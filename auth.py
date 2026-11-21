"""Accounts, roles and two-factor sign-in for the caregiver dashboard (no Streamlit).

Three roles:
  patient my day: big "I am OK" / "I need help" buttons, how walking feels, a QR code for the phone page
  family  simple view: is my person OK, where are they, recent freezes
  care    doctor / care team: everything, plus the care report and access management

Sign-in has two steps:
  1. who you are: "Sign in with Google" (when .streamlit/secrets.toml has an
     [auth] section) or an email + password kept on this laptop
  2. a 6-digit code from Google Authenticator (or any TOTP app), RFC 6238

Accounts live in users.json on the laptop (git-ignored). Passwords are stored
as PBKDF2-SHA256 hashes, TOTP secrets as base32. Five wrong attempts lock the
account for 5 minutes, and a code cannot be used twice.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import struct
import time
import urllib.parse

USERS_FILE = os.environ.get("STEPCUE_USERS", "users.json")
ROLES = {"patient": "Patient", "family": "Family", "care": "Care team"}
ISSUER = "StepCue"
PBKDF2_ROUNDS = 240_000
MAX_FAILS = 5
LOCK_SEC = 300
TOTP_STEP = 30
TOTP_DIGITS = 6


# ----------------------------------------------------------- storage --
def load_users(path=None):
    path = path or USERS_FILE
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and isinstance(data.get("users"), list) else {"users": []}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"users": []}


def save_users(data, path=None):
    path = path or USERS_FILE
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def norm_email(email):
    return str(email or "").strip().lower()


# ------------------------------------------------------------ demo admin --
# A quick demo login (email 1, password 1) that can open all three dashboards on any computer that
# runs StepCue, even a fresh clone with no accounts yet. It is not stored in users.json and skips the
# authenticator step. Turn it off with STEPCUE_DEMO_ADMIN=off before real use, or limit it to the
# hosting computer itself with STEPCUE_ADMIN_LOCAL_ONLY=1.
ADMIN_EMAIL = "1"
ADMIN_PASSWORD = "1"
ADMIN_NAME = "1"


def admin_enabled():
    return os.environ.get("STEPCUE_DEMO_ADMIN", "on").lower() not in ("off", "0", "false", "no")


def check_admin(email, password):
    """The demo admin user dict if these are the admin login and it is enabled, else None."""
    if not admin_enabled():
        return None
    ok_e = hmac.compare_digest(str(email or "").strip().encode(), ADMIN_EMAIL.encode())
    ok_p = hmac.compare_digest(str(password or "").encode(), ADMIN_PASSWORD.encode())
    if ok_e and ok_p:
        return {"email": ADMIN_EMAIL, "name": ADMIN_NAME, "role": "admin", "admin": True}
    return None


def find_user(data, email):
    e = norm_email(email)
    return next((u for u in data["users"] if u.get("email") == e), None)


def add_user(data, email, name, role, password=None):
    if role not in ROLES:
        raise ValueError(f"role must be one of {list(ROLES)}")
    e = norm_email(email)
    if "@" not in e:
        raise ValueError("enter a valid email address")
    if find_user(data, e):
        raise ValueError("that email already has access")
    user = {"email": e, "name": str(name or e.split("@")[0]).strip(), "role": role,
            "created": time.strftime("%Y-%m-%d %H:%M")}
    if password:
        if len(password) < 10:
            raise ValueError("password must be at least 10 characters")
        user["password"] = hash_password(password)
    data["users"].append(user)
    return user


def remove_user(data, email):
    e = norm_email(email)
    data["users"] = [u for u in data["users"] if u.get("email") != e]


# ---------------------------------------------------------- passwords --
def hash_password(password, salt=None, rounds=PBKDF2_ROUNDS):
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password, stored):
    try:
        algo, rounds, salt, dk = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt), int(rounds))
        return hmac.compare_digest(test, base64.b64decode(dk))
    except (ValueError, AttributeError, TypeError):
        return False


# --------------------------------------------------------------- TOTP --
def new_totp_secret():
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _key(secret):
    s = secret.strip().replace(" ", "").upper()
    return base64.b32decode(s + "=" * (-len(s) % 8))


def totp(secret, t=None, step=TOTP_STEP, digits=TOTP_DIGITS, counter=None):
    counter = int((time.time() if t is None else t) // step) if counter is None else counter
    mac = hmac.new(_key(secret), struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return f"{code:0{digits}d}"


def match_totp(secret, code, t=None, window=1, step=TOTP_STEP):
    """Counter of the matching time step (allowing +/- window steps of clock drift), or None."""
    code = str(code or "").strip().replace(" ", "")
    if not (code.isdigit() and len(code) == TOTP_DIGITS):
        return None
    now = int((time.time() if t is None else t) // step)
    for c in range(now - window, now + window + 1):
        if hmac.compare_digest(totp(secret, counter=c), code):
            return c
    return None


def provisioning_uri(secret, email, issuer=ISSUER):
    label = urllib.parse.quote(f"{issuer}:{email}")
    q = urllib.parse.urlencode({"secret": secret, "issuer": issuer, "digits": TOTP_DIGITS, "period": TOTP_STEP})
    return f"otpauth://totp/{label}?{q}"


def qr_svg(text):
    """SVG markup of a QR code (pure Python, no image library)."""
    import qrcode
    import qrcode.image.svg
    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    svg = img.to_string(encoding="unicode")
    return svg[svg.find("<svg"):]


# ------------------------------------------------------------ sign-in --
class AuthError(Exception):
    pass


def _check_lock(user, now):
    if user.get("locked_until", 0) > now:
        mins = max(1, round((user["locked_until"] - now) / 60))
        raise AuthError(f"Too many attempts. Try again in {mins} min.")


def _fail(data, user, now, path):
    user["fails"] = user.get("fails", 0) + 1
    if user["fails"] >= MAX_FAILS:
        user["locked_until"] = now + LOCK_SEC
        user["fails"] = 0
    save_users(data, path)


def check_password(email, password, path=None, now=None):
    """Step 1 for local accounts. Returns the user dict or raises AuthError."""
    now = time.time() if now is None else now
    data = load_users(path)
    user = find_user(data, email)
    if not user or not user.get("password"):
        hash_password("x" * 10)                       # same cost either way
        raise AuthError("Email or password is not right.")
    _check_lock(user, now)
    if not verify_password(password, user["password"]):
        _fail(data, user, now, path)
        raise AuthError("Email or password is not right.")
    user["fails"] = 0
    save_users(data, path)
    return user


def check_code(email, code, path=None, now=None, secret=None):
    """Step 2. Checks a TOTP code (against `secret` while enrolling). Returns the user."""
    now = time.time() if now is None else now
    data = load_users(path)
    user = find_user(data, email)
    if not user:
        raise AuthError("This account no longer has access.")
    _check_lock(user, now)
    sec = secret or user.get("totp_secret")
    if not sec:
        raise AuthError("Two-step verification is not set up yet.")
    c = match_totp(sec, code, now)
    if c is None or c <= user.get("totp_last", -1):
        _fail(data, user, now, path)
        raise AuthError("That code is not right, or was already used. Wait for the next one.")
    user["totp_last"] = c
    user["fails"] = 0
    if secret:                                         # enrolment confirmed
        user["totp_secret"] = secret
    user["last_login"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
    save_users(data, path)
    return user


def reset_2fa(email, path=None):
    data = load_users(path)
    user = find_user(data, email)
    if user:
        user.pop("totp_secret", None)
        user.pop("totp_last", None)
        save_users(data, path)
    return user is not None
