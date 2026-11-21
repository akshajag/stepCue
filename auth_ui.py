"""Streamlit sign-in screens for the dashboard (logic lives in auth.py).

  user = require_login()   # stops the script until someone is fully signed in

Step 1: "Sign in with Google" when .streamlit/secrets.toml has an [auth]
section (Streamlit's built-in OIDC), otherwise email + password.
Step 2: a 6-digit code from Google Authenticator; first sign-in shows a QR code.
Only emails listed in users.json get in, whatever Google says.

Set STEPCUE_AUTH=off to skip sign-in on a demo laptop (a banner says so).
"""
import html
import os
import time

import streamlit as st

import auth

SESSION_HOURS = 12

CSS = """
<style>
.sc-auth {max-width:440px; margin:6vh auto 0 auto; font-family:var(--font);}
.sc-auth .ttl {font:700 1.7rem var(--font); color:var(--ink); margin:0 0 4px 0;}
.sc-auth p {color:var(--muted); font-size:1rem; margin:0 0 18px 0; line-height:1.45;}
.sc-auth .qr {background:#fff; border:1px solid var(--line); border-radius:14px; padding:12px; width:220px;
  margin:6px auto 10px auto;}
.sc-auth .qr svg {width:100%; height:auto; display:block;}
.sc-auth code {font-size:0.95rem; letter-spacing:0.08em; word-break:break-all;}
.sc-steps {display:flex; gap:8px; margin-bottom:14px;}
.sc-steps span {flex:1; height:5px; border-radius:3px; background:var(--line);}
.sc-steps span.on {background:var(--ok);}
.sc-authoff {background:var(--watch-tint); color:#7A4300; border-radius:10px; padding:8px 12px;
  font:400 0.9rem var(--font); margin-bottom:10px;}
</style>
"""


def google_configured():
    try:
        return "auth" in st.secrets
    except Exception:                      # no secrets.toml at all
        return False


def google_email():
    try:
        if st.user.is_logged_in:
            return auth.norm_email(st.user.get("email"))
    except Exception:
        pass
    return None


def _header(title, sub, step):
    st.markdown(CSS, unsafe_allow_html=True)
    bars = "".join(f'<span class="{"on" if i <= step else ""}"></span>' for i in (1, 2))
    st.markdown(f'<div class="sc-auth"><div class="sc-steps">{bars}</div><div class="ttl">{html.escape(title)}</div>'
                f'<p>{sub}</p></div>', unsafe_allow_html=True)


def _first_run():
    _header("Set up StepCue access", "No accounts yet. Create the first care-team account. "
            "You can add family members afterwards.", 1)
    _, mid, _ = st.columns([1, 2, 1])
    with mid, st.form("first_run"):
        name = st.text_input("Your name")
        email = st.text_input("Email" + (" (the Google account you will sign in with)" if google_configured() else ""))
        pw = st.text_input("Password (at least 10 characters)" + (", optional with Google" if google_configured() else ""),
                           type="password")
        if st.form_submit_button("Create account", type="primary", width="stretch"):
            try:
                data = auth.load_users()
                if data["users"]:
                    st.rerun()
                if not pw and not google_configured():
                    raise ValueError("a password is needed when Google sign-in is not set up")
                auth.add_user(data, email, name, "care", password=pw or None)
                auth.save_users(data)
                st.session_state["auth_email"] = auth.norm_email(email)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc).capitalize())
    if auth.admin_enabled():
        with mid:
            st.markdown('<div style="text-align:center;color:var(--muted);margin:14px 0 6px">or use the demo login</div>',
                        unsafe_allow_html=True)
            with st.form("first_run_admin"):
                email = st.text_input("Email", key="fr_admin_email")
                pw = st.text_input("Password", type="password", key="fr_admin_pw")
                if st.form_submit_button("Sign in", width="stretch"):
                    admin = auth.check_admin(email, pw)
                    if admin and admin_allowed_here():
                        st.session_state["auth_user"] = dict(admin, at=time.time())
                        st.rerun()
                    st.error("Email or password is not right.")
    st.stop()


def _step_one():
    _header("Sign in to StepCue", "Caregiver dashboard. Two steps: your account, then a code "
            "from your authenticator app.", 1)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if google_configured():
            g = google_email()
            if g:
                if auth.find_user(auth.load_users(), g):
                    st.session_state["auth_email"] = g
                    st.rerun()
                st.error(f"{g} has not been given access. Ask the care team to add you.")
                if st.button("Use a different Google account", width="stretch"):
                    st.logout()
            else:
                st.button("Sign in with Google", type="primary", width="stretch", on_click=st.login)
            st.markdown('<div style="text-align:center;color:var(--muted);margin:10px 0">or</div>',
                        unsafe_allow_html=True)
        with st.form("pw_login"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Continue", width="stretch",
                                     type="secondary" if google_configured() else "primary"):
                admin = auth.check_admin(email, pw)
                if admin:
                    if not admin_allowed_here():
                        st.error("The demo admin login is limited to the computer running StepCue.")
                    else:
                        st.session_state["auth_user"] = dict(admin, at=time.time())
                        st.rerun()
                else:
                    try:
                        u = auth.check_password(email, pw)
                        st.session_state["auth_email"] = u["email"]
                        st.rerun()
                    except auth.AuthError as exc:
                        st.error(str(exc))
    st.stop()


def _step_two(email):
    data = auth.load_users()
    user = auth.find_user(data, email)
    if not user:
        st.session_state.pop("auth_email", None)
        st.rerun()
    enrolling = not user.get("totp_secret")
    if enrolling:
        secret = st.session_state.setdefault("enroll_secret", auth.new_totp_secret())
        uri = auth.provisioning_uri(secret, email)
        _header("Set up two-step verification",
                "Open Google Authenticator (or any authenticator app), tap <b>+</b>, scan this code, "
                "then type the 6 digits it shows.", 2)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(f'<div class="sc-auth" style="margin-top:0"><div class="qr">{auth.qr_svg(uri)}</div>'
                        f'<p style="text-align:center">Can&#39;t scan? Enter this key:<br>'
                        f'<code>{" ".join(secret[i:i + 4] for i in range(0, len(secret), 4))}</code></p></div>',
                        unsafe_allow_html=True)
    else:
        secret = None
        _header("Enter your code", f"Signed in as <b>{html.escape(email)}</b>. Type the 6-digit code "
                "from your authenticator app.", 2)
        _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("totp"):
            code = st.text_input("6-digit code", max_chars=6, placeholder="123456")
            if st.form_submit_button("Verify", type="primary", width="stretch"):
                try:
                    u = auth.check_code(email, code, secret=secret)
                    st.session_state["auth_user"] = {"email": u["email"], "name": u.get("name", ""),
                                                     "role": u["role"], "at": time.time()}
                    st.session_state.pop("enroll_secret", None)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))
        if st.button("Back", width="stretch"):
            sign_out()
    st.stop()


def admin_allowed_here():
    """The demo admin works from any device unless STEPCUE_ADMIN_LOCAL_ONLY=1 limits it to the host."""
    if os.environ.get("STEPCUE_ADMIN_LOCAL_ONLY", "").lower() not in ("1", "on", "true", "yes"):
        return True
    try:
        ip = st.context.ip_address
    except Exception:
        return False
    return ip in (None, "127.0.0.1", "::1")


def sign_out():
    for k in ("auth_user", "auth_email", "enroll_secret"):
        st.session_state.pop(k, None)
    if google_email():
        st.logout()
    st.rerun()


def require_login():
    if os.environ.get("STEPCUE_AUTH", "on").lower() == "off":
        st.markdown('<div class="sc-authoff">Sign-in is turned off on this laptop (STEPCUE_AUTH=off).</div>',
                    unsafe_allow_html=True)
        return {"email": "demo@local", "name": "Demo", "role": os.environ.get("STEPCUE_ROLE", "care"),
                "demo": True}
    u = st.session_state.get("auth_user")
    if u and u.get("admin"):
        if time.time() - u["at"] < SESSION_HOURS * 3600 and auth.admin_enabled() and admin_allowed_here():
            return u
        st.session_state.pop("auth_user", None)
        u = None
    if u:
        fresh = time.time() - u["at"] < SESSION_HOURS * 3600
        still = auth.find_user(auth.load_users(), u["email"])
        if fresh and still:
            u["role"] = still["role"]                 # role changes apply immediately
            return u
        st.session_state.pop("auth_user", None)
    if not auth.load_users()["users"]:
        _first_run()
    email = st.session_state.get("auth_email")
    if not email:
        _step_one()
    _step_two(email)
