"""Data from the wearer's phone and smartwatch (companion site), and what it means.

companion.json is written only by companion_server.py (atomic tmp + replace)
and read by the dashboard and the alert notifier. It holds:

  checkins       "I'm OK" / "I need help" / how walking feels today
  confirmations  the wearer's answer to "Did you just freeze?" per episode
  phone          latest phone-motion reading + a short history
  watch          heart rate and steps from a smartwatch (Health Auto Export,
                 SensorLog or any JSON with heart_rate / steps)

The extra verification: a freeze the wearer confirms, or one where the phone
was also still, is more trustworthy than one where the phone saw walking.
Descriptive only; nothing here is a medical measurement.
"""
import json
import os
import time
from datetime import datetime

COMPANION_FILE = os.environ.get("STEPCUE_COMPANION", "companion.json")
KEY_FILE = os.environ.get("STEPCUE_COMPANION_KEY_FILE", "companion_key.txt")
COMPANION_PORT = int(os.environ.get("STEPCUE_COMPANION_PORT", "8443"))
MAX_CHECKINS = 200
MAX_SERIES = 900
CHECKIN_KINDS = ("ok", "help", "walking")
WALKING_FEEL = ("easy", "some trouble", "hard")
PHONE_MOTION = ("still", "moving", "walking")
CONFIRM_WINDOW_SEC = 600      # ask the wearer about freezes that ended in the last 10 min
VERIFY_NEAR_SEC = 10


def empty():
    return {"checkins": [], "confirmations": {}, "phone": None, "phone_series": [],
            "watch": {}, "hr_series": [], "prefs": {}}


def read_companion(path=None):
    path = path or COMPANION_FILE
    for _ in range(20):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            base = empty()
            if isinstance(data, dict):
                base.update({k: v for k, v in data.items() if k in base})
            return base
        except FileNotFoundError:
            return empty()
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(0.01)
    return empty()


def write_companion(data, path=None):
    path = path or COMPANION_FILE
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.01)
    os.replace(tmp, path)


# ------------------------------------------------------------- pairing --
def pairing_key(path=None, create=True):
    """Shared secret the phone needs to post data (kept in companion_key.txt, git-ignored)."""
    path = path or KEY_FILE
    env = os.environ.get("STEPCUE_COMPANION_KEY")
    if env:
        return env
    try:
        with open(path, encoding="utf-8") as f:
            k = f.read().strip()
        if k:
            return k
    except FileNotFoundError:
        pass
    if not create:
        return None
    import secrets
    k = secrets.token_urlsafe(12)
    with open(path, "w", encoding="utf-8") as f:
        f.write(k)
    return k


def lan_ip():
    """This laptop's address on the local network (no packets are sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def companion_url(host=None, port=None, key=None):
    return f"https://{host or lan_ip()}:{port or COMPANION_PORT}/#key={key or pairing_key()}"


# ------------------------------------------------------------- updates --
def add_checkin(data, kind, value=None, now=None):
    if kind not in CHECKIN_KINDS:
        raise ValueError("unknown check-in")
    if kind == "walking" and value not in WALKING_FEEL:
        raise ValueError("unknown answer")
    rec = {"t": time.time() if now is None else now, "kind": kind, "value": value}
    data["checkins"].append(rec)
    del data["checkins"][:-MAX_CHECKINS]
    return rec


def add_confirmation(data, episode_id, froze, now=None):
    data["confirmations"][str(int(episode_id))] = {"froze": bool(froze),
                                                   "t": time.time() if now is None else now}


def add_phone(data, motion, rms=None, now=None):
    if motion not in PHONE_MOTION:
        raise ValueError("unknown motion")
    t = time.time() if now is None else now
    data["phone"] = {"t": t, "motion": motion, "rms": None if rms is None else round(float(rms), 4)}
    data["phone_series"].append([round(t, 1), motion])
    del data["phone_series"][:-MAX_SERIES]


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _when(s, default):
    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(str(s), fmt).timestamp()
        except ValueError:
            continue
    return default


def parse_health(payload, now=None):
    """Heart rate / steps from Health Auto Export, SensorLog or simple JSON.

    Returns {"heart_rate": [(t, bpm)...], "steps": (t, count) or None, "lat/lon"...}.
    """
    now = time.time() if now is None else now
    out = {"heart_rate": [], "steps": None, "location": None}
    if not isinstance(payload, dict):
        return out
    metrics = (payload.get("data") or {}).get("metrics") if isinstance(payload.get("data"), dict) else None
    if isinstance(metrics, list):                                   # Health Auto Export
        for m in metrics:
            name = str(m.get("name", "")).lower()
            for d in m.get("data") or []:
                t = _when(d.get("date"), now)
                if name == "heart_rate":
                    v = _num(d.get("Avg", d.get("qty")))
                    if v and 25 <= v <= 240:
                        out["heart_rate"].append((t, v))
                elif name == "step_count":
                    q = _num(d.get("qty"))
                    if q is not None:
                        prev = out["steps"][1] if out["steps"] else 0
                        out["steps"] = (t, prev + q)
        return out
    # simple JSON or SensorLog-style flat keys
    for k, v in payload.items():
        kl = str(k).lower()
        n = _num(v)
        if n is None:
            continue
        if "heart" in kl and 25 <= n <= 240:
            out["heart_rate"].append((now, n))
        elif kl in ("steps", "step_count", "pedometernumberofsteps"):
            out["steps"] = (now, n)
    lat = _num(payload.get("locationLatitude", payload.get("lat")))
    lon = _num(payload.get("locationLongitude", payload.get("lon")))
    if lat is not None and lon is not None:
        out["location"] = (lat, lon, _num(payload.get("locationHorizontalAccuracy", payload.get("acc"))))
    return out


def add_health(data, parsed, source="watch"):
    hr = sorted(parsed["heart_rate"])
    if hr:
        t, v = hr[-1]
        data["watch"]["heart_rate"] = {"t": t, "bpm": round(v), "source": source}
        data["hr_series"] += [[round(a, 1), round(b)] for a, b in hr]
        data["hr_series"].sort()
        del data["hr_series"][:-MAX_SERIES]
    if parsed["steps"]:
        t, n = parsed["steps"]
        data["watch"]["steps"] = {"t": t, "count": int(n), "source": source}
    return bool(hr or parsed["steps"])


# ---------------------------------------------------------- view logic --
def episode_epoch(stamp, now):
    from location_model import episode_epoch as ee
    return ee(stamp, now)


def pending_confirmations(episodes, companion, now=None):
    """Closed episodes from the last 10 minutes the wearer has not answered about."""
    now = time.time() if now is None else now
    out = []
    for e in episodes or []:
        if e.get("recovery_sec") is None or str(e.get("id")) in companion["confirmations"]:
            continue
        t = episode_epoch(e.get("timestamp"), now)
        if t is not None and now - t <= CONFIRM_WINDOW_SEC:
            out.append({"id": e["id"], "time": str(e.get("timestamp", ""))[:5],
                        "duration_sec": e.get("duration_sec")})
    return out[-3:]


def verification(episode, companion, now=None):
    """('confirmed'|'denied'|'phone_agrees'|'phone_walking'|'unknown', plain text)."""
    now = time.time() if now is None else now
    c = companion["confirmations"].get(str(episode.get("id")))
    if c is not None:
        return ("confirmed", "Wearer confirmed") if c["froze"] else ("denied", "Wearer said no freeze")
    t = episode_epoch(episode.get("timestamp"), now)
    if t is None:
        return "unknown", ""
    dur = float(episode.get("duration_sec") or 0)
    near = [m for ts, m in companion["phone_series"] if t - 2 <= ts <= t + max(dur, 2) + VERIFY_NEAR_SEC / 2]
    if not near:
        return "unknown", ""
    if near.count("walking") > len(near) / 2:
        return "phone_walking", "Phone saw walking"
    return "phone_agrees", "Phone agrees"


def latest_checkin(companion, kinds=("ok", "help")):
    for c in reversed(companion["checkins"]):
        if c["kind"] in kinds:
            return c
    return None


def ago(t, now=None):
    now = time.time() if now is None else now
    s = max(0, now - t)
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return time.strftime("%b %d", time.localtime(t))


def companion_summary(companion, episodes, now=None):
    """Plain facts for the dashboards."""
    now = time.time() if now is None else now
    ck = latest_checkin(companion)
    feel = latest_checkin(companion, ("walking",))
    hr = companion["watch"].get("heart_rate")
    steps = companion["watch"].get("steps")
    phone = companion["phone"]
    ver = [verification(e, companion, now)[0] for e in episodes or [] if e.get("recovery_sec") is not None]
    help_open = bool(ck and ck["kind"] == "help" and now - ck["t"] < 3600)
    return {
        "checkin": None if not ck else {"kind": ck["kind"], "ago": ago(ck["t"], now),
                                        "text": "Asked for help" if ck["kind"] == "help" else "Checked in: I'm OK"},
        "help_open": help_open,
        "walking_feel": None if not feel else {"value": feel["value"], "ago": ago(feel["t"], now)},
        "heart_rate": None if not hr or now - hr["t"] > 1800 else {"bpm": hr["bpm"], "ago": ago(hr["t"], now),
                                                                    "source": hr.get("source", "watch")},
        "steps": None if not steps else {"count": steps["count"], "ago": ago(steps["t"], now)},
        "phone": None if not phone or now - phone["t"] > 60 else {"motion": phone["motion"],
                                                                   "ago": ago(phone["t"], now)},
        "verified": {"confirmed": ver.count("confirmed"), "denied": ver.count("denied"),
                     "phone_agrees": ver.count("phone_agrees"), "phone_walking": ver.count("phone_walking"),
                     "total": len(ver)},
    }
