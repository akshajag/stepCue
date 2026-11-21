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
            "watch": {}, "hr_series": [], "prefs": {}, "android": None, "android_series": [],
            "android_days": {}, "android_minutes": [], "android_steps_seen": None}


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


def android_url(host=None, port=8080, key=None):
    """Install page for the Android app (plain HTTP on the watch-data port; the key stays in the hash)."""
    return f"http://{host or lan_ip()}:{port}/android#key={key or pairing_key()}"


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


# Fields the Android app may send (POST /api/android). Anything else is dropped. Location goes to
# location.json (on this laptop) and is not kept in the android record.
ANDROID_NUM = ("cadence", "cadence_steps", "acc_rms", "gyro_rms", "vert_rms", "yaw_rate", "turn_deg",
               "freeze_index", "hz", "samples", "steps_10s", "steps_session", "pressure_hpa", "alt_change_m",
               "light_lux", "heading", "pitch", "roll", "battery", "android", "speed_ms")
ANDROID_BOOL = ("turning", "freeze_suspect", "proximity_near", "in_pocket", "charging")
STAIRS = ("up", "down", "level")


def add_android(data, body, now=None):
    """Store one feature summary from the Android app. Returns the cleaned record."""
    motion = body.get("motion")
    if motion not in PHONE_MOTION:
        raise ValueError("unknown motion")
    t = time.time() if now is None else now
    rec = {"t": t, "motion": motion}
    for k in ANDROID_NUM:
        v = _num(body.get(k))
        if v is not None and abs(v) < 1e7:
            rec[k] = round(v, 3)
    for k in ANDROID_BOOL:
        if isinstance(body.get(k), bool):
            rec[k] = body[k]
    if body.get("stairs") in STAIRS:
        rec["stairs"] = body["stairs"]
    sensors = body.get("sensors")
    if isinstance(sensors, list):
        rec["sensors"] = [str(x)[:40] for x in sensors[:24]]
    if body.get("device"):
        rec["device"] = str(body["device"])[:60]
    prev = data.get("android")
    data["android"] = rec
    _android_history(data, rec, None if not prev else prev.get("t"))
    data["android_series"].append([round(t, 1), motion, rec.get("cadence", 0), rec.get("freeze_index", 0),
                                   bool(rec.get("freeze_suspect")), bool(rec.get("turning"))])
    del data["android_series"][:-MAX_SERIES]
    add_phone(data, motion, rec.get("acc_rms"), now=t)
    return rec


MAX_DAYS = 30
MAX_MINUTES = 1440        # one day of per-minute rows


def day_key(t):
    return time.strftime("%Y-%m-%d", time.localtime(t))


def _android_history(data, rec, prev_t):
    """Roll each 2 s reading into per-day totals and per-minute rows for the charts."""
    t = rec["t"]
    seen, now_steps = data.get("android_steps_seen"), rec.get("steps_session")
    delta = 0
    if now_steps is not None:
        if seen is None:
            delta = 0                                   # first reading: we don't know when these happened
        elif now_steps >= seen:
            delta = now_steps - seen
        else:
            delta = now_steps                           # the app restarted its count
        data["android_steps_seen"] = now_steps
    delta = int(min(max(delta, 0), 400))                # a 2 s gap can't hold more than a few steps
    gap = 0 if prev_t is None else min(max(t - prev_t, 0), 5)
    walking = rec["motion"] == "walking"

    days = data.setdefault("android_days", {})
    d = days.setdefault(day_key(t), {"steps": 0, "walk_sec": 0, "freeze_sec": 0, "readings": 0})
    d["steps"] += delta
    d["walk_sec"] = round(d["walk_sec"] + (gap if walking else 0), 1)
    d["freeze_sec"] = round(d["freeze_sec"] + (gap if rec.get("freeze_suspect") else 0), 1)
    d["readings"] += 1
    for k in sorted(days)[:-MAX_DAYS]:
        del days[k]

    mins = data.setdefault("android_minutes", [])
    m0 = int(t // 60) * 60
    if not mins or mins[-1][0] != m0:
        mins.append([m0, 0, 0.0, 0, 0.0, 0, 0, 0, 0, None])
    row = mins[-1]           # [minute, readings, cadence_sum, cadence_n, fi_max, steps, walking, suspect, turning, battery]
    row[1] += 1
    if rec.get("cadence"):
        row[2] = round(row[2] + rec["cadence"], 1)
        row[3] += 1
    row[4] = max(row[4], min(float(rec.get("freeze_index", 0)), 20.0))
    row[5] += delta
    row[6] += int(walking)
    row[7] += int(bool(rec.get("freeze_suspect")))
    row[8] += int(bool(rec.get("turning")))
    if rec.get("battery") is not None:
        row[9] = rec["battery"]
    del mins[:-MAX_MINUTES]


def android_days(companion, n=7, now=None):
    """The last n days, oldest first. Days with no phone data have steps None (not 0)."""
    now = time.time() if now is None else now
    days = companion.get("android_days") or {}
    out = []
    for i in range(n - 1, -1, -1):
        t = now - i * 86400
        k = day_key(t)
        d = days.get(k)
        out.append({"date": k, "label": "Today" if i == 0 else time.strftime("%a", time.localtime(t)),
                    "steps": None if not d else d["steps"],
                    "walk_min": None if not d else round(d["walk_sec"] / 60, 1),
                    "freeze_sec": None if not d else d.get("freeze_sec", 0)})
    return out


def android_minutes(companion, hours=3, now=None):
    """Per-minute rows from the last few hours as dicts for charting."""
    now = time.time() if now is None else now
    out = []
    for r in companion.get("android_minutes") or []:
        if now - r[0] > hours * 3600:
            continue
        out.append({"t": r[0], "cadence": round(r[2] / r[3]) if r[3] else None, "freeze_index": r[4],
                    "steps": r[5], "walking": r[6] / r[1] if r[1] else 0, "suspect": r[7] > 0,
                    "turning": r[8] > 0, "battery": r[9]})
    return out


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
    lo, hi = t - 2, t + max(dur, 2) + VERIFY_NEAR_SEC / 2
    if any(r[4] for r in companion.get("android_series") or [] if lo <= r[0] <= hi):
        return "phone_agrees", "Phone felt the freeze too"
    near = [m for ts, m in companion["phone_series"] if lo <= ts <= hi]
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
        "android": android_summary(companion.get("android"), now),
        "verified": {"confirmed": ver.count("confirmed"), "denied": ver.count("denied"),
                     "phone_agrees": ver.count("phone_agrees"), "phone_walking": ver.count("phone_walking"),
                     "total": len(ver)},
    }


def android_summary(rec, now=None):
    """Latest Android reading in plain words, or None when the app has not sent in the last minute."""
    now = time.time() if now is None else now
    if not rec or now - rec.get("t", 0) > 60:
        return None
    out = {k: rec.get(k) for k in ("motion", "cadence", "steps_10s", "steps_session", "turning", "turn_deg",
                                   "freeze_index", "freeze_suspect", "stairs", "in_pocket", "battery",
                                   "charging", "device", "hz") if rec.get(k) is not None}
    out["sensors"] = len(rec.get("sensors") or [])
    out["ago"] = ago(rec["t"], now)
    return out
