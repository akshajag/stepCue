"""Companion website for the wearer's iPhone (Safari) plus a data intake for watches.

  python companion_server.py              # https://<laptop-ip>:8443  (phone page + API)
                                          # http://<laptop-ip>:8080   (API only, for watch apps)

The phone page (companion/index.html) lets the wearer:
  - tap "I'm OK" or "I need help" (help -> the notifier pushes and calls the caregiver)
  - answer "Did you just freeze?" after each episode (extra verification)
  - say how walking feels today
  - share phone motion and GPS while the page is open (motion needs HTTPS in Safari)

Smartwatch data comes in on POST /api/health from the Health Auto Export app
(Apple Health: heart rate, steps) or SensorLog / any JSON with heart_rate, steps.

Every API call needs the pairing key (companion_key.txt, shown as a QR code on
the dashboard's Access page) in an X-StepCue-Key header or ?key=. Everything is
written to companion.json (and phone GPS to location.json) on this laptop only.

HTTPS uses a self-signed certificate made on first run (certs/). Safari warns
once; tap Show Details, then visit this website.
"""
import argparse
import datetime
import hmac
import ipaddress
import json
import mimetypes
import os
import ssl
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import companion_store as cs
from alert_notifier import load_dotenv
import personal_model as pm
from location_io import LOCATION_FILE, add_fix, read_location, write_json
from state_io import read_state, state_age_sec

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companion")
CERT_DIR = "certs"
MAX_BODY = 512 * 1024
LOCK = threading.Lock()
HTTP_PORT = 8080                 # set from --http-port in main()
# served on both ports without the pairing key: the Android install page and the app itself
PUBLIC_FILES = {"/android": "android.html", "/stepcue.apk": "stepcue.apk", "/icon.png": "icon.png"}
STATUS_WORD = {"WALKING": "Walking", "STATIONARY": "Standing", "FREEZE": "Freeze detected", "OFFLINE": "No signal"}


def log(msg):
    print(time.strftime("%H:%M:%S"), "[companion]", msg, flush=True)


# ---------------------------------------------------------------- cert --
def ensure_cert(ip, cert_dir=CERT_DIR):
    """Self-signed cert for this laptop's IP (and localhost). Returns (cert, key) paths."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    os.makedirs(cert_dir, exist_ok=True)
    cert_p, key_p = os.path.join(cert_dir, "companion.crt"), os.path.join(cert_dir, "companion.key")
    if os.path.exists(cert_p) and os.path.exists(key_p):
        try:
            with open(cert_p, "rb") as f:
                c = x509.load_pem_x509_certificate(f.read())
            ips = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
                x509.IPAddress)
            if ipaddress.ip_address(ip) in ips and c.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc):
                return cert_p, key_p
        except Exception:
            pass
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "StepCue companion")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    if ip != "127.0.0.1":
        san.append(x509.IPAddress(ipaddress.ip_address(ip)))
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=397))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    with open(key_p, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    with open(cert_p, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    log(f"made a self-signed certificate for {ip} in {cert_dir}/")
    return cert_p, key_p


# ------------------------------------------------------------- handlers --
def status_payload(now=None):
    now = time.time() if now is None else now
    state = read_state()
    age = state_age_sec()
    comp = cs.read_companion()
    status = state.get("status", "STATIONARY") if age < 5 else "OFFLINE"
    ck = cs.latest_checkin(comp)
    hr = comp["watch"].get("heart_rate")
    return {"status": status, "status_text": STATUS_WORD.get(status, status),
            "pending": cs.pending_confirmations(state.get("episodes", []), comp, now),
            "last_checkin": None if not ck else {"kind": ck["kind"], "ago": cs.ago(ck["t"], now)},
            "watch": None if not hr else {"bpm": hr["bpm"], "ago": cs.ago(hr["t"], now), "t": hr["t"]},
            "setup": setup_info(),
            "cue": {"bpm": int(state.get("bpm") or 100), "haptic": int(state.get("haptic") or 0)},
            "voice": voice_info(comp),
            "training": pm.phone_status(now),
            "alerts": cs.alert_payload(comp, state.get("episodes", []), status, now),
            "phase": (comp.get("android") or {}).get("phase") if comp.get("android")
            and now - comp["android"].get("t", 0) < 10 else None,
            "server_time": now}


def voice_info(comp):
    """Which Riva voice cues the phone should play (made by nvidia_voice.py), in the wearer's language."""
    lang = (comp.get("prefs") or {}).get("lang") or os.environ.get("STEPCUE_LANG", "en")
    try:
        import nvidia_voice
        man = nvidia_voice.read_manifest()
    except Exception:
        man = {}
    entry = man.get(lang) or man.get("en")
    return {"lang": lang, "files": {k: "voice/" + v["file"] for k, v in (entry or {}).get("files", {}).items()},
            "texts": {k: v["text"] for k, v in (entry or {}).get("files", {}).items()},
            "by": (entry or {}).get("by"), "spoken_checkin": bool(nvidia_key_set())}


def nvidia_key_set():
    try:
        import nvidia_nim
        return bool(nvidia_nim.api_key())
    except Exception:
        return False


def voice_checkin(comp, body, now):
    """Spoken check-in -> Nemotron Omni -> a normal check-in (help is never dropped)."""
    import nvidia_nim
    import nvidia_voice_in
    try:
        heard = nvidia_voice_in.understand(body.get("audio"))
    except nvidia_nim.NIMError:
        return 503, {"error": "Spoken check-ins need the NVIDIA key on the laptop."}
    except ValueError:
        raise
    except Exception as exc:
        log(f"voice check-in failed: {type(exc).__name__}")
        return 502, {"error": "Could not understand the recording. Please use the buttons."}
    rec = None
    if heard["kind"] in ("ok", "help") or (heard["kind"] == "walking" and heard["walking_feel"]):
        rec = cs.add_checkin(comp, heard["kind"], heard["walking_feel"] if heard["kind"] == "walking" else None, now=now)
        rec["said"] = heard["transcript"]
        rec["via"] = "voice"
        cs.write_companion(comp)
        log(f"spoken check-in: {rec['kind']}")
    return 200, {"ok": True, "heard": heard["transcript"], "kind": heard["kind"],
                 "walking_feel": heard["walking_feel"], "saved": rec is not None}


def train_action(body, now):
    """Training screen buttons: label, stop, patient, build."""
    action = body.get("action")
    if action == "label":
        pm.set_label(body.get("label"), now)
    elif action == "stop":
        pm.stop_label()
    elif action == "patient":
        pm.set_active(str(body.get("id", "")))
    elif action == "train":
        pid, _ = pm.active()
        if not pid:
            raise ValueError("add a patient on the dashboard first")
        pm.stop_label()
        r = pm.train(pid, now)
        best = r.get("both") or r["phone"]
        log(f"personal model trained: {r['n']} samples, {best['accuracy']:.0%}")
        return 200, {"ok": True, "accuracy": round(best["accuracy"], 3), "n": r["n"], "labels": r["labels"],
                     "skipped": r["skipped"], "cv": r["cv"], "uses_ankle": r.get("both") is not None,
                     "pace": r.get("pace")}
    else:
        raise ValueError("unknown action")
    return 200, {"ok": True, "training": pm.phone_status(now)}


def setup_info():
    """What the setup page needs to connect a watch through the free Shortcuts app."""
    return {"shortcut_url": os.environ.get("STEPCUE_SHORTCUT_URL", "").strip() or None,
            "shortcut_name": os.environ.get("STEPCUE_SHORTCUT_NAME", "StepCue"),
            "watch_url": (f"http://{cs.lan_ip()}:{HTTP_PORT}/api/health?key={cs.pairing_key()}"
                          if HTTP_PORT else None)}


def handle_api(path, body, now=None):
    """Pure-ish API logic. Returns (http_code, payload). Writes companion.json / location.json."""
    now = time.time() if now is None else now
    if path == "/api/status":
        return 200, status_payload(now)
    if not isinstance(body, dict):
        return 400, {"error": "send a JSON object"}
    try:
        with LOCK:
            comp = cs.read_companion()
            if path == "/api/checkin":
                rec = cs.add_checkin(comp, body.get("kind"), body.get("value"), now=now)
                cs.write_companion(comp)
                log(f"check-in: {rec['kind']}{' ' + rec['value'] if rec['value'] else ''}")
                return 200, {"ok": True}
            if path == "/api/voice":
                return voice_checkin(comp, body, now)
            if path == "/api/prefs":
                lang = str(body.get("lang", "")).lower()
                import nvidia_translate
                if lang not in nvidia_translate.LANGUAGES:
                    raise ValueError("unknown language")
                comp.setdefault("prefs", {})["lang"] = lang
                cs.write_companion(comp)
                return 200, {"ok": True}
            if path == "/api/alertlog":
                rec = cs.add_alert_log(comp, body, now=now)
                cs.write_companion(comp)
                log(f"phone alert: {rec['kind']} (call {'placed' if rec['call'] else 'not placed'})")
                return 200, {"ok": True}
            if path == "/api/confirm":
                cs.add_confirmation(comp, body["episode_id"], body["froze"], now=now)
                cs.write_companion(comp)
                return 200, {"ok": True}
            if path == "/api/phone":
                if body.get("motion"):
                    cs.add_phone(comp, body["motion"], body.get("rms"), now=now)
                    cs.write_companion(comp)
                placed = False
                if body.get("lat") is not None and body.get("lon") is not None:
                    loc = read_location()
                    placed = add_fix(loc, body["lat"], body["lon"], t=now, acc=body.get("acc"), source="companion")
                    if placed:
                        write_json(LOCATION_FILE, loc)
                return 200, {"ok": True, "location": placed}
            if path == "/api/android":
                rec = cs.add_android(comp, body, now=now)
                state = read_state()
                saved = pm.record(rec, state, now)            # per-patient training, when a label is on
                try:
                    phase = None if pm.session(now)["label"] else pm.predict(rec, state, now=now)
                except Exception as exc:                      # a broken model must never stop the data
                    log(f"personal model failed: {type(exc).__name__}")
                    phase = None
                if phase:
                    comp["android"]["phase"] = {k: phase[k] for k in ("label", "word", "conf", "source", "probs")}
                cs.write_companion(comp)
                placed = False
                if body.get("lat") is not None and body.get("lon") is not None:
                    loc = read_location()
                    placed = add_fix(loc, body["lat"], body["lon"], t=now, acc=body.get("acc_m"), source="android")
                    if placed:
                        write_json(LOCATION_FILE, loc)
                return 200, {"ok": True, "location": placed, "training_sample": saved}
            if path == "/api/train":
                return train_action(body, now)
            if path == "/api/health":
                parsed = cs.parse_health(body, now=now)
                src = "SensorLog" if any(str(k).startswith(("locationLatitude", "motion", "logging"))
                                         for k in body) else ("Health Auto Export" if "data" in body else "watch")
                got = cs.add_health(comp, parsed, src)
                if got:
                    cs.write_companion(comp)
                if parsed["location"]:
                    loc = read_location()
                    lat, lon, acc = parsed["location"]
                    if add_fix(loc, lat, lon, t=now, acc=acc, source="watch"):
                        write_json(LOCATION_FILE, loc)
                return 200, {"ok": True, "heart_rate": len(parsed["heart_rate"]),
                             "steps": parsed["steps"] is not None}
    except (KeyError, TypeError, ValueError) as exc:
        return 400, {"error": str(exc) or "bad request"}
    return 404, {"error": "unknown endpoint"}


def make_handler(key, serve_page):
    class Handler(BaseHTTPRequestHandler):
        server_version = "StepCueCompanion"

        def log_message(self, fmt, *args):          # quiet; never log keys or bodies
            pass

        def _send(self, code, payload=None, ctype="application/json", raw=None):
            data = raw if raw is not None else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self, query):
            given = self.headers.get("X-StepCue-Key") or (query.get("key") or [""])[0]
            return bool(given) and hmac.compare_digest(given.encode(), key.encode())

        def _route(self, method):
            u = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(u.query)
            if method == "GET" and u.path in PUBLIC_FILES:     # Android install page + app, no key
                return self._static("/" + PUBLIC_FILES[u.path])
            if not u.path.startswith("/api/"):
                if method == "GET" and serve_page:
                    return self._static(u.path)
                return self._send(404, {"error": "not found"})
            if not self._authorised(query):
                return self._send(401, {"error": "pairing key missing or wrong"})
            body = None
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._send(413, {"error": "too large"})
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._send(400, {"error": "not JSON"})
            elif u.path != "/api/status":
                return self._send(405, {"error": "use POST"})
            code, payload = handle_api(u.path, body)
            self._send(code, payload)

        def _static(self, path):
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            full = os.path.normpath(os.path.join(STATIC_DIR, rel))
            if not full.startswith(STATIC_DIR + os.sep) or not os.path.isfile(full):
                return self._send(404, {"error": "not found"})
            with open(full, "rb") as f:
                data = f.read()
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if full.endswith(".webmanifest"):
                ctype = "application/manifest+json"
            elif full.endswith(".apk"):
                ctype = "application/vnd.android.package-archive"
            self._send(200, raw=data, ctype=ctype)

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

    return Handler


def beacon_loop(key, port, every=2.0):
    """Tell phones on the same Wi-Fi where this laptop is, so a paired phone finds it again
    after the laptop's address changes. Only a short hash of the key is sent, never the key."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    msg = cs.beacon_message(port, key)
    while True:
        targets = {"255.255.255.255"}
        ip = cs.lan_ip()
        try:
            targets.add(str(ipaddress.ip_network(ip + "/24", strict=False).broadcast_address))
        except ValueError:
            pass
        for t in targets:
            try:
                sock.sendto(msg, (t, cs.BEACON_PORT))
            except OSError:
                pass
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=cs.COMPANION_PORT, help="HTTPS port for the phone page")
    ap.add_argument("--http-port", type=int, default=8080, help="plain HTTP port for watch apps (0 = off)")
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    global HTTP_PORT
    HTTP_PORT = a.http_port
    load_dotenv()                                   # STEPCUE_SHORTCUT_URL may live in .env
    key = cs.pairing_key()
    ip = cs.lan_ip()
    cert, pkey = ensure_cert(ip)
    httpsd = ThreadingHTTPServer((a.host, a.port), make_handler(key, serve_page=True))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, pkey)
    httpsd.socket = ctx.wrap_socket(httpsd.socket, server_side=True)
    servers = [httpsd]
    if a.http_port:
        servers.append(ThreadingHTTPServer((a.host, a.http_port), make_handler(key, serve_page=False)))
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    if a.http_port:
        threading.Thread(target=beacon_loop, args=(key, a.http_port), daemon=True).start()
    log(f"phone page: https://{ip}:{a.port}  (scan the QR code on the dashboard's Access page)")
    if a.http_port:
        log(f"watch data: http://{ip}:{a.http_port}/api/health  (header X-StepCue-Key)")
        log(f"Android app: http://{ip}:{a.http_port}/android  (scan the Android QR code on the dashboard)")
    try:
        httpsd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
