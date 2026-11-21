"""Receive the wearer's outdoor position and keep location.json up to date.

The ESP32s have no GPS, so the position comes from the phone the wearer
already carries:

  Real phone (OwnTracks, free and open source, iOS + Android):
    python location_server.py
    In OwnTracks: Settings -> Connection -> Mode "HTTP",
      URL  http://<laptop IP>:5010/pub      (on StepCue_AP that is 192.168.4.2)
    Optional shared secret:  python location_server.py --token abc123
      and set the OwnTracks password to abc123.

  Demo without a phone (walks a route through places.json):
    python location_server.py --simulate
  The simulated walker only moves while the gait classifier says WALKING,
  so freezes stay where they happened on the map.

Everything stays on this laptop: no cloud geolocation, no third-party server.
"""
import argparse
import base64
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from location_io import (LOCATION_FILE, add_fix, haversine_m, read_location,
                         write_json)

PORT = 5010
LOCK = threading.Lock()

# Demo route (lat, lon): Home -> park path -> grocery store -> community center -> Home.
ROUTE = [
    (37.77105, -122.44020), (37.77180, -122.44045), (37.77240, -122.44190),
    (37.77255, -122.44480), (37.77262, -122.44790), (37.77270, -122.45060),
    (37.77380, -122.45120), (37.77455, -122.45180), (37.77360, -122.45080),
    (37.77140, -122.44990), (37.76960, -122.44860), (37.76990, -122.44560),
    (37.77030, -122.44260), (37.77105, -122.44020),
]
WALK_MPS = 1.0          # a slow, careful walking pace


def store_fix(lat, lon, t=None, acc=None, source="phone"):
    with LOCK:
        loc = read_location()
        if add_fix(loc, lat, lon, t, acc, source):
            write_json(LOCATION_FILE, loc)
            return True
    return False


# --------------------------------------------------------------- HTTP --
class Handler(BaseHTTPRequestHandler):
    token = None

    def log_message(self, fmt, *args):      # keep the console quiet
        pass

    def _authorized(self):
        if not self.token:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                return pw == self.token
            except ValueError:
                return False
        return self.headers.get("X-StepCue-Token") == self.token

    def _reply(self, code, body=b"[]"):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?")[0] not in ("/pub", "/"):
            return self._reply(404)
        if not self._authorized():
            return self._reply(401)
        try:
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._reply(400)
        if isinstance(msg, dict) and msg.get("_type") == "location":
            ok = store_fix(msg.get("lat"), msg.get("lon"), msg.get("tst"), msg.get("acc"), "owntracks")
            print(f"[location {time.strftime('%H:%M:%S')}] fix {msg.get('lat')}, {msg.get('lon')} "
                  f"+/-{msg.get('acc')} m {'stored' if ok else 'ignored'}", flush=True)
        self._reply(200)                     # OwnTracks expects a JSON array back


# ---------------------------------------------------------- simulator --
def gait_status():
    try:
        from state_io import read_state, state_age_sec
        if state_age_sec() > 3:
            return "WALKING"                 # no classifier running: just walk
        return read_state().get("status", "WALKING")
    except Exception:
        return "WALKING"


def simulate(period=2.0, seed=0, speed=1.0):
    rng = random.Random(seed)
    seg, frac = 0, 0.0
    print(f"[location] simulated walk, a fix every {period:.0f} s (moves only while WALKING)", flush=True)
    while True:
        a, b = ROUTE[seg], ROUTE[(seg + 1) % len(ROUTE)]
        seg_len = max(1.0, haversine_m(*a, *b))
        if gait_status() == "WALKING":
            frac += WALK_MPS * speed * period / seg_len
            while frac >= 1.0:
                frac -= 1.0
                seg = (seg + 1) % len(ROUTE)
                a, b = ROUTE[seg], ROUTE[(seg + 1) % len(ROUTE)]
                seg_len = max(1.0, haversine_m(*a, *b))
        lat = a[0] + (b[0] - a[0]) * frac
        lon = a[1] + (b[1] - a[1]) * frac
        acc = rng.uniform(4, 12)
        # GPS noise of a few metres
        lat += rng.gauss(0, acc * 0.3) / 111_320
        lon += rng.gauss(0, acc * 0.3) / (111_320 * math.cos(math.radians(lat)))
        store_fix(lat, lon, time.time(), round(acc, 1), "simulated")
        time.sleep(period)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--token", help="shared secret (OwnTracks password)")
    ap.add_argument("--simulate", action="store_true", help="demo walk instead of a phone")
    ap.add_argument("--reset", action="store_true", help="clear the stored trail first")
    ap.add_argument("--speed", type=float, default=1.0, help="simulated walk speed-up (demo)")
    a = ap.parse_args()
    if a.reset:
        write_json(LOCATION_FILE, {"fix": None, "trail": []})
    if a.simulate:
        threading.Thread(target=simulate, kwargs={"speed": a.speed}, daemon=True).start()
    Handler.token = a.token
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[location] listening for OwnTracks on http://0.0.0.0:{a.port}/pub", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
