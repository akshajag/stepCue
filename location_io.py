"""Outdoor location for the caregiver map: storage and geometry helpers.

The ESP32s have no GPS, so position comes from the wearer's phone (OwnTracks,
an open-source app that POSTs JSON to our own laptop; see location_server.py)
or from the simulated walk used in demos.

Kept in its own file (location.json) so the locked state.json contract is
untouched. Same safety rules: atomic write (tmp + os.replace), tolerant reads.
Location never leaves the laptop.
"""
import json
import math
import os
import time

LOCATION_FILE = os.environ.get("STEPCUE_LOCATION", "location.json")
PLACES_FILE = os.environ.get("STEPCUE_PLACES", "places.json")
TRAIL_MAX = 2000                 # ~1.5 h at one fix every 3 s
MAX_ACCURACY_M = 60.0            # fixes worse than this are ignored
MIN_STEP_M = 3.0                 # GPS jitter below this is not counted as distance

EMPTY = {"fix": None, "trail": []}


def read_json(path, default):
    for _ in range(20):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return json.loads(json.dumps(default))
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(0.01)
    return json.loads(json.dumps(default))


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # Windows: a reader has the file open
            time.sleep(0.01)
    os.replace(tmp, path)


def read_location():
    loc = read_json(LOCATION_FILE, EMPTY)
    if not isinstance(loc, dict):
        return dict(EMPTY)
    loc.setdefault("fix", None)
    loc.setdefault("trail", [])
    return loc


def read_places():
    places = read_json(PLACES_FILE, {"places": []})
    out = []
    for p in places.get("places", []) if isinstance(places, dict) else []:
        try:
            out.append({"name": str(p["name"]), "lat": float(p["lat"]), "lon": float(p["lon"]),
                        "radius_m": float(p.get("radius_m", 60)), "kind": p.get("kind", "place")})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def add_fix(loc, lat, lon, t=None, acc=None, source="phone"):
    """Append a fix to a location dict (in place). Returns False if rejected."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return False
    if acc is not None and float(acc) > MAX_ACCURACY_M:
        return False
    t = float(t if t is not None else time.time())
    trail = loc.setdefault("trail", [])
    if trail and t < trail[-1][2]:
        return False                       # out-of-order (OwnTracks replays queued fixes)
    loc["fix"] = {"lat": lat, "lon": lon, "t": t, "acc": None if acc is None else float(acc),
                  "source": source}
    trail.append([round(lat, 6), round(lon, 6), round(t, 1)])
    del trail[:-TRAIL_MAX]
    return True


# ----------------------------------------------------------- geometry --
def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def trail_distance_m(trail, since=None):
    """Distance along the trail, ignoring jitter smaller than MIN_STEP_M."""
    pts = [p for p in trail if since is None or p[2] >= since]
    total, last = 0.0, None
    for lat, lon, _ in pts:
        if last is None:
            last = (lat, lon)
            continue
        d = haversine_m(last[0], last[1], lat, lon)
        if d >= MIN_STEP_M:
            total += d
            last = (lat, lon)
    return total


def place_at(lat, lon, places):
    """The closest place whose radius contains the point, or None."""
    best = None
    for p in places:
        d = haversine_m(lat, lon, p["lat"], p["lon"])
        if d <= p["radius_m"] and (best is None or d < best[0]):
            best = (d, p)
    return best[1] if best else None
