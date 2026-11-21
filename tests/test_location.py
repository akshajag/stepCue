import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import location_io  # noqa: E402
from location_io import add_fix, haversine_m, place_at, trail_distance_m  # noqa: E402
from location_model import episode_epoch, freshness, location_view, visits  # noqa: E402

PLACES = [
    {"name": "Home", "kind": "home", "lat": 37.77105, "lon": -122.44020, "radius_m": 45},
    {"name": "Park path", "kind": "outdoor", "lat": 37.77250, "lon": -122.44650, "radius_m": 140},
]
NOW = time.mktime((2026, 11, 21, 15, 0, 0, 0, 0, -1))
M_LAT = 1 / 111_320          # degrees per metre (latitude)


def walk(start, end, t0, n, dt=3.0):
    return [[start[0] + (end[0] - start[0]) * i / (n - 1), start[1] + (end[1] - start[1]) * i / (n - 1),
             t0 + i * dt] for i in range(n)]


class Geometry(unittest.TestCase):
    def test_haversine_100m_north(self):
        self.assertAlmostEqual(haversine_m(37.0, -122.0, 37.0 + 100 * M_LAT, -122.0), 100, delta=0.5)

    def test_jitter_is_not_distance(self):
        trail = [[37.7710 + (i % 2) * 1 * M_LAT, -122.4402, i] for i in range(50)]   # 1 m wobble
        self.assertEqual(trail_distance_m(trail), 0.0)

    def test_straight_walk_distance(self):
        trail = walk((37.7700, -122.4400), (37.7700 + 300 * M_LAT, -122.4400), 0, 31)
        self.assertAlmostEqual(trail_distance_m(trail), 300, delta=3)

    def test_place_at_picks_containing_place(self):
        self.assertEqual(place_at(37.77106, -122.44021, PLACES)["name"], "Home")
        self.assertIsNone(place_at(37.7800, -122.4300, PLACES))


class Fixes(unittest.TestCase):
    def test_rejects_bad_and_inaccurate_fixes(self):
        loc = {"fix": None, "trail": []}
        self.assertFalse(add_fix(loc, 999, 0))
        self.assertFalse(add_fix(loc, 37.7, -122.4, acc=500))
        self.assertTrue(add_fix(loc, 37.7, -122.4, t=100, acc=8))
        self.assertFalse(add_fix(loc, 37.7, -122.4, t=50))         # out of order
        self.assertEqual(len(loc["trail"]), 1)
        self.assertEqual(loc["fix"]["acc"], 8.0)

    def test_trail_is_bounded(self):
        loc = {"fix": None, "trail": []}
        for i in range(location_io.TRAIL_MAX + 50):
            add_fix(loc, 37.7, -122.4, t=i)
        self.assertEqual(len(loc["trail"]), location_io.TRAIL_MAX)


class Freshness(unittest.TestCase):
    def test_states(self):
        fix = {"lat": 0, "lon": 0, "t": NOW - 5, "acc": 8, "source": "owntracks"}
        self.assertEqual(freshness(fix, NOW), {"key": "live", "text": "Live, phone GPS ±8 m"})
        self.assertEqual(freshness(dict(fix, t=NOW - 120), NOW)["text"], "Last seen 2 min ago")
        self.assertEqual(freshness(dict(fix, t=NOW - 600), NOW)["key"], "stale")
        self.assertEqual(freshness(None, NOW)["key"], "none")


class View(unittest.TestCase):
    def setUp(self):
        home, park = (PLACES[0]["lat"], PLACES[0]["lon"]), (PLACES[1]["lat"], PLACES[1]["lon"])
        t0 = NOW - 1800
        self.trail = (walk(home, home, t0, 10)                      # 30 s at home
                      + walk(home, park, t0 + 30, 200)              # ~10 min walk
                      + walk(park, park, t0 + 630, 20))             # 1 min at the park
        self.loc = {"fix": {"lat": park[0], "lon": park[1], "t": self.trail[-1][2], "acc": 6,
                            "source": "simulated"}, "trail": self.trail}

    def test_episode_epoch_today_and_yesterday(self):
        self.assertEqual(episode_epoch("14:30:00", NOW), NOW - 1800)
        self.assertEqual(episode_epoch("23:00:00", NOW), NOW - 16 * 3600)     # yesterday 23:00
        self.assertIsNone(episode_epoch("bad", NOW))

    def test_visits(self):
        v = visits(self.trail, PLACES)
        self.assertEqual([x["place"] for x in v], ["Home", "Park path"])

    def test_freeze_lands_on_the_map_at_its_place(self):
        stamp = time.strftime("%H:%M:%S", time.localtime(self.trail[-5][2]))
        eps = [{"id": 1, "timestamp": stamp, "duration_sec": 3.2, "recovery_sec": 2.0, "during_turn": True}]
        v = location_view(self.loc, PLACES, eps, now=self.trail[-1][2] + 5)
        self.assertEqual(len(v["spots"]), 1)
        self.assertEqual(v["spots"][0]["place"], "Park path")
        self.assertEqual(v["freezes_by_place"], [{"place": "Park path", "count": 1, "share": 1.0}])
        self.assertIn("Freeze 3.2 s in a turn", [e["title"] for e in v["timeline"]])
        self.assertEqual(v["here"], "Park path")

    def test_freeze_without_nearby_gps_is_not_guessed(self):
        eps = [{"id": 1, "timestamp": "03:00:00", "duration_sec": 2.0, "recovery_sec": 1.0}]
        v = location_view(self.loc, PLACES, eps, now=NOW)
        self.assertEqual(v["spots"], [])

    def test_distance_and_time_away(self):
        v = location_view(self.loc, PLACES, [], now=self.trail[-1][2] + 5)
        self.assertAlmostEqual(v["distance_km"], haversine_m(PLACES[0]["lat"], PLACES[0]["lon"],
                                                             PLACES[1]["lat"], PLACES[1]["lon"]) / 1000, delta=0.02)
        self.assertGreater(v["away_min"], 9)
        self.assertLess(v["away_min"], 12)
        self.assertEqual(v["places_visited"], ["Home", "Park path"])

    def test_empty(self):
        v = location_view({"fix": None, "trail": []}, PLACES, [], now=NOW)
        self.assertIsNone(v["here"])
        self.assertEqual(v["distance_km"], 0.0)
        self.assertEqual(v["freshness"]["key"], "none")


class OwnTracksReceiver(unittest.TestCase):
    def test_post_is_stored_and_token_enforced(self):
        import location_server
        from http.server import ThreadingHTTPServer
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "location.json")
            old = location_server.LOCATION_FILE, location_io.LOCATION_FILE
            location_server.LOCATION_FILE = location_io.LOCATION_FILE = path
            location_server.Handler.token = "s3cret"
            srv = ThreadingHTTPServer(("127.0.0.1", 0), location_server.Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{srv.server_address[1]}/pub"
            body = json.dumps({"_type": "location", "lat": 37.771, "lon": -122.44, "tst": NOW, "acc": 9}).encode()
            try:
                bad = urllib.request.Request(url, body, {"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(bad, timeout=3)
                self.assertEqual(cm.exception.code, 401)
                import base64
                auth = "Basic " + base64.b64encode(b"wearer:s3cret").decode()
                ok = urllib.request.Request(url, body, {"Content-Type": "application/json", "Authorization": auth})
                with urllib.request.urlopen(ok, timeout=3) as r:
                    self.assertEqual(r.read(), b"[]")
                with open(path, encoding="utf-8") as f:
                    loc = json.load(f)
                self.assertEqual(loc["fix"]["source"], "owntracks")
                self.assertEqual(len(loc["trail"]), 1)
            finally:
                srv.shutdown()
                srv.server_close()
                location_server.Handler.token = None
                location_server.LOCATION_FILE, location_io.LOCATION_FILE = old


if __name__ == "__main__":
    unittest.main()
