import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import companion_store as cs  # noqa: E402

NOW = time.mktime((2026, 11, 21, 15, 0, 0, 0, 0, -1))


def stamp(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def ep(i, t, dur=3.0, closed=True):
    return {"id": i, "timestamp": stamp(t), "duration_sec": dur, "recovery_sec": 2.0 if closed else None}


class Updates(unittest.TestCase):
    def test_checkins_validated_and_bounded(self):
        d = cs.empty()
        cs.add_checkin(d, "ok", now=NOW)
        cs.add_checkin(d, "walking", "hard", now=NOW)
        with self.assertRaises(ValueError):
            cs.add_checkin(d, "walking", "terrible")
        with self.assertRaises(ValueError):
            cs.add_checkin(d, "diagnose")
        for _ in range(cs.MAX_CHECKINS + 5):
            cs.add_checkin(d, "ok", now=NOW)
        self.assertEqual(len(d["checkins"]), cs.MAX_CHECKINS)

    def test_phone_motion(self):
        d = cs.empty()
        cs.add_phone(d, "walking", 0.21, now=NOW)
        self.assertEqual(d["phone"]["motion"], "walking")
        with self.assertRaises(ValueError):
            cs.add_phone(d, "running")


class Android(unittest.TestCase):
    BODY = {"motion": "walking", "cadence": 104, "acc_rms": 2.1, "freeze_index": 0.4, "freeze_suspect": False,
            "turning": True, "turn_deg": 62, "stairs": "up", "battery": 81, "charging": False,
            "sensors": ["gyroscope", "barometer"], "device": "Pixel 8", "lat": 37.7, "lon": -122.4,
            "evil": "x" * 9999, "hz": float("inf")}

    def test_add_android_whitelists_and_feeds_phone(self):
        d = cs.empty()
        rec = cs.add_android(d, self.BODY, now=NOW)
        self.assertEqual((rec["cadence"], rec["stairs"], rec["sensors"]), (104, "up", ["gyroscope", "barometer"]))
        for k in ("evil", "lat", "lon", "hz"):
            self.assertNotIn(k, rec)                      # unknown fields, location and inf dropped
        self.assertEqual(d["phone"]["motion"], "walking")
        self.assertEqual(d["android_series"][-1][1:], ["walking", 104, 0.4, False, True])
        with self.assertRaises(ValueError):
            cs.add_android(d, {"motion": "teleporting"})

    def test_android_summary_and_freeze_vote(self):
        d = cs.empty()
        cs.add_android(d, dict(self.BODY, motion="moving", freeze_suspect=True, cadence=0), now=NOW - 2)
        summ = cs.companion_summary(d, [ep(1, NOW - 4)], now=NOW)
        self.assertTrue(summ["android"]["freeze_suspect"])
        self.assertEqual(summ["android"]["sensors"], 2)
        self.assertEqual(summ["verified"]["phone_agrees"], 1)
        self.assertIsNone(cs.android_summary(d["android"], now=NOW + 120))   # stale after a minute

    def test_daily_steps_and_minutes(self):
        d = cs.empty()
        t = NOW
        for i, steps in enumerate([10, 14, 20, 3, 8]):          # count restarts after 20 (app restarted)
            cs.add_android(d, {"motion": "walking", "cadence": 100, "steps_session": steps,
                               "freeze_index": 0.5 + i, "turning": i == 2}, now=t + 2 * i)
        today = cs.android_days(d, 7, now=t + 10)
        self.assertEqual(len(today), 7)
        self.assertIsNone(today[0]["steps"])                     # no data is not zero
        self.assertEqual(today[-1]["steps"], 4 + 6 + 3 + 5)      # first reading only sets the baseline
        self.assertGreater(today[-1]["walk_min"], 0)
        mins = cs.android_minutes(d, 3, now=t + 10)
        self.assertTrue(mins and mins[-1]["cadence"] == 100)
        self.assertEqual(max(m["freeze_index"] for m in mins), 4.5)
        self.assertTrue(any(m["turning"] for m in mins))

    def test_gauge_svg_handles_missing_and_out_of_range(self):
        import phone_ui
        self.assertIn("<svg", phone_ui.gauge_svg(None, 0, 100))
        self.assertIn("<svg", phone_ui.gauge_svg(500, 0, 140, bands=[(0, 70, "#000")], ticks=(0, 140)))

    def test_android_url_keeps_key_in_hash(self):
        url = cs.android_url("192.168.4.2", 8080, "k123")
        self.assertTrue(url.startswith("http://192.168.4.2:8080/android?n="))
        self.assertTrue(url.endswith("#key=k123"))
        self.assertNotIn("k123", url.split("#")[0])               # the key never reaches the server

    def test_beacon_carries_hash_not_key(self):
        import json
        msg = json.loads(cs.beacon_message(8080, "secret-key-123"))
        self.assertEqual((msg["stepcue"], msg["port"], len(msg["id"])), (1, 8080, 16))
        self.assertNotIn("secret-key-123", json.dumps(msg))
        self.assertEqual(msg["id"], cs.laptop_id("secret-key-123"))


class Health(unittest.TestCase):
    def test_health_auto_export_format(self):
        payload = {"data": {"metrics": [
            {"name": "heart_rate", "units": "count/min",
             "data": [{"date": "2026-11-21 14:58:00 -0600", "Avg": 71, "Min": 68, "Max": 75},
                      {"date": "2026-11-21 14:59:00 -0600", "Avg": 74}]},
            {"name": "step_count", "data": [{"date": "2026-11-21 14:00:00 -0600", "qty": 300},
                                            {"date": "2026-11-21 14:30:00 -0600", "qty": 200}]}]}}
        p = cs.parse_health(payload, now=NOW)
        self.assertEqual([v for _, v in p["heart_rate"]], [71, 74])
        self.assertEqual(p["steps"][1], 500)
        d = cs.empty()
        self.assertTrue(cs.add_health(d, p, "Health Auto Export"))
        self.assertEqual(d["watch"]["heart_rate"]["bpm"], 74)
        self.assertEqual(d["watch"]["steps"]["count"], 500)

    def test_simple_and_sensorlog_json(self):
        p = cs.parse_health({"heartRate": "82", "steps": 1200, "locationLatitude": 37.77,
                             "locationLongitude": -122.44, "locationHorizontalAccuracy": 6}, now=NOW)
        self.assertEqual(p["heart_rate"], [(NOW, 82.0)])
        self.assertEqual(p["steps"], (NOW, 1200.0))
        self.assertEqual(p["location"], (37.77, -122.44, 6.0))

    def test_implausible_values_ignored(self):
        p = cs.parse_health({"heart_rate": 900, "steps": "abc"}, now=NOW)
        self.assertEqual(p["heart_rate"], [])
        self.assertIsNone(p["steps"])
        self.assertEqual(cs.parse_health("nope")["heart_rate"], [])


class Verification(unittest.TestCase):
    def test_pending_only_recent_closed_unanswered(self):
        d = cs.empty()
        eps = [ep(1, NOW - 3000), ep(2, NOW - 120), ep(3, NOW - 60), ep(4, NOW - 10, closed=False)]
        cs.add_confirmation(d, 3, True, now=NOW)
        self.assertEqual([p["id"] for p in cs.pending_confirmations(eps, d, NOW)], [2])

    def test_verification_levels(self):
        d = cs.empty()
        e1, e2, e3, e4 = ep(1, NOW - 100), ep(2, NOW - 200), ep(3, NOW - 300), ep(4, NOW - 400)
        cs.add_confirmation(d, 1, True, now=NOW)
        cs.add_confirmation(d, 2, False, now=NOW)
        for k in range(4):
            cs.add_phone(d, "still", now=NOW - 300 + k)
            cs.add_phone(d, "walking", now=NOW - 400 + k)
        self.assertEqual(cs.verification(e1, d, NOW)[1], "Wearer confirmed")
        self.assertEqual(cs.verification(e2, d, NOW)[1], "Wearer said no freeze")
        self.assertEqual(cs.verification(e3, d, NOW)[0], "phone_agrees")
        self.assertEqual(cs.verification(e4, d, NOW)[0], "phone_walking")
        self.assertEqual(cs.verification(ep(5, NOW - 900), d, NOW)[0], "unknown")

    def test_summary(self):
        d = cs.empty()
        cs.add_checkin(d, "walking", "some trouble", now=NOW - 400)
        cs.add_checkin(d, "help", now=NOW - 30)
        cs.add_health(d, cs.parse_health({"heart_rate": 77}, now=NOW - 100), "watch")
        s = cs.companion_summary(d, [ep(1, NOW - 100)], NOW)
        self.assertTrue(s["help_open"])
        self.assertEqual(s["checkin"]["text"], "Asked for help")
        self.assertEqual(s["walking_feel"]["value"], "some trouble")
        self.assertEqual(s["heart_rate"]["bpm"], 77)
        self.assertIsNone(s["phone"])
        self.assertEqual(s["verified"]["total"], 1)


if __name__ == "__main__":
    unittest.main()
