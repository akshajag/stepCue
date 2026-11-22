"""Caretaker calls placed by the wearer's own Android phone (laptop side). No network."""
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import companion_server as srv  # noqa: E402
import companion_store as cs  # noqa: E402


class CleanPhone(unittest.TestCase):
    def test_accepts_common_formats(self):
        self.assertEqual(cs.clean_phone("+1 (555) 123-4567"), "+15551234567")
        self.assertEqual(cs.clean_phone("555.123.4567"), "5551234567")

    def test_rejects_bad_numbers(self):
        for bad in ("", "123", "call me", "+1+555123456", "5551234567;rm", "1" * 16):
            with self.assertRaises(ValueError, msg=bad):
                cs.clean_phone(bad)


def lt_stamp(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


class Payload(unittest.TestCase):
    def test_none_until_number_set(self):
        self.assertIsNone(cs.alert_payload(cs.empty(), [], "FREEZE"))

    def test_disabled_sends_nothing(self):
        comp = cs.empty()
        cs.set_alert_contact(comp, "5551234567", enabled=False)
        self.assertIsNone(cs.alert_payload(comp, [], "WALKING"))

    def test_freeze_duration_and_help(self):
        now = time.time()
        comp = cs.empty()
        cs.set_alert_contact(comp, "+15551234567")
        cs.add_checkin(comp, "help", now=now - 30)
        eps = [{"id": 1, "timestamp": lt_stamp(now - 100), "recovery_sec": 2.0},
               {"id": 2, "timestamp": lt_stamp(now - 42), "recovery_sec": None}]
        p = cs.alert_payload(comp, eps, "FREEZE", now)
        self.assertEqual(p["phone"], "+15551234567")
        self.assertEqual(p["freeze_id"], 2)
        self.assertAlmostEqual(p["freeze_for"], 42, delta=1)
        self.assertEqual(p["help_t"], now - 30)
        self.assertNotIn("texts", p)                      # calls only, no text messages

    def test_old_help_and_walking_status(self):
        now = time.time()
        comp = cs.empty()
        cs.set_alert_contact(comp, "5551234567")
        cs.add_checkin(comp, "help", now=now - 600)
        p = cs.alert_payload(comp, [{"id": 3, "timestamp": lt_stamp(now - 5), "recovery_sec": None}],
                             "WALKING", now)
        self.assertIsNone(p["help_t"])
        self.assertIsNone(p["freeze_for"])
        self.assertIsNone(p["freeze_id"])

    def test_midnight_wrap(self):
        now = time.mktime((2026, 11, 22, 0, 0, 10, 0, 0, -1))
        self.assertEqual(cs.ongoing_freeze_sec([{"timestamp": "23:59:50", "recovery_sec": None}], now), 20)

    def test_test_call_counter_survives_resave(self):
        comp = cs.empty()
        with self.assertRaises(ValueError):
            cs.request_test_call(comp)
        cs.set_alert_contact(comp, "5551234567")
        self.assertEqual(cs.request_test_call(comp), 1)
        cs.set_alert_contact(comp, "5559999999")
        self.assertEqual(cs.alert_payload(comp, [], "WALKING")["test_id"], 1)


class Api(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = mock.patch.object(cs, "COMPANION_FILE", os.path.join(tmp.name, "companion.json"))
        p.start()
        self.addCleanup(p.stop)

    def test_status_carries_alerts_and_log_is_saved(self):
        comp = cs.read_companion()
        cs.set_alert_contact(comp, "5551234567")
        cs.write_companion(comp)
        with mock.patch.object(srv, "read_state", return_value={"status": "FREEZE", "episodes": []}), \
                mock.patch.object(srv, "state_age_sec", return_value=1):
            code, out = srv.handle_api("/api/status", None)
        self.assertEqual(code, 200)
        self.assertEqual(out["alerts"]["phone"], "5551234567")
        code, _ = srv.handle_api("/api/alertlog", {"kind": "help", "call": True})
        self.assertEqual(code, 200)
        rec = cs.read_companion()["alerts_log"][-1]
        self.assertEqual((rec["kind"], rec["call"]), ("help", True))
        self.assertNotIn("sms", rec)
        self.assertEqual(srv.handle_api("/api/alertlog", {"kind": "prank"})[0], 400)


if __name__ == "__main__":
    unittest.main()
