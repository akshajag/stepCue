import base64
import os
import sys
import tempfile
import unittest
import urllib.parse
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alert_notifier as an  # noqa: E402

BASE = {"status": "WALKING", "freeze_index": 0.3, "bpm": 96, "haptic": 1, "alert_level": "normal",
        "caretaker_brief": "Brief from the agent.", "episodes": []}


def frozen(dur, level="normal"):
    return dict(BASE, status="FREEZE", alert_level=level,
                episodes=[{"id": 1, "timestamp": "14:00:00", "duration_sec": dur, "recovery_sec": None}])


def done(level="normal"):
    return dict(BASE, alert_level=level,
                episodes=[{"id": 1, "timestamp": "14:00:00", "duration_sec": 6.0, "recovery_sec": 3.0}])


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakePush:
    def __init__(self):
        self.sent = []

    def send(self, title, body, priority, tag):
        self.sent.append((title, body, priority, tag))
        return "HTTP 200"


class FakeCall:
    def __init__(self):
        self.sent = []

    def send(self, spoken):
        self.sent.append(spoken)
        return "HTTP 201 CA123"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(an, "NOTIFY_LOG", os.path.join(self.tmp.name, "log.jsonl"))
        p.start()
        self.addCleanup(p.stop)
        self.clock, self.push, self.call = Clock(), FakePush(), FakeCall()
        self.n = an.Notifier(self.push, self.call, clock=self.clock)

    def feed(self, state, age=0.2, dt=1.0):
        self.clock.t += dt
        return self.n.step(state, age)


class Decisions(Base):
    def test_short_freeze_is_quiet(self):
        self.feed(BASE)
        self.feed(frozen(2.0))
        self.feed(done())
        self.assertEqual(self.push.sent, [])
        self.assertEqual(self.call.sent, [])

    def test_prolonged_freeze_pushes_and_calls_once(self):
        self.feed(BASE)
        for d in (2.0, 4.5, 6.0, 8.0, 12.0):
            self.feed(frozen(d))
        self.assertEqual(len(self.push.sent), 1)
        title, body, prio, _ = self.push.sent[0]
        self.assertEqual((title, prio), ("StepCue: please check in", 5))
        self.assertIn("A freeze has lasted 4.5 s", body)
        self.assertEqual(len(self.call.sent), 1)
        self.assertIn("Please check in", self.call.sent[0])

    def test_all_clear_after_check_in(self):
        self.feed(frozen(5.0))
        self.feed(done())
        self.assertEqual([s[0] for s in self.push.sent], ["StepCue: please check in", "StepCue: walking again"])
        self.assertEqual(self.push.sent[1][2], 2)

    def test_call_cooldown_across_episodes(self):
        self.feed(frozen(5.0)); self.feed(done())
        self.feed(frozen(5.0), dt=60); self.feed(done())
        self.assertEqual(len(self.call.sent), 1)                  # second within 10 min: no call
        self.feed(frozen(5.0), dt=700)
        self.assertEqual(len(self.call.sent), 2)

    def test_advisor_watch_pushes_once(self):
        self.feed(done("watch"))
        self.feed(done("watch"))
        self.assertEqual([s[2] for s in self.push.sent], [3])

    def test_offline_only_after_being_live_and_two_minutes(self):
        n = an.Notifier(self.push, None, clock=self.clock)
        self.clock.t += 1; n.step(BASE, age_sec=99)                # starts offline: no alert
        self.clock.t += 300; n.step(BASE, age_sec=400)
        self.assertEqual(self.push.sent, [])
        self.clock.t += 1; n.step(BASE, age_sec=0.2)               # live
        self.clock.t += 5; n.step(BASE, age_sec=5)                 # offline begins
        self.clock.t += 60; n.step(BASE, age_sec=65)
        self.assertEqual(self.push.sent, [])
        self.clock.t += 70; n.step(BASE, age_sec=135)
        self.clock.t += 70; n.step(BASE, age_sec=205)
        self.assertEqual([s[0] for s in self.push.sent], ["StepCue: no signal"])

    def test_dry_run_sends_nothing_but_logs(self):
        n = an.Notifier(self.push, self.call, dry_run=True, clock=self.clock)
        self.clock.t += 1
        recs = n.step(frozen(6.0), 0.2)
        self.assertEqual(self.push.sent, [])
        self.assertEqual(self.call.sent, [])
        self.assertTrue(all(r["result"] == "dry run (not sent)" for r in recs))
        self.assertEqual(len(an.read_notifications()), 2)

    def test_unconfigured_is_dry_run(self):
        self.assertTrue(an.Notifier(None, None).dry_run)

    def test_send_failure_is_logged_not_raised(self):
        class Broken:
            def send(self, *a):
                raise OSError("network down")
        n = an.Notifier(Broken(), None, clock=self.clock)
        self.clock.t += 1
        recs = n.step(frozen(6.0), 0.2)
        self.assertTrue(recs[0]["result"].startswith("failed"))

    def test_help_request_pushes_and_calls_once_per_tap(self):
        comp = {"checkins": [{"t": 900.0, "kind": "help"}]}            # 100 s before "now"
        self.clock.t = 1000.0
        self.n.step_companion(comp)
        self.n.step_companion(comp)
        self.assertEqual([(s[0], s[2]) for s in self.push.sent], [("StepCue: help requested", 5)])
        self.assertEqual(len(self.call.sent), 1)
        self.assertNotIn("location", self.push.sent[0][1].lower())
        comp["checkins"].append({"t": 1005.0, "kind": "ok"})
        self.clock.t = 1010.0
        self.n.step_companion(comp)
        self.assertEqual(len(self.push.sent), 1)                       # "I'm OK" is not an alert
        comp["checkins"].append({"t": 1020.0, "kind": "help"})
        self.clock.t = 1021.0
        self.n.step_companion(comp)
        self.assertEqual(len(self.push.sent), 2)

    def test_old_help_request_ignored_on_start(self):
        self.clock.t = 10_000.0
        self.n.step_companion({"checkins": [{"t": 1000.0, "kind": "help"}]})
        self.assertEqual(self.push.sent, [])


class Messages(unittest.TestCase):
    def test_no_brief_unless_enabled(self):
        _, body, _ = an.message_for("alert", frozen(6.0))
        self.assertNotIn("Brief from the agent", body)
        _, body, _ = an.message_for("alert", frozen(6.0), include_brief=True)
        self.assertIn("Brief from the agent", body)

    def test_cue_wording(self):
        _, body, _ = an.message_for("alert", frozen(6.0))
        self.assertIn("light and vibration cue at 96 bpm", body)


class Requests(unittest.TestCase):
    def test_ntfy_request(self):
        req = an.Ntfy("stepcue-abc", "https://ntfy.example/", "tk_1").request("T", "hello", 5, "rotating_light")
        self.assertEqual(req.full_url, "https://ntfy.example/stepcue-abc")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.data, b"hello")
        self.assertEqual(req.headers["Priority"], "5")
        self.assertEqual(req.headers["Authorization"], "Bearer tk_1")

    def test_twilio_request(self):
        c = an.TwilioCall("AC1", "secret", "+15550001111", "+15552223333")
        req = c.request("Freeze <now> & check")
        self.assertEqual(req.full_url, "https://api.twilio.com/2010-04-01/Accounts/AC1/Calls.json")
        form = urllib.parse.parse_qs(req.data.decode())
        self.assertEqual(form["To"], ["+15552223333"])
        self.assertEqual(form["From"], ["+15550001111"])
        self.assertIn("Freeze &lt;now&gt; &amp; check", form["Twiml"][0])       # XML-escaped
        auth = base64.b64decode(req.headers["Authorization"].split()[1]).decode()
        self.assertEqual(auth, "AC1:secret")

    def test_channels_from_env(self):
        push, call = an.channels_from_env({"NTFY_TOPIC": "t"})
        self.assertIsNotNone(push)
        self.assertIsNone(call)
        self.assertEqual(push.url, "https://ntfy.sh")
        _, call = an.channels_from_env({"TWILIO_ACCOUNT_SID": "a", "TWILIO_AUTH_TOKEN": "b",
                                        "TWILIO_FROM": "c", "CAREGIVER_PHONE": "d"})
        self.assertIsNotNone(call)
        self.assertEqual(an.channel_status({}), {"push": False, "call": False, "push_server": None})


if __name__ == "__main__":
    unittest.main()
