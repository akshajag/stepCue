import os
import random
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import companion_server as srv  # noqa: E402
import companion_store as cs  # noqa: E402
import location_io  # noqa: E402
import personal_model as pm  # noqa: E402

NOW = 1_800_000_000.0

# what each activity roughly looks like from a pocket phone (+ jitter per sample)
LOOK = {
    "sitting": dict(motion="still", acc_rms=0.05, gyro_rms=0.03, vert_rms=0.02, freeze_index=0.3, cadence=0,
                    steps_10s=0, pitch=-80, roll=5, in_pocket=True),
    "standing": dict(motion="still", acc_rms=0.12, gyro_rms=0.06, vert_rms=0.06, freeze_index=0.6, cadence=0,
                     steps_10s=0, pitch=-5, roll=2, in_pocket=True),
    "walking": dict(motion="walking", acc_rms=2.4, gyro_rms=1.1, vert_rms=1.6, freeze_index=0.4, cadence=104,
                    steps_10s=17, pitch=-10, roll=4, in_pocket=True),
    "frozen": dict(motion="moving", acc_rms=0.9, gyro_rms=0.5, vert_rms=0.6, freeze_index=3.5, cadence=0,
                   steps_10s=1, pitch=-12, roll=3, in_pocket=True),
}


def reading(label, rnd):
    r = dict(LOOK[label])
    for k in ("acc_rms", "gyro_rms", "vert_rms", "freeze_index"):
        r[k] = max(0.0, r[k] * rnd.uniform(0.75, 1.25))
    r["pitch"] += rnd.uniform(-6, 6)
    if r["cadence"]:
        r["cadence"] += rnd.uniform(-8, 8)
    return r


def ankle_state(label, rnd, t):
    base = {"sitting": 0.1, "standing": 0.3, "walking": 2.5, "frozen": 1.2}[label]
    feats = [base * rnd.uniform(0.8, 1.2) for _ in pm.ANKLE_FEATURES]
    feats[3] = 3.0 if label == "frozen" else 0.5                # ankle freeze index
    return {"ankle": {"t": t, "features": feats}}


class Personal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(pm, "TRAIN_DIR", self.tmp.name)
        p.start()
        self.addCleanup(p.stop)
        pm._MODEL_CACHE.clear()
        pm._RECENT.clear()

    def record_all(self, with_ankle=True, segments=2, secs=25):
        rnd = random.Random(1)
        t = NOW
        for _ in range(segments):
            for label in pm.LABELS:
                pm.set_label(label, now=t)
                for i in range(secs + 3):                       # first 3 s are skipped (settle)
                    t += 1
                    st = ankle_state(label, rnd, t) if with_ankle else {}
                    pm.record(reading(label, rnd), st, now=t)
        pm.stop_label()
        return rnd, t

    def test_profiles_and_labels(self):
        with self.assertRaises(ValueError):
            pm.set_label("walking")                             # no patient yet
        pid = pm.add_patient("Akshaj Agadi")
        self.assertEqual(pm.active(), (pid, "Akshaj Agadi"))
        with self.assertRaises(ValueError):
            pm.set_label("dancing")
        pm.set_label("sitting", now=NOW)
        self.assertFalse(pm.record(LOOK["sitting"], {}, now=NOW + 1))      # still settling
        self.assertTrue(pm.record(LOOK["sitting"], {}, now=NOW + 4))
        self.assertEqual(pm.session(now=NOW + pm.SESSION_TIMEOUT_SEC + 5)["label"], None)  # forgotten label stops
        pm.stop_label()
        self.assertFalse(pm.record(LOOK["sitting"], {}, now=NOW + 10))
        self.assertEqual(pm.dataset_summary(pid)["counts"]["sitting"], 1)

    def test_train_needs_two_labels(self):
        pm.add_patient("P")
        pm.set_label("walking", now=NOW)
        for i in range(30):
            pm.record(LOOK["walking"], {}, now=NOW + 4 + i)
        with self.assertRaises(ValueError):
            pm.train(pm.active()[0])

    def test_train_and_predict_with_ankle(self):
        pid = pm.add_patient("Akshaj Agadi")
        rnd, t = self.record_all()
        r = pm.train(pid, now=t)
        self.assertEqual(r["labels"], list(pm.LABELS))
        self.assertEqual(r["cv"], "by recording segment")
        self.assertIsNotNone(r["both"])
        self.assertGreater(r["phone"]["accuracy"], 0.9)
        self.assertGreater(r["both"]["accuracy"], 0.9)
        self.assertTrue(90 <= r["pace"]["median"] <= 115)
        for label in pm.LABELS:
            pm._RECENT.clear()
            out = pm.predict(reading(label, rnd), ankle_state(label, rnd, t), pid, now=t)
            self.assertEqual(out["label"], label)
            self.assertEqual(out["source"], "phone + ankle")
        out = pm.predict(reading("frozen", rnd), {}, pid, now=t, smooth=False)   # ankle missing -> phone model
        self.assertEqual((out["label"], out["source"]), ("frozen", "phone"))

    def test_one_segment_is_flagged_optimistic(self):
        pid = pm.add_patient("P")
        self.record_all(with_ankle=False, segments=1, secs=15)
        r = pm.train(pid)
        self.assertIn("optimistic", r["cv"])
        self.assertIsNone(r["both"])

    def test_server_train_actions_and_live_phase(self):
        comp = os.path.join(self.tmp.name, "c.json")
        loc = os.path.join(self.tmp.name, "l.json")
        for target, attr, val in ((cs, "COMPANION_FILE", comp), (srv, "LOCATION_FILE", loc),
                                  (location_io, "LOCATION_FILE", loc)):
            p = mock.patch.object(target, attr, val)
            p.start()
            self.addCleanup(p.stop)
        with mock.patch.object(srv, "read_state", return_value={}):
            self.assertEqual(srv.handle_api("/api/train", {"action": "train"})[0], 400)   # no patient
            pm.add_patient("Akshaj Agadi")
            code, out = srv.handle_api("/api/train", {"action": "label", "label": "walking"})
            self.assertEqual((code, out["training"]["label"]), (200, "walking"))
            self.assertEqual(srv.handle_api("/api/train", {"action": "label", "label": "nap"})[0], 400)
            srv.handle_api("/api/train", {"action": "stop"})
            self.record_all(with_ankle=False, segments=2, secs=12)
            code, out = srv.handle_api("/api/train", {"action": "train"})
            self.assertEqual(code, 200)
            self.assertGreater(out["accuracy"], 0.8)
            self.assertFalse(out["uses_ankle"])
            rnd = random.Random(5)
            for _ in range(3):
                code, _ = srv.handle_api("/api/android", reading("walking", rnd))
            self.assertEqual(code, 200)
            status = srv.status_payload()
        self.assertEqual(status["phase"]["label"], "walking")
        self.assertEqual(status["training"]["model"]["labels"], list(pm.LABELS))
        self.assertEqual(cs.read_companion()["android"]["phase"]["label"], "walking")


if __name__ == "__main__":
    unittest.main()
