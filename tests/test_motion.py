import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion import MotionTracker, track  # noqa: E402

RATE = 50


def signal(seconds, stride_hz=0.0, swing=200.0, yaw_dps=0.0, gyro_bias=(0, 0, 0), seed=0):
    """Sensor flat (gravity on +z, MPU6050 style): swing about x, turning about z."""
    r = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    acc = np.tile([0.0, 0.0, 1.0], (n, 1)) + 0.003 * r.standard_normal((n, 3))
    gyr = np.zeros((n, 3)) + np.asarray(gyro_bias, float)
    if stride_hz:
        # one big positive swing peak per stride, small backswing
        ph = 2 * math.pi * stride_hz * t
        gyr[:, 0] += swing * np.clip(np.sin(ph), 0, None) ** 2 - 0.15 * swing * np.clip(-np.sin(ph), 0, None)
    gyr[:, 2] += yaw_dps
    gyr += 0.3 * r.standard_normal((n, 3))
    return np.hstack([acc, gyr])


def rotation(seed=3):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def rotate(data, R):
    return np.hstack([data[:, :3] @ R.T, data[:, 3:] @ R.T])


def turns(events):
    return [e for e in events if e["type"] == "turn"]


class Stillness(unittest.TestCase):
    def test_still_has_no_steps_or_turns(self):
        m, ev = track(signal(20))
        self.assertEqual(m.steps, 0)
        self.assertEqual(turns(ev), [])
        self.assertIsNone(m.cadence())

    def test_gyro_bias_does_not_become_a_turn(self):
        # 4 deg/s of bias on the vertical axis would integrate to 240 deg per minute
        m, ev = track(signal(60, gyro_bias=(0, 0, 4.0)))
        self.assertEqual(turns(ev), [])
        self.assertAlmostEqual(m.bias[2], 4.0, delta=0.5)


class Walking(unittest.TestCase):
    def test_counts_one_stride_per_swing(self):
        m, ev = track(signal(30, stride_hz=0.9))
        self.assertAlmostEqual(m.strides, 27, delta=1)
        self.assertEqual(m.steps, 2 * m.strides)
        self.assertAlmostEqual(m.cadence(), 108, delta=3)
        self.assertEqual(turns(ev), [])

    def test_slow_parkinsonian_cadence(self):
        m, _ = track(signal(30, stride_hz=0.6))       # 72 steps/min
        self.assertAlmostEqual(m.cadence(), 72, delta=3)

    def test_small_shuffle_is_not_a_stride(self):
        m, _ = track(signal(20, stride_hz=1.5, swing=40))
        self.assertEqual(m.strides, 0)

    def test_cadence_goes_away_after_stopping(self):
        data = np.vstack([signal(10, stride_hz=1.0), signal(5)])
        m, _ = track(data)
        self.assertIsNone(m.cadence())


class Turning(unittest.TestCase):
    def test_left_turn_while_walking(self):
        data = np.vstack([signal(5, stride_hz=1.0), signal(2, stride_hz=1.0, yaw_dps=90),
                          signal(5, stride_hz=1.0)])
        m, ev = track(data)
        t = turns(ev)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["direction"], "left")
        self.assertAlmostEqual(t[0]["angle_deg"], 180, delta=20)
        self.assertEqual(m.summary()["turns_left"], 1)

    def test_right_turn(self):
        data = np.vstack([signal(3), signal(1.5, yaw_dps=-60), signal(3)])
        t = turns(track(data)[1])
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["direction"], "right")
        self.assertAlmostEqual(t[0]["angle_deg"], -90, delta=15)

    def test_tiny_heading_change_is_ignored(self):
        data = np.vstack([signal(3), signal(1.0, yaw_dps=25), signal(3)])   # 25 deg
        self.assertEqual(turns(track(data)[1]), [])

    def test_mirrored_sign_convention(self):
        data = np.vstack([signal(3), signal(2, yaw_dps=90), signal(3)])
        t = turns(track(data, yaw_sign=-1.0)[1])
        self.assertEqual(t[0]["direction"], "right")

    def test_turning_flag_live(self):
        m = MotionTracker()
        flags = []
        for row in np.vstack([signal(3), signal(3, yaw_dps=90)]):
            m.update(row)
            flags.append(m.turning)
        self.assertFalse(any(flags[:140]))
        self.assertTrue(flags[-1])


class Robustness(unittest.TestCase):
    def test_any_mounting_orientation_gives_same_answer(self):
        data = np.vstack([signal(5, stride_hz=1.0), signal(2, stride_hz=1.0, yaw_dps=90),
                          signal(5, stride_hz=1.0)])
        m0, e0 = track(data)
        m1, e1 = track(rotate(data, rotation()))
        self.assertEqual(m0.strides, m1.strides)
        self.assertEqual([t["direction"] for t in turns(e0)], [t["direction"] for t in turns(e1)])
        self.assertAlmostEqual(turns(e0)[0]["angle_deg"], turns(e1)[0]["angle_deg"], delta=5)

    def test_missing_samples_keep_time(self):
        data = signal(12, stride_hz=1.0)
        data[200:210] = np.nan
        m, _ = track(data)
        self.assertEqual(m.n, len(data))
        self.assertAlmostEqual(m.strides, 12, delta=1)


if __name__ == "__main__":
    unittest.main()
