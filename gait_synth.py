"""Synthetic ankle-IMU signal generator (stand-in until real recordings exist).

Used by sim_esp32.py (live stream) and train_classifier.py (bootstrap data).
It is a hand-built approximation of ankle motion, NOT real patient data:
  WALKING     periodic stride swing with heel-strike impacts
  STATIONARY  gravity plus sensor noise and slow sway
  FREEZE      3-8 Hz leg trembling with almost no stride-rate motion
Output per sample: ax, ay, az (g), gx, gy, gz (deg/s).
"""
import math

import numpy as np

from config import SAMPLE_RATE_HZ

MODES = ("WALKING", "STATIONARY", "FREEZE")


def _rotation(rng, max_deg):
    """Small random mounting rotation so the model can't rely on exact axes."""
    a, b, c = np.radians(rng.uniform(-max_deg, max_deg, 3))
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz


class GaitSynth:
    def __init__(self, seed=None, rate=SAMPLE_RATE_HZ, cadence=None, tilt_deg=20.0):
        self.rng = np.random.default_rng(seed)
        self.rate = rate
        self.dt = 1.0 / rate
        self.rot = _rotation(self.rng, tilt_deg)
        self.base_cadence = cadence
        self.mode = "STATIONARY"
        self.phase = 0.0
        self.tremor_phase = 0.0
        self.sway_phase = 0.0
        self.impact = 0.0
        self.yaw_dps = 0.0              # body turning rate (left positive), any mode
        self._new_segment()

    def _new_segment(self):
        r = self.rng
        self.cadence = self.base_cadence or r.uniform(70, 125)   # steps/min
        self.walk_amp = r.uniform(0.6, 1.3)
        self.tremor_hz = r.uniform(4.0, 7.0)
        self.tremor_amp = r.uniform(0.6, 1.4)
        self.noise = r.uniform(0.7, 1.5)

    def set_mode(self, mode, cadence=None):
        if mode not in MODES:
            raise ValueError(mode)
        if mode != self.mode:
            self.mode = mode
            self._new_segment()
        if cadence:
            self.cadence = cadence

    def sample(self):
        r, dt = self.rng, self.dt
        acc = np.array([0.0, 0.0, 1.0])          # gravity along sensor z
        gyr = np.zeros(3)
        self.sway_phase += 2 * math.pi * 0.3 * dt

        if self.mode == "WALKING":
            f = self.cadence / 120.0 * (1 + 0.02 * r.standard_normal())
            prev = self.phase
            self.phase = (self.phase + 2 * math.pi * f * dt) % (2 * math.pi)
            if self.phase < prev:                # heel strike once per stride
                self.impact = r.uniform(0.8, 1.6)
            p, A = self.phase, self.walk_amp
            acc += A * np.array([
                0.55 * math.sin(p) + 0.25 * math.sin(2 * p + 0.6),
                0.12 * math.sin(p + 1.0),
                0.35 * math.sin(p + 1.3) + 0.15 * math.sin(2 * p),
            ])
            acc += self.impact * np.array([-0.3, 0.05, 1.0])
            self.impact *= 0.45
            gyr += A * np.array([
                18 * math.sin(p + 0.4),
                170 * math.sin(p) + 60 * math.sin(2 * p + 0.9),
                22 * math.sin(p + 2.0),
            ])
            an, gn = 0.04, 4.0
        elif self.mode == "FREEZE":
            self.tremor_hz += 0.05 * r.standard_normal()
            self.tremor_hz = min(7.5, max(3.5, self.tremor_hz))
            self.tremor_phase += 2 * math.pi * self.tremor_hz * dt
            q = self.tremor_phase
            env = self.tremor_amp * (0.75 + 0.25 * math.sin(self.sway_phase * 1.7))
            acc += env * np.array([0.22 * math.sin(q), 0.06 * math.sin(q + 0.8),
                                   0.16 * math.sin(q + 1.6)])
            gyr += env * np.array([6 * math.sin(q + 0.3), 38 * math.sin(q), 5 * math.sin(q + 1.1)])
            an, gn = 0.025, 3.0
        else:  # STATIONARY
            acc += 0.01 * np.array([math.sin(self.sway_phase), math.cos(self.sway_phase * 0.7), 0])
            gyr += 1.0 * np.array([math.sin(self.sway_phase), 0, math.cos(self.sway_phase)])
            an, gn = 0.008, 0.8

        gyr[2] += self.yaw_dps                   # turning: rotation about gravity
        acc += self.noise * an * r.standard_normal(3)
        gyr += self.noise * gn * r.standard_normal(3)
        return np.concatenate([self.rot @ acc, self.rot @ gyr])

    def run(self, mode, seconds):
        self.set_mode(mode)
        return np.array([self.sample() for _ in range(int(seconds * self.rate))])
