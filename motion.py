"""Steps, cadence and turns from the single ankle IMU (no magnetometer, no anchors).

Works sample by sample at 50 Hz, so the live loop can feed it the same packets
the classifier sees. It does not care how the sensor is strapped on:

  * gravity direction  = slow low-pass of the accelerometer
  * yaw rate (turning) = gyro component along gravity  (left turn positive)
                         The MPU6050 reads +1 g "up" at rest and its gyro is
                         right-handed, so a left (counter-clockwise) turn is
                         positive. FoG-STAR's sensors use the opposite sign,
                         so its benchmark passes yaw_sign=-1 (checked against
                         its turn-left / turn-right labels).
  * leg swing          = gyro component perpendicular to gravity

Turns: the yaw rate is smoothed over ~1 stride (which cancels the shank's own
swing wobble), and a turn is any stretch that keeps turning one way for at
least TURN_MIN_DEG. Steps: each big swing peak is one stride of this leg,
i.e. two steps.

Relative measures only. Heading drifts over minutes because there is no
magnetometer, so this never claims a position on a map.
"""
from collections import deque

import numpy as np

from config import SAMPLE_RATE_HZ

GRAVITY_TAU_SEC = 1.0
YAW_SMOOTH_SEC = 1.0
TURN_START_DPS = 20.0      # smoothed yaw rate that opens a turn
TURN_END_DPS = 10.0        # ... and that closes it once it stays below this
TURN_END_HOLD_SEC = 0.4
TURN_MIN_DEG = 45.0
TURN_MIN_SEC = 0.5
SWING_HIGH_DPS = 100.0     # swing peak must exceed this to count as a stride
SWING_LOW_RATIO = 0.5      # swing ends when it falls below this share of the peak
STRIDE_MIN_SEC = 0.6       # refractory period between strides of one leg
STILL_WIN_SEC = 0.5        # stillness window used to learn gyro bias
CADENCE_STRIDES = 6


class MotionTracker:
    def __init__(self, rate=SAMPLE_RATE_HZ, yaw_sign=1.0):
        self.dt = 1.0 / rate
        self.yaw_sign = yaw_sign
        self.n = 0
        self.g = None
        self.g_alpha = self.dt / GRAVITY_TAU_SEC
        self.bias = np.zeros(3)
        self.still = deque(maxlen=int(STILL_WIN_SEC * rate))
        self.yaw_buf = deque(maxlen=int(YAW_SMOOTH_SEC * rate))
        self.yaw_sum = 0.0
        self.heading = 0.0                 # relative, drifts
        # turns
        self.turn = None                   # open turn: dict(start, angle, sign, quiet)
        self.turns = []                    # finished turns
        # strides
        self.in_swing = False
        self.swing_peak = 0.0
        self.last_stride = -1e9
        self.stride_times = []             # recent stride times (bounded)
        self.stride_count = 0

    # ----------------------------------------------------------- outputs --
    @property
    def strides(self):
        return self.stride_count

    @property
    def steps(self):
        return 2 * self.strides

    @property
    def turning(self):
        return self.turn is not None and abs(self.turn["angle"]) >= TURN_MIN_DEG / 2

    def cadence(self, now_sec=None):
        """Steps/min from the last few strides, or None if not walking now."""
        t = self.stride_times[-CADENCE_STRIDES:]
        now = self.n * self.dt if now_sec is None else now_sec
        if len(t) < 3 or now - t[-1] > 2.5:
            return None
        return int(round(120.0 / float(np.median(np.diff(t)))))

    def summary(self):
        left = sum(1 for t in self.turns if t["direction"] == "left")
        return {"steps": self.steps, "cadence_spm": self.cadence(), "turns": len(self.turns),
                "turns_left": left, "turns_right": len(self.turns) - left,
                "turning": self.turning}

    # ------------------------------------------------------------ update --
    def update(self, sample):
        """Feed one sample [ax, ay, az (g), gx, gy, gz (deg/s)].

        Returns a list of events finished on this sample:
          {"type": "turn", "direction", "angle_deg", "duration_sec", "t_end"}
          {"type": "stride", "t"}
        """
        s = np.asarray(sample, dtype=float)
        self.n += 1                                    # time advances even on a bad sample
        if not np.isfinite(s).all():
            return []
        acc, gyr = s[:3], s[3:6]
        t = self.n * self.dt
        events = []

        # gravity direction
        self.g = acc.copy() if self.g is None else self.g + self.g_alpha * (acc - self.g)
        gn = np.linalg.norm(self.g)
        if gn < 1e-6:
            return events
        gh = self.g / gn

        # gyro bias: learn it while the leg is perfectly still
        self.still.append(s)
        if len(self.still) == self.still.maxlen:
            w = np.array(self.still)
            if (np.linalg.norm(w[:, :3], axis=1).std() < 0.02
                    and w[:, 3:].std(axis=0).max() < 1.5
                    and np.abs(w[:, 3:].mean(axis=0)).max() < 10):
                self.bias += 0.05 * (w[:, 3:].mean(axis=0) - self.bias)
        gyr = gyr - self.bias

        along = float(gyr @ gh)
        yaw = self.yaw_sign * along                    # left turn positive
        swing = float(np.linalg.norm(gyr - along * gh))
        self.heading += yaw * self.dt

        # ------------------------------------------------------- turns --
        if len(self.yaw_buf) == self.yaw_buf.maxlen:
            self.yaw_sum -= self.yaw_buf[0]
        self.yaw_buf.append(yaw)
        self.yaw_sum += yaw
        smooth = self.yaw_sum / len(self.yaw_buf)

        if self.turn is None:
            if abs(smooth) >= TURN_START_DPS:
                # the moving average lags half a window: count that part too
                half = list(self.yaw_buf)[len(self.yaw_buf) // 2:]
                self.turn = {"start": t - len(half) * self.dt, "angle": sum(half) * self.dt,
                             "sign": np.sign(smooth), "quiet": 0}
        else:
            tr = self.turn
            tr["angle"] += yaw * self.dt
            if abs(smooth) < TURN_END_DPS or np.sign(smooth) != tr["sign"]:
                tr["quiet"] += 1
            else:
                tr["quiet"] = 0
            if tr["quiet"] * self.dt >= TURN_END_HOLD_SEC:
                end = t - tr["quiet"] * self.dt
                dur = end - tr["start"]
                if abs(tr["angle"]) >= TURN_MIN_DEG and dur >= TURN_MIN_SEC:
                    ev = {"type": "turn", "direction": "left" if tr["angle"] > 0 else "right",
                          "angle_deg": round(float(tr["angle"])), "duration_sec": round(dur, 1),
                          "t_start": round(tr["start"], 2), "t_end": round(end, 2)}
                    self.turns.append(ev)
                    events.append(ev)
                self.turn = None

        # ----------------------------------------------------- strides --
        if not self.in_swing:
            if swing >= SWING_HIGH_DPS and t - self.last_stride >= STRIDE_MIN_SEC:
                self.in_swing, self.swing_peak = True, swing
        else:
            self.swing_peak = max(self.swing_peak, swing)
            if swing < SWING_LOW_RATIO * self.swing_peak:
                self.in_swing = False
                self.last_stride = t
                self.stride_times.append(t)
                self.stride_count += 1
                del self.stride_times[:-200]
                events.append({"type": "stride", "t": round(t, 2)})
        return events


def track(data, rate=SAMPLE_RATE_HZ, yaw_sign=1.0):
    """Run a whole recording (N, 6) through a fresh tracker. Returns (tracker, events)."""
    m = MotionTracker(rate, yaw_sign)
    events = []
    for row in data:
        events += m.update(row)
    return m, events
