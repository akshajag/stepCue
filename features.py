"""Window features for the gait classifier.

A window is an (N, 6) array of [ax, ay, az, gx, gy, gz] at 50 Hz
(accel in g, gyro in deg/s). Band powers are summed over the three axes of a
sensor, which makes them independent of how the ankle node is rotated.

Freeze Index = power in the 3-8 Hz "freeze" band / power in the 0.5-3 Hz
"locomotor" band of the ankle acceleration (Moore et al. 2008). A small floor
on the denominator keeps it near 0 when the leg is simply still.
"""
import numpy as np

from config import SAMPLE_RATE_HZ

LOCO_BAND = (0.5, 3.0)
FREEZE_BAND = (3.0, 8.0)
FI_FLOOR = 0.005      # g^2; below this the leg is effectively not moving
FI_MAX = 10.0


def _band_powers(x, rate):
    """x: (N, 3). Returns (loco, freeze, total, dominant_hz), axes summed.

    Powers are scaled so that total ~= summed variance of the three axes.
    """
    n = x.shape[0]
    x = x - x.mean(axis=0)
    win = np.hanning(n)[:, None]
    spec = np.abs(np.fft.rfft(x * win, axis=0)) ** 2
    spec = spec.sum(axis=1)
    spec *= 2.0 / (n * (win ** 2).sum())
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    loco = spec[(freqs >= LOCO_BAND[0]) & (freqs < LOCO_BAND[1])].sum()
    frz = spec[(freqs >= FREEZE_BAND[0]) & (freqs <= FREEZE_BAND[1])].sum()
    total = spec[1:].sum()
    dom = freqs[1:][np.argmax(spec[1:])] if n > 2 else 0.0
    return loco, frz, total, dom


def freeze_index(window, rate=SAMPLE_RATE_HZ):
    """Freeze Index of the accelerometer part of a window, clipped to 0-10."""
    loco, frz, _, _ = _band_powers(np.asarray(window, dtype=float)[:, :3], rate)
    return float(min(FI_MAX, frz / (loco + FI_FLOOR)))


ACCEL_FEATURES = [
    "a_loco", "a_freeze", "a_total", "a_fi", "a_dom_hz",
    "a_mag_mean", "a_mag_std", "a_mag_range", "a_jerk_rms",
]
GYRO_FEATURES = [
    "g_loco", "g_freeze", "g_total", "g_ratio", "g_dom_hz",
    "g_mag_mean", "g_mag_std", "g_mag_max",
]


def feature_names(use_gyro=True):
    return ACCEL_FEATURES + (GYRO_FEATURES if use_gyro else [])


def extract_features(window, use_gyro=True, rate=SAMPLE_RATE_HZ):
    w = np.asarray(window, dtype=float)
    acc = w[:, :3]
    loco, frz, total, dom = _band_powers(acc, rate)
    mag = np.linalg.norm(acc, axis=1)
    jerk = np.diff(acc, axis=0) * rate
    feats = [
        np.log10(loco + 1e-6), np.log10(frz + 1e-6), np.log10(total + 1e-6),
        min(FI_MAX, frz / (loco + FI_FLOOR)), dom,
        mag.mean(), mag.std(), np.ptp(mag),
        np.log10(np.sqrt((jerk ** 2).sum(axis=1).mean()) + 1e-6),
    ]
    if use_gyro:
        gyr = w[:, 3:6]
        gl, gf, gt, gd = _band_powers(gyr, rate)
        gmag = np.linalg.norm(gyr, axis=1)
        feats += [
            np.log10(gl + 1e-3), np.log10(gf + 1e-3), np.log10(gt + 1e-3),
            min(FI_MAX, gf / (gl + 25.0)), gd,
            gmag.mean(), gmag.std(), gmag.max(),
        ]
    return np.asarray(feats, dtype=float)


def estimate_cadence(samples, rate=SAMPLE_RATE_HZ):
    """Steps/min from a few seconds of walking data, or None if unclear.

    One ankle swings once per stride (two steps), so cadence = 120 * stride Hz.
    """
    w = np.asarray(samples, dtype=float)
    if w.shape[0] < rate * 2:
        return None
    sig = w[:, 3:6] if np.abs(w[:, 3:6]).max() > 1.0 else w[:, :3]
    sig = sig - sig.mean(axis=0)
    n = 1024
    spec = (np.abs(np.fft.rfft(sig * np.hanning(len(sig))[:, None], n=n, axis=0)) ** 2).sum(axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    band = (freqs >= 0.5) & (freqs <= 1.5)       # 60-180 steps/min
    if spec[band].max() < 4 * np.median(spec[1:]) + 1e-12:
        return None
    return int(round(120.0 * freqs[band][np.argmax(spec[band])]))
