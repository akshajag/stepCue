"""Check motion.py (turns and steps) against FoG-STAR's video-annotated activities.

  python benchmark_motion.py

FoG-STAR labels every sample as walk / sit / stand / transitions / turn right /
turn left, so turn detection can be scored directly. There are no step labels,
so steps are checked indirectly:
  * both ankles of the same person should count the same number of strides
  * stride-based cadence should match an autocorrelation estimate of the stride period
  * sitting and standing should produce (almost) no strides
It also reports how many freezes happened during turns, the pattern a
caregiver would see.
"""
import argparse
import os

import numpy as np

from motion import track

CACHE = os.path.join("datasets", "cache", "fogstar_motion_v1.npz")
RATE = 50
WALK, SIT, STAND, TURN_R, TURN_L = 1, 2, 3, 6, 7
FOGSTAR_YAW_SIGN = -1.0   # FoG-STAR gyro/accel sign convention is mirrored vs the MPU6050


def load(folder):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        n = len([f for f in z.files if f.startswith("d")])
        return [{"data": z[f"d{k}"], "act": z[f"a{k}"], "fog": z[f"f{k}"],
                 "subject": int(z[f"m{k}"][0]), "session": int(z[f"m{k}"][1]),
                 "side": "ankleR" if z[f"m{k}"][2] == 0 else "ankleL"} for k in range(n)]
    import pandas as pd
    df = pd.read_csv(os.path.join(folder, "sensor_data.csv"))
    out, streams = {}, []
    for (subj, sess), g in df.groupby(["subjectID", "sessionID"], sort=True):
        t = g["timestamp"].to_numpy(float)
        t50 = np.arange(t[0], t[-1], 1.0 / RATE)
        idx = np.clip(np.searchsorted(t, t50), 0, len(t) - 1)
        for side in ("ankleR", "ankleL"):
            cols = [f"{side}_{s}_{a}" for s in ("acc", "gyro") for a in "xyz"]
            k = len(streams)
            st = {"data": np.column_stack([np.interp(t50, t, g[c].to_numpy(float)) for c in cols]),
                  "act": g["activity"].to_numpy()[idx], "fog": g["fog"].to_numpy()[idx],
                  "subject": int(subj), "session": int(sess), "side": side}
            streams.append(st)
            out.update({f"d{k}": st["data"], f"a{k}": st["act"], f"f{k}": st["fog"],
                        f"m{k}": np.array([subj, sess, 0 if side == "ankleR" else 1])})
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return streams


def autocorr_cadence(data):
    """Reference cadence (steps/min): first autocorrelation peak of gyro magnitude
    between 0.6 and 2.5 s, i.e. one stride of this leg."""
    sig = np.linalg.norm(data[:, 3:6], axis=1)
    sig = sig - sig.mean()
    ac = np.correlate(sig, sig, "full")[len(sig) - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo, hi = int(0.6 * RATE), min(int(2.5 * RATE), len(ac) - 2)
    for i in range(lo, hi):
        if ac[i] > 0.3 and ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]:
            return 120.0 * RATE / i
    return None


def runs(mask):
    m = np.r_[0, mask.astype(int), 0]
    return list(zip(np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0]))   # [start, stop)


def evaluate(streams):
    turns_true = turns_hit = dir_ok = 0
    det_total = det_matched = 0
    false_walk_turns, straight_sec = 0, 0.0
    angles = []
    still_strides, still_sec = 0, 0.0
    cad_pairs = []
    lr = {}
    fog_eps = fog_in_turn_label = fog_in_turn_det = 0
    for st in streams:
        data, act, fog = st["data"], st["act"], st["fog"]
        bad = ~np.isfinite(data).all(axis=1)
        if bad.mean() > 0.5:
            continue
        m, events = track(data, yaw_sign=FOGSTAR_YAW_SIGN)
        turns = [e for e in events if e["type"] == "turn"]
        strides = np.array([e["t"] for e in events if e["type"] == "stride"])
        det = np.zeros(len(act), bool)
        for e in turns:
            det[int(e["t_start"] * RATE):int(e["t_end"] * RATE) + 1] = True

        # turns: recall and direction
        lab_turn = np.isin(act, (TURN_R, TURN_L)) & ~bad
        for a, b in runs(lab_turn):
            if (b - a) / RATE < 1.0:
                continue
            turns_true += 1
            want = "left" if act[a] == TURN_L else "right"
            lo, hi = (a - RATE // 2) / RATE, (b + RATE // 2) / RATE
            hit = [e for e in turns if e["t_end"] >= lo and e["t_start"] <= hi]
            if hit:
                turns_hit += 1
                best = max(hit, key=lambda e: abs(e["angle_deg"]))
                dir_ok += best["direction"] == want
                angles.append(abs(best["angle_deg"]))
        # turns: precision and false turns during straight walking
        near_turn = np.convolve(lab_turn, np.ones(2 * RATE + 1), "same") > 0
        for e in turns:
            det_total += 1
            seg = near_turn[int(e["t_start"] * RATE):int(e["t_end"] * RATE) + 1]
            if seg.any():
                det_matched += 1
            elif np.mean(act[int(e["t_start"] * RATE):int(e["t_end"] * RATE) + 1] == WALK) > 0.5:
                false_walk_turns += 1
        straight_sec += np.sum((act == WALK) & ~near_turn & ~bad) / RATE

        # strides while sitting / standing (excluding the second after a change)
        still = np.isin(act, (SIT, STAND)) & ~bad
        for a, b in runs(still):
            a2 = a + RATE
            if b - a2 < RATE:
                continue
            still_sec += (b - a2) / RATE
            still_strides += int(np.sum((strides >= a2 / RATE) & (strides < b / RATE)))

        # cadence on straight walking bouts of 6 s or more
        walk = (act == WALK) & ~bad & (fog == 0)
        for a, b in runs(walk):
            if (b - a) / RATE < 6:
                continue
            s = strides[(strides >= a / RATE) & (strides < b / RATE)]
            spec = autocorr_cadence(data[a:b])
            if len(s) >= 3 and spec:
                cad_pairs.append((120.0 / np.median(np.diff(s)), spec))

        key = (st["subject"], st["session"])
        lr.setdefault(key, {})[st["side"]] = (m.strides, len(turns))

        # freezes during turns (count once per person: right ankle)
        if st["side"] == "ankleR":
            for a, b in runs(fog == 1):
                fog_eps += 1
                fog_in_turn_label += bool(lab_turn[a:b].any())
                fog_in_turn_det += bool(det[max(0, a - RATE):b + RATE].any())

    pairs = [(v["ankleR"], v["ankleL"]) for v in lr.values() if len(v) == 2]
    stride_diff = [abs(r[0] - l[0]) / max(r[0], l[0], 1) for r, l in pairs]
    cad = np.array(cad_pairs)
    return {
        "turns_labelled": turns_true, "turns_found": turns_hit,
        "turn_recall": turns_hit / max(turns_true, 1),
        "direction_acc": dir_ok / max(turns_hit, 1),
        "turn_precision": det_matched / max(det_total, 1),
        "false_turns_per_min_straight": false_walk_turns / max(straight_sec / 60, 1e-9),
        "angle_median": float(np.median(angles)) if angles else None,
        "angle_iqr": tuple(np.percentile(angles, [25, 75]).round()) if angles else None,
        "stride_lr_diff_median": float(np.median(stride_diff)) if stride_diff else None,
        "cadence_bouts": len(cad),
        "cadence_mae": float(np.mean(np.abs(cad[:, 0] - cad[:, 1]))) if len(cad) else None,
        "cadence_within_10pct": float(np.mean(np.abs(cad[:, 0] - cad[:, 1]) <= 0.1 * cad[:, 1])) if len(cad) else None,
        "still_strides_per_min": still_strides / max(still_sec / 60, 1e-9),
        "fog_episodes": fog_eps, "fog_in_labelled_turn": fog_in_turn_label,
        "fog_in_detected_turn": fog_in_turn_det,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fogstar", default=os.path.join("datasets", "fogstar"))
    a = ap.parse_args()
    r = evaluate(load(a.fogstar))
    print(f"Turns  found {r['turns_found']}/{r['turns_labelled']} labelled turns ({r['turn_recall']:.0%}), "
          f"direction right {r['direction_acc']:.0%}")
    print(f"       {r['turn_precision']:.0%} of detected turns are real turns; "
          f"{r['false_turns_per_min_straight']:.2f} false turns/min on straight walking")
    print(f"       turn angle median {r['angle_median']:.0f} deg (IQR {r['angle_iqr'][0]:.0f}-{r['angle_iqr'][1]:.0f})")
    print(f"Steps  left vs right ankle stride count differ by {r['stride_lr_diff_median']:.1%} (median per session)")
    print(f"       cadence vs autocorrelation estimate: MAE {r['cadence_mae']:.1f} steps/min, "
          f"{r['cadence_within_10pct']:.0%} within 10% ({r['cadence_bouts']} walking bouts)")
    print(f"       {r['still_strides_per_min']:.2f} false strides/min while sitting or standing")
    print(f"Freezes  {r['fog_in_labelled_turn']}/{r['fog_episodes']} freeze episodes happened during a labelled turn; "
          f"detector flagged a turn around {r['fog_in_detected_turn']}")
    return r


if __name__ == "__main__":
    main()
