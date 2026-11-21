"""Benchmark the gait classifier on real freezes from people the model never saw.

  python benchmark_datasets.py              # uses datasets/fogstar and datasets/mhealth

Evaluation: 5-fold cross-validation over FoG-STAR *subjects* (each fold tests
on ~4-5 people with Parkinson's who were not in training). Compares training
sources:
  A  synthetic only            (what ships today)
  B  synthetic + MHEALTH       (real walking / standing, no freezes)
  C  FoG-STAR only             (real freezes, other people)
  D  synthetic + MHEALTH + FoG-STAR

Metrics:
  window level   per-class F1, freeze recall by freeze type
  episode level  what the wearer feels: share of freeze episodes that trigger a
                 cue (3 agreeing windows, same debounce as gait_classifier.py),
                 median time from freeze onset to cue, false cues per hour
Features are cached in datasets/cache/ so reruns are fast.
"""
import argparse
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import DEBOUNCE_WINDOWS, HOP_SAMPLES, SAMPLE_RATE_HZ, STATUSES
from train_classifier import (FOGSTAR_SEVERITY, fogstar_streams, mhealth_set,
                              stream_windows, synthetic_set)

FRZ = STATUSES.index("FREEZE")
HOP_SEC = HOP_SAMPLES / SAMPLE_RATE_HZ
CACHE = os.path.join("datasets", "cache")


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    t0 = time.time()
    out = fn()
    np.savez_compressed(path, **out)
    print(f"  cached {name} ({time.time() - t0:.0f} s)")
    return out


def build_fogstar(folder):
    Xs, ends, ys, sevs, subj, sid = [], [], [], [], [], []
    for k, st in enumerate(fogstar_streams(folder)):
        X, e, y, s = stream_windows(st, use_gyro=True)
        if not len(X):
            continue
        Xs.append(X); ends.append(e); ys.append(y); sevs.append(s)
        subj.append(np.full(len(y), st["subject"])); sid.append(np.full(len(y), k))
    return {"X": np.vstack(Xs), "end": np.concatenate(ends), "y": np.concatenate(ys),
            "sev": np.concatenate(sevs), "subject": np.concatenate(subj), "stream": np.concatenate(sid)}


def build_synthetic():
    X, y = synthetic_set(range(0, 24), use_gyro=True)
    return {"X": np.array(X), "y": np.array(y)}


def build_mhealth(folder):
    X, y, g = mhealth_set(folder, use_gyro=True)
    return {"X": np.array(X), "y": np.array(y)}


def episodes_eval(pred, fs, idx):
    """Episode metrics on the test windows `idx` (time-ordered within streams)."""
    detected, latencies, false_cues, nonfreeze_sec, n_eps = 0, [], 0, 0.0, 0
    sev_hits = {s: [0, 0] for s in FOGSTAR_SEVERITY}
    for k in np.unique(fs["stream"][idx]):
        m = idx[fs["stream"][idx] == k]
        m = m[np.argsort(fs["end"][m])]
        end, y, sev, p = fs["end"][m], fs["y"][m], fs["sev"][m], pred[np.searchsorted(idx, m)]
        # Same state machine as gait_classifier.py: the state changes only when
        # DEBOUNCE_WINDOWS consecutive predictions agree on a new class.
        state, recent, onset_list = -1, [], []
        for i in range(len(m)):
            if i > 0 and end[i] - end[i - 1] != HOP_SAMPLES:
                recent = []                      # gap in the data: start over
            recent = (recent + [p[i]])[-DEBOUNCE_WINDOWS:]
            if len(recent) == DEBOUNCE_WINDOWS and p[i] != state and all(r == p[i] for r in recent):
                state = p[i]
                if state == FRZ:
                    onset_list.append(i)
        onsets = np.array(onset_list, dtype=int)
        # true episodes: runs of windows whose recent half is FREEZE
        is_f = y == FRZ
        starts = np.where(is_f & ~np.r_[False, is_f[:-1]])[0]
        stops = np.where(is_f & ~np.r_[is_f[1:], False])[0]
        used = np.zeros(len(onsets), bool)
        for a, b in zip(starts, stops):
            n_eps += 1
            s_type = int(np.bincount(sev[a:b + 1]).argmax())
            grace = b + int(1.0 / HOP_SEC)
            hit = np.where((onsets >= a) & (onsets <= grace))[0]
            if s_type in sev_hits:
                sev_hits[s_type][1] += 1
            if len(hit):
                detected += 1
                used[hit] = True
                # freeze onset is about half a window before the first all-freeze window
                latencies.append((onsets[hit[0]] - a) * HOP_SEC + 0.6)
                if s_type in sev_hits:
                    sev_hits[s_type][0] += 1
        near = np.zeros(len(m), bool)
        for a, b in zip(starts, stops):
            near[max(0, a - 10):b + 10] = True
        false_cues += int(np.sum(~used & ~near[onsets])) if len(onsets) else 0
        nonfreeze_sec += np.sum((y != FRZ) & (y >= 0)) * HOP_SEC
    return {"episodes": n_eps, "detected": detected,
            "latency_med": float(np.median(latencies)) if latencies else None,
            "false_per_hour": false_cues / max(nonfreeze_sec / 3600, 1e-9),
            "by_type": sev_hits}


def run(fog_dir, mh_dir, folds=5, trees=60):
    print("Loading features (first run extracts and caches them)...")
    fs = cached("fogstar_v1", lambda: build_fogstar(fog_dir))
    syn = cached("synthetic_v1", build_synthetic)
    mh = cached("mhealth_v1", lambda: build_mhealth(mh_dir)) if mh_dir and os.path.isdir(mh_dir) else None
    subjects = np.unique(fs["subject"])
    rng = np.random.default_rng(0)
    order = rng.permutation(subjects)
    fold_of = {s: i % folds for i, s in enumerate(order)}
    fold = np.array([fold_of[s] for s in fs["subject"]])
    print(f"FoG-STAR: {len(fs['y'])} windows, {len(subjects)} subjects, "
          f"{np.mean(fs['y'] == FRZ):.1%} freeze windows; {folds}-fold by subject\n")

    configs = {"A synthetic only": ("syn",), "B synthetic + MHEALTH": ("syn", "mh"),
               "C FoG-STAR only": ("fs",), "D synthetic + MHEALTH + FoG-STAR": ("syn", "mh", "fs")}
    results = {}
    for name, srcs in configs.items():
        if "mh" in srcs and mh is None:
            continue
        preds = np.full(len(fs["y"]), -1)
        for f in range(folds):
            Xtr, ytr = [], []
            if "syn" in srcs:
                Xtr.append(syn["X"]); ytr.append(syn["y"])
            if "mh" in srcs:
                Xtr.append(mh["X"]); ytr.append(mh["y"])
            if "fs" in srcs:
                tr = (fold != f) & (fs["y"] >= 0)
                Xtr.append(fs["X"][tr]); ytr.append(fs["y"][tr])
            model = RandomForestClassifier(n_estimators=trees, max_depth=14, min_samples_leaf=3,
                                           class_weight="balanced", random_state=f, n_jobs=-1)
            model.fit(np.vstack(Xtr), np.concatenate(ytr))
            te = fold == f
            preds[te] = model.predict(fs["X"][te])
        lab = fs["y"] >= 0
        f1 = f1_score(fs["y"][lab], preds[lab], labels=[0, 1, 2], average=None, zero_division=0)
        rec_by_sev = {}
        for code, nm in FOGSTAR_SEVERITY.items():
            m = (fs["y"] == FRZ) & (fs["sev"] == code)
            rec_by_sev[nm] = float(np.mean(preds[m] == FRZ)) if m.any() else None
        ep = episodes_eval(preds, fs, np.arange(len(fs["y"])))
        results[name] = (f1, rec_by_sev, ep)
        print(f"{name}")
        print(f"  window F1   walking {f1[0]:.2f}  standing {f1[1]:.2f}  freeze {f1[2]:.2f}")
        print("  freeze recall by type  " + "  ".join(
            f"{k} {v:.0%}" for k, v in rec_by_sev.items() if v is not None))
        lat = f"{ep['latency_med']:.1f} s" if ep["latency_med"] is not None else "n/a"
        print(f"  episodes cued {ep['detected']}/{ep['episodes']} ({ep['detected'] / max(ep['episodes'], 1):.0%}), "
              f"median onset-to-cue {lat}, false cues {ep['false_per_hour']:.1f}/h")
        print("  episodes cued by type  " + "  ".join(
            f"{FOGSTAR_SEVERITY[k]} {h}/{n}" for k, (h, n) in ep["by_type"].items() if n) + "\n")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fogstar", default=os.path.join("datasets", "fogstar"))
    ap.add_argument("--mhealth", default=os.path.join("datasets", "mhealth", "MHEALTHDATASET"))
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    run(a.fogstar, a.mhealth, a.folds)
