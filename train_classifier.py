"""Train the StepCue gait classifier (Random Forest on 1.2 s windows).

Data sources, combinable:
  (default)            synthetic ankle signals from gait_synth.py
  --recordings DIR     labelled CSVs from record_session.py / record_data.py (searched recursively)
  --daphnet DIR        Daphnet FoG dataset .txt files (ankle accelerometer only,
                       so the model is trained without gyro features)
  --mhealth DIR        UCI MHEALTH logs: real 50 Hz ankle accel + gyro for
                       WALKING / STATIONARY (no freezes)
  --fogstar DIR        FoG-STAR folder with sensor_data.csv: real freezes from
                       22 people with PD, both ankles, accel + gyro
  --no-synthetic       leave the synthetic data out

Examples:
  python train_classifier.py
  python train_classifier.py --recordings recordings
  python train_classifier.py --daphnet path/to/dataset --no-synthetic
  python train_classifier.py --mhealth datasets/MHEALTHDATASET --recordings recordings
"""
import argparse
import glob
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config import (HOP_SAMPLES, MODEL_FILE, SAMPLE_RATE_HZ, STATUSES,
                    WINDOW_SAMPLES)
from features import extract_features, feature_names
from gait_synth import MODES, GaitSynth

RECENT = WINDOW_SAMPLES // 2     # a window takes the label of its most recent half


def windows_from_stream(data, labels, use_gyro, hop=HOP_SAMPLES):
    """Slide over a continuous stream; keep windows whose recent half is one class."""
    X, y = [], []
    for end in range(WINDOW_SAMPLES, len(data) + 1, hop):
        recent = labels[end - RECENT:end]
        if recent[0] < 0 or np.any(recent != recent[0]):
            continue
        X.append(extract_features(data[end - WINDOW_SAMPLES:end], use_gyro))
        y.append(recent[0])
    return X, y


def synthetic_set(seeds, use_gyro, seconds_per_subject=150):
    X, y = [], []
    for seed in seeds:
        synth = GaitSynth(seed=seed)
        rng = np.random.default_rng(seed + 10_000)
        chunks, labels, total, last = [], [], 0.0, None
        while total < seconds_per_subject:
            mode = rng.choice([m for m in MODES if m != last])
            dur = float(rng.uniform(2.0, 7.0))
            seg = synth.run(mode, dur)
            chunks.append(seg)
            labels.append(np.full(len(seg), STATUSES.index(mode)))
            total += dur
            last = mode
        Xi, yi = windows_from_stream(np.vstack(chunks), np.concatenate(labels), use_gyro)
        X += Xi
        y += yi
    return X, y


def read_recording(path):
    """Read one recording CSV: ts_ms,ax,ay,az,gx,gy,gz,label (label per row).

    Works for both record_data.py files (one label throughout) and
    record_session.py files (label changes during the session). Rows labelled
    anything other than WALKING / STATIONARY / FREEZE (e.g. UNLABELED) get -1
    and are skipped when windows are cut.
    """
    rows, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 8 or parts[0] == "ts_ms":
                continue
            try:
                rows.append([float(v) for v in parts[1:7]])
            except ValueError:
                continue
            labels.append(STATUSES.index(parts[7]) if parts[7] in STATUSES else -1)
    return np.array(rows).reshape(-1, 6), np.array(labels, dtype=int)


def recordings_set(folder, use_gyro, test_frac=0.25):
    """All recording CSVs in a folder (searched recursively).

    The last quarter of every file is held out for testing, so each class
    appears in both the training and the test set.
    """
    Xtr, ytr, Xte, yte = [], [], [], []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)):
        data, lab = read_recording(path)
        if len(data) < WINDOW_SAMPLES * 2 or not np.any(lab >= 0):
            print(f"  skipping {path} ({len(data)} rows, no usable labels)")
            continue
        cut = int(len(data) * (1 - test_frac))
        a, b = windows_from_stream(data[:cut], lab[:cut], use_gyro)
        c, d = windows_from_stream(data[cut:], lab[cut:], use_gyro)
        Xtr += a
        ytr += b
        Xte += c
        yte += d
        mix = {STATUSES[k]: int(np.sum(np.array(b + d) == k)) for k in range(3) if np.any(np.array(b + d) == k)}
        print(f"  {os.path.relpath(path, folder)}: {len(b)} train / {len(d)} test windows {mix}")
    return Xtr, ytr, Xte, yte


def daphnet_set(folder):
    """Daphnet columns: time_ms, ankle(fwd, vert, lat) mg, thigh x3, trunk x3, annotation.

    annotation 0 = outside experiment, 1 = no freeze, 2 = freeze. 64 Hz.
    'No freeze' is split into WALKING / STATIONARY by ankle movement energy,
    because Daphnet does not label standing separately.
    """
    X, y, groups = [], [], []
    files = sorted(glob.glob(os.path.join(folder, "**", "S*R*.txt"), recursive=True))
    for gi, path in enumerate(files):
        raw = np.loadtxt(path)
        t = raw[:, 0] / 1000.0
        t50 = np.arange(t[0], t[-1], 1.0 / SAMPLE_RATE_HZ)
        acc = np.column_stack([np.interp(t50, t, raw[:, c]) for c in (1, 2, 3)]) / 1000.0
        ann = raw[np.clip(np.searchsorted(t, t50), 0, len(t) - 1), 10].astype(int)
        data = np.column_stack([acc, np.zeros((len(acc), 3))])
        n_before = len(y)
        for end in range(WINDOW_SAMPLES, len(data) + 1, HOP_SAMPLES * 2):
            recent = ann[end - RECENT:end]
            if recent[0] == 0 or np.any(recent != recent[0]):
                continue
            win = data[end - WINDOW_SAMPLES:end]
            if recent[0] == 2:
                label = "FREEZE"
            else:
                moving = np.linalg.norm(win[:, :3] - win[:, :3].mean(axis=0), axis=1).std()
                label = "WALKING" if moving > 0.08 else "STATIONARY"
            X.append(extract_features(win, use_gyro=False))
            y.append(STATUSES.index(label))
        groups += [gi] * (len(y) - n_before)
        print(f"  {os.path.basename(path)}: {len(y) - n_before} windows")
    return X, y, groups


MHEALTH_LABELS = {1: "STATIONARY", 2: "STATIONARY", 4: "WALKING"}   # standing, sitting, walking


def mhealth_set(folder, use_gyro):
    """UCI MHEALTH logs (mHealth_subject<N>.log): 50 Hz, left-ankle accel (m/s^2) + gyro (deg/s).

    Same rate, same body site and same sensor types as the StepCue ankle node,
    which makes it the best public source of real WALKING / STATIONARY ankle
    data. It has no freezes. Columns (0-based): 5-7 ankle accel, 8-10 ankle
    gyro, 23 activity label. Standing and sitting both leave the ankle still,
    so both map to STATIONARY; other activities (stairs, running...) are skipped.
    """
    X, y, groups = [], [], []
    files = sorted(glob.glob(os.path.join(folder, "**", "mHealth_subject*.log"), recursive=True))
    for gi, path in enumerate(files):
        raw = np.loadtxt(path)
        acc = raw[:, 5:8] / 9.80665
        gyr = raw[:, 8:11]
        data = np.column_stack([acc, gyr])
        lab = np.array([STATUSES.index(MHEALTH_LABELS[int(v)]) if int(v) in MHEALTH_LABELS else -1
                        for v in raw[:, 23]])
        n_before = len(y)
        Xi, yi = windows_from_stream(data, lab, use_gyro)
        X += Xi
        y += yi
        groups += [gi] * len(yi)
        print(f"  {os.path.basename(path)}: {len(y) - n_before} windows")
    return X, y, groups


FOGSTAR_WALK = (1, 6, 7)      # walk, turn right, turn left
FOGSTAR_STILL = (2, 3)         # sit, stand
FOGSTAR_SEVERITY = {1: "shuffling", 2: "trembling", 3: "akinesia"}


def fogstar_streams(folder, ankles=("ankleR", "ankleL")):
    """FoG-STAR (Zenodo 10.5281/zenodo.17838806, CC BY 4.0): 22 people with PD, 60 Hz.

    Yields one continuous 50 Hz stream per subject, session and ankle:
      data (N, 6) accel g + gyro deg/s, labels (N,) class index or -1,
      severity (N,) 0 / 1 shuffling / 2 trembling / 3 akinesia.
    Labels: fog == 1 -> FREEZE; walking and turning -> WALKING; sitting and
    standing -> STATIONARY; sit-to-stand transitions and unlabelled -> -1.
    """
    import pandas as pd
    path = os.path.join(folder, "sensor_data.csv")
    df = pd.read_csv(path)
    walk, still, frz = (STATUSES.index(k) for k in ("WALKING", "STATIONARY", "FREEZE"))
    for (subj, sess), g in df.groupby(["subjectID", "sessionID"], sort=True):
        t = g["timestamp"].to_numpy(float)          # seconds (README says ms; values are s)
        t50 = np.arange(t[0], t[-1], 1.0 / SAMPLE_RATE_HZ)
        idx = np.clip(np.searchsorted(t, t50), 0, len(t) - 1)
        act = g["activity"].to_numpy()[idx]
        fog = g["fog"].to_numpy()[idx]
        sev = np.nan_to_num(g["fog_severity"].to_numpy(float)[idx]).astype(int) * (fog == 1)
        lab = np.full(len(t50), -1)
        lab[np.isin(act, FOGSTAR_WALK)] = walk
        lab[np.isin(act, FOGSTAR_STILL)] = still
        lab[fog == 1] = frz
        for side in ankles:
            cols = [f"{side}_{s}_{a}" for s in ("acc", "gyro") for a in "xyz"]
            data = np.column_stack([np.interp(t50, t, g[c].to_numpy(float)) for c in cols])
            yield {"subject": int(subj), "session": int(sess), "side": side,
                   "data": data, "labels": lab, "severity": sev}


def stream_windows(stream, use_gyro, hop=HOP_SAMPLES):
    """Every window of a stream in time order (skips windows with missing data).

    Returns features, window end index, pure label of the recent half (-1 if
    mixed/unlabelled), and the freeze severity in the recent half (0 if none).
    """
    data, lab, sev = stream["data"], stream["labels"], stream["severity"]
    X, ends, y, s = [], [], [], []
    for end in range(WINDOW_SAMPLES, len(data) + 1, hop):
        win = data[end - WINDOW_SAMPLES:end]
        if not np.isfinite(win).all():
            continue
        recent = lab[end - RECENT:end]
        X.append(extract_features(win, use_gyro))
        ends.append(end)
        y.append(int(recent[0]) if recent[0] >= 0 and np.all(recent == recent[0]) else -1)
        s.append(int(np.bincount(sev[end - RECENT:end]).argmax()))
    return np.array(X), np.array(ends), np.array(y), np.array(s)


def fogstar_set(folder, use_gyro):
    X, y, groups = [], [], []
    for st in fogstar_streams(folder):
        Xi, _, yi, _ = stream_windows(st, use_gyro)
        keep = yi >= 0
        X += list(Xi[keep])
        y += list(yi[keep])
        groups += [st["subject"]] * int(keep.sum())
    print(f"  FoG-STAR: {len(y)} windows from {len(set(groups))} subjects")
    return X, y, groups


def report(name, model, X, y):
    pred = model.predict(X)
    labels = sorted(set(y) | set(pred))
    print(f"\n{name}: accuracy {np.mean(pred == np.array(y)):.3f} on {len(y)} windows")
    print(classification_report(y, pred, labels=labels,
                                target_names=[STATUSES[i] for i in labels], digits=3, zero_division=0))
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y, pred, labels=labels))


def split_by_group(X, y, groups, test_frac=0.25):
    ids = sorted(set(groups))
    n_test = max(1, int(round(len(ids) * test_frac))) if len(ids) > 1 else 0
    test_ids = set(ids[-n_test:]) if n_test else set()
    tr = [i for i, g in enumerate(groups) if g not in test_ids]
    te = [i for i, g in enumerate(groups) if g in test_ids]
    pick = lambda idx: ([X[i] for i in idx], [y[i] for i in idx])
    return pick(tr), pick(te)


def train(recordings=None, daphnet=None, synthetic=True, out=MODEL_FILE, quiet=False, mhealth=None,
          fogstar=None):
    use_gyro = daphnet is None
    Xtr, ytr, tests, sources = [], [], [], []

    if synthetic:
        Xs, ys = synthetic_set(range(0, 24), use_gyro)
        Xt, yt = synthetic_set(range(100, 108), use_gyro)
        Xtr += Xs
        ytr += ys
        tests.append(("Synthetic hold-out (unseen simulated wearers)", Xt, yt))
        sources.append("synthetic")
    if recordings:
        print(f"Loading recordings from {recordings}")
        a, b, c, d = recordings_set(recordings, use_gyro)
        if a:
            # real data is scarce and matters most: weight it up
            Xtr += a * 3
            ytr += b * 3
            if c:
                tests.append(("Real recordings hold-out (last quarter of each file)", c, d))
            sources.append("recordings")
    if daphnet:
        print(f"Loading Daphnet from {daphnet}")
        Xd, yd, gd = daphnet_set(daphnet)
        if Xd:
            (a, b), (c, d) = split_by_group(Xd, yd, gd)
            Xtr += a
            ytr += b
            if c:
                tests.append(("Daphnet hold-out (unseen files)", c, d))
            sources.append("daphnet")

    if mhealth:
        print(f"Loading MHEALTH from {mhealth}")
        Xm, ym, gm = mhealth_set(mhealth, use_gyro)
        if Xm:
            (a, b), (c, d) = split_by_group(Xm, ym, gm)
            Xtr += a
            ytr += b
            if c:
                tests.append(("MHEALTH hold-out (unseen subjects, no freezes)", c, d))
            sources.append("mhealth")

    if fogstar:
        print(f"Loading FoG-STAR from {fogstar}")
        Xf, yf, gf = fogstar_set(fogstar, use_gyro)
        if Xf:
            (a, b), (c, d) = split_by_group(Xf, yf, gf)
            Xtr += a
            ytr += b
            if c:
                tests.append(("FoG-STAR hold-out (unseen people with PD)", c, d))
            sources.append("fogstar")

    if not Xtr:
        raise SystemExit("No training data found.")

    model = RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=3,
                                   class_weight="balanced", random_state=0, n_jobs=1)
    model.fit(np.array(Xtr), np.array(ytr))

    if not quiet:
        counts = {STATUSES[i]: int(np.sum(np.array(ytr) == i)) for i in range(3)}
        print(f"\nTrained on {len(ytr)} windows {counts} from: {', '.join(sources)}")
        for name, Xt, yt in tests:
            report(name, model, np.array(Xt), yt)
        names = feature_names(use_gyro)
        top = np.argsort(model.feature_importances_)[::-1][:6]
        print("\ntop features:", ", ".join(f"{names[i]} {model.feature_importances_[i]:.2f}" for i in top))
        if sources == ["synthetic"]:
            print("\nNOTE: trained on simulated signals only. These scores show the pipeline"
                  "\nworks, not how it will do on a person. Record real data with"
                  "\nrecord_data.py and retrain with --recordings once the ankle node runs.")

    joblib.dump({"model": model, "use_gyro": use_gyro, "classes": list(STATUSES),
                 "feature_names": feature_names(use_gyro), "sources": sources,
                 "window": WINDOW_SAMPLES, "rate": SAMPLE_RATE_HZ}, out)
    if not quiet:
        print(f"\nSaved {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings")
    ap.add_argument("--daphnet")
    ap.add_argument("--mhealth")
    ap.add_argument("--fogstar")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--out", default=MODEL_FILE)
    a = ap.parse_args()
    train(a.recordings, a.daphnet, not a.no_synthetic, a.out, mhealth=a.mhealth, fogstar=a.fogstar)
