"""Per-patient training: label sitting / standing / walking / frozen, build a model for that person.

Freezing looks different for every person, so StepCue can learn each wearer. A helper (or the
wearer) taps what is happening, on the phone's Training screen or on the dashboard. While a label
is on, every phone reading (1 a second during training) is saved together with the ankle sensor's
features from the ESP32 (Bluetooth or Wi-Fi) at that moment. "Build the model" trains a Random
Forest on that person's own data and reports honest accuracy (cross-validated by recording
segment, so overlapping windows from one segment never score themselves).

Files (all under training/, gitignored: this is personal movement data and stays on the laptop):
  profiles.json                   patients + which one is active
  session.json                    the label being recorded right now (written atomically)
  <patient>/samples.jsonl         one labelled sample per line
  <patient>/model.joblib          the trained models + metrics

Usage:
  python personal_model.py --list
  python personal_model.py --train <patient-id>
"""
import argparse
import json
import os
import re
import threading
import time
from collections import Counter, deque

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "training")
LABELS = ("sitting", "standing", "walking", "frozen")
LABEL_WORD = {"sitting": "Sitting", "standing": "Standing", "walking": "Walking", "frozen": "Frozen"}
SETTLE_SEC = 3.0          # skip the first seconds after a label change (the phone window still holds the old activity)
ANKLE_FRESH_SEC = 3.0
MIN_PER_CLASS = 10        # samples a label needs before it is used for training
GOAL_SEC = 60             # suggested recording per label
SESSION_TIMEOUT_SEC = 900  # a label left on for 15 min stops on its own

PHONE_FEATURES = ("acc_rms", "gyro_rms", "vert_rms", "yaw_abs", "turn_deg", "fi_log", "cadence", "steps_10s",
                  "pitch", "roll", "in_pocket")
ANKLE_FEATURES = ("a_loco", "a_frz", "a_total", "a_fi", "a_dom", "a_mag_mean", "a_mag_std", "a_mag_ptp", "a_jerk",
                  "g_loco", "g_frz", "g_total", "g_fi", "g_dom", "g_mag_mean", "g_mag_std", "g_mag_max",
                  "a_mean_x", "a_mean_y", "a_mean_z")

_lock = threading.Lock()


# ---------------------------------------------------------------- files --
def _path(*parts):
    return os.path.join(TRAIN_DIR, *parts)


def _read_json(path, default):
    for _ in range(10):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(0.01)
    return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    os.replace(tmp, path)


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")[:40]
    return s or "patient"


# ------------------------------------------------------------- profiles --
def profiles():
    p = _read_json(_path("profiles.json"), {})
    p.setdefault("patients", {})
    p.setdefault("active", None)
    if p["active"] not in p["patients"]:
        p["active"] = next(iter(p["patients"]), None)
    return p


def add_patient(name, make_active=True):
    name = str(name).strip()[:60]
    if not name:
        raise ValueError("name needed")
    with _lock:
        p = profiles()
        pid = slug(name)
        base, i = pid, 2
        while pid in p["patients"] and p["patients"][pid]["name"] != name:
            pid, i = f"{base}-{i}", i + 1
        p["patients"].setdefault(pid, {"name": name, "created": time.time()})
        if make_active or not p["active"]:
            p["active"] = pid
        _write_json(_path("profiles.json"), p)
    return pid


def set_active(pid):
    with _lock:
        p = profiles()
        if pid not in p["patients"]:
            raise ValueError("unknown patient")
        p["active"] = pid
        _write_json(_path("profiles.json"), p)
    stop_label()


def active():
    p = profiles()
    pid = p["active"]
    return (pid, p["patients"][pid]["name"]) if pid else (None, None)


def delete_data(pid):
    """Remove a patient's samples and model (the profile stays)."""
    for f in ("samples.jsonl", "model.joblib"):
        try:
            os.remove(_path(pid, f))
        except FileNotFoundError:
            pass
    _MODEL_CACHE.pop(pid, None)


# -------------------------------------------------------------- session --
def session(now=None):
    now = time.time() if now is None else now
    s = _read_json(_path("session.json"), {})
    if s.get("label") and now - s.get("since", now) > SESSION_TIMEOUT_SEC:
        s["label"] = None
    return {"label": s.get("label"), "since": s.get("since"), "segment": s.get("segment", 0),
            "patient": s.get("patient")}


def set_label(label, now=None):
    if label not in LABELS:
        raise ValueError("unknown label")
    pid, _ = active()
    if not pid:
        raise ValueError("add a patient first")
    now = time.time() if now is None else now
    with _lock:
        s = _read_json(_path("session.json"), {})
        seg = int(s.get("segment", 0)) + 1
        _write_json(_path("session.json"), {"label": label, "since": now, "segment": seg, "patient": pid})
    return seg


def stop_label():
    with _lock:
        s = _read_json(_path("session.json"), {})
        if s.get("label"):
            s["label"] = None
            _write_json(_path("session.json"), s)


# ------------------------------------------------------------- features --
def phone_vector(rec):
    """Phone features in PHONE_FEATURES order from one /api/android reading."""
    g = lambda k: float(rec.get(k) or 0.0)  # noqa: E731
    return [g("acc_rms"), g("gyro_rms"), g("vert_rms"), abs(g("yaw_rate")), g("turn_deg"),
            float(np.log1p(min(g("freeze_index"), 50.0))), g("cadence"), g("steps_10s"),
            g("pitch"), g("roll"), 1.0 if rec.get("in_pocket") else 0.0]


def ankle_from_state(state, now=None):
    """Ankle feature vector written by gait_classifier.py into state.json, if fresh."""
    now = time.time() if now is None else now
    a = (state or {}).get("ankle") or {}
    vals = a.get("features")
    if not vals or len(vals) != len(ANKLE_FEATURES) or now - float(a.get("t", 0)) > ANKLE_FRESH_SEC:
        return None
    return [float(v) for v in vals]


def add_sample(pid, label, segment, rec, ankle, now=None):
    now = time.time() if now is None else now
    row = {"t": round(now, 2), "label": label, "segment": segment, "phone": phone_vector(rec),
           "ankle": ankle}
    os.makedirs(_path(pid), exist_ok=True)
    with _lock, open(_path(pid, "samples.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def record(rec, state, now=None):
    """Called for every phone reading. Saves it if a label is on and settled. Returns True if saved."""
    now = time.time() if now is None else now
    s = session(now)
    if not s["label"] or not s["patient"] or now - (s["since"] or now) < SETTLE_SEC:
        return False
    add_sample(s["patient"], s["label"], s["segment"], rec, ankle_from_state(state, now), now)
    return True


def load_samples(pid):
    out = []
    try:
        with open(_path(pid, "samples.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("label") in LABELS and isinstance(r.get("phone"), list) \
                        and len(r["phone"]) == len(PHONE_FEATURES):
                    out.append(r)
    except FileNotFoundError:
        pass
    return out


def dataset_summary(pid):
    rows = load_samples(pid) if pid else []
    counts = Counter(r["label"] for r in rows)
    segs = {lab: len({r["segment"] for r in rows if r["label"] == lab}) for lab in LABELS}
    ankle = sum(1 for r in rows if r.get("ankle"))
    return {"total": len(rows), "counts": {lab: counts.get(lab, 0) for lab in LABELS}, "segments": segs,
            "with_ankle": ankle, "goal": GOAL_SEC}


# --------------------------------------------------------------- training --
def _cv_predict(X, y, groups):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=0,
                                 n_jobs=1)
    n_groups = len(set(groups))
    min_class = min(Counter(y).values())
    if n_groups >= 3 and all(len({g for g, lab in zip(groups, y) if lab == c}) >= 2 for c in set(y)):
        cv, method = GroupKFold(n_splits=min(5, n_groups)), "by recording segment"
        pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
    else:
        k = max(2, min(5, min_class))
        cv, method = StratifiedKFold(n_splits=k, shuffle=True, random_state=0), "random split (optimistic: record each label twice for an honest score)"
        pred = cross_val_predict(clf, X, y, cv=cv)
    clf.fit(X, y)
    return clf, pred, method


def _metrics(y, pred, labels):
    y, pred = np.asarray(y), np.asarray(pred)
    cm = [[int(((y == a) & (pred == b)).sum()) for b in labels] for a in labels]
    recall = {lab: (float((pred[y == lab] == lab).mean()) if (y == lab).any() else None) for lab in labels}
    return {"accuracy": float((y == pred).mean()), "recall": recall, "confusion": cm, "labels": list(labels)}


def train(pid, now=None):
    """Train this patient's models. Returns the metrics dict (also saved with the model)."""
    import joblib
    now = time.time() if now is None else now
    rows = load_samples(pid)
    counts = Counter(r["label"] for r in rows)
    labels = [lab for lab in LABELS if counts.get(lab, 0) >= MIN_PER_CLASS]
    skipped = {lab: counts.get(lab, 0) for lab in LABELS if lab not in labels}
    if len(labels) < 2:
        raise ValueError(f"record at least two activities with {MIN_PER_CLASS}+ seconds each first")
    rows = [r for r in rows if r["label"] in labels]
    y = [r["label"] for r in rows]
    groups = [r["segment"] for r in rows]
    Xp = np.array([r["phone"] for r in rows], dtype=float)
    phone_clf, phone_pred, method = _cv_predict(Xp, y, groups)
    result = {"trained_at": now, "n": len(rows), "labels": labels, "skipped": skipped, "cv": method,
              "phone": _metrics(y, phone_pred, labels),
              "importance": sorted(zip(PHONE_FEATURES, phone_clf.feature_importances_.round(3).tolist()),
                                   key=lambda kv: -kv[1])[:8]}
    bundle = {"phone": phone_clf, "both": None, "labels": labels}

    both = [r for r in rows if r.get("ankle") and len(r["ankle"]) == len(ANKLE_FEATURES)]
    bc = Counter(r["label"] for r in both)
    if len(both) >= 0.6 * len(rows) and sum(1 for lab in labels if bc.get(lab, 0) >= MIN_PER_CLASS) >= 2:
        both = [r for r in both if bc.get(r["label"], 0) >= MIN_PER_CLASS]
        yb = [r["label"] for r in both]
        Xb = np.array([r["phone"] + r["ankle"] for r in both], dtype=float)
        both_clf, both_pred, bmethod = _cv_predict(Xb, yb, [r["segment"] for r in both])
        blabels = [lab for lab in labels if lab in set(yb)]
        result["both"] = dict(_metrics(yb, both_pred, blabels), cv=bmethod, n=len(both))
        result["importance"] = sorted(zip(PHONE_FEATURES + ANKLE_FEATURES,
                                          both_clf.feature_importances_.round(3).tolist()), key=lambda kv: -kv[1])[:8]
        bundle["both"] = both_clf
    else:
        result["both"] = None

    walk = [r["phone"][PHONE_FEATURES.index("cadence")] for r in rows if r["label"] == "walking"]
    walk = [c for c in walk if c > 0]
    if len(walk) >= 5:
        q = np.percentile(walk, [25, 50, 75])
        result["pace"] = {"low": int(q[0]), "median": int(q[1]), "high": int(q[2])}
    frz = [r["phone"][PHONE_FEATURES.index("fi_log")] for r in rows if r["label"] == "frozen"]
    if frz:
        result["frozen_fi"] = round(float(np.expm1(np.median(frz))), 2)
    bundle["metrics"] = result
    os.makedirs(_path(pid), exist_ok=True)
    tmp = _path(pid, "model.joblib.tmp")
    joblib.dump(bundle, tmp)
    os.replace(tmp, _path(pid, "model.joblib"))
    _MODEL_CACHE.pop(pid, None)
    return result


# -------------------------------------------------------------- predicting --
_MODEL_CACHE = {}
_RECENT = deque(maxlen=3)


def load_model(pid):
    path = _path(pid, "model.joblib") if pid else None
    if not path or not os.path.isfile(path):
        return None
    mtime = os.path.getmtime(path)
    hit = _MODEL_CACHE.get(pid)
    if hit and hit[0] == mtime:
        return hit[1]
    import joblib
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    _MODEL_CACHE[pid] = (mtime, bundle)
    return bundle


def model_info(pid):
    b = load_model(pid)
    return None if not b else b.get("metrics")


def predict(rec, state, pid=None, now=None, smooth=True):
    """The patient's own model on one reading: {"label", "conf", "source", "probs"} or None."""
    pid = pid or active()[0]
    b = load_model(pid)
    if not b:
        return None
    ankle = ankle_from_state(state, now)
    if ankle is not None and b.get("both") is not None:
        clf, x, src = b["both"], phone_vector(rec) + ankle, "phone + ankle"
    else:
        clf, x, src = b["phone"], phone_vector(rec), "phone"
    proba = clf.predict_proba(np.array([x], dtype=float))[0]
    probs = {str(c): round(float(p), 3) for c, p in zip(clf.classes_, proba)}
    label = max(probs, key=probs.get)
    if smooth:
        _RECENT.append(label)
        votes = Counter(_RECENT).most_common(1)[0]
        if votes[1] >= 2:
            label = votes[0]
    return {"label": label, "word": LABEL_WORD[label], "conf": probs.get(label, 0.0), "source": src, "probs": probs}


def phone_status(now=None):
    """Small block for /api/status so the phone's Training screen can draw itself."""
    p = profiles()
    pid = p["active"]
    s = session(now)
    m = model_info(pid) if pid else None
    return {"patient": None if not pid else {"id": pid, "name": p["patients"][pid]["name"]},
            "patients": [{"id": k, "name": v["name"]} for k, v in p["patients"].items()],
            "label": s["label"] if s["patient"] == pid else None, "since": s["since"],
            "counts": dataset_summary(pid)["counts"] if pid else {},
            "goal": GOAL_SEC,
            "model": None if not m else {"trained_at": m["trained_at"], "n": m["n"],
                                         "accuracy": round((m.get("both") or m["phone"])["accuracy"], 3),
                                         "labels": m["labels"], "pace": m.get("pace")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--train")
    a = ap.parse_args()
    if a.train:
        r = train(a.train)
        print(f"trained on {r['n']} samples ({r['cv']}): phone {r['phone']['accuracy']:.0%}"
              + (f", phone + ankle {r['both']['accuracy']:.0%}" if r.get("both") else ""))
    else:
        p = profiles()
        for pid, v in p["patients"].items():
            print(("* " if pid == p["active"] else "  ") + f"{pid}: {v['name']}  {dataset_summary(pid)['counts']}")


if __name__ == "__main__":
    main()
