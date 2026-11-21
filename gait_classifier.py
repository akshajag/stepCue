"""StepCue Tier 1: low-latency local gait loop.

  ESP32 (or sim_esp32.py) --UDP 5005, 50 Hz--> this script --UDP 5006--> ESP32

Every 100 ms the last 1.2 s of IMU data is classified WALKING / STATIONARY /
FREEZE by a Random Forest. Three agreeing windows in a row change the state.
The moment FREEZE is confirmed a CUE packet goes back to the ESP32; nothing in
this path touches the cloud or even the disk.

It also logs freeze episodes into state.json for the advisor and dashboard,
and picks up bpm / haptic changes that the Nemotron advisor writes there.
motion.py runs alongside on the same samples: steps, cadence and turns go into
an optional "motion" block, and each episode notes whether it began in a turn.

Run:  python gait_classifier.py
"""
import argparse
import os
import socket
import time
from collections import deque

import joblib
import numpy as np

from config import (CUE_PORT, DEBOUNCE_WINDOWS, DEFAULT_BPM,
                    FREEZE_THRESHOLD, HOP_SAMPLES, MODEL_FILE,
                    SAMPLE_RATE_HZ, SENSOR_PORT, STATUSES, WINDOW_SAMPLES)
from features import extract_features, freeze_index
from motion import MotionTracker
from state_io import read_state, update_state

HOP_SEC = HOP_SAMPLES / SAMPLE_RATE_HZ
STATE_WRITE_EVERY = 2          # hops -> state.json refreshed every 200 ms
CUE_REFRESH_SEC = 1.0          # re-send CUE while active (UDP can drop packets)
CUE_TAIL_SEC = 1.5             # keep the rhythm briefly after walking resumes
STOP_REPEATS = 3
STANDING_GIVE_UP_SEC = 8.0     # freeze -> standing still this long ends the episode
SIGNAL_LOSS_SEC = 2.0
# A window only flips class once about half of it shows the new pattern, so a
# confirmed change really happened roughly this long ago (half window + debounce).
DETECT_LAG_SEC = 0.8
MAX_EPISODES = 50
TURN_LINK_SEC = 1.5           # a freeze this close to a turn counts as "during a turn"


class GaitLoop:
    def __init__(self, model_path, verbose=True):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.use_gyro = bundle["use_gyro"]
        self.verbose = verbose
        self.buf = deque(maxlen=SAMPLE_RATE_HZ * 5)
        self.recent = deque(maxlen=DEBOUNCE_WINDOWS)
        self.status = "STATIONARY"
        self.status_since = time.monotonic()
        self.fi = 0.0
        self.cadence = None
        self.motion = MotionTracker()
        self.last_turn_end = -1e9      # tracker time (s) when the last turn finished
        self.esp_addr = None
        self.cue_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        st = read_state()
        self.bpm = int(st.get("bpm", DEFAULT_BPM))
        self.haptic = int(st.get("haptic", 0))
        self.next_id = max([e.get("id", 0) for e in st.get("episodes", [])], default=0) + 1

        self.episode = None            # open episode dict (the state.json fields)
        self.ep = {}                   # timing bookkeeping for the open episode
        self.cue_active = False
        self.cue_sent_at = 0.0
        self.cue_params = None
        self.stop_at = None
        self.stops_left = 0
        self.hops = 0
        self.samples_since_hop = 0
        self.last_packet = time.monotonic()
        self.dirty = True
        self.freeze_col = list(self.model.classes_).index(STATUSES.index("FREEZE"))
        # warm up the model so the first real prediction is not slow
        self.model.predict_proba(np.zeros((1, self.model.n_features_in_)))

    def classify(self, window):
        """FREEZE only when the model is at least FREEZE_THRESHOLD confident;
        otherwise the more likely of WALKING / STATIONARY."""
        proba = self.model.predict_proba(extract_features(window, self.use_gyro)[None, :])[0]
        if proba[self.freeze_col] >= FREEZE_THRESHOLD:
            return "FREEZE"
        proba[self.freeze_col] = -1.0
        return STATUSES[int(self.model.classes_[int(np.argmax(proba))])]

    def log(self, msg):
        if self.verbose:
            print(f"[gait {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # --------------------------------------------------------------- cues --
    def send(self, text):
        if self.esp_addr:
            try:
                self.cue_sock.sendto(text.encode("ascii"), (self.esp_addr, CUE_PORT))
            except OSError:
                pass

    def send_cue(self, now):
        self.send(f"CUE:{self.bpm}:{self.haptic}")
        self.cue_active = True
        self.cue_sent_at = now
        self.cue_params = (self.bpm, self.haptic)
        self.stop_at = None
        self.stops_left = 0

    def send_stop(self):
        self.send("STOP")
        self.cue_active = False
        self.stop_at = None
        self.stops_left = STOP_REPEATS - 1

    def service_cue(self, now):
        if self.stop_at and now >= self.stop_at:
            self.send_stop()
            self.log("cue stopped")
        elif self.cue_active and ((self.bpm, self.haptic) != self.cue_params
                                  or now - self.cue_sent_at >= CUE_REFRESH_SEC):
            if (self.bpm, self.haptic) != self.cue_params:
                self.log(f"cue updated -> {self.bpm} bpm, haptic {self.haptic}")
            self.send(f"CUE:{self.bpm}:{self.haptic}")
            self.cue_sent_at = now
            self.cue_params = (self.bpm, self.haptic)
        elif self.stops_left > 0:
            self.send("STOP")
            self.stops_left -= 1

    # ----------------------------------------------------------- episodes --
    def open_episode(self, now, onset):
        self.episode = {
            "id": self.next_id,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(time.time() - (now - onset))),
            "duration_sec": round(now - onset, 1),
            "pre_freeze_cadence": self.cadence,
            "cue_bpm": self.bpm,
            "cue_haptic": self.haptic,     # optional extra field: wrist vibration during this episode
            "recovery_sec": None,          # null while the episode is still open
            "analyzed": False,
            "during_turn": self.near_turn(),   # optional extra field (motion.py)
        }
        self.next_id += 1
        self.ep = {"onset": onset, "cue_start": now, "frozen": 0.0,
                   "freeze_since": onset, "standing_since": None}

    def near_turn(self):
        m = self.motion
        t = m.n * m.dt
        turning_now = m.turn is not None and abs(m.turn["angle"]) >= 15
        return bool(turning_now or t - self.last_turn_end <= TURN_LINK_SEC)

    def close_episode(self, now, reason):
        e, t = self.episode, self.ep
        if t["freeze_since"] is not None:
            t["frozen"] += now - t["freeze_since"]
        e["duration_sec"] = round(t["frozen"], 1)
        lag = DETECT_LAG_SEC if reason == "walking resumed" else 0.0
        e["recovery_sec"] = round(max(0.1, now - lag - t["cue_start"]), 1)
        self.log(f"episode #{e['id']} closed ({reason}): frozen {e['duration_sec']} s, "
                 f"{e['recovery_sec']} s from cue to end")
        self.flush_state(final_episode=e)
        self.episode, self.ep = None, {}

    def on_transition(self, old, new, now, t_packet):
        self.log(f"{old} -> {new}")
        if new == "FREEZE":
            if self.episode is None:
                onset = now - DETECT_LAG_SEC
                self.open_episode(now, onset)
                self.send_cue(now)                       # <- the latency-critical line
                lag_ms = (time.monotonic() - t_packet) * 1000
                self.log(f"FREEZE #{self.episode['id']}: CUE:{self.bpm}:{self.haptic} sent "
                         f"{lag_ms:.1f} ms after the confirming packet arrived")
            else:                                        # froze again before recovering
                self.ep["freeze_since"] = now - DETECT_LAG_SEC
                self.ep["standing_since"] = None
                self.send_cue(now)
        elif self.episode is not None:
            if self.ep["freeze_since"] is not None:
                self.ep["frozen"] += max(0.0, now - DETECT_LAG_SEC - self.ep["freeze_since"])
                self.ep["freeze_since"] = None
            if new == "WALKING":
                self.close_episode(now, "walking resumed")
                self.stop_at = now + CUE_TAIL_SEC
            else:
                self.ep["standing_since"] = now
        self.dirty = True

    # -------------------------------------------------------------- state --
    def flush_state(self, final_episode=None):
        ep = final_episode or self.episode
        status, fi = self.status, round(self.fi, 2)
        motion = self.motion.summary()
        motion["cadence_spm"] = motion["cadence_spm"] or self.cadence

        def mutate(st):
            st["status"] = status
            st["freeze_index"] = fi
            st["motion"] = motion                  # optional block, owned by this loop
            if ep is not None:
                eps = st.setdefault("episodes", [])
                for i, old in enumerate(eps):
                    if old.get("id") == ep["id"]:
                        merged = dict(ep)
                        merged["analyzed"] = old.get("analyzed", False)   # advisor owns this
                        eps[i] = merged
                        break
                else:
                    eps.append(dict(ep))
                del eps[:-MAX_EPISODES]

        st = update_state(mutate)
        # pick up cue parameters chosen by the advisor
        try:
            self.bpm = int(st.get("bpm", self.bpm))
            self.haptic = 1 if st.get("haptic") else 0
        except (TypeError, ValueError):
            pass
        self.dirty = False

    # ---------------------------------------------------------- main loop --
    def on_sample(self, sample, t_packet):
        for ev in self.motion.update(sample):
            if ev["type"] == "turn":
                self.last_turn_end = ev["t_end"]
                if self.episode is not None:       # the turn finished while frozen
                    self.episode["during_turn"] = True
                self.log(f"turn {ev['direction']} {abs(ev['angle_deg'])} deg in {ev['duration_sec']} s")
        self.buf.append(sample)
        self.samples_since_hop += 1
        if self.samples_since_hop < HOP_SAMPLES or len(self.buf) < WINDOW_SAMPLES:
            return
        self.samples_since_hop = 0
        self.hops += 1
        now = time.monotonic()

        window = np.array(list(self.buf)[-WINDOW_SAMPLES:])
        pred = self.classify(window)
        self.fi = 0.5 * self.fi + 0.5 * freeze_index(window)
        self.recent.append(pred)

        if (len(self.recent) == DEBOUNCE_WINDOWS and pred != self.status
                and all(p == pred for p in self.recent)):
            old, self.status, self.status_since = self.status, pred, now
            self.on_transition(old, pred, now, t_packet)

        # Cadence from counted strides. (The spectral estimate in features.py can
        # lock onto a harmonic and read double for slow walking.)
        if self.status == "WALKING":
            c = self.motion.cadence()
            if c and 40 <= c <= 180:
                self.cadence = c

        if self.episode is not None:
            t = self.ep
            frozen = t["frozen"] + (now - t["freeze_since"] if t["freeze_since"] is not None else 0)
            self.episode["duration_sec"] = round(frozen, 1)
            self.episode["cue_bpm"] = self.bpm
            self.episode["cue_haptic"] = self.haptic
            if t["standing_since"] and now - t["standing_since"] > STANDING_GIVE_UP_SEC:
                self.close_episode(now, "standing still")
                self.send_stop()

        self.service_cue(now)
        if self.dirty or self.hops % STATE_WRITE_EVERY == 0:
            self.flush_state()

    def on_signal_loss(self):
        self.log("no sensor packets for 2 s (sensor off or out of range)")
        if self.cue_active or self.stop_at:
            self.send_stop()
        if self.episode is not None:
            self.close_episode(time.monotonic(), "signal lost")
        self.buf.clear()
        self.recent.clear()
        self.motion.turn = None

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", SENSOR_PORT))
        sock.settimeout(0.25)
        self.log(f"listening on UDP {SENSOR_PORT}; cue {self.bpm} bpm, haptic {self.haptic}")
        self.flush_state()
        lost = True
        while True:
            try:
                data, addr = sock.recvfrom(512)
            except socket.timeout:
                if not lost and time.monotonic() - self.last_packet > SIGNAL_LOSS_SEC:
                    lost = True
                    self.on_signal_loss()
                continue
            except OSError:
                continue
            t_packet = time.monotonic()
            self.last_packet = t_packet
            if lost:
                lost = False
                self.log(f"sensor stream from {addr[0]}")
            self.esp_addr = addr[0]
            for line in data.decode("ascii", "ignore").splitlines():
                parts = line.split(",")
                if len(parts) != 7:
                    continue
                try:
                    sample = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                self.on_sample(sample, t_packet)


def main():
    ap = argparse.ArgumentParser(description="StepCue real-time gait classifier")
    ap.add_argument("--model", default=MODEL_FILE)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.model):
        print(f"{a.model} not found, training a bootstrap model on simulated data...", flush=True)
        from train_classifier import train
        train(out=a.model, quiet=True)
    loop = GaitLoop(a.model, verbose=not a.quiet)
    try:
        loop.run()
    except KeyboardInterrupt:
        loop.send_stop()
        print("stopped")


if __name__ == "__main__":
    main()
