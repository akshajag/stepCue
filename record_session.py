"""Record a labelled StepCue session for building your own dataset.

Stop gait_classifier.py first: only one program can listen on UDP 5005.

Two ways to label:

  Guided protocol (recommended, consistent between people):
    python record_session.py --participant P01 --protocol
  The script tells the wearer what to do ("Walk", "Tremble in place"...) with a
  countdown, and labels each segment automatically.

  Free labelling (someone watches and presses keys while the wearer moves):
    python record_session.py --participant P01
      w = WALKING   s = STATIONARY   f = FREEZE   u = UNLABELED (turns, transitions)
      m = mark an event   c = clap/sync marker (for lining up video)   q = stop

Each session is saved to recordings/<participant>_<time>/ :
  imu.csv       ts_ms,ax,ay,az,gx,gy,gz,label   (label per sample, same format
                train_classifier.py --recordings reads)
  events.jsonl  segment changes, freeze style, markers, with wall-clock time
  meta.json     participant, sensor placement, condition, sample-rate check, label totals

IMPORTANT: healthy volunteers can only *act out* freezes. That data is useful
for getting the pipeline and sensor placement right, but it is not the same as
real freezing of gait. Recording people with Parkinson's needs informed consent
and ethics approval. See RESEARCH.md.
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

from config import SAMPLE_RATE_HZ, SENSOR_PORT, STATUSES

KEYS = {"w": "WALKING", "s": "STATIONARY", "f": "FREEZE", "u": "UNLABELED"}

# (label, seconds, instruction, freeze style). Acted freezes follow the three
# manifestations used in recent FoG datasets: trembling, shuffling, akinesia.
PROTOCOL = [
    ("STATIONARY", 10, "Stand still, arms relaxed. Clap once at the start if filming.", None),
    ("WALKING", 20, "Walk at your normal pace in a straight line.", None),
    ("FREEZE", 8, "Freeze mid-step: keep feet in place and let the legs tremble quickly.", "trembling"),
    ("WALKING", 15, "Start walking again, normal pace.", None),
    ("UNLABELED", 5, "Turn around (this part is not used for training).", None),
    ("WALKING", 15, "Walk back.", None),
    ("FREEZE", 8, "Freeze with tiny shuffling steps that barely move forward.", "shuffling"),
    ("WALKING", 15, "Walk again.", None),
    ("STATIONARY", 10, "Stop and stand still.", None),
    ("WALKING", 10, "Walk slowly.", None),
    ("FREEZE", 6, "Freeze while turning: feet stuck, small trembling.", "trembling"),
    ("WALKING", 15, "Walk normally.", None),
    ("FREEZE", 6, "Akinetic freeze: stop completely, feet stuck, no movement.", "akinesia"),
    ("WALKING", 10, "Walk to finish.", None),
    ("STATIONARY", 8, "Stand still. Done.", None),
]


class Session:
    def __init__(self, args):
        self.args = args
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(args.out, f"{args.participant}_{stamp}")
        os.makedirs(self.dir, exist_ok=True)
        self.imu = open(os.path.join(self.dir, "imu.csv"), "w", encoding="utf-8")
        self.imu.write("ts_ms,ax,ay,az,gx,gy,gz,label\n")
        self.events = open(os.path.join(self.dir, "events.jsonl"), "w", encoding="utf-8")
        self.lock = threading.Lock()
        self.label = "UNLABELED"
        self.running = True
        self.n = 0
        self.counts = {}
        self.first_ts = self.last_ts = None
        self.started = time.time()
        self.last_ts_ms = None

    def event(self, kind, **extra):
        rec = {"wall_time": round(time.time(), 3), "sensor_ts_ms": self.last_ts_ms, "event": kind, **extra}
        with self.lock:
            self.events.write(json.dumps(rec) + "\n")
            self.events.flush()

    def set_label(self, label, style=None, instruction=None):
        with self.lock:
            self.label = label
        self.event("label", label=label, freeze_style=style, instruction=instruction)

    # ------------------------------------------------------------ inputs --
    def keyboard(self):
        try:
            import msvcrt
        except ImportError:
            msvcrt = None
        while self.running:
            if msvcrt:
                if not msvcrt.kbhit():
                    time.sleep(0.02)
                    continue
                c = msvcrt.getwch().lower()
            else:
                line = sys.stdin.readline()
                if not line:
                    return
                c = line.strip().lower()[:1]
            if c == "q":
                self.running = False
            elif c in KEYS:
                self.set_label(KEYS[c])
                print(f"\n  -> {KEYS[c]}", flush=True)
            elif c == "m":
                self.event("marker")
                print("\n  marker", flush=True)
            elif c == "c":
                self.event("sync_clap")
                print("\n  sync clap marked", flush=True)

    def protocol(self):
        print("\nGuided protocol. Read each instruction aloud to the wearer.\n", flush=True)
        for i in range(3, 0, -1):
            print(f"  starting in {i}...", flush=True)
            time.sleep(1)
        for label, secs, text, style in PROTOCOL:
            if not self.running:
                return
            self.set_label(label, style, text)
            print(f"\n[{label:<10}] {secs:>3} s  {text}", flush=True)
            end = time.time() + secs
            while self.running and time.time() < end:
                time.sleep(0.05)
        self.running = False

    # ------------------------------------------------------------ record --
    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", SENSOR_PORT))
        sock.settimeout(0.5)
        t = threading.Thread(target=self.protocol if self.args.protocol else self.keyboard, daemon=True)
        t.start()
        if self.args.protocol:
            threading.Thread(target=self.keyboard, daemon=True).start()   # still allows q / c / m
        last_status = 0.0
        try:
            while self.running:
                try:
                    data, _ = sock.recvfrom(512)
                except socket.timeout:
                    print("\r  waiting for sensor packets...          ", end="", flush=True)
                    continue
                for line in data.decode("ascii", "ignore").splitlines():
                    parts = line.strip().split(",")
                    if len(parts) != 7:
                        continue
                    with self.lock:
                        label = self.label
                        self.imu.write(f"{line.strip()},{label}\n")
                    self.n += 1
                    self.counts[label] = self.counts.get(label, 0) + 1
                    try:
                        ts = int(parts[0])
                    except ValueError:
                        continue
                    self.last_ts_ms = ts
                    self.first_ts = ts if self.first_ts is None else self.first_ts
                    self.last_ts = ts
                now = time.time()
                if now - last_status > 0.5:
                    last_status = now
                    rate = self.n / max(now - self.started, 1e-6)
                    print(f"\r  {self.label:<10} {self.n:>7} samples  {rate:5.1f} Hz   ", end="", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.finish()

    def finish(self):
        self.imu.close()
        self.events.close()
        span = ((self.last_ts - self.first_ts) / 1000.0) if self.first_ts is not None else 0.0
        meta = {
            "participant": self.args.participant,
            "condition": self.args.condition,
            "placement": self.args.placement,
            "notes": self.args.notes,
            "protocol": "guided" if self.args.protocol else "free",
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
            "samples": self.n,
            "sensor_span_sec": round(span, 1),
            "measured_rate_hz": round(self.n / span, 1) if span else None,
            "expected_rate_hz": SAMPLE_RATE_HZ,
            "seconds_per_label": {k: round(v / SAMPLE_RATE_HZ, 1) for k, v in sorted(self.counts.items())},
            "units": {"accel": "g", "gyro": "deg/s"},
            "labels": list(STATUSES) + ["UNLABELED"],
        }
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"\n\nSaved {self.n} samples to {self.dir}")
        print(f"Time per label: {meta['seconds_per_label']}")
        if meta["measured_rate_hz"] and abs(meta["measured_rate_hz"] - SAMPLE_RATE_HZ) > 5:
            print(f"WARNING: measured {meta['measured_rate_hz']} Hz, expected {SAMPLE_RATE_HZ} Hz "
                  "(packet loss or timing problem).")
        print("Retrain with:  python train_classifier.py --recordings recordings")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="anonymous ID like P01 (never a real name)")
    ap.add_argument("--protocol", action="store_true", help="guided, auto-labelled session")
    ap.add_argument("--condition", default="healthy-simulated",
                    choices=["healthy-simulated", "parkinsons-consented"],
                    help="who is wearing it; real patient data needs consent and ethics approval")
    ap.add_argument("--placement", default="right ankle, lateral, above malleolus",
                    help="where and how the sensor is mounted")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default="recordings")
    Session(ap.parse_args()).run()


if __name__ == "__main__":
    main()
