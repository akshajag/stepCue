"""Software stand-in for the StepCue ESP32 waist hub.

Speaks the locked contract exactly, so the rest of the stack cannot tell it
from hardware:
  * sends  "ts_ms,ax,ay,az,gx,gy,gz\\n"  to <dest>:5005 at 50 Hz
  * listens on :5006 for  "CUE:<bpm>:<haptic>"  and  "STOP"

Extras that exist only in the simulator:
  * keyboard: w = walk, s = stand, f = freeze, h = hard freeze, t = turn left,
    r = turn right, q = quit
  * UDP control on 127.0.0.1:5007: MODE:WALK | MODE:STAND | MODE:FREEZE | MODE:FREEZE_HARD
    | MODE:TURN (left) | MODE:TURN_RIGHT   (a turn is walking while rotating 90 deg/s)
  * a simulated wearer who resumes walking a short while after cueing starts
    (a "hard" freeze only resolves once wrist haptics are switched on)

Usage:
  python sim_esp32.py                       # interactive
  python sim_esp32.py --script walk:8,freeze:30,walk:8 --loop
  python sim_esp32.py --script walk:6,turn:2,freeze:8,walk:6 --loop    # freeze in a turn
"""
import argparse
import socket
import sys
import threading
import time

from config import (CUE_PORT, DEFAULT_BPM, SAMPLE_RATE_HZ, SENSOR_PORT,
                    SIM_CONTROL_PORT, SIM_IP)
from gait_synth import GaitSynth

CUE_FAILSAFE_SEC = 5.0      # same failsafe as the firmware
MODE_ALIASES = {"WALK": "WALKING", "STAND": "STATIONARY", "FREEZE": "FREEZE",
                "FREEZE_HARD": "FREEZE", "WALKING": "WALKING", "STATIONARY": "STATIONARY",
                "TURN": "WALKING", "TURN_LEFT": "WALKING", "TURN_RIGHT": "WALKING"}
TURN_DPS = 90.0


class SimHub:
    def __init__(self, dest, cadence, auto_recover, seed):
        self.dest = (dest, SENSOR_PORT)
        self.synth = GaitSynth(seed=seed, cadence=cadence, tilt_deg=10)
        self.cadence = cadence
        self.auto_recover = auto_recover
        self.lock = threading.Lock()
        self.running = True
        self.hard = False
        self.freeze_started = None
        self.recover_at = None
        self.cue_active = False
        self.cue_bpm = DEFAULT_BPM
        self.cue_haptic = 0
        self.cue_last_rx = 0.0
        self.t0 = time.perf_counter()

    def log(self, msg):
        print(f"[sim {time.perf_counter() - self.t0:7.2f}s] {msg}", flush=True)

    # ------------------------------------------------------------ modes --
    def set_mode(self, name):
        name = name.strip().upper()
        if name not in MODE_ALIASES:
            self.log(f"unknown mode {name!r}")
            return
        with self.lock:
            mode = MODE_ALIASES[name]
            self.hard = name == "FREEZE_HARD"
            self.synth.set_mode(mode, cadence=self.cadence if mode == "WALKING" else None)
            self.synth.yaw_dps = {"TURN": TURN_DPS, "TURN_LEFT": TURN_DPS,
                                  "TURN_RIGHT": -TURN_DPS}.get(name, 0.0)
            self.recover_at = None
            self.freeze_started = time.perf_counter() if mode == "FREEZE" else None
            if mode == "FREEZE" and self.cue_active:
                self._schedule_recovery()
        self.log(f"wearer -> {name}")

    def _schedule_recovery(self):
        """Simulated response to cueing. Purely illustrative, not a clinical model."""
        if not self.auto_recover or self.synth.mode != "FREEZE":
            return
        if self.hard and not self.cue_haptic:
            self.recover_at = None          # needs wrist haptics to break through
            return
        preferred = 0.92 * self.cadence     # slightly slower than usual pace helps most
        delay = 2.4 + 0.035 * abs(self.cue_bpm - preferred) - (0.9 if self.cue_haptic else 0.0)
        delay = max(0.8, delay + float(self.synth.rng.normal(0, 0.2)))
        at = time.perf_counter() + delay
        if self.recover_at is None or at < self.recover_at:
            self.recover_at = at

    # ------------------------------------------------------------- cues --
    def handle_command(self, text):
        text = text.strip()
        now = time.perf_counter()
        with self.lock:
            if text == "STOP":
                if self.cue_active:
                    self.log("STOP  (LED + wrist motor off)")
                self.cue_active = False
                return
            parts = text.split(":")
            if len(parts) == 3 and parts[0] == "CUE":
                try:
                    bpm, haptic = int(parts[1]), int(parts[2])
                except ValueError:
                    return
                bpm = max(30, min(200, bpm))
                haptic = 1 if haptic else 0
                changed = (not self.cue_active) or bpm != self.cue_bpm or haptic != self.cue_haptic
                first = not self.cue_active
                self.cue_active, self.cue_bpm, self.cue_haptic = True, bpm, haptic
                self.cue_last_rx = now
                if changed:
                    lag = ""
                    if first and self.freeze_started:
                        lag = f"  [{(now - self.freeze_started) * 1000:.0f} ms after freeze onset]"
                    self.log(f"CUE {bpm} bpm, LED on, wrist haptic {'ON' if haptic else 'off'}{lag}")
                    self._schedule_recovery()

    def cue_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", CUE_PORT))
        sock.settimeout(0.5)
        while self.running:
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                continue
            self.handle_command(data.decode("ascii", "ignore"))

    def control_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SIM_IP, SIM_CONTROL_PORT))
        sock.settimeout(0.5)
        while self.running:
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                continue
            text = data.decode("ascii", "ignore").strip()
            if text.startswith("MODE:"):
                self.set_mode(text[5:])

    def keyboard(self):
        keys = {"w": "WALK", "s": "STAND", "f": "FREEZE", "h": "FREEZE_HARD",
                "t": "TURN", "r": "TURN_RIGHT"}
        try:
            import msvcrt
        except ImportError:
            for line in sys.stdin:                     # non-Windows: type letter + Enter
                c = line.strip().lower()[:1]
                if c == "q":
                    self.running = False
                elif c in keys:
                    self.set_mode(keys[c])
            return
        if not sys.stdin or not sys.stdin.isatty():
            return
        while self.running:
            if msvcrt.kbhit():
                c = msvcrt.getwch().lower()
                if c == "q":
                    self.running = False
                elif c in keys:
                    self.set_mode(keys[c])
            time.sleep(0.03)

    def script(self, steps, loop):
        while self.running:
            for name, secs in steps:
                self.set_mode(name)
                end = time.perf_counter() + secs
                while self.running and time.perf_counter() < end:
                    time.sleep(0.05)
            if not loop:
                break
        self.running = False

    # ------------------------------------------------------ sensor loop --
    def stream(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        period = 1.0 / SAMPLE_RATE_HZ
        next_t = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            with self.lock:
                if self.cue_active and now - self.cue_last_rx > CUE_FAILSAFE_SEC:
                    self.cue_active = False
                    self.log("cue failsafe: no CUE refresh for 5 s, stopping")
                if self.recover_at and now >= self.recover_at and self.synth.mode == "FREEZE":
                    self.synth.set_mode("WALKING", cadence=self.cadence)
                    self.recover_at = None
                    took = now - (self.freeze_started or now)
                    self.freeze_started = None
                    self.hard = False
                    recovered = took
                else:
                    recovered = None
                s = self.synth.sample()
            if recovered is not None:
                self.log(f"wearer resumes walking ({recovered:.1f} s frozen)")
            ts_ms = int((now - self.t0) * 1000)
            pkt = "%d,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n" % (ts_ms, *s)
            try:
                sock.sendto(pkt.encode("ascii"), self.dest)
            except OSError:
                pass
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_t = time.perf_counter()        # fell behind; resync


def parse_script(text):
    steps = []
    for part in text.split(","):
        name, secs = part.split(":")
        steps.append((name.strip().upper(), float(secs)))
    return steps


def main():
    ap = argparse.ArgumentParser(description="StepCue ESP32 simulator")
    ap.add_argument("--dest", default="127.0.0.1", help="laptop IP running gait_classifier.py")
    ap.add_argument("--cadence", type=float, default=100.0, help="walking cadence, steps/min")
    ap.add_argument("--start", default="WALK", help="initial mode")
    ap.add_argument("--script", help="e.g. walk:8,freeze:30,walk:8")
    ap.add_argument("--loop", action="store_true", help="repeat --script forever")
    ap.add_argument("--no-auto-recover", action="store_true",
                    help="wearer stays frozen until you change mode yourself")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    hub = SimHub(args.dest, args.cadence, not args.no_auto_recover, args.seed)
    hub.log(f"streaming 50 Hz IMU to {args.dest}:{SENSOR_PORT}, listening for cues on :{CUE_PORT}")
    hub.log("keys: w walk | s stand | f freeze | h hard freeze | t turn left | r turn right | q quit")
    threads = [threading.Thread(target=hub.cue_listener, daemon=True),
               threading.Thread(target=hub.control_listener, daemon=True)]
    if args.script:
        threads.append(threading.Thread(target=hub.script,
                                        args=(parse_script(args.script), args.loop), daemon=True))
    else:
        hub.set_mode(args.start)
        threads.append(threading.Thread(target=hub.keyboard, daemon=True))
    for t in threads:
        t.start()
    try:
        hub.stream()
    except KeyboardInterrupt:
        pass
    hub.running = False
    hub.log("stopped")


if __name__ == "__main__":
    main()
