"""Record labelled IMU data from the ankle node (or simulator) for training.

Stop gait_classifier.py first: only one program can listen on UDP 5005.

  python record_data.py --label WALKING --seconds 60
  python record_data.py --label STATIONARY --seconds 30
  python record_data.py --label FREEZE --seconds 20     # acted trembling / shuffling in place

Each run writes recordings/<LABEL>_<time>.csv with rows
  ts_ms,ax,ay,az,gx,gy,gz,label
Then retrain:  python train_classifier.py --recordings recordings
"""
import argparse
import os
import socket
import time

from config import SENSOR_PORT, STATUSES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, choices=STATUSES)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--countdown", type=int, default=3)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.label}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", SENSOR_PORT))
    sock.settimeout(1.0)

    for i in range(a.countdown, 0, -1):
        print(f"Recording {a.label} in {i}...", flush=True)
        time.sleep(1)
    while True:                                   # drop packets queued during the countdown
        sock.settimeout(0.0)
        try:
            sock.recvfrom(256)
        except (BlockingIOError, OSError):
            break
    sock.settimeout(1.0)

    print(f"Recording {a.seconds:.0f} s of {a.label} -> {path}", flush=True)
    n, end = 0, time.monotonic() + a.seconds
    with open(path, "w", encoding="utf-8") as f:
        f.write("ts_ms,ax,ay,az,gx,gy,gz,label\n")
        while time.monotonic() < end:
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                print("  no packets... is the sensor streaming?", flush=True)
                continue
            for line in data.decode("ascii", "ignore").splitlines():
                if line.count(",") == 6:
                    f.write(f"{line.strip()},{a.label}\n")
                    n += 1
    print(f"Done: {n} samples ({n / max(a.seconds, 1e-9):.1f} Hz average)")


if __name__ == "__main__":
    main()
