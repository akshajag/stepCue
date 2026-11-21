"""StepCue shared constants.

Everything in the LOCKED section mirrors the shared architecture contract.
Do not change those values; every teammate's code depends on them.
"""
import os

# ----------------------------------------------------------------- LOCKED --
WIFI_SSID = "StepCue_AP"
WIFI_PASS = "stepcue123"
ESP32_IP = "192.168.4.1"
LAPTOP_IP = "192.168.4.2"

SENSOR_PORT = 5005          # ESP32 -> laptop, CSV "ts_ms,ax,ay,az,gx,gy,gz\n"
CUE_PORT = 5006             # laptop -> ESP32, "CUE:<bpm>:<haptic>" | "STOP"
SAMPLE_RATE_HZ = 50

STATE_FILE = os.environ.get("STEPCUE_STATE", "state.json")
STATE_TMP = STATE_FILE + ".tmp"

STATUSES = ("WALKING", "STATIONARY", "FREEZE")

# ------------------------------------------------------- project defaults --
# Units on the wire (not fixed by the contract, fixed here so firmware,
# simulator and classifier agree): accel in g, gyro in deg/s.
WINDOW_SEC = 1.2
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE_HZ)   # 60
HOP_SAMPLES = 5                                     # classify every 100 ms
DEBOUNCE_WINDOWS = 3

DEFAULT_BPM = 100
BPM_MIN, BPM_MAX = 60, 130
ALERT_LEVELS = ("normal", "watch", "alert")
PROLONGED_FREEZE_SEC = 4.0

MODEL_FILE = "gait_model.joblib"
# The model must be at least this confident before a window counts as FREEZE.
# Cross-subject benchmark on FoG-STAR (benchmark_datasets.py): 0.45 -> 78% of
# episodes cued, 305 false cues/h; 0.65 -> 79%, 204/h; 0.75 -> 72%, 128/h.
FREEZE_THRESHOLD = 0.65

# Simulator-only control channel (never used by real hardware).
SIM_IP = "127.0.0.1"
SIM_CONTROL_PORT = 5007     # "MODE:WALK" | "MODE:STAND" | "MODE:FREEZE"

DEFAULT_STATE = {
    "status": "STATIONARY",
    "freeze_index": 0.0,
    "bpm": DEFAULT_BPM,
    "haptic": 0,
    "alert_level": "normal",
    "caretaker_brief": "Mobility steady. No recent freeze episodes detected.",
    "episodes": [],
}
