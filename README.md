# StepCue

Assistive freezing-of-gait wearable and caregiver telemetry prototype. An ankle
IMU detects a mid-stride freeze, the waist hub immediately plays a rhythmic
light + wrist-vibration cue, and an NVIDIA Nemotron agent tunes the cue over
time and keeps a caregiver informed.

StepCue is about assistive cueing, pattern tracking and caregiver awareness.
It does not diagnose and gives no treatment advice.

## Run it (no hardware needed)

```powershell
.venv\Scripts\python run_demo.py
```

This opens seven windows: the ESP32 simulator, the gait classifier, the Nemotron
advisor, the caregiver alert notifier, the location server (simulated GPS walk), the
companion server (the wearer's phone page) and the dashboard at http://localhost:8501. Use the dashboard's
**Freeze** / **Long freeze** / **Turn** / **Freeze in a turn** buttons, or press
`f` / `h` / `w` / `s` / `t` (turn left) / `r` (turn right) in the simulator window.

To use Nemotron instead of the built-in fallback rules, copy `.env.example` to
`.env`, paste an NVIDIA API key, and check it with:

```powershell
.venv\Scripts\python nemotron_advisor.py --test
```

### Sign-in, family view and care view

The first visit asks you to create a care-team account, then to scan a QR code
with Google Authenticator (any TOTP app works). Every sign-in after that is
your account (Google, or email + password) plus the 6-digit code.

- **Care team** (doctor / caretaker): Overview with demo controls, Location,
  Care report (episode stats, % during turns, double-checked freezes, watch heart
  rate, agent decisions, CSV export) and Access (add people, reset authenticators,
  pair the wearer's phone).
- **Family**: Today (is my person OK, recent freezes, check-ins, where now) and Location.

Google sign-in: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
and follow the steps in it. For a quick demo without sign-in set
`$env:STEPCUE_AUTH="off"` (and `$env:STEPCUE_ROLE="family"` to see the family view).

### The wearer's phone (Safari) and watch

`companion_server.py` serves a phone page at `https://<laptop-ip>:8443`. Scan the
QR code on the Access page with the iPhone (same Wi-Fi), accept the certificate
warning once (it is self-signed), and add it to the Home Screen. The wearer can tap
**I'm OK** / **I need help** (help pushes and calls the caregiver), answer
"Did you just freeze?" after each episode, say how walking feels, and share phone
motion + GPS while the page is open. The first visit opens a large-text setup assistant
(`companion/setup.html`) built for older users, with notes for a helper. Apple Watch heart rate
and steps come free through Apple Shortcuts (build the shortcut once, see `SHORTCUT.md`), or on
`http://<laptop-ip>:8080/api/health` from the Health Auto Export app (or SensorLog).
All of it stays on the laptop in `companion.json`.

Fresh machine: `python -m venv .venv`, then
`.venv\Scripts\python -m pip install -r requirements.txt`, then
`.venv\Scripts\python train_classifier.py`.

## NVIDIA stack

One `NVIDIA_API_KEY` in `.env` (free at build.nvidia.com) turns on ten NVIDIA models, each with a
specific job. Without the key every feature runs on a labelled local fallback.

| Job | Model | What it does for the wearer and caregivers |
| --- | --- | --- |
| Care agent | Nemotron 3.5 Lightning | Multi-step tool calling: checks which cue setting worked fastest, the time-of-day pattern and the wearer's phone/watch signals, then sets tempo, wrist vibration, alert level and the brief |
| Daily report | Nemotron 3 Ultra 550B | End-of-day report for the care team; falls back to Lightning, then a template |
| Ask StepCue | Nemotron Embed 1B + Nemotron Rerank 8B + Lightning | Retrieval over today's episodes, agent decisions, check-ins and a device guide; answers cite the records |
| Topic guard | NemoGuard 8B topic control | Turns away diagnosis, medication and treatment questions before any answer is written |
| Safety guard | Nemotron Safety Guard 8B v3 | Screens questions, answers and every agent brief |
| Translation | Riva Translate 4B | Briefs for family in 11 languages, cached |
| Voice cues | Riva Translate + Riva Magpie TTS | Calm spoken cues in the wearer's language, played on the phone with a metronome at the cue tempo during a freeze |
| Spoken check-ins | Nemotron 3 Nano Omni | The wearer talks to the phone ("walking is hard today"), and it becomes a check-in. A spoken help request always counts as help |
| Sandbox | NemoClaw / OpenShell | `nemoclaw/`: an OpenClaw skill and an egress preset that allows only NVIDIA endpoints (see `nemoclaw/README.md`) |

Privacy: `nvidia_nim.scrub` strips account names, emails, phone numbers, coordinates and
street addresses from every request. Location never leaves the laptop. Every call is logged
in `nvidia_calls.jsonl` (job, model, latency, ok) and shown on the care team's **NVIDIA AI** page.

```
python nemotron_advisor.py --test          # one agent decision, prints the tools it used
python nvidia_ask.py "Is the cue working?"
python nvidia_report.py                    # writes care_report.json
python nvidia_voice.py --lang en es        # makes companion/voice/*.wav
```

## Files

| File | Role |
| --- | --- |
| `esp32_ankle/esp32_ankle.ino` | Firmware: SoftAP, MPU6050 at 50 Hz over UDP, LED + 150 ms wrist pulses on each beat |
| `sim_esp32.py` | Software ESP32 that speaks the same UDP contract, with a simulated wearer |
| `gait_classifier.py` | Tier 1: Random Forest on 1.2 s windows, 3-window debounce, fires `CUE` / `STOP` |
| `nemotron_advisor.py` | Tier 2: Nemotron agent with read tools, then `update_ankle_cue` for tempo, haptics, alert level, brief |
| `nvidia_nim.py` | Model registry, NIM client, privacy scrub, call ledger |
| `nvidia_guard.py`, `nvidia_ask.py`, `nvidia_report.py`, `nvidia_translate.py` | Guards, Ask StepCue retrieval, Ultra daily report, Riva translation |
| `nvidia_voice.py`, `nvidia_voice_in.py` | Riva TTS voice cues; Nemotron Omni spoken check-ins |
| `nvidia_ui.py`, `knowledge/stepcue_guide.md` | Dashboard pieces for the NVIDIA features; the guide Ask StepCue retrieves from |
| `nemoclaw/` | NemoClaw skill, OpenShell egress preset, privacy-safe snapshot export |
| `caretaker_app.py` | Streamlit dashboard, polls `state.json` every 0.5 s |
| `train_classifier.py` | Trains `gait_model.joblib` from synthetic data, our recordings, Daphnet and/or MHEALTH |
| `RESEARCH.md` | Public datasets, open-source tools, and the plan for our own dataset |
| `LOCATION_AND_DATA.md` | How to collect our own dataset, and location options with only two ESP32s |
| `motion.py` | Steps, cadence and turns from the ankle IMU (any mounting orientation, no magnetometer) |
| `benchmark_motion.py` | Scores turn and step detection against FoG-STAR's video-labelled turns |
| `location_server.py` | Receives phone GPS from OwnTracks (or simulates a walk) into `location.json` |
| `location_io.py`, `location_model.py` | Location storage and geometry, and the tested view-model for the dashboard's Location view |
| `places.json` | The wearer's places (home, park, shop...) used to label freezes by place |
| `auth.py`, `auth_ui.py` | Accounts, roles (family / care team), PBKDF2 passwords, RFC 6238 authenticator codes, lockout; the sign-in screens |
| `companion_server.py`, `companion/` | HTTPS phone page for the wearer and the watch-data intake (pairing key, self-signed cert) |
| `SHORTCUT.md` | One-time recipe for the free Apple Shortcuts heart-rate link the setup page uses |
| `companion_store.py` | Check-ins, freeze confirmations, phone motion, watch heart rate, and how each freeze was double-checked |
| `alert_notifier.py` | Caregiver alerts: free ntfy phone push, plus a Twilio voice call for a prolonged freeze or a help request (dry run until `.env` has keys) |
| `benchmark_datasets.py` | Cross-patient benchmark on FoG-STAR (real freezes) for different training sources |
| `record_session.py` | Guided or key-labelled recording sessions for building our own dataset |
| `record_data.py` | Records labelled data from the real ankle node for retraining |
| `state_io.py` | Atomic `state.json` access (tmp file + `os.replace`, retrying reader, writer lock) |
| `features.py`, `gait_synth.py`, `config.py` | Window features and Freeze Index, synthetic signals, shared constants |

## Locked contracts

- Wi-Fi SoftAP `StepCue_AP` / `stepcue123`, ESP32 `192.168.4.1`, laptop `192.168.4.2`
- Sensor stream, ESP32 → laptop, UDP 5005, 50 Hz: `ts_ms,ax,ay,az,gx,gy,gz\n`
- Cue commands, laptop → ESP32, UDP 5006: `CUE:<bpm>:<haptic_0_or_1>` and `STOP`
- `state.json`: fields exactly as in the shared spec

Choices made on top of the contract (it did not specify them):

- Units: accelerometer in g, gyro in deg/s.
- An episode that is still in progress has `"recovery_sec": null`. That is how
  the advisor spots a freeze lasting more than 4 s.
- `duration_sec` is time spent frozen; `recovery_sec` is time from cue start
  until walking resumed.
- `alert_level` is one of `normal`, `watch`, `alert`.
- The classifier re-sends `CUE` once a second while cueing, and the firmware
  stops by itself if no `CUE` arrives for 5 s (failsafe if the laptop dies).
- Who writes what in `state.json`: the classifier owns `status`,
  `freeze_index` and the episode fields; the advisor owns `bpm`, `haptic`,
  `alert_level`, `caretaker_brief` and each episode's `analyzed` flag.
- Outside the contract: `advisor_log.jsonl` (decision log for the dashboard)
  and UDP 5007 on localhost (simulator remote control).

## When the hardware arrives

1. Check the pin numbers at the top of `esp32_ankle.ino` against the wiring and
   flash it (ESP32 Arduino core 2.x or 3.x). The sketch has not been compiled
   or run on a board yet.
2. Join the laptop to `StepCue_AP`. It should get `192.168.4.2`; if not, set
   that address manually. Allow Python through the firewall for UDP 5005.
3. Use the dashboard sidebar: Device → ESP32 hardware → Play cue, to check the
   LED and wrist motor.
4. Record real data and retrain. The shipped model has only ever seen
   simulated signals:

   ```powershell
   .venv\Scripts\python record_data.py --label WALKING --seconds 60
   .venv\Scripts\python record_data.py --label STATIONARY --seconds 30
   .venv\Scripts\python record_data.py --label FREEZE --seconds 20
   .venv\Scripts\python train_classifier.py --recordings recordings
   ```
5. `.venv\Scripts\python run_demo.py --hardware`

Optional: `train_classifier.py --daphnet <folder>` trains on the public Daphnet
Freezing of Gait dataset (ankle accelerometer only, so gyro features are left
out).

## Data credits

The shipped model is trained on synthetic data plus two public datasets, both CC BY 4.0:

- **FoG-STAR**, Zenodo, doi:10.5281/zenodo.17838806. Cite: Borzì L. et al., *Freezing of
  gait detection: the effect of sensor type, position, activities, datasets, and machine
  learning model*, Journal of Parkinson's Disease 15(1), 163-181.
- **MHEALTH**, UCI Machine Learning Repository, dataset 319. Banos O. et al., IWAAL 2014.

To rebuild the model: download both into `datasets/` (see RESEARCH.md), then run
`.venv\Scripts\python train_classifier.py --mhealth datasets\mhealth\MHEALTHDATASET --fogstar datasets\fogstar`.
