# AGENTS.md: handoff for coding agents (Codex, Claude Code, ...)

Read this first. It is the project's state as of 2026-11-21, written so another agent can
continue without the original conversation. The README covers how to run things; this file
covers the rules, the decisions behind them, what is verified and what is not.

## What StepCue is

An assistive wearable prototype for **freezing of gait (FoG)**, built for the tidalBYTE
hackathon. Judging: Technicality, Ingenuity, Diversity, Applicability, Looks. Target prizes:
**Best Design** and **Best Use of Nemotron** (NVIDIA, prize: Brev credits).

- Hardware: ESP32 waist hub (SoftAP), MPU6050 ankle node, wrist DC motor, a second ESP32.
  No extra hardware or anchors.
- Tier 1 (local, fast): `gait_classifier.py`, a Random Forest that reads ankle IMU over UDP
  and fires walking cues.
- Tier 2 (reasoning): `nemotron_advisor.py`, a Nemotron tool-calling agent that tunes the cue
  and writes a caregiver brief.
- Caregiver dashboard: `caretaker_app.py` (Streamlit), with patient, family and care team portals.
- Wearer's phone: `companion_server.py` plus `companion/` (an HTTPS page for Safari).
- Location: OwnTracks or a simulator, kept on the laptop. Alerts: ntfy push and Twilio call.

## Rules you must keep

### Locked contracts (teammates' firmware depends on them; do not change)
- Wi-Fi SSID `StepCue_AP`, password `stepcue123`. ESP32 is 192.168.4.1, laptop is 192.168.4.2.
- Sensor in: UDP **5005**, CSV `ts_ms,ax,ay,az,gx,gy,gz\n` at 50 Hz (accel in g, gyro in deg/s).
- Cues out: UDP **5006**, `CUE:<bpm>:<haptic>` or `STOP`.
- `state.json` is written atomically (tmp file, then `os.replace`, in `state_io.py`). Readers
  must tolerate a partial or missing file. Episode fields: `id, timestamp, duration_sec,
  pre_freeze_cadence, cue_bpm, recovery_sec (null while ongoing), analyzed`, plus optional
  `during_turn` and `cue_haptic`.
- Constants live in the LOCKED section of `config.py`.

### Clinical boundary
- Assistive cueing, pattern tracking and caregiver awareness only. **Never diagnose, never
  give medication or treatment advice, never predict disease progression.**
- This is enforced in code by `nvidia_guard.py` (`OFF_LIMITS` regex, NemoGuard topic control,
  Nemotron Safety Guard) and by `sanitize()` in `nemotron_advisor.py`. Keep every new
  user-facing text path behind these.

### Privacy
- Location never leaves the laptop. Alert messages carry no names and no location.
- Every request to NVIDIA goes through `nvidia_nim.NIM`, which scrubs account names, emails,
  phone numbers, coordinates and street addresses (`nvidia_nim.scrub`). Do not call NVIDIA
  APIs any other way.
- Never put the user's personal details in external requests.
- No real notifications, calls or external uploads without the user's explicit OK.
  Tests mock the network; keep it that way.
- Never commit secrets. These are gitignored and must stay that way: `.env`, `users.json`,
  `demo_accounts.txt`, `companion_key.txt`, `certs/`, `.streamlit/secrets.toml`,
  `companion.json`, `location.json`, `state.json`, `advisor_log.jsonl`, `nvidia_calls.jsonl`,
  `embed_cache.json`, `translations.json`, `care_report.json`, `nemoclaw/snapshot/`.

### Design (the target is Best Design)
- The audience includes older adults: large type (20px base on the phone, 68px buttons), high
  contrast, color carries meaning only (teal = ok, amber = watch, red = alert), and every
  status also has a text label. Font: Atkinson Hyperlegible. Plain words, not developer words.
- On the phone, the wearer only taps buttons. Nothing should need typing or setup by a helper.
- Calm when OK, loud only when not. Stale data must look different from real data.

## Environment

- Windows 11, PowerShell, plus Git Bash. No NVIDIA GPU, no WSL. Python 3.14 venv in `.venv`.
- Run things with `.venv/Scripts/python ...`. Streamlit 1.67 is used (`st.login`, `st.fragment`,
  `st.segmented_control`, `st.pills`, `width="stretch"`).
- Streamlit caches imported modules: **restart the Streamlit server** after editing
  `auth_ui.py`, `nvidia_ui.py` or other imported modules.
- Tests: `.venv/Scripts/python -m unittest discover -s tests -q` gives **130 tests, all
  passing**. pytest is not installed.
- Repo: https://github.com/akshajag/stepCue (public), branch `main`.

### Ports
| Port | What |
| --- | --- |
| 5005 / 5006 | sensor in / cues out (locked) |
| 5007 | simulator control: `MODE:WALK`, `MODE:STAND`, `MODE:FREEZE` (sim only) |
| 5010 | location server (OwnTracks in, or `--simulate`) |
| 8443 | phone page, HTTPS with a self-signed cert in `certs/` |
| 8080 | plain HTTP watch-data intake (`/api/health`) |
| 8501 | dashboard with real sign-in |
| 8540 / 8541 / 8542 | dev-only dashboards with `STEPCUE_AUTH=off STEPCUE_ROLE=patient|care|family` |

Start everything: `.venv/Scripts/python run_demo.py`. The browser pane can't open the
self-signed HTTPS page. To preview the phone page, serve `companion_server.make_handler(key,
serve_page=True)` over plain HTTP on localhost, and set localStorage `stepcue_setup_done=1`
to skip the setup redirect.

## What has been built (by area)

### Sign-in and portals (`auth.py`, `auth_ui.py`, `caretaker_app.py`)
- Roles: `patient`, `family`, `care` (`auth.ROLES`).
- Passwords use PBKDF2-SHA256 (240k rounds). Second step is TOTP (RFC 6238, ±1 step, no code
  reuse). 5 failed tries lock the account for 300 s. Sessions last 12 h. Optional Google
  sign-in via `st.login` when `.streamlit/secrets.toml` has `[auth]` (template in
  `.streamlit/secrets.toml.example`; not configured).
- Demo accounts in the local `users.json`: Akshaj Agadi (patient, akshaj@stepcue.local),
  Neel Marripalapu (care, neel@stepcue.local), Arush Gayer (family, arush@stepcue.local).
  Passwords are in the gitignored `demo_accounts.txt`; never print them. Nobody has enrolled
  an authenticator yet; first sign-in shows the QR code.
- Portals:
  - **Patient "My day"**: big I'm OK / I need help (two taps) buttons, how walking feels,
    today's freezes, a QR code for the phone page.
  - **Family "Today"**: status, the brief (translatable), freezes, phone/watch info,
    Ask StepCue, Location.
  - **Care team**: Overview, Location, Care report, **NVIDIA AI**, Access (users, reset
    authenticator, pairing QR).
- `STEPCUE_AUTH=off` bypasses sign-in for development only.

### Phone companion (`companion_server.py`, `companion_store.py`, `companion/`)
- Pairing key in `companion_key.txt`, carried in the URL hash `#key=` (never sent to the
  server). API calls send it in the `X-StepCue-Key` header or `?key=`.
- API endpoints: `/api/status`, `/api/checkin`, `/api/confirm`, `/api/phone` (motion + GPS),
  `/api/health` (watch data), `/api/voice` (spoken check-in), `/api/prefs` (wearer language).
- `setup.html` is a buttons-only setup wizard. The Apple Watch steps stay hidden unless
  `STEPCUE_SHORTCUT_URL` is set. The user shelved the Shortcuts route (see `SHORTCUT.md`).
- `index.html` cards:
  - freeze confirmation ("Did you just freeze?")
  - I'm OK / I need help
  - **Say how you feel** (spoken check-in; hidden until the laptop has an NVIDIA key)
  - **Spoken walking cues**: a voice clip, then a WebAudio metronome at the cue tempo while
    status is FREEZE
  - how walking feels
  - share phone motion and GPS
  - Apple Watch
- Help requests trigger `alert_notifier.py` (one push plus a call per new help tap; dry run
  without keys).

### Android app (`android/`, replaces the shelved Apple Watch route)
- Plain Java, no Gradle. Build: `.venv/Scripts/python android/build.py` (SDK in
  `%LOCALAPPDATA%/StepCueAndroidSdk`, JDK 25) writes `companion/stepcue.apk`. The signing key
  `android/stepcue.jks` is local and gitignored; a different key means uninstalling first.
- Install flow: dashboard QR -> `http://<laptop-ip>:8080/android#key=...` (`companion/android.html`,
  served without a key along with `/stepcue.apk`) -> Download, Install -> "Open StepCue and connect"
  (`intent://pair?h=&p=&k=`) -> Start -> Allow. Buttons only.
- `SensorService` (foreground service, health|location, wake lock) reads accelerometer, linear accel,
  gravity, gyroscope, rotation, compass, barometer, light, proximity, step detector/counter, GPS and
  battery at 50 Hz; `Features.java` computes cadence, Moore freeze index, turns, stairs, in-pocket.
  Every 2 s it POSTs `/api/android`; every 1.5 s it reads `/api/status` and during FREEZE vibrates at
  the cue bpm and speaks one cue (Android TTS).
- Laptop: `companion_store.add_android` (whitelisted fields, location goes to location.json only),
  `android_summary`, and an Android freeze suspect counts as "phone agrees" in verification. Shown in
  the dashboard's phone cell and passed to the Nemotron `get_phone_and_watch` tool.
- Gait math test: `javac -d out android/src/com/stepcue/companion/Features.java android/test/FeaturesTest.java && java -cp out FeaturesTest`.
- Verified live with a Google Pixel 9 (12 sensors at 50 Hz).
- **Activity page** (`phone_ui.py`, care team and family): 3 SVG gauges (walking pace, freeze score,
  phone battery), 8 tiles, a per-minute pace + freeze-score chart (last 3 h) and a 7-day steps bar
  chart with a goal line (`STEPCUE_STEP_GOAL`, default 5000). History comes from
  `companion_store.android_days` / `android_minutes` (per-day totals and per-minute rows rolled up
  in `add_android`). Days without data show "no data", never 0. No clinical vitals (blood pressure,
  glucose, etc.): StepCue has no sensor for them, so they are not shown.

### Per-patient training (`personal_model.py`, `training_ui.py`)
- Labels: sitting, standing, walking, frozen. A helper taps the label on the dashboard's **Training**
  page (care team) or the phone's "Teach StepCue" screen; the label lives in `training/session.json`.
- While a label is on (after a 3 s settle), every `/api/android` reading is saved to
  `training/<patient>/samples.jsonl` with 11 phone features plus 20 ankle features. The ankle features
  come from `gait_classifier.py`, which now writes an optional `ankle` block (17 window features +
  mean accel x/y/z) into state.json. The phone sends 1 reading a second while training.
- "Build the model" (`pm.train`) fits a Random Forest per patient: phone-only, plus phone + ankle when
  most samples have ankle data. Scored with GroupKFold by recording segment (falls back to a random
  split, flagged "optimistic", when each label was recorded only once). Also learns their walking
  pace (25th to 75th percentile cadence).
- Live: `/api/android` runs the active patient's model; `status.phase` goes to the phone (which buzzes
  the cue when it says frozen with 70%+ confidence), the Activity page and the Nemotron tool.
- `training/` is gitignored (personal movement data).

### ESP32 over Bluetooth (`esp_bt_bridge.py`, `esp32_ankle_bt/`)
- The ESP32 links to the laptop by Bluetooth Classic SPP (pair "StepCue_Ankle" in Windows; COM port
  found automatically) or BLE Nordic UART (`--ble NAME`, needs `pip install bleak`). The bridge
  forwards the same CSV lines to UDP 127.0.0.1:5005 and returns cues from UDP 5006, so the locked
  contract and `gait_classifier.py` are unchanged. `python run_demo.py --bluetooth` starts it.
- `esp32_ankle_bt/esp32_ankle_bt.ino` is the BluetoothSerial version of the firmware. Not compiled
  or run on hardware yet; pyserial is optional (falls back to opening the COM port directly).

### NVIDIA stack (for Best Use of Nemotron)
All of it goes through `nvidia_nim.py`: the `MODELS` registry (each model can be overridden
in `.env`), the `NIM` client (`chat`, `text`, `embed`, `rerank`, `thinking=` toggles
`chat_template_kwargs.enable_thinking`), the privacy scrub, and the call ledger in
`nvidia_calls.jsonl`, which the NVIDIA AI page shows.

| Job | Model id | File |
| --- | --- | --- |
| care agent | `nvidia/nemotron-3.5-lightning-30b-a3b` | `nemotron_advisor.py`: up to 4 rounds; read tools `get_cue_effectiveness`, `get_time_pattern`, `get_phone_and_watch`, `get_episode_history`; the last round forces `update_ankle_cue`; the brief is guarded; logs `tools` and `guard` in `advisor_log.jsonl` |
| daily report | `nvidia/nemotron-3-ultra-550b-a55b` (then Lightning, then a template) | `nvidia_report.py` writes `care_report.json` |
| Ask StepCue | embed `nvidia/nemotron-3.5-embed-1b`, rerank `nvidia/nemotron-3.5-rerank-8b` (at `ai.api.nvidia.com/v1/retrieval/...`), answer with Lightning | `nvidia_ask.py`; corpus = today's episodes, agent decisions, check-ins and `knowledge/stepcue_guide.md`; citations like `[e3]`; keyword fallback |
| topic guard | `nvidia/llama-3.1-nemoguard-8b-topic-control` | `nvidia_guard.py` (system prompt must end with the exact on-topic/off-topic sentence) |
| safety guard | `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | `nvidia_guard.py` (`parse_safety` handles JSON and plain-text output) |
| translation | `nvidia/riva-translate-4b-instruct-v2` (system message `en-es-us` style) | `nvidia_translate.py`, cached in `translations.json` |
| voice cues | Riva Translate, then Riva Magpie TTS on `grpc.nvcf.nvidia.com:443`, function-id `877104f7-e885-42b9-8de8-f6e4c6303969` | `nvidia_voice.py` writes `companion/voice/<lang>/*.wav` and `manifest.json`; needs `nvidia-riva-client` (installed in `.venv`) |
| spoken check-ins | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (`input_audio`, wav) | `nvidia_voice_in.py`; distress words always turn into `help` |
| sandbox | NemoClaw / OpenShell | `nemoclaw/`: OpenClaw skill `skills/stepcue-care/SKILL.md`, egress preset `presets/stepcue-nvidia.yaml`, `make_snapshot.py` (scrubbed data export), README with Brev steps |

Dashboard UI for all of this is in `nvidia_ui.py` (the care team's NVIDIA AI page,
`ask_box`, `brief_in_language`, `chain` for tool-trace chips).
`gait_classifier.py` now records `cue_haptic` on each episode.

## Not verified yet (be honest about these)

1. **No live NVIDIA call has ever run.** There is no `NVIDIA_API_KEY` on this laptop (no
   `.env`). Every NVIDIA path is unit-tested with a fake client (`tests/test_nvidia.py`) and
   runs on local fallbacks. Formats came from NVIDIA docs, not live responses. Unconfirmed:
   - whether hosted Lightning accepts `tools` / `tool_choice` (the code falls back to
     `tool_choice="auto"` on a 400/422),
   - whether hosted Safety Guard applies its prompt template to plain chat messages,
   - whether Omni accepts `input_audio`,
   - the Riva TTS voice names (run `python nvidia_voice.py --list-voices`).
2. **NemoClaw has never been run.** The laptop can't run it (no GPU, no WSL). It is meant for
   the hackathon's Brev launchable. gRPC through `tls: skip` is unconfirmed.
3. Real sign-in with the three demo accounts, including authenticator enrolment, has not been
   tried end to end (the role views were checked with `STEPCUE_AUTH=off`).
4. Spoken check-ins and voice cues haven't been tested on a real iPhone
   (MediaRecorder → 16 kHz WAV conversion, audio unlock on tap).
5. Google sign-in isn't configured. The Apple Watch/Shortcuts route is shelved.

## Suggested next steps (in order)

1. Get the key: the user signs in at build.nvidia.com, generates a key and puts
   `NVIDIA_API_KEY=nvapi-...` in `.env` themselves (agents must not type or copy keys). Then:
   - `.venv/Scripts/python nemotron_advisor.py --test` (check `tools used:` shows a read tool
     before `update_ankle_cue`)
   - `.venv/Scripts/python nvidia_ask.py "Is the cue working?"` (check `retrieval` is
     `nemotron-embed+rerank` and `layers` include both guards)
   - `.venv/Scripts/python nvidia_report.py` (check `source` is the Ultra model)
   - `.venv/Scripts/python nvidia_voice.py --list-voices`, then `--lang en es`
   - Fix any format mismatches. Keep fallbacks and tests green.
2. Run `run_demo.py`, force a freeze (`MODE:FREEZE` to UDP 5007 for about 6 s, then
   `MODE:WALK`), and check the NVIDIA AI page: call ledger, agent tool chips, Ask StepCue,
   "Write it now" report.
3. Test the phone page on a real iPhone over the LAN (`https://<laptop-ip>:8443`, accept the
   cert): voice cues during a freeze, spoken check-in.
4. Demo polish for judging: a scripted 3-minute demo path, screenshots at phone width
   (~390 px) and desktop.
5. Optional: Brev + NemoClaw run following `nemoclaw/README.md`.

## Gotchas learned the hard way

- Bash heredocs with nested quotes break in this shell. Write Python edit scripts to a temp
  file and run them, or use a proper file-edit tool.
- `companion_store.read_companion()` only keeps keys present in `empty()`. Add new top-level
  keys there (that's how `prefs` was added).
- In Streamlit, create `st.columns` after the header markdown, or the content renders above it.
- Default arguments bind at definition time. Patch module constants such as `CACHE_FILE` and
  resolve them at call time (an `EmbedIndex` bug came from this).
- Don't let debug runs write to the real `nvidia_calls.jsonl`. The dashboard shows it as real
  activity.
- The desktop app's browser pane rejects the self-signed cert; that's expected.
