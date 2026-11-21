# Own dataset and location tracking with two ESP32s

Hardware we have: ESP32 waist hub (SoftAP), MPU6050 ankle node with LED,
wrist vibration motor, and a second ESP32. No anchors, beacons, UWB or GPS.

---

## Part 1. Building our own dataset

### What we already have

`record_session.py` records labelled sessions straight from the live UDP
stream, in the same format `train_classifier.py --recordings` reads:

```
python record_session.py --participant P01 --protocol      # guided, auto-labelled
python record_session.py --participant P01                 # someone presses w/s/f/u
python train_classifier.py --recordings recordings --fogstar datasets/fogstar --mhealth datasets/mhealth/MHEALTHDATASET
```

### Phase A: acted data from teammates (this week, no approvals needed)

| Step | Detail |
| --- | --- |
| People | 4-6 healthy volunteers, 2 sessions each on different days. More people beats more minutes per person. |
| Placement | Same as the demo: right ankle, lateral, just above the ankle bone, strapped tight. Write any change in `--placement`. |
| Protocol | `--protocol` runs ~2.5 min: standing, walking, three acted freeze styles (trembling, shuffling, akinesia), turns marked UNLABELED. Run it 2-3 times per session. |
| Ground truth | Film on a phone. Press `c` and clap with the sensor foot at the start: the clap is a spike in the IMU and a sound in the video, so they line up. Fix labels afterwards if the wearer reacted late. |
| Hard negatives | Record things that look like freezes but are not: standing and fidgeting, sitting down, stairs, stepping over a doorway, foot tapping. Label them STATIONARY or WALKING. These are what cut false cues. |
| Splitting | Always test on people the model never trained on (split by participant, never by window). |
| Use | Fine-tune sensor placement, units and the pipeline on *our* hardware. FoG-STAR stays the source of real freezes. |

Acted freezes are not real freezing of gait (they are cleaner and more
rhythmic), so report them separately and never claim clinical accuracy from them.

### Phase B: real patients (only with consent and ethics approval)

Partner with a clinic or Parkinson's group, get IRB / ethics approval and
written consent, use anonymous IDs (`--condition parkinsons-consented`),
store no names or video faces, and have a clinician label freezes from video.
This is the only way to get genuinely new freeze data.

### Using the second ESP32 for data collection (no add-ons)

1. **Wireless label clicker.** Every ESP32 dev board has a BOOT button on
   GPIO0. Flash the second ESP32 to join `StepCue_AP` and send a UDP packet
   to the laptop on each press (press = "freeze started", release = "ended").
   The observer can walk next to the wearer instead of sitting at the laptop,
   and labels are timestamped within milliseconds. It would use a new port
   (e.g. 5008) so the locked 5005/5006 contracts are untouched.
2. **Second-ankle node later.** If another MPU6050 shows up, both ankles
   give left/right asymmetry, which helps catch freezes during turns.
   FoG-STAR already has both ankles to test that idea first.

---

## Part 2. Location with only two ESP32s

Ranked from most to least practical. The short answer: **we can track
movement (steps, distance, turns) well, and which room someone is in
roughly, but not an x/y position on a map** without more hardware.

### 1. Inertial dead reckoning from the ankle IMU (recommended, now built)

**Built:** `motion.py` runs inside `gait_classifier.py` on the same 50 Hz
samples. It writes a `motion` block (steps, cadence, turn counts) to
state.json, and each episode gets `during_turn`. Both are optional extra
keys, and none of the locked fields change. The dashboard shows them
under the time-in-state bar, and the Nemotron advisor gets them as
context. `python benchmark_motion.py` scores it on FoG-STAR (22 people with
Parkinson's, video-labelled turns), using its 62 ankle recordings:

| Check | Result |
| --- | --- |
| Labelled turns found | 97% (492 / 505), direction correct 95% |
| Detected turns that are real | 95%; 0.13 false turns per minute of straight walking |
| Turn angle | median 166 deg (most FoG-STAR turns are about-turns) |
| Left vs right ankle stride count | 6% median difference |
| Cadence vs autocorrelation reference | 3.6 steps/min mean error, 93% within 10% |
| Strides counted while sitting/standing | 0.9 per minute |
| Freezes that began in a labelled turn | 93 of 101 (the detector flagged a turn around 86) |

Caveats: the FoG-STAR tasks are turn-heavy, so "93 of 101 freezes were in turns"
reflects the protocol as well as the people. FoG-STAR's sensors use a
mirrored gyro sign, so the benchmark flips it. For our MPU6050, check once on
hardware that a left turn is logged as "left". Stride *length* and
distance are not built, because there is no ground truth to check them against.

Side finding: the old spectral cadence (`features.estimate_cadence`) read
double the true cadence on about 20% of slow Parkinsonian walking bouts (it locked onto a
harmonic). Because the advisor sets the cue tempo from pre-freeze cadence,
the live loop now uses counted strides instead.

The ankle sensor already gives what matters most for freezing:

- **Steps, cadence and stride length**: the shank gyroscope swings once per
  stride and is still while the foot is flat. Integrating between those
  still moments (zero-velocity updates, "ZUPT") stops drift building up.
  Shank IMU studies report roughly 3-5% stride-length error, mostly on
  straight walking ([JNER 2014](https://jneuroengrehab.biomedcentral.com/track/pdf/10.1186/1743-0003-11-152),
  [Frontiers 2019](https://frontiersin.org/articles/10.3389/fphys.2019.01530/pdf)).
- **Turns and turn angle**: integrate the vertical gyro axis over a few
  seconds. Turning is one of the most common freeze triggers, so "freezes
  happen mostly during turns" is a useful caregiver pattern.
- **Distance walked per day** and walking bouts.

Limits: no magnetometer on the MPU6050, so heading drifts over minutes and
the drawn path will bend. Gait-event detection also gets worse during turns
(F1 91% during turns vs near-perfect when straight,
[JNER 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7866479/)).
Use it for relative measures (steps, turns, metres), not a map.

Open-source tools:
- [pyshoe](https://github.com/utiasSTARS/pyshoe): Python error-state EKF
  with four zero-velocity detectors and an LSTM detector, plus a public
  dataset ([paper](https://pdf.arxiv.org/pdf/1910.00529)). Built for the
  foot, but the detectors and EKF transfer to the shank.
- [OpenShoe MATLAB](https://github.com/hcarlsso/ZUPT-aided-INS): the
  reference ZUPT-aided INS (SHOE, acceleration variance, angular-rate energy
  detectors).
- Orientation filters (Madgwick / Mahony) for tilt.

We can validate turn detection offline on FoG-STAR, which labels turns
(activity codes 6-7), before touching hardware.

### 2. Room-level location from existing home Wi-Fi (RSSI fingerprinting)

The ESP32 can scan the signal strength of the home router and neighbours'
networks. Walk through each room once and record the scan, then a kNN
classifier guesses the room from new scans. No anchors needed, just the
routers already there.

- Expect room-level at best, not metres; signal strength is noisy through
  walls and bodies ([arXiv 2312.15744](https://arxiv.org/html/2312.15744v1)).
- Open ESP32 fingerprint datasets to practise on:
  [single-AP ESP32 datasets](https://ceur-ws.org/Vol-3434/paper6.pdf),
  [ESP32-C5 RSSI dataset](https://ieee-dataport.org/documents/rssi-fingerprint-dataset-low-cost-wifi-indoor-localization-using-esp32-c5-devices).
- **Catch:** the ESP32 has one radio. Scanning while it runs `StepCue_AP`
  drops the ankle stream for up to ~2.4 s
  ([Espressif FAQ](https://docs.espressif.com/projects/esp-faq/en/latest/software-framework/wifi.html),
  [forum report](https://esp32.com/viewtopic.php?p=153081)). So do the
  scanning on the **second ESP32**, carried at the waist, and send the room
  guess over UDP. The ankle loop never pauses.
- Do it on-device. Don't send scans to cloud geolocation services (privacy,
  and Mozilla Location Service has shut down).

### 3. Second ESP32 as one fixed "hotspot" beacon

Put the second ESP32 at a known trouble spot (a doorway, the kitchen
threshold) broadcasting a beacon. The waist hub reads its signal strength:
strong = near that spot. This is proximity to one place, not position, but
it answers a real question: "do freezes cluster at the bedroom doorway?"
Doorways and tight spaces are classic freeze triggers. Same single-radio
caveat: the hub only sees beacons on its own channel, so put both on the
same channel.

### 4. Wi-Fi FTM ranging (only if the boards are newer chips)

Wi-Fi Fine Timing Measurement gives roughly metre-level distance between
two ESP32s by timing packets. ESP-IDF supports it on ESP32-C2/C3/C6/S2/S3,
and the original ESP32 is not in the list
([ESP-IDF FTM example](https://github.com/espressif/esp-idf/blob/dc859c1e67e/examples/wifi/ftm/README.md)).
Check the chip name printed on the module. Even with FTM, two boards give a
distance, not a position (you need three or more for x/y). Dataset:
[ESP32-C3 FTM/RSSI](https://www.github.com/WiFiLocalization/ESP32C3_WiFi_FTM_RSSI_Indoor_Localization).

### 5. Wi-Fi CSI sensing (presence, not location)

[Espressif esp-csi](https://github.com/espressif/esp-csi) reads channel
state information on every ESP32 and its
[esp-radar](https://components.espressif.com/components/espressif/esp-radar/versions/0.3.4/readme)
component detects presence and motion. Good for "someone is moving in this
room", weak for locating a person; reviewers call it more of a hint than
reliable detection
([CNX Software](https://www.cnx-software.com/2022/08/08/esp-wifi-csi-detects-humans-with-wifi-signals-only-no-sensor-needed/?amp=1)).
Not worth hackathon time.

### What we would show caregivers

Keep the dashboard descriptive, never diagnostic:

- "4 of today's 5 freezes happened while turning."
- "Freezes cluster near the bedroom doorway" (room or beacon).
- "Walked 1.2 km today in 14 walking bouts."

Location is sensitive data: keep it on the laptop, show it only to the
caregiver, and allow it to be switched off.

### Recommended order

1. ~~Turn detection and steps from the ankle IMU, validated on FoG-STAR
   turn labels.~~ Done (`motion.py`, `benchmark_motion.py`).
2. Second ESP32 as a label clicker for our own recordings.
3. Room fingerprinting on the second ESP32 if time remains.

---

## Part 3. Outdoor GPS map (built)

The ESP32s have no GPS, so outdoor position comes from the phone the wearer
already carries, using **[OwnTracks](https://owntracks.org/booklet/tech/http/)**.
It's free and open source, on iOS and Android, and it POSTs JSON to a server you
choose. No cloud account is involved.

```
python location_server.py                  # real phone
python location_server.py --simulate       # demo walk (only moves while the classifier says WALKING)
```

In OwnTracks: Settings, Connection, Mode **HTTP**, URL
`http://192.168.4.2:5010/pub` when the phone is on `StepCue_AP`. Add
`--token <secret>` and use the same value as the OwnTracks password.

- `location.json` holds the latest fix and today's trail. It's a separate
  file, written atomically, so the locked `state.json` is untouched.
- `places.json` lists the wearer's places (home, park, shop...) with a radius.
  The demo uses public streets around a park; replace them with your own.
- The dashboard **Location** view uses a free OpenStreetMap basemap
  ([CARTO Voyager](https://carto.com/basemaps), no API key). It shows the live
  position with its accuracy circle, today's walk, and an orange dot where each
  freeze happened. It also shows freezes by place, where they are now,
  distance walked, time away from home, and a timeline of
  arrivals, departures and freezes.
- Freezes are matched to the GPS fix closest in time (within 2 minutes).
  A freeze with no nearby fix is not shown on the map rather than guessed.
- Location stays on the laptop. CARTO's free basemaps are meant for
  non-commercial use, so check their terms before any product launch.

## Part 4. Phone-call alerts: is there a free API?

No provider gives free automated voice calls indefinitely. The
options are trial credits or a few cents a minute
([comparison](https://geekflare.com/guides/best-voice-call-apis/),
[pricing overview](https://toolradar.com/guides/voice-api-pricing-compared)):

| Option | Cost | Notes |
| --- | --- | --- |
| **Twilio** Programmable Voice | Trial credit (about $15), then pay per minute | Easiest API. Trial calls only reach verified numbers and play a trial message first ([overview](https://dev.to/0012303/twilio-has-a-free-trial-api-send-sms-make-calls-and-build-communication-apps-without-upfront-1eak)). |
| **Telnyx** / **Plivo** / **Vonage** | Trial credit, cheaper per minute than Twilio | Telnyx bills its API fee and the SIP leg separately ([source](https://www.forasoft.com/blog/article/twilio-alternatives-voice-ai)). |
| **[ntfy](https://github.com/binwiederhier/ntfy)** push notification | Free, open source, self-hostable | Not a phone call, but "max" priority gives a loud, long-vibrating alert on the caregiver's phone. iOS and Android apps. Best free option. |

**Built:** `alert_notifier.py` (started by `run_demo.py`). It reads state.json
and uses the dashboard's own status rules, so it works even if the advisor is down:

| Situation | What the caregiver gets | Repeat limit |
| --- | --- | --- |
| "Please check in": a freeze over 4 s, or the advisor's `alert` | Max-priority ntfy push and a Twilio voice call | Push every 3 min, call every 10 min |
| "Keep an eye out" (advisor `watch`) | Normal push | Every 15 min |
| Walking again after a check-in | Low-priority all-clear push | |
| Sensor silent for 2 min after being live | High-priority push | Once per outage |

Setup:
1. Install the ntfy app (iOS or Android) and subscribe to a long random
   topic, e.g. `stepcue-7f3k9q2m`. Put it in `.env` as `NTFY_TOPIC=`.
2. Optional call: create a Twilio trial account, verify the caregiver's
   number, and fill `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`
   and `CAREGIVER_PHONE` in `.env`.
3. Check it with `python alert_notifier.py --test`, or `--test --call` for a call,
   or use **Send a test push** in the dashboard sidebar.

Without keys it runs as a dry run: alerts are decided and logged
(`notify_log.jsonl`, shown under "Caregiver alerts" on the dashboard) but
nothing is sent. Messages carry no names or location. The AI brief is added
only with `NOTIFY_INCLUDE_BRIEF=1`.

## Part 5. Apple Watch

Possible, but not directly from Windows or the ESP32: watch data goes
through an iPhone.

| Route | Effort | What you get |
| --- | --- | --- |
| **[SensorLog](https://apps.apple.com/us/app/sensorlog/id388014573)** app (iPhone + Watch) | Hours, no Mac needed | Streams watch accelerometer data over HTTP. Watch streaming only works while the app is in the foreground; background logging is accelerometer only, up to 50 Hz. |
| **[Sensor Logger](https://apps.apple.com/us/app/sensor-logger/id1531582925)** app | Hours | iPhone + Watch accelerometer and gyroscope, with HTTP/MQTT streaming. Check whether streaming works from the watch itself. |
| **Own watchOS app** (Swift, Core Motion + HealthKit) | Days, needs a Mac with Xcode | Full control: wrist IMU at high rate and heart rate. Free provisioning runs on your own watch; distribution needs the $99/yr developer program. |
| **Apple's Movement Disorder API** ([CMMovementDisorderManager](https://developer.apple.com/documentation/coremotion/cmmovementdisordermanager)) | Needs a special entitlement from Apple | Apple's own resting-tremor and dyskinesia scores, computed on the watch. Results come about every few minutes and are kept for 7 days. A research-grade option, not a hackathon one. |

**Built:** `companion_server.py` takes watch data on `POST /api/health`:

| Source | Setup | Arrives |
| --- | --- | --- |
| [Health Auto Export](https://apps.apple.com/us/app/health-auto-export-json-csv/id1115567069) (iPhone, reads Apple Health) | REST API automation, JSON, Heart Rate + Step Count, header `X-StepCue-Key` | Batches every few minutes in the background (automations are a paid feature of the app) |
| SensorLog (Watch or iPhone) | HTTP stream, JSON, URL ending `?key=...` | Live, but only while the app is open |
| Anything else | POST JSON with `heart_rate` and/or `steps` | Whenever it posts |

Heart rate is shown on the Care report and the family view as context only. The
phone page adds the **wearer's own confirmation** of each freeze and the phone's
motion during it, which the dashboards show as "Checked by".

For StepCue the useful extra signal is **arm swing** (it drops before a freeze) and
wrist tremor. The fastest prototype is a SensorLog → HTTP receiver feeding
the same pipeline as `location_server.py`.
