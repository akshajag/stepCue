# StepCue: open datasets, open-source tools, and our own dataset

Research for Tier 1, the local gait loop (`esp32_ankle.ino` → `gait_classifier.py`).
Researched 2026-11-21. Licenses and details come from the dataset pages or papers
linked below. Re-check each license on its own page before you publish or ship
anything trained on it.

## The short version

1. **Best public match for our hardware: FoG-STAR.** It has ankle IMUs with both
   accelerometer and gyroscope, 22 people with Parkinson's, and labels for each
   freeze, including what kind of freeze it was. CC BY 4.0. Resample 60 → 50 Hz.
2. **Second ankle source: the University of Twente semi-free-living set.**
   Right-ankle accelerometer + gyroscope, CC0, shipped as preprocessed tensors.
3. **Real walking and standing data at exactly our settings: UCI MHEALTH.**
   50 Hz, accelerometer + gyroscope on the left ankle, CC BY 4.0. It has no
   freezes, but it fixes the biggest weakness of our simulated data: real human
   walking. **A loader is already built in** (`--mhealth`).
4. **Daphnet** (loader already built in) is the classic benchmark, but it has
   accelerometer only, at 64 Hz.
5. **Make our own dataset, in two phases.** First, team members act out freezes
   using a guided protocol (`record_session.py`, built and tested). This tunes
   sensor placement and the pipeline, but acted freezes are not real freezes.
   Real patient data needs consent and ethics approval (see the last section).
6. **Most useful tool: emlearn.** It turns our scikit-learn Random Forest into C
   so the ESP32 can detect freezes itself if Wi-Fi drops.

---

## 1. Public datasets

| Dataset | People | Sensors and body site | Rate | Labels | License | Fit for StepCue |
| --- | --- | --- | --- | --- | --- | --- |
| [FoG-STAR](https://zenodo.org/records/17838806) ([paper](https://www.nature.com/articles/s41597-026-06645-1)) | 22 with PD | 4 IMUs (accel + gyro): both ankles, wrist, lower back | 60 Hz | Freeze events; freeze type (shuffling, trembling, akinesia); activities | CC BY 4.0 (stated by papers that reuse it; confirm on Zenodo) | **Excellent.** Same site and sensors |
| [Twente ankle, semi-free-living](https://data.4tu.nl/datasets/40e06061-f441-43b5-9235-006829206509) ([paper](https://www.mdpi.com/1424-8220/25/6/1895)) | PD | Right-ankle accel + gyro | not confirmed | Freeze / no freeze; all-activities and FoG subsets | CC0 | **Very good.** Already preprocessed tensors, so the window length may differ from ours |
| [Turning-in-place FoG](https://figshare.com/articles/dataset/A_public_dataset_of_video_acceleration_and_angular_velocity_in_individuals_with_Parkinson_s_disease_during_the_turning-in-place_task/14984667) ([paper](https://www.frontiersin.org/articles/10.3389/fnins.2022.832463)) | 35 with PD | 1 IMU (accel + gyro) on the shank of the more affected side, plus video | 128 Hz | Freezes marked by two movement-disorder specialists | CC BY 4.0 (per the article; confirm on figshare) | **Good**, but turning only. The video helps with learning to label |
| [Multimodal FoG (Mendeley)](https://data.mendeley.com/datasets/r8gmbtv7w2/3) ([paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9546845/)) | 12 with PD | Accelerometer from custom **MPU6050** hardware (same chip as ours), EEG, EMG, skin conductance | see paper | Freezes labelled by two physicians; 3 h 42 min total, 1 h 28 min frozen | check Mendeley page | Good, especially because it uses the same sensor chip. Check sensor placement |
| [Daphnet FoG (UCI)](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) | 10 with PD | Accelerometer only: ankle (shank), thigh, trunk | 64 Hz | Freeze / no freeze | CC BY 4.0 | OK. No gyro. **Loader built in** (`--daphnet`) |
| [Kaggle tlvmc FoG prediction](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction) ([paper](https://link.springer.com/10.1038/s41467-024-49027-0)) | 100+ with PD, about 5,000 freezes | One lower-back 3-axis accelerometer | 128 / 100 Hz | Start hesitation, turn freeze, walking freeze | Competition rules; **not confirmed**, check before use | Weak: wrong body site. Useful for reading winning approaches |
| [WearGait-PD](https://www.nature.com/articles/s41597-026-06806-2) | PD + age-matched controls | IMU accel + gyro, sensorised insoles, camera labels | see paper | Gait tasks | check page | Possible source of extra walking data |
| [UCI MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | 10 healthy | Accel + gyro on the **left ankle** (also chest and wrist) | **50 Hz** | 12 activities, including standing still and walking | CC BY 4.0 | **Excellent for WALKING / STATIONARY.** No freezes. **Loader built in** (`--mhealth`) |

### How to use them with our model

- **Units and rate:** our wire format is accelerometer in g, gyro in deg/s,
  50 Hz. Divide m/s² by 9.81, divide mg by 1000, and resample any other rate
  to 50 Hz. The Daphnet loader shows how.
- **Body site matters more than dataset size.** Ankle and shank data
  transfers to our ankle node. Lower-back data, such as the Kaggle set, mostly
  does not.
- **Freeze type matters.** Trembling and shuffling freezes show up in the
  Freeze Index (3–8 Hz power). **Akinetic freezes, where the person stops
  completely, look almost like standing still to a single ankle sensor.**
  Expect lower recall on those. FoG-STAR labels freeze types, so you can
  measure this directly.
- **Split by person, not by time.** Testing on people the model never saw is
  the honest number. The trainer already does this for Daphnet and MHEALTH.
  Our own recordings are still split by time within each file. Change that to
  a split by participant once several people have recorded.

## 2. Open-source tools

| Tool | What it gives StepCue | Notes |
| --- | --- | --- |
| [emlearn](https://github.com/emlearn/emlearn) (MIT) | Converts the scikit-learn Random Forest to plain C (`emlearn.convert(model).save(...)`) so it runs **on the ESP32 itself** | Tested on ESP32 / ESP32-S3. The docs page said RISC-V chips (ESP32-C3/C6) were not yet supported. Small trees (about 20–50) fit easily. Gives a fallback that keeps cueing without Wi-Fi |
| [aeon / sktime MiniRocket](https://www.sktime.net/) | Very fast, accurate time-series classifier to benchmark against our Random Forest | In one study MiniRocket performed best on inertial human-activity data. Cheap to try on 1.2 s windows |
| [TSFEL](https://www.csl.uni-bremen.de/cms/images/documents/publications/TSFEL_SoftwareX2020.pdf) / catch22 | Ready-made temporal, statistical and spectral feature sets | Good for finding features we are missing. catch22 is small enough for real-time use |
| [Label Studio](https://labelstud.io/templates/timeseries_audio_video) ≥ 1.20 | Label IMU data while watching synced video | Sync runs one way (time series → video). Record video at a constant frame rate. Our `sync_clap` marker lines video up with sensor time |
| [LIFT-PD](https://github.com/shovito66/LIFT-PD), [FOGSense](https://github.com/shovito66/FOGSense) | Reference code for single-sensor, real-time freeze detection | Read for ideas. I did not check their licenses |
| [Kaggle 8th-place 1D-CNN write-up](https://arxiv.org/pdf/2307.03475) | A simple deep-learning baseline for freeze detection from accelerometers | Lower-back data, but the method carries over |

Related systems with no public code, worth citing in the pitch as prior work:
[Wearable online FoG detection and cueing (Bioengineering 2024)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11505507/)
uses on-demand vibration at a 40 Hz processing rate.
[ParCuR (Sensors 2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12656022/)
is an ankle-worn IMU with haptic cueing.
[FM-FoG](https://arxiv.org/pdf/2509.24176) is a real-time foundation-model approach.
They show the idea is clinically motivated, and they frame what makes StepCue
different: the Nemotron agent that adapts the cue over time, plus the
caregiver layer.

## 3. Making our own dataset

### Phase A: acted data from the team (doable today)

Purpose: get sensor placement, strap tightness, Wi-Fi timing and labelling
right, and teach the model *our* sensor's real walking and standing signal.

```powershell
.venv\Scripts\python record_session.py --participant P01 --protocol
```

- The guided protocol (about 2.5 min) tells the wearer what to do and labels
  each segment automatically: standing, walking, and three acted freeze types
  (trembling, shuffling, akinesia), plus turns that are marked unlabelled so
  training ignores them.
- Free mode (no `--protocol`): someone watches and presses `w` / `s` / `f` /
  `u` while the wearer moves. `c` marks a clap for lining up video.
- Each session folder holds `imu.csv` (the training format), `events.jsonl`
  (what was asked and when) and `meta.json` (anonymous ID, sensor placement,
  measured sample rate, seconds per label).
- Target: 4–6 people × 2 sessions each, sensor on the same spot every time.
  That gives about 25–30 minutes of real-sensor data. Film each session on a
  phone so labels can be checked later.
- Retrain on everything:
  ```powershell
  .venv\Scripts\python train_classifier.py --recordings recordings --mhealth datasets\mhealth\MHEALTHDATASET --fogstar datasets\fogstar
  ```

**Be honest about the limits in the pitch.** Acted freezes are not real
freezing of gait. Real freezes often build up from shortening steps, happen at
doorways and turns, and vary a lot between people. Phase A data shows the
system works end to end on our hardware. FoG-STAR and the other public
datasets supply the real freeze examples.

### Phase B: real data from people with Parkinson's (after the hackathon)

- Requires **informed consent and ethics approval** (an IRB at a university),
  ideally run with a clinic or movement-disorder researchers. Don't record
  patients without it.
- Collect data that matches what published datasets record, so ours can be
  combined with theirs:
  - Ankle IMU at a fixed position.
  - Video for labelling.
  - Two independent raters marking freeze start and end, and freeze type.
  - Basic clinical context the person agrees to share (for example, a
    freezing questionnaire score).
- Common freeze triggers in these protocols are turns, narrow doorways,
  starting to walk, and walking while doing a second task.
- Privacy: anonymous IDs only, faces blurred in shared video, data kept off
  public repos unless consent covers sharing.

## 4. Results on real patient data (2026-11-21)

Both datasets are downloaded into `datasets/` (ignored by git). The FoG-STAR
license was checked on its Zenodo record: CC BY 4.0. Reproduce with
`.venv\Scripts\python benchmark_datasets.py` (about 90 s; features are cached).

**What FoG-STAR contains:**
- 22 people with Parkinson's, 91 minutes of recording, 17.9 minutes of it
  frozen, 196 freeze episodes (counting both ankles).
- Freeze time by type: **12.3 min akinesia (69%)**, 3.9 min trembling, 1.7 min
  shuffling.

**How the test works:** 5-fold cross-validation by person, so every score
comes from people the model never trained on. Both ankles count as separate
streams. Freeze episodes are cued using the same 3-window debounce as
`gait_classifier.py`.

| Trained on | Walking F1 | Standing F1 | Freeze F1 | Freeze episodes cued | Median onset → cue | False cues per hour* |
| --- | --- | --- | --- | --- | --- | --- |
| A synthetic only (old shipped model) | 0.18 | 0.67 | 0.28 | 64% | 2.5 s | 959 |
| B synthetic + MHEALTH | 0.25 | 0.65 | 0.29 | 69% | 2.3 s | 1249 |
| C FoG-STAR only | 0.93 | 0.75 | 0.64 | 79% | 1.7 s | 427 |
| D synthetic + MHEALTH + FoG-STAR | 0.93 | 0.74 | 0.65 | 78% | 1.6 s | 424 |
| **D with freeze threshold 0.65 (now shipped)** | | | | **79%** | **1.8 s** | **204** |

\*Counted during a lab protocol built to *provoke* freezes (turns, doorways,
dual tasks), for people with Parkinsonian gait. These rates are not
everyday-life rates. They are best used to compare one setting with another.

**What this means:**
1. **The simulator-trained model does not transfer to real people.** It
   labels most real Parkinsonian walking as freezing (walking F1 0.18).
   Training on real patient data is what fixes it.
2. **MHEALTH barely helps.** Healthy walking doesn't look like Parkinsonian
   walking. It is still useful as clean extra data for our own sensor.
3. **Akinetic freezes are the hard case.** They make up 69% of freeze time, and
   the confusion between standing and freezing accounts for most errors. This
   is a real limit of a single ankle sensor, not a bug. Possible fixes:
   - use the context that the person was just walking;
   - personalise to each user;
   - add the other ankle;
   - give the model more time context.
4. **Sensitivity is a setting.** `FREEZE_THRESHOLD` in `config.py` trades
   detection for false cues:

   | Threshold | Episodes cued | False cues per hour |
   | --- | --- | --- |
   | 0.45 | 78% | 305 |
   | **0.65** | **79%** | **204** |
   | 0.75 | 72% | 128 |

   A false cue costs a couple of seconds of rhythm while walking, which is low
   harm. A missed freeze is the worse failure.
5. Only cueing freezes that follow recent walking cut false cues 44% but
   missed about 10 more episodes. It is not enabled.

The shipped `gait_model.joblib` is now model D (17.6 MB). The simulator demo
still detects every simulated freeze with no false cues, and still sends the
cue 10–20 ms after the confirming packet. The old synthetic-only model is kept
at `datasets/cache/gait_model_synthetic_only.joblib`.

**Pitch-ready line:** "On 22 people with Parkinson's the model never saw, StepCue
cues 79% of real freeze episodes, a median 1.8 seconds after onset, using one
ankle sensor and no cloud."

## 5. Suggested next steps (in order)

1. Done: MHEALTH and FoG-STAR downloaded, loaders written, benchmark run.
2. Personalisation: fine-tune on a few minutes of each wearer's own walking
   and standing (from `record_session.py`) to cut false cues.
3. Record Phase A sessions as soon as the ankle node streams.
4. Retrain. Report accuracy on held-out *people*, for each source.
5. Optional stretch: emlearn on-device fallback.
