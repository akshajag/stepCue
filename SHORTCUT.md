# The StepCue shortcut (free Apple Watch heart rate)

Build this **once** on any iPhone, share it as an iCloud link, and every wearer
gets it with one tap from the setup page (`companion/setup.html`). Wearers never
type anything: the setup page runs the shortcut and hands it the address of the
laptop, and the shortcut remembers it.

Why a shortcut: Apple Watch heart rate lives in the iPhone's Health app, and a
web page cannot read Health data. Apple's free Shortcuts app can, and can post
it to the laptop.

## Build it (about 10 minutes)

Open **Shortcuts** → **+**. Name it exactly **StepCue** (or set
`STEPCUE_SHORTCUT_NAME` to whatever you call it). Tap **ⓘ** (Details) and turn
on **Show in Share Sheet** off/on as you like; nothing else is needed there.

Add these actions in order:

1. **If** → *Shortcut Input* **has any value**
   - **Save File**: *Shortcut Input*, turn off **Ask Where to Save**,
     Subpath `StepCue/address.txt`, turn on **Overwrite If File Exists**.
   - **Otherwise**: leave empty. **End If**.
2. **Get File**: turn off **Show Document Picker**, File Path `StepCue/address.txt`.
3. **Get Text from Input** (input: *File* from step 2). Rename the result
   variable to **Address**.
4. **Find Health Samples** where **Type** is **Heart Rate**,
   Sort by **Start Date**, Order **Latest First**, Limit **1**.
5. **Get Details of Health Sample**: **Value** (of the samples from step 4).
   Rename to **HeartRate**.
6. **Find Health Samples** where **Type** is **Steps** and
   **Start Date** **is today**.
7. **Calculate Statistics**: **Sum** of the step samples. Rename to **Steps**.
8. **Get Contents of URL**: URL = **Address**. Tap the arrow:
   Method **POST**, Request Body **JSON**, add two **Number** fields:
   `heart_rate` = **HeartRate**, `steps` = **Steps**.
   (Keep it flat; no nested fields.)
9. Optional: **Show Notification** "Sent to StepCue".

Run it once yourself from the editor to grant Health access (Heart Rate and
Steps) and, if asked, local network access.

## Share it

Tap **Share** → **Copy iCloud Link**. Put the link in the laptop's `.env`:

```
STEPCUE_SHORTCUT_URL=https://www.icloud.com/shortcuts/xxxxxxxx
```

Restart `companion_server.py`. The setup page now shows **Get the shortcut**.

## What the wearer sees

1. Scan the QR code from the dashboard's Access page. The setup page opens.
2. Connection check, then "Do you wear an Apple Watch?".
3. **Get the shortcut** → Shortcuts opens → **Add Shortcut** → back to Safari.
4. **Connect** → the shortcut runs with this laptop's address, asks for Health
   access the first time, sends one reading, and returns to the setup page,
   which confirms "Watch connected, heart rate 72 bpm".
5. A helper adds a daily **Time of Day** automation (Run Immediately) a few
   times a day, or the wearer taps **Send my heart rate** in StepCue.

If the laptop's address changes (new Wi-Fi or hotspot), scan the new QR code
and tap **Connect** again; the shortcut overwrites the saved address.

## Limits

- Readings are snapshots each time the shortcut runs, not a live stream, and
  the watch may take a few minutes to sync to the phone.
- `Save File` and `Get File` use the Shortcuts folder in iCloud Drive. If
  iCloud Drive is off, turn it on, or replace steps 1–3 with a fixed URL text.
- After running from the Home Screen app, the Shortcuts app returns to Safari,
  not the Home Screen icon. That is an iOS limit.
