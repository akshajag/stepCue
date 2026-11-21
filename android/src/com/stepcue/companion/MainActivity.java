package com.stepcue.companion;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * The wearer's screen. Buttons only: pairing arrives by a link from the laptop's QR code, then one
 * "Start" tap asks for the permissions and starts the background sensor service.
 */
public class MainActivity extends Activity {
    // calm when OK, loud only when not: teal = ok, amber = watch, red = alert
    static final int BG = 0xFFF6F4EF, CARD = 0xFFFFFFFF, INK = 0xFF1C2430, MUTED = 0xFF5B6573,
            TEAL = 0xFF0F766E, TEAL_BG = 0xFFE3F2EF, AMBER = 0xFF9A4D06, AMBER_BG = 0xFFFDF0DC,
            RED = 0xFFB42318, RED_BG = 0xFFFDE7E5, GREY = 0xFF6B7280, GREY_BG = 0xFFEDEEF0, LINE = 0xFFE3E0D8;

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private LinearLayout root;
    private TextView statusBig, statusSub, msg, senseText, freezeQ, pairText;
    private LinearLayout statusCard, freezeCard, pairCard, helpCard, feelCard, senseCard;
    private Button startBtn, okBtn, helpBtn, buzzBtn, stopBtn;
    private boolean helpArmed = false;
    private int pendingId = -1;
    private boolean visible = false;
    // training mode
    private LinearLayout trainCard, patientRow;
    private TextView trainHead, trainInfo, phaseText;
    private Button trainToggle, stopRecBtn, buildBtn;
    private final Map<String, Button> labelBtns = new LinkedHashMap<>();
    private boolean trainOpen = false;
    private String patientsKey = "";
    static final String[][] LABELS = {{"sitting", "Sitting"}, {"standing", "Standing"},
            {"walking", "Walking"}, {"frozen", "Frozen"}};

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().setStatusBarColor(BG);
        if (Build.VERSION.SDK_INT >= 23) getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        build();
        handlePair(getIntent());
    }

    @Override
    protected void onNewIntent(Intent i) {
        super.onNewIntent(i);
        setIntent(i);
        handlePair(i);
    }

    @Override protected void onResume() { super.onResume(); visible = true; tick.run(); }
    @Override protected void onPause() { super.onPause(); visible = false; ui.removeCallbacks(tick); }

    // ------------------------------------------------------------- pairing --
    private void handlePair(Intent i) {
        Uri u = i == null ? null : i.getData();
        if (u != null && "stepcue".equals(u.getScheme()) && "pair".equals(u.getHost())) {
            String h = u.getQueryParameter("h"), k = u.getQueryParameter("k"), p = u.getQueryParameter("p");
            if (h != null && k != null && h.matches("[A-Za-z0-9.\\-]{1,253}") && k.matches("[A-Za-z0-9_\\-]{8,128}")) {
                int port = 8080;
                try { if (p != null) port = Integer.parseInt(p); } catch (NumberFormatException ignored) { }
                Net.prefs(this).edit().putString("host", h).putInt("port", port).putString("key", k).apply();
                say("Connected to your StepCue laptop. Tap Start.", TEAL);
            } else {
                say("That link didn't look right. Scan the code on the laptop again.", RED);
            }
        }
        render();
        if (Net.paired(this) && !SensorService.running && Net.prefs(this).getBoolean("running", false)) {
            startFlow();     // it was running before (e.g. app updated): pick up again
        }
    }

    // ----------------------------------------------------------- start flow --
    private void startFlow() {
        List<String> want = new ArrayList<>();
        want.add(Manifest.permission.ACCESS_FINE_LOCATION);
        want.add(Manifest.permission.ACCESS_COARSE_LOCATION);
        if (Build.VERSION.SDK_INT >= 29) want.add(Manifest.permission.ACTIVITY_RECOGNITION);
        if (Build.VERSION.SDK_INT >= 33) want.add(Manifest.permission.POST_NOTIFICATIONS);
        List<String> missing = new ArrayList<>();
        for (String p : want) if (checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) missing.add(p);
        if (!missing.isEmpty() && !Net.prefs(this).getBoolean("asked", false)) {
            Net.prefs(this).edit().putBoolean("asked", true).apply();
            requestPermissions(missing.toArray(new String[0]), 1);
            return;
        }
        afterPermissions();
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        afterPermissions();
    }

    private void afterPermissions() {
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        SharedPreferences sp = Net.prefs(this);
        if (!pm.isIgnoringBatteryOptimizations(getPackageName()) && !sp.getBoolean("askedBattery", false)) {
            sp.edit().putBoolean("askedBattery", true).apply();
            try {
                startActivity(new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                        Uri.parse("package:" + getPackageName())));
            } catch (Exception ignored) { }
        }
        Intent s = new Intent(this, SensorService.class);
        try {
            startForegroundService(s);
            sp.edit().putBoolean("running", true).apply();
            say("StepCue is on. You can put the phone in your pocket.", TEAL);
        } catch (Exception e) {
            say("Couldn't start: " + e.getMessage(), RED);
        }
        ui.postDelayed(this::render, 600);
    }

    private void stopService() {
        startService(new Intent(this, SensorService.class).setAction("stop"));
        Net.prefs(this).edit().putBoolean("running", false).apply();
        say("Sharing stopped.", GREY);
        ui.postDelayed(this::render, 400);
    }

    // ------------------------------------------------------------------ UI --
    private int dp(float v) {
        return Math.round(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics()));
    }

    private GradientDrawable round(int fill, int stroke) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(fill);
        g.setCornerRadius(dp(18));
        if (stroke != 0) g.setStroke(dp(2), stroke);
        return g;
    }

    private TextView text(String s, float sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        t.setLineSpacing(0, 1.15f);
        if (bold) t.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        return t;
    }

    private Button button(String s, int fill, int ink, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(s);
        b.setAllCaps(false);
        b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 22);
        b.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        b.setTextColor(ink);
        b.setBackground(round(fill, 0));
        b.setMinHeight(dp(72));
        b.setStateListAnimator(null);
        b.setOnClickListener(l);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(72));
        lp.topMargin = dp(10);
        b.setLayoutParams(lp);
        return b;
    }

    private LinearLayout card() {
        LinearLayout c = new LinearLayout(this);
        c.setOrientation(LinearLayout.VERTICAL);
        c.setBackground(round(CARD, LINE));
        c.setPadding(dp(20), dp(18), dp(20), dp(20));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(14);
        c.setLayoutParams(lp);
        root.addView(c);
        return c;
    }

    private LinearLayout row(LinearLayout parent) {
        LinearLayout r = new LinearLayout(this);
        r.setOrientation(LinearLayout.HORIZONTAL);
        parent.addView(r);
        return r;
    }

    private Button half(LinearLayout r, String s, int fill, int ink, View.OnClickListener l, boolean first) {
        Button b = button(s, fill, ink, l);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(72), 1f);
        lp.topMargin = dp(10);
        if (!first) lp.leftMargin = dp(10);
        b.setLayoutParams(lp);
        r.addView(b);
        return b;
    }

    private void build() {
        ScrollView sv = new ScrollView(this);
        sv.setBackgroundColor(BG);
        sv.setFillViewport(true);
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(20), dp(18), dp(40));
        sv.addView(root);
        setContentView(sv);

        TextView title = text("StepCue", 30, INK, true);
        root.addView(title);
        root.addView(text("Walking help, with your care team", 18, MUTED, false));

        msg = text("", 19, TEAL, true);
        msg.setVisibility(View.GONE);
        msg.setPadding(0, dp(10), 0, 0);
        root.addView(msg);

        // not paired yet
        pairCard = card();
        pairCard.addView(text("Connect to your laptop", 24, INK, true));
        pairText = text("On the StepCue laptop, open the Android QR code. Point this phone's camera at it and tap the link.\n\nThen tap \"Open StepCue\" on the page that opens.", 19, INK, false);
        pairText.setPadding(0, dp(8), 0, 0);
        pairCard.addView(pairText);

        // status
        statusCard = card();
        statusBig = text("Getting ready", 32, INK, true);
        statusSub = text("", 18, MUTED, false);
        statusSub.setPadding(0, dp(4), 0, 0);
        statusCard.addView(statusBig);
        statusCard.addView(statusSub);
        phaseText = text("", 18, TEAL, true);
        phaseText.setPadding(0, dp(6), 0, 0);
        phaseText.setVisibility(View.GONE);
        statusCard.addView(phaseText);
        startBtn = button("Start", TEAL, Color.WHITE, v -> startFlow());
        statusCard.addView(startBtn);

        // freeze confirmation
        freezeCard = card();
        freezeQ = text("Did you just freeze?", 24, INK, true);
        freezeCard.addView(freezeQ);
        freezeCard.addView(text("Your answer helps the care team know the alerts are right.", 18, MUTED, false));
        LinearLayout fr = row(freezeCard);
        half(fr, "Yes", AMBER_BG, AMBER, v -> confirm(true), true);
        half(fr, "No", GREY_BG, INK, v -> confirm(false), false);

        // ok / help
        helpCard = card();
        helpCard.addView(text("How are you?", 24, INK, true));
        okBtn = button("I'm OK", TEAL, Color.WHITE, v -> checkin("ok", null, "Sent: you're OK."));
        helpBtn = button("I need help", RED_BG, RED, v -> help());
        helpCard.addView(okBtn);
        helpCard.addView(helpBtn);

        // walking feel
        feelCard = card();
        feelCard.addView(text("How is walking today?", 24, INK, true));
        Button easy = button("Easy", TEAL_BG, TEAL, v -> checkin("walking", "easy", "Thanks. Saved: easy."));
        Button some = button("Some trouble", AMBER_BG, AMBER, v -> checkin("walking", "some trouble", "Thanks. Saved: some trouble."));
        Button hard = button("Hard", RED_BG, RED, v -> checkin("walking", "hard", "Thanks. Saved: hard."));
        feelCard.addView(easy);
        feelCard.addView(some);
        feelCard.addView(hard);

        // cue setting
        LinearLayout cueCard = card();
        cueCard.addView(text("Walking beat", 24, INK, true));
        cueCard.addView(text("If you freeze, the phone buzzes a steady beat and says \"Step with the beat\".", 18, MUTED, false));
        buzzBtn = button("", GREY_BG, INK, v -> {
            SharedPreferences sp = Net.prefs(this);
            sp.edit().putBoolean("buzz", !sp.getBoolean("buzz", true)).apply();
            render();
        });
        cueCard.addView(buzzBtn);

        // training mode (a helper teaches StepCue this person's sitting, standing, walking and freezing)
        trainToggle = button("Teach StepCue (training)", BG, INK, v -> { trainOpen = !trainOpen; render(); });
        trainToggle.setBackground(round(BG, LINE));
        trainToggle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 19);
        root.addView(trainToggle);
        trainCard = card();
        trainHead = text("Teach StepCue", 24, INK, true);
        trainCard.addView(trainHead);
        trainCard.addView(text("Tap what the person is doing right now and leave it on while they do it, about a minute each. "
                + "When they freeze, tap Frozen (a helper can tap it for them).", 18, MUTED, false));
        patientRow = new LinearLayout(this);
        patientRow.setOrientation(LinearLayout.VERTICAL);
        trainCard.addView(patientRow);
        for (String[] lab : LABELS) {
            final String key = lab[0];
            Button b = button(lab[1], GREY_BG, INK, v -> trainPost("label", key, null, null));
            labelBtns.put(key, b);
            trainCard.addView(b);
        }
        stopRecBtn = button("Stop recording", BG, INK, v -> trainPost("stop", null, null, "Stopped."));
        stopRecBtn.setBackground(round(BG, LINE));
        trainCard.addView(stopRecBtn);
        buildBtn = button("Build the model", INK, Color.WHITE, v -> {
            say("Building... this takes a few seconds.", INK);
            trainPost("train", null, null, null);
        });
        trainCard.addView(buildBtn);
        trainInfo = text("", 17, MUTED, false);
        trainInfo.setPadding(0, dp(10), 0, 0);
        trainCard.addView(trainInfo);

        // what the phone senses
        senseCard = card();
        senseCard.addView(text("What the phone senses", 22, INK, true));
        senseText = text("", 17, MUTED, false);
        senseText.setPadding(0, dp(6), 0, 0);
        senseCard.addView(senseText);

        stopBtn = button("Stop sharing", BG, MUTED, v -> stopService());
        stopBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        stopBtn.setBackground(round(BG, LINE));
        root.addView(stopBtn);

        TextView foot = text("StepCue helps with walking cues and keeps your care team aware. It does not diagnose or give medical advice.", 14, MUTED, false);
        foot.setPadding(0, dp(18), 0, 0);
        foot.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(foot);
    }

    private void say(String s, int color) {
        if (msg == null) return;
        msg.setText(s);
        msg.setTextColor(color);
        msg.setVisibility(View.VISIBLE);
        ui.removeCallbacks(hideMsg);
        ui.postDelayed(hideMsg, 8000);
    }

    private final Runnable hideMsg = () -> msg.setVisibility(View.GONE);

    // ------------------------------------------------------------- actions --
    private void checkin(String kind, String value, String done) {
        if (!Net.paired(this)) return;
        io.execute(() -> {
            try {
                JSONObject j = new JSONObject();
                j.put("kind", kind);
                if (value != null) j.put("value", value);
                Net.post(this, "/api/checkin", j);
                ui.post(() -> say(done, "help".equals(kind) ? RED : TEAL));
            } catch (Exception e) {
                ui.post(() -> say("Not sent. Is the phone on the same Wi-Fi as the laptop?", RED));
            }
        });
    }

    private void help() {
        if (!helpArmed) {
            helpArmed = true;
            helpBtn.setText("Tap again to call for help");
            helpBtn.setBackground(round(RED, 0));
            helpBtn.setTextColor(Color.WHITE);
            ui.postDelayed(disarm, 6000);
            return;
        }
        disarm.run();
        checkin("help", null, "Help is on the way. Your care team has been told.");
    }

    private final Runnable disarm = () -> {
        helpArmed = false;
        helpBtn.setText("I need help");
        helpBtn.setBackground(round(RED_BG, 0));
        helpBtn.setTextColor(RED);
    };

    private void confirm(boolean froze) {
        final int id = pendingId;
        if (id < 0) return;
        freezeCard.setVisibility(View.GONE);
        io.execute(() -> {
            try {
                JSONObject j = new JSONObject();
                j.put("episode_id", id);
                j.put("froze", froze);
                Net.post(this, "/api/confirm", j);
                ui.post(() -> say("Thanks, saved.", TEAL));
            } catch (Exception e) {
                ui.post(() -> say("Not sent. Try again in a moment.", RED));
            }
        });
    }

    private void trainPost(String action, String label, String patient, String done) {
        if (!Net.paired(this)) return;
        io.execute(() -> {
            try {
                JSONObject j = new JSONObject();
                j.put("action", action);
                if (label != null) j.put("label", label);
                if (patient != null) j.put("id", patient);
                JSONObject r = Net.post(this, "/api/train", j);
                JSONObject st = SensorService.lastStatus;
                if (st != null && r.has("training")) st.put("training", r.getJSONObject("training"));
                String msg = done;
                if ("train".equals(action)) {
                    int pct = (int) Math.round(r.optDouble("accuracy", 0) * 100);
                    msg = "Model built from " + r.optInt("n") + " seconds. Right " + pct
                            + "% of the time on data it hadn't seen"
                            + (r.optBoolean("uses_ankle") ? " (phone + ankle sensor)." : " (phone only).");
                } else if ("label".equals(action)) {
                    msg = "Recording: " + word(label) + ". Keep going for about a minute.";
                }
                final String m = msg;
                ui.post(() -> { if (m != null) say(m, TEAL); render(); });
            } catch (Exception e) {
                String why = e.getMessage() == null ? "" : e.getMessage();
                ui.post(() -> say("train".equals(action) ? "Not enough recorded yet. Record at least two activities, "
                        + "10 seconds or more each." : "Not sent. Is the phone on the same Wi-Fi? " + why, RED));
            }
        });
    }

    // ------------------------------------------------------------- refresh --
    private final Runnable tick = new Runnable() {
        @Override public void run() {
            render();
            if (Net.paired(MainActivity.this) && !SensorService.running) {
                io.execute(() -> {          // service off: ask the laptop ourselves
                    try {
                        SensorService.lastStatus = Net.get(MainActivity.this, "/api/status");
                        SensorService.lastError = "";
                    } catch (Exception e) {
                        SensorService.lastError = e.getMessage() == null ? "no connection" : e.getMessage();
                        SensorService.lastStatus = null;
                    }
                });
            }
            if (visible) ui.postDelayed(this, 1500);
        }
    };

    private void render() {
        boolean paired = Net.paired(this);
        boolean on = SensorService.running;
        pairCard.setVisibility(paired ? View.GONE : View.VISIBLE);
        for (View v : new View[]{statusCard, helpCard, feelCard, senseCard, trainToggle})
            v.setVisibility(paired ? View.VISIBLE : View.GONE);
        trainCard.setVisibility(paired && trainOpen ? View.VISIBLE : View.GONE);
        trainToggle.setText(trainOpen ? "Close training" : "Teach StepCue (training)");
        startBtn.setVisibility(paired && !on ? View.VISIBLE : View.GONE);
        stopBtn.setVisibility(on ? View.VISIBLE : View.GONE);
        boolean buzz = Net.prefs(this).getBoolean("buzz", true);
        buzzBtn.setText(buzz ? "Beat is on  (tap to turn off)" : "Beat is off  (tap to turn on)");
        buzzBtn.setBackground(round(buzz ? TEAL_BG : GREY_BG, 0));
        buzzBtn.setTextColor(buzz ? TEAL : INK);
        if (!paired) { freezeCard.setVisibility(View.GONE); return; }

        JSONObject st = SensorService.lastStatus;
        String err = SensorService.lastError;
        boolean pairingBad = "pairing".equals(err);
        boolean linkOk = st != null && (err == null || err.isEmpty());
        String status = st == null ? "" : st.optString("status", "");
        int fg = INK, bg = CARD;
        String big, sub;
        if (pairingBad) {
            big = "Needs connecting again"; fg = AMBER; bg = AMBER_BG;
            sub = "The laptop has a new code. Scan the Android QR code on the laptop again.";
        } else if (!linkOk) {
            big = "Can't reach the laptop"; fg = AMBER; bg = AMBER_BG;
            sub = "Check that this phone is on the same Wi-Fi as the StepCue laptop.";
        } else if ("FREEZE".equals(status)) {
            big = "Freeze detected"; fg = RED; bg = RED_BG;
            sub = buzz ? "Step with the beat. Big steps." : "Take a breath. Big steps.";
        } else if ("WALKING".equals(status)) {
            big = "Walking"; fg = TEAL; bg = TEAL_BG; sub = "All good.";
        } else if ("OFFLINE".equals(status)) {
            big = "Ankle sensor not connected"; fg = GREY; bg = GREY_BG;
            sub = "The phone still shares its own motion.";
        } else {
            big = "Standing"; fg = TEAL; bg = TEAL_BG; sub = "All good.";
        }
        if (!on && !pairingBad) sub = "Tap Start so the phone can help. " + (linkOk ? "Connected to your laptop." : "");
        else if (on) {
            long ago = SensorService.lastOkMs == 0 ? -1 : (System.currentTimeMillis() - SensorService.lastOkMs) / 1000;
            sub += ago < 0 ? "\nPhone sensors: starting…" : ago < 10 ? "\nPhone sensors: sharing" : "\nPhone sensors: last shared " + ago + " s ago";
        }
        statusBig.setText(big);
        statusBig.setTextColor(fg);
        statusSub.setText(sub);
        statusCard.setBackground(round(bg, bg == CARD ? LINE : 0));

        // freeze confirmation
        JSONArray pend = st == null ? null : st.optJSONArray("pending");
        if (pend != null && pend.length() > 0) {
            JSONObject e = pend.optJSONObject(pend.length() - 1);
            pendingId = e.optInt("id", -1);
            freezeQ.setText("Did you just freeze" + (e.optString("time").isEmpty() ? "" : " at " + e.optString("time")) + "?");
            freezeCard.setVisibility(pendingId >= 0 ? View.VISIBLE : View.GONE);
        } else {
            pendingId = -1;
            freezeCard.setVisibility(View.GONE);
        }

        renderTraining(st);

        // live sensor readout
        JSONObject j = SensorService.lastSent;
        if (!on || j == null) {
            senseText.setText(on ? "Starting the sensors…" : "Nothing yet. Tap Start.");
        } else {
            StringBuilder s = new StringBuilder();
            s.append("Motion: ").append(word(j.optString("motion"))).append('\n');
            if (j.optDouble("cadence", 0) > 0) s.append("Steps per minute: ").append(j.optInt("cadence")).append('\n');
            s.append("Steps in last 10 s: ").append(j.optInt("steps_10s")).append('\n');
            if (j.has("steps_session")) s.append("Steps since start: ").append(j.optInt("steps_session")).append('\n');
            s.append("Turning: ").append(j.optBoolean("turning") ? "yes (" + j.optInt("turn_deg") + "°)" : "no").append('\n');
            s.append("Freeze index: ").append(j.optDouble("freeze_index")).append(j.optBoolean("freeze_suspect") ? "  (looks like a freeze)" : "").append('\n');
            if (j.has("stairs")) s.append("Stairs: ").append(j.optString("stairs")).append('\n');
            if (j.has("light_lux")) s.append("Light: ").append(j.optInt("light_lux")).append(" lux").append(j.optBoolean("in_pocket") ? " (in pocket)" : "").append('\n');
            if (j.has("battery")) s.append("Battery: ").append(j.optInt("battery")).append('%').append(j.optBoolean("charging") ? " (charging)" : "").append('\n');
            s.append("GPS: ").append(j.has("lat") ? "on (stays on your laptop)" : "waiting").append('\n');
            s.append("Sensors: ").append(SensorService.sensorNames.size()).append(" · ").append(j.optInt("hz")).append(" readings a second");
            senseText.setText(s.toString());
        }
    }

    private void renderTraining(JSONObject st) {
        JSONObject tr = st == null ? null : st.optJSONObject("training");
        JSONObject phase = st == null ? null : st.optJSONObject("phase");
        if (phase != null && (tr == null || tr.isNull("label"))) {
            phaseText.setText("StepCue thinks: " + phase.optString("word") + "  ("
                    + Math.round(phase.optDouble("conf") * 100) + "% sure, their own model)");
            phaseText.setTextColor("frozen".equals(phase.optString("label")) ? RED : TEAL);
            phaseText.setVisibility(View.VISIBLE);
        } else {
            phaseText.setVisibility(View.GONE);
        }
        if (!trainOpen) return;
        if (tr == null || tr.isNull("patient")) {
            trainHead.setText("Teach StepCue");
            trainInfo.setText("First add the person on the laptop: dashboard, Training page, Add patient.");
            for (Button b : labelBtns.values()) b.setEnabled(false);
            buildBtn.setEnabled(false);
            stopRecBtn.setVisibility(View.GONE);
            return;
        }
        buildBtn.setEnabled(true);
        JSONObject pat = tr.optJSONObject("patient");
        trainHead.setText("Teach StepCue: " + pat.optString("name"));
        JSONArray pats = tr.optJSONArray("patients");
        String key = (pats == null ? "" : pats.toString()) + pat.optString("id");
        if (!key.equals(patientsKey)) {          // person switch buttons, only when there is more than one
            patientsKey = key;
            patientRow.removeAllViews();
            if (pats != null && pats.length() > 1) {
                for (int i = 0; i < pats.length(); i++) {
                    JSONObject p = pats.optJSONObject(i);
                    final String id = p.optString("id");
                    boolean on = id.equals(pat.optString("id"));
                    Button b = button((on ? "\u2713 " : "") + p.optString("name"), on ? TEAL_BG : BG, on ? TEAL : INK,
                            v -> trainPost("patient", null, id, "Switched person."));
                    b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
                    if (!on) b.setBackground(round(BG, LINE));
                    patientRow.addView(b);
                }
            }
        }
        String active = tr.isNull("label") ? null : tr.optString("label");
        JSONObject counts = tr.optJSONObject("counts");
        int goal = tr.optInt("goal", 60);
        long since = (long) (tr.optDouble("since", 0) * 1000);
        for (String[] lab : LABELS) {
            Button b = labelBtns.get(lab[0]);
            b.setEnabled(true);
            int n = counts == null ? 0 : counts.optInt(lab[0]);
            boolean on = lab[0].equals(active);
            if (on) {
                long secs = Math.max(0, (System.currentTimeMillis() - since) / 1000);
                b.setText("\u25CF Recording " + lab[1] + "  " + secs + " s");
                b.setBackground(round("frozen".equals(lab[0]) ? RED : TEAL, 0));
                b.setTextColor(Color.WHITE);
            } else {
                b.setText(lab[1] + "  \u00B7  " + n + " / " + goal + " s" + (n >= goal ? "  \u2713" : ""));
                b.setBackground(round(n >= goal ? TEAL_BG : GREY_BG, 0));
                b.setTextColor(n >= goal ? TEAL : INK);
            }
        }
        stopRecBtn.setVisibility(active != null ? View.VISIBLE : View.GONE);
        JSONObject m = tr.optJSONObject("model");
        StringBuilder info = new StringBuilder();
        if (m != null) {
            info.append("Model ready: right ").append(Math.round(m.optDouble("accuracy") * 100))
                .append("% of the time on data it hadn't seen, from ").append(m.optInt("n")).append(" seconds.");
            JSONObject pace = m.optJSONObject("pace");
            if (pace != null) info.append("\nTheir usual walking pace: ").append(pace.optInt("low")).append("\u2013")
                .append(pace.optInt("high")).append(" steps a minute.");
            info.append("\nRecord more any time, then build again.");
        } else {
            info.append("No model yet. Record each activity, then tap Build the model.");
        }
        trainInfo.setText(info.toString());
    }

    static String word(String m) {
        switch (m == null ? "" : m) {
            case "walking": return "walking";
            case "moving": return "moving";
            case "sitting": return "sitting";
            case "standing": return "standing";
            case "frozen": return "frozen";
            default: return "still";
        }
    }

    @Override
    protected void onDestroy() {
        io.shutdownNow();
        super.onDestroy();
    }
}
