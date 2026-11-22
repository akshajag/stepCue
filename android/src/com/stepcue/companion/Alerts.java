package com.stepcue.companion;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.telecom.TelecomManager;

import org.json.JSONObject;

/**
 * Calls the caretaker from this phone over its own plan (no paid service).
 *
 * The laptop sends the caretaker's number in /api/status ("alerts"); it is cached here so a help tap
 * still calls when the laptop cannot be reached. Triggers: the wearer's help tap, a freeze longer than
 * freeze_sec, a help tap made elsewhere (web page or dashboard), and a test call from the dashboard.
 */
final class Alerts {
    static final String[] PERMISSIONS = {Manifest.permission.CALL_PHONE};

    /** What happened last, for the screen ("Calling your caretaker" / "Could not call"). */
    static volatile String lastResult = null;
    static volatile long lastResultMs = 0;

    private static long freezeSinceMs = 0;     // local count, in case the laptop gives no freeze_for

    private Alerts() { }

    static SharedPreferences prefs(Context c) {
        return Net.prefs(c);
    }

    static boolean ready(Context c) {
        return prefs(c).getString("alert_phone", null) != null;
    }

    static boolean permitted(Context c) {
        return c.checkSelfPermission(Manifest.permission.CALL_PHONE) == PackageManager.PERMISSION_GRANTED;
    }

    /** Called with each /api/status reply. `st` is the effective status (FREEZE includes the personal model). */
    static void onStatus(Context c, JSONObject s, String st) {
        JSONObject a = s.optJSONObject("alerts");
        SharedPreferences p = prefs(c);
        if (a == null) {                                   // number removed or calls turned off
            if (p.contains("alert_phone")) p.edit().remove("alert_phone").apply();
            freezeSinceMs = 0;
            return;
        }
        p.edit().putString("alert_phone", a.optString("phone"))
                .putLong("alert_cooldown_ms", a.optLong("cooldown", 300) * 1000L).apply();

        long now = System.currentTimeMillis();
        // a test call asked for on the dashboard (first sight after install only records the counter)
        int testId = a.optInt("test_id", 0);
        if (!p.contains("alert_test_seen")) {
            p.edit().putInt("alert_test_seen", testId).apply();
        } else if (testId > p.getInt("alert_test_seen", 0)) {
            p.edit().putInt("alert_test_seen", testId).apply();
            fire(c, "test");
        }
        // a help tap made somewhere else (the web page or the patient dashboard)
        double helpT = a.optDouble("help_t", Double.NaN);
        if (!Double.isNaN(helpT)) {
            long helpMs = (long) (helpT * 1000);
            if (helpMs > p.getLong("alert_help_seen", 0)) {
                p.edit().putLong("alert_help_seen", helpMs).apply();
                fire(c, "help");                           // cooldown skips it if this phone just called
            }
        }
        // a freeze that has lasted too long: once per episode
        if ("FREEZE".equals(st)) {
            if (freezeSinceMs == 0) freezeSinceMs = now;
            double limit = a.optDouble("freeze_sec", 30);
            double laptopFor = a.optDouble("freeze_for", Double.NaN);
            double localFor = (now - freezeSinceMs) / 1000.0;
            double lasted = Double.isNaN(laptopFor) ? localFor : Math.max(laptopFor, localFor);
            String episode = a.has("freeze_id") && !a.isNull("freeze_id")
                    ? "e" + a.optInt("freeze_id") : "t" + freezeSinceMs;
            if (lasted >= limit && !episode.equals(p.getString("alert_freeze_seen", ""))) {
                p.edit().putString("alert_freeze_seen", episode).apply();
                fire(c, "freeze");
            }
        } else {
            freezeSinceMs = 0;
        }
    }

    /** Call the caretaker on a background thread. */
    static void fire(Context c, String kind) {
        final Context app = c.getApplicationContext();
        new Thread(() -> call(app, kind), "stepcue-call").start();
    }

    private static void call(Context c, String kind) {
        SharedPreferences p = prefs(c);
        String phone = p.getString("alert_phone", null);
        if (phone == null) {
            result("No caretaker number yet. Add it on the laptop's Access page.");
            return;
        }
        long now = System.currentTimeMillis();
        long cooldown = p.getLong("alert_cooldown_ms", 300000L);
        if (!"test".equals(kind)) {
            synchronized (Alerts.class) {
                long last = p.getLong("alert_last_ms_" + kind, 0);
                if (now - last < cooldown) return;         // one of each kind per few minutes, never a flood
                p.edit().putLong("alert_last_ms_" + kind, now).apply();
            }
        }
        boolean called = false;
        String error = null;
        if (permitted(c)) {
            try {
                TelecomManager tm = (TelecomManager) c.getSystemService(Context.TELECOM_SERVICE);
                tm.placeCall(Uri.fromParts("tel", phone, null), new Bundle());
                called = true;
            } catch (SecurityException ex) {
                error = "call failed";
            }
        } else {
            error = "calling not allowed";
        }
        result(called ? "Calling your caretaker." : "Could not call. Allow phone calls for StepCue in Settings.");
        try {
            JSONObject j = new JSONObject();
            j.put("kind", kind);
            j.put("call", called);
            if (error != null) j.put("error", error);
            Net.post(c, "/api/alertlog", j);
        } catch (Exception ignored) { }                     // the laptop may be out of reach; that is fine
    }

    private static void result(String words) {
        lastResult = words;
        lastResultMs = System.currentTimeMillis();
    }
}
