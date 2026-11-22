package com.stepcue.companion;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/** Talks to the StepCue laptop over the local Wi-Fi (plain HTTP, the pairing key in a header). */
final class Net {
    static final String PREFS = "stepcue";
    static final int MAX_LAPTOPS = 8;

    /** When the active laptop last answered, and when it became the active one. */
    static volatile long lastAnswerMs = 0, activeSinceMs = 0;

    static SharedPreferences prefs(Context c) {
        return c.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    static boolean paired(Context c) {
        SharedPreferences p = prefs(c);
        return p.getString("host", null) != null && p.getString("key", null) != null;
    }

    static String base(Context c) {
        SharedPreferences p = prefs(c);
        return "http://" + p.getString("host", "") + ":" + p.getInt("port", 8080);
    }

    // ------------------------------------------------------- saved laptops --
    /** Public id of a pairing key: same as laptop_id() in companion_store.py. */
    static String idFor(String key) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-256").digest(("stepcue:" + key).getBytes(StandardCharsets.UTF_8));
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < 8; i++) b.append(String.format("%02x", d[i] & 0xff));
            return b.toString();
        } catch (Exception e) {
            return "";
        }
    }

    static String activeId(Context c) {
        String k = prefs(c).getString("key", null);
        return k == null ? "" : idFor(k);
    }

    static String activeName(Context c) {
        return prefs(c).getString("name", "");
    }

    /** Every laptop this phone has been paired with, newest first. */
    static synchronized JSONArray laptops(Context c) {
        SharedPreferences p = prefs(c);
        JSONArray a;
        try { a = new JSONArray(p.getString("laptops", "[]")); } catch (Exception e) { a = new JSONArray(); }
        if (a.length() == 0 && paired(c)) {         // paired before this list existed
            a.put(entry(p.getString("host", ""), p.getInt("port", 8080), p.getString("key", ""), p.getString("name", "")));
            p.edit().putString("laptops", a.toString()).apply();
        }
        return a;
    }

    private static JSONObject entry(String host, int port, String key, String name) {
        JSONObject j = new JSONObject();
        try {
            j.put("host", host).put("port", port).put("key", key).put("id", idFor(key))
             .put("name", name == null || name.isEmpty() ? "StepCue laptop" : name)
             .put("last", System.currentTimeMillis());
        } catch (Exception ignored) { }
        return j;
    }

    /** Add (or update) a laptop from a scanned QR code and make it the active one. */
    static synchronized void saveLaptop(Context c, String host, int port, String key, String name) {
        JSONArray old = laptops(c), out = new JSONArray();
        JSONObject e = entry(host, port, key, name);
        out.put(e);
        String id = e.optString("id");
        for (int i = 0; i < old.length() && out.length() < MAX_LAPTOPS; i++) {
            JSONObject o = old.optJSONObject(i);
            if (o != null && !id.equals(o.optString("id"))) out.put(o);
        }
        prefs(c).edit().putString("laptops", out.toString()).apply();
        activate(c, e);
    }

    /** Make a saved laptop the one the app talks to. */
    static synchronized void activate(Context c, JSONObject e) {
        prefs(c).edit().putString("host", e.optString("host")).putInt("port", e.optInt("port", 8080))
                .putString("key", e.optString("key")).putString("name", e.optString("name")).apply();
        activeSinceMs = System.currentTimeMillis();
        lastAnswerMs = 0;
        SensorService.lastStatus = null;
        SensorService.lastError = "";
    }

    static void activate(Context c, String id) {
        JSONArray a = laptops(c);
        for (int i = 0; i < a.length(); i++) {
            JSONObject o = a.optJSONObject(i);
            if (o != null && id.equals(o.optString("id"))) { activate(c, o); return; }
        }
    }

    /** A beacon from a laptop: follow its new address, or switch to it if ours went quiet. */
    static synchronized void seen(Context c, String id, String ip, int port, String name) {
        JSONArray a = laptops(c);
        JSONObject hit = null;
        for (int i = 0; i < a.length(); i++) {
            JSONObject o = a.optJSONObject(i);
            if (o != null && id.equals(o.optString("id"))) { hit = o; break; }
        }
        if (hit == null) return;                    // a laptop we were never paired with: ignore
        try {
            hit.put("host", ip).put("port", port).put("seen", System.currentTimeMillis());
        } catch (Exception ignored) { }
        prefs(c).edit().putString("laptops", a.toString()).apply();
        SharedPreferences p = prefs(c);
        long now = System.currentTimeMillis();
        if (id.equals(activeId(c))) {
            if (!ip.equals(p.getString("host", "")) || port != p.getInt("port", 8080)) {
                p.edit().putString("host", ip).putInt("port", port).apply();    // laptop got a new address
            }
        } else if (now - Math.max(lastAnswerMs, activeSinceMs) > Beacon.SWITCH_AFTER_MS) {
            activate(c, hit);                       // ours hasn't answered for 10 s: use this one
        }
    }

    static JSONObject get(Context c, String path) throws Exception {
        return call(c, "GET", path, null);
    }

    static JSONObject post(Context c, String path, JSONObject body) throws Exception {
        return call(c, "POST", path, body);
    }

    private static JSONObject call(Context c, String method, String path, JSONObject body) throws Exception {
        HttpURLConnection h = (HttpURLConnection) new URL(base(c) + path).openConnection();
        h.setConnectTimeout(4000);
        h.setReadTimeout(6000);
        h.setRequestMethod(method);
        h.setRequestProperty("X-StepCue-Key", prefs(c).getString("key", ""));
        if (body != null) {
            byte[] data = body.toString().getBytes(StandardCharsets.UTF_8);
            h.setDoOutput(true);
            h.setRequestProperty("Content-Type", "application/json");
            h.setFixedLengthStreamingMode(data.length);
            try (OutputStream o = h.getOutputStream()) { o.write(data); }
        }
        int code = h.getResponseCode();
        InputStream in = code < 400 ? h.getInputStream() : h.getErrorStream();
        String text = "";
        if (in != null) {
            ByteArrayOutputStream b = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            int r;
            while ((r = in.read(chunk)) > 0) b.write(chunk, 0, r);
            in.close();
            text = b.toString("UTF-8");
        }
        h.disconnect();
        if (code == 401) throw new Exception("pairing");
        if (code >= 400) throw new Exception("HTTP " + code);
        lastAnswerMs = System.currentTimeMillis();
        return text.isEmpty() ? new JSONObject() : new JSONObject(text);
    }
}
