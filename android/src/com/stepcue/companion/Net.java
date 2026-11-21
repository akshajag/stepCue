package com.stepcue.companion;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/** Talks to the StepCue laptop over the local Wi-Fi (plain HTTP, the pairing key in a header). */
final class Net {
    static final String PREFS = "stepcue";

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
        return text.isEmpty() ? new JSONObject() : new JSONObject(text);
    }
}
