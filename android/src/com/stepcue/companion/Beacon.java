package com.stepcue.companion;

import android.content.Context;
import android.net.wifi.WifiManager;

import org.json.JSONObject;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;

/**
 * Listens for "a StepCue laptop is here" broadcasts on UDP 5011. Each laptop sends only a short hash
 * of its pairing key, so the phone can tell which saved laptop it is without the key ever crossing
 * the air. Used to follow a laptop whose address changed, and to switch to another saved laptop when
 * the current one has stopped answering.
 */
final class Beacon {
    static final int PORT = 5011;
    static final long SWITCH_AFTER_MS = 10000;

    private static int users = 0;
    private static Thread thread;
    private static volatile DatagramSocket socket;
    private static WifiManager.MulticastLock lock;

    /** Start listening (counted: the screen and the background service can both hold it). */
    static synchronized void acquire(Context ctx) {
        users++;
        if (thread != null) return;
        final Context c = ctx.getApplicationContext();
        try {
            WifiManager wm = (WifiManager) c.getSystemService(Context.WIFI_SERVICE);
            if (wm != null) {
                lock = wm.createMulticastLock("stepcue-beacon");
                lock.setReferenceCounted(false);
                lock.acquire();
            }
        } catch (Exception ignored) { }
        thread = new Thread(() -> listen(c), "stepcue-beacon");
        thread.setDaemon(true);
        thread.start();
    }

    static synchronized void release() {
        if (users > 0) users--;
        if (users > 0 || thread == null) return;
        Thread t = thread;
        thread = null;
        DatagramSocket s = socket;
        if (s != null) s.close();
        t.interrupt();
        try { if (lock != null && lock.isHeld()) lock.release(); } catch (Exception ignored) { }
        lock = null;
    }

    private static void listen(Context c) {
        byte[] buf = new byte[512];
        while (!Thread.currentThread().isInterrupted()) {
            try (DatagramSocket s = new DatagramSocket(null)) {
                s.setReuseAddress(true);
                s.setBroadcast(true);
                s.setSoTimeout(5000);
                s.bind(new InetSocketAddress(PORT));
                socket = s;
                while (!Thread.currentThread().isInterrupted()) {
                    DatagramPacket p = new DatagramPacket(buf, buf.length);
                    try {
                        s.receive(p);
                    } catch (SocketTimeoutException e) {
                        continue;
                    }
                    String text = new String(p.getData(), 0, p.getLength(), StandardCharsets.UTF_8);
                    handle(c, text, p.getAddress().getHostAddress());
                }
            } catch (Exception e) {
                if (Thread.currentThread().isInterrupted()) return;
                try { Thread.sleep(3000); } catch (InterruptedException ie) { return; }
            } finally {
                socket = null;
            }
        }
    }

    static void handle(Context c, String text, String fromIp) {
        try {
            JSONObject j = new JSONObject(text);
            if (j.optInt("stepcue") != 1) return;
            String id = j.optString("id", "");
            int port = j.optInt("port", 8080);
            if (id.isEmpty() || fromIp == null || !fromIp.matches("[0-9.]{7,15}")) return;
            Net.seen(c, id, fromIp, port, j.optString("name", ""));
        } catch (Exception ignored) { }
    }
}
