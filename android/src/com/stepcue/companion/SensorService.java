package com.stepcue.companion;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.content.pm.ServiceInfo;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Runs in the background (foreground service with a notification) so the phone keeps sensing in a
 * pocket with the screen off. Every 2 s it sends a feature summary to the laptop; every 1.5 s it
 * asks the laptop for the status, and while the ankle sensor reports a freeze it vibrates the phone
 * on the cue beat and speaks a short cue once.
 */
public class SensorService extends Service implements SensorEventListener, LocationListener {
    static final String CHANNEL = "stepcue";
    static final int NOTE_ID = 7;
    static final long SEND_MS = 2000, STATUS_MS = 1500;

    // read by MainActivity for the live view
    static volatile boolean running = false;
    static volatile JSONObject lastSent = null;
    static volatile long lastOkMs = 0;
    static volatile String lastError = "";
    static volatile int sentCount = 0;
    static volatile JSONObject lastStatus = null;
    static volatile List<String> sensorNames = new ArrayList<>();

    private SensorManager sm;
    private LocationManager lm;
    private HandlerThread thread;
    private Handler h;
    private PowerManager.WakeLock wake;
    private Vibrator vib;
    private TextToSpeech tts;
    private boolean ttsReady = false;

    private final Features feat = new Features();
    private final float[] acc = new float[3], grav = {0, 0, 9.81f}, mag = new float[3], rot = new float[5];
    private boolean haveLinear = false, haveRot = false, haveMag = false, haveAcc = false;
    private float pressure = Float.NaN, light = Float.NaN, proxDist = Float.NaN, proxMax = 5f;
    private final ArrayDeque<double[]> pressHist = new ArrayDeque<>();   // [t, hPa]
    private final ArrayDeque<Double> stepTimes = new ArrayDeque<>();
    private float stepCounterStart = -1, stepCounter = -1;
    private Location loc;
    private final ArrayDeque<JSONObject> backlog = new ArrayDeque<>();
    private String cueStatus = "";
    private int cueBpm = 0;

    @Override public IBinder onBind(Intent i) { return null; }

    @Override
    public void onCreate() {
        super.onCreate();
        sm = (SensorManager) getSystemService(SENSOR_SERVICE);
        lm = (LocationManager) getSystemService(LOCATION_SERVICE);
        vib = (Vibrator) getSystemService(VIBRATOR_SERVICE);
        thread = new HandlerThread("stepcue-sensors");
        thread.start();
        h = new Handler(thread.getLooper());
        tts = new TextToSpeech(this, s -> {
            ttsReady = s == TextToSpeech.SUCCESS;
            if (ttsReady) tts.setLanguage(Locale.getDefault());
        });
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && "stop".equals(intent.getAction())) {
            Net.prefs(this).edit().putBoolean("running", false).apply();
            stopSelf();
            return START_NOT_STICKY;
        }
        goForeground();
        if (!running) {
            running = true;
            Net.prefs(this).edit().putBoolean("running", true).apply();
            PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
            wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "StepCue:sensors");
            wake.acquire();
            registerSensors();
            startLocation();
            h.postDelayed(sendLoop, SEND_MS);
            h.postDelayed(statusLoop, 500);
        }
        return START_STICKY;
    }

    private void goForeground() {
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "StepCue walking help", NotificationManager.IMPORTANCE_LOW));
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        Notification n = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(android.R.drawable.ic_menu_mylocation)
                .setContentTitle("StepCue is helping you walk")
                .setContentText("Sharing motion with your care team")
                .setContentIntent(open)
                .setOngoing(true)
                .build();
        if (Build.VERSION.SDK_INT >= 34) {
            int type = ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH;
            if (hasLocation()) type |= ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION;
            try {
                startForeground(NOTE_ID, n, type);
            } catch (RuntimeException e) {
                startForeground(NOTE_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH);
            }
        } else if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTE_ID, n, hasLocation() ? ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION : 0);
        } else {
            startForeground(NOTE_ID, n);
        }
    }

    private boolean hasLocation() {
        return checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
                || checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED;
    }

    private void registerSensors() {
        int[] types = {Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_LINEAR_ACCELERATION, Sensor.TYPE_GRAVITY,
                Sensor.TYPE_GYROSCOPE, Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_MAGNETIC_FIELD, Sensor.TYPE_PRESSURE,
                Sensor.TYPE_LIGHT, Sensor.TYPE_PROXIMITY, Sensor.TYPE_STEP_DETECTOR, Sensor.TYPE_STEP_COUNTER,
                Sensor.TYPE_GAME_ROTATION_VECTOR};
        List<String> names = new ArrayList<>();
        for (int t : types) {
            Sensor s = sm.getDefaultSensor(t);
            if (s == null) continue;
            if (t == Sensor.TYPE_STEP_DETECTOR || t == Sensor.TYPE_STEP_COUNTER) {
                if (Build.VERSION.SDK_INT >= 29 && checkSelfPermission(Manifest.permission.ACTIVITY_RECOGNITION)
                        != PackageManager.PERMISSION_GRANTED) continue;
            }
            int rate = (t == Sensor.TYPE_PRESSURE || t == Sensor.TYPE_LIGHT || t == Sensor.TYPE_PROXIMITY)
                    ? SensorManager.SENSOR_DELAY_NORMAL : 20000;           // 50 Hz for motion sensors
            if (sm.registerListener(this, s, rate, h)) names.add(label(t));
            if (t == Sensor.TYPE_PROXIMITY) proxMax = s.getMaximumRange();
        }
        sensorNames = names;
    }

    static String label(int t) {
        switch (t) {
            case Sensor.TYPE_ACCELEROMETER: return "accelerometer";
            case Sensor.TYPE_LINEAR_ACCELERATION: return "linear acceleration";
            case Sensor.TYPE_GRAVITY: return "gravity";
            case Sensor.TYPE_GYROSCOPE: return "gyroscope";
            case Sensor.TYPE_ROTATION_VECTOR: return "rotation";
            case Sensor.TYPE_GAME_ROTATION_VECTOR: return "game rotation";
            case Sensor.TYPE_MAGNETIC_FIELD: return "compass";
            case Sensor.TYPE_PRESSURE: return "barometer";
            case Sensor.TYPE_LIGHT: return "light";
            case Sensor.TYPE_PROXIMITY: return "proximity";
            case Sensor.TYPE_STEP_DETECTOR: return "step detector";
            case Sensor.TYPE_STEP_COUNTER: return "step counter";
            default: return "sensor " + t;
        }
    }

    private void startLocation() {
        if (!hasLocation()) return;
        try {
            for (String p : new String[]{LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER}) {
                if (lm.getAllProviders().contains(p)) lm.requestLocationUpdates(p, 10000, 3f, this, thread.getLooper());
            }
        } catch (SecurityException ignored) { }
    }

    // ------------------------------------------------------------ sensors --
    @Override
    public void onSensorChanged(SensorEvent e) {
        double t = e.timestamp / 1e9;
        switch (e.sensor.getType()) {
            case Sensor.TYPE_ACCELEROMETER:
                haveAcc = true;
                System.arraycopy(e.values, 0, acc, 0, 3);
                if (!haveLinear) {                     // no linear-acceleration sensor: low-pass gravity
                    for (int i = 0; i < 3; i++) grav[i] = 0.9f * grav[i] + 0.1f * acc[i];
                    feat.gravity(grav[0], grav[1], grav[2]);
                    feat.linear(t, acc[0] - grav[0], acc[1] - grav[1], acc[2] - grav[2]);
                }
                break;
            case Sensor.TYPE_LINEAR_ACCELERATION:
                haveLinear = true;
                feat.linear(t, e.values[0], e.values[1], e.values[2]);
                break;
            case Sensor.TYPE_GRAVITY:
                System.arraycopy(e.values, 0, grav, 0, 3);
                feat.gravity(e.values[0], e.values[1], e.values[2]);
                break;
            case Sensor.TYPE_GYROSCOPE:
                feat.gyro(e.values[0], e.values[1], e.values[2]);
                break;
            case Sensor.TYPE_ROTATION_VECTOR:
                haveRot = true;
                System.arraycopy(e.values, 0, rot, 0, Math.min(5, e.values.length));
                break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                haveMag = true;
                System.arraycopy(e.values, 0, mag, 0, 3);
                break;
            case Sensor.TYPE_PRESSURE:
                pressure = e.values[0];
                pressHist.addLast(new double[]{t, pressure});
                while (!pressHist.isEmpty() && t - pressHist.peekFirst()[0] > 12) pressHist.removeFirst();
                break;
            case Sensor.TYPE_LIGHT:
                light = e.values[0];
                break;
            case Sensor.TYPE_PROXIMITY:
                proxDist = e.values[0];
                break;
            case Sensor.TYPE_STEP_DETECTOR:
                stepTimes.addLast(t);
                while (!stepTimes.isEmpty() && t - stepTimes.peekFirst() > 10) stepTimes.removeFirst();
                break;
            case Sensor.TYPE_STEP_COUNTER:
                stepCounter = e.values[0];
                if (stepCounterStart < 0) stepCounterStart = stepCounter;
                break;
            default:
                break;
        }
    }

    @Override public void onAccuracyChanged(Sensor s, int a) { }

    @Override public void onLocationChanged(Location l) {
        if (loc == null || l.getAccuracy() <= loc.getAccuracy() + 20 || l.getTime() - loc.getTime() > 30000) loc = l;
    }
    @Override public void onProviderEnabled(String p) { }
    @Override public void onProviderDisabled(String p) { }
    @Override public void onStatusChanged(String p, int s, Bundle b) { }

    // --------------------------------------------------------- send loop --
    private final Runnable sendLoop = new Runnable() {
        @Override public void run() {
            try {
                JSONObject j = summary();
                lastSent = j;
                backlog.addLast(j);
                while (backlog.size() > 30) backlog.removeFirst();
                while (!backlog.isEmpty()) {
                    Net.post(SensorService.this, "/api/android", backlog.peekFirst());
                    backlog.removeFirst();
                    sentCount++;
                    lastOkMs = System.currentTimeMillis();
                    lastError = "";
                }
            } catch (Exception ex) {
                lastError = ex.getMessage() == null ? ex.getClass().getSimpleName() : ex.getMessage();
            }
            if (running) h.postDelayed(this, training() ? 1000 : SEND_MS);   // 1 a second while teaching
        }
    };

    /** True while a helper is recording a label for the patient's own model. */
    static boolean training() {
        JSONObject s = lastStatus;
        JSONObject tr = s == null ? null : s.optJSONObject("training");
        return tr != null && !tr.isNull("label");
    }

    private JSONObject summary() throws Exception {
        double now = SystemClock.elapsedRealtimeNanos() / 1e9;
        Features.Out o = feat.compute(now);
        JSONObject j = new JSONObject();
        j.put("t", System.currentTimeMillis() / 1000.0);
        j.put("motion", o.motion);
        j.put("cadence", round(o.cadence, 0));
        j.put("acc_rms", round(o.accRms, 3));
        j.put("gyro_rms", round(o.gyroRms, 3));
        j.put("vert_rms", round(o.vertRms, 3));
        j.put("yaw_rate", round(o.yawRate, 3));
        j.put("turn_deg", round(o.turnDeg, 0));
        j.put("turning", o.turning);
        j.put("freeze_index", round(Math.min(o.freezeIndex, 50), 2));
        j.put("freeze_suspect", o.freezeSuspect);
        j.put("hz", round(o.hz, 0));
        j.put("samples", o.samples);

        int steps10 = 0;
        for (double st : stepTimes) if (now - st <= 10) steps10++;
        j.put("steps_10s", steps10);
        if (steps10 >= 3) j.put("cadence_steps", steps10 * 6);
        if (stepCounter >= 0) j.put("steps_session", (int) (stepCounter - stepCounterStart));

        if (!Float.isNaN(pressure)) {
            j.put("pressure_hpa", round(pressure, 2));
            if (pressHist.size() > 2) {
                double dm = Features.metresFromHpa(pressure - pressHist.peekFirst()[1]);
                j.put("alt_change_m", round(dm, 1));
                j.put("stairs", dm > 2.0 ? "up" : dm < -2.0 ? "down" : "level");
            }
        }
        if (!Float.isNaN(light)) j.put("light_lux", round(light, 0));
        boolean near = !Float.isNaN(proxDist) && proxDist < proxMax;
        if (!Float.isNaN(proxDist)) j.put("proximity_near", near);
        if (!Float.isNaN(proxDist) || !Float.isNaN(light)) j.put("in_pocket", near && (Float.isNaN(light) || light < 15));

        float[] r = new float[9], ori = new float[3];
        if (haveRot) {
            SensorManager.getRotationMatrixFromVector(r, rot);
            SensorManager.getOrientation(r, ori);
            j.put("heading", round((Math.toDegrees(ori[0]) + 360) % 360, 0));
            j.put("pitch", round(Math.toDegrees(ori[1]), 0));
            j.put("roll", round(Math.toDegrees(ori[2]), 0));
        } else if (haveMag && haveAcc && SensorManager.getRotationMatrix(r, null, grav, mag)) {
            SensorManager.getOrientation(r, ori);
            j.put("heading", round((Math.toDegrees(ori[0]) + 360) % 360, 0));
        }

        if (loc != null && System.currentTimeMillis() - loc.getTime() < 60000) {
            j.put("lat", loc.getLatitude());
            j.put("lon", loc.getLongitude());
            j.put("acc_m", round(loc.getAccuracy(), 0));
            if (loc.hasSpeed()) j.put("speed_ms", round(loc.getSpeed(), 2));
        }
        Intent b = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (b != null) {
            int lvl = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1), sc = b.getIntExtra(BatteryManager.EXTRA_SCALE, 100);
            if (lvl >= 0) j.put("battery", Math.round(100f * lvl / sc));
            int plug = b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0);
            j.put("charging", plug != 0);
        }
        JSONArray names = new JSONArray();
        for (String s : sensorNames) names.put(s);
        j.put("sensors", names);
        j.put("device", Build.MANUFACTURER + " " + Build.MODEL);
        j.put("android", Build.VERSION.SDK_INT);
        return j;
    }

    static double round(double v, int d) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return 0;
        double p = Math.pow(10, d);
        return Math.round(v * p) / p;
    }

    // ---------------------------------------------- cue from the laptop --
    private final Runnable statusLoop = new Runnable() {
        @Override public void run() {
            try {
                JSONObject s = Net.get(SensorService.this, "/api/status");
                lastStatus = s;
                String st = s.optString("status", "");
                JSONObject cue = s.optJSONObject("cue");
                int bpm = cue != null ? cue.optInt("bpm", 100) : 100;
                boolean want = Net.prefs(SensorService.this).getBoolean("buzz", true);
                JSONObject phase = s.optJSONObject("phase");
                boolean personal = phase != null && "frozen".equals(phase.optString("label"))
                        && phase.optDouble("conf", 0) >= 0.7;      // the patient's own model sees a freeze
                if (personal && !"FREEZE".equals(st)) st = "FREEZE";
                if ("FREEZE".equals(st) && want) {
                    if (!"FREEZE".equals(cueStatus) || bpm != cueBpm) startBuzz(bpm, !"FREEZE".equals(cueStatus));
                } else if ("FREEZE".equals(cueStatus)) {
                    stopBuzz();
                    if ("WALKING".equals(st) && want) say("You're walking again. Well done.");
                }
                cueStatus = st;
                cueBpm = bpm;
            } catch (Exception ignored) { }
            if (running) h.postDelayed(this, STATUS_MS);
        }
    };

    private void startBuzz(int bpm, boolean speak) {
        if (speak) say("Let's walk together. Step with the beat.");
        if (vib == null || !vib.hasVibrator()) return;
        long beat = Math.max(300, Math.round(60000.0 / Math.max(40, bpm)));
        long on = 110;
        long[] pattern = {0, on, beat - on};
        vib.vibrate(VibrationEffect.createWaveform(pattern, 1));
    }

    private void stopBuzz() {
        if (vib != null) vib.cancel();
    }

    private void say(String text) {
        if (ttsReady) tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "stepcue");
    }

    @Override
    public void onDestroy() {
        running = false;
        h.removeCallbacksAndMessages(null);
        try { sm.unregisterListener(this); } catch (Exception ignored) { }
        try { lm.removeUpdates(this); } catch (Exception ignored) { }
        stopBuzz();
        if (tts != null) tts.shutdown();
        if (wake != null && wake.isHeld()) wake.release();
        thread.quitSafely();
        super.onDestroy();
    }
}
