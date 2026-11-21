package com.stepcue.companion;

import java.util.ArrayDeque;
import java.util.Iterator;

/**
 * Gait features from the phone's motion sensors. Pure Java (no Android imports) so it can be
 * unit-tested on a laptop (android/test/FeaturesTest.java).
 *
 * Inputs are linear acceleration (m/s^2, gravity removed), gravity (m/s^2) and gyroscope (rad/s)
 * samples with timestamps in seconds. Every window we compute:
 *   accRms, gyroRms        overall movement and rotation
 *   vertical accel         linear acceleration projected on gravity, whatever way the phone is held
 *   yawRate, turnDeg       rotation about the vertical axis: turns (a common freeze trigger)
 *   freezeIndex            Moore's Freeze Index: power 3-8 Hz / power 0.5-3 Hz of vertical accel.
 *                          Trembling in place ("freeze band") vs. steady stepping ("locomotor band")
 *   cadence                steps/min from the dominant locomotor frequency (fallback when the
 *                          step detector sensor is missing)
 *   motion                 still / moving / walking
 *   freezeSuspect          was walking a moment ago, now not stepping, trembling (high FI)
 * Non-uniform sample times are handled by evaluating the Fourier sum at each time stamp.
 */
public final class Features {
    public static final double WINDOW_SEC = 4.0;
    public static final double TURN_DEG = 40.0;
    public static final double FI_FREEZE = 2.0;
    public static final double STILL_RMS = 0.25;      // m/s^2
    public static final double WALK_RMS = 0.9;

    static final class S { double t, ax, ay, az, gx, gy, gz, wx, wy, wz; }

    private final ArrayDeque<S> buf = new ArrayDeque<>();
    private double gx = 0, gy = 0, gz = 9.81;           // latest gravity
    private double wx, wy, wz;                          // latest gyro
    private double lastWalkT = -1e9;

    public void gravity(double x, double y, double z) { gx = x; gy = y; gz = z; }
    public void gyro(double x, double y, double z) { wx = x; wy = y; wz = z; }

    public void linear(double t, double x, double y, double z) {
        S s = new S();
        s.t = t; s.ax = x; s.ay = y; s.az = z; s.gx = gx; s.gy = gy; s.gz = gz; s.wx = wx; s.wy = wy; s.wz = wz;
        buf.addLast(s);
        while (!buf.isEmpty() && t - buf.peekFirst().t > WINDOW_SEC) buf.removeFirst();
    }

    public int size() { return buf.size(); }

    /** Result of one window. */
    public static final class Out {
        public double accRms, gyroRms, vertRms, yawRate, turnDeg, freezeIndex, locoPower, freezePower, cadence, hz;
        public String motion = "still";
        public boolean turning, freezeSuspect;
        public int samples;
    }

    public Out compute(double now) {
        Out o = new Out();
        int n = buf.size();
        o.samples = n;
        if (n < 20) return o;
        double[] t = new double[n], v = new double[n];
        double sa = 0, sw = 0, yawInt = 0, yawSum = 0, prevT = Double.NaN;
        int i = 0;
        for (Iterator<S> it = buf.iterator(); it.hasNext(); i++) {
            S s = it.next();
            double gn = Math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz);
            if (gn < 1e-6) gn = 1;
            double ux = s.gx / gn, uy = s.gy / gn, uz = s.gz / gn;
            t[i] = s.t;
            v[i] = s.ax * ux + s.ay * uy + s.az * uz;
            sa += s.ax * s.ax + s.ay * s.ay + s.az * s.az;
            sw += s.wx * s.wx + s.wy * s.wy + s.wz * s.wz;
            double yaw = s.wx * ux + s.wy * uy + s.wz * uz;
            yawSum += yaw;
            if (!Double.isNaN(prevT)) {
                double dt = s.t - prevT;
                if (dt > 0 && dt < 0.2) yawInt += yaw * dt;
            }
            prevT = s.t;
        }
        double span = t[n - 1] - t[0];
        o.hz = span > 0 ? (n - 1) / span : 0;
        o.accRms = Math.sqrt(sa / n);
        o.gyroRms = Math.sqrt(sw / n);
        o.yawRate = yawSum / n;
        // turning over the last 2 s only
        double turn2 = 0; prevT = Double.NaN;
        for (S s : buf) {
            if (now - s.t > 2.0) { prevT = s.t; continue; }
            double gn = Math.max(1e-6, Math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz));
            double yaw = (s.wx * s.gx + s.wy * s.gy + s.wz * s.gz) / gn;
            if (!Double.isNaN(prevT)) { double dt = s.t - prevT; if (dt > 0 && dt < 0.2) turn2 += yaw * dt; }
            prevT = s.t;
        }
        o.turnDeg = Math.toDegrees(Math.abs(turn2));
        o.turning = o.turnDeg >= TURN_DEG;

        double mean = 0;
        for (double x : v) mean += x;
        mean /= n;
        double vr = 0;
        for (int k = 0; k < n; k++) { v[k] -= mean; vr += v[k] * v[k]; }
        o.vertRms = Math.sqrt(vr / n);

        // band powers with a non-uniform Fourier sum, 0.25 Hz steps
        double loco = 0, frz = 0, best = 0, bestF = 0;
        for (double f = 0.5; f <= 8.0001; f += 0.25) {
            double re = 0, im = 0, w = 2 * Math.PI * f;
            for (int k = 0; k < n; k++) { re += v[k] * Math.cos(w * t[k]); im -= v[k] * Math.sin(w * t[k]); }
            double p = (re * re + im * im) / ((double) n * n);
            if (f < 3.0) { loco += p; if (p > best) { best = p; bestF = f; } }
            else frz += p;
        }
        o.locoPower = loco;
        o.freezePower = frz;
        o.freezeIndex = loco > 1e-9 ? frz / loco : (frz > 1e-6 ? 10 : 0);

        double cad = bestF * 60.0;
        boolean stepping = o.accRms >= WALK_RMS && cad >= 50 && cad <= 150 && loco > frz;
        o.cadence = stepping ? cad : 0;
        if (o.accRms < STILL_RMS) o.motion = "still";
        else if (stepping) o.motion = "walking";
        else o.motion = "moving";
        if ("walking".equals(o.motion)) lastWalkT = now;

        o.freezeSuspect = !"walking".equals(o.motion) && now - lastWalkT <= 6.0
                && o.freezeIndex >= FI_FREEZE && o.accRms >= STILL_RMS;
        return o;
    }

    /** Altitude change (m) from a pressure change (hPa), standard atmosphere near sea level. */
    public static double metresFromHpa(double dHpa) { return -8.43 * dHpa; }
}
