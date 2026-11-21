import com.stepcue.companion.Features;

/** Laptop-side checks for the phone gait math: javac + java, no Android needed. */
public class FeaturesTest {
    static int fails = 0;

    static void check(boolean ok, String what) {
        System.out.println((ok ? "ok   " : "FAIL ") + what);
        if (!ok) fails++;
    }

    /** Feed a synthetic signal: vertical accel = amp*sin(2 pi f t), with gravity on z and optional yaw. */
    static Features.Out run(Features f, double t0, double secs, double freq, double amp, double yawRad, double rate) {
        f.gravity(0, 0, 9.81);
        double t = t0;
        Features.Out o = null;
        for (int i = 0; i < secs * rate; i++) {
            t = t0 + i / rate;
            f.gyro(0, 0, yawRad);
            double jitter = (i % 7) * 0.0006;                       // slightly uneven sampling
            f.linear(t + jitter, 0.1 * Math.sin(3 * t), 0, amp * Math.sin(2 * Math.PI * freq * t));
        }
        return f.compute(t);
    }

    public static void main(String[] a) {
        Features f = new Features();
        Features.Out walk = run(f, 0, 6, 1.75, 2.5, 0, 50);           // 105 steps/min
        check("walking".equals(walk.motion), "steady 1.75 Hz stepping is walking (" + walk.motion + ")");
        check(Math.abs(walk.cadence - 105) <= 8, "cadence about 105/min (" + walk.cadence + ")");
        check(walk.freezeIndex < 1, "freeze index low while walking (" + walk.freezeIndex + ")");
        check(!walk.freezeSuspect, "no freeze while walking");

        Features.Out frz = run(f, 6, 5, 5.5, 1.6, 0, 50);             // trembling in place at 5.5 Hz
        check(frz.freezeIndex >= Features.FI_FREEZE, "freeze index high when trembling (" + frz.freezeIndex + ")");
        check(frz.freezeSuspect, "walking then trembling is a freeze suspect");

        Features g = new Features();
        Features.Out still = run(g, 0, 5, 1, 0.02, 0, 50);
        check("still".equals(still.motion), "tiny signal is still");
        check(!still.freezeSuspect, "standing still is not a freeze");

        Features h = new Features();
        Features.Out turn = run(h, 0, 4, 1.75, 2.5, Math.toRadians(60), 50);   // 60 deg/s yaw
        check(turn.turning && turn.turnDeg > 90, "turning detected (" + turn.turnDeg + " deg in 2 s)");

        check(Math.abs(Features.metresFromHpa(-0.36) - 3.0) < 0.1, "0.36 hPa drop is about 3 m up");
        System.out.println(fails == 0 ? "ALL PASSED" : fails + " FAILED");
        System.exit(fails == 0 ? 0 : 1);
    }
}
