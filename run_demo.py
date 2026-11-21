"""Start the whole StepCue stack with one command.

  python run_demo.py              # simulator + classifier + advisor + location + companion + dashboard
  python run_demo.py --hardware   # real ESP32: no simulator (join StepCue_AP Wi-Fi first)
  python run_demo.py --bluetooth  # real ESP32 over Bluetooth: no simulator, starts esp_bt_bridge.py
  python run_demo.py --same-window  # keep all logs in this terminal

By default each part opens in its own console window on Windows, so you can
press w / s / f / h in the simulator window. Ctrl+C here stops everything.
"""
import argparse
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", action="store_true", help="do not start the simulator")
    ap.add_argument("--bluetooth", action="store_true",
                    help="ESP32 connected over Bluetooth: start esp_bt_bridge.py instead of the simulator")
    ap.add_argument("--same-window", action="store_true")
    ap.add_argument("--no-dashboard", action="store_true")
    a = ap.parse_args()

    py = sys.executable
    parts = [("classifier", [py, "-u", "gait_classifier.py"]),
             ("advisor", [py, "-u", "nemotron_advisor.py"]),
             ("alerts", [py, "-u", "alert_notifier.py"])]   # sends only if .env has ntfy/Twilio keys
    if a.bluetooth:
        a.hardware = True
        parts.append(("bluetooth", [py, "-u", "esp_bt_bridge.py"]))
    if not a.hardware:
        parts.append(("simulator", [py, "-u", "sim_esp32.py"]))
    # Outdoor location: a simulated walk in the demo, the wearer's phone (OwnTracks) on hardware.
    parts.append(("location", [py, "-u", "location_server.py"] + ([] if a.hardware else ["--simulate"])))
    # Wearer's phone page (Safari) and watch data intake: https://<laptop-ip>:8443, http://<laptop-ip>:8080
    parts.append(("companion", [py, "-u", "companion_server.py"]))
    if not a.no_dashboard:
        parts.append(("dashboard", [py, "-m", "streamlit", "run", "caretaker_app.py"]))

    flags = 0
    if sys.platform == "win32" and not a.same_window:
        flags = subprocess.CREATE_NEW_CONSOLE
    procs = []
    try:
        for name, cmd in parts:
            procs.append((name, subprocess.Popen(cmd, creationflags=flags)))
            print(f"started {name}")
            time.sleep(1.0 if name == "classifier" else 0.3)
        print("\nStepCue is running. Dashboard: http://localhost:8501   (Ctrl+C to stop)")
        while True:
            time.sleep(1)
            for name, p in procs:
                if p.poll() is not None:
                    print(f"{name} exited with code {p.returncode}")
                    procs.remove((name, p))
                    break
            if not procs:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for name, p in procs:
            p.terminate()
        print("stopped")


if __name__ == "__main__":
    main()
