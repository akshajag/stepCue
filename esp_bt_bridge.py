"""Bridge the ESP32 ankle sensor over Bluetooth into the normal StepCue pipeline.

The ESP32 talks to this laptop over Bluetooth instead of Wi-Fi. The data is the same CSV line as
the locked Wi-Fi contract (`ts_ms,ax,ay,az,gx,gy,gz` at 50 Hz, accel in g, gyro in deg/s), and the
cues are the same text (`CUE:<bpm>:<haptic>` or `STOP`). This bridge:

  ESP32 --Bluetooth--> bridge --UDP 127.0.0.1:5005--> gait_classifier.py
  ESP32 <--Bluetooth-- bridge <--UDP 127.0.0.1:5006-- gait_classifier.py (cues go back to the sender)

so gait_classifier.py, training and the dashboards work unchanged.

Two Bluetooth kinds:
  Classic (Serial Port Profile), the default. Pair "StepCue_Ankle" once in Windows Settings >
  Bluetooth; Windows makes a COM port for it. Then:
      python esp_bt_bridge.py               (finds the Bluetooth COM port by itself)
      python esp_bt_bridge.py --port COM7
  BLE (Nordic UART Service), for ESP32-S3/C3 boards without Classic Bluetooth:
      pip install bleak
      python esp_bt_bridge.py --ble StepCue_Ankle

pyserial is used for COM ports when installed (pip install pyserial); without it the port is
opened directly through Windows.
"""
import argparse
import asyncio
import re
import socket
import subprocess
import sys
import threading
import time

from config import CUE_PORT, SENSOR_PORT

NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"      # laptop -> ESP32 (write)
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"      # ESP32 -> laptop (notify)
CSV = re.compile(rb"^-?\d+(\.\d+)?(,-?\d+(\.\d+)?([eE]-?\d+)?){6}$")


def log(msg):
    print(f"[bt {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Forwarder:
    """Splits the byte stream into lines, forwards valid sensor lines over UDP, counts them."""

    def __init__(self, target=("127.0.0.1", SENSOR_PORT), cue_port=CUE_PORT):
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = target
        self.cue = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cue.bind(("127.0.0.1", cue_port))           # gait_classifier answers to the sender's address
        self.cue.settimeout(0.5)
        self.buf = b""
        self.lines = 0
        self.bad = 0

    def feed(self, data):
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        if len(self.buf) > 512:                           # garbage without newlines
            self.buf = b""
        sent = 0
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if CSV.match(line):
                # send from the cue socket so the classifier's cues come back to us
                self.cue.sendto(line + b"\n", self.target)
                self.lines += 1
                sent += 1
            else:
                self.bad += 1
        return sent

    def next_cue(self):
        try:
            data, _ = self.cue.recvfrom(256)
        except (socket.timeout, OSError):
            return None
        text = data.decode("ascii", "ignore").strip()
        return text if text == "STOP" or re.fullmatch(r"CUE:\d{2,3}:[01]", text) else None


# ------------------------------------------------------------ classic --
def bluetooth_com_ports():
    """COM ports Windows made for paired Bluetooth devices (Serial Port Profile)."""
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()
                if "bluetooth" in (p.description or "").lower() or "BTHENUM" in (p.hwid or "").upper()]
    except ImportError:
        pass
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match '\\(COM\\d+\\)' "
                              "-and ($_.Name -match 'Bluetooth' -or $_.PNPDeviceID -match 'BTHENUM') } | "
                              "ForEach-Object { $_.Name }"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return re.findall(r"\((COM\d+)\)", out)


class ComPort:
    def __init__(self, name):
        self.name = name
        try:
            import serial
            self.s = serial.Serial(name, 115200, timeout=0.5, write_timeout=1)
            self.read = lambda: self.s.read(self.s.in_waiting or 1)
            self.write = self.s.write
        except ImportError:                                # raw Windows device, no pyserial
            self.s = open(r"\\.\%s" % name, "r+b", buffering=0)
            self.read = lambda: self.s.read(64)
            self.write = self.s.write

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def probe(name, seconds=6):
    """Open a COM port and wait for sensor lines. Returns the open port or None."""
    result = {}

    def go():
        try:
            port = ComPort(name)
        except OSError as exc:
            result["err"] = exc
            return
        result["port"] = port
        buf, t0 = b"", time.time()
        while time.time() - t0 < seconds:
            try:
                buf += port.read() or b""
            except OSError:
                break
            if any(CSV.match(x.strip()) for x in buf.split(b"\n")[:-1]):
                result["ok"] = True
                return

    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(seconds + 1)
    if result.get("ok"):
        return result["port"]
    if result.get("port"):
        result["port"].close()
    return None


def run_classic(fwd, port_name):
    while True:
        names = [port_name] if port_name and port_name != "auto" else bluetooth_com_ports()
        port = None
        for n in names:
            log(f"trying {n}...")
            port = probe(n)
            if port:
                break
        if not port:
            log("no sensor data on " + (", ".join(names) or "any Bluetooth COM port")
                + ". Is the ankle sensor on and paired (Settings > Bluetooth > StepCue_Ankle)? Retrying in 5 s")
            time.sleep(5)
            continue
        log(f"receiving ankle data on {port.name}")
        stop = threading.Event()

        def cues():
            while not stop.is_set():
                c = fwd.next_cue()
                if c:
                    try:
                        port.write((c + "\n").encode("ascii"))
                    except OSError:
                        stop.set()

        threading.Thread(target=cues, daemon=True).start()
        last = time.time()
        while not stop.is_set():
            try:
                data = port.read()
            except OSError:
                break
            if data:
                fwd.feed(data)
                last = time.time()
            elif time.time() - last > 3:
                break
        stop.set()
        port.close()
        log("Bluetooth link lost, reconnecting")


# ---------------------------------------------------------------- BLE --
async def run_ble(fwd, name):
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError:
        sys.exit("BLE needs bleak: pip install bleak")
    while True:
        log(f"looking for {name}...")
        dev = await BleakScanner.find_device_by_name(name, timeout=10)
        if not dev:
            continue
        try:
            async with BleakClient(dev) as client:
                log(f"connected to {name} over BLE")
                await client.start_notify(NUS_TX, lambda _h, data: fwd.feed(bytes(data)))
                loop = asyncio.get_running_loop()
                while client.is_connected:
                    cue = await loop.run_in_executor(None, fwd.next_cue)
                    if cue:
                        await client.write_gatt_char(NUS_RX, (cue + "\n").encode("ascii"), response=False)
        except Exception as exc:                          # bleak raises many kinds; just reconnect
            log(f"BLE link lost ({type(exc).__name__}), reconnecting")
        await asyncio.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="auto", help="COM port of the paired ESP32 (default: find it)")
    ap.add_argument("--ble", metavar="NAME", help="use BLE (Nordic UART) with this device name instead")
    a = ap.parse_args()
    fwd = Forwarder()

    def stats():
        while True:
            time.sleep(10)
            if fwd.lines:
                log(f"{fwd.lines / 10:.0f} lines/s forwarded to UDP {SENSOR_PORT}" + (f", {fwd.bad} bad" if fwd.bad else ""))
            fwd.lines = fwd.bad = 0

    threading.Thread(target=stats, daemon=True).start()
    try:
        if a.ble:
            asyncio.run(run_ble(fwd, a.ble))
        else:
            run_classic(fwd, a.port)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
