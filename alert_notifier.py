"""Caregiver alerts: phone push via ntfy (free) and a voice call via Twilio.

  python alert_notifier.py            # watch state.json and alert the caregiver
  python alert_notifier.py --test     # send one test push (and a test call with --call)
  python alert_notifier.py --dry-run  # decide and log, but send nothing

It only reads state.json and uses the same status rules as the dashboard
(dashboard_model.screen_state), so it works even if the advisor is down:

  "Please check in" (freeze over 4 s, or advisor alert)  -> max-priority push + phone call
  "Keep an eye out" (advisor watch)                      -> normal push
  walking again after a "check in"                       -> low-priority all-clear push
  sensor silent for 2 min after being live               -> high-priority push
  wearer taps "I need help" on the companion site        -> max-priority push + phone call

Cooldowns stop repeat alerts, so a long episode does not keep calling.

Setup (.env, see .env.example). Nothing is sent until these are set:
  NTFY_TOPIC=stepcue-<long random string>   # install the ntfy app, subscribe to it
  NTFY_URL=https://ntfy.sh                  # or your own ntfy server
  NTFY_TOKEN=                               # optional access token
  TWILIO_ACCOUNT_SID=  TWILIO_AUTH_TOKEN=  TWILIO_FROM=+1...  CAREGIVER_PHONE=+1...

Privacy: messages say what is happening and nothing else. No names, no
location, and the AI brief only if NOTIFY_INCLUDE_BRIEF=1. Anyone who knows the
ntfy topic on the public server can read it, so use a long random topic or
your own server.
"""
import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from xml.sax.saxutils import escape

from config import PROLONGED_FREEZE_SEC
from dashboard_model import screen_state
from companion_store import read_companion
from state_io import read_state, state_age_sec

NOTIFY_LOG = "notify_log.jsonl"
POLL_SEC = 1.0
OFFLINE_NOTIFY_SEC = 120.0
COOLDOWN_SEC = {"alert_push": 180, "alert_call": 600, "watch": 900, "offline": 1800, "clear": 60}
PRIORITY = {"alert": 5, "help": 5, "offline": 4, "watch": 3, "clear": 2, "test": 3}
TAGS = {"alert": "rotating_light", "help": "sos", "offline": "electric_plug", "watch": "eyes",
        "clear": "white_check_mark", "test": "bell"}
HELP_MAX_AGE_SEC = 300


def load_dotenv(path=".env"):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def log(msg):
    print(f"[alerts {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------- channels --
class Ntfy:
    name = "ntfy push"

    def __init__(self, topic, url="https://ntfy.sh", token=None):
        self.topic, self.url, self.token = topic, url.rstrip("/"), token or None

    def request(self, title, message, priority, tag):
        headers = {"Title": title, "Priority": str(priority), "Tags": tag,
                   "Content-Type": "text/plain; charset=utf-8"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return urllib.request.Request(f"{self.url}/{urllib.parse.quote(self.topic)}",
                                      data=message.encode("utf-8"), headers=headers, method="POST")

    def send(self, title, message, priority, tag):
        with urllib.request.urlopen(self.request(title, message, priority, tag), timeout=10) as r:
            return f"HTTP {r.status}"


class TwilioCall:
    name = "Twilio call"
    API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json"

    def __init__(self, sid, token, from_number, to_number):
        self.sid, self.token, self.from_number, self.to_number = sid, token, from_number, to_number

    @staticmethod
    def twiml(spoken):
        line = escape(spoken)
        return (f'<Response><Pause length="1"/><Say>{line}</Say><Pause length="1"/>'
                f'<Say>Again. {line}</Say></Response>')

    def request(self, spoken):
        body = urllib.parse.urlencode({"To": self.to_number, "From": self.from_number,
                                       "Twiml": self.twiml(spoken)}).encode()
        auth = base64.b64encode(f"{self.sid}:{self.token}".encode()).decode()
        return urllib.request.Request(self.API.format(sid=self.sid), data=body, method="POST",
                                      headers={"Authorization": f"Basic {auth}",
                                               "Content-Type": "application/x-www-form-urlencoded"})

    def send(self, spoken):
        with urllib.request.urlopen(self.request(spoken), timeout=15) as r:
            sid = json.loads(r.read() or b"{}").get("sid", "")
            return f"HTTP {r.status} {sid}".strip()


def channels_from_env(env=None):
    env = os.environ if env is None else env
    push = call = None
    if env.get("NTFY_TOPIC"):
        push = Ntfy(env["NTFY_TOPIC"], env.get("NTFY_URL") or "https://ntfy.sh", env.get("NTFY_TOKEN"))
    keys = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "CAREGIVER_PHONE")
    if all(env.get(k) for k in keys):
        call = TwilioCall(*(env[k] for k in keys))
    return push, call


def channel_status(env=None):
    """For the dashboard: which channels are set up (no secrets)."""
    push, call = channels_from_env(env)
    return {"push": push is not None, "call": call is not None,
            "push_server": push.url if push else None}


# ----------------------------------------------------------- decision --
def message_for(kind, state, include_brief=False):
    """(title, push text, spoken text) for an alert kind. No names, no location."""
    eps = state.get("episodes") or []
    ongoing = next((e for e in eps if e.get("recovery_sec") is None), None)
    cue = ("light and vibration cue" if state.get("haptic") else "light cue") + f" at {state.get('bpm')} bpm"
    if kind == "alert":
        if ongoing:
            dur = float(ongoing.get("duration_sec") or 0)
            body = f"A freeze has lasted {dur:.1f} s. The {cue} is playing."
            spoken = (f"StepCue alert. A freeze has lasted more than {PROLONGED_FREEZE_SEC:.0f} seconds. "
                      "Please check in.")
        else:
            body = "Freezes are frequent and long right now."
            spoken = "StepCue alert. Freezes are frequent and long right now. Please check in."
        title = "StepCue: please check in"
    elif kind == "watch":
        title, body, spoken = "StepCue: keep an eye out", f"{len(eps)} freezes this session.", ""
    elif kind == "clear":
        title, body, spoken = "StepCue: walking again", "The freeze has ended and walking resumed.", ""
    elif kind == "help":
        title = "StepCue: help requested"
        body = "The wearer tapped \"I need help\" on their phone. Please check in now."
        spoken = "StepCue alert. The wearer asked for help on their phone. Please check in now."
    elif kind == "offline":
        title = "StepCue: no signal"
        body = "The ankle sensor has been silent for over 2 minutes. It may be off or out of range."
        spoken = ""
    else:
        title, body, spoken = "StepCue test", "Caregiver alerts are working.", "This is a StepCue test call."
    brief = str(state.get("caretaker_brief") or "").strip()
    if include_brief and brief and kind in ("alert", "watch"):
        body = f"{body}\n{brief}"
    return title, body, spoken


class Notifier:
    """Turns successive dashboard states into alerts. Pure apart from send()."""

    def __init__(self, push=None, call=None, dry_run=False, include_brief=False, clock=time.time):
        self.push, self.call = push, call
        self.dry_run = dry_run or (push is None and call is None)
        self.include_brief = include_brief
        self.clock = clock
        self.last_sent = {}
        self.prev_key = None
        self.seen_live = False
        self.offline_since = None
        self.offline_sent = False
        self.alerted = False           # a "check in" is outstanding, send all-clear later
        self.help_seen = None          # time of the newest help request already handled

    def _ready(self, slot, now):
        return now - self.last_sent.get(slot, -1e12) >= COOLDOWN_SEC[slot]

    def step(self, state, age_sec):
        """Feed one snapshot. Returns the list of actions taken."""
        now = self.clock()
        key = screen_state(state, age_sec)["key"]
        actions = []

        if key == "offline":
            self.offline_since = self.offline_since or now
            if (self.seen_live and not self.offline_sent and now - self.offline_since >= OFFLINE_NOTIFY_SEC
                    and self._ready("offline", now)):
                actions += self.fire("offline", state, now)
                self.offline_sent = True
        else:
            self.seen_live = True
            self.offline_since, self.offline_sent = None, False

        if key == "alert":
            if self._ready("alert_push", now) and (self.prev_key != "alert" or not self.alerted):
                actions += self.fire("alert", state, now)
                self.alerted = True
            if self.call is not None and self._ready("alert_call", now):
                actions += self.fire_call(state, now)
        elif key == "watch" and self.prev_key not in ("watch", "alert") and self._ready("watch", now):
            actions += self.fire("watch", state, now)
        elif key == "steady" and self.alerted and self._ready("clear", now):
            actions += self.fire("clear", state, now)
            self.alerted = False

        self.prev_key = key
        return actions

    def step_companion(self, companion):
        """Alert once per new "I need help" tap from the companion site (push + call, no cooldown)."""
        now = self.clock()
        helps = [c["t"] for c in companion.get("checkins", []) if c.get("kind") == "help"]
        if not helps:
            return []
        newest = max(helps)
        if self.help_seen is None:
            self.help_seen = newest if now - newest > HELP_MAX_AGE_SEC else newest - 1e-6
        if newest <= self.help_seen:
            return []
        self.help_seen = newest
        if now - newest > HELP_MAX_AGE_SEC:
            return []
        actions = self.fire("help", {}, now)
        if self.call is not None:
            _, _, spoken = message_for("help", {})
            actions.append(self.deliver("call", "help", "phone call", spoken, lambda: self.call.send(spoken)))
        return actions

    def fire(self, kind, state, now):
        title, body, _ = message_for(kind, state, self.include_brief)
        slot = "alert_push" if kind == "alert" else kind
        self.last_sent[slot] = now
        return [self.deliver("push", kind, title, body,
                             lambda: self.push.send(title, body, PRIORITY[kind], TAGS[kind]) if self.push else None)]

    def fire_call(self, state, now):
        _, _, spoken = message_for("alert", state)
        self.last_sent["alert_call"] = now
        return [self.deliver("call", "alert", "phone call", spoken, lambda: self.call.send(spoken))]

    def deliver(self, channel, kind, title, body, send):
        rec = {"time": time.strftime("%H:%M:%S"), "channel": channel, "kind": kind, "title": title,
               "message": body}
        if self.dry_run or (channel == "push" and self.push is None):
            rec["result"] = "dry run (not sent)"
        else:
            try:
                rec["result"] = send() or "sent"
            except urllib.error.HTTPError as exc:
                rec["result"] = f"failed: HTTP {exc.code}"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                rec["result"] = f"failed: {exc}"
        log(f"{channel} [{kind}] {title}: {rec['result']}")
        try:
            with open(NOTIFY_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        return rec


def read_notifications(limit=3):
    try:
        with open(NOTIFY_LOG, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out[::-1]


def send_test(with_call=False):
    """One test push (and optionally a call). Returns the delivery records."""
    load_dotenv()
    push, call = channels_from_env()
    n = Notifier(push, call)
    recs = []
    if push:
        recs += n.fire("test", {}, time.time())
    if with_call and call:
        recs.append(n.deliver("call", "test", "phone call", "This is a StepCue test call.",
                              lambda: call.send("This is a StepCue test call.")))
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="send one test push, then exit")
    ap.add_argument("--call", action="store_true", help="with --test, also place a test call")
    ap.add_argument("--dry-run", action="store_true", help="never send anything")
    a = ap.parse_args()
    load_dotenv()
    if a.test:
        recs = send_test(a.call)
        if not recs:
            print("Nothing configured. Set NTFY_TOPIC (and the TWILIO_* keys) in .env first.")
        return
    push, call = channels_from_env()
    n = Notifier(push, call, dry_run=a.dry_run,
                 include_brief=os.environ.get("NOTIFY_INCLUDE_BRIEF") == "1")
    log(f"push: {'ntfy ' + push.url if push else 'off'} | call: {'Twilio' if call else 'off'}"
        f"{' | DRY RUN' if n.dry_run else ''}")
    try:
        while True:
            n.step(read_state(), state_age_sec())
            n.step_companion(read_companion())
            time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
