"""StepCue Tier 2: NVIDIA Nemotron reasoning layer.

Polls state.json. When a freeze episode completes, or one has lasted longer
than 4 s, it asks Nemotron (NVIDIA NIM, OpenAI-compatible API) to call the
`update_ankle_cue` tool: choose the cue tempo, switch wrist haptics on or off,
set the caregiver alert level, flag an uncertain sensor signal and write a
short caregiver brief. The decision is validated, written back to state.json
(the classifier applies it to the ESP32) and appended to advisor_log.jsonl for
the dashboard.

Setup:  put NVIDIA_API_KEY=nvapi-... in a .env file next to this script
        (or set it as an environment variable). Optional: NEMOTRON_MODEL.
Run:    python nemotron_advisor.py
Check:  python nemotron_advisor.py --test      # one sample call, prints the result

Without a key (or if a call fails) it falls back to simple built-in rules so
the demo keeps running; those decisions are labelled "rules" in the log.

Boundary: this layer only tunes assistive cueing and summarises mobility
patterns for a caregiver. It never diagnoses or gives treatment advice.
"""
import argparse
import json
import os
import re
import time

from config import (ALERT_LEVELS, BPM_MAX, BPM_MIN, DEFAULT_BPM,
                    PROLONGED_FREEZE_SEC)
from state_io import read_state, update_state

NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
LOG_FILE = "advisor_log.jsonl"
POLL_SEC = 0.5
MAX_BPM_STEP = 20              # largest tempo change allowed in one decision
CALM_AFTER_SEC = 90            # no freezes this long -> alert level back to normal
BRIEF_MAX_CHARS = 320

# Words that would take a brief outside "assistive cueing and awareness".
_OFF_LIMITS = re.compile(
    r"diagnos|prescri|medicat|dosage|\bdose\b|levodopa|treatment|therapy|disease progress|prognos",
    re.I)

SYSTEM_PROMPT = f"""You are the reasoning layer of StepCue, an assistive wearable prototype.
An ankle sensor detects freezing-of-gait episodes (the wearer's feet suddenly stop mid-stride).
The device answers with rhythmic cues: a flashing ankle light and, optionally, wrist vibration pulses
on each beat. Your job is to tune those cues and keep a caregiver informed.

You MUST respond by calling the update_ankle_cue tool exactly once.

How to choose the settings:
- bpm: the cue tempo, {BPM_MIN}-{BPM_MAX}. A tempo near or slightly below (about 5-10%) the wearer's
  pre-freeze cadence is a sensible starting point. If recovery after the cue is quick (under ~2 s),
  keep what works. If recovery is slow (over ~3 s) or getting slower, adjust the tempo toward that
  range. Change by at most {MAX_BPM_STEP} bpm at a time.
- haptic: 1 turns wrist vibration on, 0 leaves only the light. Turn it on when the light alone is
  not enough (slow recovery, a freeze lasting over {PROLONGED_FREEZE_SEC:.0f} s, repeated episodes).
  Turn it back off only after several quick recoveries in a row.
- alert_level: "normal" (nothing for the caregiver to do), "watch" (pattern worth keeping an eye on,
  e.g. several episodes close together or slowing recovery), "alert" (a freeze is ongoing and
  prolonged right now, or episodes are frequent and long: the caregiver should check in).
- signal_uncertain: true if the data looks unreliable, e.g. episodes shorter than 1 s, implausible
  cadence values, or rapid flickering between states. Then prefer small or no changes.
- caretaker_brief: one or two plain sentences for a family caregiver. Say what happened, what the
  device changed, and whether to check in. Use concrete numbers from the data. If most freezes
  began during turns, say so plainly (a pattern, not a cause).
- reasoning: one or two sentences on why you chose these settings.

Hard boundary: you only describe mobility patterns and cueing. Never diagnose, never mention
medication or treatment, never speculate about the wearer's condition or its progression."""

TOOL = {
    "type": "function",
    "function": {
        "name": "update_ankle_cue",
        "description": "Set the rhythmic cue parameters on the StepCue wearable and update the caregiver view.",
        "parameters": {
            "type": "object",
            "properties": {
                "bpm": {"type": "integer", "minimum": BPM_MIN, "maximum": BPM_MAX,
                        "description": "Cue tempo in beats per minute."},
                "haptic": {"type": "integer", "enum": [0, 1],
                           "description": "1 = wrist vibration pulses on each beat, 0 = light only."},
                "alert_level": {"type": "string", "enum": list(ALERT_LEVELS)},
                "signal_uncertain": {"type": "boolean",
                                     "description": "True if the sensor data looks unreliable."},
                "caretaker_brief": {"type": "string",
                                    "description": "One or two plain sentences for the caregiver."},
                "reasoning": {"type": "string", "description": "Short justification for these settings."},
            },
            "required": ["bpm", "haptic", "alert_level", "signal_uncertain", "caretaker_brief", "reasoning"],
        },
    },
}


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
    print(f"[advisor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------- context ----
def build_context(state, trigger, focus_ids):
    eps = state.get("episodes", [])
    done = [e for e in eps if e.get("recovery_sec") is not None]
    recov = [e["recovery_sec"] for e in done]
    return {
        "trigger": trigger,
        "now": time.strftime("%H:%M:%S"),
        "current_settings": {"bpm": state.get("bpm"), "haptic": state.get("haptic"),
                             "alert_level": state.get("alert_level")},
        "live": {"status": state.get("status"), "freeze_index": state.get("freeze_index")},
        "session_summary": {
            "episodes_total": len(eps),
            "mean_recovery_sec": round(sum(recov) / len(recov), 1) if recov else None,
            "last_three_recovery_sec": recov[-3:],
            "episodes_during_turns": sum(1 for e in eps if e.get("during_turn")),
        },
        "movement": state.get("motion"),
        "episodes_needing_review": [e for e in eps if e.get("id") in focus_ids],
        "recent_episodes": eps[-8:],
        "field_notes": {
            "duration_sec": "time spent frozen",
            "recovery_sec": "time from cue start until walking resumed; null = still ongoing",
            "pre_freeze_cadence": "steps/min just before the freeze; null = unknown",
            "cue_bpm": "cue tempo used for that episode",
            "during_turn": "true if the freeze began while turning or within 1.5 s of a turn",
            "movement": "steps, cadence and turn counts from the ankle sensor this session",
        },
    }


# ------------------------------------------------------- agent tools ------
READ_TOOLS = [
    {"type": "function", "function": {
        "name": "get_cue_effectiveness",
        "description": "How quickly walking resumed for each cue setting used so far (tempo and wrist vibration). "
                       "Use it before changing bpm or haptic to pick the setting that has worked best.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_time_pattern",
        "description": "Freeze episodes grouped by hour of day and by whether they began in a turn.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_phone_and_watch",
        "description": "Signals from the wearer's phone and watch: whether the wearer confirmed or denied each "
                       "freeze, the latest check-in (OK / asked for help), how walking feels today, heart rate "
                       "and steps if a watch is connected.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_episode_history",
        "description": "Full list of this session's freeze episodes (older ones too), newest last.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 50}}}}},
]
MAX_ROUNDS = 4

AGENT_NOTE = """

You also have read-only tools. Before changing the tempo or vibration, check get_cue_effectiveness
so you move toward the setting that has actually worked fastest for this wearer. Use get_phone_and_watch
when the wearer's own confirmation, a help request or heart rate would change what the caregiver
should hear (for example: the wearer denied a freeze, so treat it as a possible false alarm). Keep it
to the tools you need; then call update_ankle_cue exactly once."""


def cue_effectiveness(episodes):
    groups = {}
    for e in episodes:
        if e.get("recovery_sec") is None:
            continue
        key = (int(e.get("cue_bpm") or 0), int(bool(e.get("cue_haptic", 0))))
        groups.setdefault(key, []).append(float(e["recovery_sec"]))
    rows = [{"cue_bpm": b, "haptic": h, "episodes": len(v),
             "mean_recovery_sec": round(sum(v) / len(v), 2), "best_sec": round(min(v), 2)}
            for (b, h), v in sorted(groups.items())]
    best = min(rows, key=lambda r: (r["mean_recovery_sec"], -r["episodes"])) if rows else None
    return {"by_setting": rows, "fastest_setting": best,
            "note": "Fewer than 2 episodes for a setting is weak evidence." if rows else "No completed episodes yet."}


def time_pattern(episodes):
    hours = {}
    for e in episodes:
        hh = str(e.get("timestamp", ""))[:2]
        if hh.isdigit():
            hours[hh] = hours.get(hh, 0) + 1
    turns = sum(1 for e in episodes if e.get("during_turn"))
    return {"by_hour": dict(sorted(hours.items())), "during_turns": turns, "total": len(episodes)}


def phone_and_watch(episodes):
    try:
        import companion_store as cs
        summ = cs.companion_summary(cs.read_companion(), episodes)
    except Exception as exc:                                  # no companion data yet
        return {"available": False, "why": type(exc).__name__}
    ck, wf, hr, steps = (summ.get(k) for k in ("checkin", "walking_feel", "heart_rate", "steps"))
    return {"available": True,
            "latest_checkin": ck and {"text": ck.get("text"), "ago": ck.get("ago")},
            "help_requested_last_hour": bool(summ.get("help_open")),
            "walking_feels": wf.get("value") if isinstance(wf, dict) else wf,
            "heart_rate_bpm": hr.get("bpm") if isinstance(hr, dict) else hr,
            "steps_today": steps.get("count") if isinstance(steps, dict) else steps,
            "freeze_verification": summ.get("verified")}


def run_read_tool(name, args, state):
    eps = state.get("episodes", [])
    if name == "get_cue_effectiveness":
        return cue_effectiveness(eps)
    if name == "get_time_pattern":
        return time_pattern(eps)
    if name == "get_phone_and_watch":
        return phone_and_watch(eps)
    if name == "get_episode_history":
        n = max(1, min(50, int((args or {}).get("limit") or 20)))
        return eps[-n:]
    return {"error": f"unknown tool {name}"}


# -------------------------------------------------------------- Nemotron --
class Nemotron:
    """Nemotron as a small agent: it may call read tools (up to MAX_ROUNDS) before it must
    call update_ankle_cue. `trace` records which tools it used, for the dashboard."""

    def __init__(self, api_key=None, model=None, nim=None):
        import nvidia_nim
        self.nim = nim or nvidia_nim.NIM(key=api_key, timeout=25.0)
        self.model = model or nvidia_nim.model_for("agent")
        self.forced_choice_ok = True
        self.trace = []

    def _create(self, messages, tools, final):
        choice = ({"type": "function", "function": {"name": "update_ankle_cue"}}
                  if final and self.forced_choice_ok else "auto")
        try:
            return self.nim.chat("agent", messages, model=self.model, tools=tools, tool_choice=choice, thinking=False,
                                 temperature=0.2, max_tokens=1024)
        except Exception as exc:
            if choice == "auto" or getattr(exc, "status_code", None) not in (400, 422):
                raise
            self.forced_choice_ok = False      # this model wants tool_choice="auto"
            return self.nim.chat("agent", messages, model=self.model, tools=tools, tool_choice="auto", thinking=False,
                                 temperature=0.2, max_tokens=1024)

    def decide(self, context, state=None):
        state = state if state is not None else {"episodes": context.get("recent_episodes", [])}
        self.trace = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + AGENT_NOTE},
            {"role": "user", "content": "Device data:\n" + json.dumps(context, indent=1)
                                        + "\n\nLook up anything you need, then call update_ankle_cue."},
        ]
        for rnd in range(MAX_ROUNDS):
            final = rnd == MAX_ROUNDS - 1
            resp = self._create(messages, [TOOL] if final else READ_TOOLS + [TOOL], final)
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            if not calls:
                match = re.search(r"\{.*\}", msg.content or "", re.S)   # arguments written as text
                if match:
                    data = json.loads(match.group(0))
                    self.trace.append("update_ankle_cue")
                    return data.get("arguments") or data.get("parameters") or data
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"role": "user", "content": "Call update_ankle_cue now."})
                continue
            final_call = next((c for c in calls if c.function.name == "update_ankle_cue"), None)
            if final_call:
                self.trace.append("update_ankle_cue")
                return json.loads(final_call.function.arguments or "{}")
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name,
                                                          "arguments": c.function.arguments or "{}"}}
                                            for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except ValueError:
                    args = {}
                self.trace.append(c.function.name)
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": json.dumps(run_read_tool(c.function.name, args, state))})
        raise ValueError("model did not call update_ankle_cue")


def guard_brief(brief, llm):
    """Run the brief past the NVIDIA guard models (the local check always runs)."""
    try:
        import nvidia_guard
        return nvidia_guard.check_reply(brief, "Caregiver status update from the wearable",
                                        nim=getattr(llm, "nim", None))
    except Exception as exc:
        return {"ok": True, "layers": ["local"], "reason": f"guard unavailable: {type(exc).__name__}"}


# -------------------------------------------------- built-in fallback -----
def turn_note(context):
    """'3 of 4 freezes began during a turn.' when that is the usual pattern, else ''."""
    s = context["session_summary"]
    n, k = s["episodes_total"], s.get("episodes_during_turns") or 0
    return f"{k} of {n} freezes began during a turn." if n >= 2 and k * 2 > n else ""


def rule_decision(context):
    d = _rule_decision(context)
    note = turn_note(context)
    if note and context["trigger"] != "prolonged_freeze" and not d["signal_uncertain"]:
        d["caretaker_brief"] = f'{d["caretaker_brief"]} {note}'
    return d


def _rule_decision(context):
    cur = context["current_settings"]
    bpm, haptic = int(cur.get("bpm") or DEFAULT_BPM), int(cur.get("haptic") or 0)
    focus = context["episodes_needing_review"] or context["recent_episodes"][-1:]
    last = focus[-1] if focus else {}
    cadence = last.get("pre_freeze_cadence")
    target = int(round(0.92 * cadence)) if cadence and 50 <= cadence <= 160 else bpm
    n = context["session_summary"]["episodes_total"]
    dur = last.get("duration_sec") or 0
    uncertain = bool(last) and last.get("recovery_sec") is not None and dur < 1.0

    if context["trigger"] == "prolonged_freeze":
        return {"bpm": target, "haptic": 1, "alert_level": "alert", "signal_uncertain": False,
                "caretaker_brief": f"A freeze has lasted over {PROLONGED_FREEZE_SEC:.0f} s. Wrist vibration cues are "
                                   f"now on at {target} bpm. Please check in.",
                "reasoning": "Light cue alone has not restarted stepping within 4 s."}
    rec = last.get("recovery_sec") or 0
    if uncertain:
        return {"bpm": bpm, "haptic": haptic, "alert_level": cur.get("alert_level") or "normal",
                "signal_uncertain": True,
                "caretaker_brief": f"A very brief freeze reading ({dur:.1f} s) was logged. The sensor "
                                   "signal may be unreliable, so cue settings were left unchanged.",
                "reasoning": "Sub-second episode is more likely sensor noise than a freeze."}
    if rec > 3.0:
        return {"bpm": target, "haptic": 1, "alert_level": "watch", "signal_uncertain": False,
                "caretaker_brief": f"Freeze of {dur:.1f} s; stepping resumed {rec:.1f} s after cueing. "
                                   f"Wrist vibration is on and the cue tempo is {target} bpm for next time.",
                "reasoning": ("Recovery slower than 3 s; the stronger cue is already in place."
                              if haptic and target == bpm else
                              "Recovery slower than 3 s, so strengthen the cue.")}
    level = "watch" if n >= 3 else "normal"
    return {"bpm": bpm, "haptic": haptic, "alert_level": level, "signal_uncertain": False,
            "caretaker_brief": f"Freeze of {dur:.1f} s; stepping resumed {rec:.1f} s after cueing at "
                               f"{bpm} bpm. {n} episode(s) this session. Settings unchanged.",
            "reasoning": "Recovery was quick, current cue is working."}


# ------------------------------------------------------------ validation --
def sanitize(raw, state):
    """Clamp a decision to safe values. Raises ValueError if unusable."""
    cur_bpm = int(state.get("bpm") or DEFAULT_BPM)
    bpm = int(round(float(raw["bpm"])))
    bpm = max(cur_bpm - MAX_BPM_STEP, min(cur_bpm + MAX_BPM_STEP, bpm))
    bpm = max(BPM_MIN, min(BPM_MAX, bpm))
    haptic = 1 if str(raw.get("haptic")).strip().lower() in ("1", "true", "1.0") else 0
    level = str(raw.get("alert_level", "normal")).lower()
    if level not in ALERT_LEVELS:
        level = "watch"
    uncertain = raw.get("signal_uncertain") in (True, "true", "True", 1)
    brief = " ".join(str(raw.get("caretaker_brief", "")).split())
    if not brief:
        raise ValueError("empty caretaker brief")
    if _OFF_LIMITS.search(brief):
        raise ValueError("brief strayed into medical language")
    if uncertain and "signal" not in brief.lower() and "sensor" not in brief.lower():
        brief += " Sensor signal looked uncertain; check the ankle strap."
    if len(brief) > BRIEF_MAX_CHARS:
        brief = brief[:BRIEF_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return {"bpm": bpm, "haptic": haptic, "alert_level": level, "signal_uncertain": bool(uncertain),
            "caretaker_brief": brief, "reasoning": " ".join(str(raw.get("reasoning", "")).split())[:400]}


# ------------------------------------------------------------------ loop --
class Advisor:
    def __init__(self, llm):
        self.llm = llm
        self.prolonged_done = set()
        self.last_episode_time = time.monotonic()

    def decide(self, state, trigger, focus_ids):
        context = build_context(state, trigger, focus_ids)
        source, t0, err = "rules", time.monotonic(), None
        decision, tools, guard = None, [], None
        if self.llm:
            try:
                decision = sanitize(self.llm.decide(context, state), state)
                source = self.llm.model
                tools = list(getattr(self.llm, "trace", []))
            except Exception as exc:                      # network, quota, bad output...
                err = f"{type(exc).__name__}: {exc}"[:200]
                log(f"Nemotron call failed, using built-in rules ({err})")
        if decision is not None:
            guard = guard_brief(decision["caretaker_brief"], self.llm)
            if not guard["ok"]:
                err = f"brief blocked by guard ({guard['reason']})"
                log(err + ", using the built-in brief")
                decision["caretaker_brief"] = sanitize(rule_decision(context), state)["caretaker_brief"]
        if decision is None:
            decision = sanitize(rule_decision(context), state)
        latency = time.monotonic() - t0
        self.last_extra = {"tools": tools, "guard": guard["layers"] if guard else ["local"]}
        return decision, source, latency, err

    def apply(self, decision, source, latency, err, trigger, focus_ids, before):
        def mutate(st):
            st["bpm"] = decision["bpm"]
            st["haptic"] = decision["haptic"]
            st["alert_level"] = decision["alert_level"]
            st["caretaker_brief"] = decision["caretaker_brief"]
            if trigger == "episode_completed":
                for e in st.get("episodes", []):
                    if e.get("id") in focus_ids and e.get("recovery_sec") is not None:
                        e["analyzed"] = True
        update_state(mutate)
        entry = {"time": time.strftime("%H:%M:%S"), "trigger": trigger, "episode_ids": sorted(focus_ids),
                 "source": source, "latency_sec": round(latency, 2), "error": err,
                 "before": before, "decision": decision, **getattr(self, "last_extra", {})}
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        log(f"{trigger} #{sorted(focus_ids)} via {source} in {latency:.1f}s: "
            f"bpm {before['bpm']}->{decision['bpm']}, haptic {before['haptic']}->{decision['haptic']}, "
            f"alert {decision['alert_level']}"
            + (", SIGNAL UNCERTAIN" if decision["signal_uncertain"] else ""))
        log(f"  brief: {decision['caretaker_brief']}")
        if decision["reasoning"]:
            log(f"  why:   {decision['reasoning']}")

    def step(self):
        state = read_state()
        eps = state.get("episodes", [])
        before = {"bpm": state.get("bpm"), "haptic": state.get("haptic"),
                  "alert_level": state.get("alert_level")}

        ongoing = [e for e in eps if e.get("recovery_sec") is None]
        for e in ongoing:
            self.last_episode_time = time.monotonic()
            if (e.get("duration_sec") or 0) > PROLONGED_FREEZE_SEC and e["id"] not in self.prolonged_done:
                self.prolonged_done.add(e["id"])
                ids = {e["id"]}
                self.apply(*self.decide(state, "prolonged_freeze", ids), "prolonged_freeze", ids, before)
                return

        todo = {e["id"] for e in eps if e.get("recovery_sec") is not None and not e.get("analyzed")}
        if todo:
            self.last_episode_time = time.monotonic()
            self.apply(*self.decide(state, "episode_completed", todo), "episode_completed", todo, before)
            return

        if (state.get("alert_level") != "normal" and not ongoing
                and time.monotonic() - self.last_episode_time > CALM_AFTER_SEC):
            def calm(st):
                st["alert_level"] = "normal"
                st["caretaker_brief"] = (f"Mobility steady. No freeze episodes in the last "
                                         f"{CALM_AFTER_SEC} s. Cue settings kept at {st.get('bpm')} bpm.")
            update_state(calm)
            log("no recent episodes, alert level back to normal")

    def run(self):
        while True:
            try:
                self.step()
            except Exception as exc:
                log(f"error: {type(exc).__name__}: {exc}")
            time.sleep(POLL_SEC)


def make_llm():
    load_dotenv()
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
    model = os.environ.get("NEMOTRON_MODEL", DEFAULT_MODEL)
    if not key:
        log("NVIDIA_API_KEY not set: running on built-in rules only (no Nemotron calls)")
        return None
    log(f"using {model} via {NIM_BASE_URL}")
    return Nemotron(key, model)


SAMPLE_STATE = {
    "status": "WALKING", "freeze_index": 0.4, "bpm": 100, "haptic": 0, "alert_level": "normal",
    "caretaker_brief": "", "episodes": [
        {"id": 1, "timestamp": "14:10:02", "duration_sec": 2.1, "pre_freeze_cadence": 98,
         "cue_bpm": 100, "recovery_sec": 1.9, "analyzed": True},
        {"id": 2, "timestamp": "14:12:05", "duration_sec": 5.8, "pre_freeze_cadence": 96,
         "cue_bpm": 100, "recovery_sec": 5.4, "analyzed": False}]}


def main():
    ap = argparse.ArgumentParser(description="StepCue Nemotron advisor")
    ap.add_argument("--test", action="store_true", help="one sample call, print result, exit")
    a = ap.parse_args()
    llm = make_llm()
    if a.test:
        context = build_context(SAMPLE_STATE, "episode_completed", {2})
        if llm:
            t0 = time.monotonic()
            raw = llm.decide(context, SAMPLE_STATE)
            print("tools used: " + " -> ".join(llm.trace))
            print(f"raw tool call from {llm.model} ({time.monotonic() - t0:.1f}s):")
            print(json.dumps(raw, indent=2))
        else:
            raw = rule_decision(context)
        print("validated decision:")
        print(json.dumps(sanitize(raw, SAMPLE_STATE), indent=2))
        return
    try:
        Advisor(llm).run()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
