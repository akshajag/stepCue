"""Daily care report written by NVIDIA Nemotron 3 Ultra (550B), for the care team.

The real-time agent uses the fast Lightning model; this is the slow, thorough pass:
Ultra reads the whole day (every episode, every agent decision, phone check-ins,
cue-effectiveness and time-of-day statistics) and writes a structured report:
headline, what happened, what the device did, patterns, and points to raise with
the care team. Model routing: Ultra first, then Lightning, then a local template,
and the report says which one wrote it.

Every line passes the guard (no diagnosis, medication or treatment).
Run:  python nvidia_report.py     -> writes care_report.json
"""
import json
import os
import re
import time

import nvidia_guard
import nvidia_nim
from nemotron_advisor import cue_effectiveness, time_pattern

REPORT_FILE = os.environ.get("STEPCUE_REPORT", "care_report.json")
FIELDS = ("headline", "what_happened", "what_the_device_did", "patterns", "for_the_care_team")

SYSTEM = """You write the end-of-day mobility report for the care team of someone who wears StepCue,
an assistive wearable that detects freezing-of-gait episodes and answers with rhythmic light and
vibration cues. You are given the day's data as JSON.

Return ONLY a JSON object with these keys:
  "headline": one plain sentence summarising the day,
  "what_happened": 2-4 short bullet strings with numbers (episodes, durations, recovery times, when),
  "what_the_device_did": 1-3 bullet strings on cue tempo / vibration changes and whether they helped,
  "patterns": 1-3 bullet strings on patterns in the data (time of day, turns, wearer confirmations),
              stated as observations, never causes,
  "for_the_care_team": 1-3 bullet strings: observations worth raising at the next visit, phrased as
              things to discuss (e.g. "Most freezes began in turns; worth mentioning at the next visit").

Rules: use only the data given; say "not enough data" rather than guess; refer to "the wearer";
never diagnose, never mention medication, treatment or therapy, never predict progression."""


def day_data(state, decisions, companion_summary=None):
    eps = state.get("episodes", [])
    done = [e for e in eps if e.get("recovery_sec") is not None]
    rec = [e["recovery_sec"] for e in done]
    return {
        "date": time.strftime("%Y-%m-%d"),
        "episodes": eps[-60:],
        "totals": {"episodes": len(eps), "completed": len(done),
                   "mean_recovery_sec": round(sum(rec) / len(rec), 2) if rec else None,
                   "longest_freeze_sec": max((e.get("duration_sec") or 0 for e in eps), default=None)},
        "cue_effectiveness": cue_effectiveness(eps),
        "time_pattern": time_pattern(eps),
        "movement": state.get("motion"),
        "current_cue": {"bpm": state.get("bpm"), "haptic": state.get("haptic")},
        "agent_decisions": [{"time": d.get("time"), "trigger": d.get("trigger"),
                             "by": "rules" if d.get("source") == "rules" else "nemotron",
                             "tools_used": d.get("tools"), "before": d.get("before"),
                             "after": {k: d.get("decision", {}).get(k) for k in ("bpm", "haptic", "alert_level")},
                             "reasoning": d.get("decision", {}).get("reasoning")}
                            for d in list(decisions)[:20]],
        "phone": companion_summary or {},
    }


def parse_report(text):
    m = re.search(r"\{.*\}", nvidia_nim.strip_thinking(text), re.S)
    if not m:
        raise ValueError("no JSON in report")
    raw = json.loads(m.group(0))
    out = {"headline": " ".join(str(raw.get("headline", "")).split())[:240]}
    for k in FIELDS[1:]:
        v = raw.get(k) or []
        if isinstance(v, str):
            v = [v]
        out[k] = [" ".join(str(x).split())[:260] for x in v if str(x).strip()][:4]
    if not out["headline"]:
        raise ValueError("empty headline")
    return out


def guard_report(rep):
    """Drop any line that strays into medical language; refuse the report if the headline does."""
    if not nvidia_guard.local_ok(rep["headline"]):
        raise ValueError("headline strayed into medical language")
    for k in FIELDS[1:]:
        rep[k] = [x for x in rep[k] if nvidia_guard.local_ok(x)]
    return rep


def local_report(data):
    t = data["totals"]
    best = data["cue_effectiveness"]["fastest_setting"]
    tp = data["time_pattern"]
    n = t["episodes"]
    rep = {
        "headline": (f"{n} freeze episode{'s' if n != 1 else ''} today"
                     + (f"; walking resumed after {t['mean_recovery_sec']:.1f} s on average." if t["mean_recovery_sec"] else ".")),
        "what_happened": [f"{n} episodes, {t['completed']} completed."]
                         + ([f"Longest freeze {t['longest_freeze_sec']:.1f} s."] if t["longest_freeze_sec"] else []),
        "what_the_device_did": [f"Cue now {data['current_cue']['bpm']} bpm, wrist vibration "
                                f"{'on' if data['current_cue']['haptic'] else 'off'}; "
                                f"{len(data['agent_decisions'])} agent decision(s)."],
        "patterns": ([f"{tp['during_turns']} of {tp['total']} began during a turn."] if tp["total"] else ["Not enough data."]),
        "for_the_care_team": ([f"Fastest recovery so far at {best['cue_bpm']} bpm "
                               f"({best['mean_recovery_sec']:.1f} s average over {best['episodes']})."] if best else []),
    }
    return rep


def write_report(state, decisions, companion_summary=None, nim=None, path=None):
    nim = nim or nvidia_nim.shared()
    data = day_data(state, decisions, companion_summary)
    t0 = time.monotonic()
    rep, source, tried = None, "local", []
    if nim.available:
        for job in ("report", "agent"):                  # Ultra first, then Lightning
            model = nvidia_nim.model_for(job)
            tried.append(model)
            try:
                text = nim.text("report", [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": json.dumps(data, indent=1)}],
                                model=model, temperature=0.3, max_tokens=1800)
                rep = guard_report(parse_report(text))
                source = model
                break
            except Exception:
                rep = None
    if rep is None:
        rep = guard_report(local_report(data))
    rep.update(source=source, tried=tried, generated_at=time.strftime("%Y-%m-%d %H:%M"),
               episodes=data["totals"]["episodes"], latency_sec=round(time.monotonic() - t0, 1))
    path = path or REPORT_FILE
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return rep


def read_report(path=None):
    try:
        with open(path or REPORT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    from state_io import read_state
    decisions = []
    try:
        with open("advisor_log.jsonl", encoding="utf-8") as f:
            decisions = [json.loads(x) for x in f.readlines()[-20:]][::-1]
    except OSError:
        pass
    try:
        import companion_store as cs
        st = read_state()
        summ = cs.companion_summary(cs.read_companion(), st.get("episodes", []))
    except Exception:
        st, summ = read_state(), None
    rep = write_report(st, decisions, summ)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
