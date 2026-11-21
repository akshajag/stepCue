"""Pure view-model logic for the caregiver dashboard (no Streamlit, fully tested).

See DESIGN.md for the status model these functions implement.
"""
from config import PROLONGED_FREEZE_SEC

STALE_AFTER_SEC = 3.0


def _ongoing(episodes):
    return next((e for e in episodes or [] if e.get("recovery_sec") is None), None)


def screen_state(state, age_sec, stale_after=STALE_AFTER_SEC):
    """The single answer the screen leads with. First matching rule wins."""
    episodes = state.get("episodes") or []
    ongoing = _ongoing(episodes)
    level = state.get("alert_level", "normal")
    bpm = state.get("bpm")
    cue = ("Light and vibration cue" if state.get("haptic") else "Light cue") + f" playing at {bpm} bpm"

    if age_sec >= stale_after:
        return {"key": "offline", "headline": "No signal from the ankle sensor",
                "detail": "Check that the sensor is switched on and within Wi-Fi range."}
    if ongoing:
        dur = float(ongoing.get("duration_sec") or 0.0)
        if dur > PROLONGED_FREEZE_SEC or level == "alert":
            return {"key": "alert", "headline": "Please check in",
                    "detail": f"Freeze has lasted {dur:.1f} s. {cue}."}
        return {"key": "freeze", "headline": "Freeze in progress",
                "detail": f"Frozen for {dur:.1f} s. {cue}."}
    if level == "alert":
        return {"key": "alert", "headline": "Please check in",
                "detail": "Recent freezes were long. A quick check-in is a good idea."}
    if level == "watch":
        return {"key": "watch", "headline": "Keep an eye out",
                "detail": "Freezes are taking longer to resolve than usual."}
    n = len(episodes)
    detail = ("No freeze episodes so far this session." if n == 0 else
              f"{n} freeze episode{'s' if n != 1 else ''} this session, all resolved.")
    return {"key": "steady", "headline": "All steady", "detail": detail}


def freshness(age_sec):
    if age_sec == float("inf"):
        return "Waiting for first data"
    if age_sec < STALE_AFTER_SEC:
        return f"Live, updated {age_sec:.1f} s ago"
    if age_sec < 120:
        return f"Last update {int(age_sec)} s ago"
    return f"Last update {int(age_sec // 60)} min ago"


def episode_stats(episodes):
    episodes = episodes or []
    done = [e for e in episodes if e.get("recovery_sec") is not None]
    rec = [float(e["recovery_sec"]) for e in done]
    stats = {"count": len(episodes), "completed": len(done),
             "avg_recovery": sum(rec) / len(rec) if rec else None,
             "last_recovery": rec[-1] if rec else None, "last_vs_avg": None}
    if len(rec) >= 2:
        earlier = sum(rec[:-1]) / len(rec[:-1])
        ratio = rec[-1] / earlier if earlier else 1.0
        stats["last_vs_avg"] = "faster" if ratio < 0.85 else "slower" if ratio > 1.15 else "similar"
    return stats


def time_in_state(history, now, window):
    """history: [(t, status), ...] sorted by time; each status holds until the next entry."""
    if not history:
        return {}
    start = now - window
    totals = {}
    for i, (t, status) in enumerate(history):
        t_end = history[i + 1][0] if i + 1 < len(history) else now
        a, b = max(t, start), min(t_end, now)
        if b > a:
            totals[status] = totals.get(status, 0.0) + (b - a)
    span = sum(totals.values())
    return {k: v / span for k, v in totals.items()} if span else {}


def decision_summary(entry):
    dec, before = entry.get("decision", {}), entry.get("before", {})
    parts = []
    if before.get("bpm") != dec.get("bpm"):
        parts.append(f"Tempo {before.get('bpm')} → {dec.get('bpm')} bpm")
    if before.get("haptic") != dec.get("haptic"):
        parts.append("wrist vibration " + ("on" if dec.get("haptic") else "off"))
    if not parts:
        parts.append("Kept the current cue")
    if dec.get("signal_uncertain"):
        parts.append("sensor signal flagged as unclear")
    text = ", ".join(parts)
    return text[0].upper() + text[1:]


def movement_summary(state):
    """Plain-language movement facts from the optional "motion" block, or None.

    Returns {"items": [(label, value, detail)], "turn_pattern": str | None, "turning": bool}.
    """
    m = state.get("motion")
    if not isinstance(m, dict):
        return None
    steps = int(m.get("steps") or 0)
    turns = int(m.get("turns") or 0)
    cad = m.get("cadence_spm")
    items = [("Steps", f"{steps:,}", "this session"),
             ("Walking pace", f"{int(cad)}" if cad else "–", "steps/min" if cad else "not walking now"),
             ("Turns", str(turns),
              f"{int(m.get('turns_left') or 0)} left, {int(m.get('turns_right') or 0)} right" if turns else "")]
    eps = state.get("episodes") or []
    k = sum(1 for e in eps if e.get("during_turn"))
    pattern = None
    if eps and k:
        pattern = (f"{k} of {len(eps)} freeze{'s' if len(eps) > 1 else ''} began during a turn"
                   if len(eps) > 1 else "The freeze began during a turn")
    return {"items": items, "turn_pattern": pattern, "turning": bool(m.get("turning"))}
