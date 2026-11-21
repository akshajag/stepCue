"""Activity page: what the wearer's Android phone measures, for the care team and family.

Tiles, gauges and charts built only from data StepCue really has (the phone's sensors, the ankle
sensor's freezes, an optional watch). No clinical numbers: walking pace, steps and the freeze score
are descriptions of movement, not a diagnosis.
"""
import html
import math
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

import companion_store as cs
import personal_model as pm

STEP_GOAL = int(os.environ.get("STEPCUE_STEP_GOAL", "5000"))
FI_WATCH, FI_FREEZE = 1.5, 2.0          # same threshold as the phone app (Features.FI_FREEZE)
TEAL, AMBER, RED, GREY, TRACK = "#0F766E", "#A35A00", "#9B1C1C", "#94A3B8", "#E7EBEF"

CSS = """<style>
.pa-head {display:flex; justify-content:space-between; align-items:flex-end; gap:12px; flex-wrap:wrap; margin:2px 0 12px;}
.pa-head h2 {font:700 1.5rem var(--font); color:var(--ink); margin:0; padding:0;}
.pa-head p {font:400 0.95rem var(--font); color:var(--muted); margin:2px 0 0;}
.pa-live {display:inline-flex; align-items:center; gap:8px; font:700 0.92rem var(--font); color:var(--ok);
          background:var(--ok-tint); border-radius:999px; padding:7px 14px; white-space:nowrap;}
.pa-live i {width:9px; height:9px; border-radius:50%; background:var(--ok); display:inline-block;}
.pa-live.stale {color:var(--offline); background:var(--offline-tint); border:1.5px dashed #AEB8C5;}
.pa-live.stale i {background:#B8C2CE;}
.pa-tiles {display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:10px; margin-bottom:12px;}
.pa-tile {display:flex; gap:12px; align-items:center; background:var(--surface); border:1px solid var(--line);
          border-radius:14px; padding:12px 14px; min-width:0; font-family:var(--font);}
.pa-tile .ico {flex:none; width:40px; height:40px; border-radius:11px; display:grid; place-items:center;
               background:var(--ok-tint); color:var(--ok);}
.pa-tile .ico.w {background:var(--watch-tint); color:var(--watch);}
.pa-tile .ico.a {background:#FBE9E9; color:var(--alert);}
.pa-tile .ico.n {background:#EEF1F4; color:var(--offline);}
.pa-tile .v {font-size:1.3rem; font-weight:700; color:var(--ink); line-height:1.2; font-variant-numeric:tabular-nums;
             white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.pa-tile .v small {font-size:0.85rem; font-weight:400; color:var(--muted); margin-left:3px;}
.pa-tile .k {font-size:0.85rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.pa-tile.dim .v {color:#9AA5B3;}
.pa-gauges {display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:10px; margin-bottom:14px;}
.pa-gauge {background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:12px 10px 10px;
           text-align:center; font-family:var(--font); min-width:0;}
.pa-gauge .t {font-weight:700; font-size:0.98rem; color:var(--ink);}
.pa-gauge .st {display:inline-block; margin-top:2px; font-weight:700; font-size:0.85rem; border-radius:999px; padding:2px 10px;}
.pa-gauge svg {width:100%; max-width:220px; display:block; margin:4px auto 0;}
.pa-gauge .n {font-size:0.82rem; color:var(--muted); line-height:1.35;}
.st-ok {background:var(--ok-tint); color:var(--ok);} .st-w {background:var(--watch-tint); color:#7A4300;}
.st-a {background:#FBE9E9; color:var(--alert);} .st-n {background:#EEF1F4; color:var(--offline);}
.pa-chips {display:flex; flex-wrap:wrap; gap:6px; margin-top:4px;}
.pa-chips span {font:400 0.82rem var(--font); background:#EEF1F4; color:var(--ink); border-radius:999px; padding:3px 10px;}
</style>"""

_ICON = {
    "steps": '<path d="M8 3c1.7 0 3 2 3 4.5S9.7 12 8 12s-3-2-3-4.5S6.3 3 8 3z"/><path d="M6 15h4v2.5a2 2 0 0 1-4 0z"/>'
             '<path d="M16 7c1.7 0 3 2 3 4.5S17.7 16 16 16s-3-2-3-4.5S14.3 7 16 7z"/><path d="M14 19h4"/>',
    "walk": '<circle cx="13" cy="4" r="2"/><path d="m9 21 3-7 3 3v4"/><path d="m6 12 3-4 4 1 3 4"/>',
    "pace": '<path d="M4 14a8 8 0 0 1 16 0"/><path d="m12 14 4-4"/><circle cx="12" cy="14" r="1.2"/>',
    "turn": '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/>',
    "stairs": '<path d="M3 20h5v-5h5v-5h5V5h3"/>',
    "heart": '<path d="M12 20s-7-4.4-9-9.1C1.7 7.6 4 4.5 7.2 4.5c2 0 3.4 1.1 4.8 2.9 1.4-1.8 2.8-2.9 4.8-2.9 3.2 0 5.5 3.1 4.2 6.4C19 15.6 12 20 12 20z"/>',
    "pocket": '<path d="M5 4h14v8a7 7 0 0 1-14 0z"/><path d="M9 9h6"/>',
    "battery": '<rect x="3" y="7" width="16" height="10" rx="2"/><path d="M21 11v2"/><path d="M6 10v4"/><path d="M9 10v4"/>',
    "freeze": '<path d="M12 3v18"/><path d="m5 7 14 10"/><path d="m19 7-14 10"/>',
}


def esc(v):
    return html.escape(str(v), quote=True)


def icon(name, size=22):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICON[name]}</svg>')


def tile(ico, value, label, tone="", dim=False):
    return (f'<div class="pa-tile{" dim" if dim else ""}"><div class="ico {tone}">{icon(ico)}</div>'
            f'<div style="min-width:0"><div class="v">{value}</div><div class="k">{esc(label)}</div></div></div>')


# ------------------------------------------------------------------ gauges --
def _pt(cx, cy, r, deg):
    a = math.radians(deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def _arc(cx, cy, r, a0, a1):
    x0, y0 = _pt(cx, cy, r, a0)
    x1, y1 = _pt(cx, cy, r, a1)
    return f"M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}"


def gauge_svg(value, lo, hi, bands=None, ticks=(), fill=TEAL):
    """Half-circle gauge. bands = [(from, to, color)] draws a coloured scale; otherwise a value arc."""
    cx, cy, r = 100, 100, 78

    def ang(v):
        f = 0.0 if v is None else (min(max(v, lo), hi) - lo) / float(hi - lo)
        return 180 - 180 * f

    parts = [f'<path d="{_arc(cx, cy, r, 180, 0)}" stroke="{TRACK}" stroke-width="18" fill="none"/>']
    if bands:
        for a, b, c in bands:
            parts.append(f'<path d="{_arc(cx, cy, r, ang(a), ang(b))}" stroke="{c}" stroke-width="18" fill="none" '
                         f'opacity="0.9"/>')
    elif value is not None and value > lo:
        parts.append(f'<path d="{_arc(cx, cy, r, 180, ang(value))}" stroke="{fill}" stroke-width="18" fill="none"/>')
    for tv in ticks:
        tx, ty = _pt(cx, cy, r - 22, ang(tv))
        parts.append(f'<text x="{tx:.1f}" y="{ty + 4:.1f}" font-size="10" fill="#556274" text-anchor="middle">{tv:g}</text>')
    if value is not None:
        nx, ny = _pt(cx, cy, r - 6, ang(value))
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#16202B" stroke-width="4" '
                     f'stroke-linecap="round"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="#16202B"/>')
    return f'<svg viewBox="0 0 200 112" role="img" aria-hidden="true">{"".join(parts)}</svg>'


def gauge_card(title, svg, big, status, tone, note):
    return (f'<div class="pa-gauge"><div class="t">{esc(title)}</div>'
            f'<span class="st st-{tone}">{esc(status)}</span>{svg}'
            f'<div style="font-size:1.45rem;font-weight:700;color:var(--ink);margin-top:-4px">{big}</div>'
            f'<div class="n">{esc(note)}</div></div>')


# ------------------------------------------------------------------ charts --
def _axis(**kw):
    base = dict(title=None, labelColor="#556274", labelFont="Atkinson Hyperlegible", labelFontSize=12,
                gridColor="#EEF1F4", domain=False, ticks=False)
    base.update(kw)
    return alt.Axis(**base)


def minutes_chart(rows):
    df = pd.DataFrame(rows)
    df["time"] = (df["t"] * 1000).astype("int64")
    x = alt.X("time:T", axis=_axis(format="%H:%M", grid=False, tickCount=6, tickMinStep=60000))
    pace = alt.Chart(df.dropna(subset=["cadence"])).mark_line(
        color=TEAL, strokeWidth=2.6, interpolate="monotone", point=alt.OverlayMarkDef(size=26, color=TEAL)).encode(
        x=x, y=alt.Y("cadence:Q", scale=alt.Scale(domain=[0, 150]), axis=_axis(tickCount=4)),
        tooltip=[alt.Tooltip("time:T", title="Time", format="%H:%M"), alt.Tooltip("cadence:Q", title="Steps/min")])
    pace = pace.properties(height=150, title=alt.TitleParams("Walking pace (steps per minute)", anchor="start",
                                                              fontSize=13, color="#16202B", font="Atkinson Hyperlegible"))
    df["level"] = df["freeze_index"].apply(lambda v: "Freeze-like" if v >= FI_FREEZE else
                                           ("Unsteady" if v >= FI_WATCH else "Steady"))
    df["freeze_index"] = df["freeze_index"].clip(upper=8)
    top = max(4.0, float(df["freeze_index"].max()) * 1.1)
    bars = alt.Chart(df).mark_bar(width=3, cornerRadiusEnd=1).encode(
        x=x, y=alt.Y("freeze_index:Q", scale=alt.Scale(domain=[0, top]), axis=_axis(tickCount=4)),
        color=alt.Color("level:N", scale=alt.Scale(domain=["Steady", "Unsteady", "Freeze-like"],
                                                    range=["#9CC9C2", "#E3A857", RED]),
                        legend=alt.Legend(orient="bottom", title=None, labelColor="#16202B", labelFontSize=12,
                                          symbolType="square")),
        tooltip=[alt.Tooltip("time:T", title="Time", format="%H:%M"),
                 alt.Tooltip("freeze_index:Q", title="Freeze score", format=".1f"), alt.Tooltip("level:N", title="")])
    rule = alt.Chart(pd.DataFrame({"y": [FI_FREEZE]})).mark_rule(color=RED, strokeDash=[5, 4], opacity=0.7).encode(y="y:Q")
    frz = alt.layer(bars, rule).properties(height=130, title=alt.TitleParams(
        "Freeze score (trembling vs. stepping, highest each minute)", anchor="start", fontSize=13, color="#16202B",
        font="Atkinson Hyperlegible"))
    return alt.vconcat(pace, frz, spacing=14).resolve_scale(x="shared").configure_view(strokeWidth=0)


def steps_chart(days):
    df = pd.DataFrame(days)
    df["has"] = df["steps"].notna()
    df["shown"] = df["steps"].fillna(0)
    df["text"] = df.apply(lambda r: f'{int(r["steps"]):,}' if r["has"] else "no data", axis=1)
    order = list(df["label"])
    x = alt.X("label:N", sort=order, axis=_axis(grid=False, labelAngle=0))
    ymax = max(STEP_GOAL * 1.25, float(df["shown"].max()) * 1.15)
    y = alt.Y("shown:Q", scale=alt.Scale(domain=[0, ymax]), axis=_axis(tickCount=5, format="~s"))
    bars = alt.Chart(df[df["has"]]).mark_bar(color=TEAL, cornerRadiusTopLeft=5, cornerRadiusTopRight=5,
                                             size=34).encode(
        x=x, y=y, tooltip=[alt.Tooltip("date:N", title="Day"), alt.Tooltip("steps:Q", title="Steps", format=",")])
    labels = alt.Chart(df).mark_text(dy=-9, fontSize=12, font="Atkinson Hyperlegible").encode(
        x=x, y=y, text="text:N", color=alt.condition("datum.has", alt.value("#16202B"), alt.value("#9AA5B3")))
    rule = alt.Chart(pd.DataFrame({"g": [STEP_GOAL]})).mark_rule(
        color=AMBER, strokeWidth=2, strokeDash=[6, 4]).encode(y="g:Q", tooltip=alt.value(f"Daily goal {STEP_GOAL:,}"))
    return (alt.layer(bars, labels, rule).properties(height=300, width="container")
            .configure_view(strokeWidth=0))


# ------------------------------------------------------------------- page --
def _pace_status(cad, motion):
    if motion != "walking" or not cad:
        return ("Standing still" if motion == "still" else "Not stepping"), "n"
    if cad < 70:
        return "Short, slow steps", "w"
    return "Walking", "ok"


@st.fragment(run_every=3)
def activity_page(read_state):
    st.markdown(CSS, unsafe_allow_html=True)
    now = time.time()
    comp = cs.read_companion()
    rec = comp.get("android") or {}
    age = now - rec["t"] if rec.get("t") else None
    live = age is not None and age <= 60
    summ = cs.companion_summary(comp, read_state().get("episodes", []), now)
    days = cs.android_days(comp, 7, now)
    today = days[-1]
    mins = cs.android_minutes(comp, 3, now)

    if rec:
        pill = (f'<span class="pa-live"><i></i>Live · {esc(rec.get("device", "Android phone"))}</span>' if live else
                f'<span class="pa-live stale"><i></i>Phone not sharing · last seen {esc(cs.ago(rec["t"], now))}</span>')
    else:
        pill = '<span class="pa-live stale"><i></i>No phone connected yet</span>'
    st.markdown(f'<div class="pa-head"><div><h2>Activity</h2><p>Measured by the phone in their pocket: '
                f'{len(rec.get("sensors") or [])} sensors, {int(rec.get("hz") or 0)} readings a second.</p></div>'
                f'{pill}</div>', unsafe_allow_html=True)
    if not rec:
        st.info("Install the StepCue Android app from the QR code on the Access page. Steps, walking pace, "
                "turns and stairs appear here as soon as the phone starts sharing.")
        return

    cad, motion = rec.get("cadence") or 0, rec.get("motion", "still")
    fi = float(rec.get("freeze_index") or 0)
    hr = summ.get("heart_rate")
    batt = rec.get("battery")
    walk_min = today["walk_min"] or 0
    alt_m = rec.get("alt_change_m")
    stairs = rec.get("stairs")
    turns_hr = sum(1 for m in mins if m["turning"] and now - m["t"] <= 3600)
    d = not live                       # dim live-only values when the phone is quiet

    phase = rec.get("phase") if live else None
    pid = pm.active()[0]
    model = pm.model_info(pid) if pid else None
    tiles = [
        tile("walk", esc(phase["word"]) + f'<small>{phase["conf"]:.0%} sure</small>' if phase else "—",
             "Doing now (their own model)" if model else "Doing now (train a model first)",
             "a" if phase and phase["label"] == "frozen" else ("" if phase else "n"), dim=not phase),
        tile("steps", f'{today["steps"] or 0:,}<small>steps</small>', "Steps today (phone)"),
        tile("walk", f'{walk_min:.0f}<small>min</small>', "Time walking today"),
        tile("pace", (f'{int(cad)}<small>steps/min</small>' if live and cad else "—"), "Walking pace now", dim=d),
        tile("turn", f'{turns_hr}<small>{"turn" if turns_hr == 1 else "turns"}</small>', "Turns in the last hour",
             "w" if live and rec.get("turning") else ""),
        tile("stairs", ({"up": "Going up", "down": "Going down"}.get(stairs, "Level") if live else "—")
             + (f'<small>{alt_m:+.0f} m</small>' if live and alt_m is not None and abs(alt_m) >= 1 else ""),
             "Stairs (barometer)", dim=d),
        tile("heart", f'{hr["bpm"]}<small>bpm</small>' if hr else "—",
             f'Heart rate ({hr["source"]})' if hr else "Heart rate (no watch)", "a" if hr else "n", dim=not hr),
        tile("pocket", ("In pocket" if rec.get("in_pocket") else "In hand / out") if live else "—",
             f'Phone position · {int(rec.get("light_lux") or 0)} lux', "n", dim=d),
        tile("battery", f'{int(batt)}<small>%{" charging" if rec.get("charging") else ""}</small>'
             if batt is not None else "—", "Phone battery",
             "a" if batt is not None and batt < 10 else ("w" if batt is not None and batt < 20 else "")),
    ]
    tiles_html = f'<div class="pa-tiles">{"".join(tiles)}</div>'

    pace_word, pace_tone = _pace_status(cad, motion) if live else ("No signal", "n")
    g1 = gauge_card("Walking pace", gauge_svg(cad if live else None, 0, 140, ticks=(0, 70, 140),
                                              fill=TEAL if pace_tone == "ok" else AMBER),
                    f'{int(cad)}<small style="font-size:0.9rem;color:var(--muted)"> steps/min</small>'
                    if live and cad else "—", pace_word, pace_tone,
                    (f'Their usual pace: {model["pace"]["low"]}–{model["pace"]["high"]} steps a minute (from training).'
                     if model and model.get("pace") else "Most adults walk about 90–120 steps a minute."))
    if not live:
        fw, ft = "No signal", "n"
    elif fi >= FI_FREEZE and rec.get("freeze_suspect"):
        fw, ft = "Freeze-like", "a"
    elif fi >= FI_WATCH:
        fw, ft = "Unsteady", "w"
    else:
        fw, ft = "Steady", "ok"
    g2 = gauge_card("Freeze score", gauge_svg(min(fi, 5) if live else None, 0, 5,
                                              bands=[(0, FI_WATCH, "#7DBDB3"), (FI_WATCH, FI_FREEZE, "#E3A857"),
                                                     (FI_FREEZE, 5, "#C25450")], ticks=(0, 2, 5)),
                    f'{fi:.1f}' if live else "—", fw, ft,
                    "Trembling in place vs. normal stepping. 2 or more while stopped looks like a freeze.")
    bt = "n" if batt is None else ("a" if batt < 10 else "w" if batt < 20 else "ok")
    g3 = gauge_card("Phone battery", gauge_svg(batt, 0, 100, ticks=(0, 50, 100),
                                               fill={"a": RED, "w": AMBER}.get(bt, TEAL)),
                    f'{int(batt)}%' if batt is not None else "—",
                    {"a": "Charge now", "w": "Low", "ok": "Charging" if rec.get("charging") else "OK",
                     "n": "Unknown"}[bt], bt,
                    "StepCue stops sensing if the phone runs out.")
    st.markdown(f'<div class="pa-gauges">{g1}{g2}{g3}</div>{tiles_html}', unsafe_allow_html=True)

    c1, c2 = st.columns([7, 5], gap="medium")
    with c1, st.container(border=True):
        st.markdown('<div class="sc-h">Walking over the last 3 hours</div>'
                    '<div class="sc-sub">One point per minute from the phone. Gaps mean the phone was not sharing.</div>',
                    unsafe_allow_html=True)
        if mins:
            st.altair_chart(minutes_chart(mins), width="stretch")
        else:
            st.caption("Collecting. The first minute appears shortly after the phone starts sharing.")
    with c2, st.container(border=True):
        have = sum(1 for x in days if x["steps"] is not None)
        st.markdown('<div class="sc-h">Steps, last 7 days</div>'
                    f'<div class="sc-sub">Counted by the phone&#39;s step sensor. '
                    f'{"Only today so far: older days fill in as the phone keeps sharing." if have <= 1 else ""}</div>',
                    unsafe_allow_html=True)
        st.altair_chart(steps_chart(days), width="stretch")
        st.markdown(f'<div class="sc-sub" style="margin:0"><b style="color:#7A4300">Dashed line: daily goal '
                    f'{STEP_GOAL:,} steps</b>, set by the care team '
                    '(STEPCUE_STEP_GOAL). Grey "no data" means the phone wasn&#39;t sharing that day.</div>',
                    unsafe_allow_html=True)

    chips = "".join(f"<span>{esc(s)}</span>" for s in rec.get("sensors") or [])
    st.markdown(f'<div class="sc-sub" style="margin:10px 0 0">Sensors in use</div><div class="pa-chips">{chips}'
                f'<span>GPS (stays on this laptop)</span><span>battery</span></div>'
                '<div class="sc-foot">Movement descriptions from a phone, to help the family and care team notice '
                'patterns. Not a medical device; it does not diagnose or give medical advice.</div>',
                unsafe_allow_html=True)
