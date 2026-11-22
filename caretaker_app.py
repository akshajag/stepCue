"""StepCue caregiver dashboard (presentation layer).

  streamlit run caretaker_app.py

Polls state.json every 0.5 s. Leads with one answer, "is my person OK and do I
need to do anything?", then shows recent episodes, the live Freeze Index, time
spent walking / standing / frozen, and what the Nemotron care agent changed and
why. Layout and status rules are documented in DESIGN.md; the logic lives in
dashboard_model.py (unit tested).

Assistive cueing and awareness only: shows mobility patterns, not a medical device.
"""
import html
import json
import os
import socket
import threading
import time

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

import alert_notifier
import auth
import auth_ui
import companion_store as cs
import nvidia_ui
import phone_ui
import training_ui
from config import (CUE_PORT, ESP32_IP, PROLONGED_FREEZE_SEC, SIM_CONTROL_PORT,
                    SIM_IP)
from dashboard_model import (decision_summary, episode_stats, freshness,
                             movement_summary, screen_state, time_in_state)
from location_io import LOCATION_FILE, read_location, read_places, write_json
from location_model import location_view
from state_io import read_state, reset_state, state_age_sec

ADVISOR_LOG = "advisor_log.jsonl"
CHART_SEC = 60
SUMMARY_SEC = 600

STATUS_WORD = {"WALKING": "Walking", "STATIONARY": "Standing", "FREEZE": "Frozen", "OFFLINE": "No signal"}
BAR_COLORS = {"WALKING": "#0F766E", "STATIONARY": "#B8C2CE", "FREEZE": "#A35A00", "OFFLINE": "#E2E7ED"}

# Simple stroke icons (no emoji), drawn in currentColor.
_ICON = {
    "steady": '<path d="M20 6 9 17l-5-5"/>',
    "freeze": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "alert": '<path d="M12 3 2 21h20L12 3z"/><path d="M12 10v5"/><path d="M12 18h.01"/>',
    "watch": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "offline": '<path d="M2 8.5a15 15 0 0 1 20 0"/><path d="M5.5 12a10 10 0 0 1 13 0"/>'
               '<path d="M9 15.5a5 5 0 0 1 6 0"/><path d="M12 19h.01"/><path d="M3 3l18 18"/>',
    "pin": '<path d="M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    "route": '<circle cx="6" cy="19" r="2.5"/><circle cx="18" cy="5" r="2.5"/>'
             '<path d="M8.5 19H16a3.5 3.5 0 0 0 0-7H8a3.5 3.5 0 0 1 0-7h7.5"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "arrive": '<path d="M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
    "leave": '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
    "heart": '<path d="M12 20s-7-4.4-9-9.1C1.7 7.6 4 4.5 7.2 4.5c2 0 3.4 1.1 4.8 2.9 1.4-1.8 2.8-2.9 4.8-2.9 3.2 0 5.5 3.1 4.2 6.4C19 15.6 12 20 12 20z"/>',
    "phone": '<rect x="7" y="2.5" width="10" height="19" rx="2.5"/><path d="M11 18h2"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.7 2.7L16 10"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5"/><path d="M12 16.5h.01"/>',
}


def icon(name, size=34):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON[name]}</svg>')


st.set_page_config(page_title="StepCue", layout="wide")
alert_notifier.load_dotenv()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
:root {
  --ink:#16202B; --muted:#556274; --bg:#F4F6F8; --surface:#FFFFFF; --line:#E1E6EC;
  --ok:#0F766E; --ok-tint:#E3F2EE; --watch:#A35A00; --watch-tint:#FBEFD9;
  --alert:#9B1C1C; --offline:#475569; --offline-tint:#EDF0F4;
  --font:'Atkinson Hyperlegible', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp {background:var(--bg);}
.block-container {padding-top:3.4rem; padding-bottom:2rem; max-width:1240px;}
[data-testid="stHeader"] {background:transparent;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility:hidden;}
.stApp, .stMarkdown, .stButton button, label, input {font-family:var(--font);}

.sc-top {display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px;}
.sc-brand {font:700 1.55rem var(--font); color:var(--ink); letter-spacing:-0.01em;}
.sc-brand small {font-weight:400; font-size:0.95rem; color:var(--muted); margin-left:10px;}
.sc-fresh {font:400 0.92rem var(--font); color:var(--muted); display:flex; align-items:center; gap:8px;
           font-variant-numeric:tabular-nums; white-space:nowrap;}
.sc-fresh i {width:9px; height:9px; border-radius:50%; background:var(--ok); display:inline-block;}
.sc-fresh.stale i {background:#B8C2CE;}

.sc-banner {display:flex; align-items:center; gap:22px; border-radius:18px; padding:24px 28px;
            border:1.5px solid transparent; font-family:var(--font);}
.sc-banner .ic {flex:none; width:64px; height:64px; border-radius:50%; display:grid; place-items:center;}
.sc-banner h1 {font:700 clamp(1.8rem,3.4vw,2.6rem)/1.1 var(--font); margin:0; padding:0; letter-spacing:-0.015em;}
.sc-banner p {font-size:1.1rem; margin:6px 0 0 0; line-height:1.4;}
.sc-banner .now {margin-left:auto; text-align:right; flex:none;}
.sc-banner .now span {display:block; font-size:0.9rem; opacity:0.85;}
.sc-banner .now b {font-size:1.35rem;}
.b-steady {background:var(--ok-tint); color:var(--ok); border-color:#CBE6DF;}
.b-steady h1, .b-steady p {color:var(--ink);} .b-steady .ic {background:#fff; color:var(--ok);}
.b-freeze, .b-watch {background:var(--watch-tint); border-color:#F0D9AE;}
.b-freeze h1, .b-watch h1 {color:#7A4300;} .b-freeze p, .b-watch p, .b-freeze .now, .b-watch .now {color:var(--ink);}
.b-freeze .ic, .b-watch .ic {background:#fff; color:var(--watch);}
.b-alert {background:var(--alert); color:#fff;} .b-alert h1, .b-alert p {color:#fff;}
.b-alert .ic {background:rgba(255,255,255,0.16); color:#fff;}
.b-offline {background:var(--offline-tint); border:1.5px dashed #AEB8C5;}
.b-offline h1 {color:var(--offline);} .b-offline p, .b-offline .now {color:var(--ink);}
.b-offline .ic {background:#fff; color:var(--offline);}
.pulse {animation:scpulse 1.2s ease-in-out infinite;}
@keyframes scpulse {0%,100% {box-shadow:0 0 0 0 rgba(255,255,255,0.0);} 50% {box-shadow:0 0 0 10px rgba(163,90,0,0.18);}}
.b-alert .pulse {animation:scpulse-a 1.2s ease-in-out infinite;}
@keyframes scpulse-a {0%,100% {box-shadow:0 0 0 0 rgba(255,255,255,0);} 50% {box-shadow:0 0 0 12px rgba(255,255,255,0.22);}}
@media (prefers-reduced-motion: reduce) {.pulse, .b-alert .pulse {animation:none;}}

.sc-metrics {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin:14px 0 4px 0;}
.sc-metric {background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:16px 18px;
            font-family:var(--font);}
.sc-metric .k {font-size:0.92rem; color:var(--muted);}
.sc-metric .v {font-size:2rem; font-weight:700; color:var(--ink); line-height:1.2;
               font-variant-numeric:tabular-nums; white-space:nowrap;}
.sc-metric .v small {font-size:1rem; font-weight:400; color:var(--muted); margin-left:4px;}
.sc-metric .s {font-size:0.92rem; color:var(--muted); margin-top:2px;}
.sc-metric .s b {color:var(--ok); font-weight:700;} .sc-metric .s b.slower {color:var(--watch);}
.dim .v, .dim .s {color:#9AA5B3 !important;}

.sc-h {font:700 1.08rem var(--font); color:var(--ink); margin:0 0 2px 0;}
.sc-sub {font:400 0.9rem var(--font); color:var(--muted); margin:0 0 10px 0;}
[data-testid="stVerticalBlockBorderWrapper"] {background:var(--surface); border-color:var(--line) !important;
                                               border-radius:16px !important;}

.sc-bar {display:flex; height:14px; border-radius:7px; overflow:hidden; background:#EEF1F4; margin:6px 0 10px 0;}
.sc-bar div {height:100%;}
.sc-legend {display:flex; flex-wrap:wrap; gap:6px 18px; font:400 0.92rem var(--font); color:var(--ink);}
.sc-legend span i {display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px;}
.sc-legend b {font-variant-numeric:tabular-nums;}
.sc-move {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:10px; margin-top:16px;
  padding-top:14px; border-top:1px solid var(--line);}
.sc-move .k {font-size:0.88rem; color:var(--muted);}
.sc-move .v {font-size:1.35rem; font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums;}
.sc-move .d {font-size:0.85rem; color:var(--muted); min-height:1.1em;}
.sc-move-note {margin-top:10px; font-size:0.95rem; color:var(--ink);}
.sc-move-note i {display:inline-block; width:8px; height:8px; border-radius:4px; background:var(--watch);
  margin-right:8px; vertical-align:middle;}

.sc-agent-src {font:400 0.88rem var(--font); color:var(--muted); display:flex; align-items:center; gap:7px;}
.sc-agent-src i {width:8px; height:8px; border-radius:50%; background:#76B900; display:inline-block;}
.sc-agent-src.rules i {background:#B8C2CE;}
.sc-brief {font:400 1.16rem/1.5 var(--font); color:var(--ink); margin:10px 0 16px 0; padding:14px 16px;
           background:#F7F9FA; border-radius:12px;}
.sc-dec {border-top:1px solid var(--line); padding:12px 0 10px 0; font-family:var(--font);}
.sc-dec .t {font-size:0.85rem; color:var(--muted); font-variant-numeric:tabular-nums;}
.sc-dec .c {font-size:1rem; font-weight:700; color:var(--ink); margin:3px 0;}
.sc-dec .w {font-size:0.93rem; color:#3B4756; line-height:1.4;}
.sc-tag {display:inline-block; font-size:0.78rem; padding:1px 8px; border-radius:999px; margin-left:6px;
         background:#EEF1F4; color:var(--muted);}
.sc-tag.alert {background:#F8E1E1; color:var(--alert);} .sc-tag.watch {background:var(--watch-tint); color:#7A4300;}
.sc-alerts {border-top:1px solid var(--line); margin-top:8px; padding-top:12px; font-family:var(--font);}
.sc-alerts .st {font-size:0.88rem; color:var(--muted); display:flex; flex-wrap:wrap; gap:6px 14px;}
.sc-alerts .st span i {display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;
  background:#B8C2CE;}
.sc-alerts .st span.on i {background:var(--ok);}
.sc-alerts .n {font-size:0.93rem; color:var(--ink); margin-top:6px;}
.sc-alerts .n small {color:var(--muted); font-variant-numeric:tabular-nums;}
.sc-empty {font:400 0.98rem/1.5 var(--font); color:var(--muted); padding:8px 0;}
.sc-foot {font:400 0.85rem var(--font); color:var(--muted); margin-top:18px;}

.sc-kpis {display:grid; grid-template-columns:repeat(auto-fit, minmax(200px,1fr)); gap:14px; margin:0 0 14px 0;}
.sc-kpi {background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:16px 18px;
  display:flex; gap:14px; align-items:center; font-family:var(--font); min-width:0;}
.sc-kpi .ico {flex:none; width:46px; height:46px; border-radius:12px; display:grid; place-items:center;
  background:var(--ok-tint); color:var(--ok);}
.sc-kpi .ico.w {background:var(--watch-tint); color:var(--watch);}
.sc-kpi .ico.n {background:#EEF1F4; color:var(--offline);}
.sc-kpi .v {font-size:1.6rem; font-weight:700; color:var(--ink); line-height:1.15; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; font-variant-numeric:tabular-nums;}
.sc-kpi .v.wrap {white-space:normal; font-size:1.3rem; line-height:1.2;}
.sc-kpi .v small {font-size:0.95rem; font-weight:400; color:var(--muted); margin-left:3px;}
.sc-kpi .k {font-size:0.9rem; color:var(--muted);}
.sc-chip {display:inline-flex; align-items:center; gap:8px; font:700 0.92rem var(--font); color:var(--ink);
  background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:5px 12px; margin-bottom:8px;}
.sc-chip i {width:9px; height:9px; border-radius:50%; background:var(--ok); display:inline-block;}
.sc-chip.recent i {background:var(--watch);} .sc-chip.stale i, .sc-chip.none i {background:#B8C2CE;}
.sc-chip span {font-weight:400; color:var(--muted);}
.sc-zone {display:grid; grid-template-columns:minmax(0,9rem) 1fr auto; gap:12px; align-items:center;
  font:400 0.98rem var(--font); color:var(--ink); padding:7px 0;}
.sc-zone .track {height:10px; background:#EEF1F4; border-radius:5px; overflow:hidden;}
.sc-zone .track div {height:100%; background:var(--watch); border-radius:5px;}
.sc-zone b {font-variant-numeric:tabular-nums;}
.sc-here {background:#F7F9FA; border-radius:12px; padding:12px 14px; margin:8px 0 14px 0; font-family:var(--font);}
.sc-here .k {font-size:0.85rem; color:var(--muted);} .sc-here .v {font-size:1.2rem; font-weight:700; color:var(--ink);}
.sc-tl {display:grid; grid-auto-flow:column; grid-auto-columns:minmax(120px,1fr); overflow-x:auto;
  padding:6px 0 4px 0; font-family:var(--font);}
.sc-tl .ev {text-align:center; position:relative; padding:0 6px;}
.sc-tl .ev::before {content:""; position:absolute; top:44px; left:0; right:0; height:2px; background:var(--line);}
.sc-tl .ev:first-child::before {left:50%;} .sc-tl .ev:last-child::before {right:50%;}
.sc-tl .tm {font-size:0.85rem; color:var(--muted); font-variant-numeric:tabular-nums; height:20px;}
.sc-tl .dot {position:relative; z-index:1; margin:4px auto 8px auto; width:38px; height:38px; border-radius:50%;
  display:grid; place-items:center; background:var(--ok); color:#fff; border:3px solid var(--surface);}
.sc-tl .dot.freeze {background:var(--watch);} .sc-tl .dot.leave {background:var(--offline);}
.sc-tl .tt {font-size:0.98rem; font-weight:700; color:var(--ink);}
.sc-tl .pl {font-size:0.88rem; color:var(--muted);}

.sc-help {display:flex; align-items:center; gap:16px; background:var(--alert); color:#fff; border-radius:16px;
  padding:16px 22px; margin:0 0 14px 0; font-family:var(--font);}
.sc-help .ic {flex:none; width:48px; height:48px; border-radius:50%; display:grid; place-items:center;
  background:rgba(255,255,255,0.16);}
.sc-help b {font-size:1.35rem; display:block;} .sc-help span {font-size:1rem; opacity:0.92;}
.sc-acct {background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:10px 12px;
  font-family:var(--font); margin-bottom:8px;}
.sc-acct b {display:block; color:var(--ink);} .sc-acct span {font-size:0.88rem; color:var(--muted);}
.sc-acct .role {display:inline-block; font-size:0.78rem; padding:1px 8px; border-radius:999px; margin-top:4px;
  background:var(--ok-tint); color:var(--ok);}
.sc-comp {display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px; margin-top:6px;}
.sc-comp > div {background:#F7F9FA; border-radius:12px; padding:10px 12px; font-family:var(--font); min-width:0;}
.sc-comp .k {font-size:0.85rem; color:var(--muted); display:flex; align-items:center; gap:6px;}
.sc-comp .v {font-size:1.2rem; font-weight:700; color:var(--ink); line-height:1.25;}
.sc-comp .v small {font-size:0.9rem; font-weight:400; color:var(--muted);}
.sc-comp .d {font-size:0.85rem; color:var(--muted);}
.sc-comp .v.help {color:var(--alert);}
.sc-fam {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin:14px 0;}
.sc-recent {font-family:var(--font);}
.sc-recent .r {display:flex; gap:12px; align-items:baseline; padding:9px 0; border-top:1px solid var(--line);
  font-size:1.02rem; color:var(--ink);}
.sc-recent .r:first-child {border-top:none;}
.sc-recent .r small {color:var(--muted); font-variant-numeric:tabular-nums; min-width:3.2rem;}
.sc-recent .r span {color:var(--muted); font-size:0.92rem; margin-left:auto; text-align:right;}
.sc-stat {display:grid; grid-template-columns:repeat(auto-fit, minmax(170px,1fr)); gap:12px; margin:0 0 14px 0;}
.sc-stat > div {background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:14px 16px;
  font-family:var(--font);}
.sc-stat .k {font-size:0.88rem; color:var(--muted);}
.sc-stat .v {font-size:1.7rem; font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums;}
.sc-stat .v small {font-size:0.95rem; font-weight:400; color:var(--muted); margin-left:3px;}
.sc-stat .d {font-size:0.85rem; color:var(--muted);}
.sc-qr {background:#fff; border:1px solid var(--line); border-radius:14px; padding:10px; max-width:220px;}
.sc-qr svg {width:100%; height:auto; display:block;}

@media (max-width: 760px) {
  .sc-fam {grid-template-columns:1fr; gap:10px;}
  .sc-comp {grid-template-columns:1fr;}
  .sc-kpis {grid-template-columns:repeat(auto-fit, minmax(150px,1fr)); gap:10px;}
  .sc-kpi {padding:12px; gap:10px;} .sc-kpi .ico {width:38px; height:38px;}
  .sc-kpi .v {font-size:1.25rem;}
  .block-container {padding-left:1rem; padding-right:1rem;}
  .sc-banner {flex-wrap:wrap; padding:18px; gap:14px;}
  .sc-banner .ic {width:48px; height:48px;}
  .sc-banner .now {margin-left:0; text-align:left; width:100%;}
  .sc-metrics {grid-template-columns:1fr; gap:10px;}
  .sc-brand small {display:none;}
}
</style>
""", unsafe_allow_html=True)


def esc(value):
    return html.escape(str(value))


def send_udp(text, ip, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(text.encode("ascii"), (ip, port))
        return True
    except OSError:
        return False


def read_decisions(limit=5):
    try:
        with open(ADVISOR_LOG, encoding="utf-8") as f:
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


# ----------------------------------------------------------------- sign-in --
user = auth_ui.require_login()          # stops here until signed in (Google or password, then 2FA code)
if user.get("admin"):                     # demo admin: switch between all three dashboards
    with st.sidebar:
        view = st.segmented_control("Dashboard", ["care", "family", "patient"], default="care",
                                    format_func=lambda r: auth.ROLES[r], key="admin_view") or "care"
    user = dict(user, role=view)
CARE = user["role"] == "care"


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.markdown(f'<div class="sc-acct"><b>{esc(user.get("name") or user["email"])}</b>'
                f'<span>{esc(user["email"])}</span><br>'
                f'<span class="role">{esc(auth.ROLES.get(user["role"], user["role"]))} view</span></div>',
                unsafe_allow_html=True)
    if not user.get("demo") and st.button("Sign out", width="stretch"):
        auth_ui.sign_out()

if CARE:
  with st.sidebar:
    st.markdown("### Demo controls")
    target = st.radio("Device", ["Simulator", "ESP32 hardware"], horizontal=True)
    cue_ip = SIM_IP if target == "Simulator" else ESP32_IP

    if target == "Simulator":
        st.caption("Tell the simulated wearer what to do.")
        c1, c2 = st.columns(2)
        if c1.button("Walk", width="stretch"):
            send_udp("MODE:WALK", SIM_IP, SIM_CONTROL_PORT)
        if c2.button("Stand still", width="stretch"):
            send_udp("MODE:STAND", SIM_IP, SIM_CONTROL_PORT)
        if c1.button("Freeze", width="stretch", type="primary"):
            send_udp("MODE:FREEZE", SIM_IP, SIM_CONTROL_PORT)
        if c2.button("Long freeze", width="stretch",
                     help="Stays frozen until wrist vibration is switched on, so the "
                          f"over-{PROLONGED_FREEZE_SEC:.0f}-second path runs."):
            send_udp("MODE:FREEZE_HARD", SIM_IP, SIM_CONTROL_PORT)
        if c1.button("Turn", width="stretch", help="Walks while turning left about 180 degrees."):
            send_udp("MODE:TURN", SIM_IP, SIM_CONTROL_PORT)
            threading.Timer(2.0, send_udp, ("MODE:WALK", SIM_IP, SIM_CONTROL_PORT)).start()
        if c2.button("Freeze in a turn", width="stretch",
                     help="Starts a turn, then freezes part-way through it."):
            send_udp("MODE:TURN", SIM_IP, SIM_CONTROL_PORT)
            threading.Timer(1.6, send_udp, ("MODE:FREEZE", SIM_IP, SIM_CONTROL_PORT)).start()
    else:
        st.caption(f"Join the StepCue_AP Wi-Fi first. Cues go to {ESP32_IP}.")

    st.markdown("### Cue check")
    st.caption("Plays the cue for about 5 seconds to check the light and wrist motor.")
    t1, t2 = st.columns(2)
    test_bpm = t1.number_input("Tempo (bpm)", 60, 130, 100, 5)
    test_haptic = t2.toggle("Vibration", value=True)
    b1, b2 = st.columns(2)
    if b1.button("Play cue", width="stretch"):
        send_udp(f"CUE:{int(test_bpm)}:{1 if test_haptic else 0}", cue_ip, CUE_PORT)
    if b2.button("Stop", width="stretch"):
        send_udp("STOP", cue_ip, CUE_PORT)

    st.markdown("### Caregiver alerts")
    _cs = alert_notifier.channel_status()
    if _cs["push"]:
        st.caption(f"Push alerts go to the ntfy topic in .env ({_cs['push_server']}). "
                   + ("Calls are set up through Twilio." if _cs["call"] else "Calls are not set up."))
        if st.button("Send a test push", width="stretch"):
            recs = alert_notifier.send_test()
            st.toast(recs[0]["result"] if recs else "Nothing sent")
    else:
        st.caption("Not set up. Add NTFY_TOPIC (free phone push) and optionally the TWILIO_* keys "
                   "to .env, then run alert_notifier.py. Until then alerts are only logged.")

    st.markdown("### Session")
    if st.button("Start a new session", width="stretch",
                 help="Clears the episode log and the agent's decision history."):
        reset_state()
        write_json(LOCATION_FILE, {"fix": None, "trail": []})
        try:
            open(ADVISOR_LOG, "w").close()
        except OSError:
            pass
        st.session_state.pop("chart", None)
        st.session_state.pop("states", None)


# ------------------------------------------------------------- live view --
def remember(status, fi, now):
    chart = st.session_state.setdefault("chart", [])
    chart.append((now, fi, status))
    while chart and now - chart[0][0] > CHART_SEC:
        chart.pop(0)
    states = st.session_state.setdefault("states", [])
    if not states or states[-1][1] != status:
        states.append((now, status))
    while len(states) > 1 and states[1][0] < now - SUMMARY_SEC:
        states.pop(0)
    return chart, states


def banner(view, live_status, state, offline):
    key = view["key"]
    pulse = " pulse" if key in ("freeze", "alert") else ""
    if offline:
        now_html = f'<span>Last known</span><b>{esc(STATUS_WORD.get(live_status, live_status))}</b>'
    else:
        now_html = f'<span>Right now</span><b>{esc(STATUS_WORD.get(live_status, live_status))}</b>'
    st.markdown(
        f'<div class="sc-banner b-{key}" role="status" aria-live="polite">'
        f'<div class="ic{pulse}">{icon(key)}</div>'
        f'<div><h1>{esc(view["headline"])}</h1><p>{esc(view["detail"])}</p></div>'
        f'<div class="now">{now_html}</div></div>', unsafe_allow_html=True)


def metrics(state, stats, offline):
    n, ongoing = stats["count"], stats["count"] - stats["completed"]
    ep_sub = "None yet this session" if n == 0 else (
        f"{ongoing} in progress now" if ongoing else "All resolved")
    if stats["avg_recovery"] is None:
        rec_v, rec_sub = "–", "Shown after the first episode"
    else:
        rec_v = f'{stats["avg_recovery"]:.1f}<small>s</small>'
        cmp_ = stats["last_vs_avg"]
        last = f'Last one {stats["last_recovery"]:.1f} s'
        if cmp_ == "faster":
            rec_sub = f'{last}, <b>faster than usual</b>'
        elif cmp_ == "slower":
            rec_sub = f'{last}, <b class="slower">slower than usual</b>'
        elif cmp_ == "similar":
            rec_sub = f"{last}, about usual"
        else:
            rec_sub = last
    haptic = bool(state.get("haptic"))
    dim = " dim" if offline else ""
    st.markdown(
        f'<div class="sc-metrics">'
        f'<div class="sc-metric{dim}"><div class="k">Freeze episodes</div>'
        f'<div class="v">{n}</div><div class="s">{ep_sub}</div></div>'
        f'<div class="sc-metric{dim}"><div class="k">Typical time to walk again</div>'
        f'<div class="v">{rec_v}</div><div class="s">{rec_sub}</div></div>'
        f'<div class="sc-metric{dim}"><div class="k">Cue setting</div>'
        f'<div class="v">{int(state.get("bpm") or 0)}<small>bpm</small></div>'
        f'<div class="s">Ankle light{" and wrist vibration" if haptic else " only"}</div></div>'
        f'</div>', unsafe_allow_html=True)


def fi_chart(chart, now):
    df = pd.DataFrame({"t": [c[0] - now for c in chart], "fi": [c[1] for c in chart],
                       "status": [c[2] for c in chart]})
    spans, start = [], None
    for t, s in zip(df["t"], df["status"]):
        if s == "FREEZE" and start is None:
            start = t
        elif s != "FREEZE" and start is not None:
            spans.append({"x0": start, "x1": t})
            start = None
    if start is not None:
        spans.append({"x0": start, "x1": 0.0})
    dom = alt.Scale(domain=[-CHART_SEC, 0])
    x = alt.X("t:Q", scale=dom, axis=alt.Axis(title=None, values=[-60, -45, -30, -15, 0], grid=False,
              labelExpr="datum.value == 0 ? 'now' : (-datum.value) + ' s ago'", labelColor="#556274",
              labelFont="Atkinson Hyperlegible", labelFontSize=12, domainColor="#E1E6EC", tickColor="#E1E6EC"))
    y = alt.Y("fi:Q", scale=alt.Scale(domain=[0, max(4.0, float(df["fi"].max() if len(df) else 0) * 1.15)]),
              axis=alt.Axis(title=None, tickCount=4, gridColor="#EEF1F4", domain=False, ticks=False,
                            labelColor="#556274", labelFontSize=12))
    layers = []
    if spans:
        layers.append(alt.Chart(pd.DataFrame(spans)).mark_rect(color="#A35A00", opacity=0.14).encode(
            x=alt.X("x0:Q", scale=dom), x2="x1:Q"))
    layers.append(alt.Chart(df).mark_line(color="#0F766E", strokeWidth=2.4, interpolate="monotone").encode(x=x, y=y))
    st.altair_chart(alt.layer(*layers).properties(height=210).configure_view(strokeWidth=0),
                    width="stretch")


def state_bar(states, now):
    frac = time_in_state(states, now, SUMMARY_SEC)
    if not frac:
        st.markdown('<div class="sc-empty">Collecting data...</div>', unsafe_allow_html=True)
        return
    order = ["WALKING", "STATIONARY", "FREEZE", "OFFLINE"]
    segs = "".join(f'<div style="width:{frac[k] * 100:.2f}%;background:{BAR_COLORS[k]}" '
                   f'title="{STATUS_WORD[k]}"></div>' for k in order if frac.get(k))
    legend = "".join(f'<span><i style="background:{BAR_COLORS[k]}"></i>{STATUS_WORD[k]} '
                     f'<b>{round(frac[k] * 100)}%</b></span>' for k in order if frac.get(k, 0) >= 0.005)
    span_min = min(SUMMARY_SEC, now - states[0][0]) / 60
    label = "last 10 minutes" if span_min >= 9.9 else (
        f"last {max(1, round(span_min))} minute{'s' if round(span_min) > 1 else ''}")
    st.markdown(f'<div class="sc-h" style="margin-top:6px">Time in each state, {label}</div>'
                f'<div class="sc-bar">{segs}</div><div class="sc-legend">{legend}</div>',
                unsafe_allow_html=True)


def movement(state, offline):
    m = movement_summary(state)
    if m is None:
        return
    dim = ' style="opacity:.55"' if offline else ""
    cells = "".join(f'<div><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
                    f'<div class="d">{esc(d)}</div></div>' for k, v, d in m["items"])
    note = (f'<div class="sc-move-note"><i></i>{esc(m["turn_pattern"])}</div>'
            if m["turn_pattern"] else "")
    st.markdown(f'<div class="sc-move"{dim}>{cells}</div>{note}', unsafe_allow_html=True)


def agent_panel(state, decisions):
    rules = bool(decisions) and decisions[0].get("source") == "rules"
    if not decisions:
        src = '<div class="sc-agent-src rules"><i></i>Waiting for the first episode</div>'
    elif rules:
        src = ('<div class="sc-agent-src rules"><i></i>Nemotron not connected, '
               'using built-in rules</div>')
    else:
        model = decisions[0].get("source", "").split("/")[-1]
        src = f'<div class="sc-agent-src"><i></i>NVIDIA Nemotron, {esc(model)}</div>'
    st.markdown(f'<div class="sc-h">Care agent</div>{src}'
                f'<div class="sc-brief">{esc(state.get("caretaker_brief", ""))}</div>',
                unsafe_allow_html=True)
    if not decisions:
        st.markdown('<div class="sc-empty">After each freeze the agent reviews how quickly walking '
                    'resumed and tunes the cue tempo and wrist vibration. Its decisions and reasons '
                    'appear here.</div>', unsafe_allow_html=True)
        return
    for d in decisions[:2]:
        level = d.get("decision", {}).get("alert_level", "normal")
        tag = {"alert": '<span class="sc-tag alert">check in</span>',
               "watch": '<span class="sc-tag watch">keep an eye out</span>'}.get(level, "")
        trig = "Long freeze" if d.get("trigger") == "prolonged_freeze" else "Episode review"
        who = "rules" if d.get("source") == "rules" else "Nemotron"
        why = d.get("decision", {}).get("reasoning", "")
        st.markdown(
            f'<div class="sc-dec"><div class="t">{esc(d.get("time", ""))}, {trig}, {who} in '
            f'{d.get("latency_sec", 0):.1f} s{tag}</div>'
            + (nvidia_ui.chain(d.get("tools"), d.get("guard"), d.get("source")) if d.get("tools") else "")
            + f'<div class="c">{esc(decision_summary(d))}</div>'
            + (f'<div class="w">{esc(why)}</div>' if why else "") + '</div>',
            unsafe_allow_html=True)


def alerts_block():
    cs = alert_notifier.channel_status()
    push = '<span class="on"><i></i>Phone push on</span>' if cs["push"] else '<span><i></i>Phone push not set up</span>'
    call = '<span class="on"><i></i>Phone call on</span>' if cs["call"] else '<span><i></i>Phone call not set up</span>'
    rows = ""
    for n in alert_notifier.read_notifications(2):
        how = "Call" if n.get("channel") == "call" else "Push"
        res = n.get("result", "")
        note = " (not sent, dry run)" if res.startswith("dry run") else (" (failed)" if res.startswith("failed") else "")
        rows += (f'<div class="n"><small>{esc(n.get("time", ""))}</small> {how}: '
                 f'{esc(n.get("title", "").replace("StepCue: ", "").capitalize())}{esc(note)}</div>')
    st.markdown(f'<div class="sc-alerts"><div class="sc-h" style="font-size:1rem">Caregiver alerts</div>'
                f'<div class="st">{push}{call}</div>{rows}</div>', unsafe_allow_html=True)


def episode_log(episodes, companion=None):
    st.markdown('<div class="sc-h">Episode log</div><div class="sc-sub">Newest first. "Checked by" is the '
                'wearer&#39;s own answer on their phone, or what the phone&#39;s motion showed.</div>',
                unsafe_allow_html=True)
    now = time.time()
    if not episodes:
        st.markdown('<div class="sc-empty">No freeze episodes recorded this session.</div>',
                    unsafe_allow_html=True)
        return
    rows = [{
        "Time": e.get("timestamp", ""),
        "Frozen for": f'{e.get("duration_sec") or 0:.1f} s',
        "Walking again after": ("still frozen" if e.get("recovery_sec") is None
                                else f'{e["recovery_sec"]:.1f} s'),
        "Steps/min before": e.get("pre_freeze_cadence"),
        "In a turn": {True: "yes", False: "no"}.get(e.get("during_turn"), ""),
        "Cue (bpm)": e.get("cue_bpm"),
        "Checked by": ("" if companion is None or e.get("recovery_sec") is None
                       else cs.verification(e, companion, now)[1]),
        "Agent review": "in progress" if e.get("recovery_sec") is None
                        else ("done" if e.get("analyzed") else "pending"),
    } for e in episodes[::-1]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 height=min(300, 38 + 35 * len(rows)))


# ---------------------------------------------------------- location view --
def hex_rgb(h, a=255):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def location_map(view, places):
    # Centre on the wearer's places so the map does not jump on every refresh;
    # fall back to the GPS trail when no places are set up.
    pts = [(p["lat"], p["lon"]) for p in places] or [(p[0], p[1]) for p in view["trail"][-300:]]
    if not pts and view["fix"]:
        pts.append((view["fix"]["lat"], view["fix"]["lon"]))
    if not pts:
        st.markdown('<div class="sc-empty">No places or GPS fixes yet. Start location_server.py '
                    '(or add --simulate for the demo).</div>', unsafe_allow_html=True)
        return
    lat0 = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2
    lon0 = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2
    layers = [
        pdk.Layer("ScatterplotLayer", data=[dict(p, label=p["name"]) for p in places],
                  get_position="[lon, lat]", get_radius="radius_m",
                  get_fill_color=hex_rgb("#0F766E", 40), get_line_color=hex_rgb("#0F766E", 160),
                  stroked=True, line_width_min_pixels=1.5, pickable=True),
        pdk.Layer("TextLayer", data=places, get_position="[lon, lat]", get_text="name",
                  get_size=13, get_color=hex_rgb("#16202B"), get_pixel_offset=[0, -18],
                  font_family="Atkinson Hyperlegible, sans-serif", font_weight=700,
                  outline_width=3, outline_color=[255, 255, 255, 230], font_settings={"sdf": True}),
    ]
    if len(view["trail"]) > 1:
        layers.append(pdk.Layer("PathLayer", data=[{"path": [[p[1], p[0]] for p in view["trail"]],
                                                    "label": "Today's walk"}],
                                get_path="path", get_color=hex_rgb("#1D6FD8", 220), width_min_pixels=4,
                                cap_rounded=True, joint_rounded=True, pickable=True))
    if view["spots"]:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=[dict(s, label=f"Freeze {s['duration_sec']:.1f} s at {s['time']}, {s['place']}"
                          + (" (in a turn)" if s["during_turn"] else "")) for s in view["spots"]],
            get_position="[lon, lat]", get_radius=1, radius_min_pixels=8, radius_max_pixels=8,
            get_fill_color=hex_rgb("#A35A00"), get_line_color=[255, 255, 255, 255],
            stroked=True, line_width_min_pixels=2.5, pickable=True))
    if view["fix"]:
        f = view["fix"]
        fresh = view["freshness"]["key"] == "live"
        layers += [
            pdk.Layer("ScatterplotLayer", data=[{"lat": f["lat"], "lon": f["lon"], "r": f.get("acc") or 10,
                                                 "label": "GPS accuracy"}],
                      get_position="[lon, lat]", get_radius="r",
                      get_fill_color=hex_rgb("#1D6FD8", 45 if fresh else 20)),
            pdk.Layer("ScatterplotLayer", data=[{"lat": f["lat"], "lon": f["lon"],
                                                 "label": "Now" if fresh else "Last known position"}],
                      get_position="[lon, lat]", get_radius=1, radius_min_pixels=9, radius_max_pixels=9,
                      get_fill_color=hex_rgb("#1D6FD8" if fresh else "#8A96A6"),
                      get_line_color=[255, 255, 255, 255], stroked=True, line_width_min_pixels=3,
                      pickable=True),
        ]
    deck = pdk.Deck(layers=layers, map_provider="carto", map_style=pdk.map_styles.CARTO_ROAD,
                    initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=15.3),
                    tooltip={"text": "{label}"})
    st.pydeck_chart(deck, height=430)


def kpi(ico, value, label, tone="", wrap=False):
    return (f'<div class="sc-kpi"><div class="ico {tone}">{icon(ico, 24)}</div><div style="min-width:0">'
            f'<div class="v{" wrap" if wrap else ""}">{value}</div><div class="k">{esc(label)}</div></div></div>')


@st.fragment(run_every=3)
def location_page():
    state = read_state()
    places = read_places()
    view = location_view(read_location(), places, state.get("episodes", []))
    fr = view["freshness"]

    st.markdown(f'<div class="sc-top"><div class="sc-brand">StepCue<small>Where they are</small></div>'
                f'<div class="sc-fresh{"" if fr["key"] == "live" else " stale"}"><i></i>{esc(fr["text"])}</div></div>',
                unsafe_allow_html=True)
    n_out = len(view["spots"])
    away = view["away_min"]
    away_v = f'{away / 60:.1f}<small>h</small>' if away >= 90 else f'{away:.0f}<small>min</small>'
    st.markdown('<div class="sc-kpis">'
                + kpi("pin", esc(view["here"] or "Unknown"), "Where now", "" if fr["key"] == "live" else "n",
                      wrap=True)
                + kpi("route", f'{view["distance_km"]:.2f}<small>km</small>', "Walked outdoors today")
                + kpi("clock", away_v, "Away from home today")
                + kpi("freeze", str(n_out), "Freezes on the map", "w" if n_out else "n")
                + '</div>', unsafe_allow_html=True)

    left, right = st.columns([1.75, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown(f'<div class="sc-chip {fr["key"]}"><i></i>{esc(fr["text"])}'
                        f'<span>· {len(view["trail"])} fixes today</span></div>', unsafe_allow_html=True)
            location_map(view, places)
            st.markdown('<div class="sc-sub" style="margin:8px 0 0 0">Blue line: today&#39;s walk. '
                        'Orange dots: where freezes happened (hover for details). Map data &copy; OpenStreetMap '
                        'contributors, &copy; CARTO.</div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="sc-h">Where freezes happen</div>'
                        '<div class="sc-sub">Freezes today, by place</div>', unsafe_allow_html=True)
            if view["freezes_by_place"]:
                rows = "".join(
                    f'<div class="sc-zone"><span>{esc(z["place"])}</span>'
                    f'<div class="track"><div style="width:{z["share"] * 100:.0f}%"></div></div>'
                    f'<b>{z["count"]}</b></div>' for z in view["freezes_by_place"])
                st.markdown(rows, unsafe_allow_html=True)
            else:
                st.markdown('<div class="sc-empty">No freezes matched to a location yet.</div>',
                            unsafe_allow_html=True)
            visited = ", ".join(view["places_visited"]) or "None yet"
            st.markdown(f'<div class="sc-here"><div class="k">Places visited today</div>'
                        f'<div class="v">{esc(visited)}</div></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="sc-h">Today&#39;s timeline</div><div class="sc-sub">Latest six events</div>',
                    unsafe_allow_html=True)
        if view["timeline"]:
            evs = "".join(
                f'<div class="ev"><div class="tm">{esc(e["time"])}</div>'
                f'<div class="dot {e["kind"]}">{icon(e["kind"], 18)}</div>'
                f'<div class="tt">{esc(e["title"])}</div><div class="pl">{esc(e["place"])}</div></div>'
                for e in view["timeline"])
            st.markdown(f'<div class="sc-tl">{evs}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sc-empty">Arrivals, departures and freezes appear here as the day goes on.'
                        '</div>', unsafe_allow_html=True)

    st.markdown('<div class="sc-foot">Location comes from the wearer&#39;s phone and stays on this laptop. '
                'StepCue shows mobility patterns for caregiver awareness and does not diagnose or advise on care.'
                '</div>', unsafe_allow_html=True)


# ------------------------------------------------------- companion pieces --
def help_banner(summary):
    if summary["help_open"]:
        st.markdown(f'<div class="sc-help" role="alert"><div class="ic">{icon("help", 28)}</div>'
                    f'<div><b>Help requested from their phone</b>'
                    f'<span>Tapped "I need help" {esc(summary["checkin"]["ago"])}. Please check in now.</span></div></div>',
                    unsafe_allow_html=True)


def companion_cells(summary, with_phone=True):
    ck, feel, hr, steps, ph = (summary["checkin"], summary["walking_feel"], summary["heart_rate"],
                               summary["steps"], summary["phone"])
    cells = []
    if ck:
        cells.append(("check", "Last check-in", f'<span class="v{" help" if ck["kind"] == "help" else ""}">'
                      f'{esc("Needs help" if ck["kind"] == "help" else "I am OK")}</span>', ck["ago"]))
    else:
        cells.append(("check", "Last check-in", '<span class="v"><small>None yet</small></span>', ""))
    cells.append(("steady", "Walking today feels",
                  f'<span class="v">{esc(feel["value"].capitalize())}</span>' if feel
                  else '<span class="v"><small>Not answered</small></span>', feel["ago"] if feel else ""))
    cells.append(("heart", "Heart rate (watch)",
                  f'<span class="v">{hr["bpm"]}<small> bpm</small></span>' if hr
                  else '<span class="v"><small>No watch data</small></span>',
                  f'{hr["ago"]}, {hr["source"]}' if hr else ""))
    an = summary.get("android")
    if with_phone and an:
        bits = [f'{int(an["cadence"])} steps/min' if an.get("cadence") else None,
                "turning" if an.get("turning") else None,
                f'stairs {an["stairs"]}' if an.get("stairs") in ("up", "down") else None,
                f'battery {int(an["battery"])}%' if an.get("battery") is not None else None]
        word = "Felt a freeze" if an.get("freeze_suspect") else an["motion"].capitalize()
        cells.append(("phone", "Android phone",
                      f'<span class="v{" help" if an.get("freeze_suspect") else ""}">{esc(word)}</span>',
                      ", ".join([an["ago"]] + [b for b in bits if b]) + f' · {an.get("sensors", 0)} sensors'))
    elif with_phone:
        cells.append(("phone", "Phone motion",
                      f'<span class="v">{esc(ph["motion"].capitalize())}</span>' if ph
                      else '<span class="v"><small>Not sharing</small></span>',
                      ph["ago"] if ph else (f'Watch steps today: {steps["count"]:,}' if steps else "")))
    return "".join(f'<div><div class="k">{icon(i, 15)}{esc(k)}</div><div>{v}</div><div class="d">{esc(d)}</div></div>'
                   for i, k, v, d in cells)


def companion_panel(summary):
    v = summary["verified"]
    if v["total"]:
        checked = v["confirmed"] + v["denied"] + v["phone_agrees"] + v["phone_walking"]
        line = (f'{checked} of {v["total"]} freezes double-checked: {v["confirmed"]} confirmed by the wearer, '
                f'{v["phone_agrees"]} matched by the phone'
                + (f', {v["denied"]} the wearer said were not freezes' if v["denied"] else "") + ".")
    else:
        line = "Freezes are double-checked here once the wearer answers on their phone."
    st.markdown(f'<div class="sc-alerts"><div class="sc-h" style="font-size:1rem">From their phone and watch</div>'
                f'<div class="sc-comp">{companion_cells(summary)}</div>'
                f'<div class="sc-sub" style="margin:8px 0 0 0">{esc(line)}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------- care: overview --
@st.fragment(run_every=0.5)
def live():
    state = read_state()
    age = state_age_sec()
    view = screen_state(state, age)
    offline = view["key"] == "offline"
    live_status = state.get("status", "STATIONARY")
    now = time.time()
    chart, states = remember("OFFLINE" if offline else live_status,
                             0.0 if offline else float(state.get("freeze_index") or 0.0), now)
    episodes = state.get("episodes", [])
    stats = episode_stats(episodes)
    comp = cs.read_companion()
    csum = cs.companion_summary(comp, episodes, now)

    stale = " stale" if offline or age == float("inf") else ""
    st.markdown(f'<div class="sc-top"><div class="sc-brand">StepCue<small>Care team view</small></div>'
                f'<div class="sc-fresh{stale}"><i></i>{esc(freshness(age))}</div></div>',
                unsafe_allow_html=True)
    help_banner(csum)
    banner(view, live_status, state, offline)
    metrics(state, stats, offline)

    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="sc-h">Freeze Index over the last minute</div>'
                        '<div class="sc-sub">Leg trembling compared with stepping. Shaded where a freeze '
                        'was detected and the cue played.</div>', unsafe_allow_html=True)
            fi_chart(chart, now)
            state_bar(states, now)
            movement(state, offline)
    with right:
        with st.container(border=True):
            agent_panel(state, read_decisions())
            companion_panel(csum)
            alerts_block()

    with st.container(border=True):
        episode_log(episodes, comp)

    st.markdown('<div class="sc-foot">StepCue is an assistive cueing prototype. It tracks mobility '
                'patterns for caregiver awareness and does not diagnose or advise on care.</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------- family: today --
FAMILY_HEADLINE = {"steady": "Doing fine", "freeze": "Having a freeze, the cue is helping",
                   "watch": "Doing OK, a few freezes", "alert": "Please check in",
                   "offline": "The sensor is not connected"}


@st.fragment(run_every=2)
def family_today():
    state = read_state()
    age = state_age_sec()
    view = screen_state(state, age)
    offline = view["key"] == "offline"
    now = time.time()
    episodes = state.get("episodes", [])
    stats = episode_stats(episodes)
    comp = cs.read_companion()
    csum = cs.companion_summary(comp, episodes, now)
    first = (user.get("name") or "").split(" ")[0]

    st.markdown(f'<div class="sc-top"><div class="sc-brand">StepCue<small>Family view'
                f'{", hi " + esc(first) if first and not user.get("demo") else ""}</small></div>'
                f'<div class="sc-fresh{" stale" if offline else ""}"><i></i>{esc(freshness(age))}</div></div>',
                unsafe_allow_html=True)
    help_banner(csum)
    fview = dict(view, headline=FAMILY_HEADLINE.get(view["key"], view["headline"]))
    if csum["help_open"] and view["key"] in ("steady", "watch"):
        fview = dict(view, key="watch", headline="Asked for help on their phone")
    banner(fview, state.get("status", "STATIONARY"), state, offline)

    last = next((e for e in reversed(episodes) if e.get("recovery_sec") is not None), None)
    last_v = "None yet" if not last else esc(str(last.get("timestamp", ""))[:5])
    last_d = "" if not last else f'Walking again after {last["recovery_sec"]:.1f} s'
    rec = "–" if stats["avg_recovery"] is None else f'{stats["avg_recovery"]:.1f}<small>s</small>'
    st.markdown(
        '<div class="sc-fam">'
        f'<div class="sc-metric"><div class="k">Freezes today</div><div class="v">{stats["count"]}</div>'
        f'<div class="s">{"None so far" if not stats["count"] else "Each one got a walking cue"}</div></div>'
        f'<div class="sc-metric"><div class="k">Last freeze</div><div class="v">{last_v}</div>'
        f'<div class="s">{esc(last_d)}</div></div>'
        f'<div class="sc-metric"><div class="k">Usual time to walk again</div><div class="v">{rec}</div>'
        f'<div class="s">After a freeze</div></div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.3, 1], gap="medium")
    with left:
        with st.container(border=True):
            hd, lg = st.columns([3, 1.2])
            hd.markdown('<div class="sc-h">How the day is going</div>', unsafe_allow_html=True)
            with lg:
                brief = nvidia_ui.brief_in_language(state.get("caretaker_brief") or "No summary yet today.", "fam")
            st.markdown(f'<div class="sc-brief">{esc(brief)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sc-h">Recent freezes</div>', unsafe_allow_html=True)
            done = [e for e in episodes if e.get("recovery_sec") is not None][-5:][::-1]
            if done:
                rows = "".join(
                    f'<div class="r"><small>{esc(str(e.get("timestamp", ""))[:5])}</small>'
                    f'Stopped for {e.get("duration_sec") or 0:.1f} s'
                    f'{" while turning" if e.get("during_turn") else ""}'
                    f'<span>{esc(cs.verification(e, comp, now)[1])}</span></div>' for e in done)
                st.markdown(f'<div class="sc-recent">{rows}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sc-empty">No freezes today.</div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="sc-h">From their phone and watch</div>'
                        '<div class="sc-sub">What they told us, and what their watch shows</div>'
                        f'<div class="sc-comp">{companion_cells(csum, with_phone=False)}</div>',
                        unsafe_allow_html=True)
            fr = location_view(read_location(), read_places(), episodes)
            where = fr["here"] or ("Out" if fr["fix"] else "Unknown")
            st.markdown(f'<div class="sc-here" style="margin-top:12px"><div class="k">Where now</div>'
                        f'<div class="v">{esc(where)}</div><div class="k">{esc(fr["freshness"]["text"])}. '
                        f'Open Location for the map.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sc-foot">If you are worried, call them or their care team. StepCue shows walking '
                'patterns for awareness only and does not diagnose or advise on care.</div>', unsafe_allow_html=True)


# ------------------------------------------------------- care: report --
def stat_cell(k, v, d=""):
    return f'<div><div class="k">{esc(k)}</div><div class="v">{v}</div><div class="d">{esc(d)}</div></div>'


def care_report():
    state = read_state()
    episodes = state.get("episodes", [])
    now = time.time()
    comp = cs.read_companion()
    done = [e for e in episodes if e.get("recovery_sec") is not None]
    stats = episode_stats(episodes)
    csum = cs.companion_summary(comp, episodes, now)
    turn = [e for e in done if e.get("during_turn")]
    cad = [e["pre_freeze_cadence"] for e in done if e.get("pre_freeze_cadence")]
    durs = [float(e.get("duration_sec") or 0) for e in done]
    decisions = read_decisions(50)
    v = csum["verified"]

    st.markdown('<div class="sc-top"><div class="sc-brand">StepCue<small>Care report, this session</small></div>'
                f'<div class="sc-fresh"><i></i>{time.strftime("%b %d, %H:%M")}</div></div>', unsafe_allow_html=True)
    help_banner(csum)
    ongoing = stats["count"] - len(done)
    st.markdown(
        '<div class="sc-stat">'
        + stat_cell("Freeze episodes", str(len(done)), f"{ongoing} in progress" if ongoing else "")
        + stat_cell("Average length", f'{sum(durs) / len(durs):.1f}<small>s</small>' if durs else "–",
                    f"Longest {max(durs):.1f} s" if durs else "")
        + stat_cell("Average time to walk again",
                    f'{stats["avg_recovery"]:.1f}<small>s</small>' if stats["avg_recovery"] is not None else "–")
        + stat_cell("During a turn", f'{round(100 * len(turn) / len(done))}<small>%</small>' if done else "–",
                    f"{len(turn)} of {len(done)}" if done else "")
        + stat_cell("Steps/min before freezing", f'{sum(cad) / len(cad):.0f}' if cad else "–")
        + stat_cell("Double-checked",
                    f'{v["confirmed"] + v["phone_agrees"]}<small>of {v["total"]}</small>' if v["total"] else "–",
                    f'{v["denied"]} marked "not a freeze" by the wearer' if v["denied"] else "By the wearer or phone")
        + '</div>', unsafe_allow_html=True)

    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="sc-h">Freeze length and recovery, by episode</div>'
                        '<div class="sc-sub">Seconds. Orange: frozen. Green: time until walking again.</div>',
                        unsafe_allow_html=True)
            if done:
                df = pd.DataFrame([{"Episode": f'#{e["id"]} {str(e.get("timestamp", ""))[:5]}', "n": i,
                                    "Frozen": float(e.get("duration_sec") or 0),
                                    "Walking again after": float(e["recovery_sec"])} for i, e in enumerate(done)])
                long = df.melt(id_vars=["Episode", "n"], var_name="What", value_name="Seconds")
                ch = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
                    x=alt.X("Episode:N", sort=alt.SortField("n"),
                            axis=alt.Axis(title=None, labelAngle=0, labelColor="#556274", labelFontSize=11)),
                    xOffset="What:N",
                    y=alt.Y("Seconds:Q", axis=alt.Axis(title=None, gridColor="#EEF1F4", domain=False, ticks=False,
                                                       labelColor="#556274")),
                    color=alt.Color("What:N", scale=alt.Scale(domain=["Frozen", "Walking again after"],
                                                              range=["#A35A00", "#0F766E"]),
                                    legend=alt.Legend(orient="top", title=None, labelColor="#16202B")),
                    tooltip=["Episode", "What", "Seconds"])
                st.altair_chart(ch.properties(height=230).configure_view(strokeWidth=0), width="stretch")
            else:
                st.markdown('<div class="sc-empty">No completed episodes yet.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="sc-h">Heart rate from the watch, last 6 hours</div>'
                        '<div class="sc-sub">Context next to freezes; not a medical reading.</div>',
                        unsafe_allow_html=True)
            hr = [p for p in comp["hr_series"] if now - p[0] < 6 * 3600]
            if hr:
                hdf = pd.DataFrame({"time": pd.to_datetime([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p[0]))
                                                            for p in hr]),
                                    "bpm": [p[1] for p in hr]})
                st.altair_chart(alt.Chart(hdf).mark_line(color="#9B1C1C", strokeWidth=2, point=len(hr) < 30).encode(
                    x=alt.X("time:T", axis=alt.Axis(title=None, format="%H:%M", labelColor="#556274", grid=False)),
                    y=alt.Y("bpm:Q", scale=alt.Scale(zero=False),
                            axis=alt.Axis(title=None, gridColor="#EEF1F4", labelColor="#556274")),
                    tooltip=[alt.Tooltip("time:T", format="%H:%M"), "bpm"])
                    .properties(height=170).configure_view(strokeWidth=0), width="stretch")
            else:
                st.markdown('<div class="sc-empty">No watch data yet. Set up Health Auto Export or SensorLog '
                            '(see the Access page).</div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="sc-h">What the care agent changed</div>'
                        f'<div class="sc-sub">{len(decisions)} reviews this session</div>', unsafe_allow_html=True)
            if decisions:
                st.dataframe(pd.DataFrame([{"Time": d.get("time", ""),
                                            "By": "rules" if d.get("source") == "rules" else "Nemotron",
                                            "Change": decision_summary(d)} for d in decisions]),
                             hide_index=True, width="stretch", height=min(260, 38 + 35 * len(decisions)))
            else:
                st.markdown('<div class="sc-empty">No agent reviews yet.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="sc-h">Wearer check-ins</div>', unsafe_allow_html=True)
            cks = comp["checkins"][-8:][::-1]
            if cks:
                label = {"ok": "I'm OK", "help": "I need help"}
                rows = "".join(f'<div class="r"><small>{time.strftime("%H:%M", time.localtime(c["t"]))}</small>'
                               f'{esc(label.get(c["kind"]) or "Walking feels " + str(c.get("value")))}</div>'
                               for c in cks)
                st.markdown(f'<div class="sc-recent">{rows}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sc-empty">None yet.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        episode_log(episodes, comp)
        if done:
            out = pd.DataFrame([dict(e, checked_by=cs.verification(e, comp, now)[1]) for e in done])
            st.download_button("Download episodes (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name=f"stepcue_episodes_{time.strftime('%Y%m%d')}.csv", mime="text/csv")

    st.markdown('<div class="sc-foot">Descriptive summary of what the device detected and did. It is not a '
                'diagnosis and is not treatment advice.</div>', unsafe_allow_html=True)


# ------------------------------------------------------- care: access --
def access_page():
    st.markdown('<div class="sc-top"><div class="sc-brand">StepCue<small>Access and devices</small></div></div>',
                unsafe_allow_html=True)
    if user.get("demo"):
        st.info("Sign-in is turned off (STEPCUE_AUTH=off). Accounts added here are used once it is back on.")
    data = auth.load_users()
    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="sc-h">People with access</div><div class="sc-sub">Family see the simple view. '
                        'The care team see everything, including this page.</div>', unsafe_allow_html=True)
            if data["users"]:
                st.dataframe(pd.DataFrame([{"Name": u.get("name", ""), "Email": u["email"],
                                            "View": auth.ROLES.get(u["role"], u["role"]),
                                            "Sign-in": "Google or password" if u.get("password") else "Google",
                                            "Authenticator": "set up" if u.get("totp_secret") else "not yet",
                                            "Last sign-in": u.get("last_login", "never")} for u in data["users"]]),
                             hide_index=True, width="stretch")
            with st.form("add_user", clear_on_submit=True):
                st.markdown('<div class="sc-h" style="font-size:1rem">Add someone</div>', unsafe_allow_html=True)
                a, b = st.columns(2)
                name = a.text_input("Name")
                email = b.text_input("Email (their Google account, if they use Google)")
                c, d = st.columns(2)
                role = c.selectbox("View", list(auth.ROLES), format_func=lambda r: auth.ROLES[r])
                pw = d.text_input("Password (optional with Google, 10+ characters)", type="password")
                if st.form_submit_button("Add", type="primary"):
                    try:
                        if not pw and not auth_ui.google_configured():
                            raise ValueError("a password is needed until Google sign-in is set up")
                        auth.add_user(data, email, name, role, password=pw or None)
                        auth.save_users(data)
                        st.success("Added. They set up their authenticator app the first time they sign in.")
                    except ValueError as exc:
                        st.error(str(exc).capitalize())
            others = [u["email"] for u in data["users"] if u["email"] != user["email"]]
            if others:
                e1, e2, e3 = st.columns([2, 1, 1], vertical_alignment="bottom")
                who = e1.selectbox("Manage", others)
                if e2.button("Reset authenticator", width="stretch"):
                    auth.reset_2fa(who)
                    st.toast("They will scan a new QR code next time they sign in.")
                if e3.button("Remove access", width="stretch"):
                    target = auth.find_user(data, who)
                    cares = [u for u in data["users"] if u["role"] == "care"]
                    if target and target["role"] == "care" and len(cares) <= 1:
                        st.error("Keep at least one care-team account.")
                    else:
                        auth.remove_user(data, who)
                        auth.save_users(data)
                        st.rerun()
    with right:
        import alerts_ui
        alerts_ui.alert_card()
        with st.container(border=True):
            st.markdown('<div class="sc-h">Android phone (recommended)</div><div class="sc-sub">Scan with the '
                        'Android phone&#39;s camera (same Wi-Fi as this laptop). The page installs the StepCue app '
                        'and connects it in one tap. The app then streams gait features from the gyroscope, '
                        'accelerometer, barometer, step sensors and GPS, and buzzes the walking beat during a '
                        'freeze.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sc-qr">{auth.qr_svg(cs.android_url())}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sc-sub" style="margin-top:6px">First install: Android asks to allow installs '
                        'from Chrome once. Rebuild the app with <code>python android/build.py</code>.</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="sc-sub" style="margin-top:6px"><b>Any laptop can host StepCue.</b> Each computer '
                        f'running StepCue shows its own code here (this one is <b>{html.escape(cs.laptop_name())}</b>, '
                        f'{html.escape(cs.lan_ip())}). Scan it with a phone that already has the app and tap '
                        '<b>Connect</b>: the phone remembers every laptop and finds them again when their address '
                        'changes. On a new Windows laptop, allow Python on <b>private networks</b> when the '
                        'firewall asks, or the phone can&#39;t reach it.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="sc-h">iPhone (web page)</div><div class="sc-sub">Run '
                        'companion_server.py, then scan this with the iPhone camera (same Wi-Fi as this laptop). '
                        'A step-by-step setup opens on the phone: connection check, Apple Watch, Home Screen. '
                        'Large text, one step per screen, with notes for a helper.</div>', unsafe_allow_html=True)
            url = cs.companion_url()
            st.markdown(f'<div class="sc-qr">{auth.qr_svg(url)}</div>', unsafe_allow_html=True)
            st.code(url, language=None)
            st.markdown('<div class="sc-sub" style="margin-top:6px">Safari warns that the certificate is not trusted '
                        '(it was made on this laptop). Tap Show Details, then visit this website.</div>',
                        unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="sc-h">Apple Watch and Health data</div>', unsafe_allow_html=True)
            st.markdown('<div class="sc-sub"><b>Free:</b> the phone setup connects the watch through Apple&#39;s '
                        'Shortcuts app once STEPCUE_SHORTCUT_URL is in .env (build the shortcut once with '
                        'SHORTCUT.md).' + ('' if os.environ.get("STEPCUE_SHORTCUT_URL") else
                                           ' <b>Not set yet.</b>') + '</div>', unsafe_allow_html=True)
            st.markdown('<div class="sc-sub" style="margin-top:6px"><b>Make it automatic (helper, 2 min, on the '
                        'wearer&#39;s phone):</b> Shortcuts &rarr; Automation &rarr; + &rarr; Time of Day (e.g. 9:00), '
                        'Daily &rarr; Run Immediately &rarr; pick StepCue. Repeat for 12:00, 15:00 and 18:00. '
                        'Optional: add the StepCue page to the Home Screen (Share &rarr; Add to Home Screen).</div>',
                        unsafe_allow_html=True)
            ip, key = cs.lan_ip(), cs.pairing_key()
            st.markdown('<div class="sc-sub">Health Auto Export (iPhone): add a REST API automation for Heart Rate '
                        'and Step Count, JSON format, with this URL and header.</div>', unsafe_allow_html=True)
            st.code(f"URL:    http://{ip}:8080/api/health\nHeader: X-StepCue-Key: {key}", language=None)
            st.markdown('<div class="sc-sub">SensorLog (Watch or iPhone): HTTP streaming, JSON, the same URL with '
                        f'?key={esc(key)} on the end. It streams only while the app is open.</div>',
                        unsafe_allow_html=True)


# -------------------------------------------------------- patient: my day --
PATIENT_HEADLINE = {"steady": "You're doing well", "freeze": "Freeze detected, follow the light and buzz",
                    "watch": "A few freezes today", "alert": "Your care team has been told",
                    "offline": "Your ankle sensor is not connected"}
PATIENT_CSS = """<style>
.st-key-p_ok button, .st-key-p_help button, .st-key-p_help2 button {min-height:84px; border-radius:18px;}
.st-key-p_ok button p, .st-key-p_help button p, .st-key-p_help2 button p {font-size:1.35rem !important; font-weight:700;}
.st-key-p_ok button {background:var(--ok) !important; color:#fff !important; border:0 !important;}
.st-key-p_help button {background:#fff !important; color:var(--alert) !important; border:2.5px solid var(--alert) !important;}
.st-key-p_help2 button {background:var(--alert) !important; color:#fff !important; border:0 !important;}
[class*="st-key-p_feel"] button {min-height:58px; border-radius:14px;}
[class*="st-key-p_feel"] button p {font-size:1.05rem !important; font-weight:700;}
</style>"""


def patient_checkin(kind, value=None):
    comp = cs.read_companion()
    cs.add_checkin(comp, kind, value)
    cs.write_companion(comp)


@st.fragment(run_every=3)
def patient_view():
    state = read_state()
    age = state_age_sec()
    view = screen_state(state, age)
    offline = view["key"] == "offline"
    now = time.time()
    episodes = state.get("episodes", [])
    stats = episode_stats(episodes)
    comp = cs.read_companion()
    csum = cs.companion_summary(comp, episodes, now)
    first = (user.get("name") or "").split(" ")[0]

    st.markdown(PATIENT_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="sc-top"><div class="sc-brand">StepCue<small>'
                f'{"Hi " + esc(first) if first and not user.get("demo") else "My day"}</small></div>'
                f'<div class="sc-fresh{" stale" if offline else ""}"><i></i>{esc(freshness(age))}</div></div>',
                unsafe_allow_html=True)
    pview = dict(view, headline=PATIENT_HEADLINE.get(view["key"], view["headline"]))
    banner(pview, state.get("status", "STATIONARY"), state, offline)

    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        with st.container(border=True):
            ck = csum["checkin"]
            st.markdown('<div class="sc-h">How are you?</div><div class="sc-sub">'
                        + (f'Last sent: {esc(ck["text"].replace("Checked in: ", ""))}, {esc(ck["ago"])}.' if ck
                           else "Let your family and care team know.") + '</div>', unsafe_allow_html=True)
            if st.button("✓  I'm OK", key="p_ok", width="stretch"):
                patient_checkin("ok")
                st.toast("Sent: I'm OK")
            if st.session_state.get("p_armed", 0) > now - 5:
                if st.button("Tap again to send help request", key="p_help2", width="stretch"):
                    st.session_state["p_armed"] = 0
                    patient_checkin("help")
                    st.toast("Help request sent to your caregiver")
            elif st.button("I need help", key="p_help", width="stretch"):
                st.session_state["p_armed"] = now
                st.rerun(scope="fragment")
            st.markdown('<div class="sc-sub" style="margin-top:6px">"I need help" alerts your caregiver. '
                        'In an emergency, call your local emergency number.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="sc-h">How is walking today?</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for col, v in zip(cols, cs.WALKING_FEEL):
                if col.button({"some trouble": "So-so"}.get(v, v.capitalize()), key=f"p_feel_{v}", width="stretch"):
                    patient_checkin("walking", v)
                    st.toast("Thanks, noted")
    with right:
        with st.container(border=True):
            last = next((e for e in reversed(episodes) if e.get("recovery_sec") is not None), None)
            st.markdown(
                '<div class="sc-h">Today</div><div class="sc-recent">'
                f'<div class="r"><small>Freezes</small>{stats["count"]}'
                f'<span>{"none so far" if not stats["count"] else "each one got a walking cue"}</span></div>'
                f'<div class="r"><small>Last</small>{esc(str(last.get("timestamp", ""))[:5]) if last else "–"}'
                f'<span>{"walking again after " + format(last["recovery_sec"], ".1f") + " s" if last else ""}</span></div>'
                f'<div class="r"><small>Cue</small>{int(state.get("bpm") or 0)} bpm'
                f'<span>{"light and wrist buzz" if state.get("haptic") else "ankle light"}</span></div>'
                '</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="sc-h">StepCue on your phone</div><div class="sc-sub">Android: scan with the '
                        'phone&#39;s camera, tap Download, Install, then Open StepCue. The app uses the phone&#39;s '
                        'motion sensors to help when you freeze.</div>'
                        f'<div class="sc-qr">{auth.qr_svg(cs.android_url())}</div>', unsafe_allow_html=True)
            with st.expander("Using an iPhone?"):
                st.markdown(f'<div class="sc-qr">{auth.qr_svg(cs.companion_url())}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sc-foot">StepCue helps you start walking again with light and vibration cues and keeps '
                'your family and care team informed. It does not diagnose or give medical advice.</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------- page switch --
PATIENT = user["role"] == "patient"
PAGES = (["Overview", "Activity", "Training", "Location", "Care report", "NVIDIA AI", "Access"] if CARE
         else ["My day"] if PATIENT else ["Today", "Activity", "Location"])
page = st.segmented_control("View", PAGES, default=PAGES[0], key=f"page_{user['role']}",
                            label_visibility="collapsed") or PAGES[0]
if page == "Activity":
    phone_ui.activity_page(read_state)
elif page == "Training":
    training_ui.training_page()
elif page == "Location":
    location_page()
elif page == "Care report":
    care_report()
elif page == "Access":
    access_page()
elif page == "NVIDIA AI":
    nvidia_ui.ai_page(read_state, read_decisions, cs.read_companion, cs.companion_summary)
elif CARE:
    live()
elif PATIENT:
    patient_view()
else:
    family_today()
    nvidia_ui.css()
    with st.container(border=True):
        st.markdown('<div class="sc-h">Ask StepCue</div><div class="sc-sub">Ask about today&#39;s walking, freezes or '
                    'alerts. Answers come from StepCue&#39;s records.</div>', unsafe_allow_html=True)
        nvidia_ui.ask_box("fam", read_state(), read_decisions(15), cs.read_companion(),
                          examples=("How was today?", "What does the alert mean?"))
