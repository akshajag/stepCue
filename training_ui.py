"""Training page: teach StepCue one person's sitting, standing, walking and freezing.

A helper labels what the wearer is doing (here or on the phone's Training screen), StepCue saves
the phone + ankle sensor readings under that label, and "Build the model" trains a model for that
person alone. Results are shown honestly: accuracy on recordings the model did not train on.
"""
import html
import time

import altair as alt
import pandas as pd
import streamlit as st

import companion_store as cs
import personal_model as pm

CSS = """<style>
.tr-head h2 {font:700 1.5rem var(--font); color:var(--ink); margin:0; padding:0;}
.tr-head p {font:400 0.98rem var(--font); color:var(--muted); margin:4px 0 12px; max-width:760px; line-height:1.45;}
.tr-conn {display:flex; flex-wrap:wrap; gap:8px; margin:0 0 10px;}
.tr-conn span {font:700 0.9rem var(--font); border-radius:999px; padding:6px 12px;}
.tr-on {background:var(--ok-tint); color:var(--ok);} .tr-off {background:var(--offline-tint); color:var(--offline); border:1.5px dashed #AEB8C5;}
.tr-rec {font:700 1.35rem var(--font); color:var(--ink); margin:2px 0 8px;}
.tr-rec.on {color:var(--ok);} .tr-rec.frz {color:var(--alert);}
.tr-rec small {font-weight:400; font-size:0.95rem; color:var(--muted); margin-left:8px;}
.tr-bars > div {display:grid; grid-template-columns:110px 1fr 96px; gap:10px; align-items:center; margin:7px 0;
                font:400 0.95rem var(--font); color:var(--ink);}
.tr-bars .track {height:14px; background:#EEF1F4; border-radius:7px; overflow:hidden;}
.tr-bars .fill {height:100%; background:var(--ok); border-radius:7px;}
.tr-bars .fill.part {background:#E3A857;}
.tr-bars .n {text-align:right; font-variant-numeric:tabular-nums; color:var(--muted);}
.tr-big {font:700 2.6rem var(--font); color:var(--ink); line-height:1;}
.tr-big small {font-size:1rem; font-weight:400; color:var(--muted); margin-left:6px;}
.tr-note {font:400 0.9rem var(--font); color:var(--muted); line-height:1.45;}
.tr-phase {font:700 1.6rem var(--font); color:var(--ok);} .tr-phase.frz {color:var(--alert);}
div[data-testid="stButton"] button[kind="primary"] {background:var(--ok); border-color:var(--ok);}
[class*="st-key-lab_"] button {min-height:68px; border-radius:16px;}
[class*="st-key-lab_"] button p {font-size:1.25rem; font-weight:700;}
.st-key-lab_frozen button[kind="primary"] {background:var(--alert); border-color:var(--alert);}
</style>"""

ICON = {"sitting": "Sitting", "standing": "Standing", "walking": "Walking", "frozen": "Frozen"}


def esc(v):
    return html.escape(str(v), quote=True)


def seed_patients():
    """First visit: offer the patient accounts from sign-in as training profiles."""
    if pm.profiles()["patients"]:
        return
    try:
        import auth
        names = [u.get("name") for u in auth.load_users()["users"] if u.get("role") == "patient" and u.get("name")]
    except Exception:
        names = []
    for n in names:
        pm.add_patient(n, make_active=False)


def patient_picker():
    p = pm.profiles()
    left, right = st.columns([3, 2], gap="medium")
    with left:
        if p["patients"]:
            ids = list(p["patients"])
            pick = st.selectbox("Person being trained", ids, index=ids.index(p["active"]) if p["active"] in ids else 0,
                                format_func=lambda i: p["patients"][i]["name"])
            if pick != p["active"]:
                pm.set_active(pick)
                st.rerun()
        else:
            st.markdown('<div class="tr-note">No one yet. Add the person you are training.</div>', unsafe_allow_html=True)
    with right:
        with st.form("add_patient", clear_on_submit=True, border=False):
            name = st.text_input("Add a person", placeholder="Name")
            if st.form_submit_button("Add", width="stretch") and name.strip():
                pm.add_patient(name)
                st.rerun()


def bars(summary):
    rows = []
    for lab in pm.LABELS:
        n = summary["counts"][lab]
        pct = min(100, round(100 * n / summary["goal"]))
        rows.append(f'<div><b>{ICON[lab]}</b><div class="track"><div class="fill{" part" if n < summary["goal"] else ""}" '
                    f'style="width:{pct}%"></div></div><span class="n">{n} / {summary["goal"]} s</span></div>')
    return f'<div class="tr-bars">{"".join(rows)}</div>'


@st.fragment(run_every=1)
def recorder(pid):
    now = time.time()
    comp = cs.read_companion()
    rec = comp.get("android") or {}
    phone_live = bool(rec.get("t")) and now - rec["t"] < 10
    try:
        from state_io import read_state
        ankle_live = pm.ankle_from_state(read_state(), now) is not None
    except Exception:
        ankle_live = False
    st.markdown('<div class="tr-conn">'
                f'<span class="{"tr-on" if phone_live else "tr-off"}">Phone: {"sharing" if phone_live else "not sharing"}</span>'
                f'<span class="{"tr-on" if ankle_live else "tr-off"}">Ankle sensor (ESP32): '
                f'{"connected" if ankle_live else "not connected"}</span></div>', unsafe_allow_html=True)

    s = pm.session(now)
    lab = s["label"] if s["patient"] == pid else None
    if lab:
        secs = int(now - s["since"])
        settling = secs < pm.SETTLE_SEC
        st.markdown(f'<div class="tr-rec {"frz" if lab == "frozen" else "on"}">&#9679; Recording {ICON[lab]} '
                    f'{secs} s<small>{"settling, not saved yet" if settling else "saving 1 reading a second"}</small></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="tr-rec">Not recording<small>Tap what the person is doing right now.</small></div>',
                    unsafe_allow_html=True)
    if not phone_live:
        st.caption("Readings are saved from the phone, so start the StepCue app first (and the ankle sensor if you have it).")

    cols = st.columns(4)
    for c, label in zip(cols, pm.LABELS):
        if c.button(ICON[label], key=f"lab_{label}", type="primary" if label == lab else "secondary",
                    width="stretch"):
            pm.set_label(label)
            st.rerun(scope="fragment")
    if lab and st.button("Stop recording", key="stop_rec", width="stretch"):
        pm.stop_label()
        st.rerun(scope="fragment")

    summ = pm.dataset_summary(pid)
    st.markdown(bars(summ), unsafe_allow_html=True)
    if summ["total"]:
        st.markdown(f'<div class="tr-note">{summ["total"]} readings saved, {summ["with_ankle"]} with the ankle '
                    f'sensor. Separate recordings per activity: '
                    + ", ".join(f'{ICON[k]} {v}' for k, v in summ["segments"].items())
                    + '. Record each activity at least twice for an honest score.</div>', unsafe_allow_html=True)

    phase = rec.get("phase") if phone_live else None
    if phase and not lab:
        st.markdown(f'<div class="tr-note" style="margin-top:12px">Their own model right now</div>'
                    f'<div class="tr-phase{" frz" if phase["label"] == "frozen" else ""}">{esc(phase["word"])}'
                    f'<span class="tr-note"> &nbsp;{phase["conf"]:.0%} sure, from {esc(phase["source"])}</span></div>',
                    unsafe_allow_html=True)
        probs = phase.get("probs") or {}
        if probs:
            df = pd.DataFrame({"Activity": [ICON.get(k, k) for k in probs], "p": list(probs.values())})
            st.altair_chart(alt.Chart(df).mark_bar(cornerRadiusEnd=4, color="#0F766E").encode(
                x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None, grid=False)),
                y=alt.Y("Activity:N", sort=[ICON[k] for k in pm.LABELS], axis=alt.Axis(title=None))
            ).properties(height=130).configure_view(strokeWidth=0), width="stretch")


def confusion_chart(m):
    labels = [ICON[k] for k in m["labels"]]
    rows = []
    for i, a in enumerate(labels):
        total = sum(m["confusion"][i]) or 1
        for j, b in enumerate(labels):
            n = m["confusion"][i][j]
            rows.append({"Really": a, "Model said": b, "n": n, "share": n / total})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(x=alt.X("Model said:N", sort=labels, axis=alt.Axis(orient="top", labelAngle=0)),
                                y=alt.Y("Really:N", sort=labels))
    heat = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=["#F4F6F8", "#0F766E"]), legend=None),
        tooltip=["Really", "Model said", "n"])
    text = base.mark_text(fontSize=14, font="Atkinson Hyperlegible").encode(
        text="n:Q", color=alt.condition("datum.share > 0.5", alt.value("white"), alt.value("#16202B")))
    return (heat + text).properties(height=60 * len(labels) + 20).configure_view(strokeWidth=0)


def results(pid):
    m = pm.model_info(pid)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="sc-h">Build the model</div><div class="sc-sub">Trains on this person&#39;s '
                    'recordings only. Takes a few seconds.</div>', unsafe_allow_html=True)
    with c2:
        if st.button("Build the model", type="primary", width="stretch"):
            try:
                pm.stop_label()
                with st.spinner("Training..."):
                    pm.train(pid)
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc).capitalize() + ".")
    if not m:
        st.markdown('<div class="tr-note">No model yet for this person.</div>', unsafe_allow_html=True)
        return
    best = m.get("both") or m["phone"]
    src = "phone + ankle sensor" if m.get("both") else "phone only"
    a, b = st.columns([2, 3], gap="large")
    with a:
        st.markdown(f'<div class="tr-big">{best["accuracy"]:.0%}<small>right on unseen data</small></div>'
                    f'<div class="tr-note" style="margin-top:6px">Uses the {src}, {m["n"]} readings. Scored '
                    f'{esc(m["cv"])}. Trained {esc(cs.ago(m["trained_at"]))}.</div>', unsafe_allow_html=True)
        if m.get("both"):
            st.markdown(f'<div class="tr-note">Phone alone: {m["phone"]["accuracy"]:.0%}.</div>',
                        unsafe_allow_html=True)
        rec_rows = "".join(f'<div><b>{ICON[k]}</b><div class="track"><div class="fill{"" if (v or 0) >= 0.8 else " part"}" '
                           f'style="width:{round(100 * (v or 0))}%"></div></div><span class="n">{(v or 0):.0%} found</span></div>'
                           for k, v in best["recall"].items())
        st.markdown(f'<div class="sc-h" style="margin-top:14px">How often each one is caught</div>'
                    f'<div class="tr-bars">{rec_rows}</div>', unsafe_allow_html=True)
        if m.get("skipped") and any(m["skipped"].values()):
            st.markdown('<div class="tr-note">Not enough yet (need 10 s): '
                        + ", ".join(f"{ICON[k]} {v} s" for k, v in m["skipped"].items()) + '</div>',
                        unsafe_allow_html=True)
        if m.get("pace"):
            p = m["pace"]
            st.markdown(f'<div class="sc-h" style="margin-top:14px">Their walking pace</div>'
                        f'<div class="tr-note">Usually {p["low"]}&ndash;{p["high"]} steps a minute '
                        f'(middle {p["median"]}). The Activity page compares against this.</div>',
                        unsafe_allow_html=True)
    with b:
        st.markdown('<div class="sc-h">What the model mixes up</div><div class="sc-sub">Rows: what was really '
                    'happening. Columns: what the model said. Big numbers on the diagonal are good.</div>',
                    unsafe_allow_html=True)
        st.altair_chart(confusion_chart(best), width="stretch")
        imp = pd.DataFrame(m["importance"], columns=["Signal", "Weight"])
        st.markdown('<div class="sc-h">Signals it leans on most</div>', unsafe_allow_html=True)
        st.altair_chart(alt.Chart(imp).mark_bar(color="#5B8DB8", cornerRadiusEnd=3).encode(
            x=alt.X("Weight:Q", axis=alt.Axis(title=None, grid=False, format=".0%")),
            y=alt.Y("Signal:N", sort="-x", axis=alt.Axis(title=None))).properties(height=200)
            .configure_view(strokeWidth=0), width="stretch")


def training_page():
    st.markdown(CSS, unsafe_allow_html=True)
    seed_patients()
    st.markdown('<div class="tr-head"><h2>Training</h2><p>Teach StepCue how this person sits, stands, walks and '
                'freezes. Everyone moves differently, so each person gets their own data and their own model. '
                'Tap what they are doing while they do it (about a minute each, twice), then build the model. '
                'The same buttons are on the phone under "Teach StepCue".</p></div>', unsafe_allow_html=True)
    patient_picker()
    pid, name = pm.active()
    if not pid:
        return
    with st.container(border=True):
        st.markdown(f'<div class="sc-h">Record: {esc(name)}</div>', unsafe_allow_html=True)
        recorder(pid)
    with st.container(border=True):
        results(pid)
    with st.expander("Delete this person's training data"):
        st.caption("Removes their recordings and model from this laptop. This can't be undone.")
        sure = st.checkbox(f"Yes, delete {name}'s training data", key=f"del_{pid}")
        if st.button("Delete", disabled=not sure, key=f"delbtn_{pid}"):
            pm.delete_data(pid)
            st.rerun()
    st.markdown('<div class="sc-foot">Training data stays on this laptop. The model describes movement to help '
                'with walking cues; it does not diagnose or give medical advice.</div>', unsafe_allow_html=True)
