"""Streamlit pieces for StepCue's NVIDIA features (used by caretaker_app.py).

ai_page()            care team: the model pipeline with live call stats, Ask StepCue,
                     the Nemotron Ultra daily report, the agent's tool trace, voice cues
ask_box(key, ...)    Ask StepCue (any role)
brief_in_language()  the latest caregiver brief, translated with Riva Translate
"""
import html
import json
import time

import streamlit as st

import nvidia_ask
import nvidia_nim
import nvidia_report
import nvidia_translate

CSS = """<style>
.nv-pipe {display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:10px; margin:6px 0 4px;}
.nv-m {background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:12px 14px; font-family:var(--font);}
.nv-m .job {font:700 0.98rem var(--font); color:var(--ink); display:flex; align-items:center; gap:8px;}
.nv-m .job i {width:9px; height:9px; border-radius:50%; background:#B8C2CE; display:inline-block; flex:none;}
.nv-m.on .job i {background:#76B900;} .nv-m.err .job i {background:var(--watch);}
.nv-m .id {font:500 0.78rem ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); margin:3px 0 6px; word-break:break-all;}
.nv-m .what {font-size:0.88rem; color:var(--ink); line-height:1.35;}
.nv-m .num {font-size:0.82rem; color:var(--muted); margin-top:7px; font-variant-numeric:tabular-nums;}
.nv-key {font:600 0.9rem var(--font); border-radius:12px; padding:10px 14px; margin:4px 0 10px;
         background:var(--watch-tint); color:#7A4300; border:1px solid #F0D9AE;}
.nv-key.on {background:#EEF7E0; color:#3D6100; border-color:#D5EBB0;}
.nv-chain {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:6px 0 2px;}
.nv-chip {font:600 0.78rem var(--font); padding:3px 9px; border-radius:999px; background:var(--offline-tint); color:var(--ink);}
.nv-chip.g {background:#EEF7E0; color:#3D6100;} .nv-chip.b {background:var(--watch-tint); color:#7A4300;}
.nv-arrow {color:var(--muted); font-size:0.8rem;}
.nv-ans {font:400 1.05rem/1.5 var(--font); color:var(--ink); background:var(--surface); border:1px solid var(--line);
         border-radius:14px; padding:14px 16px; margin:6px 0;}
.nv-ans.blocked {border-color:#F0D9AE; background:var(--watch-tint);}
.nv-src {font-size:0.86rem; color:var(--muted); border-top:1px solid var(--line); padding:7px 0;}
.nv-src b {color:var(--ink); font-weight:700; margin-right:6px;}
.nv-rep h3 {font:700 1.15rem var(--font); margin:4px 0 8px; color:var(--ink);}
.nv-rep h4 {font:700 0.86rem var(--font); color:var(--muted); margin:12px 0 4px;}
.nv-rep ul {margin:0 0 0 1.1rem; padding:0;} .nv-rep li {margin:2px 0; font-size:0.97rem;}
.nv-by {font-size:0.84rem; color:var(--muted); margin-top:10px;}
</style>"""

JOB_TITLE = {"agent": "Care agent", "report": "Daily report", "ask": "Ask StepCue", "voice_in": "Spoken check-ins",
             "embed": "Retrieval", "rerank": "Re-ranking", "safety": "Safety guard", "topic": "Topic guard",
             "translate": "Translation", "tts": "Voice cues"}
TOOL_WORD = {"get_cue_effectiveness": "cue history", "get_time_pattern": "time pattern",
             "get_phone_and_watch": "phone + watch", "get_episode_history": "episodes",
             "update_ankle_cue": "set cue"}


def esc(v):
    return html.escape(str(v), quote=True)


def css():
    st.markdown(CSS, unsafe_allow_html=True)


def key_banner():
    on = bool(nvidia_nim.api_key())
    if on:
        st.markdown('<div class="nv-key on">Connected to NVIDIA NIM. Names, emails, phone numbers and '
                    'coordinates are removed before any request.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="nv-key">Not connected to NVIDIA yet: every feature below is running on its '
                    'local fallback. Add <b>NVIDIA_API_KEY</b> to the <b>.env</b> file (free key at '
                    'build.nvidia.com) and restart.</div>', unsafe_allow_html=True)
    return on


def pipeline(summary):
    cells = []
    for job, (_, _, what) in nvidia_nim.MODELS.items():
        s = summary.get(job)
        model = nvidia_nim.model_for(job)
        if s:
            ok = s["ok"] / s["calls"] if s["calls"] else 0
            cls = "on" if ok >= 0.5 else "err"
            med = f', median {s["median_sec"]:.1f} s' if s.get("median_sec") is not None else ""
            num = f'{s["calls"]} call{"s" if s["calls"] != 1 else ""} this hour, {ok:.0%} ok{med}'
        else:
            cls, num = "", "No calls this hour"
        cells.append(f'<div class="nv-m {cls}"><div class="job"><i></i>{esc(JOB_TITLE.get(job, job))}</div>'
                     f'<div class="id">{esc(model)}</div><div class="what">{esc(what)}</div>'
                     f'<div class="num">{esc(num)}</div></div>')
    st.markdown(f'<div class="nv-pipe">{"".join(cells)}</div>', unsafe_allow_html=True)


def chain(tools, guard=None, source=None):
    chips = []
    if tools:
        for i, t in enumerate(tools):
            if i:
                chips.append('<span class="nv-arrow">→</span>')
            chips.append(f'<span class="nv-chip{" g" if t == "update_ankle_cue" else ""}">{esc(TOOL_WORD.get(t, t))}</span>')
    elif source == "rules":
        chips.append('<span class="nv-chip b">built-in rules</span>')
    for g in guard or []:
        label = {"local": "boundary check", "nemotron-safety": "Safety Guard", "nemoguard-topic": "Topic guard"}.get(g, g)
        chips.append(f'<span class="nv-chip g">{esc(label)}</span>')
    return f'<div class="nv-chain">{"".join(chips)}</div>'


def ask_box(key, state, decisions, companion, examples=()):
    with st.form(f"ask_{key}", clear_on_submit=False, border=False):
        q = st.text_input("Ask about today's walking", key=f"askq_{key}",
                          placeholder="e.g. When did freezes happen today? Is the cue working?")
        go = st.form_submit_button("Ask StepCue")
    if examples:
        ex = st.pills("Try", list(examples), selection_mode="single", key=f"askex_{key}",
                      label_visibility="collapsed")
        if ex and ex != st.session_state.get(f"askexlast_{key}"):
            q, go = ex, True
        st.session_state[f"askexlast_{key}"] = ex
    if go and q:
        with st.spinner("Checking the records…"):
            st.session_state[f"askr_{key}"] = nvidia_ask.ask(q, state, decisions, companion)
    r = st.session_state.get(f"askr_{key}")
    if not r:
        return
    st.markdown(f'<div class="nv-ans{" blocked" if r.get("blocked") else ""}">{esc(r["answer"])}</div>',
                unsafe_allow_html=True)
    how = (f'Retrieval: {r.get("retrieval")}. Answer: {"local fallback" if r.get("mode") == "local" else r.get("mode")}. '
           f'{r.get("latency_sec", 0):.1f} s.')
    st.markdown(chain([], r.get("layers")) + f'<div class="nv-by">{esc(how)}</div>', unsafe_allow_html=True)
    cited = set(r.get("cited") or [])
    srcs = [s for s in r.get("sources", []) if s["id"] in cited] or r.get("sources", [])[:3]
    if srcs and not r.get("blocked"):
        with st.expander(f"Records used ({len(srcs)})"):
            st.markdown("".join(f'<div class="nv-src"><b>[{esc(s["id"])}]</b>{esc(s["text"])}</div>' for s in srcs),
                        unsafe_allow_html=True)


def report_card(state, decisions, companion_summary):
    rep = nvidia_report.read_report()
    c1, c2 = st.columns([3, 1])
    c1.markdown('<div class="sc-h">Daily care report</div><div class="sc-sub">Written by NVIDIA Nemotron 3 Ultra '
                '(550B) from every episode, agent decision and check-in. Observations only.</div>',
                unsafe_allow_html=True)
    if c2.button("Write it now", key="nv_report", width="stretch"):
        with st.spinner("Nemotron Ultra is reading the day…"):
            rep = nvidia_report.write_report(state, decisions, companion_summary)
    if not rep:
        st.markdown('<div class="sc-empty">No report yet today. Press "Write it now".</div>', unsafe_allow_html=True)
        return
    parts = [f'<div class="nv-rep"><h3>{esc(rep["headline"])}</h3>']
    for k, title in (("what_happened", "What happened"), ("what_the_device_did", "What the device did"),
                     ("patterns", "Patterns"), ("for_the_care_team", "For the care team")):
        if rep.get(k):
            parts.append(f"<h4>{title}</h4><ul>" + "".join(f"<li>{esc(x)}</li>" for x in rep[k]) + "</ul>")
    who = "local template (NVIDIA not reachable)" if rep.get("source") == "local" else rep.get("source")
    parts.append(f'<div class="nv-by">{esc(rep.get("generated_at", ""))}, {rep.get("episodes", 0)} episodes, by '
                 f'{esc(who)}, {rep.get("latency_sec", 0)} s</div></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def agent_trace(decisions):
    st.markdown('<div class="sc-h">How the care agent decided</div><div class="sc-sub">Each review: the tools '
                'Nemotron chose to call, then the guard checks on its brief.</div>', unsafe_allow_html=True)
    if not decisions:
        st.markdown('<div class="sc-empty">Appears after the first freeze episode.</div>', unsafe_allow_html=True)
        return
    for d in decisions[:4]:
        dec = d.get("decision", {})
        who = "rules" if d.get("source") == "rules" else d.get("source", "").split("/")[-1]
        st.markdown(f'<div class="sc-dec"><div class="t">{esc(d.get("time", ""))}, episode '
                    f'{esc(", ".join(map(str, d.get("episode_ids", []))))}, {esc(who)}, {d.get("latency_sec", 0):.1f} s</div>'
                    + chain(d.get("tools"), d.get("guard"), d.get("source"))
                    + f'<div class="c">Cue {esc(d.get("before", {}).get("bpm"))} → {esc(dec.get("bpm"))} bpm, '
                      f'vibration {"on" if dec.get("haptic") else "off"}, {esc(dec.get("alert_level", ""))}</div>'
                    + (f'<div class="w">{esc(dec.get("reasoning"))}</div>' if dec.get("reasoning") else "")
                    + '</div>', unsafe_allow_html=True)


def voice_card(companion):
    import nvidia_voice
    man = nvidia_voice.read_manifest()
    lang = (companion.get("prefs") or {}).get("lang", "en")
    st.markdown('<div class="sc-h">Wearer\'s language and voice cues</div><div class="sc-sub">Spoken cues on the '
                'phone are made once with Riva Translate and Riva Magpie TTS. Spoken check-ins use Nemotron Omni.</div>',
                unsafe_allow_html=True)
    codes = list(nvidia_translate.LANGUAGES)
    pick = st.selectbox("Wearer's language", codes, index=codes.index(lang) if lang in codes else 0,
                        format_func=lambda c: f"{nvidia_translate.LANGUAGES[c][1]} ({nvidia_translate.LANGUAGES[c][0]})",
                        key="nv_wearer_lang")
    if pick != lang:
        import companion_store as cs
        comp = cs.read_companion()
        comp.setdefault("prefs", {})["lang"] = pick
        cs.write_companion(comp)
        st.toast("Saved. The phone uses this language for spoken cues.")
    if man:
        rows = "".join(f'<div class="nv-src"><b>{esc(k)}</b>{esc(v.get("voice") or "default voice")}, '
                       f'{len(v.get("files", {}))} cues, {esc(v.get("made", ""))}</div>' for k, v in man.items())
        st.markdown(rows, unsafe_allow_html=True)
    else:
        st.markdown('<div class="sc-empty">No voice cues yet; the phone uses its built-in voice. Make them with '
                    '<code>python nvidia_voice.py --lang en ' + esc(pick if pick != "en" else "es") + '</code>.</div>',
                    unsafe_allow_html=True)


def brief_in_language(brief, key):
    codes = list(nvidia_translate.LANGUAGES)
    lang = st.session_state.get(f"lang_{key}", "en")
    if lang != "en" and brief:
        r = nvidia_translate.translate(brief, lang)
        if r["ok"]:
            brief = r["text"]
    st.selectbox("Language", codes, key=f"lang_{key}", label_visibility="collapsed",
                 format_func=lambda c: nvidia_translate.LANGUAGES[c][1])
    return brief


def ai_page(read_state, read_decisions, read_companion, companion_summary):
    css()
    state = read_state()
    decisions = read_decisions(20)
    comp = read_companion()
    st.markdown('<div class="sc-top"><div class="sc-brand">StepCue<small>NVIDIA AI</small></div>'
                f'<div class="sc-fresh"><i></i>{time.strftime("%H:%M")}</div></div>', unsafe_allow_html=True)
    key_banner()
    summary = nvidia_nim.ledger_summary(nvidia_nim.read_ledger(400))
    pipeline(summary)

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="sc-h">Ask StepCue</div><div class="sc-sub">Answers come only from StepCue\'s own '
                        'records, with the records cited. Questions about diagnosis or medication are turned '
                        'away by NemoGuard.</div>', unsafe_allow_html=True)
            ask_box("care", state, decisions, comp,
                    examples=("Is the cue working?", "When do freezes happen?", "What did the agent change?"))
        with st.container(border=True):
            try:
                summ = companion_summary(comp, state.get("episodes", []))
            except Exception:
                summ = None
            report_card(state, decisions, summ)
    with right:
        with st.container(border=True):
            agent_trace(decisions)
        with st.container(border=True):
            voice_card(comp)
    with st.expander("Raw NVIDIA call ledger (last 30)"):
        rows = nvidia_nim.read_ledger(30)[::-1]
        st.code("\n".join(json.dumps(r) for r in rows) or "No calls yet.", language="json")
