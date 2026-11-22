"""Care team card: the caretaker number the wearer's Android phone calls (free, over its own plan).

The number stays in companion.json on this laptop and goes only to the paired phone. Nothing is dialed
from here: the phone calls when help is tapped, when a freeze lasts too long, or when someone presses
"Test call".
"""
import html

import streamlit as st

import companion_store as cs


def alert_card():
    comp = cs.read_companion()
    a = cs.alert_settings(comp) or {}
    with st.container(border=True):
        st.markdown('<div class="sc-h">Calls from the phone</div><div class="sc-sub">The wearer&#39;s Android '
                    'phone calls this number when they tap <b>I need help</b> or a freeze lasts over '
                    f'{cs.ALERT_FREEZE_SEC} seconds. It uses the phone&#39;s own plan, so there is no extra '
                    'cost.</div>', unsafe_allow_html=True)
        with st.form("alert_contact"):
            number = st.text_input("Caretaker's phone number (the phone that should ring)",
                                   value=a.get("phone", ""), placeholder="+1 555 123 4567",
                                   help="The patient's phone, the one with the StepCue app, calls this number.")
            on = st.checkbox("Turned on", value=a.get("enabled", True))
            if st.form_submit_button("Save", type="primary", width="stretch"):
                try:
                    comp = cs.read_companion()
                    if number.strip():
                        cs.set_alert_contact(comp, number, enabled=on)
                        st.toast("Saved. The phone picks it up within a few seconds.")
                    else:
                        cs.clear_alert_contact(comp)
                        st.toast("Number removed. The phone will not call.")
                    cs.write_companion(comp)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc).capitalize())
        if a:
            b1, b2 = st.columns(2)
            if b1.button("Test call", width="stretch", help="The phone calls this number once."):
                comp = cs.read_companion()
                cs.request_test_call(comp)
                cs.write_companion(comp)
                st.toast("Asked the phone to make a test call.")
            b2.markdown(f'<div class="sc-sub" style="padding-top:10px">Calls are '
                        f'<b>{"on" if a.get("enabled", True) else "off"}</b>.</div>', unsafe_allow_html=True)
        log = list(reversed(comp.get("alerts_log") or []))[:5]
        if log:
            words = {"help": "Help tap", "freeze": "Long freeze", "test": "Test"}
            rows = "".join(
                f'<div class="sc-sub">{words.get(r["kind"], r["kind"])}: '
                f'{"call placed" if r.get("call") else "no call"}'
                f'{" (" + html.escape(r["error"]) + ")" if r.get("error") else ""}'
                f' &middot; {cs.ago(r["t"])}</div>' for r in log)
            st.markdown('<div class="sc-h" style="font-size:1rem;margin-top:8px">What the phone did</div>' + rows,
                        unsafe_allow_html=True)
        elif a:
            st.markdown('<div class="sc-sub">No calls yet.</div>', unsafe_allow_html=True)
