"""NVIDIA integrations, tested with a fake NIM client (no network)."""
import base64
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nemotron_advisor as adv  # noqa: E402
import nvidia_ask  # noqa: E402
import nvidia_guard  # noqa: E402
import nvidia_nim  # noqa: E402
import nvidia_report  # noqa: E402
import nvidia_translate  # noqa: E402
import nvidia_voice_in  # noqa: E402


def msg(content="", tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


def call(name, args, cid="c1"):
    return NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))


class FakeClient:
    """Replies are chosen by model id; every request is kept for inspection."""

    def __init__(self, replies=None, embed=None):
        self.replies = replies or {}
        self.requests = []
        self.embed_fn = embed or (lambda texts: [[float(len(t) % 7), 1.0, float("freeze" in t.lower())] for t in texts])
        self.chat = NS(completions=NS(create=self._chat))
        self.embeddings = NS(create=self._embed)

    def _chat(self, **kw):
        self.requests.append(kw)
        r = self.replies.get(kw["model"])
        if callable(r):
            return r(kw)
        if isinstance(r, list):
            return r.pop(0)
        if isinstance(r, Exception):
            raise r
        return msg(r or "")

    def _embed(self, **kw):
        self.requests.append(kw)
        return NS(data=[NS(embedding=v) for v in self.embed_fn(kw["input"])])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        users = os.path.join(self.tmp.name, "users.json")
        with open(users, "w") as f:
            json.dump({"users": [{"name": "Akshaj Agadi", "email": "a@x.local", "role": "patient"}]}, f)
        for target, attr, val in ((nvidia_nim, "LEDGER_FILE", os.path.join(self.tmp.name, "ledger.jsonl")),
                                  (nvidia_ask, "CACHE_FILE", os.path.join(self.tmp.name, "embed.json")),
                                  (nvidia_translate, "CACHE_FILE", os.path.join(self.tmp.name, "tr.json"))):
            p = mock.patch.object(target, attr, val)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {"STEPCUE_USERS": users})
        p.start()
        self.addCleanup(p.stop)

    def nim(self, client):
        return nvidia_nim.NIM(key="test", client=client)


class Privacy(Base):
    def test_scrub_removes_personal_details(self):
        text = ("Akshaj froze at 30.6187, -96.3365 near 123 Main Street; call +1 979 555 0100 "
                "or mail akshaj@stepcue.local")
        out = nvidia_nim.scrub(text)
        for leaked in ("Akshaj", "30.6187", "Main Street", "555", "@"):
            self.assertNotIn(leaked, out)
        self.assertIn("the wearer", out)

    def test_chat_requests_are_scrubbed_and_logged(self):
        fc = FakeClient({"m": "ok"})
        self.nim(fc).text("ask", [{"role": "user", "content": "How is Akshaj Agadi today?"}], model="m")
        self.assertNotIn("Akshaj", json.dumps(fc.requests[0]["messages"]))
        led = nvidia_nim.read_ledger()
        self.assertEqual((led[-1]["job"], led[-1]["ok"]), ("ask", True))
        self.assertEqual(nvidia_nim.ledger_summary(led)["ask"]["calls"], 1)

    def test_thinking_switch_goes_to_chat_template_kwargs(self):
        fc = FakeClient({"m": "ok"})
        self.nim(fc).text("agent", [{"role": "user", "content": "hi"}], model="m", thinking=False)
        self.assertFalse(fc.requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_no_key_means_unavailable(self):
        nim = nvidia_nim.NIM(key="")
        self.assertFalse(nim.available)
        with self.assertRaises(nvidia_nim.NIMError):
            nim.text("ask", [{"role": "user", "content": "x"}])


STATE = {"status": "WALKING", "bpm": 100, "haptic": 0, "alert_level": "normal", "episodes": [
    {"id": 1, "timestamp": "09:10:02", "duration_sec": 2.1, "pre_freeze_cadence": 98, "cue_bpm": 100,
     "cue_haptic": 0, "recovery_sec": 3.9, "analyzed": True, "during_turn": True},
    {"id": 2, "timestamp": "09:40:05", "duration_sec": 1.8, "pre_freeze_cadence": 96, "cue_bpm": 90,
     "cue_haptic": 1, "recovery_sec": 1.2, "analyzed": True, "during_turn": True},
    {"id": 3, "timestamp": "14:12:05", "duration_sec": 5.8, "pre_freeze_cadence": 96, "cue_bpm": 100,
     "cue_haptic": 0, "recovery_sec": 5.4, "analyzed": False}]}


class Agent(Base):
    def test_agent_uses_read_tools_then_decides(self):
        model = "nvidia/nemotron-3.5-lightning-30b-a3b"
        final = {"bpm": 90, "haptic": 1, "alert_level": "watch", "signal_uncertain": False,
                 "caretaker_brief": "Freeze of 5.8 s. Cue moved to 90 bpm with vibration, which worked fastest before.",
                 "reasoning": "90 bpm with vibration recovered in 1.2 s earlier."}
        fc = FakeClient({model: [msg(tool_calls=[call("get_cue_effectiveness", {})]),
                                 msg(tool_calls=[call("update_ankle_cue", final, "c2")])]})
        llm = adv.Nemotron(nim=self.nim(fc), model=model)
        ctx = adv.build_context(STATE, "episode_completed", {3})
        out = llm.decide(ctx, STATE)
        self.assertEqual(llm.trace, ["get_cue_effectiveness", "update_ankle_cue"])
        self.assertEqual(out["bpm"], 90)
        tool_msg = [m for m in fc.requests[1]["messages"] if m["role"] == "tool"][0]
        best = json.loads(tool_msg["content"])["fastest_setting"]
        self.assertEqual((best["cue_bpm"], best["haptic"]), (90, 1))

    def test_last_round_forces_the_cue_tool(self):
        model = "m"
        loop = [msg(tool_calls=[call("get_time_pattern", {}, f"c{i}")]) for i in range(3)]
        loop.append(msg(tool_calls=[call("update_ankle_cue", {"bpm": 100, "haptic": 0, "alert_level": "normal",
                                                              "signal_uncertain": False, "caretaker_brief": "ok",
                                                              "reasoning": "x"}, "c9")]))
        fc = FakeClient({model: loop})
        adv.Nemotron(nim=self.nim(fc), model=model).decide(adv.build_context(STATE, "episode_completed", {3}), STATE)
        last = fc.requests[-1]
        self.assertEqual(last["tool_choice"]["function"]["name"], "update_ankle_cue")
        self.assertEqual([t["function"]["name"] for t in last["tools"]], ["update_ankle_cue"])

    def test_cue_effectiveness_and_time_pattern(self):
        eff = adv.cue_effectiveness(STATE["episodes"])
        self.assertEqual(eff["fastest_setting"]["cue_bpm"], 90)
        self.assertEqual(adv.time_pattern(STATE["episodes"])["by_hour"], {"09": 2, "14": 1})


class Guards(Base):
    def test_local_boundary(self):
        self.assertFalse(nvidia_guard.local_ok("Consider adjusting the levodopa dose."))
        self.assertTrue(nvidia_guard.local_ok("StepCue does not diagnose or give medication advice."))

    def test_parse_safety_formats(self):
        self.assertFalse(nvidia_guard.parse_safety('{"User Safety": "safe", "Response Safety": "unsafe", '
                                                   '"Safety Categories": "Violence"}')["safe"])
        self.assertTrue(nvidia_guard.parse_safety('{"User Safety": "safe"}')["safe"])
        self.assertFalse(nvidia_guard.parse_safety("User Safety: safe\nResponse Safety: unsafe")["safe"])

    def test_topic_control_blocks_off_topic(self):
        fc = FakeClient({nvidia_nim.model_for("topic"): "off-topic", nvidia_nim.model_for("safety"): '{"User Safety": "safe"}'})
        r = nvidia_guard.check_question("Write me a poem about cars", self.nim(fc))
        self.assertEqual((r["ok"], r["reason"]), (False, "off-topic"))
        self.assertIn("nemoguard-topic", r["layers"])
        sys_prompt = fc.requests[0]["messages"][0]["content"]
        self.assertTrue(sys_prompt.endswith('You must respond with "on-topic" or "off-topic".'))


class Ask(Base):
    def test_offline_keyword_answer_cites_records(self):
        r = nvidia_ask.ask("How long was the 14:12 freeze?", STATE, nim=nvidia_nim.NIM(key=""))
        self.assertEqual(r["mode"], "local")
        self.assertIn("e3", r["cited"])

    def test_medical_question_refused_before_any_model(self):
        fc = FakeClient()
        r = nvidia_ask.ask("Should we change the medication dose?", STATE, nim=self.nim(fc))
        self.assertTrue(r["blocked"])
        self.assertEqual(fc.requests, [])

    def test_full_pipeline_embed_rerank_answer_guard(self):
        ask_model = nvidia_nim.model_for("ask")
        fc = FakeClient({nvidia_nim.model_for("topic"): "on-topic",
                         nvidia_nim.model_for("safety"): '{"User Safety": "safe", "Response Safety": "safe"}',
                         ask_model: "The longest freeze was 5.8 s at 14:12 [e3]."})
        nim = self.nim(fc)
        with mock.patch.object(nim, "rerank", side_effect=lambda q, texts: list(range(len(texts)))[::-1]):
            r = nvidia_ask.ask("What was the longest freeze?", STATE, nim=nim)
        self.assertEqual(r["retrieval"], "nemotron-embed+rerank")
        self.assertEqual(r["mode"], ask_model)
        self.assertEqual(r["cited"], ["e3"])
        self.assertEqual(r["layers"], ["local", "nemoguard-topic", "nemotron-safety"])
        kinds = [q.get("extra_body", {}).get("input_type") for q in fc.requests if "input" in q]
        self.assertIn("passage", kinds)
        self.assertIn("query", kinds)


class Translate(Base):
    def test_riva_prompt_and_cache(self):
        model = nvidia_nim.model_for("translate")
        fc = FakeClient({model: "Congelación de 5.8 s."})
        nim = self.nim(fc)
        r = nvidia_translate.translate("Freeze of 5.8 s.", "es", nim)
        self.assertEqual((r["text"], r["ok"]), ("Congelación de 5.8 s.", True))
        self.assertEqual(fc.requests[0]["messages"][0]["content"], "en-es-us")
        self.assertEqual(nvidia_translate.translate("Freeze of 5.8 s.", "es", nim)["source"], "cache")
        self.assertEqual(len(fc.requests), 1)


class Report(Base):
    def test_ultra_report_parsed_and_guarded(self):
        text = json.dumps({"headline": "3 freezes; recovery fastest at 90 bpm with vibration.",
                           "what_happened": ["3 episodes"], "what_the_device_did": ["Tempo 100 to 90"],
                           "patterns": ["2 of 3 began in turns", "Consider a medication review"],
                           "for_the_care_team": ["Mention turns at the next visit"]})
        fc = FakeClient({nvidia_nim.model_for("report"): "<think>…</think>" + text})
        path = os.path.join(self.tmp.name, "rep.json")
        rep = nvidia_report.write_report(STATE, [], None, nim=self.nim(fc), path=path)
        self.assertEqual(rep["source"], nvidia_nim.model_for("report"))
        self.assertEqual(rep["patterns"], ["2 of 3 began in turns"])        # medical line dropped
        self.assertEqual(nvidia_report.read_report(path)["headline"], rep["headline"])

    def test_falls_back_to_lightning_then_local(self):
        fc = FakeClient({nvidia_nim.model_for("report"): RuntimeError("busy"),
                         nvidia_nim.model_for("agent"): "not json"})
        rep = nvidia_report.write_report(STATE, [], None, nim=self.nim(fc), path=os.path.join(self.tmp.name, "r.json"))
        self.assertEqual(rep["source"], "local")
        self.assertEqual(len(rep["tried"]), 2)


class VoiceIn(Base):
    def wav(self):
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\0\0" * 1600)
        return base64.b64encode(buf.getvalue()).decode()

    def test_help_words_always_mean_help(self):
        r = nvidia_voice_in.parse('{"transcript": "I fell in the kitchen", "kind": "other"}')
        self.assertEqual(r["kind"], "help")

    def test_omni_request_carries_audio(self):
        model = nvidia_nim.model_for("voice_in")
        fc = FakeClient({model: '{"transcript": "walking is hard today", "kind": "walking", "walking_feel": "hard"}'})
        r = nvidia_voice_in.understand(self.wav(), self.nim(fc))
        self.assertEqual((r["kind"], r["walking_feel"]), ("walking", "hard"))
        part = fc.requests[0]["messages"][0]["content"][1]
        self.assertEqual((part["type"], part["input_audio"]["format"]), ("input_audio", "wav"))

    def test_rejects_non_wav(self):
        with self.assertRaises(ValueError):
            nvidia_voice_in.understand(base64.b64encode(b"hello").decode(), self.nim(FakeClient()))


if __name__ == "__main__":
    unittest.main()
