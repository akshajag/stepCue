"""Ask StepCue: caregiver questions answered from StepCue's own records with NVIDIA models.

Pipeline (each step is a separate NVIDIA model):
  1. guard the question       NemoGuard topic control + Nemotron Safety Guard (nvidia_guard)
  2. retrieve                 Nemotron Embed indexes today's episodes, the care agent's decisions,
                              phone check-ins and the StepCue guide; cosine search on the laptop
  3. answer                   Nemotron answers only from the retrieved passages, citing [ids]
  4. guard the answer         Nemotron Safety Guard + the local no-medical-advice check

Without a key it still works: keyword retrieval and a plain answer built from the top
passages, labelled "local".

Run:  python nvidia_ask.py "When did freezes happen today?"
"""
import hashlib
import json
import math
import os
import re
import sys
import time

import nvidia_guard
import nvidia_nim

GUIDE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "stepcue_guide.md")
CACHE_FILE = os.environ.get("STEPCUE_EMBED_CACHE", "embed_cache.json")
TOP_K = 6

SYSTEM = """You are Ask StepCue, a helper for the family and care team of someone who wears StepCue,
an assistive walking-cue wearable for freezing of gait. Answer the question using ONLY the passages
provided. Cite the passages you used like [e3] or [g2] right after the sentence that uses them.
If the passages don't contain the answer, say you don't have that information.
Be brief (2-4 sentences), concrete, and use numbers from the passages.
Never diagnose, never mention medication or treatment, never predict how a condition will progress.
Refer to the person as "the wearer"."""


# ------------------------------------------------------------- corpus -----
def guide_passages(path=GUIDE_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []
    out = []
    for i, block in enumerate(re.split(r"^## ", text, flags=re.M)[1:], 1):
        title, _, body = block.partition("\n")
        body = " ".join(body.split())
        if body:
            out.append({"id": f"g{i}", "kind": "guide", "title": title.strip(), "text": f"{title.strip()}: {body}"})
    return out


def episode_passage(e, companion=None):
    bits = [f"Freeze episode {e.get('id')} began at {e.get('timestamp', '?')} and lasted {e.get('duration_sec', 0):.1f} s."]
    if e.get("recovery_sec") is None:
        bits.append("It was still ongoing.")
    else:
        bits.append(f"Walking resumed {e['recovery_sec']:.1f} s after the cue started.")
    bits.append(f"Cue tempo {e.get('cue_bpm', '?')} bpm, wrist vibration {'on' if e.get('cue_haptic') else 'off'}.")
    if e.get("pre_freeze_cadence"):
        bits.append(f"Walking pace just before: {e['pre_freeze_cadence']} steps/min.")
    if e.get("during_turn"):
        bits.append("It began during a turn.")
    conf = ((companion or {}).get("confirmations") or {}).get(str(e.get("id")))
    if conf is not None:
        bits.append("The wearer confirmed it on the phone." if conf.get("froze")
                    else "The wearer said on the phone it was not a freeze (possible false alarm).")
    return " ".join(bits)


def decision_passage(d):
    dec = d.get("decision", {})
    who = "built-in rules" if d.get("source") == "rules" else "the Nemotron care agent"
    tools = d.get("tools") or []
    used = f" It checked {', '.join(t for t in tools if t != 'update_ankle_cue')} first." if len(tools) > 1 else ""
    return (f"At {d.get('time')} {who} reviewed episode(s) {d.get('episode_ids')}: cue tempo "
            f"{d.get('before', {}).get('bpm')} to {dec.get('bpm')} bpm, wrist vibration "
            f"{'on' if dec.get('haptic') else 'off'}, alert level {dec.get('alert_level')}.{used} "
            f"Reason: {dec.get('reasoning', '')} Brief: {dec.get('caretaker_brief', '')}")


def checkin_passage(c):
    when = time.strftime("%H:%M", time.localtime(c.get("t", 0)))
    kind = c.get("kind")
    text = {"ok": "tapped I'm OK", "help": "asked for help"}.get(kind, f"said walking feels {c.get('value')}")
    return f"At {when} the wearer {text} on the phone."


def build_corpus(state, decisions=(), companion=None):
    eps = state.get("episodes", [])
    done = [e for e in eps if e.get("recovery_sec") is not None]
    rec = [e["recovery_sec"] for e in done]
    summary = (f"Today so far: {len(eps)} freeze episode(s); "
               + (f"average time to walk again {sum(rec) / len(rec):.1f} s, fastest {min(rec):.1f} s, slowest {max(rec):.1f} s; "
                  if rec else "")
               + f"{sum(1 for e in eps if e.get('during_turn'))} began during a turn. "
               f"Current cue {state.get('bpm')} bpm, wrist vibration {'on' if state.get('haptic') else 'off'}, "
               f"alert level {state.get('alert_level')}. Status right now: {str(state.get('status', '')).lower()}.")
    m = state.get("motion") or {}
    if m.get("steps") is not None:
        summary += f" Steps this session: {m.get('steps')}."
    out = [{"id": "s1", "kind": "summary", "title": "Today", "text": summary}]
    out += [{"id": f"e{e.get('id')}", "kind": "episode", "title": f"Episode {e.get('id')}",
             "text": episode_passage(e, companion)} for e in eps[-40:]]
    out += [{"id": f"d{i}", "kind": "decision", "title": f"Agent {d.get('time')}", "text": decision_passage(d)}
            for i, d in enumerate(list(decisions)[:15], 1)]
    out += [{"id": f"c{i}", "kind": "checkin", "title": "Check-in", "text": checkin_passage(c)}
            for i, c in enumerate(((companion or {}).get("checkins") or [])[-10:], 1)]
    return out + guide_passages()


# ---------------------------------------------------------- retrieval -----
def _cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in on at and or is was were did do does how what when why which who it its i my "
            "for with this that today so far there any are be".split())


def keyword_rank(question, passages, k=TOP_K):
    q = {w for w in _WORD.findall(question.lower()) if w not in _STOP}
    scored = []
    for p in passages:
        words = set(_WORD.findall(p["text"].lower()))
        score = len(q & words) + (0.5 if p["kind"] == "summary" else 0)
        scored.append((score, p))
    scored.sort(key=lambda t: -t[0])
    return [p for s, p in scored[:k] if s > 0] or [p for _, p in scored[:2]]


class EmbedIndex:
    """Nemotron Embed vectors for passages, cached by text hash so each passage is embedded once."""

    def __init__(self, nim, path=None):
        self.nim, self.path = nim, path or CACHE_FILE
        try:
            with open(self.path, encoding="utf-8") as f:
                self.cache = json.load(f)
        except (OSError, ValueError):
            self.cache = {}

    @staticmethod
    def _h(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def vectors(self, passages):
        model = nvidia_nim.model_for("embed")
        missing = [p for p in passages if f"{model}:{self._h(p['text'])}" not in self.cache]
        for i in range(0, len(missing), 32):
            chunk = missing[i:i + 32]
            for p, v in zip(chunk, self.nim.embed([p["text"] for p in chunk], "passage")):
                self.cache[f"{model}:{self._h(p['text'])}"] = v
        if missing:
            if len(self.cache) > 2000:
                self.cache = dict(list(self.cache.items())[-1500:])
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
            os.replace(tmp, self.path)
        return [self.cache[f"{model}:{self._h(p['text'])}"] for p in passages]

    def rank(self, question, passages, k=TOP_K):
        vecs = self.vectors(passages)
        q = self.nim.embed([question], "query")[0]
        order = sorted(zip(passages, vecs), key=lambda t: -_cos(q, t[1]))
        return [p for p, _ in order[:k]]


# ------------------------------------------------------------- answer -----
def local_answer(question, top):
    facts = [p for p in top if p["kind"] != "guide"][:2] or top[:2]
    return " ".join(f"{p['text']} [{p['id']}]" for p in facts)


def ask(question, state, decisions=(), companion=None, nim=None):
    """Answer a caregiver question. Returns a dict the dashboard can show as-is."""
    nim = nim or nvidia_nim.shared()
    question = " ".join(str(question or "").split())[:400]
    t0 = time.monotonic()
    result = {"question": question, "answer": "", "sources": [], "layers": [], "mode": "local",
              "retrieval": "keyword", "blocked": False}
    if not question:
        result["answer"] = "Ask about today's freezes, the cue settings, alerts or how to use StepCue."
        return result

    gate = nvidia_guard.check_question(question, nim)
    result["layers"] = list(gate["layers"])
    if not gate["ok"]:
        result.update(answer=nvidia_guard.REFUSAL, blocked=True, reason=gate["reason"])
        result["latency_sec"] = round(time.monotonic() - t0, 2)
        return result

    passages = build_corpus(state, decisions, companion)
    top = None
    if nim.available:
        try:
            top = EmbedIndex(nim).rank(question, passages, k=TOP_K * 2)
            result["retrieval"] = "nemotron-embed"
        except Exception:
            top = None
        if top:
            try:
                order = nim.rerank(question, [p["text"] for p in top])
                top = [top[i] for i in order if 0 <= i < len(top)][:TOP_K] or top[:TOP_K]
                result["retrieval"] = "nemotron-embed+rerank"
            except Exception:
                top = top[:TOP_K]
    top = top or keyword_rank(question, passages)
    result["sources"] = [{"id": p["id"], "kind": p["kind"], "title": p["title"], "text": p["text"]} for p in top]

    answer = None
    if nim.available:
        ctx = "\n".join(f"[{p['id']}] {p['text']}" for p in top)
        try:
            answer = nim.text("ask", [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"}],
                              temperature=0.2, max_tokens=600, thinking=False)
            result["mode"] = nvidia_nim.model_for("ask")
        except Exception:
            answer = None
    answer = answer or local_answer(question, top)

    check = nvidia_guard.check_reply(answer, question, nim)
    for layer in check["layers"]:
        if layer not in result["layers"]:
            result["layers"].append(layer)
    if not check["ok"]:
        result.update(blocked=True, reason=check["reason"], answer=nvidia_guard.REFUSAL)
    else:
        result["answer"] = answer
    result["cited"] = sorted(set(re.findall(r"\[([egdcs]\d+)\]", result["answer"])))
    result["latency_sec"] = round(time.monotonic() - t0, 2)
    return result


def main():
    from state_io import read_state
    import companion_store as cs
    q = " ".join(sys.argv[1:]) or "How did walking go today?"
    decisions = []
    try:
        with open("advisor_log.jsonl", encoding="utf-8") as f:
            decisions = [json.loads(x) for x in f.readlines()[-15:]][::-1]
    except OSError:
        pass
    out = ask(q, read_state(), decisions, cs.read_companion())
    print(json.dumps({k: v for k, v in out.items() if k != "sources"}, indent=2))
    print("sources:", [s["id"] for s in out["sources"]])


if __name__ == "__main__":
    main()
