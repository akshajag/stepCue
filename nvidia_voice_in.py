"""Spoken check-ins: the wearer talks to their phone, NVIDIA Nemotron Omni understands it.

Older adults often find speaking easier than finding the right button. The phone
records a few seconds (16 kHz WAV), the laptop sends it to Nemotron 3 Nano Omni
(audio in, JSON out), and the reply becomes a normal check-in:

  "I'm fine, just a bit tired"      -> ok
  "I fell, I can't get up"          -> help   (a spoken help request is never ignored)
  "walking was really hard today"   -> walking: hard

Safety rule: if the model is unsure but the words sound like distress, we choose
help. The transcript is kept on the laptop and shown to the caregiver; audio is
not stored.
"""
import base64
import json
import re

import nvidia_guard
import nvidia_nim

MAX_AUDIO_BYTES = 360_000           # about 11 s of 16 kHz mono 16-bit WAV
KINDS = ("ok", "help", "walking", "other")
FEEL = ("easy", "some trouble", "hard")
_HELP_WORDS = re.compile(r"\b(help|fell|fallen|fall|hurt|can't get up|cannot get up|emergency|pain|stuck|ambulance)\b", re.I)

PROMPT = """The audio is a short spoken check-in from an older adult who wears StepCue, a walking-cue wearable.
Transcribe exactly what they said, then classify it. Reply with ONLY this JSON:
{"transcript": "...", "kind": "ok" | "help" | "walking" | "other",
 "walking_feel": "easy" | "some trouble" | "hard" | null, "summary": "a few words for the caregiver"}
Rules:
- "help" if they ask for help, mention a fall, injury, pain, being stuck, or sound distressed. When in doubt between help and anything else, choose help.
- "walking" if they describe how walking or moving feels today; set walking_feel.
- "ok" if they say they're fine.
- "other" for anything else.
Never give medical advice."""


def wav_ok(data):
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def parse(text):
    m = re.search(r"\{.*\}", nvidia_nim.strip_thinking(text), re.S)
    raw = json.loads(m.group(0)) if m else {}
    transcript = " ".join(str(raw.get("transcript", "")).split())[:300]
    kind = str(raw.get("kind", "other")).lower()
    kind = kind if kind in KINDS else "other"
    feel = raw.get("walking_feel")
    feel = feel if feel in FEEL else None
    if kind == "walking" and not feel:
        kind = "other"
    if kind != "help" and _HELP_WORDS.search(transcript):     # safety first
        kind, feel = "help", None
    summary = " ".join(str(raw.get("summary", "")).split())[:120]
    if not nvidia_guard.local_ok(summary):
        summary = ""
    return {"transcript": transcript, "kind": kind, "walking_feel": feel, "summary": summary}


def understand(audio_b64, nim=None):
    """Returns {"transcript", "kind", "walking_feel", "summary", "model"}; raises ValueError on bad audio,
    nvidia_nim.NIMError if NVIDIA is not configured."""
    nim = nim or nvidia_nim.shared()
    try:
        data = base64.b64decode(audio_b64 or "", validate=True)
    except (ValueError, TypeError):
        raise ValueError("audio is not base64")
    if not wav_ok(data):
        raise ValueError("send a WAV recording")
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError("recording too long")
    model = nvidia_nim.model_for("voice_in")
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": "wav"}}]}]
    text = nim.text("voice_in", msgs, model=model, private=False, thinking=False,
                    temperature=0.0, max_tokens=300)
    out = parse(text)
    out["model"] = model
    return out
