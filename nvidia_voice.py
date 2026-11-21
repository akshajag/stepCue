"""Spoken walking cues for the wearer's phone, made with NVIDIA Riva.

Auditory cueing (a calm voice plus a steady beat) is a common way to help someone
start walking again after a freeze. This script builds a small set of voice cues
once, ahead of time, so the phone can play them instantly and offline:

  1. Riva Translate (riva-translate-4b-instruct) turns the English phrases into the
     wearer's language (es, fr, de, zh, hi, ...),
  2. Riva Magpie TTS (magpie-tts-multilingual, hosted on NVCF over gRPC) speaks them
     in a calm voice,
  3. the WAV files land in companion/voice/<lang>/ with a manifest.json that the
     phone page reads. During a freeze the phone plays the cue, then a metronome
     click at the cue tempo.

Nothing personal is sent: the phrases are fixed text.

Run:   python nvidia_voice.py                 # English
       python nvidia_voice.py --lang es fr    # more languages
       python nvidia_voice.py --list-voices
Needs: pip install nvidia-riva-client, and NVIDIA_API_KEY in .env.
"""
import argparse
import json
import os
import time
import wave

import nvidia_nim

TTS_URI = os.environ.get("RIVA_TTS_URI", "grpc.nvcf.nvidia.com:443")
TTS_FUNCTION_ID = os.environ.get("RIVA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")
SAMPLE_RATE = 22050
VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companion", "voice")

PHRASES = {
    "freeze": "Let's walk together. Step with the beat.",
    "count": "Left. Right. Left. Right.",
    "big_step": "Take a big step forward.",
    "turn": "Turn slowly, in small steps.",
    "resumed": "You're walking again. Well done.",
}

# app language -> (Riva locale, preferred voices; the first one the server offers is used)
LOCALES = {
    "en": ("en-US", ["Magpie-Multilingual.EN-US.Aria.Calm", "Magpie-Multilingual.EN-US.Aria",
                     "Magpie-Multilingual.EN-US.Sofia", "Magpie-Multilingual.EN-US.Leo"]),
    "es": ("es-US", ["Magpie-Multilingual.ES-US.Isabela", "Magpie-Multilingual.ES-US.Diego"]),
    "fr": ("fr-FR", ["Magpie-Multilingual.FR-FR.Louise", "Magpie-Multilingual.FR-FR.Pascal"]),
    "de": ("de-DE", []),
    "zh": ("zh-CN", []),
    "hi": ("hi-IN", []),
    "ja": ("ja-JP", []),
    "ko": ("ko-KR", []),
    "pt": ("pt-BR", []),
    "ar": ("ar-AR", []),
}


def log(msg):
    print(f"[voice {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def riva_service(key):
    import riva.client
    auth = riva.client.Auth(uri=TTS_URI, use_ssl=True, metadata_args=[
        ["function-id", TTS_FUNCTION_ID], ["authorization", f"Bearer {key}"]])
    return riva.client.SpeechSynthesisService(auth)


def list_voices(service):
    """Voice names the server offers, grouped by locale."""
    from riva.client.proto import riva_tts_pb2
    cfg = service.stub.GetRivaSynthesisConfig(riva_tts_pb2.RivaSynthesisConfigRequest(),
                                              metadata=service.auth.get_auth_metadata())
    voices = {}
    for model in cfg.model_config:
        params = model.parameters
        lang = params.get("language_code", "?")
        for name in params.get("voice_name", "").split(","):
            if name.strip():
                voices.setdefault(lang, []).append(name.strip())
        sub = params.get("subvoices", "")
        base = params.get("voice_name", "")
        for sv in sub.split(","):
            if sv.strip() and base:
                voices.setdefault(lang, []).append(f"{base}.{sv.split(':')[0].strip()}")
    return voices


def pick_voice(lang, offered):
    locale, preferred = LOCALES[lang]
    pool = [v for vs in offered.values() for v in vs] if offered else []
    for v in preferred:
        if not pool or v in pool:
            return locale, v
    match = [v for v in pool if f".{locale.upper()}." in v.upper()]
    return locale, (match[0] if match else None)


def write_wav(path, pcm, rate=SAMPLE_RATE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def phrases_for(lang, nim):
    if lang == "en":
        return dict(PHRASES), "original"
    import nvidia_translate
    out, src = {}, "riva-translate"
    for k, text in PHRASES.items():
        r = nvidia_translate.translate(text, lang, nim)
        if not r["ok"]:
            raise RuntimeError(f"translation to {lang} failed")
        out[k] = r["text"]
    return out, src


def build(lang, service, nim, offered=None, out_dir=VOICE_DIR):
    import riva.client
    locale, voice = pick_voice(lang, offered or {})
    texts, how = phrases_for(lang, nim)
    files = {}
    for key, text in texts.items():
        t0 = time.monotonic()
        try:
            resp = service.synthesize(text, voice, locale, sample_rate_hz=SAMPLE_RATE,
                                      encoding=riva.client.AudioEncoding.LINEAR_PCM)
        except Exception as exc:
            nvidia_nim.record("tts", "magpie-tts-multilingual", time.monotonic() - t0, False, str(exc))
            raise
        nvidia_nim.record("tts", "magpie-tts-multilingual", time.monotonic() - t0, True)
        rel = f"{lang}/{key}.wav"
        write_wav(os.path.join(out_dir, lang, f"{key}.wav"), resp.audio)
        files[key] = {"file": rel, "text": text}
        log(f"{lang} {key}: {text!r} -> {rel}")
    return {"lang": lang, "locale": locale, "voice": voice, "phrases": how, "files": files,
            "made": time.strftime("%Y-%m-%d %H:%M"), "by": "NVIDIA Riva Magpie TTS"}


def read_manifest(out_dir=VOICE_DIR):
    try:
        with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_manifest(man, out_dir=VOICE_DIR):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))


def main():
    ap = argparse.ArgumentParser(description="Make StepCue voice cues with NVIDIA Riva")
    ap.add_argument("--lang", nargs="+", default=["en"], choices=sorted(LOCALES))
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()
    key = nvidia_nim.api_key()
    if not key:
        raise SystemExit("Set NVIDIA_API_KEY in .env first (free key at build.nvidia.com).")
    service = riva_service(key)
    try:
        offered = list_voices(service)
    except Exception as exc:
        log(f"could not list voices ({type(exc).__name__}); using the default voice names")
        offered = {}
    if a.list_voices:
        print(json.dumps(offered, indent=1))
        return
    man = read_manifest()
    nim = nvidia_nim.NIM(key=key)
    for lang in a.lang:
        man[lang] = build(lang, service, nim, offered)
        save_manifest(man)
    log(f"done: {', '.join(man)} in {VOICE_DIR}")


if __name__ == "__main__":
    main()
