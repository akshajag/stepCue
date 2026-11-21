"""Briefs and alerts in the family's own language, with NVIDIA Riva Translate (riva-translate-4b-instruct).

Many caregivers of older adults are more comfortable in a language other than English.
`translate(text, "es")` returns the Spanish text, cached in translations.json so each
brief is translated once. The translation is checked by the local no-medical-advice
guard; if the model is unavailable the English text is returned with ok=False.
"""
import hashlib
import json
import os
import threading

import nvidia_guard
import nvidia_nim

CACHE_FILE = os.environ.get("STEPCUE_TRANSLATIONS", "translations.json")
CACHE_KEEP = 600

# code -> (English name, name in that language, Riva Translate target code)
LANGUAGES = {
    "en": ("English", "English", "en"),
    "es": ("Spanish", "Español", "es-us"),
    "zh": ("Simplified Chinese", "简体中文", "zh-cn"),
    "hi": ("Hindi", "हिन्दी", "hi"),
    "fr": ("French", "Français", "fr"),
    "de": ("German", "Deutsch", "de"),
    "pt": ("Brazilian Portuguese", "Português", "pt-br"),
    "ja": ("Japanese", "日本語", "ja"),
    "ko": ("Korean", "한국어", "ko"),
    "ru": ("Russian", "Русский", "ru"),
    "ar": ("Arabic", "العربية", "ar"),
}

_lock = threading.Lock()


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(cache, path):
    if len(cache) > CACHE_KEEP:
        cache = dict(list(cache.items())[-CACHE_KEEP:])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, path)


def prompt(text, lang):
    """Riva Translate's instruction format: the system message is "<source>-<target>" in lowercase."""
    return [{"role": "system", "content": f"en-{LANGUAGES[lang][2]}"},
            {"role": "user", "content": text}]


def cached(text, lang, path=None):
    if lang == "en" or not text:
        return text
    key = f"{lang}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"
    return _load(path or CACHE_FILE).get(key)


def translate(text, lang, nim=None, path=None):
    """Returns {"text", "lang", "ok", "source"}; source is "cache", the model id, or "english"."""
    path = path or CACHE_FILE
    if not text or lang == "en" or lang not in LANGUAGES:
        return {"text": text, "lang": "en", "ok": True, "source": "english"}
    key = f"{lang}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"
    with _lock:
        hit = _load(path).get(key)
    if hit:
        return {"text": hit, "lang": lang, "ok": True, "source": "cache"}
    nim = nim or nvidia_nim.shared()
    if not nim.available:
        return {"text": text, "lang": "en", "ok": False, "source": "english"}
    try:
        out = nim.text("translate", prompt(text, lang), temperature=0.0, max_tokens=400).strip().strip('"')
    except Exception:
        return {"text": text, "lang": "en", "ok": False, "source": "english"}
    if not out or not nvidia_guard.local_ok(out):
        return {"text": text, "lang": "en", "ok": False, "source": "english"}
    with _lock:
        cache = _load(path)
        cache[key] = out
        _save(cache, path)
    return {"text": out, "lang": lang, "ok": True, "source": nvidia_nim.model_for("translate")}
