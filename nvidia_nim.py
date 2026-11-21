"""One door to NVIDIA: the model registry, the API client, the call ledger and the privacy scrub.

Every StepCue feature that talks to NVIDIA goes through `NIM` so that:
  * the right model is used for each job (MODELS below, each overridable from .env),
  * nothing personal leaves the laptop (`scrub` removes names, emails, phone
    numbers, coordinates and street addresses before any request),
  * every call is recorded in nvidia_calls.jsonl (model, job, latency, ok) for the
    dashboard's "NVIDIA stack" panel, and
  * with no NVIDIA_API_KEY, or when a call fails, features fall back to local
    rules and say so, instead of breaking the demo.

The API is NVIDIA NIM's OpenAI-compatible endpoint (integrate.api.nvidia.com/v1).
Get a free key at build.nvidia.com and put NVIDIA_API_KEY=nvapi-... in .env.
"""
import json
import os
import re
import threading
import time

BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
RERANK_URL = os.environ.get("NIM_RERANK_URL", "https://ai.api.nvidia.com/v1/retrieval")
LEDGER_FILE =os.environ.get("STEPCUE_NVIDIA_LEDGER", "nvidia_calls.jsonl")
LEDGER_KEEP = 400

# job -> (env var, default model, what it does for StepCue)
MODELS = {
    "agent":     ("NEMOTRON_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b",
                  "Real-time care agent: reads the sensor history with tools, tunes the cue, writes the brief"),
    "report":    ("NEMOTRON_REPORT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b",
                  "Deep daily care report for the care team (slow, thorough reasoning)"),
    "ask":       ("NEMOTRON_ASK_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b",
                  "Answers caregiver questions from retrieved StepCue records"),
    "voice_in":  ("NEMOTRON_OMNI_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                  "Understands a spoken check-in from the wearer's phone"),
    "embed":     ("NEMOTRON_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b",
                  "Embeds episodes and guide passages for retrieval"),
    "rerank":    ("NEMOTRON_RERANK_MODEL", "nvidia/nemotron-3.5-rerank-8b",
                  "Re-orders retrieved passages so the answer uses the most relevant records"),
    "safety":    ("NEMOTRON_SAFETY_MODEL", "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
                  "Screens every message before a family member or caregiver sees it"),
    "topic":     ("NEMOGUARD_TOPIC_MODEL", "nvidia/llama-3.1-nemoguard-8b-topic-control",
                  "Keeps 'Ask StepCue' on mobility and the device, never diagnosis or medication"),
    "translate": ("RIVA_TRANSLATE_MODEL", "nvidia/riva-translate-4b-instruct-v2",
                  "Translates briefs and alerts into the family's language"),
    "tts":       ("RIVA_TTS_VOICE", "magpie-tts-multilingual",
                  "Speaks walking cues on the wearer's phone (Riva Magpie TTS)"),
}


def model_for(job):
    env, default, _ = MODELS[job]
    return os.environ.get(env) or default


def load_dotenv(path=".env"):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def api_key():
    load_dotenv()
    return os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY") or ""


# ------------------------------------------------------------ privacy -----
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){9,14}\d(?!\d)")
_COORD = re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}")
_ADDRESS = re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Dr|Drive|Ln|Lane|Way|Ct|Court)\b\.?")


def known_names(users_path=None):
    """First and last names of everyone with an account, so they can be removed from requests."""
    path = users_path or os.environ.get("STEPCUE_USERS", "users.json")
    try:
        with open(path, encoding="utf-8") as f:
            users = json.load(f).get("users", [])
    except (OSError, ValueError):
        return []
    names = set()
    for u in users:
        for part in str(u.get("name", "")).split():
            if len(part) >= 3:
                names.add(part)
    extra = os.environ.get("STEPCUE_PRIVATE_NAMES", "")
    names.update(n.strip() for n in extra.split(",") if len(n.strip()) >= 3)
    return sorted(names, key=len, reverse=True)


def scrub(text, names=None):
    """Remove personal details from text before it leaves the laptop."""
    if not isinstance(text, str) or not text:
        return text
    out = _EMAIL.sub("[email]", text)
    out = _COORD.sub("[location]", out)
    out = _ADDRESS.sub("[address]", out)
    out = _PHONE.sub("[phone]", out)
    for n in (known_names() if names is None else names):
        out = re.sub(rf"\b{re.escape(n)}\b", "the wearer", out, flags=re.I)
    out = re.sub(r"\bthe wearer(\s+the wearer)+\b", "the wearer", out)
    return out


def scrub_obj(obj, names=None):
    names = known_names() if names is None else names
    if isinstance(obj, str):
        return scrub(obj, names)
    if isinstance(obj, list):
        return [scrub_obj(v, names) for v in obj]
    if isinstance(obj, dict):
        return {k: scrub_obj(v, names) for k, v in obj.items()}
    return obj


# ------------------------------------------------------------- ledger -----
_ledger_lock = threading.Lock()


def record(job, model, latency, ok, note="", path=None):
    path = path or LEDGER_FILE
    entry = {"t": round(time.time(), 2), "job": job, "model": model,
             "latency_sec": round(latency, 3), "ok": bool(ok), "note": str(note)[:160]}
    with _ledger_lock:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if os.path.getsize(path) > 200_000:            # keep the file small
                with open(path, encoding="utf-8") as f:
                    lines = f.readlines()[-LEDGER_KEEP:]
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                os.replace(tmp, path)
        except OSError:
            pass
    return entry


def read_ledger(limit=200, path=None):
    path = path or LEDGER_FILE
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def ledger_summary(entries, now=None, window=3600):
    """Per-job call counts, success rate and median latency over the last hour."""
    now = time.time() if now is None else now
    jobs = {}
    for e in entries:
        if now - e.get("t", 0) > window:
            continue
        j = jobs.setdefault(e.get("job", "?"), {"calls": 0, "ok": 0, "lat": [], "model": e.get("model")})
        j["calls"] += 1
        j["ok"] += 1 if e.get("ok") else 0
        if e.get("ok"):
            j["lat"].append(e.get("latency_sec", 0))
        j["model"] = e.get("model") or j["model"]
    for j in jobs.values():
        lat = sorted(j.pop("lat"))
        j["median_sec"] = lat[len(lat) // 2] if lat else None
    return jobs


# ------------------------------------------------------------- client -----
class NIMError(RuntimeError):
    pass


class NIM:
    """Thin wrapper over the OpenAI client pointed at NVIDIA NIM.

    `client` can be injected for tests (anything with .chat.completions.create and
    .embeddings.create). Without a key `available` is False and calls raise NIMError,
    which every feature catches and turns into its local fallback.
    """

    def __init__(self, key=None, client=None, timeout=30.0, ledger=None):
        self.key = api_key() if key is None else key
        self.ledger = ledger
        self._client = client
        if client is None and self.key:
            from openai import OpenAI
            self._client = OpenAI(base_url=BASE_URL, api_key=self.key, timeout=timeout, max_retries=1)

    @property
    def available(self):
        return self._client is not None

    def _call(self, job, model, fn):
        if not self.available:
            raise NIMError("NVIDIA_API_KEY not set")
        t0 = time.monotonic()
        try:
            out = fn()
        except Exception as exc:
            record(job, model, time.monotonic() - t0, False, f"{type(exc).__name__}: {exc}", self.ledger)
            raise
        record(job, model, time.monotonic() - t0, True, "", self.ledger)
        return out

    def chat(self, job, messages, model=None, private=True, thinking=None, **kwargs):
        """Chat completion. Messages are scrubbed of personal details unless private=False.
        thinking=True/False switches Nemotron reasoning on or off (chat_template_kwargs.enable_thinking)."""
        model = model or model_for(job)
        if thinking is not None:
            extra = dict(kwargs.pop("extra_body", None) or {})
            extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(thinking)
            kwargs["extra_body"] = extra
        if private:
            names = known_names()
            messages = [dict(m, content=scrub_obj(m.get("content"), names)) if m.get("content") else m
                        for m in messages]
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("max_tokens", 1024)
        return self._call(job, model, lambda: self._client.chat.completions.create(
            model=model, messages=messages, **kwargs))

    def text(self, job, messages, **kwargs):
        resp = self.chat(job, messages, **kwargs)
        return strip_thinking(resp.choices[0].message.content or "")

    def embed(self, texts, input_type="passage", model=None):
        model = model or model_for("embed")
        names = known_names()
        texts = [scrub(t, names) for t in texts]
        resp = self._call("embed", model, lambda: self._client.embeddings.create(
            model=model, input=texts, encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"}))
        return [d.embedding for d in resp.data]

    def rerank(self, query, texts, model=None, http=None):
        """Nemotron Rerank: returns passage indexes, most relevant first."""
        model = model or model_for("rerank")
        names = known_names()
        body = {"model": model, "query": {"text": scrub(query, names)},
                "passages": [{"text": scrub(t, names)} for t in texts], "truncate": "END"}
        url = f"{RERANK_URL}/{model}/reranking"

        def post():
            if http is not None:
                return http(url, body)
            import httpx
            r = httpx.post(url, json=body, timeout=20.0,
                           headers={"Authorization": f"Bearer {self.key}", "Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        if not self.key and http is None:
            raise NIMError("NVIDIA_API_KEY not set")
        out = self._call("rerank", model, post)
        return [r["index"] for r in sorted(out.get("rankings", []), key=lambda r: -r.get("logit", 0))]


def strip_thinking(text):
    """Drop a <think>...</think> block that reasoning models may put before the answer."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.split("</think>")[-1].strip()


_shared = None


def shared():
    global _shared
    if _shared is None:
        _shared = NIM()
    return _shared
