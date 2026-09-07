"""Provider selection for the clip-selection LLM call.

Why this exists: the whole video->shorts pipeline depends on exactly ONE model
call (``main.get_viral_clips``), and it is text-only with a JSON schema. Every
other AI stage in this repo already runs locally (faster-whisper, TransNetV2,
MediaPipe, YOLO), so serving that one call from a local Ollama is the whole
difference between "needs a paid API key" and "runs offline".

The Ollama client here DUCK-TYPES the google-genai surface
(``client.models.generate_content(model=, contents=, config=)`` returning an
object with ``.parsed`` / ``.text`` / ``.candidates`` / ``.usage_metadata``)
instead of introducing a neutral interface. That is deliberate:
``main._run_gemini_stage`` and ``_run_stage_split`` encode two production
incidents in their retry and bisect logic (22-jul-2026 empty bodies,
27-aug-2026 prompt-filter combinations), and their tests inject exactly this
shape. Matching it means neither the logic nor its tests change at all — the
swap is one line at the client construction site.

Standard-library + httpx + pydantic only. No cv2/torch/google.genai at import
time, so this module imports in the thin CI env like ``clip_selection`` does;
the Gemini SDK is imported lazily inside the Gemini branch.
"""
import copy
import json
import os
import time
from typing import Optional, Tuple

import httpx
from pydantic import BaseModel

# Candidates tried in order when OLLAMA_BASE_URL is unset: the first works from
# inside a Docker container on Desktop, the second from a bare-metal uvicorn.
_BASE_URL_CANDIDATES = ("http://host.docker.internal:11434", "http://localhost:11434")

DEFAULT_MODEL = "qwen2.5:7b-instruct"
_PROBE_TTL_SECONDS = 60.0

# Cache: (base_url, model_present, expires_at). A cold start that fails should
# not pin the process to Gemini forever — starting Ollama later must be enough.
_probe_cache: Optional[Tuple[Optional[str], bool, float]] = None
_http: Optional[httpx.Client] = None


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def _model_name() -> str:
    return os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL


def _num_ctx() -> int:
    try:
        return int(os.environ.get("OLLAMA_NUM_CTX") or 8192)
    except ValueError:
        return 8192


def _timeout() -> httpx.Timeout:
    """httpx defaults every phase to 5s, which would fail every generation:
    a cold 4.7GB model load alone takes 10-30s."""
    try:
        read = float(os.environ.get("OLLAMA_TIMEOUT") or 300)
    except ValueError:
        read = 300.0
    return httpx.Timeout(connect=5.0, read=read, write=30.0, pool=5.0)


def _client_http() -> httpx.Client:
    """One pooled client for the process — pairs with keep_alive so eight
    scoring calls share a single loaded model and a single connection."""
    global _http
    if _http is None:
        _http = httpx.Client(timeout=_timeout())
    return _http


# --------------------------------------------------------------------------
# schema handling
# --------------------------------------------------------------------------

def inline_refs(schema: dict) -> dict:
    """Resolve ``$ref``/``$defs`` and drop ``title`` keys.

    Ollama accepts $ref (verified against ScoreResponse), so this is not a
    workaround for a failure — it just hands llama.cpp's schema->GBNF converter
    a flat grammar and strips titles that are pure token noise. Both schemas
    here are shallow and non-recursive, so the walk is total.
    """
    defs = schema.get("$defs", {}) or {}

    def walk(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.rsplit("/", 1)[-1])
                if target is not None:
                    return walk(copy.deepcopy(target))
            return {k: walk(v) for k, v in node.items() if k not in ("title", "$defs")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _pydantic_schema(schema) -> Optional[type]:
    """The schema only counts when it is a real BaseModel subclass.

    ``_run_gemini_stage`` is called with ``schema=object`` and ``schema=None``
    from the test suite; in those cases we send no ``format`` at all and let the
    caller's text-parsing fallback handle the body."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    return None


# --------------------------------------------------------------------------
# duck-typed response
# --------------------------------------------------------------------------

class _Usage:
    """Shaped for ``gemini_worker._calculate_cost_analysis``."""

    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_token_count = prompt_tokens or 0
        self.candidates_token_count = completion_tokens or 0
        self.thoughts_token_count = 0


class _OllamaResponse:
    """Mirrors the google-genai response surface that main.py reads.

    Deliberately has NO ``prompt_feedback`` attribute and an empty
    ``candidates`` list: ``gemini_worker.raise_if_blocked`` reads both through
    ``getattr(..., None)``, so it no-ops and the content-policy bisect in
    ``_run_stage_split`` is simply never reached. Ollama has no content policy,
    so "the mechanism never fires" is the correct behaviour, not a special case.
    """

    def __init__(self, text: str, parsed, usage: Optional[_Usage]):
        self.text = text
        self.parsed = parsed
        self.candidates = []
        self.usage_metadata = usage


# --------------------------------------------------------------------------
# ollama client
# --------------------------------------------------------------------------

class _OllamaModels:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate_content(self, model=None, contents=None, config=None, **_ignored):
        if not isinstance(contents, str):
            # Multimodal parts (frames, uploaded video) never reach this path —
            # only the text-only clip-selection call is routed to Ollama.
            raise RuntimeError(
                "The local provider only serves text prompts; this call sent "
                f"{type(contents).__name__}. Set LLM_PROVIDER=gemini for it.")

        schema = _pydantic_schema(getattr(config, "response_schema", None))
        num_ctx = _num_ctx()
        options = {"num_ctx": num_ctx, "temperature": _temperature_for(schema)}
        seed = os.environ.get("OLLAMA_SEED")
        if seed:
            try:
                options["seed"] = int(seed)
            except ValueError:
                pass

        body = {
            "model": (model or _model_name()).split("ollama/", 1)[-1],
            "messages": [{"role": "user", "content": contents}],
            "stream": False,
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE") or "10m",
            "options": options,
        }
        if schema is not None:
            body["format"] = inline_refs(schema.model_json_schema())

        data = self._post(body)

        prompt_tokens = data.get("prompt_eval_count") or 0
        # Silent front-truncation is the single most likely way this ships
        # "working" while quietly scoring windows the model never saw.
        if prompt_tokens and prompt_tokens > 0.9 * num_ctx:
            print(f"⚠️  Ollama prompt used {prompt_tokens}/{num_ctx} context tokens "
                  f"(>90%). Raise OLLAMA_NUM_CTX or lower the batch size — the "
                  f"prompt may be silently truncated.")

        if data.get("done_reason") == "length":
            # Deterministic and expensive to repeat: fail fast with the remedy
            # rather than burning three attempts on the same truncation.
            raise RuntimeError(
                "Ollama truncated the answer (done_reason=length). Raise "
                "OLLAMA_NUM_CTX or lower SCORE_BATCH in main.get_viral_clips.")

        content = ((data.get("message") or {}).get("content") or "").strip()
        if not content:
            # Reuses a token the caller's retry loop already treats as transient.
            raise ValueError("Ollama returned an empty response body.")

        usage = _Usage(prompt_tokens, data.get("eval_count") or 0)

        parsed = None
        if schema is not None:
            try:
                parsed = schema.model_validate(json.loads(content))
            except Exception:
                # Leave .parsed as None: main._run_gemini_stage then falls back
                # to gemini_worker._parse_json_response_text, which either
                # recovers the dict or raises a message the retry loop knows.
                parsed = None

        return _OllamaResponse(text=content, parsed=parsed, usage=usage)

    def _post(self, body: dict) -> dict:
        url = f"{self.base_url}/api/chat"
        try:
            r = _client_http().post(url, json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            # Transient on purpose: covers a daemon restart or a cold load.
            raise RuntimeError(f"503 UNAVAILABLE: cannot reach Ollama at {url}: {e}")
        except httpx.ReadTimeout as e:
            raise RuntimeError(f"504 UNAVAILABLE: Ollama read timeout: {e}")

        if r.status_code >= 500:
            raise RuntimeError(f"{r.status_code} INTERNAL: {r.text[:200]}")
        if r.status_code >= 400:
            # No retry token: a missing model or a bad schema is deterministic,
            # and should fail in 2s instead of after 35s of backoff.
            raise RuntimeError(
                f"Ollama rejected the request ({r.status_code}): {r.text[:300]}. "
                f"Check that model '{body.get('model')}' is pulled "
                f"(ollama pull {body.get('model')}).")
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"Ollama returned an empty response body (unparseable): {e}")


def _temperature_for(schema) -> float:
    """Scoring wants precision, detail wants some creativity.

    ``_run_gemini_stage`` sends no temperature at all, but the repo already
    reasons about this split in ``gemini_worker._config_for_strategy``. The
    schema identifies the stage, so we recover the distinction without touching
    the call site. Matched by name to avoid importing gemini_worker (which
    would pull in the Gemini SDK).
    """
    name = getattr(schema, "__name__", "")
    env = "OLLAMA_TEMPERATURE_DETAIL" if name == "DetailResponse" else "OLLAMA_TEMPERATURE_SCORE"
    default = 0.7 if name == "DetailResponse" else 0.2
    try:
        return float(os.environ.get(env) or default)
    except ValueError:
        return default


class OllamaClient:
    """google-genai-shaped client backed by Ollama's native /api/chat.

    Native rather than the OpenAI-compatible /v1 endpoint because ``num_ctx`` is
    native-only: without it the server default (possibly 4096) silently
    truncates a 3-4k-token prompt with no error. Native also surfaces
    ``done_reason``, ``prompt_eval_count`` and ``eval_count`` directly.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.models = _OllamaModels(base_url)


# --------------------------------------------------------------------------
# provider resolution
# --------------------------------------------------------------------------

def _probe(base_url: str, timeout: float = 1.5):
    """Return the model-name list at ``base_url``, or None if unreachable."""
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if r.status_code != 200:
            return None
        return [m.get("name") for m in (r.json().get("models") or [])]
    except Exception:
        return None


def _resolve_local(force: bool = False) -> Tuple[Optional[str], bool]:
    """(base_url, model_present). Cached briefly so a per-request availability
    check does not probe on every call."""
    global _probe_cache
    now = time.time()
    if not force and _probe_cache and _probe_cache[2] > now:
        return _probe_cache[0], _probe_cache[1]

    configured = os.environ.get("OLLAMA_BASE_URL")
    candidates = [configured] if configured else list(_BASE_URL_CANDIDATES)

    base_url, present = None, False
    for candidate in candidates:
        names = _probe(candidate)
        if names is None:
            continue
        base_url = candidate
        present = _model_name() in names
        break

    _probe_cache = (base_url, present, now + _PROBE_TTL_SECONDS)
    return base_url, present


def local_available() -> bool:
    """True when Ollama answers AND the configured model is actually pulled.

    Checking the model list matters: a reachable daemon without the model turns
    every generation into a 404 at job time instead of a clear message now.
    """
    base_url, present = _resolve_local()
    return bool(base_url) and present


async def local_available_async() -> bool:
    """Async twin of :func:`local_available`, for the FastAPI request path."""
    global _probe_cache
    now = time.time()
    if _probe_cache and _probe_cache[2] > now:
        return bool(_probe_cache[0]) and _probe_cache[1]

    configured = os.environ.get("OLLAMA_BASE_URL")
    candidates = [configured] if configured else list(_BASE_URL_CANDIDATES)
    base_url, present = None, False
    async with httpx.AsyncClient(timeout=1.5) as client:
        for candidate in candidates:
            try:
                r = await client.get(f"{candidate.rstrip('/')}/api/tags")
                if r.status_code != 200:
                    continue
                names = [m.get("name") for m in (r.json().get("models") or [])]
            except Exception:
                continue
            base_url, present = candidate, _model_name() in names
            break

    _probe_cache = (base_url, present, now + _PROBE_TTL_SECONDS)
    return bool(base_url) and present


def resolve_provider() -> str:
    """'ollama' or 'gemini'. Raises when neither can serve the call."""
    choice = (os.environ.get("LLM_PROVIDER") or "auto").strip().lower()

    if choice == "ollama":
        return "ollama"
    if choice == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
        return "gemini"

    # auto: local first, and only fall back to a paid API when it cannot serve.
    base_url, present = _resolve_local()
    if base_url and present:
        return "ollama"
    if os.environ.get("GEMINI_API_KEY"):
        if base_url and not present:
            print(f"⚠️  Ollama is up at {base_url} but model '{_model_name()}' is "
                  f"not pulled — falling back to Gemini.")
        else:
            print("⚠️  Ollama unreachable — falling back to Gemini.")
        return "gemini"

    raise RuntimeError(
        f"No LLM provider available. Either start Ollama and pull the model "
        f"(ollama pull {_model_name()}), or set GEMINI_API_KEY.")


def make_client():
    """Return ``(client, model_name)`` for the clip-selection call.

    ``model_name`` carries the ``ollama/`` prefix for local runs so the cost
    lookup in ``clip_selection.lookup_model_prices`` reports $0 and the UI shows
    which brain actually picked the clips.
    """
    provider = resolve_provider()

    if provider == "ollama":
        base_url, _ = _resolve_local()
        if not base_url:
            base_url = os.environ.get("OLLAMA_BASE_URL") or _BASE_URL_CANDIDATES[0]
        model = _model_name()
        print(f"🧠 Provider: ollama ({model}) at {base_url}")
        return OllamaClient(base_url), f"ollama/{model}"

    # Imported lazily so a keyless local install never needs the SDK at import.
    from google import genai
    model = os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite"
    print(f"🧠 Provider: gemini ({model})")
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY")), model
