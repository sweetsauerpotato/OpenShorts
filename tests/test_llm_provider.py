"""Contract tests for the Ollama provider shim.

No live Ollama: every HTTP exchange is faked with httpx.MockTransport, so these
run anywhere llm_provider imports (stdlib + httpx + pydantic only).

The point of these tests is that the shim keeps the promises main.py's retry
loop and gemini_worker's helpers silently rely on.
"""
import json

import httpx
import pytest
from pydantic import BaseModel

import llm_provider


# --- schemas mirroring gemini_worker's (importing it would pull the Gemini SDK)

class _Row(BaseModel):
    id: str
    score: int


class ScoreLike(BaseModel):
    windows: list[_Row]


class DetailResponse(BaseModel):
    """Name matters: the shim picks its temperature from the schema name."""
    shorts: list[_Row]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Provider probes are cached and the http client is a singleton."""
    llm_provider._probe_cache = None
    llm_provider._http = None
    for var in ("LLM_PROVIDER", "GEMINI_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
                "OLLAMA_NUM_CTX", "OLLAMA_SEED", "OLLAMA_TEMPERATURE_SCORE",
                "OLLAMA_TEMPERATURE_DETAIL", "OLLAMA_KEEP_ALIVE"):
        monkeypatch.delenv(var, raising=False)
    yield
    llm_provider._probe_cache = None
    llm_provider._http = None


def _mount(handler):
    """Point the shim's pooled client at a mock transport."""
    llm_provider._http = httpx.Client(transport=httpx.MockTransport(handler))


def _ok(content, **extra):
    body = {"message": {"content": content}, "done_reason": "stop",
            "prompt_eval_count": 100, "eval_count": 20}
    body.update(extra)
    return httpx.Response(200, json=body)


def _models(schema=ScoreLike, capture=None):
    def handler(request):
        if capture is not None:
            capture.append(json.loads(request.content))
        return _ok(json.dumps({"windows": [{"id": "w1", "score": 90}]}))
    _mount(handler)
    return llm_provider.OllamaClient("http://fake:11434").models


class _Cfg:
    """Stands in for genai_types.GenerateContentConfig."""
    def __init__(self, response_schema):
        self.response_schema = response_schema


# --- schema handling -------------------------------------------------------

def test_inline_refs_flattens_defs_and_drops_titles():
    inlined = llm_provider.inline_refs(ScoreLike.model_json_schema())
    assert "$defs" not in inlined
    assert "title" not in inlined
    items = inlined["properties"]["windows"]["items"]
    assert "$ref" not in items
    assert items["properties"]["score"]["type"] == "integer"
    assert set(items["required"]) == {"id", "score"}


def test_inline_refs_is_sent_as_format():
    sent = []
    models = _models(capture=sent)
    models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    fmt = sent[0]["format"]
    assert "$defs" not in json.dumps(fmt)
    assert fmt["properties"]["windows"]["items"]["type"] == "object"


@pytest.mark.parametrize("schema", [object, None, "nonsense"])
def test_non_pydantic_schema_sends_no_format(schema):
    """_run_gemini_stage is called with schema=object and schema=None by the
    existing suite; those must not become a bad grammar."""
    sent = []
    models = _models(capture=sent)
    resp = models.generate_content(model="m", contents="hi", config=_Cfg(schema))
    assert "format" not in sent[0]
    assert resp.parsed is None          # nothing to validate against
    assert resp.text                    # text path still available


# --- response surface ------------------------------------------------------

def test_parsed_is_a_validated_model():
    models = _models()
    resp = models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert isinstance(resp.parsed, ScoreLike)
    # main._run_gemini_stage calls .model_dump() on it
    assert resp.parsed.model_dump()["windows"][0]["score"] == 90


def test_invalid_json_leaves_parsed_none_so_caller_falls_back():
    _mount(lambda r: _ok("not json at all"))
    models = llm_provider.OllamaClient("http://fake:11434").models
    resp = models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert resp.parsed is None
    assert resp.text == "not json at all"


def test_schema_mismatch_leaves_parsed_none():
    _mount(lambda r: _ok(json.dumps({"unexpected": 1})))
    models = llm_provider.OllamaClient("http://fake:11434").models
    resp = models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert resp.parsed is None


def test_usage_metadata_shape_matches_cost_helper():
    models = _models()
    resp = models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    u = resp.usage_metadata
    assert (u.prompt_token_count, u.candidates_token_count, u.thoughts_token_count) == (100, 20, 0)


def test_response_cannot_trip_the_content_block_check():
    """gemini_worker.raise_if_blocked reads prompt_feedback + candidates via
    getattr; Ollama has no content policy, so both must be inert."""
    models = _models()
    resp = models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert not hasattr(resp, "prompt_feedback")
    assert resp.candidates == []


# --- request shaping -------------------------------------------------------

def test_num_ctx_is_always_sent(monkeypatch):
    """Without it Ollama uses the server default and may silently truncate."""
    sent = []
    models = _models(capture=sent)
    models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert sent[0]["options"]["num_ctx"] == 8192

    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    sent.clear()
    models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert sent[0]["options"]["num_ctx"] == 16384


def test_temperature_splits_by_stage():
    sent = []
    models = _models(capture=sent)
    models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert sent[0]["options"]["temperature"] == 0.2      # scoring: precise
    sent.clear()
    models.generate_content(model="m", contents="hi", config=_Cfg(DetailResponse))
    assert sent[0]["options"]["temperature"] == 0.7      # detail: creative


def test_ollama_prefix_is_stripped_from_the_model_name():
    """make_client returns 'ollama/x' for cost lookup; the wire wants 'x'."""
    sent = []
    models = _models(capture=sent)
    models.generate_content(model="ollama/qwen2.5:7b-instruct", contents="hi",
                            config=_Cfg(ScoreLike))
    assert sent[0]["model"] == "qwen2.5:7b-instruct"


def test_seed_is_forwarded_when_set(monkeypatch):
    monkeypatch.setenv("OLLAMA_SEED", "42")
    sent = []
    models = _models(capture=sent)
    models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert sent[0]["options"]["seed"] == 42


def test_non_text_contents_is_refused():
    models = _models()
    with pytest.raises(RuntimeError, match="only serves text"):
        models.generate_content(model="m", contents=[{"image": "x"}], config=_Cfg(ScoreLike))


# --- error mapping ---------------------------------------------------------

def _is_transient(exc):
    """Whether main._run_gemini_stage retries it: the loop's own decision
    function, not a copy of its token list that could drift."""
    from clip_selection import classify_gemini_error
    return classify_gemini_error(exc) is not None


def test_connect_error_is_retryable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)
    _mount(handler)
    models = llm_provider.OllamaClient("http://fake:11434").models
    with pytest.raises(RuntimeError) as e:
        models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert _is_transient(e.value)


def test_server_error_is_retryable():
    _mount(lambda r: httpx.Response(500, text="boom"))
    models = llm_provider.OllamaClient("http://fake:11434").models
    with pytest.raises(RuntimeError) as e:
        models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert _is_transient(e.value)


def test_empty_body_is_retryable():
    """Reuses the token main.py already recognises from the 22-jul-2026 incident."""
    _mount(lambda r: _ok("   "))
    models = llm_provider.OllamaClient("http://fake:11434").models
    with pytest.raises(ValueError) as e:
        models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert _is_transient(e.value)


def test_missing_model_is_NOT_retried():
    """A 404 is deterministic — failing in 2s beats 35s of pointless backoff."""
    _mount(lambda r: httpx.Response(404, text='{"error":"model not found"}'))
    models = llm_provider.OllamaClient("http://fake:11434").models
    with pytest.raises(RuntimeError) as e:
        models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert not _is_transient(e.value)
    assert "ollama pull" in str(e.value)


def test_truncation_is_NOT_retried_and_names_the_remedy():
    _mount(lambda r: _ok(json.dumps({"windows": []}), done_reason="length"))
    models = llm_provider.OllamaClient("http://fake:11434").models
    with pytest.raises(RuntimeError) as e:
        models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert not _is_transient(e.value)
    assert "OLLAMA_NUM_CTX" in str(e.value)


def test_context_overflow_warns(capsys):
    _mount(lambda r: _ok(json.dumps({"windows": [{"id": "w1", "score": 1}]}),
                         prompt_eval_count=7800))
    models = llm_provider.OllamaClient("http://fake:11434").models
    models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
    assert "context tokens" in capsys.readouterr().out


def test_unreachable_daemon_waits_like_an_overload_but_a_read_timeout_does_not():
    """Its message starts with 503, so a daemon restart or cold load gets the
    long overload wait; a read timeout keeps the short 3-attempt budget."""
    from clip_selection import classify_gemini_error

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    def stall(request):
        raise httpx.ReadTimeout("slow", request=request)

    for handler, kind in ((refuse, "overload"), (stall, "transient")):
        _mount(handler)
        models = llm_provider.OllamaClient("http://fake:11434").models
        with pytest.raises(RuntimeError) as e:
            models.generate_content(model="m", contents="hi", config=_Cfg(ScoreLike))
        assert classify_gemini_error(e.value) == kind


# --- provider resolution ---------------------------------------------------

def _fake_probe(monkeypatch, names, url="http://fake:11434"):
    monkeypatch.setattr(llm_provider, "_probe",
                        lambda base, timeout=1.5: names if base == url else None)
    monkeypatch.setenv("OLLAMA_BASE_URL", url)


def test_auto_prefers_local_when_model_present(monkeypatch):
    _fake_probe(monkeypatch, ["qwen2.5:7b-instruct"])
    monkeypatch.setenv("GEMINI_API_KEY", "should-be-ignored")
    assert llm_provider.resolve_provider() == "ollama"


def test_auto_falls_back_to_gemini_when_ollama_down(monkeypatch):
    monkeypatch.setattr(llm_provider, "_probe", lambda base, timeout=1.5: None)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert llm_provider.resolve_provider() == "gemini"


def test_auto_falls_back_when_model_not_pulled(monkeypatch):
    """A reachable daemon without the model would 404 on every generation."""
    _fake_probe(monkeypatch, ["some-other-model"])
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert llm_provider.resolve_provider() == "gemini"
    assert llm_provider.local_available() is False


def test_auto_with_nothing_available_names_both_remedies(monkeypatch):
    monkeypatch.setattr(llm_provider, "_probe", lambda base, timeout=1.5: None)
    with pytest.raises(RuntimeError) as e:
        llm_provider.resolve_provider()
    msg = str(e.value)
    assert "ollama pull" in msg and "GEMINI_API_KEY" in msg


def test_explicit_ollama_never_reads_the_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_provider, "_probe", lambda base, timeout=1.5: None)
    assert llm_provider.resolve_provider() == "ollama"


def test_explicit_gemini_without_a_key_fails_clearly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm_provider.resolve_provider()


def test_make_client_prefixes_the_model_for_cost_lookup(monkeypatch):
    _fake_probe(monkeypatch, ["qwen2.5:7b-instruct"])
    client, model_name = llm_provider.make_client()
    assert model_name == "ollama/qwen2.5:7b-instruct"
    assert isinstance(client, llm_provider.OllamaClient)
    # duck-typing contract with main._run_gemini_stage
    assert hasattr(client, "models") and hasattr(client.models, "generate_content")


def test_local_model_reports_zero_cost():
    """The 'ollama/' prefix must resolve to free in the pricing table."""
    from clip_selection import lookup_model_prices
    assert lookup_model_prices("ollama/qwen2.5:7b-instruct") == (0.0, 0.0)
