"""Retry behaviour of the Gemini scoring/detail stage.

Prod (22-jul-2026) lost 3 jobs to Gemini answering 200 with an empty body.
That raises while parsing, not while calling, so it used to escape the retry
loop and kill the job on the first blip.
"""
import types
import pytest

# main pulls in cv2/torch/mediapipe at import time; the minimal CI env lacks
# them, so skip there. Runs fully in the container/local where deps exist.
main = pytest.importorskip("main")


class _FakeResponse:
    def __init__(self, parsed=None):
        self.parsed = parsed
        self.candidates = []
        self.usage_metadata = None

    @property
    def text(self):
        return "" if self.parsed is None else "{}"


class _FakeModels:
    """Returns an empty body for the first `blips` calls, then a good one."""

    def __init__(self, blips, payload=None):
        self.blips = blips
        self.calls = 0
        self.payload = payload if payload is not None else {"windows": [{"id": "w0", "score": 90}]}

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls <= self.blips:
            return _FakeResponse(parsed=None)
        return _FakeResponse(parsed=self.payload)


def _client(models):
    return types.SimpleNamespace(models=models)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)


def test_recovers_from_a_single_empty_body(monkeypatch):
    models = _FakeModels(blips=1)
    parsed, _cost = main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 2
    assert parsed["windows"][0]["score"] == 90


def test_recovers_from_two_consecutive_blips():
    models = _FakeModels(blips=2)
    parsed, _cost = main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 3
    assert parsed["windows"]


def test_gives_up_after_three_attempts():
    models = _FakeModels(blips=99)
    with pytest.raises(Exception) as exc:
        main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 3
    assert "empty response body" in str(exc.value)


def test_non_transient_errors_are_not_retried():
    class _Boom:
        calls = 0

        def generate_content(self, **kwargs):
            _Boom.calls += 1
            raise ValueError("400 INVALID_ARGUMENT: bad request")

    with pytest.raises(ValueError):
        main._run_gemini_stage(_client(_Boom()), "m", "prompt", object)
    assert _Boom.calls == 1


def test_succeeds_without_retrying_when_the_first_call_is_fine():
    models = _FakeModels(blips=0)
    main._run_gemini_stage(_client(models), "m", "prompt", object)
    assert models.calls == 1


class _BlockedResponse:
    """Mimics the real prod shape: 200, empty candidates, block_reason set."""
    text = ""
    candidates = []
    usage_metadata = None

    class _PF:
        class _Reason:
            name = "PROHIBITED_CONTENT"
        block_reason = _Reason()

    prompt_feedback = _PF()


def test_policy_block_fails_fast_without_retrying():
    # Prod 23-jul-2026: PROHIBITED_CONTENT is deterministic — 3 retries just
    # burned quota and reported a misleading "empty response body".
    class _Models:
        calls = 0

        def generate_content(self, **kwargs):
            _Models.calls += 1
            return _BlockedResponse()

    import gemini_worker
    with pytest.raises(gemini_worker.GeminiBlockedError) as exc:
        main._run_gemini_stage(_client(_Models()), "m", "prompt", object)
    assert _Models.calls == 1
    assert "PROHIBITED_CONTENT" in str(exc.value)


def test_blocked_finish_reason_also_raises():
    import gemini_worker

    class _Cand:
        class _FR:
            name = "SAFETY"
        finish_reason = _FR()
        content = None

    class _Resp:
        prompt_feedback = None
        candidates = [_Cand()]

    with pytest.raises(gemini_worker.GeminiBlockedError):
        gemini_worker.raise_if_blocked(_Resp())


def test_clean_response_is_not_flagged_as_blocked():
    import gemini_worker

    class _Resp:
        prompt_feedback = None
        candidates = []

    gemini_worker.raise_if_blocked(_Resp())  # must not raise


# --- 503 overload, replayed through the REAL google-genai client -------------------
# 14-sep-2026: 8 of the calls in 10 harness runs hit Gemini's 503 "high demand";
# one was still overloaded after the old 15 s of retrying and failed its run.
# These feed the real SDK the body Gemini actually sent, over a fake transport,
# so the error type, its code and the parsed success are the real ones.

import json  # noqa: E402

BODY_503 = {"error": {"code": 503, "status": "UNAVAILABLE", "message":
                      "This model is currently experiencing high demand. Spikes in demand "
                      "are usually temporary. Please try again later."}}
BODY_429 = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded."}}
BODY_OK = {"candidates": [{"index": 0, "finishReason": "STOP", "content": {"role": "model", "parts": [
               {"text": json.dumps({"windows": [{"id": "window_001", "start": 0.0, "end": 90.0,
                                                 "score": 80, "reason": "r"}]})}]}}],
           "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 50}}


def _sdk_client(answers):
    """A real genai.Client whose HTTP answers are ``answers`` in order, the last repeating."""
    import httpx
    from google import genai
    from google.genai import types as genai_types

    requests = []

    def handler(request):
        status, body = answers[min(len(requests), len(answers) - 1)]
        requests.append(request)
        return httpx.Response(status, json=body)

    client = genai.Client(api_key="test-key", http_options=genai_types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
    return client, requests


@pytest.fixture()
def sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(main.time, "sleep", slept.append)
    monkeypatch.setattr(main.random, "uniform", lambda a, b: 1.0)  # no jitter
    monkeypatch.delenv("GEMINI_OVERLOAD_WAIT_SECONDS", raising=False)
    return slept


def _score_stage(client):
    import gemini_worker
    return main._run_gemini_stage(client, "gemini-3.1-flash-lite", "prompt",
                                  gemini_worker.ScoreResponse)


def test_a_503_burst_longer_than_the_old_budget_now_recovers(sleeps, capsys):
    client, requests = _sdk_client([(503, BODY_503)] * 4 + [(200, BODY_OK)])
    parsed, cost = _score_stage(client)
    assert parsed["windows"][0]["score"] == 80 and cost["input_tokens"] == 1000
    assert len(requests) == 5
    assert sleeps == [5, 10, 20, 40]
    assert "Gemini answered after 4 overload retries (75s of waiting)" in capsys.readouterr().out


def test_a_503_that_outlasts_the_budget_fails_with_the_real_reason(sleeps):
    import gemini_worker
    client, requests = _sdk_client([(503, BODY_503)])
    with pytest.raises(gemini_worker.GeminiOverloadedError) as exc:
        _score_stage(client)
    assert sleeps == [5, 10, 20, 40, 60, 45]
    assert len(requests) == 7
    assert "Nothing is wrong with this video" in str(exc.value)
    assert exc.value.__cause__.code == 503


def test_the_wait_budget_comes_from_the_environment(sleeps, monkeypatch):
    import gemini_worker
    monkeypatch.setenv("GEMINI_OVERLOAD_WAIT_SECONDS", "15")
    client, requests = _sdk_client([(503, BODY_503)])
    with pytest.raises(gemini_worker.GeminiOverloadedError):
        _score_stage(client)
    assert sleeps == [5, 10] and len(requests) == 3


def test_a_rate_limit_keeps_three_attempts(sleeps):
    from google.genai import errors
    client, requests = _sdk_client([(429, BODY_429)])
    with pytest.raises(errors.ClientError):
        _score_stage(client)
    assert sleeps == [5, 10] and len(requests) == 3


def test_get_viral_clips_reports_the_overload_instead_of_no_clips(sleeps, monkeypatch):
    import gemini_worker
    monkeypatch.setenv("GEMINI_OVERLOAD_WAIT_SECONDS", "0")
    client, _ = _sdk_client([(503, BODY_503)])
    monkeypatch.setattr(main.llm_provider, "make_client", lambda: (client, "gemini-3.1-flash-lite"))
    transcript = {"language": "en", "segments": [
        {"start": i * 25.0, "end": i * 25.0 + 25.0, "text": f"part {i}",
         "words": [{"word": "part", "start": i * 25.0, "end": i * 25.0 + 1.0}]} for i in range(12)]}
    with pytest.raises(gemini_worker.GeminiOverloadedError):
        main.get_viral_clips(transcript, 300.0)
