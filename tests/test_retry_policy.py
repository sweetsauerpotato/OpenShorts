"""Which failed Gemini calls are retried, and for how long.

14-sep-2026, 10 harness runs: 8 calls hit Gemini's 503 "high demand"; 6 went
through after one retry, 1 after two, and 1 was still overloaded after the old
budget (3 attempts, 5 s + 10 s of waiting) and failed its run. A 503 now backs
off for up to GEMINI_OVERLOAD_WAIT_SECONDS; every other error keeps the budget
it always had. Pure policy, so this runs in the thin CI env too.
"""
import pytest

import clip_selection as cs

# The body Gemini actually sent on 14-sep-2026.
BODY_503 = {"error": {"code": 503, "status": "UNAVAILABLE", "message":
                      "This model is currently experiencing high demand. Spikes in demand "
                      "are usually temporary. Please try again later."}}


class _CodedError(Exception):
    def __init__(self, code, text):
        super().__init__(text)
        self.code = code


class TestClassify:
    def test_the_real_sdk_503_is_an_overload(self):
        errors = pytest.importorskip("google.genai.errors")
        assert cs.classify_gemini_error(errors.ServerError(503, BODY_503)) == "overload"

    @pytest.mark.parametrize("code, status", [(429, "RESOURCE_EXHAUSTED"), (500, "INTERNAL")])
    def test_rate_limits_and_500s_keep_the_short_budget(self, code, status):
        errors = pytest.importorskip("google.genai.errors")
        error = errors.APIError(code, {"error": {"code": code, "status": status, "message": "x"}})
        assert cs.classify_gemini_error(error) == "transient"

    def test_a_message_starting_with_503_is_an_overload(self):
        error = RuntimeError("503 UNAVAILABLE: cannot reach Ollama at http://x:11434")
        assert cs.classify_gemini_error(error) == "overload"

    @pytest.mark.parametrize("text", [
        "Gemini returned an empty response body.",
        "Failed to parse Gemini JSON response: Expecting value: line 1 column 503 (char 502)",
    ])
    def test_broken_bodies_are_transient_even_when_they_mention_503(self, text):
        assert cs.classify_gemini_error(ValueError(text)) == "transient"

    def test_other_unavailable_errors_keep_the_short_budget(self):
        assert cs.classify_gemini_error(RuntimeError("504 UNAVAILABLE: Ollama read timeout")) == "transient"

    def test_a_coded_error_is_judged_by_its_code_not_its_text(self):
        assert cs.classify_gemini_error(_CodedError(400, "503 tokens over the limit")) != "overload"

    def test_deterministic_errors_are_not_retried(self):
        assert cs.classify_gemini_error(ValueError("400 INVALID_ARGUMENT: bad request")) is None


def _schedule(kind, budget, jitter=1.0):
    waits, waited, failures = [], 0.0, 0
    while True:
        failures += 1
        wait = cs.gemini_retry_delay(kind, failures, waited, budget, jitter=jitter)
        if wait is None:
            return waits
        waits.append(wait)
        waited += wait


class TestDelay:
    def test_an_overload_backs_off_until_the_budget_is_used(self):
        assert _schedule("overload", 180.0) == [5, 10, 20, 40, 60, 45]

    def test_the_old_budget_would_have_stopped_after_15_seconds(self):
        assert _schedule("transient", 180.0) == [5, 10]

    def test_jitter_never_exceeds_the_cap_or_the_budget(self):
        assert cs.gemini_retry_delay("overload", 1, 0.0, 180.0, jitter=1.2) == pytest.approx(6.0)
        assert cs.gemini_retry_delay("overload", 5, 75.0, 180.0, jitter=1.2) == 60.0
        assert cs.gemini_retry_delay("overload", 6, 170.0, 180.0, jitter=1.2) == 10.0
        assert sum(_schedule("overload", 180.0, jitter=1.2)) == pytest.approx(180.0)

    def test_a_zero_budget_gives_up_at_once(self):
        assert cs.gemini_retry_delay("overload", 1, 0.0, 0.0) is None


class TestBudget:
    @pytest.mark.parametrize("raw, expected", [
        (None, 180.0), ("600", 600.0), ("0", 0.0), ("-5", 0.0),
        ("99999", 3600.0), ("soon", 180.0), ("nan", 180.0), ("inf", 180.0),
    ])
    def test_read_from_the_environment(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv("GEMINI_OVERLOAD_WAIT_SECONDS", raising=False)
        else:
            monkeypatch.setenv("GEMINI_OVERLOAD_WAIT_SECONDS", raw)
        assert cs.gemini_overload_budget() == expected
