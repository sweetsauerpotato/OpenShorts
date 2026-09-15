"""GET /api/transcript/{job_id} and MCP get_transcript: the agent's view of a
transcript. Sentences as "[start-end] text" (the lines Gemini's pass 2 reads),
paged under an MCP client's output limit, with the picking rules and the
creator's instructions on the first page; one stretch word by word for exact
cuts; honest about estimated (pasted) times.
"""
import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")
import mcp_server  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = "5a1e0f5e-0000-4000-8000-000000000001"


def _words(*spec):
    """(text, start, end) -> Whisper word dicts with the leading space."""
    return [{"word": " " + t, "start": s, "end": e} for t, s, e in spec]


TRANSCRIPT = {"language": "en", "segments": [
    {"start": 1.0, "end": 4.0, "text": " Money is not the point.", "words": _words(
        ("Money", 1.0, 1.4), ("is", 1.4, 1.6), ("not", 1.6, 1.9), ("the", 1.9, 2.0), ("point.", 2.0, 2.6))},
    {"start": 5.0, "end": 9.0, "text": " So why does everyone chase it?", "words": _words(
        ("So", 5.0, 5.2), ("why", 5.2, 5.5), ("does", 5.5, 5.7), ("everyone", 5.7, 6.2),
        ("chase", 6.2, 6.6), ("it?", 6.6, 7.0))},
    {"start": 10.0, "end": 14.0, "text": " Here is the answer.", "words": _words(
        ("Here", 10.0, 10.3), ("is", 10.3, 10.5), ("the", 10.5, 10.6), ("answer.", 10.6, 11.2))},
]}


def _get(path, params=None):
    async def _do():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                                     base_url="http://t") as c:
            return await c.get(path, params=params)
    return asyncio.run(_do())


def _mcp(arguments):
    async def _do():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                                     base_url="http://t") as c:
            return await c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                              "params": {"name": "get_transcript",
                                                         "arguments": arguments}})
    return asyncio.run(_do()).json()["result"]


@pytest.fixture()
def job(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / JOB).mkdir(parents=True)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    rules = tmp_path / "rules.md"
    rules.write_text("# Rules\nOnly bangers.\n", encoding="utf-8")
    monkeypatch.setattr(app_module, "CLIP_RULES_PATH", str(rules))
    meta = {"shorts": [], "awaiting_clips": True, "selection": "agent",
            "transcript": TRANSCRIPT, "duration": 15.0}
    (out / JOB / f"{JOB}_Money_talk_metadata.json").write_text(json.dumps(meta))
    (out / JOB / "clip_instructions.txt").write_text("Skip the ad.", encoding="utf-8")
    monkeypatch.setitem(app_module.jobs, JOB, {"status": "completed", "logs": [],
                                               "result": {"clips": [], "awaiting_clips": True}})
    return out / JOB


class TestSentences:
    def test_first_call_has_sentences_rules_and_instructions(self, job):
        resp = _get(f"/api/transcript/{JOB}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["sentences"] == ["[1.0-2.6] Money is not the point.",
                                     "[5.0-7.0] So why does everyone chase it?",
                                     "[10.0-11.2] Here is the answer."]
        assert data["rules"] == "# Rules\nOnly bangers."
        assert data["instructions"] == "Skip the ad."
        assert data["title"] == "Money talk"           # the upload's job-id prefix is gone
        assert (data["duration"], data["language"], data["timing"]) == (15.0, "en", "exact")
        assert data["awaiting_clips"] is True and data["next_start"] is None
        assert data["stats"] == {"sentences": 3, "words": 15}

    def test_pages_join_without_gaps_or_repeats(self, job, monkeypatch):
        monkeypatch.setattr(app_module, "TRANSCRIPT_PAGE_CHARS", 80)  # the first two lines are 75
        first = _get(f"/api/transcript/{JOB}").json()
        assert len(first["sentences"]) == 2 and first["next_start"] == 10.0
        second = _get(f"/api/transcript/{JOB}", {"start": first["next_start"]}).json()
        assert second["sentences"] == ["[10.0-11.2] Here is the answer."]
        assert second["next_start"] is None
        assert "rules" not in second   # already sent with the first page

    def test_a_single_long_sentence_still_comes_back(self, job, monkeypatch):
        monkeypatch.setattr(app_module, "TRANSCRIPT_PAGE_CHARS", 5)
        page = _get(f"/api/transcript/{JOB}").json()
        assert len(page["sentences"]) == 1 and page["next_start"] == 5.0

    def test_start_and_end_select_sentences(self, job):
        data = _get(f"/api/transcript/{JOB}", {"start": 4, "end": 9.9}).json()
        assert data["sentences"] == ["[5.0-7.0] So why does everyone chase it?"]

    def test_pasted_times_are_marked_estimated(self, job):
        meta_path = next(job.glob("*_metadata.json"))
        meta = json.loads(meta_path.read_text())
        meta["transcript"] = dict(TRANSCRIPT, origin="pasted", exact_ranges=[[0.0, 8.0]])
        meta_path.write_text(json.dumps(meta))
        data = _get(f"/api/transcript/{JOB}").json()
        assert data["timing"] == "estimated" and data["exact_ranges"] == [[0.0, 8.0]]

    def test_missing_rules_file_is_not_an_error(self, job, monkeypatch):
        monkeypatch.setattr(app_module, "CLIP_RULES_PATH", str(job / "nope.md"))
        assert _get(f"/api/transcript/{JOB}").json()["rules"] is None


class TestWords:
    def test_a_stretch_word_by_word(self, job):
        data = _get(f"/api/transcript/{JOB}", {"start": 5, "end": 7, "words": "true"}).json()
        assert data["words"] == [[5.0, 5.2, "So"], [5.2, 5.5, "why"], [5.5, 5.7, "does"],
                                 [5.7, 6.2, "everyone"], [6.2, 6.6, "chase"], [6.6, 7.0, "it?"]]
        assert "sentences" not in data and "rules" not in data

    @pytest.mark.parametrize("params,detail", [
        ({"words": "true"}, "needs start and end"),
        ({"start": 0, "words": "true"}, "needs start and end"),
        ({"start": 0, "end": 301, "words": "true"}, "at most 300 s"),
        ({"start": 9, "end": 9}, "end must be after start"),
    ])
    def test_bad_ranges_are_400s(self, job, params, detail):
        resp = _get(f"/api/transcript/{JOB}", params)
        assert resp.status_code == 400 and detail in resp.json()["detail"]


class TestNotReady:
    def test_unknown_job(self, job):
        assert _get("/api/transcript/no-such-job").status_code == 404

    def test_job_still_processing(self, job, monkeypatch):
        for meta in job.glob("*_metadata.json"):
            meta.unlink()
        monkeypatch.setitem(app_module.jobs, JOB, {"status": "processing", "logs": []})
        resp = _get(f"/api/transcript/{JOB}")
        assert resp.status_code == 409 and "processing" in resp.json()["detail"]

    def test_video_without_speech(self, job):
        meta_path = next(job.glob("*_metadata.json"))
        meta_path.write_text(json.dumps({"shorts": [], "transcript": {"segments": []}}))
        resp = _get(f"/api/transcript/{JOB}")
        assert resp.status_code == 409 and "no transcribed speech" in resp.json()["detail"]


class TestMcp:
    def test_listed_with_its_arguments(self):
        tool = next(t for t in mcp_server.TOOLS if t["name"] == "get_transcript")
        assert set(tool["inputSchema"]["properties"]) == {"job_id", "start", "end", "words"}
        assert tool["inputSchema"]["required"] == ["job_id"]

    def test_through_the_transport(self, job):
        result = _mcp({"job_id": JOB})
        assert result["isError"] is False
        assert result["structuredContent"]["sentences"][0] == "[1.0-2.6] Money is not the point."
        words = _mcp({"job_id": JOB, "start": 10, "end": 12, "words": True})["structuredContent"]
        assert [w[2] for w in words["words"]] == ["Here", "is", "the", "answer."]

    def test_errors_are_tool_errors(self, job):
        result = _mcp({"job_id": JOB, "start": 0, "end": 400, "words": True})
        assert result["isError"] is True
        assert result["structuredContent"]["http_status"] == 400


def test_the_rules_file_ships_with_the_image():
    assert os.path.isfile(os.path.join(REPO, "clip_rules.md"))
    with open(os.path.join(REPO, ".dockerignore"), encoding="utf-8") as fh:
        assert "!clip_rules.md" in fh.read().split()
