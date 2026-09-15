"""render_clips: the agent's clips rendered from a finished job's video.

/api/process with source_job_id + clips (MCP render_clips) validates everything
before any work, links the source job's video and transcript into a new job,
keeps the look the source job asked for, and hands main.py --clips-file. These
tests own that contract, the MCP surface and list_clips' pieces, main.py's edge
placement (with fakes) and the real command line rendering pieces with a hook
and captions (container).
"""
import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys

import httpx
import pytest

app_module = pytest.importorskip("app")
import mcp_server  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = "a9e7c4d2-0000-4000-8000-00000000000a"


def _w(text, s, e):
    return {"word": " " + text, "start": s, "end": e}


TRANSCRIPT = {"language": "en", "segments": [
    {"start": 1.0, "end": 9.0, "text": " They had elections. They were not real.",
     "words": [_w("They", 1.0, 1.3), _w("had", 1.3, 1.6), _w("elections.", 1.6, 2.4),
               _w("They", 3.0, 3.2), _w("were", 3.2, 3.4), _w("not", 3.4, 3.7), _w("real.", 3.7, 4.2)]},
    {"start": 10.0, "end": 30.0, "text": " You get one choice.",
     "words": [_w("You", 10.0, 10.3), _w("get", 10.3, 10.6), _w("one", 10.6, 10.9),
               _w("choice.", 10.9, 11.6)]},
]}
CLIPS = [{"segments": [{"start": 10.0, "end": 11.8}, {"start": 0.8, "end": 4.4}],
          "hook": "One choice is no choice", "title": "Fake elections", "score": 88,
          "reason": "Punchline first"},
         {"segments": [{"start": 0.8, "end": 11.8}], "hook": "Elections without a choice"}]


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                             base_url="http://t", headers={"X-Gemini-Key": "test-key"})


def _post(body):
    async def _do():
        async with _client() as c:
            return await c.post("/api/process", json=body)
    return asyncio.run(_do())


@pytest.fixture()
def source(tmp_path, monkeypatch):
    out = tmp_path / "output"
    job_dir = out / SOURCE
    job_dir.mkdir(parents=True)
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads"))
    (job_dir / "I_dont_get_democracy.mp4").write_bytes(b"the source video")
    (job_dir / "I_dont_get_democracy_metadata.json").write_text(json.dumps({
        "shorts": [], "awaiting_clips": True, "transcript": TRANSCRIPT,
        "source_video": "I_dont_get_democracy.mp4", "duration": 600.0}))
    (job_dir / app_module.AGENT_JOB_FILE).write_text(json.dumps({"selection": "agent", "render": {
        "output_format": "square", "layouts": ["split"], "auto_hook": True,
        "auto_hook_style": "yellow", "captions": False}}))
    monkeypatch.setitem(app_module.jobs, SOURCE, {"status": "completed", "logs": [], "user_id": None,
                                                  "result": {"clips": [], "awaiting_clips": True}})
    return job_dir


def _render(**extra):
    return _post({"source_job_id": SOURCE, "clips": CLIPS, "acknowledged": True, **extra})


def _job(resp):
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    return job_id, app_module.jobs[job_id]


class TestSubmit:
    def test_a_render_job_gets_the_source_transcript_and_clips(self, source):
        job_id, job = _job(_render())
        cmd = job["cmd"]
        video = cmd[cmd.index("-i") + 1]
        assert os.path.dirname(video) == os.path.join(app_module.OUTPUT_DIR, job_id)
        assert os.path.basename(video) == "I_dont_get_democracy.mp4"   # the source job's title
        assert open(video, "rb").read() == b"the source video"
        with open(cmd[cmd.index("--transcript") + 1], encoding="utf-8") as fh:
            assert json.load(fh) == TRANSCRIPT
        with open(cmd[cmd.index("--clips-file") + 1], encoding="utf-8") as fh:
            sent = json.load(fh)
        assert [c["segments"] for c in sent] == [CLIPS[0]["segments"], CLIPS[1]["segments"]]
        assert sent[0]["viral_hook_text"] == "One choice is no choice" and sent[0]["predicted_score"] == 88
        assert "--transcribe-only" not in cmd and job["attestation"]["source"] == "agent_render"

    def test_the_source_jobs_look_is_kept(self, source):
        _, job = _job(_render())
        assert job["cmd"][job["cmd"].index("--format") + 1] == "square"
        env = job["env"]
        assert env["AUTO_HOOK"] == "1" and env["AUTO_HOOK_STYLE"] == "yellow"
        assert env["AUTO_CAPTIONS"] == "0" and env.get("SPLIT_LAYOUT") == "1"

    def test_the_request_can_change_the_look(self, source):
        _, job = _job(_render(captions=True, auto_hook=False, output_format="vertical"))
        assert "AUTO_CAPTIONS" not in job["env"] and "AUTO_HOOK" not in job["env"]
        assert job["cmd"][job["cmd"].index("--format") + 1] == "vertical"

    def test_a_source_without_agent_settings_gets_hook_and_captions(self, source):
        os.remove(source / app_module.AGENT_JOB_FILE)
        _, job = _job(_render())
        assert job["env"]["AUTO_HOOK"] == "1" and "AUTO_CAPTIONS" not in job["env"]

    def test_no_ai_provider_needed(self, source, monkeypatch):
        async def _none(request):
            return None

        async def _no_local():
            return False
        monkeypatch.setattr(app_module, "resolve_gemini", _none)
        monkeypatch.setattr(app_module, "llm_available_without_key", _no_local)
        _job(_render())


class TestRejected:
    @pytest.mark.parametrize("body,status,detail", [
        ({"clips": CLIPS}, 400, "needs both source_job_id and clips"),
        ({"source_job_id": SOURCE}, 400, "needs both source_job_id and clips"),
        ({"source_job_id": SOURCE, "clips": CLIPS, "url": "https://youtu.be/x"}, 400, "don't also send"),
        ({"source_job_id": SOURCE, "clips": CLIPS, "selection": "agent", "target_clips": 3},
         400, "selection=agent, target_clips can't be used"),
        ({"source_job_id": SOURCE, "clips": CLIPS, "transcript": "0:00 hi"}, 400, "transcript can't"),
        ({"source_job_id": SOURCE, "clips": [{"segments": [{"start": 0, "end": 2}]}]}, 400,
         "clips: clip 1 is 2.0 s long"),
        ({"source_job_id": "no-such-job", "clips": CLIPS}, 404, "Source job not found"),
    ])
    def test_before_any_work(self, source, body, status, detail):
        jobs_before = set(app_module.jobs)
        resp = _post({"acknowledged": True, **body})
        assert resp.status_code == status and detail in resp.json()["detail"], resp.text
        assert set(app_module.jobs) == jobs_before

    def test_the_source_video_is_gone(self, source):
        os.remove(source / "I_dont_get_democracy.mp4")
        resp = _render()
        assert resp.status_code == 409 and "no longer on the server" in resp.json()["detail"]

    def test_the_source_has_no_transcript_yet(self, source, monkeypatch):
        os.remove(source / "I_dont_get_democracy_metadata.json")
        monkeypatch.setitem(app_module.jobs, SOURCE, {"status": "processing", "logs": []})
        resp = _render()
        assert resp.status_code == 409 and "processing" in resp.json()["detail"]

    def test_rights_still_confirmed(self, source):
        resp = _post({"source_job_id": SOURCE, "clips": CLIPS})
        assert resp.status_code == 400 and "rights" in resp.json()["detail"]


class TestMcp:
    def test_render_clips_forwards_and_says_what_next(self):
        tool = next(t for t in mcp_server.TOOLS if t["name"] == "render_clips")
        assert tool["inputSchema"]["required"] == ["job_id", "clips"]

        class _Client:
            body = None

            async def post(self, path, json=None):
                _Client.body = (path, json)
                return httpx.Response(200, json={"job_id": "r1", "status": "queued"})

        data, is_error = asyncio.run(mcp_server._tool_render_clips(_Client(), {
            "job_id": SOURCE, "clips": CLIPS}))
        assert not is_error and data["clips"] == 2 and "list_clips" in data["hint"]
        assert _Client.body == ("/api/process", {"source_job_id": SOURCE, "clips": CLIPS,
                                                  "acknowledged": True})

    def test_through_the_transport(self, source):
        async def _do():
            async with _client() as c:
                return await c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                  "params": {"name": "render_clips", "arguments": {
                                                      "job_id": SOURCE, "clips": [{"segments": [{"start": 0, "end": 1}]}]}}})
        result = asyncio.run(_do()).json()["result"]
        assert result["isError"] is True and "at least 5 s" in result["structuredContent"]["error"]

    def test_list_clips_shows_the_pieces(self):
        clips = mcp_server._clip_summaries("j", {"clips": [
            {"start": 0.8, "end": 11.8, "recipe": {"segments": CLIPS[0]["segments"]},
             "auto_hook": {"text": "One choice is no choice"}, "predicted_score": 88,
             "reason": "Punchline first", "video_url": "/videos/j/c1.mp4"},
            {"start": 5.0, "end": 25.0, "viral_hook_text": "Hook", "video_url": "/videos/j/c2.mp4"}]})
        assert clips[0]["segments"] == CLIPS[0]["segments"]
        assert clips[0]["duration_seconds"] == 5.4            # the pieces, not the covering stretch
        assert (clips[0]["hook"], clips[0]["score"], clips[0]["reason"]) == (
            "One choice is no choice", 88, "Punchline first")
        assert clips[1]["segments"] == [{"start": 5.0, "end": 25.0}] and clips[1]["duration_seconds"] == 20.0


# --- main.py (container: needs cv2 and the ML stack) -----------------------------

class TestMainEdges:
    @pytest.fixture()
    def main(self):
        return pytest.importorskip("main")

    def test_pieces_on_whisper_words_move_into_the_gaps(self, main):
        clips = [{"segments": [{"start": 10.05, "end": 11.5}, {"start": 1.1, "end": 4.0}],
                  "start": 1.1, "end": 11.5}]
        main.cut_agent_clips_on_words(TRANSCRIPT, clips)
        # "You" starts at 10.0 (0.05 in: keep it) and "choice." ends at 11.6 (edge 0.1 before its end)
        # "They" starts at 1.0 and "real." ends at 4.2; each edge takes its lead/tail of the silence
        assert clips[0]["segments"] == [{"start": 9.5, "end": 12.0}, {"start": 0.5, "end": 4.6}]
        assert (clips[0]["start"], clips[0]["end"]) == (0.5, 12.0)

    def test_pasted_pieces_are_carried_onto_the_exact_words(self, main, monkeypatch):
        from transcript_import import parse_transcript
        pasted = parse_transcript("0:00\nThey had elections. They were not real.\n"
                                  "0:10\nYou get one choice. That is the whole story.", language="en")
        exact = [dict(seg) for seg in TRANSCRIPT["segments"]]   # Whisper: the real times
        calls = []

        def fake_range(video, lo, hi, duration, language=None):
            calls.append((lo, hi))
            return exact
        monkeypatch.setattr(main, "transcribe_range", fake_range)
        # chosen on the pasted times: "You get one choice." and "They ... real."
        est = [w for seg in pasted["segments"] for w in seg["words"]]
        choice_end = next(w for w in est if w["word"] == " choice.")["end"]
        real_end = next(w for w in est if w["word"] == " real.")["end"]
        clips = [{"segments": [{"start": 10.0, "end": choice_end}, {"start": 0.0, "end": real_end}],
                  "start": 0.0, "end": choice_end}]
        main.refine_pasted_transcript("v.mp4", pasted, clips, 600.0)
        [first, second] = clips[0]["segments"]
        assert 11.6 <= first["end"] <= 12.05 and first["start"] <= 10.0   # whole "You ... choice."
        assert second["start"] <= 1.0 and 4.2 <= second["end"] <= 4.65     # whole "They ... real."
        assert calls[0][0] == 0.0

    @staticmethod
    def _fake_whisper(main, monkeypatch, real):
        calls = []

        def fake_range(video, lo, hi, duration, language=None):
            calls.append((lo, hi))
            return [{"start": lo, "end": hi, "text": "", "words": [
                w for seg in real for w in seg["words"] if lo <= w["start"] and w["end"] <= hi]}]
        monkeypatch.setattr(main, "transcribe_range", fake_range)
        return calls

    def test_a_long_pasted_line_is_heard_whole(self, main, monkeypatch):
        from transcript_import import parse_transcript
        # One 40 s pasted line whose words are spread over its first 6.6 s; the
        # real "You get one choice." is at 30 s, far past 8 s of padding.
        pasted = parse_transcript("0:00\nThey had elections. They were not real. You get one choice.\n"
                                  "0:40\nThe end of it all here.", language="en")
        real = [{"words": TRANSCRIPT["segments"][0]["words"]},
                {"words": [_w("You", 30.0, 30.3), _w("get", 30.3, 30.6), _w("one", 30.6, 30.9),
                           _w("choice.", 30.9, 31.6)]}]
        calls = self._fake_whisper(main, monkeypatch, real)
        est = [w for seg in pasted["segments"] for w in seg["words"]]
        start = next(w for w in est if w["word"] == " You")["start"]
        end = next(w for w in est if w["word"] == " choice.")["end"]
        clips = [{"segments": [{"start": start, "end": end}], "start": start, "end": end}]
        main.refine_pasted_transcript("v.mp4", pasted, clips, 600.0)
        assert calls == [(0.0, 48.0)]                           # the whole line, plus 8 s
        assert clips[0]["segments"] == [{"start": 29.5, "end": 32.0}]

    def test_a_cut_that_moves_past_the_transcribed_stretch_gets_it_transcribed(self, main, monkeypatch):
        from transcript_import import parse_transcript
        monkeypatch.setenv("CLIP_MIN_SECONDS", "15")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "60")
        pasted = parse_transcript("\n".join([
            "0:10", "We start here with a point.", "0:15", "It is a good point.",
            "0:20", "Then more words come.", "0:25", "And this last sentence runs on and on",
            "0:30", "until it finally ends right here.", "0:40", "Next topic now."]), language="en")
        spoken = [("We", 10.2), ("start", 10.5), ("here", 10.8), ("with", 11.1), ("a", 11.4), ("point.", 11.6),
                  ("It", 15.1), ("is", 15.4), ("a", 15.6), ("good", 15.8), ("point.", 16.1),
                  ("Then", 20.2), ("more", 20.6), ("words", 21.0), ("come.", 21.4),
                  ("And", 25.0), ("this", 25.6), ("last", 26.2), ("sentence", 26.8), ("runs", 27.6),
                  ("on", 28.2), ("and", 28.8), ("on", 29.5),
                  ("until", 30.0), ("it", 31.2), ("finally", 32.4), ("ends", 34.0), ("right", 35.5),
                  ("here.", 37.0), ("Next", 40.0), ("topic", 40.4), ("now.", 40.8)]
        real = [{"words": [_w(t, s, s + 0.4 if t != "here." else 37.5) for t, s in spoken]}]
        calls = self._fake_whisper(main, monkeypatch, real)
        est = [w for seg in pasted["segments"] for w in seg["words"]]
        clips = [{"start": 10.0, "end": [w for w in est if w["word"] == " on"][-1]["end"]}]  # mid-sentence
        refined = main.refine_pasted_transcript("v.mp4", pasted, clips, 600.0)
        # Finishing the sentence took the end to "here." (37.5 s), past the first
        # stretch's last second: that stretch was transcribed too.
        assert len(calls) == 2 and calls[1][1] > calls[0][1]
        assert 37.5 <= clips[0]["end"] <= 38.0
        assert 9.85 <= clips[0]["start"] <= 10.2   # "We" at 10.2, with the pipeline's lead
        assert any(lo <= clips[0]["start"] and clips[0]["end"] + 1 <= hi
                   for lo, hi in refined["exact_ranges"])


def _video(tmp_path, seconds=14):
    video = tmp_path / "democracy.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=640x360:rate=25",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    "-shortest", "-pix_fmt", "yuv420p", str(video)], check=True)
    return video


def test_cli_renders_pieces_with_hook_and_captions(tmp_path):
    """The real command line on a generated video, horizontal (no reframe, so it
    stays fast): a clip in two pieces punchline-first, and a plain one."""
    pytest.importorskip("main")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    from agent_clips import normalize_agent_clips
    video = _video(tmp_path)
    transcript = tmp_path / "t.json"
    transcript.write_text(json.dumps(TRANSCRIPT))
    clips_file = tmp_path / "clips.json"
    clips_file.write_text(json.dumps(normalize_agent_clips(
        [{"segments": [{"start": 10.0, "end": 11.8}, {"start": 0.8, "end": 4.4}], "hook": "One choice"},
         {"segments": [{"start": 0.5, "end": 12.0}], "hook": "Elections"}], source_duration=14)))
    out = tmp_path / "job"
    run = subprocess.run(
        [sys.executable, "-u", "main.py", "-i", str(video), "-o", str(out), "--format", "horizontal",
         "--transcript", str(transcript), "--clips-file", str(clips_file)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
        env=dict(os.environ, AUTO_HOOK="1", GEMINI_API_KEY="dummy-key-for-tests",
                 LLM_PROVIDER="gemini", AUTO_LAYOUT="0", PYTHONIOENCODING="utf-8"))
    log = run.stdout + run.stderr
    assert run.returncode == 0, log[-3000:]
    assert "Provider:" not in log and "Transcribing" not in log   # no model, no Whisper
    ready = dict(line.split(" ", 2)[1:] for line in log.splitlines() if line.startswith("CLIP_READY"))
    assert ready["0"].startswith("subtitled_") and "_hooked_" in ready["0"] and "_recut_" in ready["0"]
    assert ready["1"].startswith("subtitled_") and "_hooked_" in ready["1"] and "_recut_" not in ready["1"]

    meta = json.loads((out / "democracy_metadata.json").read_text())
    first, second = meta["shorts"]
    assert first["recipe"]["segments"] == first["segments"]
    assert first["recipe"]["canonical_range"] == {"start": first["start"], "end": first["end"]}
    assert first["auto_hook"]["text"] == "One choice" and second["auto_hook"]["text"] == "Elections"
    assert "recipe" not in second
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(out / ready["0"])], capture_output=True, text=True)
    total = sum(p["end"] - p["start"] for p in first["segments"])
    assert abs(float(probe.stdout) - total) < 0.35, (probe.stdout, total)
