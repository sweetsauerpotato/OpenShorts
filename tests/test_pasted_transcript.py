"""A pasted transcript: the job chooses clips from it and never transcribes the
whole video, only the chosen clips (8 s either side) for exact cuts and
captions. Covers the /api/process contract (parsed at submit, 400s before any
work, a transcript for another video refused, handed over via --transcript),
the MCP field, transcribe_media's language hint, and main.py's clip-only
transcription and re-cut (with fakes, and the real command line).
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys

import httpx
import pytest

from transcript_import import TRANSCRIPT_MAX_CHARS, parse_transcript

app_module = pytest.importorskip("app")
import mcp_server  # noqa: E402

URL = "https://www.youtube.com/watch?v=ok"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANEL = "\n".join([
    "0:00", "so today I want to talk about democracy",
    "0:04", "and why most people get it wrong.",
    "0:09", "Here is the thing nobody tells you.",
])


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                             base_url="http://testserver", headers={"X-Gemini-Key": "test-key"})


def _post(json_body=None, files=None, data=None):
    async def _do():
        async with _client() as c:
            return await c.post("/api/process", json=json_body, files=files, data=data)
    return asyncio.run(_do())


@pytest.fixture()
def api_dirs(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads"))

    async def _probe(url):
        return {"max_height": 1080, "duration": 600}
    monkeypatch.setattr(app_module, "_probe_youtube_quality", _probe)
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 600.0)
    return tmp_path


def _cmd(resp):
    assert resp.status_code == 200, resp.text
    return app_module.jobs[resp.json()["job_id"]]["cmd"]


def _handed_over(cmd):
    with open(cmd[cmd.index("--transcript") + 1], encoding="utf-8") as fh:
        return json.load(fh)


# --- /api/process ------------------------------------------------------------------

class TestProcessEndpoint:
    def test_url_job_gets_the_parsed_transcript(self, api_dirs):
        cmd = _cmd(_post({"url": URL, "acknowledged": True, "transcript": PANEL}))
        assert cmd.count("--transcript") == 1
        handed = _handed_over(cmd)
        assert handed["origin"] == "pasted" and handed["timing"] == "line"
        assert handed == json.loads(json.dumps(parse_transcript(PANEL)))

    def test_upload_form_field_works_too(self, api_dirs):
        resp = _post(files={"file": ("talk.mp4", b"\x00" * 64, "video/mp4")},
                     data={"acknowledged": "true", "transcript": PANEL})
        assert _handed_over(_cmd(resp))["segments"][0]["text"].startswith("so today")

    @pytest.mark.parametrize("value", [None, "", "  \n\t "])
    def test_blank_means_transcribe_as_usual(self, api_dirs, value):
        assert "--transcript" not in _cmd(_post({"url": URL, "acknowledged": True,
                                                  "transcript": value}))

    def test_unreadable_transcript_is_a_400_that_says_how_to_fix_it(self, api_dirs):
        resp = _post({"url": URL, "acknowledged": True,
                      "transcript": "just the words of the video without any times at all"})
        assert resp.status_code == 400
        assert resp.json()["detail"].startswith("transcript: No timestamps found")

    def test_non_text_is_a_400(self, api_dirs):
        resp = _post({"url": URL, "acknowledged": True, "transcript": ["0:00 hi"]})
        assert resp.status_code == 400 and resp.json()["detail"] == "transcript must be text"

    def test_transcript_of_a_longer_video_is_refused_before_any_work(self, api_dirs, monkeypatch):
        async def _probe(url):
            return {"max_height": 1080, "duration": 5}   # the transcript's last line is at 0:09
        monkeypatch.setattr(app_module, "_probe_youtube_quality", _probe)
        monkeypatch.setattr(app_module, "MIN_SOURCE_SECONDS", 0)
        monkeypatch.setattr(app_module, "QUALITY_GATE_MIN_HEIGHT", 720)  # the probe runs
        jobs_before = set(app_module.jobs)
        resp = _post({"url": URL, "acknowledged": True, "transcript": PANEL})
        assert resp.status_code == 400
        assert "another video" in resp.json()["detail"]
        assert set(app_module.jobs) == jobs_before
        assert os.listdir(api_dirs / "output") == []

    def test_upload_of_a_shorter_video_is_refused_and_removed(self, api_dirs, monkeypatch):
        monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 5.0)
        monkeypatch.setattr(app_module, "MIN_SOURCE_SECONDS", 0)
        resp = _post(files={"file": ("talk.mp4", b"\x00" * 64, "video/mp4")},
                     data={"acknowledged": "true", "transcript": PANEL})
        assert resp.status_code == 400 and "another video" in resp.json()["detail"]
        assert os.listdir(api_dirs / "uploads") == [] and os.listdir(api_dirs / "output") == []

    def test_agent_upload_slot_survives_a_refused_transcript(self, api_dirs, monkeypatch):
        async def _slot():
            async with _client() as c:
                slot = (await c.post("/api/uploads", json={"filename": "talk.mp4"})).json()
                put = await c.put(f"/api/uploads/{slot['upload_id']}", content=b"x" * 100)
                return slot, put
        slot, put = asyncio.run(_slot())
        assert put.status_code == 200
        monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 5.0)
        monkeypatch.setattr(app_module, "MIN_SOURCE_SECONDS", 0)
        resp = _post({"upload_id": slot["upload_id"], "acknowledged": True, "transcript": PANEL})
        assert resp.status_code == 400
        kept = app_module.pending_uploads[slot["upload_id"]]   # resend with the right one
        assert os.path.exists(kept["path"])
        app_module.pending_uploads.pop(slot["upload_id"], None)

    def test_pasted_transcript_wins_over_the_studio_one(self, api_dirs, monkeypatch):
        video = api_dirs / "uploads" / "thumb_s1_source.mp4"
        video.write_bytes(b"fake")
        studio = {"segments": [{"start": 0, "end": 1, "text": "studio words", "words": []}]}
        monkeypatch.setitem(app_module.thumbnail_sessions, "s1", {
            "user_id": None, "video_path": str(video), "transcript_ready": True,
            "transcript": studio})
        cmd = _cmd(_post({"thumbnail_session_id": "s1", "acknowledged": True,
                          "transcript": PANEL}))
        assert cmd.count("--transcript") == 1
        assert _handed_over(cmd)["origin"] == "pasted"


# --- MCP ---------------------------------------------------------------------------

def test_mcp_advertises_and_forwards_the_transcript():
    tool = next(t for t in mcp_server.TOOLS if t["name"] == "process_video")
    prop = tool["inputSchema"]["properties"]["transcript"]
    assert prop["type"] == "string" and prop["maxLength"] == TRANSCRIPT_MAX_CHARS

    class _Client:
        body = None

        async def post(self, path, json=None):
            _Client.body = json
            return httpx.Response(200, json={"job_id": "j", "status": "queued"})

    asyncio.run(mcp_server._tool_process_video(_Client(), {
        "source_url": URL, "confirm_rights": True, "transcript": PANEL}))
    assert _Client.body["transcript"] == PANEL


# --- transcribe_media's language hint -------------------------------------------

class TestLanguageHint:
    @pytest.fixture()
    def tb(self, monkeypatch):
        tb = pytest.importorskip("transcribe_backends")
        monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisper")
        monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
        return tb

    def test_the_known_language_is_passed_on(self, tb, monkeypatch):
        calls = []
        monkeypatch.setattr(tb, "_transcribe_with_whisper",
                            lambda path, language=None: calls.append(language) or {"ok": 1})
        assert tb.transcribe_media("a.wav", language="es") == {"ok": 1}
        assert tb.transcribe_media("a.wav") == {"ok": 1}
        assert calls == ["es", None]

    def test_a_code_whisper_does_not_know_falls_back_to_detection(self, tb, monkeypatch):
        calls = []

        def fake(path, language=None):
            calls.append(language)
            if language:
                raise ValueError(f"'{language}' is not a valid language code")
            return {"ok": 1}
        monkeypatch.setattr(tb, "_transcribe_with_whisper", fake)
        assert tb.transcribe_media("a.wav", language="zz") == {"ok": 1}
        assert calls == ["zz", None]


# --- main.py: exact words only where the clips are (container: needs cv2) --------

def _exact(*words):
    """Whisper-like segment from (word, start, end) triples."""
    return {"start": words[0][1], "end": words[-1][2], "text": " ".join(w for w, _, _ in words),
            "words": [{"word": " " + w, "start": s, "end": e} for w, s, e in words]}


class TestRefine:
    @pytest.fixture()
    def main(self, monkeypatch):
        main = pytest.importorskip("main")
        monkeypatch.setenv("CLIP_MIN_SECONDS", "5")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "60")
        return main

    def _pasted(self):
        return parse_transcript("\n".join([
            "0:30", "this is where the point starts",
            "0:34", "and then it keeps going for a bit",
            "0:38", "until it finally lands right here.",
            "0:42", "Then a brand new sentence starts.",
        ]), language="en")

    def test_clips_are_cut_again_on_the_exact_words(self, main, monkeypatch):
        calls = []
        exact_segments = [
            _exact(("this", 30.4, 30.6), ("is", 30.6, 30.8), ("where", 30.8, 31.1),
                   ("the", 31.1, 31.2), ("point", 31.2, 31.6), ("starts,", 31.6, 32.2)),
            _exact(("and", 34.5, 34.7), ("then", 34.7, 35.0), ("it", 35.0, 35.1),
                   ("keeps", 35.1, 35.5), ("going", 35.5, 35.9), ("for", 35.9, 36.1),
                   ("a", 36.1, 36.2), ("bit", 36.2, 36.6)),
            # the pasted line says 0:38, but the speech runs until 42.6 s
            _exact(("until", 38.9, 39.3), ("it", 39.3, 39.5), ("finally", 39.5, 40.4),
                   ("lands", 40.4, 41.0), ("right", 41.0, 41.6), ("here.", 41.6, 42.6)),
            _exact(("Then", 43.1, 43.4), ("a", 43.4, 43.5), ("brand", 43.5, 43.9),
                   ("new", 43.9, 44.1), ("sentence", 44.1, 44.6), ("starts.", 44.6, 45.2)),
        ]

        def fake_range(input_video, lo, hi, duration, language=None):
            calls.append((lo, hi, language))
            return exact_segments
        monkeypatch.setattr(main, "transcribe_range", fake_range)

        pasted = self._pasted()
        # Selection cut on the estimates: the end at 42.0 is 0.6 s before "here." ends.
        clips = [{"start": 30.0, "end": 42.0}]
        refined = main.refine_pasted_transcript("video.mp4", pasted, clips, 120.0)

        # 8 s either side, once — and out to the end of the pasted line the cut
        # sits in (0:42's line reaches 45.6), since a line-timed word can be
        # anywhere inside its line.
        assert calls == [(22.0, 53.6, "en")]
        assert refined["exact_ranges"] == [[22.0, 53.6]]
        words = [w for seg in refined["segments"] for w in seg["words"]]
        assert ("here.", 41.6) in [(w["word"].strip(), w["start"]) for w in words]
        assert clips[0]["end"] >= 42.6                   # the last word is whole now
        assert clips[0]["end"] < 43.1                    # and "Then" stays out
        assert clips[0]["start"] <= 30.4                 # "this" is whole too

    def test_a_whisper_transcript_is_left_alone(self, main, monkeypatch):
        monkeypatch.setattr(main, "transcribe_range",
                            lambda *a, **k: pytest.fail("must not transcribe"))
        whisper = {"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi", "words": []}]}
        clips = [{"start": 0.0, "end": 20.0}]
        assert main.refine_pasted_transcript("video.mp4", whisper, clips, 60.0) is whisper
        assert clips == [{"start": 0.0, "end": 20.0}]

    def test_when_transcription_fails_the_clips_keep_the_pasted_timing(self, main, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no audio")
        monkeypatch.setattr(main, "transcribe_range", boom)
        pasted = self._pasted()
        clips = [{"start": 30.0, "end": 42.0}]
        assert main.refine_pasted_transcript("video.mp4", pasted, clips, 120.0) is pasted
        assert clips == [{"start": 30.0, "end": 42.0}]

    def test_transcribe_range_puts_words_on_the_video_clock(self, main, monkeypatch):
        import transcribe_backends
        monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: None)
        seen = {}

        def fake_media(path, language=None):
            seen["language"] = language
            return {"segments": [{"text": " cut off word one two", "words": [
                {"word": " cut", "start": 0.05, "end": 0.3},     # half a word at the edge
                {"word": " word", "start": 1.0, "end": 1.4},
                {"word": " one", "start": 5.0, "end": 5.3},
                {"word": " two", "start": 9.7, "end": 9.95},     # cut by the far edge
            ]}]}
        monkeypatch.setattr(transcribe_backends, "transcribe_media", fake_media)

        segments = main.transcribe_range("video.mp4", 100.0, 110.0, 600.0, language="en")
        assert [(w["word"], w["start"]) for w in segments[0]["words"]] == [
            (" word", 101.0), (" one", 105.0)]
        assert seen["language"] == "en"
        # at the very start and end of the video nothing is cut off
        segments = main.transcribe_range("video.mp4", 0.0, 10.0, 10.0)
        assert [w["word"] for w in segments[0]["words"]] == [" cut", " word", " one", " two"]


def _make_video(tmp_path, seconds=4):
    video = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=25",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    "-shortest", "-pix_fmt", "yuv420p", str(video)], check=True)
    return video


@pytest.mark.parametrize("case", ["misfit", "transcribe_only"])
def test_cli_with_a_pasted_transcript(tmp_path, case):
    """The real command line: a transcript for a longer video fails before
    anything else, and a fitting one skips transcription entirely."""
    pytest.importorskip("main")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    video = _make_video(tmp_path)
    text = PANEL if case == "misfit" else "0:00 one two three four five six seven eight nine ten"
    path = tmp_path / "source_transcript.json"
    path.write_text(json.dumps(parse_transcript(text, language="en")))
    out = tmp_path / "job"
    run = subprocess.run(
        [sys.executable, "-u", "main.py", "-i", str(video), "-o", str(out),
         "--transcript", str(path), "--transcribe-only"],
        cwd=REPO, env=dict(os.environ, GEMINI_API_KEY="dummy-key-for-tests",
                           LLM_PROVIDER="gemini", PYTHONIOENCODING="utf-8"),
        capture_output=True, text=True, timeout=300)
    log = run.stdout + run.stderr
    assert "Transcribing" not in log, log[-2000:]
    if case == "misfit":
        assert run.returncode == 1
        assert "runs to 0:09 but the video is 0:04 long" in log
    else:
        assert run.returncode == 0, log[-2000:]
        assert "Using the transcript you provided" in log
        meta = json.loads((out / "talk_metadata.json").read_text())
        assert meta["transcript"]["origin"] == "pasted" and meta["awaiting_clips"] is True
