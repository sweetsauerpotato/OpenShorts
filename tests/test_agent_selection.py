"""selection=agent, part 1: the job only downloads and transcribes.

An agent (Claude over MCP) chooses the clips instead of Gemini: the job stops
after transcription with the transcript saved and no clips, and the agent's
clips are rendered later by a job of their own. These tests own that first
half: the /api/process contract (flag, 400s, no AI provider needed, render
settings saved for the later job), the "awaiting clips" result through
completion, recovery from disk and the webhook, the MCP surface, and main.py's
--transcribe-only path (in the container, where main imports).
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys

import httpx
import pytest

app_module = pytest.importorskip("app")
import mcp_server  # noqa: E402  (after the app import, like the other MCP tests)

URL = "https://www.youtube.com/watch?v=ok"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _post(json_body):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post("/api/process", json=json_body,
                                headers={"X-Gemini-Key": "test-key"})
    return asyncio.run(_do())


def _mcp(tool, arguments):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post("/mcp", json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": tool, "arguments": arguments}})
    return asyncio.run(_do()).json()["result"]


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
    return tmp_path / "output"


def _job(resp):
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    return job_id, app_module.jobs[job_id]


# --- /api/process ------------------------------------------------------------------

class TestProcessEndpoint:
    def test_agent_job_only_transcribes(self, api_dirs):
        _, job = _job(_post({"url": URL, "acknowledged": True, "selection": "agent"}))
        assert "--transcribe-only" in job["cmd"]
        # The source must survive for the render job: URL jobs keep it.
        assert "--keep-original" in job["cmd"]

    @pytest.mark.parametrize("extra", [{}, {"selection": "ai"}, {"selection": " AI "},
                                       {"selection": None}, {"selection": ""}])
    def test_default_is_the_ai_selection(self, api_dirs, extra):
        _, job = _job(_post({"url": URL, "acknowledged": True, **extra}))
        assert "--transcribe-only" not in job["cmd"]

    @pytest.mark.parametrize("value", ["claude", "manual", ["agent"], 1])
    def test_unknown_selection_is_rejected(self, api_dirs, value):
        resp = _post({"url": URL, "acknowledged": True, "selection": value})
        assert resp.status_code == 400
        assert "selection must be" in resp.json()["detail"]

    @pytest.mark.parametrize("field,value", [("target_clips", 5), ("clip_min_seconds", 20),
                                             ("clip_max_seconds", 40)])
    def test_agent_rejects_the_ai_count_and_length_controls(self, api_dirs, field, value):
        resp = _post({"url": URL, "acknowledged": True, "selection": "agent", field: value})
        assert resp.status_code == 400
        assert field in resp.json()["detail"]
        assert "agent decides" in resp.json()["detail"]

    def test_agent_keeps_the_creator_instructions(self, api_dirs):
        _, job = _job(_post({"url": URL, "acknowledged": True, "selection": "agent",
                             "clip_instructions": "Only the money talk."}))
        path = job["cmd"][job["cmd"].index("--instructions-file") + 1]
        with open(path, encoding="utf-8") as fh:
            assert fh.read() == "Only the money talk."

    def test_render_settings_are_saved_for_the_render_job(self, api_dirs):
        job_id, _ = _job(_post({
            "url": URL, "acknowledged": True, "selection": "agent",
            "output_format": "square", "layouts": ["split", " punch_in"],
            "auto_hook": True, "auto_hook_style": "yellow", "captions": False}))
        with open(api_dirs / job_id / app_module.AGENT_JOB_FILE, encoding="utf-8") as fh:
            saved = json.load(fh)
        assert saved == {"selection": "agent", "render": {
            "output_format": "square", "layouts": ["split", "punch_in"],
            "auto_hook": True, "auto_hook_style": "yellow", "captions": False}}

    def test_render_settings_defaults(self, api_dirs):
        job_id, _ = _job(_post({"url": URL, "acknowledged": True, "selection": "agent",
                                "auto_hook_style": "not-a-style"}))
        with open(api_dirs / job_id / app_module.AGENT_JOB_FILE, encoding="utf-8") as fh:
            render = json.load(fh)["render"]
        assert render == {"output_format": "auto", "layouts": [], "auto_hook": False,
                          "auto_hook_style": None, "captions": True}

    def test_ai_job_writes_no_agent_file(self, api_dirs):
        job_id, _ = _job(_post({"url": URL, "acknowledged": True}))
        assert not os.path.exists(api_dirs / job_id / app_module.AGENT_JOB_FILE)

    def test_agent_job_needs_no_ai_provider(self, api_dirs, monkeypatch):
        async def _no_key(request):
            return None

        async def _no_local():
            return False
        monkeypatch.setattr(app_module, "resolve_gemini", _no_key)
        monkeypatch.setattr(app_module, "llm_available_without_key", _no_local)

        ai = _post({"url": URL, "acknowledged": True})
        assert ai.status_code == 400
        assert "No AI provider" in ai.json()["detail"]

        _, job = _job(_post({"url": URL, "acknowledged": True, "selection": "agent"}))
        assert "--transcribe-only" in job["cmd"]

    def test_agent_job_still_requires_attestation(self, api_dirs):
        resp = _post({"url": URL, "selection": "agent"})
        assert resp.status_code == 400
        assert "rights" in resp.json()["detail"]


# --- the job result ----------------------------------------------------------------

class TestAwaitingClipsResult:
    def test_transcribe_only_metadata_marks_the_result(self):
        assert app_module._job_result({"shorts": [], "awaiting_clips": True}, []) == {
            "clips": [], "cost_analysis": None, "awaiting_clips": True}

    def test_a_normal_result_is_unchanged(self):
        clip = {"start": 1.0, "end": 20.0}
        assert app_module._job_result({"cost_analysis": {"total_cost": 0.01}}, [clip]) == {
            "clips": [clip], "cost_analysis": {"total_cost": 0.01}}

    def test_recovered_from_disk_after_a_restart(self, api_dirs):
        job_dir = api_dirs / "agent-recovered-job"
        job_dir.mkdir()
        (job_dir / "Some_video_metadata.json").write_text(json.dumps(
            {"shorts": [], "awaiting_clips": True, "transcript": {"segments": []}}))
        try:
            app_module._recover_jobs_from_disk()
            job = app_module.jobs["agent-recovered-job"]
            assert job["status"] == "completed"
            assert job["result"]["awaiting_clips"] is True
        finally:
            app_module.jobs.pop("agent-recovered-job", None)

    def test_webhook_says_the_job_awaits_clips(self, monkeypatch):
        sent = []

        def _capture(url, body, secret):
            sent.append(json.loads(body))
            return asyncio.sleep(0)
        monkeypatch.setattr(app_module, "_deliver_webhook", _capture)
        app_module.jobs["agent-hook-job"] = {
            "status": "completed", "logs": [], "webhook_url": "https://example.com/hook",
            "result": {"clips": [], "cost_analysis": None, "awaiting_clips": True}}
        try:
            asyncio.run(app_module._notify_job_webhook("agent-hook-job"))
        finally:
            app_module.jobs.pop("agent-hook-job", None)
        assert sent == [{"event": "job.completed", "job_id": "agent-hook-job",
                         "status": "completed", "clips": [], "awaiting_clips": True}]


# --- MCP ---------------------------------------------------------------------------

class TestMcp:
    def test_process_video_advertises_and_forwards_selection(self):
        tool = next(t for t in mcp_server.TOOLS if t["name"] == "process_video")
        assert tool["inputSchema"]["properties"]["selection"]["enum"] == ["ai", "agent"]

        class _Client:
            body = None

            async def post(self, path, json=None):
                _Client.body = json
                return httpx.Response(200, json={"job_id": "j", "status": "queued"})

        data, is_error = asyncio.run(mcp_server._tool_process_video(_Client(), {
            "source_url": URL, "confirm_rights": True, "selection": "agent"}))
        assert not is_error
        assert _Client.body["selection"] == "agent"
        assert "awaiting_clips" in data["hint"] and "get_transcript" in data["hint"]

    def test_status_of_a_transcribed_job_points_to_the_transcript(self):
        app_module.jobs["mcp-agent-job"] = {
            "status": "completed", "logs": ["📝 Transcript ready"],
            "result": {"clips": [], "cost_analysis": None, "awaiting_clips": True}}
        try:
            result = _mcp("get_job_status", {"job_id": "mcp-agent-job"})
            listed = _mcp("list_clips", {"job_id": "mcp-agent-job"})
        finally:
            app_module.jobs.pop("mcp-agent-job", None)
        assert result["isError"] is False
        data = result["structuredContent"]
        assert data["awaiting_clips"] is True
        assert "clips" not in data
        assert "get_transcript" in data["hint"]
        # list_clips must not answer "zero clips" as if the job had failed to find any.
        assert listed["isError"] is True
        assert listed["structuredContent"]["awaiting_clips"] is True


# --- main.py --transcribe-only (container: needs cv2 and the ML stack) --------------

SPEECH = {"text": "", "language": "en", "segments": [
    {"start": 0.0, "end": 3.5, "text": " So here is the one thing nobody tells you about money.",
     "words": [{"word": f" w{i}", "start": 0.3 * i, "end": 0.3 * i + 0.25} for i in range(11)]},
]}


class TestTranscribeOnlyMain:
    @pytest.fixture()
    def main(self):
        return pytest.importorskip("main")

    def test_metadata_keeps_the_transcript_and_no_clips(self, main):
        meta = main.agent_transcript_metadata(SPEECH, "/app/output/j/My video.mp4", "auto", 3.99951)
        assert meta == {"shorts": [], "awaiting_clips": True, "selection": "agent",
                        "transcript": SPEECH, "source_video": "My video.mp4",
                        "output_format": "auto", "duration": 4.0}

    @pytest.mark.parametrize("transcript", [None, {"segments": []},
                                            {"segments": [{"text": "Uh uh"}]}])
    def test_no_speech_fails_and_names_the_way_out(self, main, transcript):
        with pytest.raises(RuntimeError, match="selection=ai"):
            main.require_agent_transcript(transcript, 120.0)

    def test_speech_passes(self, main):
        main.require_agent_transcript(SPEECH, 4.0)

    def test_cli_writes_the_transcript_and_calls_no_model(self, main, tmp_path):
        """The real command line app.py builds, on a generated 4 s video. A dummy
        key with AUTO_LAYOUT=1: a layout pick or a clip-selection call would show
        up as a log line or a failed job, and a render as clip files."""
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")
        video = tmp_path / "talk.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=25",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                        "-shortest", "-pix_fmt", "yuv420p", str(video)], check=True)
        transcript = tmp_path / "source_transcript.json"
        transcript.write_text(json.dumps(SPEECH))
        out = tmp_path / "job"

        env = dict(os.environ, GEMINI_API_KEY="dummy-key-for-tests", LLM_PROVIDER="gemini",
                   AUTO_LAYOUT="1", PYTHONIOENCODING="utf-8")
        run = subprocess.run(
            [sys.executable, "-u", "main.py", "-i", str(video), "-o", str(out),
             "--transcribe-only", "--transcript", str(transcript)],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
        log = run.stdout + run.stderr
        assert run.returncode == 0, log[-2000:]
        assert "Transcript ready" in log
        assert "Choosing a layout" not in log
        assert "Provider:" not in log  # llm_provider.make_client prints this

        assert sorted(os.listdir(out)) == ["talk_metadata.json"]
        meta = json.loads((out / "talk_metadata.json").read_text())
        assert meta["shorts"] == [] and meta["awaiting_clips"] is True
        assert meta["transcript"] == SPEECH
        assert meta["source_video"] == "talk.mp4"
        assert os.path.exists(video)
