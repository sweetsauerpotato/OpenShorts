"""A clip the editor rebuilds keeps its hook.

Re-cut (/api/clip/rerender, MCP recut_clip) and re-frame (/api/clip/reframe)
render from hook-less files: the canonical clip or the source. The hook was
never burned again, so every edit silently dropped it while the metadata kept
claiming it. Now recut.perform_recut burns the clip's recorded hook
(``auto_hook``) under the captions, the pipeline's layer order, and the
metadata keeps ``auto_hook`` only when the served file carries one.
"""
import asyncio
import json
import os

import httpx
import pytest

import recut

HOOK = {"text": "Why dictatorships call themselves democracies", "style": "yellow",
        "position": "top", "duration_seconds": 5.0}
CAPTIONS = {"segments": [{"words": [{"word": " a", "start": 0, "end": 1}]}]}


def _seg(start, end):
    return {"start": float(start), "end": float(end)}


def _touching_runner(cmd):
    with open(cmd[-1], "wb") as fh:
        fh.write(b"x")


class _Layers:
    """Fake hooker and captioner that record the order they ran in."""

    def __init__(self, hook_fails=False):
        self.events = []
        self.hook_fails = hook_fails

    def hooker(self, src, hook, dst):
        self.events.append(("hook", os.path.basename(src), hook["text"]))
        with open(dst, "wb") as fh:
            fh.write(b"half")
        if self.hook_fails:
            raise RuntimeError("ffmpeg overlay failed")
        with open(dst, "wb") as fh:
            fh.write(b"hooked")

    def captioner(self, path, transcript, start, end):
        self.events.append(("captions", os.path.basename(path)))
        out = os.path.join(os.path.dirname(path), f"subtitled_1_{os.path.basename(path)}")
        with open(out, "wb") as fh:
            fh.write(b"captioned")
        return out


def _recut(tmp_path, layers, **kw):
    return recut.perform_recut(
        input_path=kw.pop("input_path", "clip.mp4"), segments=[_seg(0, 10)],
        output_dir=str(tmp_path), clean_name="t_clip_1.mp4", runner=_touching_runner,
        captioner=layers.captioner, hooker=layers.hooker, **kw)


class TestPerformRecut:
    def test_the_hook_goes_under_the_captions(self, tmp_path):
        layers = _Layers()
        served, clean = _recut(tmp_path, layers, hook=HOOK, captions_transcript=CAPTIONS)
        hooked = layers.events[1][1]
        assert layers.events == [("hook", clean, HOOK["text"]), ("captions", hooked)]
        assert hooked.startswith("hooked_") and hooked.endswith(f"_{clean}")
        assert served == f"subtitled_1_{hooked}"
        for name in (clean, hooked, served):   # every layer stays for later re-styling
            assert os.path.exists(tmp_path / name)

    def test_without_captions_the_hooked_file_is_served(self, tmp_path):
        served, clean = _recut(tmp_path, _Layers(), hook=HOOK)
        assert served.startswith("hooked_") and served.endswith(f"_{clean}")

    @pytest.mark.parametrize("hook", [None, {}, {"text": "   "}])
    def test_no_hook_to_burn_changes_nothing(self, tmp_path, hook):
        layers = _Layers()
        served, clean = _recut(tmp_path, layers, hook=hook, captions_transcript=CAPTIONS)
        assert layers.events == [("captions", clean)]
        assert served == f"subtitled_1_{clean}"

    def test_a_failing_hook_ships_the_recut_without_it(self, tmp_path):
        layers = _Layers(hook_fails=True)
        served, clean = _recut(tmp_path, layers, hook=HOOK, captions_transcript=CAPTIONS)
        assert served == f"subtitled_1_{clean}"
        assert not [f for f in os.listdir(tmp_path) if f.startswith("hooked_")]

    def test_stacked_stretches_follow_the_hook_layer(self, tmp_path):
        """Captions on a SPLIT stretch sit on the seam only if the file they
        burn onto has the layout sidecar."""
        import layout_ranges
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"canonical")
        layout_ranges.write(str(source), [(2.0, 8.0, "split")])
        served, clean = _recut(tmp_path, _Layers(), hook=HOOK, input_path=str(source))
        assert layout_ranges.read(str(tmp_path / served)) == layout_ranges.read(str(tmp_path / clean))
        assert layout_ranges.read(str(tmp_path / served))[0]["layout"] == "split"


class TestBurnRecordedHook:
    @pytest.fixture()
    def calls(self, monkeypatch):
        hooks = pytest.importorskip("hooks")
        seen = []
        monkeypatch.setattr(hooks, "add_hook_to_video",
                            lambda src, text, dst, **kw: seen.append((src, text, dst, kw)))
        return hooks, seen

    def test_a_manual_hook_comes_back_as_it_was(self, calls):
        hooks, seen = calls
        hooks.burn_recorded_hook("in.mp4", {"text": "Big claim", "style": "red", "position": "bottom",
                                            "duration_seconds": None, "size": "L"}, "out.mp4")
        assert seen == [("in.mp4", "Big claim", "out.mp4",
                         {"position": "bottom", "font_scale": 1.3, "duration": None, "style": "red"})]

    def test_the_pipelines_auto_hook_record(self, calls):
        hooks, seen = calls
        hooks.burn_recorded_hook("in.mp4", {"text": "Hook"}, "out.mp4")  # older records: text only
        assert seen[0][3] == {"position": "top", "font_scale": 1.0, "duration": None, "style": "classic"}


# --- the endpoints ---------------------------------------------------------------

app_module = pytest.importorskip("app")
JOB = "hook-after-recut-job"
TRANSCRIPT = {"language": "en", "segments": [
    {"start": 0.0, "end": 60.0, "text": "hello world",
     "words": [{"word": " hello", "start": 12.0, "end": 12.5},
               {"word": " world", "start": 20.0, "end": 20.4}]}]}


def _post(path, body):
    async def _do():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                                     base_url="http://t") as c:
            return await c.post(path, json=body)
    return asyncio.run(_do())


@pytest.fixture()
def job(tmp_path, monkeypatch):
    out = tmp_path / "output"
    job_dir = out / JOB
    job_dir.mkdir(parents=True)
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads"))
    clip = {"start": 10.0, "end": 40.0, "video_url": f"/videos/{JOB}/t_clip_1.mp4",
            "auto_hook": dict(HOOK)}
    meta_path = job_dir / "t_metadata.json"
    meta_path.write_text(json.dumps({"shorts": [clip], "transcript": TRANSCRIPT,
                                     "source_video": "src.mp4", "output_format": "auto"}))
    (job_dir / "t_clip_1.mp4").write_bytes(b"canonical")
    (job_dir / "src.mp4").write_bytes(b"source")
    monkeypatch.setitem(app_module.jobs, JOB, {
        "status": "completed", "logs": [], "user_id": None, "watermark": False,
        "result": {"clips": [dict(clip)]}})
    return meta_path


@pytest.fixture()
def fake_recut(monkeypatch):
    """The render seam, layering real files the way perform_recut names them."""
    calls = []

    def fake(**kw):
        calls.append(kw)
        out_dir, clean = kw["output_dir"], f"recut_1_{kw['clean_name']}"
        top = clean
        names = [clean]
        if kw.get("hook") and not getattr(fake, "hook_fails", False):
            top = f"hooked_2_{clean}"
            names.append(top)
        served = f"subtitled_3_{top}"
        names.append(served)
        for name in names:
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(b"x")
        return served, clean

    monkeypatch.setattr(recut, "perform_recut", fake)
    fake.calls = calls
    return fake


def _saved_clip(meta_path):
    return json.loads(meta_path.read_text())["shorts"][0]


class TestRerender:
    def test_the_hook_is_burned_back_on_and_stays_recorded(self, job, fake_recut):
        resp = _post("/api/clip/rerender", {"job_id": JOB, "clip_index": 0,
                                            "segments": [{"start": 12, "end": 30}]})
        assert resp.status_code == 200, resp.text
        assert fake_recut.calls[0]["hook"] == HOOK
        assert resp.json()["burned_hook"] == HOOK
        assert resp.json()["new_video_url"].endswith("subtitled_3_hooked_2_recut_1_t_clip_1.mp4")
        assert _saved_clip(job)["auto_hook"] == HOOK
        assert app_module.jobs[JOB]["result"]["clips"][0]["auto_hook"] == HOOK

    def test_reapply_hook_false_drops_it_everywhere(self, job, fake_recut):
        resp = _post("/api/clip/rerender", {"job_id": JOB, "clip_index": 0, "reapply_hook": False,
                                            "segments": [{"start": 12, "end": 30}]})
        assert resp.status_code == 200
        assert fake_recut.calls[0]["hook"] is None
        assert resp.json()["burned_hook"] is None
        assert "auto_hook" not in _saved_clip(job)
        assert "auto_hook" not in app_module.jobs[JOB]["result"]["clips"][0]

    def test_a_failed_burn_is_not_recorded_as_a_hook(self, job, fake_recut):
        fake_recut.hook_fails = True
        resp = _post("/api/clip/rerender", {"job_id": JOB, "clip_index": 0,
                                            "segments": [{"start": 12, "end": 30}]})
        assert resp.status_code == 200
        assert resp.json()["burned_hook"] is None and "auto_hook" not in _saved_clip(job)

    def test_a_clip_without_a_hook_gets_none(self, job, fake_recut):
        meta = json.loads(job.read_text())
        meta["shorts"][0].pop("auto_hook")
        job.write_text(json.dumps(meta))
        _post("/api/clip/rerender", {"job_id": JOB, "clip_index": 0,
                                     "segments": [{"start": 45, "end": 55}]})   # source path
        assert fake_recut.calls[0]["hook"] is None and fake_recut.calls[0]["reframe"] is True


class TestReframe:
    def test_reframing_keeps_the_hook(self, job, fake_recut):
        resp = _post("/api/clip/reframe", {"job_id": JOB, "clip_index": 0,
                                           "crop_overrides": {"0": 0.4}})
        assert resp.status_code == 200, resp.text
        assert fake_recut.calls[0]["hook"] == HOOK
        assert resp.json()["burned_hook"] == HOOK and _saved_clip(job)["auto_hook"] == HOOK

    def test_reapply_hook_false(self, job, fake_recut):
        _post("/api/clip/reframe", {"job_id": JOB, "clip_index": 0,
                                    "crop_overrides": {"0": 0.4}, "reapply_hook": False})
        assert fake_recut.calls[0]["hook"] is None and "auto_hook" not in _saved_clip(job)


class TestManualHookRecordsItsSize:
    def test_size_is_recorded_and_scaled(self, job, monkeypatch):
        seen = {}

        def fake_burn(src, text, dst, **kw):
            seen.update(kw)
            with open(dst, "wb") as fh:
                fh.write(b"hooked")
        monkeypatch.setattr(app_module, "add_hook_to_video", fake_burn)
        resp = _post("/api/hook", {"job_id": JOB, "clip_index": 0, "text": "Big claim",
                                   "size": "L", "style": "red", "position": "top"})
        assert resp.status_code == 200, resp.text
        assert seen["font_scale"] == 1.3
        assert json.loads(job.read_text())["shorts"][0]["auto_hook"]["size"] == "L"


def test_mcp_recut_clip_forwards_reapply_hook():
    import mcp_server
    tool = next(t for t in mcp_server.TOOLS if t["name"] == "recut_clip")
    assert tool["inputSchema"]["properties"]["reapply_hook"]["type"] == "boolean"

    class _Client:
        body = None

        async def post(self, path, json=None):
            _Client.body = json
            return httpx.Response(200, json={"success": True})

    asyncio.run(mcp_server._tool_recut_clip(_Client(), {
        "job_id": JOB, "clip_index": 0, "segments": [{"start": 1, "end": 2}],
        "reapply_hook": False}))
    assert _Client.body["reapply_hook"] is False


def test_the_longest_edited_name_fits_the_filesystem():
    """A captioned hook on a recut of a long non-Latin title: three decorations."""
    main = pytest.importorskip("main")
    stem = main.sanitize_filename("বাংলা" * 40)   # 120-byte cap, 3 bytes a character
    name = f"subtitled_1789473379_hooked_1789473346_recut_1789473300_a1b2c3_{stem}_clip_10.mp4"
    assert len(name.encode("utf-8")) < 255
    assert len(f"hooked_1789473346_recut_1789473300_a1b2c3_{stem}_clip_10.mp4.layout.json"
               .encode("utf-8")) < 255
