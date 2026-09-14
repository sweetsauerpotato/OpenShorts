"""Per-job download quality for URL sources.

The bug this exists for (3-sep-2026): the HD attempt's PO-token script failed
once, the 360p-only fallback took over, and the job "succeeded" with 640x360
clips upscaled 5.35x to 1080x1920. The contract now: the caller picks an
"up to" height, HD failures retry, and above 360p the job stops instead of
silently falling back.
"""
import asyncio

import httpx
import pytest

main = pytest.importorskip("main")
app_module = pytest.importorskip("app")


# --- format selector ---------------------------------------------------------

LEGACY_1080 = ('bestvideo[vcodec^=avc1][height<=1080][ext=mp4]+bestaudio[ext=m4a]/'
               'bestvideo[vcodec^=avc1][height<=1080]+bestaudio/'
               'best[height<=1080][ext=mp4]/best[ext=mp4]/best')
LEGACY_720_CAPPED = ('bestvideo[vcodec^=avc1][height<=720][ext=mp4]+bestaudio[ext=m4a]/'
                     'bestvideo[vcodec^=avc1][height<=720]+bestaudio/'
                     'best[height<=720][ext=mp4]/best[height<=720]/best')


class TestDownloadFormat:
    def test_default_is_byte_identical_to_the_old_selector(self):
        """Callers that never pick a quality must download exactly as before."""
        assert main.download_format() == LEGACY_1080
        assert main.download_format(1080) == LEGACY_1080

    def test_paid_proxy_cap_is_byte_identical_and_wins(self):
        assert main.download_format(1080, capped=True) == LEGACY_720_CAPPED
        # The per-GB bandwidth cap still applies to a 4K pick.
        assert main.download_format(2160, capped=True) == LEGACY_720_CAPPED

    @pytest.mark.parametrize("h", [360, 480, 720, 1080])
    def test_up_to_1080_stays_h264(self, h):
        fmt = main.download_format(h)
        assert f"[height<={h}]" in fmt
        assert "vp09" not in fmt and "av01" not in fmt

    @pytest.mark.parametrize("h", [1440, 2160])
    def test_above_1080_asks_vp9_first_then_h264(self, h):
        """YouTube has no H.264 above 1080p. VP9 must be restricted to >1080 so
        a video whose VP9 tops out at 1080p still gets the H.264 stream."""
        fmt = main.download_format(h)
        assert fmt.startswith(f"bestvideo[vcodec^=vp09][height>1080][height<={h}]")
        assert "bestvideo[vcodec^=avc1]" in fmt  # fallback chain kept
        assert "av01" not in fmt                 # never AV1


# --- attempt plan -------------------------------------------------------------

class TestFallbackGate:
    def test_default_keeps_the_fallback(self):
        assert main.plan_download_attempts(False, [], None, True) == [
            ('HD', False, None), ('fallback', False, None)]

    def test_disallowed_fallback_is_dropped(self):
        assert main.plan_download_attempts(False, [], None, True, allow_fallback=False) == [
            ('HD', False, None)]

    def test_fallback_kept_when_it_is_the_only_strategy(self):
        """No HD strategy configured: an empty plan would download nothing."""
        assert main.plan_download_attempts(False, [], None, False, allow_fallback=False) == [
            ('fallback', False, None)]

    def test_direct_file_urls_unaffected(self):
        assert main.plan_download_attempts(False, [], None, True, youtube=False,
                                           allow_fallback=False) == [('direct', False, None)]


# --- size estimate ----------------------------------------------------------------

class TestEstimate:
    def test_filesize_wins(self):
        assert main.estimate_download_bytes([{"filesize": 1000, "tbr": 9999}], 60) == 1000

    def test_bitrate_times_duration_when_size_missing(self):
        # 1600 kbit/s for 10 s = 16000 kbit = 2,000,000 B
        assert main.estimate_download_bytes([{"tbr": 1600}], 10) == 2_000_000

    def test_video_plus_audio_are_summed(self):
        got = main.estimate_download_bytes([{"filesize": 700}, {"filesize_approx": 300}], 5)
        assert got == 1000

    def test_unknown_is_zero_not_a_crash(self):
        assert main.estimate_download_bytes([{}], None) == 0
        assert main.estimate_download_bytes(None, 10) == 0


# --- the real download loop, with yt-dlp faked --------------------------------------

DENO_FAILURE = ("Command '['deno', 'run', '--allow-env', '--allow-net', "
                "'/opt/bgutil-provider/server/build/generate_once.js']' returned non-zero exit status 1.")


class _FakeYDL:
    """Stands in for yt_dlp.YoutubeDL. Every metadata pass records the format
    selector it was given; ``behavior`` decides what that pass does."""
    calls = []
    behavior = None

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        _FakeYDL.calls.append(self.opts.get("format"))
        return _FakeYDL.behavior(len(_FakeYDL.calls))

    def download(self, urls):
        with open(self.opts["outtmpl"].replace("%(ext)s", "mp4"), "wb") as f:
            f.write(b"\x00")


def _hd_info(n):
    return {"title": "video", "duration": 600,
            "requested_formats": [{"height": 1080, "vcodec": "avc1.640028", "filesize": 10_000},
                                  {"vcodec": "none", "filesize": 1_000}]}


@pytest.fixture()
def fake_ytdlp(monkeypatch):
    _FakeYDL.calls = []
    sleeps = []
    monkeypatch.setattr(main.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(main.time, "sleep", sleeps.append)
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    # Self-host shape: a PO-token script, no proxies, no cookies.
    monkeypatch.setenv("BGUTIL_SCRIPT_PATH", "/fake/generate_once.js")
    for var in ("BGUTIL_BASE_URL", "PROXY_URL", "STATIC_PROXY_URLS", "DIRECT_FIRST", "YOUTUBE_COOKIES"):
        monkeypatch.delenv(var, raising=False)
    return sleeps


YT = "https://www.youtube.com/watch?v=9_SZFIW7tus"
FALLBACK_FMT = "best[ext=mp4]/best"


def _always_fail(n):
    raise RuntimeError(DENO_FAILURE)


def test_hd_failure_retries_then_stops_instead_of_360p(fake_ytdlp, tmp_path, capsys):
    """The 3-sep-2026 job, replayed: the token script fails on every try."""
    _FakeYDL.behavior = _always_fail
    with pytest.raises(RuntimeError):
        main.download_youtube_video(YT, str(tmp_path), max_height=1080, allow_low_fallback=False)
    assert _FakeYDL.calls == [main.download_format(1080)] * 3  # three HD tries...
    assert FALLBACK_FMT not in _FakeYDL.calls                   # ...and never the 360p path
    assert fake_ytdlp[:2] == [3, 6]                             # backoff between tries
    # (the trailing 0.5 s is the error handler flushing logs before it raises)
    out = capsys.readouterr().out
    assert "OpenShorts stopped instead" in out
    assert "up to 1080p" in out


def test_legacy_callers_still_fall_back_after_hd_retries(fake_ytdlp, tmp_path):
    _FakeYDL.behavior = _always_fail
    with pytest.raises(RuntimeError):
        main.download_youtube_video(YT, str(tmp_path))  # defaults: allow_low_fallback=True
    assert _FakeYDL.calls[:3] == [main.download_format(1080)] * 3
    assert _FakeYDL.calls[3:] == [FALLBACK_FMT]


def test_one_transient_hd_failure_recovers_at_full_quality(fake_ytdlp, tmp_path, capsys):
    def flaky(n):
        if n == 1:
            raise RuntimeError(DENO_FAILURE)
        return _hd_info(n)
    _FakeYDL.behavior = flaky
    path, title = main.download_youtube_video(YT, str(tmp_path), max_height=1080,
                                              allow_low_fallback=False)
    assert _FakeYDL.calls == [main.download_format(1080)] * 2
    assert path.endswith("video.mp4") and title == "video"
    out = capsys.readouterr().out
    assert "Download succeeded (HD)" in out
    assert "📐 Selected 1080p avc1 (up to 1080p requested)" in out


def test_not_enough_disk_stops_at_once(fake_ytdlp, tmp_path, monkeypatch, capsys):
    import collections
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(main.shutil, "disk_usage", lambda p: usage(100 * 1024**3, 0, 5 * 1024**3))

    def huge(n):
        return {"title": "video", "duration": 3287,
                "requested_formats": [{"height": 2160, "vcodec": "vp09.00.51.08",
                                       "filesize": 14 * 1024**3}, {"vcodec": "none"}]}
    _FakeYDL.behavior = huge
    with pytest.raises(main.DownloadSpaceError):
        main.download_youtube_video(YT, str(tmp_path), max_height=2160, allow_low_fallback=False)
    assert len(_FakeYDL.calls) == 1   # no retry, no fallback: the disk won't change
    assert "Not enough disk space for 2160p" in capsys.readouterr().out


def test_app_and_main_quality_lists_match():
    assert app_module.DOWNLOAD_QUALITIES == main.DOWNLOAD_QUALITIES
    assert app_module.DEFAULT_DOWNLOAD_QUALITY == main.DEFAULT_DOWNLOAD_QUALITY


# --- /api/process ---------------------------------------------------------------

def _post(json_body=None, files=None, data=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post("/api/process", json=json_body, files=files, data=data,
                                headers={"X-Gemini-Key": "test-key"})
    return asyncio.run(_do())


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads"))


def _stub_probe(monkeypatch, max_height=1080, duration=600):
    async def _probe(url):
        return {"max_height": max_height, "duration": duration}
    monkeypatch.setattr(app_module, "_probe_youtube_quality", _probe)


def _cmd(resp):
    assert resp.status_code == 200, resp.text
    return app_module.jobs[resp.json()["job_id"]]["cmd"]


URL = "https://www.youtube.com/watch?v=ok"


def _quality_arg(cmd):
    return cmd[cmd.index("--quality") + 1]


def test_absent_quality_downloads_at_1080(dirs, monkeypatch):
    _stub_probe(monkeypatch)
    assert _quality_arg(_cmd(_post({"url": URL, "acknowledged": True}))) == "1080"


@pytest.mark.parametrize("sent, expected", [(720, "720"), ("1440", "1440"), ("2160p", "2160")])
def test_quality_is_forwarded(dirs, monkeypatch, sent, expected):
    _stub_probe(monkeypatch)
    assert _quality_arg(_cmd(_post({"url": URL, "acknowledged": True, "quality": sent}))) == expected


@pytest.mark.parametrize("bad", [999, "4k", "hd", 0])
def test_invalid_quality_is_a_400(dirs, monkeypatch, bad):
    _stub_probe(monkeypatch)
    resp = _post({"url": URL, "acknowledged": True, "quality": bad})
    assert resp.status_code == 400
    assert "quality must be one of" in resp.json()["detail"]


def test_gate_respects_a_deliberate_low_pick(dirs, monkeypatch):
    """480p picked on a 480p-max video: a choice, not something to warn about."""
    _stub_probe(monkeypatch, max_height=480)
    resp = _post({"url": URL, "acknowledged": True, "quality": 480})
    assert "job_id" in resp.json()


def test_gate_still_warns_when_the_video_is_below_the_pick(dirs, monkeypatch):
    _stub_probe(monkeypatch, max_height=480)
    resp = _post({"url": URL, "acknowledged": True})  # default 1080
    assert resp.json().get("needs_confirmation") is True


def test_uploads_never_get_a_quality_arg(dirs, monkeypatch):
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 600.0)
    resp = _post(files={"file": ("clip.mp4", b"\x00" * 64, "video/mp4")},
                 data={"acknowledged": "true", "quality": "720"})
    assert "--quality" not in _cmd(resp)
