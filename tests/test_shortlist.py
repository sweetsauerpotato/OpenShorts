"""Pass-1 shortlist: how the scoring answers decide which windows reach the detail pass.

Measured 14-sep-2026 on a 55-min documentary (51 windows): the scoring prompt
asked for "up to 3" windows per batch of 8, so only 19-20 windows ever got a
score and the other 31-32 could never be picked. The 10 shortlisted scores
were 9-10 windows in ties, broken by whatever order the answers came back in.
The prompt now scores every window on an anchored scale; this pins how those
answers become a shortlist.
"""
import ast
import os
import re
import types

import pytest

import clip_selection as cs


def _windows(n):
    return [{"id": f"window_{i:03d}", "start": (i - 1) * 60.0, "end": (i - 1) * 60.0 + 90.0,
             "text": f"text {i}"} for i in range(1, n + 1)]


def _ids(windows):
    return [w["id"] for w in windows]


# --- best_window_scores ---------------------------------------------------------

class TestBestWindowScores:
    def test_repeated_id_keeps_the_highest(self):
        scored = [{"id": "window_001", "score": 40}, {"id": "window_001", "score": 85},
                  {"id": "window_001", "score": 60}]
        assert cs.best_window_scores(scored, _windows(2)) == {"window_001": 85}

    def test_ids_that_are_not_windows_are_ignored(self):
        scored = [{"id": "window_999", "score": 100}, {"id": "window_002", "score": 50}]
        assert cs.best_window_scores(scored, _windows(3)) == {"window_002": 50}

    @pytest.mark.parametrize("bad", [None, "high", float("nan"), [90]])
    def test_entries_without_a_numeric_score_are_ignored(self, bad):
        scored = [{"id": "window_001", "score": bad}, {"id": "window_002"},
                  {"id": "window_003", "score": 70}]
        assert cs.best_window_scores(scored, _windows(3)) == {"window_003": 70}

    def test_junk_entries_are_ignored(self):
        assert cs.best_window_scores([None, "window_001", 7], _windows(2)) == {}
        assert cs.best_window_scores(None, _windows(2)) == {}


# --- build_shortlist ------------------------------------------------------------

class TestBuildShortlist:
    def test_best_first_and_capped_at_target(self):
        scores = {"window_001": 10, "window_002": 95, "window_003": 60, "window_004": 80}
        assert _ids(cs.build_shortlist(scores, _windows(4), 3)) == [
            "window_002", "window_004", "window_003"]

    def test_ties_go_to_the_earlier_window_whatever_order_scores_arrive_in(self):
        windows = _windows(5)
        forward = {"window_002": 90, "window_004": 90, "window_005": 90, "window_001": 50}
        backward = dict(reversed(list(forward.items())))
        for scores in (forward, backward):
            assert _ids(cs.build_shortlist(scores, windows, 2)) == ["window_002", "window_004"]

    def test_fewer_scored_than_target_returns_only_the_scored(self):
        assert _ids(cs.build_shortlist({"window_003": 70}, _windows(6), 4)) == ["window_003"]

    def test_nothing_usable_falls_back_to_the_first_windows(self):
        windows = _windows(6)
        assert _ids(cs.build_shortlist({}, windows, 3)) == _ids(windows[:3])
        assert _ids(cs.build_shortlist({"window_999": 99}, windows, 3)) == _ids(windows[:3])

    def test_target_above_the_window_count_returns_them_all(self):
        scores = {w["id"]: i for i, w in enumerate(_windows(3))}
        assert _ids(cs.build_shortlist(scores, _windows(3), 10)) == [
            "window_003", "window_002", "window_001"]

    def test_returns_the_window_objects_themselves(self):
        windows = _windows(2)
        assert cs.build_shortlist({"window_002": 1}, windows, 1)[0] is windows[1]


# --- the scoring prompt ---------------------------------------------------------

def _score_template():
    """Read via ast, so this runs without the ML stack."""
    mod = ast.parse(open(os.path.join(os.path.dirname(__file__), "..",
                                      "gemini_worker.py"), encoding="utf-8").read())
    return next(node.value.value for node in mod.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "SCORE_PROMPT_TEMPLATE")


def test_score_prompt_asks_for_every_window():
    template = _score_template()
    assert "up to 3" not in template  # the cap that left 31-32 of 51 windows unscored
    assert "Score EVERY window" in template


def test_score_prompt_anchors_every_band_of_the_scale():
    template = _score_template()
    for band in ("90-100:", "70-89:", "40-69:", "0-39:"):
        assert band in template


# --- the real selection function, with the AI faked -------------------------------

# 13 windows: 7 tops, 002/004 and 006/010 tie, 003/008/013 tie at the cut.
SCORES = {"window_001": 40, "window_002": 90, "window_003": 70, "window_004": 90,
          "window_005": 20, "window_006": 85, "window_007": 95, "window_008": 70,
          "window_009": 60, "window_010": 85, "window_011": 10, "window_012": 75,
          "window_013": 70}
EXPECTED_SHORTLIST = ["window_007", "window_002", "window_004", "window_006", "window_010",
                      "window_012", "window_003", "window_008", "window_013", "window_009"]


class _ScoringClient:
    """Scores from SCORES, answered in REVERSE order with a lower duplicate and a
    made-up id mixed in, so the shortlist cannot lean on the answer order."""

    def __init__(self):
        self.models = self
        self.detail_ids = []

    def generate_content(self, model=None, contents=None, config=None):
        schema = config.response_schema
        ids = re.findall(r'"id": "(window_\d+)"', contents)
        if schema.__name__ == "ScoreResponse":
            rows = [{"id": i, "start": 0.0, "end": 1.0, "score": SCORES[i], "reason": "r"}
                    for i in reversed(ids)]
            rows += [{"id": ids[0], "start": 0.0, "end": 1.0, "score": 0, "reason": "dup"},
                     {"id": "window_999", "start": 0.0, "end": 1.0, "score": 100, "reason": "x"}]
            payload = {"windows": rows}
        else:
            self.detail_ids.append(ids)
            payload = {"shorts": [{"start": 10.0, "end": 40.0, "source_window_id": ids[0],
                                   "predicted_score": 80, "video_description_for_tiktok": "t",
                                   "video_description_for_instagram": "i",
                                   "video_title_for_youtube_short": "y", "viral_hook_text": "h"}]}
        return types.SimpleNamespace(parsed=schema.model_validate(payload), text="{}",
                                     candidates=[], usage_metadata=None)


def _transcript(n_segments=40, seconds=25.0):
    segments = []
    for i in range(n_segments):
        start = i * seconds
        segments.append({"start": start, "end": start + seconds, "text": f"part {i}",
                         "words": [{"word": "part", "start": start, "end": start + 1.0}]})
    return {"language": "en", "segments": segments}


def test_detail_pass_gets_the_top_windows_best_first(monkeypatch, capsys):
    main = pytest.importorskip("main")
    for var in ("CLIP_MIN_SECONDS", "CLIP_MAX_SECONDS", "CLIP_TARGET_MIN", "CLIP_TARGET_MAX"):
        monkeypatch.delenv(var, raising=False)
    client = _ScoringClient()
    monkeypatch.setattr(main.llm_provider, "make_client", lambda: (client, "fake-model"))

    windows = cs.build_transcript_windows(_transcript(), 1000.0)
    assert _ids(windows) == sorted(SCORES), "window layout changed; update SCORES"

    main.get_viral_clips(_transcript(), 1000.0)  # target for 1000 s is 10

    assert client.detail_ids == [EXPECTED_SHORTLIST]
    assert "Scored 13/13 window(s); shortlisted 10" in capsys.readouterr().out
