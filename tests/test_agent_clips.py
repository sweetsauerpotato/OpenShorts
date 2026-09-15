"""agent_clips: the clips an agent sends to render_clips, and where their edges go."""
import pytest

import agent_clips as ac
from agent_clips import AgentClipsError, normalize_agent_clips, snap_edge


def _clip(**kw):
    clip = {"segments": [{"start": 40.0, "end": 45.0}, {"start": 10.0, "end": 30.0}],
            "hook": "  They had elections. Fake ones. ", "title": "Why dictators hold elections",
            "tiktok_description": "Elections with one choice #history", "score": 84.6,
            "reason": "Punchline first, then the setup"}
    clip.update(kw)
    return clip


class TestNormalize:
    def test_shapes_the_clip_like_the_pipelines(self):
        [clip] = normalize_agent_clips([_clip()], source_duration=600)
        assert clip == {
            "segments": [{"start": 40.0, "end": 45.0}, {"start": 10.0, "end": 30.0}],  # play order kept
            "start": 10.0, "end": 45.0,                                               # the covering stretch
            "viral_hook_text": "They had elections. Fake ones.",
            "video_title_for_youtube_short": "Why dictators hold elections",
            "video_description_for_tiktok": "Elections with one choice #history",
            "video_description_for_instagram": "",
            "predicted_score": 85, "reason": "Punchline first, then the setup",
            "selected_by": "agent",
        }

    def test_only_segments_are_required(self):
        [clip] = normalize_agent_clips([{"segments": [{"start": 0, "end": 20}]}])
        assert clip["viral_hook_text"] == "" and clip["predicted_score"] is None

    @pytest.mark.parametrize("clips,match", [
        (None, "non-empty list"), ([], "non-empty list"), ("x", "non-empty list"),
        ([{"segments": [{"start": 0, "end": 20}]}] * 16, "at most 15 clips"),
        (["nope"], "clip 1 must be an object"),
        ([_clip(segments=[])], "clip 1: segments must be a non-empty list"),
        ([_clip(segments=[{"start": 1, "end": 1.2}, {"start": 5, "end": 20}])], "segment 1 is shorter"),
        ([_clip(segments=[{"start": 0, "end": 4}])], "is 4.0 s long; at least 5 s"),
        ([_clip(segments=[{"start": 0, "end": 10}, {"start": 400, "end": 410}])], "within 180 s"),
        ([_clip(hook="x" * 151)], "hook is 151 characters; at most 150"),
        ([_clip(title="x" * 101)], "title is 101 characters"),
        ([_clip(instagram_description="x" * 2201)], "instagram_description is 2201"),
        ([_clip(reason=["not", "text"])], "reason must be text"),
        ([_clip(score=101)], "score must be a number from 0 to 100"),
        ([_clip(score=True)], "score must be"),
        ([_clip(score="90")], "score must be"),
    ])
    def test_rejections_name_the_clip_and_the_problem(self, clips, match):
        with pytest.raises(AgentClipsError, match=match):
            normalize_agent_clips(clips, source_duration=600)

    def test_pieces_are_clamped_to_the_video(self):
        [clip] = normalize_agent_clips([_clip(segments=[{"start": 590, "end": 640}])], source_duration=600)
        assert clip["segments"] == [{"start": 590.0, "end": 600.0}]

    def test_the_second_clip_is_named(self):
        with pytest.raises(AgentClipsError, match="clip 2"):
            normalize_agent_clips([_clip(), _clip(score=-1)], source_duration=600)


# Words with real gaps: "so[1.0-1.3] (0.4 gap) today[1.7-2.2] we're[2.2-2.5] (1.0 gap) going[3.5-4.0]"
WORDS = [{"w": "so", "s": 1.0, "e": 1.3}, {"w": "today", "s": 1.7, "e": 2.2},
         {"w": "we're", "s": 2.2, "e": 2.5}, {"w": "going", "s": 3.5, "e": 4.0}]


class TestSnapEdge:
    def test_a_start_in_a_gap_leads_into_the_silence_before_its_word(self):
        # gap 1.3-1.7: lead min(0.5, half of 0.4) = 0.2
        assert snap_edge(1.5, WORDS, "start") == pytest.approx(1.5)
        # gap 2.5-3.5: half of it, 0.5, is the cap
        assert snap_edge(3.0, WORDS, "start") == pytest.approx(3.0)

    def test_an_end_in_a_gap_trails_into_the_silence_after_its_word(self):
        assert snap_edge(1.5, WORDS, "end") == pytest.approx(1.5)      # 1.3 + min(0.4, 0.2)
        assert snap_edge(3.0, WORDS, "end") == pytest.approx(2.9)      # 2.5 + 0.4

    def test_an_edge_inside_a_word_takes_its_nearer_side(self):
        # "today" 1.7-2.2: 1.8 is nearer its start -> a start keeps the word, an end drops it
        assert snap_edge(1.8, WORDS, "start") == pytest.approx(1.5)
        assert snap_edge(1.8, WORDS, "end") == pytest.approx(1.5)
        # 2.1 is nearer its end -> a start drops the word, an end keeps it
        assert snap_edge(2.1, WORDS, "start") == pytest.approx(2.2)    # "we're" touches it: no gap
        assert snap_edge(2.1, WORDS, "end") == pytest.approx(2.2)

    def test_touching_words_cut_exactly_between_them(self):
        assert snap_edge(2.2, WORDS, "start") == pytest.approx(2.2)
        assert snap_edge(2.2, WORDS, "end") == pytest.approx(2.2)

    def test_beyond_the_words(self):
        assert snap_edge(0.2, WORDS, "start") == pytest.approx(0.5)    # before the first word
        assert snap_edge(9.0, WORDS, "end") == pytest.approx(4.4)      # after the last word
        assert snap_edge(9.0, WORDS, "start") == 9.0                   # no word after it
        assert snap_edge(0.2, WORDS, "end") == 0.2                     # no word before it
        assert snap_edge(5.0, [], "end") == 5.0

    def test_never_before_zero(self):
        assert snap_edge(0.0, [{"w": "hi", "s": 0.05, "e": 0.4}], "start") == 0.0

    def test_padding_is_the_modules(self):
        assert (ac.PIECE_LEAD, ac.PIECE_TAIL) == (0.5, 0.4)
