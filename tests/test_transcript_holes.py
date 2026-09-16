"""transcript_holes: the stretches a whole-video Whisper pass dropped."""
import pytest

import transcript_holes as th


def _w(word, s, e):
    return {"word": word, "start": s, "end": e}


def _seg(start, end, text, words):
    return {"start": start, "end": end, "text": text,
            "words": [_w(t, a, b) for t, a, b in words]}


# "so"[1.0-1.3] (0.4 gap) "today"[1.7-2.2] (8.0 HOLE) "later"[10.2-10.6]
TRANSCRIPT = {
    "language": "en",
    "text": "so today later",
    "segments": [_seg(1.0, 2.2, " so today", [(" so", 1.0, 1.3), (" today", 1.7, 2.2)]),
                 _seg(10.2, 10.6, " later", [(" later", 10.2, 10.6)])],
}


class TestFindHoles:
    def test_finds_the_gap_between_words(self):
        assert th.find_holes(TRANSCRIPT) == [(2.2, 10.2)]

    def test_a_pause_under_the_threshold_is_not_a_hole(self):
        # the 0.4 s gap between "so" and "today" never qualifies
        assert all(a >= 2.2 for a, _ in th.find_holes(TRANSCRIPT))
        assert th.HOLE_SECONDS == 2.0

    def test_a_whole_song_is_skipped_as_too_long(self):
        t = {"segments": [_seg(0, 1, "a", [("a", 0.0, 1.0)]),
                          _seg(900, 901, "b", [("b", 900.0, 901.0)])]}
        assert th.find_holes(t) == []
        assert th.find_holes(t, max_seconds=1000) == [(1.0, 900.0)]

    def test_nothing_before_the_first_or_after_the_last_word(self):
        # only gaps BETWEEN words: a transcript that starts at 60 s says
        # nothing about whether anyone spoke before it.
        t = {"segments": [_seg(60, 61, "a", [("a", 60.0, 61.0)])]}
        assert th.find_holes(t) == []

    def test_an_empty_transcript_has_no_holes(self):
        assert th.find_holes({}) == [] and th.find_holes(None) == []


class TestWorthGating:
    def test_a_single_word_is_noise(self):
        # every 1-word result in the 36-hole measurement was a fragment of a
        # neighbouring sentence Whisper had already transcribed.
        one = _seg(3.0, 3.2, " Oh", [(" Oh", 3.0, 3.2)])
        two = _seg(4.0, 4.9, " Oh God", [(" Oh", 4.0, 4.3), (" God", 4.5, 4.9)])
        assert th.worth_gating([one, two]) == [two]

    def test_empty_text_is_dropped_and_the_rest_is_ordered(self):
        blank = _seg(1.0, 1.9, "   ", [("a", 1.0, 1.4), ("b", 1.5, 1.9)])
        late = _seg(9.0, 9.9, " hey there", [(" hey", 9.0, 9.4), (" there", 9.5, 9.9)])
        early = _seg(5.0, 5.9, " yes now", [(" yes", 5.0, 5.4), (" now", 5.5, 5.9)])
        assert th.worth_gating([blank, late, early]) == [early, late]


class TestParseGate:
    def test_only_dialogue_is_kept(self):
        answer = {"fragments": [{"id": 0, "kind": "dialogue"}, {"id": 1, "kind": "music"},
                                {"id": 2, "kind": "noise"}]}
        assert th.parse_gate(answer, 3) == [True, False, False]

    @pytest.mark.parametrize("answer", [
        None, {}, {"fragments": None}, {"fragments": ["nope"]}, "not a dict",
        {"fragments": [{"id": "x", "kind": "dialogue"}]},      # unusable id
        {"fragments": [{"id": 9, "kind": "dialogue"}]},        # out of range
        {"fragments": [{"id": 0}]},                            # no verdict
    ])
    def test_anything_unusable_keeps_nothing(self, answer):
        # A dropped line only leaves today's transcript; a kept lyric is published.
        assert th.parse_gate(answer, 2) == [False, False]

    def test_a_fragment_the_gate_skipped_is_not_kept(self):
        assert th.parse_gate({"fragments": [{"id": 1, "kind": "dialogue"}]}, 3) == \
            [False, True, False]


class TestMerge:
    def test_recovered_segments_are_spliced_in_order_and_marked(self):
        found = _seg(4.0, 5.0, "in the hole", [("in", 4.0, 4.3), ("hole", 4.6, 5.0)])
        out = th.merge_recovered(TRANSCRIPT, [found])
        assert [s["start"] for s in out["segments"]] == [1.0, 4.0, 10.2]
        assert out["segments"][1]["recovered"] is True
        assert "recovered" not in out["segments"][0]
        assert out["text"] == "so today in the hole later"

    def test_nothing_recovered_leaves_the_transcript_alone(self):
        assert th.merge_recovered(TRANSCRIPT, []) is TRANSCRIPT

    def test_the_original_is_not_mutated(self):
        found = _seg(4.0, 5.0, "x y", [("x", 4.0, 4.3), ("y", 4.6, 5.0)])
        th.merge_recovered(TRANSCRIPT, [found])
        assert len(TRANSCRIPT["segments"]) == 2
