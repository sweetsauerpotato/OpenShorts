"""transcript_import: a transcript the user pasted, read into the pipeline's shape.

Fixtures follow the real formats: YouTube's "Show transcript" panel as copied
(timestamp lines, a screen-reader duration line, a header), SRT with numbered
cues and markup, WebVTT including YouTube's rolling auto-captions with a time
per word, and OpenShorts' own JSON. Standard library only, so these run in CI.
"""
import json

import pytest

import transcript_import as ti
from transcript_import import TranscriptFormatError, parse_transcript


def _texts(t):
    return [seg["text"] for seg in t["segments"]]


def _all_words(t):
    return [w for seg in t["segments"] for w in seg["words"]]


# --- YouTube's transcript panel ------------------------------------------------

PANEL = """Transcript
0:00
so today I want to talk about democracy
0:04
2 seconds
and why most people get it wrong.
0:09
[Music]
0:12
Here is the thing nobody tells you.
"""


class TestYouTubePanel:
    def test_timestamp_lines_then_words(self):
        t = parse_transcript(PANEL, language="en")
        assert t["origin"] == "pasted" and t["timing"] == "line"
        assert _texts(t) == ["so today I want to talk about democracy",
                             "and why most people get it wrong.",
                             "Here is the thing nobody tells you."]
        assert [seg["start"] for seg in t["segments"]] == [0.0, 4.0, 12.0]

    def test_header_screen_reader_line_and_sound_tag_are_not_words(self):
        words = [w["word"].strip() for w in _all_words(parse_transcript(PANEL, language="en"))]
        assert "Transcript" not in words and "seconds" not in words
        assert not any("Music" in w for w in words)

    def test_a_line_ends_at_the_next_timestamp_or_its_word_budget(self):
        t = parse_transcript(PANEL, language="en")
        first, second, last = t["segments"]
        assert first["end"] == 4.0                       # capped by the next line
        assert second["end"] == pytest.approx(4.0 + 7 * ti.MAX_SECONDS_PER_WORD)  # pause after it
        assert last["end"] == pytest.approx(12.0 + 7 * ti.MAX_SECONDS_PER_WORD)

    def test_words_are_spread_over_the_line_in_order(self):
        seg = parse_transcript(PANEL, language="en")["segments"][0]
        words = seg["words"]
        assert all(w["word"].startswith(" ") for w in words)  # the Whisper word convention
        assert words[0]["start"] == seg["start"]
        assert words[-1]["end"] == pytest.approx(seg["end"], abs=0.01)
        assert all(a["end"] == pytest.approx(b["start"], abs=0.002) for a, b in zip(words, words[1:]))
        # longer words get more time
        by_word = {w["word"].strip(): w["end"] - w["start"] for w in words}
        assert by_word["democracy"] > by_word["so"]

    @pytest.mark.parametrize("text", [
        "0:00 so today I want to talk about\n0:04 why most people get it wrong",
        "[00:00] so today I want to talk about\n[00:04] why most people get it wrong",
        "(0:00) so today I want to talk about\n(0:04) why most people get it wrong",
        "0:00\tso today I want to talk about\n0:04\twhy most people get it wrong",
    ])
    def test_inline_and_bracketed_timestamps(self, text):
        t = parse_transcript(text, language="en")
        assert [seg["start"] for seg in t["segments"]] == [0.0, 4.0]
        assert _texts(t)[1] == "why most people get it wrong"

    def test_hours(self):
        t = parse_transcript("59:58 one two three four\n1:00:03 five six seven eight", language="en")
        assert [seg["start"] for seg in t["segments"]] == [3598.0, 3603.0]

    def test_lines_sharing_a_second_become_one_line(self):
        t = parse_transcript("0:01\nso here we go\n0:01\nand another bit\n0:03\nthe end of it", language="en")
        assert _texts(t) == ["so here we go and another bit", "the end of it"]

    def test_a_sentence_starting_with_a_clock_is_not_a_timestamp(self):
        t = parse_transcript("10:00\nwe met at\n10:03\n9:30 in the morning they said\n10:07\nand left",
                             language="en")
        assert _texts(t) == ["we met at", "9:30 in the morning they said", "and left"]

    def test_no_timestamps_names_the_fix(self):
        with pytest.raises(TranscriptFormatError, match="Show transcript"):
            parse_transcript("so today I want to talk about democracy and why it matters a lot")


# --- SRT -----------------------------------------------------------------------

SRT = """1
00:00:01,000 --> 00:00:04,500
Hello there, this is <i>an SRT</i>
file with two lines.

2
00:00:05,000 --> 00:00:08,000
{\\an8}And a second cue with more words.

3
00:00:05,000 --> 00:00:08,000
{\\an8}And a second cue with more words.
"""


class TestSrt:
    def test_cues_markup_and_repeats(self):
        t = parse_transcript(SRT, language="en")
        assert t["timing"] == "line"
        assert _texts(t) == ["Hello there, this is an SRT file with two lines.",
                             "And a second cue with more words."]
        assert [(seg["start"], seg["end"]) for seg in t["segments"]] == [(1.0, 4.5), (5.0, 8.0)]

    def test_whitespace_only_separator_keeps_the_next_number_out(self):
        text = "1\n00:00:01,000 --> 00:00:03,000\none two three four\n \n2\n00:00:03,000 --> 00:00:05,000\nfive six seven eight"
        assert _texts(parse_transcript(text, language="en")) == ["one two three four",
                                                                 "five six seven eight"]

    def test_cues_out_of_order_are_sorted(self):
        text = ("2\n00:00:05,000 --> 00:00:06,000\nsecond cue words here\n\n"
                "1\n00:00:01,000 --> 00:00:02,000\nfirst cue words here")
        t = parse_transcript(text, language="en")
        assert _texts(t) == ["first cue words here", "second cue words here"]


# --- WebVTT ----------------------------------------------------------------------

YOUTUBE_AUTO_VTT = """WEBVTT
Kind: captions
Language: es

00:00:00.000 --> 00:00:02.310 align:start position:0%

hoy<00:00:00.320><c> vamos</c><00:00:00.640><c> a</c><00:00:00.960><c> hablar</c>

00:00:02.310 --> 00:00:02.320 align:start position:0%
hoy vamos a hablar


00:00:02.320 --> 00:00:04.960 align:start position:0%
hoy vamos a hablar
de<00:00:02.640><c> la</c><00:00:02.960><c> democracia</c><00:00:03.400><c> moderna</c>

00:00:04.960 --> 00:00:04.970 align:start position:0%
de la democracia moderna

"""


class TestVtt:
    def test_youtube_auto_captions_keep_each_words_time(self):
        t = parse_transcript(YOUTUBE_AUTO_VTT)
        assert t["timing"] == "word"
        assert t["language"] == "es"  # from the header, no guessing
        assert _texts(t) == ["hoy vamos a hablar", "de la democracia moderna"]
        assert [(w["word"], w["start"]) for w in _all_words(t)] == [
            (" hoy", 0.0), (" vamos", 0.32), (" a", 0.64), (" hablar", 0.96),
            (" de", 2.32), (" la", 2.64), (" democracia", 2.96), (" moderna", 3.4)]
        # a word never stretches across a pause: 0.6 s at most
        assert all(w["end"] - w["start"] <= ti.MAX_SECONDS_PER_WORD + 1e-6 for w in _all_words(t))

    def test_space_only_lines_stripped_by_an_editor(self):
        # YouTube writes a " " line between a cue's time and its words; editors
        # that strip trailing spaces leave it empty. Both must read the same.
        stripped = "\n".join(line.rstrip() for line in YOUTUBE_AUTO_VTT.split("\n"))
        with_spaces = YOUTUBE_AUTO_VTT.replace(
            "position:0%\n\nhoy<", "position:0%\n \nhoy<")
        assert parse_transcript(stripped)["segments"] == parse_transcript(with_spaces)["segments"]

    def test_plain_vtt_short_timestamps(self):
        text = "WEBVTT\n\n00:01.000 --> 00:03.000\nfirst cue has words\n\n00:03.500 --> 00:06.000\n<v Speaker>second cue has words"
        t = parse_transcript(text, language="en")
        assert t["timing"] == "line"
        assert _texts(t) == ["first cue has words", "second cue has words"]
        assert t["segments"][1]["start"] == 3.5


# --- JSON ------------------------------------------------------------------------

WHISPER = {"text": "", "language": "en", "segments": [
    {"start": 1.0, "end": 3.0, "text": " one two three four",
     "words": [{"word": " one", "start": 1.0, "end": 1.4}, {"word": " two", "start": 1.5, "end": 1.9},
               {"word": " three", "start": 2.0, "end": 2.5}, {"word": " four", "start": 2.6, "end": 3.0}]},
    {"start": 3.2, "end": 5.0, "text": " five six seven eight",
     "words": [{"word": " five", "start": 3.2, "end": 3.6}, {"word": " six", "start": 3.7, "end": 4.0},
               {"word": " seven", "start": 4.1, "end": 4.5}, {"word": " eight", "start": 4.6, "end": 5.0}]},
]}


class TestJson:
    def test_whisper_json_keeps_its_words(self):
        t = parse_transcript(json.dumps(WHISPER))
        assert t["timing"] == "word" and t["language"] == "en"
        assert _all_words(t)[2] == {"word": " three", "start": 2.0, "end": 2.5}

    def test_a_jobs_metadata_file_works(self):
        t = parse_transcript(json.dumps({"shorts": [], "transcript": WHISPER}))
        assert len(t["segments"]) == 2

    def test_segments_without_words_get_estimated_ones(self):
        segs = [{"start": 0, "end": 4, "text": "one two three four"},
                {"start": 4, "end": 8, "text": "five six seven eight"}]
        t = parse_transcript(json.dumps(segs), language="en")
        assert t["timing"] == "line" and len(_all_words(t)) == 8

    @pytest.mark.parametrize("text,match", [
        ('{"segments": [{"end": 2, "text": "x"}]}', "segment 1"),
        ('{"nope": []}', "segments"),
        ('{"segments": [', "not valid JSON"),
    ])
    def test_bad_json_says_what_is_wrong(self, text, match):
        with pytest.raises(TranscriptFormatError, match=match):
            parse_transcript(text)


# --- limits ----------------------------------------------------------------------

class TestLimits:
    @pytest.mark.parametrize("text,match", [
        (None, "must be text"),
        ("   \n ", "empty"),
        ("0:01 too short", "only 2 words"),
    ])
    def test_rejected(self, text, match):
        with pytest.raises(TranscriptFormatError, match=match):
            parse_transcript(text)

    def test_too_long_is_rejected_not_cut(self):
        text = "0:00 " + "word " * (ti.TRANSCRIPT_MAX_CHARS // 5 + 1)
        with pytest.raises(TranscriptFormatError, match="too long"):
            parse_transcript(text)

    def test_windows_line_endings_and_bom(self):
        t = parse_transcript("﻿0:00\r\nso today I want to talk\r\n0:03\r\nabout this one thing", language="en")
        assert _texts(t) == ["so today I want to talk", "about this one thing"]


class TestFitsTheVideo:
    def test_transcript_of_a_longer_video(self):
        t = parse_transcript(PANEL, language="en")     # last line starts at 0:12
        assert "runs to 0:12 but the video is 0:09" in ti.transcript_misfit(t, 9.0)
        assert ti.transcript_misfit(t, 10.0) is None    # within the 2 s tolerance
        assert ti.transcript_misfit(t, None) is None    # unknown duration: no verdict

    def test_coverage_note_when_it_stops_early(self):
        t = parse_transcript(PANEL, language="en")
        assert "0:16 of a 10:00 video" in ti.coverage_note(t, 600)
        assert ti.coverage_note(t, 18) is None

    def test_clock(self):
        assert ti.format_clock(59.9) == "0:59"
        assert ti.format_clock(3723) == "1:02:03"


# --- exact words where the clips are --------------------------------------------

class TestClipRanges:
    def test_padded_clamped_merged(self):
        assert ti.clip_ranges([(100, 130), (5, 20), (135, 150)], 155) == [(0.0, 28.0), (92.0, 155.0)]

    def test_no_duration_no_clamp(self):
        assert ti.clip_ranges([(10, 20)], None, pad=1.0) == [(9.0, 21.0)]


class TestMergeExactWords:
    def _pasted(self):
        return parse_transcript("0:00\none two three four five\n0:05\nsix seven eight nine ten\n"
                                "0:10\neleven twelve thirteen fourteen", language="en")

    def test_estimates_inside_the_range_are_replaced(self):
        exact = [{"start": 5.2, "end": 9.0, "text": " Six, seven, eight.",
                  "words": [{"word": " Six,", "start": 5.2, "end": 5.6},
                            {"word": " seven,", "start": 6.0, "end": 6.5},
                            {"word": " eight.", "start": 7.0, "end": 9.0}]}]
        merged = ti.merge_exact_words(self._pasted(), exact, [(5.0, 9.5)])
        words = [(w["word"].strip(), w["start"]) for w in _all_words(merged)]
        assert ("Six,", 5.2) in words and ("six", 5.0) not in words
        assert [w for w, _ in words][:5] == ["one", "two", "three", "four", "five"]
        assert "eleven" in [w for w, _ in words]
        assert merged["exact_ranges"] == [[5.0, 9.5]]
        assert [seg["start"] for seg in merged["segments"]] == sorted(seg["start"] for seg in merged["segments"])

    def test_a_line_crossing_the_range_keeps_only_its_outside_words(self):
        # estimated line 0:05-0:08 "six seven eight nine ten": "nine" starts at
        # 6.92, "ten" at 7.52; the exact range starts at 7.0
        merged = ti.merge_exact_words(self._pasted(), [], [(7.0, 9.9)])
        crossing = [seg for seg in merged["segments"] if seg["start"] == 5.0][0]
        assert crossing["text"] == "six seven eight nine"
        assert crossing["words"][-1]["end"] == 7.0   # cut short, never into the exact stretch

    def test_exact_words_outside_their_range_are_ignored(self):
        exact = [{"start": 20, "end": 21, "text": "stray", "words": [{"word": " stray", "start": 20, "end": 21}]}]
        merged = ti.merge_exact_words(self._pasted(), exact, [(5.0, 9.5)])
        assert "stray" not in [w["word"].strip() for w in _all_words(merged)]

    def test_ranges_accumulate(self):
        once = ti.merge_exact_words(self._pasted(), [], [(0.0, 2.0)])
        twice = ti.merge_exact_words(once, [], [(1.5, 4.0), (10.0, 11.0)])
        assert twice["exact_ranges"] == [[0.0, 4.0], [10.0, 11.0]]
