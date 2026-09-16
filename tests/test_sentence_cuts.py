"""Clips are cut on whole sentences.

15-sep-2026: of 73 clips from saved harness runs, 17 (23%) ended on a finished
sentence, and a real 50-min job ended 4 of 6 clips mid-statement although each
sentence finished 2.4-4.3 s later. Two causes: pass 2 closes on a Whisper line
and 44-57% of those lines end mid-sentence; and snap_clip_to_words takes the
NEAREST word end, so an end placed on the next line's start grabbed that line's
first word whenever the pause before it (0.54 s measured) was longer than the
word ("And", 0.36 s). The fixtures below reproduce those timings.
"""
import types

import pytest

import clip_selection as cs

PAUSE = 0.55  # silence before a new sentence; 0.54-0.60 s in the job that failed


def _speech(sentences, t0=0.0, word=0.3, gap=0.05):
    words, t = [], t0
    for sentence in sentences:
        for token in sentence.split():
            words.append({"w": " " + token, "s": round(t, 3), "e": round(t + word, 3)})
            t += word + gap
        t += PAUSE - gap
    return words


TALK = [
    "We tried the obvious fix first and it did not work at all.",          # 0
    "Nobody on the team expected what happened in the next few weeks.",    # 1
    "The numbers kept climbing even after we shut the old system down.",   # 2
    "So we went back to the logs and read every single line.",             # 3
    "That is where the real problem was hiding the whole time.",           # 4
    "It was a timer that nobody had touched in eleven years.",             # 5
    "And the person who wrote it had left the company long ago.",          # 6
    "We rewrote it in an afternoon and the graphs went flat.",             # 7
    "The lesson is simple and it is the one everyone skips.",              # 8
    "Read the boring parts of the system before you blame the new ones.",  # 9
    "Most outages we see start in code that nobody thinks about anymore.", # 10
    "That is the whole story and it still makes me laugh today.",          # 11
]
WORDS = _speech(TALK)
SPANS = cs.sentence_spans(WORDS)


def _sentence(i):
    return SPANS[i]


def _cut(start, end, words=WORDS, lo=15.0, hi=60.0, **kw):
    return cs.snap_clip_to_sentences(start, end, words, 500.0, min_duration=lo, max_duration=hi, **kw)


def _last_word(words, start, end):
    return [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.05][-1]["w"].strip()


def _first_word(words, start, end):
    return [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.05][0]["w"].strip()


# --- sentence_spans -------------------------------------------------------------

class TestSentenceSpans:
    def test_one_span_per_punctuated_sentence(self):
        assert len(SPANS) == len(TALK)
        assert SPANS[3]["text"] == TALK[3]
        assert WORDS[SPANS[3]["last"]]["w"].strip() == "line."

    @pytest.mark.parametrize("token", ['done."', "done?", "done!", "done…", "done.)", "done?’"])
    def test_terminal_punctuation_with_closing_quotes(self, token):
        spans = cs.sentence_spans(_speech([f"it is {token} next one here."]))
        assert [s["text"].split()[-1] for s in spans] == [token, "here."]

    def test_an_unpunctuated_stretch_is_split_at_a_real_pause(self):
        first = [{"w": f" a{k}", "s": k * 0.4, "e": k * 0.4 + 0.3} for k in range(60)]   # 24 s
        second = [{"w": f" b{k}", "s": 26.0 + k * 0.4, "e": 26.3 + k * 0.4} for k in range(60)]
        spans = cs.sentence_spans(first + second, max_span_seconds=30.0)
        assert [(s["first"], s["last"]) for s in spans] == [(0, 59), (60, 119)]

    def test_a_run_with_no_pauses_splits_into_balanced_chunks_not_single_words(self):
        """The documentary's 50 s stretch: no punctuation and every gap 0.00 s.
        Splitting at the 'longest' pause peeled off one word at a time."""
        run = [{"w": f" w{k}", "s": k * 0.35, "e": (k + 1) * 0.35} for k in range(180)]  # 63 s
        spans = cs.sentence_spans(run, max_span_seconds=30.0)
        lengths = [s["end"] - s["start"] for s in spans]
        assert max(lengths) <= 30.0
        assert min(lengths) >= 10.0, lengths

    def test_such_a_run_splits_before_a_capitalised_word(self):
        run = [{"w": f" w{k}", "s": k * 0.35, "e": (k + 1) * 0.35} for k in range(180)]
        run[95]["w"] = " But"
        spans = cs.sentence_spans(run, max_span_seconds=40.0)
        assert any(s["text"].startswith("But") for s in spans)

    def test_empty(self):
        assert cs.sentence_spans([]) == []


# --- the end of a clip ----------------------------------------------------------

class TestEnd:
    def test_an_end_on_the_next_sentences_start_does_not_grab_its_first_word(self):
        """The measured failure: the end sits on the start of the 'And' sentence."""
        start, end = _sentence(1)["start"], _sentence(6)["start"]
        assert cs.snap_clip_to_words(start, end, WORDS, 500.0)[1] > WORDS[_sentence(6)["first"]]["s"]
        s, e = _cut(start, end)
        assert _last_word(WORDS, s, e) == "years."

    def test_an_already_cut_clip_that_includes_that_word_is_cut_back(self):
        """What a job cut by the old code looks like: '...years. And'."""
        start = _sentence(1)["start"]
        end = WORDS[_sentence(6)["first"]]["e"]
        s, e = _cut(start, end)
        assert _last_word(WORDS, s, e) == "years."

    def test_a_mid_sentence_end_finishes_the_sentence(self):
        start = _sentence(1)["start"]
        end = WORDS[_sentence(7)["first"] + 4]["e"]           # "We rewrote it in an|"
        s, e = _cut(start, end)
        assert _last_word(WORDS, s, e) == "flat."
        assert e - end <= 8.0

    def test_an_end_that_would_take_too_long_to_finish_goes_to_the_nearest_sentence_end(self):
        long_talk = TALK[:3] + [" ".join(["on"] * 40) + " and on."] + TALK[3:]
        words = _speech(long_talk)
        spans = cs.sentence_spans(words)
        end = words[spans[3]["first"] + 3]["e"]                # 4 words into a 14 s sentence
        s, e = _cut(words[0]["s"], end, words=words, lo=5.0)
        assert _last_word(words, s, e) == "down."

    def test_finishing_past_the_maximum_length_pulls_back_instead(self):
        start = _sentence(0)["start"]
        end = WORDS[_sentence(4)["first"] + 5]["e"]            # finishing "hiding the whole time." breaks a 16 s max
        s, e = _cut(start, end, hi=16.0, lo=5.0)
        assert _last_word(WORDS, s, e) == "down."
        assert e - s <= 16.0

    def test_a_clean_end_is_only_padded_into_the_silence(self):
        start, end = _sentence(1)["start"], _sentence(5)["end"]
        s, e = _cut(start, end)
        assert _last_word(WORDS, s, e) == "years."
        assert 0 < e - end <= 0.45


# --- the start of a clip ----------------------------------------------------------

class TestStart:
    def test_a_mid_sentence_start_moves_back_to_the_sentence_start(self):
        start = WORDS[_sentence(2)["first"] + 3]["s"]          # "...climbing even after..."
        s, e = _cut(start, _sentence(7)["end"])
        assert _first_word(WORDS, s, e) == "The"
        assert start - s <= 8.0

    def test_a_start_never_moves_later_into_the_sentence(self):
        start = WORDS[_sentence(2)["first"] + 3]["s"]
        s, _ = _cut(start, _sentence(7)["end"], max_shift=0.5)  # too far back to fix
        assert s <= start + 0.05

    def test_a_two_word_tail_of_the_previous_sentence_is_dropped(self):
        start = WORDS[_sentence(2)["last"] - 1]["s"]           # "...old system down." -> "system down."
        s, e = _cut(start, _sentence(8)["end"])
        assert _first_word(WORDS, s, e) == "So"
        assert s - start <= 1.5


# --- guarantees -------------------------------------------------------------------

def test_the_band_always_holds():
    for i, a in enumerate(WORDS[:60]):
        for b in WORDS[i + 40::7]:
            s, e = _cut(a["s"], b["e"], lo=15.0, hi=30.0)
            if 15.0 <= b["e"] - a["s"] <= 30.0:
                assert 15.0 - 0.01 <= e - s <= 30.0 + 0.01, (a, b, s, e)


def test_no_sentence_end_that_fits_falls_back_to_word_snapping():
    words = _speech([" ".join(["talk"] * 70) + " end."])      # one 24 s sentence
    start, end = words[0]["s"], words[50]["e"]
    assert (_cut(start, end, words=words, lo=5.0, hi=10.0)
            == cs.snap_clip_to_words(start, end, words, 500.0, 5.0, 10.0))


def test_no_words_is_word_snapping():
    assert _cut(10.0, 40.0, words=[]) == (10.0, 40.0)


# --- the pipeline uses it -----------------------------------------------------------

class _DetailEndsOnTheNextLine:
    """Scores every window; the detail answer ends on the start of the 'And' sentence."""

    def __init__(self):
        self.models = self
        self.detail_prompts = []

    def generate_content(self, model=None, contents=None, config=None):
        import re
        schema = config.response_schema
        ids = re.findall(r'"id": "(window_\d+)"', contents)
        if schema.__name__ != "ScoreResponse":
            self.detail_prompts.append(contents)
        if schema.__name__ == "ScoreResponse":
            payload = {"windows": [{"id": i, "start": 0.0, "end": 1.0, "score": 80, "reason": "r"}
                                   for i in ids]}
        else:
            payload = {"shorts": [{"start": _sentence(1)["start"], "end": _sentence(6)["start"],
                                   "source_window_id": ids[0], "predicted_score": 80,
                                   "video_description_for_tiktok": "t",
                                   "video_description_for_instagram": "i",
                                   "video_title_for_youtube_short": "y", "viral_hook_text": "h"}]}
        return types.SimpleNamespace(parsed=schema.model_validate(payload), text="{}",
                                     candidates=[], usage_metadata=None)


def _run_pipeline(monkeypatch):
    main = pytest.importorskip("main")
    for var in ("CLIP_MIN_SECONDS", "CLIP_MAX_SECONDS", "CLIP_TARGET_MIN", "CLIP_TARGET_MAX"):
        monkeypatch.delenv(var, raising=False)
    client = _DetailEndsOnTheNextLine()
    monkeypatch.setattr(main.llm_provider, "make_client", lambda: (client, "fake"))
    segments = [{"start": WORDS[sp["first"]]["s"], "end": WORDS[sp["last"]]["e"], "text": sp["text"],
                 "words": [{"word": w["w"], "start": w["s"], "end": w["e"]}
                           for w in WORDS[sp["first"]:sp["last"] + 1]]} for sp in SPANS]
    return client, main.get_viral_clips({"language": "en", "segments": segments}, WORDS[-1]["e"])


def test_get_viral_clips_cuts_on_the_sentence(monkeypatch):
    _, result = _run_pipeline(monkeypatch)
    clip = result["shorts"][0]
    assert _last_word(WORDS, clip["start"], clip["end"]) == "years."


def test_pass_two_reads_whole_sentences_with_their_start_and_end(monkeypatch):
    client, _ = _run_pipeline(monkeypatch)
    prompt = client.detail_prompts[0]
    for sp in SPANS:
        assert f"[{sp['start']:.1f}-{sp['end']:.1f}] {sp['text']}" in prompt


# --- the pass-2 prompt ----------------------------------------------------------

def _detail_template():
    import ast
    import os
    mod = ast.parse(open(os.path.join(os.path.dirname(__file__), "..", "gemini_worker.py"),
                         encoding="utf-8").read())
    return next(node.value.value for node in mod.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "DETAIL_PROMPT_TEMPLATE")


def test_the_prompt_takes_the_end_from_the_closing_sentence():
    template = _detail_template()
    # The instruction that put 71 of 73 raw ends on the next line's start.
    assert "line just AFTER" not in template
    assert "`end`   = the SECOND number of the sentence you close on." in template
    assert "`start` = the FIRST number of the sentence you open on." in template
    assert "END ON A FINISHED THOUGHT" in template

class TestSentenceCue:
    """The cue that stands in for punctuation the transcriber left out."""

    def test_a_pause_is_a_cue(self):
        w = [{"w": " one", "s": 0.0, "e": 0.5}, {"w": " two", "s": 1.0, "e": 1.5}]
        assert cs.sentence_cue(w, 0) is True

    def test_a_capital_after_an_ordinary_word_is_a_cue(self):
        w = [{"w": " business", "s": 0.0, "e": 0.5}, {"w": " Every", "s": 0.5, "e": 1.0}]
        assert cs.sentence_cue(w, 0) is True

    @pytest.mark.parametrize("binder", ["the", "a", "of", "my", "in", "and", "The"])
    def test_a_capital_after_a_binder_is_a_proper_noun_not_a_cue(self, binder):
        # "that is the | Al-Aqsa Mosque." -- the failure a 30 s split threshold
        # produced on the documentary. No sentence starts straight after "the".
        w = [{"w": f" {binder}", "s": 0.0, "e": 0.5},
             {"w": " Al-Aqsa", "s": 0.5, "e": 1.0}]
        assert cs.sentence_cue(w, 0) is False

    def test_a_pause_beats_the_binder_guard(self):
        # The guard only suppresses the CAPITAL signal; a real breath still counts.
        w = [{"w": " the", "s": 0.0, "e": 0.5}, {"w": " Al-Aqsa", "s": 1.2, "e": 1.7}]
        assert cs.sentence_cue(w, 0) is True

    def test_i_is_never_a_cue(self):
        for word in (" I", " I'm", " I’ve"):
            w = [{"w": " said", "s": 0.0, "e": 0.5}, {"w": word, "s": 0.5, "e": 1.0}]
            assert cs.sentence_cue(w, 0) is False

    def test_out_of_range_is_false_not_an_error(self):
        w = [{"w": " one", "s": 0.0, "e": 0.5}]
        assert cs.sentence_cue(w, 0) is False and cs.sentence_cue(w, -1) is False


class TestStartInsideAnUnreachableSentence:
    """A start deep inside a long unpunctuated stretch."""

    def _run(self):
        # 30 s with no punctuation at all, one capitalised cue at 20 s.
        words = []
        for k in range(60):
            word = " Every" if k == 40 else f" w{k}"
            words.append({"w": word, "s": k * 0.5, "e": k * 0.5 + 0.5})
        words.append({"w": " end.", "s": 30.0, "e": 30.5})
        return words

    def test_it_opens_on_the_cue_instead_of_mid_phrase(self):
        # The sentence starts at 0 s and the clip at 24 s -- 24 s back is far
        # past max_shift, so before this the start simply stayed mid-phrase.
        words = self._run()
        start, end = _cut(24.0, 30.5, words=words, lo=5.0, hi=60.0)
        assert start == pytest.approx(20.0)          # the " Every" cue
        assert start >= 24.0 - 8.0                   # never further than max_shift

    def test_it_never_moves_the_start_later(self):
        words = self._run()
        start, _ = _cut(24.0, 30.5, words=words, lo=5.0, hi=60.0)
        assert start <= 24.0

    def test_no_cue_in_reach_leaves_the_start_alone(self):
        # Every word lowercase and contiguous: nothing to snap to, so the
        # model's own start stands rather than being dragged somewhere worse.
        words = [{"w": f" w{k}", "s": k * 0.5, "e": k * 0.5 + 0.5} for k in range(60)]
        words.append({"w": " end.", "s": 30.0, "e": 30.5})
        start, _ = _cut(24.0, 30.5, words=words, lo=5.0, hi=60.0)
        assert 23.0 <= start <= 24.5
