"""Speech Whisper dropped, and keeping music out of what comes back.

A whole-video Whisper pass silently loses stretches of clean speech. On the
Jake Paul source (16-sep-2026) it emitted no segment at all between 746.73 and
760.97 s, jumping straight from one to the next, although that stretch is
ordinary dialogue: re-transcribing those 14 s on their own returns 90 words,
identically with ``vad_filter`` on and off. So it is not a VAD decision and not
non-speech -- it is the long-form pass dropping audio it did transcribe fine in
isolation.

Measured over the whole 23.6-min video: 36 gaps of >= 2 s (370 s, 26% of the
runtime), and re-transcribing them recovers **478 words on a 3,566-word
transcript (+13.4%)** for 114 s of CPU. Those words matter twice -- they are
captions the viewer was not getting, and they are input pass 1 scores on, so
the clip selector stops being blind to the moments inside them.

**The catch, and why this module is not just a loop.** A hole is where the
music is. Re-transcribing all 36 returns song lyrics as if they were dialogue:
"All my new friends, all my fake friends" (400.9 s), Fortunate Son (1194.6 s),
a Rick Ross verse (963.2 s). Burning those into captions would be wrong and a
copyright problem, and feeding them to the scorer would be worse.

Whisper's own confidence signals DO NOT separate them -- measured on the 26
holes that returned words, against hand labels:

    lyrics  400.9   no_speech_prob 0.019  (the lowest of all)  avg_logprob -0.423
    lyrics  963.2   no_speech_prob 0.204  avg_logprob -0.236   (the best of all)
    speech    8.9   no_speech_prob 0.366  (the highest of all)
    speech  746.7   no_speech_prob 0.083  avg_logprob -0.403

The two most confident-looking rows in the table are both music. That is the
same failure the layout picker documents from four earlier attempts: a
continuous measure that does not separate the populations. So the filter here
asks for a **decision between closed options** instead -- text-only, which is
what `llm_provider` already serves on Gemini or a local Ollama, needing no new
key and no GPU. Fragments the gate cannot be asked about are DROPPED, never
kept: today's transcript is the safe answer.

Standard library only; the Whisper and model calls live in main.py.
"""

# Same 2 s line as agent_clips.GAP_IS_A_HOLE, and the same measurement behind
# it: inter-word gaps over two real transcripts collapse between 0.5 and 1 s
# (ordinary pauses) and flatten after 1.5-2 s. Kept as its own constant because
# the two answer different questions -- that one is where a cut may land, this
# one is what is worth re-transcribing.
HOLE_SECONDS = 2.0

# Context handed to Whisper on each side. Words outside the hole are discarded:
# the padding only stops the model opening mid-utterance with nothing to latch
# onto. Words the range edge cuts in half are dropped by transcribe_range.
HOLE_PAD = 2.0

# A hole that comes back with one word is noise, not recovered speech. Every
# 1-word result in the 36-hole measurement was a fragment of a neighbouring
# sentence Whisper had already transcribed ("Oh", "Back", "I").
MIN_RECOVERED_WORDS = 2

# Don't re-transcribe a whole song: the longest holes are montages. This is a
# cost guard, not a filter -- the gate is what decides what is kept.
MAX_HOLE_SECONDS = 120.0


def transcript_words(transcript):
    """Every word of a transcript, sorted by start."""
    words = [w for seg in (transcript or {}).get("segments") or []
             for w in seg.get("words") or []]
    return sorted(words, key=lambda w: float(w["start"]))


def find_holes(transcript, min_seconds=HOLE_SECONDS, max_seconds=MAX_HOLE_SECONDS):
    """Gaps between consecutive words worth re-transcribing, in video seconds.

    Only gaps BETWEEN words: before the first word and after the last one the
    transcript is no evidence that speech is missing.
    """
    words = transcript_words(transcript)
    holes = []
    for a, b in zip(words, words[1:]):
        gap = float(b["start"]) - float(a["end"])
        if min_seconds <= gap <= max_seconds:
            holes.append((round(float(a["end"]), 3), round(float(b["start"]), 3)))
    return holes


def words_in(segments, start, end):
    """The words of ``segments`` that fall inside [start, end]."""
    return [w for seg in segments for w in seg.get("words") or []
            if float(w["start"]) >= start - 0.05 and float(w["end"]) <= end + 0.05]


def worth_gating(segments):
    """Recovered segments long enough to be real speech, in time order."""
    keep = []
    for seg in segments:
        words = seg.get("words") or []
        if len(words) >= MIN_RECOVERED_WORDS and (seg.get("text") or "").strip():
            keep.append(seg)
    return sorted(keep, key=lambda s: float(s["start"]))


def parse_gate(answer, count):
    """Which recovered fragments the gate kept: a list of ``count`` bools.

    Anything unparseable, missing or unrecognised is False -- dropping a real
    line only leaves today's transcript, while keeping a lyric publishes it.
    """
    verdicts = [False] * count
    items = (answer or {}).get("fragments") if isinstance(answer, dict) else None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < count:
            verdicts[i] = str(item.get("kind", "")).strip().lower() == "dialogue"
    return verdicts


def merge_recovered(transcript, segments):
    """A transcript with ``segments`` spliced in, in time order.

    Each added segment is marked ``recovered`` so later stages can tell it from
    what the whole-video pass produced. ``text`` is rebuilt from the result.
    """
    if not segments:
        return transcript
    out = dict(transcript or {})
    merged = list(out.get("segments") or []) + [dict(s, recovered=True) for s in segments]
    merged.sort(key=lambda s: float(s["start"]))
    out["segments"] = merged
    out["text"] = " ".join(
        part for part in ((s.get("text") or "").strip() for s in merged) if part)
    return out
