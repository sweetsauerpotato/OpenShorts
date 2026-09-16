"""Clips an agent chose: validation, and where their edges go.

``render_clips`` (MCP) / ``/api/process`` with ``source_job_id`` hand these to a
render job (``main.py --clips-file``). A clip is one or more pieces of the
source video in play order, so it can open on its punchline and leave dead
parts out, plus the text the pipeline burns and publishes.

Standard library only (plus recut's EDL helpers, also light): the API
validates at submit and CI imports it without the ML stack.
"""

from bisect import bisect_left

import recut

MAX_CLIPS = 15
# All pieces of one clip come out of one reframed stretch of the source (the
# clip's canonical file), so they must lie close together: reframing a
# 10-minute stretch for a 40 s clip would cost minutes of CPU per clip, and the
# clip editor's fast re-cut relies on that canonical stretch.
MAX_SPAN_SECONDS = 180.0
MIN_CLIP_SECONDS = 5.0
MAX_HOOK_CHARS = 150
MAX_TITLE_CHARS = 100          # YouTube's title limit
MAX_DESCRIPTION_CHARS = 2200   # TikTok's and Instagram's caption limit
MAX_REASON_CHARS = 500
# Silence kept around a piece, and never more than half the gap, so the
# neighbouring word is never audible. Generous at the start because a word's
# onset after a pause is where Whisper is least sure: on 16-sep-2026 two runs
# over the same audio put "Those" at 148.32 s and 147.84 s, the waveform says
# 147.87, and a 0.3 s lead cut into the word. Half of a long pause is silence
# either way; a short pause still cuts tight.
PIECE_LEAD = 0.5
PIECE_TAIL = 0.4

# A gap at least this long is not a pause between two words: it is a HOLE, a
# stretch the transcriber returned nothing for. The distinction matters because
# the padding above assumes a gap is silence, and a hole is not - on the Jake
# Paul source (16-sep-2026) the 14.2 s hole at 746.7-761.0 runs at -17.5 to
# -27.9 dB RMS, louder than the speech around it: it is the punch landing, the
# laughing, and "Welcome to team 11". Whisper heard none of it. Pulling an edge
# back out of a hole therefore throws content away, and it threw away the
# payoff of the best clip of that video.
#
# 2.0 s is where the two populations separate. Inter-word gaps over two real
# transcripts (3,566 and 8,440 words):
#
#     gap >=      0.5s    1.0s    1.5s    2.0s    3.0s    5.0s
#     Jake Paul    68      45      38      36      28      19
#     democracy   272      54      33      26      21      13
#
# Both collapse between 0.5 and 1.0 s - ordinary pauses - and flatten after
# 1.5-2.0 s. Below the threshold the old trimming is kept, so a breath still
# cuts tight.
GAP_IS_A_HOLE = 2.0


class AgentClipsError(ValueError):
    """Invalid clips: safe to show as a 400."""


def normalize_agent_clips(clips, source_duration=None):
    """Validate the agent's clips and shape them like the pipeline's.

    Returns clip dicts with ``segments`` (play order), the covering ``start``/
    ``end`` and the metadata field names the renderer, publishing and the
    dashboard already read (``viral_hook_text``, ``video_title_for_youtube_short``,
    ...). Raises AgentClipsError naming the clip and what is wrong.
    """
    if not isinstance(clips, list) or not clips:
        raise AgentClipsError("clips must be a non-empty list")
    if len(clips) > MAX_CLIPS:
        raise AgentClipsError(f"at most {MAX_CLIPS} clips per render (got {len(clips)})")

    out = []
    for n, clip in enumerate(clips, 1):
        where = f"clip {n}"
        if not isinstance(clip, dict):
            raise AgentClipsError(f"{where} must be an object")
        try:
            pieces = recut.normalize_segments(clip.get("segments"), source_duration)
        except recut.RecutError as e:
            raise AgentClipsError(f"{where}: {e}")
        start = min(p["start"] for p in pieces)
        end = max(p["end"] for p in pieces)
        total = recut.total_duration(pieces)
        if total < MIN_CLIP_SECONDS:
            raise AgentClipsError(f"{where} is {total:.1f} s long; at least {MIN_CLIP_SECONDS:g} s")
        if end - start > MAX_SPAN_SECONDS:
            raise AgentClipsError(
                f"{where}: its pieces spread over {end - start:.0f} s of the source; keep them "
                f"within {MAX_SPAN_SECONDS:.0f} s (make distant moments separate clips)")

        text = {}
        for field, limit in (("hook", MAX_HOOK_CHARS), ("title", MAX_TITLE_CHARS),
                             ("tiktok_description", MAX_DESCRIPTION_CHARS),
                             ("instagram_description", MAX_DESCRIPTION_CHARS),
                             ("reason", MAX_REASON_CHARS)):
            value = clip.get(field)
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise AgentClipsError(f"{where}: {field} must be text")
            value = value.strip()
            if len(value) > limit:
                raise AgentClipsError(
                    f"{where}: {field} is {len(value)} characters; at most {limit}")
            text[field] = value

        score = clip.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
                raise AgentClipsError(f"{where}: score must be a number from 0 to 100")
            score = int(round(score))

        out.append({
            "segments": pieces,
            "start": start,
            "end": end,
            "viral_hook_text": text["hook"],
            "video_title_for_youtube_short": text["title"],
            "video_description_for_tiktok": text["tiktok_description"],
            "video_description_for_instagram": text["instagram_description"],
            "predicted_score": score,
            "reason": text["reason"],
            "selected_by": "agent",
        })
    return out


def snap_edge(t, words, kind, max_lead=PIECE_LEAD, max_tail=PIECE_TAIL):
    """Put a piece edge in the silence beside the word it belongs to.

    ``words``: {'w','s','e'} sorted by start. ``kind``: "start" or "end". An edge
    inside a word goes to the nearer side of it, so a word is either whole in
    the piece or out of it; then it sits in the gap next to that word, at most
    ``max_lead``/``max_tail`` into the silence and never past the gap's middle.

    Inside a hole (``GAP_IS_A_HOLE``) that rule is wrong - the gap is not known
    to be silent - so an edge the caller put there deliberately is KEPT rather
    than pulled back to the neighbouring word. It still stops a full
    ``max_lead``/``max_tail`` clear of the word on the far side, so no
    neighbouring word can become audible. Only a hole bounded by a word on both
    sides qualifies: past the first or last word the transcript is no evidence
    of anything, and trimming stays the safe default.
    """
    t = float(t)
    if not words:
        return round(t, 3)
    starts = [float(w["s"]) for w in words]
    # Last word starting at or before t: the one t may fall inside.
    i = bisect_left(starts, t + 1e-9) - 1
    inside = i if 0 <= i < len(words) and float(words[i]["s"]) + 0.05 < t < float(words[i]["e"]) - 0.05 else None

    if kind == "start":
        if inside is not None:
            first = inside if t - float(words[inside]["s"]) <= float(words[inside]["e"]) - t else inside + 1
        else:
            first = bisect_left(starts, t - 0.05)
        if first >= len(words):
            return round(t, 3)
        word_start = float(words[first]["s"])
        gap = word_start - float(words[first - 1]["e"]) if first > 0 else 2 * max_lead
        tight = word_start - min(max_lead, max(0.0, gap) / 2)
        if inside is None and first > 0 and gap >= GAP_IS_A_HOLE and t < tight:
            # Deliberately inside a hole: keep it, clear of the previous word.
            floor = float(words[first - 1]["e"]) + max_tail
            return round(max(0.0, min(tight, max(t, floor))), 3)
        return round(max(0.0, tight), 3)

    if inside is not None:
        last = inside if float(words[inside]["e"]) - t <= t - float(words[inside]["s"]) else inside - 1
    else:
        last = next((k for k in range(min(i, len(words) - 1), -1, -1)
                     if float(words[k]["e"]) <= t + 0.05), -1)
    if last < 0:
        return round(t, 3)
    word_end = float(words[last]["e"])
    bounded = last + 1 < len(words)
    gap = float(words[last + 1]["s"]) - word_end if bounded else 2 * max_tail
    tight = word_end + min(max_tail, max(0.0, gap) / 2)
    if inside is None and bounded and gap >= GAP_IS_A_HOLE and t > tight:
        # Deliberately inside a hole: keep it, clear of the next word.
        ceiling = float(words[last + 1]["s"]) - max_lead
        return round(max(tight, min(t, ceiling)), 3)
    return round(tight, 3)
