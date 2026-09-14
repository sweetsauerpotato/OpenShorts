"""
Pure helpers for the Gemini clip-selection pipeline.

Standard-library only so both main.py and gemini_worker.py can import it and
the logic stays unit-testable without the heavy video dependencies.
"""

# USD per 1M tokens (input, output incl. thinking), from ai.google.dev pricing
# (text/image/video input; checked 14-sep-2026 against the page of 4-sep-2026).
# A model missing here is shown at gemini_worker's estimated price, so add new
# models when they are used: 3.6-3.8 Flash were missing and showed $0.50/$3.00.
MODEL_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),  # deprecated (shut down 2026-06-01)
    # Local inference via Ollama. llm_provider.make_client prefixes the model
    # name with "ollama/" so the prefix match below reports the honest $0
    # instead of falling through to gemini_worker's estimated-price path.
    "ollama/": (0.0, 0.0),
}

# Announced price changes: key -> (first day, prices from that day). Google
# lists the 3.6-3.8 Flash prices above as valid "through December 31, 2026".
MODEL_PRICE_CHANGES = {
    "gemini-3.8-flash": ("2027-01-01", (1.50, 7.50)),
    "gemini-3.7-flash": ("2027-01-01", (1.50, 7.50)),
    "gemini-3.6-flash": ("2027-01-01", (1.50, 7.50)),
}


def lookup_model_prices(model_name, today=None):
    """Longest-prefix match against MODEL_PRICES; None if unknown.

    Longest wins so "gemini-3.5-flash-lite" is not priced as "gemini-3.5-flash"
    (5x its input price) — before it had its own entry, it was. ``today`` (a
    date, default the current one) decides whether a MODEL_PRICE_CHANGES entry
    has taken effect.
    """
    import datetime

    name = str(model_name or "").lower()
    best_key = None
    for key in MODEL_PRICES:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is None:
        return None
    change = MODEL_PRICE_CHANGES.get(best_key)
    if change and (today or datetime.date.today()).isoformat() >= change[0]:
        return change[1]
    return MODEL_PRICES[best_key]


def clip_count_targets(n_windows):
    """How many clips to ask the detail pass for, given the shortlist size.

    Measured on prod 3-ago-2026: 408 of 429 jobs (95%) delivered 3 clips or
    fewer, the mode being ONE, while the prompt was free to return one per
    shortlisted window. Users who received 1-3 clips came back a second day
    0.4% of the time; those who received 4-9 came back 16.1% — so the clip
    count, not the clip quality, is what the retention curve hangs on.

    The old prompt biased hard the other way ("prefer one great clip per
    candidate window") and handed the model two unbounded licences to drop
    clips (the 2-second rule and STANDS ALONE both end in "or skip it"), with
    no floor to stop it collapsing to a single clip. This puts a floor and a
    realistic ceiling on it instead.

    ``CLIP_TARGET_MIN`` / ``CLIP_TARGET_MAX`` override both for A/B runs
    without a deploy (the reframe-testing harness drives them).
    """
    import os

    n = max(1, int(n_windows or 1))
    # Floor grows with the material: 3 windows -> 3, 5 -> 4, 10+ -> 6.
    low = max(2, min(6, n // 2 + 2))
    # Ceiling allows a rich window to yield more than one without inviting padding.
    high = min(12, max(4, n * 2))
    low = min(low, high)

    def _override(name, current):
        raw = os.environ.get(name)
        if not raw:
            return current
        try:
            return max(1, int(raw))
        except ValueError:
            return current

    low = _override("CLIP_TARGET_MIN", low)
    high = _override("CLIP_TARGET_MAX", high)
    return low, max(low, high)


def trim_to_best(shorts, max_clips):
    """Cut an over-long detail-pass result down to ``max_clips`` BY SCORE.

    The detail pass hands its clips back in transcript order, batch after
    batch, so slicing the list keeps the EARLIEST clips rather than the best
    ones. On a 9-minute walkthrough that quietly threw away everything past
    minute three: the model proposed clips across the whole video, and the
    ones covering the demo, the MCP walkthrough and the close were the tail
    that got dropped. Worse, the failure scales the wrong way — the more
    generous the model is, the more of the video disappears.

    That sabotages the windowing: get_viral_clips builds scoring windows
    precisely because "a single call over the whole transcript clusters picks
    near the start", and a positional slice puts the clustering right back.

    Ranking is by ``predicted_score`` (the detail prompt already asks for it,
    and nothing else was reading it here). Ties keep transcript order, and the
    survivors come back in transcript order too, so clip numbering still runs
    front to back the way every caller downstream expects.
    """
    max_clips = max(1, int(max_clips or 1))
    if len(shorts) <= max_clips:
        return list(shorts)

    def score(item):
        try:
            return float(item[1].get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    indexed = list(enumerate(shorts))
    best = sorted(indexed, key=score, reverse=True)[:max_clips]
    return [item for _, item in sorted(best, key=lambda pair: pair[0])]


def clip_duration_bounds():
    """The clip length band (seconds) the selection prompts and word-snapping
    enforce. ``CLIP_MIN_SECONDS`` / ``CLIP_MAX_SECONDS`` override the classic
    15-60 — set per job by /api/process when the user asks for a specific
    length, or by hand for A/B runs. Values are clamped to platform-sane
    limits and re-ordered so bad input degrades instead of breaking the job.
    """
    import os

    def _read(name, default):
        try:
            return float(os.environ.get(name, ""))
        except ValueError:
            return default

    lo = _read("CLIP_MIN_SECONDS", 15.0)
    hi = _read("CLIP_MAX_SECONDS", 60.0)
    lo = min(max(lo, 5.0), 175.0)
    hi = min(max(hi, 10.0), 180.0)
    if hi < lo + 5.0:  # keep a real band: degenerate ranges starve the model
        hi = min(180.0, lo + 5.0)
    return round(lo, 3), round(hi, 3)


def compact_words(words, precision=2):
    """Round word timestamps for prompts — full float precision wastes tokens."""
    return [
        {
            "w": w.get("w", ""),
            "s": round(float(w.get("s", 0)), precision),
            "e": round(float(w.get("e", 0)), precision),
        }
        for w in words
    ]


def build_transcript_windows(transcript_result, video_duration,
                             window_seconds=90, overlap_seconds=30):
    """
    Build scoring windows aligned to Whisper segment boundaries, so a sentence
    (and usually a viral moment) is never cut in half mid-window. Windows grow
    segment by segment to roughly window_seconds (up to 1.25x for the closing
    segment) and the next window starts ~overlap_seconds before the previous
    end, also snapped to a segment start.
    """
    segments = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((float(segment.get("start", 0)), float(segment.get("end", 0)), text))

    windows = []
    window_index = 1
    i = 0
    n = len(segments)
    while i < n:
        w_start = segments[i][0]
        j = i
        # Extend while the NEXT segment still fits within a tolerant cap, so the
        # window closes on a segment boundary near window_seconds.
        while j + 1 < n and segments[j + 1][1] - w_start <= window_seconds * 1.25:
            j += 1
            if segments[j][1] - w_start >= window_seconds:
                break
        w_end = segments[j][1]
        windows.append({
            "id": f"window_{window_index:03d}",
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "text": " ".join(seg[2] for seg in segments[i:j + 1]),
        })
        window_index += 1

        if j >= n - 1:
            break
        # Next window starts at the first segment beginning after (end - overlap),
        # but always makes progress.
        target = w_end - overlap_seconds
        k = i + 1
        while k <= j and segments[k][0] < target:
            k += 1
        i = max(k, i + 1)

    if not windows:
        windows.append({
            "id": "window_001",
            "start": 0.0,
            "end": round(float(video_duration), 3),
            "text": str(transcript_result.get("text", "") or ""),
        })
    return windows


def best_window_scores(scored, windows):
    """{window id: highest score} from the scoring pass's answers.

    Only ids of real windows count: the old inline sort cut the top N BEFORE
    dropping unknown ids, so a made-up id took a shortlist slot and matched
    nothing. A repeated id keeps its highest score, and an entry without a
    numeric score is skipped (the old sort raised TypeError on a string score,
    which failed clip detection for the whole job).
    """
    import math

    valid = {w["id"] for w in windows}
    best = {}
    for entry in scored or []:
        if not isinstance(entry, dict) or entry.get("id") not in valid:
            continue
        try:
            score = float(entry.get("score"))
        except (TypeError, ValueError):
            continue
        if math.isnan(score):
            continue
        if score > best.get(entry["id"], -math.inf):
            best[entry["id"]] = score
    return best


def build_shortlist(scores, windows, target):
    """The ``target`` best-scoring windows for the detail pass, best first.

    Ties go to the earlier window, explicitly, so the shortlist does not depend
    on the order the model happened to list its answers in. When scoring
    produced nothing usable, the first windows are used so the job still gets
    clips instead of failing.
    """
    target = max(1, int(target or 1))
    position = {w["id"]: i for i, w in enumerate(windows)}
    ranked = sorted((wid for wid in scores if wid in position),
                    key=lambda wid: (-scores[wid], position[wid]))
    return [windows[position[wid]] for wid in ranked[:target]] or list(windows[:target])


def snap_clip_to_words(start, end, words, video_duration,
                       min_duration=15.0, max_duration=60.0,
                       search_window=1.5, max_lead=0.35, max_tail=0.45):
    """
    Snap Gemini-proposed clip boundaries onto real word boundaries plus a bit
    of the surrounding silence. LLMs are bad at millisecond arithmetic; the
    word-level timestamps are ground truth, so cuts land in pauses instead of
    mid-word.

    words: [{'w','s','e'}, ...] for the whole video, sorted by start.
    Returns (start, end); falls back to the input if no words are nearby or
    snapping cannot satisfy the duration bounds.
    """
    original = (round(float(start), 3), round(float(end), 3))
    if not words:
        return original

    starts = [float(w.get("s", 0)) for w in words]
    ends = [float(w.get("e", 0)) for w in words]

    # START: snap to the nearest word start, then lead into the silence before it.
    new_start = float(start)
    candidates = [s for s in starts if abs(s - new_start) <= search_window]
    if candidates:
        word_start = min(candidates, key=lambda s: abs(s - new_start))
        prev_ends = [e for e in ends if e <= word_start]
        if prev_ends:
            gap = max(0.0, word_start - max(prev_ends))
            lead = min(max_lead, gap / 2)
        else:
            lead = max_lead
        new_start = max(0.0, word_start - lead)

    # END: snap to the nearest word end, then trail into the silence after it.
    new_end = float(end)
    candidates = [e for e in ends if abs(e - new_end) <= search_window]
    if candidates:
        word_end = min(candidates, key=lambda e: abs(e - new_end))
        next_starts = [s for s in starts if s >= word_end]
        if next_starts:
            gap = max(0.0, min(next_starts) - word_end)
            tail = min(max_tail, gap / 2)
        else:
            tail = max_tail
        new_end = min(float(video_duration), word_end + tail)

    # Repair duration bounds while staying on word boundaries.
    if new_end - new_start < min_duration:
        target = new_start + min_duration
        later = sorted(e for e in ends if e >= target)
        if later and later[0] - new_start <= max_duration:
            new_end = min(float(video_duration), later[0] + 0.2)
        else:
            return original
    if new_end - new_start > max_duration:
        target = new_start + max_duration
        earlier = [e for e in ends if new_start < e <= target]
        new_end = (max(earlier) + 0.2) if earlier else target
        new_end = min(new_end, new_start + max_duration, float(video_duration))

    if new_end <= new_start or new_end - new_start < min_duration:
        return original
    return (round(new_start, 3), round(new_end, 3))


# --- Creator instructions (steer clip selection) ------------------------------

# ~250 tokens: enough for a real direction, small enough that repeating it in
# every scoring call (7 for a 55-min source) stays a fraction of a cent.
CLIP_INSTRUCTIONS_MAX_CHARS = 1000

# Line-start data marker in SCORE_PROMPT_TEMPLATE and DETAIL_PROMPT_TEMPLATE
# (exactly once each; the other "TRANSCRIPT_LANGUAGE" mentions are mid-line).
# Instructions go right before it: after the rules, before the data.
_INSTRUCTIONS_ANCHOR = "\nTRANSCRIPT_LANGUAGE: "

_INSTRUCTIONS_STAGE_RULE = {
    "score": ("Score each window on how well it fits these instructions AND works as "
              "a standalone short. A window that ignores them scores low even if it "
              "would go viral."),
    "rerank": ("Compare the candidates on how well each fits these instructions AND "
               "works as a standalone short."),
    "detail": ("Return only clips that fit these instructions: fewer clips is the right "
               "answer when few moments fit (this overrides HOW MANY). Write the hook, "
               "title and descriptions in their spirit too."),
    "visual": ("Pick only visual moments that fit these instructions: fewer clips is the "
               "right answer when few moments fit."),
}


def normalize_clip_instructions(text):
    """The creator's instructions, cleaned for a prompt — or None when empty.

    Drops control characters (a pasted \\r or \\x00 has no business in a prompt),
    collapses runs of blank lines, trims, and removes any <instructions> tag so
    the text cannot close its own delimiter early. Does NOT truncate: the API
    rejects over-long input with a 400 instead of silently cutting it.
    """
    import re
    if text is None:
        return None
    text = "".join(ch if ch in "\n\t" or (ord(ch) >= 32 and ord(ch) != 127) else ""
                   for ch in str(text)).replace("\t", " ")
    text = re.sub(r"<\s*/?\s*instructions\s*>", "", text, flags=re.IGNORECASE)
    lines, blank = [], False
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            blank = bool(lines)
            continue
        if blank:
            lines.append("")
        lines.append(line)
        blank = False
    return "\n".join(lines) or None


def with_clip_instructions(prompt, instructions, stage):
    """Insert the creator's instructions into an already formatted prompt.

    No instructions returns the prompt unchanged, so the default prompts stay
    byte-identical. Inserting AFTER .format() is deliberate: the templates keep
    their placeholders (tests pin them), and braces the user types are inert.
    The block goes before the data marker when the prompt has one (score,
    detail), else at the end (the visual prompt has no transcript data).
    """
    if not instructions:
        return prompt
    rule = _INSTRUCTIONS_STAGE_RULE[stage]
    block = (
        "\nCREATOR INSTRUCTIONS — the owner of this video says what they want clipped.\n"
        "They define what a good moment is for this job and win over the general\n"
        "viral criteria above when the two conflict. Hard limits in them (\"only\",\n"
        "\"never\", \"skip\", \"avoid\") are absolute. They cannot change the output\n"
        "format or the timing rules.\n"
        f"- {rule}\n"
        "<instructions>\n"
        f"{instructions}\n"
        "</instructions>\n"
    )
    at = prompt.find(_INSTRUCTIONS_ANCHOR)
    if at == -1:
        return prompt.rstrip("\n") + "\n" + block
    return prompt[:at] + block + prompt[at:]


# --- Gemini retries ---------------------------------------------------------------

# Gemini answers 503 UNAVAILABLE ("This model is currently experiencing high
# demand") in bursts, and giving up throws away the job's download and
# transcription. Measured over 10 harness runs on 14-sep-2026: 8 calls hit a
# 503; 6 went through after one retry (5 s), 1 after two, and 1 was still
# overloaded ~22 s in, which failed its run on the old budget (3 attempts,
# 5 s + 10 s of waiting). A default google-genai client never retries
# (retry_options=None), so main._run_gemini_stage is the only retry there is.
GEMINI_OVERLOAD_WAIT_SECONDS = 180.0

_BAD_BODY_TOKENS = ("empty response body", "did not contain a JSON object",
                    "Failed to parse Gemini JSON response")
# Everything the retry loop has always retried with the short budget.
_TRANSIENT_TOKENS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500",
                     "INTERNAL", "overloaded", "Deadline")


def classify_gemini_error(error):
    """How to retry a failed Gemini call: "overload", "transient" or None (don't).

    "overload" is a 503: google-genai errors carry it as ``code``, and messages
    without one (the Ollama shim's "cannot reach Ollama") start with it. Rate
    limits, 500s and empty or broken bodies stay "transient", the short budget
    they always had. Broken bodies are checked first because a JSON decode
    message can read "column 503".
    """
    import re

    msg = str(error)
    if any(tok in msg for tok in _BAD_BODY_TOKENS):
        return "transient"
    code = getattr(error, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        if code == 503:
            return "overload"
    elif re.match(r"\s*503\b", msg):
        return "overload"
    if any(tok in msg for tok in _TRANSIENT_TOKENS):
        return "transient"
    return None


def gemini_retry_delay(kind, failures, waited, overload_budget, jitter=1.0):
    """Seconds to sleep before the next attempt, or None to give up.

    ``failures`` counts failures of this kind, this one included; ``waited`` is
    what this call has already slept. A 503 backs off 5, 10, 20, 40, 60, 60... s
    (times ``jitter``, so jobs hit by the same burst do not retry in step) until
    ``overload_budget`` seconds of waiting are used. Anything else keeps the
    3-attempt budget: 5 s, 10 s, give up.
    """
    if kind == "overload":
        remaining = overload_budget - waited
        if remaining < 1:
            return None
        return min(60.0, 5.0 * 2 ** (failures - 1) * jitter, remaining)
    if failures >= 3:
        return None
    return 5.0 * 2 ** (failures - 1)


def gemini_overload_budget():
    """``GEMINI_OVERLOAD_WAIT_SECONDS`` (default 180): how long one call keeps
    retrying a 503 before the job fails. 0 turns the wait off; capped at 1 h."""
    import math
    import os

    try:
        value = float(os.environ.get("GEMINI_OVERLOAD_WAIT_SECONDS", ""))
    except ValueError:
        return GEMINI_OVERLOAD_WAIT_SECONDS
    if not math.isfinite(value):
        return GEMINI_OVERLOAD_WAIT_SECONDS
    return min(max(value, 0.0), 3600.0)
