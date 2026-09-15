"""A transcript the user already has, turned into the pipeline's transcript.

Transcription is the slow part of a job before any render (a 3-min slice took
64 s of CPU on the self-host box), and YouTube already shows a transcript for
most videos. The dashboard and MCP accept one pasted in; the job then skips
transcription entirely (``main.py --transcript``).

Accepted, detected automatically:

- YouTube's "Show transcript" panel copied as it is: a timestamp ("1:02",
  "1:02:03") on its own line or starting the line, the words after it.
  Transcript tools' "[01:02]" and "(1:02)" prefixes read the same way.
- SRT and WebVTT caption files. YouTube's auto-caption VTT carries a time
  per word and its rolling repeated lines are dropped.
- OpenShorts/Whisper transcript JSON ({"segments": [...]}), or a job's
  metadata file (its "transcript").

The pipeline wants a time per word: cuts snap to words and captions show them
word by word. Only VTT word tags and JSON words carry that. Everything else
gets each line's words spread over the line by length, so their times are
estimates, and the result says so ("timing": "line").

Standard library only: app.py validates a paste at submit (a clear 400
instead of a job that fails minutes later) and CI imports it without the ML
stack.
"""

import html
import json
import re

# ~4.5 h of English speech as YouTube prints it. Also keeps a multipart form
# field under Starlette's 1 MiB part limit even at 3 bytes per character.
TRANSCRIPT_MAX_CHARS = 300_000
MIN_WORDS = 8
# An estimated line never spreads its words wider than this per word: a line
# followed by a pause (music, a cut) would otherwise stretch its last words
# across the silence, and a clip cut there would hold a caption over nothing.
MAX_SECONDS_PER_WORD = 0.6
MIN_LINE_SECONDS = 0.5
# A transcript whose last line starts this long after the video ends belongs
# to another video (or another cut of it).
FIT_TOLERANCE_SECONDS = 2.0
# Whisper transcribes this much around every chosen clip. It matches
# snap_clip_to_sentences' max_shift (8 s): a cut may move that far to finish or
# open its sentence, and it has to find exact words when it does.
EXACT_PAD_SECONDS = 8.0


class TranscriptFormatError(ValueError):
    """The text is not a transcript this module can read (safe as a 400)."""


_TS = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_CUE = re.compile(rf"^\s*{_TS}\s*-->\s*{_TS}")
_BRACKETED_TS = re.compile(rf"^[\[(]{_TS}[\])]\s*(.*)$")
_BARE_TS = re.compile(rf"^{_TS}(?:\s+(.*))?$")
# Screen-reader duration YouTube can put under a panel timestamp ("2 seconds",
# "1 minute, 5 seconds"); never speech when it directly follows a timestamp.
_UNIT = r"(?:seconds?|minutes?|hours?|segundos?|minutos?|horas?)"
_A11Y = re.compile(rf"^\d+\s+{_UNIT}(?:,?\s+(?:and\s+|y\s+)?\d+\s+{_UNIT})*$", re.I)
_WORD_TIME = re.compile(r"<((?:\d{2}:)?\d{2}:\d{2}\.\d{3})>")
_MARKUP = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}")
# Not speech: kept in the line text, never turned into words (a caption would
# otherwise burn "[MUSIC]"). YouTube's auto-captions mark a new speaker with
# ">>" and a bleeped word with "[ __ ]": a real 23-min paste had 288 and 44.
_SOUND_TAG = re.compile(r"\[[^\]]*\]|♪+|>>+")
_VTT_LANGUAGE = re.compile(r"^\s*Language:\s*([A-Za-z]{2,3})\b")


def parse_transcript(text, language=None):
    """Read ``text`` and return the transcript contract of transcribe_backends.

    Adds ``"origin": "pasted"`` and ``"timing": "word" | "line"``. Raises
    TranscriptFormatError with a message meant for the person who pasted it.
    """
    if not isinstance(text, str):
        raise TranscriptFormatError("The transcript must be text.")
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > TRANSCRIPT_MAX_CHARS:
        raise TranscriptFormatError(
            f"The transcript is too long ({len(text):,} characters; the limit is "
            f"{TRANSCRIPT_MAX_CHARS:,}).")
    stripped = text.strip()
    if not stripped:
        raise TranscriptFormatError("The transcript is empty.")

    if stripped[0] in "{[" and not _BRACKETED_TS.match(stripped.split("\n", 1)[0]):
        segments, timing, found_language = _parse_json(stripped)
    elif any(_CUE.match(line) for line in stripped.split("\n")):
        segments, timing, found_language = _parse_cues(stripped)
    else:
        segments, timing, found_language = _parse_timestamped_lines(stripped)

    n_words = sum(len(seg["words"]) for seg in segments)
    if n_words < MIN_WORDS:
        raise TranscriptFormatError(
            f"The transcript has only {n_words} words with times; "
            f"at least {MIN_WORDS} are needed to find clips.")

    full_text = " ".join(seg["text"].strip() for seg in segments)
    return {
        "text": full_text,
        "language": (language or found_language or _detect_language(full_text)),
        "segments": segments,
        "origin": "pasted",
        "timing": timing,
    }


def transcript_misfit(transcript, video_duration):
    """Why this transcript cannot belong to a video this long, or None."""
    segments = (transcript or {}).get("segments") or []
    if not segments or not video_duration or video_duration <= 0:
        return None
    last_start = max(float(seg["start"]) for seg in segments)
    if last_start > float(video_duration) + FIT_TOLERANCE_SECONDS:
        return (f"The transcript runs to {format_clock(last_start)} but the video is "
                f"{format_clock(video_duration)} long. Is it the transcript of another video?")
    return None


def coverage_note(transcript, video_duration):
    """A warning when the transcript stops well before the video does, or None."""
    segments = (transcript or {}).get("segments") or []
    if not segments or not video_duration or video_duration <= 0:
        return None
    last_end = max(float(seg["end"]) for seg in segments)
    if last_end < 0.8 * float(video_duration):
        return (f"The transcript stops at {format_clock(last_end)} of a "
                f"{format_clock(video_duration)} video: clips can only come from the part "
                f"it covers.")
    return None


def clip_ranges(bounds, video_duration, pad=EXACT_PAD_SECONDS):
    """Padded (start, end) ranges around clip bounds, overlaps merged, in order."""
    ranges = []
    for start, end in sorted((float(s), float(e)) for s, e in bounds):
        lo = max(0.0, start - pad)
        hi = end + pad
        if video_duration:
            hi = min(float(video_duration), hi)
        if ranges and lo <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], hi)
        else:
            ranges.append([lo, hi])
    return [(round(lo, 3), round(hi, 3)) for lo, hi in ranges]


def merge_exact_words(transcript, exact_segments, ranges):
    """The pasted transcript with exact words (Whisper's) inside ``ranges``.

    Estimated words inside a range are dropped, even where Whisper heard
    nothing (music, silence): an estimate there is a word nobody says. Outside
    the ranges the estimates stay, cut short where they would run into an
    exact word. ``exact_ranges`` records where the times are real.
    """
    def inside(t):
        return any(lo <= t < hi for lo, hi in ranges)

    kept = []
    for seg in (transcript or {}).get("segments") or []:
        words = []
        for w in seg.get("words") or []:
            start, end = float(w["start"]), float(w["end"])
            if inside(start):
                continue
            for lo, _hi in ranges:
                if start < lo < end:
                    end = lo
            words.append({**w, "end": round(end, 3)})
        if not words:
            continue
        if len(words) == len(seg.get("words") or []):
            kept.append({**seg, "words": words})
        else:
            kept.append({"start": words[0]["start"], "end": words[-1]["end"],
                         "text": "".join(w["word"] for w in words).strip(), "words": words})

    exact = []
    for seg in exact_segments or []:
        words = [w for w in seg.get("words") or [] if inside(float(w["start"]))]
        if words:
            exact.append({"start": words[0]["start"], "end": words[-1]["end"],
                          "text": str(seg.get("text") or "").strip(), "words": words})

    segments = sorted(kept + exact, key=lambda seg: float(seg["start"]))
    merged = dict(transcript or {})
    merged["segments"] = segments
    merged["text"] = " ".join(seg["text"].strip() for seg in segments)
    merged["exact_ranges"] = [list(r) for r in clip_ranges(
        [tuple(r) for r in (transcript or {}).get("exact_ranges") or []] + list(ranges),
        None, pad=0.0)]
    return merged


def map_to_exact(t, estimated_words, exact_words, prefer=None):
    """Where time ``t`` on a pasted transcript's estimated words falls on exact ones.

    Both are {'w','s','e'} word lists. They are aligned by their text, and
    ``t`` is interpolated between the nearest matched words on either side,
    so an edge chosen "after the word X" lands after the real X even when the
    estimate was seconds off (YouTube's panel lines ran ~20 s on a real paste,
    and each word's time inside them is a guess). None when nothing matches.

    ``prefer`` settles the tie where one word ends exactly where the next
    begins, which is most of a transcript: "end" keeps the earlier time (stay
    with the word before the edge), "start" the later one (go with the word
    after it). Without it, a clip's end jumped onto the next word.
    """
    import difflib

    def norm(w):
        return re.sub(r"[^\w]", "", str(w.get("w", "")).lower())

    est = [w for w in estimated_words if norm(w)]
    exa = [w for w in exact_words if norm(w)]
    matcher = difflib.SequenceMatcher(None, [norm(w) for w in est], [norm(w) for w in exa],
                                      autojunk=False)
    anchors = []
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            anchors.append((float(est[a + k]["s"]), float(exa[b + k]["s"])))
            anchors.append((float(est[a + k]["e"]), float(exa[b + k]["e"])))
    if not anchors:
        return None
    anchors.sort()
    # A slip in the alignment must never fold time back on itself.
    steady = []
    for est_t, exact_t in anchors:
        if not steady or exact_t >= steady[-1][1]:
            steady.append((est_t, exact_t))
    t = float(t)
    before = [p for p in steady if p[0] <= t]
    after = [p for p in steady if p[0] > t]

    def pick(group, at_end):
        """Of the anchors sharing one estimated time, the one this edge wants."""
        tied = [p for p in group if p[0] == (group[-1][0] if at_end else group[0][0])]
        if prefer in ("end", "start"):
            return tied[0] if prefer == "end" else tied[-1]
        return tied[-1] if at_end else tied[0]

    if before and after:
        (e0, x0), (e1, x1) = pick(before, True), pick(after, False)
        if e1 <= e0:
            return x0
        return x0 + (t - e0) * (x1 - x0) / (e1 - e0)
    e0, x0 = pick(before, True) if before else pick(after, False)
    return x0 + (t - e0)


def line_span(transcript, t):
    """The stretch the pasted line holding ``t`` really covers: from its
    timestamp to the next line's. The line times are reliable (a real YouTube
    paste: median 0.26 s from Whisper's, never 2 s off); where a word sits
    inside a 20 s line is only an estimate, so the word can be anywhere in it."""
    segments = (transcript or {}).get("segments") or []
    t = float(t)
    for i, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        reach = max(end, float(segments[i + 1]["start"])) if i + 1 < len(segments) else end
        if start <= t < reach:
            return start, reach
    return t, t


def covered(start, end, exact_ranges, video_duration=None, margin=1.0):
    """True when [start, end] lies inside one exact range with ``margin`` to
    spare on each side (none needed where the range meets the video's edge)."""
    for lo, hi in exact_ranges or []:
        lo_ok = start - lo >= margin or lo <= 0.0
        hi_ok = hi - end >= margin or (video_duration is not None and hi >= video_duration - 0.05)
        if lo <= start and end <= hi and lo_ok and hi_ok:
            return True
    return False


def format_clock(seconds):
    seconds = int(max(0.0, float(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# --- formats -------------------------------------------------------------------

def _parse_json(text):
    try:
        data = json.loads(text)
    except ValueError as e:
        raise TranscriptFormatError(f"The transcript looks like JSON but is not valid JSON ({e}).")
    language = None
    if isinstance(data, dict) and isinstance(data.get("transcript"), dict):
        data = data["transcript"]  # a job's metadata file
    if isinstance(data, dict):
        language = data.get("language") if isinstance(data.get("language"), str) else None
        data = data.get("segments")
    if not isinstance(data, list):
        raise TranscriptFormatError(
            'JSON transcripts need a "segments" list of {"start", "end", "text"}.')

    segments = []
    timing = "word"
    for i, seg in enumerate(data):
        try:
            start = float(seg["start"])
            end = float(seg["end"])
            line = str(seg.get("text") or "")
        except (KeyError, TypeError, ValueError):
            raise TranscriptFormatError(
                f"JSON segment {i + 1} needs numeric start and end, and text.")
        words = _json_words(seg.get("words"))
        if words is None:
            timing = "line"
            words = _spread_words(line, start, max(end, start))
        if words:
            segments.append({"start": start, "end": max(end, start), "text": line,
                             "words": words})
    segments.sort(key=lambda seg: seg["start"])
    return segments, timing, language


def _json_words(raw):
    if not isinstance(raw, list) or not raw:
        return None
    words = []
    for w in raw:
        try:
            token = str(w["word"])
            words.append({"word": token if token.startswith(" ") else " " + token,
                          "start": float(w["start"]), "end": float(w["end"])})
        except (KeyError, TypeError, ValueError):
            return None
    return words


def _parse_cues(text):
    """SRT or WebVTT: cue times, cue text; YouTube's word-timed VTT included."""
    language = None
    cues = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        match = _CUE.match(lines[i])
        if not match:
            found = _VTT_LANGUAGE.match(lines[i])
            if found and not cues:
                language = found.group(1).lower()
            i += 1
            continue
        start = _seconds(match.groups()[0:4])
        end = _seconds(match.groups()[4:8])
        body, i = _cue_body(lines, i + 1)
        if not body and i + 1 < len(lines) and lines[i] == "" and lines[i + 1].strip() \
                and not _CUE.match(lines[i + 1]):
            # YouTube's VTT opens a cue with a space-only line before the words;
            # an editor or a paste that strips trailing spaces leaves it empty.
            body, i = _cue_body(lines, i + 1)
        if start is not None and end is not None:
            cues.append((start, end, body))

    word_timed = any(_WORD_TIME.search(line) for _, _, body in cues for line in body)
    segments = []
    last_text = None
    for start, end, body in cues:
        if end - start < 0.05:
            continue  # YouTube's 10 ms "hold" cues repeat the previous line
        if word_timed:
            # Rolling auto-captions: the first line repeats what was already
            # shown, the last line is the new words with their times.
            candidates = [line for line in body if _clean(line)]
            if not candidates:
                continue
            raw = candidates[-1]
            line = _clean(_WORD_TIME.sub(" ", raw))
            if line == last_text:
                continue
            words = _tagged_words(raw, start, end)
        else:
            line = _clean(" ".join(body))
            if not line or line == last_text:
                continue
            words = _spread_words(line, start, end)
        last_text = line
        if words:
            segments.append({"start": start, "end": end, "text": line, "words": words})
    segments.sort(key=lambda seg: seg["start"])
    return segments, ("word" if word_timed else "line"), language


def _parse_timestamped_lines(text):
    """YouTube's transcript panel and similar: a timestamp, then its words."""
    entries = []  # [start, [text lines]]
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        stamp, rest = _line_timestamp(line)
        # A time far behind the previous one is a line that happens to start
        # with a clock ("10:30 is when..."), not a new timestamp.
        if stamp is not None and (not entries or stamp >= entries[-1][0] - 1.0):
            entries.append([stamp, [rest] if rest else []])
            continue
        if not entries:
            continue  # a header before the first timestamp ("Transcript") is not speech
        if not entries[-1][1] and _A11Y.match(line):
            continue
        entries[-1][1].append(line)

    if not entries:
        raise TranscriptFormatError(
            "No timestamps found. Paste the transcript with its times: on YouTube, "
            "open the description, click 'Show transcript' and copy the list as it is "
            "(lines like '1:02 so today we…'). SRT and VTT caption files work too.")

    # Several lines can share a whole-second timestamp: one line, in order.
    entries.sort(key=lambda entry: entry[0])
    merged = []
    for stamp, parts in entries:
        if merged and abs(stamp - merged[-1][0]) < 1e-6:
            merged[-1][1].extend(parts)
        else:
            merged.append([stamp, list(parts)])

    segments = []
    for index, (start, parts) in enumerate(merged):
        line = _clean(" ".join(parts))
        if not line:
            continue
        n_words = len(_speech_tokens(line))
        end = start + max(MIN_LINE_SECONDS, n_words * MAX_SECONDS_PER_WORD)
        if index + 1 < len(merged):
            end = min(end, merged[index + 1][0])
        words = _spread_words(line, start, end)
        if words:
            segments.append({"start": start, "end": end, "text": line, "words": words})
    return segments, "line", None


# --- helpers -------------------------------------------------------------------

def _cue_body(lines, i):
    """A cue's text lines from ``i``, and the index after them.

    Only an EMPTY line ends a cue: YouTube's VTT puts a space-only line
    between the time and the words.
    """
    body = []
    while i < len(lines) and lines[i] != "" and not _CUE.match(lines[i]):
        if lines[i].strip().isdigit() and i + 1 < len(lines) and _CUE.match(lines[i + 1]):
            break  # the next SRT cue's number, after a whitespace-only separator
        body.append(lines[i])
        i += 1
    return body, i


def _line_timestamp(line):
    match = _BRACKETED_TS.match(line) or _BARE_TS.match(line)
    if not match:
        return None, ""
    groups = match.groups()
    stamp = _seconds(groups[0:4])
    if stamp is None:
        return None, ""
    return stamp, (groups[4] or "").strip()


def _seconds(parts):
    hours, minutes, secs, fraction = parts
    if minutes is None or secs is None or int(secs) >= 60:
        return None
    if hours is not None and int(minutes) >= 60:
        return None
    value = int(hours or 0) * 3600 + int(minutes) * 60 + int(secs)
    if fraction:
        value += int(fraction) / (10 ** len(fraction))
    return float(value)


def _clean(line):
    # Caption files escape ">>" and "&" as HTML entities.
    return " ".join(html.unescape(_MARKUP.sub(" ", line)).split())


def _speech_tokens(line):
    return _SOUND_TAG.sub(" ", line).split()


def _spread_words(line, start, end):
    """Words of a line spread over [start, end] by length (estimated times)."""
    tokens = _speech_tokens(line)
    if not tokens:
        return []
    span = max(0.0, end - start)
    total = sum(len(token) + 1 for token in tokens)
    words = []
    cursor = start
    for token in tokens:
        duration = span * (len(token) + 1) / total
        words.append({"word": " " + token, "start": round(cursor, 3),
                      "end": round(cursor + duration, 3)})
        cursor += duration
    return words


def _tagged_words(raw, cue_start, cue_end):
    """YouTube VTT: 'so<00:00:00.320><c> today</c>' -> words with their times."""
    chunks = _WORD_TIME.split(_MARKUP.sub("", raw))
    # chunks = [text, time, text, time, text, ...]; the first text starts with the cue
    timed = [(cue_start, chunks[0])]
    for k in range(1, len(chunks) - 1, 2):
        timed.append((_clock(chunks[k]), chunks[k + 1]))
    words = []
    for n, (start, chunk) in enumerate(timed):
        next_start = timed[n + 1][0] if n + 1 < len(timed) else cue_end
        tokens = _speech_tokens(chunk)
        if not tokens:
            continue
        end = min(max(next_start, start), start + len(tokens) * MAX_SECONDS_PER_WORD)
        words.extend(_spread_words(" ".join(tokens), start, end))
    return words


def _clock(stamp):
    parts = [float(p) for p in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _detect_language(text):
    """Same classifier transcribe_backends uses for Parakeet; "en" without it."""
    sample = (text or "").strip()
    if len(sample) < 20:
        return "en"
    try:
        import py3langid
        return py3langid.classify(sample[:4000])[0]
    except Exception:
        return "en"
