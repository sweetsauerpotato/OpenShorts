"""What the user thought of a clip, and the context it was produced in.

The point of this store is not the thumbs. It is that **no change to clip
selection can be called better without it.** The two days before it was written
made that concrete: the picker's own numbers cannot be trusted as a quality
signal (pass 1 answers on a coarse grid, 72/75/78/80/82/85/88, with 7-9 tied
pairs in the top 10), run-to-run clip agreement is only 3/7-4/7, and the
transcript underneath varies 15.7% of its words between two runs of the SAME
file. A vision feature that improves things by less than that noise is
undetectable without labels to measure against.

So every row records **the context that produced the clip**, not just the
verdict. Without `mode`, `provider` and `niche` on the row, the data can never
answer "did vision help" — it can only say that some clips were liked.

``predicted_score`` is stored next to the human verdict on purpose: it makes
"does the model's own score predict what the user actually wants" a question
this file can answer on its own, with no extra work.

Reasons are a CLOSED list, never free text, for the same reason
``DELETION_REASONS`` is: a free-text field on a row designed to outlive the job
it describes collects things nobody planned to store, and it cannot be counted.

Standard library only, like the other helpers here; the API and MCP layers do
the I/O. Rows are append-only JSONL: the latest row for a (job, clip) wins, so
re-rating is just another append and a crash mid-write loses one line, not the
file.
"""

import json
import os
import time

# Thumbs, plus the way back out. "unrated" is a real row rather than a delete
# because the store is append-only: clicking a thumb off is another append that
# the latest-wins rule turns into "no opinion", and the history of what was
# thought when stays intact. Counting excludes it everywhere.
VERDICTS = ("good", "bad", "unrated")
UNRATED = "unrated"

# Why a clip was not worth posting. Closed, countable, and aligned with the
# failure modes clip_rules.md already names, so a count here points straight at
# the rule to rewrite.
REASONS = (
    "no_payoff",         # the setup never lands
    "needs_context",     # makes no sense to someone who saw nothing else
    "nothing_to_watch",  # reads fine, watches badly — the vision case
    "boring",            # nothing wrong with it, nobody would stop
    "wrong_moment",      # the right topic, cut in the wrong place
    "bad_cut",           # starts or ends in the wrong place
)

STORE_DIRNAME = "verdicts"
STORE_FILENAME = "verdicts.jsonl"


class VerdictError(ValueError):
    """Invalid verdict: safe to show as a 400."""


def normalize(verdict, reason=None):
    """Validate a verdict and its reason, or raise VerdictError."""
    v = str(verdict or "").strip().lower()
    if v not in VERDICTS:
        raise VerdictError(f"verdict must be one of {', '.join(VERDICTS)}")
    if v == UNRATED:
        return v, None          # clearing carries no reason
    r = str(reason or "").strip().lower() or None
    if r is not None and r not in REASONS:
        raise VerdictError(f"reason must be one of {', '.join(REASONS)}")
    return v, r


def build_row(job_id, clip_index, verdict, reason=None, clip=None,
              context=None, user=None, now=None):
    """One verdict row: the judgement, the clip, and how the clip was made.

    ``context`` carries the provenance the harness needs later — ``mode``,
    ``provider``, ``niche``, ``model``, the source and its duration. Unknown
    keys are kept as given so a field added by a later phase (a vision mode, a
    niche) lands in the row without changing this signature.
    """
    v, r = normalize(verdict, reason)
    clip = clip or {}
    row = {
        "job_id": str(job_id),
        "clip_index": int(clip_index),
        "verdict": v,
        "reason": r,
        "at": float(now if now is not None else time.time()),
        "user": str(user) if user else None,
        # The clip as it was served, so a row still means something after the
        # job directory is swept (24 h) and the video is gone.
        "clip": {
            "title": (clip.get("video_title_for_youtube_short")
                      or clip.get("title") or ""),
            "hook": clip.get("viral_hook_text") or clip.get("hook") or "",
            "start": clip.get("start"),
            "end": clip.get("end"),
            "predicted_score": clip.get("predicted_score", clip.get("score")),
            "selected_by": clip.get("selected_by") or "ai",
        },
        # How it was produced. Absent fields stay null rather than being
        # guessed: a null mode is honest, a wrong one poisons the measurement.
        "context": dict(context or {}),
    }
    return row


def store_path(output_dir):
    return os.path.join(output_dir, STORE_DIRNAME, STORE_FILENAME)


def append_row(output_dir, row):
    """Append one row, creating the store directory if needed."""
    path = store_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_rows(output_dir):
    """Every row ever written, oldest first. A damaged line is skipped."""
    path = store_path(output_dir)
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("job_id") is not None:
                    rows.append(row)
    except OSError:
        return []
    return rows


def latest(rows):
    """The current verdict per (job, clip): the last row written wins."""
    out = {}
    for row in rows:
        try:
            key = (str(row["job_id"]), int(row["clip_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        prev = out.get(key)
        if prev is None or float(row.get("at") or 0) >= float(prev.get("at") or 0):
            out[key] = row
    return out


def for_job(rows, job_id):
    """Current verdicts for one job, by clip index. Cleared clips are absent."""
    return {k[1]: v for k, v in latest(rows).items()
            if k[0] == str(job_id) and v.get("verdict") != UNRATED}


def summarise(rows):
    """Counts that say whether there is enough data to measure anything yet.

    ``score_split`` is the one to watch: the mean predicted_score of clips the
    user liked against those they did not. If those two numbers do not separate,
    the picker's own score is not measuring what the user wants, and no amount
    of re-ranking on it will help.
    """
    # A cleared clip is not an opinion, so it counts for nothing here.
    current = [r for r in latest(rows).values() if r.get("verdict") != UNRATED]
    by_verdict, by_reason, by_mode = {}, {}, {}
    liked, disliked = [], []
    for row in current:
        v = row.get("verdict")
        by_verdict[v] = by_verdict.get(v, 0) + 1
        if row.get("reason"):
            by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        mode = (row.get("context") or {}).get("mode") or "unknown"
        by_mode[mode] = by_mode.get(mode, 0) + 1
        score = (row.get("clip") or {}).get("predicted_score")
        if isinstance(score, (int, float)):
            (liked if v == "good" else disliked).append(float(score))

    def mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    return {
        "rated": len(current),
        "by_verdict": by_verdict,
        "by_reason": by_reason,
        "by_mode": by_mode,
        "score_split": {"good": mean(liked), "bad": mean(disliked),
                        "n_good": len(liked), "n_bad": len(disliked)},
    }


def drop_user(rows, user):
    """Every row except that user's — for account erasure (GDPR art. 17).

    Verdicts are the user's own words about their own clips, so they go when the
    account does. Rows with no user (self-host, where there are no accounts)
    are kept.
    """
    uid = str(user)
    return [r for r in rows if str(r.get("user") or "") != uid]


def rewrite(output_dir, rows):
    """Replace the store with ``rows``. Only erasure should need this."""
    path = store_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path
