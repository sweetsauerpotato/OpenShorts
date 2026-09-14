"""Compare clip-selection quality across LLM providers on the same transcript.

Phase 2 decision gate: is a local 7B good enough to replace Gemini for
``main.get_viral_clips``? This runs the REAL pipeline function under each
provider and measures the answer, rather than eyeballing one sample.

Read-only with respect to the pipeline: ``get_viral_clips`` touches no disk and
renders no video, so a comparison costs only the model calls. Transcription is
the expensive part and is cached beside the video.

    # inside the backend container. Keep sources in .cache/harness/ (gitignored),
    # NEVER uploads/: the app deletes uploads older than UPLOAD_TTL_SECONDS (6h),
    # and the transcript cache sits beside the source, so both vanish — a 55-min
    # transcript (~15 min of CPU) was lost that way between sessions.
    python tools/compare_selection.py --video .cache/harness/source.webm
    python tools/compare_selection.py --transcript output/<job>/x_metadata.json
    python tools/compare_selection.py --video x.mp4 --providers ollama --runs 2

Metrics, cheapest signal first:

  Tier 1 (mechanical — catches "broken", needs no baseline)
    schema compliance, truncation, hallucinated window ids, out-of-window
    clips, duration-band violations measured BEFORE snap_clip_to_words
    repairs them.
  Tier 2 (agreement — needs two providers)
    pass-1 rank overlap, pass-2 temporal IoU.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# --------------------------------------------------------------------------
# recording proxy — captures what get_viral_clips does internally
# --------------------------------------------------------------------------

class _RecordingModels:
    """Wraps client.models, recording every call for later analysis.

    get_viral_clips snaps clips to word boundaries before returning, which
    repairs band violations. Recording at the transport gives us the model's
    RAW answer, which is what actually measures the model.
    """

    def __init__(self, inner, calls):
        self._inner = inner
        self._calls = calls

    def generate_content(self, **kwargs):
        schema = getattr(kwargs.get("config"), "response_schema", None)
        stage = getattr(schema, "__name__", "?").replace("Response", "").lower()
        t0 = time.time()
        record = {"stage": stage, "ok": False, "parsed_ok": False,
                  "error": None, "seconds": 0.0,
                  "prompt_tokens": 0, "output_tokens": 0, "payload": None}
        # Which windows this call was SHOWN, read from the prompt itself. For the
        # detail stage that is the real shortlist, however main.py built it —
        # so the ranking metrics stay true when the shortlisting logic changes.
        contents = kwargs.get("contents")
        record["window_ids"] = (re.findall(r'"id": "(window_\d+)"', contents)
                                if isinstance(contents, str) else [])
        try:
            resp = self._inner.generate_content(**kwargs)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            record["seconds"] = time.time() - t0
            self._calls.append(record)
            raise
        record["seconds"] = time.time() - t0
        record["ok"] = True
        parsed = getattr(resp, "parsed", None)
        record["parsed_ok"] = parsed is not None
        if parsed is not None:
            record["payload"] = (parsed.model_dump()
                                 if hasattr(parsed, "model_dump") else parsed)
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            record["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
            record["output_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
        self._calls.append(record)
        return resp


class _RecordingClient:
    def __init__(self, inner, calls):
        self._inner = inner
        self.models = _RecordingModels(inner.models, calls)


# --------------------------------------------------------------------------
# transcript sourcing
# --------------------------------------------------------------------------

def load_transcript(path):
    """Accept a job metadata.json, a transcript checkpoint, or a raw dump."""
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    for key in ("transcript",):
        if isinstance(blob.get(key), dict) and blob[key].get("segments"):
            return blob[key]
    if blob.get("segments"):
        return blob
    raise SystemExit(f"No transcript with segments found in {path}")


def transcribe(video_path):
    """Transcribe once and cache — this is the only expensive step here."""
    cache = f"{video_path}.transcript.json"
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        # An empty cache is worse than none: it produces a run that looks
        # successful and measures nothing. Ignore it and transcribe again.
        if cached.get("segments"):
            print(f"   using cached transcript: {cache}")
            return cached
        print(f"   ignoring empty cached transcript ({cache}) — re-transcribing")
    from transcribe_backends import transcribe_media
    print(f"   transcribing {video_path} (cached afterwards)...")
    t0 = time.time()
    transcript = transcribe_media(video_path)
    print(f"   transcribed in {time.time() - t0:.0f}s")
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(transcript, fh, ensure_ascii=False)
    return transcript


def assert_usable(transcript, duration):
    """Refuse sources that cannot exercise clip selection.

    Three ways a source is useless here, each of which otherwise yields a run
    that prints numbers but measures nothing:
      - no speech  -> the pipeline routes to get_visual_clips (Gemini vision),
                      not the path under test
      - too short  -> below the app's own MIN_SOURCE_SECONDS gate
      - one window -> nothing to rank, so pass-1 scoring is a no-op
    """
    from clip_selection import build_transcript_windows, clip_duration_bounds

    n_words = sum(len((s.get("text") or "").split())
                  for s in transcript.get("segments", []))
    problems = []
    if n_words == 0:
        problems.append(
            "the source has no speech (silent or music-only). Such videos route "
            "to get_visual_clips, which is Gemini-vision-only — not the Ollama "
            "path this compares.")
    min_source = float(os.environ.get("MIN_SOURCE_SECONDS", "45"))
    if duration < min_source:
        problems.append(f"only {duration:.0f}s long; the app rejects sources "
                        f"under MIN_SOURCE_SECONDS ({min_source:.0f}s).")
    _, max_secs = clip_duration_bounds()
    windows = build_transcript_windows(transcript, duration,
                                       window_seconds=max(90, int(max_secs * 1.5)))
    if len(windows) < 3:
        problems.append(f"produces only {len(windows)} scoring window(s); pass-1 "
                        f"ranking needs several to mean anything. Use a source of "
                        f"a few minutes or more.")
    if problems:
        print("\nThis source cannot exercise clip selection:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    return windows


def source_duration(video_path):
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True).stdout.strip()
    return float(out)


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------

def clip_text(clip, transcript):
    """Transcript text spoken inside a clip's [start, end]."""
    s, e = float(clip.get("start", 0)), float(clip.get("end", 0))
    return " ".join(seg.get("text", "") for seg in transcript.get("segments", [])
                    if seg.get("end", 0) > s and seg.get("start", 0) < e)


def shortlist_spread(ids, by_id):
    """(overlapping pairs, seconds sent, unique seconds) for a shortlist.

    Windows overlap by ~30 s, so two adjacent windows repeat each other's
    transcript. A ranking that clusters its picks spends detail slots on the
    same moment twice; this makes that visible."""
    spans = sorted((by_id[i]["start"], by_id[i]["end"]) for i in ids if i in by_id)
    pairs = sum(1 for a in range(len(spans)) for b in range(a + 1, len(spans))
                if spans[b][0] < spans[a][1])
    sent = sum(e - s for s, e in spans)
    unique, cur = 0.0, None
    for s, e in spans:
        if cur and s < cur[1]:
            cur[1] = max(cur[1], e)
            continue
        if cur:
            unique += cur[1] - cur[0]
        cur = [s, e]
    if cur:
        unique += cur[1] - cur[0]
    return pairs, round(sent), round(unique)


def score_bands(scores):
    """Counts per band of the scoring prompt's scale: 0-39, 40-69, 70-89, 90-100."""
    bands = [0, 0, 0, 0]
    for s in scores:
        bands[0 if s < 40 else 1 if s < 70 else 2 if s < 90 else 3] += 1
    return bands


def topic_hits(clips, transcript, keywords):
    """(on-topic clips, total): a clip is on topic when its own transcript text
    contains any keyword (case-insensitive substring, so 'crusad' matches
    crusade/crusader/crusades). An objective check that instructions steer."""
    kws = [k.strip().lower() for k in keywords or [] if k.strip()]
    hits = [any(k in clip_text(c, transcript).lower() for k in kws) for c in clips]
    return sum(hits), len(hits)


def run_once(provider, transcript, duration, seed=None, instructions=None, keywords=None):
    import main
    import llm_provider
    from clip_selection import build_transcript_windows, clip_duration_bounds

    os.environ["LLM_PROVIDER"] = provider
    if seed is not None:
        os.environ["OLLAMA_SEED"] = str(seed)
    llm_provider._probe_cache = None

    calls = []
    real_make_client = llm_provider.make_client

    def recording_make_client():
        client, model_name = real_make_client()
        return _RecordingClient(client, calls), model_name

    llm_provider.make_client = recording_make_client
    main.llm_provider.make_client = recording_make_client
    try:
        t0 = time.time()
        try:
            result = main.get_viral_clips(transcript, duration, instructions=instructions)
            error = None
        except Exception as e:
            result, error = None, f"{type(e).__name__}: {e}"
        wall = time.time() - t0
    finally:
        llm_provider.make_client = real_make_client
        main.llm_provider.make_client = real_make_client

    min_secs, max_secs = clip_duration_bounds()
    windows = build_transcript_windows(
        transcript, duration, window_seconds=max(90, int(max_secs * 1.5)))
    valid_ids = {w["id"] for w in windows}
    by_id = {w["id"]: w for w in windows}

    # --- Tier 1, measured on the RAW model output (pre-snap) ---
    scored, raw_clips, hallucinated = [], [], []
    for rec in calls:
        payload = rec.get("payload") or {}
        if rec["stage"] == "score":
            for w in payload.get("windows", []) or []:
                scored.append(w)
                if w.get("id") not in valid_ids:
                    hallucinated.append(w.get("id"))
        elif rec["stage"] == "detail":
            raw_clips.extend(payload.get("shorts", []) or [])

    band_violations, out_of_window, echoed = [], [], []
    for c in raw_clips:
        try:
            start, end = float(c.get("start", 0)), float(c.get("end", 0))
        except (TypeError, ValueError):
            continue
        dur = end - start
        if dur < min_secs - 0.01 or dur > max_secs + 0.01:
            band_violations.append(round(dur, 1))
        w = by_id.get(c.get("source_window_id"))
        if w and not (w["start"] - 0.01 <= start and end <= w["end"] + 0.01):
            out_of_window.append(c.get("source_window_id"))
        # The failure that sank qwen2.5:7b: returning the candidate window's
        # own bounds as the clip. Snapping later clamps it to exactly max_secs,
        # so it is invisible in final_clips — only the raw answer shows it.
        if w and abs(start - w["start"]) < 1.0 and abs(end - w["end"]) < 1.0:
            echoed.append(c.get("source_window_id"))

    # --- ranking quality: coverage, resolution, ties in the real shortlist ---
    best_score = {}
    for w in scored:
        wid = w.get("id")
        if wid in valid_ids:
            best_score[wid] = max(best_score.get(wid, -1), w.get("score", 0))
    shortlist_ids = []
    for rec in calls:
        if rec["stage"] == "detail":
            for wid in rec.get("window_ids") or []:
                if wid not in shortlist_ids:
                    shortlist_ids.append(wid)
    shortlist_scores = [best_score.get(wid) for wid in shortlist_ids]
    tie_counts = Counter(s for s in shortlist_scores if s is not None)
    shortlist_in_ties = sum(n for n in tie_counts.values() if n > 1)
    shortlist_tied_pairs = sum(n * (n - 1) // 2 for n in tie_counts.values())
    overlap_pairs, sent_seconds, unique_seconds = shortlist_spread(shortlist_ids, by_id)

    stage_tokens = {}
    for r in calls:
        tokens = stage_tokens.setdefault(r["stage"], [0, 0])
        tokens[0] += r["prompt_tokens"]
        tokens[1] += r["output_tokens"]

    final = (result or {}).get("shorts", [])
    playbook_hooks = [c.get("viral_hook_text") for c in final
                      if _copies_playbook(c.get("viral_hook_text") or "")]
    language = str(transcript.get("language") or "")
    foreign = [text for c in final for text in _copy_fields(c)
               if _foreign_script(text, language)]

    return {
        "provider": provider,
        "seed": seed,
        "instructions": instructions,
        "topic_hits": topic_hits(final, transcript, keywords) if keywords else None,
        "error": error,
        "wall_seconds": round(wall, 1),
        "n_windows": len(windows),
        "calls": len(calls),
        "failed_calls": sum(1 for r in calls if not r["ok"]),
        # Over calls that RETURNED: a 503 retry says nothing about the schema,
        # and counting it (14-sep-2026, Gemini under load) read as 0.615.
        "schema_compliance": (
            round(sum(1 for r in calls if r["parsed_ok"]) / sum(1 for r in calls if r["ok"]), 3)
            if any(r["ok"] for r in calls) else None),
        "prompt_tokens": sum(r["prompt_tokens"] for r in calls),
        "output_tokens": sum(r["output_tokens"] for r in calls),
        "scored": scored,
        "hallucinated_ids": hallucinated,
        "raw_clips": raw_clips,
        "band_violations": band_violations,
        "band_bounds": [min_secs, max_secs],
        "out_of_window": out_of_window,
        "echoed": echoed,
        "windows_scored": len(best_score),
        "distinct_scores": len(set(best_score.values())),
        "shortlist_ids": shortlist_ids,
        "shortlist_scores": shortlist_scores,
        "shortlist_in_ties": shortlist_in_ties,
        "shortlist_tied_pairs": shortlist_tied_pairs,
        "shortlist_overlap_pairs": overlap_pairs,
        "shortlist_sent_seconds": sent_seconds,
        "shortlist_unique_seconds": unique_seconds,
        "score_bands": score_bands(best_score.values()),
        "stage_tokens": stage_tokens,
        "playbook_hooks": playbook_hooks,
        "foreign_script": foreign,
        "final_clips": final,
        "cost": (result or {}).get("cost_analysis"),
        "call_seconds": [round(r["seconds"], 1) for r in calls],
    }


# --------------------------------------------------------------------------
# copy checks
# --------------------------------------------------------------------------

def _playbook():
    """(examples, labels) read from the LIVE detail prompt, so this check can
    never drift from the playbook the model is actually shown."""
    import re
    import gemini_worker
    tpl = gemini_worker.DETAIL_PROMPT_TEMPLATE
    section = tpl.split("HOOK PLAYBOOK", 1)[-1].split("(These are", 1)[0]
    examples = re.findall(r'"([^"]+)"', section)
    labels = [m.strip().lower() for m in re.findall(r"^- ([^:\n]+):", section, re.M)]
    return examples, labels


def _words(text):
    import re
    # Crude stem (drop a trailing "s") so "gets"/"get" count as the same word.
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in re.findall(r"[a-z0-9%']+", text.lower())}


def _copies_playbook(hook):
    """Verbatim or near-verbatim reuse of an example, or a leaked label.

    Exact matching missed 'Why everyone gets this wrong.' against the example
    'Why does everyone get this wrong?' on the first run, so near-copies are
    judged by word overlap. A leaked label ('Story loop: ...') counts too —
    the 3rd qwen2.5 run produced exactly that. 'POV:' alone is a legitimate use
    of the pattern, so that label's first word is not flagged on its own.
    """
    examples, labels = _playbook()
    hw = _words(hook)
    if not hw:
        return False
    for ex in examples:
        ew = _words(ex)
        if ew and len(hw & ew) / len(hw | ew) >= 0.6:
            return True
    low = hook.strip().lower()
    for label in labels:
        for part in (p.strip() for p in label.split("/")):
            if part and part != "pov" and low.startswith(part + ":"):
                return True
    return False


def _copy_fields(clip):
    return [str(clip.get(k) or "") for k in (
        "viral_hook_text", "video_title_for_youtube_short",
        "video_description_for_tiktok", "video_description_for_instagram")]


def _foreign_script(text, language):
    """CJK/Kana/Hangul in copy for a transcript that is none of those.

    qwen2.5 leaked 'Herod's huge扩建' into an English hook. Narrow on purpose:
    it only flags scripts that cannot belong to the transcript's language.
    """
    if language[:2] in ("zh", "ja", "ko"):
        return False
    return any("一" <= ch <= "鿿" or "぀" <= ch <= "ヿ"
               or "가" <= ch <= "힯" for ch in text)


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def iou(a, b):
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def rank_overlap(run_a, run_b, top_n=5):
    def top(run):
        rows = sorted(run["scored"], key=lambda w: w.get("score", 0), reverse=True)
        seen, out = set(), []
        for r in rows:
            wid = r.get("id")
            if wid and wid not in seen:
                seen.add(wid)
                out.append(wid)
            if len(out) >= top_n:
                break
        return out
    a, b = top(run_a), top(run_b)
    if not a or not b:
        return None, a, b
    return len(set(a) & set(b)) / len(set(a)), a, b


def report(run):
    print(f"\n--- {run['provider']}" + (f" (seed={run['seed']})" if run['seed'] else "") + " ---")
    if run["error"]:
        print(f"  FAILED: {run['error']}")
    lo, hi = run["band_bounds"]
    print(f"  wall {run['wall_seconds']}s over {run['calls']} call(s) "
          f"({run['n_windows']} windows) | tokens in/out "
          f"{run['prompt_tokens']}/{run['output_tokens']}")
    if run["call_seconds"]:
        print(f"  per-call seconds: {run['call_seconds']} "
              f"(median {statistics.median(run['call_seconds'])})")
    if run.get("stage_tokens"):
        print("  tokens in/out by stage: " + ", ".join(
            f"{stage} {t[0]}/{t[1]}" for stage, t in run["stage_tokens"].items()))
    print(f"  TIER 1")
    print(f"    schema compliance : {run['schema_compliance']}   (1.0 required)")
    print(f"    failed calls      : {run['failed_calls']}")
    print(f"    hallucinated ids  : {len(run['hallucinated_ids'])} {run['hallucinated_ids'][:5]}"
          f"   <- silently degrades to 'first N windows'")
    print(f"    band violations   : {len(run['band_violations'])}/{len(run['raw_clips'])} "
          f"raw clips outside {lo:g}-{hi:g}s {run['band_violations'][:6]}")
    print(f"    out-of-window     : {len(run['out_of_window'])}")
    print(f"    WHOLE-WINDOW ECHO : {len(run['echoed'])}/{len(run['raw_clips'])} "
          f"{run['echoed'][:6]}   <- returns the window instead of choosing")
    print(f"    playbook hooks    : {len(run['playbook_hooks'])} {run['playbook_hooks'][:4]}")
    print(f"    foreign script    : {len(run['foreign_script'])} {run['foreign_script'][:3]}")
    print(f"  RANKING")
    print(f"    windows scored    : {run['windows_scored']}/{run['n_windows']}"
          f"   <- unscored windows can never be picked")
    print(f"    distinct scores   : {run['distinct_scores']}")
    print(f"    score bands       : {run['score_bands']}   <- windows scored 0-39 / 40-69 / 70-89 / 90-100")
    print(f"    shortlist (detail): {len(run['shortlist_ids'])} windows, pass-1 scores "
          f"{run['shortlist_scores']}")
    print(f"    shortlist ties    : {run['shortlist_in_ties']} windows in ties, "
          f"{run['shortlist_tied_pairs']} tied pairs   <- ties are broken by transcript order")
    print(f"    shortlist spread  : {run['shortlist_overlap_pairs']} overlapping pairs, "
          f"{run['shortlist_unique_seconds']}s unique of {run['shortlist_sent_seconds']}s sent")
    if run.get("topic_hits"):
        on, total = run["topic_hits"]
        print(f"  INSTRUCTIONS : {run['instructions']!r}")
        print(f"    on-topic clips    : {on}/{total}   <- transcript inside the clip mentions a keyword")
    print(f"    final clips       : {len(run['final_clips'])} (after word-snapping)")
    for c in run["final_clips"]:
        d = c.get("end", 0) - c.get("start", 0)
        print(f"      {c.get('start'):.1f}-{c.get('end'):.1f} ({d:.0f}s) "
              f"score={c.get('predicted_score')} hook={c.get('viral_hook_text')!r}")
        print(f"        title: {c.get('video_title_for_youtube_short')!r}")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="source video (transcribed once, then cached)")
    src.add_argument("--transcript", help="job metadata.json / checkpoint / transcript dump")
    ap.add_argument("--duration", type=float, help="source duration; inferred when omitted")
    ap.add_argument("--providers", default="ollama,gemini",
                    help="comma separated (default: ollama,gemini)")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat each provider N times to measure self-agreement")
    ap.add_argument("--out", help="write the full record to this JSON file")
    ap.add_argument("--instructions", help="creator clip instructions to steer selection")
    ap.add_argument("--topic-keywords",
                    help="comma separated; reports how many final clips mention one")
    args = ap.parse_args()
    from clip_selection import normalize_clip_instructions
    instructions = normalize_clip_instructions(args.instructions)
    keywords = [k for k in (args.topic_keywords or "").split(",") if k.strip()]

    print("=" * 74)
    print("CLIP SELECTION COMPARISON")
    print("=" * 74)

    if args.video:
        transcript = transcribe(args.video)
        duration = args.duration or source_duration(args.video)
    else:
        transcript = load_transcript(args.transcript)
        duration = args.duration
        if not duration:
            ends = [s.get("end", 0) for s in transcript.get("segments", [])]
            duration = max(ends) if ends else 0
    n_words = sum(len((s.get("text") or "").split())
                  for s in transcript.get("segments", []))
    print(f"source: {duration:.0f}s, {len(transcript.get('segments', []))} segments, "
          f"{n_words} words, language={transcript.get('language')}")
    windows = assert_usable(transcript, duration)
    print(f"        {len(windows)} scoring windows -> "
          f"{-(-len(windows) // 8)} pass-1 call(s) per provider")

    runs = []
    for provider in [p.strip() for p in args.providers.split(",") if p.strip()]:
        for i in range(args.runs):
            seed = (1000 + i) if (provider == "ollama" and args.runs > 1) else None
            print(f"\n>>> running {provider} ({i + 1}/{args.runs})...")
            run = run_once(provider, transcript, duration, seed=seed,
                           instructions=instructions, keywords=keywords)
            runs.append(run)
            report(run)

    # --- Tier 2: agreement -------------------------------------------------
    by_provider = {}
    for r in runs:
        by_provider.setdefault(r["provider"], []).append(r)

    if len(by_provider) > 1:
        names = list(by_provider)
        a, b = by_provider[names[0]][0], by_provider[names[1]][0]
        print("\n" + "=" * 74)
        print(f"TIER 2 — AGREEMENT: {names[0]} vs {names[1]}")
        overlap, top_a, top_b = rank_overlap(a, b)
        print(f"  pass-1 top-5 overlap : {overlap}")
        print(f"    {names[0]}: {top_a}")
        print(f"    {names[1]}: {top_b}")
        pairs = []
        for ca in a["final_clips"]:
            best = max((iou((ca["start"], ca["end"]), (cb["start"], cb["end"]))
                        for cb in b["final_clips"]), default=0.0)
            pairs.append(round(best, 2))
        print(f"  pass-2 best IoU per {names[0]} clip: {pairs}")
        if pairs:
            print(f"    >=0.5 overlap: {sum(1 for p in pairs if p >= 0.5)}/{len(pairs)}")

    # Self-agreement: if a provider disagrees with ITSELF more than with the
    # other, that is a temperature/context bug, not a capability limit.
    for provider, group in by_provider.items():
        if len(group) > 1:
            a, b = group[0], group[1]
            overlap, ta, tb = rank_overlap(a, b)
            print(f"\n  SELF-AGREEMENT {provider} (run1 vs run2)")
            print(f"    pass-1 top-5 overlap : {overlap}")
            print(f"      {ta}\n      {tb}")
            sa, sb = set(a["shortlist_ids"]), set(b["shortlist_ids"])
            if sa and sb:
                print(f"    shortlist overlap    : {len(sa & sb)}/{max(len(sa), len(sb))} "
                      f"windows sent to the detail pass in both runs")
            pairs = [max((iou((ca["start"], ca["end"]), (cb["start"], cb["end"]))
                          for cb in b["final_clips"]), default=0.0)
                     for ca in a["final_clips"]]
            if pairs:
                print(f"    final clips          : {sum(1 for p in pairs if p >= 0.5)}/{len(pairs)} "
                      f"of run1's clips reappear in run2 (IoU >= 0.5)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(runs, fh, ensure_ascii=False, indent=2)
        print(f"\nfull record -> {args.out}")


if __name__ == "__main__":
    main_cli()
