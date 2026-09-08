"""Compare clip-selection quality across LLM providers on the same transcript.

Phase 2 decision gate: is a local 7B good enough to replace Gemini for
``main.get_viral_clips``? This runs the REAL pipeline function under each
provider and measures the answer, rather than eyeballing one sample.

Read-only with respect to the pipeline: ``get_viral_clips`` touches no disk and
renders no video, so a comparison costs only the model calls. Transcription is
the expensive part and is cached beside the video.

    # inside the backend container
    python tools/compare_selection.py --video demo.mp4
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
import statistics
import sys
import time

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

def run_once(provider, transcript, duration, seed=None):
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
            result = main.get_viral_clips(transcript, duration)
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

    band_violations, out_of_window = [], []
    for c in raw_clips:
        try:
            dur = float(c.get("end", 0)) - float(c.get("start", 0))
        except (TypeError, ValueError):
            continue
        if dur < min_secs - 0.01 or dur > max_secs + 0.01:
            band_violations.append(round(dur, 1))
        w = by_id.get(c.get("source_window_id"))
        if w and not (w["start"] - 0.01 <= float(c.get("start", 0))
                      and float(c.get("end", 0)) <= w["end"] + 0.01):
            out_of_window.append(c.get("source_window_id"))

    return {
        "provider": provider,
        "seed": seed,
        "error": error,
        "wall_seconds": round(wall, 1),
        "n_windows": len(windows),
        "calls": len(calls),
        "failed_calls": sum(1 for r in calls if not r["ok"]),
        "schema_compliance": (
            round(sum(1 for r in calls if r["parsed_ok"]) / len(calls), 3)
            if calls else None),
        "prompt_tokens": sum(r["prompt_tokens"] for r in calls),
        "output_tokens": sum(r["output_tokens"] for r in calls),
        "scored": scored,
        "hallucinated_ids": hallucinated,
        "raw_clips": raw_clips,
        "band_violations": band_violations,
        "band_bounds": [min_secs, max_secs],
        "out_of_window": out_of_window,
        "final_clips": (result or {}).get("shorts", []),
        "cost": (result or {}).get("cost_analysis"),
        "call_seconds": [round(r["seconds"], 1) for r in calls],
    }


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
    print(f"  TIER 1")
    print(f"    schema compliance : {run['schema_compliance']}   (1.0 required)")
    print(f"    failed calls      : {run['failed_calls']}")
    print(f"    hallucinated ids  : {len(run['hallucinated_ids'])} {run['hallucinated_ids'][:5]}"
          f"   <- silently degrades to 'first N windows'")
    print(f"    band violations   : {len(run['band_violations'])}/{len(run['raw_clips'])} "
          f"raw clips outside {lo:g}-{hi:g}s {run['band_violations'][:6]}")
    print(f"    out-of-window     : {len(run['out_of_window'])}")
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
    args = ap.parse_args()

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
            run = run_once(provider, transcript, duration, seed=seed)
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
            overlap, ta, tb = rank_overlap(group[0], group[1])
            print(f"\n  SELF-AGREEMENT {provider} (run1 vs run2) top-5 overlap: {overlap}")
            print(f"    {ta}\n    {tb}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(runs, fh, ensure_ascii=False, indent=2)
        print(f"\nfull record -> {args.out}")


if __name__ == "__main__":
    main_cli()
