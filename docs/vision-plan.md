# Vision for clip selection — implementation plan

**Goal**: let the picker see the video, so it stops choosing moments purely on
what was said. Success is not "vision works"; it is **clips worth posting**,
measured against labels the user gives (Phase 0.2). Everything below is staged
so each phase ships, is measured, and can be abandoned without unpicking the
next one.

Decisions already taken (16-sep-2026): Claude cost is acceptable if the result
earns it; local models are a **support** option only; the dev box is an
i7-13th / RTX 4070 laptop with **8 GB VRAM** and 16 GB RAM.

---

## 1. What exists today, and why none of it can score

| Module | What it extracts | Cadence | When it runs |
|---|---|---|---|
| `layout_picker.py` | one of `none`/`screencast`/`split`, from 12 JPEGs @1024px | **12 frames per whole video** | before selection |
| `active_speaker.py` | frame-difference in a mouth rect, gated by audio RMS | 0.4 s windows | during reframe |
| `reframe_v2` TRACK | MediaPipe face + YOLOv8 person boxes | per frame | **after** selection |
| `scene_detection.py` | TransNetV2 shot cuts | per clip | after selection |
| `thumbnail.py` | face area + Laplacian sharpness per frame | 10 frames | on request |

Two facts kill reuse. `layout_picker` samples **1 frame per 118 s** on a 23.6-min
video — it is a router, not a reader, and cannot see a 1-second laugh. Everything
semantic (`reframe_v2`, `scene_detection`) runs **after** the clip is already
chosen. And `get_viral_clips(transcript_result, video_duration, instructions)`
has no video argument at all. That signature is the change.

What IS reusable: `layout_picker.sample_frames()` (frames to JPEG bytes at a set
width) and `thumbnail.py`'s frame-quality scoring.

## 2. The three constraints that shape the design

**a. Claude takes images, not video.** There is no video content block; frames
are extracted here. Gemini does take video but bills it at ~300 tok/s — an hour
of source is ~1.08M tokens, past the context window before it starts — so frames
are the only viable path for either provider.

**b. Frames are ~4x dearer on Claude than on Gemini.** This repo measured 12
frames @1024px at ~3k Gemini tokens (~250/frame). Claude's image tokens are
roughly `(w*h)/750`, so a 1024x576 frame is ~1,050. **Step 0.1 measures this with
`count_tokens` rather than trusting the formula.** This is why the design below
spends frames on the *shortlist*, not on every window.

**c. 8 GB VRAM rules out a local VLM.** `qwen2.5:7b-instruct` alone is ~5.3 GB
(CLAUDE.md) and reframe already wants the GPU for YOLOv8. So local vision means
**classical CV only** (MediaPipe, OpenCV), and the VLM is always remote. With no
Claude key the pipeline falls back to **today's text-only behaviour**, never to a
local VLM.

## 3. The one rule this repo has already paid for four times

`layout_picker.py`'s docstring: four attempts (Canny, MSER, temporal coverage,
width) asked a model or a heuristic to **measure** something, and none separated
a spreadsheet from a corner scoreboard. Asking for a **decision between closed
options** worked at 92-96%. The same pattern just worked again for the
speech/lyrics gate in `transcript_holes`, where `no_speech_prob` and
`avg_logprob` provably could not separate the populations.

**So: never ask the vision model for a score.** Ask it to choose, rank, or
categorise. This is the strongest constraint on every prompt below.

It also aims vision at a second problem. CLAUDE.md records that pass 1 answers on
a coarse grid (72, 75, 78, 80, 82, 85, 88) with 7-9 tied pairs in the top 10, and
that run-to-run clip agreement is only 3/7-4/7. **A comparative visual re-rank
breaks those ties and adds vision in the same call.**

---

## Phase 0 — the measurement floor (no model spend)

Without this, every later phase is unfalsifiable. Do not skip it.

**0.1 Real image-token cost.** `client.messages.count_tokens` on 1/4/8 frames at
512, 768 and 1024px. Produces the cost table that sets the frame budget. Half a
day. *Output: replaces the estimates in section 6.*

**0.2 A labelled set — the hard prerequisite.** There is no ground truth for
"good clip". Build one from material we already have: the Jake Paul video (3
runs), the democracy video, plus 2 more sources. Pool every candidate window
across runs, sample ~60 moments, and the **user labels each `would post` /
`would not post`** plus a reason from a closed list (`no payoff`, `needs
context`, `nothing to watch`, `boring`). About an hour of the user's clicking; it
is the only thing that turns "better" into a fact. *Output:
`.cache/eval/labels.json`.*

**0.3 Teach the harness to score.** Extend `tools/compare_selection.py` with
precision@k and recall against `labels.json`, plus run-to-run agreement. It
already replays saved answers with no API calls. *Output: a baseline number for
today's text-only picker.*

## Phase 1 — visual re-rank of the shortlist  *(the paid one, build first)*

**Why here.** The shortlist is ~10 windows regardless of video length, so **cost
is flat whether the source is 20 minutes or 2 hours** — the property that makes
this affordable at Claude's frame prices. It is also where the decision is
actually made, and where the ties are.

**Shape.** After `build_shortlist`, before pass 2:

1. 4-6 frames sampled *inside* each shortlisted window (not uniformly across the
   video), at the resolution 0.1 justifies.
2. One Claude call carrying all shortlisted windows: frames plus that window's
   text.
3. Ask for a **closed** answer, never a score:
   - an **ordering** of the windows, best first;
   - one **category** per window from `talking_head` / `visual_action` /
     `screen_or_text` / `nothing_to_watch`;
   - one short reason each.
4. `nothing_to_watch` **demotes**; the ordering breaks pass-1 ties; pass 2 then
   details the reordered shortlist.

**Rules.** Additive only, like `layout_picker.apply()` — vision may reorder and
demote, never invent a window pass 1 never saw (that is Phase 2's job). Any
failure — no key, bad JSON, a 503 outlasting the retry budget — falls back to
today's pass-1 order, exactly as the lyrics gate falls back to today's
transcript.

**Model.** Default `claude-opus-5`; `VISION_MODEL` overrides. Measure
`claude-sonnet-5` against it on 0.2's labels before considering a downgrade —
cost is the user's call, not a default to make quietly. Structured outputs
(`output_config.format`) replace the Pydantic `response_schema` pattern, and
prompt caching goes on the stable rules prefix (the repo re-sends that block on
every batched call today, which is pure waste).

**Measured by**: precision@k vs labels; run-to-run agreement (today 3/7-4/7);
tied pairs in the top 10 (today 7-9); cost per video; seconds added.

**Ship**: `VISION_RERANK=1`, off by default.

## Phase 2 — local dense signal that nominates  *(free, CPU)*

Phase 1 can only reorder what pass 1 surfaced. A purely visual moment — the fire
truck reveal, which the first Gemini run missed entirely — scores low on text and
never reaches the shortlist. Phase 2 fixes that without a model.

A per-second feature track over the whole video, all local:

- **MediaPipe FaceLandmarker blendshapes** — 52 coefficients including
  `mouthSmile`, `jawOpen`, `browInnerUp`. This is the actual expression reader,
  and it runs at video rate on CPU.
- audio RMS peaks and laughter-shaped bursts (the 748 s punch is an RMS peak);
- scene-cut density (TransNetV2, already in the repo);
- motion magnitude; speaker-overlap density (reuse `active_speaker`).

These **nominate** windows into the shortlist that text scored low. They never
decide — a smile spike does not know whether the joke landed; Phase 1 judges the
nominee. That division is what keeps this on the right side of section 3.

**Risk, stated plainly**: this is the shape of the four attempts that failed. The
mitigation is that thresholds here only *add a candidate for a model to judge*,
so a false nomination costs a few frames, not a bad clip.

**Measured by**: labelled-good moments surfaced that text-only dropped; false
nomination rate; CPU seconds added (budget: under 2 min on a 23.6-min source,
against the 114 s the hole repair already costs).

**Ship**: `VISION_NOMINATE=1`, off by default.

## Phase 3 — pre-render visual QC  *(cheap, protects what gets posted)*

Before rendering, check the chosen clip's own frames: is the subject on screen,
is it a static shot of a laptop, is the hook moment visible? A closed
`post` / `flag` answer. Stops a clip that reads well and watches badly from
reaching a social account. Only worth building once Phases 1-2 are measured.

---

## 4. Cross-cutting

- **Everything off by default**, matching `SPLIT_LAYOUT`, `SCREENCAST_LAYOUT`,
  `AUTO_LAYOUT`, `REPAIR_HOLES`. A shadow mode like `AUTO_LAYOUT=shadow` (decide,
  log, change nothing) is the cheapest way to learn what it says about real
  uploads before it can damage a paid clip.
- **`get_viral_clips` gains a `video_path`** and passes it down. That is the
  structural change; keep it optional so every existing test and the
  transcribe-only / agent paths are unaffected.
- **Provider shape.** `llm_provider` duck-types google-genai deliberately, and
  CLAUDE.md is explicit that this is why 754 tests passed unmodified. Claude is a
  third shape. Do **not** bend it into the duck-type — give vision its own small
  client module and leave the text path alone.
- **Agent eyes.** `render_clips` clips come from Claude-in-chat, which cannot see
  the video either. Phase 1's frames should be offered through `get_transcript`
  (or a sibling tool) so the agent gets the same eyes. This is plausibly the
  biggest single win for "post more", and nearly free once Phase 1 exists.

## 5. What could make this not work

1. **No labels, no truth.** If 0.2 does not happen, every later number is
   decoration. This is the main risk.
2. **The input is noisy underneath.** Transcription varies 15.7% of words run to
   run (measured 16-sep). A vision gain smaller than that noise is not
   detectable — so measure on a **fixed** transcript, never a fresh one per run.
3. **Latency.** Frame extraction from a 530 MB source with random seeks is
   unmeasured. If it costs minutes, Phase 1 must pull frames during a pass that
   already decodes the video.
4. **Vision may not beat text on talking-head content.** The Jake Paul video is
   unusually visual. On a podcast, frames may add nothing — in which case the
   honest answer is to gate vision on the layout picker's own verdict and not
   spend on it.

## 6. Cost — estimates, pending Step 0.1

Per video: shortlist of 10 windows x 5 frames = 50 frames. **Flat with video
length**, because the shortlist is capped.

| | input | est. $/video | vs today ($0.009) |
|---|---|---|---|
| Claude Opus 5 | ~60k tok | ~$0.31 | 34x |
| Claude Sonnet 5 | ~60k tok | ~$0.13 | 14x |
| Gemini 3.7-flash (frames ~4x cheaper) | ~20k tok | ~$0.02 | 2x |

A dime to thirty cents per video, against a clip that actually gets posted, is
the right trade at the stated priority. The Gemini row is there only to show the
frame-price gap; the plan defaults to Claude.

## 7. Order of work

1. **0.1** token measurement — half a day, unblocks every cost decision.
2. **0.2** labels — the user's hour, and the gate on everything after.
3. **0.3** harness scoring, and today's baseline number.
4. **Phase 1** behind `VISION_RERANK`, shadow first, then measured on 0.2.
5. **Agent eyes** (section 4) — cheap once Phase 1 exists, closest to "post more".
6. **Phase 2**, then **Phase 3**, each only if the previous one measured well.
