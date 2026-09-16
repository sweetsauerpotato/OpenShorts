# Vision for clip selection — implementation plan

**Goal**: let the picker see the video, so it stops choosing moments purely on
what was said. Success is not "vision works"; it is **clips worth posting**,
measured against verdicts the user gives. Every phase ships behind a flag, is
measured, and can be abandoned without unpicking the next one.

Decisions taken (16-sep-2026): Claude cost is acceptable if the result earns it;
local models are a **support** option only; the dev box is an i7-13th / RTX 4070
laptop with **8 GB VRAM** and 16 GB RAM; the system is expected to improve over
time from the user's own judging and from social engagement data.

---

## 0. Corrections to the assumptions behind this round

**a. A 1-hour cap does not make full-video Gemini work.** At the repo's measured
~300 tok/s, an hour of source is **1,080,000 tokens — past a 1M context window
before the prompt is added**. The real ceiling is ~45 min, and ~30 min is the
comfortable one. So a 1-hour split warning is right for the *pipeline in
general*, but "full video vision" has to be gated at a **different, lower
threshold** than the warning, or offered only for sources under ~30 min.

| length | video tokens | flash-lite | 3.7-flash | fits 1M? |
|---|---|---|---|---|
| 10 min | 180k | $0.045 | $0.135 | yes |
| 20 min | 360k | $0.090 | $0.270 | yes |
| 30 min | 540k | $0.135 | $0.405 | yes |
| 45 min | 810k | $0.203 | $0.608 | tight |
| 60 min | 1.08M | — | — | **no** |

**b. This repo has already measured frames against whole-video, and frames won.**
`layout_picker.py`: whole video scored 94% / 92% / 96%, twelve frames @1024px
scored 92%, and the docstring's conclusion is that the difference "falls inside
the variance the video mode already has", at **2.2 s per clip instead of ~15 s**.
That was a *static* question (is there a spreadsheet on screen). For a *temporal*
question — where is the laugh, when does the punch land — twelve frames over an
hour is one frame per 5 minutes and obviously cannot answer it, but the fix is
**more frames at the right moments**, not a 1M-token upload. Full-video mode
stays in the plan as a measured experiment, not as the default.

**c. Engagement data will not train anything, and should not be planned as if it
will.** It is sparse (a handful of clips), confounded (posting time, account
size, thumbnail, hook, platform), and delayed by days. Neither it nor your
verdicts fine-tune a model. What they actually buy is: (1) **measurement**, so a
change can be proven better; (2) **few-shot examples** in the prompt, which is
real and cheap; (3) **niche criteria that get rewritten from evidence**. Your own
verdicts are worth far more per item than engagement numbers for a long time —
expect engagement to become a useful tiebreak only after dozens of posts.

**d. Niche criteria belong in presets, not only in the free-text box.** Free text
is per-video and disappears; a preset is named, versioned, reusable and
improvable from the verdict data. The box stays, and stacks on top.

---

## 1. Modes — what the user picks

Two independent axes, not one switch:

| axis | values | notes |
|---|---|---|
| **provider** | `gemini` (default) · `claude` · `local` | extends today's `LLM_PROVIDER`; `local` stays support-only |
| **vision** | `off` (classical, today) · `frames` · `full` | `full` is Gemini-only and length-capped (§0a) |

So `classical` and `vision` exist for **both** providers, as asked. The
combinations that matter:

- `gemini + off` — today's pipeline, unchanged, still the cheapest.
- `claude + off` — text-only selection on Claude. Worth having on its own: no
  frames, no vision cost, just better judgement on the same transcript.
- `gemini + frames` / `claude + frames` — Phase 1.
- `gemini + full` — the experiment in §0b, offered only under ~30 min.

Every mode must degrade to `gemini + off` on any failure. No key, bad JSON, a 503
outlasting the retry budget — fall back, never fail the job. This is exactly how
the lyrics gate in `transcript_holes` behaves.

**UI**: a "how to choose clips" control in advanced options — Classic / Vision —
plus a provider select, remembered in `localStorage` like `os_layout` already is.
Grey out `full` above the length cap rather than failing after submit.

## 2. Niches — telling the model what counts as good *here*

Today there are two layers: `clip_rules.md` (global, user-editable, read on every
`get_transcript`) and `clip_instructions` (per-video free text). A niche is the
missing **third layer between them**: named, reusable, versioned.

Ship as `niches/<name>.md`, selected by a `niche` parameter, injected through the
existing `with_clip_instructions` mechanism so blank stays byte-identical and
user braces stay inert. Two to start:

**`tech_podcast`** — intellectual / interview content. Look for: a claim that
contradicts consensus, a concrete number or mechanism, a confident prediction, an
admission or disagreement between hosts, a clean explanation that stands alone.
Skip: throat-clearing, agreement chains, definitions without a payoff, anything
needing an earlier segment.

**`creator_chaos`** — influencer / streamer (Jake Paul, Speed). Look for: a
reaction, a stunt, a dare, an argument, an absurd object or situation, genuine
surprise, physical comedy. Skip: sponsor reads, subscribe pitches, the "later in
the video" tease, travel filler. Note that the payoff is often **non-verbal** —
which is precisely why this niche is where vision should pay off first, and the
right one to measure Phase 1 on.

**Precedent that this works**: CLAUDE.md records that a 56-character instruction
moved on-topic clips from 1 of 6 to 3 of 3 in both runs, and made the job
*cheaper* (1.51 → 1.46 cents) because fewer clips meant less pass-2 output.
Niche text costs ~100-200 input tokens and should pay for itself the same way.

**Where it goes**: pass 1 **and** pass 2 **and** `get_transcript`'s rules payload,
so the agent path gets the same niche. Pass 1 matters most — it is where most
windows are eliminated.

## 3. The verdict store — the thing that makes everything else measurable

This is now a **product feature**, not a one-off eval, because the user will keep
judging clips.

- **Capture**: thumbs up / down on each clip in the dashboard and in
  `list_clips`, plus a reason from a closed list (`no payoff`, `needs context`,
  `nothing to watch`, `boring`, `wrong moment`). Closed list, never free text —
  the same reasoning as `DELETION_REASONS`.
- **Store**: `verdicts.json` per job plus an append-only roll-up, keyed by
  (job, clip index, source, niche, mode, provider). Record the **mode that
  produced it**, or the data cannot answer "did vision help".
- **Engagement**: when a clip is posted through Upload-Post, store the returned
  ids and pull metrics later into the same row. Treat as a weak late signal per
  §0c.
- **Use**: (1) the harness scores precision@k against it; (2) once there are
  enough, the best-rated clips become **few-shot examples** in the niche prompt;
  (3) the reasons tell us which niche rule to rewrite.

**Build this first.** Every clip judged from today is data; every day it does not
exist is data lost.

## 4. What exists today, and why none of it can score

| Module | What it extracts | Cadence | When it runs |
|---|---|---|---|
| `layout_picker.py` | `none`/`screencast`/`split` from 12 JPEGs @1024px | 12 frames per whole video | before selection |
| `active_speaker.py` | frame-difference in a mouth rect, audio-gated | 0.4 s windows | during reframe |
| `reframe_v2` TRACK | MediaPipe face + YOLOv8 person boxes | per frame | **after** selection |
| `scene_detection.py` | TransNetV2 shot cuts | per clip | after selection |
| `thumbnail.py` | face area + Laplacian sharpness | 10 frames | on request |

Everything semantic runs **after** the clip is chosen, and
`get_viral_clips(transcript_result, video_duration, instructions)` has no video
argument. That signature is the structural change. Reusable:
`layout_picker.sample_frames()` and `thumbnail.py`'s frame scoring.

## 5. Constraints

- **Claude takes images, not video.** Frames are extracted here. Gemini takes
  video, at the cost in §0a.
- **A frame costs ~4x more on Claude** (~1,050 tok vs Gemini's measured ~250
  @1024px). Step 0.1 measures this with `count_tokens` instead of trusting the
  formula. This is why frames go on the **shortlist**, not every window — which
  also makes cost **flat with video length**, since the shortlist is capped at 10.
- **8 GB VRAM rules out a local VLM.** `qwen2.5:7b` alone is ~5.3 GB and reframe
  wants the GPU. Local vision is classical CV only.
- **Length guard**: over 1 hour, return the existing `needs_confirmation` shape
  (the pattern `force_low_quality` already uses) advising a manual split, with a
  `force_long` override. Separately, cap `vision=full` at ~30 min per §0a.

## 6. The rule this repo has paid for four times

`layout_picker.py` records four attempts (Canny, MSER, temporal coverage, width)
that asked for a **measurement** and all failed the same way. A **closed choice**
got 92-96%. The lyrics gate repeated the lesson: `no_speech_prob` and
`avg_logprob` could not separate music from speech at all.

**So vision is never asked for a score.** It ranks, chooses, or categorises. This
also attacks a second problem: pass 1 answers on a coarse grid with 7-9 tied
pairs in the top 10 and run-to-run agreement of only 3/7-4/7. A comparative
re-rank breaks those ties and adds vision in the same call.

---

## 7. Phases

**Phase A — verdict store (§3).** No model spend. Capture, store, roll up. Starts
collecting immediately.

**Phase B — niches (§2).** Text-only, cheap, improves output now. Measurable
against Phase A's verdicts once there are ~30 of them.

**Phase C — length guard (§5).** Small, reuses `needs_confirmation`.

**Phase D — 0.1 token measurement.** `count_tokens` on 1/4/8 frames at 512/768/
1024px. Replaces the estimates in §8.

**Phase E — modes plumbing (§1).** `provider` and `vision` through `/api/process`,
MCP, and the dashboard, with `vision=off` behaviour byte-identical to today.

**Phase F — frames re-rank.** After `build_shortlist`, before pass 2: 4-6 frames
sampled *inside* each shortlisted window; one call; closed output = an **ordering**
plus a **category** per window (`talking_head` / `visual_action` /
`screen_or_text` / `nothing_to_watch`); `nothing_to_watch` demotes. Additive only
— it may reorder and demote, never invent a window pass 1 never saw. Default
model `claude-opus-5`, `VISION_MODEL` overrides; measure Sonnet 5 against it
before any downgrade, since cost is the user's call. Shadow mode first.

**Phase G — agent eyes.** Offer the same frames through `get_transcript` so the
in-chat agent can see the video. Nearly free once F exists, and the agent path is
the one whose clips the user rated best.

**Phase H — full-video Gemini experiment (§0b).** Under ~30 min only, measured
against F on the same sources. Likely to lose on cost and latency; worth one
honest measurement.

**Phase I — local nomination.** MediaPipe FaceLandmarker blendshapes
(`mouthSmile`, `jawOpen`, `browInnerUp`), audio RMS peaks, scene-cut density,
motion, speaker overlap — free, CPU, whole video. These **nominate** windows that
text scored low; they never decide. Risk stated plainly: this is the shape of the
four attempts that failed, and the mitigation is that a threshold here only adds
a candidate for a model to judge.

**Phase J — pre-render visual QC.** Closed `post` / `flag` on the chosen clip's
own frames, before it reaches a social account.

## 8. Cost — estimates pending Phase D

Shortlist of 10 windows x 5 frames = 50 frames, **flat with video length**.

| | est. $/video | vs today ($0.009) |
|---|---|---|
| Claude Opus 5 | ~$0.30 | 33x |
| Claude Sonnet 5 | ~$0.12 | 13x |
| Claude Haiku 4.5 | ~$0.06 | 7x |
| Gemini frames | ~$0.02 | 2x |
| Gemini full video, 30 min | ~$0.14 (flash-lite) | 15x |

## 9. What could make this not work

1. **No verdicts, no truth.** Phase A is the gate on every later number.
2. **The input is noisy underneath.** Transcription varies 15.7% of words run to
   run (measured 16-sep). Measure on a **fixed** transcript, never a fresh one.
3. **Latency.** Frame extraction from a 530 MB source with random seeks is
   unmeasured.
4. **Vision may not beat text on talking-head content.** Measure `creator_chaos`
   first, where the payoff is non-verbal; if `tech_podcast` shows no gain, gate
   vision off for that niche rather than paying for it.

## 10. Start here

1. **Phase A**, the verdict store — it gates everything and starts collecting today.
2. **Phase B**, the two niches — the fastest real improvement to what gets posted.
3. **Phase C**, the length guard — small, and stops a bad submit.

Then D → E → F, with F measured on `creator_chaos` against the verdicts A has by
then.
