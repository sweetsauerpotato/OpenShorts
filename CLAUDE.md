# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenShorts is an AI-powered vertical video generator that transforms long YouTube videos or local uploads into viral-ready short clips (9:16 format) for TikTok, Instagram Reels, and YouTube Shorts. Viral moment detection and title generation run on Google Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`, overridable with `GEMINI_MODEL`) **or on a local Ollama** — see "Clip-selection provider" below.

### Clip-selection provider (`llm_provider.py`)

**The whole video→shorts pipeline depends on exactly one model call.**
Transcription (faster-whisper), shot detection (TransNetV2), face/person
tracking (MediaPipe + YOLOv8), reframing, subtitles and hooks are all local.
The only critical-path AI call is `main.get_viral_clips`, and it is **text-only
with a Pydantic JSON schema** — which is why a 7B local model can serve it.

`LLM_PROVIDER` (`auto` default) picks between them: Ollama when it answers *and*
`OLLAMA_MODEL` is actually pulled, else Gemini when a key exists, else a raise
naming both remedies. Checking the model list matters — a reachable daemon
without the model turns every generation into a 404 at job time.

`OllamaClient` **duck-types the google-genai surface**
(`client.models.generate_content(...)` → `.parsed`/`.text`/`.candidates`/
`.usage_metadata`) rather than introducing a neutral interface. That is
deliberate: `main._run_gemini_stage` and `_run_stage_split` encode two
production incidents in their retry and bisect logic, and their tests inject
exactly that shape. Matching it means the swap is one line at the construction
site and **all 754 existing tests pass unmodified**. Consequences worth keeping:

- The response has **no `prompt_feedback`** and empty `candidates`, so
  `gemini_worker.raise_if_blocked` no-ops and the content-policy bisect is
  never reached. Ollama has no content policy; "never fires" is correct.
- Errors are raised with text the retry loop already matches (`503`,
  `UNAVAILABLE`, `empty response body`). Deterministic failures — a missing
  model (404) or a truncated answer (`done_reason=length`) — deliberately carry
  **no** matching token, so they fail in 2s instead of after 35s of backoff.
- Native `/api/chat`, not the OpenAI-compatible `/v1`, because `options.num_ctx`
  is native-only. Without it Ollama uses the server default (possibly 4096) and
  **silently truncates** a 3-4k-token prompt — no error, just a model scoring
  windows it never read. The shim always sends it and warns above 90% use.
- Local runs report the model as `ollama/<name>`; `clip_selection.MODEL_PRICES`
  has an `"ollama/"` prefix entry so cost reports an honest $0.

Measured: a batch of 8 scoring windows is ~2.7k tokens, so 8192 context is
ample; qwen2.5:7b-instruct occupies ~5.3GB VRAM. Budget +3-5 min per job versus
Gemini. Do not parallelise the batches — one GPU, Ollama serialises anyway.

**Still Gemini-only** (no local equivalent, all already degrade without a key):
silent/sparse-speech clip selection (`get_visual_clips` uploads the whole
video), `/api/edit` and `/api/effects/generate`, thumbnail image generation,
SaaSShorts research (Google Search grounding).

### How clips are chosen: scoring and creator instructions

`get_viral_clips` works on ~90 s transcript windows overlapping by 30 s. **Pass
1** scores every window in batches of 8 (`SCORE_PROMPT_TEMPLATE`);
`clip_selection.build_shortlist` keeps the best `max(3, min(10, duration // 90
+ 2))`. **Pass 2** (`DETAIL_PROMPT_TEMPLATE`) reads per-sentence timestamps to
place the cut inside each shortlisted window and writes the hook, title and
descriptions. Both passes use `GEMINI_MODEL`; there is no per-pass setting.

**Every window gets a score** (14-sep-2026). The prompt used to ask for "up to 3
windows" per batch, so on a 55-min documentary only 19-20 of 51 windows were
ever scored. It now scores all of them on an anchored scale (90-100 hook AND
payoff, 70-89 one of the two, 40-69 needs context, 0-39 filler). Measured with
the harness, 2 runs each; it fixed less than the coverage number suggests:
- coverage 19-20 -> 51 of 51; distinct scores 6-8 -> 21-22
- ties were NOT fixed: flash-lite answers on a grid (72, 75, 78, 80, 82, 85,
  88), tied pairs in the top 10 went 10-14 -> 7-9, and ties at the shortlist
  cut (the only ones that change what reaches pass 2) left out 1-2 windows
  both before and after
- no final clip on that video came from a window the old prompt had not
  scored. The cap bites when strong moments are adjacent, because adjacent
  windows share a batch: with a topic instruction 3 of the top 4 sat in one.
- on short sources pass 1 stops filtering: a 10-min slice sends 8 of 9 windows
  to pass 2 (was 4) and gets 6 clips (was 4); the 2 extra score lowest in pass 2
- 1.17 -> 1.51 cents per 55-min video on flash-lite (pass-1 output ~3.8k tokens)

`build_shortlist` keeps the best score of a repeated id, ignores ids that are
not windows (the old inline sort cut the top N first, so a made-up id took a
slot) and breaks ties by position in the video, never by answer order.

**Creator instructions** (`clip_instructions` on `/api/process`, the dashboard's
"what to clip" box, MCP `process_video`) are free text of at most 1000
characters after `normalize_clip_instructions`; longer input is a 400, never a
silent cut. app.py writes `<job>/clip_instructions.txt` and passes
`--instructions-file`, so they survive a resume (the manifest keeps `cmd`, not
the job's env). `with_clip_instructions` inserts a delimited block after
`.format()` into pass 1, pass 2 and `get_visual_clips`: blank instructions keep
every prompt byte-identical and braces the user types stay inert. Pass 1 must
get them, since that is where most windows are eliminated. Measured with "Only
moments about the Crusades and the Knights Templar." (9 of 51 windows mention
either): on-topic clips 1 of 6 without, 3/3 in both runs with, the same clips
both times. The block costs +128 input tokens per call for that 56-character
text, but fewer clips means less pass-2 output (6x the input price on
flash-lite), so the job got cheaper: 1.51 -> 1.46 cents. The box is cleared for
each video on purpose.

**Clips are cut on whole sentences** (15-sep-2026, `clip_selection.
snap_clip_to_sentences`, replacing plain word snapping in `get_viral_clips`).
Of 73 saved clip answers only 17 (23%) ended on a finished sentence, and a real
50-min job ended 4 of 6 clips mid-statement although each sentence finished
2.4-4.3 s later. Two causes, neither the model's judgement: pass 2 closes on a
Whisper line and 44-57% of Whisper lines end mid-sentence; and
`snap_clip_to_words` takes the NEAREST word end, so an end placed on the next
line's start grabbed that line's first word ("And", "So") whenever the pause
before it (0.54 s) outlasted the word (0.36 s) — 20 of the 73. The rule: an end
on a finished sentence stays; one or two words of a new sentence are cut back;
a mid-sentence end finishes its sentence if that takes ≤ 8 s, else the nearest
sentence end that keeps the band; a start moves earlier to its sentence start
(≤ 8 s) or drops a ≤ 2-word tail of the previous one, never later, so the
chosen opening stays. Sentences come from punctuation (`sentence_spans`).
Whisper sometimes leaves it out, so a run-on "sentence" over 45 s is split near
its middle, at a pause or before a capitalised word. Two versions of that were
wrong: splitting at the "longest" pause peeled one word at a time off a 50 s
stretch where every gap was 0.00 s (55 one-word lines, and pass 2 closed a
clip on "that"); and a 30 s limit split run-ons of 30-37 s before proper
nouns ("that is the | Al-Aqsa Mosque."). `tools/compare_selection.py --replay`
re-cuts saved raw answers with no API calls: finished endings 16% -> 100% on
the 55-min records, 56% -> 97% on the 10-min slice (its miss is where the slice
itself cuts the transcript), 2/6 -> 6/6 on the job; no start moved later, no
clip left 15-60 s. The clip editor and MCP `recut_clip` keep word snapping on
purpose: their cuts are deliberate.

**Pass 2 reads whole timed sentences.** Its `lines` are
`"[912.4-918.9] sentence"` (the sentences that start in the window), and the
prompt takes `start`/`end` straight from the opening and closing sentence and
requires ending on a finished thought. The old lines were Whisper segments with
only a start, so `end` had to be "the next line's start". Live, 55-min x2 twice
(15-sep-2026): the model's own end lands on a finished sentence 53% -> 81%
(22/27; the misses sit where Whisper left out a period, e.g. "...impure state |
But"), cutting then moves ends by a median 0.3 s instead of repairing them,
run-to-run clip agreement 2/6 -> 3/7 and 4/7, Tier 1 clean, pass-2 input
~7.0k -> ~7.2k tokens and cost unchanged (1.51-1.57 cents).

**The randomness left is in pass 2, not the ranking.** On the 10-min slice both
runs sent the identical 8 windows to pass 2 and only 3 of 6 final clips
matched; `_run_gemini_stage` sets no temperature or thinking level (API
defaults). A side-by-side rerank of the top candidates would end the ties, but
ties at the cut moved only 1-2 windows — measure pass-2 variance before building
one. `GEMINI_THINKING_SCORE` looks like a control for this and is not: only the
standalone `gemini_worker.py` CLI reads it. Flash models (3.6-3.8) think at
"medium" by default, billed as output.

**Measuring it:** `tools/compare_selection.py` runs the real `get_viral_clips`
on a cached transcript and records the raw answers: schema compliance, window
echo, band violations, copied playbook hooks, windows scored, score bands,
shortlist ties and overlap, tokens per stage, run-to-run agreement, and
on-topic clips with `--instructions` / `--topic-keywords`. Keep sources in
`.cache/harness/`, never `uploads/` (deleted after 6 h, transcript cache
included). A 55-min run costs ~1.5 cents.

**Gemini 503 "high demand" gets a long retry.** It hit 8 calls in 10 harness
runs on 14-sep-2026: 6 went through after one retry, 1 after two, and 1 was
still overloaded after the old budget (3 attempts, 15 s of waiting) and failed
its run. A default google-genai client does not retry at all (`retry_options` is
None), so `_run_gemini_stage` is the only retry. A 503 now backs off 5, 10, 20,
40, 60... s (±20% jitter) for up to `GEMINI_OVERLOAD_WAIT_SECONDS` (180) of
waiting, then raises `GeminiOverloadedError`, which `get_viral_clips` re-raises
so the job says Gemini was overloaded instead of "did not return usable clips".
Rate limits, 500s and empty bodies keep 3 attempts (`clip_selection.
classify_gemini_error`). Replayed through the real SDK with the 503 body Gemini
sent: bursts of 3-6 consecutive 503s used to fail the job and now recover.
`get_visual_clips` (silent video) and the layout picker make their own
single-attempt calls and do not use this loop.

### Pasted transcripts (`transcript_import.py`)

`transcript` on `/api/process` (dashboard: the "I have the transcript" box;
MCP `process_video`) takes a transcript the user already has: YouTube's "Show
transcript" list copied as is, SRT, WebVTT (YouTube's word-timed auto-caption
VTT included) or OpenShorts JSON. It is parsed at submit, so a paste the app
cannot read is a 400 that says how to fix it (at most 300,000 characters, which
also stays under Starlette's 1 MiB form-part limit), and refused when its last
line starts more than 2 s after the video ends: that is another video's
transcript. main.py gets it as `--transcript` with `origin: "pasted"` and never
transcribes the whole video; a misfit only found after the download fails the
job instead of transcribing.

**It chooses the clips; Whisper still cuts them** (15-sep-2026, decided with
the measurement). Only VTT word tags and JSON words carry real word times.
Everything else gets each line's words spread over the line by length. Turning
Whisper's own transcripts into what a user pastes and re-cutting 82 saved clip
answers against the true words:
- YouTube panel copy (whole-second times): median word error 0.5 s; 33-43% of
  clips cut off their last word, and 50-67% without punctuation, which is how
  auto-captions come
- SRT with exact times: 0.15 s; 0-4% of clips cut off their last word
- Whisper (today): 0%

So `main.refine_pasted_transcript` runs Whisper on the chosen clips alone,
8 s either side (the sentence cut's `max_shift`), merges those exact words in
(`merge_exact_words`; `exact_ranges` says where the times are real) and cuts
every clip again on them. That was 13-19% of the video on two real jobs.
Live through `/mcp` (3.1-min upload, a panel-format paste made from its own
Whisper transcript, 1 clip):
- the cut on the estimates, 112.95-152.04, ended inside "concept." and opened
  mid-sentence
- Whisper on 0.9 of the 3.1 min (21 s of CPU; 95% the same words as
  whole-video Whisper, median time difference 0.00 s) moved it to
  111.75-152.64: whole sentences, and the captions end on "CONCEPT." at the
  clip's last second
`transcribe_media(language=)` takes the pasted transcript's language, because
Whisper guesses from the first 30 s and a short stretch can open on music.
Agent jobs with a transcript skip Whisper entirely (8 s instead of 93 s on a
3.1-min upload). The clips an agent sends will need the same clip-only pass
when they are rendered.

## Development Commands

### Local Development (Docker)
```bash
docker compose up --build   # Build and run full stack
```
- Backend: http://localhost:8000 (FastAPI/Uvicorn)
- Frontend: http://localhost:5175 (Vite proxies API calls to backend)
- After editing a module the API process imports at startup (`app.py`,
  `clip_selection.py`, `mcp_server.py`, ...), `docker restart openshorts-backend`.
  Jobs run `main.py` in a fresh process, but the clip editor imports `main.py`
  lazily inside the API process: on 15-sep-2026 a new `main.py` met the
  startup-cached `clip_selection` there and every rerender failed with an
  ImportError until the restart.

### Frontend Only (Dashboard)
```bash
cd dashboard
npm install
npm run dev       # Dev server with HMR (port 5173)
npm run build     # Production build
npm run lint      # ESLint (strict, --max-warnings 0)
```

### Backend Only
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Architecture

### Core Processing Pipeline
1. **Ingest** - YouTube download (yt-dlp) or local upload
2. **Transcription** - faster-whisper with word-level timestamps
3. **Scene Detection** - PySceneDetect for segment boundaries
4. **AI Analysis** - Gemini identifies 3-15 viral moments (15-60 sec each)
5. **FFmpeg Extraction** - Precise clip cutting
6. **AI Cropping** - Vertical reframing with subject tracking
7. **Effects/Subtitles** - Optional AI-generated FFmpeg filters
8. **Hook Overlay** - Text overlays with styled fonts
9. **Voice Dubbing** - Optional ElevenLabs AI translation (30+ languages)
10. **S3 Backup** - Silent background upload
11. **Social Distribution** - Upload-Post API (async upload)

### Key Files
| File | Purpose |
|------|---------|
| `main.py` | Core video processing: transcription, scene detection, clip extraction, vertical reframing |
| `clip_selection.py` | Stdlib-only clip-selection helpers: windows, shortlist, sentence cuts, creator instructions, Gemini retry policy, model prices |
| `tools/compare_selection.py` | Harness that measures clip selection on a cached transcript (see "How clips are chosen") |
| `transcript_import.py` | Stdlib-only reader for pasted transcripts (YouTube panel, SRT, VTT, JSON) and the clip-range merge of exact words (see "Pasted transcripts") |
| `app.py` | FastAPI server with async job queue and REST endpoints |
| `editor.py` | Gemini AI integration for dynamic video effects (FFmpeg filter generation) |
| `hooks.py` | Hook text overlay generation with font rendering |
| `s3_uploader.py` | AWS S3 upload with caching |
| `subtitles.py` | SRT generation, FFmpeg subtitle burning, and dubbed video transcription |
| `translate.py` | ElevenLabs dubbing API for AI voice translation |
| `dashboard/src/App.jsx` | Main React component with state management |
| `dashboard/src/components/TranslateModal.jsx` | Voice dubbing UI with language selection |
| `dashboard/vite-plugin-seo.js` | Build-time SEO surface: injects crawler-visible homepage content, emits static pages, sitemap.xml and llms.txt |
| `dashboard/seo/data.js` | Single source of truth for pricing, pipeline and competitor facts used by every generated page |

### SEO / AI-crawler surface

The dashboard is a client-rendered SPA with hash routing, so the HTML served for
`/` used to contain an empty `<div id="root">`. Googlebot renders JavaScript and
saw the real page; GPTBot, ClaudeBot and PerplexityBot do not and measured the
homepage as zero characters of text. `vite-plugin-seo.js` fixes that at build time:

- Injects the content of `seo/landing-fallback.js` into `#root`. React's
  `createRoot().render()` replaces it on mount, so users get the app and
  non-executing clients get the copy. **Keep it in sync with `Landing.jsx`.**
- Emits the standalone pages (the `/alternatives` cluster, the clip-generator,
  open-source, use-case and automation pages, and `/mcp`; the full list is
  `buildPages()` in `seo/pages.js`) as flat `.html` files.
  nginx resolves the clean URL through `try_files $uri $uri.html`; serving them as
  directories instead makes nginx 301 to a trailing slash and every canonical
  would then point at a redirect.
- Generates `sitemap.xml` and `llms.txt` from the same page list, so they cannot
  drift. Do not add a static `public/sitemap.xml` back.

When editing pricing anywhere, edit `seo/data.js` too. Nothing on the site should
say "OpenShorts is free" without naming the Cloud price in the same breath: both
are true of different editions and quoting only the first one is what makes AI
answers describe the paid product as free.

### Cómo se elige el layout

`POST /api/process` acepta `layouts`: una lista (JSON) o cadena separada por
comas con `auto`, `split`, `screencast`, `speaker_cut`, `punch_in` y `none`.
Cada nombre enciende su variable de entorno para **ese** trabajo
(`app.py:layout_env`); `none` apaga el picker aunque prod corra con
`AUTO_LAYOUT=1` (recorte simple y nada más). Sin `layouts` manda el env del
despliegue, que desde el 25-ago-2026 es `AUTO_LAYOUT=1`. El dashboard lo expone
en opciones avanzadas ("vertical layout": auto / split / screencast / none,
`MediaInput.jsx`, recordado en `localStorage.os_layout`).

`auto` activa `layout_picker.py`: **una** llamada a Gemini por vídeo de origen
(no por clip) que elige entre `none` / `screencast` / `split`. Medido sobre el
corpus de 48 contra etiquetas revisadas a mano: 94% / 92% / 96% en tres pasadas,
con 0-1 falsos positivos sobre los 28 clips que no deben tocarse, y solo 2 clips
que cambian de respuesta entre pasadas.

**Manda 12 fotogramas a 1024px, no el vídeo.** Gemini factura vídeo a ~300
tokens por segundo: una hora de fuente son ~1,08M de tokens (no cabe en una
ventana de 1M) y una subida de 1-2 GB para recibir una palabra. Doce fotogramas
cuestan ~3k tokens **dure lo que dure la fuente**, que es lo que hace viable
esto con los podcasts de una hora que entran de verdad. La resolución importa y
el número de fotogramas no: a 640px detecta 15 de 20 (una hoja de cálculo es
ilegible), a 1024px sube a 17, y pasar a 24 fotogramas lo empeora. A 1024px la
diferencia con mandar el vídeo entero cae dentro de la varianza que ya tiene el
propio modo vídeo, a 2,2 s por clip en vez de ~15 s.

Lo que hace que funcione, y que conviene no deshacer: se le pide una **decisión
entre opciones cerradas**, no una medida. Los cuatro intentos anteriores (Canny,
MSER, cobertura temporal, anchura) le pedían un número y ninguno separó una hoja
de cálculo de un marcador de esquina. La varianza que este repo atribuía a
Gemini era de las medidas continuas, no del modelo.

`layout_picker.apply()` sólo **añade**: una elección explícita del usuario nunca
se desactiva porque el modelo diga `none`.

### Thumbnail Studio (`thumbnail.py`, `/api/thumbnail/*`)

Titles come from the transcript plus 10 frames at 1024px, never the whole
video (same reasoning as the layout picker: an hour of video is ~1M tokens for
a text task). Two calls: a 25-title brainstorm across fixed styles, then a
critic that scores, dedupes by angle and returns 10, each paired with a 1-4
word `thumbnail_text` that complements the title rather than repeating it.
Rules baked in: payoff inside 50 characters (phones cut there), keyword in the
first 3 words, same language as the transcript. Text model is
`GEMINI_MODEL_THUMBNAIL` (default `gemini-3.7-flash`), deliberately not
`GEMINI_MODEL`: flash-lite is fine for a closed-choice layout pick and visibly
worse at creative titles. Image model is `GEMINI_IMAGE_MODEL` (default
`gemini-3.1-flash-image`).

Thumbnails are `count` **different concepts**, not one prompt repeated: a text
call designs each (hook text, side for the text, palette, scene prompt), then
one image call per concept in parallel. By default (`burn_text=true`) the
image model is told to leave that side as negative space and PIL sets the text
in Anton with a black stroke, so accents and spelling are never wrong; the
`AI painted` toggle lets the model render the text itself. Every output is
cover-cropped to 1280x720 and saved under YouTube's 2 MB limit.
`GET /api/thumbnail/frames/{session}` scores sampled frames by face area and
sharpness (MediaPipe + Laplacian), keeps them spread across the runtime, and
the dashboard offers them as the person reference so the thumbnail shows the
creator instead of a stranger; an uploaded face photo still wins.

### Video Reframing Modes

**A source already shot vertical is passed through untouched.**
`reframe_v2.source_already_fits()` gates it: every layout below reorganises the
frame to buy back width the crop threw away, and on a 9:16 upload there is none
to buy. GENERAL was the visible failure — its 0.42 height ratio, which buys
presence on a landscape source by overflowing the sides, scaled a 1080x1920
source down to a 453px sliver floating over a blurred copy of itself, and the
scene classifier routes every face-less shot (a slide, a screen recording) there.
So the picker is skipped (one Gemini call saved per upload), the classifier is
skipped, and every scene renders TRACK, whose crop is the whole frame.
`general_filtergraph` additionally floors the foreground at the height where the
source fills the output width, so the editor's explicit GENERAL override on a
portrait clip cannot reproduce the shrink either.

- **TRACK Mode** (single subject): MediaPipe face detection + YOLOv8 fallback with "Heavy Tripod" stabilization
- **GENERAL Mode** (groups/landscapes): Blurred background layout preserving full width
- **SPLIT Mode** (two-shot conversation, `split_layout.py`): both speakers stacked
  in half-frames. Off by default (`SPLIT_LAYOUT=1`); v2 engine only, so a
  fallback to the v1 loop silently renders GENERAL instead. It upgrades scenes
  the classifier already sent to GENERAL, never TRACK ones, and needs both faces
  visible **in the same frame** for at least half the sampled frames — that is
  what separates a real two-shot from a plano/contraplano, where stacking would
  show the same person twice. `SPLIT_TIGHTNESS` (default 0.8) trades a little
  upscale for keeping the other speaker out of each half. Captions on a SPLIT
  stretch sit on the seam between the halves (`{\an5}` per word event in
  `subtitles.generate_ass`), the one place they cover nobody; the render
  records which stretches are stacked in a `<clip>.layout.json` sidecar
  (`layout_ranges.py`) and every metadata writer copies it into the clip's
  `layout_ranges`, so `/api/subtitle` finds it after a restyle too. The fast
  rerender (cut without reframe) carries the canonical clip's ranges through
  the new cut (`layout_ranges.remap`, in `recut.perform_recut`). Only the
  ASS path can do this; SRT burns keep one alignment for the whole file.
- **SCREENCAST / WIDE Modes** (`screencast_layout.py`, `SCREENCAST_LAYOUT=1`):
  for scenes whose meaning lives outside the centre. Gemini reports each range's
  **width_fraction**, and that is the gate — coverage was tried before and did
  not separate a spreadsheet from a corner ticker, while width does (a bug spans
  ~15% and survives any crop, a spreadsheet spans ~100% and cannot). Content
  narrower than 0.5 moves nothing. Between 0.5 and 0.85 there is room beside the
  content, so SCREENCAST stacks it over the presenter. Above 0.85 the presenter
  is composited **on top of** the content and stacking would show it twice, so
  those scenes get WIDE: the GENERAL layout with side-cropping disabled.
- **INSET Mode** (`camera_inset.py`): pantalla a ancho completo arriba, el
  recuadro de la webcam ampliado abajo. Para el caso de una sola fuente con la
  cámara compuesta en una esquina (OBS, VOD de stream). Se encadena detrás de
  la decisión `screencast`, **no** se le pregunta a Gemini: ofrecido como cuarta
  opción respondió `screencast` en los 5 clips que tienen recuadro, en dos
  pasadas, y la exactitud global cayó de 92% a 83-85%. El detector geométrico
  encuentra esos 5 sin falsos positivos. Los tres filtros que hacen falta, cada
  uno pagado con una iteración: sujeto **pequeño**, **descentrado en
  horizontal** (una cara de talking head está centrada aunque esté alta), y
  **quieto entre muestras** (3-11px frente a 316px de una persona real).
- **ALTERNATE Mode** (`active_speaker.py`, `SPEAKER_SIGNAL=1` + `SPEAKER_CUT=1`):
  hard cuts to whoever is talking, rendered through the TRACK path as a
  trajectory with jumps. `SPEAKER_SIGNAL=1` alone just gates SPLIT on both people
  actually speaking. Mouth activity **must** be normalised per speaker before
  comparing (`normalise_activity`): raw frame-difference magnitude scales with
  local contrast and lighting, and on a real two-shot it handed one speaker
  90-100% of the scene.
- **Punch-in** (`punch_in.py`, `PUNCH_IN=1`): not a layout. A ~12% push on the
  clip's beats, riding the TRACK path by widening its per-frame crop command
  from x-only to w/h/x/y. Beats currently come from the audio envelope;
  `emphasis_times` is a plain list of seconds so the transcript's hook words can
  replace it without touching the module.

### Key Classes
- `SmoothedCameraman` - Stabilized camera movement with safe zone logic (prevents jitter)
- `SpeakerTracker` - Prevents rapid speaker switching, handles temporary occlusions

### API Endpoints
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/process` | Submit video for processing |
| GET | `/api/status/{job_id}` | Poll job status and logs |
| POST | `/api/edit` | Apply AI video effects |
| POST | `/api/subtitle` | Generate and apply subtitles (auto-transcribes dubbed videos) |
| POST | `/api/hook` | Add text hook overlays |
| POST | `/api/translate` | AI voice dubbing via ElevenLabs |
| GET | `/api/translate/languages` | List supported dubbing languages |
| POST | `/api/social/post` | Post to social media (async upload) |
| POST | `/mcp` | MCP server (JSON-RPC): the pipeline as agent tools |
| POST/GET/DELETE | `/api/keys` | User API keys (cloud mode, session JWT only) |
| DELETE | `/api/account` | Erase the account and everything in it (GDPR art. 17) |

### Agent access (MCP, API keys, webhooks)

- **API keys** (`cloud/api_keys.py`): `osk_...` tokens, sha256-stored, created in
  the dashboard account page. `cloud/auth.get_current_user_optional` accepts
  them (`Bearer osk_...` or `X-API-Key`) and resolves the owner, so metering,
  entitlement, plan priority and job ownership apply to agents with zero
  endpoint changes. Key management itself refuses API-key auth: a leaked key
  cannot mint replacements.
- **MCP server** (`mcp_server.py`, mounted always): stateless Streamable-HTTP
  JSON-RPC at `/mcp` — no SDK dependency, ~3 methods + 8 tools. Each tool calls
  back into this same app in-process (`httpx.ASGITransport`) forwarding the
  caller's auth headers, so it can never drift from the REST behavior. Cloud
  mode 401s without a resolvable user; self-host stays BYOK-open.
- **OAuth for MCP clients** (`cloud/mcp_oauth.py`, cloud mode only): claude.ai
  and ChatGPT connect by URL, so the server publishes RFC 9728/8414 metadata
  under `/.well-known/`, accepts dynamic client registration (`POST
  /oauth/register`, public clients, PKCE S256 mandatory) and bounces
  `GET /oauth/authorize` to the dashboard consent screen (`#/oauth/authorize`),
  because the session JWT lives in localStorage on the frontend host and a
  bare API GET cannot see it. `POST /api/oauth/authorize` (session auth) mints
  the code; `POST /oauth/token` redeems it by **minting an ordinary `osk_`
  key** named after the client and returning it as the access token. No new
  auth path, no refresh tokens: the key shows up in Account → API keys and
  revoking it disconnects the app. The `/mcp` 401 carries
  `WWW-Authenticate: Bearer resource_metadata=...` so clients find the flow.
  `oauth_codes` is in `USER_OWNED_TABLES`; `oauth_clients` deliberately not.
- **Webhooks**: `POST /api/process` takes `webhook_url` + optional
  `webhook_secret` (HMAC-SHA256, `X-OpenShorts-Signature`). Validated with
  `security_utils.assert_public_url` at submit AND at delivery (DNS rebinding).
  Fired once per job from `run_job_wrapper` after the R2 archive so the payload
  can carry durable download links; survives redeploys via the resume manifest.
  `PUBLIC_API_URL` env sets the absolute-URL base when behind a proxy.
- **Agent selection** (`selection=agent` on `/api/process` and MCP
  `process_video`, 15-sep-2026): the agent, not Gemini, chooses the clips.
  The job downloads and transcribes, then stops (`main.py --transcribe-only`):
  metadata with `shorts: []`, `awaiting_clips: true`, the transcript and the
  source (a downloaded one is always kept). No layout pick and no model call,
  so no Gemini key is needed. A video without usable speech fails and names
  `selection=ai`, whose silent-video path is a Gemini call. `target_clips` and
  the clip length band are 400s (the agent decides); instructions are kept.
  What the clips should look like (format, layouts, hook, hook style,
  captions) goes to `<job>/agent_job.json`, for the render job the agent's
  clips start. `awaiting_clips` rides the job result (also when recovered
  after a restart), the webhook payload and MCP `get_job_status`, and
  `list_clips` refuses such a job instead of answering "0 clips". Measured
  through `/mcp`: a 3.1-min upload reached `awaiting_clips` 93 s after submit
  (64 s transcribing on CPU, no model call); a 60 s tone failed with that
  message. Self-host gap: `create_upload` hands out an `upload_url` on
  `http://openshorts.internal` (the MCP tools' in-process base) unless
  `PUBLIC_API_URL` is set; PUT to `http://localhost:8000/api/uploads/<id>`.
- **get_transcript** (`GET /api/transcript/{job_id}` and MCP, 15-sep-2026):
  the agent's view of a job's transcript, for any finished job.
  - Sentences come as `[start-end] text`, the same lines pass 2 reads, in pages
    of 60,000 characters of lines (`next_start` points to the next page).
  - The first page adds `rules` and the job's `instructions`. The rules are
    `clip_rules.md`, which the user edits; it is read on every call, and
    `.dockerignore` re-includes it past `*.md`.
  - `words=true` returns one stretch of at most 300 s as `[start, end, word]`,
    for exact cuts.
  - `timing: "estimated"` marks a pasted transcript.

  Measured through `/mcp`: the 50-min democracy job is one page of 63,798
  characters (~16k tokens at 4 characters per token, under Claude Code's
  default 25k output limit), 624 sentences, returned in 0.15 s. The 23.6-min
  Jake Paul job is 25,902 characters; 60 s word by word is 5,375.

  Two things about Claude Code as the client: it gives the model a tool result
  once (not both `content` and `structuredContent`), and it keeps the tool list
  from session start, so new tools and fields need an MCP reconnect.

### Account erasure (GDPR art. 17)

`DELETE /api/account` (`cloud/account.py`, dashboard: Account → Delete account)
is immediate and irreversible: there is no recovery window because after the
delete there is nothing left to authenticate a recovery request against. It
refuses API-key auth (a leaked `osk_` must not destroy its own account) and
requires the caller to retype the account email.

The order of the steps is the design, and each one is a failure mode:
**Stripe cancel first**, aborting the whole thing if it fails, so we never erase
a user we are still billing; **R2 before the database**, because those rows are
the only index of which objects are theirs and dropping them first turns a
failed purge into permanent orphans; the DB delete is **one transaction** over
an explicit table list (`USER_OWNED_TABLES`) rather than the declared ON DELETE
CASCADEs, since `create_all` never ALTERs an existing table and a constraint
added after a table shipped exists in the models but not in production.
`tests/test_account_erasure.py` fails if a new table references `users.id`
without joining that list.

`app.py` registers a callback for the local working files, which record
ownership three different ways: the `.owner` file clip jobs write (so jobs
recovered from disk after a restart count too), `saas_jobs`, and
`thumbnail_sessions`. That last one is the only thing that ever deletes
generated thumbnails: the hourly sweep skips their directory and they are
served publicly at `/thumbnails/`.

What deliberately survives: the Stripe customer and its invoices (6-year
retention, Spanish commercial law) and one `account_deletions` row holding a
sha256 of the email as proof the erasure happened, itself purged after 5 years.
The "why are you leaving" answer is a closed list (`DELETION_REASONS`), never
free text — anything the user could type would land in a row designed to
outlive them. Deleting users also made one webhook path reachable that never
was before: `_apply_topup` reads the user id from Stripe metadata, so it now
confirms the row still exists before inserting, or the FK violation makes
Stripe retry the same doomed event for three days.

### Concurrency Model
Async job queue with semaphore-based concurrency control. Configure via `MAX_CONCURRENT_JOBS` env var (default: 5). Jobs auto-cleanup after 1 hour.

### Deploys and running jobs (handover + drain)

Every push to `main` redeploys the API container. Coolify starts the NEW
container before stopping the old one (rolling update) and both share
`output/`, so `app.py` coordinates them instead of relying on a fast swap:

- Each instance writes its id to `output/.instance` at startup. An instance
  that sees another id there is the old one and **drains**: it finishes the
  jobs it is running, starts none, and leaves queued manifests on disk.
- A running job heartbeats its `.resume.json` every 10 s. The resume scan
  (startup + every 30 s) re-enqueues only manifests nobody heartbeated for
  60 s, so no job runs twice and none is lost. Max 2 resume attempts.
- SIGTERM (`docker stop`) drains too, up to `DRAIN_TIMEOUT_SECONDS` (840),
  then hands the signal to uvicorn. The app's Coolify stop grace period is
  900 s (`application_settings.stop_grace_period`); keep the timeout below it.
  After the drain hands the signal to uvicorn, `--timeout-graceful-shutdown 15`
  (Dockerfile) caps the wait for in-flight connections: uvicorn's default is
  unbounded, and one open range download kept a drained container alive for
  the full grace period while Traefik still routed half the traffic to its
  closed port.
- `/health/ready` + the Dockerfile `HEALTHCHECK` are what keep Traefik off a
  dying container: its docker provider only routes to `healthy` containers,
  so an instance answers 503 from the moment it gets SIGTERM (out of rotation
  within ~10 s, socket still open) and a booting one gets no traffic until it
  answers. Only SIGTERM flips it, not the marker drain: at that point the new
  container is still booting and nobody else would be routable. The Coolify
  app has its health check enabled on that path so it waits for the new
  container to be `healthy` before stopping the old one. With that option on,
  Coolify replaces the Dockerfile HEALTHCHECK with its own curl/wget command
  AND its own interval/retries (5 s × 3), so the image must ship `curl` or
  every deploy rolls back as unhealthy, and a stopping container takes 15 s
  to turn `unhealthy`. That is why the drain keeps serving for
  `PROXY_DRAIN_SECONDS` (20) after the jobs are done before it hands the
  signal to uvicorn: closing the socket earlier is 502s until Traefik
  notices (measured ~60 s per deploy with retries=12 and no grace). And
  `HARD_EXIT_SECONDS` (30) after that the process is ended outright: uvicorn
  finishing does not end the interpreter while an executor thread hangs in
  a network probe, and that kept a drained container alive for the full 900 s.
  `/health` stays a plain liveness probe for the external watcher.
- `/api/status` answers from disk for a job this instance never held, so a
  poll landing on either container during the handover is fine.
- `main.py` leaves `.transcript_checkpoint.json` in the job dir so a job that
  does get re-run skips the paid transcription (download and Gemini repeat).

Before pushing, still batch small commits (tests, docs) with the next real
change: every deploy is a ~5 min build plus a handover.
