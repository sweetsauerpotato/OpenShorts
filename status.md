# Status

A running log of what changed, when and why. Newest first. `CLAUDE.md` says how
the app works today; this file says what moved and what it measured.

Add an entry per commit: date, commit, what changed, why, files, how it was
verified.

## Where the work stands

Goal: give Claude a link in chat and get clips back, with Claude choosing the
moments instead of Gemini (MCP). Steps 1-4 and **3.5 are done** and work end to
end through `/mcp`. **Pushed to `origin/main` on 16-sep-2026** at `30f32ed`
(`6eb29f1..30f32ed`, 15 commits) — the first push of this whole line of work,
so that deploy carries sentence cuts, pasted transcripts, the MCP agent tools,
the hook fix, the transcript-hole repair and their tests all at once.

3.5 was run on the Jake Paul video (23.6 min) on 16-sep-2026: Gemini's 6 clips
(job `216fe922`) against Claude's 5 (job `01e40a09`), same source, same
transcript, same instructions — only the chooser differed. The user judged
Claude's set better on every axis and singled out the cold-open re-ordering
(clip 3 opens on the worst line of the 2017 email, then plays the setup).

Next, in order:
- **The vision feature — highest priority.** Plan: `docs/vision-plan.md`
  (16-sep). Blocked on 0.2, the labelled set, which needs an hour of the user's
  judging; without it no later number means anything. Selection is text-only today and
  that is its ceiling: see the 14.2 s hole below. Architecture options were
  brainstormed 16-sep; the plan is still to draft. Decisions already taken:
  Claude API cost is acceptable if the results earn it, local models stay a
  **support** option only, and the dev box is an i7-13th/RTX 4070 laptop with
  **8 GB VRAM** and 16 GB RAM — a 7B VLM alongside the existing stack will not
  fit comfortably (qwen2.5:7b-instruct alone is ~5.3 GB). Claude takes images
  only; there is no video input, so frames must be extracted here. Open
  questions: per-video budget, CPU budget for a dense local pass, and that
  there is no labelled "funny moment" set to measure against.
- `docs/how-openshorts-works.md` is written but not committed.

Sources are kept in `.cache/headtohead/` (Jake Paul `216fe922`, democracy
`c4e6388d`) because `output/` deletes jobs after 24 h — `216fe922` was swept
mid-session and the cached copy is the only one left. The render job
`01e40a09` still holds the source (it was hard-linked, so the sweep did not
free it) and works as a `source_job_id`.

---

## 2026-09-16 — `8f07418` + `916b4e6` feat(verdicts): what the user thought of a clip

**What**: `verdicts.py`, `POST /api/verdict`, `GET /api/verdicts`, MCP
`rate_clip`, and a "worth posting?" row on every dashboard clip card. Thumbs
up/down plus an optional reason from a closed list (`no_payoff`,
`needs_context`, `nothing_to_watch`, `boring`, `wrong_moment`, `bad_cut`).

**Why**: Phase A of `docs/vision-plan.md`, and the gate on everything after it.
Nothing measured this week can say whether clip selection got *better*: pass 1
answers on a coarse grid with 7-9 tied pairs in the top 10, run-to-run clip
agreement is 3/7-4/7, and the transcript underneath varies 15.7% of its words
between two runs of the same file. A vision change worth less than that noise
is undetectable without labels.

**The design points that matter**:
- Every row records **how the clip was produced** — mode, provider, model,
  niche, source, duration, instructions — or the data can only say some clips
  were liked, never whether vision helped. `mode` is `classical` until the
  modes land; unknown provenance stays null rather than guessed.
- `predicted_score` sits beside the human verdict, so `score_split` answers
  whether the picker's own score tracks what the user wants. If those two means
  never separate, re-ranking on that score cannot help.
- Append-only JSONL, latest row per (job, clip) wins, so re-rating is one more
  append and a crash mid-write loses a line rather than the file.
- Protected from every `OUTPUT_DIR` sweep alongside `thumbnails/`: a rating is
  only worth collecting if it outlives the 24 h purge that takes the clip.
- Account erasure drops that user's rows, since nothing else ever would.

**Files**: `verdicts.py` (new), `app.py`, `mcp_server.py`, `CLAUDE.md`,
`dashboard/src/components/ClipVerdict.jsx` (new), `ResultCard.jsx`,
`tests/test_verdicts.py` (+24).

**Verified**: 1129 tests (was 1105); dashboard lint and build. Live on a real
job: verdicts stored and read back with full provenance (duration resolved to
1417.49 s from the transcript), free-text reason 400, out-of-range clip 400,
unknown job 404, `rate_clip` served as the 11th MCP tool. The rows written
during that test were wiped, so the store starts empty.

**Not done**: the clip card has not been seen rendered — reaching one means
running a job through the UI — so lint and build are what stand behind the
markup. Engagement metrics are not wired; per the plan they are a weak late
signal, not the point.

## 2026-09-16 — `1343352` docs: the hole measurement was run-dependent, and repair changes what Gemini picks

**What**: corrects the numbers recorded for `6d63ccb` and adds the end-to-end
result. No behaviour change.

**The correction**: "36 gaps, 370 s, 26% of the runtime, +13.4% of words" was
measured on ONE transcription run and stated as a property of the video. It is
not. The same file through the same pipeline and settings:

    run A   3,566 words   266 segments   36 gaps >= 2 s   370 s
    run B   4,125 words   183 segments   17 gaps >= 2 s   133 s

Run B's holes are a strict subset of run A's (13 shared, 23 only in A, **0 only
in B**), so the runs do not fail in different places — one dropped more. A bad
draw loses 15.7% of the words a good draw gets, and which one you get is luck.
The 13 shared holes are the real music and montage stretches. So this is a
robustness net against a bad run, not a fix for a systematic loss.

**The result worth having**: a full job with `REPAIR_HOLES=1` drew the bad
transcription (3,566 words exactly) and repair took it to 3,983 — against the
good run's 4,125. What that changed:

    bad transcript, no repair    body shot 729.09-745.77  16.7 s  score 80
    bad transcript + repair      body shot 729.09-756.98  27.9 s  score 90
    good transcript, no repair   body shot 728.94-758.59  29.6 s  score 92

Same video, same instructions. Gemini extended its own clip by 11.2 s and
raised its own score once the words existed — the moment that had to be
hand-picked in the 3.5 head-to-head. Cost unchanged ($0.0082 vs $0.0092).
Repair left 27 holes (241 s) alone, and the known false keep (a Rick Ross
verse at 991-1003 s) survived into the transcript, though no clip used it.

**Files**: `CLAUDE.md`, `status.md`.

**Verified**: live job `367946a9`, 6 clips, same source and instructions as the
control job `6bf64b65` the user ran from the dashboard.

## 2026-09-16 — `6d63ccb` feat(transcript): recover the speech the whole-video pass drops

**What**: `transcript_holes.py` + `main.repair_transcript_holes` (off unless
`REPAIR_HOLES=1`). Gaps of >= 2 s between words are re-transcribed on their own
and the recovered words are spliced back in, after a closed-choice gate drops
anything that is song lyrics or noise.

**Why**: this started as "caption the silent stretch as [LAUGHTER]" and that
premise was wrong. The stretch is not non-speech — Whisper transcribes it
perfectly in isolation (90 words, identical with `vad_filter` on and off); the
whole-video pass emitted no segment there at all. `[LAUGHTER]` over "Welcome to
team 11, bro" would have been a caption that lies. The real defect is that the
transcript is missing 13.4% of its words, which costs captions AND clip
selection, since pass 1 can only score what it can read.

**The hard part**: a hole is where the music is. Naive repair returns Fortunate
Son and a Rick Ross verse as dialogue. Whisper's own confidence signals do not
separate music from speech — the two most confident-looking fragments in the
measurement were both lyrics (`no_speech_prob` 0.019 and `avg_logprob` -0.236,
best of all 26), the same failure mode the layout picker records from four
attempts at a continuous measure. So the gate is a closed choice
(dialogue/music/noise) over one text-only call through `llm_provider`, gated
per Whisper segment so mixed stretches are judged in pieces. Every failure path
— no key, bad JSON, unknown verdict, a 503 outlasting the retry budget —
returns "keep nothing", because a dropped line just leaves today's transcript
while a kept lyric gets published.

**Files**: `transcript_holes.py` (new), `main.py`, `gemini_worker.py`,
`CLAUDE.md`, `tests/test_transcript_holes.py` (+20).

**Verified**: 1105 tests in the container (was 1085). Live on the Jake Paul
source: 36 holes / 370 s (26% of runtime) re-transcribed in 114 s of CPU,
recovering 478 words (+13.4%); the gate answered 69 fragments in 6.8 s on
flash-lite, keeping 60 — **1 genuine false keep**, 1 segment where Whisper had
merged a lyric with real speech, 2 harmless drops of repetitive interjections,
and it correctly dropped both "All my new friends" and the Fortunate Son lines
while keeping the real narration inside the same hole. A `gemini-3.7-flash`
comparison could not be obtained (503 for the full 180 s budget, twice) — which
did confirm the fail-safe: the gate returned nothing and the transcript was
left untouched.

**Not done**: `REPAIR_HOLES` is off by default and the recovered words have not
yet been put through clip selection end to end — the point of the +13.4% is
that pass 1 should now see moments it was blind to, and that is unmeasured.

## 2026-09-16 — `857eece` fix(agent-clips): an edge placed inside a transcript hole is kept

**What**: `agent_clips.snap_edge` no longer drags an edge back to the nearest
word when the caller put it inside a gap of `GAP_IS_A_HOLE` (2 s) or more. It
still stops a full lead/tail clear of the word on the far side, and only holes
bounded by a word on both sides qualify — past the first or last word the trim
stays. Below 2 s nothing changes.

**Why**: the padding assumes a gap is silence. A hole is a stretch Whisper
returned nothing for, which is not the same thing. Found during the 3.5
head-to-head: the best clip of the Jake Paul video asked to end at 760.60 and
was silently cut to 747.13, deleting the punch landing, the laughing and
"Welcome to team 11". Measured with ffmpeg, that 14.2 s hole (746.7-761.0) runs
at **-17.5 to -27.9 dB RMS** — louder than the speech around it, and the
loudest single second in the window (748 s) is inside it. The same 14 s is why
Gemini's transcript-only scoring ended its own clip at 745.77: both halves of
the pipeline are blind to non-verbal moments, and this fixes the half that
ignores an agent saying otherwise.

**Threshold**: 2 s is where the two populations separate. Inter-word gaps over
two real transcripts (3,566 and 8,440 words) — `>= 0.5 s`: 68 / 272;
`>= 1 s`: 45 / 54; `>= 2 s`: 36 / 26; `>= 5 s`: 19 / 13. Both collapse between
0.5 and 1 s (ordinary pauses) and flatten after 1.5-2 s.

**Files**: `agent_clips.py`, `CLAUDE.md`, `tests/test_agent_clips.py` (+6).

**Verified**: 1085 tests in the container (was 1079). Replayed over the 13
pieces of the real 5-clip set, **exactly 1 edge moved** (+13.3 s on the broken
one) and the other 12 were byte-identical. Live through `/mcp` after a restart:
`render_clips` with the original `760.6` produced segments
`726.27-734.29 + 737.55-760.47` with no hand repair, where before the fix the
same request rendered a 17.6 s clip ending at 747.13.

## 2026-09-16 — `350ef48` feat(mcp): render_clips — the agent's own clips, in pieces

**What**: `/api/process` with `source_job_id` + `clips` (MCP `render_clips`)
renders clips an agent chose, in a job of its own. A clip is pieces of the
source in play order (punchline first, dead parts dropped), plus hook, title,
descriptions, score, reason.
- New `agent_clips.py`: validates before any work (15 clips, 12 pieces, 5 s
  total, 180 s span, platform text limits) and places piece edges in the
  silence beside their word (≤ 0.5 s before, ≤ 0.4 s after, never past the
  middle of a gap).
- `main.py --clips-file`: cuts and reframes the covering stretch, then
  `recut.perform_recut` takes the pieces out of it, burns hook and captions on
  the concatenated timeline, records the clip's `recipe` (the editor's fast
  re-cut keeps working).
- Pasted transcripts: edges are carried onto Whisper's words first
  (`map_to_exact`), whole pasted lines are transcribed around each cut, and a
  cut that moves past what was heard triggers a second stretch.

**Why**: this is the missing half of "Claude picks the clips". Two bugs the
live runs caught: a clip end jumped onto the next word (a tie at a word
boundary, plus a zero-length Whisper word), and a cut started inside a word
(two Whisper runs disagreed by 0.45 s; the waveform decided it).

**Files**: `agent_clips.py` (new), `app.py`, `main.py`, `mcp_server.py`,
`transcript_import.py`, `CLAUDE.md`, tests (`test_agent_clips.py`,
`test_render_clips.py`, +`test_transcript_import.py`, `test_pasted_transcript.py`).

**Verified**: 1079 tests in the container, 954 + 35 skipped in a clean CI
container. Live through `/mcp`: a 4-piece 23.6 s clip, 172 s from call to
finished clip, captions following the reordered pieces; the same clip chosen on
pasted estimates lands within 0.06 s of the exact-transcript cut on 3 of 4
edges and keeps every piece's first and last word.

## 2026-09-15 — `679b5fe` fix(editor): a re-cut or re-frame burns the clip's hook back on

**What**: `recut.perform_recut(hook=)` burns the clip's recorded hook under the
captions; `/api/clip/rerender` and `/api/clip/reframe` take `reapply_hook`
(default true) and keep `auto_hook` only when the served file really has one;
`/api/hook` records the hook size.

**Why**: both editor rebuilds start from hook-less files, so every edit
silently dropped the hook while the metadata still claimed it. Needed for
clips made of pieces.

**Files**: `recut.py`, `app.py`, `hooks.py`, `mcp_server.py`,
`dashboard/src/App.jsx`, `CLAUDE.md`, `tests/test_hook_after_recut.py`.

**Verified**: 1018 tests, 898 + 30 skipped in CI. Live: a 5-piece re-cut came
back with the hook over its first 5 s (frames checked).

## 2026-09-15 — `43c8018` feat(mcp): get_transcript and the clip picking rules

**What**: `GET /api/transcript/{job_id}` and MCP `get_transcript`. Sentences as
`[start-end] text` in pages of 60,000 characters; `words=true` returns up to
300 s word by word for exact cuts; `timing` flags a pasted transcript. The
first page carries `clip_rules.md` (the user edits it; read on every call) and
the job's "what to clip" instructions.

**Why**: the agent needs the transcript with times to choose clips, and a
stable rule set so every chat picks the same way.

**Files**: `app.py`, `mcp_server.py`, `clip_rules.md` (new), `.dockerignore`,
`CLAUDE.md`, `tests/test_get_transcript.py`.

**Verified**: 1000 tests, 881 + 29 skipped in CI. Live: the 50-min democracy
job is one page of 63,798 characters (624 sentences) in 0.15 s.

## 2026-09-15 — `1bc98d3` + `3212424` feat(transcript): paste a transcript to skip transcribing

**What**: a `transcript` field on `/api/process`, an "I have the transcript"
box in the dashboard, and a `transcript` option in MCP `process_video`. New
`transcript_import.py` reads YouTube's "Show transcript" copy, SRT, WebVTT
(including word-timed auto-captions) and OpenShorts JSON, and rejects an
unreadable paste or one belonging to another video at submit.
`main.refine_pasted_transcript` then runs Whisper on the chosen clips only
(8 s either side) and cuts them again on those exact words.

**Why**: transcription is the slow part of a job, and YouTube already has a
transcript. But a pasted transcript's word times are estimates: on a simulated
panel copy, 33-43% of clips cut off their last word, so the paste chooses and
Whisper still cuts.

**Files**: `transcript_import.py` (new), `app.py`, `main.py`,
`transcribe_backends.py`, `mcp_server.py`, `dashboard/src/App.jsx`,
`dashboard/src/components/MediaInput.jsx`, `CLAUDE.md`, tests
(`test_transcript_import.py`, `test_pasted_transcript.py`).

**Verified**: 982 tests, 863 + 29 skipped in CI, dashboard lint and build.
Live: a 3.1-min upload's cut on estimates ended inside a word; after Whisper on
0.9 of the 3.1 min it ended on the finished sentence. An agent job with a
transcript finished in 8 s instead of 93 s.

## 2026-09-15 — `85b8bc5` feat(mcp): selection=agent jobs only download and transcribe

**What**: `selection: "ai" | "agent"` on `/api/process` and MCP
`process_video`. An agent job downloads, transcribes and stops
(`main.py --transcribe-only`): no layout pick, no model call, no Gemini key
needed. `awaiting_clips` rides the job result, the webhook and
`get_job_status`; the requested look is saved for the render job.

**Why**: the first half of letting the agent choose the clips.

**Files**: `app.py`, `main.py`, `mcp_server.py`, `CLAUDE.md`,
`tests/test_agent_selection.py`.

**Verified**: 924 tests, 811 + 23 skipped in CI. Live: a 3.1-min upload
reached `awaiting_clips` 93 s after submit; a video without speech failed with
a message naming the fix.

## 2026-09-15 — `bcf9d3c` feat(clip-selection): pass 2 reads whole timed sentences

**What**: the second Gemini pass reads `[start-end] sentence` lines and takes
the clip's start and end straight from them, and must end on a finished
thought.

**Why**: it used to close on a Whisper line, and 44-57% of those end
mid-sentence.

**Files**: `main.py`, `gemini_worker.py`, `clip_selection.py`,
`tools/compare_selection.py`, `CLAUDE.md`, `tests/test_sentence_cuts.py`.

**Verified**: live 55-min runs ×2: the model's own end lands on a finished
sentence 53% → 81%, cost unchanged.

## 2026-09-15 — `672c4d5` fix(clip-selection): cut clips on whole sentences

**What**: `clip_selection.snap_clip_to_sentences` replaces word snapping in
`get_viral_clips`: an end finishes its sentence (≤ 8 s), a start only moves
earlier, the length band is kept.

**Why**: the user's job ended 4 of 6 clips mid-statement; only 23% of 73 saved
clip answers ended on a finished sentence. The cause was mechanical, not the
model's judgement.

**Files**: `clip_selection.py`, `main.py`, `tools/compare_selection.py`,
`CLAUDE.md`, `tests/test_sentence_cuts.py`.

**Verified**: replaying 73 saved answers offline: finished endings 16% → 100%.
The user watched the re-cut clips and called the endings good.

---

## Notes for the next session

- **Restarting the backend kills a running job.** Check first: scan
  `/proc/*/cmdline` in the container for an argv element ending in `main.py`
  (the container has no `ps`). One of the user's jobs was lost this way on
  15-sep; a separate task is queued for the underlying resume bug (a job
  interrupted mid-render is recovered as "completed" with no clips).
- **Claude Code caches the MCP tool list at session start**; new tools need a
  reconnect or a new session.
- Working rules: one step at a time, tests and real measurements reported per
  step, a commit per step, and push only when the user asks.
