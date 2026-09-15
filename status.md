# Status

A running log of what changed, when and why. Newest first. `CLAUDE.md` says how
the app works today; this file says what moved and what it measured.

Add an entry per commit: date, commit, what changed, why, files, how it was
verified.

## Where the work stands

Goal: give Claude a link in chat and get clips back, with Claude choosing the
moments instead of Gemini (MCP). Steps 1-4 are done and work end to end
through `/mcp`. **Nothing below is pushed**: 8 local commits, `672c4d5`..`350ef48`.

Left to do:
- **3.5**: the real end-to-end run from the chat (not curl), then Gemini's
  clips vs Claude's on the same video, judged by the user.
- Open question for the user: which video for that comparison. Sources kept in
  `.cache/headtohead/` (Jake Paul job `216fe922`, democracy job `c4e6388d`)
  because `output/` deletes jobs after 24 h.
- `docs/how-openshorts-works.md` is written but not committed.

---

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
