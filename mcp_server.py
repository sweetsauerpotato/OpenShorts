"""Remote MCP server: the whole pipeline as agent-callable tools at ``/mcp``.

Stateless Streamable-HTTP transport (plain JSON responses, no SSE, no session
ids) implemented directly over FastAPI — the protocol surface an MCP client
actually needs for tools is three methods (initialize / tools/list / tools/call)
plus notifications, and owning those ~200 lines beats carrying the full SDK as
a dependency for them.

Tools don't reimplement anything: each one is an in-process HTTP call back into
this same app (httpx ASGITransport) with the caller's auth headers forwarded.
Cloud mode therefore meters, gates and scopes agent calls exactly like
dashboard calls — an ``osk_`` API key IS the user (see cloud/api_keys.py).
Self-host keeps its BYOK semantics: no key required, ``X-Gemini-Key`` /
env fallbacks apply unchanged.

Connect with any MCP client, e.g.:
    claude mcp add --transport http openshorts https://mcp.openshorts.app/mcp \
        --header "Authorization: Bearer osk_..."

(mcp.openshorts.app is a domain alias of the API app; api.openshorts.app/mcp
serves the identical endpoint.)
"""
import json
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

import mcp_ui

router = APIRouter()

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "openshorts", "title": "OpenShorts", "version": "1.0.0"}
INSTRUCTIONS = (
    "OpenShorts turns long videos (YouTube URLs or direct video files) into "
    "viral-ready vertical clips. When the user gives you a video URL, hand it "
    "to process_video exactly as written: OpenShorts downloads, transcribes "
    "and analyses the video on its own servers. Do NOT try to open, fetch, "
    "search for, summarise or transcribe the URL yourself first; you cannot "
    "reach the video and it is not needed. Typical flow: process_video -> "
    "poll get_job_status until 'completed' (a job takes minutes; poll every "
    "30-60s or pass webhook_url) -> list_clips -> optionally add_subtitles / "
    "recut_clip / publish_clip. Check get_quota before large jobs. The user "
    "must own the content or hold the rights: ask once, then pass "
    "confirm_rights=true. To choose the clips yourself instead of OpenShorts' "
    "AI: process_video with selection='agent' (add the user's transcript if "
    "they have one) -> get_job_status until awaiting_clips -> get_transcript "
    "(follow its rules) -> render_clips -> get_job_status on the new job -> "
    "list_clips."
)

# Headers an MCP caller may use to authenticate / bring their own keys; they are
# forwarded verbatim to the internal endpoints so every existing auth path works.
_FORWARD_HEADERS = ("authorization", "x-api-key", "x-gemini-key", "x-upload-post-key")

_LOG_TAIL = 10  # status logs are for humans; agents only need the tail


TOOLS = [
    {
        "name": "process_video",
        "title": "Process a video into short clips",
        "description": (
            "Start clipping a video from its URL. OpenShorts downloads the "
            "source itself, transcribes it, finds the most viral moments with AI "
            "and renders vertical (9:16) clips. Captions and the AI hook line are "
            "burned by default; pass captions=false or auto_hook=false to skip either. "
            "Call this directly "
            "with the URL the user gave you; do not fetch, search or inspect the "
            "URL yourself first (you cannot access the video, and it is not "
            "needed). Returns a job_id immediately — the work takes minutes; "
            "poll get_job_status or pass webhook_url to be called back. The "
            "caller must own the content or hold the rights to process it "
            "(confirm_rights)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_url": {
                    "type": "string",
                    "description": "Public video URL, passed through exactly as the user gave it "
                                   "(YouTube watch/short/live URL, a direct video file URL, or a "
                                   "tmpfiles.org link from the create_upload fallback). The server "
                                   "does the downloading. Omit when using upload_id.",
                },
                "upload_id": {
                    "type": "string",
                    "description": "Instead of source_url: the id from create_upload after the "
                                   "file was PUT to its upload_url. Use when the user gave you a "
                                   "video file rather than a link.",
                },
                "auto_hook": {
                    "type": "boolean",
                    "description": "Burn the AI-written hook line (the clip's title) over the first "
                                   "seconds of each clip, as the dashboard does. Default true; set "
                                   "false for clean clips.",
                },
                "hook_style": {
                    "type": "string",
                    "enum": ["classic", "dark", "yellow", "red", "outline", "outline_yellow"],
                    "description": "Look of the hook text (with auto_hook). Default classic.",
                },
                "captions": {
                    "type": "boolean",
                    "description": "Default true: burn word-level captions on every clip. Set false "
                                   "when the source already has subtitles burned in (they would "
                                   "stack) or the user wants clean clips; add_subtitles can still "
                                   "caption a clip later.",
                },
                "confirm_rights": {
                    "type": "boolean",
                    "description": "Must be true: the user owns the content or has rights to process it.",
                },
                "layouts": {
                    "type": "array",
                    "items": {"type": "string",
                              "enum": ["auto", "split", "screencast", "speaker_cut", "punch_in"]},
                    "description": "Optional extra reframe layouts. 'auto' lets AI pick per video.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["auto", "vertical", "horizontal", "square"],
                    "description": "Clip aspect. Default auto (vertical).",
                },
                "webhook_url": {
                    "type": "string",
                    "description": "Optional public HTTPS URL POSTed once when the job finishes or fails.",
                },
                "webhook_secret": {
                    "type": "string",
                    "description": "Optional secret; the webhook body is then HMAC-SHA256 signed (X-OpenShorts-Signature).",
                },
                "force_low_quality": {
                    "type": "boolean",
                    "description": "Set true to proceed after a needs_confirmation low-resolution warning.",
                },
                "target_clips": {
                    "type": "integer", "minimum": 1, "maximum": 15,
                    "description": "How many clips to aim for. A target, not a guarantee: "
                                   "fewer come back when the material doesn't hold them. "
                                   "Default: the AI decides (usually 2-6).",
                },
                "clip_min_seconds": {
                    "type": "number", "minimum": 5, "maximum": 175,
                    "description": "Minimum clip length in seconds (default 15).",
                },
                "clip_max_seconds": {
                    "type": "number", "minimum": 10, "maximum": 180,
                    "description": "Maximum clip length in seconds (default 60). Must be ≥ 5s above the minimum.",
                },
                "clip_instructions": {
                    "type": "string", "maxLength": 1000,
                    "description": "Optional direction for choosing clips, in plain words: what to "
                                   "look for and what to skip (e.g. 'only moments where the guest "
                                   "gives a concrete tip; skip the sponsor read'). Steers every "
                                   "selection stage; 'only'/'never' are treated as hard limits, and "
                                   "fewer clips come back when few moments fit. Cannot change clip "
                                   "length rules.",
                },
                "transcript": {
                    "type": "string", "maxLength": 300000,
                    "description": "Optional: the video's transcript WITH its timestamps, when the user "
                                   "has one, passed exactly as they gave it. Accepted: YouTube's 'Show "
                                   "transcript' list copied as is ('1:02' then the words), SRT, WebVTT "
                                   "(YouTube's auto-caption VTT too), or OpenShorts transcript JSON. "
                                   "The whole video is then never transcribed: the clips are chosen "
                                   "from this transcript and only the chosen clips are transcribed, "
                                   "for exact cuts and captions. A transcript without timestamps is "
                                   "rejected.",
                },
                "selection": {
                    "type": "string", "enum": ["ai", "agent"],
                    "description": "Who chooses the clips. 'ai' (default): OpenShorts picks and renders "
                                   "them. 'agent': OpenShorts only downloads and transcribes; when "
                                   "get_job_status reports awaiting_clips, read the transcript with "
                                   "get_transcript, choose the clips yourself and send them with "
                                   "render_clips. target_clips and the clip length options do not "
                                   "apply; auto_hook, hook_style, captions, layouts and output_format "
                                   "are kept for that render.",
                },
                "quality": {
                    "type": "integer", "enum": [360, 480, 720, 1080, 1440, 2160],
                    "description": "source_url only: download up to this resolution (pixels tall). "
                                   "Default 1080. A video that tops out lower gets its best. Above 360 "
                                   "the job stops rather than fall back to YouTube's 360p copy. "
                                   "1440/2160 are much larger downloads.",
                },
            },
            "required": ["confirm_rights"],
        },
    },
    {
        "name": "create_upload",
        "title": "Reserve an upload slot for a local video file",
        "description": (
            "Use when the user hands you a video FILE instead of a link. Returns "
            "an upload_url: send the file's raw bytes to it with an HTTP PUT "
            "(for example `curl -T video.mp4 <upload_url>`), then call "
            "process_video with the returned upload_id. Unused slots are "
            "deleted after expires_in seconds (6 h); max size is max_mb. "
            "FALLBACK if your sandbox cannot reach upload_url: upload the file "
            "to tmpfiles.org (`curl -F file=@video.mp4 https://tmpfiles.org/api/v1/upload`), "
            "and pass the returned url to process_video as source_url right "
            "away (their files expire after 60 minutes; the server refreshes "
            "the signed download link itself). Otherwise ask the user to run "
            "the curl command or to share a public link."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Original file name (optional, for the extension)."},
            },
        },
    },
    {
        "name": "get_job_status",
        "title": "Get processing job status",
        "description": (
            "Status of a processing job: 'queued', 'processing', 'completed' or "
            "'failed', with recent log lines and, once completed, the clips."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "list_clips",
        "title": "List a job's clips",
        "description": (
            "The clips of a completed job, with titles, platform-ready "
            "descriptions and download URLs. In MCP Apps-capable clients the "
            "result also renders as an interactive clip picker."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        # Both spellings of the tool->template link: MCP Apps hosts read
        # _meta.ui.resourceUri, the ChatGPT Apps SDK reads openai/outputTemplate.
        "_meta": {
            "ui": {"resourceUri": mcp_ui.CLIP_PICKER_URI},
            "openai/outputTemplate": mcp_ui.CLIP_PICKER_URI,
        },
    },
    {
        "name": "get_transcript",
        "title": "Read a job's transcript to choose clips",
        "description": (
            "The transcript of a job's video, to choose the clips yourself: after "
            "process_video with selection='agent' reports awaiting_clips (any finished "
            "job works). With only job_id: every sentence as '[start-end] text' in "
            "seconds of the source video, plus the clip picking rules and the "
            "creator's instructions to follow. Long videos come in pages: call again "
            "with start=next_start until next_start is null. With start, end and "
            "words=true (at most 300 s apart): that stretch word by word as "
            "[start, end, word], to place exact cuts, drop filler or pauses, or lift "
            "a punchline to the front. timing='estimated' means the times come from "
            "a pasted transcript: good enough to choose; the render transcribes the "
            "chosen clips and cuts on exact words."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "start": {"type": "number",
                          "description": "Seconds: begin here (a page's next_start, or a stretch to read word by word)."},
                "end": {"type": "number", "description": "Seconds: stop here."},
                "words": {"type": "boolean",
                          "description": "Word by word instead of sentences; needs start and end, at most 300 s apart."},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "render_clips",
        "title": "Render the clips you chose",
        "description": (
            "Render your own choice of clips from a job's video: after get_transcript on a job "
            "from process_video with selection='agent' (any finished job whose video is still "
            "on the server works). Each clip is one or more pieces of the source in play order, "
            "so it can open on its punchline and leave dead parts out, plus the hook text burned "
            "over its first seconds, the title and the descriptions. OpenShorts cuts, reframes "
            "and captions them like its own clips; the hook style, captions, layout and format "
            "of the process_video call are kept. Put piece edges in the silence between words "
            "(get_transcript with words=true); each edge is moved into the gap beside its word, "
            "and with a pasted transcript the pieces are transcribed first and cut on the exact "
            "words. A clip's pieces must lie within 180 s of the source. Returns a new job_id: "
            "poll get_job_status (renders take minutes, ~4 per clip on a CPU), then list_clips."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string",
                           "description": "The job whose video and transcript the clips come from."},
                "clips": {
                    "type": "array", "minItems": 1, "maxItems": 15,
                    "items": {
                        "type": "object",
                        "properties": {
                            "segments": {
                                "type": "array", "minItems": 1, "maxItems": 12,
                                "items": {"type": "object",
                                          "properties": {"start": {"type": "number"},
                                                         "end": {"type": "number"}},
                                          "required": ["start", "end"]},
                                "description": "Pieces in play order, seconds in the source video.",
                            },
                            "hook": {"type": "string", "maxLength": 150,
                                     "description": "On-screen hook for the first seconds (max 10 words)."},
                            "title": {"type": "string", "maxLength": 100,
                                      "description": "YouTube Shorts title."},
                            "tiktok_description": {"type": "string", "maxLength": 2200},
                            "instagram_description": {"type": "string", "maxLength": 2200},
                            "score": {"type": "number", "minimum": 0, "maximum": 100,
                                      "description": "Your 0-100 score on the picking rules' scale."},
                            "reason": {"type": "string", "maxLength": 500,
                                       "description": "Why it hooks, one line."},
                        },
                        "required": ["segments"],
                    },
                },
            },
            "required": ["job_id", "clips"],
        },
    },
    {
        "name": "rate_clip",
        "title": "Record what the user thought of a clip",
        "description": (
            "Record the user's verdict on one clip, so clip selection can be "
            "measured and improved. Call it when the user says a clip is good "
            "or bad -- never guess a verdict for them. 'bad' takes an optional "
            "reason from a closed list: no_payoff, needs_context, "
            "nothing_to_watch, boring, wrong_moment, bad_cut. Re-rating the "
            "same clip replaces the previous verdict. Returns the running "
            "summary, including score_split -- the mean predicted_score of "
            "clips the user liked against the ones they did not."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "clip_index": {"type": "number",
                               "description": "0-based index from list_clips."},
                "verdict": {"type": "string", "enum": ["good", "bad"]},
                "reason": {
                    "type": "string",
                    "enum": ["no_payoff", "needs_context", "nothing_to_watch",
                             "boring", "wrong_moment", "bad_cut"],
                    "description": "Why it was not worth posting. For 'bad'.",
                },
            },
            "required": ["job_id", "clip_index", "verdict"],
        },
    },
    {
        "name": "get_quota",
        "title": "Get plan and remaining minutes",
        "description": (
            "The authenticated user's plan and remaining processing minutes. "
            "Call before large jobs; process_video fails with quota_exceeded "
            "when the balance is insufficient."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_subtitles",
        "title": "Burn styled captions onto a clip",
        "description": (
            "Re-style the captions of one clip (clips already ship with default "
            "captions). style 'karaoke' highlights the active word."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "clip_index": {"type": "integer", "description": "0-based index from list_clips."},
                "style": {"type": "string", "enum": ["classic", "karaoke"]},
                "position": {"type": "string", "enum": ["top", "middle", "bottom"]},
                "font_size": {"type": "integer"},
                "font_name": {"type": "string"},
                "font_color": {"type": "string", "description": "Hex color, e.g. #FFFFFF."},
                "highlight_color": {"type": "string", "description": "Karaoke active-word color."},
                "uppercase": {"type": "boolean"},
            },
            "required": ["job_id", "clip_index"],
        },
    },
    {
        "name": "recut_clip",
        "title": "Re-cut a clip from an edited segment list",
        "description": (
            "Re-render one clip from a new list of source-video segments (the "
            "same engine behind the dashboard's clip editor). Times are seconds "
            "in the ORIGINAL source video; segments are concatenated in the "
            "given order, so you can trim, extend, drop a dead moment in the "
            "middle, or reorder. Segments inside the clip's original range "
            "re-render in seconds; going outside needs the retained source "
            "video and re-runs the reframe engine. The clip's hook text and "
            "captions are burned back on unless reapply_hook / "
            "reapply_captions is false."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "clip_index": {"type": "integer", "description": "0-based index from list_clips."},
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number", "description": "Seconds in the source video."},
                            "end": {"type": "number", "description": "Seconds in the source video."},
                        },
                        "required": ["start", "end"],
                    },
                    "minItems": 1,
                    "description": "Ordered source segments the new clip is made of.",
                },
                "snap_to_words": {
                    "type": "boolean",
                    "description": "Snap each boundary onto transcript word boundaries (recommended).",
                },
                "reapply_captions": {
                    "type": "boolean",
                    "description": "Burn default captions back on after the recut (default true).",
                },
                "reapply_hook": {
                    "type": "boolean",
                    "description": "Burn the clip's hook text back on after the recut (default true); "
                                   "false leaves the recut without a hook.",
                },
                "framing": {
                    "type": "string", "enum": ["auto", "full", "track"],
                    "description": "Layout override: 'full' shows the whole source frame "
                                   "(no side-cropping), 'track' forces the subject-tracking "
                                   "crop, 'auto' resets to the AI classifier. Omit to keep "
                                   "the clip's current framing. Non-auto values re-run the "
                                   "reframe engine and need the retained source video.",
                },
            },
            "required": ["job_id", "clip_index", "segments"],
        },
    },
    {
        "name": "publish_clip",
        "title": "Publish a clip to social platforms",
        "description": (
            "Post one clip to the user's connected accounts (TikTok lands as a "
            "draft in the app; Instagram and YouTube publish directly). Requires "
            "a connected social profile (cloud) or an Upload-Post key (self-host). "
            "Optionally schedule with an ISO-8601 scheduled_date."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "clip_index": {"type": "integer"},
                "platforms": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["tiktok", "instagram", "youtube"]},
                },
                "title": {"type": "string"},
                "description": {"type": "string"},
                "scheduled_date": {"type": "string", "description": "ISO-8601; omit to post now."},
                "timezone": {"type": "string"},
            },
            "required": ["job_id", "clip_index", "platforms"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Internal dispatch: each tool is an in-process call to the existing REST API.
# --------------------------------------------------------------------------- #
def _client(request: Request) -> httpx.AsyncClient:
    headers = {k: v for k, v in request.headers.items() if k.lower() in _FORWARD_HEADERS}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=request.app, raise_app_exceptions=False),
        base_url="http://openshorts.internal",
        headers=headers,
        timeout=300.0,
    )


def _api_error(resp: httpx.Response) -> dict:
    try:
        detail = resp.json().get("detail")
    except Exception:
        detail = resp.text[:300]
    return {"error": detail or f"HTTP {resp.status_code}", "http_status": resp.status_code}


async def _tool_process_video(client, args):
    if not args.get("confirm_rights"):
        return {"error": "confirm_rights must be true: the user must own the "
                         "content or hold the rights to process it."}, True
    if not args.get("source_url") and not args.get("upload_id"):
        return {"error": "Give source_url (a public video link) or upload_id (from create_upload)."}, True
    body = {
        "url": args.get("source_url"),
        "upload_id": args.get("upload_id"),
        "acknowledged": True,
        "layouts": args.get("layouts") or [],
        "output_format": args.get("output_format"),
        "force_low_quality": bool(args.get("force_low_quality")),
        "webhook_url": args.get("webhook_url"),
        "webhook_secret": args.get("webhook_secret"),
    }
    for k in ("target_clips", "clip_min_seconds", "clip_max_seconds", "captions", "quality",
              "clip_instructions", "selection", "transcript"):
        if args.get(k) is not None:
            body[k] = args[k]
    # Same default as the dashboard: hook on unless the caller opts out. The
    # REST endpoint keeps "absent = off" for older integrations; the MCP
    # tool is newer than the hook and its users expect the dashboard output.
    if args.get("auto_hook", True):
        body["auto_hook"] = True
        if args.get("hook_style"):
            body["auto_hook_style"] = args["hook_style"]
    resp = await client.post("/api/process", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    data = resp.json()
    if data.get("needs_confirmation"):
        data["hint"] = ("Source resolution is below the quality gate. Ask the "
                        "user, then retry with force_low_quality=true to proceed.")
        return data, False
    if body.get("selection") == "agent":
        data["hint"] = ("Downloading and transcribing only; this takes minutes. Poll "
                        "get_job_status every 30-60s until it reports awaiting_clips, then "
                        "call get_transcript.")
        return data, False
    data["hint"] = ("Processing takes minutes. Poll get_job_status every 30-60s"
                    + ("" if body["webhook_url"] else " (or re-run with webhook_url for a callback)") + ".")
    return data, False


async def _tool_create_upload(client, args):
    resp = await client.post("/api/uploads", json={"filename": args.get("filename") or "video.mp4"})
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


async def _tool_get_job_status(client, args):
    resp = await client.get(f"/api/status/{args['job_id']}")
    if resp.status_code >= 400:
        return _api_error(resp), True
    data = resp.json()
    out = {"job_id": args["job_id"], "status": data.get("status"),
           "recent_logs": (data.get("logs") or [])[-_LOG_TAIL:]}
    if data.get("status") == "completed":
        result = data.get("result") or {}
        if result.get("awaiting_clips"):
            out["awaiting_clips"] = True
            out["hint"] = ("Transcribed, no clips yet (selection=agent): call get_transcript, "
                           "choose the clips, then send them with render_clips.")
        else:
            out["clips"] = _clip_summaries(args["job_id"], result)
    return out, data.get("status") == "failed"


def _clip_summaries(job_id, result):
    base = os.environ.get("PUBLIC_API_URL", "").rstrip("/")
    out = []
    for i, clip in enumerate(result.get("clips") or []):
        rel = clip.get("video_url") or ""
        timed = (isinstance(clip.get("start"), (int, float))
                 and isinstance(clip.get("end"), (int, float)))
        # The clip's timeline: an edited or agent clip's pieces, else its stretch.
        segments = (clip.get("recipe") or {}).get("segments") or (
            [{"start": clip["start"], "end": clip["end"]}] if timed else [])
        out.append({
            "index": i,
            "title": clip.get("title") or clip.get("video_title_for_youtube_short"),
            "duration_seconds": (round(sum(s["end"] - s["start"] for s in segments), 1)
                                 if segments else None),
            "segments": segments,
            "hook": (clip.get("auto_hook") or {}).get("text") or clip.get("viral_hook_text"),
            "score": clip.get("predicted_score"),
            "reason": clip.get("reason") or None,
            "video_url": f"{base}{rel}" if base and rel.startswith("/") else rel,
            "youtube_title": clip.get("video_title_for_youtube_short"),
            "tiktok_description": clip.get("video_description_for_tiktok"),
            "instagram_description": clip.get("video_description_for_instagram"),
        })
    return out


async def _tool_list_clips(client, args):
    out, is_error = await _tool_get_job_status(client, args)
    if is_error:
        return out, True
    if out.get("status") != "completed":
        return {"error": f"Job is {out.get('status')}, clips are not ready yet.",
                "status": out.get("status")}, True
    if out.get("awaiting_clips"):
        return {"error": "This job only transcribed (selection=agent), so it has no clips yet: "
                         "call get_transcript, choose the clips, then send them with "
                         "render_clips.", "awaiting_clips": True}, True
    return {"job_id": args["job_id"], "clips": out.get("clips") or []}, False


async def _tool_get_transcript(client, args):
    params = {k: args[k] for k in ("start", "end") if args.get(k) is not None}
    if args.get("words"):
        params["words"] = "true"
    resp = await client.get(f"/api/transcript/{args['job_id']}", params=params)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


async def _tool_render_clips(client, args):
    # The rights were confirmed when the source job was submitted; this renders
    # parts of that same video.
    resp = await client.post("/api/process", json={
        "source_job_id": args["job_id"], "clips": args["clips"], "acknowledged": True})
    if resp.status_code >= 400:
        return _api_error(resp), True
    data = resp.json()
    data["clips"] = len(args["clips"])
    data["hint"] = ("Rendering takes minutes (~4 per clip on a CPU). Poll get_job_status on this "
                    "job_id, then list_clips.")
    return data, False


async def _tool_rate_clip(client, args):
    body = {"job_id": args.get("job_id"), "clip_index": args.get("clip_index"),
            "verdict": args.get("verdict")}
    if args.get("reason"):
        body["reason"] = args["reason"]
    resp = await client.post("/api/verdict", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


async def _tool_get_quota(client, args):
    resp = await client.get("/api/me")
    # 401: anonymous. 404: self-host, where /api/me isn't even mounted (the
    # cloud router only registers under BILLING_ENABLED). Neither is an error
    # from the agent's point of view — there is simply no quota to report.
    if resp.status_code in (401, 404):
        return {"self_host_or_anonymous": True,
                "note": "No authenticated cloud user; if this is a self-hosted "
                        "instance there is no minute quota."}, False
    if resp.status_code >= 400:
        return _api_error(resp), True
    data = resp.json()
    return {"plan": data.get("plan"), "entitled": data.get("entitled"),
            "minutes": data.get("minutes"),
            "upload_post_profile": data.get("upload_post_profile")}, False


async def _tool_add_subtitles(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"]}
    for k in ("style", "position", "font_size", "font_name", "font_color",
              "highlight_color", "uppercase"):
        if args.get(k) is not None:
            body[k] = args[k]
    resp = await client.post("/api/subtitle", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


async def _tool_recut_clip(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"],
            "segments": args["segments"]}
    for k in ("snap_to_words", "reapply_captions", "reapply_hook", "framing"):
        if args.get(k) is not None:
            body[k] = args[k]
    resp = await client.post("/api/clip/rerender", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


async def _tool_publish_clip(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"],
            "platforms": args["platforms"]}
    for k in ("title", "description", "scheduled_date", "timezone"):
        if args.get(k) is not None:
            body[k] = args[k]
    resp = await client.post("/api/social/post", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


_TOOL_IMPLS = {
    "process_video": _tool_process_video,
    "create_upload": _tool_create_upload,
    "get_job_status": _tool_get_job_status,
    "list_clips": _tool_list_clips,
    "get_transcript": _tool_get_transcript,
    "render_clips": _tool_render_clips,
    "rate_clip": _tool_rate_clip,
    "get_quota": _tool_get_quota,
    "add_subtitles": _tool_add_subtitles,
    "recut_clip": _tool_recut_clip,
    "publish_clip": _tool_publish_clip,
}


async def call_tool(request: Request, name: str, args: dict) -> tuple[dict, bool]:
    impl = _TOOL_IMPLS.get(name)
    if impl is None:
        return {"error": f"Unknown tool: {name}"}, True
    try:
        async with _client(request) as client:
            return await impl(client, args or {})
    except KeyError as e:
        return {"error": f"Missing required argument: {e}"}, True
    except Exception as e:
        return {"error": f"Tool failed: {e}"}, True


# --------------------------------------------------------------------------- #
# JSON-RPC / MCP protocol layer (pure: testable without the app)
# --------------------------------------------------------------------------- #
def _rpc_error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _rpc_result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


async def handle_message(msg, tool_caller) -> Optional[dict]:
    """One JSON-RPC message in, one response dict out (None for notifications).

    ``tool_caller(name, args) -> (result_dict, is_error)`` is injected so this
    layer stays free of HTTP and app state.
    """
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _rpc_error(None, -32600, "Expected a JSON-RPC 2.0 message")
    method = msg.get("method")
    msg_id = msg.get("id")

    if method is None:
        # A response from the client (has 'result'/'error') — nothing to do.
        return None
    if msg_id is None:
        return None  # notification (e.g. notifications/initialized): accept silently

    if method == "initialize":
        client_version = (msg.get("params") or {}).get("protocolVersion")
        version = client_version if client_version in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _rpc_result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False},
                             "resources": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return _rpc_result(msg_id, {})
    if method == "tools/list":
        return _rpc_result(msg_id, {"tools": TOOLS})
    if method == "resources/list":
        return _rpc_result(msg_id, {"resources": mcp_ui.RESOURCES})
    if method == "resources/templates/list":
        return _rpc_result(msg_id, {"resourceTemplates": []})
    if method == "resources/read":
        uri = (msg.get("params") or {}).get("uri") or ""
        # Per-call URIs (ui://openshorts/clip-picker/<job>) resolve to the same
        # template; the data those carried was baked into the tool result.
        if uri == mcp_ui.CLIP_PICKER_URI or uri.startswith(mcp_ui.CLIP_PICKER_URI + "/"):
            return _rpc_result(msg_id, {"contents": [{
                "uri": uri,
                "mimeType": mcp_ui.MIME_TYPE,
                "text": mcp_ui.clip_picker_html(),
            }]})
        return _rpc_error(msg_id, -32002, f"Resource not found: {uri}")
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        result, is_error = await tool_caller(name, params.get("arguments") or {})
        content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
        # A successful list_clips additionally ships the picker with its data
        # baked in, so hosts that render embedded resources need no bridge.
        # Non-UI clients ignore extra content entries.
        if name == "list_clips" and not is_error and result.get("clips"):
            content.append({"type": "resource", "resource": {
                "uri": f"{mcp_ui.CLIP_PICKER_URI}/{result.get('job_id', 'result')}",
                "mimeType": mcp_ui.MIME_TYPE,
                "text": mcp_ui.clip_picker_html(result),
            }})
        return _rpc_result(msg_id, {
            "content": content,
            "structuredContent": result,
            "isError": is_error,
        })
    return _rpc_error(msg_id, -32601, f"Method not found: {method}")


# --------------------------------------------------------------------------- #
# Transport endpoints
# --------------------------------------------------------------------------- #
def _billing_enabled() -> bool:
    return os.environ.get("BILLING_ENABLED", "").lower() in ("1", "true", "yes")


async def _authorized(request: Request) -> bool:
    """Cloud mode requires a resolvable user (API key or JWT) before any RPC.

    The internal endpoints would each reject anonymous calls anyway; failing
    once here with a clear 401 is what lets MCP clients surface 'add your API
    key' instead of a per-tool 402. Self-host stays open (BYOK)."""
    if not _billing_enabled():
        return True
    from cloud.auth import get_current_user_optional
    return (await get_current_user_optional(request)) is not None


@router.post("/mcp")
async def mcp_endpoint(request: Request):
    if not await _authorized(request):
        # OAuth-capable clients (claude.ai, ChatGPT) read resource_metadata off
        # this header and run the login flow themselves; everyone else gets the
        # API-key hint in the body.
        from cloud import mcp_oauth
        u = request.base_url
        return JSONResponse(
            {"error": "Authentication required. Connect with OAuth (claude.ai, ChatGPT) "
                      "or pass an OpenShorts API key: Authorization: Bearer osk_... "
                      "(create one in the dashboard)."},
            status_code=401,
            headers={"WWW-Authenticate": mcp_oauth.www_authenticate(f"{u.scheme}://{u.netloc}")},
        )
    try:
        msg = json.loads(await request.body())
    except Exception:
        return JSONResponse(_rpc_error(None, -32700, "Parse error"), status_code=400)
    if isinstance(msg, list):
        return JSONResponse(_rpc_error(None, -32600, "Batching is not supported"),
                            status_code=400)

    async def tool_caller(name, args):
        return await call_tool(request, name, args)

    response = await handle_message(msg, tool_caller)
    if response is None:
        return Response(status_code=202)
    return JSONResponse(response)


@router.get("/mcp")
async def mcp_get():
    # Stateless server: no server-initiated SSE stream to offer.
    return Response(status_code=405, headers={"Allow": "POST"})
