"""What counts as a good moment in this KIND of video.

Three layers decide what gets clipped, and they are deliberately separate:

  1. ``clip_rules.md``      — what makes any short work. Global, rarely edited.
  2. a niche (this module) — what works in THIS kind of content. Named,
                             reusable, versioned, improvable from verdict data.
  3. ``clip_instructions`` — what the creator wants from THIS video. Free text,
                             per job, and it wins over both.

The middle layer was missing. Free text in the box is per-video and disappears
with it, so the same guidance had to be retyped and could never be improved
from evidence; a niche is a file we can rewrite when `verdicts.summarise` shows
which reason keeps coming back.

The precedent for expecting this to work is measured: a 56-character
instruction moved on-topic clips from 1 of 6 to 3 of 3 in both runs, and made
the job *cheaper* (1.51 -> 1.46 cents), because fewer clips means less pass-2
output at 6x the input price. A niche costs a similar handful of input tokens.

Niches are FILES, not constants, for the same reason ``clip_rules.md`` is: they
are content the user edits, and a prompt change should not need a deploy.
Read fresh on every call — the file is a few hundred bytes, and a stale cache
would silently keep serving the version someone just rewrote.

Standard library only.
"""

import os
import re

NICHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "niches")

# A niche name reaches the filesystem, so it is restricted to a safe shape
# rather than sanitised: "../../etc/passwd" is not a typo to fix, it is a
# request to refuse.
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,30}$")

MAX_NICHE_CHARS = 4000


class NicheError(ValueError):
    """Unknown or unusable niche: safe to show as a 400."""


def available():
    """Niche names on disk, sorted. Never raises."""
    try:
        return sorted(
            f[:-3] for f in os.listdir(NICHE_DIR)
            if f.endswith(".md") and _NAME.match(f[:-3]))
    except OSError:
        return []


def load(name):
    """The text of a niche, or None when ``name`` is blank.

    Raises NicheError for a name that is not on disk, so a typo in the API is a
    400 that lists the real ones instead of a job that silently ran unguided.
    """
    key = str(name or "").strip().lower()
    if not key:
        return None
    if not _NAME.match(key):
        raise NicheError(
            f"niche '{name}' is not a valid name; choose one of: "
            f"{', '.join(available()) or '(none installed)'}")
    path = os.path.join(NICHE_DIR, key + ".md")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        raise NicheError(
            f"unknown niche '{key}'; choose one of: "
            f"{', '.join(available()) or '(none installed)'}")
    text = _clean(text)
    if not text:
        raise NicheError(f"niche '{key}' is empty")
    return text[:MAX_NICHE_CHARS]


def _clean(text):
    """Strip control characters and the delimiter the block uses."""
    text = "".join(ch for ch in (text or "") if ch == "\n" or ch >= " ")
    text = re.sub(r"</?niche>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# How each stage should use the niche. Mirrors the shape of
# clip_selection._INSTRUCTIONS_STAGE_RULE, and says "guidance" rather than the
# absolute language the creator's own instructions get: a niche describes the
# kind of content, the creator decides this video.
_STAGE_RULE = {
    "score": ("Use this to judge what a strong moment looks like in this kind of "
              "video. A window matching what it says to look for scores higher; "
              "one matching what it says to skip scores low."),
    "rerank": "Compare the candidates by what this says makes a moment work here.",
    "detail": ("Choose and cut the clips by what this says works in this kind of "
               "video, and write the hook and title in that register."),
    "visual": "Pick visual moments of the kind this describes.",
}


def with_niche(prompt, niche_text, stage):
    """Insert the niche block into an already formatted prompt.

    No niche returns the prompt unchanged, so default prompts stay
    byte-identical. Insert this BEFORE the creator's instructions so their block
    sits nearer the data and its "win over the general criteria above" wording
    stays true — the niche is part of what it outranks.
    """
    if not niche_text:
        return prompt
    # Imported here rather than at module scope: clip_selection imports nothing
    # from this file, and a cycle would be the only thing gained.
    from clip_selection import _INSTRUCTIONS_ANCHOR

    rule = _STAGE_RULE.get(stage, _STAGE_RULE["score"])
    block = (
        "\nCONTENT TYPE — this video is a known kind, and these are the moments that\n"
        "work in it. Guidance, not a hard limit: the general criteria above still\n"
        "apply, and the creator's own instructions win over both.\n"
        f"- {rule}\n"
        "<niche>\n"
        f"{niche_text}\n"
        "</niche>\n"
    )
    at = prompt.find(_INSTRUCTIONS_ANCHOR)
    if at == -1:
        return prompt.rstrip("\n") + "\n" + block
    return prompt[:at] + block + prompt[at:]
