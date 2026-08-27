"""FOUND 2026-08-26 (assistant-tools sweep): the "Spotify isn't set up"
line Jarvis actually SPEAKS is not the line the app prewarms into the
speech cache, and it says "spotify" in lower case mid-sentence.

jarvis/tools/spotify.py defines the persona line as

    SETUP_LINE = f"I'll need Spotify set up, sir; the notes are in {DOCS_HINT}."

and exports it in PERSONA_LINES, which jarvis/app.py:335 feeds to
TTS.prewarm() at boot (_canned_phrases).  But _Spotify.setup_line()
(jarvis/tools/spotify.py:561-570) prefers cfg.setup_line("spotify"), and
"spotify" is NOT one of assistant_config.SETUP_LINES' sections, so the
generic fallback in AssistantConfig.setup_line() renders

    "I'll need spotify set up, sir; the notes are in docs/assistant-setup.md."

(lower-case "spotify", from `section.replace("_", " ")`).  Every one of
the six spotify tools returns that string as ToolResult.speak with
placeholder credentials — measured, all six identical — so the prewarmed
entry is never hit and the line is synthesised by TTS every time.

Either give assistant_config.SETUP_LINES a "spotify" entry whose text is
spotify.SETUP_LINE, or have _Spotify.setup_line() fall back to the module
constant only when cfg has a bespoke line for the section.
"""
import pytest

import jarvis.tools.spotify as spotify_mod
from jarvis.assistant_config import AssistantConfig


@pytest.mark.xfail(reason="cfg.setup_line('spotify') falls through to the "
                          "generic template, so the spoken line differs from "
                          "the prewarmed PERSONA_LINES entry",
                   strict=True)
def test_spoken_spotify_setup_line_is_the_prewarmed_one():
    cfg = AssistantConfig({"spotify": {"client_id": "", "client_secret": ""}})
    tools = spotify_mod.make_tools(cfg, None)
    spoken = {t.handler(**_min_args(t)).speak for t in tools}
    assert spoken == {spotify_mod.SETUP_LINE}, spoken
    assert spotify_mod.SETUP_LINE in spotify_mod.PERSONA_LINES


def _min_args(spec):
    """The smallest argument set each spotify tool accepts."""
    required = (spec.parameters or {}).get("required") or []
    props = (spec.parameters or {}).get("properties") or {}
    args = {}
    for name in required:
        enum = (props.get(name) or {}).get("enum")
        args[name] = enum[0] if enum else "x"
    return args
