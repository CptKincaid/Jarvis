"""Regression tests for a FOUND DEFECT (failure-mode sweep 2026-08-26).

DEFECT: jarvis/brain.py assumes an Ollama /api/chat reply is a dict whose
"message" is a dict whose "content" is a str, and jarvis/brain.py::_http
assumes the body is always valid JSON.

  * _http() (brain.py ~line 383) ends with `json.loads(body or b"{}")`
    with no guard, so a non-JSON body (a proxy's 502 HTML page, a
    truncated response) raises ValueError.
  * _chat_sync() (brain.py ~line 1020) does `msg = data.get("message")
    or {}` then `msg.get("tool_calls")` / `msg.get("content")`, so a
    non-dict `data`, a non-dict `message`, or a list/dict `content`
    raises AttributeError/TypeError.

ValueError / AttributeError / TypeError are NOT in the
`except (TimeoutError, urllib.error.URLError, OSError)` clause that
produces the designed persona fallbacks (MODEL_SLOW_LINE /
MODEL_DOWN_LINE), so they escape _chat_sync entirely.  JarvisBrain.chat()
catches them generically and SPEAKS the raw Python message, e.g.

    "I'm afraid I hit an error, sir. 'str' object has no attribute 'get'"

These shapes are reachable in practice: the user runs a LiteLLM proxy on
this machine, and OpenAI-compatible proxies return `content` as a list of
content blocks and error pages as HTML.

FIXED 2026-08-26 (brain work item, finding H8): _http() now raises
brain.MalformedReply instead of letting json.loads' ValueError out,
_message_parts() validates the /api/chat shape before anything reads it,
and _chat_sync catches (MalformedReply, ValueError) into MODEL_EMPTY_LINE.
JarvisBrain.chat()'s generic handler no longer speaks str(exc) either — it
speaks INTERNAL_ERROR_LINE and logs the traceback.

These tests were the strict-xfail record of the defect; they now pass.
"""
import pytest

from jarvis import brain
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec


@pytest.fixture
def chatty_brain(monkeypatch):
    reg = ToolRegistry()
    reg.register(ToolSpec(name="get_weather", description="weather",
                          handler=lambda **kw: ToolResult(text="fine")))
    return brain.JarvisBrain(context=None, memory=None, registry=reg)


def _reply(monkeypatch, data):
    monkeypatch.setattr(brain, "_http",
                        lambda path, payload=None, timeout=None: data)


PERSONA_LINES = (brain.MODEL_DOWN_LINE, brain.MODEL_SLOW_LINE,
                 brain.MODEL_EMPTY_LINE)


@pytest.mark.parametrize("data", [
    {"message": "It's sunny, sir."},              # message is a str
    {"message": {"content": [{"type": "text", "text": "hi"}]}},  # block list
    {"message": {"content": {"text": "hi"}}},     # content is a dict
    {"message": {"tool_calls": "not-a-list"}},
    ["nope"],                                     # top level is a list
    "nope",                                       # top level is a str
    None,                                         # top level is None
])
def test_malformed_ollama_reply_degrades_to_a_persona_line(
        chatty_brain, monkeypatch, data):
    """A malformed reply must degrade to one of the designed persona
    lines, never raise out of _chat_sync (where the caller speaks the raw
    Python error)."""
    _reply(monkeypatch, data)
    tags = chatty_brain._chat_sync("what's the weather")
    spoken = [text for tag, text in tags if tag == "SPEAK"]
    assert spoken and spoken[0] in PERSONA_LINES


def test_non_json_body_degrades_to_a_persona_line(chatty_brain, monkeypatch):
    """Ollama (or a proxy in front of it) answering with an HTML error
    page must not surface a json.loads ValueError to the user."""
    import json

    def html_body(path, payload=None, timeout=None):
        return json.loads(b"<html><body>502 Bad Gateway</body></html>")

    monkeypatch.setattr(brain, "_http", html_body)
    tags = chatty_brain._chat_sync("what's the weather")
    spoken = [text for tag, text in tags if tag == "SPEAK"]
    assert spoken and spoken[0] in PERSONA_LINES
