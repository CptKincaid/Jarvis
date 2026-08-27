"""Regression test for a gap in JarvisBrain._chat_sync()'s spoken guards.

DEFECT (jarvis/brain.py, _chat_sync, the speak branch):

    if speak is not None:
        spoken = trim_spoken(speak.strip(), cap=HARD_SPOKEN_CHARS)
    else:
        spoken = _finish_spoken(final, guard_ctx, text, cap)

A ToolResult that carries ``speak=`` skipped ``_finish_spoken()``
entirely, so it skipped ``clean_ollama_reply()`` (emoji, bullet markers,
stage directions, file extensions) and skipped the MAX_SPOKEN_SENTENCES
cap. Only ``trim_spoken`` at the 500-char TTS hard limit remained.

For tools whose spoken line is a fixed authored string that is fine. But
``jarvis/tools/notes.py`` returns the *user's own note text* down this
path:

    if act == "list":
        line = s.list_text(k)
        return ToolResult(text=line, speak=line)          # notes.py:467-468

and ``_NoteStore.list_text`` builds that line from up to ten stored items
via ``_spoken_item``, which normalises whitespace and truncates but does
not strip emoji or markdown (notes.py:98-104, 321-335). Whatever the user
typed into a note was therefore read aloud verbatim, and ten notes were
spoken past the two-sentence cap.

Reproduced through the real production path (JarvisBrain._chat_sync, a
stub notes handler standing in for the store's output):

    spoken: 'Noted, sir: buy milk 🥛 **urgent**. One. Two. Three. Four. Five.'
    emoji survived: True   markdown survived: True   sentences: 6

FIXED 2026-08-26 (brain work item, finding M8): the speak branch runs
_finish_spoken() too, so an authored line gets the same guards as a model
line — cleaner, markdown, clock guard, the sentence cap and trim_spoken.
The tests below drive the real _chat_sync rather than re-deriving the old
expression, so they pin the production path and not a copy of it.

These tests were the strict-xfail record of the defect; they now pass.
"""
import pytest

from jarvis import brain
from jarvis.brain import MAX_SPOKEN_SENTENCES, split_sentences
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec

# What notes.list_text can hand back when the stored items contain emoji
# or asterisks — the exact string is the tool's `speak=` value.
NOTE_LINE = "Noted, sir: buy milk 🥛 **urgent**. One. Two. Three. Four. Five."


@pytest.fixture
def spoken(monkeypatch):
    """spoken(line) -> what TTS would receive for ToolResult(speak=line),
    through the real tool loop."""
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="notes", description="notes and to-dos",
        handler=lambda **kw: ToolResult(text=NOTE_LINE, speak=NOTE_LINE)))
    monkeypatch.setattr(brain, "_http", lambda path, payload=None,
                        timeout=None: {"message": {
                            "content": "",
                            "tool_calls": [{"function": {"name": "notes",
                                                         "arguments": {}}}]}})
    b = brain.JarvisBrain(context=None, memory=None, registry=reg)

    def run():
        tags = b._chat_sync("read me my notes")
        said = [text for tag, text in tags if tag == "SPEAK"]
        assert said, tags
        return said[0]

    return run


def test_finish_spoken_would_have_cleaned_it():
    """Not a defect — the guards work, they are simply not on this path."""
    cleaned = brain.clean_ollama_reply(NOTE_LINE)
    assert not any(ord(c) > 0x1F000 for c in cleaned)


def test_speak_branch_strips_emoji(spoken):
    said = spoken()
    assert not any(ord(c) > 0x1F000 for c in said), said
    assert "**" not in said, said


def test_speak_branch_applies_the_sentence_cap(spoken):
    said = spoken()
    assert len(split_sentences(said)) <= MAX_SPOKEN_SENTENCES, said
