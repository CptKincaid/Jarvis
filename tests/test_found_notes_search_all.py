"""FOUND 2026-08-26 (adversarial re-check of the assistant-tools sweep):
notes(action="search", kind="all") searches only the notes and answers
"Nothing about X in your notes, sir." while a matching to-do exists.

jarvis/tools/notes.py:78 _kind("all") returns "" (KINDS is note/todo only)
and :427 _kind_for() then falls back to "note" for every action except
"done".  The list path has the same hole (kind="all" lists notes and
silently drops the to-dos), but search is worse: the user gets a
confident negative about an item that is on their list.

Reproduced in-process through the real ToolRegistry, scratch SQLite:

    notes add  todo "buy milk"                -> "Added to your list, sir."
    notes list kind="all"                     -> "Two notes, sir: ..."   (to-dos missing)
    notes search kind="all" text="milk"       -> "Nothing about milk in your notes, sir."
    notes search kind="todo" text="milk"      -> finds it

manage_schedule already accepts kind="all" across its three kinds, so the
inconsistency is visible to the model.  ("all" is not in the tool schema's
enum, but the schema is a hint, not a constraint - the model emits it.)

This test asserts what "all" should mean; it fails today.
"""
from types import SimpleNamespace

import pytest

from jarvis.assistant_config import AssistantConfig
from jarvis.tools.notes import make_tools
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_MEMORY_DIR", str(tmp_path))
    from jarvis import config as jconfig
    monkeypatch.setattr(jconfig.PATHS, "MEMORY_DIR", tmp_path, raising=False)
    reg = ToolRegistry()
    reg.register_many(make_tools(AssistantConfig({}),
                                 SimpleNamespace(notes=None)))
    return reg


@pytest.mark.xfail(reason='_kind("all") -> "" and _kind_for() defaults to '
                          '"note", so to-dos are never searched or listed',
                   strict=True)
def test_search_all_finds_a_todo(registry):
    registry.call("notes", {"action": "add", "kind": "note",
                            "text": "the boiler service is due in October"})
    registry.call("notes", {"action": "add", "kind": "todo",
                            "text": "buy milk"})

    found = registry.call("notes", {"action": "search", "kind": "todo",
                                    "text": "milk"})
    assert "milk" in found.text                      # the item is there

    both = registry.call("notes", {"action": "search", "kind": "all",
                                   "text": "milk"})
    assert "milk" in both.text, both.text            # fails: "Nothing about milk"

    listed = registry.call("notes", {"action": "list", "kind": "all"})
    assert "buy milk" in listed.text, listed.text
