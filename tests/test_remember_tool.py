"""The model-callable `remember` tool (jarvis/tools/remember.py).

His ruling, 2026-09-04 18:15: "yes give him a remember tool, read it back".
The spoken rung ("remember that ...") stored and read back since 09-04;
the model had no tool that stores a fact, so "put it in your memory that
I graduate December 10th" went to the model, which said "I have noted
that, sir" and stored nothing -- the class the claim guard exists for.

ONE door: memory.store_fact_from_speech, the rung's own helper. The tool
refuses what the rung refuses (a bare head, a question, a pointer) and
what is somebody else's job (a to-do, a reminder, a list item), and its
read-back is an authored line -- spoken verbatim, ending the turn -- so
the model never paraphrases what was filed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis import brain as brain_mod
from jarvis.memory import JarvisMemory
from jarvis.tools import remember as rt
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def memory(tmp_path):
    return JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                        semantic=False)


def _tool(memory):
    (spec,) = rt.make_tools(None, SimpleNamespace(memory=memory))
    return spec


def test_the_spec_is_small_and_takes_one_fact(memory):
    spec = _tool(memory)
    assert spec.name == "remember"
    assert len(spec.description.split()) <= 20
    params = spec.schema()["function"]["parameters"]
    assert list(params["properties"]) == ["fact"]
    assert params["required"] == ["fact"]


def test_a_fact_is_filed_through_the_one_door_and_read_back_verbatim(memory):
    spec = _tool(memory)
    r = spec.handler(fact="I graduate December 10th 2026 with an electrical engineering degree")
    assert r.ok
    hits = memory.recall("graduate")
    assert len(hits) == 1
    stored = hits[0]["value"]
    assert stored.lower() == "you graduate december 10th 2026 with an electrical engineering degree"
    assert r.speak == f"Noted, sir: {stored}."            # the words filed, verbatim


def test_the_same_fact_said_twice_is_one_fact(memory):
    spec = _tool(memory)
    spec.handler(fact="my dentist is Dr Patel")
    spec.handler(fact="My dentist is Dr Patel.")
    assert len(memory.recall("dentist")) == 1


def test_a_person_statement_also_fills_the_people_book(memory):
    spec = _tool(memory)
    r = spec.handler(fact="my advisor is Dr Peyrovi, email hp@tamu.edu")
    assert r.ok
    assert any("peyrovi" in str(v).lower() for v in memory.people().values())


@pytest.mark.parametrize("fact, why", [
    ("", "empty"),
    ("me", "one word"),
    ("that", "a pointer"),
    ("for later", "a pointer"),
    ("everything I just said", "a pointer"),
    ("when do I graduate?", "a question"),
    ("remind me to call mom at 5", "a reminder -- set_reminder's job"),
    ("add milk to my shopping list", "a list item -- the notes tool's job"),
    ("set a timer for ten minutes", "a timer"),
])
def test_what_is_not_a_fact_is_refused_and_nothing_is_stored(memory, fact, why):
    spec = _tool(memory)
    r = spec.handler(fact=fact)
    assert not r.ok, why
    assert r.speak is None, why                      # the model answers, honestly
    assert memory._facts == {}


def test_the_refusal_names_the_right_tool_for_the_model(memory):
    spec = _tool(memory)
    assert "reminder" in spec.handler(fact="remind me to call mom at 5").text.lower()
    assert "list" in spec.handler(fact="add milk to my shopping list").text.lower()
    assert "question" in spec.handler(fact="when do I graduate?").text.lower()


def test_no_memory_service_is_an_honest_failure(tmp_path):
    (spec,) = rt.make_tools(None, SimpleNamespace())
    r = spec.handler(fact="my dentist is Dr Patel")
    assert not r.ok and "memory" in r.text.lower()


def test_the_tool_registers_and_the_claim_guard_knows_it(memory):
    reg = ToolRegistry()
    reg.register_many(rt.make_tools(None, SimpleNamespace(memory=memory)))
    assert reg.has("remember")
    assert "remember" in brain_mod.CLAIM_BACKERS["memory"]
    assert brain_mod.claim_backed("I have noted that", ran=["remember"])
    assert not brain_mod.claim_backed("I have noted that", ran=["get_time"])


def test_the_app_registers_it():
    from jarvis.app import TOOL_MODULES
    assert "jarvis.tools.remember" in TOOL_MODULES


def test_the_nudge_now_names_the_tool():
    assert "remember" in brain_mod.UNBACKED_MEMORY_NUDGE
    assert "no tool that stores facts" not in brain_mod.UNBACKED_MEMORY_NUDGE
