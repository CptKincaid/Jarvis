"""A KNOWN person gets answers, never his data -- through the tool loop.

Review finding 2 (09-04). ``gate.allowed_for`` refuses the turns that NAME
his things (mail, calendar, notes...). A question that does not --
"anything on today?", "what did I write down?" -- is admitted as plain
chat, and the gate's comment promised "no tools behind it". Nothing
enforced that: the turn reached gemma4 with every registered tool offered,
and get_calendar / notes / get_mail were one tool call away.

Now the tool loop reads the addressee once per turn: a non-owner is
offered ``brain.KNOWN_TOOLS`` (time, weather) and every call outside it --
model-chosen or commander-forced -- is refused with an authored line and
the handler is never run. Against the mocked Ollama from
tests/test_brain_tools.py: no network, no model.
"""
import pathlib
import re

import pytest

from jarvis import brain as brain_mod
from tests.test_brain_tools import (FakeContext, FakeMemory, FakeOllama,
                                    make_registry, text_reply, tool_reply)

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(monkeypatch):
    brain_mod.reset_static_prompt()
    record = []
    reg = make_registry(record)
    monkeypatch.setattr(brain_mod, "_REGISTRY", reg)
    fake = FakeOllama()
    monkeypatch.setattr(brain_mod, "_http", fake)
    b = brain_mod.JarvisBrain(context=FakeContext(), memory=FakeMemory())
    return b, fake, record, reg


def _offered(payload):
    return sorted(t["function"]["name"] for t in payload.get("tools", []))


def test_the_allow_list_is_the_time_and_the_weather():
    assert brain_mod.KNOWN_TOOLS == frozenset({"get_time", "get_weather"})
    for his in ("get_calendar", "notes", "get_mail", "get_briefing",
                "recap_day", "canvas_due", "set_reminder", "add_event",
                "ask_docs", "screen_qa", "get_location"):
        assert not brain_mod.tool_in_scope(his, owner=False), his
        assert brain_mod.tool_in_scope(his, owner=True), his


def test_a_known_persons_turn_is_offered_only_the_open_tools(setup):
    b, fake, record, reg = setup
    brain_mod.set_addressee("Heather", "ma'am")
    assert _offered({"tools": brain_mod._registry_schemas(reg)}) == \
        ["get_time", "get_weather"]
    brain_mod.set_addressee("", "sir")
    assert _offered({"tools": brain_mod._registry_schemas(reg)}) == \
        sorted(reg.names())


def test_a_hallucinated_notes_call_never_runs_for_a_known_person(setup):
    """The model was not offered ``notes`` and asks for it anyway."""
    b, fake, record, reg = setup
    brain_mod.set_addressee("Heather", "ma'am")
    fake.replies = [tool_reply(("notes", {"action": "list"})),
                    text_reply("Here is what you wrote down."),
                    text_reply("spare")]
    tags = b._chat_sync("what did I write down?")
    assert record == [], "his notes handler ran for a guest"
    assert _offered(fake.chat_payloads()[0]) == ["get_time", "get_weather"]
    assert tags == [("SPEAK", "That one's Hunter's, Heather. I can give "
                              "you the time and the weather.")]


def test_a_commander_short_cut_cannot_force_his_tool_for_a_guest(setup):
    """``_h_recap`` and the briefing routes force a tool by name; the
    scope sits below them, so the forced call is refused too and the
    turn ends on the authored line with no model round."""
    b, fake, record, reg = setup
    brain_mod.set_addressee("Heather", "ma'am")
    tags = b._chat_sync("good morning", force_tool="get_briefing")
    assert record == []
    assert fake.chat_payloads() == []
    assert tags == [("SPEAK", "That one's Hunter's, Heather. I can give "
                              "you the time and the weather.")]


def test_the_open_tools_still_work_for_her(setup):
    b, fake, record, reg = setup
    brain_mod.set_addressee("Heather", "ma'am")
    fake.replies = [tool_reply(("get_time", {})),
                    text_reply("It's five past four, ma'am.")]
    tags = b._chat_sync("what time is it?")
    assert record == [("get_time", None)]
    assert tags == [("SPEAK", "It's five past four, ma'am.")]


def test_the_owner_is_offered_everything_and_his_tools_run(setup):
    b, fake, record, reg = setup
    brain_mod.set_addressee("", "sir")
    fake.replies = [text_reply("Seventy-two and sunny, sir.")]
    tags = b._chat_sync("good morning", force_tool="get_briefing")
    assert record == [("get_briefing",)]
    assert _offered(fake.chat_payloads()[0]) == sorted(reg.names())
    assert tags[-1] == ("SPEAK", "Seventy-two and sunny, sir.")


def test_the_scope_is_read_once_per_turn_beside_the_prompt(setup):
    """The gate names the NEXT person while a guest's turn is still on the
    worker. The turn keeps the reading it started with."""
    b, fake, record, reg = setup
    brain_mod.set_addressee("Heather", "ma'am")

    def flip_then_reply(path, payload=None, timeout=None):
        brain_mod.set_addressee("", "sir")          # Hunter walks in
        return fake(path, payload, timeout)
    fake.replies = [tool_reply(("get_briefing", {})), text_reply("spare")]
    b._http = flip_then_reply
    brain_mod._http = flip_then_reply
    tags = b._chat_sync("anything on today?")
    assert record == []
    assert tags[0][1].startswith("That one's Hunter's, Heather.")


def test_every_tool_call_in_the_loop_goes_through_the_scope():
    """One ``registry.call(`` in brain.py, and it is inside _scoped_call."""
    src = (ROOT / "jarvis" / "brain.py").read_text(encoding="utf-8")
    calls = [m.start() for m in re.finditer(r"registry\.call\(", src)]
    assert len(calls) == 1
    start = src.index("def _scoped_call(")
    end = src.index("\ndef ", start + 1)
    assert start < calls[0] < end
