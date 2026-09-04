"""A KNOWN person's turn through the COMMANDER: default-deny, one scope.

Round-2 review (09-04), finding B -- measured with the reviewer's own probe
scripts (attack.py PART 3, cal.py), which these tests are. ``gate.allowed_for``
admitted "what's my next class", "what's on my to-do list", "what did I
miss", "who is my doctor", "what did I say about the dentist" as plain
questions, and commander's Tier-1 handlers answered them from his calendar,
his notes, his held notifications and his memory -- no model, no scope.
brain.KNOWN_TOOLS guarded only the tool loop.

Now ``Commander.handle`` reads the turn's addressee ONCE (jarvis/scope.py)
and a known non-owner takes ``_handle_known``: the voice I/O words, the
courtesies as plain lines, ``KNOWN_TIER1`` by name (the clock, arithmetic),
the authored refusal for anything else the table matches, the
background-chat gate, then the model with her own scope. Nothing below it
runs for her -- not the pending yes/no rungs, not dictation, not
auto-type. Fakes only.
"""
import pathlib
import re
import time
import types
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from jarvis import scope as scope_mod
from jarvis.commander import (CATCH_ALL_COMMANDS, KNOWN_TIER1, REGISTRY,
                              Commander, CommandResult, IntentClassifier,
                              clock_reply)
from jarvis.config import CONFIG
from jarvis.identity import ROLE_KNOWN
from jarvis import gate as gate_mod
from tests.test_found_next_class import NOW, WISENBAKER, _cal, _course

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARA_LINE = "That one's Hunter's, Mara. I can give you the time and the weather."
MARA = ("Mara", "ma'am")

# The reviewer's list, verbatim (attack.py PART 3), plus cal.py's three.
HIS_QUESTIONS = [
    "what's on my to-do list",
    "what do I have to do today",
    "what's on my shopping list",
    "what's my next class",
    "when is my next class",
    "where is my next lecture",
    "when is my next exam",
    "what did I miss",
    "what do you see",
    "what was I working on",
    "who is my doctor",
    "what did I say about the dentist",
    "show clipboard",
    "how's my week looking",
    "recap my day",
    "read my last email",
    "how did yesterday go",
    "show my notes",
]
# Words gate._HIS refuses at ADMISSION (calendar, inbox) and never lets
# reach the commander for a known person. Should they arrive anyway, no
# Tier-1 matcher accepts them unprefixed, so they go to the model -- whose
# own scope (brain.KNOWN_TOOLS) offers her the time and the weather.
GATE_VETOED = ["what's on my calendar", "what's in my inbox",
               "what's on my calendar tomorrow"]

# The reviewer's set (attack.py), plus the two the known path may touch.
HIS_SERVICES = ("desktop", "workflows", "brain", "memory", "context",
                "calendar", "notes", "quiet", "canvas", "mail", "study",
                "reminders", "briefing", "journal", "people", "claude",
                "scenes", "winddown")


@pytest.fixture(autouse=True)
def _config(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "auto_type", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    scope_mod.clear_addressee()
    yield
    scope_mod.clear_addressee()


def _services():
    """The reviewer's harness: every service a MagicMock that records,
    with the routing probes stubbed the way the owner's path needs."""
    svc = types.SimpleNamespace()
    for n in HIS_SERVICES:
        setattr(svc, n, MagicMock(name=n))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.notes.list_text.return_value = "1. buy milk. 2. call the bank about the mortgage."
    svc.quiet.release.return_value = "Held: Heather texted about Saturday."
    svc.context.describe_screen.return_value = "A Gmail window and a bank statement."
    svc.honorific = lambda: "ma'am"
    svc.tts = MagicMock(name="tts")
    svc.tts.last_text = "It's five past four."
    return svc


def _touched(svc):
    """Every service that was CALLED. ``brain.register()`` -- the spoken
    register a courtesy line is worded in -- is not a read of his."""
    out = []
    for n in vars(svc):
        v = getattr(svc, n)
        if not isinstance(v, MagicMock):
            continue
        calls = [mc for mc in v.mock_calls
                 if not (n == "brain" and str(mc).startswith("call.register("))]
        if calls:
            out.append(n)
    return sorted(out)


def _commander(svc=None):
    svc = svc or _services()
    c = Commander(svc)
    c.intent.classify = lambda text: (IntentClassifier.YES, 0.99)
    return c, svc


# ------------------------------------------------- the reviewer's probes
@pytest.mark.parametrize("phrase", HIS_QUESTIONS)
def test_a_known_person_is_refused_and_no_service_of_his_is_read(phrase):
    """attack.py PART 3: for every phrase, the authored line, spoken, and
    NOT ONE service touched -- not the calendar, not notes, not quiet,
    not memory, not the brain, not even the habit log."""
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    res = c.handle(phrase, source="voice")
    assert res.handled and res.speak
    assert res.reply == MARA_LINE, (phrase, res.reply)
    assert _touched(svc) == [], (phrase, _touched(svc))


@pytest.mark.parametrize("phrase", GATE_VETOED)
def test_a_gate_vetoed_question_that_arrives_anyway_reaches_only_the_scoped_model(phrase):
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    res = c.handle(phrase, source="voice")
    assert res.handled and res.done is False
    svc.brain.chat.assert_called_once_with(phrase, addressee=MARA)
    assert _touched(svc) == ["brain"], (phrase, _touched(svc))


def test_the_same_questions_reach_his_data_for_him():
    """The control: the owner's path is not what changed. The reviewer's
    measured readers, still read for HIM."""
    c, svc = _commander()
    c.handle("what's on my to-do list", source="voice")
    assert svc.notes.list_text.called
    c, svc = _commander()
    c.handle("what did I miss", source="voice")
    assert svc.quiet.release.called
    c, svc = _commander()
    c.handle("who is my doctor", source="voice")
    assert svc.memory.resolve_person.called
    c, svc = _commander()
    c.handle("what did I say about the dentist", source="voice")
    assert svc.memory.recall.called


def _calendar_commander():
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(),
        calendar=_cal(_course("BIOSENSORS", NOW + timedelta(hours=2), WISENBAKER)))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    return Commander(svc), svc


@pytest.mark.parametrize("phrase", ["what's my next class", "when is my next class",
                                    "where is my next lecture"])
def test_cal_py_his_next_class_is_not_hers(phrase):
    """cal.py: the gate admits the words; the calendar rung answered them
    from his real calendar shape with no brain. Now: the line, and the
    control still answers BIOSENSORS for him."""
    g = gate_mod.OwnerGate.__new__(gate_mod.OwnerGate)
    ok, _line = gate_mod.OwnerGate.allowed_for(g, ROLE_KNOWN, phrase)
    assert ok, "the gate's regexes still admit it; the scope must not"
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _calendar_commander()
    res = c.handle(phrase, source="voice")
    assert res.reply == MARA_LINE
    assert "BIOSENSORS" not in (res.status or "")
    assert not svc.brain.chat.called and not svc.brain.think.called
    scope_mod.clear_addressee()
    c, svc = _calendar_commander()
    res = c.handle(phrase, source="voice")
    assert "BIOSENSORS" in res.reply and "Wisenbaker" in res.reply


# ------------------------------------------------------- what she gets
def test_the_clock_the_sums_and_the_courtesies_are_hers():
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    res = c.handle("what time is it", source="voice")
    assert res.status == "Clock" and res.speak
    assert res.reply.split(":")[0] == \
        clock_reply(__import__("datetime").datetime.now(), "time").split(":")[0]
    res = c.handle("what's 18 percent of 74", source="voice")
    assert res.status.startswith("Maths") and "13.3" in res.reply
    res = c.handle("thank you", source="voice")
    assert res.status == "Courtesy" and res.reply
    res = c.handle("good morning", source="voice")
    assert res.status == "Courtesy" and res.reply
    assert _touched(svc) == [], _touched(svc)


def test_good_night_from_a_guest_starts_nothing_of_his():
    """The owner's "good night" starts his wind-down and evening preview;
    hers is the line and nothing else."""
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    res = c.handle("good night", source="voice")
    assert res.status == "Courtesy" and res.reply and res.speak
    assert _touched(svc) == [], _touched(svc)


def test_quiet_and_say_again_work_and_quiet_does_not_cancel_his_claude():
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    res = c.handle("say again", source="voice")
    assert res.status == "Repeating" and res.reply == "It's five past four."
    res = c.handle("stop talking", source="voice")
    assert res.status == "Quiet"
    assert not svc.claude.cancel.called
    assert svc.tts.interrupt.called or svc.tts.stop.called


def test_a_plain_question_goes_to_the_model_with_no_forced_tool():
    """"I'll answer what I can": the model, whose own scope offers her the
    time and the weather (tests/test_known_tool_scope.py). Not the
    router: no Claude session, no web one-shot, no forced tool."""
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    res = c.handle("hey jarvis, how far is the moon?", source="voice")
    assert res.handled and res.done is False
    # The commander's ONE reading travels with the text: the tool loop on
    # the worker scopes the turn to her even if the attribution expires or
    # the gate names the next person before the worker reads it.
    svc.brain.chat.assert_called_once_with("how far is the moon?",
                                           addressee=MARA)
    assert not svc.claude.submit.called
    assert _touched(svc) == ["brain"]


def test_a_compound_with_his_half_in_it_is_refused_whole():
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    res = c.handle("what time is it and what's on my to-do list", source="voice")
    assert res.reply == MARA_LINE
    assert _touched(svc) == []


def test_background_chat_from_a_guest_is_dropped_and_never_carded():
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    cards = []
    c.on_uncertain = cards.append
    c.intent.classify = lambda text: (IntentClassifier.UNCERTAIN, 0.5)
    res = c.handle("we could go there tomorrow", source="voice")
    assert res.status == "Ignored (background chat)"
    assert cards == [] and _touched(svc) == []


def test_her_yes_does_not_answer_his_read_back():
    """He was asked "Cancel all three alarms, sir?" and Mara says yes.
    The read-back stays armed for HIM; nothing runs."""
    ran = []

    def run():
        ran.append(1)
        return CommandResult(handled=True, reply="Cancelled.", speak=True,
                             status="Cancelled")
    c, svc = _commander()
    c._pending_destructive = (run, "Cancel all three alarms, sir?",
                              time.monotonic())
    scope_mod.set_addressee("Mara", "ma'am")
    c.handle("yes", source="voice")
    assert ran == []
    assert c._pending_destructive is not None, "her turn dropped his question"
    scope_mod.clear_addressee()
    res = c.handle("yes", source="voice")
    assert ran == [1] and res.reply == "Cancelled."


def test_her_words_are_never_typed_into_his_window():
    """The owner's fallthrough auto-types; hers cannot."""
    scope_mod.set_addressee("Mara", "ma'am")
    svc = _services()
    svc.brain = None
    c = Commander(svc)
    c.intent.classify = lambda text: (IntentClassifier.YES, 0.99)
    res = c.handle("the quick brown fox", source="voice")
    assert res.handled is False and res.status == "No route (no brain)"
    assert not svc.desktop.type_text.called


def test_the_commander_scope_expires_with_the_same_ttl(monkeypatch):
    monkeypatch.setattr(scope_mod, "_now",
                        lambda now: 0.0 if now is None else float(now))
    scope_mod.set_addressee("Mara", "ma'am")
    c, svc = _commander()
    assert c.handle("what's on my to-do list", source="voice").reply == MARA_LINE
    assert not svc.notes.list_text.called
    monkeypatch.setattr(scope_mod, "_now",
                        lambda now: 600.0 if now is None else float(now))
    c.handle("what's on my to-do list", source="voice")
    assert svc.notes.list_text.called


def test_every_one_of_his_entries_is_refused_by_name_and_no_handler_runs(monkeypatch):
    """Not the reviewer's sample of 18: EVERY REGISTRY entry outside
    KNOWN_TIER1 and the one catch-all (140 of 143 when written). Each in
    turn is made to accept a sentence nothing else in the table matches,
    every handler in the table is a recorder, and Mara's turn must come
    back as the authored line with no handler run, no service touched and
    the owner's _handle_inner never entered. The control, per entry: the
    same words on HIS turn enter _handle_inner and never _handle_known.
    So a new entry is refused for her the day it is added, by default,
    and the allow-list is the only thing that can open one."""
    probe = "zzz qqq wibble"
    ran = []

    def spy(name):
        def h(c, t, m):
            ran.append(name)
            return CommandResult(handled=True, reply="LEAKED " + name,
                                 status=name)
        return h
    for e in REGISTRY:
        monkeypatch.setattr(e, "handler", spy(e.name))
    inner = []
    monkeypatch.setattr(
        Commander, "_handle_inner",
        lambda self, text, source, gate=True: (
            inner.append(text), CommandResult(handled=True, status="inner"))[1])
    his = [e for e in REGISTRY
           if e.name not in KNOWN_TIER1 and e.name not in CATCH_ALL_COMMANDS]
    assert len(his) == len(REGISTRY) - len(KNOWN_TIER1) - len(CATCH_ALL_COMMANDS)
    assert len(his) >= 140, len(his)
    refused = []
    for e in his:
        orig = e.matcher
        e.matcher = lambda t: True
        try:
            scope_mod.set_addressee(*MARA)
            c, svc = _commander()
            res = c.handle(probe, source="voice")
            assert res.reply == MARA_LINE and res.status == "Not Mara's", \
                (e.name, res)
            assert res.speak is True, e.name
            assert ran == [], (e.name, ran)
            assert _touched(svc) == [], (e.name, _touched(svc))
            assert inner == [], (e.name, inner)
            refused.append(e.name)
            scope_mod.clear_addressee()
            c, svc = _commander()
            c.handle(probe, source="voice")
            assert inner == [probe], (e.name, inner)
            inner.clear()
        finally:
            e.matcher = orig
    assert sorted(refused) == sorted(e.name for e in his)
    assert ran == []


# ------------------------------------------------------------ the shape
def test_the_only_catch_all_matcher_is_workflow():
    """The probe skips matchers that accept ANY sentence; if a second one
    appears, a guest's every question would be refused as his."""
    nonsense = "zzz qqq wibble"
    assert [e.name for e in REGISTRY if e.matcher(nonsense)] == ["workflow"]
    assert CATCH_ALL_COMMANDS == frozenset({"workflow"})


def test_the_allow_list_is_the_clock_and_the_sums_and_they_read_nothing_of_his():
    assert KNOWN_TIER1 == frozenset({"clock", "math"})
    names = {cmd.name for cmd in REGISTRY}
    assert KNOWN_TIER1 <= names
    src = (ROOT / "jarvis" / "commander.py").read_text(encoding="utf-8")
    for fn in ("_h_clock", "_h_math"):
        start = src.index(f"\ndef {fn}(")
        body = src[start:src.index("\ndef ", start + 1)]
        assert "_svc(" not in body and "services" not in body, fn


def test_handle_reads_the_scope_once_and_the_known_path_never_reaches_his():
    """Source guards: one read at the top of handle(); _handle_known calls
    neither _handle_inner nor _route_text nor _try_assistant nor
    _try_registry nor _dispatch_router, and every REGISTRY entry it runs
    is gated by name."""
    src = (ROOT / "jarvis" / "commander.py").read_text(encoding="utf-8")

    def method(name):
        start = src.index(f"    def {name}(")
        return src[start:src.index("\n    def ", start + 1)]
    handle = method("handle")
    assert len(re.findall(r"scope_mod\.addressee\(\)", handle)) == 1
    assert "self._handle_known(text, source, who, hon)" in handle
    known = method("_handle_known")
    for banned in ("_handle_inner(", "_route_text(", "_try_assistant(",
                   "_try_registry(", "_dispatch_router(", "_try_multi(",
                   "_try_desktop(", "type_text(", "_try_approval(",
                   "_try_destructive_confirm(", "_h_courtesy(",
                   "_h_greeting(", "_h_quiet(", "_h_next_class(",
                   "_try_custom_phrase(", "_h_read_aloud("):
        assert banned not in known, banned
    assert "if name and name not in KNOWN_TIER1:" in known
    assert "self._refused(name, who)" in known
    assert "if entry.name not in KNOWN_TIER1:\n                continue" in known
    # The one scope function, and the commander's refusal goes through it.
    refused = method("_refused")
    assert "scope_mod.owner_only(" in refused
