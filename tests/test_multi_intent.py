"""Two Tier-1 commands in one breath (2026-08-30).

Desktop chains have always split on "and then / then / and / ,", but the
registry did not, so "set a timer for ten minutes and add milk to my todo
list" fell through to the model and came back with half an answer. After
the WHOLE-utterance attempt fails, the same conjunctions split the text and
each half runs through the same ladder.

Whole-match-first is the safety argument, and it is what protects a body
that legitimately contains "and": "remind me at 5 pm to buy milk and eggs"
matches _REMIND_RX whole and is never split. So does a three-way list.

Real commander and a real NotesStore; the timekeeper is a mock and the
model is a MagicMock that must not be reached.
"""
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis.commander import Commander, IntentClassifier, split_clauses
from jarvis.config import CONFIG
from jarvis.router import Router
from jarvis.tools.notes import NotesStore


class Cfg:
    def __init__(self):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


@pytest.fixture
def rich(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    notes = NotesStore(tmp_path / "notes.db")
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(), assistant=Cfg(),
        timekeeper=MagicMock(), notes=notes, approvals=MagicMock(),
        claude=MagicMock())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.timekeeper.ringing = None
    svc.timekeeper.parse_when.return_value = datetime(2026, 8, 31, 17, 0)
    svc.timekeeper.describe_due.return_value = "at 5 pm"
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.brain.local_line.return_value = "Right away, sir."
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    yield Commander(svc), svc
    notes.close()


def _todos(svc):
    return [n["text"] for n in svc.notes.list("todo")]


# ------------------------------------------------------------ the split
def test_split_clauses_takes_exactly_two_halves():
    assert split_clauses("set a timer for ten minutes and add milk to my todo list") == \
        ["set a timer for ten minutes", "add milk to my todo list"]
    assert split_clauses("check the diary then read my notes") == \
        ["check the diary", "read my notes"]
    # one command, or three: not this feature's business
    assert split_clauses("set a timer for ten minutes") == []
    assert split_clauses("add milk, eggs and bread to my todo list") == []
    assert split_clauses("") == []


def test_split_clauses_drops_a_wake_word_from_either_half():
    assert split_clauses("Jarvis set a timer and add milk to my todo list") == \
        ["set a timer", "add milk to my todo list"]


# --------------------------------------------------------- both clauses
@pytest.mark.parametrize("said", [
    "set a timer for ten minutes and add milk to my todo list",
    "jarvis set a timer for ten minutes and add milk to my todo list",
    "set a timer for ten minutes, add milk to my todo list",
])
def test_a_compound_runs_both_halves(rich, said):
    c, svc = rich
    res = c.handle(said, source="typed")
    assert svc.timekeeper.add_timer.call_args.args[0] == 600
    assert _todos(svc) == ["milk"]
    assert res.handled and res.speak
    assert res.reply == ("10 minutes, sir; I'll let you know. "
                         "Added to your list, sir.")
    svc.brain.chat.assert_not_called()


def test_the_combined_status_names_both(rich):
    c, svc = rich
    res = c.handle("set a timer for ten minutes and add milk to my todo list",
                   source="typed")
    assert res.status == "Timer set: 10 minutes + To-do: milk"
    assert res.done is True


def test_one_follow_up_window_for_the_pair(rich):
    """done=False only if a half is still working: the app opens the
    follow-up window once, on the combined result."""
    c, svc = rich
    res = c.handle("set a timer for ten minutes and add milk to my todo list",
                   source="typed")
    assert res.done and res.ack is False


# ------------------------------------------------- what must NOT split
def test_a_reminder_body_with_and_is_never_split(rich):
    """The whole utterance matches _REMIND_RX, so the split is never
    reached and "milk and eggs" survives as one errand."""
    c, svc = rich
    c.handle("remind me at 5 pm to buy milk and eggs", source="typed")
    svc.timekeeper.add_reminder.assert_called_once_with(
        datetime(2026, 8, 31, 17, 0).timestamp(), "buy milk and eggs")
    assert _todos(svc) == []


def test_a_todo_body_with_and_is_never_split(rich):
    c, svc = rich
    c.handle("add milk and eggs to my todo list", source="typed")
    assert _todos(svc) == ["milk and eggs"]


def test_a_half_that_matches_nothing_runs_neither(rich):
    """Half an answer is worse than none: the compound goes to the model
    whole, exactly as it did before."""
    c, svc = rich
    c.handle("set a timer for ten minutes and call my mother", source="typed")
    svc.timekeeper.add_timer.assert_not_called()
    assert svc.brain.chat.call_args.args == (
        "set a timer for ten minutes and call my mother",)


def test_three_clauses_are_refused(rich):
    c, svc = rich
    c.handle("set a timer for ten minutes and add milk to my todo list "
             "and add eggs to my todo list", source="typed")
    svc.timekeeper.add_timer.assert_not_called()
    assert _todos(svc) == [] and svc.brain.chat.called


def test_a_plain_single_command_is_untouched(rich):
    c, svc = rich
    res = c.handle("set a timer for ten minutes", source="typed")
    assert res.status == "Timer set: 10 minutes"
    assert svc.timekeeper.add_timer.call_count == 1


# ------------------------------------------------------- voice + gate
def test_a_spoken_compound_bypasses_the_intent_gate(rich, monkeypatch):
    """Each half is a Tier-1 match, so the classifier -- which calls short
    phrases background chat -- is never consulted."""
    c, svc = rich

    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(c.intent, "classify", boom)
    res = c.handle("set a timer for ten minutes and add milk to my todo list",
                   source="voice")
    assert res.handled and _todos(svc) == ["milk"]


def test_each_half_is_its_own_turn_for_the_raw_text(rich):
    """_h_remind_me re-reads the raw utterance for its casing; left whole,
    _REMIND_RX would re-match the compound and swallow the other half."""
    c, svc = rich
    c.handle("add milk to my todo list and remind me at 5 pm to call Mum",
             source="typed")
    svc.timekeeper.add_reminder.assert_called_once_with(
        datetime(2026, 8, 31, 17, 0).timestamp(), "call Mum")
    assert _todos(svc) == ["milk"]


def test_the_desktop_chain_still_owns_a_mixed_compound(rich):
    """_try_desktop runs first and has always chained; this feature must
    not steal a compound it already handles."""
    c, svc = rich
    svc.desktop.parse_action = lambda part: SimpleNamespace(part=part)
    res = c.handle("jarvis switch to opera and go back", source="typed")
    assert res.status == "Desktop command"
    svc.desktop.execute_actions.assert_called_once()


def test_a_shaky_compound_is_declined(rich, monkeypatch):
    """Splitting is itself a guess about the words, and the creation
    read-backs cannot cover a pair (the second clause would overwrite the
    first one's pending question). Below confirm.shaky_logprob the whole
    thing goes to the model, as it did before the feature."""
    c, svc = rich
    monkeypatch.setattr(c.intent, "classify",
                        lambda t: (IntentClassifier.YES, 0.9))
    said = "set a timer for ten minutes and add milk to my todo list"
    c.handle(said, source="voice", confidence=-0.9)
    svc.timekeeper.add_timer.assert_not_called()
    assert _todos(svc) == [] and svc.brain.chat.call_args.args == (said,)
    # and the same words, heard clearly, still chain
    c.handle(said, source="voice", confidence=-0.2)
    assert svc.timekeeper.add_timer.call_count == 1 and _todos(svc) == ["milk"]
