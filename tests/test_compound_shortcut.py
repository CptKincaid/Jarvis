"""A Tier-1 shortcut must be the WHOLE request (live 2026-08-31 14:35).

By voice: "What's on my calendar and what's on my latest email?" -- one
breath, two questions. _LAST_MAIL_RX is a `search`, so it matched the TAIL
("latest email"); the whole-utterance Tier-1 pass claimed the sentence and
forced get_mail on it (force_tool=get_mail, unread_only=False), and the
calendar half was dropped without a word:

    14:35:37.408  tier-1 match 'last mail' bypasses the intent gate:
                  "What's on my calendar and what's on my latest email?"

The sibling fix in brain.py guarantees the MAIL half now gets spoken. This
one is about the calendar half: the compound must reach the full tool loop,
which answers both (tests/test_brain_tools.py::
test_the_1435_turn_with_the_mail_fix_has_budget_to_spare).

The precedence itself is unchanged and must stay unchanged: the whole
utterance is still tried before split_clauses, or "remind me to buy milk
and eggs" becomes two commands. What changed is that a match which covers
only ONE clause of a two-request sentence no longer claims the other.

The router is the real one; the brain is a mock. No Ollama, no network.
"""
import types
from unittest.mock import MagicMock

import pytest

from jarvis.commander import (ASSISTANT_TIER1, Command, Commander,
                              IntentClassifier, local_tool_clause)
from jarvis.config import CONFIG
from jarvis.router import Router

LIVE = "What's on my calendar and what's on my latest email?"


class Cfg:
    def __init__(self, **over):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


@pytest.fixture
def rich(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "feedback.jsonl",
                        raising=False)
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(), assistant=Cfg(),
        timekeeper=MagicMock(), notes=MagicMock(), approvals=MagicMock(),
        claude=MagicMock())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.interpret_intent.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.timekeeper.ringing = None
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.brain.local_line.return_value = "Right away, sir."
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    return Commander(svc), svc


def _tier1(name: str) -> Command:
    for cmd in ASSISTANT_TIER1:
        if cmd.name == name:
            return cmd
    raise AssertionError(f"no Tier-1 command named {name!r}")


def _chat(svc):
    assert svc.brain.chat.call_count == 1, svc.brain.chat.call_args_list
    return svc.brain.chat.call_args


# ------------------------------------------------------------ the live turn
def test_the_1435_utterance_is_not_hijacked_by_last_mail(rich):
    """THE REGRESSION. Both questions must reach the model's tool loop:
    forcing get_mail answers one of them and drops the other in silence."""
    c, svc = rich
    res = c.handle(LIVE, source="voice")

    assert res.handled
    (said,), kw = _chat(svc)
    assert kw.get("force_tool") is None, \
        "the calendar half was dropped: the turn is pinned to get_mail"
    assert "calendar" in said.lower() and "email" in said.lower(), said


def test_the_1435_utterance_still_gets_past_the_intent_gate(rich):
    """The gate bypass was never the bug and must not be collateral: the
    classifier calls long sentences background chat and drops them."""
    c, svc = rich
    c.intent.classify = lambda t: (IntentClassifier.NO, 0.99)
    res = c.handle(LIVE, source="voice")
    assert res.status != "Ignored (background chat)"
    assert svc.brain.chat.call_count == 1


def test_the_same_hijack_on_the_prefixed_table(rich):
    """"Jarvis, <the same sentence>" goes through REGISTRY, which holds the
    identical 'last mail' entry."""
    c, svc = rich
    res = c.handle("jarvis, " + LIVE, source="voice")
    assert res.handled
    _, kw = _chat(svc)
    assert kw.get("force_tool") is None


def test_the_mail_half_first_is_left_alone_for_now(rich):
    """The mirror image ("read me my latest email and what's on my
    calendar") keeps its shortcut: a command that OPENS the utterance is
    usually the fuller answer, and handing it to the model would lose the
    handlers the model has no tool for. Documented, not accidental."""
    c, svc = rich
    c.handle("read me my latest email and what's on my calendar",
             source="voice")
    _, kw = _chat(svc)
    assert kw["force_tool"] == "get_mail"


# ------------------------------------------------- the bare commands stay fast
@pytest.mark.parametrize("said", [
    "last mail",
    "what's my latest email",
    "what was my last email about?",
    "read me my most recent email",
    "tell me about my last e-mail",
])
def test_the_bare_mail_shortcut_keeps_its_fast_path(rich, said):
    c, svc = rich
    res = c.handle(said, source="voice")
    assert res.handled and res.done is False
    _, kw = _chat(svc)
    assert kw["force_tool"] == "get_mail"
    assert kw["force_args"]["unread_only"] is False


@pytest.mark.parametrize("said,name", [
    ("last mail", "last mail"),
    ("what's my latest email", "last mail"),
    ("set a timer for 10 minutes", "timer"),
    ("set an alarm for 7 am", "alarm"),
    ("give me the briefing", "briefing"),
    ("standup", "standup"),
    ("good morning", "greeting"),
])
def test_a_bare_command_is_never_a_compound(rich, said, name):
    """The guard is inert on anything that does not split in two."""
    c, _ = rich
    assert c._compound_hijack(said, _tier1(name)) is False


# ---------------------------------------------- "and" inside ONE request
@pytest.mark.parametrize("said,name", [
    # the body of one command that happens to contain "and"
    ("remind me to buy milk and eggs", "remind me"),
    ("remind me to call the dentist and the vet", "remind me"),
    ("set a timer for one and a half minutes", "timer"),
    ("take a note salt and pepper", "take note"),
    ("add salt and pepper to my shopping list", "list add"),
    # a leading Tier-1 command whose tail is not a request of its own
    ("remind me to buy milk and check the weather", "remind me"),
])
def test_an_and_inside_one_request_keeps_its_shortcut(rich, said, name):
    """A naive "contains 'and'" guard would cost every one of these its
    fast path. The tail of each is a fragment, not a second intent."""
    c, _ = rich
    assert c._compound_hijack(said, _tier1(name)) is False, said


def test_the_briefing_still_owns_a_good_morning(rich):
    """"Good morning and what's on my calendar" opens with the command:
    the briefing answers the calendar too, and the model could not."""
    c, _ = rich
    said = "good morning and what's on my calendar"
    assert c._compound_hijack(said, _tier1("greeting")) is False
    assert c._compound_hijack(said, _tier1("briefing")) is False


# --------------------------------------------------- the two shapes it catches
def test_a_trailing_match_over_a_leading_request_declines(rich):
    c, _ = rich
    for said in ("what's on my calendar and what's on my latest email",
                 "what's the weather and what was my last email about",
                 "what's on my calendar tomorrow and read me my latest email"):
        assert c._compound_hijack(said, _tier1("last mail")) is True, said


def test_two_tier1_commands_defer_to_the_splitter(rich):
    """The precedence bug in the other direction: _try_multi was built for
    "remind me to call mum and set a timer for ten minutes" and never saw
    it, because the whole-utterance pass matched 'remind me' first."""
    c, _ = rich
    said = "remind me to call mum and set a timer for ten minutes"
    assert c._multi_match(said), "both halves must be Tier-1 for this test"
    assert c._compound_hijack(said, _tier1("remind me")) is True


def test_a_shaky_transcript_keeps_the_whole_match(rich):
    """_try_multi refuses to split a shaky transcript, so deferring to it
    would drop the command entirely."""
    c, _ = rich
    said = "remind me to call mum and set a timer for ten minutes"
    c._confidence = -5.0
    assert c.shaky_transcript()
    assert c._compound_hijack(said, _tier1("remind me")) is False


def test_a_matcher_that_covers_both_halves_is_not_a_compound(rich):
    """Nor is one that covers neither: the match then belongs to the whole
    sentence, not to one clause of it."""
    c, _ = rich
    both = Command("both", lambda t: True, lambda *a: None)
    neither = Command("neither", lambda t: t == "x", lambda *a: None)
    assert c._compound_hijack(LIVE.lower(), both) is False
    assert c._compound_hijack(LIVE.lower(), neither) is False


def test_a_raising_matcher_never_declines(rich):
    def boom(_t):
        raise ValueError("no")
    c, _ = rich
    bad = Command("boom", boom, lambda *a: None)
    assert c._compound_hijack(LIVE.lower(), bad) is False


# ----------------------------------------------------- the second-intent probe
@pytest.mark.parametrize("clause,kind", [
    ("what's on my calendar", "calendar"),
    ("what's the weather", "weather"),
    ("read me my latest email", "mail"),
    ("what time is it", "clock"),
    ("add milk to my to-dos", "notes"),
])
def test_a_clause_the_router_names_a_tool_for(clause, kind):
    assert local_tool_clause(clause) == kind


@pytest.mark.parametrize("clause", [
    "eggs", "a half minutes", "pepper", "in the morning",
    "take an umbrella", "the vet", "check the weather",
])
def test_a_fragment_is_not_a_second_request(clause):
    """Weak topic nouns must NOT count: "...and check the weather" is the
    body of the reminder that opened the sentence."""
    assert local_tool_clause(clause) == ""


def test_the_probe_survives_a_broken_router(monkeypatch):
    import jarvis.commander as commander

    def boom(_t):
        raise RuntimeError("router exploded")
    monkeypatch.setattr(commander, "local_cues", boom)
    assert local_tool_clause("what's on my calendar") == ""
