"""Working sessions (jarvis/dialogue.py): the framework, the commander's
rung, the mic window, and the first tenant ("let us plan the week").

Nothing here touches Canvas, the timekeeper or a mic: the framework is
pure, EchoSession is the test double the ladder is proved with, and the
planner's filer is a list-appending fake.
"""
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
import jarvis.dialogue as dialogue
from jarvis.commander import ASSISTANT_TIER1, REGISTRY, Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.dialogue import (
    NO_ROOM_LINE,
    PLAN_EMPTY_LINE,
    PLAN_NOTHING_LINE,
    SESSION_STALE_S,
    SESSION_WINDOW_S,
    EchoSession,
    PlanItem,
    WeekPlanner,
    collect_items,
    enough_kind,
    week_days,
)

MONDAY = date(2026, 8, 31)          # a Monday, so weekday names are stable


# ------------------------------------------------------------- constants
def test_session_window_is_long_enough_to_answer_a_question():
    """The whole point of the protocol: CONFIG.followup_window (4 s) is a
    beat, not a conversation. Regression guard on the number that makes
    multi-turn possible at all."""
    assert SESSION_WINDOW_S >= 15.0
    assert SESSION_WINDOW_S > CONFIG.followup_window


def test_stale_constant_is_not_the_quiz_answer_window():
    from jarvis.tools.quiz import ANSWER_WINDOW_S
    assert SESSION_STALE_S == 90.0 and ANSWER_WINDOW_S != SESSION_STALE_S


@pytest.mark.parametrize("phrase,hit", [
    ("that's enough", True), ("enough", True), ("that'll do", True),
    ("we're done", True), ("enough for now", True), ("that's it", True),
    ("do it", False), ("enough of that nonsense", False), ("yes", False),
])
def test_enough_kind(phrase, hit):
    assert enough_kind(phrase) is hit


# ---------------------------------------------------------- the protocol
def test_echo_session_holds_the_floor_and_finishes():
    s = EchoSession(limit=2)
    assert s.ask(now=100.0) == "Question 1?"
    assert s.settle("hello") == "You said hello."
    assert not s.finished
    assert s.settle("again") == "You said again."
    assert s.finished and s.stop() == "2 noted, sir."


def test_stale_is_measured_from_the_last_ask():
    s = EchoSession()
    assert not s.stale(now=1_000.0)                # never asked: never stale
    s.ask(now=1_000.0)
    assert not s.stale(now=1_000.0 + SESSION_STALE_S - 1)
    assert s.stale(now=1_000.0 + SESSION_STALE_S + 1)


# ------------------------------------------------------- commander rung
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
        context=MagicMock(), tts=MagicMock(), brain=MagicMock(),
        notes=None, timekeeper=None, assistant=None)
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.replies = []
    svc.reply = lambda text, speak=True: svc.replies.append((text, speak))
    return Commander(svc)


def test_a_session_answers_turn_after_turn_without_a_wake_word(cmdr):
    session = EchoSession(limit=3)
    assert cmdr.open_session(session) is True
    for n in (1, 2):
        res = cmdr.handle(f"line {n}", source="voice")
        assert res.handled and res.speak and res.done
        assert f"You said line {n}." in res.reply
        assert cmdr._pending_session is session
    res = cmdr.handle("line 3", source="voice")
    assert "3 noted, sir." in res.reply
    assert cmdr._pending_session is None            # finished: dropped


def test_session_results_are_done_and_spoken(cmdr):
    """app._after_dispatch arms the follow-up mic ONLY for a done, spoken
    reply; anything else silently kills the dialogue."""
    cmdr.open_session(EchoSession())
    res = cmdr.handle("hello", source="voice")
    assert res.done is True and res.speak is True and res.reply


def test_quiet_drops_a_session_without_speaking(cmdr):
    cmdr.open_session(EchoSession())
    res = cmdr.handle("quiet", source="voice")
    assert res.handled and res.speak is False
    assert cmdr._pending_session is None


def test_that_is_enough_closes_with_the_read_back(cmdr):
    session = EchoSession()
    cmdr.open_session(session)
    cmdr.handle("one", source="voice")
    res = cmdr.handle("that's enough", source="voice")
    assert res.speak and "1 noted, sir." in res.reply
    assert cmdr._pending_session is None


def test_a_stale_session_is_dropped_and_the_words_route(cmdr):
    session = EchoSession()
    cmdr.open_session(session)
    session.asked_at = 1.0                          # asked long ago
    res = cmdr.handle("what time is it", source="typed")
    assert cmdr._pending_session is None
    assert res.handled and "session" not in (res.status or "").lower()


def test_an_unrecognised_utterance_drops_the_session_and_routes(cmdr):
    """settle() -> None is the escape hatch: nothing is ever trapped."""
    cmdr.open_session(EchoSession())
    res = cmdr.handle("what's the weather", source="typed")
    assert cmdr._pending_session is None
    assert "You said" not in (res.reply or "")


def test_dictation_and_lecture_notes_refuse_to_open_a_session(cmdr):
    cmdr.dictation = True
    assert cmdr.open_session(EchoSession()) is False
    assert cmdr._pending_session is None
    cmdr.dictation = False
    cmdr.lecture_course = "biosensors"
    assert cmdr.open_session(EchoSession()) is False
    assert cmdr._pending_session is None


def test_a_broken_tenant_does_not_take_the_turn_down(cmdr):
    class Boom(EchoSession):
        def settle(self, text):
            raise RuntimeError("nope")

    cmdr.open_session(Boom())
    res = cmdr.handle("hello", source="typed")
    assert res.handled and res.reply == commander.SESSION_LOST_LINE
    assert cmdr._pending_session is None


def test_a_slim_commander_has_no_session_rung():
    c = object.__new__(Commander)
    assert c._try_session("anything") is None


# ---------------------------------------------------------- the mic window
def test_capture_window_asks_the_commander_for_the_session_window():
    from jarvis.app import JarvisApp
    app = object.__new__(JarvisApp)
    app.commander = SimpleNamespace(_pending_session=None, lecture_course=None)
    app.assistant = SimpleNamespace(get=lambda k, d=None: d)
    assert app._capture_window() is None
    app.commander._pending_session = EchoSession()
    assert app._capture_window() == max(SESSION_WINDOW_S, CONFIG.followup_window)


def test_capture_window_ignores_a_finished_session_and_keeps_lecture():
    from jarvis.app import JarvisApp
    app = object.__new__(JarvisApp)
    done = EchoSession(limit=0)
    app.commander = SimpleNamespace(_pending_session=done, lecture_course="biosensors")
    app.assistant = SimpleNamespace(get=lambda k, d=None: 20)
    assert done.finished
    assert app._capture_window() == 20.0


# ------------------------------------------------------------ the planner
def _items(*titles):
    return [PlanItem(title=t) for t in titles]


def test_week_days_covers_calendar_days_not_only_weekdays():
    days = week_days(date(2026, 9, 4), days=3)      # Friday
    assert [d.strftime("%A") for d in days] == ["Friday", "Saturday", "Sunday"]


def test_collect_items_puts_dated_work_first_and_drops_the_past():
    now = datetime(2026, 8, 31, 9, 0).timestamp()
    later = datetime(2026, 9, 3, 23, 59)
    sooner = datetime(2026, 9, 1, 23, 59)
    past = datetime(2026, 8, 30, 23, 59)
    items = collect_items(
        [{"course": "BIOSENSORS", "title": "Lab 3 report", "due": later},
         {"course": "", "title": "Reading", "due": sooner},
         {"course": "X", "title": "Old thing", "due": past},
         {"course": "X", "title": "", "due": sooner}],
        [{"text": "buy milk"}, "call mum", {"text": "  "}], now=now)
    assert [i.title for i in items] == [
        "Reading", "Lab 3 report for BIOSENSORS", "buy milk", "call mum"]
    assert [i.source for i in items] == ["canvas", "canvas", "todo", "todo"]


def test_the_planner_proposes_takes_moves_and_skips():
    plan = WeekPlanner(items=_items("lab report", "reading", "buy milk"),
                       today=MONDAY)
    assert plan.ask(now=10.0) == "lab report — today at four?"
    assert plan.settle("yes") == "today at four, then."
    assert plan.ask(now=11.0) == "reading — today at seven?"
    assert plan.settle("move that to thursday") == "Thursday instead,"
    assert plan.ask(now=12.0) == "reading — Thursday at four?"
    assert plan.settle("yes")
    # Monday-at-seven was freed when the reading moved to Thursday
    assert plan.ask(now=13.0) == "buy milk — today at seven?"
    assert plan.settle("skip it") == dialogue.SKIP_LINE
    assert plan.finished
    assert [(s.day.strftime("%A"), s.hour) for s in plan.slots] == \
        [("Monday", 16), ("Thursday", 16)]


def test_a_bare_weekday_moves_the_proposal():
    plan = WeekPlanner(items=_items("lab report"), today=MONDAY)
    plan.ask(now=10.0)
    assert plan.settle("wednesday") == "Wednesday instead,"
    assert plan.proposal.day == MONDAY + timedelta(days=2)


def test_the_planner_never_proposes_past_a_due_date():
    due = datetime(2026, 9, 1, 23, 59).timestamp()          # Tuesday
    plan = WeekPlanner(items=[PlanItem("lab report", due=due)], today=MONDAY)
    line = plan.ask(now=10.0)
    assert "today" in line
    assert plan.settle("move that to friday") == NO_ROOM_LINE


def test_an_unplaceable_item_is_apologised_for_once_and_dropped():
    past_due = datetime(2026, 8, 31, 0, 1).timestamp()
    plan = WeekPlanner(items=[PlanItem("a", due=past_due - 86_400),
                              PlanItem("b", due=past_due - 86_400),
                              PlanItem("c")], today=MONDAY)
    line = plan.ask(now=10.0)
    assert line.count(NO_ROOM_LINE) == 1 and line.endswith("c — today at four?")


def test_an_unrelated_utterance_is_not_an_answer():
    plan = WeekPlanner(items=_items("lab report"), today=MONDAY)
    plan.ask(now=10.0)
    assert plan.settle("what's the weather like") is None


def test_the_plan_is_filed_once_at_the_end_and_read_back():
    filed = []
    plan = WeekPlanner(items=_items("lab report", "reading"), today=MONDAY,
                       filer=lambda slots: filed.extend(slots) or True)
    plan.ask(now=10.0)
    plan.settle("yes")
    plan.ask(now=11.0)
    plan.settle("yes")
    line = plan.stop()
    assert len(filed) == 2 and plan.filed is True
    assert line.startswith("That's the week, sir: today at four, lab report;")
    assert "I've set a reminder for each." in line


def test_a_failed_filing_is_admitted_not_papered_over():
    plan = WeekPlanner(items=_items("lab report"), today=MONDAY,
                       filer=lambda slots: (_ for _ in ()).throw(OSError("nope")))
    plan.ask(now=10.0)
    plan.settle("yes")
    assert "couldn't file" in plan.stop() and plan.filed is False


def test_nothing_agreed_files_nothing():
    calls = []
    plan = WeekPlanner(items=_items("lab report"), today=MONDAY,
                       filer=lambda slots: calls.append(slots) or True)
    plan.ask(now=10.0)
    plan.settle("no")
    assert plan.stop() == PLAN_EMPTY_LINE and calls == []


# ---------------------------------------------------- the plan-week command
def test_plan_week_is_registered_in_tier_one():
    assert "plan week" in {c.name for c in REGISTRY}
    assert "plan week" in {c.name for c in ASSISTANT_TIER1}


@pytest.mark.parametrize("phrase", [
    "let's plan the week", "lets plan the week", "plan my week",
    "plan the week ahead", "sort out my week",
])
def test_plan_week_phrases_match(phrase):
    assert commander._PLAN_WEEK_RX.match(phrase)


@pytest.mark.parametrize("phrase", [
    "how's my week looking", "plan a trip", "what's on this week",
])
def test_plan_week_does_not_shadow_the_week_briefing(phrase):
    assert not commander._PLAN_WEEK_RX.match(phrase)


def test_plan_week_opens_a_session_and_asks_the_first_question(cmdr, monkeypatch):
    monkeypatch.setattr(commander, "_plan_items",
                        lambda c: _items("lab report", "buy milk"))
    res = cmdr.handle("let's plan the week", source="voice")
    assert res.ack and res.done is False
    assert isinstance(cmdr._pending_session, WeekPlanner)
    line, speak = cmdr.services.replies[-1]
    assert speak and "2 things to place this week" in line and "lab report" in line
    # and the next utterance is its answer, not a command
    res = cmdr.handle("yes", source="voice")
    assert "today at four, then." in res.reply


def test_plan_week_with_nothing_to_place_says_so_and_opens_nothing(cmdr, monkeypatch):
    monkeypatch.setattr(commander, "_plan_items", lambda c: [])
    cmdr.handle("plan my week", source="voice")
    assert cmdr.services.replies[-1][0] == PLAN_NOTHING_LINE
    assert cmdr._pending_session is None


def test_plan_week_refuses_while_lecture_notes_are_open(cmdr, monkeypatch):
    monkeypatch.setattr(commander, "_plan_items", lambda c: _items("lab report"))
    cmdr.lecture_course = "biosensors"
    res = commander._h_plan_week(cmdr, "plan my week", None)
    assert res.reply == commander.PLAN_BUSY_LINE
    assert cmdr._pending_session is None


def test_file_plan_writes_reminders_and_survives_a_bad_slot():
    calls = []
    tk = SimpleNamespace(add_reminder=lambda due, text: calls.append((due, text)))
    c = SimpleNamespace(_svc=lambda n: tk if n == "timekeeper" else None)
    plan = WeekPlanner(items=_items("lab report"), today=MONDAY)
    plan.ask(now=10.0)
    plan.settle("yes")
    assert commander._file_plan(c, plan.slots) is True
    assert calls and calls[0][1] == "lab report"
    # the reminder leads the slot rather than landing on it
    assert calls[0][0] < plan.slots[0].when_epoch()
    c_none = SimpleNamespace(_svc=lambda n: None)
    assert commander._file_plan(c_none, plan.slots) is False
