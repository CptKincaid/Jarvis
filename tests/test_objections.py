"""Reasoned dissent (jarvis/objections.py + the commander half).

Two halves, tested separately because they fail differently. The
predicates are pure functions of explicit datetimes and rows -- no clock,
no services -- and each must name the row it objected from. The resolver
is about the INVERTED default: unlike the destructive read-back, an
unclear reply, a changed subject and silence must all end with the alarm
set, because the alarm was asked for and only the opinion was volunteered.

Fakes only: a timekeeper stand-in with a deterministic parse_when /
describe_due (so "wake me at two" resolves the same on every run), a stub
calendar, and a snapshot list where Canvas would be. Nothing reaches the
network.
"""
import os
import types
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

import jarvis.objections as ob
from jarvis.commander import Commander, CommandResult, IntentClassifier
from jarvis.config import CONFIG
from jarvis.router import Router

TZ = ZoneInfo("America/Chicago")

# The fixtures below are built in America/Chicago and handed to the product
# as absolute timestamps, which the product -- rightly -- renders in the
# MACHINE's local zone. That made the file silently require the machine to
# be in Chicago: measured 2026-09-05, green there at all 24 hours and red in
# UTC, Tokyo, Kiritimati, Kolkata and London. A test about a house in Texas
# should SAY so rather than assume it.
pytestmark = pytest.mark.local_tz("America/Chicago")
NOW = datetime(2026, 9, 14, 21, 30, tzinfo=TZ)       # a Monday evening
TWO_AM = datetime(2026, 9, 15, 2, 0, tzinfo=TZ)


@pytest.fixture(autouse=True)
def _firewall(monkeypatch):
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)


def _ev(title, start, minutes=80, all_day=False):
    return SimpleNamespace(title=title, start=start,
                           end=start + timedelta(minutes=minutes), all_day=all_day)


def _alarm(due, label=""):
    return SimpleNamespace(kind="alarm", due=due.timestamp(),
                           effective_due=due.timestamp(), label=label)


# --------------------------------------------------------- the predicates
def test_a_small_hours_alarm_names_the_lecture_and_the_hours():
    lecture = _ev("BIOSENSORS lecture", datetime(2026, 9, 15, 9, 10, tzinfo=TZ))
    obj = ob.sleep_window(TWO_AM, [lecture], NOW)
    assert obj is not None
    assert obj.reason == ("your BIOSENSORS lecture is at 9:10 am and that "
                          "leaves you under four and a half hours")
    assert "BIOSENSORS lecture" in obj.row      # the objection can point at its row
    assert obj.line("a 2:00 am alarm").startswith(
        "I would advise against a 2:00 am alarm, sir; your BIOSENSORS")
    assert obj.line("x").endswith("Shall I set it anyway?")


def test_a_small_hours_alarm_with_a_free_day_is_not_objected_to():
    """No calendar row = no reason. "You'll be tired" is a preference about
    bedtimes, and that is the insufferable objection this module refuses."""
    assert ob.sleep_window(TWO_AM, [], NOW) is None


def test_an_all_day_row_is_never_the_reason():
    holiday = _ev("Rosh Hashanah", datetime(2026, 9, 15, tzinfo=TZ), all_day=True)
    assert ob.sleep_window(TWO_AM, [holiday], NOW) is None


def test_a_normal_morning_alarm_is_never_objected_to():
    lecture = _ev("BIOSENSORS lecture", datetime(2026, 9, 15, 9, 10, tzinfo=TZ))
    seven = datetime(2026, 9, 15, 7, 0, tzinfo=TZ)
    assert ob.sleep_window(seven, [lecture], NOW) is None


def test_plenty_of_sleep_before_a_small_hours_alarm_passes():
    """Set at lunchtime for 2 am: he is not losing sleep he has not had."""
    noon = datetime(2026, 9, 14, 12, 0, tzinfo=TZ)
    lecture = _ev("BIOSENSORS lecture", datetime(2026, 9, 15, 9, 10, tzinfo=TZ))
    assert ob.sleep_window(TWO_AM, [lecture], noon) is None


def test_a_duplicate_alarm_names_the_one_already_set():
    existing = _alarm(datetime(2026, 9, 15, 6, 55, tzinfo=TZ), "lecture")
    obj = ob.duplicate_item(datetime(2026, 9, 15, 7, 0, tzinfo=TZ), [existing])
    assert obj is not None and obj.source == "duplicate alarm"
    assert obj.reason == "you already have an alarm at 6:55 am (lecture)"


def test_an_alarm_an_hour_from_another_is_not_a_duplicate():
    existing = _alarm(datetime(2026, 9, 15, 6, 0, tzinfo=TZ))
    assert ob.duplicate_item(datetime(2026, 9, 15, 7, 0, tzinfo=TZ), [existing]) is None


def test_a_timer_is_never_matched_by_the_alarm_duplicate_rule():
    timer = SimpleNamespace(kind="timer", due=datetime(2026, 9, 15, 7, 0,
                                                       tzinfo=TZ).timestamp(),
                            effective_due=None, label="")
    assert ob.duplicate_item(datetime(2026, 9, 15, 7, 0, tzinfo=TZ), [timer]) is None


def test_an_alarm_set_after_a_deadline_names_the_canvas_row():
    due = datetime(2026, 9, 15, 8, 0, tzinfo=TZ)
    items = [{"course": "BIOSENSORS", "title": "Lab 3 report",
              "due": datetime(2026, 9, 15, 0, 0, tzinfo=TZ)}]
    obj = ob.deadline_clash(due, items, NOW)
    assert obj is not None
    assert obj.reason == "Lab 3 report for BIOSENSORS is due at 12:00 am, before that"


def test_a_deadline_after_the_alarm_is_not_a_clash():
    due = datetime(2026, 9, 15, 8, 0, tzinfo=TZ)
    items = [{"course": "BIOSENSORS", "title": "Lab 3 report",
              "due": datetime(2026, 9, 15, 23, 59, tzinfo=TZ)}]
    assert ob.deadline_clash(due, items, NOW) is None


def test_a_quiet_window_the_alarm_lands_in_is_named():
    quiet = SimpleNamespace(reason=lambda ts: "BIOSENSORS until 10:30")
    obj = ob.quiet_conflict(datetime(2026, 9, 15, 9, 30, tzinfo=TZ), quiet)
    assert obj is not None and obj.reason == "you're BIOSENSORS until 10:30 then"


def test_being_out_is_not_an_objection():
    quiet = SimpleNamespace(reason=lambda ts: "you're out")
    assert ob.quiet_conflict(datetime(2026, 9, 15, 9, 30, tzinfo=TZ), quiet) is None


def test_the_most_concrete_row_wins():
    """A duplicate beats a reason he has to reason about."""
    lecture = _ev("BIOSENSORS lecture", datetime(2026, 9, 15, 9, 10, tzinfo=TZ))
    existing = _alarm(TWO_AM, "")
    obj = ob.for_alarm(TWO_AM, NOW, events=[lecture], pending=[existing])
    assert obj.source == "duplicate alarm"


def test_a_source_that_raises_never_takes_the_alarm_down():
    class Exploding:
        @property
        def start(self):
            raise RuntimeError("bad row")
        all_day = False
        title = "boom"
    assert ob.for_alarm(TWO_AM, NOW, events=[Exploding()]) is None


def test_the_ledger_forgets_tomorrow_but_not_this_afternoon(tmp_path):
    clock = SimpleNamespace(now=NOW)
    ledger = ob.ObjectionLedger(tmp_path / "obj.json", now=lambda: clock.now)
    obj = ob.Objection("sleep window", "r", "row", "sleep|x")
    assert not ledger.already_said(obj)
    ledger.record(obj)
    assert ledger.already_said(obj)
    # A restart re-reads the file rather than starting clean.
    assert ob.ObjectionLedger(tmp_path / "obj.json",
                             now=lambda: clock.now).already_said(obj)
    clock.now = NOW + timedelta(days=1)
    assert not ledger.already_said(obj)


def test_a_hand_edited_ledger_does_not_raise(tmp_path):
    """headsup.py learned this from a real file: one stray null used to
    raise on every pass before the save could run."""
    path = tmp_path / "obj.json"
    path.write_text('{"2026-09-14|sleep|x": null, "ok": "2026-09-14T21:30:00"}')
    ledger = ob.ObjectionLedger(path, now=lambda: NOW)
    ledger.record(ob.Objection("s", "r", "row", "k"))
    assert "ok" in ledger._seen


def test_override_words_the_yes_no_parser_cannot_read():
    from jarvis.commander import parse_yes_no
    for phrase in ("set it anyway", "anyway", "I know", "regardless",
                   "even so", "that's fine"):
        assert parse_yes_no(phrase) is not True, phrase   # the gap being filled
        assert ob.is_override(phrase), phrase
    for phrase in ("no", "never mind", "what's the weather"):
        assert not ob.is_override(phrase), phrase


# ------------------------------------------------------------- the wiring
class Cfg:
    def __init__(self, **over):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


class TK:
    """Deterministic: "wake me at two" is always tomorrow at 2 am."""
    ringing = None

    def __init__(self):
        self.added = []

    def parse_when(self, when, now):
        return TWO_AM

    def describe_due(self, due, now):
        return "for 2:00 am tomorrow"

    def add_alarm(self, due, label="", repeat="once"):
        self.added.append((due, label, repeat))
        return SimpleNamespace(kind="alarm", due=due, label=label)

    def list(self, kind="all", include_done=False):
        return []


@pytest.fixture
def rich(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    spoken = []
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(), assistant=Cfg(),
        timekeeper=TK(), notes=None, approvals=MagicMock(), claude=MagicMock(),
        calendar=None, deadlines=None, quiet=None, alarm_offer=None,
        speak=lambda text, proactive=True, kind="message": spoken.append(
            (text, proactive, kind)))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    c = Commander(svc)
    c._objections = ob.ObjectionLedger(tmp_path / "obj.json")
    return c, svc, spoken


def _object_with(monkeypatch, reason="your BIOSENSORS lecture is at 9:10 am"):
    obj = ob.Objection("sleep window", reason, "calendar BIOSENSORS lecture",
                       "sleep|BIOSENSORS")
    monkeypatch.setattr(ob, "for_alarm", lambda *a, **k: obj)
    return obj


def test_an_unobjected_alarm_and_an_overruled_one_are_byte_identical(rich, monkeypatch):
    """The whole point of splitting _h_alarm into a stashed closure: the
    objection must change WHEN the alarm is set, never WHAT is said once it
    is. A drift here means the user is punished for being argued with."""
    c, svc, _ = rich
    plain = c.handle("wake me at two", source="typed")
    assert plain.reply == "Alarm for 2:00 am tomorrow, sir."

    _object_with(monkeypatch)
    offered = c.handle("wake me at two", source="typed")
    assert offered.reply.startswith("I would advise against a 2:00 am alarm, sir;")
    assert offered.status == "Advising against"
    assert len(svc.timekeeper.added) == 1          # not set yet

    overruled = c.handle("set it anyway", source="typed")
    assert (overruled.reply, overruled.speak, overruled.status) == \
           (plain.reply, plain.speak, plain.status)
    assert len(svc.timekeeper.added) == 2


def test_a_plain_yes_also_sets_it(rich, monkeypatch):
    c, svc, _ = rich
    _object_with(monkeypatch)
    c.handle("wake me at two", source="typed")
    res = c.handle("yes", source="typed")
    assert res.reply == "Alarm for 2:00 am tomorrow, sir."
    assert len(svc.timekeeper.added) == 1


def test_an_explicit_no_is_the_only_way_to_lose_the_alarm(rich, monkeypatch):
    c, svc, _ = rich
    _object_with(monkeypatch)
    c.handle("wake me at two", source="typed")
    res = c.handle("no, forget it", source="typed")
    assert res.reply == ob.DROPPED_LINE
    assert svc.timekeeper.added == []


def test_a_changed_subject_sets_the_alarm_and_keeps_its_own_meaning(rich, monkeypatch):
    """The inverted default. _try_destructive_confirm would DROP here --
    right for a delete, wrong for something he asked for out loud. The
    alarm is set, he is told so, and "what time is it" is still answered as
    the question it is."""
    c, svc, spoken = rich
    _object_with(monkeypatch)
    c.handle("wake me at two", source="typed")
    res = c.handle("what time is it", source="typed")
    assert len(svc.timekeeper.added) == 1
    assert spoken and spoken[0][0].startswith("Setting it anyway, sir.")
    assert spoken[0][1] is False              # never held for a digest
    assert "It's" in (res.reply or "")         # the clock still answered


def test_silence_sets_the_alarm_too(rich, monkeypatch):
    """A man who heard the objection, agreed with it and went to bed must
    not wake up to no alarm at all."""
    c, svc, spoken = rich
    _object_with(monkeypatch)
    monkeypatch.setattr(c, "_objection_cancel_timer", lambda: None)
    c.handle("wake me at two", source="typed")
    c._objection_timer.cancel()
    res = c.objection_timeout()
    assert len(svc.timekeeper.added) == 1
    assert res.reply == "Alarm for 2:00 am tomorrow, sir."
    assert spoken[0][0] == "Setting it anyway, sir. Alarm for 2:00 am tomorrow, sir."
    assert spoken[0][1] is False              # never held for a digest


def test_he_never_objects_to_the_same_thing_twice_in_a_day(rich, monkeypatch):
    c, svc, _ = rich
    _object_with(monkeypatch)
    first = c.handle("wake me at two", source="typed")
    assert first.status == "Advising against"
    c.handle("set it anyway", source="typed")
    second = c.handle("wake me at two", source="typed")
    assert second.reply == "Alarm for 2:00 am tomorrow, sir."


def test_dissent_can_be_switched_off(rich, monkeypatch):
    c, svc, _ = rich
    _object_with(monkeypatch)
    svc.assistant.data["confirm.dissent"] = False
    res = c.handle("wake me at two", source="typed")
    assert res.reply == "Alarm for 2:00 am tomorrow, sir."


def test_another_open_question_is_never_talked_over(rich, monkeypatch):
    """Correction 5: a dissent offer and an existing offer can never both be
    live -- two questions and one "yes" is a coin toss. The pipeline mostly
    guarantees it (both earlier stages clear their slot on an unrelated
    utterance), so the guard is asserted where it lives."""
    c, svc, _ = rich
    _object_with(monkeypatch)
    # clock-hygiene: the wall clock is the FIXTURE here, not the
    # expectation -- `now` is handed straight to objection_for_alarm as
    # its own clock, and every assertion is relative to it (is None /
    # is not None), so no hour is written down anywhere. Measured
    # 2026-09-05 under libfaketime at all 24 hours: 131 passed each
    # time. This file is zone-pinned, so --clock-at cannot re-check
    # that for you; re-measure by hand if this test grows an
    # assertion about a particular time of day.
    now = datetime.now().astimezone()
    due = (now + timedelta(hours=3)).timestamp()
    c.stash_destructive(lambda: CommandResult(handled=True), "Cancel all three, sir?")
    assert c.objection_for_alarm(due, now) is None
    c._pending_destructive = None
    svc.alarm_offer = {"due": due, "time": "7:00 am"}
    assert c.objection_for_alarm(due, now) is None
    svc.alarm_offer = None
    assert c.objection_for_alarm(due, now) is not None


def test_a_timer_is_never_objected_to(rich, monkeypatch):
    """Rule 4: nothing reversible in under a minute earns an argument."""
    c, svc, _ = rich
    called = []
    monkeypatch.setattr(ob, "for_alarm", lambda *a, **k: called.append(1))
    c.handle("set a timer for five minutes", source="typed")
    assert called == []
