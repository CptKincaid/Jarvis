"""The Aside (jarvis/aside.py): one volunteered sentence, on a budget.

The templates themselves are judged by reading a transcript, not by
assertion -- scratchpad/aside_dryrun.py replays twenty turns and prints
what he would have said. What is asserted here is everything that must
never drift: the anchor requirement, the two doors that must not both
speak, the quiet drop that costs no budget, the bucket, and the kill
phrase.

Fakes only: a calendar stub, a deadline snapshot list where Canvas would
be, a frozen clock. Nothing reaches the network, and the module never
touches canvas.fetch_due at all -- there is a test for that.
"""
import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import jarvis.aside as aside_mod
from jarvis.aside import AsideEngine, kill_phrase

TZ = ZoneInfo("America/Chicago")

# The fixtures below are built in America/Chicago and handed to the product
# as absolute timestamps, which the product -- rightly -- renders in the
# MACHINE's local zone. That made the file silently require the machine to
# be in Chicago: measured 2026-09-05, green there at all 24 hours and red in
# UTC, Tokyo, Kiritimati, Kolkata and London. A test about a house in Texas
# should SAY so rather than assume it.
pytestmark = pytest.mark.local_tz("America/Chicago")
MON = datetime(2026, 9, 14, tzinfo=TZ)


def at(day, hh, mm=0):
    return day.replace(hour=hh, minute=mm)


@pytest.fixture(autouse=True)
def _firewall(monkeypatch):
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)


class Cfg(dict):
    def get(self, key, default=None):
        node = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


class Cal:
    configured = True

    def __init__(self, events=()):
        self._events = list(events)

    def events(self):
        return list(self._events)


def _ev(title, start, minutes=90, all_day=False):
    return SimpleNamespace(title=title, start=start,
                           end=start + timedelta(minutes=minutes), all_day=all_day)


def item(kind, due, label=""):
    """What the timekeeper hands back: the anchor, and nothing else."""
    return SimpleNamespace(id="x", kind=kind, label=label,
                           due=due.timestamp(), created=0.0)


LAB = {"course": "BIOSENSORS", "title": "Lab 3 report", "due": at(MON, 23, 59)}


def _engine(tmp_path, now, items=(LAB,), events=(), quiet=None, **over):
    cfg = Cfg({"aside": dict({"enabled": True, "per_day": 2, "gap_min": 45,
                              "horizon_hours": 18}, **over)})
    return AsideEngine(cfg=cfg, get_calendar=Cal(events),
                       get_deadlines=SimpleNamespace(snapshot=lambda: list(items)),
                       quiet=quiet, state_path=tmp_path / "aside.json",
                       filed_paths=(tmp_path / "deadlines_state.json",),
                       now=lambda: now)


# ------------------------------------------------------------- the anchor
def test_a_reminder_earns_the_line_from_the_pitch(tmp_path):
    eng = _engine(tmp_path, at(MON, 8, 41))
    line = eng.consider("remind me to email my advisor at noon", "",
                        item("reminder", at(MON, 12, 0)))
    assert line == ("Incidentally, Lab 3 report for BIOSENSORS is due at "
                    "11:59 pm that night.")


def test_no_resolved_anchor_means_no_aside(tmp_path):
    """Correction 6: the reply text may contain the word "seven" and that
    is not a datetime. Guessing is how an aside lands on the wrong day."""
    eng = _engine(tmp_path, at(MON, 8, 41))
    assert eng.consider("wake me at seven", "Alarm for seven, sir.", None) is None
    assert eng.consider("x", "y", SimpleNamespace(kind="alarm")) is None
    assert eng.consider("x", "y", {"label": "seven"}) is None


def test_a_timer_is_a_countdown_not_a_plan(tmp_path):
    """The first dry run hung "you have BIOSENSORS lecture at 9:10" off a
    ten-minute kitchen timer. A timer names no point he is planning around."""
    eng = _engine(tmp_path, at(MON, 8, 13))
    assert eng.consider("set a timer for ten minutes", "",
                        item("timer", at(MON, 8, 23))) is None


def test_an_epoch_a_datetime_and_a_dict_all_resolve():
    when = at(MON, 12, 0)
    for action in (SimpleNamespace(due=when.timestamp()),
                   SimpleNamespace(start=when),
                   {"when": when}, {"due": when.timestamp()}):
        assert AsideEngine.anchor_of(action) is not None
    assert AsideEngine.anchor_of({"due": True}) is None    # a flag is not a time


# ------------------------------------------------------------- selection
def test_a_lecture_he_attends_every_week_is_not_news(tmp_path):
    """Ordinary calendar rows are excluded outright: the meeting heads-up
    already says them ten minutes ahead, and a weekly lecture volunteered
    three hours early is the tic this feature exists to avoid."""
    eng = _engine(tmp_path, at(MON, 8, 41), items=(),
                  events=(_ev("BIOSENSORS lecture", at(MON, 9, 10)),))
    assert eng.consider("wake me at nine", "", item("alarm", at(MON, 9, 0))) is None


def test_an_exam_on_the_calendar_is_news(tmp_path):
    eng = _engine(tmp_path, at(MON, 11, 5), items=(),
                  events=(_ev("Midterm 1", at(MON, 13, 0)),))
    line = eng.consider("remind me to leave at noon", "",
                        item("reminder", at(MON, 12, 0)))
    assert line == "Incidentally, Midterm 1 is at 1:00 pm that afternoon."


def test_nothing_before_the_anchor_is_ever_volunteered(tmp_path):
    eng = _engine(tmp_path, at(MON, 8, 0))
    assert eng.consider("wake me at seven tomorrow", "",
                        item("alarm", at(MON + timedelta(days=1), 7, 0))) is None


def test_beyond_the_horizon_is_not_that_night(tmp_path):
    eng = _engine(tmp_path, at(MON, 8, 41),
                  items=({"course": "THERMO", "title": "Problem set 6",
                          "due": at(MON + timedelta(days=3), 17)},))
    assert eng.consider("remind me at noon", "",
                        item("reminder", at(MON, 12, 0))) is None


# ------------------------------------------------------------- the doors
def test_what_the_heads_up_will_say_out_loud_is_never_volunteered(tmp_path):
    """Correction 3: DeadlineHeadsUp files a spoken timekeeper reminder for
    this very item. Without the cross-check the lab report gets a reminder
    at 09:00 and an aside at 09:20 -- the tic arriving by a second door."""
    key = f"BIOSENSORS|Lab 3 report|{LAB['due'].astimezone(ZoneInfo('UTC')).isoformat()}"
    (tmp_path / "deadlines_state.json").write_text(json.dumps({key: "x"}))
    eng = _engine(tmp_path, at(MON, 8, 41))
    assert eng.consider("remind me at noon", "", item("reminder", at(MON, 12))) is None


def test_the_same_thing_is_never_volunteered_twice(tmp_path):
    eng = _engine(tmp_path, at(MON, 8, 41))
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is not None
    later = _engine(tmp_path, at(MON, 14, 0))            # a fresh process
    assert later.consider("b", "", item("reminder", at(MON, 16))) is None


def test_a_second_candidate_survives_the_first_being_spent(tmp_path):
    """The first draft `return`ed on an already-said candidate, so a whole
    turn went silent because of one stale ledger row."""
    quiz = {"course": "BIOSENSORS", "title": "Reading quiz 4", "due": at(MON, 20, 0)}
    eng = _engine(tmp_path, at(MON, 8, 41), items=(quiz, LAB))
    first = eng.consider("a", "", item("reminder", at(MON, 12)))
    eng._state["spoken_at"] = []                     # budget aside, ledger kept
    second = eng.consider("b", "", item("reminder", at(MON, 12)))
    assert "Reading quiz 4" in first and "Lab 3 report" in second


# -------------------------------------------------------------- the gates
def test_quiet_drops_the_aside_and_costs_no_budget(tmp_path):
    """Correction 2: never _say(proactive=True). A held aside is drained
    into a catch-up digest hours later, detached from the question that
    provoked it -- the exact opposite of the effect."""
    quiet = SimpleNamespace(should_hold=lambda: True,
                            reason=lambda: "BIOSENSORS until 10:30",
                            hold=lambda *a, **k: pytest.fail("an aside was HELD"))
    eng = _engine(tmp_path, at(MON, 9, 40), quiet=quiet)
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is None
    assert eng._state["spoken_at"] == []             # the token was not spent


def test_the_budget_is_two_a_day_forty_five_minutes_apart(tmp_path):
    items = [{"course": "C", "title": f"Task {n}", "due": at(MON, 20 + 0, n)}
             for n in range(5)]
    clock = SimpleNamespace(now=at(MON, 8, 0))
    eng = _engine(tmp_path, None, items=items)
    eng._now = lambda: clock.now
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is not None
    clock.now = at(MON, 8, 30)
    assert eng.consider("b", "", item("reminder", at(MON, 12))) is None   # the gap
    clock.now = at(MON, 9, 0)
    assert eng.consider("c", "", item("reminder", at(MON, 12))) is not None
    clock.now = at(MON, 11, 0)
    assert eng.consider("d", "", item("reminder", at(MON, 12))) is None   # 2/2
    clock.now = at(MON + timedelta(days=1), 8, 0)
    assert eng.consider("e", "", item("reminder",
                                      at(MON + timedelta(days=1), 12))) is None


def test_both_budget_numbers_are_config(tmp_path):
    eng = _engine(tmp_path, at(MON, 8, 41), per_day=0)
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is None


def test_off_by_default(tmp_path):
    eng = AsideEngine(cfg=Cfg({}), get_calendar=Cal(),
                      get_deadlines=SimpleNamespace(snapshot=lambda: [LAB]),
                      state_path=tmp_path / "a.json", now=lambda: at(MON, 8, 41))
    assert not eng.enabled
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is None


# --------------------------------------------------------------- the kill
def test_the_kill_phrase_is_specific():
    for phrase in ("no more asides", "that's enough of that", "no more of that",
                   "Jarvis, no more asides."):
        assert kill_phrase(phrase), phrase
    # "stop that" is barge-in AND read-aloud steering; it must never be this.
    for phrase in ("stop that", "stop", "quiet", "skip", "enough"):
        assert not kill_phrase(phrase), phrase


def test_the_kill_zeroes_the_day_and_is_counted(tmp_path, caplog):
    eng = _engine(tmp_path, at(MON, 8, 41))
    with caplog.at_level("INFO"):
        assert eng.silence("no more asides") == aside_mod.KILL_LINE
    assert any("aside: silenced for the day" in r.message for r in caplog.records)
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is None
    # …but only for the day.
    tomorrow = _engine(tmp_path, at(MON + timedelta(days=1), 8, 41),
                       items=({"course": "C", "title": "T",
                               "due": at(MON + timedelta(days=1), 20)},))
    assert tomorrow.consider("a", "",
                             item("reminder", at(MON + timedelta(days=1), 12))) is not None


def test_the_silenced_line_matches_the_day_review_counter():
    from jarvis.dayreview import COUNTERS
    rx = dict(COUNTERS)["asides_silenced"]
    assert rx.search("aside: silenced for the day by 'no more asides' (1 on record)")


# ---------------------------------------------------------------- state
def test_a_hand_edited_state_file_does_not_raise(tmp_path):
    """headsup.py learned this from a real file: str -> ISO only."""
    (tmp_path / "aside.json").write_text(
        '{"said": {"k": null, "ok": "2026-09-14T00:00:00+00:00"},'
        ' "spoken_at": [1, "2026-09-14T06:00:00-05:00"], "silenced_on": 7}')
    eng = _engine(tmp_path, at(MON, 8, 41))
    assert eng._state["said"] == {"ok": "2026-09-14T00:00:00+00:00"}
    assert eng._state["silenced_on"] == ""
    # The one good stamp still counts against the budget (1 of 2 today, and
    # far enough back to clear the gap), so this turn may speak.
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is not None
    assert eng.consider("b", "", item("reminder", at(MON, 12))) is None   # 2/2


def test_the_state_write_is_atomic(tmp_path):
    eng = _engine(tmp_path, at(MON, 8, 41))
    eng.consider("a", "", item("reminder", at(MON, 12)))
    assert json.loads((tmp_path / "aside.json").read_text())["said"]
    assert not (tmp_path / "aside.json.tmp").exists()


def test_the_reply_path_never_fetches(tmp_path, monkeypatch):
    """Correction 1: canvas.fetch_due is a live REST round-trip and this
    runs between the question and the answer."""
    import jarvis.tools.canvas as canvas_mod
    monkeypatch.setattr(canvas_mod, "fetch_due",
                        lambda *a, **k: pytest.fail("the aside path fetched Canvas"))
    eng = _engine(tmp_path, at(MON, 8, 41))
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is not None


def test_a_source_that_raises_never_breaks_the_turn(tmp_path):
    class Boom:
        def snapshot(self):
            raise RuntimeError("canvas down")
    eng = _engine(tmp_path, at(MON, 8, 41))
    eng._get_deadlines = Boom()
    assert eng.consider("a", "", item("reminder", at(MON, 12))) is None
