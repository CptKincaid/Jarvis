"""Calendar anomaly watch (jarvis/calwatch.py).

The diff is pure and gets the bulk of the coverage: the four kinds of
change, the relocation a key diff cannot see, the sliding-window
intersection that would otherwise report a mass cancellation every
midnight, and the removal fuse. The watch itself is exercised against a
fake CalendarSource -- no network, no threads beyond an explicit start/stop.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from jarvis.calwatch import (ADDED, CANCELLED, MOVED, RELOCATED, CalendarWatch,
                             Change, change_id, change_line, diff_events,
                             overlap, spoken_horizon)
from jarvis.tools.calendar import Event

TZ = timezone(timedelta(hours=-5))
DAY = datetime(2026, 8, 30, 0, 0, tzinfo=TZ)
WINDOW = (DAY - timedelta(days=1), DAY + timedelta(days=14))


def ev(hour, title="BIOSENSORS", location="ETB 1003", day=0, minute=10,
       dur=50):
    start = DAY + timedelta(days=day, hours=hour, minutes=minute)
    return Event(start=start, end=start + timedelta(minutes=dur),
                 title=title, location=location)


def now_at(dt=DAY + timedelta(hours=7)):
    return lambda: dt


# ------------------------------------------------------------------ diff
def test_an_unchanged_calendar_produces_nothing():
    events = [ev(9), ev(11, "SEMINAR"), ev(14, "LAB", location="")]
    assert diff_events(events, list(events), WINDOW) == []


def test_a_vanished_event_is_a_cancellation():
    before = [ev(9), ev(11, "SEMINAR")]
    changes = diff_events(before, [ev(11, "SEMINAR")], WINDOW)
    assert [c.kind for c in changes] == [CANCELLED]
    assert changes[0].event.title == "BIOSENSORS"


def test_a_new_event_is_an_addition():
    changes = diff_events([ev(9)], [ev(9), ev(15, "ADVISOR")], WINDOW)
    assert [c.kind for c in changes] == [ADDED]
    assert changes[0].event.title == "ADVISOR"


def test_a_push_is_one_move_not_a_cancellation_plus_an_addition():
    """Event.key() carries the times, so a push is gone AND appeared; the
    pairing pass is what keeps it from being read as a cancellation."""
    changes = diff_events([ev(9)], [ev(10)], WINDOW)
    assert [c.kind for c in changes] == [MOVED]
    assert changes[0].prev.start.hour == 9 and changes[0].event.start.hour == 10


def test_a_room_change_is_invisible_to_a_key_diff_and_is_still_found():
    before, after = [ev(9)], [ev(9, location="ETB 1020")]
    assert before[0].key() == after[0].key()          # location is NOT in the key
    changes = diff_events(before, after, WINDOW)
    assert [c.kind for c in changes] == [RELOCATED]
    assert changes[0].prev.location == "ETB 1003"
    assert changes[0].event.location == "ETB 1020"


def test_a_move_that_also_changes_the_room_reports_the_move():
    changes = diff_events([ev(9)], [ev(10, location="ETB 1020")], WINDOW)
    assert [c.kind for c in changes] == [MOVED]


def test_a_same_title_event_on_another_day_is_not_a_move():
    """The pairing pass matches on title AND date; next week's lecture is
    not this morning's lecture moved."""
    changes = diff_events([ev(9)], [ev(9, day=7)], WINDOW)
    assert sorted(c.kind for c in changes) == [ADDED, CANCELLED]


def test_events_outside_the_window_are_not_compared():
    outside = ev(9, day=60)
    assert diff_events([outside], [], WINDOW) == []
    assert diff_events([], [outside], WINDOW) == []


def test_the_midnight_window_roll_is_not_a_mass_cancellation():
    """_window is anchored at yesterday-midnight and slides every day, so
    without the intersection everything on the trailing edge reads as
    cancelled and the new far edge as newly scheduled."""
    old_win = (DAY - timedelta(days=1), DAY + timedelta(days=14))
    new_win = (DAY, DAY + timedelta(days=15))
    leaving = ev(9, day=-1)                 # yesterday: in old, out of new
    staying = ev(9)
    arriving = ev(9, "FAR EDGE", day=14)    # the far edge that just entered
    before, after = [leaving, staying], [staying, arriving]
    naive = diff_events(before, after, old_win)
    assert sorted(c.kind for c in naive) == [CANCELLED]
    assert diff_events(before, after, overlap(old_win, new_win)) == []


def test_overlap_is_none_when_the_windows_do_not_meet():
    assert overlap(None, WINDOW) is None
    assert overlap(WINDOW, None) is None
    far = (DAY + timedelta(days=40), DAY + timedelta(days=50))
    assert overlap(WINDOW, far) is None


# --------------------------------------------------------------- wording
def test_the_spoken_lines_name_the_time_and_the_change():
    today = DAY.date()
    assert change_line(Change(CANCELLED, ev(9)), today) == \
        "Your 9:10 am BIOSENSORS has been cancelled, sir."
    assert change_line(Change(MOVED, ev(10), ev(9)), today) == \
        "Your 9:10 am BIOSENSORS has moved to 10:10 am, sir."
    assert change_line(Change(RELOCATED, ev(9, location="ETB 1020"), ev(9)),
                       today) == \
        "Your 9:10 am BIOSENSORS has moved to ETB 1020, sir."
    assert change_line(Change(ADDED, ev(15, "ADVISOR")), today) == \
        "ADVISOR has been added to today at 3:10 pm, sir."


def test_tomorrows_change_says_so():
    today = DAY.date()
    line = change_line(Change(CANCELLED, ev(9, day=1)), today)
    assert line.startswith("Tomorrow's 9:10 am")


def test_only_today_and_tomorrow_are_spoken():
    today = DAY.date()
    assert spoken_horizon(Change(CANCELLED, ev(9)), today)
    assert spoken_horizon(Change(CANCELLED, ev(9, day=1)), today)
    assert not spoken_horizon(Change(CANCELLED, ev(9, day=3)), today)
    # a move is dated by where it WAS: he is waiting at the old time
    assert spoken_horizon(Change(MOVED, ev(9, day=4), ev(9)), today)


def test_change_id_separates_the_kinds_and_the_shapes():
    a = change_id(Change(CANCELLED, ev(9)))
    assert a != change_id(Change(ADDED, ev(9)))
    assert a != change_id(Change(CANCELLED, ev(10)))
    assert a == change_id(Change(CANCELLED, ev(9)))


# ----------------------------------------------------------------- watch
class FakeCal:
    """A CalendarSource with the three members the watch touches."""

    def __init__(self, events, window=WINDOW, ok=True):
        self.configured = True
        self._events, self._window, self.ok = list(events), window, ok
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1
        return self.ok

    def events(self):
        return list(self._events)

    def window(self):
        return self._window


class Spy:
    def __init__(self):
        self.said, self.held = [], []

    def say(self, text, proactive=False, kind="message"):
        self.said.append((text, proactive, kind))

    def hold(self, text, kind="message"):
        self.held.append((text, kind))


def _watch(cal, tmp_path, spy=None, **kw):
    spy = spy or Spy()
    w = CalendarWatch(lambda: cal, say=spy.say, quiet=spy,
                      state_path=tmp_path / "calwatch.json",
                      now=now_at(), **kw)
    w.spy = spy
    return w


def test_the_first_tick_learns_and_says_nothing(tmp_path):
    cal = FakeCal([ev(9)])
    w = _watch(cal, tmp_path)
    assert w.tick() == [] and w.spy.said == []
    assert (tmp_path / "calwatch.json").exists()


def test_a_cancellation_between_two_ticks_is_spoken_once(tmp_path):
    cal = FakeCal([ev(9), ev(11, "SEMINAR")])
    w = _watch(cal, tmp_path)
    w.tick()
    cal._events = [ev(11, "SEMINAR")]
    changes = w.tick()
    assert [c.kind for c in changes] == [CANCELLED]
    assert w.spy.said == [("Your 9:10 am BIOSENSORS has been cancelled, sir.",
                           True, "message")]
    # the same absence on the next refresh is not news
    assert w.tick() == [] and len(w.spy.said) == 1


def test_only_one_line_is_spoken_the_rest_wait_for_the_digest(tmp_path):
    cal = FakeCal([ev(9), ev(11, "SEMINAR"), ev(13, "LAB")])
    w = _watch(cal, tmp_path)
    w.tick()
    cal._events = []
    w.tick()
    assert len(w.spy.said) == 1 and len(w.spy.held) == 2
    assert all(kind == "message" for _, kind in w.spy.held)


def test_a_failed_refresh_is_never_a_day_of_cancellations(tmp_path):
    cal = FakeCal([ev(9), ev(11, "SEMINAR")])
    w = _watch(cal, tmp_path)
    w.tick()
    cal.ok, cal._events = False, []
    assert w.tick() == [] and w.spy.said == []


def test_the_removal_fuse_suppresses_an_implausible_diff(tmp_path):
    cal = FakeCal([ev(9 + i, f"CLASS {i}") for i in range(6)])
    w = _watch(cal, tmp_path, max_removed=4)
    w.tick()
    cal._events = []
    assert w.tick() == [] and w.spy.said == []


def test_changes_beyond_tomorrow_are_filed_silently(tmp_path):
    cal = FakeCal([ev(9, day=5)])
    w = _watch(cal, tmp_path)
    w.tick()
    cal._events = []
    changes = w.tick()
    assert [c.kind for c in changes] == [CANCELLED]
    assert w.spy.said == [] and w.spy.held == []
    assert w.tick() == []                     # and still announced only once


def test_state_survives_a_restart(tmp_path):
    cal = FakeCal([ev(9), ev(11, "SEMINAR")])
    w = _watch(cal, tmp_path)
    w.tick()
    cal._events = [ev(11, "SEMINAR")]
    w.tick()
    fresh = _watch(cal, tmp_path)
    assert fresh.tick() == [] and fresh.spy.said == []


def test_a_hand_mangled_state_file_costs_one_silent_cycle(tmp_path):
    path = tmp_path / "calwatch.json"
    path.write_text(json.dumps({"snapshot": {"events": [{"start": "nope"}]},
                                "announced": {"a": 1, "b": "2026-08-30"}}))
    cal = FakeCal([ev(9)])
    w = CalendarWatch(lambda: cal, say=None, state_path=path, now=now_at())
    assert w.tick() == []                     # no crash; the snapshot is relearned
    assert w.tick() == []


def test_an_unconfigured_or_missing_calendar_is_silent(tmp_path):
    w = CalendarWatch(lambda: None, state_path=tmp_path / "s.json", now=now_at())
    assert w.tick() == []
    cal = FakeCal([ev(9)])
    cal.configured = False
    w2 = CalendarWatch(lambda: cal, state_path=tmp_path / "s2.json", now=now_at())
    assert w2.tick() == [] and cal.refreshes == 0


def test_the_watch_can_be_switched_off(tmp_path):
    class Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)
    cal = FakeCal([ev(9)])
    w = CalendarWatch(lambda: cal, cfg=Cfg({"calendar.anomaly_watch": False}),
                      state_path=tmp_path / "s.json", now=now_at())
    assert w.tick() == [] and cal.refreshes == 0


def test_a_raising_calendar_does_not_kill_the_tick(tmp_path):
    class Boom(FakeCal):
        def refresh(self):
            raise RuntimeError("caldav on fire")
    w = _watch(Boom([ev(9)]), tmp_path)
    assert w.tick() == []


def test_start_stop_joins_the_thread(tmp_path):
    w = _watch(FakeCal([ev(9)]), tmp_path)
    w.start()
    assert w._thread is not None and w._thread.is_alive()
    w.start()                                 # alive-guard: no second thread
    w.stop()
    assert not w._thread.is_alive()


@pytest.mark.parametrize("kind", [CANCELLED, ADDED, MOVED, RELOCATED])
def test_every_kind_renders_a_sentence(kind):
    ch = Change(kind, ev(9, location="ETB 1020"), ev(9))
    line = change_line(ch, DAY.date())
    assert line.endswith("sir.") and len(line) > 20
