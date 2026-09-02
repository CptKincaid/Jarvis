"""Tests for jarvis.tools.timekeeper (spec 2026-08-26, section 6.1).

Pure tests: parse_when phrase table (fixed `now`), describe_due wording,
duration coercion.  Scheduler tests drive `tick()` with a fake clock —
no thread, no sleeps beyond 0.2 s, no audio: the subprocess seam `_run`
is a recorder, speech is a list, bus events are captured.

Firewall: every Timekeeper here gets a tmp db and a tmp cache dir;
tests/conftest.py already points JARVIS_LOG_DIR / JARVIS_ASSISTANT_CONFIG /
JARVIS_CACHE_DIR at throwaway dirs.  Nothing touches /tmp/vss_voice.
"""
import json
import os
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis import events
from jarvis.tools import timekeeper as tkm
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.timekeeper import (
    ALARM_DEFAULT_LABEL, CANT_PARSE_LINE, HOW_LONG_LINE, NO_TIMEKEEPER_LINE,
    NOTHING_RINGING_LINE, Timekeeper, describe_due, make_tools, parse_duration,
    parse_when, parse_when_full, sentence_case, split_when, write_alarm_wav,
)

LIVE = Path("/tmp/vss_voice")
NOW = datetime(2026, 8, 26, 9, 0)          # Wednesday 9:00 am
D = lambda *a: datetime(*a)                # noqa: E731


def test_firewall():
    from jarvis import logs
    assert logs.LOG_DIR != LIVE
    assert not str(os.environ.get("JARVIS_ASSISTANT_CONFIG", "")).startswith(
        str(Path.home() / ".config" / "jarvis"))


# ---------------------------------------------------------------- helpers
class FakeClock:
    def __init__(self, start: datetime):
        self.t = start.timestamp()

    def now(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds
        return self.t

    def set(self, dt: datetime):
        self.t = dt.timestamp()
        return self.t


class Recorder:
    """Replaces the `_run` seam: records argv, never runs anything."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, background=False, timeout=30.0):
        self.calls.append(list(argv))
        return None

    def of(self, prog):
        return [c for c in self.calls if c and c[0] == prog]


class FakeAssistantConfig:
    """The slice of W1's AssistantConfig the timekeeper uses."""

    def __init__(self, **alarms):
        self.data = {"alarms": {"sound": "", "volume": 0.8, "escalate": True,
                                "max_ring_s": 300, "snooze_min": 10}}
        self.data["alarms"].update(alarms)

    def get(self, dotted, default=None):
        obj = self.data
        for part in dotted.split("."):
            if not isinstance(obj, dict) or part not in obj:
                return default
            obj = obj[part]
        return obj


@pytest.fixture
def capture():
    """Captured bus events; the bus delivers synchronously (no Tk root)."""
    got = []
    subs = []
    for etype in (events.ReminderFired, events.AlarmFired, events.AlarmStopped):
        fn = events.bus.subscribe(etype, got.append)
        subs.append((etype, fn))
    yield got
    for etype, fn in subs:
        events.bus.unsubscribe(etype, fn)


@pytest.fixture
def tk(tmp_path, capture):
    """A Timekeeper on a fake clock with recorded speech + subprocesses."""
    clock = FakeClock(NOW)
    said = []
    run = Recorder()
    cfg = FakeAssistantConfig(volume=0.5)
    t = Timekeeper(tmp_path / "tk.db", say=said.append, cfg=cfg, now=clock.now,
                   run=run, tick_s=0.01, ring=True, cache_dir=tmp_path / "cache")
    t.clock, t.said, t.run_rec, t.events = clock, said, run, capture
    yield t
    t.close()


def kinds(evs, cls):
    return [e for e in evs if isinstance(e, cls)]


# ------------------------------------------------------------ parse_when
@pytest.mark.parametrize("text,expected", [
    ("in 10 minutes", D(2026, 8, 26, 9, 10)),
    ("in 5 mins", D(2026, 8, 26, 9, 5)),
    ("in 90 seconds", D(2026, 8, 26, 9, 1, 30)),
    ("in 2 hours", D(2026, 8, 26, 11, 0)),
    ("in an hour", D(2026, 8, 26, 10, 0)),
    ("in an hour and a half", D(2026, 8, 26, 10, 30)),
    ("in half an hour", D(2026, 8, 26, 9, 30)),
    ("in one and a half hours", D(2026, 8, 26, 10, 30)),
    ("in 2 hours and 15 minutes", D(2026, 8, 26, 11, 15)),
    ("in twenty minutes", D(2026, 8, 26, 9, 20)),
    ("in a couple of minutes", D(2026, 8, 26, 9, 2)),
    ("in 1.5 hours", D(2026, 8, 26, 10, 30)),
    ("in 3 days", D(2026, 8, 29, 9, 0)),
    ("in 2 days at 9", D(2026, 8, 28, 9, 0)),
    ("at 3", D(2026, 8, 26, 15, 0)),               # next 3 is this afternoon
    ("at 3 pm", D(2026, 8, 26, 15, 0)),
    ("at 3pm", D(2026, 8, 26, 15, 0)),
    ("3 pm", D(2026, 8, 26, 15, 0)),
    ("at 6:30", D(2026, 8, 26, 18, 30)),           # 6:30 am already past
    ("at 6:30 am", D(2026, 8, 27, 6, 30)),
    ("at 10", D(2026, 8, 26, 10, 0)),
    ("at 7", D(2026, 8, 26, 19, 0)),               # next 7 is 7 pm
    ("at 12", D(2026, 8, 26, 12, 0)),              # bare 12 = noon
    ("at 17:30", D(2026, 8, 26, 17, 30)),
    ("at seven thirty pm", D(2026, 8, 26, 19, 30)),
    ("half past 6", D(2026, 8, 26, 18, 30)),
    ("quarter to 8", D(2026, 8, 26, 19, 45)),
    ("at noon", D(2026, 8, 26, 12, 0)),
    ("at midnight", D(2026, 8, 27, 0, 0)),
    ("at 9 in the morning", D(2026, 8, 27, 9, 0)),  # 9:00 is not > now
    ("at 7 in the evening", D(2026, 8, 26, 19, 0)),
    ("tomorrow morning", D(2026, 8, 27, 8, 0)),
    ("tomorrow afternoon", D(2026, 8, 27, 14, 0)),
    ("tomorrow evening", D(2026, 8, 27, 18, 0)),
    ("tomorrow night", D(2026, 8, 27, 21, 0)),
    ("tonight", D(2026, 8, 26, 21, 0)),
    ("this evening", D(2026, 8, 26, 18, 0)),
    ("tonight at 8", D(2026, 8, 26, 20, 0)),
    ("tomorrow at 8", D(2026, 8, 27, 8, 0)),
    ("tomorrow at 3", D(2026, 8, 27, 15, 0)),      # 1-6 on a named day = pm
    ("tomorrow at noon", D(2026, 8, 27, 12, 0)),
    ("tomorrow", D(2026, 8, 27, 9, 0)),
    ("on friday at 9", D(2026, 8, 28, 9, 0)),
    ("friday at 9 pm", D(2026, 8, 28, 21, 0)),
    ("next monday", D(2026, 8, 31, 9, 0)),
    ("next monday at 7", D(2026, 8, 31, 7, 0)),
    ("wednesday at 10", D(2026, 8, 26, 10, 0)),    # today, still ahead
    ("wednesday at 8", D(2026, 9, 2, 8, 0)),       # today's 8 am is past -> next week
    ("every day at 7", D(2026, 8, 26, 19, 0)),
    ("every day at 7 am", D(2026, 8, 27, 7, 0)),
    ("on august 30 at 2 pm", D(2026, 8, 30, 14, 0)),
    ("2026-08-27 07:00", D(2026, 8, 27, 7, 0)),
    ("10 minutes", D(2026, 8, 26, 9, 10)),
    ("day after tomorrow at 5", D(2026, 8, 28, 17, 0)),
])
def test_parse_when_table(text, expected):
    assert parse_when(text, NOW) == expected


@pytest.mark.parametrize("text,expected", [
    ("at 7", D(2026, 8, 27, 7, 0)),                # alarms: bare hour is am
    ("at 6:30", D(2026, 8, 27, 6, 30)),
    ("every weekday at 6:30", D(2026, 8, 27, 6, 30)),
    ("tomorrow at 6", D(2026, 8, 27, 6, 0)),
    ("tomorrow", D(2026, 8, 27, 7, 0)),
    ("at 10", D(2026, 8, 26, 10, 0)),
    ("at 3 pm", D(2026, 8, 26, 15, 0)),
])
def test_parse_when_prefer_morning(text, expected):
    assert parse_when(text, NOW, prefer="morning") == expected


def test_bare_hour_next_occurrence_rule():
    assert parse_when("at 7", D(2026, 8, 26, 6, 0)) == D(2026, 8, 26, 7, 0)
    assert parse_when("at 7", D(2026, 8, 26, 9, 0)) == D(2026, 8, 26, 19, 0)
    assert parse_when("at 7", D(2026, 8, 26, 20, 0)) == D(2026, 8, 27, 7, 0)
    assert parse_when("at noon", D(2026, 8, 26, 13, 0)) == D(2026, 8, 27, 12, 0)
    # "tonight" once 9 pm has gone: the next whole hour, not tomorrow night
    assert parse_when("tonight", D(2026, 8, 26, 22, 30)) == D(2026, 8, 26, 23, 0)


def test_every_weekday_skips_the_weekend():
    friday = D(2026, 8, 28, 9, 0)
    assert parse_when("every weekday at 6:30", friday, "morning") == D(2026, 8, 31, 6, 30)
    assert parse_when("weekdays at 6:30 am", friday) == D(2026, 8, 31, 6, 30)


def test_parse_when_none_and_repeat():
    assert parse_when("", NOW) is None
    assert parse_when("call mum", NOW) is None
    assert parse_when("today at 3", D(2026, 8, 26, 16, 0)) is None   # explicitly today, gone
    assert parse_when_full("every day at 7", NOW)[1] == "daily"
    assert parse_when_full("every weekday at 6:30", NOW)[1] == "weekdays"
    assert parse_when_full("daily at 9 pm", NOW)[1] == "daily"
    assert parse_when_full("at 6 am every day", NOW)[1] == "daily"
    assert parse_when_full("at 6 am", NOW)[1] == ""
    aware = NOW.replace(tzinfo=__import__("datetime").timezone.utc)
    got = parse_when("in 10 minutes", aware)
    assert got.tzinfo is not None and got == aware + timedelta(minutes=10)


@pytest.mark.parametrize("text,dt,rest", [
    ("in 10 minutes to call mum", D(2026, 8, 26, 9, 10), "call mum"),
    ("call Mum at 3", D(2026, 8, 26, 15, 0), "call Mum"),
    ("at 3 pm take the bins out", D(2026, 8, 26, 15, 0), "take the bins out"),
    ("tomorrow morning: dentist appointment", D(2026, 8, 27, 8, 0), "dentist appointment"),
    ("remind me in an hour to stretch", D(2026, 8, 26, 10, 0), "stretch"),
    ("take 2 pills at 3", D(2026, 8, 26, 15, 0), "take 2 pills"),
    ("buy 3 apples tomorrow", D(2026, 8, 27, 9, 0), "buy 3 apples"),
])
def test_split_when_keeps_the_text(text, dt, rest):
    assert split_when(text, NOW) == (dt, rest)


@pytest.mark.parametrize("value,seconds", [
    ("15", 900), (15, 900), (1.5, 90), ("90 seconds", 90), ("1.5 hours", 5400),
    ("an hour and a half", 5400), ("in 20 min", 1200), ("abc", None), (None, None),
    (0, None), ("0", None), ("3 pm", None),
])
def test_parse_duration(value, seconds):
    assert parse_duration(value) == seconds


# ---------------------------------------------------------- describe_due
@pytest.mark.parametrize("delta,expected", [
    # A 30 s reminder used to be confirmed as "set now, sir" (fixed 2026-08-26:
    # only |delta| < NOW_WINDOW_S is "now").
    (2, "now"), (30, "in 30 seconds"), (60, "in a minute"), (600, "in 10 minutes"), (3600, "in an hour"),
    (5400, "in an hour and a half"), (7200, "at 11:00 am"), (6 * 3600, "at 3:00 pm"),
    (3 * 3600, "at noon"), (23 * 3600, "at 8:00 am tomorrow"),
    (3 * 86400, "at 9:00 am on Saturday"), (10 * 86400, "at 9:00 am on the 5th of September"),
    (-600, "10 minutes ago"), (-86400, "at 9:00 am yesterday"),
])
def test_describe_due(delta, expected):
    assert describe_due(NOW.timestamp() + delta, NOW.timestamp()) == expected
    assert describe_due(NOW + timedelta(seconds=delta), NOW) == expected


def test_duration_words():
    assert tkm.duration_words(300) == "5-minute"
    assert tkm.duration_words(90) == "1-and-a-half-minute"
    assert tkm.duration_words(45) == "45-second"
    assert tkm.duration_words(7200) == "2-hour"
    assert tkm.count_words(2) == "two" and tkm.count_words(40) == "40"
    assert tkm.join_and(["a", "b", "c"]) == "a, b, and c"


# ---------------------------------------------------------- scheduling
def test_add_and_list_sorted(tk):
    t0 = tk.clock.now()
    r = tk.add_reminder(t0 + 600, "call the dentist")
    a = tk.add_alarm(t0 + 3600, "Gym", "weekdays")
    m = tk.add_timer(120, "pasta")
    assert [i.id for i in tk.list()] == [m.id, r.id, a.id]
    assert [i.kind for i in tk.list("alarms")] == ["alarm"]
    assert tk.list("reminder")[0].label == "call the dentist"
    assert a.repeat == "weekdays" and r.repeat == "" and m.duration == 120
    with pytest.raises(ValueError):
        tk.add_timer(0)


def test_reminder_fires_on_tick(tk):
    t0 = tk.clock.now()
    tk.add_reminder(t0 + 600, "call the dentist")
    tk.clock.advance(599)
    assert tk.tick() == 0 and tk.said == []
    tk.clock.advance(1)
    assert tk.tick() == 1
    assert tk.said == ["Sir, this is your reminder. Call the dentist."]
    fired = kinds(tk.events, events.ReminderFired)
    assert fired and fired[0].text == "call the dentist"
    toast = tk.run_rec.of("notify-send")
    assert toast and toast[0][-1] == "call the dentist" and "-a" in toast[0]
    assert tk.list() == [] and tk.list(include_done=True)[0].state == "done"
    assert tk.tick() == 0                         # fires once


@pytest.mark.parametrize("raw,expected", [
    ("call the dentist", "Call the dentist."),
    ("Water the plants.", "Water the plants."),
    ("ring Pepper", "Ring Pepper."),
    ("wake up!", "Wake up!"),
    ("", ""),
    (None, ""),
])
def test_sentence_case(raw, expected):
    assert sentence_case(raw) == expected


def test_spoken_user_text_is_a_sentence(tk):
    """Reminder text and alarm labels start a spoken sentence, so they are
    capitalised and full-stopped (they arrive lower-case from the model)."""
    tk.add_reminder(tk.clock.now() + 60, "feed the cat")
    tk.add_alarm(tk.clock.now() + 120, "get up and run")
    tk.clock.advance(60)
    tk.tick()
    tk.clock.advance(60)
    tk.tick()
    assert tk.said[0] == "Sir, this is your reminder. Feed the cat."
    assert tk.said[1] == "Sir, it's 9:02 am. Get up and run."


def test_timer_wording(tk):
    tk.add_timer(300)
    tk.add_timer(90, "pasta")
    tk.clock.advance(300)
    assert tk.tick() == 2
    assert "Sir, your 5-minute timer is up." in tk.said
    assert "Sir, your 1-and-a-half-minute pasta timer is up." in tk.said
    texts = {e.text for e in kinds(tk.events, events.ReminderFired)}
    assert texts == {"5-minute timer", "pasta"}


def test_persists_across_reopen(tmp_path, capture):
    clock = FakeClock(NOW)
    db = tmp_path / "tk.db"
    first = Timekeeper(db, say=lambda s: None, cfg={}, now=clock.now, run=Recorder(),
                       cache_dir=tmp_path / "c")
    first.add_reminder(clock.now() + 120, "water the plants")
    first.add_alarm(clock.now() + 3600, "Gym", "daily")
    first.close()
    said = []
    second = Timekeeper(db, say=said.append, cfg={}, now=clock.now, run=Recorder(),
                        cache_dir=tmp_path / "c")
    try:
        labels = [i.label for i in second.list()]
        assert labels == ["water the plants", "Gym"]
        clock.advance(120)
        assert second.tick() == 1
        assert said == ["Sir, this is your reminder. Water the plants."]
    finally:
        second.close()


def test_alarm_rings_escalates_and_times_out(tk):
    tk.clock.set(D(2026, 8, 26, 6, 59, 0))
    t0 = tk.clock.now()
    alarm = tk.add_alarm(t0 + 60, "")
    tk.clock.advance(60)                          # 7:00:00
    assert tk.tick() == 1
    assert tk.said == [f"Sir, it's 7:00 am. {ALARM_DEFAULT_LABEL}"]
    fired = kinds(tk.events, events.AlarmFired)
    assert fired and fired[0].alarm_id == alarm.id and fired[0].due_text == "7:00 am" \
        and fired[0].kind == "alarm"
    assert tk.ringing is not None and tk.ringing.id == alarm.id
    plays = tk.run_rec.of("paplay")
    assert plays == [["paplay", "--volume=32768", tkm.DEFAULT_SOUND]]   # 0.5 * 65536
    crit = [c for c in tk.run_rec.of("notify-send") if "critical" in c]
    assert crit
    # 1 s gaps at the configured volume until 30 s, then full volume
    for _ in range(29):
        tk.clock.advance(1)
        tk.tick()
    assert all(c[1] == "--volume=32768" for c in tk.run_rec.of("paplay"))
    assert len(tk.run_rec.of("paplay")) == 30
    tk.clock.advance(1)
    tk.tick()
    assert tk.run_rec.of("paplay")[-1][1] == "--volume=65536"
    # rings out at max_ring_s -> missed + the missed line + AlarmStopped(timeout)
    tk.clock.set(D(2026, 8, 26, 7, 5, 0))
    tk.tick()
    assert tk.ringing is None
    assert tk.list(include_done=True)[0].state == "missed"
    assert tk.said[-1] == "You missed your alarm at 7:00 am, sir; I've stopped ringing."
    stopped = kinds(tk.events, events.AlarmStopped)
    assert stopped[-1].action == "timeout" and stopped[-1].alarm_id == alarm.id
    n = len(tk.run_rec.of("paplay"))
    tk.clock.advance(5)
    tk.tick()
    assert len(tk.run_rec.of("paplay")) == n            # silence after timeout


def test_no_escalation_when_disabled(tmp_path, capture):
    clock = FakeClock(D(2026, 8, 26, 7, 0))
    run = Recorder()
    t = Timekeeper(tmp_path / "tk.db", say=lambda s: None, now=clock.now, run=run,
                   cfg={"alarms": {"volume": 1.0, "escalate": False}}, cache_dir=tmp_path)
    try:
        t.add_alarm(clock.now(), "up")
        for _ in range(40):
            t.tick()
            clock.advance(1)
        assert {c[1] for c in run.of("paplay")} == {"--volume=65536"}
        assert len(run.of("paplay")) == 40
    finally:
        t.close()


def test_dismiss_stops_ringing(tk):
    tk.clock.set(D(2026, 8, 26, 7, 0))
    alarm = tk.add_alarm(tk.clock.now(), "Gym")
    tk.tick()
    assert tk.said == ["Sir, it's 7:00 am. Gym."]
    assert tk.stop_ringing("dismiss") is True
    assert tk.ringing is None
    assert tk.list(include_done=True)[0].state == "done"
    stopped = kinds(tk.events, events.AlarmStopped)
    assert stopped[-1].action == "dismiss" and stopped[-1].alarm_id == alarm.id
    n = len(tk.run_rec.of("paplay"))
    tk.clock.advance(3)
    tk.tick()
    assert len(tk.run_rec.of("paplay")) == n
    assert tk.stop_ringing() is False             # nothing ringing now


def test_snooze_rings_again(tk):
    tk.clock.set(D(2026, 8, 26, 7, 0))
    alarm = tk.add_alarm(tk.clock.now(), "")
    tk.tick()
    assert tk.snooze(5) is True
    item = tk.list()[0]
    assert item.state == "snoozed" and item.snooze_until == tk.clock.now() + 300
    stopped = kinds(tk.events, events.AlarmStopped)
    assert stopped[-1].action == "snooze" and stopped[-1].snooze_min == 5
    assert "snoozed until 7:05 am" in tk.list_text("alarm")
    tk.clock.advance(299)
    tk.tick()
    assert tk.ringing is None
    tk.clock.advance(1)
    tk.tick()
    assert tk.ringing is not None and tk.ringing.id == alarm.id
    assert tk.said[-1] == "Sir, it's 7:05 am. Time to get up."
    assert len(kinds(tk.events, events.AlarmFired)) == 2
    # default snooze length comes from cfg (10 min)
    assert tk.snooze() is True
    assert tk.list()[0].snooze_until == tk.clock.now() + 600


def test_repeat_alarms_reschedule_on_stop(tk):
    tk.clock.set(D(2026, 8, 28, 7, 0))            # Friday
    daily = tk.add_alarm(tk.clock.now(), "daily one", "daily")
    weekdays = tk.add_alarm(tk.clock.now() + 1, "weekday one", "weekdays")
    tk.tick()
    assert tk.ringing.id == daily.id
    tk.stop_ringing()
    d = tk.list(include_done=True)
    assert {i.id: i.state for i in d}[daily.id] == "pending"
    assert datetime.fromtimestamp([i for i in d if i.id == daily.id][0].due) == D(2026, 8, 29, 7, 0)
    tk.clock.advance(1)
    tk.tick()                                     # second alarm now rings (one at a time)
    assert tk.ringing.id == weekdays.id
    tk.stop_ringing()
    wd = [i for i in tk.list() if i.id == weekdays.id][0]
    assert datetime.fromtimestamp(wd.due) == D(2026, 8, 31, 7, 0, 1)   # Monday
    assert tk.list_text("alarm").endswith(", weekdays.")


def test_catch_up_59_minutes_late_fires_with_preamble(tk):
    t0 = tk.clock.now()
    tk.add_reminder(t0 - 28 * 60, "call the dentist")   # 8:32 am
    tk.add_timer(60, "pasta")                           # 9:01 am
    tk.clock.advance(31 * 60)                           # 9:31: 59 min / 30 min late
    handled = tk.catch_up()
    assert {i.kind for i in handled} == {"reminder", "timer"}
    assert "While I was down, sir: call the dentist, due at 8:32 am." in tk.said
    assert "While I was down, sir: pasta, due at 9:01 am." in tk.said
    assert {e.text for e in kinds(tk.events, events.ReminderFired)} == {"call the dentist", "pasta"}
    assert all(i.state == "done" for i in tk.list(include_done=True))


def test_catch_up_61_minutes_late_is_missed_once(tk):
    t0 = tk.clock.now()
    tk.add_reminder(t0 - 61 * 60, "call the dentist")
    tk.add_alarm(t0 - 2 * 3600, "Gym")
    daily = tk.add_alarm(t0 - 3 * 3600, "wake", "daily")
    handled = tk.catch_up()
    assert len(handled) == 3
    assert tk.said == ["You missed wake at 6:00 am, Gym at 7:00 am, and call the dentist "
                       "at 7:59 am while I was down, sir."]
    assert kinds(tk.events, events.ReminderFired) == []
    states = {i.label: i.state for i in tk.list(include_done=True)}
    assert states == {"call the dentist": "missed", "Gym": "missed", "wake": "pending"}
    nxt = [i for i in tk.list() if i.id == daily.id][0]
    assert datetime.fromtimestamp(nxt.due) == D(2026, 8, 27, 6, 0)   # daily one carries on
    assert tk.catch_up() == [] and len(tk.said) == 1                 # announced once


def test_catch_up_alarm_late_rings(tk):
    tk.clock.set(D(2026, 8, 26, 7, 30))
    tk.add_alarm(tk.clock.now() - 1800, "Gym")
    tk.catch_up()
    assert tk.said == ["While I was down, sir: Gym, due at 7:00 am."]
    assert tk.ringing is not None and tk.run_rec.of("paplay")


def test_stale_ringing_row_is_recovered_on_boot(tk):
    alarm = tk.add_alarm(tk.clock.now() - 600, "Gym")
    tk._update(alarm.id, state="ringing")         # app died mid-ring
    tk.catch_up()
    assert tk.ringing is not None and tk.ringing.id == alarm.id


def test_start_runs_catch_up_and_scheduler_thread(tmp_path, capture):
    clock = FakeClock(NOW)
    said = []
    t = Timekeeper(tmp_path / "tk.db", say=said.append, cfg={}, now=clock.now,
                   run=Recorder(), tick_s=0.01, cache_dir=tmp_path)
    try:
        t.add_reminder(clock.now() - 60, "late one")
        t.add_reminder(clock.now() + 5, "soon")
        t.start()
        assert said == ["While I was down, sir: late one, due at 8:59 am."]
        clock.advance(5)
        deadline = time.time() + 0.2
        while len(said) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert said[-1] == "Sir, this is your reminder. Soon."
        t.start()                                 # idempotent
    finally:
        t.stop()
        t.close()


def test_import_legacy(tk, tmp_path):
    legacy = tmp_path / "reminders.json"
    due = tk.clock.now() + 900
    legacy.write_text(json.dumps({"reminders": [
        {"id": "abc123", "task": "feed the cat", "due": due},
        {"id": "def456", "task": "old one", "due": tk.clock.now() - 100}]}))
    assert tk.import_legacy(legacy) == 2
    assert not legacy.exists() and (tmp_path / "reminders.json.migrated").exists()
    labels = [i.label for i in tk.list("reminder")]
    assert labels == ["old one", "feed the cat"]
    assert tk.import_legacy(legacy) == 0          # gone
    (tmp_path / "reminders.json.migrated").rename(legacy)
    assert tk.import_legacy(legacy) == 0          # ids already imported
    assert tk.import_legacy(tmp_path / "nope.json") == 0
    tk.catch_up()                                 # the overdue one fires late
    assert tk.said == ["While I was down, sir: old one, due at 8:58 am."]


def test_list_text_wording(tk):
    assert tk.list_text() == "Nothing scheduled, sir."
    assert tk.list_text("reminder") == "No reminders set, sir."
    assert tk.list_text("timers") == "No timers running, sir."
    assert tk.list_text("alarm") == "No alarms set, sir."
    t0 = tk.clock.now()
    tk.add_reminder(t0 + 6 * 3600, "call the dentist")
    tk.add_reminder(t0 + 34 * 3600, "take the bins out")
    assert tk.list_text("reminder") == ("Two reminders, sir: call the dentist at 3:00 pm "
                                        "and take the bins out at 7:00 pm tomorrow.")
    tk.add_timer(480, "")
    tk.add_alarm(t0 + 22 * 3600, "", "daily")
    assert tk.list_text() == (
        "Two reminders, sir: call the dentist at 3:00 pm and take the bins out at "
        "7:00 pm tomorrow. One timer: the 8-minute timer in 8 minutes. "
        "One alarm: an alarm at 7:00 am tomorrow, every day.")
    tk.add_reminder(t0 + 60, "stretch")
    assert tk.list_text("reminder").startswith("Three reminders, sir: stretch in a minute, ")
    assert tk.list_text("reminder", now=NOW) == tk.list_text("reminder")


def test_cancel_variants(tk):
    t0 = tk.clock.now()
    a = tk.add_reminder(t0 + 100, "call the dentist")
    b = tk.add_reminder(t0 + 200, "take the bins out")
    c = tk.add_timer(300, "pasta")
    d = tk.add_alarm(t0 + 400, "Gym")
    assert tk.cancel("last") == 1 and d.id not in {i.id for i in tk.list()}
    assert tk.cancel("dentist") == 1 and a.id not in {i.id for i in tk.list()}
    assert tk.cancel("all", kind="timer") == 1 and c.id not in {i.id for i in tk.list()}
    assert tk.cancel(b.id) == 1 and tk.list() == []
    assert tk.cancel("all") == 0
    e = tk.add_reminder(t0 + 100, "water the plants")
    assert tk.cancel("the plants reminder") == 1
    assert tk.list(include_done=True)[-1].state == "cancelled" and e.id
    tk.add_reminder(t0 + 10, "x")
    tk.add_reminder(t0 + 20, "y")
    assert tk.cancel("everything") == 2 and tk.list() == []


def test_cancel_ringing_alarm_stops_the_ringer(tk):
    tk.add_alarm(tk.clock.now(), "Gym")
    tk.tick()
    assert tk.ringing is not None
    assert tk.cancel("gym") == 1
    assert tk.ringing is None
    assert kinds(tk.events, events.AlarmStopped)[-1].action == "dismiss"
    n = len(tk.run_rec.of("paplay"))
    tk.clock.advance(2)
    tk.tick()
    assert len(tk.run_rec.of("paplay")) == n


def test_ring_disabled_still_publishes(tmp_path, capture):
    clock = FakeClock(D(2026, 8, 26, 7, 0))
    run = Recorder()
    t = Timekeeper(tmp_path / "tk.db", say=lambda s: None, cfg=None, now=clock.now, run=run,
                   ring=False, notify=False, cache_dir=tmp_path)
    try:
        t.add_alarm(clock.now(), "Gym")
        t.tick()
        assert t.ringing is not None and kinds(capture, events.AlarmFired)
        assert run.calls == []                    # no paplay, no notify-send
        assert t.stop_ringing() is True and t.ringing is None
    finally:
        t.close()


def test_fallback_wav_is_generated_once(tmp_path, capture, monkeypatch):
    monkeypatch.setattr(tkm, "DEFAULT_SOUND", str(tmp_path / "missing.oga"))
    clock = FakeClock(NOW)
    run = Recorder()
    cache = tmp_path / "cache"
    t = Timekeeper(tmp_path / "tk.db", say=lambda s: None, now=clock.now, run=run,
                   cfg={"alarms": {"sound": str(tmp_path / "also-missing.wav"), "volume": 0.8}},
                   cache_dir=cache)
    try:
        t.add_alarm(clock.now(), "")
        t.tick()
        wav = cache / "alarm.wav"
        assert wav.is_file()
        assert run.of("paplay")[0] == ["paplay", "--volume=52428", str(wav)]
        with wave.open(str(wav)) as wf:
            assert wf.getnchannels() == 1 and wf.getframerate() == 22050
            assert wf.getnframes() / wf.getframerate() >= 1.5
        mtime = wav.stat().st_mtime
        clock.advance(2)
        t.tick()
        assert wav.stat().st_mtime == mtime       # written once
    finally:
        t.close()
    out = write_alarm_wav(tmp_path / "x" / "beep.wav", seconds=0.5)
    assert out.is_file()


def test_configured_sound_and_volume(tmp_path, capture):
    sound = tmp_path / "mine.wav"
    write_alarm_wav(sound, seconds=0.3)
    clock = FakeClock(NOW)
    run = Recorder()
    cfg = SimpleNamespace(alarms=SimpleNamespace(sound=str(sound), volume=1.5, escalate=True,
                                                 max_ring_s=300, snooze_min=10))
    t = Timekeeper(tmp_path / "tk.db", say=lambda s: None, now=clock.now, run=run, cfg=cfg,
                   cache_dir=tmp_path)
    try:
        t.add_alarm(clock.now(), "")
        t.tick()
        assert run.of("paplay")[0] == ["paplay", "--volume=65536", str(sound)]   # clamped
    finally:
        t.close()


def test_cfg_get_accessors():
    assert tkm._cfg_get(None, "alarms.volume", 0.8) == 0.8
    assert tkm._cfg_get({}, "alarms.volume", 0.8) == 0.8
    assert tkm._cfg_get({"alarms": {"volume": 0.3}}, "alarms.volume") == 0.3
    assert tkm._cfg_get(SimpleNamespace(alarms=SimpleNamespace(volume=0.4)), "alarms.volume") == 0.4
    assert tkm._cfg_get(FakeAssistantConfig(volume=0.6), "alarms.volume") == 0.6
    assert tkm._cfg_get(FakeAssistantConfig(), "alarms.nope", 7) == 7


# ---------------------------------------------------------------- tools
@pytest.fixture
def tools(tk):
    services = SimpleNamespace(timekeeper=tk)
    specs = make_tools(tk.cfg, services)
    reg = ToolRegistry()
    reg.register_many(specs)
    return reg


def test_tool_specs(tools):
    assert tools.names() == ["set_reminder", "set_timer", "set_alarm", "manage_schedule"]
    for s in tools.schemas():
        fn = s["function"]
        assert len(fn["description"].split()) <= 20
        assert fn["parameters"]["type"] == "object"
    props = tools.schemas()[3]["function"]["parameters"]["properties"]
    assert props["action"]["enum"] == ["list", "cancel", "extend", "shorten",
                                       "stop", "snooze"]
    assert props["kind"]["enum"] == ["reminder", "timer", "alarm", "all"]
    # `repeat` lost its enum when interval nudges arrived: no closed list can
    # hold "every 45 minutes". The description has to carry the shape instead.
    for i in (0, 2):
        rep = tools.schemas()[i]["function"]["parameters"]["properties"]["repeat"]
        assert "enum" not in rep
        for word in ("once", "daily", "weekdays", "every"):
            assert word in rep["description"]


def test_tool_set_reminder(tools, tk):
    r = tools.call("set_reminder", {"when": "at 3 pm", "text": "call the dentist", "extra": 1})
    assert r.ok and r.text == "Reminder set for 3:00 pm, sir." and r.speak == r.text
    assert tk.list("reminder")[0].label == "call the dentist"
    r = tools.call("set_reminder", {"when": "in 10 minutes", "text": "stretch"})
    assert r.text == "Reminder set for 10 minutes from now, sir."
    r = tools.call("set_reminder", {"when": "tomorrow morning", "text": "dentist"})
    assert r.text == "Reminder set for 8:00 am tomorrow, sir."
    # the model put the time inside the text
    r = tools.call("set_reminder", {"when": "", "text": "take the bins out at 7 pm"})
    assert r.ok and r.text == "Reminder set for 7:00 pm, sir."
    assert "take the bins out" in {i.label for i in tk.list("reminder")}
    # the model put everything in `when`
    r = tools.call("set_reminder", {"when": "in an hour to water the plants"})
    assert r.ok and "water the plants" in {i.label for i in tk.list("reminder")}
    r = tools.call("set_reminder", {"when": "whenever", "text": "x"})
    assert not r.ok and r.speak == CANT_PARSE_LINE


def test_tool_set_timer(tools, tk):
    r = tools.call("set_timer", {"minutes": "15"})
    assert r.ok and r.text == "15-minute timer set, sir."
    assert tk.list("timer")[0].duration == 900
    r = tools.call("set_timer", {"minutes": 1.5, "label": "eggs"})
    assert r.text == "1-and-a-half-minute timer set for eggs, sir."
    r = tools.call("set_timer", {"minutes": "90 seconds"})
    assert r.text == "1-and-a-half-minute timer set, sir."
    r = tools.call("set_timer", {"minutes": "soon"})
    assert not r.ok and r.speak == HOW_LONG_LINE
    r = tools.call("set_timer", {})
    assert not r.ok and r.speak == HOW_LONG_LINE


def test_tool_set_alarm(tools, tk):
    r = tools.call("set_alarm", {"when": "at 7"})
    assert r.ok and r.text == "Alarm set for 7:00 am tomorrow, sir."
    r = tools.call("set_alarm", {"when": "every weekday at 6:30", "label": "Gym"})
    assert r.text == "Alarm set for 6:30 am tomorrow, weekdays, sir."
    assert tk.list("alarm")[0].repeat == "weekdays"
    r = tools.call("set_alarm", {"when": "at 8", "repeat": "daily"})
    assert r.text == "Alarm set for 8:00 am tomorrow, every day, sir."
    r = tools.call("set_alarm", {"when": "in 20 minutes", "repeat": "once"})
    assert r.text == "Alarm set for 20 minutes from now, sir."
    assert tk.list("alarm")[0].repeat == ""
    r = tools.call("set_alarm", {"when": ""})
    assert not r.ok and r.speak == CANT_PARSE_LINE


def test_tool_manage_schedule(tools, tk):
    r = tools.call("manage_schedule", {"action": "list"})
    assert r.text == "Nothing scheduled, sir." and r.speak == r.text and r.max_sentences == 3
    tools.call("set_reminder", {"when": "at 3 pm", "text": "call the dentist"})
    tools.call("set_timer", {"minutes": 5})
    r = tools.call("manage_schedule", {"action": "list", "kind": "reminders"})
    assert r.text == "One reminder, sir: call the dentist at 3:00 pm."
    r = tools.call("manage_schedule", {"action": "cancel", "kind": "timer"})
    assert r.ok and r.text == "Cancelled, sir." and tk.list("timer") == []
    r = tools.call("manage_schedule", {"action": "delete", "which": "dentist"})
    assert r.text == "Cancelled, sir."
    r = tools.call("manage_schedule", {"action": "cancel", "which": "all"})
    assert not r.ok and r.text == "Nothing to cancel, sir."
    tools.call("set_reminder", {"when": "at 3 pm", "text": "a"})
    tools.call("set_reminder", {"when": "at 4 pm", "text": "b"})
    r = tools.call("manage_schedule", {"action": "cancel", "which": "all", "kind": "reminder"})
    assert r.text == "Cancelled two reminders, sir."
    r = tools.call("manage_schedule", {"action": "stop"})
    assert not r.ok and r.speak == NOTHING_RINGING_LINE
    r = tools.call("manage_schedule", {"action": "snooze"})
    assert not r.ok and r.speak == NOTHING_RINGING_LINE
    tk.add_alarm(tk.clock.now(), "Gym")
    tk.tick()
    r = tools.call("manage_schedule", {"action": "snooze", "minutes": "5"})
    assert r.ok and r.text == "Snoozed for five minutes, sir."
    tk.clock.advance(300)
    tk.tick()
    assert tk.ringing is not None
    r = tools.call("manage_schedule", {"action": "dismiss"})
    assert r.ok and r.text == "Very good, sir." and tk.ringing is None
    r = tools.call("manage_schedule", {"action": "snooze"})
    assert not r.ok
    r = tools.call("manage_schedule", {"action": "fly"})
    assert not r.ok


def test_tools_without_timekeeper():
    reg = ToolRegistry()
    reg.register_many(make_tools({}, SimpleNamespace()))
    for name, args in [("set_reminder", {"when": "at 3", "text": "x"}), ("set_timer", {"minutes": 1}),
                       ("set_alarm", {"when": "at 7"}), ("manage_schedule", {"action": "list"})]:
        r = reg.call(name, args)
        assert not r.ok and r.speak == NO_TIMEKEEPER_LINE


def test_missed_alarms_reach_the_quiet_digest_as_alarms(tmp_path):
    """kind='alarm' on the rang-out and catch-up missed lines: held during
    quiet hours they must digest as 'one alarm', not 'one reminder'."""
    clock = FakeClock(NOW)
    rec = []

    def say(line, proactive=True, kind="reminder"):
        rec.append((line, proactive, kind))
    t = Timekeeper(tmp_path / "tk.db", say=say, cfg={}, now=clock.now,
                   run=lambda *a, **k: None, ring=False, notify=False,
                   cache_dir=tmp_path / "cache")
    try:
        t.add_alarm(clock.now() - 61 * 60, "wake")
        t.catch_up()
        missed = [k for line, _p, k in rec if "missed" in line.lower()]
        assert missed == ["alarm"], rec
    finally:
        t.close()


# ------------------------------------------------- interval nudges (n=24)
@pytest.mark.parametrize("text,repeat,due,label", [
    ("drink water every 45 minutes", "45m", D(2026, 8, 26, 9, 45), "drink water"),
    ("stand up every hour", "60m", D(2026, 8, 26, 10, 0), "stand up"),
    ("every two hours check the oven", "120m", D(2026, 8, 26, 11, 0), "check the oven"),
    ("take a break every 90 minutes", "90m", D(2026, 8, 26, 10, 30), "take a break"),
    ("every half an hour look away", "30m", D(2026, 8, 26, 9, 30), "look away"),
    ("stretch every minute", "1m", D(2026, 8, 26, 9, 1), "stretch"),
    # unchanged: the fixed repeats still win, and a plain duration is a one-off
    ("call mum every day at 6", "daily", D(2026, 8, 26, 18, 0), "call mum"),
    ("stretch in 10 minutes", "", D(2026, 8, 26, 9, 10), "stretch"),
])
def test_every_n_minutes_parses_as_an_interval(text, repeat, due, label):
    dt, rep, rest = parse_when_full(text, NOW)
    assert (dt, rep, rest) == (due, repeat, label)


@pytest.mark.parametrize("text", [
    "water every 5 seconds",          # a mis-hear, not a nudge
    "ping me every week",             # longer than the 24 h ceiling
    "every 20 the thing",             # a bare number is a half-heard sentence
])
def test_an_out_of_range_every_is_left_alone(text):
    """Rejected intervals must not CONSUME their words either: swallowing
    them would leave a reminder with a label and no time at all."""
    dt, rep, rest = parse_when_full(text, NOW)
    assert rep == "" and dt is None and rest == text


def test_an_interval_is_re_filed_from_now_not_from_the_due_time():
    """Walking forward from `due` after a two-hour outage would fire the
    nudge three times in ten seconds."""
    due = NOW.timestamp()
    late = due + 3 * 3600
    assert tkm.next_repeat(due, "45m", late) == late + 45 * 60
    # daily still walks from the due hour, DST and all
    assert tkm.next_repeat(due, "daily", late) == (NOW + timedelta(days=1)).timestamp()


@pytest.mark.parametrize("repeat,words", [
    ("45m", ", every 45 minutes"), ("60m", ", every hour"),
    ("120m", ", every two hours"), ("1m", ", every minute"),
    ("daily", ", every day"), ("weekdays", ", weekdays"), ("", ""),
])
def test_repeat_suffix_reads_back(repeat, words):
    assert tkm.repeat_suffix(repeat) == words


def test_interval_token_round_trips_and_bounds():
    assert tkm.interval_token(45 * 60) == "45m"
    assert tkm.interval_token(30) == ""                    # under a minute
    assert tkm.interval_token(25 * 3600) == ""             # over a day
    assert tkm.interval_minutes("45m") == 45
    assert tkm.interval_minutes("1441m") is None
    assert tkm.interval_minutes("daily") is None
    assert tkm.is_repeating("45m") and tkm.is_repeating("daily")
    assert not tkm.is_repeating("") and not tkm.is_repeating("5000m")
    assert tkm.normalize_repeat("45m") == "45m"
    assert tkm.normalize_repeat("every 2 hours") == "120m"


def test_a_nudge_fires_again_and_again_on_the_interval(tk):
    it = tk.add_reminder(tk.clock.now() + 45 * 60, "drink water", repeat="45m")
    for _ in range(3):
        tk.clock.advance(45 * 60 + 1)
        assert tk.tick() == 1
    assert len(tk.said) == 3 and all("drink water" in s.lower() for s in tk.said)
    again = tk.get(it.id)
    assert again.state == "pending"
    assert again.due == pytest.approx(tk.clock.now() + 45 * 60, abs=1)


def test_a_nudge_missed_during_downtime_is_re_filed_silently(tmp_path):
    """catch_up() reads a missed reminder back ("You missed … at 3:15").
    For a nudge that is precisely the stale backlog it exists to avoid."""
    clock = FakeClock(NOW)
    rec = []

    def say(line, proactive=True, kind="reminder"):
        rec.append((line, kind))
    t = Timekeeper(tmp_path / "tk.db", say=say, cfg={}, now=clock.now,
                   run=lambda *a, **k: None, ring=False, notify=False,
                   cache_dir=tmp_path / "cache")
    try:
        it = t.add_reminder(clock.now() + 60, "drink water", repeat="45m")
        clock.advance(3 * 3600)               # three hours down: four nudges owed
        t.catch_up()
        assert rec == []
        again = t.get(it.id)
        assert again.state == "pending"
        assert again.due == pytest.approx(clock.now() + 45 * 60, abs=1)
    finally:
        t.close()


def test_a_nudge_is_listed_with_its_interval(tk):
    tk.add_reminder(tk.clock.now() + 45 * 60, "drink water", repeat="45m")
    assert ", every 45 minutes" in tk.list_text("reminder")


def test_the_tool_takes_an_interval_repeat(tools, tk):
    r = tools.call("set_reminder", {"when": "every 45 minutes", "text": "drink water"})
    assert r.ok and ", every 45 minutes" in r.text
    item = tk.list("reminder")[0]
    assert item.repeat == "45m" and item.label == "drink water"
    r = tools.call("set_reminder", {"when": "in 20 minutes", "text": "stand up",
                                    "repeat": "every 30 minutes"})
    assert r.ok and ", every 30 minutes" in r.text
    by_label = {i.label: i.repeat for i in tk.list("reminder")}
    assert by_label == {"drink water": "45m", "stand up": "30m"}


# ============================================ adjust: extend / shorten (2026-09-01)
# 20:41:49 "Extend that timer by 10 minutes." -> the model picked `list` and
# read the timer back, because the timekeeper had no way to move a due time.
def _ids(items):
    return [i.id for i in items]


def test_adjust_pending_timer_moves_the_due_and_nothing_else(tk):
    t0 = tk.clock.now()
    timer = tk.add_timer(600, "pack up the chicken")
    got = tk.adjust("last", "timer", 600)
    assert _ids(got) == [timer.id]
    fresh = got[0]
    assert fresh.due == pytest.approx(t0 + 1200)
    assert fresh.created == timer.created and fresh.state == "pending"
    assert fresh.snooze_until is None and fresh.repeat == ""
    assert tk.get(timer.id).due == pytest.approx(t0 + 1200)     # persisted
    assert "pack up the chicken in 20 minutes" in tk.list_text("timer")


def test_adjust_snoozed_alarm_moves_snooze_until_only(tk):
    tk.clock.set(D(2026, 8, 26, 7, 0))
    alarm = tk.add_alarm(tk.clock.now(), "Gym")
    tk.tick()
    assert tk.snooze(5)
    before = tk.get(alarm.id)
    got = tk.adjust("last", "alarm", 300)
    assert _ids(got) == [alarm.id]
    assert got[0].state == "snoozed"
    assert got[0].snooze_until == pytest.approx(before.snooze_until + 300)
    assert got[0].due == before.due                        # the original due is kept
    assert "snoozed until 7:10 am" in tk.list_text("alarm")


def test_adjust_shortening_past_now_clamps_to_now_and_fires_next_tick(tk):
    t0 = tk.clock.now()
    tk.add_timer(120, "tea")
    got = tk.adjust("last", "timer", -600)
    assert len(got) == 1 and got[0].due == pytest.approx(t0)  # never in the past
    assert not kinds(tk.events, events.ReminderFired)
    tk.tick()
    fired = kinds(tk.events, events.ReminderFired)
    assert len(fired) == 1 and fired[0].kind == "timer"
    assert tk.said[-1] == "Sir, your tea timer is up."
    assert tk.list("timer") == []


def test_adjust_repeating_reminder_shifts_this_occurrence_only(tk):
    t0 = tk.clock.now()
    rem = tk.add_reminder(t0 + 600, "drink water", repeat="every 45 minutes")
    assert tkm.interval_minutes(rem.repeat) == 45
    got = tk.adjust("water", "reminder", 300)
    assert _ids(got) == [rem.id]
    assert got[0].due == pytest.approx(t0 + 900)
    assert got[0].repeat == rem.repeat                    # the interval is untouched
    tk.clock.advance(900)
    tk.tick()
    again = tk.get(rem.id)
    assert again.state == "pending" and again.due == pytest.approx(tk.clock.now() + 45 * 60)


def test_adjust_by_label_words_hits_the_right_timer(tk):
    t0 = tk.clock.now()
    pasta = tk.add_timer(300, "pasta")
    chicken = tk.add_timer(600, "pack up the chicken")
    got = tk.adjust("the chicken timer", "timer", 60)
    assert _ids(got) == [chicken.id]
    assert tk.get(chicken.id).due == pytest.approx(t0 + 660)
    assert tk.get(pasta.id).due == pytest.approx(t0 + 300)


def test_adjust_last_is_the_most_recently_created(tk):
    t0 = tk.clock.now()
    late = tk.add_timer(3600, "slow roast")                  # created first, due last
    tk.clock.advance(1)
    quick = tk.add_timer(60, "eggs")                         # created last, due first
    got = tk.adjust("last", "all", 120)
    assert _ids(got) == [quick.id]
    assert tk.get(quick.id).due == pytest.approx(t0 + 1 + 180)
    assert tk.get(late.id).due == pytest.approx(t0 + 3600)
    # "that" / "it" are the same item
    assert _ids(tk.adjust("that", "timer", 60)) == [quick.id]
    assert _ids(tk.adjust("it", "timer", 60)) == [quick.id]


def test_adjust_ringing_alarm_with_a_positive_delta_snoozes_it(tk):
    tk.clock.set(D(2026, 8, 26, 7, 0))
    alarm = tk.add_alarm(tk.clock.now(), "Gym")
    tk.tick()
    assert tk.ringing is not None
    n_plays = len(tk.run_rec.of("paplay"))
    got = tk.adjust("last", "alarm", 600)
    assert _ids(got) == [alarm.id]
    assert got[0].state == "snoozed"
    assert got[0].snooze_until == pytest.approx(tk.clock.now() + 600)
    assert tk.ringing is None                                # ring killed
    stopped = kinds(tk.events, events.AlarmStopped)
    assert stopped[-1].action == "snooze" and stopped[-1].snooze_min == 10
    assert stopped[-1].alarm_id == alarm.id
    tk.clock.advance(5)
    tk.tick()
    assert len(tk.run_rec.of("paplay")) == n_plays           # stays quiet
    tk.clock.advance(600)
    tk.tick()
    assert tk.ringing is not None and tk.ringing.id == alarm.id


def test_adjust_ringing_alarm_with_a_negative_delta_is_left_alone(tk):
    tk.clock.set(D(2026, 8, 26, 7, 0))
    alarm = tk.add_alarm(tk.clock.now(), "Gym")
    tk.tick()
    before = tk.get(alarm.id)
    n_events = len(tk.events)
    assert tk.adjust("last", "alarm", -300) == []
    after = tk.get(alarm.id)
    assert after.state == "ringing" and after.due == before.due
    assert tk.ringing is not None and len(tk.events) == n_events


def test_adjust_empty_store_and_zero_delta_return_nothing(tk):
    assert tk.adjust("last", "all", 600) == []
    tk.add_timer(300, "tea")
    assert tk.adjust("last", "timer", 0) == []
    assert tk.adjust("last", "timer", None) == []
    assert tk.adjust("last", "timer", "soon") == []
    assert tk.adjust("last", "alarm", 600) == []             # wrong kind
    assert tk.adjust("laundry", "timer", 600) == []          # no such label


def test_cancel_and_adjust_share_one_targeting_rule(tk):
    """_resolve is the one place that reads "which"; cancel's own behaviour
    is pinned by test_cancel_variants above."""
    t0 = tk.clock.now()
    a = tk.add_reminder(t0 + 100, "call the dentist")
    b = tk.add_timer(300, "pasta")
    assert _ids(tk._resolve("dentist", "all")) == [a.id]
    assert _ids(tk._resolve("last", "all")) == [b.id]
    assert set(_ids(tk._resolve("all", "all"))) == {a.id, b.id}
    assert _ids(tk._resolve("next", "all")) == [a.id]
    assert _ids(tk._resolve(b.id, "all")) == [b.id]
    assert tk._resolve("nothing like this", "all") == []


def test_adjust_wording_helpers():
    assert tkm.adjust_amount_words(600) == "Ten minutes"
    assert tkm.adjust_amount_words(-300) == "Five minutes"
    assert tkm.adjust_amount_words(60) == "A minute"
    assert tkm.adjust_amount_words(3600) == "An hour"
    assert tkm.adjust_amount_words(900) == "15 minutes"
    assert tkm.adjust_amount_words(30) == "Thirty seconds" or \
        tkm.adjust_amount_words(30) == "30 seconds"
    assert tkm.adjust_line(600, "pack up the chicken in 20 minutes") == \
        "Ten minutes added, sir: pack up the chicken in 20 minutes."
    assert tkm.adjust_line(-300, ["the tea timer in 3 minutes"]) == \
        "Five minutes off, sir: the tea timer in 3 minutes."
    assert tkm.nothing_to_adjust_line("timer", 600) == "No timer to extend, sir."
    assert tkm.nothing_to_adjust_line("timer", -600) == "No timer to shorten, sir."
    assert tkm.nothing_to_adjust_line("reminder", 600) == "No reminders to push back, sir."
    assert tkm.nothing_to_adjust_line("alarm", -60) == "No alarm to bring forward, sir."
    assert tkm.nothing_to_adjust_line("all", 60) == "Nothing to extend, sir."


# --------------------------------------------- manage_schedule: extend / shorten
def test_tool_manage_schedule_extend_and_shorten(tools, tk):
    t0 = tk.clock.now()
    tools.call("set_timer", {"minutes": 10, "label": "pack up the chicken"})
    r = tools.call("manage_schedule", {"action": "extend", "kind": "timer", "minutes": 10})
    assert r.ok and r.speak == r.text
    assert r.text == "Ten minutes added, sir: pack up the chicken in 20 minutes."
    assert tk.list("timer")[0].due == pytest.approx(t0 + 1200)
    r = tools.call("manage_schedule", {"action": "shorten", "kind": "timer", "minutes": "5"})
    assert r.ok and r.text == "Five minutes off, sir: pack up the chicken in 15 minutes."
    assert tk.list("timer")[0].due == pytest.approx(t0 + 900)
    # the magnitude is what counts: a negative "minutes" with extend still adds
    r = tools.call("manage_schedule", {"action": "extend", "which": "chicken", "minutes": -5})
    assert r.ok and r.text.startswith("Five minutes added, sir:")
    assert tk.list("timer")[0].due == pytest.approx(t0 + 1200)
    # label words reach the same resolution as cancel
    tools.call("set_timer", {"minutes": 3, "label": "tea"})
    r = tools.call("manage_schedule", {"action": "extend", "which": "the tea timer",
                                       "minutes": 2})
    assert r.text == "Two minutes added, sir: tea in 5 minutes."


def test_tool_manage_schedule_extend_asks_for_the_amount(tools, tk):
    tools.call("set_timer", {"minutes": 10})
    for minutes in (None, 0, "", "0"):
        r = tools.call("manage_schedule", {"action": "extend", "minutes": minutes})
        assert not r.ok and r.text == "By how many minutes, sir?" and r.speak == r.text
    r = tools.call("manage_schedule", {"action": "shorten"})
    assert not r.ok and r.speak == "By how many minutes, sir?"
    assert tk.list("timer")[0].duration == 600                 # nothing moved


def test_tool_manage_schedule_extend_with_nothing_to_extend(tools, tk):
    r = tools.call("manage_schedule", {"action": "extend", "kind": "timer", "minutes": 10})
    assert not r.ok and r.text == "No timer to extend, sir." and r.speak == r.text
    r = tools.call("manage_schedule", {"action": "shorten", "kind": "reminder", "minutes": 5})
    assert r.text == "No reminders to bring forward, sir."
    r = tools.call("manage_schedule", {"action": "extend", "minutes": 5})
    assert r.text == "Nothing to extend, sir."


@pytest.mark.parametrize("alias,sign", [
    ("add", 1), ("more", 1), ("longer", 1), ("push back", 1), ("push", 1),
    ("delay", 1), ("defer", 1), ("prolong", 1), ("lengthen", 1),
    ("reduce", -1), ("cut", -1), ("less", -1), ("shorter", -1),
    ("bring forward", -1), ("earlier", -1), ("sooner", -1), ("subtract", -1),
    ("take off", -1),
])
def test_tool_manage_schedule_aliases(tools, tk, alias, sign):
    t0 = tk.clock.now()
    tools.call("set_timer", {"minutes": 10, "label": "tea"})
    r = tools.call("manage_schedule", {"action": alias, "kind": "timer", "minutes": 2})
    assert r.ok, (alias, r.text)
    assert tk.list("timer")[0].due == pytest.approx(t0 + 600 + sign * 120)


def test_tool_postpone_extends_when_nothing_rings_and_snoozes_when_it_does(tools, tk):
    t0 = tk.clock.now()
    tools.call("set_timer", {"minutes": 10, "label": "tea"})
    r = tools.call("manage_schedule", {"action": "postpone", "minutes": 5})
    assert r.ok and r.text == "Five minutes added, sir: tea in 15 minutes."
    assert tk.list("timer")[0].due == pytest.approx(t0 + 900)
    r = tools.call("manage_schedule", {"action": "later", "minutes": 5})
    assert r.ok and tk.list("timer")[0].due == pytest.approx(t0 + 1200)
    # ...and while an alarm rings, "postpone" is the snooze it always was
    tk.add_alarm(tk.clock.now(), "Gym")
    tk.tick()
    assert tk.ringing is not None
    r = tools.call("manage_schedule", {"action": "postpone", "minutes": "5"})
    assert r.ok and r.text == "Snoozed for five minutes, sir."
    assert tk.ringing is None and tk.list("alarm")[0].state == "snoozed"
    assert tk.list("timer")[0].due == pytest.approx(t0 + 1200)  # the timer was not touched
    # "snooze" itself is unchanged: nothing ringing -> the old line
    r = tools.call("manage_schedule", {"action": "snooze"})
    assert not r.ok and r.speak == NOTHING_RINGING_LINE


def test_tool_manage_schedule_unknown_action_and_description(tools):
    r = tools.call("manage_schedule", {"action": "fly", "minutes": 5})
    assert not r.ok and r.text == "unknown action 'fly'"
    fn = tools.schemas()[3]["function"]
    assert "extend" in fn["description"] and "shorten" in fn["description"]
    assert "extend" in fn["parameters"]["properties"]["minutes"]["description"]
