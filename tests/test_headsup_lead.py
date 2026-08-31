"""How much notice Jarvis gives before a calendar event.

Hunter asked for an unprompted heads-up 30 minutes before calendar events.
That is configuration, not new code -- jarvis/headsup.py has spoken a lead
line since the first wave -- but it takes TWO knobs, and the coupling is
the part that is easy to get wrong:

* ``calendar.heads_up_min`` is the bare "TITLE in N minutes" lead
  (jarvis/headsup.py, wired at app.py:3705).
* ``dossier.lead_min`` is the pre-class dossier's lead (jarvis/dossier.py).
  The dossier OWNS course events -- ``ClassDossier.owns`` makes the bare
  heads-up stand down so the title is not said twice -- so a
  ``calendar.heads_up_min`` raised on its own moves ONLY the one-off
  appointments and leaves every single class on the old lead.

These tests pin both paths at 30 minutes so a future edit that moves one
knob without the other fails here instead of in his kitchen.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis.dossier import ClassDossier
from jarvis.headsup import MeetingHeadsUp
from jarvis.tools.calendar import Event

TZ = timezone(timedelta(hours=-5))
MON = datetime(2026, 8, 31, tzinfo=TZ)
BIO_ROOM = "College Station Wisenbaker Engineering Bldg 049"
LEAD = 30


def bio_series():
    """BIOSENSORS at 09:10 on three Mondays -- courses.recurring_courses
    only calls a slot a course once it has seen it on several dates, so a
    single event is not a class and the dossier would not own it."""
    return [ev("BIOSENSORS", 9, 10, BIO_ROOM, day=d) for d in (0, 7, 14)]


def ev(title, hour, minute, location="", day=0, minutes=50):
    start = MON + timedelta(days=day, hours=hour, minutes=minute)
    return Event(start=start, end=start + timedelta(minutes=minutes),
                 title=title, location=location, calendar="icloud")


class FakeCal:
    configured = True

    def __init__(self, events):
        self._events = events

    def events(self):
        return list(self._events)


class Cfg:
    def __init__(self, **sections):
        self.data = {}
        for k, v in sections.items():
            self.data.setdefault(k, {}).update(v)

    def get(self, dotted, default=None):
        obj = self.data
        for part in dotted.split("."):
            if not isinstance(obj, dict) or part not in obj:
                return default
            obj = obj[part]
        return obj


class FakeTimekeeper:
    def __init__(self):
        self.silent, self.reminders = [], []

    def add_silent_timer(self, seconds, label=""):
        item = SimpleNamespace(id=f"id{len(self.silent)}", label=label,
                               seconds=seconds)
        self.silent.append(item)
        return item

    def add_reminder(self, due, text, repeat=""):
        self.reminders.append((due, text))


@pytest.fixture
def at_eight_thirty():
    """08:30 on the Monday: 40 minutes before the 09:10 class."""
    return MON + timedelta(hours=8, minutes=30)


def _clock(now):
    return lambda tzinfo=None: now.astimezone(tzinfo) if tzinfo else now


# ------------------------------------------------- the one-off appointment
def test_a_one_off_appointment_is_announced_thirty_minutes_ahead(
        at_eight_thirty, tmp_path):
    tk = FakeTimekeeper()
    chiro = [ev("Chiro", 9, 10)]
    h = MeetingHeadsUp(lambda: FakeCal(chiro), tk, lead_min=LEAD,
                       state_path=tmp_path / "h.json",
                       now=_clock(at_eight_thirty))
    assert h.tick() == 1
    due, text = tk.reminders[0]
    assert text == "Chiro in 30 minutes"
    spoken_at = datetime.fromtimestamp(due, TZ)
    assert spoken_at == MON + timedelta(hours=8, minutes=40), \
        "a 09:10 appointment is announced at 08:40, not 09:00"


# ------------------------------------------------------------- the class
def test_a_class_gets_the_same_thirty_minutes_through_the_dossier(
        at_eight_thirty, tmp_path):
    """The dossier is what speaks for a course event, so the lead Hunter
    actually experiences for a class is dossier.lead_min, not
    calendar.heads_up_min."""
    tk = FakeTimekeeper()
    cfg = Cfg(docs={"paths": [str(tmp_path / "docs")]}, dossier={"mail": False})
    d = ClassDossier(cfg, tk, get_calendar=lambda: FakeCal(bio_series()),
                     services=SimpleNamespace(speak=lambda *a, **k: None,
                                              memory=SimpleNamespace(people=lambda: {})),
                     state_path=tmp_path / "d.json", lead_min=LEAD,
                     now=_clock(at_eight_thirty), bg=lambda fn: fn(),
                     publish=lambda *a, **k: None, fetch_due=lambda *a, **k: [])
    try:
        assert d.tick() == 1
        item, = tk.silent
        assert "BIOSENSORS" in item.label
        # 09:10 start, 08:30 now, 30-minute lead -> fires in 10 minutes.
        assert abs(item.seconds - 10 * 60) < 1, \
            f"T-30 for a 09:10 class at 08:30 is 10 minutes out, got {item.seconds}s"
    finally:
        d.stop()


def test_the_bare_heads_up_still_stands_down_for_a_class(
        at_eight_thirty, tmp_path):
    """Both at 30 minutes would say "BIOSENSORS" twice at 08:40."""
    tk = FakeTimekeeper()
    cfg = Cfg(docs={"paths": [str(tmp_path / "docs")]}, dossier={"mail": False})
    cal = FakeCal(bio_series())
    d = ClassDossier(cfg, tk, get_calendar=lambda: cal,
                     services=SimpleNamespace(speak=lambda *a, **k: None,
                                              memory=SimpleNamespace(people=lambda: {})),
                     state_path=tmp_path / "d.json", lead_min=LEAD,
                     now=_clock(at_eight_thirty), bg=lambda fn: fn(),
                     publish=lambda *a, **k: None, fetch_due=lambda *a, **k: [])
    try:
        h = MeetingHeadsUp(lambda: cal, tk, lead_min=LEAD,
                           state_path=tmp_path / "h.json",
                           now=_clock(at_eight_thirty), skip=d.owns)
        assert h.tick() == 0, "the dossier speaks for a class; the bare line must not"
        assert tk.reminders == []
    finally:
        d.stop()


# ------------------------------------------------------------ the horizon
def test_thirty_minutes_is_well_inside_the_ninety_minute_horizon(tmp_path):
    """HORIZON_MIN is 90 in both modules; a 30-minute lead must not be
    clipped by it, and an event beyond the horizon must still be ignored."""
    tk = FakeTimekeeper()
    now = MON + timedelta(hours=8, minutes=30)
    far = [ev("Chiro", 13, 0)]          # 4.5 h out: outside the horizon
    h = MeetingHeadsUp(lambda: FakeCal(far), tk, lead_min=LEAD,
                       state_path=tmp_path / "far.json", now=_clock(now))
    assert h.tick() == 0 and tk.reminders == []

    near = [ev("Chiro", 9, 45)]         # 75 min out: inside it
    h2 = MeetingHeadsUp(lambda: FakeCal(near), tk, lead_min=LEAD,
                        state_path=tmp_path / "near.json", now=_clock(now))
    assert h2.tick() == 1
    assert tk.reminders[0][1] == "Chiro in 30 minutes"


def test_an_event_already_inside_the_lead_is_announced_at_once(tmp_path):
    """A restart at 08:55 before a 09:00 appointment must still say
    something, and must say the TRUE remaining minutes, not "30"."""
    tk = FakeTimekeeper()
    now = MON + timedelta(hours=8, minutes=55)
    h = MeetingHeadsUp(lambda: FakeCal([ev("Chiro", 9, 0)]), tk, lead_min=LEAD,
                       state_path=tmp_path / "s.json", now=_clock(now))
    assert h.tick() == 1
    assert tk.reminders[0][1] == "Chiro in 5 minutes"
