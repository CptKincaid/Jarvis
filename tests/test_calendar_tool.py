"""Tests for jarvis.tools.calendar (spec 6.4): ICS parsing with recurrences
and all-day events, exact wording across today / tomorrow / week / next at
a fixed `now`, the conditional-GET refresh, the disk cache and its "as of"
wording, the never-fetch-synchronously rule, the iCloud path through a fake
CalDAV client, timezone handling and the unconfigured excuse.

No network: the module's ``_fetch`` seam and the ``dav_client`` factory are
replaced.  Cache files live in tmp (tests/conftest.py also redirects
JARVIS_CACHE_DIR / JARVIS_ASSISTANT_CONFIG).  No secrets: the fake iCloud
password is a made-up placeholder and the tests assert it never reaches
the cache file."""
import json
import os
import threading
from pathlib import Path
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.calendar as calendar
from jarvis.tools.calendar import (CalendarSource, Event, as_of_words,
                                   coerce_range, describe_due, format_events,
                                   parse_ics)
from jarvis.tools.location import Response
from jarvis.tools.registry import ToolRegistry

CHI = ZoneInfo("America/Chicago")
NOW = datetime(2026, 8, 26, 9, 0, tzinfo=CHI)          # Wednesday 9:00 am CDT
EPOCH = NOW.timestamp()
URL = "https://calendar.google.com/calendar/ical/placeholder-secret-token/basic.ics"

FIXTURES = Path(__file__).parent / "fixtures"
ICS = (FIXTURES / "calendar_sample.ics").read_bytes()

TODAY_TEXT = ("Today: 7:00 am Run, 10:00 am Dentist for an hour at Main Street Dental, "
              "2:30 pm Standup; nothing else.")
WEEK_TEXT = ("This week: today 7:00 am Run, 10:00 am Dentist for an hour at Main Street "
             "Dental, 2:30 pm Standup; tomorrow all day: Mum's birthday; Friday 10:00 am "
             "UTC call for an hour and a half; Saturday all day: Conference; Sunday all "
             "day: Conference; Monday 3:00 am Sync with London; nothing else.")


# ------------------------------------------------------------- fixtures
class FakeCfg:
    def __init__(self, urls=(), apple_id="", app_password="", url=""):
        self.data = {"google_ical_urls": list(urls),
                     "icloud": {"apple_id": apple_id, "app_password": app_password,
                                "url": url or "https://caldav.icloud.com"}}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def setup_line(self, section):
        return f"I'll need your {section} set up, sir; the notes are in docs/assistant-setup.md."


class IcsServer:
    """Fake ``_fetch`` for one ICS URL with ETag / Last-Modified support."""

    def __init__(self, body=ICS, etag='"v1"', fail=False):
        self.body, self.etag, self.fail = body, etag, fail
        self.calls = []

    def __call__(self, url, timeout=8, headers=None):
        self.calls.append((url, dict(headers or {})))
        if self.fail:
            raise OSError("offline")
        if headers and headers.get("If-None-Match") == self.etag:
            r = Response(b"")
            r.status, r.headers = 304, {"etag": self.etag}
            return r
        r = Response(self.body)
        r.status = 200
        r.headers = {"etag": self.etag, "last-modified": "Wed, 26 Aug 2026 13:00:00 GMT"}
        return r


class FakeObj:
    def __init__(self, data):
        self.data = data


class FakeCal:
    def __init__(self, name, objs, fail=False):
        self.name, self.objs, self.fail = name, objs, fail
        self.searches = []

    def search(self, **kw):
        self.searches.append(kw)
        if self.fail:
            raise RuntimeError("dav search failed")
        return self.objs


class FakePrincipal:
    def __init__(self, cals):
        self._cals = cals

    def calendars(self):
        return self._cals


class FakeDAV:
    """Records the constructor args; ``calendars`` is set by the test."""
    instances = []
    calendars = []
    raise_on_connect = None

    def __init__(self, url, username, password):
        self.url, self.username, self.password = url, username, password
        FakeDAV.instances.append(self)
        if FakeDAV.raise_on_connect:
            raise FakeDAV.raise_on_connect

    def principal(self):
        return FakePrincipal(FakeDAV.calendars)


ICLOUD_ICS = (FIXTURES / "icloud_sample.ics").read_text()


@pytest.fixture(autouse=True)
def _reset_fake_dav():
    FakeDAV.instances, FakeDAV.calendars, FakeDAV.raise_on_connect = [], [], None
    yield


def make_source(tmp_path, cfg=None, fetch=None, clock=None, **kw):
    clock = clock or (lambda: EPOCH)
    return CalendarSource(cfg or FakeCfg(urls=[URL]), cache_path=tmp_path / "cal.json",
                          fetch=fetch or IcsServer(), dav_client=FakeDAV, clock=clock,
                          tz=CHI, **kw)


def events_from_fixture():
    return parse_ics(ICS, NOW - timedelta(days=1), NOW + timedelta(days=15), tz=CHI)


# --------------------------------------------------------------- parsing
def test_parse_ics_expands_recurrence_all_day_and_timezones():
    evs = events_from_fixture()
    titles = [e.title for e in evs]
    assert titles.count("Standup") == 3                 # Aug 26, Sep 2, Sep 9 (window 15 d)
    assert "Lunch (cancelled)" not in titles
    by = {e.title: e for e in evs}
    assert by["Dentist"].calendar == "Personal" and by["Dentist"].location == "Main Street Dental"
    assert by["Mum's birthday"].all_day and by["Mum's birthday"].start == \
        datetime(2026, 8, 27, tzinfo=CHI)
    assert by["Mum's birthday"].end == datetime(2026, 8, 28, tzinfo=CHI)
    assert by["UTC call"].start == datetime(2026, 8, 28, 10, 0, tzinfo=CHI)      # 15:00Z
    assert by["UTC call"].end - by["UTC call"].start == timedelta(minutes=90)     # DURATION
    assert by["Sync with London"].start == datetime(2026, 8, 31, 3, 0, tzinfo=CHI)  # 9:00 BST
    assert by["Conference"].on(datetime(2026, 8, 29).date()) and \
        by["Conference"].on(datetime(2026, 8, 30).date()) and \
        not by["Conference"].on(datetime(2026, 8, 31).date())
    assert all(e.start.tzinfo is not None for e in evs)


def test_parse_ics_floating_time_is_local_and_bad_ics_raises():
    floating = (b"BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:x\nBEGIN:VEVENT\nUID:f\n"
                b"DTSTART:20260826T160000\nSUMMARY:Floating\nEND:VEVENT\nEND:VCALENDAR\n")
    evs = parse_ics(floating, NOW - timedelta(days=1), NOW + timedelta(days=2), tz=CHI)
    assert evs[0].start == datetime(2026, 8, 26, 16, 0, tzinfo=CHI) and evs[0].end == evs[0].start
    assert evs[0].calendar == "" and evs[0].title == "Floating"
    with pytest.raises(Exception):
        parse_ics(b"not a calendar", NOW, NOW + timedelta(days=1), tz=CHI)


def test_event_round_trips_through_json():
    ev = events_from_fixture()[0]
    back = Event.from_dict(json.loads(json.dumps(ev.to_dict())), CHI)
    assert back == ev


# --------------------------------------------------------------- wording
def test_format_events_exact_strings():
    evs = events_from_fixture()
    assert format_events(evs, "today", NOW) == TODAY_TEXT
    assert format_events(evs, "tomorrow", NOW) == "Tomorrow: all day: Mum's birthday; nothing else."
    assert format_events(evs, "week", NOW) == WEEK_TEXT
    assert format_events(evs, "this week", NOW) == WEEK_TEXT
    assert format_events(evs, "next", NOW) == \
        "Next: Dentist in an hour, at 10:00 am, for an hour, at Main Street Dental."
    later = NOW.replace(hour=15)
    assert format_events(evs, "next", later) == \
        "Next: Mum's birthday, all day tomorrow."
    late = NOW.replace(hour=23, minute=30) + timedelta(days=1)          # Thu 11:30 pm
    assert format_events(evs, "next", late) == \
        "Next: UTC call at 10:00 am tomorrow, for an hour and a half."
    sunday = datetime(2026, 8, 30, 12, 0, tzinfo=CHI)
    assert format_events(evs, "next", sunday) == "Next: Sync with London at 3:00 am tomorrow."
    monday = datetime(2026, 8, 31, 12, 0, tzinfo=CHI)
    assert format_events(evs, "next", monday) == "Next: Standup on Wednesday at 2:30 pm."
    next_week = datetime(2026, 9, 2, 15, 0, tzinfo=CHI)
    assert format_events(evs, "next", next_week) == \
        "Next: Board meeting at 10:00 am tomorrow, for 2 hours."


def test_format_events_empty_and_all_day_first():
    assert format_events([], "today", NOW) == "Nothing on today, sir."
    assert format_events([], "tomorrow", NOW) == "Nothing on tomorrow, sir."
    assert format_events([], "week", NOW) == "Nothing on this week, sir."
    assert format_events([], "next", NOW) == "Nothing coming up in the next two weeks, sir."
    evs = [Event(NOW.replace(hour=8), NOW.replace(hour=8, minute=30), False, "Coffee"),
           Event(NOW.replace(hour=0), NOW.replace(hour=0) + timedelta(days=1), True, "Holiday")]
    assert format_events(evs, "today", NOW) == "Today: all day: Holiday, 8:00 am Coffee; nothing else."
    # duplicates from two sources collapse
    assert format_events(evs + evs, "today", NOW) == \
        "Today: all day: Holiday, 8:00 am Coffee; nothing else."


def test_describe_due_table():
    cases = [(timedelta(seconds=30), "now"), (timedelta(minutes=1), "in 1 minute"),
             (timedelta(minutes=10), "in 10 minutes"), (timedelta(minutes=59), "in 59 minutes"),
             (timedelta(hours=1), "in an hour"), (timedelta(hours=1, minutes=30), "in an hour and a half"),
             (timedelta(hours=2), "in 2 hours"), (timedelta(hours=2, minutes=40), "in 2 and a half hours"),
             (timedelta(hours=2, minutes=50), "in 3 hours"), (timedelta(hours=10), "at 7:00 pm today"),
             (timedelta(hours=22), "at 7:00 am tomorrow"), (timedelta(days=2), "on Friday at 9:00 am"),
             (timedelta(days=10), "on the 5th of September at 9:00 am")]
    for delta, want in cases:
        assert describe_due(NOW + delta, NOW) == want, delta


def test_as_of_words_and_range_coercion():
    assert as_of_words(NOW.replace(minute=10).timestamp(), NOW) == "9:10 am"
    assert as_of_words((NOW - timedelta(days=1)).timestamp(), NOW) == "9:00 am yesterday"
    assert as_of_words((NOW - timedelta(days=2)).timestamp(), NOW) == "9:00 am on Monday"
    assert as_of_words((NOW - timedelta(days=9)).timestamp(), NOW) == "9:00 am on the 17th of August"
    for raw, want in (("this week", "week"), ("Week", "week"), ("tmrw", "tomorrow"),
                      ("what's next", "next"), ("upcoming", "next"), ("", "today"),
                      (None, "today"), ("Today.", "today"), ("next 7 days", "week"), (3, "today")):
        assert coerce_range(raw) == want, raw


# ---------------------------------------------------------------- source
def test_unconfigured_and_placeholders(tmp_path):
    assert not make_source(tmp_path, FakeCfg()).configured
    assert not make_source(tmp_path, FakeCfg(urls=["<paste the secret ical url>"],
                                             apple_id="<apple id>", app_password="<pw>")).configured
    assert make_source(tmp_path, FakeCfg(urls=[URL])).configured
    assert make_source(tmp_path, FakeCfg(apple_id="hunter@example.com",
                                         app_password="xxxx-xxxx-xxxx-xxxx")).configured
    assert make_source(tmp_path, FakeCfg(apple_id="hunter@example.com")).configured is False


def test_refresh_conditional_get_and_disk_cache(tmp_path):
    server = IcsServer()
    src = make_source(tmp_path, fetch=server)
    assert src.fetched_at is None and src.is_stale()
    assert src.refresh(NOW) is True
    assert src.errors == [] and src.fetched_at == EPOCH and not src.is_stale()
    assert format_events(src.events(), "today", NOW) == TODAY_TEXT
    assert server.calls[0][1] == {}                     # first GET unconditional
    assert src.refresh(NOW) is True                     # second: 304 keeps the events
    assert server.calls[1][1] == {"If-None-Match": '"v1"',
                                  "If-Modified-Since": "Wed, 26 Aug 2026 13:00:00 GMT"}
    assert format_events(src.events(), "today", NOW) == TODAY_TEXT
    # the disk cache carries the events and stamps but never the secret URL
    raw = (tmp_path / "cal.json").read_text()
    assert "placeholder-secret-token" not in raw and URL not in raw
    saved = json.loads(raw)
    assert saved["sources"]["google-1"]["fetched_at"] == EPOCH
    # a fresh process answers from disk without any fetch
    offline = IcsServer(fail=True)
    boot = make_source(tmp_path, fetch=offline, clock=lambda: EPOCH + 3600)
    assert offline.calls == [] and boot.fetched_at == EPOCH and boot.is_stale()
    assert format_events(boot.events(), "week", NOW) == WEEK_TEXT


def test_failed_source_keeps_previous_events(tmp_path):
    server = IcsServer()
    src = make_source(tmp_path, fetch=server)
    src.refresh(NOW)
    server.fail = True
    assert src.refresh(NOW) is False
    assert src.errors == ["google-1: OSError"]
    assert format_events(src.events(), "today", NOW) == TODAY_TEXT
    assert src.fetched_at == EPOCH


def test_get_never_fetches_and_triggers_refresh_when_stale(tmp_path):
    server = IcsServer()
    src = make_source(tmp_path, fetch=server)
    triggered = []
    src.trigger_refresh = lambda: triggered.append(1)
    snap = src.get("today", NOW)
    assert snap.events == [] and snap.fetched_at is None and snap.stale
    assert triggered == [1] and server.calls == []
    src.refresh(NOW)
    snap = src.get("today", NOW)
    assert not snap.stale and triggered == [1]
    src._clock = lambda: EPOCH + 601
    snap = src.get("today", NOW)
    assert snap.stale and triggered == [1, 1] and len(server.calls) == 1


def test_worker_thread_refreshes_and_stops(tmp_path):
    fetched = threading.Event()

    class Server(IcsServer):
        def __call__(self, url, timeout=8, headers=None):
            try:
                return super().__call__(url, timeout, headers)
            finally:
                fetched.set()
    src = make_source(tmp_path, fetch=Server(), refresh_s=60)
    src.start()
    assert fetched.wait(3.0)
    src.stop()
    assert not src._thread.is_alive()
    assert format_events(src.events(), "tomorrow", NOW) == \
        "Tomorrow: all day: Mum's birthday; nothing else."
    # the tool's trigger wakes the running worker instead of spawning threads
    fetched.clear()
    src.start()
    src.trigger_refresh()
    assert fetched.wait(3.0)
    src.stop()


def test_icloud_path_with_fake_client(tmp_path):
    FakeDAV.calendars = [FakeCal("Work", [FakeObj(ICLOUD_ICS)]),
                         FakeCal("Broken", [], fail=True)]
    cfg = FakeCfg(urls=[URL], apple_id="hunter@example.com",
                  app_password="placeholder-app-password")
    src = make_source(tmp_path, cfg=cfg)
    assert src.refresh(NOW) is True
    dav = FakeDAV.instances[0]
    assert (dav.url, dav.username, dav.password) == \
        ("https://caldav.icloud.com", "hunter@example.com", "placeholder-app-password")
    search = FakeDAV.calendars[0].searches[0]
    assert search["event"] is True and search["expand"] is True
    assert search["start"] <= NOW <= search["end"]
    gym = [e for e in src.events() if e.title == "Gym"][0]
    assert gym.calendar == "Work" and gym.start == datetime(2026, 8, 26, 18, 0, tzinfo=CHI)
    assert format_events(src.events(), "today", NOW) == (
        "Today: 7:00 am Run, 10:00 am Dentist for an hour at Main Street Dental, "
        "2:30 pm Standup, 6:00 pm Gym for an hour; nothing else.")
    assert src.errors == []                     # one broken calendar is only logged
    assert "placeholder-app-password" not in (tmp_path / "cal.json").read_text()
    # the iCloud login failing keeps the Google events and reports the source
    FakeDAV.raise_on_connect = RuntimeError("401 for placeholder-app-password")
    assert src.refresh(NOW) is True
    assert src.errors == ["icloud: RuntimeError"]
    assert [e.title for e in src.events() if e.title == "Gym"] == ["Gym"]


def test_icloud_only_configuration(tmp_path):
    FakeDAV.calendars = [FakeCal("Home", [FakeObj(ICLOUD_ICS)])]
    src = make_source(tmp_path, cfg=FakeCfg(apple_id="a@b.c", app_password="pw-placeholder"),
                      fetch=IcsServer(fail=True))
    assert src.refresh(NOW) is True
    assert [e.title for e in src.events()] == ["Gym"] and src.errors == []


# ------------------------------------------------------------------ tool
def fixed_datetime(monkeypatch, at):
    """Freeze the module's ``now_local`` seam.  (Patching the ``datetime``
    name itself would break the ``isinstance`` checks in ``parse_ics``, which
    is how all-day events are told from timed ones.)"""
    monkeypatch.setattr(calendar, "now_local",
                        lambda tz=None: at.astimezone(tz) if tz else at)


def tool_registry(tmp_path, cfg, monkeypatch, fetch=None, clock=None):
    services = SimpleNamespace()
    src = make_source(tmp_path, cfg=cfg, fetch=fetch, clock=clock)
    services.calendar = src
    reg = ToolRegistry()
    reg.register_many(calendar.make_tools(cfg, services))
    assert services.calendar is src              # make_tools reuses the parked source
    fixed_datetime(monkeypatch, NOW)
    return reg, src


def test_tool_spec_and_unconfigured_excuse(tmp_path, monkeypatch):
    specs = calendar.make_tools(FakeCfg(), SimpleNamespace())
    assert [s.name for s in specs] == ["get_calendar"]
    assert len(specs[0].description.split()) <= 20
    enum = specs[0].parameters["properties"]["range"]["enum"]
    assert enum[:4] == ["today", "tomorrow", "week", "next"]
    # weekday names joined the enum 2026-08-28 so "agenda for Monday" can be
    # asked for directly instead of degrading to "next" (one event).
    assert "monday" in enum and "sunday" in enum
    reg, _ = tool_registry(tmp_path, FakeCfg(), monkeypatch)
    r = reg.call("get_calendar", {"range": "today"})
    assert not r.ok and r.text == \
        "I'll need your google_ical set up, sir; the notes are in docs/assistant-setup.md."
    assert r.speak == r.text


def test_tool_loading_fresh_and_stale_wording(tmp_path, monkeypatch):
    clock = {"t": EPOCH - 600}                     # fetched at 8:50 am
    reg, src = tool_registry(tmp_path, FakeCfg(urls=[URL]), monkeypatch,
                             clock=lambda: clock["t"])
    src.trigger_refresh = lambda: None
    r = reg.call("get_calendar", {"range": "today"})
    assert not r.ok and r.text == "calendar still loading, ask again in a moment"
    src.refresh(NOW)
    clock["t"] = EPOCH - 300                       # 5 min later: fresh
    assert reg.call("get_calendar", {"range": "today"}).text == TODAY_TEXT
    assert reg.call("get_calendar", {"range": "this week"}).text == WEEK_TEXT
    assert reg.call("get_calendar", {"range": "tomorrow", "junk": 1}).text == \
        "Tomorrow: all day: Mum's birthday; nothing else."
    assert reg.call("get_calendar", {}).text == TODAY_TEXT
    clock["t"] = EPOCH + 100                       # 11 min after the fetch: stale
    r = reg.call("get_calendar", {"range": "next"})
    assert r.ok and r.text == ("Next: Dentist in an hour, at 10:00 am, for an hour, "
                               "at Main Street Dental. That's as of 8:50 am.")


def test_tool_unreachable_wording(tmp_path, monkeypatch):
    reg, src = tool_registry(tmp_path, FakeCfg(urls=[URL]), monkeypatch,
                             fetch=IcsServer(fail=True))
    src.trigger_refresh = lambda: None
    src.refresh(NOW)
    r = reg.call("get_calendar", {"range": "today"})
    assert not r.ok and r.text == "calendar unreachable"


def test_briefing_call_shape(tmp_path, monkeypatch):
    """W4's briefing calls registry.call('get_calendar', {'range': 'today'})."""
    reg, src = tool_registry(tmp_path, FakeCfg(urls=[URL]), monkeypatch)
    src.refresh(NOW)
    assert reg.call("get_calendar", {"range": "today"}).text == TODAY_TEXT


# ------------------------------------------------------------------ live
@pytest.mark.skipif(not os.environ.get("JARVIS_LIVE"), reason="JARVIS_LIVE=1 only")
def test_live_public_google_ics(tmp_path):
    """Read-only: Google's public US-holidays calendar (no secret URL)."""
    url = ("https://calendar.google.com/calendar/ical/"
           "en.usa%23holiday%40group.v.calendar.google.com/public/basic.ics")
    src = CalendarSource(FakeCfg(urls=[url]), cache_path=tmp_path / "cal.json",
                         tz=CHI, window_days=120)
    assert src.refresh() is True and src.errors == []
    assert any(e.all_day for e in src.events())
    assert src.refresh() is True                       # conditional GET path


# ------------------------------------------------------- named weekdays
#
# 2026-08-28 01:03: "What's on my agenda for Monday?" returned only
# "Next: BIOSENSORS on Monday at 9:10 am" -- one event, when Monday held four.
# The tool's range enum was (today, tomorrow, week, next), so "Monday" had
# nowhere to land and the model picked "next", which means the single next
# event. The model answered correctly for the vocabulary it was given.
# Friday 28 Aug 2026; the following Monday is the 31st.
FRI = datetime(2026, 8, 28, 9, 0).astimezone()
MON = FRI + timedelta(days=3)


def _ev(when, title):
    return Event(start=when, end=when + timedelta(hours=1), title=title,
                 calendar="Navigate360 - Courses")


MONDAY_CLASSES = [
    _ev(MON.replace(hour=9, minute=10), "BIOSENSORS"),
    _ev(MON.replace(hour=12, minute=40), "MAGNETIC RESONANCE ENGR"),
    _ev(MON.replace(hour=16, minute=10), "ELECTRICAL DESIGN LAB II"),
    _ev(MON.replace(hour=18, minute=0), "MAGNETIC RESONANCE ENGR"),
]


def test_weekday_names_are_a_valid_range():
    assert "monday" in calendar.RANGES
    for word, want in (("monday", "monday"), ("Monday", "monday"),
                       ("on monday", "monday"), ("for Monday", "monday"),
                       ("this monday", "monday")):
        assert coerce_range(word) == want, word


def test_a_named_weekday_lists_every_event_that_day():
    """The actual regression: four classes, not just the earliest."""
    text = format_events(MONDAY_CLASSES, "monday", FRI)
    for title in ("BIOSENSORS", "MAGNETIC RESONANCE ENGR",
                  "ELECTRICAL DESIGN LAB II"):
        assert title in text, f"{title} missing from {text!r}"
    assert text.count("MAGNETIC RESONANCE ENGR") == 2, "both sittings"


def test_a_weekday_resolves_forward_not_backward():
    """Asked on Friday, 'Monday' means the coming Monday."""
    text = format_events(MONDAY_CLASSES, "monday", FRI)
    assert "BIOSENSORS" in text
    # an event on the PREVIOUS Monday must not be picked up
    old = [_ev((FRI - timedelta(days=4)).replace(hour=9), "OLD CLASS")]
    assert "OLD CLASS" not in format_events(old, "monday", FRI)


def test_todays_own_weekday_means_today():
    friday_ev = [_ev(FRI.replace(hour=15), "OFFICE HOURS")]
    assert "OFFICE HOURS" in format_events(friday_ev, "friday", FRI)


def test_an_empty_weekday_says_so():
    text = format_events([], "monday", FRI)
    assert "monday" in text.lower() and "nothing" in text.lower()
