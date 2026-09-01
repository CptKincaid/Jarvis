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

TODAY_TEXT = ("Today: 7:00 am Run for 30 minutes, 10:00 am Dentist for an hour at Main "
              "Street Dental, 2:30 pm Standup for 15 minutes; nothing else.")
WEEK_TEXT = ("This week: today 7:00 am Run for 30 minutes, 10:00 am Dentist for an hour "
             "at Main Street Dental, 2:30 pm Standup for 15 minutes; tomorrow all day: "
             "Mum's birthday; Friday 10:00 am UTC call for an hour and a half; Saturday "
             "all day: Conference; Sunday all day: Conference; Monday 3:00 am Sync with "
             "London for 30 minutes; nothing else.")


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


# ------------------------------------------------- the link in the body
# His Canvas feed (and every Zoom invitation) leaves LOCATION empty and puts
# the join link in the DESCRIPTION, so the dossier's JOIN row was blank for
# exactly the classes that are online.
ZOOM_URL = "https://tamu.zoom.us/j/94324046592?pwd=Ic6ifN6gKbxvhRCvfFypmo6hSM0kYw.1"


def _ics_with_description(body: str) -> bytes:
    return ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:x\r\nBEGIN:VEVENT\r\nUID:z\r\n"
            "DTSTART:20260826T160000\r\nSUMMARY:Office hours\r\n"
            f"DESCRIPTION:{body}\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n").encode()


def test_parse_ics_carries_the_description_and_finds_the_join_link():
    raw = _ics_with_description(f"Join Zoom Meeting\\n{ZOOM_URL}\\n\\nPasscode: 12345")
    ev = parse_ics(raw, NOW - timedelta(days=1), NOW + timedelta(days=2), tz=CHI)[0]
    assert ZOOM_URL in ev.description
    assert ev.meeting_url() == ZOOM_URL
    assert ev.location == ""


def test_a_location_url_still_wins_over_the_body():
    """An organiser who put the URL in LOCATION meant that one."""
    ev = Event(start=NOW, end=NOW, location="https://meet.google.com/abc-defg-hij",
               description=f"old link {ZOOM_URL}")
    assert ev.meeting_url() == "https://meet.google.com/abc-defg-hij"


@pytest.mark.parametrize("body, url", [
    (f"Join: {ZOOM_URL}", ZOOM_URL),
    ("https://teams.microsoft.com/l/meetup-join/19%3ameeting_abc/0",
     "https://teams.microsoft.com/l/meetup-join/19%3ameeting_abc/0"),
    ("Video call: https://meet.google.com/abc-defg-hij.",
     "https://meet.google.com/abc-defg-hij"),          # the full stop is prose
    ("(https://tamu.webex.com/meet/hunter)", "https://tamu.webex.com/meet/hunter"),
    ("", ""),
])
def test_meeting_url_takes_conferencing_links(body, url):
    assert calendar.meeting_url(body) == url


@pytest.mark.parametrize("body", [
    "Read https://canvas.tamu.edu/courses/12345/assignments/9 before class",
    "Slides at https://drive.google.com/file/d/abc/view",
    "unsubscribe: https://example.com/u?token=1",
    "no link at all",
])
def test_a_body_full_of_other_links_is_not_a_join_link(body):
    """Reading the assignment URL out as the way into a lecture is worse
    than saying nothing, so the hosts are an allow-list."""
    assert calendar.meeting_url(body) == ""


def test_a_long_body_keeps_the_link_past_the_cap():
    """A Canvas body opens with prose and puts the Zoom block underneath --
    precisely where a blind truncation would cut the one thing carried."""
    body = ("lorem ipsum " * 200) + ZOOM_URL
    trimmed = calendar.trim_description(body)
    assert len(trimmed) <= calendar.DESCRIPTION_CHARS
    assert calendar.meeting_url(trimmed) == ZOOM_URL


def test_the_description_survives_the_disk_cache(tmp_path):
    ev = Event(start=NOW, end=NOW, title="Office hours",
               description=f"Join Zoom Meeting {ZOOM_URL}")
    back = Event.from_dict(ev.to_dict())
    assert back.meeting_url() == ZOOM_URL
    # A cache file written before descriptions existed has no such key.
    old = {k: v for k, v in ev.to_dict().items() if k != "description"}
    assert Event.from_dict(old).meeting_url() == ""


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
    assert format_events(evs, "next", sunday) == \
        "Next: Sync with London at 3:00 am tomorrow, for 30 minutes."
    monday = datetime(2026, 8, 31, 12, 0, tzinfo=CHI)
    assert format_events(evs, "next", monday) == \
        "Next: Standup on Wednesday at 2:30 pm, for 15 minutes."
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
    assert format_events(evs, "today", NOW) == "Today: all day: Holiday, 8:00 am Coffee for 30 minutes; nothing else."
    # duplicates from two sources collapse
    assert format_events(evs + evs, "today", NOW) == \
        "Today: all day: Holiday, 8:00 am Coffee for 30 minutes; nothing else."


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


# ------------------------------------------------- a source that is down
#
# LIVE 2026-08-31: the Canvas subscription ("google-2") timed out on every
# refresh from 12:40 to 14:54.  Three separate faults fell out of that, one
# test each.
class TwoFeeds:
    """Two ICS urls; the second one always times out."""

    def __init__(self, bad_from=0):
        self.calls = []
        self.bad_from = bad_from        # call index after which "b" starts failing

    def __call__(self, url, timeout=8, headers=None):
        self.calls.append(url)
        if url.endswith("b.ics") and self.calls.count(url) > self.bad_from:
            raise TimeoutError("The read operation timed out")
        r = Response(ICS)
        r.status, r.headers = 200, {}
        return r


def _two_feed_source(tmp_path, fetch, clock):
    cfg = FakeCfg(urls=["https://a.test/a.ics", "https://b.test/b.ics"])
    return CalendarSource(cfg, cache_path=tmp_path / "cal.json", fetch=fetch,
                          dav_client=FakeDAV, clock=clock, tz=CHI)


def test_a_dead_feed_is_backed_off_instead_of_retried_every_cycle(tmp_path):
    """FETCH_TIMEOUT per cycle per dead feed, for hours, is not a plan."""
    t = {"now": EPOCH}
    server = TwoFeeds()
    src = _two_feed_source(tmp_path, server, lambda: t["now"])
    for _ in range(6):                       # six refresh cycles
        src.refresh(NOW)
        t["now"] += calendar.REFRESH_S
    good = server.calls.count("https://a.test/a.ics")
    bad = server.calls.count("https://b.test/b.ics")
    assert good == 6, server.calls          # the healthy feed is never skipped
    # back-off 600, then 1200, then 2400: tried on cycles 1, 2 and 4 only.
    # Without it that is 6 fetches and 6 x FETCH_TIMEOUT of worker time.
    assert bad == 3, server.calls
    # and a skipped source still reads as down, never as healthy
    assert src.errors == ["google-2: TimeoutError"], src.errors
    assert [d.id for d in src.down()] == ["google-2"]


def test_one_dead_feed_does_not_make_the_others_look_stale(tmp_path):
    """fetched_at was min() over EVERY source, so one dead feed pinned the
    whole cache to its last success and every answer said "as of ...""" \
        """ hours ago." """
    t = {"now": EPOCH}
    server = TwoFeeds(bad_from=1)            # b answers once, then dies
    src = _two_feed_source(tmp_path, server, lambda: t["now"])
    src.refresh(NOW)
    assert not src.is_stale() and src.down() == []
    t["now"] = EPOCH + 4 * calendar.REFRESH_S     # much later; b keeps failing
    src.refresh(NOW)
    assert src.fetched_at == t["now"], src.fetched_at   # the fresh feed decides
    assert not src.is_stale()
    down = src.down()
    assert [d.id for d in down] == ["google-2"]
    assert down[0].since == EPOCH            # the last time it DID answer


def test_a_stale_cache_kicks_one_refresh_per_period_not_one_per_question(tmp_path):
    """Every get() used to queue another full refresh -- another
    FETCH_TIMEOUT on the dead feed -- so asking twice cost twice."""
    t = {"now": EPOCH + 10 * calendar.REFRESH_S}
    server = TwoFeeds()
    src = _two_feed_source(tmp_path, server, lambda: t["now"])
    triggered = []
    src.trigger_refresh = lambda: triggered.append(t["now"])
    for _ in range(5):
        snap = src.get("today", NOW)
        assert snap.stale
    assert triggered == [t["now"]], triggered
    t["now"] += calendar.REFRESH_S           # a period later, ask again
    src.get("today", NOW)
    assert len(triggered) == 2, triggered


def test_get_calendar_names_the_feed_that_went_down(tmp_path, monkeypatch):
    """Hunter's real case: a feed that answered this morning and then
    stopped. It has told us its name, so Jarvis can use it."""
    t = {"now": EPOCH}
    server = TwoFeeds(bad_from=1)
    src = _two_feed_source(tmp_path, server, lambda: t["now"])
    src.refresh(NOW)
    t["now"] = EPOCH + 4 * calendar.REFRESH_S
    src.refresh(NOW)
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: NOW)
    src.trigger_refresh = lambda: None
    reg = ToolRegistry()
    reg.register_many(calendar.make_tools(FakeCfg(urls=["u"]), SimpleNamespace(calendar=src)))
    text = reg.call("get_calendar", {"range": "today"}).text
    assert "nothing else" not in text.lower(), text     # no completeness claim
    # "Personal" is X-WR-CALNAME in the fixture: the feed's own name, which
    # is the only name Hunter would recognise ("google-2" is not one)
    assert "I can't reach your Personal calendar, sir" in text, text
    assert "since 9:00 am" in text, text                # when it last answered


def test_worker_thread_refreshes_and_stops(tmp_path, monkeypatch):
    # The worker calls refresh() with now=None, so its window comes from the
    # module's wall-clock seam and NOT from the injected `clock` (which is
    # only the staleness stamp).  Unfrozen, this test rotted on its own: the
    # window is yesterday-midnight .. +15 days, so once the real date passed
    # 2026-08-28 the fixture's 2026-08-27 "Mum's birthday" fell out of it and
    # the assertion below started reading "Nothing on tomorrow, sir."  Freeze
    # the seam the way now_local's docstring says to.
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: NOW)
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
        "Today: 7:00 am Run for 30 minutes, 10:00 am Dentist for an hour at Main "
        "Street Dental, 2:30 pm Standup for 15 minutes, 6:00 pm Gym for an hour; "
        "nothing else.")
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


def test_a_confident_add_parks_its_undo_on_the_source(tmp_path, monkeypatch):
    """The confident path never asks, so it returns a ToolResult and not a
    CommandResult -- and a ToolResult has no undo slot. Without the parked
    entry "scratch that" would have nothing to reach for, which is how an
    add ended up being un-take-back-able in the first place."""
    reg, src = tool_registry(tmp_path, FakeCfg(urls=[URL]), monkeypatch)
    removed = []

    class Saved:
        def delete(self):
            removed.append(1)

    class WritableCal:
        name = "Calendar"

        def get_supported_components(self):
            return ["VEVENT"]

        def save_event(self, ical):
            return Saved()

    monkeypatch.setattr(src, "icloud_calendars", lambda: [WritableCal()])
    r = reg.call("add_event", {"text": "lab presentation on friday at 9 am"})
    assert r.ok and "Added" in r.text
    assert callable(src.last_add["undo"]) and src.last_add["title"]
    assert src.last_add["undo"]() == calendar.UNDONE_LINE
    assert removed == [1]


def test_tool_spec_and_unconfigured_excuse(tmp_path, monkeypatch):
    specs = calendar.make_tools(FakeCfg(), SimpleNamespace())
    specs = sorted(specs, key=lambda sp: sp.name != "get_calendar")
    # add_event joined get_calendar 2026-08-28 (calendar writing)
    assert sorted(s.name for s in specs) == ["add_event", "get_calendar"]
    assert all(len(sp.description.split()) <= 20 for sp in specs)
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


def test_raw_ics_hands_back_the_bodies_it_already_fetched(tmp_path):
    """The Canvas coursework adapter needs a TERM-long view of the feed and
    this cache keeps only ``window_days``; handing back the bytes lets it
    re-parse the SAME response instead of putting a second request on the
    wire (jarvis/tools/canvas_ical.deep_rows)."""
    server = IcsServer()
    src = make_source(tmp_path, fetch=server)
    assert src.raw_ics() == {}, "nothing fetched yet"
    src.refresh()
    assert src.raw_ics() == {URL: ICS}
    # a 304 keeps the body: the conditional GET must not blank it
    src.refresh()
    assert src.raw_ics() == {URL: ICS}
    assert any(h.get("If-None-Match") for _u, h in server.calls)
    # a source reloaded from the disk cache has parsed events but no bodies
    cold = make_source(tmp_path, fetch=server)
    assert cold.events() and cold.raw_ics() == {}


def test_get_calendar_owns_the_bare_day_question():
    """LIVE MISS 2026-08-31: "What's on today?" reached get_briefing instead
    of get_calendar on 3 of 5 tries against gemma4:26b (brainstudy bench,
    case cal_today), so Hunter asked for his events and got the whole
    morning briefing composed at him.

    The tool descriptions are the ENTIRE boundary -- they are all the model
    sees when it picks -- and the two overlapped: get_briefing opened
    "Briefing: today's weather, calendar, ..." while get_calendar opened
    "Hunter's calendar for a weekday name, ...".  So the summary tool owned
    both "today" and "calendar" and the events tool led with a phrasing
    nobody says.  Sharpening the two took the case to 5/5.

    These strings ride in Ollama's cached static prefix, so they are worth
    pinning: the boundary is the fix, not the wording of any one file."""
    from jarvis.tools import briefing as br

    cal = next(sp for sp in calendar.make_tools(FakeCfg(), SimpleNamespace())
               if sp.name == "get_calendar")
    (brief,) = br.make_tools(FakeCfg(), SimpleNamespace())
    cal_d, brief_d = cal.description.lower(), brief.description.lower()

    # The events tool leads with EVENTS, not with "a weekday name".
    assert cal_d.split()[0].startswith("event")
    # The words Hunter actually says belong to the tool that answers them.
    assert "what's on today" in cal_d
    # ...and the summary tool must not claim the day word back.
    assert "today" not in brief_d
    # Still inside the budget register() warns above; these ship on every turn.
    assert cal.description_words() <= 20 and brief.description_words() <= 20


# ------------------------------------------------- 2026-08-31 evening bugs
def test_tomorrow_shorthand_in_a_whole_sentence_is_not_today():
    """LIVE 2026-08-31 20:38, typed: "whats on tmws calendar".

    commander.calendar_range hands coerce_range the WHOLE utterance, and the
    abbreviation branch was an equality test (``text in ("tmrw", "tmr")``),
    so it could only ever fire for a bare word.  The sentence fell through
    to the "today" default, the log shows ``route short-cut:
    get_calendar({'range': 'today'})``, the tool handed the model today's
    four events, and Jarvis answered: "I'm afraid I can't see tomorrow's
    schedule, sir; I only have access to your entries for today." -- about
    a day that is inside the 14-day cache the whole time."""
    for said in ("whats on tmws calendar", "what's on my calendar tmrw",
                 "anything on tmw?", "do i have anything tomorrow",
                 "what's on my calendar tomorrow?", "tmr"):
        assert coerce_range(said) == "tomorrow", said
    # ...and a sentence with no day word still means today, as before.
    assert coerce_range("whats on my calendar") == "today"
    # A weekday still wins over a stray abbreviation-looking token.
    assert coerce_range("what's on Monday") == "monday"


def test_a_fifty_minute_class_reports_its_length():
    """LIVE 2026-08-31 21:00: "How long is my biosensors lab tomorrow?" ->
    "I'm afraid I don't have that information, sir."

    _duration_words returned "" for anything under an hour, so three of his
    four daily classes (all 50-minute lectures -- 9:10-10:00 BIOSENSORS) went
    into the tool text with no length at all and the question was genuinely
    unanswerable.  The duration is derivable from start and end; say it."""
    start = NOW.replace(hour=9, minute=10)
    fifty = calendar.Event(start=start, end=start + timedelta(minutes=50),
                           title="BIOSENSORS")
    assert format_events([fifty], "today", NOW) == \
        "Today: 9:10 am BIOSENSORS for 50 minutes; nothing else."
    quarter = calendar.Event(start=start, end=start + timedelta(minutes=15),
                             title="Standup")
    assert "for 15 minutes" in format_events([quarter], "today", NOW)
    # A zero-length marker is not a booking: still no length.
    marker = calendar.Event(start=start, end=start, title="Rent due")
    assert format_events([marker], "today", NOW) == \
        "Today: 9:10 am Rent due; nothing else."
    # The hour-and-over wording is untouched.
    two = calendar.Event(start=start, end=start + timedelta(minutes=110),
                         title="BIOSENSORS")
    assert "for about 2 hours" in format_events([two], "today", NOW)


def test_get_calendar_description_offers_duration():
    """The model picks the tool from its description alone.  Nothing in the
    tool list mentioned how long anything lasts, so "how long is my
    biosensors lab tomorrow?" was routed local:question and answered from
    nothing rather than from get_calendar."""
    cal = next(sp for sp in calendar.make_tools(FakeCfg(), SimpleNamespace())
               if sp.name == "get_calendar")
    assert "how long" in cal.description.lower()
    assert "what's on today" in cal.description.lower()   # still wins the day word
    assert cal.description_words() <= 20
