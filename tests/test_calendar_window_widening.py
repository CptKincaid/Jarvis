"""Widening the calendar's reach without a flood of "you have a new booking".

WINDOW_DAYS was 14, so "what about october 3rd" -- a day he genuinely has
-- landed on the right day and was then honestly refused for reach.  The
widening is cheap -- MEASURED on this file's own two feeds, a refresh goes
8.4 ms -> 9.7 ms and the cache file 5.1 KB -> 15.6 KB (28 events -> 90),
once every ten minutes.  The DANGER is jarvis/calwatch.py, which diffs two
snapshots: the
first refresh after the window grows sees thirty-one days of events that
were never in the previous snapshot, and every one of them is shaped
exactly like a new booking.

MEASURED on this file's own harness, at an injected Saturday 2026-09-05,
against two invented feeds of one event a day for sixty days:

    scenario                          before the fix   after the fix
    both feeds answer the wide fetch        0                0
    one feed misses the FIRST wide fetch   31                0

The zero on the left is the window-intersection already in ``diff_events``
(``overlap`` of the two snapshots' windows).  The 31 on the right of it is
the hole that intersection does not cover: ``CalendarSource.window()``
returned the window the source PROMISES, while a source in back-off still
holds the events it fetched under the OLD one.  One tick later both
snapshots carry the wide window, the recovered feed's far events arrive
all at once, and the diff has no way to know they are not new.

THE FIX: the source now reports the window its cached events were actually
GATHERED over (``covered_window``) -- the intersection of the promise with
every source's own last fetch window -- and calwatch diffs over that.  A
feed that has not caught up holds the diff back instead of exploding it.

THE CLOCK: every test here injects ``now`` (``now_local`` is monkeypatched
and CalendarWatch takes its own ``now``).  Nothing reads the wall clock.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.calendar as calendar
from jarvis.calwatch import ADDED, CalendarWatch

CHI = ZoneInfo("America/Chicago")
NOW = datetime(2026, 9, 5, 14, 0, tzinfo=CHI)      # Saturday
FEEDS = ["https://example.invalid/a.ics", "https://example.invalid/b.ics"]
SPAN = 60                                          # invented days per feed


class _Cfg:
    def __init__(self, urls):
        self.data = {"google_ical_urls": list(urls),
                     "icloud": {"apple_id": "", "app_password": "", "url": ""}}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _ics(tag: str, hour: int) -> bytes:
    """One invented event a day for SPAN days.  Nothing here is his."""
    out = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//test//EN",
           f"X-WR-CALNAME:{tag}"]
    for i in range(SPAN):
        day = NOW.date() + timedelta(days=i)
        out += ["BEGIN:VEVENT", f"UID:{tag}-{i}@invalid",
                f"DTSTART;TZID=America/Chicago:{day:%Y%m%d}T{hour:02d}0000",
                f"DTEND;TZID=America/Chicago:{day:%Y%m%d}T{hour:02d}5000",
                f"SUMMARY:{tag} day {i}", "END:VEVENT"]
    out.append("END:VCALENDAR")
    return ("\r\n".join(out) + "\r\n").encode()


class _Wire:
    """Both feeds, with one of them switchable to "not answering"."""

    def __init__(self):
        self.bodies = {FEEDS[0]: _ics("ALPHA", 9), FEEDS[1]: _ics("BETA", 13)}
        self.down: set = set()
        self.calls = 0

    def __call__(self, url, timeout=None, headers=None):
        self.calls += 1
        if url in self.down:
            raise TimeoutError("feed not answering")
        return self.bodies[url]


def _source(tmp_path, wire, window_days, clock):
    return calendar.CalendarSource(
        _Cfg(FEEDS), cache_path=tmp_path / "calendar_cache.json",
        fetch=wire, tz=CHI, window_days=window_days, clock=clock)


def _watch(tmp_path, source):
    return CalendarWatch(lambda: source, state_path=tmp_path / "calwatch.json",
                         now=lambda: NOW)


def _added(changes) -> int:
    return sum(1 for c in changes if c.kind == ADDED)


@pytest.fixture()
def frozen(monkeypatch):
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: NOW)
    return NOW


def _run(tmp_path, frozen, *, feed_b_misses_first_wide_fetch: bool) -> int:
    """The whole widening, end to end.  Returns the ADDED count the first
    wide refresh and the one after it would emit between them."""
    ticks = [1000.0]

    def clock():
        return ticks[0]

    wire = _Wire()
    # --- life before the widening: two ticks at WINDOW_DAYS = 14 ---------
    narrow = _source(tmp_path, wire, 14, clock)
    watch = _watch(tmp_path, narrow)
    watch.tick()                                   # learns, says nothing
    ticks[0] += 600
    assert watch.tick() == []                      # steady state

    # --- the restart that widens the window ------------------------------
    ticks[0] += 600
    if feed_b_misses_first_wide_fetch:
        wire.down = {FEEDS[1]}
    wide = _source(tmp_path, wire, 45, clock)
    watch2 = _watch(tmp_path, wide)
    burst = _added(watch2.tick())
    # ...and the tick after it, once the slow feed comes back.
    wire.down = set()
    ticks[0] += 3600                               # past any back-off
    burst += _added(watch2.tick())
    ticks[0] += 600
    burst += _added(watch2.tick())
    return burst


def test_widening_announces_nothing_when_both_feeds_answer(tmp_path, frozen):
    assert _run(tmp_path, frozen, feed_b_misses_first_wide_fetch=False) == 0


def test_widening_announces_nothing_when_a_feed_misses_the_first_fetch(tmp_path, frozen):
    """THE MEASURED HOLE: 31 "added" lines before the fix, 0 after."""
    assert _run(tmp_path, frozen, feed_b_misses_first_wide_fetch=True) == 0


def test_the_covered_window_is_the_events_reach_not_the_promise(tmp_path, frozen):
    """The mechanism, on its own.  ``window()`` says what the NEXT fetch will
    ask for; ``covered_window()`` says what the events in hand actually
    cover, and while a feed is behind those are different windows."""
    ticks = [1000.0]
    wire = _Wire()
    narrow = _source(tmp_path, wire, 14, lambda: ticks[0])
    assert narrow.refresh() is True
    assert narrow.covered_window() == narrow.window()

    ticks[0] += 600
    wide = _source(tmp_path, wire, 45, lambda: ticks[0])
    wire.down = {FEEDS[1]}
    assert wide.refresh() is True                  # ALPHA answered; BETA did not
    promise, covered = wide.window(), wide.covered_window()
    assert covered is not None
    assert covered[1] < promise[1]                 # held back to BETA's reach
    assert covered[1] == narrow.window()[1]

    wire.down = set()
    ticks[0] += 3600
    assert wide.refresh() is True
    assert wide.covered_window() == wide.window()  # both feeds caught up


def test_a_source_that_never_said_its_reach_costs_one_silent_cycle(tmp_path, frozen):
    """A cache written before 2026-09-05 carries no per-source window.  That
    is "unknown", not "the promise": calwatch must learn and say nothing."""
    ticks = [1000.0]
    source = _source(tmp_path, _Wire(), 45, lambda: ticks[0])
    source._sources = {"google-1": {"fetched_at": ticks[0], "events": []}}
    assert source.covered_window() is None
    watch = _watch(tmp_path, source)
    assert watch.tick() == []


def test_the_window_survives_the_cache_file(tmp_path, frozen):
    """It has to: the first refresh after a restart is where the widening
    lands, and a source that fails there keeps only what the file holds."""
    ticks = [1000.0]
    wire = _Wire()
    first = _source(tmp_path, wire, 14, lambda: ticks[0])
    first.refresh()
    again = _source(tmp_path, wire, 14, lambda: ticks[0])
    assert again.covered_window() == first.covered_window()
