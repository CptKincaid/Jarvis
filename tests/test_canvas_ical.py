"""Canvas coursework read out of the calendar FEED (jarvis/tools/canvas_ical.py).

His university blocks personal Canvas access tokens, so the REST path can
never answer on this box; the same coursework rides the Canvas calendar
subscription as VEVENTs whose SUMMARY ends in a bracket of course codes.
These are his REAL title shapes, copied out of the live feed on 31 Aug
2026.

Firewall: no network anywhere (urlopen is booby-trapped), the coursework
cache is redirected to tmp, JARVIS_LOG_DIR is the conftest temp one.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from jarvis.tools import canvas_ical as ci

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 8, 31, 9, 0, tzinfo=TZ)              # Monday morning

# Verbatim from the feed (34 events, all of this shape).
LAB1 = ("Lab 1:  Introduction to the AD2 SDK [BMEN-427:501,502,503,504,"
        "BMEN-627:600,601,602,603,ECEN-463:501,503,504,ECEN-763:601,603,604]")
HW1 = "HW#1 [MSEN-222:599,M99]"
PRES = "Update Presentation #1 [ECEN-404:901,902,903]"
PRELAB = ("Prelab for Lab  3  (canvas quiz) [BMEN-427:501,502,503,504,"
          "BMEN-627:600,601,602,603]")
ETHICS = "Ethics Quiz - Professional Engineering [ECEN-404:901,902,903]"


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    # Per-test coursework cache: deep_rows mirrors its parse to disk, and a
    # file shared across tests would leak rows from one into the next.
    monkeypatch.setattr(ci, "_cache_path",
                        lambda path=None: Path(path) if path else tmp_path / "cw.json")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    ci.clear_cache()
    yield
    ci.clear_cache()


def _ev(title, start, all_day=False, calendar="Hunter Peyrovi Calendar (Canvas)"):
    return SimpleNamespace(title=title, start=start, all_day=all_day,
                           calendar=calendar)


def _day(offset):
    """Local midnight ``offset`` days from NOW -- how an all-day VEVENT
    lands in the calendar cache."""
    return (NOW + timedelta(days=offset)).replace(hour=0, minute=0, second=0,
                                                  microsecond=0)


# ------------------------------------------------------------- the tell
def test_the_trailing_bracket_of_course_codes_is_the_tell():
    assert ci.split_coursework(LAB1) == \
        ("BMEN 427", "Lab 1: Introduction to the AD2 SDK")
    assert ci.split_coursework(HW1) == ("MSEN 222", "HW#1")
    assert ci.split_coursework(PRES) == ("ECEN 404", "Update Presentation #1")
    # the double space Canvas leaves after the colon is collapsed
    assert ci.split_coursework(PRELAB)[1] == "Prelab for Lab 3 (canvas quiz)"
    # cross-listed sections: every code is read, the FIRST names the row
    assert ci.course_codes("BMEN-427:501,502,BMEN-627:600,ECEN-463:501") == \
        ["BMEN 427", "BMEN 627", "ECEN 463"]
    # other Canvas installs write the code without the dash
    assert ci.split_coursework("Reading quiz [ENGL 210:001]")[0] == "ENGL 210"
    assert ci.split_coursework("Reading quiz [ENGL210]")[0] == "ENGL 210"


def test_titles_that_are_not_coursework():
    for title in ("BIOSENSORS",                      # his Navigate360 class row
                  "ELECTRICAL DESIGN LAB II- Presentation",
                  "Dinner with Sam [tentative]",     # a bracket, no course code
                  "Lecture [Section 101]",           # "Section" is not a dept
                  "Chiro", "", None,
                  "HW#1 [MSEN-222:599,M99] and more",  # bracket not at the end
                  "[BMEN-427:501]"):                 # a bracket and nothing else
        assert ci.split_coursework(title) is None, title
        assert not ci.is_coursework(title), title


def test_other_events_and_has_coursework():
    events = [_ev(LAB1, _day(5), all_day=True), _ev("Chiro", NOW + timedelta(hours=6))]
    assert ci.has_coursework(events) is True
    assert [e.title for e in ci.other_events(events)] == ["Chiro"]
    assert ci.has_coursework([_ev("Chiro", NOW)]) is False
    assert ci.has_coursework([]) is False and ci.has_coursework(None) is False


# ---------------------------------------------------------------- rows
def test_an_all_day_assignment_is_due_at_11_59_pm_local():
    """Canvas's own meaning. Midnight would move every deadline a whole day
    early -- the heads-up would fire on the wrong evening."""
    row = ci.row_from_event(_ev(HW1, _day(6), all_day=True))
    assert row["course"] == "MSEN 222" and row["title"] == "HW#1"
    assert row["due"] == _day(6).replace(hour=23, minute=59)
    assert row["due"].strftime("%H:%M %Z") == "23:59 CDT"


def test_a_timed_assignment_keeps_its_own_time():
    """The canvas-quiz prelabs carry DTSTART with a UTC time; 17:40Z is
    12:40 pm in his zone and must stay there."""
    when = datetime(2026, 9, 14, 17, 40, tzinfo=timezone.utc)
    row = ci.row_from_event(_ev(PRELAB, when))
    assert row["due"] == when
    assert row["due"].astimezone(TZ).strftime("%H:%M") == "12:40"


def test_the_11_59_pm_rule_survives_the_dst_change():
    """A November all-day assignment is 11:59 pm CST, not 10:59 pm: the
    replace() keeps the zone and the offset is resolved at render time."""
    nov = datetime(2026, 11, 4, 0, 0, tzinfo=TZ)          # after the change
    row = ci.row_from_event(_ev(PRES, nov, all_day=True))
    assert row["due"].strftime("%H:%M %Z") == "23:59 CST"
    assert row["due"].utcoffset() == timedelta(hours=-6)


def test_rows_from_events_windows_drops_the_past_and_sorts():
    events = [_ev(PRES, _day(9), all_day=True),
              _ev(HW1, _day(6), all_day=True),
              _ev(LAB1, _day(5), all_day=True),
              _ev("Handed in yesterday [MSEN-222:599]", _day(-1), all_day=True),
              _ev("Chiro", _day(2))]
    rows = ci.rows_from_events(events, 7, NOW)
    assert [(r["course"], r["title"]) for r in rows] == [
        ("BMEN 427", "Lab 1: Introduction to the AD2 SDK"),
        ("MSEN 222", "HW#1")]
    assert [r["title"] for r in ci.rows_from_events(events, 30, NOW)] == [
        "Lab 1: Introduction to the AD2 SDK", "HW#1", "Update Presentation #1"]
    assert ci.rows_from_events(events, None, NOW)[0]["due"] > NOW
    assert ci.rows_from_events([], 7, NOW) == []


def test_an_hour_past_its_due_time_a_row_is_gone():
    """The feed cannot say whether the work went in, so the due time
    passing is the only 'done' signal there is -- the same rule
    canvas.fetch_due applies to the planner."""
    events = [_ev(LAB1, _day(0), all_day=True)]
    assert len(ci.rows_from_events(events, 7, NOW)) == 1        # 9 am, due 11:59 pm
    late = NOW.replace(hour=23, minute=59) + timedelta(minutes=1)
    assert ci.rows_from_events(events, 7, late) == []


def test_merge_rows_lets_the_rest_reading_win_a_duplicate():
    """The planner knows the course's real NAME and whether the work was
    handed in; the feed knows neither, so REST goes first. The key is
    title-and-day because the feed's 11:59 pm is a convention, not a
    timestamp Canvas published."""
    rest = [{"course": "BIOSENSORS", "title": "HW#1",
             "due": _day(6).replace(hour=23, minute=59, second=30)}]
    feed = [{"course": "MSEN 222", "title": "HW#1", "due": _day(6).replace(hour=23, minute=59)},
            {"course": "ECEN 404", "title": "Update Presentation #1", "due": _day(9)}]
    merged = ci.merge_rows(rest, feed)
    assert [(r["course"], r["title"]) for r in merged] == [
        ("BIOSENSORS", "HW#1"), ("ECEN 404", "Update Presentation #1")]
    # junk rows are dropped rather than raising
    assert ci.merge_rows([None, {}, {"title": "x"}, "nope"], []) == []


# ------------------------------------------------------- the deep window
FEED_ICS = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:icalendar-ruby
X-WR-CALNAME:Hunter Peyrovi Calendar (Canvas)
BEGIN:VEVENT
UID:event-assignment-3211520
DTSTART;VALUE=DATE:20260905
SUMMARY:Lab 1:  Introduction to the AD2 SDK [BMEN-427:501\\,502\\,503\\,504\\,B
 MEN-627:600\\,601]
END:VEVENT
BEGIN:VEVENT
UID:event-assignment-3211554
DTSTART:20260914T174000Z
DTEND:20260914T174000Z
SUMMARY:Prelab for Lab  3  (canvas quiz) [BMEN-427:501\\,502]
END:VEVENT
BEGIN:VEVENT
UID:event-assignment-3121216
DTSTART;VALUE=DATE:20261118
SUMMARY:Final Presentation [ECEN-404:901\\,902\\,903]
END:VEVENT
BEGIN:VEVENT
UID:event-calendar-event-1
DTSTART:20260901T210000Z
DTEND:20260901T220000Z
SUMMARY:Office hours
END:VEVENT
END:VCALENDAR
""".replace("\n", "\r\n").encode()

PERSONAL_ICS = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:personal-1
DTSTART:20260901T210000Z
DTEND:20260901T220000Z
SUMMARY:Chiro
END:VEVENT
END:VCALENDAR
""".replace("\n", "\r\n").encode()


class _Source:
    """A CalendarSource stand-in: cached events plus the raw bodies it
    already fetched (calendar.CalendarSource.raw_ics)."""

    def __init__(self, events=(), raws=None):
        self._events = list(events)
        self._raws = dict(raws or {})
        self.raw_calls = 0

    configured = True

    def events(self):
        return list(self._events)

    def raw_ics(self):
        self.raw_calls += 1
        return dict(self._raws)


def test_looks_like_coursework_feed_sees_through_line_folding():
    assert ci.looks_like_coursework_feed(FEED_ICS)
    assert not ci.looks_like_coursework_feed(PERSONAL_ICS)
    folded = b"SUMMARY:HW#1 [MSEN-\r\n 222:599]\r\n"
    assert ci.looks_like_coursework_feed(folded)


def test_deep_rows_reach_past_the_calendar_cache_without_a_request():
    """The calendar cache keeps a fortnight; a final in November is 79 days
    out. deep_rows re-parses the bytes the calendar already fetched."""
    src = _Source(raws={"https://canvas.tamu.edu/feeds/x.ics": FEED_ICS,
                        "https://calendar.google.com/y.ics": PERSONAL_ICS})
    rows = ci.deep_rows(src, now=NOW)
    assert [(r["course"], r["title"]) for r in rows] == [
        ("BMEN 427", "Lab 1: Introduction to the AD2 SDK"),
        ("BMEN 427", "Prelab for Lab 3 (canvas quiz)"),
        ("ECEN 404", "Final Presentation")]
    # "Office hours" has no bracket, so it is not coursework; the personal
    # feed never went near the parser at all
    assert all("Office" not in r["title"] for r in rows)
    assert rows[-1]["due"] == datetime(2026, 11, 18, 23, 59, tzinfo=TZ)


def test_deep_rows_parse_is_memoised_per_response(monkeypatch):
    src = _Source(raws={"a.ics": FEED_ICS})
    calls = []
    real = ci._parse_window
    monkeypatch.setattr(ci, "_parse_window",
                        lambda raw, s, e, tz: calls.append(1) or real(raw, s, e, tz))
    first = ci.deep_rows(src, now=NOW)
    again = ci.deep_rows(src, now=NOW + timedelta(hours=2))
    assert len(calls) == 1, "a spoken turn must not re-parse the feed"
    assert [r["title"] for r in first] == [r["title"] for r in again]
    # a NEW body is a new key
    ci.deep_rows(_Source(raws={"a.ics": FEED_ICS.replace(b"Final", b"FINAL")}), now=NOW)
    assert len(calls) == 2


def test_deep_rows_fall_back_to_the_disk_cache_before_the_first_refresh(tmp_path):
    """A just-started process has no raw yet (the calendar cache holds
    parsed events, not bodies), so the last deep parse is mirrored to
    disk and read back."""
    path = tmp_path / "coursework.json"
    ci.deep_rows(_Source(raws={"a.ics": FEED_ICS}), now=NOW, cache_path=path)
    saved = json.loads(path.read_text())
    assert saved["version"] == ci.CACHE_VERSION and len(saved["rows"]) == 3
    cold = _Source()                     # no raw_ics content at all
    rows = ci.deep_rows(cold, now=NOW, cache_path=path)
    assert [r["title"] for r in rows][-1] == "Final Presentation"
    # rows decay by their own due time: an abandoned cache goes to nothing
    assert ci.deep_rows(cold, now=datetime(2027, 1, 1, tzinfo=TZ),
                        cache_path=path) == []


def test_deep_rows_never_raise_and_never_invent(tmp_path):
    path = tmp_path / "coursework.json"
    # no calendar at all / a source that throws
    assert ci.deep_rows(None, now=NOW, cache_path=path) == []
    assert ci.deep_rows(SimpleNamespace(raw_ics=lambda: 1 / 0), now=NOW,
                        cache_path=path) == []
    # a source with feeds but no coursework among them is a real "no"
    assert ci.deep_rows(_Source(raws={"a.ics": PERSONAL_ICS}), now=NOW,
                        cache_path=path) == []
    # an unparsable body keeps whatever was last written rather than
    # overwriting it with the failure
    ci.deep_rows(_Source(raws={"a.ics": FEED_ICS}), now=NOW, cache_path=path)
    ci.clear_cache()
    broken = b"BEGIN:VCALENDAR\r\nSUMMARY:HW#1 [MSEN-222:599]\r\nnot an ics"
    assert len(ci.deep_rows(_Source(raws={"b.ics": broken}), now=NOW,
                            cache_path=path)) == 3
    # a corrupt cache reads as empty
    path.write_text("{not json")
    assert ci.deep_rows(_Source(), now=NOW, cache_path=path) == []
