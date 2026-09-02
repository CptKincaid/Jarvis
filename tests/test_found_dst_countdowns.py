"""Regression tests for the DST family that survived the mail/location sweep.

DEFECT (three sites, all reached from the same shape of ``now``):

Every production caller on these paths builds ``now`` as
``datetime.now().astimezone()`` -- build_briefing (briefing.py:669),
commander._h_next_exam (commander.py:3454) and the syllabus scan rung
(commander.py:5643). Its tzinfo is a FIXED-offset SNAPSHOT of today's
offset, ``timezone(-5h, 'CDT')`` in summer, not a zone. Handed to
``astimezone(now.tzinfo)`` it converts every instant with today's offset
instead of the one that actually applied.

  1. jarvis/tools/canvas.py countdown_words -- measured with now =
     2026-10-30 07:05 CDT and a Canvas exam at 2026-11-02T05:59Z
     (2026-11-01 23:59 CST):

       exam_words -> 'Midterm 1 for BIOSENSORS, in 3 days, Monday at 12:59 am'
       truth      ->                            in 2 days, Sunday at 11:59 pm

     Wrong count, wrong weekday, wrong clock time -- spoken by the morning
     briefing (briefing.py sections["exam"]), by "when's my next exam", and
     by the syllabus read-back (syllabus.py:185).

  2. jarvis/tools/briefing.py _exam_days -- the same conversion, so the
     count that gates the study section is a day out. With now =
     2026-10-27 07:05 CDT and the same exam (5 whole days away,
     briefing.study_days = 5) it returned 6, ``days > near`` fired, and the
     deck line AND "Shall we run four now, sir?" were suppressed entirely
     on the morning they should first be spoken.

  3. jarvis/syllabus.py _local -- stamps the NAIVE ``datetime.combine(day,
     clock)`` from the model's rows, so the stored INSTANT is wrong, not
     just its label. A scan dated 2026-10-02 CDT with a row
     "2026-11-15T09:00" stored 2026-11-15 09:00-05:00 = 08:00 local while
     the read-back still said "9:00 am": he approves nine and the reminder
     fires at eight, which is the one thing the read-back rung exists to
     prevent.

Sites 1 and 2 were live on the path the quiz-clock lane reviewed;
canvas.when_words, 300 lines above countdown_words in the same file,
already carried the guard. All three now share it (canvas.in_local).

FOURTH SITE, jarvis/tools/location.py system_tz(): it resolves
``os.environ["TZ"]`` as a ZoneInfo key without calling time.tzset(), so a
process that changed TZ after start would get a zone it is not running on
-- and bare "UTC"/"CST6CDT"/"EST5EDT" ARE real tzdata keys, so they
resolve rather than falling through (measured: TZ=EST5EDT gave
ZoneInfo('EST5EDT') while datetime.now().astimezone() in the same process
still read -05:00). The key is now trusted only while it agrees with the
offset the interpreter itself is using.

THE ZONE IS PINNED WITH time.tzset(), not just with a tzinfo stamp: the
fix is the NO-ARGUMENT astimezone(), which reads the C library's zone, so
a test that only labelled its datetimes would pass here and prove nothing
on a machine in another zone.
"""
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

import jarvis.syllabus as syl
from jarvis.tools import canvas as cv
from jarvis.tools import location
from jarvis.tools.briefing import _exam_days, _study_section
from jarvis.tools.quiz import FlashcardStore

# 2026-11-02T05:59Z is 2026-11-01 23:59 CST -- the Sunday night of the
# fall-back, the shape _parse_iso hands back for a Canvas due date.
EXAM_UTC = datetime(2026, 11, 2, 5, 59, tzinfo=timezone.utc)


@pytest.fixture
def chicago():
    """The PROCESS on America/Chicago, restored afterwards."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "America/Chicago"
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def _now(*args) -> datetime:
    """``now`` exactly as production builds it: the naive wall time through
    a bare astimezone(), whose tzinfo is the fixed-offset snapshot."""
    got = datetime(*args).astimezone()
    assert isinstance(got.tzinfo, timezone), "the premise of all of this"
    return got


class Cfg:
    def __init__(self, **briefing):
        self.data = {"briefing": briefing}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def test_a_countdown_over_the_fall_back_keeps_the_day_and_the_hour(chicago):
    now = _now(2026, 10, 30, 7, 5)                    # Friday, still CDT
    cand = {"course": "BIOSENSORS", "title": "Midterm 1", "when": EXAM_UTC}
    assert cv.countdown_words(EXAM_UTC, now) == "in 2 days, Sunday at 11:59 pm"
    assert cv.exam_words(cand, now) == \
        "Midterm 1 for BIOSENSORS, in 2 days, Sunday at 11:59 pm"
    # the sibling that always had the guard, as the control
    assert cv.when_words(EXAM_UTC, now) == "Sun 11:59 pm"
    assert _exam_days(cand, now) == 2


def test_an_explicit_zone_is_still_honoured_over_the_system_one(chicago):
    from zoneinfo import ZoneInfo
    now = datetime(2026, 10, 30, 7, 5, tzinfo=ZoneInfo("Australia/Sydney"))
    assert cv.countdown_words(EXAM_UTC, now).endswith("at 4:59 pm")


def test_the_study_offer_is_made_on_the_morning_it_becomes_due(chicago, tmp_path):
    """study_days = 5 and the exam is 5 whole days out, so the deck line is
    due TODAY; counting it as 6 dropped it silently."""
    now = _now(2026, 10, 27, 7, 5)
    store = FlashcardStore(tmp_path / "flashcards.db")
    try:
        store.add_cards([{"question": f"q{i}?", "answer": f"a{i}"} for i in range(4)],
                        source="biosensors-notes.md", topic="BIOSENSORS",
                        now=now.timestamp() - 1)
        exam = {"course": "BIOSENSORS", "title": "Midterm 1", "when": EXAM_UTC,
                "kind": "exam", "all_day": False}
        assert _exam_days(exam, now) == 5
        line, offer = _study_section(Cfg(study_days=5), exam, store, now)
        assert line == ("4 cards due on your BIOSENSORS deck, 4 of them in box "
                        "one. Shall we run four now, sir?")
        assert offer["course"] == "BIOSENSORS" and offer["n"] == 4
    finally:
        store.close()


def test_the_syllabus_reads_back_the_time_it_actually_stored(chicago):
    """He says yes to what he HEARS; the stored instant has to be it."""
    now = _now(2026, 10, 2, 9, 0)                     # scanned in CDT
    rows = syl.parse_rows(
        [{"title": "Midterm 1", "course": "CS 101", "due": "2026-11-15T09:00"}], now)
    assert len(rows) == 1
    assert rows[0]["due"] == datetime(2026, 11, 15, 9, 0).astimezone()
    assert rows[0]["due"].utcoffset() == timedelta(hours=-6)      # CST, not CDT
    assert "Sun 15 Nov at 9:00 am" in syl.read_back_line(rows, now)


def test_a_stored_row_survives_the_trip_back_out_of_the_store(chicago, tmp_path):
    """stored_items() re-localises an already-aware row; converting it with
    the snapshot moved the wall clock an hour every winter."""
    now = _now(2026, 10, 2, 9, 0)
    due = datetime(2026, 11, 15, 9, 0).astimezone()
    syl.save([{"title": "Midterm 1", "course": "CS 101", "due": due,
               "all_day": False}], tmp_path / "s.json")
    items = syl.stored_items(now, tmp_path / "s.json")
    assert [i["due"] for i in items] == [due]
    assert items[0]["due"].hour == 9


def test_system_tz_refuses_a_key_the_process_is_not_running_on(chicago):
    """TZ is read without time.tzset(), and EST5EDT resolves as a real
    tzdata key -- so the guard is the offset check, not the key lookup."""
    assert getattr(location.system_tz(), "key", "") == "America/Chicago"
    os.environ["TZ"] = "EST5EDT"                     # deliberately no tzset()
    zone = location.system_tz()
    assert getattr(zone, "key", "") != "EST5EDT"
    local = datetime.now()
    assert zone.utcoffset(local) == local.astimezone().utcoffset()
