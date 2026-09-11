"""ROUND THREE of the 2026-09-05 calendar-date bug: THE FRAME RULE.

Round one taught ``coerce_range`` to read a named date.  Round two measured
it on a 180-phrasing grid at an injected Saturday 2026-09-05 14:00
America/Chicago and closed the ordinal-as-rank hole ("the 2nd lab"), the
eaten year ("12 september 2027"), the dropped offset ("the day after the
12th"), word ordinals ("september twelfth") and the two doors' quarrel over
"yesterday".  Measured again at the same instant, on the same grid, THIS
round found what was still open -- printed exactly as it came back:

    "i got 9/10 on the quiz"              -> 2026-09-10    (a mark, not a day)
    "the score was 3/5"                   -> 2026-03-05
    "i need a 3/4 cup"                    -> 2027-03-04
    "i scored 12/20"                      -> 2026-12-20
    "meet me at 9/10 sharp"               -> 2026-09-10
    "my ratio is 9/12"                    -> 2026-09-12
    "is it open 24/7"                     -> "did you mean the 24th of July?"
    "is my 9/10 quiz on my calendar"      -> Thursday the 10th, through BOTH doors
    "what is on my calendar in december 2026"
                                          -> forced door ASKS, model door says TODAY
    "what do i have in december"          -> today, in silence
    "what do i have in two days"          -> today, in silence
    "what do i have a week on tuesday"    -> tuesday    (the 8th; he meant the 15th)
    "the last friday of september"        -> friday     (the 11th; he meant the 25th)
    "the 12th to the 14th"                -> the 12th, and not a word about the 14th
    "tuesday the 8th" + "and the next day"
                                          -> "Wednesday the 8th" -> a question about a
                                             weekday he never said (the follow-up
                                             that worked before round one)

THE RULE this file pins, in his register:

    Jarvis reads a date only when the sentence is about WHEN -- a calendar
    word or a preposition stands in front of the number.  A number nobody
    framed as a date is not a date.  A thing that looks like a date and
    cannot be read is a question back to him, never today.  And the same
    eight words mean the same day through every door.

Three defaults taken for him are pinned here and flagged in the report:
a dashed or dotted pair ("9-12", "9.12") ASKS rather than assumes; "next
Monday" keeps its old meaning (the coming Monday); WINDOW_DAYS is 45 only
because calwatch's first-refresh burst is suppressed
(tests/test_calendar_window_widening.py).

THE CLOCK: every test injects ``now``; nothing here reads the wall clock,
and the tables are re-run at other instants at the end.
"""
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.calendar as calendar
from jarvis.tools.calendar import (Event, ask_words, as_date, coerce_range,
                                   is_ask, make_tools)
from jarvis.tools.registry import ToolRegistry

CHI = ZoneInfo("America/Chicago")
NOW = datetime(2026, 9, 5, 14, 0, tzinfo=CHI)          # the grid's Saturday
TODAY = NOW.date()


def _not_a_day(rng, said):
    assert as_date(rng) is None, f"{said!r} -> {rng!r}"
    assert not is_ask(rng), f"{said!r} -> {rng!r}"


def _asks(rng, said):
    assert is_ask(rng), f"{said!r} -> {rng!r}"
    assert ask_words(rng).endswith("?"), rng
    assert rng != "today"


# ============================================ (1b) a number is not a date
#
# ``_D_NUM_RX`` took a bare N/N ANYWHERE in the sentence with nothing in
# front of it.  Nine of the grid's non-date rows became dates or questions
# that way, and "is my 9/10 quiz on my calendar" listed Thursday the 10th
# through both doors.  A slashed pair is a date only when a date word frames
# it: a preposition ("on 9/12", "for 9/12", "the week of 9/12", "due by
# 9/12") or a calendar word ("what's on my calendar 9/12", "am i free
# 9/12") stands right in front, and what follows is not the noun of a rank.
UNFRAMED_PAIRS = [
    "i got 9/10 on the quiz",
    "the score was 3/5",
    "i need a 3/4 cup",
    "is it open 24/7",
    "it is a 50/50 chance",
    "i scored 12/20",
    "set it to 3/4 speed",
    "meet me at 9/10 sharp",
    "my ratio is 9/12",
    "is my 9/10 quiz on my calendar",
    "is my gym on my calendar 24/7",
    "a 9/10 quiz",
    "the 3/4 mark",
    "what's 9/12 as a decimal",
    "is my 9/10 quiz graded",
    "on 9/12 speed",                    # framed, but a rank: the tail rule
]


@pytest.mark.parametrize("said", UNFRAMED_PAIRS)
def test_a_slashed_pair_nobody_framed_as_a_date_is_not_one(said):
    _not_a_day(coerce_range(said, NOW), said)


FRAMED_PAIRS = [
    ("what do i have on 9/12", "2026-09-12"),
    ("anything for 9/12", "2026-09-12"),
    ("what about 9/12", "2026-09-12"),
    ("the week of 9/12", "2026-09-12"),
    ("what's due by 9/12", "2026-09-12"),
    ("what's on my calendar 9/12", "2026-09-12"),
    ("what do i have 9/12", "2026-09-12"),
    ("am i free 9/12", "2026-09-12"),
    ("what do i have on 9/12 at 3", "2026-09-12"),
    ("anything on 9/12 please", "2026-09-12"),
    ("what do i have on 9/12/27", "2027-09-12"),
    # the whole value, the way the model sends ``range``
    ("9/12", "2026-09-12"),
    ("12th", "2026-09-12"),
    ("the 12th", "2026-09-12"),
    ("september 12", "2026-09-12"),
    ("saturday the 12th", "2026-09-12"),
]


@pytest.mark.parametrize("said,want", FRAMED_PAIRS)
def test_a_slashed_pair_a_date_word_frames_is_a_date(said, want):
    assert coerce_range(said, NOW) == want, said


@pytest.mark.parametrize("said", [
    "what do i have on 9/31", "anything on 13/5", "what about 24/7",
    "what do i have on 0/5",
])
def test_a_framed_pair_that_cannot_be_a_date_still_asks(said):
    _asks(coerce_range(said, NOW), said)


# ------------------------------------------------ end to end, both doors
class _Cfg:
    def __init__(self):
        self.data = {"google_ical_urls": ["https://example.invalid/secret.ics"],
                     "icloud": {"apple_id": "", "app_password": "", "url": ""}}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _ev(start, title, minutes=60):
    return Event(start=start, end=start + timedelta(minutes=minutes), title=title)


EVENTS = [
    _ev(datetime(2026, 9, 5, 16, 0, tzinfo=CHI), "TODAY-EVENT"),
    _ev(datetime(2026, 9, 10, 9, 0, tzinfo=CHI), "TENTH-EVENT"),
    _ev(datetime(2026, 9, 12, 9, 10, tzinfo=CHI), "TWELFTH-EVENT"),
    _ev(datetime(2026, 10, 2, 9, 0, tzinfo=CHI), "OCTOBER-SECOND-EVENT"),
]


@pytest.fixture
def reg(tmp_path, monkeypatch):
    """The real ToolSpec over INVENTED events: no network, no real calendar,
    no wall clock (now_local is frozen and the source takes its clock)."""
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: NOW)
    services = SimpleNamespace()
    src = calendar.make_source(_Cfg(), services,
                               cache_path=tmp_path / "calendar_cache.json",
                               tz=CHI, fetch=lambda *a, **k: b"",
                               clock=lambda: NOW.timestamp())
    src._sources = {"google-1": {"fetched_at": NOW.timestamp(),
                                 "events": list(EVENTS)}}
    r = ToolRegistry()
    r.register_many(make_tools(_Cfg(), services))
    return r


def _both_doors(reg, said):
    from jarvis.commander import forced_call

    forced = forced_call("local:calendar", said, NOW)
    assert forced is not None, f"{said!r} did not force"
    forced_text = reg.call(forced[0], forced[1]).text
    # The model door: the model sends the old wrong value and the deriver
    # reads HIS words in its place.
    model_text = reg.call("get_calendar", {"range": "today"}, from_model=True,
                          utterance=said).text
    print(f"\n  HE SAYS : {said!r}\n  FORCED  : range={forced[1]['range']!r}"
          f"\n            -> {forced_text}\n  MODEL   : -> {model_text}")
    return forced[1]["range"], forced_text, model_text


@pytest.mark.parametrize("said,not_this", [
    ("is my 9/10 quiz on my calendar", "TENTH-EVENT"),
    ("is my gym on my calendar 24/7", "24th of July"),
    ("when is the 2nd lab on my calendar", "OCTOBER-SECOND-EVENT"),
    ("what time is the 3rd lecture on my calendar", "Nothing on"),
])
def test_end_to_end_a_thing_that_is_not_a_day_is_not_answered_as_one(
        reg, said, not_this):
    rng, forced_text, model_text = _both_doors(reg, said)
    assert as_date(rng) is None and not is_ask(rng), rng
    assert not_this not in forced_text, forced_text
    assert not_this not in model_text, model_text
    assert "October" not in forced_text and "October" not in model_text
    assert "did you mean" not in forced_text and "did you mean" not in model_text
    assert forced_text == model_text or rng == "next"


# ======================================== a dashed pair asks, never assumes
#
# DEFAULT TAKEN FOR HIM: "9-12" and "9.12" are a time range at least as
# often as a date.  Round two read "on 9-12" as the 12th of September; this
# round takes the safe direction instead -- a question that offers the date
# reading, never a guess and never a silent today.  Three parts with a year
# ("9-12-2027") cannot be a clock and stay a date.
@pytest.mark.parametrize("said", [
    "what do i have on 9-12", "what do i have on 09-12", "what do i have on 9.12",
    "what do i have on 9-12 from nine", "anything for 9-12", "what about 9-12",
    "9-12",
])
def test_a_dashed_or_dotted_pair_asks_and_offers_the_date_reading(said):
    rng = coerce_range(said, NOW)
    _asks(rng, said)
    assert "12th of September" in ask_words(rng), rng


@pytest.mark.parametrize("said,want", [
    ("anything on 9-12-2027", "2027-09-12"),
    ("anything on 9.12.2027", "2027-09-12"),
    ("what do i have on 9-12-27", "2027-09-12"),
])
def test_a_dashed_triple_with_a_year_is_a_date(said, want):
    assert coerce_range(said, NOW) == want, said


@pytest.mark.parametrize("said", ["what do i have on 13-5", "anything on 9-31",
                                  "what about 13.13"])
def test_a_dashed_pair_that_cannot_be_read_month_first_asks_plainly(said):
    rng = coerce_range(said, NOW)
    _asks(rng, said)
    assert "or a time" not in ask_words(rng), rng      # no date reading to offer


@pytest.mark.parametrize("said", [
    "i'm free 9-12", "the lab runs 2-4", "on 3-4 hours of sleep",
    "i'm free from 9-12", "meet me at 9-12", "version 3.12", "section 3.2",
])
def test_a_dashed_pair_nobody_framed_as_a_date_is_left_alone(said):
    _not_a_day(coerce_range(said, NOW), said)


# ====================== (3) date-shaped and unreadable is a question, both doors
#
# "what is on my calendar in december 2026" asked through the forced door
# and said TODAY through the model door: the date guard lived in
# ``coerce_range``, which the model door's deriver does not call.  The
# guard now lives in the one reader, so a month with no day in it is a
# question through every door -- and "what do i have in december", which
# was a silent today through both, asks which day.
@pytest.mark.parametrize("said,month", [
    ("what is on my calendar in december 2026", "December 2026"),
    ("what do i have in december", "December"),
    ("anything in september 2027", "September 2027"),
    ("what's on for october", "October"),
    ("anything on september 2027", "September 2027"),
    ("what do i have in may", "May"),
    ("what's happening during november", "November"),
])
def test_a_month_with_no_day_in_it_asks_which_day(said, month):
    rng = coerce_range(said, NOW)
    _asks(rng, said)
    assert month in ask_words(rng), rng


@pytest.mark.parametrize("said", ["maybe 3 things", "may 3 people come",
                                  "i may go", "you may be right"])
def test_a_month_word_doing_another_job_is_not_a_question(said):
    _not_a_day(coerce_range(said, NOW), said)


# ============================================ two days in one breath
#
# "the 12th to the 14th" was answered about the 12th with no word about the
# 14th: a confident partial answer.  One range value cannot carry two days,
# so a span is a question -- never the first day of it.
SPANS = [
    "what do i have the 12th to the 14th",
    "what do i have from the 12th to the 14th",
    "what do i have between the 12th and the 14th",
    "what do i have september 12th to september 14th",
    "what's on the 12th and the 13th",
    "what do i have on 9/12 or 9/13",
    "what do i have on the 5th and the 6th of october",
    "was anything on my calendar yesterday and the 14th",
    "what did i have on my calendar yesterday, and the 12th",
]


@pytest.mark.parametrize("said", SPANS)
def test_two_days_in_one_breath_is_a_question_not_the_first_of_them(said):
    rng = coerce_range(said, NOW)
    _asks(rng, said)
    assert "one day at a time" in ask_words(rng), rng


@pytest.mark.parametrize("said,want", [
    ("anything on the 12th and what time is it", "2026-09-12"),
    ("what's on the 12th at 3 and 5", "2026-09-12"),
    ("anything on the 12th at 3", "2026-09-12"),
    ("anything on the 12th of this month", "2026-09-12"),
    ("what's on the 12th, sir", "2026-09-12"),
])
def test_one_day_with_a_second_clause_is_still_that_day(said, want):
    assert coerce_range(said, NOW) == want, said


# ===================================== (2) an offset from today or a weekday
#
# "in two days" was a silent today; "a week on tuesday" was the coming
# Tuesday, seven days short.  Each is a confident wrong day.
@pytest.mark.parametrize("said,want", [
    ("what do i have in two days", "2026-09-07"),
    ("what do i have in 2 days", "2026-09-07"),
    ("what do i have in a week", "2026-09-12"),
    ("what do i have in two weeks", "2026-09-19"),
    ("what do i have in a couple of days", "2026-09-07"),
    ("what do i have in a few days", "2026-09-08"),
    ("what do i have a week on tuesday", "2026-09-15"),
    ("what do i have a week from tuesday", "2026-09-15"),
    ("what do i have two days from now", "2026-09-07"),
    ("what do i have a week from tomorrow", "2026-09-13"),
    ("what do i have a week from today", "2026-09-12"),
    ("what did i have three days ago", "2026-09-02"),
    ("what did i have a week ago", "2026-08-29"),
])
def test_an_offset_from_today_or_a_weekday_lands_on_the_day(said, want):
    assert coerce_range(said, NOW) == want, said


@pytest.mark.parametrize("said", [
    "for the 3rd week running", "3 hours on the 2nd shift", "in 20 minutes",
    "in a while", "in a moment", "i need 20 minutes", "in the next few days",
    "a week is a long time", "two days is plenty",
])
def test_a_duration_that_is_not_an_offset_is_left_alone(said):
    _not_a_day(coerce_range(said, NOW), said)


@pytest.mark.parametrize("said,want", [
    ("what do i have on the last friday of september", "2026-09-25"),
    ("what do i have on the first monday of october", "2026-10-05"),
    ("what do i have on the 2nd tuesday of this month", "2026-09-08"),
    ("what's on the third wednesday in october", "2026-10-21"),
    ("what do i have on the first monday of next month", "2026-10-05"),
    ("what do i have on the last monday of the month", "2026-09-28"),
])
def test_the_nth_weekday_of_a_month_is_that_day(said, want):
    """"the last friday of september" was "friday" -- the 11th, not the 25th."""
    assert coerce_range(said, NOW) == want, said


def test_a_weekday_the_month_does_not_have_is_a_question():
    rng = coerce_range("what do i have on the 5th friday of september", NOW)
    _asks(rng, "5th friday")
    assert "5th Friday" in ask_words(rng) and "September" in ask_words(rng)


# ==================================== (4) one resolver behind every door
#
# WHY THE DAY-SHIFT WAS LOST: ``day_shift_followup`` moved the WEEKDAY word
# and nothing else.  Before round one that worked by accident -- "tuesday
# the 8th" became "Wednesday the 8th", ``coerce_range`` ignored the ordinal
# and answered about Wednesday.  Once the ordinal was read, the same rewrite
# named a weekday the 8th is not, and the follow-up he had been using became
# a question.  The rewrite now moves the DAY the sentence names, read by the
# same resolver the two doors use.
@pytest.mark.parametrize("prev,want", [
    ("what do i have on tuesday the 8th", "2026-09-09"),
    ("what do i have on the 12th", "2026-09-13"),
    ("what do i have on september 12th", "2026-09-13"),
    ("what do i have on 9/12", "2026-09-13"),
    ("what do i have on september twelfth", "2026-09-13"),
    ("what do i have on september 30th", "2026-10-01"),
    ("what did i have on the 3rd", "2026-09-04"),
    ("what's on the day after the 12th", "2026-09-14"),
    ("what do i have on the 5th", "2026-09-06"),
])
def test_and_the_next_day_after_a_dated_question_moves_the_date(prev, want):
    from jarvis.commander import calendar_range, day_shift_followup

    meant = day_shift_followup(prev, "and the next day", TODAY)
    assert meant, prev
    got = calendar_range(meant, NOW)
    assert got == want, (prev, meant, got)
    assert not is_ask(got)


def test_the_follow_up_keeps_his_own_words_around_the_day():
    from jarvis.commander import day_shift_followup

    meant = day_shift_followup("What do I have on Tuesday the 8th?",
                               "and the next day", TODAY)
    assert meant.startswith("What do I have on ")
    assert "Tuesday" not in meant and "8th" not in meant


@pytest.mark.parametrize("prev", [
    "anything on friday the 20th",            # a question is not a day to move
    "what do i have the 12th to the 14th",
    "what do i have on the 5th floor",        # no day named at all
])
def test_a_question_or_a_non_day_is_not_moved(prev):
    from jarvis.commander import day_shift_followup

    assert day_shift_followup(prev, "and the next day", TODAY) is None


def test_the_weekday_only_follow_up_is_unchanged():
    from jarvis.commander import day_shift_followup

    monday = date(2026, 8, 31)
    assert day_shift_followup("What do I have going on tomorrow?",
                              "and the next day", monday) == \
        "What do I have going on Wednesday?"
    assert day_shift_followup("What's on my calendar Sunday?",
                              "and the next day", monday) is None


def _deriver(monkeypatch, now=NOW):
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: now)
    spec = [t for t in make_tools(_Cfg(), SimpleNamespace())
            if t.name == "get_calendar"][0]
    return spec.derive


def test_every_door_reads_the_day_through_the_one_resolver(monkeypatch):
    """The structural pin: the forced door and the model door go through
    ``calendar.sentence_date``, the day-shift rewrite through
    ``calendar.date_span``, and all three through the ONE reader beneath
    them, ``calendar._read`` -- so a rule added there reaches all three,
    and none can drift on its own."""
    from jarvis.commander import calendar_range, day_shift_followup

    seen = []
    real = calendar._read

    def spy(raw, today, backward):
        seen.append(raw)
        return real(raw, today, backward)

    monkeypatch.setattr(calendar, "_read", spy)
    derive = _deriver(monkeypatch)
    calendar_range("what do i have on the 12th", NOW)
    assert len(seen) == 1
    derive("what do i have on the 12th")
    assert len(seen) == 2
    day_shift_followup("what do i have on the 12th", "and the next day", TODAY)
    assert len(seen) >= 3


ALL_ROWS = ([(s, None) for s in UNFRAMED_PAIRS] + FRAMED_PAIRS +
            [(s, "ask") for s in SPANS])


@pytest.mark.parametrize("said,want", ALL_ROWS)
def test_the_two_doors_agree_on_every_new_row(said, want, monkeypatch):
    from jarvis.commander import calendar_range

    derive = _deriver(monkeypatch)
    forced = calendar_range(said, NOW)
    derived = derive(said)
    if want is None:                       # not a day: the model keeps its own
        assert derived == {}, f"{said!r}: {derived!r}"
        _not_a_day(forced, said)
    elif want == "ask":
        assert is_ask(forced) and derived == {"range": forced}, said
    else:
        assert forced == want and derived == {"range": want}, said


def test_the_month_without_a_day_asks_through_the_model_door_too(reg):
    said = "what is on my calendar in december 2026"
    rng, forced_text, model_text = _both_doors(reg, said)
    assert is_ask(rng)
    assert forced_text == model_text == ask_words(rng)
    assert "TODAY-EVENT" not in model_text


# ============================================ the other two defaults
def test_next_monday_still_means_the_coming_monday():
    """LEFT FOR HIM.  "next Monday" is the coming Monday here, as it always
    was; to some ears it is the Monday after that.  Pinned so it cannot
    drift by accident; his to change."""
    assert coerce_range("what do i have next monday", NOW) == "monday"
    assert coerce_range("what do i have next friday", NOW) == "friday"


def test_the_window_is_wide_only_with_the_burst_suppressed():
    """WINDOW_DAYS 14 -> 45 is allowed only because CalendarSource reports
    the window its events were GATHERED over, which is what stops calwatch
    announcing thirty far-out events as new bookings on the first wide
    refresh (tests/test_calendar_window_widening.py measures 31 -> 0)."""
    assert calendar.WINDOW_DAYS == 45
    assert callable(getattr(calendar.CalendarSource, "covered_window", None))


# ============================================================= the clock
CLOCKS = [
    datetime(2026, 9, 5, 0, 0, tzinfo=CHI),
    datetime(2026, 9, 5, 23, 59, tzinfo=CHI),
    datetime(2026, 3, 8, 1, 30, tzinfo=CHI),               # spring forward
    datetime(2026, 11, 1, 1, 30, tzinfo=CHI),              # fall back
    datetime(2026, 12, 31, 23, 59, tzinfo=CHI),
    datetime(2028, 2, 29, 12, 0, tzinfo=CHI),
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("Pacific/Auckland")),
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("UTC")),
]


@pytest.mark.parametrize("now", CLOCKS)
def test_the_frame_rule_holds_at_every_instant(now, monkeypatch):
    from jarvis.commander import calendar_range

    derive = _deriver(monkeypatch, now)
    for said in UNFRAMED_PAIRS:
        _not_a_day(coerce_range(said, now), said)
        assert derive(said) == {}, (said, now)
    for said in SPANS:
        _asks(coerce_range(said, now), said)
    for said, want in FRAMED_PAIRS:
        rng = coerce_range(said, now)
        # "saturday the 12th" is a date on the grid's Saturday and, at an
        # instant where the next 12th is not a Saturday, the weekday
        # question -- either way, never today and never silent.
        assert as_date(rng) is not None or (
            is_ask(rng) and "not a Saturday" in ask_words(rng)), (said, now, rng)
        assert derive(said) == {"range": rng}, (said, now)
    _asks(coerce_range("what do i have on 9-12", now), "9-12")
    assert calendar_range("what do i have in two days", now) == \
        (now.date() + timedelta(days=2)).isoformat()
