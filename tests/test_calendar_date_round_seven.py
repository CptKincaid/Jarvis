"""ROUND SEVEN of the 2026-09-05 calendar-date bug: the tail rule stops
being a whitelist.

SIX rounds have each closed the named rows and each adversary has found
the same bug in a new coat: a sentence that named a day answered about
TODAY, silently.  Round six's adversary re-ran its 956-phrasing grid
across the three commits and measured 320 -> 125 -> 55 wrong at the
primary instant -- and then found the sixth coat.

WHAT THE SIXTH COAT WAS.  _TAIL_WORDS listed, by hand, the ~200 words
that may follow a bare ordinal and leave it a DATE.  Every other word in
English made it a RANK, and a rank falls to today.  The code comment
claimed round five had INVERTED the rule ("a preposition, a conjunction,
a question word, a verb, a pronoun, an adverb ... is a DATE"); it had
not.  It was a finite list, and MEASURED 2026-09-06, 72 of 80 ordinary
English tails behind "the 12th" were a silent today while the identical
80 behind "september 12th" were 0 wrong:

    "what's the 12th like on my calendar"            -> TODAY
    "on the 12th throughout the day"                 -> TODAY
    "what's on my calendar the 12th other than the lab" -> TODAY
    "anything on the 12th besides lunch"             -> TODAY
    "what's on the 12th apart from lunch"            -> TODAY
    ... 'including travel', 'within work hours', 'over lunch',
    'regardless', 'otherwise', 'roughly', 'yeah', 'right' -- all today.

THE RULE SHAPE THIS ROUND TAKES.  A bare ordinal that reaches the
calendar is a DATE -- full stop -- UNLESS a PROVED RANK SHAPE follows
it.  The proved shapes are a CLOSED list, grown only from measured
phrases and each carrying its phrase beside it in the source:

  * a RANK NOUN immediately after it -- "the 3rd floor", "the 1st try",
    "the 2nd opinion";
  * the rank's own FRAMES -- "in the list", "in line", "in a row",
    "to last", "and final", "and last", "and only", "time this month",
    "of three parts";
  * a slashed PAIR, which keeps round six's caution whole (a fraction, a
    score and a ratio wear the same shape as a date and only the frame
    word in front made "3/4" a day).

A word the list does not know now makes the ordinal a DATE, never a
rank.  THE TRADE, stated plainly: a rank noun the list lacks will answer
about a date -- "the 3rd rehearsal on my calendar" gives the 3rd -- and
that is a wrong day HE CAN HEAR AND CORRECT, where the silent today was
a wrong day he could not.

THE OTHER FIVE the round-six adversary measured, end to end through both
doors with calendar.now_local RAISING at nine instants:

(2) THE NEAR-MONTH SWALLOW, a REGRESSION FROM THE BASE.  "what's on my
    calendar the 12th of febuary" ASKED on 61f0945 and answered TODAY on
    560abad: _near_month_ask iterates _NEAR_MONTH_RX with finditer, and
    alternative three ("<word> the <N>th") consumed "calendar the 12th"
    so alternative one never saw "12th of febuary".  44 lead x misspelt
    month combinations, 0 wrong on the base and 40 wrong on the tip.
(3) THE UNFRAMED MISSPELT MONTH.  "febuary 12th" with nothing in front
    of it is a silent today on all three trees: neither _NEAR_MONTH_RX
    nor _BARE_VALUE_RX admits a month-shaped word directly followed by
    a bare day.
(4) THE STEP GUARD'S OTHER HALF (the guard-one-half pattern, a fifth
    time).  _stepped_span's _FROM_RX branch guarded only
    unit.startswith("month") while _IN_RX guarded ("month", "year"), so
    "on the 12th a year from now" gave 2027-09-05; and _TAIL_WORDS
    excluded a/an/one, so "on the 12th a month from now" was today.
(5) THE DAY-SHIFT REWRITE DOUBLE-STEPS.  _shift_named_day writes an
    absolute date back into a sentence that still carries the step that
    produced it: "in a month on the 12th" + "what about the next day"
    became "in a month on Tuesday the 13th of October", which reads as
    2026-10-06 -- the step applied twice.  246 of 324 trials.
(a) THE PAST TENSE BEATS THE MODEL'S YEAR.  "what did i have on
    september 12th" reads LAST September; the model sends next
    September and reconcile_model_day let it stand.  The model's year is
    a refinement only when the tense does not contradict it.
(6) A MONTH WITH A TRAILING CALENDAR NOUN AND NO FRAME -- "december on
    my calendar" -- was a silent today: _MONTH_ONLY_RX wanted a frame
    word IN FRONT of the name and _LONE_MONTH_RX wanted the month to end
    the sentence.

DEFAULTS TAKEN FOR HIM AND NOT CHANGED HERE:
  (b) a month word followed by a proper-noun-ish word keeps the question
      -- "is march madness on my calendar" still asks "Which day in
      March, sir?".  It is never a wrong day, and it is his to overrule.
  (c) the standing pins: "next monday" is the coming Monday, "the monday
      after next" is the 14th, the three fraction frames, and 2026/09/12
      is the day it names.

THE CLOCK: every test injects ``now`` and replaces calendar.now_local
with a function that RAISES, so nothing here can read the wall clock.
"""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.calendar as calendar
from jarvis.commander import calendar_range, day_shift_followup
from jarvis.tools.calendar import (ask_words, as_date, coerce_range, is_ask,
                                   reconcile_model_day, sentence_date)

CHI = ZoneInfo("America/Chicago")
NZ = ZoneInfo("Pacific/Auckland")
NOW = datetime(2026, 9, 5, 14, 0, tzinfo=CHI)          # the grid's Saturday
TODAY = NOW.date()

# The nine instants the round-six adversary measured at: both 2026 DST
# edges for America/Chicago, New Year's Eve and New Year, two 31sts, a
# leap day and a +12 zone.
INSTANTS = [
    ("primary", datetime(2026, 9, 5, 14, 0, tzinfo=CHI)),
    ("nye", datetime(2026, 12, 31, 23, 30, tzinfo=CHI)),
    ("newyear", datetime(2027, 1, 1, 0, 30, tzinfo=CHI)),
    ("dst-spring", datetime(2026, 3, 8, 1, 30, tzinfo=CHI)),
    ("dst-fall", datetime(2026, 11, 1, 1, 30, fold=0, tzinfo=CHI)),
    ("31st", datetime(2026, 10, 31, 9, 0, tzinfo=CHI)),
    ("jan31", datetime(2026, 1, 31, 8, 0, tzinfo=CHI)),
    ("leap", datetime(2028, 2, 29, 12, 0, tzinfo=CHI)),
    ("utc+12", datetime(2026, 8, 31, 23, 59, tzinfo=NZ)),
]


def _boom(tz=None):
    raise AssertionError("the wall clock was read")


@pytest.fixture
def no_clock(monkeypatch):
    monkeypatch.setattr(calendar, "now_local", _boom)


def _not_a_day(rng, said):
    assert as_date(rng) is None and not is_ask(rng), f"{said!r} -> {rng!r}"


def _asks(rng, said, *names):
    assert is_ask(rng), f"{said!r} -> {rng!r}"
    assert ask_words(rng).endswith("?"), rng
    for name in names:
        assert name in ask_words(rng), (said, ask_words(rng))


def _shift_ym(year, month, step):
    idx = year * 12 + (month - 1) + step
    return idx // 12, idx % 12 + 1


def _month_end(year, month):
    ny, nm = _shift_ym(year, month, 1)
    return date(ny, nm, 1) - timedelta(days=1)


def _next_dom(today, day, back=False):
    """The next day-of-month at or after today -- month by month, never a
    year at a time.  The oracle, written from his words, not from the
    module."""
    step = -1 if back else 1
    year, month = today.year, today.month
    for _ in range(24):
        if day <= _month_end(year, month).day:
            got = date(year, month, day)
            if (got <= today) if back else (got >= today):
                return got
        year, month = _shift_ym(year, month, step)
    raise AssertionError("no such day within two years")


def _next_md(today, month, day, back=False):
    step = -1 if back else 1
    for i in range(12):
        year = today.year + step * i
        if day <= _month_end(year, month).day:
            got = date(year, month, day)
            if (got <= today) if back else (got >= today):
                return got
    raise AssertionError("no such day within twelve years")


def _step_months(day, months):
    year, month = _shift_ym(day.year, day.month, months)
    return date(year, month, min(day.day, _month_end(year, month).day))


# =====================================================================
# (1)  THE TAIL RULE IS NO LONGER A WHITELIST
# =====================================================================
# The twenty the round-six adversary wrote in HIS register.  Every one
# was a silent TODAY on 560abad.
IN_REGISTER = [
    "what's the 12th like on my calendar",
    "on the 12th throughout the day",
    "what's on my calendar the 12th other than the lab",
    "anything on the 12th besides lunch",
    "what's on the 12th apart from lunch",
    "the 12th aside from the lab on my calendar",
    "what's on my calendar the 12th including travel",
    "what's on the 12th within work hours",
    "anything on the 12th over lunch",
    "what's on my calendar the 12th regardless",
    "what's on the 12th otherwise",
    "what's on my calendar the 12th roughly",
    "the 12th yeah what's on my calendar",
    "the 12th right what's on my calendar",
    "what's on my calendar on the 12th give or take",
    "what's on the 12th excluding lunch",
    "what's on my calendar the 12th minus lunch",
    "what's on the 12th plus travel",
    "what's on my calendar the 12th depending on traffic",
    "what's on the 12th except lunch",
]

# Eighty ordinary English tails.  The SAME eighty go behind "the 12th"
# and behind "september 12th": the month-named form was 0 wrong on
# 560abad and the bare ordinal 72 wrong, which is the whole finding.
WIDE_TAILS = """throughout the day
other than the lab
besides the lab
apart from lunch
aside from lunch
including travel
within work hours
over lunch
regardless
otherwise
roughly
yeah
right
give or take
barring the lab
excluding lunch
minus lunch
plus travel
notwithstanding
albeit briefly
provided i'm free
assuming i'm free
considering travel
depending on traffic
save for lunch
except lunch
per the schedule
according to the plan
alongside the lab
amid the chaos
among other things
beyond lunch
beneath it all
underneath everything
above all
across town
against the clock
ahead of time
atop everything
away from home
back to back
because of travel
behind schedule
below capacity
beside the point
beyond doubt
close to noon
despite everything
down the line
due to travel
en route
following lunch
inside work hours
instead of lunch
into the evening
irrespective
like last week
near the end
next to nothing
onto the calendar
opposite the lab
out of hours
outside work hours
overall
prior to lunch
pursuant to the plan
regarding travel
round about noon
straight through
subsequent to lunch
thanks to travel
together with lunch
toward the end
under an hour
unlike last week
upon arrival
via the office
whilst travelling
without a break
worth checking
give me the gist""".strip().split("\n")


@pytest.mark.parametrize("said", IN_REGISTER)
def test_his_own_twenty_reach_the_day_he_named(said, no_clock):
    assert coerce_range(said, NOW) == _next_dom(TODAY, 12).isoformat(), said


@pytest.mark.parametrize("tail", WIDE_TAILS)
def test_a_bare_ordinal_survives_any_ordinary_tail(tail, no_clock):
    said = f"what's on my calendar on the 12th {tail}"
    assert coerce_range(said, NOW) == _next_dom(TODAY, 12).isoformat(), said


@pytest.mark.parametrize("tail", WIDE_TAILS)
def test_the_same_tail_behind_a_named_month_is_unchanged(tail, no_clock):
    said = f"what's on my calendar on september 12th {tail}"
    assert coerce_range(said, NOW) == _next_md(TODAY, 9, 12).isoformat(), said


# ...and the CLOSED list of proved rank shapes still holds every one of
# them back.  Each phrase here is the measured evidence for its word.
RANK_SHAPES = [
    "the 3rd floor",
    "on the 5th floor",
    "the 1st try",
    "the 2nd attempt",
    "for the 3rd time",
    "the 4th quarter",
    "the 9th inning",
    "the 6th sense",
    "the 2nd half",
    "the 3rd person",
    "the 1st place",
    "the 2nd version",
    "the 3rd row",
    "the 4th item",
    "the 5th chapter",
    "the 1st draft",
    "the 2nd opinion",
    "the 3rd party",
    "the 1st amendment",
    "the 2nd language",
    "the 1st leg",
    "the 22nd amendment",
    "the 2nd of three parts",
    "the 1st of many",
    "the 4th one",
    "the 3rd option",
    "the 2nd choice",
    "the 7th grade",
    "the 3rd shift",
    "the 1st round",
    "the 2nd base",
    "the 8th wonder",
    "the 4th wall",
    "the 3rd in the list",
    "the 1st in line",
    "the 3rd in a row",
    "the 2nd in command",
    "the 4th in order",
    "the 2nd in charge",
    "the 3rd in the series",
    "the 5th in succession",
    "the 2nd in the queue",
    "the 2nd to last meeting",
    "the 3rd to last",
    "the 2nd to last",
    "the 3rd and final lecture",
    "the 2nd and last meeting",
    "the 1st and only class",
    "the 4th and final round",
    "the 1st or 2nd floor",
    "when is the 2nd lab",
    "my 2nd class",
    "when is my 3rd lecture",
    "the 2nd lecture",
    "the 3rd meeting",
    "the 4th class",
    "the 2nd lab",
    "the 1st exam",
    "the 3rd quiz",
    "the 2nd session",
    "the 3rd lap",
    "the 2nd set",
    "the 4th game",
    "the 3rd period",
    "the 2nd semester",
    "the 5th page",
]


@pytest.mark.parametrize("said", RANK_SHAPES)
def test_a_proved_rank_shape_is_still_not_a_day(said, no_clock):
    _not_a_day(coerce_range(said, NOW), said)
    _not_a_day(coerce_range(said + " on my calendar", NOW), said)


# The trade, asserted so it cannot be walked back by accident: a rank
# noun the CLOSED list does not know answers about a DATE.  He can hear
# that and correct it; the silent today he could not.
def test_a_rank_noun_the_list_lacks_answers_about_a_date(no_clock):
    said = "the 3rd rehearsal on my calendar"
    assert coerce_range(said, NOW) == _next_dom(TODAY, 3).isoformat(), said


# =====================================================================
# (2)  THE NEAR-MONTH SWALLOW -- a REGRESSION from 61f0945
# =====================================================================
NEAR_MONTH_LEADS = [
    "what's on my calendar", "what do i have on my calendar",
    "anything on my calendar", "what's my calendar look like",
    "check my calendar", "look at my calendar",
    "what's on my schedule", "what's my schedule",
    "anything on my schedule", "check my schedule",
    "what's on my agenda",
]
MISSPELT_MONTHS = ["febuary", "nevember", "septmber", "octber"]


@pytest.mark.parametrize("lead", NEAR_MONTH_LEADS)
@pytest.mark.parametrize("month", MISSPELT_MONTHS)
def test_a_lead_in_front_never_swallows_the_misspelt_month(lead, month, no_clock):
    said = f"{lead} the 12th of {month}"
    _asks(coerce_range(said, NOW), said, "12th")


# =====================================================================
# (3)  THE UNFRAMED MISSPELT MONTH
# =====================================================================
UNFRAMED_NEAR_MONTH = [f"{m} {d}" for m in MISSPELT_MONTHS
                       for d in ("12th", "the 12th", "12")]


@pytest.mark.parametrize("said", UNFRAMED_NEAR_MONTH)
def test_a_month_shaped_word_in_front_of_a_bare_day_asks(said, no_clock):
    _asks(coerce_range(said, NOW), said, "12th")


# =====================================================================
# (4)  THE STEP GUARD'S OTHER HALF -- the 40-row matrix
# =====================================================================
def _step_rows():
    rows = []
    for unit, count, months in (("month", "a", 1), ("month", "an", 1),
                                ("month", "one", 1), ("months", "two", 2),
                                ("year", "a", 12), ("year", "an", 12),
                                ("year", "one", 12), ("years", "two", 24)):
        for shape in (f"on the 12th {count} {unit} from now",
                      f"{count} {unit} from now on the 12th",
                      f"in {count} {unit} on the 12th",
                      f"on the 12th in {count} {unit}",
                      f"what's on my calendar on the 12th {count} {unit} from now"):
            rows.append((shape, months))
    return rows


STEP_ROWS = _step_rows()


@pytest.mark.parametrize("said,months", STEP_ROWS)
def test_a_step_and_an_ordinal_agree_in_both_orders(said, months, no_clock):
    year, month = _shift_ym(TODAY.year, TODAY.month, months)
    assert coerce_range(said, NOW) == date(year, month, 12).isoformat(), said


# =====================================================================
# (5)  THE DAY-SHIFT REWRITE MUST NOT DOUBLE-STEP
# =====================================================================
SHIFT_PREVS = [
    "in a month on the 12th",
    "on the 12th in a month",
    "in two months on the 12th",
    "a month from now on the 12th",
    "what's on my calendar in a month on the 12th",
    "in a year on the 12th",
    "in two weeks on the 12th",
    "on the 12th of october",
    "on september 12th",
    "on the 12th",
    "on 9/12",
    "what's on my calendar on the 12th",
    "in three days",
    "two weeks from now",
    "on tuesday the 8th",
    "in a month on the 3rd",
    "in a month",
]
SHIFT_FOLLOWUPS = [
    "what about the next day",
    "and the next day",
    "what about the day after",
    "the day after that",
    "and the day after that",
    "what about the following day",
]


@pytest.mark.parametrize("iname,now", INSTANTS)
@pytest.mark.parametrize("prev", SHIFT_PREVS)
@pytest.mark.parametrize("follow", SHIFT_FOLLOWUPS)
def test_the_day_shift_moves_exactly_one_day(iname, now, prev, follow, no_clock):
    today = now.date()
    before = as_date(coerce_range(prev, now))
    if before is None:
        pytest.skip(f"{prev!r} names no day at {iname}")
    moved = day_shift_followup(prev, follow, today)
    assert moved is not None, (iname, prev, follow)
    assert coerce_range(moved, now) == (before + timedelta(days=1)).isoformat(), \
        (iname, prev, follow, moved)


# =====================================================================
# (a)  THE PAST TENSE BEATS THE MODEL'S YEAR
# =====================================================================
PAST_TENSE = [
    "what did i have on september 12th",
    "what was on my calendar on september 12th",
    "what did i have on the 12th of september",
    "what was i doing on september 12th",
    "what did i have on september 12th on my calendar",
]


@pytest.mark.parametrize("said", PAST_TENSE)
def test_a_past_question_is_not_answered_about_a_future_year(said, no_clock):
    want = _next_md(TODAY, 9, 12, back=True)
    heard = sentence_date(said, TODAY)
    assert heard == want.isoformat(), said
    # the model sends NEXT September; the tense contradicts it, so his
    # words win rather than the model's year refining them.
    got = reconcile_model_day(said, heard, "2027-09-12", NOW)
    assert got == want.isoformat(), (said, got)


def test_a_future_question_still_lets_the_model_refine_the_year(no_clock):
    said = "what do i have on september 12th"
    heard = sentence_date(said, TODAY)
    assert reconcile_model_day(said, heard, "2027-09-12", NOW) is None, said


# =====================================================================
# (6)  A MONTH WITH A TRAILING CALENDAR NOUN AND NO FRAME
# =====================================================================
TRAILING_CAL_NOUN = [
    "december on my calendar",
    "october on my calendar",
    "december on my schedule",
    "october on my agenda",
    "and october on my calendar",
    "october then on my calendar",
]


@pytest.mark.parametrize("said", TRAILING_CAL_NOUN)
def test_a_month_before_a_calendar_noun_asks_which_day(said, no_clock):
    _asks(coerce_range(said, NOW), said)


# =====================================================================
# THE 188 NON-DATES MUST NOT MOVE -- the ones round six pinned
# =====================================================================
STANDING_NON_DATES = [
    "is my calendar 3/4 full",
    "is my calendar 9/10 full",
    "i'm about 3/4 through it",
    "we're 2/3 done",
    "it's 50/50",
    "i got 9/10 on the quiz",
    "the score was 3/5",
    "is it open 24/7",
    "a 1/2 inch gap",
    "the 3/4 mark",
    "on 3-4 hours of sleep",
    "at about 9.30",
    "i'm free 9-12",
    "for 2000 dollars",
    "in 20 minutes",
    "the 5 day forecast",
    "for the 3 of us",
    "on the 2 hour drive",
    "on the 12 o'clock train",
    "maybe 3 things",
    "may 3 people come",
    "is this the 3rd time this month",
]


@pytest.mark.parametrize("said", STANDING_NON_DATES)
def test_the_pinned_non_dates_do_not_move(said, no_clock):
    _not_a_day(coerce_range(said, NOW), said)


# (c) the standing pins, re-asserted so the new rule cannot quietly move
# one of them.
STANDING_PINS = [
    ("next monday", "monday"),
    ("what do i have next monday", "monday"),
    ("2026/09/12", "2026-09-12"),
    ("what do i have on 9/12 from 2 to 4", "2026-09-12"),
    ("is the 12th free", "2026-09-12"),
    ("what's on the 12th at 3pm", "2026-09-12"),
    ("on the 12th in the morning", "2026-09-12"),
    ("on the 12th with john", "2026-09-12"),
]


@pytest.mark.parametrize("said,want", STANDING_PINS)
def test_the_standing_pins_hold(said, want, no_clock):
    assert coerce_range(said, NOW) == want, said


# BOTH DOORS agree on every phrase this round touches: the forced door
# (commander.calendar_range) and the model door (sentence_date) read the
# same sentence with the same reader.
BOTH_DOORS = (IN_REGISTER + [s for s, _ in STEP_ROWS] + PAST_TENSE
              + TRAILING_CAL_NOUN + UNFRAMED_NEAR_MONTH + RANK_SHAPES[:20])


@pytest.mark.parametrize("said", BOTH_DOORS)
def test_both_doors_read_the_same_sentence_the_same_way(said, no_clock):
    forced = calendar_range(said, NOW)
    coerced = coerce_range(said, NOW)
    assert forced == coerced or (coerced == "today" and forced == "next"), said
    heard = sentence_date(said, TODAY)
    if heard is not None:
        assert heard == coerced, said
