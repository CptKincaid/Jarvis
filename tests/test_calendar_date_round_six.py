"""ROUND SIX of the 2026-09-05 calendar-date bug: what round five's
inversion widened, and the silent todays under it.

Round five inverted the tail rule -- a rank needs a NOUN after it -- and
closed the ten "on the 12th from 2 to 4" rows, the year hole in the
deriver, the counts, the month-only possessives and the model-month
question.  Re-running the round-4 ADVERSARY'S OWN 561-row grid on that
tree (2026-09-06, wall clock raising, injected Saturday 2026-09-05)
measured 17 rows still wrong, FOUR of them made wrong BY the inversion:

    "is my calendar 3/4 full"         nondate on 61f0945 -> the 4th of March
    "is my calendar 9/10 full"        nondate on 61f0945 -> the 10th
    "i'm about 3/4 through it"        nondate on 61f0945 -> the 4th of March
    "september 12 through 14"         a question on 61f0945 -> the 12th, and
                                      "through 14" dropped in silence

CAUSE of those four: the inversion is right for an ORDINAL -- "the 12th"
says date by itself -- but a slashed pair says nothing of the kind.  A
fraction, a score and a ratio wear the same shape, and the only thing
that made "3/4" a date was the frame word in front of it.  So the tail
rule behind a PAIR keeps its old caution (an adjective is not
transparent, "there" and "through it" end it as a quantity), and a month
name with two bare numbers joined by to / through is a SPAN, which is a
question, never the first of the two.

The other thirteen were wrong before round five as well, every one a
silent TODAY or a day he never named:

    "is the 3rd and final lecture on my calendar" -> the 3rd of October
    "is this the 3rd time this month"             -> "Which day this month?"
    "what's on for the month" / "december" / "October" / "in 2027" -> TODAY
    "on the last day of the month"                -> TODAY
    "in half a year"                              -> TODAY
    "in two weeks on tuesday"                     -> Saturday the 19th

DEFAULTS TAKEN FOR HIM in this round:
  (f) "in half a year" is six months; "half a month" is not a day Jarvis
      will name and it asks;
  (g) a unit step with a weekday after it -- "in two weeks on tuesday" --
      steps first and then moves FORWARD to that weekday, the same shape
      round five gave "in a month on the 12th";
  (h) "the last day of the month" is the last day of THIS month, "the
      first day of next month" the 1st of the next -- read, not asked
      about;
  (i) a month standing alone -- "december", "and october" -- is a month
      with no day, so it asks which day.

STANDING and not reopened: the four fraction frames behind of / by ("the
odds of 1/3", "a score of 9/10", "a ratio of 3/4", "9 divided by 3/4")
read as days, 2026/09/12 is the day it names, "next monday" is the coming
Monday.

THE CLOCK: every test injects ``now`` and replaces ``calendar.now_local``
with a function that RAISES.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.calendar as calendar
from jarvis.tools.calendar import (ask_words, as_date, coerce_range, is_ask,
                                   sentence_date)

CHI = ZoneInfo("America/Chicago")
NOW = datetime(2026, 9, 5, 14, 0, tzinfo=CHI)          # the grid's Saturday
TODAY = NOW.date()


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


# ==================================== a slashed pair is not a fraction
# Each of the first three was a NON-DATE on 61f0945 and a day after round
# five: the inversion of the tail rule reached the pairs.
FRACTIONS = [
    "is my calendar 3/4 full",
    "is my calendar 9/10 full",
    "is my calendar 1/2 booked",
    "is my week 2/3 full",
    "i'm about 3/4 through it",
    "i'm about 3/4 there",
    "we're about 1/2 there",
    "we're 2/3 of the way there",
    "i'm 9/10 done",
    "the room is 1/2 empty",
]

# The tail rule still applies to the lone YEAR the new rule reads: a
# four-digit number with a noun after it counts money, not days.
NOT_A_YEAR = [
    "is that on my calendar for 2000 dollars",
    "what do i have for 2000 dollars",
    "is the 2000 mile drive on my calendar",
]


@pytest.mark.parametrize("said", FRACTIONS)
def test_a_fraction_is_not_a_day(said, no_clock):
    _not_a_day(coerce_range(said, NOW), said)


@pytest.mark.parametrize("said", NOT_A_YEAR)
def test_a_four_digit_number_with_a_noun_after_it_is_not_a_year(said, no_clock):
    _not_a_day(coerce_range(said, NOW), said)


# ...and the pairs that ARE days keep their reading.
PAIRS_THAT_ARE_DAYS = [
    ("what do i have on 9/12 from 2 to 4", "2026-09-12"),
    ("what do i have on 9/12 with john", "2026-09-12"),
    ("what do i have on my calendar 9/12", "2026-09-12"),
    ("what do i have on 9/12", "2026-09-12"),
    ("what do i have the day after 9/12", "2026-09-13"),
    ("is the 12th free", "2026-09-12"),          # the ordinal keeps the adjective
    ("does the 12th look free", "2026-09-12"),
]


@pytest.mark.parametrize("said,want", PAIRS_THAT_ARE_DAYS)
def test_a_framed_pair_is_still_a_day(said, want, no_clock):
    assert coerce_range(said, NOW) == want


# The four fraction frames behind of / by are STANDING: not reopened here.
@pytest.mark.parametrize("said,want", [
    ("the odds of 1/3", "2027-01-03"),
    ("a score of 9/10", "2026-09-10"),
    ("a ratio of 3/4", "2027-03-04"),
    ("what's 9 divided by 3/4", "2027-03-04"),
])
def test_the_standing_fraction_frames_did_not_move(said, want, no_clock):
    assert coerce_range(said, NOW) == want


# ============================================ a month and two bare numbers
@pytest.mark.parametrize("said", [
    "what do i have september 12 through 14",
    "what do i have september 12 to 14",
    "what do i have september 12 thru 14",
    "what do i have october 12 until 14",
    "what do i have september 12-14",
])
def test_a_month_with_two_numbers_is_a_span_not_the_first_of_them(
        said, no_clock):
    _asks(coerce_range(said, NOW), said, "one day at a time")


@pytest.mark.parametrize("said,want", [
    ("what do i have on the 12th until 5", "2026-09-12"),     # a clock, not a span
    ("what do i have on the 12th from 2 to 4", "2026-09-12"),
    ("what do i have on september 12 2027", "2027-09-12"),
    ("what do i have on september 12", "2026-09-12"),
    # a span runs FORWARD; a smaller number after a month-day is a clock
    ("what do i have on september 12 until 5", "2026-09-12"),
    ("what do i have on september 12 to 4", "2026-09-12"),
    ("what do i have on october 12 at 3", "2026-10-12"),
])
def test_a_clock_after_a_date_is_not_a_span(said, want, no_clock):
    assert coerce_range(said, NOW) == want


# ================================================== "the 3rd and final X"
@pytest.mark.parametrize("said", [
    "is the 3rd and final lecture on my calendar",
    "is the 2nd and last meeting on my calendar",
    "is the 1st and only class on my calendar",
    "what time is the 3rd and last lab",
    "is the 2nd and final rehearsal on my calendar",
])
def test_an_ordinal_paired_with_final_or_last_is_a_rank(said, no_clock):
    _not_a_day(coerce_range(said, NOW), said)


@pytest.mark.parametrize("said", [
    "what do i have on the 12th and 13th of october",
    "what do i have on the 12th and the 14th",
    "what's on the 12th and the 13th",
])
def test_two_ordinals_are_still_a_span(said, no_clock):
    _asks(coerce_range(said, NOW), said, "one day at a time")


# ============================================ a rank's period is not a date
@pytest.mark.parametrize("said", [
    "is this the 3rd time this month",
    "is that the 2nd time this week",
    "is this the 4th time this year",
])
def test_the_period_of_a_rank_is_not_a_day_to_ask_about(said, no_clock):
    _not_a_day(coerce_range(said, NOW), said)


@pytest.mark.parametrize("said,want", [
    ("what do i have this month", None),                      # still asks
    ("what do i have on the 12th of this month", "2026-09-12"),
    ("what do i have on the 12th next month", "2026-10-12"),
])
def test_the_month_he_really_named_still_lands(said, want, no_clock):
    got = coerce_range(said, NOW)
    if want is None:
        _asks(got, said, "this month")
    else:
        assert got == want


# ================================================ a month with no day at all
@pytest.mark.parametrize("said,names", [
    ("what's on for the month", "this month"),
    ("what do i have for the month", "this month"),
    ("what do i have in the month", "this month"),
    ("december", "December"),
    ("October", "October"),
    ("and october", "October"),
    ("october then", "October"),
    ("um december please", "December"),
    ("what do i have in 2027", "2027"),
    ("what do i have for 2028", "2028"),
])
def test_a_month_or_a_year_with_no_day_asks_which_day(said, names, no_clock):
    _asks(coerce_range(said, NOW), said, names)


@pytest.mark.parametrize("said,want", [
    ("what do i have in 2027 on september 12th", "2027-09-12"),
    ("what do i have for the month of october", None),
    ("what do i have on september 12th of 2027", "2027-09-12"),
])
def test_a_year_beside_a_day_is_still_that_day(said, want, no_clock):
    got = coerce_range(said, NOW)
    if want is None:
        _asks(got, said, "October")
    else:
        assert got == want


@pytest.mark.parametrize("said", [
    "what do i have this year",
    "what do i have next year",
    "what do i have for the rest of the year",
    "what do i have in the coming months",
    "what do i have over the next few weeks",
])
def test_a_stretch_of_time_with_no_day_in_it_asks(said, no_clock):
    _asks(coerce_range(said, NOW), said, "which day")


@pytest.mark.parametrize("said,want", [
    ("what do i have next year on september 12th", "2027-09-12"),
    ("what do i have on the 12th of september next year", "2027-09-12"),
])
def test_a_year_word_beside_a_day_is_still_that_day(said, want, no_clock):
    assert coerce_range(said, NOW) == want


# ================================================ the edge days of a month
@pytest.mark.parametrize("said,want", [
    ("what do i have on the last day of the month", "2026-09-30"),
    ("what do i have on the last day of this month", "2026-09-30"),
    ("what do i have on the last day of next month", "2026-10-31"),
    ("what do i have on the last day of october", "2026-10-31"),
    ("what do i have on the first day of next month", "2026-10-01"),
    ("what do i have on the first day of october", "2026-10-01"),
    ("what do i have on the last day of february", "2027-02-28"),
    # THE YEAR RULE on the day he named, not on the 1st of its month: the
    # last day of September is still ahead of him on the 5th, and
    # resolving the 1st first made it September 2027.
    ("what do i have on the last day of september", "2026-09-30"),
    ("what do i have on the first day of september", "2027-09-01"),
    ("what do i have on the last day of september 2027", "2027-09-30"),
])
def test_the_first_and_last_day_of_a_month_are_days(said, want, no_clock):
    assert coerce_range(said, NOW) == want


def test_the_last_weekday_of_a_month_did_not_move(no_clock):
    assert coerce_range("what do i have on the last monday of the month",
                        NOW) == "2026-09-28"
    assert coerce_range("what do i have on the last friday of september",
                        NOW) == "2026-09-25"


# ============================================================ half a unit
def test_half_a_year_is_six_months(no_clock):
    assert coerce_range("what do i have in half a year", NOW) == "2027-03-05"


def test_a_year_and_a_half_carries_the_half(no_clock):
    assert coerce_range("what do i have in a year and a half",
                        NOW) == "2028-03-05"


@pytest.mark.parametrize("said", [
    "what do i have in half a month",
    "what do i have in half a week",
])
def test_half_a_month_is_asked_about_not_guessed(said, no_clock):
    _asks(coerce_range(said, NOW), said)


# ============================================== a step with a weekday after it
@pytest.mark.parametrize("said,want", [
    ("what do i have in two weeks on tuesday", "2026-09-22"),
    ("what do i have two weeks from now on tuesday", "2026-09-22"),
    ("what do i have in a month on friday", "2026-10-09"),
    ("what do i have in three weeks on monday", "2026-09-28"),
])
def test_a_step_moves_forward_to_the_weekday_he_named(said, want, no_clock):
    assert coerce_range(said, NOW) == want


@pytest.mark.parametrize("said,want", [
    ("what do i have in two weeks", "2026-09-19"),
    ("what do i have in a month", "2026-10-05"),
    ("what do i have in a month on the 12th", "2026-10-12"),
    ("what do i have a week on tuesday", "2026-09-15"),
])
def test_a_step_with_no_weekday_did_not_move(said, want, no_clock):
    assert coerce_range(said, NOW) == want


# ===================================================== both doors, one reader
@pytest.mark.parametrize("said", [row for row in FRACTIONS] + [
    "is the 3rd and final lecture on my calendar",
    "is this the 3rd time this month",
    "what do i have september 12 through 14",
    "what do i have on the last day of the month",
    "what do i have in two weeks on tuesday",
    "december",
    "what do i have in 2027",
])
def test_the_forced_and_model_doors_read_the_new_rows_alike(said, no_clock):
    """coerce_range (the forced door) and sentence_date (the reader the
    model door's deriver uses) must not drift apart on the new shapes."""
    forced = coerce_range(said, NOW)
    read = sentence_date(said.lower().strip(" .?!"), TODAY)
    if read is None:
        assert as_date(forced) is None
    else:
        assert forced == read, (said, forced, read)


# =========================================================== other instants
@pytest.mark.parametrize("now", [
    datetime(2026, 9, 5, 0, 0, tzinfo=CHI),
    datetime(2026, 9, 5, 23, 59, tzinfo=CHI),
    datetime(2026, 12, 31, 23, 59, tzinfo=CHI),
    datetime(2027, 1, 31, 12, 0, tzinfo=CHI),
    datetime(2028, 2, 29, 12, 0, tzinfo=CHI),
])
def test_the_new_rules_hold_at_every_instant(now, no_clock):
    """The class of each answer -- a day, a question, no date at all -- may
    not depend on WHEN he asks."""
    for said in FRACTIONS + ["is the 3rd and final lecture on my calendar",
                             "is this the 3rd time this month"]:
        _not_a_day(coerce_range(said, now), said)
    for said in ("december", "what do i have in 2027",
                 "what's on for the month",
                 "what do i have september 12 through 14",
                 "what do i have in half a month"):
        assert is_ask(coerce_range(said, now)), said
    for said in ("what do i have on the last day of the month",
                 "what do i have in two weeks on tuesday",
                 "what do i have in half a year"):
        assert as_date(coerce_range(said, now)) is not None, said
