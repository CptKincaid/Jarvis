"""The model already answers "is this a day or a position?", and the
deriver used to throw that answer away.

Eight rounds hand-grew word lists to tell "the 12th" (a day) from "the
3rd rehearsal" (a position), and each round an adversary found the same
bug in a new coat, because the two readings wear the SAME SHAPE.  A
design pass measured the local model on that question -- gemma4:26b,
temperature 0, one call per row carrying this module's own get_calendar
schema, n=20 -- and got an ISO date on 11 of 11 true dates and a WORD
range on 8 of 8 true positions.  19 of 19, on a call Jarvis already
makes and already pays for.

``reconcile_model_day`` discarded it: ``as_date`` is None for a word
range, so "week" -- the model's clearest possible "this is not a
specific day" -- fell through to the reader's bare-ordinal date.

Every date here is injected; ``now_local`` is never read (there is a
guard for that below); every phrase is invented.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from jarvis.tools import calendar as C


NOW = datetime(2026, 9, 6, 14, 0)          # a Sunday, his own day
TODAY = NOW.date()


@pytest.fixture(autouse=True)
def _no_wall_clock(monkeypatch):
    """Nothing in this file may read the clock."""
    def boom(*a, **k):
        raise AssertionError("a test read the wall clock")
    monkeypatch.setattr(C, "now_local", boom)


def _derive(said: str, model_range, now: datetime = NOW):
    """The deriver as ``_range_from_words`` calls it."""
    heard = C.sentence_date(said, now.date())
    if not heard:
        return "reader-silent"
    return C.reconcile_model_day(said, heard, model_range, now)


# ---- a word range on a bare ordinal is the model saying "not a day" ----
@pytest.mark.parametrize("said", [
    "the 3rd rehearsal",
    "the 2nd standup",
    "the 4th briefing",
    "the 2nd errand on the list",
    "the 3rd practice",
    "what is the first on my calendar",
])
@pytest.mark.parametrize("model_range", ["today", "tomorrow", "week", "next", "friday"])
def test_the_model_may_say_it_is_not_a_day(said, model_range):
    """A word range from the model beats the reader's bare-ordinal date.

    Before this rule every one of these answered about a day in October
    that he never named."""
    assert _derive(said, model_range) == C.coerce_range(model_range, NOW)


@pytest.mark.parametrize("said", [
    "the 3rd rehearsal", "the 2nd standup", "what is the first on my calendar",
])
def test_before_the_rule_these_answered_about_a_day(said):
    """The reader on its own still reads these as a day -- which is why
    the rule is needed, and what it is overriding."""
    assert C.sentence_date(said, TODAY) is not None


# ---- and it never touches a date, an ask, or a pinned phrase ----------
@pytest.mark.parametrize("said, model_range", [
    ("what do i have on the 12th", "2026-09-12"),
    ("what's on the 12th except lunch", "2026-09-12"),
    ("on the 12th from 2 to 4", "2026-09-12"),
    ("september 12th", "2026-09-12"),
    ("the 12th of october", "2026-10-12"),
])
def test_a_model_date_still_stands_where_it_agrees(said, model_range):
    assert _derive(said, model_range) is None


@pytest.mark.parametrize("said, model_range", [
    ("september 12th", "week"),
    ("the 12th of october", "today"),
    ("sept 12th of next year", "next"),
])
def test_his_words_pinned_it_so_the_model_cannot_say_otherwise(said, model_range):
    """A month or a year HE said outranks the model's word range: the
    rule fires only when his words pinned nothing at all."""
    got = _derive(said, model_range)
    assert got not in (None, "week", "today", "next"), got
    assert C.as_date(got) is not None


def test_the_year_refinement_survives():
    """Round seven's rule: day and month said, no year, model supplies
    the year -- the model still stands."""
    assert _derive("sept 12th of next year", "2027-09-12") is None


def test_an_ask_is_never_overridden():
    heard = C.sentence_date("the 12th of febuary", TODAY)
    assert C.is_ask(heard)
    assert C.reconcile_model_day("the 12th of febuary", heard, "week", NOW) == heard


def test_a_model_ask_does_not_become_the_answer():
    """``coerce_range`` is a fixed point on an ask, so an ask coming back
    from the model side must not be mistaken for a word range."""
    ask = C._ask("Which day, sir?")
    assert C.coerce_range(ask, NOW) == ask
    heard = C.sentence_date("the 3rd rehearsal", TODAY)
    assert C.reconcile_model_day("the 3rd rehearsal", heard, ask, NOW) == heard


def test_no_model_range_changes_nothing():
    heard = C.sentence_date("the 3rd rehearsal", TODAY)
    assert C.reconcile_model_day("the 3rd rehearsal", heard, None, NOW) == heard
    assert C.reconcile_model_day("the 3rd rehearsal", heard, "", NOW) == heard


# ---- the exposure, pinned so nobody rediscovers it as a surprise ------
@pytest.mark.parametrize("said", [
    "what do i have on the 12th",
    "on the 12th from 2 to 4",
    "for the 12th",
    "on monday 14th",
])
def test_a_preposition_is_his_words_saying_it_IS_a_day(said):
    """THE NARROWING, and it is what keeps round four's pin true.  His
    date-shaped ordinals carry a preposition and a position carries a
    bare "the" -- the branch's own discriminator (see _ORD_PRONOUN_RX).
    So the model's "not a day" is trusted only where his words framed
    nothing: with "on"/"for" in front, ONE wrong word range from the
    model would carry a real date away, and that is the failure he
    cannot hear."""
    got = _derive(said, "today")
    assert C.as_date(got) is not None, (said, got)


def test_the_exposure_is_a_bare_ordinal_and_is_named():
    """THE COST OF TRUSTING THE MODEL, measured and deliberate, and now
    NARROWER than the design pass proposed: a true date carrying nothing
    but a bare ordinal -- no preposition, no month, no year -- on which
    the model wrongly sends a word range is answered about that range.

    Measured at 0 of 11 rows in the design pass, and it cannot be zero by
    construction: it is bounded by how well the model reads a date, which
    is the thing being trusted.  A preposition protects it, and so does a
    month or a year he said.  This test exists so the trade is visible in
    the suite rather than in a comment."""
    assert _derive("the 12th", "week") == "week"
    assert C.as_date(_derive("september 12th", "week")) == date(2026, 9, 12)
    assert C.as_date(_derive("what do i have on the 12th", "week")) == date(2026, 9, 12)


# ---- the rule holds at every instant ---------------------------------
@pytest.mark.parametrize("now", [
    datetime(2026, 1, 1, 0, 1),            # New Year
    datetime(2026, 3, 8, 2, 30),           # a DST edge
    datetime(2026, 11, 1, 1, 30),          # the other DST edge
    datetime(2026, 12, 31, 23, 59),        # New Year's Eve
    datetime(2026, 8, 31, 12, 0),          # a 31st
    datetime(2028, 2, 29, 12, 0),          # a leap day
])
def test_the_rule_does_not_move_with_the_clock(now):
    assert _derive("the 3rd rehearsal", "week", now) == "week"
    assert _derive("the 2nd standup", "today", now) == "today"


# ---- HIS RULING on "next monday", 2026-09-06 -------------------------
@pytest.mark.parametrize("now, want", [
    (datetime(2026, 9, 6, 14, 0), date(2026, 9, 7)),    # a Sunday -> tomorrow
    (datetime(2026, 9, 2, 9, 0), date(2026, 9, 7)),     # a Wednesday
    (datetime(2026, 9, 8, 9, 0), date(2026, 9, 14)),    # a Tuesday -> six days
    (datetime(2026, 12, 31, 9, 0), date(2027, 1, 4)),   # across a year
])
def test_next_monday_is_the_coming_monday(now, want):
    """HIS RULING, 2026-09-06, in his own words: "Next Monday is the the
    coming Mondays".

    Carried as an open default since round three because to some ears
    "next Monday" means the Monday after that.  Settled now, and pinned
    here so nobody reopens it.  Driven end to end: the reader answers the
    weekday, and format_events resolves it to the next day of that name,
    so an event ON that day is the one he is told about.

    NOT settled by his ruling and deliberately not asserted: what "next
    monday" means when today IS a Monday.  The code answers TODAY
    (``(want - today.weekday()) % 7``); most ears would say the Monday
    after.  A separate question, and his to rule on if it ever bites.
    """
    assert C.coerce_range("next monday", now) == "monday"

    def _event(day):
        return C.Event(start=datetime(day.year, day.month, day.day, 9, 0),
                       end=datetime(day.year, day.month, day.day, 10, 0),
                       title="INVENTED-%s" % day.isoformat())
    wanted, decoy = _event(want), _event(want + timedelta(days=7))
    said = C.format_events([wanted, decoy], "monday", now=now)
    assert wanted.title in said, (now.date(), said)
    assert decoy.title not in said, (now.date(), said)
