"""ROUND FIVE of the 2026-09-05 calendar-date bug: the tail rule inverted,
the year hole in the deriver, counts to thirty, the month-only
possessives, and the model-month question.

Round four closed "month the Nth", the relative month, the units and the
deriver rule; its adversary (2026-09-06) confirmed all of that on a fresh
436-row grid and BLOCKED on a fifth coat of the same bug, every count
measured on 61f0945 with the wall clock raising, through both doors over
invented events:

    "what do i have on the 12th from 2 to 4"   -> TODAY   (10 of 10 such:
        after 5 / before noon / around 3 / until 5 / with john / during
        the day / "on the 12th what do i have" / "on 9/12 from 2 to 4")
    "the 12th through the 14th"                -> the 14th, no question (4/4)
    "sept 12th of next year"                   -> THIS September's 12th,
        and the model's correct 2027-09-12 was REPLACED by it (3 of 262)
    "in eleven months", "in fifteen days"      -> TODAY (word counts stopped at ten)
    "in about a month"                         -> TODAY (3/3)
    "how does MY october look"                 -> TODAY through the forced door (5/5)
    "the 3rd in the list"                      -> the 3rd of October
    "on monday 14th"                           -> the coming Monday, the 7th
    "what did i have last tuesday"             -> NEXT Tuesday's list
    "on the 3rd tuesday"                       -> Tuesday the 8th
    "on febuary the 12th"                      -> September the 12th
    "in a month on the 12th"                   -> October the 5th
    and the round-four rule let the model MOVE THE MONTH of a bare
        ordinal with no question: 49 of 262 dated phrasings.

CAUSE, one class: the round-three tail rule was a WHITELIST of words that
may follow a date ("at", "in", "and"...), so every other continuation --
a clock phrase, a name, a question -- made the ordinal a RANK and dropped
it.  The rule is now the other way round: a rank needs a NOUN after it
("the 3rd lab", "the 2nd floor", "the 3rd in the list"); an ordinal
followed by anything that cannot be that noun -- a preposition, a
conjunction, a question word, a verb, a pronoun, a clock word, nothing
at all -- is a DATE.

DEFAULTS TAKEN FOR HIM in this round, each pinned below:
  (a) the model REFINES what his words left unsaid and never overrides
      what they said: day and month said, no year -> the model's year
      stands; day said and no month -> the reader's nearest-upcoming
      month is his, and a model that names a DIFFERENT month is neither
      followed nor overruled -- Jarvis ASKS which month;
  (b) a rank needs a noun after it; a clock, company, question or span
      phrase after an ordinal makes it a date;
  (c) "last <weekday>" is the most recent one BEFORE today -- a day the
      cache usually cannot reach, which is refused BY NAME, never
      answered with the coming one;
  (d) word counts read to thirty; about / roughly / around before a
      count is the count;
  (e) a misspelt month in front of "the Nth" asks, as it does behind.

STANDING and not reopened: "next monday" is the coming Monday, "the
monday after next" is the one after that, the fraction frames, and
2026/09/12 read as the day it names.

THE CLOCK: every test injects ``now``; the pure-reader tables run with
``calendar.now_local`` replaced by a function that RAISES, and are re-run
at other instants at the end.
"""
import re
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


def _next_dom(day: int, today: date) -> str:
    """THE YEAR RULE for a bare ordinal: the next such day-of-month at or
    after today."""
    y, m = today.year, today.month
    for _ in range(24):
        try:
            got = date(y, m, day)
        except ValueError:
            got = None
        if got is not None and got >= today:
            return got.isoformat()
        m += 1
        if m > 12:
            m, y = 1, y + 1
    raise AssertionError("no such day")


# ============================ (1) an ordinal followed by a clock or company
ORDINAL_THEN_PHRASE = [
    ("what do i have on the 12th from 2 to 4", "2026-09-12"),
    ("what do i have on the 12th after 5", "2026-09-12"),
    ("what do i have on the 12th before noon", "2026-09-12"),
    ("what do i have on the 12th around 3", "2026-09-12"),
    ("what do i have on the 12th until 5", "2026-09-12"),
    ("what do i have on the 12th with john", "2026-09-12"),
    ("what do i have on the 12th during the day", "2026-09-12"),
    ("what do i have on the 12th between 2 and 4", "2026-09-12"),
    ("what do i have on the 12th by noon", "2026-09-12"),
    ("what do i have on the 12th in the morning", "2026-09-12"),
    ("on the 12th what do i have", "2026-09-12"),
    ("on the 12th how busy am i", "2026-09-12"),
    ("is the 12th free", "2026-09-12"),
    ("does the 12th work", "2026-09-12"),
    ("what do i have on 9/12 from 2 to 4", "2026-09-12"),
    ("what do i have on 9/12 with john", "2026-09-12"),
    ("what do i have on the 12th 3 pm", "2026-09-12"),
    ("what do i have on the 12th he said", "2026-09-12"),
]


@pytest.mark.parametrize("said,want", ORDINAL_THEN_PHRASE)
def test_an_ordinal_followed_by_a_clock_or_company_phrase_is_a_date(
        said, want, no_clock):
    """MEASURED before the fix: 10 of 10 answered TODAY.  "from", "after",
    "before", "around", "until", "with", "during", "between", "what" and
    "how" were not in the round-three whitelist, so the ordinal was a
    rank and was dropped."""
    assert coerce_range(said, NOW) == want, said


@pytest.mark.parametrize("said", [
    "what do i have on the 12th through the 14th",
    "what do i have on the 12th until the 14th",
    "what do i have on the 12th till the 14th",
    "what do i have on the 12th thru the 14th",
    "what do i have the 12th through to the 14th",
])
def test_two_days_joined_by_through_or_until_is_a_question(said, no_clock):
    """MEASURED before the fix: the 14th silently (4/4).  The first
    ordinal failed the tail rule before the span check could see it."""
    rng = coerce_range(said, NOW)
    _asks(rng, said, "one day at a time")


# ============================================== (2) the year hole
YEAR_ROWS = [
    ("what do i have on sept 12th of next year", "2027-09-12"),
    ("what do i have on september 12th of next year", "2027-09-12"),
    ("what do i have on the 12th of september of next year", "2027-09-12"),
    ("what do i have next year on september 12th", "2027-09-12"),
    ("what do i have in 2027 on september 12th", "2027-09-12"),
    ("what do i have on september 12th in 2027", "2027-09-12"),
    ("what did i have last year on september 12th", "2025-09-12"),
    ("what do i have next year on the 12th of september", "2027-09-12"),
]


@pytest.mark.parametrize("said,want", YEAR_ROWS)
def test_the_year_he_says_is_read_wherever_he_puts_it(said, want, no_clock):
    """MEASURED before the fix: 2026-09-12 for every one of these -- the
    year rule read "next year" only when it stood right after the day."""
    assert coerce_range(said, NOW) == want, said


def test_the_day_after_a_slashed_pair_is_read(no_clock):
    """"the day after 9/12" was "Which day is that after, sir?": the pair
    after the offset had nothing framing it.  The offset IS the frame."""
    assert coerce_range("what's on the day after 9/12", NOW) == "2026-09-13"
    assert coerce_range("what's on two days after 9/12", NOW) == "2026-09-14"
    assert coerce_range("what's on the day before 9/12", NOW) == "2026-09-11"


# ============================================== (3) counts and months
UNITS = [
    ("what do i have in eleven months", "2027-08-05"),
    ("what do i have in twelve months", "2027-09-05"),
    ("what do i have in fifteen days", "2026-09-20"),
    ("what do i have in twenty days", "2026-09-25"),
    ("what do i have in twenty-one days", "2026-09-26"),
    ("what do i have in twenty one days", "2026-09-26"),
    ("what do i have in thirty days", "2026-10-05"),
    ("what do i have in eighteen months", "2028-03-05"),
    ("what do i have in about a month", "2026-10-05"),
    ("what do i have in roughly a month", "2026-10-05"),
    ("what do i have in around two months", "2026-11-05"),
    ("what do i have in about two weeks", "2026-09-19"),
    ("what do i have in approximately three weeks", "2026-09-26"),
    ("what's on fifteen days after the 12th", "2026-09-27"),
    ("what did i have fifteen days ago", "2026-08-21"),
]


@pytest.mark.parametrize("said,want", UNITS)
def test_word_counts_read_to_thirty_and_about_is_the_count(said, want, no_clock):
    """DEFAULT (d).  MEASURED before the fix: "in eleven months", "in
    fifteen days", "in about a month" -> TODAY; "in about two weeks" ->
    the week range."""
    assert coerce_range(said, NOW) == want, said


MONTH_ONLY = [
    ("how does my october look on my calendar", "October"),
    ("what does my october look like on my calendar", "October"),
    ("how's october looking on my calendar", "October"),
    ("what's my october like on my calendar", "October"),
    ("how is my october looking", "October"),
    ("what do i have for the rest of the month", "this month"),
    ("what's on for the rest of the month", "this month"),
    ("what do i have in nevember", "November"),
    ("what do i have for octobre", "October"),
]


@pytest.mark.parametrize("said,names", MONTH_ONLY)
def test_a_month_with_no_day_asks_whoever_owns_it(said, names, no_clock):
    """MEASURED before the fix: 5 of 5 silently today through the FORCED
    door -- the month-only rule wanted the month right after in / for /
    does / is, and "my" stood in the way."""
    _asks(coerce_range(said, NOW), said, names)


# ================================================= (4) non-dates
NONDATES = [
    "what's the 3rd in the list on my calendar",
    "what's the 1st in line on my calendar",
    "which is the 3rd in a row on my calendar",
    "what's the 2nd to last meeting on my calendar",
    "what's the 2nd last meeting on my calendar",
    "the 3rd person in line",
    "what's the 5th item on my agenda",
    "is the 11th hour meeting on my calendar",
    "what's on for the 3rd period",
    "the 2nd of three parts",
    "what's the 1st thing on my calendar",
    "is the 2nd round interview on my calendar",
    "is the 21st century lecture on my calendar",
    "what's the 4th session about",
    "for the 2nd time is my meeting on my calendar",
    "on the 3rd try did it go on my calendar",
    "may 3 people come",
    "on 9/12 speed",
    "on 3-4 hours of sleep",
    "what's on the 5 day forecast",
    "is the 12 o'clock slot free",
]


@pytest.mark.parametrize("said", NONDATES)
def test_a_rank_needs_a_noun_after_it(said, no_clock):
    """DEFAULT (b).  "the 3rd in the list" and "the 1st in line" were the
    3rd and 1st of October: "in" was a whitelisted tail word.  A rank is
    the Nth of some noun; these all have one."""
    _not_a_day(coerce_range(said, NOW), said)


# ================================================= (5) other wrong days
OTHERS = [
    ("what do i have on monday 14th", "2026-09-14"),
    ("what do i have on tuesday 8th", "2026-09-08"),
    ("what did i have last tuesday", "2026-09-01"),
    ("what did i have last friday", "2026-09-04"),
    ("what did i have last saturday", "2026-08-29"),
    ("what do i have on the 3rd tuesday", "2026-09-15"),
    ("what do i have on the 2nd monday", "2026-09-14"),
    ("what do i have on the last friday", "2026-09-25"),   # the month's last
    ("what do i have on the 12th of the following month", "2026-10-12"),
    ("what do i have on the 12th of the coming month", "2026-10-12"),
    ("what did i have on the 12th of the previous month", "2026-08-12"),
    ("what do i have in a month on the 12th", "2026-10-12"),
    ("what do i have in two months on the 12th", "2026-11-12"),
    ("what do i have on the 12th two months from now", "2026-11-12"),
    ("what do i have on the 12th in a month", "2026-10-12"),
]


@pytest.mark.parametrize("said,want", OTHERS)
def test_the_other_wrong_days_now_land(said, want, no_clock):
    """MEASURED before the fix: "monday 14th" -> the 7th; "last tuesday"
    -> the coming Tuesday; "the 3rd tuesday" -> the 8th; "in a month on
    the 12th" -> the 5th of October; "the following month" -> today."""
    assert coerce_range(said, NOW) == want, said


def test_a_month_step_does_not_swallow_the_ordinal_it_was_stepping_to(no_clock):
    """"a month from the 12th" is still a step FROM the 12th."""
    assert coerce_range("what do i have a month from the 12th", NOW) == "2026-10-12"
    assert coerce_range("what do i have a month after the 12th", NOW) == "2026-10-12"
    assert coerce_range("what do i have in a month", NOW) == "2026-10-05"


@pytest.mark.parametrize("said,names", [
    ("what do i have on febuary the 12th", ("February",)),
    ("what's on my calendar febuary the 12th", ("February",)),
    ("what do i have on septmber the 3rd", ("September",)),
    ("what do i have on the 12th of nevember", ("November",)),
    ("what do i have on the 12", ("12th",)),
    ("what do i have on the 12th of september or october", ()),
    ("what do i have on the 12th of december or january", ()),
    ("what do i have in january the 2nd week", ("January",)),
    ("what do i have on the 12th of next year", ()),
])
def test_the_unreadable_that_was_a_confident_day_now_asks(said, names, no_clock):
    """DEFAULT (e): the misspelt month in front asks like the one behind.
    "on the 12" (the suffix the transcriber drops) asks rather than
    guessing; two months joined by "or" is a question."""
    _asks(coerce_range(said, NOW), said, *names)


def test_the_standing_defaults_did_not_move(no_clock):
    assert coerce_range("what do i have next monday", NOW) == "monday"
    assert coerce_range("what do i have the monday after next", NOW) == "2026-09-14"
    assert coerce_range("what do i have on 2026/09/12", NOW) == "2026-09-12"
    assert is_ask(coerce_range("what do i have on 9-12", NOW))
    _not_a_day(coerce_range("i got 9/10 on the quiz", NOW), "9/10")
    _not_a_day(coerce_range("what's on my calendar at about 9.30", NOW), "at 9.30")
    assert coerce_range("what do i have on the last friday of september", NOW) \
        == "2026-09-25"
    assert coerce_range("what do i have on the 12th next month", NOW) == "2026-10-12"
    assert coerce_range("what did i have on the 12th last month", NOW) == "2026-08-12"
    assert coerce_range("what about september 12th class", NOW) == "2026-09-12"


# ============================================== THE DERIVER, refined
class _Cfg:
    def __init__(self):
        self.data = {"google_ical_urls": ["https://example.invalid/invented.ics"],
                     "icloud": {"apple_id": "", "app_password": "", "url": ""}}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _ev(d: date, hour=9, title=None) -> Event:
    start = datetime(d.year, d.month, d.day, hour, 0, tzinfo=CHI)
    return Event(start=start, end=start + timedelta(hours=1),
                 title=title or f"EV-{d:%m%d}")


# One INVENTED event on EVERY day of the 45-day window, titled by its day,
# so the day answered is visible in the text; TODAY-EV on the Saturday.
_LO, _HI = calendar.reachable(TODAY)
EVENTS = [_ev(_LO + timedelta(days=i)) for i in range((_HI - _LO).days + 1)
          if _LO + timedelta(days=i) != TODAY] + [_ev(TODAY, 16, "TODAY-EV")]


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: NOW)
    services = SimpleNamespace()
    src = calendar.make_source(_Cfg(), services, cache_path=tmp_path / "c.json",
                               tz=CHI, fetch=lambda *a, **k: b"",
                               clock=lambda: NOW.timestamp())
    src._sources = {"google-1": {"fetched_at": NOW.timestamp(),
                                 "events": list(EVENTS),
                                 "window": src.window(NOW)}}
    r = ToolRegistry()
    r.register_many(make_tools(_Cfg(), services))
    return r


def _spec(monkeypatch, now=NOW):
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: now)
    return [t for t in make_tools(_Cfg(), SimpleNamespace())
            if t.name == "get_calendar"][0]


def _about(text: str, want: str) -> bool:
    d = date.fromisoformat(want)
    if d == TODAY:
        return "TODAY-EV" in text
    if "TODAY-EV" in text:
        return False
    if calendar.out_of_reach(d, TODAY):
        return calendar.date_words(d, TODAY) in text
    return f"EV-{d:%m%d}" in text and "?" not in text


def _is_question(text: str) -> bool:
    return "?" in text and "EV-" not in text


@pytest.mark.parametrize("said", [s for s, _ in YEAR_ROWS if "2027" in _])
def test_the_models_correct_year_is_never_replaced_by_the_readers_default(reg, said):
    """MEASURED 3 of 262: the model sent 2027-09-12, the reader gave
    2026-09-12 (its own default year, not one he said), counted as pinned
    because a month was named, and REPLACED the model's value."""
    text = reg.call("get_calendar", {"range": "2027-09-12"}, from_model=True,
                    utterance=said).text
    assert _about(text, "2027-09-12"), (said, text)


def test_the_year_is_a_refinement_even_with_a_hole_in_the_reader(reg, monkeypatch):
    """STRUCTURAL, default (a): put the round-four year hole back -- a
    ``_YEAR`` that reads "next year" only right after the day -- so the
    reader once more gives 2026-09-12 for "sept 12th of next year".  The
    model's 2027-09-12 must STILL stand: his words named the day and the
    month, and no year the reader can hold against it."""
    hole = r"(?:,?\s+(?:of\s+)?(?P<y>\d{4})|\s+(?P<ry>this|next|last)\s+year)?"
    md = re.compile(rf"\b{calendar._MONTH}\s+(?:the\s+)?(?P<d>\d{{1,2}})(?!\d)"
                    rf"(?P<ord>{calendar._ORD})?{hole}", re.I)
    monkeypatch.setattr(calendar, "_D_MD_RX", md)
    monkeypatch.setattr(calendar, "_YEAR_CTX_RX", re.compile(r"(?!x)x"))
    said = "what do i have on sept 12th of next year"
    assert calendar.sentence_date(said, TODAY) == "2026-09-12"      # the hole
    text = reg.call("get_calendar", {"range": "2027-09-12"}, from_model=True,
                    utterance=said).text
    assert _about(text, "2027-09-12"), text


@pytest.mark.parametrize("said,model,want", [
    # (a) day and month said, no year: the model's year is a refinement
    ("what do i have on september 12th", "2027-09-12", "2027-09-12"),
    ("what do i have on 9/12", "2027-09-12", "2027-09-12"),
    # ...unless his words named the year: then they contradict it
    ("what do i have on september 12th 2026", "2027-09-12", "2026-09-12"),
    ("what do i have on 9/12/26", "2027-09-12", "2026-09-12"),
    ("what do i have on september 12th this year", "2027-09-12", "2026-09-12"),
    # a month his words named contradicts a different one
    ("what do i have on september 12th", "2026-10-12", "2026-09-12"),
    ("what do i have on the 12th next month", "2026-09-12", "2026-10-12"),
    ("what do i have in a month on the 12th", "2026-09-12", "2026-10-12"),
    # a step from today is his words' entirely
    ("what do i have in twelve months", "2027-10-05", "2027-09-05"),
    ("what do i have in fifteen days", "2026-09-21", "2026-09-20"),
    # a different day-of-month is a contradiction, as before
    ("what do i have on the 12th", "2026-09-13", "2026-09-12"),
    ("what do i have on monday 14th", "2026-09-15", "2026-09-14"),
    ("what do i have on the 12th from 2 to 4", "2026-09-13", "2026-09-12"),
    # agreement
    ("what do i have on the 12th", "2026-09-12", "2026-09-12"),
    ("what do i have on the 12th with john", "2026-09-12", "2026-09-12"),
])
def test_the_model_refines_what_his_words_left_unsaid(reg, said, model, want):
    text = reg.call("get_calendar", {"range": model}, from_model=True,
                    utterance=said).text
    assert _about(text, want), (said, model, text)


MONTH_MOVED = [
    ("what do i have on the 12th", "2026-10-12", "September", "October"),
    ("what do i have for the 12th", "2026-10-12", "September", "October"),
    ("what do i have on saturday the 12th", "2026-10-12", "September", "October"),
    ("what's on the day after the 12th", "2026-10-13", "September", "October"),
    ("what do i have on the 1st", "2026-11-01", "October", "November"),
    ("what do i have on the 12th", "2027-09-12", "September", "September 2027"),
    ("what do i have on the 12th with john", "2026-10-12", "September", "October"),
    ("what do i have the saturday after the 12th", "2026-10-19", "September", "October"),
    ("what do i have a month from the 12th", "2026-11-12", "October", "November"),
]


@pytest.mark.parametrize("said,model,mine,theirs", MONTH_MOVED)
def test_a_model_that_moves_the_month_of_a_bare_ordinal_is_asked_about(
        reg, said, model, mine, theirs):
    """DEFAULT (a), the design exposure the adversary measured: 49 of 262
    dated phrasings let the model's month stand over a bare ordinal with
    no question.  His words gave only a day-of-month; the reader's
    nearest-upcoming month is one reading, the model's another; Jarvis
    follows neither silently."""
    text = reg.call("get_calendar", {"range": model}, from_model=True,
                    utterance=said).text
    assert _is_question(text), (said, model, text)
    assert "which month" in text and mine in text and theirs in text, text
    assert "TODAY-EV" not in text


def test_the_month_question_is_the_deriver_rule_in_one_place():
    from jarvis.tools.calendar import model_day_stands, reconcile_model_day

    said = "what do i have on the 12th"
    assert reconcile_model_day(said, "2026-09-12", "2026-09-12", NOW) is None
    assert model_day_stands(said, "2026-09-12", "2026-09-12", NOW)
    got = reconcile_model_day(said, "2026-09-12", "2026-10-12", NOW)
    assert is_ask(got) and "which month" in ask_words(got), got
    assert not model_day_stands(said, "2026-09-12", "2026-10-12", NOW)
    assert reconcile_model_day(said, "2026-09-12", "2026-09-13", NOW) == "2026-09-12"
    assert reconcile_model_day(said, "2026-09-12", "today", NOW) == "2026-09-12"
    assert reconcile_model_day(said, "2026-09-12", None, NOW) == "2026-09-12"
    # day and month said: the year is the model's to refine
    assert reconcile_model_day("what do i have on september 12th", "2026-09-12",
                               "2027-09-12", NOW) is None
    assert reconcile_model_day("what do i have on september 12th 2026",
                               "2026-09-12", "2027-09-12", NOW) == "2026-09-12"
    # an ask from his words stands over everything
    ask = "ask:The 20th ...?"
    assert reconcile_model_day("anything on friday the 20th", ask, "2026-09-20",
                               NOW) == ask


def test_a_question_from_the_words_still_stands_over_a_model_day(reg):
    text = reg.call("get_calendar", {"range": "2026-09-12"}, from_model=True,
                    utterance="what do i have on febuary the 12th").text
    assert "February" in text and text.endswith("?")
    assert "EV-0912" not in text


def test_last_tuesday_is_refused_by_name_through_both_doors(reg):
    """DEFAULT (c).  The 1st is four days back and the cache keeps only
    yesterday: the answer names the day it cannot reach, and never lists
    the coming Tuesday."""
    from jarvis.commander import forced_call

    said = "what did i have last tuesday on my calendar"
    forced = forced_call("local:calendar", said, NOW)
    assert forced is not None and forced[1]["range"] == "2026-09-01", forced
    f_txt = reg.call(forced[0], forced[1]).text
    m_txt = reg.call("get_calendar", {"range": "tuesday"}, from_model=True,
                     utterance=said).text
    assert f_txt == m_txt, (f_txt, m_txt)
    assert "Tuesday the 1st" in f_txt and "EV-0908" not in f_txt, f_txt
    # yesterday's weekday IS reachable, and is answered
    said = "what did i have last friday on my calendar"
    forced = forced_call("local:calendar", said, NOW)
    assert "EV-0904" in reg.call(forced[0], forced[1]).text, forced


# ======================================== BOTH DOORS on every new row
ALL_NEW = (ORDINAL_THEN_PHRASE + YEAR_ROWS + UNITS + OTHERS +
           [(s, "ask") for s, _ in MONTH_ONLY] +
           [(s, None) for s in NONDATES] +
           [("what do i have on the 12th through the 14th", "ask"),
            ("what do i have on febuary the 12th", "ask"),
            ("what do i have on the 12", "ask"),
            ("what's on the day after 9/12", "2026-09-13")])


@pytest.mark.parametrize("said,want", ALL_NEW)
def test_the_two_doors_agree_on_every_new_row(said, want, monkeypatch):
    from jarvis.commander import calendar_range

    spec = _spec(monkeypatch)
    forced = calendar_range(said, NOW)
    derived = spec.derive(said, {"range": "today"})
    if want is None:
        assert derived == {}, f"{said!r}: {derived!r}"
        _not_a_day(forced, said)
    elif want == "ask":
        assert is_ask(forced) and derived == {"range": forced}, (said, forced)
    else:
        assert forced == want and derived == {"range": want}, (said, forced)


@pytest.mark.parametrize("said,want", [
    ("what do i have on the 12th from 2 to 4 on my calendar", "2026-09-12"),
    ("what do i have on the 12th with john on my calendar", "2026-09-12"),
    ("on the 12th what do i have on my calendar", "2026-09-12"),
    ("what do i have on monday 14th on my calendar", "2026-09-14"),
    ("what do i have in fifteen days on my calendar", "2026-09-20"),
    ("what do i have in a month on the 12th on my calendar", "2026-10-12"),
    ("what do i have on the 3rd tuesday on my calendar", "2026-09-15"),
])
def test_both_doors_end_to_end(reg, said, want):
    from jarvis.commander import forced_call

    forced = forced_call("local:calendar", said, NOW)
    assert forced is not None and forced[1]["range"] == want, (said, forced)
    f_txt = reg.call(forced[0], forced[1]).text
    m_txt = reg.call("get_calendar", {"range": "today"}, from_model=True,
                     utterance=said).text
    assert f_txt == m_txt, (said, f_txt, m_txt)
    assert _about(f_txt, want), (said, f_txt)


@pytest.mark.parametrize("said", [
    "how does my october look on my calendar",
    "what's my october like on my calendar",
    "what do i have for the rest of the month on my calendar",
    "what do i have on febuary the 12th on my calendar",
    "what do i have on the 12th through the 14th on my calendar",
])
def test_the_forced_door_asks_where_it_answered_today(reg, said):
    from jarvis.commander import forced_call

    forced = forced_call("local:calendar", said, NOW)
    assert forced is not None and is_ask(forced[1]["range"]), forced
    text = reg.call(forced[0], forced[1]).text
    assert text.endswith("?") and "TODAY-EV" not in text, text


@pytest.mark.parametrize("said", [
    "what's the 3rd in the list on my calendar",
    "what's the 1st in line on my calendar",
])
def test_a_rank_is_not_answered_as_a_day_through_either_door(reg, said):
    from jarvis.commander import forced_call

    forced = forced_call("local:calendar", said, NOW)
    assert forced is not None
    _not_a_day(forced[1]["range"], said)
    f_txt = reg.call(forced[0], forced[1]).text
    m_txt = reg.call("get_calendar", {"range": "today"}, from_model=True,
                     utterance=said).text
    for txt in (f_txt, m_txt):
        assert "October" not in txt and "?" not in txt, txt


# ============================================ the day-shift follow-up
@pytest.mark.parametrize("prev,want", [
    ("what do i have on monday 14th", "2026-09-15"),
    ("what do i have on sept 12th of next year", "2027-09-13"),
    ("what do i have next year on september 12th", "2027-09-13"),
    ("what do i have in 2027 on september 12th", "2027-09-13"),
    ("what do i have on the 3rd of the month", "2026-09-04"),
    ("what do i have on saturday, september 12", "2026-09-13"),
    ("what do i have on the 12th from 2 to 4", "2026-09-13"),
    ("what do i have in fifteen days", "2026-09-21"),
    ("what do i have on the 3rd tuesday", "2026-09-16"),
    ("what do i have on the 8th of sept, a tuesday", "2026-09-09"),
    ("what do i have on tuesday the 8th", "2026-09-09"),
])
def test_and_the_next_day_moves_the_new_forms_too(prev, want):
    from jarvis.commander import calendar_range, day_shift_followup

    meant = day_shift_followup(prev, "and the next day", TODAY)
    assert meant, prev
    got = calendar_range(meant, NOW)
    assert got == want, (prev, meant, got)


# ============================================================= the clock
CLOCKS = [
    datetime(2026, 9, 5, 0, 0, tzinfo=CHI),
    datetime(2026, 9, 5, 23, 59, tzinfo=CHI),
    datetime(2026, 3, 8, 1, 30, tzinfo=CHI),               # spring forward
    datetime(2026, 11, 1, 1, 30, tzinfo=CHI),              # fall back
    datetime(2026, 12, 31, 23, 59, tzinfo=CHI),
    datetime(2027, 1, 31, 12, 0, tzinfo=CHI),              # a 31st
    datetime(2028, 2, 29, 12, 0, tzinfo=CHI),              # leap day
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("Pacific/Auckland")),
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("UTC")),
]


@pytest.mark.parametrize("now", CLOCKS)
def test_the_rules_hold_at_every_instant(now, monkeypatch):
    from jarvis.commander import calendar_range

    spec = _spec(monkeypatch, now)
    today = now.date()
    twelfth = _next_dom(12, today)
    sept_12 = date(today.year, 9, 12)
    if sept_12 < today:
        sept_12 = date(today.year + 1, 9, 12)
    for said, _ in ORDINAL_THEN_PHRASE:
        rng = coerce_range(said, now)
        want = sept_12.isoformat() if "9/12" in said else twelfth
        assert rng == want, (said, now, rng)
        assert spec.derive(said, {"range": "today"}) == {"range": rng}, (said, now)
    for said in NONDATES:
        _not_a_day(coerce_range(said, now), said)
        assert spec.derive(said, {"range": "today"}) == {}, (said, now)
    for said, _ in MONTH_ONLY:
        assert is_ask(coerce_range(said, now)), (said, now)
    assert calendar_range("what do i have in fifteen days", now) == \
        (today + timedelta(days=15)).isoformat()
    assert calendar_range("what do i have in eleven months", now) == \
        calendar.step_months(today, 11).isoformat()
    last_tuesday = today - timedelta(days=(today.weekday() - 1) % 7 or 7)
    assert calendar_range("what did i have last tuesday", now) == \
        last_tuesday.isoformat()
    assert calendar_range("what do i have on september 12th of next year", now) == \
        f"{today.year + 1}-09-12"
    assert is_ask(calendar_range("what do i have on the 12th through the 14th", now))
