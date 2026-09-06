"""ROUND FOUR of the 2026-09-05 calendar-date bug: "MONTH THE Nth", the
relative month, the units that were not units, and THE DERIVER RULE.

Round three closed the frame rule and an adversary confirmed it on a
298-row grid at the injected Saturday 2026-09-05 14:00 America/Chicago:
26 rows still disagreed.  Measured end to end through BOTH doors on
2026-09-06, printed exactly as it came back:

    "what do i have on january the 5th"     -> TODAY's list
    "what do i have on december the 25th"   -> Friday the 25th of SEPTEMBER
    "what do i have on october the 12th"    -> September 12th
        12 months x {1st, 5th, 12th, 25th}: 44 of 48 silently wrong, right
        only for September; "<month> the 5th": 12 of 12 answered TODAY
    the model sends 2027-01-05 for "january the 5th" and Jarvis STILL
        answers today (6/6): the deriver REPLACED a correct model value
    "on the 12th next month", "next month on the 12th",
    "in october on the 12th", "on the 12th in october"  -> September 12th
    "what did i have on the 12th last month"            -> TODAY
    "the friday after next"                             -> the 11th (he meant the 18th)
    "in a month", "in two months", "a month from now", "in a year" -> TODAY
    "how does october look on my calendar"              -> TODAY (forced door)
    "mid september"                                     -> TODAY
    "what's the 2nd to last meeting on my calendar"     -> Friday the 2nd of October

CAUSES: ``_D_MD_RX`` allowed no "the" between month and day, so
``_D_ORD_RX`` read the bare ordinal and DROPPED the month; the bare-ordinal
reader knew no month context; a month and a year were not units; "to" was
a tail word even before "last"; and ``registry.call`` let a deriver
overwrite the model's arguments unconditionally.

THE RULES this file pins, in his register:

* "<month> the Nth" is that month, in the year the year rule gives;
* a bare ordinal takes its month from "next month", "last month", "this
  month" or "in <month>" wherever they stand in the sentence;
* "<weekday> after next" is the one after the coming one; "<weekday>
  after <day>" is the first such weekday past that day;
* a month or a year is a unit, exactly like a day or a week;
* a month with no day in it -- "how does October look", "mid September",
  "next month" -- is a QUESTION, never today;
* "the 2nd to last" and "the 3rd in a row" are ranks;
* "at about 9.30" is a CLOCK -- a DEFAULT TAKEN FOR HIM: a dotted pair
  after "at" is a time, not a date to ask about;
* THE DERIVER RULE: when the model hands over a value that reads as a
  real day, the deriver may refine it but may never replace it with today
  or with a different day.  His words override the model only when they
  CONTRADICT it -- a different day-of-month, or a month they name
  themselves -- and the rule holds even when the reader has a hole.

THE CLOCK: every test injects ``now``.  Nothing here reads the wall clock:
the pure-reader tables run with ``calendar.now_local`` replaced by a
function that RAISES, and the tables are re-run at other instants at the
end.
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
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def _boom(tz=None):
    raise AssertionError("the wall clock was read")


@pytest.fixture
def no_clock(monkeypatch):
    """The pure-reader tests: a wall-clock read is a loud failure."""
    monkeypatch.setattr(calendar, "now_local", _boom)


def _next(month: int, day: int, today: date = TODAY) -> str:
    """THE YEAR RULE: the next such day at or after today."""
    got = date(today.year, month, day)
    if got < today:
        got = date(today.year + 1, month, day)
    return got.isoformat()


# ================================================= (1) MONTH THE Nth
MONTH_THE_NTH = [(f"what do i have on {m} the {d}{calendar._suffix(d)}",
                  _next(i + 1, d))
                 for i, m in enumerate(MONTHS) for d in (1, 5, 12, 25)]


def test_month_the_nth_grid_forced_door(no_clock):
    """The 48-row grid.  Before the fix: 44 of 48 silently wrong, right
    only for September, because the month was dropped and the bare
    ordinal read alone."""
    wrong = [(said, want, coerce_range(said, NOW))
             for said, want in MONTH_THE_NTH]
    wrong = [w for w in wrong if w[1] != w[2]]
    print(f"\n  MONTH THE Nth: {len(wrong)} of {len(MONTH_THE_NTH)} wrong")
    for said, want, got in wrong:
        print(f"    {said!r}: want {want} got {got}")
    assert wrong == []


def test_month_the_5th_is_never_today_by_accident(no_clock):
    """"<month> the 5th" on the 5th: twelve of twelve answered TODAY --
    right for September and wrong for the other eleven."""
    today = [m for m in MONTHS if m != "september"
             if coerce_range(f"what do i have on {m} the 5th", NOW) == TODAY.isoformat()]
    print(f"\n  '<month> the 5th' -> today: {len(today)} of 11 (want 0)")
    assert today == []


@pytest.mark.parametrize("said,want", [
    ("what do i have on september 12th next year", "2027-09-12"),
    ("what do i have on october 12th of 2027", "2027-10-12"),
    ("what do i have on the 12th of september next year", "2027-09-12"),
    ("what do i have on december the 25th 2027", "2027-12-25"),
    ("what do i have on friday december the 25th", "2026-12-25"),
    ("what do i have on may the 4th", "2027-05-04"),
    ("what do i have on sept the 12th", "2026-09-12"),
    ("what about the 12th of oct", "2026-10-12"),
])
def test_the_year_he_says_is_the_year_he_gets(said, want, no_clock):
    assert coerce_range(said, NOW) == want, said


# ================================================= (2) THE RELATIVE MONTH
RELATIVE_MONTH = [
    ("what do i have on the 12th next month", "2026-10-12"),
    ("what do i have next month on the 12th", "2026-10-12"),
    ("what do i have in october on the 12th", "2026-10-12"),
    ("what do i have on the 12th in october", "2026-10-12"),
    ("what do i have on the 5th next month", "2026-10-05"),
    ("what did i have on the 12th last month", "2026-08-12"),
    ("what do i have on the 3rd this month", "2026-09-03"),
    ("what do i have on the 12th of next month", "2026-10-12"),
    ("what do i have on the 12th of this month", "2026-09-12"),
    ("what do i have the friday after next", "2026-09-18"),
    ("what do i have the monday after next", "2026-09-14"),
    ("what do i have the saturday after the 12th", "2026-09-19"),
    ("what do i have the friday before the 12th", "2026-09-11"),
    ("what do i have in december on the 25th", "2026-12-25"),
    ("what did i have in august on the 30th", "2026-08-30"),
    ("what do i have on the 1st next month", "2026-10-01"),
    ("what do i have next month on the 1st", "2026-10-01"),
    ("what do i have last month on the 20th", "2026-08-20"),
    ("what do i have on the 25th in january", "2027-01-25"),
    ("what do i have in january on the 5th", "2027-01-05"),
]


def test_relative_month_grid(no_clock):
    """Before the fix 18 of 20 were the wrong day with no question."""
    wrong = [(said, want, coerce_range(said, NOW)) for said, want in RELATIVE_MONTH]
    wrong = [w for w in wrong if w[1] != w[2]]
    print(f"\n  RELATIVE MONTH: {len(wrong)} of {len(RELATIVE_MONTH)} wrong")
    for said, want, got in wrong:
        print(f"    {said!r}: want {want} got {got}")
    assert wrong == []


def test_the_week_after_next_still_asks(no_clock):
    """"<weekday> after next" is a day; "the WEEK after next" names no day
    to step from and stays the question round two pinned."""
    assert is_ask(coerce_range("what's on the week after next", NOW))


def test_a_weekday_after_nothing_nameable_is_just_the_weekday(no_clock):
    """"friday after the meeting" steps from nothing: the weekday word
    alone, as before -- never an ask, never today."""
    assert coerce_range("what do i have on friday after the meeting", NOW) == "friday"


# ============================================ (3) UNITS AND MONTH-ONLY
UNITS = [
    ("what do i have in a month", "2026-10-05"),
    ("what do i have in two months", "2026-11-05"),
    ("what do i have in 2 months", "2026-11-05"),
    ("what do i have a month from now", "2026-10-05"),
    ("what do i have in a year", "2027-09-05"),
    ("what do i have a year from today", "2027-09-05"),
    ("what did i have a month ago", "2026-08-05"),
    ("what do i have in 100 days", "2026-12-14"),
    ("what do i have a month after the 12th", "2026-10-12"),
    ("what do i have in three months", "2026-12-05"),
]


def test_a_month_and_a_year_are_units(no_clock):
    """Before the fix: "in a month", "in two months", "a month from now",
    "in a year" -> TODAY, 5 of 5, because only day / week / fortnight
    were units and "in 100 days" needed a two-digit count."""
    wrong = [(said, want, coerce_range(said, NOW)) for said, want in UNITS]
    wrong = [w for w in wrong if w[1] != w[2]]
    print(f"\n  UNITS: {len(wrong)} of {len(UNITS)} wrong")
    for said, want, got in wrong:
        print(f"    {said!r}: want {want} got {got}")
    assert wrong == []


def test_a_month_step_from_the_31st_lands_on_the_last_day(no_clock):
    """31 August + a month is 30 September, not an error and not October."""
    aug31 = datetime(2026, 8, 31, 9, 0, tzinfo=CHI)
    assert coerce_range("what do i have in a month", aug31) == "2026-09-30"
    jan31 = datetime(2027, 1, 31, 9, 0, tzinfo=CHI)
    assert coerce_range("what do i have in a month", jan31) == "2027-02-28"


MONTH_ONLY = [
    ("how does october look on my calendar", "October"),
    ("what do i have mid september", "September"),
    ("what do i have mid-september", "September"),
    ("what do i have in early october", "October"),
    ("what do i have in late september", "September"),
    ("is october busy on my calendar", "October"),
    ("what do i have in september next year", "September 2027"),
    ("what do i have next month", "next month"),
    ("what do i have this month", "this month"),
    ("what did i have last month", "last month"),
]


@pytest.mark.parametrize("said,names", MONTH_ONLY)
def test_a_month_with_no_day_asks_which_day(said, names, no_clock):
    """"how does october look" and "mid september" were silent todays:
    they are about WHEN, they name no day the tool can list, so they are
    the question "which day"."""
    rng = coerce_range(said, NOW)
    assert is_ask(rng), f"{said!r} -> {rng!r}"
    assert names in ask_words(rng), (said, ask_words(rng))


# ================================================= (4) NON-DATES
NONDATES = [
    "what's the 2nd to last meeting on my calendar",
    "which is the 3rd in a row on my calendar",
    "what's on my calendar at about 9.30",       # the "at" default
    "am i free at 9.30",
    "what do i have at about 3",
    "what's on my calendar at around 9.30",
]


def _not_a_day(rng, said):
    assert as_date(rng) is None and not is_ask(rng), f"{said!r} -> {rng!r}"


@pytest.mark.parametrize("said", NONDATES)
def test_a_rank_or_a_clock_is_not_a_day(said, no_clock):
    _not_a_day(coerce_range(said, NOW), said)


def test_the_dotted_pair_default_is_unchanged_away_from_at(no_clock):
    """DEFAULTS STANDING: "on 9-12" and "on 9.12" still ask; only a pair
    behind "at" is a clock."""
    for said in ("what do i have on 9-12", "what do i have on 9.12",
                 "what do i have about 9.12"):
        rng = coerce_range(said, NOW)
        assert is_ask(rng) and "time" in ask_words(rng), (said, rng)


# ===================================== the unreadable that were today
def test_a_misheard_month_is_a_question_not_today(no_clock):
    rng = coerce_range("what do i have on the 12th of nevember", NOW)
    assert is_ask(rng), rng
    assert "November" in ask_words(rng)
    # a short word near a short month is NOT a near-miss: "many" is not May
    _not_a_day(coerce_range("the 1st of many", NOW), "the 1st of many")
    _not_a_day(coerce_range("the 2nd of three parts", NOW), "the 2nd of three")


def test_what_about_still_frames_a_dashed_pair(no_clock):
    """The "at" guard must not see the "at" inside "what": "what about
    9-12" asks, as round three pinned."""
    assert is_ask(coerce_range("what about 9-12", NOW))
    assert is_ask(coerce_range("what about 13.13", NOW))


def test_two_ranked_ordinals_do_not_make_the_first_a_date(no_clock):
    rng = coerce_range("what do i have the 1st or 2nd week of october", NOW)
    assert is_ask(rng), rng
    assert "October" in ask_words(rng)


def test_a_slashed_iso_date_is_read_as_the_day_it_names(no_clock):
    """DECIDED: 2026/09/12 can only be one day, so it is read, not asked
    about (the adversary's grid wanted a question here; a question about
    a date with one reading would be the dumber Jarvis)."""
    assert coerce_range("what do i have on 2026/09/12", NOW) == "2026-09-12"


# ============================================== THE DERIVER RULE
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


# One INVENTED event on every day this file asks about that lies inside
# the 45-day window, titled by its day so the day answered is visible.
DAYS = [date(2026, 9, 3), date(2026, 9, 6), date(2026, 9, 7), date(2026, 9, 11),
        date(2026, 9, 12), date(2026, 9, 13), date(2026, 9, 14), date(2026, 9, 18),
        date(2026, 9, 19), date(2026, 9, 25), date(2026, 10, 1), date(2026, 10, 5),
        date(2026, 10, 12), date(2026, 10, 13)]
EVENTS = [_ev(d) for d in DAYS] + [_ev(TODAY, hour=16, title="TODAY-EV")]


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


def _deriver(monkeypatch, now=NOW):
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: now)
    spec = [t for t in make_tools(_Cfg(), SimpleNamespace())
            if t.name == "get_calendar"][0]
    return spec


def _answers_about(text: str, want: str) -> bool:
    """The answer lists that day's invented event, or -- outside the
    window -- refuses BY NAME; and never lists today's instead."""
    d = date.fromisoformat(want)
    if d == TODAY:
        return "TODAY-EV" in text
    if "TODAY-EV" in text:
        return False
    if calendar.out_of_reach(d, TODAY):
        return calendar.date_words(d, TODAY) in text
    return f"EV-{d:%m%d}" in text


JANUARY_THE_5TH = [
    "what do i have on january the 5th",
    "what's on my calendar january the 5th",
    "anything on january the 5th",
    "what is on my calendar on january the 5th",
    "do i have anything on january the 5th",
    "what about january the 5th",
]


def test_the_model_got_january_the_5th_right_and_jarvis_answered_today(reg):
    """THE OVERRIDE, measured 6/6: the model sends 2027-01-05 and the
    deriver's bare-ordinal reading (today) REPLACED it."""
    bad = []
    for said in JANUARY_THE_5TH:
        text = reg.call("get_calendar", {"range": "2027-01-05"}, from_model=True,
                        utterance=said).text
        if not _answers_about(text, "2027-01-05"):
            bad.append((said, text))
    print(f"\n  model=2027-01-05 overridden: {len(bad)} of {len(JANUARY_THE_5TH)}")
    for said, text in bad:
        print(f"    {said!r} -> {text}")
    assert bad == []


def test_a_correct_model_date_stands_on_every_dated_row(reg, monkeypatch):
    """THE DERIVER RULE over every dated row in this file: the model sends
    exactly the day he meant, and the answer is about that day."""
    spec = _deriver(monkeypatch)
    rows = MONTH_THE_NTH + RELATIVE_MONTH + UNITS
    bad = []
    for said, want in rows:
        derived = spec.derive(said, {"range": want})
        if derived not in ({}, {"range": want}):
            bad.append((said, want, derived))
            continue
        text = reg.call("get_calendar", {"range": want}, from_model=True,
                        utterance=said).text
        if not _answers_about(text, want):
            bad.append((said, want, text))
    print(f"\n  correct model value overridden: {len(bad)} of {len(rows)}")
    for row in bad:
        print(f"    {row}")
    assert bad == []


def test_the_deriver_rule_holds_even_with_a_hole_in_the_reader(reg, monkeypatch):
    """STRUCTURAL: put the round-three hole back -- a _D_MD_RX with no
    "the" -- so the reader once more reads "january the 5th" as a bare
    5th (today).  The model's 2027-01-05 is never traded for TODAY.
    ROUND FIVE: the reader's month (this one) and the model's (January)
    disagree over a bare day-of-month, so Jarvis ASKS which -- following
    neither silently -- rather than letting the model's month stand as
    round four did (tests/test_calendar_date_round_five.py)."""
    hole = re.compile(rf"\b{calendar._MONTH}\s+(?P<d>\d{{1,2}})(?!\d)"
                      rf"(?P<ord>{calendar._ORD})?{calendar._YEAR}", re.I)
    monkeypatch.setattr(calendar, "_D_MD_RX", hole)
    monkeypatch.setattr(calendar, "_NEAR_MONTH_RX", re.compile(r"(?!x)x"))
    assert calendar.sentence_date("what do i have on january the 5th", TODAY) \
        == TODAY.isoformat()                                # the hole is back
    text = reg.call("get_calendar", {"range": "2027-01-05"}, from_model=True,
                    utterance="what do i have on january the 5th").text
    assert "TODAY-EV" not in text
    assert text.endswith("?") and "which month" in text, text
    assert "September" in text and "January 2027" in text, text


@pytest.mark.parametrize("said,model,want", [
    # his words CONTRADICT the model: a different day-of-month
    ("what do i have on the 12th", "2026-09-13", "2026-09-12"),
    # his words name the month themselves: they contradict a different one
    ("what do i have on september 12th", "2026-10-12", "2026-09-12"),
    ("what do i have on 9/12", "2026-10-12", "2026-09-12"),
    ("what do i have on the 12th next month", "2026-09-12", "2026-10-12"),
    # a step from today is pinned by the words entirely
    ("what do i have in two days", "2026-10-07", "2026-09-07"),
    ("what do i have the day after tomorrow", "2026-09-06", "2026-09-07"),
    # the old wrong value, and no value at all: the words decide
    ("what do i have on the 12th", "today", "2026-09-12"),
    ("what do i have on the 12th", "", "2026-09-12"),
    # his words gave only a day-of-month and the model moved its month:
    # round four called that a refinement; round five ASKS which month
    # (tests/test_calendar_date_round_five.py::MONTH_MOVED)
    ("what do i have on the 12th", "2026-10-12", "ask"),
    ("what's on the day after the 12th", "2026-10-13", "ask"),
    # the model and the words agree
    ("what do i have on the 12th", "2026-09-12", "2026-09-12"),
])
def test_words_override_the_model_only_when_they_contradict_it(
        reg, said, model, want):
    text = reg.call("get_calendar", {"range": model}, from_model=True,
                    utterance=said).text
    if want == "ask":
        assert text.endswith("?") and "which month" in text, (said, model, text)
        assert "EV-" not in text
        return
    assert _answers_about(text, want), (said, model, text)


def test_a_question_from_his_words_is_never_traded_for_a_guess(reg):
    """The words cannot be read ("friday the 20th" is a Sunday) and the
    model sent a day anyway: the QUESTION stands.  A question is neither
    today nor a different day."""
    text = reg.call("get_calendar", {"range": "2026-09-20"}, from_model=True,
                    utterance="anything on friday the 20th").text
    assert "?" in text and "not a Friday" in text
    assert "EV-0920" not in text and "TODAY-EV" not in text


def test_model_day_stands_is_the_rule_in_one_place():
    from jarvis.tools.calendar import model_day_stands

    said = "what do i have on the 12th"
    # round four: a refinement; round five: a QUESTION (which month)
    assert not model_day_stands(said, "2026-09-12", "2026-10-12", NOW)
    assert not model_day_stands(said, "2026-09-12", "2026-09-13", NOW)  # contradiction
    assert not model_day_stands(said, "2026-09-12", "today", NOW)       # not a day
    assert not model_day_stands(said, "2026-09-12", None, NOW)
    assert not model_day_stands("what do i have on september 12th",
                                "2026-09-12", "2026-10-12", NOW)        # month named
    assert not model_day_stands("anything on friday the 20th",
                                "ask:The 20th ...?", "2026-09-20", NOW)  # an ask stands
    assert model_day_stands(said, "2026-09-12", "2026-09-12", NOW)      # agreement


def test_a_follow_up_the_model_resolved_is_still_not_stolen(reg):
    """No date in the utterance: the model's own value stands, as round
    one pinned."""
    text = reg.call("get_calendar", {"range": "2026-09-13"}, from_model=True,
                    utterance="and the next day").text
    assert "EV-0913" in text


# ======================================== BOTH DOORS on every new row
ALL_NEW = (MONTH_THE_NTH + RELATIVE_MONTH + UNITS +
           [(s, "ask") for s, _ in MONTH_ONLY] +
           [(s, None) for s in NONDATES] +
           [("what do i have on the 12th of nevember", "ask"),
            ("what do i have the 1st or 2nd week of october", "ask"),
            ("what do i have on 2026/09/12", "2026-09-12")])


@pytest.mark.parametrize("said,want", ALL_NEW)
def test_the_two_doors_agree_on_every_new_row(said, want, monkeypatch):
    from jarvis.commander import calendar_range

    spec = _deriver(monkeypatch)
    forced = calendar_range(said, NOW)
    derived = spec.derive(said)
    if want is None:
        assert derived == {}, f"{said!r}: {derived!r}"
        _not_a_day(forced, said)
    elif want == "ask":
        assert is_ask(forced) and derived == {"range": forced}, (said, forced)
    else:
        assert forced == want and derived == {"range": want}, (said, forced)


@pytest.mark.parametrize("said,want", [
    ("what do i have on january the 5th on my calendar", "2027-01-05"),
    ("what is on my calendar on the 12th next month", "2026-10-12"),
    ("what is on my calendar in october on the 12th", "2026-10-12"),
    ("what is on my calendar the friday after next", "2026-09-18"),
    ("what is on my calendar in a month", "2026-10-05"),
])
def test_both_doors_end_to_end(reg, said, want):
    from jarvis.commander import forced_call

    forced = forced_call("local:calendar", said, NOW)
    assert forced is not None and forced[1]["range"] == want, (said, forced)
    f_txt = reg.call(forced[0], forced[1]).text
    m_txt = reg.call("get_calendar", {"range": "today"}, from_model=True,
                     utterance=said).text
    assert f_txt == m_txt, (said, f_txt, m_txt)
    assert _answers_about(f_txt, want), (said, f_txt)


def test_how_does_october_look_asks_through_the_forced_door(reg):
    from jarvis.commander import forced_call

    forced = forced_call("local:calendar", "how does october look on my calendar", NOW)
    assert forced is not None and is_ask(forced[1]["range"]), forced
    text = reg.call(forced[0], forced[1]).text
    assert "October" in text and text.endswith("?")
    assert "TODAY-EV" not in text


# ============================================ the day-shift follow-up
@pytest.mark.parametrize("prev,want", [
    ("what do i have on january the 5th", "2027-01-06"),
    ("what do i have on the 12th next month", "2026-10-13"),
    ("what do i have in october on the 12th", "2026-10-13"),
    ("what do i have the friday after next", "2026-09-19"),
    ("what do i have in a month", "2026-10-06"),
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
    datetime(2028, 2, 29, 12, 0, tzinfo=CHI),
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("Pacific/Auckland")),
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("UTC")),
]


@pytest.mark.parametrize("now", CLOCKS)
def test_the_rules_hold_at_every_instant(now, monkeypatch):
    from jarvis.commander import calendar_range

    spec = _deriver(monkeypatch, now)
    today = now.date()
    for i, m in enumerate(MONTHS):
        for d in (1, 5, 12, 25):
            said = f"what do i have on {m} the {d}{calendar._suffix(d)}"
            rng = coerce_range(said, now)
            assert rng == _next(i + 1, d, today), (said, now, rng)
            assert spec.derive(said) == {"range": rng}, (said, now)
    for said in NONDATES:
        _not_a_day(coerce_range(said, now), said)
        assert spec.derive(said) == {}, (said, now)
    for said, _ in MONTH_ONLY:
        assert is_ask(coerce_range(said, now)), (said, now)
    assert calendar_range("what do i have in a month", now) == \
        calendar.step_months(today, 1).isoformat()
    friday = today + timedelta(days=(4 - today.weekday()) % 7 + 7)
    assert calendar_range("what do i have the friday after next", now) == \
        friday.isoformat()
