"""HIS REPORT 2026-09-05: "I just tried to have Jarvis tell me about a
specific calendar date and he didn't get it."

Reproduced with invented dates before a line was written.  ``coerce_range``
understood ELEVEN values -- today, tomorrow, week, next and the seven
weekday names -- and its last line was a bare ``return today``, so every
explicit date became TODAY in silence.  Measured, exactly as printed:

    "what do i have on september 12th"        -> today
    "anything on the 12th"                    -> today
    "what about october 3rd"                  -> today
    "what is on my calendar on the 20th"      -> today
    "do i have anything on sept 12"           -> today
    "what do i have on 9/12"                  -> today
    "what about the 15th of october"          -> today
    "anything on friday the 20th"             -> friday   <- the NEXT Friday

He was not merely unanswered: he was answered CONFIDENTLY ABOUT THE WRONG
DAY, and "friday the 20th" is the sharpest case because it looks handled.

THE RULES THIS FILE PINS (each decision is stated where it is tested):

* an explicit date is carried as an ISO ``YYYY-MM-DD`` range string, which
  ``coerce_range`` leaves alone -- both paths coerce, so it must be a fixed
  point;
* a future-worded question resolves FORWARD (the next occurrence at or
  after today), a past-worded one BACKWARD;
* a date outside the fortnight the cache actually holds is REFUSED by name,
  never answered "nothing on it";
* a date that cannot be read -- the 31st of September, the 32nd, 13/5, a
  weekday that does not fall on the day named -- is ASKED ABOUT.  Nothing
  unrecognised may become "today" again.

THE CLOCK: every test injects its own ``now``.  Nothing here reads the wall
clock, and ``test_any_hour_any_date_any_timezone`` re-runs the whole
phrasing table at ten different instants -- including both 2026 US DST
transitions and a southern-hemisphere zone -- so the file cannot rot into a
34th clock-bound test.
"""
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.calendar as calendar
from jarvis.tools.calendar import (Event, ask_words, as_date, coerce_range,
                                   format_events, is_ask, make_tools)
from jarvis.tools.registry import ToolRegistry

CHI = ZoneInfo("America/Chicago")           # he is in Texas
# Saturday 5 September 2026, 9:00 am.  The reachable window from here is
# 2026-09-04 (yesterday) through 2026-09-18 -- CalendarSource._window
# anchors at yesterday-midnight and reaches WINDOW_DAYS + 1 days.
NOW = datetime(2026, 9, 5, 9, 0, tzinfo=CHI)
TODAY = NOW.date()


def _ev(start: datetime, title: str, minutes: int = 60) -> Event:
    return Event(start=start, end=start + timedelta(minutes=minutes), title=title)


# ------------------------------------------------------------ his report
HIS_PHRASINGS = [
    # (what he said, the day he meant, asked on Saturday 2026-09-05)
    ("what do i have on september 12th", "2026-09-12"),
    ("anything on the 12th", "2026-09-12"),
    ("what about october 3rd", "2026-10-03"),
    ("what is on my calendar on the 20th", "2026-09-20"),
    ("do i have anything on sept 12", "2026-09-12"),
    ("what do i have on 9/12", "2026-09-12"),
    ("what about the 15th of october", "2026-10-15"),
]


@pytest.mark.parametrize("said,want", HIS_PHRASINGS)
def test_each_failing_phrasing_answers_about_the_right_day(said, want):
    """The seven that all came back "today".  Each must now name its day.

    Asserted on the RANGE, not on the wording, because the range is what
    both the forced path and the model path carry and it is the thing that
    was wrong.
    """
    got = coerce_range(said, NOW)
    assert got == want, f"{said!r} -> {got!r}"
    assert as_date(got) == date.fromisoformat(want)


def test_no_phrasing_of_his_comes_back_as_today_any_more():
    """The bug in one line: none of the eight is "today" or a weekday."""
    for said, _ in HIS_PHRASINGS:
        assert coerce_range(said, NOW) not in calendar.RANGES, said
    # ...and the eighth, which came back "friday" -- the WRONG Friday.
    assert coerce_range("anything on friday the 20th", NOW) != "friday"


# ------------------------------------------------- the regression guard
#
# This matters more than the feature.  The eleven values that worked on
# 2026-09-05 morning must still work, unchanged, plus every loose form the
# existing suite already pinned (test_calendar_tool.py).
def test_the_eleven_values_still_work_unchanged():
    for word in calendar.RANGES:
        assert coerce_range(word, NOW) == word, word
    assert len(calendar.RANGES) == 11


@pytest.mark.parametrize("raw,want", [
    ("this week", "week"), ("Week", "week"), ("tmrw", "tomorrow"),
    ("what's next", "next"), ("upcoming", "next"), ("", "today"),
    (None, "today"), ("Today.", "today"), ("next 7 days", "week"), (3, "today"),
    ("on monday", "monday"), ("for Monday", "monday"), ("this monday", "monday"),
    ("whats on tmws calendar", "tomorrow"), ("what's on my calendar tmrw", "tomorrow"),
    ("anything on tmw?", "tomorrow"), ("do i have anything tomorrow", "tomorrow"),
    ("what's on my calendar tomorrow?", "tomorrow"), ("tmr", "tomorrow"),
    ("whats on my calendar", "today"), ("what's on Monday", "monday"),
    ("seven days", "week"), ("coming up", "next"), ("soon", "next"),
])
def test_every_loose_form_the_suite_already_pinned(raw, want):
    assert coerce_range(raw, NOW) == want


@pytest.mark.parametrize("said", [
    "what's my 2nd class",              # an ordinal that is not a date
    "when is my 3rd lecture tomorrow",  # ...with a real day word behind it
    "what's on in the next 7 days",
    "how long is my 1st meeting",
])
def test_an_ordinal_that_is_not_a_date_does_not_become_one(said):
    """"2nd class" is a rank, not the 2nd of the month.  A bare ordinal is
    only read as a date after on / for / the, which is how every phrasing
    he actually used says it."""
    assert as_date(coerce_range(said, NOW)) is None
    assert not is_ask(coerce_range(said, NOW))


# ------------------------------------------------------ friday the 20th
#
# DECISION: the DATE wins, and a weekday that contradicts it is never
# silently discarded.  Jarvis says which day the date really is and offers
# both readings.  Picking either one quietly is the failure he reported.
def test_friday_the_twentieth_when_the_twentieth_is_a_friday():
    now = datetime(2026, 11, 15, 9, 0, tzinfo=CHI)     # Sunday; Fri 20 Nov
    assert date(2026, 11, 20).strftime("%A") == "Friday"
    assert coerce_range("anything on friday the 20th", now) == "2026-11-20"


def test_friday_the_twentieth_when_the_twentieth_is_not_a_friday():
    """Asked on 2026-09-05 the next 20th is SUNDAY 20 September.  Jarvis
    must not pick the 20th silently, nor the next Friday silently -- the
    old code picked the next Friday (the 25th) and said nothing."""
    rng = coerce_range("anything on friday the 20th", NOW)
    assert is_ask(rng), rng
    said = ask_words(rng)
    assert "Sunday" in said and "Friday" in said
    assert "20th" in said and "18th" in said        # the nearest Friday
    assert said.endswith("?")


def test_a_weekday_that_agrees_with_the_date_is_not_questioned():
    assert coerce_range("what's on saturday the 12th", NOW) == "2026-09-12"
    assert coerce_range("monday the 14th of september", NOW) == "2026-09-14"


# --------------------------------------------------------- the year rule
#
# DECISION: a future-worded question means the NEXT occurrence of that
# month-and-day at or after today; a past-worded one the most recent at or
# before today.  Nothing else is a rule -- "January 5th" in September is
# next January precisely because it is the next one.
@pytest.mark.parametrize("now,said,want", [
    # a date just AHEAD: this month, this year.
    (NOW, "what do i have on september 6th", "2026-09-06"),
    (NOW, "what do i have on september 5th", "2026-09-05"),      # today itself
    # a date just PAST, asked in the present tense: the next one, next year.
    (NOW, "what do i have on september 3rd", "2027-09-03"),
    # the far side of the year boundary, both ways.
    (NOW, "what do i have on january 5th", "2027-01-05"),
    (datetime(2026, 1, 1, 9, 0, tzinfo=CHI), "anything on december 31st", "2026-12-31"),
    (datetime(2026, 12, 31, 23, 30, tzinfo=CHI), "anything on january 1st", "2027-01-01"),
    # a bare day-of-month rolls to the next month, not the next year.
    (NOW, "anything on the 3rd", "2026-10-03"),
    (NOW, "anything on the 5th", "2026-09-05"),                  # today itself
    (NOW, "anything on the 31st", "2026-10-31"),                 # September has none
])
def test_the_year_and_month_rule_at_its_boundaries(now, said, want):
    assert coerce_range(said, now) == want


def test_the_twenty_ninth_of_february_in_a_non_leap_year():
    """2026, 2027 have no 29 February; the next one is 2028.  "the next
    occurrence" answers this without a special case -- and it must not
    become the 1st of March, which would be the wrong day again."""
    assert coerce_range("what do i have on february 29th", NOW) == "2028-02-29"
    # Asked ON a leap day, the same words mean that day.
    leap = datetime(2028, 2, 29, 9, 0, tzinfo=CHI)
    assert coerce_range("what do i have on february 29th", leap) == "2028-02-29"


def test_an_explicit_year_is_taken_as_written():
    assert coerce_range("what do i have on september 12th 2029", NOW) == "2029-09-12"
    assert coerce_range("anything on 9/12/2029", NOW) == "2029-09-12"
    # ...including one in the past, which the window check then refuses.
    assert coerce_range("what do i have on september 12th 2024", NOW) == "2024-09-12"


# ---------------------------------------------------------- past dates
#
# DECISION: past-worded questions resolve BACKWARD, and are then answered
# only for the one past day the cache actually holds -- yesterday.  The
# window is anchored at yesterday-midnight (CalendarSource._window), so
# anything earlier is not absent from his calendar, it is absent from the
# cache, and saying "nothing on it" would be the same confident wrong
# answer in a new coat.
def test_a_past_worded_question_looks_backward():
    assert coerce_range("what did i have on the 3rd", NOW) == "2026-09-03"
    assert coerce_range("what did i have on september 12th", NOW) == "2025-09-12"
    assert coerce_range("was i busy on the 4th", NOW) == "2026-09-04"
    # The same day, asked forwards, is a different day. This is the whole
    # reason the two questions are told apart.
    assert coerce_range("what do i have on the 3rd", NOW) == "2026-10-03"


def test_yesterday_is_a_day_too():
    """Not in his list, but the same bug: "yesterday" was "today" as well,
    and yesterday is the one past day the cache genuinely holds."""
    assert coerce_range("what did i have yesterday", NOW) == "2026-09-04"
    assert format_events([_ev(datetime(2026, 9, 4, 10, 0, tzinfo=CHI), "Dentist")],
                         "yesterday", NOW) == \
        "Friday the 4th: 10:00 am Dentist for an hour; nothing else."


def test_a_past_date_beyond_the_cache_is_refused_by_name():
    text = format_events([], coerce_range("what did i have on the 3rd", NOW), NOW)
    assert "Nothing" not in text
    assert "Thursday the 3rd" in text
    assert "yesterday" in text.lower()


# ------------------------------------------------- the reachable window
#
# DECISION: the cache holds yesterday through today+13 and nothing else, so
# a date outside it is refused with its own name and the edge date, never
# answered empty.  Widening WINDOW_DAYS is a live-system decision (it moves
# what calwatch reads as a new booking) and is deliberately NOT taken here.
def test_a_future_date_beyond_the_cache_is_refused_by_name():
    text = format_events([], "2026-10-03", NOW)
    assert "Nothing" not in text
    assert "Saturday the 3rd of October" in text
    assert "Friday the 18th" in text            # the far edge, named
    # His own words land here, and get a straight answer about reach.
    assert format_events([], coerce_range("what about october 3rd", NOW), NOW) == text


def test_the_edges_of_the_window_are_inclusive():
    assert "Nothing on Friday the 4th, sir." == format_events([], "2026-09-04", NOW)
    assert "Nothing on Friday the 18th, sir." == format_events([], "2026-09-18", NOW)
    for out in ("2026-09-03", "2026-09-19"):
        assert "Nothing on" not in format_events([], out, NOW), out


def test_the_window_follows_the_source_that_owns_it():
    """A source with a shorter window must refuse sooner: the promise is
    the cache's, not a constant this module made up."""
    assert "Nothing on" in format_events([], "2026-09-10", NOW, window_days=14)
    assert "Nothing on" not in format_events([], "2026-09-10", NOW, window_days=3)


# ------------------------------------------------- dates he cannot mean
#
# DECISION: an unreadable date is a QUESTION, never a guess.  The silent
# fallback to today is the whole bug, so every branch below must ask.
@pytest.mark.parametrize("said", [
    "what do i have on september 31st",     # September has 30 days
    "anything on february 30th",            # never, in any year
    "what about the 32nd",                  # not a day of any month
    "anything on 13/13",                    # neither number is a month
    "what do i have on 9/45",               # no 45th
    "anything on 0/12",
])
def test_a_date_he_cannot_have_meant_is_asked_about(said):
    rng = coerce_range(said, NOW)
    assert is_ask(rng), f"{said!r} -> {rng!r}"
    assert rng != "today"
    assert ask_words(rng).endswith("?")


def test_the_ask_reaches_him_as_the_spoken_line():
    rng = coerce_range("what do i have on september 31st", NOW)
    assert format_events([], rng, NOW) == ask_words(rng)
    assert "September" in ask_words(rng) and "31" in ask_words(rng)


# ------------------------------------------------------------ ambiguity
#
# DECISION: he is in Texas, so a slashed date is month-first.  A first
# number above 12 cannot be, and rather than quietly switching convention
# for one input -- exactly the "looks handled and is not" trap -- Jarvis
# names the day-month reading and asks.
def test_a_slashed_date_is_read_the_american_way():
    assert coerce_range("what do i have on 9/12", NOW) == "2026-09-12"
    assert coerce_range("anything on 12/9", NOW) == "2026-12-09"
    assert coerce_range("what about 1/2", NOW) == "2027-01-02"


def test_a_slashed_date_that_cannot_be_american_asks():
    rng = coerce_range("what do i have on 13/5", NOW)
    assert is_ask(rng), rng
    said = ask_words(rng)
    assert "13th of May" in said
    assert said.endswith("?")
    # 13/13 has no reading at all; it still asks rather than guessing.
    assert is_ask(coerce_range("anything on 13/13", NOW))


# ----------------------------------------------------------- the wording
def test_a_date_with_no_events():
    assert format_events([], "2026-09-12", NOW) == "Nothing on Saturday the 12th, sir."


def test_a_date_with_several_events():
    day = datetime(2026, 9, 12, tzinfo=CHI)
    evs = [_ev(day.replace(hour=9, minute=10), "BIOSENSORS", 50),
           _ev(day.replace(hour=12, minute=40), "MAGNETIC RESONANCE ENGR", 50),
           _ev(day.replace(hour=16, minute=10), "ELECTRICAL DESIGN LAB II", 110)]
    assert format_events(evs, "2026-09-12", NOW) == (
        "Saturday the 12th: 9:10 am BIOSENSORS for 50 minutes, "
        "12:40 pm MAGNETIC RESONANCE ENGR for 50 minutes, "
        "4:10 pm ELECTRICAL DESIGN LAB II for about 2 hours; nothing else.")
    # Only that day: the day before and the day after are not swept in.
    assert "BIOSENSORS" not in format_events(evs, "2026-09-11", NOW)


def test_a_date_that_is_today_or_tomorrow_is_worded_as_he_would_say_it():
    """Answering "September 12th" with "Saturday the 12th" is right; doing
    it for TODAY would be a stilted way to say a word he has."""
    evs = [_ev(NOW.replace(hour=14, minute=30), "Standup", 15)]
    assert format_events(evs, "2026-09-05", NOW) == \
        "Today: 2:30 pm Standup for 15 minutes; nothing else."
    assert format_events([], "2026-09-06", NOW) == "Nothing on tomorrow, sir."


def test_the_month_is_named_when_the_date_leaves_this_month():
    """"the 3rd" alone is unambiguous inside September and dangerous
    outside it -- naming the month is how he hears a wrong pick."""
    assert "Saturday the 12th" in format_events([], "2026-09-12", NOW)
    assert "Saturday the 3rd of October" in format_events([], "2026-10-03", NOW)
    assert "Friday the 3rd of September 2027" in format_events([], "2027-09-03", NOW)


def test_the_completeness_claim_is_still_taken_back_on_a_dead_feed():
    """drop_completeness() matches the wording format_events produces; a
    new day label must not slip past it and keep claiming the day is fully
    accounted for while a subscription is unread."""
    assert calendar.drop_completeness("Nothing on Saturday the 12th, sir.") == \
        "Nothing on Saturday the 12th in the calendars I can reach, sir."
    assert calendar.drop_completeness(
        "Saturday the 12th: 9:10 am BIOSENSORS; nothing else.") == \
        "Saturday the 12th: 9:10 am BIOSENSORS."


# ------------------------------------------------------- both the paths
#
# The claim in the brief was that BOTH the forced path and the model path
# funnel through coerce_range, so one fix serves both.  Verified, not
# assumed -- once through commander.forced_call, once through the registry
# with from_model=True.
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


def _registry(tmp_path, monkeypatch, events, now):
    """The real tool spec over a source primed by hand: no network, no
    refresh thread, and ``now_local`` -- the module's own clock seam --
    frozen, so these pass at any hour on any date."""
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: now)
    services = SimpleNamespace()
    source = calendar.make_source(_Cfg(), services,
                                  cache_path=tmp_path / "calendar_cache.json",
                                  tz=CHI, fetch=lambda *a, **k: b"",
                                  clock=lambda: now.timestamp())
    source._sources = {"google-1": {"fetched_at": now.timestamp(),
                                    "events": list(events)}}
    reg = ToolRegistry()
    reg.register_many(make_tools(_Cfg(), services))
    return reg, source


def test_the_forced_path_carries_the_date(tmp_path):
    from jarvis.commander import calendar_range, forced_call

    assert calendar_range("what do i have on september 12th", NOW) == "2026-09-12"
    # forced_call needs a calendar NOUN (_CAL_READ_RX), so of his eight only
    # the "on my calendar" one is forced at all -- MEASURED here, and the
    # reason the model path below has to be fixed too rather than instead.
    assert forced_call("local:calendar",
                       "what do i have on september 12th", NOW) is None
    assert forced_call("local:calendar",
                       "what is on my calendar on the 20th", NOW) == \
        ("get_calendar", {"range": "2026-09-20"})
    # The repair that was already there survives: a bare "when" is the next
    # event, not today.
    assert calendar_range("when's my meeting", NOW) == "next"
    # An unreadable date reaches the tool as the ask, not as today.
    _tool, args = forced_call("local:calendar",
                              "what's on my calendar on the 32nd", NOW)
    assert is_ask(args["range"])


def test_the_model_path_carries_the_date(tmp_path, monkeypatch):
    """The model fills ``range`` from his words.  Whatever it writes, the
    deriver reads the date off the utterance itself."""
    ev = _ev(datetime(2026, 9, 12, 9, 10, tzinfo=CHI), "BIOSENSORS", 50)
    reg, _src = _registry(tmp_path, monkeypatch, [ev], NOW)
    # The model gets it right: an ISO date passes through untouched.
    good = reg.call("get_calendar", {"range": "2026-09-12"}, from_model=True,
                    utterance="what do i have on september 12th")
    assert "BIOSENSORS" in good.text
    # The model gets it WRONG -- "today", the exact old failure -- and the
    # deriver overrules it from what he actually said.
    fixed = reg.call("get_calendar", {"range": "today"}, from_model=True,
                     utterance="what do i have on september 12th")
    assert "BIOSENSORS" in fixed.text
    # No date in the utterance: the model's own value stands, so a
    # follow-up it resolved from context ("and the next day") is not stolen.
    kept = reg.call("get_calendar", {"range": "week"}, from_model=True,
                    utterance="and the next day")
    assert "this week" in kept.text        # the week, not the 12th
    assert "BIOSENSORS" not in kept.text   # which is a week out, so absent


def test_an_iso_range_is_a_fixed_point_of_coerce_range():
    """Both paths coerce, the forced one twice.  A second pass must not
    move the day -- and must not move it a year on, either."""
    for iso in ("2026-09-12", "2026-09-05", "2027-01-05", "2024-09-12"):
        assert coerce_range(iso, NOW) == iso
        assert coerce_range(coerce_range(iso, NOW), NOW) == iso


def test_an_ask_is_a_fixed_point_too():
    once = coerce_range("what do i have on september 31st", NOW)
    assert coerce_range(once, NOW) == once


def test_an_unreadable_date_asks_instead_of_answering(tmp_path, monkeypatch):
    reg, _src = _registry(tmp_path, monkeypatch, [], NOW)
    res = reg.call("get_calendar", {"range": "september 31st"})
    assert res.ok is False
    assert res.speak and res.speak.endswith("?")
    assert "Nothing on today" not in res.text


def test_a_date_past_the_window_reaches_him_as_a_refusal(tmp_path, monkeypatch):
    reg, _src = _registry(tmp_path, monkeypatch, [], NOW)
    res = reg.call("get_calendar", {"range": "2026-10-03"})
    assert res.ok is False
    assert "Saturday the 3rd of October" in res.text
    assert "Nothing" not in res.text


# ------------------------------------------------------- the tool schema
def test_the_schema_lets_the_model_send_a_date():
    """An enum of the eleven values made a date impossible for the model to
    express -- half the bug lived here, not in coerce_range."""
    spec = [t for t in make_tools(_Cfg(), SimpleNamespace())
            if t.name == "get_calendar"][0]
    rng = spec.parameters["properties"]["range"]
    assert "enum" not in rng
    assert "2026-09-12" in rng["description"]
    assert spec.description_words() <= 20          # it ships on every turn
    assert "what's on today" in spec.description   # the gemma4 routing fix
    assert spec.derive is not None


# ------------------------------------------------------- no silent today
def test_the_source_has_no_bare_return_today_on_an_unrecognised_path():
    """The proof asked for: grep the module.  ``coerce_range`` may return
    "today" from exactly ONE line, and that line is reached only when
    nothing date-shaped was said."""
    import inspect

    src = inspect.getsource(calendar.coerce_range)
    assert src.count('return "today"') == 1
    body = src.split('return "today"')[0]
    assert "_DATEISH_RX" in body, "the last line must be guarded by the date guard"


@pytest.mark.parametrize("said", [
    "what do i have on september 12th", "anything on the 12th",
    "what about october 3rd", "what is on my calendar on the 20th",
    "do i have anything on sept 12", "what do i have on 9/12",
    "what about the 15th of october", "anything on friday the 20th",
    "what do i have on september 31st", "anything on 13/5",
    "what about the 32nd", "what did i have on the 3rd",
])
def test_nothing_date_shaped_ever_becomes_today_again(said):
    rng = coerce_range(said, NOW)
    assert rng != "today", said
    assert is_ask(rng) or as_date(rng) is not None, f"{said!r} -> {rng!r}"


# ------------------------------------------------------------ the clock
#
# The suite has just been swept for wall-clock dependence.  This one runs
# the whole table at ten instants -- midnight and one minute to midnight,
# both 2026 US DST transitions, a leap day, a year end, and two zones on
# the other side of the world -- so a date test that only passes in the
# afternoon in Chicago cannot survive here.
CLOCKS = [
    datetime(2026, 9, 5, 0, 0, tzinfo=CHI),
    datetime(2026, 9, 5, 23, 59, tzinfo=CHI),
    datetime(2026, 3, 8, 1, 30, tzinfo=CHI),               # spring forward
    datetime(2026, 3, 8, 3, 30, tzinfo=CHI),
    datetime(2026, 11, 1, 1, 30, tzinfo=CHI),              # fall back
    datetime(2026, 12, 31, 23, 59, tzinfo=CHI),
    datetime(2028, 2, 29, 12, 0, tzinfo=CHI),              # a leap day
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("Pacific/Auckland")),
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    datetime(2026, 9, 5, 9, 0, tzinfo=ZoneInfo("UTC")),
]


@pytest.mark.parametrize("now", CLOCKS)
def test_any_hour_any_date_any_timezone(now):
    """The invariants, restated so they hold at every instant above: a
    resolved date is real, forward for a present-tense question, backward
    for a past-tense one, and the eleven old values are untouched."""
    today = now.date()
    for said, _ in HIS_PHRASINGS:
        rng = coerce_range(said, now)
        day = as_date(rng)
        assert day is not None, f"{said!r} at {now} -> {rng!r}"
        assert day >= today, f"{said!r} at {now} -> {day} is behind {today}"
        assert (day - today).days < 400
    back = as_date(coerce_range("what did i have on the 3rd", now))
    assert back is not None and back <= today
    for word in calendar.RANGES:
        assert coerce_range(word, now) == word
    assert coerce_range("whats on my calendar", now) == "today"
    assert coerce_range("what's on my calendar tmrw", now) == "tomorrow"


@pytest.mark.parametrize("now", CLOCKS)
def test_the_day_named_is_the_day_listed(now):
    """End to end at every instant: put one event on the day the words
    resolve to, and the answer must name it -- the wrong-day failure would
    show up here as an empty day."""
    for said in ("anything on the 12th", "what do i have on september 12th"):
        rng = coerce_range(said, now)
        day = as_date(rng)
        start = datetime(day.year, day.month, day.day, 9, 10, tzinfo=now.tzinfo)
        text = format_events([_ev(start, "BIOSENSORS", 50)], rng, now,
                             window_days=4000)
        assert "BIOSENSORS" in text, f"{said!r} at {now}: {text}"
