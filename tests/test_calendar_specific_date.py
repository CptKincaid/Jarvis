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
* a date outside the window the cache actually holds is REFUSED by name,
  never answered "nothing on it" (the window is 45 days since 2026-09-05;
  see tests/test_calendar_window_widening.py for why it could not simply be
  widened);
* a date that cannot be read -- the 31st of September, the 32nd, 13/5, a
  weekday that does not fall on the day named -- is ASKED ABOUT.  Nothing
  unrecognised may become "today" again.

ROUND TWO (same day, after a 180-phrasing grid): a bare ordinal became a
date in twenty of thirty-six NON-date rows, an explicit year was dropped, an
offset phrase was silently ignored, word ordinals and dash dates were still
silently today, and the two doors disagreed on eight of his own words.  All
four are pinned at the end of this file, as CLASSES rather than examples --
the round before pinned four ordinal phrases and shipped twenty.

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
# 2026-09-04 (yesterday) through 2026-10-19 -- CalendarSource._window
# anchors at yesterday-midnight and reaches WINDOW_DAYS + 1 days, and
# WINDOW_DAYS is 45.
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
# DECISION: the cache holds yesterday through today + WINDOW_DAYS - 1 and
# nothing else, so a date outside it is refused with its own name and the
# edge date, never answered empty.  WINDOW_DAYS went 14 -> 45 on 2026-09-05
# so that a date a month out -- "what about october 3rd", which he asked --
# is ANSWERED rather than honestly refused; see
# tests/test_calendar_window_widening.py for the calwatch burst that had to
# be made impossible first.
def test_the_date_a_month_out_is_answered_now_not_refused():
    """The whole point of the widening: 2026-10-03 was 28 days away, inside
    no 14-day cache and inside this one."""
    assert calendar.WINDOW_DAYS >= 45
    text = format_events([], coerce_range("what about october 3rd", NOW), NOW)
    assert text == "Nothing on Saturday the 3rd of October, sir."
    day = datetime(2026, 10, 3, 11, 0, tzinfo=CHI)
    assert "ADVISOR" in format_events([_ev(day, "ADVISOR")],
                                      coerce_range("what about october 3rd", NOW), NOW)


def test_a_future_date_beyond_the_cache_is_refused_by_name():
    text = format_events([], "2026-12-03", NOW)
    assert "Nothing" not in text
    assert "Thursday the 3rd of December" in text
    assert "Monday the 19th of October" in text          # the far edge, named
    # His own words land here, and get a straight answer about reach.
    assert format_events([], coerce_range("what about december 3rd", NOW), NOW) == text


def test_the_edges_of_the_window_are_inclusive():
    assert "Nothing on Friday the 4th, sir." == format_events([], "2026-09-04", NOW)
    assert "Nothing on Monday the 19th of October, sir." == \
        format_events([], "2026-10-19", NOW)
    for out in ("2026-09-03", "2026-10-20"):
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
    res = reg.call("get_calendar", {"range": "2026-12-03"})
    assert res.ok is False
    assert "Thursday the 3rd of December" in res.text
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


# ============================================================ round two
#
# Four holes measured on a 180-phrasing grid at this same injected Saturday,
# after the build above shipped.  Each is another instance of the ONE failure
# that matters -- a confident answer about the wrong day -- so none of them
# is cosmetic.


# ------------------------------- (1) a non-date must not become a date
#
# TWENTY of thirty-six non-date rows were read as dates.  The mechanism was
# ``_D_ORD_RX``: it took a bare ordinal as a date after "on", "for" or "the",
# and "the" there is the ARTICLE.  "when is the 2nd lab on my calendar" was
# answered "I only hold the calendar out to Friday the 18th, sir; Friday the
# 2nd of October is past that."
#
# The build pinned FOUR such phrases and shipped twenty, which is what
# pinning examples instead of a class buys.  This is the class: ranks, rooms
# and floors, gates and flights, versions, quantities, durations, and plain
# numbers that were never ordinals at all.  A rank is "the Nth <noun>"; a
# date is a bare ordinal with nothing after it but punctuation or a word that
# cannot be the noun a rank counts.
NOT_DATES = [
    # -- ranks -----------------------------------------------------------
    "what's my 2nd class",
    "when is the 2nd lab on my calendar",
    "how long is my 1st meeting",
    "when is the 3rd lecture",
    "what's the 4th session about",
    "what's the 2nd item on my list",
    "is the 1st period cancelled",
    "read me the 5th question",
    "it worked on the 2nd try",
    "that's for the 3rd time",
    "the 2nd half starts soon",
    "where's the 1st draft",
    "the 21st century",
    "how were the 3rd quarter results",
    "who came in 2nd",
    "the 6th man award",
    "what's the 2nd best option",
    "the 1st amendment",
    "the 7th grade parents evening",
    "the 3rd person in line",
    # -- rooms, floors, gates, flights -----------------------------------
    "the 3rd floor conference room",
    "meet me on the 2nd floor",
    "boarding at the 4th gate",
    "the 12th floor lab",
    "room 12 on the 3rd level",
    "gate 5 for the 1st flight",
    "the 1st flight out",
    "the 2nd leg of the trip",
    "flight 1204 lands tonight",
    "room 305 is booked",
    "gate 12 closes early",
    # -- versions --------------------------------------------------------
    "the 2nd version of the doc",
    "open the 3rd revision",
    "python 3 the 2nd edition",
    "version 2 of the plan",
    # -- quantities and durations ----------------------------------------
    "the 2nd of three parts",
    "the 1st of many",
    "3 hours on the 2nd shift",
    "i need the 4th copy",
    "the 1st hour of the meeting",
    "for the 3rd week running",
    "the 2nd day of class",
    "the 90th minute",
    "the 100th day of term",
    "wait a second",
    "give me one second",
    # -- a month word doing a month word's other job ---------------------
    # Found on this round's own grid, and the same class: "may 3 people
    # come" was answered about the 3rd of May.  A month name beside a BARE
    # number obeys the tail rule too; an ordinal or a year settles it.
    "may 3 people come",
    "march 3 people in",
    "maybe 3 things",
    "he finished 2nd of 40",
    "chapter 3 of 12",
    "page 2 of 5",
]


@pytest.mark.parametrize("said", NOT_DATES)
def test_a_non_date_never_becomes_a_date(said):
    """No row here names a day, so none may resolve to one -- and none may
    ASK either: a question on "what's my 2nd class" is its own wrong answer."""
    rng = coerce_range(said, NOW)
    assert as_date(rng) is None, f"{said!r} -> {rng!r}"
    assert not is_ask(rng), f"{said!r} -> {rng!r}"


def test_the_non_date_table_covers_the_classes_it_claims_to():
    """A guard on the guard: forty-odd rows, and the shapes named above."""
    assert len(NOT_DATES) >= 40
    assert len(set(NOT_DATES)) == len(NOT_DATES)


@pytest.mark.parametrize("said,want", [
    ("what's on the 12th of this month", "2026-09-12"),
    ("what do i have on the 1st of next month", "2026-10-01"),
    ("what did i have on the 3rd of last month", "2026-08-03"),
    ("anything on the 12th of the month", "2026-09-12"),
    ("what's on the day after the 12th of this month", "2026-09-13"),
])
def test_a_month_named_without_naming_it(said, want):
    """Also found on this round's grid, also a silent TODAY: "the 12th of
    this month" carries a month the tail rule reads as a rank ("the 2nd of
    three parts") and _D_DM_RX cannot see, because there is no month NAME."""
    assert coerce_range(said, NOW) == want, said


def test_a_month_name_beside_a_bare_number_still_reads_as_a_date():
    """The tail rule on _D_MD_RX must not cost him the phrasing he used."""
    for said, want in [("do i have anything on sept 12", "2026-09-12"),
                       ("anything on sep. 12", "2026-09-12"),
                       ("what's on september 12 at 3", "2026-09-12"),
                       ("anything on september 12 2027", "2027-09-12"),
                       ("what about september 12th class", "2026-09-12")]:
        assert coerce_range(said, NOW) == want, said


def test_the_ordinal_phrasings_he_actually_used_still_read_as_dates():
    """The tail rule must not buy the table above with his own sentences."""
    for said, want in HIS_PHRASINGS:
        assert coerce_range(said, NOW) == want, said
    for said, want in [("anything on the 12th at 3", "2026-09-12"),
                       ("what's on the 12th and the 13th", "2026-09-12"),
                       ("is there anything on the 12th please", "2026-09-12"),
                       ("what's on the 12th, sir", "2026-09-12"),
                       ("anything on the 12th this month", "2026-09-12")]:
        assert coerce_range(said, NOW) == want, said


# --------------------------- (2) a date must not resolve to another day
def test_an_explicit_year_after_a_month_name_is_not_eaten_as_the_day():
    """MEASURED before the fix: "on 12 september 2027" -> 2026-09-20.  The
    day regex ate the first two digits of the YEAR ("20"), the ordinal and
    the year were both optional, and the year he said was dropped."""
    assert coerce_range("on 12 september 2027", NOW) == "2027-09-12"
    assert coerce_range("what do i have on 12 september 2027", NOW) == "2027-09-12"
    assert coerce_range("anything on september 12 2027", NOW) == "2027-09-12"
    assert coerce_range("anything on 12th september 2027", NOW) == "2027-09-12"
    # and the year-less forms still resolve by the one-rule year inference
    assert coerce_range("on 12 september", NOW) == "2026-09-12"
    assert coerce_range("anything on september 2027", NOW) != "2026-09-20"


@pytest.mark.parametrize("said,want", [
    ("what's on the day after the 12th", "2026-09-13"),
    ("what's on the day before the 12th", "2026-09-11"),
    ("anything a week after the 12th", "2026-09-19"),
    ("anything a week before the 12th", "2026-09-05"),
    ("what's on two days after the 12th", "2026-09-14"),
    ("what's on three days before september 12th", "2026-09-09"),
    ("what's on the day after tomorrow", "2026-09-07"),
    ("what's on the day before yesterday", "2026-09-03"),
    ("what's on the day after today", "2026-09-06"),
    ("what's on the day after monday", "2026-09-08"),
    ("what did i have the day before the 3rd", "2026-09-02"),
    ("what's on 2 weeks after the 12th", "2026-09-26"),
])
def test_an_offset_is_understood_not_dropped(said, want):
    """MEASURED before the fix: "the day after the 12th" -> the 12th.  The
    offset was found by no rule and silently discarded, which is the whole
    complaint in miniature -- a confident answer about a day he had just
    stepped away from."""
    assert coerce_range(said, NOW) == want, said


@pytest.mark.parametrize("said", [
    "what's on the day after the exam",
    "anything two days before the deadline",
    "what's on the week after next",
])
def test_an_offset_whose_base_is_unnameable_asks(said):
    """Understood as an offset, unable to name what it is an offset FROM.
    An ask; never the base day, and never today."""
    rng = coerce_range(said, NOW)
    assert is_ask(rng), f"{said!r} -> {rng!r}"
    assert ask_words(rng).endswith("?")


# ------------------- (3) an unreadable date must not still be today
@pytest.mark.parametrize("said,want", [
    ("what do i have on september twelfth", "2026-09-12"),
    ("anything on the twelfth", "2026-09-12"),
    ("what about october third", "2026-10-03"),
    ("what is on my calendar on the twentieth", "2026-09-20"),
    ("anything on the twenty-third", "2026-09-23"),
    ("anything on the twenty third", "2026-09-23"),
    ("what's on the thirty-first of october", "2026-10-31"),
    ("anything on the first", "2026-10-01"),
    ("what did i have on the second", "2026-09-02"),
    ("what's on september twenty-ninth", "2026-09-29"),
])
def test_an_ordinal_said_as_a_word_reads_as_a_date(said, want):
    """MEASURED before the fix: "on september twelfth" -> today, answered
    "Today: ...".  Every date regex wanted digits."""
    assert coerce_range(said, NOW) == want, said


@pytest.mark.parametrize("said,want", [
    ("what do i have on 9-12", "2026-09-12"),
    ("what do i have on 9.12", "2026-09-12"),
    ("anything on 9-12-2027", "2027-09-12"),
    ("what's on 12-9", "2026-12-09"),
])
def test_a_dash_separated_date_reads_as_a_date(said, want):
    """MEASURED before the fix: "on 9-12" -> today.  A bare "9-12" really is
    a time range more often than a date; one he put "on" in front of is not."""
    assert coerce_range(said, NOW) == want, said


@pytest.mark.parametrize("said", [
    "i'm free 9-12",                  # no date cue: a time range, left alone
    "the lab runs 2-4",
    "on 3-4 hours of sleep",          # the tail rule: a duration, not a date
])
def test_a_dash_pair_that_is_not_a_date_is_not_one(said):
    rng = coerce_range(said, NOW)
    assert as_date(rng) is None, f"{said!r} -> {rng!r}"
    assert not is_ask(rng), f"{said!r} -> {rng!r}"


def test_a_dash_pair_that_could_be_a_clock_asks():
    rng = coerce_range("what do i have on 9-12 from nine", NOW)
    assert is_ask(rng), rng
    assert "9-12" in ask_words(rng) and ask_words(rng).endswith("?")


@pytest.mark.parametrize("said", [
    "what do i have on september thirty-first",   # September has 30 days
    "anything on february thirtieth",
    "what about the 32nd",
    "anything on 13-13",
])
def test_a_word_or_dash_date_that_cannot_be_read_still_asks(said):
    rng = coerce_range(said, NOW)
    assert is_ask(rng), f"{said!r} -> {rng!r}"
    assert rng != "today"
    assert ask_words(rng).endswith("?")


def test_the_one_line_that_may_answer_today_is_still_the_only_one():
    """Re-run of the grep guard, after the round-two rewrite: coerce_range
    may return "today" from exactly ONE line, still behind the date guard."""
    import inspect

    src = inspect.getsource(calendar.coerce_range)
    assert src.count('return "today"') == 1
    body = src.split('return "today"')[0]
    assert "_DATEISH_RX" in body, "the last line must be guarded by the date guard"


# ------------------------------ (4) the two doors must not disagree
#
# MEASURED before the fix: "what was on my calendar yesterday" was the 4th
# through the forced door (commander.calendar_range -> coerce_range) and
# TODAY through the model door, on the same eight words.  The CAUSE was two
# readers: the model door's deriver called ``_explicit_date``, which knows
# nothing of "yesterday" or of an offset, so it derived nothing and the
# model's own wrong ``range`` stood.  There is now ONE reader,
# ``calendar.sentence_date``, and this pins that both doors use it.
DOOR_ROWS = HIS_PHRASINGS + [
    ("what was on my calendar yesterday", "2026-09-04"),
    ("what did i have yesterday", "2026-09-04"),
    ("what's on the day after the 12th", "2026-09-13"),
    ("what do i have on september twelfth", "2026-09-12"),
    ("what do i have on 9-12", "2026-09-12"),
    ("on 12 september 2027", "2027-09-12"),
    ("what did i have on the 3rd", "2026-09-03"),
]


def _deriver(monkeypatch):
    monkeypatch.setattr(calendar, "now_local", lambda tz=None: NOW)
    spec = [t for t in make_tools(_Cfg(), SimpleNamespace())
            if t.name == "get_calendar"][0]
    return spec.derive


@pytest.mark.parametrize("said,want", DOOR_ROWS)
def test_both_doors_read_the_same_day_off_the_same_words(said, want, monkeypatch):
    from jarvis.commander import calendar_range

    derive = _deriver(monkeypatch)
    assert calendar_range(said, NOW) == want, f"forced: {said!r}"
    assert derive(said) == {"range": want}, f"model: {said!r}"


@pytest.mark.parametrize("said", NOT_DATES + [
    "what do i have on september 31st", "anything on 13/5", "what about the 32nd",
    "what's on the day after the exam", "whats on my calendar",
    "what's on my calendar tmrw", "and the next day", "what's on monday",
])
def test_the_doors_agree_on_every_row_of_the_table(said, monkeypatch):
    """The invariant, stated over the whole non-date table plus the asks and
    the word ranges: whenever the forced door names a DAY or asks a
    QUESTION, the model door derives exactly that; and whenever the model
    door derives anything at all, the forced door agrees with it."""
    from jarvis.commander import calendar_range

    derive = _deriver(monkeypatch)
    forced = calendar_range(said, NOW)
    derived = derive(said)
    if as_date(forced) is not None or is_ask(forced):
        assert derived == {"range": forced}, f"{said!r}: {forced!r} vs {derived!r}"
    if derived:
        assert derived["range"] == forced, f"{said!r}: {forced!r} vs {derived!r}"


def test_yesterday_reaches_the_tool_the_same_way_through_the_model_door(
        tmp_path, monkeypatch):
    """End to end, the exact eight words that disagreed: one event on
    yesterday, and the model sending the wrong range."""
    ev = _ev(datetime(2026, 9, 4, 10, 0, tzinfo=CHI), "Dentist")
    reg, _src = _registry(tmp_path, monkeypatch, [ev], NOW)
    said = "what was on my calendar yesterday"
    fixed = reg.call("get_calendar", {"range": "today"}, from_model=True,
                     utterance=said)
    assert "Dentist" in fixed.text and "Friday the 4th" in fixed.text
    assert fixed.text == format_events([ev], coerce_range(said, NOW), NOW)


@pytest.mark.parametrize("now", CLOCKS)
def test_the_doors_agree_at_every_instant(now, monkeypatch):
    """The clock axis, restated for round two: the agreement is not a
    property of Saturday afternoon in Chicago."""
    from jarvis.commander import calendar_range

    monkeypatch.setattr(calendar, "now_local", lambda tz=None: now)
    spec = [t for t in make_tools(_Cfg(), SimpleNamespace())
            if t.name == "get_calendar"][0]
    for said in [r[0] for r in DOOR_ROWS] + NOT_DATES:
        forced = calendar_range(said, now)
        derived = spec.derive(said)
        if as_date(forced) is not None or is_ask(forced):
            assert derived == {"range": forced}, f"{said!r} at {now}"
        if derived:
            assert derived["range"] == forced, f"{said!r} at {now}"
