"""Calendar tool (spec 2026-08-26, section 6.4): read-only, zero OAuth.

Sources: the Google "secret address in iCal format" URLs in
``cfg.google_ical_urls`` (conditional GET every 10 min, parsed with
``icalendar`` + ``recurring_ical_events``) and iCloud through ``caldav``
(app-specific password; principal -> calendars -> search(expand=True)).
Both merge into ``Event`` rows in local time.  ``CalendarSource`` owns a
refresh thread and a disk cache (``~/.cache/jarvis/calendar_cache.json``
with per-source ``fetched_at``) so ``get_calendar`` never fetches on the
model's thread: it answers from the cache and, when that is older than 10
minutes, kicks a background refresh and says "as of 9:10 am".

Every HTTP request of this module goes through ``_fetch``; the CalDAV
client is injected through ``dav_client`` (a factory) for tests.  Secret
URLs and passwords are never logged or written to the cache.
"""
from __future__ import annotations

import builtins
import difflib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.location import (cache_dir, cfg_get, clock_words, http_get,
                                   setup_line, system_tz)
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.calendar")

REFRESH_S = 600                 # refresh period and the "stale" threshold
FETCH_TIMEOUT = 8
# A feed that has stopped answering must not be retried at full price for
# the rest of the day.  LIVE 2026-08-31: the Canvas subscription
# ("google-2") timed out on every refresh from 12:40 to 14:54 -- sixteen
# times, FETCH_TIMEOUT seconds each.  Six of those sixteen were OFF-cycle,
# kicked by get_calendar itself: a failed source keeps its old stamp,
# ``fetched_at`` was the MINIMUM over sources, so one dead feed made the
# whole cache read stale forever and every calendar answer both triggered
# another refresh and was labelled "as of 12:40 pm" although the events it
# named were seconds old.  The back-off doubles from one refresh period,
# so a blip is retried on the next cycle and an all-day outage is tried
# once an hour; ONE success clears it.
SOURCE_BACKOFF_S = REFRESH_S
MAX_SOURCE_BACKOFF_S = 3600
# How far ahead the cache reaches.  14 until 2026-09-05, which made "what
# about october 3rd" -- a day he genuinely has -- land on the right date and
# then be refused for reach, which is honest and still not an answer.  45
# covers the rest of a term.  MEASURED 2026-09-05 on two synthetic feeds of
# one event a day: a refresh goes 8.4 ms -> 9.7 ms and the cache file 5.1 KB
# -> 15.6 KB (28 events -> 90), once per ten minutes.  Nothing.
# The real cost was never the bytes: calwatch diffs two
# snapshots, so the first refresh after the widening could announce every
# event past the old edge as a new booking -- 31 of them, measured, in
# tests/test_calendar_window_widening.py.  CalendarSource.covered_window()
# is what makes that impossible; do not widen this without it.
WINDOW_DAYS = 45
# Named weekdays are ranges too. Without them "what's on my agenda for
# Monday?" had nowhere to land and the model fell back to "next", which
# answers with a single event -- seen 2026-08-28 when Monday held four.
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday")
RANGES = ("today", "tomorrow", "week", "next") + WEEKDAYS
ICLOUD_URL = "https://caldav.icloud.com"
CACHE_VERSION = 1


def _fetch(url: str, timeout: float = FETCH_TIMEOUT, headers: Optional[dict] = None) -> bytes:
    """Test seam: every request of this module goes through here."""
    return http_get(url, timeout=timeout, headers=headers)


# ------------------------------------------------------------------ Event
@dataclass
class Event:
    start: datetime             # aware, local time; midnight for all-day
    end: datetime               # exclusive; next midnight for all-day
    all_day: bool = False
    title: str = ""
    calendar: str = ""
    location: str = ""
    # The ICS DESCRIPTION, whitespace-collapsed and capped. Carried for ONE
    # reason: half the world puts the meeting link in the body and leaves
    # LOCATION empty (his Canvas feed does), so the dossier's JOIN row was
    # blank for exactly the classes that need it most.
    description: str = ""

    def key(self) -> tuple:
        return (self.start.isoformat(), self.end.isoformat(), self.title.lower(),
                self.all_day)

    def meeting_url(self) -> str:
        """The link to click for this event, or "".

        LOCATION first -- an organiser who put the URL there meant it -- then
        the description. Never any old URL from the body: see meeting_url()."""
        if str(self.location or "").strip().lower().startswith(("http://", "https://")):
            return self.location.strip()
        return meeting_url(self.description)

    def on(self, day: date) -> bool:
        """True when the event touches ``day`` (all-day spans included)."""
        if self.all_day:
            return self.start.date() <= day < max(self.end.date(),
                                                  self.start.date() + timedelta(days=1))
        return self.start.date() == day

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "all_day": self.all_day, "title": self.title,
                "calendar": self.calendar, "location": self.location,
                "description": self.description}

    @classmethod
    def from_dict(cls, d: dict, tz=None) -> "Event":
        start = datetime.fromisoformat(d["start"])
        end = datetime.fromisoformat(d["end"])
        if tz is not None:
            start, end = start.astimezone(tz), end.astimezone(tz)
        return cls(start=start, end=end, all_day=bool(d.get("all_day")),
                   title=str(d.get("title") or ""),
                   calendar=str(d.get("calendar") or ""),
                   location=str(d.get("location") or ""),
                   # Absent from every cache file written before 2026-08-31;
                   # a missing key is an event with no body, not a crash.
                   description=str(d.get("description") or ""))


# The hosts a link has to belong to before Jarvis will call it "the join
# link". An event body is full of URLs -- the Canvas assignment, a syllabus
# PDF, an unsubscribe footer -- and reading one of those out as the way into
# a lecture is worse than saying nothing at all, so this is an allow-list of
# conferencing hosts rather than "the first http:// in the description".
MEETING_HOSTS = (
    "zoom.us", "zoom.com", "teams.microsoft.com", "teams.live.com",
    "meet.google.com", "webex.com", "whereby.com", "gotomeeting.com",
    "bluejeans.com", "chime.aws", "meet.jit.si",
)
# Trailing punctuation an organiser's prose leaves stuck to a URL: "join at
# https://tamu.zoom.us/j/123." must not carry the full stop into the link.
# Brackets close pairs, so they are trimmed here too.
_URL_TAIL = ">).,;:!?\"'”’]}"
_URL_RX = re.compile(r"https?://[^\s<>\"']+", re.I)
DESCRIPTION_CHARS = 800        # enough for the join block; not the whole email


def meeting_url(text) -> str:
    """The first conferencing link in ``text``, or "".

    ICS bodies arrive with escaped newlines and commas (RFC 5545 folds and
    escapes them), and Google/Zoom both wrap the URL in prose, so the
    matching is done over the raw text and the punctuation is trimmed off
    the end rather than trying to parse the body's structure.
    """
    raw = str(text or "").replace("\\n", " ").replace("\\,", ",")
    for hit in _URL_RX.findall(raw):
        url = hit.rstrip(_URL_TAIL)
        host = url.split("//", 1)[-1].split("/", 1)[0].split("@")[-1].lower()
        if any(host == h or host.endswith("." + h) for h in MEETING_HOSTS):
            return url
    return ""


def now_local(tz=None) -> datetime:
    """The current wall clock, as one module-level seam.  Tests freeze this
    instead of the ``datetime`` name, which ``isinstance`` checks rely on."""
    return datetime.now(tz or system_tz())


def _clean(text) -> str:
    return " ".join(str(text or "").split())


def trim_description(text) -> str:
    """The body, whitespace-collapsed and capped -- with the join link kept
    even when it falls past the cap.

    A Canvas or Outlook body opens with a paragraph of prose and puts the
    Zoom block underneath, which is precisely where a blind [:800] would cut
    the one thing this field is carried for.
    """
    body = _clean(text)
    if len(body) <= DESCRIPTION_CHARS:
        return body
    url = meeting_url(body)
    cut = body[:DESCRIPTION_CHARS]
    if url and url not in cut:
        cut = f"{cut[:max(0, DESCRIPTION_CHARS - len(url) - 1)].rstrip()} {url}"
    return cut


def _localize(value, tz) -> datetime:
    """date / naive / aware -> aware datetime in ``tz``."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=tz)
        return value.astimezone(tz)
    return datetime(value.year, value.month, value.day, tzinfo=tz)


def parse_ics(raw, start: datetime, end: datetime, calendar: str = "",
              tz=None) -> list[Event]:
    """Every occurrence (recurrences expanded) between ``start`` and
    ``end`` as local-time Events.  Cancelled events are dropped."""
    import icalendar
    import recurring_ical_events

    tz = tz or start.tzinfo or system_tz()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    cal = icalendar.Calendar.from_ical(raw)
    name = calendar or _clean(cal.get("X-WR-CALNAME", ""))
    out: list[Event] = []
    query = recurring_ical_events.of(cal, skip_bad_series=True)
    for comp in query.between(start, end):
        if _clean(comp.get("STATUS", "")).upper() == "CANCELLED":
            continue
        ds = comp.get("DTSTART")
        if ds is None:
            continue
        ds = ds.dt
        de = comp.get("DTEND")
        if de is not None:
            de = de.dt
        elif comp.get("DURATION") is not None:
            de = ds + comp.get("DURATION").dt
        else:
            de = ds + timedelta(days=1) if not isinstance(ds, datetime) else ds
        all_day = not isinstance(ds, datetime)
        s, e = _localize(ds, tz), _localize(de, tz)
        if all_day:
            s = s.replace(hour=0, minute=0, second=0, microsecond=0)
            e = e.replace(hour=0, minute=0, second=0, microsecond=0)
            if e <= s:
                e = s + timedelta(days=1)
        elif e < s:
            e = s
        out.append(Event(start=s, end=e, all_day=all_day,
                         title=_clean(comp.get("SUMMARY", "")) or "untitled",
                         calendar=name, location=_clean(comp.get("LOCATION", "")),
                         # Capped: the body can be a whole invitation email,
                         # and every event of the window is held in memory
                         # and written to the disk cache.
                         description=trim_description(comp.get("DESCRIPTION", ""))))
    return out


def merge_events(*groups) -> list[Event]:
    """Union, de-duplicated, all-day first within a day, then by start."""
    seen, out = set(), []
    for group in groups:
        for ev in group:
            k = ev.key()
            if k in seen:
                continue
            seen.add(k)
            out.append(ev)
    out.sort(key=lambda e: (e.start.date(), not e.all_day, e.start, e.title.lower()))
    return out


# --------------------------------------------------------------- wording
# Below this an event is a marker, not a booking, and "for 1 minute" is
# noise. Anything from a quarter of an hour up is a real length.
MIN_DURATION_MINUTES = 5


def _duration_words(ev: Event) -> str:
    minutes = int((ev.end - ev.start).total_seconds() // 60)
    if minutes < MIN_DURATION_MINUTES:
        return ""
    # A sub-hour event used to return "" -- so "how long is my biosensors
    # lab?" had NOTHING to answer from for a 50-minute class (his 9:10 am
    # BIOSENSORS lecture is 9:10-10:00), and Jarvis said "I'm afraid I don't
    # have that information, sir" about a length both ends of which were in
    # the event. The duration is derivable from start and end; say it.
    if minutes < 60:
        return f" for {minutes} minutes"
    hours, rem = divmod(minutes, 60)
    if rem == 0:
        return " for an hour" if hours == 1 else f" for {hours} hours"
    if rem == 30:
        return " for an hour and a half" if hours == 1 else f" for {hours} and a half hours"
    return f" for about {hours + (1 if rem > 30 else 0)} hours" if rem > 15 else \
        (" for an hour" if hours == 1 else f" for {hours} hours")


def _event_words(ev: Event, with_time: bool = True) -> str:
    if ev.all_day:
        text = f"all day: {ev.title}"
    else:
        text = f"{clock_words(ev.start)} {ev.title}{_duration_words(ev)}" if with_time \
            else f"{ev.title}{_duration_words(ev)}"
    if ev.location:
        text += f" at {ev.location}"
    return text


def _day_events(events, day: date) -> list[Event]:
    return [e for e in events if e.on(day)]


def day_label(day: date, today: date) -> str:
    """"today" / "tomorrow" / "Friday". Public because the console's room
    slab (jarvis/app.py _room_next_event) labels its NEXT row with it: the
    panel and the spoken answer must not word the same day differently."""
    return _day_label(day, today)


def _day_label(day: date, today: date) -> str:
    if day == today:
        return "today"
    if day == today + timedelta(days=1):
        return "tomorrow"
    return day.strftime("%A")


def describe_due(due: datetime, now: datetime) -> str:
    """"in 10 minutes" / "in 2 hours" / "at 7:00 am tomorrow" / "on Friday
    at 9:00 am" — the wording section 6.1 gives the timekeeper."""
    delta = (due - now).total_seconds()
    if delta < 60:
        return "now"
    minutes = int(delta // 60)
    if minutes < 60:
        return f"in {minutes} minute{'s' if minutes != 1 else ''}"
    if delta < 6 * 3600:
        hours, rem = divmod(minutes, 60)
        if rem >= 45:
            hours += 1
            rem = 0
        half = " and a half" if 15 <= rem < 45 else ""
        if hours == 1:
            return f"in an hour{half}"
        return f"in {hours}{half} hours"
    day = _day_label(due.date(), now.date())
    if day in ("today", "tomorrow"):
        return f"at {clock_words(due)} {day}"
    if (due.date() - now.date()).days < 7:
        return f"on {day} at {clock_words(due)}"
    return f"on the {due.day}{_suffix(due.day)} of {due.strftime('%B')} at {clock_words(due)}"


def _suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# Every way "tomorrow" reaches this function, INCLUDING typed shorthand.
# LIVE 2026-08-31 20:38: "whats on tmws calendar" was coerced to "today",
# commander logged ``route short-cut: get_calendar({'range': 'today'})``,
# the tool handed the model TODAY's four events and Jarvis answered "I'm
# afraid I can't see tomorrow's schedule, sir; I only have access to your
# entries for today" -- about a day that is squarely inside the 14-day
# cache. The old test was ``text in ("tmrw", "tmr")``: an EQUALITY check on
# a string that is the whole utterance (commander.calendar_range passes the
# sentence, not a word), so the abbreviation branch could never fire in a
# sentence. A word-boundary search is the whole fix.
_TOMORROW_RX = re.compile(
    r"\b(?:tomorrow|tomorow|tomarrow|2morrow|tmrw|tmrws|tmrrw|tmrs|tmr|"
    r"tmws|tmw|tmoro|tmoros)\b", re.I)


# ------------------------------------------------------- explicit dates
#
# HIS REPORT 2026-09-05: "I just tried to have Jarvis tell me about a
# specific calendar date and he didn't get it."  coerce_range understood
# ELEVEN values -- today, tomorrow, week, next and the seven weekday names
# -- and ended in a bare ``return today``, so every explicit date became
# TODAY in silence.  Measured before the fix, exactly as printed:
#
#     "what do i have on september 12th"         -> today
#     "anything on the 12th"                     -> today
#     "what about october 3rd"                   -> today
#     "what is on my calendar on the 20th"       -> today
#     "do i have anything on sept 12"            -> today
#     "what do i have on 9/12"                   -> today
#     "what about the 15th of october"           -> today
#     "anything on friday the 20th"              -> friday   (the NEXT Friday)
#
# He was not merely unanswered: he was answered CONFIDENTLY ABOUT THE WRONG
# DAY, and the last of those is the sharpest case because it looks handled.
#
# A date is carried as an ISO "YYYY-MM-DD" range string.  Both doors coerce
# -- the forced path through commander.calendar_range and again in
# get_calendar, the model path once -- so an ISO date has to be a FIXED
# POINT of this function, and it is (tests pin the second pass).  What
# cannot be read comes back as ASK_PREFIX + the question to put; nothing
# unreadable is turned into today ever again.
ASK_PREFIX = "ask:"
_ISO_DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3,
           "mar": 3, "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
           "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
           "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11,
           "nov": 11, "december": 12, "dec": 12}
# The twelve in full, for the near-miss the transcriber hands over
# ("nevember"): see _NEAR_MONTH_RX.
_MONTH_NAMES = ("january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november",
                "december")
# Longest first so "sept" is not eaten as "sep"; the trailing (?:\.|\b)
# accepts "sept." AND keeps "may" out of "maybe" -- a bare [a-z]* suffix
# matched "maybe 3 things" as the 3rd of May.
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH = rf"(?P<mon>{_MONTH_ALT})(?:\.|\b)"
_ORD = r"(?:st|nd|rd|th)"
# ", 2027", "of 2027" -- and "next year" / "last year", which round four
# found dropped: "september 12th next year" was September 2026.  Read
# through _year_of, never through group("y") alone.
_YEAR = (r"(?:,?\s+(?:of\s+)?(?P<y>\d{4})|\s+(?P<ry>this|next|last)\s+year)?")
# A four-digit year first can only be one day whichever way it is
# separated: 2026/09/12 is read, not asked about.
_D_ISO_RX = re.compile(r"\b(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})\b")
# (?!\d) on the DAY.  Without it "september 2027" read as SEPTEMBER THE 20th:
# \d{1,2} ate the first two digits of the year, the ordinal and the year were
# both optional, and the explicit year he had just said was dropped.  MEASURED
# 2026-09-05: "on 12 september 2027" -> 2026-09-20, wrong in both fields.
# (?:the\s+)? between month and day.  ROUND FOUR, 2026-09-06: without it
# "january the 5th" was not a month-day at all, _D_ORD_RX then read the
# bare "the 5th" and DROPPED the month -- "december the 25th" answered
# about the 25th of SEPTEMBER, and "<month> the 5th" answered TODAY twelve
# months out of twelve.  MEASURED over 12 months x {1st, 5th, 12th, 25th}:
# 44 of 48 silently wrong, right only for September.
_D_MD_RX = re.compile(
    rf"\b{_MONTH}\s+(?:the\s+)?(?P<d>\d{{1,2}})(?!\d)(?P<ord>{_ORD})?{_YEAR}",
    re.I)
_D_DM_RX = re.compile(
    rf"\b(?:the\s+)?(?P<d>\d{{1,2}})(?!\d)(?P<ord>{_ORD})?\s+(?:of\s+)?"
    rf"{_MONTH}{_YEAR}", re.I)
# THE TAIL RULE, the fix for the biggest hole of the 2026-09-05 build.
#
# "the" in "on / for / the" is the ARTICLE as often as it is a date cue, so
# the old guard let TWENTY of thirty-six non-date phrasings become dates:
# "when is the 2nd lab on my calendar" was answered "I only hold the calendar
# out to Friday the 18th, sir; Friday the 2nd of October is past that."  The
# build pinned ONE such phrase ("my 2nd class") and shipped twenty, which is
# what pinning an example instead of a class buys.
#
# The class: "the Nth <noun>" is a RANK -- a lab, a floor, a flight, a
# version, a try, an hour.  A bare ordinal is a DATE only when nothing
# follows it but punctuation or one of the small set of words below, none of
# which can be the noun a rank counts.  "of" is deliberately NOT in the set:
# "the 12th of October" is matched by _D_DM_RX before this regex is reached,
# so leaving it out costs nothing and keeps "the 2nd of three parts" a rank.
#
# "to" and "in" are tail words -- "the 12th to the 14th", "the 3rd in
# october" -- except in front of a rank of their own: "the 2nd TO LAST
# meeting" was Friday the 2nd of October through both doors, and "the 3rd
# IN A ROW" the 3rd (round four, 2026-09-06).
_ORD_TAIL = (r"at|on|in(?!\s+a\s+row\b)|and|or|for|to(?!\s+last\b)|this|next|"
             r"please|sir|then|too|instead|"
             r"is|was|are|were|do|does|did|i|my|me|we|you|there|anything|"
             r"something|else|yet|still|though|but|if|so|that|it|ok|okay|"
             r"look|looks")
_TAIL_OK = rf"(?!\s+(?!(?:{_ORD_TAIL})\b)\w)"
# The same test applied to a string rather than inline in a pattern.
_NOT_TAIL_RX = re.compile(rf"\s+(?!(?:{_ORD_TAIL})\b)\w", re.I)


def _tail_ok(rest: str) -> bool:
    """True when what follows a bare number cannot be the noun of a rank."""
    return _NOT_TAIL_RX.match(rest or "") is None
# An ordinal paired by and/or with a RANKED ordinal is a rank itself: "the
# 1st or 2nd week of october" was the 1st of October (round four).
_ORD_PAIR_RANK = (rf"(?!\s+(?:and|or)\s+(?:the\s+)?\d{{1,2}}{_ORD}\s+"
                  rf"(?!(?:{_ORD_TAIL})\b)\w)")
_D_ORD_RX = re.compile(
    rf"\b(?:on|for|the)\s+(?:the\s+)?(?P<d>\d{{1,2}}){_ORD}\b"
    rf"{_ORD_PAIR_RANK}{_TAIL_OK}", re.I)
# "the 12th OF this month" names a month without naming it.  It has its own
# rule because the tail rule above deliberately treats a following "of" as a
# rank ("the 2nd of three parts"), and _D_DM_RX only fires on a month NAME --
# so without this "what's on the 12th of this month" was silently TODAY,
# measured on the same grid as the four blockers.  The "of" is optional
# since round four: "on the 12th next month" was September the 12th.
_D_REL_MONTH_RX = re.compile(
    rf"\b(?:the\s+)?(?P<d>\d{{1,2}})(?!\d){_ORD}?\s+(?:of\s+)?"
    r"(?P<rel>this|next|last|the)\s+month\b", re.I)
# A bare ordinal takes its month from the REST of the sentence: "next month
# on the 12th", "in october on the 12th", "on the 12th in october", "what
# did i have on the 12th last month".  MEASURED 2026-09-06: every one of
# those was September the 12th, and "on the 5th next month" was TODAY --
# the tail rule let "next" follow the ordinal, and the ordinal was then
# read with no month at all.  18 of 20 relative-month phrasings wrong.
_MONTH_CTX_RX = re.compile(
    rf"\b(?:in|for|during|of|about)\s+(?:(?:this|next|last)\s+)?{_MONTH}{_YEAR}"
    r"|\b(?P<rel>this|next|last)\s+month\b", re.I)
# THE FRAME RULE (round three, 2026-09-05).  A number is a date only when
# the sentence is about WHEN: a preposition or a calendar word stands right
# in front of it.  Round two's ``_D_NUM_RX`` took a bare N/N ANYWHERE, so
# "i got 9/10 on the quiz" was September the 10th, "the score was 3/5" the
# 5th of March, "is it open 24/7" a question about the 24th of July, and
# "is my 9/10 quiz on my calendar" listed Thursday the 10th through BOTH
# doors -- nine of the grid's non-date rows, every one a pair that nothing
# framed.  A month name frames itself; a bare ordinal is framed by on / for
# / the plus the tail rule; a slashed pair needs one of THESE in front of
# it, and the tail rule still applies after it ("on 9/12 speed" is a rank).
# "got" is deliberately absent ("i got 9/10"); "have" is present ("what do
# i have 9/12").  Two strengths of frame: a PREPOSITION is strong enough
# that a pair it fronts which cannot be read is a question ("on 13/5" ->
# "did you mean the 13th of May?"); a calendar NOUN is weaker -- it reads a
# clean date ("on my calendar 9/12") and nothing else, so "is my gym on my
# calendar 24/7" is about opening hours, not the 24th of July.
_PREP_FRAME = (r"on|for|about|of|from|until|till|through|thru|by|before|"
               r"after|since|due")
_NOUN_FRAME = (r"calendar|schedule|agenda|diary|have|doing|free|busy|booked|"
               r"planned|happening|scheduled")
_DATE_FRAME = _PREP_FRAME + "|" + _NOUN_FRAME
# Month-first, because he is in Texas.
_D_NUM_RX = re.compile(
    rf"\b(?:{_DATE_FRAME})\s+(?:the\s+)?(?P<a>\d{{1,2}})\s*/\s*"
    rf"(?P<b>\d{{1,2}})(?:\s*/\s*(?P<y>\d{{2,4}}))?\b{_TAIL_OK}", re.I)
# DEFAULT TAKEN FOR HIM (round three): a dash- or dot-separated PAIR after
# on / for / about / of is a QUESTION, not a date.  Round two read "on 9-12"
# as the 12th of September; but "9-12" is a time range at least as often
# as a date, and the safe direction is a question that offers the date
# reading -- never a guess, and never the silent today it was before round
# two.  Three parts with a year ("9-12-2027") cannot be a clock and stay a
# date.  A clock word may follow ("on 9-12 from nine"), so the tail rule
# here admits the clock words too; "on 3-4 hours of sleep" is still a
# duration.  A bare pair with no frame ("i'm free 9-12") is left alone.
# DEFAULT TAKEN FOR HIM (round four): a pair behind "at" -- "at about
# 9.30" -- is a CLOCK and is left alone too; "about" frames a date only
# when "at" does not stand in front of it.
_DASH_TAIL_OK = (rf"(?!\s+(?!(?:{_ORD_TAIL}|from|between|till|until|through|"
                 rf"thru|am|pm|noon|midnight)\b)\w)")
_D_DASH_RX = re.compile(
    r"\b(?<!\bat\s)(?:on|for|about|of)\s+(?P<a>\d{1,2})\s*[-.]\s*(?P<b>\d{1,2})"
    r"(?:\s*[-.]\s*(?P<y>\d{2,4}))?\b" + _DASH_TAIL_OK, re.I)
# Date-shaped and UNREADABLE: the shapes that can only be a date and that
# every rule above declined -- a month name beside a number, an ISO shape.
# Applied in sentence_date, so BOTH doors and the day-shift rewrite meet
# it; it used to live in coerce_range alone, and "in december 2026" was a
# question through the forced door and TODAY through the model door.  A
# bare N/N is deliberately absent now: nothing framed it, so under the
# frame rule it is not date-shaped.  A bare ordinal is absent for the same
# reason (see _D_ORD_RX).
_DATEISH_RX = re.compile(
    rf"\b(?:{_MONTH_ALT})(?:\.|\b)\s*(?:the\s+)?\d{{1,2}}{_ORD}?{_TAIL_OK}|"
    rf"\b\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}\b", re.I)
# A month with no day in it -- "in december", "for october", "in september
# 2027" -- was a silent today.  It is about WHEN, it names no day the tool
# can list, so it is a question: which day.  ROUND FOUR added the month
# that nothing frames but that the sentence is plainly about -- "how does
# october look", "is october busy", "mid september", "early october" --
# and the month named without naming it: "what do i have next month".
_MONTH_ONLY_RX = re.compile(
    rf"\b(?:in|for|during|of|about|on|does|is|mid|early|late|"
    rf"(?:end|start|beginning|middle|rest)\s+of)[\s-]+"
    rf"(?:this\s+|next\s+|last\s+)?{_MONTH}{_YEAR}"
    r"|\b(?P<relm>this|next|last)\s+month\b", re.I)
# A month the transcriber nearly heard -- "the 12th of nevember" -- was a
# silent today: no rule read it, nothing was date-shaped.  Near enough to
# one of the twelve, it is a question that offers the month.  Five letters
# at least: "the 1st of many" is not the 1st of May.
_NEAR_MONTH_RX = re.compile(
    rf"\b(?:the\s+)?(?P<d>\d{{1,2}}){_ORD}?\s+of\s+(?P<w>[a-z]{{5,9}})\b"
    rf"|\b(?:in|on|for|during)\s+(?P<w2>[a-z]{{5,9}})\s+(?:the\s+)?"
    rf"(?P<d2>\d{{1,2}}){_ORD}?\b", re.I)
_WD_ALT = "|".join(WEEKDAYS)
# The WHOLE value, the way the model sends ``range`` -- "9/12", "12th", "the
# 12th", "september 12", "saturday the 12th" -- has nothing in front of it
# to frame it.  The value IS the frame: it is read as if he had said "on".
_BARE_VALUE_RX = re.compile(
    rf"^(?:(?:{_WD_ALT})\s+)?(?:the\s+)?(?:\d{{1,2}}{_ORD}?"
    rf"|\d{{1,2}}\s*[/.-]\s*\d{{1,2}}(?:\s*[/.-]\s*\d{{2,4}})?"
    rf"|(?:{_MONTH_ALT})(?:\.|\b)\s*(?:the\s+)?\d{{1,2}}{_ORD}?(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}{_ORD}?\s+(?:of\s+)?(?:{_MONTH_ALT})(?:\.|\b)(?:,?\s+\d{{4}})?)"
    r"\s*$", re.I)
# "the last friday of september", "the first monday of october", "the 2nd
# tuesday of this month".  MEASURED 2026-09-05: "the last friday of
# september" was "friday" -- the 11th, a fortnight short of the 25th he
# named -- because the weekday loop won and the rest of the phrase was
# never read.  Digit ordinals arrive already rewritten ("first" -> "1st").
_NTH_WD_RX = re.compile(
    rf"\b(?:the\s+)?(?P<n>[1-5](?:st|nd|rd|th)|last)\s+(?P<wd>{_WD_ALT})\s+"
    rf"(?:of|in)\s+(?:{_MONTH}|(?P<rel>this|next|last|the)\s+month\b){_YEAR}",
    re.I)
# "what DID I have on the 3rd" is a different question from "what DO I
# have on the 3rd" -- one looks back, one looks forward.
_PAST_RX = re.compile(r"\b(?:did|was|were|had)\b", re.I)
_YESTERDAY_RX = re.compile(r"\byesterday\b", re.I)
# Bounded scans (house rule): the widest gap between leap days is 8 years,
# and every day 1..31 falls in some month inside twelve.
_YEAR_SCAN = 12
_MONTH_SCAN = 24

# ------------------------------------------------- ordinals said as WORDS
#
# "on september twelfth" was silently TODAY as well: every date regex above
# wants digits.  Rather than teach five regexes to read English, the words
# are rewritten to digits ONCE, up front -- "twelfth" -> "12th",
# "twenty-third" -> "23rd" -- and every rule downstream, the tail rule
# included, then applies unchanged.  Nothing above 31 is rewritten, so
# "thirty-second" stays the time unit it usually is.
_ORD_UNITS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
              "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}
_ORD_WORDS = dict(_ORD_UNITS)
_ORD_WORDS.update({"tenth": 10, "eleventh": 11, "twelfth": 12, "twelth": 12,
                   "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
                   "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
                   "nineteenth": 19, "twentieth": 20, "thirtieth": 30})
for _tens, _tval in (("twenty", 20), ("thirty", 30)):
    for _unit, _uval in _ORD_UNITS.items():
        if _tval + _uval <= 31:
            _ORD_WORDS[f"{_tens} {_unit}"] = _tval + _uval
# Longest first so "twenty first" wins over "first"; [-\s]+ accepts the
# hyphen he types and the space the transcriber hears.
_ORD_WORD_RX = re.compile(
    r"\b(?:" + "|".join(sorted(_ORD_WORDS, key=len, reverse=True))
    .replace(" ", r"[-\s]+") + r")\b", re.I)


def digit_ordinals(text: str) -> str:
    """"the twenty-third" -> "the 23rd"; everything else untouched."""
    def swap(match):
        n = _ORD_WORDS[re.sub(r"[-\s]+", " ", match.group(0).lower())]
        return f"{n}{_suffix(n)}"
    return _ORD_WORD_RX.sub(swap, text or "")


# ------------------------------------------------------- offset phrases
#
# "the day after the 12th" resolved to the 12th: the offset was found by no
# rule and silently dropped, so he was answered confidently about a day he
# had explicitly stepped away from.  DECIDED 2026-09-05: an offset is
# UNDERSTOOD when its base day is one Jarvis can name, and ASKED ABOUT when
# it is not.  It is never ignored.
_COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "couple": 2, "few": 3}
_OFFSET_RX = re.compile(
    r"\b(?:the\s+)?(?:(?P<n>a|an|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|couple(?:\s+of)?|few|\d{1,3})\s+)?"
    r"(?P<unit>days?|weeks?|fortnights?|months?|years?)\s+"
    r"(?P<dir>after|before|following|preceding|prior\s+to|"
    r"later\s+than|earlier\s+than|ahead\s+of)\b\s*", re.I)
_BACKWARD_DIRS = re.compile(r"before|preceding|prior|earlier", re.I)
# "the friday after next" is the Friday after the coming one; "the saturday
# after the 12th" the first Saturday past that day.  MEASURED 2026-09-06:
# the first was "friday" -- the 11th, a week short of the 18th he named --
# because the weekday loop won; the second was the 12th itself.
_WD_AFTER_NEXT_RX = re.compile(
    rf"\b(?:the\s+)?(?P<wd>{_WD_ALT})\s+after\s+next\b", re.I)
_WD_OFFSET_RX = re.compile(
    rf"\b(?:the\s+)?(?P<wd>{_WD_ALT})\s+(?P<dir>after|before|following|"
    r"preceding)\s+", re.I)
# "in two days", "three days ago", "a week on tuesday", "two days from now":
# a count of days or weeks stepped from today or from a day he can name.
# MEASURED 2026-09-05: "in two days" was a silent today, and "a week on
# tuesday" the coming Tuesday -- the 8th, seven days short of the 15th.
# The COUNT is required: "in the next few days" is not an offset and "in 20
# minutes" is not a day.  ROUND FOUR: a month and a year are units too --
# "in a month", "in two months", "a month from now", "in a year" were all
# a silent today (5 of 5) -- and the count may run to "100 days".
_COUNT_ALT = (r"a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
              r"couple(?:\s+of)?|few|\d{1,3}")
_UNIT = r"(?P<unit>days?|weeks?|fortnights?|months?|years?)"
_IN_RX = re.compile(
    rf"\bin\s+(?:a\s+)?(?P<n>{_COUNT_ALT})\s+{_UNIT}(?:'s|s')?(?:\s+time)?\b",
    re.I)
_AGO_RX = re.compile(rf"\b(?:a\s+)?(?P<n>{_COUNT_ALT})\s+{_UNIT}\s+ago\b", re.I)
_FROM_RX = re.compile(
    rf"\b(?:a\s+)?(?P<n>{_COUNT_ALT})\s+{_UNIT}\s+(?:from|on)\s+", re.I)
_DAY_WORD_RX = re.compile(
    rf"\b(?P<w>today|tonight|now|yesterday|{'|'.join(WEEKDAYS)})\b", re.I)


def is_ask(range) -> bool:
    """True when coerce_range gave back a QUESTION rather than a day."""
    return isinstance(range, str) and range.startswith(ASK_PREFIX)


def ask_words(range) -> str:
    """The question to put to him, or ""."""
    return range[len(ASK_PREFIX):] if is_ask(range) else ""


def _ask(question: str) -> str:
    return ASK_PREFIX + question


def as_date(range) -> Optional[date]:
    """The day an ISO range names, or None for the word ranges and asks."""
    if isinstance(range, str) and _ISO_DATE_RX.match(range):
        try:
            return date.fromisoformat(range)
        except ValueError:                  # 2026-02-30 and friends
            return None
    return None


def _month_name(month: int) -> str:
    return date(2000, month, 1).strftime("%B")


def _year_of(match, today: date) -> Optional[int]:
    """The year a _YEAR match names outright ("2027") or by stepping ("next
    year"), or None when he said none."""
    groups = match.groupdict()
    if groups.get("y"):
        return int(groups["y"])
    rel = groups.get("ry")
    if rel:
        return today.year + {"next": 1, "last": -1}.get(rel.lower(), 0)
    return None


def _month_context(text: str, today: date) -> tuple:
    """(month, year) the sentence gives a bare ordinal -- "in october",
    "next month" -- or (None, None) when it gives none.  A relative month
    is explicit in both fields; a named month leaves the year to the year
    rule."""
    match = _MONTH_CTX_RX.search(text)
    if match is None:
        return None, None
    if match.group("rel"):
        step = {"next": 1, "last": -1}.get(match.group("rel").lower(), 0)
        month = today.month + step
        return (month - 1) % 12 + 1, today.year + (month - 1) // 12
    return _MONTHS[match.group("mon").rstrip(".").lower()], _year_of(match, today)


def step_months(day: date, months: int) -> date:
    """``day`` moved by whole months, the day-of-month clamped to the month
    it lands in: 31 August + 1 is 30 September, not an error."""
    m = day.month - 1 + months
    y, m = day.year + m // 12, m % 12 + 1
    last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day
    return date(y, m, min(day.day, last))


def _stepped(day: date, count: int, unit: str, sign: int = 1) -> date:
    """``day`` moved ``count`` units forward (sign 1) or back (-1)."""
    unit = unit.lower()
    if unit.startswith("year"):
        return step_months(day, 12 * count * sign)
    if unit.startswith("month"):
        return step_months(day, count * sign)
    return day + timedelta(days=count * _unit_days(unit) * sign)


def _make_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _resolve_day(month, day: int, year, today: date,
                 backward: bool) -> Optional[date]:
    """The day he meant, or None when no real date fits those numbers.

    THE YEAR RULE, decided 2026-09-05: a date with no year is the NEXT
    occurrence at or after today, or -- when he asked in the past tense --
    the most recent at or before it.  "January 5th" asked in September is
    next January because it is the next one; "September 12th" asked on the
    5th is this month for the same reason; a bare "the 3rd" past the 3rd
    rolls to next MONTH, not next year.  29 February needs no special case
    at all: the next one is 2028, which is what the scan finds.
    """
    if not 1 <= day <= 31:
        return None
    if month is not None and not 1 <= month <= 12:
        return None
    if year is not None:
        return _make_date(year, month or today.month, day)
    step = -1 if backward else 1
    if month is not None:
        for i in range(_YEAR_SCAN):
            got = _make_date(today.year + step * i, month, day)
            if got is not None and (got <= today if backward else got >= today):
                return got
        return None
    y, m = today.year, today.month
    for _ in range(_MONTH_SCAN):
        got = _make_date(y, m, day)
        if got is not None and (got <= today if backward else got >= today):
            return got
        m += step
        if m > 12:
            m, y = 1, y + 1
        elif m < 1:
            m, y = 12, y - 1
    return None


def _ask_impossible(month, day: int) -> str:
    if month and 1 <= month <= 12 and 1 <= day <= 31:
        return _ask(f"There's no {day}{_suffix(day)} of {_month_name(month)}, "
                    "sir — which day did you mean?")
    return _ask("I couldn't make that out as a date, sir — which day did you mean?")


def _weekday_disagrees(text: str, day: date) -> Optional[str]:
    """The question to ask when he named a weekday the date is not.

    DECIDED 2026-09-05: "friday the 20th" where the 20th is a Sunday must
    NOT quietly become either one.  The old code took the weekday and
    answered about the next Friday -- a day he never named -- and said
    nothing about it.  Jarvis now says which day the date really is and
    offers both readings.
    """
    named = None
    for i, word in enumerate(WEEKDAYS):
        if re.search(rf"\b{word}\b", text, re.I):
            named = i
            break
    if named is None or named == day.weekday():
        return None
    delta = (named - day.weekday()) % 7
    if delta > 3:                            # the NEAREST such weekday
        delta -= 7
    alt = day + timedelta(days=delta)
    actual = day.strftime("%A")
    wanted = WEEKDAYS[named].capitalize()
    return _ask(
        f"The {day.day}{_suffix(day.day)} of {_month_name(day.month)} is a "
        f"{actual}, sir, not a {wanted} — did you mean {actual} the "
        f"{day.day}{_suffix(day.day)}, or {wanted} the "
        f"{alt.day}{_suffix(alt.day)}?")


def _dated(text: str, month, day: int, year, today: date, backward: bool) -> str:
    got = _resolve_day(month, day, year, today, backward)
    if got is None:
        return _ask_impossible(month, day)
    return _weekday_disagrees(text, got) or got.isoformat()


_FRAME_LEAD_RX = re.compile(rf"^(?:{_DATE_FRAME})\s+", re.I)
_PREP_LEAD_RX = re.compile(rf"^(?:{_PREP_FRAME})\s+", re.I)
_JOINER_RX = re.compile(
    r"^\s*,?\s*(?:to|and|or|through|thru|till|until|&|-)\s+", re.I)
SPAN_ASK = ("I can only look at one day at a time, sir — which of those "
            "did you want?")


def _span(match) -> tuple:
    """(start, end) of the day-words inside a match.  The framing
    preposition is not part of them, so a caller that rewrites the day
    (commander.day_shift_followup) keeps his "on"."""
    lead = _FRAME_LEAD_RX.match(match.group(0))
    return match.start() + (lead.end() if lead else 0), match.end()


def _find_date(text: str, today: date, backward: bool = False,
               bare_ordinal: bool = True) -> Optional[tuple]:
    """(value, start, end) for the first date-shaped thing in ``text`` that
    the frame rule admits -- ``value`` an ISO date or an ask -- or None
    when it names no date at all.  It NEVER yields "today".

    ``bare_ordinal=False`` leaves the last rule out -- an ordinal whose
    month the sentence does not give ("the 12th") -- which is how
    model_day_stands tells a month HIS WORDS pinned from one the reader
    inferred."""
    match = _D_ISO_RX.search(text)
    if match:
        got = _make_date(int(match.group("y")), int(match.group("m")),
                         int(match.group("d")))
        value = got.isoformat() if got is not None else \
            _ask_impossible(int(match.group("m")), int(match.group("d")))
        return value, match.start(), match.end()
    match = _NTH_WD_RX.search(text)
    if match:
        return (_nth_weekday_date(match, today, backward), *_span(match))
    match = _D_MD_RX.search(text) or _D_DM_RX.search(text)
    # A month name beside a BARE number ("may 3", "march 3") is a date only
    # under the same tail rule as a bare ordinal: "may 3 people come" was
    # answered about the 3rd of May, which is the non-date class of blocker
    # (1) wearing a month name.  An ordinal suffix or an explicit year
    # settles it on its own ("september 12th class", "september 12 2027").
    if match and not match.group("ord") and _year_of(match, today) is None \
            and not _tail_ok(text[match.end():]):
        match = None
    if match:
        value = _dated(text, _MONTHS[match.group("mon").rstrip(".").lower()],
                       int(match.group("d")), _year_of(match, today), today,
                       backward)
        return (value, *_span(match))
    match = _D_NUM_RX.search(text)
    if match and not _PREP_LEAD_RX.match(match.group(0)) and not (
            1 <= int(match.group("a")) <= 12
            and _make_date(2000, int(match.group("a")), int(match.group("b")))):
        match = None                        # a noun frame reads only a clean date
    if match:
        return (_separated(text, match, "/", today, backward), *_span(match))
    match = _D_REL_MONTH_RX.search(text)
    if match:
        step = {"next": 1, "last": -1}.get(match.group("rel").lower(), 0)
        month = today.month + step
        year = today.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        value = _dated(text, month, int(match.group("d")), year, today, backward)
        return (value, *_span(match))
    match = _D_DASH_RX.search(text)
    if match:
        return (_dashed(text, match, today, backward), *_span(match))
    match = _D_ORD_RX.search(text)
    if match:
        # The month, when the rest of the sentence gives one ("next month
        # on the 12th", "on the 12th in october"); otherwise the year rule
        # picks the next such day-of-month.
        month, year = _month_context(text, today)
        if month is None and not bare_ordinal:
            return None
        value = _dated(text, month, int(match.group("d")), year, today, backward)
        return (value, *_span(match))
    return None


def _explicit_date(text: str, today: date, backward: bool = False) -> Optional[str]:
    """An ISO date, or an ASK, for anything date-shaped in ``text``; None
    when he named no date at all.  It NEVER returns "today"."""
    found = _find_date(text, today, backward)
    return found[0] if found else None


def _sep_of(match) -> str:
    """The separator he actually used, so the question quotes his words."""
    return "-" if "-" in match.group(0) else "."


def _separated(text: str, match, sep: str, today: date, backward: bool) -> str:
    """A 9/12 pair (or a 9-12-2027 triple), read month-first."""
    a, b = int(match.group("a")), int(match.group("b"))
    raw_year = match.group("y")
    year = None
    if raw_year:
        year = int(raw_year) + (2000 if len(raw_year) == 2 else 0)
    if a > 12:
        # AMBIGUITY, decided 2026-09-05: separated dates are month-first
        # because he is in Texas, so 9/12 is September 12th.  13/5
        # cannot be month-first -- and quietly switching convention for
        # one input is exactly the "looks handled and is not" trap that
        # made "friday the 20th" the worst case of this bug.  Name the
        # day-month reading and ask.
        if 1 <= b <= 12 and 1 <= a <= 31:
            return _ask(f"{a}{sep}{b} isn't a date I can read month-first, sir "
                        f"— did you mean the {a}{_suffix(a)} of "
                        f"{_month_name(b)}?")
        return _ask(f"I couldn't read {a}{sep}{b} as a date, sir — which day "
                    "did you mean?")
    return _dated(text, a, b, year, today, backward)


def _dashed(text: str, match, today: date, backward: bool) -> str:
    """A 9-12 or 9.12 pair: with a year it is a date (a clock has no year);
    without one it is a QUESTION that offers the date reading."""
    sep = _sep_of(match)
    if match.group("y"):
        return _separated(text, match, sep, today, backward)
    a, b = int(match.group("a")), int(match.group("b"))
    if 1 <= a <= 12 and _make_date(2000, a, b) is not None:
        return _ask(f"Was {a}{sep}{b} the {b}{_suffix(b)} of {_month_name(a)}, "
                    "sir, or a time?")
    return _ask(f"I couldn't read {a}{sep}{b} as a date, sir — which day did "
                "you mean?")


def _nth_weekday(year: int, month: int, weekday: int, nth) -> Optional[date]:
    """The nth (1..5) or "last" such weekday of the month; None when the
    month has no such day (a 5th Friday, most months)."""
    if nth == "last":
        last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
        return last - timedelta(days=(last.weekday() - weekday) % 7)
    first = date(year, month, 1)
    got = first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (nth - 1))
    return got if got.month == month else None


def _nth_weekday_date(match, today: date, backward: bool) -> str:
    raw_n = match.group("n").lower()
    nth = "last" if raw_n == "last" else int(raw_n[:-2])
    weekday = WEEKDAYS.index(match.group("wd").lower())
    rel = match.group("rel")
    if rel:
        # "of this month" is explicit: no forward/backward search.
        step = {"next": 1, "last": -1}.get(rel.lower(), 0)
        month = today.month + step
        years = [today.year + (month - 1) // 12]
        month = (month - 1) % 12 + 1
    else:
        month = _MONTHS[match.group("mon").rstrip(".").lower()]
        if _year_of(match, today) is not None:
            years = [_year_of(match, today)]
        else:
            # THE YEAR RULE (see _resolve_day): the next such day at or
            # after today, or the most recent at or before it.
            step = -1 if backward else 1
            years = [today.year + step * i for i in range(_YEAR_SCAN)]
    fixed = bool(rel or _year_of(match, today) is not None)
    for year in years:
        got = _nth_weekday(year, month, weekday, nth)
        if got is None:
            return _ask(f"There's no {raw_n} {WEEKDAYS[weekday].capitalize()} in "
                        f"{_month_name(month)}, sir — which day did you mean?")
        if fixed or (got <= today if backward else got >= today):
            return got.isoformat()
    return _ask("I couldn't make that out as a date, sir — which day did you "
                "mean?")                          # pragma: no cover - scan-bounded


def _word_day(word: str, today: date) -> date:
    """today / tonight / now, tomorrow, yesterday, or the NEXT such weekday
    counting today -- the same rule format_events uses, so they cannot
    drift."""
    word = word.lower()
    if word in ("today", "tonight", "now"):
        return today
    if word == "tomorrow":
        return today + timedelta(days=1)
    if word == "yesterday":
        return today - timedelta(days=1)
    return today + timedelta(days=(WEEKDAYS.index(word) - today.weekday()) % 7)


def _base_span(text: str, today: date, backward: bool,
               bare_ordinal: bool = True) -> Optional[tuple]:
    """(value, end) for the day an OFFSET is measured from: an explicit
    date, "today", "tomorrow", "yesterday", "now" or a weekday.  None when
    ``text`` names none."""
    found = _find_date(text, today, backward, bare_ordinal)
    if found:
        return found[0], found[2]                # an ISO date, or an ask
    match = _TOMORROW_RX.search(text)
    if match:
        return (today + timedelta(days=1)).isoformat(), match.end()
    match = _DAY_WORD_RX.search(text)
    if match:
        return _word_day(match.group("w"), today).isoformat(), match.end()
    return None


def _base_day(text: str, today: date, backward: bool) -> Optional[str]:
    got = _base_span(text, today, backward)
    return got[0] if got else None


def _count(raw) -> int:
    raw = (raw or "one").lower().removesuffix(" of").strip()
    got = _COUNT_WORDS.get(raw)
    if got is None:
        try:
            got = int(raw)
        except ValueError:                  # pragma: no cover - regex-bounded
            got = 1
    return got


def _unit_days(unit: str) -> int:
    unit = unit.lower()
    return 7 if unit.startswith("week") else 14 if unit.startswith("fortnight") else 1


def _offset_span(text: str, today: date, backward: bool,
                 bare_ordinal: bool = True) -> Optional[tuple]:
    """"the day after the 12th" -> the 13th, with the span of the whole
    phrase; an ISO date or an ask, and None when the sentence carries no
    offset phrase at all."""
    match = _OFFSET_RX.search(text)
    if match is None:
        return None
    count = _count(match.group("n"))
    sign = -1 if _BACKWARD_DIRS.match(match.group("dir")) else 1
    base = _base_span(text[match.end():], today, backward, bare_ordinal)
    if base is None:
        # Understood as an offset, unable to name its base: ASK.  Dropping
        # the offset and answering about the base is the bug itself.
        return (_ask(f"Which day is that {'before' if sign < 0 else 'after'}, "
                     "sir?"), match.start(), match.end())
    value, end = base
    if is_ask(value):
        return value, match.start(), match.end() + end
    got = _stepped(date.fromisoformat(value), count, match.group("unit"), sign)
    return got.isoformat(), match.start(), match.end() + end


def _offset_date(text: str, today: date, backward: bool) -> Optional[str]:
    got = _offset_span(text, today, backward)
    return got[0] if got else None


def _weekday_offset_span(text: str, today: date, backward: bool,
                         bare_ordinal: bool = True) -> Optional[tuple]:
    """"the friday after next" -> the Friday after the coming one; "the
    saturday after the 12th" -> the first Saturday past the 12th.  None
    when the sentence has neither, or when a "<weekday> after ..." steps
    from nothing nameable ("friday after the meeting"): that is the
    weekday alone, which the caller's weekday loop still reads."""
    match = _WD_AFTER_NEXT_RX.search(text)
    if match:
        got = _word_day(match.group("wd"), today) + timedelta(days=7)
        return got.isoformat(), match.start(), match.end()
    match = _WD_OFFSET_RX.search(text)
    if match is None:
        return None
    base = _base_span(text[match.end():], today, backward, bare_ordinal)
    if base is None:
        return None
    value, end = base
    if is_ask(value):
        return value, match.start(), match.end() + end
    weekday = WEEKDAYS.index(match.group("wd").lower())
    day = date.fromisoformat(value)
    if _BACKWARD_DIRS.match(match.group("dir")):
        got = day - timedelta(days=(day.weekday() - weekday) % 7 or 7)
    else:
        got = day + timedelta(days=(weekday - day.weekday()) % 7 or 7)
    return got.isoformat(), match.start(), match.end() + end


def _stepped_span(text: str, today: date, backward: bool,
                  bare_ordinal: bool = True) -> Optional[tuple]:
    """"in two days", "three days ago", "a week on tuesday": see _IN_RX."""
    match = _IN_RX.search(text)
    if match:
        got = _stepped(today, _count(match.group("n")), match.group("unit"))
        return got.isoformat(), match.start(), match.end()
    match = _AGO_RX.search(text)
    if match:
        got = _stepped(today, _count(match.group("n")), match.group("unit"), -1)
        return got.isoformat(), match.start(), match.end()
    match = _FROM_RX.search(text)
    if match:
        base = _base_span(text[match.end():], today, backward, bare_ordinal)
        if base is not None:
            value, end = base
            if is_ask(value):
                return value, match.start(), match.end() + end
            got = _stepped(date.fromisoformat(value), _count(match.group("n")),
                           match.group("unit"))
            return got.isoformat(), match.start(), match.end() + end
    return None


def _second_date(text: str, today: date, backward: bool) -> bool:
    """True when ``text`` -- the words before or after a date already read
    -- carries another day.  A joiner ("to", "and", "or") stands in for the
    frame the second date usually lacks."""
    lead = _JOINER_RX.match(text)
    if lead:
        text = "on " + text[lead.end():]
    return _find_date(text, today, backward) is not None


def _unreadable(text: str, today: date) -> Optional[str]:
    """The question for a thing that is date-shaped and cannot be read;
    None when nothing date-shaped is there."""
    match = _MONTH_ONLY_RX.search(text)
    if match:
        if match.group("relm"):
            return _ask(f"Which day {match.group('relm').lower()} month, sir?")
        month = _month_name(_MONTHS[match.group("mon").rstrip(".").lower()])
        year = _year_of(match, today)
        return _ask(f"Which day in {month}{f' {year}' if year else ''}, sir?")
    match = _NEAR_MONTH_RX.search(text)
    if match:
        word = (match.group("w") or match.group("w2")).lower()
        near = difflib.get_close_matches(word, _MONTH_NAMES, n=1, cutoff=0.8)
        if near and word not in _MONTHS:
            day = int(match.group("d") or match.group("d2"))
            return _ask(f"Did you mean the {day}{_suffix(day)} of "
                        f"{near[0].capitalize()}, sir?")
    if _DATEISH_RX.search(text):
        return _ask("I couldn't work out which date you meant, sir — "
                    "which day did you want?")
    return None


def _read(raw: str, today: date, backward,
          bare_ordinal: bool = True) -> Optional[tuple]:
    """THE ONE READER, over prepared text: (value, start, end), or None
    when the sentence names no specific day.  ``value`` is an ISO date or
    an ask; the positions are where his day-words sit in ``raw``.

    The three positional parameters are a contract (a structural test
    spies on them); ``bare_ordinal`` is model_day_stands' seam only."""
    if backward is None:
        backward = bool(_PAST_RX.search(raw))
    got = _offset_span(raw, today, backward, bare_ordinal)
    if got:
        return got
    got = _weekday_offset_span(raw, today, backward, bare_ordinal)
    if got:
        return got
    got = _stepped_span(raw, today, backward, bare_ordinal)
    if got:
        return got
    found = _find_date(raw, today, backward, bare_ordinal)
    if found:
        value, start, end = found
        if is_ask(value):
            return found
        # TWO days in one breath -- "the 12th to the 14th", "yesterday and
        # the 14th" -- were answered about the first with no word about the
        # second: a confident partial answer.  One range value cannot carry
        # two days, so a span is a question, never the first day of it.
        if _YESTERDAY_RX.search(raw) or _second_date(raw[:start], today, backward) \
                or _second_date(raw[end:], today, backward):
            return _ask(SPAN_ASK), start, end
        return found
    match = _YESTERDAY_RX.search(raw)
    if match:
        return (today - timedelta(days=1)).isoformat(), match.start(), match.end()
    got = _unreadable(raw, today)
    if got:
        return got, 0, len(raw)
    return None


def _prepared(text, clean: bool) -> tuple:
    """(the text the rules read, the offset its positions carry)."""
    raw = digit_ordinals(_clean(text) if clean else (text or "")).lower()
    if _BARE_VALUE_RX.match(raw):
        return "on " + raw, 3
    return raw, 0


def sentence_date(text, today: date, backward=None) -> Optional[str]:
    """The specific DAY this sentence names -- an ISO date or an ask -- and
    None when it names no specific day.

    THE ONE READER BOTH DOORS USE.  The forced path (commander.calendar_range
    -> coerce_range) and the model path (get_calendar's ``derive``) used to
    read the sentence with two different functions, and MEASURED 2026-09-05
    they disagreed: "what was on my calendar yesterday" was the 4th through
    one door and TODAY through the other, on the same eight words.  Since
    round three the day-shift rewrite (commander.day_shift_followup) reads
    through here as well, via date_span.  Word ranges ("tomorrow",
    "monday") are deliberately NOT resolved here -- the model door must
    leave those to the model, which can resolve a follow-up from the
    conversation that this function cannot see.

    THE FRAME RULE, in his register: a date is read only when the sentence
    is about WHEN -- a calendar word or a preposition stands in front of the
    number.  A number nobody framed as a date is not a date.  A thing that
    looks like a date and cannot be read is a question back to him, never
    today.
    """
    raw, _shift = _prepared(text, clean=True)
    got = _read(raw, today, backward)
    return got[0] if got else None


def date_span(text, today: date, backward=None) -> Optional[tuple]:
    """(value, start, end): the day ``text`` names, read by the SAME reader
    as sentence_date, and where his day-words sit in ``digit_ordinals(text)``
    -- so commander.day_shift_followup can move the day in place and keep
    the rest of his sentence.  None when it names no day, or when the
    positions cannot be trusted (lower-casing moved the letters)."""
    plain = digit_ordinals(text or "")
    raw, shift = _prepared(plain, clean=False)
    if len(raw) - shift != len(plain):
        return None
    got = _read(raw, today, backward)
    if not got:
        return None
    value, start, end = got
    return value, max(start - shift, 0), max(end - shift, 0)


def coerce_range(value, now: datetime = None) -> str:
    """Loose model values -> a range: one of RANGES ("this week" -> "week"),
    an ISO date ("september 12th" -> "2026-09-12"), or an ASK.

    ``now`` is the seam; every test injects its own.  Production reads the
    clock once here, which is the same clock get_calendar is about to use.
    """
    raw = _clean(value)
    if raw.startswith(ASK_PREFIX):
        return raw                       # already decided; both doors coerce
    text = raw.lower().strip(" .?!") or "today"
    if text in RANGES:
        return text
    today = (now or now_local()).date()
    # BEFORE the weekday loop, or "friday the 20th" is a Friday he did not
    # ask for -- that is exactly how the worst case of this bug happened.
    # sentence_date covers offsets, "yesterday" (the one past day the cache
    # genuinely holds) and every explicit date, and is the SAME reader the
    # model path uses, so the two doors cannot drift apart again.
    dated = sentence_date(text, today)
    if dated:
        return dated
    # "on monday", "for Monday", "this monday" -- and "next monday", which
    # keeps its old meaning here: the COMING Monday, the day "monday" names.
    # DEFAULT LEFT FOR HIM (round three, 2026-09-05): to some ears "next
    # Monday" is the Monday after that.  Not changed without his say.
    for day in WEEKDAYS:
        if re.search(rf"\b{day}\b", text):
            return day
    if _TOMORROW_RX.search(text):
        return "tomorrow"
    if "week" in text or "7 day" in text or "seven day" in text:
        return "week"
    if "next" in text or "upcoming" in text or "soon" in text or "coming up" in text:
        return "next"
    # THE ONE line that may answer "today": he named no date, so today is
    # the honest default.  Every unrecognised DATE left through an ask in
    # sentence_date above (_unreadable, _DATEISH_RX): the guard lives THERE,
    # where both doors and the day-shift rewrite meet it, not only here.
    return "today"


def _pinned_by_words(text, today: date) -> bool:
    """True when his words settle the month themselves -- a month name, a
    numeric or ISO date, "next month", a step from today -- and False when
    the reader INFERRED it from a bare ordinal ("the 12th", "the day after
    the 12th")."""
    raw, _shift = _prepared(text, clean=True)
    full = _read(raw, today, None)
    if not full:
        return False
    strict = _read(raw, today, None, bare_ordinal=False)
    return strict is not None and strict[0] == full[0]


def model_day_stands(said, heard: str, model_range, now: datetime) -> bool:
    """THE DERIVER RULE (round four, 2026-09-06): when the model hands over
    a value that reads as a real day, the deriver may refine it but may
    never replace it with today or with a different day.

    MEASURED before the rule: the model sent 2027-01-05 for "january the
    5th", the reader (with the round-three hole) read a bare "the 5th" as
    today, and registry.call let the reading REPLACE the model's correct
    value -- 6 of 6 phrasings answered today.  A reader will always have
    a hole somewhere; this rule is what stops the next one from throwing
    away a value the model got right.

    His words override the model only when they CONTRADICT it: a
    different day-of-month, or a month they name themselves.  When the
    words gave only a day-of-month and the model's day agrees with it,
    the model's month and year are a refinement the words cannot
    contradict, and they stand.  A QUESTION from the words is neither
    today nor a different day, so it always stands over a guess.
    """
    if not model_range or is_ask(heard):
        return False
    model_day = as_date(coerce_range(model_range, now))
    heard_day = as_date(heard)
    if model_day is None or heard_day is None:
        return False
    if heard_day == model_day:
        return True
    if _pinned_by_words(said, now.date()):
        return False
    return heard_day.day == model_day.day


# A day commonly holds three or four events. At the default two
# sentences Jarvis named the first and dropped the rest.
CALENDAR_MAX_SENTENCES = 4


def date_words(day: date, today: date) -> str:
    """"Saturday the 12th" inside this month, "Saturday the 3rd of October"
    outside it, and the year too when it is not this one.

    Naming the month is how he HEARS a wrong pick.  "the 3rd" is
    unambiguous inside September and dangerous outside it, and the whole
    complaint was a confident answer about a day he did not ask for.
    """
    stem = f"{day.strftime('%A')} the {day.day}{_suffix(day.day)}"
    if day.year != today.year:
        return f"{stem} of {_month_name(day.month)} {day.year}"
    if day.month != today.month:
        return f"{stem} of {_month_name(day.month)}"
    return stem


def _date_label(day: date, today: date) -> tuple:
    """(the words after "Nothing on", the words that open a list).

    A date that IS today or tomorrow is worded with the word he has:
    answering "September 12th" with "Saturday the 12th" is right, doing it
    for today would be a stilted way to say "today"."""
    if day == today:
        return "today", "Today"
    if day == today + timedelta(days=1):
        return "tomorrow", "Tomorrow"
    words = date_words(day, today)
    return words, words


def reachable(today: date, window_days: int = WINDOW_DAYS) -> tuple:
    """The first and last day the cache actually holds.

    ``CalendarSource._window`` anchors at YESTERDAY-midnight and reaches
    ``window_days + 1`` days, so the last whole day inside it is
    ``today + window_days - 1``.  The two must not drift.
    """
    return today - timedelta(days=1), today + timedelta(days=window_days - 1)


def out_of_reach(day: date, today: date, window_days: int = WINDOW_DAYS) -> str:
    """"" when the day is inside the cache, else the sentence that says so.

    DECIDED 2026-09-05: a day outside the window is REFUSED BY NAME, never
    answered "nothing on it".  Those events are missing from the CACHE, not
    from his calendar, and "Nothing on the 3rd of October, sir" would be
    the same confident wrong answer he reported, in a new coat.  The window
    is now 45 days rather than 14 (see WINDOW_DAYS), so this refusal is
    rare; it still has to be right when it fires.
    """
    lo, hi = reachable(today, window_days)
    if day < lo:
        return (f"I only keep the calendar back to yesterday, sir; I can't "
                f"look as far back as {date_words(day, today)}.")
    if day > hi:
        return (f"I only hold the calendar out to {date_words(hi, today)}, "
                f"sir; {date_words(day, today)} is past that.")
    return ""


def format_events(events, range: str = "today", now: datetime = None,
                  window_days: int = WINDOW_DAYS) -> str:
    """The compact text for a date / today / tomorrow / week / next at ``now``."""
    now = now or now_local()
    range = coerce_range(range, now)
    today = now.date()
    if is_ask(range):
        return ask_words(range)
    events = merge_events(events)
    day = as_date(range)
    if day is not None:
        beyond = out_of_reach(day, today, window_days)
        if beyond:
            return beyond
        label, opener = _date_label(day, today)
        todays = _day_events(events, day)
        if not todays:
            return f"Nothing on {label}, sir."
        return f"{opener}: " + ", ".join(_event_words(e) for e in todays) + \
            "; nothing else."
    if range in WEEKDAYS:
        # The NEXT such day, counting today. Asked on Friday, "Monday" is the
        # coming Monday, never the one just gone.
        want = WEEKDAYS.index(range)
        day = today + timedelta(days=(want - today.weekday()) % 7)
        todays = _day_events(events, day)
        if not todays:
            return f"Nothing on {range.capitalize()}, sir."
        return f"{range.capitalize()}: " + \
            ", ".join(_event_words(e) for e in todays) + "; nothing else."
    if range in ("today", "tomorrow"):
        day = today if range == "today" else today + timedelta(days=1)
        todays = _day_events(events, day)
        if not todays:
            return f"Nothing on {range}, sir."
        return f"{range.capitalize()}: " + ", ".join(_event_words(e) for e in todays) + \
            "; nothing else."
    if range == "week":
        parts = []
        for i in builtins.range(7):
            day = today + timedelta(days=i)
            todays = _day_events(events, day)
            if todays:
                parts.append(f"{_day_label(day, today)} " +
                             ", ".join(_event_words(e) for e in todays))
        if not parts:
            return "Nothing on this week, sir."
        return "This week: " + "; ".join(parts) + "; nothing else."
    # next: the first event that has not started yet
    upcoming = [e for e in events if e.start > now]
    if not upcoming:
        return "Nothing coming up in the next two weeks, sir."
    ev = min(upcoming, key=lambda e: (e.start, not e.all_day))
    if ev.all_day:
        label = _day_label(ev.start.date(), today)
        when = f"all day {label}" if label in ("today", "tomorrow") else f"all day on {label}"
        text = f"Next: {ev.title}, {when}"
    else:
        rel = describe_due(ev.start, now)
        when = f"{rel}, at {clock_words(ev.start)}" if rel.startswith("in ") or rel == "now" \
            else rel
        text = f"Next: {ev.title} {when}"
        if _duration_words(ev):
            text += f",{_duration_words(ev)}"
    if ev.location:
        text += f", at {ev.location}"
    return text + "."


_NOTHING_ELSE_RX = re.compile(r";\s*nothing else\.\s*$", re.I)
_NOTHING_AT_ALL_RX = re.compile(r"^Nothing (on|coming up) (.+), sir\.\s*$", re.I)


def drop_completeness(text: str) -> str:
    """Take back the claim that the day is fully accounted for.

    ``format_events`` closes a list with "; nothing else." and an empty day
    with "Nothing on today, sir." -- both are true only when every feed
    answered.  With one subscription unread they are the confident
    half-answer FOUND 2026-08-26
    (tests/test_found_calendar_partial_failure.py), so a snapshot with a
    dead source gets the claim taken back before it is spoken."""
    text = (text or "").strip()
    if _NOTHING_ELSE_RX.search(text):
        return _NOTHING_ELSE_RX.sub(".", text)
    match = _NOTHING_AT_ALL_RX.match(text)
    if match:
        return (f"Nothing {match.group(1)} {match.group(2)} "
                "in the calendars I can reach, sir.")
    return text


def down_words(down, now: datetime) -> str:
    """One spoken sentence naming the feeds that are not answering.

    Named from the calendar's OWN name (X-WR-CALNAME, carried on the events
    it last served) because "google-2" means nothing to anybody; the last
    time it did answer is the useful half of the news."""
    if not down:
        return ""
    named = [d.name for d in down if d.name]
    if len(named) == len(down) == 1:
        subject = f"your {named[0]} calendar"
    elif named and len(named) == len(down):
        subject = "your " + ", ".join(named[:-1]) + f" and {named[-1]} calendars"
    elif len(down) == 1:
        # A feed that has never answered has never told us its name.
        subject = "one of your calendars"
    else:
        subject = f"{len(down)} of your calendars"
    since = [d.since for d in down if d.since]
    if len(down) == 1 and since:
        return (f"I can't reach {subject}, sir \u2014 nothing from it since "
                f"{as_of_words(min(since), now)}.")
    return f"I can't reach {subject}, sir."


def as_of_words(fetched_at: float, now: datetime) -> str:
    """"9:10 am" / "9:10 am yesterday" / "9:10 am on Monday"."""
    when = datetime.fromtimestamp(fetched_at, now.tzinfo)
    days = (now.date() - when.date()).days
    if days <= 0:
        return clock_words(when)
    if days == 1:
        return f"{clock_words(when)} yesterday"
    if days < 7:
        return f"{clock_words(when)} on {when.strftime('%A')}"
    return f"{clock_words(when)} on the {when.day}{_suffix(when.day)} of {when.strftime('%B')}"


# --------------------------------------------------------------- source
@dataclass
class SourceDown:
    """One configured feed that is not answering."""
    id: str                                 # google-2 / icloud
    name: str = ""                          # its own name ("Canvas"), when known
    since: Optional[float] = None           # last time it DID answer
    error: str = ""                         # the exception type, never the URL


@dataclass
class Snapshot:
    events: list = field(default_factory=list)
    fetched_at: Optional[float] = None      # oldest source that is ANSWERING
    stale: bool = True
    errors: list = field(default_factory=list)
    down: list = field(default_factory=list)   # SourceDown per dead feed


def _window_words(window) -> Optional[list]:
    """A (start, end) pair as two ISO strings, for the cache file."""
    if not window:
        return None
    return [window[0].isoformat(), window[1].isoformat()]


def _window_from(words) -> Optional[tuple]:
    """The pair back, or None when it is missing or unreadable."""
    try:
        return (datetime.fromisoformat(words[0]), datetime.fromisoformat(words[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _default_dav_client(url: str, username: str, password: str):
    import caldav
    return caldav.DAVClient(url=url, username=username, password=password,
                            timeout=FETCH_TIMEOUT)


class CalendarSource:
    """Cached, background-refreshed view over every configured calendar."""

    def __init__(self, cfg, cache_path=None, fetch: Callable = None,
                 dav_client: Callable = None, clock: Callable = time.time,
                 tz=None, refresh_s: float = REFRESH_S, window_days: int = WINDOW_DAYS):
        self.cfg = cfg
        self.cache_path = Path(cache_path) if cache_path else cache_dir() / "calendar_cache.json"
        self._fetch = fetch
        self._dav_client = dav_client or _default_dav_client
        self._clock = clock
        self._tz = tz
        self.refresh_s = refresh_s
        self.window_days = window_days
        self._lock = threading.Lock()
        self._sources: dict[str, dict] = {}     # id -> {"fetched_at", "events"}
        self._etags: dict[str, dict] = {}       # url -> {"etag", "last_modified", "raw"}
        # id -> {"count", "since", "error", "until"} for a source that is not
        # answering; guarded by _lock, cleared by one success (SOURCE_BACKOFF_S).
        self._failures: dict[str, dict] = {}
        self._last_trigger: Optional[float] = None
        self.errors: list[str] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._refresh_lock = threading.Lock()
        self._load_cache()

    # -- config -----------------------------------------------------
    @property
    def tz(self):
        return self._tz or system_tz()

    @property
    def ical_urls(self) -> list[str]:
        urls = cfg_get(self.cfg, "google_ical_urls") or []
        if isinstance(urls, str):
            urls = [urls]
        return [u.strip() for u in urls if isinstance(u, str) and u.strip()
                and not u.strip().startswith("<")]

    @property
    def icloud(self) -> Optional[dict]:
        user = _clean(cfg_get(self.cfg, "icloud.apple_id"))
        pw = str(cfg_get(self.cfg, "icloud.app_password") or "")
        if not user or not pw or user.startswith("<") or pw.startswith("<"):
            return None
        return {"url": _clean(cfg_get(self.cfg, "icloud.url")) or ICLOUD_URL,
                "username": user, "password": pw}

    @property
    def configured(self) -> bool:
        return bool(self.ical_urls) or self.icloud is not None

    # -- cache file ---------------------------------------------------
    def _load_cache(self) -> None:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001 - a bad cache is a miss
            log.warning("calendar cache unreadable: %s", exc)
            return
        sources = data.get("sources") if isinstance(data, dict) else None
        if not isinstance(sources, dict):
            return
        loaded = {}
        for sid, entry in sources.items():
            try:
                loaded[sid] = {
                    "fetched_at": float(entry.get("fetched_at") or 0),
                    # The window this source's events were GATHERED over.
                    # Absent from every cache file written before 2026-09-05:
                    # None then means "unknown", which holds calwatch's diff
                    # back for one cycle rather than letting it invent
                    # bookings (covered_window).
                    "window": _window_from(entry.get("window")),
                    "events": [Event.from_dict(d, self.tz) for d in entry.get("events", [])]}
            except Exception as exc:  # noqa: BLE001 - skip a broken source
                log.warning("calendar cache source %s skipped: %s", sid, exc)
        with self._lock:
            self._sources = loaded

    def _save_cache(self) -> None:
        stamp = self.fetched_at        # takes _lock; never call it inside the block
        with self._lock:
            payload = {"version": CACHE_VERSION,
                       "fetched_at": stamp,
                       "sources": {sid: {"fetched_at": e["fetched_at"],
                                         "window": _window_words(e.get("window")),
                                         "events": [ev.to_dict() for ev in e["events"]]}
                                   for sid, e in self._sources.items()}}
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.cache_path)
        except Exception as exc:  # noqa: BLE001 - best effort
            log.warning("calendar cache not written: %s", exc)

    # -- state -------------------------------------------------------
    @property
    def fetched_at(self) -> Optional[float]:
        """How fresh the answer is: the oldest source that is ANSWERING.

        A source in back-off keeps its events -- they are still the best we
        have -- but its stamp may not decide the freshness of the whole
        cache.  As the minimum over every source, one dead feed made
        ``is_stale`` true for the rest of the day, which both mislabelled
        every fresh answer "as of <hours ago>" and put another refresh on
        the worker for every single calendar question (SOURCE_BACKOFF_S).
        When nothing is answering the whole set is used again, so a fully
        offline calendar still says honestly how old it is."""
        with self._lock:
            stamps = [e["fetched_at"] for sid, e in self._sources.items()
                      if sid not in self._failures]
            if not stamps:
                stamps = [e["fetched_at"] for e in self._sources.values()]
        return min(stamps) if stamps else None

    def down(self) -> list:
        """The configured feeds that are not answering, as SourceDown."""
        with self._lock:
            return [SourceDown(id=sid, name=self._name_locked(sid),
                               since=state.get("since"),
                               error=str(state.get("error") or ""))
                    for sid, state in self._failures.items()]

    def _name_locked(self, sid: str) -> str:
        """The feed's own name ("Canvas"), from the events it last served.
        Caller holds _lock."""
        entry = self._sources.get(sid) or {}
        for ev in entry.get("events") or []:
            if getattr(ev, "calendar", ""):
                return str(ev.calendar)
        return ""

    def _held_off(self, sid: str, now: float) -> str:
        """"" when the source may be fetched, else its remembered error."""
        with self._lock:
            state = self._failures.get(sid)
            if not state or now >= state.get("until", 0):
                return ""
            return str(state.get("error") or "unavailable")

    def _record_outcomes(self, wanted, answered, failed, stamp) -> None:
        """Back-off bookkeeping. Caller holds _lock."""
        for sid in list(self._failures):
            if sid not in wanted:
                del self._failures[sid]        # no longer configured
        for sid in answered:
            self._failures.pop(sid, None)      # one answer clears the back-off
        for sid, error in failed.items():
            state = self._failures.get(sid) or {"count": 0, "since": None}
            state["count"] += 1
            state["error"] = error
            if state["since"] is None:
                # the last time it DID answer, for "nothing since 12:40 pm"
                state["since"] = (self._sources.get(sid) or {}).get("fetched_at")
            state["until"] = stamp + min(
                MAX_SOURCE_BACKOFF_S,
                SOURCE_BACKOFF_S * (2 ** (state["count"] - 1)))
            self._failures[sid] = state

    pending_event = None      # set by add_event when it needs a yes
    # The last event written and how to take it back: {"undo", "at", "title"}.
    # Parked here rather than returned because the confident path writes from
    # inside a TOOL call, whose ToolResult has no undo slot -- and that is
    # the path most adds take, so without this "scratch that" would still
    # have nothing to reach for. Commander._try_undo consults it, subject to
    # the same staleness window as any other undo.
    last_add = None

    def icloud_calendars(self):
        """Live caldav Calendar objects, for writing. Raises when unconfigured."""
        creds = self.icloud
        if creds is None:
            raise ValueError("I don't have your iCloud calendar set up, sir")
        client = self._dav_client(url=creds["url"], username=creds["username"],
                                  password=creds["password"])
        return list(client.principal().calendars())

    def events(self) -> list[Event]:
        with self._lock:
            groups = [list(e["events"]) for e in self._sources.values()]
        return merge_events(*groups)

    def raw_ics(self) -> dict:
        """{url: the last ics body fetched} for the subscriptions, empty
        before the first refresh.

        The Canvas coursework adapter (jarvis/tools/canvas_ical.py) needs a
        TERM-long view of the Canvas feed and this cache keeps only
        ``window_days``; handing back the bytes lets it re-parse the very
        same response over a wider window instead of putting a second
        request on the wire for a body already in memory."""
        try:
            state = list(self._etags.items())
        except RuntimeError:        # a refresh added a source mid-iteration
            return {}
        return {url: s["raw"] for url, s in state
                if isinstance(s, dict) and s.get("raw")}

    def is_stale(self, now: float = None) -> bool:
        fetched = self.fetched_at
        now = self._clock() if now is None else now
        return fetched is None or now - fetched > self.refresh_s

    # -- fetching ----------------------------------------------------
    def window(self, now: datetime = None) -> tuple:
        """The (start, end) the cached events cover, as a public seam.

        jarvis/calwatch.py diffs two snapshots and MUST know this: the
        window is anchored at yesterday-midnight and slides every day, so
        without it everything before yesterday reads as a mass
        cancellation and every new far edge as a new booking, once per
        midnight."""
        return self._window(now)

    def covered_window(self, now: datetime = None) -> Optional[tuple]:
        """The window the cached events were actually GATHERED over, or None
        when that is not knowable yet.

        ``window()`` is a PROMISE -- what the next fetch will ask for.  A
        source that failed or is in back-off keeps the events of its last
        fetch, gathered over the window it had THEN, so after a widening (or
        across a midnight slide) the promise over-states what the cache
        holds.  jarvis/calwatch.py diffs two snapshots and calls anything
        inside the window that is new an ADDED event, so handing it the
        promise while a feed catches up announces that feed's whole far half
        as new bookings.  MEASURED 2026-09-05 widening 14 -> 45 days with one
        of two feeds missing the first wide fetch: 31 spurious "has been
        added" lines; 0 with this window
        (tests/test_calendar_window_widening.py).

        None -- any source that has never told us its reach -- costs calwatch
        one silent cycle, which is the right price.
        """
        promise = self._window(now)
        with self._lock:
            windows = [entry.get("window") for entry in self._sources.values()]
        if not windows or any(w is None for w in windows):
            return None
        start = max([promise[0]] + [w[0] for w in windows])
        end = min([promise[1]] + [w[1] for w in windows])
        return (start, end) if start < end else None

    def _window(self, now: datetime = None) -> tuple:
        now = now or now_local(self.tz)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        return start, start + timedelta(days=self.window_days + 1)

    def _get_ical(self, url: str) -> bytes:
        fetch = self._fetch or _fetch
        state = self._etags.get(url) or {}
        headers = {}
        if state.get("raw") is not None:
            if state.get("etag"):
                headers["If-None-Match"] = state["etag"]
            if state.get("last_modified"):
                headers["If-Modified-Since"] = state["last_modified"]
        resp = fetch(url, timeout=FETCH_TIMEOUT, headers=headers or None)
        status = getattr(resp, "status", 200)
        if status == 304 and state.get("raw") is not None:
            return state["raw"]
        rheaders = getattr(resp, "headers", None) or {}
        self._etags[url] = {"etag": rheaders.get("etag", ""),
                            "last_modified": rheaders.get("last-modified", ""),
                            "raw": bytes(resp)}
        return bytes(resp)

    def _fetch_icloud(self, start: datetime, end: datetime) -> list[Event]:
        creds = self.icloud
        client = self._dav_client(url=creds["url"], username=creds["username"],
                                  password=creds["password"])
        events: list[Event] = []
        for cal in client.principal().calendars():
            name = _clean(getattr(cal, "name", "")) or "iCloud"
            try:
                found = cal.search(start=start, end=end, event=True, expand=True)
            except Exception as exc:  # noqa: BLE001 - one calendar at a time
                log.warning("icloud calendar %s search failed: %s", name, exc)
                continue
            for obj in found or []:
                data = getattr(obj, "data", None)
                if not data:
                    continue
                try:
                    events.extend(parse_ics(data, start, end, calendar=name, tz=self.tz))
                except Exception as exc:  # noqa: BLE001 - skip a bad object
                    log.warning("icloud event in %s unparsable: %s", name, exc)
        return events

    def refresh(self, now: datetime = None) -> bool:
        """Fetch every source now (worker thread / tests).  A failing
        source keeps its previous events; True when at least one source
        answered.  Refreshes are serialised: a caller arriving while one is
        in flight waits for it and then runs its own (cheap: conditional
        GETs answer 304).  A source that keeps failing is SKIPPED rather
        than retried at FETCH_TIMEOUT a cycle (SOURCE_BACKOFF_S); one
        answer clears the back-off."""
        with self._refresh_lock:
            return self._refresh(now)

    def _refresh(self, now: datetime = None) -> bool:
        start, end = self._window(now)
        stamp = self._clock()
        errors, fresh, ok_any = [], {}, False
        failed: dict[str, str] = {}
        for i, url in enumerate(self.ical_urls, 1):
            sid = f"google-{i}"
            held = self._held_off(sid, stamp)
            if held:
                # Still down as far as we know: say so in errors (a skipped
                # source must not read as a healthy one) but do not spend
                # another FETCH_TIMEOUT on it.
                errors.append(f"{sid}: {held}")
                continue
            try:
                raw = self._get_ical(url)
                fresh[sid] = parse_ics(raw, start, end, tz=self.tz)
                ok_any = True
            except Exception as exc:  # noqa: BLE001 - never log the URL
                errors.append(f"{sid}: {type(exc).__name__}")
                failed[sid] = type(exc).__name__
                log.warning("calendar %s failed: %s", sid, _redact(str(exc), url))
        if self.icloud is not None:
            held = self._held_off("icloud", stamp)
            if held:
                errors.append(f"icloud: {held}")
            else:
                try:
                    fresh["icloud"] = self._fetch_icloud(start, end)
                    ok_any = True
                except Exception as exc:  # noqa: BLE001 - never log the password
                    errors.append(f"icloud: {type(exc).__name__}")
                    failed["icloud"] = type(exc).__name__
                    log.warning("icloud calendar failed: %s",
                                _redact(str(exc), self.icloud["password"]))
        wanted = {f"google-{i}" for i in builtins.range(1, len(self.ical_urls) + 1)}
        if self.icloud is not None:
            wanted.add("icloud")
        with self._lock:
            for sid in list(self._sources):
                if sid not in wanted:
                    del self._sources[sid]
            for sid, events in fresh.items():
                # The window is stored WITH the events it produced: a source
                # in back-off keeps yesterday's reach, and covered_window()
                # needs to know that rather than trust the promise.
                self._sources[sid] = {"fetched_at": stamp, "events": events,
                                      "window": (start, end)}
            self._record_outcomes(wanted, set(fresh), failed, stamp)
        self.errors = errors
        if fresh:
            self._save_cache()
        return ok_any

    # -- threading ---------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="calendar-refresh",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.configured:
                try:
                    self.refresh()
                except Exception:  # noqa: BLE001 - the loop must survive
                    log.exception("calendar refresh crashed")
            self._wake.wait(self.refresh_s)
            self._wake.clear()

    def trigger_refresh(self) -> None:
        """Ask for a refresh without waiting: wake the worker, or run one
        one-shot daemon thread when the worker was never started."""
        if not self.configured:
            return
        if self._thread and self._thread.is_alive():
            self._wake.set()
            return
        threading.Thread(target=self.refresh, name="calendar-refresh-once",
                         daemon=True).start()

    def get(self, range: str = "today", now: datetime = None) -> Snapshot:
        """Never fetches: the cached events plus staleness; a stale cache
        triggers a background refresh, at most one per refresh period.

        The rate limit is the whole point of ``_last_trigger``: while any
        source is down the cache reads stale on every call, and without it
        each calendar question queued another full refresh -- another
        FETCH_TIMEOUT wait on the dead feed -- on the worker thread.  Six
        such off-cycle Canvas timeouts are in the log of 2026-08-31."""
        stale = self.is_stale()
        if stale:
            self._trigger_if_due()
        return Snapshot(events=self.events(), fetched_at=self.fetched_at,
                        stale=stale, errors=list(self.errors),
                        down=self.down())

    def _trigger_if_due(self) -> bool:
        """One background refresh per refresh period, however often asked."""
        now = self._clock()
        last = self._last_trigger
        if last is not None and now - last < self.refresh_s:
            return False
        self._last_trigger = now
        self.trigger_refresh()
        return True


def _redact(text: str, *secrets: str) -> str:
    for secret in secrets:
        if secret and len(secret) > 3:
            text = text.replace(secret, "•••")
    return re.sub(r"https?://\S+", "<url>", text)


# ----------------------------------------------------------------- tools
def make_source(cfg, services=None, **kw) -> CalendarSource:
    """The one CalendarSource of the process, parked on ``services.calendar``."""
    source = getattr(services, "calendar", None) if services is not None else None
    if isinstance(source, CalendarSource):
        return source
    source = CalendarSource(cfg, **kw)
    if services is not None:
        try:
            setattr(services, "calendar", source)
        except Exception:  # noqa: BLE001 - read-only namespaces are fine
            pass
    return source


# --------------------------------------------------------------- writing
#
# Writing is a different category from reading: a misheard time becomes a real
# object on the user's phone. Policy (chosen 2026-08-28): add outright when the
# parse is unambiguous, confirm when it is not -- so this predicate carries the
# whole safety of the feature and is deliberately conservative.
_DAY_WORDS = ("today", "tomorrow", "tonight", "monday", "tuesday", "wednesday",
              "thursday", "friday", "saturday", "sunday",
              "january", "february", "march", "april", "may", "june", "july",
              "august", "september", "october", "november", "december")
_EXPLICIT_TIME = re.compile(
    r"\b\d{1,2}\s*[:.]\s*\d{2}\b"      # 4:10, 09.15
    r"|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)\b"   # 4 pm, 11am
    r"|\bnoon\b|\bmidnight\b", re.I)
_DATE_NUMERIC = re.compile(r"\b\d{1,2}[/-]\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b")


DEFAULT_WRITE_CALENDAR = "Calendar"
# Feeds that mirror data Jarvis does not own -- the university's advising and
# LMS calendars. Writing there could corrupt a subscription the user cannot
# easily repair, so they are refused even when named explicitly.
PROTECTED_CALENDARS = ("Navigate360", "Navigate Student", "Canvas")


def _cal_name(cal) -> str:
    try:
        import caldav
        props = cal.get_properties([caldav.elements.dav.DisplayName()])
        name = props.get("{DAV:}displayname")
        if name:
            return str(name)
    except Exception:                       # noqa: BLE001 - fakes and odd servers
        pass
    return str(getattr(cal, "name", "") or "")


def pick_write_calendar(calendars, requested: Optional[str]):
    """The calendar to write an event to. Raises ValueError with a spoken-
    friendly message rather than guessing, because guessing here means the
    event lands somewhere the user will not look."""
    named = [(c, _cal_name(c)) for c in calendars]

    def takes_events(cal) -> bool:
        try:
            return "VEVENT" in cal.get_supported_components()
        except Exception:                   # noqa: BLE001
            return True                     # servers that do not say: assume yes

    if requested:
        want = requested.strip().lower()
        for cal, name in named:
            if name.lower() == want:
                if any(p.lower() in name.lower() for p in PROTECTED_CALENDARS):
                    raise ValueError(f"{name} is read-only, sir")
                if not takes_events(cal):
                    raise ValueError(f"{name} does not take events, sir")
                return cal
        raise ValueError(f"I don't have a calendar called {requested}, sir")

    for cal, name in named:
        if name.lower() == DEFAULT_WRITE_CALENDAR.lower() and takes_events(cal):
            return cal
    for cal, name in named:
        if takes_events(cal) and not any(
                p.lower() in name.lower() for p in PROTECTED_CALENDARS):
            return cal
    raise ValueError("I don't have a calendar I can write to, sir")


def build_vevent(title: str, start: datetime, end: datetime) -> bytes:
    """A minimal, valid VEVENT. iCloud rejects events without UID/DTSTAMP."""
    import uuid as _uuid

    import icalendar

    cal = icalendar.Calendar()
    cal.add("prodid", "-//Jarvis//EN")
    cal.add("version", "2.0")
    ev = icalendar.Event()
    ev.add("summary", title)
    ev.add("dtstart", start)
    ev.add("dtend", end)
    ev.add("dtstamp", datetime.now(timezone.utc))
    ev.add("uid", f"{_uuid.uuid4()}@jarvis")
    cal.add_component(ev)
    return cal.to_ical()


# The undo of an add. Three outcomes, and the third is the point: when the
# server hands back no deletable object there IS no undo, and saying so is
# the only honest answer -- "scratch that" used to fall through in silence,
# which reads as success while the event sits in his calendar.
UNDONE_LINE = "Taken back off your calendar, sir."
UNDO_FAILED_LINE = "I couldn't remove {title} from your calendar, sir."
CANNOT_UNDO_LINE = ("I can put events on your calendar, sir, but I can't take "
                    "one back off — you'll want to delete {title} yourself.")


def _undo_add(saved, title: str):
    """The closure that removes the event just written, or one that says why
    it cannot.

    ``caldav``'s ``save_event`` hands back the created Event object, whose
    ``.delete()`` is the only handle to it we ever get: nothing else knows
    its UID or its href. A server (or a stand-in) that returns something
    without a delete gives us nothing to remove, and the caller must be able
    to tell the difference.
    """
    delete = getattr(saved, "delete", None)
    if not callable(delete):
        def _cannot() -> str:
            log.info("calendar: no delete path for %r; undo refused", title)
            return CANNOT_UNDO_LINE.format(title=title)
        return _cannot

    def _undo() -> str:
        try:
            delete()
        except Exception:                   # noqa: BLE001 - server boundary
            log.exception("calendar: deleting %r failed", title)
            return UNDO_FAILED_LINE.format(title=title)
        log.info("calendar: %r removed again", title)
        return UNDONE_LINE
    return _undo


def add_event(calendars, title: str, start: datetime, end: datetime,
              calendar_name: Optional[str] = None) -> tuple:
    """Add one event -> (line to speak, undo callable). Raises on refusal or
    failure.

    Server errors propagate rather than being smoothed into a success line:
    telling the user an event was added when it was not is the worst outcome
    available here. The undo callable never raises -- it returns the line to
    say, including the honest refusal when this server gives no delete path.
    """
    target = pick_write_calendar(calendars, calendar_name)   # raises ValueError
    name = _cal_name(target) or DEFAULT_WRITE_CALENDAR
    saved = target.save_event(build_vevent(title, start, end))
    log.info("calendar: added %r to %s at %s", title, name, start.isoformat())
    when = start.strftime("%A at %-I:%M %p").replace(" 0", " ")
    # The default list is literally called "Calendar"; "your Calendar calendar"
    # reads like a stutter out loud.
    where = "your calendar" if name.lower() == DEFAULT_WRITE_CALENDAR.lower() \
        else f"your {name} calendar"
    return f"Added {title}, {when}, to {where}, sir.", _undo_add(saved, title)


# The tool handler inside make_tools is ALSO called add_event -- that is the
# name the model calls -- and shadows this one inside that closure. This
# alias is how it reaches the real writer.
_add_event = add_event


def write_event(calendars, title: str, start: datetime, end: datetime,
                calendar_name: Optional[str] = None) -> str:
    """``add_event`` without the undo, for callers that cannot use one."""
    return add_event(calendars, title, start, end, calendar_name)[0]


def event_confidence(text: str, now: datetime) -> tuple[bool, str]:
    """(confident, reason). Confident means: add it without asking.

    Requires explicit evidence of BOTH a day and a time in what the user
    actually said, plus something left over to use as a title. It does NOT
    trust parse_when_full simply returning a datetime: that resolves "at four"
    by a heuristic (bare hours become 7-11am / 12 noon / 1-6pm), and a guess is
    precisely the case confirmation exists for.
    """
    from jarvis.tools.timekeeper import parse_when_full

    raw = (text or "").strip()
    if not raw:
        return False, "nothing to add"

    lowered = raw.lower()
    has_day = any(w in lowered for w in _DAY_WORDS) or bool(_DATE_NUMERIC.search(lowered))
    has_time = bool(_EXPLICIT_TIME.search(lowered))

    try:
        when, _repeat, leftover = parse_when_full(raw, now)
    except Exception:                       # noqa: BLE001 - parser is best effort
        log.exception("event parse failed")
        return False, "I could not work out when"

    if when is None:
        return False, "I could not work out when"
    if not (leftover or "").strip():
        return False, "I did not catch a title for it"
    if not has_day:
        return False, "you did not say which day"
    if not has_time:
        return False, "you did not say a clear time"
    return True, ""


def expand_people(services, title: str) -> str:
    """'lunch with mom' -> 'lunch with Linda Peyrovi' through the people
    book (jarvis.memory.expand_aliases); unchanged when nothing matches
    or no memory is wired."""
    memory = getattr(services, "memory", None) if services is not None else None
    expand = getattr(memory, "expand_aliases", None)
    if not callable(expand) or not title:
        return title
    try:
        return str(expand(title) or title)
    except Exception:                                # noqa: BLE001
        log.debug("expand_aliases failed", exc_info=True)
        return title


def make_tools(cfg, services) -> list[ToolSpec]:
    source = make_source(cfg, services)

    def get_calendar(range="today", **_) -> ToolResult:
        if not source.configured:
            line = setup_line(cfg, "google_ical")
            return ToolResult(text=line, ok=False, speak=line)
        now = now_local(source.tz)
        rng = coerce_range(range, now)
        # A date he cannot have meant is a QUESTION, not a guess. The silent
        # fall-back to today is the whole of the 2026-09-05 bug, so it dies
        # here as well as in coerce_range: speak= puts the question to him
        # verbatim instead of letting the model narrate around it.
        if is_ask(rng):
            line = ask_words(rng)
            return ToolResult(text=line, ok=False, speak=line)
        # A real day the cache does not hold is refused BY NAME rather than
        # answered "nothing on it" -- see out_of_reach().
        want = as_date(rng)
        if want is not None:
            beyond = out_of_reach(want, now.date(), source.window_days)
            if beyond:
                return ToolResult(text=beyond, ok=False, speak=beyond)
        snap = source.get(rng, now)
        if snap.fetched_at is None:
            if snap.errors:
                return ToolResult(text="calendar unreachable", ok=False)
            return ToolResult(text="calendar still loading, ask again in a moment",
                              ok=False)
        text = format_events(snap.events, rng, now, source.window_days)
        if snap.down:
            # A subscription is unread, so the day is NOT accounted for:
            # take back "; nothing else." before saying which feed is out.
            text = drop_completeness(text)
        if snap.stale:
            text += f" That's as of {as_of_words(snap.fetched_at, now)}."
        if snap.down:
            text += " " + down_words(snap.down, now)
        return ToolResult(text=text, max_sentences=CALENDAR_MAX_SENTENCES)

    def _range_from_words(said: str, model_args=None) -> dict:
        """The date off HIS words, for a model call that named one.

        Returns {} unless the utterance names a specific day, so the model
        keeps every range it can still get right -- including the ones it
        resolves from the conversation rather than from the sentence in
        hand.  ``sentence_date`` is the SAME reader the forced path uses:
        it used to be ``_explicit_date`` here and the whole of
        ``coerce_range`` there, and the two disagreed on "what was on my
        calendar yesterday" -- the 4th forced, TODAY from the model.

        ``model_args`` is what the model wrote (registry.call hands it
        over because the spec says derive_takes_args).  A range in it that
        reads as a real day his words do not contradict STANDS -- see
        model_day_stands; the deriver never again replaces a value the
        model got right with a worse one of its own.
        """
        try:
            now = now_local(source.tz)
            got = sentence_date(said, now.date())
            if not got:
                return {}
            if model_day_stands(said, got, (model_args or {}).get("range"), now):
                return {}
        except Exception:                    # noqa: BLE001 - tool boundary
            log.debug("calendar range derive failed", exc_info=True)
            return {}
        return {"range": got}

    def add_event(text="", calendar=None, **_) -> ToolResult:
        """Add one event. Writes outright when the parse is unambiguous;
        otherwise reads it back and waits for a yes."""
        from jarvis.tools.timekeeper import parse_when_full

        now = now_local(source.tz)
        raw = _clean(text)
        confident, reason = event_confidence(raw, now)
        try:
            when, _repeat, title = parse_when_full(raw, now)
        except Exception:                    # noqa: BLE001
            log.exception("event parse failed")
            when, title = None, ""
        # The alias expands here, after the time words are gone and before
        # either the write or the read-back: "lunch with Mom" lands on the
        # phone as lunch with her name.
        title = expand_people(services, _clean(title)) or "an event"

        if when is None:
            source.pending_event = None
            line = f"I couldn't work out when, sir — {reason}."
            return ToolResult(text=line, ok=False, speak=line)

        end = when + timedelta(hours=1)
        if not confident:
            # Stash the interpretation and read it back. Nothing is written
            # until the user says yes: a misheard time would otherwise become
            # a real event on their phone.
            # made_at: the read-back takes the next yes for OFFER_TTL_S
            # (commander._try_event_confirm) and counts as an open
            # question for exactly that long (Commander.question_open).
            source.pending_event = {"title": title, "start": when, "end": end,
                                    "calendar": calendar,
                                    "made_at": time.monotonic()}
            words = when.strftime("%A at %I:%M %p").replace(" 0", " ").lstrip("0")
            line = f"I have {title}, {words} — {reason}. Shall I add it, sir?"
            return ToolResult(text=line, speak=line)

        source.pending_event = None
        try:
            line, undo = _add_event(source.icloud_calendars(), title, when, end,
                                    calendar_name=calendar)
        except ValueError as exc:            # refusal: protected / unknown
            return ToolResult(text=str(exc), ok=False, speak=str(exc))
        except Exception as exc:             # noqa: BLE001 - server trouble
            log.exception("calendar write failed")
            line = f"I couldn't add that, sir — {type(exc).__name__}."
            return ToolResult(text=line, ok=False, speak=line)
        # Park the way back out: a ToolResult carries no undo slot, and this
        # is the path a confident add takes, so "scratch that" would
        # otherwise have nothing to reach for.
        source.last_add = {"undo": undo, "at": time.monotonic(), "title": title}
        return ToolResult(text=line, speak=line)

    return [ToolSpec(
        name="add_event",
        description=("Add one event to Hunter's calendar: what it is and when."),
        parameters={"type": "object", "properties": {
            "text": {"type": "string",
                     "description": "the event with its day and time, as said"},
            "calendar": {"type": "string",
                         "description": "target calendar name; omit for default"}},
            "required": ["text"]},
        handler=add_event), ToolSpec(
        name="get_calendar",
        # Leads with EVENTS and with the literal words "what's on today":
        # gemma4 sent "What's on today?" to get_briefing on 3 of 5 live
        # tries, because get_briefing's description opened with "today's
        # weather, calendar, ..." and this one opened with "a weekday
        # name". The two descriptions are the only thing separating a bare
        # event list from the composed morning summary, so the day word the
        # user actually says has to sit in THIS one. <= 20 words: the
        # description rides in the cached static prefix on every turn.
        # "; how long they last" is the hint that makes the model reach for
        # THIS tool on "how long is my biosensors lab tomorrow?" -- a
        # question that got "I'm afraid I don't have that information, sir"
        # because nothing in the tool list mentioned duration.
        # "a date" earns its word in a description this tight because the
        # model has to know a date is askable at all: with the enum below
        # gone it is now expressible, and 2026-09-05 showed the model
        # sending range="today" for "what do i have on september 12th".
        # ("named" gave up the word; the day word he says is what matters.)
        description=("Events on Hunter's calendar: what's on today, "
                     "tomorrow, a weekday, a date, the week, or next; "
                     "how long they last."),
        parameters={"type": "object", "properties": {
            # No enum. An enum of the eleven word ranges made an explicit
            # date IMPOSSIBLE for the model to express -- half of the
            # 2026-09-05 bug lived here rather than in coerce_range.
            "range": {"type": "string",
                      "description": ("today, tomorrow, a weekday, week, "
                                      "next, or a date like 2026-09-12")}},
            "required": ["range"]},
        # The model fills range from his words and got it wrong for every
        # explicit date. Only an EXPLICIT date is taken over its value: a
        # follow-up it resolved from the conversation ("and the next day")
        # names no date here and keeps its own answer -- and so does a
        # date it got RIGHT (derive_takes_args: the deriver sees the
        # model's value and applies model_day_stands).
        derive=_range_from_words,
        derive_takes_args=True,
        handler=get_calendar)]
