"""Timekeeper: persistent reminders, timers and alarms.

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 6.1
(also 3.1 events, 3.3 test policy, 3.4 persona lines, 4.1 registry).

One SQLite table (``items``) survives restarts; a daemon scheduler thread
ticks once a second and fires whatever is due.  Reminders and timers are
spoken once and published as ``ReminderFired``; alarms ring through
``paplay`` until dismissed / snoozed / timed out and publish ``AlarmFired``
/ ``AlarmStopped`` for the UI modal.  On boot ``catch_up()`` fires items
that came due while the app was down (< 1 h late, with the "while I was
down" preamble) or announces them as missed.

Seams for tests: ``now`` (a clock callable), ``run`` (the subprocess seam
``_run``: paplay / notify-send), ``say`` (speech), ``tick()`` (one
scheduler pass, no thread needed).  ``parse_when`` / ``describe_due`` are
pure functions of the ``now`` they are given.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from jarvis.events import AlarmFired, AlarmStopped, ReminderFired, bus
from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.timekeeper")

# ------------------------------------------------------------- constants
DEFAULT_SOUND = "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"
MAX_VOLUME = 65536                 # paplay --volume scale (100 %)
DEFAULT_VOLUME = 0.8
DEFAULT_MAX_RING_S = 300
DEFAULT_SNOOZE_MIN = 10
ESCALATE_AFTER_S = 30.0
RING_GAP_S = 1.0
ESCALATED_GAP_S = 0.3
LATE_GRACE_S = 3600.0
NOW_WINDOW_S = 5.0                 # |delta| under this is described as "now"              # < 1 h late on boot -> fire, else missed
MAX_SALVAGE_BYTES = 512 * 1024 * 1024   # sanity bound on a padded copy
MAX_SALVAGE_ROWS = 100_000              # sanity bound on a rowid scan
SALVAGE_GIVE_UP = 5_000                 # consecutive unreadable rowids -> stop
DEFAULT_HOUR = 9                   # "tomorrow" / "on friday" with no time
MORNING_HOUR = 7                   # same, for alarms (prefer="morning")

KINDS = ("reminder", "timer", "alarm")
ACTIVE_STATES = ("pending", "ringing", "snoozed")
STATES = ACTIVE_STATES + ("done", "missed", "cancelled")
REPEATS = ("", "daily", "weekdays")

# Persona lines (spec 3.4) — fixed strings the app prewarms.
REMINDER_LINE = "Sir, this is your reminder. {text}"
TIMER_LINE = "Sir, your {n} timer is up."
TIMER_LABEL_LINE = "Sir, your {n} {label} timer is up."
ALARM_LINE = "Sir, it's {time}. {label}"
ALARM_DEFAULT_LABEL = "Time to get up."
LATE_LINE = "While I was down, sir: {label}, due at {time}."
MISSED_LINE = "You missed {what} while I was down, sir."
RING_TIMEOUT_LINE = "You missed {label} at {time}, sir; I've stopped ringing."
CANT_PARSE_LINE = ("I couldn't make out the time, sir; try 'in ten minutes'"
                   " or 'at seven'.")
CORRUPT_STORE_LINE = (
    "Sir, my schedule store was damaged, so anything I had scheduled may be"
    " lost. I've set the old file aside as {name} and started a fresh one.")
CORRUPT_RECOVERED_LINE = (
    "Sir, my schedule store was damaged, but I recovered {n} of your"
    " scheduled items. The old file is set aside as {name}.")
NO_TIMEKEEPER_LINE = "I'm afraid my timekeeper isn't running, sir."
NOTHING_RINGING_LINE = "Nothing's ringing, sir."
HOW_LONG_LINE = "How long for, sir?"

_COUNT_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"]


# --------------------------------------------------------------- seams
def _run(argv, background=False, timeout=30.0):
    """Subprocess seam (tests replace it with a recorder).  ``background``
    returns the Popen handle (or None), otherwise waits.  Never raises."""
    try:
        if background:
            return subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        return subprocess.run(argv, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout)
    except Exception as exc:                # noqa: BLE001 - seam boundary
        log.warning("run %s failed: %s", argv[:1], exc)
        return None


def _cfg_get(cfg, key: str, default=None):
    """``cfg.get("alarms.volume")`` for an AssistantConfig, a nested dict,
    a SimpleNamespace tree, or None (-> default)."""
    if cfg is None:
        return default
    if not isinstance(cfg, dict):
        getter = getattr(cfg, "get", None)
        if callable(getter):
            try:
                value = getter(key, default)
                return default if value is None else value
            except TypeError:
                pass
    obj = cfg
    for part in key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj


def _default_cache_dir() -> Path:
    try:
        from jarvis.config import PATHS
        return Path(getattr(PATHS, "CACHE_DIR", Path.home() / ".cache" / "jarvis"))
    except Exception:                       # noqa: BLE001
        return Path.home() / ".cache" / "jarvis"


def _default_legacy_path() -> Path:
    try:
        from jarvis.config import PATHS
        return Path(PATHS.REMINDERS)
    except Exception:                       # noqa: BLE001
        return Path.home() / ".aiws_trainer" / "jarvis_memory" / "reminders.json"


# ------------------------------------------------------------ wording
def count_words(n: int) -> str:
    n = int(n)
    return _COUNT_WORDS[n] if 0 <= n < len(_COUNT_WORDS) else str(n)


def sentence_case(text: str) -> str:
    """User text spoken as its own sentence: capitalised, full-stopped.
    Words that are already capitalised inside the text are left alone."""
    t = str(text or "").strip()
    if not t:
        return t
    if t[0].islower():
        t = t[0].upper() + t[1:]
    if not t.endswith((".", "!", "?", ":", ";")):
        t += "."
    return t


def join_and(parts) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def duration_words(seconds: float) -> str:
    """'5-minute', '90-second', '2-hour', '1-and-a-half-hour'."""
    s = int(round(float(seconds)))
    if s >= 3600 and s % 3600 == 0:
        return f"{s // 3600}-hour"
    if s >= 3600 and s % 1800 == 0:
        return f"{s // 3600}-and-a-half-hour"
    if s >= 60:
        m, r = divmod(s, 60)
        if r == 30:
            return f"{m}-and-a-half-minute"
        return f"{int(round(s / 60))}-minute"
    return f"{s}-second"


def _to_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromtimestamp(float(value))


def fmt_clock(value) -> str:
    """'7:00 am' / '12:05 pm' — the plain clock reading."""
    d = _to_dt(value)
    h = d.hour % 12 or 12
    return f"{h}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'}"


def _time_words(d: datetime) -> str:
    if d.hour == 12 and d.minute == 0:
        return "noon"
    if d.hour == 0 and d.minute == 0:
        return "midnight"
    return fmt_clock(d)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _day_words(d: datetime, n: datetime) -> str:
    """'' (today) / 'tomorrow' / 'yesterday' / 'on Friday' / 'on the 4th of
    September'."""
    days = (d.date() - n.date()).days
    if days == 0:
        return ""
    if days == 1:
        return "tomorrow"
    if days == -1:
        return "yesterday"
    if 1 < days < 7:
        return "on " + d.strftime("%A")
    if -7 < days < -1:
        return "last " + d.strftime("%A")
    return f"on the {_ordinal(d.day)} of {d.strftime('%B')}"


def _when_words(d: datetime, n: datetime) -> str:
    """'7:00 am' / '7:00 am yesterday' / '7:00 am on Monday'."""
    day = _day_words(d, n)
    return _time_words(d) + (f" {day}" if day else "")


def _rel_words(seconds: float) -> str:
    """Relative wording for < 2 h: 'a minute', '10 minutes', 'an hour and
    a half'."""
    m = int(round(seconds / 60.0))
    if seconds < 45:
        return f"{int(round(seconds))} seconds"
    if m <= 1:
        return "a minute"
    if m < 60:
        return f"{m} minutes"
    if m == 60:
        return "an hour"
    if m == 90:
        return "an hour and a half"
    if m < 120:
        return f"an hour and {m - 60} minutes"
    h, r = divmod(m, 60)
    if r == 0:
        return f"{h} hours"
    if r == 30:
        return f"{h} and a half hours"
    return f"{h} hours and {r} minutes"


def describe_due(due, now=None) -> str:
    """'in 10 minutes' / 'at 3:00 pm' / 'at 7:00 am tomorrow' / 'at 9:00 am
    on Friday' / '10 minutes ago'."""
    d = _to_dt(due)
    n = _to_dt(now if now is not None else time.time())
    delta = (d - n).total_seconds()
    if delta < -NOW_WINDOW_S:
        if -delta < 7200 and d.date() == n.date():
            return f"{_rel_words(-delta)} ago"
        return f"at {_when_words(d, n)}"
    if abs(delta) < NOW_WINDOW_S:
        return "now"
    if delta < 7200 and d.date() == n.date():
        return f"in {_rel_words(delta)}"
    return f"at {_when_words(d, n)}"


# --------------------------------------------------------------- parsing
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90}
_TENS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
         "ninety"}
_UNIT_S = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
    "day": 86400, "days": 86400, "week": 604800, "weeks": 604800}
_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6}
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_PERIOD_HOUR = {"morning": 8, "afternoon": 14, "evening": 18, "night": 21}
_FILLER_LEAD = {"to", "that", "about", "and", "then", "please", "for", "me",
                "of", "-", "—", ":"}

_GLUED = re.compile(
    r"^(\d{1,2}:\d{2}|\d+(?:\.\d+)?)"
    r"(am|pm|oclock|mins?|minutes?|secs?|seconds?|hrs?|hours?|h|m|s)$")
_TIME_RE = re.compile(
    r"(?:\b(?P<at>at|around|by)\s+)?\b(?:"
    r"(?P<h>\d{1,2})(?:(?::|\s)(?P<m>[0-5]\d))?"
    r"(?:\s*(?P<ap>am|pm)|\s*(?P<oc>oclock))?"
    r"|(?P<noon>noon|midday)|(?P<mid>midnight))\b")
_PAST_RE = re.compile(
    r"\b(?P<frac>half|quarter)\s+(?P<dir>past|to)\s+(?P<h>\d{1,2})"
    r"(?:\s*(?P<ap>am|pm))?\b")
_PERIOD_RE = re.compile(
    r"\b(?:in\s+the\s+|at\s+|this\s+)?(?P<p>morning|afternoon|evening|night)\b")
_REPEAT_RE = re.compile(
    r"\b(?:(?:every|each)\s+(?P<what>day|morning|afternoon|evening|night|"
    r"weekday|weekdays|week\s*day)|(?P<daily>daily)|"
    r"(?:on\s+)?(?P<wk>weekdays))\b")
_DAY_RE = re.compile(
    r"\b(?:(?P<dat>day\s+after\s+tomorrow)|(?P<tom>tomorrow|tmrw)|"
    r"(?P<tonight>tonight)|(?P<today>today)|(?P<nextweek>next\s+week)|"
    r"(?:(?P<mod>next|this|on|coming)\s+)?(?P<wd>monday|mon|tuesday|tues|tue|"
    r"wednesday|wed|thursday|thurs|thur|thu|friday|fri|saturday|sat|"
    r"sunday|sun))\b")
_DATE_RE = re.compile(
    r"\b(?:on\s+)?(?:(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\.?\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?"
    r"|(?P<d2>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?"
    r"(?P<mon2>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)\b")
_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[T ](\d{1,2}:\d{2})(?::\d{2})?)?\b")


class _Scan:
    """Tokenised text with per-token 'used' flags so the leftover words
    (the reminder text) can be recovered with their original spelling."""

    def __init__(self, text: str):
        self.orig = text.split()
        self.norm: list[str] = []
        self.span: list[tuple[int, int]] = []     # norm idx -> orig [i0, i1)
        for i, word in enumerate(self.orig):
            w = word.lower()
            w = (w.replace("a.m.", "am").replace("p.m.", "pm")
                 .replace("a.m", "am").replace("p.m", "pm")
                 .replace("o'clock", "oclock").replace("o’clock", "oclock"))
            w = re.sub(r"^(\d{1,2})\.(\d{2})$", r"\1:\2", w)
            w = w.strip(".,;!?\"'()[]")
            for sub in w.replace("-", " ").replace("/", " ").split():
                m = _GLUED.match(sub)
                if m:
                    self._add(m.group(1), i)
                    self._add(m.group(2), i)
                else:
                    self._add(sub, i)
        self._numbers()
        self.used = [False] * len(self.norm)

    def _add(self, tok: str, i: int):
        self.norm.append(tok)
        self.span.append((i, i + 1))

    def _numbers(self):
        """'forty five' -> '45', 'seven' -> '7' (never 'a'/'an')."""
        out, spans = [], []
        i = 0
        while i < len(self.norm):
            t = self.norm[i]
            if t in _TENS and i + 1 < len(self.norm) and \
                    self.norm[i + 1] in _WORD_NUMS and \
                    0 < _WORD_NUMS[self.norm[i + 1]] < 10:
                out.append(str(_WORD_NUMS[t] + _WORD_NUMS[self.norm[i + 1]]))
                spans.append((self.span[i][0], self.span[i + 1][1]))
                i += 2
                continue
            if t in _WORD_NUMS:
                out.append(str(_WORD_NUMS[t]))
            else:
                out.append(t)
            spans.append(self.span[i])
            i += 1
        self.norm, self.span = out, spans

    # -- matching on the unused tokens ---------------------------------
    def joined(self):
        """(string of unused tokens, [(char_start, char_end, norm_idx)])."""
        parts, idx, pos = [], [], 0
        for i, t in enumerate(self.norm):
            if self.used[i]:
                continue
            idx.append((pos, pos + len(t), i))
            parts.append(t)
            pos += len(t) + 1
        return " ".join(parts), idx

    def search(self, regex):
        text, idx = self.joined()
        m = regex.search(text)
        if not m:
            return None
        a, b = m.span()
        for s, e, i in idx:
            if s < b and e > a:
                self.used[i] = True
        return m

    def use(self, i0: int, i1: int):
        for i in range(i0, i1):
            self.used[i] = True

    def rest(self) -> str:
        gone = set()
        for i, u in enumerate(self.used):
            if u:
                gone.update(range(*self.span[i]))
        words = [w for i, w in enumerate(self.orig) if i not in gone]
        text = " ".join(words)
        text = re.sub(r"^(?:remind\s+me|set\s+(?:a\s+|an\s+)?(?:reminder|timer|alarm)"
                      r"(?:\s+for)?|wake\s+me(?:\s+up)?)\b\s*", "", text, flags=re.I)
        words = text.split()
        while words and words[0].lower().strip(",:;-—") in _FILLER_LEAD | {""}:
            words.pop(0)
        while words and words[-1].lower().strip(",:;-—.") in {"please", "", "at", "on", "in"}:
            words.pop()
        return " ".join(words).strip(" ,;:-—")


def _num(tok: str):
    try:
        return float(tok)
    except ValueError:
        return None


def _duration_at(norm, used, i, allow_bare=True):
    """Parse '10 minutes', 'an hour and a half', 'half an hour', '2 hours
    and 15 minutes', 'a couple of minutes' starting at norm[i].
    ``allow_bare``: a lone number counts as minutes ("in 5").
    Returns (seconds, next_index) or None."""
    total = 0.0
    j = i
    groups = 0
    while j < len(norm) and not used[j]:
        t = norm[j]
        n = None
        k = j
        if t in ("a", "an"):
            n, k = 1.0, j + 1
            if k < len(norm) and norm[k] == "couple":
                n, k = 2.0, k + 1
                if k < len(norm) and norm[k] == "of":
                    k += 1
            elif k < len(norm) and norm[k] == "few":
                n, k = 3.0, k + 1
        elif t == "half" and j + 1 < len(norm) and norm[j + 1] in ("a", "an"):
            n, k = 0.5, j + 2
        elif t == "quarter" and j + 2 < len(norm) and norm[j + 1] == "of" \
                and norm[j + 2] in ("a", "an"):
            n, k = 0.25, j + 3
        elif _num(t) is not None:
            n, k = _num(t), j + 1
        else:
            break
        # "one and a half hours"
        if k + 2 < len(norm) and norm[k] == "and" and norm[k + 1] == "a" \
                and norm[k + 2] in ("half", "quarter"):
            n += 0.5 if norm[k + 2] == "half" else 0.25
            k += 3
        if k < len(norm) and norm[k] in _UNIT_S:
            unit = _UNIT_S[norm[k]]
            k += 1
            # "an hour and a half"
            if k + 2 < len(norm) and norm[k] == "and" and norm[k + 1] == "a" \
                    and norm[k + 2] in ("half", "quarter"):
                n += 0.5 if norm[k + 2] == "half" else 0.25
                k += 3
        elif groups == 0 and allow_bare and _num(t) is not None and \
                (k >= len(norm) or norm[k] not in _UNIT_S) and \
                (k >= len(norm) or norm[k] not in ("am", "pm", "oclock")):
            unit = 60                         # bare "in 5" -> minutes
        else:
            break
        total += n * unit
        groups += 1
        j = k
        if j < len(norm) and norm[j] == "and" and j + 1 < len(norm) and \
                (norm[j + 1] in ("a", "an") or _num(norm[j + 1]) is not None):
            j += 1
            continue
        if j < len(norm) and (_num(norm[j]) is not None or norm[j] in ("a", "an")):
            continue
        break
    if groups == 0:
        return None
    return total, j


def _find_duration(sc: _Scan):
    """'in <dur>' / 'after <dur>' / leading '<dur>' (+ 'from now')."""
    for i, t in enumerate(sc.norm):
        if sc.used[i]:
            continue
        if t in ("in", "after") and i + 1 < len(sc.norm):
            got = _duration_at(sc.norm, sc.used, i + 1)
            if got:
                sc.use(i, got[1])
                return got[0]
    got = _duration_at(sc.norm, sc.used, 0, allow_bare=(len(sc.norm) == 1))
    if got:
        sc.use(0, got[1])
        j = got[1]
        if j + 1 < len(sc.norm) and sc.norm[j] == "from" and sc.norm[j + 1] == "now":
            sc.use(j, j + 2)
        return got[0]
    return None


def repeat_from_text(text: str) -> str:
    """'' | 'daily' | 'weekdays' from 'every day', 'daily', 'every weekday',
    'weekdays', 'every morning'."""
    m = _REPEAT_RE.search((text or "").lower())
    if not m:
        return ""
    what = (m.group("what") or "").replace(" ", "")
    if m.group("wk") or what in ("weekday", "weekdays"):
        return "weekdays"
    return "daily"


def normalize_repeat(value) -> str:
    v = str(value or "").strip().lower()
    if v in ("", "once", "no", "none", "never", "false", "0"):
        return ""
    if "weekday" in v or "week day" in v or v in ("mon-fri", "monday to friday"):
        return "weekdays"
    if v in ("daily", "every day", "everyday", "each day", "day", "true", "yes",
             "repeat", "every morning", "every night", "every evening"):
        return "daily"
    return repeat_from_text(v) or ""


def parse_when_full(text: str, now: datetime, prefer: str = "next"):
    """Full parse: (datetime | None, repeat, leftover_text).

    ``prefer`` = 'next' (reminders: a bare hour is the next occurrence; a
    bare hour on a named day is 7-11 am, 12 noon, 1-6 pm) or 'morning'
    (alarms: a bare hour is am)."""
    text = (text or "").strip()
    if not text:
        return None, "", ""
    tz = now.tzinfo
    now_naive = now.replace(tzinfo=None)

    iso = _ISO_RE.search(text)
    if iso:
        try:
            dt = datetime.fromisoformat(iso.group(1) + (" " + iso.group(2) if iso.group(2) else ""))
            if not iso.group(2):
                dt = dt.replace(hour=DEFAULT_HOUR)
            rest = (text[:iso.start()] + " " + text[iso.end():]).strip()
            return (dt.replace(tzinfo=tz) if tz else dt), repeat_from_text(text), rest
        except ValueError:
            pass

    sc = _Scan(text)
    repeat = ""
    m = sc.search(_REPEAT_RE)
    period = None
    if m:
        what = (m.group("what") or "").replace(" ", "")
        repeat = "weekdays" if (m.group("wk") or what in ("weekday", "weekdays")) else "daily"
        if what in _PERIOD_HOUR:
            period = what

    duration = _find_duration(sc)

    # -- time of day ------------------------------------------------------
    hour = minute = None
    ampm = None
    explicit24 = False
    m = sc.search(_PAST_RE)
    if m:
        h = int(m.group("h"))
        frac = 30 if m.group("frac") == "half" else 15
        if m.group("dir") == "past":
            hour, minute = h, frac
        else:
            hour, minute = (h - 1) % 24, 60 - frac
        ampm = m.group("ap")
    else:
        text_now, _ = sc.joined()
        for cand in _TIME_RE.finditer(text_now):
            if cand.group("noon") or cand.group("mid"):
                break
            bare = not (cand.group("at") or cand.group("ap") or cand.group("oc")
                        or cand.group("m") is not None)
            if bare:
                continue
            h = int(cand.group("h"))
            if h > 24 or (cand.group("ap") and h > 12):
                continue
            break
        else:
            cand = None
        if cand is not None:
            m = sc.search(re.compile(re.escape(cand.group(0))))
            if cand.group("noon"):
                hour, minute, ampm = 12, 0, "pm"
            elif cand.group("mid"):
                hour, minute, ampm = 0, 0, "am"
                explicit24 = True
            else:
                hour = int(cand.group("h"))
                minute = int(cand.group("m") or 0)
                ampm = cand.group("ap")
                if hour > 12 or hour == 0:
                    explicit24 = True
                    hour %= 24
    if hour is not None and ampm:
        hour = hour % 12 + (12 if ampm == "pm" else 0)

    # -- day ------------------------------------------------------------
    day_offset = None            # days from today
    weekday = None
    weekday_strict = False
    explicit_today = False
    date_md = None
    m = sc.search(_DAY_RE)
    if m:
        if m.group("dat"):
            day_offset = 2
        elif m.group("tom"):
            day_offset = 1
        elif m.group("tonight"):
            day_offset, explicit_today, period = 0, True, "night"
        elif m.group("today"):
            day_offset, explicit_today = 0, True
        elif m.group("nextweek"):
            day_offset = 7
        elif m.group("wd"):
            weekday = _WEEKDAYS[m.group("wd")]
            weekday_strict = (m.group("mod") == "next")
    else:
        m = sc.search(_DATE_RE)
        if m:
            mon = _MONTHS[(m.group("mon") or m.group("mon2"))]
            day = int(m.group("d") or m.group("d2"))
            date_md = (mon, day)

    m = sc.search(_PERIOD_RE)
    if m:
        period = m.group("p")
        if m.group(0).startswith("this"):
            explicit_today = True
            day_offset = 0 if day_offset is None else day_offset

    # -- resolve --------------------------------------------------------
    has_day = day_offset is not None or weekday is not None or date_md is not None
    if duration is not None and hour is None and period is None and not has_day:
        dt = now_naive + timedelta(seconds=duration)
        return (dt.replace(tzinfo=tz) if tz else dt), repeat, sc.rest()

    if duration is not None:
        base = (now_naive + timedelta(seconds=duration)).date()
    elif day_offset is not None:
        base = now_naive.date() + timedelta(days=day_offset)
    elif weekday is not None:
        ahead = (weekday - now_naive.weekday()) % 7
        if weekday_strict and ahead == 0:
            ahead = 7
        base = now_naive.date() + timedelta(days=ahead)
    elif date_md is not None:
        try:
            base = date(now_naive.year, date_md[0], date_md[1])
        except ValueError:
            return None, repeat, sc.rest()
        if base < now_naive.date():
            base = date(now_naive.year + 1, date_md[0], date_md[1])
    else:
        base = None

    extra_day = 0
    bare = hour is not None and not ampm and not explicit24
    if hour is not None:
        if bare and period:
            if period == "morning":
                hour = hour % 12 if hour != 12 else 12
            else:
                if hour == 12 and period == "night":
                    hour, extra_day = 0, 1
                elif hour < 12:
                    hour += 12
            bare = False
    elif period:
        hour, minute = _PERIOD_HOUR[period], 0
    elif base is not None or duration is not None:
        hour, minute = (MORNING_HOUR if prefer == "morning" else DEFAULT_HOUR), 0
    else:
        return None, repeat, sc.rest()

    def at(d: date, h: int, mi: int) -> datetime:
        return datetime.combine(d, datetime.min.time()).replace(hour=h, minute=mi)

    dt = None
    if bare:
        if base is None or (explicit_today and day_offset == 0):
            today = now_naive.date()
            if hour == 12:
                cands = [at(today, 12, minute), at(today + timedelta(days=1), 12, minute)]
            elif prefer == "morning":
                cands = [at(today, hour, minute), at(today + timedelta(days=1), hour, minute)]
            else:
                cands = [at(today, hour, minute), at(today, hour + 12, minute),
                         at(today + timedelta(days=1), hour, minute)]
            if explicit_today:
                cands = [c for c in cands if c.date() == today]
            for c in cands:
                if c > now_naive:
                    dt = c
                    break
            if dt is None:
                return None, repeat, sc.rest()
        else:
            if prefer == "morning" or hour == 12 or hour >= 7:
                hh = hour
            else:
                hh = hour + 12
            dt = at(base, hh, minute)
    else:
        if base is None:
            dt = at(now_naive.date(), hour, minute) + timedelta(days=extra_day)
            if dt <= now_naive:
                dt += timedelta(days=1)
        else:
            dt = at(base, hour, minute) + timedelta(days=extra_day)

    if dt <= now_naive:
        if weekday is not None:
            dt += timedelta(days=7)
        elif explicit_today and not bare and period and hour == _PERIOD_HOUR.get(period):
            # "tonight" / "this evening" already past: the next whole hour
            dt = (now_naive + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        elif explicit_today:
            return None, repeat, sc.rest()
        elif date_md is not None:
            dt = dt.replace(year=dt.year + 1)
    if repeat == "weekdays":
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
    return (dt.replace(tzinfo=tz) if tz else dt), repeat, sc.rest()


def parse_when(text: str, now: datetime, prefer: str = "next") -> Optional[datetime]:
    """Natural-language time -> datetime (same tz-awareness as ``now``), or
    None when no time can be found.  See ``parse_when_full``."""
    return parse_when_full(text, now, prefer)[0]


def split_when(text: str, now: datetime, prefer: str = "next"):
    """(datetime | None, leftover words) — 'in 10 minutes to call mum'
    -> (now+10m, 'call mum')."""
    dt, _, rest = parse_when_full(text, now, prefer)
    return dt, rest


def parse_duration(value) -> Optional[float]:
    """'15' -> 900 (minutes), 15 -> 900, '90 seconds' -> 90, '1.5 hours'
    -> 5400, 'an hour and a half' -> 5400.  None when unparseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * 60 if value > 0 else None
    s = str(value).strip().lower()
    if not s:
        return None
    n = _num(s)
    if n is not None:
        return n * 60 if n > 0 else None
    sc = _Scan(s)
    got = _duration_at(sc.norm, sc.used, 0)
    if got is None and sc.norm and sc.norm[0] in ("in", "for", "after"):
        got = _duration_at(sc.norm, sc.used, 1)
    if got and got[0] > 0:
        return float(got[0])
    return None


# ------------------------------------------------------------------ data
@dataclass
class Item:
    id: str
    kind: str
    label: str
    due: float
    created: float
    repeat: str = ""
    state: str = "pending"
    snooze_until: Optional[float] = None
    fired_at: Optional[float] = None
    note: str = ""
    seq: int = 0                       # sqlite rowid: insertion order

    @property
    def effective_due(self) -> float:
        if self.state == "snoozed" and self.snooze_until:
            return float(self.snooze_until)
        return float(self.due)

    @property
    def duration(self) -> float:
        """Timers: the length they were set for."""
        return max(0.0, float(self.due) - float(self.created))

    def spoken_label(self) -> str:
        """What we call it in a sentence."""
        if self.kind == "timer":
            return self.label or f"your {duration_words(self.duration)} timer"
        if self.kind == "alarm":
            return self.label or "your alarm"
        return self.label


_SCHEMA = """
CREATE TABLE IF NOT EXISTS items(
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    due REAL NOT NULL,
    created REAL NOT NULL,
    repeat TEXT DEFAULT '',
    state TEXT NOT NULL DEFAULT 'pending',
    snooze_until REAL,
    fired_at REAL,
    note TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS items_state_due ON items(state, due);
"""
_COLS = ("id", "kind", "label", "due", "created", "repeat", "state",
         "snooze_until", "fired_at", "note")


def normalize_kind(kind) -> str:
    k = str(kind or "all").strip().lower().rstrip("s")
    if k in ("", "al", "everything", "any", "every", "schedule"):
        return "all"
    if k in KINDS:
        return k
    if k in ("wake", "wakeup", "wake-up", "wake up"):
        return "alarm"
    return "all"


def write_alarm_wav(path, seconds: float = 2.0, rate: int = 22050) -> Path:
    """Generated fallback sound: alternating 880 / 660 Hz beeps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    beep, gap = 0.22, 0.08
    t = 0.0
    tone = 0
    while t < seconds:
        freq = 880.0 if tone % 2 == 0 else 660.0
        n_beep = int(rate * beep)
        for i in range(n_beep):
            env = min(1.0, i / (rate * 0.01), (n_beep - i) / (rate * 0.01))
            frames += struct.pack("<h", int(12000 * env * math.sin(2 * math.pi * freq * i / rate)))
        frames += b"\x00\x00" * int(rate * gap)
        t += beep + gap
        tone += 1
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))
    return path


class _Ring:
    __slots__ = ("item_id", "started", "next_play", "proc", "escalated", "plays")

    def __init__(self, item_id: str, started: float):
        self.item_id = item_id
        self.started = started
        self.next_play = started
        self.proc = None
        self.escalated = False
        self.plays = 0


# ------------------------------------------------------------ timekeeper
class Timekeeper:
    """See the module docstring.  Every public method is thread-safe."""

    def __init__(self, db_path, say=None, cfg=None, now=time.time, run=_run,
                 tick_s: float = 1.0, ring: bool = True, cache_dir=None,
                 notify: bool = True):
        self.db_path = Path(db_path)
        self._say = say or (lambda text: None)
        self.cfg = cfg
        self._now = now
        self._run = run
        self.tick_s = float(tick_s)
        self.ring_enabled = bool(ring)
        self.notify_enabled = bool(notify)
        self._cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ring: Optional[_Ring] = None
        self._sound: Optional[str] = None
        self._db: Optional[sqlite3.Connection] = None
        self.quarantined: Optional[Path] = None   # set if the store was corrupt
        self.recovered = 0
        self._open()

    # ---------------------------------------------------------- storage
    def _open(self):
        """Open the store, quarantining it first if the file is corrupt.

        A damaged SQLite file used to raise straight out of ``__init__``,
        which left the app with no timekeeper at all -- for good, since
        every later boot re-read the same bad bytes.  Now the bad file is
        salvaged as far as it can be, moved aside under a timestamped
        name (never deleted), a fresh store is created, and the user is
        told in persona what happened and where the old file went."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connect()
        except sqlite3.DatabaseError as exc:
            log.warning("timekeeper store %s is corrupt (%s); quarantining",
                        self.db_path, exc)
            self._quarantine(exc)

    def _connect(self):
        """Connect and prove the file is a healthy database (or raise)."""
        db = sqlite3.connect(str(self.db_path), check_same_thread=False,
                             timeout=5.0)
        try:
            db.row_factory = sqlite3.Row
            row = db.execute("PRAGMA quick_check(1)").fetchone()
            if row is not None and str(row[0]).lower() != "ok":
                raise sqlite3.DatabaseError(f"integrity check: {row[0]}")
            db.executescript(_SCHEMA)
            db.commit()
        except Exception:                       # noqa: BLE001
            try:
                db.close()
            except Exception:                   # noqa: BLE001
                pass
            raise
        self._db = db

    def _salvage(self) -> list[tuple]:
        """Best effort: whatever rows the corrupt file still yields.

        Two passes.  A file that is merely unhealthy still answers a plain
        scan, so try that first, row by row (a torn page then costs only
        the rows after it).  A *truncated* file answers nothing at all --
        SQLite sees a page count larger than the file and calls every read
        malformed -- so pad a throwaway copy back up to the header's page
        count with zeroes and read it a rowid at a time: the pages that
        survived truncation come back, the missing ones just raise."""
        rows = self._scan(self.db_path, by_rowid=False)
        if rows:
            return rows
        pad = self._padded_copy()
        if pad is None:
            return rows
        try:
            return self._scan(pad, by_rowid=True)
        finally:
            try:
                pad.unlink()
            except OSError:
                log.warning("timekeeper: could not remove scratch copy %s", pad)

    def _padded_copy(self) -> Optional[Path]:
        """A copy of a truncated db zero-padded to its header's page count,
        or None if the file isn't a SQLite database at all."""
        try:
            raw = self.db_path.read_bytes()
            if not raw.startswith(b"SQLite format 3\x00") or len(raw) < 100:
                return None
            page = struct.unpack(">H", raw[16:18])[0] or 65536
            pages = struct.unpack(">I", raw[28:32])[0]
            want = page * pages
            if not 0 < want <= MAX_SALVAGE_BYTES or want <= len(raw):
                return None
            pad = self.db_path.with_name(self.db_path.name + ".salvage")
            pad.write_bytes(raw + b"\x00" * (want - len(raw)))
            return pad
        except (OSError, struct.error, ValueError):
            log.warning("timekeeper: could not prepare %s for salvage", self.db_path)
            return None

    def _scan(self, path: Path, by_rowid: bool) -> list[tuple]:
        rows: list[tuple] = []
        cols = ",".join(_COLS)
        try:
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                                 check_same_thread=False, timeout=5.0)
        except sqlite3.Error:
            return rows
        try:
            if not by_rowid:
                cur = db.execute(f"SELECT {cols} FROM items")
                while True:
                    try:
                        r = cur.fetchone()
                    except sqlite3.DatabaseError as exc:
                        log.warning("timekeeper: salvage stopped after %d row(s): %s",
                                    len(rows), exc)
                        break
                    if r is None:
                        break
                    rows.append(tuple(r))
                return rows
            top = MAX_SALVAGE_ROWS
            try:
                top = min(top, int(db.execute("SELECT max(rowid) FROM items").fetchone()[0]))
            except (sqlite3.DatabaseError, TypeError, ValueError):
                pass                            # unknown: scan up to the cap
            misses = 0
            for rid in range(1, top + 1):
                try:
                    r = db.execute(f"SELECT {cols} FROM items WHERE rowid=?",
                                   (rid,)).fetchone()
                except sqlite3.DatabaseError:
                    r = None
                if r is None:
                    misses += 1
                    if misses >= SALVAGE_GIVE_UP:
                        break
                else:
                    misses = 0
                    rows.append(tuple(r))
        except sqlite3.Error as exc:
            log.warning("timekeeper: nothing salvageable from %s (%s)", path, exc)
        finally:
            try:
                db.close()
            except Exception:                   # noqa: BLE001
                pass
        return rows

    def _quarantine_path(self) -> Path:
        stamp = datetime.fromtimestamp(float(self._now())).strftime("%Y%m%d-%H%M%S")
        base = self.db_path.with_name(f"{self.db_path.name}.bad-{stamp}")
        bad, n = base, 2
        while bad.exists():
            bad = base.with_name(f"{base.name}.{n}")
            n += 1
        return bad

    def _quarantine(self, exc: Exception):
        """Move the corrupt file aside, recreate, restore what we can, and
        tell the user.  Never raises: a broken file must not cost the app
        its timekeeper."""
        salvaged = self._salvage()
        bad = self._quarantine_path()
        try:
            os.replace(self.db_path, bad)
        except OSError:
            log.exception("timekeeper: could not move %s aside", self.db_path)
            raise sqlite3.DatabaseError(str(exc)) from exc
        for suffix in ("-wal", "-shm"):         # stale side files poison the new db
            side = self.db_path.with_name(self.db_path.name + suffix)
            if side.exists():
                try:
                    os.replace(side, bad.with_name(bad.name + suffix))
                except OSError:
                    log.exception("timekeeper: could not move %s aside", side)
        self._connect()                         # fresh, empty store
        restored = 0
        for row in salvaged:
            try:
                self._db.execute(
                    f"INSERT OR IGNORE INTO items({','.join(_COLS)}) "
                    f"VALUES({','.join('?' * len(_COLS))})", row)
                restored += 1
            except (sqlite3.Error, TypeError, ValueError):
                log.warning("timekeeper: unsalvageable row skipped")
        if restored:
            self._db.commit()
        self.quarantined = bad
        self.recovered = restored
        log.warning("timekeeper: corrupt store moved to %s; %d item(s) recovered",
                    bad, restored)
        line = (CORRUPT_RECOVERED_LINE.format(n=count_words(restored), name=bad.name)
                if restored else CORRUPT_STORE_LINE.format(name=bad.name))
        self._speak(line)

    def close(self):
        self.stop()
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:               # noqa: BLE001
                    log.exception("timekeeper db close failed")
                self._db = None

    def _row(self, r) -> Item:
        item = Item(**{c: r[c] for c in _COLS})
        item.seq = int(r["rowid"])
        return item

    def _get(self, item_id: str) -> Optional[Item]:
        r = self._db.execute("SELECT rowid, * FROM items WHERE id=?", (item_id,)).fetchone()
        return self._row(r) if r else None

    def _insert(self, item: Item) -> Item:
        with self._lock:
            self._db.execute(
                f"INSERT INTO items({','.join(_COLS)}) VALUES({','.join('?' * len(_COLS))})",
                tuple(getattr(item, c) for c in _COLS))
            self._db.commit()
        log.info("timekeeper: %s %r due %s (%s)", item.kind, item.label,
                 datetime.fromtimestamp(item.due).strftime("%Y-%m-%d %H:%M"),
                 item.repeat or "once")
        return item

    def _update(self, item_id: str, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE items SET {cols} WHERE id=?",
                             tuple(fields.values()) + (item_id,))
            self._db.commit()

    def _select(self, where: str = "1", params=()) -> list[Item]:
        rows = self._db.execute(f"SELECT rowid, * FROM items WHERE {where}", params).fetchall()
        return [self._row(r) for r in rows]

    # ---------------------------------------------------------- adding
    def _new(self, kind: str, label: str, due: float, repeat: str = "",
             note: str = "", created: Optional[float] = None) -> Item:
        item = Item(id=uuid.uuid4().hex[:12], kind=kind, label=(label or "").strip(),
                    due=float(due), created=float(self._now() if created is None else created),
                    repeat=repeat, state="pending", note=note or "")
        return self._insert(item)

    def add_reminder(self, due: float, text: str, repeat: str = "") -> Item:
        return self._new("reminder", text, due, repeat=normalize_repeat(repeat))

    def add_timer(self, seconds: float, label: str = "") -> Item:
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("timer length must be positive")
        now = self._now()
        return self._new("timer", label, now + seconds, created=now)

    def add_alarm(self, due: float, label: str = "", repeat: str = "once") -> Item:
        return self._new("alarm", label, due, repeat=normalize_repeat(repeat))

    # --------------------------------------------------------- reading
    def list(self, kind: str = "all", include_done: bool = False) -> list[Item]:
        kind = normalize_kind(kind)
        where, params = [], []
        if kind != "all":
            where.append("kind=?")
            params.append(kind)
        if not include_done:
            where.append(f"state IN ({','.join('?' * len(ACTIVE_STATES))})")
            params.extend(ACTIVE_STATES)
        with self._lock:
            items = self._select(" AND ".join(where) or "1", tuple(params))
        items.sort(key=lambda i: (i.effective_due, i.created, i.seq))
        return items

    @property
    def ringing(self) -> Optional[Item]:
        with self._lock:
            items = self._select("state='ringing'")
        return items[0] if items else None

    def _describe_item(self, it: Item, now: float) -> str:
        if it.state == "ringing":
            return f"{it.spoken_label()} ringing now"
        due = describe_due(it.effective_due, now)
        if it.kind == "alarm":
            head = it.label if it.label else "an alarm"
            tail = {"daily": ", every day", "weekdays": ", weekdays"}.get(it.repeat, "")
            if it.state == "snoozed":
                return f"{head} snoozed until {_time_words(_to_dt(it.effective_due))}{tail}"
            return f"{head} {due}{tail}"
        if it.kind == "timer":
            return f"{it.label or 'the ' + duration_words(it.duration) + ' timer'} {due}"
        return f"{it.label} {due}"

    def list_text(self, kind: str = "all", now=None) -> str:
        kind = normalize_kind(kind)
        now = self._now() if now is None else (now.timestamp() if isinstance(now, datetime) else float(now))
        items = self.list(kind)
        if not items:
            return {"reminder": "No reminders set, sir.",
                    "timer": "No timers running, sir.",
                    "alarm": "No alarms set, sir."}.get(kind, "Nothing scheduled, sir.")
        sentences = []
        for k in KINDS:
            group = [i for i in items if i.kind == k]
            if not group:
                continue
            noun = k + ("" if len(group) == 1 else "s")
            head = f"{count_words(len(group)).capitalize()} {noun}"
            head += ", sir: " if not sentences else ": "
            sentences.append(head + join_and(self._describe_item(i, now) for i in group) + ".")
        return " ".join(sentences)

    # -------------------------------------------------------- cancelling
    def cancel(self, which="last", kind: str = "all") -> int:
        which = str(which if which is not None else "last").strip()
        w = which.lower()
        items = self.list(kind)
        if not items:
            return 0
        if w in ("", "last", "latest", "the last one", "last one", "most recent",
                 "the latest", "that", "it", "that one", "this one"):
            targets = [max(items, key=lambda i: (i.created, i.seq))]
        elif w in ("all", "everything", "all of them", "every", "every one", "them all"):
            targets = items
        elif w in ("next", "the next one", "first", "soonest", "the first one"):
            targets = [items[0]]
        elif any(i.id == which for i in items):
            targets = [i for i in items if i.id == which]
        else:
            needle = " ".join(t for t in re.split(r"\s+", w)
                              if t not in ("the", "my", "a", "an", "one", "reminder",
                                           "reminders", "timer", "timers", "alarm",
                                           "alarms", "for", "to", "about"))
            targets = [i for i in items if needle and needle in i.label.lower()]
            if not targets and needle:
                words = set(needle.split())
                targets = [i for i in items if words & set(i.label.lower().split())]
        effects = []
        with self._lock:
            for it in targets:
                if self._ring is not None and self._ring.item_id == it.id:
                    self._kill_ring()
                    effects.append(lambda it=it: bus.publish(
                        AlarmStopped(alarm_id=it.id, action="dismiss", snooze_min=0)))
                self._update(it.id, state="cancelled")
                log.info("timekeeper: cancelled %s %r", it.kind, it.label)
        for fx in effects:
            fx()
        return len(targets)

    # ------------------------------------------------------ ring control
    def _kill_ring(self):
        r = self._ring
        self._ring = None
        if r is not None and r.proc is not None:
            for name in ("terminate", "kill"):
                fn = getattr(r.proc, name, None)
                if fn is None:
                    continue
                try:
                    fn()
                    break
                except Exception:               # noqa: BLE001
                    continue

    def _finish_alarm(self, item: Item, now: float, state: str):
        """Dismiss/timeout: repeat alarms come back tomorrow, others end."""
        if item.repeat in ("daily", "weekdays"):
            nxt = next_repeat(item.due, item.repeat, now)
            self._update(item.id, state="pending", due=nxt, snooze_until=None, fired_at=None)
            log.info("timekeeper: alarm %r rescheduled for %s", item.label,
                     datetime.fromtimestamp(nxt).strftime("%a %H:%M"))
        else:
            self._update(item.id, state=state)

    def stop_ringing(self, action: str = "dismiss") -> bool:
        with self._lock:
            item = self.ringing
            if item is None:
                return False
            now = self._now()
            self._kill_ring()
            self._finish_alarm(item, now, "done")
        log.info("timekeeper: alarm %r %s", item.label, action)
        bus.publish(AlarmStopped(alarm_id=item.id, action=action or "dismiss", snooze_min=0))
        return True

    def snooze(self, minutes=None) -> bool:
        mins = _coerce_int(minutes) or _coerce_int(
            _cfg_get(self.cfg, "alarms.snooze_min", DEFAULT_SNOOZE_MIN)) or DEFAULT_SNOOZE_MIN
        with self._lock:
            item = self.ringing
            if item is None:
                return False
            now = self._now()
            self._kill_ring()
            self._update(item.id, state="snoozed", snooze_until=now + mins * 60)
        log.info("timekeeper: alarm %r snoozed %d min", item.label, mins)
        bus.publish(AlarmStopped(alarm_id=item.id, action="snooze", snooze_min=int(mins)))
        return True

    # ---------------------------------------------------------- firing
    def _speak(self, line: str):
        try:
            self._say(line)
        except Exception:                       # noqa: BLE001
            log.exception("timekeeper: say failed")

    def _toast(self, title: str, text: str, urgency: str = "normal"):
        if not self.notify_enabled:
            return
        icon = "alarm-clock" if urgency == "critical" else "appointment-soon"
        self._run(["notify-send", "-a", "Jarvis", "-u", urgency, "-t",
                   "0" if urgency == "critical" else "10000", "-i", icon,
                   title, text], background=True)

    def _fire(self, item: Item, now: float, late: bool, effects: list) -> bool:
        """Fire one due item.  Called under the lock; speech / events are
        appended to ``effects`` and run after the lock is released."""
        due_dt = _to_dt(item.due)
        now_dt = _to_dt(now)
        if item.kind == "alarm":
            if self._ring is not None or self._select("state='ringing'"):
                return False                  # one alarm at a time; stays pending
            label = sentence_case(item.label or ALARM_DEFAULT_LABEL)
            if late:
                line = LATE_LINE.format(label=item.spoken_label(), time=_when_words(due_dt, now_dt))
            else:
                line = ALARM_LINE.format(time=fmt_clock(now_dt), label=label)
            self._update(item.id, state="ringing", fired_at=now, snooze_until=None)
            if self.ring_enabled:
                self._ring = _Ring(item.id, now)
                self._service_ring(now, effects)      # first paplay right away
            effects.append(lambda: self._speak(line))
            effects.append(lambda: self._toast("Jarvis alarm", item.label or "Alarm", "critical"))
            effects.append(lambda: bus.publish(AlarmFired(
                alarm_id=item.id, label=item.label, kind="alarm",
                due_text=fmt_clock(due_dt))))
            log.info("timekeeper: alarm %r ringing", item.label)
            return True
        if item.kind == "timer":
            n = duration_words(item.duration)
            if late:
                line = LATE_LINE.format(label=item.spoken_label(), time=_when_words(due_dt, now_dt))
            elif item.label:
                line = TIMER_LABEL_LINE.format(n=n, label=item.label)
            else:
                line = TIMER_LINE.format(n=n)
            text = f"{item.label or n + ' timer'}"
            title = "Jarvis timer"
        else:
            if late:
                line = LATE_LINE.format(label=item.label, time=_when_words(due_dt, now_dt))
            else:
                line = REMINDER_LINE.format(text=sentence_case(item.label))
            text = item.label
            title = "Jarvis reminder"
        if item.repeat in ("daily", "weekdays"):
            nxt = next_repeat(item.due, item.repeat, now)
            self._update(item.id, state="pending", due=nxt, fired_at=now,
                         snooze_until=None)
            log.info("timekeeper: %s %r rescheduled for %s", item.kind, item.label,
                     datetime.fromtimestamp(nxt).strftime("%a %H:%M"))
        else:
            self._update(item.id, state="done", fired_at=now)
        effects.append(lambda: self._speak(line))
        effects.append(lambda: self._toast(title, text, "normal"))
        effects.append(lambda: bus.publish(ReminderFired(text=text)))
        log.info("timekeeper: %s fired %r", item.kind, item.label)
        return True

    def _due_items(self, now: float) -> list[Item]:
        items = self._select(
            "(state='pending' AND due<=?) OR (state='snoozed' AND snooze_until<=?)",
            (now, now))
        items.sort(key=lambda i: (i.effective_due, i.created))
        return items

    def _ring_timeout(self, item: Item, now: float, effects: list):
        self._kill_ring()
        self._finish_alarm(item, now, "missed")
        line = RING_TIMEOUT_LINE.format(label=item.spoken_label(),
                                        time=_when_words(_to_dt(item.due), _to_dt(now)))
        effects.append(lambda: self._speak(line))
        effects.append(lambda: bus.publish(AlarmStopped(alarm_id=item.id, action="timeout", snooze_min=0)))
        log.info("timekeeper: alarm %r rang out (missed)", item.label)

    def _service_ring(self, now: float, effects: list):
        r = self._ring
        if r is None:
            return
        item = self._get(r.item_id)
        if item is None or item.state != "ringing":
            self._kill_ring()
            return
        elapsed = now - r.started
        max_ring = float(_cfg_get(self.cfg, "alarms.max_ring_s", DEFAULT_MAX_RING_S) or DEFAULT_MAX_RING_S)
        if elapsed >= max_ring:
            self._ring_timeout(item, now, effects)
            return
        escalate = bool(_cfg_get(self.cfg, "alarms.escalate", True))
        if escalate and not r.escalated and elapsed >= ESCALATE_AFTER_S:
            r.escalated = True
            log.info("timekeeper: alarm %r escalating to full volume", item.label)
        gap = ESCALATED_GAP_S if r.escalated else RING_GAP_S
        if r.proc is not None:
            poll = getattr(r.proc, "poll", None)
            if callable(poll) and poll() is None:
                return                        # still playing
            r.proc = None
            r.next_play = now + gap
            return
        if now < r.next_play:
            return
        volume = MAX_VOLUME if r.escalated else self._volume()
        r.proc = self._run(["paplay", f"--volume={volume}", self._sound_path()], background=True)
        r.plays += 1
        if r.proc is None:
            r.next_play = now + gap

    def _volume(self) -> int:
        try:
            v = float(_cfg_get(self.cfg, "alarms.volume", DEFAULT_VOLUME))
        except (TypeError, ValueError):
            v = DEFAULT_VOLUME
        return int(max(0.0, min(1.0, v)) * MAX_VOLUME)

    def _sound_path(self) -> str:
        if self._sound and Path(self._sound).is_file():
            return self._sound
        for cand in (_cfg_get(self.cfg, "alarms.sound", ""), DEFAULT_SOUND):
            if cand and Path(os.path.expanduser(str(cand))).is_file():
                self._sound = str(Path(os.path.expanduser(str(cand))))
                return self._sound
        path = self._cache_dir / "alarm.wav"
        if not path.is_file():
            try:
                write_alarm_wav(path)
            except Exception:                   # noqa: BLE001
                log.exception("timekeeper: could not write fallback alarm sound")
        self._sound = str(path)
        return self._sound

    # ------------------------------------------------------- scheduling
    def tick(self, now=None) -> int:
        """One scheduler pass: service the ringer, fire what is due.
        Returns the number of items fired."""
        now = float(self._now() if now is None else now)
        effects: list = []
        fired = 0
        with self._lock:
            self._service_ring(now, effects)
            for it in self._due_items(now):
                if self._fire(it, now, False, effects):
                    fired += 1
        for fx in effects:
            try:
                fx()
            except Exception:                   # noqa: BLE001
                log.exception("timekeeper: effect failed")
        return fired

    def catch_up(self, now=None) -> list[Item]:
        """Boot: items that came due while down.  < 1 h late -> fire with
        the 'while I was down' preamble; else -> missed (repeat alarms are
        rescheduled), announced once in a single sentence."""
        now = float(self._now() if now is None else now)
        effects: list = []
        handled: list[Item] = []
        missed: list[Item] = []
        with self._lock:
            stale = self._select("state='ringing'")
            for it in stale:                  # crashed mid-ring: treat as pending
                self._update(it.id, state="pending")
            for it in self._due_items(now):
                late = now - it.effective_due
                if late < LATE_GRACE_S:
                    if self._fire(it, now, True, effects):
                        handled.append(it)
                else:
                    self._finish_alarm(it, now, "missed")
                    missed.append(it)
                    log.info("timekeeper: %s %r missed (%.0f s late)", it.kind, it.label, late)
        if missed:
            parts = [f"{m.spoken_label()} at {_when_words(_to_dt(m.effective_due), _to_dt(now))}"
                     for m in missed]
            effects.append(lambda: self._speak(MISSED_LINE.format(what=join_and(parts))))
        for fx in effects:
            try:
                fx()
            except Exception:                   # noqa: BLE001
                log.exception("timekeeper: effect failed")
        with self._lock:                        # snapshots must not be stale
            return [self._get(it.id) or it for it in handled + missed]

    def start(self, catch_up: bool = True):
        if self._thread is not None and self._thread.is_alive():
            return
        if catch_up:
            try:
                self.catch_up()
            except Exception:                   # noqa: BLE001
                log.exception("timekeeper: catch-up failed")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="timekeeper", daemon=True)
        self._thread.start()
        log.info("timekeeper: scheduler started (%d active)", len(self.list()))

    def stop(self):
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._kill_ring()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:                   # noqa: BLE001
                log.exception("timekeeper: tick failed")
            wait = self.tick_s
            r = self._ring
            if r is not None:
                wait = min(wait, ESCALATED_GAP_S if r.escalated else RING_GAP_S)
            self._stop.wait(wait)

    # ---------------------------------------------------------- legacy
    def import_legacy(self, path=None) -> int:
        """workflows.Reminders' reminders.json -> reminders, once (the file
        is renamed *.migrated).  Overdue ones are left for catch_up()."""
        path = Path(path) if path else _default_legacy_path()
        if not path.is_file():
            return 0
        try:
            data = json.loads(path.read_text())
            entries = data.get("reminders", []) if isinstance(data, dict) else data
        except Exception:                       # noqa: BLE001
            log.exception("timekeeper: legacy reminders unreadable: %s", path)
            return 0
        count = 0
        with self._lock:
            known = {i.note for i in self._select("note LIKE 'legacy:%'")}
            for r in entries:
                try:
                    note = f"legacy:{r.get('id', '')}"
                    if r.get("id") and note in known:
                        continue
                    self._new("reminder", str(r.get("task", "")).strip(), float(r["due"]), note=note)
                    count += 1
                except Exception:               # noqa: BLE001
                    log.warning("timekeeper: skipped legacy reminder %r", r)
        try:
            path.rename(path.with_name(path.name + ".migrated"))
        except OSError:
            log.exception("timekeeper: could not rename %s", path)
        log.info("timekeeper: imported %d legacy reminder(s)", count)
        return count


def next_repeat(due: float, repeat: str, now: float) -> float:
    """The next wall-clock occurrence after ``now`` for a daily / weekdays
    alarm originally due at ``due`` (keeps the hour across DST)."""
    d = _to_dt(due)
    n = _to_dt(now)
    while True:
        d = d + timedelta(days=1)
        if repeat == "weekdays" and d.weekday() >= 5:
            continue
        if d > n:
            return d.timestamp()


def _coerce_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        pass
    sc = _Scan(str(value))
    for t in sc.norm:
        if _num(t) is not None:
            return int(round(_num(t)))
    return None


# ----------------------------------------------------------------- tools
def _set_line(noun: str, desc: str, suffix: str = "") -> str:
    """'Reminder set for 3:00 pm, sir.' / 'Reminder set for 10 minutes
    from now, sir.'"""
    if desc.startswith("in "):
        return f"{noun} set for {desc[3:]} from now{suffix}, sir."
    if desc.startswith("at "):
        return f"{noun} set for {desc[3:]}{suffix}, sir."
    return f"{noun} set {desc}{suffix}, sir."


def make_tools(cfg, services) -> list[ToolSpec]:
    """ToolSpecs for the local tool loop (spec 4.1)."""

    def tk() -> Optional[Timekeeper]:
        return getattr(services, "timekeeper", None)

    def unavailable() -> ToolResult:
        return ToolResult(text="timekeeper not running", ok=False, speak=NO_TIMEKEEPER_LINE)

    def now_of(t: Timekeeper) -> tuple[float, datetime]:
        epoch = float(t._now())
        return epoch, datetime.fromtimestamp(epoch)

    def set_reminder(when="", text="", repeat="once", **_) -> ToolResult:
        t = tk()
        if t is None:
            return unavailable()
        epoch, now = now_of(t)
        when, text = str(when or "").strip(), str(text or "").strip()
        dt, rep_text, rest = parse_when_full(when, now)
        if dt is None and text:
            dt, rep_text, rest2 = parse_when_full(text, now)
            if dt is not None:
                text = rest2 or when
        if dt is None:
            return ToolResult(text=f"could not understand the time '{when}'", ok=False,
                              speak=CANT_PARSE_LINE)
        if not text:
            text = rest or "your reminder"
        rep = normalize_repeat(repeat) or rep_text or repeat_from_text(text)
        item = t.add_reminder(dt.timestamp(), text, rep)
        suffix = {"daily": ", every day", "weekdays": ", weekdays"}.get(item.repeat, "")
        line = _set_line("Reminder", describe_due(item.due, epoch), suffix)
        return ToolResult(text=line, speak=line)

    def set_timer(minutes=None, label="", **_) -> ToolResult:
        t = tk()
        if t is None:
            return unavailable()
        seconds = parse_duration(minutes)
        if seconds is None:
            return ToolResult(text="timer length missing or not a number", ok=False,
                              speak=HOW_LONG_LINE)
        label = str(label or "").strip()
        item = t.add_timer(seconds, label)
        n = duration_words(item.duration)
        line = f"{n} timer set for {label}, sir." if label else f"{n} timer set, sir."
        line = line[0].upper() + line[1:]
        return ToolResult(text=line, speak=line)

    def set_alarm(when="", label="", repeat="once", **_) -> ToolResult:
        t = tk()
        if t is None:
            return unavailable()
        epoch, now = now_of(t)
        when = str(when or "").strip()
        dt, rep_text, _rest = parse_when_full(when, now, prefer="morning")
        if dt is None:
            return ToolResult(text=f"could not understand the time '{when}'", ok=False,
                              speak=CANT_PARSE_LINE)
        rep = normalize_repeat(repeat) or rep_text
        item = t.add_alarm(dt.timestamp(), str(label or "").strip(), rep)
        suffix = {"daily": ", every day", "weekdays": ", weekdays"}.get(item.repeat, "")
        line = _set_line("Alarm", describe_due(item.due, epoch), suffix)
        return ToolResult(text=line, speak=line)

    def manage_schedule(action="list", kind="all", which="last", minutes=None, **_) -> ToolResult:
        t = tk()
        if t is None:
            return unavailable()
        a = str(action or "list").strip().lower()
        if a in ("list", "show", "what", "check", "status", "read", "tell", "get"):
            a = "list"
        elif a in ("cancel", "delete", "remove", "clear", "drop", "kill", "forget", "scrap"):
            a = "cancel"
        elif a in ("stop", "dismiss", "off", "silence", "quiet", "turn off", "shut up"):
            a = "stop"
        elif a in ("snooze", "later", "postpone"):
            a = "snooze"
        kind = normalize_kind(kind)
        if a == "list":
            text = t.list_text(kind)
            return ToolResult(text=text, speak=text, max_sentences=3)
        if a == "cancel":
            n = t.cancel(which if which not in (None, "") else "last", kind)
            if n == 0:
                line = "Nothing to cancel, sir."
            elif n == 1:
                line = "Cancelled, sir."
            else:
                noun = "items" if kind == "all" else kind + "s"
                line = f"Cancelled {count_words(n)} {noun}, sir."
            return ToolResult(text=line, ok=n > 0, speak=line)
        if a == "stop":
            ok = t.stop_ringing("dismiss")
            line = "Very good, sir." if ok else NOTHING_RINGING_LINE
            return ToolResult(text=line, ok=ok, speak=line)
        if a == "snooze":
            mins = _coerce_int(minutes) or _coerce_int(
                _cfg_get(cfg, "alarms.snooze_min", DEFAULT_SNOOZE_MIN)) or DEFAULT_SNOOZE_MIN
            ok = t.snooze(mins)
            line = f"Snoozed for {count_words(mins)} minutes, sir." if ok else NOTHING_RINGING_LINE
            return ToolResult(text=line, ok=ok, speak=line)
        return ToolResult(text=f"unknown action '{action}'", ok=False)

    return [
        ToolSpec(
            name="set_reminder",
            description="Set a spoken reminder at a natural-language time.",
            parameters={"type": "object",
                        "properties": {
                            "when": {"type": "string",
                                     "description": "e.g. 'in 10 minutes', 'at 3 pm', 'tomorrow morning'"},
                            "text": {"type": "string", "description": "what to remind"},
                            "repeat": {"type": "string",
                                       "enum": ["once", "daily", "weekdays"]}},
                        "required": ["when", "text"]},
            handler=set_reminder),
        ToolSpec(
            name="set_timer",
            description="Start a countdown timer in minutes.",
            parameters={"type": "object",
                        "properties": {
                            "minutes": {"type": "number"},
                            "label": {"type": "string"}},
                        "required": ["minutes"]},
            handler=set_timer),
        ToolSpec(
            name="set_alarm",
            description="Set a wake-up alarm that rings until dismissed.",
            parameters={"type": "object",
                        "properties": {
                            "when": {"type": "string",
                                     "description": "e.g. 'at 7', '6:30 am', 'tomorrow at 8'"},
                            "label": {"type": "string"},
                            "repeat": {"type": "string", "enum": ["once", "daily", "weekdays"]}},
                        "required": ["when"]},
            handler=set_alarm),
        ToolSpec(
            name="manage_schedule",
            description="List or cancel reminders, timers and alarms; stop or snooze a ringing alarm.",
            parameters={"type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["list", "cancel", "stop", "snooze"]},
                            "kind": {"type": "string", "enum": ["reminder", "timer", "alarm", "all"]},
                            "which": {"type": "string",
                                      "description": "'last', 'all' or words from the item"},
                            "minutes": {"type": "number"}},
                        "required": ["action"]},
            handler=manage_schedule),
    ]
