"""Quiet hours, do-not-disturb and the spoken catch-up digest.

One question, asked by the app's single speech door (``JarvisApp._say``)
whenever a line is PROACTIVE -- something Jarvis decided to say on his own
(a memory warning, a reminder, a meeting heads-up, a line from the hooks
narrator) rather than an answer to something Hunter said::

    policy.should_hold()      # True -> park the line, say it later

Four things make him hold his tongue:

* ``quiet.dnd_until``     -- "do not disturb for an hour" (a timestamp)
* ``quiet.hours``         -- "quiet hours from eleven to seven" (a daily
                             window; overnight windows wrap midnight)
* the calendar            -- a timed event running NOW whose title contains
                             one of ``quiet.calendar_keywords``
                             (class / exam / meeting / busy)
* being away              -- ``services.presence.is_home()`` says the phone
                             left (jarvis/presence.py); off with
                             ``quiet.hold_when_away``

"I am free" ends the current window early (``quiet.free_until`` = the end of
whatever was blocking) and reads the digest. Otherwise the policy's own
thread notices the window closing and reads it back: "While you were busy,
sir: two reminders and a warning. <the lines>." Alarms are never held --
the timekeeper marks its alarm line non-proactive -- and answers to direct
questions never come through here at all: the gate is keyed on the caller's
``proactive=True`` flag, not on ``_say`` itself.

Held lines are capped (``HOLD_MAX``): a night of watchdog warnings must not
become a ten-minute monologue at seven in the morning.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("quiet")

HOLD_MAX = 12
TICK_S = 30.0
DEFAULT_KEYWORDS = ("class", "exam", "meeting", "busy")

# Persona lines (the app prewarms the fixed ones).
BUSY_PREFIX = "While you were busy, sir"
AWAY_PREFIX = "While you were out, sir"
NOTHING_HELD_LINE = "Nothing was held back, sir."
DND_SET_LINE = "Very good, sir; I'll hold my tongue until {until}."
DND_ALREADY_FREE_LINE = "You weren't in a quiet period, sir."
QUIET_HOURS_SET_LINE = "Quiet hours are now {start} to {end}, sir."
QUIET_HOURS_OFF_LINE = "Quiet hours are off, sir."
QUIET_STATUS_QUIET_LINE = "I'm holding my tongue, sir: {reason}."
QUIET_STATUS_FREE_LINE = "I'm not holding anything back, sir."
FREE_LINE = "Very good, sir."

_KIND_NOUNS = {
    "reminder": ("reminder", "reminders"),
    "timer": ("timer", "timers"),
    "warning": ("warning", "warnings"),
    "message": ("message", "messages"),
    "alarm": ("alarm", "alarms"),
}
_COUNT_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"]


def _count(n: int) -> str:
    return _COUNT_WORDS[n] if 0 <= n < len(_COUNT_WORDS) else str(n)


def _join_and(parts) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def fmt_clock(hour: int, minute: int) -> str:
    h = hour % 12 or 12
    ampm = "am" if hour < 12 else "pm"
    return f"{h}:{minute:02d} {ampm}" if minute else f"{h} {ampm}"


# ------------------------------------------------------------ clock parse
_HOUR_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_MINUTE_WORDS = {
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "quarter": 15, "half": 30,
}
_CLOCK_RX = re.compile(
    r"^(?:(?P<half>half|quarter)\s+past\s+)?"
    r"(?P<h>\d{1,2}|" + "|".join(_HOUR_WORDS) + r")"
    r"(?:[:.](?P<m>\d{2})|\s+(?P<mw>" + "|".join(_MINUTE_WORDS) +
    r")(?:[\s-](?P<mw2>one|two|three|four|five|six|seven|eight|nine))?)?"
    r"\s*(?P<ampm>a\.?m\.?|p\.?m\.?|in the morning|in the evening|at night|"
    r"in the afternoon|tonight)?\s*$", re.I)


def parse_clock(text: str, default: str = "") -> Optional[tuple]:
    """'eleven' / '11 pm' / 'seven thirty' / '7:30 am' / 'midnight' -> (h, m).

    ``default`` decides a bare 1-11 with no am/pm: 'pm' or 'am' (quiet
    hours run overnight, so a start defaults to pm and an end to am).
    None when the text is not a clock time."""
    s = re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".!?")
    if not s:
        return None
    if s in ("midnight", "twelve midnight", "12 midnight"):
        return (0, 0)
    if s in ("noon", "midday", "twelve noon", "12 noon"):
        return (12, 0)
    m = _CLOCK_RX.match(s)
    if not m:
        return None
    h_tok = m.group("h")
    hour = int(h_tok) if h_tok.isdigit() else _HOUR_WORDS[h_tok]
    minute = 0
    if m.group("m"):
        minute = int(m.group("m"))
    elif m.group("mw"):
        minute = _MINUTE_WORDS[m.group("mw")]
        if m.group("mw2"):
            minute += _HOUR_WORDS[m.group("mw2")]
    if m.group("half"):
        minute = _MINUTE_WORDS[m.group("half")]
    if hour > 23 or minute > 59:
        return None
    ampm = (m.group("ampm") or "").replace(".", "")
    if hour <= 12:
        if ampm.startswith("p") or ampm in ("in the evening", "at night",
                                            "in the afternoon", "tonight"):
            hour = hour % 12 + 12
        elif ampm.startswith("a") or ampm == "in the morning":
            hour = hour % 12
        elif h_tok.isdigit() and len(h_tok) == 2 and h_tok[0] == "0":
            pass                                  # "07:00" is 24 h
        elif default == "pm" and 1 <= hour <= 11:
            hour += 12
        elif default == "am" and hour == 12:
            hour = 0
        elif default == "pm" and hour == 12:
            hour = 0                              # "from twelve" = midnight
    return (hour, minute)


def _hhmm(value) -> Optional[tuple]:
    try:
        h, m = str(value or "").strip().split(":")
        h, m = int(h), int(m)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return (h, m)
    except ValueError:
        pass
    return None


# ------------------------------------------------------------------ policy
class QuietPolicy:
    """See the module docstring. Thread-safe; every state change goes
    through the assistant config (so a DND set at eleven survives a
    restart at midnight) with an in-memory fallback for a config object
    that cannot save."""

    def __init__(self, cfg, get_calendar: Optional[Callable] = None,
                 is_home: Optional[Callable[[], bool]] = None,
                 say: Optional[Callable[[str], None]] = None,
                 can_speak: Optional[Callable[[], bool]] = None,
                 now: Callable[[], float] = time.time,
                 hold_max: int = HOLD_MAX, tick_s: float = TICK_S):
        self._cfg = cfg
        self._get_calendar = get_calendar
        self._is_home = is_home
        self._say = say
        self._can_speak = can_speak
        self._now = now
        self.tick_s = float(tick_s)
        self._held: deque = deque(maxlen=max(1, int(hold_max)))
        self._overlay: dict = {}          # cfg without set(): values live here
        self._lock = threading.RLock()
        self._last_quiet: Optional[bool] = None
        self._last_reason = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ config
    def _get(self, key: str, default=None):
        if key in self._overlay:
            return self._overlay[key]
        get = getattr(self._cfg, "get", None)
        if callable(get):
            try:
                value = get(key, default)
                return default if value is None else value
            except Exception:  # noqa: BLE001 - a config hiccup must not mute him
                log.debug("quiet: cfg.get(%s) failed", key, exc_info=True)
        return default

    def _set(self, key: str, value) -> None:
        setter = getattr(self._cfg, "set", None)
        if callable(setter):
            self._overlay.pop(key, None)
            try:
                setter(key, value)
                return
            except Exception:  # noqa: BLE001
                log.exception("quiet: cfg.set(%s) failed; kept in memory", key)
        self._overlay[key] = value

    # ------------------------------------------------------------ state
    def now(self) -> float:
        """The policy's clock (epoch seconds); the commander computes
        "until seven" against THIS so a fake clock in tests lines up."""
        return float(self._now())

    def dnd_until(self) -> float:
        try:
            return float(self._get("quiet.dnd_until", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def quiet_hours(self) -> Optional[tuple]:
        """((h, m), (h, m)) or None when unset / malformed."""
        hours = self._get("quiet.hours", {}) or {}
        if not isinstance(hours, dict):
            return None
        start, end = _hhmm(hours.get("start")), _hhmm(hours.get("end"))
        if start is None or end is None or start == end:
            return None
        return start, end

    def keywords(self) -> tuple:
        kws = self._get("quiet.calendar_keywords", list(DEFAULT_KEYWORDS))
        if isinstance(kws, str):
            kws = [kws]
        return tuple(str(k).strip().lower() for k in (kws or []) if str(k).strip())

    # ---------------------------------------------------------- reasons
    def _hours_reason(self, dt: datetime) -> str:
        win = self.quiet_hours()
        if win is None:
            return ""
        (sh, sm), (eh, em) = win
        cur, start, end = dt.hour * 60 + dt.minute, sh * 60 + sm, eh * 60 + em
        inside = start <= cur < end if start < end else (cur >= start or cur < end)
        return f"quiet hours until {fmt_clock(eh, em)}" if inside else ""

    def _hours_end(self, dt: datetime) -> Optional[float]:
        win = self.quiet_hours()
        if win is None:
            return None
        (eh, em) = win[1]
        end = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
        if end <= dt:
            end += timedelta(days=1)
        return end.timestamp()

    def _calendar_event(self, dt: datetime):
        """The running timed event whose title matches, or None."""
        if not self._get("quiet.calendar", True) or self._get_calendar is None:
            return None
        try:
            cal = self._get_calendar() if callable(self._get_calendar) else self._get_calendar
            if cal is None:
                return None
            conf = getattr(cal, "configured", True)
            if callable(conf):
                conf = conf()
            if not conf:
                return None
            events = list(cal.events())
        except Exception:  # noqa: BLE001 - the cache is best-effort
            log.debug("quiet: calendar unavailable", exc_info=True)
            return None
        kws = self.keywords()
        if not kws:
            return None
        for ev in events:
            if getattr(ev, "all_day", False):
                continue
            start, end = getattr(ev, "start", None), getattr(ev, "end", None)
            title = (getattr(ev, "title", "") or "").strip()
            if start is None or end is None or not title:
                continue
            now = dt.astimezone(start.tzinfo) if start.tzinfo else dt.replace(tzinfo=None)
            try:
                if not (start <= now < end):
                    continue
            except TypeError:
                continue
            words = re.findall(r"[a-z]+", title.lower())
            if any(k in words for k in kws):
                return ev
        return None

    def reason(self, now: Optional[float] = None) -> str:
        """Why he is quiet right now, in his words -- "" when he is not."""
        ts = self._now() if now is None else float(now)
        try:
            free_until = float(self._get("quiet.free_until", 0) or 0)
        except (TypeError, ValueError):
            free_until = 0.0
        dt = datetime.fromtimestamp(ts).astimezone()
        if free_until > ts:
            # "I am free" overrides everything for the rest of the window,
            # the presence probe included: a man who just spoke to Jarvis
            # is in the room whatever his sleeping phone says.
            return ""
        until = self.dnd_until()
        if until > ts:
            return f"do not disturb until {fmt_clock(*_hm(until))}"
        if self._is_home is not None and self._get("quiet.hold_when_away", True):
            try:
                if not self._is_home():
                    return "you're out"
            except Exception:  # noqa: BLE001
                log.debug("quiet: is_home failed", exc_info=True)
        hours = self._hours_reason(dt)
        if hours:
            return hours
        ev = self._calendar_event(dt)
        if ev is not None:
            end = getattr(ev, "end", None)
            when = f" until {fmt_clock(end.hour, end.minute)}" if end is not None else ""
            return f"{ev.title}{when}"
        return ""

    def is_quiet(self, now: Optional[float] = None) -> bool:
        return bool(self.reason(now))

    def should_hold(self) -> bool:
        """The app's gate for proactive speech."""
        with self._lock:
            reason = self.reason()
            self._last_quiet = bool(reason)
            if reason:
                self._last_reason = reason
            return bool(reason)

    # --------------------------------------------------------- commands
    def set_dnd(self, seconds: float, now: Optional[float] = None) -> float:
        ts = self._now() if now is None else float(now)
        until = ts + max(60.0, float(seconds))
        self._set("quiet.dnd_until", until)
        self._set("quiet.free_until", 0)
        log.info("quiet: do not disturb until %s", fmt_clock(*_hm(until)))
        return until

    def set_hours(self, start: Optional[tuple], end: Optional[tuple]) -> None:
        if start is None or end is None:
            self._set("quiet.hours", {"start": "", "end": ""})
            log.info("quiet: quiet hours off")
            return
        self._set("quiet.hours", {"start": "%02d:%02d" % start, "end": "%02d:%02d" % end})
        log.info("quiet: quiet hours %s-%s", fmt_clock(*start), fmt_clock(*end))

    def free(self, now: Optional[float] = None) -> str:
        """"I am free": end the current window early and read the digest.
        Returns what to say."""
        ts = self._now() if now is None else float(now)
        with self._lock:
            was = self.reason(ts)
            self._set("quiet.dnd_until", 0)
            dt = datetime.fromtimestamp(ts).astimezone()
            # Override quiet hours / a calendar block for the rest of THIS
            # window only; tomorrow's quiet hours are untouched.
            ends = [self._hours_end(dt)] if self._hours_reason(dt) else []
            ev = self._calendar_event(dt)
            if ev is not None and getattr(ev, "end", None) is not None:
                ends.append(ev.end.timestamp())
            if ends:
                self._set("quiet.free_until", max(ends))
            digest = self.release(prefix=BUSY_PREFIX)
            if digest:
                return f"{FREE_LINE} {digest}"
            return FREE_LINE if was else DND_ALREADY_FREE_LINE

    # ------------------------------------------------------------- hold
    def hold(self, text: str, kind: str = "message") -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._held.append((self._now(), text, kind or "message"))
        log.info("quiet: held (%s): %.80s", kind, text)

    @property
    def held(self) -> list:
        with self._lock:
            return list(self._held)

    def release(self, prefix: Optional[str] = None) -> str:
        """Drain the held lines into one spoken digest ("" when empty)."""
        with self._lock:
            items = list(self._held)
            self._held.clear()
            reason = self._last_reason
        if not items:
            return ""
        if prefix is None:
            prefix = AWAY_PREFIX if reason == "you're out" else BUSY_PREFIX
        return digest(items, prefix)

    # ----------------------------------------------------------- thread
    def tick(self) -> str:
        """Notice a quiet window closing; speak the digest through ``say``.
        Returns what was said ("" when nothing)."""
        with self._lock:
            quiet = self.is_quiet()
            was = self._last_quiet
            self._last_quiet = quiet
            if quiet:
                self._last_reason = self.reason()
                return ""
            if was is False or not self._held:
                return ""
            if self._can_speak is not None:
                try:
                    ok = bool(self._can_speak())
                except Exception:  # noqa: BLE001 - a probe failure must not mute him
                    ok = True
                if not ok:
                    # Mid-capture or mid-turn: the digest talking over an
                    # open mic is the exact interruption quiet hours exist
                    # to prevent. Keep the backlog; retry next tick.
                    self._last_quiet = True
                    return ""
            text = self.release()
        if text and callable(self._say):
            try:
                self._say(text)
            except Exception:  # noqa: BLE001 - a TTS failure must not kill the loop
                log.exception("quiet: digest speak failed")
        return text

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="quiet-policy",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("quiet: tick failed")
            self._stop.wait(max(0.01, self.tick_s))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def _hm(ts: float) -> tuple:
    dt = datetime.fromtimestamp(ts).astimezone()
    return dt.hour, dt.minute


def digest(items, prefix: str = BUSY_PREFIX) -> str:
    """"While you were busy, sir: two reminders and a warning. <lines>"."""
    items = list(items)
    if not items:
        return ""
    counts: dict[str, int] = {}
    for _, _, kind in items:
        counts[kind] = counts.get(kind, 0) + 1
    parts = []
    for kind, n in counts.items():
        one, many = _KIND_NOUNS.get(kind, ("message", "messages"))
        parts.append(f"{_count(n)} {one if n == 1 else many}")
    lines = []
    for _, text, _ in items:
        text = text.strip()
        if text and text[-1] not in ".!?":
            text += "."
        lines.append(text)
    return f"{prefix}: {_join_and(parts)}. " + " ".join(lines)
