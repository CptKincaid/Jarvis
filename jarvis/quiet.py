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
                             (class / exam / meeting / busy), OR one that is
                             a recurring course whatever it is called
                             (``quiet.calendar_courses``, jarvis/courses.py)
* being away              -- ``services.presence.is_home()`` says the phone
                             left (jarvis/presence.py); off with
                             ``quiet.hold_when_away``
* a focus block           -- ``services.focus`` is mid-pomodoro
                             (jarvis/focus.py); off with ``focus.dnd``.
                             Only the BLOCK holds: the break is exactly
                             when the digest should be read, and the
                             session's own "time for a break" lines pierce
                             the hold by being non-proactive.

"I am free" ends the current window early (``quiet.free_until`` = the end of
whatever was blocking) and reads the digest. Otherwise the policy's own
thread notices the window closing and reads it back: "While you were busy,
sir: two reminders and a warning. <the lines>." Alarms are never held --
the timekeeper marks its alarm line non-proactive -- and answers to direct
questions never come through here at all: the gate is keyed on the caller's
``proactive=True`` flag, not on ``_say`` itself.

Held lines are capped (``HOLD_MAX``): a night of watchdog warnings must not
become a ten-minute monologue at seven in the morning. Some kinds are not
held at all: an interval nudge (``kind="nudge"``, jarvis/tools/timekeeper.py)
EXPIRES when the window is closed, because "stand up, sir" is only true at
the moment it was due.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Optional

from jarvis import address
from jarvis import courses as courses_mod
from jarvis.logs import get_logger

log = get_logger("quiet")

HOLD_MAX = 12
TICK_S = 30.0
DEFAULT_KEYWORDS = ("class", "exam", "meeting", "busy")
FOCUS_REASON = "your study block"

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

# A kind that is only worth hearing AT its moment. A stand-up nudge read
# back after a two-hour meeting is noise, and a 45-minute water nudge held
# through a three-hour block would arrive five deep. Held -> expired.
# "nudge" here is an interval reminder (timekeeper.NUDGE_KIND), not the
# app's "Sir?" cue -- that one is an answer and never reaches this gate.
EPHEMERAL_KINDS = ("nudge",)

_KIND_NOUNS = {
    "reminder": ("reminder", "reminders"),
    # A nudge normally never reaches a digest (hold() expires it); the entry
    # keeps a stray one from reading as "one message".
    "nudge": ("nudge", "nudges"),
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
                 get_focus: Optional[Callable] = None,
                 say: Optional[Callable[[str], None]] = None,
                 can_speak: Optional[Callable[[], bool]] = None,
                 now: Callable[[], float] = time.time,
                 hold_max: int = HOLD_MAX, tick_s: float = TICK_S):
        self._cfg = cfg
        self._get_calendar = get_calendar
        self._is_home = is_home
        self._get_focus = get_focus
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

    def hours_end(self, now: Optional[float] = None) -> Optional[float]:
        """When quiet hours next close, as a timestamp -- None when they are
        off. Public seam for the bedtime wind-down (jarvis/winddown.py),
        which arms DND for exactly the window it is joining rather than
        guessing a morning of its own."""
        ts = self._now() if now is None else float(now)
        return self._hours_end(datetime.fromtimestamp(ts).astimezone())

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
        # His courses are not called "class". The shipped keyword list
        # (class / exam / meeting / busy) matches none of BIOSENSORS,
        # MAGNETIC RESONANCE ENGR or ELECTRICAL DESIGN LAB II, so the
        # calendar leg of quiet hours had never once fired for him. A title
        # at the same weekday and clock time on two or more weeks IS a
        # class, whatever it is called (jarvis/courses.py).
        names: tuple = ()
        if self._get("quiet.calendar_courses", True):
            try:
                names = tuple(courses_mod.recurring_courses(events, now=dt))
            except Exception:  # noqa: BLE001 - the cache is best-effort
                log.debug("quiet: course detection failed", exc_info=True)
        if not kws and not names:
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
            if kws and any(k in words for k in kws):
                return ev
            if names and courses_mod.course_for(title, names):
                return ev
        return None

    def _focus_reason(self) -> str:
        """"your study block" while a focus session is mid-block.

        Only ``phase == "block"``. A break is precisely when the backlog
        SHOULD be read, so the policy goes free there and the existing
        ``tick()`` speaks the digest for free. The session's own lines
        ("Time for a break, sir") pierce this because ``FocusSession._speak``
        marks them ``proactive=False`` -- the gate keys on proactive, never
        on kind.
        """
        if self._get_focus is None or not self._get("focus.dnd", True):
            return ""
        try:
            focus = self._get_focus() if callable(self._get_focus) else self._get_focus
            if focus is None or str(getattr(focus, "phase", "") or "") != "block":
                return ""
        except Exception:  # noqa: BLE001 - a focus hiccup must not mute him
            log.debug("quiet: focus probe failed", exc_info=True)
            return ""
        return FOCUS_REASON

    def reason(self, now: Optional[float] = None, calendar: bool = True) -> str:
        """Why he is quiet right now, in his words -- "" when he is not.

        ``calendar=False`` asks the same question WITHOUT the running-event
        leg, which is what a caller that is itself acting on that event
        needs (jarvis/classflow.py stages a class the instant it starts, and
        the class is the quiet window)."""
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
        focus = self._focus_reason()
        if focus:
            return focus
        if self._is_home is not None and self._get("quiet.hold_when_away", True):
            try:
                if not self._is_home():
                    return "you're out"
            except Exception:  # noqa: BLE001
                log.debug("quiet: is_home failed", exc_info=True)
        hours = self._hours_reason(dt)
        if hours:
            return hours
        ev = self._calendar_event(dt) if calendar else None
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
            frags = self.release_fragments(prefix=BUSY_PREFIX)
            if frags:
                # THE JOIN (7 sentences, 5 sirs measured). "Very good, sir."
                # in front of a digest that opens "While you were busy, sir:"
                # says it twice before a held line has been read at all; the
                # acknowledgement is first, so it is the one that survives.
                # The digest's own fragments, NOT the string release() would
                # have joined: thinning a finished multi-sentence string is
                # the mode that killed the first attempt (jarvis/address.py).
                return address.join_fragments([FREE_LINE] + frags)
            return FREE_LINE if was else DND_ALREADY_FREE_LINE

    # ------------------------------------------------------------- hold
    def hold(self, text: str, kind: str = "message") -> bool:
        """Park a line for the digest. False when it was DROPPED instead --
        an ephemeral kind (a nudge) is only worth hearing at its moment, so
        it expires rather than queueing behind the window."""
        text = (text or "").strip()
        if not text:
            return False
        kind = kind or "message"
        if kind in EPHEMERAL_KINDS:
            log.info("quiet: expired (%s): %.80s", kind, text)
            return False
        with self._lock:
            self._held.append((self._now(), text, kind))
        log.info("quiet: held (%s): %.80s", kind, text)
        return True

    @property
    def held(self) -> list:
        with self._lock:
            return list(self._held)

    def take_fragments(self, prefix: Optional[str] = None):
        """The backlog drained for ONE ATTEMPT at speaking it, with a way back.

        Returns ``(fragments, put_back)``. ``release_fragments`` is the
        one-way form of this and it is right for a caller that speaks what
        it took there and then. It is WRONG for a caller that can still
        decide not to speak, and on 2026-09-03 that cost the backlog: the
        arrival catch-up drained the held lines on the pump thread, went
        off to read a mailbox, and then dropped the digest as stale --
        taking the drained lines with it. They are the things he missed
        while he was out and there is no second copy of them anywhere, so
        a caller that might not speak has to be able to give them back.

        ``put_back()`` returns the very items taken, in order, to the FRONT
        of the backlog and re-arms the falling edge, so the policy's own
        next tick reads them out (see ``tick``): held lines nothing will
        ever say are lost by another name. It is idempotent -- twice puts
        them back once -- and returns how many went back. The deque's
        ``maxlen`` still applies to the merged backlog, dropping the oldest
        exactly as it would have had these never been taken.
        """
        with self._lock:
            items = list(self._held)
            self._held.clear()
            reason = self._last_reason
        done = [False]

        def put_back() -> int:
            if done[0] or not items:
                done[0] = True
                return 0
            done[0] = True
            with self._lock:
                merged = items + list(self._held)
                self._held.clear()
                # extend, not extendleft: with a maxlen the deque drops from
                # the end it is not being fed, so feeding it in order drops
                # the OLDEST on overflow -- the same line hold() would have
                # dropped had these never been taken.
                self._held.extend(merged)
                # THE FALLING EDGE, RE-ARMED. tick() releases the digest on
                # the edge where quiet stops being true and never again; the
                # edge that would have said these has already gone by. Put
                # the lines back without this and they sit held until the
                # next quiet window opens and closes -- kept, and mute.
                self._last_quiet = True
            log.info("quiet: %d held line(s) put back unspoken", len(items))
            return len(items)

        if not items:
            return [], put_back
        if prefix is None:
            prefix = AWAY_PREFIX if reason == "you're out" else BUSY_PREFIX
        return digest_fragments(items, prefix), put_back

    def release_fragments(self, prefix: Optional[str] = None) -> list:
        """Drain the held lines into the FRAGMENTS of one spoken digest
        ([] when empty) -- the prefix line and then each held line, still
        separate.

        This is the primitive and ``release`` is the joined form of it: a
        burst that is joined and then thinned again as a finished string is
        exactly what jarvis/address.py refuses to do, so a caller that has
        more to say in the same burst (the arrival cue, "I am free") takes
        the fragments and joins once.

        ONE-WAY. The lines are gone the moment this returns, so the caller
        must be about to say them; one that may still change its mind takes
        ``take_fragments`` and gives them back.
        """
        return self.take_fragments(prefix)[0]

    def release(self, prefix: Optional[str] = None) -> str:
        """Drain the held lines into one spoken digest ("" when empty)."""
        return address.join_fragments(self.release_fragments(prefix))

    # ----------------------------------------------------------- thread
    def tick(self) -> str:
        """Notice a quiet window closing; speak the digest through ``say``.
        Returns what was said ("" when nothing).

        THE LINES ARE ONLY SPENT ONCE THEY HAVE BEEN SAID. This used to
        ``release()`` -- one-way -- and then swallow a TTS failure, so a
        digest that could not be spoken took the whole backlog with it.
        It is the same class of bug the arrival catch-up shipped on
        2026-09-03 (drained on one thread, dropped on another) sitting one
        level up in this file, and one fix does not close a class while
        the other instance stands. A failed speak puts the lines back and
        re-arms the edge, exactly as the ``can_speak`` deferral below
        already does: kept, and retried next tick.
        """
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
            frags, put_back = self.take_fragments()
            text = address.join_fragments(frags)
        if not text:
            # Nothing speakable came out of them. They are not spent.
            put_back()
            return ""
        if callable(self._say):
            try:
                self._say(text)
            except Exception:  # noqa: BLE001 - a TTS failure must not kill the loop
                # ...and must not cost him the lines either.
                log.exception("quiet: digest speak failed; the lines go back")
                put_back()
                return ""
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


def digest_fragments(items, prefix: str = BUSY_PREFIX) -> list:
    """The digest as SEPARATE fragments: the prefix line, then each held line.

    Every one of them is still a whole authored line here, which is the only
    state in which they can safely be thinned against each other -- see
    jarvis/address.py."""
    items = list(items)
    if not items:
        return []
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
    # Every word of the head is quiet.py's own -- the prefix constant and the
    # counts it just made, no slot anywhere in it -- so it is certified
    # (jarvis/address.py, Authored). The colon no longer needs the
    # certificate (a clause the sentence runs on from is droppable on its
    # own); it is kept because the head is the one fragment in this repo
    # that is provably Jarvis's end to end, and because the certificate is
    # what a later prefix ending in a full stop would need.
    head = address.authored(f"{prefix}: {_join_and(parts)}.")
    return [head] + lines


def digest(items, prefix: str = BUSY_PREFIX) -> str:
    """"While you were busy, sir: two reminders and a warning. <lines>".

    THE JOIN. Measured at 6 sentences and 4 sirs: the prefix says it once and
    then every held line says it again -- and with five held reminders it is
    "Sir, this is your reminder" five times over. The prefix's own "sir" is
    the first and therefore the survivor; the second and later copies of one
    summons are thinned to the sentence behind them (jarvis/address.py)."""
    return address.join_fragments(digest_fragments(items, prefix))
