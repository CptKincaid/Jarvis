"""Time to leave, learned from the room in the event -- no maps API.

The meeting heads-up (jarvis/headsup.py) speaks ONE global lead: "BIOSENSORS
in ten minutes, sir". Ten minutes is right for a call and useless for a
lecture on the far side of campus. This module adds a second, earlier,
PER-BUILDING heads-up -- "You want to be walking in 5 minutes, sir" -- and
learns the walk itself by asking once, in passing, after the heads-up:
"How long do you need to get to Wisenbaker, sir?"

Nothing is guessed. A building he has never answered for gets no leave
line at all, ever, and no maps API is consulted; a wrong "you have twenty
minutes" is worse than silence.

The correctness core is ``building_key``. His calendar's location strings
are stable week to week but room-specific::

    College Station Wisenbaker Engineering Bldg 049
    College Station Emerging Technologies Building 1003
    College Station Emerging Technologies Building 1020
    College Station Zachry Engineering Ed. Complex 330
    College Station Jack E. Brown Chem Engn Bldg 731A
    https://tamu.zoom.us/j/94324046592?pwd=...          <- a Zoom link
    ""                                                  <- no location

Twenty-one events, four real buildings. The key strips the city prefix and
the trailing room token so 1003 and 1020 collapse to one walk, and returns
None for the empty strings and the Zoom URL -- asking "how long is the walk
to https://tamu.zoom.us/..." even once is exactly the failure that kills
trust in the feature.

State: the learned minutes live in long-term memory
(``memory.set_preference("leave_lead.<key>", minutes)``) because they are
facts about him, not settings; the filed/asked bookkeeping is an atomic
JSON file under PATHS.MEMORY_DIR in headsup.py's idiom, so a restart
neither re-asks nor double-files.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("leavetime")

INTERVAL_S = 120.0             # a leave line is minutes-accurate; 2 min ticks
HORIZON_MIN = 120              # how far ahead a leave reminder is filed
DEFAULT_NOTICE_MIN = 5         # "you want to be walking in FIVE minutes"
DEFAULT_HEADSUP_MIN = 10       # calendar.heads_up_min, for the ask window
MAX_LEAD_MIN = 120             # a "walk" longer than this is a misheard number
PREF_PREFIX = "leave_lead."
ASK_LINE = "How long do you need to get to {place}, sir?"
UNKNOWN_LINE = "I don't know that walk yet, sir."
LEARNED_LINE = "{place}, {minutes} minutes. I'll have you moving in good time, sir."
FORGOT_LINE = "I've forgotten the walk to {place}, sir."

# The campus city that prefixes every location string his university writes.
# Configurable so a move does not need a code change.
CITY_PREFIXES = ("College Station",)
# A trailing room: "049", "1003", "731A", "110".
_ROOM_RX = re.compile(r"\s+\d{1,4}[A-Za-z]?$")
_URL_RX = re.compile(r"^\s*(?:https?://|www\.)", re.I)
# Words that describe what a building IS rather than which one it is; a
# spoken name drops them ("Wisenbaker Engineering Bldg" -> "Wisenbaker").
_GENERIC_TOKENS = {
    "bldg", "building", "buildings", "hall", "complex", "center", "centre",
    "ed", "engineering", "engn", "chem", "annex", "tower", "lab", "labs",
    "laboratory", "room", "rm",
}


# --------------------------------------------------------------- the key
def building_key(location, city_prefixes=CITY_PREFIXES) -> Optional[str]:
    """A location string -> the stable building it names, or None.

    None for an empty location and for anything that looks like a URL (a
    Zoom link is not a building and must never be asked about). Otherwise
    the city prefix and the trailing room number come off, so the two ETB
    rooms share one walk.
    """
    text = " ".join(str(location or "").split())
    if not text or _URL_RX.match(text):
        return None
    low = text.lower()
    for prefix in city_prefixes:
        p = str(prefix or "").strip().lower()
        if p and low.startswith(p + " "):
            text = text[len(p):].strip()
            break
    text = _ROOM_RX.sub("", text).strip(" ,-")
    return text or None


def speech_name(key: str) -> str:
    """The building as he would say it: "Wisenbaker Engineering Bldg" ->
    "Wisenbaker", "Emerging Technologies Building" -> "Emerging
    Technologies". Never empties the name -- a building called only
    "Building" keeps its word."""
    parts = str(key or "").split()
    while len(parts) > 1 and parts[-1].strip(".").lower() in _GENERIC_TOKENS:
        parts.pop()
    return " ".join(parts)


def match_key(spoken: str, keys) -> Optional[str]:
    """The stored key he means by "Wisenbaker" / "the ETB". Substring
    match on the key and on its spoken name, longest first so "Emerging
    Technologies" cannot be beaten by a shorter accidental hit."""
    said = " ".join(str(spoken or "").split()).strip(" .,?!").lower()
    if not said:
        return None
    best = None
    for key in keys:
        for form in (str(key).lower(), speech_name(key).lower()):
            if not form:
                continue
            if said == form or said in form or form in said:
                if best is None or len(str(key)) > len(str(best)):
                    best = key
    return best


# ------------------------------------------------------------- durations
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twentyfive": 25, "thirty": 30, "forty": 40, "fortyfive": 45,
    "fifty": 50, "sixty": 60, "half": 30,
}
_MIN_RX = re.compile(
    r"(?P<n>\d{1,3}|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) +
    r")(?:[\s-](?P<n2>one|two|three|four|five|six|seven|eight|nine))?"
    r"\s*(?:-\s*)?(?P<unit>minutes?|mins?|hours?|hrs?)\b", re.I)


def parse_minutes(text: str) -> Optional[int]:
    """"about ten minutes" / "15" / "a quarter of an hour" -> minutes.

    None when the utterance carries no duration -- the ask then stays open
    rather than storing a guess.
    """
    s = " ".join(str(text or "").split()).lower().strip(" .!?")
    if not s:
        return None
    s = s.replace("quarter of an hour", "15 minutes").replace(
        "quarter an hour", "15 minutes").replace("half an hour", "30 minutes")
    m = _MIN_RX.search(s)
    if m:
        tok = m.group("n")
        n = int(tok) if tok.isdigit() else _NUM_WORDS.get(tok, 0)
        if m.group("n2"):
            n += _NUM_WORDS.get(m.group("n2"), 0)
        if m.group("unit").lower().startswith(("hour", "hr")):
            n *= 60
        return n if 0 < n <= MAX_LEAD_MIN else None
    # A bare number is an answer only to a question about minutes.
    bare = re.fullmatch(r"(?:about\s+|around\s+|maybe\s+)?"
                        r"(\d{1,3}|" + "|".join(_NUM_WORDS) + r")", s)
    if bare:
        tok = bare.group(1)
        n = int(tok) if tok.isdigit() else _NUM_WORDS.get(tok, 0)
        return n if 0 < n <= MAX_LEAD_MIN else None
    return None


# The spoken answer to "How long do you need to get to Wisenbaker, sir?".
# Deliberately STRICT: the question stays open for a few minutes and a
# loose duration match would steal "set a timer for five minutes" out of
# the middle of it. An answer is a duration and almost nothing else.
_ANSWER_FILLER = ("it's", "its", "it is", "about", "around", "maybe",
                  "roughly", "probably", "takes", "it takes", "i need",
                  "i'd need", "i would need", "call it", "say", "like",
                  "oh", "uh", "um", "a good")
_ANSWER_TAIL = ("or so", "or so sir", "sir", "walk", "away", "from here",
                "on foot", "there", "to get there", "each way", "please")
ANSWER_MAX_WORDS = 4


def answer_minutes(text: str) -> Optional[int]:
    """A duration ANSWER, or None. Fillers and trailing courtesies come
    off; what is left must be short and must parse, so a command that
    merely contains minutes never counts as an answer."""
    s = " ".join(str(text or "").split()).lower().strip(" .!?,")
    s = s.replace("quarter of an hour", "15 minutes").replace(
        "quarter an hour", "15 minutes").replace("half an hour", "30 minutes")
    changed = True
    while changed and s:
        changed = False
        for word in _ANSWER_FILLER:
            if s.startswith(word + " "):
                s, changed = s[len(word) + 1:].strip(), True
        for tail in _ANSWER_TAIL:
            if s.endswith(" " + tail):
                s, changed = s[:-len(tail) - 1].strip(" .,"), True
    if not s or len(s.split()) > ANSWER_MAX_WORDS:
        return None
    return parse_minutes(s)


# "it takes ten minutes to get to Wisenbaker" / "it's a ten minute walk to
# the ETB" / "the walk to Zachry is twelve minutes".
_SET_RXS = (
    re.compile(r"^(?:it(?:'s| is)?\s+)?(?:it\s+)?takes?\s+(?:me\s+)?"
               r"(?P<dur>.+?)\s+to\s+(?:get|walk|drive|cycle|bike)\s+to\s+"
               r"(?P<place>.+)$", re.I),
    re.compile(r"^(?:it(?:'s| is)?\s+)?(?P<dur>.+?)\s+"
               r"(?:walk|ride|drive)\s+to\s+(?P<place>.+)$", re.I),
    re.compile(r"^(?:the\s+)?(?:walk|ride|drive)\s+to\s+(?P<place>.+?)\s+is\s+"
               r"(?P<dur>.+)$", re.I),
    re.compile(r"^(?P<dur>.+?)\s+(?:to|from here to)\s+(?:get\s+to\s+|walk\s+to\s+)"
               r"(?P<place>.+)$", re.I),
)
# "make that ten next time" -- the pitch's edit. correction_kind() does NOT
# match this (it only reads "no, I said X" / "not X, Y"), so it is its own
# pattern rather than a free ride on the existing regex.
_AMEND_RX = re.compile(
    r"^(?:make|call)\s+(?:that|it)\s+(?P<dur>.+?)"
    r"(?:\s+next\s+time)?$", re.I)
_QUERY_RX = re.compile(
    r"^how\s+(?:long|far)\s+"
    r"(?:is\s+it\s+|does\s+it\s+take\s+|do\s+i\s+need\s+|have\s+i\s+got\s+)?"
    r"to\s+(?:get\s+to\s+|walk\s+to\s+)?(?P<place>[a-z][^?]{1,40})$", re.I)


def leave_set_kind(text: str):
    """"it takes ten minutes to get to Wisenbaker" -> ("Wisenbaker", 10).

    None when the utterance is not a walk statement OR when the duration
    does not parse -- a half-heard number must reach the model, never the
    table."""
    s = " ".join(str(text or "").split()).strip(" .!?")
    for rx in _SET_RXS:
        m = rx.match(s)
        if not m:
            continue
        minutes = parse_minutes(m.group("dur"))
        place = m.group("place").strip(" .,")
        if minutes and place:
            return (place, minutes)
    return None


def leave_amend_kind(text: str) -> Optional[int]:
    """"make that ten next time" -> 10 (the building is the last one used)."""
    m = _AMEND_RX.match(" ".join(str(text or "").split()).strip(" .!?"))
    if not m:
        return None
    return parse_minutes(m.group("dur"))


def leave_query_kind(text: str) -> Optional[str]:
    """"how long to Wisenbaker" -> "Wisenbaker"."""
    m = _QUERY_RX.match(" ".join(str(text or "").split()).strip(" .!?"))
    return m.group("place").strip(" .,") if m else None


# "forget the walk to Wisenbaker" -- the way OUT of the table. A walk is
# taught by voice from a half-heard number ("it takes ten minutes to get to
# Wisenbaker"), so a wrong one is entirely routine, and until this matcher
# existed FORGOT_LINE and LeadTable.forget() were both dead: there was no
# command, so a mistaught lead could only be overwritten, never revoked,
# and a building learned as "2 minutes" kept its silent-and-wrong heads-up
# forever. "the walk"/"how long" both appear because they are the two
# phrasings the set and query commands already teach him to say.
_FORGET_RX = re.compile(
    r"^forget\s+(?:the\s+)?(?:walk\s+to\s+|ride\s+to\s+|drive\s+to\s+"
    r"|how\s+long\s+(?:it\s+takes\s+)?to\s+(?:get\s+to\s+|walk\s+to\s+)?)"
    r"(?P<place>[a-z][^?]{1,40})$", re.I)


def leave_forget_kind(text: str) -> Optional[str]:
    """"forget the walk to Wisenbaker" -> "Wisenbaker".

    Deliberately narrow: the bare undo words ("forget it", "scratch that")
    belong to _UNDO_RX / the no-phrases in commander.py, and a matcher that
    swallowed them would eat every dismissal in the app."""
    m = _FORGET_RX.match(" ".join(str(text or "").split()).strip(" .!?"))
    return m.group("place").strip(" .,") if m else None


def leave_line(notice_min: int, place: str, lead_min: int) -> str:
    """The heads-up itself. ``notice_min`` 0 means "right now"."""
    walk = f"{place} is a {lead_min} minute walk" if lead_min else place
    if notice_min <= 0:
        return f"You want to be walking now, sir; {walk}."
    unit = "minute" if notice_min == 1 else "minutes"
    return f"You want to be walking in {notice_min} {unit}, sir; {walk}."


# --------------------------------------------------------------- storage
class LeadTable:
    """The learned walks, kept in long-term memory's preferences.

    Tolerates a missing / broken memory service: an in-memory dict then
    holds the answers for the session so the ask is not repeated in a
    loop, and nothing raises on the tick thread.
    """

    def __init__(self, memory=None):
        self._memory = memory
        self._fallback: dict = {}

    def _get_pref(self, key, default=None):
        get = getattr(self._memory, "get_preference", None)
        if callable(get):
            try:
                return get(PREF_PREFIX + key, default)
            except Exception:  # noqa: BLE001 - a store hiccup is "unknown"
                log.debug("leavetime: get_preference(%s) failed", key,
                          exc_info=True)
        return self._fallback.get(key, default)

    def get(self, key: str) -> Optional[int]:
        value = self._get_pref(key)
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            return None
        return minutes if 0 < minutes <= MAX_LEAD_MIN else None

    def set(self, key: str, minutes: int) -> int:
        minutes = max(1, min(MAX_LEAD_MIN, int(minutes)))
        self._fallback[key] = minutes
        setter = getattr(self._memory, "set_preference", None)
        if callable(setter):
            try:
                setter(PREF_PREFIX + key, minutes)
            except Exception:  # noqa: BLE001 - kept in memory for the session
                log.exception("leavetime: set_preference(%s) failed", key)
        return minutes

    def forget(self, key: str) -> None:
        self._fallback.pop(key, None)
        setter = getattr(self._memory, "set_preference", None)
        if callable(setter):
            try:
                setter(PREF_PREFIX + key, 0)
            except Exception:  # noqa: BLE001
                log.exception("leavetime: forget(%s) failed", key)

    def keys(self) -> list:
        """Every building he has answered for."""
        out = set(self._fallback)
        getter = getattr(self._memory, "get_all_preferences", None)
        if callable(getter):
            try:
                prefs = getter() or {}
            except Exception:  # noqa: BLE001
                log.debug("leavetime: get_all_preferences failed", exc_info=True)
                prefs = {}
            for name, value in dict(prefs).items():
                if str(name).startswith(PREF_PREFIX):
                    try:
                        if int(value) > 0:
                            out.add(str(name)[len(PREF_PREFIX):])
                    except (TypeError, ValueError):
                        continue
        return sorted(k for k in out if self.get(k))


# ----------------------------------------------------------------- watch
class LeaveTimes:
    """The per-building leave heads-up and the ask that learns it.

    ``ask(key, place)`` is the app's door: it speaks the question and arms
    the commander's pending answer. It is called at most ONCE per building,
    only inside the meeting heads-up window (so it lands in passing after
    "BIOSENSORS in ten minutes, sir"), and never while the quiet policy
    says he is heads-down.
    """

    def __init__(self, get_calendar: Callable, timekeeper, table: LeadTable,
                 ask: Optional[Callable[[str, str], bool]] = None,
                 cfg=None, quiet=None, state_path: Optional[Path] = None,
                 now: Optional[Callable] = None,
                 horizon_min: int = HORIZON_MIN):
        self._get_calendar = get_calendar
        self._tk = timekeeper
        self.table = table
        self._ask = ask
        self._cfg = cfg
        self._quiet = quiet
        self._state_path = Path(state_path) if state_path else None
        self._now = now or (lambda tz=None: datetime.now(tz))
        self.horizon_min = horizon_min
        state = self._load()
        self._filed: dict = state.get("filed", {})
        self._asked: dict = state.get("asked", {})
        self.last_key: str = str(state.get("last_key") or "")
        # When a walk was last TAUGHT, monotonic and in memory only --
        # deliberately never restored from disk. `last_key` survives a
        # restart, so a disk-loaded one is a building nothing has been said
        # about; commander._leave_key_fresh honours whichever of this and
        # its own per-turn stamp is newer, and this is the half that covers
        # a teaching path the commander never handles. NOT stamped by
        # note_key: the background reminder tick and _maybe_ask both call
        # that with no user turn behind them, and "make it twenty" must not
        # come to mean a building a thread re-pointed at.
        self.last_touch: float = 0.0
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------ config
    def _cfg_int(self, key: str, default: int) -> int:
        get = getattr(self._cfg, "get", None)
        if not callable(get):
            return default
        try:
            return max(0, int(get(key, default)))
        except Exception:  # noqa: BLE001 - a config hiccup keeps the default
            return default

    @property
    def notice_min(self) -> int:
        return max(1, self._cfg_int("calendar.leave_notice_min",
                                    DEFAULT_NOTICE_MIN))

    @property
    def headsup_min(self) -> int:
        return max(1, self._cfg_int("calendar.heads_up_min",
                                    DEFAULT_HEADSUP_MIN))

    @property
    def enabled(self) -> bool:
        get = getattr(self._cfg, "get", None)
        if not callable(get):
            return True
        try:
            return bool(get("calendar.leave_times", True))
        except Exception:  # noqa: BLE001
            return True

    # ------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if not isinstance(data, dict):
                    return {}
                # str -> ISO-string only, headsup.py's lesson: one stray int
                # or null from a hand edit raised TypeError on every tick.
                out = {"last_key": str(data.get("last_key") or "")}
                for name in ("filed", "asked"):
                    raw = data.get(name)
                    out[name] = {k: v for k, v in (raw or {}).items()
                                 if isinstance(k, str) and isinstance(v, str)} \
                        if isinstance(raw, dict) else {}
                return out
        except (OSError, ValueError):
            log.debug("leavetime state unreadable", exc_info=True)
        return {}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps({"filed": self._filed, "asked": self._asked,
                                       "last_key": self.last_key}))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("leavetime state save failed", exc_info=True)

    def known_keys(self) -> list:
        """Every building he could mean: the ones already learned plus the
        ones standing in his calendar. Nothing else -- "how long to the
        moon" resolves to None and the handler falls through to the model
        rather than inventing a walk."""
        keys = set(self.table.keys())
        for ev in self._events():
            key = building_key(getattr(ev, "location", ""))
            if key:
                keys.add(key)
        return sorted(keys)

    def resolve(self, spoken: str) -> Optional[str]:
        """The stored key he means by "Wisenbaker", or None."""
        return match_key(spoken, self.known_keys())

    def note_key(self, key: str) -> None:
        """Remember the building "make that ten next time" refers to."""
        if key and key != self.last_key:
            self.last_key = key
            self._save()

    def learn(self, key: str, minutes: int) -> int:
        """Store a walk and stop asking about it."""
        value = self.table.set(key, minutes)
        self._asked[key] = self._now().isoformat()
        # A walk just taught IS the building on the table, so "make that
        # ten next time" may follow straight on. Stamped here rather than
        # in note_key so the amend window opens for every teaching path,
        # including one that reaches this store without going through a
        # commander handler.
        self.last_touch = time.monotonic()
        self.note_key(key)
        self._save()
        log.info("leavetime: %s learned as %d min", key, value)
        return value

    def forget(self, key: str) -> None:
        """Drop a learned walk and let it be asked about again.

        `LeadTable.forget` alone is not enough: `_asked` is what stops the
        proactive "How long do you need to get to X, sir?" from ever firing
        twice, so forgetting without clearing it left the building unknown
        AND unaskable -- no leave line and no way back to one short of an
        outright "it takes N minutes to get to X". `_filed` is keyed by
        event, not by building, and those reminders are already in the
        timekeeper's hands, so they are left alone."""
        self.table.forget(key)
        self._asked.pop(key, None)
        self._save()
        log.info("leavetime: %s forgotten", key)

    # -------------------------------------------------------------- tick
    def _events(self) -> list:
        cal = self._get_calendar() if callable(self._get_calendar) else self._get_calendar
        if cal is None:
            return []
        try:
            conf = getattr(cal, "configured", True)
            if callable(conf):                 # CalendarSource exposes a property;
                conf = conf()                  # a fake may expose a method
            if not conf:
                return []
            return list(cal.events())
        except Exception:  # noqa: BLE001 - the cache is best-effort
            log.debug("leavetime: calendar unavailable", exc_info=True)
            return []

    def _holding(self) -> bool:
        should = getattr(self._quiet, "should_hold", None)
        if not callable(should):
            return False
        try:
            return bool(should())
        except Exception:  # noqa: BLE001 - a gate failure must not ask anyway
            log.debug("leavetime: quiet gate failed", exc_info=True)
            return True

    def tick(self) -> int:
        """File the leave heads-ups that are due and, at most once per
        building, ask for the walk. Returns how many reminders were filed."""
        if not self.enabled or self._tk is None:
            return 0
        filed, unknown = 0, []
        notice = self.notice_min
        for ev in self._events():
            start = getattr(ev, "start", None)
            if start is None or getattr(ev, "all_day", False):
                continue
            key = building_key(getattr(ev, "location", ""))
            if key is None:
                continue
            now = self._now(start.tzinfo)
            ahead = start - now
            if ahead <= timedelta(0) or ahead > timedelta(minutes=self.horizon_min):
                continue
            lead = self.table.get(key)
            if lead is None:
                unknown.append((ahead, key, ev))
                continue
            title = (getattr(ev, "title", "") or "").strip() or "your appointment"
            fkey = f"leave|{title}|{start.isoformat()}"
            if fkey in self._filed:
                continue
            leave_at = start - timedelta(minutes=lead)
            due = leave_at - timedelta(minutes=notice)
            if due <= now:
                # We came in late: say how much of the notice is actually
                # left rather than a stale "in 5 minutes".
                left = int((leave_at - now).total_seconds() // 60)
                if left < 0:
                    continue                   # the walk has already started
                due, said = now + timedelta(seconds=5), max(0, left)
            else:
                said = notice
            text = leave_line(said, speech_name(key), lead)
            try:
                self._tk.add_reminder(due.timestamp(), text)
            except Exception:  # noqa: BLE001 - one event must not kill the tick
                log.exception("leavetime: reminder for %r failed", title)
                continue
            self._filed[fkey] = start.isoformat()
            self.note_key(key)
            filed += 1
            log.info("leavetime: %r filed for %s", text, due.strftime("%H:%M"))
        self._maybe_ask(unknown)
        self._prune()
        if filed:
            self._save()
        return filed

    def _maybe_ask(self, unknown: list) -> bool:
        """Ask about the SOONEST unknown building, once ever, and only
        inside the heads-up window so it lands in passing after "BIOSENSORS
        in ten minutes, sir" rather than out of nowhere."""
        if not callable(self._ask) or not unknown:
            return False
        window = timedelta(minutes=self.headsup_min)
        for ahead, key, _ev in sorted(unknown, key=lambda t: t[0]):
            if key in self._asked or ahead > window:
                continue
            if self._holding():
                return False                   # heads-down: try the next tick
            place = speech_name(key)
            try:
                asked = self._ask(key, place)
            except Exception:  # noqa: BLE001 - the tick must survive the app
                log.exception("leavetime: ask for %r failed", key)
                return False
            if not asked:
                return False                   # a turn is in flight; retry later
            self._asked[key] = self._now().isoformat()
            self.note_key(key)
            self._save()
            log.info("leavetime: asked for the walk to %s", place)
            return True
        return False

    def _prune(self) -> None:
        cutoff = (self._now() - timedelta(days=1)).isoformat()
        stale = [k for k, v in self._filed.items()
                 if not isinstance(v, str) or v < cutoff]
        for k in stale:
            self._filed.pop(k, None)
        if stale:
            self._save()

    # ----------------------------------------------------------- thread
    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again and a dead thread must not block a fresh one.
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="leavetime")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _run(self) -> None:
        # first pass a little after boot so the calendar has refreshed
        if self._stop.wait(25.0):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("leavetime tick failed")
            if self._stop.wait(INTERVAL_S):
                return
