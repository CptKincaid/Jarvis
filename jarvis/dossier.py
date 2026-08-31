"""The pre-class dossier: the heads-up stops being a bare title.

``headsup.tick()`` files "BIOSENSORS in 10 minutes" with the timekeeper and
that is the whole of it -- the room he has to walk to, the notes he took
last time, the report due Thursday and the mail from the TA are all data
Jarvis already holds and Hunter has to go and find. This module turns that
one line into::

    Your 9:10 is BIOSENSORS, Wisenbaker 049, sir. Last time you noted
    electrode drift; Lab 3 report is due Thursday, and there's unread mail
    from the TA.

plus ONE card on the HUD carrying the raw room (or the join link, when the
event's location is a URL -- his 9/4 Zoom is exactly that), the last notes
file, the course's deadlines and the matching unread mail.

**Why it does not ride the reminder.** ``Timekeeper.add_reminder(due, text)``
takes a string: a longer string is a longer spoken line and still no card.
So the dossier files a SILENT timer instead (``add_silent_timer``, the
focus.py pattern), catches its own ``item_id`` off ``ReminderFired``,
gathers, and publishes ``BriefingReady`` itself -- which also means the
speech goes through ``services.speak(proactive=True)`` and is held by quiet
hours / DND like every other thing he decided to say on his own.

**Every section is optional and dark by default on this box.** ~/Documents
does not exist, so ``lecture.notes_dir`` resolves under a missing tree and
there is no previous note to find; ``canvas.token`` is empty, so there are
no deadlines. What survives is the room, and the room alone is worth the
line. Nothing here raises: a source that fails is a section that is absent.

**Nothing blocks the tick thread.** ``tick()`` only files timers. The
gathering -- an IMAP round trip and a Canvas call -- happens on a worker
thread with a wall-clock budget when the silent timer fires, off both the
300 s tick thread and the Tk thread the bus delivers on.
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

from jarvis import courses as courses_mod
from jarvis import lecture as lecture_mod
from jarvis.events import BriefingReady, ReminderFired, bus
from jarvis.logs import get_logger

log = get_logger("dossier")

INTERVAL_S = 300.0             # the same cadence as headsup.py
HORIZON_MIN = 90
DEFAULT_LEAD_MIN = 10
DEFAULT_DUE_DAYS = 7
DEFAULT_MAIL_HOURS = 72
DEFAULT_BUDGET_S = 25.0        # the whole gather, IMAP included
MAIL_MAX = 3
DUE_MAX = 3
TOPIC_CHARS = 70
COURSE_MEMO_S = 60.0           # re-derive the course list at most this often

# Location wording (the VOICE line only; the card keeps the raw string).
CITY_PREFIXES = ("college station",)
BUILDING_WORDS = ("bldg", "building", "complex", "hall", "center", "centre", "ctr")
GENERIC_TAIL = ("engineering", "engr", "engn", "ed", "ed.", "chem", "chemical",
                "tech", "academic", "sciences", "science")
_ROOM_RX = re.compile(r"^\d{1,4}[A-Za-z]?$")

NOTE_LINE_RX = re.compile(r"^-\s+\d{1,2}:\d{2}\s+(.*\S)")
NOTE_NAME_RX = re.compile(r"^(?P<slug>.+)-(?P<date>\d{4}-\d{2}-\d{2})\.md$")


# ----------------------------------------------------------------- wording
def is_link(location) -> bool:
    return str(location or "").strip().lower().startswith(("http://", "https://"))


def clock(dt: datetime) -> str:
    """9:10 -- the bare clock. "Your 9:10 is BIOSENSORS" wants no meridiem:
    a class ten minutes away is not in the other half of the day."""
    return f"{dt.hour % 12 or 12}:{dt.minute:02d}"


def room_words(location) -> str:
    """'College Station Wisenbaker Engineering Bldg 049' -> 'Wisenbaker 049'.

    The iCloud room strings are a city, a building name, a building word and
    a number. Only the building name and the number are worth saying aloud;
    the card still carries the string exactly as the calendar wrote it."""
    raw = " ".join(str(location or "").split())
    if not raw or is_link(raw):
        return ""
    words = raw.split()
    room = ""
    if _ROOM_RX.match(words[-1]):
        room, words = words[-1], words[:-1]
    low = " ".join(words).lower()
    for city in CITY_PREFIXES:
        if low.startswith(city + " "):
            words = words[len(city.split()):]
            break
    cut = [i for i, w in enumerate(words) if w.lower().strip(".") in BUILDING_WORDS]
    if cut:
        words = words[:cut[0]]
    while len(words) > 1 and words[-1].lower().strip(".") in GENERIC_TAIL:
        words = words[:-1]
    name = " ".join(words[-3:]).strip()
    out = f"{name} {room}".strip()
    return out or raw


def _join_sentence(parts: list) -> str:
    """'A; B, and C.' -- the pitch's shape, with 1 and 2 parts handled."""
    parts = [p.strip().rstrip(".") for p in parts if str(p).strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        body = parts[0]
    else:
        body = "; ".join(parts[:-1]) + ", and " + parts[-1]
    return body[0].upper() + body[1:] + "."


def _mail_who(mails: list) -> str:
    names, seen = [], set()
    for m in mails:
        who = (getattr(m, "from_name", "") or getattr(m, "from_addr", "") or "").strip()
        key = who.lower()
        if who and key not in seen:
            seen.add(key)
            names.append(who)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} and {len(names) - 1} others"


# ------------------------------------------------------------ the builder
def build_dossier(ev, cfg, *, course: str = "", notes=None, due=None,
                  mail=None, now: Optional[datetime] = None) -> tuple:
    """(sections, spoken) for one class. Pure: every source is injected and
    every section is skipped when its input is empty or missing.

    ``cfg`` is in the signature because the caller has one and a wording
    preference will land here (there is nothing to read from it yet); the
    sources are all resolved before the call precisely so that this
    function never touches a disk or a socket.

    ``notes``  {"path": Path, "date": "YYYY-MM-DD", "topic": str}
    ``due``    canvas.fetch_due items already filtered to this course
    ``mail``   jarvis.tools.mail.Mail objects already filtered to this course
    """
    from jarvis.tools.canvas import when_words

    title = " ".join(str(getattr(ev, "title", "") or "").split()) or "your class"
    course = " ".join(str(course or "").split()) or title
    start = getattr(ev, "start", None)
    location = " ".join(str(getattr(ev, "location", "") or "").split())
    now = now or (start - timedelta(minutes=DEFAULT_LEAD_MIN) if start else datetime.now())

    sections: dict = {}
    if start is not None:
        sections["calendar"] = [f"{clock(start)} — {title}"]
    if is_link(location):
        sections["join"] = location
    elif location:
        sections["room"] = location

    # -- the spoken head. Room first: it is the one thing he has to act on.
    where = room_words(location) if not is_link(location) else ""
    when = clock(start) if start is not None else "next"
    if where:
        head = f"Your {when} is {title}, {where}, sir."
    elif is_link(location):
        head = f"Your {when} is {title}, sir; the link's on the card."
    else:
        head = f"Your {when} is {title}, sir."

    parts = []
    if notes:
        topic = " ".join(str(notes.get("topic") or "").split())[:TOPIC_CHARS]
        path = notes.get("path")
        if topic:
            parts.append(f"last time you noted {topic}")
        label = topic or "notes"
        if path:
            sections["last"] = f"{label} ({Path(path).name})"
        elif topic:
            sections["last"] = label

    rows = []
    for item in (due or [])[:DUE_MAX]:
        if not isinstance(item, dict):
            continue
        what = " ".join(str(item.get("title") or "").split())
        at = item.get("due")
        if not what or not isinstance(at, datetime):
            continue
        rows.append(f"{what} — {when_words(at, now)}")
        if len(rows) == 1:
            parts.append(f"{what} is due {when_words(at, now)}")
    if rows:
        sections["due"] = rows

    mails = [m for m in (mail or [])][:MAIL_MAX]
    if mails:
        sections["mail"] = [
            f"{(getattr(m, 'from_name', '') or getattr(m, 'from_addr', '') or 'someone')}"
            f" — {getattr(m, 'subject', '') or '(no subject)'}" for m in mails]
        who = _mail_who(mails)
        parts.append(f"there's unread mail from {who}" if who else "there's unread mail")

    tail = _join_sentence(parts)
    spoken = f"{head} {tail}" if tail else head
    return sections, spoken


# --------------------------------------------------------------- sources
def last_notes(cfg, course: str, before: Optional[datetime] = None) -> Optional[dict]:
    """The newest lecture-notes file for ``course`` from a day BEFORE
    ``before``, with the first line he actually noted. None when the folder
    is missing (it is, on this box), empty, or holds only headers.

    The first noted line, not a summary: it is what the last lecture opened
    with, it is deterministic, and it needs no model on a proactive path."""
    slug = lecture_mod.slugify(course)
    day = (before or datetime.now()).date().isoformat()
    try:
        folder = lecture_mod.notes_dir(cfg)
        names = sorted(p for p in folder.glob(f"{slug}-*.md") if p.is_file())
    except OSError:
        log.debug("dossier: notes folder unreadable", exc_info=True)
        return None
    for path in reversed(names):
        m = NOTE_NAME_RX.match(path.name)
        if not m or m.group("slug") != slug or m.group("date") >= day:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            hit = NOTE_LINE_RX.match(line.strip())
            if hit:
                return {"path": path, "date": m.group("date"),
                        "topic": hit.group(1).strip()}
        # header only (the stager primes an empty file): keep looking back
    return None


def course_due(items, course: str) -> list:
    """The Canvas items belonging to ``course``, soonest first."""
    out = []
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        name = str(it.get("course") or "")
        if courses_mod.course_for(name, [course]) or courses_mod.course_for(course, [name]):
            out.append(it)
    out.sort(key=lambda i: i.get("due") or datetime.max)
    return out


def _course_tokens(course: str) -> list:
    """The words of a course title distinctive enough to look for in a
    subject line: 'BIOSENSORS' yes, 'II' and 'LAB' no."""
    return [w for w in courses_mod.fold(course).split() if len(w) >= 5]


def mentions_course(mail, course: str) -> bool:
    hay = f"{getattr(mail, 'subject', '') or ''} {getattr(mail, 'snippet', '') or ''}"
    hay = courses_mod.fold(hay)
    toks = _course_tokens(course)
    return bool(toks) and any(t in hay for t in toks)


def from_people_book(mail, people: dict) -> bool:
    """True when the sender is someone in the people book (jarvis/memory.py):
    an unread newsletter mentioning the course is not news."""
    from jarvis.tools.mail import sender_matches

    for person in (people or {}).values():
        if not isinstance(person, dict):
            continue
        for field in ("email", "name"):
            wanted = str(person.get(field) or "").strip()
            if wanted and sender_matches(mail, wanted):
                return True
    return False


def course_mail(mails, course: str, people: dict) -> list:
    return [m for m in (mails or [])
            if mentions_course(m, course) and from_people_book(m, people)]


# ------------------------------------------------------------ the filer
class ClassDossier:
    """Files a silent timer ``lead_min`` before each course event, then
    gathers and delivers the dossier when it fires."""

    def __init__(self, cfg, timekeeper, get_calendar: Callable = None,
                 services=None, lead_min: int = DEFAULT_LEAD_MIN,
                 state_path: Optional[Path] = None, now: Callable = None,
                 horizon_min: int = HORIZON_MIN, bg: Optional[Callable] = None,
                 fetch_mail: Optional[Callable] = None,
                 fetch_due: Optional[Callable] = None,
                 publish: Optional[Callable] = None):
        self._cfg = cfg
        self._tk = timekeeper
        self._get_calendar = get_calendar
        self._services = services
        self.lead_min = max(1, int(lead_min or DEFAULT_LEAD_MIN))
        self.horizon_min = horizon_min
        self._state_path = Path(state_path) if state_path else None
        self._now = now or (lambda tz=None: datetime.now(tz))
        self._bg = bg or self._thread_bg
        self._fetch_mail = fetch_mail
        self._fetch_due = fetch_due
        self._publish = publish or bus.publish
        self._lock = threading.RLock()
        self._workers: list = []
        self._courses: Optional[tuple] = None       # (monotonic, names)
        state = self._load()
        self._filed: dict = state.get("filed", {})
        self._pending: dict = state.get("pending", {})
        self._stop = threading.Event()
        self._thread = None
        bus.subscribe(ReminderFired, self._on_reminder)

    # ------------------------------------------------------------ config
    def _get(self, key: str, default=None):
        get = getattr(self._cfg, "get", None)
        if not callable(get):
            return default
        try:
            value = get(key, default)
        except Exception:                   # noqa: BLE001 - config boundary
            log.debug("dossier: config read failed for %s", key, exc_info=True)
            return default
        return default if value is None else value

    @property
    def enabled(self) -> bool:
        return bool(self._get("dossier.enabled", True))

    # ------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict):
                    filed = data.get("filed")
                    pending = data.get("pending")
                    return {"filed": filed if isinstance(filed, dict) else {},
                            "pending": pending if isinstance(pending, dict) else {}}
        except (OSError, ValueError):
            log.debug("dossier state unreadable", exc_info=True)
        return {"filed": {}, "pending": {}}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps({"filed": self._filed, "pending": self._pending}))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("dossier state save failed", exc_info=True)

    # ----------------------------------------------------------- calendar
    def _events(self) -> list:
        cal = self._get_calendar() if callable(self._get_calendar) else self._get_calendar
        if cal is None:
            return []
        try:
            conf = getattr(cal, "configured", True)
            if callable(conf):
                conf = conf()
            if not conf:
                return []
            return list(cal.events())
        except Exception:                   # noqa: BLE001 - the cache is best-effort
            log.debug("dossier: calendar unavailable", exc_info=True)
            return []

    def courses(self, events=None) -> tuple:
        """The course titles, memoised for COURSE_MEMO_S -- headsup asks
        once per event and the stager asks every tick."""
        clock_now = time.monotonic()
        memo = self._courses
        # An EMPTY answer is memoised too: headsup asks owns() once per
        # event, and a box with no recurring calendar would otherwise
        # re-derive the (empty) course list for every event on every tick.
        if memo is not None and clock_now - memo[0] < COURSE_MEMO_S:
            return memo[1]
        evs = self._events() if events is None else events
        names = tuple(courses_mod.recurring_courses(evs))
        # Canvas names too when something has already warmed its cache; it
        # is read-only and answers [] with no token, which is the case here.
        try:
            from jarvis.tools.canvas import cached_course_names
            for name in cached_course_names():
                if not courses_mod.course_for(name, list(names)):
                    names = names + (name,)
        except Exception:                   # noqa: BLE001 - source boundary
            log.debug("dossier: canvas roster unavailable", exc_info=True)
        self._courses = (clock_now, names)
        return names

    def owns(self, ev) -> bool:
        """True when this event is a class the dossier will speak for --
        headsup.py asks so its bare "TITLE in ten minutes" stands down."""
        if not self.enabled or getattr(ev, "all_day", False):
            return False
        return bool(courses_mod.course_for(getattr(ev, "title", ""), list(self.courses())))

    # -------------------------------------------------------------- tick
    def tick(self) -> int:
        """File a silent timer for each upcoming class; returns how many."""
        if self._tk is None or not self.enabled:
            return 0
        events = self._events()
        if not events:
            return 0
        names = list(self.courses(events))
        if not names:
            return 0
        filed = 0
        for ev in events:
            start = getattr(ev, "start", None)
            if start is None or getattr(ev, "all_day", False):
                continue
            course = courses_mod.course_for(getattr(ev, "title", ""), names)
            if not course:
                continue
            now = self._now(start.tzinfo)
            if start <= now or start - now > timedelta(minutes=self.horizon_min):
                continue
            title = " ".join(str(getattr(ev, "title", "") or "").split())
            key = f"{title}|{start.isoformat()}"
            with self._lock:
                if key in self._filed:
                    continue
                delay = max(1.0, (start - now).total_seconds()
                            - self.lead_min * 60.0)
                try:
                    item = self._tk.add_silent_timer(delay, f"dossier {title}")
                except Exception:
                    log.exception("dossier: silent timer for %r failed", title)
                    continue
                self._filed[key] = start.isoformat()
                self._pending[str(item.id)] = {
                    "key": key, "title": title, "course": course,
                    "start": start.isoformat(),
                    "location": " ".join(str(getattr(ev, "location", "") or "").split()),
                }
                filed += 1
            log.info("dossier: %r filed for %s (T-%d)", title,
                     (start - timedelta(minutes=self.lead_min)).strftime("%H:%M"),
                     self.lead_min)
        self._prune()
        if filed:
            self._save()
        return filed

    def _prune(self) -> None:
        cutoff = (self._now() - timedelta(days=1)).isoformat()
        with self._lock:
            stale = [k for k, v in self._filed.items()
                     if not isinstance(v, str) or v < cutoff]
            for k in stale:
                self._filed.pop(k, None)
            gone = [i for i, e in self._pending.items()
                    if not isinstance(e, dict) or str(e.get("start", "")) < cutoff]
            for i in gone:
                self._pending.pop(i, None)
        if stale or gone:
            self._save()

    # ------------------------------------------------------- the delivery
    def _on_reminder(self, ev) -> None:
        item_id = str(getattr(ev, "item_id", "") or "")
        if not item_id:
            return
        with self._lock:
            entry = self._pending.pop(item_id, None)
        if entry is None:
            return                          # someone else's timer
        self._save()
        # OFF the Tk thread the bus delivers on: IMAP and Canvas are seconds.
        self._bg(lambda: self.deliver(entry))

    def _thread_bg(self, fn) -> None:
        t = threading.Thread(target=fn, daemon=True, name="dossier-gather")
        with self._lock:
            self._workers = [w for w in self._workers if w.is_alive()]
            self._workers.append(t)
        t.start()

    def deliver(self, entry: dict) -> str:
        """Gather, build, publish the card and speak. Returns what was
        said, or "" when the state entry was unusable."""
        try:
            from jarvis.tools.calendar import Event
            ev = Event.from_dict({"start": entry["start"], "end": entry["start"],
                                  "all_day": False, "title": entry.get("title", ""),
                                  "location": entry.get("location", "")})
        except Exception:                   # noqa: BLE001 - state boundary
            log.exception("dossier: unusable state entry %r", entry)
            return ""
        course = entry.get("course") or ev.title
        deadline = time.monotonic() + self._budget()
        notes = self._gather(lambda: last_notes(self._cfg, course, ev.start), "notes",
                             deadline) if self._get("dossier.notes", True) else None
        due = self._gather(lambda: self._due_items(course), "due", deadline) or []
        mail = self._gather(lambda: self._mail_items(course), "mail", deadline) or []
        sections, spoken = build_dossier(ev, self._cfg, course=course, notes=notes,
                                         due=due, mail=mail,
                                         now=self._now(ev.start.tzinfo))
        try:
            self._publish(BriefingReady(sections=sections, spoken=spoken))
        except Exception:                   # noqa: BLE001 - bus boundary
            log.exception("dossier: card publish failed")
        self._speak(spoken)
        log.info("dossier: %r — %d section(s)", course, len(sections))
        return spoken

    def _budget(self) -> float:
        try:
            return max(1.0, float(self._get("dossier.budget_s", DEFAULT_BUDGET_S)))
        except (TypeError, ValueError):
            return DEFAULT_BUDGET_S

    def _gather(self, fn, what: str, deadline: float):
        """One source, never fatal: past the budget it is simply skipped."""
        if time.monotonic() >= deadline:
            log.info("dossier: %s skipped (out of budget)", what)
            return None
        try:
            return fn()
        except Exception:                   # noqa: BLE001 - source boundary
            log.info("dossier: %s unavailable", what, exc_info=True)
            return None

    def _due_items(self, course: str) -> list:
        from jarvis.tools.canvas import canvas_settings, fetch_due
        settings = canvas_settings(self._cfg)
        if settings is None:
            return []                       # no token: the section stays dark
        days = int(self._get("dossier.due_days", DEFAULT_DUE_DAYS) or DEFAULT_DUE_DAYS)
        items = (self._fetch_due or fetch_due)(settings, days)
        return course_due(items, course)

    def _mail_items(self, course: str) -> list:
        if not self._get("dossier.mail", True):
            return []
        from jarvis.tools.mail import fetch_unread
        hours = int(self._get("dossier.mail_hours", DEFAULT_MAIL_HOURS)
                    or DEFAULT_MAIL_HOURS)
        mails = (self._fetch_mail or fetch_unread)(self._cfg, hours, 30)
        people = {}
        memory = getattr(self._services, "memory", None)
        getter = getattr(memory, "people", None)
        if callable(getter):
            try:
                people = getter() or {}
            except Exception:               # noqa: BLE001 - store boundary
                log.debug("dossier: people book unavailable", exc_info=True)
        return course_mail(mails, course, people)

    def _speak(self, text: str) -> None:
        say = getattr(self._services, "speak", None)
        if not text or not callable(say):
            return
        try:
            try:
                # proactive: quiet hours / DND hold it for the catch-up
                # digest, where it reads as a heads-up rather than a warning
                # (the services speak lambda defaults to the watchdog's kind).
                say(text, proactive=True, kind="heads-up")
            except TypeError:
                say(text)                   # a bare test seam takes text only
        except Exception:                   # noqa: BLE001 - speech boundary
            log.exception("dossier: speak failed")

    # ----------------------------------------------------------- thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="dossier")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            bus.unsubscribe(ReminderFired, self._on_reminder)
        except Exception:                   # noqa: BLE001 - bus boundary
            log.debug("dossier: unsubscribe failed", exc_info=True)
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        with self._lock:
            workers = list(self._workers)
        for w in workers:                   # an IMAP fetch must not outlive quit
            if w is not threading.current_thread():
                w.join(timeout=2.0)

    def _run(self) -> None:
        if self._stop.wait(25.0):           # let the calendar refresh first
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("dossier tick failed")
            if self._stop.wait(INTERVAL_S):
                return
