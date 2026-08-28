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
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.location import (cache_dir, cfg_get, clock_words, http_get,
                                   setup_line, system_tz)
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.calendar")

REFRESH_S = 600                 # refresh period and the "stale" threshold
FETCH_TIMEOUT = 8
WINDOW_DAYS = 14                # how far ahead the cache reaches
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

    def key(self) -> tuple:
        return (self.start.isoformat(), self.end.isoformat(), self.title.lower(),
                self.all_day)

    def on(self, day: date) -> bool:
        """True when the event touches ``day`` (all-day spans included)."""
        if self.all_day:
            return self.start.date() <= day < max(self.end.date(),
                                                  self.start.date() + timedelta(days=1))
        return self.start.date() == day

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "all_day": self.all_day, "title": self.title,
                "calendar": self.calendar, "location": self.location}

    @classmethod
    def from_dict(cls, d: dict, tz=None) -> "Event":
        start = datetime.fromisoformat(d["start"])
        end = datetime.fromisoformat(d["end"])
        if tz is not None:
            start, end = start.astimezone(tz), end.astimezone(tz)
        return cls(start=start, end=end, all_day=bool(d.get("all_day")),
                   title=str(d.get("title") or ""),
                   calendar=str(d.get("calendar") or ""),
                   location=str(d.get("location") or ""))


def now_local(tz=None) -> datetime:
    """The current wall clock, as one module-level seam.  Tests freeze this
    instead of the ``datetime`` name, which ``isinstance`` checks rely on."""
    return datetime.now(tz or system_tz())


def _clean(text) -> str:
    return " ".join(str(text or "").split())


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
                         calendar=name, location=_clean(comp.get("LOCATION", ""))))
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
def _duration_words(ev: Event) -> str:
    minutes = int((ev.end - ev.start).total_seconds() // 60)
    if minutes < 60:
        return ""
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


def coerce_range(value) -> str:
    """Loose model values -> one of RANGES ("this week" -> "week")."""
    text = _clean(value).lower().strip(" .?!")
    if not text:
        return "today"
    if text in RANGES:
        return text
    # "on monday", "for Monday", "this monday"
    for day in WEEKDAYS:
        if re.search(rf"\b{day}\b", text):
            return day
    if "tomorrow" in text or text in ("tmrw", "tmr"):
        return "tomorrow"
    if "week" in text or "7 day" in text or "seven day" in text:
        return "week"
    if "next" in text or "upcoming" in text or "soon" in text or "coming up" in text:
        return "next"
    return "today"


def format_events(events, range: str = "today", now: datetime = None) -> str:
    """The compact text for today / tomorrow / week / next at ``now``."""
    range = coerce_range(range)
    now = now or now_local()
    today = now.date()
    events = merge_events(events)
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
class Snapshot:
    events: list = field(default_factory=list)
    fetched_at: Optional[float] = None      # oldest successful source stamp
    stale: bool = True
    errors: list = field(default_factory=list)


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
        with self._lock:
            stamps = [e["fetched_at"] for e in self._sources.values()]
        return min(stamps) if stamps else None

    def events(self) -> list[Event]:
        with self._lock:
            groups = [list(e["events"]) for e in self._sources.values()]
        return merge_events(*groups)

    def is_stale(self, now: float = None) -> bool:
        fetched = self.fetched_at
        now = self._clock() if now is None else now
        return fetched is None or now - fetched > self.refresh_s

    # -- fetching ----------------------------------------------------
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
        GETs answer 304)."""
        with self._refresh_lock:
            return self._refresh(now)

    def _refresh(self, now: datetime = None) -> bool:
        start, end = self._window(now)
        stamp = self._clock()
        errors, fresh, ok_any = [], {}, False
        for i, url in enumerate(self.ical_urls, 1):
            sid = f"google-{i}"
            try:
                raw = self._get_ical(url)
                fresh[sid] = parse_ics(raw, start, end, tz=self.tz)
                ok_any = True
            except Exception as exc:  # noqa: BLE001 - never log the URL
                errors.append(f"{sid}: {type(exc).__name__}")
                log.warning("calendar %s failed: %s", sid, _redact(str(exc), url))
        if self.icloud is not None:
            try:
                fresh["icloud"] = self._fetch_icloud(start, end)
                ok_any = True
            except Exception as exc:  # noqa: BLE001 - never log the password
                errors.append(f"icloud: {type(exc).__name__}")
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
                self._sources[sid] = {"fetched_at": stamp, "events": events}
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
        triggers a background refresh."""
        stale = self.is_stale()
        if stale:
            self.trigger_refresh()
        return Snapshot(events=self.events(), fetched_at=self.fetched_at,
                        stale=stale, errors=list(self.errors))


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


def make_tools(cfg, services) -> list[ToolSpec]:
    source = make_source(cfg, services)

    def get_calendar(range="today", **_) -> ToolResult:
        rng = coerce_range(range)
        if not source.configured:
            line = setup_line(cfg, "google_ical")
            return ToolResult(text=line, ok=False, speak=line)
        now = now_local(source.tz)
        snap = source.get(rng, now)
        if snap.fetched_at is None:
            if snap.errors:
                return ToolResult(text="calendar unreachable", ok=False)
            return ToolResult(text="calendar still loading, ask again in a moment",
                              ok=False)
        text = format_events(snap.events, rng, now)
        if snap.stale:
            text += f" That's as of {as_of_words(snap.fetched_at, now)}."
        return ToolResult(text=text)

    return [ToolSpec(
        name="get_calendar",
        description=("Hunter's calendar for a weekday name, today, tomorrow, "
                     "week, or next for only the next event."),
        parameters={"type": "object", "properties": {
            "range": {"type": "string", "enum": list(RANGES),
                      "description": "today, tomorrow, week or next"}},
            "required": ["range"]},
        handler=get_calendar)]
