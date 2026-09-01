"""Canvas LMS tools: what is due, current grades, recent announcements.

Read-only against the Canvas REST API v1 (Instructure; the user's is
``canvas.tamu.edu``) with a personal access token the user mints under
Account > Settings > New Access Token. Stdlib ``urllib`` only, every
request through the ONE seam ``_fetch`` (tests replace it) and every
request carrying a timeout; a whole tool call is capped by ``BUDGET_S``
so a slow Canvas cannot stall the tool loop.

Endpoint choices, fewest calls first:

* ``canvas_due``: ONE call to ``/api/v1/planner/items`` bounded by
  start/end date -- it spans every course, so it replaces the courses
  call plus one ``/assignments?bucket=upcoming`` per course. Only when
  the planner answers 403/404 (accounts with the planner switched off)
  does it fall back to that per-course walk.
* ``canvas_grades``: the course list with ``include[]=total_scores``;
  the scores ride on the student enrolment of each course.
* ``canvas_announcements``: ``/api/v1/announcements`` for every active
  course's context code.

The course list is cached ten minutes per base URL because grades and
announcements both need it and a course roster changes once a term.

The token is a secret: it is never logged, never in a URL, only in the
Authorization header; log lines carry the host and counts.

A university that blocks personal access tokens (his does) leaves this
whole path dark, so "what is due" has a SECOND source: the Canvas calendar
feed, adapted to these same rows by ``jarvis/tools/canvas_ical.py``.
``read_due`` merges the two and says which existed, and every caller that
would otherwise ask for a token goes through it -- ``SETUP_LINE`` is spoken
only when neither source has anything at all. Grades and announcements
have no feed equivalent and still need the token.
"""
from __future__ import annotations

import html as _html
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable, NamedTuple, Optional

from jarvis.logs import get_logger
from jarvis.tools import canvas_ical
from jarvis.tools.location import clock_words
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.canvas")

DEFAULT_BASE_URL = "https://canvas.tamu.edu"
REQUEST_TIMEOUT = 5.0          # per request; the budget below trims it further
BUDGET_S = 6.0                 # wall-clock cap for one tool call, all pages
MAX_PAGES = 5                  # Link-header pagination cap (50 items a page)
PER_PAGE = 50
COURSE_CACHE_S = 10 * 60
MAX_DAYS = 30
DEFAULT_DUE_DAYS = 7
DEFAULT_ANNOUNCE_DAYS = 3
DUE_ITEMS = 12                 # fact-sheet lines; the model picks from these
ANNOUNCE_ITEMS = 6
SNIPPET_CHARS = 120
USER_AGENT = "Jarvis/3 (+https://github.com/hunterp/Jarvis)"
MAX_SENTENCES = 4

# Persona lines (fixed so the speech cache can prewarm them).
SETUP_LINE = ("I'll need a Canvas access token set up, sir; the notes are in "
              "docs/assistant-setup.md.")
BAD_TOKEN_LINE = "Canvas rejected the token, sir."
UNREACHABLE_LINE = "I can't reach Canvas, sir."
INSECURE_LINE = "Canvas is set to plain http, sir; I won't send the token in the clear."
NOTHING_DUE_LINE = "Nothing due this week, sir."
NO_GRADES_LINE = "No grades posted yet, sir."
NO_ANNOUNCEMENTS_LINE = "No announcements in the last {span}, sir."
PERSONA_LINES = [SETUP_LINE, BAD_TOKEN_LINE, UNREACHABLE_LINE, NOTHING_DUE_LINE,
                 NO_GRADES_LINE, INSECURE_LINE]

# Planner item kinds that are "due": class sessions (calendar_event), notes
# and pages are not homework.
DUE_TYPES = {"assignment", "quiz", "discussion_topic"}

# Template values docs / a first save leave behind (same shapes mail.py
# refuses): empty, "<paste here>", "your-token", "changeme", "xxxx".
_PLACEHOLDER = re.compile(r"^\s*$|^<.*>$|placeholder|^(paste|your|my)[-_ ]|"
                          r"^x{3,}$|change[ -_]?me", re.I)


class CanvasError(RuntimeError):
    """kind: setup | auth | unreachable | insecure | api (HTTP status other
    than 401)."""

    def __init__(self, kind: str, detail: str = "", status: int = 0):
        super().__init__(detail or kind)
        self.kind = kind
        self.detail = detail
        self.status = status


# ------------------------------------------------------------- config
def _cfg_get(cfg, dotted: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        cur = cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return default if cur is None else cur
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            val = get(dotted, default)
            return default if val is None else val
        except Exception:
            log.debug("cfg.get(%s) failed", dotted, exc_info=True)
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def _is_placeholder(value) -> bool:
    return not isinstance(value, str) or bool(_PLACEHOLDER.search(value))


def canvas_settings(cfg) -> Optional[dict]:
    """{base_url, token} or None when the token is missing / a placeholder.

    Deliberately NOT ``cfg.is_configured("canvas")``: AssistantConfig
    answers False for any section it does not know, which would keep this
    tool asking for a token that is already in the file."""
    token = _cfg_get(cfg, "canvas.token", "")
    if _is_placeholder(token):
        return None
    base = str(_cfg_get(cfg, "canvas.base_url", "") or DEFAULT_BASE_URL).strip()
    base = base.rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "https://" + base
    return {"base_url": base, "token": token.strip()}


# ---------------------------------------------------------------- http
def _fetch(url: str, headers: dict, timeout: float) -> tuple[int, dict, bytes]:
    """The ONE network seam: GET -> (status, lower-cased headers, body).
    HTTP errors come back as their status (the caller reads 401 / 404);
    transport errors raise."""
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (int(resp.status or 200),
                    {k.lower(): v for k, v in resp.headers.items()},
                    resp.read())
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:          # noqa: BLE001 - body is optional here
            body = b""
        return int(exc.code), {k.lower(): v for k, v in (exc.headers or {}).items()}, body


Fetch = Callable[[str, dict, float], tuple[int, dict, bytes]]
_clock = time.time             # cache clock (test seam)
_monotonic = time.monotonic    # budget clock (test seam)


class _Budget:
    """Wall-clock cap for one tool call: each request gets what is left,
    never more than REQUEST_TIMEOUT."""

    def __init__(self, seconds: float = BUDGET_S):
        self.deadline = _monotonic() + seconds

    def remaining(self) -> float:
        return self.deadline - _monotonic()

    def timeout(self) -> float:
        rem = self.remaining()
        if rem <= 0.1:
            raise CanvasError("unreachable", "time budget exhausted")
        return min(REQUEST_TIMEOUT, rem)


def _next_link(headers: dict) -> Optional[str]:
    """rel="next" from a Canvas Link header, or None on the last page."""
    for url, rel in re.findall(r'<([^>]+)>\s*;\s*rel="([^"]+)"', headers.get("link", "") or ""):
        if rel == "next":
            return url
    return None


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc or url[:40]


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
_WARNED_INSECURE: set = set()


def _insecure(url: str) -> bool:
    """True for plain http to anything but this machine: the bearer token
    rides in a header, and canvas_settings accepts an http:// base_url
    verbatim, so without this a typo in assistant.json would put the token
    on the wire in cleartext on every call."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http":
        return False
    return (parts.hostname or "").lower() not in _LOCAL_HOSTS


def get_json(settings: dict, path_or_url: str, fetch: Fetch, budget: _Budget,
             params: Optional[list[tuple[str, str]]] = None) -> tuple[object, Optional[str]]:
    """One GET -> (parsed JSON, next-page url). Raises CanvasError."""
    url = path_or_url if path_or_url.startswith("http") else settings["base_url"] + path_or_url
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    if _insecure(url):
        host = _host(url)
        if host not in _WARNED_INSECURE:            # once per host, not per call
            _WARNED_INSECURE.add(host)
            log.warning("canvas: refusing to send the token over plain http to %s", host)
        raise CanvasError("insecure", host)
    headers = {"Authorization": f"Bearer {settings['token']}",
               "Accept": "application/json", "User-Agent": USER_AGENT}
    try:
        status, resp_headers, body = fetch(url, headers, budget.timeout())
    except CanvasError:
        raise
    except Exception as exc:                  # noqa: BLE001 - transport boundary
        log.warning("canvas: %s unreachable (%s)", _host(url), type(exc).__name__)
        raise CanvasError("unreachable", type(exc).__name__) from exc
    if status == 401:
        log.warning("canvas: %s rejected the token", _host(url))
        raise CanvasError("auth", "401", status)
    if status >= 400:
        log.warning("canvas: HTTP %s from %s", status, url.split("?")[0][:80])
        raise CanvasError("api", f"HTTP {status}", status)
    try:
        data = json.loads(body or b"null")
    except ValueError as exc:
        raise CanvasError("api", "bad JSON") from exc
    return data, _next_link(resp_headers)


def get_all(settings: dict, path: str, fetch: Fetch, budget: _Budget,
            params: Optional[list[tuple[str, str]]] = None) -> list:
    """Every page of a list endpoint, up to MAX_PAGES / the budget. A page
    that would overrun the budget is dropped rather than failing the call:
    a partial list beats an excuse."""
    items: list = []
    url: Optional[str] = path
    for page in range(MAX_PAGES):
        if page and budget.remaining() < 0.5:
            log.info("canvas: budget spent after %d pages", page)
            break
        data, url = get_json(settings, url, fetch, budget, params if page == 0 else None)
        if isinstance(data, list):
            items.extend(data)
        if not url:
            break
    return items


# ------------------------------------------------------------- courses
_CACHE_LOCK = threading.Lock()
_COURSES: dict[str, tuple[float, list]] = {}


def clear_cache() -> None:
    with _CACHE_LOCK:
        _COURSES.clear()


def active_courses(settings: dict, fetch: Fetch, budget: _Budget) -> list[dict]:
    """[{id, name, code, score, grade}] for every active enrolment, cached
    COURSE_CACHE_S per base URL. ``score``/``grade`` come from the student
    enrolment's total_scores (None when nothing is posted yet)."""
    key = settings["base_url"]
    now = _clock()
    with _CACHE_LOCK:
        hit = _COURSES.get(key)
        if hit and now - hit[0] < COURSE_CACHE_S:
            return list(hit[1])
    raw = get_all(settings, "/api/v1/courses", fetch, budget,
                  [("enrollment_state", "active"), ("include[]", "total_scores"),
                   ("per_page", str(PER_PAGE))])
    courses = []
    for c in raw:
        if not isinstance(c, dict) or c.get("access_restricted_by_date"):
            continue
        name = " ".join(str(c.get("name") or "").split())
        if not name or c.get("id") is None:
            continue
        score = grade = None
        student = False
        enrollments = c.get("enrollments")
        for enr in enrollments if isinstance(enrollments, list) else []:
            if not isinstance(enr, dict) or enr.get("type") not in ("student", "StudentEnrollment"):
                continue
            student = True
            score = enr.get("computed_current_score")
            grade = enr.get("computed_current_grade")
            break
        # Non-student enrolments (a TA post, an observed course) stay in the
        # roster for announcements but carry no grade and no homework.
        try:
            cid = int(c["id"])
        except (TypeError, ValueError):
            continue
        courses.append({"id": cid, "name": name,
                        "code": str(c.get("course_code") or ""),
                        "student": student, "score": score, "grade": grade})
    log.info("canvas: %d active courses at %s", len(courses), _host(key))
    with _CACHE_LOCK:
        _COURSES[key] = (now, courses)
    return list(courses)


def cached_course_names() -> list:
    """Course names already in the module cache, tidied for the Whisper
    prompt (jarvis/vocab.py). Read-only by design: the transcriber asks
    on every turn and must NEVER trigger a network fetch, so an empty
    list before the first Canvas tool call is the correct price -- and
    expiry is ignored on purpose, because a stale course name still
    biases recognition the right way."""
    with _CACHE_LOCK:
        courses = [c for _, cached in _COURSES.values() for c in cached]
    names, seen = [], set()
    for c in courses:
        name = tidy_course(str(c.get("name") or "")) if isinstance(c, dict) else ""
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


_LEADING_CODE = re.compile(r"^[A-Z]{2,4}[\s\-_]*\d{3,4}[A-Z]?(?:[\s\-_]*\d{3,4})?[\s\-_:.]*",
                           re.I)
_TRAILING_TERM = re.compile(r"[\s\-_,(]*(?:(?:FA|SP|SU|WI)\s?\d{2}|(?:FALL|SPRING|SUMMER|WINTER)"
                            r"\s*\d{2,4}|\d{4,6})\)?[\s\-_]*$", re.I)


def tidy_course(name: str) -> str:
    """'BMEN 420 500 BIOSENSORS FA26' -> 'BIOSENSORS': the spoken form drops
    the catalogue number and the term, which Canvas glues onto every title.
    Anything that would strip to nothing keeps the original."""
    text = " ".join(str(name or "").split())
    out = _TRAILING_TERM.sub("", text)
    out = _LEADING_CODE.sub("", out)
    out = out.strip(" -_:,")
    if len(out) < 3:
        out = text
    return out[:48]


# --------------------------------------------------------------- dates
def _parse_iso(text) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def when_words(dt: datetime, now: datetime) -> str:
    """today 11:59 pm / tomorrow 11:59 pm / Tue 11:59 pm / Tue 15 Sep 11:59 pm,
    in ``now``'s timezone (the local one at call time)."""
    # datetime.now().astimezone() hands back a FIXED offset; a due date on
    # the other side of a DST change (inside every 30-day window in March
    # and November) would render an hour off and often a day off. A bare
    # astimezone() follows the system zone through the change.
    if now.tzinfo is None:
        local = dt
    elif isinstance(now.tzinfo, timezone):
        local = dt.astimezone()
    else:
        local = dt.astimezone(now.tzinfo)
    today = now.date()
    day = local.date()
    clock = clock_words(local)
    if day == today:
        return f"today {clock}"
    if day == today + timedelta(days=1):
        return f"tomorrow {clock}"
    if day == today - timedelta(days=1):
        return f"yesterday {clock}"
    if 0 < (day - today).days < 7:
        return f"{local.strftime('%a')} {clock}"
    return f"{local.strftime('%a')} {local.day} {local.strftime('%b')} {clock}"


def _now() -> datetime:
    return datetime.now().astimezone()


def _days_arg(value, default: int) -> int:
    try:
        return max(1, min(MAX_DAYS, int(float(str(value)))))
    except (TypeError, ValueError, OverflowError):   # int(float("inf"))
        return default


def _span_words(days: int) -> str:
    return "day" if days == 1 else "week" if days == 7 else f"{days} days"


# ----------------------------------------------------------------- due
def _submitted(item: dict) -> bool:
    sub = item.get("submissions")
    if isinstance(sub, dict) and (sub.get("submitted") or sub.get("graded")):
        return True
    override = item.get("planner_override")
    return bool(isinstance(override, dict) and override.get("marked_complete"))


def _planner_due(settings: dict, fetch: Fetch, budget: _Budget,
                 now: datetime, days: int) -> list[dict]:
    start = now.astimezone(timezone.utc)
    end = start + timedelta(days=days)
    raw = get_all(settings, "/api/v1/planner/items", fetch, budget,
                  [("start_date", start.strftime("%Y-%m-%dT%H:%M:%SZ")),
                   ("end_date", end.strftime("%Y-%m-%dT%H:%M:%SZ")),
                   ("per_page", str(PER_PAGE))])
    out = []
    for it in raw:
        if not isinstance(it, dict) or it.get("plannable_type") not in DUE_TYPES:
            continue
        if _submitted(it):
            continue
        plannable = it.get("plannable") if isinstance(it.get("plannable"), dict) else {}
        due = _parse_iso(plannable.get("due_at") or it.get("plannable_date"))
        title = " ".join(str(plannable.get("title") or "").split())
        # The window is re-applied here: the server bounds are a request,
        # and an item outside them would be read out as due this week.
        if due is None or not title or due < start or due > end:
            continue
        out.append({"course": tidy_course(it.get("context_name") or ""),
                    "title": title, "due": due})
    return out


def _course_due(settings: dict, fetch: Fetch, budget: _Budget,
                now: datetime, days: int) -> list[dict]:
    """Fallback: bucket=upcoming per active course (N+1 calls)."""
    end = now + timedelta(days=days)
    out = []
    for course in active_courses(settings, fetch, budget):
        if not course["student"]:
            continue
        if budget.remaining() < 0.5:
            log.info("canvas: budget spent walking courses")
            break
        try:
            raw = get_all(settings, f"/api/v1/courses/{course['id']}/assignments",
                          fetch, budget, [("bucket", "upcoming"), ("per_page", str(PER_PAGE)),
                                          ("order_by", "due_at")])
        except CanvasError as exc:
            # One course refusing (a concluded section answers 403) must
            # not blank the rest; an outage or a bad token still does.
            if exc.kind != "api":
                raise
            continue
        for a in raw:
            if not isinstance(a, dict):
                continue
            due = _parse_iso(a.get("due_at"))
            title = " ".join(str(a.get("name") or "").split())
            if due is None or not title or due < now or due > end:
                continue
            out.append({"course": tidy_course(course["name"]), "title": title, "due": due})
    return out


def fetch_due(settings: dict, days: int, fetch: Fetch = None,
              now: Optional[datetime] = None) -> list[dict]:
    """Outstanding work due within ``days``, soonest first. Raises
    CanvasError (setup errors are the caller's; see canvas_settings)."""
    fetch = fetch or _fetch
    now = now or _now()
    budget = _Budget()
    try:
        items = _planner_due(settings, fetch, budget, now, days)
    except CanvasError as exc:
        if exc.kind != "api" or exc.status not in (403, 404):
            raise
        log.info("canvas: planner unavailable (%s); walking courses", exc.status)
        items = _course_due(settings, fetch, budget, now, days)
    items.sort(key=lambda i: i["due"])
    return items


class DueRead(NamedTuple):
    """What one "what's due" reading found, and from where.

    ``canvas`` / ``feed`` are the two SOURCES: a caller says "I'll need a
    Canvas access token set up" only when neither of them exists, because a
    box whose university blocks tokens still has the coursework -- it is in
    the calendar feed (jarvis/tools/canvas_ical.py)."""
    items: list                      # merged rows, soonest first
    canvas: bool                     # the REST API was read successfully
    feed: bool                       # the calendar feed carries coursework
    error: Optional[CanvasError]     # why the REST read failed, if it did


def feed_due(calendar, days: Optional[int] = None,
             now: Optional[datetime] = None) -> list[dict]:
    """fetch_due-shaped rows from the Canvas calendar feed. Never fetches:
    the CalendarSource's cached events are the input."""
    return canvas_ical.rows_from_events(_calendar_events(calendar), days, now)


def read_due(cfg, days: int, calendar=None, now: Optional[datetime] = None,
             fetch: Fetch = None) -> DueRead:
    """The one "what is due" reading: Canvas REST when a token is set,
    merged with the calendar feed's coursework, in ONE shape.

    The REST rows go first into the merge because they know the course's
    real name and whether the work was handed in; a feed row is only
    dropped when the REST reading already has that title on that day."""
    now = now or _now()
    events = _calendar_events(calendar)
    feed_items = canvas_ical.rows_from_events(events, days, now)
    feed = bool(feed_items) or canvas_ical.has_coursework(events)
    settings = canvas_settings(cfg)
    rest: list[dict] = []
    ok = False
    err: Optional[CanvasError] = None
    if settings is not None:
        try:
            rest = fetch_due(settings, days, fetch or _fetch, now)
            ok = True
        except CanvasError as exc:
            err = exc
            log.info("canvas: REST due unavailable (%s); the feed has %d row(s)",
                     exc.kind, len(feed_items))
    return DueRead(canvas_ical.merge_rows(rest, feed_items), ok, feed, err)


def due_sheet(items: list[dict], days: int, now: datetime) -> str:
    span = _span_words(days)
    head = f"Due in the next {span} ({len(items)}):" if days != 7 else \
        f"Due this week ({len(items)}):"
    lines = [head]
    for i, it in enumerate(items[:DUE_ITEMS], 1):
        lines.append(f"{i}) {it['course']} - {it['title']}, {when_words(it['due'], now)}")
    if len(items) > DUE_ITEMS:
        lines.append(f"and {len(items) - DUE_ITEMS} more")
    return "\n".join(lines)


# --------------------------------------------------------------- exams
# "Exam-ish" titles across Canvas items and calendar events. Quizzes are
# their own kind: "when's my next exam" must not answer with Quiz 3, but
# "how long until the quiz" must find it. "test" is deliberately absent --
# "Unit test lab", "Test your knowledge" and "COVID test" all carry it.
_EXAM_KINDS = {"exam": ("exam", "exams", "midterm", "midterms", "final", "finals"),
               "quiz": ("quiz", "quizzes")}
_EXAM_WORDS = {w: k for k, words in _EXAM_KINDS.items() for w in words}
_SPECIFIC = {"midterm": "midterm", "midterms": "midterm", "final": "final", "finals": "final"}
# Whisper writes what he says, apostrophe and all. LIVE 2026-08-31 21:00:
# "How many days until my biosensor's midterm?" -- the needle came out as
# "biosensor's", which is not a substring of "ECEN 414 BIOSENSORS", so
# next_exam matched nothing while "biosensors midterm" matched fine. The
# possessive and a trailing plural both have to come off before the
# substring test, and off the kind words too ("my finals'").
_POSSESSIVE_RX = re.compile(r"(?:'s|s'|')$")


def _bare(word: str) -> str:
    """A query word without its possessive: "biosensor's" -> "biosensor"."""
    return _POSSESSIVE_RX.sub("", word)


def _stem(word: str) -> str:
    """``_bare`` plus a trailing plural, for substring matching only:
    "biosensors" -> "biosensor", which is still inside "BIOSENSORS"."""
    w = _bare(word)
    return w[:-1] if len(w) > 3 and w.endswith("s") else w
_WORD_RX = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# Question filler that must not become a title filter ("how long until
# the biosensors midterm" -> just "biosensors").
_QUERY_STOP = {"when", "whens", "when's", "is", "was", "are", "my", "the", "a", "an",
               "next", "how", "long", "many", "days", "until", "till", "before",
               "to", "do", "i", "have", "it", "in", "of", "for", "sir", "jarvis", "my"}
NO_EXAM_LINE = "Nothing that looks like an exam on the books, sir."
NO_QUIZ_LINE = "No quiz on the books, sir."
EXAM_LOOKAHEAD_DAYS = MAX_DAYS       # one planner call; 30 days is Canvas's cap
EXAM_EVE_HOUR = 19                   # the evening-before heads-up fires at 7 pm


def exam_kind(title: str) -> Optional[str]:
    """'exam' / 'quiz' when the title names one, else None."""
    for w in _WORD_RX.findall(str(title or "").lower()):
        kind = _EXAM_WORDS.get(w)
        if kind:
            return kind
    return None


def exam_candidates(items: list, events: list, now: datetime) -> list[dict]:
    """[{course, title, when, kind, all_day, source}] soonest first, from
    Canvas due items ({course, title, due}) and calendar events (.title,
    .start, .all_day, .calendar). The same exam in both (the iCloud
    'Canvas' subscription mirrors Canvas) keeps the Canvas copy: it carries
    the course name; a calendar copy is kept only when no Canvas item shares
    its title on that day."""
    out: list[dict] = []
    seen: set = set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        title = " ".join(str(it.get("title") or "").split())
        when = it.get("due")
        kind = exam_kind(title)
        if not kind or not isinstance(when, datetime) or when <= now:
            continue
        seen.add((title.lower(), when.astimezone(now.tzinfo).date()))
        out.append({"course": str(it.get("course") or ""), "title": title, "when": when,
                    "kind": kind, "all_day": False, "source": "canvas"})
    for ev in events or []:
        title = " ".join(str(getattr(ev, "title", "") or "").split())
        when = getattr(ev, "start", None)
        kind = exam_kind(title)
        if not kind or not isinstance(when, datetime) or when.tzinfo is None:
            continue
        all_day = bool(getattr(ev, "all_day", False))
        # An all-day exam "starts" at midnight; the day is what matters.
        if (when <= now and not all_day) or (all_day and when.date() < now.astimezone(when.tzinfo).date()):
            continue
        day_key = (title.lower(), when.astimezone(now.tzinfo).date())
        if day_key in seen:
            continue
        seen.add(day_key)
        out.append({"course": "", "title": title, "when": when, "kind": kind,
                    "all_day": all_day, "source": "calendar"})
    out.sort(key=lambda c: c["when"])
    return out


def next_exam(items: list, events: list, now: datetime, query: str = "") -> Optional[dict]:
    """The soonest exam matching ``query`` ("next exam", "biosensors
    midterm", "quiz"), or None. Kind words in the query pick the kind
    (exam words never match a quiz and vice versa; no kind word -> exams
    only); every other content word must appear in the course or title."""
    words = [_bare(w) for w in _WORD_RX.findall(str(query or "").lower())]
    kinds = {_EXAM_WORDS[w] for w in words if w in _EXAM_WORDS} or {"exam"}
    needles = [_stem(w) for w in words
               if w not in _EXAM_WORDS and w not in _QUERY_STOP and len(w) > 1]
    # "final" and "midterm" name a particular exam: "when's my next final"
    # must not answer with the midterm. "exam" alone is generic.
    specific = {_SPECIFIC[w] for w in words if w in _SPECIFIC}
    for cand in exam_candidates(items, events, now):
        if cand["kind"] not in kinds:
            continue
        hay = f"{cand['course']} {cand['title']}".lower()
        if specific and not any(s in cand["title"].lower() for s in specific):
            continue
        if all(n in hay for n in needles):
            return cand
    return None


def countdown_words(when: datetime, now: datetime, all_day: bool = False) -> str:
    """'today at 2:00 pm' / 'tomorrow at 9:00 am' / 'in 6 days, Friday at
    9:00 am' / 'in 19 days, Wed 15 Oct at 9:00 am', in ``now``'s zone."""
    local = when.astimezone(now.tzinfo) if now.tzinfo else when
    days = (local.date() - now.date()).days
    clock = "" if all_day else f" at {clock_words(local)}"
    if days <= 0:
        return f"today{clock}"
    if days == 1:
        return f"tomorrow{clock}"
    if days < 7:
        return f"in {days} days, {local.strftime('%A')}{clock}"
    return f"in {days} days, {local.strftime('%a')} {local.day} {local.strftime('%b')}{clock}"


def exam_words(cand: dict, now: datetime) -> str:
    """'Midterm 1 for BIOSENSORS, in 6 days, Friday at 9:00 am' -- the
    countdown fragment the briefing and the Tier-1 answer both speak."""
    what = cand["title"]
    if cand.get("course"):
        what += f" for {cand['course']}"
    return f"{what}, {countdown_words(cand['when'], now, cand.get('all_day', False))}"


def _calendar_events(calendar) -> list:
    """Events from a CalendarSource-like object; [] when absent, unconfigured
    or broken (the calendar half must never take the Canvas half down)."""
    if calendar is None:
        return []
    try:
        conf = getattr(calendar, "configured", True)
        if callable(conf):
            conf = conf()
        if not conf:
            return []
        return list(calendar.events())
    except Exception:                          # noqa: BLE001 - source boundary
        log.debug("canvas: calendar events unavailable", exc_info=True)
        return []


def find_next_exam(cfg, calendar=None, query: str = "", now: Optional[datetime] = None,
                   fetch: Fetch = None) -> tuple[Optional[dict], bool]:
    """(exam or None, canvas_checked). Canvas items (when the token is set;
    a CanvasError is swallowed and logged -- the calendar half still
    answers) merged with the coursework in the calendar FEED and with the
    calendar's own events. ``canvas_checked`` is False only when NEITHER
    Canvas source exists, so a caller can fall back to the canvas_due tool
    turn, whose setup line explains what is missing."""
    now = now or _now()
    settings = canvas_settings(cfg)
    items: list = []
    checked = settings is not None
    if checked:
        try:
            items = fetch_due(settings, EXAM_LOOKAHEAD_DAYS, fetch or _fetch, now)
        except CanvasError as exc:
            log.info("canvas: exam lookup skipped Canvas (%s)", exc.kind)
    # The feed's coursework, over a TERM rather than the calendar cache's
    # fortnight -- a midterm is nearly always further out than 14 days, so
    # reading only the cached events would answer "nothing on the books"
    # about an exam that is plainly in the feed. deep_rows re-parses bytes
    # the calendar already fetched; [] falls back to the cached fortnight.
    events = _calendar_events(calendar)
    feed_items = canvas_ical.deep_rows(calendar, now=now) or \
        canvas_ical.rows_from_events(events, EXAM_LOOKAHEAD_DAYS, now)
    if feed_items or canvas_ical.has_coursework(events):
        checked = True
    items = canvas_ical.merge_rows(items, feed_items)
    # Coursework now arrives as ITEMS, with a clean title and a course; the
    # raw VEVENT it came from must not also arrive as an event or every
    # assignment would be a candidate twice, once with its bracket of
    # section numbers still attached.
    events = canvas_ical.other_events(events)
    # The third source: exams accepted from a syllabus scan (jarvis/syllabus.py).
    # It has to be merged HERE and not only in deadlines.tick, or he would
    # call an exam eve for a midterm and then deny having one when asked --
    # the reminder and the answer must come from the same set. Imported
    # lazily because syllabus.py reads the documents index, which this
    # module has no business pulling in on a grades lookup.
    try:
        from jarvis import syllabus as syllabus_mod
        items = syllabus_mod.merge_items(items, syllabus_mod.stored_items(now))
    except Exception:                          # noqa: BLE001 - source boundary
        log.debug("canvas: syllabus items unavailable", exc_info=True)
    return next_exam(items, events, now, query), checked


# -------------------------------------------------------------- grades
def _score_words(course: dict) -> str:
    score, grade = course.get("score"), course.get("grade")
    if score is None and not grade:
        return "no score yet"
    parts = []
    if score is not None:
        try:
            parts.append(f"{float(score):.1f}%".replace(".0%", "%"))
        except (TypeError, ValueError):
            pass
    if grade:
        parts.append(f"({grade})" if parts else str(grade))
    return " ".join(parts) or "no score yet"


def grades_sheet(courses: list[dict]) -> str:
    courses = [c for c in courses if c.get("student")]
    lines = [f"Current grades ({len(courses)} courses):"]
    for i, c in enumerate(courses, 1):
        lines.append(f"{i}) {tidy_course(c['name'])} {_score_words(c)}")
    return "\n".join(lines)


# ------------------------------------------------------- announcements
_TAG = re.compile(r"<[^>]+>")
_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)


def snippet(raw_html: str, limit: int = SNIPPET_CHARS) -> str:
    text = _STYLE.sub(" ", str(raw_html or ""))
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", " ", text, flags=re.I)
    flat = " ".join(_html.unescape(_TAG.sub(" ", text)).split())
    if len(flat) > limit:
        flat = flat[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return flat


def fetch_announcements(settings: dict, days: int, fetch: Fetch = None,
                        now: Optional[datetime] = None) -> list[dict]:
    """Announcements across active courses posted within ``days``, newest
    first. Raises CanvasError."""
    fetch = fetch or _fetch
    now = now or _now()
    budget = _Budget()
    courses = active_courses(settings, fetch, budget)
    if not courses:
        return []
    names = {c["id"]: tidy_course(c["name"]) for c in courses}
    params = [("context_codes[]", f"course_{c['id']}") for c in courses]
    since = now - timedelta(days=days)
    params += [("start_date", since.strftime("%Y-%m-%d")),
               ("end_date", (now + timedelta(days=1)).strftime("%Y-%m-%d")),
               ("per_page", str(PER_PAGE))]
    raw = get_all(settings, "/api/v1/announcements", fetch, budget, params)
    out = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        posted = _parse_iso(a.get("posted_at")) or _parse_iso(a.get("delayed_post_at"))
        title = " ".join(str(a.get("title") or "").split())
        if not title or (posted is not None and posted < since):
            continue
        try:
            cid = int(str(a.get("context_code") or "").rpartition("_")[2])
        except ValueError:
            cid = None
        out.append({"course": names.get(cid, "Canvas"), "title": title,
                    "posted": posted, "snippet": snippet(a.get("message"))})
    out.sort(key=lambda a: a["posted"] or since, reverse=True)
    return out


def announcements_sheet(items: list[dict], days: int, now: datetime) -> str:
    lines = [f"Announcements in the last {_span_words(days)} ({len(items)}):"]
    for i, a in enumerate(items[:ANNOUNCE_ITEMS], 1):
        line = f"{i}) {a['course']} - {a['title']}"
        if a["posted"] is not None:
            line += f" ({when_words(a['posted'], now)})"
        if a["snippet"]:
            line += f": {a['snippet']}"
        lines.append(line)
    if len(items) > ANNOUNCE_ITEMS:
        lines.append(f"and {len(items) - ANNOUNCE_ITEMS} more")
    return "\n".join(lines)


# ---------------------------------------------------------------- tool
def _excuse(exc: CanvasError) -> ToolResult:
    if exc.kind == "auth":
        return ToolResult(text="Canvas rejected the access token (401)", ok=False,
                          speak=BAD_TOKEN_LINE)
    if exc.kind == "api":
        return ToolResult(text=f"Canvas answered {exc.detail}", ok=False,
                          speak=UNREACHABLE_LINE)
    if exc.kind == "insecure":
        return ToolResult(text=f"Canvas base_url is plain http ({exc.detail}); "
                               "token not sent", ok=False, speak=INSECURE_LINE)
    return ToolResult(text="Canvas unreachable", ok=False, speak=UNREACHABLE_LINE)


def make_tools(cfg, services) -> list[ToolSpec]:
    def _settings():
        return canvas_settings(cfg)

    def _calendar():
        # Resolved at CALL time, not here: calendar.make_tools parks the
        # CalendarSource on services during the same boot loop and the
        # order of TOOL_MODULES is not this module's business.
        return getattr(services, "calendar", None) if services is not None else None

    def canvas_due(days=DEFAULT_DUE_DAYS, **_) -> ToolResult:
        days = _days_arg(days, DEFAULT_DUE_DAYS)
        now = _now()
        read = read_due(cfg, days, _calendar(), now)
        if read.items:
            return ToolResult(text=due_sheet(read.items, days, now),
                              max_sentences=MAX_SENTENCES)
        if read.error is not None and not read.feed:
            return _excuse(read.error)
        if read.canvas or read.feed:
            # A source was read and it is empty: that is an answer.
            line = NOTHING_DUE_LINE if days == 7 else \
                f"Nothing due in the next {_span_words(days)}, sir."
            return ToolResult(text=f"nothing due in the next {days} days", speak=line)
        # Neither Canvas nor a coursework feed exists -- only NOW is the
        # token worth mentioning.
        return ToolResult(text=SETUP_LINE, ok=False, speak=SETUP_LINE)

    def canvas_grades(**_) -> ToolResult:
        settings = _settings()
        if settings is None:
            return ToolResult(text=SETUP_LINE, ok=False, speak=SETUP_LINE)
        try:
            courses = active_courses(settings, _fetch, _Budget())
        except CanvasError as exc:
            return _excuse(exc)
        if not any(c["student"] and (c["score"] is not None or c["grade"])
                   for c in courses):
            return ToolResult(text="no grades posted in any active course",
                              speak=NO_GRADES_LINE)
        return ToolResult(text=grades_sheet(courses), max_sentences=MAX_SENTENCES)

    def canvas_announcements(days=DEFAULT_ANNOUNCE_DAYS, **_) -> ToolResult:
        days = _days_arg(days, DEFAULT_ANNOUNCE_DAYS)
        settings = _settings()
        if settings is None:
            return ToolResult(text=SETUP_LINE, ok=False, speak=SETUP_LINE)
        now = _now()
        try:
            items = fetch_announcements(settings, days, _fetch, now)
        except CanvasError as exc:
            return _excuse(exc)
        if not items:
            span = _span_words(days)
            return ToolResult(text=f"no announcements in the last {days} days",
                              speak=NO_ANNOUNCEMENTS_LINE.format(span=span))
        return ToolResult(text=announcements_sheet(items, days, now),
                          max_sentences=MAX_SENTENCES)

    days_param = {"type": "integer",
                  "description": "how many days ahead to look (default 7, max 30)"}
    return [
        ToolSpec(
            name="canvas_due",
            # <= 20 words: the descriptions ride in every prompt.
            description=("Canvas coursework due soon: outstanding assignments, "
                         "quizzes and discussions for the next N days."),
            parameters={"type": "object", "properties": {"days": days_param}},
            handler=canvas_due),
        ToolSpec(
            name="canvas_grades",
            description=("Canvas grades: Hunter's current score and letter "
                         "grade in each active course."),
            parameters={"type": "object", "properties": {}},
            handler=canvas_grades),
        ToolSpec(
            name="canvas_announcements",
            description=("Recent Canvas course announcements from instructors, "
                         "for the last N days."),
            parameters={"type": "object", "properties": {"days": {
                "type": "integer",
                "description": "how many days back to look (default 3, max 30)"}}},
            handler=canvas_announcements),
    ]
