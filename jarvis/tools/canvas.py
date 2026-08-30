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
from typing import Callable, Optional

from jarvis.logs import get_logger
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
NOTHING_DUE_LINE = "Nothing due this week, sir."
NO_GRADES_LINE = "No grades posted yet, sir."
NO_ANNOUNCEMENTS_LINE = "No announcements in the last {span}, sir."
PERSONA_LINES = [SETUP_LINE, BAD_TOKEN_LINE, UNREACHABLE_LINE, NOTHING_DUE_LINE,
                 NO_GRADES_LINE]

# Planner item kinds that are "due": class sessions (calendar_event), notes
# and pages are not homework.
DUE_TYPES = {"assignment", "quiz", "discussion_topic"}

# Template values docs / a first save leave behind (same shapes mail.py
# refuses): empty, "<paste here>", "your-token", "changeme", "xxxx".
_PLACEHOLDER = re.compile(r"^\s*$|^<.*>$|placeholder|^(paste|your|my)[-_ ]|"
                          r"^x{3,}$|change[ -_]?me", re.I)


class CanvasError(RuntimeError):
    """kind: setup | auth | unreachable | api (HTTP status other than 401)."""

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


def get_json(settings: dict, path_or_url: str, fetch: Fetch, budget: _Budget,
             params: Optional[list[tuple[str, str]]] = None) -> tuple[object, Optional[str]]:
    """One GET -> (parsed JSON, next-page url). Raises CanvasError."""
    url = path_or_url if path_or_url.startswith("http") else settings["base_url"] + path_or_url
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
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
        courses.append({"id": int(c["id"]), "name": name,
                        "code": str(c.get("course_code") or ""),
                        "student": student, "score": score, "grade": grade})
    log.info("canvas: %d active courses at %s", len(courses), _host(key))
    with _CACHE_LOCK:
        _COURSES[key] = (now, courses)
    return list(courses)


_LEADING_CODE = re.compile(r"^[A-Z]{2,4}[\s\-_]*\d{3}[A-Z]?(?:[\s\-_]*\d{3})?[\s\-_:.]*",
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
    local = dt.astimezone(now.tzinfo) if now.tzinfo else dt
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
    except (TypeError, ValueError):
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
    return ToolResult(text="Canvas unreachable", ok=False, speak=UNREACHABLE_LINE)


def make_tools(cfg, services) -> list[ToolSpec]:
    def _settings():
        return canvas_settings(cfg)

    def canvas_due(days=DEFAULT_DUE_DAYS, **_) -> ToolResult:
        days = _days_arg(days, DEFAULT_DUE_DAYS)
        settings = _settings()
        if settings is None:
            return ToolResult(text=SETUP_LINE, ok=False, speak=SETUP_LINE)
        now = _now()
        try:
            items = fetch_due(settings, days, _fetch, now)
        except CanvasError as exc:
            return _excuse(exc)
        if not items:
            line = NOTHING_DUE_LINE if days == 7 else \
                f"Nothing due in the next {_span_words(days)}, sir."
            return ToolResult(text=f"nothing due in the next {days} days", speak=line)
        return ToolResult(text=due_sheet(items, days, now), max_sentences=MAX_SENTENCES)

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
