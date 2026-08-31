"""Canvas coursework read from the CALENDAR FEED, for a box with no token.

His university blocks personal Canvas access tokens, so the REST path in
``jarvis/tools/canvas.py`` can never answer here -- and until this module
existed, "what's due this week" and "when's my next exam" both replied
"I'll need a Canvas access token set up, sir" while the answer sat in the
calendar he had already subscribed to.

The same coursework rides the Canvas calendar subscription (an .ics in
``google_ical_urls``) as VEVENTs whose SUMMARY ends in a bracket of course
codes::

    HW#1 [MSEN-222:599,M99]
    Lab 1: Introduction to the AD2 SDK [BMEN-427:501,502,503,504,BMEN-627:...]
    Update Presentation #1 [ECEN-404:901,902,903]

That trailing bracket is the tell: it is what Canvas appends and nothing
else in his calendars carries it, so recognition never depends on which
feed an event arrived through or what the subscription happens to be
named.

Rows come out in EXACTLY the shape ``canvas.fetch_due`` returns --
``{"course": str, "title": str, "due": aware datetime}`` -- so canvas_due,
the briefing's Due section, jarvis/deadlines.py and canvas.find_next_exam
all keep consuming ONE shape and learn nothing new.

Two things the feed does not carry, and this module does not pretend to:

* **Submission state.** Canvas's planner marks handed-in work; an .ics
  cannot. A feed row is "outstanding" by assumption, so work already
  submitted stays on the list until its due time passes. The REST reading
  wins a duplicate for exactly this reason (see ``merge_rows``).
* **The course NAME.** The bracket carries the catalogue code, so a feed
  row says "BMEN 427" where the REST reading says "BIOSENSORS".

Horizon: an all-day VEVENT is the whole story for a date, but the calendar
cache only keeps a fortnight (``calendar.WINDOW_DAYS``) and a midterm is
further out than that.  ``deep_rows`` therefore re-parses the feed's RAW
ics -- the bytes ``CalendarSource`` already fetched, never a new request --
over a term-long window, memoised per response so a spoken turn pays for
the parse at most once, and mirrored into a small JSON cache so a
just-started process has an answer before the first refresh lands.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from jarvis.logs import get_logger

log = get_logger("tools.canvas_ical")

# A Canvas feed names the course(s) in a bracket at the END of the SUMMARY.
# Cross-listed sections all land in the one bracket, which is why it can be
# long; bounded anyway so a pathological title cannot make this quadratic.
_TAIL_RX = re.compile(r"\s*\[([^\[\]]{2,400})\]\s*$")
# TAMU writes DEPT-###; other Canvas installs write "DEPT 123" or "DEPT123".
# Two-to-four letters then three-or-four digits is what a catalogue code is,
# and it is what keeps "[tentative]" or "[Section 101]" from reading as one.
_CODE_RX = re.compile(r"\b([A-Za-z]{2,4})[\s\-_]?(\d{3,4}[A-Za-z]?)\b")
# The same shape in raw ics bytes, for the cheap prescreen in _feed_raws:
# only a feed that looks like coursework is worth handing to icalendar.
_RAW_HINT = re.compile(rb"\[[A-Za-z]{2,4}[ \-_]?\d{3,4}")

# Canvas shows an all-day assignment as "due 11:59 pm" and grades it that
# way; the .ics carries only the DATE. Midnight would move every deadline
# a whole day early -- the heads-up would fire on the wrong evening and
# "due today" would read as "due yesterday" all afternoon.
DUE_HOUR, DUE_MINUTE = 23, 59

FEED_DAYS = 120                 # a term; the calendar cache keeps 14 days
CACHE_NAME = "canvas_coursework.json"
CACHE_VERSION = 1
MAX_CACHE_ROWS = 400            # a term of coursework is ~50; this is a fuse


# ------------------------------------------------------------- the tell
def course_codes(bracket: str) -> list[str]:
    """['BMEN 427', 'BMEN 627', 'ECEN 463', ...] from a bracket's contents,
    in the order Canvas wrote them, deduplicated."""
    out: list[str] = []
    seen: set = set()
    for dept, num in _CODE_RX.findall(str(bracket or "")):
        code = f"{dept.upper()} {num.upper()}"
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def split_coursework(title) -> Optional[tuple[str, str]]:
    """('BMEN 427', 'Lab 1: Introduction to the AD2 SDK'), or None when the
    title is not Canvas coursework.

    The FIRST code wins: a cross-listed lab is one piece of work, and the
    section list behind each code is noise nobody says out loud."""
    text = " ".join(str(title or "").split())
    m = _TAIL_RX.search(text)
    if m is None:
        return None
    codes = course_codes(m.group(1))
    if not codes:
        return None                     # "[tentative]" is not a course code
    # Canvas double-spaces after the colon in a lot of these titles
    # ("Lab 0:  Introduction to Python"); the split above already collapsed
    # it. Strip trailing punctuation the bracket was hanging off.
    clean = " ".join(text[:m.start()].split()).strip(" -:–—,")
    if not clean:
        return None
    return codes[0], clean


def is_coursework(title) -> bool:
    return split_coursework(title) is not None


def other_events(events) -> list:
    """The calendar events that are NOT coursework.

    Callers that consume coursework as ROWS must hand this, not the raw
    list, to anything that also reads events (canvas.exam_candidates): the
    same assignment would otherwise appear twice, once cleanly and once
    with its bracket of section numbers still attached."""
    return [ev for ev in (events or [])
            if not is_coursework(getattr(ev, "title", ""))]


def has_coursework(events) -> bool:
    """True when the calendar carries Canvas coursework at all -- the probe
    that says the feed is a real source, so "Nothing due this week, sir" is
    an answer rather than a guess about a source nobody read. Deliberately
    ignores dates: a term with nothing due in the next two days is still a
    term Jarvis can speak for."""
    return any(is_coursework(getattr(ev, "title", "")) for ev in (events or []))


# ---------------------------------------------------------------- rows
def row_from_event(ev, tz=None) -> Optional[dict]:
    """One calendar Event -> a ``canvas.fetch_due`` row, or None."""
    parsed = split_coursework(getattr(ev, "title", ""))
    start = getattr(ev, "start", None)
    if parsed is None or not isinstance(start, datetime):
        return None
    course, title = parsed
    if getattr(ev, "all_day", False):
        # Canvas's own meaning for a dateless assignment (see DUE_HOUR).
        due = start.replace(hour=DUE_HOUR, minute=DUE_MINUTE, second=0,
                            microsecond=0)
    else:
        due = start
    if due.tzinfo is None:
        # A naive start is a parser accident, not a UTC instant; read it in
        # the zone the caller is working in rather than shifting it.
        due = due.replace(tzinfo=tz or timezone.utc)
    return {"course": course, "title": title, "due": due}


def rows_from_events(events, days: Optional[int] = None, now: Optional[datetime] = None,
                     tz=None) -> list[dict]:
    """Outstanding coursework in ``events`` within ``days``, soonest first.

    Past due times are dropped, exactly as ``canvas.fetch_due`` drops them:
    an .ics cannot say whether the work went in, so the due time passing is
    the only "done" signal there is."""
    now = now or datetime.now().astimezone()
    tz = tz or now.tzinfo
    end = now + timedelta(days=int(days)) if days else None
    out = []
    for ev in (events or []):
        row = row_from_event(ev, tz)
        if row is None or row["due"] <= now:
            continue
        if end is not None and row["due"] > end:
            continue
        out.append(row)
    return merge_rows(out)


def merge_rows(*groups) -> list[dict]:
    """Union of ``{course, title, due}`` rows, soonest first; the FIRST
    group wins a duplicate (same title, same local day).

    Callers pass the REST reading first on purpose: it knows the course's
    real name and whether the work was handed in, and the feed knows
    neither. The key is title-and-day rather than title-and-instant because
    the feed's all-day 11:59 pm is a convention, not a timestamp Canvas
    published, so it will not match the planner's to the second."""
    seen: set = set()
    out: list[dict] = []
    for group in groups:
        for row in (group or []):
            if not isinstance(row, dict):
                continue
            due = row.get("due")
            title = " ".join(str(row.get("title") or "").split())
            if not isinstance(due, datetime) or not title:
                continue
            try:
                day = due.astimezone().date()
            except (ValueError, OverflowError, OSError):
                continue
            key = (title.lower(), day)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
    out.sort(key=lambda r: r["due"])
    return out


# --------------------------------------------------------- deep horizon
_LOCK = threading.Lock()
_PARSED: dict = {}              # (digest, window start, days) -> rows


def _unfold(raw: bytes) -> bytes:
    """RFC 5545 line folding undone, so a SUMMARY split across lines still
    shows its bracket to the prescreen."""
    return re.sub(rb"\r?\n[ \t]", b"", raw)


def looks_like_coursework_feed(raw) -> bool:
    if not isinstance(raw, (bytes, bytearray)):
        raw = str(raw or "").encode("utf-8", "replace")
    return bool(_RAW_HINT.search(_unfold(bytes(raw))))


def _feed_raws(calendar) -> Optional[list[bytes]]:
    """The raw ics bodies that look like coursework feeds, or None when the
    source cannot hand any back (no CalendarSource, or nothing fetched yet
    this run). None and [] mean different things -- see ``deep_rows``."""
    getter = getattr(calendar, "raw_ics", None)
    if not callable(getter):
        return None
    try:
        raws = getter() or {}
    except Exception:                       # noqa: BLE001 - source boundary
        log.debug("canvas feed: raw ics unavailable", exc_info=True)
        return None
    bodies = [raw for raw in raws.values()
              if isinstance(raw, (bytes, bytearray)) and looks_like_coursework_feed(raw)]
    return bodies if raws else None


def _parse_window(raw: bytes, start: datetime, end: datetime, tz) -> list:
    from jarvis.tools.calendar import parse_ics       # heavy; and avoids a cycle
    return parse_ics(raw, start, end, tz=tz)


def deep_rows(calendar, days: int = FEED_DAYS, now: Optional[datetime] = None,
              cache_path=None) -> list[dict]:
    """Coursework over a TERM rather than the calendar cache's fortnight.

    No network: the bytes are the ones ``CalendarSource`` already fetched
    with its conditional GET. Parsing is memoised per (response, window) so
    a spoken turn pays for it at most once, and the result is mirrored to
    disk so a process that has not refreshed yet still answers.

    Returns [] on any failure -- the caller falls back to the fortnight of
    events it already has, which is a smaller answer, never a wrong one."""
    now = now or datetime.now().astimezone()
    tz = now.tzinfo
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=max(1, int(days)))
    raws = _feed_raws(calendar)
    if raws is None:
        return _load_cache(now, days, cache_path)
    if not raws:
        # A CalendarSource that answered, with no coursework feed among its
        # subscriptions: that is a real "no", not a cold cache.
        return []
    rows: list[dict] = []
    parsed_any = False
    for raw in raws:
        key = (hashlib.sha1(bytes(raw)).hexdigest(), start.date().isoformat(), int(days))
        with _LOCK:
            hit = _PARSED.get(key)
        if hit is None:
            try:
                events = _parse_window(bytes(raw), start, end, tz)
            except Exception:               # noqa: BLE001 - a bad feed is a miss
                log.debug("canvas feed: deep parse failed", exc_info=True)
                continue
            hit = rows_from_events(events, None, now, tz)
            with _LOCK:
                if len(_PARSED) > 8:        # one live feed, one window: a fuse
                    _PARSED.clear()
                _PARSED[key] = hit
        parsed_any = True
        rows = merge_rows(rows, hit)
    if not parsed_any:
        # Every feed refused to parse: fall back to what was last written
        # rather than overwriting a good cache with the failure.
        return _load_cache(now, days, cache_path)
    # The memo is keyed by the DAY, so a row that came due since the parse
    # is still in it; drop it here rather than re-parsing every hour.
    rows = [r for r in rows if r["due"] > now]
    _save_cache(rows, cache_path)
    return rows


# ------------------------------------------------------------ disk cache
def _cache_path(path=None) -> Path:
    if path is not None:
        return Path(path)
    from jarvis.tools.location import cache_dir
    return cache_dir() / CACHE_NAME


def _save_cache(rows: list[dict], path=None) -> None:
    try:
        target = _cache_path(path)
        payload = {"version": CACHE_VERSION,
                   "rows": [{"course": r["course"], "title": r["title"],
                             "due": r["due"].isoformat()}
                            for r in rows[:MAX_CACHE_ROWS]]}
        target.parent.mkdir(parents=True, exist_ok=True)
        # The scratch name carries the writer's identity: a briefing pool
        # and a spoken turn can both land here at once, and two writers
        # sharing one tmp file would publish half of each other's JSON.
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)                 # atomic: never half a file
    except Exception:                       # noqa: BLE001 - best effort
        log.debug("canvas feed: coursework cache not written", exc_info=True)


def _load_cache(now: datetime, days: int, path=None) -> list[dict]:
    """The last deep parse, filtered to the window asked for. Corrupt or
    missing reads as empty; a stale row is dropped by its own due time, so
    an abandoned cache decays to nothing rather than inventing homework."""
    try:
        data = json.loads(_cache_path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception:                       # noqa: BLE001 - a bad cache is a miss
        log.debug("canvas feed: coursework cache unreadable", exc_info=True)
        return []
    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
        return []
    end = now + timedelta(days=max(1, int(days)))
    out = []
    for row in (data.get("rows") or [])[:MAX_CACHE_ROWS]:
        if not isinstance(row, dict):
            continue
        try:
            due = datetime.fromisoformat(str(row.get("due")))
        except (TypeError, ValueError):
            continue
        if due.tzinfo is None:
            due = due.replace(tzinfo=now.tzinfo or timezone.utc)
        title = " ".join(str(row.get("title") or "").split())
        if not title or due <= now or due > end:
            continue
        out.append({"course": str(row.get("course") or ""), "title": title, "due": due})
    return merge_rows(out)


def clear_cache() -> None:
    """Drop the in-process parse memo (tests; a config reload)."""
    with _LOCK:
        _PARSED.clear()
