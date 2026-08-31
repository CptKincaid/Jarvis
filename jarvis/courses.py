"""Course identity, taken from the calendar rather than from Canvas.

Canvas cannot name his courses on this box: ``canvas.token`` is empty and
``canvas.cached_course_names()`` is read-only by design (it must never
trigger a fetch on the transcriber's path), so it answers ``[]`` until some
other Canvas call warms the module cache -- which cannot happen without a
token. The calendar, however, knows: a timed event whose title appears at
the SAME weekday and clock time on two or more distinct dates is a class,
and a one-off meeting never is.

``recurring_courses`` is what everything else hangs on: the pre-class
dossier (jarvis/dossier.py), the class-start stager (jarvis/classflow.py)
and the calendar leg of quiet hours (jarvis/quiet.py, whose shipped keyword
list -- class / exam / meeting / busy -- matches none of his course titles,
so calendar-driven quiet had never once fired for him).

Pure and I/O-free: every function takes the events it is given.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

MIN_DATES = 2                  # two weeks of the same slot is a course
DEFAULT_WEEKS = 3              # how far either side of "now" a slot may reach


def _norm(title) -> str:
    """The title as written, whitespace collapsed."""
    return " ".join(str(title or "").split())


def fold(title) -> str:
    """Comparison form: lowercase, punctuation to single spaces. Turns
    'ELECTRICAL DESIGN LAB II- Presentation' into
    'electrical design lab ii presentation', so the dash-jammed variant of
    a course title still lines up with the plain one."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _norm(title).lower()).split())


def slot(ev) -> Optional[tuple]:
    """(folded title, weekday, 'HH:MM') for a timed event; None otherwise."""
    start = getattr(ev, "start", None)
    if start is None or getattr(ev, "all_day", False):
        return None
    key = fold(getattr(ev, "title", ""))
    if not key:
        return None
    try:
        return (key, start.weekday(), start.strftime("%H:%M"))
    except (AttributeError, ValueError):
        return None


def recurring_courses(events, weeks: int = DEFAULT_WEEKS,
                      min_dates: int = MIN_DATES,
                      now: Optional[datetime] = None) -> list:
    """The course titles in ``events``: every slot (title + weekday + clock
    time) seen on ``min_dates`` or more distinct dates, earliest slot first.

    ``weeks`` bounds how far from ``now`` an event may sit and still count;
    with ``now`` unset every event counts, which is what the tests and the
    one-shot callers want (the calendar cache is a bounded window anyway).
    """
    horizon = timedelta(days=max(1, int(weeks)) * 7)
    groups: dict = {}
    for ev in (events or []):
        key = slot(ev)
        if key is None:
            continue
        start = ev.start
        if now is not None:
            try:
                if abs(start.date() - now.date()) > horizon:
                    continue
            except (AttributeError, TypeError):
                continue
        g = groups.setdefault(key, {"dates": set(), "titles": Counter(),
                                    "first": start})
        g["dates"].add(start.date())
        g["titles"][_norm(ev.title)] += 1
        if start < g["first"]:
            g["first"] = start
    picked = []
    for g in groups.values():
        if len(g["dates"]) < max(1, int(min_dates)):
            continue
        # The title as it reads most often: one week's "…LAB II- Presentation"
        # must not rename the course for every other week.
        name = g["titles"].most_common(1)[0][0]
        picked.append((g["first"], name))
    picked.sort(key=lambda p: (p[0], fold(p[1])))
    out, seen = [], set()
    for _, name in picked:
        if fold(name) in seen:
            continue
        seen.add(fold(name))
        out.append(name)
    return out


def course_for(title, courses) -> str:
    """The course in ``courses`` that ``title`` names, or "".

    Exact (folded) first, then the LONGEST course that the title starts
    with or that starts with the title -- 'ELECTRICAL DESIGN LAB II-
    Presentation' is that course, 'LAB' on its own is not.
    """
    t = fold(title)
    if not t:
        return ""
    best, best_len = "", 0
    for c in (courses or []):
        f = fold(c)
        if not f:
            continue
        if f == t:
            return _norm(c)
        if (t.startswith(f + " ") or f.startswith(t + " ")) and len(f) > best_len:
            best, best_len = _norm(c), len(f)
    return best


def course_events(events, course: str) -> list:
    """Every event in ``events`` belonging to ``course``, in start order."""
    if not fold(course):
        return []
    hits = [ev for ev in (events or [])
            if slot(ev) is not None and course_for(ev.title, [course])]
    hits.sort(key=lambda e: e.start)
    return hits


def next_class(events, courses, now: datetime, within: Optional[timedelta] = None):
    """(event, course) for the soonest course event starting after ``now``,
    or (None, "") -- the stager's and the dossier's shared lookup."""
    best, best_course = None, ""
    for ev in (events or []):
        if slot(ev) is None:
            continue
        try:
            if ev.start <= now or (within is not None and ev.start - now > within):
                continue
        except TypeError:
            continue
        name = course_for(ev.title, courses)
        if not name:
            continue
        if best is None or ev.start < best.start:
            best, best_course = ev, name
    return best, best_course
