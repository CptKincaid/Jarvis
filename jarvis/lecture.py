"""Lecture notes by voice: "notes for biosensors" ... "end notes".

While a course is open, every accepted utterance is appended as a
timestamped line to ``<docs folder>/notes/<course-slug>-<YYYY-MM-DD>.md``
(the first folder in assistant.json ``docs.paths``, the same folder
``ask_docs`` indexes, so the next reindex makes the notes searchable) and
mirrored into the NotesStore tagged ``course:<slug>``. The commander owns
the mode flag (``Commander.lecture_course``, checked before every other
router exactly like dictation); this module owns the file.

The course name is tidied against the Canvas roster when a token is set
("biosensors" -> "BIOSENSORS" as Canvas titles it, minus catalogue number
and term); with no token, or when Canvas is unreachable, the spoken name is
used as said. The Canvas call is bounded by canvas._Budget and cached.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.docs import doc_paths

log = get_logger("lecture")

NOTES_SUBDIR = "notes"
HEADER = "# {course} — {date}\n\n"
LINE = "- {time}  {text}\n"

START_LINE = "Taking notes for {course}, sir; say end notes when you're done."
END_LINE = "Notes closed, sir: {n} for {course}."
END_NONE_LINE = "Notes closed, sir; nothing was noted."
FAIL_LINE = "I couldn't write to the notes folder, sir."
PERSONA_LINES = [END_NONE_LINE, FAIL_LINE]


def slugify(course: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(course or "").lower()).strip("-")
    return slug[:48] or "course"


def notes_dir(cfg) -> Path:
    paths = doc_paths(cfg)
    base = paths[0] if paths else Path(os.path.expanduser("~/Documents/Jarvis Docs"))
    return base / NOTES_SUBDIR


def note_path(cfg, course: str, when: Optional[datetime] = None) -> Path:
    day = (when or datetime.now()).date().isoformat()
    return notes_dir(cfg) / f"{slugify(course)}-{day}.md"


def _match_course(spoken: str, names: list[str]) -> Optional[str]:
    """The roster name the spoken form points at, or None. Exact (case
    folded) first, then substring either way, then the most words in
    common -- 'biosensors' must find 'BIOSENSORS' and 'signals' must find
    'SIGNALS AND SYSTEMS' without picking 'SYSTEMS PHYSIOLOGY' for
    'systems' by accident when a closer name exists."""
    said = " ".join(str(spoken or "").lower().split())
    if not said or not names:
        return None
    lowered = [(n, " ".join(n.lower().split())) for n in names if n]
    for name, low in lowered:
        if low == said:
            return name
    for name, low in lowered:
        if said in low or low in said:
            return name
    words = set(said.split())
    best, score = None, 0
    for name, low in lowered:
        hit = len(words & set(low.split()))
        if hit > score:
            best, score = name, hit
    return best


def resolve_course(cfg, spoken: str, fetch: Optional[Callable] = None) -> str:
    """Canvas roster name when a token is set and the call succeeds; the
    spoken name otherwise. Never raises."""
    spoken = " ".join(str(spoken or "").split())
    try:
        from jarvis.tools import canvas
        settings = canvas.canvas_settings(cfg)
        if settings is None:
            return spoken
        courses = canvas.active_courses(settings, fetch or canvas._fetch, canvas._Budget())
        names = [canvas.tidy_course(c.get("name", "")) for c in courses
                 if isinstance(c, dict)]
        hit = _match_course(spoken, names)
        if hit:
            return hit
    except Exception:                      # noqa: BLE001 - network boundary
        log.info("lecture: canvas roster unavailable; using %r as said", spoken,
                 exc_info=True)
    return spoken


class LectureNotes:
    """One open capture: the file for today and a line counter."""

    def __init__(self, cfg, course: str, notes=None,
                 now: Callable[[], datetime] = datetime.now):
        self.course = " ".join(str(course or "").split()) or "notes"
        self.slug = slugify(self.course)
        self._notes = notes
        self._now = now
        self.started = now()
        self.path = note_path(cfg, self.course, self.started)
        self.count = 0
        # Create the file (and notes/) now: a folder that cannot be written
        # is refused at "notes for X", not silently at the first sentence.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(HEADER.format(course=self.course,
                                               date=self.started.date().isoformat()))

    def add(self, text: str) -> int:
        text = " ".join(str(text or "").split())
        if not text:
            return self.count
        stamp = self._now().strftime("%H:%M")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(LINE.format(time=stamp, text=text))
        self.count += 1
        if self._notes is not None:
            try:
                self._notes.add("note", text, tags=f"course:{self.slug}")
            except Exception:              # noqa: BLE001 - store boundary
                log.exception("lecture: notes store add failed")
        return self.count

    def close(self) -> str:
        """The spoken confirmation."""
        log.info("lecture: %d line(s) for %r in %s", self.count, self.course, self.path)
        if self.count == 0:
            return END_NONE_LINE
        n = "one line" if self.count == 1 else f"{self.count} lines"
        return END_LINE.format(n=n, course=self.course)
