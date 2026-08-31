"""Syllabus ingestion: the exam the professor never put in Canvas.

The common TAMU case, and the one hole in the deadline machinery. Canvas
carries assignments; the midterm dates live in a PDF in
``~/Documents/Jarvis Docs``, already chunked and embedded by DocsIndex --
and therefore searchable by ``ask_docs`` but invisible to "when's my next
exam", to the evening-before heads-up and to the briefing countdown.

The pass is: retrieve the syllabus-shaped chunks (``topic_chunks`` on
"exam schedule" / "due dates" / "grading and assignment schedule"), ONE
gemma call proposing ``{title, course, due}`` rows, then a spoken read-back
-- the destructive-confirm pattern, because a model reading dates out of a
PDF is exactly the place a wrong year files a reminder for the wrong week.
Only a "yes" writes ``PATHS.MEMORY_DIR/syllabus_deadlines.json``.

That file is then a THIRD source beside Canvas and the calendar, merged in
three places, not one:

  * ``deadlines.tick``          -> the lead-hours reminder and the exam eve
  * ``canvas.find_next_exam``   -> "when's my next exam" and the briefing

Merging into only the first would have him remind Hunter about an exam he
would then deny having when asked -- the reminder and the answer must come
from the same set.

Nothing here talks to Ollama or to chroma: the retrieval seam is a
DocsIndex passed in, the model seam is ``brain.read_syllabus``, and both
are stubbed in tests.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger
from jarvis.tools.docs import EmbedError, topic_chunks

log = get_logger("syllabus")

STATE_NAME = "syllabus_deadlines.json"
# Three retrievals, not one: a syllabus keeps the exam table, the weekly
# schedule and the grading breakdown in different sections, and one query
# for "dates" lands in whichever the embedder likes best.
TOPICS = ("exam schedule", "due dates", "grading and assignment schedule")
CHUNKS_PER_TOPIC = 4
MAX_SOURCE_CHARS = 6000        # what one gemma call reads
MAX_ROWS = 12                  # a semester's exams and majors, not every reading
# A syllabus that is being scanned is for THIS academic year; anything the
# model dates further out than this is a hallucinated year, not a deadline.
MAX_AHEAD_DAYS = 400
# 11:59 pm is the Canvas convention and the one a syllabus means by "due
# Friday"; an exam with no time is all-day and says so.
DEFAULT_DUE_HOUR = 23
DEFAULT_DUE_MINUTE = 59

NO_SYLLABUS_LINE = ("I can't find a syllabus in your documents, sir; drop one in "
                    "the folder and ask me again.")
NO_DATES_LINE = "I couldn't pull any dates out of your syllabus, sir."
FILED_LINE = "Filed, sir; {n} on the books."
FILED_ONE_LINE = "Filed, sir; one on the books."
SCANNING_LINE = "Let me read through your syllabus, sir."

_ISO_RX = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2}))?")


# --------------------------------------------------------------- retrieval
def gather(index, topics=TOPICS, k: int = CHUNKS_PER_TOPIC) -> list[dict]:
    """The syllabus-shaped chunks, deduplicated, in reading order.

    Raises EmbedError (Ollama down) so the caller can say so; a per-topic
    store failure is logged and skipped, because two good topics still
    make a usable pass."""
    out: list[dict] = []
    seen: set = set()
    for topic in topics:
        try:
            hits = topic_chunks(index, topic, k=k)
        except EmbedError:
            raise
        except Exception:                      # noqa: BLE001 - store boundary
            log.exception("syllabus: topic %r failed", topic)
            continue
        for hit in hits:
            key = (hit.get("name"), hit.get("chunk"))
            if key in seen:
                continue
            seen.add(key)
            out.append(hit)
    out.sort(key=lambda h: (str(h.get("name") or ""), int(h.get("chunk") or 0)))
    return out


def source_text(chunks: list[dict], limit: int = MAX_SOURCE_CHARS) -> str:
    """The fact sheet the model reads, file name first so it can fill in
    the course when the chunk itself never names it."""
    parts, total = [], 0
    for hit in chunks:
        line = f"From {hit.get('name', 'the syllabus')}: {' '.join(str(hit.get('text', '')).split())}"
        room = limit - total
        if room <= 0:
            break
        if len(line) > room:
            # The FIRST chunk is truncated rather than dropped: one chunk
            # longer than the budget used to make the whole sheet empty,
            # and the model was then asked to find dates in nothing.
            if parts:
                break
            line = line[:room].rstrip() + "\u2026"
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


# ----------------------------------------------------------------- parsing
def _local(dt: datetime, now: datetime) -> datetime:
    return dt.replace(tzinfo=now.tzinfo) if dt.tzinfo is None else dt.astimezone(now.tzinfo)


def parse_rows(raw, now: datetime, max_rows: int = MAX_ROWS,
               max_ahead_days: int = MAX_AHEAD_DAYS) -> list[dict]:
    """[{title, course, due, all_day}] from the model's rows.

    Everything questionable is DROPPED rather than guessed: a date the
    model could not write as ISO, a date already past (a syllabus scanned
    in October should not file September's midterm), and a date more than
    MAX_AHEAD_DAYS out -- that last one is the year hallucination, the
    failure this whole read-back exists for."""
    out: list[dict] = []
    seen: set = set()
    horizon = now + timedelta(days=max_ahead_days)
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or "").split())
        course = " ".join(str(item.get("course") or "").split())
        m = _ISO_RX.match(str(item.get("due") or "").strip())
        if not title or m is None:
            continue
        try:
            day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue                            # 2026-02-31 and friends
        all_day = m.group(4) is None
        clock = dtime(DEFAULT_DUE_HOUR, DEFAULT_DUE_MINUTE) if all_day else \
            dtime(int(m.group(4)) % 24, int(m.group(5)) % 60)
        due = _local(datetime.combine(day, clock), now)
        if due <= now or due > horizon:
            log.info("syllabus: dropped %r at %s (outside the window)", title, due.date())
            continue
        key = (title.lower(), course.lower(), due.date())
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "course": course, "due": due, "all_day": all_day})
    out.sort(key=lambda r: r["due"])
    return out[:max_rows]


def read_back_line(rows: list[dict], now: datetime) -> str:
    """The question he answers yes or no to. Every row is read out: this
    is the only moment a wrong date can be caught, and "I found four
    dates, shall I add them?" hides exactly the thing being confirmed."""
    from jarvis.tools.canvas import countdown_words
    parts = []
    for row in rows:
        what = f"{row['title']} for {row['course']}" if row.get("course") else row["title"]
        parts.append(f"{what}, {countdown_words(row['due'], now, row.get('all_day', False))}")
    head = "one date" if len(rows) == 1 else f"{len(rows)} dates"
    return f"I found {head} in your syllabus, sir: " + "; ".join(parts) + \
        ". Shall I put them on the books?"


# ------------------------------------------------------------------- store
def state_path() -> Path:
    return PATHS.MEMORY_DIR / STATE_NAME


def save(rows: list[dict], path: Optional[Path] = None) -> None:
    """Atomic write, the deadlines.py idiom: a half-written file here would
    silently drop every syllabus date until the next scan."""
    path = Path(path) if path else state_path()
    data = [{"title": r["title"], "course": r.get("course", ""),
             "due": r["due"].isoformat(), "all_day": bool(r.get("all_day", False))}
            for r in rows]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        log.exception("syllabus: could not save %s", path)


def load(path: Optional[Path] = None) -> list[dict]:
    """Every stored row with its due time parsed; [] when absent or bad."""
    path = Path(path) if path else state_path()
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("syllabus: state unreadable at %s", path)
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            due = datetime.fromisoformat(str(item.get("due")))
        except (TypeError, ValueError):
            continue
        title = " ".join(str(item.get("title") or "").split())
        if not title:
            continue
        out.append({"title": title, "course": str(item.get("course") or ""),
                    "due": due, "all_day": bool(item.get("all_day", False))})
    out.sort(key=lambda r: r["due"])
    return out


def add(rows: list[dict], path: Optional[Path] = None) -> int:
    """Merge accepted rows into the store; how many are new. Re-scanning
    the same syllabus must not double every date."""
    existing = load(path)
    seen = {(r["title"].lower(), str(r.get("course", "")).lower(), r["due"].date())
            for r in existing}
    added = 0
    for row in rows:
        key = (row["title"].lower(), str(row.get("course", "")).lower(), row["due"].date())
        if key in seen:
            continue
        seen.add(key)
        existing.append(row)
        added += 1
    existing.sort(key=lambda r: r["due"])
    save(existing, path)
    return added


def stored_items(now: datetime, path: Optional[Path] = None) -> list[dict]:
    """The store as Canvas-shaped items ({course, title, due}) still in the
    future -- the shape fetch_due returns, so every consumer already knows
    how to read them."""
    out = []
    for row in load(path):
        due = _local(row["due"], now)
        if due <= now:
            continue
        out.append({"course": row.get("course", ""), "title": row["title"],
                    "due": due, "source": "syllabus"})
    return out


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def merge_items(items: list, rows: list) -> list:
    """Canvas items plus the syllabus rows Canvas does not already carry.

    A professor who posts the midterm to Canvas AND lists it in the
    syllabus would otherwise get two reminders and two exam-eve calls, so
    a row matching a Canvas title on the same local day is dropped. Canvas
    wins because its due time is the authoritative one."""
    known = set()
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("due"), datetime):
            known.add((_norm(it.get("title")), it["due"].date()))
    out = list(items)
    for row in rows:
        if (_norm(row.get("title")), row["due"].date()) in known:
            continue
        out.append(row)
    out.sort(key=lambda i: i["due"])
    return out
