"""Nightly flashcards: last week's lecture notes, turned into a deck while
he sleeps.

The exam-week study briefing (tools/briefing._study_section) already reads
a deck and counts what is due, and its empty-deck line says so out loud:
"Nothing on your BIOSENSORS deck yet, sir; say quiz me on BIOSENSORS and
I'll build one."  That was the whole feature -- cards existed only if he
asked for them, in the week he had the least time to ask.  The comment in
that function names this pass as the missing half.

So: once a night, in the small hours, for each course whose lecture notes
have changed, the newest notes file goes to ``brain.make_quiz`` and what
comes back is filed as Leitner cards (``tools/quiz.FlashcardStore``) with
``topic`` set to the course, which is exactly the key the briefing filters
its deck by.  In the morning the deck is simply there.

Rules, each one a way this could have gone wrong:

* **The model is never taken from him.**  Skipped -- not queued -- when
  ``brain.is_lent`` (the GPU is with a trainer) or the brain is busy with
  a turn, exactly like jarvis/garden.py.  Tonight's notes are still there
  tomorrow night.
* **One file, once.**  The state file remembers each notes file by name
  and the number of lines it had when it was carded, so a re-run does
  nothing and a lecture continued after the pass is carded again.  The
  store de-duplicates by question within a source as well, so even that
  second look cannot double the deck.
* **Only fresh notes.**  Nothing older than ``max_age_days``: on the first
  ever run, a folder of a whole term must not become four hundred cards.
* **Only real notes.**  A file with fewer than ``min_lines`` noted lines
  is a mis-fire of the lecture mode, not a lecture; carding it produces
  questions about nothing.
* **Silent.**  Nothing is spoken here.  The cards surface where cards
  already surface -- "review my flashcards" and the exam-week briefing
  line.  A 3 am announcement about homework he did not ask for is the
  opposite of the feature.

Seams for tests: ``make_quiz``, ``store``, ``busy``, ``lent``, ``notes_dir``
and ``now``; the whole pass runs offline.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis import lecture as lecture_mod
# One definition of the notes-file shape, shared with the pre-class dossier:
# two regexes that drifted apart would card files the dossier cannot read.
from jarvis.dossier import NOTE_LINE_RX, NOTE_NAME_RX
from jarvis.logs import get_logger

log = get_logger("studycards")

TICK_S = 900.0                 # a quarter hour, like the garden and the reviewer
FIRST_TICK_S = 120.0
DEFAULT_PER_COURSE = 5
DEFAULT_MAX_COURSES = 3
DEFAULT_RUN_BEFORE_HOUR = 5    # the small hours: he is asleep, the GPU is idle
DEFAULT_MAX_AGE_DAYS = 7
DEFAULT_MIN_LINES = 3
SOURCE_PREFIX = "notes:"       # what the cards say they came from
STATE_VERSION = 1
NOTES_CHARS = 6000             # what make_quiz is handed; its own cap is lower


def _cfg(cfg, dotted: str, default):
    if cfg is None:
        return default
    try:
        value = cfg.get(dotted, default)
    except Exception:                              # noqa: BLE001 - config boundary
        return default
    return default if value is None else value


def _int_cfg(cfg, dotted: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(int(_cfg(cfg, dotted, default)), high))
    except (TypeError, ValueError):
        return default


def read_notes(path: Path) -> dict:
    """{course, date, lines: [str]} for one notes file, or {} when it is
    unreadable or holds nothing he actually noted.

    The course comes from the file's own header (``# BIOSENSORS — 2026-08-30``)
    rather than the slug in its name: the deck is filtered by the course as
    Canvas titles it, and "biosensors" is not that string.
    """
    name = NOTE_NAME_RX.match(path.name)
    if not name:
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.debug("studycards: %s unreadable", path.name, exc_info=True)
        return {}
    course, lines = "", []
    for raw in text.splitlines():
        line = raw.strip()
        if not course and line.startswith("# "):
            head = line[2:].strip()
            # HEADER is "{course} — {date}"; the em dash is the separator
            # lecture.py writes, and a course name never contains one.
            course = head.split("—")[0].strip() or head
            continue
        hit = NOTE_LINE_RX.match(line)
        if hit:
            lines.append(hit.group(1).strip())
    if not lines:
        return {}
    return {"course": course or name.group("slug"), "date": name.group("date"),
            "lines": lines}


def newest_notes(folder: Path, now: datetime, max_age_days: int) -> list[dict]:
    """The newest notes file per course inside the age window, newest
    course first. One file per course: two lectures in a night is a
    student catching up, and the older one has already had its turn."""
    try:
        paths = sorted(p for p in folder.glob("*.md") if p.is_file())
    except OSError:
        log.debug("studycards: notes folder unreadable", exc_info=True)
        return []
    floor = (now - timedelta(days=max(1, int(max_age_days)))).date().isoformat()
    best: dict[str, dict] = {}
    for path in paths:
        m = NOTE_NAME_RX.match(path.name)
        if not m or m.group("date") < floor:
            continue
        slug = m.group("slug")
        if slug in best and best[slug]["date"] >= m.group("date"):
            continue
        best[slug] = {"slug": slug, "date": m.group("date"), "path": path}
    return sorted(best.values(), key=lambda e: e["date"], reverse=True)


class NightlyCards:
    """The nightly pass. ``store`` is a quiz.FlashcardStore (or anything
    with ``add_cards``); ``make_quiz`` is the model seam."""

    def __init__(self, cfg=None, store=None, make_quiz: Optional[Callable] = None,
                 busy: Optional[Callable] = None, lent: Optional[Callable] = None,
                 state_path: Optional[Path] = None, now: Optional[Callable] = None,
                 notes_dir: Optional[Callable] = None):
        self._cfg = cfg
        self._store = store
        self._make_quiz = make_quiz
        self._busy = busy
        self._lent = lent
        self._state_path = Path(state_path) if state_path else None
        self._now = now or datetime.now
        self._notes_dir = notes_dir or (lambda: lecture_mod.notes_dir(cfg))
        self._lock = threading.RLock()
        self._state = self._load()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------- config
    @property
    def enabled(self) -> bool:
        return bool(_cfg(self._cfg, "study_cards.enabled", True))

    @property
    def per_course(self) -> int:
        return _int_cfg(self._cfg, "study_cards.per_course", DEFAULT_PER_COURSE, 1, 10)

    @property
    def max_courses(self) -> int:
        return _int_cfg(self._cfg, "study_cards.max_courses", DEFAULT_MAX_COURSES, 1, 8)

    @property
    def run_before_hour(self) -> int:
        return _int_cfg(self._cfg, "study_cards.run_before_hour",
                        DEFAULT_RUN_BEFORE_HOUR, 0, 23)

    @property
    def max_age_days(self) -> int:
        return _int_cfg(self._cfg, "study_cards.max_age_days",
                        DEFAULT_MAX_AGE_DAYS, 1, 60)

    @property
    def min_lines(self) -> int:
        return _int_cfg(self._cfg, "study_cards.min_lines", DEFAULT_MIN_LINES, 1, 50)

    # --------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict):
                    data.setdefault("done", {})
                    if not isinstance(data["done"], dict):
                        data["done"] = {}
                    return data
        except (OSError, ValueError):
            log.debug("studycards state unreadable", exc_info=True)
        return {"version": STATE_VERSION, "done": {}}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self._state, indent=1))
            os.replace(tmp, self._state_path)      # atomic: never half a file
        except OSError:
            log.debug("studycards state save failed", exc_info=True)

    # ---------------------------------------------------------- scheduling
    def due(self, now: Optional[datetime] = None) -> bool:
        """True when tonight has had no pass and the hour is small enough.

        Keyed on the DATE the pass ran, so a box left on all day runs once;
        a box that is only awake in the evening simply never reaches the
        hour, which is the right answer -- carding notes while he is at the
        desk would take the model mid-conversation.
        """
        if not self.enabled:
            return False
        now = now or self._now()
        if self._state.get("day") == now.date().isoformat():
            return False
        return now.hour < self.run_before_hour

    def _model_free(self) -> str:
        """"" when the model may be used, else why it may not."""
        for fn, why in ((self._lent, "lent to a trainer"), (self._busy, "busy")):
            try:
                if fn is not None and fn():
                    return why
            except Exception:                      # noqa: BLE001
                log.debug("studycards: model gate failed", exc_info=True)
        return ""

    def _fingerprint(self, note: dict) -> str:
        """What makes a notes file "already carded": its name and how many
        lines it had. A lecture continued after the pass is a new
        fingerprint and gets a second look."""
        return f"{len(note['lines'])}"

    # ----------------------------------------------------------- the pass
    def run_pass(self, now: Optional[datetime] = None) -> dict:
        """One pass, schedule ignored (the tick and the tests both call it).
        Returns the record it filed."""
        now = now or self._now()
        record = {"day": now.date().isoformat(),
                  "at": now.isoformat(timespec="seconds"),
                  "made": [], "skipped": ""}
        why = self._model_free()
        if why:
            # NOT recorded: the notes are still there at the next tick.
            log.info("studycards: skipped, the model is %s", why)
            record["skipped"] = why
            return record
        if self._store is None or self._make_quiz is None:
            record["skipped"] = "no deck"
            return record
        candidates = newest_notes(self._notes_dir(), now, self.max_age_days)
        for entry in candidates:
            if len(record["made"]) >= self.max_courses:
                break
            note = read_notes(entry["path"])
            if not note or len(note["lines"]) < self.min_lines:
                continue
            key = entry["path"].name
            if self._state.get("done", {}).get(key) == self._fingerprint(note):
                continue                           # carded already, unchanged
            made = self._card(note, key)
            if made:
                record["made"].append({"course": note["course"], "file": key,
                                       "cards": made})
            # Recorded either way: a course whose notes the model could make
            # nothing of must not be retried every quarter hour all night.
            with self._lock:
                self._state.setdefault("done", {})[key] = self._fingerprint(note)
        self._prune(now)
        with self._lock:
            self._state["day"] = record["day"]
            self._state["at"] = record["at"]
            self._state["made"] = record["made"]
            self._save()
        log.info("studycards: %s filed %d card(s) across %d course(s)",
                 record["day"], sum(m["cards"] for m in record["made"]),
                 len(record["made"]))
        return record

    def _card(self, note: dict, key: str) -> int:
        """Notes -> questions -> cards. Returns how many cards were filed."""
        text = "\n".join(note["lines"])[:NOTES_CHARS]
        course = note["course"]
        try:
            pairs = self._make_quiz(text, n=self.per_course, topic=course) or []
        except Exception:                          # noqa: BLE001 - model boundary
            log.exception("studycards: question generation failed for %r", course)
            return 0
        if not pairs:
            log.info("studycards: no questions came back for %r", course)
            return 0
        try:
            # topic=course is what the briefing filters the deck by; source
            # names the file so a card can be traced back to the lecture.
            cards = self._store.add_cards(pairs, source=f"{SOURCE_PREFIX}{key}",
                                          topic=course)
        except Exception:                          # noqa: BLE001 - store boundary
            log.exception("studycards: filing cards for %r failed", course)
            return 0
        log.info("studycards: %d card(s) for %r from %s", len(cards), course, key)
        return len(cards)

    def _prune(self, now: datetime) -> None:
        """Forget files older than the age window: the state file must not
        grow a row per lecture per term for ever."""
        floor = (now - timedelta(days=self.max_age_days * 4)).date().isoformat()
        with self._lock:
            done = self._state.get("done") or {}
            for key in list(done):
                m = NOTE_NAME_RX.match(key)
                if m and m.group("date") < floor:
                    done.pop(key, None)
        # No _save here: run_pass writes the state file once, at the end.

    def tick(self) -> Optional[dict]:
        if not self.due():
            return None
        return self.run_pass()

    def last_pass(self) -> dict:
        """What the last pass filed -- for the log triage and diagnostics.
        Never spoken on its own."""
        with self._lock:
            return {"day": self._state.get("day", ""),
                    "made": list(self._state.get("made") or [])}

    # ------------------------------------------------------------- thread
    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again (tests, a config reload), and a dead thread must
        # not block a fresh one.
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="studycards")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            # a pass in flight holds the local model, like the garden's
            t.join(timeout=5.0)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        if self._stop.wait(FIRST_TICK_S):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("studycards tick failed")
            if self._stop.wait(TICK_S):
                return


__all__ = ["NightlyCards", "read_notes", "newest_notes", "SOURCE_PREFIX"]
