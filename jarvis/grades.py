"""New-grade watch: "A grade posted for CIRCUITS, sir: the course is now 94%."

Canvas's course list already carries the student enrolment's total_scores
(``canvas.active_courses`` -> score/grade per course), so a poll every
INTERVAL_S and a diff against the last snapshot is the whole detector. The
first tick after a fresh install only records the baseline -- announcing
every course's standing at boot would be a monologue, not news.

total_scores is a COURSE total, so a delta alone can only say "CIRCUITS is
now 94%". Naming the thing that moved it takes one extra read-only call --
``/courses/:id/students/submissions`` for self, newest graded first -- made
ONLY on a delta (never on a quiet tick) and only for the first
MAX_NAMED_LOOKUPS courses that moved, so a term-start grade dump cannot turn
into a dozen requests. Any failure there just drops the name; the course
line still goes out.

Silent by design without a token: canvas_settings() answers None and tick()
returns 0 without touching the network.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools import canvas as canvas_mod
from jarvis.tools.canvas import CanvasError, canvas_settings
from jarvis.watchers import PollingWatcher, cfg_get

log = get_logger("grades")

# 15 min. active_courses caches for 10 (COURSE_CACHE_S), so every tick
# crosses the cache and sees a fresh score rather than re-reading its own.
INTERVAL_S = 900.0
MAX_LINES = 3                  # a term-start dump is not read out in full
MAX_NAMED_LOOKUPS = 2          # submissions calls per tick, at most
NAMED_WITHIN_DAYS = 3          # a graded_at older than this is not "just posted"
# A percentage wobbles in the last decimal when Canvas recomputes a total;
# only a real move is news.
EPSILON = 0.05


def _pct(score) -> str:
    try:
        return f"{float(score):.1f}%".replace(".0%", "%")
    except (TypeError, ValueError):
        return ""


def _changed(old, new) -> bool:
    """True when the score moved by more than EPSILON, or appeared/vanished."""
    if old is None or new is None:
        return (old is None) != (new is None)
    try:
        return abs(float(new) - float(old)) > EPSILON
    except (TypeError, ValueError):
        return str(old) != str(new)


class GradeWatch(PollingWatcher):
    NAME = "grades"
    INTERVAL_S = INTERVAL_S

    def __init__(self, cfg, announce: Optional[Callable] = None,
                 state_path=None, now: Callable = None, fetch: Callable = None):
        super().__init__(state_path=state_path, announce=announce, now=now)
        self._cfg = cfg
        # The one network seam, so tests never reach Canvas.
        self._fetch = fetch or canvas_mod._fetch
        self._snapshot = self._clean(self._state)

    @staticmethod
    def _clean(raw: dict) -> dict:
        """{course_id: {"score":…, "grade":…, "name":…}} only; a hand-edited
        or older state file must not crash the diff."""
        out = {}
        for key, val in (raw or {}).items():
            if isinstance(key, str) and isinstance(val, dict):
                out[key] = {"score": val.get("score"), "grade": val.get("grade"),
                            "name": str(val.get("name") or "")}
        return out

    @property
    def enabled(self) -> bool:
        return bool(cfg_get(self._cfg, "watch.grades", True))

    # ------------------------------------------------------------ source
    def _courses(self) -> list[dict]:
        settings = canvas_settings(self._cfg)
        if settings is None:
            return []                             # no token: silent
        try:
            return [c for c in canvas_mod.active_courses(
                settings, self._fetch, canvas_mod._Budget()) if c.get("student")]
        except CanvasError as exc:
            log.debug("grades: Canvas unavailable (%s)", exc.kind)
        except Exception:                         # noqa: BLE001 - source boundary
            log.debug("grades: Canvas fetch failed", exc_info=True)
        return []

    def _recent_assignment(self, course_id) -> str:
        """The name of the most recently graded submission in this course,
        or "" -- the delta says a grade moved, this says which one."""
        settings = canvas_settings(self._cfg)
        if settings is None:
            return ""
        try:
            rows = canvas_mod.get_all(
                settings, f"/api/v1/courses/{int(course_id)}/students/submissions",
                self._fetch, canvas_mod._Budget(),
                [("student_ids[]", "self"), ("include[]", "assignment"),
                 ("workflow_state", "graded"), ("order", "graded_at"),
                 ("order_direction", "descending"), ("per_page", "10")])
        except CanvasError as exc:
            log.debug("grades: submissions unavailable (%s)", exc.kind)
            return ""
        except Exception:                         # noqa: BLE001 - source boundary
            log.debug("grades: submissions call failed", exc_info=True)
            return ""
        cutoff = self._now() - timedelta(days=NAMED_WITHIN_DAYS)
        best_at = best_name = None
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            graded = canvas_mod._parse_iso(row.get("graded_at"))
            assignment = row.get("assignment")
            name = " ".join(str(assignment.get("name") or "").split()) \
                if isinstance(assignment, dict) else ""
            # ``order`` is a hint, not a promise: pick the newest ourselves,
            # and refuse a stale one so an old name is never read as new.
            if not name or graded is None or graded < cutoff:
                continue
            if best_at is None or graded > best_at:
                best_at, best_name = graded, name
        return best_name or ""

    # ------------------------------------------------------------- line
    @staticmethod
    def _line(course_name: str, title: str, old: dict, new: dict) -> str:
        course = canvas_mod.tidy_course(course_name)
        lead = f"{title} posted for {course}" if title else f"A grade posted for {course}"
        words = canvas_mod._score_words(new)
        if words == "no score yet":
            return f"{lead}, sir."
        before, after = _pct(old.get("score")), _pct(new.get("score"))
        move = ""
        if before and after and before != after:
            try:
                up = float(new["score"]) > float(old["score"])
            except (TypeError, ValueError):
                up = None
            if up is not None:
                move = f"{', up from ' if up else ', down from '}{before}"
        return f"{lead}, sir: the course is now {words}{move}."

    # ------------------------------------------------------------- tick
    def tick(self) -> int:
        if not self.enabled:
            return 0
        courses = self._courses()
        if not courses:
            return 0
        first_run = not self._snapshot
        moved, fresh = [], {}
        for c in courses:
            key = str(c["id"])
            new = {"score": c.get("score"), "grade": c.get("grade"),
                   "name": c.get("name") or ""}
            fresh[key] = new
            old = self._snapshot.get(key)
            if old is None:
                continue                          # a new course is not a grade
            if _changed(old.get("score"), new["score"]) or \
                    str(old.get("grade") or "") != str(new["grade"] or ""):
                moved.append((c, old, new))
        # Courses that vanished (a term rollover) drop out with the snapshot.
        # Saved BEFORE speaking: a crash mid-announcement must not leave the
        # old snapshot behind and say all of it again on the next tick.
        self._snapshot = fresh
        self._state = dict(fresh)
        self._save()
        if first_run:
            # Baseline only: on a fresh install every course looks new.
            log.info("grades: baseline recorded for %d courses", len(fresh))
            return 0
        said = 0
        for i, (c, old, new) in enumerate(moved[:MAX_LINES]):
            title = self._recent_assignment(c["id"]) if i < MAX_NAMED_LOOKUPS else ""
            if self._speak("Canvas grade", self._line(new["name"], title, old, new)):
                said += 1
        if len(moved) > MAX_LINES:
            log.info("grades: %d more courses moved, not spoken", len(moved) - MAX_LINES)
        return said
