"""Deadline heads-up: "Sir, this is your reminder. Lab 3 report for
BIOSENSORS is due in 3 hours."

The meeting heads-up (jarvis/headsup.py) skips all-day items and leads by
ten minutes, so an 11:59 pm Canvas deadline got at best an 11:49 pm nudge.
This thread is its twin for coursework: every INTERVAL_S it asks Canvas
for what is due in the next two days and files ONE reminder with the
timekeeper ``lead_hours`` before each deadline. The same pass files an
evening-before heads-up for anything that looks like an exam (Canvas
items and calendar events alike, see tools/canvas.exam_candidates): "Midterm
1 for BIOSENSORS is tomorrow at 9:00 am" at EXAM_EVE_HOUR the day before.

A THIRD source joins the two: dates a syllabus scan proposed and Hunter
accepted (jarvis/syllabus.py), which is how an exam a professor never put
in Canvas gets the same heads-up. Canvas wins a duplicate, since its due
time is the authoritative one. canvas.find_next_exam merges the same
source, deliberately -- an exam eve for something "when's my next exam"
would then deny is worse than no exam eve at all.

The Canvas half no longer needs a token: his university blocks personal
ones, so ``canvas_ical`` reads the same coursework out of the Canvas
calendar feed and hands it over in the same {course, title, due} rows.
With neither a token nor a coursework feed there is simply nothing to
file, and tick() returns 0 without a log line, so a box with no coursework
at all never nags. A CanvasError (outage, bad token) is logged at debug
and the REST half is skipped; the feed and the calendar still run. State
(what has been filed) is
an atomic JSON file under PATHS.MEMORY_DIR so a restart cannot file the
same deadline twice -- the timekeeper persists the reminders themselves.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools import canvas as canvas_mod
from jarvis.tools import canvas_ical
from jarvis.tools.canvas import CanvasError, canvas_settings, exam_candidates
from jarvis.tools.location import clock_words

log = get_logger("deadlines")

INTERVAL_S = 900.0             # one planner call per tick; 15 min is plenty
DUE_DAYS = 2                   # planner window: covers tonight and tomorrow's exam
DEFAULT_LEAD_HOURS = 3
# A reminder is filed only once its due time is within the next tick or so:
# filing days ahead would speak "due in 3 hours" for work already handed
# in, and a filed reminder cannot be withdrawn.
FILE_AHEAD = timedelta(minutes=45)
EXAM_EVE_HOUR = canvas_mod.EXAM_EVE_HOUR


def _hours_words(delta: timedelta) -> str:
    minutes = max(1, int(round(delta.total_seconds() / 60)))
    if minutes < 60:
        return f"in {minutes} minute{'s' if minutes != 1 else ''}"
    hours, rem = divmod(minutes, 60)
    if rem >= 45:
        hours, rem = hours + 1, 0
    half = " and a half" if 15 <= rem < 45 else ""
    if hours == 1:
        return f"in an hour{half}"
    return f"in {hours}{half} hours"


class DeadlineHeadsUp:
    def __init__(self, cfg, timekeeper, lead_hours: float = DEFAULT_LEAD_HOURS,
                 state_path: Optional[Path] = None, now: Callable = None,
                 get_calendar: Callable = None, fetch_due: Callable = None,
                 syllabus_path: Optional[Path] = None):
        self._cfg = cfg
        self._syllabus_path = Path(syllabus_path) if syllabus_path else None
        self._tk = timekeeper
        try:
            self.lead_hours = max(0.25, float(lead_hours))
        except (TypeError, ValueError):
            self.lead_hours = float(DEFAULT_LEAD_HOURS)
        self._state_path = Path(state_path) if state_path else None
        # Local zone, not UTC: "the evening before" is a local date.
        self._now = now or (lambda: datetime.now().astimezone())
        self._get_calendar = get_calendar
        self._fetch_due = fetch_due or canvas_mod.fetch_due
        self._filed: dict[str, str] = self._load()
        # The last Canvas planner rows this thread fetched, for readers on
        # the SPOKEN path (jarvis/aside.py). fetch_due is a live REST
        # round-trip; putting it behind a reply would add network latency to
        # a turn and stall it behind a Canvas timeout on an outage. This
        # tick already runs every 15 min on its own thread, so a snapshot
        # that old is indistinguishable from a fresh one in a sentence.
        self._snapshot: list[dict] = []
        self._snapshot_at: Optional[datetime] = None
        self._stop = threading.Event()
        self._thread = None

    def snapshot(self) -> list[dict]:
        """The last Canvas rows seen, never a fetch. Copy: the caller must
        not be able to mutate the list this thread rewrites."""
        return list(self._snapshot)

    @property
    def snapshot_at(self) -> Optional[datetime]:
        return self._snapshot_at

    # ------------------------------------------------------------ state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if not isinstance(data, dict):
                    return {}
                # str -> ISO-string only: _prune compares values as strings
                # (headsup.py learned this from a hand-edited file).
                return {k: v for k, v in data.items()
                        if isinstance(k, str) and isinstance(v, str)}
        except (OSError, ValueError):
            log.debug("deadlines state unreadable", exc_info=True)
        return {}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self._filed))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("deadlines state save failed", exc_info=True)

    # ------------------------------------------------------------ sources
    def _canvas_items(self, now: datetime, events: Optional[list] = None) -> list[dict]:
        """The REST reading merged with the calendar feed's coursework.

        The feed is what makes this thread work at all on a box whose
        university blocks Canvas tokens: without it, no token meant no
        rows, which meant no heads-up for an 11:59 pm deadline sitting in
        plain sight in his calendar. REST rows go first -- they know
        whether the work was already handed in."""
        rest: list[dict] = []
        settings = canvas_settings(self._cfg)
        if settings is not None:
            try:
                rest = list(self._fetch_due(settings, DUE_DAYS, None, now))
            except CanvasError as exc:
                log.debug("deadlines: Canvas unavailable (%s)", exc.kind)
            except Exception:                     # noqa: BLE001 - source boundary
                log.debug("deadlines: Canvas fetch failed", exc_info=True)
        feed = canvas_ical.rows_from_events(
            self._calendar_events() if events is None else events, DUE_DAYS, now)
        return canvas_ical.merge_rows(rest, feed)

    def _calendar_events(self) -> list:
        cal = self._get_calendar() if callable(self._get_calendar) else self._get_calendar
        return canvas_mod._calendar_events(cal)

    def _syllabus_items(self, now: datetime) -> list[dict]:
        """The third source: dates accepted from a syllabus scan. Canvas
        wins a tie (its due time is authoritative), so a professor who
        posts the midterm AND lists it in the syllabus is not reminded
        about it twice."""
        try:
            from jarvis import syllabus as syllabus_mod
            return syllabus_mod.stored_items(now, self._syllabus_path)
        except Exception:                         # noqa: BLE001 - source boundary
            log.debug("deadlines: syllabus items unavailable", exc_info=True)
            return []

    # ------------------------------------------------------------- tick
    def tick(self) -> int:
        """File the reminders that are due to be filed; returns how many."""
        if self._tk is None:
            return 0
        now = self._now()
        from jarvis import syllabus as syllabus_mod
        events = self._calendar_events()
        items = syllabus_mod.merge_items(self._canvas_items(now, events),
                                         self._syllabus_items(now))
        # Stash before any filing: a tick that raises later must still leave
        # the reply path a usable snapshot. The MERGED list, so an aside can
        # also see a date a syllabus scan proposed and Canvas never had.
        self._snapshot, self._snapshot_at = list(items), now
        filed = 0
        filed += self._file_deadlines(items, now)
        # Coursework is in ``items`` now, cleanly; its raw VEVENT must not
        # also be a candidate or an exam-eve would be filed twice, once
        # under a title still carrying its bracket of section numbers.
        filed += self._file_exam_eves(
            exam_candidates(items, canvas_ical.other_events(events), now), now)
        self._prune(now)
        if filed:
            self._save()
        return filed

    def _file(self, key: str, when: datetime, due: datetime, text: str) -> bool:
        try:
            self._tk.add_reminder(due.timestamp(), text)
        except Exception:
            log.exception("deadlines: reminder %r failed", text)
            return False
        self._filed[key] = when.astimezone(timezone.utc).isoformat()
        log.info("deadlines: %r filed for %s", text, due.astimezone().strftime("%a %H:%M"))
        return True

    def _file_deadlines(self, items: list, now: datetime) -> int:
        lead = timedelta(hours=self.lead_hours)
        count = 0
        for it in items:
            due_at = it.get("due") if isinstance(it, dict) else None
            title = " ".join(str(it.get("title") or "").split()) if isinstance(it, dict) else ""
            if not isinstance(due_at, datetime) or not title or due_at <= now:
                continue
            if due_at - now > lead + FILE_AHEAD:
                continue
            course = str(it.get("course") or "")
            key = f"{course}|{title}|{due_at.astimezone(timezone.utc).isoformat()}"
            if key in self._filed:
                continue
            fire = due_at - lead
            if fire <= now:                       # inside the lead: say so now
                fire = now + timedelta(seconds=5)
                left = due_at - now
            else:
                left = lead
            what = f"{title} for {course}" if course else title
            if self._file(key, due_at, fire, f"{what} is due {_hours_words(left)}"):
                count += 1
        return count

    def _file_exam_eves(self, exams: list, now: datetime) -> int:
        count = 0
        tz = now.tzinfo
        for ex in exams:
            when = ex["when"]
            local = when.astimezone(tz) if tz else when
            if (local.date() - now.date()).days != 1:
                continue                          # only "tomorrow" gets an eve call
            eve = datetime.combine(local.date() - timedelta(days=1),
                                   dtime(EXAM_EVE_HOUR, 0), tzinfo=tz)
            if eve - now > FILE_AHEAD:
                continue                          # not yet evening
            key = f"exam|{ex['course']}|{ex['title']}|{when.astimezone(timezone.utc).isoformat()}"
            if key in self._filed:
                continue
            fire = eve if eve > now else now + timedelta(seconds=5)
            what = f"{ex['title']} for {ex['course']}" if ex.get("course") else ex["title"]
            clock = "" if ex.get("all_day") else f" at {clock_words(local)}"
            if self._file(key, when, fire, f"{what} is tomorrow{clock}"):
                count += 1
        return count

    def _prune(self, now: datetime) -> None:
        cutoff = (now.astimezone(timezone.utc) - timedelta(days=1)).isoformat()
        stale = [k for k, v in self._filed.items()
                 if not isinstance(v, str) or v < cutoff]
        for k in stale:
            self._filed.pop(k, None)
        if stale:
            self._save()

    # ----------------------------------------------------------- thread
    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again (tests, a config reload), and a dead thread must
        # not block a fresh one.
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="deadlines")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            # a tick in flight (Discord post, Canvas fetch) must not outlive
            # stop_assistant into the teardown
            t.join(timeout=2.0)

    def _run(self) -> None:
        # first pass a little after boot: the calendar refresh and the
        # network are both still settling
        if self._stop.wait(30.0):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("deadlines tick failed")
            if self._stop.wait(INTERVAL_S):
                return
