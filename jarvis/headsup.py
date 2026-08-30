"""Meeting heads-up: "BIOSENSORS in ten minutes, sir."

Every few minutes, look at the calendar's cached events and file a reminder
with the timekeeper `lead_min` before each timed event that starts within the
horizon. The timekeeper then does what it always does with a reminder: speaks
it once and publishes ReminderFired. A state file remembers what has been
filed so a restart (the timekeeper persists reminders; this class does not)
cannot file the same meeting twice. All-day events are skipped: nothing
starts at midnight.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("headsup")

INTERVAL_S = 300.0
HORIZON_MIN = 90


class MeetingHeadsUp:
    def __init__(self, get_calendar: Callable, timekeeper, lead_min: int = 10,
                 state_path: Optional[Path] = None, now: Callable = None,
                 horizon_min: int = HORIZON_MIN):
        self._get_calendar = get_calendar
        self._tk = timekeeper
        self.lead_min = max(1, int(lead_min or 10))
        self.horizon_min = horizon_min
        self._state_path = Path(state_path) if state_path else None
        self._now = now or (lambda tz=None: datetime.now(tz))
        self._filed: dict[str, str] = self._load()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------ state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if not isinstance(data, dict):
                    return {}
                # Only str -> ISO-string entries survive: _prune compares
                # the values as strings, and one stray int or null (a hand
                # edit, a half-written file) raised TypeError on every tick
                # before the save could ever run.
                clean = {k: v for k, v in data.items()
                         if isinstance(k, str) and isinstance(v, str)}
                if len(clean) != len(data):
                    log.debug("headsup state: dropped %d malformed entries",
                              len(data) - len(clean))
                return clean
        except (OSError, ValueError):
            log.debug("headsup state unreadable", exc_info=True)
        return {}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic: a crash mid-write must leave the old file behind, not
            # half a JSON document that the next start reads as "nothing
            # filed" (and then files every meeting again).
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self._filed))
            os.replace(tmp, self._state_path)
        except OSError:
            log.debug("headsup state save failed", exc_info=True)

    # ------------------------------------------------------------- tick
    def tick(self) -> int:
        """File reminders for upcoming timed events; returns how many."""
        cal = self._get_calendar() if callable(self._get_calendar) else self._get_calendar
        if cal is None:
            return 0
        try:
            conf = getattr(cal, "configured", True)
            if callable(conf):                 # CalendarSource exposes a property;
                conf = conf()                  # a fake may expose a method
            if not conf:
                return 0
            events = list(cal.events())
        except Exception:
            log.debug("headsup: calendar unavailable", exc_info=True)
            return 0
        filed = 0
        for ev in events:
            start = getattr(ev, "start", None)
            if start is None or getattr(ev, "all_day", False):
                continue
            title = (getattr(ev, "title", "") or "").strip() or "your appointment"
            now = self._now(start.tzinfo)
            lead = timedelta(minutes=self.lead_min)
            if start <= now or start - now > timedelta(minutes=self.horizon_min):
                continue
            key = f"{title}|{start.isoformat()}"
            if key in self._filed:
                continue
            due = start - lead
            if due <= now:                       # inside the lead already: say so now
                due = now + timedelta(seconds=5)
                minutes = max(1, int(round((start - now).total_seconds() / 60)))
            else:
                minutes = self.lead_min
            text = f"{title} in {minutes} minute{'s' if minutes != 1 else ''}"
            try:
                self._tk.add_reminder(due.timestamp(), text)
            except Exception:
                log.exception("headsup: reminder for %r failed", title)
                continue
            self._filed[key] = start.isoformat()
            filed += 1
            log.info("headsup: %r filed for %s", text, due.strftime("%H:%M"))
        self._prune()
        if filed:
            self._save()
        return filed

    def _prune(self) -> None:
        cutoff = (self._now() - timedelta(days=1)).isoformat()
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
        self._thread = threading.Thread(target=self._run, daemon=True, name="headsup")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            # a tick in flight (Discord post, Canvas fetch) must not outlive
            # stop_assistant into the teardown
            t.join(timeout=2.0)

    def _run(self) -> None:
        # first pass a little after boot so the calendar has refreshed
        if self._stop.wait(20.0):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("headsup tick failed")
            if self._stop.wait(INTERVAL_S):
                return
