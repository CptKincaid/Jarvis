"""Calendar anomaly watch -- "your 9:10 has just been cancelled, sir."

The calendar refreshes every ten minutes and silently overwrites itself, so
a class cancelled overnight simply is not there in the morning and there is
no way for him to learn it ever existed: ``parse_ics`` drops
``STATUS:CANCELLED`` outright. This module remembers what the last
successful refresh saw, diffs the next one against it, and speaks ONE line
when something today or tomorrow changed.

Four kinds of change, and only two of them fall out of key differencing.
``Event.key()`` is ``(start, end, title.lower(), all_day)`` -- the times are
in it and the LOCATION is not -- so:

* **relocated**  same key, different location: ETB 1003 -> 1020. Invisible
  to a key diff; found by comparing the locations of the events present in
  BOTH snapshots.
* **moved**      gone and appeared, same title on the same day: pushed an
  hour. Found by a second pairing pass over the unmatched sets.
* **cancelled**  gone with no partner.
* **added**      appeared with no partner.

Two things make this safe rather than noisy:

* **The sliding window.** ``CalendarSource._window`` is anchored at
  yesterday-midnight and runs ``window_days + 1`` forward, so it slides
  every midnight: everything before yesterday leaves (and reads as
  cancelled), a new far-edge day enters (and reads as newly scheduled).
  The diff therefore runs ONLY over the intersection of the previous
  snapshot's window and the current one, and the window is recorded in the
  snapshot file. Recurrence itself is fine: expansion over a fixed window
  is deterministic, so an unchanged series re-expands to identical keys.
* **The refresh gate.** ``CalendarSource.refresh()`` keeps a failed
  source's previous events and returns True only when some source
  answered, so a merged diff cannot see a network outage as deletions. The
  "more than N removals" fuse below is belt and braces, not the defence.

Delivery: the soonest today/tomorrow change is spoken through the app's
proactive door (quiet hours hold it for the catch-up digest like any other
proactive line); the rest of today/tomorrow is handed straight to
``quiet.hold`` so it rolls into the digest instead of becoming a monologue;
anything further out is filed silently. Every emitted change is recorded by
key so a change is announced exactly once.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.calendar import Event
from jarvis.tools.location import clock_words

log = get_logger("calwatch")

INTERVAL_S = 600.0             # the calendar's own refresh period
SNAPSHOT_VERSION = 1
MAX_REMOVED = 4                # a bigger removal is a data fault, not a day
ANNOUNCE_KEEP_DAYS = 3
CANCELLED, ADDED, MOVED, RELOCATED = "cancelled", "added", "moved", "relocated"


@dataclass
class Change:
    kind: str                  # cancelled | added | moved | relocated
    event: Event               # the event as it is NOW (the old one when gone)
    prev: Optional[Event] = None


def change_id(ch: Change) -> str:
    """Stable per-change identity for the announced-once bookkeeping.
    The current shape carries the identity; a second move of the same
    class to a third time is a different change and is announced again."""
    ev = ch.event
    return "|".join([ch.kind, ev.start.isoformat(), ev.title.lower().strip(),
                     (ev.location or "").strip()])


def _in_window(ev: Event, window) -> bool:
    if not window:
        return True
    start, end = window
    return start <= ev.start < end


def diff_events(prev, cur, window=None) -> list[Change]:
    """What changed between two snapshots, over ``window`` only.

    ``window`` must already be the INTERSECTION of the two snapshots'
    windows; outside it, "missing" only means "out of range".
    """
    old = {e.key(): e for e in prev if _in_window(e, window)}
    new = {e.key(): e for e in cur if _in_window(e, window)}
    changes: list[Change] = []

    # 1. Same key, different room: the one change a key diff cannot see.
    for key, ev in new.items():
        was = old.get(key)
        if was is None:
            continue
        if (was.location or "").strip() != (ev.location or "").strip():
            changes.append(Change(RELOCATED, ev, was))

    # 2. Pair the unmatched sets on (title, date) before calling anything a
    #    cancellation: a class pushed an hour is gone AND appeared.
    gone = [ev for key, ev in old.items() if key not in new]
    fresh = [ev for key, ev in new.items() if key not in old]
    taken = set()
    for was in gone:
        mate = None
        for i, ev in enumerate(fresh):
            if i in taken:
                continue
            if ev.title.strip().lower() == was.title.strip().lower() and \
                    ev.start.date() == was.start.date():
                mate, taken = ev, taken | {i}
                break
        if mate is None:
            changes.append(Change(CANCELLED, was))
        elif mate.start != was.start or mate.end != was.end:
            changes.append(Change(MOVED, mate, was))
        elif (mate.location or "").strip() != (was.location or "").strip():
            changes.append(Change(RELOCATED, mate, was))
    for i, ev in enumerate(fresh):
        if i not in taken:
            changes.append(Change(ADDED, ev))
    changes.sort(key=lambda c: c.event.start)
    return changes


def _subject(when: Event, clock: Event, today) -> str:
    """"Your 9:10" / "Tomorrow's 9:10" / "Thursday's 9:10" -- ``when``
    dates the line (the OLD event for a move) and ``clock`` gives the time
    being named."""
    day = when.start.date()
    if day == today:
        return f"Your {clock_words(clock.start)}"
    if day == today + timedelta(days=1):
        return f"Tomorrow's {clock_words(clock.start)}"
    return f"{when.start.strftime('%A')}'s {clock_words(clock.start)}"


def _day_word(ev: Event, today) -> str:
    """"today" / "tomorrow" / "Thursday", for the lines that need a noun."""
    day = ev.start.date()
    if day == today:
        return "today"
    if day == today + timedelta(days=1):
        return "tomorrow"
    return ev.start.strftime("%A")


def change_line(ch: Change, today) -> str:
    """One spoken sentence for one change."""
    ev, was = ch.event, ch.prev
    title = (ev.title or "your appointment").strip()
    if ch.kind == CANCELLED:
        return f"{_subject(ev, ev, today)} {title} has been cancelled, sir."
    if ch.kind == MOVED:
        old = was if was is not None else ev
        return (f"{_subject(old, old, today)} {title} has moved to "
                f"{clock_words(ev.start)}, sir.")
    if ch.kind == RELOCATED:
        where = (ev.location or "").strip() or "somewhere new"
        return (f"{_subject(ev, ev, today)} {title} has moved to "
                f"{where}, sir.")
    return (f"{title} has been added to {_day_word(ev, today)} at "
            f"{clock_words(ev.start)}, sir.")


def spoken_horizon(ch: Change, today) -> bool:
    """Today and tomorrow are spoken; further out is filed silently -- a
    room change three weeks out is not worth interrupting for."""
    day = (ch.prev or ch.event).start.date()
    return today <= day <= today + timedelta(days=1)


def overlap(win_a, win_b):
    """The intersection of two (start, end) windows, or None."""
    if not win_a or not win_b:
        return None
    start, end = max(win_a[0], win_b[0]), min(win_a[1], win_b[1])
    return (start, end) if start < end else None


class CalendarWatch:
    """Diff every successful refresh; speak what changed today/tomorrow."""

    def __init__(self, get_calendar: Callable, say: Optional[Callable] = None,
                 quiet=None, cfg=None, state_path: Optional[Path] = None,
                 now: Optional[Callable] = None,
                 max_removed: int = MAX_REMOVED):
        self._get_calendar = get_calendar
        self._say = say
        self._quiet = quiet
        self._cfg = cfg
        self._state_path = Path(state_path) if state_path else None
        self._now = now or (lambda: datetime.now().astimezone())
        self.max_removed = int(max_removed)
        state = self._load()
        self._snapshot = state.get("snapshot")
        self._announced: dict = state.get("announced", {})
        self._stop = threading.Event()
        self._thread = None

    @property
    def enabled(self) -> bool:
        get = getattr(self._cfg, "get", None)
        if not callable(get):
            return True
        try:
            return bool(get("calendar.anomaly_watch", True))
        except Exception:  # noqa: BLE001 - a config hiccup keeps it on
            return True

    # ------------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if not isinstance(data, dict):
                    return {}
                announced = data.get("announced")
                snap = data.get("snapshot")
                return {
                    "snapshot": snap if isinstance(snap, dict) else None,
                    # str -> ISO-string only (headsup.py's lesson: one stray
                    # int from a hand edit raised on every prune).
                    "announced": {k: v for k, v in (announced or {}).items()
                                  if isinstance(k, str) and isinstance(v, str)}
                    if isinstance(announced, dict) else {}}
        except (OSError, ValueError):
            log.debug("calwatch state unreadable", exc_info=True)
        return {}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps({"version": SNAPSHOT_VERSION,
                                       "snapshot": self._snapshot,
                                       "announced": self._announced}))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("calwatch state save failed", exc_info=True)

    def _snapshot_of(self, events, window) -> dict:
        return {"saved_at": self._now().isoformat(),
                "window": [window[0].isoformat(), window[1].isoformat()]
                if window else None,
                "events": [ev.to_dict() for ev in events]}

    @staticmethod
    def _restore(snapshot: dict) -> tuple:
        """(events, window) from a stored snapshot; ([], None) when it is
        unusable -- a bad snapshot must cost one silent cycle, not a crash."""
        if not isinstance(snapshot, dict):
            return [], None
        try:
            events = [Event.from_dict(d) for d in snapshot.get("events") or []]
        except Exception:  # noqa: BLE001 - a hand-edited file
            log.debug("calwatch: snapshot events unreadable", exc_info=True)
            return [], None
        win = snapshot.get("window")
        try:
            window = (datetime.fromisoformat(win[0]),
                      datetime.fromisoformat(win[1])) if win else None
        except (TypeError, ValueError, IndexError):
            window = None
        return events, window

    # -------------------------------------------------------------- tick
    def tick(self) -> list:
        """One refresh + diff. Returns the changes that were emitted."""
        if not self.enabled:
            return []
        cal = self._get_calendar() if callable(self._get_calendar) else self._get_calendar
        if cal is None:
            return []
        try:
            conf = getattr(cal, "configured", True)
            if callable(conf):
                conf = conf()
            if not conf:
                return []
            # Gate on refresh() itself: it keeps a failed source's previous
            # events and answers True only when SOME source came back, so a
            # network outage can never present as a day of cancellations.
            if not cal.refresh():
                return []
            events = list(cal.events())
            # The window the events were GATHERED over, not the one the
            # source promises to fetch next time. They differ whenever a
            # feed is behind -- in back-off, or simply slower than the
            # widening of WINDOW_DAYS -- and the difference is the whole of
            # the 2026-09-05 burst: 31 far-out events arriving at once from
            # a recovered feed, every one of them shaped like a new booking.
            # None means "not knowable yet" and costs one silent cycle.
            covered = getattr(cal, "covered_window", None)
            window = covered() if callable(covered) else cal.window()
        except Exception:  # noqa: BLE001 - source boundary
            log.exception("calwatch: refresh failed")
            return []
        prev_events, prev_window = self._restore(self._snapshot)
        self._snapshot = self._snapshot_of(events, window)
        if not self._snapshot.get("window") or prev_window is None:
            self._save()                       # first run: learn, say nothing
            return []
        win = overlap(prev_window, window)
        if win is None:
            self._save()
            return []
        changes = diff_events(prev_events, events, win)
        removed = sum(1 for c in changes if c.kind == CANCELLED)
        if removed > self.max_removed:
            # Belt and braces behind the refresh gate: a diff that deletes
            # half the term is a data fault, and speaking it would be the
            # single worst thing this feature could do.
            log.warning("calwatch: %d removals in one diff; suppressed", removed)
            self._save()
            return []
        fresh = [c for c in changes if change_id(c) not in self._announced]
        self._deliver(fresh)
        stamp = self._now().isoformat()
        for c in fresh:
            self._announced[change_id(c)] = stamp
        self._prune()
        self._save()
        return fresh

    def _deliver(self, changes: list) -> None:
        if not changes:
            return
        today = self._now().date()
        spoken, filed = [], []
        for c in changes:
            (spoken if spoken_horizon(c, today) else filed).append(c)
        for c in filed:
            log.info("calwatch: %s (filed, out of horizon)", change_line(c, today))
        if not spoken:
            return
        lines = [change_line(c, today) for c in spoken]
        first, rest = lines[0], lines[1:]
        if callable(self._say):
            try:
                # proactive: quiet hours / DND / a running class hold it for
                # the catch-up digest exactly like any other unbidden line.
                self._say(first, proactive=True, kind="message")
            except Exception:  # noqa: BLE001 - the tick must survive TTS
                log.exception("calwatch: speak failed")
        else:
            log.info("calwatch: %s", first)
        hold = getattr(self._quiet, "hold", None)
        for line in rest:
            # ONE spoken line per refresh; the remainder waits for the
            # digest rather than becoming a monologue.
            if callable(hold):
                try:
                    hold(line, "message")
                    continue
                except Exception:  # noqa: BLE001
                    log.exception("calwatch: hold failed")
            log.info("calwatch: %s (not spoken)", line)

    def _prune(self) -> None:
        cutoff = (self._now() - timedelta(days=ANNOUNCE_KEEP_DAYS)).isoformat()
        for k in [k for k, v in self._announced.items()
                  if not isinstance(v, str) or v < cutoff]:
            self._announced.pop(k, None)

    # ----------------------------------------------------------- thread
    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again and a dead thread must not block a fresh one.
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="calwatch")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            # a fetch in flight must not outlive stop_assistant
            t.join(timeout=3.0)

    def _run(self) -> None:
        # first pass a little after boot: the calendar's own refresh and the
        # network are both still settling, and a diff against a snapshot
        # from last week should not race the first fetch.
        if self._stop.wait(40.0):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("calwatch tick failed")
            if self._stop.wait(INTERVAL_S):
                return
