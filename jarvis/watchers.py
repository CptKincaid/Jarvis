"""The bones every background watcher shares: a joinable poll thread and an
atomic JSON state file under PATHS.MEMORY_DIR.

DeadlineHeadsUp (jarvis/deadlines.py) and MeetingHeadsUp (jarvis/headsup.py)
each carry their own copy of this loop, and three more watchers landing at
once would have made five. The loop itself is the same every time -- wait a
little after boot so the network and the calendar refresh have settled, tick,
sleep INTERVAL_S, and join out on stop() so a fetch in flight cannot outlive
stop_assistant into the teardown -- so it lives here once and the watchers
below are only their tick().

What is deliberately NOT here: the announcement. A watcher speaks through the
``announce`` callback the app hands it (``JarvisApp._announce`` -> ``_say(
proactive=True)`` + ``_alert``), so quiet hours hold the line for the catch-up
digest and presence routes it to Discord -- a watcher that called the TTS
directly would talk over a lecture.

Every watcher here is DARK-SAFE: with no Canvas token, no mailbox and no
keywords, tick() returns 0 without a log line and without touching the
network. A box that has not been set up is never nagged.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("watchers")


def cfg_get(cfg, dotted: str, default=None):
    """Dotted read that works for a plain dict (tests) and for
    AssistantConfig (the app). Same contract as canvas._cfg_get, which is
    private to that module; a None value reads as the default so a config
    key explicitly set to null does not poison a watcher."""
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
        except Exception:                       # noqa: BLE001 - config boundary
            log.debug("cfg.get(%s) failed", dotted, exc_info=True)
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def keyword_hit(haystack: str, keywords) -> str:
    """The first keyword that occurs in ``haystack`` as a whole word (or
    whole phrase), else "". Word boundaries, not a bare substring: "AI"
    must not fire on "again", and a student watching "lab" does not want
    every "collaboration"."""
    hay = " ".join(str(haystack or "").split()).lower()
    if not hay:
        return ""
    for kw in keywords:
        if not kw:
            continue
        if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", hay):
            return kw
    return ""


class PollingWatcher:
    """Subclass and implement ``tick() -> int`` (how many lines it spoke)."""

    NAME = "watcher"
    INTERVAL_S = 900.0
    FIRST_DELAY_S = 30.0
    # Seen-id state is unbounded otherwise: a mailbox's worth of keys would
    # grow the file forever. Entries older than this are pruned each tick.
    KEEP_DAYS = 14
    MAX_STATE = 2000

    def __init__(self, state_path: Optional[Path] = None,
                 announce: Optional[Callable] = None, now: Callable = None):
        self._state_path = Path(state_path) if state_path else None
        # Local zone, not UTC: these lines are read out loud with local times.
        self._now = now or (lambda: datetime.now().astimezone())
        self._announce = announce
        self._state: dict = self._load()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------ state
    def _load(self) -> dict:
        try:
            if self._state_path and self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                if isinstance(data, dict):
                    # str keys only: a hand-edited file must not crash a tick
                    return {k: v for k, v in data.items() if isinstance(k, str)}
        except (OSError, ValueError):
            log.debug("%s state unreadable", self.NAME, exc_info=True)
        return {}

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(self._state))
            os.replace(tmp, self._state_path)     # atomic: never half a file
        except OSError:
            log.debug("%s state save failed", self.NAME, exc_info=True)

    def _prune_seen(self) -> None:
        """Drop seen-ids older than KEEP_DAYS (values are UTC ISO strings),
        then hard-cap the file. Called by the seen-id watchers only."""
        cutoff = (self._now().astimezone(timezone.utc)
                  - timedelta(days=self.KEEP_DAYS)).isoformat()
        for key in [k for k, v in self._state.items()
                    if not isinstance(v, str) or v < cutoff]:
            self._state.pop(key, None)
        if len(self._state) > self.MAX_STATE:
            # oldest first; ISO strings sort chronologically
            for key in sorted(self._state,
                              key=lambda k: str(self._state[k]))[:len(self._state) - self.MAX_STATE]:
                self._state.pop(key, None)

    def _mark(self, key: str) -> None:
        self._state[key] = self._now().astimezone(timezone.utc).isoformat()

    # -------------------------------------------------------- announcing
    def _speak(self, title: str, text: str) -> bool:
        """One unprompted line. Failures never abort the tick: a Discord
        outage must not stop the rest of the batch being marked seen, or
        the next tick would say all of it again."""
        if not text or self._announce is None:
            return False
        try:
            self._announce(title, text)
        except Exception:                       # noqa: BLE001 - sink boundary
            log.exception("%s: announce failed", self.NAME)
            return False
        log.info("%s: %s", self.NAME, text)
        return True

    # ------------------------------------------------------------- tick
    def tick(self) -> int:
        raise NotImplementedError

    # ----------------------------------------------------------- thread
    def start(self) -> None:
        # Alive-guard + clear, like deadlines.py: a stopped instance can be
        # started again (tests, a config reload), and a dead thread must not
        # block a fresh one.
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=self.NAME)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _run(self) -> None:
        if self._stop.wait(self.FIRST_DELAY_S):
            return
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("%s tick failed", self.NAME)
            if self._stop.wait(self.INTERVAL_S):
                return
