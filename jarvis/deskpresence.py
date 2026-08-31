"""Desk presence: is he actually at the keyboard?

``jarvis/presence.py`` answers "is he in the building" from his phone's
Wi-Fi address. That probe is dead on this box -- ``presence.phone_ip`` and
``presence.phone_mac`` are both unset, so the sentinel logs "sentinel
idle" and the whole welcome-back / catch-up machine has never fired in the
room. This module answers the smaller, cheaper question that needs no
configuration at all: how long since the mouse or keyboard last moved.

GNOME's Mutter keeps that number and hands it out on the session bus
without a password::

    gdbus call --session --dest org.gnome.Mutter.IdleMonitor \\
        --object-path /org/gnome/Mutter/IdleMonitor/Core \\
        --method org.gnome.Mutter.IdleMonitor.GetIdletime
    -> (uint64 13478107,)          # milliseconds

Verified on the live session (rc=0, no sudo); the RUNNING app already
carries ``DBUS_SESSION_BUS_ADDRESS`` and ``DISPLAY`` in its environment, so
there is no new plumbing.

Three rules the rest of the app depends on:

* ``desk_idle_s()`` returns ``float | None``. A nonzero rc, a missing
  gdbus, a timeout or unparsable stdout is None -- "no signal" -- NEVER
  0.0, which would read as "sitting right there", and never "away". The
  Mutter interface is not guaranteed across sessions and a failure must
  degrade to silence, not to a wrong answer.
* The sentinel only ever SUPPRESSES. It never says "nobody's home" out
  loud: the away signal feeds the quiet policy's existing
  ``hold_when_away`` seam (``QuietPolicy(is_home=...)``) rather than a
  second suppression path, and the return is greeted through the app's ONE
  presence handler, so the phone probe and this one cannot both say
  "Welcome back, sir".
* Away is a threshold, not hysteresis: 25 minutes of no keyboard or mouse
  by default (``presence.desk_away_after_min``). Idle time is
  keyboard/mouse only -- reading a paper at the desk looks like an empty
  chair -- so the threshold is generous and the consequences are gentle (a
  held line, a dimmed board), never a spoken claim.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Callable, Optional

from jarvis.events import DeskState, bus
from jarvis.logs import get_logger

log = get_logger("deskpresence")

DEFAULT_AWAY_MIN = 25
DEFAULT_POLL_S = 30.0
# JARVIS_DESK_PRESENCE=0 switches the sentinel off for a whole process.
# The test suite (tests/conftest.py) sets it, because unlike every other
# state this repo firewalls, the idle monitor is not a file that can be
# redirected: a test that builds the real App would read the DEVELOPER'S
# live session bus and, with the chair empty for the afternoon, quietly
# start holding proactive speech in unrelated tests.
ENV_OFF = "JARVIS_DESK_PRESENCE"
_OFF_VALUES = ("0", "off", "false", "no")
CALL_TIMEOUT_S = 4.0
IDLE_ARGV = [
    "gdbus", "call", "--session",
    "--dest", "org.gnome.Mutter.IdleMonitor",
    "--object-path", "/org/gnome/Mutter/IdleMonitor/Core",
    "--method", "org.gnome.Mutter.IdleMonitor.GetIdletime",
]
# "(uint64 13478107,)" -- the only shape GetIdletime returns.
_IDLE_RX = re.compile(r"uint64\s+(\d+)")


def _run_default(argv, timeout):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def parse_idletime(stdout) -> Optional[float]:
    """gdbus stdout -> seconds idle, or None when it is not an answer."""
    m = _IDLE_RX.search(str(stdout or ""))
    if not m:
        return None
    try:
        return int(m.group(1)) / 1000.0
    except ValueError:                       # a 30-digit number, say
        return None


def desk_idle_s(run: Callable = _run_default) -> Optional[float]:
    """Seconds since the last keyboard or mouse event, or None.

    ``run(argv, timeout=)`` is the seam (tests pass a fake returning an
    object with ``returncode`` / ``stdout``). None means "no signal" and
    every consumer must treat it as unknown, never as away.
    """
    try:
        res = run(IDLE_ARGV, timeout=CALL_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - no gdbus, no session bus, a timeout
        log.debug("desk: gdbus call failed", exc_info=True)
        return None
    if getattr(res, "returncode", 1) != 0:
        return None
    return parse_idletime(getattr(res, "stdout", ""))


def _cfg_get(cfg, key, default=None):
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            value = get(key, default)
            return default if value is None else value
        except Exception:  # noqa: BLE001 - a config hiccup must not mute him
            log.debug("desk: cfg.get(%s) failed", key, exc_info=True)
    return default


class DeskSentinel:
    """Polls ``desk_idle_s`` and publishes ``DeskState`` at the threshold.

    ``at_desk`` is None until a reading lands, then True/False. Both
    ``is_at_desk()`` and ``idle_s()`` fail OPEN (True / None): a session
    without the Mutter interface must behave exactly like today's app.
    """

    def __init__(self, cfg, publish: Callable = bus.publish,
                 idle_fn: Callable = desk_idle_s,
                 now: Callable[[], float] = time.time,
                 poll_s: Optional[float] = None):
        self._cfg = cfg
        self._publish = publish
        self._idle = idle_fn
        self._now = now
        self._poll_s = poll_s
        self.at_desk: Optional[bool] = None
        self.since: float = 0.0
        self.last_idle: Optional[float] = None
        self._logged_no_signal = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ config
    @property
    def enabled(self) -> bool:
        env = os.environ.get(ENV_OFF)
        if env is not None and env.strip().lower() in _OFF_VALUES:
            return False
        return bool(_cfg_get(self._cfg, "presence.desk", True))

    @property
    def away_after_s(self) -> float:
        try:
            mins = float(_cfg_get(self._cfg, "presence.desk_away_after_min",
                                  DEFAULT_AWAY_MIN))
        except (TypeError, ValueError):
            mins = DEFAULT_AWAY_MIN
        return max(60.0, mins * 60.0)

    @property
    def poll_s(self) -> float:
        if self._poll_s is not None:
            return float(self._poll_s)
        try:
            return max(5.0, float(_cfg_get(self._cfg, "presence.desk_poll_s",
                                           DEFAULT_POLL_S)))
        except (TypeError, ValueError):
            return DEFAULT_POLL_S

    # ------------------------------------------------------------- state
    @property
    def state(self) -> str:
        with self._lock:
            if self.at_desk is None:
                return "unknown"
            return "at-desk" if self.at_desk else "away"

    def is_at_desk(self) -> bool:
        """True unless a reading has established the chair is empty."""
        with self._lock:
            return self.at_desk is not False

    def idle_s(self) -> Optional[float]:
        """The last reading in seconds; None when there is no signal.
        This is what ``services.desk_idle_s()`` hands out."""
        with self._lock:
            return self.last_idle

    # -------------------------------------------------------------- tick
    def tick(self) -> Optional[DeskState]:
        """One reading; returns the DeskState published, if any."""
        if not self.enabled:
            return None
        try:
            idle = self._idle()
        except Exception:  # noqa: BLE001 - the loop must survive anything
            log.exception("desk: idle probe failed")
            return None
        if idle is None:
            # No signal is not absence: leave the state exactly as it was
            # (unknown on a session without Mutter) and say so once.
            if not self._logged_no_signal:
                self._logged_no_signal = True
                log.info("desk: no idle signal from the session bus; "
                         "desk presence is off for this session")
            return None
        self._logged_no_signal = False
        now, threshold = self._now(), self.away_after_s
        at_desk = idle < threshold
        with self._lock:
            self.last_idle = idle
            if self.at_desk is at_desk:
                return None
            returned = at_desk and self.at_desk is False
            self.at_desk, self.since = at_desk, now
            event = DeskState(at_desk=at_desk, idle_s=idle, since=now,
                              returned=returned)
        log.info("desk: %s after %.0f s idle%s",
                 "at the desk" if at_desk else "away", idle,
                 " (returned)" if event.returned else "")
        try:
            self._publish(event)
        except Exception:  # noqa: BLE001
            log.exception("desk: publish failed")
        return event

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        # Alive-guard + clear, like presence.py: a stopped instance can be
        # started again and a dead thread must not block a fresh one.
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.enabled:
            log.info("desk: presence.desk is off; sentinel idle")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="deskpresence",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            # a gdbus call in flight must not outlive stop_assistant
            t.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("desk: tick failed")
            self._stop.wait(max(0.01, self.poll_s))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
