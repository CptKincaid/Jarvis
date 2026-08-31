"""The fault lane: one place that knows the box is in trouble, and says so once.

The detectors already exist and are already latched. ``health.Watchdog``
raises memory / hogs / trainer alerts once per episode (``_level``,
``_hogs_alerted``, the ``REARM_MARGIN_GB`` hysteresis) and publishes a
``Status``. What was missing is PERSISTENCE: ``main_window.set_status``
holds a warn chip for 4 s and an error for 6 s, so a fault that fires
while he is out of the room evaporates with no trace outside jarvis.log.

This module is the store behind the board's FAULT row:

``FaultBoard``  Tk-free, thread-safe. Subscribes to ``FaultRaised`` and
                keeps the newest live fault until a ``cleared`` event
                lifts it. ``token`` is what the engine card shows (ten
                characters of mono; "2 TRAINERS" fits, a sentence does
                not), ``describe()`` is what he says when asked.
``FaultLog``    The state file (``PATHS.MEMORY_DIR/faults.json``, atomic
                os.replace like jarvis/deadlines.py). It answers ONE
                question: has this exact fault already been spoken
                recently? The watchdog's latches only live as long as the
                process, so a restart into a still-tight pool would
                announce the same episode again -- and "he told me twice"
                is how a warning stops being believed.

Deliberately NOT a second latch on the watchdog's rules (a UI-side latch
drifts out of sync on recovery; ``check()`` publishes an ok Status when
the pool recovers, and that is the clear signal). Deliberately no polling
of its own: the fault lane rides the watchdog's existing 30 s tick, which
calls ``snapshot(gpu=False)`` -- the wedge this feature exists to warn
about is exactly the state in which nvidia-smi blocks in D-state.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from jarvis.config import PATHS
from jarvis.events import FaultRaised, bus
from jarvis.logs import get_logger

log = get_logger("faults")

# The engine card draws its values in one mono font, ellipsized to
# ~104 design px: about ten characters. Longer tokens are cut here rather
# than by the canvas, so what survives is the front of the token.
TOKEN_CHARS = 10
CLEAR_TOKEN = "--"
# The same fault spoken again inside this window is not news. Only
# consulted across restarts in practice -- within one process the
# watchdog's own latches have already stopped the repeat.
SPOKEN_COOLDOWN_S = 3600.0
STATE_VERSION = 1

NOTHING_WRONG_LINE = "Nothing wrong that I can see, sir."


def state_path() -> Path:
    return PATHS.MEMORY_DIR / "faults.json"


@dataclass
class Fault:
    """One live fault. ``rule`` is the detector (memory | hogs | trainers),
    ``token`` the card text, ``line`` what he said about it."""
    rule: str = ""
    kind: str = "warn"                 # warn | error
    token: str = ""
    text: str = ""                     # the status-chip sentence
    line: str = ""                     # the spoken sentence
    since: float = field(default_factory=time.time)

    @property
    def signature(self) -> str:
        """What makes this fault the SAME fault as an earlier one: the
        rule and its token, never the free text (a memory warning whose
        gigabytes drifted by 0.1 is the same episode)."""
        return f"{self.rule}:{self.token}"


def token_for(rule: str, text: str) -> str:
    """A <= TOKEN_CHARS token for the FAULT row. The row can carry roughly
    a ten-character word, so it names the incident ("2 TRAINERS") and the
    sentence stays in the spoken line and the toast."""
    text = " ".join(str(text or "").split())
    if not text:
        return (rule or "FAULT")[:TOKEN_CHARS]
    lowered = text.lower()
    head = lowered.split()[0]
    if rule == "trainers" and head.isdigit():
        return f"{head} TRAINERS"[:TOKEN_CHARS]
    if rule == "hogs" and head.isdigit():
        return f"{head} HOGS"[:TOKEN_CHARS]
    if rule == "memory":
        # "Memory critical: 7.6 GB free" -> "MEM 7.6G"; the number is the
        # part he needs, so it survives the cut, not the word "memory".
        for word in text.split():
            bare = word.rstrip(":,")
            try:
                float(bare)
            except ValueError:
                continue
            return f"MEM {bare}G"[:TOKEN_CHARS]
        return "MEMORY"
    return text[:TOKEN_CHARS].strip().upper()


class FaultLog:
    """The spoken-once state file. Never raises: an unreadable or
    unwritable state file costs the de-duplication, not the warning."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else state_path()
        self._lock = threading.Lock()
        self._data: dict = {"version": STATE_VERSION, "spoken": {}}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:  # noqa: BLE001 - a corrupt file is not fatal
            log.warning("fault state unreadable; starting clean", exc_info=True)
            return
        if isinstance(raw, dict) and isinstance(raw.get("spoken"), dict):
            self._data = {"version": STATE_VERSION, "spoken": dict(raw["spoken"])}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self._data), encoding="utf-8")
            os.replace(tmp, self.path)     # atomic: a torn file loses the state
        except Exception:  # noqa: BLE001
            log.warning("fault state not written", exc_info=True)

    def should_speak(self, signature: str, now: Optional[float] = None,
                     cooldown: float = SPOKEN_COOLDOWN_S) -> bool:
        """True the first time this fault is seen, and again once the
        cooldown has passed. Marks it spoken as a side effect -- callers
        speak on True and stay silent on False, so the mark belongs here
        rather than in a second call the caller can forget."""
        now = time.time() if now is None else float(now)
        if not signature:
            return True
        with self._lock:
            last = self._data["spoken"].get(signature)
            try:
                fresh = last is not None and (now - float(last)) < float(cooldown)
            except (TypeError, ValueError):
                fresh = False
            if fresh:
                log.info("fault %s already spoken %.0f s ago; staying quiet",
                         signature, now - float(last))
                return False
            self._data["spoken"][signature] = now
            self._prune(now, cooldown)
            self._save()
        return True

    def clear(self, signature: str) -> None:
        """The episode ended: the next occurrence is news again."""
        with self._lock:
            if self._data["spoken"].pop(signature, None) is not None:
                self._save()

    def _prune(self, now: float, cooldown: float) -> None:
        cutoff = now - max(float(cooldown), SPOKEN_COOLDOWN_S) * 4
        stale = [k for k, v in self._data["spoken"].items()
                 if not isinstance(v, (int, float)) or v < cutoff]
        for key in stale:
            self._data["spoken"].pop(key, None)


class FaultBoard:
    """The live fault, held until it clears. One per process; the app
    parks it on ``services.faults`` and the window reads it at 1 Hz.

    Thread-safe because the bus delivers on the Tk thread while the
    commander reads it from the voice worker.
    """

    def __init__(self, subscribe: bool = False):
        self._lock = threading.Lock()
        self._fault: Optional[Fault] = None
        self._history: list = []
        if subscribe:
            self.subscribe()

    def subscribe(self):
        bus.subscribe(FaultRaised, self.on_event)
        return self

    # ------------------------------------------------------------ state
    @property
    def current(self) -> Optional[Fault]:
        with self._lock:
            return self._fault

    @property
    def token(self) -> str:
        """What the engine card's FAULT row shows: the token, or "--"."""
        fault = self.current
        return fault.token if fault is not None else CLEAR_TOKEN

    @property
    def kind(self) -> str:
        fault = self.current
        return fault.kind if fault is not None else "ok"

    def on_event(self, ev: FaultRaised) -> None:
        try:
            self.apply(ev)
        except Exception:  # noqa: BLE001 - the bus must not lose a subscriber
            log.exception("fault board rejected %r", getattr(ev, "rule", "?"))

    def apply(self, ev: FaultRaised) -> Optional[Fault]:
        """Fold one event in. A cleared event for the live fault's rule
        lifts it; any other cleared event is ignored, so a recovery on the
        memory rule cannot silently wipe a trainer warning."""
        with self._lock:
            if ev.cleared:
                if self._fault is not None and (not ev.rule or
                                                self._fault.rule == ev.rule):
                    log.info("fault cleared: %s", self._fault.signature)
                    self._fault = None
                return self._fault
            fault = Fault(rule=ev.rule or "fault", kind=ev.kind or "warn",
                          token=(ev.token or token_for(ev.rule, ev.text)).upper(),
                          text=ev.text or "", line=ev.line or "")
            # An error outranks a warning; a second warning of the same
            # rule refreshes the text but keeps `since` (one episode).
            live = self._fault
            if live is not None and live.kind == "error" and fault.kind != "error":
                return live
            if live is not None and live.signature == fault.signature:
                fault.since = live.since
            self._fault = fault
            self._history.append(fault)
            del self._history[:-20]
            return fault

    def clear(self) -> None:
        with self._lock:
            self._fault = None

    # ----------------------------------------------------------- words
    def describe(self, now: Optional[float] = None) -> str:
        """"What's wrong?" -- the live fault in his own words plus how
        long it has been standing. Empty when the board is clear, which is
        the caller's cue to fall through to the log triage."""
        fault = self.current
        if fault is None:
            return ""
        line = fault.line or fault.text
        now = time.time() if now is None else float(now)
        minutes = int(max(0.0, now - fault.since) // 60)
        if minutes >= 1:
            since = (f"That's been standing {minutes} minute"
                     f"{'s' if minutes != 1 else ''}, sir.")
            return f"{line} {since}".strip()
        return line
