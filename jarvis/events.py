"""Event bus for Jarvis V3.

Worker threads publish; subscribers run on the Tk main thread once a root is
attached (the bus pumps its queue via root.after every 30ms). Before a root is
attached — and in tests — call drain() to deliver synchronously.

No module outside jarvis/ui may import tkinter; this bus is the only bridge.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

from jarvis.logs import get_logger

log = get_logger("events")


# ---------------------------------------------------------------- events
@dataclass
class Event:
    pass


@dataclass
class AudioLevel(Event):
    level: float                      # 0..1 RMS
    waveform: list = field(default_factory=list)   # up to 64 bars, 0..1


@dataclass
class RecordingStarted(Event):
    pass


@dataclass
class RecordingStopped(Event):
    reason: str = "manual"            # manual | silence | cap | abort


@dataclass
class PartialText(Event):
    text: str = ""


@dataclass
class Transcribed(Event):
    text: str = ""
    confidence: float = 0.0
    speaker_score: float = 1.0
    accepted: bool = True
    reject_reason: str = ""           # "" | speaker | confidence


@dataclass
class HotwordDetected(Event):
    score: float = 0.0


@dataclass
class MicState(Event):
    available: bool = True
    device_name: str = ""


@dataclass
class Status(Event):
    text: str = ""
    kind: str = "info"                # ok | info | busy | warn | error


@dataclass
class UserUtterance(Event):
    text: str = ""
    source: str = "voice"             # voice | typed


@dataclass
class JarvisReply(Event):
    text: str = ""
    speak: bool = False


@dataclass
class BrainState(Event):
    state: str = "idle"               # idle | thinking


@dataclass
class SpeakingState(Event):
    active: bool = False
    amplitude: float = 0.0            # 0..1, streamed ~12Hz while active


@dataclass
class ReminderFired(Event):
    text: str = ""


@dataclass
class ModelInfo(Event):
    """Backend summary for the header chip, e.g. 'small · GPU fp16'."""
    text: str = ""


@dataclass
class AppQuit(Event):
    pass


# ------------------------------------------------------------------ bus
class Bus:
    _PUMP_MS = 30

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._subs: dict[type, list[Callable]] = {}
        self._lock = threading.Lock()
        self._root = None

    def subscribe(self, etype: type, fn: Callable) -> Callable:
        """Register fn(event) for events of etype (exact type). Returns fn."""
        with self._lock:
            self._subs.setdefault(etype, []).append(fn)
        return fn

    def unsubscribe(self, etype: type, fn: Callable):
        with self._lock:
            try:
                self._subs.get(etype, []).remove(fn)
            except ValueError:
                pass

    def publish(self, event: Event):
        """Thread-safe. Queued for the Tk thread when attached."""
        self._q.put(event)
        if self._root is None:
            # No UI yet (tests, headless): deliver immediately on this thread.
            self.drain()

    def attach_tk(self, root):
        self._root = root
        self._pump()

    def _pump(self):
        self.drain()
        try:
            self._root.after(self._PUMP_MS, self._pump)
        except Exception:
            self._root = None   # window gone; fall back to inline delivery

    def drain(self):
        while True:
            try:
                ev = self._q.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                subs = list(self._subs.get(type(ev), ()))
            for fn in subs:
                try:
                    fn(ev)
                except Exception:
                    log.exception("subscriber failed for %s", type(ev).__name__)


bus = Bus()


# ------------------------------------------------------------------
# Personal-assistant events (spec: docs/specs/2026-08-26-jarvis-personal-
# assistant.md, section 3). Published by worker threads; the UI and the
# app subscribe. Field names are the contract — do not rename.
# ------------------------------------------------------------------
@dataclass
class ClaudeTaskState(Event):
    """One Claude task's lifecycle. state: queued | running | waiting
    (permission question pending) | done | failed | cancelled."""
    project: str = ""                 # project slug, e.g. "jarvis"
    task_id: str = ""
    state: str = "running"
    text: str = ""                    # one-line human summary (transcript)


@dataclass
class ClaudeProgress(Event):
    """A compact transcript line from the stream-json parser, e.g.
    'Edit jarvis/router.py' or 'Tests: 272 passed'. milestone=True means
    the app also speaks `line` (speech is the app's job, never the UI's)."""
    project: str = ""
    task_id: str = ""
    line: str = ""
    milestone: bool = False


@dataclass
class ActiveProject(Event):
    """The active Claude project changed (status-bar chip, terminal
    button tooltip). Empty slug = no active project."""
    slug: str = ""
    path: str = ""


@dataclass
class ApprovalRequested(Event):
    """Claude asked for permission outside the allowed dirs; the app
    speaks `question`, the UI shows it with ALLOW / DENY."""
    request_id: str = ""
    question: str = ""                # persona sentence
    tool_name: str = ""
    detail: str = ""                  # the command / path being asked about
    project: str = ""


@dataclass
class ApprovalResolved(Event):
    request_id: str = ""
    allowed: bool = False
    source: str = ""                  # typed | voice | discord | ui | timeout | policy


@dataclass
class UncertainUtterance(Event):
    """The commander could not tell whether an utterance was meant for
    Jarvis. He asks aloud and the UI shows a card with YES / NO; the
    answer feeds Commander.resolve_uncertain, which also trains the
    intent classifier via log_feedback."""
    request_id: str = ""
    text: str = ""                    # the utterance in question
    question: str = ""                # what he actually says


@dataclass
class UncertainResolved(Event):
    request_id: str = ""
    yes: bool = False
    source: str = ""                  # ui | voice | superseded


@dataclass
class AlarmFired(Event):
    """An alarm (or timer/reminder promoted to ringing) started ringing.
    The UI shows the modal with DISMISS / SNOOZE; the timekeeper rings."""
    alarm_id: str = ""
    label: str = ""
    kind: str = "alarm"               # alarm | timer | reminder
    due_text: str = ""                # "7:00 am"


@dataclass
class AlarmStopped(Event):
    alarm_id: str = ""
    action: str = "dismiss"           # dismiss | snooze | timeout
    snooze_min: int = 0


@dataclass
class BriefingReady(Event):
    """A briefing was produced: `sections` renders as ONE transcript card
    (the reply card for that turn — no separate JarvisReply card), `spoken`
    is what the app says."""
    sections: dict = field(default_factory=dict)
    spoken: str = ""
