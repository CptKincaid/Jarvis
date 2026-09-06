"""Event bus for Jarvis V3.

Worker threads publish; subscribers run on the Tk main thread once a root is
attached (the bus pumps its queue via root.after every 30ms). Before a root is
attached — and in tests — call drain() to deliver synchronously.

No module outside jarvis/ui may import tkinter; this bus is the only bridge.
"""
from __future__ import annotations

import queue
import threading
import time
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
    # Publisher-side clock: the bus queues events for the Tk thread when
    # the UI is attached, so a subscriber's own clock reads drain time.
    t: float = field(default_factory=time.monotonic)


@dataclass
class RecordingStopped(Event):
    reason: str = "manual"            # manual | silence | cap | abort
    endpoint: str = ""                # which detector ended it: vad | energy | voice_id | manual | cap
    dead_air_s: float | None = None   # silence waited through before stopping (turn ledger)
    # Filler holds this capture (jarvis/recorder.py): pauses after a trailing
    # "um"/"uh" the stop waited CONFIG.filler_hold_s longer for. Reaches the
    # turn ledger line as holds=N so a week of turns can say how often it fired.
    filler_holds: int = 0
    # Spelling holds this capture (jarvis/spelling.py): pauses between two
    # spoken CHARACTERS the stop waited CONFIG.spell_hold_s longer for.
    # Counted apart from filler_holds so the ledger's holds=N keeps meaning
    # what it meant, and so a spelled address cannot eat his three ums.
    spell_holds: int = 0
    # The session was opened without a wake word (the follow-up window).
    # The app's "did not catch that" policy stays silent on those: nothing
    # said into a follow-up window is the normal case, not a lost turn.
    followup: bool = False
    # Publisher-side clock: the bus queues events for the Tk thread when
    # the UI is attached, so a subscriber's own clock reads drain time.
    t: float = field(default_factory=time.monotonic)


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
    # The result came from a decode started during the endpoint silence,
    # before the recorder stopped (JarvisApp._maybe_speculate). The turn
    # ledger notes it so a short "stt" figure is never mistaken for a
    # faster model.
    speculative: bool = False
    # Publisher-side clock: the bus queues events for the Tk thread when
    # the UI is attached, so a subscriber's own clock reads drain time.
    t: float = field(default_factory=time.monotonic)


@dataclass
class HotwordDetected(Event):
    score: float = 0.0
    # Publisher-side clock: the bus queues events for the Tk thread when
    # the UI is attached, so a subscriber's own clock reads drain time.
    t: float = field(default_factory=time.monotonic)


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
    # Correlation for turns whose replies stream back over a channel (the
    # command socket): "" = untagged, today's behaviour everywhere else.
    turn_id: str = ""


@dataclass
class BrainState(Event):
    state: str = "idle"               # idle | thinking
    turn_id: str = ""                 # see JarvisReply.turn_id


@dataclass
class SpeakingState(Event):
    active: bool = False
    amplitude: float = 0.0            # 0..1, streamed ~12Hz while active
    # An amplitude-only tick: read `amplitude`, IGNORE `active`.
    #
    # 2026-09-02 08:56:19. The TTS amplitude feeder signs a chunk off by
    # publishing amplitude 0.0 so the mouth closes in the gap before the
    # next chunk renders; it had to send active=True to avoid handing every
    # subscriber a false end-of-speech mid-burst. Its 80 ms sleeps drift
    # ~1 %/chunk behind the player, so on a long line that sign-off landed
    # AFTER the worker's real active=False -- and all five subscribers that
    # keep `active` as a level (mixer, roomtone, reactor, main_window's
    # pill, app._tts_active) latched True with no falling edge left in the
    # world. His music stayed at 30 %, the board read "Speaking" while he
    # typed, and _tts_active being stuck True disables the nudge and the
    # guest decline for the rest of the boot.
    #
    # The tick is neither a rising nor a falling edge, so it now says so and
    # nobody has to guess. Only the TTS worker publishes edges.
    amplitude_only: bool = False
    # Publisher-side clock: the bus queues events for the Tk thread when
    # the UI is attached, so a subscriber's own clock reads drain time.
    t: float = field(default_factory=time.monotonic)


@dataclass
class ReminderFired(Event):
    text: str = ""
    # Added for the focus session (jarvis/focus.py): a silent item (label
    # prefixed timekeeper.SILENT_PREFIX) is never spoken or toasted by the
    # timekeeper itself, so the owner needs the id to know WHICH of its items
    # fired; `late` says it fired from the boot catch-up rather than on time.
    item_id: str = ""
    kind: str = ""                    # timer | reminder ("" from older publishers)
    silent: bool = False
    late: bool = False


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
class RunProgress(Event):
    """A training run's lifecycle, from the run ledger (jarvis/runwatch.py).

    `kind`: started | progress | finished. Published on CHANGE ONLY --
    never a heartbeat -- from the health watchdog's existing 30 s tick.
    `label` is the script ("finetune_piper.py"), `elapsed_s` the run's
    true age (read from /proc/<pid>/stat, so it survives a Jarvis restart
    mid-run), `line` the sentence he said about it (empty when he stayed
    quiet: a brief run, or "quietly, please").

    This is the board's live-tail lane contract. The lane is OPTIONAL: the
    default degrade path is the spoken beats alone, so a renderer that
    never appears costs nothing.
    """
    kind: str = "started"             # started | progress | finished
    pid: int = 0
    label: str = ""
    elapsed_s: float = 0.0
    epoch: int = 0
    loss: float = 0.0                 # 0.0 when the tail carried none
    line: str = ""


@dataclass
class FaultRaised(Event):
    """Something is wrong with the box and it is NEW (jarvis/faults.py).

    Published by the health watchdog beside its Status, because a Status
    is a 4-6 second chip: main_window holds a warn for WARN_HOLD_S and an
    error for ERROR_HOLD_S, so a fault raised while the room is empty
    leaves no trace outside jarvis.log. This event is what the board's
    FAULT lane latches on to.

    `rule` names the detector (memory | hogs | trainers), `token` is the
    <=10-character card text ("2 TRAINERS"), `text` the status sentence,
    `line` what he actually said. `cleared=True` lifts the fault for that
    rule -- the watchdog publishes one on recovery, and the board must
    take the clear from the detector rather than latching a second time.
    """
    rule: str = ""
    kind: str = "warn"                # warn | error
    token: str = ""
    text: str = ""
    line: str = ""
    cleared: bool = False


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
class Presence(Event):
    """The phone came onto / left the Wi-Fi (jarvis/presence.py). Published
    only on a transition; `returned` is True on away -> home, which is the
    one the app greets ("Welcome back, sir") and reads the held lines on."""
    home: bool = True
    since: float = 0.0                # time.time() of the transition
    returned: bool = False


@dataclass
class SensingChanged(Event):
    """Offline mode changed (jarvis/sensing.py). Published by the spoken
    switch so the console badge turns over in the same breath as the reply;
    the window ALSO re-reads the policy on its own 5 s pass, because the
    21:00 curfew edge arrives with nobody saying anything."""
    camera: bool = True
    radar: bool = True
    offline: bool = False
    reason: str = ""                  # "" | offline | timed | curfew | failsafe
    until: float | None = None        # a timed offline's end, epoch seconds


@dataclass
class RoomChanged(Event):
    """He moved between rooms (jarvis/roomfabric.py). Published only when the
    ACTIVE room changes, never on every reading -- the fabric polls three
    radars every two seconds and a per-tick event would be a metronome on
    the bus. `previous` is "" on the first sighting of a session, and the
    change has already survived the enter hold and the switch floor, so a
    subscriber may treat it as settled rather than debouncing it again."""
    room: str = ""                    # the room name from presence.rooms[].name
    label: str = ""                   # what to say out loud
    previous: str = ""
    at: float = 0.0                   # time.time() of the change


@dataclass
class DeskState(Event):
    """He sat down at / walked away from the keyboard (jarvis/deskpresence.py,
    GNOME's Mutter idle monitor). Published only on a threshold crossing;
    `returned` is True on away -> at-desk, which the app greets through the
    SAME handler as Presence so the two probes cannot both say "Welcome
    back, sir". `idle_s` is the reading that crossed the threshold."""
    at_desk: bool = True
    idle_s: float = 0.0
    since: float = 0.0                # time.time() of the transition
    returned: bool = False


@dataclass
class BriefingReady(Event):
    """A briefing was produced: `sections` renders as ONE transcript card
    (the reply card for that turn — no separate JarvisReply card), `spoken`
    is what the app says."""
    sections: dict = field(default_factory=dict)
    spoken: str = ""
    turn_id: str = ""                 # see JarvisReply.turn_id


@dataclass
class ArcChanged(Event):
    """The house moved to a new hour (jarvis/arc.py). `phase` is one of
    arc.PHASES; `forced` says an override (quiet hours, DND, an empty room,
    a focus block) named it rather than the sun. Published ONLY on an
    accepted transition -- never per tick -- so a subscriber may treat every
    event as a real change. The arc itself does nothing with it: everything
    that is felt lives in a consumer."""
    phase: str = ""
    previous: str = ""
    since: float = 0.0                # time.time() the phase was adopted
    forced: bool = False
    sunrise: float = 0.0              # 0.0 when unknown (no coords, polar)
    sunset: float = 0.0


# ------------------------------------------------------------------
# The Board and the console's ambient modes (2026-08-30). The Board is a
# SECOND borderless surface (jarvis/ui/board.py); its state is composed off
# the Tk thread by jarvis/board.board_state() and published here so the bus
# does the thread marshalling it exists for.
# ------------------------------------------------------------------
@dataclass
class BoardUpdate(Event):
    """A fresh Board state from the 5 s poll thread (BoardFeed). `state` is
    a jarvis.board.BoardState — typed loosely here so events.py keeps its
    "no imports from feature modules" shape."""
    state: object = None


@dataclass
class BoardCommand(Event):
    """A spoken instruction FOR the Board surface: "bring up the board",
    "close the board", "focus on the sessions".

    A command rather than a fact, deliberately. The commander runs on a
    worker thread and the app must never hold a reference to the window
    (CLAUDE.md: modules publish, the window subscribes), so this is the one
    honest way for a voice command to reach a Tk surface. The SPOKEN read
    is composed app-side and is not carried here."""
    action: str = "show"              # show | hide | focus
    panel: str = ""                   # focus: the resolved panel key


@dataclass
class PowerUp(Event):
    """First activity at the desk after a long overnight gap: the console
    (and the Board, when it is up) play the staged population sweep, once a
    day. Published by the app, which owns the date latch in
    briefing_state.json — the UI must never decide what day it is."""
    reason: str = ""                  # presence | hotword
    gap_h: float = 0.0                # how long the machine was idle


@dataclass
class ClearTranscript(Event):
    """"Clear the transcript": empty the console's conversation pane.

    A command rather than a fact, for the reason BoardCommand carries the
    same shape: the commander runs on worker threads and must never hold a
    reference to a Tk surface, so this is the one honest way for a spoken
    verb to reach one.

    THE SCREEN ONLY. Nothing here touches the conversation the model sees
    (jarvis/memory.py, the context engine) -- the pane is glass, the memory
    is not, and the two must never be cleared by the same event. The spoken
    line says which he got."""
    pass
