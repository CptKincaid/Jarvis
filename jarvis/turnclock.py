"""Per-turn timing ledger: where the seconds go between the wake word and
the first audio of the reply.

Every latency change so far was argued from ad-hoc log arithmetic (grep the
timestamps, subtract by hand). This makes the breakdown a first-class log
line, emitted once per voice turn:

    turn: wake→mic 61ms · speech 1.9s · dead-air 0.8s · stt 0.37s ·
          route 1.2s · tts 0.2s · wait 2.6s (stop=vad)

`wait` is the number the user feels: from the moment they stopped talking
to the moment Jarvis started. The same record is appended as JSON to
PATHS.LOG_DIR/turns.jsonl so a week of turns can be plotted, not eyeballed.

Pure and thread-agnostic: marks arrive from the hotword, recorder, STT,
commander and TTS threads; a lock orders them. The app owns the wiring
(JarvisApp._wire_turn_clock and the _turn_on_* handlers in jarvis/app.py);
nothing here imports the bus.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("turn")

# Stage order. A stage may be missing (a button press has no "wake"; a
# rejected clip never reaches "handle"), so the report prints what it has.
STAGES = ("wake", "mic", "speech_end", "stop", "stt", "handle", "filler", "audio")
STALE_S = 90.0          # a turn that never produced audio is dropped at the next wake


def _fmt(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.1f}s"


class TurnLedger:
    """Collect stage timestamps for the current voice turn and report once.

    ``clock`` is injectable for tests. ``emit`` receives the finished record
    (a dict) and is where logging / JSONL writing happens; the default writes
    the log line and appends to ``jsonl_path`` when given.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 emit: Optional[Callable[[dict], None]] = None,
                 jsonl_path=None):
        self._clock = clock
        self._emit = emit or self._default_emit
        self._jsonl_path = jsonl_path
        self._lock = threading.Lock()
        self._marks: dict[str, float] = {}
        self._notes: dict[str, str] = {}
        self._open = False
        # The last mark of ANY turn, kept across turns. This is the ledger's
        # answer to "when did the microphone last hear him?", and it is what
        # the departure cue (jarvis/arrival.py) vetoes on -- a sleeping phone
        # radio must never settle the room around a man who just spoke.
        self._last_mark: Optional[float] = None

    # ---------------------------------------------------------------- marks
    def mark(self, stage: str, at: Optional[float] = None, **notes: str) -> None:
        """Record ``stage`` now (or at ``at``, a clock value in the past --
        the recorder reports when speech ended after the fact). "wake" and
        "mic" open a turn; anything else with no open turn is ignored (typed
        commands, TTS from the speak queue, alarms -- not voice turns)."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        now = self._clock() if at is None else float(at)
        with self._lock:
            # A wake opens a turn; so does a mic open with no turn, or a mic
            # open while the previous turn has already had its own (the mic
            # button pressed again): whatever was open is reported as
            # superseded rather than swallowing the new turn's marks.
            if stage == "wake" or (stage == "mic" and
                                   (not self._open or "mic" in self._marks)):
                if self._open:
                    self._discard_locked("superseded")
                self._marks, self._notes, self._open = {}, {}, True
            if not self._open:
                return
            if stage == "audio" and "stop" not in self._marks:
                return                       # TTS from elsewhere: not this turn's answer
            if stage in self._marks and stage != "speech_end":
                return                       # first mark wins (first audio chunk)
            self._marks[stage] = now         # speech_end: the LAST one wins
            if self._last_mark is None or now > self._last_mark:
                self._last_mark = now
            self._notes.update({k: str(v) for k, v in notes.items()})
            if stage == "audio":
                self._finish_locked("audio")

    def abandon(self, reason: str) -> None:
        """The turn ended without a reply (rejected, ignored, no speech)."""
        with self._lock:
            if self._open:
                self._finish_locked(reason)

    @property
    def open(self) -> bool:
        with self._lock:
            return self._open

    @property
    def last_mark(self) -> Optional[float]:
        """Clock value of the most recent mark of any turn (None: never)."""
        with self._lock:
            return self._last_mark

    def idle_s(self) -> Optional[float]:
        """Seconds since the microphone last did anything (None: never)."""
        with self._lock:
            if self._last_mark is None:
                return None
            return max(0.0, self._clock() - self._last_mark)

    # -------------------------------------------------------------- report
    def _finish_locked(self, outcome: str) -> None:
        m = self._marks
        self._open = False
        rec = self.compute(m, outcome, self._notes)
        try:
            self._emit(rec)
        except Exception:
            log.exception("turn ledger emit failed")

    def _discard_locked(self, reason: str) -> None:
        if self._marks and self._clock() - min(self._marks.values()) < STALE_S:
            self._finish_locked(reason)
        self._open = False

    @staticmethod
    def compute(m: dict, outcome: str, notes: Optional[dict] = None) -> dict:
        """Durations between the marks that exist. Pure; used by tests."""
        def span(a, b):
            return (m[b] - m[a]) if a in m and b in m else None
        speech_end = m.get("speech_end", m.get("stop"))
        rec = {
            "outcome": outcome,
            "wake_to_mic": span("wake", "mic"),
            # from the mic opening to the last speech the endpointer saw
            "speech": (speech_end - m["mic"]) if "mic" in m and speech_end else None,
            # the silence the recorder waited through before stopping
            "dead_air": (m["stop"] - m["speech_end"]) if "stop" in m and "speech_end" in m else None,
            "stt": span("stop", "stt"),
            "route": span("handle", "audio") if "handle" in m else None,
            # a "still working on it" line was spoken before the answer
            "filler": (m["filler"] - speech_end) if "filler" in m and speech_end else None,
            "stt_to_handle": span("stt", "handle"),
            # what the user feels: from their last word to Jarvis's first
            "wait": (m["audio"] - speech_end) if "audio" in m and speech_end else None,
            "total": (m["audio"] - m["wake"]) if "audio" in m and "wake" in m else None,
        }
        rec.update({k: v for k, v in (notes or {}).items()})
        return rec

    @staticmethod
    def format(rec: dict) -> str:
        parts = [f"wake→mic {_fmt(rec.get('wake_to_mic'))}",
                 f"speech {_fmt(rec.get('speech'))}",
                 f"dead-air {_fmt(rec.get('dead_air'))}",
                 f"stt {_fmt(rec.get('stt'))}"]
        if rec.get("route") is not None:
            parts.append(f"route {_fmt(rec.get('route'))}")
        if rec.get("filler") is not None:
            parts.append(f"filler@{_fmt(rec.get('filler'))}")
        parts.append(f"wait {_fmt(rec.get('wait'))}")
        tail = " ".join(f"{k}={v}" for k, v in rec.items()
                        if k not in TurnLedger._NUMERIC and k != "outcome")
        outcome = rec.get("outcome", "")
        head = "turn" if outcome == "audio" else f"turn[{outcome}]"
        return f"{head}: " + " · ".join(parts) + (f" ({tail})" if tail else "")

    _NUMERIC = frozenset({"wake_to_mic", "speech", "dead_air", "stt", "route",
                          "filler", "stt_to_handle", "wait", "total"})

    def _default_emit(self, rec: dict) -> None:
        log.info("%s", self.format(rec))
        if self._jsonl_path is None:
            return
        try:
            rec = dict(rec, at=time.time())
            with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            log.debug("turns.jsonl append failed", exc_info=True)
