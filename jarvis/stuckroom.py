"""A room that reads occupied and that nothing else ever agrees with.

WHY THIS SURVIVES HIS LOGIC RATHER THAN BEING REPLACED BY IT. His three
legs do outvote a latched sensor -- when the other two are honest. On
2026-09-05 they were not: his phone was out with him (NO) and the camera
had been stopped since 17:04:25 (BLIND, not "looked and saw nobody"). That
is cell 6 of the voter, the one cell his three rules cannot reach, because
rule 1 needs a camera that LOOKED. With one leg silent and one leg latched
there is no majority, and something has to break the tie. The only thing
that can is history: has anything ever agreed with this ON run? That
question is this module. Without it, tonight's exact chain still ends in
silence after his logic ships.

And the same cell is the NAPPING-PHONE-AT-THE-DESK case, where the truth
is the reverse -- he is sitting still in the office with his phone asleep.
The three legs read identically in both. Only the run's history separates
them, which is why the answer had to be a clock and not another sensor.

WHAT COUNTS AS CORROBORATION -- any one of three, all of which the tree
already produces:

  * the phone probe answered present (an ARP hit or a ping reply)
  * the camera returned an identity or faces >= 1
  * the mic heard a turn (the turn ledger arrival.departure_ready already
    reads for its own 10-minute veto)

CORROBORATION IS HOUSE-LEVEL, NOT PER ROOM, and deliberately so. The phone
answering says "he is in the flat", not "he is in the office"; only the
camera could say the latter, and only for the room it points at. Crediting
every open run is the CONSERVATIVE direction -- it makes this detector
fire LESS -- and for a detector whose whole risk is crying wolf on a man
who is simply working, less is the right way to be wrong.

WHY 45 MINUTES DOES NOT CRY WOLF, and it is not a duration test. The phone
alone is polled every 60 s while home. A man at his desk with his phone in
his pocket corroborates roughly 45 times in 45 minutes. ZERO hits in 45
minutes is not a quiet afternoon; it is a leg that is not being told the
truth. The number itself is a starting point and is labelled as one --
see FAULT_AFTER_S_PROVENANCE.

THE CLOCK MUST OUTLIVE THE PROCESS, and this is the measured reason the
detector that already exists has never once fired. ``roomfabric.Room``
keeps ``true_since`` in memory on ``time.monotonic``, reset on every
start. On 2026-09-05 Jarvis started NINE times (01:45, 11:58, 12:20,
12:34, 12:50, 12:53, 13:22, 13:59, 20:22); the longest uptime was 10.2 h
and the uptime before the incident 6.38 h. A 12 h in-memory threshold has
never been reachable on this box and never will be at that restart
cadence -- and the 20:22 restart did exactly the damage that implies: it
reset the office run to zero 21 minutes before he walked in. So this
module keeps WALL CLOCK stamps in a small state file beside
``faults.json``, and restarting Jarvis cannot launder a stuck sensor.

WHAT IT DOES, AND WHAT IT MUST NEVER DO.

  DOES  drop the room from the voter's rooms leg, so the house falls
        through to the phone -- the honest leg, and the one that would
        have saved tonight (``presencevote.rooms_leg(faulted=...)``).
  DOES  raise a standing fault on the fault board (jarvis/faults.py),
        which persists, unlike a four-second status chip.
  DOES  say it once per run.
  NEVER publish Presence(home=False). A BROKEN SENSOR IS NOT EVIDENCE
        ABOUT WHERE HE IS. It removes a vote; it does not cast one. There
        is no method on this class that can express "he is out", and
        tests/test_stuckroom.py pins that.
  NEVER fire a greeting, directly or by arming DoorWatch. Dropping the
        room can make the voter reach "away" on the PHONE's evidence,
        which is legitimate; the fault itself is not a path to speech.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from jarvis.config import PATHS
from jarvis.events import FaultRaised
from jarvis.logs import get_logger

log = get_logger("stuckroom")

RULE = "stuckroom"

# 45 minutes. GUESSED, NOT MEASURED -- there is no figure for how long he
# sits at that desk with his phone out of range and the mic silent, so
# this is a starting point to be revised from his own logs. Said out loud
# here because this project has twice been damaged by a confident number
# with no source.
FAULT_AFTER_S = 45 * 60.0
FAULT_AFTER_S_PROVENANCE = (
    "GUESSED, not measured: no ground truth exists for how long he sits at "
    "the desk with his phone out of range and the mic silent. Revise from "
    "his own logs.")

# The duration-only fallback for a box with no other leg at all, lowered
# from 12 h. MEASURED justification: on 2026-09-05 the office was the
# active room with no office/kitchen transition from 17:01:27 to 18:43:19
# -- 1 h 42 min of genuinely sitting still. A threshold at or under 2 h
# cries wolf on a man who is simply working; 4 h clears that session by
# 2.3x. (It is still a blunt instrument: the corroboration clock above is
# the real detector, and this only exists for a box with nothing else.)
DEFAULT_STUCK_AFTER_H = 4.0


def state_path() -> Path:
    return PATHS.MEMORY_DIR / "roomruns.json"


@dataclass
class _Run:
    """One room's current occupied run, in WALL CLOCK seconds.

    ``corroborated`` is 0.0 until something independent has agreed, and
    that zero is meaningful: it is the difference between "he has been at
    his desk for an hour" and "this radar has been lying for an hour".
    """

    started: float = 0.0
    corroborated: float = 0.0
    faulted: bool = False
    said: bool = False

    def clock_from(self) -> float:
        """The moment the uncorroborated stretch began."""
        return max(self.started, self.corroborated)


class StuckRooms:
    """Per-room uncorroborated occupied runs. Thread-safe enough for one
    poll loop; owns no thread and no socket.

    ``publish`` takes a ``FaultRaised``. ``now`` is wall clock on purpose:
    see the module docstring on why a monotonic clock made the existing
    detector unfireable.
    """

    def __init__(self, path: Optional[Path] = None,
                 publish: Optional[Callable] = None,
                 now: Callable[[], float] = time.time,
                 fault_after_s: float = FAULT_AFTER_S):
        self.path = Path(path) if path is not None else state_path()
        self._publish = publish
        self._now = now
        self.fault_after_s = float(fault_after_s)
        self.runs: dict = {}
        self._load()

    # ------------------------------------------------------------ state
    def _load(self) -> None:
        """Never raises: an unreadable state file costs the history and
        nothing else. A lost run only delays a fault by its own length."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for room, raw in dict(data.get("runs") or {}).items():
                self.runs[str(room)] = _Run(
                    started=float(raw.get("started") or 0.0),
                    corroborated=float(raw.get("corroborated") or 0.0),
                    faulted=bool(raw.get("faulted")),
                    said=bool(raw.get("said")))
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - corrupt, unreadable, wrong shape
            log.warning("stuckroom: %s is unreadable; starting with no run "
                        "history", self.path, exc_info=True)

    def _save(self) -> None:
        """Atomic, and never raises -- faults.py's own pattern."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"runs": {
                room: {"started": r.started, "corroborated": r.corroborated,
                       "faulted": r.faulted, "said": r.said}
                for room, r in self.runs.items()}}, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 - state is a nicety, not the feature
            log.debug("stuckroom: could not write %s", self.path, exc_info=True)

    # ------------------------------------------------------- the inputs
    def corroborate(self, source: str = "", at: Optional[float] = None) -> None:
        """Something independent agreed that somebody is here.

        Stamps EVERY open run (see the module docstring on why house-level
        corroboration is the conservative choice).
        """
        at = self._now() if at is None else float(at)
        touched = False
        for room, run in self.runs.items():
            if run.started and run.corroborated < at:
                run.corroborated = at
                touched = True
                if run.faulted:
                    self._clear(room, run,
                                "%s corroborated it" % (source or "a leg"))
        if touched:
            self._save()

    def observe(self, room: str, occupied: Optional[bool],
                at: Optional[float] = None) -> Optional[str]:
        """One room reading. Returns the room name when a fault fires now.

        ``None`` is no opinion and neither starts nor ends a run -- the
        same rule ``roomfabric.Room.observe`` uses, and for the same
        reason: a two-poll network hiccup is not an edge.
        """
        at = self._now() if at is None else float(at)
        room = str(room)
        if occupied is None:
            return None
        run = self.runs.get(room)
        if not occupied:
            if run is not None and run.started:
                if run.faulted:
                    self._clear(room, run, "the room is reading empty again")
                self.runs.pop(room, None)
                self._save()
            return None
        if run is None or not run.started:
            run = _Run(started=at)
            self.runs[room] = run
            self._save()
            return None
        if run.faulted:
            return None
        if at - run.clock_from() >= self.fault_after_s:
            run.faulted = True
            self._save()
            self._raise(room, run, at)
            return room
        return None

    # ------------------------------------------------------- the outputs
    def faulted(self) -> frozenset:
        """The rooms the voter must DROP. Not a verdict about him."""
        return frozenset(r for r, run in self.runs.items() if run.faulted)

    def status(self, now: Optional[float] = None) -> dict:
        """Numbers only, for the instrument and the setup sheet."""
        now = self._now() if now is None else float(now)
        out = {}
        for room, run in self.runs.items():
            out[room] = {
                "run_s": round(now - run.started, 1) if run.started else 0.0,
                "corroborated_s_ago": (round(now - run.corroborated, 1)
                                       if run.corroborated else None),
                "faulted": run.faulted,
            }
        return out

    # -------------------------------------------------------- the fault
    def _raise(self, room: str, run: _Run, at: float) -> None:
        mins = int((at - run.clock_from()) / 60.0)
        text = ("the %s sensor has read occupied for %d minutes with "
                "nothing agreeing -- not your phone, not the camera, not "
                "the mic" % (room, mins))
        line = ("The %s sensor has been reading occupied for %d minutes and "
                "nothing else agrees, so I have stopped counting its vote. A "
                "fan or a curtain inside the beam is the usual cause."
                % (room, mins))
        if not run.said:
            log.warning("stuckroom: %s", text)
            run.said = True
            self._save()
        self._emit(FaultRaised(rule=RULE, kind="error",
                               token=("%s ON" % room.upper())[:10],
                               text=text, line=line))

    def _clear(self, room: str, run: _Run, why: str) -> None:
        run.faulted, run.said = False, False
        log.info("stuckroom: %s is back in the picture (%s)", room, why)
        self._emit(FaultRaised(rule=RULE, cleared=True,
                               token=("%s ON" % room.upper())[:10],
                               text="the %s sensor is answering normally "
                                    "again" % room))

    def _emit(self, event) -> None:
        if self._publish is None:
            return
        try:
            self._publish(event)
        except Exception:  # noqa: BLE001 - the bus must not break the detector
            log.debug("stuckroom: publish failed", exc_info=True)
