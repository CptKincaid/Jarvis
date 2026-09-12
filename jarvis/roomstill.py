"""Is a room's OCCUPANCY a body, or a fixture the radar has locked onto?

HIS BUG, reported 2026-09-11: "Was also in my bedroom and he thought i was
out. there is something wrong with the priorities on what is considered
out." The priorities were defensible. The radar was not.

MEASURED THE SAME EVENING, on his own office LD2410, two fifteen-minute
recordings taken through scripts/room_trace.py:

    flat EMPTY (527 samples)   still distance held 306-313 cm
    him SEATED (514 samples)   still distance ranged 299-347 cm

THE WINDOW LENGTH CAME OUT OF THE DATA, and the first guess was wrong.
A rolling SPREAD, swept across window lengths at his measured 1.7 s
cadence:

    window    empty max    seated min    gap
      60 s            7             8     +1     <- unusable
      90 s            7            10     +3     <- unusable
     120 s            7            24    +17
     180 s            7            34    +27     <- chosen
     300 s            7            38    +31

One minute is NOT enough: his quietest minute at the desk came within a
centimetre of the empty room's widest, which is no margin at all. Three
minutes separates them by 27 cm, nearly five times the empty room's whole
range. The cost of the longer window is only how quickly a phantom is
dismissed, and a phantom that has run for hours can wait three minutes.

The threshold is the middle of the 180 s gap. Nothing here was chosen for
looking reasonable; every number is the smallest one his own two
recordings would support.

WHY THE SPREAD AND NOT THE BIT. His geometry is sensor -> open space ->
the back of his chair -> him -> desk and monitors -> a cement wall, and he
has nowhere else to mount it. Everything past the chair is a hard static
reflector, so the sensor reports OCCUPIED for ever: 527 of 527 samples
with the flat empty, and it read the same with him home in another room.
The BIT carries no information about him in that room. The DISTANCE does,
because a man breathes and shifts and a desk does not.

WHY NOT JUST SHORTEN ITS RANGE. Measured too, reversibly: dropping the
gate from 4 to 3 killed the phantom outright (0 of 18 samples). It also
cuts at 300 cm, and he reads 299-347 -- it would have cut off most of him.

WHAT THIS MUST NEVER DO is turn a real occupancy into an empty room on a
reading it could not take. Every unknown here returns UNKNOWN, and the
caller keeps whatever the sensor said. A false away is the error this
whole lane exists to prevent.
"""
from __future__ import annotations

from typing import Optional

from jarvis.logs import get_logger

log = get_logger("roomstill")

# The rolling window, in SECONDS of wall clock rather than in samples, so a
# fabric that changes its poll cadence does not silently change the
# measurement this module was calibrated against. Chosen by the sweep
# above, not by taste.
WINDOW_S = 180.0
# Below this, in centimetres, the return is a fixture. The middle of the
# 180 s gap: empty topped out at 7 cm, he never went below 34 cm.
SPREAD_CM = 20.0
# Enough of the window to judge on. At the fabric's 2 s cadence three
# minutes is 90 readings; 50 is a little over half of one and still about
# 100 s of evidence. Fewer than this is UNKNOWN, never "furniture".
MIN_SAMPLES = 50

PERSON = "person"
FIXTURE = "fixture"
UNKNOWN = "unknown"


class StillWindow:
    """One room's recent still-distance readings, and what they mean.

    Pure: it holds numbers and takes its clock from the caller, so the
    tests drive it with his measured traces rather than with a sleep.
    """

    def __init__(self, window_s: float = WINDOW_S,
                 spread_cm: float = SPREAD_CM,
                 min_samples: int = MIN_SAMPLES):
        self.window_s = float(window_s)
        self.spread_cm = float(spread_cm)
        self.min_samples = int(min_samples)
        self._seen: list[tuple[float, float]] = []     # (at, cm)

    def add(self, cm, at: float) -> None:
        """Record one reading. A None or unparseable cm is DROPPED, not
        stored as a zero -- a missing reading must not look like a
        fixture sitting at the origin."""
        try:
            value = float(cm)
        except (TypeError, ValueError):
            return
        if value <= 0.0:
            return
        try:
            when = float(at)
        except (TypeError, ValueError):
            return
        self._seen.append((when, value))
        cutoff = when - self.window_s
        self._seen = [(t, v) for t, v in self._seen if t >= cutoff]

    def clear(self) -> None:
        """Forget everything -- for a run that has ended. A new run's
        readings must not be judged against the old run's."""
        self._seen = []

    @property
    def samples(self) -> int:
        return len(self._seen)

    def spread(self) -> Optional[float]:
        """Centimetres between the nearest and furthest recent reading,
        or None when there is not enough to say."""
        if len(self._seen) < self.min_samples:
            return None
        values = [v for _t, v in self._seen]
        return max(values) - min(values)

    def verdict(self) -> str:
        """PERSON, FIXTURE or UNKNOWN. Only FIXTURE is ever acted on."""
        spread = self.spread()
        if spread is None:
            return UNKNOWN
        return FIXTURE if spread < self.spread_cm else PERSON


def occupancy_is_real(verdict: str, occupied: Optional[bool]) -> Optional[bool]:
    """What the room should be READ as, given the sensor and the window.

    The asymmetry is the whole point and it is the same one this lane
    states everywhere else: only a FIXTURE verdict may take an occupancy
    away, and nothing here may ever manufacture one.

      * the sensor says empty          -> empty, whatever the window says
      * the sensor says occupied, and
          the window says FIXTURE      -> False: the radar is locked on
          anything else                -> unchanged
    """
    if occupied is not True:
        return occupied
    if verdict == FIXTURE:
        return False
    return True
