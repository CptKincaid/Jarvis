"""Is a room's OCCUPANCY a body, or a fixture the radar has locked onto?

HIS BUG, reported 2026-09-11: "Was also in my bedroom and he thought i was
out. there is something wrong with the priorities on what is considered
out." The priorities were defensible. The radar was not.

MEASURED THE SAME EVENING, on his own office LD2410, two fifteen-minute
recordings taken through scripts/room_trace.py:

    flat EMPTY (527 samples)   still distance held 306-313 cm
    him SEATED (514 samples)   still distance ranged 288-337 cm
                               (~/room-trace-AT-DESK.csv, office_still_cm)

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
cuts at 300 cm, and he reads 288-337 -- it would have cut off most of him.

JUDGED ON THE SPAN, NOT THE COUNT (the 2026-09-12 attack round). The first
draft judged as soon as MIN_SAMPLES readings existed, which is 50 x the
cadence -- about 100 s at 2 s polls, 12.5 s at 0.25 s -- and not the 180 s
the sweep above chose. Measured on his own at-desk recording, 4 of 465
run-start offsets called him a fixture on their first 50 readings (spread
8-10 cm in the first minute and a half of sitting down). So a verdict now
needs readings spanning at least SPAN_FRACTION of the window as well:
with the span rule, 0 of 423 offsets do, empty tops out at 7 cm and seated
bottoms out at 35 cm. MIN_SAMPLES stays as a DENSITY floor only, and the
fabric warns once if its poll cadence can never reach it. Trimming the
extremes to resist a single stray reading was measured and rejected: his
seated excursions ARE single readings, and dropping one each side pulls
the seated minimum from 35 cm to 26 cm.

THE WINDOW IS PRUNED BY THE CLOCK, not only by the next reading. A dead
distance entity used to freeze the last verdict for ever -- a FIXTURE
kept reading a man at his desk as empty on evidence hours old. ``expire``
lets the caller age the window every poll, so an unreadable distance
falls back to UNKNOWN, and the sensor's word, inside one window.

WHAT THIS MUST NEVER DO is turn a real occupancy into an empty room on a
reading it could not take. Every unknown here returns UNKNOWN, and the
caller keeps whatever the sensor said. A false away is the error this
whole lane exists to prevent.
"""
from __future__ import annotations

import math
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
# The DENSITY floor. At the fabric's 2 s cadence three minutes is 90
# readings; fewer than 50 across the span is too thin to trust (a 10 s poll
# would hand a seated man's 35 cm of movement to chance). Fewer than this
# is UNKNOWN, never "furniture".
MIN_SAMPLES = 50
# The span floor: the readings must cover this much of the window before
# anything is judged. 0.9 x 180 s = 162 s; the sweep already separated the
# two recordings by 17 cm at 120 s, and at 162 s+ by 28 cm.
SPAN_FRACTION = 0.9

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
        """Record one reading. A None, unparseable, non-finite or boolean
        cm is DROPPED, not stored -- a missing reading must not look like
        a fixture sitting at the origin, and a flag is not a distance."""
        if isinstance(cm, bool):
            return
        try:
            value = float(cm)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or value <= 0.0:
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

    def expire(self, at: float) -> None:
        """Age the window by the clock alone. Called every poll whether or
        not a reading arrived, so a verdict never outlives its evidence."""
        try:
            cutoff = float(at) - self.window_s
        except (TypeError, ValueError):
            return
        self._seen = [(t, v) for t, v in self._seen if t >= cutoff]

    @property
    def samples(self) -> int:
        return len(self._seen)

    def span(self) -> float:
        """Seconds between the oldest and newest reading held."""
        seen = self._seen
        return (seen[-1][0] - seen[0][0]) if len(seen) > 1 else 0.0

    def spread(self) -> Optional[float]:
        """Centimetres between the nearest and furthest recent reading,
        or None when there is not enough to say -- too few readings, or
        readings that do not yet cover the window."""
        seen = self._seen          # one snapshot: add()/clear() rebind, never mutate
        if len(seen) < self.min_samples:
            return None
        if seen[-1][0] - seen[0][0] < SPAN_FRACTION * self.window_s:
            return None
        values = [v for _t, v in seen]
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
