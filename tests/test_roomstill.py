"""A radar's occupancy is a body or a fixture, told apart by DISTANCE.

His office LD2410 reports OCCUPIED for ever: 527 of 527 samples with the
flat empty, and the same reading with him home in another room. Its bit
carries no information about him. Its still DISTANCE does.

Every threshold in jarvis/roomstill.py comes from two fifteen-minute
recordings he took on 2026-09-11 through scripts/room_trace.py, and the
tests below replay those recordings rather than inventing numbers. If the
CSVs are not on the box the replay tests skip; the arithmetic tests do
not, so the module is never untested.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from jarvis import roomstill as R

EMPTY_CSV = Path.home() / "room-trace-EMPTY-FLAT.csv"
DESK_CSV = Path.home() / "room-trace-AT-DESK.csv"


def _still(path):
    with open(path) as fh:
        return [(float(r["t"]), float(r["office_still_cm"]))
                for r in csv.DictReader(fh)
                if r.get("office_still_cm") not in ("", "None", None)]


def _verdicts(rows, **kw):
    """Every verdict a full window would have produced over the trace."""
    win = R.StillWindow(**kw)
    out = []
    for at, cm in rows:
        win.add(cm, at=at)
        if win.samples >= win.min_samples:
            out.append(win.verdict())
    return out


# ---------------------------------------------------------- arithmetic
def test_a_pinned_reading_is_a_fixture():
    win = R.StillWindow()
    for i in range(60):
        win.add(310 + (i % 3), at=float(i))          # 2 cm of jitter
    assert win.verdict() == R.FIXTURE


def test_a_wandering_reading_is_a_person():
    win = R.StillWindow()
    for i in range(60):
        win.add(300 + (i % 2) * 34, at=float(i))     # 34 cm, his measured
    assert win.verdict() == R.PERSON


def test_too_little_evidence_is_UNKNOWN_never_fixture():
    """The failure this module must never commit is a false empty room."""
    win = R.StillWindow()
    for i in range(R.MIN_SAMPLES - 1):
        win.add(310, at=float(i))
    assert win.verdict() == R.UNKNOWN


def test_a_missing_reading_is_dropped_not_stored_as_zero():
    """A None kept as 0.0 would make a 310 cm spread out of nothing and
    read a locked sensor as a person -- the bug pointing backwards."""
    win = R.StillWindow()
    for i in range(60):
        win.add(310, at=float(i))
    for i in range(60, 70):
        win.add(None, at=float(i))
        win.add("", at=float(i))
        win.add(0, at=float(i))
    assert win.verdict() == R.FIXTURE


def test_the_window_forgets_what_is_older_than_the_window():
    win = R.StillWindow(window_s=60.0, min_samples=20)
    for i in range(40):
        win.add(200, at=float(i))                    # long gone
    for i in range(100, 140):
        win.add(310 + (i % 3), at=float(i))
    assert win.verdict() == R.FIXTURE, "a stale reading widened the spread"


def test_clearing_forgets_the_old_run():
    win = R.StillWindow()
    for i in range(60):
        win.add(310, at=float(i))
    win.clear()
    assert win.verdict() == R.UNKNOWN


# ------------------------------------------------------- the asymmetry
def test_only_a_FIXTURE_may_take_an_occupancy_away():
    assert R.occupancy_is_real(R.FIXTURE, True) is False
    assert R.occupancy_is_real(R.PERSON, True) is True
    assert R.occupancy_is_real(R.UNKNOWN, True) is True


def test_nothing_here_can_MANUFACTURE_an_occupancy():
    for v in (R.PERSON, R.FIXTURE, R.UNKNOWN):
        assert R.occupancy_is_real(v, False) is False
        assert R.occupancy_is_real(v, None) is None


# ------------------------------------------- replayed from his own box
@pytest.mark.skipif(not EMPTY_CSV.exists(), reason="his trace is not here")
def test_HIS_EMPTY_FLAT_reads_as_a_fixture_throughout():
    v = _verdicts(_still(EMPTY_CSV))
    assert v, "no full windows in the trace"
    assert all(x == R.FIXTURE for x in v), \
        f"{sum(1 for x in v if x != R.FIXTURE)}/{len(v)} windows were not fixture"


@pytest.mark.skipif(not DESK_CSV.exists(), reason="his trace is not here")
def test_HIM_AT_HIS_DESK_reads_as_a_person_throughout():
    """The one that matters. A single FIXTURE window here is Jarvis
    deciding he is out while he is sitting at his desk."""
    v = _verdicts(_still(DESK_CSV))
    assert v, "no full windows in the trace"
    assert all(x == R.PERSON for x in v), \
        f"{sum(1 for x in v if x != R.PERSON)}/{len(v)} windows called him furniture"


@pytest.mark.skipif(not (EMPTY_CSV.exists() and DESK_CSV.exists()),
                    reason="his traces are not here")
def test_the_threshold_sits_in_a_gap_with_NO_overlap():
    """Empty max 7 cm, seated min 24 cm, across 923 measured windows. The
    threshold is the middle of that gap, and this fails the day a change
    narrows it -- which is the moment to re-measure, not to nudge it."""
    def spreads(path):
        win, out = R.StillWindow(), []
        for at, cm in _still(path):
            win.add(cm, at=at)
            s = win.spread()
            if s is not None:
                out.append(s)
        return out
    empty, desk = spreads(EMPTY_CSV), spreads(DESK_CSV)
    assert max(empty) < R.SPREAD_CM < min(desk), \
        f"empty max {max(empty)}, threshold {R.SPREAD_CM}, seated min {min(desk)}"
