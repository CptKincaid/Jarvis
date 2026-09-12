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
    """Every verdict a full window would have produced over the trace --
    only once the window could judge (enough readings, spanning it)."""
    win = R.StillWindow(**kw)
    out = []
    for at, cm in rows:
        win.add(cm, at=at)
        if win.spread() is not None:
            out.append(win.verdict())
    return out


# ---------------------------------------------------------- arithmetic
def _fill(win, cm_of, n=90, step=2.0):
    """n readings at the fabric's cadence -- a full window at 2 s."""
    for i in range(n):
        win.add(cm_of(i), at=i * step)
    return win


def test_a_pinned_reading_is_a_fixture():
    win = _fill(R.StillWindow(), lambda i: 310 + (i % 3))     # 2 cm of jitter
    assert win.verdict() == R.FIXTURE


def test_a_wandering_reading_is_a_person():
    win = _fill(R.StillWindow(), lambda i: 300 + (i % 2) * 34)  # 34 cm, his measured
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
    win = _fill(R.StillWindow(), lambda i: 310)
    for i in range(90, 100):
        win.add(None, at=i * 2.0)
        win.add("", at=i * 2.0)
        win.add(0, at=i * 2.0)
    assert win.verdict() == R.FIXTURE


def test_the_window_forgets_what_is_older_than_the_window():
    win = R.StillWindow(window_s=60.0, min_samples=20)
    for i in range(40):
        win.add(200, at=float(i))                    # long gone
    for i in range(100, 160):
        win.add(310 + (i % 3), at=float(i))
    assert win.verdict() == R.FIXTURE, "a stale reading widened the spread"


def test_clearing_forgets_the_old_run():
    win = _fill(R.StillWindow(), lambda i: 310)
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


# ------------------------------------------ the attack round, 2026-09-12
def test_the_verdict_needs_the_whole_window_not_just_fifty_readings():
    """Fifty readings in 12.5 s is not three minutes of evidence: the
    calibration was a 180 s WINDOW, and MIN_SAMPLES only guards density.
    Judged on the span, a 0.25 s poll cannot judge early and his own
    at-desk recording has no run-start offset that reads him a fixture."""
    fast = R.StillWindow()
    for i in range(60):
        fast.add(310 + (i % 3), at=i * 0.25)          # 15 s of readings
    assert fast.verdict() == R.UNKNOWN
    full = R.StillWindow()
    for i in range(90):
        full.add(310 + (i % 3), at=i * 2.0)           # 178 s of readings
    assert full.verdict() == R.FIXTURE


def test_expire_lets_a_verdict_decay_when_readings_stop():
    """A dead distance entity must not freeze a FIXTURE for ever: the
    window is pruned by the clock, not only by the next reading, so an
    unreadable distance returns the sensor's word inside one window."""
    win = R.StillWindow()
    for i in range(90):
        win.add(310, at=i * 2.0)
    assert win.verdict() == R.FIXTURE
    win.expire(at=180.0 + 400.0)
    assert win.samples == 0
    assert win.verdict() == R.UNKNOWN


def test_a_bool_nan_or_inf_reading_is_dropped():
    win = R.StillWindow()
    for i in range(90):
        win.add(310, at=i * 2.0)
    win.add(True, at=181.0)
    win.add(float("nan"), at=182.0)
    win.add(float("inf"), at=183.0)
    assert win.samples == 90
    assert win.verdict() == R.FIXTURE


@pytest.mark.skipif(not DESK_CSV.exists(), reason="his trace is not here")
def test_NO_run_start_in_his_desk_recording_reads_him_as_a_fixture():
    """The first verdict of a run, from every possible start offset. Under
    a 50-reading rule 4 of 465 offsets called him furniture (measured
    2026-09-12); under the span rule none may."""
    rows = _still(DESK_CSV)
    bad = 0
    for start in range(len(rows)):
        win = R.StillWindow()
        for at, cm in rows[start:]:
            win.add(cm, at=at)
            v = win.verdict()
            if v != R.UNKNOWN:
                bad += v == R.FIXTURE
                break
    assert bad == 0
