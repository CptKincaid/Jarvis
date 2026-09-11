"""The uncorroborated-occupied-run detector.

Two things have to be true at once and they pull against each other: a
radar latched on with the flat empty must be caught, and a man who sits at
his desk all afternoon must never be called a fault. The separator is not
duration -- it is whether anything else ever agreed.
"""
from __future__ import annotations

import json

import pytest

from jarvis import stuckroom


@pytest.fixture()
def board(tmp_path):
    events = []
    sr = stuckroom.StuckRooms(path=tmp_path / "roomruns.json",
                              publish=events.append)
    return sr, events


# ------------------------------------------------------------ catching
def test_a_room_occupied_for_hours_with_nothing_agreeing_is_a_fault(board):
    sr, events = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    assert sr.faulted() == frozenset()
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    assert sr.faulted() == frozenset({"office"})


def test_the_fault_names_the_room_on_the_board_and_says_it_once(board):
    sr, events = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 60.0)
    raised = [e for e in events if not e.cleared]
    assert len(raised) == 1
    ev = raised[0]
    assert ev.rule == stuckroom.RULE
    assert "OFFICE" in ev.token.upper()
    assert len(ev.token) <= 10
    assert "office" in ev.text.lower()


def test_the_fault_clears_the_instant_the_room_reads_empty(board):
    sr, events = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    sr.observe("office", False, at=t + stuckroom.FAULT_AFTER_S + 2.0)
    assert sr.faulted() == frozenset()
    assert [e for e in events if e.cleared], "no clear was published"


# ------------------------------------------------- NOT crying wolf
def test_a_genuine_day_indoors_is_never_a_fault(board):
    """His measured session: the office was the active room from 17:01:27
    to 18:43:19 -- 1 h 42 min -- while he sat and worked. His phone is
    polled every 60 s and answers, so the run is corroborated ~102 times.
    """
    sr, events = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    for i in range(1, 103):                      # 102 minutes, one poll a minute
        at = t + i * 60.0
        sr.corroborate("phone", at=at)           # the phone answered
        sr.observe("office", True, at=at)
        assert sr.faulted() == frozenset(), "cried wolf at %d min" % i
    assert not [e for e in events if not e.cleared]


def test_one_corroboration_resets_the_clock_rather_than_excusing_the_run(board):
    sr, events = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    sr.corroborate("mic", at=t + stuckroom.FAULT_AFTER_S - 60.0)
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    assert sr.faulted() == frozenset()           # the stamp moved the clock
    sr.observe("office", True, at=t + 2 * stuckroom.FAULT_AFTER_S + 1.0)
    assert sr.faulted() == frozenset({"office"})  # and then it ran out again


def test_any_of_the_three_legs_counts_as_corroboration(board):
    for source in ("phone", "camera", "mic"):
        sr, _ = board
        t = 1_000_000.0
        sr.observe("office", True, at=t)
        sr.corroborate(source, at=t + 10.0)
        sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S - 5.0)
        assert sr.faulted() == frozenset(), source


def test_a_room_that_flickers_off_starts_a_fresh_run(board):
    sr, _ = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    sr.observe("office", False, at=t + 100.0)
    sr.observe("office", True, at=t + 200.0)
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 150.0)
    assert sr.faulted() == frozenset()


def test_a_reading_with_no_opinion_neither_starts_nor_ends_a_run(board):
    sr, _ = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    sr.observe("office", None, at=t + 10.0)
    assert sr.faulted() == frozenset()
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    assert sr.faulted() == frozenset({"office"})


def test_rooms_are_tracked_apart(board):
    sr, _ = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    sr.observe("kitchen", False, at=t)
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    sr.observe("kitchen", False, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    assert sr.faulted() == frozenset({"office"})


# --------------------------------------------------------- persistence
def test_restarting_jarvis_does_not_launder_a_stuck_sensor(tmp_path):
    """The 20:22 restart reset the office run to zero 21 minutes before he
    walked in. The in-memory clock is why a 12 h threshold was never once
    reachable on this box."""
    path = tmp_path / "roomruns.json"
    t = 1_000_000.0
    first = stuckroom.StuckRooms(path=path, publish=lambda e: None)
    first.observe("office", True, at=t)
    first.observe("office", True, at=t + 600.0)

    second = stuckroom.StuckRooms(path=path, publish=lambda e: None)
    # 10 minutes into the run when the process died; the rest elapses after.
    second.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    assert second.faulted() == frozenset({"office"})


def test_the_corroboration_stamp_survives_a_restart_too(tmp_path):
    path = tmp_path / "roomruns.json"
    t = 1_000_000.0
    first = stuckroom.StuckRooms(path=path, publish=lambda e: None)
    first.observe("office", True, at=t)
    first.corroborate("phone", at=t + 600.0)

    second = stuckroom.StuckRooms(path=path, publish=lambda e: None)
    second.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    assert second.faulted() == frozenset(), "the stamp was lost across restart"


def test_the_state_file_is_wall_clock_not_monotonic(tmp_path):
    """roomfabric ticks on time.monotonic, which restarts at zero with the
    process. Persisting that would be persisting nothing."""
    path = tmp_path / "roomruns.json"
    sr = stuckroom.StuckRooms(path=path, publish=lambda e: None)
    sr.observe("office", True, at=1_700_000_000.0)
    data = json.loads(path.read_text())
    started = data["runs"]["office"]["started"]
    assert started > 1_000_000_000.0


def test_an_unreadable_state_file_costs_the_history_and_nothing_else(tmp_path):
    path = tmp_path / "roomruns.json"
    path.write_text("{ not json")
    sr = stuckroom.StuckRooms(path=path, publish=lambda e: None)
    sr.observe("office", True, at=1_000_000.0)
    assert sr.faulted() == frozenset()


def test_an_unwritable_state_path_never_raises(tmp_path):
    sr = stuckroom.StuckRooms(path=tmp_path / "nope" / "deep" / "x.json",
                              publish=lambda e: None)
    sr.observe("office", True, at=1_000_000.0)          # must not raise


def test_a_publish_that_raises_does_not_break_the_detector(tmp_path):
    def boom(_ev):
        raise RuntimeError("bus down")
    sr = stuckroom.StuckRooms(path=tmp_path / "s.json", publish=boom)
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    assert sr.faulted() == frozenset({"office"})


# --------------------------------------------- what it may NEVER do
def test_the_detector_has_no_way_to_say_he_is_out(board):
    sr, events = board
    t = 1_000_000.0
    sr.observe("office", True, at=t)
    sr.observe("office", True, at=t + stuckroom.FAULT_AFTER_S + 1.0)
    for ev in events:
        assert not hasattr(ev, "home")
    assert not hasattr(sr, "presence")
    assert not hasattr(sr, "away")


def test_the_detector_exposes_no_way_to_greet(board):
    sr, _ = board
    for name in ("greet", "arrival", "speak", "say"):
        assert not hasattr(sr, name)


def test_the_threshold_is_labelled_as_a_starting_point_not_a_measurement():
    assert "GUESS" in stuckroom.FAULT_AFTER_S_PROVENANCE.upper()


def test_the_duration_only_fallback_clears_his_measured_session():
    """1 h 42 min measured; the fallback must not fire on it, with margin."""
    assert stuckroom.DEFAULT_STUCK_AFTER_H * 3600.0 > 2.0 * (102 * 60.0)


def test_status_is_numbers_only(board):
    """UPDATED 2026-09-06, not relaxed: a room whose FIRST post-boot reading
    is True has a latch of unknown age, and ``run_s`` says None rather than
    stamping "now" and calling that a beginning. An ordinary rising edge
    still measures exactly as it did."""
    sr, _ = board
    sr.observe("office", True, at=1_000_000.0)
    st = sr.status(now=1_000_100.0)
    assert st["office"]["run_s"] is None
    assert st["office"]["pre_existing"] is True
    assert st["office"]["corroborated_s_ago"] is None
    assert st["office"]["faulted"] is False


def test_status_measures_an_ordinary_run_exactly_as_it_always_did(board):
    sr, _ = board
    sr.observe("kitchen", False, at=999_990.0)
    sr.observe("kitchen", True, at=1_000_000.0)
    st = sr.status(now=1_000_100.0)
    assert st["kitchen"]["run_s"] == pytest.approx(100.0)
    assert st["kitchen"]["pre_existing"] is False
    assert st["kitchen"]["corroborated_s_ago"] is None
    assert st["kitchen"]["faulted"] is False
