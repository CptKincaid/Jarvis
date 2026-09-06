"""The voter wired into the sentinel, and the 2026-09-05 chain replayed.

The point of this file is the last test: with the office radar latched and
his phone out of the flat, the sentinel must reach a strict "away" so that
``door_arrival`` can fire when the kitchen lights up. Tonight it did not,
and he got silence.
"""
from __future__ import annotations

import pytest

from jarvis import presence, presencevote as pv, roomfabric


class FakeSensor:
    configured = True
    blocked = ""
    paused = False

    def __init__(self, value=None):
        self.value = value

    def read(self):
        return self.value

    def status(self):
        return {}


def build_fabric(office=None, kitchen=None, now=None):
    clock = {"t": 1000.0}
    fab = roomfabric.from_readers(
        [("office", FakeSensor(office)), ("kitchen", FakeSensor(kitchen))],
        now=(now or (lambda: clock["t"])), enter_hold_s=0.0, poll_s=2.0)
    fab._clock = clock
    return fab


@pytest.fixture()
def stuck(tmp_path):
    from jarvis import stuckroom
    return stuckroom.StuckRooms(path=tmp_path / "runs.json",
                                publish=lambda e: None)


@pytest.fixture()
def clocked(tmp_path):
    """A detector on a clock the test drives, so the stamps the test writes
    and the stamps ThreeLegProbe writes come from the same timeline."""
    from jarvis import stuckroom
    now = {"t": 1_000_000.0}
    sr = stuckroom.StuckRooms(path=tmp_path / "runs.json",
                              publish=lambda e: None, now=lambda: now["t"])
    return sr, now


# ------------------------------------------------------- the leg shapes
def test_the_probe_wears_the_sentinels_shape(stuck):
    fab = build_fabric(office=False, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True)
    assert leg("1.2.3.4", "") is True
    assert callable(leg)


def test_home_and_bed_both_read_as_present_to_the_sentinel(stuck):
    fab = build_fabric(office=False, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True)
    assert leg("1.2.3.4", "") is True
    assert leg.verdict.state in (pv.HOME, pv.BED)


def test_an_unknown_verdict_is_none_so_the_sentinel_holds(stuck):
    """Cells 9/17/18/26/27. The sentinel's existing None path holds the
    state and does not count it towards the away grace."""
    fab = build_fabric(office=None, kitchen=None)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: None)
    assert leg("1.2.3.4", "") is None
    assert leg.verdict.state == pv.UNKNOWN


def test_the_sentinel_keeps_the_grace_and_the_voter_does_not_double_it(stuck):
    """The voter's phone leg runs with a zero grace ON PURPOSE: the
    sentinel's away_after_min is the one and only hysteresis, and two
    graces stacked would make "away" take 24 minutes."""
    fab = build_fabric(office=False, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False)
    assert leg("1.2.3.4", "") is False
    assert leg.grace_s == 0.0


def test_a_camera_that_is_not_wired_is_blind_and_never_votes_no(stuck):
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True)
    leg("1.2.3.4", "")
    assert leg.legs["camera"] == pv.CAM_BLIND


def test_an_eye_that_raises_is_blind_rather_than_fatal(stuck):
    def boom():
        raise RuntimeError("no feed")
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck, eye=boom,
                                 phone=lambda ip, mac: True)
    assert leg("1.2.3.4", "") is True
    assert leg.legs["camera"] == pv.CAM_BLIND


def test_the_eye_is_asked_for_a_name_and_a_count_and_never_a_frame(stuck):
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    seen = []

    def eye():
        seen.append(1)
        return ("hunter", 1, True)
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck, eye=eye,
                                 phone=lambda ip, mac: False)
    assert leg("1.2.3.4", "") is True          # P1: a named him wins
    assert leg.legs["camera"] == pv.CAM_SAW
    assert seen, "the eye was never asked"


def test_every_verdict_carries_a_reason(stuck):
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True)
    leg("1.2.3.4", "")
    assert leg.verdict.reason.strip()
    assert leg.verdict.cell > 0


def test_the_reason_reaches_the_log_at_info(stuck, caplog):
    import logging
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True)
    with caplog.at_level(logging.INFO, logger="presence"):
        leg("1.2.3.4", "")
    assert any("cell" in r.getMessage() for r in caplog.records)


def test_the_reason_is_logged_once_per_change_not_once_per_poll(stuck, caplog):
    import logging
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True)
    with caplog.at_level(logging.INFO, logger="presence"):
        for _ in range(20):
            leg("1.2.3.4", "")
    mine = [r for r in caplog.records if r.name.endswith("presence")]
    assert len(mine) == 1


# ------------------------------------------------------ the room hint
def test_the_hint_comes_from_last_true_and_not_from_the_active_room():
    """where() drops the active room after 90 s; every bedroom trip that
    matters is longer than that, so the hint has to be last_true."""
    clock = {"t": 1000.0}
    fab = roomfabric.from_readers(
        [("office", FakeSensor(True)), ("kitchen", FakeSensor(False))],
        now=lambda: clock["t"], enter_hold_s=0.0, poll_s=2.0)
    fab.tick()
    fab.rooms[0].sensor.value = False
    clock["t"] += 600.0                       # ten minutes later
    fab.tick()
    assert fab.where().room == ""             # the active room is long gone
    name, age = fab.last_seen_room()
    assert name == "office"
    assert age == pytest.approx(600.0, abs=5.0)


def test_no_room_has_ever_seen_anybody_is_an_empty_hint():
    fab = build_fabric(office=False, kitchen=False)
    fab.tick()
    assert fab.last_seen_room() == ("", None)


def test_the_office_last_and_a_clear_house_says_he_never_left_the_office(stuck):
    clock = {"t": 1000.0}
    fab = roomfabric.from_readers(
        [("office", FakeSensor(True)), ("kitchen", FakeSensor(False))],
        now=lambda: clock["t"], enter_hold_s=0.0, poll_s=2.0)
    fab.tick()
    fab.rooms[0].sensor.value = False
    clock["t"] += 300.0
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True)
    assert leg("1.2.3.4", "") is True
    assert leg.verdict.state == pv.HOME
    assert "office" in leg.verdict.reason


# ------------------------------------------- a faulted room is dropped
def test_a_faulted_room_stops_pinning_the_house_occupied(clocked):
    stuck, now = clocked
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False)
    # Corroborated by nothing, and the run runs out.
    stuck.observe("office", True)
    now["t"] += 3000.0
    stuck.observe("office", True)
    assert stuck.faulted() == frozenset({"office"})
    assert leg("1.2.3.4", "") is False
    assert leg.legs["rooms"] == pv.ROOMS_CLEAR


def test_the_fabrics_own_stuck_flag_is_honoured_too(stuck):
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    fab.rooms[0].stuck = True
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False)
    leg("1.2.3.4", "")
    assert leg.legs["rooms"] == pv.ROOMS_CLEAR


# ================================================================
# THE 2026-09-05 CHAIN, REPLAYED
# ================================================================
def test_tonight_the_latched_office_no_longer_costs_him_the_greeting(clocked):
    """20:43:13. The office has read occupied since 19:07:11 with the flat
    empty; the kitchen is correctly clear; his phone is out with him; the
    camera has been stopped since 17:04:25 so it cannot look.

    Before: HouseView called the house occupied because ANY room was, the
    phone was never even asked ("if seen: return True"), the sentinel
    never said away, and door_arrival needs away and only away.

    After: nothing has corroborated the office run, so the room is dropped,
    the house falls through to the phone, and the phone says no.
    """
    stuck, now = clocked
    stuck.observe("office", True)                            # 19:07:11
    now["t"] += 96 * 60.0                                    # 20:43:13
    stuck.observe("office", True)
    assert stuck.faulted() == frozenset({"office"})

    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False)
    assert leg("192.168.50.34", "") is False
    assert leg.verdict.strictly_away is True

    from jarvis import arrival
    watch = arrival.DoorWatch()
    assert watch.observe(room="kitchen", away=True, state="away") is True


def test_the_same_cell_with_him_at_the_desk_does_not_call_him_out(clocked):
    """The mirror danger: phone napping, camera off, he is sitting still.
    Identical three legs; only the run's history separates them."""
    stuck, now = clocked
    stuck.observe("office", True)
    for _ in range(96):                         # his phone answers each minute
        now["t"] += 60.0
        stuck.corroborate("phone")
        stuck.observe("office", True)
    assert stuck.faulted() == frozenset()

    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: False)
    assert leg("192.168.50.34", "") is True
    assert leg.verdict.cell == 6
    assert leg.verdict.strictly_away is False


def test_a_phone_answering_corroborates_the_run_by_itself(clocked):
    stuck, _now = clocked
    fab = build_fabric(office=True, kitchen=False)
    fab.tick()
    leg = presence.ThreeLegProbe(fabric=fab, stuck=stuck,
                                 phone=lambda ip, mac: True)
    leg("1.2.3.4", "")
    assert stuck.status()["office"]["corroborated_s_ago"] is not None
