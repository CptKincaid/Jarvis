"""The three-leg voter, cell by cell, and the departure sequence.

The parametrised table below IS his truth table. It is written out in full
rather than derived, because a derived table would agree with the code by
construction and prove nothing: the point of the test is that a human can
read the 27 rows against the design and see the same answers.
"""
from __future__ import annotations

import pytest

from jarvis import presencevote as pv

Y, N, U = pv.PHONE_YES, pv.PHONE_NO, pv.PHONE_UNKNOWN
S, L, X = pv.CAM_SAW, pv.CAM_LOOKED, pv.CAM_BLIND
ON, CLR, UNR = pv.ROOMS_ON, pv.ROOMS_CLEAR, pv.ROOMS_UNREACHABLE
HOME, AWAY, BED, UNK = pv.HOME, pv.AWAY, pv.BED, pv.UNKNOWN

# cell, rooms, phone, camera, expected state
TABLE = [
    # --- sensor ON -----------------------------------------------------
    (1, ON, Y, S, HOME), (2, ON, Y, L, HOME), (3, ON, Y, X, HOME),
    (4, ON, N, S, HOME), (5, ON, N, L, AWAY), (6, ON, N, X, HOME),
    (7, ON, U, S, HOME), (8, ON, U, L, HOME), (9, ON, U, X, UNK),
    # --- sensor ALL CLEAR ----------------------------------------------
    (10, CLR, Y, S, HOME), (11, CLR, Y, L, BED), (12, CLR, Y, X, BED),
    (13, CLR, N, S, HOME), (14, CLR, N, L, AWAY), (15, CLR, N, X, AWAY),
    (16, CLR, U, S, HOME), (17, CLR, U, L, UNK), (18, CLR, U, X, UNK),
    # --- sensor UNREACHABLE --------------------------------------------
    (19, UNR, Y, S, HOME), (20, UNR, Y, L, HOME), (21, UNR, Y, X, HOME),
    (22, UNR, N, S, HOME), (23, UNR, N, L, AWAY), (24, UNR, N, X, AWAY),
    (25, UNR, U, S, HOME), (26, UNR, U, L, UNK), (27, UNR, U, X, UNK),
]


@pytest.mark.parametrize("cell,rooms,phone,camera,want", TABLE,
                         ids=[str(r[0]) for r in TABLE])
def test_every_cell_of_his_truth_table(cell, rooms, phone, camera, want):
    # Cell 6 is the one cell whose answer depends on history; the default
    # (a corroborated run) is the "he is at his desk" reading. Cells 11/12
    # default to the bare rule 2 with no room hint.
    v = pv.decide(phone=phone, camera=camera, rooms=rooms)
    assert v.state == want, "cell %d: %s" % (cell, v.reason)
    assert v.cell == cell
    assert v.reason.strip(), "cell %d produced no reason" % cell


def test_the_table_covers_all_twenty_seven_cells_exactly_once():
    assert len(TABLE) == 27
    assert len({r[0] for r in TABLE}) == 27
    assert len({(r[1], r[2], r[3]) for r in TABLE}) == 27


# ---------------------------------------------------------------- P1/P2
def test_a_camera_that_names_him_wins_outright_in_every_sensor_state():
    """P1. Six cells, and none of them may answer anything but HOME."""
    for rooms in (ON, CLR, UNR):
        for phone in (N, U):
            v = pv.decide(phone=phone, camera=S, rooms=rooms)
            assert v.state == HOME, v.reason


def test_sensor_on_never_carries_a_verdict_by_itself():
    """P2, and it is tonight. ON with both honest legs negative is AWAY."""
    assert pv.decide(phone=N, camera=L, rooms=ON).state == AWAY


# ------------------------------------------------------------- cell 5/6
def test_rule_one_is_his_and_it_fires_on_a_camera_that_actually_looked():
    v = pv.decide(phone=N, camera=L, rooms=ON)
    assert v.state == AWAY and v.cell == 5
    assert "looked" in v.reason


def test_a_camera_that_could_not_look_is_never_read_as_not_seeing_him():
    """The whole reason tonight is not covered by rule 1.

    A false away is worse than a false home: it fires a greeting at a man
    already sitting down. So X must not reach cell 5.
    """
    v = pv.decide(phone=N, camera=X, rooms=ON)
    assert v.cell == 6 and v.state != AWAY


def test_cell_six_splits_on_whether_anything_agreed_with_the_run_RECENTLY():
    """Not "ever". One phone hit an hour ago is not agreement now -- see
    tests/test_presence_recency.py for the two failures that came of it."""
    seated = pv.decide(phone=N, camera=X, rooms=ON, agreed_s_ago=60.0,
                       mic=pv.MIC_SILENT)
    latched = pv.decide(phone=N, camera=X, rooms=ON, agreed_s_ago=3600.0,
                        mic=pv.MIC_SILENT)
    assert seated.state == HOME and seated.cell == 6
    assert latched.state == AWAY and latched.cell == 6
    assert "agreed" in latched.reason


def test_a_stale_run_is_the_only_thing_that_can_turn_cell_six_away():
    """The history clock may not leak into any other cell's answer."""
    for cell, rooms, phone, camera, want in TABLE:
        if cell == 6:
            continue
        assert pv.decide(phone=phone, camera=camera, rooms=rooms,
                         agreed_s_ago=3600.0,
                         mic=pv.MIC_SILENT).state == want, cell


# --------------------------------------------------------- the bedroom
def test_rule_two_says_bedroom_when_the_kitchen_saw_him_last():
    v = pv.decide(phone=Y, camera=X, rooms=CLR,
                  last_room="kitchen", last_room_age_s=120.0)
    assert v.state == BED
    assert "kitchen" in v.reason


def test_the_office_seeing_him_last_means_he_never_left_the_office():
    """The corridor argument: the office is only reachable through the
    kitchen, so a clear house with the OFFICE last is a dropped still
    body, not a bedroom trip. Opposite answer to his bare rule 2, and it
    must not be BED."""
    v = pv.decide(phone=Y, camera=X, rooms=CLR,
                  last_room="office", last_room_age_s=120.0)
    assert v.state == HOME
    assert "office" in v.reason


def test_a_stale_room_hint_falls_back_to_his_bare_rule_two():
    v = pv.decide(phone=Y, camera=X, rooms=CLR,
                  last_room="office", last_room_age_s=pv.HINT_MAX_AGE_S + 1.0)
    assert v.state == BED


def test_the_room_hint_is_slugged_the_way_the_fabric_slugs_it():
    v = pv.decide(phone=Y, camera=X, rooms=CLR,
                  last_room="  Office!  ", last_room_age_s=10.0)
    assert v.state == HOME


def test_bed_is_a_home_sub_state_and_never_a_step_towards_away():
    v = pv.decide(phone=Y, camera=L, rooms=CLR,
                  last_room="kitchen", last_room_age_s=60.0)
    assert v.state == BED
    assert v.home is True
    assert v.strictly_away is False
    assert v.greetable is False


# ------------------------------------------------------------- holding
def test_only_a_room_still_claiming_occupancy_holds_the_last_verdict():
    """Cell 9 holds; the other unknowns have nothing to hold on to."""
    assert pv.decide(phone=U, camera=X, rooms=ON).hold is True
    for rooms in (CLR, UNR):
        for camera in (L, X):
            assert pv.decide(phone=U, camera=camera, rooms=rooms).hold is False


def test_no_unknown_cell_is_ever_reported_as_away():
    for cell, rooms, phone, camera, want in TABLE:
        v = pv.decide(phone=phone, camera=camera, rooms=rooms)
        if v.state == UNK:
            assert v.strictly_away is False
            assert v.home is True, "unknown must not mute him (cell %d)" % cell


def test_a_bad_leg_value_is_unknown_rather_than_an_exception():
    v = pv.decide(phone="banana", camera=None, rooms="")
    assert v.state == UNK
    assert v.strictly_away is False


# --------------------------------------------------- the leg builders
def test_the_phone_leg_only_says_no_once_the_grace_has_run_out():
    assert pv.phone_leg(answer=True, unseen_s=0.0, grace_s=720.0) == Y
    assert pv.phone_leg(answer=False, unseen_s=60.0, grace_s=720.0) == U
    assert pv.phone_leg(answer=False, unseen_s=800.0, grace_s=720.0) == N
    assert pv.phone_leg(answer=None, unseen_s=9999.0, grace_s=720.0) == U


def test_a_napping_radio_inside_the_grace_is_unknown_not_absent():
    """away_after_min exists because a dropped phone is not a departure."""
    assert pv.phone_leg(answer=False, unseen_s=1.0, grace_s=720.0) != N


def test_the_camera_leg_tells_could_not_look_from_looked_and_saw_nobody():
    assert pv.camera_leg(identity="hunter", faces=1, live=True) == S
    assert pv.camera_leg(identity="", faces=0, live=True) == L
    assert pv.camera_leg(identity="", faces=0, live=False) == X
    assert pv.camera_leg(identity="", faces=None, live=True) == X


def test_a_camera_that_is_not_wired_is_blind_and_never_looked():
    """services.camera_feed is None on his box today: "" must be X."""
    assert pv.camera_leg(identity="", faces=None, live=False) == X


def test_the_rooms_leg_is_clear_only_when_every_room_answered():
    assert pv.rooms_leg({"office": True, "kitchen": False}) == ON
    assert pv.rooms_leg({"office": False, "kitchen": False}) == CLR
    assert pv.rooms_leg({"office": False, "kitchen": None}) == UNR
    assert pv.rooms_leg({}) == UNR


def test_a_faulted_room_is_dropped_rather_than_voting_no():
    """The fault removes a vote; it does not cast one. Office latched and
    kitchen genuinely clear must read CLEAR, not ON."""
    assert pv.rooms_leg({"office": True, "kitchen": False},
                        faulted=("office",)) == CLR
    assert pv.rooms_leg({"office": True}, faulted=("office",)) == UNR


# ------------------------------------------------- the departure order
def test_his_departure_sequence_needs_the_three_in_order():
    seq = pv.DepartureSequence()
    assert seq.room(room="office", at=0.0) is False
    assert seq.room(room="kitchen", at=12.0) is False
    assert seq.phone_gone(at=200.0) is True
    assert seq.stage == pv.SEQ_LEFT


def test_the_kitchen_alone_is_not_a_departure():
    seq = pv.DepartureSequence()
    seq.room(room="kitchen", at=0.0)
    assert seq.phone_gone(at=100.0) is False


def test_the_wrong_order_is_not_a_departure():
    """kitchen then office is him coming back to his desk."""
    seq = pv.DepartureSequence()
    seq.room(room="kitchen", at=0.0)
    seq.room(room="office", at=10.0)
    assert seq.phone_gone(at=100.0) is False


def test_a_slow_walk_between_the_two_rooms_breaks_the_sequence():
    seq = pv.DepartureSequence()
    seq.room(room="office", at=0.0)
    seq.room(room="kitchen", at=pv.SEQ_STEP_S + 1.0)
    assert seq.phone_gone(at=pv.SEQ_STEP_S + 30.0) is False


def test_a_coffee_run_does_not_become_a_departure_an_hour_later():
    seq = pv.DepartureSequence()
    seq.room(room="office", at=0.0)
    seq.room(room="kitchen", at=10.0)
    assert seq.phone_gone(at=10.0 + pv.SEQ_WINDOW_S + 1.0) is False


def test_going_back_to_the_office_disarms_the_sequence():
    seq = pv.DepartureSequence()
    seq.room(room="office", at=0.0)
    seq.room(room="kitchen", at=10.0)
    seq.room(room="office", at=40.0)
    assert seq.phone_gone(at=100.0) is False


def test_the_departure_sequence_is_silent():
    """arrival.py is explicit: a valediction to an empty room is a
    notification pretending to be a presence."""
    seq = pv.DepartureSequence()
    seq.room(room="office", at=0.0)
    seq.room(room="kitchen", at=5.0)
    seq.phone_gone(at=100.0)
    assert seq.speaks is False
    assert pv.departure_note(seq) .startswith("departure:")


def test_the_sequence_re_arms_after_he_comes_home():
    seq = pv.DepartureSequence()
    seq.room(room="office", at=0.0)
    seq.room(room="kitchen", at=5.0)
    assert seq.phone_gone(at=100.0) is True
    seq.home(at=200.0)
    assert seq.stage == pv.SEQ_IDLE
    seq.room(room="office", at=300.0)
    seq.room(room="kitchen", at=310.0)
    assert seq.phone_gone(at=400.0) is True


def test_the_window_numbers_are_stated_and_defensible():
    """The step window must clear a measured walk with real margin and the
    outer window must clear the 12-minute away grace."""
    assert pv.SEQ_STEP_S >= 60.0
    assert pv.SEQ_WINDOW_S > 12 * 60.0


# ============================================================
# THE SENSOR-ONLY BOX: P2 has nothing to outvote it with
# ============================================================
def test_a_box_with_no_phone_leg_at_all_still_gets_a_verdict():
    """P2 ("sensor ON never carries a verdict alone") is a rule about
    OUTVOTING, and on a box with no phone configured there is nothing to
    outvote with. Holding "unknown" for ever there would mean presence
    simply never works -- so the rooms answer, exactly as RoomOrPhone
    always made them answer on a sensor-only install.

    This is NOT his box: presence.phone_ip is set on the Spark, so the
    27-cell table above is what runs for him.
    """
    assert pv.decide_rooms_only(rooms=ON, camera=X).state == HOME
    assert pv.decide_rooms_only(rooms=CLR, camera=X).state == AWAY
    assert pv.decide_rooms_only(rooms=UNR, camera=X).state == UNK


def test_the_sensor_only_verdict_still_says_why():
    v = pv.decide_rooms_only(rooms=ON, camera=X)
    assert "phone" in v.reason and v.reason.strip()


def test_a_camera_that_names_him_still_wins_on_a_sensor_only_box():
    assert pv.decide_rooms_only(rooms=CLR, camera=S).state == HOME


def test_a_sensor_only_box_never_derives_away_from_an_unreadable_room():
    assert pv.decide_rooms_only(rooms=UNR, camera=X).strictly_away is False
    assert pv.decide_rooms_only(rooms=UNR, camera=L).strictly_away is False
