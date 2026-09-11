"""THE WALK-OUT THAT NEVER REGISTERED, and the instrument that would have
answered it in the log instead of costing an afternoon.

He left for a massage at about 10:51 on 2026-09-06. There were 45 kitchen
lines before the restart and NONE after it until at least 12:31, so
``DepartureSequence`` never left SEQ_DESK and "away" arrived only at
11:05:19 off the 15-minute corroboration window -- about fourteen minutes
late.

MEASURED, twice and two ways: the kitchen LD2410's absence delay is 10 s
(the redacted sensor profile's ``timeout_s``, and a read-only GET on the
device's own "Absence delay" number). So the bit stays ON for ~10 s after
the last sign of life and a one-second walk-through should show True across
several consecutive polls. The enter hold therefore cannot be what lost the
walk-out UNLESS the bit never rose or the polls never landed -- and today
those two look IDENTICAL in the log. That is what this instrument fixes.

BE HONEST ABOUT QUANTISATION. At ``poll_s`` 2.0 a single-poll run measures
0.0 s, not 0.8 s. The line may not invent sub-poll precision.
"""
from __future__ import annotations

import logging


from jarvis import presencevote as pv
from jarvis import roomfabric as rf
from jarvis.events import RoomChanged


class Reader:
    def __init__(self, values):
        self.values = list(values)
        self.configured = True

    def read(self):
        return self.values.pop(0) if self.values else None


def build(office, kitchen, poll_s=2.0, hold=2.0, publish=None, now=None):
    clock = {"t": 1000.0}
    fab = rf.from_readers([("office", Reader(office)),
                           ("kitchen", Reader(kitchen))],
                          poll_s=poll_s, enter_hold_s=hold, publish=publish,
                          now=(now or (lambda: clock["t"])))
    return fab, clock


def pump(fab, clock, ticks):
    for _ in range(ticks):
        fab.tick()
        clock["t"] += fab.poll_s


# =====================================================================
# 15. THE GLIMPSE LINE EXISTS AND IS HONEST
# =====================================================================
def test_a_one_poll_kitchen_run_is_logged_as_a_glimpse(caplog):
    caplog.set_level(logging.INFO, logger="roomfabric")
    events = []
    fab, clock = build(office=[False] * 4, kitchen=[False, True, False, False],
                       publish=events.append)
    pump(fab, clock, 4)
    lines = [r.getMessage() for r in caplog.records if "glimpse" in r.getMessage()]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "kitchen" in line
    assert "1 poll" in line
    assert "0.0 s" in line, "one poll measures ZERO seconds, not 0.8"
    assert "0.8" not in line, "no sub-poll precision may be invented"
    assert "2.0 s" in line          # the hold and the poll period are both named
    assert not any(isinstance(e, RoomChanged) for e in events)
    assert fab.where().room == ""


def test_a_glimpse_never_becomes_the_active_room():
    fab, clock = build(office=[False] * 4, kitchen=[False, True, False, False])
    pump(fab, clock, 4)
    assert fab.where().room == ""
    assert fab.glimpses == 1


def test_a_run_that_clears_the_hold_is_an_arrival_and_not_a_glimpse(caplog):
    caplog.set_level(logging.INFO, logger="roomfabric")
    events = []
    fab, clock = build(office=[False] * 6,
                       kitchen=[False, True, True, True, False, False],
                       publish=events.append)
    pump(fab, clock, 6)
    assert any(isinstance(e, RoomChanged) for e in events)
    assert fab.glimpses == 0
    assert not [r for r in caplog.records if "glimpse" in r.getMessage()]


# =====================================================================
# 16. UNREACHABLE IS NOT GLIMPSED
# =====================================================================
def test_a_room_that_never_answers_is_a_different_line(caplog):
    caplog.set_level(logging.INFO, logger="roomfabric")
    fab, clock = build(office=[False] * 12, kitchen=[None] * 12)
    pump(fab, clock, 12)
    said = [r.getMessage() for r in caplog.records
            if "kitchen" in r.getMessage() and "answer" in r.getMessage()]
    assert said, "an unanswered room must say so"
    assert fab.glimpses == 0
    assert not [r for r in caplog.records if "glimpse" in r.getMessage()]
    assert len(said) == 1, "once per silence, not once per poll"


def test_the_unanswered_line_and_the_glimpse_line_are_not_the_same_diagnosis(caplog):
    caplog.set_level(logging.INFO, logger="roomfabric")
    fab, clock = build(office=[False] * 12, kitchen=[None] * 12)
    pump(fab, clock, 12)
    text = caplog.text.lower()
    assert "never saw" not in text
    assert "we do not know" in text or "not answered" in text


# =====================================================================
# 17. THE GLIMPSE AS THE SEQUENCE'S KITCHEN STEP
# =====================================================================
def test_office_then_a_kitchen_glimpse_then_the_phone_is_a_departure():
    seq = pv.DepartureSequence()
    seq.room(room="office", at=100.0)
    seq.room(room="kitchen", at=140.0, glimpse=True)
    assert seq.phone_gone(at=300.0) is True
    assert seq.left is True
    note = pv.departure_note(seq)
    assert "glimpse" in note, "the note must name the evidence it used"


def test_a_kitchen_glimpse_with_no_office_step_first_does_not_arm():
    seq = pv.DepartureSequence()
    seq.room(room="kitchen", at=140.0, glimpse=True)
    assert seq.phone_gone(at=300.0) is False


def test_a_glimpse_too_long_after_the_office_does_not_arm():
    seq = pv.DepartureSequence()
    seq.room(room="office", at=100.0)
    seq.room(room="kitchen", at=100.0 + pv.SEQ_STEP_S + 1.0, glimpse=True)
    assert seq.phone_gone(at=300.0) is False


def test_a_glimpse_of_the_desk_room_does_not_re_arm_the_sequence():
    """Only a settled RoomChanged means he sat down. A glimpse of the office
    through the wall must not make a departure EASIER to reach."""
    seq = pv.DepartureSequence()
    seq.room(room="office", at=10.0, glimpse=True)
    seq.room(room="kitchen", at=20.0, glimpse=True)
    assert seq.phone_gone(at=100.0) is False


def test_the_note_still_names_a_real_kitchen_change_as_a_real_one():
    seq = pv.DepartureSequence()
    seq.room(room="office", at=100.0)
    seq.room(room="kitchen", at=140.0)
    assert seq.phone_gone(at=300.0) is True
    assert "glimpse" not in pv.departure_note(seq)


def test_the_false_departure_cost_over_a_day_of_synthetic_edges():
    """THE COST, MEASURED rather than asserted to be small.

    A day of his flat: 16 hours of desk work with a kitchen trip every 40
    minutes, and the radar reading through plasterboard often enough to
    throw one spurious kitchen glimpse an hour. A false departure needs a
    glimpse AND his phone to nap past the 900 s window inside it, which
    nothing in this day does -- so the count is what the test prints.
    """
    seq = pv.DepartureSequence()
    false_departures = 0
    real_departures = 0
    t = 0.0
    # 16 h, one event a minute: bounded at 960 iterations.
    for minute in range(960):
        t = minute * 60.0
        seq.room(room="office", at=t)
        if minute % 60 == 30:                      # a spurious through-wall glimpse
            seq.room(room="kitchen", at=t + 5.0, glimpse=True)
            if seq.phone_gone(at=t + 5.0 + pv.SEQ_WINDOW_S + 1.0):
                false_departures += 1
        if minute % 40 == 0 and minute:            # a real kitchen trip, he comes back
            seq.room(room="kitchen", at=t + 10.0)
            seq.room(room="office", at=t + 200.0)
    # ...and one genuine walk out at the end of the day.
    seq.room(room="office", at=t + 300.0)
    seq.room(room="kitchen", at=t + 340.0, glimpse=True)
    if seq.phone_gone(at=t + 400.0):
        real_departures += 1
    assert false_departures == 0, \
        "a glimpse alone is not a departure: the phone must go inside the window"
    assert real_departures == 1
    assert pv.DepartureSequence.speaks is False, \
        "the sequence has no speak path and must never grow one"


# =====================================================================
# 18. THE HINT SKIPS FAULTED AND STUCK ROOMS
# =====================================================================
def test_last_seen_room_skips_a_stuck_room():
    fab, clock = build(office=[True] * 3, kitchen=[False] * 3)
    fab.stuck_after_s = 1.0
    pump(fab, clock, 3)
    assert fab.room("office").stuck is True
    name, age = fab.last_seen_room()
    assert name != "office", "a sensor the voter has thrown out cannot be the hint"


def test_last_seen_room_skips_a_room_the_caller_names():
    fab, clock = build(office=[True] * 3, kitchen=[False] * 3)
    pump(fab, clock, 3)
    assert fab.last_seen_room()[0] == "office"
    assert fab.last_seen_room(skip=("office",))[0] == ""


def test_cell_12_does_not_claim_he_never_left_a_room_the_voter_dropped():
    """One verdict, two contradictory readings of one sensor: rooms_leg
    computed CLEAR precisely BECAUSE the office was faulted, and the hint
    then said the office had just seen him."""
    state, reason = pv.bedroom_split(last_room="", age_s=None)
    assert "never left the office" not in reason
    assert state == pv.BED


# =====================================================================
# 19. A RETURN IS NOT CONTINUITY
# =====================================================================
def test_a_return_after_two_and_a_half_hours_is_explained_as_a_return():
    state, reason = pv.bedroom_split(last_room="office", age_s=5.0,
                                     away_s=2.5 * 3600.0)
    assert state == pv.HOME
    assert "never left" not in reason
    assert "dropped a still body" not in reason
    assert "back" in reason
    assert "2 h" in reason, "the sentence must carry how long he was out"


def test_continuity_still_works_when_he_never_went_anywhere():
    state, reason = pv.bedroom_split(last_room="office", age_s=5.0, away_s=None)
    assert state == pv.HOME
    assert "never left the office" in reason


def test_the_return_sentence_reaches_the_verdict():
    out = pv.decide(phone=pv.PHONE_YES, camera=pv.CAM_BLIND,
                    rooms=pv.ROOMS_CLEAR, last_room="office",
                    last_room_age_s=5.0, away_s=9000.0)
    assert out.cell == 12
    assert out.home is True
    assert "never left" not in out.reason


# =====================================================================
# THE REGRESSION FENCE
# =====================================================================
def test_the_readings_and_the_stuck_set_are_unchanged_by_the_instrument():
    fab, clock = build(office=[True] * 3, kitchen=[False, True, False])
    pump(fab, clock, 3)
    assert fab.readings() == {"office": True, "kitchen": False}
    assert fab.anywhere() is True
