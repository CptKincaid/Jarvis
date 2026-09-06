"""THE TWO FALSE AWAYS THAT FIRED ON HIS LIVE BOX ON 2026-09-06.

Both were measured off the running log, both had him at the desk, and both
are the SAME defect wearing two coats: a leg that could not answer was read
as a leg answering NO.

  15:10:57  he said "camera off for ten minutes". Sensing-off switched the
            ROOM radars off with the lens, so the rooms leg went
            UNREACHABLE, and the voter printed

              away (cell 24) -- rooms-unreachable, phone-no, mic-HEARD

            with him at the desk TALKING TO IT. Two things are wrong on
            that one line. A leg HE switched off is BLIND, never a "no" --
            that is this module's own stated asymmetry, violated by its own
            table. And the mic had HEARD him inside its window, which cell
            6 already treats as proof he is in the flat, but cell 24 never
            asks the mic at all.

  15:28:13  and again at 17:07:28: "away (cell 6)" at the desk. His phone
            was napping, the camera leg was dark, the mic had gone quiet
            past its 10-minute window, so the 15-minute corroboration
            expired and the latched-radar rule voted away. HE HAD TYPED A
            COMMAND AT 15:17. jarvis/deskpresence.py was measuring exactly
            that and no leg was reading it.

THE RULE BOTH HALVES OBEY, and it is his module's own:

    A LEG THAT CANNOT ANSWER VOTES UNKNOWN, NEVER "NO", and a leg that has
    him IN THE FLAT outvotes every leg that merely failed to find him.

The symmetry pin at the bottom is deliberate: the mic veto and the desk
veto are ONE guard with two inputs, and this repo has now had four defects
in two days from a guard built for one of a symmetric pair and never
applied to its twin.
"""
from __future__ import annotations

import itertools

import pytest

from jarvis import presence as pres
from jarvis import presencevote as pv
from jarvis import roomfabric as rf


class Cfg:
    def __init__(self, **kw):
        self._d = dict(kw)

    def get(self, key, default=None):
        return self._d.get(key, default)


class Reader:
    """A room reader a test scripts. ``blocked`` is the sensor's own word
    for "the sensing policy forbids the radar right now" -- the same string
    jarvis/roomsensor.RoomSensor.blocked answers with."""

    def __init__(self, value=None, blocked=""):
        self.value = value
        self.blocked = blocked
        self.configured = True
        self.reads = 0

    def read(self):
        self.reads += 1
        return None if self.blocked else self.value


def fabric(office=True, kitchen=False, blocked=""):
    return rf.from_readers(
        [("office", Reader(office, blocked)),
         ("kitchen", Reader(kitchen, blocked))],
        poll_s=2.0, enter_hold_s=2.0)


# Every (rooms, phone, camera) triple the table answers AWAY to, computed
# from the table rather than listed, so a new away cell joins these tests
# automatically instead of quietly escaping them.
AWAY_CELLS = tuple(
    key for key in itertools.product(
        (pv.ROOMS_ON, pv.ROOMS_CLEAR, pv.ROOMS_UNREACHABLE),
        (pv.PHONE_YES, pv.PHONE_NO, pv.PHONE_UNKNOWN),
        (pv.CAM_SAW, pv.CAM_LOOKED, pv.CAM_BLIND))
    if pv.decide(rooms=key[0], phone=key[1], camera=key[2],
                 agreed_s_ago=9999.0).state == pv.AWAY)


def test_the_away_cells_are_the_ones_his_log_named():
    """A sanity anchor for the parametrised tests below: if this list moves,
    the table moved, and the vetoes below are being asked a new question."""
    cells = sorted(pv.decide(rooms=r, phone=p, camera=c,
                             agreed_s_ago=9999.0).cell
                   for r, p, c in AWAY_CELLS)
    assert cells == [5, 6, 14, 15, 23, 24]


# =====================================================================
# (5a) SENSING OFF IS NOT A "NO"
# =====================================================================
@pytest.mark.parametrize("rooms,phone,camera", AWAY_CELLS)
def test_no_away_cell_survives_his_own_privacy_switch(rooms, phone, camera):
    """He switched a leg off. That is the absence of an answer, and the
    absence of an answer is never the answer "he is out"."""
    out = pv.decide(rooms=rooms, phone=phone, camera=camera,
                    agreed_s_ago=9999.0, sensing_off=True)
    assert out.state != pv.AWAY
    assert out.hold is True, "hold the last verdict; do not blank it"


def test_the_15_10_57_line_is_impossible():
    """His exact legs at 15:10:57, minus the mic (that half is below), so
    this pins the switch on its own."""
    out = pv.decide(rooms=pv.ROOMS_UNREACHABLE, phone=pv.PHONE_NO,
                    camera=pv.CAM_BLIND, mic=pv.MIC_SILENT, sensing_off=True)
    assert out.cell == 24
    assert out.state == pv.UNKNOWN and out.hold is True
    assert "switched off" in out.reason


def test_a_box_with_no_radar_at_all_still_reaches_cell_24():
    """THE COST OF THE GUARD, stated. Cell 24 is the phone-only away this
    box gave before any radar was bought, and it must still work: the guard
    is about a leg SWITCHED OFF, not about a leg that was never there."""
    out = pv.decide(rooms=pv.ROOMS_UNREACHABLE, phone=pv.PHONE_NO,
                    camera=pv.CAM_BLIND, mic=pv.MIC_SILENT, sensing_off=False)
    assert out.cell == 24 and out.state == pv.AWAY


def test_the_switch_never_manufactures_a_home():
    """It may only ever WITHHOLD an away. A cell that says home or bed says
    exactly the same thing with the switch off."""
    for rooms, phone, camera in itertools.product(
            (pv.ROOMS_ON, pv.ROOMS_CLEAR, pv.ROOMS_UNREACHABLE),
            (pv.PHONE_YES, pv.PHONE_NO, pv.PHONE_UNKNOWN),
            (pv.CAM_SAW, pv.CAM_LOOKED, pv.CAM_BLIND)):
        plain = pv.decide(rooms=rooms, phone=phone, camera=camera,
                          agreed_s_ago=9999.0)
        if plain.state == pv.AWAY:
            continue
        off = pv.decide(rooms=rooms, phone=phone, camera=camera,
                        agreed_s_ago=9999.0, sensing_off=True)
        assert (off.state, off.reason) == (plain.state, plain.reason)


def test_the_probe_reads_the_switch_off_the_rooms_themselves():
    """The leg's OWN word for it. ``RoomSensor.blocked`` is non-empty
    exactly while the sensing policy forbids the radar, and the fabric
    reports the house blocked only when EVERY configured room is."""
    fab = fabric(blocked="offline")
    probe = pres.ThreeLegProbe(fabric=fab, cfg=Cfg(),
                               phone=lambda ip, mac: False,
                               grace_s=0.0, boot_grace_s=0.0)
    probe("10.0.0.5", "")
    assert probe.legs["rooms"] == pv.ROOMS_UNREACHABLE
    assert probe.verdict.state != pv.AWAY
    assert probe.verdict.hold is True


def test_one_radar_still_looking_is_not_a_switched_off_house():
    """Half the house blocked is not the house blocked -- ``HouseView``'s
    own rule, and the reason a single unplugged sensor cannot mute away."""
    fab = rf.from_readers([("office", Reader(False, "offline")),
                           ("kitchen", Reader(False, ""))],
                          poll_s=2.0, enter_hold_s=2.0)
    assert fab.blocked == ""


# =====================================================================
# (5b) A MIC THAT HEARD HIM VETOES AWAY IN EVERY CELL
# =====================================================================
@pytest.mark.parametrize("rooms,phone,camera", AWAY_CELLS)
def test_a_turn_inside_the_window_vetoes_away_everywhere(rooms, phone, camera):
    """Cell 6 has consulted the mic since the voter shipped. The other five
    away cells never did, and 15:10:57 was cell 24."""
    out = pv.decide(rooms=rooms, phone=phone, camera=camera,
                    agreed_s_ago=9999.0,
                    mic=pv.MIC_HEARD, mic_s_ago=30.0)
    assert out.state == pv.HOME
    assert "mic" in out.reason


def test_the_15_10_57_line_with_his_actual_mic():
    """His legs at 15:10:57 exactly: the switch AND the heard turn. Either
    one alone must stop the away; both together must land on HOME, because
    a leg that has him in the flat beats a leg that merely went dark."""
    out = pv.decide(rooms=pv.ROOMS_UNREACHABLE, phone=pv.PHONE_NO,
                    camera=pv.CAM_BLIND, mic=pv.MIC_HEARD, mic_s_ago=4.0,
                    sensing_off=True)
    assert out.state == pv.HOME
    assert out.cell == 24, "the cell it came from is still reported"


def test_a_silent_mic_vetoes_nothing():
    """The veto is POSITIVE evidence only. Silence past the window is not
    evidence of an empty flat and must leave every cell where it was."""
    for rooms, phone, camera in AWAY_CELLS:
        for mic in (pv.MIC_SILENT, pv.MIC_UNKNOWN):
            out = pv.decide(rooms=rooms, phone=phone, camera=camera,
                            agreed_s_ago=9999.0, mic=mic, mic_s_ago=None)
            assert out.state == pv.AWAY, (rooms, phone, camera, mic)


# =====================================================================
# (6) THE DESK IS A LEG: A MAN TYPING IS HOME
# =====================================================================
def test_the_desk_leg_reads_idle_seconds_and_nothing_else():
    assert pv.desk_leg(idle_s=25.0, window_s=1500.0) == pv.DESK_AT
    assert pv.desk_leg(idle_s=1499.0, window_s=1500.0) == pv.DESK_AT
    assert pv.desk_leg(idle_s=1500.0, window_s=1500.0) == pv.DESK_IDLE
    assert pv.desk_leg(idle_s=9000.0, window_s=1500.0) == pv.DESK_IDLE


@pytest.mark.parametrize("bad", [None, "", "soon", float("nan"),
                                 float("inf"), -1.0])
def test_an_unreadable_idle_monitor_is_unknown_never_away(bad):
    """deskpresence.py's first rule, verbatim: a nonzero rc, a missing
    gdbus, a timeout or unparsable stdout is None -- "no signal" -- NEVER
    0.0, which would read as sitting right there, and never "away"."""
    assert pv.desk_leg(idle_s=bad, window_s=1500.0) == pv.DESK_UNKNOWN


@pytest.mark.parametrize("rooms,phone,camera", AWAY_CELLS)
def test_a_man_typing_is_never_away(rooms, phone, camera):
    out = pv.decide(rooms=rooms, phone=phone, camera=camera,
                    agreed_s_ago=9999.0,
                    desk=pv.DESK_AT, desk_s_ago=45.0)
    assert out.state == pv.HOME
    assert "keyboard" in out.reason or "mouse" in out.reason


@pytest.mark.parametrize("rooms,phone,camera", AWAY_CELLS)
def test_an_empty_chair_is_not_an_empty_flat(rooms, phone, camera):
    """DESK_IDLE must be worth exactly nothing. He reads papers at that
    desk and the bedroom has no sensor at all; an idle keyboard is not
    evidence he left the building, and reading it as one would be the same
    defect this file exists to kill, pointing the other way."""
    idle = pv.decide(rooms=rooms, phone=phone, camera=camera,
                     agreed_s_ago=9999.0, desk=pv.DESK_IDLE, desk_s_ago=None)
    blank = pv.decide(rooms=rooms, phone=phone, camera=camera,
                      agreed_s_ago=9999.0)
    assert (idle.state, idle.reason) == (blank.state, blank.reason)


def test_cell_6_takes_the_desk_exactly_as_it_takes_the_mic():
    """THE 15:28:13 AND 17:07:28 LINES. Phone napping, camera dark, mic
    silent past its window, the 15-minute corroboration expired -- and him
    typing."""
    away, _ = pv.cell6(agreed_s_ago=9999.0, mic=pv.MIC_SILENT)
    assert away == pv.AWAY, "the line he actually got"
    state, reason = pv.cell6(agreed_s_ago=9999.0, mic=pv.MIC_SILENT,
                             desk=pv.DESK_AT, desk_s_ago=660.0)
    assert state == pv.HOME
    assert "11 min" in reason, "say how stale the evidence is"


def test_the_mic_and_the_desk_are_one_guard_with_two_inputs():
    """THE SYMMETRY PIN. Four defects in two days on this repo came from a
    guard built for one of a pair and never applied to its twin. For every
    cell, a heard turn and a moved mouse must land on the same verdict."""
    for rooms, phone, camera in itertools.product(
            (pv.ROOMS_ON, pv.ROOMS_CLEAR, pv.ROOMS_UNREACHABLE),
            (pv.PHONE_YES, pv.PHONE_NO, pv.PHONE_UNKNOWN),
            (pv.CAM_SAW, pv.CAM_LOOKED, pv.CAM_BLIND)):
        heard = pv.decide(rooms=rooms, phone=phone, camera=camera,
                          agreed_s_ago=9999.0, mic=pv.MIC_HEARD,
                          mic_s_ago=60.0)
        typed = pv.decide(rooms=rooms, phone=phone, camera=camera,
                          agreed_s_ago=9999.0, desk=pv.DESK_AT,
                          desk_s_ago=60.0)
        assert heard.state == typed.state, (rooms, phone, camera)
        assert heard.cell == typed.cell


def test_a_leg_value_this_voter_does_not_know_cannot_cost_the_vote():
    """The desk is the fifth leg and, like the mic, it is not one of the
    three. A bad value there silences it and nothing else."""
    out = pv.decide(rooms=pv.ROOMS_ON, phone=pv.PHONE_YES,
                    camera=pv.CAM_LOOKED, desk="desk-banana")
    assert out.cell == 2 and out.state == pv.HOME


# =====================================================================
# THE PROBE END: THE NUMBER COMES OFF deskpresence.py
# =====================================================================
def test_the_probe_turns_idle_seconds_into_the_leg():
    probe = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(),
                               desk=lambda: 120.0)
    assert probe._desk_leg() == (pv.DESK_AT, 120.0)

    quiet = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(),
                               desk=lambda: 99999.0)
    assert quiet._desk_leg() == (pv.DESK_IDLE, None)

    none = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(), desk=None)
    assert none._desk_leg() == (pv.DESK_UNKNOWN, None)


def test_a_desk_reader_that_raises_is_unknown_not_away():
    def boom():
        raise RuntimeError("no session bus")

    probe = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(), desk=boom)
    assert probe._desk_leg() == (pv.DESK_UNKNOWN, None)


def test_the_desk_window_is_his_own_desk_away_number():
    """ONE NUMBER, ONE MEANING -- the same discipline the mic window
    follows. ``presence.desk_away_after_min`` already means "how long since
    the keyboard moved before the chair counts as empty"."""
    probe = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(),
                               desk=lambda: 0.0)
    assert probe.desk_window_s == pv.DEFAULT_DESK_WINDOW_S == 1500.0
    his = pres.ThreeLegProbe(
        fabric=fabric(), cfg=Cfg(**{"presence.desk_away_after_min": 40}),
        desk=lambda: 0.0)
    assert his.desk_window_s == 2400.0


def test_the_15_28_13_line_replayed_through_the_probe(tmp_path):
    """His legs at 15:28:13: the office radar occupied with its
    corroboration long expired, his phone napping past the 12-minute grace,
    the camera leg dark, the mic silent past its window -- and a command
    typed 11 minutes earlier.

    The chair goes empty at the end because a leg that WITHHOLDS an away
    must not LATCH one: the moment the desk stops answering, cell 6 is free
    to say away again on its own evidence.
    """
    from jarvis import stuckroom as sr

    clock = {"t": 5000.0}
    typed = {"idle": 11 * 60.0}
    stuck = sr.StuckRooms(path=tmp_path / "roomruns.json",
                          now=lambda: clock["t"], publish=lambda ev: None)
    office = Reader(False)
    fab = rf.from_readers([("office", office), ("kitchen", Reader(False))],
                          poll_s=2.0, enter_hold_s=2.0)
    probe = pres.ThreeLegProbe(
        fabric=fab, cfg=Cfg(), stuck=stuck,
        phone=lambda ip, mac: False, mic=lambda: 40 * 60.0,
        desk=lambda: typed["idle"], now=lambda: clock["t"],
        grace_s=0.0, boot_grace_s=0.0)
    # An ORDINARY rising edge: the office reads empty first, so the run that
    # follows has an honest beginning and is not the pre-existing latch the
    # boot grace holds on (tests/test_presence_boot_grace.py).
    probe("10.0.0.5", "")
    office.value = True
    probe("10.0.0.5", "")           # opens the run, corroborated by nothing
    clock["t"] += 20 * 60.0         # past the 15-minute corroboration window
    out = probe("10.0.0.5", "")
    assert probe.verdict.cell == 6
    assert probe.verdict.state == pv.HOME, "he was typing eleven minutes ago"
    assert out is True
    assert probe.legs["desk"] == pv.DESK_AT

    typed["idle"] = 3600.0          # the chair genuinely empty for an hour
    probe("10.0.0.5", "")
    assert probe.legs["desk"] == pv.DESK_IDLE
    assert probe.verdict.state == pv.AWAY


def test_the_15_10_57_line_replayed_through_the_probe():
    """Sensing off, phone silent, and him talking to it."""
    clock = {"t": 5000.0}
    probe = pres.ThreeLegProbe(
        fabric=fabric(blocked="offline"), cfg=Cfg(),
        phone=lambda ip, mac: False, mic=lambda: 4.0,
        now=lambda: clock["t"], grace_s=0.0, boot_grace_s=0.0)
    out = probe("10.0.0.5", "")
    assert probe.legs["rooms"] == pv.ROOMS_UNREACHABLE
    assert probe.legs["mic"] == pv.MIC_HEARD
    assert probe.verdict.state == pv.HOME
    assert out is True


def test_the_sentinel_never_publishes_an_away_while_sensing_is_off():
    """END TO END, because the vote is only half of it: the sentinel is
    what turns a verdict into the Presence event the greeter reads."""
    clock = {"t": 5000.0}
    published = []
    probe = pres.ThreeLegProbe(
        fabric=fabric(blocked="offline"), cfg=Cfg(),
        phone=lambda ip, mac: False, now=lambda: clock["t"],
        grace_s=0.0, boot_grace_s=0.0)
    s = pres.PresenceSentinel(
        Cfg(**{"presence.phone_ip": "192.168.50.34"}),
        publish=published.append, probe_fn=probe,
        now=lambda: clock["t"], poll_s=60.0)
    for _ in range(6):
        clock["t"] += 900.0
        s.tick()
    assert published == []
    assert s.state != "away"


def test_a_sensor_only_box_gets_both_guards_too():
    """``decide_rooms_only`` is the ONE-LEG install, which has LESS to
    outvote a wrong away with, not more. Leaving the guards off that path
    would be exactly the guard-one-half shape this file's symmetry pin
    exists to catch."""
    plain = pv.decide_rooms_only(rooms=pv.ROOMS_CLEAR)
    assert plain.state == pv.AWAY, "unchanged when nothing witnessed him"

    heard = pv.decide_rooms_only(rooms=pv.ROOMS_CLEAR, mic=pv.MIC_HEARD,
                                 mic_s_ago=20.0)
    typed = pv.decide_rooms_only(rooms=pv.ROOMS_CLEAR, desk=pv.DESK_AT,
                                 desk_s_ago=20.0)
    assert heard.state == typed.state == pv.HOME

    off = pv.decide_rooms_only(rooms=pv.ROOMS_CLEAR, sensing_off=True)
    assert off.state == pv.UNKNOWN and off.hold is True

    idle = pv.decide_rooms_only(rooms=pv.ROOMS_CLEAR, desk=pv.DESK_IDLE)
    assert idle.state == pv.AWAY, "an empty chair is still not an empty flat"


def test_the_log_line_names_the_desk_leg():
    """He reads these lines. A leg that decided a vote has to appear in the
    sentence that reports it."""
    probe = pres.ThreeLegProbe(fabric=fabric(), cfg=Cfg(),
                               phone=lambda ip, mac: True,
                               desk=lambda: 30.0, grace_s=0.0,
                               boot_grace_s=0.0)
    probe("10.0.0.5", "")
    assert probe.legs["desk"] == pv.DESK_AT
