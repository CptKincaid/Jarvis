"""The still-distance filter, wired into the fabric (jarvis/roomfabric.py).

His office LD2410 reads OCCUPIED for ever -- 527 of 527 samples with the
flat empty -- because everything past his chair is a hard reflector. The
bit carries nothing; the still DISTANCE does (jarvis/roomstill.py, every
threshold from his two 2026-09-11 recordings). These tests hold down the
WIRING: metres become centimetres, the window is keyed to the radar's raw
run and never to its own verdict, a missing distance keeps the sensor's
word, and the corrected bit is what the vote and the house see.

Nothing here opens a socket: the reader is a seam and the clock is
injected, as in tests/test_roomfabric.py.
"""
from __future__ import annotations

import csv
import itertools
import logging
from pathlib import Path

import pytest

from jarvis import presencevote, roomsensor
from jarvis import roomstill as R
from jarvis.roomfabric import Room, RoomFabric, RoomSpec, room_specs

EMPTY_CSV = Path.home() / "room-trace-EMPTY-FLAT.csv"
DESK_CSV = Path.home() / "room-trace-AT-DESK.csv"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def tick(self, s):
        self.t += s


class Cfg:
    def __init__(self, **flat):
        self.data = flat

    def get(self, key, default=None):
        return self.data.get(key, default)


class StillSensor:
    """A RoomSensor stand-in: the bit, and a still distance in METRES --
    which is how ``RoomSensor.read_distance`` reports it."""

    def __init__(self, value=None, metres=None,
                 url="http://10.0.0.1/binary_sensor/presence"):
        self.value, self.url = value, url
        self._metres = metres
        self.reads = self.dist_reads = 0
        self.entities: list = []
        self.raise_distance = False
        self.configured, self.paused = True, False

    def read(self):
        self.reads += 1
        return self.value

    def read_distance(self, entity=roomsensor.DETECTION_ENTITY):
        self.dist_reads += 1
        self.entities.append(entity)
        if self.raise_distance:
            raise OSError("radar went away")
        m = self._metres
        return next(m) if hasattr(m, "__next__") else m

    def status(self):
        return {"url": self.url, "reads": self.reads}


class BitOnly:
    """A reader with no distance at all -- the satellite lane's shape."""

    def __init__(self, value=None):
        self.value = value
        self.configured, self.paused = True, False

    def read(self):
        return self.value


def pinned():
    """His empty office: 306-313 cm, a 7 cm spread."""
    return itertools.cycle([3.06, 3.08, 3.10, 3.13, 3.11, 3.07])


def wandering():
    """Him at the desk: 288-337 cm in his recording, a 49 cm spread; this
    synthetic run keeps the same spread."""
    return itertools.cycle([2.99, 3.20, 3.47, 3.10, 3.33])


def fabric(office=None, kitchen=None, clock=None, office_still=True, **kw):
    clock = clock or Clock()
    rooms = [
        Room(spec=RoomSpec(name="office", url="http://10.0.0.1", label="office",
                           primary=True, still_check=office_still),
             sensor=office if office is not None else StillSensor()),
        Room(spec=RoomSpec(name="kitchen", url="http://10.0.0.2", label="kitchen"),
             sensor=kitchen if kitchen is not None else StillSensor(False, None)),
    ]
    return RoomFabric(rooms, now=clock.now, **kw), clock


def run(fab, clock, polls, step=2.0):
    """Poll at the fabric's real cadence; the window is calibrated in wall
    clock, so the clock must move the way the running app's does."""
    for _ in range(polls):
        clock.tick(step)
        fab.tick()
    return fab.readings()


# ---------------------------------------------------------- the ghost
def test_a_pinned_still_distance_reads_the_office_empty_once_the_window_fills():
    """Judged on the SPAN of the evidence: 60 polls (120 s) is not yet a
    window, 100 polls (200 s) is."""
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office)
    assert run(fab, clock, 60)["office"] is True                  # too little to say
    assert run(fab, clock, 40)["office"] is False                 # a fixture
    assert fab.anywhere() is False
    assert fab.room("office").still_verdict == R.FIXTURE


def test_the_distance_asked_for_is_the_still_one_and_only_while_the_bit_is_on():
    office = StillSensor(True, pinned())
    kitchen = StillSensor(False, pinned())
    fab, clock = fabric(office=office, kitchen=kitchen)
    run(fab, clock, 10)
    assert set(office.entities) == {roomsensor.STILL_ENTITY}
    assert kitchen.dist_reads == 0        # an empty room costs one GET, not two


def test_a_wandering_still_distance_keeps_the_office_occupied():
    """In METRES. Unconverted, 0.48 m would read as 'under 20' and the man
    at his desk would become a fixture -- the false empty this lane exists
    to prevent."""
    fab, clock = fabric(office=StillSensor(True, wandering()))
    assert run(fab, clock, 120)["office"] is True
    assert fab.room("office").still_verdict == R.PERSON


# ----------------------------------------------------- his recordings
def _trace(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _replay(rows):
    """Both radars driven from one of his recordings, the fabric's clock
    following the recording's own timestamps."""
    class Playback:
        def __init__(self, room):
            self.room, self.row = room, None
            self.configured, self.paused = True, False
            self.dist_reads = 0

        def read(self):
            return self.row[f"{self.room}_presence"] == "True"

        def read_distance(self, entity=roomsensor.DETECTION_ENTITY):
            self.dist_reads += 1
            cm = self.row[f"{self.room}_still_cm"]
            return None if cm in ("", "None") else float(cm) / 100.0

        def status(self):
            return {}

    office, kitchen = Playback("office"), Playback("kitchen")
    clock = Clock(float(rows[0]["t"]))
    fab, _ = fabric(office=office, kitchen=kitchen, clock=clock)
    seen = []
    for row in rows:
        office.row = kitchen.row = row
        clock.t = float(row["t"])
        fab.tick()
        seen.append(dict(fab.readings()))
    return fab, seen, office, kitchen


@pytest.mark.skipif(not EMPTY_CSV.exists(), reason="his empty-flat recording is not on this box")
def test_his_empty_flat_recording_ends_with_the_office_read_empty():
    rows = _trace(EMPTY_CSV)
    fab, seen, office, kitchen = _replay(rows)
    assert seen[-1] == {"office": False, "kitchen": False}
    # Once there was enough to judge on it never wavered, and the judgement
    # came inside one window of the start.
    first = next(i for i, s in enumerate(seen) if s["office"] is False)
    assert float(rows[first]["t"]) - float(rows[0]["t"]) <= R.WINDOW_S + 5.0
    assert not any(s["office"] for s in seen[first:])
    assert kitchen.dist_reads == 0          # the kitchen read empty, raw


@pytest.mark.skipif(not DESK_CSV.exists(), reason="his at-desk recording is not on this box")
def test_his_at_desk_recording_keeps_him_in_the_office_throughout():
    fab, seen, office, kitchen = _replay(_trace(DESK_CSV))
    assert all(s["office"] is True for s in seen)
    assert fab.room("office").still_verdict == R.PERSON


# ------------------------------------------------ the window's lifetime
def test_a_false_reading_forgets_the_window_so_the_next_run_starts_unknown():
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office)
    assert run(fab, clock, 100)["office"] is False
    office.value = False
    run(fab, clock, 1)
    office.value = True
    assert run(fab, clock, 1)["office"] is True      # one reading is no verdict
    assert fab.room("office").still.samples == 1


def test_a_silence_past_the_gap_limit_forgets_the_window_too():
    """The twin of the test above (the guard-one-half pattern): observe()
    ends a run on a long silence, and the window must end with it."""
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office)
    assert run(fab, clock, 100)["office"] is False
    office.value = None
    gap = fab.room("office").gap_limit_s
    run(fab, clock, int(gap // 2.0) + 2)             # past the gap, at cadence
    office.value = True
    assert run(fab, clock, 1)["office"] is True
    assert fab.room("office").still.samples == 1


def test_a_two_poll_hiccup_keeps_the_window():
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office)
    assert run(fab, clock, 100)["office"] is False
    office.value = None
    run(fab, clock, 2)
    office.value = True
    assert run(fab, clock, 1)["office"] is False      # still a fixture
    assert fab.room("office").still.samples > R.MIN_SAMPLES


def test_a_fixture_verdict_never_erases_its_own_evidence():
    """If the corrected False ended the run and cleared the window, the
    office would flap True for 100 s and False for a moment, for ever."""
    fab, clock = fabric(office=StillSensor(True, pinned()))
    run(fab, clock, 100)
    seen = [run(fab, clock, 1)["office"] for _ in range(200)]   # 400 s, past the window
    assert seen == [False] * 200


# ----------------------------------------------- when it cannot judge
def test_no_distance_at_all_keeps_the_sensors_word():
    fab, clock = fabric(office=StillSensor(True, None))
    assert run(fab, clock, 120)["office"] is True
    assert fab.room("office").still_verdict == R.UNKNOWN


def test_a_distance_read_that_raises_costs_nothing():
    office = StillSensor(True, pinned())
    office.raise_distance = True
    fab, clock = fabric(office=office)
    assert run(fab, clock, 120)["office"] is True
    assert fab.polls == 120


def test_a_reader_with_no_distance_is_left_alone():
    fab, clock = fabric(office=BitOnly(True))
    assert run(fab, clock, 120)["office"] is True


# ------------------------------------------------------------- config
def test_still_check_is_on_by_default_and_off_per_room_by_the_entry():
    cfg = Cfg(**{"presence.room_sensor_enabled": True,
                 "presence.rooms": [
                     {"name": "office", "url": "http://10.0.0.1", "still_check": False},
                     {"name": "kitchen", "url": "http://10.0.0.2"}]})
    assert [s.still_check for s in room_specs(cfg)] == [False, True]
    assert RoomSpec(name="x").still_check is True


def test_a_room_switched_off_in_config_is_not_filtered():
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office, office_still=False)
    assert run(fab, clock, 120)["office"] is True
    assert office.dist_reads == 0
    assert fab.room("office").still is None


# -------------------------------------------------------------- the log
def test_one_line_per_verdict_change_not_per_poll(caplog):
    caplog.set_level(logging.INFO, logger="jarvis.roomfabric")
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office)
    run(fab, clock, 200)
    fixture = [r.getMessage() for r in caplog.records if "fixture" in r.getMessage()]
    assert len(fixture) == 1
    assert "office" in fixture[0] and "cm" in fixture[0]
    office._metres = wandering()
    run(fab, clock, 200)
    body = [r.getMessage() for r in caplog.records if "moving again" in r.getMessage()]
    assert len(body) == 1
    assert fab.readings()["office"] is True


def test_status_says_why():
    fab, clock = fabric(office=StillSensor(True, pinned()))
    run(fab, clock, 100)
    st = fab.room("office").status()["still"]
    assert st["verdict"] == R.FIXTURE
    assert st["corrected"] is True
    assert st["samples"] >= R.MIN_SAMPLES
    assert 0 <= st["spread_cm"] < R.SPREAD_CM


# ---------------------------------------------------------- downstream
def test_the_vote_sees_the_corrected_office():
    fab, clock = fabric(office=StillSensor(True, pinned()))
    run(fab, clock, 5)
    assert presencevote.rooms_leg(fab.readings(), fab.stuck_rooms()) == presencevote.ROOMS_ON
    run(fab, clock, 100)
    assert presencevote.rooms_leg(fab.readings(), fab.stuck_rooms()) == presencevote.ROOMS_CLEAR


def test_the_slow_stuck_latch_yields_to_the_fast_one_and_still_covers_a_blind_radar():
    fab, clock = fabric(office=StillSensor(True, pinned()), stuck_after_h=0.02)  # 72 s
    run(fab, clock, 200)
    assert fab.room("office").stuck is False        # corrected, never latched
    assert fab.readings()["office"] is False
    blind, clock2 = fabric(office=StillSensor(True, None), stuck_after_h=0.02)
    run(blind, clock2, 200)
    assert blind.room("office").stuck is True       # the old latch still works


# ------------------------------------------ the attack round, 2026-09-12
def test_a_dead_distance_leg_returns_the_sensors_word_within_the_window(caplog):
    """A FIXTURE verdict must not outlive its evidence. When the still
    distance stops answering, the window ages out by the clock and the
    room goes back to the bit -- and the log says the check went blind,
    once, and once when it can see again."""
    caplog.set_level(logging.INFO, logger="jarvis.roomfabric")
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office)
    assert run(fab, clock, 100)["office"] is False
    office._metres = None                                   # the entity died
    seen = [run(fab, clock, 1)["office"] for _ in range(200)]   # 400 s
    assert seen[-1] is True
    assert seen.index(True) * 2.0 <= R.WINDOW_S + 4.0        # back inside one window
    assert fab.room("office").still_verdict == R.UNKNOWN
    blind = [r.getMessage() for r in caplog.records if "blind" in r.getMessage()]
    assert len(blind) == 1 and "office" in blind[0]
    office._metres = pinned()                               # and it came back
    run(fab, clock, 100)
    back = [r.getMessage() for r in caplog.records if "can see again" in r.getMessage()]
    assert len(back) == 1
    assert fab.readings()["office"] is False


def test_a_slow_poll_cadence_is_warned_about_once(caplog):
    """WINDOW_S is wall clock, MIN_SAMPLES a count: above 3.6 s a poll the
    window can never fill and the filter would be silently inert."""
    caplog.set_level(logging.WARNING, logger="jarvis.roomfabric")
    fabric(office=StillSensor(True, pinned()), poll_s=4.0)
    slow = [r.getMessage() for r in caplog.records if "still" in r.getMessage()]
    assert len(slow) == 1 and "4.0" in slow[0]
    caplog.clear()
    fabric(office=StillSensor(True, pinned()), poll_s=2.0)
    assert not [r for r in caplog.records if "still" in r.getMessage()]


def test_a_fixture_run_that_was_never_a_person_leaves_no_hint_and_no_glimpse(caplog):
    """The boot ghost: the office reads occupied until the window can
    judge, then FIXTURE. Those readings were the furniture's, so they may
    not become the bedroom hint (the 2026-09-06 stuck-room lesson, twin)
    nor be logged as a pass-through glimpse."""
    caplog.set_level(logging.INFO, logger="jarvis.roomfabric")
    published = []
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office, publish=published.append)
    run(fab, clock, 100)
    assert fab.readings()["office"] is False
    assert fab.last_seen_room() == ("", None)
    assert not [r for r in caplog.records if "glimpsed" in r.getMessage()]
    assert not [e for e in published if type(e).__name__ == "RoomGlimpsed"]


def test_a_real_sitting_keeps_its_hint_when_he_goes_still():
    """The other half: he was a PERSON in the run, then sat dead still for
    three minutes. The window says fixture now, but the sighting was real
    and the hint must still name the office."""
    office = StillSensor(True, wandering())
    fab, clock = fabric(office=office)
    run(fab, clock, 100)
    assert fab.readings()["office"] is True
    office._metres = pinned()
    run(fab, clock, 100)
    assert fab.readings()["office"] is False
    room, age = fab.last_seen_room()
    assert room == "office" and age is not None


def test_a_gap_clear_says_nothing_about_moving(caplog):
    caplog.set_level(logging.INFO, logger="jarvis.roomfabric")
    office = StillSensor(True, pinned())
    fab, clock = fabric(office=office)
    run(fab, clock, 100)
    office.value = None
    run(fab, clock, int(fab.room("office").gap_limit_s // 2.0) + 2)
    office.value = True
    run(fab, clock, 1)
    assert fab.room("office").still_verdict == R.UNKNOWN
    assert not [r for r in caplog.records if "moving again" in r.getMessage()]


def test_status_value_is_the_corrected_bit_even_when_the_sensor_reports_its_own():
    class Chatty(StillSensor):
        def status(self):
            return {"url": self.url, "reads": self.reads, "value": self.value}
    fab, clock = fabric(office=Chatty(True, pinned()))
    run(fab, clock, 100)
    st = fab.room("office").status()
    assert st["value"] is False and st["raw"] is True
    assert st["still"]["corrected"] is True


def test_the_singular_config_has_a_still_check_twin():
    cfg = Cfg(**{"presence.room_sensor_enabled": True,
                 "presence.room_sensor_url": "http://10.0.0.9",
                 "presence.room_sensor_still_check": False})
    assert [s.still_check for s in room_specs(cfg)] == [False]


def test_still_check_typed_as_the_string_false_is_off():
    cfg = Cfg(**{"presence.room_sensor_enabled": True,
                 "presence.rooms": [
                     {"name": "office", "url": "http://10.0.0.1", "still_check": "false"},
                     {"name": "kitchen", "url": "http://10.0.0.2", "still_check": "off"}]})
    assert [s.still_check for s in room_specs(cfg)] == [False, False]
