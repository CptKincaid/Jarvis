"""The sensor fabric (jarvis/roomfabric.py): three rooms fused into one
picture.

Nothing here opens a socket or a thread: the reader is a seam and the
fabric's clock is injected. The five things being held down are the ones
that cost him something if they are wrong -- the handoff does not flap at
a doorway, a dead room can never report an empty house, a fan cannot make
the house occupied for ever, three radars all get switched off by offline
mode, and "someone else is here" is refused rather than guessed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from jarvis import roomfabric as rf
from jarvis.roomfabric import (CERTAIN, STALE, UNKNOWN, HouseView, Room,
                               RoomFabric, RoomPolicy, RoomSpec, build,
                               from_readers, room_specs)

ON = '{"id":"binary_sensor-presence","value":true,"state":"ON"}'


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def tick(self, s):
        self.t += s


class Cfg:
    """The AssistantConfig surface roomfabric.py uses: get() over dotted keys."""

    def __init__(self, **flat):
        self.data = flat

    def get(self, key, default=None):
        return self.data.get(key, default)


class FakeSensor:
    """A RoomSensor stand-in: a value you set, and a read counter."""

    def __init__(self, value=None, url="http://10.0.0.1/binary_sensor/presence"):
        self.value, self.url, self.reads = value, url, 0
        self.configured, self.paused = True, False

    def read(self):
        self.reads += 1
        return self.value

    def status(self):
        return {"url": self.url, "reads": self.reads}


def fabric(names=("office", "kitchen", "bedroom"), clock=None, **kw):
    clock = clock or Clock()
    rooms = [Room(spec=RoomSpec(name=n, url="http://10.0.0.%d" % (i + 1),
                                label=n, primary=(i == 0)),
                  sensor=FakeSensor())
             for i, n in enumerate(names)]
    return RoomFabric(rooms, now=clock.now, **kw), clock


def see(fab, **values):
    for name, value in values.items():
        fab.room(name).sensor.value = value


def advance(fab, clock, seconds, step=2.0):
    """Run the fabric at its REAL cadence. Jumping the clock between ticks
    would fake a poll gap the running app never has, and the gap is itself
    a signal (Room.observe)."""
    end = clock.t + seconds
    while clock.t < end:
        clock.tick(step)
        fab.tick()
    return fab.where()


# ----------------------------------------------------------------- config
def test_rooms_list_is_the_plural_and_the_singular_still_works():
    """Shaped after gmail.accounts: a list, with the old keys as fallback."""
    single = Cfg(**{"presence.room_sensor_enabled": True,
                    "presence.room_sensor_url": "http://192.168.50.60"})
    specs = room_specs(single)
    assert [s.name for s in specs] == ["room"]
    assert specs[0].primary is True          # the only room is the primary

    many = Cfg(**{"presence.room_sensor_enabled": True,
                  "presence.room_sensor_url": "http://192.168.50.60",
                  "presence.rooms": [
                      {"name": "Office", "url": "http://192.168.50.60"},
                      {"name": "kitchen", "url": "http://192.168.50.61",
                       "primary": True}]})
    specs = room_specs(many)
    assert [s.name for s in specs] == ["office", "kitchen"]
    # The list WINS: the singular url is not appended as a third room.
    assert [s.primary for s in specs] == [False, True]


def test_master_switch_still_turns_the_whole_fabric_off():
    """docs/room-sensor.md section 10 must stay true with three sensors."""
    cfg = Cfg(**{"presence.room_sensor_enabled": False,
                 "presence.rooms": [{"name": "office", "url": "http://a"}]})
    assert room_specs(cfg) == []


@pytest.mark.parametrize("entry", [
    {"url": "http://a"},                       # no name
    {"name": "office"},                        # no url
    {"name": "office", "url": "http://a", "enabled": False},
    "not a dict",
])
def test_one_unfinished_room_does_not_take_the_others_down(entry):
    cfg = Cfg(**{"presence.room_sensor_enabled": True,
                 "presence.rooms": [entry,
                                    {"name": "kitchen", "url": "http://b"}]})
    assert [s.name for s in room_specs(cfg)] == ["kitchen"]


def test_a_duplicate_room_name_is_skipped_not_merged():
    cfg = Cfg(**{"presence.room_sensor_enabled": True,
                 "presence.rooms": [{"name": "office", "url": "http://a"},
                                    {"name": "Office ", "url": "http://b"}]})
    assert [s.url for s in room_specs(cfg)] == ["http://a"]


def test_first_room_becomes_primary_when_he_did_not_say():
    cfg = Cfg(**{"presence.room_sensor_enabled": True,
                 "presence.rooms": [{"name": "office", "url": "http://a"},
                                    {"name": "kitchen", "url": "http://b"}]})
    assert [s.primary for s in room_specs(cfg)] == [True, False]


# --------------------------------------------------------------- handoff
def test_walking_office_to_kitchen_hands_over_after_the_enter_hold():
    fab, clock = fabric()
    see(fab, office=True, kitchen=False, bedroom=False)
    advance(fab, clock, 300.0)             # a morning at the desk
    assert fab.where().room == "office"

    edge = clock.t
    see(fab, office=False, kitchen=True)   # he is in the kitchen
    clock.tick(2.0)
    fab.tick()
    assert fab.where().room == "office"    # too soon: the enter hold
    clock.tick(2.0)
    fab.tick()
    w = fab.where()
    assert (w.room, w.confidence) == ("kitchen", CERTAIN)
    # The whole handoff, edge to spoken room, inside two polls.
    assert clock.t - edge <= 4.0


def test_a_doorway_cannot_flap_the_active_room():
    """Standing in a doorway BOTH radars see him (they see through the
    wall). Without the switch floor, newest-edge alone ping-pongs."""
    fab, clock = fabric(enter_hold_s=2.0, switch_min_s=6.0)
    see(fab, office=True, kitchen=False, bedroom=False)
    advance(fab, clock, 10.0)
    assert fab.where().room == "office"

    changes = []
    for _ in range(12):                    # 48 s of the kitchen flickering
        clock.tick(2.0)
        see(fab, office=True, kitchen=True)
        fab.tick()
        clock.tick(2.0)
        see(fab, office=True, kitchen=False)
        fab.tick()
        changes.append(fab.where().room)
    assert set(changes) <= {"office", "kitchen"}
    assert changes.count("kitchen") <= 48 / 6


def test_a_pass_through_never_takes_the_room():
    """He crosses the kitchen to the back door: one poll of presence."""
    fab, clock = fabric()
    see(fab, office=True, kitchen=False, bedroom=False)
    advance(fab, clock, 10.0)
    see(fab, kitchen=True)
    clock.tick(1.0)
    fab.tick()                             # 1 s in the beam
    see(fab, kitchen=False)
    clock.tick(1.0)
    fab.tick()
    assert fab.where().room == "office"


def test_the_leave_hold_survives_him_leaning_out_of_the_beam():
    fab, clock = fabric()
    see(fab, office=True, kitchen=False, bedroom=False)
    advance(fab, clock, 10.0)
    see(fab, office=False)
    advance(fab, clock, 4.0)
    assert fab.where().confidence == CERTAIN       # inside leave_hold_s
    advance(fab, clock, 8.0)
    assert fab.where().confidence == STALE         # past it, but still named
    advance(fab, clock, 120.0)
    w = fab.where()
    assert (w.room, w.confidence) == ("", UNKNOWN)


def test_two_rooms_occupied_the_newest_edge_wins():
    fab, clock = fabric()
    see(fab, office=True, kitchen=False, bedroom=False)
    advance(fab, clock, 20.0)
    see(fab, kitchen=True)                          # both alight now
    advance(fab, clock, 8.0)                        # past the switch floor
    assert fab.where().room == "kitchen"
    assert set(fab.where().occupied) == {"office", "kitchen"}


# --------------------------------------------------------------- failures
def test_one_dead_room_can_never_report_an_empty_house():
    """The whole point: with three sensors "all empty" is a claim about
    coverage, and a room with an open breaker means we do not have it."""
    fab, _ = fabric()
    see(fab, office=False, kitchen=False, bedroom=None)   # bedroom unreachable
    fab.tick()
    assert fab.anywhere() is None                         # NOT False
    see(fab, bedroom=False)
    fab.tick()
    assert fab.anywhere() is False


def test_one_live_room_still_answers_yes_while_another_is_down():
    fab, _ = fabric()
    see(fab, office=True, kitchen=None, bedroom=None)
    fab.tick()
    assert fab.anywhere() is True


def test_a_room_that_raises_costs_only_itself():
    fab, _ = fabric()

    def raiser():
        raise OSError("wifi")
    fab.room("kitchen").sensor.read = raiser
    see(fab, office=True, bedroom=False)
    fab.tick()
    assert fab.anywhere() is True
    assert fab.where().unknown == ("kitchen",)


def test_a_stuck_room_is_dropped_and_taken_back_when_it_clears():
    """A pedestal fan inside the beam is the documented failure and it
    would otherwise make the house occupied for ever."""
    fab, clock = fabric(stuck_after_h=12.0)
    see(fab, office=False, kitchen=True, bedroom=False)
    advance(fab, clock, 13 * 3600.0)
    assert fab.room("kitchen").stuck is True
    assert fab.anywhere() is False               # the fan is not a person
    assert fab.where().room == ""
    see(fab, kitchen=False)
    fab.tick()
    assert fab.room("kitchen").stuck is False
    see(fab, kitchen=True)
    fab.tick()
    assert fab.anywhere() is True


def test_no_opinion_does_not_reset_the_enter_clock():
    """A two-poll network hiccup must not make him "arrive" in the room he
    was already sitting in -- but a real outage must not preserve the
    morning's edge either."""
    fab, clock = fabric()
    see(fab, office=True, kitchen=False, bedroom=False)
    fab.tick()
    started = fab.room("office").true_since
    see(fab, office=None)
    advance(fab, clock, 4.0)
    see(fab, office=True)
    clock.tick(2.0)
    fab.tick()
    assert fab.room("office").true_since == started

    see(fab, office=None)
    advance(fab, clock, 600.0)
    see(fab, office=True)
    clock.tick(2.0)
    fab.tick()
    assert fab.room("office").true_since == clock.t


# ------------------------------------------------------- what it cannot do
def test_someone_else_is_refused_not_guessed():
    fab, _ = fabric()
    see(fab, office=True, kitchen=True, bedroom=False)
    fab.tick()
    assert fab.others() is None
    assert "not a count" in fab.others_reason


# ------------------------------------------------------------ offline mode
def test_three_radars_all_register_with_the_sensing_policy(tmp_path):
    """Measured on 2026-09-02: WITHOUT the adapter, SensingPolicy.attach
    keys on the device NAME and drops the duplicate, so three RoomSensors
    collapse to one and offline mode cuts only the last one's power."""
    from jarvis.roomsensor import RoomSensor
    from jarvis.sensing import SensingPolicy

    state = tmp_path / "sensing.json"
    state.write_text(json.dumps({"version": 1, "offline": False, "until": None}))
    policy = SensingPolicy(cfg=None, path=state)
    posts: list = []
    for i in (1, 2, 3):
        RoomSensor("http://10.0.0.%d" % i, policy=policy,
                   power_url="http://10.0.0.%d/switch/radar_power" % i,
                   post=lambda u, t: posts.append(u))
    assert len(policy._devices) == 1                      # the collision

    policy2 = SensingPolicy(cfg=None, path=state)
    posts2: list = []
    for i, label in ((1, "office"), (2, "kitchen"), (3, "bedroom")):
        RoomSensor("http://10.0.0.%d" % i,
                   policy=RoomPolicy(policy2, label),
                   power_url="http://10.0.0.%d/switch/radar_power" % i,
                   post=lambda u, t: posts2.append(u))
    assert [d.name for d in policy2._devices] == \
        ["office radar", "kitchen radar", "bedroom radar"]
    out = policy2.disable(source="test")
    assert out.stopped == ("office radar", "kitchen radar", "bedroom radar")
    assert len(posts2) == 3                               # every radar cut


def test_the_spoken_line_reads_as_english():
    from jarvis.commander import _sensing_join
    assert _sensing_join(("office radar", "kitchen radar")) == \
        "the office radar and the kitchen radar"


def test_a_single_room_keeps_the_bare_device_name():
    """His current sentence -- "THE RADAR is down" -- must not change under
    him just because the plural exists."""
    calls = []
    policy = SimpleNamespace(attach=lambda n, s, p=None, r=None: calls.append(n),
                             allowed=lambda k: True)
    RoomPolicy(policy, "room", plural=False).attach("radar", lambda: True)
    RoomPolicy(policy, "office", plural=True).attach("radar", lambda: True)
    assert calls == ["radar", "office radar"]


def test_offline_mode_makes_every_room_have_no_opinion(tmp_path):
    from jarvis.roomsensor import RoomSensor
    from jarvis.sensing import SensingPolicy

    state = tmp_path / "sensing.json"
    state.write_text(json.dumps({"version": 1, "offline": False, "until": None}))
    policy = SensingPolicy(cfg=None, path=state)
    rooms = []
    for i, label in ((1, "office"), (2, "kitchen"), (3, "bedroom")):
        rooms.append(Room(
            spec=RoomSpec(name=label, url="http://10.0.0.%d" % i, label=label,
                          primary=(i == 1)),
            sensor=RoomSensor("http://10.0.0.%d" % i,
                              policy=RoomPolicy(policy, label),
                              get=lambda u, t: ON)))
    fab = RoomFabric(rooms)
    fab.tick()
    assert fab.anywhere() is True
    reads = [r.sensor.reads for r in rooms]

    policy.disable(source="test")
    fab.tick()
    # Not "the readings were ignored": no request was sent at all.
    assert [r.sensor.reads for r in rooms] == reads
    assert fab.anywhere() is None                   # never False
    assert fab.where().unknown == ("office", "kitchen", "bedroom")


# ------------------------------------------------------ the presence leg
def test_the_fast_loop_and_the_sentinel_do_not_both_poll():
    """Two callers on one RoomSensor would double the LAN traffic and race
    the breaker's counters for nothing."""
    fab, _ = fabric()
    view = HouseView(fab)
    see(fab, office=True, kitchen=False, bedroom=False)
    view.read()
    before = [r.sensor.reads for r in fab.rooms]
    assert before == [1, 1, 1]
    fab._thread = SimpleNamespace(is_alive=lambda: True)   # the loop is up
    assert view.read() is True
    assert [r.sensor.reads for r in fab.rooms] == before


def test_house_view_wears_the_room_sensor_shape():
    fab, _ = fabric()
    view = HouseView(fab)
    see(fab, office=True, kitchen=False, bedroom=False)
    assert view.configured is True
    assert view.read() is True
    see(fab, office=False)
    assert view.read() is False
    see(fab, kitchen=None)
    assert view.read() is None


def test_room_or_phone_needs_no_change_for_three_rooms():
    """The existing asymmetry is already the right rule for N rooms: a room
    seeing somebody beats a sleeping phone, three rooms seeing nobody never
    beats a phone that answers."""
    from jarvis.presence import RoomOrPhone

    fab, _ = fabric()
    view = HouseView(fab)
    asked = []

    def phone(ip, mac):
        asked.append(ip)
        return False

    see(fab, office=True, kitchen=False, bedroom=False)
    assert RoomOrPhone(view, phone)("10.0.0.9", "") is True
    assert asked == []                       # a hit ends the tick

    see(fab, office=False)
    assert RoomOrPhone(view, phone)("10.0.0.9", "") is False
    assert asked == ["10.0.0.9"]             # empty rooms defer to the phone

    see(fab, bedroom=None)
    RoomOrPhone(view, phone)("10.0.0.9", "")
    assert asked == ["10.0.0.9", "10.0.0.9"]  # a blind spot also defers


# ------------------------------------------------------------- building
def test_build_returns_none_when_nothing_is_configured():
    assert build(Cfg()) is None
    assert build(Cfg(**{"presence.room_sensor_enabled": True})) is None


def test_build_skips_a_room_whose_url_is_not_a_url():
    cfg = Cfg(**{"presence.room_sensor_enabled": True,
                 "presence.rooms": [{"name": "office", "url": "192.168.50.60"},
                                    {"name": "kitchen", "url": "http://b"}]})
    fab = build(cfg)
    assert [r.name for r in fab.rooms] == ["kitchen"]


def test_build_reads_the_timers_from_config():
    cfg = Cfg(**{"presence.room_sensor_enabled": True,
                 "presence.rooms": [{"name": "office", "url": "http://a"}],
                 "presence.rooms_enter_hold_s": 5,
                 "presence.rooms_poll_s": "nonsense"})
    fab = build(cfg)
    assert fab.enter_hold_s == 5.0
    assert fab.poll_s == rf.DEFAULT_POLL_S


def test_from_readers_takes_anything_that_reads():
    """The fusion has no opinion about the wire, so the satellite lane's
    Satellite drops in through the same hole a RoomSensor does."""
    a, b = FakeSensor(True), FakeSensor(False)
    fab = from_readers([("Office", a), ("Kitchen", b)])
    fab.tick()
    assert [r.name for r in fab.rooms] == ["office", "kitchen"]
    assert fab.anywhere() is True
    assert fab.where().occupied == ("office",)


def test_a_room_change_publishes_one_event_not_a_metronome():
    from jarvis.events import RoomChanged

    seen: list = []
    fab, clock = fabric(publish=seen.append)
    see(fab, office=True, kitchen=False, bedroom=False)
    advance(fab, clock, 20.0)
    assert len(seen) == 1
    assert isinstance(seen[0], RoomChanged)
    assert (seen[0].room, seen[0].previous) == ("office", "")
