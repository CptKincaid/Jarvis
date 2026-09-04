"""Zones (jarvis/zones.py): where in the room, and the record of it.

Nothing here opens a socket, a lens or a real state directory.
``RoomSensor(get=...)`` is the transport seam it already had, the camera is
a callable that returns a LABEL and never an image, and every log test
writes into ``tmp_path``.

The three things worth pinning, in the order they will break:

* **precedence** -- the camera overrules the radar, and a radar with no
  opinion is never read as an empty room;
* **hysteresis** -- a distance sitting on a band edge chatters, and the
  dwell has to turn twenty crossings into one transition;
* **the record** -- one JSONL line per committed change, 0600, capped, and
  carrying a face LABEL and nothing the lens saw.
"""
from __future__ import annotations

import copy
import json
import os
import stat

import pytest

from jarvis import zones as zn
from jarvis.assistant_config import DEFAULTS
from jarvis.roomsensor import RoomSensor
from jarvis.zones import (ABSENT, NO_OPINION, RULE_BAND, RULE_CAMERA,
                          RULE_EMPTY, RULE_SILENT, RULE_UNPLACED, UNPLACED,
                          Band, CameraOpinion, ZoneLog, ZoneMap, ZoneTracker,
                          ZoneWatcher, verdict)

ESPHOME_ON = '{"id":"binary_sensor/Presence","value":true,"state":"ON"}'
ESPHOME_OFF = '{"id":"binary_sensor/Presence","value":false,"state":"OFF"}'


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def tick(self, s):
        self.t += s


class Http:
    """Fake transport, keyed by URL so the order of reads is not the test."""

    def __init__(self, answers=None, default=None):
        self.answers = dict(answers or {})
        self.default = default
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        answer = self.answers.get(url, self.default)
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            raise OSError("no answer scripted for %s" % url)
        return answer

    @property
    def calls(self):
        return len(self.urls)


def office_map():
    return ZoneMap.office()


# ------------------------------------------------------------------ bands
def test_a_band_is_half_open_so_touching_bands_neither_overlap_nor_leave_a_gap():
    band = Band("mid", 1.5, 3.0)
    assert band.holds(1.5) and band.holds(2.99)
    assert not band.holds(3.0)          # 3.0 belongs to the NEXT band
    assert not band.holds(1.49)


def test_two_bands_that_overlap_are_refused_when_the_map_is_built():
    with pytest.raises(ValueError, match="overlap"):
        ZoneMap("office", (Band("a", 0.75, 2.0), Band("b", 1.5, 3.0)))


def test_a_band_whose_far_edge_is_not_past_its_near_edge_is_refused():
    with pytest.raises(ValueError, match="past"):
        ZoneMap("office", (Band("a", 2.0, 2.0),))


def test_two_bands_with_one_name_are_refused_because_a_verdict_must_name_a_place():
    with pytest.raises(ValueError, match="twice"):
        ZoneMap("office", (Band("a", 0.75, 1.5), Band("a", 1.5, 3.0)))


def test_a_map_with_no_bands_at_all_is_refused():
    with pytest.raises(ValueError, match="no bands"):
        ZoneMap("office", ())


def test_the_gaps_are_named_including_the_blind_zone_and_everything_past_the_far_gate():
    # "A reading that falls in no band is unplaced, which is a real answer,
    # not an error" -- so the gaps have to be inspectable, not inferred.
    gaps = office_map().gaps()
    assert (0.0, 0.75) in gaps                       # the module's blind zone
    assert any(lo == 4.5 and hi is None for lo, hi in gaps)   # past the far gate
    assert len(gaps) == 2                            # and no accidental ones


def test_the_office_bands_are_gate_aligned_because_the_ld2410_resolves_no_finer():
    # One distance gate is 0.75 m (scripts/room_sensor.py GATE_M). An edge
    # inside a gate is finer than the sensor and would be a fiction.
    for band in office_map().bands:
        assert abs(band.near_m / zn.GATE_M - round(band.near_m / zn.GATE_M)) < 1e-9
        assert abs(band.far_m / zn.GATE_M - round(band.far_m / zn.GATE_M)) < 1e-9


def test_the_office_map_starts_at_the_blind_zone_and_ends_at_the_configured_far_gate():
    m = office_map()
    assert m.bands[0].near_m == zn.BLIND_M == 0.75
    assert m.bands[-1].far_m == 4.5      # max move gate 6 x 0.75 m, measured 09-03


def test_no_office_band_is_the_desk_because_the_radar_is_on_the_desk_facing_away():
    # The profile: "on the desk at the BACK edge, aimed OUT across the room
    # at the door -- not at the chair". Sitting, he is behind the module and
    # inside its 0.75 m blind zone, so the nearest band is the floor in
    # FRONT of the desk, not the chair. The chair belongs to the camera.
    m = office_map()
    assert m.camera_zone == "at the desk"
    assert "at the desk" not in [b.name for b in m.bands]


def test_placing_a_distance_names_the_band_or_nothing_at_all():
    m = office_map()
    assert m.place(1.0) == "just off the desk"
    assert m.place(2.4) == "the middle of the room"
    assert m.place(3.6) == "by the door"          # measured walk topped at 3.6 m
    assert m.place(0.4) is None                   # blind zone
    assert m.place(5.2) is None                   # past the far gate
    assert m.place(None) is None


# -------------------------------------------------------------- precedence
def test_the_camera_overrules_the_radar_even_when_the_radar_says_the_room_is_empty():
    # His words: "camera recognition overrules sensor detection since he can
    # literally see me at my desk".
    v = verdict(office_map(), presence=False, distance_m=None,
                camera=CameraOpinion(known=True, label="hunter"))
    assert v.zone == "at the desk"
    assert v.rule == RULE_CAMERA


def test_the_camera_overrules_a_radar_that_places_him_somewhere_else():
    v = verdict(office_map(), presence=True, distance_m=3.6,
                camera=CameraOpinion(known=True, label="hunter"))
    assert (v.zone, v.rule) == ("at the desk", RULE_CAMERA)


def test_a_camera_that_recognised_nobody_is_not_a_veto_and_the_radar_still_places_him():
    # The lens is dark, or he is turned away. That is not evidence of an
    # empty desk, so it may only ever ADD -- exactly like the room sensor
    # next door, which can make him home sooner and never make him away.
    v = verdict(office_map(), presence=True, distance_m=2.0,
                camera=CameraOpinion(known=False))
    assert (v.zone, v.rule) == ("the middle of the room", RULE_BAND)


def test_presence_with_a_distance_in_a_band_is_that_band():
    v = verdict(office_map(), presence=True, distance_m=1.0)
    assert (v.zone, v.rule) == ("just off the desk", RULE_BAND)


def test_presence_with_a_distance_in_no_band_is_in_the_room_but_unplaced():
    v = verdict(office_map(), presence=True, distance_m=0.4)
    assert (v.zone, v.rule) == (UNPLACED, RULE_UNPLACED)
    assert v.zone == "in the room, unplaced"


def test_presence_with_no_distance_at_all_is_unplaced_and_never_a_guess():
    v = verdict(office_map(), presence=True, distance_m=None)
    assert (v.zone, v.rule) == (UNPLACED, RULE_UNPLACED)


def test_presence_false_is_not_in_the_room():
    v = verdict(office_map(), presence=False, distance_m=None)
    assert (v.zone, v.rule) == (ABSENT, RULE_EMPTY)


def test_a_radar_with_no_opinion_is_never_read_as_not_in_the_room():
    # roomsensor.read() returns None for a timeout, a 404, offline mode or
    # an open breaker. Collapsing that into "empty" is how Jarvis goes
    # silent on a man sitting in the room.
    v = verdict(office_map(), presence=None, distance_m=None)
    assert v.zone == NO_OPINION
    assert v.zone != ABSENT
    assert v.rule == RULE_SILENT


# -------------------------------------------------------------- hysteresis
def _tracker(clock, **kw):
    kw.setdefault("dwell_s", 3.0)
    return ZoneTracker("office", office_map(), now=clock.now,
                       wall=lambda: 1_756_000_000.0 + clock.now(), **kw)


def test_a_zone_change_only_commits_after_the_dwell_has_elapsed():
    c = Clock()
    t = _tracker(c)
    assert t.observe(presence=True, distance_m=2.0) is None   # candidate only
    assert t.zone == NO_OPINION
    c.tick(2.0)
    assert t.observe(presence=True, distance_m=2.0) is None   # 2.0s < 3.0s
    c.tick(2.0)
    change = t.observe(presence=True, distance_m=2.0)         # 4.0s >= 3.0s
    assert change is not None
    assert (change.old, change.new) == (NO_OPINION, "the middle of the room")
    assert change.held_s == pytest.approx(4.0)


def test_a_candidate_that_changes_before_the_dwell_elapses_starts_the_clock_again():
    c = Clock()
    t = _tracker(c)
    t.observe(presence=True, distance_m=2.0)
    c.tick(2.0)
    t.observe(presence=True, distance_m=1.0)      # a different candidate
    c.tick(2.0)
    # 2.0s into the SECOND candidate, not 4.0s into the first: the clock
    # restarted, so nothing has committed yet.
    assert t.observe(presence=True, distance_m=1.0) is None
    assert t.zone == NO_OPINION
    c.tick(1.5)
    assert t.observe(presence=True, distance_m=1.0) is not None
    assert t.zone == "just off the desk"


def test_a_reading_oscillating_across_a_band_edge_makes_one_transition_not_twenty():
    # The chatter case, pinned. 1.45 m and 1.55 m straddle the 1.50 m edge
    # between "just off the desk" and "the middle of the room"; a naive
    # commit-on-every-reading would log one line per sample.
    c = Clock()
    t = _tracker(c)
    for _ in range(6):                       # settle in the middle of the room
        c.tick(1.0)
        t.observe(presence=True, distance_m=2.4)
    assert t.zone == "the middle of the room"

    naive, committed, last = 0, [], t.zone
    for i in range(20):
        c.tick(0.4)                          # faster than the dwell, on purpose
        metres = 1.45 if i % 2 else 1.55
        naive += 1 if office_map().place(metres) != last else 0
        last = office_map().place(metres)
        change = t.observe(presence=True, distance_m=metres)
        if change is not None:
            committed.append(change)
    assert naive >= 19                       # what a memoryless model would log
    assert committed == []                   # what the dwell logs

    for _ in range(6):                       # he sits down in front of the desk
        c.tick(1.0)
        change = t.observe(presence=True, distance_m=1.2)
        if change is not None:
            committed.append(change)
    assert [ch.new for ch in committed] == ["just off the desk"]


def test_a_single_missed_poll_does_not_commit_no_opinion():
    # One timeout is a transient; roomsensor's own breaker needs three in a
    # row before it even calls the sensor down. The log must not gain a
    # "no opinion" line per dropped packet.
    c = Clock()
    t = _tracker(c)
    for _ in range(4):
        c.tick(2.0)
        t.observe(presence=True, distance_m=2.4)
    assert t.zone == "the middle of the room"
    c.tick(2.0)
    assert t.observe(presence=None, distance_m=None) is None
    c.tick(2.0)
    assert t.observe(presence=True, distance_m=2.4) is None   # back before dwell
    assert t.zone == "the middle of the room"


def test_the_dwell_applies_to_the_camera_rule_too():
    c = Clock()
    t = _tracker(c)
    seen = CameraOpinion(known=True, label="hunter")
    assert t.observe(presence=True, distance_m=2.0, camera=seen) is None
    c.tick(3.0)
    change = t.observe(presence=True, distance_m=2.0, camera=seen)
    assert change is not None and change.rule == RULE_CAMERA


def test_a_zero_dwell_commits_immediately_for_a_caller_that_wants_raw_edges():
    c = Clock()
    t = _tracker(c, dwell_s=0.0)
    change = t.observe(presence=True, distance_m=2.0)
    assert change is not None and change.new == "the middle of the room"


# --------------------------------------------------------------- the record
def _commit(tmp_path, **kw):
    c = Clock()
    log_file = ZoneLog(tmp_path / "state" / "zones.jsonl", **kw)
    t = ZoneTracker("office", office_map(), dwell_s=0.0, now=c.now,
                    wall=lambda: 1_756_000_000.0, log=log_file)
    return c, t, log_file


def test_a_committed_transition_is_one_json_line_naming_the_rule_that_decided(tmp_path):
    c, t, log_file = _commit(tmp_path)
    t.observe(presence=True, distance_m=2.4, moving=True, still=False,
              camera=CameraOpinion(known=False))
    lines = log_file.path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["room"] == "office"
    assert rec["old"] == NO_OPINION
    assert rec["new"] == "the middle of the room"
    assert rec["rule"] == RULE_BAND
    assert rec["distance_m"] == 2.4
    assert rec["presence"] is True and rec["moving"] is True and rec["still"] is False
    assert rec["camera"] == {"known": False, "label": ""}
    assert rec["at"] == 1_756_000_000.0 and rec["iso"].startswith("20")


def test_no_line_is_written_for_a_reading_that_changes_nothing(tmp_path):
    c, t, log_file = _commit(tmp_path)
    t.observe(presence=True, distance_m=2.4)
    t.observe(presence=True, distance_m=2.5)      # same band
    assert len(log_file.path.read_text().splitlines()) == 1


def test_the_record_carries_a_face_label_and_nothing_the_lens_saw(tmp_path):
    # HARD BOUNDARY: a name is fine, an image is not. The record is built
    # from a fixed set of fields, so there is nowhere for a frame, a crop or
    # an embedding to be smuggled in, and an over-long label is truncated.
    c, t, log_file = _commit(tmp_path)
    t.observe(presence=True, distance_m=1.0,
              camera=CameraOpinion(known=True, label="x" * 500))
    rec = json.loads(log_file.path.read_text().splitlines()[0])
    assert set(rec) == set(zn.RECORD_FIELDS)
    assert set(rec["camera"]) == {"known", "label"}
    assert len(rec["camera"]["label"]) == zn.MAX_LABEL_CHARS
    assert len(log_file.path.read_bytes()) < 400      # a line, not a picture


def test_the_transition_log_is_created_0600_inside_a_0700_directory(tmp_path):
    c, t, log_file = _commit(tmp_path)
    t.observe(presence=True, distance_m=1.0)
    assert stat.S_IMODE(log_file.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(log_file.path.parent.stat().st_mode) == 0o700


def test_the_log_rotates_at_its_cap_and_keeps_exactly_one_generation(tmp_path):
    c, t, log_file = _commit(tmp_path, max_bytes=600)
    metres = [1.0, 2.4, 3.6]
    for i in range(40):
        t.observe(presence=True, distance_m=metres[i % 3])
    rolled = log_file.path.with_name(log_file.path.name + ".1")
    assert rolled.exists()
    assert not log_file.path.with_name(log_file.path.name + ".2").exists()
    assert log_file.path.stat().st_size <= 600
    assert rolled.stat().st_size <= 600
    assert stat.S_IMODE(rolled.stat().st_mode) == 0o600


def test_the_cap_can_never_be_set_below_one_line_or_it_would_rotate_every_write(tmp_path):
    assert ZoneLog(tmp_path / "z.jsonl", max_bytes=10).max_bytes == zn.MAX_LINE_BYTES
    assert ZoneLog(tmp_path / "z.jsonl", max_bytes="nonsense").max_bytes == \
        zn.DEFAULT_MAX_BYTES


def test_a_log_that_cannot_be_written_costs_a_reading_and_not_the_poll(tmp_path):
    # A logging feature may not take the poll loop down with it.
    blocked = tmp_path / "nope" / "zones.jsonl"
    blocked.parent.mkdir()
    blocked.parent.chmod(0o500)
    try:
        log_file = ZoneLog(blocked)
        t = ZoneTracker("office", office_map(), dwell_s=0.0, log=log_file)
        change = t.observe(presence=True, distance_m=1.0)
        assert change is not None            # the tracker still answers
        assert log_file.writes == 0
        assert log_file.failures == 1
    finally:
        blocked.parent.chmod(0o700)


# --------------------------------------------------------------- the config
def test_the_office_map_in_the_code_and_the_one_in_the_config_cannot_drift():
    entry = DEFAULTS["zones"]["rooms"][0]
    assert entry["name"] == "office"
    assert entry["camera_zone"] == ZoneMap.office().camera_zone
    assert [(b["name"], b["near_m"], b["far_m"]) for b in entry["bands"]] == \
           [(b.name, b.near_m, b.far_m) for b in ZoneMap.office().bands]


def test_the_zone_maps_come_from_the_config():
    cfg = {"zones": {"enabled": True, "rooms": [
        {"name": "kitchen", "camera_zone": "at the counter",
         "bands": [{"name": "the doorway", "near_m": 0.75, "far_m": 2.25}]}]}}
    maps = zn.zone_maps(_Cfg(cfg))
    assert list(maps) == ["kitchen"]
    assert maps["kitchen"].place(1.0) == "the doorway"
    assert maps["kitchen"].camera_zone == "at the counter"


def test_a_room_whose_bands_overlap_is_skipped_and_the_others_survive():
    cfg = {"zones": {"enabled": True, "rooms": [
        {"name": "broken", "bands": [{"name": "a", "near_m": 0.75, "far_m": 3.0},
                                     {"name": "b", "near_m": 1.5, "far_m": 4.5}]},
        {"name": "office", "bands": [{"name": "a", "near_m": 0.75, "far_m": 4.5}]}]}}
    maps = zn.zone_maps(_Cfg(cfg))
    assert list(maps) == ["office"]


def test_zones_switched_off_in_the_config_builds_nothing():
    assert zn.zone_maps(_Cfg({"zones": {"enabled": False, "rooms": [
        {"name": "office", "bands": [{"name": "a", "near_m": 0.75, "far_m": 4.5}]}]}})) == {}


def test_the_dwell_and_the_cap_are_read_from_the_config():
    cfg = _Cfg({"zones": {"dwell_s": 7.5, "log_max_bytes": 4096}})
    assert zn.dwell_s(cfg) == 7.5
    assert zn.log_max_bytes(cfg) == 4096
    assert zn.dwell_s(_Cfg({"zones": {"dwell_s": "nonsense"}})) == zn.DEFAULT_DWELL_S


class _Cfg:
    """The dotted ``cfg.get`` shape every jarvis module reads.

    Faithful to ``AssistantConfig.get`` on the one point that matters
    here: a key that is PRESENT and null comes back as None, not as the
    default. A stub that folded the two together was why the first repair
    looked complete.
    """

    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


# -------------------------------------------------------------- the watcher
def _watcher(http, tmp_path, **kw):
    sensor = RoomSensor("http://10.0.0.9/binary_sensor/Presence", get=http)
    kw.setdefault("dwell_s", 0.0)
    return ZoneWatcher("office", sensor, office_map(),
                       log_file=ZoneLog(tmp_path / "zones.jsonl"), **kw)


def test_the_distance_is_read_at_the_entity_name_and_arrives_in_centimetres(tmp_path):
    http = Http({
        "http://10.0.0.9/binary_sensor/Presence": ESPHOME_ON,
        "http://10.0.0.9/sensor/Detection%20distance":
            '{"id":"sensor/Detection distance","value":242,"state":"242 cm"}',
        "http://10.0.0.9/sensor/Moving%20distance":
            '{"id":"sensor/Moving distance","value":242,"state":"242 cm"}',
        "http://10.0.0.9/sensor/Still%20distance":
            '{"id":"sensor/Still distance","value":0,"state":"0 cm"}'})
    w = _watcher(http, tmp_path)
    change = w.poll()
    assert change is not None
    assert change.new == "the middle of the room"      # 242 cm -> 2.42 m
    assert change.distance_m == pytest.approx(2.42)
    assert change.moving is True and change.still is False


def test_the_radar_is_not_asked_for_a_distance_while_it_says_the_room_is_empty(tmp_path):
    http = Http({"http://10.0.0.9/binary_sensor/Presence": ESPHOME_OFF})
    w = _watcher(http, tmp_path)
    change = w.poll()
    assert change.new == ABSENT
    assert http.urls == ["http://10.0.0.9/binary_sensor/Presence"]


def test_a_sensor_that_says_nothing_gives_no_opinion_and_costs_one_request(tmp_path):
    # And writes NO line: the tracker starts at "no opinion", so a radar
    # that never answers is not a transition, it is the status quo.
    http = Http(default=OSError("unreachable"))
    w = _watcher(http, tmp_path)
    assert w.poll() is None
    assert w.zone == NO_OPINION
    assert http.calls == 1
    assert w.log_file.writes == 0


def test_the_camera_hook_is_a_callable_the_watcher_never_opens_itself(tmp_path):
    # zones.py contains no camera code at all: an opinion is handed in.
    http = Http({"http://10.0.0.9/binary_sensor/Presence": ESPHOME_OFF})
    w = _watcher(http, tmp_path,
                 camera=lambda: CameraOpinion(known=True, label="hunter"))
    assert w.poll().new == "at the desk"


def test_a_camera_hook_that_raises_is_no_opinion_and_not_a_crash(tmp_path):
    def boom():
        raise RuntimeError("the gate is closed")

    http = Http({"http://10.0.0.9/binary_sensor/Presence": ESPHOME_OFF})
    w = _watcher(http, tmp_path, camera=boom)
    assert w.poll().new == ABSENT


def test_the_watcher_does_not_poll_the_radar_at_all_while_sensing_is_denied(tmp_path):
    # The offline-mode assertion is "no request was sent", not "the reading
    # was ignored" -- see jarvis/roomsensor.py. A distance read that went
    # around the policy would be a hole in exactly that promise.
    class Denied:
        def allowed(self, kind):
            return False

    http = Http(default=ESPHOME_ON)
    sensor = RoomSensor("http://10.0.0.9/binary_sensor/Presence", get=http,
                        policy=Denied())
    w = ZoneWatcher("office", sensor, office_map(), dwell_s=0.0,
                    log_file=ZoneLog(tmp_path / "zones.jsonl"))
    assert w.poll() is None
    assert w.zone == NO_OPINION
    assert sensor.reads == 0 and http.calls == 0


def test_a_distance_entity_that_404s_never_opens_the_presence_breaker(tmp_path):
    # The 09-03 bug was a silent 404 on a wrong URL shape. If a bad distance
    # entity could open the shared breaker it would take PRESENCE down with
    # it, which is the one thing the room sensor exists to provide.
    http = Http({"http://10.0.0.9/binary_sensor/Presence": ESPHOME_ON},
                default="<html>404</html>")
    sensor = RoomSensor("http://10.0.0.9/binary_sensor/Presence", get=http)
    w = ZoneWatcher("office", sensor, office_map(), dwell_s=0.0,
                    log_file=ZoneLog(tmp_path / "zones.jsonl"))
    changes = [c for c in (w.poll() for _ in range(8)) if c is not None]
    assert [c.new for c in changes] == [UNPLACED]   # present, nowhere placeable
    assert sensor.read() is True            # presence still answers
    assert not sensor.paused


def test_the_distance_reads_stop_after_the_entity_has_failed_enough_times(tmp_path):
    http = Http({"http://10.0.0.9/binary_sensor/Presence": ESPHOME_ON},
                default="<html>404</html>")
    sensor = RoomSensor("http://10.0.0.9/binary_sensor/Presence", get=http)
    w = ZoneWatcher("office", sensor, office_map(), dwell_s=0.0,
                    log_file=ZoneLog(tmp_path / "zones.jsonl"), read_bits=False)
    for _ in range(3):
        w.poll()
    before = http.calls
    w.poll()
    assert http.calls == before + 1          # presence only; the distance is skipped


def test_a_still_distance_stands_in_when_the_detection_entity_is_missing(tmp_path):
    http = Http({
        "http://10.0.0.9/binary_sensor/Presence": ESPHOME_ON,
        "http://10.0.0.9/sensor/Detection%20distance": "<html>404</html>",
        "http://10.0.0.9/sensor/Moving%20distance":
            '{"id":"sensor/Moving distance","value":0,"state":"0 cm"}',
        "http://10.0.0.9/sensor/Still%20distance":
            '{"id":"sensor/Still distance","value":110,"state":"110 cm"}'})
    w = _watcher(http, tmp_path)
    change = w.poll()
    assert change.new == "just off the desk"
    assert change.still is True and change.moving is False


def test_the_default_log_path_is_under_the_state_directory_not_the_tmpfs():
    # /tmp is wiped at every boot on this box; the record is meant to
    # outlive one, so it goes to ~/.local/state/jarvis (JARVIS_STATE_DIR in
    # the suite, see tests/conftest.py).
    assert str(zn.default_log_path()).endswith("/zones.jsonl")
    assert "/tmp/vss_voice" not in str(zn.default_log_path())
    assert str(zn.default_log_path()) == os.environ["JARVIS_STATE_DIR"] + "/zones.jsonl"


def test_an_eye_that_recognised_nobody_is_no_opinion_and_not_an_empty_chair():
    # jarvis/eye.py's identify() collapses "under the bar", "no gallery",
    # "the recogniser raised" and "score too low" into one ("", 0.0). It
    # says so on purpose, so this must not invent a distinction back.
    assert CameraOpinion.from_identify("", 0.0) is None
    assert CameraOpinion.from_identify("  ") is None
    seen = CameraOpinion.from_identify("hunter", 0.71)
    assert seen == CameraOpinion(known=True, label="hunter")


def test_the_built_in_office_ladder_stands_in_for_a_config_written_before_zones():
    assert zn.zone_map_for(_Cfg({}), "office").camera_zone == "at the desk"
    assert zn.zone_map_for(_Cfg({}), "kitchen") is None


def test_the_off_switch_is_not_walked_around_by_the_built_in_ladder():
    off = _Cfg({"zones": {"enabled": False}})
    assert zn.zone_map_for(off, "office") is None


def test_the_log_path_and_the_kept_generation_are_configurable(tmp_path):
    cfg = _Cfg({"zones": {"log_path": str(tmp_path / "elsewhere.jsonl"),
                          "log_keep": 0}})
    assert zn.log_path(cfg) == tmp_path / "elsewhere.jsonl"
    assert zn.log_keep(cfg) == 0
    assert zn.log_path(_Cfg({})) == zn.default_log_path()
    assert zn.log_keep(_Cfg({})) == 1


def test_keeping_no_generation_discards_the_old_file_instead_of_rolling_it(tmp_path):
    c, t, log_file = _commit(tmp_path, max_bytes=600, keep=0)
    for i in range(40):
        t.observe(presence=True, distance_m=[1.0, 2.4, 3.6][i % 3])
    assert not log_file.path.with_name(log_file.path.name + ".1").exists()
    assert log_file.path.stat().st_size <= 600


def test_the_ladder_describes_itself_with_its_gaps_named(tmp_path):
    text = office_map().describe()
    assert "at the desk" in text and "camera" in text
    assert text.count("gap") == 2               # the blind zone and past 4.5 m
    assert "0.75 -  1.50 m   just off the desk" in text


# --------------------------------------------------- the repairs (review 2)
# A verifier found eight problems with the first pass. These pin the two
# that mattered: a config error that silently kept recording against the
# OLD bands, and a still target the fallback never reached.

def test_a_room_whose_bands_are_broken_is_refused_not_quietly_replaced():
    # THE FAILURE THE WHOLE LOG EXISTS TO AVOID. He edits the office
    # ladder, mistypes it, and the record goes on being written against
    # the bands he thought he had replaced -- and looks like it worked.
    # A rejected room must record NOTHING.
    broken = _Cfg({"zones": {"rooms": [
        {"name": "office", "bands": [{"name": "mine", "near_m": 3.0,
                                      "far_m": 1.0}]}]}})
    assert zn.zone_maps(broken) == {}
    assert zn.zone_map_for(broken, "office") is None
    # Keyed by the dotted config path -- the line he has to go and edit --
    # and repeated under the room it costs.
    assert "zones.rooms[0].bands" in zn.rejected_rooms(broken)
    assert "office" in zn.read_zones(broken).room_refusals


def test_a_rejected_room_names_the_config_key_that_is_wrong():
    broken = _Cfg({"zones": {"rooms": [
        {"name": "kitchen", "bands": [{"name": "a", "near_m": 0.75}]}]}})
    # The missing key is far_m of the first band, and that exact path is
    # what comes back -- not "the kitchen is broken somewhere".
    assert "zones.rooms[0].bands[0].far_m" in zn.rejected_rooms(broken)
    why = zn.read_zones(broken).room_refusals["kitchen"]
    assert "zones.rooms[0].bands[0].far_m" in why and "missing" in why


def test_a_room_switched_off_in_its_own_entry_is_refused_not_replaced():
    off = _Cfg({"zones": {"rooms": [
        {"name": "office", "enabled": False,
         "bands": [{"name": "a", "near_m": 0.75, "far_m": 4.5}]}]}})
    assert zn.zone_map_for(off, "office") is None
    assert "zones.rooms[0].enabled" in zn.rejected_rooms(off)
    assert "is false" in zn.read_zones(off).room_refusals["office"]


def test_a_config_that_lists_rooms_at_all_gets_no_built_in_ladder():
    # The built-in office ladder is a bridge for a config written BEFORE
    # this section existed. Once zones.rooms is there, the config is the
    # only authority -- otherwise deleting the office entry silently
    # reinstates the shipped bands.
    listed = _Cfg({"zones": {"rooms": [
        {"name": "kitchen", "bands": [{"name": "a", "near_m": 0.75,
                                       "far_m": 4.5}]}]}})
    assert zn.zone_map_for(listed, "office") is None
    assert zn.zone_map_for(_Cfg({}), "office") is not None   # still bridged


def test_a_room_named_twice_is_refused_because_two_ladders_cannot_both_win():
    twice = _Cfg({"zones": {"rooms": [
        {"name": "office", "bands": [{"name": "a", "near_m": 0.75, "far_m": 4.5}]},
        {"name": "office", "bands": [{"name": "b", "near_m": 0.75, "far_m": 2.0}]}]}})
    assert zn.zone_map_for(twice, "office") is None
    why = zn.read_zones(twice).room_refusals["office"]
    # BOTH entries are named, because "one of these two" is not an
    # instruction he can act on.
    assert "zones.rooms[1]" in why and "zones.rooms[0]" in why
    assert "zones.rooms[1]" in zn.rejected_rooms(twice)


def _zero_http(detection, **rest):
    answers = {"http://10.0.0.9/binary_sensor/Presence": ESPHOME_ON,
               "http://10.0.0.9/sensor/Detection%20distance": detection}
    answers.update(rest)
    return Http(answers)


def test_a_detection_distance_of_zero_falls_through_to_the_still_distance(tmp_path):
    # 0 is a REAL reading meaning "no target of this kind" (parse_cm), not
    # a man standing on the module. The unverified still-only case is
    # exactly what the fallback was written for, and `if distance is None`
    # never reached it.
    http = _zero_http(
        '{"id":"sensor/Detection distance","value":0,"state":"0 cm"}',
        **{"http://10.0.0.9/sensor/Moving%20distance": '{"value":0}',
           "http://10.0.0.9/sensor/Still%20distance": '{"value":180}'})
    w = _watcher(http, tmp_path)
    change = w.poll()
    assert change.new == "the middle of the room"
    assert change.distance_m == pytest.approx(1.8)
    assert change.still is True and change.moving is False


def test_a_zero_from_every_distance_entity_is_no_distance_not_zero_metres(tmp_path):
    http = _zero_http('{"value":0}',
                      **{"http://10.0.0.9/sensor/Moving%20distance": '{"value":0}',
                         "http://10.0.0.9/sensor/Still%20distance": '{"value":0}'})
    w = _watcher(http, tmp_path)
    change = w.poll()
    assert change.new == UNPLACED
    assert change.distance_m is None        # never 0.0, which reads as a place


def test_a_detection_distance_of_zero_is_not_a_place_without_the_bits_either(tmp_path):
    http = _zero_http('{"value":0}')
    w = _watcher(http, tmp_path, read_bits=False)
    change = w.poll()
    assert change.new == UNPLACED
    assert change.distance_m is None


def test_a_healthy_distance_entity_does_not_reset_a_broken_one(tmp_path):
    # One counter shared by three entities meant a permanently 404ing
    # entity was re-asked on every single poll for ever.
    http = Http({"http://10.0.0.9/binary_sensor/Presence": ESPHOME_ON,
                 "http://10.0.0.9/sensor/Detection%20distance": '{"value":242}',
                 "http://10.0.0.9/sensor/Moving%20distance": '{"value":242}',
                 "http://10.0.0.9/sensor/Still%20distance": "<html>404</html>"})
    sensor = RoomSensor("http://10.0.0.9/binary_sensor/Presence", get=http)
    w = ZoneWatcher("office", sensor, office_map(), dwell_s=0.0,
                    log_file=ZoneLog(tmp_path / "zones.jsonl"))
    for _ in range(8):
        w.poll()
    asked = [u for u in http.urls if u.endswith("Still%20distance")]
    assert len(asked) == sensor.fail_after      # then it backs off
    assert w.zone == "the middle of the room"   # and the good ones carry on


def test_a_presence_value_that_is_not_a_bool_is_no_opinion_not_an_empty_chair():
    # RoomSensor.read() only ever answers True/False/None today. This is
    # the guard for the next reader: "" or 0 from a future source must not
    # become "not in the room", which is how Jarvis goes quiet on a man
    # sitting three feet away.
    for odd in (0, 0.0, "", [], {}, "unknown", "unavailable", 1, "ON"):
        v = verdict(office_map(), presence=odd, distance_m=2.0)
        assert (v.zone, v.rule) == (NO_OPINION, RULE_SILENT), odd
    assert verdict(office_map(), presence=False).zone == ABSENT
    assert verdict(office_map(), presence=True, distance_m=2.0).zone == \
        "the middle of the room"


def test_the_worst_line_over_the_office_vocabulary_is_measured_not_assumed():
    # The first pass quoted 242/272/330 bytes as a "ceiling"; they were
    # three particular transitions. This enumerates every old/new pair the
    # office can produce and pins the real maximum under the one-line
    # floor, so the (keep+1)*max_bytes ceiling holds.
    vocab = [b.name for b in office_map().bands] + \
            [UNPLACED, ABSENT, NO_OPINION, office_map().camera_zone]
    cams = [None, CameraOpinion(known=False),
            CameraOpinion(known=True, label="x" * zn.MAX_LABEL_CHARS)]
    worst = 0
    for old in vocab:
        for new in vocab:
            for cam in cams:
                t = zn.Transition(
                    room="office", old=old, new=new, rule=RULE_UNPLACED,
                    at=1_756_000_000.125, iso="2026-09-03T12:34:56",
                    held_s=1234.56, distance_m=4.44, presence=True,
                    moving=True, still=False, camera=cam)
                worst = max(worst, len(json.dumps(t.as_record()).encode()) + 1)
    assert worst <= zn.MAX_LINE_BYTES
    assert 1_000_000 // worst >= 2_850      # the claim in the docstring


def test_a_room_with_enormous_names_still_cannot_write_past_its_cap(tmp_path):
    # Room and band names are arbitrary config strings; MAX_LINE_BYTES is
    # only a real floor if a record cannot exceed it.
    cfg = _Cfg({"zones": {"rooms": [
        {"name": "k" * 300,
         "bands": [{"name": "b" * 300, "near_m": 0.75, "far_m": 2.0},
                   {"name": "c" * 300, "near_m": 2.0, "far_m": 4.5}]}]}})
    zmap = list(zn.zone_maps(cfg).values())[0]
    log_file = ZoneLog(tmp_path / "zones.jsonl", max_bytes=zn.MAX_LINE_BYTES)
    t = ZoneTracker(zmap.room, zmap, dwell_s=0.0, log=log_file)
    # known=False so the label lands in the record without the camera rule
    # taking over the zone: the worst line is a long room, two long band
    # names and a long label all at once.
    for i in range(20):
        t.observe(presence=True, distance_m=[1.0, 3.0][i % 2],
                  camera=CameraOpinion(known=False, label="c" * 300))
    rolled = log_file.path.with_name(log_file.path.name + ".1")
    assert log_file.writes == 20
    assert log_file.path.stat().st_size <= zn.MAX_LINE_BYTES
    assert not rolled.exists() or rolled.stat().st_size <= zn.MAX_LINE_BYTES


def _zone_log_script():
    """scripts/zone_log.py, imported by path (scripts/ is not a package)."""
    import importlib.util
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "scripts", "zone_log.py")
    spec = importlib.util.spec_from_file_location("zone_log_script", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["zone_log_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_instrument_is_governed_by_offline_mode_like_the_app(tmp_path):
    # scripts/zone_log.py is meant to be left running for an hour. It said
    # it polled "under the same offline-mode policy Jarvis itself uses" and
    # then built a RoomSensor with no policy at all, so the sensing switch
    # did not reach it. A SensingPolicy with no state file starts OFFLINE
    # (the fail-safe), so this asserts at the wire: nothing is sent.
    zl = _zone_log_script()
    sensor = zl.build_sensor(None, "http://10.0.0.9/binary_sensor/Presence",
                             policy_path=tmp_path / "sensing.json")
    assert sensor.blocked == "offline"
    assert sensor.read() is None
    assert sensor.read_distance() is None
    assert sensor.reads == 0


# --------------------------------------------------- the repairs (review 3)
# A second verifier proved the first repair incomplete: it told "no rooms
# key" from "a rooms LIST", not from "a rooms key of the wrong shape", so
# the commonest JSON slip in that file -- dropping the [ ] around the one
# room -- still landed on the built-in ladder, silently. These pin that,
# plus the two the repair itself introduced.

def _rooms(value):
    return _Cfg({"zones": {"rooms": value}})


ONE_ROOM = {"name": "office",
            "bands": [{"name": "my new near band", "near_m": 0.75,
                       "far_m": 2.0}]}


def test_a_rooms_key_of_the_wrong_shape_is_refused_and_never_bridged():
    # THE SLIP: he edits his one room and drops the square brackets. The
    # config TRIED to say something about rooms and got it wrong, which is
    # not the same as a config that stayed silent -- only silence may fall
    # back. Anything else records nothing, loudly.
    for shape in (ONE_ROOM,                       # the [ ] dropped
                  {"office": ONE_ROOM},           # an object map
                  "office",                       # a bare string
                  3,                              # a number
                  None):                          # an explicit null
        cfg = _rooms(shape)
        assert zn.zone_map_for(cfg, "office") is None, shape
        assert zn.zone_maps(cfg) == {}, shape
        why = zn.rejected_rooms(cfg)
        assert "zones.rooms" in why, shape
        assert "list" in why["zones.rooms"], shape


def test_the_wrong_shape_is_refused_through_the_real_config_too(tmp_path):
    # Not a stub: AssistantConfig._deep_merge replaces a list with the
    # user's dict, so the DEFAULTS ladder does not survive the slip -- and
    # zones.rooms coming out of DEFAULTS is exactly why the bridge must
    # not answer here.
    from jarvis.assistant_config import AssistantConfig
    path = tmp_path / "assistant.json"
    path.write_text(json.dumps({"zones": {"rooms": ONE_ROOM}}),
                    encoding="utf-8")
    cfg = AssistantConfig.load(path)
    assert isinstance(cfg.get("zones.rooms"), dict)      # the slip survived
    assert zn.zone_map_for(cfg, "office") is None
    assert "zones.rooms" in zn.rejected_rooms(cfg)


def test_only_a_missing_rooms_key_still_gets_the_built_in_ladder():
    # The one fallback left, and it has to keep working: a config written
    # before this section existed.
    assert zn.zone_map_for(_Cfg({}), "office") is not None
    assert zn.zone_map_for(_Cfg({"zones": {"enabled": True}}), "office") \
        is not None
    assert zn.rejected_rooms(_Cfg({})) == {}


def test_two_band_names_that_differ_only_past_the_cap_say_so():
    # The 64-character cap the last repair added can turn two DIFFERENT
    # names into one, and ZoneMap then refused the room for a duplicate he
    # never wrote -- sending him looking for a second entry that is not
    # there. Refusing is right; the reason has to be the true one.
    prefix = "the corner of the office behind the filing cabinet by the window"
    assert len(prefix) == zn.MAX_NAME_CHARS
    cfg = _rooms([{"name": "office", "bands": [
        {"name": prefix + " near", "near_m": 0.75, "far_m": 2.0},
        {"name": prefix + " far", "near_m": 2.0, "far_m": 4.5}]}])
    why = zn.read_zones(cfg).room_refusals["office"]
    assert zn.zone_map_for(cfg, "office") is None      # still refused
    assert "twice" not in why                          # but not as a duplicate
    # And it says what ACTUALLY happened. These names are ASCII, so the
    # character cap is the one that bit and both numbers are 64.
    assert "64 characters and 64 escaped bytes" in why


def test_two_room_names_that_differ_only_past_the_cap_say_so_too():
    # Same trap one level up: two rooms, one lookup key, and the old
    # message would have called them the same room.
    prefix = "the workshop at the far end of the garage behind the big freezer"
    assert len(prefix) == zn.MAX_NAME_CHARS
    band = [{"name": "a", "near_m": 0.75, "far_m": 4.5}]
    cfg = _rooms([{"name": prefix + " left wall", "bands": band},
                  {"name": prefix + " right wall", "bands": band}])
    why = list(zn.rejected_rooms(cfg).values())[0]
    assert zn.zone_map_for(cfg, prefix) is None
    assert "twice" not in why
    assert "64 characters and 64 escaped bytes" in why


def test_a_genuine_duplicate_room_name_still_reads_as_a_duplicate():
    cfg = _rooms([{"name": "office",
                   "bands": [{"name": "a", "near_m": 0.75, "far_m": 4.5}]},
                  {"name": "OFFICE",
                   "bands": [{"name": "b", "near_m": 0.75, "far_m": 2.0}]}])
    assert zn.zone_map_for(cfg, "office") is None
    why = zn.read_zones(cfg).room_refusals["office"]
    assert "again" in why and "zones.rooms[0]" in why


def test_a_genuine_duplicate_band_name_still_reads_as_a_duplicate():
    cfg = _rooms([{"name": "office", "bands": [
        {"name": "here", "near_m": 0.75, "far_m": 2.0},
        {"name": "here", "near_m": 2.0, "far_m": 4.5}]}])
    assert zn.zone_map_for(cfg, "office") is None
    assert "twice" in zn.read_zones(cfg).room_refusals["office"]


def test_a_huge_room_name_does_not_land_whole_in_the_complaint():
    # ZoneMap capped self.room only AFTER building its error strings, so a
    # 300-character room name reached the log line and rejected_rooms in
    # full. Every name that leaves this module is capped, errors included.
    with pytest.raises(ValueError) as caught:
        ZoneMap("R" * 300, (Band("a", 3.0, 1.0),))
    assert len(str(caught.value)) < 300
    assert "R" * (zn.MAX_NAME_CHARS + 1) not in str(caught.value)


def test_a_name_is_capped_in_BYTES_not_only_in_characters():
    # json.dumps escapes non-ASCII, so one CJK character costs 6 bytes on
    # the line and an emoji 12. A 64-CHARACTER cap is not a 64-byte cap,
    # and MAX_LINE_BYTES is a byte figure.
    for ch in ("居", "\U0001f600", "é", "x"):
        name = ch * 300
        assert len(json.dumps(zn._short(name))) - 2 <= zn.MAX_NAME_CHARS, ch


def test_a_room_named_in_cjk_or_emoji_cannot_write_past_its_cap(tmp_path):
    for ch in ("居", "\U0001f600"):
        # The names differ at the FRONT: two that differ only past the cap
        # are a clash, which the test above owns.
        cfg = _rooms([{"name": ch * 300, "bands": [
            {"name": "a" + ch * 300, "near_m": 0.75, "far_m": 2.0},
            {"name": "b" + ch * 300, "near_m": 2.0, "far_m": 4.5}]}])
        maps = zn.zone_maps(cfg)
        assert maps, ch          # capping must not collapse them into one
        zmap = list(maps.values())[0]
        path = tmp_path / ("z-%s.jsonl" % ord(ch))
        log_file = ZoneLog(path, max_bytes=zn.MAX_LINE_BYTES)
        t = ZoneTracker(zmap.room, zmap, dwell_s=0.0, log=log_file)
        for i in range(20):
            t.observe(presence=True, distance_m=[1.0, 3.0][i % 2],
                      camera=CameraOpinion(known=False, label=ch * 300))
        rolled = path.with_name(path.name + ".1")
        on_disk = path.stat().st_size + \
            (rolled.stat().st_size if rolled.exists() else 0)
        assert log_file.failures == 0, ch
        assert on_disk <= (log_file.keep + 1) * log_file.max_bytes, ch


def test_the_widest_record_the_fields_allow_is_computed_not_asserted():
    # The last repair called 499 bytes "the worst possible record" and it
    # was not: a wider float reaches 538. So compute the maximum over the
    # whole space instead of pinning a number a future field invalidates.
    # A float's repr is at most 24 characters, so the widest number here is
    # the widest number there is.
    # The names go in RAW: as_record is the last gate before the file and
    # caps them itself, so the size is a property of the format and not of
    # every caller's good behaviour.
    names = ["x" * 300, "居" * 300, "\U0001f600" * 300, '"\\' * 300,
             "é" * 300]
    numbers = [1.7976931348623157e+308, -1.7976931348623157e+308,
               -1234567890123.456, 99999999999.999, 1e16, -1e16,
               float("inf"), float("-inf"), float("nan"), -0.0]
    rules = [RULE_BAND, RULE_CAMERA, RULE_EMPTY, RULE_SILENT, RULE_UNPLACED]
    worst = 0
    for raw in names:
        for number in numbers:
            for rule in rules:
                t = zn.Transition(
                    room=raw, old=raw, new=raw, rule=rule, at=number,
                    iso="2026-09-03T12:34:56", held_s=number,
                    distance_m=number, presence=True, moving=True,
                    still=True,
                    camera=CameraOpinion(known=True, label=raw))
                worst = max(worst,
                            len(json.dumps(t.as_record()).encode()) + 1)
    assert worst <= zn.MAX_LINE_BYTES, worst
    # The numbers above are the widest a float can print: 24 characters.
    # iso is the one field this does not bound, and append refuses it.
    assert max(len(json.dumps(round(float(n), 3))) for n in numbers) == 24


def test_a_line_that_would_break_the_ceiling_is_refused_not_written(tmp_path):
    # The (keep+1)*max_bytes ceiling is only true if no single line can
    # exceed MAX_LINE_BYTES. Names are capped so it cannot happen through
    # the config; this is the backstop for a field that is not a name, and
    # it makes the bound ENFORCED rather than enumerated.
    path = tmp_path / "zones.jsonl"
    log_file = ZoneLog(path, max_bytes=zn.MAX_LINE_BYTES)
    fat = zn.Transition(room="office", old=UNPLACED, new="by the door",
                        rule=RULE_BAND, at=1.0, iso="i" * 4000, held_s=1.0)
    assert log_file.append(fat) is False
    assert log_file.writes == 0 and log_file.failures == 1
    assert not path.exists() or path.stat().st_size == 0


def test_the_instrument_refuses_a_rooms_key_of_the_wrong_shape(tmp_path,
                                                               capsys,
                                                               monkeypatch):
    # The point of the instrument is an hour of truthful record. Told to
    # record against a ladder it cannot read, it must stop with a non-zero
    # exit and the reason, not start with the shipped one.
    path = tmp_path / "assistant.json"
    path.write_text(json.dumps({"zones": {"rooms": ONE_ROOM}}),
                    encoding="utf-8")
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(path))
    zl = _zone_log_script()
    assert zl.main(["--describe"]) == 2
    err = capsys.readouterr().err
    assert "NOTHING will be recorded" in err
    assert zn.ROOMS_KEY in err


# --------------------------------------------------- the repairs (review 4)
# THREE rounds, three versions of ONE finding: a malformed band list fell
# back to the built-in ladder, then a zones.rooms of the wrong shape fell
# back, then a zones key of the wrong shape fell back -- each repair fixed
# the level it was shown and left the level above it open. And zones.enabled
# was read with bool(), so "false", "no", "off", "0" and null all left
# recording switched ON.
#
# So these do not test another instance. They test the CLASS: the declared
# shape of the whole section, crossed with every wrong shape there is. A
# level added to the config later and not to the declaration fails
# test_every_key_the_shipped_config_has_is_declared; a level in the
# declaration with no check behind it fails the cross product.

# One kind of wrong per column. None of them is ever the right shape for
# anything: the cross product below skips a value that satisfies the level
# it is aimed at, so "a number" is not tested against 7.
WRONG_SHAPES = {
    "a mapping": {"name": "office"},
    "a list": ["office"],
    "text": "office",
    "an int": 7,
    "a float": 1.5,
    "true or false": True,
    "null": None,
    "a nested wrong shape": [{"name": {"deep": ["wrong"]}}],
    "not a number": float("nan"),
}

GOOD_BAND = {"name": "mine near", "near_m": 0.75, "far_m": 1.5}
GOOD_ROOM = {"name": "office", "enabled": True, "camera_zone": "at the desk",
             "bands": [dict(GOOD_BAND),
                       {"name": "mine far", "near_m": 1.5, "far_m": 3.0}]}
GOOD_ZONES = {"enabled": True, "dwell_s": 3.0, "log_path": "",
              "log_max_bytes": 1000000, "log_keep": 1,
              "rooms": [copy.deepcopy(GOOD_ROOM)]}


def _levels():
    """Every level of the section, straight off the DECLARATION.

    Read from ``SECTION_SHAPE``/``ROOM_SHAPE``/``BAND_SHAPE`` rather than
    written out here, so a key added to the declaration is tested the
    moment it is declared and cannot be added without a check behind it.
    Each entry is (dotted path, declared shape, a setter that puts a value
    at that path in a copy of GOOD_ZONES).
    """
    def at_section(key):
        return lambda z, v: (z.__setitem__(key, v), z)[1]

    def at_room(key):
        return lambda z, v: (z["rooms"][0].__setitem__(key, v), z)[1]

    def at_band(key):
        return lambda z, v: (z["rooms"][0]["bands"][0].__setitem__(key, v), z)[1]

    out = [("zones", "a mapping", lambda z, v: v)]
    out += [("zones.%s" % k, want, at_section(k)) for k, want, _
            in zn.SECTION_SHAPE]
    out.append(("zones.rooms[0]", "a mapping",
                lambda z, v: (z["rooms"].__setitem__(0, v), z)[1]))
    out += [("zones.rooms[0].%s" % k, want, at_room(k)) for k, want, _
            in zn.ROOM_SHAPE]
    out.append(("zones.rooms[0].bands[0]", "a mapping",
                lambda z, v: (z["rooms"][0]["bands"].__setitem__(0, v), z)[1]))
    out += [("zones.rooms[0].bands[0].%s" % k, want, at_band(k)) for k, want, _
            in zn.BAND_SHAPE]
    return out


def _real_cfg(tmp_path, zones_section, n=[0]):
    """The REAL AssistantConfig off a real file, not a stub.

    The stub is where round two hid: _deep_merge replaces on a type
    mismatch, so a user's unwrapped dict beats the DEFAULTS list and the
    shipped ladder does NOT survive the slip.
    """
    from jarvis.assistant_config import AssistantConfig
    n[0] += 1
    path = tmp_path / ("assistant-%d.json" % n[0])
    path.write_text(json.dumps({"zones": zones_section}), encoding="utf-8")
    return AssistantConfig.load(path)


CROSS = [(path, want, kind, value)
         for path, want, _setter in _levels()
         for kind, value in WRONG_SHAPES.items()
         if not zn.SHAPES[want](value)]
SETTERS = {path: setter for path, _want, setter in _levels()}


def test_the_good_config_the_cross_product_starts_from_is_actually_good():
    # Without this the whole table below could pass by refusing everything.
    cfg = _Cfg({"zones": copy.deepcopy(GOOD_ZONES)})
    zmap = zn.zone_map_for(cfg, "office")
    assert zmap is not None
    assert [b.name for b in zmap.bands] == ["mine near", "mine far"]
    assert zn.rejected_rooms(cfg) == {}
    assert zn.dwell_s(cfg) == 3.0 and zn.log_keep(cfg) == 1


@pytest.mark.parametrize("path,want,kind,value", CROSS,
                         ids=["%s=%s" % (p, k) for p, _w, k, _v in CROSS])
def test_every_level_crossed_with_every_wrong_shape_is_refused_by_name(
        path, want, kind, value, tmp_path):
    # THE RULE, and it has no exempt level: a config that TRIES to say
    # something about zones and gets it wrong is refused BY NAME and
    # records nothing. Only an absent key falls back to a default.
    section = SETTERS[path](copy.deepcopy(GOOD_ZONES), copy.deepcopy(value))
    for cfg in (_Cfg({"zones": copy.deepcopy(section)}),
                _real_cfg(tmp_path, copy.deepcopy(section))):
        where = "%s %s at %s" % (kind, value, path)
        # 1. it records NOTHING, and in particular is not quietly handed
        #    the ladder built into jarvis/zones.py
        assert zn.zone_map_for(cfg, "office") is None, where
        assert zn.zone_maps(cfg) == {}, where
        # 2. it is refused BY NAME, under the exact dotted path he must go
        #    and edit -- not "something in zones is wrong"
        refused = zn.rejected_rooms(cfg)
        assert path in refused, (where, sorted(refused))
        assert refused[path].startswith(path), refused[path]
        assert want in refused[path], (where, refused[path])


# The one column the cross product above cannot carry: a container whose
# SHAPE is right and whose CONTENTS are wrong. A list of rooms really is a
# list, so the table skips it -- and "one level up" is the exact shape of
# the bug that came back three times, so it gets its own table.
# (container path, the value put there, the DEEPER path that must be named)
NESTED_WRONG = [
    ("zones", {"enabled": "false", "rooms": [copy.deepcopy(GOOD_ROOM)]},
     "zones.enabled"),
    ("zones.rooms", [{"name": {"deep": ["wrong"]},
                      "bands": [dict(GOOD_BAND)]}],
     "zones.rooms[0].name"),
    # a break in the SECOND room is named at index 1, and takes only the
    # kitchen with it -- one bad room is survivable, unlike a bad section
    ("zones.rooms", [copy.deepcopy(GOOD_ROOM),
                     {"name": "kitchen", "bands": 3}],
     "zones.rooms[1].bands"),
    ("zones.rooms[0]", {"name": "office", "bands": "not a list"},
     "zones.rooms[0].bands"),
    ("zones.rooms[0].bands", [{"name": 5, "near_m": 0.75, "far_m": 1.5}],
     "zones.rooms[0].bands[0].name"),
    ("zones.rooms[0].bands", [dict(GOOD_BAND),
                              {"name": "b", "near_m": None, "far_m": 3.0}],
     "zones.rooms[0].bands[1].near_m"),
    ("zones.rooms[0].bands[0]", {"name": "a", "near_m": "0.75", "far_m": 1.5},
     "zones.rooms[0].bands[0].near_m"),
]


@pytest.mark.parametrize("container,value,deeper", NESTED_WRONG,
                         ids=[d for _c, _v, d in NESTED_WRONG])
def test_a_container_of_the_right_shape_with_wrong_contents_is_refused_too(
        container, value, deeper, tmp_path):
    section = SETTERS[container](copy.deepcopy(GOOD_ZONES),
                                 copy.deepcopy(value))
    # A break under zones.rooms[1] costs the kitchen and NOT the office:
    # one bad room is survivable, a bad section is not.
    costs_office = not deeper.startswith("zones.rooms[1]")
    for cfg in (_Cfg({"zones": copy.deepcopy(section)}),
                _real_cfg(tmp_path, copy.deepcopy(section))):
        assert (zn.zone_map_for(cfg, "office") is None) is costs_office, deeper
        assert zn.zone_map_for(cfg, "kitchen") is None, deeper
        refused = zn.rejected_rooms(cfg)
        # named at the DEEPER path, not blamed on the container it sits in
        assert deeper in refused, (deeper, sorted(refused))
        assert container not in refused or container == deeper


def test_a_number_that_float_itself_cannot_hold_is_refused_not_raised():
    # My own first draft of the shape check called math.isfinite on the raw
    # value, and math.isfinite(10 ** 400) raises OverflowError -- so a
    # hand-edited config with an absurd integer in it took the exception
    # out through read_zones to the caller instead of being refused.
    for value in (10 ** 400, -10 ** 400, float("nan"), float("inf"),
                  float("-inf")):
        cfg = _Cfg({"zones": {"dwell_s": value, "rooms": [ONE_ROOM]}})
        assert zn.zone_map_for(cfg, "office") is None, repr(value)
        assert "zones.dwell_s" in zn.rejected_rooms(cfg), repr(value)
        assert zn.dwell_s(cfg) == zn.DEFAULT_DWELL_S, repr(value)
        band = _rooms([{"name": "office",
                        "bands": [{"name": "a", "near_m": value,
                                   "far_m": 4.5}]}])
        assert zn.zone_map_for(band, "office") is None, repr(value)
        assert "zones.rooms[0].bands[0].near_m" in zn.rejected_rooms(band)


def test_every_key_the_shipped_config_has_is_declared_in_the_shape():
    # The other half of the guarantee. The cross product covers every
    # DECLARED level; this is what makes a level added to the config and
    # NOT declared fail the suite instead of shipping unvalidated.
    zones = DEFAULTS["zones"]
    assert set(zones) == {k for k, _w, _d in zn.SECTION_SHAPE}
    for room in zones["rooms"]:
        assert set(room) == {k for k, _w, _d in zn.ROOM_SHAPE}
        for band in room["bands"]:
            assert set(band) == {k for k, _w, _d in zn.BAND_SHAPE}
    # and every shape a table names is one the walker can actually check
    for table in (zn.SECTION_SHAPE, zn.ROOM_SHAPE, zn.BAND_SHAPE):
        for key, want, _default in table:
            assert want in zn.SHAPES, (key, want)


def test_nothing_reads_a_zones_key_around_the_validator():
    # There must be no SECOND DOOR. Three rounds of this bug were three
    # different readers of the same section disagreeing about what counts
    # as an answer, so the fix is only a fix while there is one reader.
    import pathlib
    import re
    dotted = re.compile(r"""get(?:_option)?\(\s*["']zones""")
    root = pathlib.Path(zn.__file__).resolve().parents[1]
    # jarvis/ui/sensors_page.py joined the list on 2026-09-03: it is the
    # second consumer of this section and, unlike the log, it WRITES it.
    # A writer needs the rooms list as the file holds it, and it takes that
    # from ZonesConfig.raw_rooms rather than reading the key itself --
    # otherwise the page is a second door onto the same section, which is
    # the shape of every bug in this file's history.
    for name in ("jarvis/zones.py", "scripts/zone_log.py",
                 "jarvis/ui/sensors_page.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert not dotted.search(text), name     # no cfg.get("zones.x")
    source = (root / "jarvis/zones.py").read_text(encoding="utf-8")
    # exactly one raw read of the config, and it asks for the SECTION --
    # a dotted read answers "absent" for every key under a zones that is a
    # string, which is how the whole section slipped past round three.
    assert source.count("_cfg_raw(") == 2        # the def and its one call
    assert "_cfg_raw(cfg, SECTION_KEY)" in source


def test_an_enabled_flag_of_the_wrong_shape_does_not_leave_recording_on():
    # The verifier's own list: "false", "no", "off", "0" and null all left
    # recording enabled, because bool("false") is True and a present null
    # was folded into the default. An off switch that is only off when it
    # is spelled the one right way is not an off switch.
    for value in ("false", "no", "off", "0", "", 0, 1, None, [], {}):
        cfg = _Cfg({"zones": {"enabled": value, "rooms": [ONE_ROOM]}})
        assert zn.zone_map_for(cfg, "office") is None, repr(value)
        assert "zones.enabled" in zn.rejected_rooms(cfg), repr(value)
    # A real bool still works, both ways, and switching off is an ANSWER
    # rather than a mistake -- there is nothing to go and fix, so nothing
    # is named.
    on = _Cfg({"zones": {"enabled": True, "rooms": [ONE_ROOM]}})
    off = _Cfg({"zones": {"enabled": False, "rooms": [ONE_ROOM]}})
    assert zn.zone_map_for(on, "office") is not None
    assert zn.zone_map_for(off, "office") is None
    assert zn.rejected_rooms(off) == {}


def test_a_broken_key_above_the_rooms_costs_every_room_not_just_one():
    # A refusal above the room level has no entries to salvage and no
    # default it would be honest to use, so it is not survivable the way
    # one bad room is.
    two = [ONE_ROOM, {"name": "kitchen",
                      "bands": [{"name": "a", "near_m": 0.75, "far_m": 4.5}]}]
    fine = _Cfg({"zones": {"rooms": copy.deepcopy(two)}})
    assert sorted(zn.zone_maps(fine)) == ["kitchen", "office"]
    broken = _Cfg({"zones": {"dwell_s": "three seconds",
                             "rooms": copy.deepcopy(two)}})
    assert zn.zone_maps(broken) == {}
    assert zn.zone_map_for(broken, "kitchen") is None
    assert zn.zone_map_for(broken, "office") is None
    assert "zones.dwell_s" in zn.rejected_rooms(broken)
    # but one bad ROOM still only takes itself
    one_bad = _Cfg({"zones": {"rooms": [
        {"name": "office", "bands": "not a list"}, copy.deepcopy(two[1])]}})
    assert sorted(zn.zone_maps(one_bad)) == ["kitchen"]


def test_the_clash_message_counts_what_was_really_cut_not_always_sixty_four():
    # _short cuts at MAX_NAME_CHARS characters AND at MAX_NAME_CHARS
    # escaped bytes, and json.dumps escapes one CJK character to six bytes
    # -- so two CJK names become one after TEN characters, not sixty-four.
    # The message said "share their first 64 characters" whatever the
    # alphabet, which sent him to a column that does not exist.
    prefix = "居" * 12
    cfg = _rooms([{"name": "office", "bands": [
        {"name": prefix + "A", "near_m": 0.75, "far_m": 2.0},
        {"name": prefix + "B", "near_m": 2.0, "far_m": 4.5}]}])
    why = zn.read_zones(cfg).room_refusals["office"]
    assert zn.zone_map_for(cfg, "office") is None
    assert "10 characters and 60 escaped bytes" in why, why
    assert "64 characters" not in why, why       # the sentence that was false
    # and the number in the message is the truth about this string
    assert len(zn._short(prefix)) == 10
    assert len(json.dumps(zn._short(prefix))) - 2 == 60


def test_the_log_never_raises_at_the_caller_even_while_building_the_line(
        tmp_path):
    # ZoneLog.append built the JSON line OUTSIDE its try, so a Transition
    # carrying a field that is not a number raised ValueError at the
    # caller -- inside the poll loop -- against this class's documented
    # contract that it never does.
    path = tmp_path / "z.jsonl"
    log_file = ZoneLog(path)

    def t(**kw):
        base = dict(room="office", old=UNPLACED, new="by the door",
                    rule=RULE_BAND, at=1.0, iso="2026-09-03T00:00:00",
                    held_s=1.0)
        base.update(kw)
        return zn.Transition(**base)

    bad = [t(at="not a clock"),            # float() -> ValueError
           t(held_s=object()),             # float() -> TypeError
           t(distance_m="near"),           # float() -> ValueError
           t(presence={1, 2}),             # json.dumps -> TypeError
           object()]                       # not a Transition -> AttributeError
    for item in bad:
        assert log_file.append(item) is False
    assert log_file.writes == 0
    assert log_file.failures == len(bad)
    assert not path.exists() or path.stat().st_size == 0
    # and it is a refusal, not a poisoned log: a good record still writes
    assert log_file.append(t()) is True
    assert log_file.writes == 1


def test_the_ceiling_is_arithmetic_on_two_constants_not_an_average(tmp_path):
    # The only records-per-MB figure this module still states. Three passes
    # quoted a measured "typical" line size and a verifier failed to
    # reproduce it three times, so the measured claim is gone and what is
    # left divides one enforced constant by another.
    assert zn.DEFAULT_MAX_BYTES // zn.MAX_LINE_BYTES == 1562
    assert zn.ZoneLog(tmp_path / "z.jsonl", max_bytes=1).max_bytes == \
        zn.MAX_LINE_BYTES                        # no cap below one record
    # and no per-record byte figure survives in the prose, where it could
    # go stale without failing anything
    import pathlib
    for name in ("jarvis/zones.py", "jarvis/assistant_config.py"):
        text = (pathlib.Path(zn.__file__).resolve().parents[1]
                / name).read_text(encoding="utf-8")
        for stale in ("2,890", "2,850", "234-346", "234 to 346", "234-350",
                      "542 bytes", "499 bytes"):
            assert stale not in text, (name, stale)


def test_a_log_path_that_is_text_but_unusable_is_refused_by_name(tmp_path):
    # The class again, one level DOWN from shape this time. "~nobody/x" is
    # text, so the shape table passes it, and Path.expanduser then RAISES
    # RuntimeError -- which came out of read_zones at the caller instead of
    # being refused. A null byte is the same story through os.open, which
    # raises ValueError and so walked past ZoneLog's OSError handler too.
    import pathlib
    for value in ("~nosuchuser12345/zones.jsonl", "\0zones.jsonl",
                  "/var/\0/zones.jsonl"):
        for cfg in (_Cfg({"zones": {"log_path": value, "rooms": [ONE_ROOM]}}),
                    _real_cfg(tmp_path, {"log_path": value,
                                         "rooms": [copy.deepcopy(ONE_ROOM)]})):
            zones = zn.read_zones(cfg)          # and it does NOT raise
            assert "zones.log_path" in zones.refused, repr(value)
            assert zn.zone_map_for(cfg, "office") is None, repr(value)
            assert zn.zone_maps(cfg) == {}, repr(value)
            # and it falls back to nothing, not to the default file
            assert zones.poisoned, repr(value)
    # a path that is merely absent, or ordinary, still resolves
    plain = _Cfg({"zones": {"rooms": [ONE_ROOM]}})
    assert zn.log_path(plain) == zn.default_log_path()
    tilde = _Cfg({"zones": {"log_path": "~/zones.jsonl", "rooms": [ONE_ROOM]}})
    assert zn.log_path(tilde) == pathlib.Path("~/zones.jsonl").expanduser()


def test_the_log_never_raises_when_the_path_itself_cannot_be_opened():
    # ZoneLog says it never raises at the caller. os.stat and os.open raise
    # ValueError -- not OSError -- on a path with a null byte in it, and
    # that path can come straight out of the config, so the promise was
    # only true for the errors somebody had thought of.
    t = zn.Transition(room="office", old=UNPLACED, new="by the door",
                      rule=RULE_BAND, at=1.0, iso="2026-09-03T00:00:00",
                      held_s=1.0)
    for bad in ("\0zones.jsonl", "/var/\0/zones.jsonl"):
        log_file = ZoneLog(bad)
        assert log_file.append(t) is False, bad
        assert log_file.failures == 1 and log_file.writes == 0, bad


# ------------------------------- the repairs (review 5): the room with no lens
# THE SAME DISEASE, ONE LEVEL LOWER. Four rounds closed "a config value of the
# wrong SHAPE is silently replaced". This is a config value of the right shape
# that is silently replaced: ``camera_zone: ""`` -- which is how a room says
# there is no lens in it -- was `self.camera_zone or DEFAULT_CAMERA_ZONE`, so
# his lensless kitchen described itself as "at the desk", and "   " collapsed
# to "" and was written into the record as a NAMELESS zone while a band with
# no name was refused outright. Blank now MEANS something, and it means it
# everywhere.
def _room(name="kitchen", camera=None, bands=None):
    entry = {"name": name,
             "bands": bands or [{"name": "the kitchen",
                                 "near_m": 0.75, "far_m": 3.0},
                                {"name": "at the door",
                                 "near_m": 3.0, "far_m": 3.75}]}
    if camera is not None:
        entry["camera_zone"] = camera
    return entry


def test_a_blank_camera_zone_means_the_room_has_no_camera_at_all():
    zmap = ZoneMap("kitchen", (Band("the kitchen", 0.75, 3.0),), "")
    assert zmap.camera_zone == zn.NO_CAMERA == ""
    assert zmap.has_camera is False
    # ... and the camera rule cannot fire for it. A recognised face in a
    # room with no lens is somebody else's mistake, and the verdict falls
    # through to the radar rather than naming a place that does not exist.
    v = verdict(zmap, presence=True, distance_m=1.0,
                camera=CameraOpinion(known=True, label="hunterp"))
    assert (v.zone, v.rule) == ("the kitchen", RULE_BAND)


def test_a_camera_zone_of_only_spaces_is_no_camera_and_never_a_nameless_zone():
    # The verifier's own case: "   " is truthy, so it walked past the `or`,
    # collapsed to "" in _short, and the camera rule then wrote {"new": ""}
    # into the record -- a zone with no name, in a file whose whole job is
    # to say where he was.
    for blank in ("   ", "\t", "\n  \n", ""):
        zmap = ZoneMap("kitchen", (Band("the kitchen", 0.75, 3.0),), blank)
        assert zmap.has_camera is False, repr(blank)
        v = verdict(zmap, presence=False,
                    camera=CameraOpinion(known=True, label="hunterp"))
        assert v.zone == ABSENT and v.rule == RULE_EMPTY, repr(blank)


def test_an_absent_camera_zone_still_takes_the_default_because_absence_is_silence():
    # The one rule this module has never bent: a key that is not there at
    # all is silence, and silence may take a default.
    cfg = _Cfg({"zones": {"rooms": [_room("office", camera=None)]}})
    zmap = zn.zone_map_for(cfg, "office")
    assert zmap.camera_zone == zn.DEFAULT_CAMERA_ZONE
    assert zmap.has_camera is True


def test_his_kitchen_has_no_camera_and_says_so_instead_of_at_the_desk():
    # His config, 2026-09-03: the kitchen entry carries camera_zone "".
    cfg = _Cfg({"zones": {"rooms": [_room("kitchen", camera="")]}})
    zmap = zn.zone_map_for(cfg, "kitchen")
    assert zmap.has_camera is False
    text = zmap.describe()
    assert zn.DEFAULT_CAMERA_ZONE not in text
    assert "no camera" in text
    # and a room that HAS one still prints it against the word "camera"
    office = zn.zone_map_for(
        _Cfg({"zones": {"rooms": [_room("office", camera="at the desk")]}}),
        "office")
    assert "at the desk" in office.describe()
    assert "camera" in office.describe()


def test_a_room_with_no_camera_is_never_even_asked_for_an_opinion(tmp_path):
    # Cheaper, and one less way to touch the lens: the hook is not called
    # at all for a room the config says has no camera.
    calls = []
    http = Http(default=ESPHOME_OFF)
    sensor = RoomSensor("http://10.0.0.9/binary_sensor/Presence", get=http)
    zmap = ZoneMap("kitchen", (Band("the kitchen", 0.75, 3.0),), "")
    watcher = ZoneWatcher("kitchen", sensor, zmap, dwell_s=0.0,
                          log_file=ZoneLog(tmp_path / "z.jsonl"),
                          camera=lambda: calls.append(1) or
                          CameraOpinion(known=True, label="hunterp"))
    watcher.poll()
    assert calls == []
    assert watcher.zone == ABSENT


def test_no_configuration_of_the_camera_zone_can_write_a_zone_with_no_name():
    # THE CLASS, not the instance. Every rule, crossed with every way a
    # camera_zone can be blank and every camera opinion: a committed
    # transition's zone is always a NAMED place. A nameless zone in the
    # record is unreadable exactly when he goes looking for where he was.
    zmap_bands = (Band("the kitchen", 0.75, 3.0), Band("at the door", 3.0, 3.75))
    cameras = [None, CameraOpinion(known=True, label="hunterp"),
               CameraOpinion(known=False)]
    for camera_zone in ("", " ", "\t\n", "at the desk", "  at the desk  "):
        zmap = ZoneMap("kitchen", zmap_bands, camera_zone)
        for camera in cameras:
            for presence in (True, False, None):
                for distance in (None, 0.5, 1.0, 3.2, 9.9):
                    v = verdict(zmap, presence=presence, distance_m=distance,
                                camera=camera)
                    assert v.zone.strip(), (camera_zone, camera, presence,
                                            distance)
                    assert v.rule in (RULE_CAMERA, RULE_BAND, RULE_UNPLACED,
                                      RULE_EMPTY, RULE_SILENT)
                    if v.rule == RULE_CAMERA:
                        assert zmap.has_camera


def test_every_declared_text_key_says_what_a_blank_value_means():
    # The OTHER half, and the reason this is a class fix rather than one
    # more `or`. A text key that is PRESENT and blank tried to say
    # something; what it says has to be declared, not left to whichever
    # `or` happens to be on the path. A text key added to the shape
    # without an entry here fails this test instead of shipping with a
    # silent substitution behind it.
    declared = set(zn.BLANK_TEXT_MEANS)
    text_keys = set()
    for prefix, table in (("zones", zn.SECTION_SHAPE),
                          ("zones.rooms[]", zn.ROOM_SHAPE),
                          ("zones.rooms[].bands[]", zn.BAND_SHAPE)):
        for key, want, _default in table:
            if want == "text":
                text_keys.add("%s.%s" % (prefix, key))
    assert text_keys == declared, text_keys ^ declared
    # and each declaration is TRUE of the code, not just written down
    for blank in ("", "   "):
        # name: refused, and named by its dotted path
        cfg = _Cfg({"zones": {"rooms": [_room(blank)]}})
        assert zn.zone_map_for(cfg, "kitchen") is None
        assert any("name" in k for k in zn.rejected_rooms(cfg))
        # band name: refused too
        bad = _room("kitchen", bands=[{"name": blank, "near_m": 0.75,
                                       "far_m": 3.0}])
        cfg = _Cfg({"zones": {"rooms": [bad]}})
        assert zn.zone_map_for(cfg, "kitchen") is None
        assert zn.rejected_rooms(cfg)
        # camera_zone: no camera, and the room still records
        cfg = _Cfg({"zones": {"rooms": [_room("kitchen", camera=blank)]}})
        zmap = zn.zone_map_for(cfg, "kitchen")
        assert zmap is not None and zmap.has_camera is False
        # log_path: the default path, which is what "" has always meant
        cfg = _Cfg({"zones": {"log_path": blank, "rooms": [_room()]}})
        assert zn.log_path(cfg) == zn.default_log_path()
