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
    """The dotted ``cfg.get`` shape every jarvis module reads."""

    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node


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
