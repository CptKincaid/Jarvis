"""Offline mode: the one owner of "am I allowed to sense right now".

Everything here is about a sensor that is LIVE while Hunter believes it is
off, so the assertions are deliberately about the DEVICE -- the fake camera
records whether it was ever opened, and RoomSensor's ``reads`` counter is
the proof that no HTTP request left the box -- and never about a consumer
politely ignoring a reading.

His ruling of 2026-09-02, in his words: "It also needs to have a Jarvis
offline mode and it will shutdown/disable sensors and cameras", with the
microphone deliberately LEFT LIVE so he can speak to bring it back.
"""
import json
import os

import pytest

from jarvis import sensing
from jarvis.sensing import (CAMERA, RADAR, CameraGate, SensingPolicy,
                            in_window, parse_hhmm)


# 2026-09-02 12:00:00 local -- a Wednesday midday, outside the 21:00-07:00
# curfew, so a test that says nothing about the clock is in the open.
def _clock(hour: int, minute: int = 0, day: int = 2):
    import datetime as _dt
    return _dt.datetime(2026, 9, day, hour, minute).timestamp()


def _policy(tmp_path, cfg=None, now=None, **kw):
    return SensingPolicy(cfg=cfg, path=tmp_path / "sensing.json",
                         now=now or (lambda: _clock(12)), **kw)


class FakeCfg:
    """The AssistantConfig surface sensing.py actually uses."""

    def __init__(self, data=None):
        self.data = dict(data or {})
        self.writes = []

    def get(self, dotted, default=None):
        value = self.data.get(dotted, default)
        return default if value is None else value

    def set(self, dotted, value):
        self.data[dotted] = value
        self.writes.append((dotted, value))
        return True


class FakeCamera:
    """A camera device that RECORDS whether it was ever opened.

    This is the whole point of the enforcement test: a gate that merely
    discards frames would still show opens here.
    """

    def __init__(self):
        self.opens = 0
        self.closes = 0
        self.open_now = False

    def open(self):
        self.opens += 1
        self.open_now = True
        return self

    def close(self):
        self.closes += 1
        self.open_now = False
        return True


# ----------------------------------------------------------- clock helpers
def test_parse_hhmm_takes_the_config_shape_and_rejects_junk():
    assert parse_hhmm("21:00") == (21, 0)
    assert parse_hhmm("07:30") == (7, 30)
    assert parse_hhmm("7:30") == (7, 30)
    for bad in ("", None, "nine", "25:00", "21:60", "21", "21:00:00", []):
        assert parse_hhmm(bad) is None, bad


def test_in_window_wraps_midnight():
    # The curfew he asked for is 21:00 -> 07:00, which is the wrapping case
    assert in_window((21, 0), (7, 0), (22, 30))
    assert in_window((21, 0), (7, 0), (3, 0))
    assert in_window((21, 0), (7, 0), (21, 0))      # inclusive at the start
    assert not in_window((21, 0), (7, 0), (7, 0))   # exclusive at the end
    assert not in_window((21, 0), (7, 0), (12, 0))
    # a same-day window still works, for a curfew he moves into the daytime
    assert in_window((9, 0), (17, 0), (12, 0))
    assert not in_window((9, 0), (17, 0), (8, 59))


# --------------------------------------------------------------- fail-safe
def test_a_missing_state_file_starts_offline():
    """FAIL TO OFFLINE. He chose this over persisting-and-trusting: with no
    readable state the honest answer is "I do not know", and not knowing
    must not open a lens."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = SensingPolicy(path=os.path.join(tmp, "nope", "sensing.json"),
                          now=lambda: _clock(12))
        st = p.state()
        assert st.failsafe is True
        assert st.camera is False and st.radar is False
        assert p.allowed(CAMERA) is False and p.allowed(RADAR) is False


@pytest.mark.parametrize("body", [
    "{ not json",
    "[]",                       # valid JSON, wrong shape
    '{"offline": "maybe"}',     # right shape, unusable value
    "",
    "null",
])
def test_a_corrupt_state_file_starts_offline(tmp_path, body):
    (tmp_path / "sensing.json").write_text(body)
    p = _policy(tmp_path)
    assert p.state().failsafe is True
    assert p.allowed(CAMERA) is False and p.allowed(RADAR) is False


def test_an_unreadable_state_file_starts_offline(tmp_path):
    path = tmp_path / "sensing.json"
    path.write_text(json.dumps({"offline": False}))
    path.chmod(0o000)
    try:
        p = _policy(tmp_path)
        assert p.allowed(CAMERA) is False
    finally:
        path.chmod(0o600)


def test_a_saved_offline_survives_the_restart(tmp_path):
    p = _policy(tmp_path)
    p.enable()                       # clear the first-run fail-safe
    assert p.allowed(CAMERA) is True
    p.disable()
    again = _policy(tmp_path)
    assert again.state().offline is True
    assert again.allowed(CAMERA) is False and again.allowed(RADAR) is False


def test_a_saved_online_survives_the_restart(tmp_path):
    _policy(tmp_path).enable()
    again = _policy(tmp_path)
    assert again.state().failsafe is False
    assert again.allowed(CAMERA) is True and again.allowed(RADAR) is True


def test_going_offline_reports_when_it_could_not_be_saved(tmp_path):
    """A write that failed means the NEXT start would read "online". The
    spoken line has to be able to say so, so the outcome carries it."""
    p = _policy(tmp_path)
    p.enable()
    (tmp_path / "sensing.json").unlink()
    tmp_path.chmod(0o500)            # no new file can be created here
    try:
        out = p.disable()
        assert out.persisted is False
        assert p.allowed(CAMERA) is False    # the running process still obeys
    finally:
        tmp_path.chmod(0o700)


# ----------------------------------------------------------------- curfew
def test_the_camera_curfew_closes_the_lens_and_leaves_the_radar(tmp_path):
    cfg = FakeCfg()
    at_ten_pm = _policy(tmp_path, cfg=cfg, now=lambda: _clock(22))
    at_ten_pm.enable()
    assert at_ten_pm.allowed(CAMERA) is False
    assert at_ten_pm.allowed(RADAR) is True, "the curfew is the LENS, not presence"
    assert at_ten_pm.state().reason == sensing.REASON_CURFEW


def test_the_curfew_wraps_past_midnight(tmp_path):
    p = _policy(tmp_path, now=lambda: _clock(3))
    p.enable()
    assert p.allowed(CAMERA) is False
    day = _policy(tmp_path, now=lambda: _clock(12))
    assert day.allowed(CAMERA) is True


def test_the_curfew_window_is_configurable(tmp_path):
    cfg = FakeCfg({"sensing.curfew.start": "13:00", "sensing.curfew.end": "14:00"})
    p = _policy(tmp_path, cfg=cfg, now=lambda: _clock(13, 30))
    p.enable()
    assert p.allowed(CAMERA) is False
    assert p.curfew() == ((13, 0), (14, 0))


def test_a_malformed_curfew_falls_back_to_the_shipped_window(tmp_path):
    """A typo in assistant.json must not silently delete the curfew."""
    cfg = FakeCfg({"sensing.curfew.start": "nine o'clock", "sensing.curfew.end": ""})
    p = _policy(tmp_path, cfg=cfg, now=lambda: _clock(22))
    p.enable()
    assert p.curfew() == (sensing.DEFAULT_CURFEW_START, sensing.DEFAULT_CURFEW_END)
    assert p.allowed(CAMERA) is False


def test_set_curfew_writes_both_ends(tmp_path):
    cfg = FakeCfg()
    p = _policy(tmp_path, cfg=cfg)
    assert p.set_curfew((22, 0), (6, 30)) is True
    assert cfg.data["sensing.curfew.start"] == "22:00"
    assert cfg.data["sensing.curfew.end"] == "06:30"
    assert p.curfew() == ((22, 0), (6, 30))


def test_the_manual_switch_outranks_the_curfew(tmp_path):
    p = _policy(tmp_path, now=lambda: _clock(12))
    p.disable()
    st = p.state()
    assert st.reason == sensing.REASON_OFFLINE      # not "curfew"
    assert st.camera is False and st.radar is False


# ------------------------------------------------------------ timed offline
def test_a_timed_offline_expires_on_its_own(tmp_path):
    clock = {"t": _clock(12)}
    p = _policy(tmp_path, now=lambda: clock["t"])
    p.enable()
    p.disable(until=clock["t"] + 7200)
    assert p.allowed(RADAR) is False
    assert p.state().reason == sensing.REASON_TIMED
    clock["t"] += 7201
    assert p.allowed(RADAR) is True, "an explicit 'for two hours' ends"


def test_a_timed_offline_that_expired_while_the_process_was_dead_comes_back(tmp_path):
    p = _policy(tmp_path, now=lambda: _clock(12))
    p.enable()
    p.disable(until=_clock(13))
    later = _policy(tmp_path, now=lambda: _clock(14))
    assert later.allowed(RADAR) is True


def test_an_open_ended_offline_does_not_expire(tmp_path):
    """He never said whether offline should auto-expire. The safer reading
    is that it does not: an unbounded 'go offline' means until he says
    otherwise, and a switch that healed itself overnight would put a lens
    back on without him ever asking for it."""
    p = _policy(tmp_path, now=lambda: _clock(12))
    p.enable()
    p.disable()
    tomorrow = _policy(tmp_path, now=lambda: _clock(12, day=9))
    assert tomorrow.allowed(CAMERA) is False


# --------------------------------------------------- the microphone is not here
def test_the_microphone_is_not_a_governed_sensor():
    """His explicit choice: the mic stays live so he can speak offline mode
    off again. It is not in SENSORS, and an unknown kind is DENIED, so no
    future caller can gate the mic through this object by accident."""
    assert set(sensing.SENSORS) == {CAMERA, RADAR}
    assert "mic" not in sensing.SENSORS and "microphone" not in sensing.SENSORS


def test_an_unknown_sensor_kind_is_denied(tmp_path):
    p = _policy(tmp_path)
    p.enable()
    assert p.allowed(CAMERA) is True
    assert p.allowed("lidar") is False
    assert p.allowed("") is False
    assert p.allowed(None) is False


def test_nothing_in_the_voice_path_consults_the_policy():
    """The mic, the wake word and the transcriber must not import sensing:
    a gate there would make offline mode unspeakable-out-of."""
    import pathlib
    root = pathlib.Path(sensing.__file__).parent
    for name in ("recorder.py", "hotword.py", "transcriber.py", "endpoint.py",
                 "speaker.py"):
        text = (root / name).read_text()
        assert "sensing" not in text, f"{name} must not gate the mic"


# ------------------------------------------------------------ device stops
def test_disable_stops_the_attached_devices_and_names_the_failures(tmp_path):
    p = _policy(tmp_path)
    p.enable()
    stopped = []
    p.attach("camera", lambda: stopped.append("camera") or True)
    p.attach("radar", lambda: False)          # a stop that did not work
    out = p.disable()
    assert stopped == ["camera"]
    assert out.stopped == ("camera",)
    assert out.failed == ("radar",)


def test_a_device_that_is_not_there_is_not_claimed_as_stopped(tmp_path):
    """The spoken line must not say "the camera is down" on a box with no
    camera wired: there is none, and claiming otherwise is the exact kind
    of promise he ruled out."""
    p = _policy(tmp_path)
    p.enable()
    p.attach("camera", lambda: True, present=lambda: False)
    out = p.disable()
    assert out.stopped == ()
    assert out.absent == ("camera",)


def test_a_stop_that_raises_is_a_failure_not_a_crash(tmp_path):
    p = _policy(tmp_path)
    p.enable()

    def boom():
        raise OSError("device busy")

    p.attach("radar", boom)
    out = p.disable()
    assert out.failed == ("radar",)
    assert p.allowed(RADAR) is False


# -------------------------------------------------------------- CameraGate
def test_the_camera_device_is_never_opened_while_offline(tmp_path):
    dev = FakeCamera()
    p = _policy(tmp_path)
    p.enable()
    gate = CameraGate(p, dev.open, closer=lambda d: d.close())
    assert gate.open() is dev and dev.opens == 1
    gate.release()
    p.disable()
    assert gate.open() is None
    assert dev.opens == 1, "the opener was CALLED while offline"


def test_the_camera_device_is_never_opened_during_the_curfew(tmp_path):
    dev = FakeCamera()
    p = _policy(tmp_path, now=lambda: _clock(23))
    p.enable()
    gate = CameraGate(p, dev.open, closer=lambda d: d.close())
    assert gate.open() is None
    assert dev.opens == 0


def test_going_offline_closes_a_camera_that_is_already_open(tmp_path):
    """The curfew arriving at 21:00 has to shut a lens that opened at
    20:59; a gate that only guards open() would leave it streaming."""
    dev = FakeCamera()
    p = _policy(tmp_path)
    p.enable()
    gate = CameraGate(p, dev.open, closer=lambda d: d.close())
    gate.open()
    assert dev.open_now is True
    out = p.disable()
    assert dev.open_now is False and dev.closes == 1
    assert "camera" in out.stopped


def test_a_camera_gate_refuses_when_the_policy_itself_is_broken(tmp_path):
    class Broken:
        def allowed(self, kind):
            raise RuntimeError("policy exploded")

    dev = FakeCamera()
    gate = CameraGate(Broken(), dev.open)
    assert gate.open() is None
    assert dev.opens == 0, "a broken policy must fail to OFFLINE"


def test_the_camera_gate_reports_a_close_that_failed(tmp_path):
    dev = FakeCamera()
    p = _policy(tmp_path)
    p.enable()

    def bad_close(_d):
        raise OSError("cannot release")

    gate = CameraGate(p, dev.open, closer=bad_close)
    gate.open()
    out = p.disable()
    assert out.failed == ("camera",)


# ------------------------------------------------------------------ status
def test_status_says_which_sensor_is_off_and_why(tmp_path):
    p = _policy(tmp_path, now=lambda: _clock(22))
    p.enable()
    st = p.status()
    assert st["camera"] is False and st["radar"] is True
    assert st["reason"] == sensing.REASON_CURFEW
    assert st["curfew"] == "21:00-07:00"


# ------------------------------------------------------ the radar, at the wire
class _Transport:
    """Records every URL asked for, so "no request left the box" is an
    assertion and not a hope."""

    def __init__(self, body='{"value": true}'):
        self.calls = []
        self.body = body

    def __call__(self, url, timeout):
        self.calls.append(url)
        return self.body


def _sensor(policy=None, transport=None, url="http://10.0.0.9/binary_sensor/presence"):
    from jarvis import roomsensor
    return roomsensor.RoomSensor(url, get=transport or _Transport(),
                                 policy=policy)


def test_the_radar_issues_no_request_while_offline(tmp_path):
    p = _policy(tmp_path)
    p.enable()
    tr = _Transport()
    s = _sensor(p, tr)
    assert s.read() is True and tr.calls
    p.disable()
    tr.calls.clear()
    assert s.read() is None
    assert tr.calls == [], "the radar was polled while offline"
    assert s.paused is True
    assert s.status()["blocked"] == "offline"


def test_the_radar_stays_up_through_the_camera_curfew(tmp_path):
    """The curfew is about a LENS. Killing presence at night as well would
    cost the arrival cue and the away grace for no privacy gain -- the
    radar produces no image."""
    p = _policy(tmp_path, now=lambda: _clock(23))
    p.enable()
    tr = _Transport()
    assert _sensor(p, tr).read() is True
    assert len(tr.calls) == 1


def test_a_broken_policy_stops_the_radar(tmp_path):
    class Broken:
        def allowed(self, kind):
            raise RuntimeError("boom")

    tr = _Transport()
    assert _sensor(Broken(), tr).read() is None
    assert tr.calls == []


def test_the_radar_registers_its_own_stop_with_the_policy(tmp_path):
    p = _policy(tmp_path)
    p.enable()
    tr = _Transport()
    s = _sensor(p, tr)
    out = p.disable()
    assert "radar" in out.stopped
    assert s.read() is None and tr.calls == []


def test_an_unconfigured_radar_is_absent_not_stopped(tmp_path):
    p = _policy(tmp_path)
    p.enable()
    _sensor(p, _Transport(), url="")           # no URL -> nothing to stop
    out = p.disable()
    assert out.absent == ("radar",) and out.stopped == ()


def test_the_radar_powers_the_device_down_when_it_has_a_switch(tmp_path):
    """ESPHome exposes a switch as POST /switch/<name>/turn_off. With one
    configured, "offline" cuts the radar itself instead of merely not
    listening to it."""
    from jarvis import roomsensor
    posts = []
    p = _policy(tmp_path)
    p.enable()
    s = roomsensor.RoomSensor("http://10.0.0.9/binary_sensor/presence",
                              get=_Transport(), policy=p,
                              power_url="http://10.0.0.9/switch/radar_power",
                              post=lambda url, timeout: posts.append(url))
    out = p.disable()
    assert posts == ["http://10.0.0.9/switch/radar_power/turn_off"]
    assert "radar" in out.stopped
    p.enable()
    s.resume()
    assert posts[-1] == "http://10.0.0.9/switch/radar_power/turn_on"


def test_a_power_cut_that_failed_is_reported_not_swallowed(tmp_path):
    from jarvis import roomsensor

    def boom(url, timeout):
        raise OSError("no route to host")

    p = _policy(tmp_path)
    p.enable()
    roomsensor.RoomSensor("http://10.0.0.9/binary_sensor/presence",
                          get=_Transport(), policy=p,
                          power_url="http://10.0.0.9/switch/radar_power",
                          post=boom)
    out = p.disable()
    assert out.failed == ("radar",), "he must be told the radar may still be on"


# ------------------------------------------- what presence does while offline
def test_presence_holds_rather_than_calling_the_room_empty(tmp_path):
    """The honest degradation. A blocked radar has NO OPINION, which is
    already the module's dark-safe path: with no phone leg the tick returns
    None and the sentinel holds its state, instead of drifting to "away"
    and muting him for the evening."""
    from jarvis.presence import RoomOrPhone
    p = _policy(tmp_path)
    p.enable()
    tr = _Transport()
    s = _sensor(p, tr)
    p.disable()
    assert RoomOrPhone(s)("", "") is None
    assert tr.calls == []
    # ...and with a phone configured the phone decides, exactly as it did
    # before the sensor existed.
    assert RoomOrPhone(s, lambda ip, mac: True)("10.0.0.5", "") is True


# ------------------------------------------- what the consumers do, exactly
def test_the_sentinel_holds_unknown_instead_of_drifting_away(tmp_path):
    """The end-to-end degradation, on the shape that matters: a box whose
    ONLY presence leg is the radar. While offline the leg has no opinion,
    so the sentinel must hold "unknown" -- not run out the away grace and
    publish Presence(home=False), which is what mutes every proactive line
    he has and dims the board he is sitting in front of."""
    from jarvis.presence import PresenceSentinel
    cfg = FakeCfg({"presence.room_sensor_enabled": True,
                   "presence.room_sensor_url": "http://10.0.0.9",
                   "presence.phone_ip": "", "presence.phone_mac": ""})
    p = _policy(tmp_path, cfg=cfg)
    p.disable()
    published = []
    clock = {"t": 1_000_000.0}
    s = PresenceSentinel(cfg, publish=published.append,
                         now=lambda: clock["t"], poll_s=1.0, policy=p)
    assert s.sensor is not None and s.configured
    for _ in range(20):
        clock["t"] += 600.0            # ten minutes a tick: two hours in all
        assert s.tick() is None
    assert published == [], "offline mode published a presence transition"
    assert s.state == "unknown"
    assert s.is_home() is True, "an unknown room is not an empty one"
    assert s.sensor.reads == 0, "the radar was polled while offline"


def test_the_quiet_policy_and_the_desk_probe_are_left_alone():
    """Neither is governed, and the reasons are different. The desk probe
    reads GNOME's idle monitor -- his own keyboard, not a sensor pointed at
    the room -- and quiet hours only ever CONSUME presence, so gating them
    would turn a privacy switch into a silence switch he never asked for."""
    import pathlib
    root = pathlib.Path(sensing.__file__).parent
    for name in ("quiet.py", "deskpresence.py"):
        assert "sensing" not in (root / name).read_text(), name
