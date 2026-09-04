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


def test_a_state_file_that_is_not_valid_utf8_starts_offline(tmp_path):
    """An interrupted write or a bad block leaves bytes that are not UTF-8.

    UnicodeDecodeError is a ValueError, so it escaped the `except OSError`
    around the read, escaped __init__, and was swallowed by app._construct
    -- leaving no policy at all, the radar polling ungoverned and the badge
    reading SENSING. A fail-ONLINE on exactly the corrupt input his ruling
    names, reached by the one path a fail-safe inside the object cannot
    cover: the object not existing.
    """
    (tmp_path / "sensing.json").write_bytes(b'\xff\xfe\x00\x01{"offline": true}')
    p = _policy(tmp_path)
    assert p.state().failsafe is True
    assert p.allowed(CAMERA) is False and p.allowed(RADAR) is False


def test_a_sensor_whose_owner_could_not_be_built_does_not_sense():
    """The belt for the same failure: app._construct hands back None, and a
    RoomSensor with policy=None falls through to "nobody is stopping me".
    The app hands it sensing.DENIED instead, which never says yes."""
    tr = _Transport()
    s = _sensor(sensing.DENIED, tr)
    assert s.read() is None and tr.calls == []
    assert s.blocked == "offline"
    dev = FakeCamera()
    assert CameraGate(sensing.DENIED, dev.open).open() is None
    assert dev.opens == 0
    assert sensing.DENIED.status()["offline"] is True


def test_the_app_gives_the_room_sensor_a_denying_policy_when_the_owner_is_gone():
    """Driven through app's own construction path, with no `sensing`
    attribute at all -- which is exactly what _construct leaves behind."""
    import types

    from jarvis.app import JarvisApp
    cfg = FakeCfg({"presence.room_sensor_enabled": True,
                   "presence.room_sensor_url": "http://10.0.0.9"})
    sentinel = JarvisApp._make_presence(types.SimpleNamespace(assistant=cfg))
    assert sentinel is not None and sentinel.sensor is not None
    assert sentinel.sensor.blocked == "offline"
    assert sentinel.sensor.read() is None


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


def test_set_curfew_reports_a_write_the_config_refused(tmp_path):
    """AssistantConfig.set / update return False rather than raising when
    the file was not written. Confirming a window that never reached the
    disk is a fail-ONLINE by the slow route: he widens the curfew by voice,
    is told it took, and the camera comes back at the old hour on the next
    restart."""
    class Refusing(FakeCfg):
        def set(self, dotted, value):
            self.writes.append((dotted, value))
            return False

    cfg = Refusing()
    p = _policy(tmp_path, cfg=cfg)
    assert p.set_curfew((22, 0), (6, 0)) is False
    assert p.set_curfew(None, None) is False, "the off branch reports too"


def test_set_curfew_writes_the_whole_window_in_one_save(tmp_path):
    """Three separate saves could be refused in the middle and leave the
    persisted start and end disagreeing, which is a privacy window nobody
    chose."""
    class Updating(FakeCfg):
        saves = 0

        def update(self, values):
            type(self).saves += 1
            self.data.update(values)
            self.writes.extend(values.items())
            return True

    Updating.saves = 0
    cfg = Updating()
    p = _policy(tmp_path, cfg=cfg)
    assert p.set_curfew((22, 0), (6, 30)) is True
    assert Updating.saves == 1
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


def test_disable_never_reports_that_sensing_is_allowed(tmp_path):
    """A method whose contract is "deny" must not be able to answer
    "allowed". The outcome used to be built from a state() call made
    OUTSIDE the lock, and state() expires a timed offline -- so a hold
    whose end passed during the call (disable can block for the ESPHome
    timeout) re-enabled everything while the line still said "offline
    until"."""
    now = _clock(12)
    p = _policy(tmp_path, now=lambda: now)
    p.enable()
    out = p.disable(until=now - 5)
    assert out.state.offline is True and out.state.camera is False
    assert out.state.until is None, "a bound already past is dropped, not honoured"
    assert p.allowed(CAMERA) is False and p.allowed(RADAR) is False


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


def test_a_device_that_reads_absent_is_still_stopped_on_the_way_off(tmp_path):
    """camera.device "0" is a cv2 INDEX, not a path, so present() read a lit
    camera as absent and the stopper was skipped: disable() said "nothing
    was sensing to stop" with the handle still open (F53, measured). Now the
    off path stops whatever this process may be holding regardless, and
    "absent" is only the wording."""
    p = _policy(tmp_path)
    p.enable()
    stopped = []
    p.attach("camera", lambda: stopped.append("camera") or True,
             present=lambda: False)
    out = p.disable()
    assert stopped == ["camera"]           # the stop ran anyway
    assert out.stopped == ()               # ...but is not CLAIMED
    assert out.absent == ("camera",)


def test_an_absent_device_is_not_resumed_on_the_way_back_on(tmp_path):
    """The symmetric case keeps its old rule: "back online" must not try to
    open a lens present() cannot see."""
    p = _policy(tmp_path)
    p.disable()
    resumed = []
    p.attach("camera", lambda: True, present=lambda: False,
             resume=lambda: resumed.append("camera") or True)
    p.enable()
    assert resumed == []


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


def test_the_spoken_switch_closes_a_camera_that_is_already_open(tmp_path):
    """"Offline mode" has to shut a lens that is already streaming, not
    merely refuse the next open()."""
    dev = FakeCamera()
    p = _policy(tmp_path)
    p.enable()
    gate = CameraGate(p, dev.open, closer=lambda d: d.close())
    gate.open()
    assert dev.open_now is True
    out = p.disable()
    assert dev.open_now is False and dev.closes == 1
    assert "camera" in out.stopped


# ------------------------------------------------- the curfew, on the clock
def test_the_curfew_closes_a_camera_that_is_already_open(tmp_path):
    """21:00 arrives with NOBODY having said anything.

    This is the control that runs unattended every night, and CameraGate
    only re-checks permission inside open(): a lens opened at 20:59 is
    still physically open at 21:00 unless something walks the devices on
    the clock. Meanwhile the badge has already flipped to CAM OFF --
    the exact lie the feature exists to prevent.
    """
    dev = FakeCamera()
    clock = {"t": _clock(20, 59)}
    p = _policy(tmp_path, now=lambda: clock["t"])
    p.enable()
    gate = CameraGate(p, dev.open, closer=lambda d: d.close())
    assert gate.open() is dev and dev.open_now is True
    clock["t"] = _clock(21, 0)
    assert p.allowed(CAMERA) is False       # the verdict has already turned
    assert gate.is_open is True             # ...and nothing has run yet
    out = p.enforce()
    assert dev.open_now is False and dev.closes == 1
    assert gate.is_open is False, "the lens was still open inside the curfew"
    assert out.stopped == ("camera",)


def test_the_curfew_edge_leaves_the_radar_running(tmp_path):
    """The curfew is about a LENS. The radar makes no image, so stopping it
    at night would cost presence for no privacy."""
    acted = []
    clock = {"t": _clock(20, 59)}
    p = _policy(tmp_path, now=lambda: clock["t"])
    p.enable()
    p.attach(RADAR, lambda: acted.append("stop") or True)
    clock["t"] = _clock(21, 0)
    p.enforce()
    assert acted == []


def test_the_guard_puts_the_camera_back_when_the_curfew_ends(tmp_path):
    """...and acts only on the CHANGE, so a device already down is not
    re-stopped every pass (which for a wired radar would be an HTTP POST
    every few seconds, all night)."""
    acted = []
    clock = {"t": _clock(22)}
    p = _policy(tmp_path, now=lambda: clock["t"])
    p.enable()
    p.attach(CAMERA, lambda: acted.append("stop") or True,
             resume=lambda: acted.append("resume") or True)
    p.enforce()
    assert acted == ["stop"]
    p.enforce()
    assert acted == ["stop"]
    clock["t"] = _clock(8)                  # morning, outside 21:00-07:00
    p.enforce()
    assert acted == ["stop", "resume"]


def test_a_stop_that_failed_is_retried_on_the_next_pass(tmp_path):
    """Privacy, not tidiness: a device that refused to stop keeps being
    asked, because the alternative is a lens left open until he speaks."""
    tries = {"n": 0}

    def flaky():
        tries["n"] += 1
        return tries["n"] > 2

    clock = {"t": _clock(22)}
    p = _policy(tmp_path, now=lambda: clock["t"])
    p.enable()
    p.attach(CAMERA, flaky)
    assert p.enforce().failed == ("camera",)
    assert p.enforce().failed == ("camera",)
    assert p.enforce().stopped == ("camera",)
    p.enforce()
    assert tries["n"] == 3, "a device that DID stop is not asked again"


def test_the_enforcement_runs_without_the_console(tmp_path):
    """The guard is the policy's own thread on purpose. A privacy control
    that stops working because the Tk window was never built, or its thread
    died, is not a privacy control."""
    import threading
    shut = threading.Event()
    clock = {"t": _clock(20, 59)}
    p = _policy(tmp_path, now=lambda: clock["t"])
    p.enable()
    p.attach(CAMERA, lambda: shut.set() or True)
    clock["t"] = _clock(21, 0)
    p.start(interval_s=0.01)
    try:
        assert shut.wait(5.0), "nothing closed the lens at the curfew edge"
    finally:
        p.stop()
    assert p.running is False


def test_coming_back_online_does_not_resume_what_the_curfew_still_denies(tmp_path):
    """"Back online" at ten at night must not open the lens: the nightly
    window is still running, and this is the one place it could be
    silently overridden."""
    acted = []
    p = _policy(tmp_path, now=lambda: _clock(22))
    p.disable()
    p.attach(CAMERA, lambda: True, resume=lambda: acted.append("camera") or True)
    p.attach(RADAR, lambda: True, resume=lambda: acted.append("radar") or True)
    out = p.enable()
    assert acted == ["radar"]
    assert out.resumed == ("radar",)


def test_a_camera_open_racing_a_spoken_offline_does_not_deadlock(tmp_path):
    """CameraGate takes its own lock and THEN asks the policy, so the policy
    must never run a device's stopper while holding its own lock -- the two
    orders meet and take out the vision thread and the voice thread
    together, permanently. The save inside the switch is a real window."""
    import threading
    import time as _time
    dev = FakeCamera()
    p = _policy(tmp_path)
    p.enable()
    gate = CameraGate(p, dev.open, closer=lambda d: d.close())
    gate.open()
    real_save = p._save
    p._save = lambda: (_time.sleep(0.3), real_save())[1]
    done = []
    voice = threading.Thread(target=lambda: (p.disable(), done.append("voice")),
                             daemon=True)
    vision = threading.Thread(target=lambda: (gate.open(), done.append("vision")),
                              daemon=True)
    voice.start()
    _time.sleep(0.05)                 # voice is inside the lock, mid-save
    vision.start()
    voice.join(10)
    vision.join(10)
    assert sorted(done) == ["vision", "voice"], "the two lock orders deadlocked"


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
    assert "radar" in out.partial       # no power switch wired: see below
    assert s.read() is None and tr.calls == []


def test_a_radar_with_no_power_switch_is_not_claimed_as_stopped(tmp_path):
    """THE LIVE CONFIGURATION. Nothing is flashed and no MOSFET is wired,
    so "offline" stops Jarvis asking while the LD2410 keeps radiating and
    keeps serving presence to anyone on the LAN. Bucketing that as
    `stopped` is what turns the spoken report into a promise -- and this is
    the only case that exists on his box today.
    """
    p = _policy(tmp_path)
    p.enable()
    tr = _Transport()
    s = _sensor(p, tr)
    assert s.power_url == ""
    out = p.disable()
    assert out.partial == ("radar",)
    assert out.stopped == () and out.failed == ()
    assert s.read() is None and tr.calls == [], "the POLLING really did stop"


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


def _radar_only_sentinel(tmp_path, policy, body):
    """A box whose ONLY presence leg is the radar -- the shape his is."""
    from jarvis.presence import PresenceSentinel
    cfg = FakeCfg({"presence.room_sensor_enabled": True,
                   "presence.room_sensor_url": "http://10.0.0.9",
                   "presence.phone_ip": "", "presence.phone_mac": ""})
    published, clock = [], {"t": 1_000_000.0}
    s = PresenceSentinel(cfg, publish=published.append,
                         now=lambda: clock["t"], poll_s=1.0, policy=policy)
    s.sensor._get = _Transport(body)
    return s, published, clock


def test_offline_stops_the_sentinel_asserting_away(tmp_path):
    """The frozen verdict, in the direction that MUTES him.

    The radar reports an empty room long enough to run the away grace out;
    then sensing goes off and the leg is gone. Holding "away" for the whole
    blackout makes jarvis/quiet.py answer "you're out" and swallow every
    proactive line while he is sitting in the room, and publishes
    presence: away to the ambient slab -- a state nobody can verify.
    """
    p = _policy(tmp_path)
    p.enable()
    s, published, clock = _radar_only_sentinel(tmp_path, p, '{"value": false}')
    for _ in range(4):
        clock["t"] += 600.0
        s.tick()
    assert s.state == "away" and s.is_home() is False
    assert len(published) == 1
    p.disable()
    clock["t"] += 600.0
    assert s.tick() is None
    assert s.state == "unknown", "a room nobody can see is not an empty one"
    assert s.is_home() is True, "an unknown room must not hold his speech"
    assert len(published) == 1, "no transition was invented on the way out"
    assert s.sensor.reads == 4, "and the radar was not polled while offline"


def test_offline_stops_the_sentinel_asserting_home(tmp_path):
    """The mirror, in the direction that talks to an empty room: a frozen
    "home" speaks proactive lines at nobody and then never says "welcome
    back", because there is no transition left to make."""
    p = _policy(tmp_path)
    p.enable()
    s, published, clock = _radar_only_sentinel(tmp_path, p, '{"value": true}')
    clock["t"] += 600.0
    s.tick()
    assert s.state == "home"
    p.disable()
    clock["t"] += 600.0
    assert s.tick() is None
    assert s.state == "unknown" and s.is_home() is True


def test_a_breaker_outage_still_HOLDS_rather_than_forgetting(tmp_path):
    """The distinction the fix turns on. A dead ESP32 is out for 30 s and
    comes back; holding the last verdict across that is right, and dropping
    to unknown on every transient would flap the ambient row."""
    p = _policy(tmp_path)
    p.enable()
    s, _published, clock = _radar_only_sentinel(tmp_path, p, '{"value": true}')
    clock["t"] += 600.0
    s.tick()
    assert s.state == "home"

    def dead(url, timeout):
        raise OSError("no route to host")

    s.sensor._get = dead
    for _ in range(6):                 # trips the breaker and keeps going
        clock["t"] += 600.0
        s.tick()
    assert s.state == "home", "a transient outage is not a privacy blackout"


def test_the_quiet_policy_and_the_desk_probe_are_left_alone():
    """Neither is governed, and the reasons are different. The desk probe
    reads GNOME's idle monitor -- his own keyboard, not a sensor pointed at
    the room -- and quiet hours only ever CONSUME presence, so gating them
    would turn a privacy switch into a silence switch he never asked for."""
    import pathlib
    root = pathlib.Path(sensing.__file__).parent
    for name in ("quiet.py", "deskpresence.py"):
        assert "sensing" not in (root / name).read_text(), name


# ------------------------- the same promise, on the MULTI-ROOM leg
#
# 2026-09-03 regression. `presence.rooms` swaps the sentinel's leg from one
# `RoomSensor` to `roomfabric.HouseView`, and the view shipped without a
# `blocked` property. `_blacked_out()` read it inside a broad `except` at
# debug level, so the AttributeError was swallowed, the check answered
# "nothing is blocked", and every test above stopped applying to the only
# configuration that has more than one radar in it. Measured on the branch
# before the fix: a rooms-only box held "home" through a full blackout
# where the single-sensor box reached "unknown".
def _rooms_only_sentinel(policy, body='{"value": true}'):
    """A box whose ONLY presence leg is a two-room fabric -- his, once the
    kitchen is flashed. No phone_ip, no phone_mac."""
    from jarvis.presence import PresenceSentinel
    cfg = FakeCfg({"presence.room_sensor_enabled": True,
                   "presence.rooms": [
                       {"name": "office", "url": "http://10.0.0.9",
                        "primary": True},
                       {"name": "kitchen", "url": "http://10.0.0.8"}],
                   "presence.phone_ip": "", "presence.phone_mac": ""})
    published, clock = [], {"t": 1_000_000.0}
    s = PresenceSentinel(cfg, publish=published.append,
                         now=lambda: clock["t"], poll_s=1.0, policy=policy)
    for room in s.fabric.rooms:
        room.sensor._get = _Transport(body)
    return s, published, clock


def test_the_house_view_can_say_it_is_blocked_at_all(tmp_path):
    """The attribute itself, pinned so it cannot go missing again. Presence
    reads `sensor.blocked` and nothing else asks the question, so a leg
    without it silently disables offline mode's reach into presence."""
    p = _policy(tmp_path)
    p.enable()
    s, _pub, _clock = _rooms_only_sentinel(p)
    view = s.sensor
    assert hasattr(type(view), "blocked"), "HouseView must wear `blocked`"
    assert view.blocked == ""
    p.disable()
    assert view.blocked == "offline"


def test_a_rooms_only_box_goes_unknown_when_sensing_is_off(tmp_path):
    """The end-to-end promise, on the leg that broke it. Two radars report
    the room occupied, then sensing goes off: the house must go to
    "unknown", not hold "home" and speak proactive lines into a room
    nobody can see."""
    p = _policy(tmp_path)
    p.enable()
    s, published, clock = _rooms_only_sentinel(p, '{"value": true}')
    clock["t"] += 600.0
    s.tick()
    assert s.state == "home"
    p.disable()
    clock["t"] += 600.0
    assert s.tick() is None
    assert s.state == "unknown", "a house nobody can see is not a home"
    assert s.is_home() is True
    assert len(published) == 1, "no transition was invented on the way out"
    assert [r.sensor.reads for r in s.fabric.rooms] == [1, 1], \
        "the radars were polled while offline"


def test_a_rooms_only_box_stops_asserting_AWAY_while_blacked_out(tmp_path):
    """The direction that MUTES him, which is why finding 1 mattered: a
    frozen "away" makes jarvis/quiet.py answer "you're out" and swallow
    every proactive line while he is sitting in the kitchen."""
    p = _policy(tmp_path)
    p.enable()
    s, published, clock = _rooms_only_sentinel(p, '{"value": false}')
    for _ in range(4):
        clock["t"] += 600.0
        s.tick()
    assert s.state == "away" and s.is_home() is False
    assert len(published) == 1
    p.disable()
    clock["t"] += 600.0
    assert s.tick() is None
    assert s.state == "unknown" and s.is_home() is True
    assert len(published) == 1


def test_a_breaker_outage_on_the_rooms_leg_still_HOLDS(tmp_path):
    """Same distinction as the single-sensor case: a dead ESP32 is a
    transient, not a privacy blackout, and dropping to unknown on every
    hiccup would flap the ambient row."""
    p = _policy(tmp_path)
    p.enable()
    s, _published, clock = _rooms_only_sentinel(p, '{"value": true}')
    clock["t"] += 600.0
    s.tick()
    assert s.state == "home"

    def dead(url, timeout):
        raise OSError("no route to host")

    for room in s.fabric.rooms:
        room.sensor._get = dead
    for _ in range(6):
        clock["t"] += 600.0
        s.tick()
    assert s.state == "home", "a transient outage is not a privacy blackout"


def test_a_leg_that_cannot_answer_blocked_is_reported_LOUDLY(caplog):
    """The swallow itself. `_blacked_out` used to read `.blocked` inside a
    bare `except Exception` logged at DEBUG, so a leg without the attribute
    turned offline mode off without a word. It still answers False -- a
    broken leg must not blank presence -- but at ERROR, naming the type."""
    import logging
    from jarvis.presence import PresenceSentinel

    class LegWithNoBlocked:
        configured = True

        def read(self):
            return None

    cfg = FakeCfg({"presence.phone_ip": "", "presence.phone_mac": ""})
    s = PresenceSentinel(cfg, publish=lambda ev: None)
    s.sensor = LegWithNoBlocked()
    with caplog.at_level(logging.ERROR):
        assert s._blacked_out() is False
    assert any("LegWithNoBlocked" in r.getMessage() and r.levelno >= logging.ERROR
               for r in caplog.records), "the missing attribute was swallowed"


def test_a_leg_whose_blocked_EXPLODES_does_not_cost_the_whole_TICK():
    """The other half of the same repair, and the one it narrowed.

    Making a MISSING `blocked` loud moved the read out of the `try`, so a
    leg whose `blocked` RAISES anything else -- an unreadable policy file,
    a property with a bug -- escaped `_blacked_out()`, escaped `tick()`
    (which has no guard of its own; only `_loop` does) and lost the entire
    poll, the blackout `_forget()` included. The old bare `except
    Exception` caught it. Loud about the case we can name, still caught
    for the ones we cannot.
    """
    from jarvis.presence import PresenceSentinel

    class ExplodingBlocked:
        configured = True

        @property
        def blocked(self):
            raise RuntimeError("the policy file is unreadable")

        def read(self):
            return None

    cfg = FakeCfg({"presence.phone_ip": "", "presence.phone_mac": ""})
    s = PresenceSentinel(cfg, publish=lambda ev: None)
    s.sensor = ExplodingBlocked()
    assert s._blacked_out() is False
    assert s.tick() is None          # and NOT a RuntimeError out of the tick


def test_the_missing_blocked_ERROR_is_said_once_not_once_a_POLL(caplog):
    """A leg with no `blocked` is wrong for as long as it is wired, and the
    sentinel polls every `poll_s` (60 s by default). One ERROR names the
    bug; one an hour, for ever, is how a real error gets filtered out."""
    import logging
    from jarvis.presence import PresenceSentinel

    class LegWithNoBlocked:
        configured = True

        def read(self):
            return None

    cfg = FakeCfg({"presence.phone_ip": "", "presence.phone_mac": ""})
    s = PresenceSentinel(cfg, publish=lambda ev: None)
    s.sensor = LegWithNoBlocked()
    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            assert s._blacked_out() is False
    said = [r for r in caplog.records
            if "LegWithNoBlocked" in r.getMessage() and r.levelno >= logging.ERROR]
    assert len(said) == 1, "one ERROR a poll, for as long as the leg is wrong"


# =====================================================================
# The privacy path is loud HOWEVER the leg fails to answer
# =====================================================================
# Third pass on the same three lines. It went silent once with the read
# inside a bare `except ... debug`; then a MISSING `blocked` was made loud
# and the OTHER branch of that same `if` -- a `blocked` that RAISES -- was
# left exactly as it was, caught at debug and swallowed. There is no branch
# left to forget now: one `try` asks the question and every way of failing
# to answer it is the same reported event.
class _NoBlocked:
    configured = True

    def read(self):
        return None


class _BlockedRaises:
    configured = True

    @property
    def blocked(self):
        raise RuntimeError("the policy file is unreadable")

    def read(self):
        return None


class _UnBoolable:
    """A value that is there and still cannot be read as a yes or a no."""

    class _Odd:
        def __bool__(self):
            raise ValueError("this cannot be read as a yes or a no")

    configured = True
    blocked = _Odd()

    def read(self):
        return None


def _dark_sentinel(leg):
    from jarvis.presence import PresenceSentinel
    cfg = FakeCfg({"presence.phone_ip": "", "presence.phone_mac": ""})
    s = PresenceSentinel(cfg, publish=lambda ev: None)
    s.sensor = leg
    return s


@pytest.mark.parametrize("leg", [_NoBlocked, _BlockedRaises, _UnBoolable])
def test_EVERY_way_of_failing_to_answer_blocked_is_reported_loudly(leg, caplog):
    """Absent, raising, or refusing bool(): one ERROR, naming the leg, and
    False rather than an invented blackout. A privacy path that fails
    silently is the worst failure in this file."""
    import logging
    s = _dark_sentinel(leg())
    with caplog.at_level(logging.ERROR):
        assert s._blacked_out() is False
    loud = [r for r in caplog.records
            if leg.__name__ in r.getMessage() and r.levelno >= logging.ERROR]
    assert len(loud) == 1, f"{leg.__name__} failed quietly"


@pytest.mark.parametrize("leg", [_NoBlocked, _BlockedRaises, _UnBoolable])
def test_no_way_of_failing_to_answer_blocked_costs_the_TICK(leg):
    """`tick()` has no guard of its own (only `_loop` does), so anything
    escaping here loses the whole poll -- the blackout's own `_forget()`
    included."""
    s = _dark_sentinel(leg())
    assert s.tick() is None


@pytest.mark.parametrize("leg", [_NoBlocked, _BlockedRaises, _UnBoolable])
def test_the_unreadable_blocked_ERROR_is_said_once_not_once_a_poll(leg, caplog):
    """`poll_s` is 60 s and the leg stays wrong for as long as it is wired:
    one line an hour for ever is how a real error gets filtered out."""
    import logging
    s = _dark_sentinel(leg())
    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            assert s._blacked_out() is False
    loud = [r for r in caplog.records
            if leg.__name__ in r.getMessage() and r.levelno >= logging.ERROR]
    assert len(loud) == 1


def test_a_leg_that_starts_failing_a_NEW_way_says_so_again(caplog):
    """Once per leg AND reason. A leg that was merely missing the attribute
    and then starts raising is a different fact about the privacy path, and
    the log has to carry it."""
    import logging
    leg = _NoBlocked()
    s = _dark_sentinel(leg)
    with caplog.at_level(logging.ERROR):
        s._blacked_out()
        s._blacked_out()
        type(leg).blocked = property(
            lambda self: (_ for _ in ()).throw(OSError("the socket died")))
        try:
            s._blacked_out()
            s._blacked_out()
        finally:
            del type(leg).blocked
    loud = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(loud) == 2, [r.getMessage() for r in loud]
    assert "AttributeError" in loud[0].getMessage()
    assert "OSError" in loud[1].getMessage()


# ------------------------------------------ a second reader of the same file
# F33 (thawed 2026-09-04). scripts/face_enrol.py builds its OWN SensingPolicy
# in another process; a spoken "offline mode" to the live Jarvis has to reach
# that policy's CameraGate mid-run, or the docstrings that say the script
# obeys "the same rules through the same objects" are false.


def test_a_spoken_offline_in_another_process_is_seen_by_a_second_reader(
        tmp_path):
    """Two policies on one sensing.json, at midday (outside the curfew).
    The live one is told "offline mode"; the script's one must answer False
    on its very next read, and True again when he comes back online.
    Reproduced 2026-09-03: the second reader stayed True forever."""
    live = _policy(tmp_path)
    live.enable(source="test")
    script = _policy(tmp_path)
    assert script.allowed(CAMERA) is True

    live.disable(source="voice")
    assert live.allowed(CAMERA) is False
    assert script.allowed(CAMERA) is False
    assert script.state().reason == sensing.REASON_OFFLINE

    live.enable(source="voice")
    assert script.allowed(CAMERA) is True


def test_the_second_readers_gate_shuts_on_the_next_open(tmp_path):
    """Through the gate, which is what the enrolment feed actually asks:
    an open device, then somebody else says offline, then the next open is
    None and the device was released."""
    live = _policy(tmp_path)
    live.enable(source="test")
    script = _policy(tmp_path)
    dev = FakeCamera()
    gate = CameraGate(script, dev.open, closer=lambda d: d.close())
    assert gate.open() is dev
    live.disable(source="voice")
    assert gate.open() is None
    assert dev.closes == 1 and gate.is_open is False


def test_a_timed_offline_from_another_process_carries_its_bound(tmp_path):
    live = _policy(tmp_path)
    live.enable(source="test")
    script = _policy(tmp_path)
    live.disable(until=_clock(14), source="voice")
    st = script.state()
    assert st.camera is False and st.reason == sensing.REASON_TIMED
    assert st.until == _clock(14)


def test_a_vanished_file_is_not_a_change(tmp_path):
    """Deleting sensing.json out from under a running owner must not flip
    it: the in-memory verdict stands and the next save puts the file back.
    (A MISSING file at construction is still offline -- that rule is
    unchanged and tested above.)"""
    p = _policy(tmp_path)
    p.enable(source="test")
    assert p.allowed(CAMERA) is True
    os.unlink(tmp_path / "sensing.json")
    assert p.allowed(CAMERA) is True
    p.disable(source="voice")
    assert (tmp_path / "sensing.json").exists()
    assert p.allowed(CAMERA) is False


def test_an_owner_does_not_reread_its_own_write_as_somebody_elses(tmp_path,
                                                                caplog):
    """The owner's own saves are recorded as seen, so its every ``state()``
    is not a reload and the log does not say another process changed the
    file on each of its own switches."""
    import logging
    p = _policy(tmp_path)
    with caplog.at_level(logging.INFO):
        p.enable(source="test")
        p.disable(source="voice")
        p.enable(source="voice")
        for _ in range(5):
            p.state()
    assert not [r for r in caplog.records
                if "another process" in r.getMessage()]


def test_a_corrupt_file_written_by_somebody_else_fails_shut(tmp_path):
    """The re-read goes through ``_load``, so its fail-safe applies: a
    second reader that finds garbage where the record was lands OFFLINE,
    the way a fresh start on that file would."""
    live = _policy(tmp_path)
    live.enable(source="test")
    script = _policy(tmp_path)
    assert script.allowed(CAMERA) is True
    (tmp_path / "sensing.json").write_text("{not json", encoding="utf-8")
    assert script.allowed(CAMERA) is False
    assert script.state().failsafe is True
