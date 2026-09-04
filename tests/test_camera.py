"""The camera wired to the sensing owner -- tested with a device that
records whether it was ever opened.

The assertion that matters is never "the consumer ignored the frame". A
gate that merely discarded frames would still leave a lit camera light in
his room, so every test here counts OPENS on a fake device, exactly as
tests/test_sensing.py does for ``CameraGate`` itself.

What is new here, and why it needed a module rather than two lines of glue:

* ``Eye`` owns the two permission checks (before the open, and again after
  the grab) and ``CameraGate`` owns the device and the attachment to
  ``SensingPolicy.enforce``. Handing ``Eye`` the gate's raw device would
  leave the gate believing it still held one after ``Eye.close()``, so the
  device that crosses between them is WRAPPED: its ``release`` goes back
  through the gate, and its ``read`` refuses once the gate has let go.
* The 21:00 curfew edge arrives with nobody speaking. That is
  ``SensingPolicy.enforce()`` calling ``CameraGate.release``, and the test
  for it opens a device at 20:59 and asserts it is released at 21:01.
* The field of view is CONFIGURED, never assumed. ``lens_from_config``
  refuses to guess.

No cv2, no /dev/video*, no display.
"""
from __future__ import annotations

import datetime as _dt
import logging

import pytest

from jarvis import camera as cam
from jarvis.sensing import CAMERA, SensingPolicy


def _clock(hour: int, minute: int = 0, day: int = 2) -> float:
    return _dt.datetime(2026, 9, day, hour, minute).timestamp()


class FakeDevice:
    """cv2.VideoCapture's surface, and a record of what happened to it."""

    def __init__(self, fail_read: bool = False):
        self.reads = 0
        self.released = 0
        self.fail_read = fail_read

    def read(self):
        self.reads += 1
        if self.fail_read:
            return False, None
        return True, [[self.reads]]

    def release(self):
        self.released += 1


class Opener:
    """Counts the only thing that matters: how often a device was OPENED."""

    def __init__(self, fail: bool = False):
        self.opens = 0
        self.devices = []
        self.fail = fail

    def __call__(self):
        self.opens += 1
        if self.fail:
            raise OSError("no such device")
        dev = FakeDevice()
        self.devices.append(dev)
        return dev


class FakeCfg:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, dotted, default=None):
        value = self.data.get(dotted, default)
        return default if value is None else value


def _online_policy(tmp_path, now=None):
    path = tmp_path / "sensing.json"
    path.write_text('{"version": 1, "offline": false, "until": null}')
    return SensingPolicy(path=path, now=now or (lambda: _clock(12)))


def _feed(tmp_path, policy=None, opener=None, **kw):
    policy = policy or _online_policy(tmp_path)
    opener = opener or Opener()
    return cam.CameraFeed(policy, opener, **kw), policy, opener


# --------------------------------------------------- the device is not opened
def test_offline_never_opens_the_device(tmp_path):
    feed, policy, opener = _feed(tmp_path)
    policy.disable(source="test")
    assert feed.capture() is None
    assert feed.capture() is None
    assert opener.opens == 0
    assert feed.device_open is False


def test_a_missing_state_file_starts_offline_and_still_opens_nothing(tmp_path):
    """The fail-safe reaches the device. A first run has no state file, which
    ``SensingPolicy`` reads as OFFLINE, and the camera must inherit that
    rather than deciding for itself that nobody said no."""
    policy = SensingPolicy(path=tmp_path / "nope.json",
                           now=lambda: _clock(12))
    feed, _p, opener = _feed(tmp_path, policy=policy)
    assert feed.capture() is None
    assert opener.opens == 0


def test_the_curfew_never_opens_the_device(tmp_path):
    policy = _online_policy(tmp_path, now=lambda: _clock(22))
    feed, _p, opener = _feed(tmp_path, policy=policy)
    assert policy.state().camera is False
    assert feed.capture() is None
    assert opener.opens == 0


def test_online_and_outside_the_curfew_opens_once_and_reuses(tmp_path):
    feed, _p, opener = _feed(tmp_path)
    frames = [feed.capture() for _ in range(4)]
    assert all(f is not None for f in frames)
    assert opener.opens == 1
    assert opener.devices[0].reads == 4


def test_a_broken_policy_is_not_permission(tmp_path):
    class Exploding:
        def allowed(self, kind):
            raise RuntimeError("the state file caught fire")

        def attach(self, *a, **k):
            pass

    feed, _p, opener = _feed(tmp_path, policy=Exploding())
    assert feed.capture() is None
    assert opener.opens == 0


# ------------------------------------------- the curfew edge, nobody speaking
def test_the_curfew_edge_releases_a_device_already_open(tmp_path):
    """A lens opened at 20:59 is still a lit camera at 21:01 unless something
    walks the devices on the clock. That something is
    ``SensingPolicy.enforce`` calling the gate's release, and this is the
    whole reason the camera attaches to the policy instead of only asking it.
    """
    clock = {"t": _clock(20, 59)}
    policy = _online_policy(tmp_path, now=lambda: clock["t"])
    feed, _p, opener = _feed(tmp_path, policy=policy)
    assert feed.capture() is not None
    assert feed.device_open is True
    dev = opener.devices[0]

    clock["t"] = _clock(21, 1)
    out = policy.enforce()

    assert CAMERA in out.stopped
    assert dev.released == 1
    assert feed.gate.is_open is False


def test_after_the_curfew_edge_the_next_capture_opens_nothing(tmp_path):
    clock = {"t": _clock(20, 59)}
    policy = _online_policy(tmp_path, now=lambda: clock["t"])
    feed, _p, opener = _feed(tmp_path, policy=policy)
    feed.capture()
    clock["t"] = _clock(21, 1)
    policy.enforce()
    assert feed.capture() is None
    assert opener.opens == 1        # still just the one, from before the edge


def test_the_gate_and_the_eye_do_not_disagree_about_who_is_open(tmp_path):
    """``Eye`` releases the device on its own deny edge. If it released the
    RAW device the gate would still believe it held one, and the next
    permitted capture would hand back a released handle instead of opening a
    fresh one. The wrapper is what keeps the two in step."""
    policy = _online_policy(tmp_path)
    feed, _p, opener = _feed(tmp_path, policy=policy)
    assert feed.capture() is not None
    policy.disable(source="test")
    assert feed.capture() is None
    assert feed.gate.is_open is False
    assert opener.devices[0].released == 1
    policy.enable(source="test")
    assert feed.capture() is not None
    assert opener.opens == 2


def test_a_frame_in_flight_when_offline_is_set_is_dropped(tmp_path):
    """Offline said mid-grab must not recognise a face captured a
    millisecond earlier. ``Eye``'s second permission check is what does it;
    this asserts it survives the composition."""
    policy = _online_policy(tmp_path)
    feed, _p, opener = _feed(tmp_path, policy=policy)

    class Flipping(FakeDevice):
        def read(self):
            policy.disable(source="mid-grab")
            return FakeDevice.read(self)

    dev = Flipping()
    feed, _p, _o = _feed(tmp_path, policy=policy, opener=lambda: dev)
    assert feed.capture() is None
    assert feed.frames_dropped == 1
    assert dev.reads == 1
    assert dev.released == 1


def test_a_dead_read_closes_rather_than_latching(tmp_path):
    opener = Opener()
    feed, _p, _o = _feed(tmp_path, opener=opener)

    def bad():
        opener.opens += 1
        d = FakeDevice(fail_read=True)
        opener.devices.append(d)
        return d

    feed, _p, _o = _feed(tmp_path, opener=bad)
    assert feed.capture() is None
    assert feed.capture() is None
    assert opener.opens == 2          # re-opened rather than latching shut


def test_an_opener_that_raises_is_no_opinion_not_a_crash(tmp_path):
    feed, _p, opener = _feed(tmp_path, opener=Opener(fail=True))
    assert feed.capture() is None
    assert feed.status()["open"] is False


# ------------------------------------------------ absence, told honestly
def test_a_box_with_no_camera_is_reported_absent_not_stopped(tmp_path):
    """"The camera is off" is a lie when there was never a camera. The
    presence callable is the gate's own, and offline mode reports it."""
    policy = _online_policy(tmp_path)
    feed, _p, opener = _feed(tmp_path, policy=policy,
                             present=lambda: False)
    out = policy.disable(source="test")
    assert out.absent == (CAMERA,)
    assert out.stopped == ()
    assert opener.opens == 0
    assert feed is not None


def test_device_present_maps_a_digit_to_the_video_node(monkeypatch):
    """open_capture treats "0" as a cv2 index; present() must ask about
    /dev/video0, not about a file called "0" in the working directory
    (which made a lit camera read as absent -- F53)."""
    asked = []
    monkeypatch.setattr(cam.os.path, "exists", lambda p: asked.append(p) or True)
    assert cam.device_present("0") is True
    assert asked == ["/dev/video0"]


def test_device_present_reads_the_node_not_a_config_flag(tmp_path):
    node = tmp_path / "video7"
    assert cam.device_present(str(node)) is False
    node.write_text("")
    assert cam.device_present(str(node)) is True


# -------------------------------------------------- the lens is configured
def test_the_lens_comes_from_the_config_and_is_not_assumed(tmp_path):
    cfg = FakeCfg({"camera.width": 1280, "camera.height": 720,
                   "camera.hfov_deg": 65.64})
    lens = cam.lens_from_config(cfg)
    assert (lens.width_px, lens.height_px) == (1280, 720)
    assert lens.hfov_deg == pytest.approx(65.64, abs=0.01)
    assert lens.face_px(95.0) == pytest.approx(167, abs=1)


def test_a_missing_field_of_view_is_refused_rather_than_guessed():
    """The shipped comment reasoned from ~90 deg horizontal, which belongs to
    a camera he does not own; every real candidate is 65-82. A default here
    is how that error propagated in the first place."""
    with pytest.raises(ValueError) as exc:
        cam.lens_from_config(FakeCfg({"camera.width": 1280,
                                      "camera.height": 720}))
    assert "hfov_deg" in str(exc.value)


def test_an_absurd_field_of_view_is_refused():
    for bad in (0.0, -3.0, 180.0, 400.0):
        with pytest.raises(ValueError):
            cam.lens_from_config(FakeCfg({"camera.width": 1280,
                                          "camera.height": 720,
                                          "camera.hfov_deg": bad}))


def test_a_diagonal_field_of_view_is_converted_with_tangents(tmp_path):
    """Webcams are SOLD by the diagonal. Accepting one and converting it is
    better than inviting him to type 73 into a horizontal field."""
    cfg = FakeCfg({"camera.width": 1280, "camera.height": 720,
                   "camera.diag_fov_deg": 73.0})
    assert cam.lens_from_config(cfg).hfov_deg == pytest.approx(65.64, abs=0.02)


def test_the_thresholds_come_from_his_config(tmp_path):
    cfg = FakeCfg({"camera.min_conf": 0.62, "camera.cone_deg": 18.0,
                   "camera.cone_hysteresis_deg": 4.0, "camera.dwell_s": 0.7,
                   "camera.identity_min": 0.4, "camera.cone_centre_deg": 30.0})
    th = cam.thresholds_from_config(cfg)
    assert (th.min_conf, th.cone_deg, th.dwell_s) == (0.62, 18.0, 0.7)
    assert th.cone_centre_deg == 30.0


# ------------------------------------------------------------ the safe build
def test_build_never_raises_and_says_why_it_declined(tmp_path):
    cfg = FakeCfg({"camera.enabled": False})
    feed, why = cam.build(cfg, _online_policy(tmp_path))
    assert feed is None and "camera.enabled" in why


def test_build_declines_a_config_it_cannot_read_without_crashing(tmp_path):
    class Hostile:
        def get(self, *a, **k):
            raise RuntimeError("config on fire")

    feed, why = cam.build(Hostile(), _online_policy(tmp_path))
    assert feed is None and why


def test_build_wires_the_policy_when_the_config_is_complete(tmp_path):
    cfg = FakeCfg({"camera.enabled": True, "camera.width": 1280,
                   "camera.height": 720, "camera.hfov_deg": 65.64})
    opener = Opener()
    feed, why = cam.build(cfg, _online_policy(tmp_path), opener=opener)
    assert feed is not None and why == ""
    assert feed.lens.hfov_deg == pytest.approx(65.64, abs=0.01)
    assert feed.capture() is not None
    assert opener.opens == 1


# ------------------------------------------------- models, absent-safely
def test_no_weights_means_no_detector_and_a_reason_not_a_crash(tmp_path):
    """With nothing on disk the camera path must degrade to no opinion --
    today's behaviour exactly -- and say which file is missing. It must not
    raise, and it must not return something else that runs."""
    cfg = FakeCfg({"camera.model_dir": str(tmp_path / "nothing-here")})
    det, why = cam.detector_from_config(cfg)
    from jarvis import facemodels as fm
    assert det is None
    assert fm.backend_for().detector.filename in why
    assert "missing" in why


def test_the_model_directory_is_configurable(tmp_path):
    """The weights are 38 MB and live outside the repo. Which outside is his
    choice, and the config key is how he makes it."""
    from jarvis import facemodels as fm
    here = tmp_path / "weights"
    here.mkdir()
    for model in fm.ALL_MODELS:
        (here / model.filename).write_bytes(b"\0" * model.size)
    # camera.face_backend is the one-line reversal; this is also the test
    # that it reaches the loader.
    cfg = FakeCfg({"camera.model_dir": str(here), "camera.identity": True,
                   "camera.face_backend": "opencv"})
    calls = []
    import jarvis.facedetect as fdmod
    real = fdmod._create_yunet
    try:
        fdmod._create_yunet = lambda path, **kw: calls.append(path) or object()
        det, why = cam.detector_from_config(cfg)
    finally:
        fdmod._create_yunet = real
    assert det is not None and why == ""
    assert calls and calls[0].startswith(str(here))


def test_identity_off_means_the_recogniser_is_never_built(tmp_path):
    cfg = FakeCfg({"camera.identity": False})
    rec, why = cam.recogniser_from_config(cfg)
    assert rec is None and "camera.identity" in why


# --------------------------------------------- what the driver actually gave
class FakeCap:
    """A capture that GRANTS a different mode from the one asked for, which
    is what v4l2 does rather than erroring."""

    def __init__(self, granted=None, refuse=()):
        self.granted = dict(granted or {})
        self.refuse = set(refuse)
        self.sets = []

    def get(self, prop):
        return self.granted.get(prop, -1.0)

    def set(self, prop, value):
        self.sets.append((prop, value))
        if prop in self.refuse:
            return False
        self.granted[prop] = value
        return True


class _FakeVideoCapture:
    """cv2.VideoCapture's surface for open_capture: records every set, in
    order, and answers get() with what was set."""

    def __init__(self, target, opened=True):
        self.target = target
        self.sets = []
        self.props = {}
        self.released = 0
        self._opened = opened

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.sets.append((prop, float(value)))
        self.props[prop] = float(value)
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def release(self):
        self.released += 1


class _FakeCv2:
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT = 3, 4
    CAP_PROP_FPS, CAP_PROP_FOURCC = 5, 6
    CAP_PROP_CONVERT_RGB, CAP_PROP_BUFFERSIZE = 16, 38

    def __init__(self, opened=True):
        self.caps = []
        self._opened = opened

    @staticmethod
    def VideoWriter_fourcc(*chars):
        return sum(ord(c) << (8 * i) for i, c in enumerate(chars))

    def VideoCapture(self, target):
        cap = _FakeVideoCapture(target, self._opened)
        self.caps.append(cap)
        return cap


def test_open_capture_asks_for_one_buffer_and_logs_the_granted_mode(
        monkeypatch, caplog):
    """ONE buffer is requested, and the history of that number is on
    camera.CAPTURE_BUFFERS: set to 1 to keep a slow consumer off a frame
    up to three intervals old; measured in the PROBE to cost exactly half
    the rate (c228a01: 2.00x on eight rows of eight, so it was set to
    None); put back the same evening because the APP gained nothing from
    the driver's buffers -- 7.6 fps / 132 ms on its own rate line -- and
    he saw the staleness at once (9ba1c56). An earlier version of this
    test was named for c228a01's world and asserted 9ba1c56's.

    The log line says "requested", not "driver": cv2's read-back of
    CAP_PROP_BUFFERSIZE is OpenCV's own stored request, and the count the
    driver allocated is not observable through it (P12).

    The granted mode is still LOGGED, because the running app never read it
    back and a night was spent guessing that MJPG had been declined (it had
    not; the probe showed it granted in either set order, twice). FOURCC
    still goes before the size."""
    fake = _FakeCv2()
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    with caplog.at_level(logging.INFO, logger="jarvis.camera"):
        cap = cam.open_capture("", 1280, 720, "MJPG")
    props = [p for p, _ in cap.sets]
    assert props.index(fake.CAP_PROP_FOURCC) < props.index(
        fake.CAP_PROP_FRAME_WIDTH)
    assert cam.CAPTURE_BUFFERS == 1
    assert cap.props[fake.CAP_PROP_BUFFERSIZE] == 1.0
    assert cap.props[fake.CAP_PROP_FRAME_WIDTH] == 1280.0
    lines = [r.getMessage() for r in caplog.records
             if r.name == "jarvis.camera"]
    assert any("asked 1280x720 MJPG" in m and "granted 1280x720 MJPG" in m
               for m in lines), lines
    assert any("1 buffer(s) requested" in m for m in lines), lines
    assert not any("driver buffer" in m for m in lines), lines
    assert cap.released == 0


def test_open_capture_raises_when_the_device_will_not_open(monkeypatch):
    fake = _FakeCv2(opened=False)
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    with pytest.raises(OSError):
        cam.open_capture("/dev/video9")
    assert fake.caps[0].released == 1            # and lets go of the handle


def test_the_granted_mode_is_read_back_not_assumed():
    cv2 = pytest.importorskip("cv2")
    cap = FakeCap({cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                   cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
                   cv2.CAP_PROP_FPS: 30.0,
                   cv2.CAP_PROP_FOURCC: float(
                       cv2.VideoWriter_fourcc(*"MJPG"))})
    mode = cam.capture_mode(cap)
    assert (mode["width"], mode["height"]) == (1280.0, 720.0)
    assert mode["fourcc"] == "MJPG"
    assert mode["fps"] == 30.0


def test_an_unsupported_property_is_a_number_not_an_exception():
    pytest.importorskip("cv2")
    mode = cam.capture_mode(FakeCap())
    assert mode["width"] == -1.0 and mode["fourcc"] == ""


def test_focus_reports_whether_it_could_actually_be_pinned():
    cv2 = pytest.importorskip("cv2")
    cap = FakeCap({cv2.CAP_PROP_AUTOFOCUS: 1.0},
                  refuse={cv2.CAP_PROP_FOCUS})
    probe = cam.focus_probe(cap)
    assert probe["autofocus"]["pinned"] is True
    assert probe["focus"]["pinned"] is False


def test_fourcc_decodes_and_shrugs_at_nonsense():
    assert cam.fourcc_name(0) == ""
    assert cam.fourcc_name("nope") == ""
    cv2 = pytest.importorskip("cv2")
    assert cam.fourcc_name(cv2.VideoWriter_fourcc(*"YUYV")) == "YUYV"


# ------------------------------------------- the harness runs THROUGH the gate
def test_the_rig_source_obeys_offline_mode(tmp_path):
    """A rig pointed at a raw VideoCapture would be a second code path that
    never asks the sensing owner. It reads the feed instead."""
    from jarvis.visionrig import Rig, Thresholds
    from jarvis.facemodels import LIFECAM_CINEMA

    policy = _online_policy(tmp_path)
    feed, _p, opener = _feed(tmp_path, policy=policy)
    source = cam.FeedSource(feed)
    ok, frame = source.read()
    assert ok and frame is not None

    policy.disable(source="test")
    ok, frame = source.read()
    assert ok is False and frame is None

    class Det:
        name, input_size = "stub", (320, 180)

        def detect(self, frame):
            return None

    report = Rig(source, Det(), LIFECAM_CINEMA,
                 Thresholds(min_conf=0.6, cone_deg=20.0,
                            cone_hysteresis_deg=5.0, dwell_s=0.6,
                            identity_min=0.363)).run(frames=5)
    assert report.frames == 0            # offline: nothing was read
    assert opener.opens == 1


# ------------------------------------------- the curfew against a live grab
def test_a_close_waits_for_a_grab_already_in_flight(tmp_path):
    """``SensingPolicy.enforce`` runs on its own thread, and
    cv2.VideoCapture is not thread-safe. Releasing a capture while another
    thread is inside read() is undefined behaviour, and "the process
    crashed, which did close the camera" is not the enforcement anybody
    wants. So a close waits for the grab."""
    import threading

    clock = {"t": _clock(20, 59)}
    policy = _online_policy(tmp_path, now=lambda: clock["t"])
    in_read = threading.Event()
    may_finish = threading.Event()
    order = []

    class Slow(FakeDevice):
        def read(self):
            in_read.set()
            may_finish.wait(5.0)
            order.append("read done")
            return FakeDevice.read(self)

        def release(self):
            order.append("released")
            FakeDevice.release(self)

    dev = Slow()
    feed, _p, _o = _feed(tmp_path, policy=policy, opener=lambda: dev)
    grab = threading.Thread(target=feed.capture, daemon=True)
    grab.start()
    assert in_read.wait(5.0)

    clock["t"] = _clock(21, 1)
    closer = threading.Thread(target=policy.enforce, daemon=True)
    closer.start()
    may_finish.set()
    grab.join(5.0)
    closer.join(5.0)

    assert order == ["read done", "released"]
    assert dev.released == 1


def test_a_wedged_grab_does_not_postpone_the_curfew_forever(tmp_path,
                                                            monkeypatch):
    """The other half of the same decision: a camera that has stopped
    answering must not be able to hold the lens open indefinitely."""
    import threading

    monkeypatch.setattr(cam, "CLOSE_WAIT_S", 0.05)
    clock = {"t": _clock(20, 59)}
    policy = _online_policy(tmp_path, now=lambda: clock["t"])
    in_read = threading.Event()
    stuck = threading.Event()

    class Wedged(FakeDevice):
        def read(self):
            in_read.set()
            stuck.wait(10.0)
            return FakeDevice.read(self)

    dev = Wedged()
    feed, _p, _o = _feed(tmp_path, policy=policy, opener=lambda: dev)
    grab = threading.Thread(target=feed.capture, daemon=True)
    grab.start()
    assert in_read.wait(5.0)

    clock["t"] = _clock(21, 1)
    out = policy.enforce()                 # must return, not hang
    assert CAMERA in out.stopped
    assert dev.released == 1
    stuck.set()
    grab.join(5.0)


# ==========================================================================
# WHY THE CAMERA WOULD NOT OPEN, said truthfully (2026-09-04)
# ==========================================================================
# THE INCIDENT THIS FIXES. On 2026-09-03 scripts/face_enrol.py could not open
# the camera and reported "the frame source stopped delivering -- sensing
# denied the camera, or the device went away". BOTH HALVES WERE FALSE:
# sensing said camera=True and the device was present. It was simply HELD by
# the running Jarvis, because v4l2 capture is exclusive. The message was
# true-sounding and useless, and it cost half an hour of checking the two
# things that were already fine.
#
# So FeedSource now answers the question from what it can actually check --
# the gate, the node, whether the device ever opened, and who has it -- and
# these tests pin each branch. Nothing here opens a device or reads a frame;
# device_holder is a directory listing and a readlink, and opens nothing.
class _StatusFeed:
    """A CameraFeed-shaped stub: just the four things ``reason`` reads."""

    def __init__(self, allowed=True, opens=1, frames=0, device="",
                 policy_reason=""):
        self._status = {"allowed": allowed, "opens": opens, "frames": frames,
                        "name": "camera", "open": False}
        self.device = device
        self.policy = type("P", (), {
            "status": staticmethod(lambda: {"reason": policy_reason})})()

    def status(self):
        return dict(self._status)


def _readlink_or_blank(path: str) -> str:
    import os
    try:
        return os.readlink(path)
    except OSError:
        return ""


def test_a_denied_feed_says_sensing_and_names_the_reason():
    src = cam.FeedSource(_StatusFeed(allowed=False, policy_reason="curfew"))
    assert "sensing" in src.reason
    assert "curfew" in src.reason


def test_an_absent_device_says_so_rather_than_blaming_sensing(monkeypatch):
    monkeypatch.setattr(cam, "device_present", lambda _d="": False)
    src = cam.FeedSource(_StatusFeed(device="/dev/video9"))
    assert "not there" in src.reason
    assert "/dev/video9" in src.reason


def test_a_device_that_never_opened_is_reported_as_held_by_somebody(
        monkeypatch):
    """THE 2026-09-03 CASE. Sensing allows it, the node exists, and we never
    once got it open -- which on a v4l2 device means somebody else has it."""
    monkeypatch.setattr(cam, "device_present", lambda _d="": True)
    monkeypatch.setattr(cam, "device_holder",
                        lambda _d="": (1835163, "python3"))
    said = cam.FeedSource(_StatusFeed(opens=0, device="/dev/video0")).reason
    assert "already open" in said
    assert "1835163" in said and "python3" in said
    # ...and it does NOT repeat either of the two false claims.
    assert "sensing denied" not in said
    assert "went away" not in said


def test_an_unidentifiable_holder_is_still_reported_honestly(monkeypatch):
    """A process this user may not read must produce "somebody, and I cannot
    say who" -- never a guess, and never the old sentence."""
    monkeypatch.setattr(cam, "device_present", lambda _d="": True)
    monkeypatch.setattr(cam, "device_holder", lambda _d="": (0, ""))
    said = cam.FeedSource(_StatusFeed(opens=0, device="/dev/video0")).reason
    assert "another process is holding it" in said


def test_a_device_that_opened_and_then_stopped_says_that_instead(monkeypatch):
    monkeypatch.setattr(cam, "device_present", lambda _d="": True)
    said = cam.FeedSource(_StatusFeed(opens=1, frames=42)).reason
    assert "stopped answering" in said and "42" in said


def test_the_reason_never_raises_on_a_feed_that_cannot_answer():
    """It is read on a failure path, so it must not be able to add a second
    failure on top of the first."""
    class Broken:
        device = ""

        def status(self):
            raise RuntimeError("no")

    assert cam.FeedSource(Broken()).reason == ""


def test_device_holder_really_walks_proc_and_skips_our_own_pid(tmp_path):
    """Proves the /proc walk WORKS rather than merely returning (0, "") for
    everything, which is how this could pass while being broken.

    A plain temp file stands in for the video node: device_holder matches an
    fd's readlink target and has no opinion about what kind of file that is.
    Our own pid is skipped by design -- the question is "who has it INSTEAD
    of me" -- so both halves are asserted: nobody is reported, and the fd is
    nevertheless right there to be found at the path that was skipped.
    """
    import os
    probe = tmp_path / "video-probe"
    probe.write_text("x")
    fh = open(probe, "rb")
    try:
        assert cam.device_holder(str(probe)) == (0, "")
        mine = "/proc/%d/fd" % os.getpid()
        found = [f for f in os.listdir(mine)
                 if _readlink_or_blank("%s/%s" % (mine, f)) == str(probe)]
        assert found, "the /proc walk is looking in the wrong place"
    finally:
        fh.close()


def test_device_holder_says_nobody_for_a_node_that_does_not_exist():
    assert cam.device_holder("/dev/video-nope-99") == (0, "")


# ----------------------------------------- the exposure controls (2026-09-04)
# Everything below runs against _FakeCv2 / _FakeVideoCapture: a recorder of
# sets and gets, never a device. The real cv2 property numbers are used
# where a test needs them so the fake answers the same question the driver
# would; nothing opens /dev/video*.
_FakeCv2.CAP_PROP_GAIN, _FakeCv2.CAP_PROP_EXPOSURE = 14, 15
_FakeCv2.CAP_PROP_AUTO_EXPOSURE = 21


class _MeteredCapture(_FakeVideoCapture):
    """A capture that reports his LifeCam's controls as measured 2026-09-04:
    auto-exposure on (3), a cached exposure of 156, and NO gain control
    (get returns -1, the way v4l2 answers for a control the camera lacks).
    """

    def __init__(self, target, opened=True, auto=3.0, exposure=156.0,
                 refuse=()):
        super().__init__(target, opened)
        self.props[21] = float(auto)
        self.props[15] = float(exposure)
        self.props[14] = -1.0
        self.refuse = set(refuse)

    def set(self, prop, value):
        if prop in self.refuse:
            self.sets.append((prop, float(value)))
            return False
        return super().set(prop, value)


class _MeteredCv2(_FakeCv2):
    def __init__(self, opened=True, **kw):
        super().__init__(opened)
        self._kw = kw

    def VideoCapture(self, target):
        cap = _MeteredCapture(target, self._opened, **self._kw)
        self.caps.append(cap)
        return cap


def test_the_exposure_controls_are_read_as_numbers_and_an_absent_one_is_minus_one(
        monkeypatch):
    fake = _MeteredCv2()
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    cap = fake.VideoCapture(0)
    ctl = cam.exposure_probe(cap)
    assert ctl == {"auto_exposure": 3.0, "exposure": 156.0, "gain": -1.0}
    assert cap.sets == []                        # a probe changes nothing
    # a capture that raises on get is still a number, never an exception
    class Angry:
        def get(self, _prop):
            raise RuntimeError("no")
    assert cam.exposure_probe(Angry()) == {"auto_exposure": -1.0,
                                           "exposure": -1.0, "gain": -1.0}


def test_every_open_logs_the_exposure_controls_beside_the_granted_mode(
        monkeypatch, caplog):
    """The line that was missing on 2026-09-03: the preview halved from 7.5
    to 3.7 fps and nothing in the log could say whether the camera had been
    re-metered. Now every open says auto/manual and the three numbers, and
    says in the same breath that under auto the exposure figure is a cached
    value, so 156 is never again read as 'the room is bright'."""
    fake = _MeteredCv2()
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    with caplog.at_level(logging.INFO, logger="jarvis.camera"):
        cap = cam.open_capture("", 1280, 720, "MJPG")
    lines = [r.getMessage() for r in caplog.records if r.name == "jarvis.camera"]
    granted = [i for i, m in enumerate(lines) if "granted 1280x720" in m]
    controls = [i for i, m in enumerate(lines) if "controls at open" in m]
    assert granted and controls
    assert controls[0] == granted[0] + 1         # next to it, not somewhere
    line = lines[controls[0]]
    assert "auto_exposure 3  exposure 156  gain -1 (auto;" in line, line
    assert "cached manual value" in line
    # and by default NOTHING about exposure was written to the driver
    assert not any(p in (fake.CAP_PROP_AUTO_EXPOSURE, fake.CAP_PROP_EXPOSURE)
                   for p, _ in cap.sets)


def test_a_manual_control_is_logged_as_manual(monkeypatch, caplog):
    fake = _MeteredCv2(auto=1.0, exposure=20.0)
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    with caplog.at_level(logging.INFO, logger="jarvis.camera"):
        cam.open_capture("", 1280, 720, "MJPG")
    assert any("auto_exposure 1  exposure 20  gain -1 (manual;" in
               r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


def test_camera_exposure_pins_manual_exposure_at_open_and_reads_it_back(
        monkeypatch, caplog):
    """The lever, MEASURED 2026-09-04 grab-only with the app closed: auto
    had put his LifeCam on its 7.5 fps sensor tier (3.75 fps through the
    single driver buffer); manual 156 took the same open to 15-16 fps.
    Auto is switched off BEFORE the value is written -- v4l2 ignores an
    exposure_absolute written while exposure_auto is still 3 -- and the
    read-back is what the log prints, because a refused set is silent."""
    fake = _MeteredCv2()
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    with caplog.at_level(logging.INFO, logger="jarvis.camera"):
        cap = cam.open_capture("", 1280, 720, "MJPG", exposure=156)
    props = [p for p, _ in cap.sets]
    assert fake.CAP_PROP_AUTO_EXPOSURE in props and fake.CAP_PROP_EXPOSURE in props
    assert props.index(fake.CAP_PROP_AUTO_EXPOSURE) < props.index(
        fake.CAP_PROP_EXPOSURE)
    # and both AFTER the mode: the format negotiation comes first
    assert props.index(fake.CAP_PROP_FRAME_HEIGHT) < props.index(
        fake.CAP_PROP_AUTO_EXPOSURE)
    assert cap.props[fake.CAP_PROP_AUTO_EXPOSURE] == float(cam.EXPOSURE_MANUAL)
    assert cap.props[fake.CAP_PROP_EXPOSURE] == 156.0
    lines = [r.getMessage() for r in caplog.records if r.name == "jarvis.camera"]
    assert any("exposure pinned manual 156 -> accepted; driver reads back "
               "auto_exposure 1  exposure 156  gain -1" in m for m in lines), \
        lines
    assert not any("controls at open" in m for m in lines)   # one line, not two


def test_a_refused_exposure_pin_is_logged_as_refused_not_assumed(
        monkeypatch, caplog):
    fake = _MeteredCv2(refuse={15})               # the driver declines EXPOSURE
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    with caplog.at_level(logging.INFO, logger="jarvis.camera"):
        cam.open_capture("", 1280, 720, "MJPG", exposure=156)
    assert any("exposure pinned manual 156 -> REFUSED" in r.getMessage()
               for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


def test_pin_exposure_reports_the_read_back_and_never_raises():
    class Dead:
        def set(self, *_a):
            raise RuntimeError("gone")

        def get(self, *_a):
            raise RuntimeError("gone")
    import types
    fake = types.SimpleNamespace(CAP_PROP_AUTO_EXPOSURE=21,
                                 CAP_PROP_EXPOSURE=15, CAP_PROP_GAIN=14)
    import jarvis.camera as mod
    orig = mod._import_cv2
    mod._import_cv2 = lambda: fake
    try:
        out = cam.pin_exposure(Dead(), 156)
    finally:
        mod._import_cv2 = orig
    assert out["pinned"] is False and out["asked"] == 156
    assert out["auto_exposure"] == -1.0


def test_build_hands_camera_exposure_to_the_opener_and_zero_means_auto(
        tmp_path, monkeypatch):
    seen = []

    def fake_open(device, width, height, fourcc, exposure=0):
        seen.append((device, width, height, fourcc, exposure))
        return object()
    monkeypatch.setattr(cam, "open_capture", fake_open)
    base = {"camera.enabled": True, "camera.width": 1280,
            "camera.height": 720, "camera.hfov_deg": 65.6}
    feed, why = cam.build(FakeCfg(base), _online_policy(tmp_path))
    assert feed is not None, why
    feed._opener()
    feed2, why = cam.build(FakeCfg({**base, "camera.exposure": 156}),
                           _online_policy(tmp_path))
    assert feed2 is not None, why
    feed2._opener()
    assert [s[4] for s in seen] == [0, 156]
    assert all(s[3] == "MJPG" for s in seen)


def test_the_shipped_config_leaves_exposure_on_auto():
    """0 in the defaults: the lever exists, and it ships untouched. Whether
    the pane is still watchable at a fixed exposure in his evening light is
    his to read off the log, not this file's to decide."""
    from jarvis.assistant_config import DEFAULTS
    assert DEFAULTS["camera"]["exposure"] == 0
