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


def test_open_capture_leaves_the_buffer_queue_alone_and_drains_it_instead(
        monkeypatch, caplog):
    """THE BUFFERS COME BACK, because the drain is what makes that safe.

    The history of that number is on camera.CAPTURE_BUFFERS. It was set to 1
    to keep a slow consumer off a frame up to three intervals old; measured
    in the PROBE to cost exactly half the rate (c228a01: a clean 2.00x on
    eight rows of eight); put back to 1 the same evening because the APP
    gained nothing from the driver's buffers and he saw the staleness at
    once (9ba1c56). What 9ba1c56 restored was a NAKED buffer count with no
    drain under it, and that is the thing this replaces -- not the staleness
    verdict, which stands. ``DrainingCapture`` keeps the driver's four
    buffers AND discards the stale ones before retrieving, so the two go
    together: the property is never written, and the handle that comes back
    wraps the VideoCapture rather than being it.

    The log line therefore says "driver buffer(s)" rather than "requested":
    nothing is requested any more, so the number printed is only ever the
    read-back. The device underneath is still asked for exactly the same
    mode in exactly the same order, and the granted mode is still LOGGED,
    because the running app never read it back and a night was spent
    guessing that MJPG had been declined (it had not; the probe showed it
    granted in either set order, twice). FOURCC still goes before the
    size."""
    fake = _FakeCv2()
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    with caplog.at_level(logging.INFO, logger="jarvis.camera"):
        cap = cam.open_capture("", 1280, 720, "MJPG")
    raw = cap.raw
    props = [p for p, _ in raw.sets]
    assert props.index(fake.CAP_PROP_FOURCC) < props.index(
        fake.CAP_PROP_FRAME_WIDTH)
    assert cam.CAPTURE_BUFFERS is None
    # the whole point: the property is never written, at all
    assert fake.CAP_PROP_BUFFERSIZE not in props
    assert raw.props[fake.CAP_PROP_FRAME_WIDTH] == 1280.0
    lines = [r.getMessage() for r in caplog.records
             if r.name == "jarvis.camera"]
    assert any("asked 1280x720 MJPG" in m and "granted 1280x720 MJPG" in m
               for m in lines), lines
    assert any("driver buffer(s)" in m for m in lines), lines
    assert not any("requested" in m for m in lines), lines
    assert raw.released == 0


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
    raw = cap.raw
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
                   for p, _ in raw.sets)


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
    props = [p for p, _ in cap.raw.sets]
    assert fake.CAP_PROP_AUTO_EXPOSURE in props and fake.CAP_PROP_EXPOSURE in props
    assert props.index(fake.CAP_PROP_AUTO_EXPOSURE) < props.index(
        fake.CAP_PROP_EXPOSURE)
    # and both AFTER the mode: the format negotiation comes first
    assert props.index(fake.CAP_PROP_FRAME_HEIGHT) < props.index(
        fake.CAP_PROP_AUTO_EXPOSURE)
    assert cap.raw.props[fake.CAP_PROP_AUTO_EXPOSURE] == float(
        cam.EXPOSURE_MANUAL)
    assert cap.raw.props[fake.CAP_PROP_EXPOSURE] == 156.0
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


# ===================================================== draining the queue
# CAP_PROP_BUFFERSIZE was set to 1 to stop a slow consumer being handed a
# frame that had waited in the driver's queue. It cost half the frame rate
# (measured 2026-09-03, 8 rows of 8, scripts/camera_mode_probe.py), so the
# setting went and the staleness came back. ``DrainingCapture`` is the
# proper fix: keep the driver's buffers and DISCARD the stale ones before
# retrieving, so a slow consumer still gets the newest frame.
#
# NOTHING HERE IS A PICTURE. The double's "frame" is the INTEGER ORDINAL of
# the frame the modelled device produced, so every assertion below is about
# WHICH frame came back -- the newest or an old one -- and no pixel exists
# anywhere in this file.
class VirtualClock:
    """A clock the model moves. The queue advances it exactly where real
    time would have been spent: inside a grab that had to wait."""

    def __init__(self, t: float = 1000.0):
        self.t = float(t)

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += max(0.0, float(dt))


class QueuedCapture:
    """A V4L2-SHAPED capture double: a fixed-depth queue the device fills on
    a fixed interval, ``grab()`` that pops the oldest or WAITS for the next
    one, and ``retrieve()`` that hands back the last buffer grabbed.

    This is the shape the whole problem lives in. OpenCV's V4L2 backend
    queues four driver buffers, so a consumer slower than the device is
    handed whatever has been sitting at the head of that queue; and a grab
    that finds the queue EMPTY blocks until the device delivers, which is
    the cost the drain must not pay more than once.
    """

    def __init__(self, clock, interval_s: float = 1.0 / 15.0,
                 depth: int = 4, grab_cost_s: float = 0.0001,
                 fail_after=None, raise_after=None):
        self.clock = clock
        self.interval = float(interval_s)
        self.depth = int(depth)
        self.grab_cost = float(grab_cost_s)
        self.fail_after = fail_after
        self.raise_after = raise_after
        self.next_at = clock.now() + self.interval
        self.queue = []
        self.produced = 0
        self.overrun = 0          # frames the DRIVER threw away: queue full
        self.grabs = 0
        self.retrieves = 0
        self.held = None
        self.released = 0

    def _fill(self) -> None:
        while self.clock.now() >= self.next_at:
            self.produced += 1
            self.queue.append(self.produced)
            if len(self.queue) > self.depth:
                self.queue.pop(0)
                self.overrun += 1
            self.next_at += self.interval

    def grab(self) -> bool:
        self.grabs += 1
        if self.raise_after is not None and self.grabs > self.raise_after:
            raise OSError("the device stopped answering")
        self._fill()
        if not self.queue:
            self.clock.advance(max(0.0, self.next_at - self.clock.now()))
            self._fill()
        self.clock.advance(self.grab_cost)
        if self.fail_after is not None and self.grabs > self.fail_after:
            return False
        if not self.queue:
            return False
        self.held = self.queue.pop(0)
        return True

    def retrieve(self, *_a):
        self.retrieves += 1
        if self.held is None:
            return False, None
        return True, self.held

    def release(self) -> None:
        self.released += 1


class FloodCapture:
    """A queue that never empties and a grab that never waits -- the case
    the bound exists for."""

    def __init__(self, clock=None, cost_s: float = 0.0):
        self.clock = clock
        self.cost = float(cost_s)
        self.grabs = 0
        self.retrieves = 0

    def grab(self) -> bool:
        self.grabs += 1
        if self.clock is not None:
            self.clock.advance(self.cost)
        return True

    def retrieve(self, *_a):
        self.retrieves += 1
        return True, self.grabs

    def release(self) -> None:
        pass


def _drain(cap, clock, **kw):
    return cam.DrainingCapture(cap, now=clock.now, **kw)


# --------------------------------------------- it must help only when it helps
def test_a_consumer_at_the_device_rate_drains_nothing():
    """The half of the claim that is easy to get wrong. A consumer keeping
    pace has an EMPTY queue when it arrives, so the first grab is the one
    that waits for the device -- and the drain must stop there and cost
    exactly what an undrained read costs: one grab, one retrieve."""
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=1.0 / 15.0)
    drain = _drain(cap, clock)
    got = []
    for _ in range(30):
        ok, frame = drain.read()
        assert ok
        got.append(frame)
    assert drain.dropped == 0
    assert cap.grabs == 30 and cap.retrieves == 30
    assert got == list(range(1, 31))      # every frame, none skipped
    assert cap.overrun == 0               # and the driver never had to


def test_a_slow_consumer_is_handed_the_newest_frame_not_the_oldest():
    """The other half, and the A/B that proves the drain does something.

    Three frames wait in the queue. Undrained, the read is handed the one
    that has been sitting there longest -- three intervals old, which at
    the 15.0 fps his LifeCam delivers is 200 ms of visible lag. Drained,
    the same queue yields the newest ordinal the device has produced."""
    def run(max_drops):
        clock = VirtualClock()
        cap = QueuedCapture(clock, interval_s=1.0 / 15.0, depth=4)
        drain = _drain(cap, clock, max_drops=max_drops)
        clock.advance(3.5 / 15.0)          # nobody read for 3.5 intervals
        ok, frame = drain.read()
        return ok, frame, cap, drain

    ok, frame, cap, drain = run(0)         # max_drops 0 == today's read()
    assert ok and frame == 1               # the OLDEST: three intervals stale
    assert drain.dropped == 0

    ok, frame, cap, drain = run(cam.DRAIN_MAX_DROPS)
    assert ok and frame == cap.produced    # the NEWEST the device has made
    assert drain.dropped == 3
    assert cap.retrieves == 1              # one retrieve either way


def test_the_drain_never_waits_for_more_than_one_frame_interval():
    """It must not block. A drain that waits for the queue to run dry waits
    at exactly the rate the device delivers; this one stops at the FIRST
    grab that had to wait, so a drained read costs one frame interval at
    worst -- the same wait an undrained read pays when the queue is empty."""
    interval = 1.0 / 15.0
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=interval, depth=4)
    drain = _drain(cap, clock)
    clock.advance(3.5 * interval)
    t0 = clock.now()
    drain.read()
    assert drain.dropped == 3                  # it did drain, and then
    assert (clock.now() - t0) <= interval * 1.05

    # and at the device rate the two cost the same wait
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=interval)
    drained = _drain(cap, clock)
    t0 = clock.now()
    drained.read()
    cost_drained = clock.now() - t0
    clock2 = VirtualClock()
    cap2 = QueuedCapture(clock2, interval_s=interval)
    plain = _drain(cap2, clock2, max_drops=0)
    t0 = clock2.now()
    plain.read()
    assert abs((clock2.now() - t0) - cost_drained) < 1e-9


def test_the_drain_is_bounded_by_the_max_drops_backstop():
    """A driver that reports a queue deeper than OpenCV asks for would let
    the depth bound run away, so ``DRAIN_MAX_DROPS`` sits above it. A device
    whose queue never empties -- 99 buffers deep, and a consumer a whole
    second behind -- stops at the backstop, and stopping short of what the
    ledger counted is what ``bounded`` means."""
    clock = VirtualClock()
    flood = FloodCapture(clock)
    drain = _drain(flood, clock, depth=99)
    clock.advance(1.0)                      # a second away: ~30 arrivals
    ok, _frame = drain.read()
    assert ok
    assert flood.grabs == cam.DRAIN_MAX_DROPS + 1
    assert drain.dropped == cam.DRAIN_MAX_DROPS
    assert drain.bounded == 1


def test_the_drain_is_bounded_by_a_clock_too():
    """The count alone assumes the drops are cheap, and MEASURED at his
    camera they are not always: one busy Python thread in the process turns
    a 2 us dequeue into a 5.2 ms round trip. Nine of those would be 47 ms
    inside one read; the budget stops the chain at 30 ms, comfortably inside
    one 67.8 ms frame interval."""
    clock = VirtualClock()
    started = clock.now()
    flood = FloodCapture(clock, cost_s=MEASURED_DEQUEUE_S)
    drain = _drain(flood, clock, depth=99)
    clock.advance(1.0)
    drain.read()
    assert flood.grabs < cam.DRAIN_MAX_DROPS + 1
    assert drain.bounded == 1
    assert (clock.now() - started) <= 1.0 + cam.DRAIN_BUDGET_S \
        + MEASURED_DEQUEUE_S


# ------------------------------------------------ it must not drop the only one
def test_a_grab_that_fails_mid_drain_still_yields_the_last_good_frame():
    """The frame already in hand must survive. If the (n+1)th grab never
    arrives, the consumer gets the nth -- not nothing."""
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=1.0 / 15.0, depth=4, fail_after=2)
    drain = _drain(cap, clock)
    clock.advance(3.5 / 15.0)
    ok, frame = drain.read()
    assert ok and frame == 2               # the last one actually grabbed
    assert cap.retrieves == 1


def test_a_first_grab_that_fails_is_no_frame_rather_than_a_stale_one():
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=1.0 / 15.0, fail_after=0)
    drain = _drain(cap, clock)
    assert drain.read() == (False, None)
    assert cap.retrieves == 0               # nothing to retrieve, so it did not


def test_a_grab_that_raises_is_no_opinion_not_a_crash():
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=1.0 / 15.0, raise_after=0)
    assert _drain(cap, clock).read() == (False, None)


def test_a_retrieve_that_raises_is_no_opinion_not_a_crash():
    clock = VirtualClock()

    class Broken(FloodCapture):
        def retrieve(self, *_a):
            raise OSError("decode failed")

    assert _drain(Broken(clock), clock).read() == (False, None)


def test_a_device_that_cannot_grab_is_read_straight_through():
    """A fake with only ``read()`` -- every device double in this file, and
    any backend without grab/retrieve -- must behave exactly as before."""
    dev = FakeDevice()
    drain = cam.DrainingCapture(dev)
    ok, frame = drain.read()
    assert ok and frame is not None
    assert dev.reads == 1
    assert drain.dropped == 0
    assert drain.drains == 0                # the loop never ran


def test_a_device_far_faster_than_the_consumer_is_drained_not_given_up_on():
    """A 1000 fps device used to switch the drain OFF: its frame interval
    was shorter than the 2 ms a grab had to beat to count as a dequeue, so
    the old rule could not tell a wait from one and refused to judge. There
    is nothing to judge now. A consumer 100 frames behind such a device
    finds the queue FULL, which is exactly when draining it matters, and the
    count comes off the depth."""
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=0.001, depth=4)   # 1000 fps
    drain = _drain(cap, clock, nominal_fps=1000.0)
    got = []
    for _ in range(6):
        clock.advance(0.1)                  # 100 frames behind, every read
        ok, frame = drain.read()
        assert ok
        got.append(cap.produced - frame)
    assert drain.draining is True
    assert drain.dropped == 6 * 3           # depth - 1 every read
    # ...and the newest there was. At 1000 fps another frame can land
    # DURING the four dequeues, which is a frame this read never had.
    assert max(got) <= 1


# ------------------------------------- the numbers measured at HIS hardware
# 2026-09-03, device free (Jarvis stopped), 1280x720 MJPG, the queue backed
# up and then grabbed until the first blocking grab. grab() only -- nothing
# retrieved, decoded, shown or saved. Written up in
# scratch-0903/drain/MEASURED-dequeue.md:
#
#   condition           n     p50        p90     p99     max        >= 2.0 ms
#   idle box           135    0.001 ms   0.035   0.036   0.037 ms     0.00%
#   1 busy Py thread   135    5.222 ms   5.460   5.545   5.605 ms    89.63%
#   4 busy Py threads  120    0.004 ms   0.118   5.357   15.987 ms    9.17%
#
#   grabs that WAITED for the device: p50 67.8 ms, n=397
#
# A real dequeue costs 2 MICROSECONDS. What the drain can time is the whole
# round trip, and grab() releases the GIL for the ioctl and has to take it
# back to return -- so one competing Python thread turns that 2 us into
# 5.2 ms. Jarvis is never an idle box.
MEASURED_DEQUEUE_S = 0.00522        # p50 under ONE busy Python thread
MEASURED_WORST_DEQUEUE_S = 0.01599  # max under four
MEASURED_INTERVAL_S = 0.0678        # p50 of the grabs that waited


def test_a_busy_process_does_not_turn_every_dequeue_into_a_wait():
    """THE BUG THIS DRAIN SHIPPED WITH. The staleness test was "did this
    grab take less than 2 ms", and 2 ms is 8x BELOW the noise floor of the
    process it runs in: with one busy Python thread 89.63% of dequeues took
    longer than that (measured, table above). So the drain stopped at the
    FIRST queued buffer, handed back the OLDEST frame -- the exact lag it
    exists to remove -- and reported dropped 0, which the numbers line says
    means healthy.

    The queue is full and every dequeue costs the measured 5.2 ms -- 2.6x
    the threshold that used to mean "this one waited for the device". The
    drain must still empty it and reach the newest frame."""
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=MEASURED_INTERVAL_S, depth=4,
                        grab_cost_s=MEASURED_DEQUEUE_S)
    drain = _drain(cap, clock)
    for _ in range(3):                      # learn the delivered rate
        clock.advance(MEASURED_INTERVAL_S)
        drain.read()
    before = drain.dropped
    clock.advance(4.5 * MEASURED_INTERVAL_S)     # the queue fills to depth
    ok, frame = drain.read()
    assert ok
    assert frame == cap.produced            # the NEWEST, not the oldest
    assert drain.dropped - before == 3      # depth - 1, all of them stale
    assert drain.draining is True


def test_the_drain_is_counted_off_the_drivers_own_queue_depth():
    """How many buffers can be stale is not a guess: the driver's queue is
    ``CAP_PROP_BUFFERSIZE`` deep and readable, so at most depth - 1 can be
    waiting behind the newest one. That number is the bound -- not eight,
    which was twice a four-deep queue with no reason for the factor."""
    clock = VirtualClock()
    flood = FloodCapture(clock, cost_s=MEASURED_DEQUEUE_S)
    drain = _drain(flood, clock, depth=4)
    clock.advance(1.0)                      # a second away: ~30 arrivals
    drain.read()
    assert drain.dropped == 3               # depth - 1, and not 8
    assert flood.grabs == 4


def _work_paced(max_drops, work_s, cycles=30,
                interval_s=MEASURED_INTERVAL_S,
                grab_cost_s=0.0001):
    """A consumer slowed by its OWN WORK rather than by a sleep it can
    shorten. Every drain test before this one paced with a compensating
    sleep -- ``sleep(period - elapsed)`` -- which ABSORBS whatever the drain
    costs and hides it. Real work does not: the drain's cost lands on top of
    it, cycle after cycle."""
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=interval_s, depth=4,
                        grab_cost_s=grab_cost_s)
    drain = _drain(cap, clock, max_drops=max_drops)
    started = clock.now()
    ages = []
    for _ in range(cycles):
        clock.advance(work_s)               # decode, detect, draw
        ok, frame = drain.read()
        assert ok
        ages.append(cap.produced - frame)   # frames behind the newest
    elapsed = clock.now() - started
    return {"fps": cycles / elapsed, "age": sum(ages) / len(ages),
            "cycle_s": elapsed / cycles, "dropped": drain.dropped}


def test_a_consumer_slowed_by_its_own_work_keeps_its_frame_rate():
    """80 ms of work against a 67.8 ms device: the queue always has a buffer
    waiting, so a read that INSISTS on a frame newer than the one in hand
    has to wait for the next one -- and pays that wait on top of its own
    80 ms, every cycle. Measured at 38% of the frame rate before this test
    existed. The drain may not cost frame rate, so it takes the newest
    buffer ALREADY QUEUED and does not wait for a newer one."""
    for cost in (0.0001, MEASURED_DEQUEUE_S):
        drained = _work_paced(cam.DRAIN_MAX_DROPS, 0.080, grab_cost_s=cost)
        plain = _work_paced(0, 0.080, grab_cost_s=cost)
        assert drained["fps"] >= plain["fps"] * 0.95, cost
        assert drained["age"] <= plain["age"], cost


def test_a_consumer_far_slower_than_the_device_still_gets_the_newest():
    """The other end of the same axis: 300 ms of work per cycle against the
    same device fills the four-deep queue between reads, and the drain has
    to empty it rather than hand back the oldest."""
    for cost in (0.0001, MEASURED_DEQUEUE_S):
        drained = _work_paced(cam.DRAIN_MAX_DROPS, 0.300, grab_cost_s=cost)
        plain = _work_paced(0, 0.300, grab_cost_s=cost)
        assert drained["age"] <= 0.5, cost   # essentially always the newest
        assert plain["age"] >= 2.0, cost     # the lag the drain exists for
        # The only thing a drained cycle can cost over an undrained one is
        # the dequeues themselves, and the budget is the ceiling on those.
        assert drained["cycle_s"] <= plain["cycle_s"] + cam.DRAIN_BUDGET_S


# --------------------------------- the queue that refills while it is drained
def test_a_queue_that_fills_mid_drain_is_counted_and_still_terminates():
    """THE LEDGER IS NOT READ ONCE AND ACTED ON. The device does not stop
    producing while the drain is discarding, so a drain that decided how
    many to take on the way in would hand back a buffer that was already
    stale by the time it stopped. ``_arrivals`` therefore runs again after
    every grab, counting the grab's OWN wall clock as time the device spent
    working.

    3.5 intervals pass, so buffers 1, 2 and 3 exist when the read begins and
    buffer 4 DOES NOT. Four dequeues at 4 ms each carry the clock past the
    fifth interval boundary, buffer 4 lands mid-drain, and the drain takes
    it -- one more buffer than the queue held when it started. It still
    terminates: the depth bound stops it at four grabs with the newest frame
    in hand."""
    interval, dequeue = 0.010, 0.004
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=interval, depth=4,
                        grab_cost_s=dequeue)
    drain = _drain(cap, clock, nominal_fps=1.0 / interval)
    clock.advance(3.5 * interval)
    ok, frame = drain.read()
    assert ok
    assert cap.grabs == 4                  # one more than were queued
    assert frame > 3                       # a buffer that did not yet exist
    assert frame == cap.produced           # ...and the newest there is
    assert drain.dropped == 3
    assert drain.bounded == 1              # the depth bound, not a runaway
    assert cap.retrieves == 1

    # ...against the undrained arm on the identical queue: it takes the one
    # that has been waiting longest and never learns the others exist.
    clock2 = VirtualClock()
    cap2 = QueuedCapture(clock2, interval_s=interval, depth=4,
                         grab_cost_s=dequeue)
    plain = _drain(cap2, clock2, max_drops=0, nominal_fps=1.0 / interval)
    clock2.advance(3.5 * interval)
    ok, stale = plain.read()
    assert ok and stale == 1
    assert cap2.produced - stale == 2       # two intervals behind already


# ------------------------------------------------- the device that goes away
def test_a_device_that_disappears_mid_stream_is_no_opinion_from_then_on():
    """An unplugged camera is not an exception the app may raise. The read
    in flight keeps the last buffer that DID arrive, and every read after it
    is (False, None) -- which is what ``CameraFeed.capture`` already reads as
    no opinion -- rather than a crash on the preview thread or a stale frame
    served forever."""
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=1.0 / 15.0, depth=4, raise_after=2)
    drain = _drain(cap, clock)
    clock.advance(3.5 / 15.0)
    ok, frame = drain.read()
    assert ok and frame == 2               # the last one that did arrive
    assert drain.failures == 1
    for _ in range(3):
        assert drain.read() == (False, None)
    assert drain.failures == 4
    assert cap.retrieves == 1              # never retrieved off a dead device
    assert drain.status()["failures"] == 4


# ------------------------------------------------------------ the rate window
def test_the_rate_window_averages_the_last_few_buffers_and_no_more():
    """``interval_s`` is what the whole ledger divides by, so how it is
    measured decides everything. It is the MEAN SPACING over the last
    ``DRAIN_RATE_WINDOW`` buffers -- long enough that one slow grab cannot
    move it, short enough to follow a device that changes its own rate, and
    his LifeCam changes its own rate with the light (measured 2026-09-03).

    A window that long has an exact signature and this pins it: after the
    device halves its interval, the estimate is dragged by the old gaps for
    exactly ``DRAIN_RATE_WINDOW - 1`` buffers and then reads the new rate
    with none of the old left in it."""
    clock = VirtualClock()
    flood = FloodCapture(clock)
    drain = _drain(flood, clock, max_drops=0, nominal_fps=1000.0)
    # Before there are two buffers there is no spacing to report, and the
    # nominal stands in -- which a driver can only overstate.
    assert drain.spacing_s == 0.0
    assert drain.interval_s == pytest.approx(0.001)

    slow, fast = 0.100, 0.040
    for _ in range(cam.DRAIN_RATE_WINDOW * 2):
        clock.advance(slow)
        drain.read()
    assert drain.spacing_s == pytest.approx(slow)
    assert drain.interval_s == pytest.approx(slow)
    assert drain.bounded == 0              # an inert arm reports no bound

    gaps = cam.DRAIN_RATE_WINDOW - 1       # N stamps hold N-1 gaps
    clock.advance(fast)
    drain.read()
    assert drain.spacing_s == pytest.approx(
        ((gaps - 1) * slow + fast) / gaps)
    for _ in range(gaps - 1):
        clock.advance(fast)
        drain.read()
    assert drain.spacing_s == pytest.approx(fast)   # nothing old left
    assert drain.interval_s == pytest.approx(fast)


# ------------------------------------------------ what reaches the feed
def test_the_feed_carries_the_drains_own_state_not_only_its_count(tmp_path):
    """An INERT drain has to be visible. ``draining``, the interval it
    learned and the times a bound stopped it never left the object, so a
    drain switched off by its own guard looked exactly like a drain with
    nothing to do: drop 0.0/s either way."""
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=MEASURED_INTERVAL_S, depth=4)
    drain = cam.DrainingCapture(cap, now=clock.now)
    feed, _p, _o = _feed(tmp_path, opener=lambda: drain)
    for _ in range(8):
        assert feed.capture() is not None
    state = feed.status()["drain"]
    assert state["on"] is True
    assert state["depth"] == 4
    # the DELIVERED interval it measured, not the nominal it started from
    assert state["interval_ms"] == pytest.approx(
        MEASURED_INTERVAL_S * 1000.0, abs=1.0)
    assert state["interval_ms"] > 1000.0 / cam.DRAIN_NOMINAL_FPS
    assert state["bounded"] >= 0


def test_a_feed_with_no_drain_under_it_says_so_rather_than_raising(tmp_path):
    feed, _p, _o = _feed(tmp_path)
    assert feed.capture() is not None
    assert feed.status()["drain"]["on"] is False


# ------------------------------------------------------- through the real seam
def test_open_capture_wraps_the_device_in_the_drain_and_still_reads_it_back(
        monkeypatch, caplog):
    """``open_capture`` is where the buffer decision lives, so it is where
    the drain is attached. The wrapper still answers ``get``/``set``, which
    is what ``capture_mode`` and ``focus_probe`` ask it -- and what
    scripts/vision_selfcheck.py calls on the gate's own device."""
    fake = _FakeCv2()
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)
    with caplog.at_level(logging.INFO, logger="jarvis.camera"):
        cap = cam.open_capture("", 1280, 720, "MJPG")
    assert isinstance(cap, cam.DrainingCapture)
    assert cap.raw is fake.caps[0]
    mode = cam.capture_mode(cap)            # through the wrapper
    assert mode["width"] == 1280.0 and mode["fourcc"] == "MJPG"
    cap.release()
    assert fake.caps[0].released == 1


def test_the_drains_two_inputs_come_from_the_driver_not_from_a_constant(
        monkeypatch):
    """How deep the queue is and how fast the device claims to run are the
    two numbers the whole ledger is built on, and both are READ BACK at the
    open rather than assumed. A driver that answers 0 or -1 leaves the
    module fallbacks in place -- a queue of no depth would switch the drain
    off silently, which is the failure the ``draining`` flag exists for."""
    fake = _FakeCv2()
    monkeypatch.setattr(cam, "_import_cv2", lambda: fake)

    def granted(**kw):
        base = {"width": 1280.0, "height": 720.0, "fps": 30.0,
                "buffersize": 4.0, "convert_rgb": 1.0, "fourcc": "MJPG"}
        base.update(kw)
        monkeypatch.setattr(cam, "capture_mode", lambda _c: base)
        return cam.open_capture("", 1280, 720, "MJPG")

    cap = granted(buffersize=6.0, fps=25.0)
    assert cap.depth == 6                    # the driver's, not DRAIN_DEPTH
    assert cap.interval_s == pytest.approx(1.0 / 25.0)
    cap = granted(buffersize=-1.0, fps=0.0)
    assert cap.depth == cam.DRAIN_DEPTH
    assert cap.interval_s == pytest.approx(1.0 / cam.DRAIN_NOMINAL_FPS)


def test_the_feed_counts_the_frames_the_drain_threw_away(tmp_path):
    """The count has to reach the preview, and the only path from the device
    to the preview is the feed. A drain silently eating half the stream must
    be findable by grep, not by a night of guessing."""
    clock = VirtualClock()
    cap = QueuedCapture(clock, interval_s=1.0 / 15.0, depth=4)
    drain = cam.DrainingCapture(cap, now=clock.now)
    clock.advance(3.5 / 15.0)
    feed, _p, _o = _feed(tmp_path, opener=lambda: drain)
    assert feed.capture() == cap.produced
    assert feed.stale_dropped == 3
    assert feed.status()["stale_dropped"] == 3
    assert feed.capture() is not None
    assert feed.stale_dropped == drain.dropped


def test_a_device_with_no_drain_leaves_the_count_at_zero(tmp_path):
    """Not every opener returns a draining capture -- the suite's do not,
    and neither does a future backend. The count degrades to 0, not to an
    AttributeError inside a read."""
    feed, _p, _o = _feed(tmp_path)
    assert feed.capture() is not None
    assert feed.stale_dropped == 0
    assert feed.status()["stale_dropped"] == 0


def test_a_reopened_device_does_not_double_count_the_drops(tmp_path):
    """The drop baseline is PER WRAPPER. ``_GatedDevice`` is made fresh at
    every open, so a device re-opened after a curfew edge starts from its
    own cumulative count and the feed's total climbs by what the new device
    actually dropped -- not by the whole of its history a second time."""
    clock = VirtualClock()
    # Both devices are built NOW, so both queues have been filling since
    # before either open -- a lazily built one starts its own clock at the
    # open and would have nothing to drain.
    ready = [cam.DrainingCapture(
        QueuedCapture(clock, interval_s=1.0 / 15.0, depth=4), now=clock.now)
        for _ in range(2)]
    made = []

    def opener():
        made.append(ready[len(made)])
        return made[-1]

    feed, _p, _o = _feed(tmp_path, opener=opener)
    clock.advance(3.5 / 15.0)
    assert feed.capture() is not None
    assert feed.stale_dropped == 3
    assert made[0].dropped == 3
    feed.close()                              # the curfew edge, say
    clock.advance(3.5 / 15.0)
    assert feed.capture() is not None
    assert len(made) == 2                     # a second device, fresh counts
    assert made[1].dropped == 3               # its OWN three, from zero
    assert feed.stale_dropped == 6            # 3 + 3, not 3 + 3 + 3
