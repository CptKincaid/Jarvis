"""The camera, wired to the one thing allowed to say yes.

``jarvis/sensing.py`` already owns the answer to "may this sensor run" --
offline mode, the 21:00-07:00 curfew, the fail-to-offline rule, and
``enforce()`` walking the devices on a 15 s clock so a lens opened at 20:59
does not stay lit at 21:01. ``jarvis/eye.py`` already owns the two
permission checks around a single frame. Neither of those is re-implemented
here and neither may be: a second copy of the curfew is a copy that can
disagree, and the one that disagrees quietly is the one that leaves the lens
open.

What this module adds is the JOIN, and it exists because the obvious glue is
wrong in a way that only shows up at the device:

* ``CameraGate`` holds the device handle and is what ``SensingPolicy``
  reaches through when the curfew edge arrives with nobody speaking.
* ``Eye`` closes the device itself on its own deny edge.
* So if ``Eye`` were handed the gate's RAW device, ``Eye.close()`` would
  release it while the gate went on believing it held one -- and the gate,
  believing that, would hand the same dead handle back on the next
  permitted capture instead of opening a fresh one.

The device that crosses between them is therefore WRAPPED: ``release()``
goes back through the gate so both agree who is open, and a wrapper whose
device the gate has since released refuses to read at all (the ``epoch``
counter). That is the entire trick, and ``tests/test_camera.py`` pins each
half of it with a fake device that records its own opens and releases.

THE FIELD OF VIEW IS CONFIGURED, NEVER ASSUMED. The shipped config's comment
reasoned from "~90 deg horizontal", which is the 98 deg-diagonal Arducam
docs/vision.md section 9 recommends BUYING; every camera actually in play is
narrower -- the LifeCam Cinema he owns is 65.6 deg H (73 is its DIAGONAL),
a C930e 82.2 -- and a pixel-to-angle map that assumes 90 overstates every
off-axis angle by up to 1.4x, which is most of a 20 deg attention cone.
``lens_from_config`` therefore REFUSES to default: an unset ``hfov_deg`` is
an error, not a 90.

Nothing here imports cv2 at module scope, so this file loads on a box with
no OpenCV and no camera, and ``build()`` returns a reason instead of raising.
"""
from __future__ import annotations

import glob
import os
import threading
from typing import Any, Callable, Optional

from jarvis.eye import Eye
from jarvis.facemodels import Lens, horizontal_fov_deg
from jarvis.logs import get_logger
from jarvis.sensing import CAMERA, CameraGate
from jarvis.visionrig import MIN_DETECT_PX, HeadModel, Thresholds

log = get_logger("camera")

DEVICE_GLOB = "/dev/video*"
# MJPEG rather than raw YUY2: at 1280x720 a USB 2.0 camera cannot deliver
# uncompressed 30 fps, and the driver's silent answer to asking is to grant
# a slower mode instead of an error -- his LifeCam grants 1280x720 YUYV at a
# nominal 10 fps and 1280x720 MJPG at 30 (scripts/camera_mode_probe.py,
# 2026-09-03, both fourcc-before-size and size-before-fourcc). The granted
# mode is read back and logged in open_capture for exactly that reason.
DEFAULT_FOURCC = "MJPG"
# One driver buffer, not OpenCV's default four. See open_capture.
# None means LEAVE THE DRIVER'S DEFAULT ALONE, which is what the fast probe
# run actually did. Setting this to 1 costs exactly half the frame rate --
# MEASURED 2026-09-03, one A/B pair back to back in the same light with
# scripts/camera_mode_probe.py, 60 timed grabs per row, 8 rows out of 8:
#
#   mode              set(1)              driver default
#   720p MJPG      7.5 fps / 132.2 ms   15.0 fps /  67.9 ms
#   480p MJPG      7.5 fps / 132.2 ms   15.0 fps /  67.9 ms
#   720p YUYV      5.0 fps / 200.0 ms   10.0 fps / 100.0 ms
#   480p YUYV      7.5 fps / 132.1 ms   15.0 fps /  67.9 ms
#
# A clean 2.00x on every row, and with the default the 720p YUYV mode hits
# its granted 10.0 fps exactly -- so the device was never the cap. OpenCV's
# V4L2 backend requeues a dequeued buffer only at the NEXT grab, so with one
# buffer the driver holds none in between and every grab waits a full extra
# frame interval.
CAPTURE_BUFFERS = None
# How long a close waits for a grab already in flight before releasing the
# device anyway. One frame at the idle tier's 1.5 fps is 670 ms; a second is
# a grab that is not coming back.
CLOSE_WAIT_S = 1.0


def _import_cv2():
    """cv2, imported HERE and nowhere at module scope, so this file loads on
    a box without OpenCV and ``build()`` can report that as a reason."""
    import cv2                       # noqa: PLC0415 - deliberately lazy
    return cv2


def device_nodes() -> list:
    """Every /dev/video* node, sorted. No camera means an empty list, which
    is the honest answer on this box today."""
    try:
        return sorted(glob.glob(DEVICE_GLOB))
    except OSError:  # pragma: no cover - a broken /dev is not our problem
        return []


def device_present(device: str = "") -> bool:
    """Is there a camera to stop? ``SensingPolicy`` reports a device that is
    absent separately from one it stopped, because "the camera is off" is a
    lie when there was never a camera."""
    if device:
        # open_capture takes a bare digit as a cv2 index, so the question
        # "is /dev/video<N> there" is the one to ask; os.path.exists("0")
        # is never true and read a lit camera as absent (jarvis/sensing.py
        # _switch_devices, which now stops regardless -- this keeps the
        # spoken WORDING honest too).
        if device.isdigit():
            device = "/dev/video%d" % int(device)
        return os.path.exists(device)
    return bool(device_nodes())


# ------------------------------------------------------------- the config
def _cfg_get(cfg, key: str, default=None):
    get = getattr(cfg, "get", None)
    if not callable(get):
        return default
    value = get(key, default)
    return default if value is None else value


def lens_from_config(cfg) -> Lens:
    """Build the lens geometry from his config, or refuse.

    ``camera.hfov_deg`` is the horizontal field, in degrees, of the camera
    that is actually plugged in. ``camera.diag_fov_deg`` is accepted because
    webcams are SOLD by the diagonal and converting it here is better than
    inviting him to type 73 into a horizontal field -- the conversion is a
    ratio of tangents, not of numbers, and getting it wrong turns a 73 deg
    LifeCam into a field 11% wider than it has.

    There is deliberately no default. A guessed field of view is the exact
    error this whole lane exists to undo.
    """
    width = int(_cfg_get(cfg, "camera.width", 0) or 0)
    height = int(_cfg_get(cfg, "camera.height", 0) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("camera.width/camera.height are not set")
    hfov = float(_cfg_get(cfg, "camera.hfov_deg", 0.0) or 0.0)
    if hfov:
        return Lens(width, height, hfov)
    diag = float(_cfg_get(cfg, "camera.diag_fov_deg", 0.0) or 0.0)
    if diag:
        return Lens(width, height,
                    horizontal_fov_deg(diag, float(width), float(height)))
    raise ValueError(
        "camera.hfov_deg is not set. There is no default: the LifeCam "
        "Cinema is 65.6 deg horizontal, a C930e 82.2, and the ~90 the old "
        "comment reasoned from belongs to a camera that is not plugged in. "
        "Set camera.hfov_deg, or camera.diag_fov_deg if the spec sheet "
        "quotes the diagonal.")


def head_from_config(cfg) -> HeadModel:
    ratio = float(_cfg_get(cfg, "camera.nose_ratio", 0.0) or 0.0)
    return HeadModel(ratio) if ratio else HeadModel()


def thresholds_from_config(cfg) -> Thresholds:
    return Thresholds(
        min_conf=float(_cfg_get(cfg, "camera.min_conf", 0.6)),
        cone_deg=float(_cfg_get(cfg, "camera.cone_deg", 20.0)),
        cone_hysteresis_deg=float(
            _cfg_get(cfg, "camera.cone_hysteresis_deg", 5.0)),
        cone_centre_deg=float(_cfg_get(cfg, "camera.cone_centre_deg", 0.0)),
        dwell_s=float(_cfg_get(cfg, "camera.dwell_s", 0.6)),
        identity_min=float(_cfg_get(cfg, "camera.identity_min", 0.363)),
        min_detect_px=float(_cfg_get(cfg, "camera.min_detect_px",
                                     MIN_DETECT_PX)),
    )


# ------------------------------------------------------------- the device
def open_capture(device: str = "", width: int = 1280, height: int = 720,
                 fourcc: str = DEFAULT_FOURCC):
    """cv2.VideoCapture, opened, asked for a mode, and the GRANTED mode read
    back and logged. Raises if it will not open -- ``Eye`` reads that as no
    opinion, which is the same behaviour as having no camera at all.

    THE GRANTED MODE IS LOGGED HERE, once per open, because until 2026-09-03
    nothing in the running app read it back and the one time it mattered
    nobody could tell what the device was doing: the preview logged
    ``7.5 fps`` at 1280x720 whatever rate it asked for, and the explanation
    everyone reached for first -- that the driver had quietly declined MJPG
    and granted a bandwidth-capped YUYV stream -- was wrong. Measured with
    scripts/camera_mode_probe.py, twice (2026-09-03 02:36 and 07:17, Jarvis
    stopped, ``grab()`` only, nothing retrieved): MJPG IS granted at
    1280x720 in either set order at a nominal 30 fps, and the device
    delivered 3.7-3.9 fps in every 30 fps mode it has, 640x480 included,
    both runs -- so the rate is the device's own, not the format's and not
    the bus's. What sets it is not proven; the whole-multiple frame
    intervals (268 ms = 8 x 33 ms at the probe, 133 ms = 4 x 33 ms with him
    at the desk) are what auto-exposure lengthening the interval for a dim
    scene looks like. The line printed here puts asked-against-granted
    beside the preview's own rate line so that the next such question is a
    grep of the log rather than a night of guessing.

    ``CAP_PROP_BUFFERSIZE`` IS LEFT AT THE DRIVER'S DEFAULT, and that is a
    correction. It was set to ONE to stop a slow consumer being handed a
    frame that had waited in the queue -- up to three intervals old, which
    is lag he can see, and the reasoning was right. The cost was not
    measured until 2026-09-03, and the cost is HALF THE FRAME RATE: an A/B
    pair in the same light gave a clean 2.00x on all eight rows (see
    CAPTURE_BUFFERS above), and with the default the 720p YUYV mode reaches
    its granted 10.0 fps exactly, so the device was never the cap.

    Half the rate is the worse trade. At his configured ``preview_fps`` of
    15 the consumer now keeps pace with the device (15.0 fps delivered), so
    the queue does not build and the staleness this was fighting does not
    arise; it only bit when the consumer ran far slower than the device
    (the old ``6.0 fps  grab 11 ms`` line, 6 requested against 15 delivered).

    THE PROPER FIX IS TO DRAIN, NOT TO STARVE: keep the driver's buffers and
    discard the stale ones before retrieving, so a slow consumer still gets
    the newest frame at full rate. That is not built yet, and until it is,
    a consumer configured well below the delivered rate can still be handed
    a frame up to three intervals old.

    cv2 is imported HERE, not at module scope, so that a box without OpenCV
    still loads jarvis.camera and still reports honestly.
    """
    cv2 = _import_cv2()
    target: Any = device if device else 0
    if isinstance(target, str) and target.isdigit():
        target = int(target)
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        cap.release()
        raise OSError("could not open camera %r" % (device or 0))
    try:
        # FOURCC before the size. The probe found this driver honours both
        # orders, but the order in which OpenCV's V4L2 backend re-negotiates
        # has changed between versions and format-first is the one that
        # has never been reported broken.
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if CAPTURE_BUFFERS is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFERS)
    except Exception:  # noqa: BLE001 - an unsupported mode is not a failure
        log.debug("camera: the driver refused a mode request", exc_info=True)
    try:
        got = capture_mode(cap)
        log.info("camera: asked %dx%d %s; granted %.0fx%.0f %s at %.1f fps "
                 "nominal, %.0f driver buffer(s) -- the delivered rate is "
                 "what the preview's own line reports",
                 int(width), int(height), fourcc or "-", got["width"],
                 got["height"], got["fourcc"] or "?", got["fps"],
                 got["buffersize"])
    except Exception:  # noqa: BLE001 - a read-back is not worth a crash
        log.debug("camera: could not read the granted mode back",
                  exc_info=True)
    return cap


class _GatedDevice:
    """The handle ``Eye`` holds: a device whose close goes through the gate.

    ``read`` delegates. ``release`` calls the GATE's release rather than the
    device's, so a close initiated by ``Eye`` and a close initiated by
    ``SensingPolicy.enforce`` leave the two objects agreeing about who holds
    what. A wrapper made before a close is stale afterwards and reads False,
    so a handle released by the curfew cannot be read from by a loop that
    has not noticed yet.
    """

    __slots__ = ("_feed", "_raw", "_epoch")

    def __init__(self, feed: "CameraFeed", raw, epoch: int):
        self._feed = feed
        self._raw = raw
        self._epoch = epoch

    def read(self):
        if self._epoch != self._feed.epoch:
            return False, None
        # Held across the grab so the curfew cannot release the device out
        # from under an in-flight read. cv2.VideoCapture is not thread-safe
        # and ``SensingPolicy.enforce`` runs on its own thread; releasing a
        # capture mid-read is undefined behaviour, and "the process crashed,
        # which did close the camera" is not the enforcement anybody wants.
        # Nothing else is taken while it is held, so the only lock order
        # that exists is gate -> read, and there is no cycle.
        with self._feed.read_lock:
            if self._epoch != self._feed.epoch:
                return False, None
            return self._raw.read()

    def release(self) -> None:
        self._feed.gate.release()


class CameraFeed:
    """``CameraGate`` + ``Eye``, joined so neither can be bypassed.

    Every frame costs two permission reads and at most one device open, and
    the device is attached to the policy so the clock-driven guard can shut
    it without anybody calling this object at all.
    """

    def __init__(self, policy, opener: Callable[[], Any], *,
                 present: Optional[Callable[[], bool]] = None,
                 on_blind: Optional[Callable[[], None]] = None,
                 name: str = CAMERA, lens: Optional[Lens] = None):
        self.name = name
        self.lens = lens
        self.policy = policy
        self._opener = opener
        self._lock = threading.RLock()
        # NOT reentrant: _close_raw waits on it with a timeout, which an
        # RLock owned by this same thread would grant instantly and defeat.
        self.read_lock = threading.Lock()
        self.epoch = 0
        self.gate = CameraGate(policy, opener, closer=self._close_raw,
                               name=name, present=present)
        self.eye = Eye(allow=lambda: self._allowed(),
                       open_device=self._open_gated, on_blind=on_blind)
        self.frames = 0

    # -------------------------------------------------------------- wiring
    def _allowed(self) -> bool:
        """``Eye`` treats anything but exactly True as a no, and so does the
        gate. Routing through the gate keeps ONE reading of the policy."""
        return self.gate.allowed() is True

    def _close_raw(self, dev) -> None:
        """The gate's closer. Bumping the epoch is what makes every wrapper
        handed out before this moment refuse to read.

        It WAITS, briefly, for a read already in flight -- see
        ``_GatedDevice.read``. The wait is bounded because the alternative to
        a bounded wait is an unbounded one: a camera that has stopped
        answering can block a grab for seconds, and a privacy control that
        can be postponed indefinitely by a wedged device is not one. So after
        ``CLOSE_WAIT_S`` the device is released regardless and the fact is
        logged, which is the honest ordering of the two risks.
        """
        got = self.read_lock.acquire(timeout=CLOSE_WAIT_S)
        try:
            with self._lock:
                self.epoch += 1
            if not got:
                log.warning("camera: a grab was still in flight after %.1fs; "
                            "releasing the device anyway", CLOSE_WAIT_S)
            release = (getattr(dev, "release", None)
                       or getattr(dev, "close", None))
            if callable(release):
                release()
        finally:
            if got:
                self.read_lock.release()

    def _open_gated(self):
        raw = self.gate.open()
        if raw is None:
            # Denied between Eye's own check and this call. Eye reads an
            # exception as no opinion, which is the correct reading.
            raise PermissionError("sensing denied the camera")
        return _GatedDevice(self, raw, self.epoch)

    # ------------------------------------------------------------- capture
    def capture(self):
        """One frame, or None meaning NO OPINION -- denied, absent, unplugged
        or dropped because offline was set mid-grab. A consumer that has to
        tell those apart will get one of them wrong, and the safe reading of
        all four is "behave as if there were no camera"."""
        frame = self.eye.capture()
        if frame is not None:
            self.frames += 1
        return frame

    def close(self) -> None:
        self.eye.close()
        self.gate.release()

    # -------------------------------------------------------------- report
    @property
    def device_open(self) -> bool:
        return self.gate.is_open

    @property
    def opens(self) -> int:
        return self.eye.opens

    @property
    def denials(self) -> int:
        return self.eye.denials

    @property
    def frames_dropped(self) -> int:
        return self.eye.reads_dropped

    def status(self) -> dict:
        st = self.gate.status()
        st.update({"opens": self.opens, "denials": self.denials,
                   "frames": self.frames, "dropped": self.frames_dropped,
                   "blind_edges": self.eye.blind_edges})
        return st


class FeedSource:
    """A ``CameraFeed`` as the frame source ``visionrig.Rig`` expects.

    The rig's contract is cv2.VideoCapture's ``read() -> (ok, frame)``; the
    feed's is "a frame, or None meaning no opinion". This is the two-line
    join, and it exists as a class so the harness runs against the GATED
    camera rather than around it: a rig pointed at a raw VideoCapture would
    be a second code path that ignores offline mode.
    """

    def __init__(self, feed: CameraFeed):
        self.feed = feed

    def read(self):
        frame = self.feed.capture()
        return (frame is not None), frame

    def release(self) -> None:
        self.feed.close()


def fourcc_name(value) -> str:
    """The four characters behind cv2's packed integer, e.g. 'MJPG'."""
    try:
        code = int(value)
    except (TypeError, ValueError):
        return ""
    if code <= 0:
        return ""
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def capture_mode(cap) -> dict:
    """What the driver actually GRANTED, which is not what it was asked for.

    A camera asked for a mode it does not have does not raise -- v4l2 picks
    the nearest thing and says nothing. That is how ``width: 1920`` sat in
    the config for a 720p camera without anybody finding out, so the
    self-check reads the mode back and prints both numbers.
    """
    cv2 = _import_cv2()
    out = {}
    for name, prop in (("width", cv2.CAP_PROP_FRAME_WIDTH),
                       ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                       ("fps", cv2.CAP_PROP_FPS),
                       ("buffersize", cv2.CAP_PROP_BUFFERSIZE),
                       ("convert_rgb", cv2.CAP_PROP_CONVERT_RGB)):
        try:
            out[name] = float(cap.get(prop))
        except Exception:  # noqa: BLE001 - an unsupported property is data
            out[name] = -1.0
    try:
        out["fourcc"] = fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))
    except Exception:  # noqa: BLE001
        out["fourcc"] = ""
    return out


def focus_probe(cap) -> dict:
    """Can focus be PINNED? Numbers only, and no judgement about the scene.

    A camera that keeps hunting refocuses on whatever moved, which at a desk
    is his hands; the fix is autofocus off plus a fixed focus value, and
    whether the driver will accept either is a per-camera fact nobody can
    look up. This asks, and reports what came back.
    """
    cv2 = _import_cv2()
    out = {}
    for name, prop, value in (("autofocus", cv2.CAP_PROP_AUTOFOCUS, 0.0),
                              ("focus", cv2.CAP_PROP_FOCUS, 0.0)):
        try:
            before = float(cap.get(prop))
        except Exception:  # noqa: BLE001
            before = -1.0
        try:
            accepted = bool(cap.set(prop, value))
        except Exception:  # noqa: BLE001
            accepted = False
        try:
            after = float(cap.get(prop))
        except Exception:  # noqa: BLE001
            after = -1.0
        out[name] = {"before": before, "set_accepted": accepted,
                     "after": after,
                     "pinned": bool(accepted and after == value)}
    return out


def detector_from_config(cfg, score_threshold: Optional[float] = None):
    """``(detector, reason)`` -- never raises, and NEVER falls back.

    With no weights on disk this returns ``(None, "...missing: /path...")``
    and the camera path degrades to no opinion, which is byte-for-byte
    today's behaviour. What it must never do is return a DIFFERENT detector:
    VSS shipped a yunet path that silently fell back to head_box and has
    never once run, and the whole reason this returns a reason string is so
    that failure is visible in a log and in the self-check.
    """
    from jarvis import facedetect          # noqa: PLC0415 - keeps cv2 lazy
    try:
        size = (int(_cfg_get(cfg, "camera.detect_width", 320)),
                int(_cfg_get(cfg, "camera.detect_height", 180)))
        det = facedetect.load_detector(
            min_conf=float(_cfg_get(cfg, "camera.min_conf", 0.6)),
            input_size=size,
            threads=int(_cfg_get(cfg, "camera.threads", 2)),
            score_threshold=score_threshold,
            model_dir=str(_cfg_get(cfg, "camera.model_dir", "") or "") or None)
        return det, ""
    except Exception as exc:  # noqa: BLE001 - absence is not a crash
        log.info("camera: no face detector (%s)", exc)
        return None, str(exc)


def recogniser_from_config(cfg):
    """``(recogniser, reason)``. None unless ``camera.identity`` is on: with
    it off, nothing about his face is ever computed, let alone written."""
    from jarvis import facedetect          # noqa: PLC0415
    if not bool(_cfg_get(cfg, "camera.identity", False)):
        return None, "camera.identity is false"
    try:
        return facedetect.load_recogniser(
            min_conf=float(_cfg_get(cfg, "camera.min_conf", 0.6)),
            model_dir=str(_cfg_get(cfg, "camera.model_dir", "") or "")
            or None), ""
    except Exception as exc:  # noqa: BLE001
        log.info("camera: no face recogniser (%s)", exc)
        return None, str(exc)


def build(cfg, policy, opener: Optional[Callable[[], Any]] = None,
          on_blind: Optional[Callable[[], None]] = None) -> tuple:
    """``(feed, reason)`` -- never raises, so a camera cannot take Jarvis down.

    ``feed`` is None with a reason whenever the camera is off, the config is
    incomplete or unreadable, or the geometry cannot be established. The
    reason is a sentence for the log and the self-check, not a code: the
    failure this guards against is a camera path that declines silently.
    """
    try:
        if not bool(_cfg_get(cfg, "camera.enabled", False)):
            return None, "camera.enabled is false"
        lens = lens_from_config(cfg)
        device = str(_cfg_get(cfg, "camera.device", "") or "")
        fourcc = str(_cfg_get(cfg, "camera.fourcc", DEFAULT_FOURCC) or "")
        if opener is None:
            def opener():                       # noqa: E306 - one call site
                return open_capture(device, lens.width_px, lens.height_px,
                                    fourcc)
        feed = CameraFeed(policy, opener, lens=lens, on_blind=on_blind,
                          present=lambda: device_present(device))
        return feed, ""
    except Exception as exc:  # noqa: BLE001 - a camera must not end the app
        log.warning("camera: not wired (%s: %s)", type(exc).__name__, exc)
        return None, "%s: %s" % (type(exc).__name__, exc)
