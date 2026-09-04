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
# One buffer requested, not OpenCV's default four -- and this is a REVERT
# to that, not the original choice. None means leave the driver's default
# alone. The history, in order, because two commits on one day disagreed
# and the comment here described the wrong one for a while:
#
# 1. Set to 1 (12eadf0) so a slow consumer could not be handed a frame that
#    had waited up to three intervals in the queue -- lag he can see.
# 2. MEASURED IN THE PROBE to cost exactly half the rate (c228a01): one A/B
#    pair back to back in the same light with scripts/camera_mode_probe.py,
#    60 timed grabs per row, 8 rows out of 8:
#
#      mode              set(1)              driver default
#      720p MJPG      7.5 fps / 132.2 ms   15.0 fps /  67.9 ms
#      480p MJPG      7.5 fps / 132.2 ms   15.0 fps /  67.9 ms
#      720p YUYV      5.0 fps / 200.0 ms   10.0 fps / 100.0 ms
#      480p YUYV      7.5 fps / 132.1 ms   15.0 fps /  67.9 ms
#
#    A clean 2.00x on every row, with the 720p YUYV mode hitting its
#    granted 10.0 fps exactly at the default -- so in the probe the device
#    was never the cap. That measurement stands. It was set to None.
# 3. Put back to 1 the same evening (9ba1c56), because THE APP DID NOT GET
#    THE RATE: with the driver's buffers its own rate line read 7.6 fps /
#    132 ms (19:47), the same ~7.5 it had with one buffer, and he saw the
#    staleness at once ("way less accurate", boxes on "random objects").
#    The probe grabs in a tight loop and never retrieves; the app grabs,
#    retrieves, decodes and works; something in that difference eats the
#    other half and NOBODY KNOWS WHAT YET. So the setting that demonstrably
#    keeps his picture fresh wins over a rate gain that does not reach the
#    app.
#
# OpenCV's V4L2 backend requeues a dequeued buffer only at the NEXT grab,
# so with one buffer the driver holds none in between and every grab waits
# a full extra frame interval -- the mechanism that fits the probe's 2x.
# The count the driver actually allocated is not observable through cv2:
# CAP_PROP_BUFFERSIZE's get() returns OpenCV's own stored request, never
# compared with what VIDIOC_REQBUFS granted, so "set(1) read back 1" says
# only that the request was accepted, and the log line below says
# "requested" for that reason. The proper fix is to DRAIN the queue, not
# starve it (branch camera-drain); when it lands the buffers come back.
CAPTURE_BUFFERS = 1
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


def _proc_name(pid: int) -> str:
    """``/proc/<pid>/comm``, or "". Never raises; a name is a courtesy."""
    try:
        with open("/proc/%d/comm" % int(pid), encoding="utf-8",
                  errors="replace") as fh:
            return fh.read().strip()
    except Exception:            # noqa: BLE001 - the process may have gone
        return ""


def device_holder(device: str = "") -> tuple:
    """``(pid, name)`` of a process holding ``device`` open, else ``(0, "")``.

    WHY THIS EXISTS. V4L2 capture is EXCLUSIVE: a second opener does not get
    a queue, it gets a failure, and cv2 reports that failure the same way it
    reports a missing camera. On 2026-09-03 that cost half an hour --
    ``scripts/face_enrol.py`` said "sensing denied the camera, or the device
    went away" while sensing said camera=True and the device was sitting
    right there, held by the running Jarvis on fd 14. Both halves of the
    sentence were false and both sent him to check something that was fine.

    IT OPENS NOTHING. It reads the /proc fd symlinks, which is a directory
    listing and a readlink -- the same information ``fuser`` prints, at no
    risk of taking the device away from whoever legitimately has it. A
    process this user may not read is simply skipped, so the honest failure
    here is ``(0, "")`` -- "somebody, and I cannot say who" -- never a guess.
    """
    node = str(device or "")
    if node.isdigit():
        node = "/dev/video%d" % int(node)
    if not node:
        nodes = device_nodes()
        node = nodes[0] if nodes else ""
    if not node:
        return 0, ""
    me = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:              # pragma: no cover - a broken /proc
        return 0, ""
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            # Our own fd is not an answer to "who has it instead of me".
            continue
        fd_dir = "/proc/%d/fd" % pid
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue             # not ours to read, or already exited
        for fd in fds:
            try:
                if os.readlink(os.path.join(fd_dir, fd)) != node:
                    continue
            except OSError:
                continue
            return pid, _proc_name(pid)
    return 0, ""


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
    1280x720 in either set order at a nominal 30 fps, so the format is not
    what sets the rate and neither is the bus (640x480 delivered the same
    3.7-3.9 fps in both formats). That 3.7-3.9 was the PROBE'S
    configuration in that morning's light -- one buffer, 30 fps requested
    -- not the device's ceiling: the same configuration gave 7.5 that
    afternoon and the driver's default gave 15.0 (CAPTURE_BUFFERS above),
    while the app gets ~7.5 either way. What sets the app's 7.5 is not
    proven; the whole-multiple frame intervals (268 ms = 8 x 33 ms, 133 =
    4 x 33, 68 = 2 x 33) are what auto-exposure lengthening the interval
    for a dim scene looks like, and the light was never controlled for.
    The line printed here puts asked-against-granted beside the preview's
    own rate line so that the next such question is a grep of the log
    rather than a night of guessing.

    ``CAP_PROP_BUFFERSIZE`` IS REQUESTED AS ONE, and the history of that
    number is on CAPTURE_BUFFERS: it halves the probe's rate, does not
    change the app's, and is what keeps the picture he sees fresh. The
    staleness it fights is a frame up to three intervals old handed to a
    consumer that runs slower than the device. The old ``6.0 fps  grab
    11 ms`` line at 6 requested is CONSISTENT with that -- an 11 ms grab
    from a 133 ms device is most plausibly a frame that was already waiting
    -- but frame age was never timed, and 9ba1c56's "boxes on random
    objects" with the driver's buffers is the nearest thing to a
    measurement of it. Expected mechanism, not yet measured.

    THE PROPER FIX IS TO DRAIN, NOT TO STARVE: keep the driver's buffers and
    discard the stale ones before retrieving, so a slow consumer still gets
    the newest frame at full rate. That is not built yet (branch
    camera-drain), and until it is, the single buffer stays.

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
                 "nominal, %.0f buffer(s) requested -- the delivered rate "
                 "is what the preview's own line reports",
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
                 name: str = CAMERA, lens: Optional[Lens] = None,
                 device: str = ""):
        self.name = name
        self.lens = lens
        # The node this feed was pointed at, kept ONLY so that a failure to
        # open can name the process holding it (see FeedSource.reason). It is
        # never opened from here; the opener owns that.
        self.device = str(device or "")
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

    @property
    def reason(self) -> str:
        """Why the last read came back empty, in ONE true sentence.

        ``CameraFeed.capture`` returns None for five different reasons on
        purpose -- a consumer that has to tell them apart will get one of
        them wrong -- but a HUMAN being told to go and fix it needs exactly
        that distinction, and the caller that guessed at it got both halves
        wrong on 2026-09-03 (see ``faceenrol.FRAMES_STOPPED``). So the guess
        is replaced by the four things this object can actually check: what
        sensing says, whether the node exists, whether we ever got the device
        open at all, and who has it if we did not.

        Never raises and never opens anything. "" means "I have nothing
        better than the caller's own sentence", which is an honest answer.
        """
        feed = self.feed
        try:
            st = feed.status()
        except Exception:        # noqa: BLE001 - a duck-typed feed
            log.debug("camera: the feed could not report status",
                      exc_info=True)
            return ""
        if st.get("allowed") is not True:
            why = ""
            try:
                why = str(feed.policy.status().get("reason") or "")
            except Exception:    # noqa: BLE001 - a slim/absent policy
                why = ""
            return ("sensing is holding the camera shut (%s)" % why) if why \
                else "sensing is holding the camera shut"
        device = getattr(feed, "device", "") or ""
        try:
            there = device_present(device)
        except Exception:        # noqa: BLE001
            there = True
        if not there:
            return ("the camera device is not there (%s)" % device) if device \
                else "there is no camera device"
        if not int(st.get("opens") or 0):
            # Sensing allows it, the node exists, and we never once got it
            # open. On a V4L2 device that means somebody else has it.
            pid, name = 0, ""
            try:
                pid, name = device_holder(device)
            except Exception:    # noqa: BLE001 - /proc is a courtesy
                log.debug("camera: could not look for the holder",
                          exc_info=True)
            node = device or "the camera"
            if pid and name:
                return ("%s is already open -- %s (pid %d) is holding it, "
                        "and v4l2 only allows one" % (node, name, pid))
            if pid:
                return ("%s is already open -- pid %d is holding it, and "
                        "v4l2 only allows one" % (node, pid))
            return ("%s could not be opened; sensing allows it and the "
                    "device is there, so another process is holding it"
                    % node)
        return ("the camera stopped answering after %d frame(s)"
                % int(st.get("frames") or 0))


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


def face_backend_from_config(cfg) -> str:
    """Which face model pair his config asks for. "" means the default.

    ``camera.face_backend`` is the one-line reversal: set it to "opencv" and
    the box goes back to YuNet + SFace and to the enrolment already on disk.
    An unknown value RAISES out of ``facemodels.backend_for`` rather than
    quietly leaving the old models running.
    """
    return str(_cfg_get(cfg, "camera.face_backend", "") or "")


def identity_min_warning(cfg) -> str:
    """"" or the sentence saying his identity bar was tuned for another model.

    ``camera.identity_min`` is ONE number and there is ONE of it. It was
    raised to 0.47 on 2026-09-03 from SFace scores measured on his own face
    and his own wall. SFace's cosines and ArcFace's are different
    distributions of a different model's vectors, so that number does not
    carry across -- and a threshold calibrated for one vector silently applied
    to another is precisely the failure ``jarvis/eye.py`` names in its own
    docstring.

    Nothing here CHANGES the bar. Guessing a replacement would be the same
    mistake in the other direction. What it does is refuse to let the swap
    happen quietly: the bar is unmeasured for this model until he runs
    ``scripts/face_model_compare.py`` on his own face.
    """
    from jarvis import facegallery as fgal        # noqa: PLC0415
    from jarvis import facemodels as fmod         # noqa: PLC0415
    try:
        back = fmod.backend_for(face_backend_from_config(cfg))
    except Exception:  # noqa: BLE001 - the backend error is reported elsewhere
        return ""
    if fgal.cosine_same(back.embed_model) is not None:
        return ""
    bar = float(_cfg_get(cfg, "camera.identity_min", 0.363))
    return ("camera.identity_min is %.3f and that number was measured "
            "against SFace's vectors, not %s's. No same-person cosine has "
            "been measured for %s on this machine -- run "
            "scripts/face_model_compare.py after you re-enrol and set the "
            "bar from what it reports."
            % (bar, back.embed_model, back.embed_model))


def gallery_from_config(cfg):
    """``(gallery, reason)`` -- the enrolled faces FOR THE ACTIVE MODEL.

    ``gallery`` is None with a reason whenever identity cannot be offered, and
    after a model swap that reason is the ONE LINE that says what to do:
    re-enrol, or set ``camera.face_backend`` back. The failure this replaces
    is "nobody is enrolled" printed over thirteen enrolled samples that this
    model simply cannot read -- true, useless, and exactly how a swap turns
    into a week of confusion.

    A cross-model gallery is never merged, never scaled and never scored: an
    ArcFace 512-vector against an SFace 128-vector is not a weak comparison,
    it is not a comparison.
    """
    from jarvis.config import PATHS               # noqa: PLC0415
    from jarvis.facegallery import default_gallery  # noqa: PLC0415
    del PATHS
    try:
        gallery = default_gallery(backend=face_backend_from_config(cfg))
    except Exception as exc:  # noqa: BLE001 - a bad backend name is a reason
        return None, str(exc)
    try:
        loaded = bool(gallery.load())
    except Exception as exc:  # noqa: BLE001
        return None, "the face gallery would not open (%s)" % exc
    if not loaded or not gallery.labels():
        if gallery.foreign_generations:
            return None, gallery.reenrol_message()
        return None, "nobody is enrolled for %s" % gallery.model
    return gallery, ""


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
            model_dir=str(_cfg_get(cfg, "camera.model_dir", "") or "") or None,
            backend=face_backend_from_config(cfg))
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
        # input_size is the DETECTOR's, and only the ArcFace branch uses it:
        # every call site hands the recogniser a full capture frame with a row
        # in detector pixels, so without this the crop is taken from the
        # frame's top-left corner. See ArcFaceRecogniser.
        return facedetect.load_recogniser(
            min_conf=float(_cfg_get(cfg, "camera.min_conf", 0.6)),
            model_dir=str(_cfg_get(cfg, "camera.model_dir", "") or "")
            or None,
            backend=face_backend_from_config(cfg),
            input_size=(int(_cfg_get(cfg, "camera.detect_width", 320)),
                        int(_cfg_get(cfg, "camera.detect_height", 180)))), ""
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
                          present=lambda: device_present(device),
                          device=device)
        return feed, ""
    except Exception as exc:  # noqa: BLE001 - a camera must not end the app
        log.warning("camera: not wired (%s: %s)", type(exc).__name__, exc)
        return None, "%s: %s" % (type(exc).__name__, exc)
