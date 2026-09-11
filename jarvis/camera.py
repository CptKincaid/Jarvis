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
import time
from collections import deque
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
# NOTHING IS REQUESTED: the driver's own queue stands, and the stale
# buffers in it are DRAINED (``DrainingCapture`` below) instead of never
# being queued. None means leave the driver's default alone. Those two
# facts go together and neither is safe without the other, which is the
# whole point of the history below -- three commits on one day disagreed
# about this number and the comment here described the wrong one for a
# while:
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
# only that the request was accepted. Nothing sets it now, so the log line
# below prints that read-back as "driver buffer(s)" and claims nothing about
# a request that was never made.
#
# 4. AND THIS IS STEP 4, the one the comment used to say was owed. The
#    proper fix is to DRAIN the queue, not starve it, and it is built:
#    ``DrainingCapture`` keeps the driver's buffers and grabs the stale ones
#    away before retrieving, so a slow consumer is handed the NEWEST frame
#    without the device ever being throttled to one buffer. What step 3
#    reverted was a naked buffer count with no drain under it, and that
#    revert was right on its own terms -- the staleness he saw at 19:47 was
#    real. It is not this. The two belong together: the buffers come back
#    only because the drain exists.
#
#    WHAT IS MEASURED AND WHAT IS NOT. The 2.00x above is measured, at his
#    camera, in the probe. The drain's own arms are measured against a
#    MODELLED four-buffer queue on that camera's measured 66.7 ms interval
#    (scripts/camera_drain_probe.py --model 15) because /dev/video0 was
#    held by the running Jarvis and a number is not worth blinding the
#    assistant for: drained and undrained deliver the same frames a second
#    at every consumer rate, and the drained arm hands back a frame ~0
#    intervals old where the undrained one hands back 2-3. WHETHER THE APP
#    NOW GETS THE 15.0 IS UNMEASURED AT THE DEVICE -- that is step 3's open
#    question and it is his to settle, with Jarvis stopped, by running
#    scripts/camera_drain_probe.py with no --model.
CAPTURE_BUFFERS = None
# How long a close waits for a grab already in flight before releasing the
# device anyway. One frame at the idle tier's 1.5 fps is 670 ms; a second is
# a grab that is not coming back.
CLOSE_WAIT_S = 1.0

# ----------------------------------------------------- the drain's five dials
# HOW DEEP THE DRIVER'S QUEUE IS, and therefore how many buffers can be
# waiting BEHIND the newest one: depth - 1. That count is the drain's
# operating bound, and it is not a guess -- OpenCV's V4L2 backend asks for
# four buffers and ``CAP_PROP_BUFFERSIZE`` reads back what it got, which
# ``open_capture`` passes in. This constant is the fallback for a driver
# that will not answer, and 4 is what OpenCV asks for.
DRAIN_DEPTH = 4
# The frame interval the drain STARTS OUT assuming, as a rate, until it has
# timed the spacing between buffers for itself. A seed for the arithmetic
# below, not a claim about any camera: what makes it safe is that a driver's
# NOMINAL rate can only be faster than what it delivers -- his LifeCam grants
# 30 nominal and delivers 15.0 (measured 2026-09-03,
# scripts/camera_mode_probe.py) -- so a nominal interval is a LOWER bound on
# the real one, and a lower bound makes the drain over-estimate what is
# waiting rather than miss it. ``open_capture`` passes the driver's own
# number; this is the fallback.
DRAIN_NOMINAL_FPS = 30.0
# The backstop on drops per read, sitting ABOVE the depth bound for a driver
# that reports a deeper queue than OpenCV asks for. Twice a four-deep queue,
# so on his camera it never bites: the depth bound stops the loop at 3.
DRAIN_MAX_DROPS = 8
# ...and the backstop on the TIME per read, which is what stops a read
# blocking TWICE. A grab that has to wait for the device costs a frame
# interval -- p50 67.8 ms at his LifeCam, n=397, measured 2026-09-03 with
# the device free (scratch-0903/drain/MEASURED-dequeue.md) -- so the first
# such grab blows this budget on its own and the loop stops with the buffer
# it just took. Dequeues do not: the same run measured p50 5.222 ms each
# with one busy Python thread in the process and a worst single dequeue of
# 15.99 ms under four, so a full depth-1 drain of three of them is ~16 ms
# and fits. 30 ms is above that and less than half a frame interval. A read
# therefore costs at most one wait plus this budget, and tripping it is
# counted in ``bounded`` and printed on the preview's numbers line.
DRAIN_BUDGET_S = 0.030
# How many buffer arrivals the delivered-rate estimate averages over. Long
# enough that one slow grab cannot move it, short enough to follow a device
# that changes its own rate -- his LifeCam's delivered rate is a function of
# the light (measured 2026-09-03) -- and 16 buffers is about a second of it.
DRAIN_RATE_WINDOW = 16
# WHAT IS NO LONGER HERE, and why. Until 2026-09-03 the drain decided a
# buffer was already queued by TIMING the grab that fetched it: under
# FRESH_GRAB_S = 2 ms meant "this one was waiting", over it meant "this one
# waited for the device". The dequeue cost was then measured at his camera
# with the device free, and 2 ms is EIGHT TIMES BELOW THE NOISE FLOOR of the
# process the drain runs in:
#
#   condition           n     p50        p90     p99     max        >= 2.0 ms
#   idle box           135    0.001 ms   0.035   0.036   0.037 ms     0.00%
#   1 busy Py thread   135    5.222 ms   5.460   5.545   5.605 ms    89.63%
#   4 busy Py threads  120    0.004 ms   0.118   5.357   15.987 ms    9.17%
#
# A real dequeue costs 2 MICROSECONDS. What the drain could time was the
# whole round trip, and ``grab()`` releases the GIL for the ioctl and must
# take it back to return -- so ONE competing Python thread turned 2 us into
# 5.2 ms and 89.63% of dequeues read as "this one waited for the device".
# The drain then stopped at the first queued buffer, handed back the OLDEST
# frame, and reported drop 0.0/s, which the numbers line says means healthy.
# Jarvis is never an idle box: Whisper, TTS, the bus, the preview worker,
# two room pollers and the hotword daemon all run. So the grab's duration is
# not a usable staleness signal in this process and nothing reads it as one
# any more; what is left of it is the frame-interval ESTIMATE
# (``DrainingCapture.interval_s``), which is a 67.8 ms quantity that the
# same measured noise cannot reach.
# V4L2's exposure_auto values, which cv2's V4L2 backend passes through RAW
# on CAP_PROP_AUTO_EXPOSURE (measured 2026-09-04 on his LifeCam: get() read
# 3.0 under auto, set(1) read back 1.0 and the v4l2 control agreed). Not
# the 0.25/0.75 folklore from other backends.
EXPOSURE_MANUAL = 1
EXPOSURE_AUTO = 3
# The driver controls that decide the camera's OWN frame period. Read and
# logged at every open (``exposure_probe``), because on 2026-09-03/04 the
# preview halved from 7.5 to 3.7 fps with nothing in the log able to say
# whether the camera had been re-metered. MEASURED 2026-09-04, grab() only,
# the app closed: the LifeCam runs a 30 / 15 / 7.5 fps ladder by exposure
# tier (<=15.6 ms / 31-62 ms / >=125 ms) and auto-exposure was sitting on
# the SLOWEST rung -- at midday, lights up, so not the room: WHAT it was
# metering on is unmeasured. Forcing manual exposure 156 took it from 3.75 to
# 15-16 fps in the app's own 1-buffer configuration, and back to auto put
# it straight back. Under auto the ``exposure`` figure is the CACHED manual
# value, not a meter reading; ``auto_exposure`` is the number that matters.
EXPOSURE_CONTROLS = ("auto_exposure", "exposure", "gain")


def _import_cv2():
    """cv2, imported HERE and nowhere at module scope, so this file loads on
    a box without OpenCV and ``build()`` can report that as a reason."""
    import cv2                       # noqa: PLC0415 - deliberately lazy
    return cv2


def exposure_probe(cap) -> dict:
    """The driver's exposure controls, as NUMBERS. Never raises, never
    changes anything; -1.0 is "the driver has no such control" (his LifeCam
    has no gain), which is data and is logged as such."""
    cv2 = _import_cv2()
    out = {}
    for name, prop in (("auto_exposure", "CAP_PROP_AUTO_EXPOSURE"),
                       ("exposure", "CAP_PROP_EXPOSURE"),
                       ("gain", "CAP_PROP_GAIN")):
        try:
            out[name] = float(cap.get(getattr(cv2, prop)))
        except Exception:  # noqa: BLE001 - an absent control is a number
            out[name] = -1.0
    return out


def pin_exposure(cap, exposure: int) -> dict:
    """Manual exposure, pinned: auto off, then the value. Reports what the
    driver READ BACK, because a refused set is silent on v4l2.

    THE ONE CAVEAT IS THE CAMERA'S OWN TABLE. The LifeCam accepts any value
    in 5..20000 but runs its fastest 30 fps sensor rate only at values on
    its discrete list (5, 9, 10, 19, 20, 39, 78, 156 measured; 312 and 625
    give 15; anything off the list -- 50, 100, 200, 400 -- falls to the
    SLOWEST tier). So the number to use is one the next preview line shows
    to be fast, not one that looks reasonable. 156 is the value the camera
    itself caches under auto and is the safe first choice.
    """
    cv2 = _import_cv2()
    out = {"asked": int(exposure)}
    try:
        out["auto_accepted"] = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                                            EXPOSURE_MANUAL))
    except Exception:  # noqa: BLE001
        out["auto_accepted"] = False
    try:
        out["accepted"] = bool(cap.set(cv2.CAP_PROP_EXPOSURE,
                                       int(exposure)))
    except Exception:  # noqa: BLE001
        out["accepted"] = False
    out.update(exposure_probe(cap))
    out["pinned"] = bool(out["auto_accepted"] and out["accepted"]
                         and out["auto_exposure"] == float(EXPOSURE_MANUAL)
                         and out["exposure"] == float(int(exposure)))
    return out



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
class DrainingCapture:
    """A capture whose ``read()`` throws the STALE buffers away first.

    THE PROBLEM, and it is the one CAP_PROP_BUFFERSIZE = 1 was reaching for.
    OpenCV's V4L2 backend queues four driver buffers. A consumer reading
    SLOWER than the device delivers therefore gets whatever has been sitting
    at the head of that queue -- up to three intervals old, 200 ms at the
    15.0 fps his LifeCam delivers, and lag he can see. Setting the queue to
    one buffer did stop that, and cost exactly half the frame rate (measured
    2026-09-03, eight rows of eight, a clean 2.00x; see CAPTURE_BUFFERS).
    So the queue is the driver's again and the stale frames are DISCARDED
    here instead: ``grab()`` dequeues without decoding, so dropping one is
    nearly free, and only the last buffer is ever ``retrieve()``d.

    HOW IT KNOWS WHICH ARE STALE, and the question it stopped asking.
    "Is THIS buffer stale" cannot be answered through OpenCV's surface: the
    only observable difference between a buffer that was already waiting and
    one the device has just made is how long ``grab()`` took, and this
    process cannot time a grab. The dequeue cost was measured at his camera
    on 2026-09-03 and the table sits beside ``DRAIN_BUDGET_S`` -- one busy
    Python thread turns a 2 us dequeue into a 5.2 ms round trip, because
    ``grab()`` drops the GIL for the ioctl and has to take it back to
    return. So the drain does not ask it any more.

    It asks a question arithmetic can answer instead: HOW MANY BUFFERS CAN
    BE WAITING? Two known quantities settle that.

    * The driver's queue is ``depth`` deep -- ``CAP_PROP_BUFFERSIZE``, read
      back at the open and passed in by ``open_capture`` -- so at most
      ``depth - 1`` buffers can sit behind the newest one.
    * The device produces one buffer per frame interval, so the number that
      arrived while the consumer was away is that absence divided by the
      interval. The absence is this object's own wall clock between its last
      grab and this one; the interval is ``interval_s`` below.

    ``queued`` is the ledger those two keep: arrivals added, every grab
    subtracted, clamped at ``depth`` because the driver overwrites what will
    not fit. A read takes ``int(queued)`` buffers -- always at least one,
    never more than ``depth - 1`` beyond it -- and retrieves the last. IT
    NEVER ASKS THE DEVICE FOR A BUFFER THE LEDGER DOES NOT SAY HAS ARRIVED,
    which is the whole of how it avoids waiting.

    THE FOUR WAYS TO GET THIS WRONG, and what stops each:

    1. IT MUST NOT BLOCK. A grab the ledger was wrong about waits a frame
       interval -- 67.8 ms at his LifeCam, measured -- and the budget is
       sized so that ONE such wait blows it: ``DRAIN_BUDGET_S`` is 30 ms
       against a full depth-1 drain of dequeues at ~16 ms. So a read blocks
       at most once, which is the wait an undrained read already pays on an
       empty queue, and it can never block twice. ``bounded`` counts it.
    2. IT MUST NOT DROP THE ONLY FRAME. The buffer of the LAST SUCCESSFUL
       grab is what gets retrieved, so a grab that fails or raises part-way
       through leaves the consumer with the newest frame that did arrive,
       never with nothing. Nothing is retrieved only when the FIRST grab
       fails, which is what a plain failed ``read()`` already meant.
    3. IT MUST HELP ONLY WHEN IT HELPS. A consumer keeping pace is never
       away for a whole interval, so the ledger says nothing is waiting and
       the read is one grab and one retrieve -- byte for byte the old path.
       ``tests/test_camera.py`` pins that at 15 fps against a 15 fps device:
       30 reads, 30 grabs, 0 dropped, every ordinal in sequence.
    4. IT MUST DEGRADE. A device with no ``grab``/``retrieve`` is read
       straight through. A grab or a retrieve that raises is (False, None),
       which every caller already reads as no opinion. And the ledger's own
       errors are bounded in both directions -- see the estimate below.

    THE INTERVAL ESTIMATE, AND BOTH WAYS IT CAN BE WRONG. ``interval_s`` is
    the mean spacing between the last 16 buffers the device handed over, or
    the driver's nominal interval before there are two. A RATE OVER MANY
    FRAMES, not a verdict on one grab: every buffer grabbed is a buffer the
    device produced, so buffers over wall clock is what it delivers, and a
    5.2 ms GIL round trip on one of sixteen moves it by a third of a
    percent. The nominal is safe as a seed because a driver never delivers
    FASTER than the rate it advertises (30 granted, 15.0 delivered here), so
    it can only be shorter than the truth.

    * TOO LONG -- the queue is backing up, so buffers are reaching the drain
      more slowly than the device makes them -- under-counts arrivals, so
      the drain takes fewer buffers and hands back an older frame. It costs
      no rate, and it is where the drain leaves lag (below).
    * TOO SHORT -- the seed, or a device that has just slowed down --
      over-counts arrivals, so the drain asks for a buffer that has not
      arrived and waits for it. The budget stops that read at the first such
      wait, and the wait itself is the measurement that lengthens the
      estimate.

    A MEAN OVER A WINDOW RATHER THAN THE LONGEST GRAB. Timing the grabs
    themselves and keeping the maximum is the obvious estimator and it is
    biased low, because a grab that waits does not wait a WHOLE interval --
    it waits the remainder of one. Measured against the modelled queue, that
    bias had the drain believing a 66.7 ms device ran at 55.6 ms, counting
    22% more arrivals than existed and reaching for a buffer that was not
    there on every other read: 9.8 fps where an undrained consumer got 12.5.

    WHAT THE DRAIN COSTS AND BUYS, MEASURED -- AND NOT AT HIS CAMERA.
    /dev/video0 was held by the running Jarvis (the house rule is to say so
    rather than take the lens off the assistant for a number), so the arms
    below drive the REAL ``DrainingCapture`` against a MODELLED V4L2 queue
    -- four buffers on the 66.7 ms interval his LifeCam was measured to
    deliver on 2026-09-03 -- with real time and real sleeps:

        scripts/camera_drain_probe.py --model 15 --seconds 4

        consumer                 delivered  discarded  device  driver lost
        as fast as it will, drained   15.0     0.0/s    15.0        0
        as fast as it will, undrained 15.0     0.0/s    15.0        0
        6 fps, drained                 6.0     8.0/s    14.0        0
        6 fps, undrained               6.0     0.0/s     6.0       30
        3 fps, drained                 3.0     8.2/s    11.2       10
        3 fps, undrained               3.0     0.0/s     3.0       40
        80 ms of work, drained        12.1     0.2/s    12.3        6
        80 ms of work, undrained      12.3     0.0/s    12.3        6
        300 ms of work, drained        3.3     7.7/s    11.0       12
        300 ms of work, undrained      3.3     0.0/s     3.3       42

    FULL RATE, which is the half the last version got wrong. The drained and
    undrained arms deliver the same frames a second at every consumer rate,
    including the two paced by WORK -- 80 and 300 ms a cycle, the case a
    sleep-paced arm cannot show because the sleep absorbs whatever the drain
    costs. The previous drain lost 38% of the frame rate at 80 ms of work
    -- 7.45 fps against 12.07 where the verification found it, 7.37 against
    12.48 reproduced here -- by waiting for a frame newer than the one in
    hand. This one never asks for a buffer the ledger has not counted.

    THE NEWEST FRAME. On the drained rows delivered + dropped climbs back
    towards the device's own 15 and the driver loses little or nothing, so
    the queue is being emptied and the frame retrieved is one that has just
    arrived; on the undrained rows the driver overwrote 30 and 40 buffers in
    four seconds, which is a permanently full queue and is the staleness
    itself. Which ordinal came back is pinned in the suite instead, since
    the probe never decodes: the same queue hands back the newest ordinal
    drained and the oldest undrained
    (``test_a_slow_consumer_is_handed_the_newest_frame_not_the_oldest``).

    WHERE IT STILL LEAVES LAG, and this is the honest edge. The ledger can
    only count the buffers it GRABS -- one the driver overwrote while the
    drain was taking too few is invisible to it -- so a drain that settles
    at one buffer a read has no way to discover that two were arriving. That
    happens in the MARGINAL band, a consumer a shade slower than the device:
    at 80 ms of work against 66.7 ms the drained arm above sits within 2% of
    the undrained rate and drops almost nothing, and the frame it hands back
    is measured at 1.7 intervals old rather than 2.5. Draining that band
    properly means speculatively waiting for a frame that may not exist,
    which is precisely what cost 38% of the frame rate, so it does not.

    NOT MEASURED: the dequeue cost and the frame interval here are his
    camera's, measured with the device free
    (scratch-0903/drain/MEASURED-dequeue.md), but this whole table is the
    modelled queue. When the camera is free the same script without
    ``--model`` measures the real device -- ``drain()`` below is the grab
    loop with no retrieve in it, so the probe runs the REAL policy against
    the REAL camera without a pixel reaching Python. Retrieve is not part of
    the comparison in any case: drained or not, a read retrieves once.
    """

    def __init__(self, cap, *, now: Callable[[], float] = time.perf_counter,
                 max_drops: int = DRAIN_MAX_DROPS,
                 budget_s: float = DRAIN_BUDGET_S,
                 depth: int = DRAIN_DEPTH,
                 nominal_fps: float = DRAIN_NOMINAL_FPS):
        self.raw = cap
        self._now = now
        self.max_drops = max(0, int(max_drops))
        self.budget_s = float(budget_s)
        # The driver's answer, not a belief about it. A backend that reports
        # nothing useful (-1, or 0) leaves the fallback in place rather than
        # a queue of no depth, which would switch the drain off silently.
        try:
            self.depth = max(1, int(depth))
        except (TypeError, ValueError):
            self.depth = DRAIN_DEPTH
        try:
            nominal = float(nominal_fps)
        except (TypeError, ValueError):
            nominal = DRAIN_NOMINAL_FPS
        if not (nominal > 0.0):
            nominal = DRAIN_NOMINAL_FPS
        self.nominal_s = 1.0 / max(1.0, nominal)
        # TWO predicates, not one. The drain loop needs only ``grab`` --
        # which is what lets scripts/camera_drain_probe.py time the real
        # policy against a device it may not decode from. Returning a FRAME
        # additionally needs ``retrieve``; without it there is nothing to
        # drain towards, so ``read`` falls straight through to the device's
        # own.
        self.can_grab = callable(getattr(cap, "grab", None))
        self.can_drain = self.can_grab and \
            callable(getattr(cap, "retrieve", None))
        self.dropped = 0        # stale buffers discarded, cumulative
        self.drains = 0         # reads that ran the loop at all
        self.bounded = 0        # reads that stopped short of the ledger
        self.failures = 0       # grabs that failed or raised
        # The LEDGER: how many buffers the drain believes are waiting in the
        # driver's queue right now. Fractional, because arrivals are wall
        # clock over a frame interval and a consumer's cadence has no reason
        # to be a whole number of frames.
        #
        # IT STARTS FULL, and that is the one place the drain guesses. The
        # device has been streaming since ``cv2.VideoCapture`` opened it and
        # nobody has taken a buffer yet, so the queue at the first read is
        # the one moment it is reliably backed up. The guess is free when it
        # is wrong: an empty queue makes the first grab wait, that wait
        # blows the budget on its own, and the loop stops with one grab --
        # exactly an undrained read. When it is right the queue is cleared
        # at the open instead of over the following minute. It is also the
        # only measurement the drain gets of a device it has not yet
        # watched: the grabs it makes here are what ``spacing_s`` divides.
        self.queued = float(self.depth)
        # WHEN THE LAST FEW BUFFERS WERE HANDED OVER. The mean spacing
        # between them is the device's own delivered interval whenever the
        # queue is being kept short, which is the only number the ledger
        # needs and the only one this process can measure honestly: it is a
        # rate over many frames, not a verdict on one grab.
        self._grabbed_at = deque(maxlen=DRAIN_RATE_WINDOW)

        # Reported, never acted on: the longest single grab yet seen. A
        # drain that is waiting for the device shows it here.
        self.longest_grab_ms = 0.0
        # The wall clock the ledger has already counted.
        self._seen_at = now()

    # ------------------------------------------------------- the two numbers
    @property
    def spacing_s(self) -> float:
        """The mean wall clock between the last few buffers the device
        handed over. 0.0 until there are two of them.

        THE RATE, NOT ONE GRAB'S DURATION. Every buffer grabbed is a buffer
        the device produced, so buffers over wall clock IS its delivered
        rate whenever the queue is not backing up -- and when it is backing
        up this reads LONGER than the truth, which makes the ledger count
        fewer arrivals and the drain take fewer buffers. Both errors are on
        the side of behaving like an undrained read.

        The window is short enough to follow a device that changes rate --
        his LifeCam halves its own in dim light -- and long enough that one
        slow grab cannot move it.
        """
        stamps = self._grabbed_at
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return span / (len(stamps) - 1) if span > 0.0 else 0.0

    @property
    def interval_s(self) -> float:
        """The device's frame interval as the drain currently believes it:
        the measured spacing, or the driver's nominal interval before there
        is one. Never shorter than the nominal, which a driver can only
        overstate."""
        return max(self.spacing_s, self.nominal_s)

    @property
    def draining(self) -> bool:
        """Is this object actually discarding anything? A capture with no
        ``retrieve`` under it, or one built with ``max_drops`` 0 -- the
        undrained arm the probe compares against -- is inert, and an inert
        drain reporting drop 0.0/s looks exactly like a healthy one."""
        return self.can_drain and self.max_drops > 0

    def _arrivals(self) -> None:
        """Advance the ledger to NOW: however long the drain has not been at
        the device, divided by the frame interval, is how many buffers the
        device queued in the meantime. Clamped at ``depth`` because the
        driver overwrites the oldest rather than growing the queue.

        The cursor moves with it, so every microsecond of wall clock is
        counted exactly once -- including the time inside a grab, where a
        device that made the drain wait produced a frame of its own.
        """
        now = self._now()
        gap = max(0.0, now - self._seen_at)
        self._seen_at = now
        self.queued = min(float(self.depth),
                          self.queued + gap / self.interval_s)

    # ------------------------------------------------------------ the drain
    def drain(self) -> dict:
        """Pop the stale buffers. NOTHING IS RETRIEVED OR DECODED HERE.

        Returns numbers: how many buffers were grabbed, how many of those
        were stale and discarded, the wall clock spent, what the ledger
        still says is waiting and whether a bound stopped it short.
        ``read`` retrieves the last one; the probe script does not, which is
        how the real policy can be measured at the real device without a
        pixel existing.
        """
        out = {"grabbed": 0, "dropped": 0, "spent_ms": 0.0,
               "longest_ms": 0.0, "bounded": False, "queued": 0.0}
        if not self.can_grab:
            return out
        self.drains += 1
        # ONE grab always, then as many as the ledger says are waiting.
        ceiling = 1 if self.max_drops <= 0 else \
            1 + min(self.max_drops, max(0, self.depth - 1))
        spent = 0.0
        longest = 0.0
        while True:
            self._arrivals()
            if out["grabbed"] >= 1:
                if self.max_drops <= 0:
                    # The undrained arm the probe compares against: the loop
                    # is OFF, not bounded, and counting a bound here would
                    # report a limit biting on every single cycle.
                    break
                if self.queued < 1.0:
                    # Nothing else is waiting. Asking anyway is what makes a
                    # read block, and blocking is what the drain must not
                    # cost a consumer that is only slow because of its own
                    # work.
                    break
                if out["grabbed"] >= ceiling:
                    # The queue's own depth, or the max_drops backstop above
                    # it. Buffers the ledger counted are still in there.
                    out["bounded"] = True
                    break
                if spent >= self.budget_s:
                    # THE LEDGER WAS WRONG: a full depth-1 drain of dequeues
                    # costs ~16 ms at the measured 5.2 ms each, so 30 ms can
                    # only have been spent WAITING for the device -- which
                    # means the queue is empty and the buffer in hand is the
                    # newest there is. The ledger is reset to say so.
                    out["bounded"] = True
                    self.queued = 0.0
                    break
            start = self._now()
            try:
                ok = bool(self.raw.grab())
            except Exception:  # noqa: BLE001 - a device edge is not a crash
                log.debug("camera: a grab raised inside the drain",
                          exc_info=True)
                ok = False
            took = max(0.0, self._now() - start)
            spent += took
            longest = max(longest, took)
            self.longest_grab_ms = max(self.longest_grab_ms, took * 1000.0)
            if not ok:
                self.failures += 1
                break
            out["grabbed"] += 1
            # Immediately, not at the end of the read: this buffer's arrival
            # is part of the rate the next turn of this loop divides by.
            self._grabbed_at.append(self._now())
            self._arrivals()            # the grab's own wall clock, then...
            self.queued = max(0.0, self.queued - 1.0)   # ...its own buffer
        if out["bounded"]:
            self.bounded += 1
        out["dropped"] = max(0, out["grabbed"] - 1)
        out["spent_ms"] = spent * 1000.0
        out["longest_ms"] = longest * 1000.0
        out["queued"] = round(self.queued, 2)
        self.dropped += out["dropped"]
        return out

    # ------------------------------------------------------------- the read
    def read(self):
        """``(ok, frame)`` -- cv2.VideoCapture's own contract, and the newest
        frame the device has produced rather than the oldest one queued."""
        if not self.can_drain:
            return self.raw.read()
        got = self.drain()
        if got["grabbed"] <= 0:
            return False, None
        try:
            return self.raw.retrieve()
        except Exception:  # noqa: BLE001 - a decode failure is no opinion
            log.debug("camera: a retrieve raised", exc_info=True)
            return False, None

    # ------------------------------------------------------- the passthrough
    def grab(self) -> bool:
        return bool(self.raw.grab())

    def retrieve(self, *args):
        return self.raw.retrieve(*args)

    def get(self, prop):
        return self.raw.get(prop)

    def set(self, prop, value):
        return self.raw.set(prop, value)

    def isOpened(self) -> bool:                        # noqa: N802 - cv2's
        return bool(self.raw.isOpened())

    def getBackendName(self) -> str:                   # noqa: N802 - cv2's
        return str(self.raw.getBackendName())

    def release(self) -> None:
        self.raw.release()

    def status(self) -> dict:
        """Numbers only, for a log line or a self-check."""
        return {"dropped": self.dropped, "drains": self.drains,
                "bounded": self.bounded, "failures": self.failures,
                "longest_grab_ms": round(self.longest_grab_ms, 2),
                "interval_ms": round(self.interval_s * 1000.0, 2),
                "queued": round(self.queued, 2), "depth": self.depth,
                "draining": self.draining}


def open_capture(device: str = "", width: int = 1280, height: int = 720,
                 fourcc: str = DEFAULT_FOURCC, exposure: int = 0):
    """cv2.VideoCapture, opened, asked for a mode, and the GRANTED mode read
    back and logged. Raises if it will not open -- ``Eye`` reads that as no
    opinion, which is the same behaviour as having no camera at all.

    THE EXPOSURE CONTROLS ARE LOGGED AT EVERY OPEN, beside the granted mode,
    since 2026-09-04. What the two probes of 09-03 left "NOT proven" (below)
    was then measured, grab() only and the app closed: the delivered rate is
    the camera's own auto-exposure tier. His LifeCam runs 30 / 15 / 7.5 fps
    at exposure <=15.6 ms / 31-62 ms / >=125 ms, auto was on the slowest
    rung (3.75 fps through the app's single driver buffer) for a reason
    nobody has measured -- ambient light is refuted, the scene is not -- and
    manual exposure 156 took the same open to 15-16 fps with no other change.
    So the line printed here says ``auto_exposure``, ``exposure`` and
    ``gain`` as the driver reports them -- and says plainly that under auto
    the exposure figure is a cached manual value, not a light reading, so
    nobody reads 156 as "the room is bright" again.

    ``exposure`` > 0 PINS manual exposure at that value (``pin_exposure``),
    which is ``camera.exposure`` in his config and ships at 0 = leave auto
    alone. It is a lever, not a verdict: it trades the camera's own metering
    for a fixed sensor rate, and whether the pane and the detector are still
    usable at that exposure in his evening light is his to read off the
    next ``campreview:`` line, not this module's to assume.

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

    ``CAP_PROP_BUFFERSIZE`` IS NEVER WRITTEN, and the history of that number
    is on CAPTURE_BUFFERS: setting it to one halves the probe's rate (a
    measured 2.00x on eight rows of eight), did not change the app's, and
    was what kept the picture he sees fresh. The staleness it fought is a
    frame up to three intervals old handed to a consumer that runs slower
    than the device. The old ``6.0 fps  grab 11 ms`` line at 6 requested is
    CONSISTENT with that -- an 11 ms grab from a 133 ms device is most
    plausibly a frame that was already waiting -- but frame age was never
    timed at the device, and 9ba1c56's "boxes on random objects" with the
    driver's buffers is the nearest thing to a measurement of it. Expected
    mechanism, measured only against a modelled queue.

    THE PROPER FIX IS TO DRAIN, NOT TO STARVE: keep the driver's buffers and
    discard the stale ones before retrieving, so a slow consumer still gets
    the newest frame at full rate. THAT IS WHAT IS RETURNED HERE. The handle
    this hands back is a ``DrainingCapture`` wrapping the VideoCapture, not
    the VideoCapture itself -- it answers the same ``read``/``grab``/
    ``retrieve``/``get``/``set``/``release`` surface, so ``capture_mode``,
    ``focus_probe``, scripts/vision_selfcheck.py and scripts/face_enrol.py
    are unaffected, and its ``read`` returns the newest queued frame rather
    than the oldest. What is still owed is a measurement AT THE DEVICE with
    Jarvis stopped: scripts/camera_drain_probe.py, no --model.

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
    # THE DRAIN'S TWO INPUTS COME FROM THE DRIVER, not from a constant: how
    # deep its queue is (so the drain knows how many buffers can be stale)
    # and its nominal rate (the seed for the frame interval, which is only
    # ever a lower bound on the real one). A driver that answers -1 or 0 --
    # or a read-back that raises -- leaves the module fallbacks in place.
    depth, nominal = DRAIN_DEPTH, DRAIN_NOMINAL_FPS
    try:
        got = capture_mode(cap)
        log.info("camera: asked %dx%d %s; granted %.0fx%.0f %s at %.1f fps "
                 "nominal, %.0f driver buffer(s) -- the delivered rate is "
                 "what the preview's own line reports",
                 int(width), int(height), fourcc or "-", got["width"],
                 got["height"], got["fourcc"] or "?", got["fps"],
                 got["buffersize"])
        if got["buffersize"] >= 1:
            depth = int(got["buffersize"])
        if got["fps"] > 0:
            nominal = float(got["fps"])
    except Exception:  # noqa: BLE001 - a read-back is not worth a crash
        log.debug("camera: could not read the granted mode back",
                  exc_info=True)
    try:
        if int(exposure or 0) > 0:
            pin = pin_exposure(cap, int(exposure))
            log.info("camera: exposure pinned manual %d -> %s; driver reads "
                     "back auto_exposure %.0f  exposure %.0f  gain %.0f",
                     pin["asked"], "accepted" if pin["pinned"] else "REFUSED",
                     pin["auto_exposure"], pin["exposure"], pin["gain"])
        else:
            ctl = exposure_probe(cap)
            log.info("camera: controls at open: auto_exposure %.0f  exposure "
                     "%.0f  gain %.0f (%s; -1 = no such control; under auto "
                     "the exposure figure is the cached manual value, not a "
                     "light reading -- the camera's own rate tier shows in "
                     "the preview line)",
                     ctl["auto_exposure"], ctl["exposure"], ctl["gain"],
                     "auto" if ctl["auto_exposure"] == float(EXPOSURE_AUTO)
                     else "manual" if ctl["auto_exposure"]
                     == float(EXPOSURE_MANUAL) else "mode ?")
    except Exception:  # noqa: BLE001 - a control read is not worth a crash
        log.debug("camera: could not read the exposure controls",
                  exc_info=True)
    return DrainingCapture(cap, depth=depth, nominal_fps=nominal)


class _GatedDevice:
    """The handle ``Eye`` holds: a device whose close goes through the gate.

    ``read`` delegates. ``release`` calls the GATE's release rather than the
    device's, so a close initiated by ``Eye`` and a close initiated by
    ``SensingPolicy.enforce`` leave the two objects agreeing about who holds
    what. A wrapper made before a close is stale afterwards and reads False,
    so a handle released by the curfew cannot be read from by a loop that
    has not noticed yet.
    """

    __slots__ = ("_feed", "_raw", "_epoch", "_seen_drops")

    def __init__(self, feed: "CameraFeed", raw, epoch: int):
        self._feed = feed
        self._raw = raw
        self._epoch = epoch
        # What this wrapper has already reported of the device's own
        # cumulative drop count. Per-wrapper, so a re-open starts a fresh
        # device at a fresh baseline and cannot double-count.
        self._seen_drops = 0

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
        #
        # The DRAIN runs inside this same lock, because it is part of the
        # read: a curfew edge that landed between two grabs would otherwise
        # release the device out from under the loop.
        with self._feed.read_lock:
            if self._epoch != self._feed.epoch:
                return False, None
            out = self._raw.read()
            self._carry_drops()
            return out

    def _carry_drops(self) -> None:
        """Move the device's own drop count up to the feed, as a delta.

        A device that does not drain (every fake in the suite, any backend
        without grab/retrieve) has no such attribute and this is a no-op --
        the count degrades to 0 rather than to an AttributeError inside a
        read.
        """
        try:
            total = int(getattr(self._raw, "dropped", 0) or 0)
        except (TypeError, ValueError):    # a device with a strange attr
            return
        if total > self._seen_drops:
            self._feed.stale_dropped += total - self._seen_drops
            self._seen_drops = total
        self._carry_drain_state()

    def _carry_drain_state(self) -> None:
        """And the drain's own state, which a count alone cannot show.

        A drain that is doing NOTHING reports drop 0.0/s, and so does a
        drain with nothing to do. Until 2026-09-03 those two were
        indistinguishable from outside the object, which is how a drain
        that had switched itself off went unnoticed. ``draining``, the
        delivered interval it measured and the times a bound stopped it
        short all come up here and end on the preview's numbers line.

        A device with no drain under it leaves the feed's dict empty, which
        the preview reads as "no drain", not as an error.
        """
        raw = self._raw
        if not hasattr(raw, "interval_s"):
            return
        try:
            self._feed.drain = {
                "on": bool(getattr(raw, "draining", False)),
                "interval_ms": round(float(raw.interval_s) * 1000.0, 2),
                "longest_ms": round(float(raw.longest_grab_ms), 2),
                "bounded": int(getattr(raw, "bounded", 0)),
                "queued": round(float(getattr(raw, "queued", 0.0)), 2),
                "depth": int(getattr(raw, "depth", 0)),
            }
        except (TypeError, ValueError):        # a device with strange attrs
            return

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
        # STALE frames the drain discarded on the way to the newest one --
        # cumulative, carried up from the device by ``_GatedDevice``. This
        # is what the preview's numbers line reports per second, so a drain
        # quietly eating half the stream is a grep rather than a night of
        # guessing. It is a plain int deliberately: there is exactly one
        # capture handle and one consumer by design, so there is no second
        # writer to race with.
        self.stale_dropped = 0
        # The drain's own state, replaced wholesale on each read by
        # ``_GatedDevice._carry_drain_state``. Empty means NO DRAIN UNDER
        # THIS FEED -- every stub in the suite, any backend without
        # grab/retrieve -- which the preview prints as "drain off" rather
        # than as a healthy zero.
        self.drain: dict = {}
        # WHO IS USING THE DEVICE RIGHT NOW, by name. Two readers of one
        # v4l2 node is safe from corruption already (``_GatedDevice.read``
        # holds ``read_lock`` across the grab and ``CameraGate.open`` is
        # locked and idempotent) but it is NOT safe from a close: a
        # ``close()`` from one reader bumps the gate epoch and invalidates
        # the other's in-flight handle. So a holder releases its claim and
        # only the LAST one out closes the device.
        self._holders: set = set()

    # ------------------------------------------------------------ holders
    def hold(self, name: str) -> None:
        """Claim the device for ``name``. Idempotent."""
        with self._lock:
            self._holders.add(str(name))

    def drop(self, name: str, close: bool = True) -> bool:
        """Release ``name``'s claim; close only when it was the last one.

        ``close=False`` gives the claim up WITHOUT closing -- for the
        caller that has just noticed somebody else started capturing and
        would rather leave the lens to them than bump the epoch under it.
        Returns True when the device was actually closed here.
        """
        with self._lock:
            self._holders.discard(str(name))
            last = not self._holders
        if last and close:
            # Outside the lock: close() reaches the gate, which reaches the
            # policy, and the one lock order that exists is gate -> read.
            self.close()
            return True
        return False

    @property
    def holders(self) -> tuple:
        with self._lock:
            return tuple(sorted(self._holders))

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
                   "stale_dropped": self.stale_dropped,
                   "drain": dict(self.drain) or {"on": False},
                   "holders": len(self._holders),
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
        # 0 = auto-exposure, the camera's own metering. See open_capture.
        exposure = int(_cfg_get(cfg, "camera.exposure", 0) or 0)
        if opener is None:
            def opener():                       # noqa: E306 - one call site
                return open_capture(device, lens.width_px, lens.height_px,
                                    fourcc, exposure=exposure)
        feed = CameraFeed(policy, opener, lens=lens, on_blind=on_blind,
                          present=lambda: device_present(device),
                          device=device)
        return feed, ""
    except Exception as exc:  # noqa: BLE001 - a camera must not end the app
        log.warning("camera: not wired (%s: %s)", type(exc).__name__, exc)
        return None, "%s: %s" % (type(exc).__name__, exc)
