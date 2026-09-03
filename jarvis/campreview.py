"""The camera preview's capture side: frames for HIS screen, and nothing else.

Hunter, 2026-09-02: *"lets add a small camera with visable tracking on the
jarvis app but make me be able to turn if off in settings"*.

THIS IS THE ONE PLACE IN JARVIS WHERE A FRAME BECOMES SOMETHING TO LOOK AT,
so it is the place where his standing rule -- *"i dont want you to look at
anything the camera sees without my explicit permission"* -- has to be
mechanically true rather than merely intended. The rule is not relaxed by
this feature; the preview is rendered on his monitor, for his eyes, and the
pixels never become reachable by anyone else:

* **No frame is written to disk.** Not a debug dump, not a cache, not a
  crash artefact. There is no imwrite, no PIL ``save``, no tempfile in this
  module or in ``jarvis/ui/preview.py``, and ``tests/test_campreview.py``
  greps both files for the names to keep it that way.
* **No frame leaves the process.** The only consumer of ``PreviewShot.image``
  is the Tk canvas two modules over. Nothing is published on the bus, logged,
  or handed to a provider -- ``PreviewShot`` carries the image in an
  attribute that ``numbers_only()`` deliberately excludes, and every log line
  and every status dict here is scalars and strings.
* **The frame is reduced before it is handed over.** What crosses to the Tk
  thread is already scaled down to the pane's own size (~160x90 design px);
  the full-resolution capture is dropped inside ``grab()`` and never held on
  an attribute. A stale full frame sitting in RAM after "offline mode" is the
  thing ``jarvis/eye.py`` closes the device to prevent.
* **The tests never look either.** They drive geometry, state and timing
  against fake frames this process generates. Nothing in the suite asserts on
  pixel content, because a test that had to look at an image would be a test
  that made looking at an image normal.

WHY THE CAPTURE IS OFF THE Tk THREAD, WHICH IS THE WHOLE ENGINEERING PROBLEM.
A grab on the Tk thread is the device's frame interval spent inside the
mainloop -- 133 ms at the 7.5 fps his LifeCam delivered with him at the
desk, 268 ms at the 3.8 fps it delivered to the probe (both MEASURED,
below) --
and the console's reactor animates on 16.67 ms slot boundaries
(jarvis/ui/avatar_clock.py). A read there would stop the console dead for
eight to sixteen slots, every picture. So: a daemon thread, a latest-wins
slot, and the two never block each other. The arithmetic per picture is
small next to the wait. Measured on this box 2026-09-03 against SYNTHETIC
1280x720 frames (no device opened): MJPG decode inside ``VideoCapture.read``
2.7-3.2 ms, the reduction to the pane's box 1.8 ms, YuNet at 320x180
1.4-2.6 ms, one SFace embedding 10.4 ms. Those component costs reproduce. An
earlier draft of this docstring also carried a table of whole-pipeline
CPU-per-second totals; they did not reproduce (off by 45% and more, and
priced at a picture rate the device does not deliver), so they are gone.

WHAT THE DEVICE ACTUALLY DELIVERS, MEASURED -- and why the number this
module asks for is a CEILING, not a rate. An earlier draft said the 7.5 fps
in his log was an artefact of asking for 1920x1080. IT WAS NOT. At the
corrected 1280x720 his live log (2026-09-03, 00:04-00:22) reads ``6.0 fps
grab 11 ms`` with 6 requested, ``7.5 fps  grab 133 ms`` with 10, and
``7.4-7.6 fps`` with 15: the device hands over ~7.5 fps whatever is asked.
(The 11 ms grab at 6 requested was not a fast device -- it was a frame that
had been waiting in the driver's queue, a stale picture; see
``CAP_PROP_BUFFERSIZE`` in jarvis/camera.py.) Two explanations were then put
to the device itself by scripts/camera_mode_probe.py, TWICE -- 2026-09-03
at 02:36 and again at 07:17, Jarvis stopped both times, ``grab()`` only:
buffers dequeued and never retrieved, decoded, shown or saved; 10 warm-up +
60 timed grabs per row, 30 fps requested for every row, cv2 4.12.0 on the
V4L2 backend. The two runs agree to 0.1 fps and 1 ms; the 07:17 run is the
one tabled, with the 02:36 sustained rate beside it:

  requested       set order     granted                 sustained 07:17  02:36
  1280x720 MJPG   fourcc-first  1280x720 MJPG 30.0 fps  3.8 fps p50 268   3.7
  1280x720 MJPG   size-first    1280x720 MJPG 30.0 fps  3.9 fps p50 268   3.9
  1280x720 YUYV   fourcc-first  1280x720 YUYV 10.0 fps  5.0 fps p50 200   5.0
  1280x720 YUYV   size-first    1280x720 YUYV 10.0 fps  5.0 fps p50 200   5.0
  640x480  MJPG   fourcc-first  640x480  MJPG 30.0 fps  3.9 fps p50 268   3.7
  640x480  MJPG   size-first    640x480  MJPG 30.0 fps  3.8 fps p50 268   3.7
  640x480  YUYV   fourcc-first  640x480  YUYV 30.0 fps  3.9 fps p50 267   3.9
  640x480  YUYV   size-first    640x480  YUYV 30.0 fps  3.9 fps p50 268   3.9

  (p50 in ms; max within 3 ms of p50 on every row. auto_exposure 3.0 and
   exposure 156 read back both times, not changed; CAP_PROP_BUFFERSIZE was
   4, set(1) accepted and read back 1, both times.)

So MJPG IS granted at 720p, in either set order, at a nominal 30: the format
is not the cause, and there is no FOURCC-ordering fix to apply. Bandwidth
is not the cause either: 640x480 delivers the same 3.8-3.9 fps in both
formats, and a 640x480 YUYV stream at that rate is a tenth of what USB 2.0
carries -- so there is no resolution change worth making, and none was
made. Every 30 fps mode delivered the same fraction of its nominal rate
(30/8, a 268 ms interval) and the 10 fps mode delivered 10/2, where his
live log, with him at the desk between 00:04 and 00:22, shows the same
720p MJPG mode delivering 30/4 (7.4-7.6 fps, grab 132-136 ms). Frame
intervals that are whole multiples of the mode's own are what a UVC
camera on auto-exposure produces when it lengthens the interval to expose
a dim scene ("exposure priority"), and nothing else that was measured
explains them. It is NOT proven: the two probe runs were at 02:36 and
07:17 and nobody looked at whether the light differed between them, so
they neither confirm nor refute it. Proving it means a run with the desk
lamp on, or with the camera's exposure-priority control off, and both are
his to do. What follows from it is the rule this module keeps: THE
PICTURE RATE IS SET BY THE LENS AND THE LIGHT, AND ``camera.preview_fps``
CAN ONLY LOWER IT.

THE THREE RATES ARE NOT ONE RATE, and separating them is what keeps the pane
cheap whatever the device delivers. Hunter, 2026-09-03: *"looks good but it
lags a ton"*. Detection does not need to run on every picture, and identity
must not:

* PICTURES -- ``camera.preview_fps``, default ``DEFAULT_FPS`` = 7.5, capped at
  ``MAX_FPS`` = 30. 7.5 is the HIGHEST rate the 1280x720 mode has been
  measured to deliver -- with him at the desk (his log, above); the probe
  measured 3.8-3.9 at 02:36 and 07:17 -- and a default is a ceiling, so it
  is set at the best the mode does rather than the worst: a ceiling under
  the delivered rate throws pictures away in exactly the light where the
  pane could be smooth. 30 is the mode's granted nominal, which the device
  has never been measured to reach. Above the DELIVERED rate --
  whatever it is tonight -- there is nothing more to fetch and the capture
  thread is simply parked inside a blocking read: measured 2026-09-03, the
  loop's sleep is 100% of the period at 6 requested against 7.5 delivered,
  1.1% at 10, 0.5% at 15, 0.3% at 30. That costs no CPU (a blocked read is
  not a spin) and it is not a privacy problem (``stop()`` is synchronous and
  a frame that returns after it is dropped), but it does mean a request
  above the delivered rate is a request the device ignores. Below it,
  pictures are thrown away.
* BOXES -- ``DETECT_FPS`` = 8, a constant rather than a knob, applied as
  EVERY Nth PICTURE with ``N = max(1, round(picture_fps / DETECT_FPS))``
  against the MEASURED picture interval (``detect_stride``). Not a
  wall-clock period: a detection can only happen when a picture arrives, so
  a bare ``>= 125 ms`` test is inert at 7.5 delivered fps (every picture,
  133 ms apart) and at 10 fps quantises to every SECOND picture -- 5 Hz
  boxes, slower than the pane managed before any of this. 8 Hz is what the
  assistant's OWN vision lane runs at (``camera.armed_fps``), so the overlay
  is never staler than what Jarvis is deciding on. Between detections the
  last boxes are CARRIED FORWARD rather than blanked: a smooth picture with
  a box an eighth of a second behind reads better than a choppy one with a
  perfectly fresh box.
* NAMES -- ``IDENT_FPS`` = 2. An embedding is 10.4 ms, five to seven times a
  detection. A person does not become a different person between frames, and
  the hysteresis below needs two agreeing readings anyway, so a name settles
  in about a second.

SENSING IS CONSULTED BEFORE THE DEVICE, NOT AFTER THE FRAME. ``jarvis/
sensing.py`` is the single owner of "may this sensor run" -- offline mode, the
nightly camera curfew, and the fail-to-offline rule. This module asks it at
the top of every cycle and, when the answer is no, does not call the pipeline
at all and lets go of the device it was holding -- closing it when the preview
opened it, dropping the reference when the feed is the app's (see
``PreviewPipeline.close``). That is belt-and-braces on top of ``Eye``'s own two
checks per frame, and it is what makes the pane's stated reason trustworthy:
the words "camera off (curfew until 7 am)" are printed by the same branch that
skipped the capture.

A BLACK RECTANGLE IS NOT AN ANSWER. Every way this can decline names itself
in ``PreviewShot.reason`` -- the toggle, the sensing owner, a missing vision
pipeline, a camera that is not plugged in -- because "off because you said so"
and "broken" look identical on screen and need opposite responses from him.

WHO IT IS LOOKING AT, AND THE ONE RULE THAT CANNOT BEND. Hunter, 2026-09-03:
*"lets have the identity of the person its tracking next to their name, small
but readable"*. So the tracked face carries a name, and four things constrain
how:

* **IDENTITY IS COMPUTED ONLY FROM A DETECTION THAT ALREADY CLEARED THE
  DETECTOR BAR.** SFace scores confidently on things that are not faces --
  unrelated non-face crops match each other at cosine 0.66-0.92, well over
  the 0.363 "same person" bar -- so the embedding is not a second opinion on
  whether this is a face. ``SFaceRecogniser.embed`` refuses a row under
  ``camera.min_conf`` and this module refuses to ask it. Same rule, stated
  twice, because it is the one that turns a smudge into a name.
* **"UNKNOWN" IS AN ANSWER AND BLANK IS NOT.** A face whose best cosine is
  under ``camera.identity_min`` reads as UNKNOWN with its score, never as
  the last name seen and never as nothing at all. No chip at all means
  something different and checkable: identity is not running (the switch is
  off, the weights are missing, or the gallery is empty).
* **IT MAY NOT FLICKER.** A per-frame verdict that alternates hunter/unknown
  is unreadable, so ``IdentityHold`` holds it the way ``AttentionTracker``
  holds the cone: a NAME needs two agreeing readings to take the screen, and
  the verdict survives ``IDENT_HOLD_S`` without a confirmation before it
  goes. The asymmetry is deliberate -- the first honest "unknown" is shown at
  once, because being slow to say "I do not know who this is" is the failure
  that matters.
* **IDENTITY MAY REMOVE CAPABILITY OR ADD A NAME. IT MAY NEVER GRANT ONE.**
  Nothing outside ``jarvis/ui/`` imports this module -- pinned by
  tests/test_campreview.py -- so a recognised face here cannot reach the wake
  gate, the commander or the bus. The name is pixels on his monitor and a
  string in a log line, and there is no seam by which it could become
  permission. ``jarvis/eye.py``'s ``resolve_wake`` reads its own identity
  from its own lane and is untouched by this file.

THERE IS DELIBERATELY NO VOICE PHRASE FOR THIS, YET. He asked for a settings
control and that is what the drawer's Privacy section has. A spoken switch
would have to join the commander's regex families, and the 09-01 review of the
"ui look" family found the softer openers ("give me classic", "use classic")
matching a noun-less path -- an overheard sentence in an open listening window
would have rewritten the config. The same trap sits under "show me the camera"
and "turn the camera on", which are ordinary English about a lens rather than
an instruction to this pane, and getting it wrong points a camera. It is a
half-hour of regex plus its tests whenever he wants it; it is not free, so it
is not in this change.

WHERE THE FRAMES COME FROM, AND WHY THIS MODULE DOES NOT OPEN A DEVICE.
``jarvis/camera.py`` (branch vision-bringup) already joins ``CameraGate`` and
``Eye`` into the one gated feed, and ``jarvis/visionrig.py`` already turns a
YuNet row into geometry. A second ``cv2.VideoCapture`` in the process would be
a second path that offline mode does not reach, so there is not one here:
``resolve_feed`` prefers the feed the app hands over (``Services.camera_feed``)
and otherwise asks ``jarvis.camera.build`` for one, importing it lazily so
this module loads on a tree where that lane has not landed yet.

That preference is not a style choice. ``SensingPolicy.attach`` REPLACES a
device by name, so if the app ever builds its own camera feed and the preview
built a second one, the second would silently displace the first and the
curfew would stop reaching the app's lens. Wiring ``camera_feed`` is therefore
how the two lanes stay one lens; the fallback exists only because nothing
attaches a camera on this tree today (verified) and logs loudly when it fires.
It also means the app's feed is BORROWED, not owned: closing it here would
bump the gate epoch and release a device another consumer is reading, on every
ambient transition and every curfew edge, so ``PreviewPipeline.close`` drops a
borrowed feed instead of shutting it.

THE THREAD'S LIFETIME IS ITS OWN, NOT A SHARED FLAG. Each ``start()`` mints a
fresh stop event and hands it to that run; ``stop()`` sets it and never clears
it. A single shared event would be un-set by the next ``start()`` -- reviving
a thread whose bounded join had timed out, so two of them would drive one
device forever -- and a run that outlived its own stop could publish a frame
grabbed after the switch went off. Both are closed here: the generation's
event gates its own loop, its own capture and its own writes to the slot, and
a capture lease keeps a wedged thread and its successor from being inside the
device at the same time.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from jarvis.logs import get_logger

log = get_logger("campreview")

# assistant.json keys. The toggle defaults OFF: a camera pane that appeared
# by itself on a box whose owner asked for camera silence would be the
# feature introducing itself by breaking the rule it lives under.
OPTION_ENABLED = "camera.preview"
OPTION_FPS = "camera.preview_fps"
OPTION_IDENTITY_MIN = "camera.identity_min"
OPTION_MIN_CONF = "camera.min_conf"
# The HIGHEST rate the 1280x720 mode has been measured to deliver -- his
# live log 2026-09-03 00:06-00:22, him at the desk: 7.4-7.6 fps whether 10
# or 15 was requested. Not chosen. The request is a ceiling the device may
# not reach: the same mode delivered 3.8-3.9 fps to scripts/camera_mode_probe
# at 02:36 and again at 07:17, and asking for more than it delivers buys
# nothing but a thread parked in a blocking read -- while asking for LESS
# than it delivers throws pictures away, which is why the default is the
# best measured rate and not the worst. See the module docstring; his
# config's own value wins.
DEFAULT_FPS = 7.5
# 30 is the GRANTED nominal of the 1280x720 MJPG mode (read back by the
# probe, either set order, both runs). The parked-in-read threshold is not
# this number but the DELIVERED rate -- 3.8 to 7.6 fps measured, and the
# device has never been measured at 30. Below 1 the pane stops reading as
# live.
MIN_FPS, MAX_FPS = 1.0, 30.0
# How often the DETECTOR runs, whatever the picture rate is. 8 Hz is what the
# assistant's own vision lane runs at (camera.armed_fps), so the boxes on the
# pane are never staler than what Jarvis is deciding on; between detections
# the last boxes are carried forward. Applied as every Nth picture against
# the MEASURED picture interval (detect_stride), so at a delivered 7.5 fps it
# is every picture and at 30 it is every fourth.
DETECT_FPS = 8.0
# How often IDENTITY runs. An embedding is 10.4 ms measured (2 threads,
# 2026-09-03), against 1.4-2.6 ms for a detection, so this is the one cadence
# that would dominate the whole module if it were left at the picture rate.
IDENT_FPS = 2.0
# A NAME needs this many agreeing readings before it takes the screen; an
# honest "unknown" is shown on the first one. At IDENT_FPS that is about a
# second to name someone and half a second to stop claiming to know them.
IDENT_AGREE = 2
# How long a held verdict survives with nothing confirming it -- three
# identity periods. Beyond that the chip goes rather than sitting there over
# a face nobody has checked. A face LEAVING clears it immediately.
IDENT_HOLD_S = 1.5
# OpenCV's own documented SFace cosine for "same person"; the default for
# camera.identity_min, and the same number jarvis/facegallery.py starts from.
DEFAULT_IDENTITY_MIN = 0.363
# The detector bar an embedding may not be taken under. Mirrors
# camera.min_conf's default; the recogniser carries its own copy and raises,
# and this one keeps us from paying for the exception.
DEFAULT_MIN_CONF = 0.6
# How much two boxes have to overlap before the held name is allowed to
# follow from one to the other. A HELD VERDICT BELONGS TO A FACE, NOT TO THE
# PANE: without this the label lands on whichever face is currently largest,
# so two people at similar distance with jittering box sizes get each other's
# names -- measured 70 of 142 chip-bearing frames wrong, and never
# converging (synthetic two-face harness, 2026-09-03, re-run 07:20 with
# the same counts). 0.3 is well above where a head at a desk drifts between
# two identity ticks (a still head holds >0.9 across 500 ms) and well below
# where two separate faces could land: side-by-side boxes score 0, and a
# stranger stepping in front at twice the linear size scores at most 0.25
# even perfectly concentric. Losing the chip for one identity period is the
# safe direction -- a missing name is a question, a wrong one is an answer.
SUBJECT_IOU = 0.3
# Boxes the pane can draw. Three is a glance, not a dashboard, and the
# fourth face at his desk is a poster. The cap is applied AFTER the size
# sort -- see PreviewPipeline._faces for why the order is the whole point.
MAX_FACES = 3
# A WORK bound on the reduction, not a draw cap: every row is reduced before
# anything is sorted, so a detector with a runaway top_k cannot cost the
# capture thread a visible amount of arithmetic. YuNet returns its rows
# score-descending and a real room yields single digits, so dropping the
# 65th-most-confident detection cannot lose the face at the desk.
MAX_ROWS = 64


def detect_stride(picture_fps: float, detect_fps: float = DETECT_FPS) -> int:
    """Run the detector on every Nth picture: ``max(1, round(picture_fps /
    detect_fps))``.

    A detection can only happen when a picture arrives, so the achievable
    box rates are the picture rate divided by a whole number. A wall-clock
    period picked the first multiple AT OR ABOVE it: inert at the 7.5 fps
    the device delivers (every picture -- 133 ms is already over the 125 ms
    period) and wrong at 10 (every SECOND picture, 5 Hz boxes). Rounding to
    the nearest multiple means the saving can never cost more than it
    saves: 7.5 -> 1, 10 -> 1, 15 -> 2, 30 -> 4. An unknown rate (the first
    picture, a stalled feed) is 1 -- detect, and learn the interval.
    """
    if not picture_fps or picture_fps <= 0.0 or detect_fps <= 0.0:
        return 1
    return max(1, int(round(picture_fps / detect_fps)))

# Why there is no picture. One of these, or "" while a frame is live.
REASON_LIVE = ""
REASON_DISABLED = "disabled"     # camera.preview is off (his switch)
REASON_SENSING = "sensing"       # offline / curfew / failsafe -- detail says
REASON_PIPELINE = "pipeline"     # no vision lane, no weights, camera.enabled
REASON_NO_FRAME = "noframe"      # the device declined: unplugged, busy, dropped
REASON_WAITING = "waiting"       # started, first frame not back yet

# What the pane says for each, when the reason has no detail of its own.
REASON_WORDS = {
    REASON_DISABLED: "preview off",
    REASON_SENSING: "sensing off",
    REASON_PIPELINE: "camera not wired",
    REASON_NO_FRAME: "no frame from the camera",
    REASON_WAITING: "waking the camera…",
}


# --------------------------------------------------------------- records
@dataclass(frozen=True)
class PreviewFace:
    """One detection, as NUMBERS -- the same discipline jarvis/visionrig.py
    holds its report to. There is no crop, no landmark array and no
    embedding on this object; what the overlay draws is a rectangle and
    three readouts, and that is all it is given.

    ``x/y/w/h`` are CAPTURE pixels (the detector's rows scaled back up), so
    the pane can map them into its own box without knowing the detect size.
    """

    conf: float
    x: float
    y: float
    w: float
    h: float
    yaw_deg: float = 0.0
    attending: bool = False
    landmarks_ok: bool = True
    # Interocular distance in CAPTURE px -- the one face scalar the hand
    # stage needs (gesture.observe_hand's reach ratio is palm_diag / this),
    # carried here so the stage reads it off the SAME frame's face rather
    # than re-deriving it from a box width. 0.0 when the landmarks failed.
    eye_px: float = 0.0
    # WHO, on the tracked face only -- the same one-subject rule ``attending``
    # follows, and for the same reason: one embedding per identity tick.
    #
    # ``id_ran`` is the field that carries the difference between the two
    # silences. False means identity was NOT asked (the switch is off, the
    # weights are missing, the gallery is empty, this is not the subject) and
    # the pane draws no chip. True with an empty ``name`` means it WAS asked
    # and nothing in the gallery matched -- an answer, drawn as UNKNOWN. A
    # single boolean is the whole distinction between "not looking" and "do
    # not recognise you", and a pane that showed them the same way would be
    # unreadable exactly when he needs to read it.
    name: str = ""
    id_score: float = 0.0
    id_ran: bool = False

    def as_dict(self) -> dict:
        return {"conf": self.conf, "x": self.x, "y": self.y, "w": self.w,
                "h": self.h, "yaw_deg": self.yaw_deg,
                "attending": self.attending,
                "landmarks_ok": self.landmarks_ok, "eye_px": self.eye_px,
                "name": self.name, "id_score": self.id_score,
                "id_ran": self.id_ran}


@dataclass(frozen=True)
class PreviewShot:
    """What crosses from the capture thread to the Tk thread.

    ``image`` is a PIL Image ALREADY SCALED to the pane's box -- in memory,
    for one canvas, for one monitor. It is the only pixel-bearing field in
    this module and it is excluded from ``numbers_only()`` by name, so a
    status line or a log format string cannot pick it up by accident.
    """

    image: Any = None                # PIL.Image or None; never saved, never sent
    faces: tuple = ()
    cap_w: int = 0                   # the capture geometry the boxes are in
    cap_h: int = 0
    reason: str = REASON_WAITING
    detail: str = ""                 # the sentence the pane prints
    seq: int = 0
    at: float = 0.0
    fps: float = 0.0                 # measured delivery rate, not the target
    grab_ms: float = 0.0             # p50-ish cost of the last cycle
    # The hand stage's verdict for this frame (jarvis/handstage.HandShot),
    # SCALARS ONLY, or None for no opinion -- the stage is off, has no
    # models, or raised. Never a landmark array, never a crop.
    hand: Any = None

    @property
    def live(self) -> bool:
        return self.image is not None and self.reason == REASON_LIVE

    @property
    def primary(self) -> Optional[PreviewFace]:
        """The face the readouts talk about: the biggest one. Size, not
        confidence -- at a desk the nearest face is the one at the desk,
        and a confident 20 px face across the room is not the subject.

        THIS IS NOT THE RULE ``visionrig.Rig`` USES. The rig picks
        ``max(faces, key=conf)``; the pane picks the biggest. What the two
        share is the geometry that produces the numbers -- ``observe``,
        ``HeadModel``, ``AttentionTracker`` -- so a yaw the pane prints is
        the same yaw his self-check prints for the same face. With more
        than one face in frame they can nevertheless report attention on
        DIFFERENT heads, and that is a real difference, not an oversight:
        for a pane read at a glance from a desk chair, nearest-is-the-
        subject is the honest rule. Anyone reconciling the two should
        change the rig rather than blunt this one.
        """
        return max(self.faces, key=lambda f: f.w * f.h, default=None)

    def numbers_only(self) -> dict:
        """Everything about this shot EXCEPT the pixels. What a log line, a
        status dict or a future diagnostic may have."""
        return {"faces": len(self.faces), "cap_w": self.cap_w,
                "cap_h": self.cap_h, "reason": self.reason,
                "detail": self.detail, "seq": self.seq, "fps": self.fps,
                "grab_ms": self.grab_ms, "live": self.live,
                "face": self.primary.as_dict() if self.primary else {},
                "hand": self.hand.as_dict() if self.hand is not None else {}}


def blank(reason: str, detail: str = "", seq: int = 0,
          at: float = 0.0) -> PreviewShot:
    """A shot with no picture and a stated reason."""
    return PreviewShot(image=None, reason=reason,
                       detail=detail or REASON_WORDS.get(reason, reason),
                       seq=seq, at=at)


# ------------------------------------------------------------- the config
def _option(get_option: Optional[Callable], key: str, default):
    if not callable(get_option):
        return default
    try:
        value = get_option(key, default)
    except TypeError:                     # a reader that takes only the key
        try:
            value = get_option(key)
        except Exception:                 # noqa: BLE001 - config boundary
            log.debug("campreview: %s unreadable", key, exc_info=True)
            return default
    except Exception:                     # noqa: BLE001 - config boundary
        log.debug("campreview: %s unreadable", key, exc_info=True)
        return default
    return default if value is None else value


def _float_option(get_option: Optional[Callable], key: str,
                  default: float) -> float:
    """A number from the config, or the default. NEVER raises.

    ``_option`` already survives a reader that explodes; this survives a
    reader that returns "on" where a float was wanted. It matters because
    ``build_pipeline`` promises not to raise -- a webcam setting typed wrong
    must not be able to blank the pane, let alone the console."""
    try:
        return float(_option(get_option, key, default))
    except (TypeError, ValueError):
        log.debug("campreview: %s is not a number; using %.3f", key, default)
        return float(default)


def preview_enabled(get_option: Optional[Callable]) -> bool:
    return bool(_option(get_option, OPTION_ENABLED, False))


def preview_fps(get_option: Optional[Callable]) -> float:
    """The PICTURE rate ceiling, clamped to [MIN_FPS, MAX_FPS].

    A ceiling, not a rate: the device delivers what its granted mode and the
    light allow (3.7-7.5 fps measured at 1280x720, see the module
    docstring), and a request above that buys a thread parked inside a
    blocking read rather than more pictures. It is NOT the rate the boxes or
    the name update at -- those have their own cadences (DETECT_FPS,
    IDENT_FPS), which is what makes a smooth picture affordable."""
    try:
        fps = float(_option(get_option, OPTION_FPS, DEFAULT_FPS))
    except (TypeError, ValueError):
        fps = DEFAULT_FPS
    return max(MIN_FPS, min(MAX_FPS, fps))


def poll_ms(fps: float) -> int:
    """How often the Tk side should LOOK for a new shot.

    Twice the CONFIGURED rate, so a frame is on screen within half a period
    of arriving, and never faster than 20 ms: 67 ms at the 7.5 default,
    33 ms at his 15, 20 ms at the 30 cap. The rate went up with the cap --
    12 polls a second at the old 6 fps default became 30 at 15 and 50 at
    the ceiling -- and that is affordable because a poll that finds nothing
    new is one attribute read and an int compare (``PreviewShot.seq``),
    which at the device's real 3.7-7.5 fps is most of them. The thing that
    must not happen is a poll ON the 16.67 ms reactor slot, and the 20 ms
    floor keeps every setting clear of it.
    """
    return max(20, int(round(500.0 / max(MIN_FPS, fps))))


# ------------------------------------------------------------ the reasons
def sensing_detail(state: Any) -> str:
    """"curfew until 7 am" / "offline mode" / "sensing state unknown".

    Reads the SAME state object the header badge does (jarvis/ui/
    sensing_badge.normalise), so the pane and the badge cannot disagree
    about why the lens is shut -- two readouts two inches apart telling
    different stories about a privacy control is worse than one.
    """
    from jarvis.ui.sensing_badge import normalise   # noqa: PLC0415 - lazy: no Tk
    s = normalise(state)
    if s["reason"] == "failsafe":
        return "sensing state unknown — off until you say otherwise"
    if s["reason"] == "timed" and s["until"]:
        from datetime import datetime                # noqa: PLC0415
        from jarvis.quiet import fmt_clock           # noqa: PLC0415
        end = datetime.fromtimestamp(float(s["until"]))
        return "offline until %s" % fmt_clock(end.hour, end.minute)
    if s["offline"]:
        return "offline mode"
    if s["reason"] == "curfew" and s["curfew"]:
        from jarvis.quiet import fmt_clock           # noqa: PLC0415
        return "curfew until %s" % fmt_clock(int(s["curfew"][1][0]),
                                             int(s["curfew"][1][1]))
    if not s["camera"]:
        return "camera off"
    return ""


def camera_allowed(state: Any) -> bool:
    """May the lens be open, per the state the badge reads? Anything this
    cannot read as an explicit yes is a no.

    THE STATE HAS TO ACTUALLY SAY SO. ``sensing_badge.normalise`` defaults a
    missing ``camera`` to True, which is right for a badge -- a header that
    hid itself on a malformed reading would say nothing at all -- and wrong
    here, where the same default would open a lens because an object of the
    wrong shape arrived. So the field must be PRESENT, and only then is its
    value trusted. That is the fail-to-offline rule reaching one more
    consumer, and it is the exact shape of bug offline mode exists to stop.
    """
    from jarvis.ui.sensing_badge import normalise   # noqa: PLC0415
    declared = (("camera" in state) if isinstance(state, dict)
                else hasattr(state, "camera"))
    if not declared:
        log.warning("campreview: a sensing state with no camera field (%s); "
                    "staying dark", type(state).__name__)
        return False
    try:
        s = normalise(state)
    except Exception:                     # noqa: BLE001 - provider boundary
        log.warning("campreview: unreadable sensing state; staying dark",
                    exc_info=True)
        return False
    return bool(s["camera"]) and not s["offline"]


# ------------------------------------------------------------- who it is
class IdentityHold:
    """The tracked face's name, held steady enough to read.

    THE PROBLEM THIS SOLVES IS NOT ACCURACY, IT IS LEGIBILITY. A per-frame
    verdict that alternates hunter / unknown / hunter is a chip nobody can
    read, and worse, it is a chip that says two contradictory things a
    hundred milliseconds apart about the person in the chair. So this is the
    same shape as ``visionrig.AttentionTracker``: an observation goes in, a
    HELD verdict comes out, and changing the held one costs evidence.

    THE ASYMMETRY IS THE DESIGN. Adopting a NAME needs ``agree`` readings in
    a row; adopting the first "unknown" needs one. Being slow to put a name
    on a face is a cosmetic delay of about a second. Being slow to take one
    OFF -- leaving "HUNTER" over someone who is not him because the first
    disagreeing reading was ignored -- is the pane telling him something
    false about who Jarvis thinks is there. Once something IS held, both
    directions cost the same evidence, because from then on a single odd
    reading in either direction is just noise.

    THE VERDICT IS BOUND TO THE FACE IT WAS COMPUTED FROM, and that binding
    is load-bearing rather than tidy. Everything above is about WHEN a label
    may change; a hold with no subject also has to answer WHOSE it is, and
    without that the two rules fight each other. The pane names its largest
    face, "largest" flips between two people at similar distance as their
    boxes jitter, and the hysteresis then works against the person looking
    at it: each intervening reading of the OTHER name resets the
    disagreement counter (so the two-reading bar is never reached) and
    refreshes the timestamp (so ``hold_s`` never expires either). Measured on
    the unbound version, 2026-09-03, synthetic two-face harness at 15 fps for
    150 pictures with "largest" flipping every picture: 142 chip-bearing
    frames, 70 of them one person's name and score over the other person's
    box, never converging; flipping every third picture, 126 named and 60
    wrong.

    So a reading carries the BOX it was taken from, and a box that is not
    plausibly the same face (``SUBJECT_IOU``) is a different question rather
    than a disagreement about this one. Two rules follow, and they differ on
    purpose. A READING about a different face replaces the subject: the
    hold is cleared and the new face starts from nothing, because the hold
    is "who the identity tick last looked at" and it just looked elsewhere.
    A mere STAMP request for a different face -- ``held(box)`` between
    ticks, when "largest" jitters to the other person for a frame -- is
    refused but does not clear: the verdict is withheld from that box and
    is still there when the face it belongs to is largest again. Clearing
    there too was measured (same harness): 18 named frames of 150 and 0
    wrong; withholding keeps 0 wrong and gives the right face its name back
    without another second of agreement -- 72 named frames of 150, still 0
    wrong. (Flipping every third picture, the identity ticks land on both
    faces in turn and each reading replaces the subject: 36 named, 0 wrong,
    either way.) The candidate counter is bound the same way, so "two
    agreeing readings" means two about ONE face. This is the rule
    ``_track`` already applies to carried boxes on a capture-geometry
    change, moved to the thing that has a person's name on it.

    ``clear()`` is the hard reset for "there is no subject": ``Eye``'s own
    ``SessionIdentity.room_empty`` makes the same call for the same reason --
    an empty frame is precisely the window in which a different person sits
    down, so an identity that survives one is an identity vouching for
    whoever is there when the next face appears.

    Nothing here is persisted and nothing here is a permission. It produces a
    label and a number for a chip on his screen.
    """

    def __init__(self, agree: int = IDENT_AGREE,
                 hold_s: float = IDENT_HOLD_S,
                 now: Callable[[], float] = time.monotonic,
                 iou: float = SUBJECT_IOU):
        self.agree = max(1, int(agree))
        self.hold_s = float(hold_s)
        self.iou = float(iou)
        self._now = now
        self._ran = False          # is ANYTHING held? "" is a verdict too
        self._label = ""
        self._score = 0.0
        self._at = 0.0
        self._box: Optional[tuple] = None       # whose verdict this is
        self._cand: Optional[str] = None
        self._cand_box: Optional[tuple] = None  # …and whose candidate
        self._n = 0

    def clear(self) -> None:
        """No subject, or no identity to run: forget who it was."""
        self._ran = False
        self._label, self._score, self._at = "", 0.0, 0.0
        self._box = None
        self._cand, self._cand_box, self._n = None, None, 0

    def _same(self, a: Optional[tuple], b: Optional[tuple]) -> bool:
        """Are these two boxes plausibly the same face?

        UNKNOWN READS AS YES, deliberately. A caller that does not carry
        geometry (the hold's own unit tests, and any future consumer that
        only has a label) gets the pre-binding behaviour rather than a hold
        that clears itself on every reading. The binding is enforced where
        the geometry exists, which is ``PreviewPipeline._name``.
        """
        if a is None or b is None:
            return True
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return True                         # not geometry; do not judge
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return union > 0 and (inter / union) >= self.iou

    def observe(self, label: str, score: float,
                box: Optional[tuple] = None) -> tuple:
        """One identity reading -> the held ``(label, score, ran)``.

        ``label`` is "" for "asked, nothing matched", which is an answer and
        is held like any other. ``box`` is the detection the embedding came
        from, in capture pixels; a reading about a DIFFERENT face resets
        rather than votes.
        """
        label = str(label or "")
        score = float(score)
        now = self._now()
        if box is not None and not self._same(self._box, box):
            # Not a disagreement about this face -- a different face. The
            # held verdict has nothing to say about it and must not be
            # confirmed, contradicted or extended by it.
            self.clear()
        if self._ran and label == self._label:
            self._score, self._at = score, now      # a confirmation
            self._box = box if box is not None else self._box
            self._cand, self._cand_box, self._n = None, None, 0
            return self.held(box)
        if not self._ran and label == "":
            self._adopt("", score, now, box)        # say so at once
            return self.held(box)
        if label != self._cand or not self._same(self._cand_box, box):
            self._cand, self._n = label, 0
        self._cand_box = box if box is not None else self._cand_box
        self._n += 1
        if self._n >= self.agree:
            self._adopt(label, score, now, box)
        return self.held(box)

    def _adopt(self, label: str, score: float, now: float,
               box: Optional[tuple] = None) -> None:
        self._ran = True
        self._label, self._score, self._at = label, score, now
        self._box = box
        self._cand, self._cand_box, self._n = None, None, 0

    def held(self, box: Optional[tuple] = None) -> tuple:
        """``(label, score, ran)``, expiring a verdict nothing has confirmed
        for ``hold_s``, and refusing to lend one to a face it was not
        computed from.

        Read, not tick, because the caller only comes back when a frame
        arrives -- and a chip that outlived its evidence because nobody
        called a tick would be exactly the stale name this exists to
        prevent. ``box`` is the face the caller is about to stamp: a verdict
        that does not belong to it is WITHHELD -- ``("", 0.0, False)``, no
        chip -- rather than drawn, which is the difference between a missing
        name and a wrong one. It is not cleared: the verdict still belongs
        to the face it came from, and that face is usually largest again on
        the next picture. Only a READING (``observe``) moves the subject.
        """
        if self._ran and (self._now() - self._at) > self.hold_s:
            self.clear()
        if self._ran and box is not None and not self._same(self._box, box):
            return "", 0.0, False
        return self._label, self._score, self._ran


def _usable_embedding(vec) -> bool:
    """Finite, non-empty, non-zero -- the two failure shapes a recogniser
    can hand back without raising. The gallery applies stricter checks of
    its own (dimension, spread); those are the gallery's business and it
    answers ("", 0.0) for them, which ``_match`` then accepts as an answer.
    """
    try:
        import numpy as np                        # noqa: PLC0415 - lazy
        arr = np.asarray(vec, dtype=float).ravel()
        if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
            return False
        return float(np.linalg.norm(arr)) > 0.0
    except Exception:                             # noqa: BLE001 - not a vector
        return False


# ----------------------------------------------------------- the pipeline
class PreviewPipeline:
    """Frames -> (small image, numbers). Owns nothing that opens a device.

    ``feed`` is ``jarvis.camera.CameraFeed`` in production and a stub in the
    suite; the contract is only ``capture() -> frame or None`` and
    ``close()``, which is the same seam ``visionrig.Rig`` drives, so the
    preview runs through the gated path rather than beside it.

    ``detector`` is ``jarvis.facedetect.YuNetDetector`` (``input_size`` +
    ``detect(frame)``); None means the pane shows the reason and no boxes,
    NEVER "nobody is there". That distinction is the whole lesson of the VSS
    yunet path that silently fell back and never once ran.

    ``owned`` says whether shutting the device is this object's business --
    False for the feed the app handed over. See ``close``.
    """

    def __init__(self, feed, detector=None, lens=None, head=None,
                 tracker=None, reason: str = "", observe=None,
                 now: Callable[[], float] = time.monotonic,
                 owned: bool = True, recogniser=None, gallery=None,
                 identity_min: float = DEFAULT_IDENTITY_MIN,
                 min_conf: float = DEFAULT_MIN_CONF,
                 detect_fps: float = DETECT_FPS,
                 ident_fps: float = IDENT_FPS, hold=None, hands=None):
        self.feed = feed
        self.owned = bool(owned)
        self.detector = detector
        self.lens = lens
        self.head = head
        self.tracker = tracker
        self.reason = reason
        # Identity. Both of these are None unless camera.identity is on AND
        # the weights and an enrolment are actually there, so "no name" is
        # the resting state of this feature rather than its failure mode.
        self.recogniser = recogniser
        self.gallery = gallery
        self.identity_min = float(identity_min)
        self.min_conf = float(min_conf)
        # THE IDENTITY GATE HAS ITS OWN FLOOR RATHER THAN BORROWING THE
        # DETECTOR'S. camera.min_conf decides how many boxes get DRAWN, and
        # assistant_config records that it has already been turned down once
        # (0.7 -> 0.6) "because a miss is SILENT" -- the documented direction
        # of travel is downward. Borrowing it means a dial about how much the
        # pane shows also decides who gets NAMED: at min_conf 0.0 a
        # 0.25-scoring detection produced three embeddings and a chip reading
        # "HUNTER 0.71" over a smudge. Raising the detector bar still raises
        # this one (max, not a constant), because a detection the pane will
        # not even draw is not one to embed.
        self.id_conf = max(float(min_conf), DEFAULT_MIN_CONF)
        self._ident = hold if hold is not None else IdentityHold(now=now)
        # The three cadences. Periods rather than rates because that is what
        # the comparison against the clock wants, and a rate of 0 would be a
        # division by zero at the one place it must not happen.
        self._detect_period = 1.0 / max(0.1, float(detect_fps))
        self._ident_period = 1.0 / max(0.1, float(ident_fps))
        # None, not 0.0, and the distinction has teeth: a monotonic clock
        # that starts at zero -- every hand-wound one in the suite, and any
        # implementation that measures from process start -- would make a
        # falsy check read "never detected" on the second frame and detect
        # twice in a row for ever.
        self._last_detect: Optional[float] = None
        self._last_ident: Optional[float] = None
        # The MEASURED interval between pictures, which is what the detect
        # cadence has to round against -- see _track. Measured rather than
        # configured because the two differ: the device delivers what it
        # delivers whatever camera.preview_fps asks for.
        self._last_pic: Optional[float] = None
        self._pic_dt: Optional[float] = None
        self._since_detect = 0
        # The boxes carried forward between detections, and the capture
        # geometry they are expressed in. Faces are ALWAYS in capture pixels,
        # so a mode change makes the carried ones wrong rather than stale.
        self._held: tuple = ()
        self._held_cap: tuple = (0, 0)
        # The hand stage (jarvis/handstage.HandStage), or None. It runs on
        # THIS frame on THIS thread inside grab() -- the one place a frame
        # exists -- and answers scalars. Not a second consumer of the lens.
        self.hands = hands
        # ``visionrig.observe`` by default, resolved lazily so this module
        # loads on a tree where that lane has not landed. A seam rather than
        # an import because it lets the suite exercise THIS class's own
        # contract -- the scale-back-up, the ordering, the cap, the tracker
        # -- on a tree that has no vision lane at all, while the geometry
        # itself stays tested where it lives (tests/test_visionrig.py).
        self._observe = observe
        self._now = now
        self.frames = 0
        self.misses = 0

    def _observer(self):
        if self._observe is not None:
            return self._observe
        try:
            from jarvis.visionrig import observe   # noqa: PLC0415 - lazy lane
        except Exception:                          # noqa: BLE001
            log.debug("campreview: visionrig unavailable", exc_info=True)
            return None
        self._observe = observe
        return observe

    # ---------------------------------------------------------- geometry
    def _scales(self, frame_w: int, frame_h: int) -> tuple:
        """detect px -> capture px, per axis.

        Separate because they can DISAGREE (a 4:3 detect size against a
        16:9 capture squashes every face 1.33x horizontally), and averaging
        them would hide the mistake that produces it.
        """
        size = getattr(self.detector, "input_size", None) or (0, 0)
        dw, dh = int(size[0]), int(size[1])
        return (frame_w / dw if dw else 1.0, frame_h / dh if dh else 1.0)

    def _faces(self, frame, rows, frame_w: int, frame_h: int,
               at: float = 0.0) -> tuple:
        """YuNet rows -> ``PreviewFace``, through visionrig's geometry.

        The yaw is NOT re-derived here. ``visionrig.observe`` already owns
        the roll-invariant landmark ratio and ``HeadModel`` owns the one
        anthropometric constant that turns it into degrees; a second copy
        in a UI feature is a second copy that can disagree with the number
        the rig prints in his self-check report. (The SUBJECT rule does
        differ from the rig's -- see ``PreviewShot.primary``.)

        THE CAP IS APPLIED AFTER THE SORT, and the order is the whole
        point. YuNet hands its rows back score-descending, so slicing
        first would keep by CONFIDENCE and drop by confidence: with four
        detections -- three 20 px faces across the room at 0.99/0.98/0.97
        and the 360 px one at the desk at 0.80 -- the subject would be the
        row that got thrown away, and ``_attend`` would then put the
        attention verdict on a head two metres behind him.
        """
        # A DETECTION THAT COULD NOT BE MADE IS NOT WEAKER EVIDENCE OF
        # "NOBODY IS THERE" THAN ONE THAT CAME BACK EMPTY. Both of these
        # returns used to skip the clear() below, so a single frame where the
        # detector raised left the name held -- and the next face, a
        # different person 0.2 s later, inherited it (measured: the
        # stranger's chip read name='hunter' id_ran=True). The boxes were
        # already dropped on this path; the NAME was not, which is the half
        # that says something false about a person.
        if rows is None:
            self._ident.clear()
            return ()
        observe = self._observer()
        if observe is None:
            self._ident.clear()
            return ()
        sx, sy = self._scales(frame_w, frame_h)
        out = []
        for row in list(rows)[:MAX_ROWS]:
            try:
                obs = observe(row, self.lens, sx, sy, self.head)
            except Exception:                      # noqa: BLE001 - one bad row
                log.debug("campreview: a detection row would not reduce",
                          exc_info=True)
                continue
            # The ROW travels beside the reduced face, because SFace aligns
            # its own crop from the five landmarks and cannot work from a
            # rectangle. It is a local; nothing about it reaches an
            # attribute, a shot or a log line.
            out.append((PreviewFace(conf=float(obs.conf), x=float(obs.x),
                                    y=float(obs.y), w=float(obs.w),
                                    h=float(obs.h),
                                    yaw_deg=float(obs.yaw_deg),
                                    landmarks_ok=bool(obs.landmarks_ok),
                                    eye_px=float(getattr(obs, "eye_px", 0.0)
                                                 or 0.0)),
                        row))
        out.sort(key=lambda pair: pair[0].w * pair[0].h, reverse=True)
        out = out[:MAX_FACES]
        faces = self._attend([face for face, _ in out])
        if faces:
            faces[0] = self._name(frame, faces[0], out[0][1], at)
        else:
            self._ident.clear()
        return tuple(faces)

    def _attend(self, faces: list) -> list:
        """The attention verdict, on the BIGGEST face only.

        One tracker, one subject: hysteresis is a per-head state machine
        (visionrig.AttentionTracker) and feeding it whichever face the
        detector happened to list first would make its dwell meaningless.
        With no tracker the flag stays False rather than defaulting to
        "yes" -- an attention indicator that is right by accident is one he
        would learn to ignore.
        """
        if self.tracker is None or not faces:
            if self.tracker is not None:
                try:
                    self.tracker.update(None)     # a gap ends the dwell
                except Exception:                 # noqa: BLE001
                    log.debug("campreview: tracker refused", exc_info=True)
            return faces
        head = faces[0]
        try:
            inside = bool(self.tracker.update(
                head.yaw_deg if head.landmarks_ok else None))
        except Exception:                         # noqa: BLE001
            log.debug("campreview: tracker refused", exc_info=True)
            return faces
        # ``replace`` rather than a fresh PreviewFace listing every field:
        # the second spelling silently DROPS whatever field is added next,
        # which is how a name would quietly stop reaching the pane.
        faces[0] = replace(head, attending=inside)
        return faces

    # ------------------------------------------------------------- who
    def _name(self, frame, face: PreviewFace, row, at: float) -> PreviewFace:
        """The tracked face, with a held identity on it -- or unchanged.

        THREE GATES BEFORE AN EMBEDDING IS TAKEN, and the order matters:

        1. identity has to be wired at all (``camera.identity`` on, weights
           present, an enrolment loaded). Off is the resting state.
        2. the detection has to have cleared ``self.id_conf`` ALREADY.
           SFace answers confidently on inputs that are not faces -- unrelated
           non-face crops match each other at 0.66-0.92 against a 0.363 bar --
           so an embedding from a weak detection is not a weak answer, it is a
           confident wrong one. ``SFaceRecogniser.embed`` refuses the same row
           for the same reason; this check is what keeps the refusal off the
           exception path 15 times a second.
        3. the cadence has to be due. 10.4 ms per embedding is the most
           expensive thing in this module by a factor of five.

        Between ticks the HELD verdict is stamped on anyway, so the chip
        stays with the box rather than blinking at the identity rate -- but
        only onto the FACE IT BELONGS TO. The subject's geometry goes into
        the hold with the reading and is checked against on the way out, so
        a name cannot walk from the face it was computed from onto whoever
        happens to be largest in the next picture. See ``IdentityHold``.
        """
        if self.recogniser is None or self.gallery is None:
            return face
        box = (face.x, face.y, face.w, face.h)
        due = (self._last_ident is None
               or (at - self._last_ident) >= self._ident_period)
        if due and face.conf >= self.id_conf:
            self._last_ident = at
            label, score = self._match(frame, row)
            if label is not None:      # None = it could not be asked; hold
                self._ident.observe(label, score, box)
        label, score, ran = self._ident.held(box)
        if not ran:
            return face
        return replace(face, name=label, id_score=score, id_ran=True)

    def _match(self, frame, row) -> tuple:
        """``(label, score)`` for one detection, or ``(None, 0.0)`` when the
        question could not be asked.

        None is not "" here. "" is an ANSWER -- asked, nothing in the gallery
        matched -- and the pane draws it as UNKNOWN. None is a failure of the
        machinery (a crop that would not align, a gallery that raised), and
        the held verdict is left alone rather than being overwritten by a
        verdict nobody reached.
        """
        try:
            vec = self.recogniser.embed(frame, row)
        except Exception:                     # noqa: BLE001 - a model edge
            log.debug("campreview: the recogniser declined a row",
                      exc_info=True)
            return None, 0.0
        if not _usable_embedding(vec):
            # A crop of nothing (non-finite) or a blank one (all zero) is
            # the MACHINERY failing, not the gallery answering. The gallery
            # would return ("", 0.0) for it -- the same shape as "asked,
            # nobody matched" -- and the pane would print UNKNOWN 0.00 over
            # a face nobody actually asked about. None leaves the held
            # verdict alone, the way a recogniser exception does. A ("",
            # 0.0) that survives this check IS accepted as an answer: it is
            # what the gallery says when every enrolled sample sits at or
            # under cosine 0, which is a stranger, not a fault.
            log.debug("campreview: the embedding was not usable")
            return None, 0.0
        try:
            label, score = self.gallery.match(vec)
        except Exception:                     # noqa: BLE001 - a store edge
            log.debug("campreview: the gallery would not match", exc_info=True)
            return None, 0.0
        score = float(score)
        if not label or score < self.identity_min:
            return "", score                  # asked, and the answer is no
        return str(label), score

    def _hand(self, frame, faces, frame_w: int, frame_h: int, seq: int):
        """The gesture stage, on this frame, on this thread. Scalars out.

        None is NO OPINION -- no stage, or the stage raised -- and is
        deliberately not ``HandShot(present=False)``, which would assert
        there is no hand in the room. Its own try/except, separate from the
        detector's: a tracker that raises must leave the face path and the
        picture untouched.
        """
        stage = self.hands
        if stage is None:
            return None
        try:
            return stage.observe(frame, faces, frame_w, frame_h, seq)
        except Exception:                          # noqa: BLE001 - the stage
            log.debug("campreview: the hand stage raised", exc_info=True)
            return None

    # ----------------------------------------------------------- capture
    def grab(self, box: tuple, seq: int = 0) -> PreviewShot:
        """One cycle: a frame, its detections, and a picture scaled to
        ``box`` -- ``(width, height)`` in device pixels.

        The FULL frame does not survive this call. It is read into a local,
        reduced to a small image and a handful of floats, and dropped when
        the frame returns; nothing about it is stored on ``self``.
        """
        t0 = self._now()
        try:
            frame = self.feed.capture()
        except Exception:                          # noqa: BLE001 - device edge
            log.debug("campreview: the feed raised", exc_info=True)
            frame = None
        if frame is None:
            self.misses += 1
            return blank(REASON_NO_FRAME, seq=seq, at=time.time())
        try:
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        except (AttributeError, IndexError, TypeError):
            self.misses += 1
            return blank(REASON_NO_FRAME, "the camera returned something "
                                          "that is not a frame",
                         seq=seq, at=time.time())

        faces = self._track(frame, frame_w, frame_h, t0)
        # After the faces (attention gates the hand stage), before shrink
        # (the last use of the full frame). One capture, one consumer --
        # and between detections the faces are the ones _track carries
        # forward, which is what the stage's baseline and latch read.
        hand = self._hand(frame, faces, frame_w, frame_h, seq)

        # LETTERBOXED HERE, on this thread, so what crosses over is the
        # picture at exactly the size it will be drawn. Stretching a 4:3
        # camera into a 16:9 box would squash every face 1.33x and put
        # every overlay box slightly off the face it belongs to.
        _ox, _oy, bw, bh = fit_box(frame_w, frame_h, box[0], box[1])
        image = shrink(frame, (bw, bh))
        self.frames += 1
        detail = "" if self.detector is not None else \
            (self.reason or "no face detector")
        return PreviewShot(image=image, faces=faces, cap_w=frame_w,
                           cap_h=frame_h, reason=REASON_LIVE, detail=detail,
                           seq=seq, at=time.time(),
                           grab_ms=(self._now() - t0) * 1000.0, hand=hand)

    def _track(self, frame, frame_w: int, frame_h: int,
               at: float) -> tuple:
        """The boxes for THIS picture: a fresh detection when one is due,
        otherwise the last ones carried forward.

        WHY CARRY THEM RATHER THAN RUN THE DETECTOR EVERY FRAME. At 15 fps a
        detection on every picture is 15 x 2 ms of arithmetic a second for a
        box that moves at the speed of a head at a desk; at 8 Hz it is 16 ms
        and the box is at most 125 ms behind. He asked for the pane to stop
        looking laggy, and the lag he can see is the PICTURE stepping, not a
        box trailing an eighth of a second -- so the picture rate went up and
        the detection rate did not.

        WHY THE CARRY IS BOUNDED BY MORE THAN THE CLOCK. A carried face is in
        the CAPTURE pixels of the frame it came from, so a camera that
        changes mode mid-run makes it wrong rather than merely stale; the
        geometry check drops it. A detection that returns nothing clears it
        immediately -- an empty room has to empty the pane on the next
        detection, not linger for a hold period -- and that is the same pass
        that tells the attention tracker the dwell is over.

        WHY THE CADENCE IS EVERY Nth PICTURE RATHER THAN A WALL-CLOCK
        PERIOD. Detections can only happen when a picture arrives, so the
        achievable box rates are the picture rate divided by a whole number
        -- and a bare ``>= period`` test picks the first multiple AT OR
        ABOVE the period, which can be much slower than the rate it was
        asked for. At 10 pictures a second against a 125 ms period that is
        every SECOND picture: 5 Hz boxes, slower than the 6 Hz the pane
        managed before any of this; at the 7.5 fps the device actually
        delivers it is inert. ``detect_stride`` rounds to the nearest
        achievable multiple instead, so the saving can never cost more than
        it saves. The picture interval is MEASURED -- an EMA over the last few
        gaps between live pictures, so N follows the device's rate rather
        than any one gap, and a stall errs towards detecting MORE often for
        a few pictures -- not taken from ``camera.preview_fps``, because on
        this camera those two are not the same number (module docstring).

        The IDENTITY cadence deliberately stays on the clock. Rounding it up
        would raise the rate of the one thing here that costs real
        arithmetic (10.4 ms an embedding), and identity running slightly
        slow is invisible: the hold covers the gap by design.
        """
        dt = at - self._last_pic if self._last_pic is not None else 0.0
        self._last_pic = at
        if 0.0 < dt < 2.0:
            self._pic_dt = dt if self._pic_dt is None \
                else 0.7 * self._pic_dt + 0.3 * dt
        if self.detector is None:
            self._held = ()
            self._ident.clear()
            return ()
        if self._held_cap != (frame_w, frame_h):
            # The NAME goes with the boxes. A held verdict is expressed in
            # the same capture pixels its subject's box was, so a mode change
            # makes it unplaceable rather than merely stale -- and a camera
            # that renegotiates its mode while somebody new sits down is
            # exactly the window in which the last name is wrong.
            self._held, self._held_cap = (), (frame_w, frame_h)
            self._last_detect = None
            self._ident.clear()
        self._since_detect += 1
        stride = detect_stride(1.0 / self._pic_dt if self._pic_dt else 0.0,
                               1.0 / self._detect_period)
        if self._last_detect is not None and self._since_detect < stride:
            return self._held
        self._last_detect = at
        self._since_detect = 0
        try:
            rows = self.detector.detect(frame)
        except Exception:                          # noqa: BLE001 - a detector
            log.debug("campreview: the detector raised", exc_info=True)
            rows = None
        self._held = self._faces(frame, rows, frame_w, frame_h, at)
        return self._held

    def close(self) -> None:
        """Stop reading the device -- and shut it only if it is OURS.

        The preview closes what it opened. It does NOT close
        ``services.camera_feed``: that object is the app's lens, the
        console's own vision lane reads it, and ``CameraFeed.close()``
        bumps the gate epoch (invalidating every ``_GatedDevice`` already
        handed out, so an in-flight read elsewhere is discarded) and
        releases the raw device, which then has to be reopened and have its
        focus and exposure pinned again. This method is reached on every
        ACTIVE->AMBIENT transition, every curfew edge and every toggle-off,
        so closing a borrowed feed would cost the app's lane a device
        reopen each time the console went quiet.

        What the preview owes when it may not look is to STOP LOOKING, and
        dropping the reference is that. Whether the lens itself shuts is
        the sensing owner's ruling (``jarvis/sensing.py`` releases the
        devices it has attached), not a UI pane's.
        """
        feed, self.feed = self.feed, None
        if feed is None or not self.owned:
            return
        try:
            feed.close()
        except Exception:                          # noqa: BLE001
            log.debug("campreview: the feed would not close", exc_info=True)


def fit_box(cap_w: int, cap_h: int, box_w: int, box_h: int) -> tuple:
    """``(ox, oy, w, h)`` -- where the picture sits inside the pane's box.

    LETTERBOXED, not stretched, and the offsets matter as much as the size:
    a 4:3 camera in a 16:9 box would otherwise be squashed 1.33x and every
    face box drawn on it would sit slightly off the face it belongs to. The
    same numbers scale the overlay (``face_rect``), so the boxes cannot
    drift out of step with the picture whatever camera is plugged in.
    """
    box_w, box_h = max(1, int(box_w)), max(1, int(box_h))
    if cap_w <= 0 or cap_h <= 0:
        return 0, 0, box_w, box_h
    scale = min(box_w / float(cap_w), box_h / float(cap_h))
    w = max(1, min(box_w, int(round(cap_w * scale))))
    h = max(1, min(box_h, int(round(cap_h * scale))))
    return (box_w - w) // 2, (box_h - h) // 2, w, h


def shrink(frame, box: tuple):
    """A capture-sized array -> a pane-sized PIL Image, or None.

    Scaled HERE, on the capture thread, for two reasons that both matter:
    the Tk thread must not spend 2-4 ms per frame resampling 2.7 MB, and
    the reduced image is the only thing that ever crosses the boundary, so
    the full frame cannot be retained by a consumer that keeps a reference.

    The channel swap runs AFTER the resize (BGR -> RGB on 160x90 is 43 kB
    of work against 2.7 MB), and cv2 does the resize when it is there --
    it is already imported for the capture, and INTER_AREA is the correct
    filter for a 4:1 reduction. PIL is the fallback so a box without cv2
    still shows a picture rather than a stated failure it cannot fix.
    """
    w, h = max(1, int(box[0])), max(1, int(box[1]))
    try:
        from PIL import Image                     # noqa: PLC0415 - lazy
    except Exception:                             # noqa: BLE001
        log.debug("campreview: PIL is not available", exc_info=True)
        return None
    try:
        import numpy as np                        # noqa: PLC0415
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        try:
            import cv2                            # noqa: PLC0415 - lazy
            small = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
            return Image.fromarray(small[:, :, 2::-1])
        except Exception:                         # noqa: BLE001 - no cv2
            img = Image.fromarray(arr[:, :, 2::-1])
            return img.resize((w, h))
    except Exception:                             # noqa: BLE001 - a weird frame
        log.debug("campreview: a frame would not reduce", exc_info=True)
        return None


# ------------------------------------------------------------- resolving
def resolve_feed(services=None, get_option: Optional[Callable] = None,
                 policy=None, build=None) -> tuple:
    """``(feed, reason, owned)``. Never raises, never opens a device itself.

    The app's own feed wins whenever there is one. See the module docstring
    for why that is a correctness rule and not a preference: two feeds mean
    two ``SensingPolicy.attach`` calls under the same name, the second
    displaces the first, and the curfew stops reaching the lens the first
    one held.

    ``owned`` travels WITH the feed rather than being inferred later,
    because the two ways of getting one differ in exactly this: a feed the
    preview built is the preview's to shut, and a feed the app handed over
    is not (``PreviewPipeline.close``). Inferring it at the closing end --
    by identity against ``services``, say -- would be a second reading of a
    fact only this function actually knows.
    """
    feed = getattr(services, "camera_feed", None) if services else None
    if feed is not None:
        return feed, "", False                     # BORROWED: do not close
    if policy is None:
        return None, "sensing owner not wired", False
    maker = build
    if maker is None:
        try:
            from jarvis.camera import build as maker   # noqa: PLC0415
        except Exception as exc:                       # noqa: BLE001
            return None, "the camera lane is not installed (%s)" % exc, False
    log.warning("campreview: no services.camera_feed; building the preview's "
                "own gated feed. The app should hand one over instead -- "
                "SensingPolicy.attach replaces by name, so two feeds mean the "
                "curfew only reaches the newer one.")
    try:
        feed, reason = maker(_CfgView(get_option), policy)
    except Exception as exc:                           # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc), False
    return feed, reason, feed is not None


class _CfgView:
    """``camera.build`` asks its config for dotted keys and nothing else, so
    the settings drawer's reader is already the whole interface. Wrapping it
    here means the preview needs NO new app wiring: ``Services.get_option``
    and ``Services.sensing`` were both already there for the settings drawer
    and the header badge."""

    __slots__ = ("_get",)

    def __init__(self, get_option: Optional[Callable]):
        self._get = get_option

    def get(self, key: str, default=None):
        return _option(self._get, key, default)


def build_pipeline(services=None, get_option: Optional[Callable] = None,
                   policy=None, feed=None, build=None,
                   owned: bool = True, hands=None) -> PreviewPipeline:
    """The pipeline, or one whose ``feed`` is None and whose ``reason`` says
    why. Never raises: a camera must not be able to take the console down,
    and a console that failed to build because a webcam was unplugged would
    be a worse bug than no preview.

    ``owned`` applies only to a ``feed`` passed in directly (the suite's
    seam); a resolved one carries its own answer back from
    ``resolve_feed``."""
    cfg = _CfgView(get_option)
    reason = ""
    if feed is None:
        feed, reason, owned = resolve_feed(services, get_option, policy,
                                           build)
    if feed is None:
        return PreviewPipeline(None, reason=reason or "no camera feed",
                               hands=hands)

    detector = lens = head = tracker = None
    min_conf = _float_option(get_option, OPTION_MIN_CONF, DEFAULT_MIN_CONF)
    identity_min = _float_option(get_option, OPTION_IDENTITY_MIN,
                                 DEFAULT_IDENTITY_MIN)
    try:
        from jarvis import camera as cam            # noqa: PLC0415 - lazy lane
        from jarvis.visionrig import AttentionTracker  # noqa: PLC0415
        detector, why = cam.detector_from_config(cfg)
        reason = why or reason
        lens = getattr(feed, "lens", None) or cam.lens_from_config(cfg)
        head = cam.head_from_config(cfg)
        th = cam.thresholds_from_config(cfg)
        min_conf, identity_min = th.min_conf, th.identity_min
        tracker = AttentionTracker(th.cone_deg, th.cone_hysteresis_deg,
                                   th.cone_centre_deg)
    except Exception as exc:                        # noqa: BLE001
        log.info("campreview: the vision lane is incomplete (%s: %s)",
                 type(exc).__name__, exc)
        reason = reason or "%s: %s" % (type(exc).__name__, exc)
    recogniser, gallery = resolve_identity(cfg)
    return PreviewPipeline(feed, detector=detector, lens=lens, head=head,
                           tracker=tracker, reason=reason, owned=owned,
                           recogniser=recogniser, gallery=gallery,
                           identity_min=identity_min, min_conf=min_conf,
                           hands=hands)


def resolve_identity(cfg) -> tuple:
    """``(recogniser, gallery)`` -- both None unless a name can honestly be
    put on the pane. Never raises.

    THREE THINGS HAVE TO BE TRUE and each of them is somebody's explicit
    choice: ``camera.identity`` is on (``recogniser_from_config`` refuses
    otherwise, so nothing about his face is computed when it is off), the
    SFace weights are present and load, and the gallery holds at least one
    enrolled label. An empty gallery is not an error and not a bug -- it is
    the state before he enrols -- and the honest response to it is no chip,
    not "UNKNOWN" on every face forever.

    THE GALLERY IS OPENED READ-ONLY, and this module has no code path that
    writes one. That is deliberate: jarvis/facegallery.py exists because a
    test destroyed his voiceprint by SAVING over it, and a UI pane is the
    last thing that should be able to touch the store his face lives in.

    THE COST IS PAID ON THE CAPTURE THREAD, ONCE PER PIPELINE. Loading SFace
    is 48 ms measured (2026-09-03; YuNet is 2), and the pipeline is rebuilt
    on every ACTIVE wake and every curfew lift, so that is 48 ms added to the
    first frame after the console comes back -- on the capture thread, behind
    the pane's own "waking the camera…", and not on the thread the reactor
    animates on. Caching it across rebuilds would mean holding a loaded face
    model through offline mode and the curfew, which is a worse trade than
    50 ms.
    """
    try:
        from jarvis import camera as cam           # noqa: PLC0415 - lazy lane
        recogniser, why = cam.recogniser_from_config(cfg)
    except Exception as exc:                       # noqa: BLE001
        log.info("campreview: no face recogniser (%s: %s)",
                 type(exc).__name__, exc)
        return None, None
    if recogniser is None:
        log.info("campreview: no name beside the box (%s)", why)
        return None, None
    try:
        from jarvis.config import PATHS            # noqa: PLC0415
        from jarvis.facegallery import FaceGallery  # noqa: PLC0415
        gallery = FaceGallery(PATHS.FACE_GALLERY)
        if not gallery.load() or not gallery.labels():
            log.info("campreview: nobody is enrolled; the pane will show "
                     "boxes without names")
            return None, None
    except Exception as exc:                       # noqa: BLE001
        log.info("campreview: the face gallery would not open (%s: %s)",
                 type(exc).__name__, exc)
        return None, None
    log.info("campreview: identity on -- %d enrolled label(s)",
             len(gallery.labels()))
    return recogniser, gallery


# --------------------------------------------------------------- the loop
class PreviewWorker:
    """The capture thread, and the latest-wins slot the Tk side reads.

    START/STOP IS THE PRIVACY CONTROL, not a visibility flag. ``stop()``
    ends the capture and hands the device back, so with the toggle off --
    or with the console in standby, where nobody is looking at the pane --
    there is no capture loop, no ``capture()`` call and no device held. A
    worker that kept grabbing behind a hidden widget would be a camera
    running for nobody.

    THE DENY IS SYNCHRONOUS; THE HANDBACK NEED NOT BE. Setting the stop
    event is what guarantees no further frame: the loop checks it before
    every pass, ``cycle`` checks it before opening anything, and a grab
    already in flight is DROPPED rather than published. Waiting for the
    thread is only about giving the device back, so ``stop(join=False)``
    moves that wait off the caller -- see ``stop`` for why the Tk thread
    must not do it.

    The thread NEVER touches Tk. It writes one immutable ``PreviewShot`` to
    ``self._shot`` under a lock and the UI reads it on its own timer; there
    is no queue to back up, because a preview frame that is two frames old
    has no value and dropping it is the correct behaviour.
    """

    def __init__(self, *, get_option: Optional[Callable] = None,
                 sensing=None, services=None, box=(160, 90),
                 pipeline=None, make_pipeline=None,
                 now: Callable[[], float] = time.monotonic,
                 sleep: Optional[Callable[[float], None]] = None,
                 hands=None):
        self.get_option = get_option
        self.sensing = sensing
        self.services = services
        self.box = (int(box[0]), int(box[1]))
        self._pipeline = pipeline
        # The hand stage outlives any one pipeline (a pipeline is rebuilt
        # on every start); its state machine resets itself on a stall.
        self.hands = hands
        self._make = make_pipeline or (
            lambda: build_pipeline(services, get_option, sensing,
                                   hands=hands))
        self._now = now
        self._sleep = sleep
        self._lock = threading.Lock()            # guards self._shot
        self._pipe_lock = threading.Lock()       # guards self._pipeline
        # ONE capture at a time across generations -- see _run.
        self._lease = threading.Lock()
        # The CURRENT generation's stop event. start() mints a new one
        # rather than clearing this; nothing ever clears a stop event.
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._shot = blank(REASON_DISABLED)
        self._last_at = 0.0
        self._logged = 0.0
        self.cycles = 0

    # -------------------------------------------------------- the switch
    @property
    def fps(self) -> float:
        """The configured capture rate. Read fresh, so the pane's poll
        interval follows a config edit at the next start rather than being
        frozen at construction."""
        return preview_fps(self.get_option)

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, enabled: Optional[bool] = None) -> bool:
        """Begin capturing. Idempotent; False when the toggle says no.

        The toggle is read HERE rather than inside the loop so that "off"
        costs no thread at all, which is the difference between a switch
        and a curtain.

        ``enabled`` OVERRIDES that read with a value the caller already
        has. ``SettingsDrawer._set_option`` writes assistant.json on a
        daemon thread and echoes to the console synchronously, so the
        console's ON path holds the new value while a re-read here could
        still see the old one -- and a stale False would pack the pane and
        then refuse to capture behind it, saying "OFF -- preview off",
        with nothing to retry it until the next console mode change. Every
        other caller (the first build, a mode change) passes nothing and
        gets the config read.
        """
        if self.running:
            return True
        want = (preview_enabled(self.get_option) if enabled is None
                else bool(enabled))
        if not want:
            self._publish(blank(REASON_DISABLED))
            return False
        # A NEW event, never a cleared one. Clearing would un-set the flag
        # a PREVIOUS generation is still waiting on -- a stop whose bounded
        # join timed out leaves exactly such a thread, inside a blocking
        # read -- and revive it, so two threads would drive one device for
        # the life of the process, racing in jarvis/eye.py's lockless
        # capture() and in this object's own counters.
        self._stop = stop = threading.Event()
        self._publish(blank(REASON_WAITING))
        self._thread = threading.Thread(target=self._run, args=(stop,),
                                        daemon=True, name="campreview")
        self._thread.start()
        return True

    def stop(self, timeout: float = 2.0, join: bool = True) -> None:
        """End the capture and hand the device back. Idempotent, never
        raises.

        WHAT IS SYNCHRONOUS: the stop event is set and the slot says
        "disabled" before this returns. That is the deny, and it is
        complete -- the loop will not start another pass, ``cycle`` will
        not open anything, and a grab already in flight is dropped instead
        of published.

        WHAT MAY NOT BE: the join. The thread can be inside a 133-268 ms
        grab (one frame interval of his LifeCam, measured 2026-09-03) or,
        on a wedged camera, inside one for seconds, and this is called on
        the TK THREAD -- from ``_preview_apply`` at every ACTIVE->AMBIENT
        edge, which is every 45 s of quiet. 130 ms there is eight
        consecutive 60 Hz slots (sixteen at the probe's 268 ms): the
        console visibly stopping dead, which is precisely the symptom this
        module's threading exists to avoid. So ``join=False`` hands the
        wait and the device handback to a throwaway daemon thread and
        returns at once; the console passes it, and quit passes the
        default and waits.

        The join is bounded either way, because a privacy control that can
        be postponed indefinitely by a stuck device is not one -- the same
        ordering of risks ``CameraFeed._close_raw`` settled on.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        self._publish(blank(REASON_DISABLED))
        if thread is None or not thread.is_alive():
            self._close_pipeline()
            return
        if join:
            self._release(thread, timeout)
            return
        threading.Thread(target=self._release, args=(thread, timeout),
                         daemon=True, name="campreview-stop").start()

    def _release(self, thread: threading.Thread, timeout: float) -> None:
        """Wait briefly for the capture thread, and take the device back
        MYSELF only if it did not stop.

        A thread that exits cleanly closes the pipeline on its own way out
        (``_loop``), so closing again from here would be a no-op at best --
        and at worst would close a pipeline a NEW generation had built in
        the gap, since with ``join=False`` this runs concurrently with
        whatever the console does next. The close here is for the wedged
        case only, and it stays unconditional there because a privacy
        control that can be postponed indefinitely by a stuck device is
        not one.

        Off the Tk thread whenever ``stop(join=False)`` was used. Never
        raises: this is the last thing that runs on a teardown path.
        """
        try:
            thread.join(timeout=timeout)
        except Exception:                     # noqa: BLE001 - a teardown path
            log.debug("campreview: the join failed", exc_info=True)
        if not thread.is_alive():
            return
        log.warning("campreview: the capture thread did not stop in %.1fs; "
                    "releasing the device without it", timeout)
        self._close_pipeline()

    def set_enabled(self, enabled: bool) -> None:
        """The settings toggle, live. The value he just chose is passed
        through rather than re-read, so this does not race the drawer's
        write to assistant.json in either direction."""
        if enabled:
            self.start(enabled=True)
        else:
            self.stop()

    # ----------------------------------------------------------- reading
    def latest(self) -> PreviewShot:
        with self._lock:
            return self._shot

    def status(self) -> dict:
        """Numbers and strings. No image, by construction: this is built
        from ``PreviewShot.numbers_only``."""
        shot = self.latest()
        out = shot.numbers_only()
        out.update({"running": self.running, "cycles": self.cycles,
                    "box_w": self.box[0], "box_h": self.box[1]})
        stage = self.hands
        if stage is not None and hasattr(stage, "status"):
            try:
                out["gesture"] = stage.status()
            except Exception:                 # noqa: BLE001 - a diagnostic
                out["gesture"] = {}
        return out

    # ------------------------------------------------------------- guts
    # How often the capture thread logs a one-line summary. Numbers and
    # words only -- it is built from ``numbers_only()``, so there is no
    # code path by which a log line can carry pixels.
    LOG_EVERY_S = 60.0

    def _publish(self, shot: PreviewShot,
                 stop: Optional[threading.Event] = None) -> None:
        """Write the latest-wins slot.

        ``stop`` is the writer's own generation event. A write from a
        SUPERSEDED generation -- the thread of a stop whose join timed out,
        finishing its last pass after the pane has been switched back on --
        is dropped, so it cannot overwrite the live picture with its own
        stale verdict. Callers on the UI thread pass nothing and always
        write.
        """
        if stop is not None and stop is not self._stop:
            return
        with self._lock:
            self._shot = shot
        self._maybe_log(shot)

    def _maybe_log(self, shot: PreviewShot) -> None:
        now = self._now()
        if now - self._logged < self.LOG_EVERY_S:
            return
        self._logged = now
        data = shot.numbers_only()
        face = data.get("face") or {}
        who = ""
        if face.get("id_ran"):
            # A LABEL, not a face. "hunter" is a string he chose; the pixels
            # it was derived from were dropped inside grab() and never
            # reached this dict -- numbers_only() has no image field.
            who = "  id %s %.2f" % (face.get("name") or "unknown",
                                    face.get("id_score") or 0.0)
        log.info("campreview: %s  faces %d  %dx%d  %.1f fps  grab %.0f ms%s",
                 data["reason"] or "live", data["faces"], data["cap_w"],
                 data["cap_h"], data["fps"], data["grab_ms"], who)

    def _sensing_state(self):
        """The policy's state, or the fail-safe when there is no policy.

        Identical to the header badge's rule (``_probe_sensing``): a
        console that shows a live picture because the sensing owner failed
        to construct is the one asserting the exact thing nobody can check.
        """
        from jarvis.ui.sensing_badge import (  # noqa: PLC0415 - lazy, no Tk
            sensing_failsafe_state)
        policy = self.sensing
        if policy is None:
            return sensing_failsafe_state()
        try:
            return policy.state()
        except Exception:                     # noqa: BLE001 - provider boundary
            log.warning("campreview: the sensing owner failed; staying dark",
                        exc_info=True)
            return sensing_failsafe_state()

    def _close_pipeline(self) -> None:
        """Let go of the pipeline, and of whatever device it owns.

        Locked because two threads can reach it at once now: the releaser
        started by ``stop(join=False)`` and the capture thread's own exit
        path. Without the lock both could read the same pipeline and close
        it twice.
        """
        with self._pipe_lock:
            pipe, self._pipeline = self._pipeline, None
        if pipe is not None:
            pipe.close()

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _measure_fps(self, at: float) -> float:
        last, self._last_at = self._last_at, at
        gap = at - last
        return (1.0 / gap) if last and 0.0 < gap < 10.0 else 0.0

    def cycle(self, stop: Optional[threading.Event] = None) -> PreviewShot:
        """One pass, exposed so the suite can drive the loop body without a
        thread. Publishes and returns the shot.

        ``stop`` is the calling generation's event; the loop passes its
        own, and a direct caller gets the current one. It is checked
        TWICE: before anything is opened, and again after the grab
        returns.
        """
        stop = self._stop if stop is None else stop
        self.cycles += 1
        seq = self._next_seq()
        state = self._sensing_state()
        if not camera_allowed(state):
            # Not "grab and discard": the device is not opened, and the
            # pipeline lets go of it -- closing one the preview opened,
            # dropping a borrowed one for its owner to shut. Enforcement
            # before the device is the ruling.
            self._close_pipeline()
            shot = blank(REASON_SENSING, sensing_detail(state), seq=seq,
                         at=time.time())
            self._publish(shot, stop)
            return shot
        if stop.is_set():
            # Switched off between the loop's check and here. No device is
            # opened for a session that is already over.
            shot = blank(REASON_DISABLED, seq=seq, at=time.time())
            self._publish(shot, stop)
            return shot
        if self._pipeline is None:
            with self._pipe_lock:
                if self._pipeline is None:
                    self._pipeline = self._make()
        pipe = self._pipeline
        if pipe is None or getattr(pipe, "feed", None) is None:
            shot = blank(REASON_PIPELINE,
                         getattr(pipe, "reason", "") or
                         REASON_WORDS[REASON_PIPELINE], seq=seq,
                         at=time.time())
            self._publish(shot, stop)
            return shot
        shot = pipe.grab(self.box, seq=seq)
        if stop.is_set():
            # THE SWITCH WENT OFF WHILE THIS GRAB WAS IN FLIGHT. The frame
            # came back after he said stop, so it is dropped here rather
            # than stored in the slot -- "stop() means no frame is held"
            # has to survive a wedged 2 s read, not only a 130 ms one.
            shot = blank(REASON_DISABLED, seq=seq, at=time.time())
        elif shot.live:
            shot = PreviewShot(image=shot.image, faces=shot.faces,
                               cap_w=shot.cap_w, cap_h=shot.cap_h,
                               reason=shot.reason, detail=shot.detail,
                               seq=shot.seq, at=shot.at,
                               fps=self._measure_fps(self._now()),
                               grab_ms=shot.grab_ms, hand=shot.hand)
        self._publish(shot, stop)
        return shot

    def _run(self, stop: threading.Event) -> None:
        """One generation of capture, behind the lease.

        ONE CAPTURE AT A TIME, whatever happened to the last one. A
        ``stop()`` whose bounded join timed out leaves a thread inside a
        blocking read; without this lease it and its successor would be in
        the device together, and ``jarvis/eye.py``'s ``capture()`` has no
        lock -- so a ``release()`` on one thread can land inside a
        ``read()`` on the other, and the counters here are non-atomic
        read-modify-writes besides. The successor waits HERE, off the Tk
        thread, and starts the moment the wedged read returns.
        """
        with self._lease:
            if stop.is_set():                 # stopped while it waited
                return
            self._loop(stop)

    def _loop(self, stop: threading.Event) -> None:
        period = 1.0 / preview_fps(self.get_option)
        log.info("campreview: capturing at up to %.1f fps into a %dx%d box",
                 1.0 / period, self.box[0], self.box[1])
        while not stop.is_set():
            t0 = self._now()
            try:
                self.cycle(stop)
            except Exception:                 # noqa: BLE001 - the loop lives
                log.exception("campreview: a capture cycle failed")
                self._publish(blank(REASON_NO_FRAME, "the capture failed",
                                    seq=self._next_seq(), at=time.time()),
                              stop)
            # Wait on the STOP EVENT rather than sleeping: a toggle-off must
            # not have to wait out a frame period, and a busy-wait on a
            # 130 ms device would burn a core for nothing.
            left = period - (self._now() - t0)
            if self._sleep is not None:
                self._sleep(max(0.0, left))
            elif left > 0:
                stop.wait(left)
        self._close_pipeline()
        log.info("campreview: capture stopped after %d cycles", self.cycles)
