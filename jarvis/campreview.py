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
Measured on his LifeCam Cinema on 2026-09-02: a frame grab costs **130 ms at
p50** (the driver grants 1280x720 MJPG at a nominal 30 fps and delivers ~7.5),
while YuNet costs 2-3 ms. The capture is the bottleneck by a factor of fifty.
The console's reactor animates on 16.67 ms slot boundaries in the Tk mainloop
(jarvis/ui/avatar_clock.py), so a ``VideoCapture.read()`` on that thread would
blow through EIGHT consecutive slots -- the console would visibly stop dead
once per frame, which is a worse version of the 30 fps standby clock he
already called "really laggy" (09-01). So this module owns a daemon thread,
the UI polls a latest-wins slot, and the two never block each other.

The rate is capped deliberately at ``camera.preview_fps`` (default 6, below
the ~7.5 the device can actually deliver). Asking for more does not produce
more frames; it produces a thread that is permanently inside a blocking read,
which is the state in which a curfew edge has to wait CLOSE_WAIT_S to get the
device back.

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
from dataclasses import dataclass
from typing import Any, Callable, Optional

from jarvis.logs import get_logger

log = get_logger("campreview")

# assistant.json keys. The toggle defaults OFF: a camera pane that appeared
# by itself on a box whose owner asked for camera silence would be the
# feature introducing itself by breaking the rule it lives under.
OPTION_ENABLED = "camera.preview"
OPTION_FPS = "camera.preview_fps"
DEFAULT_FPS = 6.0
# Above the device's measured ~7.5 fps there is nothing to gain and a
# blocking read to always be inside; below 1 the pane stops reading as live.
MIN_FPS, MAX_FPS = 1.0, 10.0
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

    def as_dict(self) -> dict:
        return {"conf": self.conf, "x": self.x, "y": self.y, "w": self.w,
                "h": self.h, "yaw_deg": self.yaw_deg,
                "attending": self.attending,
                "landmarks_ok": self.landmarks_ok, "eye_px": self.eye_px}


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


def preview_enabled(get_option: Optional[Callable]) -> bool:
    return bool(_option(get_option, OPTION_ENABLED, False))


def preview_fps(get_option: Optional[Callable]) -> float:
    """The capture rate, clamped. It cannot beat the device (~7.5 fps
    measured), and a rate above that buys a thread permanently inside a
    blocking read instead of more pictures."""
    try:
        fps = float(_option(get_option, OPTION_FPS, DEFAULT_FPS))
    except (TypeError, ValueError):
        fps = DEFAULT_FPS
    return max(MIN_FPS, min(MAX_FPS, fps))


def poll_ms(fps: float) -> int:
    """How often the Tk side should LOOK for a new shot.

    Twice the capture rate, so a frame is on screen within half a period of
    arriving, and never faster than 20 ms. A poll that finds nothing new is
    an int compare (``PreviewShot.seq``), so the extra passes are free; the
    thing that must not happen is a poll on the 16.67 ms reactor slot, and
    83 ms at the default 6 fps is nowhere near it.
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
                 owned: bool = True, hands=None):
        self.feed = feed
        self.owned = bool(owned)
        self.detector = detector
        self.lens = lens
        self.head = head
        self.tracker = tracker
        self.reason = reason
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

    def _faces(self, frame, rows, frame_w: int, frame_h: int) -> tuple:
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
        if rows is None:
            return ()
        observe = self._observer()
        if observe is None:
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
            out.append(PreviewFace(conf=float(obs.conf), x=float(obs.x),
                                   y=float(obs.y), w=float(obs.w),
                                   h=float(obs.h),
                                   yaw_deg=float(obs.yaw_deg),
                                   landmarks_ok=bool(obs.landmarks_ok),
                                   eye_px=float(getattr(obs, "eye_px", 0.0)
                                                or 0.0)))
        out.sort(key=lambda f: f.w * f.h, reverse=True)
        return tuple(self._attend(out[:MAX_FACES]))

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
        faces[0] = PreviewFace(conf=head.conf, x=head.x, y=head.y, w=head.w,
                               h=head.h, yaw_deg=head.yaw_deg,
                               attending=inside,
                               landmarks_ok=head.landmarks_ok,
                               eye_px=head.eye_px)
        return faces

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

        rows = None
        if self.detector is not None:
            try:
                rows = self.detector.detect(frame)
            except Exception:                      # noqa: BLE001 - a detector
                log.debug("campreview: the detector raised", exc_info=True)
                rows = None
        faces = self._faces(frame, rows, frame_w, frame_h)
        # After the faces (attention gates the hand stage), before shrink
        # (the last use of the full frame). One capture, one consumer.
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
    try:
        from jarvis import camera as cam            # noqa: PLC0415 - lazy lane
        from jarvis.visionrig import AttentionTracker  # noqa: PLC0415
        detector, why = cam.detector_from_config(cfg)
        reason = why or reason
        lens = getattr(feed, "lens", None) or cam.lens_from_config(cfg)
        head = cam.head_from_config(cfg)
        th = cam.thresholds_from_config(cfg)
        tracker = AttentionTracker(th.cone_deg, th.cone_hysteresis_deg,
                                   th.cone_centre_deg)
    except Exception as exc:                        # noqa: BLE001
        log.info("campreview: the vision lane is incomplete (%s: %s)",
                 type(exc).__name__, exc)
        reason = reason or "%s: %s" % (type(exc).__name__, exc)
    return PreviewPipeline(feed, detector=detector, lens=lens, head=head,
                           tracker=tracker, reason=reason, owned=owned,
                           hands=hands)


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

        WHAT MAY NOT BE: the join. The thread can be inside a 130 ms grab
        (measured on his LifeCam) or, on a wedged camera, inside one for
        seconds, and this is called on the TK THREAD -- from
        ``_preview_apply`` at every ACTIVE->AMBIENT edge, which is every
        45 s of quiet. 130 ms there is eight consecutive 60 Hz slots: the
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
        log.info("campreview: %s  faces %d  %dx%d  %.1f fps  grab %.0f ms",
                 data["reason"] or "live", data["faces"], data["cap_w"],
                 data["cap_h"], data["fps"], data["grab_ms"])

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
