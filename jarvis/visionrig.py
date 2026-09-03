"""The numbers-only harness: how this pipeline is developed without looking.

Hunter's standing rule is that nothing may look at what his camera sees --
not to debug, not to confirm a detection, not once. A rule like that is only
survivable if the ordinary way to work on the camera ALREADY produces
numbers, so this module is the natural workflow rather than the workaround:
point it at a frame source, and what comes out is detection counts,
confidences, box geometry, derived angles, embedding distances, timings and
a verdict against each threshold. He runs it against his real camera and
pastes the report; nobody ever sees the room.

THREE PROPERTIES MAKE THAT MORE THAN A PROMISE.

1. **The report cannot carry an image.** ``RigReport.to_dict`` is checked by
   ``assert_numbers_only``, which walks the structure and raises on anything
   that is not a scalar, a string or a container of those. An array cannot be
   added to the output by accident, and a test asserts the checker actually
   rejects one.
2. **The rig retains no frame.** Frames are read, reduced to numbers and
   dropped inside the loop; nothing about a frame -- not a crop, not an
   embedding -- survives ``run()`` as an attribute.
3. **"I could not look" is never spelled "I saw nobody".** VSS on this same
   machine shipped a ``region_mode='yunet'`` path that silently fell back to
   head_box and never once executed; the shape of that bug is an unavailable
   detector reporting zero detections. Here a missing model, a broken model
   and an exception mid-run all set ``detector_ok`` False, name themselves in
   ``reason``, and make ``ok`` False. Zero faces with ``ok`` True means the
   room was empty. Zero faces with ``ok`` False means the rig was blind.

WHAT IS A MEASUREMENT HERE AND WHAT IS AN ASSUMPTION. Counts, confidences,
pixel geometry, timings and cosines are measured. The BEARING (angle off the
lens axis) is measured, but only because the caller supplies a real
``facemodels.Lens``: the same pixel is 17.9 deg on the LifeCam and 23.6 deg
on a C930e, which is why no default field of view exists anywhere in this
file. The YAW is a proxy through a stated rigid head model with one assumed
anthropometric constant, and the raw dimensionless ratio is reported beside
the degrees precisely so the assumption stays visible.

Nothing here imports cv2, torch or a model. The detector is a seam
(``jarvis/facedetect.py`` in production, a stub in the suite) and the frame
source is cv2.VideoCapture's own ``read()``/``release()`` interface, so the
whole module runs in the suite with no camera, no weights and no display.
"""
from __future__ import annotations

import math
import numbers
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from jarvis.logs import get_logger

log = get_logger("visionrig")

# YuNet hands back one row per face: x, y, w, h, then five landmarks as
# (x, y) pairs -- right eye, left eye, nose tip, right mouth corner, left
# mouth corner -- then the score. "Right" is the PERSON's right, which is the
# left of the image.
DETECT_COLS = 15
IDX_SCORE = 14
IDX_EYE_R, IDX_EYE_L, IDX_NOSE = 4, 6, 8

# YuNet's documented working range starts around 10 px across. At detect
# width 320 the LifeCam puts a 16 cm face at 95 cm at 42 px and the 90 deg
# Arducam at 27; a mount that lands under this cannot work, and the point of
# checking it explicitly is that the failure would otherwise arrive as
# "detected nothing" with no cause attached.
MIN_DETECT_PX = 10.0

# Nose-tip protrusion over interocular distance. AN ASSUMPTION, not a
# measurement of him: adult interocular is ~6.3 cm and the nose tip sits
# ~2.2 cm forward of the plane through the eye centres, so ~0.35. It is the
# only constant standing between a dimensionless ratio and a number in
# degrees, and its error shows up where it matters most -- at the edge of the
# attention cone a 23% error in it is ~5 deg, a quarter of a 20 deg cone. So
# the rig reports ``yaw_t`` (raw, measured) beside ``yaw_deg`` (derived,
# assumed), and the constant is a config key he can calibrate from the $0
# photo test in docs/vision.md section 9: photograph a known angle, read the
# ratio the rig prints, and solve r = t / tan(angle).
NOSE_RATIO = 0.35


# ------------------------------------------------------------ the guard rail
def assert_numbers_only(obj: Any, path: str = "report") -> None:
    """Raise unless ``obj`` is scalars, strings and containers of those.

    This is the mechanical half of the privacy rule. Reviewing print
    statements does not survive the next edit; a checker that the report must
    pass does, and ``tests/test_visionrig.py`` asserts it really does reject
    an array rather than merely being called.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return
    if isinstance(obj, numbers.Real):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError("%s: key %r is not a string" % (path, key))
            assert_numbers_only(value, "%s.%s" % (path, key))
        return
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            assert_numbers_only(value, "%s[%d]" % (path, i))
        return
    raise TypeError("%s is a %s; the report may only carry numbers, strings "
                    "and containers of them" % (path, type(obj).__name__))


def _pct(values: Sequence[float], q: float) -> float:
    """Percentile with no numpy round trip and no opinion about an empty
    list: nothing measured is 0.0, and the count beside it says which."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


# ------------------------------------------------------------- the geometry
def landmark_geometry(row: Sequence[float]) -> tuple:
    """``(t, roll_deg, eye_px, ok)`` from one detector row, in ITS pixels.

    ``t`` is the nose tip's displacement from the eye midpoint, projected
    onto the interocular direction and divided by the interocular distance.
    Projecting onto that direction rather than onto image x is what makes it
    ROLL-INVARIANT: a head tilted 30 deg must not read as a head turned 30
    deg, and the naive version does exactly that.

    ``ok`` is False when the two eye landmarks coincide -- a degenerate row
    from a bad detection, where every downstream angle would be invented.
    """
    ex = float(row[IDX_EYE_L]) - float(row[IDX_EYE_R])
    ey = float(row[IDX_EYE_L + 1]) - float(row[IDX_EYE_R + 1])
    eye_px = math.hypot(ex, ey)
    if eye_px < 1e-6:
        return 0.0, 0.0, 0.0, False
    ux, uy = ex / eye_px, ey / eye_px
    mx = (float(row[IDX_EYE_L]) + float(row[IDX_EYE_R])) / 2.0
    my = (float(row[IDX_EYE_L + 1]) + float(row[IDX_EYE_R + 1])) / 2.0
    dx = float(row[IDX_NOSE]) - mx
    dy = float(row[IDX_NOSE + 1]) - my
    t = (dx * ux + dy * uy) / eye_px
    roll = math.degrees(math.atan2(ey, ex))
    return t, roll, eye_px, True


@dataclass(frozen=True)
class HeadModel:
    """The rigid head the yaw proxy assumes. See ``NOSE_RATIO``.

    Under yaw the projected interocular distance shrinks by cos(yaw) while
    the nose tip swings by sin(yaw), so the measured ratio is
    ``t = nose_ratio * tan(yaw)`` and the inverse is one atan. Stating it as
    an object rather than a constant is what lets the number be calibrated
    per person and per detector without a second implementation appearing.
    """

    nose_ratio: float = NOSE_RATIO

    def __post_init__(self) -> None:
        if not 0.0 < float(self.nose_ratio) < 2.0:
            raise ValueError("nose_ratio out of range: %r" % (self.nose_ratio,))

    def yaw_deg(self, t: float) -> float:
        return math.degrees(math.atan(float(t) / float(self.nose_ratio)))


@dataclass(frozen=True)
class FaceObservation:
    """One face, as numbers. There is deliberately no crop, no landmark
    array and no embedding on this object: what it holds is what may be
    printed, and that is the whole design."""

    conf: float
    x: float
    y: float
    w: float
    h: float
    cx: float
    cy: float
    detect_px: float       # box width in the pixels the detector saw
    face_px: float         # the same box in capture pixels
    bearing_deg: float     # off the lens axis, from the CONFIGURED fov
    elevation_deg: float
    yaw_t: float           # measured, dimensionless
    yaw_deg: float         # derived through HeadModel -- an assumption
    roll_deg: float
    eye_px: float
    landmarks_ok: bool


def observe(row: Sequence[float], lens, scale_x: float, scale_y: float,
            head: HeadModel) -> FaceObservation:
    """One detector row -> one ``FaceObservation`` in capture pixels.

    ``scale_x``/``scale_y`` map detect space back to capture space. They are
    separate because they can DISAGREE: 320x240 for a 16:9 capture is a 1.33x
    horizontal squash of every face in the frame, and the rig's ``aspect``
    check exists to say so out loud rather than to quietly average them.
    """
    conf = float(row[IDX_SCORE])
    x, y = float(row[0]) * scale_x, float(row[1]) * scale_y
    w, h = float(row[2]) * scale_x, float(row[3]) * scale_y
    cx, cy = x + w / 2.0, y + h / 2.0
    t, roll, eye_px, ok = landmark_geometry(row)
    return FaceObservation(
        conf=conf, x=x, y=y, w=w, h=h, cx=cx, cy=cy,
        detect_px=float(row[2]), face_px=w,
        bearing_deg=lens.offset_deg(cx - lens.width_px / 2.0),
        # Vertical angle through the SAME horizontal scale, because pixels
        # are square: a separate vertical fov would be a second number to get
        # wrong for no gain.
        elevation_deg=lens.offset_deg(cy - lens.height_px / 2.0),
        yaw_t=t, yaw_deg=head.yaw_deg(t) if ok else 0.0,
        roll_deg=roll, eye_px=eye_px * scale_x, landmarks_ok=ok)


# --------------------------------------------------------- the attention cone
class AttentionTracker:
    """Is the head pointed at Jarvis, with hysteresis so a blink is not a no.

    ``centre_deg`` aims the cone OFF the lens axis, which the mount needs:
    docs/vision.md section 9 puts the camera beside the monitor at eye level,
    so "looking at Jarvis" is not necessarily "looking down the lens".

    The yaw fed in is relative to the line from the head TO THE CAMERA -- a
    face pointed at the camera is frontal wherever it sits in the frame -- so
    this is a head-direction test and not a position test. Position is the
    bearing, and it is reported separately.
    """

    def __init__(self, cone_deg: float, hysteresis_deg: float,
                 centre_deg: float = 0.0,
                 now: Callable[[], float] = time.monotonic):
        self.cone_deg = float(cone_deg)
        self.hysteresis_deg = max(0.0, float(hysteresis_deg))
        self.centre_deg = float(centre_deg)
        self._now = now
        self._attending = False
        self._since = 0.0
        self.dwell_s = 0.0

    def update(self, yaw_deg: Optional[float]) -> bool:
        """``None`` means no face: attention ends, and so does the dwell."""
        if yaw_deg is None:
            self._attending = False
            self.dwell_s = 0.0
            return False
        off = abs(float(yaw_deg) - self.centre_deg)
        bar = self.cone_deg + (self.hysteresis_deg if self._attending else 0.0)
        inside = off <= bar
        now = self._now()
        if inside and not self._attending:
            self._since = now
        self._attending = inside
        self.dwell_s = (now - self._since) if inside else 0.0
        return inside


# ------------------------------------------------------------- the verdicts
@dataclass(frozen=True)
class Thresholds:
    """The bars the rig judges against -- all of them from his config, none
    of them invented here."""

    min_conf: float
    cone_deg: float
    cone_hysteresis_deg: float
    dwell_s: float
    identity_min: float
    cone_centre_deg: float = 0.0
    min_detect_px: float = MIN_DETECT_PX
    min_detection_rate: float = 0.5


@dataclass(frozen=True)
class Check:
    """One pass/fail line. ``ok`` is None for NOT APPLICABLE, which is a real
    answer here: a run with no face in frame has nothing to say about the
    confidence bar, and reporting that as a failure would train him to ignore
    the report."""

    name: str
    ok: Optional[bool]
    detail: str

    def as_tuple(self) -> tuple:
        return (self.name, self.ok, self.detail)


@dataclass
class RigReport:
    frames: int = 0
    frames_missed: int = 0
    errors: int = 0
    seconds: float = 0.0
    detector: str = ""
    detector_ok: bool = False
    reason: str = ""
    lens_width: int = 0
    lens_height: int = 0
    lens_hfov_deg: float = 0.0
    detect_width: int = 0
    detect_height: int = 0
    scale_x: float = 0.0
    scale_y: float = 0.0
    faces_total: int = 0
    detected_frames: int = 0
    multi_face_frames: int = 0
    conf_min: float = 0.0
    conf_p05: float = 0.0
    conf_p50: float = 0.0
    conf_p95: float = 0.0
    detect_px_p50: float = 0.0
    face_px_p50: float = 0.0
    eye_px_p50: float = 0.0
    bearing_p50: float = 0.0
    bearing_min: float = 0.0
    bearing_max: float = 0.0
    yaw_t_p50: float = 0.0
    yaw_p50: float = 0.0
    yaw_min: float = 0.0
    yaw_max: float = 0.0
    roll_p50: float = 0.0
    attending_frames: int = 0
    dwell_max_s: float = 0.0
    grab_ms_p50: float = 0.0
    grab_ms_p95: float = 0.0
    detect_ms_p50: float = 0.0
    detect_ms_p95: float = 0.0
    id_ms_p50: float = 0.0
    level_p50: float = 0.0
    contrast_p50: float = 0.0
    id_pairs: int = 0
    id_cosine_min: float = 0.0
    id_cosine_p50: float = 0.0
    id_cosine_max: float = 0.0
    # The GALLERY leg: the same embedding asked "is this him" rather than "is
    # this the same face as a moment ago". Zero everywhere when no identifier
    # was supplied, which is the state until he has enrolled.
    gallery_enrolled: int = 0
    gallery_match_min: float = 0.0
    id_matched_frames: int = 0
    id_unknown_frames: int = 0
    id_gated_out: int = 0
    match_min_score: float = 0.0
    match_p50: float = 0.0
    match_max: float = 0.0
    match_ms_p50: float = 0.0
    checks: tuple = ()

    @property
    def ok(self) -> bool:
        return bool(self.detector_ok and self.errors == 0
                    and all(c.ok is not False for c in self.checks))

    def to_dict(self) -> dict:
        out = {k: v for k, v in vars(self).items() if k != "checks"}
        out["ok"] = self.ok
        out["checks"] = [list(c.as_tuple()) for c in self.checks]
        return out

    def lines(self) -> list:
        """The report as text. Every value on every line is a number, a name
        or a verdict; there is nothing else to print."""
        out = [
            "detector   %s  ok=%s%s" % (self.detector or "-", self.detector_ok,
                                        ("  (%s)" % self.reason)
                                        if self.reason else ""),
            "lens       %dx%d  hfov %.1fdeg  detect %dx%d  scale %.2fx/%.2fy"
            % (self.lens_width, self.lens_height, self.lens_hfov_deg,
               self.detect_width, self.detect_height, self.scale_x,
               self.scale_y),
            "frames     %d read, %d not delivered, %d errors, %.2fs"
            % (self.frames, self.frames_missed, self.errors, self.seconds),
            "faces      %d in %d frames (%d with 2+)"
            % (self.faces_total, self.detected_frames, self.multi_face_frames),
            "confidence min %.3f  p05 %.3f  p50 %.3f  p95 %.3f"
            % (self.conf_min, self.conf_p05, self.conf_p50, self.conf_p95),
            "size       detect %.1fpx  capture %.0fpx  interocular %.0fpx"
            % (self.detect_px_p50, self.face_px_p50, self.eye_px_p50),
            "bearing    p50 %+.1fdeg  range %+.1f..%+.1f"
            % (self.bearing_p50, self.bearing_min, self.bearing_max),
            "yaw        t p50 %+.3f -> p50 %+.1fdeg  range %+.1f..%+.1f"
            % (self.yaw_t_p50, self.yaw_p50, self.yaw_min, self.yaw_max),
            "roll       p50 %+.1fdeg" % self.roll_p50,
            "attention  %d frames, longest dwell %.2fs"
            % (self.attending_frames, self.dwell_max_s),
            "exposure   level p50 %.1f  contrast p50 %.1f"
            % (self.level_p50, self.contrast_p50),
            "timing     grab p50 %.2fms p95 %.2fms  detect p50 %.2fms "
            "p95 %.2fms  id p50 %.2fms"
            % (self.grab_ms_p50, self.grab_ms_p95, self.detect_ms_p50,
               self.detect_ms_p95, self.id_ms_p50),
        ]
        if self.id_pairs:
            out.append("identity   %d pairs  cosine min %.3f p50 %.3f max %.3f"
                       % (self.id_pairs, self.id_cosine_min,
                          self.id_cosine_p50, self.id_cosine_max))
        if self.gallery_enrolled or self.id_matched_frames or \
                self.id_unknown_frames:
            out.append("gallery    %d enrolled  %d matched  %d unknown  "
                       "%d under the detector bar  %.1fms p50"
                       % (self.gallery_enrolled, self.id_matched_frames,
                          self.id_unknown_frames, self.id_gated_out,
                          self.match_ms_p50))
            out.append("match      min %.3f  p50 %.3f  max %.3f  against "
                       "%.3f" % (self.match_min_score, self.match_p50,
                                 self.match_max, self.gallery_match_min))
        for check in self.checks:
            mark = "n/a " if check.ok is None else ("PASS" if check.ok
                                                    else "FAIL")
            out.append("  [%s] %-15s %s" % (mark, check.name, check.detail))
        out.append("VERDICT    %s" % ("ok" if self.ok else "not ok"))
        return out


class SyntheticSource:
    """Frames this process generates, so the rig can be exercised with no
    camera and no lens -- which is the state of this box today.

    WHAT IT IS GOOD FOR, exactly: plumbing, memory, and per-frame COST.
    YuNet is a fixed fully-convolutional graph, so the conv stack does
    identical work on noise and on a face and only the NMS tail scales with
    how many boxes survive, which at a desk is 0-2.

    WHAT IT IS NOT GOOD FOR: any accuracy claim whatsoever. A synthetic face
    is not a face -- a crude drawn one scores 0.33 here, which bounds the
    FALSE-POSITIVE side and nothing else. Whether a real 167 px face clears
    min_conf needs a real camera, and that measurement belongs to Hunter.

    Structured noise rather than flat grey, so any lazy early-out in the
    graph cannot fire and under-report the cost.
    """

    def __init__(self, width: int = 1280, height: int = 720, frames: int = 0,
                 seed: int = 7):
        self.width, self.height = int(width), int(height)
        self.left = int(frames)          # 0 = unlimited
        self.reads = 0
        self.released = 0
        self._rng = np.random.default_rng(seed)
        grad = np.linspace(0, 255, self.width, dtype=np.uint8)
        self._grad = np.tile(grad, (self.height, 1))

    def read(self):
        if self.left and self.reads >= self.left:
            return False, None
        self.reads += 1
        frame = self._rng.integers(0, 256, (self.height, self.width, 3),
                                   dtype=np.uint8)
        frame[:, :, 1] = self._grad
        return True, frame

    def release(self) -> None:
        self.released += 1


# -------------------------------------------------------------------- the rig
def frame_level(frame) -> tuple:
    """``(mean, std)`` over a SUBSAMPLED frame -- two scalars, and the answer
    to "does autoexposure leave any contrast at 95 cm".

    Two numbers over a whole frame are a measurement. A histogram starts to
    be a picture, so there is not one, and the subsample keeps the cost off
    the frame budget as well.
    """
    try:
        small = np.asarray(frame)[::8, ::8]
        return float(small.mean()), float(small.std())
    except Exception:  # noqa: BLE001 - a weird frame must not end a run
        return 0.0, 0.0


def _cosine(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class _Acc:
    conf: list = field(default_factory=list)
    detect_px: list = field(default_factory=list)
    face_px: list = field(default_factory=list)
    eye_px: list = field(default_factory=list)
    bearing: list = field(default_factory=list)
    yaw_t: list = field(default_factory=list)
    yaw: list = field(default_factory=list)
    roll: list = field(default_factory=list)
    grab: list = field(default_factory=list)
    detect: list = field(default_factory=list)
    ident: list = field(default_factory=list)
    level: list = field(default_factory=list)
    contrast: list = field(default_factory=list)
    cosine: list = field(default_factory=list)
    match: list = field(default_factory=list)
    ident_id: list = field(default_factory=list)


class Rig:
    """Run a frame source through a detector and emit numbers.

    ``source`` needs only cv2.VideoCapture's ``read() -> (ok, frame)``, which
    is what ``jarvis/camera.py`` hands over and what a stub provides in the
    suite. ``detector`` needs ``input_size`` and ``detect(frame)`` returning
    the (N, 15) rows OpenCV's FaceDetectorYN returns, or None.

    ``detector=None`` with a ``detector_reason`` is the ABSENT case, and it
    is the one this class is most careful about: the run does not happen at
    all, the reason is carried into the report, and ``ok`` is False. A rig
    that quietly reported "0 faces, all good" with no weights on disk would
    be the VSS bug with a new name.
    ``identifier`` is a ``jarvis.eye.FaceIdentifier`` -- the enrolled gallery,
    asked "is this him". It is separate from ``recogniser`` because the two
    ask different questions: the recogniser leg measures whether consecutive
    frames of the SAME sitting agree (a stability measurement that needs no
    enrolment), the identifier leg measures whether the person in the chair
    matches the gallery on disk. Passing both computes two embeddings per
    frame, which is ~20 ms and is what a bring-up run wants; production passes
    one.
    """

    def __init__(self, source, detector, lens, thresholds: Thresholds, *,
                 recogniser=None, head: Optional[HeadModel] = None,
                 detector_reason: str = "", identifier=None,
                 now: Callable[[], float] = time.monotonic):
        self.source = source
        self.detector = detector
        self.lens = lens
        self.thresholds = thresholds
        self.recogniser = recogniser
        self.identifier = identifier
        self.head = head or HeadModel()
        self.detector_reason = detector_reason
        self._now = now

    # ------------------------------------------------------------- running
    def run(self, frames: int = 60, seconds: Optional[float] = None,
            interval_s: float = 0.0) -> RigReport:
        rep = RigReport(
            lens_width=self.lens.width_px, lens_height=self.lens.height_px,
            lens_hfov_deg=self.lens.hfov_deg,
            detector=getattr(self.detector, "name", "") or
            (type(self.detector).__name__ if self.detector else ""),
        )
        if self.detector is None:
            rep.detector_ok = False
            rep.reason = self.detector_reason or "no detector"
            rep.detector = rep.detector or "none"
            rep.checks = (Check("detector", False, rep.reason),)
            log.warning("visionrig: no detector (%s); not reading any frames",
                        rep.reason)
            return rep

        dw, dh = getattr(self.detector, "input_size", (0, 0))
        rep.detect_width, rep.detect_height = int(dw), int(dh)
        sx = self.lens.width_px / float(dw) if dw else 1.0
        sy = self.lens.height_px / float(dh) if dh else 1.0
        rep.scale_x, rep.scale_y = sx, sy
        rep.detector_ok = True

        tracker = AttentionTracker(self.thresholds.cone_deg,
                                   self.thresholds.cone_hysteresis_deg,
                                   self.thresholds.cone_centre_deg,
                                   now=self._now)
        acc = _Acc()
        ref = None                      # the first embedding, a LOCAL only
        start = self._now()
        read = 0
        for i in range(int(frames)):
            if seconds is not None and self._now() - start >= seconds:
                break
            t0 = self._now()
            try:
                ok, frame = self.source.read()
            except Exception as exc:  # noqa: BLE001
                rep.errors += 1
                rep.detector_ok = False
                rep.reason = rep.reason or "frame source: %s" % exc
                break
            t1 = self._now()
            if not ok or frame is None:
                break
            read += 1
            acc.grab.append((t1 - t0) * 1000.0)
            level, contrast = frame_level(frame)
            acc.level.append(level)
            acc.contrast.append(contrast)

            t2 = self._now()
            try:
                rows = self.detector.detect(frame)
            except Exception as exc:  # noqa: BLE001 - counted, never swallowed
                rep.errors += 1
                rep.detector_ok = False
                if not rep.reason:
                    rep.reason = "%s: %s" % (type(exc).__name__, exc)
                log.debug("visionrig: detect failed on frame %d", i,
                          exc_info=True)
                tracker.update(None)
                continue
            acc.detect.append((self._now() - t2) * 1000.0)

            faces = [observe(row, self.lens, sx, sy, self.head)
                     for row in (rows if rows is not None else [])]
            if faces:
                rep.detected_frames += 1
            if len(faces) >= 2:
                rep.multi_face_frames += 1
            rep.faces_total += len(faces)
            for obs in faces:
                acc.conf.append(obs.conf)
                acc.detect_px.append(obs.detect_px)
                acc.face_px.append(obs.face_px)
                acc.eye_px.append(obs.eye_px)
                acc.bearing.append(obs.bearing_deg)
                if obs.landmarks_ok:
                    acc.yaw_t.append(obs.yaw_t)
                    acc.yaw.append(obs.yaw_deg)
                    acc.roll.append(obs.roll_deg)

            best = max(faces, key=lambda f: f.conf) if faces else None
            attending = tracker.update(best.yaw_deg if best is not None and
                                       best.landmarks_ok else None)
            if attending:
                rep.attending_frames += 1
                rep.dwell_max_s = max(rep.dwell_max_s, tracker.dwell_s)

            # IDENTITY IS GATED ON THE DETECTION, HARD. SFace collapses on
            # out-of-distribution input -- unrelated non-face crops match each
            # other at cosine 0.66-0.92, i.e. CONFIDENTLY above the 0.363
            # "same person" bar (jarvis/facemodels.py) -- so the embedding is
            # not a second opinion on whether this is a face and must never
            # be computed from one that did not already clear min_conf.
            if (self.recogniser is not None or self.identifier is not None) \
                    and best is not None and \
                    best.conf >= self.thresholds.min_conf:
                row = rows[int(np.argmax([r[IDX_SCORE] for r in rows]))]
                if self.recogniser is not None:
                    t3 = self._now()
                    try:
                        vec = self.recogniser.embed(frame, row)
                    except Exception as exc:  # noqa: BLE001
                        rep.errors += 1
                        if not rep.reason:
                            rep.reason = "recogniser: %s" % exc
                        vec = None
                    acc.ident.append((self._now() - t3) * 1000.0)
                    if vec is not None:
                        if ref is None:
                            ref = np.asarray(vec, dtype=np.float32).ravel()
                        else:
                            acc.cosine.append(_cosine(
                                np.asarray(vec, dtype=np.float32).ravel(),
                                ref))
                if self.identifier is not None:
                    # The gallery leg. ``identify`` re-checks the same
                    # confidence bar underneath this branch: identity may
                    # never be computed from a detection that did not clear
                    # it, and that rule does not get to depend on the caller.
                    t4 = self._now()
                    label, score = self.identifier.identify(frame, row)
                    acc.ident_id.append((self._now() - t4) * 1000.0)
                    if label:
                        rep.id_matched_frames += 1
                        acc.match.append(float(score))
                    else:
                        rep.id_unknown_frames += 1
            frame = None            # the frame does not outlive its iteration
            if interval_s:
                time.sleep(interval_s)

        rep.frames = read
        rep.frames_missed = max(0, int(frames) - read)
        rep.seconds = max(0.0, self._now() - start)
        ref = None
        self._summarise(rep, acc)
        rep.checks = tuple(self._checks(rep))
        return rep

    # --------------------------------------------------------- summarising
    def _summarise(self, rep: RigReport, acc: _Acc) -> None:
        rep.conf_min = min(acc.conf) if acc.conf else 0.0
        rep.conf_p05 = _pct(acc.conf, 0.05)
        rep.conf_p50 = _pct(acc.conf, 0.50)
        rep.conf_p95 = _pct(acc.conf, 0.95)
        rep.detect_px_p50 = _pct(acc.detect_px, 0.50)
        rep.face_px_p50 = _pct(acc.face_px, 0.50)
        rep.eye_px_p50 = _pct(acc.eye_px, 0.50)
        rep.bearing_p50 = _pct(acc.bearing, 0.50)
        rep.bearing_min = min(acc.bearing) if acc.bearing else 0.0
        rep.bearing_max = max(acc.bearing) if acc.bearing else 0.0
        rep.yaw_t_p50 = _pct(acc.yaw_t, 0.50)
        rep.yaw_p50 = _pct(acc.yaw, 0.50)
        rep.yaw_min = min(acc.yaw) if acc.yaw else 0.0
        rep.yaw_max = max(acc.yaw) if acc.yaw else 0.0
        rep.roll_p50 = _pct(acc.roll, 0.50)
        rep.grab_ms_p50 = _pct(acc.grab, 0.50)
        rep.grab_ms_p95 = _pct(acc.grab, 0.95)
        rep.detect_ms_p50 = _pct(acc.detect, 0.50)
        rep.detect_ms_p95 = _pct(acc.detect, 0.95)
        rep.id_ms_p50 = _pct(acc.ident, 0.50)
        rep.level_p50 = _pct(acc.level, 0.50)
        rep.contrast_p50 = _pct(acc.contrast, 0.50)
        rep.id_pairs = len(acc.cosine)
        rep.id_cosine_min = min(acc.cosine) if acc.cosine else 0.0
        rep.id_cosine_p50 = _pct(acc.cosine, 0.50)
        rep.id_cosine_max = max(acc.cosine) if acc.cosine else 0.0
        rep.match_min_score = min(acc.match) if acc.match else 0.0
        rep.match_p50 = _pct(acc.match, 0.50)
        rep.match_max = max(acc.match) if acc.match else 0.0
        rep.match_ms_p50 = _pct(acc.ident_id, 0.50)
        if self.identifier is not None:
            st = self.identifier.status()
            rep.gallery_enrolled = int(st.get("enrolled", 0))
            rep.gallery_match_min = float(st.get("match_min", 0.0))
            rep.id_gated_out = int(st.get("gated_out", 0))

    def _checks(self, rep: RigReport) -> list:
        th = self.thresholds
        # The reason belongs to whichever thing failed. Printing it on the
        # detector line when the RECOGNISER was what raised reads as "the
        # detector is broken, PASS", which is exactly the kind of report
        # nobody believes twice.
        out = [Check("detector", rep.detector_ok,
                     rep.reason if not rep.detector_ok else
                     "%s ran on %d frames" % (rep.detector or "detector",
                                              rep.frames))]
        out.append(Check("errors", rep.errors == 0,
                         "%d error(s) during the run%s"
                         % (rep.errors, (": " + rep.reason)
                            if (rep.errors and rep.detector_ok and rep.reason)
                            else "")))
        squash = (rep.scale_x / rep.scale_y) if rep.scale_y else 0.0
        out.append(Check(
            "aspect", abs(squash - 1.0) <= 0.01,
            "detect %dx%d against capture %dx%d is a %.2fx horizontal squash"
            % (rep.detect_width, rep.detect_height, rep.lens_width,
               rep.lens_height, squash)))
        if rep.faces_total == 0:
            out.append(Check("detection_rate", None,
                             "no face in any of %d frames -- an empty room "
                             "and a blind detector look the same here, so "
                             "read the detector line above" % rep.frames))
            out.append(Check("min_conf", None, "no face to score"))
            out.append(Check("detect_px", None, "no face to measure"))
            return out
        rate = rep.detected_frames / float(rep.frames or 1)
        out.append(Check("detection_rate", rate >= th.min_detection_rate,
                         "%d of %d frames (%.0f%%) against a %.0f%% bar"
                         % (rep.detected_frames, rep.frames, rate * 100.0,
                            th.min_detection_rate * 100.0)))
        out.append(Check("min_conf", rep.conf_p50 >= th.min_conf,
                         "p50 %.2f, p05 %.2f, min %.2f against a %.2f bar"
                         % (rep.conf_p50, rep.conf_p05, rep.conf_min,
                            th.min_conf)))
        out.append(Check("detect_px", rep.detect_px_p50 >= th.min_detect_px,
                         "%.1f px across at the detector, against YuNet's "
                         "~%.0f px floor" % (rep.detect_px_p50,
                                             th.min_detect_px)))
        if rep.attending_frames:
            out.append(Check("dwell", rep.dwell_max_s >= th.dwell_s,
                             "longest dwell %.2fs against %.2fs"
                             % (rep.dwell_max_s, th.dwell_s)))
        else:
            out.append(Check("dwell", None,
                             "no frame put a head inside the %.0f deg cone"
                             % th.cone_deg))
        if rep.id_pairs:
            out.append(Check("identity", rep.id_cosine_p50 >= th.identity_min,
                             "%d pairs, cosine p50 %.3f against %.3f"
                             % (rep.id_pairs, rep.id_cosine_p50,
                                th.identity_min)))
        if self.identifier is not None:
            seen = rep.id_matched_frames + rep.id_unknown_frames
            if not rep.gallery_enrolled:
                out.append(Check("gallery", None,
                                 "nothing enrolled -- run "
                                 "scripts/face_enrol.py"))
            elif not seen:
                out.append(Check("gallery", None,
                                 "%d embeddings enrolled, but no detection "
                                 "cleared the %.2f bar to be identified from"
                                 % (rep.gallery_enrolled, th.min_conf)))
            else:
                out.append(Check(
                    "gallery", rep.id_matched_frames > rep.id_unknown_frames,
                    "%d of %d identified frames matched the gallery at "
                    "cosine p50 %.3f (min %.3f) against a %.3f bar; %d "
                    "detections were under the %.2f detector bar and had no "
                    "embedding computed at all"
                    % (rep.id_matched_frames, seen, rep.match_p50,
                       rep.match_min_score, rep.gallery_match_min,
                       rep.id_gated_out, th.min_conf)))
        return out
