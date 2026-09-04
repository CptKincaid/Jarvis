"""Reach out, close your hand on something, throw it. Coordinates only.

He asked for a physical metaphor: reach at the screen, grab, fling it at
another machine. This module is the whole decision layer for that, and it
holds NO IMAGE. It takes 21 landmark coordinates per hand and one face
scale, and it emits three events -- ``grab``, ``throw``, ``drop``. There is
no cv2 here, no camera, no thread, no file, no clock but the injected one.
That is the same posture as ``jarvis/eye.py`` and ``jarvis/visionrig.py``,
and it is what lets the entire gesture be developed and validated without
anyone ever looking at what the lens saw. ``tests/test_gesture.py`` pins it
at source level: the vocabulary of pixels is forbidden in this file.

THE CAMERA IS 7.5 fps AND THAT SHAPES EVERY DECISION HERE. A throw lasts
250-400 ms, which is two or three samples. Nothing in this file needs a
velocity curve, a smoothed track or a Kalman filter, because none of those
can exist on three points. Every counter is in FRAMES, not seconds, because
at 7.5 fps a "300 ms dwell" is 2.25 frames and the rounding decides whether
the feature works at all. There are two wall-clock backstops, and they exist
only to stop a wedged capture thread or a starved frame rate leaving a carry
alive holding something: three frame periods of silence resets the machine,
and a carry older than ``carry_max_s`` is dropped whatever the frame count.

THE SCALE IS ``palm_diag``, NOT PALM LENGTH, AND THAT IS MEASURED. Over a
75-pose rotation sweep of a synthetic hand at a fixed 500 mm:

    measure                min      p50      max    cv%   change on closing
    palm_len   (0-9)      45.6    173.2    217.5   30.4     0.0%
    palm_w     (5-17)     29.6    112.6    140.2   17.4     0.0%
    palm_diag            121.8    201.6    258.1   19.4     0.0%   <- CHOSEN
    maxspan (all 21)     199.8    299.5    430.6   24.9   -35.3%   disqualified
    bbox_diag            203.8    335.3    478.9   24.3   -37.3%   disqualified

``palm_diag`` is the only candidate that neither collapses when the hand
tips toward the lens (2.1x range, against palm length's 4.8x) nor moves when
the hand closes -- and the depth cue has to read the same distance either
side of the grab, so a measure that shrinks 37% on closing is not a depth
cue, it is a fist detector in disguise. One hand-unit is one ``palm_diag``,
which is 127 mm on standard adult anthropometry: hypot(107 mm palm length,
68 mm MCP breadth). That anthropometry is GUESSED, not measured on him.

WHY ``gesture_geom.classify()`` IS NOT USED FOR THE FIST. The scratch
classifier normalises every finger extension by ``|lm9 - lm0|`` -- palm
length, the least stable measure in the table above -- so when the hand tips
toward the lens the normaliser collapses to 46 px and the ratios blow up.
Measured: a full fist is classified FIST in 10 of 75 poses. It is replaced
here by one scalar, C = mean(|tip_i - wrist|) / palm_diag over the four
fingers, which over the same pose envelope gives OPEN min 0.919 against FIST
max 0.680 -- a +0.239 gap the 0.70/0.85 bars sit inside, with a deliberate
dead band between them so a hand hovering at the boundary cannot chatter.

THE MIRROR IS THE TRAP IN THIS FILE, AND IT HAS ALREADY BITTEN TWICE. The
feed is NOT mirrored (``cv2.flip`` appears nowhere in jarvis/ or scripts/,
and campreview's shrink() only resizes and swaps BGR->RGB), and the LifeCam
faces him, so image +x is HIS LEFT. Every public direction in this module is
in HIS frame, and exactly ONE function -- ``to_his_frame`` -- is permitted
to change the sign of a coordinate. During the design pass the sign was got
wrong twice inside an hour, and both times the output looked entirely
plausible: correct magnitudes, believable angles, wrong side of the room. A
test that checks ``|B - A|`` passes with the sign inverted, so the tests here
assert a NAMED DIRECTION on a trajectory whose direction is in its own name.

THE CANCELS, and where each lives. Opening the hand where it is (a
release under one hand-unit of travel), pulling it back STILL CLOSED
(``reach_exit``, re-tested on every carry frame), a fling at the desk (the
DOWN sector), the spoken word and the chip click (``cancel()``), the frame
cap and the wall-clock cap (``_carry_expired``, and ``sweep()`` when no
frame arrives to test it). Every one of them is a ``drop``; none can throw.

THE THROW AXIS IS HORIZONTAL, AND THE FIELD OF VIEW FORCED THAT. At a
425-450 mm reach the frame is 4.6 hand-units wide but only 2.6 tall, so a
centred anchor has 2.29 units of lateral room and 1.29 vertical -- and an
off-centre one can have under 0.3. Measured over the design sweep: lateral
throws 48/48 correct at both 7.5 and 6.0 fps with zero wrong directions,
while the SAME vertical throw scored 12/12 or 2/12 depending only on which
way the fingers pointed. So LEFT and RIGHT are the only sectors that produce
a ``throw`` event. DOWN is the cancel: a fling toward the desk is a ``drop``
that says so. UP is measured, reported in ``toward`` and not a target. A
fling the machine cannot name confidently becomes a DROP, which is the
cheap, reversible outcome.

WHAT THIS MODULE WILL NOT DO. It never calls ``deliver()``. A grab picks a
payload up and a drop puts it back, but choosing where a throw LANDS is
config, and sending it is an outward action that his house rules say must be
proposed and confirmed out loud. The throw event names the sector and the
payload; ``jarvis/cast.py`` decides what that means. Subject resolution
(HELD > DOCUMENT > TRACK > SCREEN) lives there too; this module only knows a
``Payload`` with ``pick_up`` / ``put_back`` and an optional ``prepare`` that
is called on the reach so the name can be resolved speculatively and be
ready the instant the fist closes. A gesture is a much weaker statement of
intent than a sentence, and this is the file where that asymmetry is paid
for.

EVERY THRESHOLD BELOW IS A CALIBRATED STARTING POINT, NOT A VALIDATED VALUE.
They come from arithmetic over a synthetic hand whose proportions match the
official MediaPipe reference to within 1.5% and from his own lens constants.
The detector's real landmark noise, his lighting, his sleeve, his reach and
motion blur at a 133 ms exposure are all absent from that. Motion blur in
particular is modelled NOWHERE, which is why a throw is allowed to end by
the hand simply disappearing near an edge.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from jarvis.logs import get_logger

log = get_logger("gesture")

# MediaPipe's 21-point hand. The four finger tips whose distance from the
# wrist collapses when the hand closes, and the five points whose centroid is
# the palm -- wrist plus the four finger MCP knuckles. The thumb is in
# NEITHER: its tip barely moves between an open hand and a fist (it folds
# across rather than in), so including it would flatten the very gap the
# closed scalar is built on.
TIPS = (8, 12, 16, 20)
PALM = (0, 5, 9, 13, 17)
WRIST = 0
MID_MCP = 9
IDX_MCP = 5
PNK_MCP = 17

# A hand-unit is one palm_diag. MEASURED on the synthetic hand; the
# millimetre figure behind it is GUESSED standard adult anthropometry
# (palm length 107 mm, MCP breadth 68.2 mm -> hypot 127 mm), NOT measured on
# him. Every distance this module reports is in hand-units precisely so that
# a wrong millimetre figure cannot corrupt a decision -- it only changes what
# a unit means in the world.
HAND_UNIT_MM = 127.0
# palm_diag / interocular for the same anthropometry: 127 / 63. The reach
# ratio R is scale-free BECAUSE of this -- R = 2.01 * k(pose) * Zface/Zhand.
# His own proportions could move it by +/-15%, which is why reach_min has
# margin measured into it rather than being set to the nominal.
PALM_DIAG_OVER_IPD = 2.01

# Below this a landmark row is degenerate -- two coincident points, a
# collapsed detection -- and every ratio computed from it is invented. It is
# reported as ``ok=False``, never as a very small hand.
MIN_PALM_DIAG_PX = 1e-6

# The frame rate every frame counter below was designed at. The camera
# delivers ~7.5 fps (measured 09-02, 133 ms per frame) and camera.preview_fps
# defaults to 6.0; the counters are correct across that band and are scaled
# from this reference outside it (see CastThresholds.for_fps).
DESIGN_FPS = 7.5
DESIGN_FPS_BAND = (5.5, 8.0)

# Below this much travel a carry-end has no direction worth naming in
# ``toward``: the landmark noise floor is 0.05-0.09 units per frame (frames
# lane), so a vector shorter than this is noise wearing a compass.
TOWARD_MIN_U = 0.15

# Frame edges and the four sectors, in HIS frame. Sector centres are the
# bearings to_his_frame produces for the four cardinal directions.
SECTORS = (("right", 0.0), ("up", 90.0), ("left", 180.0), ("down", 270.0))


# --------------------------------------------------------------- geometry
@dataclass(frozen=True)
class HandObservation:
    """One hand, one frame, reduced to scalars. No array survives this.

    ``reach`` is 0.0 when there was no face in the frame, and 0.0 means NO
    OPINION, never "infinitely close". That is the same three-valued contract
    ``eye.py`` and ``roomsensor.py`` already use, and it matters here because
    the alternative -- treating a missing face as an unbounded reach -- turns
    every fist in the room into a grab the moment the face detector blinks.
    """

    palm_diag: float
    closed: float
    reach: float
    cx: float
    cy: float
    conf: float = 1.0
    ok: bool = True

    def numbers_only(self) -> dict:
        return {"palm_diag": round(self.palm_diag, 3),
                "closed": round(self.closed, 4),
                "reach": round(self.reach, 4),
                "cx": round(self.cx, 2), "cy": round(self.cy, 2),
                "conf": round(self.conf, 4), "ok": bool(self.ok)}


def observe_hand(lm, eye_px: float, conf: float = 1.0) -> HandObservation:
    """21 landmarks in CAPTURE pixels -> the four scalars that decide.

    ``eye_px`` is the interocular distance from the SAME frame, in the same
    capture pixels -- ``visionrig.observe()`` already scales it back out of
    detect space, so use ``obs.eye_px`` and do not re-derive it. Passing 0.0
    (no face this frame) yields ``reach == 0.0``, which every caller reads as
    "no opinion about depth". A face seen within the last few seconds may be
    reused by the caller as a stale median; older than that is a miss.
    """
    arr = np.asarray(lm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 21 or arr.shape[1] < 2:
        raise ValueError("landmarks must be (>=21, >=2); got %r"
                         % (arr.shape,))
    pts = arr[:21, :2]
    if not np.all(np.isfinite(pts)):
        raise ValueError("landmarks contain NaN or inf")
    wrist = pts[WRIST]
    palm_len = float(np.linalg.norm(pts[MID_MCP] - wrist))
    palm_w = float(np.linalg.norm(pts[PNK_MCP] - pts[IDX_MCP]))
    palm_diag = math.hypot(palm_len, palm_w)
    cx, cy = (float(v) for v in pts[list(PALM)].mean(axis=0))
    if palm_diag < MIN_PALM_DIAG_PX:
        # Do not invent C or R from a collapsed hand. Say so instead.
        return HandObservation(0.0, 0.0, 0.0, cx, cy, float(conf), False)
    reach_span = float(np.mean(
        [np.linalg.norm(pts[t] - wrist) for t in TIPS]))
    closed = reach_span / palm_diag
    reach = (palm_diag / float(eye_px)) if float(eye_px) > 0.0 else 0.0
    return HandObservation(palm_diag, closed, reach, cx, cy,
                           float(conf), True)


def to_his_frame(dx: float, dy: float,
                 mirrored: bool = False) -> tuple[float, float]:
    """A sensor delta -> (his_right, his_up). THE ONLY SIGN FLIP THERE IS.

    The lens faces him, so a hand moving toward image +x is moving to HIS
    LEFT, and image +y is downward in every capture buffer, so it is HIS
    DOWN. Mirroring the feed (``camera.mirrored``) flips only the horizontal.
    Nothing else in this module -- or anywhere else -- may negate a
    coordinate; that rule is the only defence against the failure that
    produced correct magnitudes and the wrong side of the room, twice.
    """
    if mirrored:
        return float(dx), -float(dy)
    return -float(dx), -float(dy)


def bearing_deg(his_right: float, his_up: float) -> float:
    """0 = his right, 90 = his up, 180 = his left, 270 = his down."""
    return math.degrees(math.atan2(float(his_up),
                                   float(his_right))) % 360.0


def _turn_between(a: float, b: float) -> float:
    """The unsigned angle between two bearings, 0..180 degrees."""
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def sector(bearing: float, half_deg: float = 35.0) -> str:
    """'right' | 'up' | 'left' | 'down' | 'ambiguous'.

    Four +/-35 deg sectors with four 20 deg gaps between them. The gaps are
    not wasted: a fling the machine cannot name confidently must become a
    DROP, and being unable to name it is exactly what the gap means.
    """
    b = float(bearing) % 360.0
    for name, centre in SECTORS:
        if abs((b - centre + 180.0) % 360.0 - 180.0) <= float(half_deg):
            return name
    return "ambiguous"


def edge_bearing(cx: float, cy: float, frame_w: float, frame_h: float,
                 mirrored: bool = False) -> tuple[str, float, float]:
    """Which edge the palm is nearest, how near, and where that points.

    ``(edge_name, nearest_fraction, bearing_deg)``. The fraction is of a HALF
    field, so 0.0 is on the edge and 1.0 is dead centre; it is clamped at
    zero for a centroid the model placed just outside the picture. The edge
    name and bearing are in HIS frame, derived by pushing the edge's outward
    normal through ``to_his_frame`` rather than by writing the mapping out --
    a second copy of that mapping is precisely how the sign gets inverted.
    """
    half_w = max(float(frame_w) / 2.0, 1e-9)
    half_h = max(float(frame_h) / 2.0, 1e-9)
    # (fraction of a half-field, outward normal in SENSOR coordinates)
    edges = (
        (max(float(cx), 0.0) / half_w, (-1.0, 0.0)),                # image L
        (max(float(frame_w) - float(cx), 0.0) / half_w, (1.0, 0.0)),
        (max(float(cy), 0.0) / half_h, (0.0, -1.0)),                # image T
        (max(float(frame_h) - float(cy), 0.0) / half_h, (0.0, 1.0)),
    )
    frac, normal = min(edges, key=lambda e: e[0])
    bearing = bearing_deg(*to_his_frame(normal[0], normal[1], mirrored))
    return sector(bearing), float(frac), bearing


# ----------------------------------------------------------- the machine
class CastState(str, Enum):
    IDLE = "idle"
    REACHING = "reaching"
    CLOSING = "closing"
    CARRYING = "carrying"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class CastThresholds:
    """Every number, where it came from, and whether it was measured.

    MEASURED means measured this design pass, by arithmetic over a synthetic
    hand and his own lens constants -- not against his room, and not against
    the detector's real noise. Treat them as a calibrated starting point.
    Frame counters are for the 5.5-8 fps band the camera actually delivers;
    ``for_fps`` rescales them for a faster feed and never loosens them for a
    slower one.
    """

    # MEASURED. Over the realistic pose envelope a fist reads C <= 0.680 and
    # an open hand C >= 0.919. 0.70 and 0.85 sit inside that gap with a dead
    # band between, so a hand at the boundary cannot flicker open/closed.
    closed_max: float = 0.70
    open_min: float = 0.85
    # MEASURED. R = palm_diag_px / eye_px. A hand at the FACE PLANE never
    # exceeded 2.03 in either sweep, so 2.35 cannot be reached without
    # actually reaching. It is also the first bar in the sweep with zero
    # false grabs AND full lateral recall: at 2.20 the everyday set produced
    # 6 false grabs in 288, at 2.35 it produced 0. It is tested on EVERY
    # dwell frame, as the measured harness did, so a fist that closes at
    # reach and then pulls back along the lens axis cannot grab.
    reach_min: float = 2.35
    reach_arm: float = 1.60
    # MEASURED (adversarial pass, 09-03). The reach is re-tested DURING the
    # carry, not only during the dwell: a fist still closed below this bar
    # is a WITHDRAWAL and the carry ends as a drop, never a throw. Without
    # it the promised cancel "a fist that leaves the reach zone still
    # closed is a drop" did not exist: a retraction to the lap or the
    # shoulder (z 700 mm) with >= 1 hand-unit of sideways image travel,
    # opened there, THREW in 32 of 112 synthetic carries. Swept through
    # the wired path at 1.75/1.85/1.95/2.00/2.05: 1.95 (0.83 of reach_min,
    # ~550 mm against the 425 mm grab) is the highest bar that leaves
    # lateral recall IDENTICAL to no bar at all (24/24 at 6.0 and 7.5 fps
    # with 0/4/8 px of landmark noise) and it takes the grid to 0/112 and a
    # shoulder-height retraction to 600 mm to 0/6; at 2.00 recall starts to
    # slip (23/24 at 6 fps, 8 px). A pull-back of only ~125 mm with a
    # 1.4-unit sideways fling still throws, and is ambiguous by eye too.
    reach_exit: float = 1.95
    # MEASURED in frames, deliberately. 3 hand-bearing closed frames is
    # 267 ms of elapsed time at 7.5 fps and 333 ms at 6.0 (the design note's
    # 400/500 ms counts the latch frame's own period as well). Seconds here
    # would round to 2 or 3 frames depending on jitter.
    dwell_frames: int = 3
    # The hand must have been SEEN OPEN recently, or a resting fist becomes a
    # grab the moment it drifts into the reach zone. 12 frames is 1.6 s at
    # 7.5 fps -- long enough to survive the detector missing a frame or two,
    # short enough that a fist held on his chin cannot inherit an open frame
    # from a minute ago.
    open_lookback_frames: int = 12
    open_frames_req: int = 1
    # MEASURED against the noise floor: landmark jitter is 0.05-0.09 units
    # per frame, so 0.60 is roughly 7x the noise and still small enough that
    # a hand already moving cannot dwell.
    anchor_drift_u: float = 0.60
    # MEASURED. Three exits, three bars. Opening the hand where it is must
    # never be a throw -- he opens his hand hundreds of times an hour -- so a
    # release in frame needs a full hand-width of travel. Leaving the picture
    # is itself the evidence of intent, so that bar drops to 0.25. Vanishing
    # in open space is the weakest evidence and needs both distance AND to
    # have been moving when we lost it.
    throw_release_u: float = 1.00
    throw_exit_u: float = 0.25
    throw_lost_u: float = 0.50
    exit_step_u: float = 0.35
    edge_frac: float = 0.30
    # 2 frames is 267 ms at 7.5 fps: long enough to ride out one missed
    # detection, short enough that an unexplained gap ends the carry rather
    # than persisting it. Failing safe here means dropping, not holding.
    lost_grace_frames: int = 2
    # The carry cap, in frames AND in seconds. Frames is the design (30 =
    # 4.0 s at 7.5 fps, 5.0 s at 6.0); the seconds are the wall-clock backstop
    # his brief asked for, so a starved frame rate cannot hold a carry open
    # for a quarter of a minute. Whichever comes first ends it.
    carry_max_frames: int = 30
    carry_max_s: float = 8.0
    # Association during a carry. 2.50 units is a long way for a hand to
    # travel in 133 ms; it is loose on purpose, because the alternative is a
    # carry that dies when he raises his other hand.
    assoc_max_u: float = 2.50
    cooldown_frames: int = 8
    sector_half_deg: float = 35.0
    # Two hands out at the lens is not this gesture. GUESSED bar, chosen so a
    # hand clearly further back cannot veto the one he is reaching with.
    second_hand_frac: float = 0.70
    # The sectors that produce a THROW. Everything else that clears a bar is
    # a named DROP: down is the cancel, up has no room (measured 12/12 or
    # 2/12 on finger direction alone) and no target.
    target_sectors: tuple = ("left", "right")

    @classmethod
    def for_fps(cls, fps: float, base: Optional["CastThresholds"] = None
                ) -> "CastThresholds":
        """The same machine at another frame rate.

        Inside the 5.5-8 fps band the thresholds are returned untouched:
        they were designed and measured there. ABOVE it every frame counter
        is scaled up from the 7.5 fps reference (dwell, grace and lookback
        rounded up, cooldown and carry cap to nearest) and the one per-frame
        velocity bar, ``exit_step_u``, is scaled DOWN by the same ratio --
        0.35 units per frame at 7.5 fps is 2.6 units per second, and at
        15 fps the same throw moves half as far between samples. MEASURED:
        without that scaling 3 of 48 lateral throws at 15 fps became drops
        with a last step of 0.34. BELOW the band nothing is lowered: a
        counter is never taken under its design value, because at 5 fps a
        2-frame dwell let 8 of 288 everyday motions through (measured) and
        at 2 fps a natural-speed throw is between samples whatever the
        counters say -- the wall-clock carry cap is what bounds a carry
        there.
        """
        base = base or cls()
        fps = float(fps)
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if fps <= DESIGN_FPS_BAND[1]:
            return base
        k = fps / DESIGN_FPS

        def up(n: int) -> int:
            return max(int(n), int(math.ceil(n * k - 1e-9)))

        def near(n: int) -> int:
            return max(int(n), int(round(n * k)))

        return replace(
            base,
            dwell_frames=up(base.dwell_frames),
            open_lookback_frames=up(base.open_lookback_frames),
            lost_grace_frames=up(base.lost_grace_frames),
            cooldown_frames=near(base.cooldown_frames),
            carry_max_frames=near(base.carry_max_frames),
            exit_step_u=float(base.exit_step_u) / k,
        )


@dataclass(frozen=True)
class CastEvent:
    """One of exactly three things: grab, throw, drop. Scalars and strings.

    ``sector`` is set ONLY on a throw and is the field a cast layer may route
    on. ``toward`` is informational: the measured sector on any carry end,
    including a drop, so the pane and the earcon can say "down -- put back"
    without a router ever mistaking it for a target. There is no image field
    and there cannot be one: ``numbers_only()`` is what goes into a log
    line, the console card and the self-check report, and
    ``visionrig.assert_numbers_only`` is asserted against it in the suite.
    """

    kind: str
    at: float
    frame: int
    dist_u: float = 0.0
    bearing_deg: float = 0.0
    sector: str = ""
    why: str = ""
    reach: float = 0.0
    closed: float = 0.0
    payload: str = ""
    toward: str = ""

    def numbers_only(self) -> dict:
        return {"kind": self.kind, "at": round(float(self.at), 4),
                "frame": int(self.frame),
                "dist_u": round(float(self.dist_u), 4),
                "bearing_deg": round(float(self.bearing_deg), 2),
                "sector": self.sector, "why": self.why,
                "reach": round(float(self.reach), 4),
                "closed": round(float(self.closed), 4),
                "payload": self.payload, "toward": self.toward}


class Payload(Protocol):
    """What the hand is holding. Opaque to this module.

    ``pick_up`` is called at the GRAB INSTANT and returns ``(handle, spoken
    name)`` so the console can name what he caught in the same cycle as the
    earcon -- a grab that does not say WHAT it grabbed is a grab he cannot
    trust. Returning ``None`` means there was nothing to pick up, and the
    machine then refuses to enter a carry at all, which turns a large share
    of false grabs into a single tone with nothing armed behind it.

    ``prepare`` is optional and is called when the reach begins, 300-400 ms
    before any grab can fire: the cast layer uses it to resolve the subject
    speculatively (a Spotify call, a window title) so ``pick_up`` is instant.
    """

    def pick_up(self) -> Optional[tuple[str, str]]: ...

    def put_back(self, handle: str) -> None: ...


class CallablePayload:
    """A Payload from two or three callables -- the adapter the wiring layer
    uses to plug ``jarvis/cast.py``'s subject ladder in without this module
    importing it.

    ``pick_up`` returns ``(handle, spoken_name)`` or ``None``; ``put_back``
    takes the handle; ``prepare`` takes nothing. Any of them may be omitted.
    """

    def __init__(self, pick_up: Optional[Callable[[], Optional[tuple]]] = None,
                 put_back: Optional[Callable[[str], None]] = None,
                 prepare: Optional[Callable[[], None]] = None) -> None:
        self._pick_up = pick_up
        self._put_back = put_back
        self._prepare = prepare

    def prepare(self) -> None:
        if self._prepare is not None:
            self._prepare()

    def pick_up(self) -> Optional[tuple[str, str]]:
        if self._pick_up is None:
            return None
        got = self._pick_up()
        if not got:
            return None
        handle, name = got[0], got[1]
        return str(handle), str(name)

    def put_back(self, handle: str) -> None:
        if self._put_back is not None:
            self._put_back(handle)


class CastGesture:
    """Frames in, three events out. Holds no frame and no landmark array.

    Drive it once per camera cycle with ``update()``. A cycle with no hand
    and a cycle where no frame arrived are THE SAME CALL -- ``update((),
    0.0, n)`` -- and that is deliberate. Distinguishing them would buy a
    slightly longer carry and cost a state the tests cannot reach; treating
    an unexplained gap as a lost hand fails safe, because the worst it does
    is end a carry he can start again.

    ``update`` runs on the capture thread; ``cancel`` arrives from wherever
    the spoken "drop it", the chip click or shutdown happens, so the three
    entry points share one lock.
    """

    def __init__(self, thresholds: CastThresholds = CastThresholds(),
                 frame_wh: tuple = (1280, 720), mirrored: bool = False,
                 payload: Optional[Payload] = None,
                 on_event: Optional[Callable[[CastEvent], None]] = None,
                 now: Callable[[], float] = time.monotonic,
                 preview_fps: float = 6.0) -> None:
        self.t = thresholds
        self.frame_w = float(frame_wh[0])
        self.frame_h = float(frame_wh[1])
        self.mirrored = bool(mirrored)
        self.payload = payload
        self.on_event = on_event
        self._now = now
        # The backstop, not the schedule: three frame periods of silence and
        # the machine gives up whatever it is holding. A wedged capture
        # thread must not leave a carry armed for the rest of the session.
        self.preview_fps = max(float(preview_fps), 0.1)
        self.stall_s = 3.0 / self.preview_fps
        self._lock = threading.Lock()
        self._reset_unlocked()

    # ------------------------------------------------------------- state
    def reset(self) -> None:
        with self._lock:
            self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        self._state = CastState.IDLE
        self._dwell = 0
        self._lost = 0
        self._carry = 0
        self._carry_started: Optional[float] = None
        self._cool = 0
        self._anchor: Optional[tuple[float, float]] = None
        self._unit = 0.0
        self._last: Optional[tuple[float, float]] = None
        self._last_fist: Optional[tuple[float, float]] = None
        self._last_step_u = 0.0
        self._dist_u = 0.0
        self._reach = 0.0
        self._closed = 0.0
        self._ambiguous = False
        self._held: Optional[tuple[str, str]] = None
        self._open_hist: deque = deque(
            maxlen=max(1, int(self.t.open_lookback_frames)))
        self._last_update: Optional[float] = None
        self._frame = -1

    @property
    def state(self) -> CastState:
        return self._state

    def retune(self, thresholds: CastThresholds,
               fps: Optional[float] = None) -> None:
        """The same machine at another frame rate, live, from any thread.

        The stage re-fits the counters to the DELIVERED rate; this is the
        only door for that. The open-history window is rebuilt to the new
        lookback under the lock, keeping what it held: it was sized once
        at construction, and a refit that assigned ``t`` alone left it at
        12 frames -- MEASURED: 0 of 48 grabs at 30 fps delivered against
        24 of 24 for the same thresholds set at construction, and 46/48 at
        15 fps. ``fps`` also resets the stall backstop to three periods.
        """
        with self._lock:
            self.t = thresholds
            self._open_hist = deque(
                self._open_hist,
                maxlen=max(1, int(thresholds.open_lookback_frames)))
            if fps is not None:
                self.preview_fps = max(float(fps), 0.1)
                self.stall_s = 3.0 / self.preview_fps

    def sweep(self) -> Optional[CastEvent]:
        """The wall-clock carry cap, applied WITHOUT a frame. Any thread.

        The cap and the stall backstop otherwise run only inside
        ``update()``, so a carry whose frames simply stopped -- the preview
        worker halted on a standby edge, a wedged read -- stayed live for
        as long as the silence lasted (MEASURED: 60 s, and a spoken throw
        then cast the stale subject). Every voice path calls this first.
        """
        with self._lock:
            now = float(self._now())
            if self._state is CastState.CARRYING and self._carry_expired(now):
                return self._end_carry("timeout", now)
            return None

    def carry_cap_s(self) -> float:
        """How long a carry can actually last, in seconds, at the rate
        this machine is tuned to: the frame cap (it ends on the frame AFTER
        ``carry_max_frames``, so 31 frames -- 4.13 s at 7.5 fps, MEASURED)
        or the wall-clock backstop, whichever comes first. The chip's
        depleting rule is drawn over THIS, not over the backstop alone,
        which was twice too long at 7.5 fps."""
        frames = (int(self.t.carry_max_frames) + 1) / max(self.preview_fps, 0.1)
        return min(float(self.t.carry_max_s), frames)

    @property
    def held(self) -> str:
        return self._held[1] if self._held else ""

    def status(self) -> dict:
        now = float(self._now())
        started = self._carry_started
        return {"state": self._state.value, "dwell": int(self._dwell),
                "lost": int(self._lost), "carry": int(self._carry),
                "carry_s": round(now - started, 3) if started else 0.0,
                "dist_u": round(float(self._dist_u), 4),
                "reach": round(float(self._reach), 4),
                "closed": round(float(self._closed), 4),
                "held": self.held, "ambiguous": bool(self._ambiguous),
                "cap_s": round(self.carry_cap_s(), 3),
                "frame": int(self._frame)}

    # -------------------------------------------------------- the cycle
    def update(self, hands: Sequence[HandObservation], eye_px: float,
               frame: int) -> Optional[CastEvent]:
        """One camera cycle. At most one event comes back.

        ``eye_px`` is accepted for symmetry with the caller and for the
        record; the reach ratio itself is already on each observation,
        because it has to be computed from the SAME frame as the hand.
        """
        with self._lock:
            now = float(self._now())
            self._frame = int(frame)
            stalled = self._check_stall(now)
            self._last_update = now
            if stalled is not None:
                return stalled

            usable = [h for h in (hands or ()) if getattr(h, "ok", False)]
            subject, ambiguous = self._pick(usable)
            self._ambiguous = ambiguous
            if subject is None:
                return self._on_miss(now)
            return self._on_hit(subject, now)

    def cancel(self, why: str = "cancelled") -> Optional[CastEvent]:
        """The spoken "drop it", the chip click and shutdown all land here.

        A live carry is put back and a ``drop`` is returned (and delivered
        to ``on_event``); a reach or a dwell is simply cleared. Nothing is
        ever thrown by a cancel, whatever the hand was doing.
        """
        with self._lock:
            now = float(self._now())
            if self._state is CastState.CARRYING:
                return self._end_carry("cancelled (%s)" % why, now)
            if self._state in (CastState.REACHING, CastState.CLOSING):
                self._to_idle()
            return None

    def _check_stall(self, now: float) -> Optional[CastEvent]:
        if self._last_update is None:
            return None
        if (now - self._last_update) <= self.stall_s:
            return None
        if self._state is CastState.CARRYING:
            return self._end_carry("stalled", now)
        self._to_idle()
        self._cool = 0
        return None

    def _pick(self, usable) -> tuple[Optional[HandObservation], bool]:
        """The subject hand, and whether the cycle is ambiguous.

        Outside a carry the subject is the largest ``palm_diag``, and a
        second hand that is ALSO at reach depth and comparably large makes
        the cycle ambiguous -- two hands out at the lens is not this gesture,
        and guessing between them is how a throw goes to the wrong side of
        the room. During a carry the rule relaxes to nearest-centroid,
        because he may well raise his other hand mid-carry and the carry
        should not die for it. Handedness is never consulted: it is the one
        field that silently inverts when somebody "fixes" the mirror.
        """
        if not usable:
            return None, False
        if self._state is CastState.CARRYING and self._last is not None:
            limit = self.t.assoc_max_u * max(self._unit, MIN_PALM_DIAG_PX)
            best, best_d = None, float("inf")
            for h in usable:
                d = math.hypot(h.cx - self._last[0], h.cy - self._last[1])
                if d < best_d:
                    best, best_d = h, d
            if best is None or best_d > limit:
                return None, False
            return best, False
        ranked = sorted(usable, key=lambda h: h.palm_diag, reverse=True)
        subject = ranked[0]
        for other in ranked[1:]:
            if (other.reach >= self.t.reach_min
                    and other.palm_diag >= (self.t.second_hand_frac
                                            * subject.palm_diag)):
                return None, True
        return subject, False

    # --------------------------------------------------------- branches
    def _on_miss(self, now: float) -> Optional[CastEvent]:
        self._open_hist.append(False)
        if self._state is CastState.COOLDOWN:
            self._tick_cooldown()
            return None
        if self._state is CastState.CARRYING:
            self._carry += 1
            if self._carry_expired(now):
                return self._end_carry("timeout", now)
            self._lost += 1
            if self._lost > self.t.lost_grace_frames:
                return self._end_carry("exit", now)
            return None
        if self._state in (CastState.REACHING, CastState.CLOSING):
            self._to_idle()
        return None

    def _on_hit(self, o: HandObservation,
                now: float) -> Optional[CastEvent]:
        self._open_hist.append(o.closed >= self.t.open_min)
        self._reach, self._closed = o.reach, o.closed

        if self._state is CastState.COOLDOWN:
            self._tick_cooldown()
            return None
        if self._state is CastState.CARRYING:
            return self._carrying(o, now)
        if self._state is CastState.IDLE:
            if o.reach >= self.t.reach_arm:
                self._state = CastState.REACHING
                self._prepare()
            return None
        if self._state is CastState.REACHING:
            return self._reaching(o)
        return self._closing(o, now)

    def _reaching(self, o: HandObservation) -> None:
        if o.closed <= self.t.closed_max and o.reach >= self.t.reach_min:
            self._latch(o, dwell=1)
            self._state = CastState.CLOSING
            return None
        if o.reach < self.t.reach_arm:
            self._to_idle()
        return None

    def _closing(self, o: HandObservation,
                 now: float) -> Optional[CastEvent]:
        if o.closed > self.t.closed_max or o.reach < self.t.reach_min:
            # He opened his hand, or it is no longer at reach, before the
            # dwell completed. Nothing was ever picked up, so nothing is put
            # down and no event fires. The reach is re-tested on every dwell
            # frame -- that is what the measured 0/288 harness did -- so a
            # fist pulled back along the lens axis cannot complete a grab.
            self._dwell = 0
            self._anchor = None
            self._state = (CastState.REACHING
                           if o.reach >= self.t.reach_arm
                           else CastState.IDLE)
            return None
        unit = max(self._unit, MIN_PALM_DIAG_PX)
        self._dist_u = math.hypot(o.cx - self._anchor[0],
                                  o.cy - self._anchor[1]) / unit
        if self._dist_u > self.t.anchor_drift_u:
            # Still moving. Re-latch where it is now and start the dwell
            # again: the anchor has to be where the throw STARTS, or the
            # displacement bar measures the approach instead of the fling.
            self._latch(o, dwell=0)
            self._dist_u = 0.0
            return None
        self._dwell += 1
        self._last = self._last_fist = (o.cx, o.cy)
        if self._dwell < self.t.dwell_frames:
            return None
        if sum(self._open_hist) < self.t.open_frames_req:
            # A fist that was never seen open is a resting fist, not a grab.
            return None
        return self._grab(o, now)

    def _carrying(self, o: HandObservation,
                  now: float) -> Optional[CastEvent]:
        self._carry += 1
        if self._carry_expired(now):
            return self._end_carry("timeout", now)
        unit = max(self._unit, MIN_PALM_DIAG_PX)
        if self._last is not None:
            self._last_step_u = math.hypot(o.cx - self._last[0],
                                           o.cy - self._last[1]) / unit
        self._last = (o.cx, o.cy)
        self._dist_u = math.hypot(o.cx - self._anchor[0],
                                  o.cy - self._anchor[1]) / unit
        if o.closed >= self.t.open_min:
            return self._end_carry("released", now)
        if 0.0 < o.reach < self.t.reach_exit:
            # Still closed and no longer at reach: he has pulled his hand
            # back with the thing in it. A withdrawal, never a throw, and
            # a reach of 0.0 is no opinion (the arm across the face), not
            # a withdrawal.
            return self._end_carry("withdrawn", now)
        self._lost = 0
        self._last_fist = (o.cx, o.cy)
        return None

    # ----------------------------------------------------------- helpers
    def _carry_expired(self, now: float) -> bool:
        if self._carry > self.t.carry_max_frames:
            return True
        started = self._carry_started
        return (started is not None
                and (now - started) > float(self.t.carry_max_s))

    def _latch(self, o: HandObservation, dwell: int) -> None:
        self._anchor = (o.cx, o.cy)
        self._unit = max(o.palm_diag, MIN_PALM_DIAG_PX)
        self._dwell = int(dwell)
        self._last = self._last_fist = (o.cx, o.cy)
        self._last_step_u = 0.0
        self._dist_u = 0.0

    def _to_idle(self) -> None:
        self._state = CastState.IDLE
        self._dwell = 0
        self._lost = 0
        self._carry = 0
        self._carry_started = None
        self._anchor = None
        self._last = None
        self._last_fist = None
        self._dist_u = 0.0
        self._last_step_u = 0.0

    def _tick_cooldown(self) -> None:
        self._cool -= 1
        if self._cool <= 0:
            self._state = CastState.IDLE
            self._cool = 0

    def _prepare(self) -> None:
        prep = getattr(self.payload, "prepare", None)
        if prep is None:
            return
        try:
            prep()
        except Exception:                                    # noqa: BLE001
            log.debug("gesture: payload.prepare raised", exc_info=True)

    def _emit(self, event: CastEvent) -> CastEvent:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:                                # noqa: BLE001
                # The capture thread must survive a bad subscriber. A
                # dropped notification is a missing chip; a raised one is a
                # dead camera pipeline.
                log.warning("gesture: on_event raised for %s", event.kind,
                            exc_info=True)
        return event

    def _grab(self, o: HandObservation, now: float) -> CastEvent:
        picked: Optional[tuple[str, str]] = None
        if self.payload is not None:
            try:
                picked = self.payload.pick_up()
            except Exception:                                # noqa: BLE001
                log.warning("gesture: payload.pick_up raised",
                            exc_info=True)
                picked = None
            if picked is None:
                # Nothing to hold. Do NOT enter a carry: an armed carry with
                # no subject is a false fire waiting to become a false
                # throw. It costs one tone and nothing else.
                self._enter_cooldown()
                return self._emit(CastEvent(
                    kind="drop", at=now, frame=self._frame,
                    why="nothing to carry", reach=o.reach,
                    closed=o.closed))
        self._held = (str(picked[0]), str(picked[1])) if picked else None
        self._state = CastState.CARRYING
        self._carry = 0
        self._carry_started = now
        self._lost = 0
        self._dist_u = 0.0
        self._last_step_u = 0.0
        return self._emit(CastEvent(
            kind="grab", at=now, frame=self._frame, dist_u=0.0,
            reach=o.reach, closed=o.closed, why="held",
            payload=self.held))

    def _enter_cooldown(self) -> None:
        self._state = CastState.COOLDOWN
        self._cool = int(self.t.cooldown_frames)
        self._dwell = 0
        self._lost = 0
        self._carry = 0
        self._carry_started = None
        self._anchor = None
        self._last = None
        self._last_fist = None

    def _end_carry(self, reason: str, now: float) -> CastEvent:
        """The whole throw decision, in one place.

        Three exits with three bars, because the evidence is not equally
        strong in each. Released in frame: he still has the hand where we can
        see it, so a full hand-width of travel is required and anything less
        is him setting the thing down. Left the picture near an edge: leaving
        is itself the evidence, and at reach distance there is often no room
        for more than a quarter of a hand-width, so the edge names the
        direction and the bar drops. Vanished in open space: the weakest
        case, so it needs both distance from the anchor AND to have been
        moving on the last step we saw. A = the anchor at the end of the
        dwell, B = the last frame the fist was seen. Then the sector: only a
        target sector (left, right) is a throw; down is the cancel and up is
        not a target, and both are reported as drops that say which way.
        """
        anchor = self._anchor or (0.0, 0.0)
        end = self._last_fist or self._last or anchor
        unit = max(self._unit, MIN_PALM_DIAG_PX)
        dx, dy = end[0] - anchor[0], end[1] - anchor[1]
        dist_u = math.hypot(dx, dy) / unit
        vec_bearing = bearing_deg(*to_his_frame(dx, dy, self.mirrored))
        why, thrown, bearing = reason, False, vec_bearing

        if reason == "released":
            thrown = dist_u >= self.t.throw_release_u
            why = "released"
        elif reason == "exit":
            edge, frac, edge_bear = edge_bearing(
                end[0], end[1], self.frame_w, self.frame_h, self.mirrored)
            # The edge may name the direction only when the hand was
            # MOVING toward that edge: a fist parked near his-left edge
            # that drifted 45 mm toward the centre and was then lost is
            # not a throw to his left (measured 3/6 before this check).
            # Within a quarter turn of the edge's outward normal counts;
            # otherwise it is judged as a hand lost in open space.
            agrees = _turn_between(vec_bearing, edge_bear) <= 90.0
            if frac <= self.t.edge_frac and agrees:
                bearing = edge_bear
                thrown = dist_u >= self.t.throw_exit_u
                why = "left frame (his %s)" % edge
            else:
                thrown = (dist_u >= self.t.throw_lost_u
                          and self._last_step_u >= self.t.exit_step_u)
                why = "lost"
        # 'timeout', 'stalled', 'withdrawn' and 'cancelled (...)' are never
        # throws: a carry that ran out, was pulled back closed, or that he
        # ended with a word, is him having put it down, not flung it.

        where = sector(bearing, self.t.sector_half_deg)
        if thrown and where == "ambiguous":
            # Measured but unnameable. Ambiguity resolves to the cheap,
            # reversible outcome, always.
            thrown = False
            why = "%s (ambiguous direction)" % why
        elif thrown and where not in self.t.target_sectors:
            thrown = False
            if where == "down":
                why = "cancelled (his down)"
            else:
                why = "%s (his %s is not a target)" % (why, where)

        name = self.held
        handle = self._held[0] if self._held else ""
        if not thrown and self.payload is not None and self._held:
            try:
                self.payload.put_back(handle)
            except Exception:                                # noqa: BLE001
                log.warning("gesture: payload.put_back raised",
                            exc_info=True)
        self._held = None
        reach, closed = self._reach, self._closed
        self._enter_cooldown()
        self._dist_u = dist_u
        return self._emit(CastEvent(
            kind="throw" if thrown else "drop", at=now, frame=self._frame,
            dist_u=dist_u, bearing_deg=bearing,
            sector=where if thrown else "", why=why, reach=reach,
            closed=closed, payload=name,
            toward=where if dist_u >= TOWARD_MIN_U else ""))


def describe(t: CastThresholds = CastThresholds(),
             fps: float = 6.0) -> str:
    """One line for a bring-up log: what the frame counters mean in time."""
    ms = 1000.0 / max(float(fps), 0.1)
    return ("cast: C<=%.2f closed / >=%.2f open, R>=%.2f grab (arm %.2f), "
            "dwell %d fr (%.0f ms), grace %d fr, cooldown %d fr, cap %d fr "
            "(%.1f s) or %.1f s, targets %s at %.1f fps"
            % (t.closed_max, t.open_min, t.reach_min, t.reach_arm,
               t.dwell_frames, t.dwell_frames * ms, t.lost_grace_frames,
               t.cooldown_frames, t.carry_max_frames,
               t.carry_max_frames * ms / 1000.0, t.carry_max_s,
               "/".join(t.target_sectors), fps))


__all__ = [
    "CallablePayload", "CastEvent", "CastGesture", "CastState",
    "CastThresholds", "DESIGN_FPS", "HandObservation", "Payload",
    "HAND_UNIT_MM", "PALM", "PALM_DIAG_OVER_IPD", "SECTORS", "TIPS",
    "bearing_deg", "describe", "edge_bearing", "observe_hand", "sector",
    "to_his_frame",
]
