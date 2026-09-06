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

A THROW MUST BE A THROW, AND LEAVING THE PICTURE IS NOT INTENT. The first
draft said it was: ``throw_exit_u`` dropped to a quarter hand-width on the
reasoning that at reach distance there is often no room for more. MEASURED
09-05 over 1232 desk sequences: 448 false fires, P(fire | not a cast
gesture) = 0.36, and every one of them the same shape -- reach at the
screen, close the hand on something, carry it out of the picture, no fling
and no release. A man carrying a mug leaves the picture. Distance cannot
tell the two apart, because a carried hand and a flung hand cover the SAME
~2 units before the frame edge takes them; SPEED can, by a factor of three.
So every distance bar is kept and one requirement is put in front of all
three of them: the hand must have been travelling at ``throw_speed_us``
within ``fling_window_s`` of the last frame that saw it. That took the same
grid to 0. What it does NOT do is separate his own slow deliberate throw
(measured floor 3.02 hand-units/s) from a mug hurried out in 1.6 s (peak
3.34): those are the same motion here, the bar sits between the mug's
ordinary speeds and his gesture, and the residual is stated in
``tests/test_gesture.py`` rather than hidden.

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

ROUND 3, AND WHAT IT SETTLED. An independent attack rebuilt the desk grid
wider -- 172 non-gesture families, 2752 sequences, all synthetic -- and
measured P(fire | NOT a cast gesture) = 0.2700 at 7.5 fps against a bar of
0.005. Its central finding is accepted here and is not re-argued: HIS
ORDINARY BRISK DESK MOTION SITS ON THE SAME SIDE OF THE SPEED BAR AS HIS
OWN THROW. A mug carried out in 1.2 s travels 433 mm/s; his deliberate
slow throw floors at about 370. No value of ``throw_speed_us`` separates
those, and this file no longer pretends one does.

Four things changed underneath it, and each was a measurement that was
wrong rather than a threshold that was loose:

  * THE HAND-UNIT. It was ``palm_diag``, latched at the grab, and
    foreshortened: 3.0 u/s was worth 381 mm/s for a flat palm square to
    the lens, 289 at his working pitch, and 204 once the hand had come
    nearer with the unit still latched. It is now measured EVERY FRAME
    and corrected for pose (``HandObservation.unit``), and every distance
    bar was rescaled by 0.86 so each keeps the millimetres it was
    measured at. Worst error over the sweep: 34% before, 11% after.
  * THE FLING GRANT. ``fling_window_s`` was measured to the last frame
    that SAW the hand and ``lost_grace_frames`` then ran on top, so the
    documented 0.55 s was 1.07 s at 7.5 fps. It is now measured to the
    moment the throw would fire, floored at the grace.
  * THE ASSOCIATION. A carry frame with more than one candidate hand is
    now AMBIGUOUS: the nearest is still followed, but its movement is not
    fling evidence.
  * THE FIST SCALAR. Measured, and then NOT gated -- see
    ``closed_ratio_max``, which is the one place round 3 says the attack's
    inference was wrong.

AND THE SECOND SIGNAL, because one scalar cannot do it: a throw now also
needs HIS HEAD to have been pointed at the screen he grabbed, from before
the reach through to the last frame that held the hand. That angle is not
in this file and never will be; ``handstage.HandStage`` measures it and
passes a tri-state into ``update()``.

WHERE THAT LEAVES THE NUMBER, stated rather than dressed up: 0.2700 ->
0.1995 at 7.5 fps on the same grid and the same seeds, with recall in the
5.5-8.0 band at 1526 of 1536 and no throw to the wrong side. The bar is
0.005. IT IS NOT MET, and it cannot be met from the hand track alone,
because the families that still fire -- a mug carried out in under 1.2 s,
a thing handed to someone, a hand that leaves the picture and comes back
-- are the same motion as the gesture by every signal this rig has.

ROUND 4 MOVED THAT NUMBER THE WRONG WAY ON PURPOSE, AND HERE IT IS.
Round 3 bought part of its 0.1995 by placing ``throw_speed_us`` inside
the spread of HIS OWN throw: measured over the full recall surface, 41 of
84 cells were below round 2 and the gesture stopped firing when his swing
was 10 mm shorter or ended 10 mm further from the lens. The bar moved to
2.45, which is at or above round 2 in every cell, and the false-fire rate
went with it:

    P(fire | NOT a cast gesture), the preserved 172-family grid,
    2752 sequences, same seeds:
      7.5 fps  0.2362  95% CI [0.2207, 0.2524]   (round 3 0.1995)
      6.0 fps  0.2322  95% CI [0.2168, 0.2483]   (round 3 0.2002)
      3.7 fps  0.1435  95% CI [0.1309, 0.1571]   (round 3 0.1134)

THE BAR IS 0.005. It is forty-seven times over. A rate that is 40x or 47x
its bar is not usefully different: both mean the gesture cannot be trusted
to stay off his screens, and a recall cliff he cannot see is a worse thing
to ship than a number that was already unshippable.

ROUND 5 ASKED HIM THE QUESTION AND MEASURED THE ANSWER. THIS IS THE END
OF THE LINE FOR THIS GESTURE, AND IT IS A LEGITIMATE OUTCOME.

Round 4 said the one remaining candidate was a true WIND-UP -- a small
backward retraction immediately before the swing -- and that only he could
say whether his throw has one. Asked directly on 2026-09-05, "when you
throw, does your hand pull back a little first, before it swings?", HE
ANSWERED YES. It is built, it is measured, and it is not enough.

It is a real signal. ``CastEvent.windup_u`` is the most negative
projection of the carry path onto the axis the throw took, over the prefix
up to the furthest forward point, in hand-units; injected retractions come
back monotonically (5 mm -> 0.049 u, 40 mm -> 0.247, 120 mm -> 0.822,
MEASURED over 8 sub-frame phases and four retraction durations).

AND IT DOES NOT SEPARATE, for two independent reasons.

  * THE AMPLITUDE IS SHARED. On the PRESERVED 172-family grid a bar of
    0.05 u takes P(fire | NOT a cast gesture) from 0.2362 to 0.0000, which
    looks like a solution and IS AN ARTEFACT: that grid was built to
    attack speed and distance and contains no retractions at all. Round 5
    added 54 families that DO retract, invented adversarially and swept to
    the same amplitudes his throw is -- pulling a mug toward himself
    before lifting it away, drawing back to get a run at something heavy,
    a hand that hesitates, settling a grip. They measure the SAME wind-up
    his throw does: at 40 mm, his throw 0.218 u against a carry's 0.239.
    On the honest 226-family grid, 4192 sequences, same seeds:

        P(fire | NOT a cast gesture), by wind-up bar, 7.5 fps
          bar 0.00 u   0.4986  95% CI [0.4834, 0.5137]   recall needs 0 mm
          bar 0.15 u   0.2600  95% CI [0.2470, 0.2735]   ...needs  40 mm
          bar 0.30 u   0.1935  95% CI [0.1818, 0.2057]   ...needs  80 mm
          bar 0.60 u   0.1054  95% CI [0.0965, 0.1151]   ...needs 120 mm
        and at 6.0 fps: 0.4945, 0.2510, 0.1868, 0.0942.

    THE BAR IS 0.005. The best point in the sweep is TWENTY-ONE TIMES over
    it and it asks him to pull his hand back 120 mm before every throw.
    That is a performance, not a gesture, and it is not shippable.

  * THE DIRECTION MAY MAKE IT INVISIBLE, which is worse than weak. The
    wind-up is measured in the IMAGE PLANE. A retraction straight back
    TOWARD HIS BODY reads 0.000 u at every amplitude up to 120 mm
    (MEASURED) -- it does not even end the carry, recall stays 1.0000 with
    the gate off -- so if that is the wind-up he makes, every bar above
    0.02 refuses every throw he makes. THE AMPLITUDE AND THE DIRECTION OF
    HIS OWN WIND-UP ARE UNKNOWN. He said it exists; he did not measure it,
    and nothing in this repo has ever seen his hand. The numbers-only
    instrument is:

        ~/vss_env/bin/python scripts/gesture_selfcheck.py \
            --seconds 30 --windup

So ``windup_min_u`` ships at 0.0, which refuses nothing, and this gesture
stays ``enabled: False`` in jarvis/assistant_config.py. The second signal
cannot help either and cannot be tuned into helping: ``yaw_hold_deg`` is a
switch, not a dial -- at or under 25 deg of head movement it vetoes
nothing and the rate is unchanged, over 25 deg it refuses every throw he
makes, and there is no setting between. Three independent signals have now
been measured against this grid and none of them separates his throw from
his desk. THE SPOKEN CAST IS NOT BEHIND THIS SWITCH and is where the work
belongs.

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
# The knuckles whose distance from the wrist is measured ALONG the finger
# axis: index, middle and ring. See PALM_LEN_MM.
AXIS_MCPS = (5, 9, 13)

# A hand-unit is one palm_diag. MEASURED on the synthetic hand; the
# millimetre figure behind it is GUESSED standard adult anthropometry
# (palm length 107 mm, MCP breadth 68.2 mm -> hypot 127 mm), NOT measured on
# him. Every distance this module reports is in hand-units precisely so that
# a wrong millimetre figure cannot corrupt a decision -- it only changes what
# a unit means in the world.
HAND_UNIT_MM = 127.0
# The two lengths the CORRECTION needs on their own. Same GUESSED
# anthropometry.
#
# PALM_LEN_MM is the mean wrist-to-knuckle distance over the INDEX,
# MIDDLE and RING MCPs (105.4 mm), not the single wrist-to-middle-MCP
# segment. Two reasons and both were measured. It is a DENOMINATOR, and
# one 2-point distance at 12 px of landmark noise cost 16 grabs in 88 on
# the round-2 gesture set -- a fist the machine stopped recognising, not
# any decision about a throw. And it must stay ALONG THE FINGER AXIS,
# because that is the whole point of dividing by it: the index and ring
# knuckles sit 12-14 degrees off that axis (a 2.6% bias) while the pinky
# sits 25 degrees off, so the pinky is left out.
PALM_LEN_MM = 105.4
PALM_W_MM = 68.2
# palm_diag / interocular for the same anthropometry: 127 / 63. The reach
# ratio R is scale-free BECAUSE of this -- R = 2.01 * k(pose) * Zface/Zhand.
# His own proportions could move it by +/-15%, which is why reach_min has
# margin measured into it rather than being set to the nominal.
PALM_DIAG_OVER_IPD = 2.01

# Below this a landmark row is degenerate -- two coincident points, a
# collapsed detection -- and every ratio computed from it is invented. It is
# reported as ``ok=False``, never as a very small hand.
MIN_PALM_DIAG_PX = 1e-6

# How many frames the hand-unit is smoothed over before a step is divided
# by it. Three is one frame either side at 7.5 fps -- 267 ms, shorter than
# the dwell -- and it is a MEDIAN, so one bad landmark row is absorbed
# rather than averaged in.
UNIT_MEDIAN_FRAMES = 3

# How many carry frames the round-5 traces keep. The carry cap is 30 frames
# and ``for_fps`` may scale it up for a faster feed; this is comfortably
# above any of that and it is a HARD BOUND -- neither trace may grow with
# session length, because an unbounded record on the capture thread is how
# this box has already lost a session.
CARRY_TRACE_FRAMES = 96

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

class _NoHead:
    """The default for ``update(looking=...)``, and it is NOT ``None``.

    ``None`` means a caller that reads his head LOOKED AND HAD NO OPINION
    this frame -- no clean face row -- and ``yaw_required`` decides what
    that costs. ``NO_HEAD`` means the caller has no head to offer at all:
    a direct-drive test, a grid of synthetic hands, an embedder that never
    wires a face detector. Collapsing the two would either make every
    hand-only harness refuse every throw, or make a missing face silently
    count as a yes. It is the same three-valued discipline ``eye.py`` and
    ``roomsensor.py`` already hold, with one more state because there are
    three things to say.
    """

    __slots__ = ()

    def __repr__(self) -> str:                       # pragma: no cover
        return "NO_HEAD"


NO_HEAD = _NoHead()

# WHY A CARRY DID NOT BECOME A THROW, as a CLOSED SET -- the same
# discipline the cast verbs are held to, and for the same reason: this
# code goes into a log line, a counter and the stage status, and a
# free-text reason cannot be counted or compared.
#
#   ""          it WAS a throw
#   "speed"     far enough, never fast enough -- the fling bar refused it
#   "distance"  it did not travel far enough from the anchor
#   "look"      his head was not on the screen he grabbed (or no face row)
#   "direction" the bearing fell between two sectors and is unnameable
#   "sector"    a clean throw at a sector that is not a target (his up)
#   "cancel"    his deliberate down-fling, which is the put-back
#   "carry"     it never got as far as the three bars: it timed out, it
#               stalled, he pulled it back, or a word ended it
#
# ROUND 4 ADDED THESE BECAUSE THE FAILURE WAS SILENT. A throw refused by
# the speed bar produced a drop, a tone, and nothing he could read; "it
# just did nothing" was the whole story he had, and he cannot debug that.
#   "windup"    ROUND 5: it never pulled back before it swung. Inert at
#               the shipped bar of 0.0 -- see ``windup_min_u``.
REFUSALS = ("", "speed", "distance", "look", "direction", "sector",
            "cancel", "carry", "windup")

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
    # ROUND 3. The two palm lengths, and the two scalars derived from them
    # that the shipped pair could not carry.
    #
    # ``unit`` is the hand-unit this module measures every distance in, and
    # it is measured ON THIS FRAME rather than latched at the grab, and
    # corrected for the pose. ``palm_diag`` alone is foreshortened: at his
    # working pitch of 35 degrees it reads 225 px where 127 mm at that
    # depth projects to 296, so ``throw_speed_us`` 3.0 -- documented as
    # "about 380 mm/s at his reach" -- was worth 289 mm/s there, 242 at
    # 50 degrees and 204 once the hand had moved to 300 mm with the unit
    # still latched at the grab distance (MEASURED, round-3 attack). The
    # correction is arithmetic, not a fudge: a rotation about the lens's x
    # axis leaves the MCP breadth (5->17) alone and foreshortens the palm
    # length (0->9); a rotation about y does the reverse. Each observed
    # length, scaled back up by its own share of the 127 mm hypotenuse, is
    # therefore a LOWER BOUND on the true unit, and the largest of the
    # three is the tightest bound available from one frame. It is exact
    # whenever either angle is near zero and never over-reads. MEASURED:
    # worst error over z 300-500 mm and pitch 0-50 deg falls from 31% to
    # under 4%.
    #
    # ``curled`` is the fist scalar that survives his reaching pose. The
    # shipped C divides the finger extension by ``palm_diag``, which keeps
    # a pitch-invariant palm WIDTH inside it, so the numerator
    # foreshortens and the denominator only half does: a RELAXED hand at
    # curl 0.40 and pitch 35 reads C = 0.602 against ``closed_max`` 0.70
    # and is taken for a grip (1921 false grabs in 2752 grid sequences,
    # MEASURED). ``curled`` divides by the palm LENGTH instead, which
    # foreshortens with the fingers, so the ratio holds still: a real fist
    # reads 0.644-0.723 across the whole pose envelope while curl 0.40 at
    # pitch 35 reads 0.770.
    palm_len: float = 0.0
    palm_w: float = 0.0
    unit: float = 0.0
    curled: float = 0.0

    def __post_init__(self) -> None:
        """A row built by hand carries only the two scalars round 2 had.
        Fall back to those rather than to zero: a ``unit`` of 0.0 is not a
        small hand, it is a caller that did not measure one, and dividing
        a step by it is how a bar becomes a hole. ``curled`` falls back to
        ``closed``, so such a row is judged on exactly what it carries."""
        if not self.unit > 0.0:
            object.__setattr__(self, "unit", float(self.palm_diag))
        if not self.curled > 0.0:
            object.__setattr__(self, "curled", float(self.closed))
        if not self.palm_len > 0.0:
            object.__setattr__(self, "palm_len",
                               float(self.palm_diag) * PALM_LEN_MM
                               / HAND_UNIT_MM)
        if not self.palm_w > 0.0:
            object.__setattr__(self, "palm_w",
                               float(self.palm_diag) * PALM_W_MM
                               / HAND_UNIT_MM)

    def numbers_only(self) -> dict:
        return {"palm_diag": round(self.palm_diag, 3),
                "closed": round(self.closed, 4),
                "curled": round(self.curled, 4),
                "unit": round(self.unit, 3),
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
    # palm_diag is UNCHANGED and is still the single wrist-to-middle-MCP
    # segment against the MCP breadth: every round-2 bar was measured on
    # it and none of them moves here.
    palm_diag = math.hypot(
        float(np.linalg.norm(pts[MID_MCP] - wrist)),
        float(np.linalg.norm(pts[PNK_MCP] - pts[IDX_MCP])))
    # The two ROBUST lengths the round-3 scalars divide by: the mean of
    # the four wrist-to-knuckle distances, and the MCP breadth.
    palm_len = float(np.mean(
        [np.linalg.norm(pts[m] - wrist) for m in AXIS_MCPS]))
    palm_w = float(np.linalg.norm(pts[PNK_MCP] - pts[IDX_MCP]))
    cx, cy = (float(v) for v in pts[list(PALM)].mean(axis=0))
    if palm_diag < MIN_PALM_DIAG_PX:
        # Do not invent C or R from a collapsed hand. Say so instead.
        return HandObservation(0.0, 0.0, 0.0, cx, cy, float(conf), False)
    reach_span = float(np.mean(
        [np.linalg.norm(pts[t] - wrist) for t in TIPS]))
    closed = reach_span / palm_diag
    reach = (palm_diag / float(eye_px)) if float(eye_px) > 0.0 else 0.0
    # The pose-corrected unit: the tightest lower bound the three lengths
    # give. See HandObservation for why this is arithmetic and not a fudge.
    unit = max(palm_diag,
               palm_len * (HAND_UNIT_MM / PALM_LEN_MM),
               palm_w * (HAND_UNIT_MM / PALM_W_MM))
    # A palm seen edge-on has no length to divide by. That is NO OPINION
    # about the fist, and no opinion must read as OPEN here, never as a
    # grip: refusing a grab costs a gesture, inventing one opens a window.
    curled = (reach_span / palm_len) if palm_len >= MIN_PALM_DIAG_PX \
        else float("inf")
    return HandObservation(palm_diag, closed, reach, cx, cy,
                           float(conf), True, palm_len=palm_len,
                           palm_w=palm_w, unit=unit, curled=curled)


def _median(values) -> float:
    """The middle of a handful of floats. ``statistics`` is not importable
    here: tests/test_gesture.py pins this module's whole import list, and
    the list is the promise that nothing in it can reach a frame."""
    rows = sorted(float(v) for v in values)
    n = len(rows)
    if not n:
        return 0.0
    mid = n // 2
    return rows[mid] if n % 2 else 0.5 * (rows[mid - 1] + rows[mid])


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
    # ROUND 3 MEASURED THIS AND THEN DID NOT SHIP THE BAR IT WAS ASKED
    # FOR, WHICH IS THE FINDING. ``closed_max`` really is pose-dependent:
    # over a slice at roll 0 and yaw 0 a curl of 0.40 at his working pitch
    # of 35 deg reads C = 0.602 against 0.70, and a relaxed hand reaching
    # at that monitor is a fist to the rig. ``curled`` (HandObservation)
    # divides by the palm LENGTH instead, which foreshortens with the
    # fingers, and over that same slice it separates them cleanly.
    #
    # OVER THE WHOLE POSE ENVELOPE IT DOES NOT. Swept over curl 0.15-1.00
    # x pitch 0-55 x roll 0/20/180/200 x yaw -20/0/20 x z 350-500 mm, a
    # real grip reads curled up to 1.162 -- the fingers-down hand, which
    # is one of the attacker's own throw families -- and the tightest bar
    # that refuses no fist admits every relaxed hand ``closed_max``
    # already admitted. At a bar of 0.85 it refuses 194 fists in 1080 to
    # exclude 47 relaxed hands in 1620. That is not a trade worth making
    # on a gesture he cannot confirm.
    #
    # AND THE PREMISE UNDERNEATH IT IS WRONG, which matters more than the
    # scalar. The attack read 1921 false grabs in 2752 grid sequences as
    # relaxed hands misread. MEASURED: adding an aggressive 0.85 bar moves
    # the grid's grab rate by about one point (972 -> 962 of 1376 at eight
    # phases). Those grabs are hands that REALLY CLOSED -- on a mug, on
    # the monitor bezel, on a pen -- so no closure bar was ever going to
    # be where the false fires come from, and every family that still
    # fires below closes a full fist.
    #
    # What ships is the measurement, not a gate: ``curled`` is computed,
    # reported in ``numbers_only`` and bounded here at a value the sweep
    # says costs no fist at any pose in the envelope. It is a sanity floor
    # -- fingertips further from the wrist than the knuckles are is not a
    # fist -- and it is one key in assistant.json, so he can tighten it
    # from his own room where the pose distribution is real rather than
    # invented.
    closed_ratio_max: float = 1.05
    # ...and the same for the other end, MEASURED AND THEN LEFT OFF. An
    # open hand at pitch 55 reads C = 0.725, under ``open_min``, so at
    # that pose the machine never sees it open -- which starves the
    # open-history a grab needs and loses the release that ends a carry
    # in frame. Letting ``curled >= 1.05`` also count as open fixes that,
    # and on the preserved grid it changed no throw at all and cost 11
    # extra false grabs in 2752 (1943 against 1932). So it ships OFF, at
    # a value nothing reaches, and it is one key in assistant.json if his
    # own room ever shows the high-pitch release being missed.
    open_ratio_min: float = 99.0
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
    anchor_drift_u: float = 0.52
    # MEASURED. Three exits, three bars. Opening the hand where it is must
    # never be a throw -- he opens his hand hundreds of times an hour -- so a
    # release in frame needs a full hand-width of travel. Leaving the picture
    # is itself the evidence of intent, so that bar drops to 0.25. Vanishing
    # in open space is the weakest evidence and needs both distance AND to
    # have been moving when we lost it.
    #
    # ALL FOUR, AND ``anchor_drift_u`` AND ``assoc_max_u`` ABOVE, WERE
    # RESCALED BY 0.86 IN ROUND 3 AND THE MEANING IS UNCHANGED. They are
    # in hand-units, and the hand-unit itself was wrong: ``palm_diag`` at
    # his working pose (425 mm, pitch 35 deg) reads 225 px where 127 mm
    # projects to 262, so every one of these bars was silently 16% wider
    # than the millimetres it was measured at -- and up to 31% wider at
    # other poses. Multiplying each by 225/262 leaves it exactly where it
    # was measured AT THAT POSE and stops it moving with his wrist
    # everywhere else. MEASURED, without the rescale: three false grabs
    # appeared on round 2's own everyday set and its retraction test
    # threw 12 where it had thrown 10.
    throw_release_u: float = 0.86
    throw_exit_u: float = 0.22
    throw_lost_u: float = 0.43
    exit_step_u: float = 0.30
    edge_frac: float = 0.30
    # MEASURED (the carry-out attack, 09-05), AND THE BAR THAT NOW DECIDES.
    # The three distance bars above cannot separate a throw from a mug: a
    # hand carried out of the picture and a hand flung out of it cover the
    # SAME ~2 units before the frame edge takes them, so raising
    # throw_exit_u buys nothing. Speed can, and by a factor of three. Peak
    # image speed of the carry, in HAND-UNITS PER SECOND (one unit is one
    # palm_diag, 127 mm on the guessed anthropometry, so 3.0 u/s is about
    # 380 mm/s at his reach):
    #
    #   the design's own throw (250-400 ms swing)   min 3.06  p05 4.90
    #   a slow deliberate throw (0.8 s swing)       min 3.02  p05 3.18
    #   a mug carried out over 2.0 s (235 mm/s)     max 2.73  p50 2.35
    #   a mug carried out over 2.4 s                max 2.35  p50 1.98
    #   a mug carried out over 3.2 s (147 mm/s)     max 1.83  p50 1.52
    #
    # over 5.5-8.0 fps with 0/3/5/8 px of landmark noise.
    #
    # ROUND 3 RE-MEASURED ALL OF THAT IN THE CORRECTED UNIT, because the
    # figures above were read against a scale 16-31% too small and so
    # overstate every speed. In the unit this module now uses
    # (``HandObservation.unit``, pose-corrected and measured every frame)
    # the same families read:
    #
    #   his deliberate 400 ms throw                 5.78-6.41
    #   his SLOW 800 ms throw, 300 mm swing         2.89-3.20  <- the floor
    #   his slow throw at the 250 mm swing          2.39-2.69  (an old hole)
    #   a mug carried out over 1.2 s (433 mm/s)     3.42
    #   a mug carried out over 1.6 s (325 mm/s)     2.58
    #   handing a thing over in 0.70 s (336 mm/s)   2.60
    #   setting a thing down in 0.7 s (300 mm/s)    2.34
    #
    # ROUND 3 SET THIS TO 2.65 ON ONE POINT OF A SURFACE AND IT CUT INTO
    # HIS OWN GESTURE. That value was chosen against a single family --
    # a 300-310 mm swing ending at z = 440 -- and round 4 measured the
    # whole surface: swings 270-330 mm in 10 mm steps, end depths 420-470
    # in 10 mm steps, both directions, the 5.5-8.0 fps band, 0/3/5 px of
    # landmark noise, 16 sub-frame phases, both finger orientations
    # (castgrid/test_r4_recall.py, 24192 sequences a pass). At 2.65,
    # 41 of the 84 cells were BELOW round 2, worst 0.7569 -> 0.1910, and
    # the machine said why on every one of them: "carried out of frame,
    # NOT FLUNG".
    #
    # AND IT WAS NOT THE DISTANCE RESCALE, which is what it looked like.
    # ABLATED one change at a time on the same grid: putting every
    # distance bar back to its round-2 value changed the surface by
    # NOTHING (219/336 either way at 7.5 fps, 0 px), and so did switching
    # off closed_ratio_max and assoc_step_u. Only this bar moved it.
    #
    # WHY IT CUT, and this is the whole of it -- MEASURED, not derived.
    # His own slow deliberate throw is not one speed, it is a spread, and
    # the machine measures it across this surface as:
    #
    #   peak u/s on HIS OWN throw, 7.5 fps, 0 px, his-left
    #     swing 270 mm   2.576 (z=420) .. 2.512 (z=470)
    #     swing 290 mm   2.777         .. 2.709
    #     swing 310 mm   2.978         .. 2.906
    #     swing 330 mm   3.180         .. 3.104
    #
    # 2.65 SITS INSIDE THAT SPREAD. It refuses every 270 and 280 mm swing
    # and most 290s, which is exactly the cliff. 2.45 sits BELOW all of it
    # (the minimum measured is 2.512), which is why recall comes back
    # wherever the frame geometry allows a throw at all.
    #
    # And the mm figure above is NOT what it says. One hand-unit measures
    # 112.0 mm at z=420 and 113.4 mm at z=470 on this hand at his working
    # pitch -- the pose correction still UNDER-reads the 127 mm it is
    # documented as by about 11%, which is the residual the round-3 attack
    # measured (34% -> 11%). So "about 337 mm/s at any pose and any depth"
    # is wrong twice over: wrong in value and wrong in claiming it does not
    # move. Any bar stated in mm/s here would be stating a number this rig
    # cannot yet support, so the value above is defended by the recall
    # surface and the false-fire grid and by nothing else.
    #
    # 2.45 IS THE HIGHEST VALUE THAT IS AT OR ABOVE ROUND 2 IN EVERY ONE
    # OF THE 84 CELLS (measured: 2.48 was below in 2, 2.50 in 6, 2.55 in
    # 24). Over the whole surface it is 22001/24192 = 0.9094 against round
    # 2's own 20712/24192 = 0.8562, with no throw to the wrong side.
    # THE FLING WINDOW WAS NOT TRADED AWAY TO GET IT: fling_window_s stays
    # at 0.55 and stays measured to the moment the throw would fire, so
    # round 3's measurement fix is kept whole.
    #
    # WHAT IT COSTS, stated rather than hidden. On the attacker's
    # preserved 172-family grid, 2752 sequences, same seeds:
    # P(fire | NOT a cast gesture) 0.1995 -> 0.2362 at 7.5 fps,
    # 0.2002 -> 0.2322 at 6.0, 0.1134 -> 0.1435 at 3.7. The bar is 0.005.
    # It was forty times over before and it is forty-seven times over now;
    # it is still under round 2's 0.2700, and the gesture ships OFF for
    # exactly this reason. See the module docstring.
    #
    # IT IS STILL NOT A SEPARATION and no value of it can be: a mug
    # hurried out in 1.2 s is genuinely faster than his own slow throw.
    #
    # PER SECOND, NOT PER FRAME, and so ``for_fps`` must not touch it: a
    # throw is fast in the world, not fast per sample, and the same fling
    # sampled at 15 fps moves half as far between frames.
    throw_speed_us: float = 2.45
    # How stale the fling may be when the carry ends. The hand must have
    # been travelling at throw speed within this much of the LAST FRAME
    # THAT SAW IT -- so a jerk at the start of a four-second carry cannot
    # be spent at the end of it. 0.55 s is four frames at 7.5 fps and one
    # at the camera's 2 fps idle rate.
    fling_window_s: float = 0.55
    # ================================================================
    # ROUND 5, AND IT IS THE SIGNAL THE LAST THREE ROUNDS RAN OUT OF.
    #
    # THE WIND-UP. Asked directly on 2026-09-05 -- "when you throw, does
    # your hand pull back a little first, before it swings?" -- HE ANSWERED
    # YES. It was absent from every throw model in this lane, whose spans
    # are reach, close, hold still, swing, which is exactly why it had to
    # be asked rather than measured. It is the one signal a carry cannot
    # fake at any speed, because carrying something away never reverses
    # before it leaves.
    #
    # It is NOT the reversal round 3 disproved. That was APPROACH versus
    # EXIT and it scored nothing, because his hand comes in from his left
    # in the throw and in the carry alike, so it only ever encoded which
    # side he threw to. THIS is a backward retraction WITHIN the throw,
    # after the dwell and before the fling, measured along the axis the
    # throw eventually took: the most negative projection of the carry path
    # onto that axis, over the prefix up to the furthest forward point, in
    # hand-units.
    #
    # THE AMPLITUDE AND DURATION OF HIS OWN WIND-UP ARE UNKNOWN. He said it
    # exists; he did not measure it, and nothing in this repo has seen his
    # hand. ``scripts/gesture_selfcheck.py --windup`` is the numbers-only
    # instrument that lets him find out.
    #
    # AND THE BAR SHIPS AT 0.0, WHICH IS INERT, BECAUSE THE MEASUREMENT
    # SAYS IT DOES NOT SEPARATE. castgrid/test_r5_windup.py sweeps the bar
    # against the preserved 172-family grid PLUS 40 new families invented
    # adversarially to contain a retraction of their own -- pulling a mug
    # toward himself before lifting it away, drawing back to get a run at
    # something heavy, a hand that hesitates before it leaves. Those
    # families retract exactly as his throw does, so every bar that keeps
    # his own recall also keeps them, and the numbers are in that file and
    # in the module docstring. A gate that only looks like it works is
    # worse than none: this one is MEASURED, REPORTED, and OFF.
    windup_min_u: float = 0.0
    # ================================================================
    # ROUND 3, AND THE SECOND INDEPENDENT SIGNAL. One scalar cannot
    # separate his throw from his desk motion: MEASURED, his brisk desk
    # motion sits on the SAME SIDE of the speed bar as his own slow throw
    # (a mug carried out in 1.2 s peaks at 433 mm/s, his deliberate slow
    # throw floors at ~370). So a throw also has to have him LOOKING at
    # the screen he grabbed, from before the reach through to the last
    # frame that saw the hand. The angle is measured by the hand stage --
    # this module has no face -- and arrives as a tri-state ``looking`` on
    # ``update()``: True, False, or None for no clean face row.
    #
    # ``yaw_hold_deg`` is how far his head may drift from where it was
    # BEFORE the reach began and still count as held on that screen. 25
    # deg is GUESSED: it is wider than the 20 deg of yaw noise the face
    # rows carry at rest and narrower than the turn to the middle screen.
    # THE HAZARD IS NAMED AND MEASURED IN THE ROUND-3 REPORT: a mug
    # carried away from that same monitor may share the yaw exactly, and
    # then this signal buys nothing at all.
    yaw_hold_deg: float = 25.0
    # What NO OPINION means. A frame with no clean face says nothing about
    # where he was looking, and that is not the same as him looking at the
    # screen. Refusing is the safe direction and it is the default; the
    # cost is a gesture that does not fire, which he can see and repeat.
    # It is one key in assistant.json (``gesture.yaw_required``) because
    # the abstain RATE is the one number no synthetic grid can produce --
    # only his own room can, and ``screens_status()['yaw_miss_pct']`` is
    # the instrument for it.
    yaw_required: bool = True
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
    assoc_max_u: float = 2.15
    # ...but a hand adopted across a GAP must not have that gap counted as
    # travel. MEASURED (round-3 attack): ``_pick`` relaxed to nearest
    # centroid with no identity check, so his OTHER hand appearing
    # 0.5-2.4 u from the carried hand's last position was taken as the
    # carried hand and the jump was measured as one frame of movement --
    # 3.75 to 18.0 u/s, up to six times the bar, and ``flung`` went True
    # with no hand having moved fast.
    #
    # THE FIX IS NOT A SPEED BAR, AND THAT IS THE POINT. The hijack at
    # 0.5 u reads 3.75 u/s, which is an ordinary fling speed -- no bar
    # separates "his other hand appeared half a unit away" from "his hand
    # flung half a unit". What separates them is that there were TWO
    # CANDIDATES and the machine cannot be sure which one it followed. So
    # a carry frame on which more than one hand passes the association
    # gate still follows the nearest, and the carry lives, but that
    # frame's movement is NOT fling evidence. A throw is an outward,
    # irreversible action; an ambiguous frame must never be the one that
    # fires it.
    #
    # Two cheap locks go with it. ``assoc_scale_frac``: a candidate whose
    # own pose-corrected unit differs from the carried hand's by more
    # than this fraction is a hand at a different distance, not that one
    # -- which is what keeps the gate narrow enough that his other hand
    # further off is never a candidate at all. ``assoc_step_u`` is the
    # backstop for one detection TELEPORTING: 2.00 units between frames
    # is 254 mm in 133 ms, which no hand does. Both GUESSED.
    assoc_scale_frac: float = 0.45
    assoc_step_u: float = 2.00
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
    # ROUND 4. The two things a silent drop never told him: how fast the
    # hand actually went, and which bar said no.
    #
    # ``speed_us`` is the FASTEST the hand was measured travelling during
    # the carry, not the last step -- the last step of a throw is the
    # follow-through and is slow by design. It is the number the fling bar
    # was compared against, so he can read "you threw at 2.41 and the bar
    # is 2.45" instead of guessing.
    # ``refused`` is one of ``REFUSALS`` and is "" on a throw.
    speed_us: float = 0.0
    refused: str = ""
    # ROUND 5. ``speed_us`` IS NOT THE NUMBER THE BAR WAS COMPARED WITH,
    # and the comment above it said it was. ``_was_flung`` asks whether the
    # hand was at ``throw_speed_us`` inside ``fling_grant_s`` OF THE MOMENT
    # THE THROW WOULD FIRE; ``speed_us`` is the peak of the WHOLE carry, so
    # a carry with one quick correction early and a slow exit is refused on
    # speed while this number sits above the bar. MEASURED on the jerk-carry
    # family: peak 2.61 against a bar of 2.45, refused on speed. This is the
    # instrument he is being handed to settle whether his own throw is
    # inside the surface, so it now carries BOTH: the carry peak, and
    # ``fling_us`` -- the fastest CREDITED step inside the window the bar
    # actually used.
    fling_us: float = 0.0
    # ROUND 5, THE SECOND SIGNAL: how far the hand pulled BACK along the
    # axis the throw eventually took, before it swung, in hand-units.
    # Measured on every carry end, gated only when ``windup_min_u`` > 0.
    windup_u: float = 0.0

    def numbers_only(self) -> dict:
        return {"kind": self.kind, "at": round(float(self.at), 4),
                "frame": int(self.frame),
                "dist_u": round(float(self.dist_u), 4),
                "bearing_deg": round(float(self.bearing_deg), 2),
                "sector": self.sector, "why": self.why,
                "reach": round(float(self.reach), 4),
                "closed": round(float(self.closed), 4),
                "speed_us": round(float(self.speed_us), 3),
                "fling_us": round(float(self.fling_us), 3),
                "windup_u": round(float(self.windup_u), 4),
                "refused": self.refused,
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
        # The fling test. ``_last_seen_at`` is the clock at the last frame
        # that actually held the carried hand -- the exit branch fires
        # LATER than that, and the speed of the hand belongs to the frame
        # that saw it, not to the frame that noticed it was gone.
        self._last_seen_at: Optional[float] = None
        self._speed_us = 0.0
        # The FASTEST step of this carry, kept so a refusal can say what it
        # was refusing. Reset wherever _fling_at is.
        self._peak_us = 0.0
        self._fling_at = -1e9
        # ROUND 5. Two bounded records of the carry, and neither is a frame.
        # ``_win`` is (clock, credited speed) per carry frame, so a refusal
        # can quote the peak INSIDE the fling window rather than the peak of
        # the whole carry -- which is the number the bar was never compared
        # against. ``_path`` is (cx, cy, unit) per carry frame, which is
        # what the wind-up is measured out of at the end, when the axis the
        # throw took is finally known. Both are hard-bounded by the carry
        # cap; neither grows with session length.
        self._win: deque = deque(maxlen=CARRY_TRACE_FRAMES)
        self._path: deque = deque(maxlen=CARRY_TRACE_FRAMES)
        self._windup_u = 0.0
        # Tri-state, and it is the SECOND SIGNAL: was he looking at the
        # screen he grabbed, from before the reach through to the last
        # frame that held the hand? None until a frame says otherwise;
        # one False anywhere in the carry sticks.
        self._look = NO_HEAD
        self._looking = NO_HEAD
        self._jumped = False
        # The last few pose-corrected units, so the step is divided by a
        # MEDIAN rather than by one frame's estimate. The correction is a
        # max over three measured lengths, and a max over noisy numbers
        # is biased upward -- which reads as a slower hand and loses
        # throws. MEASURED at 12 px of landmark noise (well past the
        # tracker's own floor): 36 of 48 lateral throws on one frame's
        # unit, 44 of 48 on a median of three.
        self._units: deque = deque(maxlen=UNIT_MEDIAN_FRAMES)
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
                "speed_us": round(float(self._speed_us), 3),
                "peak_us": round(float(self._peak_us), 3),
                "fling_us": round(float(self.fling_us(now)), 3),
                "windup_u": round(float(self.windup_u()), 4),
                "flung": bool(self._was_flung(now)),
                "looking": ("" if self._look is NO_HEAD
                            else "?" if self._look is None
                            else ("yes" if self._look else "no")),
                "reach": round(float(self._reach), 4),
                "closed": round(float(self._closed), 4),
                "held": self.held, "ambiguous": bool(self._ambiguous),
                "cap_s": round(self.carry_cap_s(), 3),
                "frame": int(self._frame)}

    # -------------------------------------------------------- the cycle
    def update(self, hands: Sequence[HandObservation], eye_px: float,
               frame: int, looking=NO_HEAD) -> Optional[CastEvent]:
        """One camera cycle. At most one event comes back.

        ``eye_px`` is accepted for symmetry with the caller and for the
        record; the reach ratio itself is already on each observation,
        because it has to be computed from the SAME frame as the hand.

        ``looking`` is the SECOND SIGNAL and it is a TRI-STATE: True when
        his head was still pointed where it was pointed before the reach
        began, False when it has turned away by more than
        ``yaw_hold_deg``, and None when there was no clean face row to
        read -- which is no opinion, not a yes. This module has no face and
        never will; ``handstage.HandStage`` measures the angle and hands
        the verdict in.
        """
        with self._lock:
            now = float(self._now())
            self._frame = int(frame)
            self._looking = (looking if looking is NO_HEAD
                             else (None if looking is None else bool(looking)))
            self._note_look(self._looking)
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

    def _is_closed(self, o: HandObservation) -> bool:
        """A fist, on BOTH scalars. See ``closed_ratio_max``: the shipped
        one alone calls a relaxed hand at his reaching pitch a grip."""
        return (o.closed <= self.t.closed_max
                and o.curled <= self.t.closed_ratio_max)

    def _is_open(self, o: HandObservation) -> bool:
        """An open hand, on EITHER scalar. Open is the cheap, reversible
        reading, so the bars that produce it are the generous ones."""
        return (o.closed >= self.t.open_min
                or o.curled >= self.t.open_ratio_min)

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
            unit = max(self._unit, MIN_PALM_DIAG_PX)
            limit = self.t.assoc_max_u * unit
            # THE CHEAP IDENTITY CHECK. Nearest-centroid alone adopted his
            # OTHER hand mid-carry; a hand at a different distance has a
            # different unit, and that costs one subtraction to notice.
            span = self.t.assoc_scale_frac * unit
            best, best_d, seen = None, float("inf"), 0
            for h in usable:
                if abs(max(h.unit, MIN_PALM_DIAG_PX) - unit) > span:
                    continue
                d = math.hypot(h.cx - self._last[0], h.cy - self._last[1])
                if d > limit:
                    continue
                # A hand that is OPEN is not the hand holding the thing,
                # so it is not a hand this one could be confused WITH.
                # It is still allowed to win the association, because the
                # carried hand opening is exactly how a release looks.
                if self._is_closed(h):
                    seen += 1
                if d < best_d:
                    best, best_d = h, d
            if best is None:
                return None, False
            # MORE THAN ONE CANDIDATE IS AN AMBIGUOUS FRAME. The nearest
            # is still followed -- a carry must not die because he raised
            # his other hand -- but ``_carrying`` will not take that
            # frame's movement as fling evidence.
            return best, seen > 1
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
        self._open_hist.append(self._is_open(o))
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
        if self._is_closed(o) and o.reach >= self.t.reach_min:
            self._latch(o, dwell=1)
            self._state = CastState.CLOSING
            return None
        if o.reach < self.t.reach_arm:
            self._to_idle()
        return None

    def _closing(self, o: HandObservation,
                 now: float) -> Optional[CastEvent]:
        if not self._is_closed(o) or o.reach < self.t.reach_min:
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
        # THE STEP IS MEASURED IN THIS FRAME'S OWN UNIT, not in the one
        # latched at the grab: a hand that comes nearer the lens covers
        # more pixels per millimetre, and the latched unit turned that
        # into speed it did not have (a 300 mm/s sweep fired inbound and
        # not outbound, MEASURED).
        self._units.append(max(o.unit, MIN_PALM_DIAG_PX))
        step_unit = max(_median(self._units), MIN_PALM_DIAG_PX)
        # A frame the association could not be certain about, or a step no
        # hand makes, is followed but not credited: see ``assoc_step_u``.
        self._jumped = bool(self._ambiguous)
        if self._last is not None:
            step = math.hypot(o.cx - self._last[0],
                              o.cy - self._last[1]) / step_unit
            if step > self.t.assoc_step_u:
                self._jumped = True
            self._last_step_u = 0.0 if self._jumped else step
        self._note_speed(now)
        if not self._jumped:
            # THE PATH THE WIND-UP IS MEASURED OUT OF, and it takes the
            # same credit rule the fling does: a frame the association
            # could not be sure about is followed but is not evidence. A
            # wind-up made of his OTHER hand appearing is not a wind-up.
            self._path.append((o.cx, o.cy, step_unit))
        self._last = (o.cx, o.cy)
        self._dist_u = math.hypot(o.cx - self._anchor[0],
                                  o.cy - self._anchor[1]) / unit
        if self._is_open(o):
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
    def _note_look(self, looking: Optional[bool]) -> None:
        """Fold one frame's verdict into the carry's. ONE False anywhere
        in the carry sticks: "he was looking at it through the grab AND
        through the swing" is an ALL, not a last-frame sample, because the
        one frame that matters is exactly the one the arm is across."""
        if self._state not in (CastState.CLOSING, CastState.CARRYING):
            return
        if looking is NO_HEAD:
            return
        if self._look is NO_HEAD:
            self._look = looking
            return
        if looking is False:
            self._look = False
        elif looking is True and self._look is not False:
            self._look = True

    def _note_speed(self, now: float) -> None:
        """How fast the hand was travelling on the step just measured, and
        WHEN it was last travelling like a fling.

        In hand-units per SECOND, off the injected clock rather than off
        the frame counter, so a dropped frame or a jittered interval
        reports the speed the hand actually had rather than half of it. A
        cycle with no measurable interval -- a hand-wound clock that has
        not moved -- falls back to one frame period at the tuned rate; it
        must never read as infinite speed, which is how a bar like this
        becomes a hole.
        """
        seen = self._last_seen_at
        dt = (now - seen) if seen is not None else 0.0
        if dt <= 0.0:
            dt = 1.0 / self.preview_fps
        self._speed_us = self._last_step_u / dt
        if self._speed_us > self._peak_us:
            self._peak_us = self._speed_us
        # A step that had to be a RE-ASSOCIATION is not travel, so it is
        # not fling evidence either -- however fast the arithmetic makes
        # it look. The hand is still followed; only the credit is refused.
        if not self._jumped:
            # THE SAMPLE THE BAR IS ACTUALLY COMPARED WITH. Credited steps
            # only, exactly like ``_fling_at`` below -- a re-association is
            # not travel, so it is not evidence and must not be quoted back
            # to him as a speed he made.
            self._win.append((now, float(self._speed_us)))
        if self._speed_us >= self.t.throw_speed_us and not self._jumped:
            self._fling_at = now
        self._last_seen_at = now

    def _was_flung(self, now: float) -> bool:
        """Was the hand still FLINGING ``fling_window_s`` ago or later?

        Leaving the picture is not intent -- a man carrying a mug leaves
        the picture -- so a carry becomes a throw only if the hand was
        travelling at ``throw_speed_us`` inside the window.

        THE WINDOW IS MEASURED TO NOW, WHICH IS WHEN THE THROW WOULD FIRE.
        Round 2 measured it to the last frame that SAW the hand and then
        let ``lost_grace_frames`` run on top, so the documented 0.55 s was
        really 1.07 s at 7.5 fps and 1.35 s at 3.7 (MEASURED: one fast
        frame, then a 0.53 s crawl at a fifth of the bar, then 0.40 s
        absent, and the exit still resolved as a throw). The grant is now
        the number in the field, grace included -- 0.55 s total at 7.5 fps,
        of which 0.40 s is the grace and 0.15 s is slack.

        WITH ONE FLOOR, and it is arithmetic rather than a concession: an
        exit fires ``lost_grace_frames`` + 1 frames after the last frame
        that saw the hand, so a window shorter than that can never be
        satisfied by an exit AT ALL. At 5.0 fps that is 0.60 s and at
        3.7 fps 0.81 s, both longer than 0.55, and without the floor the
        whole exit branch went dead below 5.5 fps (MEASURED: lateral
        throws 12 of 24 at 5.0 fps and 0 of 8 at 2.0 fps). The floor
        gives a slow feed the grace and NOTHING MORE -- no slack for a
        crawl -- which is the tightest honest answer there.
        """
        return (float(now) - self._fling_at) <= self.fling_grant_s()

    def fling_us(self, now: float) -> float:
        """The fastest CREDITED step inside the window the fling bar was
        compared against, in hand-units per second.

        ROUND 5, AND IT IS A DIAGNOSTIC FIX WITH TEETH. ``_peak_us`` is the
        peak of the WHOLE carry and the refusal quoted it, so a carry with
        one quick correction early and a slow exit was refused on speed
        while the number printed sat ABOVE the bar -- it could tell him he
        threw at 2.61 against a bar of 2.45 and was refused on speed. This
        is the number that decided it, and the law it keeps is exact:
        ``fling_us(now) >= throw_speed_us`` is true if and only if
        ``_was_flung(now)`` is, because both read the same credited samples
        over the same grant.
        """
        grant = self.fling_grant_s()
        best = 0.0
        for at, speed in self._win:
            if (float(now) - at) <= grant and speed > best:
                best = speed
        return best

    def windup_u(self) -> float:
        """How far the hand pulled BACK before it swung, in hand-units.

        Measured along the axis the throw eventually took -- anchor to the
        last frame that held the fist -- as the most negative projection of
        the carry path, over the PREFIX up to the furthest forward point.
        The prefix is what makes it a wind-up rather than a return: a hand
        that goes out and comes back (``exit-return``, one of the
        attacker's own non-gesture families) has its retraction AFTER the
        furthest point and scores zero.

        Each point is divided by the unit measured on ITS OWN frame, so the
        number is a ratio and does not move with his reach.
        """
        anchor = self._anchor
        end = self._last_fist or self._last
        if anchor is None or end is None or len(self._path) < 2:
            return 0.0
        ax, ay = end[0] - anchor[0], end[1] - anchor[1]
        span = math.hypot(ax, ay)
        if span < MIN_PALM_DIAG_PX:
            return 0.0
        ax, ay = ax / span, ay / span
        projs = [((px - anchor[0]) * ax + (py - anchor[1]) * ay)
                 / max(unit, MIN_PALM_DIAG_PX) for px, py, unit in self._path]
        if not projs:
            return 0.0
        far = projs.index(max(projs))
        back = min(projs[:far + 1])
        return -back if back < 0.0 else 0.0

    def fling_grant_s(self) -> float:
        """How long ago the hand may last have been flinging and the exit
        still be a throw, in seconds, at the rate this machine is tuned
        to. For the bring-up log and for the tests that pin the law.

        ``fling_window_s``, floored at the loss grace plus half a frame.
        The half frame is not slack for a crawl: the exit fires a whole
        number of frames after the last sighting, and requiring the fling
        to land on that exact frame turns one unlucky sub-frame phase
        into a lost gesture (MEASURED at 5.0 fps: 12 of 24 lateral throws
        with no half frame, 24 of 24 with it). At 7.5 fps the floor is
        0.47 s and does not bind at all.
        """
        period = 1.0 / max(self.preview_fps, 0.1)
        return max(float(self.t.fling_window_s),
                   (int(self.t.lost_grace_frames) + 1.5) * period)

    def _carry_expired(self, now: float) -> bool:
        if self._carry > self.t.carry_max_frames:
            return True
        started = self._carry_started
        return (started is not None
                and (now - started) > float(self.t.carry_max_s))

    def _latch(self, o: HandObservation, dwell: int) -> None:
        self._anchor = (o.cx, o.cy)
        self._unit = max(o.unit, MIN_PALM_DIAG_PX)
        self._units.clear()
        self._units.append(self._unit)
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
        self._win.clear()
        self._path.clear()
        self._windup_u = 0.0
        self._last_seen_at = None
        self._speed_us = 0.0
        self._peak_us = 0.0
        self._fling_at = -1e9
        self._look = NO_HEAD
        self._jumped = False

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
        # The carry starts here and so does the fling clock. A fist that
        # was moving before it was ever picked up brings no credit with it.
        self._last_seen_at = now
        self._speed_us = 0.0
        self._peak_us = 0.0
        self._fling_at = -1e9
        self._jumped = False
        # The traces start EMPTY except for the anchor itself, which is
        # where the wind-up is measured from. A fist that was moving before
        # it was picked up brings no credit with it, and no path either.
        self._win.clear()
        self._path.clear()
        self._path.append((o.cx, o.cy, max(self._unit, MIN_PALM_DIAG_PX)))
        self._windup_u = 0.0
        # The look starts from THIS frame's opinion: the grab is part of
        # "through the grab and through the swing".
        self._look = self._looking
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
        self._last_seen_at = None
        self._speed_us = 0.0
        self._peak_us = 0.0
        self._fling_at = -1e9
        self._win.clear()
        self._path.clear()
        self._windup_u = 0.0
        self._look = NO_HEAD
        self._jumped = False

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
        # TAKEN NOW, because _enter_cooldown below clears it and the whole
        # point of the field is that the refusal can say what it refused.
        peak_us = float(self._peak_us)
        # ...and the number the bar was ACTUALLY compared with, which is
        # not that one. See ``fling_us``.
        window_us = float(self.fling_us(now))
        # THE WIND-UP, measured before the traces are cleared. Measured on
        # every carry end, gated only when ``windup_min_u`` is above zero.
        windup = float(self.windup_u())
        self._windup_u = windup
        end = self._last_fist or self._last or anchor
        unit = max(self._unit, MIN_PALM_DIAG_PX)
        dx, dy = end[0] - anchor[0], end[1] - anchor[1]
        dist_u = math.hypot(dx, dy) / unit
        vec_bearing = bearing_deg(*to_his_frame(dx, dy, self.mirrored))
        why, thrown, bearing = reason, False, vec_bearing
        # ROUND 4: WHICH BAR REFUSED IT, from REFUSALS. A carry that never
        # reached the three-bar decision at all -- a timeout, a stall, a
        # hand pulled back, a word -- is "carry" and is NOT a gesture that
        # failed; it is him having put the thing down.
        refused = "carry"
        # A THROW MUST BE A THROW. Every distance bar below is kept exactly
        # as it was and every one of them now has this in front of it: the
        # hand has to have been FLINGING when the picture last held it.
        flung = self._was_flung(now)

        if reason == "released":
            far = dist_u >= self.t.throw_release_u
            thrown = far and flung
            why = "released" if (flung or not far) else "set down, not flung"
            refused = "" if thrown else ("distance" if not far else "speed")
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
                far = dist_u >= self.t.throw_exit_u
                thrown = far and flung
                why = ("left frame (his %s)" % edge if (flung or not far)
                       else "carried out of frame, not flung")
                refused = "" if thrown else ("distance" if not far
                                             else "speed")
            else:
                far = (dist_u >= self.t.throw_lost_u
                       and self._last_step_u >= self.t.exit_step_u)
                thrown = far and flung
                why = "lost" if (flung or not far) else "lost, not flung"
                refused = "" if thrown else ("distance" if not far
                                             else "speed")
        # 'timeout', 'stalled', 'withdrawn' and 'cancelled (...)' are never
        # throws: a carry that ran out, was pulled back closed, or that he
        # ended with a word, is him having put it down, not flung it.

        # THE SECOND SIGNAL, AND IT IS ANDED WITH THE FIRST, NOT ORED.
        # Speed is measured to be unable to separate his throw from his
        # brisk desk motion -- both sit on the same side of the bar -- so
        # the throw also needs him to have been LOOKING at the screen he
        # grabbed, from before the reach through to the last frame that
        # held the hand. A carry that he watched leave, or one made while
        # his head was turned, is a drop. NO OPINION (no clean face row)
        # falls whichever way ``yaw_required`` says, and it says refuse.
        # THE THIRD SIGNAL, ROUND 5, AND IT IS ANDED LIKE THE SECOND. A
        # throw pulls back before it swings; carrying something away never
        # reverses before it leaves. ``windup_min_u`` ships at 0.0, which
        # refuses nothing -- the measurement is the deliverable and the gate
        # is off, because the adversarial retraction families measure the
        # same retraction his throw does (castgrid/test_r5_windup.py).
        if thrown and self.t.windup_min_u > 0.0 \
                and windup < self.t.windup_min_u:
            thrown, refused = False, "windup"
            why = "%s (no wind-up: pulled back %.2f u, needs %.2f)" \
                % (why, windup, self.t.windup_min_u)

        if thrown and self._look is not NO_HEAD:
            if self._look is False:
                thrown, refused = False, "look"
                why = "%s (he was not looking at it)" % why
            elif self._look is None and self.t.yaw_required:
                thrown, refused = False, "look"
                why = "%s (no clean face to read his head from)" % why

        where = sector(bearing, self.t.sector_half_deg)
        if thrown and where == "ambiguous":
            # Measured but unnameable. Ambiguity resolves to the cheap,
            # reversible outcome, always.
            thrown, refused = False, "direction"
            why = "%s (ambiguous direction)" % why
        elif thrown and where not in self.t.target_sectors:
            thrown = False
            if where == "down":
                refused = "cancel"
                why = "cancelled (his down)"
            else:
                refused = "sector"
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
            speed_us=peak_us, fling_us=window_us, windup_u=windup,
            refused=refused,
            toward=where if dist_u >= TOWARD_MIN_U else ""))


def describe(t: CastThresholds = CastThresholds(),
             fps: float = 6.0) -> str:
    """One line for a bring-up log: what the frame counters mean in time."""
    ms = 1000.0 / max(float(fps), 0.1)
    return ("cast: C<=%.2f & curl<=%.2f closed / C>=%.2f or curl>=%.2f "
            "open, R>=%.2f grab (arm %.2f), dwell %d fr (%.0f ms), grace "
            "%d fr, cooldown %d fr, cap %d fr (%.1f s) or %.1f s, fling "
            ">=%.2f u/s within %.2f s of firing, head within %.0f deg "
            "(no face %s), targets %s at %.1f fps"
            % (t.closed_max, t.closed_ratio_max, t.open_min,
               t.open_ratio_min, t.reach_min, t.reach_arm,
               t.dwell_frames, t.dwell_frames * ms, t.lost_grace_frames,
               t.cooldown_frames, t.carry_max_frames,
               t.carry_max_frames * ms / 1000.0, t.carry_max_s,
               t.throw_speed_us, t.fling_window_s, t.yaw_hold_deg,
               "refuses" if t.yaw_required else "allows",
               "/".join(t.target_sectors), fps))


__all__ = [
    "CallablePayload", "CastEvent", "CastGesture", "CastState",
    "CastThresholds", "DESIGN_FPS", "HandObservation", "Payload",
    "HAND_UNIT_MM", "PALM", "PALM_DIAG_OVER_IPD", "PALM_LEN_MM",
    "REFUSALS",
    "PALM_W_MM", "SECTORS", "TIPS",
    "NO_HEAD",
    "bearing_deg", "describe", "edge_bearing", "observe_hand", "sector",
    "to_his_frame",
]
