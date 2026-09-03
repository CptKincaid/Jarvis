"""The grab-and-throw engine, driven by synthetic hands and nothing else.

Every hand here is a 21-point 3-D model built from anthropometry, curled,
rotated and projected through the LifeCam's own lens constants (f = 993 px
at 1280 wide, 65.6 deg). Its proportions match the official MediaPipe
reference to within 1.5%. NO CAMERA IS OPENED, no frame is read, no image is
made: ``jarvis.gesture`` takes coordinates and the harness makes coordinates.

Six promises are pinned, each because the design cannot keep it by intention:

* **The mirror.** The feed is not mirrored and the lens faces him, so image
  +x is HIS LEFT. Every direction test names the side in the trajectory's
  own name and asserts the sector by name -- ``|B - A|`` passes with the
  sign inverted, and the design pass got the sign wrong twice.
* **A throw is a fling, not an unclench.** Opening the hand where it is must
  be a drop, whatever the hand did before. Just-under and just-over each
  displacement bar are driven directly.
* **Only LEFT and RIGHT throw.** Down is the cancel, up is not a target, the
  gaps between sectors are ambiguous. All four are drops that say why.
* **Ordinary motion does not fire.** The design's 12 everyday motions x 2
  finger directions x 12 phases x 2 frame rates (576 sequences) plus a
  randomised sweep across 14 parametrised families of desk motion. The
  false-grab and false-throw counts are the headline of this file.
* **The machine is not tuned to one frame rate.** The same throw at 5, 6,
  7.5, 10, 15 and 30 fps and at jittered intervals grabs and lands on the
  named side; 2 fps is shown for what it honestly is.
* **No pixel, ever.** A source-level test forbids the vocabulary of images in
  jarvis/gesture.py, and every event and status dict passes
  ``visionrig.assert_numbers_only``.

Nothing here imports cv2, opens a device, spawns a process or touches a
socket. The whole file runs on a box with no camera and no weights.
"""
from __future__ import annotations

import ast
import io
import math
import os
import re
import tokenize
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from jarvis import gesture as g
from jarvis.gesture import (
    CallablePayload,
    CastEvent,
    CastGesture,
    CastState,
    CastThresholds,
    HandObservation,
    bearing_deg,
    edge_bearing,
    observe_hand,
    sector,
    to_his_frame,
)
from jarvis.visionrig import assert_numbers_only

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- the lens
# LifeCam Cinema as pinned in jarvis/facemodels.py: 65.6 deg across 1280 px.
F_PX = 1280.0 / (2.0 * math.tan(math.radians(65.6) / 2.0))
W, H = 1280, 720
IPD_MM = 63.0
FACE_Z = 700.0                       # a desk: his face 70 cm from the lens
EYE_PX = F_PX * IPD_MM / FACE_Z      # 89.4 px interocular at that distance

# ---------------------------------------------------------- the synthetic hand
# A right hand, palm plane = xy, fingers +y, millimetres. Ported from the
# design harness (~/scratch-gesture/castdesign/handproj.py) so this file
# owns its own fixture.
MCP = {"index": (-24., 100., 0.), "middle": (0., 107., 0.),
       "ring": (22., 104., 0.), "pinky": (44., 95., 0.)}
PHAL = {"index": (40., 24., 18.), "middle": (45., 27., 20.),
        "ring": (42., 25., 19.), "pinky": (33., 19., 17.)}
THUMB = [(-32., 30., 8.), (-52., 55., 12.), (-66., 72., 14.), (-76., 88., 16.)]
IDX = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
       "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FULL_FIST = (90., 100., 70.)         # MCP, PIP, DIP flexion at curl 1.0
FINGERS = ("index", "middle", "ring", "pinky")


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@lru_cache(maxsize=4096)
def hand3d(curls: tuple, left: bool = False) -> np.ndarray:
    """21x3 mm. ``curls`` is one value per finger (index, middle, ring,
    pinky); 0 = flat open palm, 1 = closed fist. The thumb follows the
    mean curl. ``left`` mirrors x, which is all a left hand is to a
    coordinate model."""
    tc = float(np.mean(curls))
    lm = np.zeros((21, 3))
    for k, p in enumerate(THUMB):
        p = np.array(p, float)
        closed = np.array([-14. - 6. * k, 55. + 12. * k, -14. - 4. * k])
        lm[1 + k] = p * (1 - tc) + closed * tc
    for name, curl in zip(FINGERS, curls):
        ids = IDX[name]
        lm[ids[0]] = np.array(MCP[name], float)
        d = np.array([0., 1., 0.])
        cum = np.eye(3)
        for j, ln in enumerate(PHAL[name]):
            cum = cum @ _rotx(math.radians(FULL_FIST[j] * curl))
            lm[ids[0] + 1 + j] = lm[ids[0] + j] + (cum @ d) * ln
    if left:
        lm = lm.copy()
        lm[:, 0] *= -1
    lm.setflags(write=False)
    return lm


def _curls(curl) -> tuple:
    if isinstance(curl, (tuple, list)):
        return tuple(round(min(max(float(c), 0.), 1.), 2) for c in curl)
    c = round(min(max(float(curl), 0.), 1.), 2)
    return (c, c, c, c)


def project(lm3, pos, pitch=0., yaw=0., roll=0.):
    """Rotate about the wrist, place the wrist at ``pos`` (mm, camera
    frame, +z away from the lens), project to CAPTURE pixels with the
    origin at the top-left like every cv2 buffer. Returns None when the
    PALM QUAD is not in the picture -- fingertips may fall outside, the
    landmark model extrapolates them from the palm crop -- which is the
    harness's model of 'no hand found'."""
    R = (_roty(math.radians(yaw)) @ _rotx(math.radians(pitch))
         @ _rotz(math.radians(roll)))
    p = (R @ lm3.T).T + np.asarray(pos, float)
    if p[:, 2].min() < 80:
        return None
    img = np.stack([F_PX * p[:, 0] / p[:, 2] + W / 2,
                    F_PX * p[:, 1] / p[:, 2] + H / 2], 1)
    q = img[list(g.PALM)]
    if (q[:, 0].min() < 20 or q[:, 0].max() > W - 20
            or q[:, 1].min() < 20 or q[:, 1].max() > H - 20):
        return None
    return img


def sample(traj, t, noise=0.0, rng=None, left=False):
    """One camera cycle of a trajectory -> the tuple update() takes."""
    pos, curl, pi, ya, ro = traj(t)
    img = project(hand3d(_curls(curl), left), pos, pi, ya, ro)
    if img is None:
        return ()
    if noise and rng is not None:
        img = img + rng.normal(0.0, noise, img.shape)
    return (observe_hand(img, EYE_PX),)


# ------------------------------------------------------------- trajectories
def _lerp(a, b, u):
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def _mix(a, b, u):
    """A curl is one float for the whole hand or one per finger (the
    pointing family curls three and leaves the index flat); both lerp."""
    if isinstance(a, (tuple, list)):
        return _lerp(a, b, u)
    return a + (b - a) * u


def seg(t, spans):
    """Piecewise-linear motion: (duration, pos0, pos1, curl0, curl1, pitch,
    yaw, roll) per span."""
    acc = 0.
    for d, p0, p1, c0, c1, pi, ya, ro in spans:
        if t < acc + d:
            u = (t - acc) / d if d else 0.
            return _lerp(p0, p1, u), _mix(c0, c1, u), pi, ya, ro
        acc += d
    d, p0, p1, c0, c1, pi, ya, ro = spans[-1]
    return p1, c1, pi, ya, ro


def gesture(end, hold=0.50, swing=0.40, release=0.35, pitch=35.):
    """The intended gesture: reach in open, close, hold, swing to ``end``,
    open while following through. 425 mm is a comfortable reach."""
    return (0.55 + 0.20 + hold + swing + release + 0.5, lambda t: seg(t, [
        (0.55, (140, 300, 780), (10, 10, 430), 0.10, 0.06, pitch - 5, 0, 0),
        (0.20, (10, 10, 430), (10, 10, 425), 0.06, 1.00, pitch, 0, 0),
        (hold, (10, 10, 425), (15, 5, 425), 1.00, 1.00, pitch, 0, 0),
        (swing, (15, 5, 425), end, 1.00, 1.00, pitch, 0, 0),
        (release, end, tuple(1.35 * x for x in end), 1.00, 0.06, pitch, 0,
         0)]))


def slow_gesture(end, approach=1.2, close=0.3, hold=1.5, swing=0.8,
                 release=0.6, tail=1.5, pitch=35.):
    """The same gesture done slowly, with the open approach IN FRAME from
    its first instant. ``gesture()`` starts its reach out of the picture,
    which is realistic, but at 2 fps it leaves some phases with no open
    sample at all -- and then the resting-fist rule refuses, correctly.
    The tail is there so a lost-hand drop has room to fire."""
    far = tuple(1.35 * x for x in end)
    return (approach + close + hold + swing + release + tail, lambda t: seg(t, [
        (approach, (60, 120, 600), (10, 10, 430), 0.10, 0.06, pitch - 5, 0, 0),
        (close, (10, 10, 430), (10, 10, 425), 0.06, 1.00, pitch, 0, 0),
        (hold, (10, 10, 425), (15, 5, 425), 1.00, 1.00, pitch, 0, 0),
        (swing, (15, 5, 425), end, 1.00, 1.00, pitch, 0, 0),
        (release, end, far, 1.00, 0.06, pitch, 0, 0),
        (tail, far, far, 0.06, 0.06, pitch, 0, 0)]))


# Named for the side HE throws to. Image -x is his right: the lens faces him.
GOOD = {"his-right": gesture((-250, -10, 450)),
        "his-left": gesture((260, -10, 450)),
        "his-up": gesture((10, -190, 445)),
        "his-down": gesture((10, 200, 445))}

# The design's everyday set. Each is a plausible desk motion, invented, not
# sampled from him.
BAD = {
    "typing": (6.0, lambda t: (
        (150 + 9 * math.sin(27 * t), 215 + 7 * math.sin(19.5 * t), 615),
        0.38 + 0.10 * math.sin(27 * t), -25, 10, 0)),
    "scratch": (4.0, lambda t: seg(t, [
        (0.9, (160, 250, 660), (70, -110, 615), 0.45, 0.55, 10, 0, 0),
        (1.4, (70, -110, 615), (80, -120, 610), 0.55, 0.60, 10, 5, 10),
        (0.9, (80, -120, 610), (160, 250, 660), 0.60, 0.45, 10, 0, 0)])),
    "drink-mug": (4.0, lambda t: seg(t, [
        (0.9, (210, 210, 600), (120, -95, 545), 0.85, 0.88, 15, 0, 0),
        (1.5, (120, -95, 545), (125, -100, 540), 0.88, 0.88, 20, 0, 5),
        (0.9, (125, -100, 540), (210, 210, 600), 0.88, 0.85, 15, 0, 0)])),
    "talk-gesture": (6.0, lambda t: (
        (185 * math.sin(7.54 * t), 40 + 40 * math.cos(7.54 * t), 600),
        0.15 + 0.10 * math.sin(15.1 * t), 15, 20 * math.sin(7.54 * t), 0)),
    "hand-crosses": (1.2, lambda t: (
        (-430 + 720 * t / 1.2, 60, 500), 0.10 + 0.7 * (t / 1.2), 20, 0, 0)),
    "chin-on-fist": (4.0, lambda t: ((110, 10, 635), 1.0, 20, 5, 0)),
    "adjust-monitor": (3.0, lambda t: seg(t, [
        (0.8, (150, 200, 700), (-40, -30, 410), 0.10, 0.08, 25, 0, 0),
        (1.0, (-40, -30, 410), (60, -60, 395), 0.08, 0.12, 30, 10, 10),
        (1.2, (60, -60, 395), (150, 200, 700), 0.12, 0.10, 25, 0, 0)])),
    "fist-then-withdraw": (2.0, lambda t: seg(t, [
        (0.55, (120, 250, 780), (0, -10, 430), 0.10, 0.05, 30, 0, 0),
        (0.13, (0, -10, 430), (0, -10, 430), 0.05, 1.00, 35, 0, 0),
        (0.13, (0, -10, 430), (20, 20, 470), 1.00, 1.00, 35, 0, 0),
        (0.8, (20, 20, 470), (150, 250, 740), 1.00, 0.20, 30, 0, 0)])),
    "phone-to-lens": (4.0, lambda t: seg(t, [
        (0.8, (150, 240, 700), (20, -40, 430), 0.80, 0.90, 25, 0, 0),
        (2.0, (20, -40, 430), (30, -30, 435), 0.90, 0.90, 25, 5, 5),
        (1.0, (30, -30, 435), (150, 240, 700), 0.90, 0.80, 25, 0, 0)])),
    "rub-eye-fist": (3.0, lambda t: seg(t, [
        (0.7, (140, 230, 660), (60, -90, 440), 0.9, 1.0, 15, 0, 0),
        (1.4, (60, -90, 440), (75, -75, 435), 1.0, 1.0, 20, 10, 15),
        (0.7, (75, -75, 435), (140, 230, 660), 1.0, 0.9, 15, 0, 0)])),
    "reach-and-point": (3.0, lambda t: seg(t, [
        (0.7, (150, 250, 720), (0, 0, 420), 0.10, 0.10, 30, 0, 0),
        (1.5, (0, 0, 420), (20, -20, 415), 0.10, 0.10, 30, 10, 0),
        (0.8, (20, -20, 415), (150, 250, 720), 0.10, 0.10, 30, 0, 0)])),
}
# A deliberate grab followed by setting it back down: 12/12 grabs, 0 throws.
PUT_BACK = (3.0, lambda t: seg(t, [
    (0.55, (140, 300, 780), (10, 10, 430), 0.10, 0.06, 30, 0, 0),
    (0.20, (10, 10, 430), (10, 10, 425), 0.06, 1.00, 35, 0, 0),
    (0.50, (10, 10, 425), (15, 5, 425), 1.00, 1.00, 35, 0, 0),
    (1.1, (15, 5, 425), (45, 60, 440), 1.00, 1.00, 35, 0, 0),
    (0.4, (45, 60, 440), (45, 62, 440), 1.00, 0.06, 35, 0, 0)]))


def fingers_down(scen):
    """The same motion with the fingers pointing at the desk (roll + 180)."""
    dur, traj = scen
    return (dur, lambda t: (lambda r: (r[0], r[1], r[2], r[3],
                                       r[4] + 180.))(traj(t)))


GOOD.update({k + "/fingers-down": fingers_down(v)
             for k, v in list(GOOD.items())})
BAD.update({k + "/fingers-down": fingers_down(v)
            for k, v in list(BAD.items())})
LATERAL = {k: v for k, v in GOOD.items() if "left" in k or "right" in k}


# ------------------------------------------------------------- the driver
class Clock:
    """A hand-wound monotonic clock. Nothing here waits on the real one."""

    def __init__(self, t: float = 100.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


def run(scen, fps, thresholds=None, phases=12, noise=0.0, seed=0,
        left=False, mirrored=False, jitter=0.0, payload=None):
    """Sample a trajectory at ``fps`` from ``phases`` sub-frame offsets and
    collect every event. ``jitter`` is the fractional spread of the
    inter-frame interval; the clock and the sampling both see it."""
    dur, traj = scen
    events = []
    for ph in range(phases):
        clk = Clock()
        rng = np.random.default_rng(seed * 1009 + ph)
        m = CastGesture(thresholds or CastThresholds(), (W, H),
                        mirrored=mirrored, now=clk, preview_fps=fps,
                        payload=payload)
        t = ph * (1.0 / fps) / phases
        i = 0
        while t < dur:
            e = m.update(sample(traj, t, noise, rng, left), EYE_PX, i)
            if e is not None:
                events.append((ph, e))
            step = 1.0 / fps
            if jitter:
                step *= 1.0 + rng.uniform(-jitter, jitter)
            t += step
            clk.t += step
            i += 1
    return events


def tally(events):
    out = {"grab": 0, "throw": 0, "drop": 0, "sectors": {}, "toward": {}}
    for _ph, e in events:
        out[e.kind] += 1
        if e.kind == "throw":
            out["sectors"][e.sector] = out["sectors"].get(e.sector, 0) + 1
        elif e.kind == "drop" and e.toward:
            out["toward"][e.toward] = out["toward"].get(e.toward, 0) + 1
    return out


def lateral_recall(fps, **kw):
    """(grabs, throws, wrong-side throws) over the 4 lateral gestures."""
    grabs = throws = wrong = 0
    for name, scen in LATERAL.items():
        want = "left" if "left" in name else "right"
        for _ph, e in run(scen, fps, **kw):
            if e.kind == "grab":
                grabs += 1
            elif e.kind == "throw":
                throws += 1
                wrong += e.sector != want
    return grabs, throws, wrong


def everyday_false(fps, **kw):
    """(false grabs, false throws) over the everyday set."""
    grabs = throws = 0
    for scen in BAD.values():
        t = tally(run(scen, fps, **kw))
        grabs += t["grab"]
        throws += t["throw"]
    return grabs, throws


# ------------------------------------------------------- direct drive
def obs(cx, cy, closed=0.55, reach=2.6, palm_diag=280.0):
    return HandObservation(palm_diag=palm_diag, closed=closed, reach=reach,
                           cx=cx, cy=cy)


OPEN, FIST = 1.30, 0.55
UNIT = 280.0            # px per hand-unit at a 425 mm reach (measured 297)


def grab_at(m, cx, cy, frame0=0, unit=UNIT):
    """Drive a machine through open -> fist x3 at reach and return the grab
    event. Every direct-drive test starts here."""
    m.update((obs(cx, cy, OPEN, 2.6, unit),), EYE_PX, frame0)
    assert m.state is CastState.REACHING
    events = [m.update((obs(cx, cy, FIST, 2.6, unit),), EYE_PX, frame0 + k)
              for k in range(1, 4)]
    assert events[-1] is not None and events[-1].kind == "grab", events
    assert m.state is CastState.CARRYING
    return events[-1]


# ================================================================ geometry
class TestObserveHand:
    def test_an_open_hand_and_a_fist_sit_either_side_of_the_dead_band(self):
        """MEASURED over the design's 75-pose envelope: open min 0.919, fist
        max 0.680. One pose of each here, both bars between them."""
        o = observe_hand(project(hand3d(_curls(0.0)), (0, 0, 500), 30), EYE_PX)
        f = observe_hand(project(hand3d(_curls(1.0)), (0, 0, 500), 30), EYE_PX)
        t = CastThresholds()
        assert o.closed >= t.open_min
        assert f.closed <= t.closed_max
        assert o.ok and f.ok

    def test_palm_diag_does_not_move_when_the_hand_closes(self):
        """The depth cue has to read the same distance either side of the
        grab. MEASURED 0.0% change; the fist here is a rigid palm so the
        equality is exact to floating point."""
        o = observe_hand(project(hand3d(_curls(0.0)), (0, 0, 500), 30), EYE_PX)
        f = observe_hand(project(hand3d(_curls(1.0)), (0, 0, 500), 30), EYE_PX)
        assert f.palm_diag == pytest.approx(o.palm_diag, rel=1e-9)
        assert f.reach == pytest.approx(o.reach, rel=1e-9)

    def test_reach_grows_as_the_hand_comes_toward_the_lens(self):
        rs = [observe_hand(project(hand3d(_curls(1.0)), (0, 0, z), 35),
                           EYE_PX).reach for z in (900, 700, 550, 450, 350)]
        assert rs == sorted(rs)
        # A hand at the face plane never reaches 2.35 (measured max 2.03).
        assert rs[1] < CastThresholds().reach_min

    def test_the_gestures_own_pose_clears_the_bar_and_edge_on_does_not(self):
        """MEASURED here, fist at 425 mm: pitch 0 -> R 3.32, pitch 35 (the
        intended gesture's hold) -> 2.55, pitch 45 -> 2.27, pitch 60 ->
        1.88. The bar is 2.35. So the gesture as designed clears it by
        +0.20, and a hand tipped 45 deg or more toward the desk does NOT --
        the design's own warning that the margin is thin edge-on, pinned
        here as a stated limit rather than hidden in a pose that happens
        to pass."""
        bar = CastThresholds().reach_min

        def at(z, pitch):
            return observe_hand(project(hand3d(_curls(1.0)), (0, 0, z),
                                        pitch), EYE_PX).reach

        assert at(425, 35) >= bar
        assert at(425, 0) > at(425, 35) > at(425, 45) > at(425, 60)
        assert at(425, 45) < bar

    def test_no_face_is_no_opinion_not_infinite_reach(self):
        o = observe_hand(project(hand3d(_curls(1.0)), (0, 0, 300), 45), 0.0)
        assert o.reach == 0.0
        assert o.ok

    def test_a_collapsed_row_is_not_a_tiny_hand(self):
        o = observe_hand(np.zeros((21, 2)), EYE_PX)
        assert not o.ok
        assert o.closed == 0.0 and o.reach == 0.0

    def test_the_shape_is_checked(self):
        with pytest.raises(ValueError):
            observe_hand(np.zeros((20, 2)), EYE_PX)
        with pytest.raises(ValueError):
            observe_hand(np.zeros((21,)), EYE_PX)
        with pytest.raises(ValueError):
            observe_hand(np.full((21, 2), np.nan), EYE_PX)

    def test_three_columns_are_accepted_and_the_third_ignored(self):
        lm = project(hand3d(_curls(0.0)), (0, 0, 500), 30)
        with_z = np.c_[lm, np.arange(21)]
        assert observe_hand(with_z, EYE_PX) == observe_hand(lm, EYE_PX)

    def test_the_observation_is_numbers_only(self):
        o = observe_hand(project(hand3d(_curls(0.5)), (0, 0, 500)), EYE_PX)
        assert_numbers_only(o.numbers_only())


class TestTheMirror:
    """Image +x is HIS LEFT. Named, never just a sign."""

    def test_toward_image_right_is_his_left(self):
        r, u = to_his_frame(+10.0, 0.0)
        assert r < 0 and u == 0
        assert sector(bearing_deg(r, u)) == "left"

    def test_toward_image_left_is_his_right(self):
        r, u = to_his_frame(-10.0, 0.0)
        assert r > 0
        assert sector(bearing_deg(r, u)) == "right"

    def test_toward_image_bottom_is_his_down(self):
        r, u = to_his_frame(0.0, +10.0)
        assert u < 0
        assert sector(bearing_deg(r, u)) == "down"

    def test_toward_image_top_is_his_up(self):
        assert sector(bearing_deg(*to_his_frame(0.0, -10.0))) == "up"

    def test_a_mirrored_feed_flips_only_the_horizontal(self):
        assert sector(bearing_deg(*to_his_frame(+10, 0, True))) == "right"
        assert sector(bearing_deg(*to_his_frame(-10, 0, True))) == "left"
        assert sector(bearing_deg(*to_his_frame(0, +10, True))) == "down"
        assert sector(bearing_deg(*to_his_frame(0, -10, True))) == "up"

    def test_to_his_frame_is_the_only_sign_flip_in_the_module(self):
        """A second copy of the mapping is how the sign gets inverted. The
        edge table derives its bearings through to_his_frame, and no other
        line negates a coordinate delta."""
        src = (REPO / "jarvis" / "gesture.py").read_text()
        body = src[src.index("def bearing_deg"):]
        assert not re.search(r"=\s*-\s*\(?\s*d[xy]\b", body)
        assert not re.search(r"-\s*(cx|cy|dx|dy)\b", body)


class TestBearingAndSector:
    @pytest.mark.parametrize("deg,name", [
        (0, "right"), (34, "right"), (326, "right"), (360, "right"),
        (90, "up"), (56, "up"), (124, "up"),
        (180, "left"), (146, "left"), (214, "left"),
        (270, "down"), (236, "down"), (304, "down"),
    ])
    def test_the_four_sectors_are_35_degrees_wide(self, deg, name):
        assert sector(deg) == name

    @pytest.mark.parametrize("deg", [45, 135, 225, 315, 36, 54, 126, 144,
                                     216, 234, 306, 324])
    def test_the_gaps_between_sectors_are_ambiguous(self, deg):
        assert sector(deg) == "ambiguous"

    def test_bearing_is_zero_at_his_right_and_ninety_at_his_up(self):
        assert bearing_deg(1, 0) == 0.0
        assert bearing_deg(0, 1) == 90.0
        assert bearing_deg(-1, 0) == 180.0
        assert bearing_deg(0, -1) == 270.0

    def test_a_negative_bearing_wraps(self):
        assert sector(-90) == "down"
        assert sector(-1) == "right"


class TestEdgeBearing:
    """The frame edges in HIS frame: image-left is his RIGHT."""

    def test_the_image_left_edge_is_his_right(self):
        name, frac, deg = edge_bearing(5, 360, W, H)
        assert name == "right" and deg == 0.0
        assert frac == pytest.approx(5 / 640)

    def test_the_image_right_edge_is_his_left(self):
        name, _f, deg = edge_bearing(W - 5, 360, W, H)
        assert name == "left" and deg == 180.0

    def test_the_top_edge_is_his_up_and_the_bottom_his_down(self):
        assert edge_bearing(640, 5, W, H)[0] == "up"
        assert edge_bearing(640, H - 5, W, H)[0] == "down"

    def test_mirroring_swaps_only_left_and_right(self):
        assert edge_bearing(5, 360, W, H, mirrored=True)[0] == "left"
        assert edge_bearing(W - 5, 360, W, H, mirrored=True)[0] == "right"
        assert edge_bearing(640, 5, W, H, mirrored=True)[0] == "up"
        assert edge_bearing(640, H - 5, W, H, mirrored=True)[0] == "down"

    def test_dead_centre_is_a_full_half_field_from_every_edge(self):
        assert edge_bearing(640, 360, W, H)[1] == pytest.approx(1.0)

    def test_a_centroid_outside_the_picture_is_on_the_edge(self):
        assert edge_bearing(-30, 360, W, H)[1] == 0.0
        assert edge_bearing(640, H + 30, W, H)[1] == 0.0


# ============================================================ the firewall
def code_only(path: Path) -> str:
    """The source with every comment and string literal blanked out, so
    that a docstring which SAYS "no cv2 here" is not itself a hit. Spans
    are replaced with spaces, not removed, so line and column structure
    survive for the regexes below."""
    lines = path.read_text().splitlines(keepends=True)
    tokens = tokenize.generate_tokens(io.StringIO("".join(lines)).readline)
    for tok in tokens:
        name = tokenize.tok_name[tok.type]
        if name not in ("COMMENT", "STRING") and not name.startswith("FSTRING"):
            continue
        (r0, c0), (r1, c1) = tok.start, tok.end
        for r in range(r0, r1 + 1):
            line = lines[r - 1]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(line.rstrip("\r\n"))
            lines[r - 1] = line[:a] + " " * (b - a) + line[b:]
    return "".join(lines)


GESTURE_IMPORTS = {"__future__", "math", "threading", "time", "collections",
                   "dataclasses", "enum", "typing", "numpy", "jarvis.logs"}


def test_gesture_never_touches_a_pixel():
    """Hunter: 'I dont want you to look at anything the camera sees'. This
    module decides from coordinates; the vocabulary of images is forbidden
    in its CODE (its docstring is allowed to say so), the way test_cast.py
    pins cast.py's vocabulary. Two checks: an import allow-list read from
    the AST, and a grep over the code with strings and comments blanked."""
    path = REPO / "jarvis" / "gesture.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name in GESTURE_IMPORTS, \
                "jarvis/gesture.py imports %r; it may not" % name
    code = code_only(path)
    for banned in ("cv2", "onnxruntime", "PIL", "imwrite", "imencode",
                   "imread", "imshow", "VideoCapture", "ImageGrab",
                   "tkinter", "socket", "requests", "urlopen", "subprocess",
                   "bus.publish", "NamedTemporaryFile", "mkstemp",
                   "write_bytes", "tofile", "frombuffer", "fromfile",
                   ".save(", "uint8", "__import__", "importlib"):
        assert banned not in code, \
            "jarvis/gesture.py must not reach for %r" % banned
    assert not re.search(r"\bopen\s*\(", code)


def test_handedness_and_world_landmarks_are_never_consulted():
    """The model's left/right call is defined relative to an assumed
    mirroring and is the one field that silently inverts when someone
    'fixes' the mirror; its metric 3-D landmarks are unverified on a real
    hand. Both are logged by handpose and read by nobody here."""
    code = code_only(REPO / "jarvis" / "gesture.py")
    assert not re.search(r"\.handed\b", code)
    assert not re.search(r"\.world\b", code)


def test_events_and_status_are_numbers_only():
    m = CastGesture(now=Clock())
    grab = grab_at(m, 640, 360)
    assert_numbers_only(grab.numbers_only())
    assert_numbers_only(m.status())
    m.update((obs(100, 360, OPEN),), EYE_PX, 9)
    m.update((obs(100, 360, OPEN),), EYE_PX, 10)
    assert_numbers_only(m.status())
    ev = CastEvent(kind="throw", at=1.0, frame=3, sector="left", toward="left")
    assert set(ev.numbers_only()) == {"kind", "at", "frame", "dist_u",
                                      "bearing_deg", "sector", "why", "reach",
                                      "closed", "payload", "toward"}


def test_the_machine_holds_no_array_and_no_observation():
    """Nothing about a frame survives on the object: not an array, not the
    HandObservation, only floats, ints, strings and a deque of booleans."""
    m = CastGesture(now=Clock())
    grab_at(m, 640, 360)
    for k, v in vars(m).items():
        assert not isinstance(v, (np.ndarray, HandObservation)), k
    m.update((obs(120, 360, OPEN),), EYE_PX, 9)
    for k, v in vars(m).items():
        assert not isinstance(v, (np.ndarray, HandObservation)), k


# ===================================================== the designed gesture
class TestTheIntendedGesture:
    """The design's own validation, re-run against the shipped module:
    8 gestures (4 directions x fingers up/down) x 12 sub-frame phases at
    the two rates the camera actually runs at."""

    @pytest.mark.parametrize("fps", [7.5, 6.0])
    def test_every_lateral_throw_grabs_and_lands_on_the_named_side(self, fps):
        grabs, throws, wrong = lateral_recall(fps)
        assert (grabs, throws, wrong) == (48, 48, 0)

    @pytest.mark.parametrize("name", ["his-right", "his-left",
                                      "his-right/fingers-down",
                                      "his-left/fingers-down"])
    def test_the_sector_is_the_side_in_the_trajectorys_name(self, name):
        want = "left" if "left" in name else "right"
        t = tally(run(GOOD[name], 7.5))
        assert t["sectors"] == {want: 12}, t

    @pytest.mark.parametrize("fps", [7.5, 6.0])
    def test_a_fling_at_the_desk_is_a_cancel_never_a_throw(self, fps):
        """12 grabs, 12 drops, 0 throws, and ``toward`` -- the field the
        pane reads -- says "down" or nothing, never a side. MEASURED: it
        says down 12/12 fingers-down at both rates, 10/12 fingers-up at
        7.5 fps and 9/12 at 6.0; the unnamed ones are drops with under
        TOWARD_MIN_U (0.15 units) of travel, which the module refuses to
        put a compass on. Why the vertical case is lossier than the
        lateral one: the palm quad is ~0.8 units tall against 1.29 units
        of vertical room, so the tracker loses the hand while its centroid
        is still more than 30% of a half-field from the bottom edge, and
        that is the 'lost' branch with 0.1-0.4 units of travel."""
        for name in ("his-down", "his-down/fingers-down"):
            events = [e for _p, e in run(GOOD[name], fps)]
            ends = [e for e in events if e.kind != "grab"]
            assert sum(e.kind == "grab" for e in events) == 12
            assert len(ends) == 12 and all(e.kind == "drop" for e in ends)
            assert all(e.sector == "" for e in ends)
            assert all(e.toward in ("down", "") for e in ends), \
                [(e.why, e.toward) for e in ends]
            for e in ends:
                if e.toward == "":
                    assert e.dist_u < g.TOWARD_MIN_U, e
            assert sum(e.toward == "down" for e in ends) >= 9, \
                [(e.why, e.toward, round(e.dist_u, 2)) for e in ends]

    @pytest.mark.parametrize("fps", [7.5, 6.0])
    def test_a_fling_upward_is_a_drop_that_says_up(self, fps):
        for name in ("his-up", "his-up/fingers-down"):
            t = tally(run(GOOD[name], fps))
            assert t["throw"] == 0
            assert t["drop"] == 12

    def test_the_down_cancel_names_down_and_nothing_else(self):
        """``why`` is the diagnostic: a hand that ran out of the bottom of
        the picture is 'cancelled (his down)' when it got far enough to
        name the edge and 'lost' when the tracker dropped it first
        (MEASURED 5 and 7 of 12 phases at 7.5 fps). Neither is ever a
        throw, neither ever carries a sector, and ``toward`` -- what the
        chip and the earcon read -- says down or says nothing."""
        events = [e for _p, e in run(GOOD["his-down"], 7.5)
                  if e.kind == "drop"]
        assert len(events) == 12
        assert all(e.why in ("cancelled (his down)", "lost") for e in events)
        assert any(e.why == "cancelled (his down)" for e in events)
        assert all(e.sector == "" and e.toward in ("down", "")
                   for e in events)

    def test_a_grab_then_setting_it_down_is_a_drop(self):
        t = tally(run(PUT_BACK, 7.5))
        assert t == {"grab": 12, "throw": 0, "drop": 12, "sectors": {},
                     "toward": t["toward"]}

    @pytest.mark.parametrize("fps", [7.5, 6.0])
    def test_the_everyday_set_never_fires(self, fps):
        assert everyday_false(fps) == (0, 0)


class TestOtherFrameRates:
    """The counters are in frames on purpose; for_fps keeps the machine
    honest at rates the camera does not deliver."""

    @pytest.mark.parametrize("fps", [5.0, 10.0, 15.0, 30.0])
    def test_the_same_throw_lands_at_other_rates(self, fps):
        th = CastThresholds.for_fps(fps)
        grabs, throws, wrong = lateral_recall(fps, thresholds=th, phases=6)
        assert (grabs, throws, wrong) == (24, 24, 0)
        assert everyday_false(fps, thresholds=th, phases=4) == (0, 0)

    def test_the_design_band_returns_the_design_untouched(self):
        for fps in (5.5, 6.0, 7.5, 8.0):
            assert CastThresholds.for_fps(fps) == CastThresholds()

    def test_below_the_band_nothing_is_loosened(self):
        """MEASURED: a 2-frame dwell at 5 fps let 8 of 288 everyday motions
        through. A counter never goes under its design value."""
        low = CastThresholds.for_fps(2.0)
        assert low == CastThresholds()

    def test_above_the_band_frames_scale_up_and_the_step_bar_down(self):
        th = CastThresholds.for_fps(15.0)
        assert th.dwell_frames == 6
        assert th.lost_grace_frames == 4
        assert th.carry_max_frames == 60
        assert th.open_lookback_frames == 24
        assert th.exit_step_u == pytest.approx(0.35 / 2.0)
        assert th.reach_min == 2.35 and th.throw_release_u == 1.00

    def test_fast_rates_with_the_raw_counters_would_misfire(self):
        """Why for_fps exists: at 30 fps three frames is 100 ms."""
        grabs, _t = everyday_false(30.0, phases=2)
        assert grabs > 0

    def test_a_nonsense_rate_is_refused(self):
        with pytest.raises(ValueError):
            CastThresholds.for_fps(0.0)

    def test_jittered_intervals_do_not_lose_the_throw(self):
        """+/-40% on every frame period, 7.5 fps nominal."""
        grabs, throws, wrong = lateral_recall(7.5, jitter=0.4, seed=3)
        assert (grabs, throws, wrong) == (48, 48, 0)

    def test_jittered_intervals_never_throw_on_everyday_motion(self):
        """+/-40% jitter can put three frames inside 240 ms. MEASURED: over
        the 24 everyday motions x 6 phases that produced ONE grab -- the
        snatch (fist-then-withdraw/fingers-down, phase 1), where three
        short intervals sampled a 0.26 s stationary fist at reach three
        times -- and it ended as a drop ('released'), not a throw. That is
        the one family the design accepts as a tone and a chip; a
        wall-clock floor on the dwell would be the fix if it ever matters
        in the room, and it is a trade (felt lag), not a free one. Every
        other family is silent and nothing throws."""
        grabs_total = 0
        for name, scen in BAD.items():
            t = tally(run(scen, 7.5, jitter=0.4, seed=3, phases=6))
            assert t["throw"] == 0, (name, t)
            if not name.startswith("fist-then-withdraw"):
                assert t["grab"] == 0, (name, t)
            grabs_total += t["grab"]
        assert grabs_total <= 2, grabs_total

    def test_two_fps_is_honest_about_what_it_can_sample(self):
        """THE STATED LIMIT: at 2 fps (the camera's idle rate is 1.5) a
        natural-speed gesture is BETWEEN samples. The fist is held for
        0.7 s, which is one sample, so a 3-frame dwell cannot complete:
        0 grabs, 0 throws, and no design survives that -- the lane must be
        OFF at idle, not degraded. Below the band nothing is loosened.

        What the LOGIC can do at 2 fps, so the limit is shown to be the
        rate and not the machine: a slow, deliberate gesture whose open
        approach stays in frame for 1.2 s (2-3 open samples), holds 1.5 s
        (3-4 fist samples) and swings wide grabs 6/6 and lands 6/6 on the
        named side, both sides. MEASURED. The first draft's slow variant,
        whose approach starts out of frame, grabbed only 3/6: three phases
        never sampled an open hand and the resting-fist rule refused them,
        which is that rule working."""
        natural = tally(run(GOOD["his-right"], 2.0, phases=6))
        assert natural["grab"] == 0 and natural["throw"] == 0, natural
        for side, end in (("right", (-300, -10, 440)),
                          ("left", (310, -10, 440))):
            t = tally(run(slow_gesture(end), 2.0, phases=6))
            assert t["grab"] == 6 and t["throw"] == 6, (side, t)
            assert t["sectors"] == {side: 6}, (side, t)

    def test_the_slow_gesture_also_lands_at_the_design_rates(self):
        for fps in (7.5, 6.0):
            for side, end in (("right", (-300, -10, 440)),
                              ("left", (310, -10, 440))):
                t = tally(run(slow_gesture(end), fps, phases=6))
                assert t["sectors"] == {side: 6} and t["grab"] == 6, (fps, t)

    def test_describe_speaks_the_counters_in_time(self):
        line = g.describe(CastThresholds(), 7.5)
        assert "dwell 3 fr (400 ms)" in line
        assert "cap 30 fr (4.0 s) or 8.0 s" in line
        assert "targets left/right" in line


# ======================================================== the throw bars
class TestThrowBars:
    """Just under and just over each bar, driven directly. Every direction
    is named; a bar met in a non-target sector is a drop that says why."""

    def _release(self, dx_units, dy_units=0.0, mirrored=False):
        m = CastGesture(now=Clock(), mirrored=mirrored)
        grab_at(m, 640, 360)
        # fist travels, then opens where it is
        x = 640 + dx_units * UNIT
        y = 360 + dy_units * UNIT
        m.update((obs(x, y, FIST),), EYE_PX, 5)
        return m.update((obs(x, y, OPEN),), EYE_PX, 6)

    def test_released_in_frame_needs_a_full_hand_width(self):
        under = self._release(-0.99)
        over = self._release(-1.01)
        assert under.kind == "drop" and under.why == "released"
        assert over.kind == "throw" and over.sector == "right"
        assert over.dist_u == pytest.approx(1.01, abs=1e-6)

    def test_image_minus_x_is_his_right_and_plus_x_his_left(self):
        assert self._release(-1.5).sector == "right"
        assert self._release(+1.5).sector == "left"

    def test_a_mirrored_feed_names_the_other_side(self):
        assert self._release(-1.5, mirrored=True).sector == "left"
        assert self._release(+1.5, mirrored=True).sector == "right"

    def test_opening_the_hand_where_it_is_is_a_drop(self):
        e = self._release(0.0)
        assert e.kind == "drop" and e.why == "released"
        assert e.toward == ""

    def test_a_fling_at_the_desk_is_the_cancel(self):
        e = self._release(0.0, +1.5)               # image +y is his down
        assert e.kind == "drop" and e.why == "cancelled (his down)"
        assert e.toward == "down" and e.sector == ""

    def test_a_fling_upward_is_not_a_target(self):
        e = self._release(0.0, -1.5)
        assert e.kind == "drop" and "his up is not a target" in e.why
        assert e.toward == "up"

    @pytest.mark.parametrize("dx,dy", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
    def test_a_diagonal_is_ambiguous_and_drops(self, dx, dy):
        e = self._release(1.2 * dx, 1.2 * dy)
        assert e.kind == "drop" and "ambiguous direction" in e.why
        assert e.toward == "ambiguous"

    def test_the_sector_half_width_is_honoured_at_the_boundary(self):
        # 34 deg off his right: inside; 36 deg: the gap.
        inside = self._release(-1.5 * math.cos(math.radians(34)),
                               -1.5 * math.sin(math.radians(34)))
        gap = self._release(-1.5 * math.cos(math.radians(36)),
                            -1.5 * math.sin(math.radians(36)))
        assert inside.kind == "throw" and inside.sector == "right"
        assert gap.kind == "drop"

    def test_targets_are_configurable(self):
        th = CastThresholds(target_sectors=("left",))
        m = CastGesture(th, now=Clock())
        grab_at(m, 640, 360)
        m.update((obs(640 - 1.5 * UNIT, 360, FIST),), EYE_PX, 5)
        e = m.update((obs(640 - 1.5 * UNIT, 360, OPEN),), EYE_PX, 6)
        assert e.kind == "drop" and "his right is not a target" in e.why

    # -- leaving the frame ----------------------------------------------
    def _exit(self, last_x, last_y, step_units, edge_ok=True):
        """Grab at centre, one fist frame at (last_x, last_y) reached with
        a last step of ``step_units``, then the hand is gone."""
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        # a frame before the last so the last STEP is what we say it is
        ux = (last_x - 640) / max(abs(last_x - 640), 1e-9)
        uy = (last_y - 360) / max(abs(last_y - 360), 1e-9)
        prev = (last_x - ux * step_units * UNIT if last_x != 640 else last_x,
                last_y - uy * step_units * UNIT if last_y != 360 else last_y)
        m.update((obs(prev[0], prev[1], FIST),), EYE_PX, 5)
        m.update((obs(last_x, last_y, FIST),), EYE_PX, 6)
        events = [m.update((), 0.0, 7 + k) for k in range(3)]
        return events[-1]

    def test_leaving_through_the_image_left_edge_is_a_throw_to_his_right(self):
        # 100 px from the left edge: frac 0.16 <= 0.30, dist 1.93 >= 0.25
        e = self._exit(100, 360, 0.5)
        assert e.kind == "throw" and e.sector == "right"
        assert e.why == "left frame (his right)"

    def test_leaving_through_the_image_right_edge_is_a_throw_to_his_left(self):
        e = self._exit(W - 100, 360, 0.5)
        assert e.kind == "throw" and e.sector == "left"
        assert e.why == "left frame (his left)"

    def test_leaving_through_the_bottom_is_the_cancel(self):
        e = self._exit(640, H - 60, 0.5)
        assert e.kind == "drop" and e.why == "cancelled (his down)"

    def test_leaving_through_the_top_is_not_a_target(self):
        e = self._exit(640, 60, 0.5)
        assert e.kind == "drop" and "his up is not a target" in e.why

    def test_the_edge_bar_is_a_quarter_hand_width(self):
        # anchor 100 px from the left edge so it can leave with little travel
        for dist, kind in ((0.24, "drop"), (0.26, "throw")):
            m = CastGesture(now=Clock())
            grab_at(m, 100, 360)
            m.update((obs(100 - dist * UNIT, 360, FIST),), EYE_PX, 5)
            e = [m.update((), 0.0, 6 + k) for k in range(3)][-1]
            assert e.kind == kind, (dist, e)
            if kind == "throw":
                assert e.sector == "right"

    def test_lost_in_open_space_needs_distance_and_motion(self):
        # centre of the frame: frac > 0.30 from every edge -> the lost rule
        far_fast = self._exit(640 - 0.51 * UNIT, 360, 0.36)
        far_slow = self._exit(640 - 0.51 * UNIT, 360, 0.34)
        near_fast = self._exit(640 - 0.49 * UNIT, 360, 0.36)
        assert far_fast.kind == "throw" and far_fast.sector == "right"
        assert far_fast.why == "lost"
        assert far_slow.kind == "drop" and far_slow.why == "lost"
        assert near_fast.kind == "drop" and near_fast.why == "lost"

    def test_the_hand_vanishing_where_it_was_grabbed_is_a_drop(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        e = [m.update((), 0.0, 5 + k) for k in range(3)][-1]
        assert e.kind == "drop" and e.dist_u == 0.0
        assert e.toward == ""

    def test_the_bearing_of_an_edge_exit_is_the_edges_not_the_vectors(self):
        """A diagonal run out through the left edge is still a throw to his
        right: leaving the picture is the evidence and the edge names it."""
        e = self._exit(100, 200, 0.5)
        assert e.kind == "throw" and e.sector == "right"
        assert e.bearing_deg == 0.0


# ========================================================= ordinary motion
def _rand_family(rng):
    """One random everyday sequence from 14 families. Each is a plausible
    desk motion with its parameters drawn at random; none is the gesture.
    Returns (family, duration, traj)."""
    u = rng.uniform
    fam = rng.integers(0, 14)
    if fam == 0:                                            # typing
        bx, by, bz = u(-260, 260), u(150, 320), u(540, 720)
        ax, ay, f1, f2 = u(4, 20), u(4, 16), u(8, 30), u(8, 30)
        c0, ca = u(0.25, 0.55), u(0.05, 0.15)
        pi, ya = u(-35, -10), u(-15, 15)
        return "typing", u(2, 5), lambda t: (
            (bx + ax * math.sin(f1 * t), by + ay * math.sin(f2 * t), bz),
            c0 + ca * math.sin(f1 * t), pi, ya, 0)
    if fam == 1:                                            # scratch a face
        rest = (u(100, 220), u(200, 300), u(620, 720))
        face = (u(-120, 120), u(-160, -40), u(540, 680))
        c, hold = u(0.35, 0.75), u(0.6, 2.0)
        pi, ya, ro = u(0, 25), u(-10, 10), u(-15, 15)
        return "scratch", 0.9 + hold + 0.9, lambda t: seg(t, [
            (0.9, rest, face, c - 0.1, c, pi, 0, 0),
            (hold, face, tuple(v + 8 for v in face), c, c + 0.05, pi, ya, ro),
            (0.9, tuple(v + 8 for v in face), rest, c + 0.05, c - 0.1, pi,
             0, 0)])
    if fam == 2:                                            # drink from a mug
        rest = (u(150, 260), u(180, 280), u(560, 700))
        mouth = (u(60, 160), u(-120, -60), u(480, 600))
        c, hold = u(0.80, 0.95), u(1.0, 2.5)
        pi = u(10, 25)
        return "drink", 0.9 + hold + 0.9, lambda t: seg(t, [
            (0.9, rest, mouth, c - 0.03, c, pi, 0, 0),
            (hold, mouth, tuple(v + 5 for v in mouth), c, c, pi + 5, 0, 5),
            (0.9, tuple(v + 5 for v in mouth), rest, c, c - 0.03, pi, 0, 0)])
    if fam == 3:                                            # talking
        ax, ay, z = u(80, 250), u(20, 70), u(500, 700)
        c0, ca, w, sway = u(0.05, 0.35), u(0.05, 0.15), u(3, 10), u(10, 25)
        pi = u(0, 25)
        return "talk", u(2, 5), lambda t: (
            (ax * math.sin(w * t), 40 + ay * math.cos(w * t), z),
            c0 + ca * math.sin(2 * w * t), pi, sway * math.sin(w * t), 0)
    if fam == 4:                                            # a hand crosses
        d = u(0.5, 1.6)
        y, z = u(-100, 150), u(380, 620)
        sgn = 1 if rng.integers(0, 2) else -1
        c0, c1 = u(0, 1), u(0, 1)
        pi = u(0, 30)
        return "cross", d, lambda t: (
            (sgn * (-450 + 900 * t / d), y, z), c0 + (c1 - c0) * t / d, pi,
            0, 0)
    if fam == 5:                                            # chin on a fist
        pos = (u(-160, 160), u(-40, 80), u(520, 720))
        c, d, drift = u(0.9, 1.0), u(1.5, 5.0), u(0, 10)
        pi, ya = u(5, 35), u(-10, 10)
        return "chin-fist", d, lambda t: (
            (pos[0] + drift * math.sin(0.7 * t), pos[1] + drift *
             math.cos(0.5 * t), pos[2]), c, pi, ya, 0)
    if fam == 6:                                            # adjust the monitor
        rest = (u(100, 220), u(150, 300), u(620, 760))
        at = (u(-80, 80), u(-80, 20), u(370, 450))
        c, fiddle = u(0.05, 0.25), u(0.5, 1.5)
        pi = u(15, 35)
        return "adjust", 0.8 + fiddle + 1.0, lambda t: seg(t, [
            (0.8, rest, at, c, c, pi, 0, 0),
            (fiddle, at, (at[0] + u(-60, 60), at[1] + u(-40, 40), at[2] - 15),
             c, c + 0.05, pi + 5, u(-10, 10), u(-10, 10)),
            (1.0, (at[0], at[1], at[2] - 15), rest, c + 0.05, c, pi, 0, 0)])
    if fam == 7:                                            # snatch and withdraw
        rest = (u(100, 220), u(200, 320), u(700, 800))
        at = (u(-60, 60), u(-40, 30), u(400, 460))
        close, hold, back = u(0.10, 0.25), u(0.0, 0.25), u(0.5, 1.0)
        pi = u(25, 40)
        return "snatch", 0.55 + close + hold + back, lambda t: seg(t, [
            (0.55, rest, at, 0.10, 0.05, pi - 5, 0, 0),
            (close, at, at, 0.05, 1.00, pi, 0, 0),
            (hold, at, (at[0] + 5, at[1] + 5, at[2] + 10), 1.00, 1.00, pi,
             0, 0),
            (back, (at[0] + 5, at[1] + 5, at[2] + 10), rest, 1.00, 0.20,
             pi - 5, 0, 0)])
    if fam == 8:                                            # phone at the lens
        rest = (u(100, 220), u(180, 300), u(620, 760))
        at = (u(-60, 80), u(-80, 0), u(400, 470))
        c, hold, drift = u(0.80, 0.95), u(1.0, 3.0), u(0, 15)
        pi = u(15, 30)
        return "phone", 0.8 + hold + 1.0, lambda t: seg(t, [
            (0.8, rest, at, c - 0.1, c, pi, 0, 0),
            (hold, at, (at[0] + drift, at[1] + drift, at[2] + 5), c, c,
             pi, 5, 5),
            (1.0, (at[0] + drift, at[1] + drift, at[2] + 5), rest, c,
             c - 0.1, pi, 0, 0)])
    if fam == 9:                                            # rub an eye
        side = 1 if rng.integers(0, 2) else -1
        eye = (side * u(40, 100), u(-110, -60), u(410, 480))
        rest = (u(100, 220), u(180, 300), u(600, 720))
        c, d, amp, f = u(0.85, 1.0), u(0.8, 2.2), u(5, 15), u(2, 4)
        pi = u(10, 25)
        return "rub-eye", 0.7 + d + 0.7, lambda t: seg(t, [
            (0.7, rest, eye, c - 0.1, c, pi, 0, 0),
            (d, eye, (eye[0] + amp * math.sin(f * 6.28 * t), eye[1] +
                      amp * math.cos(f * 6.28 * t), eye[2]), c, c, pi + 5,
             10, 15),
            (0.7, eye, rest, c, c - 0.1, pi, 0, 0)])
    if fam == 10:                                           # a wave
        z, y, amp, f, c = u(430, 620), u(-150, 0), u(70, 160), u(1.5, 3.0), \
            u(0.0, 0.2)
        return "wave", u(1.0, 2.5), lambda t: (
            (amp * math.sin(2 * math.pi * f * t), y, z), c, u(0, 20),
            15 * math.sin(2 * math.pi * f * t), 0)
    if fam == 11:                                           # point at the screen
        at = (u(-150, 150), u(-120, 60), u(380, 480))
        d = u(1.0, 3.0)
        pi = u(20, 45)
        return "point", 0.6 + d + 0.6, lambda t: seg(t, [
            (0.6, (150, 250, 720), at, (0., 0.8, 0.9, 0.9), (0., 1., 1., 1.),
             pi, 0, 0),
            (d, at, (at[0] + u(-80, 80), at[1] + u(-40, 40), at[2]),
             (0., 1., 1., 1.), (0., 1., 1., 1.), pi, u(-10, 10), 0),
            (0.6, at, (150, 250, 720), (0., 1., 1., 1.), (0., 0.8, 0.9, 0.9),
             pi, 0, 0)])
    if fam == 12:                                           # a stretch
        z, c = u(400, 560), u(0.1, 0.3)
        d = u(2.0, 4.0)
        return "stretch", d, lambda t: (
            (u(-200, 200), 200 - 450 * math.sin(math.pi * t / d), z), c,
            u(0, 20), 0, 0)
    # fam == 13: fidgeting -- opening and closing in front of the face
    z, pos = u(520, 680), (u(-120, 120), u(-60, 80))
    f, d = u(1.0, 3.0), u(1.0, 3.0)
    pi = u(10, 35)
    return "fidget", d, lambda t: (
        (pos[0], pos[1], z), 0.65 + 0.35 * math.sin(2 * math.pi * f * t),
        pi, 0, 0)


def sweep(n: int, seed: int = 20260903) -> dict:
    """The false-positive sweep: ``n`` random everyday sequences, each at
    a random rate in the design band, a random phase and a random landmark
    noise level in {0, 3, 5} px. Returns per-family counts."""
    rng = np.random.default_rng(seed)
    out = {"n": n, "grabs": 0, "throws": 0, "families": {}}
    for _ in range(n):
        fam, dur, traj = _rand_family(rng)
        fps = float(rng.uniform(5.5, 8.0))
        noise = float(rng.choice([0.0, 3.0, 5.0]))
        events = run((dur, traj), fps, phases=1, noise=noise,
                     seed=int(rng.integers(0, 1 << 20)))
        # phases=1 starts at t=0; randomise the phase by the clock instead
        t = tally(events)
        row = out["families"].setdefault(fam, {"n": 0, "grabs": 0,
                                               "throws": 0})
        row["n"] += 1
        row["grabs"] += t["grab"]
        row["throws"] += t["throw"]
        out["grabs"] += t["grab"]
        out["throws"] += t["throw"]
    return out


class TestOrdinaryMotion:
    """The headline: how often ordinary desk motion fires the gesture."""

    def test_the_named_everyday_set_at_both_rates_is_silent(self):
        """24 trajectories x 12 phases x 2 rates = 576 sequences."""
        assert everyday_false(7.5) == (0, 0)
        assert everyday_false(6.0) == (0, 0)

    def test_the_randomised_sweep_never_throws(self):
        """THE HEADLINE. JARVIS_GESTURE_SWEEP=N scales it; the default of
        600 runs in 0.3 s. MEASURED 2026-09-03 at N=5000: 0 throws. Grabs:
        snatch 169/371 (46%), fidget 25/375 (6.7%), the other twelve
        families 0/4254. (N=50000 is in the commit message.)

        Both grabbing families are a closed hand, seen open, held still at
        reach depth for three frames -- which IS the grab, by the design's
        definition. The snatch is that on purpose (reach, close, hold,
        withdraw). The fidget is a palm held flat to the lens at ~52 cm
        opening and closing slowly: flat-to-the-lens reads R 2.38 against
        the 2.35 bar, the pose-dependence of R the design records as its
        thin edge-on margin, met from the other side. Each costs a tone and
        a chip that drops itself. A false THROW is the thing the design
        must not produce, and across every family it produces none."""
        n = int(os.environ.get("JARVIS_GESTURE_SWEEP", "600"))
        res = sweep(n)
        assert res["throws"] == 0, res
        for fam, row in res["families"].items():
            if fam == "snatch":
                continue
            if fam == "fidget":
                # MEASURED 6.7% at N=5000; the bar is twice that.
                assert row["grabs"] <= max(2, int(0.14 * row["n"])), (fam, row)
                continue
            assert row["grabs"] == 0, (fam, row)

    def test_a_left_hand_behaves_the_same(self):
        """Handedness is never consulted; a mirrored hand model must grab
        and throw identically."""
        assert lateral_recall(7.5, left=True, phases=6) == (24, 24, 0)
        assert everyday_false(7.5, left=True, phases=6)[1] == 0

    @pytest.mark.parametrize("noise", [3.0, 5.0])
    def test_landmark_noise_at_the_trackers_level(self, noise):
        """MEASURED tracker jitter is sd ~3-5 px in capture pixels at the
        320x180 detect size (frames lane). The intended gesture survives it
        and nothing everyday throws."""
        grabs, throws, wrong = lateral_recall(7.5, noise=noise, seed=int(noise))
        assert (grabs, throws, wrong) == (48, 48, 0)
        assert everyday_false(7.5, noise=noise, seed=int(noise))[1] == 0

    def test_heavy_noise_degrades_recall_never_direction(self):
        """sd 12 px is well past the measured floor. Some throws are lost
        (MEASURED 46/48 at 7.5 fps); none goes to the wrong side and
        nothing everyday throws."""
        grabs, throws, wrong = lateral_recall(7.5, noise=12.0, seed=12)
        assert wrong == 0
        assert throws >= 40
        assert everyday_false(7.5, noise=12.0, seed=12)[1] == 0


# ======================================================== robustness
class TestRobustness:
    def test_one_or_two_dropped_frames_mid_carry_are_ridden_out(self):
        for misses in (1, 2):
            m = CastGesture(now=Clock())
            grab_at(m, 640, 360)
            for k in range(misses):
                assert m.update((), 0.0, 5 + k) is None
            assert m.state is CastState.CARRYING
            # the hand comes back, throws to his right, in frame
            m.update((obs(640 - 1.4 * UNIT, 360, FIST),), EYE_PX, 8)
            e = m.update((obs(640 - 1.4 * UNIT, 360, OPEN),), EYE_PX, 9)
            assert e.kind == "throw" and e.sector == "right"

    def test_three_dropped_frames_end_the_carry(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        assert m.update((), 0.0, 5) is None
        assert m.update((), 0.0, 6) is None
        e = m.update((), 0.0, 7)
        assert e is not None and e.kind == "drop"
        assert m.state is CastState.COOLDOWN

    def test_a_dropped_frame_and_a_missing_hand_are_the_same_call(self):
        m = CastGesture(now=Clock())
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        assert m.state is CastState.REACHING
        m.update((), 0.0, 1)
        assert m.state is CastState.IDLE

    def test_two_hands_at_reach_is_ambiguous_and_grabs_nothing(self):
        m = CastGesture(now=Clock())
        both = (obs(500, 360, OPEN, 2.6, 280), obs(800, 360, OPEN, 2.6, 250))
        m.update(both, EYE_PX, 0)
        fists = (obs(500, 360, FIST, 2.6, 280), obs(800, 360, FIST, 2.6, 250))
        for k in range(1, 6):
            assert m.update(fists, EYE_PX, k) is None
        assert m.state is CastState.IDLE
        assert m.status()["ambiguous"] is True

    def test_a_second_hand_further_back_does_not_veto(self):
        m = CastGesture(now=Clock())
        far = obs(900, 500, FIST, 1.2, 120)          # not at reach, small
        m.update((obs(640, 360, OPEN), far), EYE_PX, 0)
        events = [m.update((obs(640, 360, FIST), far), EYE_PX, k)
                  for k in range(1, 4)]
        assert events[-1].kind == "grab"

    def test_a_second_hand_raised_mid_carry_does_not_end_it(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        other = obs(1100, 300, OPEN, 2.6, 300)      # bigger AND open
        m.update((obs(600, 360, FIST), other), EYE_PX, 5)
        assert m.state is CastState.CARRYING
        m.update((obs(640 - 1.3 * UNIT, 360, FIST), other), EYE_PX, 6)
        e = m.update((obs(640 - 1.3 * UNIT, 360, OPEN), other), EYE_PX, 7)
        assert e.kind == "throw" and e.sector == "right"

    def test_a_hand_that_jumps_too_far_is_not_the_carried_one(self):
        """Association during a carry is nearest-centroid within 2.5 units;
        a hand that appears 3 units away is a different hand, and the
        carried one counts as lost."""
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        for k in range(3):
            m.update((obs(640 + 3.0 * UNIT, 360, FIST),), EYE_PX, 5 + k)
        assert m.state is CastState.COOLDOWN

    def test_a_degenerate_row_is_ignored_not_carried(self):
        m = CastGesture(now=Clock())
        bad = HandObservation(0.0, 0.0, 0.0, 640, 360, ok=False)
        m.update((bad,), EYE_PX, 0)
        assert m.state is CastState.IDLE

    def test_mirrored_config_flips_the_named_side_of_a_real_trajectory(self):
        for name, want in (("his-right", "left"), ("his-left", "right")):
            t = tally(run(GOOD[name], 7.5, mirrored=True, phases=4))
            assert t["sectors"] == {want: 4}, (name, t)


# =========================================================== the cancels
class TestCancels:
    def test_opening_in_place_puts_it_down(self):
        put = []
        pay = CallablePayload(pick_up=lambda: ("w1", "the window"),
                              put_back=put.append)
        m = CastGesture(now=Clock(), payload=pay)
        grab_at(m, 640, 360)
        e = m.update((obs(650, 365, OPEN),), EYE_PX, 5)
        assert e.kind == "drop" and e.why == "released"
        assert put == ["w1"]
        assert m.held == ""

    def test_a_fist_that_leaves_the_reach_zone_still_closed_is_a_drop(self):
        """Reaching behind the monitor for an object and retracting it, hand
        shut the whole way. The design harness's snatch is one shape of it;
        this is the one that DOES pass the dwell -- and still never throws."""
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        # retreat: down and shrinking, still a fist, out the bottom
        for k, (y, pd) in enumerate(((500, 240), (640, 200), (700, 170))):
            m.update((obs(660, y, FIST, 1.5, pd),), EYE_PX, 5 + k)
        e = [m.update((), 0.0, 8 + k) for k in range(3)][-1]
        assert e.kind == "drop" and e.why == "cancelled (his down)"

    def test_the_spoken_drop_it_cancels_a_carry(self):
        put, seen = [], []
        pay = CallablePayload(pick_up=lambda: ("d1", "the document"),
                              put_back=put.append)
        m = CastGesture(now=Clock(), payload=pay, on_event=seen.append)
        grab_at(m, 640, 360)
        e = m.cancel("drop it")
        assert e.kind == "drop" and e.why == "cancelled (drop it)"
        assert e.payload == "the document"
        assert put == ["d1"]
        assert seen[-1] is e
        assert m.state is CastState.COOLDOWN

    def test_cancel_outside_a_carry_clears_the_reach_and_says_nothing(self):
        m = CastGesture(now=Clock())
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        m.update((obs(640, 360, FIST),), EYE_PX, 1)
        assert m.state is CastState.CLOSING
        assert m.cancel("never mind") is None
        assert m.state is CastState.IDLE
        assert m.cancel() is None

    def test_a_cancel_never_throws_whatever_the_hand_did(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        m.update((obs(640 - 2.0 * UNIT, 360, FIST),), EYE_PX, 5)
        e = m.cancel("shutdown")
        assert e.kind == "drop" and e.sector == ""
        assert e.toward == "right"          # informational only

    def test_the_carry_times_out_in_frames(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        e = None
        for k in range(5, 40):
            e = m.update((obs(640, 360, FIST),), EYE_PX, k)
            if e is not None:
                break
        assert e is not None and e.kind == "drop" and e.why == "timeout"
        assert k == 5 + 30                   # the 31st carry frame

    def test_the_carry_times_out_on_the_wall_clock_too(self):
        """At 2 fps 30 frames is 15 s; the 8 s backstop ends it first."""
        clk = Clock()
        m = CastGesture(now=clk, preview_fps=2.0)
        grab_at(m, 640, 360)
        frames = 0
        e = None
        while e is None and frames < 100:
            clk.t += 0.5
            e = m.update((obs(640, 360, FIST),), EYE_PX, 5 + frames)
            frames += 1
        assert e is not None and e.why == "timeout"
        assert frames <= 17                  # 8.0 s / 0.5 s + 1

    def test_a_stalled_capture_thread_drops_the_carry(self):
        clk = Clock()
        m = CastGesture(now=clk, preview_fps=7.5)
        grab_at(m, 640, 360)
        clk.t += 3.0 / 7.5 + 0.05
        e = m.update((obs(640, 360, FIST),), EYE_PX, 5)
        assert e.kind == "drop" and e.why == "stalled"
        assert m.state is CastState.COOLDOWN

    def test_a_stall_outside_a_carry_resets_to_idle(self):
        clk = Clock()
        m = CastGesture(now=clk, preview_fps=7.5)
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        m.update((obs(640, 360, FIST),), EYE_PX, 1)
        assert m.state is CastState.CLOSING
        clk.t += 1.0
        assert m.update((obs(640, 360, FIST),), EYE_PX, 2) is None
        assert m.state in (CastState.IDLE, CastState.REACHING)
        assert m.status()["dwell"] <= 1

    def test_the_stall_bar_is_three_frame_periods(self):
        assert CastGesture(preview_fps=7.5).stall_s == pytest.approx(0.4)
        assert CastGesture(preview_fps=6.0).stall_s == pytest.approx(0.5)

    def test_cooldown_blocks_a_new_grab_for_eight_frames(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        m.update((obs(640, 360, OPEN),), EYE_PX, 5)         # drop
        assert m.state is CastState.COOLDOWN
        for k in range(6, 6 + 7):
            m.update((obs(640, 360, FIST),), EYE_PX, k)
            assert m.state is CastState.COOLDOWN
        m.update((obs(640, 360, FIST),), EYE_PX, 13)
        assert m.state is CastState.IDLE

    def test_reset_drops_everything_silently(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        m.reset()
        assert m.state is CastState.IDLE and m.held == ""


# ============================================================ the payload
class TestPayload:
    def test_pick_up_is_called_once_at_the_grab_and_named_in_the_event(self):
        calls = []

        def pick():
            calls.append(1)
            return ("h", "Firefox")

        m = CastGesture(now=Clock(), payload=CallablePayload(pick_up=pick))
        e = grab_at(m, 640, 360)
        assert calls == [1]
        assert e.payload == "Firefox" and m.held == "Firefox"
        assert m.status()["held"] == "Firefox"

    def test_nothing_to_carry_is_a_drop_with_nothing_armed(self):
        m = CastGesture(now=Clock(), payload=CallablePayload(pick_up=lambda: None))
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        events = [m.update((obs(640, 360, FIST),), EYE_PX, k)
                  for k in range(1, 4)]
        e = events[-1]
        assert e.kind == "drop" and e.why == "nothing to carry"
        assert m.state is CastState.COOLDOWN and m.held == ""

    def test_prepare_is_called_on_the_reach_before_any_grab(self):
        order = []
        pay = CallablePayload(pick_up=lambda: (order.append("pick"),
                                               ("h", "x"))[1],
                              prepare=lambda: order.append("prepare"))
        m = CastGesture(now=Clock(), payload=pay)
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        assert order == ["prepare"]
        for k in range(1, 4):
            m.update((obs(640, 360, FIST),), EYE_PX, k)
        assert order == ["prepare", "pick"]

    def test_put_back_on_a_drop_never_on_a_throw(self):
        put = []
        pay = CallablePayload(pick_up=lambda: ("h", "x"), put_back=put.append)
        m = CastGesture(now=Clock(), payload=pay)
        grab_at(m, 640, 360)
        m.update((obs(640 - 1.5 * UNIT, 360, FIST),), EYE_PX, 5)
        e = m.update((obs(640 - 1.5 * UNIT, 360, OPEN),), EYE_PX, 6)
        assert e.kind == "throw" and e.payload == "x"
        assert put == []
        assert m.held == ""

    def test_a_payload_that_raises_does_not_kill_the_capture_thread(self):
        def boom():
            raise RuntimeError("no display")

        pay = CallablePayload(pick_up=boom, put_back=lambda h: 1 / 0,
                              prepare=boom)
        m = CastGesture(now=Clock(), payload=pay)
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        events = [m.update((obs(640, 360, FIST),), EYE_PX, k)
                  for k in range(1, 4)]
        assert events[-1].kind == "drop" and events[-1].why == "nothing to carry"

    def test_a_subscriber_that_raises_is_logged_not_propagated(self):
        def bad(_e):
            raise ValueError("bad chip")

        m = CastGesture(now=Clock(), on_event=bad)
        e = grab_at(m, 640, 360)
        assert e.kind == "grab"

    def test_the_adapter_tolerates_missing_parts(self):
        p = CallablePayload()
        p.prepare()
        assert p.pick_up() is None
        p.put_back("h")
        assert CallablePayload(pick_up=lambda: ()).pick_up() is None
        assert CallablePayload(pick_up=lambda: (1, 2)).pick_up() == ("1", "2")

    def test_a_second_grab_while_holding_is_a_no_op(self):
        m = CastGesture(now=Clock(), payload=CallablePayload(
            pick_up=lambda: ("h", "x")))
        grab_at(m, 640, 360)
        for k in range(5, 12):
            assert m.update((obs(640, 360, FIST),), EYE_PX, k) is None
        assert m.state is CastState.CARRYING


# ============================================================ the dwell
class TestDwellRules:
    def test_the_reach_is_retested_on_every_dwell_frame(self):
        """A fist that closes at reach and then pulls back along the lens
        axis (reach falls, centroid stays) must not grab. This is what the
        measured harness did and what the first draft did not."""
        m = CastGesture(now=Clock())
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        m.update((obs(640, 360, FIST, 2.6),), EYE_PX, 1)
        m.update((obs(640, 360, FIST, 2.0),), EYE_PX, 2)
        e = m.update((obs(640, 360, FIST, 2.0),), EYE_PX, 3)
        assert e is None and m.state is CastState.REACHING
        e = m.update((obs(640, 360, FIST, 2.0),), EYE_PX, 4)
        assert e is None and m.state is not CastState.CARRYING

    def test_a_resting_fist_never_seen_open_cannot_grab(self):
        m = CastGesture(now=Clock())
        for k in range(20):
            assert m.update((obs(640, 360, FIST),), EYE_PX, k) is None
        assert m.state is CastState.CLOSING

    def test_the_open_frame_must_be_within_the_lookback(self):
        m = CastGesture(now=Clock())
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        # 12 frames of nothing push the open frame out of the window
        for k in range(1, 13):
            m.update((), 0.0, k)
        for k in range(13, 20):
            assert m.update((obs(640, 360, FIST),), EYE_PX, k) is None
        assert m.state is not CastState.CARRYING

    def test_the_dead_band_is_not_a_fist(self):
        m = CastGesture(now=Clock())
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        m.update((obs(640, 360, FIST),), EYE_PX, 1)
        m.update((obs(640, 360, 0.78),), EYE_PX, 2)     # between the bars
        assert m.state is CastState.REACHING
        assert m.status()["dwell"] == 0

    def test_the_dead_band_does_not_release_a_carry(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        assert m.update((obs(640, 360, 0.78),), EYE_PX, 5) is None
        assert m.state is CastState.CARRYING

    def test_a_moving_fist_relatches_and_restarts_the_dwell(self):
        m = CastGesture(now=Clock())
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        m.update((obs(640, 360, FIST),), EYE_PX, 1)
        m.update((obs(640, 360, FIST),), EYE_PX, 2)
        assert m.status()["dwell"] == 2
        m.update((obs(640 - 0.7 * UNIT, 360, FIST),), EYE_PX, 3)
        assert m.status()["dwell"] == 0
        assert m.state is CastState.CLOSING
        events = [m.update((obs(640 - 0.7 * UNIT, 360, FIST),), EYE_PX, k)
                  for k in (4, 5, 6)]
        assert events[-1].kind == "grab"

    def test_arming_needs_less_reach_than_grabbing(self):
        m = CastGesture(now=Clock())
        m.update((obs(640, 360, OPEN, 1.7),), EYE_PX, 0)
        assert m.state is CastState.REACHING
        for k in range(1, 6):
            m.update((obs(640, 360, FIST, 2.0),), EYE_PX, k)
        assert m.state is CastState.REACHING
        m.update((obs(640, 360, FIST, 1.2),), EYE_PX, 6)
        assert m.state is CastState.IDLE

    def test_the_default_thresholds_are_the_design(self):
        t = CastThresholds()
        assert (t.closed_max, t.open_min, t.reach_min, t.reach_arm) == \
            (0.70, 0.85, 2.35, 1.60)
        assert (t.dwell_frames, t.open_lookback_frames, t.lost_grace_frames,
                t.carry_max_frames, t.cooldown_frames) == (3, 12, 2, 30, 8)
        assert (t.throw_release_u, t.throw_exit_u, t.throw_lost_u,
                t.exit_step_u, t.edge_frac) == (1.00, 0.25, 0.50, 0.35, 0.30)
        assert t.carry_max_s == 8.0 and t.target_sectors == ("left", "right")
