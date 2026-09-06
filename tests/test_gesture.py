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


def withdraw(end, hold=0.50, pull=0.40, release=0.35, tail=0.5, pitch=35.):
    """Reach in, close, hold -- then PULL THE FIST BACK to ``end`` and open
    it there, shut the whole way. The second cancel: reaching past the
    monitor for an object and retracting it with the thing in your hand.
    ``end`` is named for the side it withdraws toward."""
    return (0.55 + 0.20 + hold + pull + release + tail, lambda t: seg(t, [
        (0.55, (140, 300, 780), (10, 10, 430), 0.10, 0.06, pitch - 5, 0, 0),
        (0.20, (10, 10, 430), (10, 10, 425), 0.06, 1.00, pitch, 0, 0),
        (hold, (10, 10, 425), (15, 5, 425), 1.00, 1.00, pitch, 0, 0),
        (pull, (15, 5, 425), end, 1.00, 1.00, pitch, 0, 0),
        (release, end, end, 1.00, 0.06, pitch, 0, 0),
        (tail, end, end, 0.06, 0.06, pitch, 0, 0)]))


# Shoulder height, 600 mm back from the lens: the retraction the adversarial
# pass measured throwing 68 of 68 before ``reach_exit`` existed. Image +x is
# HIS LEFT, so these are named for the side the hand travels toward.
WITHDRAW = {"his-left": withdraw((200., 50., 600.)),
            "his-right": withdraw((-200., 50., 600.))}


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
            # ROUND 3: 9 of 12 became 8 of 12. Every distance bar was
            # rescaled by 0.86 into the corrected hand-unit, so one more
            # of these short downward flings now falls under
            # ``TOWARD_MIN_U`` and is reported with no direction at all --
            # which is the honest outcome for a fling that short. Still
            # 12 drops, still no sector, still nothing thrown.
            assert sum(e.toward == "down" for e in ends) >= 8, \
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
        # ROUND 3: the base is 0.30, not 0.35 -- see CastThresholds, every
        # distance bar was rescaled by 0.86 into the corrected hand-unit.
        assert th.exit_step_u == pytest.approx(CastThresholds().exit_step_u
                                               / 2.0)
        assert th.reach_min == 2.35
        assert th.throw_release_u == CastThresholds().throw_release_u

    def test_fast_rates_with_the_raw_counters_would_misfire(self):
        """Why for_fps exists: at 30 fps three frames is 100 ms."""
        grabs, _t = everyday_false(30.0, phases=2)
        assert grabs > 0

    def test_a_refit_rebuilds_the_open_history_to_the_new_lookback(self):
        """The open-history deque is sized ONCE, at construction. A refit
        that assigned ``t`` alone left it at 12 frames while the thresholds
        asked for 48, so at 30 fps the lookback silently stayed 0.4 s and
        the resting-fist rule refused every grab -- MEASURED: 0 of 48
        against 24 of 24 for the same thresholds set at construction, and
        46 of 48 at 15 fps. ``retune()`` is the only door, and it rebuilds
        the window under the lock, keeping what it held."""
        m = CastGesture(now=Clock(), preview_fps=7.5)
        assert m._open_hist.maxlen == CastThresholds().open_lookback_frames
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        for fps in (15.0, 30.0, 7.5):
            th = CastThresholds.for_fps(fps)
            m.retune(th, fps=fps)
            assert m._open_hist.maxlen == th.open_lookback_frames, fps
            assert m.t is th
            assert m.preview_fps == fps
            assert m.stall_s == pytest.approx(3.0 / fps)
        assert list(m._open_hist) == [True]        # and it kept what it had

    def test_the_refit_is_what_makes_a_fast_feed_grab_at_all(self):
        """The same thresholds, one set at construction and one arrived at
        through ``retune``, must behave identically. MEASURED at 30 fps:
        24/24 either way now; assigning ``t`` alone gave 0/24."""
        th = CastThresholds.for_fps(30.0)

        def refit_first():
            m = CastGesture(now=Clock(), preview_fps=7.5)
            m.retune(th, fps=30.0)
            return m

        built = lateral_recall(30.0, thresholds=th, phases=6)
        assert built == (24, 24, 0)
        grabs = throws = wrong = 0
        for name, scen in LATERAL.items():
            want = "left" if "left" in name else "right"
            dur, traj = scen
            for ph in range(6):
                m = refit_first()
                clk = m._now
                rng = np.random.default_rng(ph)
                t = ph * (1.0 / 30.0) / 6
                i = 0
                while t < dur:
                    e = m.update(sample(traj, t, 0.0, rng, False), EYE_PX, i)
                    if e is not None and e.kind == "grab":
                        grabs += 1
                    elif e is not None and e.kind == "throw":
                        throws += 1
                        wrong += e.sector != want
                    t += 1.0 / 30.0
                    clk.t += 1.0 / 30.0
                    i += 1
        assert (grabs, throws, wrong) == built

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
        (3-4 fist samples) and swings wide grabs 6/6, both sides, and
        lands on the NAMED side every time it lands at all. MEASURED. The
        first draft's slow variant, whose approach starts out of frame,
        grabbed only 3/6: three phases never sampled an open hand and the
        resting-fist rule refused them, which is that rule working.

        THE THROW COUNT MOVED WHEN THE FLING TEST LANDED (09-05), and it
        moved for the reason this test is about. A throw now has to be
        measurably FAST -- ``throw_speed_us``, off the clock -- and at 2 fps
        a 0.5 s sample interval smears a 0.8 s swing together with the hold
        that came before it, so the speed READS as 1.7 hand-units/s against
        a real 3.4. It was 6/6 and 6/6; it is now 4/6 and 3/6, and the rest
        are drops that still name the side in ``toward``. That is the same
        conclusion this test already drew from the other end: below the
        band the rate is the limit, and the lane must be OFF at idle rather
        than degraded. At 5.5-8.0 fps the SAME slow gesture is untouched at
        6/6, which the next test pins."""
        natural = tally(run(GOOD["his-right"], 2.0, phases=6))
        assert natural["grab"] == 0 and natural["throw"] == 0, natural
        # ROUND 3: 4/3 became 3/3. At 2 fps the fling grant is floored at
        # the loss grace (1.5 s) and the whole gesture is between samples;
        # one fewer landing at a rate the lane is meant to be OFF at is
        # the safe direction, and the sentence above already says so.
        for side, want in (("right", 3), ("left", 3)):
            end = (-300, -10, 440) if side == "right" else (310, -10, 440)
            t = tally(run(slow_gesture(end), 2.0, phases=6))
            assert t["grab"] == 6, (side, t)
            assert t["throw"] == want, (side, t)
            assert t["sectors"] == {side: want}, (side, t)

    def test_the_slow_gesture_also_lands_at_the_design_rates(self):
        """And it still does with the fling test in front of it: MEASURED
        09-05 across the whole band, 6/6 both sides at every rate. The
        slow deliberate throw peaks at 3.4 hand-units/s against the 3.0
        bar -- a 13% margin, which is the thinnest margin in this file and
        is stated as such in ``CastThresholds.throw_speed_us``."""
        for fps in (5.5, 6.0, 7.5, 8.0):
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
        # ROUND 3: the bar is 0.86, not 1.00 -- rescaled by 0.86 into the
        # corrected hand-unit, where it means the same millimetres.
        under = self._release(-0.85)
        over = self._release(-0.87)
        assert under.kind == "drop" and under.why == "released"
        assert over.kind == "throw" and over.sector == "right"
        assert over.dist_u == pytest.approx(0.87, abs=1e-6)

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
    def _exit(self, last_x, last_y, step_units, edge_ok=True, fps=6.0):
        """Grab at centre, one fist frame at (last_x, last_y) reached with
        a last step of ``step_units``, then the hand is gone.

        THE CLOCK HERE DOES NOT ADVANCE, and that is now load-bearing: with
        no measurable interval the machine falls back to one frame period
        at ``preview_fps``, so the step this drives is a speed of
        ``step_units * fps`` hand-units per second. At the 6.0 fps default
        a 0.5-unit step is 3.0 u/s -- exactly the fling bar -- so every
        caller below states a step that is unambiguously one side of it."""
        m = CastGesture(now=Clock(), preview_fps=fps)
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
        e = self._exit(100, 360, 0.8)
        assert e.kind == "throw" and e.sector == "right"
        assert e.why == "left frame (his right)"

    def test_leaving_through_the_image_right_edge_is_a_throw_to_his_left(self):
        e = self._exit(W - 100, 360, 0.8)
        assert e.kind == "throw" and e.sector == "left"
        assert e.why == "left frame (his left)"

    def test_leaving_through_the_bottom_is_the_cancel(self):
        e = self._exit(640, H - 60, 0.8)
        assert e.kind == "drop" and e.why == "cancelled (his down)"

    def test_leaving_through_the_top_is_not_a_target(self):
        e = self._exit(640, 60, 0.8)
        assert e.kind == "drop" and "his up is not a target" in e.why

    def test_the_edge_bar_is_a_quarter_hand_width_but_no_longer_decides(self):
        """``throw_exit_u`` is still the distance floor, and it is still a
        quarter of a hand-width -- but it is no longer what decides.

        A fling has to come with it (09-05), and that changes which of
        these two bars you can actually reach. At the design rates you
        cannot reach the distance bar at all: a single step small enough to
        leave the hand 0.26 units from its anchor is 1.6 hand-units/s,
        which is not a fling by any measure. So the distance bar is driven
        here at 20 fps, where the SAME 0.26-unit step is 5.2 u/s and the
        quarter-hand-width is once again what separates the two -- and the
        row underneath shows the same geometry refusing at 6 fps, which is
        the whole of the fix in two lines."""
        # ROUND 3: the bar is 0.22, not 0.25 -- rescaled by 0.86 into the
        # corrected hand-unit, where it means the same millimetres.
        for dist, kind in ((0.21, "drop"), (0.23, "throw")):
            m = CastGesture(now=Clock(), preview_fps=20.0)
            grab_at(m, 100, 360)
            m.update((obs(100 - dist * UNIT, 360, FIST),), EYE_PX, 5)
            e = [m.update((), 0.0, 6 + k) for k in range(3)][-1]
            assert e.kind == kind, (dist, e)
            if kind == "throw":
                assert e.sector == "right"
        # the same 0.23 units, carried rather than flung: 1.4 u/s at 6 fps
        m = CastGesture(now=Clock(), preview_fps=6.0)
        grab_at(m, 100, 360)
        m.update((obs(100 - 0.23 * UNIT, 360, FIST),), EYE_PX, 5)
        e = [m.update((), 0.0, 6 + k) for k in range(3)][-1]
        assert e.kind == "drop" and e.sector == ""
        assert e.why == "carried out of frame, not flung"

    def test_lost_in_open_space_needs_distance_and_motion(self):
        """Centre of the frame: frac > 0.30 from every edge, so the lost
        rule judges it. It wants distance from the anchor AND to have been
        moving on the last step -- and since 09-05 that motion has to be a
        FLING (3.0 hand-units/s; at the 6 fps this helper implies, a step
        of 0.5 units). ``exit_step_u`` is still the per-frame bar and is
        still 0.35, which is why the 0.36-unit step below is a drop now:
        it clears the old bar and is not a fling."""
        # ROUND 3: the two bars are 0.43 and 0.30, not 0.50 and 0.35 --
        # rescaled by 0.86 into the corrected hand-unit.
        far_fast = self._exit(640 - 0.53 * UNIT, 360, 0.52)
        far_carried = self._exit(640 - 0.53 * UNIT, 360, 0.31)
        far_slow = self._exit(640 - 0.44 * UNIT, 360, 0.29)
        near_fast = self._exit(640 - 0.42 * UNIT, 360, 0.52)
        assert far_fast.kind == "throw" and far_fast.sector == "right"
        assert far_fast.why == "lost"
        assert far_carried.kind == "drop" and far_carried.why == "lost, not flung"
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
        e = self._exit(100, 200, 0.8)
        assert e.kind == "throw" and e.sector == "right"
        assert e.bearing_deg == 0.0

    def test_the_edge_names_the_direction_only_when_the_hand_went_that_way(self):
        """Grabbed 80 px from the image-right edge (his left, frac 0.125),
        drifted 0.3 hand-units TOWARD THE CENTRE, then lost. The edge is
        still the nearest thing to the hand, but he was not moving toward
        it, so it may not name the direction: judged as lost in open space,
        and 0.3 u is under that bar. MEASURED before the direction had to
        agree with the edge: 3 of 6 of these threw to his left."""
        m = CastGesture(now=Clock())
        grab_at(m, W - 80, 360)
        m.update((obs(W - 80 - 0.3 * UNIT, 360, FIST),), EYE_PX, 5)
        e = [m.update((), 0.0, 6 + k) for k in range(3)][-1]
        assert e.kind == "drop" and e.why == "lost"
        assert e.sector == ""
        # ...and the same run OUTWARD is the throw it always was, given a
        # fling: 0.9 units in one frame is 5.4 hand-units/s at 6 fps.
        m = CastGesture(now=Clock())
        grab_at(m, W - 80 - 0.9 * UNIT, 360)
        m.update((obs(W - 80, 360, FIST),), EYE_PX, 5)
        e = [m.update((), 0.0, 6 + k) for k in range(3)][-1]
        assert e.kind == "throw" and e.sector == "left"
        assert e.why == "left frame (his left)"


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
        shut the whole way. The reach is re-tested on EVERY carry frame, so
        the carry ends on the first frame below ``reach_exit`` -- it does
        not wait for the hand to open or to leave the picture.

        R 1.5 is a fist about 765 mm from the lens (MEASURED over the
        synthetic hand at pitch 35 with his face at 700 mm: R 2.40 at
        450 mm, 2.01 at 550, 1.62 at 700, 1.46 at 780) -- further back than
        his own face plane, and 0.45 below the 1.95 bar. It has left."""
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        e = m.update((obs(660, 500, FIST, 1.5, 240.0),), EYE_PX, 5)
        assert e is not None
        assert e.kind == "drop" and e.why == "withdrawn"
        assert e.sector == ""
        assert m.state is CastState.COOLDOWN

    def test_a_reach_of_zero_during_a_carry_is_no_opinion_not_a_withdrawal(self):
        """The arm crosses the face at exactly the moment it matters, and
        R is 0.0 when there is no face to scale against. 0.0 means NO
        OPINION -- the same three-valued contract eye.py uses -- so it must
        not end the carry the way a measured retreat does."""
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        assert m.update((obs(645, 362, FIST, 0.0),), EYE_PX, 5) is None
        assert m.state is CastState.CARRYING

    @pytest.mark.parametrize("fps", [7.5, 6.0])
    def test_a_shoulder_height_retraction_still_closed_never_throws(self, fps):
        """The adversarial pass's case, through the whole engine: grab at
        425 mm, pull the fist back to shoulder height 600 mm away with a
        full hand-unit of sideways image travel, open it there. MEASURED
        with the bar removed (reach_exit=0.0): 68 of 68 THREW. With it:
        0 throws, and the drop still names the side the hand went."""
        for name, want in (("his-left", "left"), ("his-right", "right")):
            ends = [e for _p, e in run(WITHDRAW[name], fps, phases=12)]
            drops = [e for e in ends if e.kind == "drop"]
            assert sum(e.kind == "grab" for e in ends) == 12, name
            assert sum(e.kind == "throw" for e in ends) == 0, name
            assert len(drops) == 12, name
            assert all(e.why == "withdrawn" for e in drops), name
            assert all(e.sector == "" for e in drops), name
            assert all(e.toward == want for e in drops), (name, want)

    def test_without_the_exit_bar_that_same_retraction_throws(self):
        """Why the bar is there, stated as the measurement that found it.
        The ONLY difference is ``reach_exit``."""
        off = CastThresholds(reach_exit=0.0)
        t = tally(run(WITHDRAW["his-left"], 7.5, thresholds=off, phases=12))
        # 10 of 12 -- it was 12 of 12 before the fling test landed, which
        # catches the other two on its own: a retraction to the shoulder is
        # a slow motion as well as a backward one. Two independent cancels
        # for one class of accident is the design working, not redundancy.
        # ROUND 3: 10 became 12 -- with ``reach_exit`` switched OFF, as
        # this test does deliberately, the corrected unit reads the
        # retraction's speed honestly and two more of them clear the
        # fling bar. The point of the test is unchanged and is made by
        # the line above it: with reach_exit ON, none of these throws.
        assert t["throw"] == 12 and t["drop"] == 0
        assert t["sectors"] == {"left": 12}

    def test_the_exit_bar_costs_no_lateral_recall(self):
        """It may only remove throws that were wrong. MEASURED at 0, 4 and
        8 px of landmark noise, both rates: identical either side."""
        off = CastThresholds(reach_exit=0.0)
        for fps in (7.5, 6.0):
            for noise in (0.0, 4.0, 8.0):
                with_bar = lateral_recall(fps, noise=noise)
                without = lateral_recall(fps, thresholds=off, noise=noise)
                assert with_bar == without, (fps, noise, with_bar, without)
                assert with_bar[2] == 0

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

    def test_a_carry_whose_frames_stop_is_swept_up_without_one(self):
        """The frame cap and the stall backstop both live inside
        ``update()``, so a carry whose frames simply STOPPED -- the worker
        halted on a standby edge, a wedged read -- had nothing to end it:
        MEASURED, 60 s of silence left the machine CARRYING and a spoken
        throw then cast the stale subject. ``sweep()`` is the wall-clock
        cap applied with no frame at all, and every voice path calls it."""
        put, seen = [], []
        pay = CallablePayload(pick_up=lambda: ("d1", "the lab report"),
                              put_back=put.append)
        clk = Clock()
        m = CastGesture(now=clk, payload=pay, on_event=seen.append)
        grab_at(m, 640, 360)
        assert m.held == "the lab report"
        clk.t += 7.9                              # inside the 8.0 s cap
        assert m.sweep() is None
        assert m.state is CastState.CARRYING and m.held == "the lab report"
        clk.t += 52.1                             # 60.0 s of silence
        e = m.sweep()
        assert e is not None and e.kind == "drop" and e.why == "timeout"
        assert e.payload == "the lab report"
        assert put == ["d1"] and m.held == ""
        assert seen[-1] is e
        assert m.state is CastState.COOLDOWN
        assert m.sweep() is None                  # nothing left to sweep

    def test_sweeping_outside_a_carry_says_nothing(self):
        clk = Clock()
        m = CastGesture(now=clk)
        assert m.sweep() is None
        m.update((obs(640, 360, OPEN),), EYE_PX, 0)
        m.update((obs(640, 360, FIST),), EYE_PX, 1)
        assert m.state is CastState.CLOSING
        clk.t += 60.0
        assert m.sweep() is None
        assert m.state is CastState.CLOSING       # only a frame moves this

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
        # ROUND 3 rescaled every distance bar by 0.86 -- the ratio between
        # palm_diag and the corrected hand-unit at his working pose -- so
        # each one means the same millimetres it was measured at and stops
        # moving with his wrist. The speed bar was re-measured outright.
        assert (t.throw_release_u, t.throw_exit_u, t.throw_lost_u,
                t.exit_step_u, t.edge_frac) == (0.86, 0.22, 0.43, 0.30, 0.30)
        assert (t.throw_speed_us, t.fling_window_s) == (2.65, 0.55)
        assert (t.closed_ratio_max, t.open_ratio_min) == (1.05, 99.0)
        assert (t.yaw_hold_deg, t.yaw_required) == (25.0, True)
        assert t.carry_max_s == 8.0 and t.target_sectors == ("left", "right")


# ============================================ a throw must be a throw
# HIS DESK, and it is the whole reason this section exists. The LifeCam
# sits ON TOP OF THE SPARK'S MONITOR -- the right-hand screen, looking
# across at him -- so a grab at that screen is close to the lens and his
# hands are beside that lens all day. Reaching for a mug there looks
# exactly like grabbing at that screen, which makes the Spark-as-source
# case the one most exposed to a false fire. Every family below is a
# motion AT that screen, and the grid is weighted accordingly.
#
# THE ATTACK, MEASURED before the fix: reach at the screen, CLOSE the hand
# on something, carry it laterally out of the picture. No fling, no
# release, no follow-through. At a 3.2 s traverse (147 mm/s, twenty times
# slower than a throw) it fired every time, because gesture.py treated
# LEAVING THE PICTURE as the evidence of intent (throw_exit_u = 0.25).
# A man carrying a mug leaves the picture.

def carry_out(side, end_z=425.0, traverse=3.2, hold=0.50, pitch=35.):
    """The attack. ``side`` names the side HE carries it toward.

    Reach beside the screen, close on something, lift it to ``end_z`` and
    carry it out of the picture over ``traverse`` seconds. The hand is
    STILL SHUT when the picture loses it: nothing here is a fling.
    """
    sgn = 1.0 if side == "his-left" else -1.0        # image +x is HIS LEFT
    lift, tail, x0 = 0.45, 1.2, 15.0
    gone = sgn * (0.72 * float(end_z) + 200.0)       # well outside the frame
    return (0.55 + 0.20 + hold + lift + traverse + tail, lambda t: seg(t, [
        (0.55, (140, 300, 780), (10, 10, 430), 0.10, 0.06, pitch - 5, 0, 0),
        (0.20, (10, 10, 430), (10, 10, 425), 0.06, 1.00, pitch, 0, 0),
        (hold, (10, 10, 425), (x0, 5, 425), 1.00, 1.00, pitch, 0, 0),
        (lift, (x0, 5, 425), (x0, 8, end_z), 1.00, 1.00, pitch, 0, 0),
        (traverse, (x0, 8, end_z), (gone, 10, end_z), 1.00, 1.00, pitch, 0, 0),
        (tail, (gone, 10, end_z), (gone, 10, end_z), 1.00, 1.00, pitch,
         0, 0)]))


def slide_aside(side, mm=200.0, secs=1.8, z=430.0):
    """Close on something beside the screen and slide it aside IN FRAME,
    then let go there. The RELEASE branch rather than the exit branch: he
    sets a thing down 200 mm to one side, which is 1.6 hand-units and
    clears ``throw_release_u`` easily."""
    sgn = 1.0 if side == "his-left" else -1.0
    ex = sgn * float(mm)
    return (0.55 + 0.20 + 0.4 + secs + 0.9, lambda t: seg(t, [
        (0.55, (140, 300, 780), (10, 10, 435), 0.10, 0.06, 30, 0, 0),
        (0.20, (10, 10, 435), (10, 10, z), 0.06, 1.00, 35, 0, 0),
        (0.40, (10, 10, z), (12, 8, z), 1.00, 1.00, 35, 0, 0),
        (secs, (12, 8, z), (ex, 12, z), 1.00, 1.00, 35, 0, 0),
        (0.90, (ex, 12, z), (ex, 14, z), 1.00, 0.06, 35, 0, 0)]))


# The traverses and the ending depths the attack swept. 425-550 mm is the
# band that stays AT reach depth (the mug slid across the desk at arm's
# length); 600 and 700 mm are already caught by ``reach_exit``, which is
# the ending-depth discriminator the attack named.
ATTACK_TRAVERSES = (3.2, 2.4, 2.0)
ATTACK_DEPTHS = (425.0, 460.0, 500.0, 550.0, 600.0, 700.0)

CARRY_OUT = {"carry-out/%s/%.1fs/%.0fmm" % (side, tv, z):
             carry_out(side, z, tv)
             for side in ("his-left", "his-right")
             for tv in ATTACK_TRAVERSES
             for z in ATTACK_DEPTHS}

SLIDE = {"slide-aside/%s/%.0fmm/%.1fs" % (side, mm, secs):
         slide_aside(side, mm, secs)
         for side in ("his-left", "his-right")
         for mm, secs in ((200.0, 1.8), (260.0, 2.4))}

# Ordinary life beside that monitor: the common case, not the edge case.
AT_THE_SCREEN = {
    "mug-to-mouth": (3.4, lambda t: seg(t, [
        (0.6, (150, 240, 700), (60, 40, 430), 0.15, 0.10, 25, 0, 0),
        (0.3, (60, 40, 430), (60, 40, 428), 0.10, 0.95, 30, 0, 0),
        (0.9, (60, 40, 428), (90, -120, 520), 0.95, 0.95, 25, 10, 5),
        (0.8, (90, -120, 520), (95, -125, 525), 0.95, 0.95, 20, 10, 5),
        (0.8, (95, -125, 525), (150, 240, 700), 0.95, 0.15, 25, 0, 0)])),
    "keyboard-hands": (5.0, lambda t: (
        (95 + 55 * math.sin(4.1 * t), 235 + 12 * math.sin(6.3 * t), 545),
        0.40 + 0.12 * math.sin(9.0 * t), -20, 8 * math.sin(2.0 * t), 0)),
    "scratch-at-the-screen": (3.4, lambda t: seg(t, [
        (0.8, (150, 240, 690), (30, -80, 445), 0.45, 0.70, 15, 0, 0),
        (1.4, (30, -80, 445), (45, -70, 440), 0.70, 0.72, 18, 8, 12),
        (1.2, (45, -70, 440), (150, 240, 690), 0.72, 0.45, 15, 0, 0)])),
    "stretch-arms-wide": (2.2, lambda t: seg(t, [
        (0.9, (60, 120, 600), (330, -160, 520), 0.10, 0.12, 10, 20, 0),
        (0.6, (330, -160, 520), (335, -165, 515), 0.12, 0.14, 10, 25, 0),
        (0.7, (335, -165, 515), (60, 120, 600), 0.14, 0.10, 10, 20, 0)])),
    "grab-the-bezel": (3.0, lambda t: seg(t, [
        (0.7, (150, 250, 700), (-30, -40, 415), 0.10, 0.90, 28, 0, 0),
        (1.0, (-30, -40, 415), (-15, -25, 405), 0.90, 0.92, 30, 10, 8),
        (1.3, (-15, -25, 405), (150, 250, 700), 0.92, 0.10, 28, 0, 0)])),
    "reach-past-the-screen": (3.2, lambda t: seg(t, [
        (0.9, (140, 250, 700), (150, -60, 430), 0.12, 0.55, 25, 15, 0),
        (1.1, (150, -60, 430), (175, -55, 425), 0.55, 0.60, 28, 20, 5),
        (1.2, (175, -55, 425), (140, 250, 700), 0.60, 0.12, 25, 15, 0)])),
}
AT_THE_SCREEN.update({k + "/fingers-down": fingers_down(v)
                      for k, v in list(AT_THE_SCREEN.items())})

# The grid. Weighted to his desk: 36 carry-outs and 4 slides at that
# monitor, 12 ordinary motions beside it, then the design's own 24
# everyday trajectories, the 2 withdrawals and the deliberate put-back.
DESK_GRID = {}
DESK_GRID.update(CARRY_OUT)
DESK_GRID.update(SLIDE)
DESK_GRID.update(AT_THE_SCREEN)
DESK_GRID.update({"everyday/" + k: v for k, v in BAD.items()})
DESK_GRID.update({"withdraw/" + k: v for k, v in WITHDRAW.items()})
DESK_GRID["put-back"] = PUT_BACK


def desk_misfire(fps=7.5, phases=16, thresholds=None, **kw):
    """P(fire | not a cast gesture) over the desk grid, and who fired."""
    fires = {}
    grabs = 0
    for name, scen in DESK_GRID.items():
        t = tally(run(scen, fps, thresholds=thresholds, phases=phases, **kw))
        grabs += t["grab"]
        if t["throw"]:
            fires[name] = t["throw"]
    seqs = len(DESK_GRID) * phases
    return {"sequences": seqs, "fires": sum(fires.values()),
            "rate": sum(fires.values()) / seqs, "grabs": grabs,
            "who": fires}


class TestAThrowMustBeAThrow:
    """Leaving the picture is not intent. A man carrying a mug leaves the
    picture; what he does not do is FLING.

    THE RULE: a carry becomes a throw only if the hand was still travelling
    at throw speed -- ``throw_speed_us``, 3.0 hand-units a second, about
    380 mm/s at his reach -- within ``fling_window_s`` of the last frame
    that saw it. The distance bars are untouched: distance cannot separate
    these two at all, because a mug carried out of the picture and a thrown
    hand cover the SAME ~2 units before the frame edge takes them.
    """

    def test_the_named_attack_a_mug_carried_out_of_the_picture(self):
        """THE BLOCKER. 3.2 s of traverse, 147 mm/s, both directions, every
        ending depth. MEASURED before the fix: it fired 18 of 18 at the
        depths that stay at reach."""
        fired = {}
        for side in ("his-left", "his-right"):
            for z in ATTACK_DEPTHS:
                t = tally(run(carry_out(side, z, 3.2), 7.5, phases=18))
                if t["throw"]:
                    fired["%s/%.0fmm" % (side, z)] = t["throw"]
        assert fired == {}, fired

    def test_the_whole_slow_carry_band_is_silent_in_both_directions(self):
        """Every traverse the attack swept, at both design rates."""
        for fps in (6.0, 7.5):
            for tv in ATTACK_TRAVERSES:
                for side in ("his-left", "his-right"):
                    for z in ATTACK_DEPTHS:
                        t = tally(run(carry_out(side, z, tv), fps, phases=8))
                        assert t["throw"] == 0, (fps, tv, side, z, t)

    def test_setting_a_thing_down_beside_the_screen_is_not_a_throw(self):
        """The same class on the RELEASE branch: he closes on something
        beside the monitor, slides it 200 mm aside -- 1.6 hand-units, well
        past ``throw_release_u`` -- and lets go. MEASURED before the fix:
        6 of 6, both directions."""
        for name, scen in SLIDE.items():
            for fps in (6.0, 7.5):
                t = tally(run(scen, fps, phases=8))
                assert t["throw"] == 0, (name, fps, t)

    def test_the_desk_misfire_rate(self):
        """THE HEADLINE, on the grid weighted to where his hands actually
        are: 77 families x 16 phases = 1232 sequences, of which 36 are the
        carry-out at that monitor and 4 are the slide-aside beside it.

        MEASURED ON THIS SAME GRID, before and after the fling test:

            P(fire | not a cast gesture)   6.0 fps   7.5 fps
            before                          0.3636    0.3636
            after                           0.0000    0.0000

        448 of 1232 became 0 of 1232. The false GRAB count barely moves
        (720 and 726) and is not meant to: a grab costs a tone and a chip
        that drops itself, and this whole file exists because a false
        THROW costs a desktop on a monitor he is working at."""
        for fps in (6.0, 7.5):
            res = desk_misfire(fps=fps)
            assert res["rate"] == 0.0, (fps, res["who"])

    def test_the_desk_misfire_rate_survives_landmark_noise(self):
        """ROUND 3: 0.0000 became 1 in 462 at 5 px on THIS grid, and that
        is a fair trade rather than a regression to hide. The speed bar
        moved from 3.0 to 2.65 in a unit that is now honest, which is a
        tighter bar in millimetres at every pose except the one this grid
        was built at; on the attacker's much wider grid the same change
        takes P(fire | not a cast) from 0.2700 to 0.1857. One misfire in
        462 is 0.0022, still inside the 0.005 bar this lane is held to."""
        for noise in (3.0, 5.0):
            res = desk_misfire(fps=7.5, phases=6, noise=noise,
                               seed=int(noise))
            assert res["rate"] <= 0.005, (noise, res["who"])

    def test_the_ordinary_throw_still_fires(self):
        """THE OTHER HALF. The same gesture has to work when he means it.
        Full lateral recall at every rate in the design band, with the
        tracker's own noise, and never to the wrong side."""
        for fps in (5.5, 6.0, 7.5, 8.0):
            grabs, throws, wrong = lateral_recall(fps, phases=6)
            assert (grabs, throws, wrong) == (24, 24, 0), fps
        for noise in (3.0, 5.0):
            grabs, throws, wrong = lateral_recall(7.5, noise=noise,
                                                  seed=int(noise), phases=6)
            assert (grabs, throws, wrong) == (24, 24, 0), noise

    def test_distance_cannot_tell_them_apart_and_speed_can(self):
        """WHY the fix is not a raised ``throw_exit_u``. Both motions cover
        the same ground before the picture loses them; only the speed of
        the hand at that moment separates them, and by a factor of three."""
        thrown = _carry_end_scalars(GOOD["his-right/fingers-down"], 7.5)
        carried = _carry_end_scalars(carry_out("his-right", 425.0, 3.2), 7.5)
        assert thrown and carried
        # the same ground... (ROUND 3: read in the corrected hand-unit,
        # which is 16% larger at this pose, so every figure here is 0.86
        # of what round 2 measured. The RATIO between the two families,
        # which is what the test is about, is untouched.)
        assert min(d for d, _s in thrown) > 0.86
        assert min(d for d, _s in carried) > 0.86
        # ...at wholly different speeds
        assert min(s for _d, s in thrown) > 2.65
        assert max(s for _d, s in carried) < 1.80

    def test_a_mug_snatched_out_faster_than_a_fling_is_the_stated_limit(self):
        """THE RESIDUAL, stated rather than hidden. A hand that leaves the
        picture at more than ~380 mm/s IS a fling by every signal this rig
        has, so a genuine snatch at that speed still fires. MEASURED at
        7.5 fps, 24 sequences per row, a mug carried out of the picture:

            3.2 s traverse  147 mm/s    0/24 fire
            2.4 s           196 mm/s    0/24
            2.0 s           235 mm/s    0/24
            1.6 s           294 mm/s   12/24   <- the boundary
            1.2 s           392 mm/s   24/24
            0.7 s           671 mm/s   24/24

        Nothing here can close that gap without refusing his own slow
        deliberate throw, which sits inside the same band: measured floor
        3.02 hand-units/s against the 1.6 s carry's peak of 3.34. Those
        two are the same motion, and the bar is placed between the mug's
        ORDINARY speeds and his gesture rather than between two things
        that can actually be told apart."""
        fast = tally(run(carry_out("his-right", 425.0, 1.0), 7.5, phases=8))
        assert fast["throw"] > 0
        slow = tally(run(carry_out("his-right", 425.0, 2.0), 7.5, phases=8))
        assert slow["throw"] == 0

    def test_the_speed_bar_is_per_second_so_for_fps_leaves_it_alone(self):
        """A throw is fast IN THE WORLD, not fast per sample. The frame
        counters rescale; this one must not."""
        base = CastThresholds()
        for fps in (5.0, 7.5, 15.0, 30.0):
            t = CastThresholds.for_fps(fps)
            assert t.throw_speed_us == base.throw_speed_us, fps
            assert t.fling_window_s == base.fling_window_s, fps

    def test_the_machine_reports_the_speed_it_judged_on(self):
        m = CastGesture(now=Clock())
        grab_at(m, 640, 360)
        st = m.status()
        assert "speed_us" in st and "flung" in st
        assert_numbers_only(st)


def _carry_end_scalars(scen, fps, phases=12):
    """(dist_u, speed_us) at every carry end past a fifth of a unit."""
    dur, traj = scen
    out = []
    clk_rate = 1.0 / fps
    for ph in range(phases):
        clk = Clock()
        rng = np.random.default_rng(ph)
        m = CastGesture(CastThresholds.for_fps(fps), (W, H), now=clk,
                        preview_fps=fps)
        t = ph * clk_rate / phases
        i, last_seen, peak = 0, None, 0.0
        while t < dur:
            hands = sample(traj, t, 0.0, rng)
            carrying = m.state is CastState.CARRYING
            e = m.update(hands, EYE_PX, i)
            if carrying and hands:
                dt = (clk.t - last_seen) if last_seen is not None else clk_rate
                peak = max(peak, m._last_step_u / (dt if dt > 0 else clk_rate))
            if hands:
                last_seen = clk.t
            if e is not None and e.kind in ("throw", "drop") and e.dist_u > 0.2:
                out.append((e.dist_u, peak))
                peak = 0.0
            t += clk_rate
            clk.t += clk_rate
            i += 1
    return out
