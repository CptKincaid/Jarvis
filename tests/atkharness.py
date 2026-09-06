"""ATTACKER's own harness for the gesture cast. Numbers only, no frames.

Independent reconstruction of the desk. NOTHING here opens a camera, a
capture device or a RustDesk session; every hand is 21 synthetic landmark
rows projected through the LifeCam constants in-process.

THE ROOM, as he stated it 09-05: the LifeCam sits ON TOP OF THE SPARK'S
OWN MONITOR, the RIGHT screen, looking across at him. HPCOMPUTER's middle
and left screens are off to one side. So:
  * z ~ 400-460 mm is the RIGHT screen and everything beside it -- his mug,
    his keyboard's right end, the bezel. His hands are there all day.
  * the grab bar reach_min 2.35 is met at z <= ~455 mm at his working pitch,
    so the right screen is the ONLY thing in easy grab range.
That is why this grid is weighted at the right monitor.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import jarvis.gesture as g
from jarvis.gesture import CastGesture, CastThresholds, observe_hand

# ---------------------------------------------------------------- the lens
F_PX = 1280.0 / (2.0 * math.tan(math.radians(65.6) / 2.0))
W, H = 1280, 720
IPD_MM = 63.0
FACE_Z = 700.0
EYE_PX = F_PX * IPD_MM / FACE_Z

MCP = {"index": (-24., 100., 0.), "middle": (0., 107., 0.),
       "ring": (22., 104., 0.), "pinky": (44., 95., 0.)}
PHAL = {"index": (40., 24., 18.), "middle": (45., 27., 20.),
        "ring": (42., 25., 19.), "pinky": (33., 19., 17.)}
THUMB = [(-32., 30., 8.), (-52., 55., 12.), (-66., 72., 14.), (-76., 88., 16.)]
IDX = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
       "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FULL_FIST = (90., 100., 70.)
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


@lru_cache(maxsize=8192)
def hand3d(curls: tuple, left: bool = False) -> np.ndarray:
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


# ------------------------------------------------------------ trajectories
def _lerp(a, b, u):
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def _mix(a, b, u):
    if isinstance(a, (tuple, list)):
        return _lerp(a, b, u)
    return a + (b - a) * u


def seg(t, spans):
    """(duration, pos0, pos1, curl0, curl1, pitch, yaw, roll) per span."""
    acc = 0.
    for d, p0, p1, c0, c1, pi, ya, ro in spans:
        if t < acc + d:
            u = (t - acc) / d if d else 0.
            return _lerp(p0, p1, u), _mix(c0, c1, u), pi, ya, ro
        acc += d
    d, p0, p1, c0, c1, pi, ya, ro = spans[-1]
    return p1, c1, pi, ya, ro


class Clock:
    def __init__(self, t: float = 100.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


def _sample(traj, t, noise, rng, left=False):
    pos, curl, pi, ya, ro = traj(t)
    img = project(hand3d(_curls(curl), left), pos, pi, ya, ro)
    if img is None:
        return ()
    if noise and rng is not None:
        img = img + rng.normal(0.0, noise, img.shape)
    return (observe_hand(img, EYE_PX),)


def run_seq(scen, fps, phase, phases, thresholds=None, noise=0.0, seed=0,
            eye_fn=None, left=False):
    """ONE sequence: one trajectory at one sub-frame offset. Events back."""
    dur, traj = scen
    clk = Clock()
    rng = np.random.default_rng(seed * 1009 + phase)
    m = CastGesture(thresholds or CastThresholds(), (W, H),
                    now=clk, preview_fps=fps)
    t = phase * (1.0 / fps) / phases
    i = 0
    events = []
    while t < dur:
        eye = EYE_PX if eye_fn is None else eye_fn(t)
        pos, curl, pi, ya, ro = traj(t)
        img = project(hand3d(_curls(curl), left), pos, pi, ya, ro)
        if img is None:
            hands = ()
        else:
            if noise:
                img = img + rng.normal(0.0, noise, img.shape)
            hands = (observe_hand(img, eye if eye > 0 else 0.0),)
        e = m.update(hands, eye, i)
        if e is not None:
            events.append(e)
        step = 1.0 / fps
        t += step
        clk.t += step
        i += 1
    return events


def wilson(k: int, n: int, z: float = 1.959963985):
    """95% Wilson score interval for k/n. Sound at k = 0, which is the
    whole reason it is used here rather than a normal approximation."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, c - h), min(1.0, c + h))
