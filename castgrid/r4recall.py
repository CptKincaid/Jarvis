"""ROUND 4's recall surface: does his own throw still fire, everywhere?

Round 3 measured recall at ONE point -- a 300-310 mm swing ending at
z = 440 -- and the round-3 adversary showed that point is the single most
favourable spot on a cliff round 3 itself created. This runner measures
the whole surface instead: swings 270-330 mm in 10 mm steps, end depths
420-470 mm in 10 mm steps, both directions, the 5.5-8.0 fps design band,
three landmark-noise levels, every sub-frame phase.

IT RUNS THE SAME MEASUREMENT AGAINST TWO CHECKOUTS. ``tests/atkharness``
and ``tests/atkgrid`` are the attacker's files and are used BYTE-FOR-BYTE;
what this adds is the ability to point the same loop at a DIFFERENT
``jarvis/gesture.py`` loaded out of a file, so round 2's numbers are
re-measured here rather than quoted from a report. Put round 2's module
somewhere and name it in ``R2_GESTURE``:

    git show 53537c7:jarvis/gesture.py > /tmp/.../gesture_53537c7.py
    R2_GESTURE=/tmp/.../gesture_53537c7.py \
      ~/.local/bin/memcap timeout 1700 ~/vss_env/bin/python -m pytest -q \
        -p no:cacheprovider castgrid/test_r4_recall.py -s

NO CAMERA, NO CAPTURE DEVICE, NO FRAME. Every hand is a synthetic 21-point
row projected in-process through his LifeCam constants.

THE HONEST LIMIT, carried forward unchanged: the hand-unit of 127 mm is
standard adult anthropometry, GUESSED, not measured on him, and the throw
model is one attacker's reconstruction of his gesture. A recall of 1.0000
here means "this model of his throw fires on this model of his hand", not
"his throw will fire".
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import numpy as np

from tests.atkharness import EYE_PX, W, H, Clock, hand3d, project, _curls
from tests import atkgrid

# The surface, and it is the one the brief names.
SWINGS = (270., 280., 290., 300., 310., 320., 330.)
DEPTHS = (420., 430., 440., 450., 460., 470.)
SIDES = {"his-left": 1.0, "his-right": -1.0}
# The design band, ends included. camera.preview_fps defaults to 6.0 and
# the camera measures ~7.5.
RATES = (5.5, 6.0, 6.5, 7.0, 7.5, 8.0)
NOISES = (0.0, 3.0, 5.0)
PHASES = 16


def load_gesture(path: Optional[str]):
    """Load a ``jarvis/gesture.py`` out of a file as its own module, so the
    same loop can be run against two checkouts in one process. Returns None
    when the path is not set or not there -- the caller then reports the
    baseline column as absent rather than inventing it."""
    if not path or not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("gesture_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    # It must be in sys.modules BEFORE it executes: @dataclass resolves
    # its own class's module out of there, and a frozen dataclass in a
    # module that is not registered raises on definition.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def throw(side: str, swing: float, z: float, *, slow=True, swing_s=0.8,
          fingers_down=False):
    """HIS deliberate throw, parameterised by how far it swings and how far
    from the lens it ends. ``atkgrid.throw_slow`` / ``throw_fast``
    unmodified; only the end point moves."""
    end = (SIDES[side] * float(swing), -10., float(z))
    scen = (atkgrid.throw_slow(end, swing=swing_s) if slow
            else atkgrid.throw_fast(end))
    return atkgrid.fingers_down(scen) if fingers_down else scen


def run_one(mod, scen, fps, phase, phases, thresholds=None, noise=0.0,
            seed=11, look=True):
    """ONE sequence against ``mod``'s machine. Events back.

    ``look`` is the head model and it is MODEL A from round 3: his head is
    on the screen he grabbed, which is what a deliberate throw looks like.
    A module with no ``looking`` argument (round 2) is called without it.
    """
    dur, traj = scen
    clk = Clock()
    rng = np.random.default_rng(seed * 1009 + phase)
    th = thresholds if thresholds is not None else mod.CastThresholds()
    m = mod.CastGesture(th, (W, H), now=clk, preview_fps=fps)
    has_look = hasattr(mod, "NO_HEAD")
    t = phase * (1.0 / fps) / phases
    i = 0
    events = []
    while t < dur:
        pos, curl, pi, ya, ro = traj(t)
        img = project(hand3d(_curls(curl), False), pos, pi, ya, ro)
        if img is None:
            hands = ()
        else:
            if noise:
                img = img + rng.normal(0.0, noise, img.shape)
            hands = (mod.observe_hand(img, EYE_PX),)
        e = (m.update(hands, EYE_PX, i, looking=look) if has_look
             else m.update(hands, EYE_PX, i))
        if e is not None:
            events.append(e)
        step = 1.0 / fps
        t += step
        clk.t += step
        i += 1
    return events


def cell(mod, side, swing, z, *, thresholds=None, rates=RATES, noises=NOISES,
         phases=PHASES, kinds=((True, False), (True, True))):
    """(fired, n, wrong_side) for one point on the surface.

    ``kinds`` is (slow, fingers_down) pairs: his slow deliberate throw with
    the fingers up and with them down, which is the same gesture made with
    the hand rolled over and is one of the attacker's own families.
    """
    want = "left" if side == "his-left" else "right"
    fired = n = wrong = 0
    for slow, fdown in kinds:
        scen = throw(side, swing, z, slow=slow, fingers_down=fdown)
        for fps in rates:
            for px in noises:
                for ph in range(phases):
                    evs = run_one(mod, scen, fps, ph, phases,
                                  thresholds=thresholds, noise=px)
                    n += 1
                    thr = [e for e in evs if e.kind == "throw"]
                    if thr:
                        fired += 1
                        if any(e.sector != want for e in thr):
                            wrong += 1
    return fired, n, wrong


def surface(mod, *, thresholds=None, swings=SWINGS, depths=DEPTHS,
            sides=("his-left", "his-right"), **kw):
    """The whole grid. ``{(side, swing, z): (fired, n, wrong)}``."""
    out = {}
    for side in sides:
        for swing in swings:
            for z in depths:
                out[(side, swing, z)] = cell(mod, side, swing, z,
                                             thresholds=thresholds, **kw)
    return out


def render(out, side, *, swings=SWINGS, depths=DEPTHS, label=""):
    """One side of the surface as a table of recall."""
    lines = []
    if label:
        lines.append(label)
    lines.append("    z(mm)  " + "".join("%8.0f" % z for z in depths))
    for swing in swings:
        row = ["  %5.0f  " % swing]
        for z in depths:
            fired, n, _w = out[(side, swing, z)]
            row.append("%8.4f" % (fired / n if n else 0.0))
        lines.append("".join(row))
    return "\n".join(lines)


def totals(out):
    fired = sum(v[0] for v in out.values())
    n = sum(v[1] for v in out.values())
    wrong = sum(v[2] for v in out.values())
    return fired, n, wrong


def floor_at(out, side, z, *, swings=SWINGS):
    """The smallest swing at which recall is 1.0000 at that depth, or None.
    The 'all-or-nothing floor' the brief names."""
    best = None
    for swing in sorted(swings):
        fired, n, _w = out[(side, swing, z)]
        if n and fired == n:
            best = swing if best is None else min(best, swing)
    return best


def worst(out, k=8):
    """The k worst cells, so a cliff cannot hide inside an average."""
    rows = [(v[0] / v[1] if v[1] else 0.0, key) for key, v in out.items()]
    rows.sort()
    return rows[:k]


def peak_us(mod, side, swing, z, *, fps=7.5, phases=8, thresholds=None):
    """The FASTEST the machine measures HIS OWN throw going, in units per
    second, over the sub-frame phases. This is the number the fling bar is
    compared against, so a bar inside this spread cuts into his gesture --
    which is what round 3's 2.65 did.

    It reads the machine's own scalar rather than dividing millimetres by
    seconds, because the throw moves in DEPTH as well as laterally and the
    image displacement is what is actually measured. A millimetre figure
    derived on the side would be arithmetic dressed as a measurement.
    """
    from tests.atkharness import F_PX                       # noqa: F401
    best = 0.0
    scen = throw(side, swing, z)
    dur, traj = scen
    th = thresholds if thresholds is not None else mod.CastThresholds()
    for ph in range(phases):
        clk = Clock()
        m = mod.CastGesture(th, (W, H), now=clk, preview_fps=fps)
        t = ph * (1.0 / fps) / phases
        i = 0
        while t < dur:
            pos, curl, pi, ya, ro = traj(t)
            img = project(hand3d(_curls(curl), False), pos, pi, ya, ro)
            hands = () if img is None else (mod.observe_hand(img, EYE_PX),)
            if hasattr(mod, "NO_HEAD"):
                m.update(hands, EYE_PX, i, looking=True)
            else:
                m.update(hands, EYE_PX, i)
            best = max(best, float(m._speed_us))
            step = 1.0 / fps
            t += step
            clk.t += step
            i += 1
    return best


def mm_per_unit(z=440.0, pitch=35.0):
    """How many millimetres ONE HAND-UNIT measures, statically, at a pose.

    This is the SIZE OF THE UNIT and not a conversion for a speed bar: the
    throw changes depth as it swings, so the image displacement the machine
    measures is not the lateral millimetres divided by this. It is here to
    show one thing only -- that the pose-corrected unit still under-reads
    the 127 mm it is documented as.
    """
    from tests.atkharness import F_PX
    img = project(hand3d((1., 1., 1., 1.)), (0., 0., z), pitch, 0., 0.)
    if img is None:
        return float("nan")
    import jarvis.gesture as g
    o = g.observe_hand(img, EYE_PX)
    return float(o.unit) * z / F_PX if hasattr(o, "unit") else float("nan")


__all__ = ["DEPTHS", "NOISES", "PHASES", "RATES", "SIDES", "SWINGS", "cell",
           "floor_at", "load_gesture", "mm_per_unit", "peak_us", "render",
           "run_one", "surface", "throw", "totals", "worst"]
