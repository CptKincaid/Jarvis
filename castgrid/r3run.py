"""ROUND 3's runner over the ATTACKER's preserved grid.

``tests/atkharness.py`` and ``tests/atkgrid.py`` are the attacker's files
and are used BYTE-FOR-BYTE: same families, same phases, same seeds, so
every number here is directly comparable with 0.2700 rather than being a
fresh grid that agrees with itself. What is added is one thing the
attacker's runner had no way to carry -- the tri-state ``looking`` the
round-3 machine now takes -- so this file re-implements ``run_seq`` with
that one extra argument and ASSERTS that with no head model it reproduces
``run_seq`` event for event.

NO CAMERA, NO CAPTURE DEVICE, NO FRAME. Every hand is a synthetic
21-point row projected in-process through his LifeCam constants.
"""
from __future__ import annotations

import numpy as np

from jarvis.gesture import (NO_HEAD, CastGesture, CastThresholds,
                            observe_hand)
from tests.atkharness import (EYE_PX, W, H, Clock, hand3d, project, _curls,
                              run_seq, wilson)                    # noqa: F401
from tests import atkgrid


def run_look(scen, fps, phase, phases, thresholds=None, noise=0.0, seed=0,
             eye_fn=None, left=False, look_fn=None):
    """One sequence, with an optional HEAD MODEL.

    ``look_fn(t)`` returns True / False / None -- the tri-state the hand
    stage would hand in: was his head still pointed where it was pointed
    before the reach began? None is no clean face row.

    With ``look_fn`` itself None the machine is told NO_HEAD, which is not
    the same thing and is the difference between a measurement and a
    tautology: a grid of hands with no faces in it has no head to offer,
    so the yaw gate abstains ENTIRELY and the number measures the
    hand-track fixes alone. Passing None per frame instead would make
    ``yaw_required`` refuse all 2752 sequences and report 0.0000, which
    would be true and useless -- precisely how round 2's 0.0000 was.
    """
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
        e = m.update(hands, eye, i,
                     looking=NO_HEAD if look_fn is None else look_fn(t))
        if e is not None:
            events.append(e)
        step = 1.0 / fps
        t += step
        clk.t += step
        i += 1
    return events


# --------------------------------------------------------- the head models
# NEITHER OF THESE IS MEASURED ON HIM, and that is the whole caveat on the
# yaw column of the round-3 table. The attacker's grid has no head in it at
# all, so a head had to be invented to measure the gate; two were, and they
# bracket the honest answer.
def head_locked(_t):
    """MODEL A -- THE HAZARD, and the pessimistic bound. His head never
    leaves the screen the camera sits on, whatever his hand is doing: he
    is reading it while he moves the mug. The gate then vetoes NOTHING."""
    return True


def head_follows(t0, lag=0.25):
    """MODEL B -- his head follows what his hand carries away. The turn
    starts ``lag`` seconds after the hand leaves the anchor and the gate
    sees it as soon as it passes ``yaw_hold_deg``."""
    def look(t):
        return t < (t0 + lag)
    return look


def no_face(_t):
    """MODEL C -- no clean face row at all: the arm is across his face, or
    the detector missed. NO OPINION, which ``yaw_required`` decides."""
    return None


LOOSE = None            # filled by the test module


def sweep(fam, fps, thresholds, phases=16, noise=0.0, seed=0, look=None):
    """(sequences, fired, grabbed, per-family fires). One sequence = one
    family at one sub-frame offset. The attacker's own shape."""
    n = fired = grabbed = 0
    per = {}
    for name, (scen, eye) in sorted(fam.items()):
        for ph in range(phases):
            evs = run_look(scen, fps, ph, phases, thresholds=thresholds,
                           noise=noise, seed=seed, eye_fn=eye,
                           look_fn=look(name, scen) if callable(look) else None)
            n += 1
            kinds = [e.kind for e in evs]
            if "grab" in kinds:
                grabbed += 1
            if "throw" in kinds:
                fired += 1
                per[name] = per.get(name, 0) + 1
    return n, fired, grabbed, per


def line(tag, n, fired):
    p, lo, hi = wilson(fired, n)
    return ("%-52s %5d of %5d   P=%.4f   95%% CI [%.4f, %.4f]"
            % (tag, fired, n, p, lo, hi))


def throw_grid(amp="long"):
    return atkgrid.throws(amp)
