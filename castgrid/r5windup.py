"""ROUND 5's WIND-UP grid: the throw that pulls back, and the carries that
also do.

WHAT THIS IS FOR. Asked directly on 2026-09-05 -- "when you throw, does your
hand pull back a little first, before it swings?" -- HE ANSWERED YES. That
span is absent from every throw model in this lane (reach, close, hold
still, swing), which is why it had to be asked rather than measured, and it
is the one signal a carry cannot fake at any speed: carrying something away
never reverses before it leaves.

AND HERE IS THE TRAP THIS FILE EXISTS TO NOT FALL INTO. A GRID THAT INVENTS
A WIND-UP AND THEN DETECTS IT HAS PROVED NOTHING. So:

  * THE WIND-UP IS A PARAMETER WITH A RANGE, never one blessed shape. Both
    its amplitude and its duration are swept, INCLUDING values so small the
    detector should miss them (0, 5, 10 mm) and values past anything
    natural (120 mm). At amplitude 0 the family is ``atkgrid``'s own
    ``throw_slow`` / ``throw_fast`` span for span, and ``test_r5_windup.py``
    asserts that it reproduces them event for event -- so the sweep has a
    fixed point that is the preserved baseline rather than a new model.
  * THE SWING KEEPS ITS SPEED, not its duration. A wind-up adds distance to
    the swing; holding the duration fixed would make the hand faster and
    the wind-up would buy recall through the SPEED bar instead of on its
    own merits. The swing's duration is scaled by the extra distance, so
    the throw is the same throw with a run-up.
  * IT PULLS BACK THREE WAYS. Laterally, opposite the throw (the pure
    case); TOWARD HIM in depth (which is what a real wind-up often is);
    and a blend. THE DEPTH ONE IS A FINDING ON ITS OWN, and not the one
    expected: it does NOT trip ``reach_exit`` -- the carry survives 120 mm
    of it and still throws (measured, recall 1.0000) -- but the detector
    reads it as 0.000 u AT EVERY AMPLITUDE, because a retraction along the
    lens axis produces almost no displacement in the image plane. If that
    is the wind-up he makes, the signal is not weak, it is ABSENT, and any
    bar above 0.02 refuses every throw he makes.
  * EVERY NON-GESTURE FAMILY IS RE-EXAMINED FOR MOTIONS THAT ALSO RETRACT.
    ``retraction_grid()`` below is 90 families invented adversarially for
    exactly that: pulling a mug toward himself before lifting it away,
    drawing back to get a run at something heavy, a hand that hesitates,
    settling a grip, handing a thing over, and a slow retract-and-leave.
    They fire, and they measure the SAME retraction his throw does at the
    same amplitude -- MEASURED at 40 mm: his throw 0.218 u, a carry that
    pulls back first 0.239 u. That is the whole finding.

THE AMPLITUDE AND DURATION OF HIS OWN WIND-UP ARE UNKNOWN. He said it
exists; he did not measure it. ``scripts/gesture_selfcheck.py --windup`` is
the numbers-only instrument that lets him find out, and until he runs it
every recall figure here is recall AT AN ASSUMED AMPLITUDE.

NO CAMERA, NO CAPTURE DEVICE, NO FRAME, NO RUSTDESK SESSION, and nothing
touches his desktop. Every hand is a synthetic 21-point row projected
in-process through his LifeCam constants. ``tests/atkgrid.py`` and
``tests/atkharness.py`` are the attacker's files and are NOT edited: this
module composes new families beside them.
"""
from __future__ import annotations

import math

import numpy as np

from jarvis.gesture import NO_HEAD, CastGesture, CastThresholds, observe_hand
from tests.atkharness import (EYE_PX, W, H, Clock, hand3d, project, _curls,
                              seg, wilson)
from tests import atkgrid
from tests.atkgrid import GRAB_Z, PITCH, SIDES, _approach

# ---------------------------------------------------------- the sweeps
# MILLIMETRES OF RETRACTION. 0 is the preserved family exactly; 5 and 10 are
# deliberately below anything a detector should see over landmark noise; 120
# is past any natural wind-up and is in the sweep so the top of the curve is
# measured rather than assumed.
AMPS_MM = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0)
# SECONDS THE RETRACTION TAKES. At 7.5 fps, 0.10 s is under one frame -- a
# wind-up the sampler can miss entirely -- and 0.35 s is three frames.
DURS_S = (0.10, 0.18, 0.25, 0.35)
# Which way it pulls back.
WAYS = ("lateral", "depth", "blend")

# The bars swept over the recorded measurements. Nothing is re-run per bar:
# each sequence's wind-up is measured once and every bar is read off it.
BARS_U = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)

# HOW LONG THE HAND IS HELD STILL BEFORE THE RETRACTION IN THE ADVERSARIAL
# FAMILIES, and it is a MEASURED correction rather than a taste. atkgrid's
# own ``_approach`` holds for 0.45 s, and the grab needs dwell_frames + the
# latch frame -- 0.53 s at 7.5 fps -- so with a 0.45 s hold the anchor
# latches PART WAY INTO the retraction and the wind-up reads the remainder
# only (measured: 0.03 u where 0.18 u was injected). 0.95 s clears it at
# every rate in the band, so these families measure the retraction they
# actually contain.
HOLD_S = 0.95


def _back_point(start, end, amp_mm, way):
    """Where the hand pulls back TO, given where it is and where it goes."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    span = math.hypot(dx, dy)
    if span < 1e-9 or amp_mm <= 0.0:
        return tuple(start)
    ux, uy = dx / span, dy / span
    if way == "lateral":
        return (start[0] - amp_mm * ux, start[1] - amp_mm * uy, start[2])
    if way == "depth":
        # Straight back toward him: the hand comes AWAY from the lens.
        return (start[0], start[1], start[2] + amp_mm)
    # blend: half of each, so the total retraction is still amp_mm
    h = amp_mm / math.sqrt(2.0)
    return (start[0] - h * ux, start[1] - h * uy, start[2] + h)


def throw_windup(end, *, swing=0.8, pitch=PITCH, amp_mm=0.0, wu_s=0.0,
                 way="lateral", slow=True):
    """HIS throw, with a wind-up of ``amp_mm`` taking ``wu_s``.

    Span for span ``atkgrid.throw_slow`` (or ``throw_fast``) with ONE span
    inserted between the hold and the swing. At ``amp_mm=0`` or ``wu_s=0``
    the inserted span has zero duration and ``seg`` skips it, so the family
    IS the preserved one -- which ``test_r5_windup.py`` asserts event for
    event rather than assuming.
    """
    far = tuple(1.35 * x for x in end)
    if slow:
        pre = [(1.2, (60., 90., 600.), (10., 10., 430.), 0.10, 0.06,
                pitch - 5, 0, 0),
               (0.3, (10., 10., 430.), (10., 10., 425.), 0.06, 1.00,
                pitch, 0, 0),
               (1.5, (10., 10., 425.), (15., 5., 425.), 1.00, 1.00,
                pitch, 0, 0)]
        post = [(0.6, end, far, 1.00, 0.06, pitch, 0, 0),
                (1.5, far, far, 0.06, 0.06, pitch, 0, 0)]
        base_dur, swing_s = 1.2 + 0.3 + 1.5, swing
        tail_dur = 0.6 + 1.5
    else:
        pre = [(0.55, (140., 190., 780.), (10., 10., 430.), 0.10, 0.06,
                pitch - 5, 0, 0),
               (0.20, (10., 10., 430.), (10., 10., 425.), 0.06, 1.00,
                pitch, 0, 0),
               (0.50, (10., 10., 425.), (15., 5., 425.), 1.00, 1.00,
                pitch, 0, 0)]
        post = [(0.35, end, tuple(1.35 * x for x in end), 1.00, 0.06,
                 pitch, 0, 0)]
        base_dur, swing_s = 0.55 + 0.20 + 0.50, 0.40
        # ``throw_fast``'s own duration runs 0.5 s past its last span, so
        # the sequence keeps sampling the held end pose. Kept exactly.
        tail_dur = 0.35 + 0.5
    anchor = (15., 5., 425.)
    back = _back_point(anchor, end, float(amp_mm), way)
    wu = float(wu_s) if (amp_mm > 0.0 and wu_s > 0.0) else 0.0
    # THE SWING KEEPS ITS SPEED. The extra distance buys extra time, so the
    # wind-up cannot smuggle recall in through the speed bar.
    reach_mm = math.hypot(end[0] - anchor[0], end[1] - anchor[1])
    back_mm = math.hypot(end[0] - back[0], end[1] - back[1])
    scaled = swing_s * (back_mm / reach_mm if reach_mm > 1e-9 else 1.0)
    spans = pre + [(wu, anchor, back, 1.00, 1.00, pitch, 0, 0),
                   (scaled, back, end, 1.00, 1.00, pitch, 0, 0)] + post
    dur = base_dur + wu + scaled + tail_dur
    return (dur, lambda t: seg(t, spans))


def throw_family(side, *, swing_mm=310.0, z=440.0, amp_mm=0.0, wu_s=0.25,
                 way="lateral", slow=True, fingers_down=False):
    end = (SIDES[side] * float(swing_mm), -10., float(z))
    scen = throw_windup(end, amp_mm=amp_mm, wu_s=wu_s, way=way, slow=slow)
    return atkgrid.fingers_down(scen) if fingers_down else scen


# ================================================= the adversarial carries
# EVERY ONE OF THESE CONTAINS A RETRACTION BEFORE THE HAND LEAVES, and every
# one of them is a NON-gesture: he is moving a thing, not casting a screen.
# They are invented deliberately and adversarially, which is the whole point
# -- a wind-up detector that has only ever been shown throws with wind-ups
# and carries without them has measured nothing.
def pull_then_carry(side, amp_mm, traverse, end_z=500.0, pull_s=0.25):
    """PULLING A MUG TOWARD HIMSELF BEFORE LIFTING IT AWAY. The commonest
    natural retraction at a desk: the mug is behind the keyboard, he draws
    it back to clear the edge and then takes it off to one side."""
    s = SIDES[side]
    a = (5., 8., GRAB_Z)
    back = (a[0] - s * amp_mm, a[1], a[2])
    end = (s * 520., 20., end_z)
    return (0.75 + HOLD_S + pull_s + traverse + 0.6, lambda t: seg(t, _approach(hold=HOLD_S) + [
        (pull_s, a, back, 1.00, 1.00, PITCH, 0, 0),
        (traverse, back, end, 1.00, 1.00, PITCH, 0, 0),
        (0.6, end, end, 1.00, 1.00, PITCH, 0, 0)]))


def runup_shove(side, amp_mm, traverse, pull_s=0.18):
    """DRAWING BACK TO GET A RUN AT SOMETHING HEAVY, then shoving it across
    the desk and out of the picture. The retraction is fast, because he is
    loading the shove -- which is exactly what a throw's wind-up is."""
    s = SIDES[side]
    a = (5., 8., GRAB_Z)
    back = (a[0] - s * amp_mm, a[1] + 6., a[2] + 10.)
    end = (s * 500., 25., GRAB_Z + 40.)
    return (0.75 + HOLD_S + pull_s + traverse + 0.7, lambda t: seg(t, _approach(
        hold=HOLD_S) + [
        (pull_s, a, back, 1.00, 1.00, PITCH, 0, 0),
        (traverse, back, end, 1.00, 1.00, PITCH, 0, 0),
        (0.7, end, end, 1.00, 1.00, PITCH, 0, 0)]))


def hesitate_carry(side, amp_mm, traverse, out=0.45):
    """A HAND THAT HESITATES. He starts to take it away, thinks better of
    it, comes back a little, then goes. The retraction is in the MIDDLE of
    the carry, which is where a wind-up detector that measures a prefix
    minimum is weakest."""
    s = SIDES[side]
    a = (5., 8., GRAB_Z)
    mid = (s * 120., 8., GRAB_Z)
    back = (mid[0] - s * amp_mm, mid[1], mid[2])
    end = (s * 520., 18., GRAB_Z + 60.)
    return (0.75 + HOLD_S + out + 0.22 + traverse + 0.6, lambda t: seg(t, _approach(hold=HOLD_S)
            + [(out, a, mid, 1.00, 1.00, PITCH, 0, 0),
               (0.22, mid, back, 1.00, 1.00, PITCH, 0, 0),
               (traverse, back, end, 1.00, 1.00, PITCH, 0, 0),
               (0.6, end, end, 1.00, 1.00, PITCH, 0, 0)]))


def rock_grip(side, amp_mm, traverse):
    """SETTLING A GRIP: two small back-and-forth rocks before the carry.
    Twice the retraction, half the amplitude each time."""
    s = SIDES[side]
    a = (5., 8., GRAB_Z)
    b1 = (a[0] - s * amp_mm, a[1], a[2])
    b2 = (a[0] - s * amp_mm * 0.6, a[1] + 4., a[2])
    end = (s * 510., 20., GRAB_Z + 30.)
    return (0.75 + HOLD_S + 0.34 + traverse + 0.6, lambda t: seg(t, _approach(hold=HOLD_S) + [
        (0.14, a, b1, 1.00, 1.00, PITCH, 0, 0),
        (0.10, b1, a, 1.00, 1.00, PITCH, 0, 0),
        (0.10, a, b2, 1.00, 1.00, PITCH, 0, 0),
        (traverse, b2, end, 1.00, 1.00, PITCH, 0, 0),
        (0.6, end, end, 1.00, 1.00, PITCH, 0, 0)]))


def pull_then_hand_over(side, amp_mm, swing=0.5):
    """Drawing it back and HANDING IT TO SOMEBODY. ``atkgrid``'s own
    hand-over already fires by every signal the rig has; this is the same
    motion with the retraction he says his throw carries."""
    s = SIDES[side]
    a = (5., 8., GRAB_Z)
    back = (a[0] - s * amp_mm, a[1], a[2])
    end = (s * 235., -30., GRAB_Z + 60.)
    return (0.75 + HOLD_S + 0.22 + swing + 0.9, lambda t: seg(t, _approach(hold=HOLD_S)
            + [(0.22, a, back, 1.00, 1.00, PITCH, 0, 0),
               (swing, back, end, 1.00, 1.00, PITCH, 0, 0),
               (0.35, end, end, 1.00, 0.06, PITCH, 0, 0),
               (0.55, end, end, 0.06, 0.06, PITCH, 0, 0)]))


def retract_then_exit(side, amp_mm, out=1.6):
    """The slow version: pull back, then simply leave the picture. It
    should not fire on speed, and it is here so a bar that lets it through
    is visible rather than assumed away."""
    s = SIDES[side]
    a = (5., 8., GRAB_Z)
    back = (a[0] - s * amp_mm, a[1], a[2])
    gone = (s * 520., 30., GRAB_Z + 40.)
    return (0.75 + HOLD_S + 0.30 + out + 0.9, lambda t: seg(t, _approach(hold=HOLD_S)
            + [(0.30, a, back, 1.00, 1.00, PITCH, 0, 0),
               (out, back, gone, 1.00, 1.00, PITCH, 0, 0),
               (0.9, gone, gone, 1.00, 1.00, PITCH, 0, 0)]))


def retraction_grid():
    """name -> (scenario, eye_fn|None). Every entry is a NON-gesture that
    CONTAINS A RETRACTION before the hand leaves."""
    fam = {}
    for side in SIDES:
        # THE ADVERSARY'S AMPLITUDES MUST REACH AS FAR AS THE THROW'S, or
        # the comparison is rigged. MEASURED first at 20/40/70 mm, where a
        # bar of 0.60 u looked like it separated -- it did not, it had simply
        # run past the biggest retraction the grid contained. 130 mm is a mug
        # drawn from behind the keyboard to the desk edge, which is an
        # ordinary thing to do and is bigger than any wind-up he is likely
        # to make.
        for amp in (20., 40., 70., 100., 130.):
            for tr in (1.6, 1.2, 0.9):
                fam["pull-carry/%s/%.0fmm/%.1fs" % (side, amp, tr)] = (
                    pull_then_carry(side, amp, tr), None)
            for tr in (1.2, 0.9):
                fam["runup-shove/%s/%.0fmm/%.1fs" % (side, amp, tr)] = (
                    runup_shove(side, amp, tr), None)
            fam["hesitate/%s/%.0fmm" % (side, amp)] = (
                hesitate_carry(side, amp, 1.2), None)
            fam["rock-grip/%s/%.0fmm" % (side, amp)] = (
                rock_grip(side, amp, 1.2), None)
            fam["pull-handover/%s/%.0fmm" % (side, amp)] = (
                pull_then_hand_over(side, amp), None)
            fam["retract-exit/%s/%.0fmm" % (side, amp)] = (
                retract_then_exit(side, amp), None)
    return fam


# ------------------------------------------------------------- the runner
def run(scen, fps, phase, phases, thresholds=None, noise=0.0, seed=0,
        eye_fn=None, left=False, look=None):
    """ONE sequence. Returns the throw events, which is all any caller
    here needs -- each carries ``windup_u`` and ``fling_us``.

    Identical in shape to ``castgrid.r3run.run_look`` and to the
    attacker's ``run_seq``; ``test_r5_windup.py`` asserts it reproduces
    ``run_seq`` event for event with no head model.
    """
    dur, traj = scen
    clk = Clock()
    rng = np.random.default_rng(seed * 1009 + phase)
    m = CastGesture(thresholds or CastThresholds(), (W, H),
                    now=clk, preview_fps=fps)
    t = phase * (1.0 / fps) / phases
    i = 0
    out = []
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
        e = m.update(hands, eye, i, looking=NO_HEAD if look is None else look)
        if e is not None:
            out.append(e)
        step = 1.0 / fps
        t += step
        clk.t += step
        i += 1
    return out


def wind_of(events, want=None):
    """The wind-up measured on the THROW events of one sequence, and which
    way they went. ``(fired, best_windup_u, wrong_side)``.

    ONE PASS, EVERY BAR. A sequence's wind-up is measured once here and
    every bar in ``BARS_U`` is read off it afterwards, so the sweep costs
    one grid pass rather than one per bar -- and every bar is measured on
    exactly the same sequences with exactly the same seeds.
    """
    thr = [e for e in events if e.kind == "throw"]
    if not thr:
        return False, 0.0, False
    best = max(float(e.windup_u) for e in thr)
    wrong = bool(want) and any(e.sector != want for e in thr)
    return True, best, wrong


def sweep_fires(fam, fps, thresholds=None, phases=16, seed=0, look=None):
    """(n, [(name, windup_u)]) over a family map. Every entry is a fire."""
    n = 0
    fires = []
    for name, (scen, eye) in sorted(fam.items()):
        for ph in range(phases):
            evs = run(scen, fps, ph, phases, thresholds=thresholds,
                      eye_fn=eye, seed=seed, look=look)
            n += 1
            fired, wind, _w = wind_of(evs)
            if fired:
                fires.append((name, wind))
    return n, fires


def rate_at(n, fires, bar):
    """P(fire) once a wind-up bar of ``bar`` is applied, with its 95%
    Wilson interval. Read off the recorded measurements; nothing is re-run,
    so every bar shares the sequences and the seeds."""
    k = sum(1 for _name, w in fires if w >= bar)
    return wilson(k, n)


def recall_pass(*, amps=AMPS_MM, wu_s=0.25, way="lateral", swings=(290., 310.),
                depths=(430., 450.), rates=(6.0, 7.5), noises=(0.0, 3.0),
                phases=8, sides=("his-left", "his-right"),
                kinds=((True, False), (True, True)), thresholds=None):
    """recall as a function of WIND-UP AMPLITUDE.

    ``{amp_mm: (n, [windup_u of each fire], wrong_side)}``. The bar sweep is
    read off the recorded wind-ups afterwards, so this is one pass per
    amplitude and not one per (amplitude, bar).
    """
    out = {}
    for amp in amps:
        n = wrong = 0
        fires = []
        for side in sides:
            want = "left" if side == "his-left" else "right"
            for swing in swings:
                for z in depths:
                    for slow, fdown in kinds:
                        scen = throw_family(side, swing_mm=swing, z=z,
                                            amp_mm=amp, wu_s=wu_s, way=way,
                                            slow=slow, fingers_down=fdown)
                        for fps in rates:
                            for px in noises:
                                for ph in range(phases):
                                    evs = run(scen, fps, ph, phases,
                                              thresholds=thresholds,
                                              noise=px, seed=11, look=True)
                                    n += 1
                                    f, w, bad = wind_of(evs, want)
                                    if f:
                                        fires.append(w)
                                    wrong += 1 if bad else 0
        out[amp] = (n, fires, wrong)
    return out


def line(tag, n, k):
    p, lo, hi = wilson(k, n)
    return ("%-46s %5d of %5d   P=%.4f   95%% CI [%.4f, %.4f]"
            % (tag, k, n, p, lo, hi))


__all__ = ["AMPS_MM", "BARS_U", "DURS_S", "WAYS", "hesitate_carry", "line",
           "pull_then_carry", "pull_then_hand_over", "rate_at", "recall_pass",
           "retract_then_exit", "retraction_grid", "rock_grip", "run",
           "runup_shove", "sweep_fires", "throw_family", "throw_windup",
           "wind_of"]
