"""The ATTACKER's non-gesture grid, weighted to HIS desk. Numbers only.

The camera sits ON TOP OF THE SPARK'S OWN MONITOR, the right screen,
looking across at him.  Two things follow and both were MEASURED here
before a single family was written:

  * DEPTH.  reach_min 2.35 is met at z <= ~455 mm at his working pitch, so
    the right screen, its bezel and everything beside it are the only
    things in grab range.  That is where his hands are all day.
  * POSE.   The closed scalar is not pose-free.  At pitch 35 (a hand
    reaching AT the lens) a curl of 0.40 already reads C = 0.586, and at
    pitch 55 a curl of 0.15 reads 0.703 -- both under closed_max 0.70.  A
    RELAXED hand coming at that monitor reads as a fist.  So this grid does
    not only test deliberate grips; it tests the natural hand.

  * FRAME.  A palm at z 430 leaves the bottom of the picture at y = 40 mm,
    so the desk surface and the keyboard are largely BELOW the field of
    view at close range.  Families are placed where the lens can see.

Every hand is a synthetic 21-point row projected in-process.  No camera,
no capture device, no frame.
"""
from __future__ import annotations

import math

from tests.atkharness import EYE_PX, seg

GRAB_Z = 425.0
REST = (150., 105., 720.)
PITCH = 35.0

CARRY_MM = 520.0
SIDES = {"his-left": 1.0, "his-right": -1.0}
TRAVERSES = (3.2, 2.4, 2.0, 1.6, 1.2, 0.9, 0.7)
END_Z = (425.0, 500.0, 600.0, 700.0)


def _approach(z=GRAB_Z, dwell=0.55, close=0.20, hold=0.45, grip=1.00,
              pitch=PITCH):
    return [(dwell, REST, (0., 10., z), 0.10, 0.06, pitch - 5, 0, 0),
            (close, (0., 10., z), (0., 10., z), 0.06, grip, pitch, 0, 0),
            (hold, (0., 10., z), (5., 8., z), grip, grip, pitch, 0, 0)]


# ------------------------------------------------------- the named attack
def carry_out(side, traverse, end_z, grip=1.00, pitch=PITCH, z0=GRAB_Z):
    """Reach at the right screen, close on a mug, carry it laterally out of
    the picture.  No fling, no release, no follow-through."""
    s = SIDES[side]
    end = (s * CARRY_MM, 20., end_z)
    return (1.20 + traverse + 0.6, lambda t: seg(t, _approach(
        z=z0, grip=grip, pitch=pitch) + [
        (traverse, (5., 8., z0), end, grip, grip, pitch, 0, 0),
        (0.6, end, end, grip, grip, pitch, 0, 0)]))


def relaxed_carry(side, traverse, pitch=48.0, grip=0.42):
    """THE POSE ATTACK.  The same reach-and-carry done with a RELAXED hand
    that is never deliberately gripped -- reaching past that monitor for a
    mug and taking it away.  At this pitch a curl of 0.42 reads closed."""
    return carry_out(side, traverse, 460., grip=grip, pitch=pitch, z0=385.)


def set_down_aside(side, move, dist=210.0, grip=1.00):
    s = SIDES[side]
    end = (s * dist, 35., GRAB_Z + 20.)
    return (1.20 + move + 0.9, lambda t: seg(t, _approach(grip=grip) + [
        (move, (5., 8., GRAB_Z), end, grip, grip, PITCH, 0, 0),
        (0.35, end, end, grip, 0.06, PITCH, 0, 0),
        (0.55, end, end, 0.06, 0.06, PITCH, 0, 0)]))


def hand_to_someone(side, swing):
    """Handing a thing to someone beside him.  This IS a throw by every
    signal the rig has; it is reported as its own family, never hidden."""
    s = SIDES[side]
    end = (s * 235., -30., GRAB_Z + 60.)
    return (1.15 + swing + 0.9, lambda t: seg(t, _approach(hold=0.40) + [
        (swing, (5., 8., GRAB_Z), end, 1.00, 1.00, PITCH, 0, 0),
        (0.35, end, end, 1.00, 0.06, PITCH, 0, 0),
        (0.55, end, end, 0.06, 0.06, PITCH, 0, 0)]))


def jerk_then_carry(side, gap):
    """THE FLING-WINDOW ATTACK, and it is about TIME, not distance.

    Carry the thing away SLOWLY (154 mm/s throughout, a fifth of a throw),
    but put ONE quick correction in the middle of it -- shifting the grip on
    a mug, catching it as it tips.  The fling test asks only whether the
    hand was at throw speed within ``fling_window_s`` of the LAST FRAME THAT
    SAW IT, so the question is how long after that correction the hand
    leaves the picture.  ``gap`` is that delay.  Everything else about the
    motion is identical across the row."""
    s = SIDES[side]
    a = (s * 120., 4., GRAB_Z)          # slow carry to here: 154 mm/s
    b = (s * 176., 0., GRAB_Z)          # THE CORRECTION: 56 mm in 0.14 s
    c = (s * (176. + 154. * (gap + 0.45)), 6., GRAB_Z)   # slow again
    return (1.05 + 0.78 + 0.14 + gap + 0.45 + 0.7,
            lambda t: seg(t, _approach(hold=0.30) + [
                (0.78, (5., 8., GRAB_Z), a, 1.00, 1.00, PITCH, 0, 0),
                (0.14, a, b, 1.00, 1.00, PITCH, 0, 0),
                (gap + 0.45, b, c, 1.00, 1.00, PITCH, 0, 0),
                (0.7, c, c, 1.00, 1.00, PITCH, 0, 0)]))


def toward_lens_sweep(side, traverse):
    """UNIT INFLATION.  The hand-unit is latched at the grab; a hand that
    then comes CLOSER to the lens covers more image units per millimetre.
    Sliding a finger down the right screen does exactly that."""
    s = SIDES[side]
    end = (s * 300., 0., 300.)
    return (1.20 + traverse + 0.6, lambda t: seg(t, _approach() + [
        (traverse, (5., 8., GRAB_Z), end, 1.00, 1.00, PITCH, 0, 0),
        (0.6, end, end, 1.00, 1.00, PITCH, 0, 0)]))


def bezel_adjust(push, z_end=360.0):
    """Grabbing the RIGHT monitor's own bezel and tilting it."""
    mid = (70., -50., z_end)
    return (1.10 + push + 1.0, lambda t: seg(t, _approach(z=395., hold=0.35) + [
        (push, (5., 8., 395.), mid, 1.00, 1.00, PITCH + 5, 8, 8),
        (0.4, mid, mid, 1.00, 0.10, PITCH, 0, 0),
        (0.6, mid, REST, 0.10, 0.10, PITCH - 10, 0, 0)]))


def mug_to_mouth(lift):
    lip = (95., -160., 600.)
    return (1.10 + lift + 2.4, lambda t: seg(t, _approach(hold=0.35) + [
        (lift, (5., 8., GRAB_Z), lip, 1.00, 1.00, 15, 0, 10),
        (1.2, lip, (100., -165., 605.), 1.00, 1.00, 20, 5, 15),
        (lift, (100., -165., 605.), (10., 10., GRAB_Z), 1.00, 1.00, 25, 0, 0),
        (0.4, (10., 10., GRAB_Z), (10., 10., GRAB_Z), 1.00, 0.10, PITCH, 0, 0)
    ]))


def keyboard_reach(z, curl, y=None):
    """Hands at the keyboard, half-curled, as far up as the lens can see."""
    yy = 0.20 * z if y is None else y
    return (4.6, lambda t: seg(t, [
        (0.7, REST, (185., yy, z), 0.12, curl, -18, 0, 0),
        (1.6, (185., yy, z), (-60., yy + 8., z + 15.),
         curl, curl + 0.08, -20, 5, 0),
        (1.6, (-60., yy + 8., z + 15.), (185., yy, z),
         curl + 0.08, curl, -18, 0, 0),
        (0.7, (185., yy, z), REST, curl, 0.12, -18, 0, 0)]))


def scratch(where):
    tgt, pit, cur = where
    off = tuple(x + d for x, d in zip(tgt, (18., -14., 6.)))
    return (3.6, lambda t: seg(t, [
        (0.8, REST, tgt, 0.35, cur, pit, 0, 0),
        (1.0, tgt, off, cur, cur + 0.12, pit, 8, 12),
        (1.0, off, tgt, cur + 0.12, cur, pit, -8, -12),
        (0.8, tgt, REST, cur, 0.35, pit, 0, 0)]))


def stretch(kind):
    """Arms up and out at the end of an hour.  Fast, and it leaves frame."""
    end, pit, cur = kind
    return (3.5, lambda t: seg(t, [
        (0.5, REST, (60., 40., 470.), 0.20, 0.45, 10, 0, 0),
        (0.30, (60., 40., 470.), (65., 20., 450.), 0.45, 0.50, pit, 0, 0),
        (0.55, (65., 20., 450.), end, 0.50, cur, pit, 0, 0),
        (0.9, end, end, cur, cur, pit, 10, 10),
        (0.6, end, REST, cur, 0.20, 10, 0, 0)]))


def point_while_talking(freq, amp, z, curls=(0.05, 0.72, 0.78, 0.74)):
    return (5.5, lambda t: (
        (amp * math.sin(freq * t), -10. + 45. * math.cos(freq * t * 0.7), z),
        curls, 22, 22. * math.sin(freq * t), 0))


def pen_gesture(freq, amp, z):
    """Talking with a pen in his hand: index and thumb pinched, the other
    three curled hard.  MEASURED C = 0.53-0.66 -- a fist as far as the rig
    is concerned -- at reach depth, moving."""
    return (5.5, lambda t: seg(t, [
        (0.7, REST, (0., 0., z), (0.15, 0.15, 0.15, 0.15),
         (0.30, 0.90, 0.95, 0.92), 30, 0, 0),
        (3.4, (0., 0., z), (0., 0., z), (0.30, 0.90, 0.95, 0.92),
         (0.30, 0.90, 0.95, 0.92), 30, 0, 0),
        (1.4, (0., 0., z), (amp, -20., z + 30.), (0.30, 0.90, 0.95, 0.92),
         (0.30, 0.90, 0.95, 0.92), 30, 0, 0)]))


def leave_chair(side):
    """He pushes off the desk edge and stands.  THE FACE GOES FIRST -- and
    a reach of 0.0 is 'no opinion', which switches the withdrawal cancel
    off for the rest of the carry."""
    s = SIDES[side]
    push = (s * 120., 60., 430.)
    end = (s * 470., -300., 500.)
    return (3.4, lambda t: seg(t, [
        (0.6, REST, push, 0.15, 0.60, 10, 0, 0),
        (0.20, push, push, 0.60, 1.00, 12, 0, 0),
        (0.55, push, (s * 130., 55., 428.), 1.00, 1.00, 12, 0, 0),
        (0.75, (s * 130., 55., 428.), end, 1.00, 1.00, 5, 0, 0),
        (1.3, end, end, 1.00, 1.00, 0, 0, 0)]))


def exit_and_return(side, away=1.1, out=0.9):
    """The hand leaves the picture and comes back, still closed."""
    s = SIDES[side]
    gone = (s * 520., 30., GRAB_Z + 40.)
    return (1.10 + out + away + 0.9 + 0.8,
            lambda t: seg(t, _approach(hold=0.35) + [
                (out, (5., 8., GRAB_Z), gone, 1.00, 1.00, PITCH, 0, 0),
                (away, gone, gone, 1.00, 1.00, PITCH, 0, 0),
                (0.9, gone, (10., 15., GRAB_Z), 1.00, 1.00, PITCH, 0, 0),
                (0.8, (10., 15., GRAB_Z), REST, 1.00, 0.15, PITCH, 0, 0)]))


def crumb_sweep(side, speed, z=440.0):
    """Brushing crumbs off the desk beside that monitor: a half-closed hand
    moving fast and low, never a deliberate grab."""
    s = SIDES[side]
    y = 0.06 * z
    return (2.4, lambda t: seg(t, [
        (0.5, REST, (-s * 150., y, z), 0.15, 0.55, 25, 0, 0),
        (speed, (-s * 150., y, z), (s * 430., y + 8., z), 0.55, 0.60, 25, 0, 0),
        (0.9, (s * 430., y + 8., z), REST, 0.60, 0.15, 25, 0, 0)]))


def put_back():
    return (3.4, lambda t: seg(t, _approach() + [
        (1.10, (5., 8., GRAB_Z), (45., 45., 442.), 1.00, 1.00, PITCH, 0, 0),
        (0.40, (45., 45., 442.), (45., 47., 442.), 1.00, 0.06, PITCH, 0, 0),
        (0.55, (45., 47., 442.), REST, 0.06, 0.06, PITCH, 0, 0)]))


def withdraw(side, pull=0.5):
    s = SIDES[side]
    end = (s * 210., 45., 640.)
    return (1.20 + pull + 0.9, lambda t: seg(t, _approach() + [
        (pull, (5., 8., GRAB_Z), end, 1.00, 1.00, PITCH, 0, 0),
        (0.35, end, end, 1.00, 0.06, PITCH, 0, 0),
        (0.55, end, end, 0.06, 0.06, PITCH, 0, 0)]))


# ---------------------------------------------------------------- the grid
def build():
    """name -> (scenario, eye_fn|None).  Every entry is a NON-gesture."""
    fam = {}
    for side in SIDES:
        for tr in TRAVERSES:
            for ez in END_Z:
                fam["carry-out/%s/%.1fs/%.0fmm" % (side, tr, ez)] = (
                    carry_out(side, tr, ez), None)
        for tr in (3.2, 2.4, 2.0, 1.6, 1.2, 0.9):
            fam["relaxed-carry/%s/%.1fs" % (side, tr)] = (
                relaxed_carry(side, tr), None)
        for mv in (1.2, 0.9, 0.7, 0.5):
            fam["set-down/%s/%.1fs" % (side, mv)] = (
                set_down_aside(side, mv), None)
        for sw in (0.35, 0.5, 0.7):
            fam["hand-over/%s/%.2fs" % (side, sw)] = (
                hand_to_someone(side, sw), None)
        for gap in (0.00, 0.15, 0.30, 0.45, 0.60, 0.90, 1.30):
            fam["jerk-carry/%s/gap%.2f" % (side, gap)] = (
                jerk_then_carry(side, gap), None)
        for tr in (2.4, 1.8, 1.2):
            fam["toward-lens/%s/%.1fs" % (side, tr)] = (
                toward_lens_sweep(side, tr), None)
        fam["leave-chair/%s" % side] = (leave_chair(side), None)
        for aw in (0.6, 1.1, 1.8):
            fam["exit-return/%s/%.1f" % (side, aw)] = (
                exit_and_return(side, aw), None)
        for out in (1.6, 2.4, 3.2):
            fam["exit-return-slow/%s/%.1f" % (side, out)] = (
                exit_and_return(side, 1.1, out=out), None)
        for sp in (0.35, 0.55, 0.85):
            fam["crumb-sweep/%s/%.2f" % (side, sp)] = (
                crumb_sweep(side, sp), None)
            fam["crumb-far/%s/%.2f" % (side, sp)] = (
                crumb_sweep(side, sp, z=500.), None)
        fam["withdraw/%s" % side] = (withdraw(side), None)
    for p in (0.5, 0.8, 1.2):
        fam["bezel/%.1fs" % p] = (bezel_adjust(p), None)
    for li in (0.6, 0.9, 1.3):
        fam["mug-to-mouth/%.1fs" % li] = (mug_to_mouth(li), None)
    for z, c in ((470., 0.36), (520., 0.44), (600., 0.52), (660., 0.40),
                 (445., 0.50), (500., 0.58)):
        fam["keyboard/%.0f" % z] = (keyboard_reach(z, c), None)
    for nm, w in (("head", ((60., -170., 590.), 8, 0.58)),
                  ("jaw", ((105., -60., 555.), 18, 0.66)),
                  ("neck", ((130., -25., 575.), 22, 0.62)),
                  ("eye-lean", ((70., -70., 455.), 30, 0.62)),
                  ("brow-lean", ((40., -100., 470.), 25, 0.68)),
                  ("nose-lean", ((90., -40., 440.), 35, 0.55))):
        fam["scratch/%s" % nm] = (scratch(w), None)
    for nm, k in (("up", ((80., -330., 520.), -25, 0.28)),
                  ("out-left", ((430., -180., 540.), -10, 0.22)),
                  ("out-right", ((-420., -170., 540.), -10, 0.22)),
                  ("yawn", ((150., -230., 470.), 5, 0.55))):
        fam["stretch/%s" % nm] = (stretch(k), None)
    for f, a, z in ((7.5, 175., 585.), (5.2, 240., 545.),
                    (9.8, 120., 620.), (6.4, 300., 500.),
                    (7.5, 150., 450.), (5.2, 190., 470.)):
        fam["point-talk/%.1f@%.0f" % (f, z)] = (
            point_while_talking(f, a, z), None)
    for f, a, z in ((5.0, 260., 440.), (7.0, 200., 465.), (6.0, 320., 425.)):
        fam["pen-talk/%.1f@%.0f" % (f, z)] = (pen_gesture(f, a, z), None)
    fam["put-back"] = (put_back(), None)

    def dark_after(t0):
        return lambda t: (0.0 if t >= t0 else EYE_PX)

    for side in SIDES:
        for tr in (3.2, 2.0, 1.2):
            fam["face-lost/%s/%.1fs" % (side, tr)] = (
                carry_out(side, tr, 500.), dark_after(1.35))
        for tr in (3.2, 2.0):
            fam["face-lost-relaxed/%s/%.1fs" % (side, tr)] = (
                relaxed_carry(side, tr), dark_after(1.35))
    return fam


# ------------------------------------------------- THE GESTURE ITSELF
# The other side of the ledger: a fix that kills the feature is also a
# failure.  Two speeds, both directions, both finger orientations.
def throw_fast(end, pitch=PITCH):
    """His deliberate gesture at natural speed: reach in open, close, hold,
    swing 400 ms, open while following through."""
    return (0.55 + 0.20 + 0.50 + 0.40 + 0.35 + 0.5, lambda t: seg(t, [
        (0.55, (140., 190., 780.), (10., 10., 430.), 0.10, 0.06,
         pitch - 5, 0, 0),
        (0.20, (10., 10., 430.), (10., 10., 425.), 0.06, 1.00, pitch, 0, 0),
        (0.50, (10., 10., 425.), (15., 5., 425.), 1.00, 1.00, pitch, 0, 0),
        (0.40, (15., 5., 425.), end, 1.00, 1.00, pitch, 0, 0),
        (0.35, end, tuple(1.35 * x for x in end), 1.00, 0.06, pitch, 0, 0)]))


def throw_slow(end, swing=0.8, pitch=PITCH):
    """The SAME gesture done deliberately slowly -- the family the fix's
    speed bar was placed to keep (its measured floor is 3.02 u/s)."""
    far = tuple(1.35 * x for x in end)
    return (1.2 + 0.3 + 1.5 + swing + 0.6 + 1.5, lambda t: seg(t, [
        (1.2, (60., 90., 600.), (10., 10., 430.), 0.10, 0.06, pitch - 5, 0, 0),
        (0.3, (10., 10., 430.), (10., 10., 425.), 0.06, 1.00, pitch, 0, 0),
        (1.5, (10., 10., 425.), (15., 5., 425.), 1.00, 1.00, pitch, 0, 0),
        (swing, (15., 5., 425.), end, 1.00, 1.00, pitch, 0, 0),
        (0.6, end, far, 1.00, 0.06, pitch, 0, 0),
        (1.5, far, far, 0.06, 0.06, pitch, 0, 0)]))


def fingers_down(scen):
    dur, traj = scen
    return (dur, lambda t: (lambda r: (r[0], r[1], r[2], r[3], r[4] + 180.))(
        traj(t)))


# Named for the side HE throws to. Image +x is his left.
# TWO AMPLITUDES, and the difference between them is a finding in itself.
# LONG is the swing the fixer's own pinned slow-gesture test uses (300-310
# mm); SHORT is the swing its own FAST gesture uses (250-260 mm).
LONG = {"his-left": (310., -10., 440.), "his-right": (-300., -10., 440.)}
SHORT = {"his-left": (260., -10., 450.), "his-right": (-250., -10., 450.)}


def throws(amp="long"):
    """name -> (scenario, wanted sector). The deliberate gesture, both
    speeds and both directions."""
    ends = LONG if amp == "long" else SHORT
    out = {}
    for side, end in ends.items():
        want = "left" if side == "his-left" else "right"
        out["throw-fast/%s" % side] = (throw_fast(end), want)
        out["throw-slow/%s" % side] = (throw_slow(end), want)
        out["throw-fast/%s/fingers-down" % side] = (
            fingers_down(throw_fast(end)), want)
        out["throw-slow/%s/fingers-down" % side] = (
            fingers_down(throw_slow(end)), want)
    return out
