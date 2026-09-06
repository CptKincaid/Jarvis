"""ROUND 4's recall surface, before and after. The whole grid, not a point.

WHAT THIS IS FOR. Round 3 measured recall at ONE point -- a 300-310 mm
swing ending at z = 440 -- and that point turned out to be the single most
favourable spot on a cliff round 3 itself created. One number from one
family is not a measurement of a surface, and this file exists so nobody
does it that way again.

HOW TO RUN IT. This directory is deliberately OUTSIDE pytest.ini's
``testpaths``, so ``pytest -q`` does not collect it: it is an instrument,
not a regression, and a full pass is 24192 sequences per checkout. From
the repo root:

    git show 53537c7:jarvis/gesture.py > /tmp/gesture_53537c7.py
    R2_GESTURE=/tmp/gesture_53537c7.py \
      ~/.local/bin/memcap timeout 1700 ~/vss_env/bin/python -m pytest -q \
        -p no:cacheprovider castgrid/test_r4_recall.py -s

Without ``R2_GESTURE`` the round-2 columns are reported ABSENT rather than
invented.

NO CAMERA, NO CAPTURE DEVICE, NO FRAME, NO RUSTDESK SESSION, and nothing
touches his desktop. Every hand is a synthetic 21-point row projected
in-process through his LifeCam constants.

THE HONEST LIMIT, unchanged from the attacker's own statement of it: the
hand-unit of 127 mm is standard adult anthropometry, GUESSED, not measured
on him, and the throw model is one attacker's reconstruction of his
gesture. 0.9094 means "this model of his throw fires on this model of his
hand at these poses". Only he can say whether his own throw is in here at
all -- and ``screens_status()["refused"]`` is now the instrument for that,
because a refused throw finally says which bar refused it.
"""
from __future__ import annotations

import os

import pytest

from dataclasses import replace

import jarvis.gesture as g4
from castgrid import r4recall as R

R2 = os.environ.get("R2_GESTURE", "")
# The value round 3 shipped, so the "before" is measured here rather than
# quoted -- the module now carries 2.45 and this file must still be able to
# show what 2.65 did.
ROUND3_SPEED = 2.65


@pytest.fixture(scope="module")
def base():
    mod = R.load_gesture(R2)
    if mod is None:
        pytest.skip("set R2_GESTURE to round 2's jarvis/gesture.py "
                    "(git show 53537c7:jarvis/gesture.py)")
    return mod


def report(tag, out):
    f, n, w = R.totals(out)
    print("\n===== %s =====" % tag)
    print("  %d of %d = %.4f     throws to the WRONG SIDE: %d" % (f, n, f/n, w))
    for side in ("his-left", "his-right"):
        print(R.render(out, side, label="\n  %s (recall by swing mm x end z mm)"
                       % side))
    short = [(round(p, 4), k) for p, k in R.worst(out, 6) if p < 1.0]
    print("\n  worst cells: %s" % (short or "none -- every cell 1.0000"))
    return f, n


def test_1_the_whole_surface_before_and_after(base):
    """THE HEADLINE. Same loop, same seeds, three checkouts."""
    print("\nswings %s mm" % (list(map(int, R.SWINGS)),))
    print("end depths %s mm" % (list(map(int, R.DEPTHS)),))
    print("fps %s   noise %s px   phases %d   both directions, both finger "
          "orientations" % (list(R.RATES), list(R.NOISES), R.PHASES))

    b = report("ROUND 2 (53537c7) -- the recall being recovered",
               R.surface(base))
    r3 = report("ROUND 3 (d4def0b) -- throw_speed_us 2.65",
                R.surface(g4, thresholds=replace(g4.CastThresholds(),
                                                 throw_speed_us=ROUND3_SPEED)))
    r4 = report("ROUND 4 -- throw_speed_us %.2f"
                % g4.CastThresholds().throw_speed_us, R.surface(g4))
    print("\n  round 2 %.4f   round 3 %.4f   round 4 %.4f"
          % (b[0]/b[1], r3[0]/r3[1], r4[0]/r4[1]))


def test_2_no_cell_is_below_round_two(base):
    """THE BAR THE BRIEF SET, cell by cell -- an average cannot hide a
    cliff, which is exactly how round 3's one favourable point hid one."""
    before = R.surface(base)
    after = R.surface(g4)
    worse = []
    for key in before:
        b = before[key][0] / before[key][1]
        a = after[key][0] / after[key][1]
        if a < b - 1e-9:
            worse.append((key, b, a))
    print("\n  cells below round 2: %d of %d" % (len(worse), len(before)))
    for key, b, a in worse:
        print("      %-9s swing %.0f z %.0f: %.4f -> %.4f"
              % (key[0], key[1], key[2], b, a))
    assert worse == []


def test_3_the_all_or_nothing_floor(base):
    """The number the brief names: the smallest swing that fires EVERY
    time, at each depth. Round 3 moved it from 280 mm to 295 mm."""
    before, after = R.surface(base), R.surface(g4)
    r3 = R.surface(g4, thresholds=replace(g4.CastThresholds(),
                                          throw_speed_us=ROUND3_SPEED))
    print("\n  smallest swing (mm) with recall 1.0000, his-left:")
    print("    z(mm)   " + "".join("%9.0f" % z for z in R.DEPTHS))
    for tag, out in (("round 2", before), ("round 3", r3), ("round 4", after)):
        row = ["    %-8s" % tag]
        for z in R.DEPTHS:
            f = R.floor_at(out, "his-left", z)
            row.append("%9s" % ("%.0f" % f if f else "none"))
        print("".join(row))


def test_4_the_bar_sits_inside_his_own_throw(base):
    """WHY IT CLIFFED, in one table, and it is the only table that
    matters: the speed the machine MEASURES on his own gesture, against
    the bar it is compared with. A bar inside this spread does not reject
    a mug, it rejects a shorter throw."""
    print("\n  peak u/s the machine measures on HIS OWN slow throw")
    print("  (7.5 fps, 0 px, his-left, 8 sub-frame phases):")
    print("    swing    " + "".join("%8.0f" % z for z in R.DEPTHS))
    lo = 9e9
    hi = 0.0
    for swing in R.SWINGS:
        row = ["    %5.0f    " % swing]
        for z in R.DEPTHS:
            v = R.peak_us(g4, "his-left", swing, z)
            lo, hi = min(lo, v), max(hi, v)
            row.append("%8.3f" % v)
        print("".join(row))
    print("\n  his throw spans %.3f - %.3f u/s." % (lo, hi))
    print("  round 3's bar was 2.65 -- INSIDE it. Round 4's is %.2f -- below"
          % g4.CastThresholds().throw_speed_us)
    print("  all of it. That is the whole of the cliff.")
    print("\n  and the unit itself, statically at pitch 35:")
    for z in (R.DEPTHS[0], R.DEPTHS[-1]):
        print("    z %.0f mm -> one hand-unit measures %.1f mm (documented "
              "as 127)" % (z, R.mm_per_unit(z)))
    print("  so no bar here may be quoted in mm/s; the recall surface and")
    print("  the false-fire grid are what defend the value.")
    assert lo > g4.CastThresholds().throw_speed_us, (lo, "the bar is inside "
                                                     "his own throw again")


def test_5_it_was_not_the_distance_rescale(base):
    """THE ABLATION, because the brief's own guess was that it was, and
    guessing at a cause is how a round wastes itself. One change put back
    at a time, on the fast slice."""
    sub = dict(rates=(7.5,), noises=(0.0,), phases=8,
               kinds=((True, False),), sides=("his-left",))
    NOW = g4.CastThresholds()
    r3 = replace(NOW, throw_speed_us=ROUND3_SPEED)
    rows = (
        ("round 3 as it shipped", r3),
        ("...distance bars back to round 2",
         replace(r3, throw_release_u=1.00, throw_exit_u=0.25,
                 throw_lost_u=0.50, exit_step_u=0.35, anchor_drift_u=0.60,
                 assoc_max_u=2.50)),
        ("...closed_ratio_max off", replace(r3, closed_ratio_max=99.0)),
        ("...assoc_step_u off", replace(r3, assoc_step_u=99.0)),
        ("...fling window neutralised", replace(r3, fling_window_s=10.0)),
        ("...speed bar at 2.45 (round 4)", replace(r3, throw_speed_us=2.45)),
    )
    print("\n  his-left, slow throw, 7.5 fps, 0 px, 8 phases:")
    for tag, th in rows:
        out = R.surface(g4, thresholds=th, **sub)
        f, n, _w = R.totals(out)
        print("    %-34s %3d/%3d = %.4f" % (tag, f, n, f/n))
    print("\n  Only the speed bar moves it. The rescale, the fist scalar and")
    print("  the teleport backstop change the surface by nothing at all.")


def test_6_what_the_recall_recovery_cost(base):
    """The other side of the ledger, on the attacker's preserved grid so
    it is comparable with 0.2700 and 0.1995 rather than being a fresh
    number that agrees with itself."""
    from castgrid.r3run import line, sweep
    from tests import atkgrid
    fam = atkgrid.build()
    print("\n  P(fire | NOT a cast gesture), 172 families x 16 phases:")
    for tag, th in (("round 3 (2.65)",
                     replace(g4.CastThresholds(),
                             throw_speed_us=ROUND3_SPEED)),
                    ("round 4 (%.2f)" % g4.CastThresholds().throw_speed_us,
                     g4.CastThresholds())):
        for fps in (7.5, 6.0, 3.7):
            n, fired, _g, _p = sweep(fam, fps, th, phases=16)
            print("  " + line("%s  %.1f fps" % (tag, fps), n, fired))
    print("\n  round 2 on this same grid: 0.2700 / 0.2594 / 0.1257.")
    print("  THE BAR IS 0.005. It was 40x over and it is 47x over. The")
    print("  gesture ships enabled: False and this round does not change")
    print("  that; see jarvis/assistant_config.py.")
