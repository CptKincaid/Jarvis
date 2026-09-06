"""ROUND 3's numbers, on the ATTACKER's preserved grid and its seeds.

172 non-gesture families x 16 sub-frame phases = 2752 sequences, the same
families and the same seeds round 2 was measured on, so 0.2700 and the
numbers below are the same measurement of the same thing.

HOW TO RUN IT. This directory is deliberately OUTSIDE pytest.ini's
``testpaths``, so ``pytest -q`` does not collect it: it is an instrument,
not a regression, and it costs about a minute. From the repo root:

    ~/.local/bin/memcap timeout 1700 ~/vss_env/bin/python -m pytest -q \
        -p no:cacheprovider castgrid/test_r3_measure.py -s

``castgrid/test_atk_*.py`` are the attacker's own four files, unmodified,
and run the same way.

EVERY HAND IS SYNTHETIC. No camera, no capture device, no frame, no
RustDesk session, and nothing here touches his desktop.

THE HONEST LIMIT, carried forward from the attacker unchanged: every row
is projected through his LifeCam constants in process. The hand-unit of
127 mm is standard anthropometry, GUESSED, not measured on him, and the
family MIX is one attacker's judgement of plausible desk motion, not
sampled from his room. 0.19 means "about a fifth of the brisk
near-monitor motions we could invent still fire", NOT misfires per hour.
Only he can produce the latter.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from jarvis.gesture import CastThresholds
from castgrid.r3run import line, run_look, sweep
from tests.atkharness import wilson
from tests import atkgrid

PHASES = 16
NOW = CastThresholds()
RATES = (7.5, 6.0, 3.7)


@pytest.fixture(scope="module")
def fam():
    return atkgrid.build()


def test_1_false_fire_rate_at_every_rate(fam):
    """THE HEADLINE, and the yaw gate is INERT for it.

    The grid has no head in it -- the attacker's families model hands, not
    faces -- so ``looking`` is never supplied and the second signal
    abstains on every sequence. That is deliberate: it means this number
    measures the hand-track fixes ALONE, against a round-2 number measured
    the same way. What the yaw gate could add is bracketed in test 5, and
    the pessimistic end of that bracket is zero.
    """
    print("\n=========== P(fire | NOT a cast gesture), round 3 ===========")
    out = {}
    for fps in RATES:
        n, fired, grabbed, per = sweep(fam, fps, NOW, phases=PHASES)
        out[fps] = (n, fired, grabbed, per)
        print(line("%.1f fps" % fps, n, fired)
              + "   false grabs %d (%.4f)" % (grabbed, grabbed / n))
    print("\nROUND 2 (53537c7) on this same grid, this same runner:")
    print("  7.5 fps  743 of 2752  P=0.2700 CI [0.2537,0.2869]  grabs 1921")
    print("  6.0 fps  714 of 2752  P=0.2594 CI [0.2434,0.2762]  grabs 1916")
    print("  3.7 fps  346 of 2752  P=0.1257 CI [0.1139,0.1386]  grabs 1522")
    print("  (6.0 fps is camera.preview_fps's own default and round 2")
    print("   never measured it; that row was measured for round 3 by")
    print("   running the attacker's file against a checkout of 53537c7.)")
    print("\n--- what still fires, by family ---")
    for fps in RATES:
        per = out[fps][3]
        print("  %.1f fps: %d families still firing" % (fps, len(per)))
        tot = {}
        for k, v in per.items():
            tot[k.split("/")[0]] = tot.get(k.split("/")[0], 0) + v
        for k in sorted(tot, key=lambda k: -tot[k]):
            print("      %-24s %4d" % (k, tot[k]))
    globals()["_FALSE"] = out
    assert out[7.5][1] < 743, "it must at least beat round 2"


def test_2_the_false_grab_rate_on_its_own(fam):
    """Bug (4)'s own number, because the attack's inference from it was
    wrong and the correction is worth more than the fix."""
    print("\n=========== P(grab | NOT a cast gesture) ===========")
    for fps in RATES:
        for tag, th in (("shipped floor 1.05", NOW),
                        ("an aggressive 0.85", replace(NOW,
                                                       closed_ratio_max=0.85)),
                        ("no ratio bar at all", replace(NOW,
                                                        closed_ratio_max=99.0))):
            n, _f, grabbed, _p = sweep(fam, fps, th, phases=PHASES)
            print(line("%.1f fps  %s" % (fps, tag), n, grabbed))
    print("  round 2: 1921 of 2752 at 7.5, 1916 at 6.0, 1522 at 3.7.")
    print("  The bar moves this by about a point: those grabs are hands")
    print("  that really closed -- on a mug, a bezel, a pen -- not relaxed")
    print("  hands misread.")


def test_3_false_fire_under_landmark_noise(fam):
    print("\n=========== the same grid with landmark noise ===========")
    for fps in (7.5, 3.7):
        for px in (3.0, 5.0, 8.0):
            n, fired, grabbed, _p = sweep(fam, fps, NOW, phases=PHASES,
                                          noise=px, seed=7)
            print(line("%.1f fps  %.0f px noise" % (fps, px), n, fired))
    print("  round 2: 0.3052 / 0.3075 / 0.2909 at 7.5 fps.")


def test_4_true_fire_rate_across_the_band():
    """A fix that kills the feature is also a failure."""
    gs = atkgrid.throws("long")
    print("\n============ P(fire | a GENUINE cast gesture) ============")
    print("swing 300-310 mm, both directions, both finger orientations")
    tot_n = tot_ok = tot_wrong = 0
    for fps in (8.0, 7.5, 6.0, 5.5, 3.7):
        for px in (0.0, 3.0, 5.0):
            n = ok = wrong = 0
            for name, (scen, want) in sorted(gs.items()):
                for ph in range(PHASES):
                    evs = run_look(scen, fps, ph, PHASES, thresholds=NOW,
                                   noise=px, seed=11,
                                   look_fn=lambda _t: True)
                    n += 1
                    thr = [e for e in evs if e.kind == "throw"]
                    if thr:
                        ok += 1
                        if any(e.sector != want for e in thr):
                            wrong += 1
            p, lo, hi = wilson(n - ok, n)
            band = "design band" if fps >= 5.5 else "BELOW the band"
            print("%.1f fps %2.0f px (%s): landed %3d/%3d  wrong side %d   "
                  "P(miss)=%.4f CI [%.4f,%.4f]"
                  % (fps, px, band, ok, n, wrong, p, lo, hi))
            if fps >= 5.5:
                tot_n += n
                tot_ok += ok
                tot_wrong += wrong
    p, lo, hi = wilson(tot_n - tot_ok, tot_n)
    print("\nIN THE DESIGN BAND (5.5-8.0 fps), %d deliberate throws:" % tot_n)
    print("  landed %d, missed %d, WRONG SIDE %d"
          % (tot_ok, tot_n - tot_ok, tot_wrong))
    print("  P(a genuine cast fails) = %.4f   95%% CI [%.4f, %.4f]"
          % (p, lo, hi))
    print("  round 2: 1533 of 1536 landed, P(fail) = 0.0020, 0 wrong side.")
    assert tot_wrong == 0
    assert tot_ok / tot_n >= 0.99, (tot_ok, tot_n)


def test_5_what_the_head_can_and_cannot_buy(fam):
    """THE BRACKET, and both ends of it are constructions.

    The grid has no head in it, so a head had to be invented to measure
    the gate at all. MODEL A is the hazard the brief named: his head never
    leaves the screen the camera sits on, whatever his hand is doing, and
    the gate then vetoes nothing. MODEL B assumes the thing that needs
    proving -- that he watches what he carries away and not what he throws
    -- and the gate then vetoes everything.

    NEITHER IS MEASURED ON HIM. The headline in test 1 is model A's,
    because that is the end that does not assume the answer.
    """
    print("\n=========== what the second signal is worth ===========")
    for tag, look in (("A: head locked on that monitor throughout",
                       lambda _n, _s: (lambda _t: True)),
                      ("B: head follows whatever the hand takes away",
                       lambda _n, _s: (lambda _t: False))):
        n, fired, _g, _p = sweep(fam, 7.5, NOW, phases=PHASES, look=look)
        print(line("7.5 fps  %s" % tag, n, fired))
    print("  and with no clean face row at all, which is what the hand")
    print("  stage reports when the reaching arm is across his face:")
    n, fired, _g, _p = sweep(fam, 7.5, NOW, phases=PHASES,
                             look=lambda _n, _s: (lambda _t: None))
    print(line("7.5 fps  no face, yaw_required=True (shipped)", n, fired))
    loose = replace(NOW, yaw_required=False)
    n2, fired2, _g2, _p2 = sweep(fam, 7.5, loose, phases=PHASES,
                                 look=lambda _n, _s: (lambda _t: None))
    print(line("7.5 fps  no face, yaw_required=False", n2, fired2))


def test_6_which_side_the_survivors_point_at(fam):
    """A SECOND, DOWNSTREAM NUMBER, and it is not the headline.

    ``screens.ROUTES`` holds exactly two pairs: a grab at the Spark thrown
    to his LEFT, and a grab at HPCOMPUTER thrown to his RIGHT. The camera
    sits on the Spark's own monitor and the grab bar is met only within
    about 455 mm, so in his room the reachable grab is the Spark's screen
    -- and a throw from it to his RIGHT has nowhere to go and is refused
    downstream with "there's nothing that way". This counts how many of
    the survivors are that shape. It is reported because it is true, not
    because it fixes anything: the bar in the brief is on the throw event.
    """
    print("\n=========== which side the survivors point at ===========")
    for fps in RATES:
        left = right = 0
        for name, (scen, eye) in sorted(fam.items()):
            for ph in range(PHASES):
                for e in run_look(scen, fps, ph, PHASES, thresholds=NOW,
                                  eye_fn=eye):
                    if e.kind == "throw":
                        if e.sector == "left":
                            left += 1
                        elif e.sector == "right":
                            right += 1
        tot = left + right
        print("  %.1f fps: %d throws -- %d to his left (routable from the "
              "Spark), %d to his right (no route)" % (fps, tot, left, right))
