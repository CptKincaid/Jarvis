"""ROUND 5's WIND-UP measurement, and it is the end of the line for this
gesture.

THE QUESTION, ASKED AND ANSWERED. 2026-09-05: "when you throw, does your
hand pull back a little first, before it swings?" HE SAID YES. Three rounds
had established that nothing in the hand track separates his throw from his
ordinary desk motion -- same approach, same dwell, same grip, same speed,
same exit through the same edge -- and a wind-up was the one signal left,
because carrying something away never reverses before it leaves.

HOW TO RUN IT. This directory is deliberately OUTSIDE pytest.ini's
``testpaths``, so ``pytest -q`` does not collect it: it is an instrument,
not a regression. From the repo root:

    ~/.local/bin/memcap timeout 1700 ~/vss_env/bin/python -m pytest -q \\
      -p no:cacheprovider castgrid/test_r5_windup.py -s

NO CAMERA, NO CAPTURE DEVICE, NO FRAME, NO RUSTDESK SESSION, and nothing
touches his desktop. Every hand is a synthetic 21-point row projected
in-process through his LifeCam constants.

THE ANSWER, UP FRONT, BECAUSE IT IS NEGATIVE AND HE SHOULD NOT HAVE TO READ
A TABLE TO GET IT:

  The wind-up is REAL, it is MEASURABLE, and it is NOT ENOUGH. On the
  preserved 172-family grid alone a bar of 0.05 u takes the false-fire rate
  from 0.2362 to 0.0000 -- which looks like a complete solution and IS AN
  ARTEFACT: that grid was built to attack speed and distance and contains
  no retractions at all. Add 54 families that DO retract -- pulling a mug
  toward himself before lifting it away, drawing back to get a run at
  something heavy, a hand that hesitates, all ordinary desk motions -- and
  the best bar in the sweep leaves P(fire | NOT a cast gesture) at 0.1054,
  which is TWENTY-ONE TIMES the 0.005 bar, and it demands a 120 mm
  pull-back before every throw. A gesture that needs him to perform is not
  the gesture he asked for.

  AND THE DIRECTION MATTERS MORE THAN THE AMPLITUDE. A wind-up that pulls
  his hand back TOWARD HIS BODY measures 0.018 u AT EVERY AMPLITUDE up to
  120 mm, because it produces almost no movement in the image plane. If
  that is the wind-up he makes, every bar above 0.02 refuses every throw
  he makes and the signal is not merely insufficient, it is absent.

THE FOUR RULES THIS FILE IS HELD TO, from the brief, because a grid that
invents a wind-up and then detects it has proved nothing:

  1. THE WIND-UP IS SWEPT, amplitude AND duration, including values too
     small to detect (0, 5, 10 mm). At amplitude 0 the family reproduces
     ``atkgrid``'s own throw event for event -- asserted, in test 1.
  2. EVERY NON-GESTURE FAMILY IS RE-EXAMINED for motions that also
     retract, invented deliberately and adversarially, and swept to the
     SAME amplitudes the throw is. (Measured first at 20-70 mm, where a
     0.60 u bar looked like it separated; it had merely run past the
     biggest retraction the grid contained. That is in test 4.)
  3. RECALL IS REPORTED AS A FUNCTION OF AMPLITUDE, so the honest sentence
     -- "your throw must pull back at least N millimetres" -- can be said
     with a number in it.
  4. HIS OWN AMPLITUDE AND DURATION ARE UNKNOWN and are stated as such.
     ``scripts/gesture_selfcheck.py --seconds 30 --windup`` is the
     numbers-only instrument that lets him find out.

THE BASELINE IS REPRODUCED FIRST (test 0) rather than quoted, so every
number here is comparable with round 4's 0.2362 / 0.2322 / 0.1435 instead
of being a fresh grid that agrees with itself.
"""
from __future__ import annotations

import statistics as st

import pytest

from castgrid import r5windup as R
from castgrid.r4recall import mm_per_unit
from castgrid.r3run import sweep
from jarvis.gesture import CastThresholds
from tests import atkgrid
from tests.atkharness import run_seq

# The round-4 numbers this round must reproduce before it may compare
# anything with them. From jarvis/gesture.py's own docstring.
ROUND4 = {7.5: 0.2362, 6.0: 0.2322, 3.7: 0.1435}
MM_PER_U = mm_per_unit(425.0)


@pytest.fixture(scope="module")
def grids():
    return atkgrid.build(), R.retraction_grid()


@pytest.fixture(scope="module")
def fires(grids):
    """One pass per (grid, rate). EVERY BAR IS READ OFF THESE.

    A sequence's wind-up is measured once and every bar in ``BARS_U`` is
    applied to the recorded value afterwards, so the whole sweep costs one
    grid pass per rate -- and, more to the point, every bar is measured on
    exactly the same sequences with exactly the same seeds.
    """
    pres, retr = grids
    out = {}
    for tag, fam in (("preserved", pres), ("retraction", retr)):
        for fps in (7.5, 6.0, 3.7):
            out[(tag, fps)] = R.sweep_fires(fam, fps, phases=16)
    return out


@pytest.fixture(scope="module")
def recall():
    return R.recall_pass(way="lateral", wu_s=0.25)


# ================================================ 0. reproduce the baseline
def test_0_the_round_four_baseline_reproduces_exactly():
    """NOTHING BELOW MAY BE COMPARED WITH A NUMBER THIS ROUND DID NOT
    RE-MEASURE. Same files, same families, same seeds, same phases."""
    fam = atkgrid.build()
    print("\n  the preserved grid, re-measured on this checkout:")
    for fps in (7.5, 6.0, 3.7):
        n, fired, _g, _p = sweep(fam, fps, CastThresholds(), phases=16)
        got = fired / n
        print("    %.1f fps  %4d of %4d = %.4f   (round 4 reported %.4f)"
              % (fps, fired, n, got, ROUND4[fps]))
        assert abs(got - ROUND4[fps]) < 1e-4, (fps, got, ROUND4[fps])
    print("  reproduced. Every number below is comparable with round 4.")


def test_1_at_amplitude_zero_the_family_is_the_preserved_one(grids):
    """THE FIXED POINT. If the wind-up family at amplitude 0 is not
    ``atkgrid``'s own throw, the sweep measures a new model rather than the
    effect of one added span."""
    checked = 0
    for side, end in sorted(atkgrid.LONG.items()):
        for slow in (True, False):
            base = (atkgrid.throw_slow(end) if slow
                    else atkgrid.throw_fast(end))
            mine = R.throw_windup(end, amp_mm=0.0, wu_s=0.25, slow=slow)
            assert abs(base[0] - mine[0]) < 1e-9, (side, slow, base[0],
                                                   mine[0])
            for fps in (6.0, 7.5):
                for ph in range(4):
                    a = [(e.kind, e.sector, round(e.dist_u, 9))
                         for e in run_seq(base, fps, ph, 4)]
                    b = [(e.kind, e.sector, round(e.dist_u, 9))
                         for e in R.run(mine, fps, ph, 4)]
                    assert a == b, (side, slow, fps, ph, a, b)
                    checked += 1
    print("\n  amplitude 0 reproduces atkgrid event for event over %d "
          "sequences" % checked)


# =========================================== 2. the wind-up IS measurable
def test_2_the_detector_measures_the_wind_up_that_was_injected():
    """It is a real signal and it is not being flattered: what goes in
    comes out, discounted by the sampler, which is honest -- the frames
    rarely land on the extreme of the retraction."""
    print("\n  one hand-unit = %.1f mm at z=425 (GUESSED anthropometry; "
          "127 mm documented)" % MM_PER_U)
    print("\n  HIS THROW, lateral wind-up, 7.5 fps, 8 phases x 4 durations:")
    print("    injected mm   fired    measured windup_u (median)   "
          "geometric mm")
    seen = []
    for amp in R.AMPS_MM:
        got, n = [], 0
        for wu in R.DURS_S:
            scen = R.throw_family("his-left", amp_mm=amp, wu_s=wu)
            for ph in range(8):
                f, w, _bad = R.wind_of(R.run(scen, 7.5, ph, 8, look=True),
                                       "left")
                n += 1
                if f:
                    got.append(w)
        med = st.median(got) if got else 0.0
        seen.append((amp, med))
        print("    %11.0f   %2d/%2d    %22.3f   %10.0f"
              % (amp, len(got), n, med, med * MM_PER_U))
    ups = [m for _a, m in seen]
    assert ups == sorted(ups), "the measurement must be monotone in what "\
        "was injected"
    assert seen[0][1] < 0.03, "amplitude 0 must measure ~nothing"
    assert seen[-1][1] > 0.6, "120 mm must be plainly visible"


def test_3_a_wind_up_toward_his_body_is_invisible_to_the_lens():
    """THE FINDING THAT MATTERS MORE THAN THE AMPLITUDE, and it is not a
    tuning problem. The wind-up is measured in the IMAGE PLANE along the
    axis the throw took. A retraction straight back toward him produces
    almost no image displacement -- the hand only gets smaller -- so it
    reads as nothing whatever its size."""
    print("\n  the SAME amplitudes, pulled back TOWARD HIM instead:")
    print("    injected mm   measured windup_u (median)")
    worst = 0.0
    for amp in R.AMPS_MM:
        got = []
        for wu in R.DURS_S:
            scen = R.throw_family("his-left", amp_mm=amp, wu_s=wu,
                                  way="depth")
            for ph in range(8):
                f, w, _b = R.wind_of(R.run(scen, 7.5, ph, 8, look=True),
                                     "left")
                if f:
                    got.append(w)
        med = st.median(got) if got else 0.0
        worst = max(worst, med)
        print("    %11.0f   %26.3f" % (amp, med))
    print("\n  120 mm of wind-up toward his body reads as %.3f u." % worst)
    print("  ANY BAR ABOVE THAT REFUSES EVERY THROW HE MAKES. If this is")
    print("  the wind-up he does, the signal is not weak -- it is ABSENT,")
    print("  and no threshold recovers it.")
    assert worst < 0.05


# ==================================== 4. what it buys, honestly
def test_4_what_the_bar_buys_on_each_grid(fires, recall):
    """THE HEADLINE TABLE. The preserved grid alone says the wind-up
    solves it. The preserved grid alone is WRONG, because it contains no
    retractions -- it was built to attack speed and distance."""
    for fps in (7.5, 6.0):
        pn, pf = fires[("preserved", fps)]
        rn, rf = fires[("retraction", fps)]
        bn, bf = pn + rn, pf + rf
        print("\n  ===== %.1f fps =====   preserved n=%d   + retraction n=%d"
              % (fps, pn, bn))
        print("   bar u   ~mm    preserved 172 families        "
              "+ the 54 that RETRACT            recall needs")
        for bar in R.BARS_U:
            p, lo, hi = R.rate_at(pn, pf, bar)
            p2, lo2, hi2 = R.rate_at(bn, bf, bar)
            need = None
            for amp in R.AMPS_MM:
                n, hits, _w = recall[amp]
                if sum(1 for w in hits if w >= bar) / n >= 0.95:
                    need = amp
                    break
            print("   %5.2f  %5.0f    %.4f [%.4f, %.4f]   %.4f [%.4f, %.4f]"
                  "     %s"
                  % (bar, bar * MM_PER_U, p, lo, hi, p2, lo2, hi2,
                     ("%.0f mm" % need) if need is not None else ">120 mm"))
        base = R.rate_at(bn, bf, 0.0)[0]
        best = R.rate_at(bn, bf, R.BARS_U[-1])[0]
        print("\n   THE BAR IS 0.005. On the honest grid the wind-up takes")
        print("   %.4f to %.4f -- still %.0f times over -- and the bar that"
              % (base, best, best / 0.005))
        print("   does it needs a 120 mm pull-back before every throw.")
        assert best > 0.005 * 5, "if this ever passes, re-read test 3 first"


def test_5_recall_as_a_function_of_wind_up_amplitude(recall):
    """RULE 3 OF THE BRIEF, and the sentence he is owed: 'your throw must
    pull back at least N millimetres'. Read the column for the bar, find
    the first row at 0.95, that is N."""
    print("\n  RECALL vs WIND-UP AMPLITUDE (lateral, 0.25 s retraction,")
    print("  6.0 + 7.5 fps, 0 and 3 px noise, both sides, both finger")
    print("  orientations, 8 sub-frame phases):")
    print("    amp mm   n     median u   " + "".join("%7.2f" % b
                                                     for b in R.BARS_U))
    for amp in R.AMPS_MM:
        n, hits, wrong = recall[amp]
        med = st.median(hits) if hits else 0.0
        row = "    %6.0f  %4d   %8.3f   " % (amp, n, med)
        for bar in R.BARS_U:
            row += "%7.4f" % (sum(1 for w in hits if w >= bar) / n)
        print(row + ("   WRONG SIDE %d" % wrong if wrong else ""))
    print("\n  Read it as: at a bar of 0.15 u his throw has to pull back")
    print("  about 40 mm to fire 95% of the time, and that bar leaves the")
    print("  false-fire rate at 0.2600. At 0.60 u it is 120 mm and 0.1054.")
    n0, hits0, _w = recall[0.0]
    assert len(hits0) / n0 > 0.99, "the no-wind-up throw must still fire "\
        "with the gate off -- otherwise this measures a broken family"


def test_6_the_adversarial_carries_retract_exactly_as_his_throw_does():
    """WHY IT CANNOT BE TUNED, in one table. These are NOT throws. Each is
    an ordinary thing to do at a desk, each contains a retraction, and each
    measures the SAME wind-up his gesture does at the same amplitude --
    slightly MORE, because a carry dwells in the retraction longer than a
    throw does."""
    print("\n  injected mm    HIS THROW (median u)    A CARRY THAT PULLS "
          "BACK FIRST (median u)")
    fam = R.retraction_grid()
    by = {}
    for name, (scen, eye) in sorted(fam.items()):
        amp = float(name.split("/")[2].replace("mm", ""))
        for ph in range(8):
            f, w, _b = R.wind_of(R.run(scen, 7.5, ph, 8, eye_fn=eye))
            if f:
                by.setdefault(amp, []).append(w)
    for amp in sorted(by):
        got = []
        for wu in R.DURS_S:
            scen = R.throw_family("his-left", amp_mm=amp, wu_s=wu)
            for ph in range(8):
                f, w, _b = R.wind_of(R.run(scen, 7.5, ph, 8, look=True),
                                     "left")
                if f:
                    got.append(w)
        thr = st.median(got) if got else float("nan")
        car = st.median(by[amp])
        print("  %11.0f    %19.3f    %30.3f" % (amp, thr, car))
        assert car > thr * 0.5, "if a carry ever measured far less than the "\
            "throw at the same amplitude, THAT would be the separation"
    print("\n  They are the same measurement. That is the whole finding:")
    print("  the wind-up is a threshold on an amplitude his throw and his")
    print("  carries share, so it trades recall for precision one for one,")
    print("  exactly as the speed bar already did.")


def test_7_the_gate_ships_off_and_the_module_says_why():
    """THE OUTCOME, AND IT IS A LEGITIMATE ONE. An honest 'this signal is
    worth X and X is not enough' is the deliverable."""
    t = CastThresholds()
    print("\n  shipped windup_min_u = %.2f (inert)" % t.windup_min_u)
    assert t.windup_min_u == 0.0
    from jarvis.assistant_config import DEFAULTS
    assert DEFAULTS["gesture"]["enabled"] is False
    print("  gesture.enabled = False, unchanged and not re-opened.")
