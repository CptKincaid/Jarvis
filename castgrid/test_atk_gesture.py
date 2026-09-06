"""ATTACK on the gesture: can an unintended movement open a screen?

MEASURED against the fixed machine at 53537c7, on a grid this attacker
built independently of the fixer's.  Every hand is synthetic.  NO CAMERA,
NO CAPTURE DEVICE, NO FRAME, NO RUSTDESK SESSION.

The BEFORE control is the same code with ``throw_speed_us = 0.0``: the
fling test then passes on every frame, which is byte-for-byte the machine
as it was before the fix.  So before and after are measured on ONE grid
with ONE set of seeds, and the two numbers are directly comparable.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from jarvis.gesture import CastThresholds
from tests.atkharness import run_seq, wilson
from tests import atkgrid

PHASES = 16
AFTER = CastThresholds()
BEFORE = replace(CastThresholds(), throw_speed_us=0.0)   # the pre-fix machine

# The camera has been measured at 7.5 fps and has run as slow as 3.7.
RATES = (7.5, 3.7)


def sweep(fam, fps, thresholds, noise=0.0, seed=0):
    """(sequences, fired, grabbed, per-family fires). One sequence = one
    family at one sub-frame offset."""
    n = fired = grabbed = 0
    per = {}
    for name, (scen, eye) in sorted(fam.items()):
        for ph in range(PHASES):
            evs = run_seq(scen, fps, ph, PHASES, thresholds=thresholds,
                          noise=noise, seed=seed, eye_fn=eye)
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
    return ("%-46s %5d of %5d   P=%.4f   95%% CI [%.4f, %.4f]"
            % (tag, fired, n, p, lo, hi))


@pytest.fixture(scope="module")
def fam():
    return atkgrid.build()


def test_the_grid_is_big_enough_and_weighted_at_that_monitor(fam):
    n = len(fam) * PHASES
    near = [k for k in fam
            if k.split("/")[0] in ("carry-out", "relaxed-carry", "set-down",
                                   "jerk-carry", "toward-lens", "bezel",
                                   "face-lost", "face-lost-relaxed",
                                   "exit-return")]
    print("\nfamilies %d, phases %d -> %d non-gesture sequences"
          % (len(fam), PHASES, n))
    print("at or beside the RIGHT monitor (the one the camera is on): "
          "%d families, %d sequences" % (len(near), len(near) * PHASES))
    assert n >= 1500, n


def test_false_fire_rate_at_every_rate_before_and_after(fam):
    print("\n================ P(fire | NOT a cast gesture) ================")
    out = {}
    for fps in RATES:
        for tag, th in (("before (no fling test)", BEFORE),
                        ("after  (53537c7)", AFTER)):
            n, fired, grabbed, per = sweep(fam, fps, th)
            out[(fps, tag)] = (n, fired, grabbed, per)
            print(line("%.1f fps  %s" % (fps, tag), n, fired)
                  + "   false grabs %d" % grabbed)
    print("\n--- what still fires AFTER the fix, by family ---")
    for fps in RATES:
        per = out[(fps, "after  (53537c7)")][3]
        n = out[(fps, "after  (53537c7)")][0]
        print("  %.1f fps: %d families still firing" % (fps, len(per)))
        for k in sorted(per, key=lambda k: -per[k]):
            print("      %-44s %2d/%d" % (k, per[k], PHASES))
    print("\n--- the same, BEFORE ---")
    for fps in RATES:
        per = out[(fps, "before (no fling test)")][3]
        print("  %.1f fps: %d families firing" % (fps, len(per)))
        for k in sorted(per, key=lambda k: -per[k])[:24]:
            print("      %-44s %2d/%d" % (k, per[k], PHASES))
    # Recorded, not asserted -- the verdict is the attacker's to state.
    globals()["_FALSE"] = out


def test_false_fire_rate_under_landmark_noise(fam):
    print("\n=========== the same grid with landmark noise ===========")
    for fps in RATES:
        for px in (3.0, 5.0, 8.0):
            n, fired, grabbed, per = sweep(fam, fps, AFTER, noise=px, seed=7)
            print(line("%.1f fps  after, %.0f px noise" % (fps, px), n, fired)
                  + "   false grabs %d" % grabbed)
            for k in sorted(per, key=lambda k: -per[k])[:12]:
                print("      %-44s %2d/%d" % (k, per[k], PHASES))


def test_the_named_attack_alone(fam):
    """The brief's own shape: reach, close, carry out laterally, no fling."""
    print("\n=========== the named attack, family by family ===========")
    only = {k: v for k, v in fam.items() if k.startswith("carry-out/")}
    for fps in RATES:
        for tag, th in (("before", BEFORE), ("after", AFTER)):
            n, fired, _g, per = sweep(only, fps, th)
            print(line("carry-out only  %.1f fps  %s" % (fps, tag), n, fired))
    print("\n  after, by traverse speed (both directions, all end depths):")
    for fps in RATES:
        for tr in atkgrid.TRAVERSES:
            sub = {k: v for k, v in only.items() if "/%.1fs/" % tr in k}
            n, fired, _g, _p = sweep(sub, fps, AFTER)
            mm = atkgrid.CARRY_MM / tr
            print("    %.1f fps  traverse %.1f s (%6.1f mm/s): %3d of %3d"
                  % (fps, tr, mm, fired, n))


# =================================================== the other side
def test_true_fire_rate_and_whether_the_two_numbers_can_coexist():
    """A fix that kills the feature is also a failure.  Deliberate throws,
    both speeds, both directions, across the design band and below it."""
    gs = atkgrid.throws("long")
    print("\n============ P(fire | a GENUINE cast gesture) ============")
    print("swing amplitude 300-310 mm -- the fixer's OWN pinned slow "
          "gesture")
    tot_n = tot_ok = tot_wrong = tot_grab = 0
    rows = []
    for fps in (8.0, 7.5, 6.0, 5.5, 3.7):
        for px in (0.0, 3.0, 5.0):
            n = ok = wrong = grabs = 0
            per = {}
            for name, (scen, want) in sorted(gs.items()):
                for ph in range(PHASES):
                    evs = run_seq(scen, fps, ph, PHASES, thresholds=AFTER,
                                  noise=px, seed=11)
                    n += 1
                    ks = [e.kind for e in evs]
                    if "grab" in ks:
                        grabs += 1
                    thr = [e for e in evs if e.kind == "throw"]
                    if thr:
                        ok += 1
                        if any(e.sector != want for e in thr):
                            wrong += 1
                            per[name] = per.get(name, 0) + 1
                    else:
                        per.setdefault("MISS:" + name, 0)
                        per["MISS:" + name] += 1
            rows.append((fps, px, n, ok, wrong, grabs, per))
            if fps >= 5.5:
                tot_n += n
                tot_ok += ok
                tot_wrong += wrong
                tot_grab += grabs
    for fps, px, n, ok, wrong, grabs, per in rows:
        p, lo, hi = wilson(n - ok, n)
        band = "design band" if fps >= 5.5 else "BELOW the band"
        print("%.1f fps %2.0f px (%s): landed %3d/%3d  grabs %3d  "
              "wrong side %d   P(miss)=%.4f CI [%.4f,%.4f]"
              % (fps, px, band, ok, n, grabs, wrong, p, lo, hi))
        miss = {k[5:]: v for k, v in per.items() if k.startswith("MISS:")}
        for k in sorted(miss, key=lambda k: -miss[k])[:6]:
            if miss[k]:
                print("        missed %-38s %2d/%d" % (k, miss[k], PHASES))
    p, lo, hi = wilson(tot_n - tot_ok, tot_n)
    print("\nIN THE DESIGN BAND (5.5-8.0 fps), %d deliberate throws:"
          % tot_n)
    print("  landed %d, missed %d, WRONG SIDE %d" % (tot_ok, tot_n - tot_ok,
                                                     tot_wrong))
    print("  P(a genuine cast fails) = %.4f   95%% CI [%.4f, %.4f]"
          % (p, lo, hi))
    assert tot_n >= 300, tot_n

    # THE SAME GESTURE, SHORTER SWING. His own fast gesture throws 250-260
    # mm; the pinned slow one throws 300-310. A slow throw at the FAST
    # gesture's own amplitude is a different animal, and this is how
    # different.
    print("\n--- the same, at the FAST gesture's own 250-260 mm swing ---")
    gs2 = atkgrid.throws("short")
    for fps in (7.5, 6.0):
        for nm in ("throw-fast", "throw-slow"):
            n = ok = 0
            for name, (scen, want) in sorted(gs2.items()):
                if not name.startswith(nm):
                    continue
                for ph in range(PHASES):
                    evs = run_seq(scen, fps, ph, PHASES, thresholds=AFTER,
                                  seed=11)
                    n += 1
                    ok += any(e.kind == "throw" for e in evs)
            print("  %.1f fps  %-12s landed %3d of %3d" % (fps, nm, ok, n))


def test_what_speed_the_machine_thinks_it_saw():
    """WHY the survivors survive: the hand-unit is latched at the GRAB, so a
    hand that comes closer to the lens afterwards is measured against a unit
    that is too small and reads FASTER than it is travelling."""
    from jarvis.gesture import CastGesture
    from tests.atkharness import (EYE_PX, Clock, W, H, hand3d, project,
                                  _curls)
    from jarvis.gesture import observe_hand
    print("\n===== world speed vs the speed the machine judged on =====")
    print("%-34s %11s %11s %8s" % ("family", "world mm/s", "peak u/s",
                                   "fires"))
    cases = [("carry-out/his-left/2.0s/425mm",
              atkgrid.carry_out("his-left", 2.0, 425.), 260.0),
             ("carry-out/his-left/1.6s/425mm",
              atkgrid.carry_out("his-left", 1.6, 425.), 325.0),
             ("carry-out/his-left/1.2s/425mm",
              atkgrid.carry_out("his-left", 1.2, 425.), 433.0),
             ("toward-lens/his-left/2.4s",
              atkgrid.toward_lens_sweep("his-left", 2.4), 300. / 2.4),
             ("toward-lens/his-left/1.8s",
              atkgrid.toward_lens_sweep("his-left", 1.8), 300. / 1.8),
             ("toward-lens/his-left/1.2s",
              atkgrid.toward_lens_sweep("his-left", 1.2), 300. / 1.2)]
    for name, scen, mm_s in cases:
        dur, traj = scen
        clk = Clock()
        m = CastGesture(AFTER, (W, H), now=clk, preview_fps=7.5)
        t = 0.0
        i = 0
        peak = 0.0
        fired = False
        while t < dur:
            pos, curl, pi, ya, ro = traj(t)
            img = project(hand3d(_curls(curl)), pos, pi, ya, ro)
            hands = () if img is None else (observe_hand(img, EYE_PX),)
            e = m.update(hands, EYE_PX, i)
            peak = max(peak, m.status()["speed_us"])
            if e is not None and e.kind == "throw":
                fired = True
            t += 1 / 7.5
            clk.t += 1 / 7.5
            i += 1
        print("%-34s %11.1f %11.2f %8s   (world u/s %.2f)"
              % (name, mm_s, peak, fired, mm_s / 127.0))


def test_the_fling_window_ladder():
    """Is ``fling_window_s`` actually 0.55 s of memory?

    One quick correction in the middle of an otherwise SLOW carry (154 mm/s
    throughout, a fifth of a throw), and then the hand leaves the picture
    ``gap`` seconds later.  Everything else in the row is identical."""
    print("\n===== a slow carry with ONE fast correction in it =====")
    print("  gap after the correction ->  fires (of %d), both directions"
          % (PHASES * 2))
    for fps in RATES:
        for gap in (0.00, 0.15, 0.30, 0.45, 0.60, 0.90, 1.30):
            n = fired = 0
            for side in atkgrid.SIDES:
                scen = atkgrid.jerk_then_carry(side, gap)
                for ph in range(PHASES):
                    evs = run_seq(scen, fps, ph, PHASES, thresholds=AFTER)
                    n += 1
                    fired += any(e.kind == "throw" for e in evs)
            print("    %.1f fps  gap %.2f s: %2d of %2d" % (fps, gap, fired, n))


def test_the_release_and_handover_ladders():
    """The RELEASE branch: setting a thing down beside that monitor, and
    handing it to somebody."""
    print("\n===== setting it down aside, by how briskly =====")
    for fps in RATES:
        for mv in (1.2, 0.9, 0.7, 0.5):
            n = fired = 0
            for side in atkgrid.SIDES:
                scen = atkgrid.set_down_aside(side, mv)
                for ph in range(PHASES):
                    evs = run_seq(scen, fps, ph, PHASES, thresholds=AFTER)
                    n += 1
                    fired += any(e.kind == "throw" for e in evs)
            print("    %.1f fps  210 mm in %.1f s (%5.0f mm/s): %2d of %2d"
                  % (fps, mv, 210. / mv, fired, n))
    print("\n===== handing it to somebody standing beside him =====")
    for fps in RATES:
        for sw in (0.35, 0.5, 0.7):
            n = fired = 0
            for side in atkgrid.SIDES:
                scen = atkgrid.hand_to_someone(side, sw)
                for ph in range(PHASES):
                    evs = run_seq(scen, fps, ph, PHASES, thresholds=AFTER)
                    n += 1
                    fired += any(e.kind == "throw" for e in evs)
            print("    %.1f fps  235 mm in %.2f s (%5.0f mm/s): %2d of %2d"
                  % (fps, sw, 235. / sw, fired, n))
