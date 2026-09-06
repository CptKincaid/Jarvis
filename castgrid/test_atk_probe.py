"""Geometry probe: what the rig actually measures, before any grid runs."""
import math

import numpy as np

import jarvis.gesture as g
from jarvis.gesture import CastThresholds, observe_hand
from tests.atkharness import EYE_PX, F_PX, hand3d, project, run_seq
from tests import atkgrid


def test_probe_the_room():
    t = CastThresholds()
    print("\n=== the lens and the bars ===")
    print("F_PX=%.1f  EYE_PX=%.2f  reach_min=%.2f reach_exit=%.2f "
          "reach_arm=%.2f  throw_speed_us=%.2f  fling_window_s=%.2f"
          % (F_PX, EYE_PX, t.reach_min, t.reach_exit, t.reach_arm,
             t.throw_speed_us, t.fling_window_s))
    print("\n=== reach and hand-unit vs depth, at his working pitch 35 ===")
    print("  z(mm)   palm_diag_px   reach R   half-frame(mm)")
    for z in (300, 350, 395, 425, 455, 500, 550, 600, 700, 780):
        img = project(hand3d((1.0, 1.0, 1.0, 1.0)), (0., 20., float(z)), 35.)
        if img is None:
            print("  %5d   (out of frame)" % z)
            continue
        o = observe_hand(img, EYE_PX)
        print("  %5d   %10.1f   %7.3f   %10.1f"
              % (z, o.palm_diag, o.reach, 0.6446 * z))
    print("\n=== what one hand-unit is worth, and the speed ladder ===")
    img = project(hand3d((1.0, 1.0, 1.0, 1.0)), (0., 20., 425.), 35.)
    unit = observe_hand(img, EYE_PX).palm_diag
    print("  unit at 425 mm = %.1f px" % unit)
    for tr in atkgrid.TRAVERSES:
        mm_s = atkgrid.CARRY_MM / tr
        print("  traverse %.1f s -> %6.1f mm/s  = %.2f world-units/s"
              % (tr, mm_s, mm_s / g.HAND_UNIT_MM))


def test_probe_the_grid_is_real():
    """Every family must actually reach the machine: a grid of motions that
    never even arm is a grid that proves nothing."""
    fam = atkgrid.build()
    print("\nfamilies: %d" % len(fam))
    armed = grabs = 0
    dead = []
    for name, (scen, eye) in sorted(fam.items()):
        evs = run_seq(scen, 7.5, 0, 16, eye_fn=eye)
        kinds = [e.kind for e in evs]
        if kinds:
            armed += 1
        grabs += kinds.count("grab")
        if not kinds:
            dead.append(name)
    print("families producing at least one event at phase 0: %d/%d"
          % (armed, len(fam)))
    print("grabs at phase 0: %d" % grabs)
    print("silent families (%d): %s" % (len(dead), ", ".join(dead[:40])))
