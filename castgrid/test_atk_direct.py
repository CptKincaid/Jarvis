"""Direct-drive attacks: the timing law, the association, the unit.

HandObservation rows only -- the same public API the camera path uses, with
no projection in the way, so a timing claim can be pinned to the frame.
NO CAMERA, NO FRAME, NO CAPTURE DEVICE.
"""
from __future__ import annotations

from jarvis.gesture import (CastGesture, CastState, CastThresholds,
                            HandObservation)

UNIT = 223.0          # px per hand-unit at his 425 mm reach, MEASURED
EYE = 89.38
OPEN, FIST = 1.30, 0.55
T = CastThresholds()


class Clock:
    def __init__(self, t=100.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def obs(cx, cy=360.0, closed=FIST, reach=2.6, palm=UNIT):
    return HandObservation(palm_diag=palm, closed=closed, reach=reach,
                           cx=cx, cy=cy)


def machine(fps=7.5):
    return CastGesture(T, (1280, 720), now=Clock(), preview_fps=fps)


def grab_at(m, cx, cy=360.0):
    i = [0]

    def step(hands):
        i[0] += 1
        m._now.t += 1.0 / m.preview_fps
        return m.update(hands, EYE, i[0])
    step((obs(cx, cy, OPEN),))
    for _ in range(3):
        e = step((obs(cx, cy, FIST),))
    assert e is not None and e.kind == "grab", e
    return step


# ------------------------------------------------- the fling window, exactly
def test_how_long_a_fling_is_remembered_and_what_that_buys():
    """A hand may be at throw speed ONCE, then crawl, then vanish, and the
    carry still resolves as a throw.  This measures the whole grant: the
    slow frames the window pays for, PLUS the grace frames the hand may be
    absent for after them."""
    print("\n=== after ONE fast frame, how many SLOW frames may follow "
          "and still throw? ===")
    print("  (slow frames move 0.08 u each -- 0.6 u/s, a FIFTH of the "
          "3.0 u/s bar)")
    for fps in (7.5, 6.0, 3.7):
        rows = []
        for slow in range(0, 9):
            m = machine(fps)
            step = grab_at(m, 900.0)
            cx = 900.0 + 1.00 * UNIT          # ONE fast frame
            step((obs(cx),))
            for _ in range(slow):
                cx += 0.08 * UNIT
                step((obs(cx),))
            t_last_seen = m._now.t
            ev = None
            for _ in range(T.lost_grace_frames + 2):
                e = step(())
                if e is not None:
                    ev = e
                    break
            fired = ev is not None and ev.kind == "throw"
            rows.append((slow, slow / fps, fired,
                         (m._now.t - t_last_seen) if fired else 0.0))
        for slow, secs, fired, absent in rows:
            print("    %.1f fps  %d slow frames (%.2f s of crawl): %-5s"
                  % (fps, slow, secs, fired)
                  + ("   + %.2f s absent before it fired" % absent
                     if fired else ""))
    # The stated law, pinned: the window is measured to the LAST SEEN frame,
    # and the grace frames are on top of it.
    m = machine(7.5)
    step = grab_at(m, 900.0)
    t_fast = m._now.t
    step((obs(900.0 + 1.00 * UNIT),))
    cx = 900.0 + 1.00 * UNIT
    for _ in range(4):
        cx += 0.08 * UNIT
        step((obs(cx),))
    ev = None
    for _ in range(4):
        e = step(())
        if e is not None:
            ev = e
            break
    assert ev is not None and ev.kind == "throw", ev
    print("\n  PINNED: at 7.5 fps the throw fired %.2f s after the last "
          "frame that was at throw speed" % (m._now.t - t_fast))
    print("          fling_window_s is %.2f s and it is measured to the "
          "last SEEN frame;" % T.fling_window_s)
    print("          lost_grace_frames (%d) is added on top of it."
          % T.lost_grace_frames)


# ------------------------------------------------ the association hijack
def test_a_second_hand_can_be_adopted_mid_carry_and_it_reads_as_a_fling():
    """During a carry ``_pick`` relaxes to NEAREST CENTROID with no identity
    check at all, up to ``assoc_max_u`` = 2.50 units away.  His other hand
    coming up mid-carry can therefore be adopted as the carried hand, and
    the jump to it is measured as one frame of travel."""
    print("\n=== his OTHER hand comes up while a carry is live ===")
    for jump_u in (0.5, 1.0, 1.5, 2.0, 2.4):
        m = machine(7.5)
        step = grab_at(m, 640.0)
        # the carried hand crawls; the other hand appears jump_u away and
        # NEARER the last-seen point, so nearest-centroid takes it.
        step((obs(650.0),))
        other = 650.0 + jump_u * UNIT
        e = step((obs(other), obs(650.0 + 2.6 * UNIT)))
        st = m.status()
        print("    other hand %.1f u away: adopted, speed read %.2f u/s, "
              "flung=%s" % (jump_u, st["speed_us"], st["flung"]))
    # and then it can be thrown
    m = machine(7.5)
    step = grab_at(m, 640.0)
    step((obs(650.0),))
    step((obs(650.0 + 2.0 * UNIT),))      # the OTHER hand, adopted
    ev = None
    for _ in range(4):
        e = step(())
        if e is not None:
            ev = e
            break
    print("    -> then the hand is lost: %s (%s), sector %r"
          % (ev.kind, ev.why, ev.sector))


# --------------------------------------------------- the unit is not 127 mm
def test_the_hand_unit_is_the_foreshortened_palm_not_127_mm():
    """``throw_speed_us = 3.0`` is documented as 'about 380 mm/s at his
    reach'.  It is not.  The unit is the MEASURED palm_diag in pixels, which
    at his reaching pitch is 25% smaller than 127 mm projects to, and the
    unit is LATCHED at the grab -- so a hand that comes nearer the lens
    afterwards is measured against a unit that is smaller still."""
    from tests.atkharness import EYE_PX, F_PX, hand3d, project, _curls
    from jarvis.gesture import observe_hand
    print("\n=== what 3.0 hand-units/second is worth in millimetres ===")
    print("%8s %8s %12s %12s %14s"
          % ("z(mm)", "pitch", "palm_diag", "127mm as px", "3.0 u/s ="))
    for z, pitch in ((425, 0), (425, 35), (425, 50), (385, 48), (350, 35),
                     (300, 35)):
        img = project(hand3d(_curls(1.0)), (0., 10., float(z)), float(pitch))
        pd = observe_hand(img, EYE_PX).palm_diag
        true_px = F_PX * 126.8 / z
        mm_s = 3.0 * pd * z / F_PX
        print("%8d %8d %12.1f %12.1f %10.0f mm/s"
              % (z, pitch, pd, true_px, mm_s))
    print("\n  and with the unit LATCHED at a 425 mm / 35 deg grab, the same")
    print("  3.0 u/s bar as the hand moves nearer the lens:")
    img = project(hand3d(_curls(1.0)), (0., 10., 425.), 35.)
    latched = observe_hand(img, EYE_PX).palm_diag
    for z in (425, 400, 375, 350, 325, 300):
        print("      at z=%d mm -> %.0f mm/s" % (z, 3.0 * latched * z / F_PX))
