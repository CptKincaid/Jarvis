"""ROUND 3: the four measurement bugs, and the second signal for a throw.

Every hand here is a synthetic 21-point landmark row or a hand-built
``HandObservation``. NO CAMERA, NO CAPTURE DEVICE, NO FRAME EVER OPENED,
and nothing in this file touches his desktop.

The numbers each test pins were measured on the preserved attack grid
(castgrid/) at 53537c7 before the fix, and the failure each one produced
is quoted in the round-3 report.
"""
from __future__ import annotations

import math

from jarvis.gesture import (CastGesture, CastThresholds, HandObservation,
                            observe_hand)
from tests.atkharness import EYE_PX, F_PX, hand3d, project, _curls

UNIT = 223.0          # palm_diag px at his 425 mm reach, pitch 35 (MEASURED)
EYE = 89.38
OPEN, FIST = 1.30, 0.55
T = CastThresholds()


class Clock:
    def __init__(self, t=100.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def obs(cx, cy=360.0, closed=FIST, reach=2.6, palm=UNIT, **kw):
    """A hand row whose pose-corrected unit equals its palm_diag unless
    asked otherwise, so a test that is not about the unit is not about the
    unit."""
    kw.setdefault("palm_len", palm * 107.0 / 127.0)
    kw.setdefault("palm_w", palm * 68.2 / 127.0)
    kw.setdefault("curled", 0.66 if closed <= 0.70 else 1.20)
    return HandObservation(palm_diag=palm, closed=closed, reach=reach,
                           cx=cx, cy=cy, **kw)


def machine(fps=7.5, t=None):
    return CastGesture(t or T, (1280, 720), now=Clock(), preview_fps=fps)


def grab_at(m, cx, cy=360.0, looking=True):
    i = [0]

    def step(hands, **kw):
        i[0] += 1
        m._now.t += 1.0 / m.preview_fps
        kw.setdefault("looking", looking)
        return m.update(hands, EYE, i[0], **kw)
    step((obs(cx, cy, OPEN),))
    for _ in range(3):
        e = step((obs(cx, cy, FIST),))
    assert e is not None and e.kind == "grab", e
    return step


def landmarks(curl, z, pitch, y=10.0):
    """None when that pose does not fit in the picture."""
    return project(hand3d(_curls(curl)), (0., y, float(z)), float(pitch))


# ============================================ BUG 4: the closed bar at his pose
def test_the_closed_scalar_is_pose_dependent_and_a_ratio_does_not_fix_it():
    """THE ATTACK WAS RIGHT ABOUT THE SCALAR AND WRONG ABOUT WHAT IT COST,
    and this test is the measurement that says so.

    ``closed_max`` is pose-dependent: over a slice at roll 0 and yaw 0, a
    curl of 0.40 at his working pitch of 35 deg reads C = 0.602 against
    0.70. ``curled`` -- the same extension divided by the palm LENGTH,
    which foreshortens with the fingers -- separates them over that slice.

    Over the WHOLE pose envelope it does not, and a bar tight enough to
    help refuses real fists, including the attacker's own fingers-down
    throw family. So what ships is the measurement plus a sanity floor,
    not a gate, and this test pins both halves.
    """
    rows = []
    for z in (350, 385, 425, 455, 500):
        for pitch in (0, 15, 25, 35, 45, 55):
            for roll in (0, 20, 180, 200):
                for yaw in (-20, 0, 20):
                    img = project(hand3d(_curls(1.0)), (0., 10., float(z)),
                                  pitch, yaw, roll)
                    if img is not None:
                        rows.append(observe_hand(img, EYE_PX))
    worst = max(o.curled for o in rows)
    print("\na real grip reads curled up to %.3f over %d poses; the shipped "
          "floor is %.2f" % (worst, len(rows), T.closed_ratio_max))
    assert worst <= T.closed_ratio_max, worst
    # ...and the slice the attack measured on, where it DOES separate.
    flat = [(c, observe_hand(landmarks(c, 425, 35), EYE_PX))
            for c in (0.30, 0.40, 1.00)]
    for curl, o in flat:
        print("  curl %.2f at pitch 35: C=%.3f curled=%.3f"
              % (curl, o.closed, o.curled))
    fist = [o for c, o in flat if c == 1.00][0]
    relaxed = [o for c, o in flat if c == 0.40][0]
    assert relaxed.closed <= T.closed_max, "the shipped scalar calls it a fist"
    assert relaxed.curled > fist.curled, "the ratio does not"


def test_the_closed_ratio_is_reachable_from_the_config_file():
    from jarvis.handstage import thresholds_from_options
    got = thresholds_from_options({"gesture.closed_ratio_max": 0.61}.get)
    assert got.closed_ratio_max == 0.61


# ================================== BUG 1: the unit is latched and foreshortened
def test_one_hand_unit_is_the_same_number_of_millimetres_at_every_pose():
    """``throw_speed_us`` is documented as "about 380 mm/s at his reach".
    MEASURED on the shipped unit: 381 mm/s for a flat palm square to the
    lens, 289 at his working pitch of 35 deg, 242 at 50 deg, and 204 once
    the hand has moved to 300 mm with the unit still latched at the grab.
    The bar has to mean one thing.
    """
    rows = []
    for z in (300, 350, 385, 425, 455, 500):
        for pitch in (0, 15, 25, 35, 45, 50):
            img = landmarks(1.0, z, pitch)
            if img is None:
                continue
            o = observe_hand(img, EYE_PX)
            # the depth of the PALM, not of the wrist: the MCP row sits
            # forward of the wrist and a pitch pushes it away from the lens.
            zc = float(z) + 55.0 * math.sin(math.radians(pitch)) * 0.8
            true_px = F_PX * 126.8 / zc
            rows.append((z, pitch, o.palm_diag, o.unit, true_px))
    print("\n%6s %6s %10s %10s %10s %8s %8s"
          % ("z", "pitch", "palm_diag", "unit", "true px", "old err", "new err"))
    worst_old = worst_new = 0.0
    for z, pitch, pd, unit, true_px in rows:
        eo = abs(pd - true_px) / true_px
        en = abs(unit - true_px) / true_px
        worst_old, worst_new = max(worst_old, eo), max(worst_new, en)
        print("%6d %6d %10.1f %10.1f %10.1f %7.1f%% %7.1f%%"
              % (z, pitch, pd, unit, true_px, 100 * eo, 100 * en))
    print("worst error: shipped %.1f%%, corrected %.1f%%"
          % (100 * worst_old, 100 * worst_new))
    assert worst_new <= 0.12, worst_new


def test_the_speed_bar_does_not_move_when_the_hand_comes_nearer_the_lens():
    """The unit is latched at the grab, so the SAME sweep reads faster
    coming toward the lens than going away. MEASURED: it fires at
    250 mm/s inbound and needs 400+ mm/s outbound."""
    def sweep(z0, z1, mm_s):
        m = machine(7.5)
        clk = m._now
        i = [0]

        def step(pos_mm, z):
            i[0] += 1
            clk.t += 1.0 / 7.5
            img = project(hand3d(_curls(1.0)), (pos_mm, 10., z), 35.)
            hands = () if img is None else (observe_hand(img, EYE_PX),)
            return m.update(hands, EYE_PX, i[0], looking=True)
        step(0.0, z0)
        for _ in range(6):
            e = step(0.0, z0)
            if e is not None and e.kind == "grab":
                break
        peak = 0.0
        x = 0.0
        z = z0
        for k in range(1, 7):
            x += mm_s / 7.5
            z = z0 + (z1 - z0) * k / 6.0
            step(x, z)
            peak = max(peak, m.status()["speed_us"])
        return peak
    inbound = sweep(425.0, 300.0, 300.0)
    outbound = sweep(425.0, 550.0, 300.0)
    print("\nsame 300 mm/s sweep: toward the lens %.2f u/s, away %.2f u/s"
          % (inbound, outbound))
    assert abs(inbound - outbound) / max(inbound, 1e-9) <= 0.20, \
        (inbound, outbound)


# ================================ BUG 2: the fling grant is 1.07 s, not 0.55 s
def test_the_fling_grant_is_the_documented_window_and_not_more():
    """``fling_window_s`` is documented as 0.55 s. MEASURED at 53537c7 and
    7.5 fps: one fast frame, then 4 crawling frames (0.53 s at a fifth of
    the bar), then 3 absent frames (0.40 s) and the exit STILL resolved as
    a throw -- a grant of 1.07 s, because the window is measured to the
    last SEEN frame and ``lost_grace_frames`` is added on top of it."""
    grants = {}
    for fps in (7.5, 6.0, 3.7):
        worst = 0.0
        for slow in range(0, 9):
            m = machine(fps)
            step = grab_at(m, 900.0)
            cx = 900.0 + 1.00 * UNIT
            t_fast = m._now.t + 1.0 / fps
            step((obs(cx),))
            for _ in range(slow):
                cx += 0.08 * UNIT
                step((obs(cx),))
            ev = None
            for _ in range(T.lost_grace_frames + 3):
                e = step(())
                if e is not None:
                    ev = e
                    break
            if ev is not None and ev.kind == "throw":
                worst = max(worst, m._now.t - t_fast)
        grants[fps] = worst
        print("  %.1f fps: the longest grant that still threw = %.2f s"
              % (fps, worst))
    # THE LAW, not a number: the grant is ``fling_window_s`` or the loss
    # grace plus half a frame, whichever is longer -- see
    # CastGesture.fling_grant_s. Below about 5.5 fps the floor is what
    # binds, because an exit cannot fire sooner than the grace.
    for fps, worst in grants.items():
        grant = machine(fps).fling_grant_s()
        print("  %.1f fps: the law says %.2f s" % (fps, grant))
        assert worst <= grant + 1e-6, (fps, worst, grant)


# =========================== BUG 3: a second hand adopted mid-carry as travel
def test_a_second_candidate_hand_can_never_supply_the_fling():
    """``_pick`` relaxed to nearest-centroid during a carry with no
    identity check at all, so his OTHER hand appearing near the carried
    one was taken as it and the jump measured as one frame of travel --
    3.75 to 18.0 u/s, up to six times the bar, with no hand having moved
    fast (MEASURED at 53537c7).

    A frame with more than one candidate in the association gate is now
    an AMBIGUOUS frame: the nearest is still followed, so the carry
    lives, but that frame's movement is not fling evidence.
    """
    for jump_u in (0.5, 1.0, 1.5, 1.9):
        m = machine(7.5)
        step = grab_at(m, 640.0)
        step((obs(650.0),))
        other = 650.0 + jump_u * UNIT
        step((obs(other), obs(650.0 + 0.30 * UNIT)))
        st = m.status()
        print("  a second candidate %.1f u away: speed read %.2f u/s, "
              "flung=%s" % (jump_u, st["speed_us"], st["flung"]))
        assert not st["flung"], (jump_u, st)


def test_a_single_hand_teleporting_is_not_travel_either():
    m = machine(7.5)
    step = grab_at(m, 640.0)
    step((obs(650.0),))
    step((obs(650.0 + 2.4 * UNIT),))
    print("\n  one hand, 2.4 u in a frame: flung=%s" % m.status()["flung"])
    assert not m.status()["flung"]


def test_the_residual_that_no_check_here_can_close():
    """STATED RATHER THAN HIDDEN. With ONE candidate in the gate, at the
    same scale, half a unit from where the carried hand was, there is
    nothing in this module that distinguishes "his other hand appeared
    there" from "his hand moved there". 0.5 u between frames at 7.5 fps
    is 3.75 u/s, which is an ordinary fling speed. Closing it needs a
    tracker that carries identity between frames, which the detector does
    not provide."""
    m = machine(7.5)
    step = grab_at(m, 640.0)
    step((obs(650.0),))
    step((obs(650.0 + 0.5 * UNIT),))
    st = m.status()
    print("\n  one candidate, 0.5 u away: speed %.2f u/s, flung=%s -- and "
          "that is correct for a hand that really did move"
          % (st["speed_us"], st["flung"]))
    assert st["flung"] is True


# ================================================= the SECOND signal: his yaw
def test_a_throw_needs_him_to_have_been_looking_at_the_source_screen():
    """One scalar cannot separate his throw from his desk motion; that is
    measured, not opinion. The second signal is where his head was
    pointing through the grab AND through the swing."""
    def run(looking):
        m = machine(7.5)
        step = grab_at(m, 640.0, looking=True)
        cx = 640.0
        for _ in range(3):
            cx += 1.10 * UNIT
            step((obs(cx),), looking=looking)
        ev = None
        for _ in range(4):
            e = step((), looking=looking)
            if e is not None:
                ev = e
                break
        return ev
    kept = run(True)
    turned = run(False)
    print("\n  head held on the source screen : %s" % kept.kind)
    print("  head turned away through the swing: %s (%s)"
          % (turned.kind, turned.why))
    assert kept is not None and kept.kind == "throw", kept
    assert turned is not None and turned.kind == "drop", turned


def test_no_opinion_about_his_head_is_configurable_and_defaults_to_refusing():
    """A frame with no clean face has NO OPINION about his yaw, and that
    is not the same as him looking at the screen. Which way that falls is
    his to set from assistant.json; it refuses by default."""
    from jarvis.handstage import thresholds_from_options
    assert CastThresholds().yaw_required is True
    loose = thresholds_from_options({"gesture.yaw_required": False}.get)
    assert loose.yaw_required is False

    def run(t):
        m = machine(7.5, t=t)
        step = grab_at(m, 640.0, looking=None)
        cx = 640.0
        for _ in range(3):
            cx += 1.10 * UNIT
            step((obs(cx),), looking=None)
        for _ in range(4):
            e = step((), looking=None)
            if e is not None:
                return e
        return None
    strict = run(CastThresholds())
    relaxed = run(loose)
    print("\n  no face row, yaw_required=True : %s (%s)"
          % (strict.kind, strict.why))
    print("  no face row, yaw_required=False: %s" % relaxed.kind)
    assert strict.kind == "drop"
    assert relaxed.kind == "throw"


def test_every_new_threshold_reaches_the_config_file():
    from jarvis.handstage import thresholds_from_options
    keys = {"closed_ratio_max": 0.61, "open_ratio_min": 1.4,
            "throw_speed_us": 2.4, "assoc_step_u": 0.9,
            "assoc_scale_frac": 0.4, "yaw_hold_deg": 30.0}
    got = thresholds_from_options({"gesture." + k: v
                                   for k, v in keys.items()}.get)
    for k, v in keys.items():
        assert getattr(got, k) == v, k


# ============================ where the head angle is actually measured
class _Face:
    """One ``PreviewFace``-shaped row. Numbers only; no frame, ever."""

    def __init__(self, yaw_deg=0.0, eye_px=EYE, ok=True, attending=True):
        self.yaw_deg = float(yaw_deg)
        self.eye_px = float(eye_px)
        self.landmarks_ok = bool(ok)
        self.attending = bool(attending)


class _Row:
    def __init__(self, lm, conf=0.9):
        self.lm = lm
        self.conf = conf


class _Tracker:
    """A fake hand tracker. It is handed a plain integer as the 'frame'
    and returns whatever landmark rows the test queued; it cannot open
    anything and there is nothing for it to open."""

    def __init__(self, queue):
        self.queue = queue

    def detect(self, _frame):
        return self.queue[0]


def _stage(queue, clock):
    from jarvis.handstage import HandStage
    g = CastGesture(T, (1280, 720), now=clock, preview_fps=7.5)
    return HandStage(g, get_option={"gesture.enabled": True}.get,
                     make_tracker=lambda: _Tracker(queue), now=clock)


def test_the_hand_stage_reads_his_head_from_before_the_reach():
    """The reference angle is latched while the machine is IDLE and frozen
    for the life of a reach: ``handstage`` already says the reaching arm
    crosses his face at exactly the moment the gesture matters, which is
    why attention is latched rather than sampled, and the same applies to
    the angle."""
    clock = Clock()
    queue = [()]
    stage = _stage(queue, clock)
    fist = landmarks(1.0, 425, 35)
    openh = landmarks(0.06, 425, 35)

    def frame(rows, yaw, ok=True):
        clock.t += 1.0 / 7.5
        queue[0] = rows
        return stage.observe(object(), [_Face(yaw, ok=ok)], 1280, 720)

    shot = frame((), 0.0)                       # resting, head straight on
    assert shot.looking == "yes"
    for _ in range(4):
        shot = frame((_Row(openh),), 2.0)       # reaching, head steady
    assert shot.looking == "yes"
    shot = frame((_Row(fist),), 40.0)           # ...and now he turns away
    print("\n  head 40 deg off the latched angle: looking=%r" % shot.looking)
    assert shot.looking == "no"
    # A single frame with no clean face is not "no opinion": the newest
    # clean sample inside YAW_STALE_S still counts, which is the whole
    # reason that window exists -- the reaching arm crosses his face.
    shot = frame((_Row(fist),), 40.0, ok=False)
    print("  one frame with no face (a stale angle still counts): %r"
          % shot.looking)
    assert shot.looking == "no"
    # ...but once the window has run out there is genuinely nothing to
    # read, and that is "" -- no opinion, not a yes.
    from jarvis.handstage import YAW_STALE_S
    clock.t += YAW_STALE_S + 1.0
    stage.gesture.stall_s = 1e9          # don't let the stall reset it here
    shot = frame((_Row(fist),), 40.0, ok=False)
    print("  the window has run out:            looking=%r" % shot.looking)
    assert shot.looking == ""
    st = stage.status()
    assert st["look_reads"] > 0 and st["look_miss_pct"] >= 0.0
    print("  the instrument he can read in his own room: %d reads, "
          "%.1f%% with no opinion" % (st["look_reads"], st["look_miss_pct"]))


def test_the_hand_stage_reports_the_angle_as_a_string_and_nothing_else():
    """The scalars-only discipline: what crosses out of the stage is
    "yes", "no" or "", never an angle a router could act on and never
    anything of the frame."""
    from jarvis.handstage import HandShot
    row = HandShot(looking="yes").as_dict()
    assert row["looking"] == "yes"
    assert all(isinstance(v, (int, float, str, bool))
               for v in row.values()), row
