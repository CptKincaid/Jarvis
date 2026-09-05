"""His two gestures, end to end: grab a screen, throw it at the other
machine (jarvis/screens.py + jarvis/castview.py through
jarvis/gesturecast.py).

Hunter, verbatim: "if i pull from the spark (right screen) and throw to the
middle, it will cast the spark to HPCOMPUTER. if i pull from theHPCOMPUTER
(middle screen) and throw to the right then it will cast HPCOMPTUER to
spark". This file drives exactly those two motions through the real hand
stage and the real courier and checks what came out.

EVERY LANDMARK ROW HERE IS SYNTHETIC. The hand is the one
tests/test_gesture.py builds from MediaPipe's published proportions,
projected through his own lens constants (LifeCam Cinema, 65.6 deg across
1280 px) by arithmetic in this process. No camera is opened, no frame is
decoded, no preview window exists and nothing is looked at. The screen map
is built from the SAME projection, so the two grab positions in it are
measured off the fixture rather than asserted -- which is the point of
``test_the_two_anchors_are_where_the_projection_puts_them``.

NOTHING STARTS A RUSTDESK SESSION. Both launchers are recorders, the relay
is a plain object, and no socket is opened.

WHAT IS PINNED HERE:

* HIS TWO GESTURES, by name, with THREE screens configured, and the two
  opposite throws REFUSING because there is nothing that way.
* OFF BY DEFAULT, and every way the preview shuts shuts this too.
* A CAST CANNOT FIRE WHILE ONE IS UP, and a fling at the desk stops it.
* AN UNLEARNED OR TOO-CLOSE MAP falls back to the board and says so once.
"""
from __future__ import annotations

import pytest

from jarvis import castview as cv
from jarvis import gesturecast as gc_mod
from jarvis import screens as sc
from jarvis.gesture import observe_hand
from jarvis.visionrig import assert_numbers_only
from tests.test_gesture import EYE_PX, W, _curls, hand3d, project, seg
from tests.test_gesturecast import STEP, build, drive, face

# ---------------------------------------------------------------- the desk
# Two grab positions in millimetres, in the camera frame, at a 425 mm reach.
# World -x is HIS RIGHT (the lens faces him), so the Spark -- his right-hand
# screen, from xrandr -- is the negative one.
SPARK_X = -170.0
MIDDLE_X = 80.0
LEFT_X = 200.0
# The placement spread the map is built with. ASSUMED, not measured on him:
# this repo has never measured how accurately he puts his hand at a named
# target, and scripts/screen_selfcheck.py is the instrument that would.
# 0.40 units of IQR is a deliberately pessimistic stand-in.
IQR_U = 0.40
IQR_T = 0.09
# Head yaw, as RAW yaw_t. Also assumed: how much of a 41 deg gaze shift he
# takes with his head rather than his eyes has never been measured either.
YAW = {"right": -0.30, "middle": 0.02, "left": 0.34}


def anchor_u(world_x: float) -> float:
    """The lateral hand position, in hand-units in HIS frame, for a fist
    held at ``world_x``. Pure arithmetic over the synthetic hand and his
    lens; this is the fixture the map below is built from."""
    img = project(hand3d(_curls(1.0)), (world_x, 10.0, 425.0), 35.0, 0.0, 0.0)
    assert img is not None, "the fixture hand must be in frame"
    obs = observe_hand(img, EYE_PX)
    return sc.hand_x_u(obs.cx, obs.palm_diag, W)


def desk_map(n: int = 200, **kw) -> sc.ScreenMap:
    """His desk: three screens, two machines, built from the projection."""
    return sc.build([
        sc.Screen("right", sc.SPARK, anchor_u(SPARK_X), IQR_U,
                  YAW["right"], IQR_T, n),
        sc.Screen("middle", sc.HPCOMPUTER, anchor_u(MIDDLE_X), IQR_U,
                  YAW["middle"], IQR_T, n),
        sc.Screen("left", sc.HPCOMPUTER, anchor_u(LEFT_X), IQR_U,
                  YAW["left"], IQR_T, n),
    ], at=1000.0, source="passive", layout="0,1920,1920,1920", **kw)


def screen_gesture(anchor_x: float, end_x: float, pitch: float = 35.0):
    """Reach in open, close at ``anchor_x``, hold still, fling to
    ``end_x``, open while following through. The same shape as
    tests/test_gesture.py's ``gesture()``, with the GRAB POSITION as a
    parameter -- which is the whole thing this lane decides on."""
    a = (anchor_x, 10.0, 425.0)
    b = (anchor_x + 5.0, 5.0, 425.0)
    end = (end_x, -10.0, 450.0)
    far = tuple(1.35 * v for v in end)
    return (0.55 + 0.20 + 0.50 + 0.40 + 0.35 + 0.5, lambda t: seg(t, [
        (0.55, (anchor_x + 130.0, 300.0, 780.0), a, 0.10, 0.06, pitch - 5, 0, 0),
        (0.20, a, a, 0.06, 1.00, pitch, 0, 0),
        (0.50, a, b, 1.00, 1.00, pitch, 0, 0),
        (0.40, b, end, 1.00, 1.00, pitch, 0, 0),
        (0.35, end, far, 1.00, 0.06, pitch, 0, 0)]))


# His two examples, and their two mirror images that must refuse.
#
# MEASURED WHILE BUILDING THIS FIXTURE, and worth recording: from a grab at
# his RIGHT-hand screen there is under one hand-unit of lateral room left to
# throw further right -- a fling to -280 mm reached dist_u 0.94 against the
# 1.00 the release bar wants, so it came out a DROP rather than a throw. The
# refusal below therefore has to travel far enough to leave the picture (the
# exit path, whose bar is 0.25) to be a throw at all. That is the design's
# "an off-centre anchor can have under 0.3 units of room" showing up in
# practice, and it cuts the safe way: the direction with no destination is
# also the direction with no room.
FROM_SPARK_LEFT = screen_gesture(SPARK_X, 260.0)        # -> HPCOMPUTER
FROM_HP_RIGHT = screen_gesture(MIDDLE_X, -250.0)        # -> the Spark
FROM_SPARK_RIGHT = screen_gesture(SPARK_X, -340.0)      # nothing that way
FROM_HP_LEFT = screen_gesture(MIDDLE_X, 300.0)          # nothing that way


def rig(*, on=True, armed=True, helper=True, learned=True, yaw=None, **kw):
    """The courier with the screen cast wired to recorders.

    ``on`` is the config switch, ``learned`` whether a map is stored,
    ``armed`` whether that map clears the bar, ``helper`` whether the
    Windows startup script is polling.
    """
    from tests.test_gesturecast import Options
    opts = Options()
    if on:
        opts.data[sc.OPTION_ENABLED] = True
    if learned:
        m = desk_map() if armed else sc.build(
            [sc.Screen("right", sc.SPARK, 1.0, IQR_U, 0.0, IQR_T, 200),
             sc.Screen("middle", sc.HPCOMPUTER, 1.05, IQR_U, 0.01, IQR_T, 200),
             sc.Screen("left", sc.HPCOMPUTER, -1.9, IQR_U, 0.34, IQR_T, 200)],
            at=1000.0)
        opts.data[sc.OPTION_MAP] = sc.to_config(m)
    launched: list = []
    stopped: list = []
    out = build(opts=opts, view_launch=launched.append,
                view_stop=lambda: stopped.append(1), **kw)
    out.launched, out.stopped = launched, stopped
    if helper:
        out.courier.relay.note(mon=1, layout="0,1920,1920,1920",
                               at=out.clock.t)
    out.faces = lambda t: (face(),)
    return out


def faces_at(name: str):
    """Face rows with his head turned at a named screen, in DEGREES --
    which is the only angle campreview.PreviewFace carries. The courier
    converts it back to the raw ratio it was derived from."""
    import math

    from jarvis.visionrig import NOSE_RATIO
    deg = math.degrees(math.atan(YAW[name] / NOSE_RATIO))

    def rows(_t):
        f = face()
        return (type(f)(**{**f.as_dict(), "yaw_deg": deg,
                           "attending": True, "landmarks_ok": True}),)
    return rows


def throw(r, scenario, faces=None):
    """Drive one gesture through the real stage. Returns the shots."""
    return drive(r, scenario, faces=faces or r.faces, step=STEP)


# ---------------------------------------------------------- the fixture
def test_the_two_anchors_are_where_the_projection_puts_them():
    """MEASURED off the synthetic hand, not asserted: a grab at his right
    screen reads positive (his right) and one at the middle reads negative,
    and they are more than two units apart -- which is what makes the
    two-zone split the design settled on comfortable rather than marginal."""
    right, middle, left = (anchor_u(x) for x in (SPARK_X, MIDDLE_X, LEFT_X))
    assert right > 0.0
    assert middle < 0.0 and left < middle
    assert (right - middle) > 1.6
    m = desk_map()
    assert m.armed, m.reason
    assert middle < m.boundary_u < right
    assert m.margin_sigma >= sc.MIN_MARGIN_SIGMA


def test_the_map_holds_three_screens_and_only_two_machines():
    m = desk_map()
    assert len(m.screens) == 3
    assert sorted(m.machines) == sorted(sc.MACHINES)
    assert_numbers_only(sc.to_config(m))


# ------------------------------------------------------- his two gestures
class TestHisTwoGestures:
    def test_pull_from_the_spark_throw_left_casts_the_spark_to_hpcomputer(self):
        r = rig()
        shots = throw(r, FROM_SPARK_LEFT)
        assert any(s and s.event == "throw" for s in shots)
        rec = r.courier.recent()
        assert rec["sink"] == "hp-view"
        assert rec["status"] == "landed"
        assert r.courier.relay.verb == cv.VERB_SHOW
        assert "Spark" in " ".join(r.rec.spoken)
        assert r.launched == []            # nothing was opened on THIS box

    def test_pull_from_hpcomputer_throw_right_casts_hpcomputer_to_the_spark(self):
        r = rig()
        shots = throw(r, FROM_HP_RIGHT)
        assert any(s and s.event == "throw" for s in shots)
        rec = r.courier.recent()
        assert rec["sink"] == "spark-view"
        assert rec["status"] == "landed"
        assert r.launched == [cv.HPCOMPUTER_HOST]
        assert r.courier.relay.verb == cv.VERB_NONE   # Windows was not asked

    def test_the_grab_position_names_the_SOURCE_not_the_direction(self):
        """The two gestures throw in OPPOSITE directions and are both
        valid; what tells them apart is where the fist closed."""
        left_first = rig()
        throw(left_first, FROM_SPARK_LEFT)
        right_first = rig()
        throw(right_first, FROM_HP_RIGHT)
        assert left_first.courier.recent()["sink"] != \
            right_first.courier.recent()["sink"]

    def test_throwing_the_other_way_refuses_because_there_is_nothing_there(self):
        for scenario in (FROM_SPARK_RIGHT, FROM_HP_LEFT):
            r = rig()
            throw(r, scenario)
            assert r.launched == []
            assert r.courier.relay.verb == cv.VERB_NONE
            assert gc_mod.NOTHING_THAT_WAY_LINE in r.rec.spoken
            assert r.courier.refusals == 1

    def test_a_refusal_never_lands_on_the_board_instead(self):
        """He asked for a screen on another monitor. A row on the board is
        not a smaller version of that; it is a different thing."""
        r = rig()
        throw(r, FROM_SPARK_RIGHT)
        assert r.rec.shows == []
        assert r.courier.recent() is None

    def test_his_head_agreeing_with_his_hand_still_routes(self):
        r = rig()
        throw(r, FROM_SPARK_LEFT, faces=faces_at("right"))
        assert r.courier.recent()["sink"] == "hp-view"

    def test_his_head_disagreeing_with_his_hand_falls_back_to_the_board(self):
        """The tripwire at the moment of use: he reaches at the Spark and
        looks at the far left screen. That grab is refused and counted,
        because a map learned from a skewed label set would look confident
        and be wrong."""
        r = rig()
        throw(r, FROM_SPARK_LEFT, faces=faces_at("left"))
        assert r.launched == []
        assert r.courier.relay.verb == cv.VERB_NONE
        assert r.courier.recent()["sink"] == "board"


# ------------------------------------------------------- off means off
class TestOffMeansOff:
    def test_it_is_off_by_default_even_with_a_map_stored(self):
        r = rig(on=False)
        throw(r, FROM_SPARK_LEFT)
        assert r.launched == []
        assert r.courier.relay.verb == cv.VERB_NONE
        assert r.courier.recent()["sink"] == "board"

    def test_the_gesture_switch_shuts_it_too(self):
        """gesture.enabled off means the tracker is never built, so there
        is no grab, no source and nothing to route."""
        r = rig()
        r.opts.data["gesture.enabled"] = False
        throw(r, FROM_SPARK_LEFT)
        assert r.tracker.calls == 0
        assert r.launched == []
        assert r.courier.recent() is None

    def test_a_preview_that_stopped_feeding_cannot_cast(self):
        """EVERY WAY THE PREVIEW SHUTS SHUTS THIS TOO. The source is read
        from the frame the fist closed on; with no fresh frame there is no
        source, and with no source the throw is an ordinary board cast.
        A spoken throw after the camera stopped is the case that reaches
        the courier without a frame at all."""
        r = rig()
        throw(r, FROM_SPARK_LEFT)
        r.courier.stop_cast()
        r.clock.t += 60.0
        assert r.courier._read_source(r.clock.t) is None
        r.launched.clear()
        line, status = r.courier.throw_by_voice("board")
        assert status in ("landed", "held")
        assert r.launched == []

    def test_a_question_on_the_floor_blocks_the_grab_and_the_cast(self):
        r = rig()
        r.commander.open = True
        throw(r, FROM_SPARK_LEFT)
        assert r.launched == []
        assert r.courier.relay.verb == cv.VERB_NONE

    def test_a_back_turned_blocks_it_the_same_way(self):
        r = rig()
        throw(r, FROM_SPARK_LEFT, faces=lambda t: (face(attending=False),))
        assert r.launched == []
        assert r.courier.relay.verb == cv.VERB_NONE


# ------------------------------------------------------ nothing learned
class TestUnlearned:
    def test_with_no_map_every_throw_goes_to_the_board(self):
        r = rig(learned=False)
        throw(r, FROM_SPARK_LEFT)
        assert r.courier.recent()["sink"] == "board"
        assert sc.UNLEARNED_LINE in r.rec.spoken

    def test_it_says_so_once_and_not_on_every_throw(self):
        r = rig(learned=False)
        throw(r, FROM_SPARK_LEFT)
        before = r.rec.spoken.count(sc.UNLEARNED_LINE)
        r.clock.t += 5.0
        throw(r, FROM_HP_RIGHT)
        assert r.rec.spoken.count(sc.UNLEARNED_LINE) == before == 1

    def test_two_machines_it_cannot_tell_apart_do_not_arm(self):
        r = rig(armed=False)
        assert r.courier.screens is not None
        assert not r.courier.screens.armed
        throw(r, FROM_SPARK_LEFT)
        assert r.courier.recent()["sink"] == "board"
        assert any("can't tell" in line for line in r.rec.spoken)

    def test_a_grab_too_near_the_boundary_refuses_quietly(self):
        """Under two sigma of source margin the throw goes to the board and
        nothing is asked of him: he waves, hears the board tone, and no
        question is on the floor. That is the "make the gesture harder to
        trigger" option, chosen over reinstating a prompt he overruled."""
        r = rig()
        boundary_x = _world_x_for(r.courier.screens.boundary_u)
        throw(r, screen_gesture(boundary_x, 300.0))
        assert r.launched == []
        assert r.courier.relay.verb == cv.VERB_NONE
        assert r.courier.recent()["sink"] == "board"

    def test_the_layout_tripwire_disarms_the_map(self):
        r = rig()
        assert not r.courier.note_layout("0,1920,1920,1920")
        assert r.courier.note_layout("0,1920,1920,1920,3840,1080")
        assert r.courier.screens is None
        assert sc.LAYOUT_CHANGED_LINE in r.rec.spoken
        throw(r, FROM_SPARK_LEFT)
        assert r.launched == []


def _world_x_for(hand_u: float) -> float:
    """Invert ``anchor_u`` by bisection: the world x whose grab lands at a
    given hand-unit position. Used to put a grab exactly on the boundary."""
    lo, hi = -400.0, 400.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if anchor_u(mid) > hand_u:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ------------------------------------------------------ one at a time
class TestOneCastAtATime:
    def test_a_second_throw_while_a_cast_is_up_is_refused(self):
        r = rig()
        throw(r, FROM_SPARK_LEFT)
        assert r.courier.cast_live == "hp-view"
        r.clock.t += 30.0
        throw(r, FROM_HP_RIGHT)
        assert r.launched == []                       # no second window
        assert any(cv.BUSY_LINE in line for line in r.rec.spoken)

    def test_a_fling_at_the_desk_stops_the_cast(self):
        """Down is already the cancel sector, so the stop costs no new
        gesture and no new vocabulary."""
        r = rig()
        throw(r, FROM_SPARK_LEFT)
        assert r.courier.cast_live
        r.clock.t += 5.0
        throw(r, _down_gesture(SPARK_X))
        assert r.courier.cast_live == ""
        assert r.courier.relay.verb == cv.VERB_STOP
        assert cv.STOPPED_LINE in r.rec.spoken

    def test_the_spoken_stop_works_with_the_camera_off(self):
        r = rig()
        throw(r, FROM_SPARK_LEFT)
        assert r.courier.stop_cast() == cv.STOPPED_LINE
        assert r.courier.cast_live == ""
        assert r.courier.stop_cast() == cv.NOTHING_UP_LINE

    def test_the_suppression_outlasts_the_gesture_cooldown(self):
        r = rig()
        throw(r, FROM_SPARK_LEFT)
        r.courier.stop_cast()
        r.launched.clear()
        line, status = r.courier.cast_screen("spark")
        assert status == "held"                        # inside the window
        r.clock.t += cv.CAST_SUPPRESS_S + 0.1
        line, status = r.courier.cast_screen("spark")
        assert status == "landed"
        assert r.launched == [cv.HPCOMPUTER_HOST]


def _down_gesture(anchor_x: float, pitch: float = 35.0):
    """Grab and fling at the desk: the cancel, and now also the stop."""
    a = (anchor_x, 10.0, 425.0)
    end = (anchor_x + 10.0, 200.0, 445.0)
    return (2.5, lambda t: seg(t, [
        (0.55, (anchor_x + 130.0, 300.0, 780.0), a, 0.10, 0.06, pitch - 5, 0, 0),
        (0.20, a, a, 0.06, 1.00, pitch, 0, 0),
        (0.50, a, (anchor_x + 5.0, 5.0, 425.0), 1.00, 1.00, pitch, 0, 0),
        (0.40, (anchor_x + 5.0, 5.0, 425.0), end, 1.00, 1.00, pitch, 0, 0),
        (0.35, end, tuple(1.2 * v for v in end), 1.00, 0.06, pitch, 0, 0),
        (0.5, end, end, 0.06, 0.06, pitch, 0, 0)]))


# --------------------------------------------------- the spoken way in
class TestTheSpokenWayIn:
    """His ruling gives TWO ways in, the gesture and a sentence, and
    neither asks for confirmation."""

    def test_casting_by_voice_needs_no_map_and_no_camera(self):
        r = rig(learned=False)
        line, status = r.courier.cast_screen("hpcomputer")
        assert status == "landed"
        assert r.courier.relay.verb == cv.VERB_SHOW
        assert "HPCOMPUTER" in line

    def test_the_other_direction_by_voice_launches_the_viewer(self):
        r = rig(learned=False)
        line, status = r.courier.cast_screen("spark")
        assert status == "landed"
        assert r.launched == [cv.HPCOMPUTER_HOST]

    def test_with_the_switch_off_the_sentence_is_refused_too(self):
        r = rig(on=False, learned=False)
        line, status = r.courier.cast_screen("spark")
        assert status == "refused"
        assert line == gc_mod.NO_SCREEN_CAST_LINE
        assert r.launched == []

    def test_an_unknown_target_is_refused_by_name(self):
        r = rig(learned=False)
        line, status = r.courier.cast_screen("the telly")
        assert status == "refused"
        assert "telly" in line


# --------------------------------------------- the honest degradations
class TestHonestDegradation:
    def test_with_no_windows_helper_running_it_holds_and_says_why(self):
        """The reverse direction -- casting the Spark ONTO HPCOMPUTER --
        is the one that needs his startup script. Without it nothing is
        queued, so a script installed an hour later cannot suddenly act on
        an hour-old intention."""
        r = rig(helper=False)
        throw(r, FROM_SPARK_LEFT)
        assert r.courier.relay.verb == cv.VERB_NONE
        assert cv.NO_HELPER_LINE in r.rec.spoken
        assert r.courier.recent()["status"] == "held"

    def test_a_helper_that_has_gone_quiet_holds_too(self):
        r = rig(helper=True)
        r.clock.t += cv.HELPER_ALIVE_S + 1.0
        throw(r, FROM_SPARK_LEFT)
        assert r.courier.relay.verb == cv.VERB_NONE
        assert cv.NO_HELPER_LINE in r.rec.spoken

    def test_with_no_viewer_on_this_box_the_other_way_holds(self):
        r = rig()
        r.courier.registry["spark-view"]._launch = None
        throw(r, FROM_HP_RIGHT)
        assert cv.NO_VIEWER_LINE in r.rec.spoken
        assert r.courier.recent()["status"] == "held"

    def test_a_held_screen_cast_never_falls_back_to_the_board(self):
        r = rig(helper=False)
        throw(r, FROM_SPARK_LEFT)
        assert r.rec.shows == []

    def test_the_status_readout_is_numbers_only(self):
        r = rig()
        throw(r, FROM_SPARK_LEFT)
        status = r.courier.screens_status()
        assert_numbers_only(status)
        assert status["on"] is True and status["armed"] is True
        assert status["source_reads"] >= 1
        assert 0.0 <= status["yaw_miss_pct"] <= 100.0

    def test_yaw_is_read_from_BEFORE_the_fist_closed_and_abstains_when_stale(self):
        """handstage.py's own docstring says the reaching arm crosses the
        face at exactly the moment the gesture matters, which is why
        attention is LATCHED for 3 s rather than sampled. So yaw is taken
        from the last clean face row in the second before the fist closed,
        and when there is none it ABSTAINS -- it never invents zero degrees,
        which would read as looking straight at the lens."""
        r = rig()
        c = r.courier
        c.note_frame({"at": 100.0, "present": True, "cx": 243.0,
                      "palm_diag": 296.0, "frame_w": float(W),
                      "yaw_deg": -40.0, "face_ok": True})
        assert c._yaw_before(100.4) is not None
        assert c._yaw_before(101.2) is None
        fresh = c._read_source(100.4)
        assert fresh is not None and fresh.yaw_used
        stale = c._read_source(101.2)
        assert stale is not None and not stale.yaw_used
        assert c.yaw_misses == 1
        assert c.screens_status()["yaw_miss_pct"] == pytest.approx(50.0)

    def test_a_face_whose_landmarks_failed_contributes_no_angle(self):
        r = rig()
        c = r.courier
        c.note_frame({"at": 100.0, "present": True, "cx": 243.0,
                      "palm_diag": 296.0, "frame_w": float(W),
                      "yaw_deg": 0.0, "face_ok": False})
        assert c._yaw_before(100.2) is None

    def test_the_row_the_stage_hands_over_carries_no_pixel(self, monkeypatch):
        """The source is read from SCALARS the stage already reports. There
        is no landmark array, no crop and no field that could hold one --
        asserted with the same checker the self-check reports pass."""
        r = rig()
        rows: list = []
        real = r.courier.note_frame
        monkeypatch.setattr(r.courier, "note_frame",
                            lambda row: (rows.append(dict(row)), real(row)))
        r.stage._watch = r.courier.note_frame
        throw(r, FROM_SPARK_LEFT)
        assert rows
        for row in rows:
            assert_numbers_only(row)


# ------------------------------------------------- the sentence route
class TestTheCommanderPatterns:
    """The spoken way in, matched. His ruling gives two ways in and neither
    asks for confirmation; this is the one that works with the camera off.

    The collision that matters: "throw this on HPCOMPUTER" is a FILE cast
    and must not become a screen cast. Those patterns take a pronoun, these
    take a named machine, so they cannot overlap -- pinned both ways.
    """

    @staticmethod
    def rx():
        from jarvis import commander as cm
        return cm._CAST_SCREEN_RX, cm._CAST_STOP_RX, cm._CAST_THROW_RX

    @pytest.mark.parametrize("said,where", [
        ("cast the spark to HPCOMPUTER", "HPCOMPUTER"),
        ("cast the spark's screen onto hpcomputer", "hpcomputer"),
        ("show hpcomputer on the spark", "spark"),
        ("mirror the hp computer to the spark", "spark"),
        ("cast my screen to the windows machine", "windows machine"),
    ])
    def test_the_spoken_cast_is_matched_and_names_the_destination(self, said,
                                                                  where):
        screen_rx, _stop, _throw = self.rx()
        m = screen_rx.match(said)
        assert m is not None, said
        assert m.group("where").lower() == where.lower()

    @pytest.mark.parametrize("said", [
        "stop the cast", "stop casting", "close the cast", "end the mirror",
        "stop that screen cast",
    ])
    def test_the_spoken_stop_is_matched(self, said):
        _screen, stop_rx, _throw = self.rx()
        assert stop_rx.match(said) is not None, said

    @pytest.mark.parametrize("said", [
        "throw this on HPCOMPUTER", "put it on the board",
        "cast this to the spark", "drop it",
    ])
    def test_a_file_cast_never_matches_the_screen_cast(self, said):
        screen_rx, stop_rx, _throw = self.rx()
        assert screen_rx.match(said) is None, said
        assert stop_rx.match(said) is None, said

    def test_the_screen_cast_is_in_the_family_that_outranks_a_gesture(self):
        from jarvis import commander as cm
        assert cm._CAST_FAMILY_RX.match("cast the spark to hpcomputer")
        assert cm._CAST_FAMILY_RX.match("stop the cast")


# ------------------------------------------------ the instrument he runs
class TestTheSelfCheck:
    """scripts/screen_selfcheck.py -- the numbers-only instrument that turns
    the two GUESSED constants in this whole lane into measurements.

    It is tested here on rows this process invented. Nothing in the suite
    opens a camera, and the script's live path is his to run.
    """

    @staticmethod
    def script():
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "scripts" / \
            "screen_selfcheck.py"
        spec = importlib.util.spec_from_file_location("screen_selfcheck",
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_the_synthetic_run_reaches_the_pass_bar_and_prints_numbers_only(
            self, capsys):
        mod = self.script()
        code = mod.main(["--synthetic"])
        assert code == 0
        out = capsys.readouterr().out
        assert "yaw_miss_pct" in out
        assert "zones" in out

    def test_the_json_report_is_numbers_only(self, capsys):
        import json as _json
        mod = self.script()
        assert mod.main(["--synthetic", "--json"]) == 0
        report = _json.loads(capsys.readouterr().out)
        assert_numbers_only(report)
        assert report["summary"]["grabs"] > 0

    def test_it_refuses_to_conclude_on_too_few_grabs(self):
        mod = self.script()
        rows, misses, looks = mod.synthetic_rows(n=6)
        got = mod.verdict(mod.summarise(rows, misses, looks))
        samples = [r for r in got if r[0] == "samples"][0]
        assert not samples[1]

    def test_a_head_that_barely_turns_fails_the_second_axis_bar(self):
        """The pass bar was written down in advance so it could not be
        moved afterwards. This is it failing, on purpose: a neck that
        hardly moves makes the yaw clusters overlap and the second axis an
        illusion."""
        mod = self.script()
        rows, misses, looks = mod.synthetic_rows(sigma_t=0.60)
        got = dict((r[0], r[1]) for r in mod.verdict(
            mod.summarise(rows, misses, looks)))
        assert got["yaw is a 2nd axis"] is False

    def test_the_zone_table_agrees_with_the_design_arithmetic(self):
        """MEASURED in the design: two zones give a 1.143-unit half-width
        and three give 0.762, on a 4.57-unit span. Those two numbers are
        what the whole two-versus-three-zones argument rests on."""
        mod = self.script()
        z = mod.zones(0.30)
        assert z["zones_2_half_u"] == pytest.approx(1.1425, abs=1e-3)
        assert z["zones_3_half_u"] == pytest.approx(0.7617, abs=1e-3)
        assert z["zones_2_pct"] > z["zones_3_pct"]
        assert z["zones_2_pct"] > 99.9
