"""Where his screens are, learned rather than told -- the decision layer
(jarvis/screens.py).

Every row in this file is SYNTHETIC: hand centroids and head-yaw ratios this
process made up, pushed through the same arithmetic the live courier uses.
No camera is opened, no frame is decoded, no landmark ever came off a lens.
That is not a limitation of the test, it is the method: the whole feature is
verifiable from numbers, and a test that could only be passed by looking at
a picture would be the wrong test.

WHAT IS PINNED HERE, and why each needs a test rather than a comment:

* HIS TWO EXAMPLES. "Pull from the Spark and throw to the middle" casts the
  Spark to HPCOMPUTER; "pull from HPCOMPUTER and throw to the right" casts
  HPCOMPUTER to the Spark. Both, by name, on a map with THREE screens.
* THE OTHER TWO ROWS REFUSE. There are four (source, direction) rows and
  only two of them can fire. The two valid directions are DISJOINT between
  the two machines, which is why a one-step source misread lands on a
  refusal and never on the wrong desktop.
* A MAP THAT CANNOT TELL TWO MACHINES APART DOES NOT ARM, and says which
  pair beat it. A pair that is too close but sits INSIDE one machine costs
  nothing and must not block: middle and left are both HPCOMPUTER.
* THE DESTINATION SET IS TWO MACHINES, MECHANICALLY. The disjoint-direction
  safety argument dies the moment a third target is registered, so a third
  machine refuses to arm rather than quietly making the analysis void.
* NOTHING LEARNED, NOTHING ROUTED. An empty or unarmed map scores no
  machine at all, and the courier's fallback is the board.
* THE STORED FORM IS NUMBERS AND SHORT NAMES ONLY -- it goes in his config,
  so ``assert_numbers_only`` is run over it.
"""
from __future__ import annotations

import math

import pytest

from jarvis import screens as sc
from jarvis.visionrig import NOSE_RATIO, assert_numbers_only

# --------------------------------------------------------------- fixtures
# One synthetic desk. Hand medians are in HAND-UNITS in HIS frame (+ is his
# right), yaw in RAW yaw_t (the nose-tip ratio; negative is turned to his
# right). The Spark is his right-hand screen, HPCOMPUTER drives the other
# two. INVENTED, not measured on him -- these are the geometry the design's
# arithmetic predicts, used here to exercise the decision, not to claim it.
RIGHT = dict(name="right", machine=sc.SPARK,
             hand_x_u_p50=1.34, hand_x_u_iqr=0.40,
             yaw_t_p50=-0.30, yaw_t_iqr=0.07, n=210)
MIDDLE = dict(name="middle", machine=sc.HPCOMPUTER,
              hand_x_u_p50=-0.63, hand_x_u_iqr=0.40,
              yaw_t_p50=0.02, yaw_t_iqr=0.07, n=380)
LEFT = dict(name="left", machine=sc.HPCOMPUTER,
            hand_x_u_p50=-1.90, hand_x_u_iqr=0.40,
            yaw_t_p50=0.34, yaw_t_iqr=0.07, n=140)


def screen(**kw):
    row = dict(RIGHT)
    row.update(kw)
    return sc.Screen(**row)


def three(**kw):
    """His desk: three screens, two machines."""
    return sc.build([sc.Screen(**RIGHT), sc.Screen(**MIDDLE),
                     sc.Screen(**LEFT)], at=1000.0, source="passive", **kw)


# ------------------------------------------------------ the four-row table
class TestHisTwoExamples:
    """Hunter, verbatim: "if i pull from the spark (right screen) and throw
    to the middle, it will cast the spark to HPCOMPUTER. if i pull from
    theHPCOMPUTER (middle screen) and throw to the right then it will cast
    HPCOMPTUER to spark". The grab names the SOURCE, the throw names the
    DESTINATION -- and with three screens on two machines that reduces to
    two firing rows out of four."""

    def test_the_map_has_three_screens_and_two_machines(self):
        m = three()
        assert [s.name for s in m.screens] == ["right", "middle", "left"]
        assert m.machines == (sc.HPCOMPUTER, sc.SPARK)
        assert m.armed, m.reason

    def test_grab_at_the_spark_throw_left_casts_the_spark_to_hpcomputer(self):
        m = three()
        v = sc.score(m, hand_u=1.30)
        assert v.machine == sc.SPARK
        assert sc.route(v.machine, "left") == sc.HPCOMPUTER

    def test_grab_at_hpcomputer_throw_right_casts_hpcomputer_to_the_spark(self):
        m = three()
        v = sc.score(m, hand_u=-0.60)
        assert v.machine == sc.HPCOMPUTER
        assert sc.route(v.machine, "right") == sc.SPARK

    def test_a_grab_at_the_far_left_screen_is_still_hpcomputer(self):
        """The third screen is not a third destination. He reaches at the
        left-hand monitor and the SOURCE is the machine driving it."""
        m = three()
        assert sc.score(m, hand_u=-1.85).machine == sc.HPCOMPUTER

    def test_the_other_two_rows_refuse_because_there_is_nothing_that_way(self):
        assert sc.route(sc.SPARK, "right") == ""
        assert sc.route(sc.HPCOMPUTER, "left") == ""

    def test_the_valid_directions_are_disjoint_so_a_source_flip_refuses(self):
        """THE WHOLE SAFETY ARGUMENT, enumerated rather than asserted. A
        one-step source misread always flips into the invalid direction for
        the machine it was misread as, and lands on a refusal. There is no
        combination that shows the wrong desktop."""
        fired = {}
        for src in sc.MACHINES:
            for word in ("left", "right"):
                dest = sc.route(src, word)
                if dest:
                    fired[(src, word)] = dest
        assert fired == {(sc.SPARK, "left"): sc.HPCOMPUTER,
                         (sc.HPCOMPUTER, "right"): sc.SPARK}
        for (src, word), _dest in fired.items():
            other = [m for m in sc.MACHINES if m != src][0]
            assert sc.route(other, word) == "", (
                "a source misread as %s throwing %s must refuse" % (other, word))

    def test_up_and_down_and_nonsense_never_route(self):
        for word in ("up", "down", "ambiguous", "", "LEFTish", None):
            assert sc.route(sc.SPARK, word) == ""
            assert sc.route(sc.HPCOMPUTER, word) == ""

    def test_a_machine_that_is_not_one_of_the_two_never_routes(self):
        assert sc.route("handoff", "left") == ""
        assert sc.route("", "right") == ""


# -------------------------------------------------------- refusing to arm
class TestArming:
    """Refusing to arm is the right answer and it is better than a coin
    flip. It is cheap here in a way it usually is not, because the fallback
    -- the board -- is a real, working destination rather than an error."""

    def test_two_screens_of_the_SAME_machine_may_sit_on_top_of_each_other(self):
        """Middle and left are both HPCOMPUTER. If he reaches at them the
        same way, nothing is lost: the design never needed to tell them
        apart, and blocking the whole feature over it would be an
        overreaction."""
        m = sc.build([sc.Screen(**RIGHT), sc.Screen(**MIDDLE),
                      sc.Screen(**dict(LEFT, hand_x_u_p50=-0.66,
                                       yaw_t_p50=0.03))], at=1000.0)
        assert m.armed, m.reason
        assert ("middle", "left") in m.unseparated or \
               ("left", "middle") in m.unseparated

    def test_two_screens_ACROSS_the_boundary_that_are_too_close_do_not_arm(self):
        """Right (Spark) and middle (HPCOMPUTER) overlapping on BOTH axes is
        the pair that matters, and the map must refuse rather than guess."""
        m = sc.build([sc.Screen(**RIGHT),
                      sc.Screen(**dict(MIDDLE, hand_x_u_p50=1.20,
                                       yaw_t_p50=-0.29)),
                      sc.Screen(**LEFT)], at=1000.0)
        assert not m.armed
        assert "right" in m.reason and "middle" in m.reason

    def test_it_says_so_in_words_he_will_hear(self):
        m = sc.build([sc.Screen(**RIGHT),
                      sc.Screen(**dict(MIDDLE, hand_x_u_p50=1.20,
                                       yaw_t_p50=-0.29)),
                      sc.Screen(**LEFT)], at=1000.0)
        line = sc.unarmed_line(m)
        assert line and line.endswith("sir.") or "sir" in line
        assert "board" in line

    def test_a_pair_separated_on_ONE_axis_is_enough(self):
        """Two weak independent signals beat one: overlapping hand clusters
        still arm when the head yaw tells them apart."""
        m = sc.build([sc.Screen(**RIGHT),
                      sc.Screen(**dict(MIDDLE, hand_x_u_p50=1.20)),
                      sc.Screen(**LEFT)], at=1000.0)
        assert m.armed, m.reason

    def test_a_third_machine_refuses_to_arm(self):
        """THE MECHANICAL PIN. The disjoint-direction safety property dies
        the moment there is a third target -- a second Windows box, the
        handoff page promoted to a destination. If this test ever fails,
        the entire misfire analysis is void and has to be redone."""
        m = sc.build([sc.Screen(**RIGHT), sc.Screen(**MIDDLE),
                      sc.Screen(**dict(LEFT, machine="laptop"))], at=1000.0)
        assert not m.armed
        assert "two machines" in m.reason

    def test_one_machine_alone_refuses_to_arm(self):
        m = sc.build([sc.Screen(**MIDDLE), sc.Screen(**LEFT)], at=1000.0)
        assert not m.armed
        assert "two machines" in m.reason

    def test_too_few_samples_refuses_to_arm(self):
        m = sc.build([sc.Screen(**dict(RIGHT, n=9)), sc.Screen(**MIDDLE),
                      sc.Screen(**LEFT)], at=1000.0)
        assert not m.armed
        assert "25" in m.reason

    def test_an_empty_map_is_not_armed_and_scores_nothing(self):
        m = sc.build([], at=1000.0)
        assert not m.armed
        assert sc.score(m, hand_u=1.3).machine == ""
        assert sc.unarmed_line(m)

    def test_a_degenerate_spread_cannot_manufacture_confidence(self):
        """Two clusters 0.02 units apart with a zero IQR would divide by
        nothing and read as infinitely separated. The sigma floor is what
        stops a map learned from three identical frames arming itself."""
        m = sc.build([sc.Screen(**dict(RIGHT, hand_x_u_p50=0.01,
                                       hand_x_u_iqr=0.0, yaw_t_iqr=0.0,
                                       yaw_t_p50=0.0)),
                      sc.Screen(**dict(MIDDLE, hand_x_u_p50=-0.01,
                                       hand_x_u_iqr=0.0, yaw_t_iqr=0.0,
                                       yaw_t_p50=0.0)),
                      sc.Screen(**dict(LEFT, hand_x_u_iqr=0.0,
                                       yaw_t_iqr=0.0))], at=1000.0)
        assert not m.armed


# ----------------------------------------------------------- the margin
class TestMargin:
    def test_a_grab_near_the_boundary_has_no_confident_source(self):
        m = three()
        v = sc.score(m, hand_u=m.boundary_u)
        assert v.margin_sigma == pytest.approx(0.0, abs=1e-6)
        assert not v.ok

    def test_a_grab_at_a_cluster_centre_is_well_clear(self):
        m = three()
        assert sc.score(m, hand_u=1.34).margin_sigma >= sc.MIN_MARGIN_SIGMA
        assert sc.score(m, hand_u=-0.63).margin_sigma >= sc.MIN_MARGIN_SIGMA

    def test_the_boundary_sits_between_the_two_machines(self):
        m = three()
        assert -0.63 < m.boundary_u < 1.34

    def test_yaw_is_read_when_it_is_there_and_abstains_when_it_is_not(self):
        m = three()
        without = sc.score(m, hand_u=1.10)
        with_yaw = sc.score(m, hand_u=1.10, yaw_t=-0.30)
        assert not without.yaw_used
        assert with_yaw.yaw_used
        assert with_yaw.margin_sigma >= without.margin_sigma

    def test_the_head_and_the_hand_disagreeing_refuses_the_grab(self):
        """The disagreement tripwire, at the moment of use. He reaches at
        the Spark and looks at the left-hand screen: that grab is refused
        and counted, because a map learned from a skewed label set would
        look confident and be wrong."""
        m = three()
        v = sc.score(m, hand_u=1.30, yaw_t=0.34)
        assert not v.ok
        assert not v.agreed
        assert v.machine == ""

    def test_a_weak_yaw_reading_does_not_veto_a_strong_hand(self):
        """Yaw abstains rather than vetoing when it is itself near ITS own
        boundary; only a confident disagreement refuses."""
        m = three()
        v = sc.score(m, hand_u=1.34, yaw_t=m.boundary_t)
        assert v.ok
        assert v.machine == sc.SPARK


# ----------------------------------------------------------- the geometry
class TestGeometry:
    def test_hand_units_are_signed_into_HIS_frame_and_nowhere_else(self):
        """Image +x is HIS LEFT because the lens faces him. This module
        must not own a second copy of that sign -- it goes through
        gesture.to_his_frame, and this is the test that would catch the
        failure that produced correct magnitudes and the wrong side of the
        room, twice."""
        frame_w, palm = 1280.0, 296.0
        his_right = sc.hand_x_u(cx=243.0, palm_diag=palm, frame_w=frame_w)
        his_left = sc.hand_x_u(cx=1037.0, palm_diag=palm, frame_w=frame_w)
        assert his_right > 0.0
        assert his_left < 0.0
        assert his_right == pytest.approx(-his_left, abs=1e-9)

    def test_a_mirrored_feed_flips_the_side(self):
        a = sc.hand_x_u(cx=243.0, palm_diag=296.0, frame_w=1280.0)
        b = sc.hand_x_u(cx=243.0, palm_diag=296.0, frame_w=1280.0,
                        mirrored=True)
        assert a == pytest.approx(-b)

    def test_a_collapsed_hand_has_no_opinion_rather_than_an_infinite_one(self):
        assert sc.hand_x_u(cx=0.0, palm_diag=0.0, frame_w=1280.0) == 0.0

    def test_yaw_t_is_recovered_from_yaw_deg_exactly(self):
        """The classifier works in RAW yaw_t, never degrees, because
        NOSE_RATIO is documented in visionrig.py as an assumption about
        adult anatomy rather than a measurement of him. PreviewFace carries
        only the derived degrees, so the inverse must be exact."""
        for t in (-0.6, -0.31, 0.0, 0.02, 0.45):
            deg = math.degrees(math.atan(t / NOSE_RATIO))
            assert sc.yaw_t_from_deg(deg) == pytest.approx(t, abs=1e-9)

    def test_looking_to_his_right_reads_negative(self):
        assert sc.yaw_t_from_deg(-40.0) < 0.0
        assert sc.yaw_t_from_deg(40.0) > 0.0


# -------------------------------------------------------------- the store
class TestStorage:
    def test_the_stored_form_is_numbers_and_short_names_only(self):
        """It goes in his config file. Nothing that could reconstruct what
        he was doing may be in it: no frame, no window title, no
        per-minute activity trace."""
        raw = sc.to_config(three(layout="0,1920,1920,1920", probe_r=0.11))
        assert_numbers_only(raw)

    def test_a_map_survives_a_round_trip(self):
        m = three(layout="0,1920,1920,1920")
        back = sc.from_config(sc.to_config(m))
        assert back is not None
        assert back.armed == m.armed
        assert back.layout == m.layout
        assert [s.name for s in back.screens] == [s.name for s in m.screens]
        assert back.boundary_u == pytest.approx(m.boundary_u)

    def test_rubbish_in_the_config_reads_as_nothing_learned(self):
        for raw in (None, {}, [], "spark", {"screens": "left"},
                    {"version": 99, "screens": []},
                    {"screens": [{"name": "x"}]}):
            assert sc.from_config(raw) is None

    def test_a_stored_map_that_would_not_arm_today_is_not_trusted(self):
        """Armed is RE-DERIVED on load, never taken on trust from the file.
        A map hand-edited to armed:true with one machine in it must not
        route."""
        raw = sc.to_config(three())
        raw["screens"] = [s for s in raw["screens"] if s["machine"] != sc.SPARK]
        raw["armed"] = True
        back = sc.from_config(raw)
        assert back is None or not back.armed

    def test_the_layout_string_is_the_tripwire(self):
        """An exact signal that he unplugged, added or moved a monitor."""
        m = three(layout="0,1920,1920,1920")
        assert not sc.layout_changed(m, "0,1920,1920,1920")
        assert sc.layout_changed(m, "0,1920,1920,1920,3840,1080")
        assert not sc.layout_changed(m, "")          # nothing said, no news


# ------------------------------------------------------------ the learner
class TestLearner:
    """The passive path: he says nothing and works across the three screens
    while the learner watches which machine he is on and where his head is
    pointed. Medians and IQRs, never means, so a minority of wrong labels
    moves nothing."""

    def test_a_label_must_hold_still_before_a_sample_counts(self):
        lrn = sc.ScreenLearner()
        # the mouse flicks onto the Spark for half a second
        lrn.observe(at=0.0, label="right", hand_u=1.3, yaw_t=-0.30)
        lrn.observe(at=0.5, label="right", hand_u=1.3, yaw_t=-0.30)
        assert lrn.counts() == {}
        lrn.observe(at=3.1, label="right", hand_u=1.3, yaw_t=-0.30)
        assert lrn.counts() == {"right": 1}

    def test_a_label_change_restarts_the_gate(self):
        lrn = sc.ScreenLearner()
        for t in (0.0, 3.1, 3.2):
            lrn.observe(at=t, label="right", hand_u=1.3, yaw_t=-0.3)
        lrn.observe(at=3.3, label="middle", hand_u=-0.6, yaw_t=0.02)
        assert lrn.counts() == {"right": 2}
        lrn.observe(at=6.5, label="middle", hand_u=-0.6, yaw_t=0.02)
        assert lrn.counts()["middle"] == 1

    def test_a_minority_of_wrong_labels_does_not_move_the_median(self):
        lrn = sc.ScreenLearner()
        t = 0.0
        for i in range(40):
            t += 4.0
            lrn.observe(at=t, label="right", hand_u=1.30 + 0.01 * (i % 3),
                        yaw_t=-0.30)
        for i in range(6):                      # his mouse rested on Windows
            t += 4.0
            lrn.observe(at=t, label="right", hand_u=-1.9, yaw_t=0.34)
            t += 0.1
            lrn.observe(at=t, label="right", hand_u=-1.9, yaw_t=0.34)
        stat = lrn.screen("right", sc.SPARK)
        assert stat.hand_x_u_p50 == pytest.approx(1.31, abs=0.02)

    def test_it_will_not_conclude_on_too_few_grabs(self):
        lrn = sc.ScreenLearner()
        t = 0.0
        for i in range(40):
            t += 4.0
            lrn.observe(at=t, label="middle", hand_u=-0.63 + 0.02 * (i % 3),
                        yaw_t=0.02)
        for _ in range(5):
            t += 4.0
            lrn.observe(at=t, label="right", hand_u=1.3, yaw_t=-0.3)
        m = lrn.build({"right": sc.SPARK, "middle": sc.HPCOMPUTER}, at=t)
        assert not m.armed
        assert "25" in m.reason

    def test_a_full_passive_session_arms(self):
        lrn = sc.ScreenLearner()
        t = 0.0
        plan = {"right": (1.34, -0.30), "middle": (-0.63, 0.02),
                "left": (-1.90, 0.34)}
        for name, (hand, yaw) in plan.items():
            for i in range(30):
                t += 4.0
                lrn.observe(at=t, label=name,
                            hand_u=hand + 0.02 * ((i % 5) - 2),
                            yaw_t=yaw + 0.01 * ((i % 3) - 1))
        m = lrn.build({"right": sc.SPARK, "middle": sc.HPCOMPUTER,
                       "left": sc.HPCOMPUTER}, at=t, source="passive")
        assert m.armed, m.reason
        assert sc.score(m, hand_u=1.30).machine == sc.SPARK
        assert sc.score(m, hand_u=-0.60).machine == sc.HPCOMPUTER

    def test_the_learner_holds_no_trace_of_what_he_was_doing(self):
        lrn = sc.ScreenLearner()
        lrn.observe(at=0.0, label="right", hand_u=1.3, yaw_t=-0.3)
        lrn.observe(at=4.0, label="right", hand_u=1.3, yaw_t=-0.3)
        assert_numbers_only(lrn.numbers_only())

    def test_an_unknown_machine_for_a_label_is_dropped_not_invented(self):
        lrn = sc.ScreenLearner()
        t = 0.0
        for _ in range(30):
            t += 4.0
            lrn.observe(at=t, label="telly", hand_u=0.2, yaw_t=0.1)
        m = lrn.build({}, at=t)
        assert not m.armed
        assert m.screens == ()
