"""The numbers-only harness (jarvis/visionrig.py), tested with no camera.

WHAT IS BEING PINNED, and why each one is a promise the design cannot keep
by intention alone:

* **Nothing but numbers leaves the rig.** Hunter's rule is that nothing may
  look at what his camera sees, so the harness that develops this pipeline
  has to be one whose OUTPUT cannot carry an image. That is asserted
  structurally -- ``assert_numbers_only`` walks the report and rejects an
  array -- rather than by reviewing the print statements, because a print
  statement can be added later.
* **"I could not look" is never spelled "I saw nobody".** VSS shipped a
  ``region_mode='yunet'`` path that silently fell back to head_box and never
  ran once; the shape of that bug is an unavailable detector reporting zero
  detections. Here a missing model and a raising detector both produce a
  NAMED failure, and the rig's own report says so.
* **Angles come from the configured lens, never from an assumed one.** The
  same pixel is a different angle on a 65.6 deg LifeCam and an 82.2 deg
  C930e, and the shipped config's comment reasoned from 90.
* **The yaw proxy is a stated head model, not a measurement.** It is
  roll-invariant, it round-trips, and its sensitivity to the one
  anthropometric constant it assumes is pinned so nobody mistakes it for
  calibrated.

No cv2, no model weights, no /dev/video*, no display. The detector is a stub
that hands back the same 15-column rows OpenCV's FaceDetectorYN does.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from jarvis import visionrig as vr
from jarvis.facemodels import C930E, LIFECAM_CINEMA


# --------------------------------------------------------------- fixtures
def head_row(yaw_deg: float, roll_deg: float = 0.0, cx: float = 160.0,
             cy: float = 90.0, eye_px: float = 12.0,
             nose_ratio: float = 0.35, conf: float = 0.9,
             box_px: float = 40.0) -> list:
    """A YuNet row synthesised from a RIGID HEAD MODEL at a known yaw.

    Eyes at +/-0.5 interocular units on the head's own x axis, nose tip
    ``nose_ratio`` units FORWARD of the eye plane and 0.6 below it. Under
    yaw the interocular axis foreshortens by cos(yaw) and the nose swings
    by sin(yaw), which is the whole content of the proxy: the round trip
    through ``landmark_geometry`` must return the yaw that went in.
    """
    b = nose_ratio
    psi = math.radians(yaw_deg)
    pts = [(-0.5 * math.cos(psi), 0.0),          # person's right eye
           (+0.5 * math.cos(psi), 0.0),          # person's left eye
           (b * math.sin(psi), 0.6),             # nose tip
           (-0.3 * math.cos(psi), 1.0),          # right mouth corner
           (+0.3 * math.cos(psi), 1.0)]          # left mouth corner
    th = math.radians(roll_deg)
    out = [cx - box_px / 2.0, cy - box_px / 2.0, box_px, box_px]
    for x, y in pts:
        rx = (x * math.cos(th) - y * math.sin(th)) * eye_px + cx
        ry = (x * math.sin(th) + y * math.cos(th)) * eye_px + cy
        out += [rx, ry]
    out.append(conf)
    return out


class StubDetector:
    """Hands back scripted rows. Stands in for jarvis.facedetect.YuNetDetector
    exactly as ``FakeDevice`` stands in for cv2.VideoCapture next door."""

    name = "stub"

    def __init__(self, script, input_size=(320, 180), raise_on=()):
        self.script = list(script)
        self.input_size = input_size
        self.raise_on = set(raise_on)
        self.calls = 0

    def detect(self, frame):
        i = self.calls
        self.calls += 1
        if i in self.raise_on:
            raise RuntimeError("detector exploded on frame %d" % i)
        rows = self.script[i % len(self.script)] if self.script else []
        if not rows:
            return None
        return np.asarray(rows, dtype=np.float32)


class StubSource:
    """A frame source with cv2.VideoCapture's interface and no lens."""

    def __init__(self, frames=10, width=1280, height=720, fail_from=None):
        self.left = frames
        self.width, self.height = width, height
        self.fail_from = fail_from
        self.reads = 0
        self.released = 0

    def read(self):
        self.reads += 1
        if self.fail_from is not None and self.reads > self.fail_from:
            return False, None
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, np.full((self.height, self.width, 3), 100, dtype=np.uint8)

    def release(self):
        self.released += 1


def thresholds(**kw):
    base = dict(min_conf=0.6, cone_deg=20.0, cone_hysteresis_deg=5.0,
                cone_centre_deg=0.0, dwell_s=0.6, identity_min=0.363)
    base.update(kw)
    return vr.Thresholds(**base)


def rig(detector, source=None, lens=LIFECAM_CINEMA, **kw):
    clock = iter(np.arange(0.0, 10000.0, 0.125))
    return vr.Rig(source or StubSource(), detector, lens, thresholds(**kw),
                  now=lambda: float(next(clock)))


# ------------------------------------------------------- the yaw proxy
@pytest.mark.parametrize("yaw", [0.0, 5.0, -5.0, 12.0, -20.0, 35.0, -47.0])
def test_the_yaw_proxy_round_trips_the_head_model_it_assumes(yaw):
    row = head_row(yaw, nose_ratio=0.35)
    t, _roll, _eye, ok = vr.landmark_geometry(row)
    assert ok
    assert vr.HeadModel(0.35).yaw_deg(t) == pytest.approx(yaw, abs=0.01)


def test_the_yaw_proxy_is_roll_invariant():
    """A tilted head must not read as a turned one. The nose displacement is
    projected onto the INTEROCULAR direction rather than onto image x, which
    is the only reason this holds."""
    upright = vr.landmark_geometry(head_row(18.0, roll_deg=0.0))[0]
    for roll in (-30.0, -8.0, 8.0, 30.0):
        tilted = vr.landmark_geometry(head_row(18.0, roll_deg=roll))
        assert tilted[0] == pytest.approx(upright, abs=1e-4)
        assert tilted[1] == pytest.approx(roll, abs=0.01)


def test_the_yaw_proxy_is_scale_invariant():
    """A face at 60 cm and the same face at 120 cm must give the same yaw;
    the ratio is normalised by the projected interocular distance."""
    near = vr.landmark_geometry(head_row(22.0, eye_px=30.0))[0]
    far = vr.landmark_geometry(head_row(22.0, eye_px=7.0))[0]
    assert near == pytest.approx(far, abs=1e-4)


def test_the_yaw_sign_says_which_way_the_head_turned():
    assert vr.HeadModel().yaw_deg(vr.landmark_geometry(head_row(25.0))[0]) > 0
    assert vr.HeadModel().yaw_deg(vr.landmark_geometry(head_row(-25.0))[0]) < 0
    assert vr.HeadModel().yaw_deg(0.0) == 0.0


def test_the_nose_ratio_is_a_head_model_and_its_error_shows_at_the_cone_edge():
    """THE HONEST CAVEAT, pinned so it cannot be quietly forgotten. The proxy
    converts a dimensionless ratio to degrees through ONE anthropometric
    constant -- nose-tip protrusion over interocular distance -- which is
    assumed, not measured on him. A 23% error in it is ~5 deg at the 20 deg
    cone edge, i.e. a quarter of the cone, which is why the rig reports the
    raw ratio alongside the degrees and why the constant is configurable."""
    t = vr.landmark_geometry(head_row(20.0, nose_ratio=0.35))[0]
    assert vr.HeadModel(0.35).yaw_deg(t) == pytest.approx(20.0, abs=0.01)
    assumed_small = vr.HeadModel(0.27).yaw_deg(t)
    assert assumed_small > 24.0            # a smaller nose reads as more turn
    assert assumed_small - 20.0 > 4.0
    with pytest.raises(ValueError):
        vr.HeadModel(0.0)


def test_degenerate_landmarks_are_refused_not_guessed():
    row = head_row(0.0)
    row[4], row[5], row[6], row[7] = 10.0, 10.0, 10.0, 10.0   # eyes coincide
    t, roll, eye_px, ok = vr.landmark_geometry(row)
    assert ok is False and t == 0.0 and eye_px == 0.0 and roll == 0.0


# ----------------------------------------------------- the lens, not 90 deg
def test_the_bearing_comes_from_the_configured_lens():
    """The same pixel offset is a different angle on the two real candidates,
    and the shipped comment reasoned from a third camera he does not own."""
    row = head_row(0.0, cx=240.0)                    # 80 px right of centre
    lifecam = vr.observe(row, LIFECAM_CINEMA, 4.0, 4.0, vr.HeadModel())
    c930e = vr.observe(row, C930E, 6.0, 6.0, vr.HeadModel())
    assert lifecam.bearing_deg == pytest.approx(17.87, abs=0.05)
    assert c930e.bearing_deg == pytest.approx(23.57, abs=0.05)
    assert c930e.bearing_deg > lifecam.bearing_deg * 1.25


def test_the_bearing_at_the_frame_edge_is_half_the_field_of_view():
    for lens, detect_w in ((LIFECAM_CINEMA, 320), (C930E, 320)):
        scale = lens.width_px / detect_w
        row = head_row(0.0, cx=float(detect_w))
        obs = vr.observe(row, lens, scale, scale, vr.HeadModel())
        assert obs.bearing_deg == pytest.approx(lens.hfov_deg / 2.0, abs=0.01)


def test_detect_space_is_mapped_back_to_capture_space():
    """YuNet sees 320x180; the box has to be reported in the pixels the
    resolution arithmetic is written in."""
    row = head_row(0.0, cx=160.0, cy=90.0, box_px=42.0)
    obs = vr.observe(row, LIFECAM_CINEMA, 4.0, 4.0, vr.HeadModel())
    assert obs.detect_px == pytest.approx(42.0)
    assert obs.face_px == pytest.approx(168.0)
    assert obs.bearing_deg == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------- the attention cone
def test_the_cone_is_measured_in_degrees_and_has_hysteresis():
    tracker = vr.AttentionTracker(cone_deg=20.0, hysteresis_deg=5.0)
    assert tracker.update(19.0) is True
    assert tracker.update(23.0) is True          # inside the released band
    assert tracker.update(26.0) is False
    assert tracker.update(23.0) is False         # must re-enter at 20, not 25
    assert tracker.update(19.0) is True


def test_the_cone_can_be_aimed_off_the_lens_axis():
    """An off-axis mount is the whole point: the camera sits beside the
    monitor, so 'looking at Jarvis' need not be 'looking down the lens'."""
    tracker = vr.AttentionTracker(cone_deg=10.0, hysteresis_deg=0.0,
                                  centre_deg=30.0)
    assert tracker.update(0.0) is False
    assert tracker.update(28.0) is True


def test_losing_the_face_ends_the_dwell():
    tracker = vr.AttentionTracker(cone_deg=20.0, hysteresis_deg=5.0)
    assert tracker.update(2.0) is True
    assert tracker.update(None) is False


# --------------------------------------------------------------- the rig
def test_the_rig_reports_counts_confidences_and_geometry():
    det = StubDetector([[head_row(4.0, conf=0.91)],
                        [head_row(6.0, conf=0.88)], []])
    report = rig(det).run(frames=6)
    assert report.frames == 6
    assert report.faces_total == 4
    assert report.detected_frames == 4
    assert report.conf_p50 == pytest.approx(0.895, abs=0.02)
    assert report.errors == 0
    assert report.detector_ok is True


def test_a_detector_that_raises_is_an_error_not_an_empty_room():
    """THE VSS SHAPE. A path that cannot run must not be indistinguishable
    from a path that ran and saw nobody -- that is exactly how
    ``region_mode='yunet'`` silently fell back to head_box and never once
    executed. An exception out of the detector is counted, named and
    reported, and the report does not claim a clean look."""
    det = StubDetector([[head_row(0.0)]], raise_on=(0, 1, 2, 3))
    report = rig(det).run(frames=4)
    assert report.frames == 4
    assert report.faces_total == 0
    assert report.errors == 4
    assert report.detector_ok is False
    assert "detector exploded" in report.reason
    assert report.ok is False


def test_an_absent_model_is_named_and_never_silently_zero():
    report = vr.Rig(StubSource(), None, LIFECAM_CINEMA, thresholds(),
                    detector_reason="missing: /nope/yunet.onnx").run(frames=3)
    assert report.detector_ok is False
    assert report.ok is False
    assert "/nope/yunet.onnx" in report.reason
    assert report.frames == 0        # nothing was even read; say so
    assert report.faces_total == 0


def test_a_source_that_stops_ends_the_run_without_pretending():
    det = StubDetector([[head_row(0.0)]])
    report = rig(det, source=StubSource(frames=3)).run(frames=10)
    assert report.frames == 3
    assert report.frames_missed == 7


# ------------------------------------------------------- numbers, and only
def test_the_report_carries_no_pixels_by_construction():
    det = StubDetector([[head_row(3.0)]])
    payload = rig(det).run(frames=4).to_dict()
    vr.assert_numbers_only(payload)          # raises if anything is not scalar
    assert "frame" not in payload and "image" not in payload


def test_assert_numbers_only_actually_rejects_an_array():
    with pytest.raises(TypeError) as exc:
        vr.assert_numbers_only({"faces": 1, "peek": {"a": np.zeros((2, 2))}})
    assert "peek" in str(exc.value) and "a" in str(exc.value)


def test_the_rig_keeps_no_frame_after_the_run():
    det = StubDetector([[head_row(0.0)]])
    r = rig(det)
    r.run(frames=3)
    for name, value in vars(r).items():
        assert not isinstance(value, np.ndarray), \
            "the rig is holding an array in %s" % name


def test_the_printed_lines_are_numbers_too():
    det = StubDetector([[head_row(3.0, conf=0.8)]])
    lines = rig(det).run(frames=3).lines()
    assert any("faces" in ln for ln in lines)
    assert all(isinstance(ln, str) for ln in lines)


# ------------------------------------------------------- pass/fail checks
def test_the_checks_state_a_verdict_against_each_threshold():
    det = StubDetector([[head_row(2.0, conf=0.82, box_px=42.0)]])
    report = rig(det).run(frames=8)
    checks = {c.name: c for c in report.checks}
    assert checks["min_conf"].ok is True
    assert checks["detect_px"].ok is True          # 42 px clears YuNet's 10
    assert checks["detection_rate"].ok is True
    assert "0.82" in checks["min_conf"].detail


def test_a_face_under_the_confidence_bar_fails_the_check_loudly():
    det = StubDetector([[head_row(2.0, conf=0.41)]])
    report = rig(det).run(frames=6)
    checks = {c.name: c for c in report.checks}
    assert checks["min_conf"].ok is False
    assert report.ok is False


def test_a_face_too_small_for_the_detector_fails_its_own_check():
    """YuNet's documented working range starts at ~10 px. The Arducam at
    320 wide puts a 16 cm face at 27 px and the LifeCam at 42; a mount that
    lands under 10 is a mount that cannot work, and the rig has to say which
    number failed rather than just detecting nothing."""
    det = StubDetector([[head_row(0.0, conf=0.9, box_px=8.0)]])
    report = rig(det).run(frames=4)
    checks = {c.name: c for c in report.checks}
    assert checks["detect_px"].ok is False
    assert "8" in checks["detect_px"].detail


def test_an_anisotropic_detect_size_is_flagged_not_silently_squashed():
    """320x240 for a 16:9 capture is a 1.33x horizontal squash of every face.
    The rig computes its own x and y scales and refuses to call that fine."""
    det = StubDetector([[head_row(0.0)]], input_size=(320, 240))
    report = vr.Rig(StubSource(), det, LIFECAM_CINEMA, thresholds(),
                    now=iter_clock()).run(frames=3)
    checks = {c.name: c for c in report.checks}
    assert checks["aspect"].ok is False
    assert "1.33" in checks["aspect"].detail


def iter_clock():
    clock = iter(np.arange(0.0, 10000.0, 0.125))
    return lambda: float(next(clock))


# ------------------------------------------------- identity is gated hard
def test_identity_is_only_ever_computed_on_a_face_that_cleared_min_conf():
    """SFace's out-of-distribution collapse (jarvis/facemodels.py) means a
    bad crop scores CONFIDENTLY against the gallery rather than low. So the
    embedding may never be a second opinion on whether this is a face, and
    the gate is structural: the recogniser is not called at all."""
    calls = []

    class Recogniser:
        def embed(self, frame, row):
            calls.append(float(row[-1]))
            return np.ones(128, dtype=np.float32)

    det = StubDetector([[head_row(0.0, conf=0.41)],
                        [head_row(0.0, conf=0.77)]])
    r = vr.Rig(StubSource(), det, LIFECAM_CINEMA, thresholds(),
               recogniser=Recogniser(), now=iter_clock())
    r.run(frames=4)
    assert calls == pytest.approx([0.77, 0.77], abs=1e-5)
    assert all(c >= 0.6 for c in calls)


def test_the_rig_reports_embedding_distances_as_numbers():
    class Recogniser:
        def embed(self, frame, row):
            return np.full(128, float(row[-1]), dtype=np.float32)

    det = StubDetector([[head_row(0.0, conf=0.9)]])
    r = vr.Rig(StubSource(), det, LIFECAM_CINEMA, thresholds(),
               recogniser=Recogniser(), now=iter_clock())
    report = r.run(frames=5)
    assert report.id_pairs > 0
    assert report.id_cosine_p50 == pytest.approx(1.0, abs=1e-5)
    vr.assert_numbers_only(report.to_dict())


# ------------------------------------------------------------ the exposure
def test_frame_level_and_contrast_are_two_scalars_not_a_histogram():
    """Question 3 of the open list -- whether autoexposure leaves enough
    contrast at 95 cm -- needs a number. Two scalars over a whole frame is a
    number; anything finer starts to be a picture."""
    det = StubDetector([[]])
    report = rig(det).run(frames=3)
    assert report.level_p50 == pytest.approx(100.0, abs=0.01)
    assert report.contrast_p50 == pytest.approx(0.0, abs=0.01)


# ------------------------------------------------ frames with no lens at all
def test_the_synthetic_source_delivers_frames_and_then_stops():
    src = vr.SyntheticSource(width=64, height=36, frames=3)
    shapes = []
    while True:
        ok, frame = src.read()
        if not ok:
            break
        shapes.append(frame.shape)
    assert shapes == [(36, 64, 3)] * 3
    src.release()
    assert src.released == 1


def test_the_synthetic_frames_are_structured_noise_not_flat_grey():
    """A flat frame would let any lazy early-out in the detector graph fire
    and under-report the real per-frame cost."""
    src = vr.SyntheticSource(width=64, height=36)
    _ok, a = src.read()
    _ok, b = src.read()
    assert float(a.std()) > 30.0
    assert not np.array_equal(a, b)


def test_a_synthetic_run_measures_cost_and_claims_no_accuracy():
    det = StubDetector([[]])
    report = vr.Rig(vr.SyntheticSource(width=64, height=36, frames=5), det,
                    LIFECAM_CINEMA, thresholds(), now=iter_clock()).run(
                        frames=5)
    assert report.frames == 5 and report.errors == 0
    checks = {c.name: c for c in report.checks}
    assert checks["min_conf"].ok is None      # nothing to say, and it says so
    assert checks["detection_rate"].ok is None


def test_a_recognisers_failure_is_not_reported_on_the_detector_line():
    """"the detector is broken, PASS" is the kind of line nobody believes
    twice. The reason belongs to whichever thing actually failed."""
    class Broken:
        def embed(self, frame, row):
            raise RuntimeError("alignCrop said no")

    det = StubDetector([[head_row(0.0, conf=0.9)]])
    report = vr.Rig(StubSource(), det, LIFECAM_CINEMA, thresholds(),
                    recogniser=Broken(), now=iter_clock()).run(frames=3)
    checks = {c.name: c for c in report.checks}
    assert checks["detector"].ok is True
    assert "alignCrop" not in checks["detector"].detail
    assert checks["errors"].ok is False
    assert "alignCrop" in checks["errors"].detail
    assert report.ok is False


# ------------------------------------------------- a fist is not a face
def _row(box, eye_r, eye_l, nose, mouth_r, mouth_l, score):
    """One detector row, built by hand. No image, no camera, no permission."""
    r = [0.0] * vr.DETECT_COLS
    r[0], r[1], r[2], r[3] = box
    r[vr.IDX_EYE_R], r[vr.IDX_EYE_R + 1] = eye_r
    r[vr.IDX_EYE_L], r[vr.IDX_EYE_L + 1] = eye_l
    r[vr.IDX_NOSE], r[vr.IDX_NOSE + 1] = nose
    r[vr.IDX_MOUTH_R], r[vr.IDX_MOUTH_R + 1] = mouth_r
    r[vr.IDX_MOUTH_L], r[vr.IDX_MOUTH_L + 1] = mouth_l
    r[vr.IDX_SCORE] = score
    return r


def _face_row(score=0.93):
    """A 100x120 box with anatomically ordinary landmarks."""
    return _row((0, 0, 100, 120), (31, 40), (69, 40), (50, 62),
                (36, 88), (64, 88), score)


def test_an_ordinary_face_row_is_plausible():
    ratio, drop, ok = vr.landmark_plausibility(_face_row())
    assert ok
    assert 0.30 < ratio < 0.55, ratio          # interocular over box width
    assert 1.0 < drop < 1.6, drop              # mouth below eyes, in eye-widths


def test_a_mouth_above_the_eyes_is_refused_however_confident():
    """The failure Hunter hit: a closed fist scored 0.71 -- over camera.min_conf
    0.6 -- and was accepted as a face. Before this gate the ONLY rejection was
    two eye landmarks at literally the same point, so any row with two distinct
    bright spots passed and was drawn, named and counted as a face."""
    fist = _row((0, 0, 100, 120), (46, 60), (54, 60), (50, 55),
                (47, 30), (53, 30), 0.71)      # "mouth" 30 px ABOVE the "eyes"
    _ratio, drop, ok = vr.landmark_plausibility(fist)
    assert drop < 0, drop
    assert not ok
    # and the old check would have waved it straight through
    assert vr.landmark_geometry(fist)[3] is True


def test_the_plausibility_gate_is_roll_invariant():
    """A head tilted over is a tilted head, not a rejected one. The drop is
    projected onto the face's own down axis, so it must not move with roll."""
    import math as _m
    upright = vr.landmark_plausibility(_face_row())[1]
    for roll in (-75.0, -30.0, 30.0, 75.0):
        a = _m.radians(roll)
        cos_a, sin_a = _m.cos(a), _m.sin(a)

        def spin(pt, cx=50.0, cy=60.0):
            dx, dy = pt[0] - cx, pt[1] - cy
            return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)

        tilted = _row((0, 0, 100, 120), spin((31, 40)), spin((69, 40)),
                      spin((50, 62)), spin((36, 88)), spin((64, 88)), 0.93)
        ratio, drop, ok = vr.landmark_plausibility(tilted)
        assert ok, roll
        assert abs(drop - upright) < 1e-6, (roll, drop, upright)


def test_observe_refuses_the_fist_and_reports_the_numbers():
    """observe() requires BOTH halves: a well-defined angle on a row that is
    not a face at all is still not a face."""
    lens = LIFECAM_CINEMA
    head = vr.HeadModel()
    good = vr.observe(_face_row(), lens, 1.0, 1.0, head)
    assert good.landmarks_ok
    assert good.mouth_drop_u > 1.0 and good.eye_box_ratio > 0.3

    fist = _row((0, 0, 100, 120), (46, 60), (54, 60), (50, 55),
                (47, 30), (53, 30), 0.71)
    bad = vr.observe(fist, lens, 1.0, 1.0, head)
    assert not bad.landmarks_ok
    assert bad.yaw_deg == 0.0          # no invented angle off a non-face
    assert bad.mouth_drop_u < 0


def test_a_row_with_no_mouth_landmarks_abstains_rather_than_refusing():
    """Absent evidence is not evidence of a fist. A row whose mouth corners are
    both at the origin reported no mouth at all; refusing it would be the same
    error as reading an unreachable sensor as 'nobody there'."""
    r = _face_row()
    r[vr.IDX_MOUTH_R] = r[vr.IDX_MOUTH_R + 1] = 0.0
    r[vr.IDX_MOUTH_L] = r[vr.IDX_MOUTH_L + 1] = 0.0
    ratio, drop, ok = vr.landmark_plausibility(r)
    assert ok and drop == 0.0
    assert ratio > 0.3, "the interocular ratio is still measured and reported"
