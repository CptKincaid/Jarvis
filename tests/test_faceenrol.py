"""Face enrolment (jarvis/faceenrol.py + scripts/face_enrol.py), with no
camera, no display, no weights and no pixels asserted on anywhere.

WHAT IS BEING PINNED, and why each one is a promise the design cannot keep by
intention alone:

* **The suite cannot reach his enrolled face.** On 2026-09-02 a test that
  built the real object and saved destroyed his voiceprint, and the copy kept
  beside it held only the fixtures -- unrecoverable, re-enrolled by hand. The
  first test below asserts the ``JARVIS_FACE_GALLERY`` redirect actually
  holds, rather than trusting that conftest set it.
* **No embedding is ever computed from a detection that failed the gate.**
  SFace scores non-faces CONFIDENTLY -- unrelated non-face crops match each
  other at cosine 0.66-0.92 against a 0.363 "same person" bar -- so a bad
  crop does not produce a low score, it produces a wrong answer that looks
  right. The test counts the recogniser's calls: a rejected sample must cost
  zero.
* **A gallery that is too tight or too loose is refused, and it says which.**
  Too tight fails him in every pose but one; too loose has somebody else in
  it. Both are silent failures, which is why neither may be saved.
* **The report carries numbers and nothing else.**
  ``assert_numbers_only`` is run over everything the script would print.
* **Recognising him grants nothing an anonymous face did not already get.**
  His ruling: identity may REMOVE capability or ADD a name, never grant it.

No test here creates a window, opens a device, loads a model, or looks at,
saves or asserts on the content of any frame.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from jarvis import faceenrol as fe
from jarvis import visionrig as vr
from jarvis.config import PATHS
from jarvis.eye import Attention, FaceIdentifier, SessionIdentity, resolve_wake
from jarvis.facegallery import (SFACE_COSINE_SAME, FaceGallery, cosine,
                                default_gallery)
from jarvis.facemodels import LIFECAM_CINEMA

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "face_enrol", os.path.join(_HERE, "scripts", "face_enrol.py"))
face_enrol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(face_enrol)


# ------------------------------------------------------------------ fakes
def head_row(yaw_deg: float, roll_deg: float = 0.0, cx: float = 160.0,
             cy: float = 90.0, eye_px: float = 40.0,
             nose_ratio: float = 0.35, conf: float = 0.9,
             box_px: float = 80.0) -> list:
    """A YuNet row synthesised from a rigid head at a known yaw, in DETECT
    pixels -- the same construction tests/test_visionrig.py uses, so the yaw
    that goes in is the yaw ``observe`` reads back."""
    b = nose_ratio
    psi = math.radians(yaw_deg)
    pts = [(-0.5 * math.cos(psi), 0.0),
           (+0.5 * math.cos(psi), 0.0),
           (b * math.sin(psi), 0.6),
           (-0.3 * math.cos(psi), 1.0),
           (+0.3 * math.cos(psi), 1.0)]
    th = math.radians(roll_deg)
    out = [cx - box_px / 2.0, cy - box_px / 2.0, box_px, box_px]
    for x, y in pts:
        rx = (x * math.cos(th) - y * math.sin(th)) * eye_px + cx
        ry = (x * math.sin(th) + y * math.cos(th)) * eye_px + cy
        out += [rx, ry]
    out.append(conf)
    return out


class ScriptedDetector:
    """Hands back one face per call, walking a list of yaws. Stands in for
    jarvis.facedetect.YuNetDetector; imports no cv2 and sees no frame."""

    name = "scripted"

    def __init__(self, yaws, conf=0.9, extra_faces=0, input_size=(320, 180),
                 **row_kw):
        self.yaws = list(yaws)
        self.conf = conf
        self.extra_faces = int(extra_faces)
        self.input_size = input_size
        self.row_kw = row_kw
        self.calls = 0

    def detect(self, frame):
        yaw = self.yaws[self.calls % len(self.yaws)]
        self.calls += 1
        conf = self.conf(self.calls - 1) if callable(self.conf) else self.conf
        rows = [head_row(yaw, conf=conf, **self.row_kw)]
        for i in range(self.extra_faces):
            rows.append(head_row(yaw, cx=60.0 + 10 * i, conf=0.7))
        return np.asarray(rows, dtype=np.float32)


class ScriptedRecogniser:
    """Deterministic 128-D vectors, and the SAME confidence gate the real
    SFaceRecogniser applies (jarvis/facedetect.py:212-226) -- so a test that
    slipped a weak row past the session would fail here too."""

    def __init__(self, vectors, min_conf: float = 0.6):
        self.vectors = list(vectors)
        self.min_conf = float(min_conf)
        self.calls = 0

    def embed(self, frame, row):
        conf = float(np.asarray(row).ravel()[-1])
        if conf < self.min_conf:
            raise ValueError("refusing to embed a detection scoring %.2f"
                             % conf)
        vec = self.vectors[self.calls % len(self.vectors)]
        self.calls += 1
        return np.asarray(vec, dtype=np.float32)


class FakeSource:
    """``read() -> (ok, frame)``, cv2.VideoCapture's own contract. The frames
    are noise this process generates; nothing here is ever looked at,
    displayed or written, and ``stop_after`` is how "sensing denied the
    camera mid-run" is simulated."""

    def __init__(self, stop_after: int = 0, width=1280, height=720, seed=3):
        self.stop_after = int(stop_after)
        self.reads = 0
        self._rng = np.random.default_rng(seed)
        self.width, self.height = width, height
        self.released = 0

    def read(self):
        if self.stop_after and self.reads >= self.stop_after:
            return False, None
        self.reads += 1
        return True, self._rng.integers(0, 256, (self.height, self.width, 3),
                                        dtype=np.uint8)

    def release(self):
        self.released += 1


def base_vec(seed: int = 11) -> np.ndarray:
    """A plausible SFace embedding: 128 float32, NOT unit norm (measured on
    this box 2026-09-02, L2 = 10.41)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(128) * 0.9).astype(np.float32)


def same_face(base, n: int, k: float = 0.4, seed: int = 100) -> list:
    """``n`` takes of ONE face: the base plus noise, which is what a real
    pool looks like -- alike, and not identical."""
    rng = np.random.default_rng(seed)
    return [(base + rng.standard_normal(base.shape).astype(np.float32) * k
             ).astype(np.float32) for _ in range(n)]


PLAN_YAWS = [0.0, 0.0, 0.0, 40.0, 40.0, 40.0, -40.0, -40.0, -40.0,
             2.0, 2.0, -2.0, -2.0]


def a_good_session(tmp_path, yaws=None, vectors=None, min_conf=0.6):
    gallery = FaceGallery(root=tmp_path / "gallery")
    det = ScriptedDetector(yaws or PLAN_YAWS)
    vecs = vectors if vectors is not None else same_face(base_vec(), 13)
    rec = ScriptedRecogniser(vecs, min_conf=min_conf)
    session = fe.EnrolmentSession(gallery, "hunter", LIFECAM_CINEMA, det, rec,
                                  fe.SampleLimits(min_conf=min_conf))
    return gallery, det, rec, session


# ------------------------------------------------------------ the firewall
def test_no_test_can_reach_his_real_face_gallery():
    """THE 2026-09-02 SHAPE, refused. A test built the real object and saved;
    the voiceprint was replaced by fixtures and was unrecoverable. So: the
    redirect is asserted, not assumed -- including through
    ``default_gallery()``, which is the exact call an unwary test would make.
    """
    real = Path.home() / ".aiws_trainer" / "face_gallery"
    assert os.environ.get("JARVIS_FACE_GALLERY"), \
        "conftest must force JARVIS_FACE_GALLERY before any jarvis import"
    assert PATHS.FACE_GALLERY != real
    assert real not in PATHS.FACE_GALLERY.parents

    g = default_gallery()
    assert g.root != real
    assert real not in Path(g.root).parents
    # And a real save through the real object lands in the throwaway dir.
    for vec in same_face(base_vec(7), 3):
        g.add("fixture", vec)
    gen = g.save(reason="firewall test")
    assert gen >= 1
    assert g.path_for(gen).exists()
    assert str(g.path_for(gen)).startswith(str(PATHS.FACE_GALLERY))
    assert not str(g.path_for(gen)).startswith(str(real))
    g.purge()


def test_the_face_model_dir_is_redirected_too():
    """A test that quietly passed only because 38 MB of SFace happened to be
    on this machine would be worse than no test."""
    from jarvis import facemodels
    assert os.environ.get("JARVIS_FACE_MODEL_DIR")
    assert facemodels.model_dir() != Path.home() / ".aiws_trainer" / \
        "models" / "face"


# ------------------------------------------------------------- the measures
def test_sharpness_separates_a_blurred_box_from_a_sharp_one():
    """Gross motion blur is what this claims to catch, and all it claims."""
    rng = np.random.default_rng(5)
    sharp = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
    # A smooth ramp is what a badly blurred face reduces to: structure with
    # almost no per-pixel gradient.
    ramp = np.tile(np.linspace(40, 210, 400, dtype=np.uint8), (400, 1))
    blurred = np.stack([ramp, ramp, ramp], axis=2)
    s_sharp = fe.sharpness(sharp, 20, 20, 200, 200)
    s_blur = fe.sharpness(blurred, 20, 20, 200, 200)
    assert s_sharp > fe.MIN_SHARPNESS
    assert s_blur < fe.MIN_SHARPNESS
    assert isinstance(s_sharp, float)


def test_sharpness_returns_a_float_and_never_a_crop():
    """The crop exists inside the call and dies there: one float comes out,
    which is the whole contract of a numbers-only pipeline."""
    frame = np.random.default_rng(1).integers(0, 256, (200, 200, 3),
                                              dtype=np.uint8)
    out = fe.sharpness(frame, 10, 10, 100, 100)
    assert isinstance(out, float)
    vr.assert_numbers_only({"sharpness": out})


def test_sharpness_survives_a_box_outside_the_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert fe.sharpness(frame, -500, -500, 20, 20) == 0.0
    assert fe.sharpness(frame, 90, 90, 200, 200) == 0.0
    assert fe.sharpness(None, 0, 0, 50, 50) == 0.0


# --------------------------------------------------------- the sample gate
def _obs(yaw=0.0, conf=0.9, roll=0.0, box_px=80.0, eye_px=40.0):
    row = head_row(yaw, roll_deg=roll, conf=conf, box_px=box_px,
                   eye_px=eye_px)
    return vr.observe(row, LIFECAM_CINEMA, 4.0, 4.0, vr.HeadModel())


def test_a_detection_under_the_bar_is_rejected_before_anything_else():
    """The detector bar is the only thing standing between a bad crop and a
    poisoned gallery, because SFace scores garbage confidently."""
    limits = fe.SampleLimits(min_conf=0.6)
    ok, why = fe.judge_sample(_obs(conf=0.45), 1, 1.0, limits)
    assert ok is False
    assert "0.45" in why and "0.60" in why
    assert "no embedding is computed" in why


def test_two_faces_in_frame_is_refused():
    """A gallery that learned a visitor identifies the wrong person
    confidently, and nothing about it looks wrong afterwards."""
    ok, why = fe.judge_sample(_obs(), 2, 1.0, fe.SampleLimits(min_conf=0.6))
    assert ok is False
    assert "2 faces" in why


@pytest.mark.parametrize("kwargs,fragment", [
    ({"box_px": 20.0}, "under SFace's"),      # 80 capture px, under 112
    ({"eye_px": 6.0}, "interocular"),          # 24 capture px, under 35
    ({"yaw": 75.0}, "yaw"),
    ({"roll": 45.0}, "roll"),
])
def test_the_quality_bars_each_reject_with_their_own_reason(kwargs, fragment):
    ok, why = fe.judge_sample(_obs(**kwargs), 1, 1.0,
                              fe.SampleLimits(min_conf=0.6))
    assert ok is False, why
    assert fragment in why


def test_a_blurred_sample_is_rejected_and_says_so():
    ok, why = fe.judge_sample(_obs(), 1, 0.001,
                              fe.SampleLimits(min_conf=0.6))
    assert ok is False
    assert "motion blur" in why


def test_a_good_sample_is_accepted_and_still_carries_a_reason():
    ok, why = fe.judge_sample(_obs(yaw=40.0), 1, 0.5,
                              fe.SampleLimits(min_conf=0.6))
    assert ok is True
    assert why == "ok"


# ------------------------------------------------------------- the session
def test_a_rejected_sample_costs_zero_embeddings(tmp_path):
    """THE RULE, IN CODE. SFace collapses on out-of-distribution input, so an
    embedding taken from a detection that failed the gate is not a weak
    opinion -- it is a confident wrong answer. Nothing may compute one."""
    gallery, det, rec, session = a_good_session(tmp_path, yaws=[0.0])
    det.conf = 0.30                       # under the 0.6 bar
    source = FakeSource()
    for _ in range(5):
        _ok, frame = source.read()
        sample = session.offer(frame, "lens")
        assert sample.accepted is False
    assert rec.calls == 0, "an embedding was computed from a weak detection"
    assert gallery.total() == 0


def test_a_frame_with_no_face_is_not_a_rejected_sample(tmp_path):
    """An empty frame while he settles into position is not a data point, and
    filling the report with them would bury the ones that matter."""

    class Empty:
        name, input_size = "empty", (320, 180)

        def detect(self, frame):
            return None

    gallery = FaceGallery(root=tmp_path / "g")
    session = fe.EnrolmentSession(gallery, "hunter", LIFECAM_CINEMA, Empty(),
                                  ScriptedRecogniser([base_vec()]),
                                  fe.SampleLimits(min_conf=0.6))
    assert session.offer(np.zeros((720, 1280, 3), np.uint8)) is None
    assert session.samples == []


def test_the_session_holds_no_frame_and_no_crop(tmp_path):
    gallery, _det, _rec, session = a_good_session(tmp_path)
    source = FakeSource()
    for _ in range(3):
        _ok, frame = source.read()
        session.offer(frame, "lens")
    held = [v for v in vars(session).values()
            if isinstance(v, np.ndarray) and v.ndim >= 2]
    assert held == [], "the session is holding an image-shaped array"


def test_a_detector_that_raises_becomes_a_named_rejection(tmp_path):
    class Boom:
        name, input_size = "boom", (320, 180)

        def detect(self, frame):
            raise RuntimeError("the detector exploded")

    gallery = FaceGallery(root=tmp_path / "g")
    session = fe.EnrolmentSession(gallery, "hunter", LIFECAM_CINEMA, Boom(),
                                  ScriptedRecogniser([base_vec()]),
                                  fe.SampleLimits(min_conf=0.6))
    sample = session.offer(np.zeros((720, 1280, 3), np.uint8))
    assert sample.accepted is False
    assert "detector raised" in sample.reason


def test_a_recogniser_that_refuses_is_reported_not_raised(tmp_path):
    """A model that says no must land in the report as a dropped sample; a
    traceback out of an enrolment loses every sample already taken."""
    gallery, det, _rec, session = a_good_session(tmp_path, yaws=[0.0])

    class Refuser:
        calls = 0

        def embed(self, frame, row):
            raise ValueError("SFace said no")

    session.recogniser = Refuser()
    _ok, frame = FakeSource().read()
    sample = session.offer(frame, "lens")
    assert sample.accepted is False
    assert "recogniser refused" in sample.reason


# -------------------------------------------------------- the gallery gate
def test_a_good_spread_passes_every_check(tmp_path):
    gallery, _det, _rec, session = a_good_session(tmp_path)
    source = FakeSource()
    for _ in range(len(PLAN_YAWS)):
        _ok, frame = source.read()
        session.offer(frame, "plan")
    rep = session.report()
    assert rep.accepted == len(PLAN_YAWS)
    assert rep.ok is True, [c.as_tuple() for c in rep.checks]
    assert rep.yaw_spread >= fe.MIN_YAW_SPREAD_DEG
    assert rep.frontal >= fe.MIN_FRONTAL and rep.offaxis >= fe.MIN_OFFAXIS


def test_too_tight_is_refused_and_named(tmp_path):
    """Near-duplicate samples: a gallery that knows him in one pose and
    rejects him in every other, which fails silently."""
    base = base_vec(21)
    dupes = same_face(base, 13, k=0.02, seed=9)     # all but identical
    gallery, _det, _rec, session = a_good_session(tmp_path, vectors=dupes)
    source = FakeSource()
    for _ in range(len(PLAN_YAWS)):
        _ok, frame = source.read()
        session.offer(frame, "plan")
    rep = session.report()
    failed = {c.name for c in rep.checks if c.ok is False}
    assert "variation" in failed
    assert rep.ok is False
    detail = [c.detail for c in rep.checks if c.name == "variation"][0]
    assert "one face repeated" in detail


def test_no_pose_variation_is_refused_even_when_the_vectors_differ(tmp_path):
    """The failure his own usage guarantees: 13 takes of one head position.
    The embeddings can look healthy and the gallery still be useless."""
    gallery, _det, _rec, session = a_good_session(
        tmp_path, yaws=[0.0, 3.0, -3.0, 5.0])
    source = FakeSource()
    for _ in range(13):
        _ok, frame = source.read()
        session.offer(frame, "lens")
    rep = session.report()
    failed = {c.name for c in rep.checks if c.ok is False}
    assert "pose_spread" in failed
    assert "variation" not in failed          # the vectors themselves differ
    assert rep.ok is False


def test_too_loose_is_refused_and_names_the_outlier(tmp_path):
    """One sample that does not cluster: a second person in frame, or a crop
    that is not a face. match() scores against the pool's BEST member, so one
    poisoned sample is all it takes."""
    good = same_face(base_vec(31), 12, seed=3)
    stranger = base_vec(999)
    vectors = good[:6] + [stranger] + good[6:]
    gallery, _det, _rec, session = a_good_session(tmp_path, vectors=vectors)
    source = FakeSource()
    for _ in range(len(PLAN_YAWS)):
        _ok, frame = source.read()
        session.offer(frame, "plan")
    rep = session.report()
    failed = {c.name for c in rep.checks if c.ok is False}
    assert "cohesion" in failed
    detail = [c.detail for c in rep.checks if c.name == "cohesion"][0]
    assert "#6" in detail
    assert "%.3f" % SFACE_COSINE_SAME in detail


def test_too_few_samples_is_refused():
    embs = same_face(base_vec(4), 3)
    samples = [fe.Sample(index=i, station="lens", faces=1, conf=0.9,
                         face_px=300.0, eye_px=140.0, yaw_deg=float(i * 20),
                         roll_deg=0.0, bearing_deg=0.0, sharpness=0.4,
                         accepted=True, reason="ok") for i in range(3)]
    checks = fe.judge_gallery(samples, embs)
    assert [c.ok for c in checks if c.name == "samples"] == [False]


def test_an_empty_pool_is_not_applicable_rather_than_a_failure():
    checks = fe.judge_gallery([], [])
    named = {c.name: c.ok for c in checks}
    assert named["samples"] is False
    assert named["pose_spread"] is None and named["cohesion"] is None


# ------------------------------------------------------------- the report
def test_the_report_carries_numbers_and_nothing_else(tmp_path):
    gallery, _det, _rec, session = a_good_session(tmp_path)
    source = FakeSource()
    for _ in range(len(PLAN_YAWS)):
        _ok, frame = source.read()
        session.offer(frame, "plan")
    rep = session.report()
    payload = rep.to_dict()
    vr.assert_numbers_only(payload)
    for line in rep.lines():
        assert isinstance(line, str)


def test_the_report_is_rejected_if_an_array_is_ever_added_to_it(tmp_path):
    """The guard rail is mechanical, not a review convention."""
    gallery, _det, _rec, session = a_good_session(tmp_path)
    payload = session.report().to_dict()
    payload["oops"] = np.zeros((4, 4))
    with pytest.raises(TypeError):
        vr.assert_numbers_only(payload)


def test_every_sample_line_carries_its_own_numbers(tmp_path):
    """He pastes these; each line has to stand alone as evidence."""
    gallery, _det, _rec, session = a_good_session(tmp_path)
    _ok, frame = FakeSource().read()
    sample = session.offer(frame, "lens")
    line = sample.line()
    for token in ("conf", "face", "eyes", "yaw", "roll", "sharp"):
        assert token in line


# ------------------------------------------------------------ the run loop
def test_the_run_walks_the_plan_and_saves_a_usable_gallery(tmp_path):
    gallery, _det, _rec, session = a_good_session(tmp_path)
    rep, ran = fe.run_enrolment(session, FakeSource(), say=lambda *_a: None,
                                gap_s=0.0)
    assert ran == len(fe.DEFAULT_PLAN)
    assert rep.accepted == sum(s.samples for s in fe.DEFAULT_PLAN)
    assert rep.ok is True, [c.as_tuple() for c in rep.checks]
    gen = gallery.save(reason="test")
    assert gen == 1
    assert gallery.total() == rep.accepted


def test_the_run_stops_when_sensing_takes_the_camera_away(tmp_path):
    """The 21:00 curfew arriving mid-enrolment, or him saying "offline mode".
    ``_GatedDevice.read`` returns (False, None) and the correct response is
    to stop and SAY so -- not to finish quietly with half a gallery."""
    gallery, _det, _rec, session = a_good_session(tmp_path)
    rep, ran = fe.run_enrolment(session, FakeSource(stop_after=4),
                                say=lambda *_a: None, gap_s=0.0)
    assert rep.detector_ok is False
    assert "sensing denied" in rep.reason
    assert rep.ok is False


def test_a_station_that_cannot_be_reached_costs_that_station_only(tmp_path):
    """His desk may not permit one of the head positions. That must cost the
    station, not the enrolment -- the pose spread is judged at the end."""
    said = []
    gallery, _det, _rec, session = a_good_session(tmp_path)
    rep, ran = fe.run_enrolment(session, FakeSource(), say=said.append,
                                gap_s=0.0, frames_per_station=2)
    assert rep.accepted == 2 * len(fe.DEFAULT_PLAN)
    assert any("moving on" in s for s in said)


def test_the_run_says_when_he_is_not_in_the_position_it_asked_for(tmp_path):
    said = []
    gallery, _det, _rec, session = a_good_session(tmp_path, yaws=[45.0])
    fe.run_enrolment(session, FakeSource(), say=said.append, gap_s=0.0,
                     plan=(fe.DEFAULT_PLAN[0],), frames_per_station=3)
    assert any("outside this station" in s for s in said)


def test_the_gap_keeps_one_pose_from_becoming_eight_near_duplicates(tmp_path):
    """Eight frames at 8 fps is one second of one pose. The spacing is what
    makes the plan produce variation rather than repetition."""
    clock = {"t": 0.0}
    gallery, _det, _rec, session = a_good_session(tmp_path)
    fe.run_enrolment(session, FakeSource(), say=lambda *_a: None, gap_s=0.5,
                     now=lambda: clock["t"],
                     sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                     plan=(fe.DEFAULT_PLAN[0],))
    assert clock["t"] >= 0.5 * (fe.DEFAULT_PLAN[0].samples - 1)


# ----------------------------------------------------- storage and safety
def test_harden_puts_0700_on_the_directory_and_0600_on_the_files(tmp_path):
    root = tmp_path / "g"
    g = FaceGallery(root=root)
    for vec in same_face(base_vec(2), 4):
        g.add("hunter", vec)
    g.save(reason="t")
    os.chmod(root, 0o755)
    os.chmod(g.path_for(1), 0o644)
    found = fe.harden(root)
    assert found["dir"] == 755
    assert found["gen-00001.npz"] == 644
    assert oct(os.stat(root).st_mode & 0o777) == "0o700"
    assert oct(os.stat(g.path_for(1)).st_mode & 0o777) == "0o600"


def test_backup_copies_every_generation_and_reads_each_one_back(tmp_path):
    """A backup nobody has read back is the shape the voiceprint's copy had:
    it existed, it was named like a backup, and it held only fixtures."""
    g = FaceGallery(root=tmp_path / "live")
    for vec in same_face(base_vec(6), 5):
        g.add("hunter", vec)
    g.save(reason="one")
    for vec in same_face(base_vec(6), 6, seed=77):
        g.add("hunter", vec)
    g.save(reason="two")
    out = fe.backup(g, tmp_path / "elsewhere")
    assert out["ok"] is True
    assert out["copied"] == 2 and out["verified"] == 2
    assert out["failed"] == []
    for name in ("gen-00001.npz", "gen-00002.npz"):
        path = tmp_path / "elsewhere" / name
        assert path.exists()
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    assert oct(os.stat(tmp_path / "elsewhere").st_mode & 0o777) == "0o700"


def test_a_backup_that_does_not_parse_is_reported_as_failed(tmp_path):
    g = FaceGallery(root=tmp_path / "live")
    for vec in same_face(base_vec(8), 4):
        g.add("hunter", vec)
    g.save(reason="one")
    g.path_for(1).write_bytes(b"not an npz")
    out = fe.backup(g, tmp_path / "elsewhere")
    assert out["ok"] is False
    assert out["verified"] == 0
    assert "did not parse" in out["failed"][0]


def test_restore_writes_a_new_generation_and_overwrites_nothing(tmp_path):
    """Restoring by overwriting is the move that made 2026-09-02
    unrecoverable. A restore of the wrong thing is one rollback away."""
    live = FaceGallery(root=tmp_path / "live")
    original = same_face(base_vec(12), 6)
    for vec in original:
        live.add("hunter", vec)
    live.save(reason="original")
    fe.backup(live, tmp_path / "backup")

    live.reset()
    for vec in same_face(base_vec(13), 6, seed=44):
        live.add("hunter", vec)
    live.save(reason="a later, different enrolment")
    assert live.generations() == [1, 2]

    out = fe.restore(live, tmp_path / "backup", reason="restore")
    assert out["restored"] == 6
    assert out["generation"] == 3
    assert out["samples"] == 6         # a restore replaces, it does not merge
    assert live.generations() == [1, 2, 3]     # nothing was overwritten
    back = FaceGallery(root=tmp_path / "live")
    back.load(generation=3)
    assert back.total() == 6
    assert cosine(back.embeddings("hunter")[0], original[0]) == \
        pytest.approx(1.0, abs=1e-6)
    # ...and the enrolment the restore replaced is still there to go back to.
    was = FaceGallery(root=tmp_path / "live")
    was.load(generation=2)
    assert was.total() == 6


def test_restore_from_an_empty_directory_says_so(tmp_path):
    live = FaceGallery(root=tmp_path / "live")
    out = fe.restore(live, tmp_path / "nothing", reason="r")
    assert out["restored"] == 0
    assert "nothing readable" in out["reason"]


# --------------------------------------------------------- the match path
def test_identity_is_never_computed_from_a_weak_detection():
    """Enforced in code, not in a comment: a row under the bar returns no
    opinion and the recogniser is not called at all."""
    gallery = FaceGallery(root=None)
    for vec in same_face(base_vec(17), 6):
        gallery.add("hunter", vec)
    rec = ScriptedRecogniser([base_vec(17)])
    ident = FaceIdentifier(gallery, rec, min_conf=0.6)
    label, score = ident.identify(None, head_row(0.0, conf=0.4))
    assert (label, score) == ("", 0.0)
    assert rec.calls == 0
    assert ident.gated_out == 1


def test_a_recognised_face_comes_back_with_its_score():
    base = base_vec(19)
    gallery = FaceGallery(root=None)
    for vec in same_face(base, 8):
        gallery.add("hunter", vec)
    rec = ScriptedRecogniser([base])
    ident = FaceIdentifier(gallery, rec, min_conf=0.6,
                           match_min=SFACE_COSINE_SAME)
    label, score = ident.identify(None, head_row(0.0, conf=0.9))
    assert label == "hunter"
    assert score >= SFACE_COSINE_SAME
    assert ident.matched == 1


def test_a_stranger_is_no_opinion_rather_than_a_wrong_name():
    gallery = FaceGallery(root=None)
    for vec in same_face(base_vec(23), 8):
        gallery.add("hunter", vec)
    rec = ScriptedRecogniser([base_vec(4242)])
    ident = FaceIdentifier(gallery, rec, min_conf=0.6)
    assert ident.identify(None, head_row(0.0, conf=0.95)) == ("", 0.0)
    assert ident.unknown == 1


def test_an_empty_gallery_computes_nothing_at_all():
    """Before enrolment there is nothing to compare against, and paying for
    an embedding to discover that is pure cost."""
    rec = ScriptedRecogniser([base_vec()])
    ident = FaceIdentifier(FaceGallery(root=None), rec, min_conf=0.6)
    assert ident.identify(None, head_row(0.0, conf=0.9)) == ("", 0.0)
    assert rec.calls == 0


def test_a_broken_recogniser_is_no_opinion_and_never_an_exception():
    class Boom:
        def embed(self, frame, row):
            raise RuntimeError("no")

    gallery = FaceGallery(root=None)
    for vec in same_face(base_vec(25), 4):
        gallery.add("hunter", vec)
    ident = FaceIdentifier(gallery, Boom(), min_conf=0.6)
    assert ident.identify(None, head_row(0.0, conf=0.9)) == ("", 0.0)
    assert ident.errors == 1


def test_recognising_him_grants_nothing_an_anonymous_face_lacked():
    """HIS STANDING RULING: identity may REMOVE capability or ADD a name; it
    must never GRANT capability the existing gates do not already grant. So
    for every input, a recognised owner and an unidentified face reach the
    same ``ok``; only the log line differs."""
    for verdict, ok in (("suppress", False), ("accept", True),
                        ("abstain", True)):
        for faces, attending, dwell in ((1, True, 0.8), (1, False, 0.8),
                                        (2, True, 0.8), (1, True, 0.1)):
            anon = resolve_wake(verdict, ok, Attention(
                faces=faces, attending=attending, dwell_s=dwell, identity=""))
            named = resolve_wake(verdict, ok, Attention(
                faces=faces, attending=attending, dwell_s=dwell,
                identity="hunter", id_score=0.9))
            assert named.ok == anon.ok
            assert named.guest_ok == anon.guest_ok


def test_a_recognised_stranger_removes_what_an_anonymous_face_would_have_got():
    """The other direction, which identity IS allowed to act in."""
    anon = resolve_wake("suppress", False, Attention(
        faces=1, attending=True, dwell_s=0.8, identity=""))
    other = resolve_wake("suppress", False, Attention(
        faces=1, attending=True, dwell_s=0.8, identity="guest", id_score=0.9))
    assert anon.ok is True
    assert other.ok is False


def test_a_recognised_stranger_drops_the_body_anchor():
    """Removing an identity is the direction this may act in."""
    session = SessionIdentity(ttl_s=600.0, match_min=0.75, now=lambda: 0.0)
    body = np.random.default_rng(2).standard_normal(768).astype(np.float32)
    session.anchor("hunter", body)
    assert session.identify(body)[0] == "hunter"

    gallery = FaceGallery(root=None)
    for vec in same_face(base_vec(27), 5):
        gallery.add("hunter", vec)
    ident = FaceIdentifier(gallery, ScriptedRecogniser([base_vec(31337)]),
                           min_conf=0.6, session=session)
    ident.identify(None, head_row(0.0, conf=0.9))
    assert session.identify(body) == ("", 0.0)


def test_recognising_him_anchors_the_body_vector():
    base = base_vec(29)
    session = SessionIdentity(ttl_s=600.0, match_min=0.75, now=lambda: 0.0)
    gallery = FaceGallery(root=None)
    for vec in same_face(base, 5):
        gallery.add("hunter", vec)
    ident = FaceIdentifier(gallery, ScriptedRecogniser([base]), min_conf=0.6,
                           session=session)
    body = np.random.default_rng(3).standard_normal(768).astype(np.float32)
    label, _score = ident.identify(None, head_row(0.0, conf=0.9),
                                   body_vec=body)
    assert label == "hunter"
    assert session.identify(body)[0] == "hunter"


def test_a_degenerate_body_vector_does_not_lose_the_face_identification():
    base = base_vec(33)
    session = SessionIdentity(ttl_s=600.0, match_min=0.75, now=lambda: 0.0)
    gallery = FaceGallery(root=None)
    for vec in same_face(base, 5):
        gallery.add("hunter", vec)
    ident = FaceIdentifier(gallery, ScriptedRecogniser([base]), min_conf=0.6,
                           session=session)
    label, _s = ident.identify(None, head_row(0.0, conf=0.9),
                               body_vec=np.zeros(768, dtype=np.float32))
    assert label == "hunter"


def test_the_identifier_status_is_numbers_only():
    ident = FaceIdentifier(FaceGallery(root=None),
                           ScriptedRecogniser([base_vec()]), min_conf=0.6)
    vr.assert_numbers_only(ident.status())


# --------------------------------------------- the rig, wired to a gallery
def test_the_rig_reports_gallery_matches_and_still_carries_no_pixels():
    base = base_vec(41)
    gallery = FaceGallery(root=None)
    for vec in same_face(base, 8):
        gallery.add("hunter", vec)
    ident = FaceIdentifier(gallery, ScriptedRecogniser([base]), min_conf=0.6)
    det = ScriptedDetector([0.0])
    rig = vr.Rig(FakeSource(), det, LIFECAM_CINEMA,
                 vr.Thresholds(min_conf=0.6, cone_deg=20.0,
                               cone_hysteresis_deg=5.0, dwell_s=0.6,
                               identity_min=SFACE_COSINE_SAME),
                 identifier=ident)
    rep = rig.run(frames=6)
    payload = rep.to_dict()
    vr.assert_numbers_only(payload)
    assert rep.id_matched_frames == 6
    assert rep.id_unknown_frames == 0
    assert rep.gallery_enrolled == 8
    assert rep.match_p50 >= SFACE_COSINE_SAME
    assert [c.ok for c in rep.checks if c.name == "gallery"] == [True]
    assert any("gallery" in line for line in rep.lines())


def test_the_rig_says_not_applicable_when_nothing_is_enrolled():
    ident = FaceIdentifier(FaceGallery(root=None),
                           ScriptedRecogniser([base_vec()]), min_conf=0.6)
    rig = vr.Rig(FakeSource(), ScriptedDetector([0.0]), LIFECAM_CINEMA,
                 vr.Thresholds(min_conf=0.6, cone_deg=20.0,
                               cone_hysteresis_deg=5.0, dwell_s=0.6,
                               identity_min=SFACE_COSINE_SAME),
                 identifier=ident)
    rep = rig.run(frames=3)
    # NOT APPLICABLE, not FAIL: nothing enrolled is a state, not a fault, and
    # a check that failed for it would train him to ignore the report.
    assert [c.ok for c in rep.checks if c.name == "gallery"] == [None]
    assert rep.errors == 0 and rep.detector_ok is True


def test_a_rig_without_an_identifier_is_byte_for_byte_what_it_was():
    """The gallery leg is additive: with no identifier the report's new
    fields stay zero and no check appears."""
    rig = vr.Rig(FakeSource(), ScriptedDetector([0.0]), LIFECAM_CINEMA,
                 vr.Thresholds(min_conf=0.6, cone_deg=20.0,
                               cone_hysteresis_deg=5.0, dwell_s=0.6,
                               identity_min=SFACE_COSINE_SAME))
    rep = rig.run(frames=3)
    assert rep.gallery_enrolled == 0
    assert rep.id_matched_frames == 0
    assert "gallery" not in {c.name for c in rep.checks}


# ------------------------------------------------------------- the command
class FakePolicy:
    """The sensing owner, as far as the script can see it. The real one is
    ``jarvis/sensing.SensingPolicy``; what is being pinned here is that the
    script ASKS it and obeys, not what it answers."""

    def __init__(self, camera=True, reason=""):
        self._camera = bool(camera)
        self._reason = reason

    def status(self):
        return {"camera": self._camera, "radar": True,
                "offline": not self._camera, "reason": self._reason,
                "until": None, "failsafe": False, "persisted": True,
                "curfew": "21:00-07:00"}

    def allowed(self, kind):
        return self._camera and kind == "camera"

    def attach(self, *a, **k):
        pass


class FakeFeed:
    """``jarvis/camera.CameraFeed``'s surface, minus the device. ``capture``
    returning None is what the gated feed does when sensing says no."""

    def __init__(self, source, lens=LIFECAM_CINEMA):
        self.source = source
        self.lens = lens
        self.closed = 0

    def capture(self):
        ok, frame = self.source.read()
        return frame if ok else None

    def close(self):
        self.closed += 1


def wire(monkeypatch, tmp_path, *, yaws=None, vectors=None, camera=True,
         source=None, feed_guard=False):
    """Point the script at fakes: a fake sensing owner, a fake feed, scripted
    models, and a gallery under tmp_path. Nothing opens a device, loads a
    model or touches his real gallery or his real config."""
    gallery = FaceGallery(root=tmp_path / "gallery")
    cfg_path = tmp_path / "assistant.json"
    real_load = face_enrol.AssistantConfig.load
    monkeypatch.setattr(face_enrol.AssistantConfig, "load",
                        staticmethod(lambda: real_load(cfg_path)))
    monkeypatch.setattr(face_enrol, "open_gallery", lambda: gallery)
    monkeypatch.setattr(face_enrol, "SensingPolicy",
                        lambda cfg=None: FakePolicy(camera=camera))
    monkeypatch.setattr(face_enrol.facedetect, "probe",
                        lambda **k: {"ready": True, "models": {}, "dir": "",
                                     "deep": False, "cv2": {}})
    det = ScriptedDetector(yaws or PLAN_YAWS)
    rec = ScriptedRecogniser(vectors if vectors is not None
                             else same_face(base_vec(), 13))
    monkeypatch.setattr(face_enrol, "build_models", lambda cfg: (det, rec, ""))
    feed = FakeFeed(source or FakeSource())
    if feed_guard:
        def _never(cfg, policy):
            raise AssertionError("the device must not be built when sensing "
                                 "has said no")
        monkeypatch.setattr(face_enrol, "build_feed", _never)
    else:
        monkeypatch.setattr(face_enrol, "build_feed",
                            lambda cfg, policy: (feed, ""))
    return gallery, feed


ENROL = ["--auto", "--gap-s", "0", "--enable-identity"]


def test_the_command_enrols_and_saves_one_generation(monkeypatch, tmp_path,
                                                     capsys):
    gallery, feed = wire(monkeypatch, tmp_path)
    code = face_enrol.main(ENROL)
    out = capsys.readouterr().out
    assert code == 0, out
    assert gallery.generations() == [1]
    saved = FaceGallery(root=gallery.root)
    saved.load()
    assert saved.total() == sum(s.samples for s in fe.DEFAULT_PLAN)
    assert saved.labels() == ("hunter",)
    assert "VERDICT    usable" in out
    assert "SAVED      generation 1" in out
    assert feed.closed == 1, "the device must be released when the run ends"


def test_the_command_prints_numbers_and_never_an_image(monkeypatch, tmp_path,
                                                       capsys):
    """THE WHOLE WORKFLOW: he pastes this, and it can be judged without
    anybody ever seeing him."""
    wire(monkeypatch, tmp_path)
    code = face_enrol.main(ENROL + ["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    vr.assert_numbers_only(payload)
    result = payload["result"]
    for key in ("conf_p50", "face_px_p50", "eye_px_min", "yaw_spread",
                "cos_p50", "cohesion_min", "saved_generation"):
        assert key in result
    for sample in result["samples"]:
        assert set(sample) >= {"conf", "face_px", "eye_px", "yaw_deg",
                               "accepted", "reason"}


def test_the_command_refuses_a_gallery_built_at_one_head_position(
        monkeypatch, tmp_path, capsys):
    """TOO TIGHT. It would know him at his lens and reject him at his
    screen, silently, which is the expensive way for this to be wrong."""
    gallery, _feed = wire(monkeypatch, tmp_path, yaws=[0.0, 2.0, -2.0])
    code = face_enrol.main(ENROL)
    out = capsys.readouterr().out
    assert code == 1
    assert gallery.generations() == [], "a bad gallery reached the disk"
    assert "[FAIL] pose_spread" in out
    assert "This gallery was NOT saved" in out


def test_the_command_refuses_a_gallery_with_somebody_else_in_it(
        monkeypatch, tmp_path, capsys):
    """TOO LOOSE. match() scores against the pool's BEST member, so one
    sample that is not him is enough to identify the wrong person."""
    good = same_face(base_vec(51), 12, seed=8)
    vectors = good[:4] + [base_vec(4242)] + good[4:]
    gallery, _feed = wire(monkeypatch, tmp_path, vectors=vectors)
    code = face_enrol.main(ENROL)
    out = capsys.readouterr().out
    assert code == 1
    assert gallery.generations() == []
    assert "[FAIL] cohesion" in out


def test_the_command_opens_no_device_when_sensing_says_no(monkeypatch,
                                                          tmp_path, capsys):
    """Not a limitation of the script: the same rule the running Jarvis
    obeys, reached through the same object. ``build_feed`` raises if called."""
    gallery, _feed = wire(monkeypatch, tmp_path, camera=False,
                          feed_guard=True)
    code = face_enrol.main(ENROL)
    out = capsys.readouterr().out
    assert code == 2
    assert "sensing says the camera may not run" in out
    assert gallery.generations() == []


def test_enrolment_is_refused_while_the_identity_gate_is_off(monkeypatch,
                                                             tmp_path,
                                                             capsys):
    """``camera.identity`` false means nothing about his face is written
    down at all, and turning it on has to be a deliberate act."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    code = face_enrol.main(["--auto", "--gap-s", "0"])
    out = capsys.readouterr().out
    assert code == 1
    assert "camera.identity is false" in out
    assert gallery.generations() == []


def test_the_curfew_arriving_mid_run_stops_it_and_says_so(monkeypatch,
                                                          tmp_path, capsys):
    gallery, _feed = wire(monkeypatch, tmp_path,
                          source=FakeSource(stop_after=5))
    code = face_enrol.main(ENROL)
    out = capsys.readouterr().out
    assert code == 1
    assert "sensing denied the camera" in out
    assert gallery.generations() == []


def test_status_reports_the_gallery_and_opens_nothing(monkeypatch, tmp_path,
                                                      capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(["--status"]) == 3        # nothing enrolled yet
    assert "nothing enrolled yet" in capsys.readouterr().out
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    assert face_enrol.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "gen 00001" in out
    assert "cosine min" in out
    assert "NOTE ON BACKUP" in out


def test_status_json_is_numbers_only(monkeypatch, tmp_path, capsys):
    wire(monkeypatch, tmp_path)
    face_enrol.main(ENROL)
    capsys.readouterr()
    face_enrol.main(["--status", "--json"])
    vr.assert_numbers_only(json.loads(capsys.readouterr().out))


def test_backup_delete_and_restore_round_trip(monkeypatch, tmp_path, capsys):
    """The three commands that make his biometric data his: put a copy
    somewhere else, destroy the original, and put it back."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    before = FaceGallery(root=gallery.root)
    before.load()
    first = before.embeddings("hunter")[0].copy()
    capsys.readouterr()

    dest = tmp_path / "elsewhere"
    assert face_enrol.main(["--backup", str(dest)]) == 0
    assert "verified by reading them back" in capsys.readouterr().out

    assert face_enrol.main(["--delete", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "no longer on this disk" in out
    assert FaceGallery(root=gallery.root).generations() == []

    assert face_enrol.main(["--restore", str(dest)]) == 0
    assert "restored" in capsys.readouterr().out
    back = FaceGallery(root=gallery.root)
    assert back.load() is True
    assert back.total() == sum(s.samples for s in fe.DEFAULT_PLAN)
    assert cosine(back.embeddings("hunter")[0], first) == \
        pytest.approx(1.0, abs=1e-6)


def test_delete_turns_the_identity_flag_off_as_well(monkeypatch, tmp_path,
                                                    capsys):
    """Recognition running against a gallery he has just destroyed is a
    feature that is on and cannot work."""
    wire(monkeypatch, tmp_path)
    face_enrol.main(ENROL)
    cfg = face_enrol.AssistantConfig.load()
    assert cfg.get("camera.identity") is True
    capsys.readouterr()
    assert face_enrol.main(["--delete", "--yes"]) == 0
    assert "camera.identity set to false: True" in capsys.readouterr().out
    assert face_enrol.AssistantConfig.load().get("camera.identity") is False


def test_delete_without_confirmation_deletes_nothing(monkeypatch, tmp_path,
                                                     capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    face_enrol.main(ENROL)
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", lambda *_a: "no")
    assert face_enrol.main(["--delete"]) == 1
    assert "Not deleted" in capsys.readouterr().out
    assert gallery.generations() == [1]


def test_rollback_undoes_the_last_enrolment(monkeypatch, tmp_path, capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    face_enrol.main(ENROL)
    face_enrol.main(ENROL)
    assert gallery.generations() == [1, 2]
    capsys.readouterr()
    assert face_enrol.main(["--rollback"]) == 0
    assert "rolled back to generation 1" in capsys.readouterr().out
    assert FaceGallery(root=gallery.root).generations() == [1]


def test_the_second_enrolment_is_a_new_generation_not_an_overwrite(
        monkeypatch, tmp_path):
    """The property the voiceprint did not have on 2026-09-02."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    face_enrol.main(ENROL)
    face_enrol.main(ENROL)
    assert gallery.generations() == [1, 2]
    old = FaceGallery(root=gallery.root)
    assert old.load(generation=1) is True
    assert old.total() == sum(s.samples for s in fe.DEFAULT_PLAN)


def test_the_saved_files_are_0600_in_a_0700_directory(monkeypatch, tmp_path):
    gallery, _feed = wire(monkeypatch, tmp_path)
    face_enrol.main(ENROL)
    assert oct(os.stat(gallery.root).st_mode & 0o777) == "0o700"
    assert oct(os.stat(gallery.path_for(1)).st_mode & 0o777) == "0o600"


def test_verify_runs_the_stored_gallery_against_live_frames(monkeypatch,
                                                            tmp_path, capsys):
    """The matching path, end to end, reported as numbers -- which is the
    only way he can show anybody that recognition works."""
    base = base_vec(61)
    gallery, _feed = wire(monkeypatch, tmp_path,
                          vectors=same_face(base, 13, seed=12))
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    # A fresh detector/recogniser for the verify run: the same face again.
    det = ScriptedDetector([0.0, 40.0])
    rec = ScriptedRecogniser([base])
    monkeypatch.setattr(face_enrol, "build_models", lambda cfg: (det, rec, ""))
    monkeypatch.setattr(face_enrol, "build_feed",
                        lambda cfg, policy: (FakeFeed(FakeSource()), ""))
    face_enrol.main(["--verify", "--frames", "8", "--json"])
    payload = json.loads(capsys.readouterr().out)
    vr.assert_numbers_only(payload)
    result = payload["result"]
    assert result["id_matched_frames"] == 8
    assert result["id_unknown_frames"] == 0
    assert result["match_p50"] >= SFACE_COSINE_SAME
    assert result["gallery_enrolled"] == sum(s.samples
                                             for s in fe.DEFAULT_PLAN)


def test_verify_reports_a_stranger_as_unmatched_rather_than_as_him(
        monkeypatch, tmp_path, capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    monkeypatch.setattr(face_enrol, "build_models",
                        lambda cfg: (ScriptedDetector([0.0]),
                                     ScriptedRecogniser([base_vec(31337)]),
                                     ""))
    monkeypatch.setattr(face_enrol, "build_feed",
                        lambda cfg, policy: (FakeFeed(FakeSource()), ""))
    code = face_enrol.main(["--verify", "--frames", "6"])
    out = capsys.readouterr().out
    assert code == 1
    assert "[FAIL] gallery" in out
    assert "0 of 6 identified frames matched" in out


def test_verify_without_a_gallery_says_there_is_nothing_to_verify(
        monkeypatch, tmp_path, capsys):
    wire(monkeypatch, tmp_path)
    assert face_enrol.main(["--verify"]) == 3
    assert "nothing enrolled" in capsys.readouterr().out


def test_verify_opens_no_device_when_sensing_says_no(monkeypatch, tmp_path,
                                                     capsys):
    wire(monkeypatch, tmp_path)
    face_enrol.main(ENROL)
    capsys.readouterr()
    monkeypatch.setattr(face_enrol, "SensingPolicy",
                        lambda cfg=None: FakePolicy(camera=False,
                                                    reason="curfew"))

    def _never(cfg, policy):
        raise AssertionError("no device may be built during a curfew")

    monkeypatch.setattr(face_enrol, "build_feed", _never)
    assert face_enrol.main(["--verify"]) == 2
    assert "curfew" in capsys.readouterr().out


def test_a_stricter_identity_min_reaches_the_cohesion_check(tmp_path):
    """``camera.identity_min`` has to be USED, not merely printed. A bar he
    tightened after measuring his own face would otherwise change the report
    header and nothing else."""
    gallery, _det, _rec, session = a_good_session(tmp_path)
    rep, _ran = fe.run_enrolment(session, FakeSource(), say=lambda *_a: None,
                                 gap_s=0.0, identity_min=0.99)
    assert rep.identity_min == 0.99
    assert [c.ok for c in rep.checks if c.name == "cohesion"] == [False]
    assert "0.990" in [c.detail for c in rep.checks
                       if c.name == "cohesion"][0]


def test_the_cohesion_line_names_the_sample_he_can_find_in_the_report(
        tmp_path):
    """The embeddings are numbered over the ACCEPTED samples and the report's
    lines over every candidate. Naming the embedding's index would point at a
    different line the moment one frame was dropped -- in a report whose
    whole job is to be pasted and read by somebody else."""
    good = same_face(base_vec(71), 13, seed=15)
    vectors = good[:5] + [base_vec(4242)] + good[5:]
    det = ScriptedDetector(PLAN_YAWS,
                           conf=lambda i: 0.30 if i == 0 else 0.90)
    gallery = FaceGallery(root=tmp_path / "g")
    session = fe.EnrolmentSession(
        gallery, "hunter", LIFECAM_CINEMA, det,
        ScriptedRecogniser(vectors), fe.SampleLimits(min_conf=0.6))
    rep, _ran = fe.run_enrolment(session, FakeSource(), say=lambda *_a: None,
                                 gap_s=0.0)
    assert rep.samples[0].accepted is False        # dropped under the bar
    detail = [c.detail for c in rep.checks if c.name == "cohesion"][0]
    assert "#6" in detail, detail                  # the SAMPLE, not the 5th
    assert rep.samples[6].accepted is True


def test_a_refused_run_leaves_the_identity_flag_alone(monkeypatch, tmp_path,
                                                      capsys):
    """Sensing is asked before the config is written. A run that cannot
    happen must not leave ``camera.identity`` turned on behind it -- that is
    the switch that says his face may be written down."""
    wire(monkeypatch, tmp_path, camera=False, feed_guard=True)
    assert face_enrol.main(ENROL) == 2
    capsys.readouterr()
    assert face_enrol.AssistantConfig.load().get("camera.identity") is False
