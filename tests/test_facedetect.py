"""Loading the face models, or saying exactly why not.

THE BUG THIS FILE EXISTS TO PREVENT ALREADY SHIPPED ONCE, on this machine:
VSS has a ``region_mode='yunet'`` path that silently falls back to head_box
and has never once executed. Nobody noticed because a silent fallback and a
working feature produce the same logs.

So every failure here is NAMED and every failure RAISES. There is no second
detector to fall through to, an absent weight file is not an empty room, and
a git-lfs pointer -- which is what the plain raw.githubusercontent URL
actually serves -- is reported as a pointer rather than as a corrupt model.

No cv2, no weights, no camera: the OpenCV constructor is an injected seam,
exactly as ``open_device`` is in tests/test_eye.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import facedetect as fd
from jarvis import facemodels as fm
from jarvis.visionrig import assert_numbers_only


@pytest.fixture()
def empty_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_FACE_MODEL_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def stocked_dir(tmp_path, monkeypatch):
    """Files of the RIGHT SIZE but the wrong content -- enough for the
    shallow check, which is what production uses."""
    monkeypatch.setenv("JARVIS_FACE_MODEL_DIR", str(tmp_path))
    for model in fm.ALL_MODELS:
        (tmp_path / model.filename).write_bytes(b"\0" * model.size)
    return tmp_path


class Recorder:
    """Stands in for cv2.FaceDetectorYN_create / FaceRecognizerSF_create."""

    def __init__(self, net=None):
        self.calls = []
        self.net = net or FakeNet()

    def __call__(self, path, **kw):
        self.calls.append((path, kw))
        return self.net


class FakeNet:
    def __init__(self, faces=None):
        self.faces = faces
        self.sizes = []

    def setInputSize(self, size):
        self.sizes.append(tuple(size))

    def detect(self, frame):
        return 1, self.faces


# ------------------------------------------------------ absence, named
def test_a_missing_detector_names_the_file_and_raises(empty_dir):
    with pytest.raises(fd.ModelUnavailable) as exc:
        fd.load_detector(create=Recorder(), backend="opencv")
    assert fm.YUNET.filename in str(exc.value)
    assert "missing" in str(exc.value)
    with pytest.raises(fd.ModelUnavailable) as exc:
        fd.load_detector(create=Recorder(), backend="insightface")
    assert fm.SCRFD_500M.filename in str(exc.value)


def test_a_missing_recogniser_names_the_file_and_raises(empty_dir):
    with pytest.raises(fd.ModelUnavailable) as exc:
        fd.load_recogniser(create=Recorder(), backend="opencv")
    assert fm.SFACE.filename in str(exc.value)
    with pytest.raises(fd.ModelUnavailable) as exc:
        fd.load_recogniser(create=Recorder(), backend="insightface")
    assert fm.ARCFACE_MBF.filename in str(exc.value)


def test_a_git_lfs_pointer_is_reported_as_a_pointer(empty_dir):
    """The plain raw.githubusercontent URL returns a ~130 byte pointer. It
    would otherwise read as a corrupt or truncated download, and the fix for
    those two is not the same fix."""
    (empty_dir / fm.YUNET.filename).write_bytes(b"version https://git-lfs" * 5)
    with pytest.raises(fd.ModelUnavailable) as exc:
        fd.load_detector(create=Recorder(), backend="opencv")
    assert "git-lfs pointer" in str(exc.value)


def test_there_is_no_second_detector_to_fall_back_to(empty_dir, monkeypatch):
    """The VSS shape, asserted directly: with the model unusable, loading
    must raise rather than return SOMETHING that runs."""
    monkeypatch.setattr(fm, "verify",
                        lambda m, deep=False, model_dir=None:
                        (False, "made up reason"))
    for loader in (fd.load_detector, fd.load_recogniser):
        with pytest.raises(fd.ModelUnavailable) as exc:
            loader(create=Recorder(), backend="opencv")
        assert "made up reason" in str(exc.value)


def test_no_opencv_is_a_named_failure_not_a_crash(stocked_dir, monkeypatch):
    def boom(*a, **k):
        raise ImportError("No module named 'cv2'")

    monkeypatch.setattr(fd, "_import_cv2", boom)
    with pytest.raises(fd.ModelUnavailable) as exc:
        fd.load_detector(backend="opencv")
    assert "cv2" in str(exc.value)


# ------------------------------------------------------------- the probe
def test_probe_reports_every_model_and_never_raises(empty_dir):
    report = fd.probe(backend="opencv")
    assert set(report["models"]) == {"detector", "recogniser"}
    assert report["ready"] is False
    assert fm.YUNET.filename in report["models"]["detector"]["reason"]
    assert report["models"]["detector"]["licence"] == "MIT"
    assert report["models"]["recogniser"]["licence"] == "Apache-2.0"
    assert_numbers_only(report)


def test_probe_says_ready_when_the_files_are_there(stocked_dir):
    report = fd.probe(backend="opencv")
    assert report["ready"] is True
    assert report["models"]["detector"]["ok"] is True
    assert report["dir"] == str(stocked_dir)


def test_probe_names_the_backend_and_the_licence_it_is_running_under(
        stocked_dir):
    """"The detector is missing" and "the detector you are not using is
    missing" are different sentences. The report has to be able to say which,
    and it has to carry the non-commercial term where he will see it."""
    report = fd.probe()
    assert report["backend"] == "insightface"
    assert report["embed_dim"] == 512
    assert report["embed_model"] == "arcface_mbf"
    assert report["commercial_ok"] is False
    assert report["licence"] == "NON-COMMERCIAL RESEARCH ONLY"
    assert set(report["backends"]) == {"opencv", "insightface"}
    assert report["backends"]["opencv"]["commercial_ok"] is True
    assert report["backends"]["opencv"]["embed_dim"] == 128
    assert_numbers_only(report)


def test_probe_reports_the_onnxruntime_providers(empty_dir):
    """The GB10 is unreachable from this ORT build, which is WHY mbf and not
    r50. A report that does not say which providers exist makes the next
    person guess."""
    report = fd.probe()
    assert "onnxruntime" in report
    assert isinstance(report["onnxruntime"]["available"], bool)
    assert isinstance(report["onnxruntime"]["providers"], str)


def test_probe_reports_whether_opencv_can_even_build_them(empty_dir):
    """A box with opencv-python-headless, or with a 4.x too old for
    FaceDetectorYN, fails at construction and not at the file check."""
    report = fd.probe()
    assert "cv2" in report
    assert isinstance(report["cv2"]["available"], bool)


# ------------------------------------------------- the pinned model choice
def test_the_opencv5_export_is_never_the_one_loaded(stocked_dir):
    rec = Recorder()
    fd.load_detector(create=rec, backend="opencv")
    path = rec.calls[0][0]
    assert path.endswith(fm.YUNET.filename)
    assert fm.DETECTOR_OPENCV5 not in path
    assert "2023mar" in path


def test_int8_is_never_the_one_loaded(stocked_dir):
    """int8 measured SLOWER than fp32 on this aarch64 build at every thread
    count -- SFace 23.2 ms against 21.1 single-threaded."""
    rec = Recorder()
    fd.load_detector(create=rec, backend="opencv")
    fd.load_recogniser(create=rec, backend="opencv")
    import os
    assert all("int8" not in os.path.basename(path)
               for path, _ in rec.calls)


# ------------------------------------ the floor is not the verdict
def test_the_detector_floor_is_lower_than_his_confidence_bar(stocked_dir):
    """THE SILENT-MISS TRAP. If the detector's own score_threshold were set
    to camera.min_conf, a face that scores 0.45 would simply not be
    reported, and the report would say "no face" instead of "your face
    scores 0.45 against your 0.6 bar". So the detector floors low and the RIG
    judges the bar."""
    rec = Recorder()
    fd.load_detector(create=rec, score_threshold=0.3, backend="opencv")
    assert rec.calls[0][1]["score_threshold"] == pytest.approx(0.3)
    assert fd.PROBE_THRESHOLD < 0.6


def test_the_input_size_and_thread_count_reach_opencv(stocked_dir):
    rec = Recorder()
    det = fd.load_detector(create=rec, input_size=(320, 180), threads=2,
                            backend="opencv")
    assert det.input_size == (320, 180)
    assert rec.calls[0][1]["input_size"] == (320, 180)
    assert rec.calls[0][1]["threads"] == 2


# --------------------------------------------------------- the detector
def test_detect_resizes_to_the_input_size_and_returns_rows(stocked_dir):
    rows = np.zeros((2, 15), dtype=np.float32)
    seen = {}

    def resize(frame, size):
        seen["size"] = size
        return frame

    det = fd.YuNetDetector(FakeNet(rows), (320, 180), resize=resize)
    out = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert seen["size"] == (320, 180)
    assert out is rows


def test_no_faces_is_none_not_an_empty_guess(stocked_dir):
    det = fd.YuNetDetector(FakeNet(None), (320, 180),
                           resize=lambda f, s: f)
    assert det.detect(np.zeros((4, 4, 3), dtype=np.uint8)) is None


# ------------------------------------------- identity is gated structurally
def test_the_recogniser_refuses_a_face_that_did_not_clear_the_bar():
    """SFace collapses on out-of-distribution input: unrelated non-face crops
    match each other at cosine 0.66-0.92, i.e. CONFIDENTLY above the 0.363
    "same person" bar. So a bad crop does not score low against the gallery,
    it scores high -- and the only defence is never to embed a detection
    that did not already clear min_conf. That is enforced here as well as in
    the rig, because two call sites will eventually exist."""
    class Net:
        def alignCrop(self, frame, row):
            return frame

        def feature(self, crop):
            return np.ones((1, 128), dtype=np.float32)

    rec = fd.SFaceRecogniser(Net(), min_conf=0.6)
    row = np.zeros(15, dtype=np.float32)
    row[14] = 0.41
    with pytest.raises(ValueError) as exc:
        rec.embed(np.zeros((8, 8, 3), dtype=np.uint8), row)
    assert "0.41" in str(exc.value)
    row[14] = 0.72
    assert rec.embed(np.zeros((8, 8, 3), dtype=np.uint8), row).shape == (128,)
