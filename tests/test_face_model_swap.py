"""The swap: SCRFD + ArcFace behind the old seam, and the gallery migration.

The migration is the half that can hurt him, so most of this file is about it.
His enrolled vectors are SFace's 128 floats. An ArcFace vector is 512 and is
not weakly comparable with them -- the cosine between them measures nothing.
What is pinned here:

* a gallery for one model REFUSES to load another model's generation, and says
  which two models by name rather than scoring anything;
* his old generations are NOT destroyed by that refusal, NOT destroyed by a
  re-enrolment, and NOT evicted by five later saves (the prune window is
  per-model, or a successful re-enrolment would silently delete the thing he
  reverts to);
* the shrink guard does not compare across models either -- without that, a
  fresh 5-take ArcFace enrolment is refused as "shrinking from 13 to 5";
* a generation with no ``_model`` key is SFace's, because nothing else ever
  wrote this store, and the on-disk FORMAT is deliberately NOT bumped so those
  files stay readable;
* there is ONE line saying what he has to do, and it says "re-enrol".

Plus the two new models behind the existing interface: a fake onnxruntime
session, synthetic tensors, and pixels this file generates itself. No camera,
no weights, no network.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import facedetect as fd
from jarvis import facegallery as fg
from jarvis import faceinsight as fi
from jarvis import facemodels as fm
from jarvis import visionrig as vr


def sface_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=128).astype(np.float32)


def arc_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ===================================================== the gallery migration
def test_a_gallery_knows_which_model_it_is_for_and_refuses_the_other(tmp_path):
    old = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    old.add("hunter", sface_vec(1))
    old.save("sface enrolment")

    new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    with pytest.raises(ValueError) as exc:
        new.add("hunter", sface_vec(2))
    assert "wrong dimension" in str(exc.value)
    assert "512" in str(exc.value) and "arcface_mbf" in str(exc.value)


def test_the_old_generation_is_not_loaded_and_not_destroyed(tmp_path, caplog):
    old = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    for i in range(13):
        old.add("hunter", sface_vec(i))
    gen = old.save("his real enrolment")

    new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    assert new.load() is False
    assert new.total() == 0
    assert new.foreign_generations == {gen: "sface"}
    assert new.foreign_sample_count() == 13
    # the file is still there, byte for byte
    assert (tmp_path / ("gen-%05d.npz" % gen)).exists()
    # and the old gallery still loads it
    back = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    assert back.load() is True and back.total() == 13


def test_the_refusal_names_both_models_rather_than_scoring_anything(
        tmp_path, caplog):
    old = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    old.add("hunter", sface_vec(3))
    old.save("sface")
    new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    with caplog.at_level("WARNING"):
        new.load()
    text = caplog.text
    assert "sface" in text and "arcface_mbf" in text
    assert "NOT comparing across models" in text


def test_there_is_one_line_telling_him_to_re_enrol(tmp_path):
    old = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    for i in range(13):
        old.add("hunter", sface_vec(i))
    old.save("his real enrolment")
    new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    new.load()
    line = new.reenrol_message()
    assert "re-enrol" in line
    assert "arcface_mbf" in line and "sface" in line
    assert "13" in line
    assert "camera.face_backend" in line          # how to go back
    assert "\n" not in line                       # ONE line, for the log


def test_a_re_enrolment_does_not_have_to_beat_the_old_models_sample_count(
        tmp_path):
    """WITHOUT the per-model shrink guard his first ArcFace enrolment is
    REFUSED: 13 SFace samples on disk against 5 new ones reads as a shrink."""
    old = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    for i in range(13):
        old.add("hunter", sface_vec(i))
    old.save("sface")

    new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    new.load()
    for i in range(5):
        new.add("hunter", arc_vec(100 + i))
    gen = new.save("arcface enrolment")          # must not raise
    assert gen > 0
    again = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    assert again.load() is True and again.total() == 5


def test_five_re_enrolments_do_not_evict_his_old_model(tmp_path):
    """The incident's shape, one model change later: the prune window keeps
    the newest five, and five ArcFace saves would carry his SFace enrolment
    out of it -- deleting the thing he reverts to as a side effect of
    something that succeeded."""
    old = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    for i in range(13):
        old.add("hunter", sface_vec(i))
    sface_gen = old.save("sface")

    for round_ in range(8):
        new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
        new.load()
        for i in range(5):
            new.add("hunter", arc_vec(round_ * 10 + i))
        new.save("arcface round %d" % round_)

    assert (tmp_path / ("gen-%05d.npz" % sface_gen)).exists()
    back = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    assert back.load() is True and back.total() == 13


def test_a_rollback_never_shreds_the_other_models_generation(tmp_path):
    old = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    old.add("hunter", sface_vec(1))
    sface_gen = old.save("sface")
    new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    new.add("hunter", arc_vec(1))
    new.save("arcface one")
    # one ArcFace generation only: there is nothing of OURS to roll back to
    assert new.rollback() == 0
    assert (tmp_path / ("gen-%05d.npz" % sface_gen)).exists()


def test_a_generation_with_no_model_key_reads_as_sface(tmp_path):
    """Every file he already has is one of these. The on-disk FORMAT is
    deliberately NOT bumped, because ``_read`` refuses an unknown format and
    bumping it would make his enrolment unreadable -- taking away the thing
    the whole reversal argument rests on."""
    gal = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    gal.add("hunter", sface_vec(4))
    gen = gal.save("legacy")
    path = tmp_path / ("gen-%05d.npz" % gen)
    with np.load(path) as data:
        kept = {k: data[k] for k in data.files if k != "_model"}
    assert "_model" not in kept
    np.savez(path.with_suffix(""), **kept)       # rewrite without the key
    assert fg.FORMAT == 1

    again = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    assert again.load() is True and again.total() == 1
    assert again.provenance()["model"] == "sface"
    arc = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    assert arc.load() is False
    assert arc.foreign_generations == {gen: "sface"}


def test_a_generation_from_an_unknown_future_model_is_reported_not_destroyed(
        tmp_path):
    gal = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    gal.add("hunter", sface_vec(5))
    gen = gal.save("sface")
    path = tmp_path / ("gen-%05d.npz" % gen)
    with np.load(path) as data:
        kept = {k: data[k] for k in data.files}
    kept["_model"] = np.array(["something_from_2027"])
    np.savez(path.with_suffix(""), **kept)

    gal2 = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    assert gal2.load() is False
    assert path.exists()


def test_the_default_gallery_follows_the_active_backend(monkeypatch):
    monkeypatch.setenv(fm.BACKEND_ENV, "opencv")
    assert fg.default_gallery().model == "sface"
    monkeypatch.setenv(fm.BACKEND_ENV, "insightface")
    assert fg.default_gallery().model == "arcface_mbf"


def test_there_is_no_borrowed_threshold_for_the_new_model():
    """0.363 is OpenCV's published number for SFace's vectors. Carried across
    to ArcFace it would be a bar that had stopped meaning anything."""
    assert fg.cosine_same("sface") == pytest.approx(0.363)
    assert fg.cosine_same("arcface_mbf") is None
    assert fg.ARCFACE_COSINE_SAME is None


def test_dimensions_are_per_model_and_an_unknown_model_raises():
    assert fg.model_dim("sface") == 128
    assert fg.model_dim("arcface_mbf") == 512
    with pytest.raises(ValueError, match="unknown embedding model"):
        fg.model_dim("buffalo_l")
    with pytest.raises(ValueError, match="unknown embedding model"):
        fg.FaceGallery(root=None, model="buffalo_l")


# ============================ deleting one person during the migration window
#
# THE WINDOW IS THE DANGEROUS PART. Between the swap landing and his
# re-enrolment, every vector on the disk is SFace's and every gallery the code
# opens is ArcFace's. A delete that runs in that window used to destroy the
# generations it could not rewrite -- taking everybody ELSE in them with the
# one person who asked to go.
def _mixed_store(root, hunter_n: int = 6, heather_n: int = 4) -> int:
    """His SFace enrolment, holding two people. Returns the generation."""
    old = fg.FaceGallery(root=root, model=fg.SFACE_MODEL)
    for i in range(hunter_n):
        old.add("hunter", sface_vec(100 + i))
    for i in range(heather_n):
        old.add("heather", sface_vec(200 + i))
    return old.save("sface enrolment")


def test_deleting_one_person_in_the_window_may_not_take_the_others(tmp_path):
    """THE ONE THAT DESTROYS HIS DATA.

    ``purge_label`` finds the generations that hold her by reading the files
    RAW, which sees across models; then ``load()`` refuses the cross-model
    generation and loads nothing; then "no survivors means no save" fires over
    a pool that is empty only because it could not be read -- and the
    generation is shredded anyway. His enrolment goes with hers."""
    gen = _mixed_store(tmp_path)
    new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    out = new.purge_label("heather", reason="test")

    assert (tmp_path / ("gen-%05d.npz" % gen)).exists(), \
        "a generation this build cannot rewrite was destroyed anyway"
    back = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    assert back.load() is True
    assert back.count("hunter") == 6, "his enrolment went with hers"
    assert out["removed"] == 0


def test_a_delete_it_could_not_carry_out_is_never_called_complete(tmp_path):
    """And she may not be told she is gone while she is still on the disk.
    The generation is COUNTED and REPORTED, exactly like an unreadable one."""
    gen = _mixed_store(tmp_path)
    new = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    out = new.purge_label("heather", reason="test")

    assert out["foreign"] == [gen]
    assert out["foreign_models"] == (fg.SFACE_MODEL,)
    assert out["complete"] is False
    back = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    back.load()
    assert back.count("heather") == 4, "the report must not be a lie"


def test_a_delete_inside_one_model_still_removes_her_and_reports_complete(
        tmp_path):
    """The control. Nothing above may make an ordinary delete timid."""
    g = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    for i in range(6):
        g.add("hunter", arc_vec(300 + i))
    for i in range(4):
        g.add("heather", arc_vec(400 + i))
    g.save("arcface enrolment")

    out = fg.FaceGallery(root=tmp_path,
                         model=fg.ARCFACE_MODEL).purge_label("heather")
    assert out["complete"] is True
    assert out["foreign"] == [] and out["not_carried"] == []
    assert out["removed"] == 1
    assert out["labels_left"] == ("hunter",)
    for path in tmp_path.iterdir():
        assert b"heather" not in path.read_bytes()


def test_a_generation_whose_survivors_were_not_rewritten_is_left_alone(
        tmp_path):
    """The same rule with no model swap in sight: the newest generation holds
    only her, an older one holds him as well. "What is left" is nothing, so
    there is no save -- and destroying the older generation would take his
    only samples with her."""
    g = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    for i in range(6):
        g.add("hunter", arc_vec(300 + i))
    for i in range(4):
        g.add("heather", arc_vec(400 + i))
    g.save("both of them")
    just_her = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    for i in range(4):
        just_her.add("heather", arc_vec(400 + i))
    just_her.save("just her", allow_shrink=True)

    out = fg.FaceGallery(root=tmp_path,
                         model=fg.ARCFACE_MODEL).purge_label("heather")
    assert out["not_carried"] == [1]
    assert out["complete"] is False
    assert out["removed"] == 1, "the one holding nobody else still goes"
    back = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    assert back.load(generation=1) is True
    assert back.count("hunter") == 6


# ------------------------------------- the two callers that dropped the model
def test_the_read_back_reads_the_generation_as_the_caller_s_own_model(
        tmp_path):
    """``_generation_holds`` used to build a gallery with NO model, so it was
    SFace's while its caller was ArcFace's: the cross-model refusal returned
    False over a generation that was still there, and the command printed
    "verified" over her embeddings. The model is required and has no
    default -- a default is how it got dropped."""
    from tests.test_faceenrol import face_enrol

    g = fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL)
    g.add("heather", arc_vec(1))
    g.add("hunter", arc_vec(2))
    g.save("arcface")

    assert face_enrol._generation_holds(tmp_path, 1, "heather",
                                        fg.ARCFACE_MODEL) is True
    assert face_enrol._generation_holds(tmp_path, 1, "nobody",
                                        fg.ARCFACE_MODEL) is False
    with pytest.raises(TypeError):
        face_enrol._generation_holds(tmp_path, 1, "heather")


def test_the_command_will_not_say_verified_over_another_model_s_generation(
        tmp_path):
    from tests.test_faceenrol import face_enrol

    _mixed_store(tmp_path)
    said = []
    code, out = face_enrol.do_delete_label(
        fg.FaceGallery(root=tmp_path, model=fg.ARCFACE_MODEL),
        "heather", None, said.append)
    text = "\n".join(said)
    assert code == 1, text
    assert "verified" not in text
    assert "--delete with no --label" in text
    assert (tmp_path / "gen-00001.npz").exists()
    assert out["complete"] is False


def test_the_voice_path_opens_the_gallery_for_the_configured_backend(
        monkeypatch):
    """``_face_gallery`` built the DEFAULT gallery, so "who do you recognise"
    and the voice delete could open the wrong model's store -- answering
    "nobody" over an enrolment that is right there."""
    from jarvis import commander as cm

    class Cfg:
        def get(self, key, default=None):
            return "opencv" if key == "camera.face_backend" else default

    class Ctx:
        def _svc(self, name):
            return Cfg() if name == "assistant" else None

    monkeypatch.setenv(fm.BACKEND_ENV, "insightface")
    assert cm._face_gallery(Ctx()).model == fg.SFACE_MODEL

# ============================================================== the detector
class FakeInput:
    name = "input.1"


class FakeSession:
    """onnxruntime's surface, with the outputs planted."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.blobs = []

    def get_inputs(self):
        return [FakeInput()]

    def run(self, _names, feed):
        self.blobs.append(feed["input.1"])
        return self.outputs


def blank_outputs(width=320, height=192):
    out = []
    for cols in (1, 4, 10):
        for stride in fi.STRIDES:
            n = (height // stride) * (width // stride) * fi.NUM_ANCHORS
            out.append(np.zeros((n, cols), dtype=np.float32))
    return out


def test_the_detector_pads_to_a_multiple_of_32_and_keeps_the_detect_space():
    """320x180 is the shipped detect size, 180 is not divisible by 32, and
    RESIZING to 192 would add a vertical stretch the rig cannot see."""
    outs = blank_outputs()
    sess = FakeSession(outs)
    det = fd.ScrfdDetector(sess, (320, 180), 0.3,
                           resize=lambda f, s: np.zeros((s[1], s[0], 3),
                                                        np.uint8))
    assert det.input_size == (320, 180)
    assert det.padded_size == (192, 320)
    det.detect(np.zeros((720, 1280, 3), np.uint8))
    blob = sess.blobs[0]
    assert blob.shape == (1, 3, 192, 320)


def test_the_padding_is_zeros_along_the_bottom_only():
    outs = blank_outputs()
    sess = FakeSession(outs)
    det = fd.ScrfdDetector(sess, (320, 180), 0.3,
                           resize=lambda f, s: np.full((s[1], s[0], 3), 200,
                                                       np.uint8))
    det.detect(np.zeros((720, 1280, 3), np.uint8))
    blob = sess.blobs[0]
    bright = (200 - 127.5) / 128.0
    dark = (0 - 127.5) / 128.0
    assert blob[0, 0, 0, 0] == pytest.approx(bright)
    assert blob[0, 0, 179, 0] == pytest.approx(bright)
    assert blob[0, 0, 180, 0] == pytest.approx(dark)


def test_the_detector_returns_yunet_shaped_rows_the_rig_can_read():
    outs = blank_outputs()
    row = (3 * 20 + 5) * fi.NUM_ANCHORS          # stride 16, cell (5, 3)
    outs[1][row, 0] = 0.91
    outs[4][row] = [1.0, 1.0, 1.0, 1.0]
    outs[7][row] = [-0.5, -0.5, 0.5, -0.5, 0.0, 0.0, -0.4, 0.6, 0.4, 0.6]
    det = fd.ScrfdDetector(FakeSession(outs), (320, 180), 0.3,
                           resize=lambda f, s: np.zeros((s[1], s[0], 3),
                                                        np.uint8))
    rows = det.detect(np.zeros((720, 1280, 3), np.uint8))
    assert rows.shape == (1, vr.DETECT_COLS)
    assert rows[0][vr.IDX_SCORE] == pytest.approx(0.91)
    assert rows[0][:4].tolist() == [64.0, 32.0, 32.0, 32.0]
    _t, roll, _eye, ok = vr.landmark_geometry(rows[0])
    assert ok and abs(roll) < 1.0
    _ratio, drop, shaped = vr.landmark_plausibility(rows[0])
    assert shaped and drop > vr.MOUTH_DROP_MIN_U


def test_no_faces_is_none_not_an_empty_array():
    """The rig's "empty room" and "blind detector" paths are different, and
    OpenCV's detector returns None. This one must too."""
    det = fd.ScrfdDetector(FakeSession(blank_outputs()), (320, 180), 0.3,
                           resize=lambda f, s: np.zeros((s[1], s[0], 3),
                                                        np.uint8))
    assert det.detect(np.zeros((720, 1280, 3), np.uint8)) is None


def test_the_detector_name_is_not_yunets():
    det = fd.ScrfdDetector(FakeSession(blank_outputs()), (320, 180), 0.3,
                           resize=lambda f, s: np.zeros((s[1], s[0], 3),
                                                        np.uint8))
    assert det.name == "scrfd_500m"
    assert fd.YuNetDetector.name == "yunet"


def test_the_score_floor_is_still_a_floor_and_never_his_bar(tmp_path,
                                                            monkeypatch):
    """The silent-miss trap, unchanged by the swap: a face scoring 0.45 must
    be REPORTED and judged, not filtered out by the detector."""
    monkeypatch.setenv("JARVIS_FACE_MODEL_DIR", str(tmp_path))
    for model in fm.INSIGHT.models():
        (tmp_path / model.filename).write_bytes(b"\0" * model.size)
    made = {}

    def create(path, threads=2, **kw):
        made["path"] = path
        return FakeSession(blank_outputs())

    det = fd.load_detector(min_conf=0.6, input_size=(320, 180), create=create)
    assert det.score_threshold == pytest.approx(fd.PROBE_THRESHOLD)
    assert det.score_threshold < 0.6
    assert made["path"].endswith("det_500m.onnx")


# ============================================================ the recogniser
class FakeArcSession(FakeSession):
    def __init__(self, vector=None):
        super().__init__(None)
        self.vector = (np.arange(512, dtype=np.float32) if vector is None
                       else vector)
        self.crops = []

    def run(self, _names, feed):
        blob = feed["input.1"]
        self.crops.append(blob)
        return [self.vector.reshape(1, 512)]


def frontal_row(conf=0.9, scale=1.0, offset=(0.0, 0.0)):
    """A YuNet-shaped row whose landmarks are the ArcFace template, moved."""
    pts = fi.ARCFACE_TEMPLATE * scale + np.asarray(offset)
    box = [pts[:, 0].min(), pts[:, 1].min(),
           np.ptp(pts[:, 0]), np.ptp(pts[:, 1])]
    return np.array(box + pts.reshape(10).tolist() + [conf], dtype=np.float32)


def test_the_recogniser_refuses_a_weak_detection_and_a_nan_one():
    """ArcFace collapses on out-of-distribution input MORE than SFace does --
    every synthetic family above 0.363 for every pair, measured 2026-09-03 --
    so this gate matters more after the swap, not less."""
    rec = fd.ArcFaceRecogniser(FakeArcSession(), min_conf=0.6)
    frame = np.zeros((200, 200, 3), np.uint8)
    with pytest.raises(ValueError, match="0.45"):
        rec.embed(frame, frontal_row(conf=0.45))
    with pytest.raises(ValueError, match="under the"):
        rec.embed(frame, frontal_row(conf=float("nan")))


def test_the_recogniser_refuses_a_row_of_the_wrong_shape():
    rec = fd.ArcFaceRecogniser(FakeArcSession(), min_conf=0.6)
    row = np.concatenate([frontal_row(), [0.0]])
    with pytest.raises(ValueError, match="column detection row"):
        rec.embed(np.zeros((200, 200, 3), np.uint8), row)


def test_the_embedding_is_512_and_unit_length():
    sess = FakeArcSession()
    rec = fd.ArcFaceRecogniser(sess, min_conf=0.6)
    vec = rec.embed(np.zeros((200, 200, 3), np.uint8), frontal_row())
    assert vec.shape == (512,)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-6)
    blob = sess.crops[0]
    assert blob.shape == (1, 3, 112, 112)


def test_the_crop_comes_from_where_the_face_is_not_the_top_left_corner():
    """THE BUG THE SFACE PATH HAS TODAY. Every call site passes the FULL
    capture frame with a row in DETECTOR pixels. Measured 2026-09-03 with
    synthetic arrays: cv2's alignCrop given that pair reads the frame's
    top-left 320x180 corner. ArcFace scales the landmarks first, so its crop
    lands on the face.

    Synthetic pixels only: one bright block, drawn here, at the CAPTURE-space
    location the detect-space landmarks correspond to.
    """
    frame = np.zeros((720, 1280, 3), np.uint8)
    row = frontal_row(scale=0.6, offset=(120.0, 60.0))
    kps = row[4:14].reshape(5, 2)
    sx, sy = 1280.0 / 320.0, 720.0 / 180.0
    big = kps * np.array([sx, sy])
    x0, y0 = int(big[:, 0].min()) - 150, int(big[:, 1].min()) - 150
    x1, y1 = int(big[:, 0].max()) + 150, int(big[:, 1].max()) + 150
    frame[y0:y1, x0:x1] = 220

    scaled = fd.ArcFaceRecogniser(FakeArcSession(), min_conf=0.6,
                                  input_size=(320, 180))
    unscaled = fd.ArcFaceRecogniser(FakeArcSession(), min_conf=0.6)
    scaled.embed(frame, row)
    unscaled.embed(frame, row)
    black = (0 - 127.5) / 128.0
    bright = (220 - 127.5) / 128.0
    # the scaled crop is looking at the block; the unscaled one at nothing
    assert scaled._session.crops[0].mean() == pytest.approx(bright, abs=1e-5)
    assert unscaled._session.crops[0].mean() == pytest.approx(black, abs=1e-5)


def test_the_alignment_puts_the_template_on_the_template():
    """A wrong template loses accuracy and raises nothing, so the check has
    to be on the matrix rather than on the output."""
    rec = fd.ArcFaceRecogniser(FakeArcSession(), min_conf=0.6)
    row = frontal_row()
    kps = np.asarray(row[4:14]).reshape(5, 2)
    matrix = fi.arcface_matrix(kps)
    placed = kps @ matrix[:, :2].T + matrix[:, 2]
    assert np.allclose(placed, fi.ARCFACE_TEMPLATE, atol=1e-8)
    assert rec.scales(np.zeros((720, 1280, 3))) == (1.0, 1.0)


def test_the_scale_is_read_from_the_frame_and_the_detector_size():
    rec = fd.ArcFaceRecogniser(FakeArcSession(), min_conf=0.6,
                               input_size=(320, 180))
    assert rec.scales(np.zeros((720, 1280, 3))) == (4.0, 4.0)
    assert rec.scales(np.zeros((480, 640, 3))) == (2.0, pytest.approx(2.6667,
                                                                     rel=1e-3))


def test_sface_is_deliberately_left_exactly_as_it_was():
    """Reverting to the OpenCV backend has to give back TODAY's behaviour,
    warts included: his enrolled vectors were built through that path, and
    fixing only the query side would make stored and live vectors measure
    different things -- a total, silent loss of recognition."""
    import inspect
    src = inspect.getsource(fd.SFaceRecogniser.embed)
    assert "alignCrop" in src
    assert "scale_landmarks" not in src
    assert "input_size" not in src
