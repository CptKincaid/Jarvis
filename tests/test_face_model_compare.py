"""The instrument he runs (scripts/face_model_compare.py), and its rules.

Accuracy needs his face; nothing here may look at his face; so the code does
not measure accuracy, it BUILDS THE THING THAT DOES and he runs it. That makes
this test file about two separate promises.

THE PRIVACY PROMISE, checked mechanically and twice: the script's own CODE
names no call that could turn data into a picture, checked by its self-check
and by a token-level grep from outside that does not trust the self-check; its
forbidden list is byte-identical to ``scripts/camera_mode_probe.py``'s so the
two cannot drift; and every line it prints is a count, a cosine, a degree or a
millisecond, enforced by ``visionrig.assert_numbers_only`` over the payload.

THE MEASUREMENT PROMISE: the numbers are the ones the question needs. His
question was *"he also is recognizing me less from the side angle"*, so the
same-person cosines are split by pose -- and a gallery whose takes carry no
angle gets a refusal to answer rather than an average over the ignorance. The
two models are never compared with each other's vectors. And the suggested
identity bar is set from FACE evidence only: folding the synthetic non-face
probe into it makes every verdict read "no separation", because both models
score two pieces of nothing at each other near 1.0.

No camera, no weights (``--no-probe``), no network.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
import tokenize

import numpy as np
import pytest

from jarvis import facegallery as fg
from jarvis.visionrig import assert_numbers_only

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_HERE, "scripts", "face_model_compare.py")
_SPEC = importlib.util.spec_from_file_location("face_model_compare", _PATH)
compare = importlib.util.module_from_spec(_SPEC)
sys.modules["face_model_compare"] = compare
_SPEC.loader.exec_module(compare)

_PROBE_PATH = os.path.join(_HERE, "scripts", "camera_mode_probe.py")


def _code_only(path: str) -> str:
    """The file with every comment and string literal removed."""
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


# ------------------------------------------------------------- the rule
def test_it_names_no_call_that_could_show_or_save_a_frame():
    source = _code_only(_PATH)
    for pattern in (r"\bread\s*\(", r"\bretrieve\b", r"\bimshow\b",
                    r"\bimwrite\b", r"\bimencode\b", r"\bimdecode\b",
                    r"\bnamedWindow\b", r"\bwaitKey\b", r"\btofile\b",
                    r"\.save\s*\(", r"\bVideoCapture\b", r"\bPIL\b"):
        assert not re.search(pattern, source), pattern
    assert compare.self_check() == []


def test_its_forbidden_list_is_the_same_one_the_mode_probe_uses():
    """Two scripts with two lists is one list that will drift."""
    probe_spec = importlib.util.spec_from_file_location("_probe", _PROBE_PATH)
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    assert compare.PIXEL_CALLS == probe.PIXEL_CALLS


def test_the_self_check_catches_a_planted_read(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import cv2\ncap = cv2.VideoCapture(0)\n"
                   "ok, frame = cap.read()\n")
    hits = compare.self_check(str(bad))
    assert hits and hits[0].startswith("read line 3")


def test_a_planted_read_stops_it_before_anything_runs(monkeypatch, capsys):
    monkeypatch.setattr(compare, "self_check", lambda: ["read line 1"])
    assert compare.main(["--no-probe"]) == 4
    assert "REFUSING TO RUN" in capsys.readouterr().out


# ------------------------------------------------------------- fixtures
def _stock(root, *, angles=True, second_person=True):
    """An SFace generation with no angles (his live one's shape), plus an
    ArcFace one with angles. Vectors are numpy, made here."""
    rng = np.random.default_rng(11)
    old = fg.FaceGallery(root=root, model=fg.SFACE_MODEL)
    base = rng.normal(size=128)
    for _ in range(6):
        old.add("hunter",
                (base + rng.normal(scale=0.3, size=128)).astype(np.float32))
    old.save("pre-swap")

    new = fg.FaceGallery(root=root, model=fg.ARCFACE_MODEL)
    b1 = rng.normal(size=512)
    for yaw in (0.0, 4.0, -3.0, 30.0, -36.0):
        v = b1 + rng.normal(scale=0.3, size=512)
        new.add("hunter", (v / np.linalg.norm(v)).astype(np.float32),
                yaw_deg=(yaw if angles else None))
    if second_person:
        b2 = rng.normal(size=512)
        for yaw in (0.0, 20.0):
            v = b2 + rng.normal(scale=0.3, size=512)
            new.add("heather", (v / np.linalg.norm(v)).astype(np.float32),
                    yaw_deg=(yaw if angles else None))
    new.save("post-swap")
    return root


class Args:
    def __init__(self, gallery, profile_deg=15.0, no_probe=True, json=False):
        self.gallery = str(gallery)
        self.profile_deg = profile_deg
        self.no_probe = no_probe
        self.json = json


class Cfg:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


# ------------------------------------------------------- what it reports
def test_the_payload_is_numbers_strings_and_containers_only(tmp_path):
    _stock(tmp_path)
    payload = compare.collect(Args(tmp_path), Cfg())
    assert_numbers_only(payload)          # raises on an array


def test_it_reports_both_models_and_never_mixes_their_vectors(tmp_path):
    _stock(tmp_path)
    payload = compare.collect(Args(tmp_path), Cfg())
    by_model = {m["model"]: m for m in payload["models"]}
    assert set(by_model) == {"sface", "arcface_mbf"}
    assert by_model["sface"]["dim"] == 128
    assert by_model["arcface_mbf"]["dim"] == 512
    assert by_model["sface"]["loaded"] and by_model["arcface_mbf"]["loaded"]
    # 6 sface takes -> 15 pairs; 7 arcface takes over two people -> 10 + 1
    assert by_model["sface"]["intra"]["n"] == 15
    assert by_model["arcface_mbf"]["intra"]["n"] == 11
    assert by_model["arcface_mbf"]["cross_person"]["n"] == 10
    # and no key anywhere carries a comparison BETWEEN the models
    assert "cross_model" not in json.dumps(payload)


def test_the_published_bar_is_named_for_sface_and_missing_for_arcface(
        tmp_path):
    _stock(tmp_path)
    payload = compare.collect(Args(tmp_path), Cfg())
    by_model = {m["model"]: m for m in payload["models"]}
    assert by_model["sface"]["published_cosine_same"] == pytest.approx(0.363)
    assert by_model["arcface_mbf"]["published_cosine_same"] is None


# ------------------------------------------------- the side-angle question
def test_the_pose_split_is_the_answer_to_his_actual_question(tmp_path):
    _stock(tmp_path)
    payload = compare.collect(Args(tmp_path), Cfg())
    arc = [m for m in payload["models"] if m["model"] == "arcface_mbf"][0]
    # hunter has 3 frontal + 2 profile -> 3 f/f, 6 f/p, 1 p/p; heather has
    # one of each -> one more f/p. Pairs are only ever formed WITHIN a label.
    assert arc["intra_frontal_frontal"]["n"] == 3
    assert arc["intra_frontal_profile"]["n"] == 7
    assert arc["intra_profile_profile"]["n"] == 1
    assert arc["angles_recorded"] == 7


def test_the_profile_cut_is_a_flag_and_moving_it_moves_the_split(tmp_path):
    _stock(tmp_path)
    wide = compare.collect(Args(tmp_path, profile_deg=45.0), Cfg())
    arc = [m for m in wide["models"] if m["model"] == "arcface_mbf"][0]
    assert arc["intra_profile_profile"]["n"] == 0     # nothing is profile now
    assert arc["intra_frontal_profile"]["n"] == 0
    assert arc["intra_frontal_frontal"]["n"] == 11    # every same-label pair


def test_a_gallery_with_no_angles_refuses_the_question_rather_than_averaging(
        tmp_path, capsys):
    """His live generation carries no angles at all. Reporting 0.0 for those
    would claim a fully frontal enrolment that was never measured."""
    _stock(tmp_path, angles=False)
    payload = compare.collect(Args(tmp_path), Cfg())
    arc = [m for m in payload["models"] if m["model"] == "arcface_mbf"][0]
    assert arc["angles_recorded"] == 0
    assert arc["intra"]["n"] > 0                      # still has a floor
    assert arc["intra_frontal_frontal"]["n"] == 0
    compare.report(payload, print)
    out = capsys.readouterr().out
    assert "CANNOT BE ANSWERED" in out
    assert "Re-enrolling" in out


# --------------------------------------------------------------- the bar
def test_the_bar_comes_from_faces_and_not_from_the_synthetic_probe(tmp_path):
    """Both models score two unrelated non-face crops near 1.0 at each other.
    Folding that into the ceiling makes every verdict read "no separation"
    even when two real people separate perfectly."""
    _stock(tmp_path)
    payload = compare.collect(Args(tmp_path), Cfg())
    arc = [m for m in payload["models"] if m["model"] == "arcface_mbf"][0]
    bar = arc["bar"]
    assert bar["ok"] is True
    assert bar["basis"] == "another enrolled person"
    assert bar["floor"] > bar["ceiling"]
    assert bar["ceiling"] == pytest.approx(arc["cross_person"]["max"])
    assert bar["suggested"] == pytest.approx(
        (bar["floor"] + bar["ceiling"]) / 2.0)


def test_one_person_alone_cannot_set_a_bar_and_says_so(tmp_path):
    _stock(tmp_path, second_person=False)
    payload = compare.collect(Args(tmp_path), Cfg())
    arc = [m for m in payload["models"] if m["model"] == "arcface_mbf"][0]
    assert arc["bar"]["ok"] is False
    assert "only one person" in arc["bar"]["basis"]
    assert arc["bar"]["floor"] > 0.0          # the floor is still known


def test_the_ood_verdict_is_reported_separately_when_the_probe_ran():
    ood = {"flat colour": compare.spread([0.9, 0.98]),
           "uniform noise": compare.spread([0.8, 0.85]),
           "_embed_ms": compare.spread([4.0])}
    verdict = compare.ood_verdict(ood)
    assert verdict["ok"] and verdict["family"] == "flat colour"
    assert verdict["worst"] == pytest.approx(0.98)
    assert verdict["n"] == 4
    assert compare.ood_verdict({})["ok"] is False


# ------------------------------------------------ the migration, end to end
def test_after_a_swap_it_says_re_enrol_rather_than_nothing_is_enrolled(
        tmp_path, capsys):
    rng = np.random.default_rng(2)
    old = fg.FaceGallery(root=tmp_path, model=fg.SFACE_MODEL)
    for _ in range(13):
        old.add("hunter", rng.normal(size=128).astype(np.float32))
    old.save("his real enrolment")
    payload = compare.collect(Args(tmp_path), Cfg())
    arc = [m for m in payload["models"] if m["model"] == "arcface_mbf"][0]
    assert arc["loaded"] is False
    assert arc["foreign_samples"] == 13
    assert "re-enrol" in arc["reason"]
    compare.report(payload, print)
    out = capsys.readouterr().out
    assert "re-enrol" in out and "13" in out


def test_it_exits_nonzero_when_nothing_is_enrolled_for_either_model(
        tmp_path, capsys):
    code = compare.main(["--gallery", str(tmp_path), "--no-probe"])
    assert code == 1
    out = capsys.readouterr().out
    assert "NOT ENROLLED" in out


def test_json_mode_prints_a_payload_and_nothing_else(tmp_path, capsys):
    _stock(tmp_path)
    code = compare.main(["--gallery", str(tmp_path), "--no-probe", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {m["model"] for m in payload["models"]} == {"sface", "arcface_mbf"}


def test_nothing_it_prints_looks_like_a_path_to_a_picture(tmp_path, capsys):
    _stock(tmp_path)
    compare.report(compare.collect(Args(tmp_path), Cfg()), print)
    out = capsys.readouterr().out
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".mp4", "/dev/video"):
        assert ext not in out
