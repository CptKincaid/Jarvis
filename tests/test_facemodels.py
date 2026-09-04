"""Face model registry and lens geometry.

No camera, no display, no GPU, no network, no cv2. The geometry tests are
closed-form arithmetic; the registry tests build fake files in tmp_path via
JARVIS_FACE_MODEL_DIR and never touch the real weights.
"""
from __future__ import annotations

import math

import pytest

from jarvis import facemodels as fm


# ------------------------------------------------------------------ licences
def test_the_opencv_pair_is_permissive_and_commercial_safe():
    """VSS ships commercially from this machine. This was once a check that
    NO research-licensed weight existed here; two now do, so the check is
    narrower and the flag that replaces it is ``commercial_ok``."""
    assert fm.YUNET.licence == "MIT"
    assert fm.SFACE.licence == "Apache-2.0"
    assert fm.YUNET.commercial_ok and fm.SFACE.commercial_ok
    assert fm.OPENCV.commercial_ok is True
    for m in fm.OPENCV.models():
        assert m.url.startswith("https://media.githubusercontent.com/media/"
                                "opencv/opencv_zoo/"), m.url
        assert "insightface" not in m.url.lower()


def test_the_insightface_pair_is_marked_non_commercial_everywhere():
    """The restriction has to be impossible to miss by somebody reusing this
    module later, so it is on the models, on the backend, and in a word that
    reads as a refusal rather than as a licence name."""
    for m in (fm.SCRFD_500M, fm.ARCFACE_MBF):
        assert m.licence == "NON-COMMERCIAL RESEARCH ONLY"
        assert m.commercial_ok is False
        assert "model_zoo" in m.licence_source
    assert fm.INSIGHT.commercial_ok is False
    assert "NON-COMMERCIAL" in fm.INSIGHT.note
    assert fm.commercial_backends() == ("opencv",)


def test_the_licence_terms_are_recorded_where_a_reader_will_hit_them():
    """The verification habit in this module is to read the model's own
    LICENSE. InsightFace has none -- the sentence in model_zoo/README.md is
    the only statement of terms -- and the docstring has to say so, or the
    next reader assumes the usual answer."""
    doc = fm.__doc__ or ""
    assert "NON-COMMERCIAL RESEARCH ONLY" in doc
    assert "LICENSE file at all" in doc
    assert "model_zoo/README.md" in doc


def test_every_declared_model_is_pinned_by_hash_and_size():
    for m in fm.ALL_MODELS:
        assert len(m.sha256) == 64
        assert m.size > 1024


# ------------------------------------------------------------------ registry
@pytest.fixture()
def fake_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_FACE_MODEL_DIR", str(tmp_path))
    return tmp_path


def test_missing_model_reports_the_path(fake_dir):
    ok, why = fm.verify(fm.YUNET)
    assert not ok and "missing" in why and fm.YUNET.filename in why
    assert not fm.ready()


def test_lfs_pointer_is_named_as_such(fake_dir):
    """The file the zoo serves over plain raw.githubusercontent is a ~130 byte
    pointer, not the model. It must not be mistaken for a truncated download."""
    (fake_dir / fm.YUNET.filename).write_bytes(b"version https://git-lfs" * 5)
    ok, why = fm.verify(fm.YUNET)
    assert not ok and "git-lfs pointer" in why


def test_wrong_size_is_rejected(fake_dir):
    (fake_dir / fm.SFACE.filename).write_bytes(b"\0" * 4096)
    ok, why = fm.verify(fm.SFACE)
    assert not ok and "size" in why


def test_correct_size_passes_shallow_but_hash_still_checked(fake_dir):
    p = fake_dir / fm.YUNET.filename
    p.write_bytes(b"\7" * fm.YUNET.size)
    assert fm.verify(fm.YUNET)[0] is True          # size-only path
    ok, why = fm.verify(fm.YUNET, deep=True)       # content is wrong
    assert not ok and "sha256" in why


def test_available_covers_the_active_backends_pair_by_role(fake_dir):
    got = fm.available()
    assert set(got) == {"detector", "recogniser"}
    assert set(fm.available(backend="opencv")) == {"detector", "recogniser"}


# ------------------------------------------------------------------ backends
def test_the_default_backend_is_the_swap_and_it_is_reversible():
    """He asked for the swap; the OpenCV pair stays fully declared so one
    config key puts it back."""
    assert fm.DEFAULT_BACKEND == "insightface"
    assert set(fm.BACKENDS) == {"opencv", "insightface"}
    assert fm.backend_for().name == "insightface"
    assert fm.backend_for("opencv").models() == (fm.YUNET, fm.SFACE)
    assert fm.backend_for("insightface").models() == (fm.SCRFD_500M,
                                                      fm.ARCFACE_MBF)


def test_an_unknown_backend_raises_rather_than_falling_back():
    """A typo that silently left the old models running is the exact class of
    bug this subsystem is written against."""
    with pytest.raises(ValueError, match="unknown face backend"):
        fm.backend_for("arcface")


def test_the_environment_can_pick_the_backend(monkeypatch):
    monkeypatch.setenv(fm.BACKEND_ENV, "opencv")
    assert fm.backend_for().name == "opencv"
    assert fm.backend_for("insightface").name == "insightface"   # arg wins


def test_the_two_backends_disagree_about_dimension_and_bar():
    assert fm.OPENCV.embed_dim == 128 and fm.INSIGHT.embed_dim == 512
    assert fm.OPENCV.embed_model == "sface"
    assert fm.INSIGHT.embed_model == "arcface_mbf"
    assert fm.OPENCV.cosine_same == pytest.approx(0.363)
    # UNMEASURED, not zero and not borrowed from SFace.
    assert fm.INSIGHT.cosine_same is None


def test_readiness_is_about_the_active_pair_not_all_four(fake_dir):
    for model in fm.backend_for("opencv").models():
        (fake_dir / model.filename).write_bytes(b"\0" * model.size)
    assert fm.ready(backend="opencv") is True
    assert fm.ready(backend="insightface") is False


# ------------------------------------------------------------------ geometry
def test_diagonal_to_horizontal_uses_tangents_not_ratios():
    """The LifeCam Cinema's 73 deg is a DIAGONAL. Treating it as horizontal,
    or scaling it linearly by 16/18.36, both give the wrong answer."""
    h = fm.horizontal_fov_deg(73.0)
    assert h == pytest.approx(65.64, abs=0.02)
    assert h != pytest.approx(73.0, abs=1.0)              # not the diagonal
    assert h != pytest.approx(73.0 * 16 / math.hypot(16, 9), abs=0.5)  # not
    #                                                       the linear ratio


def test_98_degree_diagonal_is_the_configs_90_horizontal():
    """docs/vision.md section 9 recommends a 98 deg diagonal camera and then
    reasons from '~90 deg horizontally'. That step is correct -- it is only
    wrong when applied to a camera he actually owns."""
    assert fm.horizontal_fov_deg(98.0) == pytest.approx(90.1, abs=0.1)


def test_lens_requires_an_explicit_fov():
    """No default. Assuming 90 is the bug this class exists to prevent."""
    with pytest.raises(TypeError):
        fm.Lens(1280, 720)          # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [0.0, -5.0, 180.0, 200.0])
def test_absurd_fov_rejected(bad):
    with pytest.raises(ValueError):
        fm.Lens(1280, 720, bad)


def test_lifecam_is_720p_and_so_is_the_config_now():
    """camera.width/height shipped as 1920x1080 until 2026-09-02, which is a
    mode the LifeCam does not have; the config now says 1280x720 and
    tests/test_eye.py pins it."""
    assert (fm.LIFECAM_CINEMA.width_px, fm.LIFECAM_CINEMA.height_px) == \
        (1280, 720)
    assert fm.LIFECAM_CINEMA.hfov_deg == pytest.approx(65.64, abs=0.02)


def test_face_pixels_at_the_desk():
    """A 16 cm face at 95 cm. The narrower lens and the lower resolution very
    nearly cancel, which is why the full-frame number looks fine on both."""
    assert fm.LIFECAM_CINEMA.face_px(95.0) == pytest.approx(167, abs=1)
    assert fm.ARDUCAM_IMX462.face_px(95.0) == pytest.approx(162, abs=1)


def test_the_downscale_is_where_the_cameras_diverge():
    """They cancel at full frame and do NOT after the resize to detect width,
    because the two cameras are downscaled by different factors (4x vs 6x)."""
    lifecam = fm.LIFECAM_CINEMA.detect_face_px(95.0, 320)
    arducam = fm.ARDUCAM_IMX462.detect_face_px(95.0, 320)
    assert lifecam == pytest.approx(41.8, abs=0.5)
    assert arducam == pytest.approx(26.9, abs=0.5)
    assert lifecam > arducam * 1.5
    # YuNet's documented working range is ~10x10 to 300x300; both clear it.
    assert 10 < arducam < 300


def test_offset_deg_is_rectilinear_not_linear():
    """At the edge of a wide frame the linear deg-per-pixel approximation is
    materially wrong, which matters for a 20 deg attention cone."""
    lens = fm.ARDUCAM_IMX462                     # 90.1 deg horizontal
    edge = lens.offset_deg(lens.width_px / 2.0)
    assert edge == pytest.approx(lens.hfov_deg / 2.0, abs=0.01)
    # Quarter width: the tangent map gives 26.6 deg, the linear
    # deg-per-pixel approximation 22.5 -- a 4.1 deg error, which is a fifth
    # of the whole 20 deg attention cone.
    quarter = lens.offset_deg(lens.width_px / 4.0)
    linear = lens.hfov_deg / 2.0 * 0.5
    assert quarter == pytest.approx(26.63, abs=0.05)
    assert linear == pytest.approx(22.53, abs=0.05)
    assert quarter - linear > 4.0


def test_offset_deg_is_signed_and_centred():
    lens = fm.LIFECAM_CINEMA
    assert lens.offset_deg(0.0) == 0.0
    assert lens.offset_deg(-100.0) == pytest.approx(-lens.offset_deg(100.0))


def test_a_narrower_lens_maps_the_same_pixel_to_a_smaller_angle():
    """The whole reason FOV cannot be assumed: identical pixel coordinates
    mean different angles on different cameras."""
    px = 300.0
    narrow = fm.Lens(1280, 720, 65.64).offset_deg(px)
    wide = fm.Lens(1280, 720, 90.0).offset_deg(px)
    assert narrow < wide
    assert wide / narrow > 1.3


def test_describe_mentions_both_scales(monkeypatch):
    line = fm.describe(fm.LIFECAM_CINEMA)
    assert "1280x720" in line and "65.6" in line and "167px" in line
    assert "42px" in line
