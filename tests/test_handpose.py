"""The hand models on this box, driven by procedural frames and pure noise.

NO CAMERA IS OPENED, no frame is read from a device, and no frame is
written anywhere. Every image in this file is drawn -- a palm ellipse and
five capsule fingers in a flat skin tone on a dark ground -- or is random
bytes. Hunter's rule is that nothing the lens saw is ever looked at; the
way to test a pixel pipeline under that rule is to feed it pixels that
never came from a lens, and to verify from NUMBERS: row counts, landmark
coordinates, confidences, timings, hashes.

What is pinned, and why each matters:

* **An unusable model RAISES, and the sentence names the file.** The VSS
  ``region_mode='yunet'`` fallback has never once run, and nobody noticed,
  because a silent fallback and a working feature produce identical logs.
  A missing file, a git-lfs pointer, a wrong size, a wrong hash, a missing
  ONNX Runtime: each is a different sentence and none is "no hands".
* **Provenance is on disk and it agrees with the code.** The pinned sha256
  of each weight, the SHA256SUMS the fetch wrote, and the sha256 of the
  LICENSE text the verdict was made on all have to agree -- a licence
  question before a correctness one, on a box that also runs a commercial
  project.
* **The real graphs, on synthetic input.** A drawn open hand is found with
  landmarks of the right shapes in the caller's pixel space and reads OPEN
  through ``jarvis.gesture``; noise, flat grey and black find nothing.
* **Timing at camera.threads = 2** -- the setting the app actually uses,
  not the 4 the bench used -- with the measured figure in the message.
* **No buffer survives a call.** The frame is not retained, no crop, blob
  or downsample is alive after ``detect`` returns, and the source never
  writes a frame or opens a device.
"""
from __future__ import annotations

import gc
import importlib.util
import io
import json
import math
import re
import sys
import time
import tokenize
import weakref
from pathlib import Path

import numpy as np
import pytest

from jarvis import gesture as g
from jarvis import handpose as hp
from jarvis.visionrig import assert_numbers_only

REPO = Path(__file__).resolve().parents[1]
W, H = 1280, 720
EYE_PX = 89.4                     # interocular at 70 cm, capture pixels
FACE_DIR = Path.home() / ".aiws_trainer" / "models" / "face"

needs_models = pytest.mark.skipif(
    not hp.ready(), reason="hand weights are not installed under %s"
    % hp.model_dir())


# ------------------------------------------------------------- fixtures
@pytest.fixture()
def empty_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HAND_MODEL_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def stocked_dir(tmp_path, monkeypatch):
    """Files of the RIGHT SIZE but the wrong content -- enough for the
    shallow check, which is what production uses."""
    monkeypatch.setenv("JARVIS_HAND_MODEL_DIR", str(tmp_path))
    for model in hp.MODELS:
        (tmp_path / model.filename).write_bytes(b"\0" * model.size)
    return tmp_path


class Recorder:
    """Stands in for the ORT session maker: records (path, threads)."""

    def __init__(self):
        self.calls = []

    def __call__(self, path, threads):
        self.calls.append((Path(path).name, int(threads)))
        return object()


@pytest.fixture(scope="module")
def tracker():
    if not hp.ready():
        pytest.skip("hand weights are not installed")
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    return hp.HandTracker(threads=2)


# --------------------------------------------------------- drawn frames
def synth_hand(w=W, h=H, cx=640, cy=360, scale=2.0, seed=0):
    """A hand-like blob: palm ellipse, four fingers up, a thumb out to the
    image left, flat skin tone on a dark ground with a little sensor
    noise. Ported from ~/scratch-gesture/synth.py, where the frames lane
    measured it firing the real palm detector. ``scale`` 2.0 in a
    1280x720 frame is the size that fires cleanly (MEASURED: 1.6-3.0 do,
    0.8-1.2 are too small and 3.5 too big for the crop)."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (40, 45, 55), np.uint8)
    img = (img + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
    skin = (150, 178, 214)
    s = 100.0 * scale
    cv2.ellipse(img, (int(cx), int(cy)), (int(.42 * s), int(.55 * s)),
                0, 0, 360, skin, -1)
    for ang, ln, wd in ((-78, 1.00, .15), (-90, 1.10, .15),
                        (-102, 1.02, .15), (-114, .85, .13),
                        (-160, .70, .17)):
        a = math.radians(ang)
        bx, by = cx + .30 * s * math.cos(a), cy + .30 * s * math.sin(a)
        tx, ty = cx + ln * s * math.cos(a), cy + ln * s * math.sin(a)
        cv2.line(img, (int(bx), int(by)), (int(tx), int(ty)), skin,
                 int(wd * s))
        cv2.circle(img, (int(tx), int(ty)), int(wd * s / 2), skin, -1)
    cv2.ellipse(img, (int(cx), int(cy + .18 * s)),
                (int(.40 * s), int(.34 * s)), 0, 0, 360, skin, -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def drawn_extent(cx, cy, scale):
    """The box the drawing above stays inside: fingers reach 1.1 s up,
    the thumb 0.7 s to the image left, the heel 0.6 s down."""
    s = 100.0 * scale
    return (cx - 1.0 * s, cy - 1.2 * s, cx + 0.6 * s, cy + 0.6 * s)


def synth_noise(seed=0, w=W, h=H):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3),
                                                dtype=np.uint8)


def paste_hand(canvas, cx, cy, scale, seed=1):
    """Draw a second hand onto ``canvas`` (skin pixels only)."""
    extra = synth_hand(canvas.shape[1], canvas.shape[0], cx, cy, scale, seed)
    mask = np.abs(extra.astype(int) - np.array([40, 45, 55])).sum(2) > 40
    out = canvas.copy()
    out[mask] = extra[mask]
    return out


def code_only(path: Path) -> str:
    """Source with comments and string literals blanked, so a docstring or
    a URL string is not a hit. Same helper as tests/test_gesture.py."""
    lines = path.read_text().splitlines(keepends=True)
    tokens = tokenize.generate_tokens(io.StringIO("".join(lines)).readline)
    for tok in tokens:
        name = tokenize.tok_name[tok.type]
        if name not in ("COMMENT", "STRING") and not name.startswith("FSTRING"):
            continue
        (r0, c0), (r1, c1) = tok.start, tok.end
        for r in range(r0, r1 + 1):
            line = lines[r - 1]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(line.rstrip("\r\n"))
            lines[r - 1] = line[:a] + " " * (b - a) + line[b:]
    return "".join(lines)


def arrays_in(obj, seen=None, path="obj"):
    """Every ndarray reachable through Python attributes and containers."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return [(path, obj.shape)]
    out = []
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            out += arrays_in(v, seen, "%s.%s" % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += arrays_in(v, seen, "%s[%d]" % (path, i))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += arrays_in(v, seen, "%s[%r]" % (path, k))
    return out


# ================================================= absence, named, raised
class TestUnusableModelsRaise:
    def test_a_missing_palm_model_names_the_file(self, empty_dir):
        with pytest.raises(hp.HandModelUnavailable) as exc:
            hp.HandTracker(create=Recorder())
        msg = str(exc.value)
        assert hp.PALM_DET.filename in msg and "missing" in msg
        assert "blind" in msg and "not there" in msg

    def test_a_missing_landmark_model_names_that_file(self, empty_dir):
        (empty_dir / hp.PALM_DET.filename).write_bytes(b"\0" * hp.PALM_DET.size)
        with pytest.raises(hp.HandModelUnavailable) as exc:
            hp.HandTracker(create=Recorder())
        assert hp.HANDPOSE.filename in str(exc.value)
        assert hp.PALM_DET.filename not in str(exc.value)

    def test_a_git_lfs_pointer_is_reported_as_a_pointer(self, empty_dir):
        """The plain raw.githubusercontent URL serves ~130 bytes. Reported
        as what it is, not as a corrupt model -- different fix."""
        (empty_dir / hp.PALM_DET.filename).write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n" * 3)
        with pytest.raises(hp.HandModelUnavailable) as exc:
            hp.HandTracker(create=Recorder())
        assert "git-lfs pointer" in str(exc.value)

    def test_a_wrong_size_is_reported_with_both_numbers(self, empty_dir):
        (empty_dir / hp.PALM_DET.filename).write_bytes(
            b"\0" * (hp.PALM_DET.size + 7))
        with pytest.raises(hp.HandModelUnavailable) as exc:
            hp.HandTracker(create=Recorder())
        msg = str(exc.value)
        assert "size" in msg and str(hp.PALM_DET.size) in msg
        assert str(hp.PALM_DET.size + 7) in msg

    def test_a_hash_mismatch_is_a_licence_question(self, stocked_dir):
        """Right size, wrong bytes: the shallow check passes and the deep
        one says this is not the file whose LICENSE was read."""
        assert hp.ready()                         # shallow: sizes agree
        assert not hp.ready(deep=True)
        with pytest.raises(hp.HandModelUnavailable) as exc:
            hp.HandTracker(deep=True, create=Recorder())
        assert "LICENSE" in str(exc.value)
        assert "sha256" in str(exc.value)

    def test_no_onnxruntime_is_a_named_failure(self, stocked_dir,
                                               monkeypatch):
        def boom():
            raise ImportError("No module named 'onnxruntime'")

        monkeypatch.setattr(hp, "_import_ort", boom)
        with pytest.raises(hp.HandModelUnavailable) as exc:
            hp.HandTracker()
        assert "ONNX Runtime" in str(exc.value)
        assert "onnxruntime" in str(exc.value)

    def test_a_maker_that_raises_is_wrapped_and_names_the_dir(self,
                                                             stocked_dir):
        def bad(path, threads):
            raise RuntimeError("protobuf parsing failed")

        with pytest.raises(hp.HandModelUnavailable) as exc:
            hp.HandTracker(create=bad, model_dir=stocked_dir)
        msg = str(exc.value)
        assert "protobuf parsing failed" in msg and str(stocked_dir) in msg

    def test_there_is_nothing_to_fall_back_to(self, stocked_dir, monkeypatch):
        monkeypatch.setattr(hp, "verify",
                            lambda m, deep=False, model_dir=None:
                            (False, "made up reason"))
        with pytest.raises(hp.HandModelUnavailable) as exc:
            hp.HandTracker(create=Recorder())
        assert "made up reason" in str(exc.value)

    def test_available_and_ready_on_an_empty_dir(self, empty_dir):
        got = hp.available()
        assert set(got) == {"palm", "landmarks"}
        assert all(ok is False for ok, _ in got.values())
        assert all("missing" in why for _, why in got.values())
        assert hp.ready() is False

    def test_the_probe_never_raises_and_is_numbers_only(self, empty_dir):
        report = hp.probe()
        assert report["ready"] is False
        assert set(report["models"]) == {"palm", "landmarks", "landmarks_fp32"}
        for key, row in report["models"].items():
            assert row["ok"] is False and row["file"] in row["reason"]
            assert row["licence"] == "Apache-2.0"
            assert row["commercial_ok"] is True
            assert row["licence_present"] is False
            assert row["licence_matches"] is False
        assert report["threads"] == 2 and report["detect_size"] == "320x180"
        assert report["anchors"] == 2016
        assert_numbers_only(report)

    def test_provenance_on_an_empty_dir_is_not_ok(self, empty_dir):
        prov = hp.provenance()
        assert prov["ok"] is False and prov["sums_present"] is False
        assert prov["entries"] == 0
        assert_numbers_only(prov)

    def test_provenance_rejects_sums_that_disagree_with_the_pins(self,
                                                                empty_dir):
        lines = ["%s  %s" % ("0" * 64, m.filename)
                 for m in (hp.PALM_DET, hp.HANDPOSE, hp.HANDPOSE_FP32)]
        (empty_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        for m in (hp.PALM_DET, hp.HANDPOSE):
            (empty_dir / m.licence_file).write_text("not the licence\n")
        prov = hp.provenance()
        assert prov["sums_present"] is True and prov["entries"] == 3
        assert prov["ok"] is False
        for row in prov["models"].values():
            assert row["listed"] is True and row["matches_pin"] is False
        # present is not the same as matching: the wrong text is present
        assert prov["models"]["palm"]["licence_present"] is True
        assert prov["models"]["palm"]["licence_matches"] is False


class TestWhereTheWeightsLive:
    def test_the_override_beats_the_env_beats_home(self, monkeypatch,
                                                   tmp_path):
        monkeypatch.delenv("JARVIS_HAND_MODEL_DIR", raising=False)
        assert hp.model_dir() == Path.home() / ".aiws_trainer" / "models" / "hand"
        monkeypatch.setenv("JARVIS_HAND_MODEL_DIR", str(tmp_path / "env"))
        assert hp.model_dir() == tmp_path / "env"
        assert hp.model_dir(tmp_path / "cfg") == tmp_path / "cfg"
        assert hp.model_path(hp.PALM_DET, tmp_path / "cfg") == \
            tmp_path / "cfg" / hp.PALM_DET.filename

    def test_the_weights_are_not_in_the_repo(self):
        """repo/ already swallowed 140 MB of TTS weights once."""
        for m in (hp.PALM_DET, hp.HANDPOSE, hp.HANDPOSE_FP32):
            assert not list(REPO.rglob(m.filename)), m.filename

    def test_every_model_is_pinned_and_apache(self):
        for m in (hp.PALM_DET, hp.HANDPOSE, hp.HANDPOSE_FP32):
            assert len(m.sha256) == 64 and m.size > 100_000
            assert m.licence == "Apache-2.0"
            assert len(m.licence_sha256) == 64
            assert m.url.startswith("https://media.githubusercontent.com/")
        assert hp.HANDPOSE.filename.endswith("_int8.onnx")
        assert hp.MODELS == (hp.PALM_DET, hp.HANDPOSE)


# ============================================ provenance, on the real dir
@needs_models
class TestProvenanceOnDisk:
    def test_the_weights_on_disk_are_the_pinned_bytes(self):
        for m in (hp.PALM_DET, hp.HANDPOSE, hp.HANDPOSE_FP32):
            ok, why = hp.verify(m, deep=True)
            assert ok, (m.filename, why)

    def test_the_licences_on_disk_are_the_ones_that_were_read(self):
        """Hashes, not names. And the docstring's one-byte claim, by
        diff: the palm text is the handpose text plus a trailing
        newline."""
        root = hp.model_dir()
        palm = (root / hp.PALM_DET.licence_file).read_bytes()
        pose = (root / hp.HANDPOSE.licence_file).read_bytes()
        assert hp.sha256_of(root / hp.PALM_DET.licence_file) == \
            hp.LICENCE_SHA_PALM
        assert hp.sha256_of(root / hp.HANDPOSE.licence_file) == \
            hp.LICENCE_SHA_HANDPOSE
        assert (len(palm), len(pose)) == (11358, 11357)
        assert palm == pose + b"\n"
        for text in (palm, pose):
            assert b"Apache License" in text
            assert b"Version 2.0, January 2004" in text
        assert hp.licence_text_matches(hp.PALM_DET)
        assert hp.licence_text_matches(hp.HANDPOSE)

    def test_the_palm_licence_is_byte_identical_to_the_sface_one(self):
        """jarvis/facemodels.py verified SFace's Apache text against
        apache.org; the palm text pins the same hash, so the verdict is
        inherited by bytes, not by reading twice."""
        sface = FACE_DIR / "LICENSE.sface.Apache-2.0.txt"
        if not sface.exists():
            pytest.skip("SFace licence text not installed")
        assert hp.sha256_of(sface) == hp.LICENCE_SHA_PALM

    def test_provenance_is_ok_and_numbers_only(self):
        prov = hp.provenance()
        assert prov["ok"] is True and prov["sums_present"] is True
        assert prov["entries"] >= 3
        for key, row in prov["models"].items():
            assert row["listed"] and row["matches_pin"], (key, row)
            assert row["licence_present"] and row["licence_matches"], (key, row)
        assert_numbers_only(prov)

    def test_the_probe_on_the_real_dir(self):
        report = hp.probe()
        assert report["ready"] is True
        assert report["provenance"]["ok"] is True
        assert report["ort"]["available"] and report["cv2"]["available"]
        assert "CPUExecutionProvider" in report["ort"]["providers"]
        for row in report["models"].values():
            assert row["ok"] and row["bytes"] == row["expected_bytes"]
            assert row["licence_matches"] is True
        assert_numbers_only(report)


# ============================================================ anchors
class TestAnchors:
    def test_the_anchors_are_the_two_ssd_grids(self):
        a = hp.palm_anchors()
        assert a.shape == (2016, 2) and a.dtype == np.float32
        assert float(a.min()) > 0.0 and float(a.max()) < 1.0
        # 24x24 taken twice, row-major, then 12x12 taken six times
        grid24, grid12 = a[:1152], a[1152:]
        assert len(np.unique(grid24, axis=0)) == 576
        assert len(np.unique(grid12, axis=0)) == 144
        assert np.array_equal(grid24[0], grid24[1])
        assert np.allclose(grid24[0], [1 / 48, 1 / 48])
        assert np.allclose(grid24[2], [3 / 48, 1 / 48])
        assert np.allclose(grid12[0], [1 / 24, 1 / 24])
        assert np.array_equal(grid12[0], grid12[5])
        assert not np.array_equal(grid12[0], grid12[6])

    def test_the_shared_copy_is_cached_and_read_only(self):
        one, two = hp._anchors(), hp._anchors()
        assert one is two and not one.flags.writeable
        assert np.array_equal(one, hp.palm_anchors())


# ================================= the real graphs, on synthetic input
@needs_models
class TestTheRealGraphs:
    def test_a_drawn_open_hand_is_found_in_the_right_shapes(self, tracker):
        rows = tracker.detect(synth_hand())
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, hp.HandRow)
        assert row.lm.shape == (21, 2) and row.lm.dtype == np.float32
        assert row.world.shape == (21, 3) and row.world.dtype == np.float32
        assert np.all(np.isfinite(row.lm)) and np.all(np.isfinite(row.world))
        assert 0.7 <= row.conf <= 1.0
        assert isinstance(row.handed, float)
        assert row.palm_diag > 0.0
        assert row.palm_diag == pytest.approx(
            g.observe_hand(row.lm, 0.0).palm_diag, rel=1e-6)

    def test_the_landmarks_land_on_the_drawn_hand(self, tracker):
        """Every one of the 21 points inside the drawn extent, in the
        frame's own pixel space; the wrist below the palm centroid because
        the fingers were drawn pointing up."""
        cx, cy, scale = 640, 360, 2.0
        row = tracker.detect(synth_hand(cx=cx, cy=cy, scale=scale))[0]
        x0, y0, x1, y1 = drawn_extent(cx, cy, scale)
        assert np.all(row.lm[:, 0] >= x0) and np.all(row.lm[:, 0] <= x1)
        assert np.all(row.lm[:, 1] >= y0) and np.all(row.lm[:, 1] <= y1)
        palm_y = float(row.lm[list(g.PALM), 1].mean())
        assert float(row.lm[g.WRIST, 1]) > palm_y
        assert float(row.lm[8, 1]) < palm_y            # index tip above

    def test_the_drawn_open_hand_reads_open_through_the_real_graphs(
            self, tracker):
        """MEASURED: C = 1.62 at scale 2.0 (1.33-1.72 across 1.6-3.0)
        against the 0.85 open bar. The scalar from jarvis/gesture.py on
        landmarks from the real network agrees with the design's
        synthetic-hand figure of open >= 0.919."""
        for scale in (1.6, 2.0, 2.5):
            rows = tracker.detect(synth_hand(scale=scale))
            assert len(rows) == 1, scale
            o = g.observe_hand(rows[0].lm, EYE_PX)
            assert o.ok and o.closed >= g.CastThresholds().open_min, (scale, o)
            assert o.reach == pytest.approx(rows[0].palm_diag / EYE_PX)

    def test_coordinates_are_in_the_callers_pixel_space(self, tracker):
        frame = synth_hand()
        base = tracker.detect(frame)[0].lm
        moved = tracker.detect(frame, origin=(100.0, 50.0))[0].lm
        assert np.abs((moved - base) - [100.0, 50.0]).max() < 1e-3

    def test_the_downsample_gives_the_same_hand_as_the_full_frame(self):
        """The frames lane's claim, on the real graphs: the palm detector
        letterboxes every 16:9 frame into 192x108, so 320x180 costs no
        answer. MEASURED: palm centroid within 12 px and palm_diag within
        7% of the full-frame run."""
        pytest.importorskip("cv2")
        frame = synth_hand()
        small = hp.HandTracker(threads=2).detect(frame)[0]
        full = hp.HandTracker(threads=2, detect_size=None).detect(frame)[0]
        c_small = small.lm[list(g.PALM)].mean(0)
        c_full = full.lm[list(g.PALM)].mean(0)
        assert float(np.hypot(*(c_small - c_full))) < 0.15 * small.palm_diag
        assert full.palm_diag == pytest.approx(small.palm_diag, rel=0.15)

    def test_two_hands_come_back_largest_first_and_top_k_caps_them(self):
        pytest.importorskip("cv2")
        frame = paste_hand(synth_hand(cx=380, cy=360, scale=2.2), 950, 360, 1.6)
        rows = hp.HandTracker(threads=2).detect(frame)
        assert len(rows) == 2
        assert rows[0].palm_diag > rows[1].palm_diag
        assert float(rows[0].lm[:, 0].mean()) < float(rows[1].lm[:, 0].mean())
        capped = hp.HandTracker(threads=2, top_k=1).detect(frame)
        assert len(capped) == 1
        assert capped[0].palm_diag == pytest.approx(rows[0].palm_diag)

    def test_noise_flat_grey_and_black_find_nothing_and_raise_nothing(
            self, tracker):
        """Zero hands with no exception means the room had no hand in it."""
        for seed in range(3):
            assert tracker.detect(synth_noise(seed)) == ()
        assert tracker.detect(np.full((H, W, 3), 120, np.uint8)) == ()
        assert tracker.detect(np.zeros((H, W, 3), np.uint8)) == ()
        assert tracker.detect(synth_noise(7, 320, 180)) == ()

    def test_a_frame_already_at_detect_size_is_used_as_is(self, tracker):
        frame = synth_hand(320, 180, 160, 90, 0.5)
        small, sx, sy = tracker._small(frame)
        assert small is frame and (sx, sy) == (1.0, 1.0)
        rows = tracker.detect(frame)
        assert len(rows) == 1
        assert np.all(rows[0].lm[:, 0] < 320) and np.all(rows[0].lm[:, 1] < 180)


# ===================================================== threads and time
class TestThreads:
    def test_both_sessions_are_asked_for_two_threads(self, stocked_dir):
        rec = Recorder()
        hp.HandTracker(create=rec)
        assert rec.calls == [(hp.PALM_DET.filename, 2),
                             (hp.HANDPOSE.filename, 2)]
        assert hp.DEFAULT_THREADS == 2

    def test_the_thread_count_is_passed_through(self, stocked_dir):
        rec = Recorder()
        t = hp.HandTracker(threads=4, create=rec, top_k=3)
        assert [n for _p, n in rec.calls] == [4, 4]
        assert t.threads == 4 and t.top_k == 3
        assert t.detect_size == (320, 180)

    @needs_models
    def test_the_ort_sessions_really_run_at_two_intra_threads(self, tracker):
        for stage in (tracker._palm, tracker._pose):
            opts = stage._net.session.get_session_options()
            assert opts.intra_op_num_threads == 2
            assert opts.inter_op_num_threads == 1

    @needs_models
    def test_a_hand_pass_fits_the_frame_budget_at_two_threads(self, tracker):
        """MEASURED 2026-09-03 on this box, 2 threads, 1280x720: p50 9.9 ms
        with a hand (the design's 9.0), 7.2 ms on an empty frame (the
        always-on tax; the design's 7.1). The bars below are 4x those so a
        loaded box does not flake the suite; the message carries the
        measured figure so a regression shows as a number."""
        hand, empty = synth_hand(), synth_noise(0)
        for frame, bar, label in ((hand, 40.0, "hand"), (empty, 30.0, "empty")):
            tracker.detect(frame)                         # warm
            samples = []
            for _ in range(20):
                t0 = time.perf_counter()
                tracker.detect(frame)
                samples.append((time.perf_counter() - t0) * 1000.0)
            samples.sort()
            p50 = samples[len(samples) // 2]
            assert p50 < bar, "%s frame p50 %.1f ms (bar %.0f)" % (
                label, p50, bar)


# ========================================== no buffer survives a call
@needs_models
class TestNoBufferSurvives:
    def test_the_tracker_holds_no_array_before_or_after(self, tracker):
        assert arrays_in(tracker) == []
        tracker.detect(synth_hand())
        assert arrays_in(tracker) == []
        tracker.detect(synth_noise(1))
        assert arrays_in(tracker) == []

    def test_the_frame_is_not_retained(self, tracker):
        frame = synth_hand()
        ref = weakref.ref(frame)
        before = sys.getrefcount(frame)
        tracker.detect(frame)
        assert sys.getrefcount(frame) == before
        del frame
        gc.collect()
        assert ref() is None

    def test_no_intermediate_buffer_outlives_the_call(self, tracker):
        """Not the 320x180 downsample, not the 192x192 or 224x224 blob,
        not the crop, not the rotated crop: after the call and a collect,
        no ndarray exists that did not exist before it."""
        frame = synth_hand()
        gc.collect()
        # type() rather than isinstance(): the latter reads __class__ on
        # every live object, and torch keeps deprecated attribute proxies
        # alive that warn when touched.
        before = {id(o) for o in gc.get_objects() if type(o) is np.ndarray}
        rows = tracker.detect(frame)
        del rows
        gc.collect()
        after = [o.shape for o in gc.get_objects()
                 if type(o) is np.ndarray and id(o) not in before]
        assert after == [], after

    def test_the_rows_carry_landmarks_and_never_pixels(self, tracker):
        row = tracker.detect(synth_hand())[0]
        for name, shape in arrays_in(row, path="row"):
            assert shape in ((21, 2), (21, 3)), (name, shape)


# ============================================================ the source
def test_handpose_writes_no_frame_and_opens_no_device():
    """The grep firewall tests/test_campreview.py runs over the preview
    modules, applied to the one module in the gesture path that holds a
    pixel. Strings are blanked first: the opencv_zoo URL is a string."""
    code = code_only(REPO / "jarvis" / "handpose.py")
    forbidden = (
        r"\bimwrite\b", r"\bimencode\b", r"\bimshow\b", r"\bimread\b",
        r"\bnamedWindow\b", r"\bwaitKey\b", r"\bVideoCapture\b",
        r"/dev/video", r"\.save\s*\(", r"\btofile\b",
        r"\bopen\s*\([^)]*['\"]w", r"\bNamedTemporaryFile\b", r"\bmkstemp\b",
        r"\bwrite_bytes\b", r"\bwrite_text\b", r"\bsocket\b", r"\brequests\b",
        r"\burlopen\b", r"\burllib\b", r"\bbus\.publish\b", r"\bsubprocess\b",
        r"\btkinter\b", r"\bPIL\b",
    )
    for pattern in forbidden:
        assert not re.search(pattern, code), \
            "jarvis/handpose.py: %r is a way a frame could leave" % pattern


def test_handpose_is_the_only_pixel_module_in_the_gesture_path():
    """cv2 and onnxruntime are imported lazily in handpose.py and nowhere
    in gesture.py; the split is the whole reason the state machine can be
    validated with no camera and no weights."""
    pixel = code_only(REPO / "jarvis" / "handpose.py")
    pure = code_only(REPO / "jarvis" / "gesture.py")
    assert re.search(r"^\s+import cv2\b", pixel, re.M)
    assert re.search(r"^\s+import onnxruntime\b", pixel, re.M)
    assert not re.search(r"^import cv2\b|^import onnxruntime\b", pixel, re.M)
    assert "cv2" not in pure and "onnxruntime" not in pure


def test_the_row_scale_is_gestures_scale_and_not_a_second_formula():
    """One definition of palm_diag. handpose asks gesture for it rather
    than restating hypot(|lm0-lm9|, |lm5-lm17|)."""
    code = code_only(REPO / "jarvis" / "handpose.py")
    assert "observe_hand" in code
    assert not re.search(r"hypot\s*\(", code.split("class HandTracker")[1])


# ======================================================= the self-check
def _load_selfcheck():
    spec = importlib.util.spec_from_file_location(
        "gesture_selfcheck", REPO / "scripts" / "gesture_selfcheck.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCfg:
    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)


class TestSelfCheck:
    """scripts/gesture_selfcheck.py: the instrument he runs on his own
    hand. The script's live path needs a camera and is HIS to run; what is
    pinned here runs on drawn frames and never opens a device."""

    def test_the_overlay_reads_through_and_never_writes(self):
        """AssistantConfig.set() saves to disk; the script's --device and
        the suite's overrides must go through a read-only overlay."""
        sc = _load_selfcheck()
        base = FakeCfg({"camera.device": "/dev/video0", "camera.fps": 6.0})
        over = sc.ConfigOverlay(base, {"camera.device": "/dev/video2"})
        assert over.get("camera.device") == "/dev/video2"
        assert over.get("camera.fps") == 6.0
        assert over.get("camera.absent", "dflt") == "dflt"
        assert base.data["camera.device"] == "/dev/video0"
        assert not hasattr(over, "set") and not hasattr(over, "save")
        # No config write anywhere in the script: cfg.set / cfg.update
        # save his assistant.json, and nothing here may call .save().
        code = code_only(REPO / "scripts" / "gesture_selfcheck.py")
        assert not re.search(r"\bcfg\.(set|update|save)\s*\(", code)
        assert not re.search(r"\.save\s*\(", code)
        assert not re.search(r"\.data\b", code)

    def test_thresholds_from_config_overrides_key_by_key(self):
        sc = _load_selfcheck()
        assert sc.thresholds_from_config(FakeCfg({})) == g.CastThresholds()
        t = sc.thresholds_from_config(FakeCfg({
            "gesture.reach_min": 2.5, "gesture.dwell_frames": 4,
            "gesture.target_sectors": ["left"],
            "gesture.carry_max_s": 6}))
        assert t.reach_min == 2.5 and t.dwell_frames == 4
        assert t.target_sectors == ("left",) and t.carry_max_s == 6.0
        assert isinstance(t.dwell_frames, int)
        assert t.open_min == g.CastThresholds().open_min

    def test_the_synthetic_source_keeps_the_rig_contract(self):
        pytest.importorskip("cv2")
        sc = _load_selfcheck()
        src = sc.SyntheticHandSource(640, 360, frames=2)
        ok, frame = src.read()
        assert ok and frame.shape == (360, 640, 3) and frame.dtype == np.uint8
        assert src.read()[0] is True
        assert src.read() == (False, None)
        src.release()
        assert src.released == 1

    def test_closed_3d_is_a_ratio_and_zero_on_the_wrong_shape(self):
        sc = _load_selfcheck()
        assert sc.closed_3d(np.zeros((20, 3))) == 0.0
        assert sc.closed_3d(np.zeros((21, 3))) == 0.0
        w = np.zeros((21, 3))
        w[g.MID_MCP] = (0, 0.10, 0)
        w[g.IDX_MCP], w[g.PNK_MCP] = (-0.03, 0.09, 0), (0.04, 0.09, 0)
        for tip in g.TIPS:
            w[tip] = (0, 0.18, 0)
        assert sc.closed_3d(w) == pytest.approx(0.18 / math.hypot(0.10, 0.07))

    def test_models_only_reaches_a_verdict_without_a_device(self, capsys):
        sc = _load_selfcheck()
        code = sc.main(["--models-only", "--json"])
        report = json.loads(capsys.readouterr().out)
        assert_numbers_only(report)
        assert report["exit_code"] == code
        assert set(report["hand_models"]["models"]) == {
            "palm", "landmarks", "landmarks_fp32"}
        assert "run" not in report
        assert code == (0 if hp.ready() else 1)

    @needs_models
    def test_the_synthetic_run_is_numbers_only_and_honest(self, capsys):
        """A drawn hand: found on every frame, R 0.0 (no face), no grab,
        and every claim that needs his hand marked not measurable."""
        pytest.importorskip("cv2")
        sc = _load_selfcheck()
        code = sc.main(["--synthetic", "--frames", "3", "--json"])
        report = json.loads(capsys.readouterr().out)
        assert_numbers_only(report)
        assert code == 0 and report["exit_code"] == 0
        run = report["run"]
        assert run["frames"] == 3 and run["hand_frames"] == 3
        assert run["R"]["max"] == 0.0 and run["face_frames"] == 0
        assert run["C"]["min"] >= g.CastThresholds().open_min
        assert run["grabs"] == 0 and run["throws"] == 0
        assert run["states"] == {"idle": 3}
        checks = {c["name"]: c for c in report["checks"]}
        assert checks["hand seen"]["ok"] and checks["hand seen"]["measurable"]
        for name in ("reached", "grab fired", "throw fired", "face seen"):
            assert not checks[name]["measurable"], name
        assert report["thresholds"]["reach_min"] == 2.35
        assert report["fps_mismatch"] is False

    @needs_models
    def test_the_counters_are_scaled_by_the_configured_rate(self, capsys,
                                                            monkeypatch):
        """His live config asks for preview_fps 15 (read 2026-09-03,
        default 6.0). At 15 the design's 3-frame dwell would be 200 ms,
        so the self-check applies for_fps exactly as the app must, and
        says so. The LifeCam delivers ~7.5 whatever is asked, which is why
        the script also compares delivered to configured after the run."""
        pytest.importorskip("cv2")
        sc = _load_selfcheck()
        real_load = sc.AssistantConfig.load

        def load(*a, **k):
            # An overlay, never AssistantConfig.set(): set() SAVES.
            return sc.ConfigOverlay(real_load(*a, **k),
                                    {"camera.preview_fps": 15.0})

        monkeypatch.setattr(sc.AssistantConfig, "load", staticmethod(load))
        assert sc.main(["--synthetic", "--frames", "2", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["capture_fps"] == 15.0
        assert report["counters_scaled"] is True
        assert report["thresholds"]["dwell_frames"] == 6
        assert report["thresholds"]["exit_step_u"] == pytest.approx(0.175)
        assert report["thresholds"]["reach_min"] == 2.35

    def test_the_selfcheck_writes_no_frame_and_opens_no_device_itself(self):
        """The live path reaches the camera only through jarvis/camera's
        gated feed (cam.build / CameraFeed), never a VideoCapture of its
        own, and nothing in it can write or show a frame."""
        code = code_only(REPO / "scripts" / "gesture_selfcheck.py")
        for pattern in (r"\bimwrite\b", r"\bimencode\b", r"\bimshow\b",
                        r"\bimread\b", r"\bnamedWindow\b", r"\bwaitKey\b",
                        r"\bVideoCapture\b", r"\.save\s*\(", r"\btofile\b",
                        r"\bopen\s*\([^)]*['\"]w", r"\bNamedTemporaryFile\b",
                        r"\bmkstemp\b", r"\bwrite_bytes\b", r"\bwrite_text\b",
                        r"\bsocket\b", r"\brequests\b", r"\burlopen\b",
                        r"\bsubprocess\b", r"\btkinter\b", r"\bPIL\b"):
            assert not re.search(pattern, code), pattern
        assert "cam.FeedSource" in code and "cam.build" in code
        assert "assert_numbers_only" in code
