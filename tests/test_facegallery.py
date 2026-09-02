"""The face gallery (jarvis/facegallery.py): the ONE artefact on disk that is
biometric data about a specific person.

Every test here exists because of a real incident. On 2026-09-02 a test built
a real SpeakerVerifier and called ``enroll_from_audio``; ``save()`` replaced
his six-sample voiceprint with two copies of the fixture's constant vector,
the copy kept beside it held only those fixtures, and the original was
UNRECOVERABLE -- he had to re-enrol (tests/conftest.py, the JARVIS_VOICEPRINT
paragraph; jarvis/config.py:46-55). ``jarvis/speaker.py:276`` writes
atomically, which protects against a crash mid-write and against nothing
else: a successful bad write is still a total loss.

So this store is generational. A save never overwrites; it writes the NEXT
generation and the previous ones stay on disk, which is what makes a bad
write recoverable instead of merely tidy. The three guards below (degenerate
vectors, a collapsed pool, a silent shrink) each refuse the specific shape
that incident had.

Nothing here touches the user's real gallery: PATHS.FACE_GALLERY reads
JARVIS_FACE_GALLERY, conftest points it at the throwaway dir, and every test
additionally passes its own tmp_path root.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import facegallery as fg
from jarvis.facegallery import FaceGallery, cosine, degenerate_reason


def vec(seed: int, dim: int = fg.EMBED_DIM) -> np.ndarray:
    """A plausible SFace embedding: 128-D float32, NOT unit norm (measured
    2026-09-02 on this box -- cv2.FaceRecognizerSF.feature returns L2=10.41)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(dim).astype(np.float32) * 0.9)


def near(base: np.ndarray, seed: int, k: float = 0.25) -> np.ndarray:
    """The same face again: base plus a little noise, so it matches."""
    rng = np.random.default_rng(seed)
    return (base + rng.standard_normal(base.shape).astype(np.float32) * k).astype("float32")


# ------------------------------------------------------------------ maths
def test_cosine_is_scale_free_because_sface_does_not_normalise():
    a = vec(1)
    assert cosine(a, a * 7.0) == pytest.approx(1.0, abs=1e-5)
    assert cosine(a, -a) == pytest.approx(-1.0, abs=1e-5)


def test_cosine_of_a_zero_vector_is_zero_not_nan():
    """A zero vector is what a failed crop produces; NaN would poison every
    comparison downstream and silently match everything."""
    assert cosine(vec(1), np.zeros(fg.EMBED_DIM, dtype=np.float32)) == 0.0


# ------------------------------------------------------- the three guards
def test_a_constant_vector_is_refused_this_is_the_2026_09_02_shape():
    """The fixture that destroyed the voiceprint was a constant vector --
    every element 0.07216878 (jarvis/config.py:50-52). A real embedding never
    is, so the store can refuse that exact shape at the door."""
    bad = np.full(fg.EMBED_DIM, 0.07216878, dtype=np.float32)
    assert "constant" in degenerate_reason(bad)
    g = FaceGallery(root=None)
    with pytest.raises(ValueError, match="constant"):
        g.add("hunter", bad)


def test_wrong_dimension_and_non_finite_are_refused():
    assert "dimension" in degenerate_reason(np.ones(64, dtype=np.float32) * np.arange(64))
    nan = vec(2).copy()
    nan[3] = np.nan
    assert "finite" in degenerate_reason(nan)


def test_a_collapsed_pool_cannot_be_saved(tmp_path):
    """Two copies of one vector is what the incident left behind. A pool whose
    members are all the same face-to-4-decimals is not an enrolment."""
    g = FaceGallery(root=tmp_path)
    base = vec(3)
    g.add("hunter", base)
    g.add("hunter", base + np.float32(1e-6))
    with pytest.raises(ValueError, match="collapsed"):
        g.save(reason="enrol")
    assert g.generations() == []          # nothing reached the disk


# ------------------------------------------------------- generational save
def test_saves_are_generational_so_a_bad_write_is_recoverable(tmp_path):
    g = FaceGallery(root=tmp_path)
    base = vec(4)
    for i in range(6):
        g.add("hunter", near(base, 40 + i))
    assert g.save(reason="enrol") == 1

    # A second, honest enrolment.
    g2 = FaceGallery(root=tmp_path)
    assert g2.load() is True
    assert g2.count("hunter") == 6
    for i in range(6):
        g2.add("hunter", near(base, 60 + i))
    assert g2.save(reason="re-enrol") == 2
    assert g2.generations() == [1, 2]

    # Generation 1 is still readable, byte for byte, after generation 2.
    old = FaceGallery(root=tmp_path)
    assert old.load(generation=1) is True
    assert old.count("hunter") == 6
    assert cosine(old.embeddings("hunter")[0], g.embeddings("hunter")[0]) == \
        pytest.approx(1.0, abs=1e-6)


def test_rollback_recovers_the_previous_generation(tmp_path):
    """The step that did not exist on 2026-09-02: undo the last write."""
    g = FaceGallery(root=tmp_path)
    good = vec(5)
    for i in range(4):
        g.add("hunter", near(good, 70 + i))
    g.save(reason="enrol")
    first = g.embeddings("hunter")[0].copy()

    g.reset()
    other = vec(999)                      # a different face entirely
    for i in range(4):
        g.add("hunter", near(other, 80 + i))
    g.save(reason="oops")
    assert g.generations() == [1, 2]

    g.rollback()
    assert g.generations() == [1]
    assert cosine(g.embeddings("hunter")[0], first) == pytest.approx(1.0, abs=1e-6)


def test_a_shrinking_save_needs_saying_so_out_loud(tmp_path):
    """The incident replaced six samples with two and nothing objected."""
    g = FaceGallery(root=tmp_path)
    base = vec(6)
    for i in range(6):
        g.add("hunter", near(base, 90 + i))
    g.save(reason="enrol")

    g.reset()
    for i in range(2):
        g.add("hunter", near(base, 110 + i))
    with pytest.raises(ValueError, match="shrink"):
        g.save(reason="fixture")
    assert g.generations() == [1]
    assert g.save(reason="deliberate trim", allow_shrink=True) == 2


def test_old_generations_are_pruned_but_never_below_two(tmp_path):
    g = FaceGallery(root=tmp_path)
    base = vec(7)
    for n in range(fg.KEEP_GENERATIONS + 3):
        g.reset()
        for i in range(4):
            g.add("hunter", near(base, 200 + 10 * n + i))
        g.save(reason="enrol %d" % n)
    gens = g.generations()
    assert len(gens) == fg.KEEP_GENERATIONS
    assert gens[-1] == fg.KEEP_GENERATIONS + 3      # the newest survives
    assert len(gens) >= 2                           # a rollback target always exists


def test_a_corrupt_newest_generation_falls_back_to_the_one_before(tmp_path):
    g = FaceGallery(root=tmp_path)
    base = vec(8)
    for i in range(4):
        g.add("hunter", near(base, 300 + i))
    g.save(reason="enrol")
    kept = g.embeddings("hunter")[0].copy()
    g.reset()
    for i in range(4):
        g.add("hunter", near(base, 310 + i))
    g.save(reason="enrol 2")

    path = g.path_for(2)
    path.write_bytes(b"not an npz at all")

    fresh = FaceGallery(root=tmp_path)
    assert fresh.load() is True
    assert fresh.loaded_generation == 1
    assert cosine(fresh.embeddings("hunter")[0], kept) == pytest.approx(1.0, abs=1e-6)


# -------------------------------------------------------------- on disk
def test_the_gallery_is_private_to_him_on_disk(tmp_path):
    root = tmp_path / "gallery"
    g = FaceGallery(root=root)
    base = vec(9)
    for i in range(4):
        g.add("hunter", near(base, 400 + i))
    g.save(reason="enrol")
    assert (root.stat().st_mode & 0o777) == 0o700
    assert (g.path_for(1).stat().st_mode & 0o777) == 0o600


def test_provenance_records_what_wrote_it_and_when(tmp_path):
    g = FaceGallery(root=tmp_path)
    base = vec(10)
    for i in range(4):
        g.add("hunter", near(base, 500 + i))
    g.save(reason="scripts/enroll_face.py --reset")
    p = FaceGallery(root=tmp_path)
    p.load()
    prov = p.provenance()
    assert prov["reason"] == "scripts/enroll_face.py --reset"
    assert prov["n"] == 4 and prov["format"] == fg.FORMAT
    assert prov["created_ns"] > 0


def test_a_future_format_is_refused_rather_than_misread(tmp_path):
    """Reading a newer pool as if it were this one is how an embedding set
    silently stops comparing (jarvis/speaker.py:264-271 warns about the same
    thing for the voiceprint)."""
    g = FaceGallery(root=tmp_path)
    base = vec(11)
    for i in range(4):
        g.add("hunter", near(base, 600 + i))
    g.save(reason="enrol")
    data = dict(np.load(g.path_for(1)))
    data["_format"] = np.array([fg.FORMAT + 1])
    with open(g.path_for(1), "wb") as fh:
        np.savez(fh, **data)
    fresh = FaceGallery(root=tmp_path)
    assert fresh.load() is False
    assert fresh.count("hunter") == 0


# -------------------------------------------------------------- deletion
def test_purge_actually_deletes_every_generation(tmp_path):
    """'How is it deleted' must have an answer that leaves nothing behind --
    a gallery with one surviving old generation is not deleted."""
    g = FaceGallery(root=tmp_path)
    base = vec(12)
    for n in range(3):
        g.reset()
        for i in range(4):
            g.add("hunter", near(base, 700 + 10 * n + i))
        g.save(reason="enrol")
    assert g.purge() == 3
    assert g.generations() == []
    assert list(tmp_path.glob("*.npz")) == []
    assert g.count("hunter") == 0


def test_forget_one_person_leaves_the_others(tmp_path):
    g = FaceGallery(root=tmp_path)
    a, b = vec(13), vec(14)
    for i in range(4):
        g.add("hunter", near(a, 800 + i))
        g.add("guest", near(b, 810 + i))
    g.save(reason="enrol")
    g.forget("guest")
    assert g.save(reason="forget guest", allow_shrink=True) == 2
    fresh = FaceGallery(root=tmp_path)
    fresh.load()
    assert fresh.labels() == ("hunter",)


# -------------------------------------------------------------- matching
def test_match_returns_the_nearest_label_and_its_score(tmp_path):
    g = FaceGallery(root=tmp_path)
    a, b = vec(15), vec(16)
    for i in range(5):
        g.add("hunter", near(a, 900 + i))
        g.add("guest", near(b, 910 + i))
    label, score = g.match(near(a, 999))
    assert label == "hunter" and score > 0.5
    label, score = g.match(vec(1234))
    assert score < 0.5


def test_an_empty_gallery_matches_nothing_rather_than_guessing():
    g = FaceGallery(root=None)
    assert g.match(vec(17)) == ("", 0.0)
