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


# ------------------------------------------ the holes an adversarial read found
# Five defects, all reproduced against the first version of this file before
# being fixed. Each one let the 2026-09-02 shape recur through a door the
# module docstring claimed was shut.
def test_the_shrink_guard_covers_a_gallery_THAT_NEVER_LOADED(tmp_path):
    """The incident's own shape, which the first guard missed.

    ``enroll_from_audio`` built a fresh SpeakerVerifier and saved; it never
    loaded. A shrink baseline taken only from ``self._loaded_n`` is therefore 0
    on precisely the caller that caused the loss, and measured 2026-09-02 a
    fresh FaceGallery wrote a 2-sample generation over a 6-sample one without
    a word. The baseline has to come off the disk."""
    root = tmp_path / "g"
    first = FaceGallery(root=root)
    base = vec(1)
    for i in range(6):
        first.add("hunter", near(base, 10 + i))
    assert first.save("real enrolment, 6 takes") == 1

    stray = FaceGallery(root=root)          # fresh object: loaded nothing
    stray.add("hunter", vec(20))
    stray.add("hunter", vec(21))
    with pytest.raises(ValueError, match="refusing to shrink"):
        stray.save("a stray test")
    assert FaceGallery(root=root).generations() == [1]

    # and it is still a guard, not a wall: saying so out loud still works
    assert stray.save("deliberate re-enrolment", allow_shrink=True) == 2


def test_repetition_cannot_evict_the_best_enrolment(tmp_path):
    """The generations are the backup only if repetition cannot delete them.

    KEEP_GENERATIONS is 5 and the first ``_prune`` deleted oldest-first, so
    five saves of any size at all evicted a six-take enrolment and left the
    store holding nothing but the bad writes -- the state the incident left,
    reached again five writes later. Measured 2026-09-02: [4,5,6,7,8], every
    one n=2."""
    root = tmp_path / "g"
    good = FaceGallery(root=root)
    base = vec(2)
    for i in range(6):
        good.add("hunter", near(base, 30 + i))
    good.save("real enrolment, 6 takes")

    for run in range(8):                    # well past the five-deep window
        k = FaceGallery(root=root)
        k.add("hunter", vec(40 + run))
        k.add("hunter", vec(60 + run))
        k.save("runaway %d" % run, allow_shrink=True)

    kept = FaceGallery(root=root)
    gens = kept.generations()
    assert len(gens) == fg.KEEP_GENERATIONS      # the window is still a window
    assert 1 in gens, "the six-take enrolment was pruned away: %r" % (gens,)
    assert kept.load(1) and kept.total() == 6


def test_a_tie_prunes_exactly_as_before_so_the_window_still_moves(tmp_path):
    """Protecting the richest must not freeze the window in the normal case.

    When every save is the same size the richest IS the newest, so the oldest
    still falls out -- otherwise the store would grow without bound."""
    root = tmp_path / "g"
    for run in range(9):
        k = FaceGallery(root=root)
        for i in range(3):
            k.add("hunter", vec(100 + run * 10 + i))
        k.save("even save %d" % run)
    gens = FaceGallery(root=root).generations()
    assert gens == [5, 6, 7, 8, 9]


def test_the_embedding_file_is_never_briefly_world_readable(tmp_path):
    """0600 from creation, not 0600 after np.savez returns.

    His umask is 0002, so ``open(tmp, "wb")`` creates the file 0664 and it
    stays 0664 for the whole write. This is the one module whose stated job is
    a stricter standard for data that cannot be re-issued, so the window is
    the bug, not just the end state."""
    import os

    root = tmp_path / "g"
    g = FaceGallery(root=root)
    base = vec(3)
    for i in range(3):
        g.add("hunter", near(base, 70 + i))

    seen = {}
    real_open = os.open

    def spy(path, *a, **kw):
        fd = real_open(path, *a, **kw)
        if str(path).endswith(".tmp"):
            seen["tmp"] = os.stat(path).st_mode & 0o777
            seen["dir"] = root.stat().st_mode & 0o777
        return fd

    os.open = spy
    try:
        g.save("first")
    finally:
        os.open = real_open

    assert seen["tmp"] == 0o600, "tmp was %s mid-write" % oct(seen.get("tmp", 0))
    assert seen["dir"] == 0o700, "dir was %s" % oct(seen.get("dir", 0))
    assert g.path_for(1).stat().st_mode & 0o777 == 0o600


def test_a_crashed_save_leaves_no_embeddings_behind(tmp_path):
    """``gen-00002.npz.tmp`` holds a full set of face embeddings and does not
    match the generation pattern, so the first ``purge()`` reported success and
    left one on disk at 0600 -- deleted, in the sense that nothing looked."""
    import os

    root = tmp_path / "g"
    g = FaceGallery(root=root)
    base = vec(4)
    for i in range(4):
        g.add("hunter", near(base, 80 + i))
    g.save("first")
    g.add("hunter", near(base, 90))

    real_replace = os.replace
    os.replace = lambda a, b: (_ for _ in ()).throw(OSError("disk full"))
    try:
        with pytest.raises(OSError):
            g.save("second")
    finally:
        os.replace = real_replace

    # the failed save cleaned up after itself
    assert sorted(p.name for p in root.iterdir()) == ["gen-00001.npz"]

    # and belt and braces: a tmp that somehow survives is still purged
    stray = root / "gen-00007.npz.tmp"
    stray.write_bytes(b"pretend embeddings")
    assert g.purge() == 2
    assert list(root.iterdir()) == []


def test_rollback_refuses_when_the_only_readable_generation_is_the_newest(tmp_path):
    """gen 1 unreadable (a 0-byte truncation), gen 2 the only good
    enrolment: rollback() used to shred gen 2 on a file COUNT, fail the
    load, and leave nothing on disk that parses (F13, reproduced). Now it
    proves an older generation parses BEFORE it destroys anything."""
    g = FaceGallery(root=tmp_path / "g")
    base = vec(5)
    for i in range(6):
        g.add("hunter", near(base, 300 + i))
    g.save("gen1")
    g.add("hunter", near(base, 310))
    g.save("gen2")
    assert g.generations() == [1, 2]
    g.path_for(1).write_bytes(b"")          # the shape backup() warns about
    assert g.rollback() == 0
    assert g.generations() == [1, 2], "it destroyed the only good enrolment"
    back = FaceGallery(root=g.root)
    assert back.load() is True and back.loaded_generation == 2


def test_rollback_reports_the_generation_it_is_actually_holding(tmp_path):
    """``load()`` falls back down the stack when the predecessor is also
    corrupt, but the return value was hard-coded to ``gens[-2]``. Measured
    2026-09-02: rollback() returned 2 while loaded_generation was 1. An
    enrolment script printing "rolled back to generation 2" while holding
    generation 1 is exactly the quiet mismatch this module exists to prevent."""
    root = tmp_path / "g"
    g = FaceGallery(root=root)
    base = vec(5)
    for i in range(6):
        g.add("hunter", near(base, 200 + i))
    g.save("gen1")
    g.add("hunter", near(base, 210))
    g.save("gen2")
    g.add("hunter", near(base, 211))
    g.save("gen3")

    g.path_for(2).write_bytes(b"not an npz at all")   # the predecessor is bad too
    got = g.rollback()
    assert got == g.loaded_generation == 1
    assert g.total() == 6


# ---------------------------------- destroying only what is superseded
def test_drop_generations_removes_exactly_what_it_is_given(tmp_path):
    """The primitive ``--reset`` needs so it can destroy LAST rather than
    first: the enrolment is captured and saved, and only then do the
    generations it supersedes go. ``purge()`` cannot do this job -- it
    destroys everything and empties the object, which is only ever right when
    the answer to "what does he have afterwards" is "nothing"."""
    g = FaceGallery(root=tmp_path / "g")
    base = vec(3)
    for i in range(5):
        g.add("hunter", near(base, 10 + i))
    g.save(reason="one")
    for i in range(5):
        g.add("hunter", near(base, 30 + i))
    g.save(reason="two")
    for i in range(5):
        g.add("hunter", near(base, 50 + i))
    g.save(reason="three")
    assert g.generations() == [1, 2, 3]

    assert g.drop_generations([1, 2]) == 2
    assert g.generations() == [3]
    # The in-memory pool and the loaded generation are untouched: this is the
    # tail of a save, not a reset.
    assert g.total() == 15 and g.loaded_generation == 3
    assert FaceGallery(root=g.root).load() is True

    assert g.drop_generations([1, 2]) == 0, "a missing generation is not a guess"
    assert g.drop_generations([]) == 0
    assert g.generations() == [3]


def test_every_deletion_overwrites_the_embeddings_first(tmp_path):
    """unlink drops the directory ENTRY and leaves the 128-float vectors in
    the extents until the filesystem reuses them. Read back through a second
    hard link to the same inode -- zeros, not embeddings.

    This is a filesystem-level erase and the command says so in those words;
    an SSD's controller may still hold the old blocks, which is why nothing
    here claims the data is off the device."""
    import os
    g = FaceGallery(root=tmp_path / "g")
    base = vec(9)
    for i in range(6):
        g.add("hunter", near(base, 70 + i))
    g.save(reason="one")
    for i in range(6):
        g.add("hunter", near(base, 90 + i))
    g.save(reason="two")

    for gen, name in ((1, "twin1"), (2, "twin2")):
        path = g.path_for(gen)
        body = path.read_bytes()
        assert body.strip(b"\0"), "the fixture must not already be zeros"
        os.link(path, tmp_path / name)

    assert g.rollback() == 1                      # generation 2 goes
    assert (tmp_path / "twin2").read_bytes().strip(b"\0") == b""
    assert g.purge() == 1                         # and now generation 1
    assert (tmp_path / "twin1").read_bytes().strip(b"\0") == b""
