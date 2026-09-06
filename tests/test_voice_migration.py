"""His existing voiceprint becomes a labelled pool WITHOUT a re-enrolment.

THE TEST THAT MATTERS RUNS ON A COPY OF THE REAL FILE. The synthetic tests
below prove the code path; ``test_his_real_voiceprint_migrates_on_a_copy``
proves it against the fourteen embeddings that are actually on this box --
copied into tmp first, read as NUMBERS, never written back. It is skipped
cleanly on a machine that has no voiceprint, so the suite is identical
everywhere.

Nothing here opens the microphone or plays anything back. A stored embedding
is 192 floats; there is no audio in the file to begin with.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jarvis import voicegallery as vg
from tests.synthvoice import Voices, cos

REAL_VOICEPRINT = Path.home() / ".aiws_trainer" / "voiceprint.npz"


def _voiceprint(path, vectors, fmt=2):
    arrays = {"emb_%04d" % i: np.asarray(v, dtype=np.float32)
              for i, v in enumerate(vectors)}
    arrays["_format"] = np.array([fmt])
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)


@pytest.fixture()
def gal(tmp_path):
    return vg.VoiceGallery(root=tmp_path / "voice_gallery")


# --------------------------------------------------------------- the happy path
def test_a_format_2_voiceprint_migrates_under_one_label(gal, tmp_path):
    given = Voices(seed=1).takes("hunter", 14)
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, given)

    out = gal.migrate_voiceprint("hunter", path=src)
    assert out["ok"] is True
    assert out["found"] == 14 and out["migrated"] == 14 and out["dropped"] == 0
    assert out["generation"] == 1

    back = vg.VoiceGallery(root=gal.root)
    back.load()
    assert back.labels() == ("hunter",)
    assert back.count("hunter") == 14
    for a, b in zip(given, back.embeddings("hunter")):
        assert cos(a, b) == pytest.approx(1.0, abs=1e-6)
    takes = back.takes("hunter")
    assert all(t.src == "legacy" for t in takes)
    assert all("voiceprint.npz" in t.note for t in takes)
    assert back.consent("hunter") == "owner"


def test_the_source_file_is_never_written(gal, tmp_path):
    """It is the rollback. Reading it is the whole interaction."""
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, Voices(seed=2).takes("hunter", 12))
    before = src.read_bytes()
    assert gal.migrate_voiceprint("hunter", path=src)["ok"] is True
    assert src.read_bytes() == before


def test_undoing_the_migration_is_deleting_the_generation(gal, tmp_path):
    """Provably reversible: the only new state is one file, and the file it
    came from is still there, unchanged, holding the same numbers."""
    src = tmp_path / "voiceprint.npz"
    given = Voices(seed=3).takes("hunter", 14)
    _voiceprint(src, given)
    gal.migrate_voiceprint("hunter", path=src)
    assert gal.purge() == 1
    assert gal.generations() == []
    data = np.load(src)
    keys = sorted(k for k in data.files if k.startswith("emb_"))
    assert len(keys) == 14
    for a, k in zip(given, keys):
        assert cos(a, data[k]) == pytest.approx(1.0, abs=1e-6)


def test_the_migrated_pool_scores_the_same_as_the_voiceprint_did(gal, tmp_path):
    """The vectors MOVE, they are not recomputed, so leave-one-out through
    the gallery's centroid arithmetic must equal leave-one-out through
    speaker's. If these ever diverge, the migration has changed his scores
    without telling him."""
    given = [np.asarray(v, dtype=np.float64)
             for v in Voices(seed=4).takes("hunter", 14)]
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, given)
    gal.migrate_voiceprint("hunter", path=src)

    pool = [np.asarray(v, dtype=np.float64) for v in gal.embeddings("hunter")]
    for i in range(len(given)):
        want = cos(given[i], np.mean([e for j, e in enumerate(given)
                                      if j != i], axis=0))
        got = cos(pool[i], vg.centroid([e for j, e in enumerate(pool)
                                        if j != i]))
        assert got == pytest.approx(want, abs=1e-5)


# ------------------------------------------------------------- the refusals
def test_a_format_1_voiceprint_does_not_migrate(gal, tmp_path):
    """Pre-trim embeddings are not comparable with trimmed probes. Importing
    them under a fresh format number would hide a known-bad pool behind a new
    name; the existing re-enrol message stands instead."""
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, Voices(seed=5).takes("hunter", 10), fmt=1)
    out = gal.migrate_voiceprint("hunter", path=src)
    assert out["ok"] is False
    assert "format 1" in out["why"] and "re-enrol" in out["why"].lower()
    assert gal.generations() == []


def test_migrating_twice_is_refused(gal, tmp_path):
    """A second run would store the same fourteen takes again and double the
    label's weight in its own centroid -- which makes every later margin
    wrong, silently."""
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, Voices(seed=6).takes("hunter", 14))
    assert gal.migrate_voiceprint("hunter", path=src)["ok"] is True
    again = vg.VoiceGallery(root=gal.root)
    out = again.migrate_voiceprint("hunter", path=src)
    assert out["ok"] is False and "already" in out["why"]
    assert again.generations() == [1]


def test_a_missing_voiceprint_is_a_sentence_not_a_crash(gal, tmp_path):
    out = gal.migrate_voiceprint("hunter", path=tmp_path / "nope.npz")
    assert out["ok"] is False and "no voiceprint" in out["why"]


def test_a_degenerate_vector_in_the_source_is_dropped_and_named(gal, tmp_path):
    """0.07216878 in every element is the fixture that destroyed his pool. If
    one is still in there it must not be carried into the new store."""
    given = Voices(seed=7).takes("hunter", 12)
    given.append(np.full(192, 0.07216878, dtype=np.float32))
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, given)
    out = gal.migrate_voiceprint("hunter", path=src)
    assert out["ok"] is True
    assert out["found"] == 13 and out["migrated"] == 12 and out["dropped"] == 1


def test_a_bad_label_is_refused_before_anything_is_written(gal, tmp_path):
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, Voices(seed=8).takes("hunter", 10))
    out = gal.migrate_voiceprint("Hunter", path=src)
    assert out["ok"] is False
    assert gal.generations() == []


def test_a_refused_save_leaves_the_pool_as_it_was(gal, tmp_path):
    """A collapsed source cannot half-migrate: the in-memory pool is restored
    so the object is still usable and nothing is on disk."""
    one = Voices(seed=9).take("hunter")
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, [one + np.float32(1e-4) for _ in range(6)])
    out = gal.migrate_voiceprint("hunter", path=src)
    assert out["ok"] is False and "collapsed" in out["why"]
    assert gal.total() == 0 and gal.generations() == []


# ------------------------------------------------- against the REAL numbers
@pytest.mark.skipif(not REAL_VOICEPRINT.exists(),
                    reason="no enrolled voiceprint on this machine")
def test_his_real_voiceprint_migrates_on_a_copy(tmp_path):
    """ON A COPY. The real file is copied into tmp and the migration is run
    against the copy, so the original cannot be touched even by a bug in the
    code under test -- and the test asserts both files afterwards.

    This reads stored FLOATS. There is no audio in the file.
    """
    src = tmp_path / "voiceprint.npz"
    shutil.copy2(REAL_VOICEPRINT, src)
    original = REAL_VOICEPRINT.read_bytes()
    copied = src.read_bytes()

    data = np.load(src)
    keys = sorted(k for k in data.files if k.startswith("emb_"))
    fmt = int(data["_format"][0]) if "_format" in data.files else 1
    given = [np.asarray(data[k], dtype=np.float64) for k in keys]
    if fmt != 2:
        pytest.skip("the enrolled voiceprint is format %d, which does not "
                    "migrate by design" % fmt)

    gal = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    out = gal.migrate_voiceprint("hunter", path=src)
    assert out["ok"] is True, out["why"]
    assert out["migrated"] == len(keys)
    assert out["dropped"] == 0, "a stored embedding failed the vector guard"

    # NO RE-ENROLMENT: every vector arrives identical.
    back = vg.VoiceGallery(root=gal.root)
    back.load()
    assert back.count("hunter") == len(keys)
    for a, b in zip(given, back.embeddings("hunter")):
        assert cos(a, b) == pytest.approx(1.0, abs=1e-6)

    # And his leave-one-out floor is unchanged -- measured 2026-09-04 at
    # 0.570-0.798 on this pool. The bound is deliberately loose: it is a
    # sanity check that the numbers survived the move, not a re-measurement.
    pool = [np.asarray(v, dtype=np.float64) for v in back.embeddings("hunter")]
    loo = [cos(pool[i], vg.centroid([e for j, e in enumerate(pool) if j != i]))
           for i in range(len(pool))]
    assert min(loo) > vg.ACCEPT_DEFAULT, min(loo)
    assert 0.5 < min(loo) < max(loo) < 0.9, (min(loo), max(loo))

    # Neither file moved.
    assert src.read_bytes() == copied
    assert REAL_VOICEPRINT.read_bytes() == original
