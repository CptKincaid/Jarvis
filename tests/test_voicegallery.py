"""The voice gallery's read/write half: generations, the model marker, the
four write guards, and a delete that reads back before it shreds.

Every vector is SYNTHETIC (tests/synthvoice.py). Nothing here opens the
microphone, reads a recording, or touches ~/.aiws_trainer/voiceprint.npz.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import speaker as sp
from jarvis import voicegallery as vg
from tests.synthvoice import Voices, cos


@pytest.fixture()
def gal(tmp_path):
    return vg.VoiceGallery(root=tmp_path / "voice_gallery")


def _fill(gallery, world, label, n, **kw):
    for e in world.takes(label, n):
        gallery.add(label, e, **kw)


# ------------------------------------------------------ the shared contract
def test_the_label_pattern_matches_the_other_two_stores():
    """Three stores that disagree about who exists is three stores that
    silently stop naming somebody. Asserted, not commented."""
    from jarvis import facegallery as fg
    from jarvis import identity as ident
    assert vg._LABEL_RE.pattern == fg._LABEL_RE.pattern
    assert vg._LABEL_RE.pattern == ident.LABEL_RX.pattern


def test_the_accept_bar_is_speakers_own_number():
    """0.30 is measured and it is not re-derived here. Duplicated only
    because speaker.py imports this module, so this may not import it back."""
    assert vg.ACCEPT_DEFAULT == sp.DEFAULT_THRESHOLD == 0.30


def test_the_abstain_window_is_speakers_own_number():
    assert vg.ABSTAIN_SECONDS == sp.ABSTAIN_SECONDS == 1.5


def test_the_store_is_not_the_voiceprint():
    """A new path AND a new format namespace. Sharing either is what lets an
    old single-speaker loader pool several people into one centroid."""
    from jarvis.config import PATHS
    assert PATHS.VOICE_GALLERY != PATHS.VOICEPRINT
    assert PATHS.VOICE_GALLERY.suffix == ""      # a directory, not a file


# ----------------------------------------------------------- round tripping
def test_a_pool_survives_a_save_and_a_load_bit_for_bit(gal):
    world = Voices(seed=1)
    given = world.takes("hunter", 9)
    for i, e in enumerate(given):
        gal.add("hunter", e, len_s=3.2 + i, rms=0.031, note="say your name",
                at="2026-09-04T21:00:00", src="enrol")
    gal.set_consent("hunter", "owner")
    gen = gal.save("enrol hunter")
    assert gen == 1

    back = vg.VoiceGallery(root=gal.root)
    assert back.load() is True
    assert back.labels() == ("hunter",)
    assert back.count("hunter") == 9
    for a, b in zip(given, back.embeddings("hunter")):
        assert cos(a, b) == pytest.approx(1.0, abs=1e-6)
    t = back.takes("hunter")[3]
    assert t.note == "say your name" and t.src == "enrol"
    assert t.len_s == pytest.approx(6.2)
    assert back.consent("hunter") == "owner"
    assert back.provenance()["model"] == vg.ECAPA_MODEL


def test_two_people_live_in_one_generation_without_pooling(gal):
    world = Voices(seed=2)
    _fill(gal, world, "hunter", 10)
    _fill(gal, world, "mara", 8)
    gal.save("two people")
    back = vg.VoiceGallery(root=gal.root)
    back.load()
    assert back.labels() == ("hunter", "mara")
    assert back.count("hunter") == 10 and back.count("mara") == 8
    ch, cm = back.centroid("hunter"), back.centroid("mara")
    assert cos(ch, cm) < 0.99, "two labels collapsed into one centroid"


def test_a_take_with_no_metadata_is_a_take_not_an_error(gal):
    """Optional keys at the SAME format. A pool with nothing recorded must
    round-trip exactly as it did before those keys existed."""
    world = Voices(seed=3)
    _fill(gal, world, "hunter", 8)
    gal.save("bare")
    back = vg.VoiceGallery(root=gal.root)
    assert back.load() is True
    assert back.count("hunter") == 8
    assert all(t.recorded is False for t in back.takes("hunter"))


def test_takes_are_padded_never_short(gal):
    world = Voices(seed=4)
    gal.add("hunter", world.take("hunter"), note="one")
    gal._pool["hunter"].append(world.take("hunter"))   # a list out of step
    assert len(gal.takes("hunter")) == 2


# ------------------------------------------------------------- the refusals
def test_a_format_it_does_not_know_is_refused(gal):
    world = Voices(seed=5)
    _fill(gal, world, "hunter", 8)
    gal.save("v1")
    path = gal.path_for(1)
    data = {k: v for k, v in np.load(path).items()}
    data["_format"] = np.array([FUTURE := 4])
    with open(path, "wb") as fh:
        np.savez(fh, **data)
    fresh = vg.VoiceGallery(root=gal.root)
    assert fresh.load() is False
    assert fresh.total() == 0
    assert FUTURE == 4


def test_a_generation_with_no_model_key_is_refused(gal):
    """Unlike the face gallery there is NO legacy generation to be generous
    to -- nothing but this code has ever written this store, so a file with no
    model name was not written by it."""
    world = Voices(seed=6)
    _fill(gal, world, "hunter", 8)
    gal.save("v1")
    path = gal.path_for(1)
    data = {k: v for k, v in np.load(path).items() if k != "_model"}
    with open(path, "wb") as fh:
        np.savez(fh, **data)
    assert vg.VoiceGallery(root=gal.root).load() is False


def test_another_models_generation_is_refused_by_name_and_left_alone(gal):
    world = Voices(seed=7)
    _fill(gal, world, "hunter", 8)
    gal.save("v1")
    path = gal.path_for(1)
    data = {k: v for k, v in np.load(path).items()}
    data["_model"] = np.array(["some_other_encoder"])
    with open(path, "wb") as fh:
        np.savez(fh, **data)
    before = path.read_bytes()
    fresh = vg.VoiceGallery(root=gal.root)
    assert fresh.load() is False
    assert fresh.foreign_generations == {1: "some_other_encoder"}
    assert path.read_bytes() == before, "a foreign generation was touched"


def test_a_width_that_contradicts_the_model_is_refused(gal):
    world = Voices(seed=8)
    _fill(gal, world, "hunter", 8)
    gal.save("v1")
    path = gal.path_for(1)
    data = {k: v for k, v in np.load(path).items()}
    data["_dim"] = np.array([512])
    with open(path, "wb") as fh:
        np.savez(fh, **data)
    assert vg.VoiceGallery(root=gal.root).load() is False


@pytest.mark.parametrize("bad,why", [
    (np.full(192, 0.07216878, dtype=np.float32), "constant"),
    (np.zeros(192, dtype=np.float32), "all zero"),
    (np.full(512, 0.5, dtype=np.float32), "wrong dimension"),
])
def test_a_degenerate_vector_is_refused_at_add(gal, bad, why):
    """0.07216878 is not a hypothetical: it is every element of the fixture
    that replaced his voiceprint on 2026-09-02."""
    with pytest.raises(ValueError):
        gal.add("hunter", bad)
    assert why


def test_a_non_finite_vector_is_refused(gal):
    v = Voices(seed=9).take("hunter").astype(np.float64)
    v[7] = np.nan
    with pytest.raises(ValueError):
        gal.add("hunter", v)


@pytest.mark.parametrize("label", ["", "Hunter", "hunter!", "-hunter",
                                   "x" * 32, None])
def test_a_label_the_other_stores_would_refuse_is_refused_here(gal, label):
    with pytest.raises(ValueError):
        gal.add(label, Voices(seed=10).take("x"))


def test_a_collapsed_pool_is_refused_at_save(gal):
    """Two copies of one vector is what the 2026-09-02 accident LEFT BEHIND,
    and the old save() accepted it atomically."""
    one = Voices(seed=11).take("hunter")
    for _ in range(4):
        gal.add("hunter", one + np.float32(1e-4))
    with pytest.raises(ValueError, match="collapsed"):
        gal.save("collapsed")
    assert gal.generations() == [], "a refused save still wrote a file"


def test_his_own_measured_cohesion_passes_the_collapsed_bar(gal):
    """0.90 must sit well clear of a real pool. The fixture reproduces his
    measured median pairwise cosine of 0.485."""
    world = Voices(seed=12)
    _fill(gal, world, "hunter", 14)
    med = vg.median_pairwise(gal.embeddings("hunter"))
    assert 0.35 < med < 0.65, med
    gal.save("real-ish")


def test_an_empty_gallery_is_refused_at_save(gal):
    with pytest.raises(ValueError, match="empty"):
        gal.save("nothing")


# -------------------------------------------------------- the shrink guard
def test_shrinking_needs_saying_so(gal):
    world = Voices(seed=13)
    _fill(gal, world, "hunter", 10)
    gal.save("ten")
    small = vg.VoiceGallery(root=gal.root)
    _fill(small, world, "hunter", 3)          # never loaded: baseline in memory is 0
    with pytest.raises(ValueError, match="shrink"):
        small.save("three")
    assert small.save("three", allow_shrink=True) == 2


def test_the_shrink_baseline_comes_from_the_disk_too(gal):
    """THE INCIDENT'S OWN SHAPE. The write that destroyed the voiceprint came
    from a freshly built object that had loaded nothing, so an in-memory-only
    baseline is 0 and the guard abstains on exactly its motivating case."""
    world = Voices(seed=14)
    _fill(gal, world, "hunter", 12)
    gal.save("twelve")
    naive = vg.VoiceGallery(root=gal.root)
    assert naive._loaded_n == 0
    _fill(naive, world, "hunter", 2)
    with pytest.raises(ValueError, match="12 samples to 2"):
        naive.save("two")


# ------------------------------------------------------------- generations
def test_a_save_never_overwrites(gal):
    world = Voices(seed=15)
    _fill(gal, world, "hunter", 8)
    assert gal.save("one") == 1
    gal.add("hunter", world.take("hunter"))
    assert gal.save("two") == 2
    assert gal.generations() == [1, 2]


def test_the_newest_that_parses_wins(gal):
    world = Voices(seed=16)
    _fill(gal, world, "hunter", 8)
    gal.save("good")
    gal.add("hunter", world.take("hunter"))
    gal.save("also good")
    gal.path_for(2).write_bytes(b"not an npz")
    fresh = vg.VoiceGallery(root=gal.root)
    assert fresh.load() is True
    assert fresh.loaded_generation == 1 and fresh.count("hunter") == 8


def test_the_richest_generation_is_never_pruned(gal):
    """Five bad writes must not evict a good enrolment. Measured on the face
    gallery 2026-09-02: one 6-sample enrolment plus six 2-sample saves left
    [4,5,6,7,8], every one n=2, and the enrolment gone."""
    world = Voices(seed=17)
    _fill(gal, world, "hunter", 12)
    gal.save("the enrolment")
    for i in range(6):
        small = vg.VoiceGallery(root=gal.root)
        _fill(small, world, "hunter", 2)
        small.save("small %d" % i, allow_shrink=True)
    assert 1 in gal.generations(), "the 12-sample enrolment was pruned away"
    fresh = vg.VoiceGallery(root=gal.root)
    fresh.load(1)
    assert fresh.count("hunter") == 12


def test_rollback_undoes_the_last_save(gal):
    world = Voices(seed=18)
    _fill(gal, world, "hunter", 10)
    gal.save("good")
    gal.add("hunter", np.full(192, 3.0, dtype=np.float32) +
            np.asarray(world.take("hunter"), dtype=np.float32) * 0.0001)
    gal.save("regret", allow_shrink=True)
    assert gal.rollback() == 1
    assert gal.count("hunter") == 10


def test_rollback_refuses_when_there_is_nothing_to_fall_back_to(gal):
    world = Voices(seed=19)
    _fill(gal, world, "hunter", 8)
    gal.save("only one")
    assert gal.rollback() == 0
    assert gal.generations() == [1]


def test_a_crashed_saves_tmp_is_visible_and_is_swept(gal):
    """A .tmp holds a WHOLE pool under a name generations() cannot see."""
    world = Voices(seed=20)
    _fill(gal, world, "hunter", 8)
    gal.save("one")
    (gal.root / "gen-00002.npz.tmp").write_bytes(b"a whole pool")
    assert gal.leftovers() == ["gen-00002.npz.tmp"]
    gal.add("hunter", world.take("hunter"))
    gal.save("two")
    assert gal.leftovers() == []


def test_the_files_are_private(gal):
    world = Voices(seed=21)
    _fill(gal, world, "hunter", 8)
    gal.save("one")
    assert oct(gal.root.stat().st_mode)[-3:] == "700"
    assert oct(gal.path_for(1).stat().st_mode)[-3:] == "600"


# ------------------------------------------------- withdrawing consent
def test_purge_label_removes_one_person_and_keeps_the_others(gal):
    world = Voices(seed=22)
    _fill(gal, world, "hunter", 12)
    _fill(gal, world, "mara", 8)
    gal.set_consent("mara", "typed")
    gal.save("both")
    out = gal.purge_label("mara", reason="consent withdrawn")
    assert out["complete"] is True
    assert out["still_holding"] == []
    assert "mara" not in gal.disk_labels()
    assert gal.count("hunter") == 12


def test_purge_label_destroys_nothing_it_has_not_proved_superseded(gal):
    """THE INVARIANT. With the replacement unreadable, nothing goes -- and
    the caller is told the delete is incomplete rather than told it worked."""
    world = Voices(seed=23)
    _fill(gal, world, "hunter", 10)
    _fill(gal, world, "mara", 8)
    gal.save("both")

    real_save = gal.save

    def corrupting_save(*a, **kw):
        gen = real_save(*a, **kw)
        gal.path_for(gen).write_bytes(b"truncated")
        return gen

    gal.save = corrupting_save
    out = gal.purge_label("mara")
    gal.save = real_save
    assert out["complete"] is False
    assert out["reason"]
    assert out["removed"] == 0
    assert 1 in gal.generations(), "the original was destroyed unproven"


def test_purge_label_sweeps_the_tmps_too(gal):
    world = Voices(seed=24)
    _fill(gal, world, "hunter", 10)
    _fill(gal, world, "mara", 8)
    gal.save("both")
    (gal.root / "gen-00009.npz.tmp").write_bytes(b"a whole pool")
    out = gal.purge_label("mara")
    assert out["tmp_removed"] == 1
    assert out["complete"] is True


def test_purge_label_on_somebody_who_is_not_there_destroys_nothing(gal):
    world = Voices(seed=25)
    _fill(gal, world, "hunter", 10)
    gal.save("him")
    out = gal.purge_label("heather")
    assert out["removed"] == 0 and out["complete"] is True
    assert gal.disk_labels() == ("hunter",)


def test_purge_removes_every_generation_and_every_tmp(gal):
    world = Voices(seed=26)
    _fill(gal, world, "hunter", 8)
    gal.save("one")
    gal.add("hunter", world.take("hunter"))
    gal.save("two")
    (gal.root / "gen-00007.npz.tmp").write_bytes(b"x")
    assert gal.purge() == 3
    assert gal.generations() == [] and gal.leftovers() == []


def test_forget_is_not_a_delete(gal):
    """In memory only. The generations still hold them, one rollback away --
    which is why withdrawn consent is purge_label and not this."""
    world = Voices(seed=27)
    _fill(gal, world, "hunter", 10)
    _fill(gal, world, "mara", 8)
    gal.save("both")
    assert gal.forget("mara") == 8
    assert gal.labels() == ("hunter",)
    fresh = vg.VoiceGallery(root=gal.root)
    fresh.load()
    assert "mara" in fresh.labels()


# --------------------------------------------------------- provisional-ness
def test_a_label_under_eight_takes_is_provisional(gal):
    world = Voices(seed=28)
    _fill(gal, world, "mara", 5)
    _fill(gal, world, "hunter", 14)
    assert gal.provisional("mara") is True
    assert gal.provisional("hunter") is False
    assert gal.provisional("nobody") is False


def test_eight_takes_is_where_the_centroid_has_settled():
    """The measured curve behind MIN_TAKES_TO_NAME. On his real pool
    (2026-09-04): cos(centroid of k takes, converged centroid) is k=2 0.841,
    k=5 0.943, k=8 0.975. Reproduced here on the calibrated fixture."""
    world = Voices(seed=29)
    pool = world.takes("hunter", 14)
    full = vg.centroid(pool)
    got = {k: float(np.mean([cos(vg.centroid(pool[:k]), full)
                             for _ in range(1)])) for k in (2, 5, 8, 12)}
    assert got[2] < got[5] < got[8] < got[12]
    assert got[8] > 0.95, got
