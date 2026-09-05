"""THE PREVIOUS FIX LOCKED HIM OUT, AND ORDINARY USE IS ENOUGH TO DO IT.

Three faults, measured independently, all on synthetic vectors
(tests/synthvoice.py). No microphone, no recording, no real voiceprint, no
real gallery, no config, no Jarvis process.

1. THE REPORTED SCORE IS NOT THE SCORE OF THE LABEL IT NAMES.
   ``speaker._ident`` builds the gate's ``top`` from TWO independent values::

       if verdict.scores and float(verdict.score) >= self.threshold:
           who_top = str(verdict.scores[0][0] or "")

   ``verdict.score`` is rank 1 AT IDENTIFY TIME; ``verdict.scores[0][0]`` is
   the label at rank 1 AFTER ``_strip_disowned`` has removed one. Once his own
   label is disowned the two describe different rows, and the gate is handed a
   label it is told is above the bar when it is nowhere near it. Measured here
   at apart 0.3, his own voice, his own box:

       top='mara'   who_scores={'mara': 0.14}   bar 0.30

   -0.16 under the bar, reported as over it, and ``gate._voice_leg``'s
   "the gallery's best guess is somebody else" refuses him for it.

2. PASSIVE LEARNING WALKS HIM OUT OF HIS OWN POOL, WITH NO COMMAND AT ALL.
   ``speaker.add_sample`` moves ``voiceprint.npz``'s centroid. The gallery's
   migrated copy of him does not move. ``_disowned`` compares the two on
   ``voicegallery.OWNER_POOL_COSINE`` (0.98), so ordinary use walks the two
   apart until his own label stops being read as his. Measured ladder, two
   passive samples per restart, seed 3 apart 0.3 (the same shape at apart 1.0
   and 2.0):

       passive   0    2    4    6    8   10   12   14   16 |  18   20   30
       cosine  1.000 .995 .991 .989 .986 .984 .983 .981 .981| .980 .979 .975
       admits   30   30   30   30   30   30   30   30   30 |   0    0    0
                                                     LOCKED ^ nine restarts

3. THERE IS NO SUPPORTED WAY BACK. ``migrate_voiceprint`` refuses ("hunter
   already has embeddings"), ``pool_ok`` refuses fresh takes ("would pull
   hunter away from voiceprint.npz"), and what is left is deleting a
   generation file by hand -- which is a re-enrolment by another name and
   breaks this lane's one hard rule.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from jarvis import speaker as sp
from jarvis import voicegallery as vg
from tests.synthvoice import Voices, cos
from tests.test_voice_multispeaker_wiring import _FakeEncoder, _gate

ROOT = Path(__file__).resolve().parent.parent


def _voice_enrol():
    spec = importlib.util.spec_from_file_location(
        "_voice_enrol_under_drift_test", ROOT / "scripts" / "voice_enrol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """A verifier told who the owner is, the way app.py builds it."""
    monkeypatch.setattr(sp, "VOICEPRINT_FILE", tmp_path / "voiceprint.npz")
    enc = _FakeEncoder()
    v = sp.SpeakerVerifier()
    v.owner_label = "hunter"
    v._model_loaded = True
    monkeypatch.setattr(v, "_ensure_model", lambda: True)
    monkeypatch.setattr(v, "_extract_embedding", enc)
    v.gallery = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    return v, enc, _gate(tmp_path)


def _clip(seconds, speech_s, marker):
    """A clip whose TRIMMED speech is ``speech_s`` exactly."""
    n = int(sp.SAMPLE_RATE * seconds)
    rng = np.random.default_rng(int(abs(marker) * 100000) + 5)
    a = (rng.normal(size=n) * 0.001).astype(np.float32)
    k = int(sp.SAMPLE_RATE * speech_s)
    a[:k] = (rng.normal(size=k) * 0.2).astype(np.float32)
    a[0] = np.float32(marker)
    return a


def _layout(v, enc, world, *, guest="mara", his=14, hers=10, save=True):
    """His migrated pool + one guest, and his voiceprint holding the SAME
    fourteen vectors -- what ``--migrate`` leaves behind."""
    him = world.takes("hunter", his)
    for e in him:
        v.gallery.add("hunter", e, src="legacy",
                      note="migrated from voiceprint.npz format 2")
    v.gallery.set_consent("hunter", "owner")
    if hers:
        for e in world.takes(guest, hers):
            v.gallery.add(guest, e, src="enrol")
    if save:
        v.gallery.save(reason="test layout")
    v._embeddings = list(him)
    v._recompute_centroid()
    return him


def _turns(v, enc, gate, world, n=30, base=0.9, speaker="hunter"):
    """``(admitted, [stats, ...])`` for ``n`` ordinary turns of his."""
    ok = 0
    out = []
    for i in range(n):
        clip = _clip(4.0, 3.0, base + i * 1e-4)
        enc.teach(clip, world.take(speaker))
        filt, stats = v.filter_segments(clip)
        d = gate.judge("voice", "unlock the door", stats=stats,
                       rejected=filt is None)
        ok += int(d.admit and d.who == "hunter")
        out.append(stats)
    return ok, out


def _restarts(v, enc, world, n_restarts, per=2, base=0.3):
    """``n_restarts`` restarts' worth of ordinary passive learning. The
    session cap is reset each time, which is exactly what a restart does --
    ``_passive_added`` and ``_frozen_centroid`` are process state."""
    added = 0
    for r in range(n_restarts):
        v._passive_added = 0
        v._frozen_centroid = None
        for j in range(per):
            clip = _clip(4.0, 3.0, base + r * 0.01 + j * 1e-3)
            enc.teach(clip, world.take("hunter"))
            added += int(bool(v.add_sample(clip)))
    return added


def _anchor_cos(v):
    """The cosine ``speaker._disowned`` actually judges him on."""
    return cos(v._centroid, v.gallery.centroid("hunter"))


# =================================================== 1. the reported score
def test_a_verdicts_top_label_and_its_score_are_one_measurement():
    """THE ROOT. ``score`` and ``scores[0]`` were two values that had to be
    kept in step by hand, and ``_strip_disowned`` did not keep them: it
    filtered the ranking and left the scalar, deliberately, on the argument
    that a stale number there could only ever withhold a name. It could not
    only withhold: paired with a label from the filtered list it INVENTS a
    row -- somebody else's name carrying his score.

    The type must make that impossible. Every scalar is derived from the one
    ranking, so ``dataclasses.replace(v, scores=...)`` cannot leave a stale
    one behind."""
    v = vg.VoiceVerdict(who="hunter", scores=(("hunter", 0.70),
                                              ("mara", 0.14)))
    assert v.score == pytest.approx(0.70)
    assert v.top_label == "hunter"
    assert v.margin == pytest.approx(0.56)

    stripped = dataclasses.replace(v, who="", scores=(("mara", 0.14),))
    assert stripped.top_label == "mara"
    assert stripped.score == pytest.approx(0.14), (
        "the score still describes the row that was removed")
    assert stripped.second == "" and stripped.margin is None


def test_a_verdicts_ranking_is_ranked_by_construction():
    """Not by the care of whoever builds one. ``score`` means "rank 1", so
    rank 1 has to be a fact about the tuple and not a promise about the
    caller."""
    v = vg.VoiceVerdict(scores=(("mara", 0.14), ("hunter", 0.70)))
    assert v.scores[0] == ("hunter", pytest.approx(0.70))
    assert v.top_label == "hunter" and v.score == pytest.approx(0.70)


def test_the_gate_is_never_told_a_below_bar_label_is_above_the_bar(rig):
    """THE MEASURED FAULT, end to end, on the layout ordinary use produces.

    ``gate._voice_leg`` reads ``top`` as "the label identify() ranked first
    ABOVE THE BAR". Whatever else is true of a turn, that sentence has to be
    true of the number beside it."""
    v, enc, gate = rig
    world = Voices(seed=3, apart=0.3)
    _layout(v, enc, world)
    _restarts(v, enc, world, 12)          # drift him out, the ordinary way
    _ok, stats = _turns(v, enc, gate, world)
    bad = [(s.get("top"), s.get("who_scores", {}).get(s.get("top")))
           for s in stats
           if s.get("top")
           and s["who_scores"].get(s["top"], -9.0) < v.threshold]
    assert not bad, ("%d of %d turns reported a label below the %.2f bar as "
                     "the best guess above it: %s"
                     % (len(bad), len(stats), v.threshold, bad[:3]))


def test_recomputing_only_the_reported_score_changes_nothing(rig):
    """THE ONE VALUE, ISOLATED. Judge each turn twice: once with the stats as
    the pipeline built them, once with ``top`` recomputed from the ranking it
    is supposed to come from. Two readings of one measurement must agree --
    and they must both admit him, because it is his voice on his box.

    Before the fix: as-is 0 of 30, recomputed 30 of 30."""
    v, enc, gate = rig
    world = Voices(seed=3, apart=0.3)
    _layout(v, enc, world)
    _restarts(v, enc, world, 12)
    as_is = fixed = 0
    for i in range(30):
        clip = _clip(4.0, 3.0, 0.5 + i * 1e-4)
        enc.teach(clip, world.take("hunter"))
        filt, stats = v.filter_segments(clip)
        rejected = filt is None
        as_is += int(gate.judge("voice", "unlock the door", stats=stats,
                                rejected=rejected).admit)
        ranked = sorted(stats["who_scores"].items(), key=lambda kv: -kv[1])
        top = ranked[0][0] if ranked and ranked[0][1] >= v.threshold else ""
        honest = dict(stats, top=top)
        fixed += int(gate.judge("voice", "unlock the door", stats=honest,
                                rejected=rejected).admit)
    assert (as_is, fixed) == (30, 30), (
        "as-is %d of 30, with `top` recomputed from its own ranking %d of 30"
        % (as_is, fixed))


# ========================================================= 2. the drift
@pytest.mark.parametrize("apart", [0.3, 1.0, 2.0])
def test_ordinary_passive_learning_never_walks_him_out_of_his_own_pool(
        rig, apart):
    """NO COMMAND, NO ATTACKER, NO MISTAKE -- just use. Twenty restarts is
    forty passive samples, more than twice what it took to lock him out.

    The bound is the SAME number the runtime disowns on: a passive sample is
    refused when the centroid it would produce would stop measuring as his
    gallery pool. One number, two enforcers, so the walk can never reach the
    place the door is.

    THE COSINE IS ASSERTED AT EVERY SEPARATION; THE ADMIT RATE ONLY WHERE THE
    LAYOUT CAN EXIST. At apart 2.0 ``pool_ok`` refuses to enrol the guest at
    all (measured here: median margin 0.140 against the 0.20 bar), and on the
    layout forced past it he is already admitted 3 of 30 with ZERO passive
    samples -- ``identify``'s margin rule, pre-existing, nothing to do with
    drift, and too near the floor for 30 turns to say anything. Claiming this
    fix holds his admit rate there would be reading noise as a result. What
    IS asserted at 2.0 is the invariant this fix is about: the walk stops."""
    v, enc, gate = rig
    world = Voices(seed=3, apart=apart)
    _layout(v, enc, world)
    baseline, _s = _turns(v, enc, gate, world)
    ladder = []
    for r in range(20):
        ladder.append((r * 2, _anchor_cos(v)))
        _restarts(v, enc, world, 1, base=0.3 + r * 0.05)
    ladder.append((40, _anchor_cos(v)))
    floor = min(c for _n, c in ladder)
    assert floor >= vg.OWNER_POOL_COSINE, (
        "passive learning walked him to cos %.4f, under the %.2f line his own "
        "label is read as his on. Ladder: %s"
        % (floor, vg.OWNER_POOL_COSINE,
           ", ".join("%d:%.4f" % t for t in ladder)))
    ok, _stats = _turns(v, enc, gate, world)
    if apart <= 1.0:
        assert (baseline, ok) == (30, 30), (
            "%d of 30 before 40 passive samples, %d after" % (baseline, ok))
    else:
        assert baseline < 10, (
            "apart 2.0 is supposed to be the margin rule's own floor; it "
            "admitted %d of 30 before any drift, so this branch is now "
            "measuring something else" % baseline)


def test_the_drift_bound_actually_bites(rig):
    """NOT VACUOUS. The ladder above has to be held by a REFUSAL, not by a
    seed that happened not to drift. At least one sample must be turned away,
    and the cosine must come to rest above the line rather than never having
    approached it."""
    v, enc, gate = rig
    world = Voices(seed=3, apart=0.3)
    _layout(v, enc, world)
    added = _restarts(v, enc, world, 30, base=0.3)
    assert added < 60, "every one of 60 passive samples was accepted"
    assert _anchor_cos(v) >= vg.OWNER_POOL_COSINE
    assert v.gallery.count("hunter") == 14, "the gallery must not have moved"


def test_the_drift_ladder_is_pinned_as_numbers(rig):
    """A REGRESSION HERE HAS TO BE VISIBLE AS A NUMBER, not as a lockout six
    restarts into his week. The unbounded walk is reproduced by writing the
    pool directly -- which is what a box that drifted before this fix looks
    like -- and the bounded one by the code under test."""
    v, enc, gate = rig
    world = Voices(seed=3, apart=0.3)
    him = _layout(v, enc, world)

    # The unbounded walk, reproduced: append without asking.
    pool = list(him)
    unbounded = []
    for r in range(15):
        unbounded.append(cos(np.mean(pool, axis=0), v.gallery.centroid("hunter")))
        for _j in range(2):
            pool.append(np.asarray(world.take("hunter"), dtype=np.float32))
    assert unbounded[0] == pytest.approx(1.0, abs=1e-6)
    assert min(unbounded) < vg.OWNER_POOL_COSINE, (
        "the unbounded walk no longer reproduces the lockout: %s"
        % ["%.4f" % c for c in unbounded])

    # The bounded one, through add_sample.
    _restarts(v, enc, world, 15)
    assert _anchor_cos(v) >= vg.OWNER_POOL_COSINE


def test_passive_learning_is_unchanged_when_he_has_no_gallery_pool(rig):
    """A SINGLE-SPEAKER BOX MUST BEHAVE EXACTLY AS IT DID. With no label of
    his in the gallery there is no anchor to drift away from and no invariant
    to keep, so the bound must not apply -- adding one would make this
    feature cost something on a box that never asked for it."""
    v, enc, gate = rig
    world = Voices(seed=11, apart=0.3)
    him = world.takes("hunter", 14)
    v._embeddings = list(him)
    v._recompute_centroid()
    assert v.gallery.labels() == ()
    assert _restarts(v, enc, world, 10) == 20, (
        "passive learning was bounded on a box with nothing to bound it "
        "against")


def test_the_bound_and_the_door_are_the_same_number(rig):
    """ONE NUMBER, TWO ENFORCERS. ``add_sample`` must not carry a bar of its
    own that could drift away from the one ``_disowned`` applies."""
    src = inspect.getsource(sp.SpeakerVerifier.add_sample)
    assert "OWNER_POOL_COSINE" not in src, (
        "add_sample re-derives the bar instead of asking the predicate that "
        "does the disowning")
    v, _enc, _gate = rig
    assert sp.MIGRATED_ALIAS_COSINE == vg.OWNER_POOL_COSINE


# ======================================================== 3. the recovery
def _drifted(v, enc, world, restarts=12):
    """A box that drifted BEFORE the bound shipped: the pool is written
    directly, which is the state on disk this recovery exists to repair."""
    him = _layout(v, enc, world)
    pool = list(him)
    for r in range(restarts * 2):
        pool.append(np.asarray(world.take("hunter"), dtype=np.float32))
    v._embeddings = pool
    v._recompute_centroid()
    return pool


def _write_voiceprint(path, vectors):
    np.savez(path, _format=np.array([2]),
             **{"emb_%04d" % i: np.asarray(e, dtype=np.float32)
                for i, e in enumerate(vectors)})
    return path


def test_a_disowned_owner_has_a_supported_way_back_with_no_re_enrolment(rig,
                                                                        tmp_path):
    """THE LANE'S OWN HARD RULE. He does not pay for multi-speaker with a
    re-enrolment -- not to get in, and not to get back in.

    The recovery is a RE-ANCHOR: his gallery label is refilled from the
    voiceprint he already has, no microphone, no takes, no file deleted by
    hand.

    APART 1.0, AND THE REASON MATTERS. Once the reported-score fault above is
    fixed, a drifted box at apart 0.3 admits him again anyway -- the guest is
    so far off that nothing outscores him. The lockout SURVIVES that fix
    wherever the guest's own centroid clears the 0.30 bar on his voice, which
    at apart 1.0 it does (measured 0.41 here). So the two faults really are
    independent, and this is the one the recovery is for: he is still
    DISOWNED, his own label is still not read as his, and no amount of the
    first fix gets it back.

    THE NUMBER, STATED HONESTLY. With the reported score fixed this is not
    the reported 0 of 100 any more -- it is 8 of 30 here, because on some
    clips the guest happens to fall under the bar and he gets through. A
    two-thirds refusal rate on his own box is the thing being repaired; the
    original 0 of 100 was the two faults compounding."""
    v, enc, gate = rig
    world = Voices(seed=3, apart=1.0)
    pool = _drifted(v, enc, world)
    assert _anchor_cos(v) < vg.OWNER_POOL_COSINE, "the drift did not reproduce"
    before, _s = _turns(v, enc, gate, world)
    assert before <= 10, ("expected the drift to still cost him most of his "
                          "turns; it admitted %d of 30" % before)

    src = _write_voiceprint(tmp_path / "vp.npz", pool)
    out = v.gallery.reanchor_voiceprint("hunter", path=src)
    assert out["ok"] is True, out["why"]
    assert out["migrated"] == len(pool) and out["replaced"] == 14
    v.gallery.load()
    assert _anchor_cos(v) == pytest.approx(1.0, abs=1e-6)
    after, _s = _turns(v, enc, gate, world, base=0.6)
    assert after == 30, "%d of 30 after the re-anchor" % after
    assert all(t.src == "legacy" for t in v.gallery.takes("hunter")), \
        "a re-anchor must not invent enrolment takes"
    assert v.gallery.consent("hunter") == "owner"


def test_the_recovery_leaves_the_previous_generation_to_roll_back_to(rig,
                                                                     tmp_path):
    """It is a WRITE of his pool, so it is reversible the way every other
    write to this store is: the generation before it is still on disk."""
    v, enc, gate = rig
    world = Voices(seed=5, apart=0.3)
    pool = _drifted(v, enc, world)
    gens_before = v.gallery.generations()
    src = _write_voiceprint(tmp_path / "vp.npz", pool)
    out = v.gallery.reanchor_voiceprint("hunter", path=src)
    assert out["ok"], out["why"]
    assert v.gallery.generations()[-1] > gens_before[-1]
    assert set(gens_before) <= set(v.gallery.generations())
    v.gallery.rollback()
    assert v.gallery.count("hunter") == 14


def test_the_recovery_refuses_a_voiceprint_that_is_not_his(rig, tmp_path):
    """THE MISTAKE IT CATCHES: pointing it at the wrong file. The voiceprint
    has to still measure as the pool it is replacing, at that pool's OWN
    measured floor -- the smallest leave-one-out cosine across his takes,
    which is the same instrument ``passive_ok`` bars a passive sample on.

    It is a mistake-catcher and not a lock, and the difference is stated
    rather than implied: whoever can write ``voiceprint.npz`` is already the
    owner as far as ``_owner_pools`` is concerned, so this grants nothing a
    takeover would not already have."""
    v, enc, gate = rig
    world = Voices(seed=3, apart=0.3)
    _drifted(v, enc, world)
    src = _write_voiceprint(tmp_path / "hers.npz", world.takes("mara", 14))
    out = v.gallery.reanchor_voiceprint("hunter", path=src)
    assert out["ok"] is False
    assert "%.3f" % out["cosine"] in out["why"], out["why"]
    assert v.gallery.count("hunter") == 14, "a refusal wrote something"


def test_the_recovery_refuses_a_pool_that_has_not_drifted(rig, tmp_path):
    """Nothing to repair is a REFUSAL, not a no-op that rewrites his pool:
    a healthy box must not be able to spend a generation on this by
    accident."""
    v, enc, gate = rig
    world = Voices(seed=7, apart=0.3)
    him = _layout(v, enc, world)
    src = _write_voiceprint(tmp_path / "vp.npz", him)
    out = v.gallery.reanchor_voiceprint("hunter", path=src)
    assert out["ok"] is False and "already" in out["why"].lower(), out["why"]
    assert out["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_the_recovery_will_not_create_a_pool_that_does_not_exist(rig, tmp_path):
    """A re-anchor REPAIRS; ``--migrate`` creates. Letting this one create
    would put a second door beside the one the owner guard was built on."""
    v, enc, gate = rig
    world = Voices(seed=9, apart=0.3)
    src = _write_voiceprint(tmp_path / "vp.npz", world.takes("hunter", 14))
    out = v.gallery.reanchor_voiceprint("hunter", path=src)
    assert out["ok"] is False and "--migrate" in out["why"], out["why"]


def test_the_refusals_that_used_to_be_dead_ends_point_at_the_recovery(rig,
                                                                      tmp_path):
    """THE PART THAT MADE IT A LOCKOUT: both doors said no and neither said
    where the third one was."""
    v, enc, gate = rig
    world = Voices(seed=13, apart=0.3)
    pool = _drifted(v, enc, world)
    src = _write_voiceprint(tmp_path / "vp.npz", pool)
    out = v.gallery.migrate_voiceprint("hunter", path=src)
    assert out["ok"] is False and "--reanchor" in out["why"], out["why"]

    ve = _voice_enrol()
    ok, why = ve.pool_ok(v.gallery, "hunter", world.takes("hunter", 8),
                         owner="hunter", owner_vectors=pool)
    assert ok is False and "--reanchor" in why, why


def test_the_script_offers_the_recovery_without_a_microphone(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    """It has to be a COMMAND he can run, not a method a test can call."""
    ve = _voice_enrol()
    world = Voices(seed=17, apart=0.3)
    him = world.takes("hunter", 14)
    root = tmp_path / "vg"
    g = vg.VoiceGallery(root=root)
    for e in him:
        g.add("hunter", e, src="legacy")
    g.set_consent("hunter", "owner")
    g.save(reason="migrated")
    drifted = him + world.takes("hunter", 24)
    src = _write_voiceprint(tmp_path / "voiceprint.npz", drifted)
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", root)
    monkeypatch.setattr(vg.PATHS, "VOICEPRINT", src)
    assert ve.main(["--reanchor", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "re-anchor" in out.lower()
    g2 = vg.VoiceGallery(root=root)
    g2.load()
    assert g2.count("hunter") == len(drifted)


def test_the_recovery_takes_no_label_but_his(tmp_path, monkeypatch, capsys):
    """The same rule ``--migrate`` lives under: the voiceprint has no label
    in it, and the only one it may ever be filed under is his."""
    ve = _voice_enrol()
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", tmp_path / "vg")
    assert ve.main(["--reanchor", "--label", "mara"]) == 2
    assert "mara" in capsys.readouterr().err


def test_a_failed_recovery_leaves_his_pool_exactly_as_it_was(rig, tmp_path,
                                                             monkeypatch):
    """THE STAKES ARE HIGHER HERE THAN IN A MIGRATION. A re-anchor EMPTIES
    his label before it refills it, so an exception escaping between the two
    leaves him holding part of one pool and part of another -- the same class
    of loss ``migrate_voiceprint`` learned to roll back from on 2026-09-05,
    with the pool already gone rather than merely staged."""
    v, enc, gate = rig
    world = Voices(seed=23, apart=0.3)
    pool = _drifted(v, enc, world)
    src = _write_voiceprint(tmp_path / "vp.npz", pool)
    before = [np.array(e, copy=True) for e in v.gallery.embeddings("hunter")]
    # HALF WAY THROUGH THE REFILL is the case the try has to cover, and it is
    # the one an except around save() alone would miss: the old pool is
    # already gone and the new one is not there yet.
    real_add = v.gallery.add
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise ValueError("refusing a degenerate embedding: pretend")
        return real_add(*a, **k)

    monkeypatch.setattr(v.gallery, "add", boom)
    out = v.gallery.reanchor_voiceprint("hunter", path=src)
    assert out["ok"] is False and out["why"], out
    after = v.gallery.embeddings("hunter")
    assert len(after) == len(before) == 14, (
        "his pool came back holding %d of the 14 it had" % len(after))
    assert all(np.array_equal(a, b) for a, b in zip(after, before)), \
        "his pool came back different from the one that was there"
    monkeypatch.setattr(v.gallery, "add", real_add)
    assert v.gallery.reanchor_voiceprint("hunter", path=src)["ok"] is True
    assert v.gallery.count("hunter") == len(pool)
