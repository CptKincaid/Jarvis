"""ONE MAN, TWO GALLERY LABELS -- and the margin asked about his own noise.

THE BLOCKER THE ROUND-3 REVIEW MEASURED AND THIS FILE CLOSES. With his voice
under TWO gallery labels and nobody else in the room, ``identify`` ranked the
two of them against each other, the gap between two pools of the same man
measured ~0.00 against a 0.20 bar, ``near_miss`` fired every turn, and
``gate._voice_leg``'s near-miss branch answered "that is nobody, not the
owner". Measured by the review at 100 of 100 refusals in BOTH variants -- an
exact copy of his pool (alias 1.0000) and a genuine second enrolment of the
same man (alias 0.9189) -- and identical on d41c17d, so it is older than this
round.

IT IS REACHABLE WITH THE SUPPORTED COMMANDS. ``migrate_voiceprint``'s "already
has embeddings" guard is per-LABEL, so a second ``--migrate`` under a
DIFFERENT label succeeds. That is what happens the day
``identity.owner_label(cfg)`` changes: he edits his name in assistant.json,
re-runs ``--migrate``, and the old slug's label stays on disk.

WHY THE FIX IS A DESIGN CHANGE AND NOT A CONDITION. ``MARGIN`` is derived
from splitting HIS OWN pool into two pretend people -- it is the floor on what
his own within-person noise can produce. Two labels that are the same person
produce a gap that is BY CONSTRUCTION within-person noise, so ranking them
against each other asks the margin the one question it was derived to answer
NO to. The gallery is arithmetic and cannot know who the owner is; the caller
that does (``speaker._owner_pools``, which MEASURES rather than spells) now
hands ``identify`` the grouping, and one person occupies ONE row of the
ranking before either bar is applied.

Every number here is synthetic (tests/synthvoice.py). No microphone, no
recording, no real voiceprint, no camera.
"""
from __future__ import annotations


import pytest

from jarvis import speaker as sp
from jarvis import voicegallery as vg
from tests.synthvoice import Voices, cos
from tests.test_voice_multispeaker_wiring import _clip
from tests.test_voice_owner_lockout import ROW, SLUG, _gate, _verifier


def _both_labels(v, world, *, copy=True, seed2=777):
    """His voiceprint, plus TWO gallery labels that are both him -- built the
    way the SUPPORTED commands build them, provenance stamp and all.

    ``copy`` True is the plain second ``--migrate``: the same fourteen vectors
    under the old slug and the new one (alias 1.0000). That is what happens
    the day ``identity.owner_label(cfg)`` changes -- he edits his name in
    assistant.json and re-runs ``--migrate``.

    ``copy`` False adds a re-enrolment in the middle: ``--migrate`` as the old
    slug, ``enroll_voice.py --reset`` (a genuinely new voiceprint of the same
    man), then rename and ``--migrate`` again. The old label is then STALE --
    it measures ~0.92 against the current voiceprint, under the 0.98 line --
    which is the review's second variant and the one no cosine can fold.
    """
    him = world.takes("hunter", 14)
    _carry(v.gallery, SLUG, him)
    if copy:
        second = list(him)
    else:
        w2 = Voices(seed=seed2, apart=world.apart)
        w2.shared = world.shared
        w2._identity["hunter"] = world.identity("hunter")
        second = w2.takes("hunter", 14)
    _carry(v.gallery, ROW, second)
    # voiceprint.npz is whatever the LAST enrolment wrote.
    v._embeddings = list(second)
    v._recompute_centroid()
    return him


def _carry(gallery, label, vectors):
    """``--migrate``'s write: the voiceprint's vectors under one label, with
    the provenance stamp both writers put on every take."""
    for e in vectors:
        gallery.add(label, e, src=vg.VOICEPRINT_SRC,
                    note="migrated " + vg.VOICEPRINT_NOTE)
    gallery.set_consent(label, "owner")


def _alias(v):
    """cos between the two labels' centroids -- the review's 1.0000 / 0.9189."""
    return cos(v.gallery.centroid(SLUG), v.gallery.centroid(ROW))


def _turns(v, gate, n=100, base=60.0):
    """``(admitted, near_miss_count)`` over ``n`` of HIS OWN 4 s turns."""
    world = v._world
    ok = misses = 0
    for i in range(n):
        c = _clip(4.0, base + i * 0.01)
        v._extract_embedding.teach(c, world.take("hunter"))
        out, stats = v.filter_segments(c)
        misses += bool(stats.get("near_miss"))
        d = gate.judge("voice", "read my mail", stats=stats,
                       rejected=out is None)
        ok += bool(d.admit and d.who == ROW)
    return ok, misses


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    v, enc = _verifier(tmp_path, monkeypatch, owner_label=SLUG)
    v._extract_embedding = enc
    return v, enc, _gate(tmp_path, owner=SLUG)


# ============================================== 1. the blocker, both variants
@pytest.mark.parametrize("copy,ident", [(True, "second-migrate-copy"),
                                        (False, "second-enrolment")],
                         ids=["copy", "fresh"])
@pytest.mark.parametrize("apart", [0.3, 1.0])
def test_two_labels_of_his_own_voice_never_refuse_him(rig, copy, ident, apart):
    """100 of HIS turns, alone in the room, his voice under two labels.

    Measured before the fix: 0 of 100 admitted, near_miss 100 of 100, at both
    separations and in both variants. The margin was being asked to separate
    a man from himself.
    """
    v, enc, gate = rig
    world = Voices(seed=303, apart=apart)
    v._world = world
    _both_labels(v, world, copy=copy)
    alias = _alias(v)
    ok, misses = _turns(v, gate)
    print("MEASURED two-labels %s apart=%.1f: alias=%.4f admitted=%d/100 "
          "near_miss=%d/100" % (ident, apart, alias, ok, misses))
    assert v.gallery.voiceprint_labels() == (ROW, SLUG), \
        v.gallery.voiceprint_labels()
    assert misses == 0, "the margin fired between two pools of one man"
    assert ok == 100, "he was refused his own turn %d time(s)" % (100 - ok)


def test_one_man_under_two_labels_occupies_one_row_of_the_ranking(rig):
    """The invariant under the count, not just the count. identify() must
    return ONE row per PERSON, so ``margin`` is always between people."""
    v, enc, gate = rig
    world = Voices(seed=304, apart=0.3)
    v._world = world
    _both_labels(v, world, copy=False)
    c = _clip(4.0, 91.0)
    enc.teach(c, world.take("hunter"))
    out, stats = v.filter_segments(c)
    scored = set(stats["who_scores"])
    assert len(scored & {SLUG, ROW}) == 1, scored
    assert stats["who"] in ("", SLUG, ROW)
    assert not stats["near_miss"]


def test_a_real_guest_is_still_ranked_against_him(rig):
    """The fold must not swallow anybody else: with a guest enrolled the
    margin still applies BETWEEN the two people."""
    v, enc, gate = rig
    world = Voices(seed=305, apart=0.3)
    v._world = world
    _both_labels(v, world, copy=False)
    for e in world.takes("mara", 10):
        v.gallery.add("mara", e)
    c = _clip(4.0, 92.0)
    enc.teach(c, world.take("hunter"))
    out, stats = v.filter_segments(c)
    assert "mara" in stats["who_scores"], stats["who_scores"]
    assert len(stats["who_scores"]) == 2, stats["who_scores"]
    ok, misses = _turns(v, gate, n=50, base=120.0)
    print("MEASURED two-labels+guest apart=0.3: admitted=%d/50 near_miss=%d"
          % (ok, misses))
    assert ok == 50


# ================================== 2. the route in: a second --migrate
def test_a_second_migrate_under_a_new_label_is_refused(tmp_path):
    """The layout above must not be buildable. The guard was per-LABEL; the
    owner's pool is a MEASURED fact, so the guard is now per-POOL: any label
    that already measures as voiceprint.npz blocks a second carry, and the
    refusal names the way out."""
    from tests.test_voice_migration import _voiceprint
    src = tmp_path / "voiceprint.npz"
    _voiceprint(src, Voices(seed=306).takes("hunter", 14))
    g = vg.VoiceGallery(root=tmp_path / "gallery")
    first = g.migrate_voiceprint(SLUG, path=src, reason="t")
    assert first["ok"], first["why"]
    second = g.migrate_voiceprint(ROW, path=src, reason="t")
    assert not second["ok"], "a second label of his voice was written"
    assert SLUG in second["why"]
    assert "--delete" in second["why"], second["why"]
    assert ROW not in g.disk_labels()


# ============================== 3. the fold is a rule about labels, not scores
def test_the_fold_is_pure_and_keeps_the_best_row_of_each_group():
    """``_fold_same`` decides on GROUP MEMBERSHIP alone. Nothing in it looks
    at how close two cosines are -- that would be the gallery quietly folding
    two PEOPLE who happen to sit near each other, which is the escalation this
    lane spent round 3 closing."""
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.1)]
    out, dropped = vg._fold_same(ranked, [{"a", "c"}])
    assert out == [("a", 0.9), ("b", 0.8), ("d", 0.1)]
    assert dropped == [("c", 0.7)]
    # the survivor is the group's best row, wherever it sits in the ranking
    out, dropped = vg._fold_same(ranked, [{"b", "d"}])
    assert out == [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    # a one-label group, an empty group and no groups at all change nothing
    for same in ((), [set()], [{"a"}], None):
        assert vg._fold_same(ranked, same)[0] == ranked
    # NOTHING IS AVERAGED AND NO SCORE IS RAISED: every surviving row is a row
    # that was already there.
    assert set(vg._fold_same(ranked, [{"a", "b", "c", "d"}])[0]) <= set(ranked)


def test_the_fold_never_invents_a_score_for_a_label(rig):
    """The invariant the round-2 fix established survives the fold: ``score``
    is the score OF ``top_label``, and ``who`` is never a label absent from
    ``who_scores``."""
    v, enc, gate = rig
    world = Voices(seed=307, apart=0.3)
    v._world = world
    _both_labels(v, world, copy=False)
    for e in world.takes("mara", 10):
        v.gallery.add("mara", e, src="enrol")
    for i in range(60):
        c = _clip(4.0, 200.0 + i * 0.01)
        enc.teach(c, world.take("hunter" if i % 2 else "mara"))
        _out, stats = v.filter_segments(c)
        scores = stats["who_scores"]
        assert stats["who"] in ("",) or stats["who"] in scores, stats
        assert stats["top"] in ("",) or stats["top"] in scores, stats
        if stats["top"]:
            assert scores[stats["top"]] >= v.threshold, stats


def test_a_gallery_used_alone_folds_nobody(tmp_path):
    """``identify`` with no ``same`` is unchanged: a store used on its own
    cannot silently decide two labels are one person."""
    g = vg.VoiceGallery(root=tmp_path / "g")
    world = Voices(seed=308, apart=0.3)
    him = world.takes("hunter", 14)
    for e in him:
        g.add(SLUG, e)
        g.add(ROW, e)
    v = g.identify(world.take("hunter"), 4.0)
    assert set(dict(v.scores)) == {SLUG, ROW}
    assert v.who == "" and v.margin is not None and v.margin < vg.MARGIN


# ================= 4. provenance does not weaken the impostor guard
def _hers(v, enc, gate, probes, base=400.0):
    """``probes`` is a LIST OF VECTORS, not a world: the paired control is
    only a control if both arms judge the identical embeddings. A fresh
    ``Voices`` at the same seed does NOT reproduce them -- it re-draws its
    identity directions in call order, so "mara" in a fresh world is whoever
    was asked for first in the old one."""
    ok = 0
    for i, pv in enumerate(probes):
        c = _clip(4.0, base + i * 0.01)
        enc.teach(c, pv)
        out, stats = v.filter_segments(c)
        d = gate.judge("voice", "read my mail", stats=stats,
                       rejected=out is None)
        ok += bool(d.admit and d.who == ROW)
    return ok


@pytest.mark.parametrize("apart", [0.3, 1.0])
def test_microphone_takes_under_his_name_buy_her_nothing(rig, tmp_path,
                                                         monkeypatch, apart):
    """THE GUARD THE FOLD MUST NOT COST, measured as a PAIRED CONTROL rather
    than as a rate.

    ``_owner_pools`` now has a second route in: a pool CARRIED OUT OF
    voiceprint.npz is his whatever it measures. Takes somebody recorded under
    his name AT THE MICROPHONE carry ``src="enrol"`` and can never acquire
    that stamp, so the 2026-09-05 "his name is not a credential" guard is
    untouched.

    THE RATE ALONE WOULD LIE HERE. At apart 1.0 her voice clears HIS
    VOICEPRINT's own bar -- the single-speaker false-accept this feature
    predates and does not touch -- so an absolute number would read 74 of 100
    and blame the gallery for it. The question the guard answers is what her
    label bought her, and the answer must be exactly zero: the same probe
    vectors judged with her takes under his label, and against an empty
    gallery.
    """
    v, enc, gate = rig
    world = Voices(seed=309, apart=apart)
    him = world.takes("hunter", 14)
    hers = world.takes("mara", 14)
    probes = [world.take("mara") for _ in range(100)]
    v._embeddings = list(him)
    v._recompute_centroid()
    control = _hers(v, enc, gate, probes)

    v2, enc2 = _verifier(tmp_path / "arm2", monkeypatch, owner_label=SLUG)
    v2._extract_embedding = enc2
    v2._embeddings = list(him)
    v2._recompute_centroid()
    for e in hers:                       # SHE records under HIS label
        v2.gallery.add(SLUG, e, src="enrol", note="Speak normally.")
    v2.gallery.set_consent(SLUG, "owner")
    assert not v2.gallery.carried_from_voiceprint(SLUG)
    assert v2._disowned(v2._all_centroids()) == frozenset({SLUG})
    assert SLUG not in v2._owner_pools(v2._all_centroids(matchable=True))
    attacked = _hers(v2, enc2, _gate(tmp_path / "arm2", owner=SLUG), probes)
    print("MEASURED impostor-under-his-label apart=%.1f: empty-gallery=%d/100 "
          "her-label-present=%d/100 bought=%+d"
          % (apart, control, attacked, attacked - control))
    assert attacked <= control, ("her takes under his name bought her %d "
                                 "admissions" % (attacked - control))
    if apart == 0.3:
        assert attacked == 0 and control == 0


def test_one_microphone_take_in_a_carried_pool_ends_the_attestation(tmp_path):
    """Provenance is ALL-OR-NOTHING on the enrolment takes. Mixing one mic
    take into a migrated pool must not launder the lot."""
    g = vg.VoiceGallery(root=tmp_path / "g")
    world = Voices(seed=310, apart=0.3)
    for e in world.takes("hunter", 14):
        g.add(SLUG, e, src=vg.VOICEPRINT_SRC, note="migrated " + vg.VOICEPRINT_NOTE)
    assert g.carried_from_voiceprint(SLUG)
    g.add(SLUG, world.take("mara"), src="enrol", note="Speak normally.")
    assert not g.carried_from_voiceprint(SLUG)
    assert g.voiceprint_labels() == ()


def test_a_passive_take_does_not_end_the_attestation(tmp_path):
    """A pool that started as his voiceprint and has learned a little is
    still his: passive takes are the pool teaching itself, not a second
    person at the microphone."""
    g = vg.VoiceGallery(root=tmp_path / "g")
    world = Voices(seed=311, apart=0.3)
    for e in world.takes("hunter", 14):
        g.add(SLUG, e, src=vg.VOICEPRINT_SRC, note="migrated " + vg.VOICEPRINT_NOTE)
    g.add(SLUG, world.take("hunter"), src="passive")
    assert g.carried_from_voiceprint(SLUG)


def test_a_gallery_that_raises_folds_nobody(rig):
    """``_carried_labels`` is consulted on every match. A store that raises
    there must not take the door with it."""
    v, enc, gate = rig
    world = Voices(seed=312, apart=0.3)
    him = world.takes("hunter", 14)
    v._embeddings = list(him)
    v._recompute_centroid()
    for e in him:
        v.gallery.add(SLUG, e, src=vg.VOICEPRINT_SRC,
                      note="migrated " + vg.VOICEPRINT_NOTE)

    def boom():
        raise RuntimeError("wedged")

    v.gallery.voiceprint_labels = boom
    assert v._carried_labels() == frozenset()
    c = _clip(4.0, 500.0)
    enc.teach(c, world.take("hunter"))
    out, stats = v.filter_segments(c)
    d = gate.judge("voice", "read my mail", stats=stats, rejected=out is None)
    assert d.admit and d.who == ROW, "a wedged provenance read refused him"


# ============ 5. the abstention flag, which was not literally true (C5)
def _paths(v, enc, world, tmp_path):
    """The three stats dicts the review found carrying abstained=False with
    nothing measured, plus the honest one, all from the real code paths."""
    out = {}
    c = _clip(4.0, 600.0)
    enc.teach(c, world.take("hunter"))

    # 1. THE ENROLMENT-GAP FAIL-OPEN: nobody enrolled at all.
    empty = sp.SpeakerVerifier(owner_label=SLUG)
    empty._model_loaded = True
    empty._ensure_model = lambda: True
    empty._extract_embedding = enc
    empty.gallery = vg.VoiceGallery(root=tmp_path / "empty")
    out["gap"] = empty.filter_segments(c)[1]

    # 2. THE MODEL-NOT-LOADED FAIL-SHUT.
    dead = sp.SpeakerVerifier(owner_label=SLUG)
    dead._embeddings = list(v._embeddings)
    dead._recompute_centroid()
    dead._ensure_model = lambda: False
    dead.gallery = v.gallery
    out["no_model"] = dead.filter_segments(c)[1]

    # 3. THE GALLERY FAULT: identify() raises while centroids() still works.
    def boom(*a, **kw):
        raise RuntimeError("wedged")

    v.gallery.identify = boom
    out["fault"] = v.filter_segments(c)[1]
    return out


def test_every_path_that_measured_nothing_says_so(rig, tmp_path):
    """C5. ``abstained`` claimed to be carried through EVERY path and was
    not: three of them said "measured, and named nobody" about a clip nothing
    had looked at. None can admit anybody -- they reach the gate as matched=0
    or as a fault, both refusals -- so this is the flag telling the truth
    rather than a lock. A flag that lies about the easy cases is one nobody
    can trust about the hard one."""
    v, enc, gate = rig
    world = Voices(seed=320, apart=0.3)
    v._world = world
    _both_labels(v, world)
    got = _paths(v, enc, world, tmp_path)
    for name, stats in got.items():
        assert stats["abstained"] is True, "%s carries abstained=False" % name
    assert got["fault"]["who_fault"]
    assert got["gap"]["matched"] == 0 and got["no_model"]["matched"] == 0


def test_the_honest_flag_still_admits_and_refuses_exactly_as_before(rig,
                                                                    tmp_path):
    """The three paths must not have become admissions. Whatever the flag
    now says, the gate's answer on each is unchanged: nobody."""
    v, enc, gate = rig
    world = Voices(seed=321, apart=0.3)
    v._world = world
    _both_labels(v, world)
    got = _paths(v, enc, world, tmp_path)
    for name, stats in got.items():
        d = gate.judge("voice", "read my mail", stats=stats, rejected=False)
        assert not (d.admit and d.who == ROW), \
            "%s admitted him on the strength of nothing" % name


def test_a_short_clip_still_fails_open_for_him(rig):
    """The documented fail-open the flag exists for is untouched: every
    "Yes." he says is under 1.5 s of speech."""
    v, enc, gate = rig
    world = Voices(seed=322, apart=0.3)
    v._world = world
    _both_labels(v, world)
    ok = 0
    for i in range(50):
        c = _clip(1.0, 700.0 + i * 0.01)
        enc.teach(c, world.take("hunter"))
        out, stats = v.filter_segments(c)
        assert stats["abstained"] is True
        d = gate.judge("voice", "yes", stats=stats, rejected=out is None)
        ok += bool(d.admit and d.who == ROW)
    print("MEASURED short-speech-two-labels: admitted=%d/50" % ok)
    assert ok == 50


# ===================== 6. the rename, end to end, and the two other holes
def _renamed(v, world, seed2=777):
    """What a rename leaves on disk: he migrated as the OLD slug, re-recorded
    his voiceprint, renamed himself in assistant.json, and migrated again. The
    old slug's label stays, and it is no longer what the config spells."""
    old = world.takes("hunter", 14)
    _carry(v.gallery, SLUG, old)
    w2 = Voices(seed=seed2, apart=world.apart)
    w2.shared = world.shared
    w2._identity["hunter"] = world.identity("hunter")
    new = w2.takes("hunter", 14)
    _carry(v.gallery, ROW, new)
    v._embeddings = list(new)
    v._recompute_centroid()
    return old, new


@pytest.mark.parametrize("apart", [0.3, 1.0])
def test_the_rename_that_left_a_third_name_behind(tmp_path, monkeypatch, apart):
    """THE REVIEW'S OWN SHAPE, and it needed TWO fixes rather than one.

    ``identity.owner_label(cfg)`` changed, so the label the config spells is
    the NEW one and the old slug stays on disk beside it. Measured on 9964f66:
    alias 0.9201 / 0.9194, near_miss 100 of 100, admitted 0 of 100 at both
    separations.

    The fold alone got that to 45 of 100, not 100 -- because when the OLD
    slug's row won the fold, it wore a name no registry had ever heard of and
    ``recognise`` dropped it. ``gate._voice_leg`` now reads the POOL rather
    than the spelling: a name on the owner's own pool is the owner, and
    ``speaker._reconcile`` guarantees a guest's label can never arrive on it.
    """
    v, enc = _verifier(tmp_path, monkeypatch, owner_label=ROW)
    v._extract_embedding = enc
    v._world = world = Voices(seed=303, apart=apart)
    _renamed(v, world)
    gate = _gate(tmp_path, owner=ROW)
    ok, misses = _turns(v, gate, base=1700.0)
    print("MEASURED renamed-and-remigrated apart=%.1f: admitted=%d/100 "
          "near_miss=%d/100" % (apart, ok, misses))
    assert misses == 0 and ok == 100


def test_a_stale_carried_pool_no_longer_costs_him_his_turns(tmp_path,
                                                             monkeypatch):
    """C1-SECOND. ``enroll_voice.py --reset`` after ``--migrate``: his gallery
    pool measures 0.938 of the new voiceprint, under the 0.98 line.

    Measured on 9964f66 at apart 1.0 with a guest enrolled: 10 of 100 of his
    own turns admitted, because ``_disowned`` dropped his own migrated pool
    out of the matchable set. A pool CARRIED OUT OF voiceprint.npz is his
    whatever it measures -- the impostor shape is microphone takes, which
    carry src="enrol" -- so it is read as his: 99 of 100 here. The anchor is
    still off and --status still says so; what it costs now is stalled
    passive learning, not his own machine.
    """
    v, enc = _verifier(tmp_path, monkeypatch, owner_label=SLUG)
    v._extract_embedding = enc
    v._world = world = Voices(seed=341, apart=1.0)
    _carry(v.gallery, SLUG, world.takes("hunter", 14))
    for e in world.takes("mara", 10):
        v.gallery.add("mara", e, src="enrol")
    later = Voices(seed=342, apart=1.0)
    later.shared = world.shared
    later._identity["hunter"] = world.identity("hunter")
    v._embeddings = later.takes("hunter", 14)      # he re-recorded it
    v._recompute_centroid()
    anchor = cos(v.gallery.centroid(SLUG), v._centroid)
    ok, _m = _turns(v, _gate(tmp_path, owner=SLUG), base=1300.0)
    print("MEASURED re-enrolled-voiceprint apart=1.0: anchor=%.4f admitted=%d/100"
          % (anchor, ok))
    assert anchor < vg.OWNER_POOL_COSINE, "the fixture did not go stale"
    assert ok >= 99


@pytest.mark.parametrize("apart", [0.7, 1.0])
def test_deleting_his_pool_is_what_the_guard_is_protecting(tmp_path,
                                                            monkeypatch, apart):
    """C1-THIRD, with the number the guard exists for. ``--delete --label
    <his own label>`` reaches the layout ``owner_ready`` refuses to BUILD.
    Nothing in the runtime can fix that -- the gallery genuinely cannot rank
    him against her once his pool is gone -- so the fix is the script refusing,
    and this records what it is refusing on his behalf."""
    v, enc = _verifier(tmp_path, monkeypatch, owner_label=SLUG)
    v._extract_embedding = enc
    v._world = world = Voices(seed=331, apart=apart)
    him = world.takes("hunter", 14)
    _carry(v.gallery, SLUG, him)
    for e in world.takes("mara", 10):
        v.gallery.add("mara", e, src="enrol")
    v._embeddings = list(him)
    v._recompute_centroid()
    gate = _gate(tmp_path, owner=SLUG)
    with_pool, _m = _turns(v, gate, base=900.0)
    v.gallery.forget(SLUG)
    without, _m = _turns(v, gate, base=1100.0)
    print("MEASURED delete-his-label apart=%.1f: with-his-pool=%d/100 "
          "after-delete=%d/100" % (apart, with_pool, without))
    assert with_pool >= 99
    assert without < with_pool, ("the layout the guard refuses is no longer "
                                 "worse for him; re-derive the guard")


def test_a_guests_name_can_never_arrive_on_the_owners_pool(tmp_path):
    """THE GUARD ON THE GATE'S NEW RULE. "A name on the owner's own pool is
    the owner" is only safe because ``speaker._reconcile`` withholds any name
    whose pool disagrees with the matched one -- so ``who=<guest>`` beside
    ``matched_label=""`` is a combination the pipeline cannot produce.

    Asserted from BOTH ends: the gate would indeed read such a dict as him
    (which is why it must be unbuildable), and ``_reconcile`` is what makes it
    unbuildable.
    """
    gate = _gate(tmp_path, owner=SLUG)
    forged = {"total": 1, "matched": 1, "scores": [0.7], "who": "mara",
              "who_scores": {"mara": 0.7}, "labels": ("mara",),
              "matched_label": "", "top": "mara", "provisional": "",
              "near_miss": False, "abstained": False, "who_fault": "",
              "who_is_owner": True}
    assert gate._voice_leg(forged, False) == (ROW, True), (
        "the rule is stated so its precondition can be checked, not hidden")
    # WITHOUT THE KEY NOTHING CHANGES: a stats dict that does not carry the
    # matching layer's attestation is read exactly as it was before.
    assert gate._voice_leg(dict(forged, who_is_owner=False), False) == \
        ("mara", True)

    # ...and the precondition. _reconcile takes the name off any verdict whose
    # label belongs to a pool other than the one the bar was cleared on.
    import dataclasses
    v = sp.SpeakerVerifier(owner_label=SLUG)
    v.gallery = vg.VoiceGallery(root=tmp_path / "g")
    world = Voices(seed=360, apart=0.3)
    _carry(v.gallery, SLUG, world.takes("hunter", 14))
    for e in world.takes("mara", 10):
        v.gallery.add("mara", e, src="enrol")
    v._embeddings = list(v.gallery.embeddings(SLUG))
    v._recompute_centroid()
    named_her = vg.VoiceVerdict(who="mara", scores=(("mara", 0.7),))
    assert v._reconcile(named_her, "").who == "", (
        "a guest's name survived on the owner's pool"
    )
    assert v._reconcile(named_her, "mara").who == "mara"
    # and his own label on his own pool is untouched
    named_him = vg.VoiceVerdict(who=SLUG, scores=((SLUG, 0.7),))
    assert v._reconcile(named_him, "").who == SLUG
    assert dataclasses.is_dataclass(named_him)
