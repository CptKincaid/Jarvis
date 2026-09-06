"""The owner fallback, proven AT THE GATE, and the two routes that turned it
into a privilege escalation.

THE FINDING (2026-09-04 ~02:00). ``speaker._best_score`` is a MAXIMUM over
every enrolled centroid, so a second person cleared the 0.30 bar on her OWN
centroid and set ``matched=1``; ``identify`` then withheld her name (six
takes is provisional, or the margin failed), and ``gate._voice_leg`` turned
the nameless match into the owner. Measured: 150 of 150 of her clips admitted
as him, at a centroid cosine of 0.246 to his pool. The existing test asserted
``stats["who"] == ""`` -- one layer too low, since "" was exactly the value
that became his name.

TWO FIXES, EACH WITH A TEST THAT FAILS WHEN IT IS REVERTED:

1. A provisional label cannot match as anybody (``speaker._all_centroids``
   leaves it out of the matchable set). It still scores and logs.
2. A nameless match is the owner ONLY when it was measured on HIS pool
   (``matched_label`` "" or his label) and the gallery's best guess above
   the bar (``top``) is not somebody else. The first version of this fix
   counted labels instead (one -> him, two -> nobody); the round-2 review
   measured that wrong both ways -- a guest's short window on a one-label
   box minted him, and his own short window on a two-label box was refused
   -- and tests/test_voice_per_window.py holds those fixtures now.

Two further things the gate must be able to tell apart, and now can: an
ABSTENTION (nothing measured -- the documented fail-open, unchanged by
enrolling anybody) and a FAULT (the gallery raised -- no instrument, not a
negative, not a name).

Synthetic vectors only. No microphone, no recording, no real voiceprint.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import gate as gt
from jarvis import identity as ident
from jarvis import speaker as sp
from jarvis import voicegallery as vg
from tests.synthvoice import Voices, centroid, cos
from tests.test_voice_multispeaker_wiring import (_FakeEncoder, _clip, _enrol,
                                                 _gate)


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "VOICEPRINT_FILE", tmp_path / "voiceprint.npz")
    enc = _FakeEncoder()
    v = sp.SpeakerVerifier()
    v._model_loaded = True
    monkeypatch.setattr(v, "_ensure_model", lambda: True)
    monkeypatch.setattr(v, "_extract_embedding", enc)
    v.gallery = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    gate = _gate(tmp_path)
    return v, enc, gate


def _stats(**kw):
    base = {"total": 1, "matched": 1, "scores": [0.41], "who": "",
            "who_scores": {}, "labels": (), "abstained": False,
            "who_fault": "", "matched_label": "", "top": "",
            "provisional": "", "near_miss": False}
    base.update(kw)
    return base


def _old_rule_admits_as_owner(v, emb):
    """What the code did before the fix: max over EVERY centroid, and a
    nameless match is the owner. Kept so the fixture is shown to reproduce
    the route, not merely to pass."""
    best = max(cos(emb, c) for c in v._all_centroids().values())
    who = v.gallery.identify(emb, 4.0, v.threshold).who
    return best >= v.threshold and not who


# --------------------------------------------- 1. the finding, replicated
@pytest.mark.parametrize("migrated", [False, True],
                         ids=["voiceprint+mara", "voiceprint+hunter+mara"])
def test_a_second_person_at_six_takes_is_never_admitted_as_the_owner(
        rig, migrated):
    """150 of her clips through filter_segments and then the gate, in
    ENFORCE. Before the fix every one of them was him."""
    v, enc, gate = rig
    world = Voices(seed=31, apart=0.3)
    him = world.takes("hunter", 14)
    v._embeddings = list(him)
    v._recompute_centroid()
    if migrated:
        for e in him:
            v.gallery.add("hunter", e)
    _enrol(v.gallery, world, "mara", 6)
    assert v.gallery.provisional("mara")
    gap = cos(centroid(him), v.gallery.centroid("mara"))
    assert gap < v.threshold, "fixture drifted: her centroid is inside his bar"

    as_owner = before = 0
    for i in range(150):
        emb = world.take("mara")
        clip = _clip(4.0, 10.0 + i * 0.01)
        enc.teach(clip, emb)
        before += _old_rule_admits_as_owner(v, emb)
        out, stats = v.filter_segments(clip)
        d = gate.judge("voice", "unlock the door", stats=stats,
                       rejected=out is None)
        as_owner += (d.admit and d.who == "hunter")
    assert before >= 140, ("the fixture no longer reproduces the route: "
                           "only %d of 150 would have been him" % before)
    assert as_owner == 0, "%d of 150 of her clips were admitted as him" % as_owner


def test_the_route_is_closed_at_the_gate_not_only_below_it(rig):
    """The layer the old test missed. Her clip's stats carry matched>=1 and
    who="" when her label has enough takes but the margin fails -- and THAT
    dict, judged, is nobody."""
    v, enc, gate = rig
    world = Voices(seed=32, apart=3.0)          # confusable on purpose
    _enrol(v.gallery, world, "hunter", 14)
    _enrol(v.gallery, world, "mara", 12)
    clip = _clip(4.0, 0.51)
    enc.teach(clip, world.take("mara"))
    out, stats = v.filter_segments(clip)
    if stats["matched"] < 1:
        pytest.skip("this fixture's mara did not clear the bar")
    assert out is not None and stats["who"] == ""
    d = gate.judge("voice", "unlock the door", stats=stats)
    assert d.who != "hunter"
    assert d.admit is False and d.how == gt.HOW_NOBODY


# --------------------------------------------- 2. the rule, at the gate
def test_a_nameless_match_on_her_pool_is_nobody(tmp_path):
    """Her window cleared the bar on HER centroid and identify() could not
    name her (too little speech). Whose pool matched is the fact; the
    number of labels is not."""
    d = _gate(tmp_path).judge("voice", "unlock the door",
                              stats=_stats(labels=("hunter", "mara"),
                                           matched_label="mara",
                                           abstained=True))
    assert d.who == "" and d.admit is False and d.how == gt.HOW_NOBODY


def test_a_nameless_match_on_her_pool_names_the_way_back_in(tmp_path):
    d = _gate(tmp_path).judge("voice", "unlock the door",
                              stats=_stats(labels=("hunter", "mara"),
                                           matched_label="mara"))
    assert d.line == gt.UNKNOWN_LINE


@pytest.mark.parametrize("labels", [(), ("hunter",), ("mara",),
                                    ("hunter", "mara"), ("heather", "mara")],
                         ids=["no-gallery", "his-label", "one-guest",
                              "him-and-a-guest", "two-guests-unmigrated"])
def test_a_match_on_his_pool_is_him_however_many_labels(tmp_path, labels):
    """THE LOCKOUT THE COUNT RULE CAUSED. A nameless match on the
    voiceprint is his whether the gallery holds nobody, him, one guest or
    two guests he never migrated beside: nobody else enrolling can lock him
    out of his own voiceprint."""
    d = _gate(tmp_path).judge("voice", "what's the time",
                              stats=_stats(labels=labels, matched_label=""))
    assert d.admit is True and d.who == "hunter"


def test_his_own_migrated_label_is_his_pool(tmp_path):
    d = _gate(tmp_path).judge("voice", "what's the time",
                              stats=_stats(labels=("hunter", "mara"),
                                           matched_label="hunter",
                                           abstained=True))
    assert d.admit is True and d.who == "hunter"


@pytest.mark.parametrize("extra", [dict(provisional="mara"),
                                   dict(near_miss=True)],
                         ids=["probably-mara", "margin-failed-mara-on-top"])
def test_the_gallerys_best_guess_being_somebody_else_blocks_the_owner(
        tmp_path, extra):
    """identify() ranked HER first above the bar and withheld the name --
    for takes, or for the margin. "Probably Mara" may not be minted as
    him, whichever pool the window cleared the bar on."""
    d = _gate(tmp_path).judge("voice", "unlock the door",
                              stats=_stats(labels=("hunter", "mara"),
                                           matched_label="", top="mara",
                                           **extra))
    assert d.who != "hunter" and d.admit is False


def test_a_near_miss_is_refused_with_the_near_miss_line(tmp_path):
    d = _gate(tmp_path).judge("voice", "unlock the door",
                              stats=_stats(labels=("hunter", "mara"),
                                           matched_label="mara", top="mara",
                                           near_miss=True))
    assert d.admit is False and d.line == gt.NEAR_MISS_LINE
    assert "hunter" not in d.line.lower() and "mara" not in d.line.lower()


def test_a_near_miss_is_nobody_even_with_him_on_top(tmp_path):
    """THE HOLE ROUND 3 FOUND IN ITS OWN DRAFT. The margin failed, the
    gallery's best guess was the owner and the bar was cleared on his pool
    -- and the draft admitted it as him ("nothing says it is anybody
    else"). Measured on the reviewer's grid: a confusable guest minted as
    him 5/150 (10 takes) and 22/150 (6 takes) at apart 3.0 with him
    migrated, against 0/150 under the count rule. A failed margin is a coin
    flip and names nobody, whoever is on top."""
    d = _gate(tmp_path).judge("voice", "what's the time",
                              stats=_stats(labels=("hunter", "mara"),
                                           matched_label="", top="hunter",
                                           near_miss=True))
    assert d.admit is False and d.who == "" and d.line == gt.NEAR_MISS_LINE


def test_a_rejected_clip_never_gets_the_near_miss_line(tmp_path):
    """Two provisional labels can both sit above the gallery's bar on a
    clip that matched nobody; "I can hear someone I know" would be a small
    lie to a stranger. A rejection gets the ordinary unknown line."""
    d = _gate(tmp_path).judge("voice", "unlock the door", rejected=True,
                              stats=_stats(matched=0, scores=[0.1],
                                           labels=("heather", "mara"),
                                           top="mara", near_miss=True))
    assert d.admit is False and d.line == gt.UNKNOWN_LINE


def test_a_legacy_stats_dict_with_no_labels_key_is_still_him(tmp_path):
    """test_owner_gate.py's ABSTAINED dict has no identity keys at all.
    Nothing older than this feature may start refusing him."""
    d = _gate(tmp_path).judge("voice", "Yes.",
                              stats={"total": 1, "matched": 1, "scores": [0.0]})
    assert d.admit is True and d.who == "hunter"


def test_a_named_match_with_two_labels_is_that_person(tmp_path):
    d = _gate(tmp_path).judge("voice", "what's the weather",
                              stats=_stats(who="mara",
                                           labels=("hunter", "mara")))
    assert d.who == "mara" and d.role == ident.ROLE_KNOWN


def test_an_abstention_stays_him_with_two_labels(tmp_path):
    """THE COST, STATED. Nothing was measured on a clip this short, so the
    pipeline's documented fail-open stands and enrolling a second person
    moves nothing about it: whoever says a sub-1.5 s clip inside a follow-up
    window is answered as him, exactly as before the gallery existed. The
    alternative is refusing every "Yes." he says."""
    d = _gate(tmp_path).judge("voice", "Yes.",
                              stats=_stats(scores=[0.0], abstained=True,
                                           labels=("hunter", "mara")))
    assert d.admit is True and d.who == "hunter"


def test_shadow_mode_logs_the_would_refuse_and_admits(tmp_path):
    d = _gate(tmp_path, mode="shadow").judge(
        "voice", "unlock the door", stats=_stats(labels=("hunter", "mara"),
                                                 matched_label="mara"))
    assert d.admit is True and d.would_refuse is True and d.who == ""


# --------------------------------------------- 3. provisional cannot match
def test_a_provisional_label_alone_is_shut_on_the_transcript_and_open_on_wake(
        rig):
    """Somebody with six takes and nobody else: the transcript gate fails
    SHUT (somebody IS enrolled) and the wake gate reads None, its fail-open.
    Neither direction moved."""
    v, enc, _gate_ = rig
    world = Voices(seed=33, apart=0.3)
    _enrol(v.gallery, world, "mara", 6)
    assert v.is_enrolled is True
    clip = _clip(4.0, 0.61)
    enc.teach(clip, world.take("mara"))
    assert v.score(clip) is None
    out, stats = v.filter_segments(clip)
    assert out is None and stats["matched"] == 0
    assert stats["who"] == "" and "mara" in stats["who_scores"]


def test_a_provisional_label_moves_his_wake_score_by_nothing(rig):
    v, enc, _gate_ = rig
    world = Voices(seed=34, apart=0.3)
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()
    clip = _clip(2.0, 0.62)
    enc.teach(clip, world.take("hunter"))
    before = v.score(clip)
    _enrol(v.gallery, world, "mara", 7)
    assert v.score(clip) == before
    v.gallery.add("mara", world.take("mara"))          # the eighth
    assert not v.gallery.provisional("mara")
    assert v.score(clip) >= before


# --------------------------------------------- 4. a fault is not a non-match
class _RaisesOnIdentify(vg.VoiceGallery):
    def identify(self, *a, **kw):
        raise RuntimeError("wedged")


class _RaisesOnEverything:
    def labels(self):
        raise RuntimeError("wedged")

    def centroids(self):
        raise RuntimeError("wedged")

    def identify(self, *a, **kw):
        raise RuntimeError("wedged")


def test_a_wedged_gallery_says_so_and_names_nobody(rig, tmp_path):
    """Both used to arrive as who="". The wedged one SAYS SO (who_fault),
    and the gate answers it as NOBODY -- fail shut, never the owner.

    The first version of this policy treated a fault as NO INSTRUMENT:
    admitted blind, counted toward the dead-man. The round-3 review
    measured the hole in that: identify() raising while centroids() works
    let her clip clear the bar on HER pool and arrive with owner scope,
    50/50 with the camera off. The pool is a fact from the matching
    instrument, which ran; a fault in the naming instrument is an
    abstention, and an abstention on somebody else's pool is nobody
    (tests/test_voice_owner_lockout.py measures both sides)."""
    v, enc, gate = rig
    world = Voices(seed=35, apart=0.3)
    wedged = _RaisesOnIdentify(root=tmp_path / "vg2")
    _enrol(wedged, world, "hunter", 14)
    _enrol(wedged, world, "mara", 10)
    v.gallery = wedged

    hers = _clip(4.0, 0.71)
    enc.teach(hers, world.take("mara"))
    out, stats = v.filter_segments(hers)
    assert stats["matched"] >= 1, "her own label should have matched"
    assert stats["who"] == "" and stats["who_fault"]
    d = gate.judge("voice", "unlock the door", stats=stats)
    assert d.who != "hunter"
    assert d.how == gt.HOW_NOBODY and d.admit is False

    v.gallery = vg.VoiceGallery(root=tmp_path / "vg3")
    _enrol(v.gallery, world, "hunter", 14)
    stranger = _clip(4.0, 0.72)
    enc.teach(stranger, world.take("nobody"))
    out2, stats2 = v.filter_segments(stranger)
    if stats2["matched"] >= 1:
        pytest.skip("this fixture's stranger is not far enough away")
    assert stats2["who_fault"] == ""
    d2 = gate.judge("voice", "unlock the door", stats=stats2,
                    rejected=out2 is None)
    assert d2.how == gt.HOW_NOBODY and d2.admit is False
    # distinguishable in the STATS and the log, not in the verdict
    assert bool(stats["who_fault"]) != bool(stats2["who_fault"])


def test_a_gallery_that_cannot_even_list_itself_is_a_fault(rig):
    """And a fault is nobody, on his own voiceprint too: the stated cost of
    failing shut. Typed input, the socket and the passphrase remain."""
    v, enc, gate = rig
    world = Voices(seed=36)
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()
    v.gallery = _RaisesOnEverything()
    clip = _clip(4.0, 0.73)
    enc.teach(clip, world.take("hunter"))
    out, stats = v.filter_segments(clip)
    assert stats["who_fault"]
    d = gate.judge("voice", "what's the time", stats=stats)
    assert d.how == gt.HOW_NOBODY and d.who == "" and d.admit is False


def test_a_fault_holds_the_door_rather_than_tripping_the_dead_man(tmp_path):
    """The dead-man counts turns where NOTHING was measuring. A fault is a
    leg that ran and could not name, so three of them refuse three turns
    and stand nothing down."""
    g = _gate(tmp_path)
    for _ in range(gt.DEADMAN_TURNS):
        d = g.judge("voice", "hello", stats=_stats(who_fault="wedged"))
        assert d.admit is False
    assert g.stood_down is False and d.line != gt.STANDDOWN_LINE


# --------------------------------------------- 5. the keys are always there
def test_every_path_out_of_filter_segments_carries_the_identity_keys(rig):
    v, enc, _gate_ = rig
    keys = {"who", "who_scores", "labels", "abstained", "who_fault"}
    clip = _clip(4.0, 0.81)
    _out, fresh = v.filter_segments(clip)                # nothing enrolled
    assert keys <= set(fresh)
    world = Voices(seed=37)
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()
    enc.teach(clip, world.take("hunter"))
    _out, matched = v.filter_segments(clip)             # a match
    assert keys <= set(matched) and matched["labels"] == ()
    short = _clip(1.0, 0.82)
    enc.teach(short, world.take("hunter"))
    _out, abstained = v.filter_segments(short)          # an abstention
    assert abstained["abstained"] is True and abstained["matched"] == 1
    v._ensure_model = lambda: False
    _out, shut = v.filter_segments(clip)                # fail shut
    assert keys <= set(shut)


def test_the_gate_reads_the_new_keys_and_no_score(tmp_path):
    """The shape test in test_voice_multispeaker_wiring.py pins matched/who;
    this pins the three additions, by the syntax tree."""
    import ast
    import inspect
    read = set()
    for node in ast.walk(ast.parse(inspect.getsource(gt))):
        if isinstance(node, ast.Call) and \
                getattr(node.func, "attr", "") == "get":
            for arg in node.args[:1]:
                if isinstance(arg, ast.Constant):
                    read.add(arg.value)
    assert {"labels", "abstained", "who_fault", "matched_label", "top"} <= read
    assert "scores" not in read and "who_scores" not in read


def test_his_own_migrated_voice_is_still_named_with_a_guest_enrolled(rig):
    """The other direction: none of this may refuse HIM. Voiceprint plus his
    migrated label plus a full guest label -- his clip is named hunter."""
    v, enc, gate = rig
    world = Voices(seed=38, apart=0.3)
    him = world.takes("hunter", 14)
    v._embeddings = list(him)
    v._recompute_centroid()
    for e in him:
        v.gallery.add("hunter", e)
    _enrol(v.gallery, world, "mara", 10)
    clip = _clip(4.0, 0.91)
    enc.teach(clip, world.take("hunter"))
    out, stats = v.filter_segments(clip)
    assert out is not None and stats["who"] == "hunter"
    d = gate.judge("voice", "read my mail", stats=stats)
    assert d.admit is True and d.who == "hunter" and d.role == ident.ROLE_OWNER
    assert np.isfinite(stats["who_scores"]["hunter"])
