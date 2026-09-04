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
2. With two or more labels in the gallery a nameless match is NOBODY at the
   gate, never the owner. With at most one label the fallback stands,
   because the only nameless pool that can have matched is his.

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
            "who_fault": ""}
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
def test_two_labels_and_no_name_is_nobody(tmp_path):
    d = _gate(tmp_path).judge("voice", "unlock the door",
                              stats=_stats(labels=("hunter", "mara")))
    assert d.who == "" and d.admit is False and d.how == gt.HOW_NOBODY


def test_two_labels_and_no_name_names_the_way_back_in(tmp_path):
    d = _gate(tmp_path).judge("voice", "unlock the door",
                              stats=_stats(labels=("hunter", "mara")))
    assert d.line == gt.UNKNOWN_LINE


@pytest.mark.parametrize("labels", [(), ("hunter",), ("mara",)],
                         ids=["no-gallery", "his-label", "one-guest"])
def test_at_most_one_label_keeps_the_owner_fallback(tmp_path, labels):
    """With one label the only nameless pool that can have matched is the
    voiceprint's -- a provisional label cannot match at all, and a named
    label arrives named -- so a nameless match still means him."""
    d = _gate(tmp_path).judge("voice", "what's the time",
                              stats=_stats(labels=labels))
    assert d.admit is True and d.who == "hunter"


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
        "voice", "unlock the door", stats=_stats(labels=("hunter", "mara")))
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


def test_a_wedged_gallery_is_distinguishable_from_a_non_match(rig, tmp_path):
    """Both used to arrive as who="". Now the wedged one says so, and the
    gate treats it as NO INSTRUMENT: admitted blind and counted toward the
    dead-man, never as a name -- where the stranger is refused."""
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
    assert d.how == gt.HOW_BLIND

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
    assert d.how != d2.how


def test_a_gallery_that_cannot_even_list_itself_is_a_fault(rig):
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
    assert d.how == gt.HOW_BLIND and d.who == ""


def test_a_fault_trips_the_dead_man_rather_than_holding_the_door(tmp_path):
    g = _gate(tmp_path)
    for _ in range(gt.DEADMAN_TURNS):
        d = g.judge("voice", "hello", stats=_stats(who_fault="wedged"))
    assert g.stood_down is True and d.line == gt.STANDDOWN_LINE


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
    assert {"labels", "abstained", "who_fault"} <= read
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
