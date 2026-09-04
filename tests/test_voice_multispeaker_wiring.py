"""The invariants multi-speaker voice ID must not break.

The two gates fail in OPPOSITE directions and both directions are load-bearing
(jarvis/CLAUDE.md): the wake-word gate fails OPEN, because an unwakeable
assistant is worse than an over-eager one; the transcript gate fails SHUT once
anybody is enrolled, because silently accepting every voice is how a
television reached the commander. With nothing enrolled BOTH fail open, so a
fresh box is never mute.

Adding labels may not move either direction, and every test here is one of
those directions rather than a feature.

Synthetic vectors only. No microphone, no recording, no real voiceprint.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from jarvis import gate as gt
from jarvis import identity as ident
from jarvis import speaker as sp
from jarvis import voicegallery as vg
from tests.synthvoice import Voices


class _FakeEncoder:
    """Stands in for ECAPA. Returns the vector it was handed for a clip, keyed
    by the clip's first sample -- so a test can say "this audio is Mara"
    without a model, a GPU or a microphone."""

    def __init__(self):
        self.table = {}

    def teach(self, audio, vec):
        self.table[float(audio[0])] = np.asarray(vec, dtype=np.float32)
        return audio

    def __call__(self, audio, min_seconds=None):
        return self.table.get(float(audio[0]))


def _clip(seconds, marker):
    """A clip that survives trim_silence and carries an identifying marker in
    its first sample."""
    n = int(sp.SAMPLE_RATE * seconds)
    rng = np.random.default_rng(int(abs(marker) * 1000) + 5)
    a = (rng.normal(size=n) * 0.2).astype(np.float32)
    a[0] = np.float32(marker)
    return a


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """A verifier whose encoder is a lookup table and whose gallery is in a
    throwaway directory."""
    monkeypatch.setattr(sp, "VOICEPRINT_FILE", tmp_path / "voiceprint.npz")
    enc = _FakeEncoder()
    v = sp.SpeakerVerifier()
    v._model_loaded = True
    monkeypatch.setattr(v, "_ensure_model", lambda: True)
    monkeypatch.setattr(v, "_extract_embedding", enc)
    v.gallery = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    return v, enc


def _enrol(gallery, world, label, n):
    for e in world.takes(label, n):
        gallery.add(label, e)


# ------------------------------------------- 1. the wake gate stays a boolean
def test_the_wake_gate_asks_for_a_boolean_and_never_a_name():
    """hotword._speaker_ok calls score() and asks a threshold question.
    Naming anybody there would put a coin flip in front of the identity chain:
    it runs on a 2 s ring buffer holding 0.40-0.88 s of speech on 6 of his 10
    recorded wake clips, which is BELOW the abstain floor."""
    from jarvis import hotword
    src = inspect.getsource(hotword.Hotword._speaker_ok)
    tree = ast.parse(textwrap.dedent(src))
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}
    assert "identify" not in called
    assert "verify" not in called
    assert "score" in called


def test_enrolling_somebody_can_only_make_the_wake_gate_more_permissive(rig):
    """score() is a MAXIMUM over every enrolled centroid, so a new label can
    only ever raise it. That is the fail-open direction, by construction
    rather than by care."""
    v, enc = rig
    world = Voices(seed=1, apart=0.3)
    him = world.takes("hunter", 14)
    v._embeddings = list(him)
    v._recompute_centroid()

    probe = world.take("mara")
    clip = _clip(2.0, 0.11)
    enc.teach(clip, probe)
    before = v.score(clip)

    _enrol(v.gallery, world, "mara", 10)
    after = v.score(clip)
    assert after >= before - 1e-9, (before, after)
    assert after > before, "the new label should raise this particular probe"


# ------------------------------------- 2. the transcript gate stays fail-shut
def test_an_unrecognised_voice_is_still_dropped(rig):
    v, enc = rig
    world = Voices(seed=2, apart=0.02)
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()
    clip = _clip(4.0, 0.22)
    enc.teach(clip, world.take("a-stranger"))
    out, stats = v.filter_segments(clip)
    if stats["scores"] and max(stats["scores"]) >= v.threshold:
        pytest.skip("this fixture's stranger is not far enough away")
    assert out is None
    assert stats["matched"] == 0
    assert stats["who"] == "", "a rejected clip carried a name"


def test_a_broken_model_still_fails_shut(rig):
    v, _enc = rig
    world = Voices(seed=3)
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()
    v._ensure_model = lambda: False
    out, stats = v.filter_segments(_clip(4.0, 0.33))
    assert out is None and stats["matched"] == 0
    assert stats["who"] == "" and stats["who_scores"] == {}


def test_a_gallery_only_box_still_fails_shut(rig):
    """Somebody enrolled in the GALLERY with no voiceprint at all must still
    be filtered for. If is_enrolled ignored the gallery, the person who just
    enrolled would be the one person the gate stopped protecting."""
    v, enc = rig
    world = Voices(seed=4, apart=0.02)
    _enrol(v.gallery, world, "mara", 10)
    assert v._embeddings == []
    assert v.is_enrolled is True
    clip = _clip(4.0, 0.44)
    enc.teach(clip, world.take("a-stranger"))
    out, stats = v.filter_segments(clip)
    if stats["scores"] and max(stats["scores"]) >= v.threshold:
        pytest.skip("this fixture's stranger is not far enough away")
    assert out is None and stats["matched"] == 0


# ------------------------------------------ 3. nothing enrolled fails open
def test_a_fresh_box_is_never_mute(rig):
    v, _enc = rig
    assert v._embeddings == []
    assert v.gallery.labels() == ()
    assert v.is_enrolled is False
    clip = _clip(4.0, 0.55)
    out, stats = v.filter_segments(clip)
    assert out is clip
    assert stats["matched"] == 0 and stats["total"] == 0
    assert stats["who"] == ""
    assert v.verify(clip) == (True, 1.0)


def test_an_empty_gallery_is_not_an_enrolment(rig):
    v, _enc = rig
    v.gallery = vg.VoiceGallery()
    assert v.is_enrolled is False


# --------------------------------------- 4. the owner fallback in the gate
def _gate(tmp_path, mode="enforce"):
    reg = ident.Registry(path=tmp_path / "people.json")
    reg.add_person(ident.Person(label="hunter", name="Hunter",
                                role=ident.ROLE_OWNER))
    reg.add_person(ident.Person(label="mara", name="Mara",
                                role=ident.ROLE_KNOWN))
    reg.save()
    reg = ident.Registry.load(tmp_path / "people.json")
    opts = {"owner.mode": mode}
    return gt.OwnerGate(registry=reg, owner="hunter",
                        get_option=lambda k, d=None: opts.get(k, d))


def test_a_nameless_match_is_still_him(tmp_path):
    """THE CONCRETE LOCKOUT. An abstention sets matched=1 and no name; every
    "Yes." he says arrives that way. An empty label here refuses his own
    follow-ups."""
    stats = {"total": 1, "matched": 1, "scores": [0.0], "who": "",
             "who_scores": {}}
    d = _gate(tmp_path).judge("voice", "Yes.", stats=stats)
    assert d.admit is True and d.who == "hunter"


def test_a_named_match_is_that_person(tmp_path):
    stats = {"total": 3, "matched": 2, "scores": [0.41, 0.38, 0.1],
             "who": "mara", "who_scores": {"mara": 0.41, "hunter": 0.12}}
    d = _gate(tmp_path).judge("voice", "what time is it", stats=stats)
    assert d.who == "mara"
    assert d.role == ident.ROLE_KNOWN


def test_a_label_the_registry_does_not_hold_is_not_an_identity(tmp_path):
    """recognise() already drops it, and that is what stops a stale voice
    label minting a person."""
    stats = {"total": 3, "matched": 2, "scores": [0.41], "who": "heather",
             "who_scores": {"heather": 0.41}}
    d = _gate(tmp_path).judge("voice", "unlock the door", stats=stats)
    assert d.who != "heather"


def test_the_stats_dict_keeps_its_shape(tmp_path):
    """"matched" keeps its exact present meaning; "who" and "who_scores" are
    additions. test_owner_gate.py forbids the gate reading "scores" or
    "best_score" and requires it to read "matched"."""
    read = set()
    for node in ast.walk(ast.parse(inspect.getsource(gt))):
        if isinstance(node, ast.Subscript) and \
                isinstance(node.slice, ast.Constant):
            read.add(node.slice.value)
        elif isinstance(node, ast.Call) and \
                getattr(node.func, "attr", "") == "get":
            for arg in node.args[:1]:
                if isinstance(arg, ast.Constant):
                    read.add(arg.value)
    assert "matched" in read and "who" in read
    assert "scores" not in read and "best_score" not in read


def test_the_gate_still_owns_no_threshold():
    src = inspect.getsource(gt)
    found = [tok for tok in src.replace("(", " ").replace(")", " ")
             .replace(",", " ").split()
             if _is_bar(tok)]
    assert found == [], found


def _is_bar(tok):
    try:
        val = float(tok)
    except ValueError:
        return False
    return 0.0 < val < 1.0


# ----------------------------------------------- 5. the abstain still holds
def test_a_short_clip_abstains_and_names_nobody(rig):
    v, enc = rig
    world = Voices(seed=5)
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()
    _enrol(v.gallery, world, "mara", 10)
    clip = _clip(1.0, 0.66)           # under ABSTAIN_SECONDS of speech
    enc.teach(clip, world.take("hunter"))
    ok, score, ident = v._verify_named(clip)
    assert ok is True and score == 0.0
    assert ident["who"] == "" and ident["who_scores"] == {}
    assert ident["abstained"] is True, "an abstention must say it is one"


def test_the_abstain_window_is_unchanged():
    assert sp.ABSTAIN_SECONDS == 1.5 == vg.ABSTAIN_SECONDS


# --------------------------------------------- 6. a label actually arrives
def test_the_gallery_names_the_speaker_on_a_real_capture(rig):
    v, enc = rig
    world = Voices(seed=6, apart=0.3)
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()
    _enrol(v.gallery, world, "mara", 10)
    clip = _clip(4.0, 0.77)
    enc.teach(clip, world.take("mara"))
    out, stats = v.filter_segments(clip)
    assert out is not None and stats["matched"] >= 1
    assert stats["who"] == "mara"
    assert set(stats["who_scores"]) == {"mara"}


def test_a_failed_margin_reaches_the_gate_as_no_name(rig):
    """The near miss. Two enrolled people too close to separate: matched can
    still be 1 (the audio IS somebody enrolled) and the NAME is withheld.
    With two labels the gate then reads that as NOBODY -- never the owner;
    tests/test_voice_owner_fallback.py proves it at the gate."""
    v, enc = rig
    world = Voices(seed=7, apart=3.0)
    _enrol(v.gallery, world, "hunter", 14)
    _enrol(v.gallery, world, "mara", 12)
    clip = _clip(4.0, 0.88)
    enc.teach(clip, world.take("mara"))
    _out, stats = v.filter_segments(clip)
    verdict = v.gallery.identify(world.take("mara"), 4.0, v.threshold)
    assert verdict.margin is not None and verdict.margin < vg.MARGIN
    assert stats["who"] == ""
    assert stats["labels"] == ("hunter", "mara")
    assert stats["abstained"] is False and stats["who_fault"] == ""


def test_a_provisional_label_cannot_match_at_all(rig):
    """A label with too few takes to be NAMED used to still sit inside the
    maximum ``matched`` is taken over, so it matched as a nameless somebody
    -- and a nameless match on a one-label box is the owner. It scores and
    logs; it does not open the door."""
    v, enc = rig
    world = Voices(seed=8, apart=0.3)
    _enrol(v.gallery, world, "mara", 4)
    clip = _clip(4.0, 0.99)
    enc.teach(clip, world.take("mara"))
    out, stats = v.filter_segments(clip)
    assert stats["who"] == ""
    assert out is None and stats["matched"] == 0, \
        "a provisional label matched as somebody"
    assert "mara" in stats["who_scores"], "it should still SCORE"


# ----------------------------------------------- 7. the voiceprint is safe
def test_the_voiceprint_is_never_written_with_multi_label_keys(rig):
    """It is the rollback. Multi-label keys in that file are readable as one
    pooled centroid by any build without the format refusal."""
    v, enc = rig
    world = Voices(seed=9)
    clip = _clip(4.0, 1.11)
    enc.teach(clip, world.take("hunter"))
    v.enroll_from_audio(clip)
    data = np.load(sp.VOICEPRINT_FILE)
    for key in data.files:
        if key.startswith("emb_"):
            assert vg._KEY_RE.match(key) is None, \
                "a labelled key was written into voiceprint.npz: %s" % key


def test_a_broken_gallery_does_not_cost_him_the_voiceprint(rig, tmp_path):
    v, _enc = rig
    world = Voices(seed=10)
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()

    class Broken:
        def labels(self):
            raise RuntimeError("boom")

        def centroids(self):
            raise RuntimeError("boom")

        def identify(self, *a, **kw):
            raise RuntimeError("boom")

    v.gallery = Broken()
    assert v.is_enrolled is True          # the voiceprint alone still counts
    assert v._all_centroids() != {}
    verdict, fault = v._who(world.take("hunter"), 3.0)
    assert verdict is None
    assert fault, "a raising gallery must say so, not look like a non-match"
