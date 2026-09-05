"""The OWNER's side of multi-speaker voice ID, measured.

Round 3 closed the escalation (hers as him: 0/150 in every cell) and the
round-3 review then found four ways the branch still failed HIM -- three of
them lockouts, one a policy hole. Each is reproduced here on synthetic vectors
(tests/synthvoice.py) against the pre-fix sources and asserted on the fix.

  1. A FRESH BOX. ``owner_ready`` let a guest be enrolled FIRST (no
     voiceprint, empty gallery) while its docstring claimed the opposite.
     Measured on 26be9e7 with mara@10 in the gallery and him nowhere:
     hotword._speaker_ok woke him 0/50 -- the fail-open ``None`` had become
     a number under the bar -- and the transcript gate admitted him 0/50.
     Now the script refuses with a line naming what to run, and a verifier
     that was TOLD who the owner is reads a gallery holding only other
     people as NO INSTRUMENT for him: both gates fail open, loudly. So the
     invariant "enrolling somebody can only make the wake gate more
     permissive" holds at the nothing -> guest transition, which is exactly
     where it broke; the existing invariant test starts from him enrolled
     and could not see it.
  2. A LABEL MISMATCH. ``gate._voice_leg`` handed the gate the raw GALLERY
     label and ``recognise`` drops any label the registry lacks. His gallery
     label is identity.owner_label(cfg) = slug(user.name); his registry row
     is TYPED. user.name -> "hunterpeyrovi" beside a registry row "hunter",
     migrated, mara@10, apart 0.3: his 4 s turns refused 50/50, and
     load_gallery's warning did not fire because his label IS in the
     gallery. Now ONE resolver (``OwnerGate._registry_label``) maps who /
     top / matched_label from gallery space into registry space, the way
     ``_face_leg`` already does for a face label.
  3. ``--migrate --label mara`` was accepted: rc 0, his fourteen vectors
     filed under her label, consent recorded "owner", and at runtime the
     cosine fold made "mara" his pool so he was answered AS MARA with KNOWN
     scope 30/30. ``--migrate`` takes only the owner's label and refuses
     anything else before the gallery is opened.
  4. A PARTIAL FAULT. ``identify()`` raising while ``centroids()`` works let
     her clip clear the bar on HER pool; ``_voice_leg`` saw ``who_fault``
     and returned "not running" before it read the pool, so with the camera
     off the turn was admitted BLIND with owner scope (50/50). RULING: any
     fault in the identity path is an abstention -- the leg ran and names
     NOBODY, fail shut -- never the owner. The cost is stated and measured:
     a persistently faulting gallery now refuses him on the voice path too
     (typed, the socket and the passphrase remain), where it used to stand
     the gate down after three turns.

Plus the reviewer's twelve-cell grid (three fresh seeds x apart 0.3 / 3.0 x
both layouts x 150 of her clips), re-run: hers as the owner 0/150 in every
cell. Every between-people number here is a measurement of the CODE at a
synthetic separation; none is evidence about two real people. No microphone,
no recording, no real voiceprint, no camera.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from jarvis import gate as gt
from jarvis import hotword as hw
from jarvis import identity as ident
from jarvis import speaker as sp
from jarvis import voicegallery as vg
from tests.synthvoice import Voices
from tests.test_voice_multispeaker_wiring import _clip, _enrol, _FakeEncoder
from tests.test_voice_per_window import _layout

ROOT = Path(__file__).resolve().parent.parent
# slug("Hunter Peyrovi"): what --migrate files him under (identity.owner_label)
SLUG = "hunterpeyrovi"
# what he typed into the registry (scripts/jarvis_people.py add --label)
ROW = "hunter"


# ------------------------------------------------------------------ rigs
def _verifier(tmp_path, monkeypatch, owner_label="hunter"):
    """A verifier told who the owner is, the way app.py builds it, with a
    lookup-table encoder and a throwaway gallery. No voiceprint on disk."""
    monkeypatch.setattr(sp, "VOICEPRINT_FILE", tmp_path / "voiceprint.npz")
    enc = _FakeEncoder()
    v = sp.SpeakerVerifier(owner_label=owner_label)
    v._model_loaded = True
    monkeypatch.setattr(v, "_ensure_model", lambda: True)
    monkeypatch.setattr(v, "_extract_embedding", enc)
    v.gallery = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    return v, enc


def _registry(tmp_path, owner_row=ROW):
    reg = ident.Registry(path=tmp_path / "people.json")
    reg.add_person(ident.Person(label=owner_row, name="Hunter Peyrovi",
                                role=ident.ROLE_OWNER))
    reg.add_person(ident.Person(label="mara", name="Mara",
                                role=ident.ROLE_KNOWN))
    reg.save()
    return ident.Registry.load(tmp_path / "people.json")


def _gate(tmp_path, *, owner=ROW, mode="enforce"):
    """``owner`` is what app.py passes: identity.owner_label(cfg), the
    CONFIG slug -- not necessarily the registry row."""
    opts = {"owner.mode": mode}
    return gt.OwnerGate(registry=_registry(tmp_path), owner=owner,
                        get_option=lambda k, d=None: opts.get(k, d))


def _script():
    path = ROOT / "scripts" / "voice_enrol.py"
    spec = importlib.util.spec_from_file_location("_voice_enrol_lockout", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _speaker_verify_on():
    """The one switch the wake gate consults; restored after every test."""
    was = hw.CONFIG.speaker_verify
    hw.CONFIG.speaker_verify = True
    yield
    hw.CONFIG.speaker_verify = was


def _wake(v, clip):
    """hotword._speaker_ok driven directly on a 2 s wake buffer, the way the
    listener thread calls it. No audio device, no model."""
    h = hw.Hotword.__new__(hw.Hotword)
    h._speaker = v
    h._music_playing = None
    return h._speaker_ok(clip, 16000, oww_score=0.9)


def _turn(v, gate, clip, text="read my mail", **kw):
    """One voice turn: app._decode_clip's guard (the speaker filter runs
    only when somebody is enrolled; otherwise the gate sees no instrument),
    then the gate."""
    if v.enrolled:
        out, stats = v.filter_segments(clip)
        rejected = out is None
    else:
        stats, rejected = {}, False
    return stats, gate.judge("voice", text, stats=stats, rejected=rejected,
                             **kw)


def _measured(name, **counts):
    """Printed, so a ``-s`` run reports the numbers per fixture; the
    assertions below are what the suite enforces."""
    print("MEASURED %s: %s" % (
        name, " ".join("%s=%s" % kv for kv in counts.items())))


_BASE = {"total": 1, "matched": 1, "scores": [0.41], "who": "",
         "who_scores": {}, "labels": (), "abstained": False,
         "who_fault": "", "matched_label": "", "top": "",
         "provisional": "", "near_miss": False}


def _stats(**kw):
    out = dict(_BASE)
    out.update(kw)
    return out


# =========================================================== 1. fresh box
def test_a_guest_cannot_be_enrolled_before_the_owner_exists_anywhere():
    """The script's decision, pure. The owner must have a pool SOMEWHERE
    (the voiceprint or the gallery) before anybody else is enrolled; the
    docstring no longer claims a box with neither cannot be locked out."""
    ve = _script()
    g = vg.VoiceGallery()
    ok, why = ve.owner_ready(g, "hunter", "mara", False)
    assert ok is False
    assert "hunter" in why and "mara" in why
    assert "enroll_voice" in why and "--migrate" in why
    assert "cannot be locked out" not in (ve.owner_ready.__doc__ or "")
    # the owner himself, always
    assert ve.owner_ready(g, "hunter", "hunter", False) == (True, "")
    # him in the gallery and no voiceprint: a guest may follow
    _enrol(g, Voices(seed=1301, apart=0.3), "hunter", 14)
    assert ve.owner_ready(g, "hunter", "mara", False) == (True, "")


def test_main_refuses_the_first_guest_on_a_fresh_box_before_consent(
        tmp_path, monkeypatch, capsys):
    """rc 5 -- the owner_ready refusal -- with no voiceprint and nothing in
    the gallery, before the consent prompt (3) and before the recorder."""
    ve = _script()
    g = vg.VoiceGallery(root=tmp_path / "vg")
    monkeypatch.setattr(ve.vg, "default_gallery", lambda: g)
    monkeypatch.setattr(ve, "owner_label", lambda cfg: "hunter")
    monkeypatch.setattr(ve.PATHS, "VOICEPRINT", tmp_path / "voiceprint.npz")
    assert not (tmp_path / "voiceprint.npz").exists()
    assert ve.main(["--label", "mara"]) == 5
    err = capsys.readouterr().err
    assert "REFUSED" in err and "hunter" in err
    assert g.labels() == () and not (tmp_path / "vg").exists()


def test_a_gallery_holding_only_a_guest_is_no_instrument_for_him(
        tmp_path, monkeypatch):
    """THE LOCKOUT, MEASURED. The layout the script used to build (and an
    older build's gallery can still hold): mara@10 in the gallery, no
    voiceprint, the owner nowhere. His 2 s wake buffers and his 4 s turns,
    fifty each, before and after enrolling her.

    Pre-fix: woke 50 -> 0, admitted 50 -> 0. Now both stay 50: a verifier
    told who the owner is reads that gallery as no pool of HIS and fails
    open on both gates, saying why."""
    v, enc = _verifier(tmp_path, monkeypatch, owner_label="hunter")
    world = Voices(seed=1302, apart=0.3)

    def him(base):
        # A fresh gate per phase: with no instrument the dead-man stands a
        # gate down after three turns, and a stood-down gate admits in
        # shadow, which would hide a lockout measured on the same object.
        gate = _gate(tmp_path)
        woke = admitted = 0
        for i in range(50):
            c = _clip(2.0, base + i * 0.01)
            enc.teach(c, world.take("hunter"))
            woke += bool(_wake(v, c))
            c4 = _clip(4.0, base + 5.0 + i * 0.01)
            enc.teach(c4, world.take("hunter"))
            _st, d = _turn(v, gate, c4)
            admitted += bool(d.admit)
        return woke, admitted

    assert v.is_enrolled is False
    before = him(200.0)
    _enrol(v.gallery, world, "mara", 10)
    assert v.gallery.labels() == ("mara",)
    after = him(210.0)
    _measured("fresh-box guest-first", woke_before=before[0],
              admitted_before=before[1], woke_after=after[0],
              admitted_after=after[1])
    assert before == (50, 50), "a fresh box was mute before anybody enrolled"
    assert after == (50, 50), ("enrolling a guest first locked him out: "
                               "woke %d/50, admitted %d/50" % after)
    assert v.is_enrolled is False
    gap = v.enrolment_gap()
    assert "hunter" in gap and "mara" in gap and "--migrate" in gap


def test_the_wake_gate_invariant_holds_from_nothing(tmp_path, monkeypatch):
    """The transition the existing invariant test cannot see: NOTHING ->
    a guest. score() must stay None (the wake gate's fail-open) rather
    than become a number under the bar, and a provisional guest changes
    nothing either."""
    v, enc = _verifier(tmp_path, monkeypatch, owner_label="hunter")
    world = Voices(seed=1303, apart=0.3)
    clip = _clip(2.0, 220.0)
    enc.teach(clip, world.take("hunter"))
    assert v.score(clip) is None
    _enrol(v.gallery, world, "heather", 6)
    assert v.score(clip) is None
    _enrol(v.gallery, world, "mara", 10)
    assert v.score(clip) is None, "a guest alone gave the wake gate a number"
    assert _wake(v, clip) is True
    # ...and once HE has a pool, the number is his to clear.
    v._embeddings = world.takes("hunter", 14)
    v._recompute_centroid()
    assert v.is_enrolled is True
    assert v.score(clip) >= hw.Hotword.SPEAKER_WAKE_MIN


def test_a_verifier_nobody_told_keeps_the_old_reading(tmp_path, monkeypatch):
    """STATED, NOT HIDDEN. voice_check.py, hotword_daemon.py and the
    scripts build a SpeakerVerifier without owner_label; with no voiceprint
    they cannot tell whose the gallery is, so a gallery with anybody in it
    still counts as an enrolment there -- exactly as before. The explicit
    label is the mechanism, and only app.py passes it."""
    v, _enc = _verifier(tmp_path, monkeypatch, owner_label="")
    _enrol(v.gallery, Voices(seed=1304, apart=0.3), "mara", 10)
    assert v.is_enrolled is True
    assert v.enrolment_gap() == ""


def test_the_owner_in_the_gallery_alone_is_still_an_enrolment(
        tmp_path, monkeypatch):
    """The other half of the "or" in is_enrolled: him in the gallery with
    no voiceprint (enroll_voice --reset after --migrate) is HIS pool, the
    transcript gate keeps failing shut, and a stranger is dropped."""
    v, enc = _verifier(tmp_path, monkeypatch, owner_label="hunter")
    world = Voices(seed=1305, apart=0.02)
    _enrol(v.gallery, world, "hunter", 14)
    assert v._embeddings == [] and v.is_enrolled is True
    clip = _clip(4.0, 230.0)
    enc.teach(clip, world.take("a-stranger"))
    out, stats = v.filter_segments(clip)
    if stats["scores"] and max(stats["scores"]) >= v.threshold:
        pytest.skip("this fixture's stranger is not far enough away")
    assert out is None and stats["matched"] == 0


def test_load_says_out_loud_when_the_owner_has_no_pool(
        tmp_path, monkeypatch, caplog):
    """A WARNING at load, not only a debug line on the first fail-open:
    the gallery holds somebody and he is nowhere."""
    world = Voices(seed=1306, apart=0.3)
    g = vg.VoiceGallery(root=tmp_path / "vg")
    _enrol(g, world, "mara", 10)
    g.save("test")
    monkeypatch.setattr(vg, "default_gallery",
                        lambda model=None: vg.VoiceGallery(root=tmp_path / "vg"))
    monkeypatch.setattr(sp, "VOICEPRINT_FILE", tmp_path / "voiceprint.npz")
    v = sp.SpeakerVerifier(owner_label="hunter")
    with caplog.at_level("WARNING", logger="speaker"):
        v.load()
    assert v.is_enrolled is False
    said = [r.getMessage() for r in caplog.records if "OFF" in r.getMessage()]
    assert said and any("mara" in s and "hunter" in s for s in said), said


def test_app_runs_the_filter_only_when_somebody_is_enrolled():
    """The seam _turn mirrors: with is_enrolled False the app hands the
    gate an EMPTY stats dict (no instrument), which is what makes the
    fresh-box fail-open reach the gate as a fail-open."""
    from jarvis.app import JarvisApp
    src = inspect.getsource(JarvisApp._decode_clip)
    assert "self.speaker.enrolled" in src
    assert "filter_segments" in src


# ====================================================== 2. label mismatch
@pytest.mark.parametrize("migrated", [False, True],
                         ids=["voiceprint+mara", "voiceprint+slug+mara"])
def test_his_typed_registry_row_and_his_config_slug_are_one_person(
        tmp_path, monkeypatch, migrated):
    """user.name -> "hunterpeyrovi", registry row "hunter". Migrated, the
    gallery names him by the slug and the registry has never heard of it:
    pre-fix 50/50 of his turns refused as nobody. Un-migrated was already
    0/50 (a nameless match on his pool folds to the owner) and stays so.
    Her turns are hers in both layouts, never his."""
    v, enc = _verifier(tmp_path, monkeypatch, owner_label=SLUG)
    gate = _gate(tmp_path, owner=SLUG)
    assert gate._owner_label() == ROW
    world = Voices(seed=1307, apart=0.3)
    him = world.takes("hunter", 14)
    v._embeddings = list(him)
    v._recompute_centroid()
    if migrated:
        for e in him:
            v.gallery.add(SLUG, e)
    _enrol(v.gallery, world, "mara", 10)
    refused = as_row = hers_as_him = hers_named = 0
    for i in range(50):
        c = _clip(4.0, 300.0 + i * 0.01)
        enc.teach(c, world.take("hunter"))
        st, d = _turn(v, gate, c, "read my mail")
        refused += (not d.admit)
        as_row += (d.admit and d.who == ROW and d.role == ident.ROLE_OWNER)
        if migrated:
            # the gallery still speaks its own label; the GATE translates
            assert st["who"] == SLUG, st
        c2 = _clip(4.0, 310.0 + i * 0.01)
        enc.teach(c2, world.take("mara"))
        _st2, d2 = _turn(v, gate, c2, "what's the time")
        hers_as_him += (d2.admit and d2.who == ROW)
        hers_named += (d2.who == "mara")
    _measured("label-mismatch migrated=%s" % migrated, his_refused=refused,
              his_as_row=as_row, hers_as_him=hers_as_him,
              hers_named=hers_named)
    assert refused == 0, "%d/50 of his turns refused on a label mismatch" % refused
    assert as_row == 50, as_row
    assert hers_as_him == 0 and hers_named == 50, (hers_as_him, hers_named)


def test_one_resolver_maps_every_gallery_label_the_gate_reads(tmp_path):
    """who, top and matched_label all pass through OwnerGate._registry_label:
    a registry row is itself, the config slug is the registry OWNER, and a
    label the registry lacks stays somebody (recognise() drops it)."""
    g = _gate(tmp_path, owner=SLUG)
    assert g._registry_label(SLUG) == ROW
    assert g._registry_label(ROW) == ROW
    assert g._registry_label("mara") == "mara"
    assert g._registry_label("") == ""
    assert g._registry_label("zed") == "zed"
    base = _stats(labels=(SLUG, "mara"))
    named = g.judge("voice", "read my mail", stats=dict(base, who=SLUG))
    assert named.admit and named.who == ROW
    assert named.role == ident.ROLE_OWNER and named.how == gt.HOW_VOICE
    top = g.judge("voice", "read my mail", stats=dict(base, top=SLUG))
    assert top.admit and top.who == ROW
    pool = g.judge("voice", "read my mail",
                   stats=dict(base, matched_label=SLUG))
    assert pool.admit and pool.who == ROW
    hers = g.judge("voice", "read my mail", stats=dict(base, top="mara"))
    assert hers.who != ROW and hers.admit is False
    zed = g.judge("voice", "read my mail", stats=dict(base, who="zed"))
    assert zed.who != "zed" and zed.admit is False
    # a gate whose config slug IS the registry row: nothing changes
    same = _gate(tmp_path, owner=ROW)
    assert same._registry_label(ROW) == ROW
    assert same._registry_label(SLUG) == SLUG


def test_the_resolver_is_used_for_the_three_keys_by_the_syntax_tree():
    """Not by care: every read of who / top / matched_label inside
    _voice_leg goes through the resolver."""
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(gt.OwnerGate._voice_leg)))
    wrapped = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
                getattr(node.func, "attr", "") == "_registry_label":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and \
                        getattr(inner.func, "attr", "") == "get":
                    for arg in inner.args[:1]:
                        if isinstance(arg, ast.Constant):
                            wrapped.add(arg.value)
    assert {"who", "top", "matched_label"} <= wrapped, wrapped


# ================================================ 3. --migrate --label mara
def test_migrate_takes_only_the_owners_label(tmp_path, monkeypatch, capsys):
    """rc 2, nothing opened, nothing written; then the owner's label and no
    label at all still migrate, consent "owner" under HIS label."""
    ve = _script()
    root = tmp_path / "vg"
    src = tmp_path / "voiceprint.npz"
    arrays = {"emb_%04d" % i: np.asarray(v)
              for i, v in enumerate(Voices(seed=1308).takes("hunter", 14))}
    arrays["_format"] = np.array([2])
    with open(src, "wb") as fh:
        np.savez(fh, **arrays)
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", root)
    monkeypatch.setattr(vg.PATHS, "VOICEPRINT", src)
    monkeypatch.setattr(ve, "owner_label", lambda cfg: "hunter")
    asked = []
    real = vg.VoiceGallery.migrate_voiceprint

    def spy(self, *a, **kw):
        asked.append(a)
        return real(self, *a, **kw)
    monkeypatch.setattr(vg.VoiceGallery, "migrate_voiceprint", spy)

    assert ve.main(["--migrate", "--label", "mara"]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "mara" in err and "hunter" in err
    assert asked == [], "the gallery was asked to migrate under her label"
    assert not root.exists()
    assert vg.VoiceGallery(root=root).disk_labels() == ()

    assert ve.main(["--migrate"]) == 0
    assert asked == [("hunter",)]
    g = vg.VoiceGallery(root=root)
    assert g.load() and g.labels() == ("hunter",)
    assert g.consent("hunter") == "owner" and g.count("hunter") == 14
    assert src.read_bytes() == open(src, "rb").read()   # untouched


def test_the_migrate_label_rule_is_pure():
    ve = _script()
    assert ve.migrate_label_ok("", "hunter") == (True, "")
    assert ve.migrate_label_ok("hunter", "hunter") == (True, "")
    ok, why = ve.migrate_label_ok("mara", "hunter")
    assert ok is False and "mara" in why and "hunter" in why
    assert "--label mara" in why, "the refusal names the right command"


def test_migrate_refuses_before_the_gallery_is_opened(tmp_path, monkeypatch,
                                                      capsys):
    """"Before touching the file" means before default_gallery() too."""
    ve = _script()
    monkeypatch.setattr(ve, "owner_label", lambda cfg: "hunter")

    def boom():
        raise AssertionError("the gallery was opened")
    monkeypatch.setattr(ve.vg, "default_gallery", boom)
    assert ve.main(["--migrate", "--label", "mara"]) == 2
    assert "REFUSED" in capsys.readouterr().err


# ======================================================= 4. partial fault
class _IdentifyWedged(vg.VoiceGallery):
    """centroids() works, identify() raises: the matching instrument ran
    and the naming instrument did not."""

    def identify(self, *a, **kw):
        raise RuntimeError("wedged")


def test_a_fault_in_the_identity_path_names_nobody_never_the_owner(
        tmp_path, monkeypatch):
    """Her 4 s clips on HER pool under a wedged identify(): pre-fix admitted
    BLIND 50/50 with the camera off and 20/20 with it running and naming
    nobody. Now 0 and 0. His clips under the same fault: refused 50/50 --
    the stated cost of failing shut -- and the dead-man does not trip,
    because the leg RAN."""
    v, enc = _verifier(tmp_path, monkeypatch)
    gate = _gate(tmp_path)
    world = Voices(seed=1309, apart=0.3)
    him = world.takes("hunter", 14)
    v._embeddings = list(him)
    v._recompute_centroid()
    wedged = _IdentifyWedged(root=tmp_path / "vg2")
    for e in him:
        wedged.add("hunter", e)
    _enrol(wedged, world, "mara", 10)
    v.gallery = wedged

    hers_admitted = hers_blind = hers_face = his_admitted = 0
    for i in range(50):
        c = _clip(4.0, 400.0 + i * 0.01)
        enc.teach(c, world.take("mara"))
        st, d = _turn(v, gate, c, "read my mail")
        assert st["matched"] >= 1 and st["matched_label"] == "mara", st
        assert st["who_fault"] and st["who"] == ""
        hers_admitted += bool(d.admit)
        hers_blind += (d.how == gt.HOW_BLIND)
        if i < 20:
            d2 = gate.judge("voice", "read my mail", stats=st, face="",
                            face_running=True)
            hers_face += bool(d2.admit)
        c3 = _clip(4.0, 410.0 + i * 0.01)
        enc.teach(c3, world.take("hunter"))
        st3, d3 = _turn(v, gate, c3, "read my mail")
        assert st3["matched_label"] == "" and st3["who_fault"]
        his_admitted += bool(d3.admit)
        assert d3.who != "hunter"
    _measured("identify-wedged", hers_admitted=hers_admitted,
              hers_blind=hers_blind, hers_admitted_face_running=hers_face,
              his_admitted=his_admitted)
    assert hers_admitted == 0, "%d/50 of hers admitted under a fault" % hers_admitted
    assert hers_face == 0, hers_face
    assert his_admitted == 0, "fail shut means shut for him too: %d" % his_admitted
    assert gate.stood_down is False, "a fault stood the gate down"


def test_a_fault_dict_is_nobody_at_the_gate_in_both_modes(tmp_path):
    hers = _stats(who_fault="wedged", matched_label="mara")
    d = _gate(tmp_path).judge("voice", "read my mail", stats=hers)
    assert d.admit is False and d.how == gt.HOW_NOBODY and d.who == ""
    d = _gate(tmp_path, mode="shadow").judge("voice", "read my mail",
                                             stats=hers)
    assert d.admit is True and d.would_refuse is True and d.who == ""
    assert d.how == gt.HOW_NOBODY
    # on HIS pool, or on no pool at all: still nobody, never the owner
    for st in (_stats(who_fault="wedged", matched_label=""),
               _stats(who_fault="wedged", matched_label="", matched=0),
               _stats(who_fault="wedged", abstained=True)):
        d = _gate(tmp_path).judge("voice", "read my mail", stats=st)
        assert d.admit is False and d.who == "", st


def test_a_fault_holds_the_door_and_does_not_trip_the_dead_man(tmp_path):
    """The dead-man counts turns where NOTHING was measuring. A fault is a
    leg that ran and could not name; three of them refuse three turns and
    stand nothing down."""
    g = _gate(tmp_path)
    for _ in range(gt.DEADMAN_TURNS + 1):
        d = g.judge("voice", "hello", stats=_stats(who_fault="wedged"))
        assert d.admit is False and d.line != gt.STANDDOWN_LINE
    assert g.stood_down is False


def test_the_face_leg_can_still_rescue_a_faulted_turn(tmp_path):
    """Fail shut on the VOICE leg is not a veto: a camera that names him is
    a positive on another leg (recognise rule 1), and a camera that names
    HER gives her her own scope, not his."""
    st = _stats(who_fault="wedged", matched_label="mara")
    him = _gate(tmp_path).judge("voice", "read my mail", stats=st,
                                face="hunter", face_running=True)
    assert him.admit and him.who == "hunter" and him.how == gt.HOW_FACE
    her = _gate(tmp_path).judge("voice", "read my mail", stats=st,
                                face="mara", face_running=True)
    assert her.who == "mara" and her.admit is False   # out of scope, not his


# ============================================== the reviewer's twelve cells
@pytest.mark.parametrize("seed", [1201, 1202, 1203])
@pytest.mark.parametrize("apart", [0.3, 3.0])
@pytest.mark.parametrize("migrated", [False, True],
                         ids=["voiceprint+mara", "voiceprint+hunter+mara"])
def test_the_reviewers_twelve_cells_stay_closed(tmp_path, monkeypatch, seed,
                                                apart, migrated):
    """Three fresh seeds x apart 0.3 / 3.0 x both layouts, 150 of her clips
    and 150 of his per cell. Hers as the owner: 0/150 in every cell. His
    side is asserted at 0.3 (150/150 his); at 3.0 -- a separation
    scripts/voice_enrol.pool_ok refuses to enrol -- it is reported and not
    asserted, as in test_voice_per_window."""
    v, enc = _verifier(tmp_path, monkeypatch)
    gate = _gate(tmp_path)
    world = Voices(seed=seed, apart=apart)
    _layout(v, world, migrated, [("mara", 10)])
    hers_as_owner = his_lost = his_as_mara = 0
    for i in range(150):
        c = _clip(4.0, 500.0 + i * 0.01)
        enc.teach(c, world.take("mara"))
        _st, d = _turn(v, gate, c, "unlock the door")
        hers_as_owner += (d.admit and d.who == "hunter")
        c2 = _clip(4.0, 520.0 + i * 0.01)
        enc.teach(c2, world.take("hunter"))
        _st2, d2 = _turn(v, gate, c2, "read my mail")
        his_lost += (not (d2.admit and d2.who == "hunter"))
        his_as_mara += (d2.who == "mara")
    _measured("grid seed=%d apart=%.1f migrated=%s" % (seed, apart, migrated),
              hers_as_owner=hers_as_owner, his_lost=his_lost,
              his_as_mara=his_as_mara)
    assert hers_as_owner == 0, "%d/150 of hers were him" % hers_as_owner
    if apart <= 0.3:
        assert his_lost == 0, "%d/150 of his were not him" % his_lost
        assert his_as_mara == 0, his_as_mara
