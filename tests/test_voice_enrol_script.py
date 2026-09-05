"""The enrolment script's refusals, and the instrument's.

THE MICROPHONE IS NEVER OPENED HERE. ``scripts/voice_enrol.py`` is the only
thing in this feature that may open it, and it is run by the person being
enrolled, at the keyboard. These tests exercise the decisions AROUND that --
the consent rule, the loudness floor, the cohesion and separation refusals,
the delete, and the instrument's flat refusal to suggest a bar with one person
enrolled -- all of which are pure functions or arithmetic over stored floats.

Both scripts are loaded by PATH, the way tests/test_faceenrol.py already loads
scripts/face_enrol.py: ``scripts/`` has no ``__init__.py``.
"""
from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest

from jarvis import voicegallery as vg
from tests.synthvoice import Voices

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, "scripts", "%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


voice_enrol = _load("voice_enrol")
voice_model_compare = _load("voice_model_compare")


@pytest.fixture()
def gal(tmp_path):
    return vg.VoiceGallery(root=tmp_path / "voice_gallery")


def _fill(gallery, world, label, n, **kw):
    for e in world.takes(label, n):
        gallery.add(label, e, **kw)


# ------------------------------------------------------------- no microphone
def test_importing_the_script_opens_nothing():
    """It imports jarvis.voicegallery and jarvis.config at module level and
    everything that can touch hardware INSIDE main(), so a test, a linter or
    ``--help`` never reaches the recorder."""
    import ast
    src = open(os.path.join(_HERE, "scripts", "voice_enrol.py")).read()
    top = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top.append(getattr(node, "module", "") or "")
            for alias in node.names:
                top.append(alias.name)
    joined = " ".join(top)
    assert "recorder" not in joined
    assert "speaker" not in joined
    assert "sounddevice" not in joined


# ------------------------------------------------------------ the loudness bar
@pytest.mark.parametrize("rms,ok", [(0.02, True), (0.004, True),
                                    (0.0039, False), (0.0, False)])
def test_a_silent_take_is_refused(rms, ok):
    got, why = voice_enrol.take_ok(rms)
    assert got is ok
    if not ok:
        assert "%.4f" % rms in why, "the refusal must print the number"


def test_an_unmeasurable_take_is_refused():
    assert voice_enrol.take_ok(float("nan"))[0] is False
    assert voice_enrol.take_ok(None)[0] is False


# -------------------------------------------------------- the pool's own bars
def test_a_collapsed_pool_is_refused_with_its_number(gal):
    one = Voices(seed=1).take("mara")
    vectors = [one + (i * 1e-5) for i in range(8)]
    ok, why = voice_enrol.pool_ok(gal, "mara", vectors)
    assert ok is False
    assert "median pairwise cosine" in why and "0.9" in why


def test_a_real_pool_passes(gal):
    ok, why = voice_enrol.pool_ok(gal, "mara", Voices(seed=2).takes("mara", 8))
    assert ok is True, why


def test_a_pool_too_close_to_somebody_already_enrolled_is_refused(gal):
    """The refusal that stops two people quietly breaking each other: stored
    anyway, EVERY verdict for BOTH of them would come back UNKNOWN on the
    margin, which is indistinguishable from the feature being broken."""
    world = Voices(seed=3, apart=6.0)      # deliberately confusable
    _fill(gal, world, "hunter", 14)
    ok, why = voice_enrol.pool_ok(gal, "mara", world.takes("mara", 8))
    assert ok is False
    assert "hunter" in why and "margin" in why
    assert "%.2f" % vg.MARGIN in why


def test_a_pool_far_enough_from_everybody_is_allowed(gal):
    world = Voices(seed=4, apart=0.3)
    _fill(gal, world, "hunter", 14)
    ok, why = voice_enrol.pool_ok(gal, "mara", world.takes("mara", 8))
    assert ok is True, why


def test_one_take_is_not_a_pool(gal):
    ok, _why = voice_enrol.pool_ok(gal, "mara", Voices(seed=5).takes("mara", 1))
    assert ok is False


# -------------------------------------------------------------- the consent
def _args(**kw):
    base = dict(json=False, auto=False, yes=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_consent_rule_is_imported_not_copied():
    """One rule, not two that can drift. The MECHANISM comes from
    face_enrol.consent; only the wording is this store's, because printing
    "128 numbers describing your FACE" over a voice enrolment would be a false
    statement of what is kept."""
    fe = voice_enrol.face_enrol()
    assert callable(fe.consent)
    assert "VOICE" in " ".join(voice_enrol.CONSENT_LINES)
    assert "FACE" not in " ".join(voice_enrol.CONSENT_LINES)
    assert "FACE" in " ".join(fe.CONSENT_LINES)


def test_the_consent_wording_says_no_recording_is_kept():
    text = " ".join(voice_enrol.CONSENT_LINES)
    assert "NO RECORDING IS KEPT" in text
    assert "192 numbers" in text
    assert "nothing leaves this" in text
    # And it must not sell itself as security.
    assert "not a password" in text and "not a lock" in text


def test_consent_cannot_be_given_through_json(tmp_path):
    fe = voice_enrol.face_enrol()
    ok, why = fe.consent("mara", "hunter", tmp_path, lambda _s: None,
                         _args(json=True), lines=voice_enrol.CONSENT_LINES)
    assert ok is False and "--json" in why


def test_consent_cannot_be_given_through_a_pipe(tmp_path):
    """A capture nobody has to press a key for is a capture nobody has to be
    present for. Under pytest neither stream is a terminal, which is exactly
    the condition being refused."""
    fe = voice_enrol.face_enrol()
    ok, why = fe.consent("mara", "hunter", tmp_path, lambda _s: None,
                         _args(), lines=voice_enrol.CONSENT_LINES)
    assert ok is False and "pipe" in why


def test_the_owner_needs_no_consent_prompt(tmp_path):
    fe = voice_enrol.face_enrol()
    ok, how = fe.consent("hunter", "hunter", tmp_path, lambda _s: None,
                         _args(), lines=voice_enrol.CONSENT_LINES)
    assert ok is True and how == "owner"


def test_the_face_script_still_prints_its_own_words(tmp_path):
    """The new parameter defaults to the face wording, so nothing about the
    face flow changed."""
    said = []
    fe = voice_enrol.face_enrol()
    fe.consent("mara", "hunter", tmp_path, said.append, _args(json=True))
    assert said == []          # --json refuses before printing, as before


# --------------------------------------------------------------- the driver
def test_status_on_an_empty_gallery_says_what_to_run(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", tmp_path / "vg")
    assert voice_enrol.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "nobody" in out and "--migrate" in out


def test_a_bad_label_is_refused_before_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", tmp_path / "vg")
    assert voice_enrol.main(["--label", "Mara"]) == 2
    assert "lowercase" in capsys.readouterr().err


def test_json_cannot_enrol_anybody(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", tmp_path / "vg")
    assert voice_enrol.main(["--label", "mara", "--json"]) == 2
    assert "consent" in capsys.readouterr().err


def test_enrolling_a_guest_stops_at_the_consent_prompt(tmp_path, monkeypatch,
                                                       capsys):
    """AND THAT IS THE TEST. Under pytest stdout is not a terminal, so the run
    refuses before the recorder is even imported -- which is why this file can
    exercise the enrolment path at all without a microphone.

    The owner is seeded into the gallery first: a guest on a box where he
    is enrolled nowhere is refused EARLIER (rc 5, owner_ready -- the
    fresh-box lockout, tests/test_voice_owner_lockout.py), and this test is
    about the consent step behind that."""
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", tmp_path / "vg")
    g = vg.VoiceGallery(root=tmp_path / "vg")
    _fill(g, Voices(seed=8, apart=0.3), "hunter", 14)
    g.save("the owner")
    assert voice_enrol.main(["--label", "mara"]) == 3
    assert "pipe" in capsys.readouterr().err


def test_delete_reports_incomplete_rather_than_claiming_success(
        tmp_path, monkeypatch, capsys):
    root = tmp_path / "vg"
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", root)
    g = vg.VoiceGallery(root=root)
    world = Voices(seed=6, apart=0.3)
    _fill(g, world, "hunter", 12)
    _fill(g, world, "mara", 8)
    g.save("both")
    assert voice_enrol.main(["--delete", "--label", "mara"]) == 0
    out = capsys.readouterr().out
    assert "complete" in out
    assert "mara" not in vg.VoiceGallery(root=root).disk_labels()


def test_migrate_needs_no_microphone(tmp_path, monkeypatch, capsys):
    import numpy as np
    root = tmp_path / "vg"
    src = tmp_path / "voiceprint.npz"
    arrays = {"emb_%04d" % i: np.asarray(v)
              for i, v in enumerate(Voices(seed=7).takes("hunter", 14))}
    arrays["_format"] = np.array([2])
    with open(src, "wb") as fh:
        np.savez(fh, **arrays)
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", root)
    monkeypatch.setattr(vg.PATHS, "VOICEPRINT", src)
    assert voice_enrol.main(["--migrate", "--label", "hunter"]) == 0
    out = capsys.readouterr().out
    assert "migrated 14" in out
    assert "rollback" in out
    assert vg.VoiceGallery(root=root).disk_labels() == ("hunter",)


# ------------------------------------------------------------- the instrument
def test_the_instrument_refuses_to_suggest_a_bar_with_one_label(
        tmp_path, monkeypatch, capsys):
    """THE REQUIREMENT. With one person enrolled this box holds no
    different-person evidence at all, and a bar picked from one voice's
    variation against itself is a bar picked from nothing -- which is how 0.40
    came to sit inside his own genuine band."""
    root = tmp_path / "vg"
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", root)
    g = vg.VoiceGallery(root=root)
    _fill(g, Voices(seed=8), "hunter", 14)
    g.save("him")
    assert voice_model_compare.main([]) == 0
    out = capsys.readouterr().out
    assert "NO THRESHOLD IS RECOMMENDED, AND NONE CAN BE." in out
    assert "different-person evidence at all" in out
    assert "0.377-0.397" in out, "the sentence must name the measured mistake"


def test_the_instrument_reports_pairs_once_two_people_are_enrolled(
        tmp_path, monkeypatch, capsys):
    root = tmp_path / "vg"
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", root)
    g = vg.VoiceGallery(root=root)
    world = Voices(seed=9, apart=0.3)
    _fill(g, world, "hunter", 14)
    _fill(g, world, "mara", 10)
    g.save("both")
    assert voice_model_compare.main(["--trials", "200"]) == 0
    out = capsys.readouterr().out
    assert "NO THRESHOLD IS RECOMMENDED" not in out
    assert "BETWEEN PEOPLE" in out
    assert "recommends nothing automatically" in out


def test_the_instrument_never_opens_a_device():
    import ast
    src = open(os.path.join(_HERE, "scripts", "voice_model_compare.py")).read()
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.append(getattr(node, "module", "") or "")
            names += [a.name for a in node.names]
    joined = " ".join(names)
    for forbidden in ("sounddevice", "recorder", "cv2", "torch",
                      "speechbrain", "soundfile"):
        assert forbidden not in joined, forbidden


def test_the_instrument_says_so_when_nothing_is_enrolled(tmp_path, monkeypatch,
                                                         capsys):
    monkeypatch.setattr(vg.PATHS, "VOICE_GALLERY", tmp_path / "vg")
    assert voice_model_compare.main([]) == 1
    assert "--migrate" in capsys.readouterr().out


# --------------------------------------------- the separation bar's quantity
def test_the_separation_check_measures_the_margin_not_the_centroids(gal):
    """THE REGIME THE CHECK EXISTS FOR. Two pools whose centroids sit at
    cosine ~0.71 -- under the old 0.80 centroid bar, so the old check waved
    them through -- and whose takes then fail the 0.20 margin on more than
    nine verdicts in ten. Measured 2026-09-04 on synthetic pools at his
    within-person spread (apart=2.0): centroid cosine 0.712, median margin
    0.15 / 0.12, unknown-rate 91% / 97%.

    Reverting pool_ok to the centroid comparison fails this test: the
    centroids are below 0.80 and it says ok."""
    world = Voices(seed=21, apart=2.0)
    him = world.takes("hunter", 14)
    _fill(gal, world, "hunter", 14)
    hers = world.takes("mara", 8)
    cc = vg.cosine(vg.centroid(him), vg.centroid(hers))
    assert cc < 1.0 - vg.MARGIN, "fixture drifted: the old check would fire"

    # What identify() itself would do to fresh takes of hers once stored.
    probe = vg.VoiceGallery()
    for e in gal.embeddings("hunter"):
        probe.add("hunter", e)
    for e in hers:
        probe.add("mara", e)
    fresh = world.takes("mara", 100)
    unknown = sum(1 for e in fresh
                  if probe.identify(e, 4.0, vg.ACCEPT_DEFAULT).who == "")
    assert unknown > 50, "fixture drifted: identify names her most of the time"

    ok, why = voice_enrol.pool_ok(gal, "mara", hers)
    assert ok is False, "the check passed a pool identify cannot name"
    assert "margin" in why and "hunter" in why
    assert "%.2f" % vg.MARGIN in why, "the refusal must print the bar"


def test_the_separation_margin_is_the_quantity_identify_uses(gal):
    """Leave-one-out margins and fresh-probe margins agree within 0.02 in
    median on the same pools, so the number printed is the number the
    verdicts will see, not a proxy for it."""
    world = Voices(seed=22, apart=1.5)
    him, hers = world.takes("hunter", 14), world.takes("mara", 8)
    m_hers, m_him = voice_enrol.separation_margins(hers, him)
    ch, cm = vg.centroid(him), vg.centroid(hers)
    import numpy as np
    fresh_hers = np.median([vg.cosine(e, cm) - vg.cosine(e, ch)
                            for e in world.takes("mara", 200)])
    fresh_him = np.median([vg.cosine(e, ch) - vg.cosine(e, cm)
                           for e in world.takes("hunter", 200)])
    assert abs(m_hers - fresh_hers) < 0.03, (m_hers, fresh_hers)
    assert abs(m_him - fresh_him) < 0.03, (m_him, fresh_him)


def test_the_status_report_prints_the_margin(gal):
    world = Voices(seed=23, apart=0.3)
    _fill(gal, world, "hunter", 14)
    _fill(gal, world, "mara", 8)
    lines = voice_enrol.separation_report(gal)
    assert len(lines) == 1
    assert "margin" in lines[0] and "%.2f" % vg.MARGIN in lines[0]
