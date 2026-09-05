"""``scripts/enroll_voice.py`` had no idea the voice gallery existed.

C1-SECOND FROM THE ROUND-3 REVIEW, 2026-09-05. His gallery label is a COPY of
``voiceprint.npz`` carried across by ``voice_enrol.py --migrate``. Re-recording
the voiceprint leaves that copy behind: two disjoint fourteen-take pools of the
SAME man measure 0.927-0.934 of each other, under the 0.98 line at which Jarvis
stops reading a gallery pool as the voiceprint's. The script that re-records it
contained no mention of the gallery, of ``--status`` or of ``--reanchor``, so
the command he would naturally run to re-record his own voice silently put his
pool off its anchor and only the runtime log said so.

Two things changed. The runtime no longer refuses him for it -- a pool CARRIED
OUT OF voiceprint.npz is his whatever it measures, and only takes recorded at
the MICROPHONE under his name are disowned (tests/test_voice_owner_two_labels.py
measures the paired control). And this script now says what it is about to do,
refuses without ``--yes``, and names the repair on the way out.

NO MICROPHONE IS OPENED HERE, and this file could not open one if it wanted to:
everything asserted below is a pure function or a string. The recorder lives
inside ``main()`` behind a mic check this never reaches.
"""
from __future__ import annotations

import ast
import importlib.util
import os

import pytest

from jarvis import voicegallery as vg
from tests.synthvoice import Voices

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "_enroll_voice", os.path.join(_HERE, "scripts", "enroll_voice.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


enroll_voice = _load()


def test_the_warning_is_silent_when_no_pool_was_carried():
    assert enroll_voice.anchor_warning(()) == ""
    assert enroll_voice.anchor_warning([]) == ""


def test_the_warning_names_the_label_the_number_and_the_repair():
    text = enroll_voice.anchor_warning(("hunterpeyrovi",))
    assert "hunterpeyrovi" in text
    assert "%.2f" % vg.OWNER_POOL_COSINE in text
    assert "voice_enrol.py --reanchor" in text
    assert "--yes" in text


def test_main_refuses_to_re_record_over_a_carried_pool_without_yes(monkeypatch,
                                                                   capsys):
    """The refusal happens BEFORE anything is cleared or recorded, so a run
    that stops here has cost him nothing."""
    calls = []
    monkeypatch.setattr(enroll_voice, "gallery_labels",
                        lambda: ("hunterpeyrovi",))

    class _V:
        num_samples = 14
        threshold = 0.3

        def load(self):
            calls.append("load")

        def clear(self):
            calls.append("CLEARED")

    monkeypatch.setattr(enroll_voice, "SpeakerVerifier", lambda **kw: _V())
    monkeypatch.setattr(enroll_voice.sys, "argv", ["enroll_voice.py", "--reset"])
    rc = enroll_voice.main()
    out = capsys.readouterr()
    assert rc == 6
    assert "CLEARED" not in calls, "it cleared his voiceprint before refusing"
    assert "load_model" not in calls, "it spent a model load before refusing"
    assert "hunterpeyrovi" in out.out
    assert "--reanchor" in out.out
    assert "Nothing was recorded and nothing was cleared." in out.err


def test_status_names_the_gallery(monkeypatch, capsys):
    """--status printed every number about the voiceprint and nothing at all
    about the store that holds a copy of it."""
    monkeypatch.setattr(enroll_voice, "gallery_labels",
                        lambda: ("hunterpeyrovi",))

    class _V:
        num_samples = 0
        threshold = 0.3

        def load(self):
            pass

    monkeypatch.setattr(enroll_voice, "SpeakerVerifier", lambda **kw: _V())
    monkeypatch.setattr(enroll_voice.sys, "argv",
                        ["enroll_voice.py", "--status"])
    assert enroll_voice.main() == 0
    out = capsys.readouterr().out
    assert "voice gallery" in out and "hunterpeyrovi" in out
    assert "--reanchor" in out


def test_a_wedged_gallery_never_blocks_his_own_enrolment(monkeypatch, capsys):
    """The gallery is ADVICE here, never a door: this file's job is the
    voiceprint, and a store that cannot be opened must not stop him
    re-recording his own voice."""
    def boom():
        raise OSError("wedged")

    monkeypatch.setattr(enroll_voice.vg, "default_gallery", boom)
    assert enroll_voice.gallery_labels() == ()
    assert "could not read the voice gallery" in capsys.readouterr().out


def test_gallery_labels_reads_the_real_predicate(tmp_path, monkeypatch):
    """It asks the store the same question ``speaker._owner_pools`` asks --
    ``voiceprint_labels()`` -- rather than keeping a copy of the rule."""
    g = vg.VoiceGallery(root=tmp_path / "g")
    world = Voices(seed=61, apart=0.3)
    for e in world.takes("hunter", 14):
        g.add("hunterpeyrovi", e, src=vg.VOICEPRINT_SRC,
              note="migrated " + vg.VOICEPRINT_NOTE)
    for e in world.takes("mara", 10):
        g.add("mara", e, src="enrol")
    monkeypatch.setattr(enroll_voice.vg, "default_gallery", lambda: g)
    assert enroll_voice.gallery_labels() == ("hunterpeyrovi",)


def test_the_gallery_is_imported_at_module_level():
    """``jarvis.voicegallery`` opens nothing on import, so the advice above is
    available to --status and to the refusal without reaching for a device.
    (``jarvis.recorder`` is imported here too and always has been -- importing
    it does not open a microphone; this file's own import in this test is the
    proof.)"""
    src = open(os.path.join(_HERE, "scripts", "enroll_voice.py")).read()
    top = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            base = getattr(node, "module", "") or ""
            for alias in node.names:
                top.append(("%s.%s" % (base, alias.name)).strip("."))
    assert "jarvis.voicegallery" in top
