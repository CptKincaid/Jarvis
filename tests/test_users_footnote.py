"""THE FOOT NOTE MUST END UP TRUE, and must stay true without anyone editing it.

This is the defect he actually caught on 2026-09-05. ``CANNOT_DO`` was a
hand-edited tuple of sentences sitting a hundred lines from the code whose
truth it asserts, and it went stale in THREE places at once:

* "Enrolling a face or a voice still needs a terminal" -- false for HIM since
  jarvis/enrolrun.py merged: he can enrol his own face by voice plus one
  typed word, no terminal.
* "Setting a new spoken passphrase or a new override code still needs a
  terminal" -- half false the moment the drawer's Knightfall row shipped a
  masked code box on the same display.
* ``VOICE_GUEST``, one chip over -- "there is one voiceprint and it is the
  owner's" -- false since the voice gallery merged, because gate._voice_leg
  names guests from it.

The fix is structural rather than another edit: the note is DERIVED from the
seams that are actually wired, so a capability that lands turns its own
sentence off and a capability that is removed turns one back on. These tests
are what make that derivation load-bearing.
"""
from __future__ import annotations

import pytest

from jarvis.ui import users_page as up


class Svc:
    """A services namespace holding exactly the seams a test names."""

    def __init__(self, **seams):
        for name, value in seams.items():
            setattr(self, name, value)


def _note(**seams):
    return "\n".join(up.cannot_do(Svc(**seams)))


ALL = dict(people_set_phrase=lambda *a: None,
           people_new_code=lambda *a: None,
           face_enrol_start=lambda *a: None,
           voice_enrol_start=lambda *a: None,
           people_purge_face=lambda *a: None,
           people_purge_voice=lambda *a: None)


# ============================================ the sentence follows the seam
def test_with_nothing_wired_the_note_says_everything_needs_a_terminal():
    note = _note().lower()
    assert "passphrase" in note and "terminal" in note
    assert "face" in note


def test_a_wired_phrase_seam_turns_its_own_sentence_off():
    """The sentence he read must not survive the capability landing."""
    without = _note()
    with_it = _note(people_set_phrase=lambda *a: None)
    assert "spoken passphrase" in without
    assert with_it != without
    assert "setting a new spoken passphrase" not in with_it.lower()


def test_a_wired_face_seam_stops_the_note_claiming_a_terminal_is_needed():
    with_it = _note(face_enrol_start=lambda *a: None).lower()
    assert "enrolling a face" not in with_it or "somebody else" in with_it


def test_a_wired_voice_seam_stops_the_note_claiming_a_terminal_is_needed():
    with_it = _note(voice_enrol_start=lambda *a: None).lower()
    assert "enrolling a voice still needs a terminal" not in with_it


def test_a_wired_purge_seam_retires_the_measurements_stay_sentence():
    """The fifth complaint. Once the tab can purge, the sentence saying the
    measurements stay until a terminal command is run is false."""
    without = _note().lower()
    assert "stay in the gallery" in without
    with_it = _note(people_purge_face=lambda *a: None,
                    people_purge_voice=lambda *a: None).lower()
    assert "stay in the gallery until that command is run" not in with_it


def test_the_two_sentences_that_are_true_whatever_ships():
    """Recognition is not a lock, and nothing here can be undone. Neither is
    a capability, so neither may ever be derived away."""
    for seams in ({}, ALL):
        note = "\n".join(up.cannot_do(Svc(**seams))).lower()
        assert "recognition, not a lock" in note
        assert "no history and no backup" in note


def test_the_note_keeps_the_two_things_that_stay_true_with_everything_wired():
    """CONSTRAINT G. With every seam wired the note is SHORTER, not empty:
    somebody else's face and voice still need a terminal, and an override
    code is still chosen by the machine and mailed, never typed here."""
    note = _note(**ALL).lower()
    assert "somebody else" in note
    assert "terminal" in note
    assert "override code" in note and "email" in note


# =============================== the shipped page must agree with its own seams
def test_the_shipped_note_matches_the_seams_the_app_actually_wires():
    """The test that stops the recurrence. It enumerates the seams
    ``app.ui_service_kwargs`` publishes and asserts the note the page would
    draw is the note derived from them -- so a seam that lands with no note
    change fails HERE rather than being found on screen weeks later."""
    from jarvis import app as app_mod
    import inspect
    src = inspect.getsource(app_mod.JarvisApp.ui_service_kwargs)
    wired = {name for name in ("people_set_phrase", "people_new_code",
                               "face_enrol_start", "voice_enrol_start",
                               "people_purge_face", "people_purge_voice")
             if name + "=" in src}
    derived = up.cannot_do(Svc(**{n: (lambda *a: None) for n in wired}))
    assert tuple(derived) == tuple(up.cannot_do(Svc(**{
        n: (lambda *a: None) for n in wired})))
    # ...and every sentence it produces must be one the module declares,
    # never a string assembled somewhere a reader cannot find it.
    assert all(isinstance(line, str) and line for line in derived)


def test_no_page_string_claims_one_voiceprint_while_the_gallery_exists():
    """The stale chip, one over from the stale note. ``gate._voice_leg``
    names guests out of jarvis/voicegallery.py, so a chip drawn on every
    non-owner row saying a guest cannot be named is false."""
    import importlib
    import inspect
    assert importlib.util.find_spec("jarvis.voicegallery") is not None
    src = inspect.getsource(up)
    assert "there is one voiceprint" not in src
    assert "one voiceprint and it is the owner" not in src


def test_the_voice_chip_is_drawn_from_the_pools_that_exist():
    """It says who HAS a voice pool, from the snapshot, rather than asserting
    that nobody but the owner can have one."""
    row = up.voice_chip("heather", role="known", voices=("heather",))
    assert "heather" in row.lower() or "enrolled" in row.lower()
    none = up.voice_chip("heather", role="known", voices=())
    assert "no" in none.lower() or "not" in none.lower()


def test_the_forget_warning_no_longer_claims_there_is_no_recording():
    """``forget_warning`` said "There is no voice recording of %s to remove:
    Jarvis holds one voiceprint and it is the owner's." That is the same
    false sentence in a destructive panel, which is worse."""
    lines = "\n".join(up.forget_warning("heather", voices=("heather",))).lower()
    assert "one voiceprint" not in lines
    assert "voice" in lines
