"""Removing somebody's FACE and VOICE measurements from the tab.

This is the fifth complaint, one bullet under the one he caught: forgetting
somebody removes their row and leaves their measurements on the disk until a
terminal command is run.

THE WHOLE RISK IS THE REPORT, NOT THE DELETE. ``facegallery.purge_label`` is
already the strongest code in this tree -- no file is destroyed until the
embeddings it held, MINUS hers, have been read back off disk from the new
generation, by name AND by sample count -- and ``voicegallery.purge_label`` is
the same method over the same layout. Neither needs replacing. What is new is
a surface that could ROUND AN INCOMPLETE PURGE UP TO "done", which is exactly
the class of defect he just caught in the foot note.

So every sentence here is built from a value the call RETURNED -- ``complete``,
``still_holding``, ``unreadable``, ``foreign``, ``not_carried`` -- and never
from "the call did not raise".
"""
from __future__ import annotations

import pytest

from jarvis import enrolentry as ee
from jarvis import gate as gt
from jarvis import passphrase as pp
from jarvis import app as app_mod
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry
from jarvis.ui import users_page as up

NOT_A_CODE = "zzz-not-a-real-code-zzz"


def _clean(label="heather", **kw):
    out = {"label": label, "generations_with": [1, 2, 3], "removed": 3,
           "unreadable": [], "foreign": [], "foreign_models": (),
           "not_carried": [], "still_holding": [], "tmp_removed": 0,
           "generation": 4, "loaded": 0, "left": 0, "labels_left": (),
           "labels_on_disk": ("alderman",), "reason": "", "complete": True}
    out.update(kw)
    return out


# ============================================ the report may not round up
def test_a_complete_purge_says_what_actually_happened_in_numbers():
    line = ee.purge_line(_clean(), kind="face")
    assert "3" in line
    assert "heather" in line.lower()
    assert "complete" in line.lower() or "removed" in line.lower()


def test_an_incomplete_purge_never_says_done():
    """``complete`` False means she is still on the disk. The tab must repeat
    the refusal rather than round it up -- the invariant is refusing to lie
    and a surface that smoothed that over would be the bug."""
    line = ee.purge_line(_clean(complete=False, unreadable=[2],
                                still_holding=[2], removed=2), kind="face")
    low = line.lower()
    assert "still" in low
    assert "2" in line
    assert not low.startswith("removed."), line
    assert "could not be read" in low or "unreadable" in low


def test_a_generation_another_model_wrote_is_named_and_left_alone():
    line = ee.purge_line(_clean(complete=False, foreign=[1],
                                foreign_models=("sface",),
                                still_holding=[1]), kind="face")
    assert "sface" in line.lower() or "another model" in line.lower()


def test_a_generation_holding_a_bystander_is_named_and_left_alone():
    line = ee.purge_line(_clean(complete=False, not_carried=[3],
                                still_holding=[3]), kind="face")
    low = line.lower()
    assert "3" in line
    assert "somebody else" in low or "bystander" in low or "carried" in low


def test_an_incomplete_purge_names_the_command_that_finishes_it():
    line = ee.purge_line(_clean(complete=False, still_holding=[2]),
                         kind="face")
    assert "--delete" in line or "--status" in line


def test_a_face_purge_says_it_does_not_touch_a_voice():
    """A face purge that leaves a voice pool behind is the same half-truth
    the foot note is being corrected for."""
    assert "voice" in ee.purge_line(_clean(), kind="face").lower()
    assert "face" in ee.purge_line(_clean(), kind="voice").lower()


def test_a_purge_that_found_nobody_says_so_rather_than_claiming_a_delete():
    line = ee.purge_line(_clean(generations_with=[], removed=0,
                                complete=True), kind="face")
    low = line.lower()
    assert "nothing" in low or "no " in low
    assert "3" not in line


# ============================================= the seam is gated and honest
class Stub:
    def __init__(self, gate, report=None, raises=None):
        self.gate = gate
        self._report = report
        self._raises = raises
        self.asked = []

    _people_registry = app_mod.JarvisApp._people_registry
    _people_unlock_left = app_mod.JarvisApp._people_unlock_left
    _people_open_unlock = app_mod.JarvisApp._people_open_unlock
    _people_gate = app_mod.JarvisApp._people_gate
    _people_decide = app_mod.JarvisApp._people_decide
    _purge = app_mod.JarvisApp._purge
    _stop_preview_worker = app_mod.JarvisApp._stop_preview_worker
    people_purge_face = app_mod.JarvisApp.people_purge_face
    people_purge_voice = app_mod.JarvisApp.people_purge_voice


def _app(tmp_path, *, code=False, **kw):
    path = tmp_path / "people.json"
    reg = Registry(path=path)
    reg.add_person(Person(label="alderman", name="Alderman", role=ROLE_OWNER))
    reg.add_person(Person(label="heather", name="Heather", role=ROLE_KNOWN,
                          consent="typed"))
    if code:
        reg.set_secret("alderman", "code_hash", pp.hash_secret(NOT_A_CODE))
    assert reg.save()
    gate = gt.OwnerGate(registry=Registry.load(path), owner="alderman",
                        get_option=lambda k, d=None: d)
    return Stub(gate, **kw), path


def test_a_purge_is_refused_while_the_code_is_owed_and_opens_no_gallery(
        tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(app_mod, "_open_face_gallery",
                        lambda: opened.append(1))
    app, _path = _app(tmp_path, code=True)
    ok, line = app.people_purge_face("heather")
    assert ok is False
    assert line == gt.ADMIN_CODE_OWED
    assert opened == []


def test_an_incomplete_purge_answers_not_ok(tmp_path, monkeypatch):
    """``ok`` is ``complete`` and nothing else. A toast built on "it did not
    raise" is the failure this whole file exists to refuse."""
    class Gal:
        def purge_label(self, label, reason=""):
            return _clean(label, complete=False, still_holding=[2],
                          unreadable=[2])

    monkeypatch.setattr(app_mod, "_open_face_gallery", lambda: Gal())
    app, _path = _app(tmp_path)
    ok, line = app.people_purge_face("heather")
    assert ok is False
    assert "still" in line.lower()


def test_a_complete_purge_answers_ok(tmp_path, monkeypatch):
    class Gal:
        def purge_label(self, label, reason=""):
            return _clean(label)

    monkeypatch.setattr(app_mod, "_open_face_gallery", lambda: Gal())
    app, _path = _app(tmp_path)
    ok, line = app.people_purge_face("heather")
    assert ok is True
    assert "heather" in line.lower()


def test_the_live_preview_staleness_is_said_or_the_worker_is_stopped(
        tmp_path, monkeypatch):
    """The running preview holds a gallery IN MEMORY -- ``enrolrun`` stops the
    worker after a save for exactly this reason. A purge from the tab must do
    the same or say "restart to apply"; a console that goes on recognising
    somebody who was just destroyed is a lie with a face on it."""
    class Gal:
        def purge_label(self, label, reason=""):
            return _clean(label)

    stopped = []

    class Svc:
        preview_worker = type("W", (), {"stop": lambda self: stopped.append(1)})()

    monkeypatch.setattr(app_mod, "_open_face_gallery", lambda: Gal())
    app, _path = _app(tmp_path)
    app.services = Svc()
    ok, line = app.people_purge_face("heather")
    assert ok is True
    assert stopped or "restart" in line.lower()


# ================================================== the two-gesture confirm
def test_the_confirmation_is_her_own_label_typed_and_not_a_yes():
    arm = up.ForgetArm(window_s=100.0, clock=lambda: 0.0)
    arm.press("heather")
    assert arm.confirm("heather", "yes") == "refused"
    assert arm.confirm("heather", "y") == "refused"
    assert arm.confirm("heather", "Heather") == "forget"


def test_a_wrong_answer_does_not_disarm():
    arm = up.ForgetArm(window_s=100.0, clock=lambda: 0.0)
    arm.press("heather")
    assert arm.confirm("heather", "hether") == "refused"
    assert arm.armed_for == "heather"


def test_the_purge_warning_says_what_goes_and_what_survives():
    lines = "\n".join(up.purge_warning("heather", kind="face"))
    low = lines.lower()
    assert "heather" in low
    assert "cannot be undone" in low or "no backup" in low
    assert "voice" in low, "a face purge must name what it does NOT touch"
    assert "type heather" in low
