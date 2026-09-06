"""Starting and stopping a face or a voice enrolment FROM THE TAB.

NO CAMERA IS OPENED AND NO MICROPHONE IS OPENED IN THIS FILE, and that is not
an inconvenience being worked around -- it is the rule the feature exists
under. Every claim below is made from numbers and injected devices: what the
preflight returned, whether the lease callable was ever asked for the lens,
which label the run was built with, and whether a parked run was cleared.

THE THREE THINGS BEING PINNED:

* CONSTRAINT A -- a button that starts a lens or a microphone is a GALLERY
  write and asks ``gate.admin_gate`` exactly like ``forget`` does. The voice
  path to the identical write does not ask, and that asymmetry is stated in
  the foot note rather than closed by loosening the button.
* CONSTRAINT D -- somebody else's face or voice needs THEIR consent, which a
  window cannot take, so a guest row hands over a command and never starts a
  run. The seam forces the owner's label; it is never read from a row.
* CONSTRAINT F -- the face run's progress channel stays AUDIO. A button may
  start it; it must not become a thing that only works if he can see a
  screen.
"""
from __future__ import annotations

import sys

import pytest

from jarvis import app as app_mod
from jarvis import gate as gt
from jarvis import passphrase as pp
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry

NOT_A_CODE = "zzz-not-a-real-code-zzz"


class FakeRun:
    """What a started run looks like from outside: it is stoppable and it
    says whether it is alive. Nothing here captures anything."""

    def __init__(self):
        self.aborted = []
        self.alive = True

    def abort(self, reason=""):
        self.aborted.append(reason)
        self.alive = False

    @property
    def running(self):
        return self.alive


class Services:
    """The namespace the console publishes; a run is parked on it."""

    def __init__(self, worker=object()):
        self.preview_worker = worker
        self.preview_lease = self._lease
        self.enrol_run = None
        self.voice_run = None
        self.leased = []

    def _lease(self, on):
        self.leased.append(bool(on))


class Stub:
    def __init__(self, gate, services, cfg=None, sensing=None):
        self.gate = gate
        self.services = services
        self.assistant = cfg or {}
        self.sensing = sensing

    _people_registry = app_mod.JarvisApp._people_registry
    _people_unlock_left = app_mod.JarvisApp._people_unlock_left
    _people_open_unlock = app_mod.JarvisApp._people_open_unlock
    _people_gate = app_mod.JarvisApp._people_gate
    _people_decide = app_mod.JarvisApp._people_decide
    _stop_run = app_mod.JarvisApp._stop_run
    _enrol_say = app_mod.JarvisApp._enrol_say
    face_enrol_start = app_mod.JarvisApp.face_enrol_start
    face_enrol_stop = app_mod.JarvisApp.face_enrol_stop
    voice_enrol_start = app_mod.JarvisApp.voice_enrol_start
    voice_enrol_stop = app_mod.JarvisApp.voice_enrol_stop


def _app(tmp_path, *, code=False, services=None):
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
    return Stub(gate, services if services is not None else Services()), path


# ============================================ A: the button asks for the code
def test_face_start_is_refused_while_the_code_is_owed_and_opens_no_lens(
        tmp_path, monkeypatch):
    svc = Services()
    app, _path = _app(tmp_path, code=True, services=svc)
    called = []
    monkeypatch.setattr(app_mod, "_face_preflight",
                        lambda *a, **k: called.append(1) or {"ok": True})
    ok, line = app.face_enrol_start()
    assert ok is False
    assert line == gt.ADMIN_CODE_OWED
    assert called == [], "the preflight ran before the gate did"
    assert svc.leased == [], "a refused press asked for the lens"
    assert svc.enrol_run is None


def test_voice_start_is_refused_while_the_code_is_owed(tmp_path):
    svc = Services()
    app, _path = _app(tmp_path, code=True, services=svc)
    ok, line = app.voice_enrol_start()
    assert ok is False
    assert line == gt.ADMIN_CODE_OWED
    assert svc.voice_run is None


def test_stopping_is_never_gated(tmp_path):
    """STOPPING IS THE SAFE DIRECTION and takes the widest door -- the same
    rule ``Commander._enrol_control`` already follows for the spoken stop. A
    code owed must never be the reason a lens stays open."""
    svc = Services()
    run = FakeRun()
    svc.enrol_run = run
    app, _path = _app(tmp_path, code=True, services=svc)
    ok, _line = app.face_enrol_stop()
    assert ok is True
    assert run.aborted


# ================================ A/D: the label is forced, never read from a row
def test_the_face_seam_takes_no_label_at_all(tmp_path):
    """CONSTRAINT D, made structural. ``enrolrun`` forces
    ``identity.owner_label`` by construction; a seam that ACCEPTED a label
    would be the door round it, so it does not have one."""
    import inspect
    sig = inspect.signature(app_mod.JarvisApp.face_enrol_start)
    assert [p for p in sig.parameters if p not in ("self", "now")] == []
    sig = inspect.signature(app_mod.JarvisApp.voice_enrol_start)
    assert [p for p in sig.parameters if p not in ("self", "now")] == []


def test_a_guest_row_gets_a_command_and_never_a_run():
    """CONSTRAINT D. Enrolling Heather stores a measurement of Heather, so
    Heather reads what is kept and types her own name, at a terminal."""
    from jarvis.ui import users_page as up
    line = up.enrol_command("heather", kind="face")
    assert "--label" in line and "heather" in line
    voice = up.enrol_command("heather", kind="voice")
    assert "voice_enrol" in voice and "heather" in voice


def test_building_a_guest_command_imports_no_vision_module():
    """``jarvis/enrolentry.py``'s import-lightness is an existing property --
    importing it used to pull the whole vision stack in and the Users page
    found that. This is the test that keeps it."""
    from jarvis.ui import users_page as up
    heavy = ("cv2", "torch", "jarvis.facedetect", "jarvis.camera",
             "jarvis.facegallery")
    before = {m for m in heavy if m in sys.modules}
    up.enrol_command("heather", kind="face")
    up.enrol_command("heather", kind="voice")
    after = {m for m in heavy if m in sys.modules}
    assert after == before


# ============================================= the preflight owns the refusals
def test_every_preflight_refusal_answers_its_own_sentence_and_no_lease(
        tmp_path, monkeypatch):
    svc = Services()
    app, _path = _app(tmp_path, services=svc)
    for reason, reply in (("curfew", "the camera curfew is on"),
                          ("offline", "the camera stays shut"),
                          ("identity off", "Face identity is switched off"),
                          ("models", "models aren't on disk")):
        monkeypatch.setattr(
            app_mod, "_face_preflight",
            lambda *a, _r=reason, _p=reply, **k: {
                "ok": False, "reply": _p, "reason": _r})
        ok, line = app.face_enrol_start()
        assert ok is False
        assert line == reply
    assert svc.leased == []
    assert svc.enrol_run is None


def test_a_second_press_while_a_run_is_live_starts_nothing(tmp_path,
                                                           monkeypatch):
    svc = Services()
    first = FakeRun()
    svc.enrol_run = first
    app, _path = _app(tmp_path, services=svc)
    ok, line = app.face_enrol_start()
    assert ok is False
    assert "already" in line.lower()
    assert svc.enrol_run is first


def test_a_started_run_is_parked_so_the_stop_button_can_reach_it(tmp_path,
                                                                 monkeypatch):
    svc = Services()
    app, _path = _app(tmp_path, services=svc)
    made = FakeRun()
    made.start = lambda: True
    monkeypatch.setattr(app_mod, "_face_preflight",
                        lambda *a, **k: {"ok": True, "reply": "", "reason": ""})
    monkeypatch.setattr(app_mod, "_build_face_run", lambda *a, **k: made)
    ok, line = app.face_enrol_start()
    assert ok is True
    assert svc.enrol_run is made
    assert "camera" in line.lower()
    ok, _ = app.face_enrol_stop()
    assert made.aborted


def test_a_run_that_will_not_start_is_unparked(tmp_path, monkeypatch):
    """A parked run that never began would refuse every later press with
    "already running" and there would be nothing to stop."""
    svc = Services()
    app, _path = _app(tmp_path, services=svc)
    dud = FakeRun()
    dud.start = lambda: False
    monkeypatch.setattr(app_mod, "_face_preflight",
                        lambda *a, **k: {"ok": True, "reply": "", "reason": ""})
    monkeypatch.setattr(app_mod, "_build_face_run", lambda *a, **k: dud)
    ok, _line = app.face_enrol_start()
    assert ok is False
    assert svc.enrol_run is None


# ===================================== F: the face run's channel stays audio
def test_the_button_changes_nothing_about_how_the_run_reports(tmp_path,
                                                              monkeypatch):
    """CONSTRAINT F. Two of the five stations are "turn your head the other
    way" and "sit back", where he can see no screen. The seam must build the
    run with the SAME spoken/earcon channel the voice path builds, so a press
    replaces the typed word and nothing else."""
    svc = Services()
    app, _path = _app(tmp_path, services=svc)
    built = {}
    monkeypatch.setattr(app_mod, "_face_preflight",
                        lambda *a, **k: {"ok": True, "reply": "", "reason": ""})

    def _build(**kw):
        built.update(kw)
        run = FakeRun()
        run.start = lambda: True
        return run

    monkeypatch.setattr(app_mod, "_build_face_run", _build)
    app.face_enrol_start()
    for seam in ("say", "card", "earcon", "lease", "clipboard"):
        assert callable(built.get(seam)), seam


def test_the_foot_note_says_the_spoken_door_does_not_ask_for_the_code():
    """CONSTRAINT A creates an asymmetry: a button under GUARDED is STRICTER
    than the wake word plus one typed word, which reaches the identical
    write. That is the right direction, and he is told rather than left to
    find it."""
    from jarvis.ui import users_page as up
    note = "\n".join(up.ENROL_ASYMMETRY)
    assert "code" in note and ("out loud" in note or "voice" in note
                              or "asking" in note)
