"""The two secrets the USERS tab may touch, and the two rules that bound them.

WHAT IS BEING PINNED, in his words: "Setting a new spoken passphrase or a new
override code still needs a terminal" was the foot note he read on 2026-09-05
and asked to be made untrue. It becomes untrue in two DIFFERENT ways, and the
difference is the whole of this file:

* THE SPOKEN PASSPHRASE may be typed here. Its threat model already accepts
  being overheard -- ``scripts/jarvis_people.py`` says so out loud -- so a
  masked box on his own console makes it no worse.
* THE OVERRIDE CODE MAY NOT BE TYPED HERE. Its entire value is that it never
  travels a microphone and is never chosen; the tab may only ASK for a fresh
  one to be mailed, through the rotate that already exists.

NO SECRET IS EVER REAL IN THIS FILE. Every literal below is a marked
non-secret, and the tests assert those markers never appear in a log record,
a returned line or a saved file.

NOTHING HERE OPENS A DEVICE, A MAILBOX OR HIS PEOPLE BOOK: every registry is
built in tmp_path and every mail transport is a stand-in.
"""
from __future__ import annotations

import json
import logging

import pytest

from jarvis import app as app_mod
from jarvis import gate as gt
from jarvis import passphrase as pp
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry
from jarvis.ui import users_page as up

# Marked non-secrets. If either string ever reaches a log or a line, the
# grep that finds it is finding a leak and not a fixture.
NOT_A_CODE = "zzz-not-a-real-code-zzz"
NOT_A_PHRASE = "zzz not a real passphrase at all zzz"


class Stub:
    """Just enough JarvisApp for the secret seams: a gate and nothing else."""

    def __init__(self, gate, assistant=None):
        self.gate = gate
        self.assistant = assistant

    _people_registry = app_mod.JarvisApp._people_registry
    _people_unlock_left = app_mod.JarvisApp._people_unlock_left
    _people_open_unlock = app_mod.JarvisApp._people_open_unlock
    _people_write = app_mod.JarvisApp._people_write
    _people_write_now = app_mod.JarvisApp._people_write_now
    _people_decide = app_mod.JarvisApp._people_decide
    people_unlock = app_mod.JarvisApp.people_unlock
    people_snapshot = app_mod.JarvisApp.people_snapshot
    # The two under test.
    people_set_phrase = app_mod.JarvisApp.people_set_phrase
    people_new_code = app_mod.JarvisApp.people_new_code
    _gate_mode_quietly = app_mod.JarvisApp._gate_mode_quietly
    knightfall_new_code = app_mod.JarvisApp.knightfall_new_code
    _knightfall_rotate = app_mod.JarvisApp._knightfall_rotate
    _people_gate = app_mod.JarvisApp._people_gate


def _app(tmp_path, *, code=False, known=False):
    path = tmp_path / "people.json"
    reg = Registry(path=path)
    reg.add_person(Person(label="alderman", name="Alderman", role=ROLE_OWNER,
                          voice=True, face="alderman", face_dim=128))
    if known:
        reg.add_person(Person(label="pemberton", name="Pemberton",
                              role=ROLE_KNOWN, consent="typed"))
    if code:
        reg.set_secret("alderman", "code_hash", pp.hash_secret(NOT_A_CODE))
    assert reg.save()
    gate = gt.OwnerGate(registry=Registry.load(path), owner="alderman",
                        get_option=lambda k, d=None: d)
    return Stub(gate), path


# ================================================ A: every write asks the gate
def test_set_phrase_is_refused_while_the_code_is_owed(tmp_path):
    """CONSTRAINT A. A code is set and no dwell is open, so the write is
    refused with the SAME sentence add/forget use -- which is what makes the
    page raise its unlock row rather than inventing a second refusal."""
    app, path = _app(tmp_path, code=True)
    before = path.read_bytes()
    ok, line = app.people_set_phrase("alderman", pp.hash_secret(NOT_A_PHRASE))
    assert ok is False
    assert line == gt.ADMIN_CODE_OWED
    assert path.read_bytes() == before


def test_new_code_is_refused_while_the_code_is_owed(tmp_path):
    """CONSTRAINT A, the other half. Asking for a fresh code to be MAILED is
    still a registry write, so it asks the same question."""
    app, path = _app(tmp_path, code=True)
    before = path.read_bytes()
    ok, line = app.people_new_code()
    assert ok is False
    assert line == gt.ADMIN_CODE_OWED
    assert path.read_bytes() == before


def test_the_guarded_list_has_the_new_actions_and_not_the_dead_one(tmp_path):
    """``set_face`` is dead -- it is named in its own definition and nowhere
    else in the tree -- and every new door must be in the list instead."""
    assert "set_face" not in up.GUARDED
    for action in ("set_phrase", "new_code", "face_enrol", "voice_enrol",
                   "purge_face", "purge_voice"):
        assert action in up.GUARDED


def test_the_page_refuses_a_guarded_press_while_locked():
    lock = up.Lock(window_s=0.0)
    for action in ("set_phrase", "new_code", "face_enrol", "voice_enrol",
                   "purge_face", "purge_voice"):
        ok, why = up.may(action, gt.ADMIN_CODE, lock)
        assert ok is False, action
        assert why == up.LOCKED_LINE


# =========================================== A: the write goes through the seam
def test_set_phrase_writes_a_hash_and_never_a_plaintext(tmp_path):
    app, path = _app(tmp_path)
    hashed = pp.hash_secret(NOT_A_PHRASE)
    ok, line = app.people_set_phrase("alderman", hashed)
    assert ok is True
    raw = path.read_text()
    assert NOT_A_PHRASE not in raw
    assert "zzz" not in line
    stored = Registry.load(path).person("alderman")
    assert stored.phrase_hash == hashed
    assert pp.check_secret(NOT_A_PHRASE, stored.phrase_hash)


def test_set_phrase_refuses_a_non_owner_and_writes_nothing(tmp_path):
    """``identity.Registry.set_secret`` already refuses this; the seam must
    not become the way around it."""
    app, path = _app(tmp_path, known=True)
    before = path.read_bytes()
    ok, line = app.people_set_phrase("pemberton", pp.hash_secret(NOT_A_PHRASE))
    assert ok is False
    assert "owner" in line
    assert path.read_bytes() == before


def test_set_phrase_refuses_an_empty_hash(tmp_path):
    """An empty hash would CLEAR his way back in while looking like a set."""
    app, path = _app(tmp_path)
    before = path.read_bytes()
    ok, _line = app.people_set_phrase("alderman", "")
    assert ok is False
    assert path.read_bytes() == before


def test_a_successful_set_phrase_rearms_the_dwell_on_the_code_leg(tmp_path):
    """PINNED WITH ITS REASON. ``_people_write`` re-arms only where a code was
    actually presented, so a run of edits is one code. Setting a phrase is
    the same kind of act as forgetting somebody and is treated the same way;
    the bootstrap leg still re-arms nothing."""
    app, _path = _app(tmp_path, code=True)
    ok, _line = app.people_unlock(NOT_A_CODE)
    assert ok is True
    ok, _line = app.people_set_phrase("alderman", pp.hash_secret(NOT_A_PHRASE))
    assert ok is True
    assert app._people_unlock_left() > 0.0


def test_the_bootstrap_leg_does_not_open_a_dwell(tmp_path):
    """No code is set at all, so the write is allowed -- and must still leave
    the dwell shut, or a code set seconds later falls into a free window."""
    app, _path = _app(tmp_path)
    ok, _line = app.people_set_phrase("alderman", pp.hash_secret(NOT_A_PHRASE))
    assert ok is True
    assert app._people_unlock_left() == 0.0


# ================================== B: he can never lock himself out (the code)
class FakeMail:
    """The mail module's two seams, and nothing that opens a socket."""

    def __init__(self, msgid="<zzz@example.invalid>"):
        self.msgid = msgid
        self.sent = []

    def mail_accounts(self, _assistant):
        return [{"address": "him@example.invalid"}]

    def send(self, *_a, **_kw):
        return self.msgid


def _send_notice(fake):
    def _send(_account, _subject, body, smtp=None, mail=None):
        fake.sent.append(body)
        if not fake.msgid:
            return ""
        return fake.msgid
    return _send


def test_a_mail_that_never_went_leaves_the_old_code_working(tmp_path,
                                                            monkeypatch):
    """CONSTRAINT B. No Message-ID came back, so nothing is stored and the
    code he already has still opens the tab."""
    app, path = _app(tmp_path, code=True)
    app._people_open_unlock()
    fake = FakeMail(msgid="")
    monkeypatch.setattr(app_mod.outbox, "send_notice", _send_notice(fake))
    ok, line = app.people_new_code(mail=fake)
    assert "zzz-not-a-real-code" not in line
    stored = Registry.load(path).person("alderman")
    assert pp.check_secret(NOT_A_CODE, stored.code_hash)
    assert ok is False or "stays as it is" in line or "will not work" in line


def test_a_store_that_failed_puts_the_old_hash_back_in_memory(tmp_path,
                                                              monkeypatch):
    """CONSTRAINT B. The mail went and the save did not: the inbox copy is
    dead, and the one in his head must still work IN MEMORY as it does on
    disk. There is no instant with no working code."""
    app, path = _app(tmp_path, code=True)
    app._people_open_unlock()
    fake = FakeMail()
    monkeypatch.setattr(app_mod.outbox, "send_notice", _send_notice(fake))
    monkeypatch.setattr(Registry, "save", lambda self: False)
    ok, line = app.people_new_code(mail=fake)
    assert ok is False
    assert "zzz" not in line
    person = app.gate.registry.person("alderman")
    assert pp.check_secret(NOT_A_CODE, person.code_hash)


def test_the_new_code_never_appears_in_the_line_or_the_log(tmp_path,
                                                           monkeypatch,
                                                           caplog):
    """CONSTRAINT C. The generated code goes to his inbox and nowhere else --
    not into the toast, not into a log record at any level."""
    app, _path = _app(tmp_path, code=True)
    app._people_open_unlock()
    fake = FakeMail()
    monkeypatch.setattr(app_mod.outbox, "send_notice", _send_notice(fake))
    with caplog.at_level(logging.DEBUG):
        ok, line = app.people_new_code(mail=fake)
    assert ok is True
    assert fake.sent, "the body is what carries the code"
    code = fake.sent[0].split("\n")[0].strip()
    assert code and code not in line
    for record in caplog.records:
        assert code not in record.getMessage()


def test_setting_the_first_code_can_turn_the_gate_on_and_the_page_says_so(
        tmp_path):
    """THE SHARPEST TRAP, and it is a UI action reaching the microphone.

    ``gate._mode_unsafe`` downgrades enforce to SHADOW while no owner row
    carries a code, because a wrong verdict would otherwise have no way back
    in. The moment a code exists that downgrade lifts -- so one press of
    "send me a new code" on a box configured ``owner.mode=enforce`` starts
    refusing turns. This pins the MECHANISM, and the caption test below pins
    that he is told before he presses.
    """
    path = tmp_path / "people.json"
    reg = Registry(path=path)
    reg.add_person(Person(label="alderman", name="Alderman", role=ROLE_OWNER))
    assert reg.save()
    opts = {"owner.mode": "enforce"}
    gate = gt.OwnerGate(registry=Registry.load(path), owner="alderman",
                        get_option=lambda k, d=None: opts.get(k, d))
    assert gate.effective_mode() == gt.MODE_SHADOW
    reg2 = Registry.load(path)
    reg2.set_secret("alderman", "code_hash", pp.hash_secret(NOT_A_CODE))
    assert reg2.save()
    gate.reload()
    assert gate.effective_mode() == gt.MODE_ENFORCE


def test_the_code_panel_warns_about_the_mode_before_the_press():
    """CONSTRAINT B, said on screen. The caption must name the change BEFORE
    he presses, not explain it afterwards."""
    lines = "\n".join(up.code_panel_lines({"to": "h•••@x.invalid"},
                                          mode="enforce", has_code=False))
    assert "shadow" in lines and "enforce" in lines


def test_the_code_panel_disables_itself_with_no_mailbox():
    plan = up.code_plan({"to": "", "setup": "add an account"}, mode="shadow",
                        has_code=True)
    assert plan.enabled is False
    assert "nowhere to send" in plan.line or "no mail account" in plan.line
    plan = up.code_plan({"to": "h•@x.invalid"}, mode="shadow",
                        has_code=True)
    assert plan.enabled is True


def test_there_is_no_free_text_code_setter_anywhere_on_the_page():
    """DELIBERATELY NOT BUILT. A code he TYPES is a code he chose, will
    reuse and will type again; a MAILED one is generated from an alphabet
    with no 0/o and rotates on every use. The absence is the feature, so it
    is asserted rather than left to be added by somebody later."""
    import inspect
    src = inspect.getsource(up)
    assert "people_set_code" not in src
    assert "set_secret" not in src, ("the page must not reach a generic "
                                     "secret setter; it may set a PHRASE")


# ================== C: a typed secret never persists, never travels, never logs
class Entry:
    """A stand-in Tk entry that records WHEN it was emptied."""

    def __init__(self, text=""):
        self.text = text
        self.cleared_at = None

    def get(self):
        return self.text

    def clear(self):
        self.text = ""
        self.cleared_at = len(_ORDER)
        _ORDER.append("clear")


_ORDER: list = []


class SecretSvc:
    """Reads the entries back AT THE MOMENT the service is called."""

    def __init__(self, boxes, answer=(True, "Set, sir."), raises=None):
        self.boxes = boxes
        self.answer = answer
        self.raises = raises
        self.saw = None
        self.calls = []

    def people_set_phrase(self, label, hashed):
        self.saw = [b.get() for b in self.boxes]
        self.calls.append((label, hashed))
        _ORDER.append("service")
        if self.raises is not None:
            raise self.raises
        return self.answer


def _control(first, second, svc, label="alderman"):
    later = []
    toasts = []
    ctl = up.UsersSecretControl(
        svc, label=label,
        read=lambda: (first.get(), second.get()),
        clear=lambda: (first.clear(), second.clear()),
        toast=lambda line, kind="info": toasts.append((line, kind)),
        later=lambda fn: later.append(fn),
        spawn=lambda fn, *a: fn(*a))
    return ctl, later, toasts


def test_both_boxes_are_empty_before_the_service_is_called():
    """CONSTRAINT C, and the order is the assertion. Hashing is a scrypt KDF
    at N=2^14 -- real work -- and a typed secret must not sit on screen while
    it runs, nor still be there if the tab is left mid-call."""
    _ORDER.clear()
    a, b = Entry(NOT_A_PHRASE), Entry(NOT_A_PHRASE)
    svc = SecretSvc([a, b])
    ctl, later, _toasts = _control(a, b, svc)
    assert ctl.pressed() == "started"
    for fn in later:
        fn()
    assert svc.saw == ["", ""], "the entries were still full at the call"
    assert _ORDER.index("clear") < _ORDER.index("service")


def test_the_service_is_handed_a_hash_and_never_the_typed_words():
    a, b = Entry(NOT_A_PHRASE), Entry(NOT_A_PHRASE)
    svc = SecretSvc([a, b])
    ctl, later, _toasts = _control(a, b, svc)
    ctl.pressed()
    for fn in later:
        fn()
    assert svc.calls, "nothing was called"
    _label, hashed = svc.calls[0]
    assert NOT_A_PHRASE not in hashed
    assert pp.check_secret(NOT_A_PHRASE, hashed)


def test_a_mismatch_calls_nothing_and_empties_both_boxes():
    a, b = Entry(NOT_A_PHRASE), Entry(NOT_A_PHRASE + " x")
    svc = SecretSvc([a, b])
    ctl, later, toasts = _control(a, b, svc)
    assert ctl.pressed() == "mismatch"
    assert svc.calls == []
    assert a.get() == "" and b.get() == ""
    assert toasts and "match" in toasts[0][0]
    assert "zzz" not in toasts[0][0]


def test_a_short_phrase_calls_nothing_and_says_the_length():
    a, b = Entry("short"), Entry("short")
    svc = SecretSvc([a, b])
    ctl, later, toasts = _control(a, b, svc)
    assert ctl.pressed() == "too short"
    assert svc.calls == []
    assert a.get() == "" and b.get() == ""
    assert str(pp.MIN_PHRASE_LEN) in toasts[0][0]


def test_no_log_record_at_any_level_carries_the_typed_words(caplog):
    """Across a success, a mismatch, a short phrase, and the case that bit
    before: a SERVICE THAT RAISES WITH THE VALUE IN ITS MESSAGE. Only the
    exception's type name may be recorded -- a traceback carries the
    exception's own str()."""
    _ORDER.clear()
    with caplog.at_level(logging.DEBUG):
        for first, second, svc_kw in (
                (NOT_A_PHRASE, NOT_A_PHRASE, {}),
                (NOT_A_PHRASE, NOT_A_PHRASE + " x", {}),
                ("short", "short", {}),
                (NOT_A_PHRASE, NOT_A_PHRASE,
                 {"raises": RuntimeError("the phrase %s was rejected"
                                         % NOT_A_PHRASE)})):
            a, b = Entry(first), Entry(second)
            svc = SecretSvc([a, b], **svc_kw)
            ctl, later, _t = _control(a, b, svc)
            ctl.pressed()
            for fn in later:
                fn()
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert NOT_A_PHRASE not in blob
    assert "zzz" not in blob
    assert "RuntimeError" in blob, "the TYPE is what a log is for here"


def test_the_control_thread_is_named_for_the_field_never_the_value():
    import threading
    seen = []
    real = threading.Thread

    class Spy(real):
        def __init__(self, *a, **kw):
            seen.append(kw.get("name") or "")
            super().__init__(*a, **kw)

    a, b = Entry(NOT_A_PHRASE), Entry(NOT_A_PHRASE)
    svc = SecretSvc([a, b])
    ctl = up.UsersSecretControl(
        svc, label="alderman", read=lambda: (a.get(), b.get()),
        clear=lambda: (a.clear(), b.clear()),
        toast=lambda *x, **k: None, later=lambda fn: None)
    threading.Thread = Spy
    try:
        ctl.pressed()
    finally:
        threading.Thread = real
    assert seen and all("zzz" not in n for n in seen)
    assert any("phrase" in n for n in seen)


def test_an_unwired_service_says_so_and_still_empties_the_boxes():
    a, b = Entry(NOT_A_PHRASE), Entry(NOT_A_PHRASE)
    ctl, later, toasts = _control(a, b, object())
    assert ctl.pressed() == "not wired"
    assert a.get() == "" and b.get() == ""
