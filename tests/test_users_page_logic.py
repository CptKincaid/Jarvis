"""The USERS tab's logic, WITHOUT a window.

The KnightfallControl / RestartArm split, and for the same reason: a
control that can only be exercised by building a console cannot be tested
here at all. Everything in this file is pure -- no Tk, no display, no
registry of his, no code that was ever real.

WHAT IS PINNED HERE, in the order it matters:

  * every administrative action asks for the code when one is set, and
    REFUSES when it has not been given;
  * the unlock calls ``check_override_code`` and NEVER ``open_window`` --
    a users-tab unlock must not make the microphone start answering the
    room, which is what the drawer's Knightfall button deliberately does;
  * the entry is cleared BEFORE the service is called;
  * no code and no hash reaches a row, a toast, a log record or a thread
    name;
  * the bootstrap case still lets a first owner be made -- but not over a
    file that failed to parse;
  * adding a non-owner takes their consent;
  * forgetting asks twice and says what it cannot undo.
"""
import logging

import pytest

from jarvis import consent as cs
from jarvis import gate as gt
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry
from jarvis.ui import users_page as up

FAKE_CODE = "zzz-not-a-real-code-zzz"


class Svc:
    """A stand-in app. Records what it was handed, never what was typed."""

    def __init__(self, unlock=None, snapshot=None):
        self.calls = []
        self._unlock = unlock
        self._snapshot = snapshot or {}

    def people_unlock(self, code):
        # The control hands the plaintext over ONCE. The test records that
        # it was called and its length only -- never the text.
        self.calls.append(("unlock", len(code or "")))
        if self._unlock is None:
            return True, "unlocked, sir"
        return self._unlock(code)

    def people_snapshot(self):
        self.calls.append(("snapshot", 0))
        return self._snapshot


def _control(services=None, entry="", **kw):
    """A control whose seams are lists, so every effect is inspectable."""
    box = {"text": entry}
    toasts, later, spawned = [], [], []
    ctl = up.UsersUnlockControl(
        services,
        read=lambda: box["text"],
        clear=lambda: box.update(text=""),
        toast=lambda line, kind="info": toasts.append((line, kind)),
        later=lambda fn: later.append(fn) or fn(),
        spawn=lambda fn, *a: spawned.append((fn, a)),
        **kw)
    return ctl, box, toasts, spawned


# ================================================== the lock is the tab's own
def test_the_unlock_is_a_window_that_relocks_by_itself():
    clock = {"t": 100.0}
    lock = up.Lock(window_s=120.0, clock=lambda: clock["t"])
    assert lock.locked() is True and lock.remaining() == 0.0
    lock.unlock()
    assert lock.locked() is False
    clock["t"] += 119.0
    assert lock.locked() is False
    assert lock.remaining() == pytest.approx(1.0)
    clock["t"] += 2.0
    assert lock.locked() is True
    assert lock.remaining() == 0.0


def test_a_successful_action_re_arms_the_window():
    clock = {"t": 0.0}
    lock = up.Lock(window_s=120.0, clock=lambda: clock["t"])
    lock.unlock()
    clock["t"] = 110.0
    lock.touch()
    clock["t"] = 200.0
    assert lock.locked() is False, "each action should re-arm the window"
    clock["t"] = 240.0
    assert lock.locked() is True


def test_leaving_the_tab_relocks_at_once():
    lock = up.Lock(window_s=120.0, clock=lambda: 0.0)
    lock.unlock()
    assert lock.locked() is False
    lock.lock()
    assert lock.locked() is True


# ============================================ what the lock does NOT do
def test_the_unlock_never_opens_a_voice_window(monkeypatch, tmp_path):
    """THE SHARPEST EDGE IN THIS LANE. app.knightfall_code calls
    gate.open_window because Knightfall's whole job is to open a five
    minute VOICE admission window. If this tab copied it, pressing Unlock
    would make the microphone answer whoever is standing there -- a
    capability the terminal tool never grants for an administrative
    action. This is invisible if you get it wrong, so it is a test."""
    src = open(up.__file__).read()
    assert "open_window" not in src, \
        "the users tab must never open a voice admission window"
    assert "HOW_CODE" not in src


def test_the_unlock_uses_the_gate_s_own_attempt_counter(tmp_path):
    """A second in-process Attempts silently doubles the budget from ten
    tries per five minutes to twenty. The terminal tool builds its own
    because it is a separate PROCESS; the tab is inside the app."""
    reg = Registry(path=tmp_path / "people.json")
    reg.add_person(Person(label="alderman", role=ROLE_OWNER))
    gate = gt.OwnerGate(registry=reg, owner="alderman")
    seen = []
    who, why = up.check_code(gate, "no", check=lambda r, c, attempts=None:
                             (seen.append(attempts) or ("", "nope")))
    assert who == "" and why == "nope"
    assert seen == [gate.code_attempts], \
        "the tab must pass the gate's existing counter, not a fresh one"


# =========================================== the entry, and what it must not do
def test_the_entry_is_cleared_before_the_service_is_called():
    """Not after it returns. check_override_code is a scrypt KDF looped
    over every owner with a code, so it is real work -- a typed code must
    not sit on screen while it runs, nor survive the tab being left."""
    seen = {}
    ctl, box, _toasts, spawned = _control(Svc(), entry="something typed")

    def watcher(code):
        seen["entry_was"] = box["text"]
        return True, "unlocked"

    ctl.services._unlock = watcher
    assert ctl.unlock_pressed() == "started"
    assert box["text"] == "", "the entry is cleared on the press"
    fn, args = spawned[0]
    fn(*args)
    assert seen["entry_was"] == "", \
        "the entry still held the code while the KDF ran"


def test_the_press_hands_the_code_over_exactly_once():
    ctl, _box, _toasts, spawned = _control(Svc(), entry="abcdef")
    ctl.unlock_pressed()
    fn, args = spawned[0]
    fn(*args)
    assert ctl.services.calls == [("unlock", 6)]


def test_an_empty_entry_is_refused_without_calling_anything():
    ctl, _box, toasts, spawned = _control(Svc(), entry="   ")
    assert ctl.unlock_pressed() == "empty"
    assert spawned == []
    assert ctl.services.calls == []
    assert toasts and "code" in toasts[0][0].lower()


def test_nothing_typed_can_reach_a_toast_or_a_log(caplog):
    """The toast carries check_override_code's own refusals and nothing
    else. On an exception it is a FIXED sentence, never str(exc): the
    Knightfall rule, which exists because a transport once quoted a mail
    body back and put a freshly rotated code in the line the drawer
    showed."""
    secret = "zzz-typed-in-anger-zzz"

    def boom(code):
        raise RuntimeError("the code %s was rejected by %s" % (code, code))

    caplog.set_level(logging.DEBUG)
    ctl, _box, toasts, spawned = _control(Svc(unlock=boom), entry=secret)
    ctl.unlock_pressed()
    fn, args = spawned[0]
    fn(*args)
    assert toasts
    line = toasts[-1][0]
    assert secret not in line
    assert line == up.UNLOCK_FAILED
    assert secret not in caplog.text
    for rec in caplog.records:
        assert secret not in str(rec.getMessage())
        assert secret not in str(getattr(rec, "args", "") or "")


def test_the_worker_thread_is_named_for_the_feature():
    names = []

    class Thread:
        def __init__(self, target=None, args=(), daemon=None, name=None):
            names.append(name)
            self.start = lambda: None

    import jarvis.ui.users_page as mod
    old = mod.threading.Thread
    mod.threading.Thread = Thread
    try:
        ctl = up.UsersUnlockControl(Svc(), read=lambda: "abcdef",
                                    clear=lambda: None,
                                    toast=lambda *a, **k: None,
                                    later=lambda fn: fn())
        ctl.unlock_pressed()
    finally:
        mod.threading.Thread = old
    assert names == ["users-unlock"]
    assert all("abcdef" not in (n or "") for n in names)


def test_a_refusal_is_shown_in_the_gate_s_own_words():
    ctl, _box, toasts, spawned = _control(
        Svc(unlock=lambda c: (False, "that is not a code I know")),
        entry="wrong")
    ctl.unlock_pressed()
    fn, args = spawned[0]
    fn(*args)
    assert toasts[-1] == ("that is not a code I know", "warn")


def test_an_unwired_service_says_so_instead_of_pretending():
    ctl, _box, toasts, spawned = _control(object(), entry="abcdef")
    assert ctl.unlock_pressed() == "not wired"
    assert spawned == []
    assert toasts[-1][1] == "warn"


# ======================================= every write asks for the code
@pytest.mark.parametrize("action", up.GUARDED)
def test_every_administrative_action_is_refused_while_locked(action):
    """Reading the list is not guarded -- the CLI runs `list` before
    _authorise for exactly that reason, and the list holds no secret."""
    lock = up.Lock(window_s=120.0, clock=lambda: 0.0)
    ok, why = up.may(action, gt.ADMIN_CODE, lock)
    assert ok is False
    assert "code" in why.lower()
    lock.unlock()
    assert up.may(action, gt.ADMIN_CODE, lock)[0] is True


def test_reading_the_list_never_asks_for_anything():
    lock = up.Lock(window_s=120.0, clock=lambda: 0.0)
    assert "list" not in up.GUARDED
    assert up.may("list", gt.ADMIN_CODE, lock)[0] is True


def test_an_owner_who_set_no_code_is_not_asked_for_one():
    lock = up.Lock(window_s=120.0, clock=lambda: 0.0)
    assert lock.locked() is True
    for action in up.GUARDED:
        assert up.may(action, gt.ADMIN_NOCODE, lock)[0] is True


def test_the_bootstrap_still_lets_a_first_owner_be_made():
    lock = up.Lock(window_s=120.0, clock=lambda: 0.0)
    assert up.may("add", gt.ADMIN_FIRST, lock)[0] is True


def test_a_broken_registry_refuses_every_action_including_the_first_owner():
    lock = up.Lock(window_s=120.0, clock=lambda: 0.0)
    lock.unlock()
    for action in up.GUARDED:
        ok, why = up.may(action, gt.ADMIN_REFUSE, lock)
        assert ok is False
        assert "read" in why or "repair" in why.lower()


# ============================================== what the bootstrap offers
def test_a_missing_file_offers_exactly_one_action():
    plan = up.bootstrap_plan("missing", "/tmp/nowhere/people.json")
    assert plan.may_create is True
    assert plan.first_owner_only is True
    assert "first owner" in plan.line.lower()


def test_a_registry_that_names_no_owner_keeps_the_rows_it_has():
    plan = up.bootstrap_plan("ownerless", "/tmp/nowhere/people.json")
    assert plan.may_create is True and plan.first_owner_only is True


def test_a_corrupt_registry_offers_no_create_button_at_all():
    """A GUI button is far easier to press by accident than a typed
    subcommand, and this press would write a one-row registry over rows
    that failed to parse."""
    for kind in ("malformed", "unreadable"):
        plan = up.bootstrap_plan(kind, "/tmp/nowhere/people.json")
        assert plan.may_create is False
        assert "/tmp/nowhere/people.json" in plan.line
        assert plan.first_owner_only is False


def test_a_healthy_registry_is_not_a_bootstrap():
    plan = up.bootstrap_plan("", "/tmp/nowhere/people.json")
    assert plan.may_create is False and plan.first_owner_only is False
    assert plan.line == ""


# ================================================== the rows on the page
def _snapshot(**kw):
    people = kw.pop("people", None)
    if people is None:
        people = [
            Person(label="alderman", name="Alderman", role=ROLE_OWNER,
                   voice=True, face="alderman", face_dim=128,
                   code_hash="zzz-a-hash", enrolled_at="2026-01-01T00:00:00",
                   consent="owner").redacted(),
            Person(label="pemberton", name="Pemberton", role=ROLE_KNOWN,
                   face="missing-from-gallery", consent="typed").redacted(),
        ]
    out = {"people": people, "gate_line": "owner-gate: SHADOW -- 1 owner",
           "fault_kind": "", "path": "/tmp/nowhere/people.json",
           "gallery": ["alderman"]}
    out.update(kw)
    return out


def test_a_row_carries_no_hash_in_any_form():
    """Built from Person.redacted() and never from a Person. A hash cannot
    reach a widget even by accident, and the row says only set / not set --
    not the value, not the length, not a prefix, not a masked stand-in."""
    rows = up.rows_from(_snapshot())
    blob = repr(rows)
    assert "zzz-a-hash" not in blob
    owner = rows[0]
    assert owner.has_code is True and owner.has_phrase is False
    assert "set" in owner.code_text and "not set" in rows[1].code_text
    for row in rows:
        for text in (row.code_text, row.phrase_text, row.detail):
            assert "hash" not in text.lower()


def test_the_voice_chip_tells_the_truth_about_one_voiceprint():
    """MEASURED in gate._voice_leg: there is ONE voiceprint pool and the
    leg returns the OWNER's label on a match, whoever spoke. A voice
    tickbox on a guest row would let him create a row that lies."""
    rows = up.rows_from(_snapshot())
    assert "owner only" in rows[0].voice_text
    assert rows[1].voice_text == up.VOICE_GUEST
    assert "cannot" in up.VOICE_GUEST.lower()


def test_a_face_pointer_at_nothing_is_flagged_amber():
    rows = up.rows_from(_snapshot())
    assert rows[0].face_known is True
    assert rows[1].face_known is False
    assert rows[1].face_tone == "warn"
    assert rows[0].face_tone == "ok"


def test_a_stale_face_pointer_explains_itself_and_tints_nothing_else():
    """FOUND BY RENDERING AT 1040x1760 AND LOOKING, 2026-09-05.

    ``detail`` is ONE label, and it was drawn in WARN whenever the face
    pointer was stale -- so "phrase: not set", "code: not set" and the
    enrolment date all went amber for that person and stayed muted for
    everybody else, though nothing about them differed. This tree has an
    explicit rule that amber means a fault and nothing else
    (tests/test_ui_classic_frozen.py::
    test_amber_on_the_holo_page_means_a_fault_and_nothing_else), and five
    non-faults wearing the fault colour breaks it.

    So the fault gets its OWN line, which can also say what is wrong --
    a tint never could.
    """
    rows = up.rows_from(_snapshot())
    good, stale = rows[0], rows[1]
    assert good.face_why == ""
    assert stale.face_tone == "warn"
    # it NAMES the pointer that is dangling, and says what it costs
    assert "missing-from-gallery" in stale.face_why
    assert "gallery" in stale.face_why
    # and the detail line itself stays a plain list of chips
    assert "missing-from-gallery" in stale.detail


def test_a_person_with_no_face_at_all_is_not_a_fault():
    """"face: none" is a choice, not a dangling pointer; it must not
    acquire an amber explanation."""
    rows = up.rows_from(_snapshot(people=[
        Person(label="alderman", name="Alderman", role=ROLE_OWNER,
               voice=True, face="alderman", face_dim=128,
               code_hash="zzz-a-hash").redacted(),
        Person(label="pemberton", name="Pemberton",
               role=ROLE_KNOWN).redacted()]))
    assert rows[1].face_tone == "muted"
    assert rows[1].face_why == ""


def test_the_sole_owner_cannot_be_forgotten_or_demoted_and_is_told_why():
    rows = up.rows_from(_snapshot())
    owner = rows[0]
    assert owner.can_forget is False
    assert "only owner" in owner.forget_why
    assert owner.can_change_role is False
    assert "only owner" in owner.role_why
    guest = rows[1]
    assert guest.can_forget is True and guest.forget_why == ""
    assert guest.can_change_role is True


def test_two_owners_may_each_be_demoted():
    people = [Person(label="alderman", role=ROLE_OWNER).redacted(),
              Person(label="brightwell", role=ROLE_OWNER).redacted()]
    rows = up.rows_from(_snapshot(people=people))
    assert all(r.can_forget and r.can_change_role for r in rows)


def test_an_empty_registry_makes_no_rows_and_does_not_raise():
    assert up.rows_from({"people": []}) == ()
    assert up.rows_from({}) == ()
    assert up.rows_from(None) == ()


# ==================================================== forgetting somebody
def test_forgetting_asks_twice():
    """A press ARMS and says what it destroys; the confirmation is the
    person's own label, typed. Not a yes/no: a mis-click on a list row
    must not be able to destroy a row, and typing the label names WHO is
    being destroyed in a way "Are you sure?" does not."""
    clock = {"t": 0.0}
    arm = up.ForgetArm(window_s=20.0, clock=lambda: clock["t"])
    assert arm.armed_for == ""
    assert arm.press("pemberton") == "armed"
    assert arm.armed_for == "pemberton"
    assert arm.confirm("pemberton", "no") == "refused"
    assert arm.armed_for == "pemberton", "a wrong answer does not disarm"
    assert arm.confirm("pemberton", "pemberton") == "forget"
    assert arm.armed_for == ""


def test_an_arm_that_went_cold_is_not_a_confirmation():
    clock = {"t": 0.0}
    arm = up.ForgetArm(window_s=20.0, clock=lambda: clock["t"])
    arm.press("pemberton")
    clock["t"] = 21.0
    assert arm.confirm("pemberton", "pemberton") == "expired"


def test_arming_one_person_never_confirms_another():
    arm = up.ForgetArm(window_s=20.0, clock=lambda: 0.0)
    arm.press("pemberton")
    assert arm.confirm("alderman", "alderman") == "expired"
    assert arm.armed_for == "pemberton"


def test_the_warning_says_what_survives_and_that_it_cannot_be_undone():
    """MEASURED in identity.Registry.forget: it removes the ROW and
    nothing else. The face gallery entry is untouched, and there is no
    voice pool of theirs to scrub because there is only one and it is
    his."""
    lines = "\n".join(up.forget_warning("pemberton"))
    assert "cannot be undone" in lines.lower()
    assert "gallery" in lines.lower()
    assert "voice" in lines.lower()
    assert "pemberton" in lines
    cmd = up.forget_face_command("pemberton")
    assert "--delete" in cmd and "--label pemberton" in cmd
    assert "face_enrol.py" in cmd


# ======================================================== adding a person
def test_adding_a_non_owner_takes_their_consent_on_the_console():
    """Somebody else's agreement, in the shared words, recorded as a
    console consent rather than as a terminal one."""
    shown = []
    ok, how = up.take_consent("pemberton", owner="alderman",
                              show=shown.append, ask=lambda: "pemberton",
                              mapped=lambda: True)
    assert (ok, how) == (True, cs.HOW_CONSOLE)
    assert any("pemberton" in line for line in shown)
    assert not any("128" in line for line in shown), \
        "adding a row stores no face measurement; the words must not claim it"


def test_a_refused_consent_adds_nobody():
    ok, how = up.take_consent("pemberton", owner="alderman",
                              show=lambda _l: None, ask=lambda: "",
                              mapped=lambda: True)
    assert ok is False and "nothing was written" in how


def test_the_owner_adding_himself_asks_nobody_for_permission():
    ok, how = up.take_consent("alderman", owner="alderman",
                              show=lambda _l: None, ask=lambda: "",
                              mapped=lambda: True)
    assert (ok, how) == (True, "owner")


def test_a_second_owner_has_to_name_the_first():
    """identity.add_person's confirm_existing_owner exists precisely so a
    slip of the hand cannot mint one. A pre-filled dropdown would defeat
    it entirely, so the tab asks for the label to be typed."""
    ok, why = up.owner_confirmed(ROLE_OWNER, ["alderman"], "")
    assert ok is False and "alderman" in why
    assert up.owner_confirmed(ROLE_OWNER, ["alderman"], "alderman")[0] is True
    assert up.owner_confirmed(ROLE_OWNER, ["alderman"], "Alderman")[0] is True
    assert up.owner_confirmed(ROLE_OWNER, [], "")[0] is True     # bootstrap
    assert up.owner_confirmed(ROLE_KNOWN, ["alderman"], "")[0] is True


def test_a_label_the_gallery_could_not_store_is_refused_before_anything():
    assert up.label_fault("pemberton") == ""
    assert up.label_fault("") != ""
    assert up.label_fault("Pemberton") != ""      # LABEL_RX is lower-case
    assert up.label_fault("has space") != ""
    assert up.label_fault("x" * 40) != ""


# ============================================== the page says what it cannot do
def test_the_standing_note_never_claims_to_be_a_lock():
    text = " ".join(up.CANNOT_DO).lower()
    assert "terminal" in text
    assert "camera" in text or "face" in text
    for word in ("secure", "locked out", "prevents"):
        assert word not in text
    assert "recognition, not a lock" in " ".join(up.CANNOT_DO).lower() or \
        "not a lock" in text


# ============================== the page must not load a face detector
def test_the_command_builder_needs_no_vision_module_at_all(monkeypatch):
    """FOUND ON THE PHOTO RIG, 2026-09-05. The page asks for ONE STRING --
    the command that deletes somebody's face measurements -- and that
    import pulled in jarvis.faceenrol, which imports jarvis.facedetect and
    jarvis.visionrig. The rig's lens blocker refused it outright and the
    render died.

    jarvis/enrolentry.py's own docstring claimed "there is no import in
    this file that could" open a device, and at import time that was not
    true. The heavy imports moved inside the four functions that need a
    gallery; command_line and to_clipboard need neither.
    """
    import importlib
    import sys

    blocked = ("cv2", "jarvis.camera", "jarvis.eye", "jarvis.facedetect",
               "jarvis.facemodels", "jarvis.facegallery", "jarvis.visionrig",
               "jarvis.faceenrol")

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name in blocked:
                raise ImportError("blocked for this test: %s" % name)
            return None

    def hidden(name):
        return (name in blocked or name == "jarvis.enrolentry"
                or any(name.startswith(b + ".") for b in blocked))

    # HAND THEM ALL BACK. cv2 is a PACKAGE: popping the top-level name and
    # leaving cv2.typing behind meant the next ``import cv2`` re-ran
    # cv2/__init__.py against its own stale children and raised
    # "partially initialized module". The whole subtree goes, and every
    # object comes back -- pinned by the test below.
    saved = {n: m for n, m in list(sys.modules.items()) if hidden(n)}
    for name in saved:
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(".")
    sys.meta_path.insert(0, Blocker())
    try:
        ee = importlib.import_module("jarvis.enrolentry")
        cmd = ee.command_line("pemberton", delete=True)
        # read while the blocker is still up: this is the assertion
        leaked = [n for n in blocked if n in sys.modules]
    finally:
        sys.meta_path.pop(0)
        for name in [n for n in list(sys.modules) if hidden(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    assert "--delete" in cmd and "--label pemberton" in cmd
    assert leaked == [], leaked


def test_the_blocker_dance_puts_every_vision_module_back(monkeypatch):
    """THE TEST ABOVE MUST NOT COST THE SUITE A MODULE.

    FOUND IN THE FULL SUITE, 2026-09-05: it popped ``cv2`` out of
    ``sys.modules`` and never put it back. ``cv2`` is a PACKAGE, so
    dropping the top-level name while ``cv2.typing`` and the rest stayed
    behind left the next ``import cv2`` re-executing ``cv2/__init__.py``
    against its own stale children -- "partially initialized module 'cv2'
    has no attribute 'mat_wrapper'". tests/test_webapp.py's QR decode ran
    later in the same process and was the one that died, which is why the
    failure looked like somebody else's.

    So this pins the hygiene rather than the symptom: after that test has
    run, the vision modules are exactly the objects they were before it.
    """
    import sys

    cv2 = pytest.importorskip("cv2")
    def names():
        return {n: sys.modules[n] for n in list(sys.modules)
                if n == "cv2" or n.startswith("cv2.")}

    before = names()
    assert before, "cv2 was not loaded, so this pins nothing"

    test_the_command_builder_needs_no_vision_module_at_all(monkeypatch)

    after = names()
    assert set(after) == set(before), (
        "left behind: %s / lost: %s"
        % (sorted(set(after) - set(before)), sorted(set(before) - set(after))))
    assert after["cv2"] is cv2, "cv2 was replaced by a rebuilt module"


def test_the_page_builds_the_same_command_the_voice_path_hands_over():
    """ONE builder, not two: the tab's Copy button and "forget Heather's
    face" must not drift into two different commands."""
    from jarvis import enrolentry as ee
    assert up.forget_face_command("pemberton") == \
        ee.command_line("pemberton", delete=True)


def test_no_press_reaches_a_write_without_asking_the_gate_first():
    """THE WHOLE PROMISE OF THE LOCK, checked structurally rather than one
    button at a time. Every method that calls ``_write`` must call
    ``_guard`` first -- the app seam refuses a BROKEN registry but it does
    not know about this page's unlock dwell, so a press that skipped the
    guard would simply be performed."""
    import ast
    tree = ast.parse(open(up.__file__).read())
    page = [n for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "UsersPage"][0]
    callers = []
    for fn in page.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = ast.unparse(fn)
        if "self._write(" in body and fn.name != "_write":
            callers.append((fn.name, body))
    assert {n for n, _b in callers} == {"_forget_confirm", "_role_pressed",
                                        "_role_confirm", "_create_pressed"}, \
        [n for n, _b in callers]
    for name, body in callers:
        assert "self._guard(" in body, name
        # ...and the guard comes FIRST, before anything is handed over
        assert body.index("self._guard(") < body.index("self._write("), name
