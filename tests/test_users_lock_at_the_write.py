"""THE LOCK IS A DECISION AT THE WRITE, NOT A MEMORY OF ONE.

The users tab used to decide whether an override code was owed from a
snapshot read when the tab was OPENED, and nothing below the UI re-decided
it. So the sequence the page itself teaches --

    create the first owner in the tab (no code yet, so the tab is open and
    must stay open, or a fresh install is a brick)
        -> the foot note says a new code still needs a terminal
        -> he goes and sets one
        -> he comes back to the SAME open tab

-- left every administrative action ungated: ``app.people_forget`` simply
succeeded, no code was ever asked for, and the unlock row was not even
drawn, because the only thing that re-read the file was ``show()``.

The rule is not the UI's to invent. It is
``scripts/jarvis_people.py::_authorise``'s, which is
``gate.admin_gate``'s: no registry or no owner, anybody at this keyboard
may make the FIRST owner; an owner with no code, allowed and said out
loud; an owner WITH a code, the code is asked for through
``gate.check_override_code``, never echoed, and rate limited on the
CODE's own counter so burning the spoken passphrase's attempts cannot
close the break-glass.

EVERY CLOCK HERE IS INJECTED. Nothing in this file reads the wall clock or
``time.monotonic``: the dwell is driven by an explicit ``now=``.

No registry of his is opened. Every registry is built in a tmp_path with
invented people, an invented code and an invented passphrase.
"""
import json
import logging

import pytest

from jarvis import app as app_mod
from jarvis import gate as gt
from jarvis import passphrase as pp
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry

# Invented. Neither of these is his, and neither may ever be printed.
FAKE_CODE = "zzz-not-a-real-code-zzz"
FAKE_PHRASE = "zzz not a real passphrase zzz"


class Stub:
    """Just enough JarvisApp for the people seams: a gate and nothing else."""

    def __init__(self, gate):
        self.gate = gate

    _people_registry = app_mod.JarvisApp._people_registry
    _people_unlock_left = app_mod.JarvisApp._people_unlock_left
    _people_open_unlock = app_mod.JarvisApp._people_open_unlock
    _people_write = app_mod.JarvisApp._people_write
    _people_write_now = app_mod.JarvisApp._people_write_now
    people_admin_state = app_mod.JarvisApp.people_admin_state
    people_relock = app_mod.JarvisApp.people_relock
    people_snapshot = app_mod.JarvisApp.people_snapshot
    people_unlock = app_mod.JarvisApp.people_unlock
    people_add = app_mod.JarvisApp.people_add
    people_set_role = app_mod.JarvisApp.people_set_role
    people_forget = app_mod.JarvisApp.people_forget


def _registry(path, *, known=False, code=False, phrase=False):
    reg = Registry(path=path)
    reg.add_person(Person(label="alderman", name="Alderman", role=ROLE_OWNER,
                          voice=True, face="alderman", face_dim=128))
    if known:
        reg.add_person(Person(label="pemberton", name="Pemberton",
                              role=ROLE_KNOWN, consent="typed"))
    if code:
        assert reg.set_secret("alderman", "code_hash",
                              pp.hash_secret(FAKE_CODE))[0]
    if phrase:
        assert reg.set_secret("alderman", "phrase_hash",
                              pp.hash_secret(FAKE_PHRASE))[0]
    assert reg.save()
    return reg


def _app(tmp_path, *, code=False, known=False, phrase=False, empty=False):
    path = tmp_path / "people.json"
    if not empty:
        _registry(path, known=known, code=code, phrase=phrase)
    gate = gt.OwnerGate(registry=Registry.load(path), owner="alderman",
                        get_option=lambda k, d=None: d)
    return Stub(gate), path


def _set_a_code_at_a_terminal(path):
    """What he actually does: ``jarvis_people.py set-code`` in another
    PROCESS, which this one cannot see until it re-reads the file."""
    reg = Registry.load(path)
    assert reg.set_secret("alderman", "code_hash",
                          pp.hash_secret(FAKE_CODE))[0]
    assert reg.save()


# ============================================ nothing below the UI enforced
@pytest.mark.parametrize("call", [
    lambda a: a.people_forget("pemberton", now=100.0),
    lambda a: a.people_set_role("pemberton", ROLE_OWNER,
                                confirm_existing_owner="alderman", now=100.0),
    lambda a: a.people_add(label="brightwell", role=ROLE_KNOWN,
                           consent="console", now=100.0),
])
def test_every_administrative_seam_refuses_with_no_unlock(tmp_path, call):
    """THREE RED ROWS BEFORE THE FIX. Called with no unlock at all, on a
    registry whose owner has set a code, each of these simply succeeded."""
    app, path = _app(tmp_path, code=True, known=True)
    before = path.read_text(encoding="utf-8")
    ok, why = call(app)
    assert ok is False
    assert why == gt.ADMIN_CODE_OWED
    assert path.read_text(encoding="utf-8") == before, "the file was written"


def test_the_stale_snapshot_sequence_he_would_actually_walk(tmp_path):
    """HIS SEQUENCE, END TO END, and it is the crux of this lane.

    An owner and no code: the tab is legitimately open. He reads the foot
    note, goes to a terminal, sets a code, and comes back to the tab he
    never closed -- so its snapshot still says ``open_nocode``.
    """
    app, path = _app(tmp_path, known=True)
    stale = app.people_snapshot()
    assert stale["admin"] == gt.ADMIN_NOCODE, "the tab opened unlocked"

    _set_a_code_at_a_terminal(path)

    # The page still holds `stale`. The seam must not.
    ok, why = app.people_forget("pemberton", now=100.0)
    assert ok is False
    assert why == gt.ADMIN_CODE_OWED
    labels = [p["label"]
              for p in json.loads(path.read_text(encoding="utf-8"))["people"]]
    assert "pemberton" in labels, "the row was removed with no code"


def test_a_code_set_while_the_tab_is_open_is_noticed_without_switching_tabs(
        tmp_path):
    """``show()`` used to be the only thing that re-read the file, so
    switching tabs and back was the only way to notice."""
    app, path = _app(tmp_path)
    assert app.people_admin_state(now=100.0)["admin"] == gt.ADMIN_NOCODE
    _set_a_code_at_a_terminal(path)
    live = app.people_admin_state(now=100.0)
    assert live["admin"] == gt.ADMIN_CODE
    assert live["unlocked_s"] == 0.0
    assert FAKE_CODE not in json.dumps(live)


def test_the_snapshot_is_read_off_the_file_not_the_gates_boot_time_copy(
        tmp_path):
    """THE THIRD STALE READ. ``gate.registry`` is rebound only by
    ``reload()``, which nothing calls until a write from the tab succeeds,
    so the page's "read the file NOW on open" was reading a copy that could
    be hours old: a code, a row or a role set at a terminal was invisible
    until he happened to change something else."""
    app, path = _app(tmp_path)
    assert app.people_snapshot()["admin"] == gt.ADMIN_NOCODE
    _set_a_code_at_a_terminal(path)
    reg = Registry.load(path)
    assert reg.add_person(Person(label="pemberton", role=ROLE_KNOWN,
                                 consent="typed"))[0]
    assert reg.save()
    snap = app.people_snapshot()
    assert snap["admin"] == gt.ADMIN_CODE
    assert [p["label"] for p in snap["people"]] == ["alderman", "pemberton"]
    assert snap["people"][0]["has_code"] is True
    assert FAKE_CODE not in json.dumps(snap)


# ================================================== the bootstrap must live
def test_the_bootstrap_still_makes_a_first_owner_with_no_code(tmp_path):
    """A fresh install must not be a brick, and this is the case the fix
    is most able to break."""
    app, path = _app(tmp_path, empty=True)
    assert app.people_admin_state(now=100.0)["admin"] == gt.ADMIN_FIRST
    ok, line = app.people_add(label="alderman", name="Alderman",
                              role=ROLE_OWNER, consent="owner", now=100.0)
    assert ok is True, line
    assert path.exists()
    assert app.gate.registry.role_of("alderman") == ROLE_OWNER


def test_an_owner_who_set_no_code_is_never_asked_for_one(tmp_path):
    """The second case of the shared rule, said out loud rather than
    demanded: asking for a code he never chose locks him out of the flow
    that sets one."""
    app, _path = _app(tmp_path, known=True)
    ok, line = app.people_forget("pemberton", now=100.0)
    assert ok is True, line


def test_the_bootstrap_write_arms_no_dwell(tmp_path):
    """THE SAME HOLE, ONE SIZE SMALLER. Making the first owner proves no
    code, so it must not leave a window behind that a code set seconds
    later would then fall inside."""
    app, path = _app(tmp_path, empty=True)
    assert app.people_add(label="alderman", role=ROLE_OWNER,
                          consent="owner", now=100.0)[0] is True
    assert app.people_admin_state(now=100.0)["unlocked_s"] == 0.0
    _set_a_code_at_a_terminal(path)
    ok, why = app.people_add(label="pemberton", role=ROLE_KNOWN,
                             consent="console", now=101.0)
    assert ok is False and why == gt.ADMIN_CODE_OWED


# ======================================================= the code, and only
def test_the_right_code_opens_the_dwell_and_the_writes_land(tmp_path):
    app, _path = _app(tmp_path, code=True, known=True)
    ok, line = app.people_unlock(FAKE_CODE, now=100.0)
    assert ok is True and FAKE_CODE not in line
    assert app.people_admin_state(now=100.0)["unlocked_s"] > 0
    ok2, line2 = app.people_forget("pemberton", now=101.0)
    assert ok2 is True, line2


def test_the_dwell_runs_out_and_the_next_write_is_refused(tmp_path):
    app, _path = _app(tmp_path, code=True, known=True)
    assert app.people_unlock(FAKE_CODE, now=100.0)[0] is True
    late = 100.0 + app_mod.PEOPLE_UNLOCK_S + 1.0
    assert app.people_admin_state(now=late)["unlocked_s"] == 0.0
    ok, why = app.people_forget("pemberton", now=late)
    assert ok is False and why == gt.ADMIN_CODE_OWED


def test_a_successful_write_re_arms_the_dwell_so_a_run_is_one_code(tmp_path):
    app, _path = _app(tmp_path, code=True, known=True)
    assert app.people_unlock(FAKE_CODE, now=100.0)[0] is True
    half = 100.0 + app_mod.PEOPLE_UNLOCK_S - 1.0
    assert app.people_forget("pemberton", now=half)[0] is True
    # ...and the window now runs from THAT write, not from the unlock.
    assert app.people_add(label="brightwell", role=ROLE_KNOWN,
                          consent="console",
                          now=half + 5.0)[0] is True


def test_a_wrong_code_is_refused_and_leaves_the_seams_shut(tmp_path):
    app, path = _app(tmp_path, code=True, known=True)
    ok, why = app.people_unlock("nope", now=100.0)
    assert ok is False and "not a code I know" in why
    assert app.people_admin_state(now=100.0)["unlocked_s"] == 0.0
    before = path.read_text(encoding="utf-8")
    assert app.people_forget("pemberton", now=100.0)[0] is False
    assert path.read_text(encoding="utf-8") == before


def test_a_wrong_code_is_counted_on_the_codes_own_limiter(tmp_path):
    """Burning the spoken passphrase's attempts must not close the
    break-glass, and vice versa."""
    app, _path = _app(tmp_path, code=True, phrase=True)
    before_phrase = len(app.gate.phrase_attempts._hits)
    app.people_unlock("nope", now=100.0)
    app.people_unlock("also nope", now=100.0)
    assert len(app.gate.code_attempts._hits) == 2
    assert len(app.gate.phrase_attempts._hits) == before_phrase


def test_the_unlock_re_reads_so_a_code_set_at_a_terminal_can_be_typed(
        tmp_path):
    """THE OTHER HALF OF THE STALENESS. ``check_override_code`` was handed
    the gate's IN-MEMORY registry, which has no code until something calls
    ``reload``. So the tab would demand a code and then refuse the right
    one with "no override code has been set" -- unlockable only by
    restarting Jarvis."""
    app, path = _app(tmp_path, known=True)
    _set_a_code_at_a_terminal(path)
    assert not any(p.code_hash for p in app.gate.registry.owners()), \
        "the in-memory gate must still be stale for this test to mean anything"
    ok, line = app.people_unlock(FAKE_CODE, now=100.0)
    assert ok is True, line
    assert app.people_forget("pemberton", now=100.0)[0] is True


# ================================== the rules the lock must not have eaten
def test_the_last_owner_cannot_be_removed_even_when_unlocked(tmp_path):
    app, _path = _app(tmp_path, code=True)
    assert app.people_unlock(FAKE_CODE, now=100.0)[0] is True
    ok, why = app.people_forget("alderman", now=100.0)
    assert ok is False and "only owner" in why


def test_consent_is_still_taken_for_a_non_owner_when_unlocked(tmp_path):
    app, _path = _app(tmp_path, code=True)
    assert app.people_unlock(FAKE_CODE, now=100.0)[0] is True
    ok, why = app.people_add(label="pemberton", role=ROLE_KNOWN, consent="",
                             now=100.0)
    assert ok is False and "consent" in why.lower()


def test_a_corrupt_registry_is_still_refused_before_anything_else(tmp_path):
    """REFUSE outranks the code: an unlocked tab must not write a one-row
    registry over a file that failed to parse."""
    path = tmp_path / "people.json"
    path.write_text('{"people": [ truncated', encoding="utf-8")
    gate = gt.OwnerGate(registry=Registry.load(path), owner="alderman",
                        get_option=lambda k, d=None: d)
    app = Stub(gate)
    app._people_open_unlock(now=100.0)
    before = path.read_text(encoding="utf-8")
    ok, why = app.people_add(label="pemberton", role=ROLE_KNOWN,
                             consent="console", now=100.0)
    assert ok is False
    assert why != gt.ADMIN_CODE_OWED
    assert path.read_text(encoding="utf-8") == before


def test_relocking_shuts_the_seams_at_once(tmp_path):
    """Leaving the tab must relock the APP, not only the page: the page is
    not the guard, so a page that relocked alone left the real dwell open."""
    app, _path = _app(tmp_path, code=True, known=True)
    assert app.people_unlock(FAKE_CODE, now=100.0)[0] is True
    app.people_relock()
    assert app.people_admin_state(now=100.0)["unlocked_s"] == 0.0
    ok, why = app.people_forget("pemberton", now=100.0)
    assert ok is False and why == gt.ADMIN_CODE_OWED


def test_a_seam_with_no_gate_refuses_rather_than_raising(tmp_path):
    app = Stub(None)
    assert app.people_forget("pemberton", now=100.0)[0] is False
    assert app.people_unlock(FAKE_CODE, now=100.0)[0] is False
    assert app.people_admin_state(now=100.0)["admin"] == gt.ADMIN_REFUSE
    app.people_relock()          # must not raise either


# ============================================== nothing typed may escape
def test_no_code_or_phrase_reaches_a_line_or_a_log_record(tmp_path, caplog):
    app, _path = _app(tmp_path, code=True, known=True, phrase=True)
    caplog.set_level(logging.DEBUG)
    lines = [app.people_unlock("nope", now=100.0)[1],
             app.people_unlock(FAKE_CODE, now=100.0)[1],
             app.people_forget("pemberton", now=100.0)[1],
             app.people_add(label="brightwell", role=ROLE_KNOWN,
                            consent="console", now=100.0)[1],
             app.people_set_role("brightwell", ROLE_OWNER,
                                 confirm_existing_owner="alderman",
                                 now=100.0)[1],
             json.dumps(app.people_admin_state(now=100.0)),
             json.dumps(app.people_snapshot())]
    hashed = [p.code_hash for p in Registry.load(_path).owners()]
    blob = "\n".join(lines + [r.getMessage() for r in caplog.records]
                     + [str(r.args) for r in caplog.records])
    for secret in (FAKE_CODE, FAKE_PHRASE, "nope"):
        assert secret not in blob
    for h in hashed:
        if h:
            assert h not in blob


def test_the_refusal_says_what_to_do_and_names_no_secret():
    assert "code" in gt.ADMIN_CODE_OWED.lower()
    assert gt.ADMIN_CODE_OWED != gt.ADMIN_CODE_LINE
    assert "zzz" not in gt.ADMIN_CODE_OWED


# ======================================================= the wiring itself
def test_the_new_seams_are_offered_to_the_ui():
    import inspect
    src = inspect.getsource(app_mod.JarvisApp.ui_service_kwargs)
    for name in ("people_admin_state", "people_relock"):
        assert "%s=self.%s" % (name, name) in src


def test_the_ui_declares_them():
    import dataclasses
    from jarvis.ui.main_window import Services
    names = {f.name for f in dataclasses.fields(Services)}
    assert {"people_admin_state", "people_relock"} <= names


def test_the_seam_and_the_terminal_tool_ask_the_same_decision():
    """ONE RULE, and a grep is the cheapest way to keep a second one from
    growing beside it: both must go through ``gate.admin_gate``."""
    import importlib.util
    import inspect
    import pathlib

    here = pathlib.Path(gt.__file__).parent.parent / "scripts"
    spec = importlib.util.spec_from_file_location(
        "jarvis_people_one_rule", here / "jarvis_people.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert "admin_gate" in inspect.getsource(cli._authorise)
    assert "admin_gate" in inspect.getsource(app_mod.JarvisApp._people_write)
    assert "check_override_code" in inspect.getsource(
        app_mod.JarvisApp.people_unlock)
