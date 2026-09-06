"""ONE administrative rule, shared by the terminal tool and the users tab.

Everything here is pure: no Tk, no display, no registry of his. The three
pieces the tab and ``scripts/jarvis_people.py`` must share are

  * ``identity.Registry.fault_kind`` -- WHY a registry is unusable, as a
    token rather than an English sentence, so a caller can branch on the
    one case where writing would destroy data;
  * ``gate.admin_gate`` -- the three-case decision ``_authorise`` used to
    make privately, plus the corrupt case it used to get WRONG;
  * ``jarvis/consent.py`` -- the words and the "type your own label" rule,
    with two takers (a terminal one and a console one) that record which
    of them was used.

His real people book is never touched: every registry here is built in a
tmp_path with invented labels, names and codes.
"""
import pytest

from jarvis import consent as cs
from jarvis import gate as gt
from jarvis import passphrase as pp
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry

FAKE_CODE = "zzz-not-a-real-code-zzz"


def _reg(tmp_path, *, code=False, known=False, name="people.json"):
    r = Registry(path=tmp_path / name)
    r.add_person(Person(label="alderman", name="Alderman", role=ROLE_OWNER,
                        voice=True))
    if known:
        r.add_person(Person(label="pemberton", name="Pemberton",
                            role=ROLE_KNOWN, face="pemberton",
                            consent="typed"))
    if code:
        r.set_secret("alderman", "code_hash", pp.hash_secret(FAKE_CODE))
    r.save()
    return r


# ================================================= why a registry is unusable
def test_a_missing_file_is_a_different_fault_from_a_corrupt_one(tmp_path):
    """The CLI treats all five the same ("anybody may enrol the first
    owner"). For a file that FAILED TO PARSE that is a data shredder:
    people comes back empty and the next save writes a one-row registry
    over the top. A token is what lets a caller tell them apart."""
    missing = Registry.load(tmp_path / "nothing.json")
    assert missing.usable is False
    assert missing.fault_kind == "missing"

    bad = tmp_path / "bad.json"
    bad.write_text("{not json at all")
    assert Registry.load(bad).fault_kind == "malformed"

    shaped = tmp_path / "shaped.json"
    shaped.write_text('{"format": 1, "people": "not a list"}')
    assert Registry.load(shaped).fault_kind == "malformed"

    ownerless = tmp_path / "ownerless.json"
    ownerless.write_text('{"format": 1, "people": [{"label": "pemberton", '
                         '"role": "known"}]}')
    reg = Registry.load(ownerless)
    assert reg.fault_kind == "ownerless"
    # ...and its rows SURVIVED, which is why adding an owner here is safe
    assert [p.label for p in reg.people] == ["pemberton"]


def test_a_usable_registry_has_no_fault_kind_at_all(tmp_path):
    reg = Registry.load(_reg(tmp_path).path)
    assert reg.usable is True
    assert reg.fault_kind == "" and reg.fault == ""


def test_an_unreadable_file_is_its_own_token(tmp_path):
    path = tmp_path / "people.json"
    path.write_text("{}")
    path.chmod(0o000)
    try:
        reg = Registry.load(path)
    finally:
        path.chmod(0o600)
    if reg.usable:                    # running as root: nothing to prove
        pytest.skip("this process can read a 0000 file")
    assert reg.fault_kind == "unreadable"


# ====================================================== the shared decision
def test_no_registry_at_all_lets_the_first_owner_be_made(tmp_path):
    """Otherwise a fresh install is a brick and enrolment is unreachable."""
    state, why = gt.admin_gate(Registry.load(tmp_path / "nothing.json"))
    assert state == gt.ADMIN_FIRST
    assert "first owner" in why.lower()


def test_an_owner_with_no_code_is_allowed_and_told_so(tmp_path):
    state, why = gt.admin_gate(_reg(tmp_path))
    assert state == gt.ADMIN_NOCODE
    assert "set-code" in why


def test_an_owner_who_has_set_a_code_is_asked_for_it(tmp_path):
    state, why = gt.admin_gate(_reg(tmp_path, code=True))
    assert state == gt.ADMIN_CODE
    assert why


def test_a_corrupt_registry_is_refused_rather_than_overwritten(tmp_path):
    """THE HAZARD THE CLI HAS TODAY. A file that failed to parse loads as
    zero people; ``add_person`` then ``save`` writes a one-row registry
    over whatever was in it. There is no history and no backup."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"people": [ truncated')
    state, why = gt.admin_gate(Registry.load(bad))
    assert state == gt.ADMIN_REFUSE
    assert str(bad) in why


def test_the_decision_never_raises_on_a_registry_that_cannot_answer():
    class Broken:
        usable = True
        fault_kind = ""

        def owners(self):
            raise RuntimeError("no")

    state, why = gt.admin_gate(Broken())
    assert state == gt.ADMIN_REFUSE and why


# ========================================================= the consent rule
def test_consent_is_given_by_typing_your_own_label_and_nothing_else():
    assert cs.agreed("pemberton", "pemberton") is True
    assert cs.agreed("pemberton", "  Pemberton \n") is True   # trimmed, folded
    assert cs.agreed("pemberton", "yes") is False
    assert cs.agreed("pemberton", "y") is False
    assert cs.agreed("pemberton", "") is False
    assert cs.agreed("pemberton", "alderman") is False
    assert cs.agreed("pemberton", None) is False


def test_the_two_texts_say_what_is_actually_being_stored():
    """A finding, not a preference: the face text describes storing 128
    numbers of somebody's face. Adding a REGISTRY ROW stores no face data
    at all, so the same words there would over-claim."""
    fields = {"who": "pemberton", "root": "/tmp/nowhere",
              "python": "/usr/bin/python3", "script": "/tmp/face_enrol.py"}
    face = "\n".join(cs.lines_for(cs.WHAT_FACE, fields))
    row = "\n".join(cs.lines_for(cs.WHAT_ROW, fields))
    assert "128" in face and "FACE" in face
    assert "128" not in row
    assert "pemberton" in row
    # both say what it can never do, in the same words
    assert "cannot command Jarvis" in face and "cannot command Jarvis" in row
    # and the row text says the face capture is a separate consent
    assert "face" in row.lower()


def test_a_console_consent_never_claims_it_happened_at_a_terminal():
    """The provenance string is the only durable record of HOW the consent
    was taken. A console consent that wrote "typed" would be a false
    attestation on disk, which is worse than no record."""
    assert cs.HOW_TERMINAL == "typed"
    assert cs.HOW_CONSOLE == "console"
    assert cs.HOW_TERMINAL != cs.HOW_CONSOLE


def test_the_terminal_taker_still_refuses_a_pipe(monkeypatch):
    said = []
    ok, how = cs.take_at_terminal("pemberton", what=cs.WHAT_ROW,
                                  fields={"who": "pemberton"},
                                  say=said.append, ask=lambda _p: "pemberton",
                                  isatty=lambda _s: False)
    assert ok is False
    assert "terminal" in how
    assert not said                 # nothing was shown, so nothing was agreed


def test_the_terminal_taker_returns_typed_when_they_type_their_name():
    said = []
    ok, how = cs.take_at_terminal("pemberton", what=cs.WHAT_ROW,
                                  fields={"who": "pemberton"},
                                  say=said.append, ask=lambda _p: "PEMBERTON",
                                  isatty=lambda _s: True)
    assert (ok, how) == (True, cs.HOW_TERMINAL)
    assert any("pemberton" in line for line in said)


def test_the_console_taker_returns_console_and_needs_a_real_window():
    shown = []
    ok, how = cs.take_at_console("pemberton", what=cs.WHAT_ROW,
                                 fields={"who": "pemberton"},
                                 show=shown.append,
                                 ask=lambda: "pemberton", mapped=lambda: True)
    assert (ok, how) == (True, cs.HOW_CONSOLE)
    assert shown

    # ...and a window nobody can see is not a window somebody agreed at
    shown2 = []
    ok2, how2 = cs.take_at_console("pemberton", what=cs.WHAT_ROW,
                                   fields={"who": "pemberton"},
                                   show=shown2.append,
                                   ask=lambda: "pemberton",
                                   mapped=lambda: False)
    assert ok2 is False and "window" in how2
    assert not shown2


def test_a_refused_consent_says_nothing_was_written():
    ok, how = cs.take_at_console("pemberton", what=cs.WHAT_ROW,
                                 fields={"who": "pemberton"},
                                 show=lambda _l: None,
                                 ask=lambda: "no", mapped=lambda: True)
    assert ok is False
    assert "nothing was written" in how


def test_the_owner_enrolling_himself_needs_nobody_elses_agreement():
    ok, how = cs.take_at_console("alderman", what=cs.WHAT_ROW,
                                 fields={"who": "alderman"},
                                 show=lambda _l: None,
                                 ask=lambda: "", mapped=lambda: True,
                                 owner="alderman")
    assert (ok, how) == (True, "owner")


# ============================================== the CLI uses the same rule
def test_the_terminal_tool_asks_the_shared_decision(monkeypatch, tmp_path):
    """Not a second implementation. ``_authorise`` calls ``admin_gate``."""
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "jarvis_people_under_test",
        pathlib.Path(gt.__file__).parent.parent / "scripts" / "jarvis_people.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    seen = []

    def fake(reg):
        seen.append(reg)
        return gt.ADMIN_NOCODE, "no code has been set"

    monkeypatch.setattr(gt, "admin_gate", fake)
    ok, who = mod._authorise(_reg(tmp_path))
    assert ok is True and who == "nocode"
    assert len(seen) == 1


def test_the_terminal_tool_refuses_to_write_over_a_corrupt_registry(tmp_path,
                                                                    capsys):
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "jarvis_people_corrupt", pathlib.Path(gt.__file__).parent.parent
        / "scripts" / "jarvis_people.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bad = tmp_path / "people.json"
    bad.write_text('{"people": [ truncated')
    ok, why = mod._authorise(Registry.load(bad))
    assert ok is False
    assert str(bad) in why


# ================================ the terminal tool still takes consent
def _cli():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "jarvis_people_consent", pathlib.Path(gt.__file__).parent.parent
        / "scripts" / "jarvis_people.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    """A stand-in for the argparse namespace ``do_add`` is handed.

    IT CARRIES EVERY FIELD do_add READS, and deliberately does not use
    getattr defaults on the production side to make that unnecessary. The
    honorific in particular: the security lane made ``--honorific``
    REQUIRED with no default, on the stated rule that a form of address is
    never inferred from a name -- so a stand-in that omits it is not
    "close enough", it is a namespace argparse could never produce. This
    class was written before that flag existed and lost four tests to it
    on the co-merge (AttributeError at scripts/jarvis_people.py:181);
    ``first`` and ``last`` arrived in the same lane and would have been
    next. Add the field here when do_add grows one -- never a getattr
    fallback there, which would let a missing form of address through as
    a silent "none".
    """

    def __init__(self, **kw):
        self.label = kw.get("label", "pemberton")
        self.name = kw.get("name", "Pemberton")
        self.role = kw.get("role", ROLE_KNOWN)
        self.face = kw.get("face", "")
        self.face_dim = 0
        self.voice = False
        self.confirm_owner = kw.get("confirm_owner")
        self.honorific = kw.get("honorific", "none")
        self.first = kw.get("first", "")
        self.last = kw.get("last", "")


def test_adding_a_guest_at_a_terminal_still_asks_them_to_type_their_name(
        tmp_path, monkeypatch, capsys):
    mod = _cli()
    reg = _reg(tmp_path)
    monkeypatch.setattr(cs, "_isatty", lambda _s: True)
    monkeypatch.setattr("builtins.input", lambda *_a: "no")
    code = mod.do_add(reg, None, _Args())
    out = capsys.readouterr().out
    assert code == 2
    assert reg.person("pemberton") is None
    assert "CONSENT" in out
    assert "pemberton" in out
    # the words match what `add` actually stores: a row, not a face
    assert "128" not in out


def test_a_guest_who_types_their_own_name_is_enrolled_as_a_typed_consent(
        tmp_path, monkeypatch, capsys):
    mod = _cli()
    reg = _reg(tmp_path)
    monkeypatch.setattr(cs, "_isatty", lambda _s: True)
    monkeypatch.setattr("builtins.input", lambda *_a: "pemberton")
    assert mod.do_add(reg, None, _Args()) == 0
    person = reg.person("pemberton")
    assert person is not None
    assert person.consent == cs.HOW_TERMINAL
    assert person.consent != cs.HOW_CONSOLE


def test_a_pipe_still_cannot_give_a_guests_consent(tmp_path, monkeypatch,
                                                   capsys):
    mod = _cli()
    reg = _reg(tmp_path)
    monkeypatch.setattr(cs, "_isatty", lambda _s: False)
    assert mod.do_add(reg, None, _Args()) == 2
    assert reg.person("pemberton") is None
    assert "terminal" in capsys.readouterr().out


def test_the_owner_adding_himself_at_a_terminal_asks_nobody(tmp_path,
                                                            monkeypatch):
    mod = _cli()
    path = tmp_path / "fresh.json"
    reg = Registry.load(path)
    monkeypatch.setattr(cs, "_isatty", lambda _s: True)
    assert mod.do_add(reg, None, _Args(label="alderman", role=ROLE_OWNER)) == 0
    assert reg.person("alderman").consent == cs.HOW_OWNER
