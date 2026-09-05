"""Knightfall, the TYPED path (Hunter, 2026-09-04: "a and b").

``JarvisApp.knightfall_code(code)``: the keyboard's way in. Checked with
the SAME ``check_override_code`` the people CLI uses, on the gate's own
code counter; on success the window opens on the code leg and the code
ROTATES -- a fresh one is mailed to his own address from his own first
account, and ONLY a returned Message-ID lets the new hash be stored. Mail
failed, no account, no Message-ID: the old code stands and the line says
so. ``knightfall_new_code()`` is the bootstrap: same generate -> mail ->
store, one press a minute.

Every line and every log record is searched for both plaintexts. No real
mail: the fake transport tests/test_send_file.py already uses, and the
conftest refuses the submission ports besides. No real registry.
"""
import logging
import time
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis import gate as gate_mod
from jarvis import passphrase as pp
from jarvis.identity import Registry
from jarvis.tools import mail as mail_mod
from tests.test_notes_mail import GMAIL_CFG, FakeCfg
from tests.test_owner_gate import FAKE_CODE, MATCHED, _registry
from tests.test_send_file import FakeSMTP

OK_LINE = "Knightfall accepted, sir; a new code is in your inbox."


class RefusingSMTP(FakeSMTP):
    def login(self, user, password):
        raise RuntimeError("535 5.7.8 nope")


@pytest.fixture(autouse=True)
def _fresh_transport():
    FakeSMTP.made = []
    yield
    FakeSMTP.made = []


def _app(tmp_path, *, code=True, cfg=None, mode="shadow"):
    reg = _registry(tmp_path, code=code)
    opts = {"owner.mode": mode}
    a = SimpleNamespace()
    a.assistant = FakeCfg(GMAIL_CFG if cfg is None else cfg)
    a.get_option = lambda k, d=None: opts.get(k, d)
    a.gate = gate_mod.OwnerGate(registry=reg, owner="hunter",
                                get_option=a.get_option)
    for name in ("knightfall_code", "knightfall_new_code",
                 "_knightfall_rotate"):
        setattr(a, name, getattr(app_mod.JarvisApp, name).__get__(a))
    return a


def _mailed_code():
    """The plaintext the fake transport saw, so the tests can prove it is
    nowhere else. Body: the code, then the one sentence."""
    assert FakeSMTP.made, "nothing was sent"
    msg = FakeSMTP.made[-1].sent[-1]
    lines = msg.get_content().splitlines()
    return lines[0].strip(), lines, msg


def _clean(caplog, lines, *codes):
    hay = [caplog.text] + [str(x) for x in lines]
    for record in caplog.records:
        hay.append(record.getMessage())
        hay.append(str(record.args))
    for code in codes:
        assert code, "a code to search for"
        for text in hay:
            assert code not in text, "a plaintext code leaked"


def _on_disk(tmp_path):
    return Registry.load(tmp_path / "people.json")


# ------------------------------------------------------------ success
def test_the_right_code_opens_the_window_and_rotates_by_mail(tmp_path, caplog):
    a = _app(tmp_path)
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert line == OK_LINE
    new, lines, msg = _mailed_code()
    assert msg["Subject"] == "Knightfall"
    assert msg["To"] == "hunter@example.com" and "hunter@example.com" in msg["From"]
    assert lines == [new, "Typed only, never spoken. This replaces the old one."]
    assert len(new) == 8 and set(new) <= set(pp.CODE_ALPHABET)
    # the window: a dropped clip inside it is admitted on the CODE leg
    d = a.gate.judge("voice", "read me my mail", stats=MATCHED, rejected=True)
    assert d.admit is True and d.how == gate_mod.HOW_CODE and d.who == "hunter"
    # the rotation: the new code checks true, the old one no longer does,
    # in memory and on disk
    for reg in (a.gate.registry, _on_disk(tmp_path)):
        assert gate_mod.check_override_code(reg, new)[0] == "hunter"
        assert gate_mod.check_override_code(reg, FAKE_CODE)[0] == ""
    _clean(caplog, [line], new, FAKE_CODE)
    assert "the code opened the floor to hunter" in caplog.text


def test_the_gates_own_code_counter_is_the_one_used(tmp_path, monkeypatch):
    a = _app(tmp_path)
    seen = []
    real = gate_mod.check_override_code

    def spy(registry, code, *, attempts=None):
        seen.append(attempts)
        return real(registry, code, attempts=attempts)
    monkeypatch.setattr(gate_mod, "check_override_code", spy)
    a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert seen == [a.gate.code_attempts]
    assert a.gate.code_attempts is not a.gate.phrase_attempts


# ------------------------------------------------------------ refusals
@pytest.mark.parametrize("typed,why", [
    ("xxx999", "that is not a code I know"),
    ("", "no code was given"),
    ("   ", "no code was given"),
])
def test_a_refusal_is_the_reason_and_nothing_else(tmp_path, typed, why, caplog):
    a = _app(tmp_path)
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(typed, smtp=FakeSMTP)
    assert line == "Knightfall: %s" % why
    assert FakeSMTP.made == []
    d = a.gate.judge("voice", "hello", stats=MATCHED, rejected=True)
    assert d.how == gate_mod.HOW_NOBODY
    _clean(caplog, [line], FAKE_CODE)


def test_with_no_code_set_the_typed_path_says_so(tmp_path):
    a = _app(tmp_path, code=False)
    assert a.knightfall_code(FAKE_CODE, smtp=FakeSMTP) == \
        "Knightfall: no override code has been set"
    assert FakeSMTP.made == []


def test_too_many_wrong_codes_is_a_cool_off_on_the_code_counter(tmp_path):
    a = _app(tmp_path)
    for _ in range(pp.CODE_LIMIT):
        assert a.knightfall_code("xxx999", smtp=FakeSMTP).startswith(
            "Knightfall: that is not")
    line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert line.startswith("Knightfall: too many tries; wait")
    assert FakeSMTP.made == []
    # and the phrase's counter was never touched
    assert a.gate.phrase_attempts.allow()[0] is True


# ------------------------------------------ rotation only after Message-ID
def test_mail_failure_keeps_the_old_code_and_says_so(tmp_path, caplog):
    a = _app(tmp_path)
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(FAKE_CODE, smtp=RefusingSMTP)
    assert line.startswith("Knightfall accepted, sir; the code stays as it is (mail: ")
    assert line.endswith(").")
    # accepted: the window IS open...
    assert a.gate.judge("voice", "hello", stats=MATCHED,
                        rejected=True).how == gate_mod.HOW_CODE
    # ...and the OLD code still checks true, in memory and on disk
    for reg in (a.gate.registry, _on_disk(tmp_path)):
        assert gate_mod.check_override_code(reg, FAKE_CODE)[0] == "hunter"
    _clean(caplog, [line], FAKE_CODE)


def test_no_message_id_means_no_rotation(tmp_path):
    sent = []

    def send_message(account, to_addr, subject, body, smtp=None, **kw):
        sent.append(body.splitlines()[0])
        return ""
    stub = SimpleNamespace(mail_accounts=mail_mod.mail_accounts,
                           send_message=send_message,
                           MailSendFailed=mail_mod.MailSendFailed)
    a = _app(tmp_path)
    line = a.knightfall_code(FAKE_CODE, mail=stub)
    assert line == "Knightfall accepted, sir; the code stays as it is " \
                   "(mail: no Message-ID came back)."
    assert len(sent) == 1
    assert gate_mod.check_override_code(_on_disk(tmp_path), FAKE_CODE)[0] == "hunter"
    assert gate_mod.check_override_code(_on_disk(tmp_path), sent[0])[0] == ""


def test_no_mail_account_keeps_the_old_code(tmp_path, caplog):
    a = _app(tmp_path, cfg={})
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert line == "Knightfall accepted, sir; the code stays as it is " \
                   "(mail: no mail account is configured)."
    assert FakeSMTP.made == []
    assert gate_mod.check_override_code(_on_disk(tmp_path), FAKE_CODE)[0] == "hunter"
    _clean(caplog, [line], FAKE_CODE)


def test_a_save_failure_after_a_sent_mail_leaves_the_old_code_working(
        tmp_path, caplog, monkeypatch):
    """Never a state where no code works: the mail went, the store did
    not, so the old hash is put back in memory (disk never changed) and
    the line says the one in the inbox is dead."""
    a = _app(tmp_path)
    monkeypatch.setattr(a.gate.registry, "save", lambda: False)
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    new, _, _ = _mailed_code()
    assert line.startswith("Knightfall accepted, sir; the new code could not be stored")
    assert "old one stands" in line
    for reg in (a.gate.registry, _on_disk(tmp_path)):
        assert gate_mod.check_override_code(reg, FAKE_CODE)[0] == "hunter"
        assert gate_mod.check_override_code(reg, new)[0] == ""
    assert any(r.levelno == logging.ERROR and "knightfall" in r.getMessage().lower()
               for r in caplog.records)
    _clean(caplog, [line], new, FAKE_CODE)


def test_the_plaintext_is_not_left_on_the_app_or_the_gate(tmp_path):
    a = _app(tmp_path)
    a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    new, _, _ = _mailed_code()
    for obj in (a, a.gate, a.gate.registry):
        assert new not in repr(vars(obj))
        assert FAKE_CODE not in repr(vars(obj))


# ------------------------------------------------------------ bootstrap
def test_email_me_a_new_code_mails_and_stores_one_for_the_owner(tmp_path,
                                                                caplog):
    a = _app(tmp_path, code=False)
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_new_code(smtp=FakeSMTP, now=1000.0)
    assert line == "Knightfall: a new code is in your inbox."
    new, lines, msg = _mailed_code()
    assert msg["Subject"] == "Knightfall" and lines[1].startswith("Typed only")
    assert gate_mod.check_override_code(_on_disk(tmp_path), new)[0] == "hunter"
    # it did NOT open the window: mailing a code is not typing one
    assert a.gate.judge("voice", "hello", stats=MATCHED,
                        rejected=True).how == gate_mod.HOW_NOBODY
    _clean(caplog, [line], new)


def test_one_code_a_minute(tmp_path):
    a = _app(tmp_path, code=False)
    assert a.knightfall_new_code(smtp=FakeSMTP, now=1000.0).startswith(
        "Knightfall: a new code")
    assert a.knightfall_new_code(smtp=FakeSMTP, now=1030.0) == \
        "Knightfall: one code a minute, sir."
    assert len(FakeSMTP.made) == 1
    assert a.knightfall_new_code(smtp=FakeSMTP, now=1061.0).startswith(
        "Knightfall: a new code")
    assert len(FakeSMTP.made) == 2
    assert app_mod.KNIGHTFALL_COOLDOWN_S == 60.0


def test_the_cooldown_runs_on_the_monotonic_clock_by_default(tmp_path,
                                                             monkeypatch):
    a = _app(tmp_path, code=False)
    t = [5000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    a.knightfall_new_code(smtp=FakeSMTP)
    assert a.knightfall_new_code(smtp=FakeSMTP) == \
        "Knightfall: one code a minute, sir."
    t[0] += 61.0
    assert a.knightfall_new_code(smtp=FakeSMTP).startswith("Knightfall: a new")


def test_a_new_code_that_cannot_be_mailed_stores_nothing(tmp_path):
    a = _app(tmp_path, code=True)
    line = a.knightfall_new_code(smtp=RefusingSMTP, now=1000.0)
    assert line.startswith("Knightfall: the code stays as it is (mail: ")
    assert gate_mod.check_override_code(_on_disk(tmp_path), FAKE_CODE)[0] == "hunter"


def test_with_nobody_enrolled_the_bootstrap_says_so_and_mails_nothing(
        tmp_path):
    a = _app(tmp_path, code=False)
    a.gate.registry = Registry(path=tmp_path / "empty.json")
    line = a.knightfall_new_code(smtp=FakeSMTP, now=1000.0)
    assert line.startswith("Knightfall: nobody is enrolled as an owner")
    assert FakeSMTP.made == []


def test_keyboard_is_owner_is_the_existing_rule_and_not_a_second_check():
    """check_override_code's docstring is the rule; the bootstrap does not
    invent another. Said in its docstring, not enforced twice."""
    doc = app_mod.JarvisApp.knightfall_new_code.__doc__ or ""
    assert "keyboard" in doc.lower() and "owner" in doc.lower()
    import inspect
    src = inspect.getsource(app_mod.JarvisApp.knightfall_new_code)
    assert "check_override_code" not in src.split('"""')[-1]


# ------------------------------------------------------- the generator
def test_new_code_is_eight_readable_characters_from_secrets():
    assert pp.CODE_ALPHABET == "abcdefghijkmnpqrstuvwxyz23456789"
    for bad in "0o1lO":
        assert bad not in pp.CODE_ALPHABET
    seen = {pp.new_code() for _ in range(50)}
    assert len(seen) == 50
    for code in seen:
        assert len(code) == 8 and set(code) <= set(pp.CODE_ALPHABET)
        assert pp.code_ok(code)[0] is True
    import inspect
    assert "secrets.choice" in inspect.getsource(pp.new_code)
