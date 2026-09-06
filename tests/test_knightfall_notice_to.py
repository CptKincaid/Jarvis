"""Where a Knightfall code is mailed TO (Hunter, 2026-09-05: "send the
email for rotating the code to the email the weekly encrypted back up is
sending to on oracle").

THE THING BEING PROTECTED. ``outbox.send_notice`` is the one door in this
package to ``mail.send_message`` (pinned in tests/test_send_file.py), and
it is deliberately narrower than the file lane: no attachment, and **no
recipient parameter**. That is what makes "can a tool loop mail a
stranger?" answerable. Adding a ``to=``/``to_addr=``/``cfg=`` argument
would hand the answer back to the caller, so the destination rides on the
ACCOUNT OBJECT instead -- minted by ``mail.mail_accounts()`` out of his
config file, alongside the SMTP credential, by one function from one file.

So the tests below are in two halves:

* the FEATURE -- a configured ``notice_to`` is used, an absent or blank one
  falls back to the account's own address exactly as before, and a
  malformed one is REFUSED (the old code stands; never a rotation that
  mails nowhere and stores a new hash);
* the INVARIANT -- the signature is frozen, the address is traceable to a
  config read, caller text cannot become a destination, and only one module
  under ``jarvis/`` mints one.

Every address here is invented (example.com). No real mail: the fake
transport tests/test_send_file.py already uses, and the conftest refuses
the submission ports besides.
"""
import ast
import inspect
import logging
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis import gate as gate_mod
from jarvis import outbox
from jarvis.tools import mail as mail_mod
from tests.test_knightfall_typed import (FakeSMTP, RefusingSMTP, _app,
                                         _clean, _mailed_code, _on_disk,
                                         _stub_mail)
from tests.test_notes_mail import FakeCfg
from tests.test_owner_gate import FAKE_CODE

SELF = "hunter@example.com"
VAULT = "knightfall-backup@example.com"

# The legacy single mailbox, the shape tests/test_notes_mail.py uses.
LEGACY = {"gmail": {"address": SELF, "app_password": "abcd efgh ijkl mnop",
                    "imap_host": "imap.gmail.com"}}


def _legacy(notice=None):
    cfg = {"gmail": dict(LEGACY["gmail"])}
    if notice is not None:
        cfg["gmail"]["notice_to"] = notice
    return cfg


def _listed(notice=None):
    entry = {"label": "personal", "address": SELF, "app_password": "s"}
    if notice is not None:
        entry["notice_to"] = notice
    return {"gmail": {"accounts": [entry]}}


@pytest.fixture(autouse=True)
def _fresh_transport():
    FakeSMTP.made = []
    yield
    FakeSMTP.made = []


def _capture():
    """A stub mail module that records the recipient send_message was given."""
    seen = []

    def send_message(account, to_addr, subject, body, smtp=None, **kw):
        seen.append(to_addr)
        return "<id@example.com>"
    return seen, _stub_mail(send_message)


# ================================================================
# 1. The account object carries the destination, out of his config
# ================================================================
@pytest.mark.parametrize("cfg", [_legacy(VAULT), _listed(VAULT)])
def test_a_configured_destination_rides_on_the_account(cfg):
    """Both config shapes: the legacy single mailbox reads gmail.notice_to,
    a listed account reads its own. The credential and the destination are
    minted together, by one function, out of one file."""
    account = mail_mod.mail_accounts(FakeCfg(cfg))[0]
    assert account["address"] == SELF          # the FROM does not move
    assert mail_mod.notice_destination(account) == VAULT


@pytest.mark.parametrize("cfg", [_legacy(), _listed()])
def test_with_the_key_absent_the_account_is_its_own_destination(cfg):
    """The change is INVISIBLE until he opts in: unset must mean today's
    behaviour, never "nowhere"."""
    account = mail_mod.mail_accounts(FakeCfg(cfg))[0]
    assert account.get("notice_to", "") == ""
    assert mail_mod.notice_destination(account) == SELF


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_key_falls_back_to_the_account_itself(blank):
    account = mail_mod.mail_accounts(FakeCfg(_legacy(blank)))[0]
    assert mail_mod.notice_destination(account) == SELF


def test_a_listed_account_does_not_inherit_the_top_level_key():
    """F22, the other way round: gmail.smtp_host was inherited by every
    listed account and submitted them all to the wrong server. A notice
    destination is per mailbox; a top-level one belongs to the legacy
    single mailbox and stops there."""
    cfg = {"gmail": {"notice_to": VAULT,
                     "accounts": [{"label": "personal", "address": SELF,
                                   "app_password": "s"}]}}
    account = mail_mod.mail_accounts(FakeCfg(cfg))[0]
    assert mail_mod.notice_destination(account) == SELF


def test_the_second_account_keeps_its_own_destination():
    cfg = {"gmail": {"accounts": [
        {"label": "personal", "address": SELF, "app_password": "s"},
        {"label": "school", "address": "hp@school.example.com",
         "app_password": "s", "notice_to": VAULT}]}}
    personal, school = mail_mod.mail_accounts(FakeCfg(cfg))
    assert mail_mod.notice_destination(personal) == SELF
    assert mail_mod.notice_destination(school) == VAULT


# ================================================================
# 2. A malformed destination is REFUSED, never quietly self-sent
# ================================================================
@pytest.mark.parametrize("bad", ["not-an-address", "hunter@", "@example.com",
                                 "hunter at example dot com", "hunter@host",
                                 "a@b.com, c@d.com", "a@b.com c@d.com"])
def test_a_malformed_destination_is_refused_not_silently_self_sent(bad):
    """He set the key because he expects the code at the address he named.
    A silent fallback would leave a live code in a different inbox while he
    waits at the one he chose -- so nothing is sent at all."""
    account = mail_mod.mail_accounts(FakeCfg(_legacy(bad)))[0]
    with pytest.raises(mail_mod.NoticeAddressInvalid):
        mail_mod.notice_destination(account)
    seen, stub = _capture()
    with pytest.raises(mail_mod.NoticeAddressInvalid):
        outbox.send_notice(account, "Knightfall", "body\n", mail=stub)
    assert seen == [], "nothing may reach the transport"


def test_a_refused_destination_is_a_mail_send_failure():
    """So every existing caller's `except MailSendFailed` still holds."""
    assert issubclass(mail_mod.NoticeAddressInvalid, mail_mod.MailSendFailed)


def test_a_bad_destination_keeps_the_old_code_and_mails_nothing(tmp_path,
                                                                caplog):
    a = _app(tmp_path, cfg=_legacy("not-an-address"))
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert line == ("Knightfall accepted, sir; the code stays as it is "
                    "(mail: %s)." % app_mod.KNIGHTFALL_BAD_DESTINATION)
    assert FakeSMTP.made == [], "no transport was opened"
    # the old code still works, in memory AND on disk: no new hash stored
    for reg in (a.gate.registry, _on_disk(tmp_path)):
        assert gate_mod.check_override_code(reg, FAKE_CODE)[0] == "hunter"
    _clean(caplog, [line], FAKE_CODE)


def test_the_bad_destination_line_names_the_config_not_the_exception():
    """The rotation's failure line is the exception's TYPE, because that
    text came from a transport that had just been handed a code. This one
    never reaches a transport, so it can be a sentence he can act on -- but
    it is a CONSTANT, not str(exc), so there is still no channel from an
    exception's words to the drawer."""
    src = inspect.getsource(app_mod.JarvisApp._knightfall_rotate)
    assert "KNIGHTFALL_BAD_DESTINATION" in src
    # The CODE, with the docstring dropped -- which talks about str(exc) at
    # length, because that is the bug this function was built around.
    fn = ast.parse(textwrap.dedent(src)).body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body.pop(0)
    body = ast.unparse(fn)
    assert "str(exc)" not in body
    assert "type(exc).__name__" in body, "a TYPE is still all a transport gets"
    line = app_mod.KNIGHTFALL_BAD_DESTINATION
    assert "notice_to" in line or "config" in line
    assert "@" not in line, "never an address on screen"


def test_the_bootstrap_button_stores_nothing_on_a_bad_destination(tmp_path):
    a = _app(tmp_path, code=False, cfg=_legacy("not-an-address"))
    line = a.knightfall_new_code(smtp=FakeSMTP, now=1000.0)
    assert app_mod.KNIGHTFALL_BAD_DESTINATION in line
    assert FakeSMTP.made == []
    assert not _on_disk(tmp_path).person("hunter").code_hash
    # ...and a press that mailed nothing does not burn the minute (R8b)
    assert a.knightfall_new_code(smtp=FakeSMTP, now=1005.0) == line


# ================================================================
# 3. End to end: the code actually goes where his config says
# ================================================================
def test_the_rotated_code_is_mailed_to_the_configured_address(tmp_path,
                                                              caplog):
    a = _app(tmp_path, cfg=_legacy(VAULT))
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert line == "Knightfall accepted, sir; a new code is in your inbox."
    new, lines, msg = _mailed_code()
    assert msg["To"] == VAULT
    assert SELF in msg["From"], "the FROM stays his own account"
    assert msg["Subject"] == "Knightfall"
    # and it really rotated, in memory and on disk
    for reg in (a.gate.registry, _on_disk(tmp_path)):
        assert gate_mod.check_override_code(reg, new)[0] == "hunter"
        assert gate_mod.check_override_code(reg, FAKE_CODE)[0] == ""
    _clean(caplog, [line], new, FAKE_CODE)


def test_with_no_key_set_the_code_still_goes_to_his_own_account(tmp_path):
    """The regression that matters most: unset changes nothing."""
    a = _app(tmp_path, cfg=_legacy())
    assert a.knightfall_code(FAKE_CODE, smtp=FakeSMTP) == \
        "Knightfall accepted, sir; a new code is in your inbox."
    _new, _lines, msg = _mailed_code()
    assert msg["To"] == SELF


def test_a_refusing_transport_keeps_the_old_code_with_the_key_set(tmp_path):
    """The existing failure paths are unchanged by the new destination."""
    a = _app(tmp_path, cfg=_legacy(VAULT))
    line = a.knightfall_code(FAKE_CODE, smtp=RefusingSMTP)
    assert line == ("Knightfall accepted, sir; the code stays as it is "
                    "(mail: MailSendFailed).")
    for reg in (a.gate.registry, _on_disk(tmp_path)):
        assert gate_mod.check_override_code(reg, FAKE_CODE)[0] == "hunter"


def test_no_message_id_keeps_the_old_code_with_the_key_set(tmp_path):
    a = _app(tmp_path, cfg=_legacy(VAULT))
    seen = []

    def send_message(account, to_addr, subject, body, smtp=None, **kw):
        seen.append(to_addr)
        return ""
    line = a.knightfall_code(FAKE_CODE, mail=_stub_mail(send_message))
    assert line == ("Knightfall accepted, sir; the code stays as it is "
                    "(mail: no Message-ID came back).")
    assert seen == [VAULT]
    for reg in (a.gate.registry, _on_disk(tmp_path)):
        assert gate_mod.check_override_code(reg, FAKE_CODE)[0] == "hunter"


def test_a_save_failure_still_leaves_the_old_code_working(tmp_path,
                                                          monkeypatch):
    a = _app(tmp_path, cfg=_legacy(VAULT))
    monkeypatch.setattr(type(a.gate.registry), "save", lambda self: False)
    line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert "could not be stored" in line
    new, _lines, msg = _mailed_code()
    assert msg["To"] == VAULT
    for reg in (a.gate.registry, _on_disk(tmp_path)):
        assert gate_mod.check_override_code(reg, FAKE_CODE)[0] == "hunter"
        assert gate_mod.check_override_code(reg, new)[0] == ""


# ================================================================
# 4. The invariant: a CALLER can never choose the recipient
# ================================================================
def test_send_notice_takes_no_recipient_parameter():
    """The docstring's promise, made mechanical. `to`, `to_addr`,
    `recipient`, `cfg` or **kwargs appearing here fails this test."""
    params = inspect.signature(outbox.send_notice).parameters
    assert tuple(params) == ("account", "subject", "body", "smtp", "mail")
    assert params["smtp"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["mail"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in params.values())


@pytest.mark.parametrize("kw", ["to", "to_addr", "recipient", "cfg"])
def test_a_caller_that_tries_to_name_a_recipient_is_a_type_error(kw):
    seen, stub = _capture()
    account = mail_mod.mail_accounts(FakeCfg(_legacy()))[0]
    with pytest.raises(TypeError):
        outbox.send_notice(account, "Knightfall", "body\n", mail=stub,
                           **{kw: "attacker@example.com"})
    assert seen == []


def test_caller_text_never_becomes_a_destination():
    """Taint: the two things a caller DOES control are the subject and the
    body. Neither is parsed for an address."""
    account = mail_mod.mail_accounts(FakeCfg(_legacy(VAULT)))[0]
    seen, stub = _capture()
    outbox.send_notice(account,
                       "rotate to attacker@example.com",
                       "reply-to: attacker@example.com\nBcc: x@example.com\n",
                       mail=stub)
    assert seen == [VAULT]
    assert "attacker" not in seen[0]


def test_the_mail_seam_cannot_supply_the_destination():
    """`mail=` is the module seam the tests substitute. If the destination
    were resolved THROUGH it, a caller holding that seam would choose the
    recipient after all -- so it is resolved against the real module."""
    seen = []

    def send_message(account, to_addr, subject, body, smtp=None, **kw):
        seen.append(to_addr)
        return "<id@example.com>"
    liar = SimpleNamespace(
        send_message=send_message,
        mail_accounts=mail_mod.mail_accounts,
        MailSendFailed=mail_mod.MailSendFailed,
        NoticeAddressInvalid=mail_mod.NoticeAddressInvalid,
        notice_destination=lambda account: "attacker@example.com")
    account = mail_mod.mail_accounts(FakeCfg(_legacy()))[0]
    outbox.send_notice(account, "Knightfall", "body\n", mail=liar)
    assert seen == [SELF]


def test_the_address_is_traceable_to_a_read_of_his_config():
    """"It always came from the config object", literally: the account is
    built by mail_accounts() from a config that RECORDS what was asked for,
    and the address the transport is given is what that read returned."""
    sentinel = "sentinel-9f31@example.com"

    class Recording(FakeCfg):
        def __init__(self, data):
            super().__init__(data)
            self.reads = []

        def get(self, dotted, default=None):
            self.reads.append(dotted)
            if dotted.endswith(".notice_to"):
                return sentinel
            return super().get(dotted, default)

    cfg = Recording(_legacy())
    account = mail_mod.mail_accounts(cfg)[0]
    seen, stub = _capture()
    outbox.send_notice(account, "Knightfall", "body\n", mail=stub)
    assert "gmail.notice_to" in cfg.reads
    assert seen == [sentinel]


def test_only_one_module_mints_a_notice_destination():
    """The same style of pin as tests/test_send_file.py's one-door grep:
    if a second module starts reading the key out of a config, "where can a
    destination come from?" stops having one answer."""
    root = Path(mail_mod.__file__).parent.parent
    hits = subprocess.run(["grep", "-rln", "--include=*.py", "notice_to",
                           str(root)], capture_output=True,
                          text=True).stdout.split()
    assert sorted(Path(h).name for h in hits) == ["mail.py"]


def test_the_one_door_to_the_transport_is_still_one_door():
    """tests/test_send_file.py owns this pin; asserted here too because
    this change is the kind that would break it."""
    root = Path(mail_mod.__file__).parent.parent
    hits = subprocess.run(["grep", "-rln", "--include=*.py", "send_message",
                           str(root)], capture_output=True,
                          text=True).stdout.split()
    assert sorted(Path(h).name for h in hits) == ["mail.py", "outbox.py"]
    assert [s.name for s in
            mail_mod.make_tools(FakeCfg({}), SimpleNamespace())] == ["get_mail"]


# NOT IN SCOPE, said out loud rather than pretended otherwise: `smtp=` and
# `mail=` stay caller-supplied seams (a transport class and a module).
# They are not address parameters, and anything able to pass a module
# object into this process can `import smtplib` without help. The claim
# above is over ADDRESSES: no argument to send_notice can become one.


# ================================================================
# 5. What the drawer is told (masked, and honest about zero accounts)
# ================================================================
def _status_app(cfg):
    a = SimpleNamespace(assistant=FakeCfg(cfg))
    a.knightfall_status = app_mod.JarvisApp.knightfall_status.__get__(a)
    # The caption now carries the weekly lane's sentence when there is
    # one (jarvis/knightfall_weekly.py); these tests are about the
    # DESTINATION half, so the lane has simply never run.
    a._knightfall_weekly_caption = \
        app_mod.JarvisApp._knightfall_weekly_caption.__get__(a)
    return a


def test_the_status_names_a_masked_destination_never_a_whole_address():
    st = _status_app(_legacy(VAULT)).knightfall_status()
    assert st["to"] == mail_mod._mask_address(VAULT)
    assert VAULT not in str(st), "the drawer is on screen"
    assert st["problem"] == ""


def test_with_no_key_the_status_names_his_own_masked_account():
    st = _status_app(_legacy()).knightfall_status()
    assert st["to"] == mail_mod._mask_address(SELF)
    assert SELF not in str(st)


def test_with_no_account_the_status_is_empty_and_carries_the_setup_line():
    st = _status_app({}).knightfall_status()
    assert st["to"] == "" and st["problem"] == ""
    assert st["setup"], "something he can act on"


def test_a_bad_destination_shows_as_a_problem_not_as_a_destination():
    st = _status_app(_legacy("not-an-address")).knightfall_status()
    assert st["problem"] == app_mod.KNIGHTFALL_BAD_DESTINATION
    assert st["to"] == ""


def test_a_config_that_raises_reads_as_no_account_rather_than_crashing():
    class Boom:
        def get(self, dotted, default=None):
            raise RuntimeError("nope")

    a = SimpleNamespace(assistant=Boom())
    a.knightfall_status = app_mod.JarvisApp.knightfall_status.__get__(a)
    # The caption now carries the weekly lane's sentence when there is
    # one (jarvis/knightfall_weekly.py); these tests are about the
    # DESTINATION half, so the lane has simply never run.
    a._knightfall_weekly_caption = \
        app_mod.JarvisApp._knightfall_weekly_caption.__get__(a)
    st = a.knightfall_status()
    assert st["to"] == "" and st["problem"] == ""
