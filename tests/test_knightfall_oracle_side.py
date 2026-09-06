"""Oracle's half of the weekly code: deploy/oracle/jarvis_override.py.

That file is deployed to /home/opc/knightfall/app/jarvis_override.py on the
dead-man's switch, where app/backup_main.py calls claim() before the send
and settle() after it. It is kept in this repo so what runs on Oracle is
version-controlled and so it can be tested HERE -- nothing in this file
touches Oracle, opens a socket or sends mail.

THE PROPERTY THAT MATTERS MOST: with no spool file the composed body is
BYTE-FOR-BYTE the body Knightfall has always sent. The backup of a
dead-man's switch does not change because a convenience feature was bolted
to it, and it does not fail when that feature does.

The codes here are invented.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone

import pytest

import jarvis

UTC = timezone.utc
SUN = datetime(2026, 9, 13, 7, 30, tzinfo=UTC)          # the Spark's push
COMPOSE = datetime(2026, 9, 13, 8, 30, tzinfo=UTC)      # Oracle sends
CODE = "xxx777yy"                                       # invented
PUSH_ID = "ab12cd34"

# The body app/backup_main.py composes today, character for character
# (read from Oracle read-only on 2026-09-06). The dash is the plain one
# this test file can hold; the assertion that matters is that the base is
# UNCHANGED by the module, not what the base says.
BASE = ("Attached is an encrypted snapshot of your Knightfall database.\n\n"
        "Decrypt it with your MASTER_KEY (Fernet). Keep the backup and the "
        "key in separate, safe places - together in one inbox defeats the "
        "encryption.\n\n"
        "Encrypted size: 4096 bytes.")


@pytest.fixture()
def ov(tmp_path):
    """The Oracle module, with its spool redirected into tmp_path."""
    path = (Path_of_repo() / "deploy" / "oracle" / "jarvis_override.py")
    spec = importlib.util.spec_from_file_location("jarvis_override_under_test",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SPOOL_DIR = tmp_path / "spool"
    mod.PENDING = mod.SPOOL_DIR / "pending.json"
    mod.RECEIPT = mod.SPOOL_DIR / "receipt.json"
    return mod


def Path_of_repo():
    from pathlib import Path
    return Path(jarvis.__file__).parent.parent


def _push(ov, *, code=CODE, code_id=PUSH_ID, at=SUN, for_day="2026-09-13"):
    """What the Spark's ``put`` verb writes."""
    ov._write_json(ov.PENDING, {"id": code_id, "code": code,
                                "pushed_at": at.isoformat(), "for": for_day})


def _push_now(ov, ago=timedelta(minutes=58)):
    """A push dated against the REAL clock: --check and --dry-run are run by
    a human at a keyboard and take no ``now``."""
    at = datetime.now(UTC) - ago
    _push(ov, at=at, for_day=datetime.now(UTC).date().isoformat())


def _compose(ov, now=COMPOSE, sent=True):
    """What run_backup does: claim -> body -> send -> settle."""
    line, token = ov.claim(now)
    body = BASE + line
    ov.settle(token, sent, now)
    return body, line


def _receipt(ov):
    data, why = ov._read_json(ov.RECEIPT)
    return {} if why else data


# ------------------------------------------------- the email is unchanged
def test_with_no_spool_the_body_is_byte_for_byte_todays(ov):
    body, line = _compose(ov)
    assert line == ""
    assert body == BASE, "the email Knightfall has always sent"
    assert _receipt(ov) == {}, "nothing to settle, nothing written"


def test_a_fresh_spool_appends_after_the_size_line_and_never_says_key(ov):
    _push(ov)
    body, line = _compose(ov)
    assert body.startswith(BASE), "the Knightfall text is untouched up to here"
    assert body[:len(BASE)] == BASE
    assert line.startswith("\n\nJarvis override code for this week: ")
    assert CODE in line
    # The master key is NOT in this email and never was; nothing appended
    # may be mistaken for it.
    assert "key" not in line.lower().replace("knightfall", "")
    assert line.endswith("\n")


def test_the_spool_is_deleted_and_the_receipt_says_sent(ov):
    _push(ov)
    _compose(ov)
    assert not ov.PENDING.exists(), "the plaintext does not linger"
    r = _receipt(ov)
    assert r["id"] == PUSH_ID and r["status"] == "sent" and r["at"]


def test_a_failed_send_settles_failed_and_still_ate_the_spool(ov):
    _push(ov)
    body, line = _compose(ov, sent=False)
    assert CODE in line, "it was composed"
    assert not ov.PENDING.exists()
    assert _receipt(ov)["status"] == "failed"


# ----------------------------------------------------------- the refusals
@pytest.mark.parametrize("kw,why", [
    ({"at": SUN - timedelta(days=3)}, "stale"),
    ({"at": SUN + timedelta(days=2)}, "pushed in the future"),
    ({"for_day": "2026-09-20"}, "for another day"),
])
def test_a_spool_that_is_not_for_this_compose_is_discarded_unread(ov, kw, why):
    _push(ov, **kw)
    body, line = _compose(ov)
    assert line == "" and body == BASE
    assert not ov.PENDING.exists(), "deleted unread"
    r = _receipt(ov)
    assert r["status"] == "stale" and r["why"] == why and r["id"] == PUSH_ID


def test_a_malformed_spool_is_discarded_and_the_email_is_todays(ov):
    ov._ensure_dir()
    ov.PENDING.write_text("{ not json")
    body, line = _compose(ov)
    assert line == "" and body == BASE
    assert not ov.PENDING.exists()
    assert _receipt(ov)["status"] == "stale"


def test_a_spool_without_a_code_is_refused(ov):
    ov._write_json(ov.PENDING, {"id": PUSH_ID, "pushed_at": SUN.isoformat(),
                                "for": "2026-09-13"})
    body, line = _compose(ov)
    assert line == "" and body == BASE
    assert _receipt(ov)["status"] == "stale"


# ------------------------------------------- it never raises into run_backup
def test_claim_never_raises_even_when_the_spool_cannot_be_read(ov, monkeypatch):
    _push(ov)
    monkeypatch.setattr(ov, "_read_json",
                        lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    line, token = ov.claim(COMPOSE)
    assert line == "" and token["status"] == "error"


def test_settle_never_raises_when_the_receipt_cannot_be_written(ov, monkeypatch):
    monkeypatch.setattr(ov, "_write_json",
                        lambda p, d: (_ for _ in ()).throw(OSError("read-only")))
    ov.settle({"id": PUSH_ID, "status": "pending"}, True, COMPOSE)  # no raise


def test_settle_of_no_token_writes_nothing(ov):
    ov.settle(None, True, COMPOSE)
    assert not ov.RECEIPT.exists()


# ------------------------------------------------------------- the verbs
def test_put_writes_a_private_spool_and_refuses_junk(ov, monkeypatch, capsys):
    payload = json.dumps({"id": PUSH_ID, "code": CODE,
                          "pushed_at": SUN.isoformat(), "for": "2026-09-13"})
    monkeypatch.setattr(ov.sys, "stdin", _Stdin(payload))
    assert ov.main(["put"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "id": PUSH_ID}
    assert (ov.PENDING.stat().st_mode & 0o777) == 0o600, "0600, umask is 0022"
    assert (ov.SPOOL_DIR.stat().st_mode & 0o777) == 0o700

    monkeypatch.setattr(ov.sys, "stdin", _Stdin('{"id": "nope", "code": "x"}'))
    assert ov.main(["put"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_receipt_prints_and_only_an_ack_deletes(ov, capsys):
    """2026-09-06, defect (B): the read and the delete are two verbs, so a
    Spark that read a receipt and then failed to write its own registry
    has not destroyed the only record that the email went out. The wider
    rules are in tests/test_knightfall_weekly_defects.py."""
    _push(ov)
    _compose(ov)
    assert ov.main(["receipt"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == PUSH_ID
    assert ov.RECEIPT.exists(), "a read deletes nothing"
    assert ov.main(["receipt"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == PUSH_ID

    assert ov.main(["receipt", "--ack", PUSH_ID]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "acked": True}
    assert not ov.RECEIPT.exists()
    assert ov.main(["receipt"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "none"}


def test_revoke_removes_only_that_push(ov, capsys):
    _push(ov)
    assert ov.main(["revoke", "ffffffff"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "revoked": False}
    assert ov.PENDING.exists(), "another push's id must not delete this one"
    assert ov.main(["revoke", PUSH_ID]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "revoked": True}
    assert not ov.PENDING.exists()


def test_ping_and_an_unknown_verb(ov, capsys):
    assert ov.main(["ping"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert ov.main(["shell"]) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


# --------------------------------------------------------- the rehearsals
def test_check_says_what_would_happen_and_changes_nothing(ov, capsys):
    assert ov.main(["--check"]) == 0
    assert capsys.readouterr().out.strip() == \
        "no spool: the email would be exactly as today"

    _push_now(ov)
    ov.main(["--check"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("a line WOULD be included") and PUSH_ID in out
    assert ov.PENDING.exists(), "--check deletes nothing"
    assert CODE not in out, "and never prints the code"

    _push_now(ov, ago=timedelta(days=3))
    ov.main(["--check"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("stale") and "deleted unread" in out
    assert ov.PENDING.exists()


def test_the_dry_run_prints_one_line_only_and_never_the_code(ov, capsys):
    assert ov.main(["--dry-run"]) == 0
    assert capsys.readouterr().out.strip() == "extra line: would not"
    _push_now(ov)
    assert ov.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.strip() == "extra line: would include"
    assert len(out.strip().splitlines()) == 1
    assert CODE not in out
    assert ov.PENDING.exists(), "--dry-run sends nothing and deletes nothing"
    assert not ov.RECEIPT.exists()


def test_the_module_imports_nothing_beyond_the_stdlib(ov):
    """Oracle's venv is Knightfall's, not ours: an import of anything but
    the standard library would be a new dependency on a box whose job is
    to still work in a year."""
    import ast
    src = (Path_of_repo() / "deploy" / "oracle" / "jarvis_override.py").read_text()
    allowed = {"json", "os", "sys", "tempfile", "datetime", "pathlib",
               "__future__"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in allowed, a.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed, node.module


class _Stdin:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text
