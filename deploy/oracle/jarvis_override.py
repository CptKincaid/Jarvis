"""The Jarvis override code that rides in Sunday's backup email.

DEPLOYED TO /home/opc/knightfall/app/jarvis_override.py. This copy is the
one in the Jarvis repo so that what runs on Oracle is version-controlled;
deploy/oracle/README.md says how it gets there and what else changes.

HIS RULING, 2026-09-06 11:00, verbatim: "I want the nightfall code to send
weekly in the same email as the encrypted back up."

WHAT THIS IS. The Spark makes a fresh Knightfall override code every Sunday
at 07:30 UTC and hands the PLAINTEXT to this box over ssh, into
spool/pending.json (0600, in a 0700 directory). An hour later Oracle's own
knightfall-backup.timer runs app.backup_main, which asks this module for
one extra line to put at the end of the email it already sends. Then this
module deletes the spool and leaves a receipt the Spark collects, which is
how the Spark learns it may retire the previous code.

THE THREE RULES THIS MODULE IS BUILT AROUND.

  1. IT NEVER RAISES INTO run_backup. The backup of a dead-man's switch is
     not allowed to fail because of a convenience feature bolted to it.
     Every public function catches BaseException and answers with "no
     line". With no spool file, or with this module broken, the email is
     BYTE-FOR-BYTE what it has always been.
  2. STDLIB ONLY, AND NO Config. Config.from_env() raises without
     /etc/knightfall.env (root-owned, 0600, read by systemd), so anything
     needing it cannot be rehearsed as opc. Nothing here reads the
     environment, the database or any secret.
  3. THE PLAINTEXT IS DELETED IN EVERY BRANCH OF claim(). Fresh, stale or
     malformed, the spool file goes. It exists on this disk for about the
     hour between the Spark's push and the compose, and nowhere else here:
     no log line, no receipt, no audit row carries it -- only the push id,
     which is eight hex characters that name the push and say nothing
     about the code.

THE VERBS, spoken over ssh by jarvis/knightfall_weekly.py on the Spark and
allowed one-by-one by the forced-command key in jarvis-override-gate.sh:

  put             JSON on stdin -> spool/pending.json, atomically, 0600
  receipt         print spool/receipt.json and delete it
  revoke <id>     delete spool/pending.json if it is that push's
  ping            "is the gate alive and is the spool directory there"

And two REHEARSALS that change nothing, for a human at a keyboard:

  --check         say what claim() WOULD do, deleting nothing
  --dry-run       compose the body exactly as backup_main does and print
                  ONLY "extra line: would include" / "would not". It
                  sends nothing and never prints the code itself.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# /home/opc/knightfall/app/jarvis_override.py -> /home/opc/knightfall
ROOT = Path(__file__).resolve().parent.parent
SPOOL_DIR = ROOT / "spool"
PENDING = SPOOL_DIR / "pending.json"
RECEIPT = SPOOL_DIR / "receipt.json"

# A push lands about an hour before the compose. 36 h is a belt around
# that, not a measurement: it lets a Saturday-evening push (the fallback
# time, 23:30 UTC) through and still discards a spool left by a week that
# never composed. SKEW allows an hour of disagreement between two NTP
# clocks in the other direction.
FRESH_S = 36 * 3600
SKEW_S = 3600

LEAD = "Jarvis override code for this week: "
NOTE = ("Typed only, never spoken; it replaces the previous code once "
        "Jarvis confirms this email went out.")

# Eight lowercase hex characters, the id the Spark generates.
_HEX = "0123456789abcdef"


def _is_id(text) -> bool:
    text = str(text or "")
    return len(text) == 8 and all(c in _HEX for c in text)


def _utc(now=None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if getattr(now, "tzinfo", None) is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


# ------------------------------------------------------------- the files
def _ensure_dir() -> None:
    SPOOL_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(SPOOL_DIR, 0o700)
    except OSError:
        pass


def _write_json(path: Path, data: dict) -> None:
    """0600, atomic, in a 0700 directory. umask here is 0022, so the mode
    is set explicitly rather than left to the process."""
    _ensure_dir()
    fd, tmp = tempfile.mkstemp(prefix=".jarvis-override-", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _delete(path: Path) -> None:
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


def _read_json(path: Path):
    """``(data, why)``. ``why`` is "" when the file parsed."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, "absent"
    except OSError as exc:
        return {}, "unreadable (%s)" % type(exc).__name__
    try:
        data = json.loads(raw)
    except Exception:                              # noqa: BLE001 - one fault
        return {}, "malformed"
    return (data, "") if isinstance(data, dict) else ({}, "malformed")


# ------------------------------------------------------------- the rules
def _verdict(data: dict, now: datetime):
    """May this spool go in the email? ``(ok, why, age_s)``."""
    if not str(data.get("code") or "") or not _is_id(data.get("id")):
        return False, "malformed", None
    try:
        pushed = _utc(datetime.fromisoformat(str(data.get("pushed_at") or "")))
    except Exception:                              # noqa: BLE001 - one fault
        return False, "malformed", None
    age = (now - pushed).total_seconds()
    if age < -SKEW_S:
        return False, "pushed in the future", age
    if age > FRESH_S:
        return False, "stale", age
    # A belt beside the age: the push names the Sunday it is FOR, and this
    # compose must be that day. A spool that survived a missed week cannot
    # ride in a later week's email even if the clock says it is young.
    if str(data.get("for") or "") != now.date().isoformat():
        return False, "for another day", age
    return True, "", age


def _line(code: str) -> str:
    """The two lines appended AFTER "Encrypted size: N bytes.". Never the
    word "key": the master key is not in this email and never was, and
    nothing here may be mistaken for it."""
    return "\n\n%s%s\n%s\n" % (LEAD, code, NOTE)


# ------------------------------------------- what backup_main calls (2)
def claim(now=None):
    """``(line, token)`` -- the extra body text, and something to settle.

    Deletes the spool in EVERY branch. NEVER RAISES: on any trouble at all
    the answer is ("", token) and the email goes out exactly as it always
    has.
    """
    try:
        now = _utc(now)
        data, why = _read_json(PENDING)
        if why == "absent":
            return "", None
        _delete(PENDING)                    # fresh, stale or junk: it goes
        code_id = str(data.get("id") or "") if _is_id(data.get("id")) else ""
        if why:
            return "", {"id": code_id, "status": "stale", "why": why}
        ok, bad, _age = _verdict(data, now)
        if not ok:
            return "", {"id": code_id, "status": "stale", "why": bad}
        return _line(str(data["code"])), {"id": code_id, "status": "pending"}
    except BaseException:                   # noqa: BLE001 - see the docstring
        try:
            _delete(PENDING)
        except BaseException:               # noqa: BLE001
            pass
        return "", {"id": "", "status": "error", "why": "claim failed"}


def settle(token, sent, now=None) -> None:
    """Write the receipt the Spark collects. NEVER RAISES.

    There is no Message-ID to quote: app/notifiers/email.py builds only
    From/To/Subject and returns SendResult(ok) -- so the receipt is the
    push id, whether the SMTP server accepted the message, and when. "sent"
    means accepted, not delivered, exactly as "backup_sent" has always
    meant in this app's audit log.
    """
    try:
        if not token:
            return
        status = str(token.get("status") or "error")
        if status == "pending":
            status = "sent" if sent else "failed"
        receipt = {"id": str(token.get("id") or ""), "status": status,
                   "at": _utc(now).isoformat()}
        why = str(token.get("why") or "")
        if why:
            receipt["why"] = why
        _write_json(RECEIPT, receipt)
    except BaseException:                   # noqa: BLE001 - see the docstring
        pass


# ------------------------------------------------------- the rehearsals
def peek(now=None):
    """What claim() WOULD do, deleting and writing nothing. ``(would, why,
    age_s, code_id)``."""
    now = _utc(now)
    data, why = _read_json(PENDING)
    if why == "absent":
        return False, "no spool", None, ""
    code_id = str(data.get("id") or "") if _is_id(data.get("id")) else ""
    if why:
        return False, why, None, code_id
    ok, bad, age = _verdict(data, now)
    return ok, (bad or ""), age, code_id


def _check_line(now=None) -> str:
    would, why, age, code_id = peek(now)
    if why == "no spool":
        return "no spool: the email would be exactly as today"
    hours = ("%.1f h" % (age / 3600.0)) if age is not None else "unknown age"
    if would:
        return ("a line WOULD be included (id %s, age %s)"
                % (code_id or "?", hours))
    return ("%s (id %s, age %s): would be deleted unread and noted"
            % (why or "not usable", code_id or "?", hours))


def _dry_run_line(now=None) -> str:
    """Compose the body the way run_backup does -- base text, then whatever
    claim would append -- and report ONLY whether the extra line is in it.
    The body itself is never printed: it would contain the code."""
    would, _why, _age, _id = peek(now)
    base = ("Attached is an encrypted snapshot of your Knightfall database.\n\n"
            "Decrypt it with your MASTER_KEY (Fernet). Keep the backup and "
            "the key in separate, safe places - together in one inbox "
            "defeats the encryption.\n\n"
            "Encrypted size: 0 bytes.")
    body = base + (_line("x" * 8) if would else "")
    return "extra line: %s" % ("would include" if body != base else "would not")


# -------------------------------------------------------------- the CLI
def _put(stdin_text: str) -> dict:
    data, why = ({}, "malformed")
    try:
        parsed = json.loads(stdin_text or "")
        if isinstance(parsed, dict):
            data, why = parsed, ""
    except Exception:                              # noqa: BLE001 - one fault
        pass
    if why or not _is_id(data.get("id")) or not str(data.get("code") or ""):
        return {"ok": False, "why": "a push needs an 8-hex id and a code"}
    _write_json(PENDING, {"id": str(data["id"]), "code": str(data["code"]),
                          "pushed_at": str(data.get("pushed_at") or ""),
                          "for": str(data.get("for") or "")})
    return {"ok": True, "id": str(data["id"])}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.strip().splitlines()[0])
        return 1
    verb = argv[0]
    try:
        if verb == "--check":
            print(_check_line())
            return 0
        if verb == "--dry-run":
            print(_dry_run_line())
            return 0
        if verb == "ping":
            print(json.dumps({"ok": True, "spool_dir": SPOOL_DIR.is_dir()}))
            return 0
        if verb == "put":
            print(json.dumps(_put(sys.stdin.read())))
            return 0
        if verb == "receipt":
            data, why = _read_json(RECEIPT)
            _delete(RECEIPT)
            print(json.dumps(data if not why else {"status": "none"}))
            return 0
        if verb == "revoke":
            code_id = argv[1] if len(argv) > 1 else ""
            if not _is_id(code_id):
                print(json.dumps({"ok": False, "why": "not an id"}))
                return 2
            data, why = _read_json(PENDING)
            hit = (not why) and str(data.get("id") or "") == code_id
            if hit:
                _delete(PENDING)
            print(json.dumps({"ok": True, "revoked": bool(hit)}))
            return 0
    except BaseException:                          # noqa: BLE001 - a verb
        print(json.dumps({"ok": False, "why": "the gate failed"}))
        return 2
    print(json.dumps({"ok": False, "why": "not a verb this gate allows"}))
    return 2


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
