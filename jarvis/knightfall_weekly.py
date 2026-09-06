"""The weekly Knightfall code, delivered in Oracle's backup email.

HIS RULING, 2026-09-06 11:00, verbatim: "I want the nightfall code to send
weekly in the same email as the encrypted back up."

The typed override code (jarvis/passphrase.py, jarvis/gate.py) is the
keyboard's break-glass past the owner gate. Until today it was issued ON
DEMAND only: typing it, or pressing "Email me a new code", mailed the next
one from the Spark to his notice address. The Oracle box already mails him
"Knightfall encrypted backup" every Sunday at 08:30 UTC; this lane puts
one extra line in that email.

THE SHAPE: PUSH FROM THE SPARK, TWO-PHASE, WITH A RECEIPT.

  Sun 07:30 UTC   the Spark (jarvis-knightfall-push.timer -> ``push``)
      makes an eight-character code, hashes it, hands the PLAINTEXT to
      Oracle over ssh (a 0600 spool file, /home/opc/knightfall/spool/
      pending.json), and only on a clean put stores the hash in
      people.json as PENDING, beside the current one. From this moment
      BOTH codes open the door.
  Sun 08:30 UTC   Oracle's own timer runs app.backup_main, which (one
      additive block) reads the spool if present and fresh, appends
      "Jarvis override code for this week: <code>" after the size line,
      sends, deletes the spool and writes a receipt. No spool -> the
      email is byte-for-byte what it always was.
  Sun 09:30 UTC   the Spark (jarvis-knightfall-pull.timer, Sun..Wed ->
      ``pull``) fetches the receipt. "sent" and the id matches -> the
      pending hash is PROMOTED to current and the previous one is gone.
      "failed" / "stale" / "error" -> the pending is dropped and the old
      code stands. No receipt by Wednesday 09:30 UTC -> dropped, and the
      caption says so.

WHY THE PLAINTEXT NEVER TOUCHES THE SPARK'S DISK. It is made here, written
into an ssh child's stdin, and deleted. It lives in exactly two places:
the Oracle spool for about an hour, and his inbox. No log line, toast,
receipt, state file or audit row carries it -- only its length, the push
id (eight hex characters that name the push and say nothing about the
code) and a hash prefix. tests/test_knightfall_weekly.py greps every file
this lane can write and every log record at DEBUG for it.

WHY NEVER ZERO WORKING CODES. Every path here leaves at least one hash on
disk and in memory that opens the door: the current hash is never
touched by a push, a promote replaces it only with a hash that has just
been proved deliverable, and a drop only forgets the pending. The
property is the same one _knightfall_rotate already has, and it is tested
as a property over random sequences.

TWO CLOCKS, WRITTEN IN UTC. Oracle's zone is literally "GMT"; the Spark
is America/Chicago and moves on Nov 1. Every timer here is written with a
UTC suffix so nothing shifts relative to Oracle's 08:30; the only visible
effect of the change is that his email lands at 02:30 CST instead of
03:30 CDT.

WHAT TOUCHES THE LIVE APP. ``gate.registry`` is the copy loaded at boot
and nothing watches the file, so after every write here the issuer sends
the existing NON-SPEAKING ``people reload`` verb over the command socket
(jarvis/cmdsock.py: a read of the file, no turn, no speech). The drawer's
typed check now reads the file itself (jarvis/app.py knightfall_code), so
even without the belt a weekly code is honoured; the belt keeps the
caption and the in-memory copy in step.

THE SSH IS A SEAM. ``Lane.ssh(verb, stdin)`` is injected; the real one
(``run_ssh``) is Popen + communicate(timeout) + kill-without-wait, the
discipline jarvis/tools/oracle.py already has, and nothing in the test
suite opens a socket. The far side answers four verbs: ``put`` (JSON on
stdin), ``receipt`` (prints and deletes), ``revoke <id>``, ``ping``.

NOTHING HERE INSTALLS A TIMER. The unit files ship in scripts/systemd/
and scripts/setup_knightfall_weekly.sh installs them when he runs it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from jarvis import identity
from jarvis import passphrase as pp
from jarvis.config import PATHS
from jarvis.identity import FAULT_MALFORMED, FAULT_UNREADABLE, ROLE_OWNER, Registry
from jarvis.logs import get_logger

log = get_logger("knightfall_weekly")

# ---------------------------------------------------------------- knobs
SSH_BIN = "ssh"
SSH_TIMEOUT_S = 20.0
PUT_TRIES = 3
RETRY_WAIT_S = 300.0
# The push window, UTC: Saturday 20:00 to Sunday 08:15. Persistent=true
# fires a missed timer at boot, and a push after Oracle has composed only
# makes a spool Oracle will find stale next Sunday.
WINDOW_OPEN = (20, 0)
WINDOW_CLOSE = (8, 15)
# No receipt by the Wednesday after the push, 09:30 UTC -> dropped.
DEADLINE_DAYS = 3
DEADLINE_TIME = (9, 30)
# The far side. The forced-command key (deploy/oracle/) ignores the
# command and reads the verb out of SSH_ORIGINAL_COMMAND, so this string
# has to end in the verb either way.
REMOTE_DIR = "/home/opc/knightfall"
REMOTE_PYTHON = ".venv/bin/python"
REMOTE_MODULE = "app.jarvis_override"
VERBS = ("put", "receipt", "ping")

# ------------------------------------------------------------ the lines
# One short sentence each: the drawer caption under the Knightfall button
# and the `jarvis -q status` sheet carry them, and the strip they share
# holds ~118 characters at his geometry (tests pin the length). {t} is a
# local time like "Sun 04:30 CDT". None ever carries a code.
LINE_PUSHED = ("Weekly: this week's code is in Sunday's backup email; both "
               "codes work until Jarvis confirms it went out.")
LINE_PROMOTED = ("Weekly: this week's code took effect {t}; the previous code "
                 "no longer works.")
LINE_PROMOTED_OVER_ROTATE = ("Weekly: this week's code took effect {t} and "
                             "replaced the code from your rotate; older codes "
                             "are dead.")
LINE_PROMOTED_BY_USE = ("Weekly: this week's code took effect when you typed "
                        "it {t}; the previous code no longer works.")
LINE_UNREACHABLE = ("Weekly: no code this week; Oracle was unreachable {t}; "
                    "your current code stands.")
LINE_UNSTORED = ("Weekly: could not store this week's code; Sunday's email "
                 "carries no code; your current code stands.")
LINE_DEAD_CODE = ("Weekly: the code in Sunday's email will NOT work (it could "
                  "not be stored here); your current code stands.")
LINE_SEND_FAILED = ("Weekly: Oracle's backup email failed on Sunday; no weekly "
                    "code; your current code stands.")
LINE_NO_RECEIPT = ("Weekly: no receipt for Sunday's code by Wednesday; it was "
                   "dropped; your previous code stands.")
LINE_MISSED = ("Weekly: missed this week; the Spark was off at push time ({t}); "
               "your current code stands.")
LINE_REPLACED = ("Weekly: the code in Sunday's email was replaced by a forced "
                 "push; it will not work.")
LINE_STALE = ("Weekly: Sunday's code reached Oracle too late and was discarded "
              "unread; your current code stands.")
LINE_ORACLE_ERROR = ("Weekly: Oracle could not add the code on Sunday (error); "
                     "your current code stands.")
LINE_UNREADABLE = ("Weekly: the people file could not be read; nothing was "
                   "changed; fix it at the keyboard.")


# ------------------------------------------------------------- the seam
@dataclass
class SshReply:
    ok: bool
    out: str = ""
    reason: str = ""      # "" on success; "timeout" / "no-ssh" / "failed"


@dataclass
class Outcome:
    ok: bool
    status: str
    line: str


@dataclass
class Lane:
    """Everything a push or a pull needs, all injectable. ``ssh(verb,
    stdin) -> SshReply`` is the one thing that leaves this machine."""

    registry_path: Path
    state_path: Path
    owner: str = ""
    ssh: Optional[Callable[[str, Optional[str]], SshReply]] = None
    reload_live: Optional[Callable[[], str]] = None
    sleep: Callable[[float], None] = time.sleep
    new_code: Callable[[], str] = pp.new_code
    put_tries: int = PUT_TRIES


# ------------------------------------------------------------ the clock
def _utc(now) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _when(now) -> str:
    """A local stamp for a caption: "Sun 04:30 CDT"."""
    try:
        return _utc(now).astimezone().strftime("%a %H:%M %Z")
    except Exception:  # noqa: BLE001 - a stamp, never a raise
        return _utc(now).strftime("%a %H:%M UTC")


def target_sunday(now) -> date:
    """The Sunday this push is FOR: today if it is Sunday, else the next."""
    d = _utc(now).date()
    return d + timedelta(days=(6 - d.weekday()) % 7)


def week_key(day: date) -> str:
    iso = day.isocalendar()
    return "%d-W%02d" % (iso[0], iso[1])


def in_window(now) -> bool:
    t = _utc(now)
    hm = (t.hour, t.minute)
    if t.weekday() == 5:
        return hm >= WINDOW_OPEN
    if t.weekday() == 6:
        return hm <= WINDOW_CLOSE
    return False


def drop_deadline(since_iso) -> datetime:
    """Wednesday 09:30 UTC after the Sunday the push was for."""
    since = _utc(datetime.fromisoformat(str(since_iso)))
    day = target_sunday(since) + timedelta(days=DEADLINE_DAYS)
    return datetime(day.year, day.month, day.day, *DEADLINE_TIME, tzinfo=timezone.utc)


# ------------------------------------------------------------ the state
def read_state(path) -> dict:
    """The lane's narration, or {}. NEVER RAISES."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing, unreadable, malformed: no state
        return {}
    return data if isinstance(data, dict) else {}


def write_state(path, data: dict) -> None:
    """0600, atomic, in a 0700 directory. NEVER RAISES: a caption that
    could not be written is a log line, not a failed push."""
    p = Path(path)
    payload = dict(data)
    payload["format"] = 1
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(prefix=".knightfall-weekly-", suffix=".tmp",
                                   dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:  # noqa: BLE001 - see above
        log.exception("knightfall-weekly: the state file could not be written")


def _update_state(path, **changes) -> dict:
    state = read_state(path)
    state.update(changes)
    write_state(path, state)
    return state


def caption(path) -> str:
    """The one sentence the drawer and `jarvis -q status` show. "" when
    the lane has never run. NEVER RAISES, never carries a code."""
    try:
        return str(read_state(path).get("line") or "")
    except Exception:  # noqa: BLE001
        return ""


def note_promoted_by_use(path, code_id, now=None) -> None:
    """The app promoted the pending hash because he TYPED that code (he
    can only have it from the email). Record it so the pull knows the
    receipt is already handled and the caption says what happened."""
    state = read_state(path)
    if state.get("id") not in (None, "", code_id):
        log.info("knightfall-weekly: id %s promoted by use, but the state "
                 "names %s; noting it anyway", code_id, state.get("id"))
    state.update({"id": code_id, "status": "promoted-by-use",
                  "line": LINE_PROMOTED_BY_USE.format(t=_when(now)),
                  "promoted_at": _utc(now).isoformat()})
    write_state(path, state)
    log.info("knightfall-weekly: id %s promoted by use at the keyboard; "
             "the previous code is retired", code_id)


def _fp(hashed: str) -> str:
    """A fingerprint of a STORED HASH (never of a code), so a pull can tell
    whether a manual rotate happened since the push."""
    return hashlib.sha256(str(hashed or "").encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------- helpers
def _json(text) -> dict:
    try:
        data = json.loads(text or "")
    except Exception:  # noqa: BLE001 - not JSON is not an answer
        return {}
    return data if isinstance(data, dict) else {}


def _owner(lane: Lane, reg: Registry) -> str:
    if lane.owner and reg.role_of(lane.owner) == ROLE_OWNER:
        return lane.owner
    owners = reg.owners()
    return owners[0].label if owners else ""


def _reload(lane: Lane) -> None:
    """The `people reload` belt: the live app re-reads the file so its
    boot copy and the drawer caption catch up. Non-speaking, no turn."""
    fn = lane.reload_live
    if fn is None:
        return
    try:
        answer = fn()
    except Exception as exc:  # noqa: BLE001 - a belt, never a raise
        log.warning("knightfall-weekly: the live app could not be told to "
                    "re-read the people file (%s)", type(exc).__name__)
        return
    log.info("knightfall-weekly: live app told to re-read the people file: %s",
             str(answer or "")[:120])


def _ssh(lane: Lane, verb: str, stdin: Optional[str] = None) -> SshReply:
    if lane.ssh is None:
        return SshReply(False, "", "no-ssh")
    try:
        reply = lane.ssh(verb, stdin)
    except Exception as exc:  # noqa: BLE001 - the seam must not raise past here
        log.warning("knightfall-weekly: ssh %s raised %s", verb.split()[0],
                    type(exc).__name__)
        return SshReply(False, "", "failed")
    return reply if isinstance(reply, SshReply) else SshReply(False, "", "failed")


def _put(lane: Lane, payload: str, code_id: str) -> SshReply:
    tries = max(1, int(lane.put_tries))
    last = SshReply(False, "", "no-ssh")
    for i in range(tries):
        if i:
            log.warning("knightfall-weekly: put %s failed (%s); trying again "
                        "in %.0f s (%d of %d)", code_id, last.reason or "failed",
                        RETRY_WAIT_S, i + 1, tries)
            lane.sleep(RETRY_WAIT_S)
        last = _ssh(lane, "put", payload)
        if last.ok:
            answer = _json(last.out)
            if answer.get("ok") is True and answer.get("id", code_id) == code_id:
                return last
            last = SshReply(False, last.out, "failed")
    return last


# ------------------------------------------------------------- the push
def push(lane: Lane, now=None, *, force: bool = False) -> Outcome:
    """Phase 1. Make -> hash -> PUT FIRST -> store pending -> reload.

    The put comes before the store for the reason the mail comes before
    the store in _knightfall_rotate: a hash stored for a code that never
    reached Oracle would be a code that works and is nowhere; a code on
    Oracle whose hash could not be stored is revoked, and if even that
    fails he is told the emailed code will not work. Either way the
    current code is never touched.
    """
    now = _utc(now)
    reg = Registry.load(lane.registry_path)
    if not reg.usable:
        log.error("knightfall-weekly: the people file is not usable (%s); "
                  "no code was made", reg.fault)
        _update_state(lane.state_path, status="unreadable", line=LINE_UNREADABLE,
                      at=now.isoformat())
        return Outcome(False, "unreadable", LINE_UNREADABLE)
    owner = _owner(lane, reg)
    person = reg.person(owner) if owner else None
    if person is None:
        log.error("knightfall-weekly: no owner is enrolled; no code was made")
        return Outcome(False, "no-owner", "no owner is enrolled")
    if person.pending_code_hash and not force:
        line = ("already pending (id %s, since %s); --force replaces it"
                % (person.pending_code_id, person.pending_code_since or "?"))
        log.warning("knightfall-weekly: refused: %s", line)
        return Outcome(False, "refused", line)
    if not in_window(now):
        # THE WINDOW IS THE TIMER'S GUARD, NOT HIS. Persistent=true fires a
        # missed Sunday at boot, and an automatic push after Oracle has
        # composed only makes a spool that is stale by the next Sunday --
        # so the unattended path declines and says so. ``--force`` is a
        # deliberate act at a keyboard and goes through: the flag's help,
        # the "replaced" line and the failure table all describe a forced
        # push AFTER the 08:30 compose, and a guard that silently refused
        # it would make those three a lie. What it costs him is named
        # here and in the caption, never discovered later.
        if not force:
            log.warning("knightfall-weekly: %s is outside the push window (Sat "
                        "20:00 to Sun 08:15 UTC); nothing pushed, the current "
                        "code stands", now.isoformat())
            line = LINE_MISSED.format(t=_when(now))
            write_state(lane.state_path, {"status": "missed", "line": line,
                                          "at": now.isoformat()})
            return Outcome(False, "missed", line)
        log.warning("knightfall-weekly: --force outside the push window (%s). "
                    "Oracle composes at 08:30 UTC: if it has already composed, "
                    "the code in this week's email is the PREVIOUS push's and "
                    "this one replaces it, so the emailed code dies unused; if "
                    "it has not, this spool will be stale by next Sunday. The "
                    "current code stands either way.", now.isoformat())

    for_day = target_sunday(now)
    week = week_key(for_day)
    code = str(lane.new_code())
    length = len(code)
    hashed = pp.hash_secret(code)
    code_id = secrets.token_hex(4)
    payload = json.dumps({"id": code_id, "code": code,
                          "pushed_at": now.isoformat(), "for": for_day.isoformat()})
    del code
    log.info("knightfall-weekly: pushing a %d-character code for %s (id %s, "
             "hash %s...)", length, for_day.isoformat(), code_id, hashed[:20])
    reply = _put(lane, payload, code_id)
    del payload
    if not reply.ok:
        log.error("knightfall-weekly: Oracle was unreachable for the put (%s); "
                  "no code this week; the current code stands",
                  reply.reason or "failed")
        line = LINE_UNREACHABLE.format(t=_when(now))
        write_state(lane.state_path, {"status": "unreachable", "line": line,
                                      "id": code_id, "week": week,
                                      "at": now.isoformat()})
        return Outcome(False, "unreachable", line)

    replaced = person.pending_code_id if (force and person.pending_code_hash) else ""
    since = now.isoformat()
    ok, why = identity.locked_update(
        lane.registry_path,
        lambda r: r.set_pending_code(owner, hashed, code_id, since))
    if not ok:
        log.error("knightfall-weekly: the pending hash could not be stored (%s); "
                  "revoking the spool on Oracle", why)
        rev = _ssh(lane, "revoke %s" % code_id)
        if rev.ok:
            log.error("knightfall-weekly: spool %s revoked; Sunday's email "
                      "carries no code; the current code stands", code_id)
            status, line = "unstored", LINE_UNSTORED
        else:
            log.error("knightfall-weekly: the revoke failed too (%s): the code "
                      "in Sunday's email will NOT work; the current code stands",
                      rev.reason or "failed")
            status, line = "dead-code", LINE_DEAD_CODE
        write_state(lane.state_path, {"status": status, "line": line,
                                      "id": code_id, "week": week,
                                      "at": now.isoformat()})
        return Outcome(False, status, line)

    _reload(lane)
    write_state(lane.state_path, {
        "status": "pushed", "line": LINE_PUSHED, "id": code_id, "week": week,
        "for": for_day.isoformat(), "pushed_at": since,
        "replaced_id": replaced, "current_fp": _fp(person.code_hash),
        "receipt": None})
    log.info("knightfall-weekly: pushed id %s for %s%s; both codes are honoured "
             "until the receipt", code_id, for_day.isoformat(),
             (" (replacing %s)" % replaced) if replaced else "")
    return Outcome(True, "pushed", LINE_PUSHED)


# ------------------------------------------------------------- the pull
def pull(lane: Lane, now=None) -> Outcome:
    """Phase 3. Fetch the receipt; promote, drop, or keep waiting.

    THE REGISTRY IS THE TRUTH and the state file only narrates: a pending
    hash the state has forgotten is still resolved, and a state that says
    "pushed" for a hash that is no longer pending does no harm.
    """
    now = _utc(now)
    state = read_state(lane.state_path)
    reg = Registry.load(lane.registry_path)
    if not reg.usable:
        if reg.fault_kind in (FAULT_MALFORMED, FAULT_UNREADABLE):
            log.error("knightfall-weekly: the people file could not be read "
                      "(%s); nothing changed; fix it at the keyboard", reg.fault)
            _update_state(lane.state_path, line=LINE_UNREADABLE)
            return Outcome(False, "unreadable", LINE_UNREADABLE)
        return Outcome(True, "idle", caption(lane.state_path))
    owner = _owner(lane, reg)
    person = reg.person(owner) if owner else None
    if person is None:
        return Outcome(True, "idle", caption(lane.state_path))
    pending_id = person.pending_code_id if person.pending_code_hash else ""
    by_use = (state.get("status") == "promoted-by-use"
              and state.get("receipt") is None)
    if not pending_id and not by_use:
        return Outcome(True, "idle", caption(lane.state_path))

    reply = _ssh(lane, "receipt")
    receipt = _json(reply.out) if reply.ok else {}
    rstatus = str(receipt.get("status") or "none")
    rid = str(receipt.get("id") or "")
    if not reply.ok:
        log.warning("knightfall-weekly: the receipt could not be fetched (%s)",
                    reply.reason or "failed")

    if rstatus == "none":
        return _drop_if_due(lane, now, owner, person, pending_id) or \
            _waiting(lane, pending_id)

    # a receipt for THE pending push
    if pending_id and rid == pending_id:
        if rstatus == "sent":
            return _promote(lane, now, state, owner, person, pending_id, receipt)
        return _drop(lane, owner, pending_id, rstatus, receipt)

    # a receipt for a push already promoted by typing it
    if by_use and rid == str(state.get("id") or ""):
        _update_state(lane.state_path, receipt=receipt)
        log.info("knightfall-weekly: receipt for id %s already handled (it was "
                 "promoted when typed)", rid)
        return Outcome(True, "promoted-by-use", caption(lane.state_path))

    # a receipt for a push this week's --force replaced: the emailed code
    # is dead, and the new pending still waits on its own receipt
    if rid and rid == str(state.get("replaced_id") or "") and rstatus == "sent":
        log.warning("knightfall-weekly: receipt for id %s is for a push that "
                    "--force replaced; the code in Sunday's email will not work",
                    rid)
        _update_state(lane.state_path, line=LINE_REPLACED, replaced_receipt=receipt)
        return _drop_if_due(lane, now, owner, person, pending_id) or \
            Outcome(True, "replaced", LINE_REPLACED)

    log.warning("knightfall-weekly: receipt for id %s (%s) matched nothing "
                "(pending is %s); ignored", rid or "?", rstatus,
                pending_id or "none")
    return _drop_if_due(lane, now, owner, person, pending_id) or \
        _waiting(lane, pending_id)


def _waiting(lane: Lane, pending_id: str) -> Outcome:
    if pending_id:
        log.info("knightfall-weekly: no receipt yet for id %s; both codes stay "
                 "honoured", pending_id)
    return Outcome(True, "waiting", caption(lane.state_path))


def _drop_if_due(lane, now, owner, person, pending_id) -> Optional[Outcome]:
    if not pending_id:
        return None
    try:
        due = drop_deadline(person.pending_code_since)
    except Exception:  # noqa: BLE001 - an unreadable stamp: drop now
        log.warning("knightfall-weekly: the pending stamp %r is unreadable; "
                    "treating the deadline as reached", person.pending_code_since)
        due = now
    if now < due:
        return None
    log.error("knightfall-weekly: no receipt for id %s by the deadline (%s); "
              "dropping it; the previous code stands", pending_id,
              due.isoformat())
    return _drop(lane, owner, pending_id, "no-receipt", None)


def _drop(lane, owner, pending_id, why, receipt) -> Outcome:
    ok, fault = identity.locked_update(
        lane.registry_path, lambda r: r.drop_pending(owner, pending_id))
    if not ok:
        log.error("knightfall-weekly: could not drop id %s (%s); nothing changed",
                  pending_id, fault)
        _update_state(lane.state_path, line=LINE_UNREADABLE)
        return Outcome(False, "unreadable", LINE_UNREADABLE)
    _reload(lane)
    line = {"failed": LINE_SEND_FAILED, "stale": LINE_STALE,
            "no-receipt": LINE_NO_RECEIPT}.get(why, LINE_ORACLE_ERROR)
    status = "dropped-" + (why if why in ("failed", "stale", "no-receipt") else "error")
    _update_state(lane.state_path, status=status, line=line, receipt=receipt,
                  id=pending_id)
    log.error("knightfall-weekly: dropped id %s (%s); the previous code stands",
              pending_id, why)
    return Outcome(True, status, line)


def _promote(lane, now, state, owner, person, pending_id, receipt) -> Outcome:
    rotated = bool(state.get("current_fp")) and \
        state.get("current_fp") != _fp(person.code_hash)
    ok, fault = identity.locked_update(
        lane.registry_path, lambda r: r.promote_pending(owner, pending_id))
    if not ok:
        log.error("knightfall-weekly: could not promote id %s (%s); nothing "
                  "changed; both codes stay honoured", pending_id, fault)
        _update_state(lane.state_path, line=LINE_UNREADABLE)
        return Outcome(False, "unreadable", LINE_UNREADABLE)
    _reload(lane)
    line = (LINE_PROMOTED_OVER_ROTATE if rotated else LINE_PROMOTED).format(t=_when(now))
    _update_state(lane.state_path, status="promoted", line=line, receipt=receipt,
                  id=pending_id, promoted_at=now.isoformat())
    log.info("knightfall-weekly: promoted id %s; previous code retired%s",
             pending_id, " (it had been rotated since the push)" if rotated else "")
    return Outcome(True, "promoted", line)


# ------------------------------------------------------------ real ssh
def remote_command(verb: str) -> str:
    """The far side's command for one verb. Refuses anything that is not
    exactly one of the four verbs (a revoke carries an 8-hex id), so no
    string from a config or a receipt can ever become shell on Oracle."""
    verb = str(verb or "").strip()
    parts = verb.split()
    ok = (len(parts) == 1 and parts[0] in VERBS) or \
        (len(parts) == 2 and parts[0] == "revoke"
         and identity.PENDING_ID_RX.match(parts[1]) is not None)
    if not ok:
        raise ValueError("not a verb this lane sends: %r" % (verb,))
    return "cd %s && exec %s -m %s %s" % (REMOTE_DIR, REMOTE_PYTHON, REMOTE_MODULE, verb)


def ssh_argv(target: str, key_file: str, command: str, timeout_s: float) -> list:
    """jarvis/tools/oracle.ssh_argv without ``-n``: the put feeds stdin."""
    connect = max(1, int(timeout_s) - 1)
    argv = [SSH_BIN,
            "-o", "BatchMode=yes",
            "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no",
            "-o", "NumberOfPasswordPrompts=0",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=%d" % connect,
            "-o", "IdentitiesOnly=yes"]
    if key_file:
        argv += ["-i", key_file]
    argv += ["--", target, command]
    return argv


def run_ssh(target: str, key_file: str, command: str, stdin_text: Optional[str],
            timeout_s: float = SSH_TIMEOUT_S) -> SshReply:
    """ONE bounded round trip: Popen + communicate(timeout) + kill without
    waiting (jarvis/tools/oracle.run_ssh's discipline). stdin carries the
    put's JSON and nothing else; stderr is logged by TYPE and length."""
    argv = ssh_argv(target, key_file, command, timeout_s)
    log.info("knightfall-weekly: ssh %s (%s, %.0fs budget)", target,
             command.rsplit(" ", 1)[-1] if command else "?", timeout_s)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            text=True, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("knightfall-weekly: ssh unavailable: %s", type(exc).__name__)
        return SshReply(False, "", "no-ssh")
    try:
        out, err = proc.communicate(input=stdin_text, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        log.warning("knightfall-weekly: ssh did not answer in %.0fs; not waiting "
                    "for it", timeout_s)
        return SshReply(False, "", "timeout")
    if proc.returncode != 0:
        log.warning("knightfall-weekly: ssh rc=%s (%d bytes of stderr)",
                    proc.returncode, len(err or ""))
        return SshReply(False, out or "", "failed")
    return SshReply(True, out or "", "")


def make_ssh(target: str, key_file: str, timeout_s: float = SSH_TIMEOUT_S):
    def ssh(verb, stdin=None):
        return run_ssh(target, key_file, remote_command(verb), stdin, timeout_s)
    return ssh


def reload_live_app() -> str:
    """The `people reload` verb over the command socket, if Jarvis is up.
    Non-speaking, not a turn (jarvis/cmdsock.py). Never raises."""
    sock = PATHS.COMMAND_SOCK
    if not Path(sock).exists():
        return "not running"
    from jarvis.cmdsock import ask
    try:
        for msg in ask(sock, "people reload", quiet=True, timeout=10.0):
            if isinstance(msg, dict) and msg.get("kind") == "reply":
                return str(msg.get("text") or "")
    except ConnectionError:
        return "not running"
    except Exception as exc:  # noqa: BLE001 - a belt, never a raise
        return "failed (%s)" % type(exc).__name__
    return ""


def lane_from_config() -> Optional[Lane]:
    """The real lane: oracle.host / oracle.user / oracle.key_path from his
    assistant.json (read through AssistantConfig, which redacts), the real
    people.json and state path, the real ssh and the real reload."""
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig.load()
    host = str(cfg.get("oracle.host", "") or "").strip()
    user = str(cfg.get("oracle.user", "") or "").strip()
    # A DEDICATED key for this lane when he has made one (deploy/oracle/
    # jarvis-override-gate.sh): the forced-command key that can only speak
    # the four verbs. Until then the voice lane's key, which is root on
    # the dead-man's switch -- the reason the restricted one is offered.
    key = str(cfg.get("knightfall_weekly.key_path", "") or "").strip() or \
        str(cfg.get("oracle.key_path", "") or "").strip()
    key = os.path.expanduser(key)
    if not host or not key:
        print("knightfall-weekly: oracle.host and oracle.key_path are needed in "
              "~/.config/jarvis/assistant.json; nothing done", file=sys.stderr)
        return None
    target = "%s@%s" % (user, host) if user else host
    return Lane(registry_path=PATHS.OWNER_REGISTRY, state_path=PATHS.KNIGHTFALL_WEEKLY,
                owner=identity.owner_label(cfg), ssh=make_ssh(target, key),
                reload_live=reload_live_app)


# --------------------------------------------------------------- the CLI
EXIT = {"pushed": 0, "promoted": 0, "promoted-by-use": 0, "waiting": 0,
        "idle": 0, "dropped-failed": 0, "dropped-stale": 0, "dropped-error": 0,
        "dropped-no-receipt": 0, "replaced": 0,
        "refused": 1, "missed": 1, "unstored": 1, "dead-code": 1, "no-owner": 1,
        "unreachable": 2, "unreadable": 2}


def main(argv=None, *, lane: Optional[Lane] = None, now=None) -> int:
    ap = argparse.ArgumentParser(
        prog="knightfall_weekly",
        description="the weekly Knightfall code in Oracle's backup email")
    ap.add_argument("verb", nargs="?", choices=["push", "pull"],
                    help="push: make this week's code and hand it to Oracle "
                         "(Sunday 07:30 UTC); pull: fetch the receipt and "
                         "promote or drop (Sun..Wed 09:30 UTC)")
    ap.add_argument("--force", action="store_true",
                    help="push again this week, replacing the pending code. "
                         "After Oracle has composed (08:30 UTC) the code in "
                         "Sunday's email DIES: it is the previous push's")
    ap.add_argument("--rehearse", action="store_true",
                    help="make and hash a code, print its length and hash "
                         "prefix, push and store nothing")
    ap.add_argument("--probe", action="store_true",
                    help="one ssh ping to Oracle's gate; writes nothing")
    ap.add_argument("--status", action="store_true",
                    help="print the caption the drawer shows")
    args = ap.parse_args(argv)
    if lane is None:
        lane = lane_from_config()
        if lane is None:
            return 2

    if args.status:
        print(caption(lane.state_path) or "Weekly: nothing recorded yet.")
        return 0
    if args.rehearse:
        code = str(lane.new_code())
        hashed = pp.hash_secret(code)
        n = len(code)
        del code
        t = _utc(now)
        print("rehearsal: a code of %d characters was made and hashed (hash "
              "prefix %s...); NOT pushed, NOT stored. The push window is %s "
              "now (Sat 20:00 to Sun 08:15 UTC); the next push is for %s."
              % (n, hashed[:24], "open" if in_window(t) else "closed",
                 target_sunday(t).isoformat()))
        return 0
    if args.probe:
        reply = _ssh(lane, "ping")
        answer = _json(reply.out) if reply.ok else {}
        if reply.ok and answer.get("ok") is True:
            print("Oracle gate answers: ok (spool dir %s)"
                  % ("present" if answer.get("spool_dir") else "MISSING"))
            return 0
        print("Oracle gate does not answer (%s)" % (reply.reason or "no ok"))
        return 2
    if args.verb == "push":
        out = push(lane, now, force=args.force)
    elif args.verb == "pull":
        out = pull(lane, now)
    else:
        ap.print_help()
        return 1
    print("%s: %s" % (out.status, out.line))
    return EXIT.get(out.status, 1)


if __name__ == "__main__":           # pragma: no cover - the unit's entry
    raise SystemExit(main())
