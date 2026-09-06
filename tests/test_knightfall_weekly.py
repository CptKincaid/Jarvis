"""The weekly Knightfall code, pushed from the Spark into Oracle's backup
email (Hunter, 2026-09-06 11:00, verbatim: "I want the nightfall code to
send weekly in the same email as the encrypted back up").

TWO PHASES, and this file drives both from the Spark's side with the far
side in memory. ``FakeOracle`` answers the same three verbs the real gate
on Oracle answers (put / receipt / revoke) and ``compose()`` does what
``app.backup_main`` does at 08:30 UTC: claim the spool, send, settle a
receipt. NOTHING HERE OPENS A SOCKET: the ssh runner is the injected seam
``Lane.ssh``, and the one test of the real runner replaces ``Popen``.

Every row of the failure table is a test. Two tests are the ones that
matter most: at no point in ANY sequence is there zero working codes, and
the plaintext never lands on the Spark's disk or in any log at any level.

The codes are invented (x's and digits); the registry is a tmp_path one.
"""
from __future__ import annotations

import json
import logging
import random
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis
from jarvis import gate as gt
from jarvis import identity
from jarvis import knightfall_weekly as kw
from jarvis import passphrase as pp
from jarvis.identity import Registry
from tests.test_owner_gate import FAKE_CODE, _registry

UTC = timezone.utc
SAT = datetime(2026, 9, 12, 20, 30, tzinfo=UTC)          # inside the window
SUN = datetime(2026, 9, 13, 7, 30, tzinfo=UTC)           # the push
COMPOSE = datetime(2026, 9, 13, 8, 30, tzinfo=UTC)       # Oracle sends
PULL = datetime(2026, 9, 13, 9, 30, tzinfo=UTC)          # the pull
MON = PULL + timedelta(days=1)
TUE = PULL + timedelta(days=2)
WED = PULL + timedelta(days=3)                           # the drop deadline

WEEKLY = "xxx777yy"          # what the fake generator makes; never real
FRESH_S = 36 * 3600

REPO = Path(jarvis.__file__).parent.parent


# ------------------------------------------------------------------ fakes
class FakeOracle:
    """The far side, in memory. ``spool`` is spool/pending.json, ``receipt``
    is spool/receipt.json; ``calls`` is every ssh round trip in order.

    THE RECEIPT IS READ AND ACKED SEPARATELY (2026-09-06, defect B): a
    ``receipt`` leaves the file where it is and only ``receipt --ack <id>``
    deletes it, so a Spark that read one and then failed to write its own
    registry has not lost it. This models deploy/oracle/jarvis_override.py
    verb for verb, and tests/test_knightfall_weekly_defects.py pins the
    real module against the same rules.
    """

    def __init__(self):
        self.spool = None
        self.receipt = None
        self.calls = []
        self.down = False
        self.fail_puts = 0
        self.fail_revoke = False
        self.fail_receipt = False
        self.fail_ack = False

    def __call__(self, verb, stdin=None):
        self.calls.append((verb, stdin))
        if self.down:
            return kw.SshReply(False, "", "unreachable")
        if verb == "put":
            if self.fail_puts > 0:
                self.fail_puts -= 1
                return kw.SshReply(False, "", "timeout")
            self.spool = json.loads(stdin)
            return kw.SshReply(True, json.dumps({"ok": True, "id": self.spool["id"]}))
        if verb.startswith("receipt --ack "):
            if self.fail_ack:
                return kw.SshReply(False, "", "timeout")
            code_id = verb.split()[2]
            hit = bool(self.receipt and self.receipt.get("id") == code_id)
            if hit:
                self.receipt = None
            return kw.SshReply(True, json.dumps({"ok": True, "acked": hit}))
        if verb == "receipt":
            if self.fail_receipt:
                return kw.SshReply(False, "", "timeout")
            return kw.SshReply(True, json.dumps(self.receipt or {"status": "none"}))
        if verb.startswith("revoke "):
            if self.fail_revoke:
                return kw.SshReply(False, "", "timeout")
            code_id = verb.split()[1]
            hit = bool(self.spool and self.spool.get("id") == code_id)
            if hit:
                self.spool = None
            return kw.SshReply(True, json.dumps({"ok": True, "revoked": hit}))
        if verb == "ping":
            return kw.SshReply(True, json.dumps({"ok": True, "spool_dir": True}))
        return kw.SshReply(False, "", "failed")

    def compose(self, now, send_ok=True):
        """app.backup_main at 08:30 UTC: claim (delete the spool either
        way), send, settle. Returns the extra line the email carried."""
        spool, self.spool = self.spool, None
        if spool is None:
            return ""
        pushed = datetime.fromisoformat(spool["pushed_at"])
        age = (now - pushed).total_seconds()
        if not (-3600 <= age <= FRESH_S) or spool["for"] != now.date().isoformat():
            self.receipt = {"id": spool["id"], "status": "stale",
                            "at": now.isoformat()}
            return ""
        self.receipt = {"id": spool["id"],
                        "status": "sent" if send_ok else "failed",
                        "at": now.isoformat()}
        return "\n\nJarvis override code for this week: %s\n" % spool["code"]


def _lane(tmp_path, oracle=None, *, code=WEEKLY):
    _registry(tmp_path, code=True)
    reloads = []
    sleeps = []
    lane = kw.Lane(registry_path=tmp_path / "people.json",
                   state_path=tmp_path / "state" / "knightfall-weekly.json",
                   owner="hunter", ssh=oracle or FakeOracle(),
                   reload_live=lambda: reloads.append(1) or "reloaded",
                   sleep=sleeps.append,
                   # A FIXED STRING OR A GENERATOR. ``lambda: code`` alone
                   # handed the property test's fresh_code FUNCTION back to
                   # push, which str()'d it into the same 109-character
                   # "<function fresh at 0x...>" on every push -- so every
                   # code in the sequence was identical, and the shadow
                   # bookkeeping discarded a code that still worked. The
                   # invariant it claimed to prove was never exercised.
                   new_code=(code if callable(code) else (lambda: code)))
    lane.reloads, lane.sleeps = reloads, sleeps
    return lane


def _works(lane, plain) -> bool:
    return gt.check_override_code(Registry.load(lane.registry_path), plain)[0] == "hunter"


def _pending_id(lane) -> str:
    return Registry.load(lane.registry_path).person("hunter").pending_code_id


def _state(lane) -> dict:
    return kw.read_state(lane.state_path)


# ------------------------------------------------------------ the clock
def test_the_target_sunday_and_the_week_key():
    assert kw.target_sunday(SAT) == SUN.date()
    assert kw.target_sunday(SUN) == SUN.date()
    assert kw.target_sunday(MON) == (SUN + timedelta(days=7)).date()
    assert kw.week_key(SUN.date()) == "2026-W37"
    assert kw.week_key(kw.target_sunday(SAT)) == "2026-W37"


@pytest.mark.parametrize("when,inside", [
    (SAT, True), (SUN, True),
    (datetime(2026, 9, 12, 19, 59, tzinfo=UTC), False),
    (datetime(2026, 9, 13, 8, 15, tzinfo=UTC), True),
    (datetime(2026, 9, 13, 8, 16, tzinfo=UTC), False),
    (MON, False), (COMPOSE, False),
])
def test_the_push_window_is_saturday_evening_to_before_oracle_composes(when, inside):
    assert kw.in_window(when) is inside


def test_the_drop_deadline_is_the_wednesday_after_the_push():
    assert kw.drop_deadline(SUN.isoformat()) == WED
    assert kw.drop_deadline(SAT.isoformat()) == WED


# ------------------------------------------------------- the happy week
def test_the_happy_week_pushes_stores_pending_and_promotes_on_receipt(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)

    out = kw.push(lane, SUN)
    assert out.ok and out.status == "pushed", out
    verb, stdin = oracle.calls[0]
    assert verb == "put"
    spool = json.loads(stdin)
    assert set(spool) == {"id", "code", "pushed_at", "for"}
    assert spool["code"] == WEEKLY and spool["for"] == "2026-09-13"
    assert spool["pushed_at"] == SUN.isoformat()
    assert len(spool["id"]) == 8 and int(spool["id"], 16) >= 0
    assert _pending_id(lane) == spool["id"]
    assert _works(lane, FAKE_CODE) and _works(lane, WEEKLY), "both honoured"
    assert lane.reloads == [1], "the live app was told to re-read the file"
    st = _state(lane)
    assert st["status"] == "pushed" and st["id"] == spool["id"]
    assert st["week"] == "2026-W37"
    assert kw.caption(lane.state_path).startswith("Weekly: ")
    assert "both codes work" in kw.caption(lane.state_path)

    line = oracle.compose(COMPOSE)
    assert WEEKLY in line and oracle.spool is None

    out = kw.pull(lane, PULL)
    assert out.ok and out.status == "promoted", out
    # READ, then written down here, and only THEN acked away (defect B)
    assert [v for v, _ in oracle.calls][-2:] == \
        ["receipt", "receipt --ack %s" % st["id"]]
    assert oracle.receipt is None
    assert _works(lane, WEEKLY) and not _works(lane, FAKE_CODE)
    assert _pending_id(lane) == ""
    assert lane.reloads == [1, 1]
    assert "took effect" in kw.caption(lane.state_path)
    assert "no longer works" in kw.caption(lane.state_path)

    calls = len(oracle.calls)
    out = kw.pull(lane, MON)
    assert out.status == "idle" and len(oracle.calls) == calls, "no ssh"


def test_the_put_is_retried_and_the_store_waits_for_a_clean_put(tmp_path):
    oracle = FakeOracle()
    oracle.fail_puts = 2
    lane = _lane(tmp_path, oracle)
    out = kw.push(lane, SUN)
    assert out.status == "pushed"
    assert [v for v, _ in oracle.calls] == ["put", "put", "put"]
    assert lane.sleeps == [kw.RETRY_WAIT_S, kw.RETRY_WAIT_S]
    assert _pending_id(lane) == oracle.spool["id"]


# --------------------------------------------------- the failure table
def test_row1_oracle_down_at_push_stores_nothing(tmp_path):
    oracle = FakeOracle()
    oracle.down = True
    lane = _lane(tmp_path, oracle)
    out = kw.push(lane, SUN)
    assert not out.ok and out.status == "unreachable"
    assert [v for v, _ in oracle.calls] == ["put", "put", "put"]
    assert oracle.spool is None and _pending_id(lane) == ""
    assert _works(lane, FAKE_CODE) and not _works(lane, WEEKLY)
    assert lane.reloads == []
    assert "Oracle was unreachable" in kw.caption(lane.state_path)
    assert "current code stands" in kw.caption(lane.state_path)


def test_row2_store_fails_so_the_spool_is_revoked(tmp_path, monkeypatch):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    monkeypatch.setattr(kw.identity, "locked_update",
                        lambda *a, **k: (False, "the people file could not be written"))
    out = kw.push(lane, SUN)
    assert not out.ok and out.status == "unstored"
    verbs = [v for v, _ in oracle.calls]
    assert verbs[0] == "put" and verbs[-1].startswith("revoke ")
    assert oracle.spool is None, "the plaintext no longer waits on Oracle"
    assert _pending_id(lane) == ""
    assert "carries no code" in kw.caption(lane.state_path)


def test_row2b_store_and_revoke_both_fail_names_the_dead_code(tmp_path, monkeypatch):
    oracle = FakeOracle()
    oracle.fail_revoke = True
    lane = _lane(tmp_path, oracle)
    monkeypatch.setattr(kw.identity, "locked_update",
                        lambda *a, **k: (False, "the people file could not be written"))
    out = kw.push(lane, SUN)
    assert not out.ok and out.status == "dead-code"
    assert oracle.spool is not None, "the code will go out and will not work"
    assert "will NOT work" in kw.caption(lane.state_path)
    assert _works(lane, FAKE_CODE)


def test_row3_oracles_send_failed_drops_the_pending(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE, send_ok=False)
    out = kw.pull(lane, PULL)
    assert out.status == "dropped-failed", out
    assert _works(lane, FAKE_CODE) and not _works(lane, WEEKLY)
    assert _pending_id(lane) == ""
    assert "backup email failed" in kw.caption(lane.state_path)


def test_row4_no_receipt_keeps_both_until_wednesday_then_drops(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    # Oracle never composed (or the receipt never came back)
    for when in (PULL, MON, TUE):
        out = kw.pull(lane, when)
        assert out.status == "waiting", (when, out)
        assert _works(lane, FAKE_CODE) and _works(lane, WEEKLY)
        assert oracle.calls[-1][0] == "receipt"
    out = kw.pull(lane, WED)
    assert out.status == "dropped-no-receipt", out
    assert _works(lane, FAKE_CODE) and not _works(lane, WEEKLY)
    assert "by Wednesday" in kw.caption(lane.state_path)
    assert "previous code stands" in kw.caption(lane.state_path)


def test_row4b_a_receipt_fetched_late_promotes_late(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    oracle.fail_receipt = True
    assert kw.pull(lane, PULL).status == "waiting"
    assert _works(lane, FAKE_CODE) and _works(lane, WEEKLY)
    oracle.fail_receipt = False
    out = kw.pull(lane, TUE)
    assert out.status == "promoted"
    assert _works(lane, WEEKLY) and not _works(lane, FAKE_CODE)
    assert "Tue" in kw.caption(lane.state_path)


def test_row4c_ssh_down_at_the_deadline_still_drops(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    oracle.fail_receipt = True
    for when in (PULL, MON, TUE):
        assert kw.pull(lane, when).status == "waiting"
    out = kw.pull(lane, WED)
    assert out.status == "dropped-no-receipt"
    assert _works(lane, FAKE_CODE) and not _works(lane, WEEKLY)


def test_row5_a_push_outside_the_window_is_missed_and_touches_nothing(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    out = kw.push(lane, MON)                 # Persistent=true fired at boot
    assert not out.ok and out.status == "missed"
    assert oracle.calls == [] and _pending_id(lane) == ""
    assert "missed this week" in kw.caption(lane.state_path)
    assert "current code stands" in kw.caption(lane.state_path)


def test_row6_a_second_push_is_refused_and_force_replaces(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    first = kw.push(lane, SUN)
    first_id = _pending_id(lane)
    again = kw.push(lane, SUN + timedelta(minutes=5))
    assert not again.ok and again.status == "refused"
    assert "already pending" in again.line and first_id in again.line
    assert len([v for v, _ in oracle.calls if v == "put"]) == 1
    assert _state(lane)["status"] == "pushed", "the refusal changed no state"

    forced = kw.push(lane, SUN + timedelta(minutes=30), force=True)
    assert forced.ok and forced.status == "pushed"
    new_id = _pending_id(lane)
    assert new_id != first_id and oracle.spool["id"] == new_id
    assert _state(lane)["replaced_id"] == first_id
    oracle.compose(COMPOSE)
    assert kw.pull(lane, PULL).status == "promoted"
    assert _works(lane, WEEKLY)
    assert first.status == "pushed"


def test_row6b_force_after_oracle_composed_names_the_dead_emailed_code(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    first_id = _pending_id(lane)
    oracle.compose(COMPOSE)                  # the email carries first_id's code
    kw.push(lane, COMPOSE + timedelta(minutes=15), force=True)
    new_id = _pending_id(lane)
    out = kw.pull(lane, PULL)
    assert out.status == "replaced", out
    assert _pending_id(lane) == new_id, "the forced push still waits on its own receipt"
    assert "replaced by a forced push" in kw.caption(lane.state_path)
    assert "will not work" in kw.caption(lane.state_path)
    assert _works(lane, FAKE_CODE)
    # ...and its receipt never comes, so Wednesday drops it
    assert kw.pull(lane, WED).status == "dropped-no-receipt"
    assert first_id != new_id and _works(lane, FAKE_CODE)


def test_row7_a_manual_rotate_keeps_the_pending_and_then_BEATS_it(tmp_path):
    from tests.test_notes_mail import GMAIL_CFG, FakeCfg
    from tests.test_send_file import FakeSMTP
    import jarvis.app as app_mod
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    # the drawer's rotate, between the push and the receipt
    a = SimpleNamespace(assistant=FakeCfg(GMAIL_CFG))
    a.get_option = lambda k, d=None: {"owner.mode": "shadow"}.get(k, d)
    a.gate = gt.OwnerGate(registry=Registry.load(lane.registry_path),
                          owner="hunter", get_option=a.get_option)
    for _n in ("_knightfall_rotate", "_knightfall_store"):
        setattr(a, _n, getattr(app_mod.JarvisApp, _n).__get__(a))
    FakeSMTP.made = []
    line, mailed = a._knightfall_rotate("hunter", smtp=FakeSMTP)
    assert mailed, line
    manual = FakeSMTP.made[-1].sent[-1].get_content().splitlines()[0].strip()
    FakeSMTP.made = []
    assert _works(lane, manual) and _works(lane, WEEKLY)
    assert not _works(lane, FAKE_CODE)
    assert _pending_id(lane), "the rotate did not touch the pending"

    # 2026-09-06, defect (C). This used to promote, which retired the code
    # he had been handed MOST RECENTLY and left him holding a dead one.
    # The rotate wins; Sunday's code is dropped and the caption says why.
    oracle.compose(COMPOSE)
    out = kw.pull(lane, PULL)
    assert out.status == "dropped-rotated", out
    assert _works(lane, manual) and not _works(lane, WEEKLY)
    assert _pending_id(lane) == ""
    assert "not applied" in kw.caption(lane.state_path)
    assert "rotated your own" in kw.caption(lane.state_path)


def test_row8_a_stale_spool_is_discarded_unread(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE + timedelta(days=2))     # a compose two days late
    out = kw.pull(lane, PULL + timedelta(days=2))
    assert out.status == "dropped-stale", out
    assert _works(lane, FAKE_CODE) and not _works(lane, WEEKLY)
    assert "too late" in kw.caption(lane.state_path)


def test_row9_an_error_on_oracles_side_drops_the_pending(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.spool = None
    oracle.receipt = {"id": _pending_id(lane), "status": "error",
                      "at": COMPOSE.isoformat()}
    out = kw.pull(lane, PULL)
    assert out.status == "dropped-error"
    assert _works(lane, FAKE_CODE) and not _works(lane, WEEKLY)
    assert "(error)" in kw.caption(lane.state_path)


def test_row11_a_receipt_for_an_unknown_id_is_ignored(tmp_path, caplog):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    mine = _pending_id(lane)
    oracle.receipt = {"id": "deadbeef", "status": "sent", "at": COMPOSE.isoformat()}
    with caplog.at_level(logging.INFO):
        out = kw.pull(lane, PULL)
    assert out.status == "waiting"
    assert _pending_id(lane) == mine
    assert _works(lane, FAKE_CODE) and _works(lane, WEEKLY)
    assert any("matched nothing" in r.getMessage() for r in caplog.records)


def test_row12_an_unreadable_registry_at_pull_changes_nothing(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    lane.registry_path.write_text("{ not json")
    calls = len(oracle.calls)
    out = kw.pull(lane, PULL)
    assert out.status == "unreadable"
    assert lane.registry_path.read_text() == "{ not json"
    assert len(oracle.calls) == calls, "no ssh for a registry it cannot write"
    assert oracle.receipt is not None, "the receipt waits for a readable file"
    assert "fix it at the keyboard" in kw.caption(lane.state_path)


def test_a_push_refuses_an_unusable_registry_before_any_ssh(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    lane.registry_path.write_text("{ not json")
    out = kw.push(lane, SUN)
    assert not out.ok and out.status == "unreadable"
    assert oracle.calls == []


def test_typed_before_the_pull_is_the_receipt_in_person(tmp_path, caplog):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    code_id = _pending_id(lane)
    oracle.compose(COMPOSE)
    # what the app does when the pending hash admits him at the keyboard
    ok, why = identity.locked_update(
        lane.registry_path, lambda r: r.promote_pending("hunter", code_id))
    assert ok, why
    kw.note_promoted_by_use(lane.state_path, code_id, COMPOSE + timedelta(minutes=45))
    assert "when you typed it" in kw.caption(lane.state_path)
    assert _works(lane, WEEKLY) and not _works(lane, FAKE_CODE)
    with caplog.at_level(logging.INFO):
        out = kw.pull(lane, PULL)
    assert out.status == "promoted-by-use"
    assert [v for v, _ in oracle.calls][-2:] == \
        ["receipt", "receipt --ack %s" % code_id], "the receipt is still drained"
    assert oracle.receipt is None
    assert any("already" in r.getMessage() for r in caplog.records)
    assert "when you typed it" in kw.caption(lane.state_path)
    calls = len(oracle.calls)
    assert kw.pull(lane, MON).status == "idle" and len(oracle.calls) == calls


def test_a_pending_without_state_is_still_pulled(tmp_path):
    """The state file narrates; the registry is the truth. A pending that
    the state has forgotten (a wiped state dir) is still resolved."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    lane.state_path.unlink()
    out = kw.pull(lane, PULL)
    assert out.status == "promoted"
    assert _works(lane, WEEKLY) and not _works(lane, FAKE_CODE)


# ------------------------------------------- the two properties that matter
def test_the_plaintext_never_lands_on_the_sparks_disk_or_in_any_log(
        tmp_path, caplog, capsys):
    """The code exists in exactly two places: the Oracle spool (here, the
    fake's memory) and his inbox. Every file the lane can write, every log
    record of every logger at DEBUG, and everything printed: none of them
    carry it. The hash may appear (it is what is stored); the code may not."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    with caplog.at_level(logging.DEBUG):
        kw.push(lane, SUN)
        kw.push(lane, SUN + timedelta(minutes=5))          # the refusal line
        oracle.compose(COMPOSE)
        kw.pull(lane, PULL)
        kw.pull(lane, MON)
        kw.main(["--rehearse"], lane=lane, now=SUN)
        kw.main(["--probe"], lane=lane, now=SUN)
        kw.main(["--status"], lane=lane, now=MON)
    assert json.loads(oracle.calls[0][1])["code"] == WEEKLY, "it did go to Oracle"
    for path in sorted(p for p in tmp_path.rglob("*") if p.is_file()):
        text = path.read_bytes().decode("utf-8", "replace")
        assert WEEKLY not in text, "the code is on disk in %s" % path
    for r in caplog.records:
        blob = " ".join((str(r.msg), repr(r.args), r.getMessage()))
        assert WEEKLY not in blob, (r.name, r.levelname, str(r.msg))
    printed = capsys.readouterr()
    assert WEEKLY not in printed.out and WEEKLY not in printed.err
    assert "8 characters" in printed.out and "scrypt$" in printed.out


def test_at_no_point_in_any_sequence_is_there_zero_working_codes(
        tmp_path, monkeypatch):
    """A property over random sequences of everything that can happen to
    the registry -- pushes (some refused, some forced, some with Oracle
    down or the store failing), Oracle composing well, badly or not at
    all, pulls on every day, manual rotates from the drawer, the emailed
    code typed early, and the clock -- with the set of codes that SHOULD
    work tracked beside it. After every step at least one of them opens
    the door on the file as it is on disk. scrypt is turned down for this
    one so 40 sequences fit in seconds; the invariant does not depend on
    the cost of a hash."""
    from tests.test_notes_mail import GMAIL_CFG, FakeCfg
    from tests.test_send_file import FakeSMTP
    import jarvis.app as app_mod
    monkeypatch.setattr(pp, "SCRYPT_N", 2 ** 8)
    rng = random.Random(20260906)
    counter = [0]

    def fresh_code():
        counter[0] += 1
        return "x%07d" % counter[0]

    for seq in range(40):
        base = tmp_path / ("s%02d" % seq)
        base.mkdir()
        oracle = FakeOracle()
        lane = _lane(base, oracle, code=fresh_code)
        should = {FAKE_CODE}
        now = SAT
        a = SimpleNamespace(assistant=FakeCfg(GMAIL_CFG))
        a.get_option = lambda k, d=None: {"owner.mode": "shadow"}.get(k, d)
        a.gate = gt.OwnerGate(registry=Registry.load(lane.registry_path),
                              owner="hunter", get_option=a.get_option)
        for _n in ("_knightfall_rotate", "_knightfall_store"):
            setattr(a, _n, getattr(app_mod.JarvisApp, _n).__get__(a))
        trail = []
        for step in range(10):
            op = rng.choice(["push", "push-force", "compose", "compose-bad",
                             "pull", "rotate", "type-pending", "oracle-down",
                             "store-fails", "tick", "tick"])
            trail.append((now.isoformat(), op))
            before = Registry.load(lane.registry_path).person("hunter")
            if op in ("push", "push-force"):
                oracle.down = False
                out = kw.push(lane, now, force=(op == "push-force"))
                if out.status == "pushed":
                    should.add(oracle.spool["code"])
                    # A push only reaches "pushed" over an existing pending
                    # when it REPLACED it: --force, or (2026-09-06) a
                    # scheduled push over one Oracle can no longer send.
                    if before.pending_code_hash:
                        should.discard(_plain_of(should, before.pending_code_hash))
            elif op == "oracle-down":
                oracle.down = True
                kw.push(lane, now)
                oracle.down = False
            elif op == "store-fails":
                with monkeypatch.context() as m:
                    m.setattr(kw.identity, "locked_update",
                              lambda *a_, **k: (False, "no"))
                    kw.push(lane, now)
            elif op in ("compose", "compose-bad"):
                oracle.compose(now, send_ok=(op == "compose"))
            elif op == "pull":
                out = kw.pull(lane, now)
                after = Registry.load(lane.registry_path).person("hunter")
                if out.status == "promoted":
                    should.discard(_plain_of(should, before.code_hash))
                elif out.status.startswith("dropped"):
                    should.discard(_plain_of(should, before.pending_code_hash))
                assert after.code_hash, "a current hash always exists"
            elif op == "rotate":
                FakeSMTP.made = []
                line, mailed = a._knightfall_rotate("hunter", smtp=FakeSMTP)
                if mailed and "in your inbox" in line and "will not work" not in line:
                    new = FakeSMTP.made[-1].sent[-1].get_content().splitlines()[0].strip()
                    should.discard(_plain_of(should, before.code_hash))
                    should.add(new)
                FakeSMTP.made = []
            elif op == "type-pending":
                if before.pending_code_id:
                    ok, _ = identity.locked_update(
                        lane.registry_path,
                        lambda r, i=before.pending_code_id: r.promote_pending("hunter", i))
                    if ok:
                        should.discard(_plain_of(should, before.code_hash))
                        kw.note_promoted_by_use(lane.state_path, before.pending_code_id, now)
            elif op == "tick":
                now += timedelta(hours=rng.choice([1, 6, 24, 30]))
            reg = Registry.load(lane.registry_path)
            assert reg.usable, trail
            working = [c for c in should if gt.check_override_code(reg, c)[0] == "hunter"]
            assert working, ("ZERO working codes after %r; expected one of %d"
                             % (trail, len(should)))
            # and nothing outside the tracked set works either
            assert not gt.check_override_code(reg, "nope00")[0]
    # THE PROPERTY MUST NOT BE VACUOUS. It once was: _lane handed the
    # generator function itself to push, which str()'d it, so every
    # sequence used ONE code and "at least one works" was trivially about
    # that one. Eight characters is what passphrase.new_code makes, and
    # a distinct one per push is the whole point of the sequence.
    assert counter[0] > 40, "the sequences barely pushed; the property is thin"
    assert len({fresh_code() for _ in range(20)}) == 20, "codes must be distinct"


def _plain_of(candidates, stored_hash):
    """Which tracked plaintext a stored hash is (the property test's
    bookkeeping; the lane itself never does this)."""
    for c in candidates:
        if pp.check_secret(c, stored_hash):
            return c
    return None


# -------------------------------------------------------------- the lines
def test_every_caption_fits_the_drawer_strip():
    """894 px at his geometry; ~7.5 px a character. One short sentence."""
    for name in dir(kw):
        if name.startswith("LINE_"):
            text = getattr(kw, name).format(t="Sun 04:30 CDT", id="ab12cd34")
            assert len(text) <= 118, (name, len(text), text)
            assert text.startswith("Weekly: ")


def test_the_caption_of_no_state_is_empty(tmp_path):
    assert kw.caption(tmp_path / "nope.json") == ""
    (tmp_path / "bad.json").write_text("{ not json")
    assert kw.caption(tmp_path / "bad.json") == ""


def test_the_state_file_is_private_and_atomic(tmp_path):
    lane = _lane(tmp_path)
    kw.push(lane, SUN)
    assert lane.state_path.stat().st_mode & 0o777 == 0o600
    assert lane.state_path.parent.stat().st_mode & 0o777 == 0o700
    assert not list(lane.state_path.parent.glob("*.tmp"))


# ------------------------------------------------------------- the runner
def test_the_ssh_argv_is_batch_only_and_carries_the_key_and_stdin(monkeypatch):
    proc = MagicMock()
    proc.communicate.return_value = ('{"ok": true}', "")
    proc.returncode = 0
    seen = {}

    def popen(argv, **kw_):
        seen["argv"], seen["kw"] = argv, kw_
        return proc
    monkeypatch.setattr(kw.subprocess, "Popen", popen)
    reply = kw.run_ssh("opc@1.2.3.4", "/k/key", "cd /x && py -m app.jarvis_override put",
                       '{"id": "a"}', 20.0)
    assert reply.ok and reply.out == '{"ok": true}'
    argv = seen["argv"]
    assert argv[0].endswith("ssh") and "-n" not in argv
    assert "BatchMode=yes" in argv and "PasswordAuthentication=no" in argv
    assert "IdentitiesOnly=yes" in argv and "/k/key" in argv
    assert argv[-2] == "opc@1.2.3.4"
    assert argv[-1].endswith("put")
    assert seen["kw"]["stdin"] == subprocess.PIPE
    proc.communicate.assert_called_once_with(input='{"id": "a"}', timeout=20.0)


def test_a_wedged_ssh_is_killed_without_being_waited_for(monkeypatch):
    proc = MagicMock()
    proc.communicate.side_effect = subprocess.TimeoutExpired("ssh", 20)
    monkeypatch.setattr(kw.subprocess, "Popen", lambda *a, **k: proc)
    reply = kw.run_ssh("opc@h", "/k", "cmd", None, 20.0)
    assert not reply.ok and reply.reason == "timeout"
    proc.kill.assert_called_once()
    proc.wait.assert_not_called()


def test_a_missing_ssh_binary_is_a_reply_not_a_raise(monkeypatch):
    def boom(*a, **k):
        raise OSError("no ssh")
    monkeypatch.setattr(kw.subprocess, "Popen", boom)
    reply = kw.run_ssh("opc@h", "/k", "cmd", None, 5.0)
    assert not reply.ok and reply.reason == "no-ssh"


def test_the_remote_command_is_the_verb_on_oracles_module():
    assert kw.remote_command("put").endswith("app.jarvis_override put")
    assert kw.remote_command("revoke ab12cd34").endswith("revoke ab12cd34")
    with pytest.raises(ValueError):
        kw.remote_command("revoke $(rm -rf /)")
    with pytest.raises(ValueError):
        kw.remote_command("bogus")


# --------------------------------------------------------------- the CLI
def test_rehearse_makes_a_code_hashes_it_and_pushes_nothing(tmp_path, capsys):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    rc = kw.main(["--rehearse"], lane=lane, now=SUN)
    assert rc == 0
    out = capsys.readouterr().out
    assert "8 characters" in out and "scrypt$" in out and "NOT pushed" in out
    assert WEEKLY not in out
    assert oracle.calls == [] and _pending_id(lane) == ""


def test_probe_pings_the_gate_and_writes_nothing(tmp_path, capsys):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    rc = kw.main(["--probe"], lane=lane, now=SUN)
    assert rc == 0
    assert oracle.calls == [("ping", None)]
    assert "answers" in capsys.readouterr().out
    assert _pending_id(lane) == "" and not lane.state_path.exists()


def test_the_cli_push_and_pull_are_the_module_functions(tmp_path, capsys):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    assert kw.main(["push"], lane=lane, now=SUN) == 0
    assert _pending_id(lane)
    assert kw.main(["push"], lane=lane, now=SUN) == 1       # refused
    oracle.compose(COMPOSE)
    assert kw.main(["pull"], lane=lane, now=PULL) == 0
    assert _works(lane, WEEKLY)
    assert kw.main(["--status"], lane=lane, now=MON) == 0
    assert "took effect" in capsys.readouterr().out


def test_the_cli_exit_codes_tell_the_journal_what_happened(tmp_path):
    oracle = FakeOracle()
    oracle.down = True
    lane = _lane(tmp_path, oracle)
    assert kw.main(["push"], lane=lane, now=SUN) == 2        # unreachable


# ------------------------------------------------------ the unit files
UNITS = REPO / "scripts" / "systemd"


def _unit(name) -> dict:
    text = (UNITS / name).read_text()
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out.setdefault(k.strip(), []).append(v.strip())
    return out


def test_the_push_timer_is_written_in_utc_and_persistent():
    t = _unit("jarvis-knightfall-push.timer")
    assert t["OnCalendar"] == ["Sun *-*-* 07:30:00 UTC"]
    assert t["Persistent"] == ["true"]
    assert t.get("RandomizedDelaySec", ["0"]) == ["0"]


def test_the_pull_timer_runs_sunday_to_wednesday_in_utc():
    t = _unit("jarvis-knightfall-pull.timer")
    assert t["OnCalendar"] == ["Sun,Mon,Tue,Wed *-*-* 09:30:00 UTC"]
    assert t["Persistent"] == ["true"]


@pytest.mark.parametrize("name,verb", [("jarvis-knightfall-push.service", "push"),
                                       ("jarvis-knightfall-pull.service", "pull")])
def test_the_services_run_the_module_once_from_the_venv(name, verb):
    s = _unit(name)
    assert s["Type"] == ["oneshot"]
    assert s["ExecStart"] == ["%%h/vss_env/bin/python -m jarvis.knightfall_weekly %s" % verb]
    assert s["WorkingDirectory"] == ["%h/Jarvis"]
    assert s["NoNewPrivileges"] == ["true"]
    assert "TimeoutStartSec" in s


def test_the_install_script_ships_but_is_never_run_here():
    script = REPO / "scripts" / "setup_knightfall_weekly.sh"
    text = script.read_text()
    assert script.stat().st_mode & 0o111
    assert "jarvis-knightfall-push.timer" in text
    assert "jarvis-knightfall-pull.timer" in text
    assert "systemctl --user daemon-reload" in text
    assert "attack" in text.lower() or "until" in text.lower()


def test_the_oracle_gate_script_ships_for_the_restricted_key():
    gate = REPO / "deploy" / "oracle" / "jarvis-override-gate.sh"
    text = gate.read_text()
    assert "SSH_ORIGINAL_COMMAND" in text
    for verb in ("put", "receipt", "revoke", "ping"):
        assert verb in text
    assert "restrict" in text


def test_the_unit_calendars_parse():
    """systemd-analyze is on this box; a calendar it refuses is a timer
    that never fires. Skipped where it is missing."""
    import shutil
    if not shutil.which("systemd-analyze"):
        pytest.skip("no systemd-analyze")
    for name in ("jarvis-knightfall-push.timer", "jarvis-knightfall-pull.timer"):
        spec = _unit(name)["OnCalendar"][0]
        res = subprocess.run(["systemd-analyze", "calendar", spec],
                             capture_output=True, text=True, timeout=10)
        assert res.returncode == 0, (name, res.stderr)
        assert "UTC" in res.stdout
