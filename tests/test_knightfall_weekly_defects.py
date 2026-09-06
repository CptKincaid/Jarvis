"""The six defects the 09-06 adversary pass left in the weekly lane.

None of them is a lockout -- every one leaves the code he already has
working -- and that is exactly why they are easy to ship and hard to
notice. Each is a way for the WEEK to be silently lost, or for the lane's
own narration to go missing.

  (A) ``--force`` before Friday 20:30 UTC writes a spool Oracle's 36 h
      freshness rule discards unread at the Sunday compose, and the pending
      it stored then blocks the scheduled push and is dropped at the pull.
      One exploratory --force turns the feature off for the week.
  (B) Oracle's ``receipt`` verb PRINTED AND DELETED in one step: a pull
      that fetched the receipt and then failed to write the registry had
      lost it for ever, and the Wednesday deadline dropped a code that had
      actually been sent. Now ``receipt`` reads and ``receipt --ack <id>``
      deletes, after the Spark has written down what it says.
  (C) ``_promote`` saw ``current_fp`` changed -- a manual rotate between
      the push and the receipt -- and promoted anyway, retiring the code
      he had been handed MOST RECENTLY. Now it drops and says so.
  (D) A push killed between the clean PUT and the store left no state
      line at all, so the caption could not narrate it.
  (E) ``scripts/jarvis_people.py`` checked a typed code with
      ``check_override_code`` and left a matched PENDING code pending.
  (F) The asymmetry his decision #3 settled -- the drawer BURNS a typed
      weekly code, the users tab and this CLI do not -- was nowhere
      written down.

THE INVARIANTS THE ADVERSARY PROVED ARE NOT NEGOTIABLE and every test here
re-asserts its own corner of them: never a zero-code state, no plaintext
anywhere but the Oracle spool and his inbox, and an empty spool means
today's email byte-for-byte. The codes are invented.
"""
from __future__ import annotations

import importlib.util
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import jarvis
from jarvis import gate as gt
from jarvis import identity
from jarvis import knightfall_weekly as kw
from jarvis import passphrase as pp
from jarvis.identity import Registry
from tests.test_knightfall_weekly import (COMPOSE, FakeOracle, MON, PULL, SAT,
                                          SUN, WED, WEEKLY, _lane, _pending_id,
                                          _state, _works)
from tests.test_owner_gate import FAKE_CODE

UTC = timezone.utc
REPO = Path(jarvis.__file__).parent.parent
# 36 h before Oracle composes on Sunday 13 Sept 08:30 UTC, to the second.
EDGE = datetime(2026, 9, 11, 20, 30, tzinfo=UTC)
PUSH_ID = "ab12cd34"


# =====================================================================
# (A) a --force that can only make a spool Oracle will throw away
# =====================================================================
def test_a_force_too_early_is_refused_before_anything_leaves_this_machine(
        tmp_path, caplog):
    """Wednesday. The next compose is four days off, Oracle's rule is 36 h,
    so this spool is dead on arrival -- and the pending it would store
    blocks Sunday's real push. It never happens: no ssh, no pending, no
    caption that has to explain a lost week."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    with caplog.at_level(logging.DEBUG):
        out = kw.push(lane, MON, force=True)
    assert not out.ok and out.status == "too-early", out
    assert oracle.calls == [], "nothing left this machine"
    assert oracle.spool is None
    assert _pending_id(lane) == ""
    assert _works(lane, FAKE_CODE), "his code is untouched"
    assert "36" in out.line or "too early" in out.line.lower()
    # and it says WHEN it would be allowed, so the refusal is actionable:
    # MON 14 Sept targets Sunday the 20th, whose floor is Friday the 18th
    assert "2026-09-18 20:30 UTC" in out.line, out.line
    for r in caplog.records:
        assert WEEKLY not in r.getMessage()


def test_the_refusal_names_the_hour_the_push_becomes_possible():
    """The boundary is Oracle's FRESH_S before its compose, to the second:
    Friday 20:30 UTC for a Sunday 08:30 UTC compose."""
    assert kw.compose_at(SUN.date()) == datetime(2026, 9, 13, 8, 30, tzinfo=UTC)
    assert kw.earliest_push(SUN.date()) == EDGE
    assert (kw.compose_at(SUN.date()) - EDGE).total_seconds() == kw.ORACLE_FRESH_S


def test_the_boundary_is_thirty_six_hours_and_not_a_second_more(tmp_path):
    late = FakeOracle()
    lane = _lane(tmp_path, late)
    out = kw.push(lane, EDGE, force=True)
    assert out.ok and out.status == "pushed", out
    assert late.spool["for"] == "2026-09-13"

    early = FakeOracle()
    lane2 = _lane(tmp_path / "b", early)
    out2 = kw.push(lane2, EDGE - timedelta(seconds=1), force=True)
    assert not out2.ok and out2.status == "too-early", out2
    assert early.calls == []


def test_the_scheduled_push_replaces_a_pending_oracle_can_only_discard(tmp_path):
    """The other half of (A). A pending left over from a week whose spool
    can no longer be emailed used to REFUSE Sunday's push -- so one stuck
    pending turned the feature off for every week after it. The scheduled
    push replaces it now, and never touches the current code."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    stale_since = (SUN - timedelta(days=7)).isoformat()
    ok, why = identity.locked_update(
        lane.registry_path,
        lambda r: r.set_pending_code("hunter", pp.hash_secret("old11111"),
                                     "ffffffff", stale_since))
    assert ok, why

    out = kw.push(lane, SUN)                    # the TIMER's push, no --force
    assert out.ok and out.status == "pushed", out
    assert oracle.spool["id"] == _pending_id(lane) != "ffffffff"
    assert _state(lane)["replaced_id"] == "ffffffff"
    assert _works(lane, FAKE_CODE), "the current code is never touched"
    assert _works(lane, WEEKLY) and not _works(lane, "old11111")


def test_the_scheduled_push_still_refuses_a_pending_that_can_still_be_sent(
        tmp_path):
    """Saturday evening's forced push, then the timer at 07:30 Sunday. That
    spool is 11 hours old and Oracle will email it, so the timer must NOT
    replace it -- the refusal is the correct answer and (A) must not widen
    into 'the timer always wins'."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    first = kw.push(lane, SAT)
    assert first.ok, first
    first_id = _pending_id(lane)
    again = kw.push(lane, SUN)
    assert not again.ok and again.status == "refused", again
    assert _pending_id(lane) == first_id
    assert len([v for v, _ in oracle.calls if v == "put"]) == 1


def test_a_refused_force_never_rewrites_the_caption_of_the_push_in_flight(
        tmp_path):
    """A refusal changed nothing, so it must not narrate as though it had.
    Sunday's push is in flight and both codes work; an exploratory --force
    on Tuesday is refused and the drawer goes on saying so."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    before = dict(_state(lane))
    out = kw.push(lane, PULL + timedelta(days=2), force=True)
    assert not out.ok and out.status == "too-early", out
    assert _state(lane) == before, "the refusal wrote no state"
    assert "both codes work" in kw.caption(lane.state_path)
    assert _pending_id(lane) == before["id"]
    assert len([v for v, _ in oracle.calls if v == "put"]) == 1


def test_a_pending_with_an_unreadable_stamp_is_replaced_not_obeyed(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    ok, why = identity.locked_update(
        lane.registry_path,
        lambda r: r.set_pending_code("hunter", pp.hash_secret("old11111"),
                                     "ffffffff", "not a date"))
    assert ok, why
    out = kw.push(lane, SUN)
    assert out.ok and out.status == "pushed", out
    assert _works(lane, FAKE_CODE)


def test_the_freshness_rule_is_one_number_on_both_sides():
    """The Spark refuses exactly what Oracle would discard. Two copies of
    36 h that can drift apart is how (A) would come back."""
    src = (REPO / "deploy" / "oracle" / "jarvis_override.py").read_text()
    assert "FRESH_S = 36 * 3600" in src
    assert kw.ORACLE_FRESH_S == 36 * 3600
    assert "jarvis_override" in kw.__doc__ or "Oracle" in kw.__doc__


# =====================================================================
# (B) the receipt is READ, and deleted only once it is written down here
# =====================================================================
@pytest.fixture()
def ov(tmp_path):
    """Oracle's module, spool redirected into tmp_path (nothing here
    touches Oracle)."""
    path = REPO / "deploy" / "oracle" / "jarvis_override.py"
    spec = importlib.util.spec_from_file_location("jarvis_override_defects",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SPOOL_DIR = tmp_path / "spool"
    mod.PENDING = mod.SPOOL_DIR / "pending.json"
    mod.RECEIPT = mod.SPOOL_DIR / "receipt.json"
    return mod


def _a_receipt(ov, code_id=PUSH_ID, status="sent"):
    ov._write_json(ov.RECEIPT, {"id": code_id, "status": status,
                               "at": COMPOSE.isoformat()})


def test_the_receipt_verb_reads_and_no_longer_deletes(ov, capsys):
    _a_receipt(ov)
    for _ in range(3):
        assert ov.main(["receipt"]) == 0
        assert json.loads(capsys.readouterr().out)["id"] == PUSH_ID
        assert ov.RECEIPT.exists(), "a read must not destroy the only copy"


def test_only_an_ack_for_that_id_deletes_the_receipt(ov, capsys):
    _a_receipt(ov)
    assert ov.main(["receipt", "--ack", "ffffffff"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "acked": False}
    assert ov.RECEIPT.exists(), "another push's id must not delete this one"

    assert ov.main(["receipt", "--ack", PUSH_ID]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "acked": True}
    assert not ov.RECEIPT.exists()

    assert ov.main(["receipt", "--ack", PUSH_ID]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "acked": False}


def test_a_read_and_an_ack_are_safe_with_no_receipt_at_all(ov, capsys):
    assert ov.main(["receipt"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "none"}
    assert ov.main(["receipt", "--ack", PUSH_ID]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "acked": False}
    assert ov.main(["--check"]) == 0
    assert capsys.readouterr().out.strip() == \
        "no spool: the email would be exactly as today"


def test_a_malformed_receipt_is_readable_as_none_and_ackable(ov, capsys):
    ov._ensure_dir()
    ov.RECEIPT.write_text("{ not json")
    assert ov.main(["receipt"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "none"}
    assert ov.main(["receipt", "--ack", PUSH_ID]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "acked": True}
    assert not ov.RECEIPT.exists(), "junk must not wedge the lane for ever"


@pytest.mark.parametrize("argv", [
    ["receipt", "--ack"],
    ["receipt", "--ack", "nope"],
    ["receipt", "--ack", "ab12cd34", "extra"],
    ["receipt", "--nope", "ab12cd34"],
    ["receipt", "ab12cd34"],
])
def test_a_receipt_verb_this_gate_does_not_know_is_refused(ov, capsys, argv):
    _a_receipt(ov)
    assert ov.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert ov.RECEIPT.exists(), "and it deleted nothing on the way out"


def test_the_ack_verb_is_one_the_spark_may_send_and_junk_is_not():
    cmd = kw.remote_command("receipt --ack ab12cd34")
    assert cmd.endswith("-m app.jarvis_override receipt --ack ab12cd34")
    for bad in ("receipt --ack", "receipt --ack nope", "receipt --ack ab12cd34 x",
                "receipt --ack ab12cd34; rm -rf /", "receipt ab12cd34"):
        with pytest.raises(ValueError):
            kw.remote_command(bad)


def test_the_restricted_key_allows_the_ack_and_nothing_wider():
    text = (REPO / "deploy" / "oracle" / "jarvis-override-gate.sh").read_text()
    assert "--ack" in text
    # the id is matched character by character, exactly as revoke's is
    assert text.count("[0-9a-f][0-9a-f][0-9a-f][0-9a-f]"
                      "[0-9a-f][0-9a-f][0-9a-f][0-9a-f]") == 2


def test_the_pull_acks_only_after_the_registry_is_written(tmp_path, monkeypatch):
    """THE DEFECT ITSELF. A pull that read the receipt and then could not
    write people.json used to leave Oracle with nothing, so Wednesday
    dropped a code that HAD gone out. Now the receipt survives the failure
    and the next pull promotes on it."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    assert oracle.receipt is not None

    with monkeypatch.context() as m:
        m.setattr(kw.identity, "locked_update",
                  lambda *a, **k: (False, "the people file could not be written"))
        out = kw.pull(lane, PULL)
    assert not out.ok and out.status == "unreadable", out
    assert oracle.receipt is not None, "the receipt was not thrown away"
    assert not [v for v, _ in oracle.calls if v.startswith("receipt --ack")]
    assert _works(lane, FAKE_CODE) and _works(lane, WEEKLY), "both still open"

    out = kw.pull(lane, MON)
    assert out.ok and out.status == "promoted", out
    assert oracle.receipt is None, "acked once it was safely written down"
    assert _works(lane, WEEKLY) and not _works(lane, FAKE_CODE)


def test_a_failed_ack_is_remembered_and_retried_by_the_next_pull(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    oracle.fail_ack = True
    out = kw.pull(lane, PULL)
    assert out.status == "promoted", out
    assert oracle.receipt is not None, "Oracle still holds it"
    assert _state(lane)["unacked"] == _state(lane)["id"], \
        "the lane knows exactly which receipt it owes an ack for"

    oracle.fail_ack = False
    out = kw.pull(lane, MON)                    # otherwise an idle pull
    assert out.ok, out
    assert oracle.receipt is None
    assert not _state(lane)["unacked"]


def test_a_receipt_for_nothing_this_lane_knows_is_acked_away(tmp_path, caplog):
    """A receipt nobody can apply must not sit on Oracle for ever: next
    week's pull would read it instead of its own."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.receipt = {"id": "ffffffff", "status": "sent",
                      "at": COMPOSE.isoformat()}
    with caplog.at_level(logging.WARNING):
        out = kw.pull(lane, PULL)
    assert out.status == "waiting", out
    assert oracle.receipt is None, "the stranger was acked away"
    assert _pending_id(lane), "and this week's pending still waits"


# =====================================================================
# (C) a rotate between the push and the receipt WINS
# =====================================================================
def _rotate(lane, plain):
    """What the drawer's rotate does to the file, without the mail: the
    current hash becomes a new one, the pending is untouched."""
    ok, why = identity.locked_update(
        lane.registry_path,
        lambda r: r.set_secret("hunter", "code_hash", pp.hash_secret(plain)))
    assert ok, why


def test_a_rotate_after_the_push_beats_the_weekly_code(tmp_path):
    """He rotated at the drawer on Monday; the code in his hand is the one
    he was handed MOST RECENTLY. Promoting Sunday's email over it retired
    the newer one -- so the pending is dropped and the caption says why."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    _rotate(lane, "manual22")
    assert _works(lane, "manual22") and _works(lane, WEEKLY)

    out = kw.pull(lane, PULL)
    assert out.ok and out.status == "dropped-rotated", out
    assert _works(lane, "manual22"), "the code he holds still opens the door"
    assert not _works(lane, WEEKLY), "and the email's code is not applied"
    assert _pending_id(lane) == ""
    cap = kw.caption(lane.state_path)
    assert "not applied" in cap and "rotated" in cap
    assert oracle.receipt is None, "and the receipt was still acked"


def test_without_a_rotate_the_weekly_code_still_takes_effect(tmp_path):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    out = kw.pull(lane, PULL)
    assert out.status == "promoted", out
    assert _works(lane, WEEKLY) and not _works(lane, FAKE_CODE)
    assert "took effect" in kw.caption(lane.state_path)


def test_a_lost_state_file_does_not_invent_a_rotate(tmp_path):
    """``current_fp`` is the only witness; with no state at all the pull
    must not refuse to apply the week's code."""
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    kw.push(lane, SUN)
    oracle.compose(COMPOSE)
    lane.state_path.unlink()
    out = kw.pull(lane, PULL)
    assert out.status == "promoted", out
    assert _works(lane, WEEKLY)


# =====================================================================
# (D) the gap between a clean PUT and the store
# =====================================================================
def test_a_push_killed_between_the_put_and_the_store_leaves_a_caption(
        tmp_path, monkeypatch):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)

    def killed(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(kw.identity, "locked_update", killed)
    with pytest.raises(KeyboardInterrupt):
        kw.push(lane, SUN)

    st = _state(lane)
    assert st["status"] == "in-flight", st
    assert st["id"] == oracle.spool["id"]
    cap = kw.caption(lane.state_path)
    assert "in flight" in cap and "unstored" in cap
    assert st["id"] in cap, "the id names the push he can ask Oracle about"
    assert _works(lane, FAKE_CODE), "and his code still opens the door"
    assert WEEKLY not in lane.state_path.read_text()


def test_the_in_flight_line_is_replaced_by_the_final_one(tmp_path):
    lane = _lane(tmp_path)
    kw.push(lane, SUN)
    st = _state(lane)
    assert st["status"] == "pushed"
    assert "in flight" not in kw.caption(lane.state_path)


def test_a_failed_store_replaces_the_in_flight_line_too(tmp_path, monkeypatch):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    monkeypatch.setattr(kw.identity, "locked_update",
                        lambda *a, **k: (False, "no"))
    out = kw.push(lane, SUN)
    assert out.status == "unstored", out
    assert _state(lane)["status"] == "unstored"
    assert "in flight" not in kw.caption(lane.state_path)


# =====================================================================
# (E) the people CLI leaves a typed weekly code pending
# =====================================================================
def _cli_with_pending(monkeypatch, tmp_path, *, typed=WEEKLY):
    """scripts/jarvis_people.py, a tmp registry holding a pending code, and
    the typed answer to its prompt. NOTHING here touches his real
    ~/.local/state/jarvis."""
    from tests.test_signin import _cli
    from tests.test_owner_gate import _registry

    mod, said = _cli(monkeypatch, tmp_path)
    monkeypatch.setattr(mod.PATHS, "KNIGHTFALL_WEEKLY",
                        tmp_path / "knightfall-weekly.json", raising=False)
    reg = _registry(tmp_path, code=True)
    ok, why = identity.locked_update(
        tmp_path / "people.json",
        lambda r: r.set_pending_code("hunter", pp.hash_secret(WEEKLY),
                                     PUSH_ID, SUN.isoformat()))
    assert ok, why
    reg = Registry.load(tmp_path / "people.json")
    monkeypatch.setattr(mod, "_isatty", lambda _s: True)
    monkeypatch.setattr(mod.getpass, "getpass", lambda *_a, **_k: typed)
    return mod, said, reg


def test_the_people_cli_promotes_a_weekly_code_he_typed(monkeypatch, tmp_path):
    mod, said, reg = _cli_with_pending(monkeypatch, tmp_path)
    ok, who = mod._authorise(reg)
    assert ok and who == "hunter", who

    disk = Registry.load(tmp_path / "people.json")
    assert disk.person("hunter").pending_code_hash == "", "it was promoted"
    assert disk.person("hunter").pending_code_id == ""
    assert gt.check_override_code(disk, WEEKLY)[0] == "hunter"
    assert gt.check_override_code(disk, FAKE_CODE)[0] == "", "the old one is gone"
    state = kw.read_state(tmp_path / "knightfall-weekly.json")
    assert state["status"] == "promoted-by-use" and state["id"] == PUSH_ID
    assert WEEKLY not in json.dumps(state)


def test_the_cli_can_still_save_after_it_promoted(monkeypatch, tmp_path):
    """The registry the caller goes on to change is the one this function
    was handed. Promoting behind its back would make every `add`,
    `set-role` and `forget` that asked for a code refuse with 'the people
    book changed since this command started'."""
    mod, said, reg = _cli_with_pending(monkeypatch, tmp_path)
    ok, _who = mod._authorise(reg)
    assert ok
    assert reg.person("hunter").pending_code_hash == "", "in memory too"
    reg.set_honorific("hunter", "sir")
    ok, why = reg.save_checked()
    assert ok, why


def test_the_current_code_typed_at_the_cli_changes_nothing(monkeypatch, tmp_path):
    mod, said, reg = _cli_with_pending(monkeypatch, tmp_path, typed=FAKE_CODE)
    ok, who = mod._authorise(reg)
    assert ok and who == "hunter"
    disk = Registry.load(tmp_path / "people.json")
    assert disk.person("hunter").pending_code_id == PUSH_ID, "still pending"
    assert kw.read_state(tmp_path / "knightfall-weekly.json") == {}


def test_a_wrong_code_at_the_cli_promotes_nothing(monkeypatch, tmp_path):
    mod, said, reg = _cli_with_pending(monkeypatch, tmp_path, typed="nope99")
    ok, why = mod._authorise(reg)
    assert not ok
    disk = Registry.load(tmp_path / "people.json")
    assert disk.person("hunter").pending_code_id == PUSH_ID
    assert kw.read_state(tmp_path / "knightfall-weekly.json") == {}


def test_the_cli_does_not_burn_the_code_it_promoted(monkeypatch, tmp_path):
    """HIS DECISION #3, the asymmetry (F): the DRAWER rotates and mails the
    next code when a typed one is accepted; the users tab and this terminal
    tool do not. An administrative unlock mails nothing."""
    mod, said, reg = _cli_with_pending(monkeypatch, tmp_path)
    ok, _who = mod._authorise(reg)
    assert ok
    disk = Registry.load(tmp_path / "people.json")
    assert pp.check_secret(WEEKLY, disk.person("hunter").code_hash), \
        "the code he typed is the code that still works: nothing was rotated"
    src = (REPO / "scripts" / "jarvis_people.py").read_text()
    assert "_knightfall_rotate" not in src and "send_notice" not in src


# =====================================================================
# (F) the asymmetry is written down
# =====================================================================
def test_the_module_docstring_says_who_burns_a_typed_weekly_code():
    doc = kw.__doc__
    low = doc.lower()
    assert "drawer" in low and "users tab" in low
    assert "rotate" in low or "burn" in low


def test_the_document_says_it_too():
    text = (REPO / "docs" / "assistant-setup.md").read_text()
    section = text[text.index("## 84"):]
    section = section[:section.index("## 85")]
    low = section.lower()
    assert "users tab" in low
    for phrase in ("--ack", "receipt"):
        assert phrase in low, phrase
    assert "friday" in low, "the --force boundary he can act on"


# =====================================================================
# the invariants, once more, across the changed paths
# =====================================================================
def test_no_new_path_can_leave_zero_working_codes(tmp_path, monkeypatch):
    """Every branch this change added, walked once, with the set of codes
    that should work tracked beside it."""
    monkeypatch.setattr(pp, "SCRYPT_N", 2 ** 8)
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)

    def open_codes():
        reg = Registry.load(lane.registry_path)
        return [c for c in (FAKE_CODE, WEEKLY, "manual22", "old11111")
                if gt.check_override_code(reg, c)[0] == "hunter"]

    assert open_codes() == [FAKE_CODE]
    assert kw.push(lane, MON, force=True).status == "too-early"
    assert open_codes() == [FAKE_CODE]
    assert kw.push(lane, SUN).status == "pushed"
    assert open_codes() == [FAKE_CODE, WEEKLY]
    oracle.compose(COMPOSE)
    _rotate(lane, "manual22")
    assert open_codes() == [WEEKLY, "manual22"]
    assert kw.pull(lane, PULL).status == "dropped-rotated"
    assert open_codes() == ["manual22"]
    assert kw.pull(lane, WED).status == "idle"
    assert open_codes() == ["manual22"]


def test_the_plaintext_still_never_reaches_the_disk_or_a_log(
        tmp_path, caplog, capsys):
    oracle = FakeOracle()
    lane = _lane(tmp_path, oracle)
    with caplog.at_level(logging.DEBUG):
        kw.push(lane, MON, force=True)              # too-early
        kw.push(lane, SUN)
        oracle.compose(COMPOSE)
        _rotate(lane, "manual22")
        kw.pull(lane, PULL)                          # dropped-rotated + ack
        kw.pull(lane, MON)
    for path in sorted(p for p in tmp_path.rglob("*") if p.is_file()):
        text = path.read_bytes().decode("utf-8", "replace")
        assert WEEKLY not in text, path
    for r in caplog.records:
        assert WEEKLY not in " ".join((str(r.msg), repr(r.args), r.getMessage()))
    printed = capsys.readouterr()
    assert WEEKLY not in printed.out and WEEKLY not in printed.err


def test_an_empty_spool_is_still_todays_email_byte_for_byte(ov):
    """Unchanged by (B): claim() with no spool returns no line and settles
    nothing, so run_backup composes exactly what it always has."""
    base = "Encrypted size: 4096 bytes."
    line, token = ov.claim(COMPOSE)
    ov.settle(token, True, COMPOSE)
    assert line == "" and token is None
    assert base + line == base
    assert not ov.RECEIPT.exists()
    assert ov.main(["--dry-run"]) == 0
