"""A PENDING override code beside the current one (Knightfall weekly,
2026-09-06: "I want the nightfall code to send weekly in the same email as
the encrypted back up").

The weekly lane hands a fresh code to Oracle BEFORE it knows whether the
email carrying it will go out, so for a while there are two codes that
must both open the door: the one he has, and the one in Sunday's email.
That is the whole reason a second hash exists. Three named transitions on
the registry -- set_pending_code / promote_pending / drop_pending -- and
the break-glass check tries both hashes. Nothing here sees a plaintext but
the test's own invented codes.

The cross-process lock: the issuer is a THIRD writer of people.json (after
the app and scripts/jarvis_people.py) and the code says in its own words
that nothing locked against the terminal tool. ``identity.locked_update``
takes an flock on <people.json>.lock around load -> change -> save.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from jarvis import gate as gt
from jarvis import identity
from jarvis import passphrase as pp
from jarvis.identity import Person, Registry
from tests.test_owner_gate import FAKE_CODE, _registry

WEEKLY = "xxx222"          # the invented code "in Sunday's email"
PENDING_ID = "ab12cd34"


def _pending(tmp_path):
    reg = _registry(tmp_path, code=True)
    ok, why = reg.set_pending_code("hunter", pp.hash_secret(WEEKLY),
                                   PENDING_ID, "2026-09-13T07:30:00+00:00")
    assert ok, why
    assert reg.save()
    return reg


# ------------------------------------------------------------ the fields
def test_a_person_round_trips_the_pending_fields_through_the_file(tmp_path):
    _pending(tmp_path)
    disk = Registry.load(tmp_path / "people.json")
    p = disk.person("hunter")
    assert p.pending_code_hash.startswith("scrypt$")
    assert p.pending_code_id == PENDING_ID
    assert p.pending_code_since == "2026-09-13T07:30:00+00:00"
    assert p.code_hash and p.code_hash != p.pending_code_hash


def test_redacted_says_yes_or_no_and_never_carries_a_hash(tmp_path):
    reg = _pending(tmp_path)
    red = reg.person("hunter").redacted()
    assert red["has_code"] is True and red["has_pending"] is True
    assert "scrypt" not in json.dumps(red)
    assert Person(label="x").redacted()["has_pending"] is False


def test_repr_carries_no_hash(tmp_path):
    reg = _pending(tmp_path)
    assert "scrypt" not in repr(reg.person("hunter"))


def test_a_row_without_the_keys_reads_as_no_pending(tmp_path):
    """A FORMAT-2 file written by an older build has no such keys."""
    path = tmp_path / "people.json"
    path.write_text(json.dumps({"format": 2, "people": [
        {"label": "hunter", "role": "owner", "code_hash": "scrypt$x"}]}))
    p = Registry.load(path).person("hunter")
    assert p.pending_code_hash == "" and p.pending_code_id == ""


# ------------------------------------------------------- the transitions
def test_set_pending_is_owner_only_and_needs_an_id(tmp_path):
    reg = _registry(tmp_path, code=True, known=True)
    ok, why = reg.set_pending_code("heather", "scrypt$x", "a1b2c3d4", "t")
    assert not ok and "owner" in why
    ok, why = reg.set_pending_code("nobody", "scrypt$x", "a1b2c3d4", "t")
    assert not ok
    ok, why = reg.set_pending_code("hunter", "scrypt$x", "", "t")
    assert not ok and "id" in why
    ok, why = reg.set_pending_code("hunter", "", "a1b2c3d4", "t")
    assert not ok


def test_promote_makes_the_pending_current_and_the_old_hash_is_gone(tmp_path):
    reg = _pending(tmp_path)
    old = reg.person("hunter").code_hash
    new = reg.person("hunter").pending_code_hash
    ok, why = reg.promote_pending("hunter", PENDING_ID)
    assert ok, why
    p = reg.person("hunter")
    assert p.code_hash == new
    assert p.pending_code_hash == "" and p.pending_code_id == "" \
        and p.pending_code_since == ""
    assert old not in json.dumps(p.to_json())


def test_promote_and_drop_refuse_a_wrong_id(tmp_path):
    """The receipt names an id; a receipt for a DIFFERENT push must not
    promote whatever happens to be pending now (a forced re-push)."""
    reg = _pending(tmp_path)
    ok, why = reg.promote_pending("hunter", "ffffffff")
    assert not ok and "id" in why
    ok, why = reg.drop_pending("hunter", "ffffffff")
    assert not ok
    assert reg.person("hunter").pending_code_id == PENDING_ID


def test_drop_leaves_the_current_code_untouched(tmp_path):
    reg = _pending(tmp_path)
    old = reg.person("hunter").code_hash
    ok, why = reg.drop_pending("hunter", PENDING_ID)
    assert ok, why
    p = reg.person("hunter")
    assert p.code_hash == old and p.pending_code_hash == ""


def test_promote_with_nothing_pending_says_so(tmp_path):
    reg = _registry(tmp_path, code=True)
    ok, why = reg.promote_pending("hunter", PENDING_ID)
    assert not ok and "pending" in why


def test_set_secret_still_allows_only_the_two_old_fields(tmp_path):
    reg = _registry(tmp_path)
    ok, why = reg.set_secret("hunter", "pending_code_hash", "scrypt$x")
    assert not ok


def test_a_manual_set_code_leaves_the_pending_alone(tmp_path):
    """A rotate from the drawer writes code_hash and never touches the
    pending: the code in the one email he actually reads must not die
    because he also pressed a button."""
    reg = _pending(tmp_path)
    ok, _ = reg.set_secret("hunter", "code_hash", pp.hash_secret("manual99"))
    assert ok and reg.save()
    p = Registry.load(tmp_path / "people.json").person("hunter")
    assert p.pending_code_id == PENDING_ID and p.pending_code_hash


# -------------------------------------------------------------- the gate
def test_both_codes_open_the_door_while_one_is_pending(tmp_path):
    reg = _pending(tmp_path)
    assert gt.check_override_code(reg, FAKE_CODE) == ("hunter", "")
    assert gt.check_override_code(reg, WEEKLY) == ("hunter", "")


def test_the_check_says_which_leg_admitted_him(tmp_path):
    reg = _pending(tmp_path)
    assert gt.check_override_code_leg(reg, FAKE_CODE) == ("hunter", "", "current")
    assert gt.check_override_code_leg(reg, WEEKLY) == ("hunter", "", "pending")
    who, why, leg = gt.check_override_code_leg(reg, "nope00")
    assert (who, leg) == ("", "") and why == "that is not a code I know"


def test_after_promote_only_the_weekly_code_works(tmp_path):
    reg = _pending(tmp_path)
    assert reg.promote_pending("hunter", PENDING_ID)[0]
    assert gt.check_override_code(reg, WEEKLY)[0] == "hunter"
    assert gt.check_override_code(reg, FAKE_CODE)[0] == ""


def test_after_drop_only_the_old_code_works(tmp_path):
    reg = _pending(tmp_path)
    assert reg.drop_pending("hunter", PENDING_ID)[0]
    assert gt.check_override_code(reg, FAKE_CODE)[0] == "hunter"
    assert gt.check_override_code(reg, WEEKLY)[0] == ""


def test_a_refusal_reads_the_same_with_or_without_a_pending(tmp_path):
    a = _registry(tmp_path / "a", code=True)
    b = _pending(tmp_path / "b")
    assert gt.check_override_code(a, "nope00") == gt.check_override_code(b, "nope00")


def test_a_wrong_code_costs_two_derivations_per_owner_only_with_a_pending(
        tmp_path, monkeypatch):
    calls = []
    real = pp.check_secret

    def counted(plain, stored):
        calls.append(stored[:12])
        return real(plain, stored)
    monkeypatch.setattr(gt.pp, "check_secret", counted)
    gt.check_override_code(_registry(tmp_path / "a", code=True), "nope00")
    assert len(calls) == 1
    calls.clear()
    gt.check_override_code(_pending(tmp_path / "b"), "nope00")
    assert len(calls) == 2


def test_the_pending_code_is_never_accepted_over_the_microphone(tmp_path):
    """The typed-only rule is the point of the feature, and a second hash
    must not open a second channel: judge() never calls the code check."""
    from tests.test_owner_gate import MATCHED, _gate
    reg = _pending(tmp_path)
    g = _gate(tmp_path, registry=reg, mode="enforce")
    d = g.judge("voice", WEEKLY, stats=MATCHED, rejected=True)
    assert d.admit is False


# -------------------------------------------------------------- the lock
def test_locked_update_loads_changes_and_saves(tmp_path):
    _registry(tmp_path, code=True)
    path = tmp_path / "people.json"
    ok, why = identity.locked_update(
        path, lambda r: r.set_pending_code("hunter", "scrypt$x", "a1b2c3d4", "t"))
    assert ok, why
    assert Registry.load(path).person("hunter").pending_code_id == "a1b2c3d4"
    assert not (tmp_path / "people.json.lock").exists() or \
        (tmp_path / "people.json.lock").stat().st_size == 0


def test_locked_update_refuses_a_broken_file(tmp_path):
    path = tmp_path / "people.json"
    path.write_text("{ not json")
    ok, why = identity.locked_update(
        path, lambda r: r.set_pending_code("hunter", "scrypt$x", "a1b2c3d4", "t"))
    assert not ok and "registry" in why.lower() or "read" in why.lower()
    assert path.read_text() == "{ not json", "nothing was written over it"


def test_locked_update_refuses_a_missing_file(tmp_path):
    ok, why = identity.locked_update(
        tmp_path / "people.json",
        lambda r: r.set_pending_code("hunter", "scrypt$x", "a1b2c3d4", "t"))
    assert not ok
    assert not (tmp_path / "people.json").exists()


def test_locked_update_hands_back_the_change_refusal(tmp_path):
    _registry(tmp_path, code=True)
    ok, why = identity.locked_update(
        tmp_path / "people.json",
        lambda r: r.promote_pending("hunter", "ffffffff"))
    assert not ok and "pending" in why


def test_a_second_holder_waits_and_then_gives_up_with_a_line(tmp_path):
    _registry(tmp_path, code=True)
    path = tmp_path / "people.json"
    held = threading.Event()
    release = threading.Event()

    def holder():
        with identity.registry_lock(path):
            held.set()
            release.wait(5.0)
    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert held.wait(2.0)
    t0 = time.monotonic()
    ok, why = identity.locked_update(
        path, lambda r: r.set_pending_code("hunter", "scrypt$x", "a1b2c3d4", "t"),
        timeout_s=0.3)
    assert not ok and "another writer" in why
    assert 0.25 <= time.monotonic() - t0 < 3.0
    release.set()
    t.join(2.0)
    assert Registry.load(path).person("hunter").pending_code_id == ""


def test_the_lock_is_re_entrant_on_one_thread(tmp_path):
    _registry(tmp_path, code=True)
    path = tmp_path / "people.json"
    with identity.registry_lock(path):
        ok, why = identity.locked_update(
            path, lambda r: r.set_pending_code("hunter", "scrypt$x", "a1b2c3d4", "t"),
            timeout_s=0.3)
    assert ok, why


def test_save_checked_refuses_to_write_over_a_file_that_changed(tmp_path):
    """The terminal tool loads, prompts, and saves minutes later. A save
    that finds the file changed under it refuses rather than clobbers."""
    _registry(tmp_path, code=True)
    path = tmp_path / "people.json"
    mine = Registry.load(path)
    other = Registry.load(path)
    assert other.set_pending_code("hunter", "scrypt$x", "a1b2c3d4", "t")[0]
    assert other.save()
    mine.set_secret("hunter", "code_hash", pp.hash_secret("manual99"))
    ok, why = mine.save_checked()
    assert not ok and "changed" in why
    assert Registry.load(path).person("hunter").pending_code_id == "a1b2c3d4"
    fresh = Registry.load(path)
    fresh.set_secret("hunter", "code_hash", pp.hash_secret("manual99"))
    assert fresh.save_checked() == (True, "")


@pytest.mark.parametrize("bad", ["", "not-hex!", "a" * 40])
def test_a_pending_id_is_short_hex_only(tmp_path, bad):
    """The id travels in a spool file, a receipt and a log line; it is the
    one thing about the code that is allowed to be seen, so it must not be
    able to carry anything else."""
    reg = _registry(tmp_path, code=True)
    ok, why = reg.set_pending_code("hunter", "scrypt$x", bad, "t")
    assert not ok
