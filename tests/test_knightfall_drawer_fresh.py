"""The drawer's typed check reads the FILE, and a typed pending code is
the receipt in person (Knightfall weekly, 2026-09-06).

THE PRE-EXISTING BUG THIS LANE EXPOSED. ``JarvisApp.knightfall_code``
checked the typed code against ``gate.registry`` -- the copy loaded at
BOOT -- while the users tab (``people_unlock``) re-reads the file every
time. So a code set at a terminal opened the tab and was refused by the
drawer until a reload or a restart. A weekly issuer that writes the file
from another process is useless against a drawer that keeps its boot
copy, so the drawer now decides from the file as it is, and falls back to
the boot copy ONLY when the file is unusable (a corrupted file must not
lock the break-glass that the boot copy would still open).

No real mail (tests/test_send_file.FakeSMTP), no real registry (tmp_path),
invented codes only.
"""
from __future__ import annotations

import contextlib
import logging
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis import gate as gate_mod
from jarvis import identity
from jarvis import knightfall_weekly as kw
from jarvis import passphrase as pp
from jarvis import selfstate
from jarvis.config import PATHS
from jarvis.identity import Registry
from jarvis.ui import views as views_mod
from tests.test_notes_mail import GMAIL_CFG, FakeCfg
from tests.test_owner_gate import FAKE_CODE, _registry
from tests.test_send_file import FakeSMTP

OK_LINE = "Knightfall accepted, sir; a new code is in your inbox."
WEEKLY = "xxx777yy"
TERMINAL = "yyy888zz"
PENDING_ID = "ab12cd34"


@pytest.fixture(autouse=True)
def _fresh_transport(tmp_path, monkeypatch):
    FakeSMTP.made = []
    monkeypatch.setattr(PATHS, "KNIGHTFALL_WEEKLY",
                        tmp_path / "state" / "knightfall-weekly.json")
    yield
    FakeSMTP.made = []


def _app(tmp_path, *, mode="shadow"):
    reg = _registry(tmp_path, code=True)
    opts = {"owner.mode": mode}
    a = SimpleNamespace()
    a.assistant = FakeCfg(GMAIL_CFG)
    a.get_option = lambda k, d=None: opts.get(k, d)
    a.gate = gate_mod.OwnerGate(registry=reg, owner="hunter",
                                get_option=a.get_option)
    # The REAL methods, bound to a bare namespace: this exercises the
    # production code, so every helper one of them calls has to be here
    # too (_knightfall_store is the locked half of the rotate;
    # _knightfall_weekly_caption is what knightfall_status reads).
    for name in ("knightfall_code", "knightfall_new_code", "knightfall_status",
                 "_knightfall_rotate", "_knightfall_store",
                 "_knightfall_weekly_caption", "_knightfall_check_registry",
                 "_knightfall_promote_by_use", "_people_registry",
                 "people_unlock", "_people_open_unlock", "_people_unlock_left"):
        setattr(a, name, getattr(app_mod.JarvisApp, name).__get__(a))
    return a


def _on_disk(a) -> Registry:
    return Registry.load(a.gate.registry.path)


def _mailed_code() -> str:
    assert FakeSMTP.made, "nothing was sent"
    return FakeSMTP.made[-1].sent[-1].get_content().splitlines()[0].strip()


def _set_pending(a, code=WEEKLY, code_id=PENDING_ID):
    ok, why = identity.locked_update(
        a.gate.registry.path,
        lambda r: r.set_pending_code("hunter", pp.hash_secret(code), code_id,
                                     "2026-09-13T07:30:00+00:00"))
    assert ok, why
    kw.write_state(PATHS.KNIGHTFALL_WEEKLY, {
        "id": code_id, "status": "pushed", "week": "2026-W37",
        "for": "2026-09-13", "pushed_at": "2026-09-13T07:30:00+00:00",
        "line": kw.LINE_PUSHED})


def _clean(caplog, *codes):
    hay = [caplog.text] + [r.getMessage() for r in caplog.records] + \
        [str(r.args) for r in caplog.records]
    for code in codes:
        assert code
        for text in hay:
            assert code not in text, "a plaintext code leaked"


# ------------------------------------------------ the drawer reads the file
def test_the_drawer_admits_a_code_set_at_a_terminal_since_boot(tmp_path, caplog):
    a = _app(tmp_path)
    disk = _on_disk(a)
    assert disk.set_secret("hunter", "code_hash", pp.hash_secret(TERMINAL))[0]
    assert disk.save()
    assert gate_mod.check_override_code(a.gate.registry, TERMINAL)[0] == "", \
        "the boot copy has never heard of it"
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(TERMINAL, smtp=FakeSMTP)
    assert line == OK_LINE, line
    assert a.gate._granted(None) == "hunter", "the floor did not open"
    _clean(caplog, TERMINAL, _mailed_code())


def test_the_old_code_is_refused_once_the_file_moved_on(tmp_path):
    a = _app(tmp_path)
    disk = _on_disk(a)
    assert disk.set_secret("hunter", "code_hash", pp.hash_secret(TERMINAL))[0]
    assert disk.save()
    line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert line == "Knightfall: that is not a code I know"
    assert FakeSMTP.made == []


def test_a_broken_file_falls_back_to_the_boot_copy(tmp_path):
    """A corrupted file must not lock the break-glass the boot copy would
    still open. The window opens; the rotate then cannot store (and says
    so), which is the same answer a corrupted file gives today."""
    a = _app(tmp_path)
    a.gate.registry.path.write_text("{ not json")
    line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert line.startswith("Knightfall accepted, sir;"), line
    assert a.gate._granted(None) == "hunter", "the floor did not open"
    assert a.gate.registry.path.read_text() == "{ not json", "nothing written over it"


def test_the_check_registry_prefers_the_file_and_falls_back_only_when_unusable(tmp_path):
    a = _app(tmp_path)
    reg = a._knightfall_check_registry()
    assert reg is not a.gate.registry and reg.usable
    a.gate.registry.path.write_text("{ not json")
    assert a._knightfall_check_registry() is a.gate.registry
    a.gate.registry.path.unlink()
    assert a._knightfall_check_registry() is a.gate.registry


# ---------------------------------------------- typed pending = the receipt
def test_typing_the_pending_code_promotes_it_on_the_spot(tmp_path, caplog):
    a = _app(tmp_path)
    _set_pending(a)
    with caplog.at_level(logging.DEBUG):
        line = a.knightfall_code(WEEKLY, smtp=FakeSMTP)
    assert line == OK_LINE, line
    disk = _on_disk(a)
    p = disk.person("hunter")
    assert p.pending_code_id == "" and p.pending_code_hash == "", "promoted"
    # burn-on-use, as today: the weekly code opened the door once and the
    # NEXT one is in his inbox; the code from before the week is dead.
    mailed = _mailed_code()
    assert gate_mod.check_override_code(disk, mailed)[0] == "hunter"
    assert gate_mod.check_override_code(disk, FAKE_CODE)[0] == ""
    assert gate_mod.check_override_code(disk, WEEKLY)[0] == ""
    st = kw.read_state(PATHS.KNIGHTFALL_WEEKLY)
    assert st["status"] == "promoted-by-use"
    assert "when you typed it" in kw.caption(PATHS.KNIGHTFALL_WEEKLY)
    assert any("promoted" in r.getMessage() and PENDING_ID in r.getMessage()
               for r in caplog.records)
    _clean(caplog, WEEKLY, mailed, FAKE_CODE)


def test_the_current_code_still_works_while_a_pending_waits(tmp_path):
    a = _app(tmp_path)
    _set_pending(a)
    line = a.knightfall_code(FAKE_CODE, smtp=FakeSMTP)
    assert line == OK_LINE
    p = _on_disk(a).person("hunter")
    assert p.pending_code_id == PENDING_ID, "a rotate never touches the pending"
    assert kw.read_state(PATHS.KNIGHTFALL_WEEKLY)["status"] == "pushed"


def test_the_users_tab_unlock_with_the_pending_code_promotes_too(tmp_path):
    a = _app(tmp_path)
    _set_pending(a)
    ok, line = a.people_unlock(WEEKLY)
    assert ok and line == "Unlocked, sir."
    disk = _on_disk(a)
    assert disk.person("hunter").pending_code_id == ""
    assert gate_mod.check_override_code(disk, WEEKLY)[0] == "hunter"
    assert gate_mod.check_override_code(disk, FAKE_CODE)[0] == ""
    assert kw.read_state(PATHS.KNIGHTFALL_WEEKLY)["status"] == "promoted-by-use"
    assert FakeSMTP.made == [], "an administrative unlock mails nothing"


def test_the_live_gate_is_reloaded_after_a_promotion(tmp_path):
    a = _app(tmp_path)
    _set_pending(a)
    a.people_unlock(WEEKLY)
    assert a.gate.registry.person("hunter").pending_code_id == ""
    assert gate_mod.check_override_code(a.gate.registry, WEEKLY)[0] == "hunter"


# ------------------------------------------------------------ the lock
def test_a_rotate_writes_under_the_file_lock(tmp_path, monkeypatch):
    a = _app(tmp_path)
    entered = []

    @contextlib.contextmanager
    def fake_lock(path, timeout_s=10.0):
        entered.append(str(path))
        yield
    monkeypatch.setattr(app_mod.identity_mod, "registry_lock", fake_lock)
    line, mailed = a._knightfall_rotate("hunter", smtp=FakeSMTP)
    assert mailed and line
    assert entered == [str(a.gate.registry.path)]


def test_a_rotate_that_cannot_take_the_lock_leaves_the_old_code_standing(
        tmp_path, monkeypatch):
    a = _app(tmp_path)

    @contextlib.contextmanager
    def held(path, timeout_s=10.0):
        raise identity.RegistryBusy("another writer holds the people book")
        yield  # pragma: no cover
    monkeypatch.setattr(app_mod.identity_mod, "registry_lock", held)
    line, mailed = a._knightfall_rotate("hunter", smtp=FakeSMTP)
    assert mailed and "old one stands" in line
    assert gate_mod.check_override_code(_on_disk(a), FAKE_CODE)[0] == "hunter"
    assert gate_mod.check_override_code(a.gate.registry, FAKE_CODE)[0] == "hunter"


def test_the_users_tab_write_and_the_cli_take_the_same_lock():
    """Source-level, deliberately: both writers are a page of scaffolding
    to drive and the claim is one line each."""
    import inspect
    from pathlib import Path
    src = inspect.getsource(app_mod.JarvisApp._people_write)
    assert "registry_lock" in src
    cli = (Path(app_mod.__file__).parent.parent / "scripts" / "jarvis_people.py").read_text()
    assert "save_checked" in cli
    assert "reg.save()" not in cli, "every CLI save goes through the checked path"


# ---------------------------------------------------------- the caption
def test_the_status_carries_the_weekly_caption_from_the_state_file(tmp_path):
    a = _app(tmp_path)
    assert a.knightfall_status()["weekly"] == ""
    _set_pending(a)
    st = a.knightfall_status()
    assert st["weekly"].startswith("Weekly: ")
    assert set(st) >= {"to", "problem", "setup", "weekly"}


def test_the_drawer_caption_appends_the_weekly_sentence_only_when_there_is_one():
    plain, can = views_mod.format_knightfall_status({"to": "k…@x.com"})
    assert plain == "Typed only, never spoken. The next code goes to k…@x.com."
    text, can2 = views_mod.format_knightfall_status(
        {"to": "k…@x.com", "weekly": kw.LINE_PUSHED})
    assert text == plain + " " + kw.LINE_PUSHED and can2 is can
    text, _ = views_mod.format_knightfall_status({"to": "", "weekly": kw.LINE_PUSHED})
    assert text.endswith(kw.LINE_PUSHED)


def test_the_diagnostics_line_carries_the_weekly_sentence(tmp_path):
    from tests.test_selfstate import state
    plain = selfstate.diagnostics_line(state())
    assert "Weekly" not in plain
    line = selfstate.diagnostics_line(state(knightfall_weekly=kw.LINE_PUSHED))
    assert line.startswith(plain)
    assert line.endswith(kw.LINE_PUSHED)


def test_self_state_reads_the_weekly_caption(tmp_path, monkeypatch):
    """The `jarvis -q status` sheet is built by self_state(); the caption
    is one small file read and never a socket."""
    kw.write_state(PATHS.KNIGHTFALL_WEEKLY, {
        "id": PENDING_ID, "status": "pushed", "line": kw.LINE_PUSHED})
    assert app_mod.JarvisApp._knightfall_weekly_caption(SimpleNamespace()) == kw.LINE_PUSHED
    PATHS.KNIGHTFALL_WEEKLY.unlink()
    assert app_mod.JarvisApp._knightfall_weekly_caption(SimpleNamespace()) == ""
