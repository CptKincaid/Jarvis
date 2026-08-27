"""Regression tests for a FOUND DEFECT (failure-mode sweep 2026-08-26).

DEFECT: a corrupt timekeeper.db / notes.db is never quarantined or
recreated, so the capability is lost permanently and silently.

  * Timekeeper._open()   (jarvis/tools/timekeeper.py ~line 909)
  * NotesStore.__init__() (jarvis/tools/notes.py ~line 152)

both did `sqlite3.connect(...)` + `executescript(_SCHEMA)` with NO
try/except.  On a corrupt file that raises sqlite3.DatabaseError
("file is not a database" / "database disk image is malformed").

jarvis/app.py::_construct catches it, so the app still BOOTS (good) --
but the member is left None forever and nothing ever touches the bad
file.  Every subsequent boot re-reads the same corrupt bytes, so from
then on the user hears only

    "I'm afraid my timekeeper isn't running, sir."

for every alarm, timer and reminder, with the real reason visible only in
jarvis.log.  Alarms silently stop firing.

The same codebase already implements the correct behaviour one module
over: AssistantConfig.load() (jarvis/assistant_config.py ~line 296)
moves a corrupt config to `<name>.bad` and recreates it.

STATUS
  timekeeper: FIXED 2026-08-26.  Timekeeper._open() now proves the file
    with PRAGMA quick_check, salvages what rows it can, moves the bad
    file aside as `<name>.bad-<YYYYmmdd-HHMMSS>` (timestamped so a
    second corruption cannot overwrite the first evidence, and never
    deleted), recreates the store, restores the salvaged items and tells
    the user in persona where the old file went.  The tests below are
    live.
  notes: still open -- NotesStore is owned by another work item, so its
    cases stay strict-xfail.
"""
import sqlite3
import time

import pytest

from jarvis.tools.notes import NotesStore
from jarvis.tools.timekeeper import Timekeeper

# Shapes real corruption takes: junk bytes, a torn page, all-zero blocks.
CORRUPT = {
    "junk": b"this is not a sqlite database at all\n" * 40,
    "truncated_header": b"SQLite format 3\x00",
    "zero_filled": b"\x00" * 4096,
}


def _timekeeper(path, **kw):
    return Timekeeper(path, tick_s=99, ring=False, notify=False, **kw)


def _quarantined(db):
    """Every file the store moved aside for this db."""
    return sorted(db.parent.glob(db.name + ".bad*"))


# --------------------------------------------------------------- timekeeper
@pytest.mark.parametrize("shape", sorted(CORRUPT))
def test_corrupt_timekeeper_is_quarantined_and_recreated(tmp_path, shape):
    """Opening a corrupt store must move the bad file aside (keeping the
    bytes) and start a fresh working one, not raise."""
    db = tmp_path / "timekeeper.db"
    db.write_bytes(CORRUPT[shape])

    said = []
    tk = _timekeeper(db, say=said.append)

    bad = _quarantined(db)
    assert len(bad) == 1, "the corrupt file should have been quarantined"
    assert bad[0].read_bytes() == CORRUPT[shape], "the bad bytes must be preserved"
    assert tk.quarantined == bad[0]
    assert said and bad[0].name in said[0] and "sir" in said[0].lower(), \
        f"the user must be told, in persona, where the file went: {said}"

    # ...and the fresh store works.
    tk.add_reminder(time.time() + 600, "buy milk")
    assert [i.label for i in tk.list()] == ["buy milk"]
    tk.close()

    # A second corruption keeps the first quarantine (timestamped names).
    db.write_bytes(CORRUPT[shape])
    tk2 = _timekeeper(db)
    assert len(_quarantined(db)) == 2
    tk2.close()


@pytest.mark.parametrize("shape", sorted(CORRUPT))
def test_corrupt_timekeeper_constructor_does_not_raise(tmp_path, shape):
    """The narrow version of the above: whatever the recovery strategy,
    the constructor must not raise DatabaseError at the app's boot
    boundary."""
    db = tmp_path / "timekeeper.db"
    db.write_bytes(CORRUPT[shape])
    try:
        _timekeeper(db).close()
    except sqlite3.DatabaseError as exc:
        pytest.fail(f"timekeeper constructor raised {type(exc).__name__}: {exc}")


def test_truncated_timekeeper_recovers_what_it_can(tmp_path):
    """A populated db truncated mid-file: the readable items come back and
    the user is told how many, and the scheduler still runs."""
    db = tmp_path / "timekeeper.db"
    tk = _timekeeper(db)
    for i in range(30):
        tk.add_alarm(time.time() + 3600 + i, f"alarm {i}")
    tk.close()
    raw = db.read_bytes()
    db.write_bytes(raw[:len(raw) // 2])          # torn mid-file

    said = []
    tk2 = _timekeeper(db, say=said.append)
    assert len(_quarantined(db)) == 1
    assert tk2.recovered == len(tk2.list()) > 0, \
        "recoverable items should be restored on a best-effort basis"
    assert said and "damaged" in said[0]

    # ...and the scheduler runs normally on the fresh store.
    fired = []
    tk2._say = fired.append
    tk2.add_reminder(time.time() - 1, "stretch")
    tk2.tick()
    assert any("stretch" in line.lower() for line in fired), fired
    tk2.close()


def test_fresh_store_after_quarantine_does_not_invent_missed_items(tmp_path):
    """catch_up() on the recreated store must not announce anything that
    was lost as if it had fired or been missed."""
    db = tmp_path / "timekeeper.db"
    db.write_bytes(CORRUPT["junk"])
    said = []
    tk = _timekeeper(db, say=said.append)
    said.clear()                                 # drop the quarantine notice
    assert tk.catch_up() == []
    assert said == []
    tk.close()


# -------------------------------------------------------------------- notes
@pytest.mark.xfail(reason="FOUND DEFECT: corrupt notes store is not quarantined",
                   strict=True)
@pytest.mark.parametrize("shape", sorted(CORRUPT))
def test_corrupt_notes_store_is_quarantined_and_recreated(tmp_path, shape):
    db = tmp_path / "notes.db"
    db.write_bytes(CORRUPT[shape])

    store = NotesStore(db)       # currently raises sqlite3.DatabaseError

    assert _quarantined(db), "the corrupt file should have been quarantined"
    store.close()


@pytest.mark.xfail(reason="FOUND DEFECT: notes constructor raises on a corrupt db",
                   strict=True)
def test_corrupt_notes_constructor_does_not_raise(tmp_path):
    db = tmp_path / "notes.db"
    db.write_bytes(CORRUPT["junk"])
    try:
        NotesStore(db).close()
    except sqlite3.DatabaseError as exc:
        pytest.fail(f"notes constructor raised {type(exc).__name__}: {exc}")
