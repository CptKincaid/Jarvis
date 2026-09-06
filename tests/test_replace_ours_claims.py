"""THE AUDITED HELPER WROTE ITS TEMP WITH NO CLAIM.  Round 10, item 1.

``_replace_ours`` is the chokepoint round 7 was asked to build for "replace
a file we own atomically".  Its temp name was ``<target>.tmp`` -- fixed,
guessable, and opened with a plain ``open(tmp, "w")``, which TRUNCATES
whatever is at that name.  The threat model says, in words, "CLAIM: every
write that could land where a file already is takes one operation that
creates the name or refuses".  That sentence was not true of this one.

MEASURED on the round-9 tip (1039ce8), in tmp_path, nothing of his touched:

    a file of his at ~/Desktop/Jarvis/status.txt.tmp, 100000 bytes
    one pass of the syncer
    status.txt.tmp                            GONE
    status.txt                                our status text, ~300 bytes

His bytes were truncated to our text and then renamed over status.txt.  A
narrow name and an undocumented one, but a name he can have, and the fix is
the shape this lane already uses for the pull's part file: ``O_CREAT|O_EXCL``
on a UNIQUE name (pid + counter), so a taken name is REFUSED by the kernel
and the next one is tried.  The open row in the census then honestly earns
``claim:excl-open`` instead of ``NOTHING``.

Everything here works in ``tmp_path`` against the same FAKE far side as
tests/test_foldersync.py, whose fixtures are imported rather than copied.
Nothing here opens a socket, touches ~/Desktop, or speaks to HPCOMPUTER.
"""
# ruff: noqa: F811 -- `home` is a pytest FIXTURE imported from
# tests/test_foldersync.py rather than copied.
import itertools
import os

import pytest

from jarvis import foldersync as fs

from tests.test_foldersync import home, syncer  # noqa: F401

HIS = b"h" * 100000


def test_a_file_of_his_at_the_status_temp_name_survives_a_pass(home):
    """The measured destruction, as a pin.  Before the fix: 100000 -> GONE."""
    his = home / "status.txt.tmp"
    his.write_bytes(HIS)
    s = syncer(home)

    s.run_pass()

    assert his.exists(), (
        "his 100000-byte file at status.txt.tmp is GONE after one pass: the "
        "status writer opened it, truncated it to our text and renamed it "
        "over status.txt")
    assert his.read_bytes() == HIS, (
        f"his file at status.txt.tmp is now {his.stat().st_size} bytes")
    assert (home / "status.txt").exists(), "and status.txt was still written"


def test_the_temp_name_is_claimed_not_opened(tmp_path, monkeypatch):
    """A file of his at the EXACT name we try first is refused by the kernel
    and left alone; the write goes to the next name and still lands."""
    monkeypatch.setattr(fs, "_REPLACE_SEQ", itertools.count(41))
    target = tmp_path / "status.txt"
    taken = tmp_path / f"status.txt.{os.getpid()}-41.tmp"
    taken.write_bytes(HIS)

    fs._replace_ours(target, "ours\n")

    assert taken.read_bytes() == HIS, "the taken name was written through"
    assert target.read_text() == "ours\n"
    assert not (tmp_path / f"status.txt.{os.getpid()}-42.tmp").exists(), (
        "the temp we did use was left behind")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "status.txt", taken.name]


def test_the_old_fixed_temp_name_is_not_used_at_all(tmp_path):
    """The name ``<target>.tmp`` is his to have now.  Whatever sits there is
    neither read, written nor renamed."""
    target = tmp_path / "ledger.json"
    old_name = tmp_path / "ledger.json.tmp"
    old_name.write_bytes(HIS)

    fs._replace_ours(target, "{}", fsync=True)

    assert old_name.read_bytes() == HIS
    assert target.read_text() == "{}"


def test_every_temp_name_taken_refuses_and_touches_nothing(tmp_path,
                                                            monkeypatch):
    """Bounded, like every other claim loop in this lane: after
    MAX_CLAIM_TRIES refusals it raises the OSError its callers already
    catch, and not one of his files, nor the target, has changed."""
    monkeypatch.setattr(fs, "_REPLACE_SEQ", itertools.count(1))
    target = tmp_path / "status.txt"
    target.write_text("before\n")
    his = []
    for n in range(1, fs.MAX_CLAIM_TRIES + 1):
        p = tmp_path / f"status.txt.{os.getpid()}-{n}.tmp"
        p.write_bytes(HIS)
        his.append(p)

    with pytest.raises(OSError):
        fs._replace_ours(target, "after\n")

    assert target.read_text() == "before\n", "the target changed on a refusal"
    assert all(p.read_bytes() == HIS for p in his), "one of his files changed"
    assert len(list(tmp_path.iterdir())) == len(his) + 1, "something was left"
