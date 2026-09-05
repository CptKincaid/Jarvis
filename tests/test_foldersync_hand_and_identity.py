"""Round 4: his hand in his own folder, and identity instead of pattern.

NOTHING HERE OPENS A SOCKET OR TOUCHES ~/Desktop.  Every test works in
``tmp_path`` against the same FAKE far side as tests/test_foldersync.py,
whose fixtures are imported rather than copied.

The seven things an independent attacker measured on the round-3 tip
(36d26fe) and this file pins:

L    a pass DIES when he moves his own file out of the Outbox mid-pass;
     status.txt is never rewritten, the inbound half never runs, and a
     file that HAD landed is sent again next pass as a duplicate.
M/M2 the delete guard is a NAME PATTERN: a file of his called
     jarvis-part-notes.tmp is deleted, and so is the voice lane's
     in-flight file -- which then tells him "There's already a file by
     that name", which is false.
row3 the push scp writes at a remote temp name with NO claim on it.
row5 the pull scp writes at inbox/.jarvis-part-<name> with no claim.
Q    the 400-entry listing cap is read as "not there": a landed file is
     reported missing, and an inbound file past the cap is hidden.
O    the voice staging name has no counter.
row14 the "cannot send" note is HIS filename plus a suffix, so it can
     overwrite and then delete a file of his.
"""
# ruff: noqa: F811 -- `home` is a pytest FIXTURE imported from
# tests/test_foldersync.py rather than copied; every test that takes it as a
# parameter looks like a redefinition and is not one.
import os
from pathlib import Path

import pytest

from jarvis import foldersync as fs
from jarvis.tools import remote

from tests.test_foldersync import (Cfg, FakeTransport, drop, home,  # noqa: F401
                             paths_for, syncer)


# ------------------------------------------------------------------ helpers
def stock_pull(tr, name="in.txt", data=b"inbound"):
    """One file waiting on the far side, so a pass has an inbound half."""
    tr.put("outbox", name, data)
    return name


class HandInTheOutbox(FakeTransport):
    """His hand, at the worst instant.  ``on_listing`` fires on the Nth
    listing of the pass -- the VERIFY listing is the second one -- which is
    exactly the window in which he drags a file back out of the Outbox."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.on_listing = None
        self.at = 2
        self._n = 0

    def listing(self, key):
        self._n += 1
        if self.on_listing and self._n == self.at:
            self.on_listing()
        return super().listing(key)


# --------------------------------------------------------------- FINDING L
def test_his_hand_in_the_outbox_does_not_kill_the_pass(home):
    """He drags a.txt out and drops b.txt in while the pass is verifying.

    Three things must hold and none of them did: the pass must not raise,
    status.txt must be rewritten (an untrue status file is one of his
    stated merge conditions), and the INBOUND half must still run.
    """
    tr = HandInTheOutbox()
    stock_pull(tr)
    s = syncer(home, tr)
    a = drop(home, "a.txt", b"1234567")

    def his_hand():
        a.rename(home / "Outbox" / "b.txt")      # a drag, mtime preserved

    tr.on_listing = his_hand
    events = s.run_pass()                        # must not raise

    status = (home / "status.txt")
    assert status.exists(), "status.txt was never written"
    text = status.read_text()
    assert "outbox    empty" not in text, \
        "status.txt says empty while b.txt is sitting in the Outbox"
    assert "b.txt" in text
    assert any(e.direction == "pull" and e.outcome == "received"
               for e in events), "the inbound half never ran"


def test_a_file_that_already_landed_is_never_sent_twice(home):
    """Pass 1 lands a.txt on the far side and then his hand takes the local
    original away before it can be moved to Sent.  Pass 2 -- he puts the
    same file back, the way a file manager puts it back, mtime intact --
    must NOT send a second copy.
    """
    tr = HandInTheOutbox()
    s = syncer(home, tr)
    a = drop(home, "a.txt", b"1234567")
    held = home / "held.txt"
    stat = a.stat()

    def his_hand():
        a.rename(held)

    tr.on_listing = his_hand
    s.run_pass()
    assert "a.txt" in tr.dirs["inbox"], "pass 1 did not land it"

    held.rename(a)                               # he puts it back
    os.utime(a, (stat.st_atime, stat.st_mtime))  # a move keeps the mtime
    s.run_pass()

    copies = sorted(n for n in tr.dirs["inbox"] if n.startswith("a"))
    assert copies == ["a.txt"], f"sent again as a duplicate: {copies}"


# ------------------------------------------------------------ FINDINGS M/M2
def test_a_file_of_his_that_matches_the_part_pattern_survives(home):
    """`jarvis-part-notes.tmp` is a name HE can choose.  The guard was a
    pattern, so it was deleted -- no event, no note, no line anywhere."""
    tr = FakeTransport()
    tr.put("inbox", "jarvis-part-notes.tmp", b"his own working file")
    s = syncer(home, tr)
    drop(home, "a.txt")
    s.run_pass()
    assert "jarvis-part-notes.tmp" in tr.dirs["inbox"], \
        "a file of his was deleted on HPCOMPUTER"
    assert "jarvis-part-notes.tmp" not in tr.discarded


def test_the_voice_lanes_in_flight_file_survives_a_foldersync_pass(home):
    """The voice lane stages at jarvis-part-<ITS pid>-<t>-<n>.tmp.  The
    sweep matched the pattern and ate it mid-transfer."""
    other = f"{remote.REMOTE_TEMP_PREFIX}{os.getpid() + 1}-1757000000-0" \
            f"{remote.REMOTE_TEMP_SUFFIX}"
    tr = FakeTransport()
    tr.put("inbox", other, b"the voice lane's bytes, in flight")
    s = syncer(home, tr)
    drop(home, "a.txt")
    s.run_pass()
    assert other in tr.dirs["inbox"], \
        "the voice lane's in-flight file was deleted by the folder lane"


def test_a_staging_file_that_vanished_is_not_reported_as_a_taken_name(
        tmp_path, monkeypatch):
    """M2's second half.  When the claim fails because OUR OWN staging file
    is gone, the lane must not say "There's already a file by that name" --
    nothing held the name; we took our own file away."""
    # A key that EXISTS, or missing_reason refuses before push reaches the
    # claim at all and this test is green about nothing.  (It was: the first
    # draft passed against the unfixed code for exactly that reason.)
    key = tmp_path / "hpcomputer.key"
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    conf = remote.read_config(Cfg(**{
        "remote.enabled": True, "remote.host": "192.168.50.114",
        "remote.user": "jarvis", "remote.key_path": str(key),
        "remote.inbox": "/C:/Users/h2pey/Desktop/Jarvis/Inbox",
        "remote.local_roots": (str(tmp_path),)}))
    assert remote.missing_reason(conf) == "", "this test must reach push()"
    local = tmp_path / "report.pdf"
    local.write_bytes(b"x" * 32)

    monkeypatch.setattr(remote, "run_copy",
                        lambda *a, **k: remote.SshResult(True))
    monkeypatch.setattr(remote, "sftp_rename",
                        lambda *a, **k: remote.SshResult(False,
                                                         reason="failed"))
    # The far side, asked what is in our staging area, says it is empty --
    # somebody took our file away.
    monkeypatch.setattr(remote, "run_sftp",
                        lambda *a, **k: remote.SshResult(True, out=""))
    monkeypatch.setattr(remote, "sftp_remove",
                        lambda *a, **k: remote.SshResult(True))
    monkeypatch.setattr(remote, "sftp_mkdir",
                        lambda *a, **k: remote.SshResult(True), raising=False)
    monkeypatch.setattr(remote, "sftp_rmdir",
                        lambda *a, **k: remote.SshResult(True), raising=False)

    res = remote.push(conf, local)
    assert not res.ok
    assert res.reason != "exists", \
        "a vanished staging file was reported as a name of his being taken"
    line = remote.fail_line(conf, res.reason)
    assert "already a file by that name" not in line, line


# --------------------------------------------------------------- rows 3 & 5
def test_the_push_never_writes_at_a_remote_name_it_has_not_claimed(home):
    """Row 3.  The bytes went to a temp name with NO check at all: a
    99999-byte file of his at that name was replaced by 30 bytes and the
    pass said "sent"."""
    tr = FakeTransport()
    s = syncer(home, tr)
    drop(home, "a.txt", b"x" * 30)
    s.run_pass()
    opened = [n for kind, n in tr.calls if kind == "stage_open"]
    sends = [n for kind, n in tr.calls if kind == "send"]
    assert opened, "no exclusive claim was taken before the copy"
    assert sends and all(n.startswith(opened[0] + "/") for n in sends), \
        f"scp wrote at {sends} which is outside the claimed {opened}"


def test_the_pull_claims_its_part_file_and_never_writes_over_one(home):
    """Row 5.  ``.jarvis-part-<name>`` in HIS Inbox was written with no
    O_EXCL and no counter -- 99999 bytes destroyed, "received" reported."""
    tr = FakeTransport()
    stock_pull(tr, "in.txt", b"y" * 40)
    his = home / "Inbox" / (fs.PART_PREFIX + "in.txt")
    his.write_bytes(b"H" * 99999)
    s = syncer(home, tr)
    s.run_pass()
    assert his.exists() and his.stat().st_size == 99999, \
        "the pull destroyed a file sitting at its part name"


# --------------------------------------------------------------- FINDING Q
class CappedTransport(FakeTransport):
    """A far side whose listing is truncated the way the real parse
    truncates it: the first LISTING_CAP entries, by name."""

    def listing(self, key):
        rows, why = super().listing(key)
        return sorted(rows, key=lambda e: e.name)[:remote.LISTING_CAP], why


def test_a_landed_file_is_not_reported_missing_by_a_capped_listing(home):
    """400+ files in the Windows Inbox and zebra.pdf lands past the cap.
    It was on the box at exactly the right size; the note said "HPCOMPUTER
    reports no such file" and twelve passes left five orphans."""
    tr = CappedTransport()
    for i in range(remote.LISTING_CAP):
        tr.put("inbox", f"a{i:04d}.bin", b"x")
    s = syncer(home, tr)
    drop(home, "zebra.pdf", b"z" * 12)
    events = s.run_pass()
    outcomes = [e.outcome for e in events if e.direction == "push"]
    assert outcomes == ["sent"], outcomes
    assert (home / "Sent" / "zebra.pdf").exists()
    assert not list((home / "Outbox").glob("*" + fs.NOTE_SUFFIX))


def test_the_listing_parse_does_not_hide_a_file_past_four_hundred():
    """The cap belonged to the PARSE, so entry 401 was invisible for ever:
    six passes, never fetched, nothing said anywhere, status "link OK"."""
    lines = ["sftp> ls -ln \"box\""]
    for i in range(500):
        lines.append(f"-rw-rw-r--    ? h  h   7 Sep  5 13:45 box/f{i:04d}.txt")
    rows, truncated = fs.parse_sftp_entries("\n".join(lines))
    assert len(rows) == 500, len(rows)
    assert truncated is False


def test_a_listing_too_big_to_read_is_said_out_loud_not_silently_cut(home):
    lines = ["sftp> ls -ln \"box\""]
    for i in range(fs.LISTING_HARD_CAP + 25):
        lines.append(f"-rw-rw-r--    ? h  h   7 Sep  5 13:45 box/f{i:06d}.txt")
    rows, truncated = fs.parse_sftp_entries("\n".join(lines))
    assert len(rows) == fs.LISTING_HARD_CAP
    assert truncated is True


# --------------------------------------------------------------- FINDING O
def test_two_voice_sends_in_one_second_do_not_share_a_staging_name(
        monkeypatch):
    monkeypatch.setattr(remote.time, "time", lambda: 1757000000.0)
    names = {remote.remote_temp_name() for _ in range(8)}
    assert len(names) == 8, f"the staging name has no counter: {names}"


# ---------------------------------------------------------------- row 14
def test_a_file_of_his_at_the_note_name_is_neither_overwritten_nor_deleted(
        home):
    """The note is HIS filename plus a suffix, and _sweep_notes deletes any
    such name whose base file is gone."""
    tr = FakeTransport()
    tr.send_fail = {"report.pdf": "denied"}
    s = syncer(home, tr)
    p = drop(home, "report.pdf")
    his = home / "Outbox" / ("report.pdf" + fs.NOTE_SUFFIX)
    his.write_bytes(b"HIS OWN NOTES, not Jarvis's")
    s.run_pass()
    assert his.read_bytes() == b"HIS OWN NOTES, not Jarvis's", \
        "the note overwrote a file of his"
    p.unlink()
    s.run_pass()
    assert his.exists(), "the note sweep deleted a file of his"


def test_our_own_note_is_still_written_cleared_and_swept(home):
    tr = FakeTransport()
    tr.send_fail = {"report.pdf": "denied"}
    s = syncer(home, tr)
    p = drop(home, "report.pdf")
    s.run_pass()
    note = home / "Outbox" / ("report.pdf" + fs.NOTE_SUFFIX)
    assert note.exists() and note.read_text().startswith(fs.NOTE_MARKER)
    p.unlink()
    s.run_pass()
    assert not note.exists(), "our own note was not swept"


# ------------------------------------------- the question that is his alone
def test_a_broken_copy_is_left_alone_unless_he_turns_it_on(home):
    """Verify fails, so a copy under HIS name is sitting broken on Windows.
    Taking it back is a WIDER promise than this lane has ever made, so it
    is a setting, and the setting is OFF."""
    tr = FakeTransport()
    tr.short_write = 3
    s = syncer(home, tr)
    drop(home, "a.txt", b"x" * 30)
    events = s.run_pass()
    assert [e.outcome for e in events if e.direction == "push"] == \
        ["verify-failed"]
    assert "a.txt" in tr.dirs["inbox"], \
        "the broken copy was removed with the setting off"


def test_the_broken_copy_is_taken_back_only_when_he_asks_for_it(home):
    tr = FakeTransport()
    tr.short_write = 3
    s = syncer(home, tr, remove_broken_copies=True)
    drop(home, "a.txt", b"x" * 30)
    events = s.run_pass()
    assert "a.txt" not in tr.dirs["inbox"], "the broken copy was left behind"
    assert any(e.detail and "removed" in e.detail.lower()
               for e in events if e.direction == "push")
    assert fs.DEFAULTS_REMOVE_BROKEN is False


def test_the_shipped_setting_is_off(home):
    assert fs.read_config(Cfg()).remove_broken_copies is False


# ----------------------------------------------- the claim, actually measured
SFTP_SERVER = "/usr/lib/openssh/sftp-server"


@pytest.mark.skipif(not os.environ.get("JARVIS_SLOW_MEASURE"),
                    reason="set JARVIS_SLOW_MEASURE=1: spawns a real "
                           "sftp-server (no socket, no HPCOMPUTER)")
@pytest.mark.skipif(not Path(SFTP_SERVER).exists(),
                    reason="no local sftp-server to measure against")
def test_measure_that_sftp_mkdir_really_is_an_exclusive_claim(tmp_path,
                                                              capsys):
    """ROW 3 rests on one claim: that ``mkdir`` refuses a name that is held.

    NO SOCKET AND NO HPCOMPUTER.  ``sftp -D`` pipes straight into this
    box's own sftp-server, which is how every other sftp fact in this lane
    was established.  The Windows far side is MODELLED, never reached, and
    that limit is written down beside the numbers in remote.py.
    """
    import subprocess

    def sftp(line):
        return subprocess.run(["sftp", "-q", "-b", "-", "-D", SFTP_SERVER],
                              input=line + "\n", cwd=tmp_path, text=True,
                              capture_output=True, timeout=30).returncode

    (tmp_path / "inbox").mkdir()
    stage = "inbox/" + remote.remote_stage_name()
    assert sftp(f'mkdir "{stage}"') == 0                  # free name: taken
    assert sftp(f'mkdir "{stage}"') != 0                  # held name: refused
    (tmp_path / "inbox" / "his.pdf").write_bytes(b"H" * 99999)
    assert sftp('mkdir "inbox/his.pdf"') != 0             # a FILE blocks it

    # ...and the claim survives being raced.  Bounded: 20 rounds, hard.
    winners = []
    for i in range(20):
        target = f"inbox/race{i}" + remote.REMOTE_STAGE_SUFFIX
        procs = [subprocess.Popen(
            ["sftp", "-q", "-b", "-", "-D", SFTP_SERVER],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, text=True, cwd=tmp_path)
            for _ in range(2)]
        for pr in procs:
            pr.stdin.write(f'mkdir "{target}"\n')
        for pr in procs:
            pr.stdin.close()
        winners.append(sum(1 for pr in procs if pr.wait(timeout=30) == 0))
    assert winners == [1] * 20, winners

    # The other half: rename -l ACROSS directories still refuses, and the
    # file it refused to overwrite is byte-intact.
    (tmp_path / stage / "f").write_bytes(b"o" * 40)
    assert sftp(f'rename -l "{stage}/f" "inbox/his.pdf"') != 0
    assert (tmp_path / "inbox" / "his.pdf").stat().st_size == 99999
    assert (tmp_path / stage / "f").stat().st_size == 40
    # ...and the release can never take a byte with it.
    assert sftp(f'rmdir "{stage}"') != 0                  # not empty: refused
    with capsys.disabled():
        print(f"\n  MEASURED, local sftp-server, no socket:"
              f"\n    mkdir at a free name -> taken; at a held name or over a"
              f"\n      FILE -> refused."
              f"\n    2 sessions racing one mkdir, 20 rounds: exactly one"
              f"\n      winner {sum(1 for w in winners if w == 1)}/20."
              f"\n    rename -l across directories onto a held name: refused,"
              f"\n      his 99999 bytes intact, our 40 intact."
              f"\n    rmdir of a non-empty directory: refused.")
