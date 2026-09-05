"""The table, PINNED.  Add a write to this lane and this test fails.

tests/write_census.py walks the AST of jarvis/foldersync.py and
jarvis/tools/remote.py and lists every call that can create or destroy a
name, with whatever guards it in the same function.  :data:`KNOWN` below is
the one line per write that somebody has actually thought about.

WHY THIS EXISTS RATHER THAN A PARAGRAPH IN A DOCSTRING.  Three rounds of
this lane each shipped a hand-written version of this table.  Round 2's
said "nothing else in the repo has either shape" and was false.  Round 3's
had fourteen rows and was missing TWO of the most dangerous writes in the
module -- the push's own scp at its temp name (unbounded window, measured
destroying a 99999-byte file while reporting "sent") and the pull's scp
into its part file (the whole transfer, another file destroyed).  Both were
writes with NO GUARD AT ALL, which is precisely the row a person skims past
and a machine cannot.

If this test fails, do not edit KNOWN to make it green.  Read the failure:
it is telling you that a write appeared, or moved, or lost its guard.
"""
from tests.write_census import census, table


# (module, function, write primitive) -> what can be at that name, in a line.
KNOWN = {
    # ---------------------------------------------------- the ledger, ours
    ("jarvis/foldersync.py", "Ledger.save", "mkdir"):
        "~/.local/state/jarvis, ours; exist_ok",
    ("jarvis/foldersync.py", "Ledger.save", "replace"):
        "tmp -> ledger, ours, one process holds the flock",
    ("jarvis/foldersync.py", "Ledger.save", "open"):
        "our own .tmp beside the ledger, then the dir fd for the fsync",

    # ------------------------------------------- the far side, through ssh
    ("jarvis/foldersync.py", "SshTransport.claim", "sftp_rename"):
        "CLAIM: rename -l, kernel refuses a held name; window ZERO",
    ("jarvis/foldersync.py", "SshTransport.discard", "sftp_remove"):
        "only a temp of ours or a file inside a stage of ours; guarded twice",
    ("jarvis/foldersync.py", "SshTransport.fetch", "run_copy"):
        "READS over there; writes at the local part file the caller claimed",
    ("jarvis/foldersync.py", "SshTransport.listing", "run_sftp"):
        "an ls; writes nothing",
    ("jarvis/foldersync.py", "SshTransport.remove_landed", "sftp_remove"):
        "THE GATED ONE: his filename, only with remove_broken_copies on, "
        "only a name this run took; logged at WARNING every time",
    ("jarvis/foldersync.py", "SshTransport.send", "run_copy"):
        "inside a staging directory the caller took exclusively (row 3); "
        "the stat before it is the local size, not a question about the name",
    ("jarvis/foldersync.py", "SshTransport.stage_close", "sftp_rmdir"):
        "our stage only, and the server refuses a non-empty one",
    ("jarvis/foldersync.py", "SshTransport.stage_open", "sftp_mkdir"):
        "THE CLAIM for row 3: mkdir refuses any held name; window ZERO",
    ("jarvis/foldersync.py", "SshTransport.stat", "run_sftp"):
        "an ls of ONE name; writes nothing (finding Q)",

    # --------------------------------------------------- the syncer's own
    ("jarvis/foldersync.py", "Syncer._claim_part", "open"):
        "O_CREAT|O_EXCL in his Inbox: ROW 5's claim; window ZERO",
    ("jarvis/foldersync.py", "Syncer._close_stage", "stage_close"):
        "the stage THIS pass opened",
    ("jarvis/foldersync.py", "Syncer._discard", "discard"):
        "a name inside a stage of ours",
    ("jarvis/foldersync.py", "Syncer._drop", "unlink"):
        "our own part file, claimed with O_EXCL a moment earlier",
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "land_beside"):
        "os.link into Sent; window ZERO, and a vanished source is his hand",
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "mkdir"):
        "~/Desktop/Jarvis/Sent, exist_ok",
    ("jarvis/foldersync.py", "Syncer._open_stage", "stage_open"):
        "the mkdir claim, once per pass",
    ("jarvis/foldersync.py", "Syncer._push_once", "save"):
        "the ledger, ours",
    ("jarvis/foldersync.py", "Syncer._remove_broken_copy", "remove_landed"):
        "OFF by default; identity is self._claimed_here, not a name shape",
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "save"):
        "the ledger, ours",
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "claim"):
        "rename -l onto his name; refuses, never replaces",
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "save"):
        "the ledger, ours -- and it is BEFORE the claim on purpose (L)",
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "send"):
        "ROW 3: inside the stage taken by _open_stage; window ZERO",
    ("jarvis/foldersync.py", "Syncer._sweep_my_stages", "stage_close"):
        "only a stage name in self._my_stages (findings M and M2)",
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "save"):
        "the ledger, ours",
    ("jarvis/foldersync.py", "Syncer._write_note", "fdopen"):
        "ROW 14: the fd is the one whose marker was just read",
    ("jarvis/foldersync.py", "Syncer._write_note", "open"):
        "ROW 14: O_EXCL first; an existing note is opened WITHOUT truncating",
    ("jarvis/foldersync.py", "Syncer._write_note", "truncate"):
        "ROW 14: on the checked fd, so never on a file of his",
    ("jarvis/foldersync.py", "Syncer.pull_once", "fetch"):
        "ROW 5: into the part file _claim_part took with O_EXCL",
    ("jarvis/foldersync.py", "Syncer.pull_once", "land_beside"):
        "os.link into his Inbox; window ZERO",
    ("jarvis/foldersync.py", "Syncer.pull_once", "mkdir"):
        "~/Desktop/Jarvis/Inbox, exist_ok",
    ("jarvis/foldersync.py", "Syncer.pull_once", "save"):
        "the ledger, ours",
    ("jarvis/foldersync.py", "Syncer.record", "mkdir"):
        "~/.local/state/jarvis, ours",
    ("jarvis/foldersync.py", "Syncer.record", "open"):
        "APPEND to our own history file",
    ("jarvis/foldersync.py", "Syncer.record", "write_text"):
        "our own history file, trimmed; no name of his is involved",
    ("jarvis/foldersync.py", "Syncer.write_status", "mkdir"):
        "~/Desktop/Jarvis, exist_ok",
    ("jarvis/foldersync.py", "Syncer.write_status", "replace"):
        "status.txt.tmp -> status.txt, both names reserved to us",
    ("jarvis/foldersync.py", "Syncer.write_status", "write_text"):
        "our own status.txt.tmp",
    ("jarvis/foldersync.py", "_unlink_after_landing", "unlink"):
        "the source we JUST hard-linked; failure leaves it in both places",
    ("jarvis/foldersync.py", "_unlink_if_ours", "open"):
        "O_RDONLY -- a READ the census counts as a write, over-broadly",
    ("jarvis/foldersync.py", "_unlink_if_ours", "unlink"):
        "ROW 14: only a note carrying our marker; sub-ms TOCTOU, stated",
    ("jarvis/foldersync.py", "land_beside", "link"):
        "the atomic claim; EEXIST is the answer, not an error",
    ("jarvis/foldersync.py", "land_beside", "move"):
        "over OUR OWN 0-byte O_EXCL claim (no-hardlink filesystems only)",
    ("jarvis/foldersync.py", "land_beside", "open"):
        "O_CREAT|O_EXCL: the fallback claim, FAT/exFAT only",
    ("jarvis/foldersync.py", "land_beside", "replace"):
        "over OUR OWN 0-byte claim; measured p99 0.26 ms, nothing of his",
    ("jarvis/foldersync.py", "single_instance", "mkdir"):
        "the state dir, ours",
    ("jarvis/foldersync.py", "single_instance", "open"):
        "our own lock file, opened a+ and flocked",

    # -------------------------------------------------------- remote.py
    ("jarvis/tools/remote.py", "_list_sftp", "run_sftp"):
        "an ls; writes nothing",
    ("jarvis/tools/remote.py", "_remote_folder", "replace"):
        "str.replace on a config string -- a false positive, kept visible",
    ("jarvis/tools/remote.py", "pull", "open"):
        "O_CREAT|O_EXCL on the local destination; window ZERO",
    ("jarvis/tools/remote.py", "pull", "run_copy"):
        "over our own 0-byte claim, never over a file of his",
    ("jarvis/tools/remote.py", "pull", "unlink"):
        "our own claim and whatever half a file scp left in it",
    ("jarvis/tools/remote.py", "push", "run_copy"):
        "ROW 10: inside the staging directory sftp_mkdir just claimed",
    ("jarvis/tools/remote.py", "push", "sftp_mkdir"):
        "the exclusive create; refuses a file OR a directory at that name",
    ("jarvis/tools/remote.py", "push", "sftp_remove"):
        "our own staged file, inside our own directory",
    ("jarvis/tools/remote.py", "push", "sftp_rename"):
        "rename -l onto his name; refuses, never replaces",
    ("jarvis/tools/remote.py", "push", "sftp_rmdir"):
        "our own empty staging directory",
    ("jarvis/tools/remote.py", "sftp_mkdir", "run_sftp"):
        "guarded: refuses any name that is not a stage of our shape",
    ("jarvis/tools/remote.py", "sftp_remove", "run_sftp"):
        "guarded: our own shapes, or an explicit claimed=True widening",
    ("jarvis/tools/remote.py", "sftp_rename", "run_sftp"):
        "the claim itself",
    ("jarvis/tools/remote.py", "sftp_rmdir", "run_sftp"):
        "guarded: our own stage shape, and empty-only by the server",
}

# The four writes that put bytes somewhere a file could already be.  Each
# must be behind a CLAIM -- an operation that creates the name or refuses --
# and never behind a mere question.  Rows 3, 5, 10 and 12 of the round-3
# table, two of which that table did not have at all.
MUST_BE_CLAIMED = {
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "send"),
    ("jarvis/foldersync.py", "Syncer.pull_once", "fetch"),
    ("jarvis/tools/remote.py", "push", "run_copy"),
    ("jarvis/tools/remote.py", "pull", "run_copy"),
}


def test_every_write_in_this_lane_is_in_the_table():
    """A write nobody has written a line about fails here, not in his
    Outbox.  This is the pin the last three tables did not have."""
    rows = census()
    missing = [(m, f, w, g or "NOTHING") for m, f, w, g in rows
               if (m, f, w) not in KNOWN]
    assert not missing, (
        "a write appeared that the table does not name.  Add one line to "
        "KNOWN in tests/test_write_census.py saying what can be at that "
        "name when it happens:\n  " +
        "\n  ".join(f"{m}:{f} -> {w}  (guard: {g})" for m, f, w, g in missing))


def test_the_table_has_no_rows_that_are_no_longer_true():
    """The other direction: a row for a write that has gone is a table
    drifting away from the source, which is how the last two went wrong."""
    live = {(m, f, w) for m, f, w, _g in census()}
    stale = sorted(k for k in KNOWN if k not in live)
    assert not stale, f"KNOWN describes writes that no longer exist: {stale}"


def test_the_writes_that_can_land_on_a_file_are_all_behind_a_claim():
    """Not behind a QUESTION.  Ask-then-write is the shape that destroyed a
    513-byte notes.txt, a 22222-byte file, and two 99999-byte files across
    three rounds; a claim is one operation that cannot be raced."""
    guards = {(m, f, w): g for m, f, w, g in census()}
    for key in MUST_BE_CLAIMED:
        assert key in guards, f"{key} has gone; the pin is now meaningless"
        assert guards[key].startswith("claim:"), (
            f"{key[1]} writes behind '{guards[key] or 'NOTHING'}'. A check "
            f"is not a claim -- see round 3's rows 3 and 5.")


def test_the_table_prints():
    """`python -m tests.write_census` is how the table is read now, so it
    has to actually render."""
    text = table()
    assert "NOT IN THE TABLE" not in text
    assert "Syncer._send_and_claim" in text and "push" in text
    assert len(text.splitlines()) == len(census()) + 2
