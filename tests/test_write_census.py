"""The table, PINNED.  Add a write to this lane and this test fails.

tests/write_census.py walks the AST of jarvis/foldersync.py and
jarvis/tools/remote.py and lists every call that can create or destroy a
name, with whatever guards it in the same scope.  :data:`KNOWN` below is
the one line per WRITE CALL SITE that somebody has actually thought about.

WHY THIS EXISTS RATHER THAN A PARAGRAPH IN A DOCSTRING.  Three rounds of
this lane each shipped a hand-written version of this table.  Round 2's
said "nothing else in the repo has either shape" and was false.  Round 3's
had fourteen rows and was missing TWO of the most dangerous writes in the
module -- the push's own scp at its temp name (unbounded window, measured
destroying a 99999-byte file while reporting "sent") and the pull's scp
into its part file (the whole transfer, another file destroyed).  Both were
writes with NO GUARD AT ALL, which is precisely the row a person skims past
and a machine cannot.

WHY ROUND 5 REBUILT IT.  Round 4 shipped the machine version above and an
adversary stepped over it three ways in about a minute, which is worse than
having no census at all, because a number that cannot fail reads as
coverage:

  DEFEAT 1  rows were keyed (module, function, primitive) with ONE row per
            primitive per function, so a SECOND write of a listed primitive
            in a listed function was INVISIBLE.  MEASURED: an ask-then-write
            added to Syncer.record destroyed a 100000-byte file of his at
            ok.txt.log -- 100000 bytes to 0 -- with all four tests green and
            the count unchanged at 63.  That row's line ("our own history
            file... no name of his is involved") had silently become false.
            CLOSED by keying every CALL SITE: (module, scope, primitive, n).
  DEFEAT 2  WRITES had no word for rmtree, copytree, touch, renames,
            removedirs, os.write, writelines or mkstemp -- or for run_ssh,
            the one primitive in this lane that runs an arbitrary command on
            his Windows machine, a `del` included.  MEASURED: a _tidy_inbox
            doing rmtree, touch and a run_ssh `del` on his Windows Inbox
            censused as ZERO rows, four green tests, count still 63.
            CLOSED by putting run_ssh at the head of WRITES and adding the
            rest.
  DEFEAT 3  _functions() started at FunctionDef, so a module-level write --
            or a class-body one -- censused as nothing.  CLOSED: the module
            body and every class body are scopes now.

And one more the adversary asked to be considered: a row that DOWNGRADES
its guard, from an atomic claim to a mere question, used to pass everywhere
except the four rows in MUST_BE_CLAIMED.  The guard KIND is pinned on every
row now, so any downgrade anywhere fails.

WHY ROUND 6 INVERTED IT, and this is the part that matters most.  The
rebuilt census above was walked past SEVEN more ways, every one leaving the
count frozen at 75 -- a dead ignore entry (`raw.replace`) that measured
100000 bytes of his down to 4 while eleven census tests stayed green; a bare
`replace`; `self._my_stages.discard`; a module-level alias `_rm = os.remove`;
`getattr(os, "remove")(p)`; `OPS["rm"](p)`; `(a or b).replace(c)`.  Plus one
structural hole: the walk covered TWO modules and jarvis/tools/filepick.py
was imported AND CALLED by both of them and censused by neither.

Three rounds blocked on the same instrument is a design that cannot win, not
a fixer who keeps missing things.  Rounds 4 and 5 asked "is this one of the
writes I know?" and SKIPPED the rest, and Python has unboundedly many ways
to spell a callee.  Every one of the ten defeats is a SKIP.  So the DEFAULT
IS INVERTED: the census now reports every call it cannot PROVE harmless, and
an unresolvable callee is a row that has to be EXPLAINED rather than a
silence.  tests/write_census.py states the four rules and why a chokepoint
rewrite was judged the wrong answer to this particular question.
tests/test_census_fails_closed.py plants ten routes -- seven of which nobody
had listed -- and none of them needed a per-pattern entry.

What that costs is visible right here: KNOWN grew from 75 lines to 90,
because fifteen calls this lane always made are now SAID rather than
skipped.  Four of them are subprocess.Popen -- the seam every remote write
in this lane actually goes through, which the census had no word for.

If this test fails, do not edit KNOWN to make it green.  Read the failure:
it is telling you that a write appeared, or moved, or lost its guard.

THE ONE BLIND SPOT, stated rather than discovered later.  The ordinal is
positional: SWAPPING two calls of the same primitive inside one scope keeps
the same key set, so the two prose lines would silently trade places.  A
line number instead of an ordinal would catch that and would also churn the
whole table on every unrelated edit above it, which is how a pin gets edited
to green.  A swap that also changes either guard still fails on the kind.
"""
from tests.write_census import census, kind, table


# (module, scope, write primitive, which occurrence) -> (guard kind, what can
# be at that name, in a line).
KNOWN = {
    # -------------------- jarvis/foldersync.py
    # ---- ROWS THAT ARE NOT WRITES.  Round 6 inverted the census default
    # from SKIP to REPORT, so a call it cannot follow to a name is a row
    # that has to be EXPLAINED rather than a silence.  Every line below is
    # a call round 5 skipped -- three of them by a SPELLING-matched ignore
    # list, whose dead `raw.replace` entry let a 100000-byte file of his be
    # destroyed with the census green.
    ("jarvis/foldersync.py", "<module>", "?uncensused", 1):
        ("none",
         "get_logger('foldersync') into jarvis.logs, which this census "
         "does not walk; sets up a logger, names no file of his"),
    ("jarvis/tools/remote.py", "<module>", "?uncensused", 1):
        ("none",
         'the same get_logger; jarvis.logs is out of the walked set'),
    ("jarvis/tools/filepick.py", "<module>", "?uncensused", 1):
        ("none",
         'the same get_logger; jarvis.logs is out of the walked set'),
    ("jarvis/foldersync.py", "Syncer._close_stage", "discard", 1):
        ("none",
         "set.discard on self._my_stages -- OUR OWN bookkeeping, not the "
         "transport's discard.  Round 5 waved this through by dotted name "
         "and the same free pass covered any variable spelled that way"),
    ("jarvis/foldersync.py", "Syncer._remove_broken_copy", "discard", 1):
        ("none",
         'set.discard on self._claimed_here; bookkeeping, no name on disk'),
    ("jarvis/foldersync.py", "Syncer._sweep_my_stages", "discard", 1):
        ("none",
         'set.discard on self._my_stages; bookkeeping, no name on disk'),
    ("jarvis/foldersync.py", "Syncer._half", "?unresolved", 1):
        ("none",
         'fn(now): the INJECTED half of a pass -- push or pull, passed in'),
    ("jarvis/foldersync.py", "Syncer.loop", "?unresolved", 1):
        ("none",
         'sleep(...): the injected clock seam, so tests never really wait'),
    ("jarvis/foldersync.py", "is_quiescent", "?unresolved", 1):
        ("check",
         'now(): the injected clock seam; reads a number, writes nothing'),
    ("jarvis/foldersync.py", "is_quiescent", "?unresolved", 2):
        ("check",
         'sleep(...): the injected clock seam between two stat samples'),
    ("jarvis/tools/remote.py", "_cfg_get", "?unresolved", 1):
        ("none",
         "get(dotted, default) where get = getattr(cfg, 'get', None) -- a "
         "callable resolved at RUN TIME.  Exactly the shape the census "
         "cannot follow, and now it says so instead of skipping it"),
    ("jarvis/tools/remote.py", "run_ssh", "Popen", 1):
        ("none",
         "THE SEAM UNDER EVERYTHING: subprocess.Popen is what actually "
         "runs ssh.  Round 5 censused run_ssh and not the Popen beneath "
         "it; argv is built by ssh_argv, never by a shell string"),
    ("jarvis/tools/remote.py", "run_copy", "Popen", 1):
        ("none",
         'the scp process; argv from scp_argv, no shell'),
    ("jarvis/tools/remote.py", "run_sftp", "Popen", 1):
        ("none",
         'the sftp batch process; argv from sftp_argv, no shell'),
    ("jarvis/tools/remote.py", "tailnet_state", "Popen", 1):
        ("none",
         '`tailscale status --json`, a read-only local query'),

    # -------------------- jarvis/foldersync.py
    ("jarvis/foldersync.py", "Ledger.save", "mkdir", 1):
        ("none",
         '~/.local/state/jarvis, ours; exist_ok'),
    ("jarvis/foldersync.py", "Ledger.save", "open", 1):
        ("none",
         'our own .tmp beside the ledger, then the dir fd for the fsync'),
    ("jarvis/foldersync.py", "Ledger.save", "open", 2):
        ("none",
         "the DIRECTORY of the ledger, O_RDONLY for the fsync; writes "
         "no bytes"),
    ("jarvis/foldersync.py", "Ledger.save", "replace", 1):
        ("none",
         'tmp -> ledger, ours, one process holds the flock'),
    ("jarvis/foldersync.py", "Ledger.save", "write", 1):
        ("none",
         "fh.write into our own .tmp, on the fd just opened; nothing "
         "of his"),
    ("jarvis/foldersync.py", "SshTransport.claim", "sftp_rename", 1):
        ("claim",
         'CLAIM: rename -l, kernel refuses a held name; window ZERO'),
    ("jarvis/foldersync.py", "SshTransport.discard", "sftp_remove", 1):
        ("none",
         "only a temp of ours or a file inside a stage of ours; "
         "guarded twice"),
    ("jarvis/foldersync.py", "SshTransport.fetch", "run_copy", 1):
        ("none",
         "READS over there; writes at the local part file the caller "
         "claimed"),
    ("jarvis/foldersync.py", "SshTransport.listing", "run_sftp", 1):
        ("none",
         'an ls; writes nothing'),
    ("jarvis/foldersync.py", "SshTransport.remove_landed", "sftp_remove", 1):
        ("none",
         "THE GATED ONE: his filename, only with remove_broken_copies "
         "on, only a name this run took; logged at WARNING every time"),
    ("jarvis/foldersync.py", "SshTransport.send", "run_copy", 1):
        ("check",
         "inside a staging directory the caller took exclusively (row "
         "3); the stat before it is the local size, not a question "
         "about the name"),
    ("jarvis/foldersync.py", "SshTransport.stage_close", "sftp_rmdir", 1):
        ("none",
         'our stage only, and the server refuses a non-empty one'),
    ("jarvis/foldersync.py", "SshTransport.stage_open", "sftp_mkdir", 1):
        ("claim",
         'THE CLAIM for row 3: mkdir refuses any held name; window ZERO'),
    ("jarvis/foldersync.py", "SshTransport.stat", "run_sftp", 1):
        ("none",
         'an ls of ONE name; writes nothing (finding Q)'),
    ("jarvis/foldersync.py", "Syncer._claim_part", "open", 1):
        ("claim",
         "O_CREAT|O_EXCL in his Inbox: ROW 5's claim; window ZERO"),
    ("jarvis/foldersync.py", "Syncer._close_stage", "stage_close", 1):
        ("none",
         'the stage THIS pass opened'),
    ("jarvis/foldersync.py", "Syncer._discard", "discard", 1):
        ("none",
         'a name inside a stage of ours'),
    ("jarvis/foldersync.py", "Syncer._drop", "unlink", 1):
        ("none",
         'our own part file, claimed with O_EXCL a moment earlier'),
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "land_beside", 1):
        ("claim",
         "os.link into Sent; window ZERO, and a vanished source is his "
         "hand"),
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "land_beside", 2):
        ("claim",
         "the SECOND try, at <name>.<epoch>-<pid> in Sent; os.link, "
         "window ZERO"),
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "mkdir", 1):
        ("none",
         '~/Desktop/Jarvis/Sent, exist_ok'),
    ("jarvis/foldersync.py", "Syncer._open_stage", "stage_open", 1):
        ("claim",
         'the mkdir claim, once per pass'),
    ("jarvis/foldersync.py", "Syncer._push_once", "save", 1):
        ("check",
         'the ledger, ours'),
    ("jarvis/foldersync.py", "Syncer._remove_broken_copy", "remove_landed", 1):
        ("none",
         'OFF by default; identity is self._claimed_here, not a name shape'),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "save", 1):
        ("check",
         'the ledger, ours'),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "save", 2):
        ("check",
         'the ledger, ours -- after mark_landed'),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "save", 3):
        ("check",
         'the ledger, ours -- after the original moved to Sent'),
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "claim", 1):
        ("claim",
         'rename -l onto his name; refuses, never replaces'),
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "save", 1):
        ("claim",
         'the ledger, ours -- and it is BEFORE the claim on purpose (L)'),
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "save", 2):
        ("claim",
         "the ledger, ours -- clearing the record when the claim was "
         "refused"),
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "send", 1):
        ("claim",
         'ROW 3: inside the stage taken by _open_stage; window ZERO'),
    ("jarvis/foldersync.py", "Syncer._sweep_my_stages", "stage_close", 1):
        ("none",
         'only a stage name in self._my_stages (findings M and M2)'),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "save", 1):
        ("check",
         'the ledger, ours'),
    ("jarvis/foldersync.py", "Syncer._write_note", "fdopen", 1):
        ("claim",
         'ROW 14: the fd is the one whose marker was just read'),
    ("jarvis/foldersync.py", "Syncer._write_note", "open", 1):
        ("claim",
         "ROW 14: O_EXCL first; an existing note is opened WITHOUT "
         "truncating"),
    ("jarvis/foldersync.py", "Syncer._write_note", "open", 2):
        ("claim",
         'ROW 14: the NON-truncating O_RDWR reopen of an existing note'),
    ("jarvis/foldersync.py", "Syncer._write_note", "truncate", 1):
        ("claim",
         'ROW 14: on the checked fd, so never on a file of his'),
    ("jarvis/foldersync.py", "Syncer._write_note", "write", 1):
        ("claim",
         "ROW 14: on the fd whose marker was read through this same "
         "descriptor"),
    ("jarvis/foldersync.py", "Syncer.pull_once", "fetch", 1):
        ("claim",
         'ROW 5: into the part file _claim_part took with O_EXCL'),
    ("jarvis/foldersync.py", "Syncer.pull_once", "land_beside", 1):
        ("claim",
         'os.link into his Inbox; window ZERO'),
    ("jarvis/foldersync.py", "Syncer.pull_once", "mkdir", 1):
        ("check",
         '~/Desktop/Jarvis/Inbox, exist_ok'),
    ("jarvis/foldersync.py", "Syncer.pull_once", "save", 1):
        ("claim",
         'the ledger, ours'),
    ("jarvis/foldersync.py", "Syncer.record", "mkdir", 1):
        ("none",
         '~/.local/state/jarvis, ours'),
    ("jarvis/foldersync.py", "Syncer.record", "open", 1):
        ("none",
         'APPEND to our own history file'),
    ("jarvis/foldersync.py", "Syncer.record", "write", 1):
        ("none",
         'fh.write, append mode, our own history file; no name of his'),
    ("jarvis/foldersync.py", "Syncer.record", "write_text", 1):
        ("check",
         'our own history file, trimmed; no name of his is involved'),
    ("jarvis/foldersync.py", "Syncer.write_status", "mkdir", 1):
        ("none",
         '~/Desktop/Jarvis, exist_ok'),
    ("jarvis/foldersync.py", "Syncer.write_status", "replace", 1):
        ("none",
         'status.txt.tmp -> status.txt, both names reserved to us'),
    ("jarvis/foldersync.py", "Syncer.write_status", "write_text", 1):
        ("none",
         'our own status.txt.tmp'),
    ("jarvis/foldersync.py", "_unlink_after_landing", "unlink", 1):
        ("none",
         'the source we JUST hard-linked; failure leaves it in both places'),
    ("jarvis/foldersync.py", "_unlink_if_ours", "open", 1):
        ("none",
         'O_RDONLY -- a READ the census counts as a write, over-broadly'),
    ("jarvis/foldersync.py", "_unlink_if_ours", "unlink", 1):
        ("none",
         'ROW 14: only a note carrying our marker; sub-ms TOCTOU, stated'),
    ("jarvis/foldersync.py", "land_beside", "link", 1):
        ("claim",
         'the atomic claim; EEXIST is the answer, not an error'),
    ("jarvis/foldersync.py", "land_beside", "move", 1):
        ("claim",
         'over OUR OWN 0-byte O_EXCL claim (no-hardlink filesystems only)'),
    ("jarvis/foldersync.py", "land_beside", "open", 1):
        ("claim",
         'O_CREAT|O_EXCL: the fallback claim, FAT/exFAT only'),
    ("jarvis/foldersync.py", "land_beside", "replace", 1):
        ("claim",
         'over OUR OWN 0-byte claim; measured p99 0.26 ms, nothing of his'),
    ("jarvis/foldersync.py", "single_instance", "mkdir", 1):
        ("none",
         'the state dir, ours'),
    ("jarvis/foldersync.py", "single_instance", "open", 1):
        ("none",
         'our own lock file, opened a+ and flocked'),

    # -------------------- jarvis/tools/remote.py
    ("jarvis/tools/remote.py", "_list_sftp", "run_sftp", 1):
        ("none",
         'an ls; writes nothing'),
    ("jarvis/tools/remote.py", "_remote_folder", "replace", 1):
        ("none",
         'str.replace on a config string -- a false positive, kept visible'),
    ("jarvis/tools/remote.py", "ask", "run_ssh", 1):
        ("check",
         "ARBITRARY COMMAND: query_command's read-only probe "
         "(non-Windows path); the command is built from config, never "
         "from a filename of his"),
    ("jarvis/tools/remote.py", "list_remote", "run_ssh", 1):
        ("check",
         "ARBITRARY COMMAND: `ls -1p --` on the non-Windows path, "
         "folder shell-quoted by shell_path; a listing, and it writes "
         "nothing"),
    ("jarvis/tools/remote.py", "pull", "open", 1):
        ("claim",
         'O_CREAT|O_EXCL on the local destination; window ZERO'),
    ("jarvis/tools/remote.py", "pull", "run_copy", 1):
        ("claim",
         'over our own 0-byte claim, never over a file of his'),
    ("jarvis/tools/remote.py", "pull", "unlink", 1):
        ("claim",
         'our own claim and whatever half a file scp left in it'),
    ("jarvis/tools/remote.py", "push", "run_copy", 1):
        ("claim",
         'ROW 10: inside the staging directory sftp_mkdir just claimed'),
    ("jarvis/tools/remote.py", "push", "sftp_mkdir", 1):
        ("claim",
         'the exclusive create; refuses a file OR a directory at that name'),
    ("jarvis/tools/remote.py", "push", "sftp_remove", 1):
        ("claim",
         'our own staged file, inside our own directory'),
    ("jarvis/tools/remote.py", "push", "sftp_remove", 2):
        ("claim",
         "our own staged file again, after a REFUSED rename; inside "
         "our stage"),
    ("jarvis/tools/remote.py", "push", "sftp_rename", 1):
        ("claim",
         'rename -l onto his name; refuses, never replaces'),
    ("jarvis/tools/remote.py", "push", "sftp_rmdir", 1):
        ("claim",
         'our own empty staging directory'),
    ("jarvis/tools/remote.py", "sftp_mkdir", "run_sftp", 1):
        ("none",
         'guarded: refuses any name that is not a stage of our shape'),
    ("jarvis/tools/remote.py", "sftp_remove", "run_sftp", 1):
        ("none",
         'guarded: our own shapes, or an explicit claimed=True widening'),
    ("jarvis/tools/remote.py", "sftp_rename", "run_sftp", 1):
        ("none",
         'the claim itself'),
    ("jarvis/tools/remote.py", "sftp_rmdir", "run_sftp", 1):
        ("none",
         'guarded: our own stage shape, and empty-only by the server'),
}


# THE WRITES THAT PUT BYTES SOMEWHERE A FILE COULD ALREADY BE.  Each must be
# behind a CLAIM -- an operation that creates the name or refuses -- and never
# behind a mere question.  Rows 3, 5, 10 and 12 of the round-3 table, two of
# which that table did not have at all, plus the three round 5 added: the two
# land_beside calls that put a file into a folder of HIS, and the pull's own
# O_EXCL part file.
MUST_BE_CLAIMED = {
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "send", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "fetch", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "land_beside", 1),
    ("jarvis/foldersync.py", "Syncer._claim_part", "open", 1),
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "land_beside", 1),
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "land_beside", 2),
    ("jarvis/tools/remote.py", "push", "run_copy", 1),
    ("jarvis/tools/remote.py", "pull", "run_copy", 1),
}

# The primitive that is not like the others.  run_ssh runs a command line of
# our choosing on his Windows box, so it is not "a write at a name" -- it is
# every write there is.  Round 4's census had no word for it at all.
ARBITRARY_COMMAND = "run_ssh"


def test_every_write_in_this_lane_is_in_the_table():
    """A write nobody has written a line about fails here, not in his
    Outbox.  This is the pin the last three tables did not have -- and
    since round 5 it is per CALL SITE, so a second write in a function
    that already has one cannot hide behind the first (defeat 1)."""
    rows = census()
    missing = [(m, f, w, n, g or "NOTHING") for m, f, w, n, g in rows
               if (m, f, w, n) not in KNOWN]
    assert not missing, (
        "a write appeared that the table does not name.  Add one line to "
        "KNOWN in tests/test_write_census.py saying what can be at that "
        "name when it happens:\n  " +
        "\n  ".join(f"{m}:{f} -> {w} #{n}  (guard: {g})"
                     for m, f, w, n, g in missing))


def test_the_table_has_no_rows_that_are_no_longer_true():
    """The other direction: a row for a write that has gone is a table
    drifting away from the source, which is how the last two went wrong.
    Per call site, so REMOVING one of two identical writes fails too."""
    live = {(m, f, w, n) for m, f, w, n, _g in census()}
    stale = sorted(k for k in KNOWN if k not in live)
    assert not stale, f"KNOWN describes writes that no longer exist: {stale}"


def test_the_writes_that_can_land_on_a_file_are_all_behind_a_claim():
    """Not behind a QUESTION.  Ask-then-write is the shape that destroyed a
    513-byte notes.txt, a 22222-byte file, and two 99999-byte files across
    three rounds; a claim is one operation that cannot be raced."""
    guards = {(m, f, w, n): g for m, f, w, n, g in census()}
    for key in MUST_BE_CLAIMED:
        assert key in guards, f"{key} has gone; the pin is now meaningless"
        assert guards[key].startswith("claim:"), (
            f"{key[1]} writes behind '{guards[key] or 'NOTHING'}'. A check "
            f"is not a claim -- see round 3's rows 3 and 5.")


def test_no_row_may_quietly_downgrade_its_guard():
    """The adversary's fifth observation, and it is the cheap one to miss.
    A row going from ``claim:`` to ``check:`` -- somebody replacing an
    exclusive create with an ``if not exists`` -- used to pass everywhere
    except the four rows in MUST_BE_CLAIMED.  The guard KIND is pinned on
    every row now, so a downgrade anywhere fails, and so does a change in
    the other direction, because a table that understates the guard is a
    table nobody is reading.
    """
    changed = []
    for mod, fn, w, n, guard in census():
        row = KNOWN.get((mod, fn, w, n))
        if row and kind(guard) != row[0]:
            changed.append(f"{mod}:{fn} -> {w} #{n}: pinned {row[0]!r}, "
                           f"source now {kind(guard)!r} ({guard or 'NOTHING'})")
    assert not changed, (
        "a write's guard changed.  If it got WEAKER this is the bug; if it "
        "got stronger, update the kind in KNOWN and say why:\n  " +
        "\n  ".join(changed))


def test_the_census_has_a_word_for_running_a_command_on_his_machine():
    """DEFEAT 2, pinned as a fact about the census rather than about the
    source.  run_ssh is the one primitive here that can do anything at all
    on HPCOMPUTER -- del, rmdir /s, a redirect over a file of his -- and it
    was not in WRITES.  A future edit that drops it puts the whole class of
    remote destruction back out of sight."""
    from tests.write_census import WRITES
    assert ARBITRARY_COMMAND in WRITES, (
        "run_ssh is not censused.  A method that runs `del` on his Windows "
        "Inbox will show up as ZERO rows again.")
    for word in ("rmtree", "copytree", "touch", "renames", "removedirs",
                 "write", "writelines", "mkstemp"):
        assert word in WRITES, f"{word} has no row shape; defeat 2 is open"


def test_the_census_sees_a_write_that_is_not_inside_a_function():
    """DEFEAT 3.  A module-level write, or one in a class body, ran at
    import and censused as nothing."""
    from tests.write_census import census as run
    src = ("import os\n"
           "from pathlib import Path\n"
           "Path('/tmp/x').write_text('boot')\n"
           "class C:\n"
           "    os.mkdir('/tmp/y')\n")
    rows = run(paths=("m.py",), sources={"m.py": src})
    where = {(f, w) for _m, f, w, _n, _g in rows}
    assert ("<module>", "write_text") in where, rows
    assert ("C.<class body>", "mkdir") in where, rows


def test_the_table_prints():
    """`python -m tests.write_census` is how the table is read now, so it
    has to actually render."""
    text = table()
    assert "NOT IN THE TABLE" not in text
    assert "Syncer._send_and_claim" in text and "push" in text
    assert len(text.splitlines()) == len(census()) + 2
