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
    # THE CHOKEPOINT for a file of OURS, round 7.  These rows used to be
    # TEN: the same mkdir + temp + replace dance written out once in
    # Ledger.save (with an fsync) and once in Syncer.write_status (without
    # one).  Both go through _replace_ours now, and the fsync is a named
    # argument rather than a thing one copy remembered.  See
    # tests/test_write_chokepoint.py, which pins the audited-writer set --
    # and which records that this consolidation removed FOUR of 487 rows.
    #
    # ROUND 10: THE TEMP IS CLAIMED.  Through round 9 the temp was a fixed
    # "<name>.tmp" opened with a plain open(tmp, "w"), and every row below
    # honestly said NOTHING.  MEASURED on 1039ce8: a 100000-byte file of his
    # at ~/Desktop/Jarvis/status.txt.tmp was GONE after one pass.  The temp
    # is O_CREAT|O_EXCL at "<name>.<pid>-<n>.tmp" now, so the open row
    # earns claim:excl-open and every row after it in the scope inherits
    # that guard -- five kinds went none -> claim (stronger, and this is the
    # why), and fdopen on the claimed descriptor is a new sixth row.
    ("jarvis/foldersync.py", "_replace_ours", "mkdir", 1):
        ("none",
         "the parent of a file of OURS -- ~/.local/state/jarvis for the "
         "ledger, ~/Desktop/Jarvis for status.txt; exist_ok"),
    ("jarvis/foldersync.py", "_replace_ours", "open", 1):
        ("claim",
         'CLAIM: O_CREAT|O_EXCL at our own "<name>.<pid>-<n>.tmp" beside the '
         "target, in the same directory so the rename that follows is "
         "atomic; a taken name is refused by the kernel and the next tried"),
    ("jarvis/foldersync.py", "_replace_ours", "fdopen", 1):
        ("claim",
         "a text handle over the descriptor the O_EXCL claim just returned; "
         "opens no name of its own"),
    ("jarvis/foldersync.py", "_replace_ours", "open", 2):
        ("claim",
         "the DIRECTORY of the target, O_RDONLY for the fsync; writes no "
         "bytes and only happens when fsync=True"),
    ("jarvis/foldersync.py", "_replace_ours", "replace", 1):
        ("claim",
         "our claimed tmp -> the target.  A file of OURS only: the TARGET "
         "is not claimed, on purpose, and a plain replace at a name HE "
         "chose is what destroyed a 100000-byte file in round 5.  That case "
         "is land_beside"),
    ("jarvis/foldersync.py", "_replace_ours", "write", 1):
        ("claim",
         "fh.write into our own claimed tmp, on the fd just opened; nothing "
         "of his can be at that name"),
    # THE LANE'S OWN DESTROYERS, round 10: a caller of _replace_ours, _drop
    # or _unlink_after_landing is a row now, so a NEW caller fails the pin
    # here before it fails a product test.
    ("jarvis/foldersync.py", "Ledger.save", "_replace_ours", 1):
        ("none",
         "the ledger, a file of OURS at ~/.local/state/jarvis, fsync=True; "
         "the target needs no claim because nothing of his lives there"),
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
    ("jarvis/foldersync.py", "Syncer.pull_once", "_drop", 1):
        ("claim",
         "the part file _claim_part took with O_EXCL, after a fetch that "
         "failed; OUR name, pid + counter, never his"),
    ("jarvis/foldersync.py", "Syncer.pull_once", "_drop", 2):
        ("claim",
         "the same part file, after a size that did not verify; ours"),
    ("jarvis/foldersync.py", "Syncer.pull_once", "_drop", 3):
        ("claim",
         "the same part file, when land_beside raised (denied); ours"),
    ("jarvis/foldersync.py", "Syncer.pull_once", "_drop", 4):
        ("claim",
         "the same part file, when land_beside ran out of names; ours"),
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
    ("jarvis/foldersync.py", "Syncer.write_status", "_replace_ours", 1):
        ("none",
         "status.txt, a file of OURS in ~/Desktop/Jarvis, no fsync; the "
         "target needs no claim, and the temp beside it is claimed inside"),
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
    ("jarvis/foldersync.py", "main", "?reach", 1):
        ("none",
         "argparse.ArgumentParser for the command-line entry point.  A row "
         "because argparse came OUT of HARMLESS_MODULES in round 7: "
         'argparse.FileType("w") opens a path for writing, so the module '
         "can name a file and cannot be vouched wholesale"),
    ("jarvis/foldersync.py", "main", "?uncensused", 1):
        ("check",
         "AssistantConfig.load() -- reads ~/.config/jarvis/assistant.json "
         "through the redacting loader, from a module this census does not "
         "walk.  Newly VISIBLE in round 7: the lazy `from "
         "jarvis.assistant_config import AssistantConfig` inside main() "
         "resolves now, where before the name was untypable and the call "
         "was silence"),
    ("jarvis/foldersync.py", "_unlink_after_landing", "unlink", 1):
        ("none",
         'the source we JUST hard-linked; failure leaves it in both places'),
    ("jarvis/foldersync.py", "_unlink_if_ours", "open", 1):
        ("none",
         'O_RDONLY -- a READ the census counts as a write, over-broadly'),
    ("jarvis/foldersync.py", "_unlink_if_ours", "unlink", 1):
        ("none",
         'ROW 14: only a note carrying our marker; sub-ms TOCTOU, stated'),
    ("jarvis/foldersync.py", "land_beside", "_unlink_after_landing", 1):
        ("claim",
         "the SOURCE, only after os.link has already put the same inode at "
         "the claimed dest; failure leaves it in both places"),
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

    # ---- ROUND 8: REACHING THROUGH A VOUCHED NAMESPACE (?through:)
    # Rule S used to bless any dotted name starting with a walked module's
    # path, and any name whose ROOT was on HARMLESS_MODULES.  Both are
    # prefix matches, and a prefix says nothing about what is behind it: a
    # module re-exports every module it imports, so
    # `remote.subprocess.getoutput("rm -f ...")` was NOT A ROW.  MEASURED
    # destroying a 100000-byte file of his with the census frozen at 483.
    # Both rules now stop at ONE segment, and the walked one additionally
    # requires that the segment be a real def or class of that module.
    #
    # THE SEVEN ROWS BELOW ARE THE PRICE, and they are worth reading,
    # because they are the ATTACK'S OWN SHAPE sitting in the lane's real
    # code: a module-level NAME of remote.py that is not a def and not a
    # class, called through the walked prefix.  Both are compiled regexes
    # and both are safe -- but this census cannot prove that from the
    # source, which is exactly what it now says instead of guessing.  If
    # either ever stops being a regex, these lines go false and a person
    # has to look, which is the only guarantee on offer.
    ("jarvis/foldersync.py", "windows_name_problem", "?through:match", 1):
        ("none",
         "remote.SAFE_REMOTE_NAME_RX.match -- a module-level name of "
         "remote.py, not a def or a class, so this census cannot type it; "
         "it is a compiled regex and .match only ASKS about a string"),
    ("jarvis/foldersync.py", "remote_folder_problems", "?through:search", 1):
        ("none",
         "remote._SFTP_UNQUOTABLE_RX.search -- same shape, same module-"
         "level regex; a question about a name, never a write"),
    ("jarvis/foldersync.py", "SshTransport.listing", "?through:search", 1):
        ("none",
         "remote._SFTP_UNQUOTABLE_RX.search on the path before a listing"),
    ("jarvis/foldersync.py", "SshTransport.stat", "?through:match", 1):
        ("none",
         "remote.SAFE_REMOTE_NAME_RX.match on the name before a stat"),
    ("jarvis/foldersync.py", "SshTransport.stat", "?through:search", 1):
        ("none",
         "remote._SFTP_UNQUOTABLE_RX.search on the path before a stat"),
    ("jarvis/foldersync.py", "SshTransport.fetch", "?through:match", 1):
        ("none",
         "remote.SAFE_REMOTE_NAME_RX.match on the name before a fetch -- "
         "the refusal that keeps an odd remote name out of an argv"),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?through:match", 1):
        ("check",
         "remote.SAFE_REMOTE_NAME_RX.match on each listed name; the one "
         "that decides a file is unsafe to bring across at all"),
}


# ---------------------------------------------------------------- rule T
# THE 394 ROWS ROUND 7 BOUGHT, and the prose that makes them readable.
#
# Round 6 inverted the default for the CALLEE.  Round 7 inverted it for the
# RECEIVER: a call on an object this file cannot type is a row now, whatever
# the method is called, because judging it by its NAME is the same losing
# race one level down -- the author of the attack picks the name.  That is
# what let a 100000-byte file of his be destroyed twice over with the census
# frozen at 90/90 (logging.FileHandler, subprocess.getoutput), and it is what
# hardlink_to, symlink_to and lchmod walked through needing no trick at all.
#
# THE PRICE, MEASURED: 394 more rows, 65 distinct method names.
#
# HOW THE PROSE IS SPLIT, AND WHY IT IS NOT 394 SENTENCES.  The brief for
# this round costed rule T at "one human sentence in KNOWN" per row.  Writing
# 394 of those would produce 394 sentences that nobody wrote and nobody
# reads -- and round 6 named that exact failure itself: "if that table ever
# starts getting edited to green, the instrument is dead again regardless of
# its shape."  So the pin and the prose are split, and NOTHING IS SILENCED
# by the split:
#
#   UNTYPED_SITES  every call site, pinned.  A new untypable call anywhere
#                  fails the suite, exactly as a new mutation row does.
#                  It is data, it is generated from the source, and it is
#                  meant to be regenerated rather than reasoned about.
#   UNTYPED        one sentence per METHOD NAME, 65 of them.  This is where
#                  the thinking lives, and it is the table a person reads.
#                  A NEW NAME cannot be added without writing a sentence
#                  that says what an object with that method is and why it
#                  cannot name a file of his -- and for `shred`, `lchmod`,
#                  `FileHandler` or `getoutput` no such sentence exists.
#
# WHAT THE SPLIT COSTS, said out loud: a SECOND call site of a name already
# in this table adds a line to UNTYPED_SITES and no sentence.  That is a
# real blind spot and it is the same one the per-site version would have --
# `out = []` becoming `out = EvilThing()` keeps the key, the name and the
# row, and no amount of prose about call sites notices.  What bounds it is
# unchanged and is stated in write_census.py: such an object comes either
# from a censused module, whose class methods are walked as scopes so the
# write surfaces there, or from an uncensused one, whose constructing call
# is already a rule-R row.
UNTYPED = {
    # ---- containers and strings.  The receiver is a local built from a
    # literal or a comprehension a few lines up.  Rule M runs BEFORE rule T,
    # so a container method that shares a name with a filesystem verb --
    # list.remove, dict.pop's cousin os.remove, str.replace -- is still a
    # mutation row and is not in this table.  That ordering is deliberate:
    # it is what stopped `raw.replace(dest)`.
    "append": "list.append onto a local accumulator (`lines`, `events`, "
              "`out`, `names`, `rows`, `skipped`), each built from a literal "
              "in the same scope; appends to a list in memory",
    "get": "mapping lookup with a default -- ledger dicts, a parsed JSON "
           "row, a tailscale peer, a history row; reads, never writes",
    "items": "dict.items over the ledger's own three maps and over parsed "
             "JSON; an iteration, no name on disk",
    "pop": "dict.pop / list.pop off the ledger's maps, the config-problem "
           "list and a walk stack; removes an entry from memory",
    "add": "set.add into `taken`, `seen`, `_my_stages`, `_claimed_here` -- "
           "OUR OWN bookkeeping of which names we hold, not a name on disk",
    "clear": "dict.clear / list.clear of the ledger's landed map and the "
             "config-problem list; memory only",
    "update": "dict.update merging fields into a ledger row in memory",
    "values": "dict.values over the tailscale peer map",
    "sort": "list.sort on `scored`, the candidate ranking; memory only",
    "strip": "str.strip on a configured folder, a spoken phrase or a "
             "transport line; returns a new string",
    "rstrip": "str.rstrip trimming a trailing slash off a remote folder or "
              "trailing spaces off a printed line; returns a new string",
    "lstrip": "str.lstrip on the remainder of an sftp listing line",
    "split": "str.split on a listing line, a hostname, a filename stem or "
             "this module's own __doc__; returns a list",
    "rsplit": "str.rsplit splitting an sftp long-listing line from the "
              "right, which is how the name is separated from the fields",
    "splitlines": "str.splitlines over transport output or the text just "
                  "read back from our own history file",
    "partition": "str.partition splitting a name at its first dot",
    "rpartition": "str.rpartition splitting a name at its LAST dot, which "
                  "is how the extension is found",
    "join": 'str.join on a literal separator ("\\n", ", ", " ") -- the '
            "receiver is the separator itself, so this can only be str.join "
            "and never os.path.join",
    "startswith": "str.startswith testing a name, a listing line or a "
                  "spoken phrase for a prefix",
    "endswith": "str.endswith testing a name for a suffix (our part files, "
                "the note suffix, a tailnet domain)",
    "lower": "str.lower normalising a hostname, a stem or a spoken phrase",
    "encode": "str.encode of NOTE_MARKER, our own constant, to compare "
              "against the first bytes read back from a note",
    "format": "str.format filling a fixed template line for the printout",
    "groups": "re.Match.groups pulling fields out of a matched line",

    # ---- compiled regexes.  Module-level constants built by re.compile,
    # which is the one fully-qualified `compile` spelling in PROVEN_SAFE.
    "search": "Pattern.search on a module-level regex constant "
              "(_SFTP_UNQUOTABLE_RX, _AUTH_RX, _DENIED_RX, _REACH_RX, "
              "_MISSING_RX, _MISSING_CMD_RX); reads a string",
    "match": "Pattern.match on SAFE_REMOTE_NAME_RX or _TAILNET_IP_RX -- the "
             "gate that decides a name is safe to send, and it only reads",
    "ratio": "difflib.SequenceMatcher.ratio scoring how close two names "
             "are, for the spoken 'send the invoice' pick; a number",

    # ---- the logger.  `log = get_logger(...)` binds the RESULT OF A CALL
    # into a module the census does not walk, so the census cannot prove
    # what `log` is -- and round 7 stopped pretending it could.  This is
    # exactly the shape of ROUTE A: `logging` was vouched wholesale until
    # logging.FileHandler(path, mode="w") was measured truncating a file.
    # These four rows are honest and they stay.
    "warning": "log.warning -- jarvis.logs.get_logger's return value, which "
               "this census cannot type.  A logger writes to the log file, "
               "never to a name of his",
    "info": "log.info, same logger, same reasoning",
    "debug": "log.debug, same logger, same reasoning",
    "exception": "log.exception, same logger; writes a traceback to the log",

    # ---- Path objects.  NOT vouched as a type, deliberately: pathlib is
    # where hardlink_to, symlink_to and lchmod live, and those were ROUTE C
    # -- three real methods that create or change a name and are not in the
    # mutator vocabulary.  Every Path method that reaches this table is one
    # that READS, and each is named individually for that reason.
    "resolve": "Path.resolve -- the real path with symlinks followed. The "
               "containment check in this lane is built on it; it reads",
    "relative_to": "Path.relative_to, pure string arithmetic on two already "
                   "resolved paths; raises if outside, writes nothing",
    "with_name": "Path.with_name building the '<name>.<pid>-<n>.tmp' "
                 "sibling inside _replace_ours; names a path, creates "
                 "nothing",
    "stat": "Path.stat / the transport's stat -- size and mtime, the "
            "quiescence question. A read, and the row it guards is pinned "
            "as a check rather than a claim",
    "exists": "Path.exists on a candidate base name; a question, and the "
              "lane's rule is that a question is never a guard for a write "
              "at a name of his",
    "is_dir": "Path.is_dir / DirEntry.is_dir filtering a listing",
    "is_file": "DirEntry.is_file filtering a listing",
    "iterdir": "Path.iterdir listing his Outbox and his Inbox; reads names, "
               "creates none",
    "read_text": "Path.read_text of OUR OWN files -- the ledger and the "
                 "history file. Never of anything of his",

    # ---- open file objects.  The `open` that produced them is itself a
    # row with its own line in KNOWN; these are operations on a descriptor
    # we already hold.
    "flush": "file.flush before the fsync in _replace_ours",
    "fileno": "file.fileno for os.fsync, and for the O_EXCL note descriptor",
    "seek": "file.seek back to 0 to read the marker through THE SAME "
            "descriptor the truncate happens on -- which is the whole "
            "point of that row and is why it is not two opens",
    "close": "file.close of a descriptor we opened",

    # ---- subprocess.Popen results.  The Popen call is a row of its own in
    # every case, with the argv it was given written out beside it.
    "communicate": "Popen.communicate collecting stdout/stderr from ssh, "
                   "sftp, scp or tailscale; reads two pipes",
    "kill": "Popen.kill on OUR OWN child after a timeout -- ends a process "
            "we started, touches no name",

    # ---- this lane's own objects, reached through an injected seam.  The
    # ledger and the transport are passed in, so the census cannot type
    # them -- and that is the honest reason these are rows.  Every one of
    # these names IS a method of a class in a censused module, whose body is
    # walked as its own scope, so the write inside it surfaces there: that
    # is the bound on this whole group, and it is why a new transport method
    # (ROUTE J, `self.transport.wipe_folder`) still cannot be silent.
    "bump": "Ledger.bump, the fail counter; memory plus a later save",
    "has": "Ledger.has, a membership question",
    "blocked": "Ledger.blocked, a question about the fail counter",
    "parked": "Ledger.parked, the same question as blocked asked so the "
              "answer can be SAID: when the hour is up, and what stopped it. "
              "Read-only -- the sweep of an expired row stays in blocked",
    "landed_row": "Ledger.landed_row, a lookup",
    "mark_landed": "Ledger.mark_landed, an entry in the landing record. "
                   "The ORDER matters and is the duplicate-send guarantee; "
                   "the fsync'd write is Ledger.save, which is its own row",
    "mark_pulled": "Ledger.mark_pulled, the same shape",
    "mark_claiming": "Ledger.mark_claiming, the same shape; it is also in "
                     "CLAIMS, so a write after it reads as claim-guarded",
    "clear_landed": "Ledger.clear_landed, forgetting a landing record",
    "sweep": "Ledger.sweep, refreshing what is still on the far side",
    "listing": "SshTransport.listing -- one sftp `ls`. A READ of the far "
               "side, and it is in READS for that reason",
    "row": "Event.row, the dataclass as a dict for the history line",
    "status_clock": "the injected clock seam on Syncer; returns a number",
    "is_set": "threading.Event.is_set, the stop flag in the loop",
    "loop": "Syncer.loop from main(); a method of this module, walked as "
            "its own scope",
    "run_pass": "Syncer.run_pass from main(), same",
    "check_far_side": "Syncer.check_far_side from main(), same",
    "status_text": "Syncer.status_text from main(); builds a string",

    # ---- argparse.  Also OUT of HARMLESS_MODULES in round 7, because
    # argparse.FileType("w") opens a path for writing.  The parser itself is
    # a rule-R row in main(); these two are its methods.
    "add_argument": "ArgumentParser.add_argument declaring a flag",
    "parse_args": "ArgumentParser.parse_args reading sys.argv",
}

# The 394 call sites those 65 names appear at, generated from the source
# and meant to be regenerated rather than reasoned about.  It is the PIN:
# a new untypable call anywhere fails the suite.  The thinking is in
# UNTYPED above; this is data.
UNTYPED_SITES = {
    # ---------------------------------------- jarvis/foldersync.py
    ("jarvis/foldersync.py", "Ledger._trim", "?untyped:items", 1),
    ("jarvis/foldersync.py", "Ledger._trim", "?untyped:items", 2),
    ("jarvis/foldersync.py", "Ledger._trim", "?untyped:items", 3),
    ("jarvis/foldersync.py", "Ledger._trim.<lambda> #1", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger._trim.<lambda> #2", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger._trim.<lambda> #3", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.blocked", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.blocked", "?untyped:get", 2),
    ("jarvis/foldersync.py", "Ledger.blocked", "?untyped:pop", 1),
    ("jarvis/foldersync.py", "Ledger.bump", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.bump", "?untyped:get", 2),
    ("jarvis/foldersync.py", "Ledger.clear", "?untyped:pop", 1),
    # ROUND 8.  Ledger.parked reads the deadline and the reason out of
    # the fail row so the two skip sites can SAY what they are skipping;
    # blocked keeps the decision and the sweep, and gave up these reads.
    ("jarvis/foldersync.py", "Ledger.parked", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.parked", "?untyped:get", 2),
    ("jarvis/foldersync.py", "Ledger.parked", "?untyped:get", 3),
    ("jarvis/foldersync.py", "Ledger.parked", "?untyped:get", 4),
    ("jarvis/foldersync.py", "Ledger.clear_landed", "?untyped:pop", 1),
    ("jarvis/foldersync.py", "Ledger.landed_row", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.landed_row", "?untyped:get", 2),
    ("jarvis/foldersync.py", "Ledger.landed_row", "?untyped:pop", 1),
    ("jarvis/foldersync.py", "Ledger.load", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.load", "?untyped:get", 2),
    ("jarvis/foldersync.py", "Ledger.load", "?untyped:get", 3),
    ("jarvis/foldersync.py", "Ledger.load", "?untyped:get", 4),
    ("jarvis/foldersync.py", "Ledger.load", "?untyped:items", 1),
    ("jarvis/foldersync.py", "Ledger.load", "?untyped:items", 2),
    ("jarvis/foldersync.py", "Ledger.load", "?untyped:items", 3),
    ("jarvis/foldersync.py", "Ledger.load", "?untyped:read_text", 1),
    ("jarvis/foldersync.py", "Ledger.mark_landed", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.mark_landed", "?untyped:get", 2),
    ("jarvis/foldersync.py", "Ledger.mark_pulled", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.mark_pulled", "?untyped:update", 1),
    ("jarvis/foldersync.py", "Ledger.save", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Ledger.sweep", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Ledger.sweep", "?untyped:get", 2),
    ("jarvis/foldersync.py", "Ledger.sweep", "?untyped:items", 1),
    ("jarvis/foldersync.py", "Ledger.sweep", "?untyped:pop", 1),
    ("jarvis/foldersync.py", "SshTransport.fetch", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "SshTransport.listing", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "SshTransport.send", "?untyped:stat", 1),
    ("jarvis/foldersync.py", "SshTransport.stat", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "SshTransport.target", "?untyped:partition", 1),
    ("jarvis/foldersync.py", "Syncer._candidates", "?untyped:append", 1),
    ("jarvis/foldersync.py", "Syncer._candidates", "?untyped:append", 2),
    ("jarvis/foldersync.py", "Syncer._candidates", "?untyped:endswith", 1),
    ("jarvis/foldersync.py", "Syncer._candidates", "?untyped:endswith", 2),
    ("jarvis/foldersync.py", "Syncer._candidates", "?untyped:iterdir", 1),
    ("jarvis/foldersync.py", "Syncer._candidates", "?untyped:startswith", 1),
    ("jarvis/foldersync.py", "Syncer._claim_part", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Syncer._close_stage", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer._config_clear", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer._config_clear", "?untyped:pop", 1),
    ("jarvis/foldersync.py", "Syncer._config_event", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Syncer._config_event", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Syncer._discard", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer._half", "?untyped:exception", 1),
    ("jarvis/foldersync.py", "Syncer._link_event", "?untyped:clear", 1),
    ("jarvis/foldersync.py", "Syncer._link_event", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Syncer._mark_up", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Syncer._move_to_sent", "?untyped:warning", 2),
    ("jarvis/foldersync.py", "Syncer._open_stage", "?untyped:add", 1),
    ("jarvis/foldersync.py", "Syncer._probe_link", "?untyped:listing", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:add", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 2),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 3),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 4),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 5),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 6),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 7),
    # ROUND 8: the parked note and the parked event, outbound.
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 8),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:append", 9),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:blocked", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:bump", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:debug", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:parked", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:landed_row", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:listing", 1),
    ("jarvis/foldersync.py", "Syncer._push_once", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Syncer._remove_broken_copy", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Syncer._remove_broken_copy", "?untyped:warning", 2),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "?untyped:clear", 1),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "?untyped:clear_landed", 1),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "?untyped:clear_landed", 2),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "?untyped:get", 2),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "?untyped:mark_landed", 1),
    ("jarvis/foldersync.py", "Syncer._resolve_landing", "?untyped:stat", 1),
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "?untyped:add", 1),
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "?untyped:clear_landed", 1),
    ("jarvis/foldersync.py", "Syncer._send_and_claim", "?untyped:mark_claiming", 1),
    ("jarvis/foldersync.py", "Syncer._size", "?untyped:stat", 1),
    ("jarvis/foldersync.py", "Syncer._sweep_my_stages", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer._sweep_notes", "?untyped:endswith", 1),
    ("jarvis/foldersync.py", "Syncer._sweep_notes", "?untyped:exists", 1),
    ("jarvis/foldersync.py", "Syncer._sweep_notes", "?untyped:iterdir", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:append", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:append", 2),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:append", 3),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:append", 4),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:append", 5),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:bump", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:clear", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:clear_landed", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:clear_landed", 2),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:listing", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:mark_landed", 1),
    ("jarvis/foldersync.py", "Syncer._verify_and_move", "?untyped:stat", 1),
    ("jarvis/foldersync.py", "Syncer._write_note", "?untyped:debug", 1),
    ("jarvis/foldersync.py", "Syncer._write_note", "?untyped:debug", 2),
    ("jarvis/foldersync.py", "Syncer._write_note", "?untyped:debug", 3),
    ("jarvis/foldersync.py", "Syncer._write_note", "?untyped:seek", 1),
    ("jarvis/foldersync.py", "Syncer._write_note", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Syncer.check_far_side", "?untyped:append", 1),
    ("jarvis/foldersync.py", "Syncer.check_far_side", "?untyped:append", 2),
    ("jarvis/foldersync.py", "Syncer.check_far_side", "?untyped:append", 3),
    ("jarvis/foldersync.py", "Syncer.check_far_side", "?untyped:listing", 1),
    ("jarvis/foldersync.py", "Syncer.loop", "?untyped:exception", 1),
    ("jarvis/foldersync.py", "Syncer.loop", "?untyped:is_set", 1),
    ("jarvis/foldersync.py", "Syncer.note", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Syncer.note", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:add", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 2),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 3),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 4),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 5),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 6),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 7),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 8),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 9),
    # ROUND 8: the parked note and the parked event, inbound -- the one
    # he can see sitting in the HPCOMPUTER outbox for the whole hour.
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 10),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:append", 11),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:blocked", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:bump", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:bump", 2),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:bump", 3),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:clear", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:has", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:parked", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:info", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:info", 2),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:info", 3),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:iterdir", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:listing", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:mark_pulled", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:sweep", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:warning", 2),
    ("jarvis/foldersync.py", "Syncer.pull_once", "?untyped:warning", 3),
    ("jarvis/foldersync.py", "Syncer.record", "?untyped:debug", 1),
    ("jarvis/foldersync.py", "Syncer.record", "?untyped:join", 1),
    ("jarvis/foldersync.py", "Syncer.record", "?untyped:read_text", 1),
    ("jarvis/foldersync.py", "Syncer.record", "?untyped:row", 1),
    ("jarvis/foldersync.py", "Syncer.record", "?untyped:splitlines", 1),
    ("jarvis/foldersync.py", "Syncer.run_pass", "?untyped:exception", 1),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 1),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 2),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 3),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 4),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 5),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 6),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 7),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 8),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 9),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 10),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 11),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 12),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 13),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 14),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 15),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 16),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 17),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 18),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 19),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 20),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 21),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 22),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 23),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 24),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 25),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 26),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 27),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 28),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 29),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 30),
    # ROUND 8: the three lines of the parked note itself.
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 31),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 32),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:append", 33),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:get", 1),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:join", 1),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "Syncer.status_text", "?untyped:status_clock", 1),
    ("jarvis/foldersync.py", "Syncer.write_status", "?untyped:debug", 1),
    ("jarvis/foldersync.py", "_inside", "?untyped:relative_to", 1),
    ("jarvis/foldersync.py", "_inside", "?untyped:resolve", 1),
    ("jarvis/foldersync.py", "_marked", "?untyped:encode", 1),
    ("jarvis/foldersync.py", "_marked", "?untyped:encode", 2),
    ("jarvis/foldersync.py", "_ours_to_move", "?untyped:partition", 1),
    ("jarvis/foldersync.py", "_replace_ours", "?untyped:fileno", 1),
    ("jarvis/foldersync.py", "_replace_ours", "?untyped:flush", 1),
    ("jarvis/foldersync.py", "_replace_ours", "?untyped:with_name", 1),
    ("jarvis/foldersync.py", "_unlink_after_landing", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "_unlink_if_ours", "?untyped:info", 1),
    ("jarvis/foldersync.py", "is_quiescent", "?untyped:stat", 1),
    ("jarvis/foldersync.py", "main", "?untyped:add_argument", 1),
    ("jarvis/foldersync.py", "main", "?untyped:add_argument", 2),
    ("jarvis/foldersync.py", "main", "?untyped:add_argument", 3),
    ("jarvis/foldersync.py", "main", "?untyped:check_far_side", 1),
    ("jarvis/foldersync.py", "main", "?untyped:get", 1),
    ("jarvis/foldersync.py", "main", "?untyped:get", 2),
    ("jarvis/foldersync.py", "main", "?untyped:get", 3),
    ("jarvis/foldersync.py", "main", "?untyped:get", 4),
    ("jarvis/foldersync.py", "main", "?untyped:get", 5),
    ("jarvis/foldersync.py", "main", "?untyped:info", 1),
    ("jarvis/foldersync.py", "main", "?untyped:loop", 1),
    ("jarvis/foldersync.py", "main", "?untyped:parse_args", 1),
    ("jarvis/foldersync.py", "main", "?untyped:read_text", 1),
    ("jarvis/foldersync.py", "main", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "main", "?untyped:rstrip", 2),
    ("jarvis/foldersync.py", "main", "?untyped:run_pass", 1),
    ("jarvis/foldersync.py", "main", "?untyped:split", 1),
    ("jarvis/foldersync.py", "main", "?untyped:splitlines", 1),
    ("jarvis/foldersync.py", "main", "?untyped:status_text", 1),
    ("jarvis/foldersync.py", "name_series", "?untyped:rpartition", 1),
    ("jarvis/foldersync.py", "parse_sftp_entries", "?untyped:append", 1),
    ("jarvis/foldersync.py", "parse_sftp_entries", "?untyped:join", 1),
    ("jarvis/foldersync.py", "parse_sftp_entries", "?untyped:rsplit", 1),
    ("jarvis/foldersync.py", "parse_sftp_entries", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "parse_sftp_entries", "?untyped:split", 1),
    ("jarvis/foldersync.py", "parse_sftp_entries", "?untyped:splitlines", 1),
    ("jarvis/foldersync.py", "parse_sftp_entries", "?untyped:startswith", 1),
    ("jarvis/foldersync.py", "path_problem", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "preflight", "?untyped:append", 1),
    ("jarvis/foldersync.py", "preflight", "?untyped:append", 2),
    ("jarvis/foldersync.py", "preflight", "?untyped:append", 3),
    ("jarvis/foldersync.py", "preflight", "?untyped:append", 4),
    ("jarvis/foldersync.py", "preflight", "?untyped:join", 1),
    ("jarvis/foldersync.py", "preflight", "?untyped:resolve", 1),
    ("jarvis/foldersync.py", "preflight", "?untyped:resolve", 2),
    ("jarvis/foldersync.py", "preflight", "?untyped:resolve", 3),
    ("jarvis/foldersync.py", "read_config.get", "?untyped:debug", 1),
    ("jarvis/foldersync.py", "read_config.get", "?untyped:get", 1),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:append", 1),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:append", 2),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:append", 3),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:append", 4),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:append", 5),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:join", 1),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:strip", 1),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:strip", 2),
    ("jarvis/foldersync.py", "remote_folder_problems", "?untyped:strip", 3),
    ("jarvis/foldersync.py", "single_instance", "?untyped:close", 1),
    ("jarvis/foldersync.py", "single_instance", "?untyped:fileno", 1),
    ("jarvis/foldersync.py", "single_instance", "?untyped:warning", 1),
    ("jarvis/foldersync.py", "skip_reason", "?untyped:endswith", 1),
    ("jarvis/foldersync.py", "skip_reason", "?untyped:endswith", 2),
    ("jarvis/foldersync.py", "skip_reason", "?untyped:startswith", 1),
    ("jarvis/foldersync.py", "stat_key", "?untyped:stat", 1),
    ("jarvis/foldersync.py", "windows_name_problem", "?untyped:lower", 1),
    ("jarvis/foldersync.py", "windows_name_problem", "?untyped:rstrip", 1),
    ("jarvis/foldersync.py", "windows_name_problem", "?untyped:split", 1),
    # ---------------------------------------- jarvis/tools/filepick.py
    ("jarvis/tools/filepick.py", "_norm", "?untyped:lower", 1),
    ("jarvis/tools/filepick.py", "_norm", "?untyped:strip", 1),
    ("jarvis/tools/filepick.py", "_walk", "?untyped:append", 1),
    ("jarvis/tools/filepick.py", "_walk", "?untyped:debug", 1),
    ("jarvis/tools/filepick.py", "_walk", "?untyped:is_dir", 1),
    ("jarvis/tools/filepick.py", "_walk", "?untyped:is_file", 1),
    ("jarvis/tools/filepick.py", "_walk", "?untyped:pop", 1),
    ("jarvis/tools/filepick.py", "_walk", "?untyped:startswith", 1),
    ("jarvis/tools/filepick.py", "_within", "?untyped:relative_to", 1),
    ("jarvis/tools/filepick.py", "_within", "?untyped:resolve", 1),
    ("jarvis/tools/filepick.py", "check_file", "?untyped:stat", 1),
    ("jarvis/tools/filepick.py", "describe", "?untyped:join", 1),
    ("jarvis/tools/filepick.py", "expand_roots", "?untyped:append", 1),
    ("jarvis/tools/filepick.py", "expand_roots", "?untyped:is_dir", 1),
    ("jarvis/tools/filepick.py", "expand_roots", "?untyped:resolve", 1),
    ("jarvis/tools/filepick.py", "pick", "?untyped:add", 1),
    ("jarvis/tools/filepick.py", "pick", "?untyped:append", 1),
    ("jarvis/tools/filepick.py", "pick", "?untyped:resolve", 1),
    ("jarvis/tools/filepick.py", "pick", "?untyped:resolve", 2),
    ("jarvis/tools/filepick.py", "pick", "?untyped:resolve", 3),
    ("jarvis/tools/filepick.py", "pick", "?untyped:sort", 1),
    ("jarvis/tools/filepick.py", "pick", "?untyped:startswith", 1),
    ("jarvis/tools/filepick.py", "pick", "?untyped:startswith", 2),
    ("jarvis/tools/filepick.py", "pick", "?untyped:strip", 1),
    ("jarvis/tools/filepick.py", "pick", "?untyped:strip", 2),
    ("jarvis/tools/filepick.py", "pick", "?untyped:strip", 3),
    ("jarvis/tools/filepick.py", "reason_line", "?untyped:format", 1),
    ("jarvis/tools/filepick.py", "reason_line", "?untyped:get", 1),
    ("jarvis/tools/filepick.py", "score", "?untyped:ratio", 1),
    ("jarvis/tools/filepick.py", "score", "?untyped:ratio", 2),
    ("jarvis/tools/filepick.py", "score", "?untyped:split", 1),
    ("jarvis/tools/filepick.py", "score", "?untyped:split", 2),
    # ---------------------------------------- jarvis/tools/remote.py
    ("jarvis/tools/remote.py", "_cfg_get", "?untyped:debug", 1),
    ("jarvis/tools/remote.py", "_list_sftp", "?untyped:search", 1),
    ("jarvis/tools/remote.py", "_list_sftp", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "_norm_os", "?untyped:lower", 1),
    ("jarvis/tools/remote.py", "_norm_os", "?untyped:strip", 1),
    ("jarvis/tools/remote.py", "_norm_os", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "_remote_folder", "?untyped:strip", 1),
    ("jarvis/tools/remote.py", "ask", "?untyped:join", 1),
    ("jarvis/tools/remote.py", "classify_error", "?untyped:search", 1),
    ("jarvis/tools/remote.py", "classify_error", "?untyped:search", 2),
    ("jarvis/tools/remote.py", "classify_error", "?untyped:search", 3),
    ("jarvis/tools/remote.py", "classify_error", "?untyped:search", 4),
    ("jarvis/tools/remote.py", "classify_error", "?untyped:search", 5),
    ("jarvis/tools/remote.py", "classify_error", "?untyped:search", 6),
    ("jarvis/tools/remote.py", "classify_error", "?untyped:search", 7),
    ("jarvis/tools/remote.py", "fail_line", "?untyped:format", 1),
    ("jarvis/tools/remote.py", "fail_line", "?untyped:get", 1),
    ("jarvis/tools/remote.py", "inbox_target", "?untyped:match", 1),
    ("jarvis/tools/remote.py", "inbox_target", "?untyped:rstrip", 1),
    ("jarvis/tools/remote.py", "inbox_target", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "is_remote_stage", "?untyped:endswith", 1),
    ("jarvis/tools/remote.py", "is_remote_stage", "?untyped:match", 1),
    ("jarvis/tools/remote.py", "is_remote_stage", "?untyped:startswith", 1),
    ("jarvis/tools/remote.py", "is_remote_temp", "?untyped:endswith", 1),
    ("jarvis/tools/remote.py", "is_remote_temp", "?untyped:match", 1),
    ("jarvis/tools/remote.py", "is_remote_temp", "?untyped:startswith", 1),
    ("jarvis/tools/remote.py", "list_remote", "?untyped:append", 1),
    ("jarvis/tools/remote.py", "list_remote", "?untyped:endswith", 1),
    ("jarvis/tools/remote.py", "list_remote", "?untyped:info", 1),
    ("jarvis/tools/remote.py", "list_remote", "?untyped:match", 1),
    ("jarvis/tools/remote.py", "list_remote", "?untyped:splitlines", 1),
    ("jarvis/tools/remote.py", "list_remote", "?untyped:strip", 1),
    ("jarvis/tools/remote.py", "match_remote", "?untyped:sort", 1),
    ("jarvis/tools/remote.py", "match_remote", "?untyped:strip", 1),
    ("jarvis/tools/remote.py", "missing_command", "?untyped:groups", 1),
    ("jarvis/tools/remote.py", "missing_command", "?untyped:search", 1),
    ("jarvis/tools/remote.py", "pull", "?untyped:debug", 1),
    ("jarvis/tools/remote.py", "pull", "?untyped:match", 1),
    ("jarvis/tools/remote.py", "pull", "?untyped:rstrip", 1),
    ("jarvis/tools/remote.py", "query_command", "?untyped:get", 1),
    ("jarvis/tools/remote.py", "query_command", "?untyped:get", 2),
    ("jarvis/tools/remote.py", "query_say", "?untyped:get", 1),
    ("jarvis/tools/remote.py", "query_say", "?untyped:get", 2),
    ("jarvis/tools/remote.py", "query_say", "?untyped:get", 3),
    ("jarvis/tools/remote.py", "read_config", "?untyped:items", 1),
    ("jarvis/tools/remote.py", "read_config", "?untyped:strip", 1),
    ("jarvis/tools/remote.py", "read_config", "?untyped:strip", 2),
    ("jarvis/tools/remote.py", "read_config", "?untyped:strip", 3),
    ("jarvis/tools/remote.py", "read_config", "?untyped:strip", 4),
    ("jarvis/tools/remote.py", "read_config", "?untyped:strip", 5),
    ("jarvis/tools/remote.py", "read_config", "?untyped:strip", 6),
    ("jarvis/tools/remote.py", "remote_dir", "?untyped:get", 1),
    ("jarvis/tools/remote.py", "run_copy", "?untyped:communicate", 1),
    ("jarvis/tools/remote.py", "run_copy", "?untyped:info", 1),
    ("jarvis/tools/remote.py", "run_copy", "?untyped:kill", 1),
    ("jarvis/tools/remote.py", "run_copy", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "run_copy", "?untyped:warning", 2),
    ("jarvis/tools/remote.py", "run_sftp", "?untyped:communicate", 1),
    ("jarvis/tools/remote.py", "run_sftp", "?untyped:info", 1),
    ("jarvis/tools/remote.py", "run_sftp", "?untyped:kill", 1),
    ("jarvis/tools/remote.py", "run_sftp", "?untyped:rstrip", 1),
    ("jarvis/tools/remote.py", "run_sftp", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "run_sftp", "?untyped:warning", 2),
    ("jarvis/tools/remote.py", "run_sftp", "?untyped:warning", 3),
    ("jarvis/tools/remote.py", "run_ssh", "?untyped:communicate", 1),
    ("jarvis/tools/remote.py", "run_ssh", "?untyped:info", 1),
    ("jarvis/tools/remote.py", "run_ssh", "?untyped:kill", 1),
    ("jarvis/tools/remote.py", "run_ssh", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "run_ssh", "?untyped:warning", 2),
    ("jarvis/tools/remote.py", "run_ssh", "?untyped:warning", 3),
    ("jarvis/tools/remote.py", "scp_path", "?untyped:lstrip", 1),
    ("jarvis/tools/remote.py", "scp_path", "?untyped:startswith", 1),
    ("jarvis/tools/remote.py", "scp_path", "?untyped:strip", 1),
    ("jarvis/tools/remote.py", "sftp_listing", "?untyped:append", 1),
    ("jarvis/tools/remote.py", "sftp_listing", "?untyped:rsplit", 1),
    ("jarvis/tools/remote.py", "sftp_listing", "?untyped:rstrip", 1),
    ("jarvis/tools/remote.py", "sftp_listing", "?untyped:split", 1),
    ("jarvis/tools/remote.py", "sftp_listing", "?untyped:splitlines", 1),
    ("jarvis/tools/remote.py", "sftp_listing", "?untyped:startswith", 1),
    ("jarvis/tools/remote.py", "sftp_mkdir", "?untyped:search", 1),
    ("jarvis/tools/remote.py", "sftp_mkdir", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "sftp_remove", "?untyped:search", 1),
    ("jarvis/tools/remote.py", "sftp_remove", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "sftp_remove", "?untyped:warning", 2),
    ("jarvis/tools/remote.py", "sftp_rename", "?untyped:search", 1),
    ("jarvis/tools/remote.py", "sftp_rename", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "sftp_rmdir", "?untyped:search", 1),
    ("jarvis/tools/remote.py", "sftp_rmdir", "?untyped:warning", 1),
    ("jarvis/tools/remote.py", "shell_path", "?untyped:startswith", 1),
    ("jarvis/tools/remote.py", "shell_path", "?untyped:strip", 1),
    ("jarvis/tools/remote.py", "shell_path", "?untyped:strip", 2),
    ("jarvis/tools/remote.py", "tailnet_host", "?untyped:endswith", 1),
    ("jarvis/tools/remote.py", "tailnet_host", "?untyped:lower", 1),
    ("jarvis/tools/remote.py", "tailnet_host", "?untyped:match", 1),
    ("jarvis/tools/remote.py", "tailnet_host", "?untyped:strip", 1),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:communicate", 1),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:debug", 1),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:get", 1),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:get", 2),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:get", 3),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:get", 4),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:lower", 1),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:lower", 2),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:split", 1),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:split", 2),
    ("jarvis/tools/remote.py", "tailnet_state", "?untyped:values", 1),
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


def _untyped(rows):
    return [r for r in rows if r[2].startswith("?untyped:")]


def _named(rows):
    return [r for r in rows if not r[2].startswith("?untyped:")]


def test_every_write_in_this_lane_is_in_the_table():
    """A write nobody has written a line about fails here, not in his
    Outbox.  This is the pin the last three tables did not have -- and
    since round 5 it is per CALL SITE, so a second write in a function
    that already has one cannot hide behind the first (defeat 1)."""
    missing = [(m, f, w, n, g or "NOTHING") for m, f, w, n, g in
               _named(census()) if (m, f, w, n) not in KNOWN]
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
    live = {(m, f, w, n) for m, f, w, n, _g in _named(census())}
    stale = sorted(k for k in KNOWN if k not in live)
    assert not stale, f"KNOWN describes writes that no longer exist: {stale}"


def test_every_untypable_call_site_is_pinned():
    """RULE T, the pin.  A call on a receiver this file cannot type is a row
    since round 7, and a NEW one fails here.  The fix is to regenerate
    UNTYPED_SITES -- it is generated data -- and then to look at what
    changed, because that is the part a person has to do."""
    missing = [r[:4] for r in _untyped(census()) if r[:4] not in UNTYPED_SITES]
    assert not missing, (
        "an untypable receiver appeared that nothing pins:\n  " +
        "\n  ".join(f"{m}:{f} -> {w} #{n}" for m, f, w, n in missing))


def test_the_untyped_pin_has_no_sites_that_have_gone():
    live = {r[:4] for r in _untyped(census())}
    stale = sorted(k for k in UNTYPED_SITES if k not in live)
    assert not stale, f"UNTYPED_SITES pins calls that are gone: {stale}"


def test_every_untypable_method_name_has_a_sentence():
    """AND THIS IS THE ONE THAT MATTERS.  UNTYPED_SITES is data; this is the
    table a person reads, and a NEW METHOD NAME cannot get into it without
    somebody writing down what an object with that method is and why it
    cannot name a file of his.

    That sentence is exactly what does not exist for the routes that beat
    round 6: `p.shred()`, `p.hardlink_to(q)`, `logging.FileHandler(p, "w")`,
    `subprocess.getoutput("del ...")`, `self.transport.wipe_folder(name)`.
    Every one of them arrives here as a name with no sentence.
    """
    live = {r[2].split(":", 1)[1] for r in _untyped(census())}
    missing = sorted(live - set(UNTYPED))
    assert not missing, (
        "a method name on an untypable receiver that nobody has vouched "
        f"for: {missing}.  Write one sentence per name in UNTYPED saying "
        "what that object is here and why the call cannot create, truncate "
        "or destroy a name of his.  If you cannot write the sentence, that "
        "is the finding.")


def test_the_untyped_prose_has_no_names_that_have_gone():
    live = {r[2].split(":", 1)[1] for r in _untyped(census())}
    stale = sorted(set(UNTYPED) - live)
    assert not stale, f"UNTYPED vouches for names that are gone: {stale}"


def test_no_sentence_in_the_untyped_table_may_cover_a_mutator():
    """The trapdoor, shut.  Rule M runs BEFORE rule T, so a name in the
    mutator vocabulary can never reach this table -- but a future reader
    quieting a row by deleting a word out of WRITES and adding it here would
    undo the whole round.  `raw.replace(dest)` is what that looks like when
    it goes wrong: 100000 bytes to 4, eleven census tests green."""
    from tests.write_census import WRITES
    overlap = sorted(set(UNTYPED) & WRITES)
    assert not overlap, (
        f"{overlap} is vouched for in UNTYPED and is also a mutator.  A "
        "mutator name is a row on any receiver; that is rule M and it is "
        "not negotiable by prose.")


def test_the_split_pins_every_row_exactly_once():
    """No row may fall between the two tables, and none may be in both.
    A row that is in neither is the hole this whole file exists to close."""
    rows = census()
    named = {r[:4] for r in _named(rows)}
    untyped = {r[:4] for r in _untyped(rows)}
    assert not (named & untyped), "a row is in both halves"
    assert len(named) + len(untyped) == len(rows), "a row is in neither half"
    assert named <= set(KNOWN) and untyped <= UNTYPED_SITES


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
