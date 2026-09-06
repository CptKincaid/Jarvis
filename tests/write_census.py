"""Every read-then-write pair in this lane, DERIVED FROM THE SOURCE.

Three rounds of this lane have each shipped a hand-written table of "where
could a write destroy something", and two of the three were wrong:

  round 2  swept by hand and wrote "nothing else in the repo has either
           shape".  False: the sweep looked only for a LOCAL os.replace and
           never at a REMOTE write.
  round 3  wrote a fourteen-row table into foldersync's docstring as the
           starting point for the next sweep.  It was missing the push's own
           scp at its temp name -- an unbounded window, measured destroying
           a 99999-byte file and reporting "sent" -- and it did not mention
           the pull's part file at all, which destroyed another.

A table a person maintains is a table that goes stale silently.  So this
one is not maintained: it is COMPUTED, by walking the AST of the two
modules that move his files.

AND IT ENUMERATES EVERY WRITE, not every read-then-write PAIR, which is the
correction round 3 needed and did not get.  Both rows the attacker had to
add were writes with NO CHECK IN FRONT OF THEM AT ALL -- the push's scp at
its temp name and the pull's scp into its part file -- so a census that
looked only for "ask, then write" would have walked straight past both of
them, exactly as the human sweep did.  A write with no guard is not a safe
write; it is the most dangerous row on the table.

So every call to a write primitive is a row.  Against each one the census
records what LEXICALLY PRECEDES it in the same function, split into two
kinds, because they mean opposite things:

  a CHECK  ("does that name exist?")  -- the shape that has bitten this
           lane three times.  The gap between the question and the write is
           the window, and it can be minutes.
  a CLAIM  (os.link, O_EXCL, sftp mkdir, rename -l) -- one syscall that
           CREATES the name or refuses.  There is no gap at all.

The matching is deliberately CRUDE AND OVER-BROAD: it is lexical, it does
not follow control flow, and it does not try to prove that any given row is
a real race -- a checker that tried would be a third thing to get wrong.
:data:`KNOWN` in tests/test_write_census.py is the list of writes somebody
has actually looked at, one line each saying what can be at that name when
it happens.  A write the list does not name fails that test.  That is the
whole contract: you cannot add a write to these two modules without saying
what it can land on.

Run it to print today's table:

    ~/vss_env/bin/python -m tests.write_census
"""
import ast
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODULES = ("jarvis/foldersync.py", "jarvis/tools/remote.py")

# ---------------------------------------------------------------- the shapes
# A call that can DESTROY or CREATE a name.  Matched on the dotted tail of
# the callee, so os.replace, Path.write_text and self.transport.send all
# reduce to a single word.  Over-broad on purpose: a name here that turns
# out to be harmless costs one line in KNOWN; a name MISSING here costs a
# file of his.
WRITES = frozenset({
    # THE ARBITRARY ONE, FIRST.  run_ssh runs a command line of our
    # choosing on his Windows machine, so it is every write there is --
    # del, rmdir /s, move, a redirect.  It had NO WORD HERE at all until
    # round 5, and an adversary walked straight past the census with a
    # method doing a run_ssh `del` on his Windows Inbox: zero rows, four
    # green tests, the count unchanged at 63.  It is the most dangerous
    # single name in this file and it belongs at the top of the list.
    "run_ssh",
    # local
    "replace", "rename", "renames", "link", "symlink", "unlink", "remove",
    "rmdir", "removedirs", "mkdir", "makedirs", "write_text", "write_bytes",
    "open", "fdopen", "truncate", "ftruncate", "move", "copy", "copy2",
    "copyfile", "copytree", "rmtree", "touch", "utime", "write",
    "writelines", "mkstemp", "mkdtemp", "chmod", "chown",
    # remote
    "run_copy", "sftp_rename", "sftp_remove", "sftp_mkdir", "sftp_rmdir",
    "run_sftp", "push", "pull", "send", "claim", "discard", "fetch",
    "stage_open", "stage_close", "remove_landed", "land_beside", "save",
})

# A call that ASKS whether a name is free, or what is there.  A write that
# follows one of these in the same function is the shape that has bitten
# this lane three times.
CLAIMS = frozenset({
    # One operation that CREATES a name and refuses if it is taken.  There
    # is no gap between the question and the answer, so a write behind one
    # of these has a window of zero rather than a small one.
    "link", "sftp_mkdir", "sftp_rename", "stage_open", "claim",
    "mark_claiming", "land_beside", "_open_stage", "_claim_part",
})

READS = frozenset({
    "exists", "is_file", "is_dir", "stat", "lstat", "fstat", "listdir",
    "iterdir", "glob", "scandir", "read_text", "read_bytes", "read",
    "listing", "sftp_listing", "list_remote", "parse_sftp_entries",
    "stat_key", "dedupe_name", "claim_candidates", "name_series",
    "_list_sftp", "staged_file_missing", "landed_row", "has", "blocked",
    "load", "access",
})

# Writes that are not writes at his machine or his folders: pure predicates
# and constructors that happen to share a name with something dangerous.
IGNORE_CALLEES = frozenset({
    "log.exception", "log.warning", "log.info", "log.debug", "log.error",
    "argparse.ArgumentParser", "ap.add_argument",
    # Same WORD, nothing to do with a file.  Each of these is named in full
    # rather than by its tail, so os.replace and transport.discard are
    # still counted; only these exact call sites are not.
    "replace",                    # dataclasses.replace(conf, ...)
    "raw.replace",                # str.replace
    "self._my_stages.discard",    # set.discard
    "self._claimed_here.discard",  # set.discard
})


def _dotted(node: ast.AST) -> str:
    """The callee as source text, e.g. "os.replace" or "self.transport.send"."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _tail(name: str) -> str:
    return name.rsplit(".", 1)[-1]


SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _calls_in_scope(node) -> list:
    """Every ``ast.Call`` that belongs to THIS scope and no deeper one.

    Two corrections in one.  The old census walked each function whole, so a
    write in a nested closure was counted against the OUTER function as well
    -- a single call site showing up as two rows, which is exactly the kind
    of double vision that makes an occurrence count meaningless.  And it
    started at FunctionDef, so a scope with no function around it did not
    exist: DEFEAT 3, a module-level ``write_text`` guarded by an environment
    variable, censused as NOTHING.
    """
    out = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, SCOPED):
            continue                      # its own scope; listed separately
        if isinstance(child, ast.Call):
            out.append(child)
        stack.extend(ast.iter_child_nodes(child))
    return out


def _functions(tree: ast.AST):
    """(qualified name, [calls]) for every scope in the module.

    Functions, methods, lambdas -- AND the module body itself, AND each
    class body, because both of those run at import and can write.  A
    scope's own calls only: see :func:`_calls_in_scope`.
    """
    out = [("<module>", _calls_in_scope(tree))]

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                out.append((f"{prefix}{child.name}.<class body>",
                            _calls_in_scope(child)))
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((f"{prefix}{child.name}",
                            _calls_in_scope(child)))
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.Lambda):
                out.append((f"{prefix}<lambda>", _calls_in_scope(child)))
            else:
                walk(child, prefix)
    walk(tree)
    return [(name, calls) for name, calls in out if calls]


def census(paths=MODULES, sources=None) -> list:
    """One row per WRITE CALL SITE: ``(module, scope, write, n, guard)``.

    ``n`` is which occurrence of that primitive this is within that scope,
    counting from 1 in line order.  IT IS THERE BECAUSE THE ROW KEY WITHOUT
    IT IS DEFEATABLE.  Round 4 keyed a row (module, function, primitive) and
    kept one row per primitive per function, so a SECOND write of a
    primitive already listed, in a function already listed, was invisible.
    MEASURED: an ask-then-write added to ``Syncer.record`` destroyed a
    100000-byte file of his at ``ok.txt.log`` -- 100000 bytes to 0 -- with
    all four census tests green and the row count unchanged at 63.  The
    KNOWN line for that row ("our own history file... no name of his is
    involved") had silently become false and nothing said so.

    An ORDINAL rather than a LINE NUMBER on purpose: a line number churns
    the table on every unrelated edit above it, and a table that cries wolf
    is a table people edit to green.  The ordinal is stable under insertions
    elsewhere and changes exactly when a call site is added or removed.
    Its one blind spot is stated in tests/test_write_census.py.

    ``guard`` is "claim:<what>" when an atomic claim precedes the write in
    that scope, "check:<what>" when only a question does, and "" when
    NOTHING does -- which is the row that cost this lane two files.

    ``sources`` maps a module name to source TEXT, so a test can census a
    mutated copy of this lane without writing to the working tree.
    """
    found = []
    for rel in paths:
        text = (sources or {}).get(rel)
        if text is None:
            text = (REPO / rel).read_text()
        tree = ast.parse(text)
        for qual, calls in _functions(tree):
            checks, claims, writes = [], [], []
            for node in calls:
                dotted = _dotted(node.func)
                if not dotted or dotted in IGNORE_CALLEES:
                    continue
                tail = _tail(dotted)
                line = getattr(node, "lineno", 0)
                if tail in CLAIMS or _excl_open(node, tail):
                    claims.append((line, "excl-open" if tail == "open"
                                   else tail))
                elif tail in READS:
                    checks.append((line, tail))
                if tail in WRITES:
                    writes.append((line, tail))
            seen = {}
            for wline, wtail in sorted(writes):
                seen[wtail] = seen.get(wtail, 0) + 1
                before_claim = sorted({t for ln, t in claims if ln <= wline})
                before_check = sorted({t for ln, t in checks if ln <= wline})
                if before_claim:
                    guard = "claim:" + "+".join(before_claim)
                elif before_check:
                    guard = "check:" + "+".join(before_check)
                else:
                    guard = ""
                found.append((rel, qual, wtail, seen[wtail], guard))
    return sorted(found)


def kind(guard: str) -> str:
    """A guard reduced to the only three answers that matter."""
    if guard.startswith("claim:"):
        return "claim"
    if guard.startswith("check:"):
        return "check"
    return "none"


def _excl_open(node: ast.Call, tail: str) -> bool:
    """``os.open(..., O_CREAT | O_EXCL | ...)`` is a CLAIM, not a check --
    it is the local half of the same idea as sftp mkdir."""
    if tail != "open":
        return False
    return "O_EXCL" in ast.dump(node)


def _weaker(a: str, b: str) -> bool:
    rank = {"": 0}
    return rank.get(a, 2 if a.startswith("claim") else 1) < \
        rank.get(b, 2 if b.startswith("claim") else 1)


def table(rows=None, known=None) -> str:
    """The census as the table that used to be hand-written."""
    from tests.test_write_census import KNOWN          # noqa: PLC0415
    rows = census() if rows is None else rows
    known = KNOWN if known is None else known
    def where(m, f, w, n):
        return f"{m}:{f}" + (f" #{n}" if n > 1 else "")
    width = max(len(where(m, f, w, n)) for m, f, w, n, _g in rows)
    gwidth = max(len(f"{g} -> {w}") for _m, _f, w, _n, g in rows)
    out = ["WHERE".ljust(width) + "  GUARD -> WRITE".ljust(gwidth + 2) +
           "  WINDOW"]
    out.append("-" * (width + gwidth + 40))
    for mod, fn, write, n, guard in rows:
        row = known.get((mod, fn, write, n))
        note = row[1] if row else "*** NOT IN THE TABLE ***"
        out.append(where(mod, fn, write, n).ljust(width) +
                   f"  {guard or 'NOTHING'} -> {write}".ljust(gwidth + 2) +
                   f"  {note}")
    return "\n".join(out)


if __name__ == "__main__":                # pragma: no cover - a printout
    import sys
    sys.path.insert(0, str(REPO))
    print(table())
    print(f"\n{len(census())} writes in "
          f"{', '.join(MODULES)}, derived by AST from the source at "
          f"{os.environ.get('GIT_COMMIT', 'this working tree')}.")
