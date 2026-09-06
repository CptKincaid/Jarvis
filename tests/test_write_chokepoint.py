"""THE CHOKEPOINT, as a property the census can now PROVE.

"Nothing may touch his folders except one audited function" was asked for in
round 6 and refused, on the grounds that it "buys a SHORTER TABLE, not a safer
check": to check "no mutating call outside the chokepoint" you must still
decide, of every call, whether it mutates -- and a chokepoint census that
SKIPS what it cannot resolve is walked past by ``raw.replace(dest)`` exactly
as the old one was.  That reasoning was right and it is why the chokepoint
could not be built first.

Round 7 deleted the last skip.  Rule T makes an untypable receiver a row, so
the census is now total over ``ast.Call`` with no free pass anywhere -- and
THAT is what makes the chokepoint checkable rather than decorative.  This file
is the check.  It is a policy over the code that already exists; it is not a
rewrite of 2665 lines of file-moving code that has survived six adversarial
rounds, which would put the product back at risk to tidy the instrument.

WHAT IT PINS.  A call is a PRIMITIVE write when its name is in the census's
mutator vocabulary and is NOT the name of a def or method of this lane -- so
``os.replace``, ``Path.write_text`` and ``subprocess.Popen`` are primitives,
while ``send``, ``land_beside``, ``save`` and ``run_sftp`` are this lane's own
audited verbs, each of which is a scope of its own with its own rows.  That
split is COMPUTED from the source, not listed here, so a new lane verb cannot
be smuggled in as a primitive or the other way round.

The pin is the SET OF SCOPES allowed to call a primitive directly.  Adding a
raw ``os.unlink`` to a nineteenth function fails here with the word
"chokepoint" in it, which is a different and more useful failure than "a new
row appeared".

WHAT IT MEASURED, and the number is the point.  Consolidating the two
identical "replace a file we own, atomically" dances -- ``Ledger.save`` and
``Syncer.write_status`` -- onto one audited ``_replace_ours`` took the census
from 487 rows to 483.  FOUR rows, 0.8% of the table.  Three of them are
mutation rows and exactly ONE is an untyped-receiver row: 1 of 395, 0.25%.

That is the honest answer to "take the chokepoint so the long table gets
readable".  IT DOES NOT.  The 395 untyped rows are `append`, `get`, `warning`
and `strip` on receivers nothing in this file can type, and a chokepoint over
MUTATIONS cannot touch one of them -- they are not mutations.  Rewriting more
of the product would not change that; the ceiling is the ~90 mutation rows,
and most of those are already behind this lane's own audited verbs.  So the
long table stays, and the thing that actually made it readable was splitting
the prose: 66 sentences about METHOD NAMES instead of 395 about call sites.

What the consolidation does buy is one place where the fsync decision for a
file of ours is written down -- ``Ledger.save`` fsyncs because the landing
record is the duplicate-send guarantee, ``write_status`` does not because
status.txt is a courtesy -- and that is worth having on its own, which is why
it was kept rather than reverted once the 0.8% was measured.
"""
import ast

from tests.write_census import (MODULES, REPO, WRITES, _lane_names,
                                _method_names, census)


def _primitives() -> set:
    """The mutator vocabulary MINUS every name this lane defines itself.

    Computed, never listed: `save` and `send` are lane verbs with scopes of
    their own, `replace` and `Popen` are primitives that stop here.
    """
    lane = set()
    for rel in MODULES:
        tree = ast.parse((REPO / rel).read_text())
        lane |= _lane_names(tree) | _method_names(tree)
    return WRITES - lane


# Every scope permitted to call an OS-level write primitive DIRECTLY.  Each
# one is a place where a name is created or destroyed, and each already has
# its rows and its guard kind pinned in tests/test_write_census.py.
PRIMITIVE_WRITERS = {
    # -- the audited local writers, foldersync
    ("jarvis/foldersync.py", "land_beside"),
    #    THE one function that may put a file into a folder of HIS: link,
    #    then move or replace, behind an O_EXCL claim.  Three call sites,
    #    all of which show `claim:land_beside`.
    ("jarvis/foldersync.py", "_replace_ours"),
    #    THE one function that replaces a file of OURS atomically: mkdir,
    #    write a temp beside it, optional fsync of file and directory, then
    #    os.replace.  Round 7 routed Ledger.save and Syncer.write_status
    #    through it; nothing else may hand-roll the dance.
    ("jarvis/foldersync.py", "Syncer._write_note"),
    #    the note beside his file, at a name HE can choose -- O_EXCL, then a
    #    marker read through the same descriptor it is truncated on.
    ("jarvis/foldersync.py", "Syncer._claim_part"),      # the O_EXCL part file
    ("jarvis/foldersync.py", "Syncer._drop"),            # unlink of our own part
    ("jarvis/foldersync.py", "Syncer._move_to_sent"),    # mkdir of our Sent dir
    ("jarvis/foldersync.py", "Syncer.pull_once"),        # mkdir of his Inbox
    ("jarvis/foldersync.py", "Syncer.record"),           # append to our history
    ("jarvis/foldersync.py", "_unlink_after_landing"),   # identity-checked
    ("jarvis/foldersync.py", "_unlink_if_ours"),         # identity-checked
    ("jarvis/foldersync.py", "single_instance"),         # the lock directory
    # -- the audited remote writers
    ("jarvis/tools/remote.py", "run_ssh"),      # subprocess.Popen(ssh)
    ("jarvis/tools/remote.py", "run_sftp"),     # subprocess.Popen(sftp)
    ("jarvis/tools/remote.py", "run_copy"),     # subprocess.Popen(scp)
    ("jarvis/tools/remote.py", "tailnet_state"),  # Popen(tailscale status)
    ("jarvis/tools/remote.py", "pull"),         # the O_EXCL local part file
    ("jarvis/tools/remote.py", "_remote_folder"),
    #    a FALSE POSITIVE kept on purpose: `folder.replace("\\", "/")` is
    #    str.replace on a string.  WRITES is matched on the resolved tail
    #    whatever the receiver is, which is what stopped `raw.replace(dest)`
    #    destroying a 100000-byte file -- and the price of that is this row.
    #    It is cheaper to explain one string than to type every receiver.
}


def test_no_function_outside_the_audited_set_touches_a_primitive():
    """The chokepoint, enforced.  A raw os.unlink, Path.write_text or
    subprocess.Popen in a nineteenth function fails here."""
    prim = _primitives()
    live = {(m, f) for m, f, w, _n, _g in census() if w in prim}
    extra = sorted(live - PRIMITIVE_WRITERS)
    assert not extra, (
        "a write primitive is being called outside the audited set:\n  " +
        "\n  ".join(f"{m}:{f}" for m, f in extra) +
        "\nRoute it through one of the audited writers, or add the scope "
        "here WITH a sentence saying what can be at that name.")


def test_the_audited_set_has_no_scopes_that_have_gone():
    """The other direction: a pinned scope that no longer writes is a policy
    drifting away from the source, which is how the last three tables went
    wrong."""
    prim = _primitives()
    live = {(m, f) for m, f, w, _n, _g in census() if w in prim}
    stale = sorted(PRIMITIVE_WRITERS - live)
    assert not stale, f"pinned writers that write nothing any more: {stale}"


def test_the_two_atomic_replaces_of_our_own_files_share_one_function():
    """The consolidation itself.  Ledger.save and Syncer.write_status each
    hand-rolled mkdir + write a temp + os.replace; the ledger's also fsynced
    the file and the directory, and that fsync is what makes a duplicate send
    impossible rather than unlikely.  Two copies of a dance where one of them
    has the safety step is exactly how the step goes missing."""
    src = (REPO / "jarvis/foldersync.py").read_text()
    tree = ast.parse(src)
    bodies = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies.setdefault(node.name, []).append(node)
    assert "_replace_ours" in bodies, "the audited replace is gone"
    prim = _primitives()
    for name in ("save", "write_status"):
        for node in bodies[name]:
            called = {n.func.attr for n in ast.walk(node)
                      if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute)}
            called |= {n.func.id for n in ast.walk(node)
                       if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name)}
            assert not (called & prim), (
                f"{name} hand-rolls {sorted(called & prim)} again instead of "
                "going through _replace_ours")
        assert any("_replace_ours" in ast.dump(n) for n in bodies[name]), (
            f"{name} does not route through the audited replace")


def test_the_chokepoint_removes_almost_none_of_the_untyped_rows():
    """The measurement, pinned so it cannot quietly become a claim.

    The chokepoint was asked for because "a shorter table is exactly what
    makes a 396-line KNOWN readable".  It is not: the untyped-receiver rows
    are `append`, `get`, `warning` and `strip` on receivers nothing can type,
    and a chokepoint over MUTATIONS removes none of them.  If a later round
    believes otherwise, this test is where the belief gets checked.
    """
    prim = _primitives()
    rows = census()
    untyped = [r for r in rows if r[2].startswith("?untyped:")]
    mutations = [r for r in rows if r[2] in prim]
    assert len(untyped) > 300, len(untyped)
    assert len(mutations) < len(untyped) / 8, (
        "the mutation rows are now a large fraction of the table; the "
        "measurement this test records no longer holds")
