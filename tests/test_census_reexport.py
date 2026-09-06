"""RULE S TRUSTED A PREFIX.  A prefix is not a promise about what is behind it.

ROUND 8.  Round 7 made silence require a resolved, vouched, fully-qualified
name and that was right.  What it did not notice is that TWO of the rules
that hand out that vouching match a PREFIX of the resolved name and then
stop looking:

    rule S, walked module   any(resolved.startswith(w + ".") for w in walked)
    rule S, harmless module resolved.split(".")[0] in HARMLESS_MODULES

Both were written as a statement about a MODULE.  Both are read by the code
as a statement about every name that module's namespace can reach.  Those
are not the same statement, because A MODULE RE-EXPORTS EVERY MODULE IT
IMPORTS AS AN ATTRIBUTE OF ITSELF.  ``import subprocess`` at the top of
remote.py makes ``remote.subprocess`` a real, working name, and the census
walked straight past it on the strength of the four characters before it.

MEASURED HERE on the round-7 tip (2785994), before any of this was fixed --
one line planted into ``Syncer._candidates`` of a git-archive copy under the
session scratchpad, nothing in his own folders touched:

    remote.subprocess.getoutput("rm -f .../quarterly.xlsx")

    quarterly.xlsx      100000 bytes  ->  GONE
    census              483 rows, byte-identical to the clean baseline
    census test files   42 passed

Sixteen further spellings of the same one idea were planted in memory
against that same unfixed instrument.  EVERY ONE was silent.  They are all
below, each as its own test, and they are the reason the fix is not a
patch to the two names the adversary reported:

  through a WALKED module's namespace (the reported family)
    remote.subprocess.getoutput / .getstatusoutput      the reported route
    remote.os.posix_spawn / remote.os.execve            same shape, os
    remote.<a module-level variable>                    NOT a def or a class
    remote.<a class imported from elsewhere>            FileHandler's shape
    remote.<a class attribute>                          _Ops.rm, one file over
    from jarvis.tools.remote import subprocess as _sp   the from-import
    import jarvis.tools.remote; jarvis.tools.remote...  the full dotted one
    remote.subprocess.os.posix_spawn                    two modules deep
    remote.filepick.os.posix_spawn                      walked through walked

  through a HARMLESS module's namespace (NOBODY REPORTED THIS ONE)
    shlex.os.posix_spawn        remote.py ALREADY does `import shlex`.  Zero
                                new imports, zero indirection, one line.
    contextlib.os.posix_spawn   contextlib imports os
    uuid.os.posix_spawn         uuid imports os
    traceback.linecache.os...   traceback imports linecache imports os
    typing.contextlib.os...     three deep

The second family is the one that matters most, because HARMLESS_MODULES is
the list whose own comment says the test for membership is "this module has
no function that takes a name and creates, truncates or destroys the thing
at it".  That sentence is about the module's OWN surface.  The code asked it
of the whole reachable namespace, which no stdlib module can pass: shlex,
contextlib, uuid, traceback, typing and dataclasses all re-export ``os``.

THE FIX IS ONE SENTENCE.  A vouched namespace vouches for its own members,
never for what it re-exports.  So both prefix rules now stop at ONE segment,
and the walked-module one additionally requires that the segment is a real
``def`` or ``class`` STATEMENT of that module -- because "a name bound in
that module" would have blessed ``remote.subprocess`` and
``remote.FileHandler`` all over again.

Everything below is a defeat test.  Nothing here opens a socket, touches
~/Desktop, or speaks to HPCOMPUTER; every plant is IN MEMORY through
``census(sources=...)`` and the working tree is never written.
"""
from pathlib import Path

import pytest

from tests import write_census as wc
from tests.write_census import MODULES, census

REPO = Path(__file__).resolve().parent.parent
FS = "jarvis/foldersync.py"
RM = "jarvis/tools/remote.py"

SRC = {m: (REPO / m).read_text() for m in MODULES}

# The two anchors every plant hangs off.  Both are asserted present, so a
# moved anchor fails loudly instead of silently planting nothing.
FS_SCOPE = """    def _candidates(self) -> list:
        out = []
        try:"""
FS_IMPORTS = "from jarvis.tools import filepick, remote"
RM_IMPORTS = "import shlex"
RM_SCOPE = "def shell_path("


def plant(fs_line="", fs_import="", rm_line="", rm_top=""):
    """This lane's source with ONE line added, as TEXT.  Nothing is written."""
    out = dict(SRC)
    if fs_line or fs_import:
        text = out[FS]
        if fs_import:
            assert FS_IMPORTS in text, "the foldersync import block moved"
            text = text.replace(FS_IMPORTS, FS_IMPORTS + "\n" + fs_import, 1)
        if fs_line:
            assert FS_SCOPE in text, "Syncer._candidates moved"
            text = text.replace(FS_SCOPE, FS_SCOPE.replace(
                "        try:", f"        {fs_line}\n        try:"), 1)
        out[FS] = text
    if rm_line or rm_top:
        text = out[RM]
        if rm_top:
            assert RM_IMPORTS in text, "the remote import block moved"
            text = text.replace(RM_IMPORTS, RM_IMPORTS + "\n" + rm_top, 1)
        if rm_line:
            assert RM_SCOPE in text, "remote.shell_path moved"
            text = text.replace(
                RM_SCOPE, f"def _planted_here():\n    {rm_line}\n\n\n"
                + RM_SCOPE, 1)
        out[RM] = text
    return out


def rows_for(**kw):
    return set(census(sources=plant(**kw)))


@pytest.fixture(scope="module")
def clean():
    return set(census())


# Every route, as (label, plant kwargs).  ONE table, because the whole point
# is that these are one idea and not sixteen: each reaches THROUGH a
# namespace the census vouches for, into one it does not.
WALKED_ROUTES = [
    # THE REPORTED ONE.  Measured destroying a 100000-byte file of his with
    # the round-7 census frozen at 483 rows and its 42 tests green.
    ("remote.subprocess.getoutput",
     dict(fs_line='remote.subprocess.getoutput("rm -f X")')),
    ("remote.subprocess.getstatusoutput",
     dict(fs_line='remote.subprocess.getstatusoutput("rm -f X")')),
    # getoutput was the one that was reported.  os is the same door: remote
    # does `import os` too, and the exec family is arbitrary execution under
    # a name that was not in WRITES.
    ("remote.os.posix_spawn",
     dict(fs_line='remote.os.posix_spawn("/bin/rm", ["rm", "-f", "X"], {})')),
    ("remote.os.execve",
     dict(fs_line='remote.os.execve("/bin/rm", ["rm", "-f", "X"], {})')),
    # MINE.  A module-level VARIABLE of a walked module is not a def and not
    # a class, and this file cannot prove what it holds.  It is round 5's
    # `_rm = os.remove` alias attack moved one module across -- and it is
    # ALSO the shape this lane's own SAFE_REMOTE_NAME_RX is written in, so
    # closing it costs seven honest rows rather than nothing.
    ("a walked module's module-level variable",
     dict(rm_top="_RM = os.remove", fs_line='remote._RM("X")')),
    # MINE.  Round 7 took `logging` out of HARMLESS_MODULES because
    # FileHandler(p, "w") truncates.  A walked module that imports the class
    # hands the same constructor back through a name the prefix trusts, so
    # blessing "a name bound in that module" would have reopened route A of
    # round 7 word for word.  Hence: a def or a class STATEMENT, nothing else.
    ("a class a walked module imported from elsewhere",
     dict(rm_top="from logging import FileHandler",
          fs_line='remote.FileHandler("X", "w")')),
    # MINE.  `<lane>._Ops.rm` was round 7's ninth route and it was closed
    # INSIDE a module.  Across modules it was never closed at all.
    ("a walked module's class attribute",
     dict(rm_top="class _Ops:\n    rm = os.remove",
          fs_line='remote._Ops.rm("X")')),
    # MINE.  The from-import spelling: no attribute is written at the call
    # site at all, and the dangerous name is bound at the top of the file
    # among the ordinary imports where it reads as one.
    ("from jarvis.tools.remote import subprocess as _sp",
     dict(fs_import="from jarvis.tools.remote import subprocess as _sp",
          fs_line='_sp.getoutput("rm -f X")')),
    ("the full dotted import spelling",
     dict(fs_import="import jarvis.tools.remote",
          fs_line='jarvis.tools.remote.subprocess.getoutput("rm -f X")')),
    # MINE.  Two modules deep: subprocess re-exports os in its turn, so the
    # depth behind a trusted prefix is not one hop, it is the whole import
    # graph of the standard library.
    ("two modules deep through the walked one",
     dict(fs_line='remote.subprocess.os.posix_spawn("/bin/rm", ["rm"], {})')),
    # MINE.  A walked module aliasing ANOTHER walked module.  Both ends are
    # censused, and the route between them was not.
    ("walked module aliasing a walked module",
     dict(fs_line='remote.filepick.os.posix_spawn("/bin/rm", ["rm"], {})')),
]

HARMLESS_ROUTES = [
    # MINE, AND THE ONE NOBODY REPORTED.  remote.py already has `import
    # shlex` at line 96.  No new import, no alias, no lazy import, no
    # getattr: one line inside an existing function, and `shlex` is on the
    # list of modules vouched for WHOLESALE.
    ("shlex.os.posix_spawn, inside remote.py, no new import",
     dict(rm_line='shlex.os.posix_spawn("/bin/rm", ["rm", "-f", "X"], {})')),
    ("contextlib.os.posix_spawn",
     dict(fs_import="import contextlib",
          fs_line='contextlib.os.posix_spawn("/bin/rm", ["rm"], {})')),
    ("uuid.os.posix_spawn",
     dict(fs_import="import uuid",
          fs_line='uuid.os.posix_spawn("/bin/rm", ["rm"], {})')),
    ("traceback.linecache.os.posix_spawn",
     dict(fs_import="import traceback",
          fs_line='traceback.linecache.os.posix_spawn("/bin/rm", ["rm"], {})')),
    ("typing.contextlib.os.posix_spawn",
     dict(fs_import="import typing",
          fs_line='typing.contextlib.os.posix_spawn("/bin/rm", ["rm"], {})')),
]

ALL_ROUTES = WALKED_ROUTES + HARMLESS_ROUTES


@pytest.mark.parametrize("label,kw", WALKED_ROUTES,
                         ids=[r[0] for r in WALKED_ROUTES])
def test_reaching_through_a_walked_modules_namespace_is_a_row(label, kw, clean):
    """Rule S blessed ``<walked module>.<anything at all>``.  It may now
    bless only ``<walked module>.<a def or class of that module>``."""
    new = rows_for(**kw) - clean
    assert new, (
        f"SILENT: {label} reaches through a namespace the census vouches "
        "for into one it does not, and the row count did not move.  This "
        "is arbitrary execution with the instrument green.")


@pytest.mark.parametrize("label,kw", HARMLESS_ROUTES,
                         ids=[r[0] for r in HARMLESS_ROUTES])
def test_reaching_through_a_harmless_modules_namespace_is_a_row(label, kw,
                                                                clean):
    """HARMLESS_MODULES is matched on the ROOT of the resolved name, so it
    vouched for everything that module's namespace can reach.  Its own
    comment only ever claimed the module's own surface.  Six of the
    twenty-nine names on that list re-export ``os``."""
    new = rows_for(**kw) - clean
    assert new, (
        f"SILENT: {label}.  A module on HARMLESS_MODULES vouched for a "
        "module it merely imports.")


def test_the_structure_catches_them_and_not_the_vocabulary(clean):
    """THE ACCEPTANCE BAR, and the reason this round added NOTHING to WRITES.

    The brief asked for ``getoutput``, ``getstatusoutput`` and the
    ``os.exec*``/``posix_spawn`` family to go into WRITES as defence in
    depth.  That was tried and REFUSED, because it does not add a lock, it
    takes one off:

      * every one of the sixteen routes below then became a rule-M row
        named after its own primitive, so the table said "a known write"
        where what it knows is "a call reached through a namespace this
        census does not walk" -- and only the second sentence survives a
        name nobody has thought of yet;
      * round 7's own two tests failed on it, which is what a test is for:
        ``test_none_of_these_ten_needed_a_per_attack_entry`` (no table the
        instrument consults may name a trick) and
        ``test_a_lazy_import_resolves_exactly_like_a_module_level_one``
        (which pins ``subprocess.getoutput`` resolving STRUCTURALLY, as
        ?reach; a WRITES entry silently took that away).

    So the assertion is the strong one: every route is a row, and NOT ONE
    of them is caught by its name.  ``?through:`` means rule S refused to
    vouch -- if any route needed rule M, the vocabulary would be load
    bearing and the seventeenth spelling walks through.
    """
    by_name = []
    for label, kw in ALL_ROUTES:
        new = sorted(set(census(sources=plant(**kw))) - clean)
        assert new, f"SILENT: {label}"
        if not all(r[2].startswith("?through:") for r in new):
            by_name.append(f"{label}: {[r[2] for r in new]}")
    assert not by_name, (
        "these routes are caught by their NAME being in WRITES rather than "
        "by the namespace rule, which is a per-attack entry and buys "
        "exactly one round:\n  " + "\n  ".join(by_name))

    # And the names themselves are absent from every table, which is round
    # 7's rule and now also this round's.
    tables = " ".join(str(getattr(wc, n)) for n in dir(wc)
                      if n.isupper() and isinstance(
                          getattr(wc, n), (frozenset, set, tuple, list, dict)))
    for trick in ("getoutput", "getstatusoutput", "posix_spawn", "execve",
                  "linecache", "FileHandler", "startfile"):
        assert trick not in tables, (
            f"{trick!r} is named in a table of the instrument")


def test_a_real_function_of_a_walked_module_is_still_silence(clean):
    """The other direction, and the reason the rule is 'a def or a class'
    rather than 'nothing'.  ``remote.scp_path`` is a top-level def of a
    walked module: its body is a scope of its own and is censused there, so
    calling it here must stay silent or the table drowns."""
    new = rows_for(fs_line='remote.scp_path("X")') - clean
    assert not new, (
        "a plain call into a walked module's own function became a row; "
        f"the rule is too tight: {sorted(new)}")


def test_a_harmless_modules_own_function_is_still_silence(clean):
    """``json.dumps`` is one segment past a vouched root and stays silent.
    ``json.codecs.open`` is two and does not."""
    assert not rows_for(fs_line='json.dumps({})') - clean
    assert rows_for(fs_line='json.codecs.EncodedFile(0, "utf-8")') - clean


def test_the_arbitrary_command_primitives_are_deliberately_not_in_writes():
    """The refusal, pinned so a later round cannot quietly take the offer.

    These names all run a command line of somebody's choosing.  Putting
    them in WRITES looks like free safety and is not: it converts a
    structural row into a vocabulary row, and a vocabulary row is a promise
    about the names somebody has thought of.  The lane has lost three
    rounds to that promise.  If this assertion ever needs to change, the
    thing to change is the namespace rule, not the word list.
    """
    for name in ("getoutput", "getstatusoutput", "execve", "execl", "execle",
                 "execlp", "execvpe", "posix_spawn", "posix_spawnp",
                 "startfile"):
        assert name not in wc.WRITES, (
            f"{name!r} was added to WRITES.  Read the comment on WRITES: "
            "this makes the reported route a rule-M row and stops the "
            "namespace rule being what is tested.")


def test_the_reported_route_lands_in_the_named_half_of_the_table():
    """A row nobody has to write a sentence about is not a row.  These are
    ``?through:`` rows, which ``_named`` routes to KNOWN -- so a new one
    fails ``test_every_write_in_this_lane_is_in_the_table`` and somebody has
    to say what can be at that name."""
    from tests.test_write_census import KNOWN, _named
    rows = census(sources=plant(
        fs_line='remote.subprocess.getoutput("rm -f X")'))
    planted = [r for r in rows if r[1] == "Syncer._candidates"
               and r[2].startswith("?through:")]
    assert planted, "the reported route is not a ?through: row"
    assert _named(planted) == planted, "it fell into the untyped half"
    assert all(r[:4] not in KNOWN for r in planted), (
        "the planted route is pinned in KNOWN, which would make it silent "
        "again")
