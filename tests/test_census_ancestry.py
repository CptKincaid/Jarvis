"""A CLASS IS NOT ITS BODY.  Rule S blessed one as if it were.

ROUND 9.  Rule S grants silence to a bare def or class of a walked module on
one committed sentence: "its body is a scope of its own and is censused in
its own module".  That is TRUE OF A DEF and FALSE OF A CLASS THAT SUBCLASSES
A WRITER.  What runs when you write ``_Sub(path, "w")`` is not ``_Sub``'s
body -- which is ``pass`` -- but the ``__init__`` it INHERITED, from a module
the walk never reads.

MEASURED HERE on the round-8 tip (0cfad85), in a git-archive copy under the
session scratchpad, nothing of his touched:

    class _Sub(logging.FileHandler): pass     planted in remote.py
    class _Kill(subprocess.Popen): pass       remote.py already imports
                                              subprocess -- no new import
    class _Local(logging.FileHandler): pass   the same, local to foldersync
    remote._Sub(p, "w")                       quarterly-a.xlsx 100000 -> 0
    remote._Kill(["rm", "-f", p])             quarterly-b.xlsx GONE
    _Local(p, "w")                            quarterly-c.xlsx 100000 -> 0
    census                                    502 rows, delta ZERO, not one
                                              row naming _Sub, _Kill, _Local
    census suite on that planted copy         63 passed

THE FIX IS ONE RULE, NOT A LIST.  A class may be blessed only when
everything that runs to construct it is already vouched for: every base,
to the root, and no class keyword.  A base is proven when it resolves
through a real binding to object, a builtin exception, a harmless module's
own name, a name in PROVEN_SAFE, or a class STATEMENT of a walked module
that is itself proven -- to a fixpoint, across the walked set.  Anything
else is a reason, and a reason makes the class a row at every call site,
spelled ``?inherits:<name>``.  See :func:`tests.write_census._ancestry`.

Everything below is a plant IN MEMORY through ``census(sources=...)``.
Nothing here opens a socket, touches ~/Desktop, speaks to HPCOMPUTER, or
writes to the working tree.
"""
import re
from pathlib import Path

import pytest

from tests import write_census as wc
from tests.write_census import MODULES, census

REPO = Path(__file__).resolve().parent.parent
FS = "jarvis/foldersync.py"
RM = "jarvis/tools/remote.py"
SRC = {m: (REPO / m).read_text() for m in MODULES}

# The anchors every plant hangs off, each asserted present so a moved anchor
# fails loudly instead of silently planting nothing.
FS_SCOPE = """    def _candidates(self) -> list:
        out = []
        try:"""
FS_IMPORTS = "from jarvis.tools import filepick, remote"


def plant(fs_line="", fs_import="", fs_top="", rm_top=""):
    """This lane's source with a few lines added, as TEXT.

    ``fs_line``   one statement (or an indented block) at the top of
                  ``Syncer._candidates``, the call site;
    ``fs_import`` a line after foldersync's import block;
    ``fs_top``    text appended to the END of foldersync -- a class
                  statement at module level;
    ``rm_top``    text appended to the END of remote.py.
    Nothing is written.
    """
    out = dict(SRC)
    text = out[FS]
    if fs_import:
        assert FS_IMPORTS in text, "the foldersync import block moved"
        text = text.replace(FS_IMPORTS, FS_IMPORTS + "\n" + fs_import, 1)
    if fs_line:
        assert FS_SCOPE in text, "Syncer._candidates moved"
        text = text.replace(FS_SCOPE, FS_SCOPE.replace(
            "        try:", f"        {fs_line}\n        try:"), 1)
    if fs_top:
        text = text + "\n\n" + fs_top
    out[FS] = text
    if rm_top:
        out[RM] = out[RM] + "\n\n" + rm_top
    return out


def rows_for(**kw):
    return set(census(sources=plant(**kw)))


@pytest.fixture(scope="module")
def clean():
    return set(census())


SUB = "import logging\n\n\nclass _Sub(logging.FileHandler):\n    pass\n"
KILL = "class _Kill(subprocess.Popen):\n    pass\n"


# ------------------------------------------------------- the two reported
def test_a_filehandler_subclass_in_a_walked_module_is_a_row(clean):
    """THE REPORTED ROUTE.  Measured 100000 bytes -> 0 with the census at
    502, delta zero."""
    new = rows_for(rm_top=SUB, fs_line='remote._Sub("X", "w")') - clean
    assert new, "SILENT: a class of remote.py that IS a FileHandler"
    assert {r[2] for r in new} == {"?inherits:_Sub"}, sorted(new)


def test_a_popen_subclass_needs_no_new_import_and_is_a_row(clean):
    """remote.py already says `import subprocess`; the subclass deletes a
    file through an arbitrary command line.  Measured GONE, delta zero."""
    new = rows_for(rm_top=KILL,
                   fs_line='remote._Kill(["rm", "-f", "X"])') - clean
    assert new, "SILENT: a class of remote.py that IS a Popen"
    assert {r[2] for r in new} == {"?inherits:_Kill"}, sorted(new)


def test_the_same_class_local_to_the_calling_module_is_a_row(clean):
    """Rule S's other door: `<lane>.<class>` of the CURRENT module."""
    new = rows_for(fs_import="import logging",
                   fs_top="class _Local(logging.FileHandler):\n    pass\n",
                   fs_line='_Local("X", "w")') - clean
    assert {r[2] for r in new} == {"?inherits:_Local"}, sorted(new)


def test_a_class_nested_inside_a_function_is_a_row_too(clean):
    """The lane set blesses a nested def module-wide, and used to bless a
    nested class the same way."""
    new = rows_for(
        fs_import="import logging",
        fs_line=("class _Nested(logging.FileHandler):\n"
                 "            pass\n"
                 '        _Nested("X", "w")')) - clean
    assert {r[2] for r in new} == {"?inherits:_Nested"}, sorted(new)


# ---------------------------------------- three spellings nobody listed
MINE = [
    # A base reached through an ALIAS.  The class statement names no
    # module at all; `_Base` is an ordinary module-level name.
    ("a base reached through an alias",
     dict(rm_top=("import logging\n_Base = logging.FileHandler\n\n\n"
                  "class _Via(_Base):\n    pass\n"),
          fs_line='remote._Via("X", "w")'),
     "?inherits:_Via"),
    # A MIXIN TWO LEVELS UP.  The class that is called has an innocent
    # base, whose base has an innocent base, whose base is a Popen.
    ("a mixin two levels up",
     dict(rm_top=("class _Mixin(subprocess.Popen):\n    pass\n\n\n"
                  "class _Mid(_Mixin):\n    pass\n\n\n"
                  "class _Leaf(_Mid):\n    pass\n"),
          fs_line='remote._Leaf(["rm", "-f", "X"])'),
     "?inherits:_Leaf"),
    # A base that is itself a WALKED CLASS whose own base writes: the
    # writer is in remote.py, the subclass in foldersync, and the old rule
    # blessed both ends -- `remote._W` is a class statement of a walked
    # module, `_L` is a class of this one.
    ("a walked class whose own base writes",
     dict(rm_top="class _W(subprocess.Popen):\n    pass\n",
          fs_top="class _L(remote._W):\n    pass\n",
          fs_line='_L(["rm", "-f", "X"])'),
     "?inherits:_L"),
    # And two more of the same one idea, for the record.
    ("the from-import spelling of a dirty base",
     dict(rm_top=SUB,
          fs_import="from jarvis.tools.remote import _Sub as _S",
          fs_top="class _T(_S):\n    pass\n",
          fs_line='_T("X", "w")'),
     "?inherits:_T"),
    ("a module-level alias to a dirty class, called directly",
     dict(rm_top=SUB, fs_top="_Mk = remote._Sub\n",
          fs_line='_Mk("X", "w")'),
     "?inherits:_Sub"),
]


@pytest.mark.parametrize("label,kw,prim", MINE, ids=[m[0] for m in MINE])
def test_an_ancestry_spelling_nobody_listed_is_a_row(label, kw, prim, clean):
    new = rows_for(**kw) - clean
    assert new, f"SILENT: {label}"
    assert {r[2] for r in new} == {prim}, (label, sorted(new))


# ------------------------------------------------------ the other direction
STAYS_SILENT = [
    ("no base at all",
     dict(fs_top="class _Plain:\n    pass\n", fs_line="_Plain()")),
    ("object",
     dict(fs_top="class _Obj(object):\n    pass\n", fs_line="_Obj()")),
    ("a builtin exception",
     dict(fs_top="class _Err(ValueError):\n    pass\n",
          fs_line='_Err("x")')),
    ("a proven class of this module",
     dict(fs_top="class _Rec(SyncConfig):\n    pass\n", fs_line="_Rec()")),
    ("a proven class of this module, through an alias",
     dict(fs_top="_A = SyncConfig\n\n\nclass _ViaA(_A):\n    pass\n",
          fs_line="_ViaA()")),
    ("a proven class of a walked module",
     dict(fs_top="class _Sub2(remote.RemoteConfig):\n    pass\n",
          fs_line="_Sub2()")),
    ("a harmless module's own class",
     dict(fs_import="import typing",
          fs_top="class _T(typing.NamedTuple):\n    x: int\n",
          fs_line="_T(1)")),
    ("a class in PROVEN_SAFE",
     dict(fs_top="class _P(Path):\n    pass\n", fs_line='_P("x")')),
]


@pytest.mark.parametrize("label,kw", STAYS_SILENT,
                         ids=[s[0] for s in STAYS_SILENT])
def test_a_class_with_a_proven_ancestry_stays_silent(label, kw, clean):
    """The rule is 'proven', not 'none': a table that rows every class
    drowns, and a drowned table is one nobody reads."""
    new = rows_for(**kw) - clean
    assert not new, (f"{label}: a class with a proven ancestry became a "
                     f"row; the rule is too tight: {sorted(new)}")


# ------------------------------------------------------------ fail closed
def test_a_class_whose_base_is_unresolved_is_a_row(clean):
    """Not "a base we know is bad" -- a base we cannot follow at all."""
    new = rows_for(fs_top="class _U(_never_bound_anywhere):\n    pass\n",
                   fs_line="_U()") - clean
    assert {r[2] for r in new} == {"?inherits:_U"}, sorted(new)


def test_a_class_with_a_class_keyword_is_a_row(clean):
    """``metaclass=`` is exactly "who constructs this"."""
    new = rows_for(fs_top="class _M(metaclass=type):\n    pass\n",
                   fs_line="_M()") - clean
    assert {r[2] for r in new} == {"?inherits:_M"}, sorted(new)


def test_a_class_whose_base_is_a_call_is_two_rows(clean):
    """The call in the base list was already a row (round 7, route D); the
    class built on its result is one now as well."""
    new = rows_for(fs_top="class _C(_make()):\n    pass\n",
                   fs_line="_C()") - clean
    assert {r[2] for r in new} == {"?unresolved", "?inherits:_C"}, sorted(new)


def test_a_class_whose_base_is_a_def_is_a_row(clean):
    """`remote.scp_path` is a def, so a call to it is silence; a class
    built on it is not a class built on anything proven."""
    new = rows_for(fs_top="class _D(remote.scp_path):\n    pass\n",
                   fs_line="_D()") - clean
    assert {r[2] for r in new} == {"?inherits:_D"}, sorted(new)


# ------------------------------------------------------------- the shape
def test_the_structure_catches_them_and_not_the_vocabulary(clean):
    """Every route above is a ``?inherits:`` row.  NOT ONE is caught by its
    name being in WRITES, and no table of the instrument names any of
    them, so the next spelling is caught by the same rule."""
    by_name = []
    for label, kw, _p in MINE:
        new = rows_for(**kw) - clean
        if not all(r[2].startswith("?inherits:") for r in new):
            by_name.append(f"{label}: {sorted(r[2] for r in new)}")
    assert not by_name, ("caught by vocabulary, not by ancestry:\n  "
                         + "\n  ".join(by_name))
    tables = " ".join(str(getattr(wc, n)) for n in dir(wc)
                      if n.isupper() and isinstance(
                          getattr(wc, n), (frozenset, set, tuple, list, dict)))
    for trick in ("_Sub", "_Kill", "_Local", "_Mixin", "_Leaf", "_Via",
                  "FileHandler", "logging"):
        assert trick not in tables, f"{trick!r} is named in a table"


def test_an_inherits_row_lands_in_the_named_half(clean):
    """A row nobody has to write a sentence about is not a row.  These go
    to KNOWN, so a new one fails
    ``test_every_write_in_this_lane_is_in_the_table`` until somebody says
    what the class inherits and why that cannot name a file of his."""
    from tests.test_write_census import KNOWN, _named
    rows = census(sources=plant(rm_top=SUB, fs_line='remote._Sub("X", "w")'))
    planted = [r for r in rows if r[2] == "?inherits:_Sub"]
    assert planted and _named(planted) == planted
    assert all(r[:4] not in KNOWN for r in planted)


def test_the_lanes_own_eleven_classes_are_all_proven():
    """The price of the rule on the real source is ZERO rows, and this is
    why: no class in this lane has a base.  If one grows a base the census
    cannot vouch for, this fails with the reason in it."""
    import ast
    trees = {rel: ast.parse((REPO / rel).read_text()) for rel in MODULES}
    verdicts = wc._ancestry(trees)
    unproven = {f"{rel}:{name}": why for rel, v in verdicts.items()
                for name, why in v.items() if why}
    assert not unproven, unproven
    assert sum(len(v) for v in verdicts.values()) >= 11


def test_the_count_in_the_prose_is_derived_not_typed():
    """Round 8's report said 490 rows.  The tables it shipped said 502.  A
    number in prose that nothing checks is the kind of number this project
    has been damaged by twice, so the one at the top of write_census.py is
    checked here against the census AND against the two tables that pin
    it -- derived twice, typed once."""
    from tests.test_write_census import KNOWN, UNTYPED_SITES
    m = re.search(r"TODAY'S TABLE IS (\d+) ROWS", wc.__doc__)
    assert m, "the count sentence is gone from the top of write_census.py"
    said = int(m.group(1))
    assert said == len(census()) == len(KNOWN) + len(UNTYPED_SITES), (
        f"prose says {said}, the census derives {len(census())}, the "
        f"tables pin {len(KNOWN) + len(UNTYPED_SITES)}")
