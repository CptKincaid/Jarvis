"""TEN MORE ROUTES, none of them listed in six rounds, and the instrument
must see every one WITHOUT a per-pattern entry.

Round 6 inverted the census default from SKIP to REPORT for the CALLEE and
that was right: an unresolvable callee is a row now, which caught ten routes
including seven nobody had listed.  But it left the same branch wearing a new
sign.  ``_classify`` still answered "not a row" for ``<method>.X`` -- an
attribute call on a receiver the resolver cannot type -- whenever X was not
one of the ~60 words in WRITES.  The fixer named that hole itself ("p.shred()")
and argued it was narrow because os, shutil, subprocess, pathlib and tempfile
"are not vouched wholesale".  That is true only of the MODULE-LEVEL spelling
of the import, and the fifth adversary then destroyed a 100000-byte file of
his two independent ways with the row count frozen at 90/90 and the lane suite
at its exact baseline:

  ROUTE A  logging.FileHandler(path, mode="w") truncates a file at a name.
           `logging` was vouched WHOLESALE in HARMLESS_MODULES, so the
           fully-qualified, module-level, entirely un-tricky spelling of a
           file-destroying call was rule S.
  ROUTE B  a LAZY `import subprocess` inside a function, then
           subprocess.getoutput("del ..."). The resolver only walked imports
           at MODULE level, so `subprocess` was an untypable receiver and
           "getoutput" is not in the mutator vocabulary.
  ROUTE C  p.hardlink_to / p.symlink_to / p.lchmod.  No import trick at all,
           no getattr, no subscript -- three real, documented pathlib methods
           that create or change a name, none of them in WRITES.

So the free pass is deleted: silence now requires a RESOLVED, VOUCHED,
FULLY-QUALIFIED name, and every untypable receiver is a row.  These ten tests
are the acceptance bar.  They plant into a MUTATED COPY of the source, in
memory, via ``census(sources=...)``.  Nothing here writes to the working tree,
opens a socket, touches ~/Desktop, or speaks to HPCOMPUTER.
"""
from pathlib import Path

from tests.write_census import MODULES, census

REPO = Path(__file__).resolve().parent.parent
FOLDERSYNC = "jarvis/foldersync.py"

ANCHOR = '    def note(self, target: Path, reason: str, extra: str = "") -> None:'


def _planted(body: str, tail: str = "") -> dict:
    """foldersync.py with `body` spliced in before Syncer.note."""
    src = (REPO / FOLDERSYNC).read_text()
    assert ANCHOR in src, "the anchor moved; re-read the source"
    return {FOLDERSYNC: src.replace(ANCHOR, body + "\n" + ANCHOR, 1) + tail}


def _rows(sources=None):
    return census(paths=MODULES, sources=sources)


def _new(sources):
    """Rows the plant ADDED, against the untouched tree."""
    base = {r[:4] for r in _rows()}
    return [r for r in _rows(sources) if r[:4] not in base]


def _scopes_of(rows, name):
    return [r for r in rows if r[1].endswith(name) or name in r[1]]


# ------------------------------------------------------------------ ROUTE A
def test_a_vouched_module_that_can_name_a_file():
    """MEASURED by the fifth adversary: this destroyed 100000 bytes.

    ``logging.FileHandler(str(p), mode="w")`` opens the path with "w" and
    truncates whatever is there.  There is no indirection anywhere in it --
    it is the plainest spelling Python has -- and it was silent because
    `logging` was vouched WHOLESALE.  The lesson is not "add FileHandler to
    WRITES"; it is that a wholesale module vouch has to answer the question
    its own comment asks, "can this module name a file?", and `logging` can.
    """
    rows = _new(_planted('''    def _rotate(self, p) -> None:
        logging.FileHandler(str(p), mode="w")
''', tail="\nimport logging\n"))
    assert rows, (
        "logging.FileHandler(path, mode='w') censused as NOTHING.  It "
        "truncates a file at a name of his and `logging` was vouched "
        "wholesale in HARMLESS_MODULES")


# ------------------------------------------------------------------ ROUTE B
def test_a_lazy_import_inside_a_function():
    """The resolver walked ``ast.Import`` only at module level, so a name
    imported INSIDE a function was an untypable receiver -- and then the
    method name was all rule M had.  ``subprocess.getoutput`` runs a shell
    command line and is not in the mutator vocabulary."""
    rows = _new(_planted('''    def _reap_b(self, p) -> None:
        import subprocess
        subprocess.getoutput("del C:/Users/h2pey/Desktop/Inbox/*")
'''))
    assert rows, (
        "a lazy `import subprocess` then subprocess.getoutput('del ...') "
        "censused as NOTHING")


def test_a_lazy_import_resolves_exactly_like_a_module_level_one():
    """The structural half of route B, stated as a property.  A module-level
    ``import subprocess`` makes ``subprocess.getoutput`` a rule-R row because
    subprocess is not vouched wholesale.  The lazy spelling must give the
    SAME answer -- not a different one that happens also to be a row."""
    lazy = _new(_planted('''    def _reap_b2(self, p) -> None:
        import subprocess
        subprocess.getoutput("cmd")
'''))
    assert [r[2] for r in lazy] == ["?reach"], (
        "a lazy import does not resolve like a module-level one; got "
        f"{[r[2] for r in lazy]}")


# ------------------------------------------------------------------ ROUTE C
def test_the_pathlib_methods_that_need_no_import_trick():
    """hardlink_to, symlink_to and lchmod are real, documented pathlib
    methods that create or change a name on disk.  None is in WRITES, none
    needs getattr or a subscript or a lazy import, and all three were silent
    on any receiver the resolver could not type -- which is every Path this
    lane holds in a local or a parameter."""
    for method, args in (("hardlink_to", "dest"), ("symlink_to", "dest"),
                         ("lchmod", "0o000")):
        rows = _new(_planted(f'''    def _reap_c_{method}(self, p, dest) -> None:
        p.{method}({args})
'''))
        assert rows, f"p.{method}(...) censused as NOTHING"


# ------------------------------------------------------------------ ROUTE D
def test_a_write_in_a_class_base_expression():
    """A ClassDef's BASES run where the class is written, in the enclosing
    scope, exactly like a decorator expression.  ``_calls_in_scope`` collected
    a nested scope's decorator list and its argument defaults and NOT its
    bases, and ``_scopes`` gave the ClassDef only its own body -- so a call in
    a base expression belonged to no scope at all."""
    rows = _new(_planted('''    def _reap_d(self, p) -> None:
        class _C(self._registry.obliterate(p)):
            pass
        return _C
'''))
    assert rows, "a call in a class base expression censused as NOTHING"


# ------------------------------------------------------------------ ROUTE E
def test_a_write_inside_a_nested_lambda():
    """``_scopes.walk`` appended a Lambda and did not RECURSE into it, and
    ``_calls_in_scope`` skips any SCOPED child -- so a lambda inside a lambda
    was walked by nobody.  A top-level lambda was already seen; the nested
    one was a scope with no owner."""
    rows = _new(_planted('''    def _reap_e(self, paths) -> None:
        f = lambda q: (lambda r: r.scrub())(q)
        for p in paths:
            f(p)
'''))
    inner = [r for r in rows if r[1].count("<lambda>") >= 2]
    assert inner, (
        "a call inside a NESTED lambda belonged to no scope; rows were "
        f"{sorted({r[1] for r in rows})}")


# ------------------------------------------------------------------ ROUTE F
def test_a_mutating_method_on_an_object_returned_by_a_call():
    """The receiver is the RESULT of a call, so there is nothing to type and
    the attribute name is all there is.  Round 6 landed this in
    ``<method>.X`` and then handed it a free pass when X was not one of the
    sixty words it knew."""
    rows = _new(_planted('''    def _reap_f(self, p) -> None:
        self._pick_sink().obliterate(p)
'''))
    assert rows, "a method on a call's return value censused as NOTHING"


# ------------------------------------------------------------------ ROUTE G
def test_a_module_alias_imported_under_a_different_name_inside_a_function():
    """``import shutil as _sh`` inside a function, then
    ``_sh.unpack_archive(src, his_folder)`` -- which writes arbitrarily many
    files at arbitrary names under a directory of his, and is not in WRITES.
    Two holes in series: the lazy import and the free pass."""
    rows = _new(_planted('''    def _reap_g(self, src, dest) -> None:
        import shutil as _sh
        _sh.unpack_archive(str(src), str(dest))
'''))
    assert rows, "a lazily aliased shutil.unpack_archive censused as NOTHING"


# ------------------------------------------------------------------ ROUTE H
def test_argparse_can_name_a_file_too():
    """The other candidate found by re-reading HARMLESS_MODULES against the
    question its own comment asks.  ``argparse.FileType("w")(path)`` OPENS
    the path for writing and truncates it -- argparse is imported at module
    level here, so this is the fully-qualified, un-tricky spelling."""
    rows = _new(_planted('''    def _reap_h(self, p) -> None:
        argparse.FileType("w")(str(p))
'''))
    assert any(r[2] != "?unresolved" for r in rows), (
        "argparse.FileType('w') censused as harmless; argparse can name a "
        f"file.  rows: {rows}")


# ------------------------------------------- MY OWN, 1 of 2
def test_a_primitive_parked_as_a_class_attribute():
    """Nobody has listed this one.  No getattr, no subscript, no lazy import,
    no string building: a real dotted name in the source, ``_Ops.rm(p)``,
    where ``_Ops`` is an ordinary class of this module holding
    ``rm = os.remove``.  ``_Ops`` resolves as a lane class, so the old
    resolver returned ``<method>.rm`` -- and "rm" is not a Python filesystem
    word, so rule M had nothing and rule S waved it through."""
    rows = _new(_planted('''    def _reap_i(self, p) -> None:
        _Ops.rm(p)
''', tail="\n\nclass _Ops:\n    rm = os.remove\n"))
    assert rows, (
        "os.remove parked as a class attribute and called by its dotted "
        "name censused as NOTHING")


# ------------------------------------------- MY OWN, 2 of 2
def test_a_new_method_on_this_lanes_own_injected_transport():
    """The most realistic route of the ten, and the one this lane's design
    invites.  ``self.transport`` is INJECTED -- that is the seam the whole
    product is built on -- so it can never be typed from the source.  A new
    transport method that destroys something on his Windows box is therefore
    an untypable receiver with a name of the author's choosing, which is the
    free pass exactly.  ``self.transport.send`` was censused only because
    "send" happened to be a word somebody had already written down."""
    rows = _new(_planted('''    def _reap_j(self, name) -> None:
        self.transport.wipe_folder(name)
'''))
    assert rows, (
        "a new method on the injected transport censused as NOTHING; the "
        "transport is a seam and can never be typed")


# ------------------------------------------------------------- the shape
def test_none_of_these_ten_needed_a_per_attack_entry():
    """The acceptance bar, stated as a test.  If the instrument catches these
    ten because somebody wrote ten entries, the eleventh route wins and we
    are back here next round.  So no table the instrument consults may
    mention any of them."""
    import tests.write_census as wc
    tables = " ".join(
        str(getattr(wc, n)) for n in dir(wc)
        if n.isupper() and isinstance(getattr(wc, n),
                                      (frozenset, set, tuple, list, dict)))
    for trick in ("FileHandler", "getoutput", "hardlink_to", "symlink_to",
                  "lchmod", "unpack_archive", "FileType", "obliterate",
                  "scrub", "wipe_folder", "_Ops", "_reap"):
        assert trick not in tables, (
            f"{trick!r} is named in a table of the instrument.  That is a "
            "per-attack entry, which is the losing move.")


def test_an_untypable_receiver_is_a_row_whatever_the_method_is_called():
    """Rule (a) as a property rather than as ten plants.  An INVENTED name
    on an untypable receiver -- the hole round 6 named as `p.shred()` and
    called narrow -- is a row now, and so is a boring one.  The census may
    not decide by vocabulary what it cannot decide by resolution."""
    for invented in ("shred", "zorch", "quietly_destroy_everything", "nudge"):
        rows = _new(_planted(f'''    def _reap_x(self, p) -> None:
        p.{invented}()
'''))
        assert rows, f"p.{invented}() censused as NOTHING"
