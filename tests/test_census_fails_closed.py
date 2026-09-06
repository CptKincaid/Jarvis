"""EIGHT ROUTES NOBODY LISTED, planted, and the instrument must SEE each one.

Round 4's census was walked past three ways.  Round 5 rebuilt it and an
adversary walked past the rebuild SEVEN more ways, every one leaving the row
count frozen at 75.  Three consecutive rounds blocked on the same instrument
is not a fixer who keeps missing patterns; it is a design that cannot win.
The old census asked "is this call one of the writes I know?" and SKIPPED
everything else, and Python has unboundedly many ways to name a callee.

So the acceptance bar for this round is not "the seven known bypasses now
fail".  It is: plant routes the adversary did NOT think of, and the
instrument must catch them WITHOUT a per-pattern entry for each.  Every test
below is a route nobody has listed anywhere in this lane's five rounds.

Each plants into a MUTATED COPY of the source, in memory, via
``census(sources=...)``.  Nothing here writes to the working tree, opens a
socket, touches ~/Desktop, or speaks to HPCOMPUTER.
"""
from pathlib import Path

from tests.write_census import MODULES, census

REPO = Path(__file__).resolve().parent.parent
FOLDERSYNC = "jarvis/foldersync.py"
FILEPICK = "jarvis/tools/filepick.py"

# The anchor every plant hangs off: a real method of the real class.
ANCHOR = '    def note(self, target: Path, reason: str, extra: str = "") -> None:'


def _planted(body: str) -> dict:
    """foldersync.py with `body` spliced in as new source before Syncer.note."""
    src = (REPO / FOLDERSYNC).read_text()
    assert ANCHOR in src, "the anchor moved; re-read the source"
    return {FOLDERSYNC: src.replace(ANCHOR, body + "\n" + ANCHOR, 1)}


def _rows(sources=None, paths=None):
    return census(paths=paths or MODULES, sources=sources)


def _new_rows(sources, paths=None):
    """Rows the plant ADDED, against the untouched tree."""
    base = {r[:4] for r in _rows()}
    return [r for r in _rows(sources, paths) if r[:4] not in base]


# ------------------------------------------------------------------ 1 of 8
def test_a_write_in_a_third_module_the_census_does_not_walk():
    """jarvis/tools/filepick.py is imported AND called by both censused
    modules and is censused by neither.  Move the write there and the two
    walked modules are innocent.  No new pattern: the module set must be
    CLOSED under this lane's own first-party imports."""
    src = (REPO / FILEPICK).read_text() + '''

def promote(dest, raw):
    """Planted: filepick is on this lane's import path and not censused."""
    import os
    if dest.exists():
        os.replace(raw, dest)
'''
    rows = _rows({FILEPICK: src}, paths=tuple(MODULES) + (FILEPICK,))
    assert any(r[0] == FILEPICK and r[1] == "promote" for r in rows), (
        "a write in a third module of this lane censused as nothing")
    assert FILEPICK in MODULES, (
        "filepick is imported and called by BOTH censused modules and is "
        "not in MODULES.  The census walks two files; the lane is three.")


# ------------------------------------------------------------------ 2 of 8
def test_a_callee_built_by_string_concatenation():
    """No dotted name exists at all: the method name is assembled at run
    time.  A census keyed on names cannot name this one."""
    rows = _new_rows(_planted('''    def _reap(self, p) -> None:
        import os
        getattr(os, "re" + "move")(p)
'''))
    assert rows, "a callee built from string pieces censused as NOTHING"


# ------------------------------------------------------------------ 3 of 8
def test_a_method_resolved_through_dunder_getattribute():
    """Bypasses `getattr` itself.  ``os.__getattribute__("unlink")(p)``
    never spells a write anywhere in the source text."""
    rows = _new_rows(_planted('''    def _reap2(self, p) -> None:
        import os
        os.__getattribute__("unlink")(p)
'''))
    assert rows, "__getattribute__ dispatch censused as NOTHING"


# ------------------------------------------------------------------ 4 of 8
def test_a_write_inside_a_comprehension():
    """A comprehension has its own scope in Python 3.  If the walker treats
    it as one, every write inside it disappears."""
    rows = _new_rows(_planted('''    def _reap3(self, paths) -> None:
        [p.unlink() for p in paths]
'''))
    assert rows, "a write inside a comprehension censused as NOTHING"


# ------------------------------------------------------------------ 5 of 8
def test_a_write_behind_a_decorator_expression():
    """The decorator EXPRESSION runs at import, and a walker that skips a
    FunctionDef as "its own scope" never looks at its decorator list."""
    rows = _new_rows(_planted('''    @staticmethod
    def _keep(*a):
        return lambda f: f

    @_keep(os.unlink(Path("/tmp/jarvis-decorator-canary")))
    def _decorated(self) -> None:
        return None
'''))
    assert any(r[2] == "unlink" for r in rows), (
        "the ONLY write is in the decorator expression, which runs at "
        f"import; censused as {[r[2] for r in rows] or 'NOTHING'}")


# ------------------------------------------------------------------ 6 of 8
def test_a_bound_method_stored_in_a_list_and_called_by_index():
    """The callee is a Subscript.  There is no name to match."""
    rows = _new_rows(_planted('''    def _reap4(self, p) -> None:
        import os
        ops = [os.remove, os.rmdir]
        ops[0](p)
'''))
    assert rows, "a bound method called by list index censused as NOTHING"


# ------------------------------------------------------------------ 7 of 8
def test_operator_methodcaller():
    """The standard library's own indirection primitive."""
    rows = _new_rows(_planted('''    def _reap5(self, p) -> None:
        import operator
        operator.methodcaller("unlink")(p)
'''))
    assert rows, "operator.methodcaller censused as NOTHING"


# ------------------------------------------------------------------ 8 of 8
def test_the_adversarys_promote_partials_destruction():
    """The round-5 adversary's own measured destruction, replanted.  A
    ``_promote_partials`` globbing *.partial in his Outbox, asking
    ``dest.exists()``, then ``raw.replace(dest)`` -- Path.replace IS
    os.replace and silently overwrites.  MEASURED: his quarterly.xlsx went
    from 100000 bytes to 4, with 11 census tests green and the count
    unchanged at 75."""
    rows = _new_rows(_planted('''    def _promote_partials(self) -> None:
        for raw in self.paths.outbox.glob("*.partial"):
            dest = raw.with_suffix("")
            if dest.exists():
                raw.replace(dest)
'''))
    assert rows, (
        "the adversary's measured destruction censused as NOTHING: "
        "`raw.replace` is a DEAD entry in IGNORE_CALLEES handing a free "
        "pass to any variable named raw")


# ------------------------------------------------------------- the shape
def test_none_of_the_above_needed_a_per_attack_entry():
    """The point of the round, stated as a test.

    If the instrument only catches these eight because somebody added eight
    entries, the ninth route wins and we are back here next round.  So: the
    tables the instrument consults must not mention any of these routes by
    name.  What catches them has to be structural -- an unresolvable callee
    is a ROW, a mutator name is a ROW, a reach out of the walked set is a
    ROW -- not a list of tricks.
    """
    import tests.write_census as wc
    tables = " ".join(
        str(getattr(wc, n)) for n in dir(wc)
        if n.isupper() and isinstance(getattr(wc, n), (frozenset, set,
                                                       tuple, list, dict)))
    for trick in ("methodcaller", "__getattribute__", "_promote_partials",
                  "_reap", "concat"):
        assert trick not in tables, (
            f"{trick!r} is named in a table of the instrument.  That is a "
            "per-attack entry, which is the losing move.")
    assert not hasattr(wc, "IGNORE_CALLEES"), (
        "IGNORE_CALLEES is back.  It was a per-call-site free pass matched "
        "by SPELLING, so a dead entry in it was a live hole -- that is what "
        "let `raw.replace(dest)` destroy a 100000-byte file of his with 11 "
        "census tests green.  Vouch for a resolved name in PROVEN_SAFE "
        "instead, so a local variable cannot inherit the exemption.")

    # And the positive half: the free pass that replaced it can never cover
    # a mutator.  A future reader who quiets a row by adding "os.replace" to
    # PROVEN_SAFE has to get past this.
    for vouched in wc.PROVEN_SAFE:
        assert vouched.rsplit(".", 1)[-1] not in wc.WRITES or vouched in (
            "dataclasses.replace", "re.compile"), (
            f"{vouched} vouches for a name in the mutator vocabulary")


# ------------------------------------------------- the two round-5 defeats
def test_a_module_level_alias_for_a_write():
    """`_rm = os.remove` at module level, then `_rm(p)`.  Round 5 resolved
    the callee to the string "_rm", which is not in WRITES, so it was not a
    row at all.  Resolution runs through the module's real bindings now, so
    the alias reports as the thing it is."""
    body = "    def _reap6(self, p) -> None:\n        _rm(p)\n"
    src = (REPO / FOLDERSYNC).read_text().replace(
        ANCHOR, body + "\n" + ANCHOR, 1) + "\n_rm = os.remove\n"
    rows = _new_rows({FOLDERSYNC: src})
    assert any(r[2] == "remove" for r in rows), (
        f"an alias for os.remove censused as {[r[2] for r in rows]}")


def test_run_ssh_reached_through_a_dict_of_operations():
    """The most dangerous primitive in this lane -- an arbitrary command
    line on his Windows box -- fetched out of a dict so that no dotted name
    exists in the source.  Round 5's census did `if not dotted: continue`."""
    rows = _new_rows(_planted(
        '    def _reap7(self) -> None:\n'
        '        OPS = {"x": remote.run_ssh}\n'
        '        OPS["x"](self.rconf, "del C:/Users/h2pey/Desktop/Inbox/*")\n'))
    assert rows, "a run_ssh `del` on his Windows Inbox censused as NOTHING"


def test_the_walked_set_is_closed_under_this_lanes_own_imports():
    """Rule R as a property rather than as a plant.  Every first-party
    module these modules CALL INTO must itself be walked, or the call is a
    row -- so a write cannot be made invisible by moving house.  filepick
    was imported AND called by both walked modules through five rounds and
    was censused by neither."""
    import ast

    from tests.write_census import (_ancestry, _assigned_in, _classify,
                                    _exports, _first_party, _lane_names,
                                    _method_names, _module_bindings,
                                    _resolve, _scope_imports, _scopes)
    walked = _first_party(MODULES)
    # ROUND 8.  Rule S now asks whether a name is a def or a class of the
    # OTHER walked module, so the hand rebuild has to carry that map too --
    # the same reason it had to grow _scope_imports.  A drifting rebuild is
    # an instrument wearing a new sign, which is the whole point of this
    # test, so _classify takes `exports` with no default and this call is
    # the reason it may not have one.
    # ROUND 9.  And whether that class's ANCESTRY is proven, which is a
    # question about all three modules at once, so _ancestry runs first
    # and _module_bindings and _exports both take its verdict, undefaulted.
    trees = {rel: ast.parse((REPO / rel).read_text()) for rel in MODULES}
    verdicts = _ancestry(trees)
    exports = {rel[:-3].replace("/", "."): _exports(trees[rel], verdicts[rel])
               for rel in MODULES}
    escapes = set()
    for rel in MODULES:
        tree = trees[rel]
        b, m, lane = (_module_bindings(tree, verdicts[rel]),
                      _method_names(tree), _lane_names(tree))
        for _q, calls, owner in _scopes(tree):
            local = _assigned_in(owner)
            # THE SAME BINDINGS THE CENSUS ITSELF USES, scope imports and
            # all.  This test rebuilds the resolver by hand, and a hand
            # rebuild that drifts from the real one is an instrument wearing
            # a new sign -- without the scope imports it missed
            # AssistantConfig.load() entirely and reported one escape where
            # there are two.
            scoped = dict(b, **_scope_imports(owner))
            for node in calls:
                r = _resolve(node.func, scoped, m, lane, local)
                is_row, prim = _classify(r, walked, exports)
                if is_row and prim == "?uncensused":
                    escapes.add(str(r))
    assert escapes == {"jarvis.logs.get_logger",
                       "jarvis.assistant_config.AssistantConfig.load"}, (
        "a call leaves this lane into a first-party module the census does "
        f"not walk: {sorted(escapes)}.  Either walk it (add it to MODULES) "
        "or give it a line in KNOWN -- it may not be silent.")
