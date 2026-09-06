"""ROUND 10: the lane's own destroyers are WRITES, and a parameter is a
binding.

THE LANE'S OWN DESTROYERS WERE NOT IN WRITES.  ``land_beside`` has been a
word in the mutator vocabulary since round 5, so a new call to it is a row.
``_replace_ours``, ``Syncer._drop`` and ``_unlink_after_landing`` -- the
three names in this lane that overwrite or unlink a path -- were not.  Each
resolves to ``<lane>.<name>`` and rule S blessed the CALL on the strength of
the callee's BODY being walked, which is true and is not the point: the body
is fine, and a new CALLER that hands it a name of his is the mistake.

MEASURED by the seventh adversary on the round-9 tip, in a git-archive copy,
nothing of his touched:

    a new self._drop(p) on a verify-failed path       his 100000-byte Outbox
    a new _unlink_after_landing(p, ...) there          file GONE, both ways
    a new _replace_ours(...) at the note name          100000 -> a 300-byte
                                                       note
    census, all three times                            92/92 GREEN
    product suite                                      5 / 5 / 1 tests red

So the product caught all three and this is a TIGHTENING, not a hole: the
three names are in WRITES now, every existing call site has a sentence in
KNOWN, and a new caller fails the census pin -- naming the row -- before a
product test has to.  The tests below plant each shape IN MEMORY through
``census(sources=...)``.

AND ONE LINE IN ``_assigned_in``.  Its docstring says "a locally bound name
is OPAQUE even when a def of the same name exists in the module", and it
walked assignments, loops, with, except and walrus targets -- but not the
def's own PARAMETERS, which live on the node rather than in its body.  So
``def f(dedupe_name, p): dedupe_name(p)`` resolved to ``<lane>.dedupe_name``
and was silence.  It is ``?unresolved`` now, like every other injected
callable.  MEASURED: the fix moved no row of the real table (502 -> 510 is
the eight rows above and nothing else).

Nothing here opens a socket, touches ~/Desktop, speaks to HPCOMPUTER, or
writes to the working tree.
"""
import ast
from pathlib import Path

from tests.test_write_census import KNOWN
from tests.write_census import MODULES, WRITES, _assigned_in, census

REPO = Path(__file__).resolve().parent.parent
FS = "jarvis/foldersync.py"

# Every plant hangs off an anchor asserted present, so a moved anchor fails
# loudly instead of silently planting nothing.
NOTE = '    def note(self, target: Path, reason: str, extra: str = "") -> None:'
VERIFY_FAILED = '''            if got != entry.size:
                self._drop(part)
'''

DESTROYERS = ("_replace_ours", "_drop", "_unlink_after_landing")


def _planted(before_note: str = "", at_verify_failed: str = "",
             tail: str = "") -> dict:
    src = (REPO / FS).read_text()
    if before_note:
        assert NOTE in src, "Syncer.note moved; re-read the source"
        src = src.replace(NOTE, before_note + "\n" + NOTE, 1)
    if at_verify_failed:
        assert VERIFY_FAILED in src, "the verify-failed branch moved"
        src = src.replace(VERIFY_FAILED, VERIFY_FAILED + at_verify_failed, 1)
    return {FS: src + tail}


def _rows(sources=None):
    return census(paths=MODULES, sources=sources)


def _new(sources):
    return set(_rows(sources)) - set(_rows())


# ------------------------------------------------------------ the pin itself
def test_the_three_destroyers_are_words_in_the_vocabulary():
    """Stated as a fact about the census, like run_ssh: drop one of these
    from WRITES and a new caller of it is silence again."""
    for name in DESTROYERS:
        assert name in WRITES, f"{name} is not a write; a new caller is silent"
    assert "land_beside" in WRITES, "the one that was always there"


def test_every_existing_call_of_a_destroyer_has_a_sentence():
    """The tightening's price, paid: seven call sites, each with a line."""
    sites = {(m, f, w, n) for m, f, w, n, _g in _rows() if w in DESTROYERS}
    assert len(sites) == 7, sorted(sites)
    assert sites <= set(KNOWN), sorted(sites - set(KNOWN))


# ------------------------------------------------ the adversary's three shapes
def test_a_new_drop_on_a_verify_failed_path_is_a_row():
    """The exact shape reported: a second self._drop on the path where the
    size did not verify.  Rows are per call site in LINE order, so the plant
    is #3 and the two after it shuffle up to #4 and #5 -- and #5 has no
    sentence, which is the failure that names it."""
    src = _planted(at_verify_failed=(
        "                self._drop(self.paths.outbox / entry.name)\n"))
    drops = [r for r in _rows(src) if r[1] == "Syncer.pull_once"
             and r[2] == "_drop"]
    assert len(drops) == 5, drops
    assert drops[2][4] == "claim:_claim_part", drops   # the plant, by line
    assert (FS, "Syncer.pull_once", "_drop", 5) not in KNOWN


def test_a_new_caller_of_drop_in_a_new_method_is_a_row():
    src = _planted(before_note='''    def _tidy(self, p: Path) -> None:
        self._drop(p)
''')
    new = _new(src)
    assert new == {(FS, "Syncer._tidy", "_drop", 1, "")}, new


def test_a_new_caller_of_unlink_after_landing_is_a_row():
    """On the same verify-failed path, with a dest it never linked to."""
    src = _planted(at_verify_failed=(
        "                _unlink_after_landing(self.paths.outbox / entry.name,"
        " part)\n"))
    new = _new(src)
    assert new == {(FS, "Syncer.pull_once", "_unlink_after_landing", 1,
                    "claim:_claim_part")}, new


def test_a_new_replace_ours_at_the_note_name_is_a_row():
    """The overwrite: _replace_ours at <his name>.jarvis-cannot-send.txt,
    which is a name he can have.  100000 -> 300 bytes on the round-9 tip."""
    src = _planted(before_note='''    def _leave_note(self, target: Path, text: str) -> None:
        _replace_ours(target.with_name(target.name + NOTE_SUFFIX), text)
''')
    new = _new(src)
    assert (FS, "Syncer._leave_note", "_replace_ours", 1, "") in new, new
    assert (FS, "Syncer._leave_note", "_replace_ours", 1) not in KNOWN


def test_each_destroyer_plant_moves_the_count():
    base = len(_rows())
    for name, src in (
        ("_drop", _planted(at_verify_failed=(
            "                self._drop(self.paths.outbox / entry.name)\n"))),
        ("_unlink_after_landing", _planted(at_verify_failed=(
            "                _unlink_after_landing(part, part)\n"))),
        ("_replace_ours", _planted(before_note=(
            "    def _n(self, t: Path) -> None:\n"
            "        _replace_ours(t, 'x')\n"))),
    ):
        assert len(_rows(src)) > base, f"a new {name} caller left the count"


# ------------------------------------------------ a parameter is a binding
def test_a_parameter_named_like_a_def_of_this_module_is_opaque():
    """``def f(dedupe_name, p): dedupe_name(p)`` used to be
    ``<lane>.dedupe_name`` -- blessed by a def it does not call."""
    src = _planted(
        before_note='''    def _sweep(self, stat_key, p: Path) -> None:
        stat_key(p)
''',
        tail='''

def _sweep_module(dedupe_name, p):
    dedupe_name(p)


_SWEEP = lambda name_series, p: name_series(p)  # noqa: E731
''')
    new = {(f, w) for _m, f, w, _n, _g in _new(src)}
    assert ("Syncer._sweep", "?unresolved") in new, new
    assert ("_sweep_module", "?unresolved") in new, new
    assert any(f.startswith("<lambda>") and w == "?unresolved"
               for f, w in new), new


def test_assigned_in_sees_every_kind_of_parameter():
    fn = ast.parse("def f(a, /, b, c=1, *d, e, f=2, **g):\n    pass").body[0]
    assert _assigned_in(fn) == {"a", "b", "c", "d", "e", "f", "g"}
    lam = ast.parse("h = lambda x, *y, **z: 0").body[0].value
    assert _assigned_in(lam) == {"x", "y", "z"}


def test_a_lambda_in_a_default_does_not_leak_its_parameter():
    """The default is evaluated in the ENCLOSING scope, but the lambda's own
    parameter is the lambda's, and SCOPED stops the walk there."""
    fn = ast.parse("def f(x=lambda q: q):\n    pass").body[0]
    assert _assigned_in(fn) == {"x"}


def test_the_parameter_fix_moved_no_row_of_the_real_table():
    """The real lane has no parameter that shadows a def and is then called:
    every ?unresolved row the rule could have added already had a line, so
    the count is derived elsewhere and not typed here."""
    unresolved = [r for r in _rows() if r[2] == "?unresolved"]
    assert {(m, f, w, n) for m, f, w, n, _g in unresolved} <= set(KNOWN)
