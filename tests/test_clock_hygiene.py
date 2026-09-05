"""THE CLOCK GUARD, static half: the suite must not depend on when it runs.

2026-09-05, the incident this file exists to prevent a third time: the very
same commit was 12217 passed / 0 failed at 03:50 and 4 failed / 197 passed
at 09:48. Nothing had changed but the wall clock. Four claim-guard tests
hard-coded "Good evening" while jarvis/brain.py:ground_greeting correctly
regrounds a stale greeting against the hour, so the suite was only green in
the evening -- and a suite that is only green in the evening cannot be used
as a merge gate. It had happened before: a 2026-09-02 note records "1
pre-existing time-of-day failure in test_study_briefing" and moved on. That
write-off is the thing to kill, not the four red lines.

There are two halves to the guard.

  * RUNTIME (tests/conftest.py): --clock-at=HH:MM[:SS] and --clock-tz=ZONE
    re-run the whole suite at a simulated hour or in another zone, and
    tools/clock_guard.sh sweeps them and diffs the verdicts. That catches a
    test whose RESULT moves with the clock.

  * STATIC (this file): it catches the two SHAPES that produce such a test,
    at the moment they are written, for the price of parsing tests/ once
    (~0.2 s). Both shapes are here because both have already bitten:

      1. Reading the wall clock AT IMPORT and asserting against it later.
         The module captures `TODAY = date.today()`; the product reads its
         own clock minutes later when the assertion runs; a run that crosses
         midnight -- or 31 August, or 31 December -- goes red. Three tests
         did this (test_dayreview x2, test_found_calendar_partial_failure).

      2. Building an instant by snapshotting the CURRENT UTC offset and then
         doing wall-clock arithmetic with it:
             (datetime.now().astimezone() + timedelta(days=3)).replace(hour=13)
             datetime.now().astimezone().replace(hour=23, minute=30)
         The offset belongs to today; the instant does not. Across a DST
         change it is an hour wrong, and the product -- which reads real
         local dates -- disagrees. Five tests did this (test_winddown x3,
         test_commander x2). The cure is to do the arithmetic NAIVE and
         attach the zone LAST, so Python resolves the offset that actually
         applies to the day you land on.

A line that genuinely needs one of these shapes says so ON THE LINE:

    NOW = datetime.now().astimezone()   # clock-hygiene: <why>

A waiver is deliberately awkward and deliberately local. It has to be
written next to the code, where the next person to edit that line reads it,
rather than in a list somewhere else that nobody opens -- which is how "1
pre-existing time-of-day failure" survived three days.
"""
from __future__ import annotations

import ast
import pathlib

TESTS = pathlib.Path(__file__).parent

# (receiver, attribute) pairs that read the machine's wall clock.
_CLOCK_CALLS = {
    ("date", "today"), ("datetime", "now"), ("datetime", "today"),
    ("datetime", "utcnow"), ("time", "time"), ("time", "localtime"),
}
_WAIVER = "clock-hygiene:"


def _attr_pair(func: ast.expr):
    """("datetime", "now") for datetime.now / dt.datetime.now, else None."""
    if not isinstance(func, ast.Attribute):
        return None
    base = func.value
    if isinstance(base, ast.Name):
        return (base.id, func.attr)
    if isinstance(base, ast.Attribute):
        return (base.attr, func.attr)
    return None


def _test_files():
    return sorted(p for p in TESTS.glob("*.py") if p.name != "conftest.py")


def _waived(lines: list[str], lineno: int) -> bool:
    """A waiver on the offending line, or anywhere in the run of comment
    lines directly above it.

    The comment block above the line is where the reason belongs when it
    needs a sentence or two, and it is still local -- you cannot read the
    offending line without reading it.
    """
    if 1 <= lineno <= len(lines) and _WAIVER in lines[lineno - 1]:
        return True
    n = lineno - 1                            # walk up the comment block
    while n >= 1 and lines[n - 1].lstrip().startswith("#"):
        if _WAIVER in lines[n - 1]:
            return True
        n -= 1
    return False


def _import_time_clock_reads(tree: ast.Module):
    """Clock calls that run when the module is IMPORTED -- top-level
    statements and class bodies, but not inside a def (which runs later,
    inside the test, where reading the clock is fine)."""
    out = []

    def walk(body):
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue                     # runs at call time, not import
            if isinstance(stmt, ast.ClassDef):
                walk(stmt.body)              # a class body DOES run at import
                continue
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    pair = _attr_pair(node.func)
                    if pair in _CLOCK_CALLS:
                        out.append((node.lineno, ".".join(pair) + "()"))

    walk(tree.body)
    return out


def _snapshotted_offset_arithmetic(tree: ast.Module):
    """`x.astimezone()` used as the BASE of wall-clock arithmetic: either
    `.astimezone().replace(...)` or `.astimezone() +/- delta`."""
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "astimezone"):
            out.append((node.lineno, "astimezone().replace(...)"))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            for side in (node.left, node.right):
                if (isinstance(side, ast.Call)
                        and isinstance(side.func, ast.Attribute)
                        and side.func.attr == "astimezone"):
                    out.append((node.lineno, "astimezone() +/- timedelta"))
    return out


def _scan(finder):
    found = []
    for path in _test_files():
        text = path.read_text()
        lines = text.splitlines()
        tree = ast.parse(text, str(path))
        for lineno, what in finder(tree):
            if not _waived(lines, lineno):
                found.append(f"{path.name}:{lineno}: {what}")
    return found


def test_no_test_module_reads_the_wall_clock_at_import_time():
    """Shape 1. A module-level `date.today()` is compared against a product
    that reads its own clock later; the gap is the bug, and it only shows
    when a run happens to straddle midnight."""
    offenders = _scan(_import_time_clock_reads)
    assert offenders == [], (
        "these read the wall clock when the module is IMPORTED, then assert "
        "against it later -- a run that crosses a date boundary goes red:\n  "
        + "\n  ".join(offenders)
        + "\nFix: pin the day and give the product the matching clock through "
          "its own now=/today= seam, or read the clock inside the test. If the "
          "capture is genuinely deliberate, say why on the line:  "
          "# clock-hygiene: <reason>")


def test_no_test_builds_an_instant_from_a_snapshotted_utc_offset():
    """Shape 2. `.astimezone()` first, arithmetic second, carries today's
    offset onto another day. It is right for ~363 days a year."""
    offenders = _scan(_snapshotted_offset_arithmetic)
    assert offenders == [], (
        "these snapshot the CURRENT UTC offset and then do wall-clock "
        "arithmetic with it, so they are an hour wrong across a DST change:\n  "
        + "\n  ".join(offenders)
        + "\nFix: do the arithmetic on a NAIVE datetime and call .astimezone() "
          "LAST, so the offset that actually applies to that day is used. If "
          "the shape is deliberate, say why on the line:  "
          "# clock-hygiene: <reason>")


def test_the_guard_can_see_both_shapes():
    """The guard's own smoke test: the finders must actually fire. Without
    this, a refactor that broke the AST walk would leave both tests above
    passing vacuously -- green because they found nothing, not because
    there is nothing to find."""
    bad = ast.parse(
        "import datetime\n"
        "TODAY = datetime.date.today()\n"
        "def f(now):\n"
        "    return now.astimezone().replace(hour=9)\n")
    assert [w for _, w in _import_time_clock_reads(bad)] == ["date.today()"]
    assert [w for _, w in _snapshotted_offset_arithmetic(bad)] == \
        ["astimezone().replace(...)"]
    # and a clock read INSIDE a function is not an offence
    ok = ast.parse("import datetime\ndef f():\n    return datetime.date.today()\n")
    assert _import_time_clock_reads(ok) == []
    # the waiver is honoured on the line, and in the comment block above it
    assert _waived(["x = date.today()  # clock-hygiene: why"], 1)
    assert _waived(["# clock-hygiene: why", "# more prose", "x = date.today()"], 3)
    assert not _waived(["# clock-hygiene: why", "", "x = date.today()"], 3)
    assert not _waived(["x = date.today()"], 1)
