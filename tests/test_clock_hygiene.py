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



# ----------------------------------------------------------------------
# Shape 3: a time-of-day read inside a ZONE-PINNED scope.
# ----------------------------------------------------------------------
# Added 2026-09-05 after measuring a hole in the guard's own runtime half.
# `pytest.mark.local_tz(zone)` pins the process's local zone for a file
# whose fixtures are written in a particular place. That is right -- but
# --clock-at simulates an hour BY SHIFTING THE ZONE, so the marker
# overrides it and the pinned scope is invisible to the hour sweep.
#
# Measured on this branch at a real clock of 14:52: a deliberately
# clock-dependent test added to tests/test_arc.py (module-pinned) PASSED
# --clock-at=09:00, 13:00, 21:00, 23:59:30, 03:00 and --clock-tz=UTC --
# every configuration scripts/clock_guard.sh runs -- while being plainly
# red at any real hour outside the afternoon. The runtime half cannot see
# into a pinned scope, so the static half has to.
#
# Scope is kept deliberately narrow, because a guard that cries wolf gets
# waived and then written off, which is the failure this whole branch
# exists to end:
#   * only TIME-OF-DAY reads count. `time.time()` is an absolute epoch
#     scalar and cannot by itself expose a local hour -- test_winddown
#     uses `time.time() + 8 * 3600` to build relative instants for its
#     fakes, which is hour-blind and correct.
#   * only PINNED scope counts. A module-level `pytestmark` pins the whole
#     file; a `@pytest.mark.local_tz` decorator pins only that test, and
#     the rest of the file is still swept normally by --clock-at.
#
# Inside a pinned scope, take the instant from the file's own fixtures
# rather than from the machine. A genuine exception says why on the line,
# `# clock-hygiene: <reason>`, and owes its own evidence that it holds at
# every hour.

# time.time() is excluded: see the note above.
_TIME_OF_DAY_CALLS = _CLOCK_CALLS - {("time", "time")}


def _module_pinned(tree: ast.Module) -> bool:
    """True if a module-level `pytestmark = pytest.mark.local_tz(...)`
    pins the whole file."""
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark"
                   for t in stmt.targets):
            continue
        if any(isinstance(n, ast.Attribute) and n.attr == "local_tz"
               for n in ast.walk(stmt.value)):
            return True
    return False


def _pinned_functions(tree: ast.Module):
    """The functions carrying a @pytest.mark.local_tz decorator."""
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(d, ast.Attribute) and d.attr == "local_tz"
                    or isinstance(d, ast.Call)
                    and any(isinstance(x, ast.Attribute) and x.attr == "local_tz"
                            for x in ast.walk(d.func))
                    for d in n.decorator_list)]


def _time_of_day_reads(node) -> list:
    """Every time-of-day call anywhere under `node`, including inside a
    test body -- which `_import_time_clock_reads` ignores on purpose."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            pair = _attr_pair(n.func)
            if pair in _TIME_OF_DAY_CALLS:
                out.append((n.lineno, ".".join(pair) + "()"))
    return out


def _pinned_clock_reads(tree: ast.Module):
    if _module_pinned(tree):
        return _time_of_day_reads(tree)
    out = []
    for fn in _pinned_functions(tree):
        out.extend(_time_of_day_reads(fn))
    return out


def test_a_zone_pinned_scope_does_not_read_the_time_of_day():
    """Shape 3. The runtime guard is blind inside a local_tz scope, so a
    clock-dependent test written there passes every configuration of
    scripts/clock_guard.sh and still goes red at some other hour."""
    offenders = _scan(_pinned_clock_reads)
    assert offenders == [], (
        "these read the time of day inside a scope that pins the local zone "
        "with pytest.mark.local_tz. --clock-at simulates an hour by shifting "
        "the zone, so the marker overrides it and scripts/clock_guard.sh "
        "CANNOT see these lines -- a clock bug here passes every "
        "configuration the guard runs:\n  "
        + "\n  ".join(offenders)
        + "\nFix: take the instant from the file's own fixtures rather than "
          "from the machine. If the read is genuinely safe at every hour, say "
          "why on the line, with the evidence:  # clock-hygiene: <reason>")


def test_the_guard_can_see_the_zone_pinned_shape():
    """Shape 3's smoke test: the finder must fire, or the test above is
    green because it looked at nothing."""
    module_pinned = ast.parse(
        "import pytest\n"
        "pytestmark = pytest.mark.local_tz('UTC')\n"
        "def test_x():\n"
        "    return datetime.now()\n")
    assert _module_pinned(module_pinned)
    assert [w for _, w in _pinned_clock_reads(module_pinned)] == ["datetime.now()"]
    # a per-test marker pins only that test ...
    per_test = ast.parse(
        "import pytest\n"
        "@pytest.mark.local_tz('UTC')\n"
        "def test_pinned():\n"
        "    return datetime.now()\n"
        "def test_free():\n"
        "    return date.today()\n")
    assert not _module_pinned(per_test)
    assert [w for _, w in _pinned_clock_reads(per_test)] == ["datetime.now()"]
    # ... and an unpinned file is not subject to this shape at all
    plain = ast.parse("def test_x():\n    return datetime.now()\n")
    assert _pinned_clock_reads(plain) == []
    # time.time() is an epoch scalar, not a time of day
    epoch = ast.parse(
        "import pytest\n"
        "pytestmark = pytest.mark.local_tz('UTC')\n"
        "def test_x():\n"
        "    return time.time() + 3600\n")
    assert _pinned_clock_reads(epoch) == []
