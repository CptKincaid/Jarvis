"""ROUND 3: the Windows helper script itself, read as text.

He installs it by COPYING THE FILE. Nothing here runs PowerShell, nothing
connects to HPCOMPUTER and nothing touches his desktop; this reads
scripts/windows/cast-poll.ps1 off disk and pins the order of two lines.
"""
from __future__ import annotations

import pathlib
import re

SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
          / "scripts" / "windows" / "cast-poll.ps1")


def body() -> str:
    src = SCRIPT.read_text(encoding="utf-8")
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))


def test_the_script_commits_its_receipt_only_after_a_verified_launch():
    """AS SHIPPED at 53537c7 the order was

        if ($newSeq -ne $seq) {
          $seq = $newSeq                <-- the receipt, committed HERE
          if ($verb -eq 'show-spark') { ... Start-Process ... }

    under ``$ErrorActionPreference = 'SilentlyContinue'``, so the sequence
    Jarvis reads back as a RECEIPT was committed before the launch was
    attempted and whether or not it worked. RustDesk not being at the
    hard-coded path is the ordinary way that happens.
    """
    text = body()
    commit = text.index("$seq = $newSeq")
    launch = text.index("Start-Process")
    print("\n  $seq = $newSeq at char %d, Start-Process at char %d"
          % (commit, launch))
    assert launch < commit, ("the receipt is still committed before the "
                             "launch is attempted")


def test_the_script_does_not_swallow_the_launch_error():
    text = body()
    assert "-ErrorAction Stop" in text, "Start-Process must raise"
    assert re.search(r"try\s*\{[^}]*Start-Process", text, re.S), \
        "the launch must be inside a try"
    assert "Test-Path" in text, "a missing rustdesk.exe must be noticed"


def test_the_script_reports_the_failure_back_to_jarvis():
    from jarvis import castview as cv
    text = body()
    assert "fail" in text and "failseq" in text
    for code in cv.HELPER_FAILS:
        assert "'%s'" % code in text, code


def test_the_script_still_sends_nothing_but_an_index_and_a_layout():
    """The privacy line this file has always held: no cursor coordinate,
    no window handle, no title, no process name and no path leaves that
    machine. A failure REASON is a code from a closed set for the same
    reason a verb is."""
    text = body()
    sends = re.search(r"\$body\s*=\s*@\{(.*?)\}", text, re.S)
    assert sends is not None
    fields = set(re.findall(r"(\w+)\s*=", sends.group(1)))
    print("\n  the helper sends: %s" % sorted(fields))
    assert fields == {"mon", "layout", "seq", "fail", "failseq"}, fields
    assert "$_.Exception.Message" not in text, \
        "a Windows error string may not cross the wire"
    assert "$RustDesk\"" not in text.replace("$RustDesk = ", "")


def test_the_script_cannot_retry_a_failed_launch_for_ever():
    text = body()
    assert "$failSeq" in text, "a failed sequence must be remembered"
    assert re.search(r"\$newSeq\s*-ne\s*\$failSeq", text), \
        "the loop must not relaunch a sequence it already failed"


def test_the_verbs_it_acts_on_are_still_a_closed_set_of_literals():
    from jarvis import castview as cv
    text = body()
    for verb in cv.VERBS:
        if verb == cv.VERB_NONE:
            continue
        assert "'%s'" % verb in text, verb
    assert "Invoke-Expression" not in text
    assert "iex" not in text
    assert "$verb)" not in text and "& $verb" not in text
