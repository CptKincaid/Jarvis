"""ROUND 4: a STOP that did not happen may not become a receipt.

He installs the Windows helper by COPYING THE FILE. Nothing here runs
PowerShell, nothing connects to HPCOMPUTER and nothing touches his
desktop: this reads scripts/windows/cast-poll.ps1 off disk as text.

AS SHIPPED AT d4def0b:

    function Stop-Cast {
      param([int]$ProcId)
      if ($ProcId -ne 0) {
        try { Stop-Process -Id $ProcId -Force -ErrorAction Stop } catch { }
      }
      return 0
    }

Round 3 put ``$ErrorActionPreference = 'Stop'`` at the top of the file and
made the LAUNCH report its failures as a code, and then left one empty
catch behind. It is the one that matters most for the sentence Jarvis
says: ``return 0`` is unconditional, so a viewer the script could not kill
is orphaned on his middle monitor while ``$castPid`` is cleared, ``$ok``
stays true and the receipt tells Jarvis the stop happened. The same
swallow also lets a show-spark relaunch a second viewer on top of one it
could not kill.
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


def stop_cast() -> str:
    text = body()
    start = text.index("function Stop-Cast")
    end = text.index("function Start-Viewer")
    return text[start:end]


def test_stop_cast_no_longer_swallows_a_failed_kill():
    """The empty catch is the defect. A ``catch { }`` around the only
    thing that stops a viewer is a promise that it worked."""
    fn = stop_cast()
    print("\n  Stop-Cast body:\n%s" % fn)
    assert not re.search(r"catch\s*\{\s*\}", fn), \
        "Stop-Cast still swallows the failure in an empty catch"
    assert "$script:fail" in fn, \
        "a failed Stop-Process must set a code, like a failed launch does"


def test_a_stop_that_did_not_happen_cannot_return_success():
    """``return 0`` unconditionally is what makes the receipt a lie: 0
    means 'nothing of ours is running', and the caller believes it."""
    fn = stop_cast()
    returns = re.findall(r"return\s+(\S+)", fn)
    print("\n  Stop-Cast returns: %s" % returns)
    assert "$ProcId" in returns, \
        "a viewer that would not die must be reported as still ours"
    # The FALL-THROUGH is what made it a lie: whatever happened above, the
    # function ended by telling the caller nothing of ours was running.
    assert returns[-1] == "$ProcId", returns
    assert not re.search(r"return\s+0\s*\n\s*\}\s*$", fn.rstrip()), \
        "the function still ends by claiming success unconditionally"
    # ...and every 0 it does return is guarded by an actual look at the
    # process, not by nothing at all.
    for chunk in fn.split("return 0")[:-1]:
        tail = "\n".join(chunk.strip().splitlines()[-2:])
        assert ("Get-Process" in tail or "-eq 0" in tail), tail


def test_the_stop_verb_only_receipts_when_the_viewer_is_actually_gone():
    """The main loop must read Stop-Cast's answer instead of assuming it."""
    text = body()
    branch = text[text.index("elseif ($verb -eq 'stop')"):]
    branch = branch[:branch.index("if ($ok)")]
    print("\n  the stop branch:\n%s" % branch)
    assert "$ok" in branch, "the stop branch never sets $ok"
    assert re.search(r"\$ok\s*=\s*\(\$castPid\s*-eq\s*0\)", branch), \
        "the receipt must depend on the viewer actually being gone"


def test_a_show_does_not_stack_a_second_viewer_on_one_it_could_not_kill():
    """Two RustDesk windows on his middle monitor is worse than none."""
    text = body()
    branch = text[text.index("if ($verb -eq 'show-spark')"):]
    branch = branch[:branch.index("elseif")]
    print("\n  the show branch:\n%s" % branch)
    assert "Start-Viewer" in branch
    stop_at = branch.index("Stop-Cast")
    start_at = branch.index("Start-Viewer")
    guard = branch[stop_at:start_at]
    assert re.search(r"\$castPid\s*-(ne|eq)\s*0", guard), \
        "nothing checks whether the old viewer actually died before " \
        "starting a new one"


def test_the_new_failure_code_is_in_the_closed_set_and_has_a_line():
    """A failure REASON is a code from a closed set for the same reason a
    verb is: a free-text Windows error carries paths and window titles."""
    from jarvis import castview as cv
    text = body()
    assert "stop-failed" in cv.HELPER_FAILS, cv.HELPER_FAILS
    assert "'stop-failed'" in text, "the script must send the code"
    assert "stop-failed" in cv.HELPER_FAIL_LINES, \
        "a code Jarvis can receive must be a sentence Jarvis can say"
    print("\n  HELPER_FAILS = %s" % (cv.HELPER_FAILS,))
    print("  says: %r" % cv.HELPER_FAIL_LINES["stop-failed"])


def test_the_helper_still_sends_nothing_but_an_index_and_a_layout():
    """Unchanged, and re-pinned because this round touches the sender."""
    text = body()
    sends = re.search(r"\$body\s*=\s*@\{(.*?)\}", text, re.S)
    assert sends is not None
    fields = set(re.findall(r"(\w+)\s*=", sends.group(1)))
    print("\n  the helper sends: %s" % sorted(fields))
    assert fields == {"mon", "layout", "seq", "fail", "failseq"}, fields
    assert "$_.Exception.Message" not in text
    for leak in ("$PWD", "Get-Location", "$env:USERNAME", "MainWindowTitle",
                 "$p.Path", "ProcessName"):
        assert leak not in text, leak


def test_installing_it_is_still_copying_one_file():
    """His ruling: there is nothing to run and nothing to answer."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "Copy to" in src
    assert "Install-" not in src and "New-Service" not in src
    assert "Invoke-Expression" not in src and "iex" not in src
    print("\n  one file, copied: %d lines" % len(src.splitlines()))
