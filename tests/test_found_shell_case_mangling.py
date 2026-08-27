"""Regression test for a FOUND DEFECT (failure-mode sweep 2026-08-26).

DEFECT: `jarvis run <command>` lowercases the command before executing it,
so every case-sensitive path, filename and environment variable is
mangled.

jarvis/commander.py::handle() normalises the utterance to lower case for
matching, and `_h_run_shell` (commander.py ~line 1428) then takes the
command straight out of the *normalised* match:

    shell_cmd = m.group(1).strip()      # <- lower-cased text

Commander.handle() already stores `self._raw_text = text` (commander.py
~line 1794) precisely so handlers that need the original casing can get
it -- `_h_run_shell` just doesn't use it.

Measured, with jarvis/context.py's run_shell recorded rather than run:

    spoken   'cat README.md'                    executed 'cat readme.md'
    spoken   'ls /home/hunterp/Jarvis/CLAUDE.md' executed 'ls /home/hunterp/jarvis/claude.md'
    spoken   'echo $HOME/Jarvis'                executed 'echo $home/jarvis'

The last one is the worst: `$home` is not set, so the shell expands it to
the empty string and the command silently operates on the wrong path
instead of failing loudly.

`_h_count_lines` (commander.py ~line 1446) builds its `wc -l` / `find`
command from the same lower-cased group and has the same defect.

FIXED 2026-08-26: both handlers now re-take their group from the original
utterance via `_raw_group()` / `_raw_cmd_text()` (the same trick
`_h_pronounce` already used), falling back to the lower-cased match when the
pattern no longer fits.  This test is no longer xfail.

NOTE on the fixture: the services that are asked "is this yours?" before
the shell handler runs must answer no.  `Commander.handle()` tries the
desktop chain BEFORE the registry, and `_h_workflow` sits above "run shell"
inside it; a bare MagicMock returns a truthy Mock from `parse_action()` and
`get()`, so the utterance was swallowed as a window command / a saved
workflow and never reached the shell handler at all -- the assertion that
used to fail was "the shell command never reached context.run_shell", not
the casing one this test documents.
"""
import pytest


@pytest.fixture
def commander_with_recorded_shell(monkeypatch):
    """A Commander whose context.run_shell records instead of executing."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from jarvis.commander import Commander

    executed = []
    context = SimpleNamespace(
        run_shell=lambda cmd: executed.append(cmd) or "",
        add_exchange=MagicMock(), snapshot=MagicMock(return_value=""),
        get=MagicMock(return_value=""))
    # These two must say "not mine" (see the NOTE above).
    desktop = MagicMock()
    desktop.parse_action.return_value = None
    workflows = MagicMock()
    workflows.get.return_value = None
    services = SimpleNamespace(
        context=context, desktop=desktop, memory=MagicMock(),
        workflows=workflows, brain=MagicMock(), tts=MagicMock(),
        reader=MagicMock(), history=MagicMock(), assistant=None,
        tools=None, router=None, timekeeper=None, notes=None, claude=None,
        approvals=None, alerts=None, discord=None, calendar=None,
        say=MagicMock())
    return Commander(services), executed


@pytest.mark.parametrize("command", [
    "cat README.md",
    "ls /home/hunterp/Jarvis/CLAUDE.md",
    "echo $HOME/Jarvis",
    "grep -i TODO CLAUDE.md",
])
def test_run_shell_preserves_the_original_casing(
        commander_with_recorded_shell, command):
    """The command reaching the shell must be what the user actually said,
    not the lower-cased matching text."""
    commander, executed = commander_with_recorded_shell
    commander.handle(f"jarvis run {command}", source="voice")

    # the handler runs on a background thread; give it a moment
    for _ in range(50):
        if executed:
            break
        import time
        time.sleep(0.02)

    assert executed, "the shell command never reached context.run_shell"
    assert executed[0] == command, (
        f"case mangled: said {command!r}, executed {executed[0]!r}")
