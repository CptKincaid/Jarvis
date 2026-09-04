"""The desktop firewall in tests/conftest.py.

2026-09-04 17:21. A probe ran a corpus of "remember ..." sentences through
Commander.handle as a plain script. One of them, "jarvis remember heather's
face", is a FACE ENROLMENT, and that rung hands its command line to xclip
on :1 -- so the probe overwrote his clipboard while he was working. Another,
"jarvis commit this to memory: ...", matched the bare-substring
QUICK_COMMANDS trigger "commit" and ran `git add -A` in ~/vss_env through a
shell.

These tests drive the SAME two sentences through the same Commander and
prove that, under pytest, neither forks: the attempt is recorded, refused,
and the caller degrades the way it does when xclip is simply missing.

Every canary here depends on the `desktop_attempts` fixture. Before the
firewall existed that fixture did not either, so the red run of this file
ERRORS AT SETUP -- it never reached the sentence, and never ran xclip.
"""
import subprocess

import pytest

import jarvis.commander as commander
import jarvis.desktop as desktop
import jarvis.enrolentry as enrolentry
from jarvis.commander import QUICK_COMMANDS
from tests.test_commander import cmdr, services  # noqa: F401  (fixtures)


def _programs(attempts):
    return [(owner, kind, program) for owner, kind, program, _argv in attempts]


# ------------------------------------------------- the two incident sentences
def test_remember_heathers_face_reaches_no_xclip(cmdr, services,  # noqa: F811
                                                 desktop_attempts):
    """The 17:21 sentence. It IS a face enrolment (that routing is right);
    what must not happen is the xclip write at the end of it."""
    res = cmdr.handle("jarvis remember heather's face")
    assert res.handled
    services.memory.remember.assert_not_called()
    xclip = [a for a in _programs(desktop_attempts) if a[2] == "xclip"]
    assert xclip, f"the enrolment hand-over never asked for xclip: {desktop_attempts}"
    assert all(owner == "jarvis.enrolentry" for owner, _k, _p in xclip)
    # The refusal is what xclip-missing looks like to the caller, so the
    # reply tells the truth: it is in the transcript and NOT on the clipboard.
    assert enrolentry.CLIP_FAILED_LINE in (res.reply or "")
    assert enrolentry.CLIP_OK_LINE not in (res.reply or "")


def test_commit_this_to_memory_runs_no_shell(cmdr, services,  # noqa: F811
                                             desktop_attempts):
    """The other 17:21 sentence. At 7539478 the word "commit" inside it fires
    QUICK_COMMANDS["commit"] (a shell string with `git add -A` in it). Under
    the firewall the shell is refused and the attempt recorded; the rung
    branch then retires this into "no attempt at all" once the memory
    family sits above the quick-command rung."""
    cmdr.handle("jarvis commit this to memory: i graduate december 10th 2026")
    shells = [argv for owner, kind, _p, argv in desktop_attempts
              if owner == "jarvis.commander" and kind == "run"]
    assert shells == [QUICK_COMMANDS["commit"]], desktop_attempts
    assert "git add -A" in shells[0]


# ------------------------------------------------------------ the seams
def test_enrolentry_default_runner_is_refused_and_recorded(desktop_attempts):
    assert enrolentry.to_clipboard("sentinel-that-must-never-land") is False
    assert _programs(desktop_attempts) == [("jarvis.enrolentry", "run", "xclip")]


def test_the_refusal_names_the_module_and_the_program(desktop_attempts):
    with pytest.raises(OSError) as exc:
        commander.subprocess.run("cd /tmp && true", shell=True)
    msg = str(exc.value)
    assert "jarvis.commander" in msg and "'cd'" in msg
    assert "real_subprocess" in msg, "the message must say how to opt in"
    assert _programs(desktop_attempts) == [("jarvis.commander", "run", "cd")]


def test_desktop_xdotool_is_refused_and_the_caller_degrades(desktop_attempts):
    desktop.press_key("ctrl+f")          # catches, logs, returns
    assert _programs(desktop_attempts) == [("jarvis.desktop", "run", "xdotool")]
    with pytest.raises(OSError):
        desktop.subprocess.Popen(["xdg-open", "https://example.invalid"])
    assert _programs(desktop_attempts)[-1] == ("jarvis.desktop", "Popen", "xdg-open")


def test_the_quick_command_shell_is_refused_inline(cmdr, services,  # noqa: F811
                                                   desktop_attempts):
    cmdr.handle("jarvis check disk")
    assert [p for _o, _k, p, _a in desktop_attempts] == ["cd"] or \
        [a for _o, _k, _p, a in desktop_attempts] == [QUICK_COMMANDS["check disk"]]


def test_import_time_default_runners_are_rebound(desktop_attempts):
    """reader._xclip binds subprocess.run as a DEFAULT ARGUMENT at import,
    so swapping the module attribute alone would leave it live."""
    import jarvis.reader as reader
    assert reader._xclip("clipboard") == ""
    assert _programs(desktop_attempts) == [("jarvis.reader", "run", "xclip")]


# ------------------------------------------------------ the ways through
def test_a_global_monkeypatch_of_subprocess_run_is_forwarded(monkeypatch,
                                                            desktop_attempts):
    """tests/test_oracle.py, test_remote_files_and_shell.py and others
    monkeypatch subprocess.run / Popen on the real module; the proxy hands
    the call to that fake instead of refusing it."""
    seen = []
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **kw: seen.append(argv) or
                        subprocess.CompletedProcess(argv, 0, "", ""))
    r = commander.subprocess.run(["xclip", "-o"], capture_output=True)
    assert r.returncode == 0 and seen == [["xclip", "-o"]]
    assert desktop_attempts == []


def test_a_per_module_monkeypatch_still_works(monkeypatch, cmdr, services,  # noqa: F811
                                              desktop_attempts):
    """tests/test_commander.py's own pattern: monkeypatch.setattr(
    commander.subprocess, "run", fake)."""
    ran = {}
    monkeypatch.setattr(commander.subprocess, "run",
                        lambda cmd, **kw: ran.setdefault("cmd", cmd) and
                        subprocess.CompletedProcess(cmd, 0, "ok\n", ""))
    cmdr.handle("jarvis check gpu")
    assert "nvidia-smi" in ran["cmd"]
    assert desktop_attempts == []


@pytest.mark.real_subprocess("jarvis.context")
def test_the_marker_restores_the_real_module_for_one_test():
    import jarvis.context as context
    assert context.subprocess is subprocess


def test_and_without_the_marker_it_is_the_proxy():
    import jarvis.context as context
    assert context.subprocess is not subprocess
    assert context.subprocess.CompletedProcess is subprocess.CompletedProcess


def test_the_core_modules_are_all_wrapped():
    import jarvis.workflows as workflows
    for mod in (commander, enrolentry, desktop, workflows):
        assert mod.subprocess is not subprocess, mod.__name__
        assert mod.subprocess.PIPE is subprocess.PIPE
        assert mod.subprocess.TimeoutExpired is subprocess.TimeoutExpired
        assert issubclass(mod.subprocess.Popen, subprocess.Popen)


# ------------------------------------------- the two tmp_path allowances
def test_git_in_a_tmp_path_repo_is_let_through(tmp_path, desktop_attempts):
    """tests/test_context.py and test_standup.py build repos in tmp_path
    and probe them with context._git: a repo pytest made is not his."""
    import jarvis.context as context
    repo = tmp_path / "repo"
    repo.mkdir()
    assert context._git(repo, "init", "-q") == ""      # ran: no refusal
    assert (repo / ".git").is_dir()
    assert desktop_attempts == []


def test_git_outside_tmp_path_is_refused(desktop_attempts):
    import jarvis.context as context
    assert context._git("/home/hunterp/Jarvis", "status", "--short") == ""
    assert _programs(desktop_attempts) == [("jarvis.context", "run", "git")]


def test_a_shell_string_never_uses_the_git_door(desktop_attempts):
    """`cd ~/vss_env && git add -A` is a string, not an argv list."""
    with pytest.raises(OSError):
        commander.subprocess.run("git status", shell=True, cwd="/tmp")
    assert _programs(desktop_attempts) == [("jarvis.commander", "run", "git")]


def test_a_script_written_into_tmp_path_is_let_through(tmp_path, desktop_attempts):
    """tests/test_web_route.py runs a stand-in `claude` it wrote itself."""
    import jarvis.brain as brain
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\necho hi\n")
    exe.chmod(0o755)
    proc = brain.subprocess.Popen([str(exe)], stdout=brain.subprocess.PIPE, text=True)
    assert proc.communicate()[0].strip() == "hi"
    assert desktop_attempts == []
