"""jarvis/relaunch.py -- the Restart button's helper, driven through seams.

Nothing here touches a real process: every pid is a number handed to a fake
process table, every signal lands in a list, every sleep advances a fake
clock, and every launch is a recorded popen. The live Jarvis (pid in
/tmp/vss_voice/jarvis.pid) is never looked at, not even with kill(pid, 0).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from jarvis import relaunch

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ plan
def test_plan_is_the_bare_app_command_in_the_repo_with_this_environment(
        monkeypatch):
    monkeypatch.setenv("JARVIS_RELAUNCH_PROBE", "carried")
    cmd, cwd, env = relaunch.plan()
    assert cmd == [sys.executable, "-m", "jarvis.app"]
    assert Path(cwd) == REPO_ROOT
    # a COPY of the environment: DISPLAY / XAUTHORITY / DBUS ride along,
    # and a later edit to os.environ does not reach the plan
    assert env["JARVIS_RELAUNCH_PROBE"] == "carried"
    assert env is not os.environ
    monkeypatch.setenv("JARVIS_RELAUNCH_PROBE", "changed")
    assert env["JARVIS_RELAUNCH_PROBE"] == "carried"


# ------------------------------------------------------- spawn_relauncher
class _RecordingPopen:
    """A stand-in for subprocess.Popen: records argv + kwargs, hands back a
    process with a chosen pid."""

    def __init__(self, pid=4242, probe=b""):
        self.pid = pid
        self.probe = probe
        self.calls: list[tuple[list, dict]] = []

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), dict(kw)))
        if self.probe:
            # what the real helper does with the handle it inherits: write
            # to it WHILE the parent still holds it open (the parent closes
            # its copy right after Popen, as a real Popen duplicates it)
            os.write(kw["stdout"], self.probe)

        class _P:
            pid = self.pid
        return _P()


def test_spawn_relauncher_starts_the_detached_helper_and_returns_its_pid(
        tmp_path):
    popen = _RecordingPopen(pid=777, probe=b"probe\n")
    log_path = tmp_path / "logs" / "relaunch.log"
    cmd = ["/venv/bin/python", "-m", "jarvis.app"]
    env = {"DISPLAY": ":1", "PATH": "/bin"}
    pid = relaunch.spawn_relauncher(31337, cmd, "/repo", env, log_path,
                                    grace_s=20, popen=popen)
    assert pid == 777
    assert len(popen.calls) == 1
    argv, kw = popen.calls[0]
    assert argv == [sys.executable, "-m", "jarvis.relaunch",
                    "--wait-pid", "31337", "--grace", "20",
                    "--log", str(log_path), "--cwd", "/repo",
                    "--", "/venv/bin/python", "-m", "jarvis.app"]
    # detached from the app: its own session, no terminal, fds closed
    assert kw["start_new_session"] is True
    assert kw["close_fds"] is True
    assert kw["stdin"] == subprocess.DEVNULL
    assert kw["cwd"] == "/repo"
    assert kw["env"] == env
    # stdout/stderr go to the log file (a traceback in the helper must
    # land somewhere he can read); the directory is created for it
    assert log_path.parent.is_dir()
    assert kw["stderr"] == kw["stdout"]
    assert log_path.read_text() == "probe\n"


def test_spawn_relauncher_grace_is_carried_as_the_helper_saw_it(tmp_path):
    popen = _RecordingPopen()
    relaunch.spawn_relauncher(1, ["py", "-m", "jarvis.app"], "/repo", {},
                              tmp_path / "relaunch.log", grace_s=7.5,
                              popen=popen)
    argv, _ = popen.calls[0]
    assert argv[argv.index("--grace") + 1] == "7.5"
