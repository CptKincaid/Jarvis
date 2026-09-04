"""Restart Jarvis from the Restart button: a tiny detached helper that
waits for the old process to be gone, then launches the new one.

STDLIB ONLY, at import and at run. This module is what
``python -m jarvis.relaunch`` executes after the app has quit, and it is
imported by the UI for the code-status line; ``jarvis.config`` runs
``Config.load()`` at import and ``jarvis.brain`` loads the assistant config
at import, so touching either here would make a helper whose only job is a
clean handover read (and possibly write) his config on the way. It reads
nothing under HOME; tests/test_relaunch.py measures that with
``python -X importtime``.

Two halves:

* the app side -- ``plan()`` (the command, the repo, a copy of the
  environment) and ``spawn_relauncher()`` (start the helper, detached,
  logging to ``relaunch.log``);
* the helper side -- ``main()``: poll until the old pid is GONE (a pid the
  kernel reuses for something else counts as gone: the cmdline is checked
  for ``jarvis.app``), SIGTERM after the grace period, SIGKILL five seconds
  later, then ONE launch. A failed launch is logged and exits 1, never
  retried: a loop that keeps launching a broken checkout is the one thing
  worse than no Jarvis.

WHY THE BARE COMMAND and not scripts/jarvis-autostart: the wrapper exists
to keep a first Jarvis start out of the Breeze sidecar's one-time CUDA
graph capture at login. At a button restart Jarvis has already been up,
so the sidecar, if enabled, is already up and answering; the wrapper would
be a pass-through (measured "not waiting" in 1 ms) that then exec's this
exact command. Nothing is lost by skipping it.

WHY THE OLD PID MUST BE DEAD FIRST: app._focus_running_instance reads
/tmp/vss_voice/jarvis.pid and, when that pid is ALIVE, raises its window
and exits instead of starting. A launch while the old process still holds
the pid would therefore be silently swallowed. A stale file with a dead pid
is overwritten, which is the case the helper waits for.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_MODULE = "jarvis.app"
DEFAULT_GRACE_S = 20.0


# ------------------------------------------------------------ app side
def plan() -> tuple[list[str], str, dict]:
    """The relaunch: (command, cwd, env). The same interpreter, the bare
    module, run from the repo root, with a COPY of this process's
    environment (DISPLAY, XAUTHORITY, XDG_RUNTIME_DIR, DBUS_SESSION_BUS_
    ADDRESS all ride along; a later change to os.environ does not)."""
    return [sys.executable, "-m", APP_MODULE], str(REPO_ROOT), dict(os.environ)


def helper_argv(old_pid: int, cmd: list[str], cwd, log_path,
                grace_s: float = DEFAULT_GRACE_S) -> list[str]:
    """The helper's own command line; ``--`` separates it from the launch."""
    return [sys.executable, "-m", "jarvis.relaunch",
            "--wait-pid", str(int(old_pid)),
            "--grace", str(grace_s),
            "--log", str(log_path),
            "--cwd", str(cwd),
            "--", *cmd]


def spawn_relauncher(old_pid: int, cmd: list[str], cwd, env: dict, log_path,
                     grace_s: float = DEFAULT_GRACE_S,
                     popen=subprocess.Popen) -> int:
    """Start the helper detached (its own session, stdin from /dev/null,
    stdout+stderr appended to ``log_path``, no inherited fds) and return
    its pid. The file handle is what the helper writes its own lines to as
    well, so a traceback and the timeline land in the same place."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        proc = popen(helper_argv(old_pid, cmd, cwd, log_path, grace_s),
                     cwd=str(cwd), env=env, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=fd, stderr=fd,
                     close_fds=True)
    finally:
        os.close(fd)
    return int(proc.pid)
