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

import errno
import os
import signal
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


# --------------------------------------------------------- helper side
def _read_cmdline(pid: int) -> str:
    with open(f"/proc/{int(pid)}/cmdline", "rb") as fh:
        return fh.read().decode("utf-8", "replace")


def process_alive(pid: int, kill=os.kill, read_cmdline=_read_cmdline,
                  marker: str = APP_MODULE) -> bool:
    """Is the OLD JARVIS still there? False on ESRCH, and False when the
    pid now belongs to something whose command line does not mention
    ``jarvis.app`` (the kernel recycles pids; waiting on a stranger would
    hold the relaunch until that stranger exits). An unreadable or EMPTY
    cmdline (a zombie, a process mid-exit) is NOT evidence of a stranger:
    the next poll settles it."""
    try:
        kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                      # exists, owned by someone else: alive
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ESRCH:
            return False
    try:
        cmd = read_cmdline(pid)
    except OSError:
        return True
    if not cmd:
        return True
    return marker in cmd.replace("\x00", " ")


def _send(kill, pid: int, sig: int, log) -> None:
    try:
        kill(pid, sig)
    except ProcessLookupError:
        log(f"pid {pid} was gone before {signal.Signals(sig).name} landed")
    except OSError as exc:
        log(f"{signal.Signals(sig).name} to pid {pid} failed: {exc}")


def wait_for_exit(pid: int, grace_s: float, *, alive, kill, sleep, clock,
                  log, poll_s: float = 0.25, term_wait_s: float = 5.0,
                  kill_wait_s: float = 30.0) -> str:
    """Block until ``pid`` is gone. Patience first (``grace_s``), then
    SIGTERM, then SIGKILL ``term_wait_s`` later, then a bounded wait.
    Returns one of "exited" (left on its own), "terminated", "killed" or
    "stuck" (survived SIGKILL: a D-state process; the launch would only be
    swallowed by the pid-file guard, so the caller gives up instead)."""
    pid = int(pid)
    t0 = clock()
    log(f"waiting for pid {pid} to exit (grace {grace_s:g}s, poll {poll_s:g}s)")
    while alive(pid):
        if clock() - t0 >= grace_s:
            break
        sleep(poll_s)
    else:
        log(f"pid {pid} gone after {clock() - t0:.2f}s")
        return "exited"
    log(f"pid {pid} still alive after {grace_s:g}s; sending SIGTERM")
    _send(kill, pid, signal.SIGTERM, log)
    t1 = clock()
    while alive(pid):
        if clock() - t1 >= term_wait_s:
            break
        sleep(poll_s)
    else:
        log(f"pid {pid} gone {clock() - t1:.2f}s after SIGTERM")
        return "terminated"
    log(f"pid {pid} still alive {term_wait_s:g}s after SIGTERM; sending SIGKILL")
    _send(kill, pid, signal.SIGKILL, log)
    t2 = clock()
    while alive(pid):
        if clock() - t2 >= kill_wait_s:
            log(f"pid {pid} still alive {kill_wait_s:g}s after SIGKILL; giving up")
            return "stuck"
        sleep(poll_s)
    log(f"pid {pid} gone {clock() - t2:.2f}s after SIGKILL")
    return "killed"
