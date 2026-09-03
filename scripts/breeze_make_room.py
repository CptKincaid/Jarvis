#!/usr/bin/env python3
"""Make room for the Breeze sidecar's one-time start -- or say why we will not.

THE PROBLEM THIS EXISTS FOR
---------------------------
breeze_server.py refuses to start under MemFree 33 GB, and it is right to:
the resident model costs 13.5 GB of free pages and CUDA-graph capture then
demands ~18.2 GB more, in bursts, on top of it (a real start went in at
33.88 GB and troughed at 2.39 GB -- the shape of the 2026-08-28 power-off).

At a fresh boot that gate is never in question: ollama loads no model until
something asks it to, and the only thing that asks is jarvis.brain's
residency loop, which starts with Jarvis. So the ordering that solves the
boot window is Breeze-before-Jarvis, and it is done by the unit plus
scripts/jarvis-autostart, not here. The margin is measured: a start on
2026-09-02 went in at MemFree 49.87 GB -- with brave, claude, 46 GB of page
cache, Jarvis and F5 up, only gemma4 evicted -- and troughed at 18.66 GB. A
fresh login holds strictly less, so 49.9 GB in and 18.7 GB at the trough is
a FLOOR, not an estimate.

This script exists for the ONE case boot ordering does not cover: a GNOME
LOG OUT WITHOUT A REBOOT. ollama.service is a system service and survives
the session; jarvis.brain pins gemma4:26b with keep_alive -1 and nothing
unpins it when Jarvis dies. So the next login starts Breeze against a
19 GB model that no longer has an owner, MemFree sits near 30 GB, and the
gate refuses for a reason that has already stopped being true.

WHAT IT WILL NOT DO, AND WHY THAT IS THE POINT
----------------------------------------------
It does not evict while Jarvis is running. That is not caution, it is
arithmetic: brain.RESIDENCY_INTERVAL_S is 30 s, so a live Jarvis re-pins
gemma4 within half a minute of any unload, and a 19.0 GB reload landing
inside Breeze's 31.2 GB capture window is

    49.9 GB free (measured, gemma4 evicted) - 31.2 - 19.0 = -0.3 GB

which is the power-off, not a near miss. An eviction is only free when the
process that would undo it is not there. So: Jarvis alive -> do nothing,
let the gate refuse legibly, Jarvis keeps speaking through F5.

Liveness is read from the pid file Jarvis itself writes (app.py
_focus_running_instance) and then CONFIRMED against /proc/<pid>/cmdline --
anchored on the pid file, never a search of the process table, and never
used to signal anything. A pid file whose process is gone, or whose cmdline
is somebody else's, is the stale file a wiped /tmp leaves behind. Anything
we cannot read, we treat as "Jarvis is alive": the safe direction is the one
where we change nothing.

ALWAYS EXITS 0
--------------
It runs as ExecStartPre, and a failing ExecStartPre stops the unit before
main() ever prints its refusal. The refusal -- with the two numbers and the
lever -- is the most useful thing a failed start can produce, so nothing
here is allowed to pre-empt it. Every failure is a printed line and exit 0.

Stdlib only: it runs under the system /usr/bin/python3, not the breeze venv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

MEMINFO_PATH = "/proc/meminfo"
PROC_ROOT = "/proc"
DEFAULT_PIDFILE = "/tmp/vss_voice/jarvis.pid"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# Mirrors scripts/breeze_server.py MIN_FREE_GB. Passed explicitly by the unit
# so the two can never drift apart silently.
DEFAULT_MIN_FREE_GB = 33.0
# Long enough for ollama to actually return the pages (a 19 GB unload is not
# instant), short enough that a wedged ollama does not hold up the login.
DEFAULT_WAIT_S = 30.0
POLL_S = 0.5
# The cmdline of a real Jarvis. app.py is started as `python -m jarvis.app`,
# and the pid file is the anchor -- this only has to reject an unrelated
# process that inherited a recycled pid.
JARVIS_CMDLINE_MARK = "jarvis"


# --------------------------------------------------------------- seams
# Module-level and looked up at call time, so the tests drive every branch
# with no /proc, no ollama and no Jarvis.
def read_meminfo(path: str = MEMINFO_PATH) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def mem_gb(field: str = "MemFree") -> float:
    """``field`` from /proc/meminfo in GiB, or 0.0 when it cannot be read.

    0.0 is deliberately the pessimistic answer: it reads as "no room", which
    sends us down the do-something branch rather than the silently-skip one.
    """
    try:
        text = read_meminfo()
    except OSError:
        return 0.0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") == field:
            try:
                return int(parts[1]) / (1024.0 * 1024.0)
            except ValueError:
                return 0.0
    return 0.0


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body or b"{}")


def ollama_loaded(base_url: str) -> list[str]:
    """Model names ollama currently holds resident; [] when it is down."""
    try:
        data = http_json(f"{base_url}/api/ps", timeout=5.0)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"breeze-make-room: ollama /api/ps unavailable ({exc}); "
              f"nothing to unload", flush=True)
        return []
    names = []
    for entry in data.get("models") or []:
        name = entry.get("name") or entry.get("model") or ""
        if name:
            names.append(str(name))
    return names


def ollama_unload(base_url: str, model: str) -> bool:
    """keep_alive 0 -- the same call jarvis.brain.release() makes."""
    try:
        http_json(f"{base_url}/api/generate",
                  {"model": model, "keep_alive": 0}, timeout=30.0)
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"breeze-make-room: could not unload {model}: {exc}", flush=True)
        return False


def read_cmdline(pid: int) -> str:
    with open(f"{PROC_ROOT}/{pid}/cmdline", "rb") as fh:
        return fh.read().decode("utf-8", "replace").replace("\0", " ")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                      # alive, just not ours
    except OSError:
        return True                      # unreadable: assume alive
    return True


def live_jarvis_pid(pidfile: str) -> int | None:
    """The pid in ``pidfile`` when it is a running Jarvis, else None.

    Three ways to be None, and they are all "the file is stale": no file, a
    pid nobody is using, or a pid whose cmdline belongs to something else.
    Anything unreadable returns the pid -- see the module docstring on which
    direction is the safe one.
    """
    try:
        with open(pidfile, "r", encoding="utf-8") as fh:
            pid = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    if pid <= 0 or not pid_alive(pid):
        return None
    try:
        cmdline = read_cmdline(pid)
    except OSError:
        return pid                       # cannot confirm; assume it is him
    if JARVIS_CMDLINE_MARK not in cmdline:
        return None
    return pid


# ---------------------------------------------------------------- main
def make_room(min_free_gb: float, wait_s: float, pidfile: str,
              base_url: str) -> int:
    free = mem_gb("MemFree")
    if free >= min_free_gb:
        print(f"breeze-make-room: MemFree {free:.1f}GB >= {min_free_gb:.0f}GB; "
              f"nothing to do", flush=True)
        return 0

    pid = live_jarvis_pid(pidfile)
    if pid is not None:
        print(f"breeze-make-room: MemFree {free:.1f}GB is short of "
              f"{min_free_gb:.0f}GB, but Jarvis is running (pid {pid}) and "
              f"re-pins his model every 30s -- an eviction now would land a "
              f"19GB reload inside the 31GB graph capture. Leaving it alone; "
              f"the sidecar will refuse and Jarvis keeps speaking through F5.",
              flush=True)
        return 0

    loaded = ollama_loaded(base_url)
    if not loaded:
        print(f"breeze-make-room: MemFree {free:.1f}GB is short of "
              f"{min_free_gb:.0f}GB and ollama holds nothing; the room has to "
              f"come from somewhere else", flush=True)
        return 0

    print(f"breeze-make-room: MemFree {free:.1f}GB, no Jarvis, ollama holds "
          f"{', '.join(loaded)} -- unloading (nothing will re-pin it until "
          f"Jarvis starts)", flush=True)
    for model in loaded:
        ollama_unload(base_url, model)

    deadline = time.monotonic() + wait_s
    while True:
        free = mem_gb("MemFree")
        if free >= min_free_gb:
            print(f"breeze-make-room: MemFree {free:.1f}GB; room made",
                  flush=True)
            return 0
        if time.monotonic() >= deadline:
            print(f"breeze-make-room: MemFree {free:.1f}GB after {wait_s:.0f}s "
                  f"-- still short of {min_free_gb:.0f}GB; letting the sidecar "
                  f"refuse with the real numbers", flush=True)
            return 0
        time.sleep(POLL_S)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    ap.add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S)
    ap.add_argument("--pidfile", default=DEFAULT_PIDFILE)
    ap.add_argument("--ollama-url", default=os.environ.get(
        "OLLAMA_HOST_URL", DEFAULT_OLLAMA_URL))
    args = ap.parse_args(argv)
    try:
        return make_room(args.min_free_gb, args.wait_s, args.pidfile,
                         args.ollama_url)
    except Exception as exc:               # noqa: BLE001 - see the docstring
        print(f"breeze-make-room: {type(exc).__name__}: {exc} "
              f"(ignored; the sidecar's own gate decides)", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
