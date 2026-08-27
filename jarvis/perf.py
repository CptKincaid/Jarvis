"""Opt-in performance instrumentation. Harmless when the env flags are unset.

JARVIS_PERF_DETAIL=1         thread dump, GC pass logging and per-event late
                             avatar-slot detail lines, without the profile.
JARVIS_PROFILE_SECS=N        once the avatar's full frame cycle is live (plus
                             JARVIS_PROFILE_DELAY_SECS of settle, default 45),
                             cProfile the Tk thread for N seconds and write the
                             sorted stats (tottime + cumtime, top 40) to
                             <LOG_DIR>/profile_main.txt. While the flag is set
                             the app also logs every thread's name + native id
                             (so /proc/PID/task/TID maps to names), every GC
                             pass >= 2 ms, and each late avatar slot with the
                             age of the periodic jobs that ran before it.

`mark(name)` is a one-dict-write timestamp the periodic jobs leave behind so a
late slot can be correlated with them; it is always on and costs nothing.
"""
from __future__ import annotations

import gc
import os
import threading
import time

from jarvis.logs import LOG_DIR, get_logger

log = get_logger("perf")

_marks: dict = {}          # name -> monotonic() of the last run
_gc_info: dict = {"t0": 0.0, "gen": -1, "ms": 0.0, "last": 0.0}


def profile_secs() -> float:
    try:
        return float(os.environ.get("JARVIS_PROFILE_SECS", "") or 0.0)
    except ValueError:
        return 0.0


def profile_delay_secs() -> float:
    try:
        return float(os.environ.get("JARVIS_PROFILE_DELAY_SECS", "") or 45.0)
    except ValueError:
        return 45.0


def enabled() -> bool:
    return profile_secs() > 0.0


def detail_enabled() -> bool:
    """Diagnostics WITHOUT the cProfile run: JARVIS_PERF_DETAIL=1 turns on
    the thread dump, GC pass logging and per-event late-slot detail lines
    (also implied by JARVIS_PROFILE_SECS)."""
    return enabled() or os.environ.get("JARVIS_PERF_DETAIL", "") not in (
        "", "0", "false", "no")


# ------------------------------------------------------------------ marks
def mark(name: str) -> None:
    _marks[name] = time.monotonic()


def ages(now: float = None) -> str:
    """'gc=1.23s(gen2 41.0ms) temps=0.40s …' — seconds since each mark."""
    now = time.monotonic() if now is None else now
    parts = []
    if _gc_info["last"]:
        parts.append("gc=%.2fs(gen%d %.1fms)" % (
            now - _gc_info["last"], _gc_info["gen"], _gc_info["ms"]))
    for name, t in sorted(_marks.items()):
        parts.append("%s=%.2fs" % (name, now - t))
    return " ".join(parts) or "-"


# ------------------------------------------------------------------ threads
def log_threads(tag: str) -> None:
    """Every thread's name + native id (Python threads by name; native-only
    threads such as the Tcl notifier / CUDA handlers by their comm)."""
    known = {}
    for t in threading.enumerate():
        known[t.native_id] = t.name
        log.info("thread[%s]: tid=%s name=%s daemon=%s", tag, t.native_id,
                 t.name, t.daemon)
    try:
        base = f"/proc/{os.getpid()}/task"
        for tid in sorted(os.listdir(base), key=int):
            if int(tid) in known:
                continue
            try:
                with open(f"{base}/{tid}/comm") as fh:
                    comm = fh.read().strip()
            except OSError:
                continue
            log.info("thread[%s]: tid=%s name=<native:%s>", tag, tid, comm)
    except OSError:
        pass


# ---------------------------------------------------------------------- gc
def _gc_cb(phase, info):
    if phase == "start":
        _gc_info["t0"] = time.monotonic()
        return
    now = time.monotonic()
    ms = (now - _gc_info["t0"]) * 1000.0
    _gc_info.update(gen=info.get("generation", -1), ms=ms, last=now)
    if ms >= 2.0:
        log.info("gc: gen%d %.1f ms collected=%d thread=%s",
                 info.get("generation", -1), ms, info.get("collected", 0),
                 threading.current_thread().name)


def install_gc_logging() -> None:
    if _gc_cb not in gc.callbacks:
        gc.callbacks.append(_gc_cb)
    log.info("gc: thresholds=%s counts=%s freeze=%d", gc.get_threshold(),
             gc.get_count(), gc.get_freeze_count())


# ------------------------------------------------------------------ profile
def start_profile(widget, secs: float = None, path=None) -> None:
    """cProfile the calling (Tk) thread for `secs`, then write the stats.
    Scheduled through `widget.after` so start and stop both run on the Tk
    thread (cProfile is per-thread)."""
    import cProfile
    import io
    import pstats

    secs = profile_secs() if secs is None else float(secs)
    if secs <= 0:
        return
    path = path or (LOG_DIR / "profile_main.txt")
    log_threads("profile-start")
    try:
        n_obj = len(gc.get_objects())
    except Exception:
        n_obj = -1
    log.info("profile: tk thread tid=%s objects=%d — %.0fs starts",
             threading.get_native_id(), n_obj, secs)
    prof = cProfile.Profile()
    t_wall0 = time.monotonic()
    t_cpu0 = time.thread_time()
    prof.enable()

    def _stop():
        prof.disable()
        wall = time.monotonic() - t_wall0
        cpu = time.thread_time() - t_cpu0
        out = io.StringIO()
        out.write("cProfile of the Tk thread (tid %s) for %.1fs wall; thread "
                  "CPU %.2fs = %.1f%% of one core (includes profiler "
                  "overhead)\n\n" % (threading.get_native_id(), wall, cpu,
                                     100.0 * cpu / max(wall, 1e-9)))
        for key in ("tottime", "cumtime"):
            out.write("=" * 78 + f"\nsorted by {key}\n")
            pstats.Stats(prof, stream=out).sort_stats(key).print_stats(40)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(out.getvalue())
            log.info("profile: written to %s (tk thread %.1f%% over %.1fs)",
                     path, 100.0 * cpu / max(wall, 1e-9), wall)
        except OSError:
            log.exception("profile: write failed")
        log_threads("profile-end")

    widget.after(int(secs * 1000), _stop)


def on_full_cycle(widget) -> None:
    """Reactor hook: the full frame cycle just went live. Schedules the
    profile after the settle delay when JARVIS_PROFILE_SECS is set."""
    if not enabled():
        return
    delay = profile_delay_secs()
    log.info("profile: full cycle live — profiling in %.0fs", delay)
    widget.after(int(delay * 1000), lambda: start_profile(widget))
