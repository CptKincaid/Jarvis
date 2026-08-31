"""The run ledger: a training run gets a start, a finish and an honest duration.

Today a trainer appearing on the box costs two spoken lines in a whole
run -- ``Watchdog._trainer_rule`` says "I have lent the GPU to your
trainer, sir" and, twenty-odd minutes later, "Your trainer has finished".
There is no start time, no duration, and no sense that anything happened
in between. The twenty minutes that matter are the quietest the room
ever gets.

``RunLedger`` fixes the two ends of that. It diffs successive
``health.find_trainers()`` results by pid and emits ``started`` and
``finished`` events ON CHANGE ONLY -- never a heartbeat -- and it reads
each run's TRUE start from ``/proc/<pid>/stat`` field 22 (clock ticks
since boot) plus ``/proc/stat`` btime, so a run already going when Jarvis
restarts still reports an honest elapsed. The started_at is cached in the
ledger, so the duration is still computable after /proc is gone.
``ABSENT_TICKS`` mirrors ``health.TRAINER_ABSENT_TICKS``: a run that
restarts between epochs must not be narrated as a finish.

What this module deliberately does NOT do:

* No loss curve from jarvis.log. ``logtriage.read_tail`` reads JARVIS's
  log, which knows nothing about epochs -- only the watchdog's own
  lend/reclaim lines. And ``/proc/<pid>/fd/1`` on a real run is a
  ``socket:`` or a pty, not a tailable file. Progress narration is
  therefore OPT-IN and needs ``scripts/runlog.sh``, which tees the
  trainer's stdout to ``<log_dir>/<pid>.log``. Without the wrapper the
  feature degrades to the two lifecycle beats, which is still the moment.
* No thread. The ledger is driven from the watchdog's existing 30 s tick,
  which calls ``snapshot(gpu=False)`` -- and the whole reason this
  feature exists is the state in which nvidia-smi blocks in D-state.
* No model call. ``brain.summarize`` returns the "GPU is lent" line while
  a trainer holds it (capabilities row 80), so a summary built that way
  would be a non sequitur. Speech is TTS, not the LLM, so the beats
  themselves are safe.

``is_trainer`` is a cmdline regex over interpreters, so a run started as
``./run.py`` is invisible here. That is the right failure: say nothing
rather than guess.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("runwatch")

PROC_ROOT = "/proc"
# Mirrors health.TRAINER_ABSENT_TICKS: two ticks without the pid before a
# run counts as finished, so a trainer that respawns between epochs is not
# announced as done and started again.
ABSENT_TICKS = 2
# A "run" that lasted less than this was a crash or a typo, not a run.
# The event is still emitted (the board may want it); it is flagged brief
# and the narrator stays quiet.
MIN_RUN_S = 60.0
# Progress narration: at most one line per this many seconds, AND only
# when the epoch number actually changed. The pitch's 90 s is a tic on a
# 23-minute run.
PROGRESS_GAP_S = 300.0
TAIL_BYTES = 8192               # enough for the last few progress lines

STARTED_LINE = "Your {label} has started, sir; I'll tell you when it's done."
FINISHED_LINE = "That's your {label} done, sir; {elapsed}."
FINISHED_PLAIN_LINE = "Your trainer has finished, sir; {elapsed}."
PROGRESS_LINE = "Epoch {epoch}, sir{loss}."
LOSS_FALLING = "; the loss is still falling"
LOSS_FLAT = "; the loss has stopped falling"

_EPOCH_RX = re.compile(r"\bepoch[\s:=]+(\d+)", re.I)
_LOSS_RX = re.compile(r"\bloss[\s:=]+([0-9]*\.?[0-9]+)", re.I)


# ------------------------------------------------------------- /proc seams
def clock_ticks() -> float:
    """Seam: kernel ticks per second (100 on this box)."""
    try:
        return float(os.sysconf("SC_CLK_TCK")) or 100.0
    except (ValueError, OSError, AttributeError):
        return 100.0


def read_btime(proc_root: str = PROC_ROOT) -> Optional[float]:
    """Seam: boot time as an epoch, from /proc/stat's btime line."""
    try:
        with open(os.path.join(proc_root, "stat"), encoding="ascii",
                  errors="replace") as fh:
            for line in fh:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        log.debug("btime unreadable", exc_info=True)
    return None


def read_proc_start(pid: int, proc_root: str = PROC_ROOT) -> Optional[float]:
    """Seam: when ``pid`` actually started, as an epoch. Field 22 of
    /proc/<pid>/stat is the start time in clock ticks since boot.

    Field 2 is the comm, which may contain spaces and parentheses, so the
    fields are counted from the LAST ')' -- splitting on whitespace from
    the front is the classic way to read the wrong number here.
    """
    try:
        with open(os.path.join(proc_root, str(pid), "stat"), encoding="ascii",
                  errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return None
    cut = raw.rfind(")")
    if cut < 0:
        return None
    fields = raw[cut + 2:].split()
    # after the comm, field 3 (state) is index 0, so field 22 is index 19
    if len(fields) < 20:
        return None
    btime = read_btime(proc_root)
    if btime is None:
        return None
    try:
        return btime + float(fields[19]) / clock_ticks()
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_tail(path, limit: int = TAIL_BYTES) -> str:
    """Seam: the last ``limit`` bytes of a run log ("" when unreadable)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - int(limit)))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


# ------------------------------------------------------------------ words
def elapsed_words(seconds: float) -> str:
    """"22 minutes" / "an hour and 10 minutes" / "under a minute"."""
    seconds = max(0.0, float(seconds))
    minutes = int(round(seconds / 60.0))
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, rest = divmod(minutes, 60)
    head = "an hour" if hours == 1 else f"{hours} hours"
    if rest == 0:
        return head
    return f"{head} and {rest} minute{'s' if rest != 1 else ''}"


def parse_progress(text: str) -> Optional[dict]:
    """The newest epoch (and loss, when the same tail carries one) from a
    trainer's stdout. None when the tail says nothing about either --
    which is most trainers, and the reason this is opt-in."""
    epoch = None
    for m in _EPOCH_RX.finditer(text or ""):
        epoch = int(m.group(1))
    if epoch is None:
        return None
    loss = None
    for m in _LOSS_RX.finditer(text or ""):
        try:
            loss = float(m.group(1))
        except ValueError:
            loss = None
    return {"epoch": epoch, "loss": loss}


# ----------------------------------------------------------------- events
@dataclass
class RunEvent:
    """One beat. ``kind``: started | progress | finished."""
    kind: str
    pid: int = 0
    label: str = ""
    started_at: float = 0.0
    elapsed_s: float = 0.0
    epoch: int = 0
    loss: Optional[float] = None
    falling: Optional[bool] = None     # loss vs the previous narrated epoch
    brief: bool = False                # shorter than MIN_RUN_S: not narrated

    def line(self) -> str:
        """What he would say about this beat, or "" when it is not worth
        a word (a brief run, a beat with nothing new in it)."""
        if self.brief:
            return ""
        if self.kind == "started":
            return STARTED_LINE.format(label=self.label or "trainer")
        if self.kind == "finished":
            elapsed = elapsed_words(self.elapsed_s)
            if self.label:
                return FINISHED_LINE.format(label=self.label, elapsed=elapsed)
            return FINISHED_PLAIN_LINE.format(elapsed=elapsed)
        if self.kind == "progress":
            tail = "" if self.falling is None else (
                LOSS_FALLING if self.falling else LOSS_FLAT)
            return PROGRESS_LINE.format(epoch=self.epoch, loss=tail)
        return ""


@dataclass
class Run:
    pid: int
    label: str
    started_at: float
    absent: int = 0
    epoch: int = 0
    loss: Optional[float] = None
    last_progress: float = 0.0
    seen_at: float = field(default_factory=time.time)


class RunLedger:
    """Diffs successive trainer lists into started / finished beats, and
    (opt-in) reads a tee'd run log for epoch progress.

    Pure enough to test against synthetic Proc lists: every /proc read is
    an injected callable, and ``apply`` takes ``now``.
    """

    def __init__(self, log_dir=None, start_time: Optional[Callable] = None,
                 tail: Optional[Callable] = None,
                 absent_ticks: int = ABSENT_TICKS,
                 min_run_s: float = MIN_RUN_S,
                 progress_gap_s: float = PROGRESS_GAP_S,
                 narrate: bool = True, clock: Optional[Callable] = None):
        # None (the default) disables progress entirely: without
        # scripts/runlog.sh there is no tee'd stdout to read.
        self.log_dir = Path(log_dir).expanduser() if log_dir else None
        # runwatch.narrate: False keeps the ledger, the board lane and the
        # log line, and says nothing.
        self.narrate = bool(narrate)
        self._start_time = start_time or read_proc_start
        self._tail = tail or read_tail
        self.absent_ticks = max(1, int(absent_ticks))
        self.min_run_s = float(min_run_s)
        self.progress_gap_s = float(progress_gap_s)
        # The clock apply() stamps with when the caller does not pass one.
        # Injected so the watchdog's tick can be driven at speed in tests.
        self._clock = clock or time.time
        self.runs: dict = {}                 # pid -> Run
        # "Quietly, please": narration is held for the CURRENT run and
        # lifts by itself when that run ends. Nothing to restore at quit.
        self.muted = False

    # ------------------------------------------------------------ state
    @property
    def active(self) -> list:
        return [r for r in self.runs.values() if r.absent == 0]

    def label_of(self, proc) -> str:
        return (getattr(proc, "hint", "") or getattr(proc, "name", "")
                or "trainer")

    def apply(self, trainers: list, now: Optional[float] = None) -> list:
        """One tick. Returns the beats this tick produced, in order."""
        now = self._clock() if now is None else float(now)
        out: list = []
        present = {}
        for proc in trainers or []:
            try:
                present[int(proc.pid)] = proc
            except (TypeError, ValueError, AttributeError):
                continue
        for pid, proc in present.items():
            run = self.runs.get(pid)
            if run is None:
                started = self._started_at(pid, now)
                run = Run(pid=pid, label=self.label_of(proc), started_at=started,
                          seen_at=now)
                self.runs[pid] = run
                log.info("run %d (%s) started %.0f s ago", pid, run.label,
                         max(0.0, now - started))
                out.append(RunEvent(kind="started", pid=pid, label=run.label,
                                    started_at=started,
                                    elapsed_s=max(0.0, now - started)))
                continue
            run.absent = 0
            run.seen_at = now
            ev = self._progress(run, now)
            if ev is not None:
                out.append(ev)
        for pid in list(self.runs):
            if pid in present:
                continue
            run = self.runs[pid]
            run.absent += 1
            if run.absent < self.absent_ticks:
                continue                     # an epoch restart, not a finish
            del self.runs[pid]
            elapsed = max(0.0, run.seen_at - run.started_at)
            log.info("run %d (%s) finished after %.0f s", pid, run.label, elapsed)
            out.append(RunEvent(kind="finished", pid=pid, label=run.label,
                                started_at=run.started_at, elapsed_s=elapsed,
                                brief=elapsed < self.min_run_s))
            if not self.runs:
                self.muted = False           # the mute was for THAT run
        return out

    def _started_at(self, pid: int, now: float) -> float:
        """The run's true start, so a run already going when Jarvis
        restarted still reports an honest elapsed. Falls back to now when
        /proc will not say (the run then looks as old as our knowledge of
        it, which is the only honest answer available)."""
        try:
            started = self._start_time(pid)
        except Exception:  # noqa: BLE001 - /proc races are normal
            log.debug("start time unreadable for %d", pid, exc_info=True)
            started = None
        if started is None or started > now:
            return now
        return float(started)

    # --------------------------------------------------------- progress
    def _progress(self, run: Run, now: float) -> Optional[RunEvent]:
        """One progress beat, or None. Two gates, both required: the epoch
        number must have CHANGED, and the last beat must be more than
        progress_gap_s ago. A heartbeat is what makes a narrator tiresome."""
        if self.log_dir is None:
            return None
        if run.last_progress and now - run.last_progress < self.progress_gap_s:
            return None
        try:
            text = self._tail(self.log_dir / f"{run.pid}.log")
        except Exception:  # noqa: BLE001
            log.debug("run log unreadable for %d", run.pid, exc_info=True)
            return None
        found = parse_progress(text)
        if not found or found["epoch"] == run.epoch:
            return None
        falling = None
        if found["loss"] is not None and run.loss is not None:
            falling = found["loss"] < run.loss
        run.epoch = found["epoch"]
        if found["loss"] is not None:
            run.loss = found["loss"]
        run.last_progress = now
        return RunEvent(kind="progress", pid=run.pid, label=run.label,
                        started_at=run.started_at,
                        elapsed_s=max(0.0, now - run.started_at),
                        epoch=found["epoch"], loss=found["loss"],
                        falling=falling)
