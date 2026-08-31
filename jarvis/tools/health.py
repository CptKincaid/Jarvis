"""Spark system health: a fact-sheet tool and a memory watchdog.

The DGX Spark (GB10) has 121 GB of UNIFIED memory. GPU allocations are
carved out of system RAM, so they are invisible to nvidia-smi (memory
reads N/A) AND to the OOM killer -- when the pool runs dry, allocations
stall inside the driver and the box needs a hard power-off (2026-08-28:
two trainers plus ollama did exactly that). The one number that sees the
whole pool is ``MemAvailable`` in /proc/meminfo, so both the tool and the
watchdog key on it rather than on any per-device figure.

``system_health`` renders one compact line the local model rephrases.
``Watchdog`` samples every ``health.interval_s`` seconds on a daemon
thread, publishes a ``Status`` and speaks ONCE per episode when
MemAvailable drops under ``health.warn_gb`` (again, as an error, under
``health.critical_gb``), re-arming only after a recovery above
``warn_gb + REARM_MARGIN_GB``; and warns once when two or more processes
each hold more than ``health.hog_gb`` (the two-trainers pattern).

Every probe is a module-level seam -- ``read_meminfo``, ``read_loadavg``,
``run_nvidia_smi``, ``disk_free``, ``iter_process_rss``, ``read_cmdline``
-- looked up at call time so tests monkeypatch them. Nothing here
raises past ``snapshot()``; a probe that fails simply leaves its field
None. ``make_tools`` parks a Watchdog on ``services.health_watchdog`` and
does NOT start it: the integrator starts it beside the timekeeper.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from jarvis.events import FaultRaised, Status, bus
from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.health")

MEMINFO_PATH = "/proc/meminfo"
LOADAVG_PATH = "/proc/loadavg"
PROC_ROOT = "/proc"
DISKS = ("/", "/home")
SMI_QUERY = "temperature.gpu,clocks.sm,power.draw,utilization.gpu"
SMI_TIMEOUT = 5.0                  # nvidia-smi hangs when the driver is wedged
TOP_N = 3
KB_PER_GB = 1024 * 1024            # /proc reports kB; the sheet speaks GiB
DEFAULT_WARN_GB = 16.0
DEFAULT_CRITICAL_GB = 8.0
DEFAULT_HOG_GB = 20.0
HOG_COUNT = 2                      # "two trainers": this many hogs at once
# The actual 2026-08-28 incident: TWO trainers on the unified pool at once.
# hogs() is any two processes over hog_gb -- ollama plus a browser trips it
# and is not the incident. snap.trainers already knows which processes are
# training runs, so this rule can name the thing that wedged the box.
TRAINER_COUNT = 2
REARM_MARGIN_GB = 4.0              # hysteresis above warn_gb before re-arming
DEFAULT_INTERVAL_S = 30.0
# Interpreter names that say nothing about the job; the first script /
# module on the command line is the useful label ("python (train.py)").
_INTERPRETERS = {"python", "python3", "node", "bash", "sh", "uv", "perl", "ruby"}
# A trainer is recognised by what the interpreter runs, not by its size
# (a small-batch run under hog_gb would never trip the hogs rule): train.py,
# aiws_trainer.train, finetune_piper.py, "-m trainer" ... The check is a
# word inside the hint, so "constraint.py" does not count.
TRAINER_HINT_RX = re.compile(
    r"(?:^|[^a-z])(?:train|trainer|training|pretrain|finetune|fine_tune|fine-tune|"
    r"finetuning)(?:[^a-z]|$)", re.I)
_TRAINER_NAMES = {"python", "python3", "uv", "accelerate", "torchrun", "deepspeed"}
TRAINER_ABSENT_TICKS = 2           # ticks without the trainer before reclaiming
DEFAULT_YIELD = False

UNREADABLE_LINE = "I can't read the system counters, sir."
WARN_LINE = "Memory is getting tight, sir: {free} gigabytes free{hogs}."
CRITICAL_LINE = ("Memory is critical, sir: {free} gigabytes free{hogs}. "
                 "I'd stop something before the box does.")
HOGS_LINE = ("Two processes each hold more than {hog} gigabytes, sir: {hogs}. "
             "Last time that ended in a hard power-off.")
TRAINERS_LINE = ("{n} trainers are on the pool at once, sir: {names}. "
                 "The last time that happened the box needed a hard power-off.")
# The rules whose alerts are FAULTS (jarvis/faults.py): they light the
# board's FAULT lane and are de-duplicated across restarts. "trainer" (the
# GPU-yield lend/reclaim) is deliberately absent -- lending the GPU is
# routine, not a fault.
FAULT_RULES = ("memory", "hogs", "trainers")
LENT_LINE = ("I have lent the GPU to your trainer, sir; quick answers only "
             "until it is done.")
RECLAIMED_LINE = "Your trainer has finished, sir; I'm loading my model again."


# ------------------------------------------------------------- config
def _cfg_get(cfg, dotted: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        cur = cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return default if cur is None else cur
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            val = get(dotted, default)
            return default if val is None else val
        except Exception:
            log.debug("cfg.get(%s) failed", dotted, exc_info=True)
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def _cfg_float(cfg, dotted: str, default: float) -> float:
    try:
        return float(_cfg_get(cfg, dotted, default))
    except (TypeError, ValueError):
        return float(default)


# -------------------------------------------------------------- seams
def read_meminfo() -> dict[str, int]:
    """Seam: /proc/meminfo -> {field: kB}. Raises on an unreadable file."""
    with open(MEMINFO_PATH, encoding="ascii", errors="replace") as fh:
        return parse_meminfo(fh.read())


def read_loadavg() -> tuple[float, float, float]:
    """Seam: the three load averages from /proc/loadavg."""
    with open(LOADAVG_PATH, encoding="ascii", errors="replace") as fh:
        parts = fh.read().split()
    return float(parts[0]), float(parts[1]), float(parts[2])


def run_nvidia_smi() -> Optional[str]:
    """Seam: one CSV line from nvidia-smi, None when it cannot answer.

    Popen + communicate(timeout), never subprocess.run: on a timeout run()
    kills the child and then WAITS for it, and an nvidia-smi wedged in
    uninterruptible D-state under a stuck NVRM lock (the 2026-08-28 failure
    mode on this box) never exits. Kill and walk away instead.
    """
    try:
        proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("nvidia-smi unavailable: %s", type(exc).__name__)
        return None
    try:
        out, _ = proc.communicate(timeout=SMI_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        log.warning("nvidia-smi did not answer in %ss; not waiting for it", SMI_TIMEOUT)
        return None
    if proc.returncode != 0:
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def disk_free(path: str) -> Optional[float]:
    """Seam: free GB on the filesystem holding ``path``."""
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return None


def iter_process_rss(proc_root: str = PROC_ROOT) -> Iterator[tuple[int, str, int]]:
    """Seam: (pid, name, rss_kB) for every readable /proc/<pid>/status.
    Processes that vanish or refuse mid-scan are skipped, never fatal."""
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, entry, "status"),
                      encoding="ascii", errors="replace") as fh:
                name, rss = _parse_status(fh.read())
        except OSError:
            continue
        if rss is None:
            continue                       # kernel threads carry no VmRSS
        yield int(entry), name, rss


def read_cmdline(pid: int, proc_root: str = PROC_ROOT) -> list[str]:
    """Seam: argv of ``pid`` ([] when gone or unreadable)."""
    try:
        with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as fh:
            raw = fh.read()
    except OSError:
        return []
    return [a.decode("utf-8", errors="replace") for a in raw.split(b"\0") if a]


# ------------------------------------------------------------ parsing
def parse_meminfo(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0])
    return out


def _parse_status(text: str) -> tuple[str, Optional[int]]:
    name, rss = "", None
    for line in text.splitlines():
        if line.startswith("Name:"):
            name = line[5:].strip()
        elif line.startswith("VmRSS:"):
            parts = line[6:].split()
            if parts and parts[0].isdigit():
                rss = int(parts[0])
        if name and rss is not None:
            break
    return name, rss


def _number(token: str) -> Optional[float]:
    """"2418 MHz" / "12.61 W" / "2 %" / "43" -> float; N/A forms -> None."""
    token = token.strip()
    if not token or "N/A" in token.upper():
        return None
    try:
        return float(token.split()[0])
    except (ValueError, IndexError):
        return None


def parse_nvidia_smi(line: Optional[str]) -> Optional[dict]:
    """One CSV row -> {temp_c, sm_mhz, power_w, util_pct} (None per N/A
    field); None when the row is not nvidia-smi output at all."""
    if not line:
        return None
    fields = [f.strip() for f in line.split(",")]
    if len(fields) < 4:
        return None
    vals = [_number(f) for f in fields[:4]]
    if all(v is None for v in vals):
        return None
    return {"temp_c": vals[0], "sm_mhz": vals[1], "power_w": vals[2],
            "util_pct": vals[3]}


def cmdline_hint(argv: list[str]) -> str:
    """The script or module an interpreter is running, else ""."""
    if not argv:
        return ""
    exe = os.path.basename(argv[0]).lower()
    if not any(exe == i or exe.startswith(i) for i in _INTERPRETERS):
        return ""
    args = argv[1:]
    for i, arg in enumerate(args):
        if arg == "-m" and i + 1 < len(args):
            return args[i + 1]
        if arg == "-c":
            return ""
        if arg.startswith("-"):
            continue
        return os.path.basename(arg)
    return ""


# ----------------------------------------------------------- snapshot
@dataclass
class Proc:
    pid: int
    name: str
    rss_gb: float
    hint: str = ""

    def label(self, hint: bool = True) -> str:
        text = f"{self.name} {_gb(self.rss_gb)} GB"
        if hint and self.hint:
            text += f" ({self.hint})"
        return text


@dataclass
class Snapshot:
    mem_total_gb: Optional[float] = None
    mem_avail_gb: Optional[float] = None
    load1: Optional[float] = None
    gpu: Optional[dict] = None
    disks: list[tuple[str, Optional[float]]] = field(default_factory=list)
    top: list[Proc] = field(default_factory=list)
    trainers: list[Proc] = field(default_factory=list)   # by cmdline, any size

    @property
    def readable(self) -> bool:
        return self.mem_total_gb is not None or self.load1 is not None


def _gb(value: float) -> str:
    """Whole gigabytes once past ten, one decimal below (7.6 GB matters)."""
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def top_processes(n: int = TOP_N) -> list[Proc]:
    procs: list[tuple[int, str, int]] = []
    try:
        for pid, name, rss_kb in iter_process_rss():
            procs.append((pid, name, rss_kb))
    except Exception:  # noqa: BLE001 - a bad /proc must not sink the sheet
        log.debug("process scan failed", exc_info=True)
    procs.sort(key=lambda p: -p[2])
    out = []
    for pid, name, rss_kb in procs[:n]:
        try:
            hint = cmdline_hint(read_cmdline(pid))
        except Exception:  # noqa: BLE001
            hint = ""
        out.append(Proc(pid=pid, name=name, rss_gb=rss_kb / KB_PER_GB, hint=hint))
    return out


def is_trainer(argv: list[str]) -> bool:
    """True when an interpreter's command line names a training job."""
    hint = cmdline_hint(argv)
    if hint:
        return bool(TRAINER_HINT_RX.search(hint))
    # torchrun / accelerate launch <script>: no interpreter prefix, so
    # cmdline_hint has nothing to say; look at the launcher's first script.
    if argv and os.path.basename(argv[0]).lower() in _TRAINER_NAMES:
        return any(TRAINER_HINT_RX.search(os.path.basename(a)) for a in argv[1:]
                   if not a.startswith("-"))
    return False


def find_trainers(self_pid: Optional[int] = None) -> list[Proc]:
    """Every process whose command line looks like a trainer, largest first.
    Only interpreter-named processes have their cmdline read (a few dozen
    small files, not the whole table). Never raises."""
    me = os.getpid() if self_pid is None else self_pid
    out: list[Proc] = []
    try:
        for pid, name, rss_kb in iter_process_rss():
            if pid == me or name.lower() not in _TRAINER_NAMES:
                continue
            try:
                argv = read_cmdline(pid)
            except Exception:  # noqa: BLE001
                continue
            if is_trainer(argv):
                out.append(Proc(pid=pid, name=name, rss_gb=rss_kb / KB_PER_GB,
                                hint=cmdline_hint(argv) or
                                os.path.basename(argv[-1] if argv else "")))
    except Exception:  # noqa: BLE001 - a bad /proc must not sink the tick
        log.debug("trainer scan failed", exc_info=True)
    out.sort(key=lambda p: -p.rss_gb)
    return out


def snapshot(gpu: bool = True) -> Snapshot:
    """Every probe, each failing on its own; never raises."""
    snap = Snapshot()
    try:
        mem = read_meminfo()
        if "MemTotal" in mem:
            snap.mem_total_gb = mem["MemTotal"] / KB_PER_GB
        if "MemAvailable" in mem:
            snap.mem_avail_gb = mem["MemAvailable"] / KB_PER_GB
    except Exception:  # noqa: BLE001
        log.warning("meminfo unreadable", exc_info=True)
    try:
        snap.load1 = read_loadavg()[0]
    except Exception:  # noqa: BLE001
        log.debug("loadavg unreadable", exc_info=True)
    try:
        snap.gpu = parse_nvidia_smi((run_nvidia_smi() if gpu else None))
    except Exception:  # noqa: BLE001
        log.debug("nvidia-smi failed", exc_info=True)
    for mount in DISKS:
        try:
            snap.disks.append((mount, disk_free(mount)))
        except Exception:  # noqa: BLE001
            snap.disks.append((mount, None))
    snap.top = top_processes()
    snap.trainers = find_trainers()
    return snap


# --------------------------------------------------------------- words
def _gpu_words(gpu: Optional[dict]) -> str:
    if not gpu:
        return "GPU unavailable"
    bits = []
    if gpu.get("temp_c") is not None:
        bits.append(f"{gpu['temp_c']:.0f} C")
    if gpu.get("sm_mhz") is not None:
        bits.append(f"at {gpu['sm_mhz']:.0f} MHz")
    if gpu.get("power_w") is not None:
        bits.append(f"{gpu['power_w']:.0f} W")
    if gpu.get("util_pct") is not None:
        bits.append(f"{gpu['util_pct']:.0f}% busy")
    # "43 C at 2418 MHz" reads as one clause; every other field is its own.
    head, rest = bits[0], bits[1:]
    if rest and rest[0].startswith("at "):
        head, rest = f"{head} {rest[0]}", rest[1:]
    return "GPU " + ", ".join([head, *rest])


def _disk_words(disks) -> str:
    parts = [f"{m} {_gb(free)} GB" for m, free in disks if free is not None]
    return ("disk free " + ", ".join(parts)) if parts else "disk unknown"


def hogs(snap: Snapshot, hog_gb: float = DEFAULT_HOG_GB) -> list[Proc]:
    return [p for p in snap.top if p.rss_gb > hog_gb]


def fact_sheet(snap: Snapshot, warn_gb: float = DEFAULT_WARN_GB,
               hog_gb: float = DEFAULT_HOG_GB) -> str:
    """"Memory 41 of 121 GB free; GPU 42 C at 2418 MHz, 12 W, 2% busy;
    load 3.1; disk free / 210 GB, /home 1500 GB; top: python 38 GB
    (train.py), ollama 22 GB." plus a note when a rule trips."""
    if snap.mem_avail_gb is not None and snap.mem_total_gb is not None:
        mem = f"Memory {_gb(snap.mem_avail_gb)} of {_gb(snap.mem_total_gb)} GB free"
    else:
        mem = "Memory unknown"
    load = f"load {snap.load1:.1f}" if snap.load1 is not None else "load unknown"
    top = ("top: " + ", ".join(p.label() for p in snap.top)) if snap.top else "top: unknown"
    notes = []
    if snap.mem_avail_gb is not None and snap.mem_avail_gb < warn_gb:
        notes.append("memory tight")
    if len(hogs(snap, hog_gb)) >= HOG_COUNT:
        notes.append(f"{len(hogs(snap, hog_gb))} processes over {_gb(hog_gb)} GB each")
    sheet = "; ".join([mem, _gpu_words(snap.gpu), load, _disk_words(snap.disks), top])
    if notes:
        sheet += "; note: " + ", ".join(notes)
    return sheet + "."


def _spoken_hogs(procs: list[Proc], limit: int = 2) -> str:
    """"python at 38 gigabytes and ollama at 22" for the spoken lines."""
    procs = procs[:limit]
    if not procs:
        return ""
    if len(procs) == 1:
        return f"{procs[0].name} at {procs[0].rss_gb:.0f} gigabytes"
    first = f"{procs[0].name} at {procs[0].rss_gb:.0f} gigabytes"
    rest = ", ".join(f"{p.name} at {p.rss_gb:.0f}" for p in procs[1:])
    return f"{first} and {rest}"


def memory_line(snap: Snapshot, critical: bool = False) -> str:
    tail = _spoken_hogs(snap.top)
    tail = f", with {tail}" if tail else ""
    free = f"{(snap.mem_avail_gb or 0):.0f}"
    return (CRITICAL_LINE if critical else WARN_LINE).format(free=free, hogs=tail)


def hogs_line(procs: list[Proc], hog_gb: float) -> str:
    return HOGS_LINE.format(hog=f"{hog_gb:.0f}", hogs=_spoken_hogs(procs))


def distinct_runs(procs: list[Proc]) -> list[Proc]:
    """One Proc per training RUN, largest first. torchrun / accelerate /
    DDP fan a single run out into one interpreter process per GPU worker,
    and every one of them matches is_trainer -- counting processes would
    report "4 trainers on the pool" for one job and the warning would be
    a lie the first time he saw it. Runs are keyed by the script the
    interpreter is running, which is what distinguishes the 2026-08-28
    incident (train.py AND finetune_piper.py) from a fanned-out job."""
    seen: dict = {}
    for proc in sorted(procs, key=lambda p: -p.rss_gb):
        seen.setdefault(proc.hint or proc.name, proc)
    return list(seen.values())


def trainers_line(procs: list[Proc]) -> str:
    """"2 trainers are on the pool at once, sir: train.py and finetune.py."
    Named by their hint (the script), which is what he recognises -- both
    are "python" by process name."""
    names = [p.hint or p.name for p in procs[:3]]
    if len(names) > 1:
        joined = ", ".join(names[:-1]) + f" and {names[-1]}"
    else:
        joined = names[0] if names else "unknown"
    return TRAINERS_LINE.format(n=len(procs), names=joined)


# ------------------------------------------------------------ watchdog
@dataclass
class Alert:
    kind: str                          # warn | error
    line: str                          # what he says
    status: str                        # the status-bar chip
    rule: str = "memory"               # memory | hogs | trainers | trainer


class Watchdog:
    """Samples ``snapshot(gpu=False)`` on a daemon thread and raises the alarm once
    per episode. ``check(snap)`` is the whole state machine (tests drive
    it directly); ``tick()`` is one sample; ``start()`` / ``stop()`` own
    the thread. ``speak`` may be None at construction: the callback is
    resolved at fire time from ``services.speak`` so boot order does not
    matter."""

    def __init__(self, cfg=None, speak: Optional[Callable[[str], None]] = None,
                 services=None, publish: Optional[Callable] = None,
                 interval: Optional[float] = None, brain=None, faults=None):
        # GPU yield (health.yield_to_trainer): ``brain`` is anything with
        # release() / reclaim() / is_lent(); None means jarvis.brain itself,
        # imported at fire time (tests pass a fake).
        self.yield_to_trainer = bool(_cfg_get(cfg, "health.yield_to_trainer",
                                              DEFAULT_YIELD))
        self._brain_obj = brain
        self._lent_to: Optional[int] = None      # trainer pid the model is lent to
        self._absent_ticks = 0
        # pids the user took the GPU back from ("take the GPU back" while the
        # run continues): never lend to those again, or the next tick would
        # undo a spoken order
        self._held: set[int] = set()
        self.warn_gb = _cfg_float(cfg, "health.warn_gb", DEFAULT_WARN_GB)
        # A critical threshold above warn would fire "critical" first and
        # swallow the warning; clamp so the ladder always runs warn -> critical.
        self.critical_gb = min(_cfg_float(cfg, "health.critical_gb", DEFAULT_CRITICAL_GB),
                               self.warn_gb)
        self.hog_gb = _cfg_float(cfg, "health.hog_gb", DEFAULT_HOG_GB)
        self.interval = float(interval if interval is not None
                              else _cfg_float(cfg, "health.interval_s", DEFAULT_INTERVAL_S))
        self._speak = speak
        self._services = services
        self._publish = publish or bus.publish
        self._level = 0                # 0 ok | 1 warned | 2 critical
        self._hogs_alerted = False
        self._trainers_alerted = False
        # jarvis.faults.FaultLog when the integrator wires one: the spoken
        # -once state file, so a restart INTO a still-tight pool does not
        # announce the same episode a second time. None keeps the old
        # behaviour (speak every time a rule newly trips).
        self._faults = faults
        # rule -> the fault tokens spoken for it, so a recovery can clear
        # exactly those entries from the state file (and no others).
        self._fault_tokens: dict = {}
        self._unreadable_logged = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last: Optional[Snapshot] = None

    # ---------------------------------------------------------- rules
    def check(self, snap: Snapshot) -> list[Alert]:
        """Apply the rules to one snapshot; returns the alerts it raised."""
        self.last = snap
        fired: list[Alert] = []
        avail = snap.mem_avail_gb
        if avail is None:
            if not self._unreadable_logged:
                log.warning("health watchdog: MemAvailable unreadable; memory rule idle")
                self._unreadable_logged = True
        else:
            level = 2 if avail < self.critical_gb else 1 if avail < self.warn_gb else 0
            if level > self._level:
                # Straight from fine to critical raises ONE error, not a
                # warning and an error back to back.
                critical = level == 2
                fired.append(Alert(kind="error" if critical else "warn",
                                   line=memory_line(snap, critical),
                                   status=f"Memory {'critical' if critical else 'tight'}: "
                                          f"{_gb(avail)} GB free"))
                self._level = level
            elif self._level and avail >= self.warn_gb + REARM_MARGIN_GB:
                # Hysteresis: recovering into the warn..warn+4 band keeps the
                # episode open, so a pool bouncing around 16 GB is announced
                # once, not every 30 s.
                self._level = 0
                log.info("health watchdog: memory recovered (%.1f GB free)", avail)
                self._clear_fault("memory")
                self._safe_publish(Status(text=f"Memory recovered: {_gb(avail)} GB free",
                                          kind="ok"))
        heavy = hogs(snap, self.hog_gb)
        if len(heavy) >= HOG_COUNT:
            if not self._hogs_alerted:
                fired.append(Alert(kind="warn", line=hogs_line(heavy, self.hog_gb),
                                   status=f"{len(heavy)} processes over {_gb(self.hog_gb)} GB",
                                   rule="hogs"))
                self._hogs_alerted = True
        elif self._hogs_alerted and \
                len(hogs(snap, self.hog_gb - REARM_MARGIN_GB)) < HOG_COUNT:
            # Hysteresis, as for memory: a process hovering around hog_gb must
            # not re-speak "last time that ended in a hard power-off" every
            # other tick.
            self._hogs_alerted = False
            self._clear_fault("hogs")
        # Two trainers on the pool: the 2026-08-28 incident by name. Latched
        # like the others, and cleared the moment one of them exits -- the
        # clear is what re-arms the warning for the next run.
        runners = distinct_runs(snap.trainers or [])
        if len(runners) >= TRAINER_COUNT:
            if not self._trainers_alerted:
                fired.append(Alert(kind="error", line=trainers_line(runners),
                                   status=f"{len(runners)} trainers on the pool",
                                   rule="trainers"))
                self._trainers_alerted = True
        elif self._trainers_alerted:
            self._trainers_alerted = False
            log.info("health watchdog: back to %d trainer run(s) on the pool",
                     len(runners))
            self._clear_fault("trainers")
            self._safe_publish(Status(text="One trainer on the pool", kind="ok"))
        if self.yield_to_trainer:
            self._trainer_rule(snap, fired)
        for alert in fired:
            self._fire(alert)
        return fired

    # ---------------------------------------------------- trainer yield
    def _brain(self):
        if self._brain_obj is None:
            from jarvis import brain as brain_mod   # late: brain imports tools
            self._brain_obj = brain_mod
        return self._brain_obj

    @property
    def lent_to(self) -> Optional[int]:
        """The trainer pid the model is currently lent to (None when not)."""
        return self._lent_to

    def _trainer_rule(self, snap: Snapshot, fired: list) -> None:
        """Lend the model when a trainer appears; take it back once the
        trainer has been gone for TRAINER_ABSENT_TICKS ticks (a run that
        restarts between epochs must not cost a 7 s reload each time)."""
        trainers = [p for p in (snap.trainers or []) if p.pid not in self._held]
        present = {p.pid for p in trainers}
        # forget a held pid once it is really gone, so a later run can lend
        self._held &= {p.pid for p in (snap.trainers or [])}
        if self._lent_to is None:
            if not trainers:
                return
            lead = trainers[0]
            try:
                ok = self._brain().release(reason=f"trainer pid {lead.pid} {lead.hint}")
            except Exception:  # noqa: BLE001
                log.exception("health watchdog: brain.release failed")
                return
            self._lent_to = lead.pid
            self._absent_ticks = 0
            fired.append(Alert(kind="warn", line=LENT_LINE,
                               status=f"GPU lent to {lead.hint or lead.name}"
                                      f"{'' if ok else ' (unload failed)'}",
                               rule="trainer"))
            return
        if self._lent_to in present or present:
            # the lent-to run continues, or another trainer took over: keep
            # the GPU lent, and follow the newest occupant
            if self._lent_to not in present:
                self._lent_to = trainers[0].pid
            self._absent_ticks = 0
            return
        self._absent_ticks += 1
        if self._absent_ticks < TRAINER_ABSENT_TICKS:
            return
        try:
            ok = self._brain().reclaim()
        except Exception:  # noqa: BLE001
            log.exception("health watchdog: brain.reclaim failed")
            ok = False
        self._lent_to = None
        self._absent_ticks = 0
        fired.append(Alert(kind="ok" if ok else "warn", line=RECLAIMED_LINE,
                           status="GPU reclaimed" if ok else "GPU reclaimed; model failed to load",
                           rule="trainer"))

    def manual_reclaim(self) -> bool:
        """"Take the GPU back": reclaim now and hold off the trainer rule
        for the run that is still going. Returns the warm-up verdict."""
        if self._lent_to is not None:
            self._held.add(self._lent_to)
        self._lent_to = None
        self._absent_ticks = 0
        try:
            return bool(self._brain().reclaim())
        except Exception:  # noqa: BLE001
            log.exception("health watchdog: manual reclaim failed")
            return False

    def _clear_fault(self, rule: str) -> None:
        """The episode ended: lift the board's FAULT lane and make the next
        occurrence news again. The board takes its clear from HERE -- a
        second latch in the UI drifts out of sync on recovery."""
        self._safe_publish(FaultRaised(rule=rule, cleared=True))
        tokens = self._fault_tokens.pop(rule, ())
        if self._faults is None:
            return
        for token in tokens:
            try:
                self._faults.clear(f"{rule}:{token}")
            except Exception:  # noqa: BLE001
                log.exception("fault log clear failed")

    def _fire(self, alert: Alert) -> None:
        log.warning("health watchdog [%s/%s]: %s", alert.rule, alert.kind, alert.status)
        self._safe_publish(Status(text=alert.status, kind=alert.kind))
        speak_it = True
        if alert.rule in FAULT_RULES:
            from jarvis.faults import token_for      # late: faults reads config
            token = token_for(alert.rule, alert.status)
            if self._faults is not None:
                try:
                    speak_it = bool(self._faults.should_speak(f"{alert.rule}:{token}"))
                except Exception:  # noqa: BLE001 - dedupe must not cost the alarm
                    log.exception("fault log read failed; speaking anyway")
            self._fault_tokens.setdefault(alert.rule, set()).add(token)
            self._safe_publish(FaultRaised(rule=alert.rule, kind=alert.kind,
                                           token=token, text=alert.status,
                                           line=alert.line))
        if not speak_it:
            return                    # already said once; the board still shows it
        speak = self._speak or (getattr(self._services, "speak", None)
                                if self._services is not None else None)
        if not callable(speak):
            return
        try:
            speak(alert.line)
        except Exception:  # noqa: BLE001 - a TTS failure must not kill the loop
            log.exception("health watchdog: speak failed")

    def _safe_publish(self, event) -> None:
        try:
            self._publish(event)
        except Exception:  # noqa: BLE001
            log.exception("health watchdog: publish failed")

    # --------------------------------------------------------- thread
    def tick(self) -> list[Alert]:
        try:
            return self.check(snapshot(gpu=False))
        except Exception:  # noqa: BLE001 - the loop must survive anything
            log.exception("health watchdog: tick failed")
            return []

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="health-watchdog",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(max(0.01, self.interval))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# --------------------------------------------------------------- tool
def make_tools(cfg, services) -> list[ToolSpec]:
    warn_gb = _cfg_float(cfg, "health.warn_gb", DEFAULT_WARN_GB)
    hog_gb = _cfg_float(cfg, "health.hog_gb", DEFAULT_HOG_GB)

    # Parked, not started: the integrator starts it beside the timekeeper
    # once services.speak exists (the same idiom as calendar.make_tools).
    if services is not None and getattr(services, "health_watchdog", None) is None:
        try:
            services.health_watchdog = Watchdog(cfg, services=services)
        except (AttributeError, TypeError):
            log.debug("services does not accept health_watchdog")

    def system_health(**_) -> ToolResult:
        # The TOOL may ask nvidia-smi (Popen + kill-without-wait, bounded):
        # "how's the Spark doing" is a question about the GPU. Only the
        # 30 s watchdog tick stays gpu=False -- a wedged nvidia-smi there
        # would pile up a zombie every half minute.
        try:
            snap = snapshot()
        except Exception:  # noqa: BLE001 - belt and braces; snapshot() guards
            log.exception("system_health: snapshot failed")
            return ToolResult(text="system counters unreadable", ok=False,
                              speak=UNREADABLE_LINE)
        if not snap.readable:
            return ToolResult(text="system counters unreadable", ok=False,
                              speak=UNREADABLE_LINE)
        return ToolResult(text=fact_sheet(snap, warn_gb=warn_gb, hog_gb=hog_gb),
                          max_sentences=3)

    return [ToolSpec(
        name="system_health",
        description=("Spark system health: memory, GPU temperature and clocks, "
                     "load, disk and the heaviest processes."),
        parameters={"type": "object", "properties": {}},
        handler=system_health)]
