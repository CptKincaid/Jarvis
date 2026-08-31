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

The GPU-yield rule lends the local model (``brain.release``) to whatever
holds the GPU: a trainer by pattern, plus any job NAMED in
``health.yield_to`` -- the nightly haymaker digest is not a trainer and
starved behind a pinned gemma4:26b for ~50 minutes before it was named.

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

from jarvis.events import FaultRaised, RunProgress, Status, bus
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
# A GPU CLAIMANT that is not a trainer has to be NAMED, because guessing it
# off the command line is exactly what failed. 2026-08-30: the haymaker
# digest ("/usr/bin/python3 -u digest_llm.py --cache ./digest_cache.json
# ...") asked Ollama for qwen2.5:32b and sat STARVED for ~50 minutes -- no
# runner spawned, no [GIN] line, no progress -- behind Jarvis's gemma4:26b,
# pinned with keep_alive -1 (brain.py) and re-warmed every 300 s. Nothing
# in "digest_llm.py" says "train", so TRAINER_HINT_RX never saw it and the
# whole lend/reclaim machinery below never ran. Saying "lend the GPU" by
# voice freed it and the digest loaded within seconds: the ONLY gap was
# this detector. health.yield_to NAMES the claimants; the trainer regex
# above still stands on its own.
DEFAULT_YIELD_TO = ("digest_llm",)
# How a claimant is spoken about. Anything not listed is read off its
# script name ("render_job" -> "the render job").
CLAIMANT_WORDS = {"digest_llm": "the digest"}
TRAINER_WORDS = "your trainer"
TRAINER_ABSENT_TICKS = 2           # ticks without the claimant before reclaiming
# health.yield_to_trainer's fallback when the cfg object carries no health
# section at all (a bare dict, or a boot before assistant.json is read).
# The USER-FACING default lives in assistant_config.DEFAULTS and is now
# True; this one stays False so "no configuration whatsoever" never
# unloads the model behind a process that merely LOOKS like a trainer.
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
# The same two beats for a NAMED claimant: calling the nightly digest "your
# trainer" would be a small lie every night at 04:09.
LENT_TO_LINE = ("I have lent the GPU to {what}, sir; quick answers only "
                "until it is done.")
RECLAIMED_TO_LINE = "{what} has finished, sir; I'm loading my model again."
RECLAIMED_TO_ELAPSED_LINE = ("{what} has finished, sir; that took {elapsed}. "
                             "I'm loading my model again.")
# The run ledger's duration, folded ONTO the reclaim line rather than
# spoken beside it: the count of spoken lines per run must not go up.
RECLAIMED_ELAPSED_LINE = ("Your trainer has finished, sir; that took {elapsed}. "
                          "I'm loading my model again.")


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
    # "" a plain process | "trainer" (TRAINER_HINT_RX) | the health.yield_to
    # name it matched. The lend rule reads this to decide whether the flag
    # for trainers applies and what to call the job out loud.
    claim: str = ""

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
    # Trainers PLUS the jobs named in health.yield_to (a superset of
    # ``trainers``): what the GPU-yield rule lends to. Kept separate so the
    # two-trainers fault and the run ledger keep counting training runs only.
    claimants: list[Proc] = field(default_factory=list)

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


def _claim_key(text: str) -> str:
    """A claimant name reduced to its comparable stem: no directory, no
    extension, lower case -- "./digest_llm.py" and "digest_llm" are one
    name, which is what lets the config name a job the way a human would."""
    stem = os.path.basename(str(text or "").strip()).lower()
    for ext in (".py", ".sh", ".pyc"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return stem


def claimant_names(value) -> tuple[str, ...]:
    """health.yield_to -> the claimant stems to look for. Takes a list or a
    comma-separated string. Interpreter names are DROPPED: "python3" in the
    list would make every script on the box a GPU claimant."""
    if value is None:
        return ()
    try:
        items = value.split(",") if isinstance(value, str) else list(value)
    except TypeError:            # a number or a dict in the config file
        log.warning("health.yield_to: %r is not a list of names", value)
        return ()
    out: list[str] = []
    for item in items:
        key = _claim_key(item)
        if not key or key in out:
            continue
        if key in _INTERPRETERS or key in _TRAINER_NAMES:
            log.warning("health.yield_to: ignoring %r -- an interpreter name "
                        "would match every script on the box", item)
            continue
        out.append(key)
    return tuple(out)


def claim_of(argv: list[str], names=()) -> str:
    """What this command line claims the GPU as: "trainer" when it looks
    like a training run, else the health.yield_to name it matches, else "".
    The named branch is the one the starved digest needed -- its argv is
    ["/usr/bin/python3", "-u", "digest_llm.py", "--cache", ...]."""
    if is_trainer(argv):
        return "trainer"
    if not names:
        return ""
    hint = cmdline_hint(argv)
    if hint and _claim_key(hint) in names:
        return _claim_key(hint)
    # A claimant that is not run through an interpreter (a compiled job, or
    # a wrapper script): match any non-flag word of the command line.
    for arg in argv:
        if arg.startswith("-"):
            continue
        key = _claim_key(arg)
        if key in names:
            return key
    return ""


def find_claimants(self_pid: Optional[int] = None, names=()) -> list[Proc]:
    """Every process claiming the GPU -- trainers by pattern plus the jobs
    named in health.yield_to -- largest first. Only processes that COULD be
    a claimant have their cmdline read (an interpreter, or a name from the
    config), so this stays a few dozen small files and not the whole
    table. Never raises."""
    me = os.getpid() if self_pid is None else self_pid
    names = tuple(names or ())
    out: list[Proc] = []
    try:
        for pid, name, rss_kb in iter_process_rss():
            lowered = name.lower()
            if pid == me or (lowered not in _TRAINER_NAMES
                             and _claim_key(lowered) not in names):
                continue
            try:
                argv = read_cmdline(pid)
            except Exception:  # noqa: BLE001
                continue
            claim = claim_of(argv, names)
            if not claim:
                continue
            hint = cmdline_hint(argv)
            if not hint:
                # A named claimant that is not run through an interpreter has
                # no script to point at; its own name beats argv[-1], which
                # would put a flag ("--gpu") in the spoken status line.
                hint = claim if claim != "trainer" else \
                    os.path.basename(argv[-1] if argv else "")
            out.append(Proc(pid=pid, name=name, rss_gb=rss_kb / KB_PER_GB,
                            hint=hint, claim=claim))
    except Exception:  # noqa: BLE001 - a bad /proc must not sink the tick
        log.debug("claimant scan failed", exc_info=True)
    out.sort(key=lambda p: -p.rss_gb)
    return out


def find_trainers(self_pid: Optional[int] = None) -> list[Proc]:
    """Every process whose command line looks like a TRAINING run, largest
    first: the two-trainers rule and the run ledger count these, and a
    digest is not one of them. Never raises."""
    return [p for p in find_claimants(self_pid) if p.claim == "trainer"]


def snapshot(gpu: bool = True, yield_to=()) -> Snapshot:
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
    # One /proc pass for both: trainers are the claimants that match the
    # training pattern.
    snap.claimants = find_claimants(names=yield_to)
    snap.trainers = [p for p in snap.claimants if p.claim == "trainer"]
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


def claimant_words(proc: Proc) -> str:
    """What a claimant is called out loud. A Proc with no claim is a
    trainer -- that is all this rule knew before named claimants existed."""
    if proc.claim in ("", "trainer"):
        return TRAINER_WORDS
    return CLAIMANT_WORDS.get(proc.claim, f"the {proc.claim.replace('_', ' ')} job")


def lent_line(words: str = TRAINER_WORDS) -> str:
    return LENT_LINE if words == TRAINER_WORDS else LENT_TO_LINE.format(what=words)


def reclaimed_line(words: str = TRAINER_WORDS, elapsed: str = "") -> str:
    """The reclaim beat, with the run ledger's duration folded in when there
    is one (the count of spoken lines per run must not go up)."""
    if words == TRAINER_WORDS:
        return RECLAIMED_ELAPSED_LINE.format(elapsed=elapsed) if elapsed \
            else RECLAIMED_LINE
    what = words[0].upper() + words[1:]
    return RECLAIMED_TO_ELAPSED_LINE.format(what=what, elapsed=elapsed) if elapsed \
        else RECLAIMED_TO_LINE.format(what=what)


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
                 interval: Optional[float] = None, brain=None, faults=None,
                 runs=None):
        # GPU yield (health.yield_to_trainer): ``brain`` is anything with
        # release() / reclaim() / is_lent(); None means jarvis.brain itself,
        # imported at fire time (tests pass a fake).
        self.yield_to_trainer = bool(_cfg_get(cfg, "health.yield_to_trainer",
                                              DEFAULT_YIELD))
        # Named GPU claimants (health.yield_to). Consulted even when
        # yield_to_trainer is off: that flag speaks about processes GUESSED
        # to be trainers, while a name in this list is Hunter saying "this
        # job takes the GPU" -- and the job that starved (the nightly
        # haymaker digest) is not a trainer at all.
        self.yield_to = claimant_names(_cfg_get(cfg, "health.yield_to",
                                                DEFAULT_YIELD_TO))
        self._brain_obj = brain
        self._lent_to: Optional[int] = None      # claimant pid the model is lent to
        self._lent_words = TRAINER_WORDS         # what to call it out loud
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
        # jarvis.runwatch.RunLedger when the integrator wires one: the
        # start/finish beats and the opt-in epoch progress, driven off THIS
        # tick (no second thread, and no nvidia-smi -- the wedge the run
        # ledger narrates around is exactly when nvidia-smi blocks).
        self.runs = runs
        self._run_finished = None      # the ledger's finish, for RECLAIMED_LINE
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
        self._run_ledger(snap, fired)
        if self.yield_to_trainer or self.yield_to:
            self._trainer_rule(snap, fired)
        for alert in fired:
            self._fire(alert)
        return fired

    # ------------------------------------------------------ run ledger
    def _run_ledger(self, snap: Snapshot, fired: list) -> None:
        """Drive the run ledger off this tick and turn its beats into
        spoken lines and RunProgress events.

        When the GPU-yield rule is on it owns the two lifecycle beats
        already (LENT_LINE on appear, RECLAIMED_LINE on vanish, which
        picks up the duration below), so the ledger stays silent and only
        publishes -- the spoken line count per run must not go up. With
        yield off, the ledger's own two beats are the whole feature.
        """
        self._run_finished = None
        ledger = self.runs
        if ledger is None:
            return
        try:
            events = ledger.apply(list(snap.trainers or []))
        except Exception:  # noqa: BLE001 - the tick must survive anything
            log.exception("run ledger failed")
            return
        for ev in events:
            if ev.kind == "finished":
                self._run_finished = ev
            quiet = getattr(ledger, "muted", False) or \
                not getattr(ledger, "narrate", True)
            line = "" if quiet else ev.line()
            # The lend/reclaim lines already cover appear and vanish.
            if self.yield_to_trainer and ev.kind in ("started", "finished"):
                line = ""
            self._safe_publish(RunProgress(
                kind=ev.kind, pid=ev.pid, label=ev.label,
                elapsed_s=ev.elapsed_s, epoch=ev.epoch,
                loss=float(ev.loss or 0.0), line=line))
            # Spoken directly, NOT through _fire: an alert would publish a
            # Status, and an ok/info Status clears main_window's held ERROR
            # pill -- a routine "epoch four, sir" must never wipe a fault
            # off the board.
            if line:
                self._speak_line(line)

    # ---------------------------------------------------- trainer yield
    def _brain(self):
        if self._brain_obj is None:
            from jarvis import brain as brain_mod   # late: brain imports tools
            self._brain_obj = brain_mod
        return self._brain_obj

    @property
    def lent_to(self) -> Optional[int]:
        """The claimant pid the model is currently lent to (None when not)."""
        return self._lent_to

    def _claimants(self, snap: Snapshot) -> list[Proc]:
        """The processes this tick may lend the GPU to, largest first. A
        Proc with no claim counts as a trainer -- that is what every Proc
        was before named claimants existed -- so yield_to_trainer still
        gates the pattern-guessed ones and only those."""
        procs = list(snap.claimants or snap.trainers or [])
        if not self.yield_to_trainer:
            procs = [p for p in procs if p.claim not in ("", "trainer")]
        return [p for p in procs if p.pid not in self._held]

    def _trainer_rule(self, snap: Snapshot, fired: list) -> None:
        """Lend the model when a GPU claimant appears; take it back once it
        has been gone for TRAINER_ABSENT_TICKS ticks (a run that restarts
        between epochs must not cost a 7 s reload each time)."""
        trainers = self._claimants(snap)
        present = {p.pid for p in trainers}
        # forget a held pid once it is really gone, so a later run can lend
        self._held &= {p.pid for p in (snap.claimants or snap.trainers or [])}
        if self._lent_to is None:
            if not trainers:
                return
            lead = trainers[0]
            try:
                ok = self._brain().release(
                    reason=f"{lead.claim or 'trainer'} pid {lead.pid} {lead.hint}")
            except Exception:  # noqa: BLE001
                log.exception("health watchdog: brain.release failed")
                return
            self._lent_to = lead.pid
            self._lent_words = claimant_words(lead)
            self._absent_ticks = 0
            fired.append(Alert(kind="warn", line=lent_line(self._lent_words),
                               status=f"GPU lent to {lead.hint or lead.name}"
                                      f"{'' if ok else ' (unload failed)'}",
                               rule="trainer"))
            return
        if self._lent_to in present or present:
            # the lent-to run continues, or another claimant took over: keep
            # the GPU lent, and follow the newest occupant
            if self._lent_to not in present:
                self._lent_to = trainers[0].pid
                self._lent_words = claimant_words(trainers[0])
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
        # The run ledger's finish for this tick, folded in: "that took 22
        # minutes" belongs ON this line, not spoken after it.
        done = self._run_finished
        elapsed = ""
        if done is not None and not done.brief:
            from jarvis.runwatch import elapsed_words
            elapsed = elapsed_words(done.elapsed_s)
        line = reclaimed_line(self._lent_words, elapsed)
        self._lent_words = TRAINER_WORDS
        fired.append(Alert(kind="ok" if ok else "warn", line=line,
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
        self._speak_line(alert.line)

    def _speak_line(self, line: str) -> None:
        """Say one line through whatever speak callback exists. Resolved at
        fire time from services.speak so boot order does not matter, and
        that door is proactive=True -- quiet hours hold it for the digest
        rather than narrating a 3 am run."""
        speak = self._speak or (getattr(self._services, "speak", None)
                                if self._services is not None else None)
        if not callable(speak) or not line:
            return
        try:
            speak(line)
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
            return self.check(snapshot(gpu=False, yield_to=self.yield_to))
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


def make_run_ledger(cfg):
    """The run ledger from config, or None when it cannot be built. Reads
    nothing here: log_dir is only opened once runwatch.progress is on AND
    a run is going, so the default costs a dict lookup per tick."""
    try:
        from jarvis.runwatch import RunLedger
        log_dir = _cfg_get(cfg, "runwatch.log_dir", "") \
            if _cfg_get(cfg, "runwatch.progress", False) else ""
        return RunLedger(log_dir=log_dir or None,
                         narrate=bool(_cfg_get(cfg, "runwatch.narrate", True)),
                         min_run_s=_cfg_float(cfg, "runwatch.min_run_s", 60.0),
                         progress_gap_s=_cfg_float(cfg, "runwatch.progress_gap_s",
                                                   300.0))
    except Exception:  # noqa: BLE001 - no ledger is better than no watchdog
        log.exception("run ledger unavailable")
        return None


# --------------------------------------------------------------- tool
def make_tools(cfg, services) -> list[ToolSpec]:
    warn_gb = _cfg_float(cfg, "health.warn_gb", DEFAULT_WARN_GB)
    hog_gb = _cfg_float(cfg, "health.hog_gb", DEFAULT_HOG_GB)

    # Parked, not started: the integrator starts it beside the timekeeper
    # once services.speak exists (the same idiom as calendar.make_tools).
    if services is not None and getattr(services, "health_watchdog", None) is None:
        try:
            services.health_watchdog = Watchdog(cfg, services=services,
                                                runs=make_run_ledger(cfg))
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
