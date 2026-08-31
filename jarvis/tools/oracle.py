"""Hunter's Oracle Cloud VM, read-mostly and OUTBOUND ONLY.

The box is an Oracle Linux VM (``opc@170.9.245.136``) running a Discord
"game-news" bot under pm2 out of ``~/Game-News``. Jarvis asks it questions
over ssh and runs a SHORT ALLOW-LIST of named commands. He never opens
anything the other way: there is no tunnel, no reverse tunnel, no
port-forward and no exposure of the Spark here, by design -- the Spark sits
behind the house NAT and it stays there.

Three rules this module exists to keep:

1. **Nothing free-form ever reaches a shell.** ``resolve_action`` maps a
   SPOKEN NAME to an exact command out of ``oracle.actions`` in the
   assistant config, and nothing else. Not one character of a transcript is
   ever interpolated into a command -- a misheard word can only ever fail
   to match a key, and a key that does not match is refused out loud.
2. **Nothing hangs the turn.** Every call is Popen + ``communicate(timeout)``
   + kill-WITHOUT-wait, the same shape as ``health.run_nvidia_smi``: a
   ``subprocess.run`` timeout kills the child and then *waits* for it, and a
   TCP connect stuck against a firewalled host is exactly the wedge that
   never returns. Budget is ``oracle.timeout_s`` (6 s), and when the box
   cannot answer inside it Jarvis says so rather than holding the floor.
3. **Unconfigured is safe and honest.** ``enabled`` is false by default and
   ``key_path`` ships empty. ``missing_reason`` names EXACTLY what is
   absent, and every entry point checks it before anything opens a socket.

``run_ssh`` is the single module-level seam -- looked up at call time, so a
test replaces it and no test in this repo can reach the network.

The last good reading is cached for ``oracle.cache_s`` seconds so "how's the
Oracle box" followed by "and the bot?" is one round trip, not two.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from jarvis.logs import get_logger
from jarvis.runwatch import elapsed_words
from jarvis.tools.health import parse_meminfo
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.oracle")

SSH_BIN = "ssh"
DEFAULT_TIMEOUT_S = 6.0
MIN_TIMEOUT_S = 2.0
MAX_TIMEOUT_S = 30.0
DEFAULT_CACHE_S = 25.0
CONFIG_HINT = "~/.config/jarvis/assistant.json"
DOCS_HINT = "docs/assistant-setup.md"
KB_PER_GB = 1024 * 1024                # /proc and df -Pk both report kB
LOG_LINE_CAP = 40                      # lines of `pm2 logs` kept for the card
CARD_CHAR_CAP = 4000                   # a card is read, not scrolled forever

# pm2 is installed through npm, and on this box npm is under nvm. A
# non-interactive `ssh host 'pm2 status'` runs a NON-login shell, which never
# sources the ~/.bashrc block that puts nvm's node on PATH -- so the exact
# command from the cheat sheet answers "pm2: command not found" even though
# it works the moment you ssh in and type it by hand. Every remote command
# gets this prelude so the config can hold the plain command the cheat sheet
# documents. POSIX sh only: Oracle Linux's /bin/sh is not bash.
REMOTE_PRELUDE = (
    'for d in "$HOME"/.nvm/versions/node/*/bin "$HOME"/.local/bin '
    '/usr/local/bin; do [ -d "$d" ] && PATH="$d:$PATH"; done; export PATH; '
)

# ONE round trip for the whole status answer. Everything but pm2 comes out
# of /proc and df, which cannot hang and cannot be missing; pm2 is allowed
# to fail on its own (`|| echo []`) so a dead pm2 still returns a box
# reading instead of an empty answer.
STATUS_SECTIONS = ("uptime", "load", "mem", "disk", "pm2")
STATUS_COMMAND = (
    "echo '#uptime'; cat /proc/uptime; "
    "echo '#load'; cat /proc/loadavg; "
    "echo '#mem'; cat /proc/meminfo; "
    "echo '#disk'; df -Pk /; "
    "echo '#pm2'; pm2 jlist 2>/dev/null || echo '[]'"
)

# What "this changes something" looks like in a command. Applied to the
# COMMAND, not to the spoken name, so an action Hunter adds to the table
# later ("deploy" = git pull + npm install + pm2 restart) is read back for a
# yes on its own without him having to remember to flag it.
STATE_CHANGE_RX = re.compile(
    r"(?:^|[^a-z])(?:restart|reload|stop|start|kill|delete|rm|reboot|shutdown|"
    r"poweroff|halt|install|update|upgrade|npm|git|systemctl|mv|cp|chmod|"
    r"chown|truncate|tee|dd)(?:[^a-z]|$)", re.I)

# ssh's own diagnosis, in the three registers worth telling him apart.
_AUTH_RX = re.compile(
    r"permission denied|no supported authentication|too many authentication|"
    r"unprotected private key|bad permissions|invalid format|"
    r"authentication failed", re.I)
_REACH_RX = re.compile(
    r"connection refused|no route to host|network is unreachable|"
    r"could not resolve|name or service not known|connection timed out|"
    r"operation timed out|connection closed by|broken pipe|"
    r"kex_exchange_identification", re.I)
# What counts as "mentions an error" in a log line. Deliberately literal:
# the spoken line claims a MENTION, never a diagnosis.
_ERROR_RX = re.compile(r"error|exception|traceback|fatal|\berr\b|rejection",
                       re.I)
_HOSTKEY_RX = re.compile(
    r"host key verification failed|remote host identification has changed|"
    r"known_hosts", re.I)

# Every failure gets a line that names the failure. None of them claims a
# reading Jarvis does not have -- the whole point of the 6 s budget is that
# he says "it did not answer" instead of sitting on the turn.
FAIL_LINES = {
    "timeout": "The Oracle box didn't answer in {timeout} seconds, sir; "
               "I've stopped waiting on it.",
    "auth": "The Oracle box turned my key away, sir.",
    "unreachable": "I couldn't reach the Oracle box, sir.",
    "hostkey": "The Oracle box's host key isn't the one I know, sir; "
               "I'd look at that before I talk to it again.",
    "no-ssh": "I've no ssh on this machine, sir.",
    "failed": "The Oracle box wouldn't answer that, sir.",
}
NOT_ENABLED_LINE = ("The Oracle box is switched off in my settings, sir; "
                    f"set oracle.enabled to true in {CONFIG_HINT}.")
NO_HOST_LINE = (f"I've no address for the Oracle box, sir; oracle.host in "
                f"{CONFIG_HINT} is empty.")
NO_KEY_LINE = ("I've no key for the Oracle box, sir; set oracle.key_path in "
               f"{CONFIG_HINT}. The notes are in {DOCS_HINT}.")
KEY_MISSING_LINE = ("I can't find the Oracle key at {path}, sir.")
KEY_OPEN_LINE = ("The Oracle key at {path} is readable by others, sir; "
                 "ssh will refuse it until you chmod 600 it.")
UNKNOWN_ACTION_LINE = ("I don't do {spoken} on the Oracle box, sir. "
                       "I know {known}.")
NO_ACTIONS_LINE = ("There are no Oracle actions in my settings, sir; "
                   f"oracle.actions in {CONFIG_HINT} is empty.")
READ_BACK_LINE = "{spoken} on the Oracle box, sir?"
DONE_LINE = "Done, sir."
NOTHING_BACK_LINE = "It ran without a word back, sir."

# Filler that must not decide whether a spoken name is on the list.
_FILLER = {"the", "a", "an", "my", "our", "please", "on", "oracle", "box",
           "server", "vm", "cloud"}


# --------------------------------------------------------------- config
def _cfg_get(cfg, dotted: str, default=None):
    """One dotted read that works for AssistantConfig, a plain dict and the
    SimpleNamespace fakes the tests use. Never raises."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter) and not isinstance(cfg, dict):
        try:
            val = getter(dotted, default)
            return default if val is None else val
        except Exception:                   # noqa: BLE001 - config boundary
            log.debug("oracle: cfg.get(%s) failed", dotted, exc_info=True)
    cur = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


@dataclass
class OracleConfig:
    enabled: bool = False
    host: str = ""
    user: str = ""
    key_path: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S
    cache_s: float = DEFAULT_CACHE_S
    actions: dict = field(default_factory=dict)

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    @property
    def key_file(self) -> str:
        return os.path.expanduser(self.key_path) if self.key_path else ""


def read_config(cfg) -> OracleConfig:
    """The ``oracle`` section, read fresh. The commander reads per call so
    that turning the box on in the config file takes effect on the next
    question rather than on the next restart."""
    def _float(key, default, low, high):
        try:
            return max(low, min(high, float(_cfg_get(cfg, key, default))))
        except (TypeError, ValueError):
            return float(default)

    actions = _cfg_get(cfg, "oracle.actions", None)
    return OracleConfig(
        enabled=bool(_cfg_get(cfg, "oracle.enabled", False)),
        host=str(_cfg_get(cfg, "oracle.host", "") or "").strip(),
        user=str(_cfg_get(cfg, "oracle.user", "") or "").strip(),
        key_path=str(_cfg_get(cfg, "oracle.key_path", "") or "").strip(),
        timeout_s=_float("oracle.timeout_s", DEFAULT_TIMEOUT_S,
                         MIN_TIMEOUT_S, MAX_TIMEOUT_S),
        cache_s=_float("oracle.cache_s", DEFAULT_CACHE_S, 0.0, 600.0),
        actions=dict(actions) if isinstance(actions, dict) else {})


def missing_reason(conf: OracleConfig) -> Optional[str]:
    """The ONE honest line naming what is missing, or None when the lane is
    ready. Checked by every entry point BEFORE anything opens a socket, so
    an unconfigured install never touches the network at all.

    The key-permission check is here rather than left to ssh because the
    likely way this key arrives is a copy off a Windows desktop, which lands
    it 0644 -- and ssh's own refusal ("UNPROTECTED PRIVATE KEY FILE") would
    reach him as the generic auth line instead of the fix."""
    if not conf.enabled:
        return NOT_ENABLED_LINE
    if not conf.host:
        return NO_HOST_LINE
    if not conf.key_path:
        return NO_KEY_LINE
    path = conf.key_file
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return KEY_MISSING_LINE.format(path=conf.key_path)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return KEY_OPEN_LINE.format(path=conf.key_path)
    return None


# ----------------------------------------------------------------- ssh
@dataclass
class SshResult:
    ok: bool
    out: str = ""
    err: str = ""
    reason: str = ""          # "" on success, else a FAIL_LINES key


def classify_error(err: str) -> str:
    """ssh's stderr in one word. Order matters: a changed host key also says
    "Permission denied" further down, and the host key is the one he must
    look at himself."""
    text = err or ""
    if _HOSTKEY_RX.search(text):
        return "hostkey"
    if _AUTH_RX.search(text):
        return "auth"
    if _REACH_RX.search(text):
        return "unreachable"
    return "failed"


def ssh_argv(conf: OracleConfig, command: str) -> list:
    """The exact argv. No shell on THIS side ever (no shell=True, no string
    command), and on the far side the command is the config's own text.

    ``BatchMode=yes`` plus ``NumberOfPasswordPrompts=0`` is what makes the
    timeout meaningful: without them a key with a passphrase, or a host that
    falls back to password auth, sits on a prompt for the whole budget and
    then dies without having asked anything."""
    connect = max(1, int(conf.timeout_s) - 1)
    argv = [SSH_BIN, "-n",
            "-o", "BatchMode=yes",
            "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no",
            "-o", "NumberOfPasswordPrompts=0",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={connect}",
            "-o", "IdentitiesOnly=yes"]
    if conf.key_file:
        argv += ["-i", conf.key_file]
    argv += ["--", conf.target, REMOTE_PRELUDE + command]
    return argv


def run_ssh(conf: OracleConfig, command: str) -> SshResult:
    """THE SEAM. One bounded ssh round trip.

    Popen + communicate(timeout) + kill-without-wait, never subprocess.run:
    run() kills the child on a timeout and then waits for it, and an ssh
    stuck in a TCP connect to a host that silently drops packets is exactly
    the child that does not come back. Kill it and walk away -- a leaked
    zombie costs nothing next to a wedged voice turn.

    stdin is /dev/null (``-n`` and ``stdin=DEVNULL``, belt and braces): an
    ssh that inherits the app's stdin can block reading it.
    """
    argv = ssh_argv(conf, command)
    log.info("oracle: ssh %s (%.0fs budget)", conf.target, conf.timeout_s)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("oracle: ssh unavailable: %s", type(exc).__name__)
        return SshResult(False, reason="no-ssh")
    try:
        out, err = proc.communicate(timeout=conf.timeout_s)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        log.warning("oracle: ssh did not answer in %.0fs; not waiting for it",
                    conf.timeout_s)
        return SshResult(False, reason="timeout")
    if proc.returncode != 0:
        reason = classify_error(err)
        log.warning("oracle: ssh rc=%s (%s): %s", proc.returncode, reason,
                    (err or "").strip()[:200])
        return SshResult(False, out=out or "", err=err or "", reason=reason)
    return SshResult(True, out=out or "", err=err or "")


def fail_line(conf: OracleConfig, reason: str) -> str:
    line = FAIL_LINES.get(reason) or FAIL_LINES["failed"]
    return line.format(timeout=int(conf.timeout_s))


# --------------------------------------------------------------- parsing
@dataclass
class Proc:
    name: str = ""
    status: str = ""
    restarts: int = 0
    uptime_s: Optional[float] = None
    cpu: Optional[float] = None
    mem_kb: Optional[int] = None

    @property
    def online(self) -> bool:
        return self.status == "online"


@dataclass
class Reading:
    uptime_s: Optional[float] = None
    load: Optional[tuple] = None
    mem_total_kb: Optional[int] = None
    mem_avail_kb: Optional[int] = None
    disk_total_kb: Optional[int] = None
    disk_free_kb: Optional[int] = None
    procs: list = field(default_factory=list)
    pm2_ok: bool = False              # pm2 answered (even with no apps)
    at: float = 0.0                   # monotonic stamp for the cache

    @property
    def readable(self) -> bool:
        return self.uptime_s is not None or bool(self.procs)


def _split_sections(text: str) -> dict:
    """The ``#name``-delimited blocks of STATUS_COMMAND's output."""
    out, current, buf = {}, None, []
    for line in (text or "").splitlines():
        name = line[1:].strip() if line.startswith("#") else ""
        if name in STATUS_SECTIONS:
            if current:
                out[current] = "\n".join(buf)
            current, buf = name, []
            continue
        if current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf)
    return out


def parse_pm2(text: str) -> tuple:
    """(procs, answered) from ``pm2 jlist``.

    ``raw_decode`` from the first ``[`` rather than json.loads on the whole
    string: pm2 prints update banners and node deprecation warnings to
    stdout on some versions, and one stray line would otherwise cost the
    whole process list."""
    start = (text or "").find("[")
    if start < 0:
        return [], False
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        log.warning("oracle: pm2 jlist was not JSON (%d bytes)", len(text or ""))
        return [], False
    if not isinstance(data, list):
        return [], False
    procs = []
    now_ms = time.time() * 1000.0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        env = entry.get("pm2_env") if isinstance(entry.get("pm2_env"), dict) else {}
        monit = entry.get("monit") if isinstance(entry.get("monit"), dict) else {}
        up = None
        started = env.get("pm_uptime")
        if isinstance(started, (int, float)) and str(env.get("status")) == "online":
            up = max(0.0, (now_ms - float(started)) / 1000.0)
        mem = monit.get("memory")
        procs.append(Proc(
            name=str(entry.get("name") or "?")[:40],
            status=str(env.get("status") or "?"),
            restarts=int(env.get("restart_time") or 0),
            uptime_s=up,
            cpu=float(monit["cpu"]) if isinstance(monit.get("cpu"), (int, float)) else None,
            mem_kb=int(mem) // 1024 if isinstance(mem, (int, float)) else None))
    return procs, True


def parse_status(text: str) -> Reading:
    """One STATUS_COMMAND payload -> a Reading. Never raises: a section that
    does not parse simply leaves its fields None, exactly like health.py's
    probes, because a partial answer still tells him whether the bot is up."""
    sections = _split_sections(text)
    r = Reading()
    try:
        r.uptime_s = float(sections.get("uptime", "").split()[0])
    except (IndexError, ValueError):
        pass
    try:
        parts = sections.get("load", "").split()
        r.load = (float(parts[0]), float(parts[1]), float(parts[2]))
    except (IndexError, ValueError):
        pass
    try:
        mem = parse_meminfo(sections.get("mem", ""))
        r.mem_total_kb = mem.get("MemTotal")
        # MemAvailable, not MemFree: on a 1 GB Always-Free VM the page cache
        # owns most of MemFree and "12 megabytes free" would be a lie.
        r.mem_avail_kb = mem.get("MemAvailable", mem.get("MemFree"))
    except Exception:                        # noqa: BLE001 - probe boundary
        log.debug("oracle: meminfo unparsable", exc_info=True)
    for line in sections.get("disk", "").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1].isdigit() and parts[3].isdigit():
            r.disk_total_kb, r.disk_free_kb = int(parts[1]), int(parts[3])
            break
    r.procs, r.pm2_ok = parse_pm2(sections.get("pm2", ""))
    return r


# --------------------------------------------------------------- speech
def uptime_words(seconds: Optional[float]) -> str:
    """"40 days", "3 days and 4 hours", "an hour and 10 minutes".

    runwatch.elapsed_words tops out in hours -- correct for a training run,
    useless for a server that has been up since spring ("960 hours")."""
    if seconds is None:
        return "an unknown time"
    seconds = max(0.0, float(seconds))
    days = int(seconds // 86400)
    if days < 1:
        return elapsed_words(seconds)
    hours = int((seconds - days * 86400) // 3600)
    head = "a day" if days == 1 else f"{days} days"
    if not hours:
        return head
    return f"{head} and {hours} hour{'s' if hours != 1 else ''}"


def _gb(kb: Optional[int]) -> Optional[float]:
    return None if kb is None else round(kb / KB_PER_GB, 1)


def speak_line(conf: OracleConfig, r: Reading) -> str:
    """ONE sentence. The bot comes first -- "is the game-news bot up" is the
    question behind every phrasing of this command -- and the box's own
    numbers follow it as the same sentence's second clause."""
    if r is None or not r.readable:
        return "The Oracle box answered, sir, but not with anything I could read."
    box = f"the box has been up {uptime_words(r.uptime_s)}"
    free = _gb(r.mem_avail_kb)
    if free is not None:
        box += f" with {free} gigabytes free"
    down = [p for p in r.procs if not p.online]
    if down:
        p = down[0]
        return (f"{p.name} is {p.status} on the Oracle box, sir, after "
                f"{p.restarts} restart{'s' if p.restarts != 1 else ''}; {box}.")
    if r.procs:
        p = r.procs[0]
        extra = "" if len(r.procs) == 1 else \
            f" and {len(r.procs) - 1} other{'s' if len(r.procs) > 2 else ''}"
        return (f"{p.name} has been online {uptime_words(p.uptime_s)}{extra}, "
                f"sir; {box}.")
    if r.pm2_ok:
        return f"pm2 has nothing running on the Oracle box, sir; {box}."
    return f"I couldn't get a word out of pm2, sir; {box}."


def card(conf: OracleConfig, r: Reading) -> str:
    """The plain sheet behind the spoken line -- the numbers stay readable
    and the two can never disagree, because both come off this Reading."""
    lines = [f"Oracle {conf.target}"]
    head = f"up {uptime_words(r.uptime_s)}"
    if r.load:
        head += "  ·  load " + " ".join(f"{v:.2f}" for v in r.load)
    lines.append(head)
    total, free = _gb(r.mem_total_kb), _gb(r.mem_avail_kb)
    if total is not None and free is not None:
        lines.append(f"memory  {free} GB free of {total} GB")
    dtotal, dfree = _gb(r.disk_total_kb), _gb(r.disk_free_kb)
    if dtotal is not None and dfree is not None:
        lines.append(f"disk    {dfree} GB free of {dtotal} GB")
    lines.append("")
    if not r.pm2_ok:
        lines.append("pm2     no answer")
    elif not r.procs:
        lines.append("pm2     nothing running")
    else:
        for p in r.procs:
            bits = [f"{p.name:<16}", f"{p.status:<9}"]
            bits.append(f"up {uptime_words(p.uptime_s):<22}"
                        if p.uptime_s is not None else " " * 25)
            bits.append(f"{p.cpu:>3.0f}% cpu" if p.cpu is not None else "        ")
            bits.append(f"{p.mem_kb // 1024:>5} MB" if p.mem_kb is not None else "      ")
            bits.append(f"{p.restarts} restart{'s' if p.restarts != 1 else ''}")
            lines.append("  ".join(bits).rstrip())
    return "\n".join(lines)[:CARD_CHAR_CAP]


# ---------------------------------------------------------------- status
_cache: Optional[Reading] = None
_lock = threading.Lock()
_clock = time.monotonic              # test seam


def clear_cache() -> None:
    global _cache
    with _lock:
        _cache = None


def cached(conf: OracleConfig) -> Optional[Reading]:
    """The last good reading while it is younger than ``cache_s``.

    The point of the cache is the second question: "how's the Oracle box" /
    "…and how long has the bot been up" is one 6 s round trip, not two, and
    the second answer is instant instead of another ``done=False`` turn."""
    with _lock:
        r = _cache
    if r is None or conf.cache_s <= 0:
        return None
    return r if _clock() - r.at < conf.cache_s else None


def status(conf: OracleConfig, force: bool = False) -> tuple:
    """(Reading, reason). reason is "" on success, else a FAIL_LINES key.

    Callers must have cleared ``missing_reason`` first; this opens a socket.
    """
    if not force:
        hit = cached(conf)
        if hit is not None:
            return hit, ""
    res = run_ssh(conf, STATUS_COMMAND)
    if not res.ok:
        return None, res.reason or "failed"
    reading = parse_status(res.out)
    if not reading.readable:
        return reading, ""
    reading.at = _clock()
    global _cache
    with _lock:
        _cache = reading
    return reading, ""


# --------------------------------------------------------------- actions
def key_tokens(text: str) -> tuple:
    """A spoken name reduced to the words that decide it. Filler and the
    box's own name drop out, so "restart the bot", "restart bot" and
    "restart the bot on the oracle box" are all the same key -- and a word
    that is NOT filler ("restart everything") still misses."""
    words = re.sub(r"[^a-z0-9' ]+", " ", str(text or "").lower()).split()
    return tuple(w for w in words if w not in _FILLER)


def action_names(conf: OracleConfig) -> list:
    return [str(k) for k in conf.actions if str(conf.actions.get(k) or "").strip()]


def resolve_action(conf: OracleConfig, spoken: str) -> tuple:
    """(name, command) for a spoken action, or (spoken, None) when it is not
    on the list. THE allow-list: the returned command is the config's own
    string, verbatim -- ``spoken`` only ever chooses a row."""
    want = key_tokens(spoken)
    if not want:
        return str(spoken or "").strip(), None
    for name in conf.actions:
        command = str(conf.actions.get(name) or "").strip()
        if command and key_tokens(name) == want:
            return str(name), command
    return str(spoken or "").strip(), None


def unknown_action_line(conf: OracleConfig, spoken: str) -> str:
    known = action_names(conf)
    if not known:
        return NO_ACTIONS_LINE
    if len(known) == 1:
        listed = f'only "{known[0]}"'
    else:
        listed = ", ".join(f'"{k}"' for k in known[:-1]) + f' and "{known[-1]}"'
    return UNKNOWN_ACTION_LINE.format(spoken=f'"{spoken}"' if spoken else "that",
                                      known=listed)


def changes_state(command: str) -> bool:
    """Does this command change something on the far side? Read off the
    COMMAND so a row added to the table later inherits the read-back without
    anyone having to remember to mark it."""
    return bool(STATE_CHANGE_RX.search(str(command or "")))


def run_action(conf: OracleConfig, command: str) -> tuple:
    """(ok, text). ``text`` is the far side's output on success (capped) or
    the honest failure line."""
    res = run_ssh(conf, command)
    if not res.ok:
        return False, fail_line(conf, res.reason)
    out = (res.out or "").strip()
    if not out:
        out = (res.err or "").strip()
    return True, out


def log_summary(text: str) -> tuple:
    """(spoken line, card) for a `pm2 logs` payload. The count of lines that
    MENTION an error, never a judgement about whether it is one: a grep is
    a fact, "nothing looks wrong" would be an opinion he did not ask for."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    tail = lines[-LOG_LINE_CAP:]
    # "error" as a plain substring, not \berror\b: the lines that matter on a
    # node bot are "TypeError", "ReferenceError", "UnhandledPromiseRejection"
    # -- a word boundary in front of it misses every one of them.
    errors = sum(1 for ln in tail if _ERROR_RX.search(ln))
    if not tail:
        return "The bot's log came back empty, sir.", ""
    n = len(tail)
    if errors:
        line = (f"The last {n} line{'s' if n != 1 else ''}, sir; "
                f"{errors} of them mention an error.")
    else:
        line = (f"The last {n} line{'s' if n != 1 else ''}, sir; "
                "not one of them mentions an error.")
    return line, "\n".join(tail)[:CARD_CHAR_CAP]


# ----------------------------------------------------------------- tool
def make_tools(cfg, services) -> list:
    """One tool, and ONLY when the lane is switched on and configured.

    The registry is already over its 11-tool budget and every schema is paid
    for in prefill on every model turn, so an Oracle box that is off (the
    default) costs the model nothing at all."""
    conf = read_config(cfg)
    if not conf.enabled or missing_reason(conf) is not None:
        return []

    def oracle_status(**_) -> ToolResult:
        live = read_config(cfg)                   # honour a live config edit
        reason = missing_reason(live)
        if reason:
            return ToolResult(text="oracle not configured", ok=False, speak=reason)
        reading, why = status(live)
        if reading is None:
            return ToolResult(text=f"oracle unreachable ({why})", ok=False,
                              speak=fail_line(live, why))
        return ToolResult(text=card(live, reading), max_sentences=2,
                          speak=speak_line(live, reading))

    return [ToolSpec(
        name="oracle_status",
        description=("Hunter's Oracle cloud server: uptime, load, memory, "
                     "disk and the pm2 processes."),
        parameters={"type": "object", "properties": {}},
        handler=oracle_status)]
