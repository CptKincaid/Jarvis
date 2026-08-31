"""Hunter's Oracle Cloud VM, read-mostly and OUTBOUND ONLY.

The box is ``demon-bot`` -- ``opc@163.192.101.18``, Oracle Linux Server 9.6,
verified by ssh on 2026-08-31. It runs NINE app services under **systemd**
(five Discord bots, two web services, a sync job and a dashboard) behind
nginx, with fail2ban and docker alongside them. There is no pm2 on it and
there never was: the pm2/game-news cheat sheet this module was first written
from described an OLDER server.

Jarvis asks it questions over ssh and runs a SHORT ALLOW-LIST of actions
against a KNOWN unit. He never opens anything the other way: no tunnel, no
reverse tunnel, no port-forward and no exposure of the Spark, by design --
the Spark sits behind the house NAT and it stays there.

Four rules this module exists to keep:

1. **Nothing free-form ever reaches a shell.** A spoken name is resolved to
   a row of ``oracle.services`` and the command is BUILT here from a fixed
   template plus that row's unit name, which had to match ``UNIT_RX`` to be
   loaded at all. Not one character of a transcript is ever interpolated
   into a command -- a misheard word can only fail to resolve, and an
   unresolved name is refused out loud.
2. **Nothing hangs the turn.** Every call is Popen + ``communicate(timeout)``
   + kill-WITHOUT-wait, the same shape as ``health.run_nvidia_smi``: a
   ``subprocess.run`` timeout kills the child and then *waits* for it, and a
   TCP connect stuck against a firewalled host is exactly the wedge that
   never returns. Budget is ``oracle.timeout_s`` (6 s -- the real round trip
   measures 1.0 s), and past it Jarvis says so rather than holding the floor.
3. **Unconfigured is safe and honest.** ``enabled`` is false by default.
   ``missing_reason`` names EXACTLY what is absent, and every entry point
   checks it before anything opens a socket.
4. **Restart is the only door that changes anything, and it is read back.**
   Verified on the box: a plain ``systemctl restart`` as ``opc`` is refused
   ("Authorization requires authentication", polkit, with no agent on a
   non-interactive ssh), while ``opc`` holds NOPASSWD sudo, so the restart
   goes through ``sudo -n``. The ``-n`` matters as much as the sudo: without
   it a sudo that has been locked down since would sit on a password prompt
   for the whole budget.

``run_ssh`` is the single module-level seam -- looked up at call time, so a
test replaces it and no test in this repo can reach the network.

The last good reading is cached for ``oracle.cache_s`` seconds, so "how are
the bots" followed by "and haymaker?" is one round trip, not two -- the
per-service answer is read out of the same roll-call.
"""
from __future__ import annotations

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
DEFAULT_LOG_LINES = 20
MAX_LOG_LINES = 200
CONFIG_HINT = "~/.config/jarvis/assistant.json"
DOCS_HINT = "docs/assistant-setup.md"
KB_PER_GB = 1024 * 1024                # /proc and df -Pk both report kB
LOG_LINE_CAP = 40                      # lines of journal kept for the card
CARD_CHAR_CAP = 4000                   # a card is read, not scrolled forever

# The nine services as they are actually named on demon-bot, read off
# `systemctl list-units` on 2026-08-31. This table is the source of truth;
# assistant_config.DEFAULTS ships the same rows as literal config data and
# test_the_shipped_config_matches_the_verified_table keeps the two honest.
#
# `name` is what he is called out loud AND on the card -- the systemd
# Description is longer than a spoken clause wants ("Knightfall Protocol web
# (heartbeat endpoint + admin dashboard)") and fetching it would cost a
# second round trip for a string that never changes.
DEFAULT_SERVICES = {
    "haymaker": {"unit": "haymaker-bot", "name": "Haymaker"},
    "coa": {"unit": "coa-bot", "name": "Court of Awe",
            "aliases": ["court of awe", "court of aw"]},
    "exoshock": {"unit": "exoshock-bot", "name": "Exoshock",
                 "aliases": ["exo shock"]},
    "vrider": {"unit": "vrider-bot", "name": "VRider",
               "aliases": ["v rider", "the rider"]},
    "timecard": {"unit": "timecard-bot", "name": "Timecard",
                 "aliases": ["time card"]},
    "knightfall": {"unit": "knightfall-web", "name": "Knightfall",
                   "aliases": ["knightfall protocol", "nightfall"]},
    "elevation api": {"unit": "elevation-api", "name": "the elevation API",
                      "aliases": ["elevation", "ditch grade"]},
    # No bare "monday" alias on purpose: "what about Monday?" is a question
    # about the week, and answering it with a sync service's state would be
    # the lane speaking over a scheduling conversation.
    "monday sync": {"unit": "monday-sheets-sync", "name": "Monday sync",
                    "aliases": ["sheets sync", "monday sheets"]},
    "dashboard": {"unit": "bot-dashboard", "name": "the dashboard",
                  "aliases": ["bot dashboard"]},
}

# What a systemd unit name may contain. Anything else is DROPPED at config
# read, before it can be seen by the command builder -- the config is not a
# transcript, but it is the only text that reaches a remote shell here and
# it gets checked like one.
UNIT_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:+-]{0,63}$")

STATUS_SECTIONS = ("host", "uptime", "load", "mem", "disk", "services")

# The three actions, and nothing else. `status` is not in the table because
# it is answered out of the roll-call the status probe already makes -- so
# "how's haymaker" costs no extra round trip and can be served from cache.
ACTION_LOGS = "logs"
ACTION_RESTART = "restart"
ACTIONS = (ACTION_LOGS, ACTION_RESTART)

# ssh's own diagnosis, in the registers worth telling apart.
_AUTH_RX = re.compile(
    r"permission denied|no supported authentication|too many authentication|"
    r"unprotected private key|bad permissions|invalid format|"
    r"authentication failed", re.I)
_REACH_RX = re.compile(
    r"connection refused|no route to host|network is unreachable|"
    r"could not resolve|name or service not known|connection timed out|"
    r"operation timed out|connection closed by|broken pipe|"
    r"kex_exchange_identification", re.I)
# A refused privilege on the FAR side, which is a different thing from ssh
# turning the key away and has a different fix. Checked before _AUTH_RX:
# sudo and polkit both say "authentication" in their refusals, and blaming
# the ssh key for a sudoers change would send him looking in the wrong file.
_PRIV_RX = re.compile(
    r"sudo:.*(?:password is required|a terminal is required)|"
    r"is not allowed to execute|may not run|"
    r"interactive authentication required|access denied.*polkit", re.I)
_HOSTKEY_RX = re.compile(
    r"host key verification failed|remote host identification has changed|"
    r"known_hosts", re.I)
# What counts as "mentions an error" in a log line. Deliberately literal:
# the spoken line claims a MENTION, never a diagnosis. A plain substring and
# not \berror\b -- the lines that matter say "TypeError", "ERROR -" and
# "UnhandledPromiseRejection", and a word boundary in front misses them all.
_ERROR_RX = re.compile(r"error|exception|traceback|fatal|critical|\berr\b|"
                       r"rejection", re.I)

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
    "privilege": "The Oracle box wouldn't let me do that, sir; sudo there "
                 "isn't open to me without a password.",
    "no-ssh": "I've no ssh on this machine, sir.",
    "failed": "The Oracle box wouldn't answer that, sir.",
}
NOT_ENABLED_LINE = ("The Oracle box is switched off in my settings, sir; "
                    f"set oracle.enabled to true in {CONFIG_HINT}.")
NO_HOST_LINE = (f"I've no address for the Oracle box, sir; oracle.host in "
                f"{CONFIG_HINT} is empty.")
NO_KEY_LINE = ("I've no key for the Oracle box, sir; set oracle.key_path in "
               f"{CONFIG_HINT}. The notes are in {DOCS_HINT}.")
KEY_MISSING_LINE = "I can't find the Oracle key at {path}, sir."
KEY_OPEN_LINE = ("The Oracle key at {path} is readable by others, sir; "
                 "ssh will refuse it until you chmod 600 it.")
NO_SERVICES_LINE = ("There are no services in my settings for the Oracle "
                    f"box, sir; oracle.services in {CONFIG_HINT} is empty.")
UNKNOWN_SERVICE_LINE = "I've nothing called {spoken} on the Oracle box, sir."
UNKNOWN_ACTION_LINE = ("I only do status, logs and a restart on the Oracle "
                       "box, sir.")
WHICH_SERVICE_LINE = "Which service's log, sir?"
WHICH_RESTART_LINE = "Which of them, sir?"
READ_BACK_LINE = "Restart {name} on the Oracle box, sir?"
RESTARTED_LINE = "{name} is back up, sir."
RESTART_QUIET_LINE = "{name} restarted, sir, but it isn't up yet."
UNREADABLE_LINE = ("The Oracle box answered, sir, but not with anything I "
                   "could read.")

# Filler that must not decide whether a spoken name is on the list. "bot" is
# in here on purpose: five of the nine ARE bots, so "haymaker bot",
# "haymaker" and "the haymaker" have to be one key -- and a bare "the bot",
# which reduces to nothing, resolves to no service at all rather than to
# whichever one happens to be listed first.
_FILLER = {"the", "a", "an", "my", "our", "please", "on", "oracle", "box",
           "server", "vm", "cloud", "bot", "bots", "service", "demon"}

# Small numbers are spoken as words -- "all nine services are up" is a
# sentence, "all 9 services are up" is a readout.
_NUMBERS = ("no", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve")

# systemd's ActiveState in the register a person uses. "unknown" is what a
# unit that systemd has never heard of comes back as -- a typo in the config
# rather than a service in trouble, and worth saying differently.
STATE_WORDS = {
    "active": "up",
    "inactive": "down",
    "failed": "failed",
    "activating": "still starting",
    "deactivating": "shutting down",
    "reloading": "reloading",
    "unknown": "unaccounted for",
}

# Friendly fractions for the disk, nearest match inside a tight band. "a
# third of the disk free" is the answer to the question actually being
# asked; "10405444 kilobytes" is not.
_FRACTIONS = ((0.95, "nearly all of"), (0.75, "three quarters of"),
              (0.67, "two thirds of"), (0.5, "half"), (0.33, "a third of"),
              (0.25, "a quarter of"), (0.2, "a fifth of"),
              (0.1, "a tenth of"), (0.05, "a twentieth of"))
_FRACTION_BAND = 0.035


def _count(n: int) -> str:
    return _NUMBERS[n] if 0 <= n < len(_NUMBERS) else str(n)


def _cap(text: str) -> str:
    """Sentence case without touching the rest: "the elevation API" ->
    "The elevation API", and never "The Elevation Api"."""
    text = str(text or "")
    return text[:1].upper() + text[1:]


def _join(items: list) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f" and {items[-1]}"


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
class Service:
    """One row of the allow-list: a systemd unit, a name to say, and the
    phrasings that reach it."""
    key: str
    unit: str
    name: str
    aliases: tuple = ()

    @property
    def match_keys(self) -> tuple:
        """Every token tuple that means this service. The unit name is in
        here too, so "knightfall web" finds knightfall-web without anyone
        writing that alias down."""
        phrases = [self.key, self.name, self.unit.replace("-", " ")]
        phrases += list(self.aliases)
        out, seen = [], set()
        for phrase in phrases:
            toks = key_tokens(phrase)
            if toks and toks not in seen:
                seen.add(toks)
                out.append(toks)
        return tuple(out)


def _read_services(raw) -> list:
    """The ``oracle.services`` table as Service rows, in config order.

    A row whose unit name is not a plain systemd unit is DROPPED with a
    warning rather than sanitised: this is the only config text that reaches
    a remote shell, and half-cleaning it would be worse than refusing it."""
    if not isinstance(raw, dict):
        return []
    out = []
    for key, val in raw.items():
        key = str(key or "").strip()
        if isinstance(val, str):            # "haymaker": "haymaker-bot"
            val = {"unit": val}
        if not isinstance(val, dict):
            continue
        unit = str(val.get("unit") or "").strip()
        if not unit or not UNIT_RX.match(unit):
            log.warning("oracle: dropping service %r -- %r is not a unit name",
                        key, unit)
            continue
        aliases = val.get("aliases")
        aliases = tuple(str(a) for a in aliases) if isinstance(aliases, (list, tuple)) else ()
        out.append(Service(key=key or unit, unit=unit,
                           name=str(val.get("name") or key or unit).strip(),
                           aliases=aliases))
    return out


@dataclass
class OracleConfig:
    enabled: bool = False
    host: str = ""
    user: str = ""
    key_path: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S
    cache_s: float = DEFAULT_CACHE_S
    log_lines: int = DEFAULT_LOG_LINES
    services: list = field(default_factory=list)

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    @property
    def key_file(self) -> str:
        return os.path.expanduser(self.key_path) if self.key_path else ""

    @property
    def units(self) -> list:
        return [s.unit for s in self.services]

    def by_unit(self, unit: str) -> Optional[Service]:
        for svc in self.services:
            if svc.unit == unit:
                return svc
        return None


def read_config(cfg) -> OracleConfig:
    """The ``oracle`` section, read fresh. The commander reads per call so
    that turning the box on in the config file takes effect on the next
    question rather than on the next restart."""
    def _float(key, default, low, high):
        try:
            return max(low, min(high, float(_cfg_get(cfg, key, default))))
        except (TypeError, ValueError):
            return float(default)

    return OracleConfig(
        enabled=bool(_cfg_get(cfg, "oracle.enabled", False)),
        host=str(_cfg_get(cfg, "oracle.host", "") or "").strip(),
        user=str(_cfg_get(cfg, "oracle.user", "") or "").strip(),
        key_path=str(_cfg_get(cfg, "oracle.key_path", "") or "").strip(),
        timeout_s=_float("oracle.timeout_s", DEFAULT_TIMEOUT_S,
                         MIN_TIMEOUT_S, MAX_TIMEOUT_S),
        cache_s=_float("oracle.cache_s", DEFAULT_CACHE_S, 0.0, 600.0),
        log_lines=int(_float("oracle.log_lines", DEFAULT_LOG_LINES,
                             1, MAX_LOG_LINES)),
        services=_read_services(_cfg_get(cfg, "oracle.services", None)))


def missing_reason(conf: OracleConfig) -> Optional[str]:
    """The ONE honest line naming what is missing, or None when the lane is
    ready. Checked by every entry point BEFORE anything opens a socket, so
    an unconfigured install never touches the network at all.

    The key-permission check is here rather than left to ssh because the
    key ships in ~/Downloads, where a copy off another machine lands 0644 --
    and ssh's own refusal ("UNPROTECTED PRIVATE KEY FILE") would reach him
    as the generic auth line instead of the fix."""
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
    if not conf.services:
        return NO_SERVICES_LINE
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
    "Permission denied" further down and is the one he must look at himself,
    and sudo/polkit refusals talk about authentication without the ssh key
    having anything to do with it."""
    text = err or ""
    if _HOSTKEY_RX.search(text):
        return "hostkey"
    if _PRIV_RX.search(text):
        return "privilege"
    if _AUTH_RX.search(text):
        return "auth"
    if _REACH_RX.search(text):
        return "unreachable"
    return "failed"


def ssh_argv(conf: OracleConfig, command: str) -> list:
    """The exact argv. No shell on THIS side ever (no shell=True, no string
    command), and on the far side the command is one this module built.

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
    argv += ["--", conf.target, command]
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


# -------------------------------------------------------------- commands
def status_command(conf: OracleConfig) -> str:
    """ONE round trip for the whole roll-call: 1.0 s against the real box.

    Everything but the services comes out of /proc and df, which cannot hang
    and cannot be missing. The service loop prints the unit name BESIDE its
    state rather than relying on ``systemctl is-active a b c`` answering in
    argument order -- nine extra forks on an idle VM cost nothing, and a
    reading paired to the wrong service is the one wrong answer this lane
    must never give. Every unit in it passed UNIT_RX at config read.
    POSIX sh only, and quoted: /bin/sh on Oracle Linux is bash, but the
    command has to be correct without depending on that.
    """
    head = ("echo '#host'; cat /proc/sys/kernel/hostname; "
            "echo '#uptime'; cat /proc/uptime; "
            "echo '#load'; cat /proc/loadavg; "
            "echo '#mem'; cat /proc/meminfo; "
            "echo '#disk'; df -Pk /; "
            "echo '#services';")
    units = " ".join(conf.units)
    if not units:
        return head
    return (f"{head} for u in {units}; do "
            "printf '%s %s\\n' \"$u\" \"$(systemctl is-active \"$u\" "
            "2>/dev/null)\"; done")


def logs_command(conf: OracleConfig, unit: str) -> str:
    return f"journalctl -u {unit} -n {int(conf.log_lines)} --no-pager"


def restart_command(unit: str) -> str:
    """A restart, then the state it landed in.

    ``sudo -n`` and not a bare systemctl: verified on the box, ``opc`` gets
    "Authorization requires authentication" from polkit for a plain
    ``systemctl restart`` (there is no agent on a non-interactive ssh) and
    holds NOPASSWD sudo, so this is the form that actually works. The ``-n``
    is load-bearing on its own: a sudoers change would otherwise turn this
    into a password prompt that eats the whole timeout budget.

    The ``&&`` tail means the spoken line can say whether it came back up
    instead of "done" -- and a failed restart leaves rc!=0 for
    classify_error to read.
    """
    return f"sudo -n systemctl restart {unit} && systemctl is-active {unit}"


def action_command(conf: OracleConfig, action: str, unit: str) -> Optional[str]:
    """THE ALLOW-LIST. ``action`` must be one of ACTIONS and ``unit`` must be
    a unit this config actually loaded; anything else returns None and is
    refused out loud. This is the only place a remote command is built."""
    if action not in ACTIONS or conf.by_unit(unit) is None:
        return None
    if action == ACTION_LOGS:
        return logs_command(conf, unit)
    return restart_command(unit)


# --------------------------------------------------------------- parsing
@dataclass
class ServiceState:
    key: str = ""
    name: str = ""
    unit: str = ""
    state: str = "unknown"

    @property
    def up(self) -> bool:
        return self.state == "active"

    @property
    def word(self) -> str:
        return STATE_WORDS.get(self.state, self.state or "unknown")


@dataclass
class Reading:
    host: str = ""
    uptime_s: Optional[float] = None
    load: Optional[tuple] = None
    mem_total_kb: Optional[int] = None
    mem_avail_kb: Optional[int] = None
    disk_total_kb: Optional[int] = None
    disk_free_kb: Optional[int] = None
    services: list = field(default_factory=list)
    at: float = 0.0                   # monotonic stamp for the cache

    @property
    def readable(self) -> bool:
        """A reading worth speaking. The service list is built from the
        CONFIG, so it is nine rows long even when the box said nothing at
        all -- without the state check, an ssh that returned garbage would
        be announced as nine services down, which is the most alarming
        possible way to say "I couldn't read the answer"."""
        return (self.uptime_s is not None
                or any(s.state != "unknown" for s in self.services))

    @property
    def down(self) -> list:
        return [s for s in self.services if not s.up]

    def state_of(self, unit: str) -> Optional[ServiceState]:
        for svc in self.services:
            if svc.unit == unit:
                return svc
        return None


def _split_sections(text: str) -> dict:
    """The ``#name``-delimited blocks of status_command's output."""
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


def parse_services(conf: OracleConfig, text: str) -> list:
    """The roll-call, in CONFIG order rather than reply order, so the card
    never reshuffles between readings. A unit the box said nothing about is
    "unknown" -- present and honest, not quietly dropped."""
    seen = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        seen[parts[0]] = parts[1] if len(parts) > 1 and parts[1] else "unknown"
    return [ServiceState(key=svc.key, name=svc.name, unit=svc.unit,
                         state=seen.get(svc.unit, "unknown"))
            for svc in conf.services]


def parse_status(conf: OracleConfig, text: str) -> Reading:
    """One status_command payload -> a Reading. Never raises: a section that
    does not parse simply leaves its fields None, exactly like health.py's
    probes, because a partial answer still tells him whether the bots are
    up."""
    sections = _split_sections(text)
    r = Reading(host=(sections.get("host", "") or "").strip().splitlines()[0]
                if (sections.get("host") or "").strip() else "")
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
        r.mem_avail_kb = _available_kb(mem)
    except Exception:                        # noqa: BLE001 - probe boundary
        log.debug("oracle: meminfo unparsable", exc_info=True)
    for line in sections.get("disk", "").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1].isdigit() and parts[3].isdigit():
            r.disk_total_kb, r.disk_free_kb = int(parts[1]), int(parts[3])
            break
    r.services = parse_services(conf, sections.get("services", ""))
    return r


def _available_kb(mem: dict) -> Optional[int]:
    """MemAvailable, but only when it can possibly be true.

    demon-bot's kernel (OCI aarch64, Oracle Linux 9.6) reports MemAvailable
    20512504 kB against a MemTotal of 5779324 kB -- nineteen gigabytes free
    on a five-and-a-half gigabyte box. procps' own `free` guards against
    exactly this and falls back to MemFree, which is why `free -h` says
    1.7 Gi available while /proc/meminfo says 19.6. Without this clamp the
    spoken line is a confident, absurd lie."""
    total = mem.get("MemTotal")
    avail = mem.get("MemAvailable")
    free = mem.get("MemFree")
    if avail is None or (total and avail > total):
        return free
    return avail


# --------------------------------------------------------------- speech
def uptime_words(seconds: Optional[float]) -> str:
    """"ten weeks and a day", "3 days and 4 hours", "an hour and 10 minutes".

    runwatch.elapsed_words tops out in hours -- correct for a training run,
    useless for a server that has been up since June ("1721 hours"). Past a
    fortnight it goes to weeks, because that is the unit he answers in."""
    if seconds is None:
        return "an unknown time"
    seconds = max(0.0, float(seconds))
    days = int(seconds // 86400)
    if days < 1:
        return elapsed_words(seconds)
    if days < 14:
        hours = int((seconds - days * 86400) // 3600)
        head = "a day" if days == 1 else f"{_count(days)} days"
        if not hours:
            return head
        return f"{head} and {_count(hours)} hour{'s' if hours != 1 else ''}"
    weeks, rest = divmod(days, 7)
    head = f"{_count(weeks)} weeks"
    if not rest:
        return head
    return f"{head} and {'a day' if rest == 1 else f'{_count(rest)} days'}"


def disk_words(free_kb: Optional[int], total_kb: Optional[int]) -> str:
    """"a third of the disk free", else a percentage. The question behind
    "how's the box" is whether the disk is about to bite, and a fraction
    answers it in the register it was asked in."""
    if not total_kb or free_kb is None:
        return ""
    frac = float(free_kb) / float(total_kb)
    for value, word in _FRACTIONS:
        if abs(frac - value) <= _FRACTION_BAND:
            return f"{word} the disk free"
    return f"{round(frac * 100)} percent of the disk free"


def _box_clause(r: Reading) -> str:
    bits = []
    if r.uptime_s is not None:
        bits.append(f"{uptime_words(r.uptime_s)} of uptime")
    disk = disk_words(r.disk_free_kb, r.disk_total_kb)
    if disk:
        bits.append(disk)
    return " and ".join(bits)


def _host_words(conf: OracleConfig, r: Reading) -> str:
    return r.host or conf.host or "the Oracle box"


def speak_line(conf: OracleConfig, r: Reading) -> str:
    """ONE sentence for the whole roll-call. Anything not up LEADS it -- the
    question behind every phrasing of this command is whether something has
    fallen over, and burying that behind the uptime is how a status readout
    gets ignored."""
    if r is None or not r.readable:
        return UNREADABLE_LINE
    host = _host_words(conf, r)
    tail = _box_clause(r)
    total, down = len(r.services), r.down
    if not total:
        # missing_reason refuses an empty table before anything gets here;
        # this is the belt to that brace.
        if tail:
            return f"I've no services listed for {host}, sir; {tail}."
        return f"{_cap(host)} answered, sir, but I've no services to ask about."
    if not down:
        head = f"All {_count(total)} services are up on {host}, sir"
        return f"{head}; {tail}." if tail else f"{head}."
    others = total - len(down)
    if len(down) == 1:
        head = f"{_cap(down[0].name)} is {down[0].word}, sir"
    elif len(down) <= 3:
        words = {s.word for s in down}
        word = words.pop() if len(words) == 1 else "not running"
        head = f"{_cap(_join([s.name for s in down]))} are {word}, sir"
    else:
        words = {s.word for s in down}
        word = words.pop() if len(words) == 1 else "down"
        head = (f"{_cap(_count(len(down)))} of the {_count(total)} services "
                f"are {word} on {host}, sir")
    if others:
        head += (f"; the other {_count(others)} "
                 f"{'is' if others == 1 else 'are'} up")
    else:
        head += "; not one of them is up"
    return head + "."


def service_line(conf: OracleConfig, r: Reading, svc: Service) -> str:
    """ONE sentence about ONE named service, read out of the same roll-call
    the box question uses -- so "how's haymaker" straight after "how are the
    bots" costs nothing at all."""
    if r is None or not r.readable:
        return UNREADABLE_LINE
    state = r.state_of(svc.unit)
    if state is None:
        return f"I've no reading for {svc.name} from the Oracle box, sir."
    host = _host_words(conf, r)
    others = [s for s in r.services if s.unit != svc.unit and not s.up]
    if state.up:
        line = f"{_cap(svc.name)} is up on {host}, sir"
        if others:
            line += (f", though {_join([s.name for s in others])} "
                     f"{'is' if len(others) == 1 else 'are'} not")
        elif len(r.services) > 1:
            line += f", along with the other {_count(len(r.services) - 1)}"
        return line + "."
    line = f"{_cap(svc.name)} is {state.word} on {host}, sir"
    up = len(r.services) - len(r.down)
    if up:
        line += f"; {_count(up)} of the {_count(len(r.services))} are up"
    return line + "."


def card(conf: OracleConfig, r: Reading) -> str:
    """The plain sheet behind the spoken line -- the numbers stay readable
    and the two can never disagree, because both come off this Reading."""
    host = _host_words(conf, r)
    lines = [f"{host}  ({conf.target})"]
    head = f"up {uptime_words(r.uptime_s)}"
    if r.load:
        head += "  ·  load " + " ".join(f"{v:.2f}" for v in r.load)
    lines.append(head)
    total, free = _gb(r.mem_total_kb), _gb(r.mem_avail_kb)
    if total is not None and free is not None:
        lines.append(f"memory  {round(total - free, 1)} GB used of {total} GB")
    dtotal, dfree = _gb(r.disk_total_kb), _gb(r.disk_free_kb)
    if dtotal is not None and dfree is not None:
        lines.append(f"disk    {dfree} GB free of {dtotal} GB")
    lines.append("")
    if not r.services:
        lines.append("no services configured")
    for svc in r.services:
        mark = "  " if svc.up else "! "
        lines.append(f"{mark}{svc.unit:<20} {svc.state:<10} {svc.name}")
    return "\n".join(lines)[:CARD_CHAR_CAP]


def _gb(kb: Optional[int]) -> Optional[float]:
    return None if kb is None else round(kb / KB_PER_GB, 1)


def services_card(conf: OracleConfig) -> str:
    """The list of what he can ask about, for the "which service?" turn."""
    lines = ["Oracle services"]
    for svc in conf.services:
        lines.append(f"  {svc.name:<18} {svc.unit}")
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

    The point of the cache is the second question: "how are the bots" /
    "…and haymaker?" is one round trip, and the second answer is instant
    instead of another ``done=False`` turn. A reading taken against a
    DIFFERENT service list is thrown away -- he edited the config between
    the two questions, and answering the new roll-call off the old one would
    be a wrong answer with a straight face."""
    with _lock:
        r = _cache
    if r is None or conf.cache_s <= 0:
        return None
    if [s.unit for s in r.services] != conf.units:
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
    res = run_ssh(conf, status_command(conf))
    if not res.ok:
        return None, res.reason or "failed"
    reading = parse_status(conf, res.out)
    if not reading.readable:
        return reading, ""
    reading.at = _clock()
    global _cache
    with _lock:
        _cache = reading
    return reading, ""


# --------------------------------------------------------------- naming
def key_tokens(text: str) -> tuple:
    """A spoken name reduced to the words that decide it. Filler and the
    box's own name drop out, so "haymaker", "the haymaker" and "haymaker
    bot" are one key -- and a word that is NOT filler ("haymaker digest")
    still misses."""
    words = re.sub(r"[^a-z0-9' ]+", " ", str(text or "").lower()).split()
    return tuple(w for w in words if w not in _FILLER)


def spoken_names(conf: OracleConfig) -> list:
    return [svc.name for svc in conf.services]


def resolve_service(conf: OracleConfig, spoken: str) -> Optional[Service]:
    """The Service a spoken name means, or None.

    Exact token match first, then subset ("haymaker" inside "haymaker bot"),
    and an AMBIGUOUS name resolves to nothing at all rather than to whichever
    row was listed first -- restarting the wrong bot because two matched is
    exactly the failure this lane is built to avoid."""
    want = key_tokens(spoken)
    if not want:
        return None
    exact = [s for s in conf.services if want in s.match_keys]
    if len(exact) == 1:
        return exact[0]
    if exact:
        log.info("oracle: %r matches %d services; refusing", spoken, len(exact))
        return None
    wanted = set(want)
    near = [s for s in conf.services
            if any(wanted <= set(k) for k in s.match_keys)]
    if len(near) == 1:
        return near[0]
    if near:
        log.info("oracle: %r is ambiguous across %d services", spoken, len(near))
    return None


def unknown_service_line(conf: OracleConfig, spoken: str) -> str:
    if not conf.services:
        return NO_SERVICES_LINE
    return UNKNOWN_SERVICE_LINE.format(
        spoken=f'"{spoken}"' if str(spoken or "").strip() else "that")


def service_word_pattern() -> str:
    """A regex alternation over the names of the SHIPPED services, longest
    first, for the commander's bare-name matchers ("is knightfall up").

    Built from DEFAULT_SERVICES rather than from his live config because a
    Command matcher is a module-level function with no config in reach. The
    cost is honest and small: a service he ADDS to the config is reachable
    by naming the box ("on the oracle box, restart foo") but not by its bare
    name. The gain is that "how's the weather" cannot match a service
    matcher at all -- and a matcher that accepts anything would bypass the
    intent gate for every "how's the X" sentence in the house."""
    phrases = set()
    for key, row in DEFAULT_SERVICES.items():
        phrases.add(key)
        phrases.add(row.get("name", ""))
        phrases.add(str(row.get("unit", "")).replace("-", " "))
        for alias in row.get("aliases", ()):
            phrases.add(alias)
    out = []
    for phrase in sorted(phrases, key=lambda p: (-len(p), p)):
        words = [w for w in re.split(r"[\s-]+", phrase.strip().lower()) if w]
        if not words:
            continue
        out.append(r"\s+".join(re.escape(w) for w in words))
    return "(?:" + "|".join(out) + r")(?:\s+(?:bot|service|server))?"


# --------------------------------------------------------------- actions
def run_action(conf: OracleConfig, command: str) -> tuple:
    """(ok, text). ``text`` is the far side's output on success (capped) or
    the honest failure line. ``command`` only ever comes from
    ``action_command``."""
    res = run_ssh(conf, command)
    if not res.ok:
        return False, fail_line(conf, res.reason)
    out = (res.out or "").strip()
    if not out:
        out = (res.err or "").strip()
    return True, out


def restart_line(svc: Service, output: str) -> str:
    """What to say after a restart: the state it landed in, not "done".
    ``systemctl is-active`` tailed the restart, so this is a fact."""
    state = (output or "").strip().splitlines()[-1].strip() \
        if (output or "").strip() else ""
    if state == "active":
        return RESTARTED_LINE.format(name=_cap(svc.name))
    if state in STATE_WORDS:
        return (f"{_cap(svc.name)} restarted, sir, and it's "
                f"{STATE_WORDS[state]}.")
    return RESTART_QUIET_LINE.format(name=_cap(svc.name))


def log_summary(svc: Optional[Service], text: str) -> tuple:
    """(spoken line, card) for a journalctl payload. The count of lines that
    MENTION an error, never a judgement about whether it is one: a grep is
    a fact, "nothing looks wrong" would be an opinion he did not ask for."""
    name = svc.name if svc is not None else "that service"
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    tail = lines[-LOG_LINE_CAP:]
    if not tail:
        return f"{_cap(name)}'s log came back empty, sir.", ""
    n = len(tail)
    errors = sum(1 for ln in tail if _ERROR_RX.search(ln))
    head = f"The last {n} line{'s' if n != 1 else ''} of {name}, sir"
    if errors:
        line = f"{head}; {errors} of them mention an error."
    else:
        line = f"{head}; not one of them mentions an error."
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
        description=("Hunter's Oracle server demon-bot: uptime, load, "
                     "memory, disk and its nine systemd services."),
        parameters={"type": "object", "properties": {}},
        handler=oracle_status)]
