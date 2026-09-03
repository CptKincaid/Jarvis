"""HPCOMPUTER -- files both ways and a SHORT allow-list of read-only questions.

GROUND TRUTH, measured on the Spark 2026-09-02, before a line of this was
written.  It is written down because every one of these facts changed the
design, and a future reader who assumes otherwise will build the wrong thing:

* **HPCOMPUTER is not on the tailnet.**  ``tailscale status`` has exactly
  two nodes -- ``spark`` (this box, 100.70.145.63) and ``iphone172`` (iOS,
  offline two days).  There is no HPCOMPUTER peer.
* **HPCOMPUTER is not reachable at all right now.**  ``hpcomputer.local``
  still resolves on the LAN, to 192.168.50.114, but that is a stale mDNS
  record: the host answers no ping and has nothing listening on 22, 3389,
  445 or 5900.  It is asleep or off.
* **There is no ssh key for it.**  ``~/.ssh`` holds ``known_hosts`` and the
  Oracle key, nothing else.  Key auth to HPCOMPUTER cannot work yet.
* **tailscaled here runs in USERSPACE mode** (``--tun=userspace-networking``,
  ``"TUN": false``), as a rootless user unit.  This is the fact that would
  otherwise cost an afternoon: there is NO tun device, so once HPCOMPUTER
  does join the tailnet, ``ssh user@hpcomputer.tail5323b8.ts.net`` will
  still not route.  Tailscale traffic has to go through the SOCKS5 proxy
  the daemon already runs on ``localhost:1055``, which is why
  :func:`ssh_argv` grows a ``ProxyCommand``.  ``nc -X 5 -x`` is present and
  supports SOCKS5; ssh, scp, sftp and rsync are all installed.

So this module ships ``enabled: false`` and every entry point refuses out
loud with a reason that names what is missing.  Nothing here has been run
against the real host -- by instruction, Hunter runs the first live command
himself.  What IS tested is every decision this module makes before the
socket opens, which is where the irreversible mistakes live.

------------------------------------------------------------------- safety

Copying a file onto another machine and running a command there cannot be
undone.  Three tiers, and the boundaries are the design:

1. **Read-only questions run unattended.**  A fixed table (:data:`QUERIES`)
   of named, parameterless commands -- is it up, is the disk full, what is
   the uptime.  A spoken phrase selects a ROW; the command is built here
   from that row's template.  Not one character of a transcript is ever
   interpolated into a command line.  This is the Oracle module's rule and
   it is the reason a misheard word can only fail to resolve.

2. **File transfers are read back and confirmed, every time.**  Not only on
   a shaky transcript, the way an alarm is: an alarm set wrong is an
   annoyance and this is a file leaving the machine.  The read-back names
   the FILE, the DIRECTION and the HOST before anything opens.

3. **Arbitrary commands are refused outright.**  There is no code path from
   speech to ``ssh <anything he said>``.  ``_h_remote_freeform`` in
   commander.py is a door that only says no.  A voice channel with a
   false-accept rate cannot be given a shell; the useful 5% of that feature
   is already covered by tier 1, and the other 95% is unbounded.

Two more rules that are less obvious and matter as much:

* **Speech never names a remote path.**  A push lands in ONE configured
  directory, ``remote.inbox``, and nowhere else.  "Put it in my system
  folder" cannot be said into existence.  This bounds the blast radius of a
  push to a single folder he chose while looking at a screen.
* **A pulled filename comes from the REMOTE's own listing, never from the
  transcript.**  Pull lists an allow-listed remote folder, fuzzy-matches
  what he said against the real entries, reads the winner back, and copies
  THAT.  The string handed to scp was produced by the far side, so a
  misheard word cannot become a path there.  Names outside
  :data:`SAFE_REMOTE_NAME_RX` are dropped from the listing rather than
  quoted around -- a file with a newline or a backtick in its name is not
  worth the class of bug it invites.

``run_ssh`` and ``run_copy`` are the two module-level seams, looked up at
call time, so tests replace them and no test in this repo can reach a
network, a host key or a disk it did not make.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from jarvis.logs import get_logger
from jarvis.tools import filepick

log = get_logger("tools.remote")

SSH_BIN = "ssh"
SCP_BIN = "scp"
NC_BIN = "nc"
TAILSCALE_BIN = os.path.expanduser("~/.local/bin/tailscale")
# The rootless daemon's socket.  The CLI defaults to /var/run/tailscale,
# which does not exist here -- `tailscale status` with no --socket fails
# with "failed to connect to local tailscaled", which is NOT the same thing
# as the host being down and must never be reported as if it were.
TAILSCALE_SOCKET = os.path.expanduser(
    "~/.local/share/tailscale/tailscaled.sock")

DEFAULT_TIMEOUT_S = 12.0          # a question; the box may be waking up
DEFAULT_TRANSFER_S = 120.0        # a file; bounded, but not 12 seconds
MIN_TIMEOUT_S = 3.0
MAX_TIMEOUT_S = 60.0
MAX_TRANSFER_S = 900.0
DEFAULT_MAX_MB = 100.0
LISTING_CAP = 400                 # entries read out of a remote folder
CARD_CHAR_CAP = 4000
CONFIG_HINT = "~/.config/jarvis/assistant.json"

# What a remote filename may contain to be eligible at all.  Deliberately
# narrow: this is an allow-list over names the far side reported, and a name
# that fails it is DROPPED from the listing, not escaped.  Shell
# metacharacters, newlines, leading dashes (which scp would read as flags)
# and anything path-like are all excluded by construction.
SAFE_REMOTE_NAME_RX = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9 ._+()-]{0,120}$")

# A remote directory key he may name out loud -> the config field holding
# the real path.  He says "desktop", not a path; see the pull rules above.
PULL_KEYS = ("outbox", "desktop", "downloads")


# ------------------------------------------------------- remote paths
# Every shipped remote directory starts with "~".  A tilde is expanded by
# the remote SHELL and by nothing else, so the two far sides need it
# written two DIFFERENT ways, and quoting it the obvious way breaks both.
#
# Measured on this box, 2026-09-03:
#     shlex.quote("~/x")            -> "'~/x'"
#     sh -c "ls -1p -- '~/x'"       -> No such file or directory (rc 2)
#     sh -c 'ls -1pd -- "$HOME"/x'  -> /home/hunterp/x   (rc 0)
# The failing form's stderr matches _MISSING_RX, so classify_error called it
# "not-there" and FAIL_LINES spoke a config bug as "I couldn't find that on
# HPCOMPUTER, sir." -- the whole pull half of the lane, and the `inbox`
# question, wrong AND misdiagnosed out loud.
def shell_path(path: str) -> str:
    """A remote path for a remote SHELL: "$HOME" unquoted, the rest quoted.

    Only the tilde escapes quoting, and only as the leading segment; every
    character that came from the config file is still inside single quotes,
    so ``~/my inbox; rm -rf ~`` is one directory name and not two commands.
    A ``~user`` form is left quoted deliberately -- it would fail loudly
    rather than resolve to somebody else's home.
    """
    raw = (path or "").strip()
    if not raw:
        return ""
    if raw == "~":
        return '"$HOME"'
    if raw.startswith("~/"):
        rest = raw[2:].strip("/")
        return '"$HOME"' if not rest else '"$HOME"/' + shlex.quote(rest)
    return shlex.quote(raw)


def scp_path(path: str) -> str:
    """The same path as SCP must receive it: relative to the login home.

    scp is not a shell on the far side.  OpenSSH 9.6 (this box, ``ssh -V``)
    runs scp over SFTP, where a relative path resolves against the login
    home and a tilde is a literal character; the legacy ``-O`` protocol runs
    a remote shell whose cwd is that same home.  A path relative to the home
    is therefore right in both, and ``~/x`` is right in neither.
    """
    raw = (path or "").strip()
    if raw == "~":
        return "."
    if raw.startswith("~/"):
        return raw[2:].lstrip("/") or "."
    return raw


# --------------------------------------------------------------- config
@dataclass
class RemoteConfig:
    enabled: bool = False
    host: str = ""
    user: str = ""
    key_path: str = ""
    name: str = "HPCOMPUTER"
    timeout_s: float = DEFAULT_TIMEOUT_S
    transfer_timeout_s: float = DEFAULT_TRANSFER_S
    socks_proxy: str = "127.0.0.1:1055"
    inbox: str = "~/jarvis-inbox"
    pull_dirs: dict = field(default_factory=dict)
    max_mb: float = DEFAULT_MAX_MB
    local_roots: tuple = filepick.DEFAULT_ROOTS

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


def _clamp(value, low, high, fallback):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, v))


def _cfg_get(cfg, dotted: str, default=None):
    """Dotted lookup that survives a missing or hostile config object.  Same
    convention as ``oracle._cfg_get``: the commander reads per call, so
    switching the lane on in the file takes effect on the next sentence
    rather than on the next restart."""
    get = getattr(cfg, "get", None)
    if not callable(get):
        return default
    try:
        val = get(dotted, default)
    except Exception:                          # noqa: BLE001 - config only
        log.debug("remote: cannot read %s", dotted, exc_info=True)
        return default
    return default if val is None else val


def read_config(cfg) -> RemoteConfig:
    """The ``remote`` section, defensively.  A malformed config must leave the
    lane OFF, never half-configured: a present host with a missing user is
    exactly the state that produces a baffling auth failure ten seconds
    later, so ``missing_reason`` names it before a socket opens."""
    pull = _cfg_get(cfg, "remote.pull_dirs", {})
    if not isinstance(pull, dict):
        pull = {}
    roots = _cfg_get(cfg, "remote.local_roots", None)
    return RemoteConfig(
        enabled=bool(_cfg_get(cfg, "remote.enabled", False)),
        host=str(_cfg_get(cfg, "remote.host", "") or "").strip(),
        user=str(_cfg_get(cfg, "remote.user", "") or "").strip(),
        key_path=str(_cfg_get(cfg, "remote.key_path", "") or "").strip(),
        name=str(_cfg_get(cfg, "remote.name", "") or "").strip() or "HPCOMPUTER",
        timeout_s=_clamp(_cfg_get(cfg, "remote.timeout_s", None),
                         MIN_TIMEOUT_S, MAX_TIMEOUT_S, DEFAULT_TIMEOUT_S),
        transfer_timeout_s=_clamp(
            _cfg_get(cfg, "remote.transfer_timeout_s", None),
            MIN_TIMEOUT_S, MAX_TRANSFER_S, DEFAULT_TRANSFER_S),
        socks_proxy=str(_cfg_get(cfg, "remote.socks_proxy", "") or "").strip(),
        inbox=str(_cfg_get(cfg, "remote.inbox", "")
                  or "~/jarvis-inbox").strip(),
        pull_dirs={str(k): str(v) for k, v in pull.items() if v},
        max_mb=_clamp(_cfg_get(cfg, "remote.max_mb", None), 1, 4096,
                      DEFAULT_MAX_MB),
        local_roots=tuple(roots) if roots else filepick.DEFAULT_ROOTS,
    )


def key_file(conf: RemoteConfig) -> str:
    """The expanded key path, or "" when it is unset or not actually there.

    Jarvis never reads the key -- only hands ssh its path.  That is the
    whole of the credential story for this lane, and it is why nothing new
    goes in SECRET_KEYS: there is no new secret, only a reference to one
    ssh already owns.
    """
    if not conf.key_path:
        return ""
    p = os.path.expanduser(conf.key_path)
    return p if os.path.isfile(p) else ""


def missing_reason(conf: RemoteConfig) -> str:
    """EXACTLY what is absent, so the spoken refusal can be specific.  ""
    when the lane is ready to try."""
    if not conf.enabled:
        return "disabled"
    if not conf.host:
        return "no-host"
    if not conf.user:
        return "no-user"
    if conf.key_path and not key_file(conf):
        return "bad-key"
    return ""


# ---------------------------------------------------------------- lines
FAIL_LINES = {
    "disabled": "{name} isn't set up in my settings yet, sir; "
                f"remote.enabled is false in {CONFIG_HINT}.",
    "no-host": "I've no address for {name}, sir -- it isn't on the tailnet "
               "yet, and remote.host is empty.",
    "no-user": "I don't know which account to use on {name}, sir; "
               "remote.user is empty.",
    "bad-key": "The key I'm meant to use for {name} isn't where my settings "
               "say it is, sir.",
    "no-ssh": "I've no ssh on this machine, sir.",
    "no-proxy": "I can't reach the tailnet proxy, sir; the Tailscale daemon "
                "runs without a tunnel here and its SOCKS port isn't answering.",
    "off-tailnet": "{name} isn't on the tailnet, sir -- I can't see it at all.",
    "asleep": "{name} is on the tailnet but not answering, sir; "
              "I'd say it's asleep.",
    "timeout": "{name} didn't answer in {timeout} seconds, sir; "
               "I've stopped waiting on it.",
    "auth": "{name} turned my key away, sir.",
    "unreachable": "I couldn't reach {name}, sir.",
    "hostkey": "{name}'s host key isn't the one I know, sir; I'd look at "
               "that before I talk to it again.",
    "no-space": "{name} hadn't the room for that file, sir.",
    "denied": "{name} wouldn't let me write that, sir.",
    "not-there": "I couldn't find that on {name}, sir.",
    # A LOCAL file whose name has characters this module will not put into a
    # remote path.  Refused, never escaped -- see inbox_target.
    "odd-name": "That file's name has characters I won't put on {name}, "
                "sir; rename it and I'll send it.",
    "exists": "There's already a file by that name where I'd put it, sir; "
              "I've left yours alone.",
    "failed": "{name} wouldn't answer that, sir.",
}

# The one door that always says no.  Kept here beside the failures so the
# refusal is as maintained as the successes are.
FREEFORM_REFUSAL = (
    "I don't run loose commands on {name}, sir. I'll fetch files, send "
    "them, and answer a short list of questions about it -- but not that.")


def fail_line(conf: RemoteConfig, reason: str) -> str:
    line = FAIL_LINES.get(reason) or FAIL_LINES["failed"]
    return line.format(name=conf.name, timeout=int(conf.timeout_s))


_HOSTKEY_RX = re.compile(
    r"host key.*(changed|verification failed)|REMOTE HOST IDENTIFICATION",
    re.I)
_AUTH_RX = re.compile(
    r"permission denied|too many authentication|no supported authentication|"
    r"publickey.*denied|authentication failed", re.I)
_REACH_RX = re.compile(
    r"could not resolve|name or service not known|no route to host|"
    r"connection refused|network is unreachable|connection closed by|"
    r"operation timed out|connection timed out|proxy", re.I)
_SPACE_RX = re.compile(r"no space left|disk quota exceeded", re.I)
_DENIED_RX = re.compile(r"permission denied.*(writ|creat)|read-only file system",
                        re.I)
_MISSING_RX = re.compile(r"no such file or directory|not a directory", re.I)


def classify_error(err: str) -> str:
    """ssh/scp stderr in one word.  Order matters, and each rung is a real
    confusion this avoids: a changed host key ALSO prints "Permission
    denied" further down and is the one thing he must look at himself; a
    full remote disk and a refused write both look like a generic failure
    but need different sentences; and "no such file" during a pull is a
    misremembered name, not a broken link."""
    text = err or ""
    if _HOSTKEY_RX.search(text):
        return "hostkey"
    if _SPACE_RX.search(text):
        return "no-space"
    if _DENIED_RX.search(text):
        return "denied"
    if _AUTH_RX.search(text):
        return "auth"
    if _MISSING_RX.search(text):
        return "not-there"
    if _REACH_RX.search(text):
        return "unreachable"
    return "failed"


@dataclass
class SshResult:
    ok: bool
    out: str = ""
    err: str = ""
    reason: str = ""


# ------------------------------------------------------------- transport
def proxy_args(conf: RemoteConfig) -> list:
    """The ``ProxyCommand`` that makes the tailnet reachable from a
    USERSPACE tailscaled.

    Without a tun device there is no route to 100.64/10 at all, so a plain
    ssh to a tailnet name fails with "Network is unreachable" no matter how
    healthy the tailnet is.  The daemon's SOCKS5 port is the supported way
    through, and ``nc -X 5 -x`` speaks it.  Empty when ``socks_proxy`` is
    blank, which is the right configuration the day he gives this box a real
    tun device or reaches the host over plain LAN.
    """
    if not conf.socks_proxy:
        return []
    return ["-o", f"ProxyCommand={NC_BIN} -X 5 -x "
                  f"{shlex.quote(conf.socks_proxy)} %h %p"]


def _common_opts(conf: RemoteConfig, timeout_s: float) -> list:
    """Options shared by ssh and scp.

    ``BatchMode`` plus the three ``*Authentication=no`` settings are what
    make a timeout mean anything: without them a host that falls back to
    password auth sits on a prompt for the entire budget and then dies
    without having asked a human being anything.

    ``StrictHostKeyChecking=accept-new`` takes the first key silently and
    HARD-FAILS a changed one.  That is the intended asymmetry -- first
    contact should not need a terminal, and a key that changes afterwards is
    the one event worth refusing to talk over.
    """
    connect = max(1, int(min(timeout_s, conf.timeout_s)) - 1)
    argv = ["-o", "BatchMode=yes",
            "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no",
            "-o", "NumberOfPasswordPrompts=0",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={connect}",
            "-o", "IdentitiesOnly=yes"]
    kf = key_file(conf)
    if kf:
        argv += ["-i", kf]
    argv += proxy_args(conf)
    return argv


def ssh_argv(conf: RemoteConfig, command: str, timeout_s: float = 0.0) -> list:
    """The exact argv.  No shell on this side (never ``shell=True``), and on
    the far side the command is one this module built from a template."""
    return ([SSH_BIN, "-n"] + _common_opts(conf, timeout_s or conf.timeout_s)
            + ["--", conf.target, command])


def scp_argv(conf: RemoteConfig, local: str, remote: str,
             push: bool, timeout_s: float = 0.0) -> list:
    """scp argv for one file, in one direction.

    ``--`` before the paths, and ``./`` glued to a bare local name, because
    scp reads a leading dash as a flag and a local file called ``-rf`` would
    otherwise be an argument.

    The remote half is NOT quoted here, and that is deliberate rather than
    an oversight: whether scp puts it through a remote shell at all depends
    on the protocol (SFTP on OpenSSH 9, a remote shell under ``-O``), so a
    quote would be literal in one mode and syntax in the other.  What makes
    it safe instead is that neither end of it can carry a shell character:
    :data:`SAFE_REMOTE_NAME_RX` vets the basename on BOTH directions
    (:func:`inbox_target` on a push, :func:`list_remote` and :func:`pull` on
    a fetch), and the directory half is config, never speech.
    """
    opts = _common_opts(conf, timeout_s or conf.transfer_timeout_s)
    local_arg = local if os.path.isabs(local) else os.path.join(".", local)
    remote_arg = f"{conf.target}:{remote}"
    ends = [local_arg, remote_arg] if push else [remote_arg, local_arg]
    return [SCP_BIN, "-p", "-q"] + opts + ["--"] + ends


def run_ssh(conf: RemoteConfig, command: str,
            timeout_s: float = 0.0) -> SshResult:
    """THE SEAM.  One bounded ssh round trip.

    Popen + communicate(timeout) + kill-WITHOUT-wait, never
    ``subprocess.run``: run() kills the child on a timeout and then WAITS
    for it, and an ssh wedged in a TCP connect to a sleeping host that
    silently drops packets is precisely the child that never comes back.
    Kill it and walk away; a leaked zombie costs nothing next to a voice
    turn that never ends.
    """
    budget = timeout_s or conf.timeout_s
    argv = ssh_argv(conf, command, budget)
    log.info("remote: ssh %s (%.0fs budget)", conf.name, budget)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("remote: ssh unavailable: %s", type(exc).__name__)
        return SshResult(False, reason="no-ssh")
    try:
        out, err = proc.communicate(timeout=budget)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        log.warning("remote: %s did not answer in %.0fs; not waiting",
                    conf.name, budget)
        return SshResult(False, reason="timeout")
    if proc.returncode != 0:
        reason = classify_error(err)
        log.warning("remote: ssh rc=%s (%s)", proc.returncode, reason)
        return SshResult(False, out=out or "", err=err or "", reason=reason)
    return SshResult(True, out=out or "", err=err or "")


def run_copy(conf: RemoteConfig, local: str, remote: str,
             push: bool) -> SshResult:
    """THE SEAM for a transfer.  Same bounded shape as :func:`run_ssh`, on
    the longer transfer budget -- a file is not a question."""
    budget = conf.transfer_timeout_s
    argv = scp_argv(conf, local, remote, push, budget)
    log.info("remote: scp %s %s (%.0fs budget)",
             "->" if push else "<-", conf.name, budget)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("remote: scp unavailable: %s", type(exc).__name__)
        return SshResult(False, reason="no-ssh")
    try:
        out, err = proc.communicate(timeout=budget)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        log.warning("remote: transfer did not finish in %.0fs", budget)
        return SshResult(False, reason="timeout")
    if proc.returncode != 0:
        return SshResult(False, out=out or "", err=err or "",
                         reason=classify_error(err))
    return SshResult(True, out=out or "", err=err or "")


# ------------------------------------------------------------ reachability
def tailnet_state(conf: RemoteConfig) -> str:
    """Where HPCOMPUTER stands on the tailnet: "online", "offline" (a known
    peer that is down), "absent" (no such peer -- it has never joined), or
    "unknown" (the daemon could not be asked).

    This is what lets the failure line be honest.  "I couldn't reach it" is
    true of a sleeping laptop and of a machine that was never set up, and
    those need different sentences from him -- one is "wake it", the other
    is "install Tailscale on it".
    """
    argv = [TAILSCALE_BIN, f"--socket={TAILSCALE_SOCKET}", "status", "--json"]
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, text=True,
                                errors="replace")
        out, _err = proc.communicate(timeout=5)
    except (OSError, subprocess.SubprocessError):
        log.debug("remote: tailscale status unavailable", exc_info=True)
        return "unknown"
    if proc.returncode != 0 or not out:
        return "unknown"
    try:
        data = json.loads(out)
    except (ValueError, TypeError):
        return "unknown"
    want = (conf.host or conf.name).split(".")[0].lower()
    peers = data.get("Peer") or {}
    if not isinstance(peers, dict):
        return "unknown"
    for peer in peers.values():
        if not isinstance(peer, dict):
            continue
        names = [str(peer.get("HostName") or ""),
                 str(peer.get("DNSName") or "").split(".")[0]]
        if any(n.lower() == want for n in names if n):
            return "online" if peer.get("Online") else "offline"
    return "absent"


def unreachable_reason(conf: RemoteConfig) -> str:
    """Turn a failed round trip into the MOST specific reason available, by
    asking the tailnet what it thinks.  "" when nothing better is known."""
    state = tailnet_state(conf)
    if state == "absent":
        return "off-tailnet"
    if state == "offline":
        return "asleep"
    return ""


# ---------------------------------------------------- tier 1: questions
# Named, parameterless, READ-ONLY commands.  A spoken phrase selects a row;
# the row's `cmd` is what runs.  Nothing from a transcript is interpolated,
# which is why this tier needs no confirmation: the worst a misheard word
# can do is run a different question from this table, or none.
QUERIES = {
    "up": {"cmd": "uptime -p 2>/dev/null || uptime",
           "say": "how long it's been up"},
    "disk": {"cmd": "df -Ph / | tail -1", "say": "how the disk looks"},
    "load": {"cmd": "cat /proc/loadavg 2>/dev/null | cut -d' ' -f1-3",
             "say": "what the load is"},
    "who": {"cmd": "who 2>/dev/null | head -5", "say": "who's logged in"},
    "inbox": {"cmd": "", "say": "what's in the inbox"},   # built from config
}


def query_command(conf: RemoteConfig, key: str) -> str:
    """The command for a QUERIES row.  "inbox" is the one row whose command
    depends on config rather than being a constant -- built here through
    :func:`shell_path` (quoted, but with a leading tilde left for the remote
    shell to expand), never from anything spoken."""
    if key == "inbox":
        return f"ls -1p -- {shell_path(conf.inbox)} 2>/dev/null | head -40"
    row = QUERIES.get(key)
    return row["cmd"] if row else ""


def ask(conf: RemoteConfig, key: str) -> SshResult:
    """Run one allow-listed read-only question.  Unattended by design."""
    why = missing_reason(conf)
    if why:
        return SshResult(False, reason=why)
    cmd = query_command(conf, key)
    if not cmd:
        return SshResult(False, reason="failed")
    res = run_ssh(conf, cmd)
    if not res.ok and res.reason in ("unreachable", "timeout"):
        better = unreachable_reason(conf)
        if better:
            return SshResult(False, out=res.out, err=res.err, reason=better)
    return res


# ------------------------------------------------------------ tier 2: files
def remote_dir(conf: RemoteConfig, key: str) -> str:
    """The configured remote folder for a spoken key, or "" if not allowed."""
    if key == "inbox":
        return conf.inbox
    return conf.pull_dirs.get(key, "")


def list_remote(conf: RemoteConfig, key: str) -> tuple[list, str]:
    """Filenames in an allow-listed remote folder, and a failure reason.

    ``ls -1p`` marks directories with a trailing slash so they can be
    dropped without a second round trip.  Every surviving name must match
    :data:`SAFE_REMOTE_NAME_RX`; anything else is DISCARDED rather than
    escaped, and that is deliberate -- a name containing a quote or a
    newline is not a file worth risking a quoting bug for.
    """
    folder = remote_dir(conf, key)
    if not folder:
        return [], "not-there"
    res = run_ssh(conf, f"ls -1p -- {shell_path(folder)}")
    if not res.ok:
        reason = res.reason
        if reason in ("unreachable", "timeout"):
            reason = unreachable_reason(conf) or reason
        return [], reason
    names = []
    for raw in (res.out or "").splitlines()[:LISTING_CAP]:
        line = raw.strip()
        if not line or line.endswith("/"):
            continue
        if SAFE_REMOTE_NAME_RX.match(line):
            names.append(line)
        else:
            log.info("remote: skipping an unquotable name in %s", key)
    return names, ""


def match_remote(said: str, names: Sequence[str]) -> filepick.Pick:
    """Fuzzy-match a spoken name against the REMOTE's own listing.

    Reuses the local resolver's scorer so "the budget one" means the same
    thing on both machines, and returns the same three-way Pick, so an
    ambiguous remote name ASKS exactly as an ambiguous local one does.
    """
    said = (said or "").strip()
    if not said:
        return filepick.Pick(reason="empty")
    scored = [(filepick.score(said, n), n) for n in names]
    scored = [(s, n) for s, n in scored if s >= filepick.FUZZY_FLOOR]
    if not scored:
        return filepick.Pick(reason="not-found")
    scored.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    top = scored[0][0]
    tied = [Path(n) for s, n in scored if s >= top - 0.02]
    if len(tied) > 1:
        return filepick.Pick(candidates=tied[:filepick.MAX_CANDIDATES])
    return filepick.Pick(path=Path(scored[0][1]))


def inbox_target(conf: RemoteConfig, name: str) -> str:
    """Where a pushed file lands: ALWAYS inside ``remote.inbox``.  "" when
    the local basename is not one this module will write remotely.

    The basename is taken from the LOCAL file (which ``filepick`` already
    resolved and bounded), never from the transcript, and joined to the one
    configured folder.  There is no code path by which a spoken phrase
    chooses a remote directory -- that is the whole point of this function
    existing instead of a formatted string at the call site.

    The name is run through :data:`SAFE_REMOTE_NAME_RX` here as well, which
    the pull side has always done and this side had not.  A LOCAL file may
    legitimately be called ``a;b.txt`` or hold a backtick, and that name was
    reaching the remote argument unescaped: harmless while scp speaks SFTP,
    remote command execution the day it does not (``-O``, an older scp, a
    different ``SCP_BIN``).  Refused rather than escaped, exactly as an
    unquotable remote name is dropped rather than quoted.
    """
    base = os.path.basename(name)
    if not SAFE_REMOTE_NAME_RX.match(base):
        log.warning("remote: refusing to push a name I won't write remotely")
        return ""
    return f"{scp_path(conf.inbox).rstrip('/')}/{base}"


def push(conf: RemoteConfig, local: Path) -> SshResult:
    """Copy ONE local file into the remote inbox.  Confirmed before this."""
    why = missing_reason(conf)
    if why:
        return SshResult(False, reason=why)
    roots = filepick.expand_roots(conf.local_roots)
    bad = filepick.check_file(local, roots, conf.max_mb)
    if bad:
        return SshResult(False, reason=bad)
    target = inbox_target(conf, local.name)
    if not target:
        return SshResult(False, reason="odd-name")
    res = run_copy(conf, str(local), target, push=True)
    if not res.ok and res.reason in ("unreachable", "timeout"):
        better = unreachable_reason(conf)
        if better:
            return SshResult(False, out=res.out, err=res.err, reason=better)
    return res


def pull_target(conf: RemoteConfig, name: str) -> Path:
    """Where a pulled file lands locally: the FIRST configured root (his
    Desktop by default), under the remote basename.  Same reasoning as
    ``inbox_target`` in the other direction."""
    roots = filepick.expand_roots(conf.local_roots)
    root = roots[0] if roots else Path(os.path.expanduser("~"))
    return root / os.path.basename(name)


def pull(conf: RemoteConfig, key: str, remote_name: str) -> SshResult:
    """Copy ONE file out of an allow-listed remote folder.

    ``remote_name`` must be a name this module read out of that folder's own
    listing -- callers get it from :func:`match_remote`.  It is re-checked
    here anyway, because a seam that is only safe when its caller behaves is
    not a safe seam.
    """
    why = missing_reason(conf)
    if why:
        return SshResult(False, reason=why)
    folder = remote_dir(conf, key)
    if not folder or not SAFE_REMOTE_NAME_RX.match(remote_name or ""):
        return SshResult(False, reason="not-there")
    dest = pull_target(conf, remote_name)
    if dest.exists():
        # Never clobber something of his without being told to.
        return SshResult(False, reason="exists")
    src = f"{scp_path(folder).rstrip('/')}/{remote_name}"
    res = run_copy(conf, str(dest), src, push=False)
    if not res.ok and res.reason in ("unreachable", "timeout"):
        better = unreachable_reason(conf)
        if better:
            return SshResult(False, out=res.out, err=res.err, reason=better)
    return res
