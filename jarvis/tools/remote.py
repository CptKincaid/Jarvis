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
SFTP_BIN = "sftp"
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
    # "windows" | "posix": which far side the read side talks to.  Windows
    # is shipped because that is what HPCOMPUTER is (F07); it decides the
    # question table, and whether a listing is a shell `ls` or SFTP.
    os: str = "windows"
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


_POSIX_WORDS = ("posix", "linux", "unix", "mac", "macos", "darwin", "bsd")


def _norm_os(value) -> str:
    """Two words and no third.  Anything that is not plainly POSIX is
    windows, the shipped default -- a typo must not silently switch the
    lane onto a shell that is not there."""
    word = str(value or "").strip().lower()
    if word in _POSIX_WORDS:
        return "posix"
    if word and word not in ("windows", "win", "win32", "nt"):
        log.warning("remote.os %r is not windows|posix; using windows", word)
    return "windows"


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
        os=_norm_os(_cfg_get(cfg, "remote.os", "")),
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
    # A BLANK key is a refusal, not "ready" (F11, 2026-09-03).  The shipped
    # default is "", and with IdentitiesOnly=yes and no -i, ssh offers only
    # the default-named identities -- of which ~/.ssh here has none
    # (measured: no id_rsa/id_ecdsa/id_ed25519; the key is ~/.ssh/hpcomputer).
    # So a blank line in his settings opened a socket and came back as
    # "HPCOMPUTER turned my key away, sir", blaming the far side.
    if not conf.key_path:
        return "no-key"
    if not key_file(conf):
        return "bad-key"
    return ""


# ---------------------------------------------------------------- lines
FAIL_LINES = {
    "disabled": "{name} isn't set up in my settings yet, sir; "
                f"remote.enabled is false in {CONFIG_HINT}.",
    "no-host": "I've no address for {name}, sir; remote.host is empty in "
               f"{CONFIG_HINT}.",
    "no-user": "I don't know which account to use on {name}, sir; "
               "remote.user is empty.",
    "no-key": "I've no key for {name}, sir; remote.key_path is empty in "
              f"{CONFIG_HINT}.",
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
    # The far side's OWN shell said it does not know the command (F07): a
    # POSIX row sent to cmd.exe, or a Windows row sent to sh.  That is a
    # remote.os problem on this side, and it names the command so he can
    # tell which -- the generic line hid this behind "wouldn't answer".
    "wrong-os": "{name} doesn't know that command, sir{missing}. I asked it "
                "the way I'd ask a {os_word} box; remote.os in "
                f"{CONFIG_HINT} is probably wrong.",
    "failed": "{name} wouldn't answer that, sir.",
}

# The one door that always says no.  Kept here beside the failures so the
# refusal is as maintained as the successes are.
FREEFORM_REFUSAL = (
    "I don't run loose commands on {name}, sir. I'll fetch files, send "
    "them, and answer a short list of questions about it -- but not that.")


# The command a shell did not recognise, in the three wordings that matter:
# cmd.exe ("'ls' is not recognized ..."), dash ("sh: 1: powershell: not
# found"), bash ("bash: powershell: command not found").
_MISSING_CMD_RX = re.compile(
    r"'([^'\r\n]+)' is not recognized|"
    r"(?:^|\n)(?:\w+: (?:\d+: )?)?(\S+): (?:command )?not found", re.I)


def missing_command(err: str) -> str:
    """The command name out of a wrong-os stderr, or "" when the text does
    not carry one ("The system cannot find the path specified.")."""
    m = _MISSING_CMD_RX.search(err or "")
    if not m:
        return ""
    return next((g for g in m.groups() if g), "")


def fail_line(conf: RemoteConfig, reason: str, err: str = "") -> str:
    """The spoken line for a reason.  ``err`` is the stderr it came from,
    used only by the wrong-os line to name the command (F07)."""
    line = FAIL_LINES.get(reason) or FAIL_LINES["failed"]
    cmd = missing_command(err) if reason == "wrong-os" else ""
    return line.format(name=conf.name, timeout=int(conf.timeout_s),
                       missing=f" -- it has no {cmd}" if cmd else "",
                       os_word="Windows" if conf.os == "windows" else "Linux")


_HOSTKEY_RX = re.compile(
    r"host key.*(changed|verification failed)|REMOTE HOST IDENTIFICATION",
    re.I)
# What the far side's shell says to a command that is not there.  cmd.exe
# says the first for `ls`, and the second for the `2>/dev/null` in a POSIX
# row (it reads it as a redirect into a path that does not exist); dash and
# bash say the last two to `powershell`.  Any of them means the table was
# built for the wrong OS -- remote.os, not the host, is what to look at.
_WRONG_OS_RX = re.compile(
    r"is not recognized as an internal or external command|"
    r"the system cannot find the path specified|"
    r"command not found|sh: \d+: \S+: not found", re.I)
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
# The third wording is sftp's, for a folder that is not there -- measured
# 2026-09-03 with `sftp -b -`: `Can't ls: "x" not found` on stderr, rc 1.
_MISSING_RX = re.compile(
    r"no such file or directory|not a directory|can't ls: .* not found", re.I)


def classify_error(err: str) -> str:
    """ssh/scp/sftp stderr in one word.  Order matters, and each rung is a
    real confusion this avoids: a changed host key ALSO prints "Permission
    denied" further down and is the one thing he must look at himself; a
    shell that does not know the command is a remote.os mistake on THIS
    side and must not be filed under any of the far side's failures; a
    full remote disk and a refused write both look like a generic failure
    but need different sentences; and "no such file" during a pull is a
    misremembered name, not a broken link."""
    text = err or ""
    if _HOSTKEY_RX.search(text):
        return "hostkey"
    if _WRONG_OS_RX.search(text):
        return "wrong-os"
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
_TAILNET_IP_RX = re.compile(r"^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+$")


def tailnet_host(conf: RemoteConfig) -> bool:
    """Is ``remote.host`` a TAILNET address -- the only kind the tailnet
    view and the SOCKS proxy have anything to say about?

    F08 (2026-09-03).  HPCOMPUTER is reached over the LAN (192.168.50.114 /
    hpcomputer.local; it has no Tailscale peer), and every door asked the
    tailnet about it anyway: the host's first label ("192", "hpcomputer")
    is not a peer, so ``tailnet_state`` said "absent" and the spoken line
    was "isn't on the tailnet, sir -- I can't see it at all" -- without a
    socket ever opening, and with a real unreachable/timeout (the box
    asleep) rewritten into the same wrong sentence.

    A tailnet address is a MagicDNS name (``*.ts.net``), an address in the
    CGNAT block Tailscale hands out (100.64/10), or a bare single label
    WITH the proxy configured -- that is the MagicDNS short name, and the
    proxy is the only way this box reaches MagicDNS.  A bare label with no
    proxy is the router's name for a LAN box, and ``.local`` is mDNS.
    """
    host = (conf.host or "").strip().lower()
    if not host:
        return False
    if host.endswith(".ts.net") or _TAILNET_IP_RX.match(host):
        return True
    return "." not in host and bool(conf.socks_proxy)


def proxy_args(conf: RemoteConfig) -> list:
    """The ``ProxyCommand`` that makes the tailnet reachable from a
    USERSPACE tailscaled.

    Without a tun device there is no route to 100.64/10 at all, so a plain
    ssh to a tailnet name fails with "Network is unreachable" no matter how
    healthy the tailnet is.  The daemon's SOCKS5 port is the supported way
    through, and ``nc -X 5 -x`` speaks it.  Empty when ``socks_proxy`` is
    blank (the day this box gets a real tun device) and -- F08 -- for any
    host that is not a tailnet address: the proxy is tailscaled's way to
    the TAILNET, whether it forwards to a LAN IP at all is unverified, and
    his own verified line is a direct ``ssh h2pey@192.168.50.114``.  So the
    shipped proxy default no longer has to be blanked for the LAN box.
    """
    if not conf.socks_proxy or not tailnet_host(conf):
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
    asking the tailnet what it thinks.  "" when nothing better is known --
    and always "" for a LAN address (F08), where the tailnet knows nothing
    and "absent" would be a confident wrong answer; the ssh result
    ("unreachable", "timeout") is then the honest one."""
    if not tailnet_host(conf):
        return ""
    state = tailnet_state(conf)
    if state == "absent":
        return "off-tailnet"
    if state == "offline":
        return "asleep"
    return ""


# ---------------------------------------------------- tier 1: questions
# Named, parameterless, READ-ONLY commands, one per OS.  A spoken phrase
# selects a row; the row's command for ``conf.os`` is what runs.  Nothing
# from a transcript is interpolated, which is why this tier needs no
# confirmation: the worst a misheard word can do is run a different
# question from this table, or none.
#
# The Windows column (F07, 2026-09-03).  HPCOMPUTER runs the built-in
# OpenSSH Server, whose login shell is cmd.exe unless the DefaultShell
# registry value says PowerShell, and the POSIX rows came back from it as
# "'uptime' is not recognized as an internal or external command".  Every
# Windows row is one ``powershell -Command "<expression>"``, chosen for ONE
# property: no ``$``, no backtick, no nested quote, one double-quoted
# argument.  That is the quoting that both cmd.exe and PowerShell hand to
# powershell.exe unchanged -- a ``$var`` would be interpolated by an outer
# PowerShell before it ever ran, and a nested quote is parsed differently
# by the two.  ``net``/``wmic``/``quser`` were passed over: wmic is gone
# from Windows 11 24H2 and quser is absent on Home editions.
#
# MEASURED here: the POSIX rows (this box) and sftp's batch format.  NOT
# measured: the Windows rows against HPCOMPUTER -- by instruction the first
# live command is Hunter's, and until then they are a reading of the
# PowerShell docs, not a result.  The wrong-os rung is what he hears if the
# reading was wrong.
_PS = "powershell -NoProfile -NonInteractive -Command "
QUERIES = {
    "up": {"posix": "uptime -p 2>/dev/null || uptime",
           "windows": _PS + '"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime"',
           "say": "how long it's been up", "say_windows": "up since"},
    "disk": {"posix": "df -Ph / | tail -1",
             "windows": _PS + '"[math]::Round((Get-PSDrive C).Free/1GB)"',
             "say": "how the disk looks",
             "say_windows": "free space on C, in gigabytes"},
    "load": {"posix": "cat /proc/loadavg 2>/dev/null | cut -d' ' -f1-3",
             "windows": _PS + '"(Get-CimInstance Win32_Processor).LoadPercentage"',
             "say": "what the load is", "say_windows": "the CPU load, in percent"},
    "who": {"posix": "who 2>/dev/null | head -5",
            "windows": _PS + '"(Get-CimInstance Win32_ComputerSystem).UserName"',
            "say": "who's logged in"},
    # Built from config on POSIX (query_command); listed over SFTP on
    # Windows (ask), where there is no shell to build it for.
    "inbox": {"posix": "", "windows": "", "say": "what's in the inbox"},
}


def query_command(conf: RemoteConfig, key: str) -> str:
    """The command for a QUERIES row on ``conf.os``.  "inbox" is the one row
    whose POSIX command depends on config rather than being a constant --
    built here through :func:`shell_path` (quoted, but with a leading tilde
    left for the remote shell to expand), never from anything spoken.  On
    Windows it is "" on purpose: :func:`ask` lists the inbox over SFTP."""
    row = QUERIES.get(key)
    if not row:
        return ""
    if key == "inbox":
        if conf.os == "windows":
            return ""
        return f"ls -1p -- {shell_path(conf.inbox)} 2>/dev/null | head -40"
    return row.get(conf.os) or ""


def query_say(conf: RemoteConfig, key: str) -> str:
    """How the answer is introduced -- per OS where the output differs in
    kind (a boot TIME on Windows, an uptime on POSIX)."""
    row = QUERIES.get(key) or {}
    return row.get(f"say_{conf.os}") or row.get("say") or ""


def ask(conf: RemoteConfig, key: str) -> SshResult:
    """Run one allow-listed read-only question.  Unattended by design."""
    why = missing_reason(conf)
    if why:
        return SshResult(False, reason=why)
    if key == "inbox" and conf.os == "windows":
        names, why = list_remote(conf, "inbox")
        return SshResult(not why, out="\n".join(names[:40]), reason=why)
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
def _remote_folder(conf: RemoteConfig, raw: str) -> str:
    """A configured remote folder as the far side is addressed.  On Windows
    a backslash becomes a slash: SFTP paths are slash-separated on every
    server (Windows OpenSSH takes ``C:/Users/...``), scp speaks SFTP, and to
    sftp's own tokenizer a backslash is an escape, not a separator."""
    folder = (raw or "").strip()
    if conf.os == "windows":
        folder = folder.replace("\\", "/")
    return folder


def remote_dir(conf: RemoteConfig, key: str) -> str:
    """The configured remote folder for a spoken key, or "" if not allowed."""
    if key == "inbox":
        return _remote_folder(conf, conf.inbox)
    return _remote_folder(conf, conf.pull_dirs.get(key, ""))


# ---- the SFTP listing (F07) ----
# Windows OpenSSH has no `ls`, but it has the SFTP subsystem -- the same
# one scp already speaks -- so a Windows folder is listed the way it is
# copied from: no shell on the far side at all.  Format measured on this
# box 2026-09-03 (`sftp -q -b - -D /usr/lib/openssh/sftp-server`, which
# pipes straight to the local sftp-server; no socket):
#
#     sftp> ls -ln "jarvis-outbox"                          <- the echo
#     -rw-rw-r--    ? hunterp  hunterp   1 Sep  3 12:21 jarvis-outbox/a.txt
#     drwxrwxr-x    ? hunterp  hunterp 4096 Sep  3 12:21 jarvis-outbox/sub dir
#
# Batch mode echoes each command; a name comes back PATH-PREFIXED; the mode
# column's first character tells a directory; dotfiles are absent without
# -a; a missing folder is `Can't ls: "x" not found` on stderr with rc 1.
# `-ln` rather than `-1` because -1 cannot tell a directory from a file, and
# `-n` makes the long line the LOCAL sftp client's own ls_file() format
# whatever the server sends, which is the format above.  A path is quoted
# with double quotes for sftp's tokenizer; a folder that contains one is
# refused rather than escaped, the rule this module applies to names.
_SFTP_UNQUOTABLE_RX = re.compile(r'["\r\n]')
_SFTP_LONG_FIELDS = 8              # mode links user group size mon day time


def sftp_argv(conf: RemoteConfig, timeout_s: float = 0.0) -> list:
    """``sftp -b -`` reads its commands from stdin, so the batch is Popen
    input and never an argument; the same guards as ssh's."""
    return ([SFTP_BIN, "-q", "-b", "-"]
            + _common_opts(conf, timeout_s or conf.timeout_s)
            + ["--", conf.target])


def run_sftp(conf: RemoteConfig, batch: str,
             timeout_s: float = 0.0) -> SshResult:
    """THE SEAM for a listing.  One bounded SFTP session fed ``batch`` on
    stdin; the same Popen + communicate(timeout) + kill-without-wait shape
    as :func:`run_ssh`, for the same reason."""
    budget = timeout_s or conf.timeout_s
    argv = sftp_argv(conf, budget)
    log.info("remote: sftp %s (%.0fs budget)", conf.name, budget)
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("remote: sftp unavailable: %s", type(exc).__name__)
        return SshResult(False, reason="no-ssh")
    try:
        out, err = proc.communicate(input=batch.rstrip("\n") + "\n",
                                    timeout=budget)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        log.warning("remote: %s did not answer sftp in %.0fs; not waiting",
                    conf.name, budget)
        return SshResult(False, reason="timeout")
    if proc.returncode != 0:
        reason = classify_error(err)
        log.warning("remote: sftp rc=%s (%s)", proc.returncode, reason)
        return SshResult(False, out=out or "", err=err or "", reason=reason)
    return SshResult(True, out=out or "", err=err or "")


def sftp_listing(out: str) -> list:
    """Filenames out of an ``ls -ln`` batch's stdout: the echo skipped,
    directories dropped, the path prefix removed.  Names are NOT vetted
    here -- :func:`list_remote` does that, the same for both listings."""
    names = []
    for raw in (out or "").splitlines()[:LISTING_CAP + 1]:
        line = raw.rstrip()
        if not line or line.startswith("sftp>") or line[0] == "d":
            continue
        fields = line.split(None, _SFTP_LONG_FIELDS)
        if len(fields) <= _SFTP_LONG_FIELDS:
            continue
        names.append(fields[_SFTP_LONG_FIELDS].rsplit("/", 1)[-1])
    return names


def _list_sftp(conf: RemoteConfig, folder: str) -> SshResult:
    path = scp_path(folder)
    if _SFTP_UNQUOTABLE_RX.search(path):
        log.warning("remote: refusing to list a folder I won't put in an "
                    "sftp batch line")
        return SshResult(False, reason="not-there")
    return run_sftp(conf, f'ls -ln "{path}"')


def list_remote(conf: RemoteConfig, key: str) -> tuple[list, str]:
    """Filenames in an allow-listed remote folder, and a failure reason.

    POSIX: ``ls -1p`` marks directories with a trailing slash so they can
    be dropped without a second round trip.  Windows: an SFTP ``ls -ln``,
    read by :func:`sftp_listing` (F07 -- there is no ``ls`` there).  Either
    way every surviving name must match :data:`SAFE_REMOTE_NAME_RX`;
    anything else is DISCARDED rather than escaped, and that is deliberate
    -- a name containing a quote or a newline is not a file worth risking
    a quoting bug for.
    """
    folder = remote_dir(conf, key)
    if not folder:
        return [], "not-there"
    if conf.os == "windows":
        res = _list_sftp(conf, folder)
    else:
        res = run_ssh(conf, f"ls -1p -- {shell_path(folder)}")
    if not res.ok:
        reason = res.reason
        if reason in ("unreachable", "timeout"):
            reason = unreachable_reason(conf) or reason
        return [], reason
    if conf.os == "windows":
        listed = sftp_listing(res.out)
    else:
        listed = [raw.strip() for raw in (res.out or "").splitlines()]
    names = []
    for line in listed[:LISTING_CAP]:
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
    return f"{scp_path(remote_dir(conf, 'inbox')).rstrip('/')}/{base}"


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
