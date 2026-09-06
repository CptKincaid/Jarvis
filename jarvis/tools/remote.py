"""HPCOMPUTER -- files both ways and a SHORT allow-list of read-only questions.

GROUND TRUTH, re-measured on the Spark 2026-09-03.  The 09-02 block this
replaces said ``hpcomputer.local`` was a stale mDNS record, the host
answered nothing and no key existed -- true that night, none of it now,
and the design that followed from it (tailnet name + SOCKS proxy, a POSIX
shell on the far side) targeted the wrong link.  Written down because every
one of these facts changed the design, and a future reader who assumes
otherwise will build the wrong thing:

* **HPCOMPUTER is a Windows box on the LAN, not a tailnet peer.**
  ``hpcomputer.local`` resolves to 192.168.50.114 (``getent hosts``) and
  there is a live ARP entry for it (60:cf:84:ad:fd:91).  ``tailscale
  status`` has two nodes -- ``spark`` (100.70.145.63) and ``iphone172``
  (offline) -- and no HPCOMPUTER, nor a plan for one.  So the tailnet view
  and the SOCKS proxy apply to a tailnet ADDRESS only
  (:func:`tailnet_host`); a LAN host gets ssh's own answer.
* **Its ssh is Windows OpenSSH Server** (``Add-WindowsCapability
  OpenSSH.Server``, staged and waiting on a reboot as of the 09-03
  checklist; user ``h2pey``).  Its login shell is cmd.exe or PowerShell:
  there is no ``ls``, ``df``, ``uptime`` or ``$HOME`` there.  Hence
  ``remote.os`` (windows shipped), a question table with a command per OS
  (:data:`QUERIES`), and the Windows folder listing over SFTP -- the
  channel scp already speaks -- rather than a remote shell.  Port 22 has
  NOT been probed from here (no network, by instruction), and the Windows
  rows have not been run: the first live command is Hunter's.
* **The key is ``~/.ssh/hpcomputer``** (ed25519, generated 09-03 00:44; its
  public half belongs in ``C:\\ProgramData\\ssh\\administrators_authorized_keys``
  because his account is an administrator).  ``~/.ssh`` holds NO
  default-named identity, so a blank ``remote.key_path`` offers ssh nothing
  and is refused before a socket opens ("no-key").
* **tailscaled here runs in USERSPACE mode** (``--tun=userspace-networking
  --socks5-server=localhost:1055``, a rootless user unit whose socket is
  ``~/.local/share/tailscale/tailscaled.sock``).  There is NO tun device,
  so a tailnet name routes only through that SOCKS5 port, which is why
  :func:`ssh_argv` grows a ``ProxyCommand`` for a tailnet address.  ``nc
  -X 5 -x`` is present; ssh, scp and sftp are OpenSSH 9.6p1.

So this module ships ``enabled: false`` and every entry point refuses out
loud with a reason that names what is missing.  Nothing here has been run
against the real host -- by instruction, Hunter runs the first live command
himself.  What IS tested is every decision this module makes before the
socket opens, which is where the irreversible mistakes live -- and, for
the Windows side, the reading of sftp's batch output as measured locally
(``sftp -D`` straight to this box's sftp-server; no socket).

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

``run_ssh``, ``run_copy`` and ``run_sftp`` are the three module-level
seams, looked up at call time, so tests replace them and no test in this
repo can reach a network, a host key or a disk it did not make.
"""
from __future__ import annotations

import errno
import itertools
import json
import os
import re
import shlex
import subprocess
import threading
import time
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
    # NOT the line above, and that distinction is finding M2.  When our own
    # staged copy disappears mid-flight, nothing was holding his name -- so
    # "there's already a file by that name" is simply false, and it was
    # being said every time the folder lane's sweep ate this lane's file.
    "staging-gone": "Something removed the copy I was staging on {name} "
                    "before I could put it in place, sir; nothing of yours "
                    "was touched, and I'll try it again.",
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


# ---- claiming a name on the far side (F-J) ----
#
# WHERE THESE NUMBERS COME FROM, because round 7 found them claiming a
# session nobody can show.  Until this edit, this block opened by attributing
# every number below to a first-person run against HPCOMPUTER, naming its
# server build and its SFTP protocol version.  The exact wording is in
# 36d26fe, which is where it landed and the right place to read it.
#
# It cannot be right.  docs/capabilities.md records HPCOMPUTER on 2026-09-03
# as ping 100% loss with 22/445/3389/5900/8008/2343 CLOSED, and this very
# file says 160 lines above that the Windows rows were NOT measured, "by
# instruction".  It said both things at once.  No probe of his machine is
# allowed from here, and none was run.
#
# THE NUMBERS THEMSELVES ARE RIGHT, and every one was RE-MEASURED HERE on
# 2026-09-05 against THIS BOX'S OWN /usr/lib/openssh/sftp-server, driven
# over a pipe by `scp -D` and `sftp -D`: no socket, no network, nothing of
# his touched.  tests/test_provenance.py re-derives them on every run rather
# than quoting them, so they cannot become claims nobody checks.
#
# MEASURED HERE.  scp TRUNCATES, and nothing that writes on this link can be
# made to refuse: a 1111-byte source onto a 22222-byte target left 1111
# bytes, exit 0, nothing on stdout or stderr -- in the default SFTP mode,
# under the legacy `-O` protocol, and through `sftp put` alike.  Our client
# is OpenSSH_9.6p1 and its complete flag set is -346ABCOpqRrsTv plus
# -c -D -F -i -J -l -o -P -S -X; there is no no-clobber among them.  The
# CLIENT runs on this box, so that last one is a local fact outright.  So
# the refusal cannot come from the write and has to be a separate step.
#
# That step is a RENAME, and the client offers two of them which behave
# OPPOSITELY.  MEASURED HERE, both confirmed on the wire with -vvv rather
# than read off the manual:
#
#   bare `rename`  -> SSH2_FXP_EXTENDED(posix-rename@openssh.com), which
#                     SILENTLY REPLACES.  10 of 10 rounds destroyed the file
#                     at the target name, exit status 0.  The client picks
#                     this whenever the server advertises it, and the
#                     sftp-server on this box does.
#   `rename -l`    -> SSH2_FXP_RENAME (opcode 18), which REFUSES.  10 of 10
#                     rounds refused, exit status 1, `remote rename ...:
#                     Failure` on stderr, BOTH files byte-intact.
#
# And the measurement that actually decides it: two concurrent sftp sessions
# each renaming its own temp onto the SAME name, 12 rounds -- 7 wins to 5
# (the split is scheduling and means nothing), ZERO rounds where both moved,
# ZERO corrupt, the loser's temp intact 12 of 12.  That is an atomic claim,
# not a narrowed window.
#
# INFERRED, for the far side, and said as an inference.  What HIS server
# does with opcode 18 was not measured and cannot be from here.  It is read
# off the protocol -- draft-ietf-secsh-filexfer-02 §6.5 says SSH_FXP_RENAME
# SHOULD fail when the target exists -- and off Windows' own MoveFile
# semantics.  THE SAFETY DOES NOT REST ON THAT INFERENCE: `-l` pins the
# OPCODE at the client, whatever the far server advertises, so the worst a
# wrong inference can do is make the claim fail closed rather than open.
# And if the far side ever did replace on opcode 18, the read-back after
# every transfer is the second net.
#
# So RENAME_FLAG is not a nicety.  Dropping it reproduces the bug this
# closes, silently, with no error anywhere to notice.  It is asserted in
# tests/test_foldersync.py rather than left to a reader.
RENAME_FLAG = "-l"

# Our own in-flight file on the far side.  Every byte scp writes over there
# goes at a name of THIS shape and never at a name he could have chosen, so
# a write can never land on a file of his; and it is the only shape
# :func:`sftp_remove` will delete, which is what keeps "this module cannot
# delete anything of his" true by construction rather than by care.
REMOTE_TEMP_PREFIX = "jarvis-part-"
REMOTE_TEMP_SUFFIX = ".tmp"

# A STAGING DIRECTORY of ours on the far side, and the file inside it.  See
# :func:`sftp_mkdir` for why a directory is the claim.
REMOTE_STAGE_SUFFIX = ".d"
STAGE_FILE = "f"

# FINDING O.  ``remote_temp_name()`` was pid + second + a counter the CALLER
# supplied, and remote.push supplied none -- so it defaulted to 0 and two
# spoken sends inside the same second shared a staging name, each scp
# truncating the other's bytes.  The counter belongs to the process, not to
# whoever remembers to pass one; an explicit ``seq`` is still honoured
# because foldersync numbers its own files within a pass.
_NAME_SEQ = itertools.count()
_NAME_LOCK = threading.Lock()


def _next_seq() -> int:
    with _NAME_LOCK:
        return next(_NAME_SEQ)


def remote_temp_name(seq: int = None) -> str:
    """A name in the remote inbox that is ours beyond argument: the pid, the
    second, and a counter, so two passes and two processes cannot collide
    even if the lock ever failed.  It satisfies SAFE_REMOTE_NAME_RX, so it
    goes through the same vetting every other remote name does."""
    seq = _next_seq() if seq is None else seq
    return (f"{REMOTE_TEMP_PREFIX}{os.getpid()}-{int(time.time())}-"
            f"{int(seq)}{REMOTE_TEMP_SUFFIX}")


def remote_stage_name(seq: int = None) -> str:
    """The name of a staging DIRECTORY of ours, same shape, ``.d`` on the
    end so it can never be mistaken for -- or mistake itself for -- one of
    the ``.tmp`` part files."""
    seq = _next_seq() if seq is None else seq
    return (f"{REMOTE_TEMP_PREFIX}{os.getpid()}-{int(time.time())}-"
            f"{int(seq)}{REMOTE_STAGE_SUFFIX}")


def is_remote_temp(name: str) -> bool:
    """Is this basename one of ours?  Deliberately strict on both ends."""
    base = os.path.basename(name or "")
    return (base.startswith(REMOTE_TEMP_PREFIX)
            and base.endswith(REMOTE_TEMP_SUFFIX)
            and bool(SAFE_REMOTE_NAME_RX.match(base)))


def is_remote_stage(name: str) -> bool:
    """Is this basename a staging DIRECTORY of ours?"""
    base = os.path.basename(name or "")
    return (base.startswith(REMOTE_TEMP_PREFIX)
            and base.endswith(REMOTE_STAGE_SUFFIX)
            and bool(SAFE_REMOTE_NAME_RX.match(base)))


def _ours_over_there(path: str) -> bool:
    """A remote path this module is allowed to write over or delete without
    asking anybody: a part file of our own shape, or ANYTHING inside a
    staging directory of our own shape -- which we only ever get by having
    created that directory exclusively (see :func:`sftp_mkdir`)."""
    text = str(path or "")
    return is_remote_temp(os.path.basename(text)) or \
        is_remote_stage(os.path.basename(os.path.dirname(text)))


# ---- claiming the name we WRITE at, not just the name he sees ----------
# ROW 3 and ROW 5 of the round-3 table, and the third time this same shape
# has been found here.  ``rename -l`` claims the name HE will see, but the
# bytes still went to a temp name with NO check of any kind: an attacker
# measured a 99999-byte file at that name replaced by 30 bytes, with the
# pass reporting "sent".  The window is the whole transfer and the guard
# was only that the name is improbable.
#
# SFTP has no exclusive create for a FILE -- there is no no-clobber flag on
# put, and that is exactly what round 3 measured.  It has exactly one
# operation that creates a name and refuses if it is taken: MKDIR.
#
# MEASURED on this box 2026-09-05, `sftp -q -b - -D
# /usr/lib/openssh/sftp-server` (no socket, no HPCOMPUTER):
#   * mkdir at a free name        -> rc 0
#   * mkdir again at that name    -> rc 1, `remote mkdir "...": Failure`
#   * mkdir over an existing FILE -> rc 1, the same
#   * two sessions racing one mkdir, 20 rounds -> exactly one winner 20/20,
#     never both, never neither
#   * `rename -l` ACROSS directories, onto a taken name -> rc 1 with a
#     99999-byte file at the target still 99999 bytes and our own staged
#     file intact
#   * rmdir of a NON-EMPTY directory -> rc 1: the release can never take a
#     byte with it
# So: create a directory exclusively, put the bytes INSIDE it (where no
# name can be anyone's but ours, by construction), claim his name out of it
# with `rename -l`, and rmdir the empty shell.  Two extra round trips per
# transfer; nothing else on this link can refuse a write.
#
# HONESTLY MODELLED THROUGH FIVE ROUNDS, AND NO LONGER.  The far side is
# OpenSSH for Windows and no probe of it is allowed from here, so the POSIX
# numbers above are real and the Windows behaviour used to be inferred from
# the protocol (SSH_FXP_MKDIR over CreateDirectory, which fails with
# ERROR_ALREADY_EXISTS).
#
# CONFIRMED BY HIM, 2026-09-05.  He was asked directly whether creating a
# folder on HPCOMPUTER fails when that name is already taken, and he
# answered YES.  So the one property this lane's whole safety argument
# stands on is now his statement about his own machine, not a model of it.
#
# THAT IS NOT A LOCAL MEASUREMENT and must never be written up as one.  It
# is stronger than the inference it replaces and weaker than the sftp-server
# numbers above, which were run here.  Three provenances, three different
# words, and the difference is the point: this lane has twice been damaged
# by a confident number with no source.
#
# The old degradation argument is kept, because it is what made this
# shippable while it was still only modelled: if Windows were to allow mkdir
# over an existing directory, this degrades to exactly today's behaviour and
# no further -- the inner name is still ours by shape -- so the change
# cannot be worse than what it replaces.


def sftp_rename(conf: RemoteConfig, src: str, dst: str) -> SshResult:
    """CLAIM ``dst`` for the file at ``src``.  Succeeds only if nothing
    holds that name; never replaces.

    One line per session, deliberately: in ``sftp -b`` batch mode a failing
    line ABORTS the rest of the batch unless it is prefixed with ``-``, so
    a multi-line batch would hide which line failed and swallow the ones
    after it.  One line, one exit status, no ambiguity.

    The far side's refusal is the generic word ``Failure`` -- a malformed
    path says the same thing -- so a caller must read any failure as "I did
    not get that name", never as done.  What matters is that it is never a
    silent success.
    """
    for path in (src, dst):
        if not path or _SFTP_UNQUOTABLE_RX.search(path):
            log.warning("remote: refusing a rename I won't put in an sftp "
                        "batch line")
            return SshResult(False, reason="odd-name")
    return run_sftp(conf, f'rename {RENAME_FLAG} "{src}" "{dst}"')


def sftp_mkdir(conf: RemoteConfig, path: str) -> SshResult:
    """CLAIM a staging directory on the far side.  Succeeds only if nothing
    at all holds that name -- not a file, not a directory.

    The only exclusive create SFTP offers, and therefore the only way the
    bytes of a transfer can go somewhere that is ours rather than somewhere
    that is merely improbably his.  Refuses any name that is not a staging
    directory of our own shape, so this cannot be pointed at a folder of
    his by a caller that gets confused.
    """
    if not path or not is_remote_stage(os.path.basename(path)):
        log.warning("remote: refusing to create a directory that is not one "
                    "of my own staging names")
        return SshResult(False, reason="denied")
    if _SFTP_UNQUOTABLE_RX.search(path):
        return SshResult(False, reason="odd-name")
    return run_sftp(conf, f'mkdir "{path}"')


def sftp_rmdir(conf: RemoteConfig, path: str) -> SshResult:
    """Release a staging directory of ours.  Guarded by the same name rule
    as :func:`sftp_mkdir`, and the server itself refuses a directory that
    is not empty (measured: rc 1), so this operation can never take a byte
    of anything with it."""
    if not path or not is_remote_stage(os.path.basename(path)):
        log.warning("remote: refusing to remove a directory that is not one "
                    "of my own staging names")
        return SshResult(False, reason="denied")
    if _SFTP_UNQUOTABLE_RX.search(path):
        return SshResult(False, reason="odd-name")
    return run_sftp(conf, f'rmdir "{path}"')


def sftp_remove(conf: RemoteConfig, path: str, *,
                claimed: bool = False) -> SshResult:
    """Delete ONE in-flight file OF OURS from the far side.

    The guard is the point.  Nothing in Jarvis may delete a file of his on
    HPCOMPUTER, so this refuses every name that is not one
    :func:`remote_temp_name` made, or one INSIDE a staging directory we
    created exclusively -- a caller that passes his report.pdf gets
    "denied" and no session is opened at all.

    ``claimed=True`` is the ONE widening, and it exists only for the
    setting he has not answered yet: a copy that landed under HIS filename
    and then failed its size check, which the lane can otherwise never take
    back (see ``foldersync.remove_broken_copies``, shipped OFF).  It is a
    keyword, it is never a default, the caller must have created that exact
    name in this run, and it is logged at WARNING every single time so a
    delete over there is never silent.
    """
    if not path or not (claimed or _ours_over_there(path)):
        log.warning("remote: refusing to delete a name that is not one of "
                    "my own in-flight files")
        return SshResult(False, reason="denied")
    if _SFTP_UNQUOTABLE_RX.search(path):
        return SshResult(False, reason="odd-name")
    if claimed and not _ours_over_there(path):
        log.warning("remote: removing %s on %s -- a copy I made in this run "
                    "that arrived the wrong size, because "
                    "foldersync.remove_broken_copies is on", path, conf.name)
    return run_sftp(conf, f'rm "{path}"')


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
    """Copy ONE local file into the remote inbox, at a name nothing holds.
    Confirmed out loud before this is reached.

    The two steps are not ceremony.  This used to be one scp straight at
    ``inbox/<his name>``, with no check of any kind -- and **scp truncates**
    (re-measured HERE on 2026-09-05, against this box's own sftp-server over
    a pipe: a 22222-byte file replaced by 1111 bytes, exit 0, nothing on
    stdout or stderr; the same in the legacy ``-O`` protocol and through
    ``sftp put``, and there is no no-clobber flag on any of them.  It is a
    property of the SFTP protocol and of OUR client, not of his machine,
    which is why it can be measured without touching his machine).  So a
    spoken "send that file" silently destroyed
    whatever was at that name on his Windows machine and then said it had
    arrived.

    So the bytes land at a temp name of OURS and the real name is CLAIMED
    with :func:`sftp_rename`, which refuses a name that is taken.  A refusal
    is "exists" -- the line this module has always had for the pull side and
    could never reach on this one -- and his file is untouched.

    ROUND 4 adds the half round 3 left out.  The temp name was ours only by
    being improbable: the scp that wrote it had no claim on it at all, and a
    99999-byte file at that name was measured being replaced by 30 bytes
    with the push reporting success.  So the transfer now happens inside a
    staging DIRECTORY taken with :func:`sftp_mkdir`, which is the one
    exclusive create this link has.  Four round trips instead of two --
    mkdir, scp, rename, rmdir -- and the two extra are the price of the
    write having a claim on it.

    And the false line (M2): when the rename fails, the far side says only
    "Failure", which round 3 read as "his name is taken" ALWAYS.  If our own
    staged file has gone -- which is exactly what happened when the folder
    lane's pattern sweep ate it mid-flight -- that answer is a lie: nothing
    held the name.  So a failed claim now ASKS what is in our staging
    directory, one round trip, only on failure, and says "staging-gone"
    when the honest answer is that our own copy went missing.
    """
    why = missing_reason(conf)
    if why:
        return SshResult(False, reason=why)
    roots = filepick.expand_roots(conf.local_roots)
    bad = filepick.check_file(local, roots, conf.max_mb)
    if bad:
        return SshResult(False, reason=bad)
    target = inbox_target(conf, local.name)
    stage = inbox_target(conf, remote_stage_name())
    if not target or not stage:
        return SshResult(False, reason="odd-name")
    made = sftp_mkdir(conf, stage)
    if not made.ok:
        reason = made.reason or "failed"
        if reason in ("unreachable", "timeout"):
            reason = unreachable_reason(conf) or reason
        return SshResult(False, out=made.out, err=made.err, reason=reason)
    staging = f"{stage}/{STAGE_FILE}"
    try:
        res = run_copy(conf, str(local), staging, push=True)
        if not res.ok:
            sftp_remove(conf, staging)
            if res.reason in ("unreachable", "timeout"):
                better = unreachable_reason(conf)
                if better:
                    return SshResult(False, out=res.out, err=res.err,
                                     reason=better)
            return res
        claim = sftp_rename(conf, staging, target)
        if claim.ok:
            return SshResult(True, out=res.out, err=res.err)
        # Before blaming his file for holding the name, find out whether our
        # own copy is even still there.  Only on the failure path.
        gone = staged_file_missing(conf, stage)
        # Our own bytes are sitting in his inbox under a name of ours.  Take
        # them away rather than leave litter; the guard in sftp_remove means
        # this line can never reach anything else.
        sftp_remove(conf, staging)
        if claim.reason in ("unreachable", "timeout"):
            better = unreachable_reason(conf)
            if better:
                return SshResult(False, err=claim.err, reason=better)
        if claim.reason in ("failed", ""):
            # The wire says only "Failure", for a held name and for a
            # vanished source alike.  One of those is his file; the other
            # is ours going missing, and saying the first when it was the
            # second is the whole of M2.
            reason = "staging-gone" if gone else "exists"
        else:
            reason = claim.reason
        return SshResult(False, out=claim.out, err=claim.err, reason=reason)
    finally:
        sftp_rmdir(conf, stage)


def staged_file_missing(conf: RemoteConfig, stage: str) -> bool:
    """Has something taken our own staged copy away?  True only when the far
    side ANSWERED and our file was not in the answer; a listing that could
    not be had is not evidence either way, so it says False and the caller
    falls back to the older, blunter reading."""
    res = _list_sftp(conf, stage)
    if not res.ok:
        return False
    return STAGE_FILE not in sftp_listing(res.out)


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
    # Never clobber something of his -- and never by ASKING first, either.
    # ``dest.exists()`` followed by a copy is a check and then a write, and
    # the gap between them is the whole transfer: a file he saved at that
    # name in those seconds was destroyed, because the write that follows
    # is an scp and scp truncates.  ``O_CREAT|O_EXCL`` is ONE syscall that
    # either creates the name or refuses because somebody has it, so there
    # is no gap at all; the copy then writes over a 0-byte file OF OURS.
    # (The same shape, and the same fix, as land_beside in foldersync.)
    try:
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return SshResult(False, reason="exists")
    except OSError as exc:
        return SshResult(False, reason=("not-there" if exc.errno in
                                        (errno.ENOENT, errno.ENOTDIR)
                                        else "denied"))
    os.close(fd)
    src = f"{scp_path(folder).rstrip('/')}/{remote_name}"
    res = run_copy(conf, str(dest), src, push=False)
    if not res.ok:
        # Our own claim, and whatever half a file scp left in it.  Never
        # anything of his: nothing was at that name a moment ago.
        try:
            os.unlink(dest)
        except OSError:
            log.debug("remote: cannot clear my own claim at %s", dest)
        if res.reason in ("unreachable", "timeout"):
            better = unreachable_reason(conf)
            if better:
                return SshResult(False, out=res.out, err=res.err,
                                 reason=better)
    return res
