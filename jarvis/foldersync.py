"""Two folders on his desktop that are the same folders on HPCOMPUTER.

    ~/Desktop/Jarvis/Outbox   --push-->  /C:/Users/h2pey/Desktop/Jarvis/Inbox
    ~/Desktop/Jarvis/Inbox    <--pull--  /C:/Users/h2pey/Desktop/Jarvis/Outbox

He asked for this because RustDesk -- which now carries the SCREEN -- does
not support dragging a file onto a remote desktop and never will; its answer
is a separate two-pane transfer window, which is not the feel he asked for.
Dropping a file in a folder is.

WHAT THIS IS NOT.  It is not a second transport.  ``jarvis/tools/remote.py``
already owns ssh, scp and sftp for this box, it is proven in both directions
(a real 193-byte file, round trip, byte identical, 2026-09-05), and every
byte here moves through :func:`remote.run_copy` and :func:`remote.run_sftp`.
What this module adds is the part a one-shot voice command never needed: a
LEDGER, a quiescence rule, verification before anything of his is moved, and
a way to see what happened.

------------------------------------------------------------------ the shape

**A systemd --user SERVICE, not a thread in the app and not a timer.**

* Not a thread in Jarvis: Jarvis is a Tk app that needs the desktop session,
  and a folder that only syncs while the assistant happens to be up is a
  folder he cannot trust.  This runs whether Jarvis is running or not, and
  (with lingering already on for this account) from boot, before he logs in.
* Not a timer either, and this is the closer call.  A timer restarts a
  process it cannot wedge, which is the honest argument for one.  But the
  local half of this job wants a ~2 s cadence to feel like a folder, and the
  remote half wants a ~30 s cadence so a sleeping Windows box is not handed
  120 ssh handshakes an hour -- one timer cannot be both, and two timers is
  more moving parts than one loop.  So: one service, with the wedge risk
  answered where it actually lives -- every transfer is a bounded
  ``Popen`` + ``communicate(timeout)`` + kill in remote.py, ``Restart=always``
  covers a crash, and nothing is held in memory that the on-disk ledger does
  not already hold.

------------------------------------------------------- the five hard parts

1. **A file still being copied in.**  A 2 GB drag appears instantly and
   grows; sending it half-written is the worst bug this can have.  The rule
   is :func:`is_quiescent`: the mtime must be at least ``min_quiet_s``
   (4 s) old, AND ``(size, mtime_ns)`` must be unchanged across
   ``stable_samples`` (3) observations ``stable_interval_s`` (1 s) apart.
   The sampling is SKIPPED when the mtime alone already proves more
   stillness than the sampling window would observe -- a file that has sat
   there since the last pass costs no sleep at all.

   The residual risk, stated rather than hidden: a writer that stalls for
   longer than the whole window mid-file is indistinguishable, by size and
   mtime, from a finished one.  Nothing short of an open-fd scan of /proc
   sees that, and a stall that long is a hung copy, not a slow one.

2. **Verify before moving.**  An exit code is not evidence.  After the copy,
   the far side is LISTED again and the landed name must be there at the
   byte size we sent.  Only then does the local original MOVE to
   ``~/Desktop/Jarvis/Sent/`` -- visible, recoverable, and the Outbox
   visibly empties.  Nothing here ever deletes a file of his, in either
   direction.  On a size mismatch the file STAYS in the Outbox and a
   ``<name>.jarvis-cannot-send.txt`` note beside it says both numbers.

3. **Never re-fetch.**  MEASURED: the ``jarvis`` account cannot create in
   the Windows Outbox (``dest open ".../Outbox/.writetest": Permission
   denied``), so the puller can never clear what it has taken -- and that is
   the safer design anyway, because it means nothing here can delete
   anything of his on that machine.  Instead a local ledger keys each
   remote file on ``name|size|listed-stamp``.  Same name with new contents
   -> new size or new stamp -> it comes across again, landing beside the
   old one as ``name (2).ext``.  Byte-identical and put back unchanged (a
   Windows Explorer move preserves mtime) -> same key -> never fetched
   twice, so it cannot loop.

4. **Nothing recurses.**  The puller writes into ``Inbox`` and the pusher
   reads only ``Outbox``; they are different directories and
   :func:`preflight` refuses to run if config ever makes them the same, or
   nests ``Sent`` inside the Outbox.  Our own notes and in-flight part
   files are skipped by name as well, belt and braces.

5. **The link being down.**  Backoff 30 s -> 60 -> 120 -> 240 -> 300 and
   hold, reset on the first success; one WARNING on the way down and one
   INFO on the way back up, never one per pass.  Nothing is lost -- his
   files sit in the Outbox -- and ``~/Desktop/Jarvis/status.txt`` says so
   in words, beside the folders, so the answer to "is it working?" does not
   require a terminal.

------------------------------------------------------------------- privacy

This moves his documents, so nothing here reads a byte of one.  Names, byte
sizes and outcomes go to the record; content never does.  The verification
compares SIZES, not contents, for the same reason.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import stat as statmod
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger
from jarvis.tools import filepick, remote

log = get_logger("foldersync")

# Our own furniture inside his folders.  Both are skipped by the scanner, so
# neither can be picked up and sent -- the first half of "nothing recurses".
NOTE_SUFFIX = ".jarvis-cannot-send.txt"
PART_PREFIX = ".jarvis-part-"

# Names that are somebody else's half-finished download or lock file.  A
# browser writes foo.pdf.crdownload and renames; Office leaves ~$doc.docx.
SKIP_SUFFIXES = (".part", ".partial", ".crdownload", ".download", ".tmp",
                 ".swp", NOTE_SUFFIX)
SKIP_PREFIXES = (".", "~$", PART_PREFIX)

MAX_COPIES = 50          # "name (2)" .. "name (50)", then refuse
MAX_ATTEMPTS = 5         # tries at ONE file before it is parked
RETRY_AFTER_S = 3600.0   # ...and how long it is parked for
HISTORY_LINES = 500      # the on-disk record, bounded
LEDGER_CAP = 4000        # remembered remote files, bounded
# A remote file the far side has stopped listing is forgotten after this.
# Nothing live is ever evicted by the cap: remote.LISTING_CAP bounds a
# listing at 400 entries and every one of them is refreshed on every pass,
# so the 4000 slots cannot be filled by files that still exist.
FORGET_AFTER_S = 30 * 86400.0

# Windows keeps these for devices, with or without an extension, in any
# case.  CON.txt is not a file there -- it is the console.
_RESERVED = {"con", "prn", "aux", "nul"} | \
            {f"com{i}" for i in range(1, 10)} | \
            {f"lpt{i}" for i in range(1, 10)}
# 260 including the terminating NUL, so 259 usable characters.
WINDOWS_MAX_PATH = 259

# Bytes per second assumed when sizing a transfer budget.  Deliberately
# pessimistic: the cost of guessing low is a longer timeout on a file that
# finishes early, and the cost of guessing high is a 120-second budget
# cutting a large file in half.
ASSUMED_BYTES_PER_S = 1024 * 1024


# --------------------------------------------------------------- the pieces
@dataclass(frozen=True)
class Paths:
    outbox: Path
    inbox: Path
    sent: Path
    status: Path


@dataclass
class SyncConfig:
    enabled: bool = False
    paths: Optional[Paths] = None
    pull_from: str = "outbox"        # which remote.pull_dirs key to pull from
    scan_interval_s: float = 2.0     # local only; costs nothing
    remote_interval_s: float = 30.0  # one sftp listing; costs a handshake
    max_backoff_s: float = 300.0
    stable_samples: int = 3
    stable_interval_s: float = 1.0
    min_quiet_s: float = 4.0
    max_mb: float = 0.0              # 0 -> inherit remote.max_mb


@dataclass(frozen=True)
class Entry:
    """One file on the far side, as its own listing described it."""
    name: str
    size: int
    stamp: str                       # the listing's date text, verbatim

    @property
    def key(self) -> str:
        """The ledger identity.  A FINGERPRINT, not a timestamp -- the sftp
        listing gives "Sep  3 12:21" with no year and no seconds, and
        guessing an epoch out of that is a bug waiting for New Year.  It
        changes whenever the file is written, which is all the ledger needs.

        Known and bounded: sftp flips the text from "Sep  3 12:21" to
        "Sep  3  2025" once a file is about six months old, which would look
        like a change and fetch it once more (landing as "name (2).ext").
        A file that has sat in an Outbox for six months is not the case this
        serves, and the cost is one duplicate, never a loss and never a loop.
        """
        return f"{self.name}|{self.size}|{self.stamp}"


@dataclass
class Event:
    when: float
    direction: str                   # "push" | "pull" | "link"
    name: str
    size: int
    outcome: str
    detail: str = ""

    def row(self) -> dict:
        return {"t": round(self.when, 1), "dir": self.direction,
                "name": self.name, "size": self.size,
                "outcome": self.outcome, "detail": self.detail}


# Outcomes that describe the LINK rather than a file: kept out of the record,
# which is a record of files.  The status file carries the link's state.
TRANSIENT = {"link-down"}

WHY = {
    "name-charset": "its name has characters Windows will not take (a colon, "
                    "a slash, a quote, an emoji or an accent)",
    "name-trailing": "its name ends in a dot or a space, which Windows drops",
    "name-reserved": "Windows keeps that name for a device (CON, PRN, AUX, "
                     "NUL, COM1-9, LPT1-9)",
    "name-too-long": "the path it would have on HPCOMPUTER is longer than "
                     "Windows accepts",
    "too-big": "it is too big for the limit I am given",
    "not-a-file": "it is a folder, or not an ordinary file -- I send files, "
                  "one at a time, and never a whole folder",
    "unreadable": "I am not allowed to read it",
    "outside": "it is not inside the folders I am allowed to send from",
    "too-many-copies": "there are already fifty files by that name over there",
    "verify-failed": "it arrived the wrong size, so I have not moved yours",
}


# ------------------------------------------------------------------- names
def windows_name_problem(name: str) -> str:
    """"" if Windows and the transport will both take this basename, else
    the reason.  The charset rule is deliberately the TRANSPORT's own
    (:data:`remote.SAFE_REMOTE_NAME_RX`) rather than a looser one of our
    own: a name accepted here and refused there would be reported late, with
    a worse message, after a socket had opened."""
    base = os.path.basename(name or "")
    if not base or base != name:
        return "name-charset"
    if base != base.rstrip(". "):
        return "name-trailing"
    if base.split(".")[0].lower() in _RESERVED:
        return "name-reserved"
    if not remote.SAFE_REMOTE_NAME_RX.match(base):
        return "name-charset"
    return ""


def path_problem(folder: str, name: str) -> str:
    """The name PLUS where it would land.  Separate from the name check
    because the folder half is config, and a config that is too deep is his
    to shorten, not the file's fault."""
    bad = windows_name_problem(name)
    if bad:
        return bad
    if len(f"{(folder or '').rstrip('/')}/{name}") > WINDOWS_MAX_PATH:
        return "name-too-long"
    return ""


def dedupe_name(name: str, taken) -> str:
    """``name``, or the first free ``name (2).ext``; "" past
    :data:`MAX_COPIES`.

    Both directions land through this, and neither ever overwrites: on the
    far side that would be modifying a file of his on HPCOMPUTER, and on
    this side it would be losing one of two things he was sent.  A suffix
    keeps both and makes the newer one obvious, which is what a downloads
    folder does and what he would expect.
    """
    if name not in taken:
        return name
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    for n in range(2, MAX_COPIES + 2):
        candidate = f"{stem} ({n})" + (f".{ext}" if dot else "")
        if candidate not in taken:
            return candidate
    return ""


# -------------------------------------------------------------- quiescence
def stat_key(path: Path) -> str:
    """``size|mtime_ns`` for a regular file, "" for anything else or gone."""
    try:
        st = path.stat()
    except OSError:
        return ""
    if not statmod.S_ISREG(st.st_mode):
        return ""
    return f"{st.st_size}|{st.st_mtime_ns}"


def is_quiescent(path: Path, *, samples: int = 3, interval_s: float = 1.0,
                 min_quiet_s: float = 4.0, sleep=time.sleep,
                 now=time.time) -> bool:
    """Has nothing written to ``path`` for long enough to send it?

    Two gates, and the first one does almost all the work.  A file being
    written has a mtime of NOW, so ``age < min_quiet_s`` rejects a growing
    file immediately and without sleeping -- which is what a 2 GB drag looks
    like for its whole duration.  The sampling loop is the backstop for a
    writer that pauses between blocks: ``(size, mtime_ns)`` must survive
    ``samples`` looks ``interval_s`` apart.

    And the shortcut that keeps the scan cheap: when the mtime is ALREADY
    older than the whole sampling window would be, the samples can only
    confirm what it says, so they are skipped.  A file that has sat in the
    Outbox since the last pass therefore costs one ``stat``.
    """
    key = stat_key(path)
    if not key:
        return False
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    age = now() - mtime
    if age < min_quiet_s:
        return False
    if age >= min_quiet_s + max(0, samples - 1) * interval_s:
        return True
    for _ in range(max(0, samples - 1)):
        sleep(interval_s)
        if stat_key(path) != key:
            return False
    return True


# ------------------------------------------------------------------ config
def read_config(cfg) -> SyncConfig:
    """The ``foldersync`` section, defensively -- same convention as
    :func:`remote.read_config`: a malformed value leaves the shipped
    default rather than half-configuring a thing that moves his files."""
    def get(key, default):
        try:
            val = cfg.get(f"foldersync.{key}", default)
        except Exception:                      # noqa: BLE001 - config only
            log.debug("foldersync: cannot read %s", key, exc_info=True)
            return default
        return default if val is None else val

    def num(key, default, low, high):
        try:
            return max(low, min(high, float(get(key, default))))
        except (TypeError, ValueError):
            return default

    def folder(key, default):
        return Path(os.path.expanduser(str(get(key, default) or default)))

    return SyncConfig(
        enabled=bool(get("enabled", False)),
        paths=Paths(outbox=folder("outbox", "~/Desktop/Jarvis/Outbox"),
                    inbox=folder("inbox", "~/Desktop/Jarvis/Inbox"),
                    sent=folder("sent", "~/Desktop/Jarvis/Sent"),
                    status=folder("status", "~/Desktop/Jarvis/status.txt")),
        pull_from=str(get("pull_from", "outbox") or "outbox"),
        scan_interval_s=num("scan_interval_s", 2.0, 0.5, 60.0),
        remote_interval_s=num("remote_interval_s", 30.0, 5.0, 3600.0),
        max_backoff_s=num("max_backoff_s", 300.0, 30.0, 86400.0),
        stable_samples=int(num("stable_samples", 3, 2, 10)),
        stable_interval_s=num("stable_interval_s", 1.0, 0.1, 10.0),
        min_quiet_s=num("min_quiet_s", 4.0, 0.0, 600.0),
        max_mb=num("max_mb", 0.0, 0.0, 100000.0),
    )


def effective_max_mb(rconf: remote.RemoteConfig, sconf: SyncConfig) -> float:
    """The cap this lane uses.  ``foldersync.max_mb`` of 0 inherits
    ``remote.max_mb``, because the voice lane's 100 MB is a sensible bound
    on "send that file" inside a spoken turn and not necessarily on a
    folder he drags into deliberately -- but raising it is his decision,
    made once, in one place."""
    return float(sconf.max_mb) if sconf.max_mb else float(rconf.max_mb)


def transfer_budget(rconf: remote.RemoteConfig, size_bytes: int) -> float:
    """Seconds to allow ONE copy of this size.

    ``remote.transfer_timeout_s`` is 120 s, which is right for a spoken
    "send that file" and wrong the moment the cap is raised: a 2 GB file
    cannot cross a LAN in 120 s, so the copy would be killed half-written
    every single time and the file would never leave the Outbox.  The floor
    stays 120 s; above that the budget scales with size, and it is still
    capped by remote.MAX_TRANSFER_S so a wrong number cannot hang a pass
    forever.
    """
    need = max(float(rconf.transfer_timeout_s),
               float(size_bytes) / ASSUMED_BYTES_PER_S)
    return min(need, remote.MAX_TRANSFER_S)


def preflight(rconf: remote.RemoteConfig, paths: Paths) -> list:
    """Everything that would make this unsafe to start, in his words.

    The local-roots check is the one to read twice.  ``remote.push`` refuses
    any source outside ``remote.local_roots`` with reason 'outside', and
    ``~/Desktop/Jarvis/Outbox`` is inside the shipped ``~/Desktop`` -- but
    this CHECKS rather than assumes, and if it ever fails the answer is to
    say so here, never to widen the roots quietly.  Widening them is a
    decision about which of his folders a spoken sentence can reach, and it
    is not this module's to make.
    """
    out = []
    try:
        outbox = paths.outbox.resolve()
        inbox = paths.inbox.resolve()
        sent = paths.sent.resolve()
    except (OSError, RuntimeError):
        return ["I cannot resolve the sync folders."]
    if outbox == inbox:
        out.append(f"the outbox and the inbox are the same folder "
                   f"({outbox}); that would copy every arrival straight "
                   f"back to HPCOMPUTER")
    for name, other in (("inbox", inbox), ("sent", sent)):
        if other == outbox or outbox in other.parents:
            out.append(f"the {name} folder is inside the outbox ({other}); "
                       f"everything I put there would be sent again")
    if inbox != outbox and inbox in outbox.parents:
        out.append(f"the outbox is inside the inbox ({outbox})")
    roots = filepick.expand_roots(rconf.local_roots)
    probe = outbox / ".probe"
    if not any(_inside(probe, r) for r in roots):
        out.append(
            f"the outbox {outbox} is outside remote.local_roots "
            f"({', '.join(str(r) for r in roots) or 'none of them exist'}), "
            f"so every file in it would be refused. Add the folder's root to "
            f"remote.local_roots in ~/.config/jarvis/assistant.json -- I have "
            f"not widened it myself.")
    return out


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


# ------------------------------------------------------------- the listing
_SFTP_FIELDS = 8          # mode links user group size mon day time  name


def parse_sftp_entries(out: str) -> list:
    """Name, SIZE and date text out of an ``sftp ls -ln``.

    ``remote.sftp_listing`` reads the same output and keeps only the names,
    which is all a spoken "fetch me the budget one" needs.  This lane needs
    the size (it is the verification) and the date text (it is the ledger
    key), so it reads the same measured format one field wider rather than
    opening a second kind of session to ask again.

    Format measured on this box 2026-09-03 (``sftp -q -b - -D
    /usr/lib/openssh/sftp-server``; no socket):

        sftp> ls -ln "jarvis-outbox"
        -rw-rw-r--    ? hunterp  hunterp   1 Sep  3 12:21 jarvis-outbox/a.txt
        drwxrwxr-x    ? hunterp  hunterp 4096 Sep  3 12:21 jarvis-outbox/sub dir
    """
    rows = []
    for raw in (out or "").splitlines()[:remote.LISTING_CAP + 1]:
        line = raw.rstrip()
        if not line or line.startswith("sftp>") or line[0] == "d":
            continue
        fields = line.split(None, _SFTP_FIELDS)
        if len(fields) <= _SFTP_FIELDS:
            continue
        try:
            size = int(fields[4])
        except (TypeError, ValueError):
            continue
        name = fields[_SFTP_FIELDS].rsplit("/", 1)[-1]
        if not name:
            continue
        rows.append(Entry(name, size, " ".join(fields[5:8])))
    return rows


class SshTransport:
    """The real far side, over the transport that is already proven.

    Three calls and no more: LIST a folder, SEND one file into the remote
    inbox, FETCH one file out of a remote folder.  There is deliberately no
    delete and no remote write outside ``remote.inbox`` -- the measured
    permission says the jarvis account cannot create in the Windows Outbox
    anyway, and nothing here should ever want to.
    """

    def __init__(self, conf: remote.RemoteConfig):
        self.conf = conf

    # -- where a push may land -----------------------------------------
    def target(self, name: str) -> str:
        """The remote path for a pushed basename, or "" if this module will
        not write that name.  ``name`` must already BE a basename: a caller
        that passes ``../x`` or ``a/b`` gets "" rather than a cleaned-up
        path, because silently rewriting a path is how one escapes."""
        if not name or name != os.path.basename(name):
            return ""
        if windows_name_problem(name):
            return ""
        return remote.inbox_target(self.conf, name)

    def listing(self, key: str) -> tuple:
        why = remote.missing_reason(self.conf)
        if why:
            return [], why
        folder = remote.remote_dir(self.conf, key)
        if not folder:
            return [], "not-there"
        path = remote.scp_path(folder)
        if remote._SFTP_UNQUOTABLE_RX.search(path):
            log.warning("foldersync: refusing to list a folder I will not "
                        "put in an sftp batch line")
            return [], "not-there"
        res = remote.run_sftp(self.conf, f'ls -ln "{path}"')
        if not res.ok:
            reason = res.reason
            if reason in ("unreachable", "timeout"):
                reason = remote.unreachable_reason(self.conf) or reason
            return [], reason
        return parse_sftp_entries(res.out), ""

    def send(self, local: Path, name: str, key: str = "inbox") -> str:
        why = remote.missing_reason(self.conf)
        if why:
            return why
        if key != "inbox":
            return "denied"          # nothing writes anywhere else, ever
        dest = self.target(name)
        if not dest:
            return "odd-name"
        try:
            size = local.stat().st_size
        except OSError:
            return "not-found"
        conf = replace(self.conf,
                       transfer_timeout_s=transfer_budget(self.conf, size))
        res = remote.run_copy(conf, str(local), dest, push=True)
        if res.ok:
            return ""
        if res.reason in ("unreachable", "timeout"):
            return remote.unreachable_reason(self.conf) or res.reason
        return res.reason or "failed"

    def fetch(self, key: str, name: str, dest: Path) -> str:
        why = remote.missing_reason(self.conf)
        if why:
            return why
        folder = remote.remote_dir(self.conf, key)
        if not folder or not remote.SAFE_REMOTE_NAME_RX.match(name or ""):
            return "not-there"
        src = f"{remote.scp_path(folder).rstrip('/')}/{name}"
        res = remote.run_copy(self.conf, str(dest), src, push=False)
        if res.ok:
            return ""
        if res.reason in ("unreachable", "timeout"):
            return remote.unreachable_reason(self.conf) or res.reason
        return res.reason or "failed"


# ------------------------------------------------------------------ ledger
class Ledger:
    """What has already come across, and what has failed how often.

    It exists because of a MEASURED permission: the jarvis account cannot
    create in the Windows Outbox, so the puller cannot mark a file as taken
    on the far side and must remember here instead.  It lives under
    ``~/.local/state/jarvis`` and not ``/tmp`` -- /tmp is wiped at every boot
    on this box, and a ledger that forgets at boot re-fetches everything.
    """

    def __init__(self, path: Path, cap: int = LEDGER_CAP):
        self.path = Path(path)
        self.cap = max(1, int(cap))
        self.pulled: dict = {}
        self.fails: dict = {}
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self.pulled = {k: v for k, v in (data.get("pulled") or {}).items()
                       if isinstance(v, dict)}
        self.fails = {k: v for k, v in (data.get("fails") or {}).items()
                      if isinstance(v, dict)}

    def save(self) -> None:
        self._trim()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"version": 1, "pulled": self.pulled, "fails": self.fails}))
            os.replace(tmp, self.path)
        except OSError:
            log.warning("foldersync: cannot write the ledger at %s",
                        self.path, exc_info=True)

    # -- what has come across -------------------------------------------
    def has(self, key: str) -> bool:
        return key in self.pulled

    def mark_pulled(self, key: str, landed: str = "",
                    now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        row = self.pulled.get(key) or {"first": now}
        row.update({"last_seen": now, "landed": landed})
        self.pulled[key] = row
        self._trim()

    def sweep(self, live_keys, now: Optional[float] = None) -> None:
        """Refresh everything still on the far side, forget what has been
        gone a long time.  The refresh is what makes the cap safe: a file
        still sitting in the Windows Outbox is touched on every pass, so it
        is never the oldest thing in the ledger and can never be the entry
        the cap evicts."""
        now = time.time() if now is None else now
        for key in live_keys:
            row = self.pulled.get(key)
            if row is not None:
                row["last_seen"] = now
        for key in [k for k, r in self.pulled.items()
                    if now - float(r.get("last_seen") or 0) > FORGET_AFTER_S]:
            self.pulled.pop(key, None)

    def _trim(self) -> None:
        if len(self.pulled) > self.cap:
            keep = sorted(self.pulled.items(),
                          key=lambda kv: float(kv[1].get("last_seen") or 0),
                          reverse=True)[:self.cap]
            self.pulled = dict(keep)
        if len(self.fails) > self.cap:
            keep = sorted(self.fails.items(),
                          key=lambda kv: float(kv[1].get("last") or 0),
                          reverse=True)[:self.cap]
            self.fails = dict(keep)

    # -- how often one thing has failed ---------------------------------
    def blocked(self, key: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        row = self.fails.get(key)
        if not row:
            return False
        if int(row.get("n") or 0) < MAX_ATTEMPTS:
            return False
        if now >= float(row.get("next") or 0):
            self.fails.pop(key, None)          # the parking period is over
            return False
        return True

    def bump(self, key: str, reason: str = "",
             now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        row = self.fails.get(key) or {"n": 0}
        row["n"] = int(row.get("n") or 0) + 1
        row["last"] = now
        row["why"] = reason
        if row["n"] >= MAX_ATTEMPTS:
            row["next"] = now + RETRY_AFTER_S
        self.fails[key] = row
        self._trim()
        return row["n"]

    def clear(self, key: str) -> None:
        self.fails.pop(key, None)


# ------------------------------------------------------------ one at a time
@contextmanager
def single_instance(path: Path):
    """True inside the block if this process took the lock, False if
    another holds it.  Two syncers running at once would double-push and
    race on the ledger; ``--once`` takes the same lock as the service."""
    fh = None
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fh = open(path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    except OSError:
        log.warning("foldersync: cannot take the lock at %s", path)
        yield False
    finally:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass


# ------------------------------------------------------------------ syncer
class Syncer:
    """One pass of each direction, plus the loop that repeats them."""

    def __init__(self, rconf: remote.RemoteConfig, conf: SyncConfig,
                 transport, ledger: Optional[Ledger] = None):
        self.rconf = rconf
        self.conf = conf
        self.paths = conf.paths
        self.transport = transport
        state = Path(getattr(PATHS, "STATE_DIR"))
        self.ledger = ledger or Ledger(state / "foldersync.json")
        self.history_path = state / "foldersync-history.jsonl"
        self.interval_s = conf.remote_interval_s
        self.status_clock = self._clock
        self._down_reason = ""
        self._down_since = 0.0
        self._logged_down = False
        self._pass_down = False
        self._last_ok = 0.0
        self._last_pass = 0.0
        self._recent: list = []
        self._status_text = ""
        self._skipped_names = False

    # -- small helpers ---------------------------------------------------
    @staticmethod
    def _clock() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def max_mb(self) -> float:
        return effective_max_mb(self.rconf, self.conf)

    def record(self, event: Event) -> None:
        """Names, sizes and outcomes.  Never a byte of what moved."""
        self._recent = (self._recent + [event])[-20:]
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True,
                                           mode=0o700)
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.row()) + "\n")
            lines = self.history_path.read_text(
                encoding="utf-8").splitlines()
            if len(lines) > HISTORY_LINES:
                self.history_path.write_text(
                    "\n".join(lines[-HISTORY_LINES:]) + "\n", encoding="utf-8")
        except OSError:
            log.debug("foldersync: cannot write the record", exc_info=True)

    def note(self, target: Path, reason: str, extra: str = "") -> None:
        """A plain-English note beside the file that could not go.  Visible
        in his file manager, which is where he will be looking; it is
        removed again the moment the file is fixed or taken away."""
        text = (f"Jarvis could not send {target.name}.\n\n"
                f"Why: {WHY.get(reason, reason)}.\n")
        if extra:
            text += f"\n{extra}\n"
        text += ("\nYour file has not been moved, changed or deleted. Fix the\n"
                 "reason above (usually: rename it) and I will send it on the\n"
                 "next pass. This note disappears by itself when I can.\n")
        try:
            (target.parent / (target.name + NOTE_SUFFIX)).write_text(
                text, encoding="utf-8")
        except OSError:
            log.debug("foldersync: cannot write a note", exc_info=True)

    def clear_note(self, target: Path) -> None:
        try:
            (target.parent / (target.name + NOTE_SUFFIX)).unlink()
        except OSError:
            pass

    def _sweep_notes(self) -> None:
        """Our own notes whose file has gone are ours to remove -- an Outbox
        littered with explanations of files he already dealt with is worse
        than no explanation."""
        try:
            entries = list(self.paths.outbox.iterdir())
        except OSError:
            return
        for p in entries:
            if not p.name.endswith(NOTE_SUFFIX):
                continue
            base = self.paths.outbox / p.name[:-len(NOTE_SUFFIX)]
            if not base.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    # -- what the scanner will look at -----------------------------------
    def _candidates(self) -> list:
        out = []
        try:
            entries = sorted(self.paths.outbox.iterdir(), key=lambda p: p.name)
        except OSError:
            return out
        for p in entries:
            name = p.name
            if name.startswith(SKIP_PREFIXES) or name.endswith(SKIP_SUFFIXES):
                continue
            out.append(p)
        return out

    # ------------------------------------------------------------- pushing
    def push_once(self, now: Optional[float] = None) -> list:
        now = time.time() if now is None else now
        events: list = []
        self._sweep_notes()
        roots = filepick.expand_roots(self.rconf.local_roots)
        ready: list = []
        for p in self._candidates():
            bad = filepick.check_file(p, roots, self.max_mb)
            if not bad:
                bad = path_problem(remote.remote_dir(self.rconf, "inbox"),
                                   p.name)
            if bad:
                if bad == "not-found":
                    continue                    # it went while we looked
                self.note(p, bad, self._size_note(p, bad))
                events.append(Event(now, "push", p.name, self._size(p), bad))
                continue
            if not is_quiescent(p, samples=self.conf.stable_samples,
                                interval_s=self.conf.stable_interval_s,
                                min_quiet_s=self.conf.min_quiet_s):
                log.debug("foldersync: %s is still changing; leaving it",
                          p.name)
                continue
            if self.ledger.blocked(f"push:{p.name}|{stat_key(p)}", now):
                continue
            ready.append(p)
        if not ready:
            for e in events:
                self.record(e)
            return events

        entries, why = self.transport.listing("inbox")
        if why:
            events.append(self._link_event(now, why, len(ready)))
            for e in events:
                if e.direction != "link":
                    self.record(e)
            return events
        self._mark_up(now)

        taken = {e.name for e in entries}
        sent: list = []
        for p in ready:
            landed = dedupe_name(p.name, taken)
            if not landed:
                self.note(p, "too-many-copies")
                events.append(Event(now, "push", p.name, self._size(p),
                                    "too-many-copies"))
                continue
            size = self._size(p)
            reason = self.transport.send(p, landed)
            if reason:
                if reason in remote.FAIL_LINES and reason not in (
                        "denied", "no-space", "odd-name", "exists"):
                    events.append(self._link_event(now, reason, len(ready)))
                    break
                self.ledger.bump(f"push:{p.name}|{stat_key(p)}", reason, now)
                self.note(p, reason)
                events.append(Event(now, "push", p.name, size, reason))
                continue
            taken.add(landed)
            sent.append((p, landed, size))

        if sent:
            events += self._verify_and_move(sent, now)
        for e in events:
            if e.direction != "link":
                self.record(e)
        self.ledger.save()
        return events

    def _verify_and_move(self, sent: list, now: float) -> list:
        """The far side is LISTED again and every landed name must be there
        at the byte size we sent.  Only then does his original move.

        On a mismatch nothing is deleted anywhere: the partial stays on
        HPCOMPUTER (it is in the folder we are allowed to write, under a
        name that did not exist before this pass, so it is ours -- but
        removing it would still be a write we do not need to make), his file
        stays in the Outbox, and the note says both numbers.  The attempt is
        counted, and after :data:`MAX_ATTEMPTS` that file is parked so a
        repeatedly-failing copy cannot litter the far side.
        """
        events = []
        entries, why = self.transport.listing("inbox")
        if why:
            for p, _landed, size in sent:
                events.append(Event(now, "push", p.name, size, "unverified",
                                    why))
            return events
        sizes = {e.name: e.size for e in entries}
        for p, landed, size in sent:
            there = sizes.get(landed)
            if there != size:
                key = f"push:{p.name}|{stat_key(p)}"
                n = self.ledger.bump(key, "verify-failed", now)
                self.note(p, "verify-failed",
                          f"I copied {size} bytes but HPCOMPUTER reports "
                          f"{there if there is not None else 'no such file'}. "
                          f"Attempt {n} of {MAX_ATTEMPTS}.")
                events.append(Event(now, "push", p.name, size,
                                    "verify-failed", str(there)))
                continue
            key = f"push:{p.name}|{stat_key(p)}"    # BEFORE the move: after
            moved = self._move_to_sent(p)           # it, stat_key() is ""
            self.ledger.clear(key)
            self.clear_note(p)
            events.append(Event(now, "push", p.name, size, "sent", landed))
            if moved != p.name:
                log.info("foldersync: kept %s as %s in Sent", p.name, moved)
        return events

    def _move_to_sent(self, p: Path) -> str:
        """His original, out of the Outbox and into Sent -- MOVED, never
        deleted, so the Outbox visibly empties and the file is still one
        double-click away."""
        self.paths.sent.mkdir(parents=True, exist_ok=True)
        try:
            taken = {q.name for q in self.paths.sent.iterdir()}
        except OSError:
            taken = set()
        name = dedupe_name(p.name, taken) or f"{p.name}.{int(time.time())}"
        dest = self.paths.sent / name
        try:
            os.replace(p, dest)               # same filesystem: atomic
        except OSError:
            shutil.move(str(p), str(dest))    # different one: still a move
        return name

    # ------------------------------------------------------------- pulling
    def pull_once(self, now: Optional[float] = None) -> list:
        now = time.time() if now is None else now
        if self._pass_down:
            return []                          # asked once a pass, not twice
        events: list = []
        entries, why = self.transport.listing(self.conf.pull_from)
        if why:
            return [self._link_event(now, why, 0)]
        self._mark_up(now)

        usable = []
        for e in entries:
            if remote.SAFE_REMOTE_NAME_RX.match(e.name):
                usable.append(e)
            elif not self._skipped_names:
                self._skipped_names = True
                log.info("foldersync: skipping a name in the HPCOMPUTER "
                         "outbox that I will not put in a path")
        self.ledger.sweep({e.key for e in usable}, now)

        try:
            self.paths.inbox.mkdir(parents=True, exist_ok=True)
            taken = {p.name for p in self.paths.inbox.iterdir()}
        except OSError:
            log.warning("foldersync: cannot read %s", self.paths.inbox)
            return events

        for entry in usable:
            if self.ledger.has(entry.key):
                continue
            if self.ledger.blocked(f"pull:{entry.key}", now):
                continue
            landed = dedupe_name(entry.name, taken)
            if not landed:
                events.append(Event(now, "pull", entry.name, entry.size,
                                    "too-many-copies"))
                continue
            part = self.paths.inbox / (PART_PREFIX + entry.name)
            reason = self.transport.fetch(self.conf.pull_from, entry.name,
                                          part)
            if reason:
                self._drop(part)
                if reason in ("unreachable", "timeout", "asleep",
                              "off-tailnet", "no-ssh"):
                    events.append(self._link_event(now, reason, 0))
                    break
                self.ledger.bump(f"pull:{entry.key}", reason, now)
                events.append(Event(now, "pull", entry.name, entry.size,
                                    reason))
                continue
            got = self._size(part)
            if got != entry.size:
                self._drop(part)
                n = self.ledger.bump(f"pull:{entry.key}", "verify-failed", now)
                log.warning("foldersync: %s arrived %d bytes, not %d "
                            "(attempt %d of %d)", entry.name, got, entry.size,
                            n, MAX_ATTEMPTS)
                events.append(Event(now, "pull", entry.name, entry.size,
                                    "verify-failed", str(got)))
                continue
            try:
                os.replace(part, self.paths.inbox / landed)
            except OSError:
                self._drop(part)
                self.ledger.bump(f"pull:{entry.key}", "denied", now)
                events.append(Event(now, "pull", entry.name, entry.size,
                                    "denied"))
                continue
            taken.add(landed)
            self.ledger.clear(f"pull:{entry.key}")
            self.ledger.mark_pulled(entry.key, landed, now)
            events.append(Event(now, "pull", entry.name, entry.size,
                                "received", landed))
        for e in events:
            if e.direction != "link":
                self.record(e)
        self.ledger.save()
        return events

    # ------------------------------------------------------------- the link
    def _link_event(self, now: float, why: str, waiting: int) -> Event:
        """Down, once.  The WARNING and the record entry are written on the
        transition only: a box asleep overnight must not fill either."""
        first = not self._down_reason
        if first:
            self._down_since = now
        self._down_reason = why
        self._pass_down = True
        event = Event(now, "link", "", 0, "link-down", why)
        if first:
            self.record(event)
        if not self._logged_down:
            self._logged_down = True
            log.warning("foldersync: %s (%d file(s) waiting); backing off to "
                        "%.0fs and staying quiet until it answers",
                        remote.fail_line(self.rconf, why), waiting,
                        min(self.interval_s * 2, self.conf.max_backoff_s))
        self.interval_s = min(self.interval_s * 2, self.conf.max_backoff_s)
        return event

    def _mark_up(self, now: float) -> None:
        if self._logged_down:
            log.info("foldersync: HPCOMPUTER is answering again")
        self._logged_down = False
        self._down_reason = ""
        self._down_since = 0.0
        self.interval_s = self.conf.remote_interval_s
        self._last_ok = now

    # -------------------------------------------------------------- a pass
    def run_pass(self, now: Optional[float] = None) -> list:
        now = time.time() if now is None else now
        self._pass_down = False
        events = self.push_once(now)
        events += self.pull_once(now)
        self._last_pass = now
        self.write_status()
        return events

    # ------------------------------------------------------------- status
    def status_text(self) -> str:
        waiting = [p.name for p in self._candidates()]
        lines = [f"Jarvis folder sync  --  {self.rconf.name}",
                 f"checked   {self.status_clock()}"]
        if self._down_reason:
            lines.append(f"link      DOWN: "
                         f"{remote.fail_line(self.rconf, self._down_reason)}")
            lines.append("          Your files are safe where they are; I "
                         "keep trying, more slowly.")
        else:
            lines.append("link      OK")
        lines.append(f"outbox    {len(waiting)} waiting"
                     if waiting else "outbox    empty")
        for name in waiting[:10]:
            lines.append(f"            {name}")
        if len(waiting) > 10:
            lines.append(f"            ... and {len(waiting) - 10} more")
        if self._recent:
            lines.append("")
            lines.append("recent")
            for e in reversed(self._recent[-10:]):
                when = time.strftime("%H:%M:%S", time.localtime(e.when))
                word = {"sent": "sent    ", "received": "received"}.get(
                    e.outcome, e.outcome)
                lines.append(f"  {when}  {word}  {e.name} "
                             f"({e.size} bytes)".rstrip())
        lines.append("")
        lines.append("Sent files are moved to " + str(self.paths.sent) +
                     " -- nothing here is ever deleted.")
        return "\n".join(lines) + "\n"

    def write_status(self) -> None:
        text = self.status_text()
        if text == self._status_text:
            return                              # do not churn his folder
        self._status_text = text
        try:
            self.paths.status.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.paths.status.with_name(
                self.paths.status.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.paths.status)
        except OSError:
            log.debug("foldersync: cannot write the status file", exc_info=True)

    # ---------------------------------------------------------------- loop
    def loop(self, stop=None, sleep=time.sleep) -> None:
        """Local scan often, remote poll rarely, and never both at the same
        cadence: a scan is a ``stat`` of a folder and a poll is an ssh
        handshake to a machine that may be asleep."""
        next_remote = 0.0
        seen = self.outbox_fingerprint()
        while not (stop and stop.is_set()):
            now = time.time()
            due = now >= next_remote
            if not due:
                # A NEW file is worth a pass now; the same refused file
                # sitting there is not.  Without this an unsendable name
                # (CON.txt) would poll a sleeping HPCOMPUTER every 2
                # seconds for as long as it sat in the folder -- the spin
                # this whole design is meant not to have.
                fresh = self.outbox_fingerprint()
                due = fresh != seen
                seen = fresh
            if due:
                try:
                    self.run_pass(time.time())
                except Exception:               # noqa: BLE001 - never exit
                    log.exception("foldersync: pass failed; carrying on")
                seen = self.outbox_fingerprint()
                next_remote = time.time() + self.interval_s
            sleep(self.conf.scan_interval_s)

    def outbox_fingerprint(self) -> frozenset:
        """What is in the Outbox right now, by identity rather than by
        count -- so a file REPLACED between passes counts as new."""
        return frozenset((p.name, stat_key(p)) for p in self._candidates())

    # -------------------------------------------------------------- detail
    @staticmethod
    def _size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    def _size_note(self, p: Path, reason: str) -> str:
        if reason != "too-big":
            return ""
        return (f"It is {self._size(p) / 1048576:.1f} MB and my limit is "
                f"{self.max_mb:.0f} MB. Raise foldersync.max_mb in "
                f"~/.config/jarvis/assistant.json if you want it to go.")

    @staticmethod
    def _drop(p: Path) -> None:
        """Remove OUR OWN in-flight part file.  Never anything of his."""
        try:
            p.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------- wiring
def build(cfg) -> tuple:
    """(Syncer, problems) from a live AssistantConfig."""
    rconf = remote.read_config(cfg)
    sconf = read_config(cfg)
    problems = preflight(rconf, sconf.paths)
    return Syncer(rconf, sconf, SshTransport(rconf)), problems


def main(argv=None) -> int:
    from jarvis.assistant_config import AssistantConfig

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true",
                    help="one pass of each direction, then exit")
    ap.add_argument("--status", action="store_true",
                    help="print what it has done and exit")
    ap.add_argument("--check", action="store_true",
                    help="check the folders and the config, then exit")
    args = ap.parse_args(argv)

    cfg = AssistantConfig.load()
    syncer, problems = build(cfg)

    if args.status:
        print(syncer.status_text())
        try:
            rows = syncer.history_path.read_text().splitlines()[-20:]
        except OSError:
            rows = []
        for raw in rows:
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            when = time.strftime("%m-%d %H:%M", time.localtime(r.get("t", 0)))
            print(f"{when}  {r.get('outcome',''):<14} {r.get('name','')} "
                  f"({r.get('size',0)} bytes) {r.get('detail','')}".rstrip())
        return 0

    for line in problems:
        print(f"problem: {line}", file=sys.stderr)
    if args.check:
        if not syncer.conf.enabled:
            print("foldersync.enabled is false in "
                  "~/.config/jarvis/assistant.json", file=sys.stderr)
        return 1 if problems else 0
    if problems:
        return 2
    if not syncer.conf.enabled:
        print("foldersync.enabled is false; nothing to do.", file=sys.stderr)
        return 0

    lock = Path(getattr(PATHS, "STATE_DIR")) / "foldersync.lock"
    with single_instance(lock) as mine:
        if not mine:
            print("another foldersync is already running.", file=sys.stderr)
            return 0
        if args.once:
            for e in syncer.run_pass():
                print(f"{e.outcome:<14} {e.name} ({e.size} bytes) "
                      f"{e.detail}".rstrip())
            return 0
        log.info("foldersync: watching %s and %s",
                 syncer.paths.outbox, syncer.paths.inbox)
        syncer.loop()
    return 0


if __name__ == "__main__":            # pragma: no cover - the unit's entry
    raise SystemExit(main())
