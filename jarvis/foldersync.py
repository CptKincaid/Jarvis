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

2. **Claim the name, then verify, then move.**  Three steps, and each one
   exists because the step before it is not evidence.

   The CLAIM is the part that took three rounds to get right.  **scp
   truncates** -- re-measured HERE on 2026-09-05, against this box's own
   ``/usr/lib/openssh/sftp-server`` over a pipe rather than against his
   machine: a 22222-byte file at the target name came back 1111 bytes,
   exit 0, nothing on stderr -- in the default SFTP mode, under the legacy
   ``-O`` protocol, and through ``sftp put`` alike, and there is no
   no-clobber flag on any of them.  It is a property of the SFTP protocol
   and of OUR OWN client, which is why it needs no session with HPCOMPUTER
   to establish; tests/test_provenance.py re-derives it every run.  So choosing a free name from a listing and then writing at it
   is a guess about the future, and the guess was wrong for minutes at a
   time: the sends are sequential after ONE listing, so the tenth file in
   a queue was written long after its name was checked.  Instead the bytes
   go at a name of OURS (``jarvis-part-<pid>-<second>-<n>.tmp``) and the
   real name is then taken with an sftp ``rename -l``, which the far side
   REFUSES if anything holds it -- 10 refusals out of 10, both files
   byte-intact, and two sessions racing for one name gave exactly one
   winner in 12 of 12 rounds.  A refusal costs one round trip and the next
   ``(2)``; it never costs a file.  ``rename`` WITHOUT ``-l`` is a
   different call (posix-rename@openssh.com) that silently replaced the
   target 10 times out of 10 -- see :data:`remote.RENAME_FLAG`.

   Then the VERIFY: an exit code is not evidence either, so the far side
   is LISTED again and the name WE ACTUALLY TOOK must be there at the byte
   size we sent.  Only then does the local original MOVE to
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

5. **The link being down -- and the three things that are NOT that.**
   Backoff 30 s -> 60 -> 120 -> 240 -> 300 and hold, reset on the first
   success; one WARNING on the way down and one INFO on the way back up,
   never one per pass.  Nothing is lost -- his files sit in the Outbox --
   and ``~/Desktop/Jarvis/status.txt`` says so in words, beside the
   folders, so the answer to "is it working?" does not require a terminal.

   The three impostors, each of which reached him as "link DOWN, not
   answering since 15:36" while the box was answering every 30 seconds:

   * ONE FILE that will not go.  Answered by :meth:`Syncer._probe_link`:
     only the machine may speak for the machine, so it is asked again at
     the moment of doubt.
   * A missing or mistyped FOLDER.  A listing that comes back "not there"
     is an ANSWER -- proof the box is up -- so it clears the link state,
     never backs off, never stops the other direction, and says in
     status.txt which path was not found and which setting holds it.
     MEASURED over 6 passes: 0 fetches, the interval at 300 s, and a
     control with an empty Outbox pulling fine over the same transport.
   * A SETTING with no socket behind it (no key, no host, switched off).
     It never asked, so it may claim neither that the box is up nor that
     it is down.

-------------------------------------------- every read-then-write, listed

THE TABLE THAT USED TO BE HERE IS GONE, and that is the point.

The shape that has now bitten this lane THREE times is: ask whether a name
is free, and then write at it.  Round 2 swept for it by hand and wrote
"nothing else in the repo has either shape", which was false.  Round 3
wrote a table here as the starting point for the next sweep, and that table
was ALSO wrong -- it was missing the push's own scp at its temp name (an
unbounded window, measured destroying a 99999-byte file) and it did not
mention the pull's part file at all (the whole transfer, measured
destroying another).  Two of the three rounds hand-wrote a map of this
module and got it wrong; a fourth hand-written map is not the answer.

So the table is DERIVED FROM THE SOURCE, mechanically, by an AST census of
every read-then-write pair in this module and in jarvis/tools/remote.py,
and it is pinned by a test that FAILS the moment a pair appears that the
census does not know about:

    tests/test_write_census.py          the census and the pin
    ~/vss_env/bin/python -m tests.write_census    prints today's table

Nothing about the windows themselves is any less important for being
generated -- what changed is only that a new one can no longer be left out
by somebody editing prose.

------------------------------------------------------------------- privacy

This moves his documents, so nothing here reads a byte of one.  Names, byte
sizes and outcomes go to the record; content never does.  The verification
compares SIZES, not contents, for the same reason.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import itertools
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

# ROW 14.  A note's name is HIS filename plus our suffix, so the name alone
# proves nothing -- ``report.pdf.jarvis-cannot-send.txt`` is a name he can
# choose, and it was measured being overwritten and then DELETED by the
# sweep when report.pdf went away.  A name cannot be the identity, so the
# CONTENT is: this first line is written on every note we make, checked
# through an open file descriptor before we truncate one, and required
# before we remove one.
NOTE_MARKER = "Jarvis folder sync note."

# Names that are somebody else's half-finished download or lock file.  A
# browser writes foo.pdf.crdownload and renames; Office leaves ~$doc.docx.
SKIP_SUFFIXES = (".part", ".partial", ".crdownload", ".download", ".tmp",
                 ".swp", NOTE_SUFFIX)
SKIP_PREFIXES = (".", "~$", PART_PREFIX)


def skip_reason(name: str) -> str:
    """WHY this lane is passing over ``name``, in the words of the rule that
    actually fired.  "" when no rule fires.

    Round 5 put a reason in status.txt and hardcoded two of them -- ends
    ".tmp", or starts with a dot -- while ten rules can fire.  MEASURED 3 of
    3 wrong: movie.crdownload and draft.swp were both reported as "name ends
    .tmp", and ~$report.docx as "name starts with a dot".  The count and the
    filename were right and the REASON was invented, which is worse than
    saying nothing: he reads that clause to decide what to rename.

    So the sentence is DERIVED from the tuples rather than written beside
    them.  Add an eighth suffix and this returns a true clause for it
    without being edited; tests/test_skip_reason_is_true.py walks both
    tuples and fails on any member that cannot produce one.

    LONGEST MATCH WINS.  ``.jarvis-part-x.txt`` starts with "." AND with
    PART_PREFIX, and the specific rule is the useful thing to tell him.
    """
    if name.endswith(NOTE_SUFFIX):
        # Not his file at all: a note this lane wrote.  It is filtered out
        # of the count before the sentence is built, so he is never shown
        # this one -- but the rule still has to have a true word.
        return "it is a note I wrote, not a file of yours"
    hits = [p for p in SKIP_PREFIXES if name.startswith(p)]
    if hits:
        return f'name starts with "{max(hits, key=len)}"'
    hits = [x for x in SKIP_SUFFIXES if name.endswith(x)]
    if hits:
        return f'name ends "{max(hits, key=len)}"'
    return ""

# A send or a fetch that came back with one of these has told us about the
# FILE, and nothing it says bears on whether HPCOMPUTER is answering.  Every
# OTHER reason -- including the "failed" that classify_error falls back to
# when none of its seven regexes match the stderr -- is put to the machine
# again as a question before anybody calls the link down.  See
# :meth:`Syncer._probe_link`, and the two-in-one-pass defect it closes.
FILE_REASONS = frozenset({"odd-name", "not-found", "too-big", "denied",
                          "no-space", "exists", "not-there", "name-taken",
                          "too-many-copies", "name-too-long"})

# A listing that came back with one of these has told us about a SETTING,
# not about the link, and neither of them may be answered with a back-off
# or by stopping the other half of the lane (F-K).  They differ in one
# thing only, which is whether anything actually answered:
#
#   ANSWERED -- the far side took the connection and said the folder is not
#   there.  That is proof the box is UP, so it also clears the link state.
#   MEASURED 2026-09-05: with HPCOMPUTER healthy, a missing remote Inbox
#   put "link DOWN ... not answering since 15:36" on his desk, took the
#   interval from 30 s to 300 s and stopped the inbound half for 6 passes,
#   while a control with an empty local Outbox pulled a file fine over the
#   same transport, in the same state.
#
#   NEVER ASKED -- refused here, before a socket opened, because something
#   in his settings is missing.  It says nothing about the box either way,
#   so it must not claim the link is up OR down.
CONFIG_ANSWERED = frozenset({"not-there", "wrong-os"})
CONFIG_NEVER_ASKED = frozenset({"disabled", "no-host", "no-user", "no-key",
                                "bad-key", "no-ssh"})
CONFIG_REASONS = CONFIG_ANSWERED | CONFIG_NEVER_ASKED

# Directions that describe the LANE rather than a file: kept out of the
# record loops, which record files.
NOT_A_FILE = ("link", "config")

# errno values that mean "this filesystem has no hard links", as opposed to
# "that name is taken" (EEXIST, which is the answer land_beside wants) or a
# real failure.  FAT and exFAT are the cases that reach this on his desk.
_NO_HARDLINK = frozenset(
    e for e in (getattr(errno, n, None) for n in
                ("EPERM", "EOPNOTSUPP", "ENOTSUP", "ENOSYS", "EXDEV",
                 "EMLINK"))
    if e is not None)

MAX_COPIES = 50          # "name (2)" .. "name (50)", then refuse
MAX_CLAIM_TRIES = 8      # names TRIED on the far side, per file, per pass
MAX_TEMP_SWEEP = 5       # staging folders OF THIS RUN released per pass
MAX_ATTEMPTS = 5         # tries at ONE file before it is parked
RETRY_AFTER_S = 3600.0   # ...and how long it is parked for
HISTORY_LINES = 500      # the on-disk record, bounded
LEDGER_CAP = 4000        # remembered remote files, bounded

# FINDING Q.  ``remote.LISTING_CAP`` is 400, and this lane read "not in the
# first 400 entries" as "not on the machine".  Measured with 400+ files in
# the Windows Inbox: zebra.pdf was over there at exactly the right size and
# the note said "HPCOMPUTER reports no such file", and twelve passes left
# FIVE abandoned copies under his own filenames; the same cap hid an
# INBOUND file for ever, six passes, with the status still saying "link OK".
#
# Both halves are fixed by never letting a CAP mean ABSENCE.
#  * The verification no longer scans: a name that is not in the listing is
#    ASKED ABOUT BY NAME (:meth:`SshTransport.stat`), which is an answer
#    about that name and nothing else.
#  * The listing itself is read whole, up to a bound two orders of
#    magnitude larger, and when even that is hit the truncation is a fact
#    the lane KNOWS and says in status.txt rather than a silence.
# The per-pass WORK is still bounded -- by the number of transfers, which
# is the thing that was actually expensive -- so a huge folder makes the
# lane slower and never blind.
LISTING_HARD_CAP = 20000
MAX_PULLS_PER_PASS = 100

# A landing recorded before the claim (FINDING L) is resolved on the next
# pass; if his file never comes back, the row is forgotten after this.
LANDED_FORGET_S = 7 * 86400.0

# The one question that is HIS: may the lane take back a copy that landed
# under his filename on Windows and then failed its size check?  Doing so
# widens what this lane may delete over there from "a name of my own shape"
# to "a name I created in this run", which changes a promise, so it is a
# setting and the setting ships OFF.
DEFAULTS_REMOVE_BROKEN = False
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
    # OFF, and his to turn on.  When a copy reaches HPCOMPUTER under your
    # filename but arrives the wrong size, this lets me delete that broken
    # copy of mine; with it off the broken copy stays there under your name
    # and I only tell you about it.
    #
    # WHAT TURNING IT ON COSTS, said plainly because it is the reason it is
    # a setting.  The identity is the name THIS RUN took with a rename -l,
    # and between that rename and the size check there is a window in which
    # the file at that name can stop being ours: if you replace it over
    # there inside that window, the delete removes YOUR file, not mine.
    # MEASURED and DETERMINISTIC -- it is not a rare race, it is what the
    # code does whenever that sequence happens.  Off, nothing of yours can
    # be deleted on HPCOMPUTER by this lane at all.
    remove_broken_copies: bool = DEFAULTS_REMOVE_BROKEN


@dataclass(frozen=True)
class Sent:
    """One file this pass put on the far side, and the four things the rest
    of the pass needs about it.

    IT IS A CLASS AND NOT A TUPLE BECAUSE A TUPLE ALREADY WENT WRONG.
    Round 4 grew this from three fields to four, and one reader was left
    unpacking three::

        taken = {e.name for e in fresh} | {n for _, n, _ in sent}

    which is a ValueError on the exact path finding L was about -- a good
    file ahead of a failing one in the same pass.  MEASURED 50 of 50 first
    passes crashed.  Nothing of his was destroyed and nothing was sent
    twice (the ledger ordering caught it) and the inbound half still ran,
    but round 4's whole stated point was that one file's accident is not
    the pass's death.  Fields have names now; adding a fifth cannot break
    a reader that does not ask for it.
    """
    path: Path                       # HIS file, still in the Outbox
    landed: str                      # the name it took on the far side
    size: int                        # what we copied, for the verify
    key: str                         # the ledger key, computed while it was HERE


@dataclass(frozen=True)
class Entry:
    """One file on the far side, as its own listing described it."""
    name: str
    size: int
    stamp: str                       # the listing's date text, verbatim
    is_dir: bool = False             # a FOLDER over there is a taken name

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
    # THE FOURTH SILENCE.  Not a failure of this pass -- a deliberate rest
    # after five of them.  It reads as an outcome in the record so the hour
    # is visible there too, and status_text says when the rest ends.
    "parked": "it has failed five times, so I have stopped retrying it for "
              "a while rather than hammering it every half minute",
    # Every name we offered was taken at the instant we offered it.  This is
    # what a refusal looks like when somebody is actively writing into that
    # folder, and it is the SAFE outcome -- the alternative is the write
    # that goes through and destroys what is there.
    "name-taken": "every name I tried on HPCOMPUTER was taken at the moment "
                  "I tried it, so I have written over nothing and yours is "
                  "still here. I will try again",
    # The copy itself failed and the far side did not say anything either
    # end recognises.  It is about THIS file -- everything else in the
    # folder is still going, and HPCOMPUTER answered a listing either side
    # of this attempt.  The known cause is named because it is the one that
    # can never fix itself: scp will not write a file over a FOLDER.
    "failed": "the copy failed and HPCOMPUTER did not say why. This is "
              "about this one file -- it was still answering either side "
              "of the attempt. The usual cause is a FOLDER over there "
              "with the same name",
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


def name_series(name: str) -> list:
    """``name``, then ``name (2).ext`` .. ``name (51).ext``.

    The ONE place the "(2)" shape is written.  It used to be spelled out
    three times -- in the local dedupe, in the landing and (differently) in
    the caller -- which is how the two sides drifted apart in the first
    place: the local landing was made race-free and the remote one was
    still choosing a name minutes before it wrote it.
    """
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    return [name] + [f"{stem} ({n})" + (f".{ext}" if dot else "")
                     for n in range(2, MAX_COPIES + 2)]


def dedupe_name(name: str, taken) -> str:
    """``name``, or the first free ``name (2).ext``; "" past
    :data:`MAX_COPIES`.

    A GUESS, everywhere it is now used, and never the decision.  What it is
    for is skipping names that are already known to be taken so the claim
    that follows usually succeeds first time.  On this side the decision is
    :func:`land_beside`; on the far side it is a ``rename -l`` that can
    refuse.  Reading this function's answer as "that name is free" is
    exactly the mistake both Finding F and Finding J were.
    """
    return next((c for c in name_series(name) if c not in taken), "")


def claim_candidates(name: str, taken, limit: int = 0) -> list:
    """The names to TRY on the far side, best guess first, bounded.

    Bounded because each try is an sftp round trip: a name being taken the
    instant we try it is somebody actually writing there, and walking fifty
    of those inside one pass would hold up every other file in the queue.
    """
    free = [c for c in name_series(name) if c not in taken]
    return free[:max(1, int(limit or MAX_CLAIM_TRIES))]


def land_beside(source: Path, folder: Path, name: str) -> str:
    """MOVE ``source`` into ``folder`` as ``name`` -- or the first free
    ``name (2).ext`` -- and never, at any instant, over a file already
    there.  Returns the name it took, or "" past :data:`MAX_COPIES`.

    THIS IS THE WHOLE FIX for the pull race, and the reason it is a
    function rather than three careful lines at each call site.  Asking
    "is that name free?" and then writing is a guess about the next
    microsecond, however tightly the two are pushed together: the puller
    asked at the top of the pass and wrote after a transfer that can run
    for minutes, and a 513-byte notes.txt of his was measured being
    destroyed in that window on 2026-09-05, reported as "received".

    ``os.link`` is the one POSIX operation that CREATES a name and REFUSES
    if it is taken, atomically, in the kernel -- there is no window at all
    between the question and the answer.  The link is made, then the
    source name is removed; the inode, and so his bytes, is never at risk
    in between.  On a filesystem with no hard links (FAT, exFAT) the
    fallback claims the name with ``O_CREAT|O_EXCL``, which is atomic for
    the same reason; the only thing it costs is a 0-byte file OF OURS left
    behind if the process dies between the claim and the move -- never a
    byte of his.
    """
    source, folder = Path(source), Path(folder)
    hardlinks = True
    for candidate in name_series(name):
        dest = folder / candidate
        if hardlinks:
            try:
                os.link(source, dest)
            except FileExistsError:
                continue                      # taken, by the kernel's word
            except OSError as exc:
                if exc.errno not in _NO_HARDLINK:
                    raise
                hardlinks = False             # no links here: claim instead
            else:
                _unlink_after_landing(source, dest)
                return candidate
        try:
            fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(fd)
        try:
            os.replace(source, dest)          # over OUR OWN empty claim
        except OSError:
            shutil.move(str(source), str(dest))
        return candidate
    return ""


def _marked(fd: int) -> bool:
    """Does the file behind this OPEN descriptor start with our note
    marker?  Through the fd rather than the path, so the answer is about
    the inode we are going to write to and not about whatever happens to
    hold the name a moment later."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        head = os.read(fd, len(NOTE_MARKER.encode()))
        os.lseek(fd, 0, os.SEEK_SET)
    except OSError:
        return False
    return head == NOTE_MARKER.encode()


def _unlink_if_ours(path: Path) -> bool:
    """Remove a note only if it carries our marker.  A file of his at that
    name -- and the name is his filename plus our suffix, so it is a name
    he can have -- is left exactly where it is.

    Stated rather than hidden: between reading the marker and unlinking
    there is a sub-millisecond window in which he could replace that file
    with one of his own.  It cannot be closed with a syscall (there is no
    unlink-this-inode), and it is a very long way from the measured
    behaviour it replaces, which was to delete any name ending in the
    suffix without looking inside it at all.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        ours = _marked(fd)
    finally:
        os.close(fd)
    if not ours:
        log.info("foldersync: %s is not one of my notes; leaving it",
                 path.name)
        return False
    try:
        os.unlink(path)
    except OSError:
        return False
    return True


def _unlink_after_landing(source: Path, dest: Path) -> None:
    """The second half of the move.  If it fails the file exists in BOTH
    places, which is the safe direction and is said out loud rather than
    retried into a loop."""
    try:
        os.unlink(source)
    except OSError:
        log.warning("foldersync: %s is safely at %s but I could not remove "
                    "the original; nothing has been lost", source, dest)


# The counter half of _replace_ours's unique temp name.  Process-wide, so
# the ledger and status.txt cannot collide on a name even in one pass.
_REPLACE_SEQ = itertools.count(1)


def _replace_ours(path: Path, text: str, *, fsync: bool = False,
                  mode: Optional[int] = None) -> None:
    """Replace a file of OURS atomically.  The ONE place that dance lives.

    THE CHOKEPOINT, round 7, and it is deliberately narrow.  ``land_beside``
    is already the one function allowed to put a file into a folder of HIS;
    this is its counterpart for the three files this lane owns -- the ledger,
    status.txt and anything that follows them.  ``Ledger.save`` and
    ``Syncer.write_status`` each hand-rolled the same four steps and only one
    of the two fsynced.  Two copies of a dance where one has the safety step
    is exactly how the safety step goes missing on the third copy.

    ``fsync`` is not ceremony and it is not a default.  The landing record is
    what makes a duplicate send impossible rather than unlikely, and a record
    that is only in the page cache is not a record: a power cut between
    writing it and taking the name would lose exactly the fact the next pass
    needs.  A process kill survives the page cache; the wall socket does not.
    Both the file and the directory entry, which is the standard pair.
    status.txt is a courtesy for his file manager and is rewritten every pass,
    so it does not earn two fsyncs of his disk.

    The temp name sits beside the target, so the rename is within one
    filesystem and is therefore atomic.  It carries the target's whole name
    plus ".<pid>-<n>.tmp" rather than replacing the suffix: ``ledger.json``
    and ``ledger.jsonl`` would otherwise fight over ``ledger.tmp``.

    ROUND 10: THE TEMP IS CLAIMED, NOT OPENED.  Through round 9 the name was
    a fixed ``<target>.tmp`` and it was opened with a plain ``open(tmp,
    "w")``, which truncates whatever is there.  MEASURED on 1039ce8: a
    100000-byte file of his at ``~/Desktop/Jarvis/status.txt.tmp`` was
    GONE after one pass -- truncated to our status text and renamed over
    status.txt.  A narrow name and an undocumented one, but a name he can
    have, and the threat model's own sentence -- "every write that could
    land where a file already is takes one operation that creates the name
    or refuses" -- was not true of this function.  So the temp is now
    ``O_CREAT|O_EXCL`` at a UNIQUE name (pid + counter, the shape
    ``_claim_part`` already uses): a taken name is REFUSED by the kernel
    and the next one is tried, :data:`MAX_CLAIM_TRIES` times, then it
    raises the OSError both callers already catch.  Same mode bits as the
    ``open()`` it replaces (0o666 under the umask), so nothing about the
    target's permissions changed.  The cost is the one land_beside states:
    a temp OF OURS left at a unique name if the process dies between the
    claim and the rename -- never a byte of his.

    The FINAL ``os.replace`` over the target is still unclaimed, on
    purpose: the target is a file of OURS.  NOT for a name of his -- a
    plain replace at a name he chose is the shape that destroyed a
    100000-byte file in round 5.  That is ``land_beside``, which takes the
    name first.
    """
    path.parent.mkdir(parents=True, exist_ok=True,
                      **({"mode": mode} if mode is not None else {}))
    for _try in range(MAX_CLAIM_TRIES):
        tmp = path.with_name(f"{path.name}.{os.getpid()}-"
                             f"{next(_REPLACE_SEQ)}.tmp")
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue                      # taken, by the kernel's word
        break
    else:
        raise OSError(errno.EEXIST, "every temp name beside the target is "
                      "taken; nothing was written", str(path))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
        if fsync:
            fh.flush()
            os.fsync(fh.fileno())
    os.replace(tmp, path)
    if fsync:
        dirfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)


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
        remove_broken_copies=bool(get("remove_broken_copies",
                                      DEFAULTS_REMOVE_BROKEN)),
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


def remote_folder_problems(rconf: remote.RemoteConfig,
                           sconf: SyncConfig) -> list:
    """The far-side SETTINGS this lane cannot work without, checked here,
    with no socket, before anything is waiting on them.

    Finding K's other half.  A folder that is not there was being reported
    as a dead link at the first send; the runtime half of the fix says so
    properly, and this half catches the kinds that never needed asking:
    a folder that is not configured at all, one this module would refuse to
    put in an sftp line, one so deep that no filename fits under it on
    Windows, and a ``pull_from`` that names a key ``remote.pull_dirs`` does
    not have -- which is a typo that would otherwise look exactly like
    HPCOMPUTER being asleep, for ever.

    HONEST LIMIT, and it is why the runtime half exists: a path that is
    merely WRONG -- spelled properly, quotable, short enough, and simply
    not the folder that is there -- cannot be told from a right one without
    asking the box.  ``--check`` asks; the service reports it as a folder
    problem the first time it lists.
    """
    out = []
    inbox = remote.remote_dir(rconf, "inbox")
    hint = remote.CONFIG_HINT
    if not inbox.strip():
        out.append(f"remote.inbox is empty in {hint}, so I have nowhere on "
                   f"{rconf.name} to put the files you drop in the Outbox")
    elif remote._SFTP_UNQUOTABLE_RX.search(remote.scp_path(inbox)):
        out.append(f"remote.inbox ({inbox}) has a quote or a newline in it; "
                   f"I will not put that in an sftp line, so nothing can be "
                   f"sent until it is renamed in {hint}")
    elif len(remote.scp_path(inbox).rstrip("/")) + 2 > WINDOWS_MAX_PATH:
        out.append(f"remote.inbox ({inbox}) is already longer than Windows "
                   f"accepts for a whole path, so no file could land in it; "
                   f"shorten it in {hint}")
    key = (sconf.pull_from or "").strip()
    if key not in remote.PULL_KEYS:
        out.append(f"foldersync.pull_from is '{key}', which is not one of "
                   f"the folders I am allowed to read "
                   f"({', '.join(remote.PULL_KEYS)}); nothing would ever "
                   f"come back from {rconf.name}. It is in {hint}")
    elif not remote.remote_dir(rconf, key).strip():
        out.append(f"foldersync.pull_from is '{key}' but remote.pull_dirs."
                   f"{key} is empty in {hint}, so there is no folder on "
                   f"{rconf.name} to bring files from")
    return out


def preflight(rconf: remote.RemoteConfig, paths: Paths,
              sconf: Optional[SyncConfig] = None) -> list:
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
    if sconf is not None:
        out += remote_folder_problems(rconf, sconf)
    return out


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


# ------------------------------------------------------------- the listing
_SFTP_FIELDS = 8          # mode links user group size mon day time  name


def parse_sftp_entries(out: str) -> tuple:
    """``(rows, truncated)``: name, SIZE and date text out of an ``sftp
    ls -ln``, plus whether the answer was longer than we would read.

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
    lines = (out or "").splitlines()
    # FINDING Q: the cap used to be remote.LISTING_CAP (400) and silently
    # threw the rest away, which is how entry 401 became invisible for ever.
    # It is two orders of magnitude larger now AND the caller is told when
    # it bit, so a truncation is never mistaken for an empty folder.
    truncated = False
    for raw in lines:
        if len(rows) >= LISTING_HARD_CAP:
            truncated = True
            break
        line = raw.rstrip()
        if not line or line.startswith("sftp>"):
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
        # A "d" line is KEPT, flagged.  It used to be dropped here, which
        # made a FOLDER on the Windows side invisible to dedupe_name -- so
        # a file whose name matched one was sent straight at it, scp failed
        # in a way classify_error does not recognise, and that was read as
        # the link being down.  A folder over there is a name that is taken;
        # this lane needs to see it, and never fetches one.
        rows.append(Entry(name, size, " ".join(fields[5:8]),
                          line[0] == "d"))
    return rows, truncated


def _ours_to_move(name: str) -> bool:
    """A remote name this lane may rename or delete without asking: a part
    file of our own shape, or anything inside a staging directory we took
    exclusively.  The second case is the stronger of the two -- the first
    is a name PATTERN, and a pattern is what finding M was."""
    stage, slash, inner = (name or "").partition("/")
    if slash:
        return bool(remote.is_remote_stage(stage) and inner
                    and inner == os.path.basename(inner))
    return remote.is_remote_temp(name)


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
        self.last_truncated = False

    # -- where a push may land -----------------------------------------
    def target(self, name: str) -> str:
        """The remote path for a pushed name, or "" if this module will not
        write it.  ``name`` is a basename, or ONE staging directory of our
        own shape plus a name inside it -- nothing else.  A caller that
        passes ``../x`` or ``a/b`` gets "" rather than a cleaned-up path,
        because silently rewriting a path is how one escapes.
        """
        if not name:
            return ""
        stage, slash, inner = name.partition("/")
        if slash:
            # Inside a staging directory we created EXCLUSIVELY, so every
            # name in it is ours by construction (row 3).
            if not remote.is_remote_stage(stage) or not inner:
                return ""
            if inner != os.path.basename(inner) or windows_name_problem(inner):
                return ""
            base = self.target(stage)
            return f"{base}/{inner}" if base else ""
        if name != os.path.basename(name):
            return ""
        if windows_name_problem(name):
            return ""
        return remote.inbox_target(self.conf, name)

    # -- the claim on what we WRITE, not just on what he sees ----------
    def stage_open(self, stage: str) -> str:
        """Take a staging directory on the far side, exclusively.  "" or a
        reason.  This is row 3's fix: until this returns "", nothing of ours
        writes a byte over there."""
        why = remote.missing_reason(self.conf)
        if why:
            return why
        if not remote.is_remote_stage(stage):
            return "odd-name"
        path = self.target(stage)
        if not path:
            return "odd-name"
        res = remote.sftp_mkdir(self.conf, path)
        if res.ok:
            return ""
        if res.reason in ("unreachable", "timeout"):
            return remote.unreachable_reason(self.conf) or res.reason
        # mkdir says only "Failure" when the name is held -- and a name of
        # this shape being held is either our own crashed run or something
        # very strange, so it is a reason, never a silence.
        return "name-taken" if res.reason in ("failed", "") else res.reason

    def stage_close(self, stage: str) -> str:
        """Give the staging directory back.  The far side refuses a
        directory that is not empty, so this can never take a byte."""
        if not remote.is_remote_stage(stage):
            return "denied"
        path = self.target(stage)
        if not path:
            return "odd-name"
        res = remote.sftp_rmdir(self.conf, path)
        return "" if res.ok else (res.reason or "failed")

    def listing(self, key: str) -> tuple:
        self.last_truncated = False
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
        rows, truncated = parse_sftp_entries(res.out)
        self.last_truncated = truncated
        return rows, ""

    def stat(self, name: str, key: str = "inbox") -> tuple:
        """ASK ABOUT ONE NAME.  ``(Entry, "")`` when it is there,
        ``(None, "")`` when the far side ANSWERED and it is not, and
        ``(None, reason)`` when the far side could not answer at all.

        FINDING Q's fix.  A folder listing is capped, and a capped listing
        is evidence about the first N names and about nothing else -- which
        is how a file that had landed at exactly the right size was
        reported as "HPCOMPUTER reports no such file", twelve times, each
        one leaving another copy under his own filename.  One name is one
        question and gets one answer.
        """
        why = remote.missing_reason(self.conf)
        if why:
            return None, why
        folder = remote.remote_dir(self.conf, key)
        if not folder:
            return None, "not-there"
        if not remote.SAFE_REMOTE_NAME_RX.match(name or ""):
            return None, "odd-name"
        path = f"{remote.scp_path(folder).rstrip('/')}/{name}"
        if remote._SFTP_UNQUOTABLE_RX.search(path):
            return None, "odd-name"
        res = remote.run_sftp(self.conf, f'ls -ln "{path}"')
        if not res.ok:
            # `Can't ls: "..." not found` classifies as not-there, and for
            # ONE NAME that is the answer "it is not there" -- not a folder
            # problem and not an outage.
            if res.reason == "not-there":
                return None, ""
            reason = res.reason
            if reason in ("unreachable", "timeout"):
                reason = remote.unreachable_reason(self.conf) or reason
            return None, reason
        rows, _truncated = parse_sftp_entries(res.out)
        for row in rows:
            if row.name == name:
                return row, ""
        return None, ""

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

    def claim(self, temp: str, final: str, key: str = "inbox") -> str:
        """Take the name ``final`` for the bytes already landed at ``temp``,
        or say why not.  "" is the only success; "taken" means somebody
        else holds that name and NOTHING was written.

        This is the whole of the Finding J fix.  scp cannot refuse -- it
        truncates, measured, in every mode this link offers -- so the file
        goes at a name of ours first and the name he will see is taken by a
        ``rename -l``, which the far side refuses if it is held.  Two
        sessions racing for one name gave exactly one winner in 12 of 12
        rounds, so this is a claim, not a smaller window.

        The wire says only ``Failure``, for contention and for a bad path
        alike, so every failure is read as "I did not get the name" and
        never as done.  The reasons that mean the MACHINE (a timeout, a
        refused connection) are passed straight back out instead, because
        those are not this file's problem.
        """
        why = remote.missing_reason(self.conf)
        if why:
            return why
        if key != "inbox":
            return "denied"
        if not _ours_to_move(temp):
            return "odd-name"     # we only ever rename OUR OWN part file
        src, dst = self.target(temp), self.target(final)
        if not src or not dst:
            return "odd-name"
        res = remote.sftp_rename(self.conf, src, dst)
        if res.ok:
            return ""
        if res.reason in ("unreachable", "timeout"):
            return remote.unreachable_reason(self.conf) or res.reason
        return "taken" if res.reason in ("failed", "") else res.reason

    def discard(self, temp: str, key: str = "inbox") -> str:
        """Take one in-flight file OF OURS off his machine again.  Refuses
        any other name here as well as in remote.sftp_remove -- the one
        delete this lane can do is guarded at both ends."""
        if key != "inbox" or not _ours_to_move(temp):
            return "denied"
        dest = self.target(temp)
        if not dest:
            return "odd-name"
        res = remote.sftp_remove(self.conf, dest)
        return "" if res.ok else (res.reason or "failed")

    def remove_landed(self, name: str, key: str = "inbox") -> str:
        """Take back a copy that landed under HIS filename and then failed
        its size check.  The ONLY call in this lane that deletes a name he
        could have chosen, it exists only for
        ``foldersync.remove_broken_copies`` (shipped OFF), and the caller
        must have created that exact name in this run -- see
        :meth:`Syncer._remove_broken_copy`, which holds the identity."""
        if key != "inbox":
            return "denied"
        dest = self.target(name)
        if not dest or "/" in name:
            return "odd-name"
        res = remote.sftp_remove(self.conf, dest, claimed=True)
        return "" if res.ok else (res.reason or "failed")

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
        # FINDING L.  {push key: {"name", "size", "landed": bool, "t"}} --
        # the name this process is ABOUT TO TAKE on the far side, written
        # to disk BEFORE the claim goes out.  See :meth:`mark_claiming`.
        self.landed: dict = {}
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
        self.landed = {k: v for k, v in (data.get("landed") or {}).items()
                       if isinstance(v, dict) and v.get("name")}

    def save(self) -> None:
        self._trim()
        try:
            # THROUGH THE CHOKEPOINT since round 7.  The four steps and the
            # fsync pair are unchanged and now live in ONE place
            # (_replace_ours); fsync=True is the whole reason this call is
            # not the same as write_status's, and saying it here is better
            # than a second copy of the dance that might one day lose it.
            _replace_ours(self.path, json.dumps(
                {"version": 1, "pulled": self.pulled,
                 "fails": self.fails, "landed": self.landed}),
                fsync=True, mode=0o700)
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

    # -- what is already on the far side, or about to be ----------------
    def mark_claiming(self, key: str, name: str, size: int,
                      now: Optional[float] = None) -> None:
        """Write down the name we are ABOUT TO TAKE, before we take it.

        THE ORDERING THAT MAKES THE DUPLICATE IMPOSSIBLE (finding L).  The
        only step that puts his filename on HPCOMPUTER is the ``rename -l``,
        so the record goes to disk BEFORE that call and is resolved by
        ASKING the far side afterwards:

          crash before this row exists  -> nothing at his name -> resend, and
                                           a resend is correct
          crash after it, before the rename -> next pass asks: not there ->
                                           the row is dropped and it is sent
          crash after the rename -> next pass asks: there, at our size ->
                                           it is treated as landed and his
                                           original is MOVED, never resent
          the move itself fails (his hand) -> the row survives and the same
                                           question is asked next pass

        There is no window in which "it landed" is known only in memory, so
        the duplicate is not unlikely, it is unreachable.
        """
        now = time.time() if now is None else now
        self.landed[key] = {"name": name, "size": int(size), "landed": False,
                            "t": now}
        self._trim()

    def mark_landed(self, key: str, name: str,
                    now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self.landed[key] = {"name": name,
                            "size": int((self.landed.get(key) or {})
                                        .get("size") or 0),
                            "landed": True, "t": now}

    def landed_row(self, key: str, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        row = self.landed.get(key)
        if not row:
            return {}
        if now - float(row.get("t") or 0) > LANDED_FORGET_S:
            self.landed.pop(key, None)
            return {}
        return row

    def clear_landed(self, key: str) -> None:
        self.landed.pop(key, None)

    def _trim(self) -> None:
        if len(self.landed) > self.cap:
            keep = sorted(self.landed.items(),
                          key=lambda kv: float(kv[1].get("t") or 0),
                          reverse=True)[:self.cap]
            self.landed = dict(keep)
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
    def parked(self, key: str, now: Optional[float] = None):
        """``(when it is tried again, why it stopped)`` if this key is
        parked right now, else ``None``.

        Split out of :meth:`blocked` so the callers can SAY what they are
        skipping.  For an hour after the fifth failure both of them used to
        answer this question with a bare ``continue``: no event, no line in
        status.txt, nothing -- while the file sat in plain sight in the
        HPCOMPUTER outbox or in his own Outbox.  The ledger has held the
        deadline and the reason the whole time; nobody asked it.

        READ-ONLY.  Sweeping an expired row is a decision and it stays in
        :meth:`blocked`, which is the method that makes decisions.
        """
        now = time.time() if now is None else now
        row = self.fails.get(key)
        if not row or int(row.get("n") or 0) < MAX_ATTEMPTS:
            return None
        until = float(row.get("next") or 0)
        if now >= until:
            return None
        return until, str(row.get("why") or "")

    def blocked(self, key: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if self.parked(key, now) is not None:
            return True
        row = self.fails.get(key)
        if row and int(row.get("n") or 0) >= MAX_ATTEMPTS:
            self.fails.pop(key, None)          # the parking period is over
        return False

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
        self._config_problems: dict = {}     # {folder key: (path, reason)}
        self._skipped_names = False
        self._skipped_dirs = False
        # IDENTITY, not a pattern (findings M and M2).  Only a staging
        # directory THIS process created in THIS run is ever swept, so a
        # file of his that happens to match the shape, and the voice lane's
        # in-flight file, are both simply never looked at.
        self._my_stages: set = set()
        self._stage = ""                     # the one open this pass
        self._inner_seq = 0                  # files inside it
        self._claimed_here: set = set()      # names WE took, this run
        self._noted_foreign = False
        self._foreign_temps = 0
        # THE SILENCES.  Three things he can see with his own eyes while
        # status.txt says nothing, or says "empty".  The counts already
        # existed at the point each one is skipped; all that was missing
        # was saying them out loud.  Each is set from THIS pass's evidence
        # and cleared when there is none, so none of them can go stale the
        # way _foreign_temps did.
        self._unsafe_inbound = 0     # names over there this lane will not take
        self._remote_folders = 0     # FOLDERS over there; this lane moves files
        self._skipped_outbox: list = []   # his files SKIP_* passes over
        # THE FOURTH, and the one with a clock on it.  A file that has failed
        # MAX_ATTEMPTS times is parked for RETRY_AFTER_S -- an HOUR -- and
        # both skip sites were a bare `continue`.  Once every 30 seconds for
        # that hour this lane looked at a file, decided about it, and said
        # nothing, while he could see the file the whole time.  Each list is
        # (name, when it is tried again, why it stopped), set from THIS
        # pass's evidence by the half that owns it and cleared at the top of
        # that half, so neither can go stale the way _foreign_temps did.
        self._parked_in: list = []        # over there, in the HPCOMPUTER outbox
        self._parked_out: list = []       # here, in his own Outbox
        self._listing_truncated = False
        self._note_refused = False

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
        why = WHY.get(reason)
        if not why and reason in remote.FAIL_LINES:
            # A transport word with no plain-English line of its own reads
            # as "Why: auth." otherwise.  The lane already owns a sentence
            # for every one of them.
            why = remote.fail_line(self.rconf, reason).rstrip(".")
        text = (f"{NOTE_MARKER}\n\n"
                f"Jarvis could not send {target.name}.\n\n"
                f"Why: {why or reason}.\n")
        if extra:
            text += f"\n{extra}\n"
        text += ("\nYour file has not been moved, changed or deleted. Fix the\n"
                 "reason above (usually: rename it) and I will send it on the\n"
                 "next pass. This note disappears by itself when I can.\n")
        self._write_note(target.parent / (target.name + NOTE_SUFFIX), text)

    def _write_note(self, path: Path, text: str) -> None:
        """ROW 14.  ``<his file>.jarvis-cannot-send.txt`` is a name HE can
        choose, and a plain ``write_text`` at it destroyed the file of his
        that was measured sitting there.

        So: CREATE it exclusively, or -- if something is already at that
        name -- open it WITHOUT truncating, read the marker line through
        that same file descriptor, and only then truncate and write.  The
        check and the write are on one fd bound to one inode, so a file he
        puts there afterwards cannot be hit by a decision taken about the
        file that was there before.
        """
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                fd = os.open(path, os.O_RDWR)
            except OSError:
                log.debug("foldersync: cannot open a note", exc_info=True)
                return
            if not _marked(fd):
                os.close(fd)
                if not self._note_refused:
                    self._note_refused = True
                    log.warning("foldersync: %s is a file of yours, not one "
                                "of my notes; I have left it alone and said "
                                "nothing beside the file", path.name)
                return
        except OSError:
            log.debug("foldersync: cannot write a note", exc_info=True)
            return
        try:
            with os.fdopen(fd, "r+", encoding="utf-8") as fh:
                fh.seek(0)
                fh.truncate(0)
                fh.write(text)
        except OSError:
            log.debug("foldersync: cannot write a note", exc_info=True)

    def clear_note(self, target: Path) -> None:
        _unlink_if_ours(target.parent / (target.name + NOTE_SUFFIX))

    def _sweep_notes(self) -> None:
        """Our own notes whose file has gone are ours to remove -- an Outbox
        littered with explanations of files he already dealt with is worse
        than no explanation.  OURS is decided by the marker inside, never by
        the name: the name is his filename plus a suffix (row 14)."""
        try:
            entries = list(self.paths.outbox.iterdir())
        except OSError:
            return
        for p in entries:
            if not p.name.endswith(NOTE_SUFFIX):
                continue
            base = self.paths.outbox / p.name[:-len(NOTE_SUFFIX)]
            if not base.exists():
                _unlink_if_ours(p)

    # -- what the scanner will look at -----------------------------------
    def _candidates(self) -> list:
        out = []
        try:
            entries = sorted(self.paths.outbox.iterdir(), key=lambda p: p.name)
        except OSError:
            return out
        skipped = []
        for p in entries:
            name = p.name
            if name.startswith(SKIP_PREFIXES) or name.endswith(SKIP_SUFFIXES):
                # A HALF-WRITTEN file, or a dotfile, and skipping it is
                # right -- but the skip was SILENT, and status.txt then
                # said "outbox empty" while he was looking at the file.
                if not name.endswith(NOTE_SUFFIX):
                    skipped.append(name)
                continue
            out.append(p)
        self._skipped_outbox = skipped
        return out

    # ------------------------------------------------------------- pushing
    def push_once(self, now: Optional[float] = None) -> list:
        try:
            return self._push_once(now)
        finally:
            self._close_stage()

    def _push_once(self, now: Optional[float] = None) -> list:
        now = time.time() if now is None else now
        events: list = []
        # THE COUNT IS THIS PASS'S OR IT IS NOTHING.  It used to be set only
        # inside _sweep_my_stages, which is reached only when there is
        # something to send -- so a number measured while he was sending
        # files stayed on his desk for hours afterwards, describing litter
        # that may have been tidied away in the meantime.  A stale fact in
        # status.txt is the same defect as a silence, dressed up.
        #
        # It is CLEARED rather than recomputed on purpose.  The litter is in
        # the remote INBOX, and the only thing that lists that folder is the
        # push half; recomputing every pass would mean an extra ssh round
        # trip to a machine that may be asleep, on every pass, for a number
        # we only ever REPORT and never act on.  Saying nothing about what
        # we did not look at is the honest half of the trade, and it is
        # stated here rather than left for somebody to find.
        self._foreign_temps = 0
        self._parked_out = []               # this pass's word, not the last's
        self._sweep_notes()
        roots = filepick.expand_roots(self.rconf.local_roots)
        ready: list = []
        outstanding: list = []
        for p in self._candidates():
            # FINDING L, first half.  A file with an unresolved landing
            # record is not a candidate for SENDING at all until the far
            # side has been asked whether it is already over there.
            row = self.ledger.landed_row(f"push:{p.name}|{stat_key(p)}", now)
            if row:
                outstanding.append((p, row))
                continue
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
            key = f"push:{p.name}|{stat_key(p)}"
            if self.ledger.blocked(key, now):
                # PARKED, AND SAID SO.  His file is in his own Outbox and
                # status.txt already lists it under "outbox N waiting"; what
                # it never said is why it has stopped moving, or for how
                # long.  The event goes in the record because status.txt is
                # rewritten every pass and keeps no history of its own.
                until, why = self.ledger.parked(key, now)
                self._parked_out.append((p.name, until, why))
                events.append(Event(now, "push", p.name, self._size(p),
                                    "parked", why))
                continue
            ready.append(p)
        if not ready and not outstanding:
            for e in events:
                self.record(e)
            return events

        entries, why = self.transport.listing("inbox")
        if why:
            events.append(self._listing_failed(now, "inbox", why,
                                               len(ready) + len(outstanding)))
            for e in events:
                if e.direction not in NOT_A_FILE:
                    self.record(e)
            return events
        self._mark_up(now, "inbox")
        self._listing_truncated = getattr(self.transport, "last_truncated",
                                          False)
        self._sweep_my_stages(entries)

        # Resolve first, send second: nothing may be sent while a record
        # says it may already be over there.
        for p, row in outstanding:
            events += self._resolve_landing(p, row, now)

        taken = {e.name for e in entries}
        sent: list[Sent] = []
        for p in ready:
            size = self._size(p)
            key = f"push:{p.name}|{stat_key(p)}"   # while the file is HERE
            try:
                landed, reason = self._send_and_claim(p, taken, now, key)
            except OSError:
                # His hand, on his own file, while we were reading it.  One
                # file's accident is not the pass's death (finding L).
                log.warning("foldersync: %s went while I was sending it; "
                            "carrying on", p.name, exc_info=True)
                continue
            if reason:
                # A FILE that will not go is not a LINK that is down.  The
                # only thing that can speak for the machine is the machine,
                # so it is asked again, NOW -- the listing at the top of
                # this pass is evidence about the past.  If it answers,
                # this is one file's problem: it counts against that file,
                # it parks after MAX_ATTEMPTS, the queue behind it still
                # goes, and the inbound half still runs.
                fresh, why = ((None, "") if reason in FILE_REASONS
                              else self._probe_link("inbox"))
                if why:
                    events.append(
                        self._listing_failed(now, "inbox", why, len(ready)))
                    break
                if fresh is not None:
                    taken = ({e.name for e in fresh}
                             | {s.landed for s in sent})
                if reason == "too-many-copies":
                    # Not a failure that trying again can fix, so it is not
                    # charged an attempt.
                    self.note(p, reason)
                else:
                    n = self.ledger.bump(key, reason, now)
                    self.note(p, reason, f"Attempt {n} of {MAX_ATTEMPTS}.")
                events.append(Event(now, "push", p.name, size, reason))
                continue
            taken.add(landed)
            sent.append(Sent(p, landed, size, key))

        if sent:
            events += self._verify_and_move(sent, now)
        for e in events:
            if e.direction not in NOT_A_FILE:
                self.record(e)
        self.ledger.save()
        return events

    def _resolve_landing(self, p: Path, row: dict, now: float) -> list:
        """One outstanding landing record, settled by ASKING the far side.

        The other half of finding L.  A record says "I was about to take the
        name N for this file, or I had just taken it".  Only HPCOMPUTER can
        say which, so it is asked about that ONE name -- never inferred from
        a listing that is capped (finding Q) -- and the three answers are
        the three different things to do.
        """
        key = f"push:{p.name}|{stat_key(p)}"
        name, size = str(row.get("name") or ""), int(row.get("size") or 0)
        entry, why = self.transport.stat(name, "inbox")
        if why:
            # Cannot ask.  Keep the record: guessing either way is how a
            # file gets sent twice or dropped.
            return [Event(now, "push", p.name, size, "unverified", why)]
        # RESOLVED BY SIZE, NOT BY IDENTITY, and that is a real limit
        # rather than an oversight, so it is written down.  All this can
        # ask the far side is "is there something at that name, and is it
        # the number of bytes I sent" -- there is no checksum on this link
        # and no inode to compare.  A DIFFERENT file of exactly our size
        # that appeared at that name would be read as ours and his original
        # would be moved to Sent.
        #
        # Finding L's guarantee does not rest on this and never did: it
        # rests on ORDERING.  The name is written to the ledger and fsync'd
        # -- file and directory -- BEFORE the rename -l that is the only
        # step able to put his filename over there, and the rename refuses
        # a name anything already holds.  So the name asked about here is
        # one nothing else held at the instant we took it; the size is a
        # second opinion on top of that, not the thing being trusted.
        if entry is None or entry.is_dir or entry.size != size:
            # It never landed.  Drop the record; the file goes back through
            # the ORDINARY path on the next pass -- deliberately not this
            # one, so quiescence, the size cap, the name rules and the
            # per-file attempt limit are all applied in the one place that
            # applies them, and a repeatedly-failing file cannot use this
            # branch to skip its own parking.
            self.ledger.clear_landed(key)
            self.ledger.save()
            return []
        # It IS over there, at our size.  His original has never been moved,
        # so move it now -- and do NOT send it again.
        self.ledger.mark_landed(key, name, now)
        self.ledger.save()
        moved = self._move_to_sent(p)
        if moved:
            self.ledger.clear_landed(key)
            self.ledger.clear(key)
            self.clear_note(p)
            self.ledger.save()
        log.info("foldersync: %s was already on %s as %s from an earlier "
                 "pass; moving yours to Sent rather than sending it again",
                 p.name, self.rconf.name, name)
        return [Event(now, "push", p.name, size, "sent", name)]

    def _open_stage(self) -> tuple:
        """The one staging directory this pass writes into, taken the only
        way this link can take a name exclusively.  ``(name, "")`` or
        ``("", reason)``."""
        if self._stage:
            return self._stage, ""
        folder = remote.remote_dir(self.rconf, "inbox")
        stage = remote.remote_stage_name()
        # The FILE goes inside the folder, so the folder's own name is not
        # the whole path Windows has to accept -- leave room for "/f<n>".
        if path_problem(folder, stage) or \
                path_problem(folder, f"{stage} f000000"):
            return "", "name-too-long"
        why = self.transport.stage_open(stage)
        if why:
            return "", why
        self._stage = stage
        self._my_stages.add(stage)
        self._inner_seq = 0
        return stage, ""

    def _close_stage(self) -> None:
        stage, self._stage = self._stage, ""
        if not stage:
            return
        why = self.transport.stage_close(stage)
        if why:
            log.info("foldersync: my staging folder %s is still on %s (%s); "
                     "it is empty and I will take it away on a later pass",
                     stage, self.rconf.name, why)
        else:
            self._my_stages.discard(stage)

    def _send_and_claim(self, p: Path, taken, now: float,
                        key: str) -> tuple:
        """Put ONE file of his on HPCOMPUTER without ever writing at a name
        that could be his.  ``(landed_name, "")`` or ``("", reason)``.

        THIS IS THE WHOLE FIX for the push race, and the reason it is a
        method rather than two lines in the loop.  The old shape listed the
        remote Inbox once at the top of a pass, picked a free name with
        dedupe_name, and then scp'd straight at it -- and **scp truncates**:
        re-measured HERE against this box's own sftp-server, a 22222-byte
        file replaced by 1111 bytes, exit 0, nothing on stderr, in both scp
        protocol modes and through sftp put alike.  A file of his that appeared at that name in between
        was destroyed, the pass said "sent", and his LOCAL original was then
        moved into Sent -- both copies ours, his gone, and the verify could
        not catch it because the evidence it would have compared against was
        what got destroyed.  The window was not milliseconds either: the
        sends are sequential after ONE listing, so the tenth file in a queue
        was written minutes after its name was checked.

        So the bytes go at a name of OURS (:func:`remote.remote_temp_name` --
        pid, second, counter) and the name he will see is then TAKEN with a
        ``rename -l``, which the far side refuses if anything holds it.
        MEASURED HERE 2026-09-05, against this box's own sftp-server with
        no socket: refused 10 of 10 with both files byte-intact, and two
        concurrent sessions racing for one name gave exactly one winner in
        12 of 12 rounds, zero losses.  What HIS server does with that opcode
        is INFERRED from the protocol and not measured -- but ``-l`` pins
        the opcode at OUR client, so a wrong inference can only make the
        claim fail closed.  A refusal
        costs one more round trip and the next "(2)"; it never costs a file.

        ``taken`` is only the first GUESS now -- the freshest listing we
        have, used so the usual case claims first time.  It is not the
        decision, and nothing here reads it as one.
        """
        candidates = claim_candidates(p.name, taken)
        if not candidates:
            return "", "too-many-copies"
        stage, why = self._open_stage()
        if why:
            return "", why
        temp = f"{stage}/f{self._inner_seq}"
        self._inner_seq += 1
        reason = self.transport.send(p, temp)
        if reason:
            # THE SOURCE USED TO SAY "the close at the end of the pass, or a
            # later pass, takes it away".  The server does not agree: scp
            # leaves what it managed to write, and the rmdir in stage_close
            # then REFUSES the non-empty directory, so both the partial and
            # the folder are permanent.  MEASURED (against the fake far
            # side, which models scp's leftover): 5 staging folders and 5
            # part files after 12 passes on one failing file -- the code
            # saying one thing while the machine did another.
            #
            # So it is discarded HERE, at the one moment we know the name.
            # It is a file OF OURS, inside a directory OF OURS, taken with
            # the mkdir claim -- the safest delete in this lane -- and if
            # the far side refuses that too, _discard says so and the stage
            # simply stays, which is the truth rather than a promise.
            self._discard(temp)
            return "", reason
        size = self._size(p)
        for candidate in candidates:
            # DURABLE BEFORE THE CLAIM.  The rename is the only step that
            # can put his filename on that machine, so the record of which
            # name we are taking is on disk before it goes out -- see
            # Ledger.mark_claiming for the four crash points this covers.
            self.ledger.mark_claiming(key, candidate, size, now)
            self.ledger.save()
            why = self.transport.claim(temp, candidate)
            if not why:
                self._claimed_here.add(candidate)
                return candidate, ""
            self.ledger.clear_landed(key)
            self.ledger.save()
            if why != "taken":
                self._discard(temp)
                return "", why
        self._discard(temp)
        return "", "name-taken"

    def _discard(self, temp: str) -> None:
        """One in-flight file OF OURS, off his machine again."""
        why = self.transport.discard(temp)
        if why:
            log.info("foldersync: my part file %s is still on %s (%s); I "
                     "will take it away on a later pass", temp,
                     self.rconf.name, why)

    def _sweep_my_stages(self, entries) -> None:
        """Staging folders THIS PROCESS took in THIS RUN and did not manage
        to give back.  Bounded per pass.

        FINDINGS M AND M2, and the whole reason this method is not what it
        was.  It used to delete anything matching ``jarvis-part-*.tmp``,
        which is a PATTERN -- and a pattern is a guess about who made a
        file.  MEASURED: a file of HIS at ``jarvis-part-notes.tmp`` was
        deleted with no event, no note and no line in status.txt, and the
        same sweep ate the VOICE lane's in-flight file mid-transfer, after
        which the voice lane told him "There's already a file by that name
        where I'd put it, sir" -- which was false; nothing held the name,
        we had taken our own file away.  It fired up to 120 times an hour.

        So the guard is an IDENTITY: a name is swept only if it is in the
        set this process built by CREATING it.  Anything else of that shape
        -- his, the voice lane's, or one of ours from a run that has since
        died -- is left alone and COUNTED, and the count goes in
        status.txt so litter is visible rather than tidied away by force.
        """
        listed = {e.name for e in entries if e.is_dir}
        mine = [n for n in sorted(self._my_stages)
                if n in listed and n != self._stage]
        for name in mine[:MAX_TEMP_SWEEP]:
            if not self.transport.stage_close(name):
                self._my_stages.discard(name)
        self._foreign_temps = sum(
            1 for e in entries
            if (remote.is_remote_temp(e.name)
                or (remote.is_remote_stage(e.name)
                    and e.name not in self._my_stages)))
        if self._foreign_temps and not self._noted_foreign:
            self._noted_foreign = True
            log.info("foldersync: there are %d file(s) on %s shaped like my "
                     "own in-flight ones that I did not make; I have left "
                     "every one of them alone", self._foreign_temps,
                     self.rconf.name)

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
            # Found by the same sweep as the two defects above, and it is
            # the OTHER half of the rule: a LISTING that fails is the one
            # thing that IS evidence about the link, and this one was
            # silently swallowed -- no back-off, and the inbound half then
            # opened a second connection to a box that had just refused a
            # first.  It is a link event now.  The files are NOT counted
            # against: an outage must never park a file of his.
            for s in sent:
                events.append(Event(now, "push", s.path.name, s.size,
                                    "unverified", why))
            events.append(self._listing_failed(now, "inbox", why, len(sent)))
            return events
        # A FOLDER at that name is not the file we sent, and its 4096 must
        # never be read as a byte count -- excluded, so the check fails
        # loudly instead of passing by coincidence.
        sizes = {e.name: e.size for e in entries if not e.is_dir}
        for s in sent:
            p, landed, size, key = s.path, s.landed, s.size, s.key
            # `key` was computed while his file was still in the Outbox: it
            # is the one the NEXT pass will look up if his hand takes the
            # file away before the move, so it must not be recomputed here.
            there = sizes.get(landed)
            if there is None:
                # FINDING Q.  "Not in the listing" is NOT "not there": the
                # listing is capped, and a Windows Inbox with 400+ files in
                # it made this branch report a file that had landed
                # perfectly as missing, twelve times, leaving five copies
                # under his own name.  So ASK ABOUT THAT NAME.
                entry, why = self.transport.stat(landed, "inbox")
                if why:
                    events.append(Event(now, "push", p.name, size,
                                        "unverified", why))
                    continue
                there = entry.size if entry is not None and not entry.is_dir \
                    else None
            if there != size:
                # It is over there and it is WRONG, which is a settled
                # answer: the landing record has done its job and must go,
                # or the next pass would spend a round trip re-asking a
                # question this one just answered.
                self.ledger.clear_landed(key)
                n = self.ledger.bump(key, "verify-failed", now)
                extra = (f"I copied {size} bytes but HPCOMPUTER reports "
                         f"{there if there is not None else 'no such file'}. "
                         f"Attempt {n} of {MAX_ATTEMPTS}.")
                detail = str(there)
                took_back = self._remove_broken_copy(landed)
                if took_back:
                    detail = f"{there} (broken copy removed)"
                else:
                    extra += ("\nThe broken copy is still on HPCOMPUTER under "
                              "your filename. I do not delete anything of "
                              "yours over there; set "
                              "foldersync.remove_broken_copies to true in "
                              "~/.config/jarvis/assistant.json if you would "
                              "rather I took my own bad copies back.")
                self.note(p, "verify-failed", extra)
                events.append(Event(now, "push", p.name, size,
                                    "verify-failed", detail))
                continue
            # FINDING L.  On disk BEFORE the move, so a move that fails --
            # because his hand took the file -- is resolved next pass by
            # asking, and never by sending it a second time.
            self.ledger.mark_landed(key, landed, now)
            self.ledger.save()
            moved = self._move_to_sent(p)
            self.ledger.clear(key)
            if moved:
                self.ledger.clear_landed(key)
            self.clear_note(p)
            events.append(Event(now, "push", p.name, size, "sent", landed))
            if moved and moved != p.name:
                log.info("foldersync: kept %s as %s in Sent", p.name, moved)
        return events

    def _remove_broken_copy(self, landed: str) -> bool:
        """THE ONE QUESTION THAT IS HIS, and it ships answered NO.

        When a copy reaches HPCOMPUTER under his filename and then fails
        its size check, the lane cannot take it back: the promise is that
        it never deletes anything of his over there, and a name he could
        have chosen is indistinguishable from one of his.  MEASURED: five
        orphaned copies of one file after twelve passes.  Taking it back
        means widening that promise from "a name of my own shape" to "a
        name I created in this run", which is a change to what he was
        told, so it is a SETTING and the setting is OFF.

        With it on, the identity is held HERE -- ``self._claimed_here`` is
        the set of names this process actually took with a ``rename -l`` --
        and remote.sftp_remove logs every such delete at WARNING.
        """
        if not self.conf.remove_broken_copies:
            return False
        if landed not in self._claimed_here:
            return False                    # not a name we took: not ours
        why = self.transport.remove_landed(landed)
        if why:
            log.warning("foldersync: I could not take my broken copy of %s "
                        "off %s (%s)", landed, self.rconf.name, why)
            return False
        self._claimed_here.discard(landed)
        log.warning("foldersync: removed my own broken copy of %s from %s "
                    "(foldersync.remove_broken_copies is on)", landed,
                    self.rconf.name)
        return True

    def _move_to_sent(self, p: Path) -> str:
        """His original, out of the Outbox and into Sent -- MOVED, never
        deleted, so the Outbox visibly empties and the file is still one
        double-click away.

        This side already re-read Sent immediately before renaming, which
        made the window microseconds rather than minutes -- but it was
        still a check followed by an ``os.replace``, and a window that
        small is still a window.  It lands through the same
        :func:`land_beside` as the pull now, so there is none.

        FINDING L.  It also has to survive HIS HAND: a file dragged back out
        of the Outbox while this pass was in flight raised FileNotFoundError
        out of ``os.link`` and killed the whole pass -- status.txt was never
        rewritten, the inbound half never ran, and a file that HAD landed
        was left in the Outbox and sent again next pass.  A vanished
        original is not an error here; it is him, doing the thing the folder
        is for.  It returns "" and the landing record in the ledger is what
        stops the duplicate.
        """
        try:
            self.paths.sent.mkdir(parents=True, exist_ok=True)
            moved = (land_beside(p, self.paths.sent, p.name)
                     or land_beside(p, self.paths.sent,
                                    f"{p.name}.{int(time.time())}-"
                                    f"{os.getpid()}"))
        except FileNotFoundError:
            log.info("foldersync: %s was taken out of the Outbox before I "
                     "could move it to Sent; it is on %s and I will not "
                     "send it again", p.name, self.rconf.name)
            return ""
        except OSError:
            log.warning("foldersync: could not move %s to Sent; it stays in "
                        "the Outbox and is not sent again", p.name,
                        exc_info=True)
            return ""
        if not moved:
            # Fifty-one names taken, and the stamped one too.  Say so
            # rather than report a move that did not happen: his file is
            # still in the Outbox and will go again next pass as a copy.
            log.warning("foldersync: %s was sent but I could not free a "
                        "name for it in %s; it is still in the Outbox",
                        p.name, self.paths.sent)
            return p.name
        return moved

    # ------------------------------------------------------------- pulling
    def pull_once(self, now: Optional[float] = None) -> list:
        now = time.time() if now is None else now
        if self._pass_down:
            return []                          # asked once a pass, not twice
        events: list = []
        # This pass's word, not the last one's.  A count kept from an
        # earlier listing is the _foreign_temps mistake with a different
        # name, so both are cleared before anything is asked.
        self._unsafe_inbound = self._remote_folders = 0
        self._parked_in = []
        entries, why = self.transport.listing(self.conf.pull_from)
        if why:
            return [self._listing_failed(now, self.conf.pull_from, why, 0)]
        self._mark_up(now, self.conf.pull_from)
        self._listing_truncated = (self._listing_truncated
                                   or getattr(self.transport,
                                              "last_truncated", False))

        usable = []
        for e in entries:
            if e.is_dir:
                # It is in the listing so dedupe_name can see the name is
                # taken; it is never fetched.  This lane moves files, one
                # at a time, and there is no note to leave over there --
                # the jarvis account cannot write in that folder.  He can
                # SEE the folder, so status.txt says how many there are.
                self._remote_folders += 1
                if not self._skipped_dirs:
                    self._skipped_dirs = True
                    log.info("foldersync: there is a folder in the "
                             "HPCOMPUTER outbox; I move files, one at a "
                             "time, never a folder")
                continue
            if remote.SAFE_REMOTE_NAME_RX.match(e.name):
                usable.append(e)
                continue
            # A comma, an ampersand, an accent: this lane will not put that
            # in a path, so the file sits over there for ever.  He can see
            # it there, and until round 5 status.txt said nothing at all.
            self._unsafe_inbound += 1
            if not self._skipped_names:
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

        fetched = 0
        for entry in usable:
            if self.ledger.has(entry.key):
                continue
            if self.ledger.blocked(f"pull:{entry.key}", now):
                # PARKED, AND SAID SO.  This is the one he can SEE: the file
                # stays in the HPCOMPUTER outbox for the whole hour and, until
                # now, nothing on his desk mentioned it once.
                until, why = self.ledger.parked(f"pull:{entry.key}", now)
                self._parked_in.append((entry.name, until, why))
                events.append(Event(now, "pull", entry.name, entry.size,
                                    "parked", why))
                continue
            if fetched >= MAX_PULLS_PER_PASS:
                # A WORK bound, not a visibility one (finding Q).  Every
                # entry stays in the listing and in the ledger sweep, so
                # the queue drains over the following passes; what is
                # bounded is how long one pass may spend transferring.
                log.info("foldersync: %d more file(s) waiting on %s; taking "
                         "them on the next passes", len(usable) - fetched,
                         self.rconf.name)
                break
            fetched += 1
            # A cheap look BEFORE the transfer, so a name with fifty
            # copies already costs no bytes.  It is not the decision: the
            # name is chosen and claimed in one step, after the fetch, by
            # land_beside.
            if not dedupe_name(entry.name, taken):
                events.append(Event(now, "pull", entry.name, entry.size,
                                    "too-many-copies"))
                continue
            part, reason = self._claim_part(entry.name)
            if not part:
                events.append(Event(now, "pull", entry.name, entry.size,
                                    reason))
                continue
            reason = self.transport.fetch(self.conf.pull_from, entry.name,
                                          part)
            if reason:
                self._drop(part)
                # Same rule as the push side: one file that would not come
                # is not the machine being gone.  A timeout on an oversized
                # file used to back the whole lane off.
                _fresh, why = ((None, "") if reason in FILE_REASONS
                               else self._probe_link(self.conf.pull_from))
                if why:
                    events.append(self._listing_failed(
                        now, self.conf.pull_from, why, 0))
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
                landed = land_beside(part, self.paths.inbox, entry.name)
            except OSError:
                log.warning("foldersync: cannot put %s into %s",
                            entry.name, self.paths.inbox, exc_info=True)
                self._drop(part)
                self.ledger.bump(f"pull:{entry.key}", "denied", now)
                events.append(Event(now, "pull", entry.name, entry.size,
                                    "denied"))
                continue
            if not landed:
                self._drop(part)
                events.append(Event(now, "pull", entry.name, entry.size,
                                    "too-many-copies"))
                continue
            taken.add(landed)
            self.ledger.clear(f"pull:{entry.key}")
            self.ledger.mark_pulled(entry.key, landed, now)
            events.append(Event(now, "pull", entry.name, entry.size,
                                "received", landed))
        for e in events:
            if e.direction not in NOT_A_FILE:
                self.record(e)
        self.ledger.save()
        return events

    def _claim_part(self, name: str) -> tuple:
        """``(path, "")`` for a part file in his Inbox that is OURS by the
        kernel's word, or ``("", reason)``.

        ROW 5, and the third appearance of the same shape in this lane.
        The puller wrote straight into ``.jarvis-part-<his name>`` with no
        check and no ``O_EXCL`` at all: a 99999-byte file measured sitting
        at that name was destroyed by the transfer and the pass reported
        "received".  The window was the whole transfer, up to
        remote.MAX_TRANSFER_S.

        ``O_CREAT|O_EXCL`` is one syscall that either creates the name or
        refuses because somebody holds it -- the same operation
        :func:`land_beside` and ``remote.pull`` already use.  The name also
        carries a pid and a counter now, so two passes, two processes or a
        crashed run cannot collide on it and be mistaken for his file.
        """
        for n in range(MAX_CLAIM_TRIES):
            self._inner_seq += 1
            part = self.paths.inbox / (
                f"{PART_PREFIX}{os.getpid()}-{self._inner_seq}-{name}")
            try:
                fd = os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                continue
            except OSError:
                log.warning("foldersync: cannot claim a part file in %s",
                            self.paths.inbox, exc_info=True)
                return None, "denied"
            os.close(fd)
            return part, ""
        return None, "name-taken"

    # ------------------------------------------------------------- the link
    def _probe_link(self, key: str) -> tuple:
        """``(entries, "")`` if the far side answers a listing RIGHT NOW,
        ``(None, reason)`` if it cannot.

        The one question that is allowed to condemn the link, asked at the
        moment of doubt.  A transfer that failed is evidence about a file:
        scp will not write over a folder, a name can be refused, a single
        copy can time out -- and none of that means HPCOMPUTER has gone.
        MEASURED 2026-09-05 with the far side healthy and listing fine in
        the very same pass: one such file put the status at "link DOWN:
        HPCOMPUTER wouldn't answer that, sir", stopped the inbound half for
        12 passes, and never parked.

        It costs one extra listing per failing file per pass, and only ever
        on a failure.  On the push side the entries it brings back are not
        wasted -- they are the freshest word on which names are taken.
        """
        entries, why = self.transport.listing(key)
        return (None, why) if why else (entries, "")

    def _listing_failed(self, now: float, key: str, why: str,
                        waiting: int) -> Event:
        """A listing that could not answer, sorted into the two different
        things it can mean.  Only the MACHINE not answering is the link;
        a folder that is not there is a setting (F-K), and so is a setting
        that never let a socket open at all."""
        if why in CONFIG_REASONS:
            return self._config_event(now, key, why)
        return self._link_event(now, why, waiting)

    def _config_setting(self, key: str) -> str:
        """The line in his settings file that this key comes from."""
        return ("remote.inbox" if key == "inbox"
                else f"remote.pull_dirs.{key}")

    def _config_sentence(self, key: str, path: str, why: str) -> str:
        if why in CONFIG_ANSWERED:
            return (f"{self.rconf.name} answered, but the folder I was told "
                    f"to use is not there: {path or '(not set)'}. That is "
                    f"{self._config_setting(key)} in {remote.CONFIG_HINT}. "
                    f"The link is fine, nothing is lost, and I am not "
                    f"slowing down")
        return remote.fail_line(self.rconf, why)

    def _config_event(self, now: float, key: str, why: str) -> Event:
        """A SETTING is wrong, which is not an outage.  No back-off -- the
        cost of asking again is one listing and the box is right there --
        and, crucially, ``_pass_down`` is left alone so the other direction
        still runs: a mistyped inbox must not stop the files coming IN.
        """
        path = remote.remote_dir(self.rconf, key)
        if why in CONFIG_ANSWERED:
            # It answered.  That is what "that folder is not there" IS: an
            # answer.  So the link is up, whatever it said before.
            self._mark_up(now)
        problem = (path, why)
        first = self._config_problems.get(key) != problem
        self._config_problems[key] = problem
        event = Event(now, "config", key, 0, "config-problem", path or why)
        if first:
            self.record(event)
            log.warning("foldersync: %s",
                        self._config_sentence(key, path, why))
        return event

    def _config_clear(self, key: str) -> None:
        if self._config_problems.pop(key, None) is not None:
            log.info("foldersync: the %s folder on %s is there after all",
                     key, self.rconf.name)

    def _link_event(self, now: float, why: str, waiting: int) -> Event:
        """Down, once.  The WARNING and the record entry are written on the
        transition only: a box asleep overnight must not fill either."""
        first = not self._down_reason
        if first:
            self._down_since = now
        self._down_reason = why
        # "That folder is not there" rested on the box having ANSWERED.  It
        # is not answering now, so that sentence is no longer supported and
        # must come off the status file rather than sit there beside a
        # contradiction.
        self._config_problems.clear()
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

    def _mark_up(self, now: float, key: Optional[str] = None) -> None:
        """The far side answered.  ``key`` is the folder it answered ABOUT,
        whose config complaint is thereby settled -- a listing that comes
        back is the only evidence that the folder is there."""
        if self._logged_down:
            log.info("foldersync: HPCOMPUTER is answering again")
        self._logged_down = False
        self._down_reason = ""
        self._down_since = 0.0
        self.interval_s = self.conf.remote_interval_s
        self._last_ok = now
        if key:
            self._config_clear(key)

    def check_far_side(self) -> list:
        """ASK about both folders and say, in his words, what is not there.

        Only ``--check`` calls this, deliberately: the service must not
        open sockets to answer questions nobody asked, and a preflight that
        dialled a sleeping box would turn a config check into a 12-second
        wait.  But he runs ``--check`` precisely when something is wrong,
        and the answer he needs -- "that folder is not on HPCOMPUTER" --
        cannot be had any other way.
        """
        out: list = []
        keys: list = []
        for key in ("inbox", self.conf.pull_from):
            if key and key not in keys:
                keys.append(key)
        for key in keys:
            _entries, why = self.transport.listing(key)
            if not why:
                continue
            path = remote.remote_dir(self.rconf, key)
            if why in CONFIG_REASONS:
                out.append(self._config_sentence(key, path, why))
            else:
                out.append(f"I could not look in {path or key} on "
                           f"{self.rconf.name}: "
                           f"{remote.fail_line(self.rconf, why)}")
        return out

    # -------------------------------------------------------------- a pass
    def run_pass(self, now: Optional[float] = None) -> list:
        """Both halves, and then the truth on his desk -- WHATEVER happened.

        FINDING L's third consequence.  The two halves and the status write
        used to be three statements in a row, so anything that raised in
        the first one skipped the other two: a file dragged out of the
        Outbox mid-pass left status.txt saying "outbox empty" beside a
        7-byte file of his that was sitting right there, and the inbound
        half did not run at all.  A status file that is untrue is one of
        his stated conditions for merging this, so the write is in a
        ``finally`` and each half is fenced off from the other.

        The fences are deliberately narrow: they catch, they LOG WITH A
        TRACEBACK, and they put the failure in the record as an event.
        Nothing is swallowed quietly.
        """
        now = time.time() if now is None else now
        self._pass_down = False
        self._listing_truncated = False      # this pass's word, not the last's
        events: list = []
        try:
            events += self._half(self.push_once, now, "push")
            events += self._half(self.pull_once, now, "pull")
        finally:
            self._last_pass = now
            try:
                self.write_status()
            except Exception:               # noqa: BLE001 - never leave a lie
                log.exception("foldersync: cannot write the status file")
        return events

    def _half(self, fn, now: float, direction: str) -> list:
        try:
            return fn(now)
        except Exception:                   # noqa: BLE001 - one half only
            log.exception("foldersync: the %s half of this pass failed; the "
                          "other half still runs and nothing is lost",
                          direction)
            event = Event(now, direction, "", 0, "pass-failed")
            self.record(event)
            return [event]

    # ------------------------------------------------------------- status
    def status_text(self) -> str:
        waiting = [p.name for p in self._candidates()]
        lines = [f"Jarvis folder sync  --  {self.rconf.name}",
                 f"checked   {self.status_clock()}"]
        if self._down_reason:
            lines.append(f"link      DOWN: "
                         f"{remote.fail_line(self.rconf, self._down_reason)}")
            since = time.strftime("%H:%M:%S", time.localtime(self._down_since))
            lines.append(f"          not answering since {since}. Your files "
                         f"are safe where they are;")
            lines.append("          I keep trying, more slowly.")
        elif self._last_ok:
            lines.append("link      OK, last answered " + time.strftime(
                "%H:%M:%S", time.localtime(self._last_ok)))
        else:
            # NEVER claim health that has not been measured.  A fresh
            # process has not spoken to HPCOMPUTER yet and must say so.
            lines.append("link      not checked yet")
        # A folder that is not there is NOT the link being down, and saying
        # so was the whole of Finding K: he was told the box was not
        # answering while it was answering every 30 seconds, and told
        # nothing at all about the one thing he could have fixed.
        for key in sorted(self._config_problems):
            path, why = self._config_problems[key]
            if why in CONFIG_ANSWERED:
                lines.append(f"folder    NOT THERE on {self.rconf.name}: "
                             f"{path or '(not set)'}")
                lines.append("          It answered; that folder is what "
                             "is missing. Check")
                lines.append(f"          {self._config_setting(key)} in "
                             f"{remote.CONFIG_HINT}. Nothing is lost and I")
                lines.append("          am still trying at the usual rate.")
            else:
                lines.append(f"setup     "
                             f"{remote.fail_line(self.rconf, why)}")
        if self._listing_truncated:
            # FINDING Q.  A folder too big to read whole is a FACT he is
            # told, never a silence that looks like an empty folder.
            lines.append(f"folder    more than {LISTING_HARD_CAP} files on "
                         f"{self.rconf.name}; I am working through them")
            lines.append("          a pass at a time and nothing is lost.")
        if self._foreign_temps:
            lines.append(f"note      {self._foreign_temps} file(s) on "
                         f"{self.rconf.name} look like my own in-flight")
            lines.append("          ones but I did not make them, so I have "
                         "left them alone.")
        # THE THREE SILENCES, said out loud.  Every one of these is
        # something he can see with his own eyes -- a file still sitting on
        # HPCOMPUTER, a folder there, a file still sitting in his Outbox --
        # while this file said nothing, or said "empty".  No new machinery:
        # each count already existed at the point the thing was skipped.
        if self._unsafe_inbound:
            lines.append(f"note      {self._unsafe_inbound} file(s) on "
                         f"{self.rconf.name} I cannot bring across")
            lines.append("          (their names have characters I will "
                         "not put in a path).")
            lines.append("          Rename them over there and I will "
                         "take them next pass.")
        if self._remote_folders:
            lines.append(f"note      {self._remote_folders} folder(s) in "
                         f"the {self.rconf.name} outbox. I move files,")
            lines.append("          one at a time, never a folder.")
        # THE FOURTH SILENCE, and the one with a clock on it.  Five failures
        # park a file for an hour; for that hour both halves skipped it with
        # a bare `continue` and nothing anywhere said so, while he could see
        # the file the whole time -- in the HPCOMPUTER outbox, or in his own
        # Outbox under "N waiting" with no reason beside it.  Say the count,
        # ONE example with the reason it stopped, and -- the half that makes
        # it a sentence rather than a shrug -- WHEN it starts again.
        for where, rows in (("on " + self.rconf.name, self._parked_in),
                            ("in your Outbox", self._parked_out)):
            if not rows:
                continue
            name, until, why = min(rows, key=lambda r: r[1])
            when = time.strftime("%H:%M:%S", time.localtime(until))
            lines.append(f"note      {len(rows)} file(s) {where} I have "
                         f"stopped retrying for now")
            lines.append(f"          (e.g. {name} -- {MAX_ATTEMPTS} tries, "
                         f"last said: {why or 'no reason recorded'}).")
            lines.append(f"          Nothing is lost and nothing has been "
                         f"changed. Next try {when}.")
        lines.append(f"outbox    {len(waiting)} waiting"
                     if waiting else "outbox    empty")
        for name in waiting[:10]:
            lines.append(f"            {name}")
        if len(waiting) > 10:
            lines.append(f"            ... and {len(waiting) - 10} more")
        if self._skipped_outbox:
            # The rule that ACTUALLY fired, not one of two guesses.  See
            # skip_reason: round 5's two hardcoded clauses were measured
            # wrong on 3 of 3 real names.
            why = skip_reason(self._skipped_outbox[0])
            lines.append(f"note      {len(self._skipped_outbox)} file(s) in "
                         f"your Outbox I am not sending")
            lines.append(f"          (e.g. {self._skipped_outbox[0]} -- "
                         f"{why}). A half-written file is")
            lines.append("          left alone on purpose; rename it and "
                         "I will send it.")
        if self._recent:
            lines.append("")
            lines.append("recent")
            for e in reversed(self._recent[-10:]):
                when = time.strftime("%H:%M:%S", time.localtime(e.when))
                word = {"sent": "sent    ", "received": "received"}.get(
                    e.outcome, e.outcome)
                # When it landed beside a file of his, say the name that is
                # actually on the disk.  The old line said "received
                # notes.txt" for a file that is not there under that name.
                as_ = (f" as {e.detail}"
                       if e.outcome in ("sent", "received")
                       and e.detail and e.detail != e.name else "")
                lines.append(f"  {when}  {word}  {e.name}{as_} "
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
            # THROUGH THE CHOKEPOINT since round 7.  No fsync: status.txt is
            # a courtesy for his file manager, rewritten on every pass that
            # changes it, and nothing downstream depends on it having
            # survived a power cut.  The ledger is the opposite case and
            # passes fsync=True.
            _replace_ours(self.paths.status, text)
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
    problems = preflight(rconf, sconf.paths, sconf)
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
        # The one place that is allowed to ASK.  An offline check cannot
        # tell a mistyped remote folder from a right one, and he is sitting
        # in front of this command waiting for the answer.
        live = [] if problems else syncer.check_far_side()
        for line in live:
            print(f"problem: {line}", file=sys.stderr)
        if not syncer.conf.enabled:
            print("foldersync.enabled is false in "
                  "~/.config/jarvis/assistant.json", file=sys.stderr)
        return 1 if (problems or live) else 0
    if problems:
        return 2
    if not syncer.conf.enabled:
        # 3, not 0.  The unit is Restart=always, so exiting 0 here would make
        # systemd start this every 15 seconds for as long as the switch is
        # off -- a restart loop whose only symptom is a churning journal.
        # The unit lists 3 in RestartPreventExitStatus, so it stops cleanly
        # and says why.
        print("foldersync.enabled is false in ~/.config/jarvis/"
              "assistant.json; nothing to do.", file=sys.stderr)
        return 3

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
