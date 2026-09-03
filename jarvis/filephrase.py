"""The spoken phrases a file name is hiding inside.

``jarvis.tools.filepick`` resolves a NAME -- "the budget spreadsheet" -- and
is shared with the HPCOMPUTER lane. This module handles the three shapes it
cannot, all of them from the brief and all of them things he actually says:

    "that file on my desktop"      no name at all; a FOLDER is the handle
    "the PDF I just downloaded"    a TYPE and a RECENCY, still no name
    "the lab report on my desktop" a name PLUS a folder to look in

So this is a phrase layer, not a second resolver. Everything that can be
wrong in a dangerous way is still filepick's: :func:`filepick.score` decides
how well a name matches, :func:`filepick.check_file` decides whether a
resolved path may be touched at all (containment, symlinks out of the roots,
folders, fifos, size), :data:`filepick.SKIP_DIRS` decides what is never
offered, and :func:`filepick.reason_line` says why not. Two resolvers that
disagreed about "that file" is the failure the shared module was written to
prevent, and nothing here re-implements any of that.

ONE deliberate difference, and it is the reason this file exists rather than
a flag on ``pick()``:

    **the tie band is wider here.** filepick calls two candidates tied when
    their scores are within 0.02; this lane calls them tied within
    ``TIE_SCORE`` (0.12). "The lab report" against ``lab_report.pdf`` (0.87)
    and ``lab_report_final.pdf`` (0.77) is one match under the tight rule
    and a QUESTION under this one. Both lanes read the file back before
    acting, so neither is unsafe -- but a copy to his own second machine is
    a mistake he can delete, and an attachment in someone else's inbox is
    not, so this lane buys certainty with a question and the other does not.

No transport, no config, no network: the roots are an argument.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from jarvis.logs import get_logger
from jarvis.tools import filepick

log = get_logger("filephrase")

# See the module docstring. Wider than filepick's 0.02, on purpose.
TIE_SCORE = 0.12
# "the PDF I just downloaded": how recent "just" has to be before the newest
# file is taken without asking. Twelve hours, not minutes -- a file
# downloaded before a morning lecture is still "the one I just downloaded"
# to the man who downloaded it at lunchtime. Older than this and the newest
# file is a guess, so the candidates are read back instead.
JUST_WINDOW_S = 12 * 3600.0
# Two files written within two minutes of each other are not "the newest"
# and "an older one" -- they arrived together, and separating them by
# timestamp is a coin flip. Ask.
TIE_WINDOW_S = 120.0

# Folder cues -> the root they name. "downloaded" is here as well as
# "downloads": "the PDF I just downloaded" names the folder by its verb.
FOLDERS: tuple[tuple[str, str], ...] = (
    ("desktop", "Desktop"),
    ("downloads", "Downloads"),
    ("download", "Downloads"),
    ("downloaded", "Downloads"),
    ("documents", "Documents"),
)
# Type cues -> the suffixes they allow. An entry with no suffixes ("file",
# "attachment") is recognised as a file NOUN without narrowing anything --
# that is what separates "send the file on my desktop" (a file request with
# no name) from "email Heather" (not a file request at all).
TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pdf", (".pdf",)),
    ("pdfs", (".pdf",)),
    ("spreadsheet", (".csv", ".xlsx", ".xls", ".ods", ".numbers")),
    ("excel", (".xlsx", ".xls", ".csv")),
    ("csv", (".csv",)),
    ("presentation", (".ppt", ".pptx", ".odp", ".key")),
    ("powerpoint", (".ppt", ".pptx")),
    ("slides", (".ppt", ".pptx", ".odp", ".key")),
    ("screenshot", (".png", ".jpg", ".jpeg")),
    ("photo", (".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp")),
    ("photos", (".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp")),
    ("picture", (".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp")),
    ("image", (".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp")),
    ("zip", (".zip", ".tar", ".gz", ".tgz", ".7z")),
    ("archive", (".zip", ".tar", ".gz", ".tgz", ".7z")),
    ("video", (".mp4", ".mov", ".mkv", ".webm")),
    ("audio", (".mp3", ".wav", ".m4a", ".flac")),
    ("recording", (".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mov")),
    ("word doc", (".doc", ".docx", ".odt", ".rtf")),
    ("word document", (".doc", ".docx", ".odt", ".rtf")),
    ("docx", (".docx", ".doc")),
    # Generic nouns: they say "this is about a file" and nothing more.
    ("file", ()),
    ("files", ()),
    ("document", ()),
    ("attachment", ()),
    ("doc", ()),
    ("thing", ()),
)
# A token that is already a FILE NAME ("lab_report.pdf", "Q3-2026.xlsx").
# It is protected from cue-stripping before anything else runs: "pdf" is a
# type cue and \bpdf\b matches inside "lab_report.pdf" (the dot is a word
# boundary), so without this the name he said exactly is torn into
# "lab_report." plus a type filter -- and an exact name stops matching
# itself.
_FILENAME_RX = re.compile(r"(?<![\w./~-])([\w][\w+\-]*\.[A-Za-z0-9]{1,8})(?![\w])")
_HOLD = "\x00%d\x00"
# "the latest one" -- an explicit instruction to rank by time.
_LATEST_RX = re.compile(r"\b(?:latest|most recent|newest|last)\b", re.I)
# "...I just downloaded" -- a claim about WHEN, which JUST_WINDOW_S checks.
_JUST_RX = re.compile(r"\b(?:just|a moment ago|a second ago|right now)\b", re.I)
# Grammar, not name. Everything here is removed before the leftover is
# handed to filepick.score as the name he said.
_FILLER = frozenset("""
a an the that this these those it my mine our your his her their one ones
i i've ive we you he she they me please sir jarvis on in at from to for of
with by about as is was be been am are were do does did got get and or
just now then there here over new old saved stored sitting called named
titled downloaded uploaded made created wrote put left recent recently
""".split())
# Never offered as a candidate whatever he said -- filepick's own list, so
# ~/.ssh and ~/.config are out of reach from both lanes by one rule.
SKIP_DIRS = filepick.SKIP_DIRS

# An EXPLICIT path is the one thing this lane treats differently from
# filepick, and it is a requirement, not an oversight: he asked for "email
# this file from this location", and a path he gave outright must work even
# when the location is not one of the three search folders. filepick refuses
# those outright (rightly -- a spoken NAME must never reach outside the
# roots), so the containment test is REPLACED here rather than dropped:
#
#   * nothing whose path contains a dot-component. ~/.ssh/id_ed25519,
#     ~/.gnupg, ~/.config and every credential file in this account live
#     behind a leading dot, and none of them is a document.
#   * nothing under a system tree. /etc/shadow resolves perfectly well and
#     is the exact sentence this deny-list exists to refuse.
#
# What is left is what he meant: ~/projects/thesis.pdf, /data/scans/x.png.
DENY_ROOTS = ("/etc", "/proc", "/sys", "/dev", "/boot", "/root", "/run",
              "/var", "/usr", "/bin", "/sbin", "/lib", "/opt", "/srv")


def denied(path: Path) -> bool:
    """True when an explicit path may not be used however plainly he said it."""
    try:
        real = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return True
    if any(part.startswith(".") and part not in (".", "..")
           for part in real.parts):
        return True
    text = str(real)
    return any(text == root or text.startswith(root + "/")
               for root in DENY_ROOTS)


@dataclass
class Phrase:
    """What a spoken file description actually specifies."""
    name: str = ""                       # the leftover, "" when he named none
    folder: str = ""                     # "Desktop" / "Downloads" / ...
    suffixes: tuple = ()                 # () = any type
    latest: bool = False                 # "the latest", "most recent"
    just: bool = False                   # "I just downloaded"
    noun: bool = False                   # a file word was said at all
    literals: tuple = ()                 # file names he said outright

    @property
    def has_cue(self) -> bool:
        return bool(self.folder or self.suffixes or self.latest or self.just)


@dataclass
class Match:
    """One path, several candidates, or a refusal.

    The same three-way branch as :class:`filepick.Pick`, plus ``size`` --
    which the "too-big" line needs and which a caller would otherwise have
    to re-stat off a path the refusal did not hand back.
    """
    path: Optional[Path] = None
    candidates: list = field(default_factory=list)
    reason: str = ""
    size: int = 0

    @property
    def ok(self) -> bool:
        return self.path is not None

    @property
    def ambiguous(self) -> bool:
        return self.path is None and len(self.candidates) > 1


# ------------------------------------------------------------- parsing
def _strip_cue(text: str, cue: str) -> tuple[str, bool]:
    """Remove a whole-word cue; (text, whether it was there)."""
    rx = re.compile(rf"\b{re.escape(cue)}\b", re.I)
    if not rx.search(text):
        return text, False
    return rx.sub(" ", text), True


def parse(said: str) -> Phrase:
    """Pull the folder, the type, the recency and the leftover NAME apart.

    Order matters: the longest type cues are consumed first ("word document"
    before "document"), and the folder before the filler, so "on my desktop"
    leaves nothing behind to be scored as a name.
    """
    text = " ".join(str(said or "").split())
    if not text:
        return Phrase()
    ph = Phrase()
    # Park any literal file name before the cue tables can chew on it.
    held: list[str] = []

    def _park(m):
        held.append(m.group(1))
        return _HOLD % (len(held) - 1)

    text = _FILENAME_RX.sub(_park, text)
    ph.latest = bool(_LATEST_RX.search(text))
    ph.just = bool(_JUST_RX.search(text))

    for cue, folder in FOLDERS:
        text, hit = _strip_cue(text, cue)
        if hit and not ph.folder:
            ph.folder = folder
    suffixes: list[str] = []
    for cue, sufs in sorted(TYPES, key=lambda kv: -len(kv[0])):
        text, hit = _strip_cue(text, cue)
        if hit:
            ph.noun = True
            for s in sufs:
                if s not in suffixes:
                    suffixes.append(s)
    ph.suffixes = tuple(suffixes)
    text = _LATEST_RX.sub(" ", text)
    text = _JUST_RX.sub(" ", text)
    words = [w for w in re.split(r"\s+", text) if w]
    kept = [w for w in words if w.lower().strip(".,;:'\"") not in _FILLER]
    name = " ".join(kept).strip(" .,;:'\"")
    for i, literal in enumerate(held):
        name = name.replace(_HOLD % i, literal)
    ph.name = " ".join(name.split())
    ph.literals = tuple(held)
    return ph


# ---------------------------------------------------------------- scan
def _scan(roots: Sequence[Path], depth: int) -> list[Path]:
    """Files under ``roots``, bounded, hidden and SKIP_DIRS excluded.

    Symlinks are not offered (``follow_symlinks=False``), which is
    filepick's rule and the first of its two containment layers: a Desktop
    symlink at ~/.ssh/id_ed25519 can never become a candidate. The second
    layer -- ``filepick.check_file`` on the winner -- runs in :func:`_final`.
    """
    out: list[Path] = []
    for root in roots:
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            folder, level = stack.pop()
            try:
                entries = list(os.scandir(folder))
            except OSError:
                log.debug("filephrase: cannot list %s", folder, exc_info=True)
                continue
            for e in entries:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir(follow_symlinks=False):
                        if level < depth and e.name not in SKIP_DIRS:
                            stack.append((Path(e.path), level + 1))
                    elif e.is_file(follow_symlinks=False):
                        out.append(Path(e.path))
                except OSError:
                    continue
    return out


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _final(path: Path, roots: Sequence[Path], max_mb: float) -> Match:
    """The last gate: filepick's own verdict on a resolved path."""
    why = filepick.check_file(path, roots, max_mb)
    size = 0
    try:
        size = path.stat().st_size
    except OSError:
        pass
    if why:
        log.info("filephrase: refusing %s (%s)", path.name, why)
        return Match(reason=why, size=size)
    return Match(path=path.resolve(), size=size)


# ------------------------------------------------------------- resolve
def resolve(said: str, roots: Optional[Sequence[str]] = None,
            max_mb: float = filepick.DEFAULT_MAX_MB,
            depth: int = filepick.DEFAULT_DEPTH,
            now: Optional[float] = None,
            allow_explicit_outside: bool = True) -> Match:
    """The one file ``said`` means, the rivals, or why there is none.

    ``said`` is the FILE half of the request only ("that file on my
    desktop"), never the whole sentence: the recipient's name would
    otherwise be scored as part of the file name and match heather.pdf.
    """
    text = " ".join(str(said or "").split()).strip("'\"")
    now = time.time() if now is None else float(now)
    root_paths = filepick.expand_roots(roots)
    if not root_paths:
        return Match(reason="not-found")
    if not text:
        return Match(reason="empty")

    # An explicit path. Inside the roots it is filepick's verdict verbatim,
    # so the two lanes cannot differ by a character. Outside them this lane
    # allows it and the other does not -- he asked for "this file from this
    # location", and DENY_ROOTS is what replaces the containment test there.
    # Every other leg of check_file (regular file, readable, size) runs
    # either way.
    if text.startswith(("/", "~", "./", "../")):
        try:
            real = Path(os.path.expanduser(text)).resolve()
        except (OSError, RuntimeError):
            return Match(reason="not-found")
        if filepick.check_file(real, root_paths, max_mb) != "outside":
            return _final(real, root_paths, max_mb)
        if not allow_explicit_outside or denied(real):
            log.warning("filephrase: refusing the explicit path %r", text)
            return Match(reason="outside")
        # Containment is satisfied by the fact that he named this exact
        # path, so the file's own folder stands in as the root and the
        # remaining legs of check_file are the ones that have to pass.
        return _final(real, [real.parent], max_mb)

    ph = parse(text)
    if not ph.name and not ph.has_cue:
        # "email that to Heather": no name, no folder, no type, no
        # recency. There is nothing here to resolve and nothing to rank.
        return Match(reason="empty")

    here = root_paths
    if ph.folder:
        here = [r for r in root_paths if r.name.lower() == ph.folder.lower()]
        if not here:
            log.info("filephrase: no %s folder on this box", ph.folder)
            return Match(reason="not-found")

    files = _scan(here, depth)
    if ph.suffixes:
        files = [f for f in files if f.suffix.lower() in ph.suffixes]
    if not files:
        return Match(reason="not-found")

    if ph.name:
        # A name he said EXACTLY is the answer, whatever else is close. The
        # wide tie band above exists for half-remembered names; a literal
        # "lab_report.pdf" is not half-remembered, and letting
        # lab_report_final.pdf tie with it would turn the one unambiguous
        # phrasing he has into a question.
        for literal in (ph.literals or (ph.name,)):
            exact = [f for f in files if f.name.lower() == literal.lower()]
            if len(exact) == 1:
                return _final(exact[0], root_paths, max_mb)
        scored = [(filepick.score(ph.name, f.name), f) for f in files]
        scored = [(s, f) for s, f in scored if s >= filepick.FUZZY_FLOOR]
        if not scored:
            return Match(reason="not-found")
        scored.sort(key=lambda t: (-t[0], len(t[1].name), str(t[1])))
        top = scored[0][0]
        tied = [f for s, f in scored if s >= top - TIE_SCORE]
        if len(tied) > 1:
            return Match(candidates=tied[:filepick.MAX_CANDIDATES])
        return _final(scored[0][1], root_paths, max_mb)

    # No name: the folder, the type and the recency words are the whole
    # handle, so the only safe resolutions are "there is exactly one" and
    # "he asked for the newest and one is plainly newest".
    files.sort(key=_mtime, reverse=True)
    if len(files) == 1:
        return _final(files[0], root_paths, max_mb)
    newest, runner = files[0], files[1]
    fresh = (now - _mtime(newest)) <= JUST_WINDOW_S if ph.just else True
    clear = (_mtime(newest) - _mtime(runner)) > TIE_WINDOW_S
    if (ph.latest or ph.just) and fresh and clear:
        return _final(newest, root_paths, max_mb)
    return Match(candidates=files[:filepick.MAX_CANDIDATES])
