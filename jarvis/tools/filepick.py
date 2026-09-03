"""Turn a SPOKEN file name into ONE exact path, or a question.

This module exists because "email this file to Dana" and "put that on
HPCOMPUTER" are both irreversible, and both start from a phrase that is not
a path.  Getting from "the budget thing on my desktop" to
``/home/hunterp/Desktop/budget-2026.xlsx`` is the whole risk; SMTP and scp
are the easy half.

**It is deliberately shared.**  The mail lane (email-a-file) and the remote
lane (files to and from HPCOMPUTER) resolve the same phrase against the same
folders and must not disagree about what "that file" means -- two resolvers
would drift, and the day they drift is the day one of them attaches the
wrong file.  Nothing here imports a transport, so either lane can use it.

Four rules, each of which is a way this can go wrong out loud:

1. **Ambiguity ASKS -- it never guesses.**  ``pick()`` returns every close
   match, not a best one.  Two files called "notes" is the normal case in a
   Downloads folder, and picking the newer one silently is how the wrong
   file gets sent to someone.  A caller with more than one candidate must
   read the names back.

2. **A miss is a miss.**  A half-remembered name that matches nothing
   scores nothing; ``FUZZY_FLOOR`` is a floor, not a suggestion.  "I can't
   find it" is a correct answer and the only safe one -- there is no
   threshold at which a wrong file becomes acceptable.

3. **The path must land inside a root he meant**, and that is enforced
   TWICE, on purpose.  ``_walk`` never offers a symlink as a candidate at
   all (``is_file(follow_symlinks=False)``), so a Desktop symlink pointing
   at ``~/.ssh/id_ed25519`` cannot be reached by a spoken name; and
   ``check_file`` compares the ``resolve()``d path against the
   ``resolve()``d roots, so an EXPLICIT path he said is refused by the same
   test that refuses ``../../``.  Either layer alone would leave a hole --
   the first misses paths he says outright, the second misses nothing but
   only runs on a candidate that already exists.  A name that traverses out
   is not an error to report and move past; it is the attack.

4. **Directories and specials are refused.**  ``send me the Desktop`` must
   not become a recursive copy of a folder, and a fifo would hang the
   transfer for its whole budget.

Nothing here reads file CONTENT.  Resolving a name must never be the thing
that loads a 2 GB video into memory.
"""
from __future__ import annotations

import difflib
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from jarvis.logs import get_logger

log = get_logger("tools.filepick")

# Where a spoken name is allowed to resolve.  Order matters: it is the order
# he would look himself, and it decides the winner when the SAME name sits
# in two roots ("the invoice" in Desktop AND Downloads is reported as
# ambiguous, but Desktop is listed first because that is where he said it).
DEFAULT_ROOTS = ("~/Desktop", "~/Documents", "~/Downloads")

# How far under a root to walk.  1 = the root's own entries plus one level
# of subfolders.  A spoken title must not reach a stray "notes.txt" six
# levels down a git checkout -- and an unbounded walk of ~ is also seconds
# of stat() in the middle of a voice turn.
DEFAULT_DEPTH = 1

# Below this a difflib ratio is noise rather than a memory of a name.
FUZZY_FLOOR = 0.62

# ...but a ratio ALONE cannot make this decision, and measuring it is the
# only way to see why.  On this box:
#
#     "budget" vs "gadget"          0.667      <-- the WRONG file
#     "budget" vs "budget 2026"     0.706
#     "report" vs "reports final"   0.632
#
# The wrong answer sits in the MIDDLE of the right ones, so there is no
# floor that separates them: words differing by one letter score as well as
# a genuine abbreviation.  That is exactly the mistake a mis-heard syllable
# makes, and "send budget.xlsx" turning into gadget.png is the failure this
# whole module exists to prevent.
#
# So a fuzzy match must ALSO be plausible as the same word: it shares a real
# prefix with the name, or every word he said appears whole inside it.
# "budget"/"budget-2026" shares six characters; "budget"/"gadget" shares
# none and is refused, which leaves "I can't find it" -- the correct answer.
MIN_PREFIX = 3

# More than this and reading them back is a list, not a question.  He gets
# the closest few and can say a fuller name.
MAX_CANDIDATES = 4

# Entries never offered, whatever he said.  Dotfiles are configuration and
# credentials, not documents, and "send my ssh key" should have to be typed
# by a human being who means it.
SKIP_DIRS = {".git", ".cache", "node_modules", "__pycache__", ".venv",
             "venv", ".ssh", ".gnupg", ".config", ".local"}

# A file this big is not something to put on the wire inside a voice turn.
DEFAULT_MAX_MB = 100


@dataclass
class Pick:
    """The outcome of resolving one spoken name.

    Exactly one of these is true, and the caller must branch on all three:
    ``path`` (one match -- go, after read-back), ``candidates`` (several --
    ASK), or neither (nothing -- say so).  ``reason`` explains an empty
    result so the spoken line can be specific instead of "I can't find it".
    """
    path: Optional[Path] = None
    candidates: list[Path] = field(default_factory=list)
    reason: str = ""              # "" on success, else a REASONS key

    @property
    def ok(self) -> bool:
        return self.path is not None

    @property
    def ambiguous(self) -> bool:
        return self.path is None and len(self.candidates) > 1


REASONS = {
    "empty": "I didn't catch which file, sir.",
    "not-found": "I can't find a file by that name, sir.",
    "outside": "That path leads outside the folders I look in, sir; "
               "I'd rather you moved it somewhere I can see.",
    "not-a-file": "That's a folder, sir, not a file.",
    "unreadable": "I can't read that file, sir.",
    "too-big": "That file is {mb:.0f} megabytes, sir -- past the {cap} "
               "I'll put on the wire without you saying so plainly.",
}


def expand_roots(roots: Optional[Sequence[str]] = None) -> list[Path]:
    """The search roots as absolute, existing, de-duplicated paths."""
    out: list[Path] = []
    for raw in (roots if roots is not None else DEFAULT_ROOTS):
        try:
            p = Path(os.path.expanduser(str(raw))).resolve()
        except (OSError, RuntimeError):
            continue
        if p.is_dir() and p not in out:
            out.append(p)
    return out


def _within(path: Path, roots: Sequence[Path]) -> bool:
    """Is ``path`` genuinely inside one of ``roots``?

    ``Path.resolve()`` on BOTH sides first, so this answers the question
    about the file the OS would actually open, not the string he said.  A
    Desktop symlink to ~/.ssh resolves out of the roots here and is refused
    -- which is the point; ``..`` is only the obvious half of the attack.
    """
    try:
        real = path.resolve()
    except (OSError, RuntimeError):
        return False
    for root in roots:
        try:
            real.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _walk(root: Path, depth: int) -> Iterable[Path]:
    """Files under ``root``, at most ``depth`` folders down, bounded."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        d, level = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            log.debug("filepick: cannot list %s", d, exc_info=True)
            continue
        for e in entries:
            name = e.name
            if name.startswith("."):
                continue
            try:
                if e.is_dir(follow_symlinks=False):
                    if level < depth and name not in SKIP_DIRS:
                        stack.append((Path(e.path), level + 1))
                elif e.is_file(follow_symlinks=False):
                    yield Path(e.path)
            except OSError:
                continue


def _norm(text: str) -> str:
    """A spoken name flattened for comparison: case, punctuation and the
    separator zoo ("budget_2026", "budget-2026", "budget 2026") all collapse,
    because he says one of them and the file is named another."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _prefix_len(a: str, b: str) -> int:
    """How many leading characters two normalised names share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def score(said: str, name: str) -> float:
    """How well a spoken name matches a real filename, 0..1.

    The stem is tried as well as the full name so "the budget" matches
    "budget.xlsx" without him having to say the extension, and a said name
    that is a whole WORD inside the filename is treated as a strong match --
    difflib alone rates "budget" against "2026-budget-final" poorly, and
    that is exactly the name his files have.
    """
    s, n = _norm(said), _norm(name)
    if not s or not n:
        return 0.0
    if s == n:
        return 1.0
    stem = _norm(Path(name).stem)
    if s == stem:
        return 1.0
    # Whole-word containment: "budget" in "2026 budget final".  Strong on its
    # own, and it needs no prefix -- the word IS there.
    said_words = s.split()
    name_words = set(n.split())
    if said_words and all(w in name_words for w in said_words):
        return 0.90
    # Otherwise it has to look like the same word, not merely score like it.
    want = min(MIN_PREFIX, len(s))
    if _prefix_len(s, n) < want and _prefix_len(s, stem) < want:
        return 0.0
    return max(difflib.SequenceMatcher(None, s, n).ratio(),
               difflib.SequenceMatcher(None, s, stem).ratio())


def check_file(path: Path, roots: Sequence[Path],
               max_mb: float = DEFAULT_MAX_MB) -> str:
    """"" if ``path`` is a real, readable, in-bounds, sane-sized FILE, else
    the REASONS key that says why not.  Every caller runs this before a
    transfer, including one that was handed an absolute path."""
    if not _within(path, roots):
        return "outside"
    try:
        st = path.stat()                    # follows symlinks, on purpose
    except OSError:
        return "not-found"
    if stat.S_ISDIR(st.st_mode):
        return "not-a-file"
    if not stat.S_ISREG(st.st_mode):
        return "not-a-file"                 # fifo/socket/device: never
    if not os.access(path, os.R_OK):
        return "unreadable"
    if max_mb and st.st_size > max_mb * 1024 * 1024:
        return "too-big"
    return ""


def pick(said: str, roots: Optional[Sequence[str]] = None,
         depth: int = DEFAULT_DEPTH,
         max_mb: float = DEFAULT_MAX_MB) -> Pick:
    """Resolve a spoken file name to ONE path, several candidates, or nothing.

    An absolute or ~ path he actually said is honoured, but still has to
    pass ``check_file`` -- "send /etc/shadow" resolves perfectly well and is
    refused for being outside the roots, which is the only reason the
    containment test is applied to explicit paths too.
    """
    said = (said or "").strip().strip("'\"").strip()
    if not said:
        return Pick(reason="empty")
    root_paths = expand_roots(roots)
    if not root_paths:
        return Pick(reason="not-found")

    # An explicit path: he said a path, so take it literally -- but check it.
    if said.startswith(("/", "~")) or said.startswith("./"):
        p = Path(os.path.expanduser(said))
        why = check_file(p, root_paths, max_mb)
        return Pick(path=p.resolve()) if not why else Pick(reason=why)

    scored: list[tuple[float, Path]] = []
    seen: set[Path] = set()
    for root in root_paths:
        for f in _walk(root, depth):
            try:
                real = f.resolve()
            except (OSError, RuntimeError):
                continue
            if real in seen:
                continue
            seen.add(real)
            s = score(said, f.name)
            if s >= FUZZY_FLOOR:
                scored.append((s, f))
    if not scored:
        return Pick(reason="not-found")

    scored.sort(key=lambda t: (-t[0], len(t[1].name), str(t[1])))
    top = scored[0][0]
    # Everything within a hair of the best is a genuine rival.  Two files
    # that score the same must never be silently ordered by mtime.
    tied = [p for s, p in scored if s >= top - 0.02]
    if len(tied) > 1:
        return Pick(candidates=tied[:MAX_CANDIDATES])

    winner = scored[0][1]
    why = check_file(winner, root_paths, max_mb)
    if why:
        return Pick(reason=why)
    return Pick(path=winner.resolve())


def describe(paths: Sequence[Path]) -> str:
    """Candidate names as a spoken clause: "budget.xlsx or budget-final.xlsx"."""
    names = [p.name for p in paths]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]


def reason_line(reason: str, max_mb: float = DEFAULT_MAX_MB,
                size_mb: float = 0.0) -> str:
    """The spoken refusal for a ``Pick.reason``."""
    line = REASONS.get(reason) or REASONS["not-found"]
    if "{" in line:
        return line.format(mb=size_mb, cap=f"{max_mb:.0f} megabytes")
    return line
