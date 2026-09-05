"""The address book: the ONE place a spoken name becomes an address.

"Email this to Heather" ended, every time, at "I've no address for Heather,
sir. What is it?" -- ``send_file.contacts`` had no entries and the people
book had no email fields. He said he could supply the book, so this is the
area to put it in: one hand-editable JSON file beside the config,

    ~/.config/jarvis/contacts.json        (env JARVIS_CONTACTS overrides)

with three ways in -- his editor, ``scripts/jarvis_contacts.py`` over ssh
with Jarvis down, and a page on the phone client -- and ONE way out: the
send lane READS it. Nothing spoken can write it. A misheard address stored
is the typo waiting to be mailed, so the book is his to type.

Sending a document to the wrong person cannot be undone, and this module
is shaped by that one fact:

* matching is EXACT. What he said is lowercased, its dots dropped, its
  whitespace collapsed, a leading my/our/the stripped -- and then it has
  to EQUAL a row's full name, honorific + full name, first name, surname
  or an alias. No prefix, no edit distance, no scorer of any kind. "Heathr"
  resolves to nobody; two Heathers resolve to a QUESTION, never to the
  first one. An exact full name wins outright even when another row shares
  the first name.
* addresses are checked on ENTRY for SHAPE: one @; a local part with no
  leading, trailing or doubled dot; domain labels of 1-63 letters, digits
  or hyphens that do not start or end with a hyphen; a last label of two
  or more letters; at most 254 characters; a full match, never a search.
  A row that fails on load is skipped for resolution and flagged. What
  the check cannot do is know that "gmail.con" is wrong: it is
  well-formed, and the READ-BACK is the last check for that.
* a file that cannot be read -- a trailing comma from a hand edit, the
  wrong format, a permission -- is never written over: the last good rows
  stay for RESOLVING, and every add/remove refuses until it is fixed by
  hand. Writing the last good rows back would be writing his edit away.
* a one-word name ("Heather", "Mum") that is also another row's first
  name, surname or honorific + surname is refused on add, and on load
  both rows are kept and flagged: the name resolves as a QUESTION, never
  a pick.
* the file is his: 0600, directory 0700, written atomically, unknown keys
  kept on rewrite so a hand edit is never thrown away, and re-read by
  stamp on every resolve so an edit is live on the next send with no
  restart.

The file:

    {
      "format": 1,
      "contacts": [
        {"name": "Heather Smith", "email": "heather@example.com",
         "honorific": "Dr", "aliases": ["my advisor"], "note": "PhD advisor"},
        {"name": "Heather Jones", "email": "hjones@example.com"},
        {"name": "Mum", "email": "linda@example.com", "aliases": ["mom"]}
      ]
    }

Row order is the order a "Which Heather?" question lists them in.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from jarvis import assistant_config
from jarvis.logs import get_logger

log = get_logger("contacts")

FORMAT = 1
ENV_VAR = "JARVIS_CONTACTS"
FILE_NAME = "contacts.json"

NAME_MAX = 80
EMAIL_MAX = 254
HONORIFIC_MAX = 12
ALIAS_MAX = 40
NOTE_MAX = 200
# How many rivals a "Which Heather?" question will name out loud. Past
# this it asks for the full name instead of reading a list.
WHICH_MAX = 4

# A FULL match, not a search: parse_address in outbox searches inside
# text and would accept "heather@example.com junk" as an address. Local
# part: address characters with no leading, trailing or doubled dot.
# Domain: labels of 1-63 letters/digits/hyphens, no hyphen at either end,
# at least two of them, the last one letters only and 2+ long. "a@b.c",
# "h@1.2", "x@-.-", "h@example.com-" all fail; "heather@gmail.con" passes,
# because it IS an address -- only the read-back can catch that one.
_LOCAL = r"[A-Za-z0-9_%+\-]+(?:\.[A-Za-z0-9_%+\-]+)*"
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
EMAIL_RX = re.compile(rf"{_LOCAL}@(?:{_LABEL}\.)+[A-Za-z]{{2,}}")
BAD_ADDRESS = ("bad address: name@example.com shape -- one @, no spaces, no "
               "dot at either end of the name or doubled, a domain of "
               "letters/digits/hyphens ending in letters")
# What a write path says while the file cannot be read. The CLI prefixes
# "REFUSED: "; the page sends it whole as a 409.
UNREADABLE_LINE = "contacts.json {what} ({path}) — fix it by hand first"
_LEAD_RX = re.compile(r"^(?:my|our|the)\s+")
_TAIL_PUNCT = ".,;:?!"
_NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six",
                 "seven", "eight", "nine")


# ------------------------------------------------------------- the path
def book_path() -> Path:
    """Beside the config: ``config_path().parent / contacts.json``.

    tests/conftest.py points JARVIS_ASSISTANT_CONFIG at a throwaway file,
    so the suite is off his real book for free; JARVIS_CONTACTS overrides
    outright for a test that wants its own.
    """
    raw = os.environ.get(ENV_VAR) or ""
    if raw:
        return Path(os.path.expanduser(raw))
    return assistant_config.config_path().parent / FILE_NAME


def display_path(path: Optional[Path] = None) -> str:
    """"~/.config/jarvis/contacts.json" -- for the CLI trailer and the page."""
    p = Path(path) if path is not None else book_path()
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


# ------------------------------------------------------------ normalise
def normalise(said) -> str:
    """The one spelling every comparison is made in.

    Lowercase, dots dropped ("Dr." / "J. R."), whitespace collapsed,
    trailing punctuation stripped, and a leading my/our/the stripped so
    "my advisor" and "advisor" are one key. Nothing else: this is a
    spelling, not a fuzz.
    """
    text = " ".join(str(said or "").replace(".", " ").split()).lower()
    text = text.rstrip(_TAIL_PUNCT).strip()
    text = _LEAD_RX.sub("", text).strip()
    return text


def _clean_text(value, limit: int) -> tuple[str, str]:
    """(stripped value, why) -- a one-line string of at most ``limit``."""
    if value is None:
        return "", ""
    if not isinstance(value, str):
        return "", "must be text"
    text = " ".join(value.split())
    if len(text) > limit:
        return "", f"longer than {limit} characters"
    return text, ""


def _name_chars_ok(text: str) -> bool:
    return all(ch.isalpha() or ch.isdigit() or ch in " '-." for ch in text)


# ----------------------------------------------------------------- rows
@dataclass
class Contact:
    """One validated row. ``extra`` is every key this module does not
    know, carried so a hand edit survives a rewrite."""
    name: str
    email: str
    honorific: str = ""
    aliases: list = field(default_factory=list)
    note: str = ""
    extra: dict = field(default_factory=dict)
    index: int = -1                  # position in the file, for the CLI

    @property
    def full(self) -> str:
        return normalise(self.name)

    @property
    def first(self) -> str:
        return self.full.split()[0] if self.full else ""

    @property
    def surname(self) -> str:
        return self.full.split()[-1] if self.full else ""

    @property
    def spoken(self) -> str:
        """"Dr Heather Smith" -- what the read-back says."""
        return f"{self.honorific} {self.name}".strip()

    def name_keys(self) -> set:
        """The spellings that are THIS PERSON'S NAME: full, first, surname,
        and each with the honorific in front."""
        keys = {self.full, self.first, self.surname}
        hon = normalise(self.honorific)
        if hon:
            keys.add(f"{hon} {self.full}")
            keys.add(f"{hon} {self.surname}")
        return {k for k in keys if k}

    def exact_keys(self) -> set:
        """The spellings that are this person's WHOLE name: the full name
        and honorific + full name. Saying one of these is saying all of
        it, so it wins outright over a rival who merely shares a first
        name or a surname -- unless the whole name IS a rival's first name
        or surname (the flagged one-word collision, "Heather" beside
        "Heather Jones"). Then "Heather" and "Dr Heather" alike are the
        question itself: the Dr adds nothing the other row denies."""
        keys = {self.full}
        hon = normalise(self.honorific)
        if hon:
            keys.add(f"{hon} {self.full}")
        return {k for k in keys if k}

    def alias_keys(self) -> set:
        return {normalise(a) for a in self.aliases if normalise(a)}

    def keys(self) -> set:
        return self.name_keys() | self.alias_keys()

    def row(self) -> dict:
        out = {"name": self.name, "email": self.email}
        if self.honorific:
            out["honorific"] = self.honorific
        if self.aliases:
            out["aliases"] = list(self.aliases)
        if self.note:
            out["note"] = self.note
        for key, value in self.extra.items():
            out.setdefault(key, value)
        return out

    def public(self) -> dict:
        """What the page and ``list --json`` show."""
        return {"name": self.name, "email": self.email,
                "honorific": self.honorific, "aliases": list(self.aliases),
                "note": self.note}


@dataclass
class Skipped:
    index: int
    name: str
    why: str


def validate_row(raw) -> tuple[Optional[Contact], str]:
    """(Contact, "") for a good row, (None, why) for a bad one.

    THE validator -- the CLI, the page and the loader all come through
    here, so there is one rule for what an address is. Stricter than the
    reader on purpose: a row is typed once and mailed to many times.
    """
    if not isinstance(raw, dict):
        return None, "not an object"
    name, why = _clean_text(raw.get("name"), NAME_MAX)
    if why:
        return None, f"name: {why}"
    if not name:
        return None, "name is missing"
    if "@" in name:
        return None, "name: that is an address, in the wrong box"
    if not _name_chars_ok(name):
        return None, ("name: letters, digits, spaces, apostrophes, hyphens "
                      "and full stops only")
    if not any(ch.isalpha() for ch in name):
        return None, "name: needs a letter in it"
    email, why = _clean_text(raw.get("email"), EMAIL_MAX)
    if why:
        return None, f"email: {why}"
    if not email:
        return None, "email is missing"
    if not EMAIL_RX.fullmatch(email):
        return None, BAD_ADDRESS
    honorific, why = _clean_text(raw.get("honorific"), HONORIFIC_MAX)
    if why:
        return None, f"honorific: {why}"
    if honorific and not _name_chars_ok(honorific):
        return None, "honorific: letters and full stops only"
    note, why = _clean_text(raw.get("note"), NOTE_MAX)
    if why:
        return None, f"note: {why}"
    aliases_raw = raw.get("aliases")
    if aliases_raw is None:
        aliases_raw = []
    if isinstance(aliases_raw, str):
        aliases_raw = [aliases_raw]
    if not isinstance(aliases_raw, list):
        return None, "aliases: must be a list"
    aliases = []
    for item in aliases_raw:
        alias, why = _clean_text(item, ALIAS_MAX)
        if why:
            return None, f"alias: {why}"
        if not alias:
            continue
        if "@" in alias:
            return None, "alias: that is an address, in the wrong box"
        if not _name_chars_ok(alias):
            return None, ("alias: letters, digits, spaces, apostrophes, "
                          "hyphens and full stops only")
        if not normalise(alias):
            return None, f"alias {alias!r} is nothing once my/our/the is dropped"
        if alias not in aliases:
            aliases.append(alias)
    extra = {k: v for k, v in raw.items()
             if k not in ("name", "email", "honorific", "aliases", "note")}
    return Contact(name=name, email=email, honorific=honorific,
                   aliases=aliases, note=note, extra=extra), ""


def _duplicate(existing, new: Contact) -> str:
    """The SKIP class: a second row that is the same person. On load the
    later row is dropped."""
    new_full = new.full
    new_email = new.email.lower()
    for other in existing:
        if other is new:
            continue
        if other.full == new_full:
            return f"already in the book as {other.name}"
        if other.email.lower() == new_email:
            return f"that address is already in the book as {other.name}"
    return ""


def _alias_clash(existing, new: Contact) -> str:
    """An alias that is someone else's name or alias, or a name that is
    someone else's alias. A REFUSAL on add (check_unique); on load the
    alias alone is trimmed and both people kept (Book._install), because a
    nickname he typed for one person must never cost him another."""
    new_aliases = new.alias_keys()
    new_names = new.name_keys()
    for other in existing:
        if other is new:
            continue
        clash = new_aliases & other.keys()
        if clash:
            return (f"alias {sorted(clash)[0]!r} is already how {other.name} "
                    f"is known")
        clash = new_names & other.alias_keys()
        if clash:
            return (f"{sorted(clash)[0]!r} is already an alias of "
                    f"{other.name}")
    return ""


def _how_known(contact: Contact, key: str) -> str:
    """"first name" / "surname" / "Dr + surname" -- how ``key`` is one of
    ``contact``'s own name keys."""
    if key == contact.full:
        return "name"
    if key == contact.first and key != contact.surname:
        return "first name"
    if key == contact.surname:
        return "surname"
    hon = normalise(contact.honorific)
    if hon and key == f"{hon} {contact.surname}":
        return f"{contact.honorific} + surname"
    return "name"


def name_collision(a: Contact, b: Contact) -> str:
    """Why ``a``'s FULL name is also one of ``b``'s name keys (or the other
    way round), or "". "Heather" beside "Heather Jones": a plain "Heather"
    would draft to the one-word row outright, read back "to Heather" --
    which is what he said -- and never ask. So it is a QUESTION on load
    and a refusal on add. Two multi-word names sharing a first name or a
    surname are not this: "Heather Smith" and "Heather Jones" already ask.
    """
    if a.full in b.name_keys():
        return f"{a.name!r} is also {b.name}'s {_how_known(b, a.full)}"
    if b.full in a.name_keys():
        return f"{b.name!r} is also {a.name}'s {_how_known(a, b.full)}"
    return ""


def check_unique(existing, new: Contact) -> str:
    """Why ``new`` may not join ``existing``, or "".

    No two rows with one full name (the "which" question could not be
    answered); no two rows with one address (it is one person twice); an
    alias may not equal any other row's name, first name, surname or
    alias -- it would make a plain first name ambiguous with a nickname;
    and a row's whole name may not be another row's first name, surname
    or honorific + surname ("Heather" beside "Heather Jones", "Dr Smith"
    beside Dr Heather Smith) -- a one-word name has to be unique.
    """
    why = _duplicate(existing, new) or _alias_clash(existing, new)
    if why:
        return why
    for other in existing:
        if other is new:
            continue
        why = name_collision(new, other)
        if why:
            return why + " — a one-word name has to be unique in the book"
    return ""


# ----------------------------------------------------------- resolution
@dataclass
class Resolution:
    """What a spoken name came to. Exactly one of the three shapes:
    ``addr`` set (found), ``candidates`` set (ambiguous), neither (unknown).
    """
    addr: str = ""
    name: str = ""                   # the row's full name, when found
    candidates: list = field(default_factory=list)   # full names, file order
    from_book: bool = False
    honorific: str = ""
    matched_on: str = ""             # "full name" / "first name" / ...

    @property
    def found(self) -> bool:
        return bool(self.addr)

    @property
    def ambiguous(self) -> bool:
        return not self.addr and bool(self.candidates)

    @property
    def unknown(self) -> bool:
        return not self.addr and not self.candidates


def _matched_on(contact: Contact, key: str) -> str:
    if key == contact.full:
        return "full name"
    if key == contact.first:
        return "first name"
    if key == contact.surname:
        return "surname"
    if key in contact.alias_keys():
        return "alias"
    return "honorific and name"


def which_line(said: str, candidates) -> str:
    """"Which Heather, sir — Heather Smith or Heather Jones?"

    Up to WHICH_MAX names in file order; more than that and it asks for
    the full name rather than reading a list he cannot hold in his head.
    """
    names = [str(c) for c in (candidates or ())]
    who = " ".join(str(said or "").split()).strip(_TAIL_PUNCT) or "one"
    who = _LEAD_RX.sub("", who)
    if len(names) > WHICH_MAX:
        n = len(names)
        count = _NUMBER_WORDS[n] if n < len(_NUMBER_WORDS) else str(n)
        return (f"I've {count} people called {who}, sir — the full name, "
                f"please.")
    listed = ", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 \
        else names[0]
    return f"Which {who}, sir — {listed}?"


def reask_line(candidates) -> str:
    names = [str(c) for c in (candidates or ())]
    if len(names) > WHICH_MAX:
        return "The full name, sir?"
    listed = ", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 \
        else names[0]
    return f"The full name, sir — {listed}?"


# ----------------------------------------------------------------- file
def _write_private(path: Path, text: str) -> None:
    """Atomic 0600 write: mkstemp in the same directory (0600 by mkstemp),
    fsync, os.replace, chmod to be sure. The same dance as
    assistant_config and identity keep; a third copy rather than a shared
    module because consolidating the three is its own change.

    Beside the REAL file: os.replace over a symlink replaces the link
    itself, which is how a dotfiles-managed book was silently turned into
    a regular file on its first add (the target kept the old rows and a
    later hand edit of it was invisible). Resolved first, the write lands
    in the target's directory and the link stays a link; a dangling link
    gets its target created.
    """
    path = Path(os.path.realpath(path))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=".contacts-", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def _stamp(path: Path):
    """(mtime_ns, size, ctime_ns, inode). ctime is in it so a chmod -- which
    moves neither mtime nor size -- still counts as the file having
    changed; the inode so a rename-over (an editor's atomic save, the
    CLI's own write) is a change even when it lands inside the
    filesystem's time tick with the same size. A same-size rewrite IN
    PLACE inside that tick is the one edit this cannot see, and no hand
    lands one."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size, st.st_ctime_ns, st.st_ino)
    except OSError:
        return None


class Book:
    """The book as last read, and the stamp it was read at.

    ``current()`` is the door: it re-stats the file every time and re-reads
    when the (mtime_ns, size, ctime_ns, inode) stamp has moved -- one stat
    per send, per question answer, per correction. An edit by hand, by the CLI or by the
    page is live on the next resolve with no restart, and AssistantConfig
    stays load-once.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else book_path()
        self.rows: list = []             # raw rows, file order, bad ones too
        self.contacts: list = []         # the validated ones
        self.skipped: list = []          # bad rows: not used, kept in the file
        self.flagged: list = []          # collisions: used, but as a question
        self.trimmed: list = []          # aliases not used: the row is kept
        self.extra: dict = {}            # unknown top-level keys, kept
        # Why the file on disk cannot be read right now, or "". While it is
        # set the rows above are the LAST GOOD book, for resolving only:
        # add/remove refuse rather than write them back over his edit.
        self.broken: str = ""
        self.stamp = None
        self._warned_stamp = None
        self._loaded = False
        self._lock = threading.RLock()

    # ---------------------------------------------------------- loading
    def refresh(self) -> "Book":
        """Re-read if the file moved. Never raises."""
        with self._lock:
            stamp = _stamp(self.path)
            # A BROKEN file is re-read on every refresh, never short-
            # circuited on its stamp. Kernel file times are coarse (a tick
            # of ~1 ms here), so the fix he makes can land in the same tick
            # as the break (a chmod and its undo) or restore the exact bytes
            # in the same tick as the last good write -- and against either
            # stamp the fixed file compares "unchanged" and the book stays
            # broken with the file already fixed. Seen both ways in the
            # suite, where a whole test runs inside one tick. The cost is
            # one read per resolve while the book is broken -- the state
            # the warning asks him to end -- and the stamp still gates the
            # warning to once per change.
            if self._loaded and not self.broken and stamp == self.stamp:
                return self
            if stamp is None:
                # Missing file = empty book. Not an error, not a log line:
                # a fresh install has no book yet.
                self._install([], {}, None)
                return self
            try:
                # utf-8-sig: a BOM from a Windows editor is not a syntax
                # error. A 0-byte or whitespace-only file (`touch`) is what
                # a missing file is -- an empty, writable book -- and not
                # a broken one; anything else that is not JSON stays broken.
                text = self.path.read_text(encoding="utf-8-sig")
                raw = (json.loads(text) if text.strip()
                       else {"format": FORMAT, "contacts": []})
                if not isinstance(raw, dict):
                    raise ValueError("not an object")
                if raw.get("format") != FORMAT:
                    raise ValueError("format is not %d" % FORMAT)
                rows = raw.get("contacts")
                if rows is None:
                    rows = []
                if not isinstance(rows, list):
                    raise ValueError("contacts is not a list")
            except (OSError, ValueError) as exc:
                # Keep the last good book FOR RESOLVING; say so once per
                # stamp, and say only the PATH -- never a row. Every write
                # path reads ``broken`` and refuses until he has fixed it.
                if isinstance(exc, json.JSONDecodeError):
                    what = "is not valid JSON"
                elif isinstance(exc, OSError):
                    what = "could not be read"
                else:
                    what = f"is not a format-{FORMAT} address book"
                self.broken = UNREADABLE_LINE.format(what=what, path=self.path)
                if stamp != self._warned_stamp:
                    log.warning("contacts: %s could not be read (%s); keeping "
                                "the last good book for resolving, refusing "
                                "writes", self.path, exc.__class__.__name__)
                    self._warned_stamp = stamp
                self._loaded = True
                return self
            extra = {k: v for k, v in raw.items()
                     if k not in ("format", "contacts")}
            self._install(rows, extra, stamp)
            return self

    def _install(self, rows: list, extra: dict, stamp) -> None:
        contacts, skipped, flagged, trimmed = [], [], [], []
        for i, row in enumerate(rows):
            contact, why = validate_row(row)
            if contact is not None:
                why = _duplicate(contacts, contact)
            if contact is None or why:
                shown = ""
                if isinstance(row, dict):
                    shown = " ".join(str(row.get("name") or "").split())[:NAME_MAX]
                skipped.append(Skipped(index=i, name=shown, why=why))
                continue
            contact.index = i
            contacts.append(contact)
        # An alias that is another row's NAME (full, first, surname, with
        # or without the honorific), or an earlier row's alias: the alias
        # is not used and the row is. A name beats an alias in either
        # file order; between two aliases the first in the file keeps it.
        # In memory only -- the file keeps his alias (a rewrite carries
        # the raw rows), and the page/CLI list says which one was passed
        # over and why.
        for c in contacts:
            for alias in list(c.aliases):
                key = normalise(alias)
                owner = next((o for o in contacts if o is not c
                              and key in o.name_keys()), None)
                if owner is not None:
                    why = (f"alias {alias!r} not used: it is {owner.name}'s "
                           f"{_how_known(owner, key)}")
                else:
                    owner = next((o for o in contacts if o.index < c.index
                                  and key in o.alias_keys()), None)
                    if owner is None:
                        continue
                    why = (f"alias {alias!r} not used: it is already how "
                           f"{owner.name} is known")
                c.aliases.remove(alias)
                trimmed.append(Skipped(index=c.index, name=c.name, why=why))
        # A one-word name that is also someone's first name or surname:
        # BOTH rows stay (they are his, and each is a real person) and
        # both are flagged, because the name now resolves as a question.
        for a in contacts:
            for b in contacts:
                if a.index >= b.index:
                    continue
                why = name_collision(a, b)
                if why:
                    tail = " — resolves as a question, never a pick; make it unique"
                    flagged.append(Skipped(index=a.index, name=a.name, why=why + tail))
                    flagged.append(Skipped(index=b.index, name=b.name, why=why + tail))
        self.rows = list(rows)
        self.contacts = contacts
        self.skipped = skipped
        self.flagged = flagged
        self.trimmed = trimmed
        self.extra = extra
        self.broken = ""
        self.stamp = stamp
        self._loaded = True
        if skipped:
            # Names only, never an address.
            log.warning("contacts: %d row(s) in %s skipped: %s", len(skipped),
                        self.path, "; ".join(
                            f"#{s.index + 1} {s.name or '(no name)'}: {s.why}"
                            for s in skipped))
        if flagged:
            log.warning("contacts: %d row(s) in %s collide: %s", len(flagged),
                        self.path, "; ".join(
                            f"#{f.index + 1} {f.name}: {f.why}" for f in flagged))
        if trimmed:
            log.warning("contacts: %d alias(es) in %s not used: %s", len(trimmed),
                        self.path, "; ".join(
                            f"#{t.index + 1} {t.name}: {t.why}" for t in trimmed))

    # ------------------------------------------------------------- reads
    def resolve(self, said) -> Resolution:
        """The match rule. Exact, or a question, or nothing."""
        self.refresh()
        key = normalise(said)
        if not key:
            return Resolution()
        # An exact FULL name (or honorific + full name) wins outright even
        # when another row shares the first name: he said all of it --
        # UNLESS the whole name is also another row's first name or
        # surname (a flagged collision, "Heather" beside "Heather Jones").
        # Then it is the ambiguity itself, and a question -- said with the
        # honorific too ("Dr Heather"): the Dr is on the one-word row, and
        # nothing in the book says Heather Jones is not one.
        for c in self.contacts:
            if key in c.exact_keys():
                rivals = [o for o in self.contacts if o is not c
                          and (key in o.keys() or c.full in o.keys())]
                if rivals:
                    names = [o.name for o in self.contacts if o is c or o in rivals]
                    return Resolution(candidates=names, from_book=True)
                return Resolution(addr=c.email, name=c.name, from_book=True,
                                  honorific=c.honorific, matched_on="full name")
        hits = [c for c in self.contacts if key in c.keys()]
        if not hits:
            return Resolution()
        if len(hits) == 1:
            c = hits[0]
            return Resolution(addr=c.email, name=c.name, from_book=True,
                              honorific=c.honorific,
                              matched_on=_matched_on(c, key))
        return Resolution(candidates=[c.name for c in hits], from_book=True)

    def by_name(self, name: str) -> Optional[Contact]:
        """The row whose full name this is, exactly, or None."""
        self.refresh()
        key = normalise(name)
        for c in self.contacts:
            if c.full == key:
                return c
        return None

    def pick(self, name: str) -> Optional[Resolution]:
        """The answer to "Which Heather?" as a FOUND resolution: the row
        whose full name this is, by identity -- never back through
        resolve(), whose collision rule would ask the question again for
        a one-word row ("the first one" of "Heather or Heather Jones" is
        the row called Heather, not the name Heather). None when the row
        has gone since the list was read: the caller treats that as a
        miss rather than re-resolving the name to a different person."""
        c = self.by_name(name)
        if c is None:
            return None
        return Resolution(addr=c.email, name=c.name, from_book=True,
                          honorific=c.honorific, matched_on="full name")

    def choose(self, said, candidates) -> Optional[str]:
        """The answer to "Which Heather?": ONE of ``candidates`` (full
        names), named exactly, or None.

        Accepted: the exact full name; the surname, when it is unique
        among the candidates; honorific + full name or honorific +
        surname. Nothing else -- not a first name (that is the ambiguity
        being asked about), not a prefix, not a score. Ordinals are the
        commander's, since it owns that grammar for the file question.
        """
        self.refresh()
        key = normalise(said)
        if not key:
            return None
        wanted = [normalise(c) for c in (candidates or ())]
        rows = [c for c in self.contacts if c.full in wanted]
        # ONE of the rivals may own the spelling. Two owning it ("Smith"
        # for two Smiths; "Heather" for the one-word row AND the first
        # name of "Heather Jones") is the question being asked, not an
        # answer to it.
        owners = [c for c in rows if key in c.keys()]
        # A whole name, with or without its honorific, that is also a
        # rival's first name or surname is owned by that rival too: "Dr
        # Heather" for the one-word Dr Heather beside Heather Jones is the
        # same question as "Heather", and the ordinal is the way through.
        for c in list(owners):
            if key in c.exact_keys():
                owners += [o for o in rows if o is not c and o not in owners
                           and c.full in o.keys()]
        if len(owners) != 1:
            return None
        c = owners[0]
        accepted = {c.full, c.surname}
        hon = normalise(c.honorific)
        if hon:
            accepted |= {f"{hon} {c.full}", f"{hon} {c.surname}"}
        return c.name if key in accepted else None

    # ------------------------------------------------------------ writes
    def add(self, raw: dict) -> tuple[Optional[Contact], str]:
        """Validate, check uniqueness, append, save. (Contact, "") or
        (None, why); on a refusal nothing is written. OSError from the
        write itself is left to the caller (exit 1 / HTTP 500)."""
        contact, why = validate_row(raw)
        if contact is None:
            return None, why
        with self._lock:
            self.refresh()
            if self.broken:
                return None, self.broken
            why = check_unique(self.contacts, contact)
            if why:
                return None, why
            rows = list(self.rows) + [contact.row()]
            self._save_rows(rows)
            return self.by_name(contact.name), ""

    def remove(self, name: str, email: str = "") -> tuple[Optional[Contact], str]:
        """Remove the row with EXACTLY this full name (and, when given,
        exactly this address -- the page sends both so a stale tab cannot
        remove the wrong row after a hand edit)."""
        with self._lock:
            self.refresh()
            if self.broken:
                return None, self.broken
            contact = self.by_name(name)
            if contact is None:
                return None, f"nothing in the book called {name!r}"
            if email and contact.email.lower() != str(email).strip().lower():
                return None, (f"{contact.name} is not at that address in the "
                              "book; reload and look again")
            rows = [r for i, r in enumerate(self.rows) if i != contact.index]
            self._save_rows(rows)
            return contact, ""

    def _save_rows(self, rows: list) -> None:
        payload = {"format": FORMAT}
        payload.update(self.extra)
        payload["contacts"] = rows
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        _write_private(self.path, text)
        self._loaded = False
        self.refresh()

    # ------------------------------------------------------------- views
    def public(self) -> dict:
        self.refresh()
        return {"contacts": [c.public() for c in self.contacts],
                "skipped": [{"index": s.index, "name": s.name, "why": s.why}
                            for s in self.skipped],
                "flagged": [{"index": f.index, "name": f.name, "why": f.why}
                            for f in self.flagged],
                "trimmed": [{"index": t.index, "name": t.name, "why": t.why}
                            for t in self.trimmed],
                "broken": self.broken,
                "path": display_path(self.path)}


# ------------------------------------------------------------ singleton
_BOOK: Optional[Book] = None
_BOOK_LOCK = threading.Lock()


def current() -> Book:
    """The module's one Book, re-stat'd. Rebuilt if the path moved (a
    test pointing JARVIS_CONTACTS somewhere else)."""
    global _BOOK
    path = book_path()
    with _BOOK_LOCK:
        if _BOOK is None or _BOOK.path != path:
            _BOOK = Book(path)
    return _BOOK.refresh()


def resolve(said) -> Resolution:
    return current().resolve(said)
