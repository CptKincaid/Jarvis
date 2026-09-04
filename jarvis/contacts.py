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
* addresses are validated on ENTRY (one @, a dot in the domain, no spaces,
  a full match rather than a search), and a row that fails on load is
  skipped for resolution and flagged, so a typo cannot sit there waiting.
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
# text and would accept "heather@example.com junk" as an address. One @,
# at least one dot in the domain, nothing but address characters.
EMAIL_RX = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
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
        return None, ("bad address: one @, a dot in the domain, no spaces "
                      "-- name@example.com")
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


def check_unique(existing, new: Contact) -> str:
    """Why ``new`` may not join ``existing``, or "".

    No two rows with one full name (the "which" question could not be
    answered); no two rows with one address (it is one person twice); and
    an alias may not equal any other row's name, first name, surname or
    alias -- it would make a plain first name ambiguous with a nickname.
    """
    new_full = new.full
    new_email = new.email.lower()
    new_aliases = new.alias_keys()
    new_names = new.name_keys()
    for other in existing:
        if other is new:
            continue
        if other.full == new_full:
            return f"already in the book as {other.name}"
        if other.email.lower() == new_email:
            return f"that address is already in the book as {other.name}"
        clash = new_aliases & other.keys()
        if clash:
            return (f"alias {sorted(clash)[0]!r} is already how {other.name} "
                    f"is known")
        clash = new_names & other.alias_keys()
        if clash:
            return (f"{sorted(clash)[0]!r} is already an alias of "
                    f"{other.name}")
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
    module because consolidating the three is its own change."""
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
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class Book:
    """The book as last read, and the stamp it was read at.

    ``current()`` is the door: it re-stats the file every time and re-reads
    when the (mtime_ns, size) stamp has moved -- one stat per send, per
    question answer, per correction. An edit by hand, by the CLI or by the
    page is live on the next resolve with no restart, and AssistantConfig
    stays load-once.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else book_path()
        self.rows: list = []             # raw rows, file order, bad ones too
        self.contacts: list = []         # the validated ones
        self.skipped: list = []
        self.extra: dict = {}            # unknown top-level keys, kept
        self.stamp = None
        self._warned_stamp = None
        self._loaded = False
        self._lock = threading.RLock()

    # ---------------------------------------------------------- loading
    def refresh(self) -> "Book":
        """Re-read if the file moved. Never raises."""
        with self._lock:
            stamp = _stamp(self.path)
            if self._loaded and stamp == self.stamp:
                return self
            if stamp is None:
                # Missing file = empty book. Not an error, not a log line:
                # a fresh install has no book yet.
                self._install([], {}, None)
                return self
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
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
                # Keep the last good book; say so once per stamp, and say
                # only the PATH -- never a row.
                if stamp != self._warned_stamp:
                    log.warning("contacts: %s could not be read (%s); keeping "
                                "the last good book", self.path,
                                exc.__class__.__name__)
                    self._warned_stamp = stamp
                self._loaded = True
                return self
            extra = {k: v for k, v in raw.items()
                     if k not in ("format", "contacts")}
            self._install(rows, extra, stamp)
            return self

    def _install(self, rows: list, extra: dict, stamp) -> None:
        contacts, skipped = [], []
        for i, row in enumerate(rows):
            contact, why = validate_row(row)
            if contact is not None:
                why = check_unique(contacts, contact)
            if contact is None or why:
                shown = ""
                if isinstance(row, dict):
                    shown = " ".join(str(row.get("name") or "").split())[:NAME_MAX]
                skipped.append(Skipped(index=i, name=shown, why=why))
                continue
            contact.index = i
            contacts.append(contact)
        self.rows = list(rows)
        self.contacts = contacts
        self.skipped = skipped
        self.extra = extra
        self.stamp = stamp
        self._loaded = True
        if skipped:
            # Names only, never an address.
            log.warning("contacts: %d row(s) in %s skipped: %s", len(skipped),
                        self.path, "; ".join(
                            f"#{s.index + 1} {s.name or '(no name)'}: {s.why}"
                            for s in skipped))

    # ------------------------------------------------------------- reads
    def resolve(self, said) -> Resolution:
        """The match rule. Exact, or a question, or nothing."""
        self.refresh()
        key = normalise(said)
        if not key:
            return Resolution()
        # An exact FULL name (or honorific + full name) wins outright even
        # when another row shares the first name: he said all of it.
        for c in self.contacts:
            hon = normalise(c.honorific)
            if key == c.full or (hon and key == f"{hon} {c.full}"):
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
        if not rows:
            return None
        for c in rows:
            hon = normalise(c.honorific)
            if key == c.full or (hon and key == f"{hon} {c.full}"):
                return c.name
        by_surname = [c for c in rows
                      if key == c.surname
                      or (normalise(c.honorific)
                          and key == f"{normalise(c.honorific)} {c.surname}")]
        if len(by_surname) == 1:
            return by_surname[0].name
        return None

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
