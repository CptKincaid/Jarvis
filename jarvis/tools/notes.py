"""Notes and to-dos (spec section 6.5).

``NotesStore`` keeps two SQLite tables under ``PATHS.MEMORY_DIR/notes.db``
(constructor arg; tests pass a tmp path) and renders spoken lists in the
JARVIS voice. The ``notes`` tool wraps it for the local tool loop; every
confirmation is returned as ``ToolResult.speak`` so no model turn is
needed for "note that …" / "what's on my list".

Beyond the two built-in kinds there are NAMED lists ("shopping",
"packing"): rows in ``lists`` / ``list_items`` addressed by the kind
string ``list:<name>`` (``list_kind()`` builds one, ``list_name()`` reads
one back). Every method below takes that kind, so named lists inherit the
whole resolve / ordinal / read-back plumbing rather than growing a
parallel one.

``which`` resolution (``resolve``): ``last`` / ``latest``, an ordinal or
index ("the second one", "2", "#2", "number two"), ``first``, or a
case-insensitive substring of the item text. Ordinals index the same
chronological order ``list_text`` speaks, so "remove the second one"
strikes the second item he just heard.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from jarvis.logs import get_logger
from jarvis.tools.docs import match_name
from jarvis.tools.location import cfg_get
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.notes")

KINDS = ("note", "todo")
_KIND_WORDS = {"note": "note", "notes": "note", "memo": "note", "memos": "note",
               "todo": "todo", "todos": "todo", "to-do": "todo", "to-dos": "todo",
               "to do": "todo", "to dos": "todo", "task": "todo", "tasks": "todo",
               "list": "todo", "chore": "todo", "chores": "todo", "item": "todo"}
_ACTION_WORDS = {
    "add": "add", "new": "add", "create": "add", "save": "add", "take": "add",
    "note": "add", "write": "add", "remember": "add", "put": "add",
    "list": "list", "show": "list", "read": "list", "get": "list", "what": "list",
    "remove": "remove", "delete": "remove", "drop": "remove", "forget": "remove",
    "erase": "remove", "clear": "remove", "strike": "remove", "cancel": "remove",
    "search": "search", "find": "search", "look": "search", "lookup": "search",
    "done": "done", "complete": "done", "finish": "done", "finished": "done",
    "tick": "done", "check": "done", "completed": "done", "mark": "done",
}
# A named list is a kind like any other: "list:shopping". Keeping it a
# kind STRING (rather than an extra argument on nine methods) is what lets
# resolve/remove/complete/list_text serve lists unchanged -- including the
# commander's destructive read-back, which only ever passes a kind around.
LIST_PREFIX = "list:"
# Words that are never part of a list's name.
_LIST_STOP = {"the", "a", "an", "my", "our", "your", "his", "her", "this",
              "that", "some"}
_LIST_TAIL = {"list", "lists"}
LIST_NAME_CHARS = 40

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
             "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6,
             "7th": 7, "8th": 8, "9th": 9, "10th": 10}
_NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                 "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                 "nineteen", "twenty"]
_LAST_WORDS = {"last", "latest", "newest", "recent", "previous", "that"}
_ITEM_CHARS = 100          # spoken length cap per item

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created REAL NOT NULL,
    tags TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created REAL NOT NULL,
    done INTEGER DEFAULT 0,
    done_at REAL
);
CREATE TABLE IF NOT EXISTS lists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS list_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    list_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    created REAL NOT NULL,
    done INTEGER DEFAULT 0,
    done_at REAL
);
CREATE INDEX IF NOT EXISTS list_items_by_list ON list_items(list_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def canon_list(name) -> str:
    """A spoken list name in canonical form: "the Shopping List" ->
    "shopping". Empty when nothing usable is left ("the list")."""
    words = re.findall(r"[\w'-]+", str(name or "").lower())
    while words and words[0] in _LIST_STOP:
        words.pop(0)
    while words and words[-1] in _LIST_TAIL:
        words.pop()
    words = [w for w in words if w not in _LIST_STOP]
    return " ".join(words)[:LIST_NAME_CHARS].strip()


def list_kind(name) -> str:
    """'shopping' / 'the shopping list' -> 'list:shopping'; '' when the
    name is empty or reserved (a named list must not shadow the built-in
    notes and to-dos: "my task list" stays the to-do list)."""
    c = canon_list(name)
    if not c or c in _KIND_WORDS or c in KINDS:
        return ""
    return LIST_PREFIX + c


def list_name(kind) -> Optional[str]:
    """'list:shopping' -> 'shopping'; None when the kind is not a list."""
    k = str(kind or "").strip().lower()
    if not k.startswith(LIST_PREFIX):
        return None
    return k[len(LIST_PREFIX):].strip() or None


def _kind(kind) -> str:
    k = str(kind or "").strip().lower()
    if k in KINDS:
        return k
    if k.startswith(LIST_PREFIX):
        return list_kind(k[len(LIST_PREFIX):])
    return _KIND_WORDS.get(k, "")


def number_word(n: int) -> str:
    """0..20 as words ('no', 'one', … 'twenty'), larger as digits."""
    return _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else str(n)


def _plural(kind: str, n: int) -> str:
    if kind == "todo":
        return "to-do" if n == 1 else "to-dos"
    if list_name(kind) is not None:
        return "item" if n == 1 else "items"
    return "note" if n == 1 else "notes"


def _spoken_item(text: str) -> str:
    text = " ".join(str(text).split())
    text = text.rstrip(".;: ")
    if len(text) > _ITEM_CHARS:
        cut = text[:_ITEM_CHARS].rsplit(" ", 1)[0]
        text = cut + "…"
    return text


def join_spoken(items: list[str]) -> str:
    """'a', 'a and b', 'a, b, and c' (semicolons when items hold commas)."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    sep = "; " if any("," in i for i in items) else ", "
    return sep.join(items[:-1]) + f"{sep.rstrip()} and {items[-1]}"


def parse_which(which) -> tuple[str, object]:
    """Classify a ``which`` argument -> ('last'|'all'|'index'|'text', value)."""
    if which is None:
        return "last", None
    if isinstance(which, bool):
        return "last", None
    if isinstance(which, (int, float)):
        return "index", int(which)
    w = " ".join(str(which).strip().lower().split())
    if not w:
        return "last", None
    if w in {"that one", "that", "it", "this one", "the last one", "last one"}:
        return "last", None
    w = re.sub(r"^(the|that|my|this)\s+", "", w)
    w = re.sub(r"\s+(one|item|entry|note|todo|to-do|task)$", "", w)
    if w in _LAST_WORDS:
        return "last", None
    if w in {"all", "everything", "all of them", "them all"}:
        return "all", None
    m = re.fullmatch(r"(?:number|no\.?|#)?\s*(\d{1,3})(?:st|nd|rd|th)?", w)
    if m:
        return "index", int(m.group(1))
    m = re.fullmatch(r"(?:number\s+)?([a-z]+)", w)
    if m and m.group(1) in _ORDINALS:
        return "index", _ORDINALS[m.group(1)]
    return "text", w


class NotesStore:
    """SQLite-backed notes and to-dos. Thread-safe (one lock, one
    connection with check_same_thread=False)."""

    # A whole-list clear waiting for a spoken yes ({"kind", "n", "ts"}),
    # set by the tool and consumed by Commander._try_destructive_confirm.
    pending_clear = None

    def __init__(self, db_path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def close(self):
        with self._lock:
            try:
                self._db.close()
            except Exception:
                log.debug("notes db close failed", exc_info=True)

    # ----------------------------------------------------- named lists
    def _list_id(self, name, create: bool = False) -> Optional[int]:
        """The row id of a named list, creating it on demand. None when
        the name is unusable or unknown and ``create`` is False."""
        name = canon_list(name)
        if not name:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM lists WHERE name = ? COLLATE NOCASE",
                (name,)).fetchone()
            if row is not None:
                return int(row["id"])
            if not create:
                return None
            cur = self._db.execute(
                "INSERT INTO lists(name, created) VALUES (?,?)",
                (name, time.time()))
            self._db.commit()
            log.info("list %r created (#%d)", name, cur.lastrowid)
            return int(cur.lastrowid)

    def list_names(self) -> list:
        """Every named list, oldest first (the order lists_text speaks)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT name FROM lists ORDER BY id").fetchall()
        return [str(r["name"]) for r in rows]

    def find_list(self, spoken) -> Optional[str]:
        """The existing list a spoken name means, or None. Exact match
        first, then the fuzzy name match docs.py uses for file names --
        Whisper hears "the pack in list" for "the packing list"."""
        want = canon_list(spoken)
        if not want:
            return None
        names = self.list_names()
        if want in names:
            return want
        return match_name(want, names)

    def make_list(self, spoken) -> Optional[str]:
        """Resolve a spoken name to an existing list, else create it.
        None when the name is empty or reserved."""
        found = self.find_list(spoken)
        if found is not None:
            return found
        if not list_kind(spoken):
            return None
        name = canon_list(spoken)
        self._list_id(name, create=True)
        return name

    def _scope(self, kind, create: bool = False):
        """``(table, where-fragments, params, has_done)`` for a kind, or
        None when the kind is unknown (or names a list that does not
        exist). The one place a kind becomes SQL."""
        k = _kind(kind)
        if k == "note":
            return ("notes", [], [], False)
        if k == "todo":
            return ("todos", [], [], True)
        name = list_name(k)
        if name is None:
            return None
        lid = self._list_id(name, create=create)
        if lid is None:
            return None
        return ("list_items", ["list_id = ?"], [lid], True)

    def has_list(self, kind) -> bool:
        """True when the kind names a list that exists."""
        return list_name(_kind(kind)) is not None and self._scope(kind) is not None

    @staticmethod
    def _clause(where: list) -> str:
        return (" WHERE " + " AND ".join(where)) if where else ""

    # ------------------------------------------------------------ CRUD
    def add(self, kind: str, text: str, tags: str = "",
            created: Optional[float] = None) -> int:
        k = _kind(kind)
        if not k:
            raise ValueError(f"unknown kind {kind!r}")
        text = " ".join(str(text or "").split())
        if not text:
            raise ValueError("empty text")
        created = time.time() if created is None else float(created)
        with self._lock:
            if k == "note":
                cur = self._db.execute(
                    "INSERT INTO notes(text, created, tags) VALUES (?,?,?)",
                    (text, created, tags or ""))
            elif k == "todo":
                cur = self._db.execute(
                    "INSERT INTO todos(text, created) VALUES (?,?)",
                    (text, created))
            else:
                # Adding to a list is what brings it into being: "add milk
                # to the shopping list" must not need the list declared first.
                lid = self._list_id(list_name(k), create=True)
                cur = self._db.execute(
                    "INSERT INTO list_items(list_id, text, created) VALUES (?,?,?)",
                    (lid, text, created))
            self._db.commit()
            log.info("%s added (#%d, %d chars)", k, cur.lastrowid, len(text))
            return int(cur.lastrowid)

    def delete(self, kind: str, item_id) -> bool:
        """Delete exactly one row by id -- the undo path. No matching and
        no ambiguity: the row the add returned and no other."""
        scope = self._scope(kind)
        if scope is None or item_id is None:
            return False
        table, where, params, _done = scope
        clause = self._clause(list(where) + ["id = ?"])
        with self._lock:
            cur = self._db.execute(f"DELETE FROM {table}{clause}",
                                   (*params, int(item_id)))
            self._db.commit()
        return cur.rowcount > 0

    def list(self, kind: str, limit: int = 10,
             include_done: bool = False) -> list[dict]:
        """The most recent ``limit`` items in chronological order (oldest
        of the window first) — the order ``list_text`` speaks and the
        order ordinals refer to."""
        scope = self._scope(kind)
        if scope is None:
            return []
        table, where, params, has_done = scope
        if has_done and not include_done:
            where = where + ["done = 0"]
        limit = max(1, int(limit or 10))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM {table}{self._clause(where)} "
                "ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def count(self, kind: str, include_done: bool = False) -> int:
        scope = self._scope(kind)
        if scope is None:
            return 0
        table, where, params, has_done = scope
        if has_done and not include_done:
            where = where + ["done = 0"]
        with self._lock:
            return int(self._db.execute(
                f"SELECT COUNT(*) FROM {table}{self._clause(where)}",
                tuple(params)).fetchone()[0])

    def search(self, kind: str, query: str, limit: int = 10,
               include_done: bool = True) -> list[dict]:
        scope = self._scope(kind)
        q = " ".join(str(query or "").split()).lower()
        if scope is None or not q:
            return []
        table, where, params, has_done = scope
        where = where + ["lower(text) LIKE ?"]
        params = params + [f"%{q}%"]
        if has_done and not include_done:
            where = where + ["done = 0"]
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM {table}{self._clause(where)} "
                "ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def resolve(self, kind: str, which=None, include_done: bool = False,
                window: int = 10) -> Optional[dict]:
        """Pick one item by ``which`` (see module doc). None when nothing
        matches or the index is out of range."""
        k = _kind(kind)
        if not k:
            return None
        mode, value = parse_which(which)
        items = self.list(k, limit=window, include_done=include_done)
        if not items:
            return None
        if mode == "last":
            return items[-1]
        if mode == "all":
            return None
        if mode == "index":
            idx = int(value)
            if 1 <= idx <= len(items):
                return items[idx - 1]
            return None
        # substring: exact first, else the most recent containing match
        # across the whole table (not only the spoken window)
        matches = self.search(k, value, limit=50, include_done=include_done)
        if not matches:
            return None
        exact = [m for m in matches if m["text"].lower() == value]
        return (exact or matches)[-1]

    def remove(self, kind: str, which=None) -> list[dict]:
        """Delete the matched item (or every open item for 'all').
        Returns the removed rows."""
        k = _kind(kind)
        scope = self._scope(k)
        if scope is None:
            return []
        table, where, params, has_done = scope
        mode, _ = parse_which(which)
        with self._lock:
            if mode == "all":
                rows = self.list(k, limit=1000)
                w = list(where) + (["done = 0"] if has_done else [])
                self._db.execute(f"DELETE FROM {table}{self._clause(w)}",
                                 tuple(params))
                self._db.commit()
                log.info("%s cleared (%d)", k, len(rows))
                return rows
            item = self.resolve(k, which)
            if item is None:
                return []
            self._db.execute(f"DELETE FROM {table} WHERE id = ?",
                             (item["id"],))
            self._db.commit()
            log.info("%s #%d removed", k, item["id"])
            return [item]

    def complete(self, which=None, done_at: Optional[float] = None,
                 kind: str = "todo") -> list[dict]:
        """Mark a to-do (or a named-list item) done -- 'all' completes
        every open one. Returns the rows completed."""
        k = _kind(kind) or "todo"
        scope = self._scope(k)
        if scope is None:
            return []
        table, where, params, has_done = scope
        if not has_done:
            return []                        # notes are not completable
        mode, _ = parse_which(which)
        ts = time.time() if done_at is None else float(done_at)
        with self._lock:
            if mode == "all":
                rows = self.list(k, limit=1000)
                w = list(where) + ["done = 0"]
                self._db.execute(
                    f"UPDATE {table} SET done = 1, done_at = ?{self._clause(w)}",
                    (ts, *params))
                self._db.commit()
                return rows
            item = self.resolve(k, which)
            if item is None:
                return []
            self._db.execute(
                f"UPDATE {table} SET done = 1, done_at = ? WHERE id = ?",
                (ts, item["id"]))
            self._db.commit()
            log.info("%s #%d done", k, item["id"])
            item = dict(item, done=1, done_at=ts)
            return [item]

    # --------------------------------------------------------- wording
    def lists_text(self) -> str:
        """"What lists do I have?" -- names only, never their contents."""
        names = self.list_names()
        if not names:
            return "You haven't any lists yet, sir."
        if len(names) == 1:
            return f"One list, sir: {names[0]}."
        return (f"{number_word(len(names)).capitalize()} lists, sir: "
                f"{join_spoken(names)}.")

    def list_text(self, kind: str, limit: int = 10) -> str:
        k = _kind(kind) or "note"
        name = list_name(k)
        if name is not None and self._scope(k) is None:
            # Never invent a list by reading it: an empty "packing list"
            # that only exists because he asked for one is a lie.
            return f"You haven't a {name} list, sir."
        items = self.list(k, limit=limit)
        total = self.count(k)
        if not items:
            if name is not None:
                return f"Nothing on your {name} list, sir."
            return "No notes yet, sir." if k == "note" else \
                "Nothing on your list, sir."
        if name is not None:
            n = len(items)
            body = join_spoken([_spoken_item(i["text"]) for i in items])
            head = f"{number_word(n).capitalize()} on your {name} list, sir"
            if total > n:
                head = (f"{number_word(total).capitalize()} on your {name} "
                        f"list, sir; the latest {number_word(n)}")
            return f"{head}: {body}."
        n = len(items)
        count_word = number_word(n).capitalize()
        head = f"{count_word} {_plural(k, n)}, sir"
        if total > n:
            head = f"{number_word(total).capitalize()} {_plural(k, total)}, " \
                   f"sir; the latest {number_word(n)}"
        body = join_spoken([_spoken_item(i["text"]) for i in items])
        return f"{head}: {body}."

    def search_text(self, kind: str, query: str) -> str:
        k = _kind(kind) or "note"
        q = " ".join(str(query or "").split())
        hits = self.search(k, q, limit=10)
        where = list_name(k)
        where = f"on your {where} list" if where else f"in your {_plural(k, 2)}"
        if not hits:
            return f"Nothing about {q} {where}, sir."
        n = len(hits)
        body = join_spoken([_spoken_item(h["text"]) for h in hits])
        if n == 1:
            return f"One {_plural(k, 1)} mentions {q}, sir: {body}."
        return f"{number_word(n).capitalize()} {_plural(k, n)} mention {q}, " \
               f"sir: {body}."

    def clear_line(self, kind: str, n: int) -> str:
        """The read-back a whole-list wipe asks before running."""
        name = list_name(_kind(kind))
        if name:
            return f"Clear all {number_word(n)} off your {name} list, sir?"
        return f"Clear all {number_word(n)} {_plural(_kind(kind), n)}, sir?"

    # ---------------------------------------------------------- legacy
    def import_legacy(self, memory_notes_dir) -> int:
        """Import ``note_*.txt`` voice notes (``[YYYY-MM-DD HH:MM]\\ntext``)
        once; a meta marker row makes a second call a no-op."""
        d = Path(memory_notes_dir).expanduser()
        marker = f"legacy_import:{d}"
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = ?",
                                   (marker,)).fetchone()
            if row is not None:
                return 0
            imported = 0
            if d.is_dir():
                for path in sorted(d.glob("note_*.txt")):
                    tag = f"legacy:{path.name}"
                    dup = self._db.execute(
                        "SELECT 1 FROM notes WHERE tags = ?", (tag,)).fetchone()
                    if dup:
                        continue
                    try:
                        raw = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        log.warning("legacy note unreadable: %s", path.name)
                        continue
                    created, text = _parse_legacy(raw, path)
                    if not text:
                        continue
                    self._db.execute(
                        "INSERT INTO notes(text, created, tags) VALUES (?,?,?)",
                        (text, created, tag))
                    imported += 1
            self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
                             (marker, str(imported)))
            self._db.commit()
        if imported:
            log.info("imported %d legacy notes from %s", imported, d)
        return imported


def _parse_legacy(raw: str, path: Path) -> tuple[float, str]:
    lines = raw.strip().splitlines()
    created = None
    if lines and re.fullmatch(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]", lines[0].strip()):
        try:
            created = datetime.strptime(lines[0].strip()[1:-1],
                                        "%Y-%m-%d %H:%M").timestamp()
        except ValueError:
            created = None
        lines = lines[1:]
    if created is None:
        m = re.search(r"note_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", path.name)
        if m:
            try:
                created = datetime.strptime(m.group(1),
                                            "%Y-%m-%d_%H-%M-%S").timestamp()
            except ValueError:
                created = None
    if created is None:
        try:
            created = path.stat().st_mtime
        except OSError:
            created = time.time()
    text = " ".join(" ".join(lines).split())
    return created, text


# ------------------------------------------------------------------ tool
def _action(action) -> str:
    a = str(action or "").strip().lower()
    if a in ("add", "list", "remove", "search", "done"):
        return a
    for word in a.replace("_", " ").split():
        if word in _ACTION_WORDS:
            return _ACTION_WORDS[word]
    return a


def _kind_for(kind, action: str, text: str) -> str:
    k = _kind(kind)
    if k:
        return k
    if action == "done":
        return "todo"
    if re.search(r"\b(to-?do|task|list)\b", text or "", re.I):
        return "todo"
    return "note"


def make_tools(cfg, services) -> list[ToolSpec]:
    """The ``notes`` tool. Uses ``services.notes`` when the app wired one,
    else opens the default store under PATHS.MEMORY_DIR."""
    store = getattr(services, "notes", None) if services is not None else None

    def _store() -> NotesStore:
        nonlocal store
        if store is None:
            from jarvis.config import PATHS
            path = getattr(PATHS, "NOTES_DB", None) or \
                Path(PATHS.MEMORY_DIR) / "notes.db"
            store = NotesStore(path)
        return store

    # `list` shadows the builtin deliberately: the registry calls the
    # handler with the schema's own key names (handler(**args)).
    def notes(action="list", kind=None, text=None, which=None, list=None,
              **_) -> ToolResult:
        act = _action(action)
        text = " ".join(str(text or "").split())
        s = _store()
        if list:
            # A named list ("shopping"): adding creates it, everything else
            # only ever touches one that exists.
            name = s.make_list(list) if act == "add" else s.find_list(list)
            if name is None:
                line = f"You haven't a {canon_list(list) or 'such'} list, sir."
                return ToolResult(text=f"no list named {list!r}", ok=False,
                                  speak=line)
            k = list_kind(name)
        else:
            k = _kind_for(kind, act, text)
        lname = list_name(k)
        if act == "add":
            if not text:
                line = "What shall I note down, sir?" if k == "note" else \
                    "What shall I add to the list, sir?"
                return ToolResult(text="nothing to add: no text given",
                                  ok=False, speak=line)
            s.add(k, text)
            line = "Noted, sir." if k == "note" else \
                f"Added to your {lname} list, sir." if lname else \
                "Added to your list, sir."
            return ToolResult(text=f"{k} added: {text}", speak=line)
        if act == "list":
            line = s.list_text(k)
            return ToolResult(text=line, speak=line)
        if act == "search":
            q = text or (which if isinstance(which, str) else "") or ""
            if not q.strip():
                line = "What shall I look for, sir?"
                return ToolResult(text="no query", ok=False, speak=line)
            line = s.search_text(k, q)
            return ToolResult(text=line, speak=line)
        if act == "remove":
            target = which if which not in (None, "") else (text or "last")
            mode, _ = parse_which(target)
            if mode == "all":
                # Read a whole-list wipe back before doing it: a bare
                # "clear" used to empty the list on the first transcript.
                # The store carries the offer (the commander sees the same
                # object as services.notes) and the next spoken yes runs
                # it -- the pattern calendar.py uses for pending_event.
                # A one-item list is not worth the question.
                n = s.count(k)
                if n > 1 and cfg_get(cfg, "confirm.read_back", True) is not False:
                    s.pending_clear = {"kind": k, "n": n, "ts": time.time()}
                    line = s.clear_line(k, n)
                    return ToolResult(text=f"asked before clearing {n} {k}s", speak=line)
            s.pending_clear = None
            removed = s.remove(k, target)
            if not removed:
                line = "I couldn't find that one, sir."
                return ToolResult(text=f"no {k} matched {target!r}", ok=False,
                                  speak=line)
            left = s.count(k)
            if len(removed) > 1:
                line = f"The {lname} list is clear, sir." if lname else \
                    "All cleared, sir."
            elif lname:
                line = f"Off the {lname} list, sir; {number_word(left)} left." \
                    if left else f"Off the {lname} list, sir; that's it clear."
            elif k == "todo":
                line = f"Struck off, sir; {number_word(left)} left." if left \
                    else "Struck off, sir; the list is clear."
            else:
                line = "Forgotten, sir."
            return ToolResult(text=f"removed: {'; '.join(r['text'] for r in removed)}",
                              speak=line)
        if act == "done":
            target = which if which not in (None, "") else (text or "last")
            done = s.complete(target)
            if not done:
                line = "I couldn't find that one on the list, sir."
                return ToolResult(text=f"no open to-do matched {target!r}",
                                  ok=False, speak=line)
            left = s.count("todo")
            if left == 0:
                line = "Done, sir; that clears the list."
            else:
                line = f"Done, sir; {number_word(left)} left."
            return ToolResult(text=f"done: {'; '.join(r['text'] for r in done)}",
                              speak=line)
        return ToolResult(text=f"notes: unknown action {action!r}", ok=False)

    spec = ToolSpec(
        name="notes",
        description="Add, list, search, remove or complete Hunter's notes, "
                    "to-dos and named lists (shopping, packing).",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["add", "list", "remove", "search", "done"]},
                "kind": {"type": "string", "enum": ["note", "todo"]},
                "text": {"type": "string",
                         "description": "the note / to-do text, or a search query"},
                "which": {"type": "string",
                          "description": "last, an index like 2, or words from the item"},
                "list": {"type": "string",
                         "description": "a named list such as shopping or packing; "
                                        "leave out for notes and to-dos"},
            },
            "required": ["action", "kind"],
        },
        handler=notes,
    )
    return [spec]
