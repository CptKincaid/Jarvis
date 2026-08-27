"""Notes and to-dos (spec section 6.5).

``NotesStore`` keeps two SQLite tables under ``PATHS.MEMORY_DIR/notes.db``
(constructor arg; tests pass a tmp path) and renders spoken lists in the
JARVIS voice. The ``notes`` tool wraps it for the local tool loop; every
confirmation is returned as ``ToolResult.speak`` so no model turn is
needed for "note that …" / "what's on my list".

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
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.notes")

KINDS = ("note", "todo")
_TABLE = {"note": "notes", "todo": "todos"}
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
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _kind(kind) -> str:
    k = str(kind or "").strip().lower()
    if k in KINDS:
        return k
    return _KIND_WORDS.get(k, "")


def number_word(n: int) -> str:
    """0..20 as words ('no', 'one', … 'twenty'), larger as digits."""
    return _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else str(n)


def _plural(kind: str, n: int) -> str:
    if kind == "todo":
        return "to-do" if n == 1 else "to-dos"
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

    # ------------------------------------------------------------ CRUD
    def add(self, kind: str, text: str, tags: str = "",
            created: Optional[float] = None) -> int:
        k = _kind(kind)
        if k not in KINDS:
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
            else:
                cur = self._db.execute(
                    "INSERT INTO todos(text, created) VALUES (?,?)",
                    (text, created))
            self._db.commit()
            log.info("%s added (#%d, %d chars)", k, cur.lastrowid, len(text))
            return int(cur.lastrowid)

    def list(self, kind: str, limit: int = 10,
             include_done: bool = False) -> list[dict]:
        """The most recent ``limit`` items in chronological order (oldest
        of the window first) — the order ``list_text`` speaks and the
        order ordinals refer to."""
        k = _kind(kind)
        if k not in KINDS:
            return []
        limit = max(1, int(limit or 10))
        with self._lock:
            if k == "note":
                rows = self._db.execute(
                    "SELECT * FROM notes ORDER BY id DESC LIMIT ?",
                    (limit,)).fetchall()
            else:
                where = "" if include_done else "WHERE done = 0"
                rows = self._db.execute(
                    f"SELECT * FROM todos {where} ORDER BY id DESC LIMIT ?",
                    (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def count(self, kind: str, include_done: bool = False) -> int:
        k = _kind(kind)
        with self._lock:
            if k == "note":
                return self._db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            if k == "todo":
                where = "" if include_done else "WHERE done = 0"
                return self._db.execute(
                    f"SELECT COUNT(*) FROM todos {where}").fetchone()[0]
        return 0

    def search(self, kind: str, query: str, limit: int = 10,
               include_done: bool = True) -> list[dict]:
        k = _kind(kind)
        q = " ".join(str(query or "").split()).lower()
        if k not in KINDS or not q:
            return []
        like = f"%{q}%"
        with self._lock:
            if k == "note":
                rows = self._db.execute(
                    "SELECT * FROM notes WHERE lower(text) LIKE ? "
                    "ORDER BY id DESC LIMIT ?", (like, limit)).fetchall()
            else:
                where = "" if include_done else "AND done = 0"
                rows = self._db.execute(
                    f"SELECT * FROM todos WHERE lower(text) LIKE ? {where} "
                    "ORDER BY id DESC LIMIT ?", (like, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def resolve(self, kind: str, which=None, include_done: bool = False,
                window: int = 10) -> Optional[dict]:
        """Pick one item by ``which`` (see module doc). None when nothing
        matches or the index is out of range."""
        k = _kind(kind)
        if k not in KINDS:
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
        if k not in KINDS:
            return []
        mode, _ = parse_which(which)
        with self._lock:
            if mode == "all":
                rows = self.list(k, limit=1000)
                if k == "note":
                    self._db.execute("DELETE FROM notes")
                else:
                    self._db.execute("DELETE FROM todos WHERE done = 0")
                self._db.commit()
                log.info("%s cleared (%d)", k, len(rows))
                return rows
            item = self.resolve(k, which)
            if item is None:
                return []
            self._db.execute(f"DELETE FROM {_TABLE[k]} WHERE id = ?",
                             (item["id"],))
            self._db.commit()
            log.info("%s #%d removed", k, item["id"])
            return [item]

    def complete(self, which=None, done_at: Optional[float] = None) -> list[dict]:
        """Mark a to-do done ('all' completes every open one). Returns
        the rows completed."""
        mode, _ = parse_which(which)
        ts = time.time() if done_at is None else float(done_at)
        with self._lock:
            if mode == "all":
                rows = self.list("todo", limit=1000)
                self._db.execute(
                    "UPDATE todos SET done = 1, done_at = ? WHERE done = 0", (ts,))
                self._db.commit()
                return rows
            item = self.resolve("todo", which)
            if item is None:
                return []
            self._db.execute(
                "UPDATE todos SET done = 1, done_at = ? WHERE id = ?",
                (ts, item["id"]))
            self._db.commit()
            log.info("todo #%d done", item["id"])
            item = dict(item, done=1, done_at=ts)
            return [item]

    # --------------------------------------------------------- wording
    def list_text(self, kind: str, limit: int = 10) -> str:
        k = _kind(kind) or "note"
        items = self.list(k, limit=limit)
        total = self.count(k)
        if not items:
            return "No notes yet, sir." if k == "note" else \
                "Nothing on your list, sir."
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
        if not hits:
            return f"Nothing about {q} in your {_plural(k, 2)}, sir."
        n = len(hits)
        body = join_spoken([_spoken_item(h["text"]) for h in hits])
        if n == 1:
            return f"One {_plural(k, 1)} mentions {q}, sir: {body}."
        return f"{number_word(n).capitalize()} {_plural(k, n)} mention {q}, " \
               f"sir: {body}."

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

    def notes(action="list", kind=None, text=None, which=None, **_) -> ToolResult:
        act = _action(action)
        text = " ".join(str(text or "").split())
        k = _kind_for(kind, act, text)
        s = _store()
        if act == "add":
            if not text:
                line = "What shall I note down, sir?" if k == "note" else \
                    "What shall I add to the list, sir?"
                return ToolResult(text="nothing to add: no text given",
                                  ok=False, speak=line)
            s.add(k, text)
            line = "Noted, sir." if k == "note" else "Added to your list, sir."
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
            removed = s.remove(k, target)
            if not removed:
                line = "I couldn't find that one, sir."
                return ToolResult(text=f"no {k} matched {target!r}", ok=False,
                                  speak=line)
            left = s.count(k)
            if len(removed) > 1:
                line = "All cleared, sir."
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
        description="Add, list, search, remove or complete Hunter's notes and to-dos.",
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
            },
            "required": ["action", "kind"],
        },
        handler=notes,
    )
    return [spec]
