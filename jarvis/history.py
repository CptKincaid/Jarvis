"""Typed-command history for Jarvis.

Every command the user types into the command bar is appended here
(``JarvisApp.dispatch_text`` calls ``add``), persisted as JSON lines under
``~/.aiws_trainer/jarvis_memory/typed_history.jsonl`` so it survives
restarts. Two consumers:

- the command bar's Up/Down navigation (``prev()`` / ``next()`` — shell
  semantics: Up walks back from the newest, Down walks forward, any
  ``add()`` resets the cursor);
- Jarvis himself ("what did I ask you last", ``recent()`` / ``search()``).

Voice utterances are not recorded here — they are already in the
conversation context — unless a caller passes ``source="voice"``
explicitly.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("history")

DEFAULT_FILE = PATHS.MEMORY_DIR / "typed_history.jsonl"


class TypedHistory:
    def __init__(self, path: Path | str | None = None, max_items: int = 500):
        self.path = Path(path) if path else DEFAULT_FILE
        self.max_items = max(1, int(max_items))
        self._items: list[dict] = []
        self._cursor: Optional[int] = None      # None = past the newest
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------ disk
    def _load(self) -> None:
        if not self.path.exists():
            return
        items: list[dict] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("text"):
                    items.append(obj)
        except OSError:
            log.exception("history read failed: %s", self.path)
        self._items = items[-self.max_items:]

    def _append_line(self, obj: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("history append failed: %s", self.path)

    def _rewrite(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n"
                                   for o in self._items), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            log.exception("history rewrite failed: %s", self.path)

    # ------------------------------------------------------------- api
    def add(self, text: str, source: str = "typed") -> bool:
        """Record one command. Consecutive duplicates collapse. Returns
        True when an entry was written."""
        text = (text or "").strip()
        if not text:
            return False
        with self._lock:
            self._cursor = None
            if self._items and self._items[-1].get("text") == text:
                self._items[-1]["ts"] = time.time()
                self._items[-1]["count"] = self._items[-1].get("count", 1) + 1
                return False
            obj = {"ts": time.time(), "text": text, "source": source}
            self._items.append(obj)
            if len(self._items) > self.max_items:
                self._items = self._items[-self.max_items:]
                self._rewrite()
            else:
                self._append_line(obj)
            return True

    def recent(self, n: int = 10) -> list[str]:
        """Newest first."""
        with self._lock:
            return [o["text"] for o in reversed(self._items[-max(0, n):])]

    def search(self, prefix: str, n: int = 10) -> list[str]:
        """Entries starting with ``prefix`` (case-insensitive), newest first,
        de-duplicated."""
        prefix = (prefix or "").lower()
        seen: set[str] = set()
        out: list[str] = []
        with self._lock:
            for o in reversed(self._items):
                t = o["text"]
                if t.lower().startswith(prefix) and t not in seen:
                    seen.add(t)
                    out.append(t)
                    if len(out) >= n:
                        break
        return out

    def last(self) -> Optional[str]:
        with self._lock:
            return self._items[-1]["text"] if self._items else None

    # -------------------------------------------------- cursor (Up/Down)
    def prev(self) -> Optional[str]:
        """Step back toward older entries (Up arrow)."""
        with self._lock:
            if not self._items:
                return None
            if self._cursor is None:
                self._cursor = len(self._items) - 1
            elif self._cursor > 0:
                self._cursor -= 1
            return self._items[self._cursor]["text"]

    def next(self) -> Optional[str]:
        """Step forward toward newer entries (Down arrow); past the newest
        returns None (an empty entry field)."""
        with self._lock:
            if self._cursor is None:
                return None
            if self._cursor >= len(self._items) - 1:
                self._cursor = None
                return None
            self._cursor += 1
            return self._items[self._cursor]["text"]

    def reset_cursor(self) -> None:
        with self._lock:
            self._cursor = None

    def clear(self) -> None:
        with self._lock:
            self._items = []
            self._cursor = None
            self._rewrite()

    def __len__(self) -> int:
        return len(self._items)
