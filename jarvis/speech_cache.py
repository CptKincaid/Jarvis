"""On-disk cache of synthesized speech for repeated phrases.

XTTS costs roughly real-time on the GB10 and Edge needs a network round
trip; Jarvis says the same short things many times a day ("Always, sir.",
"Good night, sir. I'll be here.", reminder preambles, the quiet-mode
acknowledgement). The cache keeps the rendered audio keyed by
(engine, voice parameters, spoken text) so a repeat plays instantly — and,
for XTTS, always with the same rendition instead of a fresh random take.

Layout: ``<cache_dir>/<sha1>.<ext>``. Bounded by file count and bytes;
pruning evicts least-recently-used (mtime is touched on every hit). Every
operation is best-effort: a cache failure never blocks speech.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("speech_cache")

DEFAULT_DIR = PATHS.AIWS / "tts_cache"


class SpeechCache:
    def __init__(self, cache_dir: Path | str | None = None,
                 max_files: int = 400, max_bytes: int = 150_000_000):
        self.dir = Path(cache_dir) if cache_dir else DEFAULT_DIR
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------- key
    @staticmethod
    def key(engine: str, text: str, **params) -> str:
        """Stable key for one rendition: engine + sorted params + text."""
        blob = json.dumps({"engine": engine, "text": text, "params": params},
                          sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _candidates(self, key: str) -> list[Path]:
        try:
            return sorted(self.dir.glob(f"{key}.*"))
        except OSError:
            return []

    # ---------------------------------------------------------- get/put
    def get(self, key: str) -> Optional[Path]:
        """Cached audio path for ``key`` or None. Touches mtime on a hit."""
        for path in self._candidates(key):
            try:
                if path.is_file() and path.stat().st_size > 0:
                    os.utime(path, None)
                    self.hits += 1
                    return path
            except OSError:
                continue
        self.misses += 1
        return None

    def put(self, key: str, src: Path | str) -> Optional[Path]:
        """Copy ``src`` into the cache under ``key`` (keeps src's suffix).
        Returns the cached path, or None when caching failed."""
        src = Path(src)
        try:
            if not src.is_file() or src.stat().st_size == 0:
                return None
            self.dir.mkdir(parents=True, exist_ok=True)
            suffix = src.suffix or ".wav"
            dst = self.dir / f"{key}{suffix}"
            tmp = self.dir / f".{key}{suffix}.tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        except Exception:
            log.exception("speech cache put failed for %s", src)
            return None
        self.prune()
        return dst

    # ----------------------------------------------------------- prune
    def _entries(self) -> list[tuple[float, int, Path]]:
        out = []
        try:
            for p in self.dir.iterdir():
                if not p.is_file() or p.name.startswith("."):
                    continue
                st = p.stat()
                out.append((st.st_mtime, st.st_size, p))
        except OSError:
            pass
        return out

    def prune(self) -> int:
        """Evict least-recently-used files past the limits. Returns count."""
        with self._lock:
            entries = self._entries()
            total = sum(size for _, size, _ in entries)
            if len(entries) <= self.max_files and total <= self.max_bytes:
                return 0
            entries.sort()                      # oldest mtime first
            removed = 0
            for _, size, path in entries:
                if len(entries) - removed <= self.max_files and \
                        total <= self.max_bytes:
                    break
                try:
                    path.unlink()
                    removed += 1
                    total -= size
                except OSError:
                    log.warning("could not evict %s", path)
            if removed:
                log.info("speech cache pruned %d file(s)", removed)
            return removed

    def clear(self) -> int:
        with self._lock:
            n = 0
            for _, _, path in self._entries():
                try:
                    path.unlink()
                    n += 1
                except OSError:
                    pass
            return n

    def stats(self) -> dict:
        entries = self._entries()
        return {"dir": str(self.dir), "files": len(entries),
                "bytes": sum(s for _, s, _ in entries),
                "hits": self.hits, "misses": self.misses,
                "checked": time.time()}
