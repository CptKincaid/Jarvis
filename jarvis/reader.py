"""Read-aloud for Jarvis: clipboard, the X selection, a file, or given text.

"Jarvis, read the clipboard" / "read this" (the highlighted text) /
"read file ~/notes.md" / "read aloud: <text>". The text is chunked at
sentence boundaries into pieces the TTS accepts whole (``TTS`` truncates a
single utterance at ``MAX_SPEAK_LENGTH``) and queued in order; the FIFO
worker plays them back to back and ``TTS.stop()`` (barge-in, "quiet")
abandons the rest.

This is deliberately outside the two-sentence reply rule: a document
being read is not a conversational reply. Long texts are read one *part*
at a time (``max_part`` chars, about three minutes of speech); Jarvis
then offers "continue reading" for the rest so a mis-fire never locks the
speakers for ten minutes.
"""
from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("reader")


@dataclass
class ReadResult:
    """``ok`` → ``message`` is a status line (the reading itself is what
    the user hears); not ok → ``message`` is Jarvis's spoken excuse."""
    ok: bool
    message: str
    chunks: int = 0
    remaining: int = 0

MAX_CHUNK_CHARS = 380          # < TTS.MAX_SPEAK_LENGTH so nothing truncates
MAX_PART_CHARS = 2400          # one "part" per request (~3 min of speech)
MAX_FILE_BYTES = 512_000
TEXT_SUFFIXES = {".txt", ".md", ".rst", ".log", ".py", ".json", ".yaml",
                 ".yml", ".toml", ".cfg", ".ini", ".csv", ".sh", ".html",
                 ".htm", ".xml", ".tex", ".org", ""}

CONTINUE_PROMPT = ("That's the first part, sir. Say 'continue reading' "
                   "for the rest.")
FINISHED_LINE = "That's the end of it, sir."

_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")


def _xclip(selection: str, run=subprocess.run) -> str:
    """Text of the X clipboard/primary selection ('' on failure)."""
    try:
        out = run(["xclip", "-selection", selection, "-o"],
                  capture_output=True, text=True, timeout=3)
    except Exception:
        log.warning("xclip %s read failed", selection, exc_info=True)
        return ""
    if getattr(out, "returncode", 1) != 0:
        return ""
    return out.stdout or ""


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into TTS-sized chunks at sentence/paragraph breaks;
    an over-long sentence is split at the last space before the limit."""
    text = re.sub(r"[ \t]+", " ", (text or "")).strip()
    if not text:
        return []
    chunks: list[str] = []
    buf = ""
    for piece in _SENT_SPLIT.split(text):
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            if cut < max_chars // 3:
                cut = max_chars
            head, piece = piece[:cut].strip(), piece[cut:].strip()
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(head)
        if not piece:
            continue
        if buf and len(buf) + 1 + len(piece) > max_chars:
            chunks.append(buf)
            buf = piece
        else:
            buf = f"{buf} {piece}" if buf else piece
    if buf:
        chunks.append(buf)
    return chunks


def looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    control = sum(1 for b in sample if b < 9 or 13 < b < 32)
    return control / len(sample) < 0.05


class ReadAloud:
    """Queue text into the TTS in readable chunks, one part at a time."""

    def __init__(self, tts, max_part: int = MAX_PART_CHARS,
                 max_chunk: int = MAX_CHUNK_CHARS,
                 run: Callable = subprocess.run,
                 search_dirs: Optional[list[Path]] = None):
        self._tts = tts
        self.max_part = max_part
        self.max_chunk = max_chunk
        self._run = run
        self._pending: list[str] = []
        self._label = ""
        self._lock = threading.Lock()
        self._search_dirs = search_dirs or [Path.cwd(), Path.home(),
                                            Path.home() / "Jarvis"]

    # ---------------------------------------------------------- sources
    def read_clipboard(self) -> ReadResult:
        text = _xclip("clipboard", self._run)
        if not text.strip():
            return ReadResult(False, "The clipboard is empty, sir.")
        return self.read_text(text, label="the clipboard")

    def read_selection(self) -> ReadResult:
        text = _xclip("primary", self._run)
        if not text.strip():
            return ReadResult(False, "Nothing is highlighted, sir.")
        return self.read_text(text, label="the selection")

    def resolve_file(self, name: str) -> Optional[Path]:
        name = (name or "").strip().strip("'\"")
        if not name:
            return None
        p = Path(name).expanduser()
        candidates = [p] if p.is_absolute() else [d / p for d in self._search_dirs]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def read_file(self, name: str) -> ReadResult:
        path = self.resolve_file(name)
        if path is None:
            return ReadResult(False, f"I can't find a file called {name}, sir.")
        try:
            data = path.read_bytes()[:MAX_FILE_BYTES]
        except OSError:
            log.exception("read_file failed: %s", path)
            return ReadResult(False, f"I couldn't open {path.name}, sir.")
        if path.suffix.lower() not in TEXT_SUFFIXES or not looks_like_text(data):
            return ReadResult(False, f"{path.name} isn't a text file, sir.")
        text = data.decode("utf-8", errors="replace")
        return self.read_text(text, label=path.name)

    # ------------------------------------------------------------ core
    def read_text(self, text: str, label: str = "") -> ReadResult:
        """Queue ``text`` (first part now, the rest behind 'continue')."""
        chunks = chunk_text(text, self.max_chunk)
        if not chunks:
            return ReadResult(False, "There's nothing to read, sir.")
        with self._lock:
            self._label = label
            self._pending = chunks
        return self._speak_next_part()

    def continue_reading(self) -> ReadResult:
        with self._lock:
            if not self._pending:
                return ReadResult(False, "That was all of it, sir.")
        return self._speak_next_part()

    def _speak_next_part(self) -> ReadResult:
        with self._lock:
            part: list[str] = []
            used = 0
            while self._pending and (not part or
                                     used + len(self._pending[0]) <= self.max_part):
                chunk = self._pending.pop(0)
                part.append(chunk)
                used += len(chunk)
            remaining = len(self._pending)
            label = self._label or "text"
        for chunk in part:
            self._tts.speak(chunk)
        if remaining:
            self._tts.speak(CONTINUE_PROMPT)
            return ReadResult(True, f"Reading {label}: {len(part)} chunk(s), "
                                    f"{remaining} more pending",
                              chunks=len(part), remaining=remaining)
        return ReadResult(True, f"Reading {label}: {len(part)} chunk(s)",
                          chunks=len(part))

    def stop(self) -> None:
        with self._lock:
            self._pending = []
        try:
            self._tts.stop()
        except Exception:
            log.exception("reader stop failed")

    @property
    def pending_chunks(self) -> int:
        with self._lock:
            return len(self._pending)
