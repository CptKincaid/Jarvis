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

Documents (.pdf / .docx) go through ``jarvis.tools.docs.extract_text``
(pdftotext / stdlib zip) and are unwrapped first: pdftotext keeps the
page's hard line breaks and form feeds, and read as-is those land as
pauses mid-sentence. ``resolve_document`` also matches a spoken name
("the biosensors lab handout") against the file names in the search
folders, which include the docs folders, so a PDF never has to be named
by its exact file name.

Steering (2026-08-30): "skip" cuts the chunk being spoken and the FIFO
carries on; "back" re-reads the previous chunk; "pause" holds the rest;
"go on" resumes. The reader owns the cursor -- ``TTS.speak`` hands back
each chunk's done-event, so the reader knows which chunk is in the air
without the TTS growing per-item ids -- and keeps its own history of
spoken chunks for "back". The verbs are only honoured while ``active``
(reading or paused) so they never collide with the Spotify transport.
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
DOC_SUFFIXES = {".pdf", ".docx"}       # via jarvis.tools.docs.extract_text
NO_TEXT_LINE = "I couldn't get any text out of {name}, sir."

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


def unwrap_text(text: str) -> str:
    """Undo pdftotext's page layout for speech: form feeds become
    paragraph breaks, a lone newline inside a paragraph becomes a space
    (a hard-wrapped line is not a sentence end) and a hyphen split across
    a wrap is joined. Paragraph breaks (blank lines) survive."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)       # "bio-\nsensor"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


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
        # Chunks handed to the TTS for the current part, with the done
        # event speak() returned (None when the TTS gave none -- counted as
        # already spoken, since nothing can be steered without a handle).
        self._queued: list[tuple[str, Optional[threading.Event]]] = []
        self._prompt_ev: Optional[threading.Event] = None
        self._spoken: list[str] = []          # history for "back"
        self._paused = False
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

    def resolve_document(self, name: str) -> Optional[Path]:
        """``resolve_file`` first; failing that, the best fuzzy match of a
        spoken name against the readable files in the search folders
        (the docs folders included). None when nothing is close."""
        path = self.resolve_file(name)
        if path is not None:
            return path
        from jarvis.tools.docs import match_name    # lazy: docs pulls chromadb hints
        candidates: dict[str, Path] = {}
        for d in self._search_dirs:
            try:
                if not d.is_dir():
                    continue
                # Depth-limited on purpose: ~ holds a whole filesystem of
                # names, and a spoken title must not resolve to a stray
                # README three levels down a checkout.
                for p in list(d.iterdir()) + [q for sub in d.iterdir()
                                              if sub.is_dir() and not sub.name.startswith(".")
                                              for q in sub.iterdir()]:
                    if p.is_file() and not p.name.startswith(".") and \
                            p.suffix.lower() in DOC_SUFFIXES | (TEXT_SUFFIXES - {""}):
                        candidates.setdefault(p.name, p)
            except OSError:
                log.debug("resolve_document: cannot list %s", d, exc_info=True)
        best = match_name(name, list(candidates))
        return candidates[best] if best else None

    def document_text(self, path: Path) -> str:
        """The spoken-ready text of a file: documents through extract_text
        and unwrap_text, text files as they are. '' when unreadable."""
        suffix = path.suffix.lower()
        if suffix in DOC_SUFFIXES:
            from jarvis.tools.docs import extract_text
            return unwrap_text(extract_text(path))[:MAX_FILE_BYTES]
        try:
            data = path.read_bytes()[:MAX_FILE_BYTES]
        except OSError:
            log.exception("document_text failed: %s", path)
            return ""
        if suffix not in TEXT_SUFFIXES or not looks_like_text(data):
            return ""
        return data.decode("utf-8", errors="replace")

    def read_file(self, name: str) -> ReadResult:
        path = self.resolve_file(name)
        if path is None:
            return ReadResult(False, f"I can't find a file called {name}, sir.")
        return self.read_document(path)

    def read_document(self, path: Path) -> ReadResult:
        """Read a resolved file: .pdf/.docx through extract_text (the
        "isn't a text file" refusal used to catch them), text as before."""
        if path.suffix.lower() in DOC_SUFFIXES:
            text = self.document_text(path)
            if not text.strip():
                return ReadResult(False, NO_TEXT_LINE.format(name=path.name))
            return self.read_text(text, label=path.name)
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
            self._queued = []
            self._prompt_ev = None
            self._spoken = []
            self._paused = False
        return self._speak_next_part()

    def continue_reading(self) -> ReadResult:
        with self._lock:
            self._paused = False
            if not self._pending:
                return ReadResult(False, "That was all of it, sir.")
        return self._speak_next_part()

    def _speak_next_part(self) -> ReadResult:
        with self._lock:
            still = self._fold_queued()
            part: list[str] = []
            used = 0
            while self._pending and (not part or
                                     used + len(self._pending[0]) <= self.max_part):
                chunk = self._pending.pop(0)
                part.append(chunk)
                used += len(chunk)
            remaining = len(self._pending)
            label = self._label or "text"
            self._paused = False
        queued = [(chunk, self._tts.speak(chunk)) for chunk in part]
        prompt_ev = self._tts.speak(CONTINUE_PROMPT) if remaining else None
        with self._lock:
            self._queued = still + queued
            self._prompt_ev = prompt_ev
        if remaining:
            return ReadResult(True, f"Reading {label}: {len(part)} chunk(s), "
                                    f"{remaining} more pending",
                              chunks=len(part), remaining=remaining)
        return ReadResult(True, f"Reading {label}: {len(part)} chunk(s)",
                          chunks=len(part))

    def stop(self) -> None:
        with self._lock:
            self._pending = []
            self._queued = []
            self._prompt_ev = None
            self._spoken = []
            self._paused = False
        try:
            self._tts.stop()
        except Exception:
            log.exception("reader stop failed")

    @property
    def pending_chunks(self) -> int:
        with self._lock:
            return len(self._pending)

    # -------------------------------------------------------- steering
    @staticmethod
    def _unplayed(ev: Optional[threading.Event]) -> bool:
        return ev is not None and not ev.is_set()

    def _fold_queued(self) -> list[tuple[str, Optional[threading.Event]]]:
        """Under the lock: move the chunks the TTS has finished (or dropped)
        into the spoken history; return the ones still in the air."""
        still = []
        for chunk, ev in self._queued:
            if self._unplayed(ev):
                still.append((chunk, ev))
            else:
                self._spoken.append(chunk)
        del self._spoken[:-200]
        self._queued = []
        return still

    @property
    def active(self) -> bool:
        """True while a part is being spoken or the reading is paused --
        the window in which "skip", "back", "pause" and "go on" belong to
        the reader rather than to Spotify."""
        with self._lock:
            if self._paused:
                return True
            if self._unplayed(self._prompt_ev):
                return True
            return any(self._unplayed(ev) for _, ev in self._queued)

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def skip(self) -> ReadResult:
        """Cut the chunk being spoken; the rest of the part follows at once
        (it is already in the TTS FIFO). Paused: drop the next held chunk."""
        with self._lock:
            if self._paused:
                if not self._pending:
                    return ReadResult(False, "That was all of it, sir.")
                self._spoken.append(self._pending.pop(0))
                return ReadResult(True, "Skipped one (paused)",
                                  remaining=len(self._pending))
            playing = any(self._unplayed(ev) for _, ev in self._queued)
        if not playing:
            return ReadResult(False, "Nothing to skip, sir.")
        try:
            self._tts.skip_current()
        except Exception:
            log.exception("reader skip failed")
            return ReadResult(False, "I couldn't skip that, sir.")
        return ReadResult(True, "Skipped")

    def back(self) -> ReadResult:
        """Re-read the previous chunk, then carry on from where we were.
        With nothing spoken yet the current chunk starts over. Paused: move
        the cursor back one and stay paused."""
        with self._lock:
            if self._paused:
                if not self._spoken:
                    return ReadResult(False, "We're at the start, sir.")
                self._pending.insert(0, self._spoken.pop())
                return ReadResult(True, "Back one (paused)",
                                  remaining=len(self._pending))
            still = [chunk for chunk, _ in self._fold_queued()]
            prev = self._spoken.pop() if self._spoken else None
            if prev is None and not still:
                return ReadResult(False, "There's nothing to go back to, sir.")
            self._pending = ([prev] if prev else []) + still + self._pending
            self._prompt_ev = None
        try:
            self._tts.stop()            # cut the current chunk; the FIFO is re-queued below
        except Exception:
            log.exception("reader back failed")
        return self._speak_next_part()

    def pause(self) -> ReadResult:
        """Hold the reading: the chunk being spoken is cut and put back at
        the front (a chunk is ~25 s, so it restarts on "go on"); everything
        queued behind it is held with the parts still to come."""
        with self._lock:
            if self._paused:
                return ReadResult(False, "Already paused, sir.")
            still = [chunk for chunk, _ in self._fold_queued()]
            self._pending = still + self._pending
            self._prompt_ev = None
            self._paused = True
            held = len(self._pending)
        try:
            self._tts.stop()
        except Exception:
            log.exception("reader pause failed")
        return ReadResult(True, "Paused, sir.", remaining=held)

    def resume(self) -> ReadResult:
        """"Go on" after a pause: the held chunks (front of the queue) are
        spoken next. Not paused -> None, so the caller can give "go on" its
        usual meaning (the next part)."""
        with self._lock:
            if not self._paused:
                return None
        return self.continue_reading()
