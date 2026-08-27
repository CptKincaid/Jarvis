"""TTS pronunciation dictionary for Jarvis.

Both engines mangle the workshop's jargon: XTTS spells "VSS" as a word,
Edge reads "GB10" as "gee-bee-ten-ish" and "Ollama" with a hard O. The
dictionary rewrites those tokens into how Jarvis should *say* them, right
before synthesis (``TTS._speak_sync``), so the transcript still shows the
real spelling while the voice says the right thing.

Two layers:
- ``DEFAULT_PRONUNCIATIONS`` — shipped jargon (whole-token, case-sensitive).
- the user file ``~/.aiws_trainer/tts_pronunciations.json`` — overrides and
  additions, edited by "Jarvis, pronounce X as Y" (``add()``) or by hand.
  Reloaded automatically when its mtime changes.

Matching is whole-token: a key matches only when it is not glued to other
word characters ("GPU" matches in "the GPU," but not in "GPUs" unless the
plural is its own entry). Keys are case-sensitive unless they are written
in lower case, in which case they match any casing ("nvidia" also matches
"Nvidia" and "NVIDIA").

Usage:
    from jarvis import pronounce
    pronounce.apply("VSS runs on the GB10")   # -> "V S S runs on the G B ten"
    pronounce.get().add("Peyrovi", "pay-ROH-vee")
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("pronounce")

USER_FILE = PATHS.AIWS / "tts_pronunciations.json"

# spelled -> spoken. Keep entries to things both engines actually get wrong;
# ordinary English needs no help. Lower-case keys match any casing.
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    # the workshop
    "VSS": "V S S",
    "GB10": "G B ten",
    "DGX": "D G X",
    "nvidia": "en-vidia",
    "GPU": "G P U",
    "GPUs": "G P Us",
    "CPU": "C P U",
    "CUDA": "kooda",
    "aarch64": "arm sixty-four",
    "HDMI": "H D M I",
    "X11": "X eleven",
    "GNOME": "nome",
    "ollama": "oh-llama",
    "llama3.2": "llama three point two",
    "qwen": "kwen",
    "XTTS": "X T T S",
    "TTS": "T T S",
    "STT": "S T T",
    "ASR": "A S R",
    "ChromaDB": "chroma D B",
    "SAM3": "sam three",
    "ONNX": "onyx",
    "YOLO": "yolo",
    "ReID": "re I D",
    "RTSP": "R T S P",
    "MJPEG": "M J peg",
    "KPI": "K P I",
    "KPIs": "K P Is",
    "AGV": "A G V",
    "AGVs": "A G Vs",
    "VLM": "V L M",
    "LLM": "L L M",
    "MoE": "M O E",
    "JARVIS": "Jarvis",
    # tools
    "pytorch": "pie torch",
    "pytest": "pie test",
    "xdotool": "X do tool",
    "xclip": "X clip",
    "nmcli": "N M C L I",
    "ffmpeg": "F F M peg",
    "tkinter": "T K inter",
    "Tk": "tee kay",
    "sudo": "soo-doo",
    "JSON": "jason",
    "YAML": "yammel",
    "SSH": "S S H",
    "API": "A P I",
    "URL": "U R L",
    "ETA": "E T A",
    "CLI": "C L I",
    "GUI": "gooey",
    "OK": "okay",
    "ok": "okay",
    "PID": "P I D",
}

# Symbols that have no word boundary; applied after the token pass.
DEFAULT_SYMBOLS: dict[str, str] = {
    "%": " percent",
    "&": " and ",
    "°C": " degrees Celsius",
    "°F": " degrees Fahrenheit",
    "~/": "home slash ",
}


class Pronunciations:
    """A pronunciation table: shipped defaults + a user JSON file."""

    def __init__(self, path: Optional[Path] = None,
                 defaults: Optional[dict] = None,
                 symbols: Optional[dict] = None):
        self.path = Path(path) if path else USER_FILE
        self._defaults = dict(DEFAULT_PRONUNCIATIONS if defaults is None
                              else defaults)
        self._symbols = dict(DEFAULT_SYMBOLS if symbols is None else symbols)
        self._user: dict[str, str] = {}
        self._mtime: float = -1.0
        self._rx: Optional[re.Pattern] = None
        self._table: dict[str, str] = {}
        self._lower: dict[str, str] = {}
        self._lock = threading.Lock()
        self.load()

    # ------------------------------------------------------------ file
    def load(self) -> None:
        """(Re)load the user file. Missing file = defaults only."""
        with self._lock:
            user: dict[str, str] = {}
            mtime = -1.0
            try:
                if self.path.exists():
                    mtime = self.path.stat().st_mtime
                    data = json.loads(self.path.read_text())
                    if isinstance(data, dict):
                        user = {str(k).strip(): str(v) for k, v in data.items()
                                if str(k).strip()}
                    else:
                        log.warning("pronunciation file is not an object: %s",
                                    self.path)
            except Exception:
                log.exception("pronunciation file unreadable: %s", self.path)
            self._user = user
            self._mtime = mtime
            self._rebuild()

    def save(self) -> None:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._user, indent=2,
                                          ensure_ascii=False, sort_keys=True))
                tmp.replace(self.path)
                self._mtime = self.path.stat().st_mtime
            except Exception:
                log.exception("pronunciation save failed: %s", self.path)

    def _maybe_reload(self) -> None:
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else -1.0
        except OSError:
            return
        if mtime != self._mtime:
            self.load()

    # ----------------------------------------------------------- table
    def _rebuild(self) -> None:
        table = dict(self._defaults)
        # A user entry whose key differs only by case replaces the default.
        for key, spoken in self._user.items():
            for existing in [k for k in table if k.lower() == key.lower()]:
                del table[existing]
            table[key] = spoken
        # An empty spoken value means "say it as written" (removes a default).
        table = {k: v for k, v in table.items() if v.strip()}
        self._table = table
        self._lower = {k.lower(): v for k, v in table.items() if k == k.lower()}
        if not table:
            self._rx = None
            return
        alts = sorted((re.escape(k) for k in table), key=len, reverse=True)
        # Whole-token: not glued to letters/digits/underscore on either side.
        self._rx = re.compile(r"(?<![\w])(?:%s)(?![\w])" % "|".join(alts),
                              re.IGNORECASE)

    def items(self) -> dict[str, str]:
        self._maybe_reload()
        return dict(self._table)

    def user_items(self) -> dict[str, str]:
        self._maybe_reload()
        return dict(self._user)

    def add(self, word: str, spoken: str) -> None:
        """Add/override one entry and persist it."""
        word = (word or "").strip()
        spoken = (spoken or "").strip()
        if not word:
            raise ValueError("word must not be empty")
        with self._lock:
            self._user[word] = spoken
            self._rebuild()
        self.save()
        log.info("pronunciation: %r -> %r", word, spoken)

    def remove(self, word: str) -> bool:
        with self._lock:
            hit = [k for k in self._user if k.lower() == word.lower()]
            for k in hit:
                del self._user[k]
            self._rebuild()
        if hit:
            self.save()
        return bool(hit)

    # ----------------------------------------------------------- apply
    def _lookup(self, token: str) -> Optional[str]:
        if token in self._table:
            return self._table[token]
        return self._lower.get(token.lower())

    def apply(self, text: str) -> str:
        """Rewrite tokens/symbols in ``text`` into their spoken forms."""
        if not text:
            return text
        self._maybe_reload()
        rx = self._rx
        if rx is not None:
            def _sub(m):
                spoken = self._lookup(m.group(0))
                return spoken if spoken is not None else m.group(0)
            text = rx.sub(_sub, text)
        for sym, spoken in self._symbols.items():
            if sym in text:
                text = text.replace(sym, spoken)
        return re.sub(r"[ \t]{2,}", " ", text).strip()


_default: Optional[Pronunciations] = None
_default_lock = threading.Lock()


def get() -> Pronunciations:
    """The process-wide table (user file at ``USER_FILE``)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Pronunciations()
        return _default


def apply(text: str) -> str:
    return get().apply(text)
