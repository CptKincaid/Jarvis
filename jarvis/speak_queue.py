"""Jarvis speak queue (V3) — in-process say() + external file watcher.

Evolves jarvis_speak_queue.py. Two paths into TTS:

- IN-PROCESS: ``say(text)`` calls the app-wired sink directly (normally
  ``tts.speak``) — no /tmp detour. The app wires it via ``set_sink(fn)``.
  If no sink is wired (or the sink raises), say() falls back to appending
  to the queue file so the text is not lost.

- EXTERNAL: other processes (VSS, Claude Code hooks, shell) still append
  lines to /tmp/vss_voice/speak_queue.txt:

      echo "Jarvis: Hello, how can I help?" >> /tmp/vss_voice/speak_queue.txt

  A watcher thread tails the file (1s poll, ported from the monolith's
  _watch_speak_queue), feeds new lines to the same sink, handles external
  truncation (size < pos → pos = 0), and truncates the file after reading
  so it never grows unbounded.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("speak_queue")

SPEAK_QUEUE = PATHS.SPEAK_QUEUE      # /tmp/vss_voice/speak_queue.txt

_sink: Optional[Callable[[str], None]] = None
_sink_lock = threading.Lock()


# ------------------------------------------------------------------ sink
def set_sink(fn: Optional[Callable[[str], None]]):
    """Wire the in-process TTS sink (the app passes e.g. ``tts.speak``).

    Pass None to unwire (say() then falls back to the queue file)."""
    global _sink
    with _sink_lock:
        _sink = fn


def get_sink() -> Optional[Callable[[str], None]]:
    with _sink_lock:
        return _sink


def say(text: str):
    """Queue a message for Jarvis to speak.

    In-process: delivered straight to the wired sink. Without a sink (or if
    the sink raises) the text is appended to the external queue file so a
    running watcher/app can pick it up."""
    text = (text or "").strip()
    if not text:
        return
    sink = get_sink()
    if sink is not None:
        try:
            sink(text)
            return
        except Exception:
            log.exception("say(): sink failed; falling back to file append")
    _append_to_file(text)


def _append_to_file(text: str):
    try:
        SPEAK_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        with open(SPEAK_QUEUE, "a") as f:
            f.write(text + "\n")
    except OSError:
        log.exception("speak-queue file append failed")


# --------------------------------------------------------------- watcher
class Watcher:
    """Tails the speak-queue file for messages from EXTERNAL producers.

    Ported from voice_input_gui._start_speak_queue_watcher /
    _watch_speak_queue (4689-4765) minus all GUI work: on boot the existing
    file content is skipped as stale backlog; each poll combines all new
    non-empty lines into one utterance and feeds it to the wired sink.

    V3 fixes: external truncation handled (size < pos → pos = 0) and the
    file is truncated after each successful read."""

    POLL_S = 1.0

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else SPEAK_QUEUE
        self._pos = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        # Skip any stale backlog present before we started (legacy 4695-4696).
        try:
            if self._path.exists():
                self._pos = self._path.stat().st_size
        except OSError:
            log.exception("speak-queue stat failed on start")
            self._pos = 0
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="speak-queue-watcher")
        self._thread.start()
        log.info("speak-queue watcher started on %s", self._path)

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2 * self.POLL_S)
        self._thread = None

    def _loop(self):
        while not self._stop.wait(self.POLL_S):
            self.poll_once()

    def poll_once(self):
        """One poll of the queue file; safe to call directly in tests."""
        try:
            if not self._path.exists():
                return
            size = self._path.stat().st_size
            if size < self._pos:          # file truncated externally
                self._pos = 0
            if size <= self._pos:
                return
            sink = get_sink()
            if sink is None:
                # No sink wired yet — leave the content in place for later.
                return
            with open(self._path, "r+") as f:
                f.seek(self._pos)
                new_lines = f.read()
                f.truncate(0)             # truncate after reading
            self._pos = 0
            # Combine all new lines into one utterance (legacy 4717-4720).
            combined = " ".join(
                line.strip() for line in new_lines.strip().splitlines()
                if line.strip()
            )
            if not combined:
                return
            log.info("talk-back queue: %.60s", combined)
            try:
                sink(combined)
            except Exception:
                log.exception("watcher sink failed for: %.60s", combined)
        except OSError:
            log.exception("speak-queue watcher poll failed")


_watcher: Optional[Watcher] = None


def start_watcher() -> Watcher:
    """Start (or restart) the module-level watcher; returns it."""
    global _watcher
    if _watcher is None:
        _watcher = Watcher()
    _watcher.start()
    return _watcher


def stop_watcher():
    if _watcher is not None:
        _watcher.stop()
