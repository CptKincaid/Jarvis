"""Shared timestamp logger for the Jarvis modules.

Every module previously had its own copy of _log() writing to
/tmp/vss_voice/gui_debug.log with a slightly different prefix. This
module replaces them all. Use get_logger("TTS") / get_logger("STT") /
etc. to get a prefix-bound logger.
"""

from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/tmp/vss_voice")
LOG_FILE = LOG_DIR / "gui_debug.log"


def _write(line: str):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_logger(prefix: str):
    """Return a function that logs with the given bracketed prefix."""
    def _log(msg):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _write(f"{ts} [{prefix}] {msg}")
    return _log
