"""Unified logging for Jarvis V3 — replaces the four scattered _log() copies.

Rotating file at /tmp/vss_voice/jarvis.log plus the legacy gui_debug.log line
format kept for the error-context grep in context.py.
"""
import logging
import logging.handlers
import os
from pathlib import Path

# JARVIS_LOG_DIR redirects everything under /tmp/vss_voice (the test suite
# sets it so it never writes into the live app's log / speak queue).
LOG_DIR = Path(os.environ.get("JARVIS_LOG_DIR") or "/tmp/vss_voice")
LOG_FILE = LOG_DIR / "jarvis.log"

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S"))
        root = logging.getLogger("jarvis")
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
    except Exception:
        pass  # logging must never take the app down
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"jarvis.{name}")
