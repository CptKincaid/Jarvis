"""Unified logging for Jarvis V3 — replaces the four scattered _log() copies.

Rotating file at /tmp/vss_voice/jarvis.log plus the legacy gui_debug.log line
format kept for the error-context grep in context.py.
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path

LIVE_LOG_DIR = "/tmp/vss_voice"
ADHOC_LOG_DIR = "/tmp/jarvis-adhoc"


def _default_log_dir(env=None, main_name=None) -> str:
    """Where this process logs. JARVIS_LOG_DIR wins (the test suite sets it);
    otherwise only a `python -m jarvis.*` process (the app, voice_check) gets
    the LIVE directory. Any other importer -- a scratch script, a one-off
    harness, an agent poking at the code -- logs to ADHOC_LOG_DIR, because
    on 2026-08-29 such runs wrote "Auto-stop on silence" and tracebacks into
    the running app's log and were read as live events, twice."""
    if env is None:
        env = os.environ.get("JARVIS_LOG_DIR") or ""
    if env:
        return env
    if main_name is None:
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        main_name = getattr(spec, "name", "") or ""
    return LIVE_LOG_DIR if main_name.startswith("jarvis.") else ADHOC_LOG_DIR


LOG_DIR = Path(_default_log_dir())
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
