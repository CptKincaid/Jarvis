"""Is he at the desk right now?

The presence sentinel (jarvis/presence.py) answers a different question --
"is he in the flat" -- and on this box it cannot answer at all:
``presence.phone_ip`` and ``phone_mac`` are both empty in the live config,
so ``PresenceSentinel.configured()`` is False and ``start()`` logs "sentinel
idle". Anything gated on it would fire into an empty room.

X knows. ``XScreenSaverQueryInfo`` on :1 reports true idle milliseconds --
keyboard and mouse, whole session, no sudo, no polling of input devices,
no new hardware. ``libXss.so.1`` is present here. When the extension is
missing (a headless box, a Wayland session, no DISPLAY) the fallback is the
last turn Jarvis handled: ``turns.jsonl`` records ``at`` per voice turn, and
a man who spoke to Jarvis four minutes ago is at his desk.

**Fail open.** ``at_desk()`` answers True when NOTHING can tell -- no X, no
ledger, an unreadable file. A gate that says "away" on a box it cannot
measure is the inert gate this module exists to replace; the cost of being
wrong is a notes file created for a class he skipped, which is nothing.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import time
from pathlib import Path
from typing import Optional

from jarvis.logs import get_logger

log = get_logger("desk")

DEFAULT_IDLE_MIN = 15
TAIL_BYTES = 8192              # enough for the last few turns.jsonl records

_MISSING = object()
_XSS = _MISSING                # (libX11, libXss) once probed, or None


class _XScreenSaverInfo(ctypes.Structure):
    # /usr/include/X11/extensions/scrnsaver.h
    _fields_ = [("window", ctypes.c_ulong), ("state", ctypes.c_int),
                ("kind", ctypes.c_int), ("til_or_since", ctypes.c_ulong),
                ("idle", ctypes.c_ulong), ("event_mask", ctypes.c_ulong)]


def _libs():
    """(libX11, libXss) with the signatures set, or None. Probed once."""
    global _XSS
    if _XSS is not _MISSING:
        return _XSS
    _XSS = None
    try:
        x11_name = ctypes.util.find_library("X11")
        xss_name = ctypes.util.find_library("Xss")
        if not x11_name or not xss_name:
            log.info("desk: no libX11/libXss; idle falls back to the turn ledger")
            return None
        x11 = ctypes.CDLL(x11_name)
        xss = ctypes.CDLL(xss_name)
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        xss.XScreenSaverAllocInfo.restype = ctypes.POINTER(_XScreenSaverInfo)
        xss.XScreenSaverQueryExtension.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)]
        xss.XScreenSaverQueryInfo.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(_XScreenSaverInfo)]
        _XSS = (x11, xss)
    except Exception:                       # noqa: BLE001 - ctypes boundary
        log.info("desk: X idle unavailable", exc_info=True)
        _XSS = None
    return _XSS


def x_idle_ms(display: Optional[str] = None) -> Optional[int]:
    """Milliseconds since the last keyboard/mouse event on ``display``, or
    None when X cannot say. Read-only: it opens a connection, asks, closes."""
    libs = _libs()
    if libs is None:
        return None
    name = display or os.environ.get("DISPLAY") or ""
    if not name:
        return None
    x11, xss = libs
    dpy = None
    try:
        dpy = x11.XOpenDisplay(name.encode())
        if not dpy:
            return None
        ev, err = ctypes.c_int(), ctypes.c_int()
        if not xss.XScreenSaverQueryExtension(dpy, ctypes.byref(ev), ctypes.byref(err)):
            return None
        info = xss.XScreenSaverAllocInfo()
        if not info:
            return None
        try:
            if not xss.XScreenSaverQueryInfo(dpy, x11.XDefaultRootWindow(dpy), info):
                return None
            return int(info.contents.idle)
        finally:
            # XScreenSaverAllocInfo mallocs; a 45 s poll for the life of the
            # app leaks without this.
            x11.XFree(info)
    except Exception:                       # noqa: BLE001 - ctypes boundary
        log.debug("desk: X idle query failed", exc_info=True)
        return None
    finally:
        if dpy:
            try:
                x11.XCloseDisplay(dpy)
            except Exception:               # noqa: BLE001
                log.debug("desk: XCloseDisplay failed", exc_info=True)


def last_turn_at(path) -> Optional[float]:
    """The ``at`` timestamp of the newest record in turns.jsonl, or None.
    Only the tail is read: the ledger grows for the life of the box."""
    if not path:
        return None
    p = Path(path)
    try:
        size = p.stat().st_size
        with p.open("rb") as fh:
            if size > TAIL_BYTES:
                fh.seek(size - TAIL_BYTES)
                fh.readline()               # drop the partial first line
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        at = rec.get("at") if isinstance(rec, dict) else None
        if isinstance(at, (int, float)):
            return float(at)
    return None


def idle_seconds(display: Optional[str] = None, turns_path=None,
                 now: Optional[float] = None) -> Optional[float]:
    """Seconds since he last touched anything, or None when nothing knows.

    X first (it sees the keyboard, which Jarvis does not); the turn ledger
    only as a fallback, because a long silent stretch of typing would look
    like hours of absence to the ledger alone."""
    ms = x_idle_ms(display)
    if ms is not None:
        return max(0.0, ms / 1000.0)
    at = last_turn_at(turns_path)
    if at is None:
        return None
    return max(0.0, (time.time() if now is None else float(now)) - at)


def at_desk(idle_min: float = DEFAULT_IDLE_MIN, display: Optional[str] = None,
            turns_path=None, now: Optional[float] = None) -> bool:
    """True when he has been active within ``idle_min`` minutes -- and True
    when nothing can measure it (see the module docstring: fail open)."""
    try:
        limit = max(0.0, float(idle_min)) * 60.0
    except (TypeError, ValueError):
        limit = DEFAULT_IDLE_MIN * 60.0
    idle = idle_seconds(display, turns_path, now)
    if idle is None:
        return True
    return idle <= limit
