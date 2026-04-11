#!/usr/bin/env python3
"""Screen capture daemon for Claude Code visual debugging.

Captures your screen on hotkey press, shows a desktop notification,
and auto-sends "check the screen" to the Claude Code terminal.

Usage:
    python scripts/utilities/screen_capture.py           # Interactive (hotkeys)
    python scripts/utilities/screen_capture.py --daemon   # Background (hotkeys only)
    python scripts/utilities/screen_capture.py --once     # Single capture and exit

Controls:
    Ctrl+Shift+S  — Capture full screen → notify → auto-send to Claude
    Ctrl+Shift+Q  — Quit daemon

Screenshots saved to: /tmp/vss_screen/latest.png
"""

import os
import sys
import signal
import subprocess
from pathlib import Path
from datetime import datetime

CAPTURE_DIR = Path("/tmp/vss_screen")
LATEST = CAPTURE_DIR / "latest.png"

# Claude Code prefixes terminal titles with spinner/status Unicode characters.
# Rather than enumerate them all, detect any non-ASCII first char followed by a space.
def _is_claude_title(name):
    return bool(name) and ord(name[0]) > 127 and len(name) > 1 and name[1] == ' '


def capture_screen():
    """Capture the full screen using PIL."""
    from PIL import ImageGrab
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M%S")
    timestamped = CAPTURE_DIR / f"screen_{timestamp}.png"

    img = ImageGrab.grab()
    img.save(str(LATEST))
    img.save(str(timestamped))

    # Keep only last 20 captures to avoid filling /tmp
    captures = sorted(CAPTURE_DIR.glob("screen_*.png"))
    for old in captures[:-20]:
        old.unlink(missing_ok=True)

    _log(f"Screenshot saved: {LATEST}")

    # Desktop notification
    try:
        subprocess.Popen(
            ["notify-send", "-i", "camera-photo", "-t", "2000",
             "Screenshot Captured",
             f"Saved to {LATEST}\nAuto-sending to Claude..."],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _log("notify-send dispatched")
    except FileNotFoundError:
        _log("notify-send not found")

    # Auto-send "check the screen" to Claude Code terminal
    _send_to_claude_terminal()

    return str(LATEST)


def _find_claude_terminal():
    """Find the Claude Code terminal window by its spinner-prefixed title."""
    try:
        result = subprocess.run(
            ["xdotool", "search", "--class", "terminal"],
            capture_output=True, text=True, timeout=2
        )
        wids = result.stdout.strip().splitlines()
        _log(f"Found {len(wids)} terminal windows")
        for wid in wids:
            wid = wid.strip()
            if not wid:
                continue
            name_result = subprocess.run(
                ["xdotool", "getwindowname", wid],
                capture_output=True, text=True, timeout=2
            )
            name = name_result.stdout.strip()
            first_char = name[0] if name else ''
            is_claude = _is_claude_title(name)
            _log(f"  WID={wid} name={name!r} first={first_char!r} ord={ord(first_char) if first_char else 0} match={is_claude}")
            if is_claude:
                return wid
    except Exception as e:
        _log(f"_find_claude_terminal EXCEPTION: {e}")
    return None


def _log(msg):
    """Write debug log."""
    with open("/tmp/vss_screen/debug.log", "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")


def _find_voice_target():
    """Fall back to the voice input GUI's saved target window."""
    try:
        import json
        settings_file = Path.home() / ".aiws_trainer" / "voice_settings.json"
        if not settings_file.exists():
            return None
        data = json.loads(settings_file.read_text())
        target_name = data.get("target_name")
        if not target_name:
            return None
        # Search for window matching the saved target name
        result = subprocess.run(
            ["xdotool", "search", "--name", target_name],
            capture_output=True, text=True, timeout=2,
        )
        wids = result.stdout.strip().splitlines()
        if wids:
            wid = wids[0].strip()
            _log(f"Voice target fallback: '{target_name}' → WID {wid}")
            return wid
    except Exception as e:
        _log(f"Voice target fallback error: {e}")
    return None


def _send_to_claude_terminal():
    """Focus Claude terminal and type 'check the screen'.

    Falls back to the voice input GUI's saved target window if no
    Claude terminal is found.
    """
    _log("_send_to_claude_terminal called")

    wid = _find_claude_terminal()
    _log(f"Found claude terminal WID: {wid}")
    if not wid:
        wid = _find_voice_target()
        _log(f"Voice target fallback WID: {wid}")
    if not wid:
        _log("No target window found, aborting")
        return

    try:
        # Put the message on clipboard (try xclip, then xsel)
        msg = "check the screen"
        copied = False
        for clip_cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
            try:
                proc = subprocess.Popen(
                    clip_cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                proc.communicate(input=msg.encode(), timeout=2)
                copied = True
                _log(f"Copied '{msg}' to clipboard via {clip_cmd[0]}")
                break
            except FileNotFoundError:
                continue
        if not copied:
            _log("No clipboard tool found (xclip/xsel) — install: sudo apt install xclip")

        # Focus the Claude terminal and auto-paste + submit
        r1 = subprocess.run(
            ["xdotool", "windowactivate", "--sync", wid],
            capture_output=True, text=True, timeout=2,
        )
        _log(f"windowactivate exit={r1.returncode} stderr={r1.stderr!r}")

        import time
        time.sleep(0.3)  # Let window fully focus

        # Do NOT use --window: it sends XSendEvent synthetic events which
        # gnome-terminal (and many others) silently ignore.
        # Without --window, xdotool uses XTest extension = real input.
        # --clearmodifiers releases Ctrl+Shift still held from the hotkey.
        r2 = subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "15",
             "check the screen"],
            capture_output=True, text=True, timeout=5,
        )
        _log(f"type exit={r2.returncode} stderr={r2.stderr!r}")
        _log("Typed 'check the screen' — user submits when ready")
    except Exception as e:
        _log(f"EXCEPTION: {e}")


def _run_xlib_listener():
    """Listen for Ctrl+Shift+S using Xlib record extension (no root needed)."""
    try:
        from Xlib import X, XK, display
        from Xlib.ext import record
        from Xlib.protocol import rq
    except ImportError:
        _run_file_watch_mode()
        return

    local_dpy = display.Display()
    record_dpy = display.Display()

    ctrl_held = False
    shift_held = False

    def callback(reply):
        nonlocal ctrl_held, shift_held
        if reply.category != record.FromServer:
            return
        if reply.client_swapped:
            return
        data = reply.data
        while len(data):
            event, data = rq.EventField(None).parse_binary_value(
                data, record_dpy.display, None, None
            )
            if event.type == X.KeyPress:
                keycode = event.detail
                keysym = local_dpy.keycode_to_keysym(keycode, 0)
                if keysym in (XK.XK_Control_L, XK.XK_Control_R):
                    ctrl_held = True
                elif keysym in (XK.XK_Shift_L, XK.XK_Shift_R):
                    shift_held = True
                elif keysym == XK.XK_s and ctrl_held and shift_held:
                    capture_screen()
                elif keysym == XK.XK_q and ctrl_held and shift_held:
                    os._exit(0)
            elif event.type == X.KeyRelease:
                keycode = event.detail
                keysym = local_dpy.keycode_to_keysym(keycode, 0)
                if keysym in (XK.XK_Control_L, XK.XK_Control_R):
                    ctrl_held = False
                elif keysym in (XK.XK_Shift_L, XK.XK_Shift_R):
                    shift_held = False

    ctx = record_dpy.record_create_context(
        0,
        [record.AllClients],
        [{
            'core_requests': (0, 0),
            'core_replies': (0, 0),
            'ext_requests': (0, 0, 0, 0),
            'ext_replies': (0, 0, 0, 0),
            'delivered_events': (0, 0),
            'device_events': (X.KeyPress, X.KeyRelease),
            'errors': (0, 0),
            'client_started': False,
            'client_died': False,
        }]
    )

    record_dpy.record_enable_context(ctx, callback)
    record_dpy.record_free_context(ctx)


def _run_file_watch_mode():
    """Fallback: watch for a trigger file."""
    import time
    trigger = CAPTURE_DIR / "trigger"
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        if trigger.exists():
            trigger.unlink(missing_ok=True)
            capture_screen()
        time.sleep(0.1)


def setup_interactive():
    """Interactive mode with banner."""
    print("\033[96m╔══════════════════════════════════════════╗\033[0m")
    print("\033[96m║   VSS Screen Capture for Claude Code     ║\033[0m")
    print("\033[96m╠══════════════════════════════════════════╣\033[0m")
    print("\033[96m║  Ctrl+Shift+S  →  Capture + send          ║\033[0m")
    print("\033[96m║  Ctrl+Shift+Q  →  Quit                   ║\033[0m")
    print("\033[96m╚══════════════════════════════════════════╝\033[0m")
    print(f"\nScreenshots → {LATEST}")
    print("Auto-pastes and sends 'check the screen' to Claude\n")

    _run_xlib_listener()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if "--once" in sys.argv:
        capture_screen()
    elif "--daemon" in sys.argv:
        _run_xlib_listener()
    else:
        setup_interactive()
