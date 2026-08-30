"""Desktop control for Jarvis V3 — xdotool windows, typing, tabs, media keys,
claude-terminal detection, screenshots.

Ported from the legacy voice_input_gui.py monolith (READ-ONLY source):
  _parse_desktop_action 3586-3663   -> parse_desktop_action + module tables
  _check_desktop_command 3548-3584  -> parse_command / DesktopControl.handle_command
  _execute_desktop_actions 3665-3752-> DesktopControl.execute
  _is_claude_title 4445-4448        -> is_claude_title
  _find_claude_terminal 4450-4473   -> find_claude_terminal (batched names)
  _type_text 4519-4586              -> DesktopControl.type_text
  _live_type_partial 4475-4517      -> DesktopControl.live_type_partial
  dictation typing 2622-2628        -> DesktopControl.type_dictation
  _get_window_list 5266-5296        -> list_windows (BATCHED: one search +
                                       one sh subprocess for all names)
  target pinning 5310-5428          -> DesktopControl pin/reset/restore/query
  _take_screenshot 3887-3933 +
  _type_then_screenshot 3854-3885   -> DesktopControl.screenshot (merged)
  "launch <app>" 3147-3170          -> launch_app

No tkinter, no widgets. The sleeps are intentional xdotool pacing and run on
the CALLER's thread — call from a worker thread, never the UI thread. Target
pin changes publish Status events on the bus.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from datetime import datetime

from jarvis.config import CONFIG, PATHS
from jarvis.events import Status, bus
from jarvis.logs import get_logger

log = get_logger("desktop")


# ------------------------------------------------------------------ tables
# Parse mappings from _parse_desktop_action (3586-3663). ORDER IS LOAD-BEARING:
# window prefixes -> tabs -> scroll -> click -> window mgmt -> media ->
# shortcuts -> wait, exactly as the original if-chain ran.
WINDOW_PREFIXES = ("switch to ", "go to ", "focus ", "open ")

TAB_KEYS = [                       # substring match, in order (3597-3604)
    ("next tab", "ctrl+Tab"),
    ("previous tab", "ctrl+shift+Tab"),
    ("prev tab", "ctrl+shift+Tab"),
    ("new tab", "ctrl+t"),
    ("close tab", "ctrl+w"),
]

WM_KEYS = [                        # substring match, in order (3629-3636)
    ("minimize", "super+h"),
    ("maximize", "super+Up"),
    ("full screen", "F11"),
    ("fullscreen", "F11"),
    ("close window", "alt+F4"),
]

MEDIA_KEYS = [                     # substring match, in order (3639-3644)
    ("volume up", "XF86AudioRaiseVolume"),
    ("volume down", "XF86AudioLowerVolume"),
    ("mute", "XF86AudioMute"),
]
MEDIA_PLAY_EXACT = {"play", "pause", "play pause"}   # exact match (3645-3646)
MEDIA_PLAY_KEY = "XF86AudioPlay"

SHORTCUT_KEYS = [                  # substring match, in order (3649-3656)
    ("copy", "ctrl+c"),
    ("paste", "ctrl+v"),
    ("undo", "ctrl+z"),
    ("redo", "ctrl+shift+z"),
    ("save", "ctrl+s"),
    ("select all", "ctrl+a"),
    ("find", "ctrl+f"),
]

# _check_desktop_command (3552-3561)
JARVIS_PREFIXES = ("jarvis ", "jarvis, ", "hey jarvis ", "hey jarvis, ")
CHAIN_SPLIT = re.compile(r"\s+and then\s+|\s+then\s+|\s+and\s+|,\s*")


# --------------------------------------------------------------- parsing
def parse_desktop_action(text):
    """Parse a single desktop action from text. Port of 3586-3663."""
    text = text.strip().lower()

    # Window switching: "switch to opera", "open terminal"
    for phrase in WINDOW_PREFIXES:
        if text.startswith(phrase):
            target = text[len(phrase):].strip()
            return ("window", target)

    # Tab controls
    for phrase, keys in TAB_KEYS:
        if phrase in text:
            return ("key", keys)

    # Scroll
    if "scroll down" in text:
        amount = 5
        m = re.search(r"(\d+)", text)
        if m:
            amount = int(m.group(1))
        return ("scroll", "down", amount)
    if "scroll up" in text:
        amount = 5
        m = re.search(r"(\d+)", text)
        if m:
            amount = int(m.group(1))
        return ("scroll", "up", amount)

    # Click
    if "double click" in text:
        return ("click", "double")
    if "right click" in text:
        return ("click", "right")
    if "click" in text:
        return ("click", "left")

    # Window management
    for phrase, keys in WM_KEYS:
        if phrase in text:
            return ("key", keys)

    # Media
    for phrase, keys in MEDIA_KEYS:
        if phrase in text:
            return ("key", keys)
    if text in MEDIA_PLAY_EXACT:
        return ("key", MEDIA_PLAY_KEY)

    # Keyboard shortcuts
    for phrase, keys in SHORTCUT_KEYS:
        if phrase in text:
            return ("key", keys)

    # Wait/pause
    m = re.match(r"wait\s+(\d+)", text)
    if m:
        return ("wait", int(m.group(1)))

    return None


def parse_command(text):
    """Parse a chained desktop command ("Jarvis, switch to opera and scroll
    down"). Returns a list of actions, or None if this is not a desktop
    command. Port of _check_desktop_command 3548-3575 (parsing half).
    """
    lower = text.strip().lower().rstrip(".")

    # Must start with "jarvis" prefix
    for prefix in JARVIS_PREFIXES:
        if lower.startswith(prefix):
            cmd_text = lower[len(prefix):].strip()
            break
    else:
        return None

    # Split on "and then", "then", "and", commas for chained commands
    parts = CHAIN_SPLIT.split(cmd_text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return None

    actions = []
    for part in parts:
        action = parse_desktop_action(part)
        if action:
            actions.append(action)
    return actions or None


# --------------------------------------------------- windows (batched)
# One sh subprocess resolves ALL window names (replaces the O(N) per-window
# getwindowname forks of 5285-5288); output is wid<TAB>name, alignment-safe.
_NAMES_SCRIPT = (
    'while IFS= read -r w; do '
    'n=$(xdotool getwindowname "$w" 2>/dev/null) || n=""; '
    'printf "%s\\t%s\\n" "$w" "$n"; '
    'done'
)


def _window_names(wids):
    """Fetch names for many window ids in a single subprocess."""
    wids = [w.strip() for w in wids if w.strip()]
    if not wids:
        return {}
    try:
        result = subprocess.run(
            ["sh", "-c", _NAMES_SCRIPT],
            input="\n".join(wids) + "\n",
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        log.exception("batched getwindowname failed")
        return {}
    names = {}
    for line in result.stdout.splitlines():
        wid, _, name = line.partition("\t")
        if wid.strip():
            names[wid.strip()] = name.strip()
    return names


def list_windows(exclude_wid=None):
    """List (wid, name) for all visible windows. Port of the INTENT of
    _get_window_list 5266-5296 with batched name resolution."""
    windows = []
    try:
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", ""],
            capture_output=True, text=True, timeout=3,
        )
        wids = [w.strip() for w in result.stdout.strip().splitlines()
                if w.strip() and w.strip() != str(exclude_wid or "")]
        names = _window_names(wids)
        for wid in wids:                      # preserve search order
            name = names.get(wid, "")
            if name and len(name) > 1:        # filter as 5290
                windows.append((wid, name))
    except Exception:
        log.exception("window list error")
    return windows


def is_claude_title(name):
    """Detect Claude Code terminal by its spinner-prefixed title.
    Port of _is_claude_title 4445-4448 (verbatim rule)."""
    return bool(name) and ord(name[0]) > 127 and len(name) > 1 and name[1] == ' '


def find_claude_terminal():
    """Find the Claude Code terminal window ID, or None.
    Port of _find_claude_terminal 4450-4473 with batched name fetch."""
    try:
        result = subprocess.run(
            ["xdotool", "search", "--class", "terminal"],
            capture_output=True, text=True, timeout=2,
        )
        wids = [w.strip() for w in result.stdout.strip().splitlines()
                if w.strip()]
        names = _window_names(wids)
        for wid in wids:                      # first match, search order
            name = names.get(wid, "")
            if is_claude_title(name):
                log.info("Found Claude terminal: WID=%s name=%r", wid, name)
                return wid
    except Exception:
        log.exception("Claude terminal search error")
    return None


# -------------------------------------------------------- primitives
def press_key(keys, timeout=2):
    """Send a key chord via xdotool."""
    try:
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", keys],
            capture_output=True, timeout=timeout,
        )
    except Exception:
        log.exception("key press error: %s", keys)


def launch_app(app):
    """Launch an application by name. Port of 3147-3170. Returns status text."""
    log.info("Launching: %s", app)
    try:
        subprocess.Popen(
            [app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return f"Launched {app}"
    except FileNotFoundError:
        # Try gtk-launch for .desktop apps
        try:
            subprocess.Popen(
                ["gtk-launch", app],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return f"Launched {app}"
        except Exception:
            log.warning("could not launch %r", app)
            return f"Could not find '{app}'"


# ------------------------------------------------------------ control
class DesktopControl:
    """Holds target-window state (pinning) and drives xdotool.

    All methods that sleep or fork are meant for worker threads. Target pin
    changes publish Status events; no widget access anywhere.
    """

    def __init__(self):
        self.target_wid = None            # pinned window id (str) or None
        self.target_name = None           # pinned window title or None
        # live-write state (was ad-hoc GUI attrs; port of 4475-4517 state)
        self._live_typed_chars = 0
        self._live_typed_text = ""

    # ------------------------------------------------- target pinning
    def pin_target(self, wid, name, persist=True):
        """Pin a specific window as the typing target (5310-5325)."""
        self.target_wid = str(wid)
        self.target_name = name
        short = name[:25] + "..." if len(name) > 25 else name
        bus.publish(Status(f"Target: {short}", "ok"))
        log.info("Target set to WID=%s name=%r", wid, name)
        if persist:
            CONFIG.update(target_name=name)

    def reset_target(self, persist=True):
        """Reset to auto-detect Claude terminal (5362-5370)."""
        self.target_wid = None
        self.target_name = None
        bus.publish(Status("Target: Auto (Claude)", "ok"))
        log.info("Target reset to Auto (Claude)")
        if persist:
            CONFIG.update(target_name="")

    def restore_target(self, name):
        """Re-find a previously pinned target window by name (5372-5383).
        Returns True if restored."""
        if not name:
            return False
        for wid, wname in list_windows():
            if name.lower() in wname.lower():
                self.target_wid = wid
                self.target_name = name
                log.info("Restored target: %s (WID %s)", name, wid)
                bus.publish(Status(f"Target: {name[:25]}", "info"))
                return True
        log.info("Could not restore target %r, using Auto", name)
        return False

    def target_by_query(self, query):
        """Find a window matching the spoken query and pin it as target.
        Port of _voice_target_window 5385-5428 (scoring verbatim).
        Returns the matched name, or None."""
        query = query.strip().lower()
        log.info("Voice target search: %r", query)
        best_wid = None
        best_name = None
        best_score = 0
        for wid, name in list_windows():
            name_lower = name.lower()
            if name_lower.startswith(query):                      # exact start
                score = 3
            elif re.search(r"\b" + re.escape(query) + r"\b", name_lower):
                score = 2                                          # word bound
            elif query in name_lower:
                score = 1                                          # substring
            else:
                continue
            if score > best_score:
                best_score = score
                best_wid = wid
                best_name = name
        if best_wid:
            # voice targeting did not persist in V1 — keep that
            self.pin_target(best_wid, best_name, persist=False)
            return best_name
        bus.publish(Status(f"No window matching '{query}'", "warn"))
        log.info("Voice target: no match for %r", query)
        return None

    # ---------------------------------------------------- execution
    def handle_command(self, text):
        """Full desktop-command pipeline: parse chained "jarvis ..." text and
        execute it on the CALLER's thread. Returns the summary text if it was
        a desktop command, else None. Port of _check_desktop_command
        3548-3584 + _execute_desktop_actions (execution now inline)."""
        actions = parse_command(text)
        if not actions:
            return None
        log.info("Desktop commands: %s", actions)
        bus.publish(Status(
            f"Executing {len(actions)} command{'s' if len(actions) > 1 else ''}...",
            "busy"))
        return self.execute(actions)

    def execute(self, actions):
        """Execute a chain of desktop actions with xdotool. Sleeps stay —
        run on a worker thread. Port of 3665-3752. Returns summary text."""
        for action in actions:
            if action[0] == "window":
                target = action[1]
                log.info("Desktop: switch to %r", target)
                try:
                    result = subprocess.run(
                        ["xdotool", "search", "--name", target],
                        capture_output=True, text=True, timeout=2,
                    )
                    wids = result.stdout.strip().splitlines()
                    if wids:
                        subprocess.run(
                            ["xdotool", "windowactivate", "--sync",
                             wids[0].strip()],
                            capture_output=True, timeout=2,
                        )
                        bus.publish(Status(f"Switched to {target}", "ok"))
                    else:
                        log.info("Desktop: window %r not found", target)
                except Exception:
                    log.exception("Desktop window error")
                time.sleep(0.3)

            elif action[0] == "key":
                keys = action[1]
                log.info("Desktop: key %s", keys)
                press_key(keys)
                time.sleep(0.2)

            elif action[0] == "scroll":
                direction = action[1]
                amount = action[2] if len(action) > 2 else 5
                btn = "5" if direction == "down" else "4"
                log.info("Desktop: scroll %s x%s", direction, amount)
                for _ in range(amount):
                    try:
                        subprocess.run(
                            ["xdotool", "click", "--clearmodifiers", btn],
                            capture_output=True, timeout=2,
                        )
                    except Exception:
                        log.exception("Desktop scroll error")
                    time.sleep(0.05)

            elif action[0] == "click":
                click_type = action[1]
                log.info("Desktop: %s click", click_type)
                try:
                    if click_type == "double":
                        subprocess.run(
                            ["xdotool", "click", "--clearmodifiers",
                             "--repeat", "2", "--delay", "100", "1"],
                            capture_output=True, timeout=2,
                        )
                    elif click_type == "right":
                        subprocess.run(
                            ["xdotool", "click", "--clearmodifiers", "3"],
                            capture_output=True, timeout=2,
                        )
                    else:
                        subprocess.run(
                            ["xdotool", "click", "--clearmodifiers", "1"],
                            capture_output=True, timeout=2,
                        )
                except Exception:
                    log.exception("Desktop click error")
                time.sleep(0.2)

            elif action[0] == "wait":
                secs = action[1]
                log.info("Desktop: wait %ss", secs)
                time.sleep(secs)

        summary = (f"Executed {len(actions)} "
                   f"command{'s' if len(actions) > 1 else ''}")
        bus.publish(Status(summary, "ok"))
        return summary

    # ------------------------------------------------------- typing
    def type_text(self, text, auto_enter=None):
        """Type text into the target window. Priority: pinned target >
        auto-detect Claude terminal > active window. Port of _type_text
        4519-4586 (Tk vars -> CONFIG). Blocking — worker thread only."""
        time.sleep(0.2)

        target_wid = None
        target_name = "active window"

        # 1. Pinned target (user picked a specific window)
        if self.target_wid:
            target_wid = self.target_wid
            short = (self.target_name or "")[:30]
            target_name = short or f"window {target_wid}"

        # 2. Auto-detect Claude terminal
        elif CONFIG.smart_target:
            wid = find_claude_terminal()
            if wid:
                target_wid = wid
                target_name = "Claude terminal"
            else:
                log.info("No Claude terminal found, typing into active window")

        # Focus the target window if we have one
        if target_wid:
            try:
                subprocess.run(
                    ["xdotool", "windowactivate", "--sync", target_wid],
                    capture_output=True, text=True, timeout=2,
                )
                time.sleep(0.15)
            except Exception:
                log.exception("Window focus error")
                target_name = "active window"

        if auto_enter is None:
            auto_enter = CONFIG.auto_enter

        try:
            # If live writing was active, backspace the partial first
            if CONFIG.live_write and self._live_typed_chars > 0:
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers"]
                    + ["BackSpace"] * self._live_typed_chars,
                    timeout=10, capture_output=True,
                )
                self._live_typed_chars = 0
                self._live_typed_text = ""
                time.sleep(0.05)

            # Type the final text
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "5", text],
                timeout=10, capture_output=True,
            )
            # Auto-press Enter if enabled
            if auto_enter:
                time.sleep(0.05)
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "Return"],
                    timeout=2, capture_output=True,
                )
                log.info("Text typed + Enter into %s", target_name)
            else:
                log.info("Text typed into %s", target_name)
            bus.publish(Status(
                f"Typed {len(text)} chars → {target_name}", "ok"))
        except Exception:
            log.exception("xdotool type error")

    def type_dictation(self, text):
        """Dictation-mode typing: text + trailing space into the active
        window, in a background thread. Port of 2622-2628 verbatim."""
        def _type(t=text):
            try:
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "5",
                     t + " "],
                    timeout=10, capture_output=True,
                )
            except Exception:
                log.exception("dictation type error")
        threading.Thread(target=_type, daemon=True).start()

    def live_type_partial(self, text, recording=True):
        """Type new words from a partial transcription into the target
        window. Port of _live_type_partial 4475-4517 (diff/replace logic
        verbatim; state now explicit fields)."""
        if not recording or not text:
            return

        old = self._live_typed_text
        # Find what's new — only type the difference
        if text.startswith(old):
            new_part = text[len(old):]
        elif old and text[:len(old) // 2] == old[:len(old) // 2]:
            # Partial changed in the middle — Whisper re-interpreted.
            new_part = None  # Signal to replace
        else:
            new_part = text  # Completely different — type fresh

        if new_part is None:
            def _replace():
                try:
                    if self._live_typed_chars > 0:
                        subprocess.run(
                            ["xdotool", "key", "--clearmodifiers"]
                            + ["BackSpace"] * self._live_typed_chars,
                            timeout=5, capture_output=True,
                        )
                    subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--delay",
                         "3", text],
                        timeout=10, capture_output=True,
                    )
                    self._live_typed_chars = len(text)
                    self._live_typed_text = text
                except Exception:
                    log.exception("live-write replace error")
            threading.Thread(target=_replace, daemon=True).start()
        elif new_part.strip():
            def _append():
                try:
                    subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--delay",
                         "3", new_part],
                        timeout=5, capture_output=True,
                    )
                    self._live_typed_chars += len(new_part)
                    self._live_typed_text = text
                except Exception:
                    log.exception("live-write append error")
            threading.Thread(target=_append, daemon=True).start()

    def clear_live_typing(self):
        """Forget live-write state (recording aborted)."""
        self._live_typed_chars = 0
        self._live_typed_text = ""

    # --------------------------------------------------- screenshot
    def screenshot(self, text=None):
        """Capture the screen and send it to the Claude terminal.

        Merge of _take_screenshot 3887-3933 and _type_then_screenshot
        3854-3885: capture first, keep the last 20, notify, then type
        "check the screen" — with the user's text appended when given.
        Returns a status message. Blocking — worker thread only."""
        try:
            from PIL import ImageGrab

            capture_dir = PATHS.SCREEN_DIR
            capture_dir.mkdir(parents=True, exist_ok=True)
            latest = capture_dir / "latest.png"
            ts = datetime.now().strftime("%H%M%S")
            timestamped = capture_dir / f"screen_{ts}.png"

            img = ImageGrab.grab()
            img.save(str(latest))
            img.save(str(timestamped))

            # Keep only last 20
            captures = sorted(capture_dir.glob("screen_*.png"))
            for old in captures[:-20]:
                old.unlink(missing_ok=True)

            log.info("Screenshot saved: %s", latest)
            try:
                subprocess.Popen(
                    ["notify-send", "-u", "normal", "-i", "camera-photo",
                     "-t", "3000", "Screenshot Captured",
                     f"Saved to {latest}\nSending to Claude..."],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass

            bus.publish(Status(f"Captured → {latest.name}", "ok"))

            # Prepend "check the screen" so Claude knows to look at it
            combined = f"check the screen, {text}" if text else "check the screen"
            self.type_text(combined)

            bus.publish(Status("Screenshot taken + sent to Claude", "ok"))
            return f"Screenshot saved to {latest}"
        except Exception as e:
            log.exception("Screenshot error")
            bus.publish(Status(f"Screenshot failed: {e}", "error"))
            return f"Screenshot failed: {e}"
