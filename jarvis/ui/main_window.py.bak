"""Jarvis V3 main window: assembly, tray, global hotkeys, bus subscriptions.

Layout (V3 spec, clean pass): header (wordmark … state pill + gear) /
reactor stage (avatar + engine card) / conversation / command bar / status
strip (wake word + CPU / GPU / MEMORY), with the SettingsDrawer sliding
over from the right. Every datum has exactly ONE site: app state = the
header StatePill (event-derived), machine telemetry = the status strip,
engine names = the reactor's engine card, RTT = the reply card head.

App-level wiring contract — class Services. The app constructs a Services
with its callables and passes it to MainWindow(root, services):

    start_recording()        begin a mic recording session
    stop_recording()         stop + transcribe the current session
    dispatch_text(text)      route a typed command (called off the Tk
                             thread; CommandBar ALSO publishes
                             UserUtterance(source='typed') — wire exactly
                             one consumer to avoid double dispatch)
    toggle_hotword(enabled)  start/stop the hotword listener
    quit()                   app teardown (bus AppQuit, thread joins);
                             MainWindow destroys the root afterwards
    calibrate_noise()        optional; run from the settings drawer
    enroll_speaker()         optional; run from the settings drawer
    train_wakeword()         optional; run from the settings drawer
    open_terminal()          pop out / raise Claude's tmux terminal,
                             attached READ-ONLY (terminal button,
                             settings drawer)
    alarm_action(alarm_id, action, minutes)
                             'dismiss' | 'snooze' from the alarm modal
    approval_answer(request_id, allowed)
                             ALLOW / DENY buttons on an approval card
    get_option(key) / set_option(key, value)
                             assistant.json dotted options for the
                             drawer's Assistant section

All callables are optional (no-ops with a warning when missing). Long
operations are invoked on daemon threads by the drawer; start/stop/quit are
called on the Tk thread and must return quickly.

Assistant events (jarvis.events) the window renders: ClaudeTaskState →
header pill WORKING / WAITING + terminal-button ring (which also draws
`open` while a terminal is attached to a jarvis-* session); ClaudeProgress
→ compact progress card; ActiveProject → status-bar PROJECT chip +
terminal tooltip; ApprovalRequested / ApprovalResolved → approval card;
AlarmFired / AlarmStopped → modal Card over the stage; BriefingReady → ONE briefing
card. Speech is never triggered here — the app subscribes too.

MainWindow attaches the event bus to the Tk root (bus.attach_tk) — the app
must NOT attach it a second time.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Callable, Optional

from jarvis import perf
from jarvis.config import CONFIG, MACHINE
from jarvis.events import (ActiveProject, AlarmFired, AlarmStopped, AppQuit,
                           ApprovalRequested, ApprovalResolved, AudioLevel,
                           UncertainResolved, UncertainUtterance,
                           BrainState, BriefingReady, ClaudeProgress,
                           ClaudeTaskState, HotwordDetected, JarvisReply,
                           MicState, ModelInfo, PartialText, RecordingStarted,
                           RecordingStopped, ReminderFired, SpeakingState,
                           Status, Transcribed, UserUtterance, bus)
from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.reactor import Reactor
from jarvis.ui.views import CommandBar, SettingsDrawer, StatusStrip, \
    TranscriptView
from jarvis.ui.widgets import (BarGradient, Card, RoundButton, StatePill,
                               Toast, px, set_scale, ui_display, ui_mono)

log = get_logger("ui.main_window")

# Design units at the 96-dpi baseline; scaled by S at runtime.
DEFAULT_W, DEFAULT_H = 520, 880
DEFAULT_GEOMETRY = f"{DEFAULT_W}x{DEFAULT_H}"
MIN_W, MIN_H = 460, 720

# StatePill words (uppercase, <= 10 chars, never ellipsized). OFFLINE is
# reserved — no event drives it today, so it is never shown. WORKING /
# WAITING come from ClaudeTaskState (a Claude task running / blocked on a
# permission question).
STATE_WORDS = {"idle": "READY", "listening": "LISTENING…",
               "thinking": "THINKING…", "speaking": "SPEAKING",
               "waiting": "WAITING", "working": "WORKING",
               "error": "ERROR"}
WARN_HOLD_S = 4.0            # warn Status: pill dot amber for this long
ERROR_HOLD_S = 6.0           # error Status: pill ERROR until ok/info or this
SNOOZE_MIN = 10              # the alarm modal's SNOOZE button
SESSION_PROBE_MS = 20000     # gap between `tmux ls` probes (2 retries)
SESSION_PROBE_RETRIES = 2
ATTACH_POLL_MS = 5000        # `tmux list-clients` while a session exists
ALARM_TITLES = {"alarm": "ALARM", "timer": "TIMER", "reminder": "REMINDER"}


def resolve_state(speaking: bool, listening: bool, thinking: bool,
                  error: bool, waiting: bool = False,
                  working: bool = False) -> str:
    """App-state precedence — the first four in the SAME order as
    Reactor.state() so the pill and the stage never disagree (the stage
    keeps idle / listen / think / speak): speaking > listening > thinking
    > waiting > working > error (held) > idle."""
    if speaking:
        return "speaking"
    if listening:
        return "listening"
    if thinking:
        return "thinking"
    if waiting:
        return "waiting"
    if working:
        return "working"
    if error:
        return "error"
    return "idle"


class ClaudeTaskTracker:
    """Tk-free bookkeeping behind the pill and the terminal button: the
    set of running task ids (working) and of those blocked on a
    permission question (waiting), from ClaudeTaskState sequences.
    queued tasks are tracked but count as neither; done / failed /
    cancelled clear a task."""

    RUNNING = ("running",)
    WAITING = ("waiting",)
    ENDED = ("done", "failed", "cancelled")

    def __init__(self):
        self._state: dict = {}        # task_id → state
        self._project: dict = {}      # task_id → project slug
        self.last_project = ""        # project of the latest live task

    def apply(self, task_id: str, state: str, project: str = ""):
        if not task_id:
            return
        if state in self.ENDED:
            self._state.pop(task_id, None)
            self._project.pop(task_id, None)
        elif state in ("queued",) + self.RUNNING + self.WAITING:
            self._state[task_id] = state
            if project:
                self._project[task_id] = project
                if state != "queued":
                    self.last_project = project
        # unknown states are ignored (the parser never raises; nor do we)

    @property
    def working(self) -> bool:
        return any(v == "running" for v in self._state.values())

    @property
    def waiting(self) -> bool:
        return any(v == "waiting" for v in self._state.values())

    @property
    def live(self) -> int:
        """Tasks that are running or waiting."""
        return sum(1 for v in self._state.values() if v != "queued")

    def terminal_state(self) -> str:
        """Terminal-button state: waiting beats working beats idle."""
        if self.waiting:
            return "waiting"
        if self.working:
            return "working"
        return "idle"


def alarm_modal_text(label: str, kind: str, due_text: str) -> tuple:
    """(title, time) for the alarm modal: the alarm's own label when it
    has one, else the kind as a chrome word (ALARM / TIMER / REMINDER);
    the due time uppercased like every HUD value ('7:00 AM')."""
    title = (label or "").strip() or ALARM_TITLES.get(
        (kind or "alarm").lower(), "ALARM")
    return title, (due_text or "").strip().upper()


def terminal_button_state(task_state: str, project: str,
                          session_seen: bool, attached: bool = False) -> str:
    """The terminal button's drawn state: the tracker's own state while a
    Claude task lives (working / waiting), otherwise `open` while a
    terminal is actually attached to a Jarvis session, `idle` when there
    is something to attach to (a project this session, or a live tmux
    session) and `no_project` when there is not."""
    if task_state in ("working", "waiting"):
        return task_state
    if attached:
        return "open"
    if (project or "").strip() or session_seen:
        return "idle"
    return "no_project"


def terminal_tooltip(project: str, attached: bool = False) -> str:
    project = (project or "").strip()
    if not project:
        return CommandBar.TIP_OPEN if attached else CommandBar.TIP_NO_PROJECT
    verb = "Claude's terminal is open" if attached else "Open Claude's terminal"
    return f"{verb} — {project}"


def terminal_available() -> bool:
    """The pop-out needs tmux and gnome-terminal on PATH."""
    return bool(shutil.which("tmux") and shutil.which("gnome-terminal"))


def session_exists(run=None) -> bool:
    """True when a `jarvis-<slug>` tmux session is alive — i.e. the
    terminal button has something to attach to right now. Probed a few
    seconds after boot and after each click (never polled: an idle app
    spawns nothing). The timeout is generous because the probe competes
    with the boot storm (avatar bake + torch/whisper preload) and a
    missed probe would leave the button faint until the next task.
    `run` is a seam for tests."""
    run = run or (lambda cmd: subprocess.run(cmd, capture_output=True,
                                             text=True, timeout=15))
    try:
        out = run(["tmux", "ls", "-F", "#{session_name}"])
    except (OSError, subprocess.SubprocessError):
        return False
    text = getattr(out, "stdout", "") or ""
    return any(line.strip().startswith("jarvis-")
               for line in text.splitlines())


def terminal_attached(run=None) -> bool:
    """True while a terminal is attached to a `jarvis-<slug>` session —
    the pop-out is on screen. One `tmux list-clients`, the same shape as
    `session_exists`; the session manager answers the same question per
    project through `terminal_open()`. No `-t` and no `-a`: bare
    list-clients lists every client on the server (tmux 3.4 has no `-a`
    flag here and rejects the command outright with it)."""
    run = run or (lambda cmd: subprocess.run(cmd, capture_output=True,
                                             text=True, timeout=15))
    try:
        out = run(["tmux", "list-clients", "-F", "#{session_name}"])
    except (OSError, subprocess.SubprocessError):
        return False
    if getattr(out, "returncode", 0) not in (0, None):
        return False
    text = getattr(out, "stdout", "") or ""
    return any(line.strip().startswith("jarvis-")
               for line in text.splitlines())


def fmt_mem_gb(total_kb: int, avail_kb: int) -> str:
    """Used system RAM for the status bar: '26.8 GB' ('--' when total is
    unknown)."""
    try:
        total, avail = int(total_kb), int(avail_kb)
    except (TypeError, ValueError):
        return "--"
    if total <= 0:
        return "--"
    return f"{max(0, total - avail) / 1048576.0:.1f} GB"


def fmt_asr(model_text: str) -> str:
    """Engine-card HEAR value: 'WHISPER SMALL' — the ModelInfo text
    ('small · GPU fp16') minus its backend/precision suffix."""
    model = (model_text or "").split("·")[0].strip()
    return f"WHISPER {model}".strip().upper()


def _xft_dpi() -> Optional[float]:
    """The desktop's effective DPI (Xft.dpi X resource — what GNOME apps
    actually render at). winfo_fpixels reports the server's physical DPI
    (162 on this box), which understates the 2x desktop (Xft.dpi=192)."""
    try:
        out = subprocess.run(["xrdb", "-query"], capture_output=True,
                             text=True, timeout=2)
        for line in out.stdout.splitlines():
            if line.startswith("Xft.dpi"):
                return float(line.split(":", 1)[1].strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        log.debug("Xft.dpi query failed", exc_info=True)
    return None


def detect_scale(root: tk.Misc) -> float:
    """Global UI scale factor S. Auto-detected from the desktop DPI
    (192dpi on this box → 2.0), quantized to quarter steps and clamped to
    [1.0, 3.0]; JARVIS_UI_SCALE overrides when set."""
    env = os.environ.get("JARVIS_UI_SCALE")
    if env:
        try:
            return max(0.5, min(3.0, float(env)))
        except ValueError:
            log.warning("bad JARVIS_UI_SCALE %r", env)
    dpi = _xft_dpi()
    if dpi is None:
        try:
            dpi = root.winfo_fpixels("1i")
        except tk.TclError:
            dpi = 96.0
    return max(1.0, min(3.0, round((dpi / 96.0) * 4) / 4))


def _noop(*_a, **_k):
    log.warning("service callable not wired")


# -------------------------------------------------------------- services
@dataclass
class Services:
    """Callables the app wires into the UI. See module docstring."""
    start_recording: Callable = _noop
    stop_recording: Callable = _noop
    dispatch_text: Optional[Callable] = None
    toggle_hotword: Callable = _noop
    quit: Optional[Callable] = None
    calibrate_noise: Optional[Callable] = None
    enroll_speaker: Optional[Callable] = None
    train_wakeword: Optional[Callable] = None
    # assistant (spec 2026-08-26 section 9.10) — all optional
    open_terminal: Callable = _noop
    alarm_action: Callable = _noop
    approval_answer: Callable = _noop
    uncertain_answer: Callable = _noop
    get_option: Callable = _noop
    set_option: Callable = _noop


# ------------------------------------------------------------------ tray
class TrayIcon:
    """System tray icon with recording state indicator.
    Ported from voice_input_gui.py 615-689 (callbacks instead of gui refs,
    theme colors instead of the old palette)."""

    def __init__(self, root, on_toggle_window, on_toggle_record, on_quit):
        self.root = root
        self.on_toggle_window = on_toggle_window
        self.on_toggle_record = on_toggle_record
        self.on_quit = on_quit
        self.icon = None
        self._available = False
        self._setup()

    def _setup(self):
        try:
            import pystray
            from PIL import Image, ImageDraw

            self._pystray = pystray
            self._Image = Image
            self._ImageDraw = ImageDraw
            self._available = True
            log.info("System tray: available")
        except Exception as e:
            log.info("System tray: not available (%s)", e)

    def start(self):
        if not self._available:
            return

        image = self._make_icon(theme.OK)
        menu = self._pystray.Menu(
            self._pystray.MenuItem("Show/Hide", self._toggle_window,
                                   default=True),
            self._pystray.MenuItem("Toggle Record", self._toggle_record),
            self._pystray.Menu.SEPARATOR,
            self._pystray.MenuItem("Quit", self._quit),
        )

        self.icon = self._pystray.Icon("jarvis", image, "Jarvis - Ready", menu)
        threading.Thread(target=self.icon.run, daemon=True,
                         name="tray-icon").start()
        log.info("Tray icon started")

    def update_state(self, recording):
        if not self._available or not self.icon:
            return
        color = theme.CYAN if recording else theme.OK
        title = "Jarvis - Recording..." if recording else "Jarvis - Ready"
        try:
            self.icon.icon = self._make_icon(color)
            self.icon.title = title
        except Exception:
            log.debug("tray update failed", exc_info=True)

    def stop(self):
        if self.icon:
            try:
                self.icon.stop()
            except Exception:
                log.debug("tray stop failed", exc_info=True)

    def _make_icon(self, color):
        # mic glyph — drawing ported verbatim from 673-680
        img = self._Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = self._ImageDraw.Draw(img)
        draw.rounded_rectangle([20, 8, 44, 38], radius=8, fill=color)
        draw.arc([14, 20, 50, 50], start=0, end=180, fill=color, width=3)
        draw.line([32, 50, 32, 58], fill=color, width=3)
        draw.line([22, 58, 42, 58], fill=color, width=3)
        return img

    def _toggle_window(self):
        self.root.after(0, self.on_toggle_window)

    def _toggle_record(self):
        self.root.after(0, self.on_toggle_record)

    def _quit(self):
        self.root.after(0, self.on_quit)


# --------------------------------------------------------- global hotkeys
class GlobalHotkey:
    """Register Ctrl+Shift+V (toggle) and Ctrl+Shift+R (push-to-talk)
    system-wide using the Xlib record extension.
    Ported from voice_input_gui.py 695-779 — same keys, same event loop;
    gui method calls replaced by injected callbacks."""

    def __init__(self, root, on_toggle, on_ptt_start, on_ptt_stop,
                 is_recording: Callable[[], bool]):
        self.root = root
        self.on_toggle = on_toggle
        self.on_ptt_start = on_ptt_start
        self.on_ptt_stop = on_ptt_stop
        self.is_recording = is_recording
        self._thread = None
        self._available = False

    def start(self):
        try:
            from Xlib import X, XK, display          # noqa: F401
            from Xlib.ext import record              # noqa: F401
            self._available = True
        except ImportError:
            log.info("Global hotkey: Xlib not available")
            return

        self._thread = threading.Thread(target=self._listener, daemon=True,
                                        name="global-hotkey")
        self._thread.start()
        log.info("Global hotkey listener started (Ctrl+Shift+V)")

    def _listener(self):
        from Xlib import X, XK, display
        from Xlib.ext import record
        from Xlib.protocol import rq

        local_dpy = display.Display()
        record_dpy = display.Display()
        ctrl_held = False
        shift_held = False

        def callback(reply):
            nonlocal ctrl_held, shift_held
            if reply.category != record.FromServer or reply.client_swapped:
                return
            data = reply.data
            while len(data):
                event, data = rq.EventField(None).parse_binary_value(
                    data, record_dpy.display, None, None
                )
                keycode = event.detail
                keysym = local_dpy.keycode_to_keysym(keycode, 0)

                if event.type == X.KeyPress:
                    if keysym in (XK.XK_Control_L, XK.XK_Control_R):
                        ctrl_held = True
                    elif keysym in (XK.XK_Shift_L, XK.XK_Shift_R):
                        shift_held = True
                    elif keysym == XK.XK_v and ctrl_held and shift_held:
                        log.info("Global hotkey: Ctrl+Shift+V")
                        self.root.after(0, self.on_toggle)
                    elif keysym == XK.XK_r and ctrl_held and shift_held:
                        # Push-to-talk: start recording on key press
                        if not self.is_recording():
                            log.info("Push-to-talk: key down")
                            self.root.after(0, self.on_ptt_start)
                elif event.type == X.KeyRelease:
                    if keysym in (XK.XK_Control_L, XK.XK_Control_R):
                        ctrl_held = False
                    elif keysym in (XK.XK_Shift_L, XK.XK_Shift_R):
                        shift_held = False
                    elif keysym == XK.XK_r and self.is_recording():
                        # Push-to-talk: stop on key release
                        log.info("Push-to-talk: key up")
                        self.root.after(0, self.on_ptt_stop)

        ctx = record_dpy.record_create_context(
            0, [record.AllClients],
            [{"core_requests": (0, 0), "core_replies": (0, 0),
              "ext_requests": (0, 0, 0, 0), "ext_replies": (0, 0, 0, 0),
              "delivered_events": (0, 0),
              "device_events": (X.KeyPress, X.KeyRelease),
              "errors": (0, 0), "client_started": False,
              "client_died": False}],
        )
        try:
            record_dpy.record_enable_context(ctx, callback)
        except Exception as e:
            log.warning("Global hotkey error: %s", e)
        finally:
            try:
                record_dpy.record_free_context(ctx)
            except Exception:
                log.debug("record_free_context failed", exc_info=True)


# ------------------------------------------------------------ main window
class MainWindow:
    """Assembles the whole V3 UI on a caller-provided tk.Tk root."""

    def __init__(self, root: tk.Tk, services: Optional[Services] = None):
        self.root = root
        self.services = services or Services()
        self._recording = False
        self._mic_available = MACHINE.has_mic
        self._last_confidence: Optional[float] = None
        self._temps_text = ""     # written by worker thread, read by Tk loop
        self._mem_text = ""       # used RAM ('26.8 GB'), same worker
        self._closing = False
        # app-state machine inputs (bus events) → the header StatePill
        self._thinking = False
        self._speaking = False
        self._error_until = 0.0
        self._warn_until = 0.0
        # assistant state: Claude tasks (pill + terminal ring), the active
        # project (status chip + tooltip), the ringing alarm's modal
        self._tasks = ClaudeTaskTracker()
        self._project = ""
        self._alarm = None               # (alarm_id, Card) while ringing
        self._term_available = terminal_available()
        self._session_seen = False       # a jarvis-* tmux session is alive
        self._term_attached = False      # …and a terminal is watching it
        self._attach_polling = False     # one list-clients chain, not N

        # Real telemetry for the reactor's engine card (plain attributes —
        # the reactor polls them at 1Hz through _telemetry()).
        self._asr_text = fmt_asr(CONFIG.model)          # refined by ModelInfo
        self._tts_text = self._tts_desc(CONFIG.tts_engine)
        try:
            from jarvis.brain import OLLAMA_MODEL       # lightweight module
            self._llm_name = OLLAMA_MODEL
        except Exception:
            self._llm_name = "?"
        self._llm_text = self._llm_name                 # re-read by _probe_llm
        self._dev_text: Optional[str] = None            # None until probed
        self._utter_ts: Optional[float] = None          # UserUtterance → reply

        # HiDPI: the V3 design couples a pixel layout (56px header, 44px
        # mic, 8px grid) to the type scale 26/15/13/11 at the 96-dpi
        # baseline. One global scale factor S (detect_scale) is
        # established ONCE here, before any widget is built: theme spacing
        # tokens mutate via apply_scale, every hardcoded dimension routes
        # through widgets.px, and fonts scale ONLY via set_font_scale
        # (wired inside set_scale — never double-scaled).
        self.scale = detect_scale(root)
        theme.apply_scale(self.scale)
        set_scale(self.scale)
        log.info("ui scale S=%.2f (dpi=%.0f)", self.scale,
                 root.winfo_fpixels("1i"))
        self._min_w, self._min_h = px(MIN_W), px(MIN_H)

        theme.resolve_fonts(root)
        root.title("Jarvis")
        # Borderless: splash-type windows are undecorated but still
        # WM-managed (focus, stacking, tray restore). Must be set pre-map.
        try:
            root.attributes("-type", "splash")
        except tk.TclError:
            log.warning("could not set splash window type")
        # Projected-panel frame: the root ground is a dim cyan step and
        # every section lives in a shell padded 1px inside it, so the
        # whole app reads as one hologram pane with a hairline outline.
        root.configure(bg=theme.FRAME)
        root.minsize(self._min_w, self._min_h)
        root.geometry(self._pick_geometry())
        self.shell = tk.Frame(root, bg=theme.BG)
        self.shell.pack(fill="both", expand=True, padx=1, pady=1)

        self.toast = Toast(root)
        self._build_header()
        # Footer packs before the expanding stage so the bottom bars always
        # keep their space; the transcript absorbs any height shortfall.
        self._build_footer()
        self._build_stage()

        self.drawer = SettingsDrawer(
            root, services=self.services,
            on_config_change=self._on_config_change, toast=self.toast)

        self.tray = TrayIcon(root, self._toggle_visibility,
                             self._toggle_recording, self._on_close)
        self.tray.start()
        self.hotkeys = GlobalHotkey(
            root,
            on_toggle=self._toggle_recording,
            on_ptt_start=self._start_recording,
            on_ptt_stop=self._stop_recording,
            is_recording=lambda: self._recording)
        self.hotkeys.start()

        self._bind_keys()
        self._subscribe()
        bus.attach_tk(root)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        if not self._mic_available:
            self.command_bar.set_mic_state("disabled")
            self.status_strip.set_hotword(False)
            self.set_status("No microphone detected — type commands below",
                            "warn")
        else:
            self.set_status("Ready", "ok")
        self._refresh_placeholder()
        self._refresh_pill()
        self._refresh_terminal()

        threading.Thread(target=self._temps_worker, daemon=True,
                         name="temps-worker").start()
        self._temps_tick()
        # after the boot storm, not in it — a `tmux ls` racing the avatar
        # bake and the model preload loses, and a lost probe is sticky
        root.after(4000, self._probe_session)

        # Borderless: strip WM decorations once mapped (Motif hints keep the
        # window managed — alt-tab, taskbar and tray restore still work,
        # unlike overrideredirect). Header drags, grip resizes.
        self._drag_off = None
        self._geom_ts = 0.0
        self._build_grip()
        self._build_frame_brackets()
        log.info("borderless window ready")

    # ----------------------------------------------------------- geometry
    def _default_geometry(self) -> str:
        """520x880 design → S-scaled, height clamped to 90% of screen."""
        w, h = px(DEFAULT_W), px(DEFAULT_H)
        try:
            h = min(h, int(self.root.winfo_screenheight() * 0.9))
        except tk.TclError:
            pass
        return f"{w}x{h}"

    def _pick_geometry(self) -> str:
        """Saved geometry, unless absent/invalid/stale. A saved size below
        the scaled minimum comes from a pre-HiDPI run — replace it with
        the scaled default (keeping the saved position)."""
        saved = CONFIG.window_geometry or ""
        m = re.fullmatch(r"(\d+)x(\d+)([+-]\d+[+-]\d+)?", saved)
        if not m:
            if saved:
                log.warning("bad saved geometry %r; using default", saved)
            return self._default_geometry()
        w, h, pos = int(m.group(1)), int(m.group(2)), m.group(3) or ""
        if w < self._min_w or h < self._min_h:
            log.info("saved geometry %r below scaled minimum; rescaling",
                     saved)
            return self._default_geometry() + pos
        return saved

    # ---------------------------------------------- borderless window mgmt
    def _move_start(self, event):
        self._drag_off = (event.x_root - self.root.winfo_x(),
                          event.y_root - self.root.winfo_y())

    def _move_drag(self, event):
        if self._drag_off is None:
            return
        now = time.monotonic()
        if now - self._geom_ts < 0.016:      # ~60Hz cap
            return
        self._geom_ts = now
        dx, dy = self._drag_off
        self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _build_grip(self):
        g = px(18)
        grip = tk.Canvas(self.root, width=g, height=g, bg=theme.SURFACE,
                         highlightthickness=0, bd=0,
                         cursor="bottom_right_corner")
        for k in range(3):
            off = px(4) + k * px(5)
            grip.create_line(off, g, g, off, fill=theme.FAINT,
                             width=max(1, px(1)))
        # SE window-frame bracket lives here (the grip owns this corner)
        lw = max(1, px(1))
        e = g - 1 - lw // 2
        grip.create_line(e - px(14), e, e, e, e, e - px(14),
                         fill=theme.EDGE, width=lw)
        grip.place(relx=1.0, rely=1.0, anchor="se")
        grip.bind("<B1-Motion>", self._grip_drag)
        tk.Misc.lift(grip)   # Canvas.lift is tag_raise; use the widget form
        self._grip = grip

    def _build_frame_brackets(self):
        """L-shaped corner brackets at the four WINDOW corners (top pair
        on the header ground, bottom-left on the status strip, bottom-
        right inside the resize grip) — with the 1px shell outline they
        make the whole app read as one projected panel."""
        arm = px(14)
        lw = max(1, px(1))
        o = lw // 2
        e = arm - 1 - o
        for corner, parent in (("nw", self._header), ("ne", self._header),
                               ("sw", self.status_strip)):
            c = tk.Canvas(parent, width=arm, height=arm,
                          bg=parent.cget("bg"), highlightthickness=0, bd=0)
            pts = {"nw": (o, arm, o, o, arm, o),
                   "ne": (0, o, e, o, e, arm),
                   "sw": (o, 0, o, e, arm, e)}[corner]
            c.create_line(*pts, fill=theme.EDGE, width=lw)
            place = {"nw": dict(x=0, y=0, anchor="nw"),
                     "ne": dict(relx=1.0, y=0, anchor="ne"),
                     "sw": dict(x=0, rely=1.0, anchor="sw")}[corner]
            c.place(**place)
            if corner in ("nw", "ne"):     # header corners stay draggable
                c.bind("<ButtonPress-1>", self._move_start, add=True)
                c.bind("<B1-Motion>", self._move_drag, add=True)

    def _grip_drag(self, event):
        now = time.monotonic()
        if now - self._geom_ts < 0.033:      # ~30Hz cap for resizes
            return
        self._geom_ts = now
        w = max(self._min_w, event.x_root - self.root.winfo_rootx())
        h = max(self._min_h, event.y_root - self.root.winfo_rooty())
        self.root.geometry(f"{w}x{h}")

    # ------------------------------------------------------------ layout
    def _build_header(self):
        header = tk.Frame(self.shell, bg=theme.BG, height=px(56))
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        self._header = header

        # Display-face wordmark; Tk has no letter tracking, so the wide
        # spacing is baked into the string. The most focal text in the app
        # is WHITE per the film budget (cyan is structure, never the
        # star), with a 1px dim-cyan hologram-fringe ghost offset behind.
        wm_font = ui_display(theme.SIZE_WORDMARK, "semibold")
        wm_text = "J A R V I S"
        wordmark = tk.Canvas(header, width=px(160), height=px(40),
                             bg=theme.BG, highlightthickness=0, bd=0)
        gh = max(1, px(1))
        wordmark.create_text(px(2) + gh, px(20) + gh, text=wm_text,
                             font=wm_font, fill=theme.RAMP40, anchor="w")
        wm_main = wordmark.create_text(px(2), px(20), text=wm_text,
                                       font=wm_font, fill=theme.FOCAL,
                                       anchor="w")
        bb = wordmark.bbox(wm_main)
        if bb:                        # fit exactly — don't starve the status
            wordmark.configure(width=bb[2] + gh + 1)
        wordmark.pack(side="left", padx=(theme.PAD, 0))

        self._close_btn = RoundButton(header, text="✕", kind="ghost",
                                      size=theme.SIZE_LABEL, pad_x=9, pad_y=5,
                                      bg=theme.BG, command=self._on_close)
        self._close_btn.pack(side="right", padx=(0, theme.PAD_S))
        self._min_btn = RoundButton(header, text="—", kind="ghost",
                                    size=theme.SIZE_LABEL, pad_x=9, pad_y=5,
                                    bg=theme.BG,
                                    command=self._minimize_to_tray)
        self._min_btn.pack(side="right")
        self._gear = RoundButton(header, text="⚙", kind="ghost",
                                 size=theme.SIZE_BODY, pad_x=9, pad_y=5,
                                 bg=theme.BG, command=self.drawer_toggle)
        self._gear.pack(side="right", padx=(0, theme.PAD_S))

        # The single app-state site: dot + word in one RAISED slab, PAD_S
        # left of the gear. Between the wordmark and the pill: nothing.
        self.pill = StatePill(header, bg=theme.BG)
        self.pill.pack(side="right", padx=(0, theme.PAD_S))

        # atmosphere: the header ground is a soft gradient (sheen behind
        # the wordmark, settling flat to the right) — flat-bg children are
        # re-tinted so nothing punches a hole in it; slightly stronger for
        # the luminous pass (the wordmark keeps its white treatment)
        self._header_grad = BarGradient(
            header, theme.BG, [(0.0, 0.022), (0.28, 0.075), (1.0, 0.015)])

        # Hairline rule under the header with three brighter dash segments
        # (classic HUD detail); redrawn only on width change.
        rule = tk.Canvas(self.shell, height=px(3), bg=theme.BG,
                         highlightthickness=0, bd=0)
        rule.pack(fill="x", side="top")
        self._rule = rule
        self._rule_w = None
        rule.bind("<Configure>", self._draw_header_rule, add=True)

        # Borderless window: the ENTIRE header strip is the drag handle —
        # the frame plus every non-button child (wordmark, state pill).
        # Only the buttons keep their own clicks.
        for handle in (header, wordmark, self.pill, rule):
            handle.bind("<ButtonPress-1>", self._move_start, add=True)
            handle.bind("<B1-Motion>", self._move_drag, add=True)

    def _draw_header_rule(self, event):
        if self._rule_w == event.width:
            return
        self._rule_w = event.width
        rule = self._rule
        rule.delete("all")
        y = px(1)
        rule.create_line(0, y, event.width, y, fill=theme.HOLO_DIM)
        x = theme.PAD
        for seg in (px(28), px(12), px(5)):
            rule.create_line(x, y, x + seg, y, fill=theme.CYAN_DIM,
                             width=px(2))
            x += seg + px(7)

    def _build_stage(self):
        self.reactor = Reactor(self.shell, height=px(300))
        self.reactor.pack(fill="x", side="top")
        self.reactor.attach_toplevel()
        self.reactor.set_telemetry(self._telemetry)

        # no pady: the reactor stage's seam band steps its ground directly
        # into the transcript's lit top row (the glow bleeds down — a BG
        # gap here would put a dark rule back between the panels)
        self.transcript = TranscriptView(self.shell, toast=self.toast)
        self.transcript.pack(fill="both", expand=True, side="top")

    def _build_footer(self):
        self.status_strip = StatusStrip(
            self.shell, on_hotword_click=self._toggle_hotword)
        self.status_strip.pack(fill="x", side="bottom")

        self.command_bar = CommandBar(self.shell,
                                      on_submit=self._on_typed,
                                      on_mic=self._toggle_recording,
                                      on_terminal=self._open_terminal,
                                      terminal_available=self._term_available)
        self.command_bar.pack(fill="x", side="bottom")
        if not self._term_available:
            log.warning("terminal button disabled: tmux or gnome-terminal "
                        "missing")

    # -------------------------------------------------------- keybindings
    def _bind_keys(self):
        # F5 binding MUST stay: hotword_daemon sends a synthetic F5 keypress
        self.root.bind("<F5>", lambda e: self._toggle_recording())
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", lambda e: self._minimize_to_tray())

    def _on_space(self, event):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Text, tk.Entry)):
            return
        self._toggle_recording()

    # ------------------------------------------------------------ actions
    def drawer_toggle(self):
        self.drawer.toggle()

    def _toggle_recording(self):
        if not self._mic_available:
            self.set_status("No microphone detected", "warn")
            return
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        try:
            self.services.start_recording()
        except Exception:
            log.exception("start_recording failed")
            self.set_status("Recording failed to start", "error")

    def _stop_recording(self):
        try:
            self.services.stop_recording()
        except Exception:
            log.exception("stop_recording failed")

    def _on_typed(self, text: str):
        # CommandBar already published UserUtterance(source='typed').
        if self.services.dispatch_text:
            threading.Thread(target=self._dispatch, args=(text,),
                             daemon=True).start()

    def _dispatch(self, text):
        try:
            self.services.dispatch_text(text)
        except Exception:
            log.exception("dispatch_text failed")
            bus.publish(Status(text="Command failed — see log", kind="error"))

    def _open_terminal(self):
        """Terminal button / drawer: services.open_terminal() spawns a
        READ-ONLY `tmux attach -r` in gnome-terminal, or raises the
        existing window — off the Tk thread."""
        fn = self.services.open_terminal

        def run():
            try:
                ok = fn()
                if ok is False:
                    bus.publish(Status(text="No Claude terminal to open",
                                       kind="warn"))
                else:
                    self._session_seen = True
                    self._after(0, self._refresh_terminal)
                    # the window takes a moment to attach; look then
                    self._after(1500, self._start_attach_poll)
            except Exception:
                log.exception("open_terminal failed")
                bus.publish(Status(text="Could not open the terminal",
                                   kind="error"))
        threading.Thread(target=run, daemon=True, name="open-terminal").start()

    def _toggle_hotword(self):
        enabled = not CONFIG.hotword
        CONFIG.update(hotword=enabled)
        self.status_strip.set_hotword(enabled)
        self._refresh_placeholder()
        try:
            self.services.toggle_hotword(enabled)
        except Exception:
            log.exception("toggle_hotword failed")

    def _toggle_visibility(self):
        # ported from voice_input_gui.py 4865-4870
        if self.root.state() == "withdrawn":
            self.root.deiconify()
            self.root.lift()
        else:
            self.root.withdraw()

    def _minimize_to_tray(self):
        # ported from voice_input_gui.py 4872-4876
        if self.tray._available:
            self.root.withdraw()
        else:
            self.root.iconify()

    def _on_close(self):
        if self._closing:
            return
        self._closing = True
        try:
            CONFIG.update(window_geometry=self.root.geometry())
        except Exception:
            log.exception("geometry save failed")
        self.tray.stop()
        bus.publish(AppQuit())
        if self.services.quit:
            try:
                self.services.quit()
            except Exception:
                log.exception("services.quit failed")
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------- status
    def set_status(self, text: str, kind: str = "info"):
        """Status-event routing. ok/info/busy → NO text anywhere (the
        state machine already covers Ready / Listening… / Thinking…; the
        text is logged). warn → toast 4 s + pill dot WARN for 4 s, word
        unchanged. error → pill ERROR held until the next ok/info Status
        or 6 s, whichever first, plus a 4 s toast."""
        log.info("status[%s]: %s", kind, text)
        now = time.monotonic()
        if kind == "warn":
            self.toast.show(text, kind="warn", ms=4000)
            self._warn_until = now + WARN_HOLD_S
            self._refresh_pill()
            self.root.after(int(WARN_HOLD_S * 1000) + 50, self._refresh_pill)
        elif kind == "error":
            self.toast.show(text, kind="error", ms=4000)
            self._error_until = now + ERROR_HOLD_S
            self._refresh_pill()
            self.root.after(int(ERROR_HOLD_S * 1000) + 50, self._refresh_pill)
        elif kind in ("ok", "info") and self._error_until:
            self._error_until = 0.0
            self._refresh_pill()

    def _app_state(self) -> str:
        now = time.monotonic()
        return resolve_state(self._speaking, self._recording, self._thinking,
                             now < self._error_until,
                             waiting=self._tasks.waiting,
                             working=self._tasks.working)

    def _refresh_pill(self):
        """Recompute the StatePill from the event-derived state machine
        (called on every state-changing event). Idle / WORKING / WAITING:
        FOCAL word, the dot carries the colour; other states take
        STATE_COLORS for both; a live warn hold turns only the dot amber."""
        state = self._app_state()
        color = theme.STATE_COLORS.get(state, theme.CYAN_DIM)
        word_color = theme.FOCAL if state in theme.FOCAL_WORD_STATES else color
        dot = theme.WARN if time.monotonic() < self._warn_until else color
        self.pill.set_state(STATE_WORDS[state], dot, word_color)

    def _refresh_terminal(self):
        """Terminal button ring from the task tracker; tooltip names the
        active project (else the latest project with a live task). With
        no project and no live tmux session the button is drawn faint —
        there is nothing to attach to yet — but stays clickable, so a
        click still says so."""
        project = self._project or self._tasks.last_project
        state = terminal_button_state(self._tasks.terminal_state(), project,
                                      self._session_seen, self._term_attached)
        self.command_bar.set_terminal_state(
            state, terminal_tooltip(project, self._term_attached))

    def _probe_session(self, retries: int = SESSION_PROBE_RETRIES):
        """One `tmux ls` off the Tk thread → the faint / normal terminal
        button. Never polled; retried a couple of times while the answer
        is 'none', because a session started by an earlier run is exactly
        the case the button is faint for."""
        if not self._term_available or self._session_seen or self._closing:
            return

        def run():
            alive = session_exists()
            log.info("tmux session probe: %s",
                     "attached" if alive else "no jarvis-* session")
            if self._closing:
                return
            if alive:
                self._session_seen = True
                self._after(0, self._refresh_terminal)
                self._after(0, self._start_attach_poll)
            elif retries > 0:
                self._after(SESSION_PROBE_MS,
                            lambda: self._probe_session(retries - 1))
        threading.Thread(target=run, daemon=True, name="tmux-probe").start()

    def _start_attach_poll(self):
        """Begin the attach poll once, however often it is asked for."""
        if not self._attach_polling:
            self._probe_attached()

    def _probe_attached(self):
        """Is a terminal watching Claude right now? One `tmux
        list-clients` off the Tk thread, repeated every 5 s — but ONLY
        while a jarvis-* session exists, so an app that has never run a
        Claude task still spawns nothing."""
        if self._closing or not self._term_available or not self._session_seen:
            self._attach_polling = False
            return
        self._attach_polling = True

        def run():
            attached = terminal_attached()
            if self._closing:
                return
            if attached != self._term_attached:
                self._term_attached = attached
                log.info("claude terminal %s",
                         "attached" if attached else "closed")
                self._after(0, self._refresh_terminal)
            self._after(ATTACH_POLL_MS, self._probe_attached)
        threading.Thread(target=run, daemon=True, name="tmux-clients").start()

    def _after(self, ms: int, fn):
        """root.after that tolerates a window on its way out."""
        try:
            self.root.after(ms, fn)
        except tk.TclError:
            log.debug("after() on a dead window", exc_info=True)

    def _refresh_placeholder(self):
        """The command-bar placeholder is the ONE idle hint; it mentions
        the wake word only when the listener is on and a mic exists."""
        self.command_bar.set_placeholder(
            bool(CONFIG.hotword) and self._mic_available)

    # ---------------------------------------------------- reactor telemetry
    @staticmethod
    def _tts_desc(engine: str) -> str:
        """Engine-card SPEAK value: 'XTTS' or 'EDGE · RYAN' (the Edge
        en-GB-Ryan neural voice) — no device/locale noise."""
        return "XTTS" if engine == "xtts" else "EDGE · RYAN"

    def _telemetry(self) -> dict:
        """Provider for Reactor.set_telemetry — cheap attribute reads on
        the Tk thread; every value is real (events, config, ollama,
        nvidia-smi). Exactly the four engine-card rows."""
        return {"asr": self._asr_text, "tts": self._tts_text,
                "llm": self._llm_text, "dev": self._dev_text}

    # ------------------------------------------------------ subscriptions
    def _subscribe(self):
        bus.subscribe(Status, self._ev_status)
        bus.subscribe(ModelInfo, self._ev_model)
        bus.subscribe(PartialText, self._ev_partial)
        bus.subscribe(Transcribed, self._ev_transcribed)
        bus.subscribe(UserUtterance, self._ev_user)
        bus.subscribe(JarvisReply, self._ev_reply)
        bus.subscribe(BrainState, self._ev_brain)
        bus.subscribe(SpeakingState, self._ev_speaking)
        bus.subscribe(MicState, self._ev_mic)
        bus.subscribe(HotwordDetected, self._ev_hotword)
        bus.subscribe(ReminderFired, self._ev_reminder)
        bus.subscribe(RecordingStarted, self._ev_rec_start)
        bus.subscribe(RecordingStopped, self._ev_rec_stop)
        bus.subscribe(AudioLevel, self._ev_audio)
        # assistant events (spec 2026-08-26, section 9)
        bus.subscribe(ClaudeTaskState, self._ev_claude_task)
        bus.subscribe(ClaudeProgress, self._ev_claude_progress)
        bus.subscribe(ActiveProject, self._ev_active_project)
        bus.subscribe(ApprovalRequested, self._ev_approval)
        bus.subscribe(ApprovalResolved, self._ev_approval_done)
        bus.subscribe(UncertainUtterance, self._ev_uncertain)
        bus.subscribe(UncertainResolved, self._ev_uncertain_done)
        bus.subscribe(AlarmFired, self._ev_alarm)
        bus.subscribe(AlarmStopped, self._ev_alarm_stopped)
        bus.subscribe(BriefingReady, self._ev_briefing)

    def _ev_status(self, ev: Status):
        self.set_status(ev.text, ev.kind)

    def _ev_model(self, ev: ModelInfo):
        self._asr_text = fmt_asr(ev.text)         # 'small · GPU fp16' → SMALL

    def _ev_partial(self, ev: PartialText):
        self.transcript.show_partial(ev.text)

    def _ev_transcribed(self, ev: Transcribed):
        self._last_confidence = ev.confidence
        if not ev.accepted:
            reason = ev.reject_reason or "low confidence"
            self.set_status(f"Rejected ({reason})", "warn")
            self.transcript.clear_partial()

    def _ev_user(self, ev: UserUtterance):
        conf = self._last_confidence if ev.source == "voice" else None
        self._last_confidence = None
        self._utter_ts = time.monotonic()         # RTT measurement start
        self.transcript.add_user(ev.text, confidence=conf)

    def _ev_reply(self, ev: JarvisReply):
        if ev.text:
            rtt = None
            if self._utter_ts is not None:
                dt = time.monotonic() - self._utter_ts
                if 0.0 < dt < 600.0:
                    rtt = dt
                self._utter_ts = None
            self.transcript.add_jarvis(ev.text, rtt=rtt)

    def _ev_brain(self, ev: BrainState):
        self._thinking = (ev.state == "thinking")
        self._refresh_pill()
        self.set_status("Thinking…" if self._thinking else "Ready",
                        "busy" if self._thinking else "ok")

    def _ev_speaking(self, ev: SpeakingState):
        if ev.active != self._speaking:
            self._speaking = ev.active
            self._refresh_pill()

    def _ev_mic(self, ev: MicState):
        self._mic_available = ev.available
        self._refresh_placeholder()
        if ev.available:
            self.command_bar.set_mic_state(
                "recording" if self._recording else "idle")
            if ev.device_name:
                self.set_status(f"Mic: {ev.device_name}", "info")
        else:
            self.command_bar.set_mic_state("disabled")
            self.set_status("No microphone detected — type commands below",
                            "warn")

    def _ev_hotword(self, ev: HotwordDetected):
        self.set_status(f"Wake word ({ev.score:.2f})", "ok")

    def _ev_reminder(self, ev: ReminderFired):
        self.toast.show(f"Reminder: {ev.text}", kind="warn", ms=6000)
        self.transcript.add_jarvis(f"Reminder: {ev.text}")

    def _ev_rec_start(self, _ev: RecordingStarted):
        self._recording = True
        self.command_bar.set_mic_state("recording")
        self.tray.update_state(True)
        self._refresh_pill()
        self.set_status("Listening…", "busy")

    def _ev_rec_stop(self, ev: RecordingStopped):
        self._recording = False
        self.command_bar.set_mic_state(
            "idle" if self._mic_available else "disabled")
        self.tray.update_state(False)
        self._refresh_pill()
        self.set_status(f"Stopped ({ev.reason})", "info")

    def _ev_audio(self, _ev: AudioLevel):
        pass  # the reactor consumes AudioLevel directly

    # ------------------------------------------------- assistant events
    def _ev_claude_task(self, ev: ClaudeTaskState):
        self._tasks.apply(ev.task_id, ev.state, ev.project)
        # a live task means a jarvis-<slug> session exists: the button has
        # something to attach to, and the attach poll has a reason to run
        if ev.state in ClaudeTaskTracker.RUNNING + ClaudeTaskTracker.WAITING:
            self._session_seen = True
            self._start_attach_poll()
        self._refresh_pill()
        self._refresh_terminal()

    def _ev_claude_progress(self, ev: ClaudeProgress):
        self.transcript.add_progress(ev.line)

    def _ev_active_project(self, ev: ActiveProject):
        self._project = (ev.slug or "").strip()
        self.status_strip.set_project(self._project)
        self._refresh_terminal()

    def _ev_approval(self, ev: ApprovalRequested):
        # the question is what this turn produced: consume the utterance
        # stamp so a later spontaneous reply cannot wear its round trip
        self._utter_ts = None
        self.transcript.add_approval(ev.request_id, ev.question,
                                     self._answer_approval)

    def _ev_approval_done(self, ev: ApprovalResolved):
        self.transcript.resolve_approval(ev.request_id, ev.allowed)

    def _ev_uncertain(self, ev: UncertainUtterance):
        """"Was that for me?" as a card that WAITS. The old behaviour was a
        4 s toast plus a text-free info Status, so the question vanished
        before it could be read, and nothing could answer it."""
        self._utter_ts = None
        self.transcript.add_approval(ev.request_id, ev.question,
                                     self._answer_uncertain,
                                     yes_text="YES", no_text="NO")

    def _ev_uncertain_done(self, ev: UncertainResolved):
        # also fires when the spoken reply answered it, so the card stops
        # inviting a click that would arrive too late
        self.transcript.resolve_approval(ev.request_id, ev.yes,
                                         yes_mark="yes", no_mark="no")

    def _answer_uncertain(self, request_id: str, yes: bool):
        fn = self.services.uncertain_answer

        def run():
            try:
                fn(request_id, yes)
            except Exception:
                log.exception("uncertain_answer failed")

        threading.Thread(target=run, daemon=True,
                         name="uncertain-answer").start()

    def _answer_approval(self, request_id: str, allowed: bool):
        fn = self.services.approval_answer

        def run():
            try:
                fn(request_id, allowed)
            except Exception:
                log.exception("approval_answer failed")
        threading.Thread(target=run, daemon=True, name="approval").start()

    def _ev_briefing(self, ev: BriefingReady):
        # the briefing card IS the answer to this turn (the app publishes
        # no JarvisReply for it), so it consumes the utterance stamp too
        self._utter_ts = None
        self.transcript.add_briefing(ev.sections)

    # ------------------------------------------------------ alarm modal
    def _ev_alarm(self, ev: AlarmFired):
        """Overlay Card centred on the stage (300 wide): label (display
        SIZE_LABEL semibold), time (mono SIZE_BODY), DISMISS / SNOOZE 10.
        The reactor keeps animating beneath; the window is brought back
        from the tray so the alarm is seen."""
        self._hide_alarm()
        title, when = alarm_modal_text(ev.label, ev.kind, ev.due_text)
        card = Card(self.reactor, fill=theme.RAISED, pad=12, bg=theme.BG)
        body = card.body
        tk.Label(body, text=title, font=ui_display(theme.SIZE_LABEL, "semibold"),
                 fg=theme.INK, bg=theme.RAISED, anchor="w", justify="left",
                 wraplength=px(300 - 2 * 12 - 4)).pack(fill="x")
        if when:
            tk.Label(body, text=when, font=ui_mono(theme.SIZE_BODY),
                     fg=theme.FOCAL, bg=theme.RAISED,
                     anchor="w").pack(fill="x", pady=(px(4), 0))
        row = tk.Frame(body, bg=theme.RAISED)
        row.pack(fill="x", pady=(px(12), 0))
        alarm_id = ev.alarm_id
        buttons = (
            RoundButton(row, text="DISMISS", kind="accent", bg=theme.RAISED,
                        command=lambda: self._alarm_action(alarm_id, "dismiss")),
            RoundButton(row, text=f"SNOOZE {SNOOZE_MIN}", kind="default",
                        bg=theme.RAISED,
                        command=lambda: self._alarm_action(alarm_id, "snooze")))
        buttons[0].pack(side="left")
        buttons[1].pack(side="left", padx=(theme.PAD_S, 0))
        card.set_edge_glow()
        card.place(in_=self.reactor, relx=0.5, rely=0.5, anchor="center",
                   width=px(300))
        tk.Misc.lift(card)
        self._alarm = (alarm_id, card, buttons)
        try:
            if self.root.state() == "withdrawn":
                self.root.deiconify()
            self.root.lift()
        except tk.TclError:
            log.debug("alarm: could not raise the window", exc_info=True)

    def _alarm_action(self, alarm_id: str, action: str):
        """DISMISS / SNOOZE → services.alarm_action(alarm_id, action, 10)
        off the Tk thread. The modal stays until AlarmStopped arrives
        (the timekeeper publishes it once the ringer actually stops); the
        buttons grey out so a click is acknowledged."""
        if self._alarm and self._alarm[0] == alarm_id:
            for btn in self._alarm[2]:
                btn.set_enabled(False)
        fn = self.services.alarm_action

        def run():
            try:
                fn(alarm_id, action, SNOOZE_MIN)
            except Exception:
                log.exception("alarm_action %s failed", action)
        threading.Thread(target=run, daemon=True, name="alarm-action").start()

    def _ev_alarm_stopped(self, ev: AlarmStopped):
        if self._alarm and ev.alarm_id and self._alarm[0] != ev.alarm_id:
            return
        self._hide_alarm()

    def _hide_alarm(self):
        if self._alarm is None:
            return
        _id, card, _btns = self._alarm
        self._alarm = None
        try:
            card.place_forget()
            card.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------ config echoes
    def _on_config_change(self, name, value):
        if name == "model":
            self._asr_text = fmt_asr(str(value))
        elif name == "hotword":
            self.status_strip.set_hotword(bool(value))
            self._refresh_placeholder()
        elif name == "tts_engine":
            self._tts_text = self._tts_desc(str(value))

    # ------------------------------------------------------------- temps
    def _temps_worker(self):
        """Reads CPU/GPU temp + utilization every 5s into a plain attribute
        (never touches widgets); the Tk after-loop displays it. Utilization
        is included because Spark temps idle nearly flat — load is what
        actually moves. Every 6th pass also asks Ollama which model is
        actually resident (for the reactor HUD's LLM readout)."""
        self._cpu_stat_prev = None
        self._probe_devices()          # one-time GPU inventory (DEV row)
        beat = 0
        while not self._closing:
            perf.mark("smi")                # nvidia-smi spawn (5 s)
            self._temps_text = self._read_temps()
            self._mem_text = self._read_mem()
            if beat % 6 == 0:
                self._probe_llm()
            beat += 1
            time.sleep(5)

    def _probe_devices(self):
        """One-time GPU inventory (worker thread) for the engine card's
        GPU row — real hardware names from nvidia-smi; 'NONE' when it
        reports nothing."""
        names = []
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=3)
            names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
        except (OSError, subprocess.SubprocessError):
            log.debug("gpu name probe failed", exc_info=True)
        if names:
            short = names[0].replace("NVIDIA", "").replace(
                "GeForce", "").strip()
            self._dev_text = (f"{len(names)}× {short}"
                              if len(names) > 1 else short)
        else:
            self._dev_text = "NONE"

    def _read_mem(self) -> str:
        """Used system RAM from /proc/meminfo → '26.8 GB'."""
        try:
            info = {}
            with open("/proc/meminfo") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    info[key] = int(rest.split()[0])
                    if "MemTotal" in info and "MemAvailable" in info:
                        break
            return fmt_mem_gb(info.get("MemTotal", 0),
                              info.get("MemAvailable", 0))
        except (OSError, ValueError, IndexError):
            log.debug("meminfo read failed", exc_info=True)
        return ""

    def _probe_llm(self):
        """Engine-card THINK value = the CONFIGURED brain model, re-read
        from jarvis.brain.OLLAMA_MODEL each pass so brain.configure() at
        app start (assistant.local_model) is reflected. No /api/ps probe:
        the user runs other agents, and a foreign resident model is not
        what Jarvis thinks with."""
        try:
            from jarvis import brain
            name = getattr(brain, "OLLAMA_MODEL", None) or self._llm_name
        except Exception:
            log.debug("brain model read failed", exc_info=True)
            name = self._llm_name
        self._llm_name = name
        self._llm_text = name

    def _cpu_percent(self):
        """CPU utilization from /proc/stat deltas between polls."""
        try:
            with open("/proc/stat") as fh:
                fields = [int(x) for x in fh.readline().split()[1:8]]
            idle = fields[3] + fields[4]
            total = sum(fields)
            prev = self._cpu_stat_prev
            self._cpu_stat_prev = (idle, total)
            if prev is None or total == prev[1]:
                return None
            d_total = total - prev[1]
            return max(0, min(100, round(100 * (1 - (idle - prev[0]) / d_total))))
        except (OSError, ValueError, IndexError):
            return None

    def _read_temps(self) -> str:
        parts = []
        try:
            from pathlib import Path
            temps = []
            for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
                try:
                    temps.append(int(zone.read_text().strip()) / 1000.0)
                except (ValueError, OSError):
                    continue
            cpu_pct = self._cpu_percent()
            if temps:
                seg = f"cpu {max(temps):.0f}°"
                if cpu_pct is not None:
                    seg += f" {cpu_pct}%"
                parts.append(seg)
        except Exception:
            log.debug("cpu temp read failed", exc_info=True)
        try:
            # memory.used reads "N/A" on GB10 unified memory (see
            # context.py:347) and was never displayed -- only vals[0] and
            # vals[1] are read below. Querying it made nvidia-smi probe a
            # field it cannot answer, and this call was timing out at 3 s
            # often enough to fill the log with tracebacks.
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            lines = out.stdout.strip().splitlines()
            for gi, line in enumerate(lines):
                vals = [v.strip() for v in line.split(",")]
                if gi == 0 and vals and vals[0].isdigit():
                    seg = f"gpu {vals[0]}°"
                    if len(vals) > 1 and vals[1].isdigit():
                        seg += f" {vals[1]}%"
                    parts.append(seg)
        except (OSError, subprocess.SubprocessError):
            log.debug("gpu temp read failed", exc_info=True)
        return " · ".join(parts)

    def _temps_tick(self):
        if self._closing:
            return
        perf.mark("temps_tick")
        self.status_strip.set_temps(self._temps_text)
        self.status_strip.set_memory(self._mem_text)
        try:
            self.root.after(2500, self._temps_tick)
        except tk.TclError:
            pass


# ---------------------------------------------------------------- helper
def create(services: Optional[Services] = None) -> MainWindow:
    """Convenience for the app: build the root + MainWindow. The caller
    still runs root.mainloop()."""
    root = tk.Tk(className="jarvis")   # WM_CLASS for the .desktop launcher
    return MainWindow(root, services)
