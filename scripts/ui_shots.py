#!/usr/bin/env python3
"""Photograph the Jarvis console in every state, on a private display, safely.

    ~/vss_env/bin/python scripts/ui_shots.py --display :91 --scale 2.0 \\
        --geometry 920x1440 --look both --out ~/scratch/ui-shots

WHAT IT MAKES. ``DIR/<look>/NN-<state>.png`` for every state below, in one
look or both (``--look holo|classic|both``), and ``DIR/manifest.json``
naming each shot, the bus events that produced it, its size, and the states
that could not be driven on this tree. One summary line per shot on stdout.

HOW IT DRIVES THE WINDOW. The REAL console is built through
``jarvis.ui.main_window.create(services)`` and then pushed through its
states by publishing the same bus events the app publishes
(``jarvis/events.py``); nothing in ``jarvis/ui`` is edited. Whatever the
window probes at construction is faked BY INJECTION before ``create()``:
the tmux probes, the nvidia-smi telemetry worker, the tray icon, the global
hotkey thread, the microphone flag, the saved geometry. The Tk loop is
``root.mainloop()`` with an ``after``-driven step chain, not an ``update()``
pump -- the reactor's backdrop worker hands its image over from a thread
with ``after()``, which is silently dropped under a pump (W2 lane A2,
2026-09-01), so pumped shots never show the stage ambience.

THE CAMERA, MECHANICALLY. Hunter, 2026-09-02: "I dont want you to look at
anything the camera sees without my explicit permission." This rig is built
so it CANNOT: ``Services.camera_feed`` is None (asserted); the capture
worker class is replaced with a thread-less stand-in before the window is
built, so ``campreview.PreviewWorker`` is never instantiated (asserted); an
import blocker refuses ``cv2`` and the vision modules for the life of the
process (asserted at the end that none of them loaded); and the one
"picture" the camera pane ever shows is a gradient this script DRAWS,
handed to the pane as a ``PreviewShot``. No device node is named, opened or
read anywhere in this file, and ``tests/test_ui_shots.py`` greps it to keep
that true.

Each look runs in its own child process (the look is read once in
``create()``; the bus is a module singleton), so ``--look both`` is two
children run one after the other, and this file merges their manifests.
Under three minutes for both looks on the Spark.

THE STATES, in capture order. This list is the contract:
tests/test_ui_shots.py pins ``STATES`` to it.

  01  boot            Status("Loading speech model…", busy) right after
                      create(), the avatar bake still filling. A busy Status
                      renders no text by design (main_window.set_status), so
                      the pill reads READY; the stage is what differs.
  02  ready           idle console, ModelInfo landed, pill READY
  03  listening       HotwordDetected + RecordingStarted + AudioLevel +
                      PartialText (the live partial-transcript ghost card)
  04  thinking        RecordingStopped + Transcribed + UserUtterance +
                      BrainState(thinking)
  05  speaking        JarvisReply + SpeakingState(active): a reply being spoken
  06  warn            Status(warn) + FaultRaised(warn): toast, amber dot,
                      the engine card's FAULT row
  07  error           Status(error): pill ERROR + toast
  08  transcript      six more turns -- one long multi-sentence reply and
                      one short one -- eight turns on screen in all
  08b transcript-clean  the same eight turns with Status("Ready") published
                      first and 07's error toast EXPIRED before the shot
                      (it lives 4 s; an ok Status clears the pill hold but
                      not the toast, so 08 at ~3.4 s still carries it)
  09  briefing        BriefingReady: the WEATHER / CALENDAR / DUE tool card
                      (the only tool-card type on this tree)
  10  uncertain       UncertainUtterance: the YES / NO "Was that for me?" card
  11  approval        ApprovalRequested: the ALLOW / DENY card
  12  alarm           AlarmFired: the DISMISS / SNOOZE modal over the stage
  13  timer           SKIPPED -- no running-countdown surface exists at 59bb901
  14  settings        the settings drawer open (drawer_toggle, the gear's command)
  15  board           BoardCommand(show) + BoardUpdate: the Board docked
                      beside the console (console + board in one frame)
  16  ambient         desk idle 120 s: the room slab over the transcript
  17  standby         desk idle 99999 s: dimmed, footer hidden, burn-in drift
  18  sensing         SensingChanged(camera + radar on): badge SENSING
  19  camera-off      SensingChanged(curfew): badge CAM OFF (the header-fit
                      word; the pane placeholder below still says CAMERA OFF)
  20  offline         SensingChanged(offline): badge OFFLINE
  21  pane-off        the camera pane packed with its capture declining:
                      the "CAMERA OFF · curfew until 7 am" placeholder
  22  pane-synthetic  the camera pane showing a DRAWN gradient frame with a
                      face box and "HUNTER 0.74" drawn into the picture
  23  claude-task     ClaudeTaskState(running) + ClaudeProgress: pill
                      WORKING, the terminal button's ring, the progress card
  24  cleared         the two open questions (10, 11) answered, then
                      ClearTranscript + the spoken confirmation: the one
                      card left on the empty glass (commander's own order)
  25  header-worst    the header alone at exactly 918 px in the worst pair:
                      LISTENING + CAM OFF
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

XVFB = os.path.expanduser("~/.local/xvfb/usr/bin/Xvfb")
SCREEN = (1920, 2200)
LOOKS = ("holo", "classic")
CHILD_TIMEOUT_S = 170.0

# (number, slug, skipped-reason-or-None). Mirrors the docstring above.
STATES = (
    ("01", "boot", None),
    ("02", "ready", None),
    ("03", "listening", None),
    ("04", "thinking", None),
    ("05", "speaking", None),
    ("06", "warn", None),
    ("07", "error", None),
    ("08", "transcript", None),
    ("08b", "transcript-clean", None),
    ("09", "briefing", None),
    ("10", "uncertain", None),
    ("11", "approval", None),
    ("12", "alarm", None),
    ("13", "timer", "no running-countdown surface exists at 59bb901"),
    ("14", "settings", None),
    ("15", "board", None),
    ("16", "ambient", None),
    ("17", "standby", None),
    ("18", "sensing", None),
    ("19", "camera-off", None),
    ("20", "offline", None),
    ("21", "pane-off", None),
    ("22", "pane-synthetic", None),
    ("23", "claude-task", None),
    ("24", "cleared", None),
    ("25", "header-worst", None),
)


def skip_reason(num: str) -> str:
    """The skip reason for state `num`, by NUMBER: indexing STATES by
    position broke the day 08b went in between 08 and 09."""
    for n, _slug, why in STATES:
        if n == num:
            return why or ""
    return ""


# The spoken confirmation the commander publishes right after
# ClearTranscript (jarvis/commander.py TRANSCRIPT_CLEAR_LINE, held == 0).
# Copied, not imported: the commander is 8000 lines the child has no other
# use for, and tests/test_ui_shots.py pins the two strings equal.
TRANSCRIPT_CLEAR_LINE = ("Screen's clear, sir. Nothing forgotten — "
                         "and nothing to bring back.")

# Modules through which a frame could reach this process. Refused for the
# life of the child, and checked at the end to have never loaded.
BLOCKED_MODULES = ("cv2", "jarvis.camera", "jarvis.eye", "jarvis.facedetect",
                   "jarvis.facemodels", "jarvis.facegallery", "jarvis.visionrig")

# assistant.json options the window reads, answered from memory.
OPTIONS = {
    "console.board": True,
    "console.ambient": True,
    "console.standby": True,
    "console.powerup": False,
    "console.look": None,
    "presence.desk_standby": True,
    "camera.preview": False,          # the pane is toggled on for 21/22 only
    "camera.preview_fps": 6.0,
}

ROOM = {"playing": "", "next": "BIOSENSORS  ·  10:00", "due": "LAB REPORT  ·  NOON",
        "temp": "71°", "arc": "MORNING", "presence": "OFFICE", "quiet": "",
        "gpu": 2}

USER_1 = "what's on my calendar tomorrow"
REPLY_1 = ("Tomorrow you have the biosensors lecture at ten in Stanley Hall, "
           "a lab meeting at two, and dinner with Sam at seven, sir.")
TURNS = (
    ("set a timer for ten minutes", "Certainly, sir. Ten minutes, starting now.", 0.9),
    ("how's the weather looking",
     "Seventy-one and clear at the moment, sir, with a high of seventy-eight "
     "and no rain expected.", 1.4),
    ("what did the training run do overnight",
     "Three things worth knowing, sir. The VSS run finished at epoch eleven "
     "with validation loss flat since epoch eight, so the checkpoint from "
     "eight is the one to keep. Nothing regressed on the held-out set. And the "
     "GPU has been idle since four this morning, so the box is free if you "
     "want to start the next one.", 2.1),
    ("thanks jarvis", "My pleasure, sir.", 0.7),
    ("is the lab report due tomorrow",
     "Noon tomorrow, sir. You have the draft from Tuesday in the docs folder; "
     "I can bring it up when you're ready.", 1.2),
    ("play something quiet", "Playing your Focus playlist, quietly.", 1.0),
)
BRIEFING = {
    "weather": "71° and clear now, high of 78°, no rain expected",
    "calendar": ["10:00  Biosensors lecture — Stanley Hall 106",
                 "14:00  Lab meeting", "19:00  Dinner with Sam"],
    "due": ["Lab report — BioE 115 — tomorrow at noon"],
}


# --------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(msg, flush=True)


def _display_alive(display: str) -> bool:
    try:
        from Xlib import display as xdisplay
        d = xdisplay.Display(display)
        d.close()
        return True
    except Exception:                          # noqa: BLE001 - probe boundary
        return False


def ensure_display(display: str, size=SCREEN) -> Optional[subprocess.Popen]:
    """Attach to a live X server at `display`, or start a private Xvfb
    there. Returns the Popen when this process started it, else None."""
    if _display_alive(display):
        log(f"display {display}: attached to a running server")
        return None
    if not os.path.exists(XVFB):
        raise SystemExit(f"no Xvfb at {XVFB} and nothing listening on {display}")
    proc = subprocess.Popen(
        [XVFB, display, "-screen", "0", f"{size[0]}x{size[1]}x24",
         "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        time.sleep(0.1)
        if _display_alive(display):
            log(f"display {display}: started Xvfb pid {proc.pid} "
                f"({size[0]}x{size[1]})")
            return proc
    proc.terminate()
    raise SystemExit(f"Xvfb did not come up on {display}")


def shot_filename(num: str, slug: str) -> str:
    return f"{num}-{slug}.png"


def describe_event(ev) -> dict:
    """A manifest-safe summary of a bus event: scalars and short strings,
    never a payload object."""
    out = {"event": type(ev).__name__}
    for f in dataclasses.fields(ev):
        if f.name == "t":
            continue
        v = getattr(ev, f.name)
        if isinstance(v, (list, tuple)) and len(v) > 8:
            v = f"<{len(v)} values>"
        elif hasattr(v, "panels"):
            v = {"panels": [p.key for p in v.panels]}
        elif not isinstance(v, (str, int, float, bool, type(None))):
            v = repr(v)[:80]
        out[f.name] = v
    return out


# ------------------------------------------------- the camera stand-ins
class RigSensing:
    """Stands in for jarvis.sensing.SensingPolicy: the two reads the
    console makes (state(), curfew()) and NOTHING that can reach a device.
    `mode` is on | curfew | offline, switched by the rig per state."""

    CURFEW = ((21, 0), (7, 0))

    def __init__(self, mode: str = "on"):
        self._state = None
        self.set(mode)

    def set(self, mode: str) -> None:
        from jarvis.sensing import SensingState
        if mode == "curfew":
            self._state = SensingState(camera=False, radar=True, offline=False,
                                       reason="curfew", curfew=self.CURFEW)
        elif mode == "offline":
            self._state = SensingState(camera=False, radar=False, offline=True,
                                       reason="offline", curfew=self.CURFEW)
        else:
            self._state = SensingState(camera=True, radar=True, offline=False,
                                       reason="", curfew=self.CURFEW)

    def state(self):
        return self._state

    def curfew(self):
        return self.CURFEW

    def event(self):
        """The SensingChanged the spoken switch would publish for this state."""
        from jarvis.events import SensingChanged
        s = self._state
        return SensingChanged(camera=s.camera, radar=s.radar, offline=s.offline,
                              reason=s.reason, until=s.until)


class RigPreviewWorker:
    """Stands in for jarvis.campreview.PreviewWorker with the surface the
    pane and the window use -- and no thread, no pipeline, no device. The
    only pictures it ever holds are the ones the rig hands it through
    `show()`, and those are drawn by this script."""

    def __init__(self, *, get_option=None, sensing=None, services=None,
                 box=(160, 90), **_ignored):
        from jarvis.campreview import REASON_DISABLED, blank
        self.get_option = get_option
        self.sensing = sensing
        self.services = services
        self.box = (int(box[0]), int(box[1]))
        self.running = False
        self.cycles = 0
        self._seq = 0
        self._shot = blank(REASON_DISABLED)
        self.staged = None            # what start() shows, set by the rig

    @property
    def fps(self) -> float:
        return float(OPTIONS.get("camera.preview_fps") or 6.0)

    def _put(self, shot) -> None:
        self._seq += 1
        self._shot = dataclasses.replace(shot, seq=self._seq, at=time.time())

    def start(self, enabled: Optional[bool] = None) -> bool:
        from jarvis.campreview import REASON_DISABLED, REASON_SENSING, blank
        want = bool(OPTIONS.get("camera.preview")) if enabled is None else bool(enabled)
        if not want:
            self._put(blank(REASON_DISABLED))
            return False
        self.running = True
        self._put(self.staged or blank(REASON_SENSING, "curfew until 7 am"))
        return True

    def stop(self, timeout: float = 2.0, join: bool = True) -> None:
        from jarvis.campreview import REASON_DISABLED, blank
        self.running = False
        self._put(blank(REASON_DISABLED))

    def set_enabled(self, enabled: bool) -> None:
        self.start(enabled=True) if enabled else self.stop()

    def latest(self):
        return self._shot

    def status(self) -> dict:
        out = self._shot.numbers_only()
        out.update({"running": self.running, "cycles": self.cycles,
                    "box_w": self.box[0], "box_h": self.box[1]})
        return out

    def show(self, shot) -> None:
        """The rig's seam: put a shot it drew into the latest-wins slot."""
        self._put(shot)


def synthetic_frame(box: tuple, cap: tuple = (1280, 720), label: str = "HUNTER 0.74",
                    face: tuple = (470, 150, 300, 340)):
    """A gradient this script DRAWS -- nothing a lens saw -- letterboxed to
    the pane's box the way campreview.grab would, with a lighter oval where
    the face box will land and `label` drawn into the picture as a chip.
    Returns a PIL Image at the pane's picture size."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from jarvis.campreview import fit_box

    _ox, _oy, w, h = fit_box(cap[0], cap[1], box[0], box[1])
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xs / max(1, w - 1)) * 0.55 + (ys / max(1, h - 1)) * 0.45
    r = 16 + 62 * t
    g = 22 + 74 * t
    b = 38 + 96 * t
    arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    sx, sy = w / float(cap[0]), h / float(cap[1])
    fx, fy, fw, fh = face
    x0, y0, x1, y1 = fx * sx, fy * sy, (fx + fw) * sx, (fy + fh) * sy
    # a head-shaped lighter oval inside the face region
    draw.ellipse((x0 + (x1 - x0) * 0.12, y0 + (y1 - y0) * 0.05,
                  x1 - (x1 - x0) * 0.12, y1 - (y1 - y0) * 0.05),
                 fill=(92, 104, 128))
    # the identity chip, drawn INTO the picture above the box
    try:
        font = ImageFont.load_default(size=11)
    except TypeError:                          # older Pillow: no size kwarg
        font = ImageFont.load_default()
    tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), label, font=font)
    tw, th = tx1 - tx0, ty1 - ty0
    cx0 = int(max(2, min(w - tw - 8, x0)))
    cy0 = int(max(2, y0 - th - 10))
    draw.rectangle((cx0, cy0, cx0 + tw + 6, cy0 + th + 4), fill=(10, 14, 24))
    draw.text((cx0 + 3, cy0 + 2), label, fill=(232, 244, 255), font=font)
    return img


def synthetic_shot(box: tuple):
    """The PreviewShot for state 22: the drawn frame plus ONE face as
    numbers, attending, confidence 0.74 (the readout column prints it)."""
    from jarvis.campreview import REASON_LIVE, PreviewFace, PreviewShot
    cap = (1280, 720)
    face = PreviewFace(conf=0.74, x=470.0, y=150.0, w=300.0, h=340.0,
                       yaw_deg=6.0, attending=True, landmarks_ok=True)
    return PreviewShot(image=synthetic_frame(box, cap), faces=(face,),
                       cap_w=cap[0], cap_h=cap[1], reason=REASON_LIVE,
                       detail="", fps=6.0, grab_ms=1.2)


# ---------------------------------------------------------- the services
class DeskIdle:
    """services.desk_idle_s: seconds since desk input, set by the rig."""

    def __init__(self):
        self.idle = 0.0

    def __call__(self) -> float:
        return float(self.idle)


def build_services(sensing: Optional[RigSensing] = None,
                   desk: Optional[DeskIdle] = None):
    """The Services the rig hands to create(): camera_feed None, a sensing
    stand-in that cannot reach a device, and only harmless callables."""
    from jarvis.ui.main_window import Services

    calls: list = []

    def quiet(name: str) -> Callable:
        def fn(*a, **k):
            calls.append((name, a, tuple(sorted(k))))
        fn.__name__ = name
        return fn

    def get_option(key, default=None):
        value = OPTIONS.get(key)
        return default if value is None else value

    def set_option(key, value):
        OPTIONS[key] = value

    svc = Services(
        start_recording=quiet("start_recording"),
        stop_recording=quiet("stop_recording"),
        cancel_recording=quiet("cancel_recording"),
        history_prev=lambda: None,
        history_next=lambda: None,
        dispatch_text=None,
        toggle_hotword=quiet("toggle_hotword"),
        quit=None,
        open_terminal=quiet("open_terminal"),
        alarm_action=quiet("alarm_action"),
        approval_answer=quiet("approval_answer"),
        uncertain_answer=quiet("uncertain_answer"),
        get_option=get_option,
        set_option=set_option,
        room_state=lambda gpu_pct=None: dict(ROOM),
        desk_idle_s=desk or DeskIdle(),
        camera_feed=None,
        sensing=sensing or RigSensing(),
        board_closed=quiet("board_closed"),
    )
    svc._rig_calls = calls               # type: ignore[attr-defined]
    return svc


# ----------------------------------------------------------- the child
class _LensBlocker:
    """A meta-path finder that refuses the modules a frame could travel
    through. Installed before any jarvis import in the child."""

    def find_spec(self, name, path=None, target=None):
        if name in BLOCKED_MODULES or name.startswith("cv2."):
            raise ImportError(f"ui_shots: import of {name} is blocked -- "
                              "this rig never touches a lens")
        return None


def _firewall_env(env_dir: str) -> None:
    """Every path the app can write goes to a throwaway directory (the
    same redirects tests/conftest.py makes), so the rig never reads his
    secrets or writes his state. His voice_settings.json is not env-
    redirectable; CONFIG.save is no-op'd in inject() for that one."""
    os.makedirs(env_dir, exist_ok=True)
    e = os.environ
    e["JARVIS_LOG_DIR"] = os.path.join(env_dir, "log")
    e["JARVIS_ASSISTANT_CONFIG"] = os.path.join(env_dir, "assistant.json")
    e["JARVIS_CACHE_DIR"] = os.path.join(env_dir, "cache")
    e["JARVIS_MEMORY_DIR"] = os.path.join(env_dir, "memory")
    e["JARVIS_LEGACY_DIR"] = os.path.join(env_dir, "legacy")
    e["JARVIS_SPOTIFY_TOKEN"] = os.path.join(env_dir, "spotify_token.json")
    e["JARVIS_INTENT_LOG"] = os.path.join(env_dir, "intent_log.json")
    e["JARVIS_VOICEPRINT"] = os.path.join(env_dir, "voiceprint.npz")
    e["JARVIS_FACE_GALLERY"] = os.path.join(env_dir, "face_gallery")
    e["JARVIS_FACE_MODEL_DIR"] = os.path.join(env_dir, "face_models")
    e["JARVIS_DOCS_INDEX_DIR"] = os.path.join(env_dir, "docs_index")
    e["JARVIS_ROOM_CONTROL"] = "0"
    e["JARVIS_DESK_PRESENCE"] = "0"
    os.makedirs(e["JARVIS_LOG_DIR"], exist_ok=True)


def inject(geometry: str) -> None:
    """Fake, by injection, everything the window probes at construction.
    Runs AFTER the jarvis imports and BEFORE create()."""
    from jarvis import campreview
    from jarvis.config import CONFIG
    from jarvis.ui import main_window as mw

    # the capture worker: the thread-less stand-in, never the real class
    campreview.PreviewWorker = RigPreviewWorker
    # his voice_settings.json must not be written (window_geometry etc.)
    CONFIG.save = lambda: None
    CONFIG.window_geometry = geometry
    # a microphone exists on his box; the rig has none
    mw.MACHINE.has_mic = True
    # tmux probes: never spawn
    mw.session_exists = lambda run=None: False
    mw.terminal_attached = lambda run=None: False
    mw.terminal_available = lambda: True
    # no tray (pystray could reach a session bus), no Xlib record thread
    mw.TrayIcon._setup = lambda self: None
    mw.GlobalHotkey.start = lambda self: None

    def temps_worker(self):
        """The 5 s telemetry worker without nvidia-smi: fixed readings for
        the status strip and the engine card, the real room and sensing
        probes on a 1 s pass so the slab and the badge follow the rig."""
        self._cpu_stat_prev = None
        self._dev_text = "GB10"
        self._temps_text = "cpu 52° 7% · gpu 44° 2%"
        self._mem_text = "26.8 GB"
        self._gpu_util_pct = 2
        while not self._closing:
            try:
                self._probe_llm()
                self._probe_room()
                self._probe_sensing()
            except Exception:              # noqa: BLE001 - the loop lives
                pass
            time.sleep(1.0)

    mw.MainWindow._temps_worker = temps_worker


class Rig:
    """The after()-driven step chain over one MainWindow."""

    def __init__(self, win, services, sensing: RigSensing, desk: DeskIdle,
                 look: str, display: str, out_dir: str, scale: float,
                 geometry: str):
        self.win = win
        self.root = win.root
        self.services = services
        self.sensing = sensing
        self.desk = desk
        self.look = look
        self.display = display
        self.out_dir = out_dir
        self.scale = scale
        self.geometry = geometry
        self.queue: list = []
        self.shots: list = []
        self.skipped: list = []
        self.errors: list = []
        self._events: list = []
        self._state_t0 = 0.0
        self._pending_state: Optional[tuple] = None

    # ------------------------------------------------------------ chain
    def step(self, fn: Callable, wait_ms: Optional[int] = 0) -> None:
        """Run fn, then continue after wait_ms. wait_ms None means fn
        continues the chain itself (by calling advance())."""
        self.queue.append((fn, wait_ms))

    def publish(self, *events) -> None:
        from jarvis.events import bus
        for ev in events:
            self._events.append(describe_event(ev))
            bus.publish(ev)

    def begin(self, num: str, slug: str) -> None:
        self._state_t0 = time.monotonic()
        self._events = []
        self._pending_state = (num, slug)

    def advance(self) -> None:
        if not self.queue:
            self.root.quit()
            return
        fn, wait = self.queue.pop(0)
        try:
            fn()
        except Exception as exc:               # noqa: BLE001 - keep shooting
            import traceback
            self.errors.append({"state": self._pending_state,
                                "error": repr(exc)})
            traceback.print_exc()
        if wait is not None:
            self.root.after(int(wait), self.advance)

    def wait_until(self, pred: Callable[[], bool], timeout_ms: int,
                   poll_ms: int = 200) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0

        def check():
            if pred() or time.monotonic() >= deadline:
                self.advance()
            else:
                self.root.after(poll_ms, check)
        check()

    def skip(self, num: str, slug: str, why: str) -> None:
        self.skipped.append({"state": f"{num}-{slug}", "why": why})
        log(f"{self.look:8s} {num}-{slug:<16s} SKIPPED: {why}")

    # ---------------------------------------------------------- capture
    def _bbox(self, tops) -> tuple:
        x0 = y0 = 10 ** 9
        x1 = y1 = -1
        for top in tops:
            top.update_idletasks()
            x, y = top.winfo_rootx(), top.winfo_rooty()
            w, h = top.winfo_width(), top.winfo_height()
            x0, y0 = min(x0, x), min(y0, y)
            x1, y1 = max(x1, x + w), max(y1, y + h)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        return (max(0, x0), max(0, y0), min(sw, x1), min(sh, y1))

    def capture(self, num: str, slug: str, bbox: Optional[tuple] = None,
                tops=None, note: str = "") -> None:
        from PIL import ImageGrab
        t0 = self._state_t0 or time.monotonic()
        if bbox is None:
            bbox = self._bbox(tops or [self.root])
        img = ImageGrab.grab(bbox=bbox, xdisplay=self.display)
        name = shot_filename(num, slug)
        path = os.path.join(self.out_dir, name)
        img.save(path)
        ms = int((time.monotonic() - t0) * 1000)
        w, h = img.size
        rel = os.path.join(self.look, name)
        entry = {"state": f"{num}-{slug}", "number": num, "slug": slug,
                 "file": rel, "look": self.look, "scale": self.scale,
                 "geometry": self.geometry, "size": [w, h],
                 "bbox": list(bbox), "ms": ms, "events": list(self._events),
                 "console_mode": getattr(self.win.modes, "mode", "?"),
                 "pill": self.win._app_state(), "note": note}
        self.shots.append(entry)
        log(f"{self.look:8s} {num}-{slug:<16s} {rel:<32s} {w}x{h}  {ms:5d} ms")

    def header_bbox(self) -> tuple:
        """The header strip alone: the shell's 1 px inset gives 918 px of
        header inside a 920 px window, plus the rule under it."""
        self.root.update_idletasks()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        hdr = self.win._header.winfo_height()
        rule = self.win._rule.winfo_height()
        return (x + 1, y + 1, x + 1 + self.win._header.winfo_width(),
                y + 1 + hdr + rule)

    # --------------------------------------------------------- the plan
    def plan(self) -> None:
        from jarvis.board import BoardState, Panel
        from jarvis.events import (ActiveProject, AlarmFired, AlarmStopped,
                                   ApprovalRequested, ApprovalResolved, AudioLevel,
                                   BoardCommand, BoardUpdate, BrainState,
                                   BriefingReady, ClaudeProgress, ClaudeTaskState,
                                   ClearTranscript, DeskState, FaultRaised,
                                   HotwordDetected, JarvisReply, ModelInfo,
                                   PartialText, RecordingStarted, RecordingStopped,
                                   SpeakingState, Status, Transcribed,
                                   UncertainResolved, UncertainUtterance,
                                   UserUtterance)
        from jarvis.ui.console_mode import ACTIVE, AMBIENT, STANDBY
        from jarvis.ui.main_window import CAMERA_PREVIEW_OPTION
        win, S = self.win, self.step

        # 01 boot ----------------------------------------------------------
        S(lambda: (self.begin("01", "boot"),
                   self.publish(Status(text="Loading speech model…", kind="busy"))),
          900)
        S(lambda: self.capture("01", "boot", note=(
            "a busy Status renders no text (main_window.set_status): the pill "
            "reads READY; the stage is mid-bake")), 0)
        # the bake: wait for the full cycle (or 30 s), then settle
        S(lambda: win.reactor.when_cycle_live(self.advance, timeout_s=30.0),
          None)

        # 02 ready ---------------------------------------------------------
        S(lambda: (self.begin("02", "ready"),
                   self.publish(ModelInfo(text="small · GPU fp16"),
                                Status(text="Ready", kind="ok"))), 900)
        S(lambda: self.capture("02", "ready"), 0)

        # 03 listening -----------------------------------------------------
        bars = [abs(math.sin(i / 3.0)) * 0.8 + 0.1 for i in range(64)]
        S(lambda: (self.begin("03", "listening"),
                   self.publish(HotwordDetected(score=0.82), RecordingStarted(),
                                AudioLevel(level=0.48, waveform=bars))), 250)
        S(lambda: self.publish(PartialText(text="what's on my calendar"),
                               AudioLevel(level=0.55, waveform=bars[::-1])), 300)
        S(lambda: self.publish(PartialText(text=USER_1),
                               AudioLevel(level=0.42, waveform=bars)), 700)
        S(lambda: self.capture("03", "listening"), 0)

        # 04 thinking ------------------------------------------------------
        S(lambda: (self.begin("04", "thinking"),
                   self.publish(RecordingStopped(reason="silence", endpoint="vad",
                                                 dead_air_s=0.8),
                                AudioLevel(level=0.0),
                                Transcribed(text=USER_1, confidence=0.93,
                                            speaker_score=0.41),
                                UserUtterance(text=USER_1, source="voice"),
                                BrainState(state="thinking"))), 900)
        S(lambda: self.capture("04", "thinking"), 0)

        # 05 speaking ------------------------------------------------------
        S(lambda: (self.begin("05", "speaking"),
                   setattr(win, "_utter_ts", time.monotonic() - 1.3),
                   self.publish(BrainState(state="idle"),
                                JarvisReply(text=REPLY_1, speak=True),
                                SpeakingState(active=True, amplitude=0.55))), 250)
        S(lambda: self.publish(SpeakingState(active=True, amplitude=0.7,
                                             amplitude_only=True)), 650)
        S(lambda: self.capture("05", "speaking"), 0)
        S(lambda: self.publish(SpeakingState(active=False, amplitude=0.0)), 300)

        # 06 warn ----------------------------------------------------------
        S(lambda: (self.begin("06", "warn"),
                   self.publish(FaultRaised(rule="memory", kind="warn",
                                            token="14 GB FREE",
                                            text="Memory is getting tight — 14 GB free"),
                                Status(text="Memory is getting tight — 14 GB free",
                                       kind="warn"))), 1200)
        S(lambda: self.capture("06", "warn"), 0)

        # 07 error ---------------------------------------------------------
        S(lambda: (self.begin("07", "error"),
                   self.publish(FaultRaised(rule="memory", cleared=True),
                                Status(text="Ollama is not responding", kind="error"))),
          900)
        S(lambda: self.capture("07", "error"), 0)
        S(lambda: self.publish(Status(text="Ready", kind="ok")), 300)

        # 08 transcript ----------------------------------------------------
        S(lambda: self.begin("08", "transcript"), 0)
        for user, reply, rtt in TURNS:
            S(lambda u=user: self.publish(UserUtterance(text=u, source="voice")), 120)
            S(lambda r=reply, d=rtt: (
                setattr(win, "_utter_ts", time.monotonic() - d),
                self.publish(JarvisReply(text=r, speak=True))), 120)
        S(lambda: None, 800)
        S(lambda: self.capture("08", "transcript"), 0)

        # 08b the same transcript, no toast --------------------------------
        # 07's error toast lives 4 s (main_window.set_status) and an ok
        # Status clears the PILL hold, not the toast -- so 08, ~3.4 s after
        # the error, still had it over the cards. Ready goes first here and
        # the rig waits on the toast's OWN timer to run out rather than
        # racing it with a guessed sleep.
        S(lambda: (self.begin("08b", "transcript-clean"),
                   self.publish(Status(text="Ready", kind="ok"))), 0)
        S(lambda: self.wait_until(lambda: win.toast._frame is None, 6000), None)
        S(lambda: None, 400)
        S(lambda: self.capture("08b", "transcript-clean", note=(
            "the eight turns of 08 with Status(Ready) published first and "
            "07's error toast expired before the shot")), 0)

        # 09 briefing card -------------------------------------------------
        S(lambda: (self.begin("09", "briefing"),
                   self.publish(UserUtterance(text="give me my morning brief",
                                              source="voice"))), 120)
        S(lambda: (setattr(win, "_utter_ts", time.monotonic() - 2.4),
                   self.publish(BriefingReady(sections=dict(BRIEFING),
                                              spoken="Here's your morning, sir."))),
          1000)
        S(lambda: self.capture("09", "briefing"), 0)

        # 10 uncertain -----------------------------------------------------
        S(lambda: (self.begin("10", "uncertain"),
                   self.publish(UncertainUtterance(
                       request_id="u-1", text="turn the lights down a bit",
                       question="Was that for me, sir?"))), 900)
        S(lambda: self.capture("10", "uncertain"), 0)

        # 11 approval ------------------------------------------------------
        S(lambda: (self.begin("11", "approval"),
                   self.publish(ApprovalRequested(
                       request_id="a-1",
                       question=("Claude wants to run the test suite in the VSS "
                                 "project. Shall I allow it, sir?"),
                       tool_name="Bash", detail="pytest -q", project="vss"))), 900)
        S(lambda: self.capture("11", "approval"), 0)

        # 12 alarm ---------------------------------------------------------
        S(lambda: (self.begin("12", "alarm"),
                   self.publish(AlarmFired(alarm_id="al-1", label="Biosensors lecture",
                                           kind="alarm", due_text="9:45 am"))), 1000)
        S(lambda: self.capture("12", "alarm"), 0)
        S(lambda: self.publish(AlarmStopped(alarm_id="al-1", action="dismiss")), 300)

        # 13 timer ---------------------------------------------------------
        S(lambda: self.skip("13", "timer", skip_reason("13")), 0)

        # 14 settings ------------------------------------------------------
        S(lambda: (self.begin("14", "settings"), win.drawer_toggle()), 900)
        S(lambda: self.capture("14", "settings"), 0)
        S(lambda: win.drawer.close(), 400)

        # 15 board ---------------------------------------------------------
        def board_state():
            return BoardState(panels=[
                Panel("vitals", "VITALS", [("GPU", "44°C · 2%"), ("MEMORY", "26.8 GB"),
                                           ("TRAINER", "none")], tone="ok"),
                Panel("turns", "TURNS", [("MEDIAN WAIT", "1.3 s"), ("LAST", "0.9 s"),
                                         ("TODAY", "38 turns")], tone="ok",
                      spark=(0.2, 0.4, 0.3, 0.9, 0.5, 0.6, 0.2, 0.7, 0.4, 0.3,
                             0.5, 0.8, 0.3, 0.2, 0.6)),
                Panel("sessions", "CLAUDE", [("RUNNING", "jarvis · header fit"),
                                             ("WAITING", "vss · permission")],
                      tone="warn"),
                Panel("deadlines", "DUE", [("LAB REPORT", "in 4 h"),
                                           ("BIOSENSORS", "tomorrow 10:00")],
                      tone="ok"),
                Panel("focus", "FOCUS", [], tone="idle"),
                Panel("quiet", "QUIET", [("UNTIL", "7:00 AM"), ("HELD", "2 messages")],
                      tone="warn"),
            ], at=time.time())

        S(lambda: (self.begin("15", "board"),
                   self.publish(BoardCommand(action="show"))), 300)
        S(lambda: self.publish(BoardUpdate(state=board_state()),
                               BoardCommand(action="focus", panel="sessions")), 1200)
        S(lambda: self.capture("15", "board", tops=[self.root] + (
            [win.board.top] if win.board is not None else [])), 0)
        S(lambda: self.publish(BoardCommand(action="hide")), 400)

        # 16 ambient -------------------------------------------------------
        S(lambda: (self.begin("16", "ambient"), setattr(self.desk, "idle", 120.0)), 0)
        S(lambda: self.wait_until(lambda: win.modes.mode == AMBIENT, 6000), None)
        S(lambda: None, 1200)
        S(lambda: self.capture("16", "ambient"), 0)

        # 17 standby -------------------------------------------------------
        S(lambda: (self.begin("17", "standby"), setattr(self.desk, "idle", 99999.0),
                   self.publish(DeskState(at_desk=False, idle_s=99999.0))), 0)
        S(lambda: self.wait_until(lambda: win.modes.mode == STANDBY, 6000), None)
        S(lambda: None, 2500)
        S(lambda: self.capture("17", "standby"), 0)
        S(lambda: (setattr(self.desk, "idle", 0.0),
                   self.publish(DeskState(at_desk=True, idle_s=0.0, returned=True),
                                HotwordDetected(score=0.79))), 0)
        S(lambda: self.wait_until(lambda: win.modes.mode == ACTIVE, 4000), None)
        S(lambda: self.publish(RecordingStarted()), 100)
        S(lambda: self.publish(RecordingStopped(reason="abort", endpoint="manual")), 600)

        # 18 / 19 / 20 the sensing badge ------------------------------------
        for num, slug, mode in (("18", "sensing", "on"),
                                ("19", "camera-off", "curfew"),
                                ("20", "offline", "offline")):
            S(lambda n=num, s=slug, m=mode: (
                self.begin(n, s), self.sensing.set(m),
                self.publish(self.sensing.event())), 800)
            S(lambda n=num, s=slug: self.capture(n, s), 0)

        # 21 the camera pane, capture declining ------------------------------
        def pane_on():
            self.sensing.set("curfew")
            self.publish(self.sensing.event())
            win.preview_worker.staged = None          # start() shows the curfew placeholder
            OPTIONS["camera.preview"] = True
            win._on_config_change(CAMERA_PREVIEW_OPTION, True)

        S(lambda: (self.begin("21", "pane-off"), pane_on()), 1000)
        S(lambda: self.capture("21", "pane-off", note=(
            "camera pane packed; the stand-in worker declines with the curfew "
            "reason -- no capture thread exists in this process")), 0)

        # 22 the camera pane, a DRAWN frame --------------------------------
        def pane_synthetic():
            self.sensing.set("on")
            self.publish(self.sensing.event())
            win.preview_worker.show(synthetic_shot(win.preview_worker.box))

        S(lambda: (self.begin("22", "pane-synthetic"), pane_synthetic()), 1000)
        S(lambda: self.capture("22", "pane-synthetic", note=(
            "the picture is a gradient drawn by scripts/ui_shots.py "
            "(synthetic_frame); the face box comes from PreviewFace numbers; "
            "'HUNTER 0.74' is drawn into the synthetic picture -- this tree's "
            "pane has no identity chip of its own")), 0)
        S(lambda: (OPTIONS.__setitem__("camera.preview", False),
                   win._on_config_change(CAMERA_PREVIEW_OPTION, False)), 500)

        # 23 a Claude task -------------------------------------------------
        S(lambda: (self.begin("23", "claude-task"),
                   self.publish(UserUtterance(text="have claude fix the header fit",
                                              source="voice"))), 120)
        S(lambda: (setattr(win, "_utter_ts", time.monotonic() - 1.1),
                   self.publish(JarvisReply(text="On it, sir. Claude is on the "
                                                 "jarvis project.", speak=True),
                                ActiveProject(slug="jarvis",
                                              path=os.path.expanduser("~/Jarvis")),
                                ClaudeTaskState(project="jarvis", task_id="t-1",
                                                state="running",
                                                text="Fix the header fit at 920 px"),
                                ClaudeProgress(project="jarvis", task_id="t-1",
                                               line="Read jarvis/ui/views.py"),
                                ClaudeProgress(project="jarvis", task_id="t-1",
                                               line="Edit jarvis/ui/views.py"),
                                ClaudeProgress(project="jarvis", task_id="t-1",
                                               line="Tests: 7310 passed"))), 1000)
        S(lambda: self.capture("23", "claude-task"), 0)
        S(lambda: self.publish(ClaudeTaskState(project="jarvis", task_id="t-1",
                                               state="done", text="Done")), 300)

        # 24 cleared -------------------------------------------------------
        # The commander's own order (commander._h_transcript_clear): the
        # ClearTranscript event, then the spoken confirmation as the one card
        # left on the empty glass. clear_all leaves an UNANSWERED question
        # standing on purpose, and two are open on this glass (10's YES / NO,
        # 11's ALLOW / DENY), so both are answered first -- as they long
        # since would have been at a real desk.
        S(lambda: (self.begin("24", "cleared"),
                   self.publish(UncertainResolved(request_id="u-1", yes=True,
                                                  source="ui"),
                                ApprovalResolved(request_id="a-1", allowed=True,
                                                 source="ui"),
                                UserUtterance(text="clear the transcript",
                                              source="voice"))), 200)
        S(lambda: (setattr(win, "_utter_ts", time.monotonic() - 0.6),
                   self.publish(ClearTranscript(),
                                JarvisReply(text=TRANSCRIPT_CLEAR_LINE,
                                            speak=True))), 900)
        S(lambda: self.capture("24", "cleared", note=(
            "ClearTranscript after the two open questions were answered "
            "(an unanswered one is kept by design); the spoken confirmation "
            "is the one card on the empty glass")), 0)

        # 25 the header alone, worst pair ----------------------------------
        S(lambda: (self.begin("25", "header-worst"), self.sensing.set("curfew"),
                   self.publish(self.sensing.event(), HotwordDetected(score=0.84),
                                RecordingStarted(),
                                AudioLevel(level=0.5, waveform=bars))), 900)
        S(lambda: self.capture("25", "header-worst", bbox=self.header_bbox(), note=(
            "header + rule only, 918 px wide: LISTENING pill + CAM OFF badge")), 0)
        S(lambda: self.publish(RecordingStopped(reason="abort", endpoint="manual")), 200)

        S(self.teardown, 200)

    def teardown(self) -> None:
        win = self.win
        win._closing = True                 # stops the injected temps worker
        try:
            if win.modes is not None:
                win.modes.stop()
            if win.desk is not None:
                win.desk.stop()
            if getattr(win, "preview", None) is not None:
                win.preview.stop()
            if win.board is not None:
                win.board.destroy()
                win.board = None
        except Exception:                   # noqa: BLE001 - teardown path
            import traceback
            traceback.print_exc()


def run_child(args) -> int:
    look, display = args.look, args.display
    look_dir = os.path.join(args.out, look)
    os.makedirs(look_dir, exist_ok=True)
    _firewall_env(os.path.join(look_dir, "_env"))
    os.environ["DISPLAY"] = display
    os.environ["JARVIS_UI_SCALE"] = str(args.scale)
    os.environ["JARVIS_LOOK"] = look
    sys.meta_path.insert(0, _LensBlocker())

    settings = os.path.expanduser("~/.aiws_trainer/voice_settings.json")
    settings_mtime = os.path.getmtime(settings) if os.path.exists(settings) else None

    t_import = time.monotonic()
    from jarvis import campreview
    from jarvis.ui import main_window as mw
    from jarvis.ui import theme
    RealWorker = campreview.PreviewWorker
    inject(args.geometry + "+0+0")

    sensing, desk = RigSensing("on"), DeskIdle()
    services = build_services(sensing, desk)
    assert services.camera_feed is None, "the rig's Services must carry no camera feed"
    t_create = time.monotonic()
    win = mw.create(services)
    root = win.root
    root.geometry(args.geometry + "+0+0")
    root.update_idletasks()
    assert theme.LOOK == look, f"look {theme.LOOK!r} != {look!r}"
    assert not isinstance(win.preview_worker, RealWorker), \
        "the real PreviewWorker was constructed"
    assert isinstance(win.preview_worker, RigPreviewWorker)
    log(f"{look:8s} window {root.winfo_width()}x{root.winfo_height()} "
        f"S={win.scale:.2f} look={theme.LOOK} import {t_create - t_import:.1f}s "
        f"create {time.monotonic() - t_create:.1f}s")

    rig = Rig(win, services, sensing, desk, look, display, look_dir,
              args.scale, args.geometry)
    rig.plan()
    t_run = time.monotonic()
    root.after(50, rig.advance)
    root.mainloop()
    try:
        root.destroy()
    except Exception:                       # noqa: BLE001
        pass

    loaded = [m for m in BLOCKED_MODULES if m in sys.modules]
    assert not loaded, f"blocked modules loaded: {loaded}"
    if settings_mtime is not None and os.path.exists(settings):
        assert os.path.getmtime(settings) == settings_mtime, \
            "voice_settings.json was written by the rig"
    manifest = {
        "look": look, "display": display, "scale": args.scale,
        "geometry": args.geometry,
        "seconds": round(time.monotonic() - t_run, 1),
        "shots": rig.shots, "skipped": rig.skipped, "errors": rig.errors,
        "guards": {"camera_feed": None, "preview_worker": type(win.preview_worker).__name__,
                   "blocked_modules_loaded": loaded,
                   "service_calls": [c[0] for c in services._rig_calls]},
    }
    manifest["window"] = [int(args.geometry.split("x")[0]),
                          int(args.geometry.split("x")[1])]
    with open(os.path.join(look_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"{look:8s} done: {len(rig.shots)} shots, {len(rig.skipped)} skipped, "
        f"{len(rig.errors)} errors, {manifest['seconds']} s")
    return 1 if rig.errors else 0


# ---------------------------------------------------------- the parent
def run_parent(args) -> int:
    looks = list(LOOKS) if args.look == "both" else [args.look]
    os.makedirs(args.out, exist_ok=True)
    xvfb = ensure_display(args.display)
    t0 = time.monotonic()
    rc = 0
    merged: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "display": args.display, "scale": args.scale, "geometry": args.geometry,
        "states": [{"number": n, "slug": s, "skipped": why} for n, s, why in STATES],
        "looks": {},
    }
    try:
        for look in looks:
            cmd = [sys.executable, os.path.abspath(__file__), "--child",
                   "--look", look, "--display", args.display,
                   "--scale", str(args.scale), "--geometry", args.geometry,
                   "--out", args.out]
            try:
                r = subprocess.run(cmd, timeout=CHILD_TIMEOUT_S)
                code = r.returncode
            except subprocess.TimeoutExpired:
                code = -1
                log(f"{look:8s} child timed out after {CHILD_TIMEOUT_S:.0f} s")
            frag = os.path.join(args.out, look, "manifest.json")
            if os.path.exists(frag):
                with open(frag) as fh:
                    merged["looks"][look] = json.load(fh)
            else:
                merged["looks"][look] = {"look": look, "shots": [], "skipped": [],
                                         "errors": [{"error": f"child exit {code}"}]}
            merged["looks"][look]["exit_code"] = code
            rc = rc or (code != 0)
    finally:
        if xvfb is not None and not args.keep_display:
            xvfb.terminate()
            try:
                xvfb.wait(timeout=5)
            except subprocess.TimeoutExpired:
                xvfb.kill()
            log(f"display {args.display}: Xvfb stopped")
    merged["seconds"] = round(time.monotonic() - t0, 1)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(merged, fh, indent=2)
    n = sum(len(v["shots"]) for v in merged["looks"].values())
    log(f"manifest: {os.path.join(args.out, 'manifest.json')}  "
        f"{n} shots in {merged['seconds']} s")
    return int(rc)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--display", default=":91")
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--geometry", default="920x1440")
    p.add_argument("--look", choices=("holo", "classic", "both"), default="holo")
    p.add_argument("--out", required=True)
    p.add_argument("--keep-display", action="store_true",
                   help="leave a rig-started Xvfb running")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
