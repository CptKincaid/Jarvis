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
  26  sensors         the SENSORS page (F9 / win.sensors_toggle) over the
                      transcript: the office radar PRESENT at 1.4 m with the
                      camera naming a face, the kitchen sensor unreachable,
                      both distance bands and the overrule toggle. The radar
                      answers come from RigRadar, a TRANSPORT stand-in --
                      SensorPoller takes `get` as its seam, so no socket is
                      opened and no address on his LAN is named (the rig's
                      two rooms are RFC 5737 TEST-NET-1 literals)
  27  sensors-fault   the same page in its two other normal states: the
                      radar unreachable long enough to trip the breaker, and
                      the camera off for the curfew. Every unknown reads NO
                      OPINION with a reason, never "nobody there"
  28  standby-over-sensors  the console goes quiet with the SENSORS page
                      still open: the tab row goes with the footer, the page
                      is shut, and the note carries the number of radar
                      requests sent across the standby dwell (it is zero)
  29  sensor-setup    the SENSOR SETUP sheet over the page, opened on an
                      INVENTED profile the rig writes into its own throwaway
                      config directory: an invented SSID, a TEST-NET-1
                      address and two invented secrets. The sheet renders
                      "set" / "not set" for a secret and never a value, so
                      nothing here can photograph one. SKIPPED IN CLASSIC at
                      run time -- the SETUP button is holo's, because
                      classic is frozen at the jarvis-v3 tip
  30  users           the USERS page in the state HE is in: an owner with an
                      override code set, so the page is LOCKED and every
                      write asks for it. The people come from RigPeople, an
                      INVENTED book -- his own
                      ~/.local/state/jarvis/people.json is never opened by
                      this script, and there is no code path here that
                      could open it
  31  users-forget    the same page unlocked, with the destructive
                      confirmation open on a guest: what is removed, what
                      SURVIVES (the face gallery), the command that removes
                      that, and the label typed to confirm
  32  users-add       the add form, with the consent paragraph named for
                      the person being added -- the words that say a row
                      stores no measurement of anybody
  33  users-phrase    the passphrase panel on the owner's row: two MASKED
                      boxes, empty, and the sentences saying why this one
                      secret may be typed here when the override code may
                      not. No value is rendered anywhere
  34  users-purge     the gallery purge armed on a guest: what is destroyed,
                      what SURVIVES, that it may refuse rather than lie, and
                      the label typed to confirm. Arming destroys nothing
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
import urllib.parse
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
    ("26", "sensors", None),
    ("27", "sensors-fault", None),
    ("28", "standby-over-sensors", None),
    ("29", "sensor-setup", None),
    ("30", "users", None),
    ("31", "users-forget", None),
    ("32", "users-add", None),
    ("33", "users-phrase", None),
    ("34", "users-purge", None),
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
    # The two rooms the SENSORS page (26/27) lists. The addresses are RFC
    # 5737 TEST-NET-1 literals, never routed anywhere, and the page's poller
    # is handed RigRadar as its transport before the page is ever shown -- so
    # neither the rig nor a bug in it can reach a device on his LAN. The
    # kitchen ESP32 really is unflashed and off the network today, which is
    # why it is the one that renders unreachable.
    "presence.room_sensor_enabled": True,
    "presence.rooms": [
        {"name": "office", "label": "office", "url": "http://192.0.2.10",
         "primary": True},
        {"name": "kitchen", "label": "kitchen", "url": "http://192.0.2.11"},
    ],
    "presence.camera_overrules": True,
    # THE ZONE LADDERS, and they are the ones in his assistant.json today
    # (2026-09-03). The office desk is the FARTHER band -- the radar sits on
    # the desk aimed out across the room, so the near space is empty and he
    # reads at 3.13 m median -- and the kitchen has NO LENS, which is what
    # the blank camera_zone says. The page edits these; there is no second
    # two-band model any more (jarvis/ui/sensors_page.py).
    # NESTED, not dotted, because jarvis/zones.py deliberately reads the
    # SECTION in one go: a dotted read answers "absent" for every key under
    # a zones that is the wrong shape, which is how a whole broken section
    # slipped past a round of review. get_option below walks a dotted key
    # into this.
    "zones": {
        "enabled": True,
        "rooms": [
            {"name": "office", "enabled": True, "camera_zone": "at the desk",
             "bands": [{"name": "empty space", "near_m": 0.75, "far_m": 2.25},
                       {"name": "at the desk", "near_m": 2.25, "far_m": 3.75}]},
            {"name": "kitchen", "enabled": True, "camera_zone": "",
             "bands": [{"name": "the kitchen", "near_m": 0.75, "far_m": 3.0},
                       {"name": "at the door", "near_m": 3.0, "far_m": 3.75}]},
        ],
    },
}

# THE INVENTED PROFILE state 29 photographs. Every value here is made up:
# the SSID is not a network that exists, the address is RFC 5737 TEST-NET-1
# like the two rooms above, and the two "secrets" are strings with the word
# invented in them. jarvis.sensorprofile puts the file in
# PATHS.ASSISTANT_CONFIG.parent/room-sensors, which _firewall_env has already
# redirected into this run's throwaway directory, so his own profiles -- which
# hold his real Wi-Fi PSK and OTA password -- are never opened, read or
# rendered. Nothing in this rig ever reads them.
RIG_SSID = "PRETEND-NET-5G"
RIG_IP = "192.0.2.10"
RIG_PSK = "invented-psk-not-his"
RIG_OTA = "invented-ota-not-his"

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


# Displays this rig must never touch. ensure_display() attaches to ANY
# live server at --display, so `--display :1` would have built the console
# on his real desktop -- a Jarvis window popping up on :1, over whatever
# he is reading, driven through an alarm and a fake camera pane. :0 and :1
# (and their .N screens) are refused before anything is imported or
# started; the guard is a pure function so tests/test_ui_shots.py can pin
# it without a display.
FORBIDDEN_DISPLAYS = (":0", ":1")


def display_allowed(display: str) -> bool:
    """False for his desktop displays (:0, :1, :0.0, :1.0, with or without
    a host part); True for anything else (:91..:99, a private Xvfb)."""
    d = (display or "").strip()
    if not d:
        return False
    d = d[d.index(":"):] if ":" in d else d          # 'localhost:1.0' -> ':1.0'
    d = d.split(".")[0]                              # ':1.0' -> ':1'
    return d not in FORBIDDEN_DISPLAYS


def refuse_live_display(display: str) -> None:
    if not display_allowed(display):
        raise SystemExit(f"ui_shots: refusing display {display!r} -- that is "
                         "his desktop, never a rig display (use :91-:99)")


def ensure_display(display: str, size=SCREEN) -> Optional[subprocess.Popen]:
    """Attach to a live X server at `display`, or start a private Xvfb
    there. Returns the Popen when this process started it, else None.
    :0 and :1 are refused outright (display_allowed)."""
    refuse_live_display(display)
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


class RigRadar:
    """Stands in for the ESPHome web_server endpoints of the room sensors.

    It is a TRANSPORT, not a device: ``SensorPoller`` takes ``get`` as its
    seam, so handing it this replaces the socket entirely -- the rig opens
    none, and the two addresses it answers for are RFC 5737 TEST-NET-1
    literals that are not routed anywhere. A host it does not know RAISES,
    which is exactly what an unreachable ESP32 does and is how state 27
    trips the circuit breaker.

    ``present`` and ``cm`` are the numbers the page renders; the bodies are
    the shapes measured off the live office radar on 2026-09-03 (see
    jarvis/roomsensor.py's docstring for the curl transcript).
    """

    OFFICE, KITCHEN = "192.0.2.10", "192.0.2.11"

    def __init__(self):
        # host -> (presence bit, detection distance in cm), or None for a
        # host that answers nothing at all (the kitchen ESP32 is unflashed)
        self.rooms = {self.OFFICE: (True, 142), self.KITCHEN: None}
        # Every attempt, reachable or not: state 28 asserts that a page shut
        # by standby sends none, and a raise is still a request.
        self.calls = 0

    def __call__(self, url: str, timeout: float) -> str:
        self.calls += 1
        host = urllib.parse.urlsplit(url).hostname or ""
        answer = self.rooms.get(host)
        if answer is None:
            raise OSError("no route to host %s" % host)
        present, cm = answer
        if "binary_sensor" in url:
            return '{"id":"binary_sensor/Presence","value":%s,"state":"%s"}' % (
                "true" if present else "false", "ON" if present else "OFF")
        return '{"id":"sensor/Detection distance","value":%d,"state":"%d cm"}' % (
            cm, cm)


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


def synthetic_shot(box: tuple, name: str = "", id_score: float = 0.0):
    """The PreviewShot for state 22: the drawn frame plus ONE face as
    numbers, attending, confidence 0.74 (the readout column prints it).

    ``name``/``id_score`` are for state 26, where the SENSORS page needs a
    RECOGNISED face to show the camera overruling the radar. They default
    to the anonymous face state 22 has always used, and ``id_ran`` follows
    the name -- False is "identity was never asked", which is a different
    row on that page from "asked, and did not know him"."""
    from jarvis.campreview import REASON_LIVE, PreviewFace, PreviewShot
    cap = (1280, 720)
    face = PreviewFace(conf=0.74, x=470.0, y=150.0, w=300.0, h=340.0,
                       yaw_deg=6.0, attending=True, landmarks_ok=True,
                       name=name, id_score=id_score,
                       id_ran=bool(name or id_score))
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


class RigPeople:
    """An INVENTED people book for the USERS page (states 30-32).

    HIS OWN REGISTRY IS NEVER OPENED. jarvis/identity.py reads
    PATHS.OWNER_REGISTRY, and nothing in this file names that path or
    constructs a Registry: the page reaches the app only through
    Services.people_snapshot, and this answers it from the rows below. The
    labels, names and dates are made up, and the two "hashes" are booleans
    -- redacted() shape, so there is nothing here that a hash could be.

    The state it renders is the one HE is in today: one owner with an
    override code set, so the page opens LOCKED.
    """

    PATH = "/home/example/.local/state/jarvis/people.json"

    def __init__(self):
        self.people = [
            {"label": "alderman", "name": "Alderman", "role": "owner",
             "voice": True, "face": "alderman", "face_dim": 128,
             "has_phrase": False, "has_code": True, "consent": "owner",
             "enrolled_at": "2026-01-04T21:12:03"},
            {"label": "pemberton", "name": "Pemberton", "role": "known",
             "voice": False, "face": "pemberton", "face_dim": 128,
             "has_phrase": False, "has_code": False, "consent": "typed",
             "enrolled_at": "2026-02-17T10:40:55"},
            {"label": "marchbanks", "name": "Marchbanks", "role": "known",
             "voice": False, "face": "marchbanks-old", "face_dim": 128,
             "has_phrase": False, "has_code": False, "consent": "console",
             "enrolled_at": "2026-03-02T18:05:11"},
        ]
        self.calls: list = []
        # The app's dwell, which is what the WRITES consult -- the page's
        # own Lock is a mirror of it. The rig carries one so the frames
        # photograph the real arrangement rather than the older app half
        # that had no such seam.
        self.unlocked = False

    def admin_state(self) -> dict:
        from jarvis import gate as gate_mod
        self.calls.append("admin_state")
        return {"admin": gate_mod.ADMIN_CODE,
                "admin_line": ("an owner has set an override code, so it is "
                               "asked for before anything is changed"),
                "unlocked_s": 120.0 if self.unlocked else 0.0}

    def relock(self) -> None:
        self.calls.append("relock")
        self.unlocked = False

    def snapshot(self) -> dict:
        from jarvis import gate as gate_mod
        return {"people": [dict(p) for p in self.people],
                "gate_line": ("owner-gate: SHADOW -- 1 owner (alderman), "
                              "voice leg live, face leg unavailable (the "
                              "gallery is empty). Nothing is being refused."),
                "fault_kind": "", "path": self.PATH,
                # marchbanks points at a gallery label that is not there,
                # which is what the amber face chip is for.
                "gallery": ["alderman", "pemberton"],
                # The VOICE gallery's labels, beside the face gallery's: the
                # chip on each row is drawn from this rather than from a rule
                # about who is allowed a pool.
                "voices": ["alderman"],
                "admin": gate_mod.ADMIN_CODE,
                "admin_line": ("an owner has set an override code, so it is "
                               "asked for before anything is changed")}

    def unlock(self, code) -> tuple:
        self.calls.append("unlock")
        del code
        self.unlocked = True
        return True, "Unlocked, sir."

    def add(self, **kw) -> tuple:
        self.calls.append(("add", kw.get("label")))
        return True, "%s is enrolled" % kw.get("label")

    def set_role(self, label, role, **kw) -> tuple:
        self.calls.append(("set_role", label, role))
        return True, "%s is now %s" % (label, role)

    def forget(self, label) -> tuple:
        self.calls.append(("forget", label))
        return True, "%s is forgotten here" % label

    # ---- the enrolment seams (states 30-35). NOTHING HERE OPENS A DEVICE:
    # each answers a line, exactly as the app's seams do, so the page draws
    # its real face without a camera, a microphone or a gallery existing.
    # The rig asserts at teardown that no vision module was imported.
    def set_phrase(self, label, hashed) -> tuple:
        # The HASH is what crosses the seam; the rig records that it was
        # asked and never what it was handed.
        self.calls.append(("set_phrase", label))
        del hashed
        for person in self.people:
            if person["label"] == label:
                person["has_phrase"] = True
        return True, "Set, sir. It is stored salted-hashed; nothing here can "\
                     "read it back."

    def new_code(self) -> tuple:
        self.calls.append("new_code")
        return True, "Knightfall: a new code is in your inbox."

    def knightfall_status(self) -> dict:
        return {"to": "h\u2022\u2022\u2022\u2022@example.invalid",
                "problem": "", "setup": ""}

    def face_start(self) -> tuple:
        self.calls.append("face_enrol_start")
        return True, ("Right. The camera is coming on. Hold still when you "
                      "hear the tone.")

    def face_stop(self) -> tuple:
        self.calls.append("face_enrol_stop")
        return True, "Stopped, sir. Nothing was written."

    def voice_start(self) -> tuple:
        self.calls.append("voice_enrol_start")
        return True, "Right, sir. Eight takes of about eight seconds."

    def voice_stop(self) -> tuple:
        self.calls.append("voice_enrol_stop")
        return True, "Stopped, sir. Nothing was written."

    def purge_face(self, label) -> tuple:
        self.calls.append(("purge_face", label))
        return True, "Removed. 3 face generation(s) held %s." % label

    def purge_voice(self, label) -> tuple:
        self.calls.append(("purge_voice", label))
        return True, "Removed. 1 voice generation(s) held %s." % label


def build_services(sensing: Optional[RigSensing] = None,
                   desk: Optional[DeskIdle] = None,
                   people: Optional["RigPeople"] = None):
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
        # Flat dotted keys first (most of OPTIONS is written that way), then
        # a walk INTO a nested section -- "zones" is nested because
        # jarvis/zones.py reads that section whole.
        if key in OPTIONS:
            value = OPTIONS[key]
        else:
            value = OPTIONS
            for part in key.split("."):
                if not isinstance(value, dict) or part not in value:
                    return default
                value = value[part]
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
    book = people or RigPeople()
    svc.people_snapshot = book.snapshot
    svc.people_unlock = book.unlock
    svc.people_add = book.add
    svc.people_set_role = book.set_role
    svc.people_forget = book.forget
    svc.people_admin_state = book.admin_state
    svc.people_relock = book.relock
    svc.people_set_phrase = book.set_phrase
    svc.people_new_code = book.new_code
    svc.knightfall_status = book.knightfall_status
    svc.face_enrol_start = book.face_start
    svc.face_enrol_stop = book.face_stop
    svc.voice_enrol_start = book.voice_start
    svc.voice_enrol_stop = book.voice_stop
    svc.people_purge_face = book.purge_face
    svc.people_purge_voice = book.purge_voice
    svc._rig_people = book               # type: ignore[attr-defined]
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
        # The radar transport for states 26/27. Built here rather than in
        # build_services because it is not a service: it replaces the
        # SENSORS page's own HTTP client, and nothing else in the window
        # ever sees it.
        self.radar = RigRadar()
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

        # 26 the SENSORS page ----------------------------------------------
        def sensors_on():
            from jarvis.ui import sensors_page as sensors
            # The transport swap comes FIRST, and it is the whole safety
            # argument for these two states: the page builds its own
            # SensorPoller in __init__ with roomsensor's real HTTP client,
            # and this replaces it before the page is ever shown.
            win.sensors.poller = sensors.SensorPoller(win.sensors.specs,
                                                      get=self.radar)
            win.preview_worker.staged = synthetic_shot(
                win.preview_worker.box, name="hunterp", id_score=0.71)
            OPTIONS["camera.preview"] = True
            win._on_config_change(CAMERA_PREVIEW_OPTION, True)
            self.sensing.set("on")
            self.publish(self.sensing.event())
            win.sensors_toggle()

        S(lambda: (self.begin("26", "sensors"), sensors_on()), 1400)
        S(lambda: self.capture("26", "sensors", note=(
            "the office radar answers PRESENT at 1.42 m and the camera names a "
            "face, so the fused verdict is AT THE DESK by CAMERA; the kitchen "
            "ESP32 (unflashed, off the network) answers nothing. Every radar "
            "body comes from RigRadar, a transport stand-in -- no socket is "
            "opened and no address on his LAN is named")), 0)

        # 27 the same page with both legs dark ------------------------------
        def sensors_fault():
            from jarvis.campreview import REASON_SENSING, blank
            self.radar.rooms[RigRadar.OFFICE] = None      # unplug the office
            self.sensing.set("curfew")
            self.publish(self.sensing.event())
            win.preview_worker.show(
                blank(REASON_SENSING, "CAMERA OFF · curfew until 7 am"))

        S(lambda: (self.begin("27", "sensors-fault"), sensors_fault()), 4200)
        S(lambda: self.capture("27", "sensors-fault", note=(
            "four failed polls at 1 Hz trip roomsensor's breaker, so both "
            "rooms read NO OPINION with the retry in the fault line and the "
            "round trip a dash (no request was sent); the camera row carries "
            "the curfew sentence. Nothing here reads 'nobody there'")), 0)
        # 28 standby WITH the page open ------------------------------------
        # main_window._set_footer_hidden's own words: in standby the panel is
        # "a clock and nothing else". The tab row is packed above the stage,
        # so it was untouched by that and shipped lit and clickable over the
        # dimmed clock. Hiding the row is only half: a page left open would
        # keep POLLING behind the clock, at a radar the curfew may just have
        # powered down. The note carries the request count for the dwell.
        S(lambda: (self.begin("28", "standby-over-sensors"),
                   setattr(self.radar, "calls", 0),
                   setattr(self.desk, "idle", 99999.0),
                   self.publish(DeskState(at_desk=False, idle_s=99999.0))), 0)
        S(lambda: self.wait_until(lambda: win.modes.mode == STANDBY, 6000), None)
        S(lambda: None, 2500)
        S(lambda: self.capture("28", "standby-over-sensors", note=(
            "the SENSORS page was open when the console went quiet: the tab "
            "row is gone with the footer, the page is shut, and the radar "
            "transport was called %d times across the standby dwell"
            % self.radar.calls)), 0)
        S(lambda: (setattr(self.desk, "idle", 0.0),
                   self.publish(DeskState(at_desk=True, idle_s=0.0,
                                          returned=True))), 0)
        S(lambda: self.wait_until(lambda: win.modes.mode == ACTIVE, 4000), None)

        # 29 the SENSOR SETUP sheet ----------------------------------------
        def setup_on():
            from jarvis import sensorprofile
            from jarvis.ui import sensors_page as sensors
            # THE INVENTED PROFILE, written through the package's own writer
            # so it lands in the firewalled config directory at 0600 -- his
            # own profiles are never opened.
            sensorprofile.write(
                "office",
                {"ssid": RIG_SSID, "ip": RIG_IP, "gateway": "192.0.2.1",
                 "subnet": "255.255.255.0", "preset": "desk",
                 "nearest_m": 2.0, "range_m": 3.5, "still": True,
                 "timeout_s": 10, "flashed": True},
                password=RIG_PSK, ota_password=RIG_OTA)
            # the transport swap again, for the same reason as state 26: the
            # page rebuilds its poller whenever its room list changes.
            win.sensors.poller = sensors.SensorPoller(win.sensors.specs,
                                                      get=self.radar)
            if not win.sensors.is_open:
                win.sensors_toggle()
            win.sensors.open_setup()
            win.sensors.setup.select("office")

        def setup_state():
            from jarvis.ui.sensors_page import restyled
            if not restyled():
                self.skip("29", "sensor-setup",
                          "the SETUP button is holo's: classic is frozen at "
                          "the jarvis-v3 tip and gains no new control")
                return
            self.begin("29", "sensor-setup")
            setup_on()

        S(setup_state, 1200)
        S(lambda: (self.capture("29", "sensor-setup", note=(
            "the setup sheet over the SENSORS page, on an INVENTED profile "
            "the rig wrote into its own throwaway config directory. The two "
            "secret boxes are empty and masked and the word beside each says "
            "only whether one is set; no value is rendered anywhere"))
            if win.sensors.setup is not None and win.sensors.setup.is_open
            else None), 0)
        S(lambda: (win.sensors.setup.hide()
                   if win.sensors.setup is not None else None), 200)

        S(lambda: (OPTIONS.__setitem__("camera.preview", False),
                   win._on_config_change(CAMERA_PREVIEW_OPTION, False)), 400)

        # 30-32 the USERS page ---------------------------------------------
        # The people are RigPeople's INVENTED rows (build_services). His own
        # registry is never opened: the page reaches the app only through
        # Services.people_snapshot, and nothing in this file names
        # PATHS.OWNER_REGISTRY or builds a Registry.
        def users_on():
            strip = getattr(win, "tabs", None)
            if strip is not None and "users" in strip.keys:
                strip.select("users")
            elif getattr(win, "users", None) is not None:
                win.users.show()

        S(lambda: (self.begin("30", "users"), users_on()), 900)
        S(lambda: self.capture("30", "users", note=(
            "the state HE is in: one owner with an override code set, so the "
            "page is LOCKED and every write asks for it. The entry is masked "
            "and empty. Marchbanks' face chip is amber because the pointer "
            "names a gallery label that is not there -- which is exactly how "
            "the face leg stops naming anybody")), 0)

        def users_forget():
            page = win.users
            # THROUGH THE APP'S DWELL, not the page's: the page Lock is a
            # mirror the tick re-seeds from RigPeople every second, so
            # unlocking only the mirror would relock a second later and the
            # frame would photograph a refusal.
            page.services._rig_people.unlocked = True
            page._lock.unlock()
            page._forget_pressed("pemberton")

        S(lambda: (self.begin("31", "users-forget"), users_forget()), 700)
        S(lambda: self.capture("31", "users-forget", note=(
            "the destructive confirmation: what is removed, what SURVIVES "
            "(the face gallery entry) with the command that removes it, that "
            "it cannot be undone, and the label typed to confirm. Arming "
            "forgets nobody")), 0)

        def users_add():
            page = win.users
            page._forget.disarm()
            page._add_pressed()
            page._add_fields["label"].insert(0, "pemberton")
            page._add_fields["name"].insert(0, "Pemberton")
            page._retitle_consent()

        S(lambda: (self.begin("32", "users-add"), users_add()), 700)
        S(lambda: self.capture("32", "users-add", note=(
            "the add form with the consent paragraph named for the person: a "
            "row stores a name, a role and a face LABEL and no measurement "
            "of anybody, so these are not the words the camera ceremony "
            "shows. They type their own label to agree")), 0)
        def users_phrase():
            page = win.users
            page._cancel()
            page.services._rig_people.unlocked = True
            page._lock.unlock()
            page._phrase_pressed("alderman")

        S(lambda: (self.begin("33", "users-phrase"), users_phrase()), 700)
        S(lambda: self.capture("33", "users-phrase", note=(
            "the spoken passphrase, set without a terminal. Two MASKED boxes "
            "and EMPTY -- the control empties both before it hashes, so "
            "nothing typed survives the press -- with the sentences saying "
            "why this secret may be typed on his own console (it is said out "
            "loud in normal use and can be overheard; that is accepted) when "
            "the override code may not. There is no box to choose a code")),
          0)

        def users_purge():
            page = win.users
            page._cancel()
            page.services._rig_people.unlocked = True
            page._lock.unlock()
            page._purge_pressed("pemberton", "face")

        S(lambda: (self.begin("34", "users-purge"), users_purge()), 700)
        S(lambda: self.capture("34", "users-purge", note=(
            "removing somebody's FACE measurements from the tab: what is "
            "destroyed, that everybody else is carried forward and READ BACK "
            "off the disk before one old byte is touched, that anything which "
            "cannot be finished honestly is left alone and named, that it "
            "does not touch their voice pool, and the label typed to confirm. "
            "Arming destroys nothing")), 0)

        S(lambda: (win.users.hide() if getattr(win, "users", None) is not None
                   else None), 200)

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
            if getattr(win, "sensors", None) is not None:
                if getattr(win.sensors, "setup", None) is not None:
                    win.sensors.setup.hide()
                win.sensors.hide()          # stops its poll thread
            if getattr(win, "users", None) is not None:
                win.users.hide()            # stops its one-second repaint
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
    refuse_live_display(args.display)      # parent AND child, before any import
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
