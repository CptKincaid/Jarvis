"""The Jarvis reactor — state-machine visualization for the main window.

States (colors from theme.STATE_COLORS):
  idle      — slow 10s rotation, core breathing ±6%, CYAN_DIM
  listening — outer ring expands with AudioLevel.level, 64-bar circular
              waveform around the core from AudioLevel.waveform, CYAN
  thinking  — amber orbital comet sweep (WARN)
  speaking  — core pulses with SpeakingState.amplitude, expanding ripple
              rings (OK)

Centerpiece (the living particulate SPHERE): CENTERPIECE selects the
  pre-rendered base art. "avatar" (default) is the film JARVIS presence —
  a rotating VOLUME of streaked blue light baked as AV_FRAMES seamless
  rotation frames by jarvis.ui.avatar_bake (the pure kernel; frame
  AV_FRAMES == frame 0 bitwise). CENTERPIECE="reactor" restores the
  Mark I–III arc-reactor disc (`_build_base`), retained intact.

Motion (clock-indexed, boundary-scheduled — see jarvis.ui.avatar_clock):
  AV_FRAMES=600 over AV_PERIOD=10 s = 60 unique frames/s, one frame per
  60 Hz refresh (the rotation RATE is unchanged from the 300-frame days —
  36°/s idle, 72°/s thinking/speaking — only the temporal resolution
  doubled; 30 fps was what read as "laggy" on a 60 Hz panel). The frame
  shown at any instant is a pure function of a monotonic clock (no
  counter is incremented, so timer jitter never accumulates) and every
  tick is scheduled for the NEXT SLOT BOUNDARY rather than a fixed
  after(16). Everything else animated on the stage (instrument arcs,
  sweep, orbits, sparks, motes) samples the same slot time, so nothing
  churns off-grid between frame swaps.

Bake (out of process): the reactor launches AV_WORKERS
  `python -m jarvis.ui.avatar_bake` subprocesses (never multiprocessing —
  spawn would re-import jarvis.app → torch, fork is unsafe in a threaded
  Tk+CUDA process). Frames stream back as raw RGB over pipes, read by
  plain threads into a SimpleQueue, and are converted to PhotoImages on
  the Tk thread inside the frame loop's spare time — as many per tick as
  the MEASURED conversion cost says fit before the next boundary with
  2 ms to spare (avatar_clock.drain_budget; <= 3). Frames land
  PROGRESSIVELY on nested grids (every 24th → 12th → 6th → 3rd → all) so
  a coarse 25-frame cycle plays within seconds of boot; the upgrade to each
  finer tier crosses over on a frame both grids contain (same phase
  angle), so it never pops. Never-mapped Labels pin each photo's Tk
  display instance so a swap is a refcount op, not an XImage rebuild.

HUD scene: static decor (rebuilt only on size settle), two looks
  (theme.LOOK, read at call time — never captured at def time):
  "classic" is the 08-31 stage token for token: the 72-tick degree ruler
  (no numbers), six unequal instrument arcs on ONE radius, a 12-dash halo
  ring, two guide circles, corner brackets, split scanlines, and the
  ENGINE CARD as a lit glass slab. "holo" (2026-09-01, the blue
  holographic overhaul — the film JARVIS HUD, gold-on-black turned blue)
  is a FILM-STYLE STAGE instead: 1px frame lines bounding the stage with
  notched (chamfered) corners and an open bottom, a segmented status
  ruler along the top, a rail of tick marks down the right frame line, a
  small 270° dial gauge bottom-right whose needle reads the REAL 1-minute
  load average per core (/proc/loadavg, sampled on the existing 1 Hz
  telemetry tick — one coords() call), and the engine card redrawn as an
  outlined thin frame (1px GLASS_EDGE, transparent fill — no slab) with
  tracked-caps captions. Nothing filled, nothing that shimmers: the
  sphere is the only light. Both looks share the seam dissolve into the
  transcript, the engine card rows HEAR / SPEAK / THINK / DEVICE / FAULT
  (label MUTED, value FOCAL) fed at 1 Hz by a provider callable
  (set_telemetry) — the reactor never imports app modules — and the
  dynamic layer: radar sweep + trail, three orbit dots, dust motes, the
  spark overlay.

Atmosphere: a full-stage backdrop PhotoImage (worker-rendered, rebuilt
  only on size settle) carries a soft radial glow pool centered on the
  ring cluster; the base squares bake the SAME analytic pool into their
  ground so the images seam. In holo the backdrop IS avatar_bake.pool_shade
  of the integer distance from the cluster centre with the pool the bases
  actually carry (_pool_used) and no vignette, so the square's border and
  the stage agree to the byte (probe: scratchpad/holo/w2/A2/seam_probe.py,
  step 0/255 in holo). Classic keeps its corner vignette, its h/2.0
  centre and the current-width pool radius — and therefore its faint
  tile (1.93/255 measured) — because classic must render exactly as it
  did.
"""
from __future__ import annotations

import math
import os
import queue
import random
import threading
import time
import tkinter as tk

from jarvis import perf as _perf
from jarvis.events import (AudioLevel, BrainState, RecordingStarted,
                           RecordingStopped, SpeakingState, bus)
from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.avatar_bake import (BakeRunner, pool_ground, pool_shade,
                                   pool_shade_point)
from jarvis.ui.avatar_clock import (TIER_STEPS, AvatarClock, ConvCost,
                                    LateCounter, drain_budget, tier_order)
from jarvis.ui.widgets import (ellipsize, get_scale, measure, px,
                               ui_display, ui_mono)

log = get_logger("ui.reactor")

FPS_MS = 33                # legacy nominal period; the loop now runs on
                           # AV_TICK slot boundaries (nothing schedules on
                           # FPS_MS any more)
LEVELS = 8                 # pre-rendered brightness levels per state
SUPER = 2                  # supersampling factor for the base render
SUPER_DROP = 500           # above this render size, supersample at 1x
                           # (bounds the numpy cost on HiDPI)
MIN_SIZE, MAX_SIZE = 170, 480   # design units; scaled by S at runtime
MAX_RENDER = 640                # hard cap on the scaled base size
DECOR_MARGIN = 64          # design units kept clear around the base for
                           # the HUD ruler/sweep/orbit ring band
WF_BARS = 64
COMET_DOTS = 16
RIPPLE_POOL = 4
SWEEP_SPEED = 20.0         # radar sweep, degrees/second
SWEEP_TRAIL = 6            # trailing arcs behind the sweep edge
STAGE_CX = 0.37            # ring cluster x as a fraction of the stage
                           # width (asymmetric by design — the right flank
                           # carries the engine card); bounded so the card
                           # never sits on the ring (see _cluster_xy)
CORE_FRACS = (0.170, 0.135, 0.105, 0.078, 0.053)   # speaking core bands /R

# Atmospheric depth — the reactor light visibly FLOODS its region and
# bleeds into the transcript below
POOL_PEAK = 0.22           # radial glow pool: peak cyan blend at the core
POOL_RADIUS = 0.80         # pool radius as a fraction of the stage WIDTH
VIGNETTE = 0.20            # max darkening toward black in extreme corners
VIGNETTE_START = 0.82      # normalized corner distance where it begins
MOTES = 16                 # dust motes on the stage (two brightness tiers)
MOTE_BRIGHT = 5            # of which this many are the brighter tier
MOTE_EVERY = 1             # coords update every tick (60 fps): the
                           # 16-mote pass is ~0.1 ms, and a 10 fps drift
                           # next to a 60 fps sphere read as stutter

# Living particulate avatar. CENTERPIECE switches the base art: "avatar"
# is the film JARVIS presence; "reactor" restores the arc-reactor disc.
CENTERPIECE = "avatar"
AV_FRAMES = 600            # baked rotation frames — a seamless loop; 600
                           # over 10 s = 60 unique frames/s = exactly one
                           # 60 Hz refresh per frame. N % 24 == 0 (tier
                           # grids) and N / P == 60 are asserted by tests.
AV_PERIOD = 10.0           # seconds per loop at 1x (idle)
AV_TICK = AV_PERIOD / AV_FRAMES        # 16.667 ms — the reactor loop slot
AV_ELLIPSE = 0.46          # overlay-plane squash for the spark orbits:
                           # they ride the tilted EQUATOR plane, so the
                           # native overlay shares the sphere's geometry
AV_SPEED = {"idle": 1, "listening": 1, "thinking": 2, "speaking": 2}
                           # INTEGER frames per slot only — 1.5 would
                           # alternate 1/2-frame steps (judder)
AV_SPARKS = 8              # live spark overlay dots (coords-only)
AV_SPARK_EVERY = 1         # spark coords update every tick (on-grid)
AV_WORKERS = 4             # bake subprocesses

# Engine card (design units): the only text on the stage
CARD_W = 176
CARD_PAD = 12
CARD_ROW = 26
CARD_ROWS = (("HEAR", "asr"), ("SPEAK", "tts"), ("THINK", "llm"),
             ("DEVICE", "dev"),     # DEVICE, not GPU: the status bar's
                                    # 'GPU 39°C' is a different datum
             ("FAULT", "fault"))    # the fault lane (jarvis/faults.py): "--"
                                    # almost always, "2 TRAINERS" when it is
                                    # not. Values are drawn in one colour and
                                    # ellipsized to ~10 mono characters, so
                                    # the TOKEN is the whole signal -- the
                                    # sentence stays in the spoken alert.
CARD_MOOD = {"listening": "asr", "thinking": "llm", "speaking": "tts"}

# Holo stage (design units; theme.LOOK == "holo" only — classic never sees
# these). Film law for the frame: thin, open, notched; captions tiny.
FRAME_INSET = 10           # frame lines this far inside the stage edges
FRAME_NOTCH = 10           # sides stop this short of a corner; a 45°
                           # chamfer joins them (the notched corner)
FRAME_FOOT = 28            # open bottom: the side lines turn inward this
                           # far and stop (the seam dissolve owns the rest)
TOP_RULER_FRAC = 0.44      # top status ruler spans this fraction of the
                           # frame width from the left notch
RAIL_PITCH = 6             # right-hand rail: a tick every RAIL_PITCH,
                           # long every 5th (the film 5/1 rhythm)
RAIL_TICK = (3, 5)         # (short, long) tick lengths — the frame is
                           # FRAME_INSET from the stage edge; in holo the
                           # card's right edge sits CARD_RAIL_GAP inside
                           # the long tips (engine_card_x1) so the ticks
                           # read as a rail, not a comb glued to the card
CARD_RAIL_GAP = 4          # holo: design px between long-tick tips and the
                           # card's right edge (the 09-01 review saw them
                           # end 1 px short of the frame, a toothed edge)
FRAME_KEEP = 3             # holo: design px the dynamic ring band keeps
                           # clear of the frame's left rule
SWEEP_LEAD = 12            # radial length of the sweep's bright lead
RULER_CAPTION = "%d FRAMES  ·  %d HZ"   # the baked cycle; '%d F' read as °F
GAUGE_R = 13               # dial radius
GAUGE_START = 225.0        # visual degrees (clockwise from 12): the dial
GAUGE_SWEEP = 270.0        # opens at the bottom, 7:30 → 4:30
GAUGE_TICKS = 9            # ticks along the sweep (ends + every 33.75°)
GAUGE_LABEL = "LOAD"       # what the needle reads: 1-min loadavg / cpus
CARD_LIFT = 10             # holo card sits AT LEAST this much above the
                           # ring centre; _draw_card lifts it further when
                           # the dial's top tick would touch its bottom
                           # edge (it did at 520 px stage, 2026-09-01)
CARD_GAP = 8               # design px kept between card bottom and dial
LOADAVG_PATH = "/proc/loadavg"


def tracked(text: str) -> str:
    """Tracked caps the way the film HUD letters its captions — Tk cannot
    letter-space a font, so a plain space between glyphs does it
    ('HEAR' -> 'H E A R'); an inner space becomes three."""
    return " ".join(text.strip())


def read_load_fraction(path: str = LOADAVG_PATH, ncpu=None):
    """The gauge's REAL signal: 1-minute load average as a fraction of the
    CPU count, clamped to [0, 1]; None when unreadable (the needle then
    rests at the dial's start). A ~20 µs file read — cheap enough for the
    1 Hz telemetry tick it rides."""
    try:
        with open(path) as f:
            load1 = float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    n = ncpu or os.cpu_count() or 1
    return max(0.0, min(1.0, load1 / float(n)))


def gauge_angle(value) -> float:
    """Needle angle in visual degrees for a [0, 1] reading (None -> the
    dial's start)."""
    v = 0.0 if value is None else max(0.0, min(1.0, float(value)))
    return (GAUGE_START + GAUGE_SWEEP * v) % 360.0


def needle_xy(gx: float, gy: float, r: float, value) -> tuple:
    """Needle line coords for a dial centred on (gx, gy) with radius r:
    from 0.22 r (clear of the hub dot) to 0.86 r along gauge_angle."""
    m = math.radians(gauge_angle(value) - 90.0)
    return (gx + math.cos(m) * r * 0.22, gy + math.sin(m) * r * 0.22,
            gx + math.cos(m) * r * 0.86, gy + math.sin(m) * r * 0.86)


def engine_card_x1(w: int, look: str | None = None) -> int:
    """Right edge of the engine card on a stage `w` wide (pure). Classic:
    PAD from the edge, as always. Holo: CARD_RAIL_GAP inside the rail's
    long-tick tips, so the ticks stop clear of the card frame instead of
    1 design px short of it. _cluster_xy clamps the ring cluster against
    this same edge, so the 12 px clearance rule follows the card."""
    look = theme.LOOK if look is None else look
    if look == "holo":
        return w - px(FRAME_INSET) - px(RAIL_TICK[1]) - px(CARD_RAIL_GAP)
    return w - theme.PAD


def ruler_caption(frames: int = None, period: float = None) -> str:
    """The top-ruler caption: '600 FRAMES  ·  60 HZ' (pure). It used to be
    '600 F', which next to the strip's 'CPU 76°' read as a temperature."""
    frames = AV_FRAMES if frames is None else frames
    period = AV_PERIOD if period is None else period
    return RULER_CAPTION % (frames, round(frames / period))


def dynamic_r_limit(h: int, cx: int, r_min: float,
                    look: str | None = None) -> float:
    """Ceiling for the DYNAMIC ring radii — sweep lead, trail arcs, orbit
    dots (pure). Always h/2 - 4 design px so nothing clips the stage top
    or bottom, and in holo also FRAME_KEEP inside the frame's left rule.

    Why the second clamp: _cluster_xy pulls the cluster left until it
    clears the engine card, and at the shipped 918x520 stage that lands
    the centre at cx=252 with the frame line at x=20 — the sweep tip
    reached x=4, i.e. the blade crossed a 1px hairline once a revolution
    and read as a glitch, not as depth (09-01 review). Classic has no
    frame, so it keeps the old ceiling to the pixel. Never below r_min
    (the lead's inner radius) or the segment would invert on a narrow
    stage."""
    r_lim = h // 2 - px(4)
    if (theme.LOOK if look is None else look) != "holo":
        return r_lim
    return min(r_lim, max(r_min, cx - px(FRAME_INSET) - px(FRAME_KEEP)))


def sweep_radii(rr: float, r_lim: float, look: str | None = None) -> tuple:
    """(inner, outer) of the radar sweep's bright lead (pure). Classic:
    Rr+8 .. Rr+20, clamped outward only — byte for byte what it always
    was. Holo: when dynamic_r_limit pulls the outer radius in off the
    frame, the inner follows so the blade keeps its SWEEP_LEAD length —
    clamping the outer alone left a 5 design px stub, which is the other
    way to lose this fight."""
    hi = min(rr + px(20), r_lim)
    lo = rr + px(8)
    if (theme.LOOK if look is None else look) == "holo":
        lo = min(lo, hi - px(SWEEP_LEAD))
    return lo, hi


def _hex_rgb(color: str) -> tuple:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _blend(c1: tuple, c2: tuple, f: float) -> tuple:
    return tuple(int(a + (b - a) * f) for a, b in zip(c1, c2))


def _to_hex(rgb: tuple) -> str:
    return "#%02x%02x%02x" % rgb


def fmt_llm(name) -> str:
    """Engine-card THINK value: the Ollama model name with ONLY the
    `:latest` tag stripped (other tags such as `:32b` stay), uppercased.
    'llama3.2:latest' → 'LLAMA3.2'; 'qwen2.5:32b' → 'QWEN2.5:32B'."""
    name = (name or "").strip()
    if name.upper().startswith("LLM "):
        name = name[4:].strip()
    if not name:
        return "--"
    if name.lower().endswith(":latest"):
        name = name[:-len(":latest")]
    return name.upper()


class Reactor(tk.Canvas):
    """Borderless canvas stage; subscribes to bus events itself (the bus
    delivers on the Tk thread once attached, so handlers only set fields)."""

    def __init__(self, parent, bg=None, **kw):
        # theme.BG resolved HERE, not in the signature: a default argument
        # is evaluated at import, i.e. under whichever look theme booted
        # with, and main_window.create() selects the real look after that
        # (2026-09-01: classic stages were rendering on the holo BG)
        bg = bg or theme.BG
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, **kw)
        self._bg_rgb = _hex_rgb(bg)
        # -- live state fed by events (Tk thread) -----------------------
        self._recording = False
        self._thinking = False
        self._speaking = False
        self._level = 0.0            # smoothed AudioLevel.level
        self._waveform: list = []
        self._speak_amp = 0.0
        self._ripples: list = []     # spawn times (loop-relative)
        self._last_ripple = 0.0
        # -- render machinery -------------------------------------------
        self._clock = AvatarClock(AV_FRAMES, AV_PERIOD, AV_TICK)
        self._t0 = self._clock.t0
        self._late = LateCounter(self._clock)   # late-slot stats, 30 s lines
        self._min_size = px(MIN_SIZE)
        self._max_size = min(px(MAX_SIZE), MAX_RENDER)
        # current base render size (196 design units — matches the default
        # stage minus DECOR_MARGIN, so boot needs no immediate re-render)
        self._size = max(self._min_size, min(self._max_size, px(196)))
        self._photos: dict = {}      # color -> [PhotoImage x LEVELS]
        self._photo_gen = 0          # invalidates stale conversions
        self._applied_gen = -1       # first finished color swaps the set
        self._img_id = None
        self._shown_photo = None     # keep a ref so Tk doesn't drop it
        self._alive = True
        self._resize_job = None
        self._render_busy = False
        # -- overlay canvas items (created lazily, updated in place) ----
        self._items: dict = {}
        self._items_state = None     # which state's items are visible
        # -- HUD decor + telemetry --------------------------------------
        self._decor: dict = {}                # static/dynamic decor ids
        self._decor_key = None                # (w, h, size) guard
        self._decor_job = None
        self._telemetry_fn = None             # provider callable (main_window)
        self._telem_cache: dict = {}
        # Standby slows the whole rotation (see set_speed_scale). An int 1
        # keeps the default arithmetic bit-identical to what it was.
        self._speed_scale = 1
        self._card_mood = None
        self._rng = random.Random(int(time.monotonic() * 1000))
        # -- living-avatar centerpiece machinery ------------------------
        self._av_frames: list = []       # PhotoImage x AV_FRAMES (gaps
                                         # while the progressive bake fills)
        self._av_holders: list = []      # unmapped Labels pinning each
                                         # photo's Tk display instance
        self._av_have = 0
        self._av_gen_applied = -1
        self._tier_counts: dict = {}
        self._bake = None                # (gen, size, BakeRunner)
        self._bake_t0 = 0.0
        self._drain_starve = 0
        self._conv_cost = ConvCost()     # measured PhotoImage cost/frame
        self._spark_beat = 0
        # -- atmospheric backdrop (glow pool + vignette) + dust motes ---
        self._backdrop_id = None
        self._backdrop_photo = None
        self._bd_gen = 0
        self._bd_key = None
        self._pool_used = self._pool_params()   # pool the bases carry
        self._mote_beat = 0
        self._mote_t = 0.0

        self._subs = [
            (AudioLevel, self._on_audio),
            (RecordingStarted, self._on_rec_start),
            (RecordingStopped, self._on_rec_stop),
            (BrainState, self._on_brain),
            (SpeakingState, self._on_speaking),
        ]
        for etype, fn in self._subs:
            bus.subscribe(etype, fn)
        self.bind("<Destroy>", self._on_destroy, add=True)
        self.bind("<Configure>", self._on_configure, add=True)

        if CENTERPIECE == "avatar":
            self._start_bake(self._size, self._photo_gen, self._pool_used)
        else:
            threading.Thread(target=self._prerender_all,
                             args=(self._size, self._photo_gen,
                                   self._pool_used),
                             daemon=True).start()
        self.after(200, self._tick)
        self.after(1000, self._telem_tick)

    # ------------------------------------------------------- bus handlers
    def _on_audio(self, ev: AudioLevel):
        self._level = 0.65 * self._level + 0.35 * max(0.0, min(1.0, ev.level))
        if ev.waveform:
            self._waveform = list(ev.waveform)[:WF_BARS]

    def _on_rec_start(self, _ev):
        self._recording = True

    def _on_rec_stop(self, _ev):
        self._recording = False
        self._level = 0.0
        self._waveform = []

    def _on_brain(self, ev: BrainState):
        self._thinking = (ev.state == "thinking")

    def _on_speaking(self, ev: SpeakingState):
        # An amplitude-only tick still shapes the mouth -- that is the whole
        # point of it -- but it must not move the speaking flag.
        if not getattr(ev, "amplitude_only", False):
            self._speaking = ev.active
        self._speak_amp = max(0.0, min(1.0, ev.amplitude))
        if CENTERPIECE != "avatar":      # ripples are a reactor-disc overlay
            now = time.monotonic() - self._t0
            if ev.active and self._speak_amp > 0.2 \
                    and now - self._last_ripple > 0.3:
                self._ripples.append(now)
                self._last_ripple = now
        if not ev.active:
            self._speak_amp = 0.0

    def _on_destroy(self, _ev):
        self._alive = False
        self._stop_bake()
        for etype, fn in self._subs:
            bus.unsubscribe(etype, fn)

    # --------------------------------------------------------- boot hooks
    @property
    def cycle_live(self) -> bool:
        """True once the full AV_FRAMES cycle is installed (no bake
        traffic left on the Tk thread)."""
        return self._bake is None and self._av_have >= AV_FRAMES

    def when_cycle_live(self, fn, timeout_s: float = 40.0,
                        poll_ms: int = 250):
        """Run `fn()` on the Tk thread once the full cycle is live, or
        after `timeout_s` at the latest (a failed bake must not hold the
        app's model loading hostage)."""
        deadline = time.monotonic() + timeout_s

        def _check():
            if not self._alive:
                return
            if self.cycle_live or time.monotonic() >= deadline:
                if not self.cycle_live:
                    log.warning("cycle-live hook: timeout after %.0fs",
                                timeout_s)
                try:
                    fn()
                except Exception:
                    log.exception("cycle-live hook failed")
                return
            self.after(poll_ms, _check)

        self.after(poll_ms, _check)

    # ----------------------------------------------------------- geometry
    def _cluster_xy(self, w: int, h: int) -> tuple:
        """Ring cluster centre: STAGE_CX of the width, pulled left as far
        as needed so the engine card (fixed width, right-aligned on PAD)
        clears the outermost ring element by 12 design px."""
        cy = h // 2
        r_out = self._size * 0.5 + px(6) + px(22)
        card_x0 = engine_card_x1(w) - px(CARD_W)
        cx = min(round(w * STAGE_CX), int(card_x0 - px(12) - r_out))
        return cx, cy

    def ground_at(self, x: float, y: float) -> str:
        """The stage ground colour under stage pixel (x, y) as hex. Holo:
        the analytic pool (avatar_bake.pool_shade) the backdrop and the
        bases carry, evaluated at that point; classic: BG. For a flat
        widget that must sit on the stage (the alarm modal) -- Tk has no
        alpha, so matching the ground under it is the only way it is not a
        darker slab."""
        if theme.LOOK != "holo":
            return theme.BG
        w, h = self.winfo_width(), self.winfo_height()
        cx, cy = self._cluster_xy(w, h)
        return pool_shade_point(math.hypot(x - cx, y - cy), self._pool_used,
                                self._bg_rgb, _hex_rgb(theme.CYAN))

    # ----------------------------------------------------------- resizing
    def _on_configure(self, event):
        if self._img_id is not None:
            self.coords(self._img_id,
                        *self._cluster_xy(event.width, event.height))
        # Debounced base re-render at the new best size.
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except tk.TclError:
                pass
        self._resize_job = self.after(300, self._maybe_rescale)
        # Debounced HUD decor rebuild (size-key guarded inside).
        if self._decor_job is not None:
            try:
                self.after_cancel(self._decor_job)
            except tk.TclError:
                pass
        self._decor_job = self.after(200, self._draw_decor)

    def attach_toplevel(self):
        """Called by main_window once packed: track window height so the
        stage grows with the window (28%, clamped)."""
        top = self.winfo_toplevel()

        def _on_top_configure(event):
            if event.widget is not top:
                return
            want = max(px(260), min(px(520), int(event.height * 0.28)))
            if abs(int(self["height"]) - want) > px(8):
                self.configure(height=want)

        top.bind("<Configure>", _on_top_configure, add=True)

    def _maybe_rescale(self):
        self._resize_job = None
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        # DECOR_MARGIN keeps a clear band around the base for the HUD
        # ruler ring, radar sweep and satellite orbits; the horizontal
        # room is bounded by the off-center cluster position.
        avail = min(int(w * 2 * STAGE_CX), h)
        want = max(self._min_size,
                   min(self._max_size, avail - px(DECOR_MARGIN)))
        # The bases bake the stage glow pool into their ground; when the
        # pool radius drifts >15% from what they carry, the seam against
        # the full-stage backdrop would show — re-render then too.
        pool = self._pool_params()
        size_ok = abs(want - self._size) < px(24)
        pool_ok = abs(pool[1] - self._pool_used[1]) \
            <= 0.15 * max(1.0, self._pool_used[1])
        if size_ok and pool_ok:
            return
        if CENTERPIECE != "avatar" and self._render_busy:
            # the disc bake can run for seconds — a rescale request
            # arriving mid-bake must retry, not vanish
            self._resize_job = self.after(400, self._maybe_rescale)
            return
        self._size = want
        self._pool_used = pool
        self._photo_gen += 1
        log.info("reactor: rescaling bases to %dpx (pool r=%.0f)",
                 want, pool[1])
        if CENTERPIECE == "avatar":
            self._start_bake(want, self._photo_gen, pool)
        else:
            threading.Thread(target=self._prerender_all,
                             args=(want, self._photo_gen, pool),
                             daemon=True).start()

    # -------------------------------------------------------- pre-render
    def _pool_params(self) -> tuple:
        """(peak, radius_px) of the stage glow pool — shared by the
        full-stage backdrop and the base image grounds so they seam."""
        w = self.winfo_width()
        if w < px(60):
            w = px(520)          # pre-layout: default stage width
        return (POOL_PEAK, POOL_RADIUS * w)

    # ------------------------------------------------- avatar bake (v9)
    def _start_bake(self, size: int, gen: int, pool: tuple):
        """Launch the out-of-process bake for one generation. Any previous
        generation's workers are killed first; its frames are dropped when
        the first frame of the new generation lands."""
        self._stop_bake()
        sup = SUPER if size <= SUPER_DROP else 1
        runner = BakeRunner(size, sup, AV_FRAMES, tier_order(AV_FRAMES),
                            pool, self._bg_rgb, _hex_rgb(theme.CYAN),
                            workers=AV_WORKERS, look=theme.LOOK)
        runner.start()
        self._bake = (gen, size, runner)
        self._bake_t0 = time.monotonic()
        self._render_busy = True
        self._conv_cost.reset()          # a new size has new per-frame costs
        log.info("avatar: bake started (%s) — %d frames @ %dpx, gen %d",
                 runner.mode, AV_FRAMES, size, gen)

    def _stop_bake(self):
        if self._bake is None:
            return
        _gen, _size, runner = self._bake
        self._bake = None
        runner.stop(wait=False)                 # kill now …
        threading.Thread(target=runner.stop, daemon=True,
                         name="bake-reaper").start()   # … reap off-thread

    def _drain_bake(self, budget: int):
        """Tk thread: convert up to `budget` baked frames waiting on the
        runner's queue into PhotoImages and install them on the grid.
        Raw bytes are dropped after conversion."""
        if self._bake is None:
            return 0
        gen, size, runner = self._bake
        from PIL import Image, ImageTk
        done = 0
        while done < budget:
            try:
                k, buf = runner.queue.get(block=False)
            except queue.Empty:
                break
            done += 1
            if gen != self._photo_gen or not self._alive:
                continue
            # the generation switch (destroying 600 holders of the old
            # cycle) is a one-off, not a per-frame cost: keep it out of
            # the timed region or it pins the budget estimate
            self._begin_av_generation(gen)
            t_conv = time.perf_counter()
            try:
                img = Image.frombuffer("RGB", (size, size), buf,
                                       "raw", "RGB", 0, 1)
                photo = ImageTk.PhotoImage(img)
            except (tk.TclError, RuntimeError, ValueError):
                log.debug("avatar frame %d conversion failed", k,
                          exc_info=True)
                continue
            self._install_av_frame(gen, k, photo)
            # the per-frame cost on the Tk thread (PIL copy, Tk master,
            # pinned display instance) is what the next slot's drain
            # budget has to fit into its spare time
            self._conv_cost.add(time.perf_counter() - t_conv)
        return done

    def _begin_av_generation(self, gen: int) -> None:
        """The first frame of a new generation swaps in a fresh frame
        list (a resize re-bake replaces the cycle only as its frames
        land), drops the OLD generation's pinned holders — their photos
        are then unreferenced and Tk frees the masters — and resets the
        tier clock. Idempotent for the current generation."""
        if self._av_gen_applied == gen:
            return
        self._av_gen_applied = gen
        self._av_frames = [None] * AV_FRAMES
        self._av_have = 0
        self._tier_counts = {s: 0 for s in TIER_STEPS}
        self._clock.reset_tiers()
        for holder in self._av_holders:
            try:
                holder.destroy()
            except tk.TclError:
                pass
        self._av_holders = []
        self.after_idle(self._draw_decor)   # ruler tracks the new size

    def _install_av_frame(self, gen: int, k: int, photo):
        """One baked frame on the grid (see _begin_av_generation for the
        first frame of a generation)."""
        self._begin_av_generation(gen)
        if self._av_frames[k] is None:
            self._av_have += 1
            for step in TIER_STEPS:
                if k % step:
                    continue
                self._tier_counts[step] += 1
                if self._tier_counts[step] != AV_FRAMES // step:
                    continue
                self._clock.set_available_step(step)
                if step == 1:
                    self._render_busy = False
                    # frame store: Tk keeps a 32-bit master copy per photo
                    # AND a 24-bit client-side XImage per pinned display
                    # instance (measured 0.59 + 0.45 MB/frame at 392 px on
                    # 09-01; the X pixmaps live server-side on top)
                    size = photo.width()
                    log.info("avatar: full %d-frame cycle live (%.1fs) — "
                             "frame store %d x %dpx, ~%.0f MB tk masters "
                             "+ ~%.0f MB pinned instances",
                             AV_FRAMES, time.monotonic() - self._bake_t0,
                             AV_FRAMES, size,
                             AV_FRAMES * size * size * 4 / 1048576.0,
                             AV_FRAMES * size * size * 3 / 1048576.0)
                    _perf.on_full_cycle(self)
                    # the workers have streamed every frame: drop the
                    # runner so the loop stops polling its queue
                    self.after_idle(self._stop_bake)
                else:
                    log.info("avatar: coarse %d-frame cycle live (%.1fs)",
                             AV_FRAMES // step,
                             time.monotonic() - self._bake_t0)
        self._av_frames[k] = photo
        # pin the photo's display instance with a never-mapped Label: the
        # cycling canvas item would otherwise drop each photo's instance
        # on swap-away and Tk would redo the XImage conversion every loop
        self._av_holders.append(tk.Label(self, image=photo))

    # ------------------------------------------ reactor-disc pre-render
    def _prerender_all(self, size: int, gen: int, pool: tuple):
        """CENTERPIECE="reactor": build 8 brightness bases per state color
        on a worker thread. The CURRENT state's color renders FIRST and is
        handed to the Tk thread immediately; the remaining colors continue
        in the background. A stale generation aborts quietly."""
        self._render_busy = True
        try:
            current = self._base_color(self.state())
            order = []
            for state in ("idle", "listening", "thinking", "speaking"):
                color = self._base_color(state)
                if color not in order:
                    order.append(color)
            if current in order:
                order.remove(current)
            order.insert(0, current)

            sup = SUPER if size <= SUPER_DROP else 1
            ground = pool_ground(size, sup, pool, self._bg_rgb,
                                 _hex_rgb(theme.CYAN))
            for color in order:
                t_start = time.monotonic()
                frames = []
                for lv in range(LEVELS):
                    if gen != self._photo_gen or not self._alive:
                        return
                    frames.append(self._build_base(
                        _hex_rgb(color), lv / LEVELS, size, ground))
                    time.sleep(0.01)   # yield during boot
                if gen != self._photo_gen or not self._alive:
                    return
                log.info("reactor: pre-rendered %s @ %dpx in %.2fs%s",
                         color, size, time.monotonic() - t_start,
                         " (current state)" if color == current else "")
                try:
                    self.after(0, lambda c=color, fr=frames:
                               self._install_color(gen, c, fr))
                except (RuntimeError, tk.TclError):
                    return
        except Exception:
            log.exception("reactor pre-render failed")
        finally:
            self._render_busy = False

    def _install_color(self, gen: int, color: str, frames, photos=None,
                       lv: int = 0):
        """Incrementally convert one color's PIL bases to PhotoImages on
        the Tk thread (2 per idle slot keeps boot and resize smooth)."""
        if gen != self._photo_gen or not self._alive:
            return
        from PIL import ImageTk
        if photos is None:
            photos = [None] * LEVELS
        done = 0
        while lv < LEVELS and done < 2:
            try:
                photos[lv] = ImageTk.PhotoImage(frames[lv])
            except (tk.TclError, RuntimeError):
                return
            lv += 1
            done += 1
        if lv < LEVELS:
            self.after(10, lambda: self._install_color(
                gen, color, frames, photos, lv))
            return
        if self._applied_gen != gen:
            self._applied_gen = gen
            self._photos = {}
            self._items_state = None
            self.after_idle(self._draw_decor)   # ruler tracks the new size
        self._photos[color] = photos

    def _build_base(self, accent: tuple, amp: float, size: int, ground):
        """One base image: the film reactor anatomy baked over the stage
        glow-pool ground, rendered at 2x and downscaled. Painter's order,
        outside → in: casing bezel disc (to 1.0R) with rim catch-lights,
        thin connector ring, the banded cyan glow annulus (filled to
        0.80R), ten dark rounded-trapezoid coil wedges 0.50–0.80R at 36°
        pitch OVER the annulus, tight core bloom, the 36-tick fine bezel
        0.18–0.25R, and the white-hot banded core disc to 0.15R. `amp`
        drives the annulus/core luminosity (the 8-level breath ramp)."""
        import numpy as np
        from PIL import Image, ImageDraw

        sup = SUPER if size <= SUPER_DROP else 1
        S2 = size * sup
        cx = cy = S2 // 2
        R = S2 * 0.48

        frame = ground.copy()

        # rim spill: soft bloom escaping past the casing edge (the region
        # under the anatomy is overpainted below, so only the spill shows)
        glow_r = R * 1.04
        y_c, x_c = np.ogrid[-cy:S2 - cy, -cx:S2 - cx]
        dist_sq = x_c * x_c + y_c * y_c
        glow_mask = dist_sq < glow_r * glow_r
        if glow_mask.any():
            dist = np.sqrt(dist_sq[glow_mask].astype(np.float32))
            f = np.clip((glow_r - dist) / (glow_r - R), 0.0, 1.0) ** 1.5
            intensity = (0.10 + amp * 0.12) * f
            for ch, tgt in enumerate(accent):
                cur = frame[glow_mask, ch].astype(np.float32)
                frame[glow_mask, ch] = np.clip(
                    cur + (tgt - cur) * intensity, 0, 255
                ).astype(np.uint8)

        img = Image.fromarray(frame, "RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        white = (255, 255, 255)
        bg = self._bg_rgb

        def circle(rf, fill=None, outline=None, width=1):
            r = R * rf
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=fill, outline=outline, width=width)

        def tone(w_f, lum):
            """BG → (accent whitened by w_f) at lum — pre-blended solid."""
            return _blend(bg, _blend(accent, white, w_f), min(1.0, lum))

        # Film law: the reactor's own light never dims with UI state — the
        # annulus/core blaze in fixed JARVIS cyan; only the instruments
        # around the casing take the state accent.
        blaze = _hex_rgb(theme.CYAN)

        def tone_blaze(w_f, lum):
            return _blend(bg, _blend(blaze, white, w_f), min(1.0, lum))

        casing = _blend(bg, (0, 0, 0), 0.45)       # coil metal, near-ground
        bezel = _blend(_blend(bg, (0, 0, 0), 0.30), accent, 0.05)
        rim_hi = _blend(bg, white, 0.16)

        # casing bezel to 1.0R with machined catch-light rings + ten rim
        # notch tabs between the coil angles (the film casing detail)
        circle(1.0, fill=bezel + (255,))
        circle(0.985, outline=rim_hi + (140,), width=max(1, sup))
        circle(0.86, outline=_blend(bg, white, 0.10) + (120,),
               width=max(1, sup))
        for k in range(10):
            a = math.radians(k * 36.0 + 18.0 - 90)
            draw.line([cx + math.cos(a) * R * 0.875,
                       cy + math.sin(a) * R * 0.875,
                       cx + math.cos(a) * R * 0.965,
                       cy + math.sin(a) * R * 0.965],
                      fill=rim_hi + (110,), width=max(2, sup * 2))
        # thin connector ring between bezel and coil band
        circle(0.835, outline=tone_blaze(0.35, 0.42 + 0.40 * amp) + (255,),
               width=max(1, sup))

        # THE glow annulus: banded radial gradient filled to 0.80R —
        # brightness peaks near 0.42R. Chest-RT levels: blazing white-cyan
        # with a HIGH floor — the reactor "never goes dark, it only swells".
        bands = 26
        for i in range(bands):
            rf = 0.80 - (i / (bands - 1)) * (0.80 - 0.24)
            p = 1.0 - min(1.0, abs(rf - 0.40) / 0.46) ** 2.0
            lum = (0.58 + 0.42 * amp) * (0.48 + 0.52 * p)
            w_f = (0.26 + 0.52 * amp) * p ** 1.1
            circle(rf, fill=tone_blaze(w_f, lum) + (255,))

        # ten dark rounded-trapezoid coil wedges at 36° pitch OVER the
        # annulus (0.50–0.80R) — these NEVER rotate; light escapes only
        # through the ten gaps
        coil_hi = _blend(casing, white, 0.10)
        for k in range(10):
            a_mid = k * 36.0
            pts = []
            for j in range(7):          # outer arc, corner radii eased in
                a = math.radians(a_mid - 12.0 + 24.0 * j / 6 - 90)
                ro = R * (0.785 if j in (0, 6) else 0.80)
                pts.append((cx + math.cos(a) * ro, cy + math.sin(a) * ro))
            for j in range(7):          # inner arc back (narrower: taper)
                a = math.radians(a_mid + 10.0 - 20.0 * j / 6 - 90)
                ri = R * (0.525 if j in (0, 6) else 0.51)
                pts.append((cx + math.cos(a) * ri, cy + math.sin(a) * ri))
            draw.polygon(pts, fill=casing + (255,))
            for s_off in (-6.5, -3.25, 0.0, 3.25, 6.5):   # winding striae
                a = math.radians(a_mid + s_off - 90)
                draw.line([cx + math.cos(a) * R * 0.54,
                           cy + math.sin(a) * R * 0.54,
                           cx + math.cos(a) * R * 0.77,
                           cy + math.sin(a) * R * 0.77],
                          fill=coil_hi + (60,), width=max(1, sup))

        # tight core bloom bridging the core into the annulus
        for rf, aa in ((0.24, 60), (0.21, 95), (0.18, 135)):
            circle(rf, fill=_blend(blaze, white, 0.35 + 0.30 * amp)
                   + (int(aa * (0.55 + 0.45 * amp)),))
        # fine tick bezel: 36 thin radial teeth 0.18–0.25R
        tick_c = _blend(casing, accent, 0.15)
        for i in range(36):
            a = math.radians(i * 10 - 90)
            draw.line([cx + math.cos(a) * R * 0.175,
                       cy + math.sin(a) * R * 0.175,
                       cx + math.cos(a) * R * 0.255,
                       cy + math.sin(a) * R * 0.255],
                      fill=tick_c + (215,), width=max(1, round(sup * 1.5)))
        # white-hot banded core disc (#fff center)
        for rf, w_f in ((0.155, 0.42), (0.130, 0.62), (0.105, 0.80),
                        (0.080, 0.92), (0.055, 1.0)):
            circle(rf, fill=_blend(accent, white,
                                   min(1.0, w_f * (0.80 + 0.20 * amp)))
                   + (255,))

        if sup == 1:
            return img
        return img.resize((size, size), Image.LANCZOS)

    # ------------------------------------------------------ state machine
    def state(self) -> str:
        if self._speaking:
            return "speaking"
        if self._recording:
            return "listening"
        if self._thinking:
            return "thinking"
        return "idle"

    @staticmethod
    def _base_color(state: str) -> str:
        """Base-art color for a state. SPEAKING keeps the cyan structure
        (green is not in the film budget; the white-hot signal comes from
        the overlay core disc + primary scan arc), so it shares the
        listening bases — one fewer color set to pre-render."""
        return theme.STATE_COLORS["listening" if state == "speaking"
                                  else state]

    # amplitude per state → brightness level of the pre-rendered base
    def _amp(self, state: str, t: float) -> float:
        if state == "listening":
            return 0.30 + 0.70 * self._level
        if state == "thinking":
            return 0.45 + 0.12 * math.sin(t * 4.5)
        if state == "speaking":
            return 0.30 + 0.70 * self._speak_amp
        # film law: the annulus BREATHES — a calm ~1Hz sinusoid riding the
        # 8-level base ramp (the coil wedges themselves never move)
        return 0.45 + 0.20 * math.sin(t * 2 * math.pi)

    # canvas has no alpha: pre-blend accent toward the background
    def _alpha(self, accent: tuple, a: int) -> str:
        return _to_hex(_blend(self._bg_rgb, accent, max(0, min(255, a)) / 255))

    # ------------------------------------------------------- frame loop
    def _tick(self):
        """One slot of the frame loop. The slot k is derived from the
        clock (never counted), everything renders at the slot's grid time
        t_k, the canvas is flushed BEFORE any other pending timer can run,
        spare time converts as many baked frames as the measured
        conversion cost says fit (drain_budget), and the next tick is
        scheduled for the next boundary AFTER rendering so render cost
        never shifts the grid."""
        if not self._alive:
            return
        clock = self._clock
        try:
            now = time.monotonic()
            k = clock.slot(now)
            late_s, skipped = self._late.observe(now, k)
            if late_s > self._late.late_s and _perf.detail_enabled() \
                    and self._late.detail_ok():
                log.info("avatar: late slot k=%d +%.1f ms (skipped %d) "
                         "ages: %s", k, late_s * 1000.0, skipped,
                         _perf.ages(now))
            line = self._late.report(now)
            if line:
                log.info(line)
            t_k = clock.slot_time(k)
            self._render(t_k)
            self.update_idletasks()
            if self._bake is not None:
                self._conv_cost.tick()      # samples expire by slot age
                spare = clock.slot_time(k + 1) - time.monotonic()
                budget = drain_budget(spare, self._conv_cost.estimate)
                if budget > 0:
                    self._drain_starve = 0
                    self._drain_bake(budget)
                else:
                    # never let a permanently short slot starve the bake
                    self._drain_starve += 1
                    if self._drain_starve >= 10:
                        self._drain_starve = 0
                        self._drain_bake(1)
        except tk.TclError:
            self._alive = False
            return
        except Exception:
            log.exception("reactor tick failed")
        if self._alive and self.winfo_exists():
            self.after(clock.next_delay_ms(time.monotonic()), self._tick)

    def _render(self, t_abs: float):
        state = self.state()
        color = self._base_color(state)
        t = t_abs - self._t0
        if CENTERPIECE == "avatar":
            frames = self._av_frames
            if not frames:
                return
            clock = self._clock
            clock.set_speed(AV_SPEED.get(state, 1) * self._speed_scale,
                            t_abs)
            idx = clock.display_index(t_abs)
            if idx is None:
                # no tier complete yet: show whatever of the coarsest grid
                # has landed, else hold the last frame
                idx = clock.index_at(t_abs)
                idx -= idx % TIER_STEPS[0]
            photo = frames[idx] or self._shown_photo
            if photo is None:
                return
        else:
            photos = self._photos.get(color)
            if not photos or photos[-1] is None:
                for c, ph in self._photos.items():   # any finished color?
                    if ph and ph[-1] is not None:
                        photos, color = ph, c
                        break
                else:
                    return
            amp = max(0.0, min(0.999, self._amp(state, t)))
            photo = photos[min(LEVELS - 1, int(amp * LEVELS))]

        w = max(self.winfo_width(), self._size)
        h = max(self.winfo_height(), self._size)
        cx, cy = self._cluster_xy(w, h)
        if self._img_id is None:
            self._img_id = self.create_image(cx, cy, anchor="center",
                                             image=photo)
            # HUD decor layers above the opaque base square (decor stays
            # outside the reactor art radius, so only ground pixels
            # overlap); the atmosphere backdrop sits below the base.
            self._fix_layers()
        elif photo is not self._shown_photo:
            self.itemconfigure(self._img_id, image=photo)
        self._shown_photo = photo

        self._ensure_items(state)
        self._update_overlays(state, _hex_rgb(color), t, cx, cy)
        self._update_decor(t)

    # ------------------------------------------------- overlay management
    def _ensure_items(self, state: str):
        """Create/show only the items the current state needs."""
        if self._items_state == state:
            return
        # first build
        if not self._items:
            it = self._items
            lw = max(2, px(2))
            # Volumetric scan arcs: each drawn 3× at the same coords —
            # heavy dim halo, mid stroke, 1px white-hot core — so they
            # read as light, not vector lines (the IM3 depth signature).
            arc_ws = (px(3), px(2), max(1, px(1)))
            # each trio carries a tag so one itemconfigure(start=) per
            # slot rotates all three strokes
            it["rot1"] = [self.create_arc(0, 0, 0, 0, style="arc", width=wd,
                                          start=0, extent=46, outline="",
                                          tags=("rot1",))
                          for wd in arc_ws]
            it["rot2"] = [self.create_arc(0, 0, 0, 0, style="arc", width=wd,
                                          start=180, extent=30, outline="",
                                          tags=("rot2",))
                          for wd in arc_ws]
            # banded white-hot core disc (reactor-disc SPEAKING only):
            # concentric ovals per the film core ramp, cyan-most band
            # created first (bottom). Never shown in avatar mode — the
            # baked frames carry the 3D knot glow and nothing 2D may be
            # painted inside the sphere's disc.
            it["core"] = [self.create_oval(0, 0, 0, 0, outline="", fill=c,
                                           state="hidden")
                          for c in reversed(theme.CORE_BANDS)]
            it["ring"] = self.create_oval(0, 0, 0, 0, width=lw, outline="")
            it["wf"] = [self.create_line(0, 0, 0, 0, width=lw, fill="")
                        for _ in range(WF_BARS)]
            it["comet"] = [self.create_oval(0, 0, 0, 0, outline="", fill="")
                           for _ in range(COMET_DOTS)]
            it["ripple"] = [self.create_oval(0, 0, 0, 0, width=lw, outline="")
                            for _ in range(RIPPLE_POOL)]
            # avatar extra: the live spark overlay (mote-style coords-only
            # dots riding the cloud's equator plane)
            sparks, params = [], []
            for i in range(AV_SPARKS):
                is_hot = i % 5 == 0
                sz = max(1, px(2 if is_hot else 1))
                sid = self.create_oval(
                    0, 0, 0, 0, outline="", state="hidden",
                    fill=theme.CORE_BANDS[1 if is_hot else 2])
                sparks.append(sid)
                params.append((sid,
                               self._rng.uniform(0.24, 0.80),   # radius /R
                               self._rng.uniform(12.0, 40.0)
                               * self._rng.choice((-1.0, 1.0)),  # deg/s
                               self._rng.uniform(0.0, 360.0),    # phase
                               self._rng.uniform(0.02, 0.06),    # wander /R
                               self._rng.uniform(0.3, 0.9),      # rad/s
                               sz))
            it["sparks"] = sparks
            it["spark_p"] = params
        it = self._items
        avatar = CENTERPIECE == "avatar"

        def show(ids, on):
            for i in (ids if isinstance(ids, list) else [ids]):
                self.itemconfigure(i, state="normal" if on else "hidden")
        show([*it["rot1"], *it["rot2"]], True)
        show(it["sparks"], avatar)
        # Avatar mode: NOTHING 2D inside the sphere's disc in any state —
        # the core disc, listening ring + waveform, thinking comet and
        # speaking ripples all live inside R and stay hidden; state shows
        # through the outer gimbal arcs' palette, AV_SPEED and the sparks.
        show(it["core"], not avatar and state == "speaking")
        show(it["ring"], not avatar and state == "listening")
        show(it["wf"], not avatar and state == "listening")
        show(it["comet"], not avatar and state == "thinking")
        show(it["ripple"], not avatar and state == "speaking")
        # Arc palette per state — structure stays cyan; SPEAKING drives
        # the primary arc white-hot (color is state, hue stays inside the
        # budget: cyan structure / white focal). Set once per transition.
        if state == "speaking":
            c1 = (theme.RAMP40, theme.BRIGHT, theme.CORE_BANDS[0])
        else:
            c1 = (theme.RAMP20, theme.RAMP40, theme.CORE_BANDS[1])
        c2 = (theme.RAMP20, theme.RAMP33, theme.CORE_BANDS[3])
        for aid, col in zip(it["rot1"], c1):
            self.itemconfigure(aid, outline=col)
        for aid, col in zip(it["rot2"], c2):
            self.itemconfigure(aid, outline=col)
        self._items_state = state
        self._set_card_mood(state)

    def _set_card_mood(self, state: str, force: bool = False):
        """Engine-card mood, no new text: the active row's LABEL turns
        FOCAL (HEAR while listening, THINK while thinking, SPEAK while
        speaking) — one itemconfigure per change."""
        labels = self._decor.get("card_lbl")
        if not labels:
            return
        active = CARD_MOOD.get(state)
        if self._card_mood == active and not force:
            return
        self._card_mood = active
        for key, iid in labels.items():
            self.itemconfigure(
                iid, fill=theme.FOCAL if key == active else theme.MUTED)

    def _update_overlays(self, state, accent, t, cx, cy):
        it = self._items
        R = self._size * 0.48

        period = 4.5 if state == "idle" else 3.0
        rot = -((t / period * 360.0) % 360.0)   # canvas angles are CCW
        # gimbal scan arcs orbit OUTSIDE the casing (the art fills the
        # base to 1.0R — instrument rings never cross it)
        r = self._size * 0.5 + px(3)
        box = (cx - r, cy - r, cx + r, cy + r)
        if box != getattr(self, "_rot_box", None):
            self._rot_box = box
            for aid in (*it["rot1"], *it["rot2"]):
                self.coords(aid, *box)
        # counter-rotation at −0.7× (film gimbal law): the trios rotate
        # together — one tagged itemconfigure(start=…) per trio
        rot2 = ((t / period * 360.0 * 0.7) % 360.0) + 180.0
        self.itemconfigure("rot1", start=rot)
        self.itemconfigure("rot2", start=rot2)

        if CENTERPIECE == "avatar":
            # sparks ride the slot time, so their motion sits on the same
            # grid as the frame swaps; no other overlay touches the disc
            self._spark_beat = (self._spark_beat + 1) % AV_SPARK_EVERY
            if self._spark_beat == 0:
                for sid, r0, spd, ph0, wa, ws, sz in it["spark_p"]:
                    a = math.radians(ph0 + t * spd)
                    sr = R * (r0 + wa * math.sin(t * ws + ph0))
                    sx = cx + math.cos(a) * sr
                    sy = cy + math.sin(a) * sr * AV_ELLIPSE
                    self.coords(sid, sx - sz, sy - sz, sx + sz, sy + sz)
            return

        if state == "listening":
            lr = R * (0.62 + 0.30 * self._level)
            self.coords(it["ring"], cx - lr, cy - lr, cx + lr, cy + lr)
            self.itemconfigure(
                it["ring"],
                outline=self._alpha(accent, 70 + int(120 * self._level)))
            wf = self._waveform
            n = len(wf)
            r0 = R * 0.30
            for i, line_id in enumerate(it["wf"]):
                v = max(0.0, min(1.0, wf[i % n])) if n else 0.0
                ang = math.radians(-90 + i * (360 / WF_BARS))
                r1 = r0 + px(3) + v * R * 0.26
                self.coords(line_id,
                            cx + math.cos(ang) * r0, cy + math.sin(ang) * r0,
                            cx + math.cos(ang) * r1, cy + math.sin(ang) * r1)
                self.itemconfigure(
                    line_id, fill=self._alpha(accent, 60 + int(190 * v)))

        elif state == "thinking":
            head = (t * 260.0) % 360.0
            cr = R * 0.70
            for k, dot_id in enumerate(it["comet"]):
                ang = math.radians(head - k * 7)
                dx = cx + math.cos(ang) * cr
                dy = cy + math.sin(ang) * cr
                sz = max(1.2, 4.0 - k * 0.18) * get_scale()
                self.coords(dot_id, dx - sz, dy - sz, dx + sz, dy + sz)
                self.itemconfigure(
                    dot_id,
                    fill=self._alpha(accent, max(0, int(230 * (1 - k / COMET_DOTS)))))

        elif state == "speaking":
            # banded white-hot core over the baked cyan one, breathing
            # slightly with the speech amplitude (coords-only, 5 calls;
            # reactor-disc mode only — avatar mode returned above)
            scale_f = 1.0 + 0.20 * self._speak_amp
            for oid, bf in zip(it["core"], CORE_FRACS):
                br_ = R * bf * scale_f
                self.coords(oid, cx - br_, cy - br_, cx + br_, cy + br_)
            self._ripples = [b for b in self._ripples if t - b < 1.2]
            for k, ring_id in enumerate(it["ripple"]):
                if k < len(self._ripples):
                    f = (t - self._ripples[k]) / 1.2
                    rr = R * (0.16 + 0.78 * f)
                    self.coords(ring_id, cx - rr, cy - rr, cx + rr, cy + rr)
                    self.itemconfigure(
                        ring_id, state="normal",
                        outline=self._alpha(accent, max(0, int(150 * (1 - f)))))
                else:
                    self.itemconfigure(ring_id, state="hidden")

    # ------------------------------------------------- atmosphere backdrop
    def _fix_layers(self):
        """Stacking, bottom → top: atmosphere backdrop, reactor base
        photo, decor + state overlays (tag_lower pushes below ALL, so the
        last lowered item lands at the very bottom)."""
        if self._img_id is not None:
            self.tag_lower(self._img_id)
        if self._backdrop_id is not None:
            self.tag_lower(self._backdrop_id)

    def _rebuild_backdrop(self, w: int, h: int):
        """Kick a worker render of the full-stage ambience image. Runs on
        size settle only; stale generations abort. Classic keys on (w, h)
        and paints the CURRENT width's pool, as it always did. Holo keys
        on (w, h, cluster centre, the pool the bases CARRY): a re-bake at
        a new pool radius, or a centre shift from a size change, must move
        the backdrop with it or the seam re-opens (_begin_av_generation
        reaches here through _draw_decor when the new generation's first
        frame lands, so old frames keep the old ground until then)."""
        centre = self._cluster_xy(w, h)
        flat = theme.LOOK == "holo"
        pool = self._pool_used if flat else self._pool_params()
        key = (w, h, centre, pool) if flat else (w, h)
        if self._bd_key == key:
            return
        self._bd_key = key
        self._bd_gen += 1
        threading.Thread(target=self._render_backdrop,
                         args=(w, h, self._bd_gen, pool, centre, flat),
                         daemon=True, name="backdrop-render").start()

    def _render_backdrop(self, w: int, h: int, gen: int, pool: tuple,
                         centre: tuple, flat: bool = False):
        """Worker. `flat` (holo): the stage ground IS the bases' ground —
        avatar_bake.pool_shade of the integer pixel offset from the anchor
        pixel the canvas places the square on, nothing else — so the
        square's pure-ground border matches the stage to the byte
        wherever it lands. Classic: the same soft radial pool with a
        float centre (h/2.0) + a darker vignette in the extreme corners,
        untouched."""
        try:
            import numpy as np
            from PIL import Image

            cyan = _hex_rgb(theme.CYAN)
            if flat:
                ccx, ccy = int(centre[0]), int(centre[1])
                y, x = np.ogrid[0:h, 0:w]
                dx = x - ccx
                dy = y - ccy
                # same ops as pool_ground(size, 1, ...): int64 squares →
                # float32 → sqrt; the float32 / 1 there is exact
                d = np.sqrt((dx * dx + dy * dy).astype(np.float32))
                img = pool_shade(d, pool, self._bg_rgb, cyan)
            else:
                peak, rp = pool
                ccx, ccy = float(centre[0]), h / 2.0
                y, x = np.ogrid[0:h, 0:w]
                dx = (x - ccx).astype(np.float32)
                dy = (y - ccy).astype(np.float32)
                d = np.sqrt(dx * dx + dy * dy)
                pf = peak * np.clip(1.0 - d / rp, 0.0, 1.0) ** 2
                nx = (x - w / 2.0).astype(np.float32) / max(w / 2.0, 1.0)
                ny = dy / max(ccy, 1.0)
                dn = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
                dark = VIGNETTE * np.clip(
                    (dn - VIGNETTE_START) / (1.0 - VIGNETTE_START),
                    0.0, 1.0) ** 1.5
                img = np.empty((h, w, 3), dtype=np.uint8)
                for ch in range(3):
                    base = self._bg_rgb[ch]
                    v = base + (cyan[ch] - base) * pf
                    img[..., ch] = (v * (1.0 - dark)).astype(np.uint8)
            pil = Image.fromarray(img, "RGB")
        except Exception:
            log.exception("backdrop render failed")
            return
        try:
            self.after(0, lambda: self._install_backdrop(gen, pil))
        except (RuntimeError, tk.TclError):
            pass

    def _install_backdrop(self, gen: int, pil):
        if gen != self._bd_gen or not self._alive:
            return
        from PIL import ImageTk
        try:
            photo = ImageTk.PhotoImage(pil)
        except (tk.TclError, RuntimeError):
            return
        w, h = pil.size
        if self._backdrop_id is None:
            self._backdrop_id = self.create_image(w // 2, h // 2,
                                                  anchor="center",
                                                  image=photo)
        else:
            self.itemconfigure(self._backdrop_id, image=photo)
            self.coords(self._backdrop_id, w // 2, h // 2)
        self._backdrop_photo = photo       # keep a ref for Tk
        self._fix_layers()

    # ------------------------------------------------------- HUD decor
    # Angle conventions: "visual" degrees run clockwise from 12 o'clock.
    # Point at visual v: (cx + cos(rad(v-90))*r, cy + sin(rad(v-90))*r).
    # Tk arcs measure CCW from 3 o'clock with y up, so an arc centered on
    # visual v uses start = 90 - v - extent/2.
    def _draw_decor(self):
        """(Re)build the static HUD scene + the engine card. Runs on size
        settle only (debounced, keyed on (w, h, base size)). Invariant:
        every DYNAMIC element (halo dashes, sweep, orbits, arcs) stays
        inside r_lim of the cluster centre — h/2 - 4 design px, so nothing
        clips at the stage top/bottom edge, and in holo also clear of the
        frame's left rule (dynamic_r_limit)."""
        self._decor_job = None
        if not self._alive:
            return
        w, h = self.winfo_width(), self.winfo_height()
        if w < px(140) or h < px(120):
            return
        # holo keys on the pool the bases carry too: a pool-only re-bake
        # (same size, width drifted >15%) has to reach _rebuild_backdrop
        key = (w, h, self._size)
        if theme.LOOK == "holo":
            key += (self._pool_used,)
        if self._decor_key == key:
            return
        self._decor_key = key
        self._rebuild_backdrop(w, h)       # keyed internally
        self.delete("decor")
        d = self._decor = {}
        cx, cy = self._cluster_xy(w, h)
        d["c"] = (cx, cy)
        thin = 1
        lw = max(1, px(1))
        Rr = self._size * 0.5 + px(6)          # degree ruler radius
        # dynamic-radius ceiling: the stage in classic, the stage AND the
        # holo frame's left rule in holo (dynamic_r_limit)
        r_lim = dynamic_r_limit(h, cx, Rr + px(8))

        # seam dissolve: 1px full-width lines stepping the stage ground
        # into the transcript's lit top tone (every 2nd row keeps the
        # ground, so the band reads as scanline texture, not a stripe) —
        # created first so all other decor stacks above
        n_seam = len(theme.SEAM_STEPS)
        for i, seam_c in enumerate(theme.SEAM_STEPS):
            y = h - 2 * (n_seam - i) + 1
            self.create_line(0, y, w, y, fill=seam_c, width=thin,
                             tags=("decor",))

        if theme.LOOK == "holo":
            self._draw_holo_decor(w, h, cx, cy)
        else:
            self._draw_classic_decor(w, h, cx, cy, Rr, lw, thin)

        # radar sweep: leading radial edge (Rr+8 .. Rr+20, slid inward as
        # one piece when holo's frame clamps r_lim — see sweep_radii) +
        # trailing arcs at Rr+12 (dynamic). The bright lead gets the
        # two-stroke treatment: a wide dim underlay created first (below).
        d["sw_r"] = sweep_radii(Rr, r_lim)
        rs = min(Rr + px(12), r_lim)
        d["rs"] = rs
        d["sweep_u"] = self.create_line(
            0, 0, 0, 0, fill=theme.RAMP33, width=px(3), tags=("decor",))
        d["sweep"] = self.create_line(
            0, 0, 0, 0, fill=theme.EDGE, width=lw, tags=("decor",))
        trail_cols = (theme.HOLO, theme.HOLO, theme.HOLO_DIM,
                      theme.HOLO_DIM, theme.SCAN, theme.SCAN)[:SWEEP_TRAIL]
        d["trail"] = [
            self.create_arc(cx - rs, cy - rs, cx + rs, cy + rs, style="arc",
                            start=0, extent=7, outline=c, width=lw,
                            tags=("decor",))
            for c in trail_cols]

        # satellite dots on separate orbits (dynamic): Rr+2 / +10 / +18
        orbits = []
        for off, speed, phase, color, sz in (
                (2, 14.0, 0.0, theme.EDGE, 2.4),
                (10, -9.0, 130.0, theme.HOLO, 1.9),
                (18, 23.0, 255.0, theme.EDGE, 1.5)):
            szp = max(2, round(sz * get_scale()))
            r = min(Rr + px(off), r_lim - szp)
            oid = self.create_oval(0, 0, 0, 0, fill=color, outline="",
                                   tags=("decor",))
            orbits.append((oid, r, speed, phase, szp))
        d["orbits"] = orbits

        # dust motes: two brightness tiers drifting slowly upward with
        # slight lateral wander, wrapping at the stage edges (coords-only
        # updates every MOTE_EVERY-th tick — see _update_decor)
        motes = []
        for i in range(MOTES):
            bright = i < MOTE_BRIGHT
            sz = px(2) if bright else max(1, px(1))
            mid = self.create_rectangle(
                0, 0, 0, 0, outline="",
                fill=theme.HOLO if bright else theme.HOLO_DIM,
                tags=("decor",))
            motes.append([mid,
                          self._rng.uniform(0, w),          # anchor x
                          self._rng.uniform(0, h),          # y
                          self._rng.uniform(px(3), px(7)),  # rise px/s
                          self._rng.uniform(0.0, math.tau),  # wander phase
                          self._rng.uniform(px(2), px(7)),  # wander amp
                          self._rng.uniform(0.25, 0.7),     # wander rad/s
                          sz])
        d["motes"] = motes
        d["wh"] = (w, h)

        self._draw_card(w, cy)
        self._telem_cache = {}
        self._apply_telemetry()
        self._update_gauge()
        self._set_card_mood(self.state(), force=True)
        self._fix_layers()

    def _draw_classic_decor(self, w: int, h: int, cx: int, cy: int,
                            Rr: float, lw: int, thin: int):
        """theme.LOOK == "classic": the 08-31 static scene, item for item
        in the order it was always created (z-order is part of the look).
        Moved out of _draw_decor unchanged on 2026-09-01 when the holo
        stage arrived."""
        # two concentric guide circles, reactor → stage edges (clip freely)
        for off in (68, 118):
            r = Rr + px(off)
            self.create_oval(cx - r, cy - r, cx + r, cy + r,
                             outline=theme.GRID, width=thin, tags=("decor",))

        # degree tick ruler: 72 ticks in the film 6/3/1 length rhythm
        # (long every 6th = 30°, medium every 3rd = 15°) — NO numbers
        for i in range(72):
            m = math.radians(i * 5 - 90)
            if i % 6 == 0:
                ln, col = px(6), theme.HOLO
            elif i % 3 == 0:
                ln, col = px(3), theme.HOLO_DIM
            else:
                ln, col = max(1, px(1)), theme.RAMP20
            self.create_line(cx + math.cos(m) * Rr, cy + math.sin(m) * Rr,
                             cx + math.cos(m) * (Rr + ln),
                             cy + math.sin(m) * (Rr + ln),
                             fill=col, width=thin, tags=("decor",))

        # segmented instrument ring: six UNEQUAL arcs (film law — never
        # equal pie slices), all on ONE radius
        Rseg = Rr + px(10)
        for a0, span in ((12, 84), (106, 40), (152, 98), (258, 26),
                         (288, 50), (345, 22)):
            self.create_arc(cx - Rseg, cy - Rseg, cx + Rseg, cy + Rseg,
                            style="arc", start=90 - a0 - span, extent=span,
                            outline=theme.RAMP47, width=max(1, px(2)),
                            tags=("decor",))

        # long-white-radial-dash outer halo ring (the clock-bezel motif):
        # 12 pill dashes, two-stroke glow treatment, pulled inside the
        # stage so nothing clips at the top/bottom edge
        rh1 = Rr + px(14)
        rh2 = min(Rr + px(20), h // 2 - px(6))
        if rh2 - rh1 >= px(2):
            for k in range(12):
                m = math.radians(k * 30 + 15 - 90)
                hx0, hy0 = cx + math.cos(m) * rh1, cy + math.sin(m) * rh1
                hx1, hy1 = cx + math.cos(m) * rh2, cy + math.sin(m) * rh2
                self.create_line(hx0, hy0, hx1, hy1, fill=theme.RAMP33,
                                 width=px(4), capstyle="round",
                                 tags=("decor",))
                self.create_line(hx0, hy0, hx1, hy1, fill=theme.FOCAL,
                                 width=max(1, px(2)), capstyle="round",
                                 tags=("decor",))

        # corner brackets
        arm, inset = px(16), px(8)
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            x0 = inset if sx > 0 else w - inset
            y0 = inset if sy > 0 else h - inset
            self.create_line(x0 + sx * arm, y0, x0, y0, x0, y0 + sy * arm,
                             fill=theme.EDGE, width=lw, tags=("decor",))

        # scanlines at 20% / 80% height, split around the base square
        half = self._size // 2 + px(4)
        for fy in (0.20, 0.80):
            y = int(h * fy)
            if abs(y - cy) < half:
                segs = [(0, cx - half), (cx + half, w)]
            else:
                segs = [(0, w)]
            for x0, x1 in segs:
                if x1 - x0 > px(8):
                    self.create_line(x0, y, x1, y, fill=theme.SCAN,
                                     width=thin, tags=("decor",))

    def _draw_holo_decor(self, w: int, h: int, cx: int, cy: int):
        """theme.LOOK == "holo": the film-style stage. Thin 1px strokes in
        the FRAME family, nothing filled but the gauge hub; the frame is
        open at the bottom (the seam dissolve owns those rows) and every
        corner is notched with a 45° chamfer. Static except the gauge
        needle (_update_gauge, 1 Hz, one coords())."""
        d = self._decor
        thin = 1
        fc = theme.FRAME
        ins, notch = px(FRAME_INSET), px(FRAME_NOTCH)
        fx0, fy0, fx1 = ins, ins, w - ins
        fy1 = h - 2 * len(theme.SEAM_STEPS) - px(6)   # above the dissolve
        foot = px(FRAME_FOOT)

        def line(*xy, fill=fc, width=thin):
            return self.create_line(*xy, fill=fill, width=width,
                                    tags=("decor",))

        # frame: top + sides stop FRAME_NOTCH short of each corner and a
        # chamfer joins them; the sides end in short inward feet
        line(fx0 + notch, fy0, fx1 - notch, fy0)
        line(fx0, fy0 + notch, fx0, fy1)
        line(fx1, fy0 + notch, fx1, fy1)
        line(fx0 + notch, fy0, fx0, fy0 + notch)
        line(fx1 - notch, fy0, fx1, fy0 + notch)
        line(fx0, fy1, fx0 + foot, fy1)
        line(fx1, fy1, fx1 - foot, fy1)

        # top status ruler: baseline + downward ticks (long every 4th) and
        # a short lit cap at its right end — the segmented bar of the film
        # frame, drawn as a ruler so it claims nothing it does not measure
        ry = fy0 + px(7)
        rx0 = fx0 + notch + px(4)
        rx1 = rx0 + int((fx1 - fx0) * TOP_RULER_FRAC)
        line(rx0, ry, rx1, ry)
        x, i = rx0, 0
        while x <= rx1:
            long_ = i % 4 == 0
            line(x, ry, x, ry + (px(4) if long_ else px(2)),
                 fill=theme.HOLO if long_ else theme.HOLO_DIM)
            x += px(8)
            i += 1
        line(rx1 - px(14), ry, rx1, ry, fill=theme.RAIL, width=max(1, px(2)))
        # frame-rate caption at the ruler's right: the baked cycle, which
        # is a fact of this build (AV_FRAMES over AV_PERIOD)
        self.create_text(rx1 + px(8), ry, anchor="w", text=ruler_caption(),
                         fill=theme.FAINT, font=ui_mono(theme.SIZE_CAPTION),
                         tags=("decor",))

        # right-hand rail: a tick every RAIL_PITCH down the right frame
        # line, long every 5th, pointing inward; the engine card's right
        # edge sits CARD_RAIL_GAP inside the long tips (engine_card_x1)
        t_short, t_long = px(RAIL_TICK[0]), px(RAIL_TICK[1])
        y, i = fy0 + notch + px(RAIL_PITCH), 0
        while y < fy1 - px(4):
            long_ = i % 5 == 0
            line(fx1 - (t_long if long_ else t_short), y, fx1, y,
                 fill=theme.HOLO if long_ else theme.HOLO_DIM)
            y += px(RAIL_PITCH)
            i += 1

        # dial gauge, bottom-right inside the frame: a 270° arc open at
        # the bottom, GAUGE_TICKS ticks (major at the ends and the middle),
        # a hub dot, the needle (dynamic) and a tracked-caps label
        r = px(GAUGE_R)
        gx = fx1 - t_long - px(8) - r
        gy = fy1 - px(6) - r
        # Tk arcs: start = 90 - visual; the sweep runs visual 225 → 495,
        # i.e. Tk 315 going CCW through 12 o'clock for 270°
        self.create_arc(gx - r, gy - r, gx + r, gy + r, style="arc",
                        start=315, extent=GAUGE_SWEEP, outline=fc,
                        width=thin, tags=("decor",))
        for i in range(GAUGE_TICKS):
            v = GAUGE_START + GAUGE_SWEEP * i / (GAUGE_TICKS - 1)
            m = math.radians(v - 90.0)
            major = i % ((GAUGE_TICKS - 1) // 2) == 0
            ln = px(3) if major else px(2)
            line(gx + math.cos(m) * r, gy + math.sin(m) * r,
                 gx + math.cos(m) * (r + ln), gy + math.sin(m) * (r + ln),
                 fill=theme.HOLO if major else theme.HOLO_DIM)
        hub = max(1, px(1))
        self.create_oval(gx - hub, gy - hub, gx + hub, gy + hub,
                         fill=theme.GLASS_EDGE, outline="", tags=("decor",))
        d["gauge"] = (gx, gy, r)
        d["needle"] = line(*needle_xy(gx, gy, r, None),
                           fill=theme.ARC_BRIGHT, width=max(1, px(1)))
        self.create_text(gx - r - px(8), gy, anchor="e",
                         text=theme.caption(GAUGE_LABEL, surface=False),
                         fill=theme.FAINT,
                         font=ui_display(theme.SIZE_CAPTION, "semibold"),
                         tags=("decor",))

    def _draw_card(self, w: int, cy: int):
        """Engine card: the only text on the stage. Right flank, right
        edge on PAD, vertically centred on the ring centre; four rows
        label (display SIZE_CAPTION semibold MUTED, anchor w) / value
        (mono SIZE_CAPTION FOCAL, anchor e). No leaders, no ring dots.
        Classic: a lit glass slab. Holo: an outlined thin frame — 1px
        GLASS_EDGE strokes, NO fill (Tk has no alpha and any flat tint
        would sit as a slab on the pool gradient), the same chamfered
        corners, labels in tracked caps, lifted CARD_LIFT or more so the
        dial below clears it by CARD_GAP."""
        d = self._decor
        thin = 1
        holo = theme.LOOK == "holo"
        x1 = engine_card_x1(w)
        x0 = x1 - px(CARD_W)
        ch = 2 * px(14) + len(CARD_ROWS) * px(CARD_ROW)
        lift = 0
        if holo:
            # clear the dial below (drawn first, so its geometry is known):
            # CARD_GAP above its top tick, but never up into the ruler
            # caption at the frame top
            lift = px(CARD_LIFT)
            g = d.get("gauge")
            if g:
                gauge_top = g[1] - g[2] - px(3)
                lift = max(lift, (cy + ch // 2) - (gauge_top - px(CARD_GAP)))
            lift = min(lift, max(0, cy - ch // 2 - px(FRAME_INSET + 7 + 14)))
        y0 = cy - ch // 2 - lift
        y1 = y0 + ch
        cut = px(8)
        if holo:
            ge = theme.GLASS_EDGE
            for xy in ((x0 + cut, y0, x1, y0), (x1, y0, x1, y1 - cut),
                       (x1, y1 - cut, x1 - cut, y1), (x1 - cut, y1, x0, y1),
                       (x0, y1, x0, y0 + cut), (x0, y0 + cut, x0 + cut, y0)):
                self.create_line(*xy, fill=ge, width=thin, tags=("decor",))
        else:
            # lit glass slab: chamfer top-left / bottom-right, hairline
            # outline, 1px inner top-edge catch-light
            self.create_polygon(
                x0 + cut, y0, x1, y0, x1, y1 - cut, x1 - cut, y1,
                x0, y1, x0, y0 + cut,
                fill=theme.RAISED, outline=theme.RAMP33, width=thin,
                tags=("decor",))
            self.create_line(x0 + cut + 1, y0 + 1, x1 - 1, y0 + 1,
                             fill=theme.GLASS_EDGE, width=thin,
                             tags=("decor",))
        d["card"] = {}
        d["card_lbl"] = {}
        lf = ui_display(theme.SIZE_CAPTION, "semibold")
        vf = ui_mono(theme.SIZE_CAPTION)
        # row keys are ROW labels, so in holo they are plain caps, not
        # tracked (theme.caption; the 09-03 U16 rule -- the card's keys
        # were the one HUD surface still tracking its rows)
        labels = {key: (theme.caption(lab, surface=False) if holo else lab)
                  for lab, key in CARD_ROWS}
        if holo:
            # the display face's labels are wider than the 40 design px
            # the classic budget assumes, and unevenly so (DEVICE vs HEAR):
            # a single budget cut from the widest label ellipsized
            # "WHISPER TURBO" on the HEAR row (seen 2026-09-01 on :97,
            # when the keys were still tracked), so holo budgets per row
            # — each value gets the card's inner width minus ITS label and
            # an 8 design px gutter, never less than 48
            inner = px(CARD_W - 2 * CARD_PAD)
            d["card_budget"] = {
                key: max(px(48), inner - measure(lf, text) - px(8))
                for key, text in labels.items()}
        else:
            d["card_budget"] = px(CARD_W - 2 * CARD_PAD - 40 - 8)
        for i, (lab, key) in enumerate(CARD_ROWS):
            ry = y0 + px(27) + i * px(CARD_ROW)
            d["card_lbl"][key] = self.create_text(
                x0 + px(CARD_PAD), ry, anchor="w", text=labels[key],
                fill=theme.MUTED, font=lf, tags=("decor",))
            d["card"][key] = self.create_text(
                x1 - px(CARD_PAD), ry, anchor="e", text="",
                fill=theme.FOCAL, font=vf, tags=("decor",))
        self._card_mood = None

    def _update_decor(self, t: float):
        """Per-slot decor dynamics: ~10 native canvas calls, plus 16 mote
        coords every MOTE_EVERY-th slot. Everything samples the slot time
        t."""
        d = self._decor
        if not d:
            return
        cx, cy = d["c"]
        r1, r2 = d["sw_r"]
        head = (t * SWEEP_SPEED) % 360.0
        m = math.radians(head - 90.0)
        sweep_xy = (cx + math.cos(m) * r1, cy + math.sin(m) * r1,
                    cx + math.cos(m) * r2, cy + math.sin(m) * r2)
        self.coords(d["sweep"], *sweep_xy)
        self.coords(d["sweep_u"], *sweep_xy)
        for k, aid in enumerate(d["trail"]):
            v = head - 7.0 - k * 7.5
            self.itemconfigure(aid, start=(86.5 - v) % 360.0)
        for oid, r, speed, phase, sz in d["orbits"]:
            a = math.radians(phase + t * speed - 90.0)
            x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
            self.coords(oid, x - sz, y - sz, x + sz, y + sz)
        # dust motes: coords-only, every MOTE_EVERY-th slot, dt from the
        # slot clock (so the drift speed is rate-independent)
        self._mote_beat = (self._mote_beat + 1) % MOTE_EVERY
        if self._mote_beat == 0 and "motes" in d:
            mw, mh = d["wh"]
            dt = max(0.0, min(0.5, t - self._mote_t))
            self._mote_t = t
            wrap = mh + px(8)
            for mo in d["motes"]:
                mid, x0, y, vy, ph, wa, ws, sz = mo
                y = (y - vy * dt) % wrap
                mo[2] = y
                x = (x0 + math.sin(t * ws + ph) * wa) % mw
                self.coords(mid, x, y - px(4), x + sz, y - px(4) + sz)

    # ------------------------------------------------------------- speed
    def set_speed_scale(self, scale) -> None:
        """Multiply the avatar's rotation speed (1 = the designed pace).

        Standby drives this to 1/2 so the disc turns slowly when nobody is
        at the desk. It is the ONLY thing standby changes about the
        reactor: AvatarClock.set_speed rebases phase0/t_speed0, so the
        change cannot pop a frame, and NOTHING here re-bakes — the bases
        are size-dependent (_prerender_all, LEVELS per state colour) and
        re-rendering them at a mode boundary is the churn class behind the
        2026-08-26 freeze."""
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return
        self._speed_scale = max(0.05, min(4.0, scale))

    # -------------------------------------------------------- telemetry
    def set_telemetry(self, fn):
        """Provider callable returning a dict of REAL strings for the
        engine card: asr ('WHISPER small'), tts ('XTTS' / 'EDGE · RYAN'),
        llm (Ollama model name), dev (GPU name / 'NONE'; None while the
        probe is pending). Called at 1Hz on the Tk thread — must be cheap
        attribute reads."""
        self._telemetry_fn = fn

    def _telem_tick(self):
        if not self._alive or not self.winfo_exists():
            return
        _perf.mark("telem")
        try:
            self._apply_telemetry()
            self._update_gauge()
        except tk.TclError:
            return
        except Exception:
            log.exception("telemetry update failed")
        self.after(1000, self._telem_tick)

    def _update_gauge(self):
        """Holo dial needle: the 1-minute load per core, one coords() per
        telemetry tick (1 Hz). No gauge (classic, or decor not built yet)
        -> nothing to do."""
        g = self._decor.get("gauge")
        if not g:
            return
        gx, gy, r = g
        self.coords(self._decor["needle"],
                    *needle_xy(gx, gy, r, read_load_fraction()))

    def _apply_telemetry(self):
        card = self._decor.get("card")
        if not card:
            return
        info = {}
        if self._telemetry_fn is not None:
            try:
                info = self._telemetry_fn() or {}
            except Exception:
                log.debug("telemetry provider failed", exc_info=True)
        # one int (classic) or a per-row dict (holo, see _draw_card)
        budget = self._decor.get("card_budget", px(104))
        vf = ui_mono(theme.SIZE_CAPTION)
        values = {"asr": info.get("asr"), "tts": info.get("tts"),
                  "llm": fmt_llm(info.get("llm")), "dev": info.get("dev"),
                  "fault": info.get("fault")}
        for key, raw in values.items():
            txt = (raw or "--").upper()
            if self._telem_cache.get(key) != txt:
                self._telem_cache[key] = txt
                b = budget[key] if isinstance(budget, dict) else budget
                self.itemconfigure(card[key], text=ellipsize(txt, vf, b))
