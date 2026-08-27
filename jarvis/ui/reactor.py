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
  AV_FRAMES=300 over AV_PERIOD=10 s = 30 unique frames/s, one frame per
  two 60 Hz refreshes. The frame shown at any instant is a pure function
  of a monotonic clock (no counter is incremented, so timer jitter never
  accumulates) and every tick is scheduled for the NEXT SLOT BOUNDARY
  rather than a fixed after(33). Everything else animated on the stage
  (instrument arcs, sweep, orbits, sparks, motes) samples the same slot
  time, so nothing churns off-grid between frame swaps.

Bake (out of process): the reactor launches AV_WORKERS
  `python -m jarvis.ui.avatar_bake` subprocesses (never multiprocessing —
  spawn would re-import jarvis.app → torch, fork is unsafe in a threaded
  Tk+CUDA process). Frames stream back as raw RGB over pipes, read by
  plain threads into a SimpleQueue, and are converted to PhotoImages on
  the Tk thread inside the frame loop's spare time (<= 2 per tick, only
  when >= 15 ms remain before the next boundary). Frames land
  PROGRESSIVELY on nested grids (every 12th → 6th → 3rd → all) so a
  coarse 25-frame cycle plays within seconds of boot; the upgrade to each
  finer tier crosses over on a frame both grids contain (same phase
  angle), so it never pops. Never-mapped Labels pin each photo's Tk
  display instance so a swap is a refcount op, not an XImage rebuild.

HUD scene: static decor (rebuilt only on size settle): the 72-tick degree
  ruler (no numbers), six unequal instrument arcs on ONE radius, a
  12-dash halo ring, two guide circles, corner brackets, split scanlines,
  the seam dissolve into the transcript, and the ENGINE CARD on the right
  flank — the only text on the stage: four rows HEAR / SPEAK / THINK /
  GPU (label MUTED, value FOCAL), fed at 1 Hz by a provider callable
  (set_telemetry) — the reactor never imports app modules. Dynamic: the
  radar sweep + trail, three orbit dots, dust motes, the spark overlay.

Atmosphere: a full-stage backdrop PhotoImage (worker-rendered, rebuilt
  only on size settle) carries a soft radial glow pool centered on the
  ring cluster plus a darker vignette in the corners; the base squares
  bake the SAME analytic pool into their ground so the images seam.
"""
from __future__ import annotations

import math
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
from jarvis.ui.avatar_bake import BakeRunner, pool_ground
from jarvis.ui.avatar_clock import (TIER_STEPS, AvatarClock, LateCounter,
                                    tier_order)
from jarvis.ui.widgets import ellipsize, get_scale, px, ui_display, ui_mono

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
MOTE_EVERY = 3             # coords update every 3rd tick (~10fps)

# Living particulate avatar. CENTERPIECE switches the base art: "avatar"
# is the film JARVIS presence; "reactor" restores the arc-reactor disc.
CENTERPIECE = "avatar"
AV_FRAMES = 300            # baked rotation frames — a seamless loop; 300
                           # over 10 s = 30 unique frames/s = exactly two
                           # 60 Hz refreshes per frame. N % 12 == 0 (tier
                           # grids) and N / P == 30 are asserted by tests.
AV_PERIOD = 10.0           # seconds per loop at 1x (idle)
AV_TICK = AV_PERIOD / AV_FRAMES        # 33.333 ms — the reactor loop slot
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
             ("DEVICE", "dev"))     # DEVICE, not GPU: the status bar's
                                    # 'GPU 39°C' is a different datum
CARD_MOOD = {"listening": "asr", "thinking": "llm", "speaking": "tts"}


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

    def __init__(self, parent, bg=theme.BG, **kw):
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
        card_x0 = w - theme.PAD - px(CARD_W)
        cx = min(round(w * STAGE_CX), int(card_x0 - px(12) - r_out))
        return cx, cy

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
                            workers=AV_WORKERS)
        runner.start()
        self._bake = (gen, size, runner)
        self._bake_t0 = time.monotonic()
        self._render_busy = True
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
            try:
                img = Image.frombuffer("RGB", (size, size), buf,
                                       "raw", "RGB", 0, 1)
                photo = ImageTk.PhotoImage(img)
            except (tk.TclError, RuntimeError, ValueError):
                log.debug("avatar frame %d conversion failed", k,
                          exc_info=True)
                continue
            self._install_av_frame(gen, k, photo)
        return done

    def _install_av_frame(self, gen: int, k: int, photo):
        """One baked frame on the grid. The first frame of a new
        generation swaps in a fresh frame list (a resize re-bake replaces
        the cycle only as its frames land) and resets the tier clock."""
        if self._av_gen_applied != gen:
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
                    # (the pinned X pixmaps live server-side on top)
                    size = photo.width()
                    log.info("avatar: full %d-frame cycle live (%.1fs) — "
                             "frame store %d x %dpx, ~%.0f MB tk masters",
                             AV_FRAMES, time.monotonic() - self._bake_t0,
                             AV_FRAMES, size,
                             AV_FRAMES * size * size * 4 / 1048576.0)
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
        spare time converts <= 2 baked frames, and the next tick is
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
                spare = clock.slot_time(k + 1) - time.monotonic()
                if spare >= 0.015:
                    self._drain_starve = 0
                    self._drain_bake(2)
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
            clock.set_speed(AV_SPEED.get(state, 1), t_abs)
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
        size settle only (keyed on (w, h)); stale generations abort."""
        key = (w, h)
        if self._bd_key == key:
            return
        self._bd_key = key
        self._bd_gen += 1
        threading.Thread(target=self._render_backdrop,
                         args=(w, h, self._bd_gen, self._pool_params(),
                               self._cluster_xy(w, h)),
                         daemon=True, name="backdrop-render").start()

    def _render_backdrop(self, w: int, h: int, gen: int, pool: tuple,
                         centre: tuple):
        """Worker: soft radial glow pool centered on the ring cluster
        (analytic falloff to BG at the stage edges) + a darker vignette in
        the extreme corners."""
        try:
            import numpy as np
            from PIL import Image

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
                (dn - VIGNETTE_START) / (1.0 - VIGNETTE_START), 0.0, 1.0) ** 1.5
            cyan = _hex_rgb(theme.CYAN)
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
        inside h/2 - 4 design px of the cluster centre — nothing clips at
        the stage top/bottom edge."""
        self._decor_job = None
        if not self._alive:
            return
        w, h = self.winfo_width(), self.winfo_height()
        if w < px(140) or h < px(120):
            return
        key = (w, h, self._size)
        if self._decor_key == key:
            return
        self._decor_key = key
        self._rebuild_backdrop(w, h)       # keyed (w, h) internally
        self.delete("decor")
        d = self._decor = {}
        cx, cy = self._cluster_xy(w, h)
        d["c"] = (cx, cy)
        thin = 1
        lw = max(1, px(1))
        Rr = self._size * 0.5 + px(6)          # degree ruler radius
        r_lim = h // 2 - px(4)                 # dynamic-radius ceiling

        # seam dissolve: 1px full-width lines stepping the stage ground
        # into the transcript's lit top tone (every 2nd row keeps the
        # ground, so the band reads as scanline texture, not a stripe) —
        # created first so all other decor stacks above
        n_seam = len(theme.SEAM_STEPS)
        for i, seam_c in enumerate(theme.SEAM_STEPS):
            y = h - 2 * (n_seam - i) + 1
            self.create_line(0, y, w, y, fill=seam_c, width=thin,
                             tags=("decor",))

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

        # radar sweep: leading radial edge (Rr+8 .. Rr+20) + trailing arcs
        # at Rr+12 (dynamic). The bright lead gets the two-stroke
        # treatment: a wide dim underlay created first (below).
        d["sw_r"] = (Rr + px(8), min(Rr + px(20), r_lim))
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
        # updates every 3rd tick — see _update_decor)
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
        self._set_card_mood(self.state(), force=True)
        self._fix_layers()

    def _draw_card(self, w: int, cy: int):
        """Engine card: the only text on the stage. Right flank, right
        edge on PAD, vertically centred on the ring centre; four rows
        label (display SIZE_CAPTION semibold MUTED, anchor w) / value
        (mono SIZE_CAPTION FOCAL, anchor e). No leaders, no ring dots."""
        d = self._decor
        thin = 1
        x1 = w - theme.PAD
        x0 = x1 - px(CARD_W)
        ch = 2 * px(14) + len(CARD_ROWS) * px(CARD_ROW)
        y0 = cy - ch // 2
        y1 = y0 + ch
        cut = px(8)
        # lit glass slab: chamfer top-left / bottom-right, hairline
        # outline, 1px inner top-edge catch-light
        self.create_polygon(
            x0 + cut, y0, x1, y0, x1, y1 - cut, x1 - cut, y1,
            x0, y1, x0, y0 + cut,
            fill=theme.RAISED, outline=theme.RAMP33, width=thin,
            tags=("decor",))
        self.create_line(x0 + cut + 1, y0 + 1, x1 - 1, y0 + 1,
                         fill=theme.GLASS_EDGE, width=thin, tags=("decor",))
        d["card"] = {}
        d["card_lbl"] = {}
        d["card_budget"] = px(CARD_W - 2 * CARD_PAD - 40 - 8)
        lf = ui_display(theme.SIZE_CAPTION, "semibold")
        vf = ui_mono(theme.SIZE_CAPTION)
        for i, (lab, key) in enumerate(CARD_ROWS):
            ry = y0 + px(27) + i * px(CARD_ROW)
            d["card_lbl"][key] = self.create_text(
                x0 + px(CARD_PAD), ry, anchor="w", text=lab,
                fill=theme.MUTED, font=lf, tags=("decor",))
            d["card"][key] = self.create_text(
                x1 - px(CARD_PAD), ry, anchor="e", text="",
                fill=theme.FOCAL, font=vf, tags=("decor",))
        self._card_mood = None

    def _update_decor(self, t: float):
        """Per-slot decor dynamics: ~10 native canvas calls, plus 16 mote
        coords every 3rd slot. Everything samples the slot time t."""
        d = self._decor
        if not d:
            return
        cx, cy = d["c"]
        rs = d["rs"]
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
        # dust motes: coords-only, every 3rd slot, dt from the slot clock
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
        except tk.TclError:
            return
        except Exception:
            log.exception("telemetry update failed")
        self.after(1000, self._telem_tick)

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
        budget = self._decor.get("card_budget", px(104))
        vf = ui_mono(theme.SIZE_CAPTION)
        values = {"asr": info.get("asr"), "tts": info.get("tts"),
                  "llm": fmt_llm(info.get("llm")), "dev": info.get("dev")}
        for key, raw in values.items():
            txt = (raw or "--").upper()
            if self._telem_cache.get(key) != txt:
                self._telem_cache[key] = txt
                self.itemconfigure(card[key], text=ellipsize(txt, vf, budget))
