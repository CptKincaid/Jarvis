"""Console modes — ONE state machine for standby, ambient and power-up.

Three proposals wanted three overlays: a room clock when the desk is empty,
a quiet room-state slab between conversations, and a staged sweep the first
time he sits down after the night. Built separately they would fight over
the same surface. Built as one machine they are three MODES of the console:

    active    a turn is live, or he is at the keyboard — the app as it is
    ambient   nobody has spoken for a while — the transcript recedes and
              one quiet slab of room state takes the stage
    standby   nobody has been at the desk for a long while — the console
              becomes the room's clock, dimmed and slowly drifting

plus one transient, POWER-UP: the first activity after an overnight gap
replays the panels in sequence before dropping into `active`.

Every rule here is a pure function of its inputs (`next_mode`, `dim_factor`,
`drift_offset`, `powerup_due`, `reveal_schedule`, `dim`); `ConsoleModes` is
the thin Tk driver that calls them once a second. That split is the same one
jarvis/ui/views.py makes, and it is what lets the whole machine be tested
with a frozen clock and no display.

Four decisions worth knowing about, all made against the corrections:

* **The desk-idle probe never runs on the Tk thread.** `DeskWatch` caches
  it on a thread of its own, so the mode tick is an attribute read. The
  XScreenSaver fallback would have been cheap enough, but the desk-presence
  module's probe shells out to `gdbus`, and a subprocess spawn on the frame
  loop is a dropped avatar frame every time it fires.

* **Dimming is CANVAS COLOUR, never xrandr.** A gamma-crushed desktop left
  behind by a crash is a far worse failure than never dimming, and the
  baked avatar PhotoImages must not be touched at all — re-baking them is
  the churn class behind the 2026-08-26 freeze. `dim()` blends toward the
  ground; the reactor's frames are left exactly as they are.

* **Burn-in drift moves the WINDOW, not the glyphs.** The reactor's decor
  is baked at fixed canvas coordinates, so drifting the art inside the
  canvas would tear it away from its rulers. Moving the whole console a few
  pixels a minute drifts the brightest thing on the panel — the reactor
  disc — for free, and is undone by restoring one geometry string.

* **The abort is generation-driven, not id-driven.** Every pending stage of
  the power-up sweep checks a counter that HotwordDetected / UserUtterance
  bump, so the reply always wins instantly without anyone tracking Tk
  after-ids (the same trap the transcript's self-rescheduling _atmo_tick
  documents).
"""
from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from datetime import datetime
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.ui import theme

log = get_logger("ui.console_mode")

ACTIVE, AMBIENT, STANDBY = "active", "ambient", "standby"
MODES = (ACTIVE, AMBIENT, STANDBY)

TICK_MS = 1000              # mode tick: an attribute read and some arithmetic
                            # (the probe itself runs on DeskWatch's thread)
AMBIENT_AFTER_S = 45.0      # quiet between conversations
STANDBY_AFTER_S = 720.0     # 12 minutes away from the keyboard
DIM_FLOOR = 0.35            # never darker than this, at any hour
AMBIENT_DIM = 0.78
DRIFT_PX_PER_MIN = 3.0      # burn-in walk speed
DRIFT_RADIUS = 42           # px; the walk stays inside this circle
POWERUP_GAP_H = 6.0         # the overnight gap that earns the sweep
POWERUP_STEP_MS = 900       # between panels in the staged reveal
POWERUP_HOLD_MS = 2600      # the finished board holds before it settles


# ------------------------------------------------------------ pure rules
def next_mode(*, idle_s: Optional[float], busy: bool = False,
              alarm: bool = False, ambient_after_s: float = AMBIENT_AFTER_S,
              standby_after_s: float = STANDBY_AFTER_S,
              ambient: bool = True, standby: bool = True) -> str:
    """The mode the console should be in right now.

    `idle_s` is seconds since the last input at the DESK (None = unknown).
    Unknown means active: a machine that cannot see the keyboard must never
    guess that nobody is there and blank itself mid-sentence.

    `busy` covers a live turn (speaking / listening / thinking) and `alarm`
    a ringing modal — both pin the console awake however long the desk has
    been idle, because an alarm that rings behind a room clock is a bug."""
    if busy or alarm or idle_s is None:
        return ACTIVE
    try:
        idle = float(idle_s)
    except (TypeError, ValueError):
        return ACTIVE
    if standby and idle >= float(standby_after_s):
        return STANDBY
    if ambient and idle >= float(ambient_after_s):
        return AMBIENT
    return ACTIVE


def night_curve(hour: float) -> float:
    """1.0 in the working day falling to 0.0 in the small hours, smoothly.

    A step at 22:00 would be visible from across the room; a cosine over
    the 20:00 → 02:00 shoulder is not. Hours 02:00–07:00 sit at the bottom
    and 09:00–20:00 at the top."""
    h = float(hour) % 24.0
    if 9.0 <= h < 20.0:
        return 1.0
    if 2.0 <= h < 7.0:
        return 0.0
    if 20.0 <= h or h < 2.0:                 # 20:00 → 02:00, falling
        t = (h - 20.0) / 6.0 if h >= 20.0 else (h + 4.0) / 6.0
    else:                                    # 07:00 → 09:00, rising
        t = 1.0 - (h - 7.0) / 2.0
    return 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(1.0, t))))


def dim_factor(mode: str, hour: float = 12.0,
               floor: float = DIM_FLOOR) -> float:
    """How bright the console is in `mode` at `hour`: 1.0 fully lit, `floor`
    at its darkest. Only standby follows the clock — the ambient slab sits
    between conversations in the middle of the day and must stay readable."""
    floor = max(0.05, min(1.0, float(floor)))
    if mode == ACTIVE:
        return 1.0
    if mode == AMBIENT:
        return max(floor, AMBIENT_DIM)
    return floor + (1.0 - floor) * night_curve(hour) * 0.55


def drift_offset(elapsed_s: float, px_per_min: float = DRIFT_PX_PER_MIN,
                 radius: int = DRIFT_RADIUS) -> tuple:
    """(dx, dy) for the burn-in walk: a slow circle of `radius` pixels
    travelled at `px_per_min`. Pure and deterministic in elapsed time, so a
    tick that arrives late lands where it should rather than accumulating
    drift error the way an incrementing offset would."""
    radius = max(0, int(radius))
    if radius == 0 or px_per_min <= 0:
        return (0, 0)
    circumference = 2.0 * math.pi * radius
    period = circumference / float(px_per_min) * 60.0
    a = 2.0 * math.pi * (float(elapsed_s) % period) / period
    # a 2:3 Lissajous, not a circle: a circle retraces one ring of pixels
    return (int(round(radius * math.sin(a))),
            int(round(radius * 0.6 * math.sin(1.5 * a))))


def _rgb(color: str) -> tuple:
    color = str(color).lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def dim(color: str, factor: float, ground: str = theme.BG) -> str:
    """`color` blended toward `ground` by `factor` (1.0 = untouched).

    Canvas items have no alpha, so every dim in this UI is a pre-blend —
    the same trick theme.py uses to build its ramps. Never applied to a
    PhotoImage: re-baking the avatar at a mode change is exactly the churn
    the 08-26 freeze came from."""
    f = max(0.0, min(1.0, float(factor)))
    if f >= 1.0:
        return color
    try:
        c, g = _rgb(color), _rgb(ground)
    except (ValueError, IndexError):
        return color
    return "#%02x%02x%02x" % tuple(
        max(0, min(255, int(gv + (cv - gv) * f))) for cv, gv in zip(c, g))


def powerup_due(last_iso: str, *, now: Optional[datetime] = None,
                idle_s: Optional[float] = None,
                gap_h: float = POWERUP_GAP_H) -> bool:
    """Has today earned the power-up sweep?

    Two gates, both required: it has not already run today (the date latch
    the app keeps beside the briefing's, so a crash-relaunch cannot replay
    it), and the machine really was left alone — `idle_s` at least `gap_h`
    hours. An unknown idle time passes the second gate: on a box where the
    desk seam is not wired, the first wake word of the day IS the signal,
    and the date latch alone keeps it to once."""
    now = now or datetime.now()
    if str(last_iso or "") == now.date().isoformat():
        return False
    if idle_s is not None:
        try:
            if float(idle_s) < float(gap_h) * 3600.0:
                return False
        except (TypeError, ValueError):
            return True
    return True


def reveal_schedule(panels: int, step_ms: int = POWERUP_STEP_MS,
                    hold_ms: int = POWERUP_HOLD_MS) -> list:
    """[(delay_ms, revealed)] for the staged sweep: the panels draw in one
    at a time, then the whole board holds before the console settles back
    to its ordinary mode (revealed=None means "everything, unstaged").

    Pure, so the choreography's timing is tuned in a test rather than by
    watching the real thing boot once a day."""
    n = max(0, int(panels))
    step = max(1, int(step_ms))
    out = [(i * step, i + 1) for i in range(n)]
    out.append((n * step + max(0, int(hold_ms)), None))
    return out


def today_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).date().isoformat()


def hour_of(now: Optional[datetime] = None) -> float:
    now = now or datetime.now()
    return now.hour + now.minute / 60.0


# ------------------------------------------------------- the idle seam
_XSS_DISPLAY = None
_XSS_DEAD = False


def xss_idle_s() -> Optional[float]:
    """Seconds since the last input on this X display, from the
    XScreenSaver extension — the last-resort desk-idle source.

    Read-only and event-driven inside the server (no polling, no sudo);
    measured 13124308 ms on :1 during vetting. The Display object is cached
    because opening one per tick leaks a socket, and one failure disables
    the probe for the life of the process rather than logging every 4 s."""
    global _XSS_DISPLAY, _XSS_DEAD
    if _XSS_DEAD:
        return None
    try:
        if _XSS_DISPLAY is None:
            from Xlib import display as _xdisplay
            _XSS_DISPLAY = _xdisplay.Display()
        info = _XSS_DISPLAY.screen().root.screensaver_query_info()
        return float(info.idle) / 1000.0
    except Exception:                       # noqa: BLE001 - probe boundary
        log.info("desk idle probe unavailable; standby stays off")
        _XSS_DEAD = True
        _XSS_DISPLAY = None
        return None


class DeskWatch:
    """Poll the desk-idle probe on a thread of its own and cache the answer.

    The probe must NOT run on the Tk thread. The XScreenSaver fallback is a
    microsecond-scale X round trip, but the desk-presence module's version
    shells out to `gdbus`, and a ~30 ms subprocess spawn on the frame loop
    is a dropped avatar frame every time it runs. Caching it here keeps the
    mode tick to an attribute read while still noticing a keystroke within
    `interval` — which is what "touch anything and it comes back" needs.

    Joinable, start/stop the same shape as jarvis/deadlines.py."""

    def __init__(self, probe: Optional[Callable] = None,
                 interval: float = 1.0):
        self._probe = probe
        self.interval = max(0.2, float(interval))
        self.idle: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def read(self) -> Optional[float]:
        """What the mode tick calls: the last cached reading, or None."""
        return self.idle

    def poll(self) -> Optional[float]:
        """One probe. A failure caches None — unknown, never "away"."""
        if not callable(self._probe):
            self.idle = None
            return None
        try:
            value = self._probe()
            self.idle = None if value is None else float(value)
        except Exception:                   # noqa: BLE001 - probe boundary
            log.debug("desk idle probe failed", exc_info=True)
            self.idle = None
        return self.idle

    def start(self) -> None:
        if self.running or not callable(self._probe):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="desk-watch",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self.interval)


def resolve_idle_fn(services=None) -> Optional[Callable]:
    """The desk-idle provider, in preference order:

      1. `services.desk_idle_s` — the seam the desk-presence group exposes
      2. `jarvis.desk.desk_idle_s` — the same module, imported directly
      3. the XScreenSaver probe above

    getattr/ImportError defensively at every step so this lands whichever
    order the two changes merge in, and so a box without either still gets
    a room clock."""
    fn = getattr(services, "desk_idle_s", None)
    if callable(fn):
        return fn
    try:
        from jarvis import desk as desk_mod       # may not exist yet
        fn = getattr(desk_mod, "desk_idle_s", None)
        if callable(fn):
            return fn
    except ImportError:
        pass
    except Exception:                       # noqa: BLE001 - import boundary
        log.debug("jarvis.desk import failed", exc_info=True)
    return xss_idle_s


# ------------------------------------------------------------- the driver
class ConsoleModes:
    """Thin Tk driver over the rules above.

    Owns exactly one self-rescheduling tick and one generation counter. All
    callbacks are injected so the driver is constructed in a test with
    plain functions and a fake clock:

        after(ms, fn)      schedule on the Tk thread (main_window._after)
        idle_fn()          seconds since desk input, or None
        on_mode(mode)      the console changes surface
        on_dim(factor)     repaint at this brightness
        on_drift(dx, dy)   move the window for burn-in
        on_stage(n)        power-up: show the first n panels (None = all)
        busy_fn()          True while a turn is live
        alarm_fn()         True while an alarm modal is up
        option(key, dflt)  assistant.json reader
    """

    def __init__(self, *, after: Callable, idle_fn: Optional[Callable] = None,
                 on_mode: Optional[Callable] = None,
                 on_dim: Optional[Callable] = None,
                 on_drift: Optional[Callable] = None,
                 on_stage: Optional[Callable] = None,
                 busy_fn: Optional[Callable] = None,
                 alarm_fn: Optional[Callable] = None,
                 option: Optional[Callable] = None,
                 clock: Callable = time.monotonic,
                 now: Callable = datetime.now):
        self._after = after
        self._idle_fn = idle_fn
        self._on_mode = on_mode
        self._on_dim = on_dim
        self._on_drift = on_drift
        self._on_stage = on_stage
        self._busy_fn = busy_fn
        self._alarm_fn = alarm_fn
        self._option = option
        self._clock = clock
        self._now = now
        self.mode = ACTIVE
        self.generation = 0
        self.sweeping = False
        self._standby_since = 0.0
        self._last_dim = 1.0
        self._stopped = False

    # ------------------------------------------------------------ config
    def _opt(self, key: str, default):
        if not callable(self._option):
            return default
        try:
            value = self._option(key, default)
        except Exception:                   # noqa: BLE001 - config boundary
            log.debug("console mode: option %s unreadable", key, exc_info=True)
            return default
        return default if value is None else value

    def _num(self, key: str, default: float) -> float:
        try:
            return float(self._opt(key, default))
        except (TypeError, ValueError):
            return float(default)

    # ------------------------------------------------------------ inputs
    def note_activity(self) -> None:
        """Anything he did: a wake word, an utterance, a keystroke. Wakes
        the console instantly and bumps the generation, which no-ops every
        pending power-up stage — "the reply wins" without tracking ids."""
        self.generation += 1
        self.sweeping = False
        if self.mode != ACTIVE:
            self._set_mode(ACTIVE)
        else:
            self._apply_dim()

    def note_output(self) -> None:
        """Something to SHOW arrived — a reply, a briefing card, an alarm
        modal, a Claude progress line. The surface comes down so the card is
        not drawn behind it, but the power-up sweep is NOT cancelled: the
        sweep is meant to play while the morning briefing is delivered over
        the top of it. Only what HE did cancels it (note_activity)."""
        if self.sweeping:
            return                  # the sweep already holds an ACTIVE console
        if self.mode != ACTIVE:
            self._set_mode(ACTIVE)

    def idle_seconds(self) -> Optional[float]:
        if not callable(self._idle_fn):
            return None
        try:
            value = self._idle_fn()
        except Exception:                   # noqa: BLE001 - probe boundary
            log.debug("desk idle probe failed", exc_info=True)
            return None
        return None if value is None else float(value)

    def _busy(self) -> bool:
        for fn in (self._busy_fn, self._alarm_fn):
            if callable(fn):
                try:
                    if fn():
                        return True
                except Exception:           # noqa: BLE001 - callback boundary
                    log.debug("console mode: state callback failed",
                              exc_info=True)
        return False

    # -------------------------------------------------------------- tick
    def tick(self) -> str:
        """One pass: read the desk, decide the mode, repaint, drift. Returns
        the mode, so a test drives it without a Tk loop."""
        if self.sweeping:
            return self.mode                # the sweep owns the surface
        want = next_mode(
            idle_s=self.idle_seconds(),
            busy=self._busy(),
            ambient_after_s=self._num("console.ambient_after_s",
                                      AMBIENT_AFTER_S),
            standby_after_s=self._num("console.standby_after_min",
                                      STANDBY_AFTER_S / 60.0) * 60.0,
            ambient=bool(self._opt("console.ambient", True)),
            standby=bool(self._opt("console.standby", True)))
        if want != self.mode:
            self._set_mode(want)
        else:
            self._apply_dim()
            self._apply_drift()
        return self.mode

    def start(self) -> None:
        self._stopped = False
        self._schedule()

    def stop(self) -> None:
        """Wake the console and leave nothing changed behind — called at
        quit and whenever the window goes away. Restoring the surface is
        not optional: a console left dimmed and displaced after a crash is
        the failure this whole machine has to avoid."""
        self._stopped = True
        self.sweeping = False
        self.generation += 1
        if self.mode != ACTIVE:
            self._set_mode(ACTIVE)

    def _schedule(self) -> None:
        if self._stopped:
            return
        try:
            self._after(TICK_MS, self._loop)
        except tk.TclError:
            log.debug("console mode: after() on a dead window", exc_info=True)

    def _loop(self) -> None:
        if self._stopped:
            return
        try:
            self.tick()
        except Exception:                   # noqa: BLE001 - tick boundary
            log.exception("console mode tick failed")
        self._schedule()

    # ------------------------------------------------------------ output
    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == STANDBY:
            self._standby_since = self._clock()
        log.info("console mode -> %s", mode)
        if callable(self._on_mode):
            try:
                self._on_mode(mode)
            except Exception:               # noqa: BLE001 - callback boundary
                log.exception("console mode callback failed")
        self._apply_dim()
        self._apply_drift()

    def _apply_dim(self) -> None:
        factor = dim_factor(self.mode, hour_of(self._now()),
                            self._num("console.standby_dim", DIM_FLOOR))
        if abs(factor - self._last_dim) < 0.005:
            return                          # no repaint for a rounding step
        self._last_dim = factor
        if callable(self._on_dim):
            try:
                self._on_dim(factor)
            except Exception:               # noqa: BLE001 - callback boundary
                log.exception("console dim callback failed")

    def _apply_drift(self) -> None:
        if not callable(self._on_drift):
            return
        if self.mode != STANDBY:
            offset = (0, 0)
        else:
            offset = drift_offset(self._clock() - self._standby_since,
                                  self._num("console.drift_px_per_min",
                                            DRIFT_PX_PER_MIN))
        try:
            self._on_drift(*offset)
        except Exception:                   # noqa: BLE001 - callback boundary
            log.exception("console drift callback failed")

    # ---------------------------------------------------------- power-up
    def power_up(self, panels: int) -> None:
        """Play the staged sweep. Every stage checks the generation, so the
        first thing he says cancels the rest of it instantly."""
        if not bool(self._opt("console.powerup", True)) or panels <= 0:
            return
        gen = self.generation
        self.sweeping = True
        if self.mode != ACTIVE:
            self._set_mode(ACTIVE)
        log.info("power-up sweep: %d panels", panels)
        for delay, count in reveal_schedule(panels):
            self._after(delay, lambda c=count, g=gen: self._stage(g, c))

    def _stage(self, gen: int, count) -> None:
        if gen != self.generation or self._stopped:
            return                          # he spoke; the sweep is over
        if count is None:
            self.sweeping = False
        if callable(self._on_stage):
            try:
                self._on_stage(count)
            except Exception:               # noqa: BLE001 - callback boundary
                log.exception("power-up stage callback failed")


__all__ = ["ACTIVE", "AMBIENT", "STANDBY", "MODES", "ConsoleModes", "DeskWatch",
           "dim", "dim_factor", "drift_offset", "next_mode", "night_curve",
           "powerup_due", "resolve_idle_fn", "reveal_schedule", "today_iso",
           "xss_idle_s"]
