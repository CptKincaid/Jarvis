"""Clock-indexed frame selection for the avatar loop (pure, Tk-free).

The reactor used to advance a phase accumulator on a fixed after(33) timer
and floor it to the frame grid: a 120-frame/10 s loop has an 83.33 ms
frame period = 2.5 timer ticks, so swaps were discovered alternately after
2 and 3 ticks (a 4/6-refresh beat at 60 Hz). Here the frame index is a
function of a monotonic clock — no counter is ever incremented — and the
next tick is scheduled for the next SLOT boundary (t0 + k·tick), so timer
jitter never accumulates: a late callback recomputes the current slot.

Progressive tiers: the bake lands coarse-to-fine on nested grids (step
24 → 12 → 6 → 3 → 1 for N = 600). `display_index` snaps to the active tier
and crosses to a finer tier only on a frame both grids contain (same phase
angle), so the upgrade never pops.

Drain budget: the frame loop converts baked frames to PhotoImages in the
spare time before the next slot boundary. `drain_budget` turns the
measured spare and a running estimate of the conversion cost (`ConvCost`)
into a frame count, so a 16.7 ms slot with ~14 ms spare drains several
frames instead of tripping a fixed "need 15 ms" threshold every slot.
"""
from __future__ import annotations

import math
import time

TIER_STEPS = (24, 12, 6, 3, 1)  # nested grids, coarse → fine; every hold
                                # is an integer number of 60 Hz refreshes
                                # (400 / 200 / 100 / 50 / 16.7 ms at N=600,
                                # so the first coarse cycle is still 25
                                # frames — motion arrives as early as it
                                # did at 300 frames)


def tier_order(n_frames: int) -> list:
    """Bake order: frames of the coarsest grid first, then the frames each
    finer grid ADDS, so tiers complete in order and each grid is a
    superset of the previous. Covers every frame exactly once."""
    seen: set = set()
    order: list = []
    for step in TIER_STEPS:
        for k in range(0, n_frames, step):
            if k not in seen:
                seen.add(k)
                order.append(k)
    return order


def tier_frames(n_frames: int, step: int) -> list:
    return list(range(0, n_frames, step))


class AvatarClock:
    """Frame index and slot scheduling from a monotonic clock.

    phase(t)  = (phase0 + (t − t_speed0)·speed / period) mod 1
    slot(t)   = floor((t − t0) / tick);  slot_time(k) = t0 + k·tick
    index(t)  = floor(phase(t)·N) mod N
    """

    def __init__(self, n_frames: int, period: float, tick: float = None,
                 speed: float = 1, now: float = None):
        self.N = int(n_frames)
        self.P = float(period)
        self.tick = float(tick) if tick else self.P / self.N
        self.t0 = time.monotonic() if now is None else float(now)
        self.speed = speed
        self.phase0 = 0.0
        self.t_speed0 = self.t0
        self.available_step = 0     # finest COMPLETE tier (0 = none yet)
        self.active_step = 0        # tier currently displayed

    # ------------------------------------------------------------ phase
    def phase(self, t: float) -> float:
        return (self.phase0 + (t - self.t_speed0) * self.speed / self.P) % 1.0

    def set_speed(self, speed, now: float) -> None:
        """Change the loop speed without a phase jump."""
        if speed == self.speed:
            return
        self.phase0 = self.phase(now)
        self.t_speed0 = now
        self.speed = speed

    # Slot times sit EXACTLY on frame boundaries (t0 + k·P/N), so
    # phase·N is nominally an integer there; floating-point rounding can
    # land it a few ulp below (k − 1e-13), which floor() would turn into a
    # held frame followed by a skipped one every few hundred slots. The
    # epsilon (in frame units) absorbs that — far below one frame.
    _EPS = 1e-6

    def index_at(self, t: float) -> int:
        return int(math.floor(self.phase(t) * self.N + self._EPS)) % self.N

    # ------------------------------------------------------------ slots
    def slot(self, now: float) -> int:
        return int(math.floor((now - self.t0) / self.tick))

    def slot_time(self, k: int) -> float:
        return self.t0 + k * self.tick

    def next_delay_ms(self, now: float) -> int:
        """Milliseconds until the next slot boundary (>= 1). Computed from
        the grid, never from the previous delay, so lateness never
        compounds."""
        target = self.slot_time(self.slot(now) + 1)
        return max(1, int(math.ceil((target - now) * 1000.0)))

    # ------------------------------------------------------------ tiers
    def reset_tiers(self) -> None:
        self.available_step = 0
        self.active_step = 0

    def set_available_step(self, step: int) -> None:
        """A tier finished baking. The first tier goes live at once; finer
        tiers wait for `display_index` to cross over on a shared frame."""
        if self.available_step and step >= self.available_step:
            return
        self.available_step = step
        if self.active_step == 0:
            self.active_step = step

    def frames_per_slot(self) -> float:
        """How far the index advances per slot at the current speed (1 at
        idle, 2 thinking/speaking, 0.5 in standby)."""
        return self.speed * self.tick * self.N / self.P

    def display_index(self, t: float):
        """Index to show at t, snapped to the active tier; None while no
        tier is complete. Crosses to a finer available tier only on a
        slot where the COARSE tier itself swaps frames — idx has just
        passed a multiple of active_step (idx % active_step is less than
        the frames the index advances per slot) — so the upgrade lands
        within one frame of the phase angle the coarse grid would have
        shown anyway: no pop. Requiring idx % active_step == 0 exactly
        would stall the upgrade for as long as speed 2 (thinking/speaking)
        keeps the index on the parity it was entered with — an odd index
        is never ≡ 0 mod 24."""
        if self.active_step == 0:
            return None
        idx = self.index_at(t)
        if self.active_step != self.available_step and \
                idx % self.active_step < max(1.0, self.frames_per_slot()):
            self.active_step = self.available_step
        return idx - idx % self.active_step


DRAIN_MARGIN_S = 0.002          # spare time left untouched after a drain
                                # so the next tick still starts on time
DRAIN_CAP = 3                   # frames per slot at most (bounds the worst
                                # case if the cost estimate is stale)
CONV_PRIOR_S = 0.004            # assumed conversion cost until measured


def drain_budget(spare_s: float, conv_s: float, cap: int = DRAIN_CAP,
                 margin_s: float = DRAIN_MARGIN_S) -> int:
    """How many baked frames fit in `spare_s` seconds before the next slot
    boundary, each costing `conv_s` (a running estimate of the measured
    PhotoImage conversion), leaving `margin_s` untouched. 0 when even one
    frame would not fit; never above `cap`. A non-positive `conv_s` (no
    measurement yet, or a clock glitch) is treated as CONV_PRIOR_S."""
    if conv_s <= 0.0:
        conv_s = CONV_PRIOR_S
    room = spare_s - margin_s
    if room < conv_s:
        return 0
    return max(0, min(int(cap), int(room // conv_s)))


class ConvCost:
    """Running estimate of a per-frame cost for `drain_budget`: the MAX of
    the last `keep` samples (pessimistic, so one slow conversion shrinks
    the next slots' budget instead of letting them overrun), `prior_s`
    until the first sample lands.

    Samples also EXPIRE by age: `tick()` is called once per slot and a
    sample `keep` slots old is dropped whether or not anything was drained
    since. Without that, a single outlier at or above the spare time (the
    FIRST frame of a generation measures 10-16 ms in the full window
    against ~2 ms for every later one — Tk's first-image one-offs plus the
    bake's start-up burst; a GIL-contended boot measured 17.7 ms) would
    return budget 0 on every slot, and the only thing that could ever push
    it out of the window — the starvation fallback's 1 frame per 10 slots
    — would take 160 slots (2.7 s) to do so. With expiry the outlier is
    gone within `keep` slots (0.27 s at 16.67 ms)."""

    def __init__(self, prior_s: float = CONV_PRIOR_S, keep: int = 16):
        self.prior_s = float(prior_s)
        self.keep = max(1, int(keep))
        self._samples: list = []            # (slot added, seconds)
        self._slot = 0

    def reset(self) -> None:
        """A new generation (re-bake at another size) starts from the
        prior: the old size's costs — and the last frame's sample, which
        carried the full-cycle bookkeeping — are not its costs."""
        self._samples = []
        self._slot = 0

    def tick(self) -> None:
        """One slot elapsed; a sample `keep` slots old falls out."""
        self._slot += 1
        self._expire()

    def add(self, seconds: float) -> None:
        self._samples.append((self._slot, max(0.0, float(seconds))))
        if len(self._samples) > self.keep:
            del self._samples[0]

    def _expire(self) -> None:
        horizon = self._slot - self.keep
        while self._samples and self._samples[0][0] <= horizon:
            del self._samples[0]

    @property
    def estimate(self) -> float:
        self._expire()
        return max(s for _, s in self._samples) if self._samples \
            else self.prior_s

    @property
    def n(self) -> int:
        self._expire()
        return len(self._samples)


LATE_MS = 8.0                   # a slot is late when its tick starts this
                                # long after the slot boundary: ~half a
                                # 16.67 ms slot (it was a quarter of the
                                # old 33 ms slot; half a slot is where a
                                # frame visibly lands on the wrong refresh)
LATE_WINDOW_S = 30.0            # summary line cadence


class LateCounter:
    """Late-slot bookkeeping for the frame loop (pure, Tk-free).

    `observe(now, k)` records a tick that started at `now` and found itself
    in slot k. The tick was DUE in the slot after the previous one; the
    slots in between (if any) were never rendered and count as late, and
    slot k itself counts as late when the tick started more than `late_ms`
    after its boundary. `report(now)` returns the summary line once per
    window ('avatar: late slots N/M (max lateness X ms)') and resets."""

    def __init__(self, clock: AvatarClock, late_ms: float = LATE_MS,
                 window_s: float = LATE_WINDOW_S, detail_max: int = 10):
        self.clock = clock
        self.late_s = float(late_ms) / 1000.0
        self.window_s = float(window_s)
        self.detail_max = int(detail_max)
        self._expected = None       # slot the next tick is due in
        self._win_t0 = None
        self._win_k0 = 0
        self._details = 0
        self.late = 0
        self.max_late_s = 0.0

    def observe(self, now: float, k: int) -> tuple:
        """-> (lateness in s relative to the slot the tick was due in,
        number of slots skipped outright)."""
        exp = k if self._expected is None else self._expected
        if self._win_t0 is None:
            self._win_t0, self._win_k0 = now, k
        skipped = max(0, k - exp)
        lateness = max(0.0, now - self.clock.slot_time(exp))
        self.late += skipped
        if now - self.clock.slot_time(k) > self.late_s:
            self.late += 1
        if lateness > self.max_late_s:
            self.max_late_s = lateness
        self._expected = k + 1
        return lateness, skipped

    def detail_ok(self) -> bool:
        """Rate limit for per-event detail lines (detail_max per window)."""
        if self._details >= self.detail_max:
            return False
        self._details += 1
        return True

    def report(self, now: float):
        if self._win_t0 is None or now - self._win_t0 < self.window_s:
            return None
        k_now = self.clock.slot(now)
        total = max(1, k_now - self._win_k0)
        line = "avatar: late slots %d/%d (max lateness %.1f ms)" % (
            min(self.late, total), total, self.max_late_s * 1000.0)
        self._win_t0, self._win_k0 = now, k_now
        self.late = 0
        self.max_late_s = 0.0
        self._details = 0
        return line
