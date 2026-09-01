"""Display-free tests for the clock-indexed avatar frame scheduler."""
import math

from jarvis.ui.avatar_clock import (TIER_STEPS, AvatarClock, ConvCost,
                                    drain_budget, tier_frames, tier_order)

N, P = 600, 10.0
TICK = P / N


def _clock(speed=1, t0=100.0):
    return AvatarClock(N, P, TICK, speed=speed, now=t0)


def test_frame_constants_match_the_60hz_grid():
    from jarvis.ui import reactor
    assert reactor.AV_FRAMES % 24 == 0            # nested 24/12/6/3/1 grids
    assert reactor.AV_FRAMES / reactor.AV_PERIOD == 60   # 60 fps = 1 refresh
    assert math.isclose(reactor.AV_TICK, reactor.AV_PERIOD / reactor.AV_FRAMES)
    assert all(isinstance(v, int) for v in reactor.AV_SPEED.values())
    assert reactor.AV_FRAMES == N and reactor.AV_PERIOD == P


def test_every_tier_hold_is_an_integer_number_of_60hz_refreshes():
    from jarvis.ui import reactor
    assert TIER_STEPS[0] == 24                    # first coarse cycle stays
    assert reactor.AV_FRAMES // TIER_STEPS[0] == 25   # 25 frames (400 ms)
    assert TIER_STEPS[-1] == 1
    for coarse, fine in zip(TIER_STEPS, TIER_STEPS[1:]):
        assert coarse % fine == 0                  # nested grids
    for step in TIER_STEPS:
        refreshes = reactor.AV_TICK * step * 60.0
        assert abs(refreshes - round(refreshes)) < 1e-9
        assert reactor.AV_FRAMES % step == 0


def test_index_is_monotonic_and_wraps_at_n():
    c = _clock()
    c.set_available_step(1)
    prev = None
    wraps = 0
    for i in range(2 * N + 5):
        t = c.t0 + i * TICK + 1e-6
        idx = c.index_at(t)
        assert 0 <= idx < N
        if prev is not None:
            if idx < prev:
                wraps += 1
                assert prev == N - 1 and idx == 0
            else:
                assert idx == prev + 1          # one frame per slot at 1x
        prev = idx
    assert wraps == 2


def test_next_delay_sums_to_the_grid_with_zero_drift():
    c = _clock()
    now = c.t0 + 0.0004
    for k in range(1, 10_001):
        d = c.next_delay_ms(now)
        assert d >= 1
        now += d / 1000.0
        assert c.slot(now) == k                 # never skips or repeats
        lateness = now - c.slot_time(k)
        assert 0.0 <= lateness < 0.0011          # ceil() error never accumulates
    assert abs(now - c.slot_time(10_000)) < 0.0011


def test_late_callback_recovers_the_current_slot():
    c = _clock()
    late = c.slot_time(5) + 0.9 * TICK           # woke almost a slot late
    assert c.slot(late) == 5
    assert c.next_delay_ms(late) == math.ceil(0.1 * TICK * 1000)


def test_set_speed_is_phase_continuous():
    c = _clock(speed=1)
    t = c.t0 + 3.21
    before = c.phase(t)
    c.set_speed(2, t)
    assert math.isclose(c.phase(t), before, abs_tol=1e-12)
    # and advances twice as fast afterwards
    assert math.isclose(c.phase(t + 1.0), (before + 2.0 / P) % 1.0,
                        abs_tol=1e-9)


def test_tier_grids_are_nested_supersets_and_cover_every_frame():
    grids = [set(tier_frames(N, s)) for s in TIER_STEPS]
    for coarse, fine in zip(grids, grids[1:]):
        assert coarse < fine
    order = tier_order(N)
    assert sorted(order) == list(range(N))
    assert len(order) == len(set(order))
    # tiers complete in order: the first 25 entries ARE the step-24 grid,
    # the first 50 the step-12 grid, and so on down every tier
    for step, grid in zip(TIER_STEPS, grids):
        assert set(order[:N // step]) == grid


def test_display_index_snaps_to_active_tier_and_crosses_on_shared_frame():
    c = _clock()
    assert c.display_index(c.t0) is None         # nothing baked yet
    c.set_available_step(12)
    assert c.active_step == 12
    for i in range(30):
        t = c.t0 + i * TICK + 1e-6
        assert c.display_index(t) % 12 == 0
    # a finer tier becomes available mid-hold: the switch must land on a
    # frame both grids contain
    t_avail = c.t0 + 7 * TICK + 1e-6             # idx 7, inside the 0..11 hold
    c.set_available_step(6)
    shown = []
    for i in range(7, 40):
        t = c.t0 + i * TICK + 1e-6
        step_before = c.active_step
        idx = c.display_index(t)
        shown.append(idx)
        if step_before != c.active_step:
            assert c.index_at(t) % 12 == 0      # crossed on a shared frame
            assert idx % 12 == 0
    assert c.active_step == 6
    assert shown[:5] == [0, 0, 0, 0, 0]          # held the coarse frame
    assert any(s % 12 == 6 for s in shown)       # then used the finer grid
    assert t_avail < c.t0 + 12 * TICK


def test_coarser_availability_never_overrides_finer():
    c = _clock()
    c.set_available_step(3)
    c.set_available_step(12)
    assert c.available_step == 3
    c.reset_tiers()
    assert c.available_step == 0 and c.active_step == 0


def test_index_on_every_slot_boundary_is_exact_no_held_or_skipped_frames():
    # regression: floor(phase*N) at t = t0 + k*P/N used to evaluate to k-1
    # every few hundred slots (float rounding) -> a 4-refresh hold then a
    # skipped frame, visible as a periodic stutter at 30 fps
    for t0 in (100.0, 12345.678, 987654.321):
        c = AvatarClock(N, P, TICK, now=t0)
        c.set_available_step(1)
        for k in range(0, 30_000):
            assert c.display_index(c.slot_time(k)) == k % N


def test_late_counter_counts_late_and_skipped_slots_per_window():
    from jarvis.ui.avatar_clock import LateCounter
    c = _clock()
    lc = LateCounter(c, late_ms=8.0, window_s=30.0)
    for k in range(10):                               # on time: +1 ms
        late, skipped = lc.observe(c.slot_time(k) + 0.001, k)
        assert skipped == 0 and late < 0.008
    assert lc.late == 0
    late, skipped = lc.observe(c.slot_time(10) + 0.020, 10)   # 20 ms late
    assert skipped == 0 and math.isclose(late, 0.020) and lc.late == 1
    # due in slot 11, lands in slot 13: 11 and 12 never rendered
    late, skipped = lc.observe(c.slot_time(13) + 0.001, 13)
    assert skipped == 2 and lc.late == 3
    assert math.isclose(late, 2 * TICK + 0.001)
    assert lc.report(c.slot_time(13) + 0.001) is None        # window open
    for k in range(14, 1801):
        lc.observe(c.slot_time(k) + 0.001, k)
    now = c.slot_time(1800) + 0.001                         # 30 s elapsed
    line = lc.report(now)
    assert line == "avatar: late slots 3/1800 (max lateness 34.3 ms)"
    assert lc.late == 0 and lc.max_late_s == 0.0            # window reset
    assert lc.report(now) is None


def test_late_threshold_is_about_half_a_slot():
    # a "late" slot is one that missed its boundary by ~half a slot; the
    # constant was tuned when a slot was 33 ms and must not silently keep
    # meaning "a quarter of a slot" now that one is 16.67 ms
    from jarvis.ui.avatar_clock import LATE_MS
    assert 0.4 * TICK <= LATE_MS / 1000.0 <= 0.6 * TICK


def test_drain_budget_fits_frames_into_the_spare_time_with_a_margin():
    # 14 ms spare, 2 ms margin, 1.5 ms per frame -> 8 fit, capped at 3
    assert drain_budget(0.014, 0.0015) == 3
    assert drain_budget(0.014, 0.0015, cap=8) == 8
    assert drain_budget(0.014, 0.0015, cap=5) == 5
    # 5 ms spare, 2 ms margin: one 2.5 ms frame fits, two do not
    assert drain_budget(0.005, 0.0025) == 1
    assert drain_budget(0.0049, 0.0025) == 1
    assert drain_budget(0.0044, 0.0025) == 0
    # nothing fits inside the margin, and a negative spare drains nothing
    assert drain_budget(0.002, 0.0001) == 0
    assert drain_budget(-0.003, 0.001) == 0
    # the old fixed threshold: 15 ms spare used to be REQUIRED; now a slot
    # with the typical ~14 ms spare at a 16.67 ms tick still drains
    assert drain_budget(0.0140, 0.003) == 3
    assert drain_budget(0.0140, 0.006) == 2


def test_drain_budget_falls_back_to_the_prior_without_a_measurement():
    from jarvis.ui.avatar_clock import CONV_PRIOR_S
    assert drain_budget(0.014, 0.0) == drain_budget(0.014, CONV_PRIOR_S)
    assert drain_budget(0.014, -1.0) == drain_budget(0.014, CONV_PRIOR_S)
    assert drain_budget(0.014, 0.0) >= 1


def test_conv_cost_is_the_max_of_the_last_16_samples():
    from jarvis.ui.avatar_clock import CONV_PRIOR_S
    cc = ConvCost()
    assert cc.estimate == CONV_PRIOR_S and cc.n == 0
    cc.add(0.001)
    assert cc.estimate == 0.001
    cc.add(0.004)                    # one slow frame shrinks the budget …
    assert cc.estimate == 0.004
    for _ in range(16):              # … until it falls out of the window
        cc.add(0.001)
    assert cc.estimate == 0.001 and cc.n == 16
    cc.add(-5.0)                     # a clock glitch never goes negative
    assert cc.estimate == 0.001


def test_tier_upgrade_is_not_stalled_by_speed_2_entered_on_an_odd_index():
    # thinking/speaking advance 2 frames per slot; entered on an odd index
    # the index stays odd, so a crossing that demanded idx % 24 == 0
    # exactly never came — the sphere sat on the 25-frame tier (5 fps at
    # speed 2) with all 600 frames installed until the state changed
    c = _clock()
    c.set_available_step(24)
    t = c.slot_time(7)
    assert c.display_index(t) == 0 and c.index_at(t) == 7
    c.set_speed(2, t)
    c.set_available_step(1)
    crossed_at = None
    for k in range(7, 7 + 24):                   # one coarse hold at speed 2
        idx = c.display_index(c.slot_time(k))
        assert idx % c.active_step == 0
        if c.active_step == 1:
            crossed_at = (k, c.index_at(c.slot_time(k)))
            break
    assert crossed_at is not None and c.active_step == 1
    # it crossed on the slot where the coarse tier would have swapped
    # anyway (index just past a multiple of 24), one frame off the shared
    # phase angle at most
    assert crossed_at[1] % 24 < 2
    # at speed 1 the rule is unchanged: exactly on the shared frame
    c = _clock()
    c.set_available_step(24)
    c.display_index(c.slot_time(5))
    c.set_available_step(1)
    for k in range(5, 40):
        before = c.active_step
        c.display_index(c.slot_time(k))
        if before != c.active_step:
            assert c.index_at(c.slot_time(k)) % 24 == 0
    assert c.active_step == 1
    # standby (half speed) reaches every integer index, so it crosses too
    c = _clock(speed=0.5)
    c.set_available_step(24)
    c.display_index(c.slot_time(3))
    c.set_available_step(1)
    for k in range(3, 60):
        c.display_index(c.slot_time(k))
    assert c.active_step == 1


def test_conv_cost_outlier_expires_by_slot_age_not_only_by_new_samples():
    # 14 ms spare, 2 ms margin -> 12 ms of room. One 13 ms sample (the
    # first frame of a generation under load) used to pin the max-of-16
    # estimate until 16 MORE frames were drained — and with budget 0 the
    # only drains were the starvation fallback's 1 per 10 slots: 160
    # slots (2.7 s) of a 6 fps bake. Age expiry evicts it in `keep` slots.
    cc = ConvCost(keep=16)
    cc.add(0.013)
    assert drain_budget(0.014, cc.estimate) == 0
    slots = 0
    while drain_budget(0.014, cc.estimate) == 0:
        cc.tick()
        slots += 1
        assert slots <= 16
    assert slots == 16                           # gone after `keep` slots
    assert cc.n == 0 and cc.estimate == cc.prior_s
    # with typical frames still landing, the outlier also ages out while
    # the newer samples stay
    cc = ConvCost(keep=16)
    cc.add(0.013)
    for _ in range(17):
        cc.tick()
        cc.add(0.0018)
    assert cc.estimate == 0.0018 and drain_budget(0.014, cc.estimate) == 3
    # reset (a new generation) drops everything back to the prior
    cc.reset()
    assert cc.n == 0 and cc.estimate == cc.prior_s
