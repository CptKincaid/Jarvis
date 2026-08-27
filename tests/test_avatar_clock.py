"""Display-free tests for the clock-indexed avatar frame scheduler."""
import math

from jarvis.ui.avatar_clock import (TIER_STEPS, AvatarClock, tier_frames,
                                    tier_order)

N, P = 300, 10.0
TICK = P / N


def _clock(speed=1, t0=100.0):
    return AvatarClock(N, P, TICK, speed=speed, now=t0)


def test_frame_constants_match_the_60hz_grid():
    from jarvis.ui import reactor
    assert reactor.AV_FRAMES % 12 == 0            # nested 12/6/3/1 grids
    assert reactor.AV_FRAMES / reactor.AV_PERIOD == 30   # 30 fps = 2 refreshes
    assert math.isclose(reactor.AV_TICK, reactor.AV_PERIOD / reactor.AV_FRAMES)
    assert all(isinstance(v, int) for v in reactor.AV_SPEED.values())


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
    # tiers complete in order: the first 25 entries ARE the step-12 grid
    assert set(order[:N // 12]) == grids[0]
    assert set(order[:N // 6]) == grids[1]
    assert set(order[:N // 3]) == grids[2]


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
    for k in range(14, 901):
        lc.observe(c.slot_time(k) + 0.001, k)
    now = c.slot_time(900) + 0.001                          # 30 s elapsed
    line = lc.report(now)
    assert line == "avatar: late slots 3/900 (max lateness 67.7 ms)"
    assert lc.late == 0 and lc.max_late_s == 0.0            # window reset
    assert lc.report(now) is None
