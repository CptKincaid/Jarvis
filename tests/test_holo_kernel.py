"""Display-free tests for the holo sphere kernel (jarvis/ui/holo_kernel.py)
and the pure geometry the holo HUD stage in reactor.py is built on.

The kernel is FROZEN (judge3 panel wf_b0295178-ce1, 3-0): the hash below
is the fingerprint of the judged art at a size small enough to render in
~10 ms. A different hash means the sphere changed and needs a new judge
round -- update it only with that round's verdict in the commit message.
"""
import hashlib
import os

import numpy as np
import pytest

from jarvis.ui import reactor
from jarvis.ui.avatar_bake import hex_rgb, pool_ground, pool_shade
from jarvis.ui.holo_kernel import HoloKernel

BG, CYAN = hex_rgb("0d1b2a"), hex_rgb("35e0ff")
BG2 = hex_rgb("050b14")            # the holo look's stage ground
SIZE, SUP, N = 64, 1, 300
POOL = (0.22, 200.0)

# sha256 of frame(0) bytes at SIZE/SUP with the constants above, recorded
# 2026-09-01 from two separate processes (it is the same in both -- all
# randomness is one seeded default_rng consumed in a fixed order)
FRAME0_SHA256 = ("3c837a8ecee4311c73fbcf15024900ad"
                 "31cb2d8e0f6342426a1785fc94944c91")


@pytest.fixture(scope="module")
def kernel():
    return HoloKernel(SIZE, SUP, N, POOL, BG, CYAN)


def _arr(img):
    return np.asarray(img)


def test_loop_is_bitwise_seamless(kernel):
    a = _arr(kernel.frame(0))
    assert a.shape == (SIZE, SIZE, 3) and a.dtype == np.uint8
    assert np.array_equal(a, _arr(kernel.frame(N)))
    assert np.array_equal(_arr(kernel.frame(7)), _arr(kernel.frame(N + 7)))
    # and the loop actually moves
    assert not np.array_equal(a, _arr(kernel.frame(1)))


def test_frame0_is_the_judged_art(kernel):
    digest = hashlib.sha256(_arr(kernel.frame(0)).tobytes()).hexdigest()
    assert digest == FRAME0_SHA256, (
        "holo sphere changed -- a new judge round, not a tweak")


def test_rendering_is_deterministic_across_instances():
    a = _arr(HoloKernel(SIZE, SUP, N, POOL, BG, CYAN).frame(11))
    b = _arr(HoloKernel(SIZE, SUP, N, POOL, BG, CYAN).frame(11))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bg", [BG, BG2])
def test_outer_border_is_the_analytic_pool_ground(bg):
    """The outer 3 % of the square is pure ground: byte-equal to
    avatar_bake.pool_ground at OUTPUT resolution, whatever the bake's sup.
    That equality is what lets the reactor backdrop hide the square."""
    ground = pool_ground(SIZE, 1, POOL, bg, CYAN)
    b = max(1, int(round(SIZE * 0.03)))
    edge = np.ones((SIZE, SIZE), dtype=bool)
    edge[b:-b, b:-b] = False
    for sup in (1, 2):
        k = HoloKernel(SIZE, sup, N, POOL, bg, CYAN)
        for i in (0, 5, 150):
            a = _arr(k.frame(i))
            assert np.array_equal(a[edge], ground[edge]), (sup, i)


def test_palette_is_purple_free(kernel):
    """Blue LUT + white-hot core, nothing else: red never exceeds green or
    blue in any pixel (a magenta/purple cast is R > B or R > G)."""
    for i in (0, 37, 149, 222):
        a = _arr(kernel.frame(i)).astype(np.int16)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        assert np.all(r <= g), i
        assert np.all(r <= b), i
    # the LUT still reaches white at the core
    a = _arr(kernel.frame(0))
    assert a.max() == 255


# ------------------------------------------------ the shared ground function
def _pre_overhaul_pool_ground(size, sup, pool, bg_rgb, cyan_rgb):
    """The 08-31 pool_ground verbatim -- pool_shade must keep these ops in
    this order or the classic frames stop hashing to their old bytes."""
    peak, rp = pool
    S2 = size * sup
    c = S2 // 2
    y, x = np.ogrid[-c:S2 - c, -c:S2 - c]
    d = np.sqrt((x * x + y * y).astype(np.float32)) / sup
    pf = peak * np.clip(1.0 - d / rp, 0.0, 1.0) ** 2
    out = np.empty((S2, S2, 3), dtype=np.uint8)
    for ch in range(3):
        base = bg_rgb[ch]
        out[..., ch] = (base + (cyan_rgb[ch] - base) * pf).astype(np.uint8)
    return out


@pytest.mark.parametrize("size,sup", [(96, 1), (64, 2)])
@pytest.mark.parametrize("pool", [POOL, (0.22, 832.0)])
def test_pool_ground_keeps_the_pre_overhaul_bytes(size, sup, pool):
    for bg in (BG, BG2):
        assert np.array_equal(pool_ground(size, sup, pool, bg, CYAN),
                              _pre_overhaul_pool_ground(size, sup, pool, bg,
                                                        CYAN))


def test_pool_shade_of_pixel_offsets_matches_pool_ground():
    """What the reactor's holo backdrop does: shade a big stage from the
    integer pixel offset to the square's anchor pixel. The square's own
    ground must be an exact sub-window of it (the seam fix)."""
    w, h, size = 200, 120, 64
    ccx, ccy = 71, 60                      # cluster centre (any ints)
    y, x = np.ogrid[0:h, 0:w]
    dx, dy = x - ccx, y - ccy
    stage = pool_shade(np.sqrt((dx * dx + dy * dy).astype(np.float32)),
                       POOL, BG2, CYAN)
    assert stage.shape == (h, w, 3) and stage.dtype == np.uint8
    x0, y0 = ccx - size // 2, ccy - size // 2   # Tk anchor="center"
    window = stage[y0:y0 + size, x0:x0 + size]
    assert np.array_equal(window, pool_ground(size, 1, POOL, BG2, CYAN))


# --------------------------------------- holo HUD geometry (reactor.py, pure)
def test_tracked_caps_spaces_out_letters():
    assert reactor.tracked("LOAD") == "L O A D"
    assert reactor.tracked(" hear ") == "h e a r"
    assert reactor.tracked("") == ""


def test_read_load_fraction_reads_loadavg_per_core(tmp_path):
    p = tmp_path / "loadavg"
    p.write_text("2.50 1.10 0.90 3/1234 5678\n")
    assert reactor.read_load_fraction(str(p), ncpu=10) == pytest.approx(0.25)
    # clamped to the dial's range
    assert reactor.read_load_fraction(str(p), ncpu=1) == 1.0
    p.write_text("garbage\n")
    assert reactor.read_load_fraction(str(p), ncpu=4) is None
    assert reactor.read_load_fraction(str(tmp_path / "missing"), 4) is None
    # the real file, if this box has one, gives a number in range
    if os.path.exists(reactor.LOADAVG_PATH):
        v = reactor.read_load_fraction()
        assert v is not None and 0.0 <= v <= 1.0


def test_gauge_angle_spans_the_270_degree_sweep():
    assert reactor.gauge_angle(0.0) == pytest.approx(reactor.GAUGE_START)
    assert reactor.gauge_angle(0.5) == pytest.approx(
        (reactor.GAUGE_START + reactor.GAUGE_SWEEP / 2) % 360.0)
    assert reactor.gauge_angle(1.0) == pytest.approx(
        (reactor.GAUGE_START + reactor.GAUGE_SWEEP) % 360.0)
    # None (no reading) and out-of-range values pin to the ends
    assert reactor.gauge_angle(None) == reactor.gauge_angle(0.0)
    assert reactor.gauge_angle(-3.0) == reactor.gauge_angle(0.0)
    assert reactor.gauge_angle(7.0) == reactor.gauge_angle(1.0)


def test_needle_stays_inside_the_dial():
    gx, gy, r = 100.0, 80.0, 26
    for v in (None, 0.0, 0.25, 0.5, 0.75, 1.0):
        x0, y0, x1, y1 = reactor.needle_xy(gx, gy, r, v)
        d0 = ((x0 - gx) ** 2 + (y0 - gy) ** 2) ** 0.5
        d1 = ((x1 - gx) ** 2 + (y1 - gy) ** 2) ** 0.5
        assert d0 < d1 < r                  # hub side in, tip short of the arc
    # zero reads at the bottom-left, full at the bottom-right, half at 12
    _, _, tx, ty = reactor.needle_xy(gx, gy, r, 0.0)
    assert tx < gx and ty > gy
    _, _, tx, ty = reactor.needle_xy(gx, gy, r, 1.0)
    assert tx > gx and ty > gy
    _, _, tx, ty = reactor.needle_xy(gx, gy, r, 0.5)
    assert abs(tx - gx) < 1e-6 and ty < gy
