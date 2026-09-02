"""holo_kernel — the JARVIS holographic particulate sphere (blue), Tk-free.

Provenance (2026-09-01): the FROZEN ship kernel from the blue-holographic
overhaul. Judge3 panel wf_b0295178-ce1 picked v3_dial 3-0 over v2 /
v3_burn; this is v3_dial with the v4 constants (HOT_GAIN 0.55 — 1.7
clipped ~1300 wall px to flat white, judged "confetti"; rail luminance
150/100/112; great-circle radii 0.955/0.930/0.900). Contract, measured at
392/sup2 n=600: 0.040 s/frame, churn 4.83 %, seamless (frame(n) ==
frame(0) bitwise), deterministic across processes, outer 3 % border
bit-exact to avatar_bake.pool_ground. Verify with
`cd scratchpad/holo/judge3 && python render_all.py v4`. Constants below are
the judged values — a change here is a new judge round, not a tweak.

The theme of the kernel is STRUCTURE AND CALM: an instrument you can read
from across the room that never shimmers.
  * ONE crisp ruler at 0.50 R: full baseline ring + 60 ticks (major every
    3rd) + 20 outer blocks.  It is perfectly 20-fold periodic, so it may
    turn 3/20 of a turn per loop and still close seamlessly; a highlight
    sweeps against its spin at 1 turn.
  * The outer wall is built 2-fold symmetric (every primitive duplicated
    at +180 deg, even-harmonic fray) so it spins 1/2 turn per loop.  Three
    distinct thin rails (0.80 / 0.865 / 0.93 R) read through the rag;
    clusters snap to them.
  * The heavy side is a luminance ENVELOPE evaluated per frame (a static
    heavy sector at the lower-left like the film plus a softer highlight
    sweeping against the spin), so the pattern stays symmetric and the
    heavy side is calm.
  * HOT PLATE: glyphs of the top luminance tier, node dots, hot arcs and
    the bright arc pieces are drawn a second time into a separate plate
    that is added with HOT_GAIN before the tone map and fed into the
    tight bloom, so hot clumps burn through to white with tight halos.
  * LUT desaturates to white from t = 0.76 (ice) to 0.90 (white).
  * Core: centre hits 255, soft halo widened toward 0.2 R; the dark
    interstitial annulus is preserved (exp tone map + thresholded bloom).

Loop: every motion is phase = 2*pi*k/n with k reduced mod n.  Bands whose
pattern has an m-fold rotational symmetry may use a multiplier that is an
integer multiple of 1/m turn: the geometry after one loop is the same set
of primitives, so the loop closes visually; frame(n) == frame(0) bitwise
because k is reduced mod n.  All randomness comes from one
np.random.default_rng(seed) consumed in a fixed order in __init__.

Ground: art light -> exp tone map -> blue LUT (derived from cyan_rgb) ->
LANCZOS downscale -> ADDED to the exact analytic pool ground
(avatar_bake.pool_ground, at OUTPUT resolution) — so the border is the
stage's ground to the byte, whatever `sup` is.  A radial mask forces the
light to zero before the outer 3% border, which is therefore pure ground.
Pure numpy + PIL; the only jarvis import is avatar_bake (itself Tk-free
and app-free, tested), and that one is what makes the seam exact.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from jarvis.ui.avatar_bake import pool_ground

TAU = 2.0 * math.pi
ART_R = 0.43          # sphere radius as a fraction of the square
TILT_DEG = 22.0       # spin axis tilt (up) away from the view direction
ROLL_DEG = -8.0       # gentle screen roll: perspective, not a lean
EDGE_IN = 0.440       # radial fraction of the square where art starts fading
EDGE_OUT = 0.4655     # ... and is zero (border ring beyond is pure ground)
Z_SPLIT = -0.12       # unit-R depth below which primitives go to the back plate
TONE_GAIN = 1.25      # t = 1 - exp(-TONE_GAIN * L / 255)
SHELL_R = 0.935       # spark shell radius (the visible ball silhouette)
HOT_GAIN = 0.55       # hot plate gain before the tone map (1.7 clipped
                      # ~1300 wall px to flat white — judged "confetti")
HEAVY = 0.75 * math.pi   # static heavy sector (disc angle; lower-left)

WALL_RINGS = (0.800, 0.865, 0.930)   # three distinct thin rails
WALL_SYM, WALL_M = 2, 0.5            # 2-fold symmetric, 1/2 turn per loop
WAVY_SYM, WAVY_M = 2, -0.5
RULER_R, RULER_N, RULER_TURNS = 0.50, 60, 3   # 60 ticks, 20-fold, 3/20 turn
DASH_R, DASH_N, DASH_TURNS = 0.34, 36, -5     # 36 dashes, 36-fold, -5/36 turn
DIAL_R, DIAL_N, DIAL_M = 0.205, 24, 1.0       # 24 ticks, +1 turn


# --------------------------------------------------------------- helpers
def _blend(c1: tuple, c2: tuple, f: float) -> tuple:
    return tuple(int(a + (b - a) * f) for a, b in zip(c1, c2))


def lut(cyan_rgb: tuple):
    """256-entry added-light ramp black -> deep blue -> CYAN -> ice -> WHITE,
    derived from the accent so a palette change re-derives the art.  The
    top of the ramp desaturates to white early (0.76 ice, 0.90 white) so
    hot clumps burn through instead of plateauing at flat neon cyan."""
    cy = tuple(int(v) for v in cyan_rgb)
    deep = tuple(int(v * fct) for v, fct in zip(cy, (0.10, 0.24, 0.62)))
    mid = _blend(deep, cy, 0.55)
    ice = _blend(cy, (255, 255, 255), 0.50)
    anchors = ((0.0, (0, 0, 0)), (0.28, deep), (0.48, mid), (0.62, cy),
               (0.76, ice), (0.90, (255, 255, 255)), (1.0, (255, 255, 255)))
    ts = np.array([a[0] for a in anchors])
    t = np.linspace(0.0, 1.0, 256)
    table = np.stack([np.interp(t, ts, [a[1][ch] for a in anchors])
                      for ch in range(3)], axis=1)
    table = table.astype(np.int16)
    table[0] = 0
    return table


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _depth(Z):
    """Brightness factor vs unit-R depth (front +1 .. back -1): 0.26..1.0."""
    return 0.26 + 0.74 * _smoothstep((np.asarray(Z, dtype=np.float64)
                                      + 0.75) / 1.5)


def _fray(th, fph, harm=(3, 5, 7)):
    """Azimuthal fray shared by every row of a band.  Odd harmonics for
    1-fold bands, even harmonics for the 2-fold symmetric bands."""
    th = np.asarray(th, dtype=np.float64)
    return (0.7 * np.sin(harm[0] * th + fph[0])
            + 0.8 * np.sin(harm[1] * th + fph[1])
            + 0.6 * np.sin(harm[2] * th + fph[2]))


def _pick(rng, values, probs, n):
    return np.asarray(values, dtype=np.float64)[
        rng.choice(len(values), size=n, p=probs)]


def _runs(mask):
    """Index ranges [s, e) of consecutive True in a boolean array."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _segments(rng, n_seg: int, cover: float, min_frac: float = 0.4,
              span: float = TAU):
    """Split an arc of length `span` into n_seg segments totalling `cover`
    of it, separated by gaps. Returns list of (a, b) radians."""
    seg_w = rng.random(n_seg) * (1.0 - min_frac) + min_frac
    seg_w = seg_w / seg_w.sum() * (span * cover)
    gap_w = rng.random(n_seg) * 0.7 + 0.3
    gap_w = gap_w / gap_w.sum() * (span * (1.0 - cover))
    a = rng.random() * span
    out = []
    for i in range(n_seg):
        out.append((a, a + seg_w[i]))
        a += seg_w[i] + gap_w[i]
    return out


def _in_segs(rng, segs, n):
    """n random angles inside the segments, weighted by segment length."""
    lens = np.array([b - a for a, b in segs])
    starts = np.array([a for a, b in segs])
    idx = rng.choice(len(segs), size=n, p=lens / lens.sum())
    return starts[idx] + rng.random(n) * lens[idx]


class _Prims:
    """Accumulates band primitives in band coordinates (angle, radius).
    Every primitive carries its spin multiplier m, an envelope weight e
    (how much the heavy-side / sweep envelope modulates it) and a hot flag
    (also drawn into the hot plate)."""

    KEYS_T = ("th", "r0", "r1", "l", "m", "w", "e", "h")
    KEYS_D = ("th", "hw", "r", "l", "m", "w", "e", "h")
    KEYS_B = ("th", "hw", "r0", "r1", "l", "m", "e")

    def __init__(self):
        self.t = {k: [] for k in self.KEYS_T}   # radial ticks
        self.d = {k: [] for k in self.KEYS_D}   # tangential dashes
        self.b = {k: [] for k in self.KEYS_B}   # quads
        self.base = []       # (th_array, r_array, lum, mult, width, e, hot)
        self.dots = []       # (th, r, rad_px, lum, mult, e, hot)
        self.hot = []        # (th_array, r, lum, mult, e)

    def tick(self, th, r0, r1, lum, m, w, e=0.0, hot=False):
        for k, v in zip(self.KEYS_T, (th, r0, r1, lum, m, int(w), e, bool(hot))):
            self.t[k].append(v)

    def dash(self, th, hw, r, lum, m, w, e=0.0, hot=False):
        for k, v in zip(self.KEYS_D, (th, hw, r, lum, m, int(w), e, bool(hot))):
            self.d[k].append(v)

    def block(self, th, hw, r0, r1, lum, m, e=0.0):
        for k, v in zip(self.KEYS_B, (th, hw, r0, r1, lum, m, e)):
            self.b[k].append(v)

    def marks(self):
        return (len(self.t["th"]), len(self.d["th"]), len(self.b["th"]),
                len(self.base), len(self.dots), len(self.hot))

    def symmetrize(self, marks, folds):
        """Duplicate every primitive appended since `marks` at
        +j*TAU/folds, j = 1..folds-1 -> folds-fold rotational symmetry."""
        nt, nd, nb, nbase, ndots, nhot = marks
        ends = (len(self.t["th"]), len(self.d["th"]), len(self.b["th"]),
                len(self.base), len(self.dots), len(self.hot))
        for j in range(1, folds):
            off = j * TAU / folds
            for store, n0, n1 in ((self.t, nt, ends[0]), (self.d, nd, ends[1]),
                                  (self.b, nb, ends[2])):
                for kk, lst in store.items():
                    if kk == "th":
                        lst.extend([v + off for v in lst[n0:n1]])
                    else:
                        lst.extend(lst[n0:n1])
            for tup in self.base[nbase:ends[3]]:
                self.base.append((tup[0] + off,) + tup[1:])
            for tup in self.dots[ndots:ends[4]]:
                self.dots.append((tup[0] + off,) + tup[1:])
            for tup in self.hot[nhot:ends[5]]:
                self.hot.append((tup[0] + off,) + tup[1:])


# ---------------------------------------------------------------- kernel
class HoloKernel:
    """Everything a frame needs besides k — built once per bake."""

    def __init__(self, size: int, sup: int, n_frames: int, pool: tuple,
                 bg_rgb: tuple, cyan_rgb: tuple, seed: int = 7):
        self.size, self.sup, self.n = int(size), int(sup), int(n_frames)
        self.S2 = self.size * self.sup
        self.bscale = self.S2 / 784.0          # bloom radii follow the art size
        self.c = self.S2 / 2.0
        self.R = self.S2 * ART_R
        self.sin_t = math.sin(math.radians(TILT_DEG))
        self.cos_t = math.cos(math.radians(TILT_DEG))
        self.sin_r = math.sin(math.radians(ROLL_DEG))
        self.cos_r = math.cos(math.radians(ROLL_DEG))
        self.table = lut(cyan_rgb)
        # ground at OUTPUT resolution: the art is downscaled first, then
        # added, so the border is exactly the stage's analytic pool
        self.ground16 = pool_ground(self.size, 1, pool, bg_rgb,
                                    cyan_rgb).astype(np.int16)
        self._build_static()
        rng = np.random.default_rng(seed)
        self._build_bands(rng)
        self._build_rings(rng)
        self._build_strands(rng)
        self._build_sparks(rng)
        self._build_knot(rng)

    # ------------------------------------------------------------ static
    def _build_static(self):
        S2, c, R = self.S2, self.c, self.R
        y, x = np.ogrid[0:S2, 0:S2]
        dx = (x - c + 0.5).astype(np.float32)
        dy = (y - c + 0.5).astype(np.float32)
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2)
        # analytic core: white-hot knot (breathes) + a soft halo out to
        # ~0.2 R + faint haze.  Centre L ~ 700 -> t 0.97 -> white.
        tight = np.zeros((S2, S2), dtype=np.float32)
        for amp, rf in ((360.0, 0.030), (170.0, 0.070)):
            tight += amp * np.exp(-d2 / (R * rf) ** 2)
        wide = np.zeros((S2, S2), dtype=np.float32)
        for amp, rf in ((85.0, 0.14), (55.0, 0.24), (18.0, 0.40), (5.0, 0.80)):
            wide += amp * np.exp(-d2 / (R * rf) ** 2)
        self.glow_tight = tight
        self.glow_wide = wide
        # edge mask: 1 inside EDGE_IN, 0 beyond EDGE_OUT (pure ground)
        m = 1.0 - _smoothstep((d / S2 - EDGE_IN) / (EDGE_OUT - EDGE_IN))
        self.mask = m.astype(np.float32)
        # output-resolution guard ring (belt and braces for the border)
        s = self.size
        yy, xx = np.ogrid[0:s, 0:s]
        dd = np.sqrt((xx - s / 2.0 + 0.5) ** 2 + (yy - s / 2.0 + 0.5) ** 2)
        self.guard = (dd / s) >= EDGE_OUT

    # -------------------------------------------------------- projection
    def _screen(self, x, y):
        """Unit-R screen-plane coords (after tilt) -> rolled pixel coords."""
        X = self.c + (x * self.cos_r - y * self.sin_r) * self.R
        Y = self.c + (x * self.sin_r + y * self.cos_r) * self.R
        return X, Y

    def _proj_xy(self, bx, by):
        """Equatorial-disc cartesian point -> screen X, Y (px), depth Z."""
        X, Y = self._screen(bx, by * self.cos_t)
        return X, Y, by * self.sin_t

    def _proj_disc(self, th, r):
        return self._proj_xy(r * np.cos(th), r * np.sin(th))

    def _proj_body(self, bu, bv, ba):
        X, Y = self._screen(bu, bv * self.cos_t - ba * self.sin_t)
        return X, Y, bv * self.sin_t + ba * self.cos_t

    def _proj_sph(self, rho, lat, th):
        cl = np.cos(lat)
        return self._proj_body(rho * cl * np.cos(th), rho * cl * np.sin(th),
                               rho * np.sin(lat))

    # ------------------------------------------------------------- bands
    def _cluster_band(self, rng, P, rings, m, e, n_seg, cover, n_cl, tpc,
                      fph, thr, harm, style, lsc, span):
        """replica-style clumped band torn by the fray: glyph clusters
        snapping to the band's rails, loose ticks, bright arc pieces and
        rails.  Angles live in [0, span); the caller symmetrizes."""
        sup = self.sup
        r0, r1 = rings[0], rings[-1]
        bw = r1 - r0
        rings_a = np.asarray(rings, dtype=np.float64)
        segs = _segments(rng, n_seg, cover, 0.5, span)
        cand = _in_segs(rng, segs, n_cl * 2)
        ok = _fray(cand, fph, harm) >= thr
        centres = cand[ok][:n_cl]
        for cth in centres.tolist():
            k = int(rng.integers(tpc[0], tpc[1] + 1))
            sig = rng.uniform(0.012, 0.040)
            th = cth + rng.normal(0.0, sig, k)
            rr = rng.uniform(r0, r1, k)
            snap = rng.random(k) < 0.55
            rr = np.where(snap, rings_a[rng.integers(0, rings_a.size, k)], rr)
            kind = rng.choice(5, size=k, p=(0.27, 0.34, 0.15, 0.18, 0.06))
            lum = _pick(rng, (50, 85, 135, 205, 250),
                        (0.28, 0.30, 0.20, 0.13, 0.09), k) * lsc
            lum = np.minimum(lum, 255.0)
            lr = rng.uniform(0.08, 0.55, k) ** 1.2 * bw
            lr = np.where(rng.random(k) < 0.04, lr * 1.6, lr)
            lt = rng.uniform(0.006, 0.040, k) ** 1.2 * 1.6 / np.maximum(rr, 0.05)
            drad = rng.uniform(1.0, 1.6, k) * sup
            rung_j = rng.integers(0, max(1, rings_a.size - 1), k)
            for i in range(k):
                kd = int(kind[i])
                hot = bool(lum[i] >= 200.0)
                w = sup + 1 + (1 if hot else 0)
                if kd == 0:                                  # radial tick
                    P.tick(th[i], rr[i] - lr[i] / 2, rr[i] + lr[i] / 2,
                           lum[i], m, w, e, hot)
                elif kd == 1:                                # tangential dash
                    P.dash(th[i], lt[i] / 2, rr[i], lum[i], m, w, e, hot)
                elif kd == 2:                                # L bracket
                    P.tick(th[i], rr[i] - lr[i] / 2, rr[i] + lr[i] / 2,
                           lum[i], m, w, e, hot)
                    P.dash(th[i] + lt[i] / 2, lt[i] / 2, rr[i] + lr[i] / 2,
                           lum[i], m, w, e, hot)
                elif kd == 3:                                # node dot
                    P.dots.append((th[i], rr[i], drad[i], min(240.0, lum[i]),
                                   m, e, hot))
                else:                                        # rung
                    j = int(rung_j[i])
                    ra, rb = (rings_a[j], rings_a[j + 1]) if rings_a.size > 1 \
                        else (r0, r1)
                    P.tick(th[i], ra, rb, lum[i] * 0.75, m, sup, e, False)
        # loose ticks between clusters
        n_loose = n_cl * 2
        th = _in_segs(rng, segs, n_loose)
        rr = rng.uniform(r0, r1, n_loose)
        lum = _pick(rng, (70, 105, 150), (0.5, 0.35, 0.15), n_loose) * lsc
        lr = rng.uniform(0.2, 0.7, n_loose) * bw
        rad = rng.random(n_loose) < 0.5
        lt = rng.uniform(0.008, 0.03, n_loose) / np.maximum(rr, 0.05)
        keep = _fray(th, fph, harm) >= thr
        for i in range(n_loose):
            if not keep[i]:
                continue
            if rad[i]:
                P.tick(th[i], rr[i] - lr[i] / 2, rr[i] + lr[i] / 2, lum[i],
                       m, sup, e, False)
            else:
                P.dash(th[i], lt[i] / 2, rr[i], lum[i], m, sup, e, False)
        # bright arc-dashes (the film's bright arc pieces) — hot tier
        n_arcd = int(rng.integers(3, 6))
        ath = _in_segs(rng, segs, n_arcd)
        asp = rng.uniform(math.radians(5), math.radians(18), n_arcd)
        arr = rng.uniform(r0, r1, n_arcd)
        alum = rng.uniform(175, 240, n_arcd) * lsc
        for i in range(n_arcd):
            n = max(4, int(asp[i] / math.radians(2.0)) + 1)
            t = ath[i] + np.linspace(0.0, asp[i], n)
            P.base.append((t, np.full(n, arr[i]), alum[i], m, sup + 1, e,
                           bool(alum[i] >= 200.0)))
        # rails
        step = math.radians(1.5)

        def rail(rad_fn, lum, w, thr_rail):
            for a, b in segs:
                n = max(2, int((b - a) / step) + 1)
                t = a + np.linspace(0.0, b - a, n)
                keep = _fray(t, fph, harm) >= thr_rail
                for s, e_ in _runs(keep):
                    if e_ - s >= 2:
                        P.base.append((t[s:e_], rad_fn(t[s:e_]), lum, m, w,
                                       e * 0.5, False))

        if style == "rings":
            # three distinct thin rails read through the rag
            for rr_, rl in zip(rings, (150.0, 100.0, 112.0)):   # outer/middle
                # dimmer so the three rails read as ONE wall with depth
                rail(lambda t, rr_=rr_: np.full(t.shape, rr_), rl * lsc,
                     sup + 1, thr - 1.3)
            # rungs between adjacent rails
            n_rung = int(span * cover / math.radians(7.0))
            rth = _in_segs(rng, segs, n_rung)
            rkeep = _fray(rth, fph, harm) >= thr
            rj = rng.integers(0, rings_a.size - 1, n_rung)
            for i in range(n_rung):
                if rkeep[i]:
                    P.tick(rth[i], rings_a[rj[i]], rings_a[rj[i] + 1],
                           62.0 * lsc, m, sup, e, False)
        elif style == "wavy":
            amp = rng.uniform(0.014, 0.022)
            fq = 2 * int(rng.integers(5, 7))          # even: 2-fold symmetric
            ph0 = rng.uniform(0.0, TAU)
            rm = (r0 + r1) / 2
            rail(lambda t: rm + amp * np.sin(fq * t + ph0), 190.0 * lsc,
                 sup + 1, thr - 0.6)
            rail(lambda t: np.full(t.shape, r0), 72.0 * lsc, sup, thr - 0.3)
        return segs

    def _micro_grain(self, rng, P, rho, mult, lum, segs, fph, thr, harm, e,
                     sig_r, hot_n):
        """Micro marks (tiny ticks/dashes into P) and grain speckles (into
        the splat lists) clustered along a ragged band; a few hot arcs."""
        g_th, g_r, g_l = [], [], []
        for a, b in segs:
            # micro marks
            n = int((b - a) * rho * 44)
            th = a + rng.random(n) * (b - a)
            r = rho + rng.normal(0.0, 0.03 if rho > 0.6 else 0.02, n)
            l_ = lum * rng.uniform(0.35, 0.85, n)
            keep = _fray(th, fph, harm) >= thr
            is_t = rng.random(n) < 0.5
            h = rng.uniform(0.006, 0.02, n)
            dhw = np.radians(rng.uniform(0.3, 0.9, n))
            for i in range(n):
                if not keep[i]:
                    continue
                if is_t[i]:
                    P.tick(th[i], r[i] - h[i] * 0.5, r[i] + h[i] * 0.5,
                           l_[i], mult, self.sup, e, False)
                else:
                    P.dash(th[i], dhw[i], r[i], l_[i], mult, self.sup, e,
                           False)
            # grain
            n = int((b - a) * rho * 120)
            nu = n // 2
            th_u = a + rng.random(nu) * (b - a)
            ncl = max(2, int((b - a) / math.radians(9)))
            centers = a + rng.random(ncl) * (b - a)
            th_c = rng.choice(centers, n - nu) \
                + rng.normal(0.0, math.radians(1.6), n - nu)
            th = np.concatenate((th_u, th_c))
            r = rho + rng.normal(0.0, sig_r, n)
            l_ = (rng.uniform(45, 140, n) * (rng.random(n) < 0.85)
                  + rng.uniform(170, 250, n) * (rng.random(n) < 0.07))
            keep = _fray(th, fph, harm) >= thr - 0.2
            g_th.append(th[keep])
            g_r.append(r[keep])
            g_l.append(l_[keep])
            # hot arcs: short bright pieces that burn white
            for _ in range(int(rng.integers(1, 3)) + hot_n):
                a0 = a + rng.random() * (b - a)
                span = math.radians(rng.uniform(1.5, 7.0))
                ths = np.linspace(a0, min(b, a0 + span), 6)
                P.hot.append((ths, rho + rng.uniform(-0.03, 0.03),
                              rng.uniform(215, 255), mult, e))
        return (np.concatenate(g_th), np.concatenate(g_r),
                np.concatenate(g_l))

    def _build_bands(self, rng):
        """Concentric bands in the equatorial disc:
             wall  0.80/0.865/0.93 R  +1/2 turn  2-fold ragged wall, 3 rails
             wavy  0.655-0.735 R      -1/2 turn  2-fold wavy rail + clusters
             ruler 0.50 R             +3/20      ONE crisp periodic ruler
             dash  0.34 R             -5/36      faint periodic dashed ring
             dial  0.205 R            +1         24-tick core dial"""
        sup = self.sup
        P = _Prims()
        g_th, g_r, g_l, g_m = [], [], [], []

        # ---- wall (outer): 2-fold symmetric, three rails, heavy envelope
        e_wall = 0.60
        fph = rng.uniform(0.0, TAU, 3)
        harm = (2, 6, 10)
        thr = -0.85
        mk = P.marks()
        segs = self._cluster_band(rng, P, WALL_RINGS, WALL_M, e_wall, 3,
                                  0.90, 105, (5, 12), fph, thr, harm,
                                  "rings", 1.05, TAU / WALL_SYM)
        gt, gr, gl = self._micro_grain(rng, P, 0.865, WALL_M, 170.0, segs,
                                       fph, thr, harm, e_wall, 0.045, 2)
        P.symmetrize(mk, WALL_SYM)
        for j in range(WALL_SYM):
            g_th.append(gt + j * TAU / WALL_SYM)
            g_r.append(gr)
            g_l.append(gl)
            g_m.append(np.full(gt.size, WALL_M))

        # ---- wavy mid band: 2-fold symmetric, -1/2 turn
        e_wavy = 0.35
        fph = rng.uniform(0.0, TAU, 3)
        thr = -1.05
        mk = P.marks()
        segs = self._cluster_band(rng, P, (0.655, 0.735), WAVY_M, e_wavy, 3,
                                  0.76, 36, (4, 9), fph, thr, harm,
                                  "wavy", 0.95, TAU / WAVY_SYM)
        gt, gr, gl = self._micro_grain(rng, P, 0.695, WAVY_M, 165.0, segs,
                                       fph, thr, harm, e_wavy, 0.03, 0)
        P.symmetrize(mk, WAVY_SYM)
        for j in range(WAVY_SYM):
            g_th.append(gt + j * TAU / WAVY_SYM)
            g_r.append(gr)
            g_l.append(gl)
            g_m.append(np.full(gt.size, WAVY_M))

        # ---- ONE crisp ruler at 0.50 R: 20-fold periodic, highlight sweep
        rho = RULER_R
        m_r = RULER_TURNS / (RULER_N / 3.0)          # 3 major pitches / loop
        e_r = 0.45
        pitch = TAU / RULER_N
        for i in range(RULER_N):
            th = i * pitch
            if i % 3 == 0:
                P.tick(th, rho - 0.012, rho + 0.050, 225.0, m_r, sup + 1, e_r,
                       False)
            else:
                P.tick(th, rho, rho + 0.027, 165.0, m_r, sup, e_r, False)
        n_arc = RULER_N // 3                          # baseline in 20 arcs
        for j in range(n_arc):
            a = j * TAU / n_arc
            ths = a + np.linspace(0.0, TAU / n_arc, 13)
            P.base.append((ths, np.full(ths.size, rho), 165.0, m_r, sup + 1,
                           e_r, False))
            P.block(a + 0.5 * TAU / n_arc, math.radians(1.7), rho + 0.064,
                    rho + 0.078, 105.0, m_r, e_r)
        for j in range(4):                            # 4 node dots (4-fold)
            P.dots.append((j * TAU / 4 + pitch * 1.5, rho + 0.071,
                           1.3 * sup, 240.0, m_r, e_r, True))

        # ---- faint periodic dashed ring at 0.34 R (36-fold), -5/36 turn
        rho = DASH_R
        m_d = DASH_TURNS / float(DASH_N)
        pitch = TAU / DASH_N
        for i in range(DASH_N):
            th = i * pitch
            P.dash(th, pitch * 0.22, rho, 120.0, m_d, sup, 0.3, False)
            if i % 2 == 0:
                P.dash(th + pitch * 0.5, pitch * 0.12, rho - 0.024, 80.0, m_d,
                       sup, 0.3, False)
        for j in range(DASH_N // 2):
            a = j * TAU / (DASH_N // 2)
            ths = a + np.linspace(0.0, TAU / (DASH_N // 2), 7)
            P.base.append((ths, np.full(ths.size, rho + 0.020), 70.0, m_d,
                           sup, 0.3, False))

        # ---- core dial at 0.205 R (+1 turn): 24 ticks + baseline + 4 dots
        rho = DIAL_R
        pitch = TAU / DIAL_N
        for i in range(DIAL_N):
            th = i * pitch
            P.tick(th, rho, rho + (0.036 if i % 2 == 0 else 0.020),
                   230.0 if i % 2 == 0 else 180.0, DIAL_M, sup, 0.0, False)
        for j in range(8):
            a = j * TAU / 8
            ths = a + np.linspace(0.0, TAU / 8, 9)
            P.base.append((ths, np.full(ths.size, rho), 200.0, DIAL_M, sup,
                           0.0, False))
        for j in range(4):
            P.dots.append((j * TAU / 4 + pitch * 0.5, rho, 1.3 * sup, 245.0,
                           DIAL_M, 0.0, True))

        f = np.asarray
        t, d, b = P.t, P.d, P.b
        self.t_th, self.t_r0, self.t_r1 = f(t["th"]), f(t["r0"]), f(t["r1"])
        self.t_l, self.t_m, self.t_w = f(t["l"]), f(t["m"]), f(t["w"])
        self.t_e, self.t_h = f(t["e"]), f(t["h"], dtype=bool)
        self.d_th, self.d_hw, self.d_r = f(d["th"]), f(d["hw"]), f(d["r"])
        self.d_l, self.d_m, self.d_w = f(d["l"]), f(d["m"]), f(d["w"])
        self.d_e, self.d_h = f(d["e"]), f(d["h"], dtype=bool)
        self.b_th, self.b_hw = f(b["th"]), f(b["hw"])
        self.b_r0, self.b_r1, self.b_l, self.b_m, self.b_e = \
            f(b["r0"]), f(b["r1"]), f(b["l"]), f(b["m"]), f(b["e"])
        self.base = P.base
        self.band_dots = P.dots
        self.hot = P.hot
        self.hl_phase = rng.random(6) * TAU   # highlight sweeps
        self.heavy2 = rng.random() * TAU      # sweeping-highlight phase
        self.g_th = np.concatenate(g_th)
        self.g_r = np.concatenate(g_r)
        self.g_l = np.concatenate(g_l)
        self.g_m = np.concatenate(g_m)
        self.g_e = np.where(self.g_r > 0.77, e_wall, e_wavy)

    # ------------------------------------------------------------- rings
    def _build_rings(self, rng):
        """3 tilted great circles, precessing together at +1 turn with
        their node azimuths 120 deg apart (they can never bunch), gapped,
        with beads sliding along them; plus 2 dashed latitude parallels."""
        self.rings = []
        n_pts = 180
        t = np.linspace(0.0, TAU, n_pts + 1)
        tm = 0.5 * (t[:-1] + t[1:])
        beta0 = rng.random() * TAU
        for j, (alpha_deg, lum, rad) in enumerate(
                ((62, 205, 0.955), (78, 185, 0.930), (50, 170, 0.900))):
            # radii trimmed (0.985/0.950 before) so no arc pokes past the
            # 0.93 wall rail and grazes the stage edge
            alpha = math.radians(alpha_deg + rng.uniform(-6, 6))
            beta = beta0 + j * TAU / 3.0 + rng.uniform(-0.15, 0.15)
            n = np.array([math.sin(alpha) * math.cos(beta),
                          math.sin(alpha) * math.sin(beta), math.cos(alpha)])
            u = np.cross(n, (0.0, 0.0, 1.0))
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            pts = rad * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
            # broken in a few places like the film's
            gaps = rng.uniform(0.0, TAU, int(rng.integers(2, 5)))
            gw = rng.uniform(0.15, 0.60, gaps.size)
            keep = np.ones(n_pts, bool)
            for g, w_ in zip(gaps, gw):
                dd = np.abs((tm - g + math.pi) % TAU - math.pi)
                keep &= dd > w_ / 2
            nb = int(rng.integers(3, 6))
            beads = {
                "t0": rng.random(nb) * TAU,
                "m": rng.choice((1, 2, -1, -2), nb),
                "rad": rng.uniform(1.1, 2.0, nb) * self.sup,
                "lum": rng.uniform(215, 255, nb),
            }
            self.rings.append({"pts": pts, "mult": 1, "lum": lum, "u": u,
                               "v": v, "rad": rad, "beads": beads,
                               "width": self.sup + 1, "keep": keep,
                               "ca": 0.0})
        # dashed latitude parallels on the sphere surface (the globe read)
        for lat_deg, lum in ((38.0, 72), (-38.0, 50)):
            lat = math.radians(lat_deg)
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 1.0, 0.0])
            rad = math.cos(lat)
            pts = rad * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
            pts[:, 2] = math.sin(lat)
            keep = (np.arange(n_pts) % 4) < 2
            nb = 3
            beads = {
                "t0": rng.random(nb) * TAU,
                "m": rng.choice((1, 2, -1), nb),
                "rad": rng.uniform(1.0, 1.6, nb) * self.sup,
                "lum": rng.uniform(200, 250, nb),
            }
            self.rings.append({"pts": pts, "mult": 1, "lum": lum, "u": u,
                               "v": v, "rad": rad, "beads": beads,
                               "width": max(1, self.sup), "keep": keep,
                               "ca": math.sin(lat)})

    # ----------------------------------------------------------- strands
    def _build_strands(self, rng):
        """Long radial data strands (with a circuit jog on ~60%), beads
        and cross ticks."""
        strands = []
        n_str = 9
        base_az = rng.random() * TAU
        for i in range(n_str):
            az = base_az + i * TAU / n_str + rng.uniform(-0.3, 0.3)
            el = rng.uniform(-0.55, 0.55) if i % 3 else rng.uniform(-0.15, 0.15)
            r0 = rng.uniform(0.05, 0.16)
            r1 = rng.uniform(0.55, 0.98)
            mult = -1 if i % 4 == 3 else 1
            # vertices (r, angular offset)
            if rng.random() < 0.6:
                rj = rng.uniform(r0 + 0.12, max(r0 + 0.14, r1 - 0.12))
                ja = rng.choice((-1.0, 1.0)) * rng.uniform(0.03, 0.10) / rj
                verts = [(r0, 0.0), (rj, 0.0), (rj, ja), (r1, ja)]
            else:
                verts = [(r0, 0.0), (r1, 0.0)]
            # densify into short pieces so depth shading varies along it
            rs, offs = [], []
            for (ra, oa), (rb, ob) in zip(verts[:-1], verts[1:]):
                nseg = 5 if ra != rb else 1
                rs.append(np.linspace(ra, rb, nseg + 1)[:-1])
                offs.append(np.linspace(oa, ob, nseg + 1)[:-1])
            rs.append(np.array([verts[-1][0]]))
            offs.append(np.array([verts[-1][1]]))
            rs = np.concatenate(rs)
            offs = np.concatenate(offs)
            nseg = rs.size - 1
            lum = np.linspace(rng.uniform(150, 200), rng.uniform(60, 95), nseg)
            nb = int(rng.integers(2, 5))
            tb = rng.uniform(0.15, 1.0, nb)
            b_r = np.interp(tb, np.linspace(0, 1, rs.size), rs)
            b_o = np.interp(tb, np.linspace(0, 1, rs.size), offs)
            nt = int(rng.integers(3, 7))
            tt = rng.uniform(0.1, 1.0, nt)
            strands.append({
                "az": az, "el": el, "rs": rs, "offs": offs, "lum": lum,
                "mult": mult, "b_r": b_r, "b_o": b_o,
                "b_rad": rng.uniform(1.0, 2.0, nb) * self.sup,
                "b_l": rng.uniform(190, 255, nb),
                "t_r": np.interp(tt, np.linspace(0, 1, rs.size), rs),
                "t_o": np.interp(tt, np.linspace(0, 1, rs.size), offs),
                "t_len": rng.uniform(0.012, 0.03, nt),
                "t_side": rng.choice((-1.0, 1.0), nt),
                "width": self.sup + 1})
        self.strands = strands

    # ------------------------------------------------------------ sparks
    def _build_sparks(self, rng):
        sup = self.sup
        # -- dust: thousands of bilinear-splat points, core-heavy ball
        nd = 5400
        u = rng.random(nd)
        rho = 0.04 + (SHELL_R + 0.02) * u ** 1.3
        disc = rng.random(nd) < 0.6
        lat = np.where(disc, rng.normal(0.0, 0.32, nd),
                       np.arcsin(rng.uniform(-1.0, 1.0, nd)))
        lat = np.clip(lat, -1.5, 1.5)
        th = rng.random(nd) * TAU
        m = rng.choice((1, 1, 1, 1, 1, 1, -1, 2), nd)
        lum = rng.uniform(45, 150, nd) * (1.0 / (0.55 + rho)) ** 0.6
        lum = np.where(rng.random(nd) < 0.07, rng.uniform(190, 250, nd), lum)
        jamp = rng.uniform(0.0, 0.02, nd)
        jf = rng.integers(1, 4, nd)
        jph = rng.random(nd) * TAU
        # -- ragged outer shell + strays (culled by azimuthal clump noise),
        #    isotropic with a polar-cap boost so the silhouette reads round
        ns = 1700
        s_rho = rng.normal(SHELL_R, 0.025, ns)
        s_lat = np.where(rng.random(ns) < 0.45, rng.normal(0.0, 0.45, ns),
                         np.arcsin(rng.uniform(-1.0, 1.0, ns)))
        s_th = rng.random(ns) * TAU
        cf = rng.integers(2, 7, 4)
        cph = rng.random(4) * TAU
        noise = sum(np.sin(cf[i] * s_th + cph[i] + 0.7 * s_lat * (i + 1))
                    for i in range(4))
        keep = noise > 0.25
        s_rho, s_lat, s_th = s_rho[keep], s_lat[keep], s_th[keep]
        ns = s_rho.size
        stray = rng.random(ns) < 0.16
        s_rho = np.where(stray, np.minimum(s_rho * rng.uniform(1.02, 1.07, ns),
                                           1.0), s_rho)
        s_lum = np.where(stray, rng.uniform(50, 100, ns),
                         rng.uniform(80, 190, ns))
        s_lum = s_lum * (1.0 + 0.6 * np.abs(np.sin(s_lat)))     # polar boost
        s_m = np.ones(ns, dtype=int)
        d_rho = np.concatenate((rho, s_rho))
        d_lat = np.concatenate((lat, s_lat))
        d_m = np.concatenate((m, s_m))
        N = d_rho.size
        # tangential tails: angular half-step so the 3-point tail spans
        # ~0.8-3.5 px * sqrt|m| at sup
        tail_px = rng.uniform(0.8, 3.5, N) * np.sqrt(np.abs(d_m)) * sup
        tail = tail_px / np.maximum(d_rho * np.cos(d_lat) * self.R, 1.0)
        self.dust = {
            "rho": d_rho, "lat": d_lat,
            "th": np.concatenate((th, s_th)),
            "m": d_m,
            "lum": np.concatenate((lum, s_lum)),
            "jamp": np.concatenate((jamp, np.zeros(ns))),
            "jf": np.concatenate((jf, np.ones(ns, dtype=int))),
            "jph": np.concatenate((jph, np.zeros(ns))),
            "tail": tail,
        }
        # -- streaks: short orbit-direction strokes
        nk = 550
        u = rng.random(nk)
        k_rho = 0.12 + 0.85 * u ** 1.25
        k_lat = np.where(rng.random(nk) < 0.7, rng.normal(0.0, 0.28, nk),
                         np.arcsin(rng.uniform(-1.0, 1.0, nk)))
        self.streaks = {
            "rho": k_rho, "lat": np.clip(k_lat, -1.5, 1.5),
            "th": rng.random(nk) * TAU,
            "m": rng.choice((1, 1, 1, -1, 2), nk),
            "s": rng.uniform(0.010, 0.035, nk),
            "lum": rng.uniform(90, 210, nk) * (1.0 / (0.6 + k_rho)) ** 0.5,
        }
        # -- embers: bright blinking nodes
        ne = 70
        self.embers = {
            "rho": rng.uniform(0.25, 0.98, ne),
            "lat": rng.normal(0.0, 0.5, ne),
            "th": rng.random(ne) * TAU,
            "m": rng.choice((1, 1, -1, 2), ne),
            "rad": rng.uniform(0.9, 1.7, ne) * sup,
            "lum": rng.uniform(200, 255, ne),
            "bf": rng.integers(1, 4, ne),
            "bph": rng.random(ne) * TAU,
        }
        # -- debris: tiny dashes (radial / tangential) scattered through
        #    the volume — technical litter between the bands
        nb = 520
        b_rho = rng.uniform(0.16, 0.93, nb) ** 0.7 * 0.93
        b_lat = np.where(rng.random(nb) < 0.6, rng.normal(0.0, 0.35, nb),
                         np.arcsin(rng.uniform(-1.0, 1.0, nb)))
        b_lat = np.clip(b_lat, -1.5, 1.5)
        self.debris = {
            "rho": b_rho, "lat": b_lat, "th": rng.random(nb) * TAU,
            "m": np.where(rng.random(nb) < 0.6, 1, -1),
            "radial": rng.random(nb) < 0.5,
            "h": rng.uniform(0.006, 0.022, nb),
            "lum": _pick(rng, (48, 75, 110), (0.45, 0.35, 0.2), nb),
            "w": np.where(rng.random(nb) < 0.4, sup, 1),
        }

    # -------------------------------------------------------------- knot
    def _build_knot(self, rng):
        sup = self.sup
        nk = 60
        self.knot = {
            "r": rng.uniform(0.03, 0.19, nk),
            "th": rng.random(nk) * TAU,
            "len": rng.uniform(0.03, 0.12, nk),
            "psi": rng.normal(0.0, 0.5, nk),      # off-tangent orientation
            "lum": rng.uniform(190, 255, nk),
            "w": rng.choice((0, 1), nk),
        }
        # partial loops (tangled scribble, 2 turns) — disc-plane polylines
        loops = []
        for _ in range(7):
            cx, cy = rng.normal(0.0, 0.03, 2)
            a = rng.uniform(0.04, 0.14)
            b = a * rng.uniform(0.35, 0.9)
            psi = rng.uniform(0.0, math.pi)
            t0 = rng.uniform(0.0, TAU)
            span = rng.uniform(math.radians(200), TAU)
            t = t0 + np.linspace(0.0, span, 36)
            ex, ey = a * np.cos(t), b * np.sin(t)
            x = cx + ex * math.cos(psi) - ey * math.sin(psi)
            y = cy + ex * math.sin(psi) + ey * math.cos(psi)
            loops.append((x, y, rng.uniform(215, 255),
                          sup + (1 if rng.random() < 0.5 else 0)))
        self.knot_loops = loops
        # hot radial streaks (2 turns)
        st = []
        for _ in range(5):
            th = rng.uniform(0.0, TAU)
            ra = rng.uniform(0.08, 0.16)
            rb = ra + rng.uniform(0.05, 0.14)
            st.append((th, ra, rb, rng.normal(0.0, 0.03),
                       rng.uniform(170, 230)))
        self.knot_streaks = st
        # long faint chords through the centre (the horizontal smear, 1 turn)
        ch = []
        for length, lum in ((0.50, 120.0), (0.32, 95.0), (0.24, 80.0)):
            psi = rng.uniform(0.0, TAU)
            off = rng.normal(0.0, 0.015, 2)
            ch.append((psi, off[0], off[1], length, lum))
        self.knot_chords = ch
        # mini arcs around the core (partial rings, fast counter-spin)
        arcs = []
        for r, mult, span in ((0.085, +3, 220), (0.12, -2, 150),
                              (0.155, +2, 100), (0.16, +2, 70),
                              (0.19, -3, 50)):
            a = rng.random() * TAU
            ths = np.linspace(a, a + math.radians(span), 24)
            arcs.append((ths, r, mult))
        self.knot_arcs = arcs

    # ------------------------------------------------------------ stroke
    @staticmethod
    def _stroke(df, db, X0, Y0, X1, Y1, Z, lum, width):
        """Route segments front/back by depth and stroke them."""
        Z = np.asarray(Z, dtype=np.float64)
        lum = np.asarray(lum, dtype=np.float64) * _depth(Z)
        fr = Z >= Z_SPLIT
        seg = np.stack((X0, Y0, X1, Y1), axis=1)
        lv = np.clip(lum, 0, 255).astype(np.int32)
        w = int(width)
        for s4, l_ in zip(seg[fr].tolist(), lv[fr].tolist()):
            df.line(s4, fill=l_, width=w)
        for s4, l_ in zip(seg[~fr].tolist(), lv[~fr].tolist()):
            db.line(s4, fill=l_, width=w)

    def _splat(self, X, Y, lum):
        """Bilinear point splat into a float plate (sub-pixel smooth)."""
        S2 = self.S2
        x0 = np.floor(X).astype(np.int64)
        y0 = np.floor(Y).astype(np.int64)
        ok = (x0 >= 0) & (x0 < S2 - 1) & (y0 >= 0) & (y0 < S2 - 1)
        x0, y0, X, Y, lum = x0[ok], y0[ok], X[ok], Y[ok], lum[ok]
        fx = X - x0
        fy = Y - y0
        acc = np.zeros(S2 * S2, dtype=np.float64)
        for ddx, ddy, wgt in ((0, 0, (1 - fx) * (1 - fy)),
                              (1, 0, fx * (1 - fy)),
                              (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            acc += np.bincount((y0 + ddy) * S2 + (x0 + ddx),
                               weights=lum * wgt, minlength=S2 * S2)
        return acc.reshape(S2, S2).astype(np.float32)

    # ------------------------------------------------------------- frame
    def frame(self, k: int) -> Image.Image:
        k = int(k) % self.n
        ph = TAU * k / self.n
        S2, R, sup = self.S2, self.R, self.sup
        front = Image.new("L", (S2, S2), 0)
        back = Image.new("L", (S2, S2), 0)
        hotim = Image.new("L", (S2, S2), 0)
        df = ImageDraw.Draw(front)
        db = ImageDraw.Draw(back)
        dh = ImageDraw.Draw(hotim)

        # ---------------------------------------------------- tick bands
        def disc_lum(th, m, lum, idx, e):
            # front/back shading across the tilted disc, a mild highlight
            # sweep against each band's own spin, and the heavy envelope:
            # a static heavy sector (lower-left, like the film) plus a
            # softer highlight sweeping against the spin at 1 turn.
            Z = np.sin(th) * self.sin_t
            shade = 0.62 + 0.40 * (Z / self.sin_t + 1.0) * 0.5
            sgn = np.sign(m)
            hl = 0.82 + 0.18 * np.cos(th + sgn * ph + self.hl_phase[idx % 6])
            hs = (0.5 + 0.5 * np.sin(th - HEAVY)) ** 1.5
            hw = (0.5 + 0.5 * np.sin(th - self.heavy2 + sgn * ph)) ** 1.5
            env = (1.0 - e) + e * 1.25 * (0.6 * hs + 0.4 * hw)
            return lum * shade * hl * env

        # radial ticks
        th = self.t_th + self.t_m * ph
        X0, Y0, _ = self._proj_disc(th, self.t_r0)
        X1, Y1, _ = self._proj_disc(th, self.t_r1)
        lv = np.clip(disc_lum(th, self.t_m, self.t_l, 0, self.t_e), 0, 255) \
            .astype(np.int32)
        for x0, y0, x1, y1, l_, w_, h_ in zip(X0.tolist(), Y0.tolist(),
                                              X1.tolist(), Y1.tolist(),
                                              lv.tolist(), self.t_w.tolist(),
                                              self.t_h.tolist()):
            df.line((x0, y0, x1, y1), fill=l_, width=w_)
            if h_:
                dh.line((x0, y0, x1, y1), fill=l_, width=w_)
        # tangential dashes
        th = self.d_th + self.d_m * ph
        X0, Y0, _ = self._proj_disc(th - self.d_hw, self.d_r)
        X1, Y1, _ = self._proj_disc(th + self.d_hw, self.d_r)
        lv = np.clip(disc_lum(th, self.d_m, self.d_l, 1, self.d_e), 0, 255) \
            .astype(np.int32)
        for x0, y0, x1, y1, l_, w_, h_ in zip(X0.tolist(), Y0.tolist(),
                                              X1.tolist(), Y1.tolist(),
                                              lv.tolist(), self.d_w.tolist(),
                                              self.d_h.tolist()):
            df.line((x0, y0, x1, y1), fill=l_, width=w_)
            if h_:
                dh.line((x0, y0, x1, y1), fill=l_, width=w_)
        # blocks (4-corner polygons)
        if self.b_th.size:
            th = self.b_th + self.b_m * ph
            ax, ay, _ = self._proj_disc(th - self.b_hw, self.b_r0)
            bx, by, _ = self._proj_disc(th + self.b_hw, self.b_r0)
            cx, cy, _ = self._proj_disc(th + self.b_hw, self.b_r1)
            dx, dy, _ = self._proj_disc(th - self.b_hw, self.b_r1)
            lv = np.clip(disc_lum(th, self.b_m, self.b_l, 2, self.b_e),
                         0, 255).astype(np.int32)
            quads = np.stack((ax, ay, bx, by, cx, cy, dx, dy), axis=1).tolist()
            for q, l_ in zip(quads, lv.tolist()):
                df.polygon(q, fill=l_)
        # rails / baselines / arc-dashes as polylines
        for ths, rs, lum, m, w, e, h_ in self.base:
            th = ths + m * ph
            X, Y, _ = self._proj_disc(th, rs)
            l_ = int(np.clip(disc_lum(th[len(th) // 2], m, lum, 3, e), 0, 255))
            pts = list(zip(X.tolist(), Y.tolist()))
            df.line(pts, fill=l_, width=int(w))
            if h_:
                dh.line(pts, fill=l_, width=int(w))
        # node dots
        for th0, r, rad, lum, m, e, h_ in self.band_dots:
            th = th0 + m * ph
            X, Y, _ = self._proj_disc(th, r)
            l_ = int(np.clip(disc_lum(th, m, lum, 2, e), 0, 255))
            box = (X - rad, Y - rad, X + rad, Y + rad)
            df.ellipse(box, fill=l_)
            if h_:
                dh.ellipse(box, fill=l_)
        # hot arcs
        for ths, r, lum, m, e in self.hot:
            th = ths + m * ph
            X, Y, _ = self._proj_disc(th, r)
            l_ = int(np.clip(disc_lum(th[3], m, lum, 4, e), 0, 255))
            pts = list(zip(X.tolist(), Y.tolist()))
            df.line(pts, fill=l_, width=sup + 1)
            dh.line(pts, fill=l_, width=sup + 1)
        # band grain (splat, added to the dust plates below)
        th = self.g_th + self.g_m * ph
        gX, gY, gZ = self._proj_disc(th, self.g_r)
        g_lum = disc_lum(th, self.g_m, self.g_l, 5, self.g_e)

        # ------------------------------------------------- great circles
        for ring in self.rings:
            phi = ring["mult"] * ph
            cph, sph = math.cos(phi), math.sin(phi)
            p = ring["pts"]
            bu = p[:, 0] * cph - p[:, 1] * sph
            bv = p[:, 0] * sph + p[:, 1] * cph
            X, Y, Z = self._proj_body(bu, bv, p[:, 2])
            Zm = 0.5 * (Z[:-1] + Z[1:])
            kp = ring["keep"]
            lum = np.full(Zm.size, float(ring["lum"]))
            self._stroke(df, db, X[:-1][kp], Y[:-1][kp], X[1:][kp], Y[1:][kp],
                         Zm[kp], lum[kp], ring["width"])
            bd = ring["beads"]
            t = bd["t0"] + bd["m"] * ph
            q = ring["rad"] * (np.outer(np.cos(t), ring["u"])
                               + np.outer(np.sin(t), ring["v"]))
            bu = q[:, 0] * cph - q[:, 1] * sph
            bv = q[:, 0] * sph + q[:, 1] * cph
            X, Y, Z = self._proj_body(bu, bv, q[:, 2] + ring["ca"])
            lv = bd["lum"] * _depth(Z)
            for x_, y_, z_, r_, l_ in zip(X.tolist(), Y.tolist(), Z.tolist(),
                                          bd["rad"].tolist(), lv.tolist()):
                d_ = df if z_ >= Z_SPLIT else db
                d_.ellipse((x_ - r_, y_ - r_, x_ + r_, y_ + r_),
                           fill=int(l_))

        # ------------------------------------------------------- strands
        for s in self.strands:
            th = s["az"] + s["mult"] * ph
            X, Y, Z = self._proj_sph(s["rs"], s["el"], th + s["offs"])
            self._stroke(df, db, X[:-1], Y[:-1], X[1:], Y[1:],
                         0.5 * (Z[:-1] + Z[1:]), s["lum"], s["width"])
            X, Y, Z = self._proj_sph(s["b_r"], s["el"], th + s["b_o"])
            for x_, y_, z_, r_, l_ in zip(X.tolist(), Y.tolist(), Z.tolist(),
                                          s["b_rad"].tolist(),
                                          (s["b_l"] * _depth(Z)).tolist()):
                d_ = df if z_ >= Z_SPLIT else db
                d_.ellipse((x_ - r_, y_ - r_, x_ + r_, y_ + r_),
                           fill=int(l_))
            X0, Y0, Z0 = self._proj_sph(s["t_r"], s["el"], th + s["t_o"])
            X1, Y1, _ = self._proj_sph(s["t_r"], s["el"],
                                       th + s["t_o"] + s["t_side"] * s["t_len"]
                                       / np.maximum(s["t_r"], 0.05))
            self._stroke(df, db, X0, Y0, X1, Y1, Z0,
                         np.full(X0.size, 120.0), sup)

        # -------------------------------------------------------- sparks
        d = self.dust
        th = d["th"] + d["m"] * ph
        rho = d["rho"] + d["jamp"] * np.sin(d["jf"] * ph + d["jph"])
        tdir = -np.sign(d["m"]) * d["tail"]
        Xs, Ys, Ls, Zs = [], [], [], []
        for j, wgt in ((0, 0.55), (1, 0.30), (2, 0.15)):
            X, Y, Z = self._proj_sph(rho, d["lat"], th + tdir * j)
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
            Ls.append(d["lum"] * wgt * _depth(Z))
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        Z = np.concatenate(Zs)
        lum = np.concatenate(Ls)
        fr = Z >= Z_SPLIT
        dust_front = self._splat(np.concatenate((X[fr], gX)),
                                 np.concatenate((Y[fr], gY)),
                                 np.concatenate((lum[fr], g_lum)))
        dust_back = self._splat(X[~fr], Y[~fr], lum[~fr])

        s = self.streaks
        th = s["th"] + s["m"] * ph
        X0, Y0, Z0 = self._proj_sph(s["rho"], s["lat"], th)
        dth = np.sign(s["m"]) * s["s"] * np.abs(s["m"]) \
            / np.maximum(s["rho"] * np.cos(s["lat"]), 0.05)
        X1, Y1, _ = self._proj_sph(s["rho"], s["lat"], th + dth)
        self._stroke(df, db, X0, Y0, X1, Y1, Z0, s["lum"], 1)

        # debris dashes
        b = self.debris
        th = b["th"] + b["m"] * ph
        rad = b["radial"]
        Xa, Ya, Za = self._proj_sph(b["rho"] - b["h"] * rad, b["lat"], th)
        Xb, Yb, _ = self._proj_sph(b["rho"] + b["h"] * rad, b["lat"], th)
        dth = b["h"] * (~rad) / np.maximum(b["rho"] * np.cos(b["lat"]), 0.05)
        Xc, Yc, _ = self._proj_sph(b["rho"], b["lat"], th - dth)
        Xd, Yd, _ = self._proj_sph(b["rho"], b["lat"], th + dth)
        X0 = np.where(rad, Xa, Xc)
        Y0 = np.where(rad, Ya, Yc)
        X1 = np.where(rad, Xb, Xd)
        Y1 = np.where(rad, Yb, Yd)
        for w in (1, sup):
            sel = b["w"] == w
            if sel.any():
                self._stroke(df, db, X0[sel], Y0[sel], X1[sel], Y1[sel],
                             Za[sel], b["lum"][sel], w)
            if sup == 1:
                break

        e = self.embers
        th = e["th"] + e["m"] * ph
        X, Y, Z = self._proj_sph(e["rho"], e["lat"], th)
        blink = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(e["bf"] * ph + e["bph"]))
        lv = e["lum"] * blink * _depth(Z)
        for x_, y_, z_, r_, l_ in zip(X.tolist(), Y.tolist(), Z.tolist(),
                                      e["rad"].tolist(), lv.tolist()):
            d_ = df if z_ >= Z_SPLIT else db
            d_.ellipse((x_ - r_, y_ - r_, x_ + r_, y_ + r_), fill=int(l_))

        # ---------------------------------------------------------- knot
        c2, s2 = math.cos(2.0 * ph), math.sin(2.0 * ph)
        kn = self.knot
        ka = kn["th"] + 2.0 * ph
        kx, ky, _ = self._proj_disc(ka, kn["r"])
        psi = ka + math.pi / 2 + kn["psi"]
        hx = np.cos(psi) * kn["len"] * R * 0.5
        hy = np.sin(psi) * kn["len"] * R * 0.5 * self.cos_t
        for x_, y_, dx_, dy_, l_, w_ in zip(kx.tolist(), ky.tolist(),
                                            hx.tolist(), hy.tolist(),
                                            kn["lum"].tolist(),
                                            kn["w"].tolist()):
            df.line((x_ - dx_, y_ - dy_, x_ + dx_, y_ + dy_), fill=int(l_),
                    width=max(1, sup + int(w_)))
        for x, y, lum, w in self.knot_loops:
            bx = x * c2 - y * s2
            by = x * s2 + y * c2
            X, Y, _ = self._proj_xy(bx, by)
            df.line(list(zip(X.tolist(), Y.tolist())), fill=int(lum),
                    width=int(w))
        for th0, ra, rb, dth, lum in self.knot_streaks:
            th = th0 + 2.0 * ph
            xa, ya, _ = self._proj_disc(th, ra)
            xb, yb, _ = self._proj_disc(th + dth, rb)
            df.line((float(xa), float(ya), float(xb), float(yb)),
                    fill=int(lum), width=max(1, sup))
        for psi0, ox, oy, length, lum in self.knot_chords:
            psi = psi0 + ph
            dx_, dy_ = math.cos(psi) * length / 2, math.sin(psi) * length / 2
            xa, ya, _ = self._proj_xy(ox - dx_, oy - dy_)
            xb, yb, _ = self._proj_xy(ox + dx_, oy + dy_)
            df.line((float(xa), float(ya), float(xb), float(yb)),
                    fill=int(lum), width=max(1, sup))
        for ths, r, m in self.knot_arcs:
            X, Y, _ = self._proj_disc(ths + m * ph, r)
            df.line(list(zip(X.tolist(), Y.tolist())), fill=210,
                    width=max(1, sup))

        # ------------------------------------------------------- compose
        fa = np.asarray(front, dtype=np.float32) + dust_front
        ba = np.asarray(back, dtype=np.float32) + dust_back
        hf = np.asarray(hotim, dtype=np.float32)
        b8 = Image.fromarray(np.clip(ba, 0, 255).astype(np.uint8))
        back_soft = np.asarray(
            b8.filter(ImageFilter.GaussianBlur(1.3 * self.bscale * 2)),
            dtype=np.float32) * 0.55
        S = fa + back_soft + HOT_GAIN * hf          # hot tier burns past 255
        Sc = np.clip(S, 0.0, 255.0)
        hi1 = Image.fromarray(np.clip(Sc - 50.0 + 0.6 * hf, 0, 255)
                              .astype(np.uint8))
        hi2 = Image.fromarray(np.clip(Sc - 130.0, 0, 255).astype(np.uint8))
        hi3 = Image.fromarray(np.clip(Sc - 185.0, 0, 255).astype(np.uint8))
        b1 = np.asarray(hi1.filter(ImageFilter.GaussianBlur(1.2 * self.bscale * 2)),
                        dtype=np.float32)
        b2 = np.asarray(hi2.filter(ImageFilter.GaussianBlur(4.5 * self.bscale * 2)),
                        dtype=np.float32)
        b3 = np.asarray(hi3.filter(ImageFilter.GaussianBlur(14.0 * self.bscale * 2)),
                        dtype=np.float32)
        breath = 1.0 + 0.08 * math.sin(2.0 * ph)
        L = S + 0.70 * b1 + 0.50 * b2 + 0.24 * b3 \
            + self.glow_wide + breath * self.glow_tight
        L *= self.mask
        t = 1.0 - np.exp(-(TONE_GAIN / 255.0) * L)
        idx = (t * 255.0 + 0.5).astype(np.uint8)
        art = self.table[idx].astype(np.uint8)
        img = Image.fromarray(art, "RGB")
        if sup > 1:
            img = img.resize((self.size, self.size), Image.LANCZOS)
        rgb = np.asarray(img, dtype=np.int16)
        rgb[self.guard] = 0
        rgb += self.ground16
        np.clip(rgb, 0, 255, out=rgb)
        return Image.fromarray(rgb.astype(np.uint8), "RGB")
