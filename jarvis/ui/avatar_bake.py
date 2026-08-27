"""Avatar sphere bake kernel — pure numpy/PIL, Tk-free, app-free.

This module holds the JARVIS particulate-sphere renderer that used to live
inside Reactor (the bodies are unchanged; `self` became parameters) plus a
stdout-streaming CLI so the reactor can bake the rotation loop OUT OF
PROCESS: W worker subprocesses each bake a shard of the frame order and
stream `4-byte little-endian k + size²·3 raw RGB` per frame. The reactor
reads them on plain threads (blocking reads release the GIL) and converts
to PhotoImages on the Tk thread inside the frame loop's spare time.

Import rules (tested): nothing from jarvis.events / jarvis.logs /
jarvis.config / tkinter — colours are PARAMETERS (the CLI takes --bg and
--cyan), so importing this module never pulls in the app or a display.

    python -m jarvis.ui.avatar_bake --size 392 --sup 2 --frames 300 \
        --pool-peak 0.22 --pool-r 832 --bg 0d1b2a --cyan 35e0ff \
        --ks 0,12,24,…  > frames.bin
"""
from __future__ import annotations

import logging
import math
import os
import queue
import struct
import subprocess
import sys
import threading
import time

log = logging.getLogger("jarvis.ui.avatar_bake")

AV_SEED = 7                 # the particle field is identical every boot
AV_TILT = 22.0              # spin-axis tilt toward the viewer, degrees
AV_BACK_LUM = (0.38, 0.14)  # back-hemisphere brightness: lo + span*(1+Z)
AV_TIER_L = (70, 120, 175, 235)   # streak luminance per brightness tier
AV_WISP_L = 40                    # faint radial wisps
ART_R = 0.48                # art radius as a fraction of the square


def _blend(c1: tuple, c2: tuple, f: float) -> tuple:
    return tuple(int(a + (b - a) * f) for a, b in zip(c1, c2))


def hex_rgb(color: str) -> tuple:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------ field
def build_field(seed: int = AV_SEED) -> dict:
    """The ONE persistent particle field (seeded numpy). Particles live ON
    A SPHERE: each band is a latitude/shell zone — (lat0, lat_sig, rho0,
    rho_sig, loop multiplier, count, streak lo/hi, tier weights, clump,
    stray, both_hemis) — multipliers are INTEGER turns per loop so frame
    N wraps seamlessly to frame 0, and adjacent bands counter-rotate at
    differing speeds. rho is the shell radius as a unit fraction of the
    art radius R; lat is radians from the equator. The ragged outer shell
    is culled by azimuthal clump noise into frayed broken arcs, a fifth of
    its survivors straying outward. Particles are sorted dim→bright per
    hemisphere plate: PIL line drawing overwrites within a plate, so
    painter's order stands in for additive blending between tiers."""
    import numpy as np
    rng = np.random.default_rng(seed)
    bands = (
        # lat0 latsig rho0  rhosig  m count s_lo  s_hi
        #    tier weights        clump stray hemi
        # inner core swirl feeding the knot (fast, 2 turns/loop)
        (0.0, 0.50, 0.30, 0.090,  2,  660, 0.012, 0.030,
         (.15, .30, .30, .25), None, 0.0, False),
        # dense equatorial turbulence — the film mid-band
        (0.0, 0.16, 0.72, 0.070, -1, 1900, 0.015, 0.038,
         (.25, .40, .25, .10), -1.2, 0.0, False),
        (0.0, 0.28, 0.60, 0.080,  1, 1550, 0.018, 0.042,
         (.30, .40, .22, .08), -1.2, 0.0, False),
        # sparser mid-latitude bands, both hemispheres
        (0.55, 0.14, 0.72, 0.070, -1,  800, 0.030, 0.100,
         (.40, .35, .18, .07), -0.2, 0.10, True),
        # ragged outer shell — frayed broken arcs + strays; the wide
        # latitude spread rounds the silhouette into a BALL
        (0.0, 0.55, 0.90, 0.050,  1,  950, 0.040, 0.140,
         (.45, .35, .15, .05), 0.25, 0.20, False),
        # dim high-latitude caps / ambient halo — fills the poles so the
        # volume reads as a full ragged sphere
        (0.90, 0.25, 0.72, 0.150, -1,  420, 0.012, 0.032,
         (.60, .30, .10, .00), None, 0.0, True),
        # wound threads: long bright arc traces in the mid cloud
        (0.0, 0.22, 0.55, 0.100,  1,   90, 0.150, 0.350,
         (.00, .15, .45, .40), None, 0.0, False),
    )
    cols: dict = {k: [] for k in ("lat", "th", "rho", "m", "s", "tier",
                                  "jamp", "jfreq", "jph", "cant", "tjamp",
                                  "tjfreq", "tjph")}
    for (lat0, lsig, rho0, rsig, m, n, s_lo, s_hi, tw, cth,
         strayf, hemi) in bands:
        lat = rng.normal(lat0, lsig, n)
        if hemi:               # mirror the band across the equator
            lat = lat * rng.choice((-1.0, 1.0), n)
        rho = rng.normal(rho0, rsig, n)
        th = rng.uniform(0, 2 * np.pi, n)
        if cth is not None:
            # azimuthal clump noise: a mild threshold gives the equatorial
            # bands turbulent density; a hard one breaks the outer shell
            # into ragged frayed arcs. Harmonics start at 3 and skip
            # multiples of 2 — a 2θ term in counter-rotating bands used to
            # beat into a faint 4-arm pinwheel read.
            p1, p2, p3, p4 = rng.uniform(0, 2 * np.pi, 4)
            clump = (0.6 * np.sin(3 * th + p1)
                     + 0.8 * np.sin(5 * th + p2)
                     + 0.7 * np.sin(7 * th + p3)
                     + 0.5 * np.sin(11 * th + p4))
            keep = clump > cth
            lat, rho, th = lat[keep], rho[keep], th[keep]
            n = rho.size
        if strayf > 0.0:
            stray = rng.random(n) < strayf
            rho = rho + np.where(stray, rng.uniform(0.0, 0.12, n), 0.0)
        cols["lat"].append(np.clip(lat, -1.45, 1.45))
        cols["rho"].append(np.clip(rho, 0.06, 1.05))
        cols["th"].append(th)
        cols["m"].append(np.full(n, m, dtype=np.int32))
        cols["s"].append(rng.uniform(s_lo, s_hi, n))
        cols["tier"].append(rng.choice(4, n, p=tw))
        cols["jamp"].append(rng.uniform(0.004, 0.018, n))
        cols["jfreq"].append(rng.choice((1, 2), n))
        cols["jph"].append(rng.uniform(0, 2 * np.pi, n))
        # cant: the streak endpoint drifts slightly in latitude —
        # turbulence, not perfect latitude rings
        cols["cant"].append(rng.normal(0.0, 0.012, n))
        # independent angular orbit jitter (integer frequencies, so it
        # wraps with the loop): each particle wobbles along its orbit on
        # its own phase — clumps shear apart instead of rotating as rigid
        # spawn groups
        cols["tjamp"].append(rng.uniform(0.015, 0.060, n))
        cols["tjfreq"].append(rng.choice((1, 2, 3), n))
        cols["tjph"].append(rng.uniform(0, 2 * np.pi, n))
    f = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.argsort(f["tier"], kind="stable")
    f = {k: v[order] for k, v in f.items()}
    # faint radial wisps bridging shells — rotating spokes in the
    # (slightly latitude-scattered) equatorial plane
    nw = 40
    f["w_th"] = rng.uniform(0, 2 * np.pi, nw)
    f["w_r0"] = rng.uniform(0.13, 0.28, nw)
    f["w_r1"] = rng.uniform(0.40, 0.85, nw)
    f["w_lat"] = rng.normal(0.0, 0.22, nw)
    # detached silhouette embers: dim short streaks orbiting past the
    # shell edge, most breathing at integer loop frequencies so they fade
    # in/out across the loop
    ne = 13
    f["e_r"] = rng.uniform(0.97, 1.04, ne)   # ≤1.04: equatorial X stays
                                             # inside the baked square
    f["e_lat"] = rng.normal(0.0, 0.35, ne)
    f["e_th"] = rng.uniform(0, 2 * np.pi, ne)
    f["e_m"] = rng.choice((-1, 1), ne).astype(np.int32)
    f["e_s"] = rng.uniform(0.025, 0.060, ne)
    f["e_l"] = rng.uniform(55, 105, ne)
    f["e_bf"] = np.where(rng.random(ne) < 0.6,
                         rng.choice((1, 2, 3), ne), 0)
    f["e_bph"] = rng.uniform(0, 2 * np.pi, ne)
    # ragged-knot streaks: short bright white chords at random positions
    # and orientations inside ~0.16R of center — the core reads as a
    # blazing knot, not a lens flare. Screen-space and always FRONT.
    nk = 10
    f["k_r"] = rng.uniform(0.060, 0.160, nk)
    f["k_th"] = rng.uniform(0, 2 * np.pi, nk)
    f["k_psi"] = rng.uniform(0, 2 * np.pi, nk)
    f["k_len"] = rng.uniform(0.025, 0.085, nk)
    f["k_l"] = rng.choice((235, 255), nk, p=(0.4, 0.6))
    f["k_w"] = rng.choice((0, 1), nk)      # mixed stroke widths
    log.info("avatar: field built — %d streaks, %d wisps, %d embers, "
             "%d knot shards", f["rho"].size, nw, ne, nk)
    return f


def lut(cyan_rgb: tuple):
    """256-entry added-light ramp: deep blue → CYAN → ice → white (the
    film's black→gold→white ramp transposed to the JARVIS blue family;
    derived from the accent so a palette change re-derives the cloud)."""
    import numpy as np
    cy = tuple(cyan_rgb)
    deep = tuple(int(v * fct) for v, fct in zip(cy, (0.10, 0.24, 0.62)))
    ice = _blend(cy, (255, 255, 255), 0.55)
    anchors = ((0.0, (0, 0, 0)), (0.30, deep), (0.58, cy),
               (0.82, ice), (1.0, (255, 255, 255)))
    ts = np.array([a[0] for a in anchors])
    t = np.linspace(0.0, 1.0, 256)
    table = np.stack([np.interp(t, ts, [a[1][ch] for a in anchors])
                      for ch in range(3)], axis=1)
    return table.astype(np.int16)


def project(rho, lat, theta, sin_t, cos_t):
    """Orthographic spherical projection. Spin axis tilted AV_TILT° toward
    the viewer: axis A=(0,-cosτ,sinτ) in screen space (y down, z toward
    the viewer), equatorial basis e1=(1,0,0), e2=(0,sinτ,cosτ). Returns
    unit-R screen X, Y and depth Z (>0 = front)."""
    import numpy as np
    cl = np.cos(lat)
    u = rho * cl * np.cos(theta)      # e1 component
    v = rho * cl * np.sin(theta)      # e2 component
    a = rho * np.sin(lat)             # axis component
    return u, v * sin_t - a * cos_t, v * cos_t + a * sin_t


def pool_ground(size: int, sup: int, pool: tuple, bg_rgb: tuple,
                cyan_rgb: tuple):
    """Ground array for the base square: BG + the analytic glow pool
    sampled around the stage center (built once per bake, shared by
    every frame)."""
    import numpy as np
    peak, rp = pool
    S2 = size * sup
    c = S2 // 2
    y, x = np.ogrid[-c:S2 - c, -c:S2 - c]
    d = np.sqrt((x * x + y * y).astype(np.float32)) / sup
    pf = peak * np.clip(1.0 - d / rp, 0.0, 1.0) ** 2
    ground = np.empty((S2, S2, 3), dtype=np.uint8)
    for ch in range(3):
        base = bg_rgb[ch]
        ground[..., ch] = (base + (cyan_rgb[ch] - base) * pf).astype(np.uint8)
    return ground


def knot_glow(size: int, sup: int):
    """Analytic knot + interior haze (static across frames): the blazing
    white center accumulates OVER the streaks, and the faint translucent
    core-haze disc lights the sphere interior so the front hemisphere
    reads as passing OVER a lit volume."""
    import numpy as np
    S2 = size * sup
    c = S2 / 2.0
    R = S2 * ART_R
    y, x = np.ogrid[0:S2, 0:S2]
    dx = (x - c).astype(np.float32)
    dy = (y - c).astype(np.float32)
    d2 = dx * dx + dy * dy
    # bloom layers kept TIGHT so the smooth gaussians only seed the knot —
    # the ragged white streak chords in the frame bake carry its character
    glow = np.zeros((S2, S2), dtype=np.float32)
    for amp_g, rf in ((235.0, 0.058), (95.0, 0.105),
                      (36.0, 0.30), (13.0, 0.72)):
        glow += amp_g * np.exp(-d2 / (R * rf) ** 2)
    # core-haze: soft-edged interior disc to ~0.8R (not gaussian — a
    # plateau, the lit inside of the sphere)
    glow += 18.0 * np.clip(1.0 - d2 / (R * 0.80) ** 2, 0.0, 1.0) ** 1.3
    return glow


def build_frame(k: int, n_frames: int, size: int, sup: int, field: dict,
                ground16, glow, table):
    """One rotation frame of the SPHERE. All particle math is numpy
    (angles + jitter ride integer-frequency sinusoids, so the loop wraps
    bitwise: frame n_frames == frame 0); PIL only draws the strokes.
    Depth cues per frame: the back hemisphere (Z<0) rasters into its own
    HALF-resolution plate at 0.35–0.5 tier brightness with relatively
    wider strokes, then bilinear-upscales + blurs — fake defocus. Front
    streaks are sharp and LONGER (arc length × (1+0.35Z)); the static
    `glow` array carries the knot bloom plus the translucent core-haze
    disc. Compose: back(soft) + haze/knot + front + front bloom → blue
    LUT → additive over the pool ground → supersample downscale."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    S2 = size * sup
    c = S2 / 2.0
    R = S2 * ART_R
    f = field
    ph = 2.0 * math.pi * k / n_frames
    sin_t = math.sin(math.radians(AV_TILT))
    cos_t = math.cos(math.radians(AV_TILT))
    Sb = S2 // 2                       # back plate at half resolution
    back = Image.new("L", (Sb, Sb), 0)
    front = Image.new("L", (S2, S2), 0)
    db = ImageDraw.Draw(back)
    df = ImageDraw.Draw(front)

    def strokes(X0, Y0, X1, Y1, Z, lum, wd_f):
        """Split segments front/back, convert ONCE to plain lists and
        stroke. `lum` is a per-segment array; back luminance dims to
        0.35–0.5 by depth."""
        fr = Z >= 0.0
        seg = np.stack((c + X0 * R, c + Y0 * R,
                        c + X1 * R, c + Y1 * R), axis=1)
        lo, span = AV_BACK_LUM
        bl = lum * (lo + span * (1.0 + np.maximum(Z, -1.0)))
        fl = np.minimum(255.0, lum * (1.0 + 0.10 * np.minimum(Z, 1.0)))
        for s4, lv in zip((seg[fr]).tolist(),
                          fl[fr].astype(np.int32).tolist()):
            df.line(s4, fill=lv, width=wd_f)
        for s4, lv in zip((seg[~fr] * 0.5).tolist(),
                          bl[~fr].astype(np.int32).tolist()):
            db.line(s4, fill=lv, width=1)

    # faint wisp spokes bridging the shells (dim → bright ordering)
    wth = f["w_th"] + ph
    wx0, wy0, wz0 = project(f["w_r0"], f["w_lat"], wth, sin_t, cos_t)
    wx1, wy1, wz1 = project(f["w_r1"], f["w_lat"], wth, sin_t, cos_t)
    strokes(wx0, wy0, wx1, wy1, (wz0 + wz1) * 0.5,
            np.full(wth.size, float(AV_WISP_L)), 1)

    # detached ember orbiters past the shell edge; integer-frequency
    # brightness modulation wraps with the loop
    ea = f["e_th"] + f["e_m"] * ph
    ea2 = ea + np.sign(f["e_m"]) * f["e_s"] \
        / np.maximum(f["e_r"] * np.cos(f["e_lat"]), 0.05)
    elum = f["e_l"] * np.where(
        f["e_bf"] > 0,
        0.30 + 0.70 * (0.5 + 0.5 * np.sin(f["e_bf"] * ph + f["e_bph"])),
        1.0)
    ex0, ey0, ez0 = project(f["e_r"], f["e_lat"], ea, sin_t, cos_t)
    ex1, ey1, _ = project(f["e_r"], f["e_lat"], ea2, sin_t, cos_t)
    strokes(ex0, ey0, ex1, ey1, ez0, elum, 1)

    # the particle shells: frame-k longitude + independent angular orbit
    # jitter; radial (shell) jitter on its own sinusoid; the streak chord
    # runs along the latitude circle, its length scaled by depth (front
    # longer, back shorter) and its endpoint canted off-latitude
    theta = f["th"] + f["m"] * ph \
        + f["tjamp"] * np.sin(f["tjfreq"] * ph + f["tjph"])
    rho = f["rho"] + f["jamp"] * np.sin(f["jfreq"] * ph + f["jph"])
    cl = np.cos(f["lat"])
    X0, Y0, Z0 = project(rho, f["lat"], theta, sin_t, cos_t)
    zn = Z0 / np.maximum(rho, 1e-3)
    dth = np.sign(f["m"]) * f["s"] * (1.0 + 0.35 * zn) \
        / np.maximum(rho * cl, 0.05)
    X1, Y1, _ = project(rho, f["lat"] + f["cant"], theta + dth,
                        sin_t, cos_t)
    bounds = np.searchsorted(f["tier"], (0, 1, 2, 3, 4))
    for ti in range(4):
        i0, i1 = bounds[ti], bounds[ti + 1]
        lum = np.full(i1 - i0, float(AV_TIER_L[ti]))
        wd = 1 if ti < 2 else max(1, sup)
        strokes(X0[i0:i1], Y0[i0:i1], X1[i0:i1], Y1[i0:i1],
                zn[i0:i1], lum, wd)

    # ragged-knot streaks last, ALWAYS front (the knot blazes over the
    # volume): white-hot chords riding the core band's 2-turn multiplier
    # — position AND orientation rotate seamlessly
    ka = f["k_th"] + 2 * ph
    kx = c + np.cos(ka) * f["k_r"] * R
    ky = c + np.sin(ka) * f["k_r"] * R
    kpsi = f["k_psi"] + 2 * ph
    kdx = np.cos(kpsi) * f["k_len"] * R * 0.5
    kdy = np.sin(kpsi) * f["k_len"] * R * 0.5
    for x_, y_, dx_, dy_, lv, w_ in zip(
            kx.tolist(), ky.tolist(), kdx.tolist(), kdy.tolist(),
            f["k_l"].tolist(), f["k_w"].tolist()):
        df.line((x_ - dx_, y_ - dy_, x_ + dx_, y_ + dy_),
                fill=int(lv), width=max(1, sup + int(w_)))

    # compose: soft wide back + lit interior (glow carries knot bloom AND
    # core-haze) + sharp front + front bloom — all additive light
    back = back.filter(ImageFilter.GaussianBlur(0.6 * sup)) \
               .resize((S2, S2), Image.BILINEAR)
    ln = np.asarray(front, dtype=np.float32)
    bloom = np.asarray(front.filter(
        ImageFilter.GaussianBlur(1.5 * sup)), dtype=np.float32)
    ln += 0.55 * bloom
    ln += np.asarray(back, dtype=np.float32)
    ln += glow
    lum8 = np.clip(ln, 0.0, 255.0).astype(np.uint8)
    rgb = table[lum8]
    rgb += ground16
    np.clip(rgb, 0, 255, out=rgb)
    img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    if sup > 1:
        img = img.resize((size, size), Image.LANCZOS)
    return img


class BakeKernel:
    """Everything a frame needs besides k — built once per bake."""

    def __init__(self, size: int, sup: int, n_frames: int, pool: tuple,
                 bg_rgb: tuple, cyan_rgb: tuple, seed: int = AV_SEED):
        import numpy as np
        self.size, self.sup, self.n = int(size), int(sup), int(n_frames)
        self.field = build_field(seed)
        self.ground16 = pool_ground(size, sup, pool, bg_rgb,
                                    cyan_rgb).astype(np.int16)
        self.glow = knot_glow(size, sup)
        self.table = lut(cyan_rgb)

    def frame(self, k: int):
        return build_frame(k, self.n, self.size, self.sup, self.field,
                           self.ground16, self.glow, self.table)


# -------------------------------------------------------------- streaming
HEADER = struct.Struct("<I")


def _read_exact(stream, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


class BakeRunner:
    """Bake `order` (frame indices) for one generation, out of process
    when possible. Frames arrive on `self.queue` as (k, rgb_bytes) —
    consume with queue.get(block=False) from the render loop.

    `start()` launches `workers` subprocesses (interleaved shards, so the
    tier order still completes coarse → fine) with one reader thread each;
    if the CLI cannot be launched, or the workers exit without delivering
    every frame, the missing frames are baked on an in-process thread
    with the same kernel. `mode` reports which path ran."""

    def __init__(self, size: int, sup: int, n_frames: int, order: list,
                 pool: tuple, bg_rgb: tuple, cyan_rgb: tuple,
                 workers: int = 4, python: str = None, seed: int = AV_SEED,
                 nice: int = 5):
        self.size, self.sup, self.n = int(size), int(sup), int(n_frames)
        self.order = list(order)
        self.pool = (float(pool[0]), float(pool[1]))
        self.bg_rgb, self.cyan_rgb = tuple(bg_rgb), tuple(cyan_rgb)
        self.workers = max(1, int(workers))
        self.python = python or sys.executable
        self.seed = seed
        self.nice = nice
        self.queue: queue.SimpleQueue = queue.SimpleQueue()
        self.mode = None
        self._procs: list = []
        self._threads: list = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._emitted: set = set()
        self._exited = 0
        self.t_start = time.monotonic()

    # ------------------------------------------------------------ control
    def start(self) -> None:
        try:
            self._spawn()
        except Exception as exc:          # CLI unavailable → in-process
            log.warning("avatar bake workers unavailable (%s); baking "
                        "in-process", exc)
            self._procs = []
            self._start_thread(self.order)

    def stop(self, wait: bool = True) -> None:
        """Kill the workers now; with wait=True also reap them and join
        the reader threads (call that off the UI thread)."""
        self._stop.set()
        for proc in self._procs:
            try:
                proc.kill()
            except OSError:
                pass
        if not wait:
            return
        for proc in self._procs:
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=2)

    def emitted(self) -> int:
        with self._lock:
            return len(self._emitted)

    # ------------------------------------------------------------ workers
    def _cmd(self, ks: list) -> list:
        return [self.python, "-m", "jarvis.ui.avatar_bake",
                "--size", str(self.size), "--sup", str(self.sup),
                "--frames", str(self.n),
                "--pool-peak", repr(self.pool[0]),
                "--pool-r", repr(self.pool[1]),
                "--bg", "%02x%02x%02x" % self.bg_rgb,
                "--cyan", "%02x%02x%02x" % self.cyan_rgb,
                "--seed", str(self.seed), "--nice", str(self.nice),
                "--ks", ",".join(str(k) for k in ks)]

    def _spawn(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = _repo_root() + os.pathsep + env.get(
            "PYTHONPATH", "")
        shards = [self.order[i::self.workers] for i in range(self.workers)]
        for ks in shards:
            if not ks:
                continue
            proc = subprocess.Popen(
                self._cmd(ks), stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                env=env, cwd=_repo_root(), bufsize=1 << 20)
            self._procs.append(proc)
        if not self._procs:
            raise RuntimeError("nothing to bake")
        self.mode = "subprocess"
        for proc in self._procs:
            t = threading.Thread(target=self._reader, args=(proc,),
                                 daemon=True, name="avatar-bake-reader")
            t.start()
            self._threads.append(t)

    def _reader(self, proc) -> None:
        nbytes = self.size * self.size * 3
        out = proc.stdout
        try:
            while not self._stop.is_set():
                hdr = _read_exact(out, HEADER.size)
                if hdr is None:
                    break
                k = HEADER.unpack(hdr)[0]
                buf = _read_exact(out, nbytes)
                if buf is None:
                    break
                with self._lock:
                    self._emitted.add(k)
                self.queue.put((k, buf))
        finally:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        with self._lock:
            self._exited += 1
            all_out = self._exited >= len(self._procs)
            missing = [k for k in self.order if k not in self._emitted]
        if all_out and missing and not self._stop.is_set():
            log.warning("avatar bake workers delivered %d/%d frames; "
                        "baking the rest in-process",
                        len(self.order) - len(missing), len(self.order))
            self._start_thread(missing)

    # ------------------------------------------------------- in-process
    def _start_thread(self, ks: list) -> None:
        self.mode = "thread" if self.mode is None else self.mode + "+thread"
        t = threading.Thread(target=self._bake_inline, args=(list(ks),),
                             daemon=True, name="avatar-bake-inline")
        t.start()
        self._threads.append(t)

    def _bake_inline(self, ks: list) -> None:
        try:
            kernel = BakeKernel(self.size, self.sup, self.n, self.pool,
                                self.bg_rgb, self.cyan_rgb, self.seed)
            for k in ks:
                if self._stop.is_set():
                    return
                img = kernel.frame(k)
                with self._lock:
                    self._emitted.add(k)
                self.queue.put((k, img.tobytes()))
                time.sleep(0.002)          # breathing room for the Tk loop
        except Exception:
            log.exception("in-process avatar bake failed")


# ------------------------------------------------------------------- CLI
def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(prog="python -m jarvis.ui.avatar_bake")
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--sup", type=int, default=1)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--pool-peak", type=float, default=0.22)
    p.add_argument("--pool-r", type=float, default=400.0)
    p.add_argument("--bg", default="0d1b2a")
    p.add_argument("--cyan", default="35e0ff")
    p.add_argument("--seed", type=int, default=AV_SEED)
    p.add_argument("--nice", type=int, default=0)
    p.add_argument("--ks", required=True,
                   help="comma-separated frame indices to bake, in order")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.nice:
        try:
            os.nice(args.nice)
        except OSError:
            pass
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    kernel = BakeKernel(args.size, args.sup, args.frames,
                        (args.pool_peak, args.pool_r),
                        hex_rgb(args.bg), hex_rgb(args.cyan), args.seed)
    out = sys.stdout.buffer
    try:
        for k in ks:
            img = kernel.frame(k)
            out.write(HEADER.pack(k))
            out.write(img.tobytes())
            out.flush()
    except BrokenPipeError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
