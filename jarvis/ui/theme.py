"""Design tokens for the Jarvis V3 UI. All colors and fonts come from here —
no hex literals anywhere else in jarvis/ui.

Direction: "luminous hologram" — the radiant workshop projection, not a dark
console: a visibly lit cold-blue ground (the light fills the space), glassy
panel fills, cyan structure linework with white focal values, chamfered
panels and dense technical ornamentation in pre-blended holo tints, with
Rajdhani as the display face over an Inter body.

Every pre-blended ramp/tint below is COMPUTED against BG at import time
(canvas items have no alpha) — change BG and the whole ladder re-derives
consistently instead of drifting.
"""
from __future__ import annotations

import tkinter.font as tkfont


def _rgb(color: str) -> tuple:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb: tuple) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in rgb)


def _mix(c1: tuple, c2: tuple, f: float) -> tuple:
    return tuple(a + (b - a) * f for a, b in zip(c1, c2))


# Ground — lifted cold blue (the hologram LIGHT fills the room; near-black
# grounds read as a dark console, which the direction overrides).
BG   = "#0d1b2a"
CYAN = "#35e0ff"

_BG_RGB = _rgb(BG)
_CY_RGB = _rgb(CYAN)
_WHITE = (255, 255, 255)


def _ramp(a: float) -> str:
    """CYAN over BG at alpha a, pre-blended."""
    return _hex(_mix(_BG_RGB, _CY_RGB, a))


def _glass(a_cyan: float, a_white: float = 0.0) -> str:
    """Translucent-glass fill: cyan tint over BG, then an icy white lift —
    the budget shifts toward ice/white luminosity, not saturated cyan."""
    return _hex(_mix(_mix(_BG_RGB, _CY_RGB, a_cyan), _WHITE, a_white))


# Ground ladder (darkest → lightest surface). SURFACE/RAISED/LINE keep the
# old relative structure but are derived from the new BG.
SURFACE = _glass(0.055, 0.012)   # bars, command strip
RAISED  = _glass(0.115, 0.030)   # cards, fields, chips — lit glass slabs
LINE    = _ramp(0.21)            # hairline separators, outlines

# Glass catch-light: the 1px lighter inner top-edge highlight on panels.
GLASS_EDGE = _glass(0.30, 0.22)

# Ink
INK   = "#e8f0f8"     # primary text (legibility is the hard constraint)
MUTED = "#93a9be"     # secondary text
FAINT = "#61788f"     # captions, timestamps, disabled

# Accent — one family
CYAN_DIM  = "#1899bd"
CYAN_SOFT = _ramp(0.26)          # fills / selected backgrounds

# Cyan structure ramp (film rule: CYAN over BG at the given alpha, computed
# above). Most linework sits at the 20-47% steps now that the ground is
# luminous; BRIGHT is reserved for corner-bracket accents.
RAMP13 = _ramp(0.13)
RAMP20 = _ramp(0.20)
RAMP27 = _ramp(0.27)
RAMP33 = _ramp(0.33)
RAMP40 = _ramp(0.40)
RAMP47 = _ramp(0.47)
RAMP60 = _ramp(0.60)
BRIGHT = _ramp(0.87)             # bracket accents, single focal strokes

# White-hot focal ramp: focal text/needles are WHITE, not cyan (the mature
# Stark HUD is white on its ground; cyan is structure, never the star).
FOCAL = "#eaf7ff"
CORE_BANDS = ("#ffffff", "#d4f4ff", "#a8e9ff", "#7de4ff", RAMP60)

# Semantic. Film budget: cyan structure / white focal / amber semantic /
# red alert — green is NOT in the palette, so the OK state reads as a
# white focal value.
OK   = FOCAL
WARN = "#ffb454"
ERR  = "#f8556d"

# Holographic decor tints — promoted one ramp step from the dark-console
# pass so the dense rim texture is clearly VISIBLE against the lit ground.
GRID     = _ramp(0.13)   # dot grid, outermost guide circles
SCAN     = _ramp(0.16)   # scanline tint
HOLO_DIM = _ramp(0.22)   # faint decor strokes
HOLO     = _ramp(0.36)   # decor strokes
EDGE     = RAMP60        # corner brackets, sweep lead, orbit dots
FRAME    = _ramp(0.27)   # 1px window outline (projected-panel edge)

# ---- Luminous-density pass (lower-panel volume) --------------------------
# The refs get their light from DENSITY of dim cyan wireframe filling the
# volume, not one bright centerpiece. The transcript ground lifts one step
# toward the reactor panel tone and every lower-panel decor tint re-derives
# against that lifted ground (canvas items have no alpha).
TV_BG = _ramp(0.05)              # transcript canvas ground (~#0f2434)
_TV_RGB = _rgb(TV_BG)


def _tv(a: float) -> str:
    """CYAN over the lifted transcript ground at alpha a, pre-blended."""
    return _hex(_mix(_TV_RGB, _CY_RGB, a))


def _tvglass(a_cyan: float, a_white: float = 0.0) -> str:
    """Glass tint over the lifted transcript ground."""
    return _hex(_mix(_mix(_TV_RGB, _CY_RGB, a_cyan), _WHITE, a_white))


TV_SCANLINE = _ramp(0.085)       # 1px scanline rows — ONE step over TV_BG
TV_POOL = tuple(_tv(a) for a in (0.02, 0.045, 0.07, 0.095))
                                 # radial light pool ovals, outermost first
                                 # (innermost stays inside the ~#12 ceiling)
TV_GRID  = _tvglass(0.18, 0.03)  # dot grid, one step brighter (~#1c4b5d)
TV_WIRE  = _hex(_mix(_rgb(_tv(0.11)), _TV_RGB, 0.35))
                                 # ambient wireframe strokes — pulled 35%
                                 # back toward the ground: the dome is an
                                 # EXTREMELY faint background, never a
                                 # competitor to the cards
TV_WIRE2 = _tv(0.16)             # brighter wireframe step / bright motes

# Lit-dome pass (finale): the lower projection dome is a WIREFRAME the
# light lives in, not a smudge — ring tone with a 1px dark offset ghost
# (emboss) and a perspective floor grid seating the dome on a lit
# surface. Ring tone likewise pulled 35% toward the ground (faint).
TV_RING       = _hex(_mix(_rgb(_tvglass(0.20, 0.02)), _TV_RGB, 0.35))
                                       # dome concentric rings (faint)
TV_RING_GHOST = _tv(0.05)              # 1px offset emboss arc (~#0e2e3a)
TV_NODE       = _tvglass(0.80, 0.40)   # node dots / mote pulses (~#8fd4ea kin)
TV_FLOOR      = _tv(0.08)              # perspective floor grid (~#123642)

# Reactor→transcript seam dissolve: the transcript's lit top row tone
# (mirrors views.GRAD_PEAK over TV_BG) and pre-blended 1px line steps
# walking the reactor ground down into it so the glow bleeds across the
# panel boundary instead of stopping at a hard edge.
TV_TOP = _hex(_mix(_TV_RGB, _CY_RGB, 0.13))
SEAM_STEPS = tuple(
    _hex(_mix(_BG_RGB, _rgb(TV_TOP), ((i + 1) / 20) ** 1.25))
    for i in range(20))

# Two-stroke fake glow (the reactor-arc treatment, exported for the rest
# of the app): a WIDE dim underlay stroke beneath a NARROW bright core.
GLOW_UNDER = _tvglass(0.20, 0.03)   # wide dim underlay (~#1d4f61)
ARC_BRIGHT = _glass(0.87, 0.18)     # narrow bright core, icy (~#55d0e8)

# Side-rail micro-labels — desaturated slate-cyan at comfortable contrast.
RAIL = _glass(0.45, 0.18)           # ~#478c9e

# Chamfer cut for HUD panels (design units; scaled with RADIUS by apply_scale)
CHAMFER = 10

# Reactor / app states. Speaking signals by driving the core and one scan
# arc WHITE-HOT over unchanged cyan structure (film rule: color is state,
# and the focal state value is white — never green).
STATE_COLORS = {
    "idle":      CYAN_DIM,
    "listening": CYAN,
    "thinking":  WARN,
    "speaking":  FOCAL,
    "waiting":   WARN,       # Claude is waiting on a permission answer
    "working":   CYAN_DIM,   # a Claude task is running
    "error":     ERR,
    "offline":   FAINT,
}

# Pill states whose WORD stays FOCAL (only the dot carries the colour):
# idle, and the two Claude-task states — the word is white so the header
# stays calm while a task runs for minutes.
FOCAL_WORD_STATES = ("idle", "working", "waiting")

# Spacing (8px grid) — design units at the 96-dpi baseline. apply_scale()
# mutates these once at startup for the global UI scale factor S.
PAD = 16
PAD_S = 8
PAD_L = 24
RADIUS = 10

_BASE_SPACING = (PAD, PAD_S, PAD_L, RADIUS, CHAMFER)


def apply_scale(scale: float) -> None:
    """Scale the spacing tokens by the global UI scale S. Called once by
    MainWindow before any widgets are built; idempotent (always derives
    from the design-unit baseline, so repeated calls don't compound)."""
    global PAD, PAD_S, PAD_L, RADIUS, CHAMFER
    PAD, PAD_S, PAD_L, RADIUS, CHAMFER = (
        max(1, round(v * scale)) for v in _BASE_SPACING)

# Type scale
SIZE_WORDMARK = 22   # 26 pre-holo; the tracked-out 'J A R V I S' wordmark
                     # needs the width back so the header status can breathe
SIZE_BODY = 15
SIZE_LABEL = 13
SIZE_CAPTION = 9    # ONE annotation size for every HUD label/value

_FAMILY = None
_FAMILY_MONO = None
_HAS_DISPLAY = False       # True when a display face (see _DISPLAY_FACES) exists
_DISPLAY = None            # the chosen display family, e.g. "Chakra Petch"

# Display faces in preference order, user's pick first (2026-08-26: Chakra
# Petch chosen from rendered samples; Rajdhani kept so the UI still renders
# if the newer family is ever missing).  Each entry maps the family to the
# fontconfig names of its medium/semibold faces.
_DISPLAY_FACES = (
    ("Chakra Petch", "Chakra Petch Medium", "Chakra Petch SemiBold"),
    ("Rajdhani", "Rajdhani Medium", "Rajdhani SemiBold"),
)


def resolve_fonts(root=None) -> str:
    """Pick the best available family. Call once after Tk root exists."""
    global _FAMILY, _FAMILY_MONO, _HAS_DISPLAY, _DISPLAY
    if _FAMILY:
        return _FAMILY
    try:
        available = set(tkfont.families(root))
    except Exception:
        available = set()
    for candidate in ("Inter", "Liberation Sans", "DejaVu Sans"):
        if candidate in available:
            _FAMILY = candidate
            break
    else:
        _FAMILY = "TkDefaultFont"
    for candidate in ("JetBrains Mono", "DejaVu Sans Mono", "Liberation Mono"):
        if candidate in available:
            _FAMILY_MONO = candidate
            break
    else:
        _FAMILY_MONO = "TkFixedFont"
    for fam, _med, _semi in _DISPLAY_FACES:
        if fam in available:
            _DISPLAY = fam
            break
    _HAS_DISPLAY = _DISPLAY is not None
    return _FAMILY


def font(size: int, weight: str = "normal") -> tuple:
    return (_FAMILY or "DejaVu Sans", size, weight)


def mono(size: int) -> tuple:
    return (_FAMILY_MONO or "DejaVu Sans Mono", size, "normal")


def display(size: int, weight: str = "normal") -> tuple:
    """Display face for HUD chrome: wordmark, labels, chips, status text.
    weight: normal | semibold | bold. The Medium/SemiBold faces are
    addressed by fontconfig name (tkfont.families lists only the base
    family, but Xft resolves the named faces); body-font fallback maps
    semibold to a plain Tk bold."""
    if _HAS_DISPLAY and _DISPLAY:
        for fam, medium, semibold in _DISPLAY_FACES:
            if fam != _DISPLAY:
                continue
            if weight == "semibold":
                return (semibold, size, "normal")
            if weight == "bold":
                return (fam, size, "bold")
            return (medium, size, "normal")
    fam = _FAMILY or "DejaVu Sans"
    return (fam, size, "bold" if weight in ("semibold", "bold") else "normal")
