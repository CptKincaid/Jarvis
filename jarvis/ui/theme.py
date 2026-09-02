"""Design tokens for the Jarvis V3 UI. All colors and fonts come from here —
no hex literals anywhere else in jarvis/ui.

Two LOOKS share one derivation (2026-09-01, the blue-holographic overhaul):

  holo     the film-black projection — a near-black navy ground, glass so
           thin nothing reads as a filled slab, 1px cyan strokes carrying
           the chrome, white focal values (Hunter: "like this but blue").
  classic  the 08-31 "luminous hologram" — a visibly lit cold-blue ground,
           glassy panel fills, cyan structure linework. Kept token-for-token
           as the FALLBACK he asked for ("create a fallback if I don't like
           the visuals"): tests/fixtures/theme_tokens_85d5066.json
           is the oracle and tests/test_theme_look.py holds us to it.

Every pre-blended ramp/tint is COMPUTED against the look's BG by _derive()
(canvas items have no alpha) — change an anchor and the whole ladder
re-derives consistently instead of drifting. select_look() re-runs the
same derivation at runtime, which is why every other UI module must read
theme.X at CALL time and never capture it in a def default or a class
body: those would freeze the import-time look.
"""
from __future__ import annotations

import os
import tkinter.font as tkfont
from typing import Callable, Mapping, Optional

from jarvis.logs import get_logger

log = get_logger("ui.theme")

LOOKS = ("holo", "classic")
DEFAULT_LOOK = "holo"
LOOK = DEFAULT_LOOK          # the current look; select_look() rewrites it
OPTION_KEY = "console.look"  # assistant.json key the voice command writes
ENV_KEY = "JARVIS_LOOK"


def _rgb(color: str) -> tuple:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb: tuple) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in rgb)


def _mix(c1: tuple, c2: tuple, f: float) -> tuple:
    return tuple(a + (b - a) * f for a, b in zip(c1, c2))


# ---- per-look anchors ---------------------------------------------------
# Only what actually differs between the looks lives here; everything else
# is a formula in _derive(). Alpha anchors are (a_cyan[, a_white]) blends
# over the look's own BG, so a look's glass is "how much light the panel
# lets through", not a colour picked by eye.
_LOOKS = {
    # Ground — lifted cold blue (the hologram LIGHT fills the room; the
    # 08-31 direction overrode near-black grounds as "a dark console").
    "classic": dict(
        BG="#0d1b2a",
        CYAN="#35e0ff",
        INK="#e8f0f8",        # primary text (legibility is the hard constraint)
        MUTED="#93a9be",      # secondary text
        FAINT="#61788f",      # captions, timestamps, disabled
        CYAN_DIM="#1899bd",
        FOCAL="#eaf7ff",
        WARN="#ffb454",
        ERR="#f8556d",
        SURFACE=(0.055, 0.012),   # bars, command strip
        RAISED=(0.115, 0.030),    # cards, fields, chips — lit glass slabs
        LINE=0.21,                # hairline separators, outlines
        GLASS_EDGE=(0.30, 0.22),  # 1px inner top-edge catch-light
        RAIL=(0.45, 0.18),        # side-rail micro-labels (~#478c9e)
        FRAME=0.27,               # 1px window outline
        GRID=0.13, SCAN=0.16, HOLO_DIM=0.22, HOLO=0.36,
        TV_LIFT=0.13,             # transcript top-row cyan lift (views.GRAD_PEAK)
    ),
    # Ground — film black with a navy cast (ref2_hud: the HUD is thin gold
    # strokes on BLACK; ours is the same on blue). The sphere kernel is
    # analytic, so it sits on any ground; CYAN stays the accent it was
    # judged with. Surfaces drop to a fraction of the classic alphas so a
    # card is a pane of glass over black, never a slab; the 1px strokes
    # (LINE / GLASS_EDGE / RAIL / FRAME) step up because on black they ARE
    # the chrome. Text lifts ~10% so MUTED/FAINT keep their contrast on the
    # darker ground. Amber stays reserved for warnings.
    "holo": dict(
        BG="#050b14",
        CYAN="#35e0ff",
        INK="#e9f2fb",
        MUTED="#a2b8cc",
        FAINT="#6e879f",
        CYAN_DIM="#1899bd",
        FOCAL="#eaf7ff",
        WARN="#ffb454",
        ERR="#f8556d",
        SURFACE=(0.030, 0.006),
        RAISED=(0.060, 0.014),
        LINE=0.30,
        GLASS_EDGE=(0.40, 0.26),
        RAIL=(0.55, 0.20),
        FRAME=0.36,
        GRID=0.15, SCAN=0.17, HOLO_DIM=0.25, HOLO=0.40,
        # FLAT transcript ground: no top lift, so the seam dissolve lands
        # on TV_BG itself. The holo cards are frames whose interior IS the
        # ground (views.card_look), and Tk has no alpha -- a gradient or
        # pool behind them would show every card as a darker slab (the
        # 09-01 review measured a ~7-25 level step). The pool ovals and
        # floor grid are not drawn in holo either (views.ground_is_flat).
        TV_LIFT=0.0,
    ),
}

# Spacing (8px grid) — design units at the 96-dpi baseline, the same in
# both looks. apply_scale() scales these for the global UI scale factor S
# and _derive() re-applies the last scale after a look switch.
_BASE_SPACING = (16, 8, 24, 10, 10)   # PAD, PAD_S, PAD_L, RADIUS, CHAMFER
_SCALE = 1.0

# Pill states whose WORD stays FOCAL (only the dot carries the colour):
# idle, and the two Claude-task states — the word is white so the header
# stays calm while a task runs for minutes.
FOCAL_WORD_STATES = ("idle", "working", "waiting")

# Type scale — look-independent.
SIZE_WORDMARK = 22   # 26 pre-holo; the tracked-out 'J A R V I S' wordmark
                     # needs the width back so the header status can breathe
SIZE_BODY = 15
SIZE_LABEL = 13
SIZE_CAPTION = 9    # ONE annotation size for every HUD label/value

# Names _derive() (re)writes — what a look switch changes. Exported so the
# def-time-capture guard test can tell a frozen colour from a frozen font
# size (the type scale is the same in both looks, so capturing it is fine).
LOOK_TOKENS: frozenset = frozenset()


def _derive(name: str) -> None:
    """(Re)compute every look-dependent token from the anchors of `name`
    and publish them as module globals. Runs once at import for the
    default look and again from select_look()."""
    global LOOK, LOOK_TOKENS
    a = _LOOKS[name]
    BG, CYAN = a["BG"], a["CYAN"]
    _BG_RGB = _rgb(BG)
    _CY_RGB = _rgb(CYAN)
    _WHITE = (255, 255, 255)

    def _ramp(alpha: float) -> str:
        """CYAN over BG at alpha, pre-blended."""
        return _hex(_mix(_BG_RGB, _CY_RGB, alpha))

    def _glass(a_cyan: float, a_white: float = 0.0) -> str:
        """Translucent-glass fill: cyan tint over BG, then an icy white
        lift — the budget shifts toward ice/white luminosity, not
        saturated cyan."""
        return _hex(_mix(_mix(_BG_RGB, _CY_RGB, a_cyan), _WHITE, a_white))

    t = {"BG": BG, "CYAN": CYAN, "_BG_RGB": _BG_RGB, "_CY_RGB": _CY_RGB,
         "_WHITE": _WHITE}

    # Ground ladder (darkest → lightest surface).
    t["SURFACE"] = _glass(*a["SURFACE"])
    t["RAISED"] = _glass(*a["RAISED"])
    t["LINE"] = _ramp(a["LINE"])
    # Glass catch-light: the 1px lighter inner top-edge highlight on panels.
    t["GLASS_EDGE"] = _glass(*a["GLASS_EDGE"])

    # Ink
    t["INK"], t["MUTED"], t["FAINT"] = a["INK"], a["MUTED"], a["FAINT"]

    # Accent — one family
    t["CYAN_DIM"] = a["CYAN_DIM"]
    t["CYAN_SOFT"] = _ramp(0.26)          # fills / selected backgrounds

    # Cyan structure ramp (film rule: CYAN over BG at the given alpha).
    # Most linework sits at the 20-47% steps; BRIGHT is reserved for
    # corner-bracket accents.
    for step in (13, 20, 27, 33, 40, 47, 60):
        t[f"RAMP{step}"] = _ramp(step / 100)
    t["BRIGHT"] = _ramp(0.87)             # bracket accents, single focal strokes

    # White-hot focal ramp: focal text/needles are WHITE, not cyan (the
    # mature Stark HUD is white on its ground; cyan is structure, never
    # the star).
    t["FOCAL"] = a["FOCAL"]
    t["CORE_BANDS"] = ("#ffffff", "#d4f4ff", "#a8e9ff", "#7de4ff", t["RAMP60"])

    # Semantic. Film budget: cyan structure / white focal / amber semantic /
    # red alert — green is NOT in the palette, so the OK state reads as a
    # white focal value.
    t["OK"] = t["FOCAL"]
    t["WARN"] = a["WARN"]
    t["ERR"] = a["ERR"]

    # Holographic decor tints — the dense rim texture must stay VISIBLE
    # against the ground, so each look sets its own alpha step.
    t["GRID"] = _ramp(a["GRID"])          # dot grid, outermost guide circles
    t["SCAN"] = _ramp(a["SCAN"])          # scanline tint
    t["HOLO_DIM"] = _ramp(a["HOLO_DIM"])  # faint decor strokes
    t["HOLO"] = _ramp(a["HOLO"])          # decor strokes
    t["EDGE"] = t["RAMP60"]               # corner brackets, sweep lead, orbit dots
    t["FRAME"] = _ramp(a["FRAME"])        # 1px window outline (projected-panel edge)

    # ---- Luminous-density pass (lower-panel volume) ----------------------
    # The refs get their light from DENSITY of dim cyan wireframe filling
    # the volume, not one bright centerpiece. The transcript ground lifts
    # one step toward the reactor panel tone and every lower-panel decor
    # tint re-derives against that lifted ground (canvas items have no
    # alpha).
    TV_BG = _ramp(0.05)                   # transcript canvas ground
    _TV_RGB = _rgb(TV_BG)
    t["TV_BG"], t["_TV_RGB"] = TV_BG, _TV_RGB

    def _tv(alpha: float) -> str:
        """CYAN over the lifted transcript ground at alpha, pre-blended."""
        return _hex(_mix(_TV_RGB, _CY_RGB, alpha))

    def _tvglass(a_cyan: float, a_white: float = 0.0) -> str:
        """Glass tint over the lifted transcript ground."""
        return _hex(_mix(_mix(_TV_RGB, _CY_RGB, a_cyan), _WHITE, a_white))

    t["TV_SCANLINE"] = _ramp(0.085)       # 1px scanline rows — ONE step over TV_BG
    t["TV_POOL"] = tuple(_tv(x) for x in (0.02, 0.045, 0.07, 0.095))
                                          # radial light pool ovals, outermost first
    t["TV_GRID"] = _tvglass(0.18, 0.03)   # dot grid, one step brighter
    t["TV_WIRE"] = _hex(_mix(_rgb(_tv(0.11)), _TV_RGB, 0.35))
                                          # ambient wireframe strokes — pulled 35%
                                          # back toward the ground: the dome is an
                                          # EXTREMELY faint background, never a
                                          # competitor to the cards
    t["TV_WIRE2"] = _tv(0.16)             # brighter wireframe step / bright motes

    # Lit-dome pass (finale): the lower projection dome is a WIREFRAME the
    # light lives in, not a smudge — ring tone with a 1px dark offset ghost
    # (emboss) and a perspective floor grid seating the dome on a lit
    # surface. Ring tone likewise pulled 35% toward the ground (faint).
    t["TV_RING"] = _hex(_mix(_rgb(_tvglass(0.20, 0.02)), _TV_RGB, 0.35))
    t["TV_RING_GHOST"] = _tv(0.05)        # 1px offset emboss arc
    t["TV_NODE"] = _tvglass(0.80, 0.40)   # node dots / mote pulses
    t["TV_FLOOR"] = _tv(0.08)             # perspective floor grid

    # Reactor→transcript seam dissolve: the transcript's lit top row tone
    # (mirrors views.GRAD_PEAK over TV_BG) and pre-blended 1px line steps
    # walking the reactor ground down into it so the glow bleeds across
    # the panel boundary instead of stopping at a hard edge.
    TV_TOP = _hex(_mix(_TV_RGB, _CY_RGB, a["TV_LIFT"]))   # == TV_BG in holo
    t["TV_TOP"] = TV_TOP
    t["SEAM_STEPS"] = tuple(
        _hex(_mix(_BG_RGB, _rgb(TV_TOP), ((i + 1) / 20) ** 1.25))
        for i in range(20))

    # Two-stroke fake glow (the reactor-arc treatment, exported for the
    # rest of the app): a WIDE dim underlay stroke beneath a NARROW bright
    # core.
    t["GLOW_UNDER"] = _tvglass(0.20, 0.03)
    t["ARC_BRIGHT"] = _glass(0.87, 0.18)

    # Side-rail micro-labels — desaturated slate-cyan at comfortable contrast.
    t["RAIL"] = _glass(*a["RAIL"])

    # Reactor / app states. Speaking signals by driving the core and one
    # scan arc WHITE-HOT over unchanged cyan structure (film rule: color is
    # state, and the focal state value is white — never green).
    t["STATE_COLORS"] = {
        "idle":      t["CYAN_DIM"],
        "listening": CYAN,
        "thinking":  t["WARN"],
        "speaking":  t["FOCAL"],
        "waiting":   t["WARN"],       # Claude is waiting on a permission answer
        "working":   t["CYAN_DIM"],   # a Claude task is running
        "error":     t["ERR"],
        "offline":   t["FAINT"],
    }

    globals().update(t)
    LOOK_TOKENS = frozenset(k for k in t if not k.startswith("_"))
    LOOK = name
    apply_scale(_SCALE)


def select_look(name: Optional[str]) -> str:
    """Switch every token to `name` ("holo" | "classic") and return the
    look actually applied. Unknown names fall back to the default with a
    warning rather than raising: a typo in assistant.json must not cost
    him the window. Case-insensitive."""
    key = (name or "").strip().lower()
    if key not in _LOOKS:
        if key:
            log.warning("unknown ui look %r; using %s", name, DEFAULT_LOOK)
        key = DEFAULT_LOOK
    _derive(key)
    return key


def resolve_look(env: Mapping[str, str] = os.environ,
                 get_option: Optional[Callable] = None) -> str:
    """Which look to start in: env JARVIS_LOOK wins (a one-off run, the
    judge harness), then assistant.json's console.look (what the voice
    command writes), then the default. Only known names count; anything
    else falls through to the next source."""
    raw = env.get(ENV_KEY) if env is not None else None
    key = (raw or "").strip().lower()
    if key in _LOOKS:
        return key
    if raw:
        log.warning("bad %s %r; ignoring", ENV_KEY, raw)
    if callable(get_option):
        try:
            opt = get_option(OPTION_KEY, None)
        except Exception:                 # noqa: BLE001 - config boundary
            log.debug("%s unreadable", OPTION_KEY, exc_info=True)
            opt = None
        key = (str(opt) if opt is not None else "").strip().lower()
        if key in _LOOKS:
            return key
        if opt is not None:
            log.warning("bad %s %r in the config; using %s",
                        OPTION_KEY, opt, DEFAULT_LOOK)
    return DEFAULT_LOOK


# Chamfer cut for HUD panels (design units; scaled with RADIUS by apply_scale)
CHAMFER = 10

# Spacing (8px grid) — design units at the 96-dpi baseline. apply_scale()
# mutates these once at startup for the global UI scale factor S.
PAD = 16
PAD_S = 8
PAD_L = 24
RADIUS = 10


def apply_scale(scale: float) -> None:
    """Scale the spacing tokens by the global UI scale S. Called once by
    MainWindow before any widgets are built; idempotent (always derives
    from the design-unit baseline, so repeated calls don't compound).
    The scale is remembered so a later select_look() keeps it."""
    global PAD, PAD_S, PAD_L, RADIUS, CHAMFER, _SCALE
    _SCALE = scale
    PAD, PAD_S, PAD_L, RADIUS, CHAMFER = (
        max(1, round(v * scale)) for v in _BASE_SPACING)


_derive(DEFAULT_LOOK)

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
