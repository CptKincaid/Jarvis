"""Composite views for the Jarvis V3 main window.

TranscriptView — scrolling conversation of flat Cards, each its own canvas
                window item on a PAD gutter over a faint UNLABELED
                holographic backdrop (click-to-copy, smart autoscroll,
                confidence left-rule on user cards, streaming partial ghost).
CommandBar    — chamfered text entry + 44px circular terminal button
                (Claude's tmux pane) + 44px circular mic button; Enter
                publishes UserUtterance(text, source='typed') on the bus.
SettingsDrawer— 320px slide-over bound to CONFIG via bind_config(), plus
                the Assistant section (briefing / autostart toggles via
                services.get_option / set_option).
StatusStrip   — wake-word state (left), the PROJECT chip (only while a
                Claude project is active) + the ONLY machine-telemetry
                site: CPU / GPU / MEMORY segments (right). No centre text.

All colors and fonts come from jarvis.ui.theme tokens. No Tk root is
constructed at import time.
"""
from __future__ import annotations

import math
import random
import re
import threading
import time
import tkinter as tk
from datetime import datetime
from typing import Callable, Optional

from jarvis.campreview import OPTION_ENABLED as CAMERA_PREVIEW_OPTION
from jarvis.handstage import OPTION_ENABLED as GESTURE_OPTION
from jarvis.config import CONFIG, MACHINE, PATHS
from jarvis.events import UserUtterance, bus
from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.widgets import (BarGradient, Card, RoundButton, Toast, Toggle,
                               Tooltip, chamfer_rect, frame_rect, measure, px,
                               ui_display, ui_font, ui_mono)

log = get_logger("ui.views")

MAX_CARDS = 200
DOT_SPACING = 24            # transcript backdrop dot grid, design units
CARD_PAD = 12               # card inner padding, design units
CARD_PAD_S = 8              # compact (progress) card inner padding
PROGRESS_MAX = 12           # visible lines in the Claude progress card
BTN = 44                    # round command-bar buttons (mic, terminal)
PROJECT_MIN_CHARS = 6       # the PROJECT chip value never shrinks below

# Transcript atmosphere: ambient vertical gradient (the reactor glow pool
# above visibly BLEEDS into the transcript top, fading deep) and sparse dim
# dust motes.
GRAD_PEAK = 0.13            # top-of-canvas cyan blend, settling to BG
GRAD_FADE = 0.95            # fraction of the height the fade spans
GRAD_EXP = 1.7              # falloff exponent — lower carries light deeper
TV_MOTES = 8                # sparser + dimmer than the reactor stage
DOME_MOTES = 8              # slow drifters INSIDE the projection dome,
                            # with an occasional brightness pulse
ATMO_MS = 33                # ~30fps coords-only atmosphere updates (was
                            # 100: a 10 fps drift under a 60 fps sphere read
                            # as stutter). The physics steps by ATMO_MS, so
                            # the drift SPEED does not change with the rate.

# Ghost projection dome: an EXTREMELY faint, UNLABELED background behind
# the conversation — light pool, floor grid, concentric arcs with a 1px
# emboss ghost, a single 1px horizon arc, motes. Centred at exactly 50%
# of the panel width. No spokes, no node dots, no ticks, zero text; the
# ring/wire tones are pulled 35% toward the ground (theme) so nothing in
# it competes with card text. Static, redrawn only with the dot grid on
# resize/scroll settle; cards (window items) always render above.
RING_CX, RING_CY = 0.50, 0.76     # center as fractions of the viewport
RING_R1, RING_R2 = 0.46, 0.58     # radii as fractions of the width
RING_EXTRA = (                    # ambient wireframe arcs: (rf, start, ext)
    (0.15, -20, 220),
    (0.22, 22, 118),
    (0.33, -12, 148),
    (0.70, 20, 130))
POOL_FRACS = (0.46, 0.34, 0.23, 0.13)  # light-pool oval radii /width, out→in
SCAN_STEP = 7                     # scanline pitch, design units (6-8 spec)

# Anchored holo-panels, not a chat app: YOU cards shrink-wrap their text
# between 35% and 70% of the usable width, hugging the right gutter;
# JARVIS cards are 85% hugging the left.
YOU_MINW, YOU_MAXW, JARVIS_MAXW = 0.35, 0.70, 0.85


def you_card_width(text_px: int, usable: int, pad: int,
                   min_frac: float = YOU_MINW,
                   max_frac: float = YOU_MAXW) -> int:
    """Shrink-wrapped YOU card width: the widest text line plus the inner
    padding, clamped to [min_frac, max_frac] of the usable column
    (pure — unit tested)."""
    lo = int(usable * min_frac)
    hi = int(usable * max_frac)
    want = int(text_px) + 2 * pad + 4
    return max(lo, min(hi, want))

# ------------------------------------------------------- the ghost card
#
# The partial ("ghost") card is a LISTENING indicator: italic, muted, YOU
# geometry, no head row. Two symptoms from the 2026-08-31 voice test come
# straight out of what it was willing to render and how long it kept it:
#
#   * "2- presentation or just 2- keeps showing up after a response and hes
#     listening back for me" -- a preview left standing after the mic shut.
#   * a card holding "2,000,000,000,000,000,..." (100+ digits) at 20:47,
#     right after "set a reminder to drink water every 45 minutes".
#
# Neither is the number formatter or the interval parser: nothing on the
# reply path ever produced those strings. Transcriber.partial() decodes a
# SHORT buffer with no VAD and no noise gate every 0.9 s, and on room noise
# Whisper babbles -- digit runs and dangling fragments are its house style.
# The decoder cannot be made honest, so the card refuses to draw the two
# shapes that are never a command in progress, and takes itself down on "".
PARTIAL_MAX_CHARS = 200     # a preview is a glance, not a document
PARTIAL_MAX_DIGITS = 9      # "2,000,000,000" is already past anything spoken
_DIGIT_RUN_RE = re.compile(r"\d(?:[\d,. \u00a0]*\d)?")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def partial_display_text(text) -> str:
    """What the ghost card should show for a raw partial decode; "" means
    take the card DOWN (pure — unit tested).

    Drops, in order: blank text; text with no letter at all ("2-", "...",
    "2,000,000" — a preview made only of digits and punctuation is noise
    babble, and a real command is never worth previewing as one); text
    carrying a run of more than PARTIAL_MAX_DIGITS digits (the 20:47
    card, which is also unbreakable by Tk's word wrap and so spills past
    the card edge). Anything longer than PARTIAL_MAX_CHARS keeps its TAIL,
    since the newest words are the point of a live preview.
    """
    text = " ".join((text or "").split())
    if not text:
        return ""
    if not _LETTER_RE.search(text):
        return ""
    for run in _DIGIT_RUN_RE.findall(text):
        if sum(ch.isdigit() for ch in run) > PARTIAL_MAX_DIGITS:
            return ""
    if len(text) > PARTIAL_MAX_CHARS:
        text = "…" + text[-(PARTIAL_MAX_CHARS - 1):]
    return text


_TEMP_RE = re.compile(r"(\d+)\s*°(?:\s+(\d+)\s*%)?")


def fmt_temps(seg: str) -> str:
    """Status-bar value from one temps segment: 'cpu 40° 1%' → '40°C · 1%',
    'gpu 38°' → '38°C', anything unparsable → '--'."""
    m = _TEMP_RE.search(seg or "")
    if not m:
        return "--"
    temp, pct = m.group(1), m.group(2)
    return f"{temp}°C · {pct}%" if pct is not None else f"{temp}°C"


def split_temps(text: str) -> dict:
    """'cpu 40° 1% · gpu 38° 5%' → {'cpu': 'cpu 40° 1%', 'gpu': …}."""
    out = {}
    for seg in (text or "").split(" · "):
        seg = seg.strip()
        if seg:
            out[seg.split()[0].lower()] = seg
    return out


def command_bar_field_px(bar_w: int, buttons: int = 2, pad: int = 16,
                         pad_s: int = 8, btn: int = BTN) -> int:
    """Width left for the command field in a `bar_w`-wide bar (design
    units): `[PAD] field [PAD_S] terminal [PAD_S] mic [PAD]`. Pure — the
    460-px minimum window must leave >= 300 for the field."""
    return bar_w - pad - pad_s - buttons * btn - (buttons - 1) * pad_s - pad


STANDBY_ALPHA = 0.45        # the board's opacity while the chair is empty
STANDBY_ALPHA_FLOOR = 0.2   # never so faint that he cannot find the window


def standby_alpha(at_desk, enabled: bool = True,
                  dim: float = STANDBY_ALPHA) -> float:
    """The window opacity for a desk state (jarvis/deskpresence.py).

    Full brightness whenever he is there, whenever standby is switched off,
    and -- the case that matters -- whenever the desk state is UNKNOWN: a
    session with no Mutter idle monitor must look exactly like today's app,
    never like a board someone dimmed and forgot.
    """
    if at_desk is not False or not enabled:
        return 1.0
    try:
        value = float(dim)
    except (TypeError, ValueError):
        value = STANDBY_ALPHA
    return max(STANDBY_ALPHA_FLOOR, min(1.0, value))


def fmt_project_chip(slug: str, budget: int) -> str:
    """Status-bar PROJECT value: the slug uppercased, ellipsized to
    `budget` characters. The budget never goes below PROJECT_MIN_CHARS
    (the ellipsis counts as one). Empty slug → ''."""
    text = (slug or "").strip().upper()
    if not text or budget <= 0:
        return ""
    if len(text) <= max(budget, PROJECT_MIN_CHARS):
        return text
    keep = max(PROJECT_MIN_CHARS, budget) - 1
    return text[:keep] + "…"


def plan_strip(total_w: int, left_w: int, project_fixed_w: int, char_w: int,
               slug_len: int, seg_ws, min_chars: int = PROJECT_MIN_CHARS):
    """Status-strip fitting plan (pure). `left_w` = wake-word segment up
    to its hairline, `project_fixed_w` = the PROJECT segment without its
    value, `char_w` = one mono character, `seg_ws` = [(name, width)] of
    the telemetry cluster including margins. Returns (value_chars,
    hidden): the value budget shrinks first (never below min_chars — or
    the slug length when shorter); when even that does not fit, MEMORY
    yields (the least urgent telemetry), then the chip itself (0 chars).
    No project → (0, []) and every telemetry segment stays."""
    if slug_len <= 0:
        return 0, []
    hidden: list = []
    yield_order = ["MEMORY"]
    need = min(min_chars, slug_len)
    while True:
        right = sum(w for name, w in seg_ws if name not in hidden)
        free = total_w - left_w - right - project_fixed_w
        chars = min(slug_len, free // char_w) if char_w > 0 else 0
        if chars >= need:
            return chars, hidden
        if yield_order:
            hidden.append(yield_order.pop(0))
            continue
        # The chip itself goes -- and with it every yield made on its
        # behalf. Until 2026-09-03 this returned (0, hidden) with MEMORY
        # still in it, so at his 918-px strip with 'jarvis' active the
        # footer lost the memory figure AND never drew the PROJECT chip
        # that had displaced it (measured: level 0 with MEMORY yielded
        # leaves 75 px for the value, 5 characters, one short of the
        # six-character floor). No chip means nothing was displaced.
        return 0, []


def curfew_choice_values(curfew) -> tuple:
    """The two picker values for a curfew window (pure).

    ``None`` -- the curfew switched off -- still shows the SHIPPED window
    rather than a blank: a picker with no value in it reads as "there is no
    such setting", and the toggle beside it is what says whether it is on.
    """
    from jarvis.sensing import (DEFAULT_CURFEW_END, DEFAULT_CURFEW_START,
                                fmt_hhmm)
    if not curfew:
        return fmt_hhmm(DEFAULT_CURFEW_START), fmt_hhmm(DEFAULT_CURFEW_END)
    return fmt_hhmm(curfew[0]), fmt_hhmm(curfew[1])


def fmt_temps_compact(seg: str) -> str:
    """Temperature alone from one temps segment: 'cpu 53° 7%' → '53°',
    unparsable → '--'. NOT what the compact strip level shows any more --
    see fmt_temps_live."""
    m = _TEMP_RE.search(seg or "")
    return f"{m.group(1)}°" if m else "--"


def fmt_temps_live(seg: str) -> str:
    """Compact status-bar value from one temps segment: 'cpu 53° 7%' →
    '53° 7%' (the °C and the middot go, the utilisation stays),
    unparsable → '--'.

    2026-09-02, verbatim: "i dont think the CPU and GPU are updating".
    They were. The strip elides to level 1 at his 920-px window and level
    1 used to drop the PERCENTAGE and keep only the temperature -- which
    on this box moves a degree or two over minutes, so the compact rung
    read as a dead strip. The utilisation is the number that actually
    moves, and the one he can act on.
    """
    m = _TEMP_RE.search(seg or "")
    if not m:
        return "--"
    temp, pct = m.group(1), m.group(2)
    return f"{temp}° {pct}%" if pct is not None else f"{temp}°"


def fmt_load(seg: str) -> str:
    """Utilisation alone from one temps segment: 'cpu 53° 7%' → '7%'.

    The narrow rung. When even 'CPU 53° 7%' will not fit, the temperature
    is what goes and the percentage is what stays: it is the number that
    moves second to second and the one he can act on, and a strip that
    drops BOTH of them (which is what the ladder used to do -- straight
    from temperature-only to memory-only) goes blank at exactly the load
    it exists to report. A segment with no percentage in it has only its
    temperature to show ('gpu 38°' → '38°'); unparsable → '--'.
    """
    m = _TEMP_RE.search(seg or "")
    if not m:
        return "--"
    return f"{m.group(2)}%" if m.group(2) is not None else f"{m.group(1)}°"


# Rungs of the telemetry ladder, widest first (telemetry_segments).
TELEMETRY_LEVELS = 4


def telemetry_segments(temps_text: str, mem_text: str, level: int) -> list:
    """The telemetry cluster at an elision `level` as [(label, value)]:

      0 full     ('CPU' '53°C · 7%')  ('GPU' '44°C · 0%')  ('MEM' '47.5/122 GB')
      1 compact  ('CPU' '53° 7%')     ('GPU' '44° 0%')     (''    '47.5/122 GB')
      2 load     ('CPU' '7%')         ('GPU' '0%')         (''    '47.5/122 GB')
      3 minimal                                            (''    '47.5/122 GB')

    Pure; the strip picks the level with plan_telemetry from measured
    widths.

    THE LOAD RUNG EXISTS BECAUSE LEVEL 1 CAN TIP. Keeping the utilisation
    at level 1 costs it: measured at S=2.0 the compact rung went 430 px →
    570, against the 650 his 918-px strip leaves beside 'WAKE WORD OFF'.
    That is still slack at 63° 7% (570 px), and it still holds with both
    processors pinned at 100% (640). But a three-digit temperature under
    load ('cpu 100° 100% · gpu 88° 100%', '121.7/122 GB') asks 654 and
    tips the plan a rung -- and the rung below used to be memory ALONE, so
    CPU and GPU vanished at precisely the load this change exists to
    surface. The load rung asks 528 in that same worst case: the ladder
    now drops the temperatures before it drops the numbers that move.

    The memory VALUE (fmt_mem_gb) carries used/total rather than the label
    carrying the word MEMORY, because every rung but the first drops the
    label and a bare '59.9 GB' on a 122 GB box is exactly what he had to
    ask about. 'MEM' rather than 'MEMORY' at level 0 pays for the wider
    value: the measured level-0 cluster comes to 718 px against the 770
    the default window leaves, so every layout that showed the full
    cluster still does.
    """
    temps = split_temps(temps_text)
    mem = (mem_text or "").strip() or "--"
    if level <= 0:
        return [("CPU", fmt_temps(temps.get("cpu", ""))),
                ("GPU", fmt_temps(temps.get("gpu", ""))),
                ("MEM", mem)]
    if level == 1:
        return [("CPU", fmt_temps_live(temps.get("cpu", ""))),
                ("GPU", fmt_temps_live(temps.get("gpu", ""))),
                ("", mem)]
    if level == 2:
        return [("CPU", fmt_load(temps.get("cpu", ""))),
                ("GPU", fmt_load(temps.get("gpu", ""))),
                ("", mem)]
    return [("", mem)]


def plan_telemetry(total_w: int, left_w: int, project_fixed_w: int,
                   char_w: int, slug_len: int, levels,
                   min_chars: int = PROJECT_MIN_CHARS):
    """Which elision level the telemetry cluster renders at (pure).
    `levels` = [seg_ws per level] as plan_strip takes them, widest
    (level 0, today's full cluster) first. Each level is planned with
    plan_strip exactly as today; the first whose left side (wake word +
    PROJECT chip) ends before its telemetry starts wins, so every layout
    that fitted before is unchanged and only a real collision elides.
    Returns (level, value_chars, hidden). 2026-09-01: at JARVIS_UI_SCALE
    2.0 on the 920-px window the wake-word text ended at 256 px and the
    full cluster started at 210 px — a 46-px overlap plan_strip could not
    see because, with no project, it has nothing to yield."""
    result = (0, 0, [])
    for i, seg_ws in enumerate(levels):
        chars, hidden = plan_strip(total_w, left_w, project_fixed_w, char_w,
                                   slug_len, seg_ws, min_chars)
        right = sum(w for name, w in seg_ws if name not in hidden)
        left_end = left_w + (project_fixed_w + chars * char_w if chars else 0)
        result = (i, chars, hidden)
        if left_end <= total_w - right:
            break
    return result


def levels_for_look(levels, look: Optional[str] = None) -> list:
    """The elision levels StatusStrip may plan with (pure) -- now all of
    them, in BOTH looks.

    Until 2026-09-02 this handed classic level 0 alone: the 09-01 review
    measured the elision as the one classic pixel change outside the holo
    sphere, and classic is the fallback that renders the 08-31 console
    token for token, so it kept the FOUND 08-26 wake-word/telemetry
    overprint by contract rather than drift.

    That contract could not survive this commit. Telling him whether the
    memory figure is used or free changed the level-0 value in BOTH looks
    ('47.5 GB' → '47.5/122 GB'), and with it classic's pixels: its cluster
    grew 710 → 718 px, i.e. the overprint on his 918-px strip went 59 px
    → 67. Given the choice between a classic that is 8 px MORE broken and
    a classic that elides, the elision wins -- and the guard's own comment
    always said so. Classic at 918 px now plans the compact rung (570 px
    beside a 267-px wake segment) instead of printing CPU over OFF.

    `look` is kept (and ignored) so the call sites and the look-explicit
    tests do not all have to change to say the same thing: this is still
    the ONE place a look-specific ladder would go.
    """
    return list(levels or [])


def fit_placeholder(avail_px: int, options) -> str:
    """The longest placeholder that fits the entry (pure). `options` =
    [(text, measured_px)] longest first; returns the first that fits, else
    the last (shortest) — a placeholder that truncates to 'Type a command
    — or say “Ja' (today's at scale 2.0: 631 px into 548) tells him
    nothing. Empty options → ''."""
    options = list(options or [])
    if not options:
        return ""
    for text, w in options:
        if w <= avail_px:
            return text
    return options[-1][0]


def header_spans(total_w: int, left_ws, right_ws) -> list:
    """Where a bar's packed children land, and how wide they end up (pure).

    A transcription of Tk 8.6's packer, generic/tkPack.c ArrangePacking:

      * each child is handed a FRAME out of a shrinking cavity, and the
        cavity is CLAMPED at zero rather than going negative, the frame
        shrinking to whatever is left instead (lines 752-756);
      * the child is then CLIPPED to its frame less its padding (786-790);
      * and a child left with no width at all is UNMAPPED -- Tk stops
        drawing it entirely (846-850).

    So right-packed pack siblings can never overlap: `frameX` for a right
    child is `cavityX + cavityWidth` and cavityWidth is never negative, so
    no parcel ever starts left of the cavity origin. An over-subscribed
    header does not print one chip on top of another; it TRUNCATES the last
    one packed, and then makes it vanish. That is the real 2026-09-02
    header report ("the word sensing is underneath the ready symbol"): at
    918 px the badge was cut to 124 px of its 168 -- capsule and the tail
    of the word sheared off, sitting flush against the wordmark -- and
    41 px of 214 in his worst state, a dot and a sliver against the pill.
    One child further along (a wider word, the extra header child a
    sibling branch is adding) Tk stops drawing it at all, and an absent
    badge and a badge reading SENSING must not look the same.

    `left_ws` / `right_ws` are [(name, req_w, pad_left, pad_right)] in
    PACKING order (left-packed children first, as the header packs them).
    Returns [(name, x0, x1, req_w, mapped)] -- x0..x1 is what is actually
    DRAWN, so `x1 - x0 < req_w` means Tk clipped it and `mapped` False
    means Tk unmapped it. spans_clipped() names both.
    """
    spans, cav_x, cav_w = [], 0, max(0, int(total_w))
    for to_left, items in ((True, left_ws), (False, right_ws)):
        for name, req, pad_l, pad_r in items or ():
            pad = pad_l + pad_r
            frame_w = req + pad
            cav_w -= frame_w
            if cav_w < 0:                 # tkPack.c 752-756: clamp, shrink
                frame_w += cav_w
                cav_w = 0
            if to_left:
                frame_x = cav_x
                cav_x += frame_w
            else:
                frame_x = cav_x + cav_w
            drawn = min(req, frame_w - pad)   # tkPack.c 786-790: clip
            x0 = frame_x + pad_l
            spans.append((name, x0, x0 + max(0, drawn), req, drawn > 0))
    return spans


def spans_clipped(spans) -> list:
    """[(name, drawn_px, requested_px)] for every `header_spans` child the
    packer had to cut down, 0 drawn px meaning Tk unmapped it outright.

    This is the header's real fitting assertion. Pack siblings cannot
    overlap (see header_spans), so "nothing collides" is free and says
    nothing; "nothing is clipped" is what keeps the sensing badge whole --
    and keeps it on screen at all.
    """
    return [(name, x1 - x0, req) for name, x0, x1, req, _m in spans or ()
            if (x1 - x0) < req]


def tracked(text: str) -> str:
    """Tracked caps for the holo captions: 'JARVIS' → 'J A R V I S'. Tk has
    no letter-spacing, so we space the glyphs by hand — the ref HUD's
    labels are small, wide-set capitals. theme.caption() carries the rule
    for WHICH labels are tracked (surface labels, never row keys)."""
    return " ".join((text or "").upper().replace(" ", ""))


def speaker_color(role: str, look: Optional[str] = None) -> str:
    """The card head's speaker label colour (pure, call-time theme).

    Classic: CYAN_DIM for JARVIS, FAINT for YOU -- the 08-31 console.
    Holo (2026-09-03, ui-polish U03): one step up each, CYAN and MUTED.
    Measured on the 09-03 shot 08b: the JARVIS label at CYAN_DIM was
    5.57:1 against the transcript ground and the 'HH:MM · 1.0 s' stamp
    beside it 4.95:1 -- two equal captions, not label > meta -- and the
    YOU label was literally the stamp's colour, (110,134,158) against
    (109,132,157). The stamp keeps FAINT (theme.py: captions, timestamps);
    the speaker lifts to 11:1 / 9:1 so the head row reads label > meta."""
    look = theme.LOOK if look is None else look
    if role == "you":
        return theme.MUTED if look == "holo" else theme.FAINT
    return theme.CYAN if look == "holo" else theme.CYAN_DIM


# ------------------------------------------------- transcript top snap
# The viewport is pinned to the newest card, so whatever sits at the top
# edge is cut wherever the arithmetic lands: measured on the 09-03 shots,
# 08b a 178-px fragment 'draft from Tuesday…' with no head row, 09 'Here's
# your morning, sir.' halved, 22 the YES / NO buttons sliced through, 23 an
# empty bracket. Tk canvas window items always paint above canvas items
# and a Tk widget has no alpha, so no drawn fade can soften the cut. What
# CAN be done is to not show the fragment: while the view rests on the
# newest card, a card whose visible remainder is shorter than SNAP_FRAGMENT
# is unmapped, and the ground shows in its place. Taller remainders stay --
# a long reply half on screen is still worth reading -- and the moment he
# scrolls up (the view unpins) every card is shown, because a cut at the
# top edge is what a scrolled list looks like.
SNAP_FRAGMENT = 96          # design px: a head row plus ~two body lines


def snap_hidden(tops, heights, view_top, min_visible) -> list:
    """Which stacked cards to unmap for a view whose top edge is at
    `view_top` (pure): True for every card that STRADDLES the edge with
    fewer than `min_visible` px left below it -- never the last card,
    which is the one the view is resting on. `tops` / `heights` are the
    laid-out card y0 and height, in order."""
    out = []
    n = len(tops)
    for i, (y, h) in enumerate(zip(tops, heights)):
        straddles = y < view_top < y + h
        out.append(bool(straddles and i < n - 1
                        and (y + h - view_top) < min_visible))
    return out


def ground_is_flat(look: Optional[str] = None) -> bool:
    """Whether the transcript ground is one flat colour (pure). Holo: yes.
    Tk canvas windows have no alpha, so a card's interior can only ever be
    ONE colour, and card_look() makes it the ground (TV_BG) so the frame
    reads as drawn on the display. That only holds if the ground really is
    TV_BG everywhere a card can sit: the 09-01 review measured the pool
    ovals, floor grid and top gradient turning every holo card into a
    7-25 level darker slab -- the homemade look the brief asked us to
    remove. So in holo TranscriptView draws none of them (theme.TV_LIFT
    0.0 lands the stage seam on TV_BG too); the 1-step scanlines and dot
    grid stay. Classic keeps its lit pool and gradient (pixel oracle)."""
    return (theme.LOOK if look is None else look) == "holo"


def card_look(role: str) -> dict:
    """Card construction kwargs per look and turn role (pure, call-time).
    Classic: the filled RAISED slab every role uses today, byte for byte.
    Holo: a thin frame on the transcript ground — JARVIS turns carry the
    bright GLASS_EDGE outline and BRIGHT corner brackets, YOU turns a
    dimmer LINE outline, the listening ghost a dashed LINE frame, and
    progress/briefing the quiet FRAME outline. The fill is the transcript
    ground (TV_BG) so the frame reads as drawn ON the display, not as a
    panel laid over it; labels inside use the same fill."""
    if theme.LOOK != "holo":
        fill = theme.SURFACE if role in ("partial", "progress") else theme.RAISED
        return dict(fill=fill, style="slab")
    ground = theme.TV_BG
    if role == "jarvis":
        return dict(fill=ground, style="frame", edge=theme.GLASS_EDGE,
                    accent=theme.BRIGHT)
    if role == "user":
        return dict(fill=ground, style="frame", edge=theme.LINE,
                    accent=theme.RAMP60)
    if role == "partial":
        return dict(fill=ground, style="frame", edge=theme.LINE,
                    accent=theme.LINE, dash=(4, 4))
    if role == "approval":
        return dict(fill=ground, style="frame", edge=theme.GLASS_EDGE,
                    accent=theme.BRIGHT)
    return dict(fill=ground, style="frame", edge=theme.FRAME,
                accent=theme.GLASS_EDGE)


def progress_card_lines(lines, max_visible: int = PROGRESS_MAX) -> list:
    """Lines shown in the Claude progress card: the newest `max_visible`,
    where older ones collapse into a first line '… N earlier steps'
    (that line counts toward the visible budget)."""
    lines = [str(ln) for ln in (lines or []) if str(ln).strip()]
    max_visible = max(2, int(max_visible))
    if len(lines) <= max_visible:
        return lines
    keep = max_visible - 1
    earlier = len(lines) - keep
    return [f"… {earlier} earlier steps"] + lines[-keep:]


def _as_lines(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()]


def briefing_rows(sections) -> list:
    """Rows (label, text) for the briefing card from BriefingReady
    .sections: WEATHER, CALENDAR (one row per line), NEWS (three rows
    'title — source'), optional SPORTS / STOCKS. A label appears once per
    section; continuation rows carry an empty label. Empty sections are
    skipped.

    The same card carries the evening preview (CANVAS, TO-DOS, ALARM and
    the wake-up OFFER), the week forecast (a WEEK summary row, then one row
    per day labelled by its name), the pre-class dossier (ROOM or JOIN,
    LAST TIME, MAIL) and the class-start stage (CLASS). Unknown keys are
    still dropped."""
    s = sections or {}
    rows: list = []
    summary = str(s.get("summary") or "").strip()
    if summary:
        rows.append(("WEEK", summary))
    weather = str(s.get("weather") or "").strip()
    if weather:
        rows.append(("WEATHER", weather))
    for i, line in enumerate(_as_lines(s.get("calendar"))):
        rows.append(("CALENDAR" if i == 0 else "", line))
    # The pre-class dossier (jarvis/dossier.py) adds four keys to the same
    # card: where the class is (a room, or the join link when the event's
    # location is a URL), what he noted last time, and the unread mail about
    # it. Additive on purpose -- DUE and EXAM keep their places below, which
    # the briefing and the evening preview depend on.
    for key, label in (("room", "ROOM"), ("join", "JOIN")):
        text = str(s.get(key) or "").strip()
        if text:
            rows.append((label, text))
    last = str(s.get("last") or "").strip()
    if last:
        rows.append(("LAST TIME", last))
    # Coursework: DUE (one row per item) and the EXAM countdown, both
    # absent when Canvas is unconfigured or nothing qualifies.
    for i, line in enumerate(_as_lines(s.get("due"))):
        rows.append(("DUE" if i == 0 else "", line))
    exam = str(s.get("exam") or "").strip()
    if exam:
        rows.append(("EXAM", exam))
    study = str(s.get("study") or "").strip()
    if study:
        rows.append(("STUDY", study))
    items = []
    for item in (s.get("news") or [])[:3]:
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            source = str(item.get("source") or "").strip()
        else:
            title, source = str(item).strip(), ""
        if title:
            items.append(f"{title} — {source}" if source else title)
    for i, line in enumerate(items):
        rows.append(("NEWS" if i == 0 else "", line))
    for key in ("sports", "stocks"):
        for i, line in enumerate(_as_lines(s.get(key))):
            rows.append((key.upper() if i == 0 else "", line))
    # the week forecast: one row per day, "clear" when nothing is on it
    for day in (s.get("days") or []):
        if not isinstance(day, dict):
            continue
        label = str(day.get("label") or "").strip().upper()
        lines = _as_lines(day.get("items"))
        if not label:
            continue
        if not lines:
            rows.append((label, "clear"))
            continue
        for i, line in enumerate(lines):
            rows.append((label if i == 0 else "", line))
    for key, label in (("canvas", "CANVAS"), ("todos", "TO-DOS"),
                       ("reminders", "REMINDERS"),
                       # dossier: unread mail about this class;
                       # classflow: what was staged at the start of it
                       ("mail", "MAIL"), ("class", "CLASS")):
        for i, line in enumerate(_as_lines(s.get(key))):
            rows.append((label if i == 0 else "", line))
    for key, label in (("alarm", "ALARM"), ("offer", "OFFER")):
        text = str(s.get(key) or "").strip()
        if text:
            rows.append((label, text))
    return rows


def _rgb(color: str) -> tuple:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


class _WrapGroup:
    """Several value labels behind one card entry slot: the transcript
    re-wraps entry[1] on width changes with configure(wraplength=) and
    reads cget('text') for YOU-card shrink-wrapping; a briefing card has a
    label column, so every value label wraps at `wrap - offset`."""

    def __init__(self, labels: list, offset: int):
        self._labels = labels
        self._offset = offset

    def configure(self, wraplength=None, **_kw):
        if wraplength is None:
            return
        w = max(px(60), int(wraplength) - self._offset)
        for lbl in self._labels:
            lbl.configure(wraplength=w)

    def cget(self, _key):
        return ""


def _bind_tree(widget, sequence, fn):
    widget.bind(sequence, fn, add=True)
    for child in widget.winfo_children():
        _bind_tree(child, sequence, fn)


# ---------------------------------------------------------- TranscriptView
class TranscriptView(tk.Frame):
    """Vertically scrolling conversation over a faint, UNLABELED
    holographic backdrop. Each utterance is a chamfered Card placed as its
    OWN canvas window item on the PAD gutter (YOU right-aligned at 70% of
    the usable width, JARVIS left-aligned at 85%), stacked newest-at-
    bottom: bottom-anchored just above the command bar while the column
    is shorter than the viewport, scrolling otherwise. User cards carry a
    2px CYAN_DIM left rule whose length maps confidence. Clicking a card
    copies its text (toast). Autoscroll stays pinned to the bottom unless
    the user has scrolled up. Idle: zero cards, zero text — the command-
    bar placeholder is the only hint."""

    def __init__(self, parent, toast: Optional[Toast] = None, **kw):
        super().__init__(parent, bg=theme.TV_BG, **kw)
        self.toast = toast
        self._pinned = True
        self._cards: list = []            # [card, body_label, role, win_id]
        self._partial = None              # [card, label, role, win_id]
        self._progress = None             # (entry, lines) — the open run
        self._approvals: dict = {}        # request_id → {stamp, buttons}
        self._dots_key = None             # (w, h, origin) dot-grid guard
        self._dots_job = None
        self._layout_job = None
        self._wrap_w = None
        self._wrap_job = None
        # atmosphere: pre-rendered vertical gradient beneath the dot grid,
        # sparse motes moved at ~30fps (coords only, ATMO_MS)
        self._grad_key = None
        self._grad_photo = None
        self._grad_id = None
        self._motes: list = []
        self._dome_motes: list = []
        self._motes_init = False
        self._t0 = time.monotonic()
        self._atmo_job = None
        self._atmo_paused = False

        self.canvas = tk.Canvas(self, bg=theme.TV_BG, highlightthickness=0,
                                bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_config, add=True)
        # X11 wheel events, scoped to pointer-over
        self.canvas.bind("<Enter>", self._grab_wheel, add=True)
        self.canvas.bind("<Leave>", self._drop_wheel, add=True)
        self._atmo_job = self.after(400, self._atmo_tick)

    # ----------------------------------------------------------- scrolling
    def _grab_wheel(self, _e):
        self.canvas.bind_all("<Button-4>", lambda e: self._wheel(-1))
        self.canvas.bind_all("<Button-5>", lambda e: self._wheel(1))
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self._wheel(-1 if e.delta > 0 else 1))

    def _drop_wheel(self, _e):
        for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            self.canvas.unbind_all(seq)

    def _wheel(self, direction):
        self.canvas.yview_scroll(direction * 2, "units")
        self._pinned = self.canvas.yview()[1] >= 0.995
        self._apply_snap()
        self._schedule_dots()

    def _apply_snap(self):
        """Map / unmap the cards per snap_hidden for the current view
        (holo only; classic shows every card as it always did). Item
        states only -- no relayout, so it is cheap enough for every
        wheel tick."""
        c = self.canvas
        entries = self._entries()
        if not entries:
            return
        hide = [False] * len(entries)
        if theme.LOOK == "holo" and self._pinned:
            try:
                view_top = int(c.canvasy(0))
                tops = [c.coords(e[3])[1] for e in entries]
            except (tk.TclError, IndexError):
                return
            heights = [e[4] if len(e) > 4 and e[4] else 0 for e in entries]
            hide = snap_hidden(tops, heights, view_top, px(SNAP_FRAGMENT))
        for entry, off in zip(entries, hide):
            try:
                c.itemconfigure(entry[3], state="hidden" if off else "normal")
            except tk.TclError:
                pass

    # ------------------------------------------------------ dot backdrop
    def _schedule_dots(self):
        """Debounced backdrop refresh: redraw once per resize/scroll
        settle. The grid is snapped to the spacing, so a redraw after a
        scroll shows an identical pattern (no perceived motion)."""
        if self._dots_job is not None:
            try:
                self.after_cancel(self._dots_job)
            except tk.TclError:
                pass
        self._dots_job = self.after(150, self._draw_dots)

    def _draw_dots(self):
        self._dots_job = None
        c = self.canvas
        if not c.winfo_exists():
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w <= 2 or h <= 2:
            return
        sp = px(DOT_SPACING)
        top = int(c.canvasy(0))
        origin = (top // sp) * sp
        key = (w, h, origin)
        if self._dots_key == key:
            return
        self._dots_key = key
        c.delete("dotgrid")
        r = max(2, px(1))
        for yy in range(origin, top + h + sp, sp):
            for xx in range(sp, w, sp):
                c.create_rectangle(xx, yy, xx + r, yy + r, fill=theme.TV_GRID,
                                   outline="", tags="dotgrid")
        self._draw_scanlines(w, h, top)
        self._draw_ring(w, h, top)
        self._draw_gradient(w, h, top)
        # layering, bottom → top: gradient, light pool, scanlines, dot
        # grid, dome wireframe, motes (card window items always render
        # above every canvas item)
        c.tag_lower("holoring")
        c.tag_lower("dotgrid")
        c.tag_lower("tvscan")
        c.tag_lower("holopool")
        if self._grad_id is not None:
            c.tag_lower(self._grad_id)

    def _draw_scanlines(self, w: int, h: int, top: int):
        """1px horizontal scanline texture across the whole panel — rows
        every SCAN_STEP design px, ONE step lighter than the panel ground.
        Static create_line loop, rebuilt only on resize/scroll settle;
        snapped to the pitch so scrolling shows an identical pattern."""
        c = self.canvas
        c.delete("tvscan")
        step = max(2, px(SCAN_STEP))
        origin = (top // step) * step
        for yy in range(origin, top + h + step, step):
            c.create_line(0, yy, w, yy, fill=theme.TV_SCANLINE, width=1,
                          tags="tvscan")

    def _draw_ring(self, w: int, h: int, top: int):
        """Extremely faint UNLABELED dome behind the conversation, centred
        at 50% of the width: radial light pool, perspective floor grid,
        embossed concentric arcs and a single 1px horizon arc. No spokes,
        no nodes, no text, no ticks — it is background, not an
        instrument. Anchored to the viewport so it never scrolls out from
        under the void."""
        c = self.canvas
        c.delete("holoring")
        c.delete("holopool")
        if h < px(200):
            return
        cx, cy = w * RING_CX, top + h * RING_CY
        lw1 = max(1, px(1))
        flat = ground_is_flat()           # holo: cards need one ground colour

        # faint radial light pool: concentric pre-blended ovals, outermost
        # first so smaller/lighter ones stack on top
        for rf, col in (() if flat else zip(POOL_FRACS, theme.TV_POOL)):
            r = w * rf
            c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=col,
                          outline="", tags="holopool")

        # perspective floor grid: horizontals bunching toward the dome
        # base + verticals converging on the dome center
        fy0 = cy + px(2)
        fy1 = top + h - px(4)
        if fy1 - fy0 > px(30) and not flat:
            rows = 6
            y_first = fy0 + (fy1 - fy0) * (1 / rows) ** 1.75
            for k in range(1, rows + 1):
                yy = fy0 + (fy1 - fy0) * (k / rows) ** 1.75
                c.create_line(0, yy, w, yy, fill=theme.TV_FLOOR, width=lw1,
                              tags="holoring")
            conv = (y_first - fy0) / max(1, fy1 - fy0)
            for i in range(10):
                xb = w * i / 9.0
                xt = cx + (xb - cx) * conv
                c.create_line(xt, y_first, xb, fy1, fill=theme.TV_FLOOR,
                              width=lw1, tags="holoring")

        # dome wireframe: ring tone over a 1px offset dark ghost arc
        gof = max(1, px(1))
        r2 = w * RING_R2
        for r, start, extent in ([(r2, 30, 100)]
                                 + [(w * rf, s, e) for rf, s, e in RING_EXTRA]):
            c.create_arc(cx - r, cy - r + gof, cx + r, cy + r + gof,
                         style="arc", start=start, extent=extent,
                         outline=theme.TV_RING_GHOST, width=lw1,
                         tags="holoring")
            c.create_arc(cx - r, cy - r, cx + r, cy + r, style="arc",
                         start=start, extent=extent, outline=theme.TV_RING,
                         width=lw1, tags="holoring")
        # horizon arc: a single 1px ring-tone stroke, no glow, no ticks
        r1 = w * RING_R1
        c.create_arc(cx - r1, cy - r1, cx + r1, cy + r1, style="arc",
                     start=8, extent=150, outline=theme.TV_RING, width=lw1,
                     tags="holoring")

    def _draw_gradient(self, w: int, h: int, top: int):
        """Pre-rendered vertical ambience beneath the dot grid: slightly
        lit at the top (the reactor stage sits directly above), settling
        to BG. Rebuilt only when the canvas size settles at a new value;
        repositioned to the viewport top on scroll settle. Not drawn on a
        flat ground (holo, see ground_is_flat)."""
        c = self.canvas
        if ground_is_flat():
            return                        # the look is fixed for the session
        if self._grad_key != (w, h):
            try:
                from PIL import Image, ImageTk
            except Exception:
                log.exception("transcript gradient unavailable")
                return
            self._grad_key = (w, h)
            bg, cyan = _rgb(theme.TV_BG), _rgb(theme.CYAN)
            fade = max(1.0, h * GRAD_FADE)
            rows = []
            for yy in range(h):
                f = GRAD_PEAK * max(0.0, 1.0 - yy / fade) ** GRAD_EXP
                rows.append(tuple(int(b + (a - b) * f)
                                  for b, a in zip(bg, cyan)))
            col = Image.new("RGB", (1, h))
            col.putdata(rows)
            self._grad_photo = ImageTk.PhotoImage(
                col.resize((w, h), Image.NEAREST))
            if self._grad_id is None:
                self._grad_id = c.create_image(0, top, anchor="nw",
                                               image=self._grad_photo)
            else:
                c.itemconfigure(self._grad_id, image=self._grad_photo)
        c.coords(self._grad_id, 0, top)

    # ------------------------------------------------------- atmosphere
    def _atmo_tick(self):
        """~30fps loop (ATMO_MS): native coords moves for the dust motes —
        no redraws, no PIL, ~16 canvas calls per pass."""
        self._atmo_job = None
        if self._atmo_paused or not self.winfo_exists():
            return
        try:
            self._update_atmo()
        except tk.TclError:
            return
        self._atmo_job = self.after(ATMO_MS, self._atmo_tick)

    def pause_atmo(self, paused: bool) -> None:
        """Stop (or resume) the atmosphere loop. The room slab covers this
        canvas completely in ambient/standby, so at 30 fps the loop would
        otherwise spend ~1 ms x 30/s all night moving motes nobody can
        see. Paused motes simply hold; the loop restarts on resume."""
        paused = bool(paused)
        if paused == self._atmo_paused:
            return
        self._atmo_paused = paused
        if paused:
            if self._atmo_job is not None:
                try:
                    self.after_cancel(self._atmo_job)
                except tk.TclError:
                    pass
                self._atmo_job = None
        elif self._atmo_job is None:
            self._atmo_tick()

    def _update_atmo(self):
        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < px(60) or h < px(60):
            return
        top = int(c.canvasy(0))
        t = time.monotonic() - self._t0
        if not self._motes_init:
            # sparse dim motes, created once
            self._motes_init = True
            rng = random.Random(0xA7)
            for i in range(TV_MOTES):
                bright = i < 3
                sz = px(2) if bright else max(1, px(1))
                mid = c.create_rectangle(
                    0, 0, 0, 0, outline="",
                    fill=theme.TV_WIRE2 if bright else theme.TV_WIRE)
                self._motes.append([mid,
                                    rng.uniform(0, w),
                                    rng.uniform(0, h),
                                    rng.uniform(px(2), px(5)),
                                    rng.uniform(0.0, math.tau),
                                    rng.uniform(px(2), px(6)),
                                    rng.uniform(0.2, 0.6),
                                    sz])
            # dome interior drifters: polar sway around the dome center,
            # each pulsing bright for ~0.6s on its own 5-11s period
            self._dome_motes = []
            for i in range(DOME_MOTES):
                mid = c.create_rectangle(0, 0, 0, 0, outline="",
                                         fill=theme.TV_WIRE2)
                self._dome_motes.append([
                    mid,
                    rng.uniform(0.10, 0.40),        # radius / width
                    rng.uniform(20.0, 160.0),       # base angle, arc °
                    rng.uniform(0.05, 0.20),        # angular sway rad/s
                    rng.uniform(4.0, 14.0),         # sway amplitude °
                    rng.uniform(0.0, math.tau),     # phase
                    px(2),                          # size
                    rng.uniform(5.0, 11.0),         # pulse period s
                    False])                         # currently pulsing?
        dt = ATMO_MS / 1000.0
        wrap = h + px(8)
        for m in self._motes:
            mid, x0, y, vy, ph, wa, ws, sz = m
            y = (y - vy * dt) % wrap
            m[2] = y
            x = (x0 + math.sin(t * ws + ph) * wa) % w
            c.coords(mid, x, top + y - px(4), x + sz, top + y - px(4) + sz)
        # dome motes: coords-only sway inside the dome; the rare pulse is
        # a single itemconfigure on state change
        rcx, rcy = w * RING_CX, top + h * RING_CY
        for m in self._dome_motes:
            mid, rf, a0, ws, aa, ph, sz, period, lit = m
            ang = math.radians(a0 + math.sin(t * ws + ph) * aa)
            rr = w * rf * (1.0 + 0.05 * math.sin(t * 0.5 + ph))
            x = rcx + math.cos(ang) * rr
            y = rcy - math.sin(ang) * rr
            c.coords(mid, x, y, x + sz, y + sz)
            pulsing = ((t + ph) % period) < 0.6
            if pulsing != lit:
                m[8] = pulsing
                c.itemconfigure(
                    mid, fill=theme.TV_NODE if pulsing else theme.TV_WIRE2)

    # ------------------------------------------------------------- layout
    def _on_canvas_config(self, e):
        self._schedule_dots()
        if self._wrap_w != e.width:
            # Re-wrapping every card is expensive; debounce it so
            # interactive resizes only pay once the drag settles.
            self._wrap_w = e.width
            if self._wrap_job is not None:
                try:
                    self.after_cancel(self._wrap_job)
                except tk.TclError:
                    pass
            self._wrap_job = self.after(
                120, lambda w=e.width: self._apply_wrap(w))
        else:
            self._schedule_layout()       # height change: re-anchor

    def _entries(self) -> list:
        entries = list(self._cards)
        if self._partial is not None:
            entries.append(self._partial)
        return entries

    def _apply_wrap(self, width):
        self._wrap_job = None
        for entry in self._entries():
            card, label, role, win = entry[:4]
            _x, cw, wrap = self._card_geo(role, width, label.cget("text"))
            label.configure(wraplength=wrap)
            self.canvas.itemconfigure(win, width=cw)
        # Unmapped card bodies get no <Configure>, so re-fit explicitly
        # once the new wraplengths have settled.
        self.after_idle(self._relayout)

    def _schedule_layout(self):
        if self._layout_job is None:
            self._layout_job = self.after_idle(self._relayout)

    def _on_card_config(self, entry, event):
        """A card's own <Configure>: when its HEIGHT differs from the one
        the last layout used (Card.sync() grew it after the nested pack
        pass settled), stack again — otherwise the newest card would sit
        flush on the command bar with its bottom PAD_S eaten."""
        if len(entry) < 5:
            return
        if event.height != entry[4]:
            self._schedule_layout()

    def _relayout(self):
        """Stack the cards: y_i = y_{i-1} + h_{i-1} + PAD_S. Bottom-
        anchored while the column is shorter than the viewport (newest
        card just above the command bar); otherwise top-anchored with a
        scrollregion, pinned to the bottom on new cards. The pending pack
        passes are flushed first so every reqheight is final."""
        self._layout_job = None
        c = self.canvas
        if not c.winfo_exists():
            return
        try:
            c.update_idletasks()      # nested pack passes → true reqheights
        except tk.TclError:
            return
        W, H = c.winfo_width(), c.winfo_height()
        if W <= 2 or H <= 2:
            return
        entries = self._entries()
        heights = []
        for entry in entries:
            card = entry[0]
            card.sync()
            h = card.winfo_reqheight()
            heights.append(h)
            if len(entry) >= 5:
                entry[4] = h              # height this layout used
        gap = theme.PAD_S
        total = sum(heights) + gap * max(0, len(entries) - 1)
        if total + 2 * gap <= H:
            y = H - gap - total
            region = (0, 0, W, H)
        else:
            y = gap
            region = (0, 0, W, gap + total + gap)
        for entry, h in zip(entries, heights):
            card, label, role, win = entry[:4]
            x, cw, _wrap = self._card_geo(role, W, label.cget("text"))
            c.coords(win, x, y)
            c.itemconfigure(win, width=cw, state="normal")
            y += h + gap
        c.configure(scrollregion=region)
        if self._pinned:
            c.yview_moveto(1.0)
        self._apply_snap()
        self._schedule_dots()

    def _text_px(self, text: str) -> int:
        """Widest line of `text` in the card body font (0 on failure)."""
        try:
            f = ui_font(theme.SIZE_BODY)
            return max(measure(f, line) for line in (text or "").split("\n"))
        except Exception:
            return 0

    def _card_geo(self, role: str, width: Optional[int] = None,
                  text: Optional[str] = None):
        """(x, card_w, wraplength) for a card role: PAD gutters both
        sides; YOU cards shrink-wrap `text` between 35% and 70% of the
        usable width on the right, JARVIS / partial cards 85% on the
        left."""
        w = width or self.canvas.winfo_width()
        if w <= 2:      # canvas not laid out yet — assume default window
            w = px(488)
        gut = theme.PAD
        usable = max(px(120), w - 2 * gut)
        if role == "you":
            cw = you_card_width(self._text_px(text) if text else 0, usable,
                                px(CARD_PAD))
            x = w - gut - cw
        else:
            cw = int(usable * JARVIS_MAXW)
            x = gut
        wrap = max(px(80), cw - 2 * px(CARD_PAD) - px(4))
        return x, cw, wrap

    # ------------------------------------------------------------- content
    def add_user(self, text: str, confidence: Optional[float] = None):
        self.clear_partial()
        card = self._make_card("YOU", speaker_color("you"), text)
        if confidence is not None:
            card.set_left_rule(theme.CYAN_DIM, max(0.12, min(1.0, confidence)))
        return card

    def add_jarvis(self, text: str, rtt: Optional[float] = None):
        """Reply card. `rtt` (seconds, utterance → reply) renders ONCE,
        muted, in the head row: 'HH:MM · 1.2 s' — the only RTT site."""
        card = self._make_card("JARVIS", speaker_color("jarvis"), text, rtt=rtt)
        if theme.LOOK == "holo":
            # frame cards draw ONE short bright bracket from _glow[0]
            card.set_edge_glow((theme.ARC_BRIGHT,))
        else:
            card.set_edge_glow()
        return card

    def show_partial(self, text: str):
        """Streaming preview: a ghost card (YOU geometry) that updates in
        place until Transcribed/UserUtterance replaces it.

        Empty (or noise-only, see partial_display_text) text RETRACTS the
        card instead of being ignored. That is the only way the producer
        can take its own preview down: it is the one that knows the mic
        has shut, and a ghost outliving the capture reads as "he is
        listening back for me" when Jarvis is not listening at all."""
        text = partial_display_text(text)
        if not text:
            self.clear_partial()
            return
        if self._partial is None:
            x, cw, wrap = self._card_geo("you", None, text)
            look = card_look("partial")
            card = Card(self.canvas, pad=CARD_PAD, **look)
            lbl = tk.Label(card.body, text=text,
                           font=ui_font(theme.SIZE_BODY, "italic"),
                           fg=theme.MUTED, bg=look["fill"], justify="left",
                           anchor="w", wraplength=wrap)
            lbl.pack(fill="x")
            win = self.canvas.create_window(x, 0, anchor="nw", window=card,
                                            width=cw, state="hidden")
            self._partial = [card, lbl, "you", win, 0]
            card.bind("<Configure>",
                      lambda e, en=self._partial: self._on_card_config(en, e),
                      add=True)
        else:
            self._partial[1].configure(text=text)
        self._schedule_layout()

    def clear_partial(self):
        if self._partial is not None:
            card, _lbl, _role, win = self._partial[:4]
            self._partial = None
            try:
                self.canvas.delete(win)
                card.destroy()
            except tk.TclError:
                pass
            self._schedule_layout()

    def clear_all(self) -> int:
        """Empty the pane — every card, the ghost preview, the open
        progress run — and return how many cards came down.

        THE SCREEN ONLY. The conversation the model sees lives in
        jarvis/memory.py and the context engine; this touches neither, and
        the spoken line that goes with it says so. There is no undo: the
        cards are destroyed, not hidden.

        ONE card is exempt. An approval still waiting on him carries the
        only hand-answerable ALLOW / DENY for a Claude run that is BLOCKED
        on it — wiping it would strand that run until the approval timeout.
        It stays, alone, and the toast says why.

        Safe in every console mode: this is canvas work under the ambient
        slab, so a wipe asked for on the way out of the room lands the same
        as one asked for at the desk, and nothing here wakes the surface.
        """
        self.clear_partial()
        held = {id(info["card"]) for info in self._approvals.values()
                if info.get("card") is not None and not info.get("answered")}
        keep, removed = [], 0
        for entry in self._cards:
            card, _lbl, _role, win = entry[:4]
            if id(card) in held:
                keep.append(entry)
                continue
            try:
                self.canvas.delete(win)
                card.destroy()
            except tk.TclError:                 # the window is going away
                pass
            removed += 1
        self._cards = keep
        self._progress = None                   # a later line opens a new run
        # He may have scrolled up to read something before asking for the
        # wipe; an empty column left unpinned lands the NEXT card off-screen.
        self._pinned = True
        for info in self._approvals.values():   # no refs to destroyed widgets
            if info.get("answered"):
                info["card"] = None
        if keep and self.toast:
            self.toast.show("Question left standing", kind="info")
        self._schedule_layout()
        return removed

    def _card_head(self, card: Card, who: str, who_color: str,
                   rtt: Optional[float] = None) -> tk.Label:
        """Head row: speaker label (display SIZE_CAPTION semibold) left,
        'HH:MM[ · N.N s]' (mono SIZE_CAPTION FAINT) right. Returns the
        stamp label (approval cards append '· allowed' to it)."""
        # bg follows the card fill (holo cards sit on the transcript
        # ground, classic on RAISED); the speaker label is tracked caps
        # in holo — the ref HUD's wide-set captions — plain in classic.
        fill = card.body.cget("bg")
        head = tk.Frame(card.body, bg=fill)
        head.pack(fill="x")
        who_text = tracked(who) if theme.LOOK == "holo" else who
        tk.Label(head, text=who_text,
                 font=ui_display(theme.SIZE_CAPTION, "semibold"),
                 fg=who_color, bg=fill).pack(side="left")
        stamp = datetime.now().strftime("%H:%M")
        if rtt is not None:
            stamp = f"{stamp} · {rtt:.1f} s"
        lbl = tk.Label(head, text=stamp, font=ui_mono(theme.SIZE_CAPTION),
                       fg=theme.FAINT, bg=fill)
        lbl.pack(side="right")
        return lbl

    def _push_entry(self, card: Card, label, role: str, x: int, cw: int):
        """Register a finished card as its own canvas window item, prune
        the oldest past MAX_CARDS and schedule a stack. Any new card ends
        the open progress run (a later progress line starts a new one)."""
        win = self.canvas.create_window(x, 0, anchor="nw", window=card,
                                        width=cw, state="hidden")
        entry = [card, label, role, win, 0]    # [4] = height last laid out
        card.bind("<Configure>",
                  lambda e, en=entry: self._on_card_config(en, e), add=True)
        self._cards.append(entry)
        if len(self._cards) > MAX_CARDS:
            old = self._cards.pop(0)
            if self._progress is not None and self._progress[0] is old:
                self._progress = None
            try:
                self.canvas.delete(old[3])
                old[0].destroy()
            except tk.TclError:
                pass
        self._schedule_layout()
        return entry

    def _make_card(self, who: str, who_color: str, text: str,
                   rtt: Optional[float] = None) -> Card:
        self._progress = None
        role = "you" if who == "YOU" else "jarvis"
        x, cw, wrap = self._card_geo(role, None, text)
        look = card_look("user" if role == "you" else "jarvis")
        card = Card(self.canvas, pad=CARD_PAD, **look)
        self._card_head(card, who, who_color, rtt)
        body = tk.Label(card.body, text=text, font=ui_font(theme.SIZE_BODY),
                        fg=theme.INK, bg=look["fill"], justify="left",
                        anchor="w", wraplength=wrap)
        body.pack(fill="x", pady=(px(4), 0))
        _bind_tree(card, "<Button-1>", lambda e, t=text: self._copy(t))
        self._push_entry(card, body, role, x, cw)
        return card

    # ----------------------------------------------- assistant cards
    def add_progress(self, line: str):
        """One compact Claude progress card (mono SIZE_CAPTION MUTED, no
        head row, SURFACE fill, JARVIS geometry): consecutive lines
        append to the same card — at most PROGRESS_MAX visible, older
        ones collapsing into '… N earlier steps'. Any user / JARVIS card
        ends the run."""
        line = (line or "").strip()
        if not line:
            return
        if self._progress is None:
            x, cw, wrap = self._card_geo("jarvis", None, line)
            look = card_look("progress")
            card = Card(self.canvas, pad=CARD_PAD_S, **look)
            lbl = tk.Label(card.body, text="", font=ui_mono(theme.SIZE_CAPTION),
                           fg=theme.MUTED, bg=look["fill"], justify="left",
                           anchor="w", wraplength=wrap)
            lbl.pack(fill="x")
            lines: list = []
            _bind_tree(card, "<Button-1>",
                       lambda e, ls=lines: self._copy("\n".join(ls)))
            entry = self._push_entry(card, lbl, "jarvis", x, cw)
            self._progress = (entry, lines)
        entry, lines = self._progress
        lines.append(line)
        entry[1].configure(
            text="\n".join(progress_card_lines(lines, PROGRESS_MAX)))
        self._schedule_layout()

    def add_approval(self, request_id: str, question: str,
                     on_answer: Optional[Callable] = None,
                     yes_text: str = "ALLOW",
                     no_text: str = "DENY") -> Optional[Card]:
        """JARVIS card carrying a yes/no question with two RoundButtons →
        on_answer(request_id, bool). One card per request id (a repeat is
        ignored). Labels default to Claude's ALLOW / DENY; the
        "Was that for me?" prompt passes YES / NO."""
        if not request_id or request_id in self._approvals:
            return None
        self._progress = None
        x, cw, wrap = self._card_geo("jarvis", None, question)
        look = card_look("approval")
        fill = look["fill"]
        card = Card(self.canvas, pad=CARD_PAD, **look)
        stamp = self._card_head(card, "JARVIS", speaker_color("jarvis"))
        body = tk.Label(card.body, text=question, font=ui_font(theme.SIZE_BODY),
                        fg=theme.INK, bg=fill, justify="left",
                        anchor="w", wraplength=wrap)
        body.pack(fill="x", pady=(px(4), 0))
        body.bind("<Button-1>", lambda e, t=question: self._copy(t), add=True)
        row = tk.Frame(card.body, bg=fill)
        row.pack(fill="x", pady=(px(8), 0))

        def answer(allowed: bool, rid=request_id):
            self.resolve_approval(rid, allowed, mark=False)
            if on_answer:
                try:
                    on_answer(rid, allowed)
                except Exception:
                    log.exception("approval answer failed")

        allow = RoundButton(row, text=yes_text, kind="accent", bg=fill,
                            command=lambda: answer(True))
        allow.pack(side="left")
        deny = RoundButton(row, text=no_text, kind="default", bg=fill,
                           command=lambda: answer(False))
        deny.pack(side="left", padx=(theme.PAD_S, 0))
        if theme.LOOK == "holo":
            card.set_edge_glow((theme.ARC_BRIGHT,))
        else:
            card.set_edge_glow()
        # `answered` and `card` are clear_all's: a question still waiting on
        # him is the one card a wipe must leave standing. `done` cannot say
        # so -- answer() above resolves with mark=False, which disables the
        # buttons without ever setting it.
        self._approvals[request_id] = {"stamp": stamp, "stamp_text": stamp.cget("text"),
                                       "buttons": (allow, deny), "done": False,
                                       "answered": False, "card": card}
        self._push_entry(card, body, "jarvis", x, cw)
        return card

    def resolve_approval(self, request_id: str, allowed: bool,
                         mark: bool = True, yes_mark: str = "allowed",
                         no_mark: str = "declined"):
        """Disable the card's buttons; with `mark` append '· allowed' /
        '· declined' to its head row (ApprovalResolved). The
        "Was that for me?" card passes 'yes' / 'no' -- approval wording on a
        plain question reads as though permission was granted."""
        info = self._approvals.get(request_id)
        if not info:
            return
        info["answered"] = True
        for btn in info["buttons"]:
            try:
                btn.set_enabled(False)
            except tk.TclError:
                pass
        if mark and not info["done"]:
            info["done"] = True
            word = yes_mark if allowed else no_mark
            try:
                info["stamp"].configure(text=f"{info['stamp_text']} · {word}")
            except tk.TclError:
                pass

    def add_briefing(self, sections) -> Card:
        """ONE JARVIS card for a briefing: head row 'JARVIS HH:MM', then
        label / value rows from briefing_rows() (labels display
        SIZE_CAPTION semibold MUTED, values body INK)."""
        self._progress = None
        rows = briefing_rows(sections)
        x, cw, wrap = self._card_geo("jarvis", None, "")
        look = card_look("briefing")
        fill = look["fill"]
        card = Card(self.canvas, pad=CARD_PAD, **look)
        self._card_head(card, "JARVIS", speaker_color("jarvis"))
        lf = ui_display(theme.SIZE_CAPTION, "semibold")
        try:
            col_w = max([measure(lf, lab) for lab, _t in rows if lab] or [0])
        except Exception:
            col_w = px(60)
        col_w += theme.PAD_S
        grid = tk.Frame(card.body, bg=fill)
        grid.pack(fill="x", pady=(px(4), 0))
        grid.grid_columnconfigure(0, minsize=col_w)
        grid.grid_columnconfigure(1, weight=1)
        values = []
        for i, (lab, text) in enumerate(rows):
            tk.Label(grid, text=lab, font=lf, fg=theme.MUTED, bg=fill,
                     anchor="nw").grid(row=i, column=0, sticky="nw",
                                       pady=(px(3), 0))
            val = tk.Label(grid, text=text, font=ui_font(theme.SIZE_BODY),
                           fg=theme.INK, bg=fill, justify="left",
                           anchor="w", wraplength=max(px(80), wrap - col_w))
            val.grid(row=i, column=1, sticky="w", pady=(0, px(2)))
            values.append(val)
        if theme.LOOK == "holo":
            card.set_edge_glow((theme.ARC_BRIGHT,))
        else:
            card.set_edge_glow()
        plain = "\n".join(f"{lab} {text}".strip() for lab, text in rows)
        _bind_tree(card, "<Button-1>", lambda e, t=plain: self._copy(t))
        self._push_entry(card, _WrapGroup(values, col_w), "jarvis", x, cw)
        return card

    def _copy(self, text: str):
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
        except tk.TclError:
            log.exception("clipboard copy failed")
            return
        if self.toast:
            self.toast.show("Copied", kind="ok")


# -------------------------------------------------------------- CommandBar
class CommandBar(tk.Frame):
    """64px SURFACE bar: rounded entry + 44px circular terminal button +
    44px circular mic button (the terminal sits LEFT of the mic).

    Enter publishes UserUtterance(text, source='typed') on the bus and
    calls on_submit(text) when provided. Up/Down walk the typed-command
    history through on_history(-1) / on_history(+1) (shell semantics: Up
    from the newest backwards, Down forwards, None past the newest = an
    empty field). The mic button calls on_mic() and supports states idle /
    recording / disabled (set_mic_state). The terminal button calls
    on_terminal() and supports idle / open / working / waiting / disabled
    (set_terminal_state)."""

    PLACEHOLDER = "Type a command"                         # hotword off / no mic
    PLACEHOLDER_HOT = "Type a command — or say “Jarvis”"    # hotword on
    PLACEHOLDER_HOT_SHORT = "Type or say “Jarvis”"          # hotword on, narrow
    TERMINAL_STATES = ("idle", "open", "working", "waiting", "no_project",
                       "disabled")
    TIP_NO_PROJECT = "No project yet"
    TIP_OPEN = "Claude's terminal is open"

    def __init__(self, parent, on_submit: Optional[Callable] = None,
                 on_mic: Optional[Callable] = None,
                 on_terminal: Optional[Callable] = None,
                 on_history: Optional[Callable] = None,
                 terminal_available: bool = True, **kw):
        super().__init__(parent, bg=theme.SURFACE, height=px(64), **kw)
        self.pack_propagate(False)
        self.on_submit = on_submit
        self.on_mic = on_mic
        self.on_terminal = on_terminal
        self.on_history = on_history
        self._mic_state = "idle"
        self._term_state = "idle" if terminal_available else "disabled"
        self._term_available = terminal_available

        # --- chamfered entry field -----------------------------------
        # explicit width: Tk's canvas default is "10c" (640px on HiDPI),
        # which would starve the mic button out of the bar; expand=True
        # stretches it to the real space.
        field = tk.Canvas(self, bg=theme.SURFACE, highlightthickness=1,
                          width=px(200), height=px(40), bd=0)
        field.configure(highlightbackground=theme.SURFACE,
                        highlightcolor=theme.CYAN_DIM)
        field.pack(side="left", fill="x", expand=True,
                   padx=(theme.PAD, theme.PAD_S), pady=px(12))
        self._field = field
        # holo: the field is a hairline frame with a dark inset interior
        # (BG) so the strip reads as a recessed display, not a lit slab
        self._field_fill = theme.BG if theme.LOOK == "holo" else theme.RAISED
        self.entry = tk.Entry(field, bd=0, bg=self._field_fill, fg=theme.FAINT,
                              insertbackground=theme.CYAN,
                              font=ui_display(theme.SIZE_BODY), relief="flat",
                              highlightthickness=0)
        self._entry_win = field.create_window(px(34), px(20), anchor="w",
                                              window=self.entry)
        field.bind("<Configure>", self._draw_field, add=True)
        self._hotword = True
        self._placeholder = self.PLACEHOLDER_HOT
        self._showing_placeholder = True
        self.entry.insert(0, self._placeholder)
        self.entry.bind("<FocusIn>", self._focus_in, add=True)
        self.entry.bind("<FocusOut>", self._focus_out, add=True)
        self.entry.bind("<Return>", self._submit, add=True)
        self._bind_history(self.entry)

        # --- circular mic button (44px design) -----------------------
        self.mic = tk.Canvas(self, width=px(44), height=px(44),
                             bg=theme.SURFACE,
                             highlightthickness=1, bd=0, takefocus=1,
                             cursor="hand2")
        self.mic.configure(highlightbackground=theme.SURFACE,
                           highlightcolor=theme.CYAN_DIM)
        self.mic.pack(side="right", padx=(0, theme.PAD), pady=px(10))
        self.mic.bind("<ButtonRelease-1>", lambda e: self._mic_clicked())
        self.mic.bind("<Return>", lambda e: self._mic_clicked())
        self.mic.bind("<space>", lambda e: self._mic_clicked())
        self._mic_tip = Tooltip(self.mic, "")
        self._draw_mic()

        # --- circular terminal button (44px design), LEFT of the mic ---
        # packed side=right AFTER the mic, so it lands to the mic's left
        self.term = tk.Canvas(self, width=px(BTN), height=px(BTN),
                              bg=theme.SURFACE, highlightthickness=1, bd=0,
                              takefocus=1, cursor="hand2")
        self.term.configure(highlightbackground=theme.SURFACE,
                            highlightcolor=theme.CYAN_DIM)
        self.term.pack(side="right", padx=(0, theme.PAD_S), pady=px(10))
        self.term.bind("<ButtonRelease-1>", lambda e: self._term_clicked())
        self.term.bind("<Return>", lambda e: self._term_clicked())
        self.term.bind("<space>", lambda e: self._term_clicked())
        self._term_tip = Tooltip(self.term, self.TIP_NO_PROJECT)
        self.set_terminal_state(self._term_state, self.TIP_NO_PROJECT)

        # atmosphere: the bar ground is a soft pre-rendered gradient (lit
        # under the transcript glow, calm at the edges) — no region keeps
        # a flat fill; slightly stronger for the luminous pass
        self._grad = BarGradient(self, theme.SURFACE,
                                 [(0.0, 0.015), (0.55, 0.070), (1.0, 0.018)])

    # ----------------------------------------------------------- entry
    def _draw_field(self, event=None):
        w = self._field.winfo_width()
        h = self._field.winfo_height()
        if w < 4:
            return
        if getattr(self, "_field_last", None) == (w, h):
            return
        self._field_last = (w, h)
        self._field.delete("chrome")
        if theme.LOOK == "holo":
            self._draw_field_holo(w, h)
            return
        # dim hairline outline; brightness only in the corner brackets
        item = chamfer_rect(self._field, 1, 1, w - 2, h - 2,
                            fill=theme.RAISED, outline=theme.RAMP47, width=1,
                            tags="chrome")
        self._field.tag_lower(item)
        cut, alen, aw = theme.CHAMFER, px(14), max(1, px(1))
        # glass catch-light along the inner top edge (lit glass slab)
        self._field.create_line(1 + cut, 2, w - 2 - cut, 2,
                                fill=theme.GLASS_EDGE, width=aw,
                                tags="chrome")
        self._field.create_line(1, 1 + cut + alen, 1, 1 + cut, 1 + cut, 1,
                                1 + cut + alen, 1, fill=theme.BRIGHT,
                                width=aw, tags="chrome")
        self._field.create_line(w - 2, h - 2 - cut - alen, w - 2,
                                h - 2 - cut, w - 2 - cut, h - 2,
                                w - 2 - cut - alen, h - 2, fill=theme.BRIGHT,
                                width=aw, tags="chrome")
        # input chevron with the two-stroke underlay glow treatment:
        # offset dim ghosts beneath the bright core (text has no width,
        # so the wide underlay is faked with four 1px-offset copies)
        chev_font = ui_display(theme.SIZE_BODY, "semibold")
        g = max(1, px(1))
        for dx, dy in ((-g, 0), (g, 0), (0, -g), (0, g)):
            self._field.create_text(px(18) + dx, h // 2 + dy, text="❯",
                                    fill=theme.GLOW_UNDER, font=chev_font,
                                    tags="chrome")
        self._field.create_text(px(18), h // 2, text="❯", fill=theme.CYAN,
                                font=chev_font, tags="chrome")
        self._field.coords(self._entry_win, px(34), h // 2)
        self._field.itemconfigure(self._entry_win, width=w - px(50))

    def _draw_field_holo(self, w: int, h: int):
        """Holo command strip: a 1px GLASS_EDGE frame with corner notches
        over the dark inset, ONE bright CYAN '[' bracket on the left edge
        (the ref HUD marks its input line with a single bright corner,
        never a lit border) and a crisp chevron — no ghost-glow copies,
        which read as blur at 2x. The entry's width follows the frame and
        the placeholder is re-fitted to it (fit_placeholder)."""
        f = self._field
        item = frame_rect(f, 1, 1, w - 2, h - 2, fill=self._field_fill,
                          outline=theme.GLASS_EDGE, accent=theme.GLASS_EDGE,
                          tags="chrome")
        f.tag_lower(item)
        bw = max(1, px(2))
        tick = px(10)
        f.create_line(1 + tick, 1, 1, 1, 1, h - 2, 1 + tick, h - 2,
                      fill=theme.CYAN, width=bw, tags="chrome",
                      capstyle="projecting", joinstyle="miter")
        f.create_text(px(18), h // 2, text="❯", fill=theme.CYAN,
                      font=ui_display(theme.SIZE_BODY, "semibold"),
                      tags="chrome")
        f.coords(self._entry_win, px(34), h // 2)
        f.itemconfigure(self._entry_win, width=w - px(50))
        self._apply_placeholder()

    def _placeholder_for(self, hotword: bool) -> str:
        """The placeholder text for the hotword state. Classic: today's
        two strings, untouched. Holo: the longest of the three that fits
        the measured entry width — at JARVIS_UI_SCALE 2.0 the long form
        (631 px) never fitted the 548-px entry and clipped to 'Type a
        command — or say “Ja'."""
        long = self.PLACEHOLDER_HOT if hotword else self.PLACEHOLDER
        if theme.LOOK != "holo" or not hotword:
            return long
        last = getattr(self, "_field_last", None)
        if not last:
            return long
        avail = last[0] - px(50) - px(6)      # entry width less a breath
        try:
            font = ui_display(theme.SIZE_BODY)
            options = [(t, measure(font, t)) for t in
                       (self.PLACEHOLDER_HOT, self.PLACEHOLDER_HOT_SHORT,
                        self.PLACEHOLDER)]
        except Exception:
            return long
        return fit_placeholder(avail, options)

    def _apply_placeholder(self):
        text = self._placeholder_for(self._hotword)
        if text == self._placeholder:
            return
        self._placeholder = text
        if self._showing_placeholder:
            self.entry.delete(0, "end")
            self.entry.insert(0, self._placeholder)

    def _focus_in(self, _e):
        if self._showing_placeholder:
            self.entry.delete(0, "end")
            self.entry.configure(fg=theme.INK)
            self._showing_placeholder = False

    def _focus_out(self, _e):
        if not self.entry.get().strip():
            self.entry.delete(0, "end")
            self.entry.insert(0, self._placeholder)
            self.entry.configure(fg=theme.FAINT)
            self._showing_placeholder = True

    def set_placeholder(self, hotword: bool):
        """Truthful idle hint: mention the wake word only when the
        listener is actually on (and a mic exists)."""
        self._hotword = bool(hotword)
        self._placeholder = self._placeholder_for(self._hotword)
        if self._showing_placeholder:
            self.entry.delete(0, "end")
            self.entry.insert(0, self._placeholder)

    # -------------------------------------------------------- history
    def _bind_history(self, entry):
        """#135 "typed-command history — arrow keys do nothing".

        jarvis/history.py's prev()/next() were written for exactly these
        two keys ("the command bar's Up/Down navigation" is in its module
        docstring) and nothing was ever bound to them: the store, the
        cursor and the JSONL file all worked, the feature had no consumer.

        A method rather than two lines in __init__ so the wire can be
        asserted without a display (tests/test_ui_turn_control.py)."""
        entry.bind("<Up>", self._history_prev, add=True)
        entry.bind("<Down>", self._history_next, add=True)

    def _history_prev(self, _e=None):
        return self._history_step(-1)

    def _history_next(self, _e=None):
        return self._history_step(1)

    def _history_step(self, delta: int):
        """Put the neighbouring history entry in the field.

        Returns "break" whenever a provider is wired, so Tk's own Entry
        binding (Up/Down move the insert cursor to the start/end of the
        line) cannot also fire and leave the caret somewhere surprising.
        With nothing wired it returns None and the default behaviour is
        untouched."""
        if self.on_history is None:
            return None
        try:
            text = self.on_history(delta)
        except Exception:
            log.exception("on_history failed")
            return "break"
        self.set_text(text or "")
        return "break"

    def set_text(self, text: str):
        """Replace the field's contents (history recall).

        An empty string CLEARS the field without putting the grey
        placeholder back -- that is Down past the newest entry, where the
        caret is still in the box and hint text under a live caret reads
        as content he would have to delete. _focus_out restores the
        placeholder when he leaves the field, as it always did."""
        self.entry.delete(0, "end")
        if text:
            self.entry.configure(fg=theme.INK)
            self._showing_placeholder = False
            self.entry.insert(0, text)
            self.entry.icursor("end")
        else:
            # Not the placeholder unless the field has also lost focus:
            # showing grey hint text under a live caret reads as content
            # he would then have to delete.
            self._showing_placeholder = False
            self.entry.configure(fg=theme.INK)

    def _submit(self, _e=None):
        if self._showing_placeholder:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        bus.publish(UserUtterance(text=text, source="typed"))
        if self.on_submit:
            try:
                self.on_submit(text)
            except Exception:
                log.exception("on_submit failed")

    # ------------------------------------------------------------- mic
    def set_mic_state(self, state: str):
        """state: idle | recording | disabled"""
        self._mic_state = state
        self._mic_tip.set_text(
            "No microphone detected" if state == "disabled" else "")
        self.mic.configure(cursor="arrow" if state == "disabled" else "hand2")
        self._draw_mic()

    def _mic_clicked(self):
        if self._mic_state == "disabled":
            return
        self.mic.focus_set()
        if self.on_mic:
            try:
                self.on_mic()
            except Exception:
                log.exception("on_mic failed")

    def _draw_mic(self):
        c = self.mic
        c.delete("all")
        p = px   # glyph drawn in 44px design units, scaled by S
        recording = self._mic_state == "recording"
        ring = theme.FAINT if self._mic_state == "disabled" else theme.CYAN
        # 4 arc segments (gaps on the diagonals) that close into a full
        # ring while recording
        box = (p(2), p(2), p(42), p(42))
        holo = theme.LOOK == "holo"
        for k in range(4):
            if recording:
                start, extent = 90 * k - 45, 90
            else:
                start, extent = 90 * k - 33, 66
            c.create_arc(*box, start=start, extent=extent, style="arc",
                         outline=ring, width=max(1, p(1)) if holo else p(2))
        if recording:
            # filled CYAN disc with a stop glyph inside the closed ring
            c.create_oval(p(7), p(7), p(37), p(37), fill=theme.CYAN,
                          outline="")
            c.create_rectangle(p(17), p(17), p(27), p(27), fill=theme.BG,
                               outline="")
            return
        glyph = ring
        if not holo:
            # classic: the glyph sits on a RAISED disc; holo is an open
            # ring of light on the bar ground (no filled disc)
            c.create_oval(p(7), p(7), p(37), p(37), fill=theme.RAISED,
                          outline="")
        # mic glyph (capsule + cradle arc + stem) — shape from the tray
        # icon drawing at voice_input_gui.py 673-680, scaled to 44px
        c.create_oval(p(18), p(11), p(26), p(19), fill=glyph, outline="")
        c.create_rectangle(p(18), p(15), p(26), p(22), fill=glyph, outline="")
        c.create_oval(p(18), p(18), p(26), p(26), fill=glyph, outline="")
        c.create_arc(p(14), p(14), p(30), p(30), start=180, extent=180,
                     style="arc", outline=glyph, width=p(2))
        c.create_line(p(22), p(30), p(22), p(34), fill=glyph, width=p(2))
        c.create_line(p(17), p(34), p(27), p(34), fill=glyph, width=p(2))

    # -------------------------------------------------------- terminal
    def set_terminal_state(self, state: str, tooltip: Optional[str] = None):
        """state: idle (gapped CYAN ring) | open (closed CYAN ring, CYAN
        glyph — a terminal is attached and watching the session) |
        working (closed ring, FOCAL glyph) | waiting (WARN ring — a
        permission question is pending) | no_project (idle drawn FAINT:
        no session to attach to yet, still clickable) | disabled (FAINT
        and inert; tmux / gnome-terminal missing). A missing tmux or
        gnome-terminal pins the button to disabled whatever is asked."""
        if state not in self.TERMINAL_STATES:
            state = "idle"
        if not self._term_available:
            state = "disabled"
        self._term_state = state
        if tooltip is not None:
            self._term_tip.set_text(tooltip)
        self.term.configure(cursor="arrow" if state == "disabled" else "hand2")
        self._draw_term()

    def _term_clicked(self):
        if self._term_state == "disabled":
            return
        self.term.focus_set()
        if self.on_terminal:
            try:
                self.on_terminal()
            except Exception:
                log.exception("on_terminal failed")

    def _draw_term(self):
        """Terminal glyph inside the mic's ring treatment: a chamfered
        18x14 rectangle outline, a '>' chevron and a 6-px underscore —
        a prompt. Glyph CYAN idle, FOCAL while a task runs, FAINT off.
        The ring is GAPPED while nothing is attached and CLOSED once a
        terminal is watching (open / working / waiting), so the button
        says at a glance whether the pop-out is already up."""
        c = self.term
        c.delete("all")
        p = px
        state = self._term_state
        if state in ("disabled", "no_project"):
            ring = glyph = theme.FAINT
        elif state == "waiting":
            ring, glyph = theme.WARN, theme.FOCAL
        elif state == "working":
            ring, glyph = theme.CYAN, theme.FOCAL
        else:
            ring, glyph = theme.CYAN, theme.CYAN
        closed = state in ("open", "working", "waiting")
        box = (p(2), p(2), p(42), p(42))
        holo = theme.LOOK == "holo"
        for k in range(4):
            if closed:
                start, extent = 90 * k - 45, 90
            else:
                start, extent = 90 * k - 33, 66
            c.create_arc(*box, start=start, extent=extent, style="arc",
                         outline=ring, width=max(1, p(1)) if holo else p(2))
        if not holo:
            c.create_oval(p(7), p(7), p(37), p(37), fill=theme.RAISED,
                          outline="")
        # chamfered 18x14 screen outline (cut 2 on every corner)
        x0, y0, x1, y1, cut = 13, 15, 31, 29, 2
        pts = [x0 + cut, y0, x1 - cut, y0, x1, y0 + cut, x1, y1 - cut,
               x1 - cut, y1, x0 + cut, y1, x0, y1 - cut, x0, y0 + cut,
               x0 + cut, y0]
        c.create_line(*[p(v) for v in pts], fill=glyph, width=max(1, p(1.5)),
                      joinstyle="miter")
        # '>' prompt chevron and a 6-px underscore cursor
        c.create_line(p(16.5), p(19), p(19.5), p(22), p(16.5), p(25),
                      fill=glyph, width=p(2), joinstyle="miter",
                      capstyle="projecting")
        c.create_line(p(21), p(25), p(27), p(25), fill=glyph, width=p(2))


# ---------------------------------------------------------- SettingsDrawer
class SettingsDrawer(tk.Frame):
    """320px slide-over from the right (RAISED, scrollable). Groups per the
    V3 spec. Every control binds CONFIG via bind_config(); changes persist
    through CONFIG.update() with a 300ms debounce."""

    WIDTH = 320                       # design units; instances scale by S
    STEPS, STEP_MS = 6, 25

    def __init__(self, host, services=None,
                 on_config_change: Optional[Callable] = None,
                 toast: Optional[Toast] = None):
        self.WIDTH = px(type(self).WIDTH)
        super().__init__(host, bg=theme.RAISED, width=self.WIDTH)
        self.host = host
        self.services = services
        self.on_config_change = on_config_change
        self.toast = toast
        self._open = False
        self._anim = None
        self._x = self.WIDTH          # offset past the right edge
        self._save_jobs: dict = {}

        # header
        head = tk.Frame(self, bg=theme.RAISED)
        head.pack(fill="x", padx=theme.PAD, pady=(theme.PAD, theme.PAD_S))
        tk.Label(head, text="SETTINGS",
                 font=ui_display(theme.SIZE_BODY, "semibold"),
                 fg=theme.INK, bg=theme.RAISED).pack(side="left")
        RoundButton(head, text="✕", kind="ghost", command=self.close,
                    pad_x=8, pad_y=4, bg=theme.RAISED).pack(side="right")

        # scrollable body
        self._canvas = tk.Canvas(self, bg=theme.RAISED, highlightthickness=0,
                                 bd=0, width=self.WIDTH)
        self._canvas.pack(fill="both", expand=True)
        self._inner = tk.Frame(self._canvas, bg=theme.RAISED)
        self._inner_win = self._canvas.create_window(
            0, 0, anchor="nw", window=self._inner, width=self.WIDTH)
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all") or (0, 0, 0, 0)),
            add=True)
        self._canvas.bind("<Enter>", self._grab_wheel, add=True)
        self._canvas.bind("<Leave>", self._drop_wheel, add=True)
        self.bind("<Escape>", lambda e: self.close(), add=True)

        self._build_sections()
        if theme.LOOK == "holo":
            # a 1px GLASS_EDGE rule on the drawer's leading edge with a
            # short bright bracket at the top: the panel slides in as a
            # framed pane, not an unbounded RAISED slab (holo only)
            edge = tk.Frame(self, bg=theme.GLASS_EDGE, width=max(1, px(1)))
            edge.place(x=0, y=0, relheight=1.0)
            tk.Frame(self, bg=theme.BRIGHT, width=max(1, px(2)),
                     height=px(28)).place(x=0, y=0)
            tk.Frame(self, bg=theme.BRIGHT, width=px(18),
                     height=max(1, px(2))).place(x=0, y=0)

    # -------------------------------------------------------- open/close
    def toggle(self):
        self.close() if self._open else self.open()

    def open(self):
        # Privacy first: it is the ONE section whose controls change while
        # the drawer is shut, because offline mode's primary path is voice.
        self._refresh_privacy()
        self._open = True
        self.place(in_=self.host, relx=1.0, y=0, x=self._x,
                   anchor="ne", relheight=1.0, width=self.WIDTH)
        self.lift()
        self.focus_set()
        self._slide(0)

    def close(self):
        if not self._open:
            return
        self._open = False
        self._slide(self.WIDTH, then_forget=True)

    def _slide(self, target, then_forget=False):
        if self._anim:
            try:
                self.after_cancel(self._anim)
            except Exception:
                pass
            self._anim = None
        step = (target - self._x) / self.STEPS

        def tick(n=self.STEPS):
            self._x = target if n <= 1 else self._x + step
            self.place_configure(x=int(self._x))
            if n > 1:
                self._anim = self.after(self.STEP_MS, tick, n - 1)
            else:
                self._anim = None
                if then_forget:
                    self.place_forget()
        tick()

    def _grab_wheel(self, _e):
        self._canvas.bind_all("<Button-4>",
                              lambda e: self._canvas.yview_scroll(-2, "units"))
        self._canvas.bind_all("<Button-5>",
                              lambda e: self._canvas.yview_scroll(2, "units"))

    def _drop_wheel(self, _e):
        for seq in ("<Button-4>", "<Button-5>"):
            self._canvas.unbind_all(seq)

    # ---------------------------------------------------- config binding
    def bind_config(self, var_name: str, widget,
                    on_change: Optional[Callable] = None):
        """Bind a widget to CONFIG.<var_name>: initialize from CONFIG and
        persist edits back via CONFIG.update() (debounced 300ms).

        Supports Toggle, tk.Scale, tk.StringVar (pickers) and tk.Entry."""
        current = getattr(CONFIG, var_name)
        caster = type(current)

        def save(value):
            try:
                value = caster(value)
            except (TypeError, ValueError):
                log.warning("bind_config: bad value %r for %s", value, var_name)
                return
            job = self._save_jobs.pop(var_name, None)
            if job:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
            self._save_jobs[var_name] = self.after(
                300, lambda: self._commit(var_name, value, on_change))

        if isinstance(widget, Toggle):
            widget.set(bool(current), animate=False)
            widget.command = save
        elif isinstance(widget, tk.Scale):
            widget.set(current)
            widget.configure(command=lambda v: save(v))
        elif isinstance(widget, tk.StringVar):
            widget.set(str(current))
            widget.trace_add("write", lambda *_: save(widget.get()))
        elif isinstance(widget, tk.Entry):
            widget.delete(0, "end")
            widget.insert(0, str(current))
            widget.bind("<FocusOut>", lambda e: save(widget.get()), add=True)
            widget.bind("<Return>", lambda e: save(widget.get()), add=True)
        else:
            raise TypeError(f"bind_config: unsupported widget {widget!r}")
        return widget

    def _commit(self, var_name, value, on_change):
        self._save_jobs.pop(var_name, None)
        CONFIG.update(**{var_name: value})
        log.info("config %s = %r", var_name, value)
        if on_change:
            try:
                on_change(value)
            except Exception:
                log.exception("on_change for %s failed", var_name)
        if self.on_config_change:
            try:
                self.on_config_change(var_name, value)
            except Exception:
                log.exception("on_config_change failed")

    # -------------------------------------------------------- section UI
    def _section(self, title: str) -> tk.Frame:
        # a section title names a SURFACE, so in holo it is tracked like
        # the board's panel titles (theme.caption; the 09-03 U16 rule)
        tk.Label(self._inner, text=theme.caption(title, surface=True),
                 font=ui_display(theme.SIZE_CAPTION, "semibold"),
                 fg=theme.CYAN_DIM,
                 bg=theme.RAISED, anchor="w").pack(
            fill="x", padx=theme.PAD, pady=(theme.PAD_L, px(4)))
        tk.Frame(self._inner, bg=theme.LINE, height=max(1, px(1))).pack(
            fill="x", padx=theme.PAD)
        box = tk.Frame(self._inner, bg=theme.RAISED)
        box.pack(fill="x", padx=theme.PAD, pady=(theme.PAD_S, 0))
        return box

    def _row(self, box, label: str) -> tk.Frame:
        row = tk.Frame(box, bg=theme.RAISED)
        row.pack(fill="x", pady=px(4))
        tk.Label(row, text=label, font=ui_font(theme.SIZE_LABEL),
                 fg=theme.MUTED, bg=theme.RAISED, anchor="w").pack(side="left")
        return row

    def _toggle_row(self, box, label, var_name, on_change=None):
        row = self._row(box, label)
        tog = Toggle(row, bg=theme.RAISED)
        tog.pack(side="right")
        self.bind_config(var_name, tog, on_change)
        return tog

    def _picker_row(self, box, label, var_name, options, on_change=None):
        row = self._row(box, label)
        var = tk.StringVar()
        menu = self._option_menu(row, var, options)
        menu.pack(side="right")
        self.bind_config(var_name, var, on_change)
        return var

    def _option_menu(self, row, var, options) -> tk.OptionMenu:
        """A themed OptionMenu. Holo (2026-09-03, ui-polish U12): the
        Motif indicator -- a tiny '=' at 2x -- is switched off and a
        drawn chevron sits at the menu's right instead (a 1px-stroke
        glyph, like every other holo mark; a font chevron would render
        from whichever fallback face has the code point)."""
        menu = tk.OptionMenu(row, var, *options)
        menu.configure(bg=theme.RAISED, fg=theme.INK,
                       activebackground=theme.CYAN_SOFT,
                       activeforeground=theme.CYAN, bd=0,
                       highlightthickness=1,
                       highlightbackground=theme.RAISED,
                       highlightcolor=theme.CYAN_DIM, relief="flat",
                       font=ui_font(theme.SIZE_LABEL))
        menu["menu"].configure(bg=theme.RAISED, fg=theme.INK,
                               activebackground=theme.CYAN_SOFT,
                               activeforeground=theme.CYAN, bd=0,
                               font=ui_font(theme.SIZE_LABEL))
        if theme.LOOK == "holo":
            menu.configure(indicatoron=False, padx=px(6))
            self._chevron(row).pack(side="right", padx=(0, px(2)))
        return menu

    @staticmethod
    def _chevron(row) -> tk.Canvas:
        """The drawn 'open me' chevron for a holo picker: two 1px CYAN_DIM
        strokes in a px(12) box, pointing down."""
        s = px(12)
        c = tk.Canvas(row, width=s, height=s, bg=theme.RAISED,
                      highlightthickness=0, bd=0)
        c.create_line(px(2), px(4), s // 2, px(8), s - px(2), px(4),
                      fill=theme.CYAN_DIM, width=max(1, px(1)),
                      joinstyle="miter")
        return c

    # Holo scale rows (U12): the Scale's value is drawn centred over the
    # knob, and at the low end of the range the knob is at the trough's
    # left edge, so the value overhangs the widget's left edge and lands
    # against the label ('Silence timeout (s)2.5', measured on the 09-03
    # shot 14). Measured at S=2 the label is 314 px and the drawer's inner
    # width 576, so the px(130) trough left 2 px for a gap: the trough
    # shortens to SCALE_LEN_HOLO and SCALE_GAP_HOLO keeps the value off
    # the label at every knob position (a 36-px value overhangs ~18).
    SCALE_LEN_HOLO = 112
    SCALE_GAP_HOLO = 12

    def _scale_row(self, box, label, var_name, lo, hi, res, on_change=None):
        row = self._row(box, label)
        holo = theme.LOOK == "holo"
        scale = tk.Scale(row, from_=lo, to=hi, resolution=res,
                         orient="horizontal",
                         length=px(self.SCALE_LEN_HOLO if holo else 130),
                         bg=theme.RAISED, fg=theme.MUTED,
                         troughcolor=theme.LINE, highlightthickness=0, bd=0,
                         activebackground=theme.CYAN,
                         font=ui_font(theme.SIZE_CAPTION),
                         showvalue=True)
        scale.pack(side="right", padx=(px(self.SCALE_GAP_HOLO) if holo else 0, 0))
        self.bind_config(var_name, scale, on_change)
        return scale

    def _button_row(self, box, label, command):
        btn = RoundButton(box, text=label, kind="default", bg=theme.RAISED,
                          command=command)
        btn.pack(anchor="w", pady=px(4))
        return btn

    def _info_row(self, box, text: str):
        """Read-only caption line (e.g. where the assistant config lives)."""
        lbl = tk.Label(box, text=text, font=ui_font(theme.SIZE_CAPTION),
                       fg=theme.FAINT, bg=theme.RAISED, anchor="w",
                       justify="left", wraplength=self.WIDTH - 2 * theme.PAD)
        lbl.pack(fill="x", pady=px(4))
        return lbl

    # ------------------------------------------- assistant.json options
    def _get_option(self, key: str, default=False):
        """services.get_option(key) — the assistant config (dotted key);
        unwired → default."""
        fn = getattr(self.services, "get_option", None) if self.services \
            else None
        if fn is None:
            return default
        try:
            value = fn(key)
        except Exception:
            log.exception("get_option %s failed", key)
            return default
        return default if value is None else value

    def _set_option(self, key: str, value):
        fn = getattr(self.services, "set_option", None) if self.services \
            else None
        if fn is None:
            if self.toast:
                self.toast.show("Assistant settings not wired", kind="warn")
            log.warning("set_option not wired (%s)", key)
            return

        def run():
            try:
                fn(key, value)
                log.info("assistant option %s = %r", key, value)
            except Exception:
                log.exception("set_option %s failed", key)
        threading.Thread(target=run, daemon=True, name="set-option").start()

    def _option_toggle_row(self, box, label: str, key: str,
                           echo: bool = False):
        """An assistant.json toggle. ``echo`` also reports the new value
        through ``on_config_change``, which is the console's "a setting
        changed, react now" seam.

        Off by default because most of these only matter at the next start
        (autostart, the morning briefing). It is on for the camera preview
        because that switch governs a LENS: a privacy control that appears
        to do nothing until a restart is one he would reasonably conclude
        was broken, and then work around.
        """
        row = self._row(box, label)
        tog = Toggle(row, bg=theme.RAISED)
        tog.pack(side="right")
        tog.set(bool(self._get_option(key, False)), animate=False)

        def changed(value, k=key, tell=echo):
            self._set_option(k, bool(value))
            if tell:
                self._echo(k, bool(value))
        tog.command = changed
        return tog

    def _echo(self, name: str, value):
        """Tell the window a setting changed. Same callback the CONFIG
        rows use (bind_config), so there is one path into the console."""
        if not self.on_config_change:
            return
        try:
            self.on_config_change(name, value)
        except Exception:
            log.exception("on_config_change failed")

    def _service(self, name):
        fn = getattr(self.services, name, None) if self.services else None

        def run():
            if fn is None:
                if self.toast:
                    self.toast.show(f"{name} not wired", kind="warn")
                log.warning("service %s not wired", name)
                return
            threading.Thread(target=self._run_service, args=(name, fn),
                             daemon=True).start()
        return run

    def _run_service(self, name, fn):
        try:
            fn()
        except Exception:
            log.exception("service %s failed", name)

    # ---------------------------------------------------------- sections
    def _languages(self):
        try:
            from jarvis.transcriber import LANGUAGES  # peer module (lazy)
            return list(LANGUAGES)
        except Exception:
            return ["English", "Spanish", "French", "German", "Italian",
                    "Portuguese", "Dutch", "Russian", "Chinese", "Japanese",
                    "Korean", "Arabic", "Hindi"]

    def _build_sections(self):
        # Audio
        box = self._section("Audio")
        mics = ["Default"] + list(MACHINE.mic_names)
        self._picker_row(box, "Microphone", "mic", mics)
        self._toggle_row(box, "Noise gate", "noise_gate")
        self._scale_row(box, "Noise threshold", "noise_threshold",
                        0.005, 0.05, 0.001)
        self._button_row(box, "Calibrate noise", self._service("calibrate_noise"))
        self._scale_row(box, "Silence timeout (s)", "silence_timeout",
                        2.0, 20.0, 0.5)
        self._toggle_row(box, "Sounds", "sound")

        # Recognition
        box = self._section("Recognition")
        self._picker_row(box, "Model size", "model",
                         ["tiny", "base", "small", "medium", "large"])
        self._picker_row(box, "Language", "language", self._languages())
        self._button_row(box, "Edit vocabulary…", self._open_vocab_editor)
        self._toggle_row(box, "Streaming preview", "streaming")
        self._toggle_row(box, "Confidence review", "review")

        # Voice ID
        box = self._section("Voice ID")
        self._toggle_row(box, "Speaker verify", "speaker_verify")
        self._scale_row(box, "Threshold", "speaker_threshold", 0.1, 0.9, 0.05)
        self._button_row(box, "Enroll my voice", self._service("enroll_speaker"))
        self._button_row(box, "Train wake word", self._service("train_wakeword"))

        # Speech
        box = self._section("Speech")
        self._toggle_row(box, "Talkback", "talkback")
        self._picker_row(box, "Engine", "tts_engine", ["edge", "xtts"])

        # Intelligence
        box = self._section("Intelligence")
        self._toggle_row(box, "Jarvis mode", "jarvis_mode")
        self._toggle_row(box, "Auto-type", "auto_type")
        self._toggle_row(box, "Smart target picker", "smart_target")
        self._toggle_row(box, "Live write", "live_write")

        # Assistant (assistant.json options via services.get/set_option)
        box = self._section("Assistant")
        self._option_toggle_row(box, "Morning briefing", "briefing.enabled")
        self._info_row(box, "Config: ~/.config/jarvis/assistant.json")
        self._button_row(box, "Open Claude's terminal",
                         self._service("open_terminal"))
        self._option_toggle_row(box, "Start at login", "autostart.enabled")

        # Privacy (jarvis/sensing.py). His ruling of 2026-09-02 was that the
        # curfew window must be changeable "with voice command or UI
        # settings buttons" -- this is the second half of that. The offline
        # switch is here too because a control he can only reach by SAYING
        # it is one bad transcript away from being unreachable.
        box = self._section("Privacy")
        self._sensing_toggle = self._sensing_toggle_row(
            box, "Offline (camera + radar off)")
        self._curfew_start = self._curfew_row(box, "Camera curfew from", 0)
        self._curfew_end = self._curfew_row(box, "…until", 1)
        # The camera preview (jarvis/campreview.py). In Privacy rather than
        # in Assistant because what it switches on is a LENS, and the two
        # rows above are the other two answers to "is the camera running" --
        # a preview toggle three sections away from the offline switch would
        # be a camera control he has to remember lives somewhere else.
        # OFF by default: a pane that turned itself on would be the feature
        # introducing itself by breaking the rule it lives under. It also
        # obeys the rows above -- with offline mode or the curfew in force
        # the pane says which, and no device is opened.
        self._preview_toggle = self._option_toggle_row(
            box, "Camera preview", CAMERA_PREVIEW_OPTION, echo=True)
        # Grab and throw (jarvis/gesturecast.py). Beside the preview
        # because it RIDES the preview: the hand stage runs inside the
        # pane's own capture, so with the row above off there is no frame
        # for it to read and nothing here opens one. No echo: the stage
        # reads the switch on every frame, so it takes effect at once.
        self._gesture_toggle = self._option_toggle_row(
            box, "Grab and throw", GESTURE_OPTION)
        self._info_row(box, "Reach at the lens, close your hand, fling it "
                            "left or right. Needs the camera preview on.")
        self._info_row(box, "The microphone stays on while offline — say "
                            "“come back online” to switch sensing back on.")

        # System
        box = self._section("System")
        self._toggle_row(box, "Wake word", "hotword", self._hotword_changed)
        self._toggle_row(box, "Voice commands", "voice_cmds")
        self._toggle_row(box, "Auto-enter", "auto_enter")
        self._toggle_row(box, "Continuous listen", "continuous")
        tk.Frame(self._inner, bg=theme.RAISED, height=theme.PAD_L).pack()

    # ------------------------------------------------- offline mode rows
    def _sensing(self):
        return getattr(self.services, "sensing", None) if self.services else None

    def _sensing_toggle_row(self, box, label: str):
        row = self._row(box, label)
        tog = Toggle(row, bg=theme.RAISED)
        tog.pack(side="right")
        pol = self._sensing()
        try:
            tog.set(bool(pol is not None and pol.state().offline), animate=False)
        except Exception:  # noqa: BLE001 - provider boundary
            log.exception("sensing state read failed")
        tog.command = self._sensing_toggled
        return tog

    def _sensing_toggled(self, offline):
        """The switch, from the drawer. Same call the voice family makes,
        so there is exactly one code path that can take a sensor down."""
        pol = self._sensing()
        if pol is None:
            if self.toast:
                self.toast.show("Offline mode not wired", kind="warn")
            return
        try:
            out = pol.disable(source="settings") if offline \
                else pol.enable(source="settings")
        except Exception:  # noqa: BLE001 - provider boundary
            log.exception("sensing switch failed")
            if self.toast:
                self.toast.show("Sensing switch failed", kind="error")
            return
        # The toast SAYS what happened, like the spoken line: a device that
        # refused to stop is the one thing he must not have to guess at.
        partial = getattr(out, "partial", ())
        if out.failed:
            if self.toast:
                self.toast.show("Could not stop: %s" % ", ".join(out.failed),
                                kind="error")
        elif partial and self.toast:
            # Same honesty as the spoken line: with no power switch wired
            # the radar is still powered and still sensing the room, and
            # "Sensing offline" would be the promise, not the report.
            self.toast.show("Stopped reading %s — still powered"
                            % ", ".join(partial), kind="warn")
        elif not out.persisted and self.toast:
            self.toast.show("Saved in memory only — it won't survive a restart",
                            kind="warn")
        elif self.toast:
            self.toast.show("Sensing offline" if offline else "Sensing on",
                            kind="ok")

    def _curfew_row(self, box, label: str, index: int):
        """One end of the camera curfew, as a picker of half-hours.

        A picker, not a text field: the value has to parse as HH:MM and a
        free-text box in a slide-over is how "9pm" ends up in a privacy
        schedule that then silently reads as no curfew at all.
        """
        from jarvis.sensing import CURFEW_END_CHOICES, CURFEW_START_CHOICES
        options = CURFEW_START_CHOICES if index == 0 else CURFEW_END_CHOICES
        row = self._row(box, label)
        var = tk.StringVar()
        pol = self._sensing()
        current = None
        if pol is not None:
            try:
                current = pol.curfew()
            except Exception:  # noqa: BLE001 - provider boundary
                log.exception("curfew read failed")
        var.set(curfew_choice_values(current)[index])
        menu = self._option_menu(row, var, options)
        menu.pack(side="right")
        var.trace_add("write", lambda *_a: self._curfew_changed())
        return var

    def _refresh_privacy(self):
        """Re-read the policy into the Privacy rows.

        Every other row in this drawer is only ever changed FROM the
        drawer, so building it once was enough. This one is not: he says
        "offline mode" and the switch flips with the drawer shut, and a
        toggle built at construction then shows the OPPOSITE of the truth
        the header badge is showing two inches away.
        """
        # BEFORE the policy guard: the preview switch is an assistant.json
        # option and does not need a sensing owner to be read, so a box
        # where the policy failed to construct must still show the truth
        # about the pane. animate=False also means Toggle.command is not
        # called, so opening the drawer cannot write the switch it read.
        tog = getattr(self, "_preview_toggle", None)
        if tog is not None:
            tog.set(bool(self._get_option(CAMERA_PREVIEW_OPTION, False)),
                    animate=False)
        tog = getattr(self, "_gesture_toggle", None)
        if tog is not None:
            tog.set(bool(self._get_option(GESTURE_OPTION, False)),
                    animate=False)
        pol = self._sensing()
        if pol is None:
            return
        try:
            offline, window = bool(pol.state().offline), pol.curfew()
        except Exception:  # noqa: BLE001 - provider boundary
            log.exception("sensing state read failed")
            return
        tog = getattr(self, "_sensing_toggle", None)
        if tog is not None:
            # animate=False also means Toggle.command is NOT called: this
            # is a read-back, and it must not write the switch it just read.
            tog.set(offline, animate=False)
        values = curfew_choice_values(window)
        self._curfew_quiet = True
        try:
            for var, value in ((getattr(self, "_curfew_start", None), values[0]),
                               (getattr(self, "_curfew_end", None), values[1])):
                if var is not None and var.get() != value:
                    var.set(value)     # fires the trace; _curfew_quiet eats it
        finally:
            self._curfew_quiet = False

    def _curfew_changed(self):
        from jarvis.sensing import parse_hhmm
        if getattr(self, "_curfew_quiet", False):
            return                     # a read-back, not his edit
        pol = self._sensing()
        start = parse_hhmm(getattr(self, "_curfew_start", None)
                           and self._curfew_start.get())
        end = parse_hhmm(getattr(self, "_curfew_end", None)
                         and self._curfew_end.get())
        if pol is None or start is None or end is None or start == end:
            return
        try:
            ok = pol.set_curfew(start, end)
        except Exception:  # noqa: BLE001 - provider boundary
            log.exception("curfew save failed")
            ok = False
        if self.toast:
            self.toast.show("Camera curfew %s–%s" % (self._curfew_start.get(),
                                                     self._curfew_end.get())
                            if ok else "Could not save the curfew",
                            kind="ok" if ok else "error")

    def _hotword_changed(self, enabled):
        fn = getattr(self.services, "toggle_hotword", None) if self.services \
            else None
        if fn:
            try:
                fn(bool(enabled))
            except Exception:
                log.exception("toggle_hotword failed")

    # ------------------------------------------------------ vocab editor
    def _open_vocab_editor(self):
        top = tk.Toplevel(self)
        top.title("Vocabulary")
        top.configure(bg=theme.BG)
        top.geometry(f"{px(360)}x{px(420)}")
        tk.Label(top, text="One word or phrase per line",
                 font=ui_font(theme.SIZE_CAPTION), fg=theme.FAINT,
                 bg=theme.BG, anchor="w").pack(fill="x", padx=theme.PAD,
                                              pady=(theme.PAD, px(4)))
        text = tk.Text(top, bg=theme.SURFACE, fg=theme.INK, bd=0,
                       insertbackground=theme.CYAN, relief="flat",
                       font=ui_mono(theme.SIZE_LABEL), padx=px(10),
                       pady=px(10),
                       highlightthickness=1,
                       highlightbackground=theme.LINE,
                       highlightcolor=theme.CYAN_DIM)
        text.pack(fill="both", expand=True, padx=theme.PAD)
        try:
            text.insert("1.0", PATHS.VOCAB_FILE.read_text())
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("vocab read failed")

        def save():
            words = "\n".join(
                line.strip() for line in text.get("1.0", "end").splitlines()
                if line.strip())
            try:
                PATHS.VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
                PATHS.VOCAB_FILE.write_text(words + ("\n" if words else ""))
                if self.toast:
                    self.toast.show("Vocabulary saved", kind="ok")
            except Exception:
                log.exception("vocab save failed")
                if self.toast:
                    self.toast.show("Vocabulary save failed", kind="error")
            top.destroy()

        bar = tk.Frame(top, bg=theme.BG)
        bar.pack(fill="x", padx=theme.PAD, pady=theme.PAD_S)
        RoundButton(bar, text="Save", kind="accent", command=save,
                    bg=theme.BG).pack(side="right")
        RoundButton(bar, text="Cancel", kind="ghost", command=top.destroy,
                    bg=theme.BG).pack(side="right", padx=(0, theme.PAD_S))


# -------------------------------------------------------------- StatusStrip
class StatusStrip(tk.Frame):
    """28px strip, segmented by hairlines: wake-word state (left,
    clickable), the PROJECT chip right after it (ONLY while a Claude
    project is active — `| PROJECT JARVIS |`; idle bar unchanged), an
    EMPTY centre (the strip has no centre text, ever), and the ONLY
    machine-telemetry site on the HUD — CPU / GPU / MEMORY segments on
    the right, each `LABEL value` (display SIZE_CAPTION semibold MUTED /
    mono SIZE_CAPTION FOCAL, '--' when unknown).

    Fitting (plan_strip): the PROJECT value ellipsizes first (never below
    6 characters); when the bar is still too narrow MEMORY yields, then
    the chip — nothing ever overlaps or clips.

    Everything is drawn as canvas items directly on the bar's gradient
    ground (no child Labels with solid backgrounds, so no slab can ever
    show behind a segment); the ground itself is flat from 30% of the
    width onward, under the telemetry cluster. Built right→left so the
    values never jump."""

    SEGMENTS = ("CPU", "GPU", "MEMORY")

    def __init__(self, parent, on_hotword_click: Optional[Callable] = None,
                 **kw):
        super().__init__(parent, bg=theme.SURFACE, height=px(28), **kw)
        self.pack_propagate(False)
        self.on_hotword_click = on_hotword_click
        self._lf = ui_display(theme.SIZE_CAPTION, "semibold")
        self._vf = ui_mono(theme.SIZE_CAPTION)
        self._hot_text = ""
        self._values = {name: "--" for name in self.SEGMENTS}
        self._temps_text = ""
        self._mem_text = ""
        self._level = 0            # telemetry elision level (plan_telemetry)
        self._project = ""
        self._project_text = ""

        # atmosphere: subtle gradient ground, lit at the hotword side and
        # FLAT (exactly SURFACE) from 30% on — the telemetry cluster sits
        # on a flat ground; the canvas is the only child, so nothing can
        # punch a differently-toned rectangle into it
        self._grad = BarGradient(self, theme.SURFACE,
                                 [(0.0, 0.055), (0.30, 0.0), (1.0, 0.0)])
        c = self.canvas = self._grad.canvas
        # holo: the wake-word state is a small ring (lit dot inside while
        # listening) beside a caps label, instead of the ●/○ glyph in the
        # text; the ring shares the "hot" tag so it toggles too
        self._hot_ring = None
        if theme.LOOK == "holo":
            self._hot_ring = (c.create_oval(0, 0, 0, 0, outline=theme.FAINT,
                                            fill="", width=max(1, px(1)),
                                            tags=("hot",)),
                              c.create_oval(0, 0, 0, 0, outline="", fill="",
                                            tags=("hot",)))
        self._hot = c.create_text(theme.PAD, 0, anchor="w", text="",
                                  font=self._lf, fill=theme.FAINT,
                                  tags=("hot",))
        c.tag_bind("hot", "<Button-1>", self._hot_clicked, add=True)
        c.tag_bind("hot", "<Enter>",
                   lambda e: c.configure(cursor="hand2"), add=True)
        c.tag_bind("hot", "<Leave>",
                   lambda e: c.configure(cursor=""), add=True)
        Tooltip(c, "Click to toggle the wake word listener", tag="hot")
        # separators: LINE hairlines in classic; holo draws them a shade
        # brighter (GLASS_EDGE) and shorter — the ref HUD's thin ticks
        sep = theme.GLASS_EDGE if theme.LOOK == "holo" else theme.LINE
        self._hot_line = c.create_line(0, 0, 0, 0, fill=sep)
        # PROJECT chip items (hidden until set_project)
        self._proj_lbl = c.create_text(0, 0, anchor="w", text="PROJECT",
                                       font=self._lf, fill=theme.MUTED,
                                       state="hidden")
        self._proj_val = c.create_text(0, 0, anchor="w", text="",
                                       font=self._vf, fill=theme.FOCAL,
                                       state="hidden")
        self._proj_line = c.create_line(0, 0, 0, 0, fill=sep,
                                        state="hidden")
        self._segs = {}
        for name in self.SEGMENTS:
            lid = c.create_text(0, 0, anchor="e", text=name, font=self._lf,
                                fill=theme.MUTED)
            vid = c.create_text(0, 0, anchor="e", text="--", font=self._vf,
                                fill=theme.FOCAL)
            hid = c.create_line(0, 0, 0, 0, fill=sep)
            self._segs[name] = (lid, vid, hid)
        c.bind("<Configure>", lambda e: self._layout(), add=True)

        self.set_hotword(bool(CONFIG.hotword))

    def _measure(self, font_spec, text: str) -> int:
        try:
            return measure(font_spec, text)
        except Exception:
            return px(7) * len(text)

    def _layout(self):
        """Place every item from the measured text widths:
        `[PAD] ○ WAKE WORD OFF [PAD_S] | [PAD_S] PROJECT [6] JARVIS [PAD_S]
        |` … `| [PAD_S] LABEL [6] value [PAD_S] | … [PAD_S] [PAD] grip`.
        The PROJECT segment and any yielding telemetry segment follow the
        pure plan_strip() decision."""
        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 4 or h < 4:
            return
        cy = h // 2
        holo = theme.LOOK == "holo"
        y0, y1 = (px(8), h - px(8)) if holo else (px(6), h - px(6))
        gap = px(6)
        hx = theme.PAD
        if self._hot_ring is not None:
            d = px(8)
            ring, dot = self._hot_ring
            c.coords(ring, hx, cy - d // 2, hx + d, cy + d // 2)
            c.coords(dot, hx + d // 4, cy - d // 4, hx + d - d // 4,
                     cy + d // 4)
            hx += d + gap
        c.coords(self._hot, hx, cy)
        x = hx + self._measure(self._lf, self._hot_text) + theme.PAD_S
        c.coords(self._hot_line, x, y0, x, y1)

        # telemetry per elision level (pure: telemetry_segments); the
        # widest level that leaves the left side clear wins
        levels, texts = [], []
        for level in range(TELEMETRY_LEVELS):
            segs = telemetry_segments(self._temps_text, self._mem_text, level)
            seg_ws, seg_txt = [], {}
            for name, (label, value) in zip(self.SEGMENTS[-len(segs):], segs):
                sw = theme.PAD_S + self._measure(self._vf, value) + theme.PAD_S
                if label:
                    sw += self._measure(self._lf, label) + gap
                if name == self.SEGMENTS[-1]:
                    sw += theme.PAD                # right margin (grip)
                seg_ws.append((name, sw))
                seg_txt[name] = (label, value)
            levels.append(seg_ws)
            texts.append(seg_txt)
        proj_fixed = (theme.PAD_S + self._measure(self._lf, "PROJECT") + gap
                      + theme.PAD_S)
        char_w = max(1, self._measure(self._vf, "M"))
        level, chars, hidden = plan_telemetry(w, x, proj_fixed, char_w,
                                              len(self._project),
                                              levels_for_look(levels))
        self._level = level
        seg_txt = texts[level]
        hidden = list(hidden) + [n for n in self.SEGMENTS if n not in seg_txt]
        self._project_text = fmt_project_chip(self._project, chars)
        if self._project_text:
            lx = x + theme.PAD_S
            c.coords(self._proj_lbl, lx, cy)
            vx = lx + self._measure(self._lf, "PROJECT") + gap
            c.coords(self._proj_val, vx, cy)
            c.itemconfigure(self._proj_val, text=self._project_text)
            px_end = vx + self._measure(self._vf, self._project_text) + theme.PAD_S
            c.coords(self._proj_line, px_end, y0, px_end, y1)
            for item in (self._proj_lbl, self._proj_val, self._proj_line):
                c.itemconfigure(item, state="normal")
        else:
            for item in (self._proj_lbl, self._proj_val, self._proj_line):
                c.itemconfigure(item, state="hidden")

        xr = w - theme.PAD - theme.PAD_S           # last value's right edge
        for name in reversed(self.SEGMENTS):       # MEMORY, GPU, CPU
            lid, vid, hid = self._segs[name]
            shown = name not in hidden
            for item in (lid, vid, hid):
                c.itemconfigure(item, state="normal" if shown else "hidden")
            if not shown:
                continue
            label, value = seg_txt[name]
            c.itemconfigure(lid, text=label)
            c.itemconfigure(vid, text=value)
            c.coords(vid, xr, cy)
            lx = xr - self._measure(self._vf, value) - (gap if label else 0)
            c.coords(lid, lx, cy)
            hx = lx - self._measure(self._lf, label) - theme.PAD_S
            c.coords(hid, hx, y0, hx, y1)
            xr = hx - theme.PAD_S

    def _set_value(self, name: str, text: str):
        text = text or "--"
        if self._values[name] == text:
            return
        self._values[name] = text
        self.canvas.itemconfigure(self._segs[name][1], text=text)
        self._layout()

    @property
    def telemetry_level(self) -> int:
        """What the last layout chose: 0 full, 1 compact ('CPU 53° 7% ·
        GPU 44° 0% · 47.5/122 GB'), 2 load ('CPU 7% · GPU 0% ·
        47.5/122 GB'), 3 minimal ('47.5/122 GB')."""
        return self._level

    def _hot_clicked(self, _e):
        if self.on_hotword_click:
            try:
                self.on_hotword_click()
            except Exception:
                log.exception("hotword click failed")

    def set_hotword(self, on: bool):
        color = theme.CYAN_DIM if on else theme.FAINT
        if self._hot_ring is not None:
            self._hot_text = "WAKE WORD ON" if on else "WAKE WORD OFF"
            ring, dot = self._hot_ring
            self.canvas.itemconfigure(ring, outline=color)
            self.canvas.itemconfigure(dot, fill=theme.CYAN if on else "")
        else:
            self._hot_text = "● WAKE WORD ON" if on else "○ WAKE WORD OFF"
        self.canvas.itemconfigure(self._hot, text=self._hot_text, fill=color)
        self._layout()

    def set_temps(self, text: str):
        """Parse the temps line ('cpu 45° 12% · gpu 38° 5%') into the CPU
        and GPU values ('45°C · 12%' / '38°C · 5%'; '--' when absent)."""
        self._temps_text = text or ""
        segs = split_temps(text)
        self._set_value("CPU", fmt_temps(segs.get("cpu", "")))
        self._set_value("GPU", fmt_temps(segs.get("gpu", "")))

    def set_memory(self, text: str):
        """Used/total system RAM, e.g. '26.8/128 GB' ('--' when unknown)."""
        self._mem_text = text or ""
        self._set_value("MEMORY", text or "--")

    def set_project(self, slug: str):
        """Active Claude project slug ('' hides the PROJECT segment)."""
        slug = (slug or "").strip()
        if slug == self._project:
            return
        self._project = slug
        self._layout()

    @property
    def project_text(self) -> str:
        """What the PROJECT chip currently shows ('' when hidden)."""
        return self._project_text
