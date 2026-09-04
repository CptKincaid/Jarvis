"""The room slab — what the console shows when nobody is talking to it.

One widget serves two of the console's three modes (jarvis/ui/console_mode):

  ambient   a quiet band of room state over the receded transcript: what is
            playing, the next commitment, the next deadline, the outside
            temperature, the hour of the house
  standby   the same facts under a large soft clock on a near-black ground —
            the room's face at two in the morning

Building them as one surface rather than two overlays is the whole point:
they show the SAME room, at two sizes and two brightnesses, and a second
widget would have meant a second set of refresh rules to keep in step.

The room dict is produced APP-side, off the Tk thread, and handed over as
pre-computed strings — the same contract Reactor.set_telemetry documents
("cheap attribute reads"). Nothing in this module makes a network call, and
`spotify.now_playing()` in particular is a live REST call with no cache
behind it, so the provider gates and backs it off; the slab just draws what
it is given.

Two rules that came out of the review and are load-bearing:

* **Quiet hours are the PALETTE, not a row.** "quiet hours: on" is a switch
  readout; a slab that goes ember and low-contrast says the same thing
  without a label, and reads from across the room.

* **Presence is hidden unless it is configured.** `presence.is_home()`
  answers True when the sentinel is unconfigured (home is None -> not False
  -> True, jarvis/presence.py:159) and phone_ip is empty today, so a naive
  panel would print a confident, false HOME. The provider sends "" and the
  row disappears.

Two LOOKS (2026-09-01, the blue-holographic overhaul, W2 lane C):

* **classic** draws exactly what shipped on 08-31 — the grey clock over a
  centred "NEXT   VALUE" list. It is the fallback Hunter asked for, so the
  legacy branches below are left byte-for-byte and every new stroke is
  gated on `theme.LOOK == "holo"`.
* **holo** follows the Iron-Man HUD reference (scratchpad/holo/ref/ref2_hud
  .png): the clock in ice-white with a faint cyan glow line under it, the
  meridiem/date in small TRACKED caps, and the room rows as caption/value
  pairs inside a thin 1px chamfered frame with corner brackets — no filled
  slab anywhere, the chrome is the 1px strokes. Amber is still reserved for
  quiet hours (slab_tone), never decoration.

Everything here reads `theme.X` at CALL time. A module constant such as the
old `QUIET_INK = theme.WARN` froze the import-time look, so a runtime
select_look() left the slab half in the other palette.
"""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from typing import Optional

from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.console_mode import ACTIVE, AMBIENT, STANDBY, dim
from jarvis.ui.widgets import ellipsize, measure, px, ui_display, ui_mono

log = get_logger("ui.ambient")

CLOCK_SIZE = 64             # standby clock, in TYPE units (ui_display scales
                            # these itself -- passing px() here would scale
                            # them twice, which is the trap widgets.py's
                            # "fonts are scaled ONLY via set_font_scale"
                            # comment exists to prevent)
CLOCK_BOX = 86              # the clock's layout box in DESIGN units (px()),
                            # ~4/3 of the type size, matching _FONT_SCALE
AMBIENT_CLOCK = 28          # the ambient band's smaller clock, type units
ROW_H = 26
PAD = 20
TICK_MS = 1000              # the clock has a minute hand to keep honest
GPU_BAR_W = 120

# ---- holo chrome (design units; classic never reads these) --------------
FRAME_CUT = 7               # the 45° corner cut on the 1px frame
BRACKET_LEN = 16            # bright corner-bracket arm length
FRAME_INSET = 10            # the ambient band's frame, in from the canvas
BAND_INDENT = 10            # holo: the band's text steps in from PAD by this
                            # so it clears the frame (20 design units in from
                            # the frame's stroke, like the standby rows)
STANDBY_FRAME_FRAC = 0.64   # standby row frame: share of the slab width…
STANDBY_FRAME_MIN = 300     # …but never narrower than this
ROW_INSET_X = 18            # text inset from the frame's vertical edges
GLOW_MIN_W = 120            # the clock's glow line, at its shortest
RAIL_LIT_FRAC = 0.22        # the ambient header rail: lit share (ref2's
                            # segmented status bar — one bright segment,
                            # the rest dim)
MERIDIEM_SIZE = 22          # the inline AM/PM after the standby digits, in
                            # TYPE units (the wordmark's size: a third of
                            # the clock, read as part of the time)
MERIDIEM_GAP = 10           # design units between the digits and the AM/PM
GLOW_STEPS = (1.0, 0.7, 0.4)  # the clock glow's nested spans, longest and
                            # dimmest first, so the line fades at its ends

# Row order for the ambient band. NOW first: what is audible in the room is
# the one fact you check without meaning to.
AMBIENT_KEYS = (("playing", "NOW"), ("next", "NEXT"), ("due", "DUE"),
                ("temp", "OUTSIDE"), ("arc", "HOUR"), ("presence", "WHERE"))
# Standby drops NOW (nothing is playing at 3 a.m.) and the arc word (the
# clock already says what hour of the house it is). It KEEPS presence: the
# standby slab is what the room shows when nobody is at the desk, which is
# exactly when "is he home" is the question worth answering, and it was
# missing here while the ambient band had it (found 2026-08-31, from the
# console).
STANDBY_KEYS = (("next", "NEXT"), ("due", "DUE"), ("temp", "OUTSIDE"),
                ("presence", "WHERE"))


# ------------------------------------------------------------ pure rules
def _rows(room, keys) -> list:
    """(LABEL, VALUE) pairs, both upper case.

    The labels were always upper case and the values were not, so the slab
    read as a caption with a sentence after it. Upper case throughout is
    what makes it a PANEL -- readable across a room at a glance, which is
    the only way this surface is ever read. Cased here, in the one place
    every row passes through, and BEFORE the renderer ellipsizes: upper
    case is wider, so truncation has to measure the text that is actually
    drawn.
    """
    room = room or {}
    out = []
    for key, label in keys:
        value = str(room.get(key) or "").strip()
        if value:
            out.append((label, value.upper()))
    return out


def room_rows(room) -> list:
    """(label, value) rows for the ambient band. Quiet hours and GPU load
    are deliberately absent: the first is the slab's colour and the second
    is a bar, because both are switch-readouts as text."""
    return _rows(room, AMBIENT_KEYS)


def standby_rows(room) -> list:
    """The lines under the room clock: next commitment, next deadline,
    outside temperature, and where he is."""
    return _rows(room, STANDBY_KEYS)


def clock_text(now: Optional[datetime] = None) -> str:
    """'2:14' — twelve hour, no leading zero, no seconds. A seconds hand on
    a room clock is a thing that moves in the corner of your eye all night."""
    now = now or datetime.now()
    hour = now.hour % 12 or 12
    return f"{hour}:{now.minute:02d}"


def meridiem(now: Optional[datetime] = None) -> str:
    return "AM" if (now or datetime.now()).hour < 12 else "PM"


def date_text(now: Optional[datetime] = None) -> str:
    """'SATURDAY 30 AUGUST' — the day name first, because that is what you
    are actually checking when you glance at a clock at 2 a.m."""
    now = now or datetime.now()
    return f"{now.strftime('%A')} {now.day} {now.strftime('%B')}".upper()


def slab_tone(room) -> str:
    """'quiet' when he is holding his tongue, 'away' when the room is
    empty and known to be, else 'normal'. The renderer maps this to a
    palette — never to a label."""
    room = room or {}
    if str(room.get("quiet") or "").strip():
        return "quiet"
    if str(room.get("presence") or "").strip().lower() == "away":
        return "away"
    return "normal"


def gpu_fraction(room) -> Optional[float]:
    """GPU load 0..1, or None when it is not known. Expressed as a bar by
    the renderer: a number here would be a readout, not a glance."""
    try:
        value = (room or {}).get("gpu")
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def visible_rows(rows, height: int, row_h: int, top: int) -> list:
    """As many rows as the slab has room for below `top` (pure, so the
    renderer never draws off its own bottom edge)."""
    room = int(height) - int(top)
    # floor division leaves the last baseline a full row above the bottom
    # edge, which is where a descender would otherwise be clipped
    return list(rows or [])[:max(0, room // max(1, int(row_h)))]


# ------------------------------------------------------ palette (call time)
def quiet_ink() -> str:
    """The ember the whole slab takes in quiet hours. A function, not a
    constant: `QUIET_INK = theme.WARN` captured the import-time look."""
    return theme.WARN


def slab_ink(tone: str, look: Optional[str] = None) -> tuple:
    """(head, body, faint) for the slab's tone, UNDIMMED, read from theme at
    call time. Quiet hours take the whole palette ember; away drops the
    contrast; normal is the focal ladder.

    `look` defaults to the current theme.LOOK and is read for the tokens
    only: the ladder is the same in both looks. Until 09-03 the holo clock
    was CORE_BANDS[2] (#a8e9ff) for a cool cast under the night dim -- and
    measured (U09, 17-standby at S=2) it came out the THIRD brightest text
    on its own slab: peak luminance 159 against 173 for the row values and
    131 for its own caption. FOCAL is the top of the ladder; the cast is
    the price of the clock owning the panel."""
    look = look or theme.LOOK
    if tone == "quiet":
        ember = quiet_ink()
        return ember, dim(ember, 0.7), theme.FAINT
    if tone == "away":
        return theme.MUTED, theme.MUTED, theme.FAINT
    return theme.FOCAL, theme.INK, theme.MUTED


# --------------------------------------------- measured-width memo
_MEASURE_CACHE: dict = {}
_MEASURE_CACHE_MAX = 512     # the room's strings number in the tens; this
                             # is a ceiling against a runaway, not a budget


def measured(font_spec, text: str) -> int:
    """widgets.measure() behind a bounded memo. The slab redraws at 1 Hz
    and asks the same handful of questions every tick (the clock digits
    change once a minute, the labels never), yet each `font measure` is a
    Tcl round-trip that profiled at ~0.45 ms — the seven the holo standby
    needs were 3 ms of an 11 ms redraw (:98 at scale 2, 2026-09-01), and
    the reactor's standby slot is 33 ms with the whole UI inside it. Keyed
    by the exact spec, so a scale or family change is a miss rather than
    a stale width; past the cap the memo is cleared, not evicted. Raises
    exactly what measure() raises — callers keep their own fallback."""
    key = (font_spec, text)
    hit = _MEASURE_CACHE.get(key)
    if hit is None:
        if len(_MEASURE_CACHE) >= _MEASURE_CACHE_MAX:
            _MEASURE_CACHE.clear()
        hit = _MEASURE_CACHE[key] = int(measure(font_spec, text))
    return hit


def fitted(text: str, font_spec, max_px: int) -> str:
    """widgets.ellipsize() through the same memo — the same trailing
    ellipsis, the same 'return the text untouched when Tk cannot answer'."""
    try:
        if measured(font_spec, text) <= max_px:
            return text
        while text and measured(font_spec, text + "…") > max_px:
            text = text[:-1]
    except Exception:                       # noqa: BLE001 - Tk font boundary
        return text
    return text + "…"


# --------------------------------------------------- holo chrome (pure)
def tracked(text: str, gap: str = " ", word_gap: str = "   ") -> str:
    """'NEXT' -> 'N E X T': letter-spaced caps, the wordmark's trick. Tk has
    no tracking, and the display face has no thin-space glyph (checked
    2026-09-01: U+2009/U+200A absent from Chakra Petch), so the gap is a
    real space and words are held apart by a wider one. Whitespace in the
    input collapses first, so a pre-spaced value cannot double up."""
    words = str(text or "").split()
    return word_gap.join(gap.join(word) for word in words)


def standby_caption(now: Optional[datetime] = None,
                    look: Optional[str] = None) -> str:
    """The line under the standby clock (pure). Classic: 'PM  ·  SATURDAY
    30 AUGUST', as it has always been. Holo: the tracked date alone -- the
    meridiem sits inline after the digits (clock_line), so it is no longer
    a detached 'PM' a caption-size below the time (U09, 09-03)."""
    now = now or datetime.now()
    if (look or theme.LOOK) == "holo":
        return tracked(date_text(now))
    return f"{meridiem(now)}  ·  {date_text(now)}"


def clock_line(cx: int, digits_w: int, meridiem_w: int, gap: int) -> tuple:
    """(digits_x, meridiem_x) for a west-anchored '4:26' and its 'PM' laid
    as ONE centred group about `cx` (pure): the digits start half the
    group's width left of centre, the meridiem `gap` past their end."""
    total = int(digits_w) + int(gap) + int(meridiem_w)
    digits_x = int(cx) - total // 2
    return digits_x, digits_x + int(digits_w) + int(gap)


def meridiem_bottom(cy: int, clock_linespace: int, clock_ascent: int,
                    meridiem_descent: int) -> int:
    """y for a south-anchored meridiem whose BASELINE meets the baseline of
    digits centred on `cy` (pure). A centred text box of `clock_linespace`
    puts its baseline at cy - linespace/2 + ascent; the smaller face's box
    bottom is that baseline plus its own descent."""
    return int(cy) - int(clock_linespace) // 2 + int(clock_ascent) \
        + int(meridiem_descent)


def font_metrics(font_spec) -> tuple:
    """(ascent, descent, linespace) of `font_spec` in device px, via the raw
    `font metrics` call (the same reason widgets.measure avoids tkfont.Font:
    a pixel size wrapped in a named font reports rounded points)."""
    root = tk._get_default_root("font metrics")
    call = root.tk.call
    return tuple(int(call("font", "metrics", font_spec, opt))
                 for opt in ("-ascent", "-descent", "-linespace"))


def frame_box(w: int, top: int, n_rows: int, row_h: int, *, frac: float,
              min_w: int, side_pad: int, inset_y: int) -> Optional[tuple]:
    """(x0, y0, x1, y1) of the standby row frame, centred, wrapping `n_rows`
    whose text centres sit at top + i*row_h. None when there are no rows:
    an empty room draws nothing, not an empty box (the same rule the rows
    themselves follow).

    Width is `frac` of the slab but never under `min_w`, and never wider
    than the slab less `side_pad` a side. Vertically the frame reaches
    half a row past the first and last text centres plus `inset_y`, which
    keeps it inside the bottom edge visible_rows() already guaranteed."""
    n = max(0, int(n_rows))
    if n == 0:
        return None
    w = int(w)
    fw = max(int(min_w), int(w * float(frac)))
    fw = min(fw, max(1, w - 2 * int(side_pad)))
    x0 = (w - fw) // 2
    half = int(row_h) // 2
    y0 = int(top) - half - int(inset_y)
    y1 = int(top) + (n - 1) * int(row_h) + half + int(inset_y)
    return (x0, y0, x0 + fw, y1)


def hud_frame_points(x0, y0, x1, y1, cut) -> list:
    """Flat polygon points for the 1px HUD frame: a rectangle with all four
    corners cut at 45° by `cut` (clamped so a tiny box still closes). The
    reactor's engine card cuts two corners; the room frames cut four so
    the corner brackets below have a diagonal to follow at every corner."""
    cut = max(0, min(int(cut), (x1 - x0) // 2, (y1 - y0) // 2))
    return [x0 + cut, y0, x1 - cut, y0, x1, y0 + cut, x1, y1 - cut,
            x1 - cut, y1, x0 + cut, y1, x0, y1 - cut, x0, y0 + cut]


def hud_bracket_points(x0, y0, x1, y1, cut, length) -> list:
    """Four polylines (flat point lists), one per corner, each an arm of
    `length` along both edges joined by the corner's diagonal — the bright
    accents ref2_hud puts where its frame lines meet. Arms are clamped so
    two brackets on a short edge never cross into each other."""
    cut = max(0, min(int(cut), (x1 - x0) // 2, (y1 - y0) // 2))
    arm = max(0, min(int(length), (x1 - x0) // 2 - cut, (y1 - y0) // 2 - cut))
    return [
        [x0, y0 + cut + arm, x0, y0 + cut, x0 + cut, y0, x0 + cut + arm, y0],
        [x1 - cut - arm, y0, x1 - cut, y0, x1, y0 + cut, x1, y0 + cut + arm],
        [x1, y1 - cut - arm, x1, y1 - cut, x1 - cut, y1, x1 - cut - arm, y1],
        [x0 + cut + arm, y1, x0 + cut, y1, x0, y1 - cut, x0, y1 - cut - arm],
    ]


def separator_ys(top: int, n_rows: int, row_h: int) -> list:
    """The y of each thin rule BETWEEN rows whose centres sit at
    top + i*row_h — n-1 of them, none for a single row."""
    n = max(0, int(n_rows))
    half = int(row_h) // 2
    return [int(top) + i * int(row_h) + half for i in range(n - 1)]


def glow_span(cx: int, text_w: int, min_w: int, max_w: int) -> tuple:
    """(x0, x1) of the clock's under-glow, centred on `cx`: 90% of the
    digits' measured width, clamped to [min_w, max_w] so '1:11' still gets
    a line and a wide '12:38' cannot run past the slab."""
    lo, hi = int(min_w), max(int(min_w), int(max_w))
    span = max(lo, min(hi, int(int(text_w) * 0.9)))
    return (int(cx) - span // 2, int(cx) + span // 2)


def glow_steps(x0: int, x1: int, steps=GLOW_STEPS) -> list:
    """[(xa, xb), …] nested spans centred on the same point, one per step
    fraction — stacked brightest-last they read as a glow that fades toward
    its ends, which one flat line never does."""
    cx = (int(x0) + int(x1)) / 2.0
    full = int(x1) - int(x0)
    out = []
    for f in steps:
        half = max(1, int(full * max(0.0, min(1.0, float(f))) / 2))
        out.append((int(cx - half), int(cx + half)))
    return out


def band_frame(w: int, h: int, inset: int, rail_y: int, top: int,
               n_rows: int, row_h: int, pad_y: int) -> tuple:
    """(x0, y0, x1, y1) of the ambient band's frame: `inset` in from the
    canvas sides and top, and FITTED to the content — half a row past the
    last row centre plus `pad_y`, or just under the header rail when there
    are no rows — never the whole slab. A frame around the empty lower
    half of the transcript area read as a box around nothing (r1 shot,
    09-01). Clamped to the canvas so a short slab keeps a closed frame."""
    n = max(0, int(n_rows))
    if n:
        y1 = int(top) + (n - 1) * int(row_h) + int(row_h) // 2 + int(pad_y)
    else:
        y1 = int(rail_y) + int(pad_y)
    y1 = min(y1, int(h) - int(inset))
    return (int(inset), int(inset), int(w) - int(inset), y1)


def rail_segments(x0: int, x1: int, lit_frac: float) -> tuple:
    """((x0, xa), (xa, x1)) — the ambient header rail split into its lit
    lead and dim remainder; lit_frac is clamped to 0..1."""
    f = max(0.0, min(1.0, float(lit_frac)))
    xa = int(x0) + int((int(x1) - int(x0)) * f)
    return ((int(x0), xa), (xa, int(x1)))


# ------------------------------------------------------------- the widget
class RoomSlab(tk.Canvas):
    """The ambient / standby surface. Placed OVER the transcript and
    withdrawn (place_forget) the instant a turn starts — it never competes
    with a reply, and it never touches the reactor, whose baked frames must
    not be re-rendered at a mode change."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=theme.BG, highlightthickness=0, bd=0,
                         takefocus=0, **kw)
        self._room: dict = {}
        self._mode = ACTIVE
        self._dim = 1.0
        self._reveal = None            # power-up: rows shown so far
        self._job = None
        self._placed = False
        self.bind("<Configure>", lambda _e: self._redraw(), add=True)

    # -------------------------------------------------------------- data
    def set_room(self, room) -> None:
        self._room = dict(room or {})
        if self._placed:
            self._redraw()

    def set_dim(self, factor: float) -> None:
        self._dim = max(0.0, min(1.0, float(factor)))
        if self._placed:
            self._redraw()

    def set_reveal(self, count) -> None:
        """Power-up: draw only the first `count` rows (None = all)."""
        self._reveal = count
        if self._placed:
            self._redraw()

    # -------------------------------------------------------------- mode
    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        if mode == ACTIVE:
            self._unplace()
            return
        self._place()

    def _place(self) -> None:
        if not self._placed:
            self.place(relx=0, rely=0, relwidth=1, relheight=1)
            tk.Misc.lift(self)          # Canvas.lift is tag_raise; widget form
            self._placed = True
        self._redraw()
        self._tick()

    def _unplace(self) -> None:
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        if self._placed:
            try:
                self.place_forget()
            except tk.TclError:
                log.debug("room slab place_forget on a dead widget",
                          exc_info=True)
            self._placed = False

    def _tick(self) -> None:
        if not self._placed:
            return
        self._redraw()
        try:
            self._job = self.after(TICK_MS, self._tick)
        except tk.TclError:
            self._job = None

    # ------------------------------------------------------------ render
    def _ink(self, tone: str):
        """(head, body, faint) for the slab's tone, already dimmed."""
        head, body, faint = slab_ink(tone)
        f = self._dim
        return dim(head, f), dim(body, f), dim(faint, f)

    def _redraw(self) -> None:
        try:
            self.delete("all")
            w, h = self.winfo_width(), self.winfo_height()
        except tk.TclError:
            return
        if w <= 1 or h <= 1 or self._mode == ACTIVE:
            return
        ground = dim(theme.TV_BG, self._dim * 0.9)
        self.configure(bg=ground)
        head, body, faint = self._ink(slab_tone(self._room))
        if self._mode == STANDBY:
            self._draw_standby(w, h, head, body, faint)
        else:
            self._draw_ambient(w, h, head, body, faint)

    def _draw_standby(self, w, h, head, body, faint) -> None:
        now = datetime.now()
        cy = int(h * 0.36)
        holo = theme.LOOK == "holo"
        clock_font = ui_display(CLOCK_SIZE, "semibold")
        clock = clock_text(now)
        if holo:
            self._draw_clock_line(w, cy, clock, clock_font, now, head, body)
        else:
            self.create_text(w // 2, cy, anchor="center", text=clock,
                             fill=head, font=clock_font)
        caption = standby_caption(now)
        self.create_text(w // 2, cy + px(CLOCK_BOX) // 2 + px(14),
                         anchor="center", text=caption,
                         fill=faint, font=ui_display(theme.SIZE_CAPTION))
        rows = standby_rows(self._room)
        top = cy + px(CLOCK_BOX) // 2 + px(48)
        if holo:
            self._draw_framed_rows(rows, w, h, top, body, faint)
            return
        self._draw_rows(rows, w, h, top, body, faint, centered=True)

    def _pad_x(self) -> int:
        """The band's horizontal text inset: PAD, stepped in by BAND_INDENT
        in holo so the text clears the frame stroke. Classic keeps PAD."""
        return px(PAD) + (px(BAND_INDENT) if theme.LOOK == "holo" else 0)

    def _draw_ambient(self, w, h, head, body, faint) -> None:
        now = datetime.now()
        holo = theme.LOOK == "holo"
        pad = self._pad_x()
        self.create_text(pad, px(PAD) + px(6), anchor="w",
                         text=clock_text(now), fill=head,
                         font=ui_display(AMBIENT_CLOCK, "semibold"))
        self.create_text(w - pad, px(PAD) + px(6), anchor="e",
                         text=tracked(date_text(now)) if holo
                         else date_text(now),
                         fill=faint, font=ui_display(theme.SIZE_CAPTION))
        y = px(PAD) + px(34)
        if holo:
            # the header rail is segmented like ref2's status bar: a short
            # lit lead, then a dim run one step above the row separators
            (a0, a1), (b0, b1) = rail_segments(pad, w - pad, RAIL_LIT_FRAC)
            self.create_line(b0, y, b1, y, fill=dim(theme.LINE, self._dim))
            self.create_line(a0, y, a1, y, fill=dim(theme.EDGE, self._dim))
        else:
            self.create_line(px(PAD), y, w - px(PAD), y,
                             fill=dim(theme.HOLO_DIM, self._dim))
        top = y + px(18)
        n = self._draw_rows(room_rows(self._room), w, h, top, body, faint)
        if holo:
            self._draw_frame(*band_frame(w, h, px(FRAME_INSET), y, top, n,
                                         px(ROW_H), px(14)))
        frac = gpu_fraction(self._room)
        if frac is not None:
            self._draw_gpu(w, h, frac, head, faint)

    def _draw_rows(self, rows, w, h, top, body, faint,
                   centered: bool = False) -> int:
        """Draws the rows; returns how many, so the holo band can fit its
        frame to what was actually drawn."""
        rows = visible_rows(rows, h, px(ROW_H), top)
        if self._reveal is not None:
            rows = rows[:max(0, int(self._reveal))]
        holo = theme.LOOK == "holo"
        pad = self._pad_x()
        lf = ui_display(theme.SIZE_CAPTION, "semibold")
        vf = ui_mono(theme.SIZE_CAPTION)
        budget = max(px(60), w - 2 * pad - px(72))
        if holo and not centered:
            # thin rules between the rows, inset to the text edges — the
            # frame carries the outline, these carry the rhythm
            for sy in separator_ys(top, len(rows), px(ROW_H)):
                self.create_line(pad, sy, w - pad, sy,
                                 fill=dim(theme.HOLO_DIM, self._dim))
        for i, (label, value) in enumerate(rows):
            y = top + i * px(ROW_H)
            text = (fitted if holo else ellipsize)(str(value), vf, budget)
            if centered:
                self.create_text(w // 2, y, anchor="center",
                                 text=f"{label}   {text}", fill=body, font=vf)
                continue
            self.create_text(pad, y, anchor="w",
                             text=(theme.caption(label, surface=False)
                                   if holo else label),
                             fill=faint, font=lf)
            self.create_text(w - pad, y, anchor="e", text=text,
                             fill=body, font=vf)
        return len(rows)

    # ------------------------------------------------------- holo chrome
    def _draw_frame(self, x0, y0, x1, y1) -> None:
        """The 1px chamfered outline plus its four bright corner brackets.
        No fill — the ground shows through; on black the strokes ARE the
        panel (ref2_hud). Dimmed with the slab so the frame never outshines
        the clock at 3 a.m."""
        if x1 - x0 < px(24) or y1 - y0 < px(24):
            return
        cut = px(FRAME_CUT)
        self.create_polygon(*hud_frame_points(x0, y0, x1, y1, cut), fill="",
                            outline=dim(theme.LINE, self._dim), width=1)
        # BRIGHT, not EDGE: at the standby dim the EDGE step was
        # indistinguishable from the LINE frame it sits on (r1 shot, 09-01)
        for pts in hud_bracket_points(x0, y0, x1, y1, cut, px(BRACKET_LEN)):
            self.create_line(*pts, fill=dim(theme.BRIGHT, self._dim), width=1)

    def _draw_clock_line(self, w, cy, clock: str, clock_font, now, head,
                         body) -> None:
        """Holo standby: '4:26' with its 'PM' inline after the digits, the
        pair centred as one group, baselines met, the glow line under the
        digits (U09, 09-03: the meridiem was a detached caption below)."""
        mer = meridiem(now)
        mer_font = ui_display(MERIDIEM_SIZE, "semibold")
        try:
            digits_w = measured(clock_font, clock)
        except Exception:                   # noqa: BLE001 - Tk font boundary
            digits_w = int(w * 0.4)
        try:
            mer_w = measured(mer_font, mer)
        except Exception:                   # noqa: BLE001 - Tk font boundary
            mer_w = px(30)
        dx, mx = clock_line(w // 2, digits_w, mer_w, px(MERIDIEM_GAP))
        self.create_text(dx, cy, anchor="w", text=clock, fill=head,
                         font=clock_font)
        try:
            asc, _desc, ls = font_metrics(clock_font)
            _a, mer_desc, _l = font_metrics(mer_font)
            self.create_text(mx, meridiem_bottom(cy, ls, asc, mer_desc),
                             anchor="sw", text=mer, fill=body, font=mer_font)
        except Exception:                   # noqa: BLE001 - Tk font boundary
            self.create_text(mx, cy, anchor="w", text=mer, fill=body,
                             font=mer_font)
        self._draw_clock_glow(w, cy + px(CLOCK_BOX) // 2, clock, clock_font,
                              cx=dx + digits_w // 2)

    def _draw_clock_glow(self, w, y, clock: str, font, cx=None) -> None:
        """The faint cyan line under the digits — the reactor's two-stroke
        fake glow (a wide dim underlay beneath a narrow brighter core),
        as long as the digits themselves, centred on `cx` (the digits'
        centre; the slab's when not given)."""
        try:
            text_w = measured(font, clock)
        except Exception:                   # noqa: BLE001 - Tk font boundary
            text_w = int(w * 0.4)
        x0, x1 = glow_span(w // 2 if cx is None else int(cx), text_w,
                           px(GLOW_MIN_W), w - 2 * px(PAD))
        spans = glow_steps(x0, x1)
        # the soft underlay sits under the MIDDLE span only, so the ends of
        # the line thin out to the 1px stroke instead of a blunt bar
        self.create_line(spans[1][0], y, spans[1][1], y,
                         fill=dim(theme.GLOW_UNDER, self._dim),
                         width=max(2, px(2)))
        for (xa, xb), tone in zip(spans, (theme.HOLO_DIM, theme.HOLO,
                                          theme.EDGE)):
            self.create_line(xa, y, xb, y, fill=dim(tone, self._dim), width=1)

    def _draw_framed_rows(self, rows, w, h, top, body, faint) -> None:
        """Standby, holo: label/value pairs inside the HUD frame with thin
        separators, instead of the classic centred 'LABEL   VALUE' line.
        Same visible_rows / reveal truncation as the classic list, so the
        frame can never reach past the slab's bottom edge either."""
        rows = visible_rows(rows, h, px(ROW_H), top)
        if self._reveal is not None:
            rows = rows[:max(0, int(self._reveal))]
        box = frame_box(w, top, len(rows), px(ROW_H), frac=STANDBY_FRAME_FRAC,
                        min_w=px(STANDBY_FRAME_MIN), side_pad=px(PAD),
                        inset_y=px(4))
        if box is None:
            return
        x0, y0, x1, y1 = box
        self._draw_frame(x0, y0, x1, y1)
        lf = ui_display(theme.SIZE_CAPTION, "semibold")
        vf = ui_mono(theme.SIZE_CAPTION)
        inset = px(ROW_INSET_X)
        rule = dim(theme.HOLO_DIM, self._dim)
        for sy in separator_ys(top, len(rows), px(ROW_H)):
            self.create_line(x0 + inset, sy, x1 - inset, sy, fill=rule)
        # the label column is the widest label; the value gets the rest of
        # the frame, and is ellipsized against exactly that. Row keys are
        # plain caps, not tracked: they name a ROW, and only a label that
        # names a surface is tracked (theme.caption, the 09-03 U16 rule)
        labels = [theme.caption(label, surface=False) for label, _v in rows]
        try:
            label_w = max(measured(lf, t) for t in labels) if labels else 0
        except Exception:                   # noqa: BLE001 - Tk font boundary
            label_w = px(90)
        budget = max(px(60), (x1 - x0) - 2 * inset - label_w - px(16))
        for i, ((_label, value), text) in enumerate(zip(rows, labels)):
            y = top + i * px(ROW_H)
            self.create_text(x0 + inset, y, anchor="w", text=text, fill=faint,
                             font=lf)
            self.create_text(x1 - inset, y, anchor="e",
                             text=fitted(str(value), vf, budget), fill=body,
                             font=vf)

    def _draw_gpu(self, w, h, frac: float, head, faint) -> None:
        """GPU load as a bar on the slab's bottom edge — a number would be
        a readout; a bar is a glance."""
        pad = self._pad_x()
        bw = min(px(GPU_BAR_W), max(px(40), w - 2 * pad))
        x0 = pad
        y = h - px(PAD)
        self.create_line(x0, y, x0 + bw, y, fill=faint, width=max(1, px(2)))
        self.create_line(x0, y, x0 + int(bw * frac), y, fill=head,
                         width=max(1, px(2)))


__all__ = ["RoomSlab", "band_frame", "clock_line", "clock_text",
           "date_text", "fitted", "font_metrics",
           "frame_box", "glow_span", "glow_steps", "gpu_fraction",
           "hud_bracket_points", "hud_frame_points", "measured",
           "meridiem", "quiet_ink", "rail_segments", "room_rows",
           "meridiem_bottom", "separator_ys", "slab_ink", "slab_tone",
           "standby_caption", "standby_rows",
           "tracked", "visible_rows", "ACTIVE", "AMBIENT", "STANDBY"]
