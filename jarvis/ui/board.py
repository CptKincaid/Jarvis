"""The Board — a second borderless surface docked down the right flank.

The console is 520x880 on a 3840x2160 panel; 93% of the desk is black. The
Board fills the right flank with the same holo theme as the reactor's
engine card: chamfered glass slabs, display-face labels at SIZE_CAPTION,
mono values, one hairline rule per panel tinted by its state.

Division of labour, deliberately:

  jarvis/board.py     WHAT it shows — pure, provider-injected, no Tk
  this module         WHERE things sit — pure geometry helpers — plus the
                      thinnest possible Tk glue over them

Every layout decision (`dock_geometry`, `panel_boxes`, `spark_points`) is a
pure function unit-tested with no widget, exactly as jarvis/ui/views.py
splits `you_card_width` / `plan_strip` out of the TranscriptView.

Three traps this window is built around:

* **The WM-managed window is not `winfo_id()`.** Tk wraps every toplevel,
  so decorations are stripped through the window the WM actually manages —
  found by TITLE here rather than by WM_CLASS, because the console already
  owns `^jarvis$` and a class search would return whichever window came
  first (main_window._wm_window_id returns the first match by design).

* **Focus churn froze :1 on 2026-08-26.** The Toplevel takes `takefocus=0`,
  is never raised on an event, and is created ONCE and withdrawn/deiconified
  rather than destroyed and rebuilt.

* **The data must not ride the Tk thread.** `BoardFeed` is a plain joinable
  thread that calls the injected state provider (which spawns nvidia-smi and
  may hit a cached Canvas answer) and PUBLISHES the result; the window
  subscribes, so the bus does the marshalling it exists to do.
"""
from __future__ import annotations

import shutil
import subprocess
import tkinter as tk
from typing import Callable, Optional

from jarvis.board import BoardState
from jarvis.events import BoardUpdate, bus
from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.widgets import ellipsize, px, ui_display, ui_mono

log = get_logger("ui.board")

WINDOW_TITLE = "Jarvis Board"     # the handle _strip_board_decorations uses
BOARD_W = 520                     # design units, same column width as the console
MARGIN = 24
MIN_W = 240
PANEL_PAD = 12                    # design units between slabs
ROW_H = 22                        # design units per (label, value) row
HEAD_H = 30                       # panel title + hairline
SPARK_H = 46                      # the turn ledger's strip chart
TALL_PANELS = frozenset({"turns"})   # panels that earn extra height
HIGHLIGHT_MS = 2600               # "focus on the sessions" — how long it lights

TONE_COLORS = {
    "ok": theme.FOCAL,
    "warn": theme.WARN,
    "error": theme.ERR,
    "idle": theme.CYAN_DIM,
    "off": theme.FAINT,
}


# ------------------------------------------------------------ pure layout
def dock_geometry(screen_w: int, screen_h: int, width: int = BOARD_W,
                  margin: int = MARGIN, console_w: int = 0) -> str:
    """A Tk geometry string docking a `width` column against the RIGHT edge,
    `margin` in from every side. The width shrinks to whatever is left
    beside the console rather than overlapping it, and never goes below
    MIN_W or off a small screen. Pure — unit tested with no display."""
    screen_w = max(1, int(screen_w))
    screen_h = max(1, int(screen_h))
    margin = max(0, int(margin))
    free = screen_w - max(0, int(console_w)) - 2 * margin
    w = min(int(width), free) if free > 0 else int(width)
    w = max(MIN_W, w)
    w = min(w, max(1, screen_w - 2 * margin))
    h = max(1, screen_h - 2 * margin)
    return f"{w}x{h}+{screen_w - w - margin}+{margin}"


def panel_boxes(height: int, keys, pad: int = PANEL_PAD,
                tall=TALL_PANELS) -> list:
    """(key, y0, y1) for a vertical stack filling `height`, `pad` between
    slabs and at both ends. Panels named in `tall` take ~1.6x the share
    (the turn ledger carries a strip chart; a name/value list does not).

    A height too small for the stack still returns ONE box per key, in
    order, with y1 >= y0 — a degenerate board is drawn empty rather than
    inverted or dropped."""
    keys = list(keys or [])
    if not keys:
        return []
    pad = max(0, int(pad))
    avail = max(0, int(height) - pad * (len(keys) + 1))
    weights = [1.6 if k in (tall or ()) else 1.0 for k in keys]
    total = sum(weights) or 1.0
    boxes = []
    y = pad
    for i, (key, weight) in enumerate(zip(keys, weights)):
        span = int(avail * weight / total)
        y1 = y + span
        if i == len(keys) - 1:
            # the last slab absorbs the rounding so the stack cannot
            # overrun the surface by a pixel or two
            y1 = max(y, int(height) - pad)
        boxes.append((key, y, y1))
        y = y1 + pad
    return boxes


def spark_points(values, x0: int, y0: int, w: int, h: int) -> list:
    """Flat [x, y, x, y, …] for a sparkline polyline: `values` are 0..1
    oldest-first (jarvis.board.sparkline), the peak touches the top and a
    zero sits on the baseline. Fewer than two points returns []: Tk cannot
    draw a one-point line."""
    vals = list(values or [])
    if len(vals) < 2:
        return []
    step = w / (len(vals) - 1)
    pts = []
    for i, v in enumerate(vals):
        v = max(0.0, min(1.0, float(v)))
        pts.append(int(x0 + i * step))
        pts.append(int(y0 + h * (1.0 - v)))
    return pts


def rows_that_fit(rows, height: int, row_h: int, head_h: int) -> list:
    """As many (label, value) rows as the slab has room for. Pure: the
    renderer never has to guess, and a short panel truncates instead of
    drawing over its neighbour."""
    room = int(height) - int(head_h)
    n = max(0, room // max(1, int(row_h)))
    return list(rows or [])[:n]


# ---------------------------------------------------- borderless toplevel
def _board_window_id(title: str = WINDOW_TITLE) -> Optional[str]:
    """The WM-managed X window for the Board, found by TITLE.

    main_window._wm_window_id searches by WM_CLASS and returns the FIRST
    match, which is correct for a single-window app and wrong the moment a
    second toplevel exists — it would hand back the console and strip its
    decorations twice while the Board kept its titlebar."""
    if not shutil.which("xdotool"):
        return None
    try:
        r = subprocess.run(["xdotool", "search", "--name", f"^{title}$"],
                           capture_output=True, text=True, timeout=3)
    except Exception:                       # noqa: BLE001 - tool boundary
        log.debug("xdotool search failed", exc_info=True)
        return None
    ids = (r.stdout or "").split()
    return ids[-1] if ids else None


def strip_board_decorations(top, title: str = WINDOW_TITLE) -> bool:
    """Undecorate the Board without changing its window TYPE — the same
    Motif-hints mechanism main_window._strip_decorations uses, addressed by
    title (see _board_window_id).

    The splash fallback is BETTER here than on the console: a docked panel
    that taskbars skip is what we want anyway; the console needed the dash
    entry, the Board does not."""
    def _fallback(why: str) -> bool:
        log.info("board: %s; using the splash window type", why)
        try:
            top.attributes("-type", "splash")
        except tk.TclError:
            log.warning("board: could not set the splash window type either")
        return False

    if not (shutil.which("xprop") and shutil.which("xdotool")):
        return _fallback("xprop/xdotool not installed")
    try:
        top.update_idletasks()
    except tk.TclError:
        return _fallback("no X window yet")
    wid = _board_window_id(title)
    if not wid:
        return _fallback("could not find the WM-managed window")
    try:
        r = subprocess.run(
            ["xprop", "-id", wid, "-f", "_MOTIF_WM_HINTS", "32c",
             "-set", "_MOTIF_WM_HINTS", "2, 0, 0, 0, 0"],
            capture_output=True, text=True, timeout=3)
    except Exception:                       # noqa: BLE001 - tool boundary
        return _fallback("xprop failed")
    if r.returncode != 0:
        return _fallback(f"xprop rc={r.returncode}")
    log.info("board: decorations stripped on window %s", wid)
    return True


# -------------------------------------------------------------- the window
class BoardWindow:
    """The docked panel itself. Built ONCE and withdrawn when closed —
    never destroyed and rebuilt, because window churn on :1 is what froze
    the desktop on 2026-08-26.

    All layout comes from the pure helpers above; this class only creates
    canvas items and moves text into them."""

    def __init__(self, master, screen: Optional[tuple] = None,
                 console_w: int = 0, on_close: Optional[Callable] = None):
        self.on_close = on_close
        self._visible = False
        self._state: Optional[BoardState] = None
        self._revealed: Optional[int] = None   # power-up: panels shown so far
        self._highlight = None
        self._geom = ""

        self.top = tk.Toplevel(master, takefocus=0)
        self.top.title(WINDOW_TITLE)
        self.top.configure(bg=theme.FRAME)
        if screen is None:
            screen = (master.winfo_screenwidth(), master.winfo_screenheight())
        self._geom = dock_geometry(screen[0], screen[1], px(BOARD_W),
                                   px(MARGIN), console_w)
        self.top.geometry(self._geom)
        self._stripped = False        # decorations come off on first show
        # A closed board is withdrawn, not destroyed: see the class docstring
        self.top.protocol("WM_DELETE_WINDOW", self.hide)
        self.canvas = tk.Canvas(self.top, bg=theme.BG, highlightthickness=0,
                                bd=0, takefocus=0)
        self.canvas.pack(fill="both", expand=True, padx=1, pady=1)
        self.canvas.bind("<Configure>", self._on_configure, add=True)
        self.top.withdraw()
        bus.subscribe(BoardUpdate, self._on_update)
        log.info("board window ready (%s)", self._geom)

    # ------------------------------------------------------- visibility
    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def panel_count(self) -> int:
        """How many slabs the last state carried — the power-up sweep needs
        it to know how many stages to play."""
        return len(self._state.panels) if self._state is not None else 0

    def show(self):
        if self._visible:
            return
        self._visible = True
        try:
            self.top.deiconify()
        except tk.TclError:
            log.debug("board deiconify on a dead window", exc_info=True)
            return
        if not self._stripped:
            # ON FIRST SHOW, not in __init__: the hints are set through the
            # WM-managed window, and a Toplevel that has never been mapped
            # has none for xdotool to find.
            self._stripped = True
            strip_board_decorations(self.top)
        self._redraw()

    def hide(self):
        if not self._visible:
            return
        self._visible = False
        try:
            self.top.withdraw()
        except tk.TclError:
            log.debug("board withdraw on a dead window", exc_info=True)
        if callable(self.on_close):
            self.on_close()

    def destroy(self):
        bus.unsubscribe(BoardUpdate, self._on_update)
        try:
            self.top.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------ data
    def _on_update(self, ev: BoardUpdate):
        """Bus handler — already on the Tk thread (bus.attach_tk)."""
        self.set_state(ev.state)

    def set_state(self, state: Optional[BoardState]):
        self._state = state
        if self._visible:
            self._redraw()

    def reveal(self, count: Optional[int]):
        """Power-up choreography: show only the first `count` panels.
        None restores the whole board."""
        self._revealed = count
        if self._visible:
            self._redraw()

    def highlight(self, key: str):
        """Light one panel while its line is spoken, then let it settle."""
        self._highlight = key or None
        if self._visible:
            self._redraw()
        try:
            self.top.after(HIGHLIGHT_MS, self._clear_highlight)
        except tk.TclError:
            self._highlight = None

    def _clear_highlight(self):
        self._highlight = None
        if self._visible:
            self._redraw()

    # ----------------------------------------------------------- render
    def _on_configure(self, _e):
        if self._visible:
            self._redraw()

    def _redraw(self):
        state, canvas = self._state, self.canvas
        try:
            canvas.delete("all")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
        except tk.TclError:
            return
        if w <= 1 or h <= 1 or state is None:
            return
        panels = list(state.panels)
        if not panels:
            return
        # The grid is laid out from EVERY panel, then only the revealed ones
        # are drawn: the power-up sweep must populate a fixed board, not
        # make each slab shrink as the next one arrives.
        shown = panels if self._revealed is None \
            else panels[:max(0, int(self._revealed))]
        boxes = panel_boxes(h, [p.key for p in panels], px(PANEL_PAD))
        by_key = {p.key: p for p in shown}
        for key, y0, y1 in boxes:
            panel = by_key.get(key)
            if panel is not None and y1 - y0 > px(HEAD_H) // 2:
                self._draw_panel(panel, px(PANEL_PAD), y0,
                                 w - 2 * px(PANEL_PAD), y1 - y0)

    def _draw_panel(self, panel, x: int, y: int, w: int, h: int):
        c = self.canvas
        cut = px(8)
        tone = TONE_COLORS.get(panel.tone, theme.CYAN_DIM)
        lit = panel.key == self._highlight
        outline = theme.BRIGHT if lit else theme.RAMP33
        c.create_polygon(x + cut, y, x + w, y, x + w, y + h - cut,
                         x + w - cut, y + h, x, y + h, x, y + cut,
                         fill=theme.RAISED, outline=outline, width=1)
        c.create_line(x + cut + 1, y + 1, x + w - 1, y + 1,
                      fill=theme.GLASS_EDGE, width=1)
        # tone rule: the panel's STATE, carried by colour rather than a word
        c.create_line(x + 1, y + cut, x + 1, y + h - 1, fill=tone,
                      width=max(1, px(2)))
        pad = px(12)
        c.create_text(x + pad, y + px(16), anchor="w", text=panel.title,
                      fill=theme.FOCAL if lit else theme.MUTED,
                      font=ui_display(theme.SIZE_CAPTION, "semibold"))
        c.create_line(x + pad, y + px(26), x + w - pad, y + px(26),
                      fill=theme.HOLO_DIM)
        head = px(HEAD_H)
        if panel.spark:
            self._draw_spark(panel, x + pad, y + head, w - 2 * pad,
                             px(SPARK_H), tone)
            head += px(SPARK_H) + px(6)
        lf = ui_display(theme.SIZE_CAPTION, "semibold")
        vf = ui_mono(theme.SIZE_CAPTION)
        budget = max(px(40), w - 2 * pad - px(70))
        rows = rows_that_fit(panel.rows, h - head, px(ROW_H), px(12))
        for i, (label, value) in enumerate(rows):
            ry = y + head + px(12) + i * px(ROW_H)
            if ry > y + h - px(4):
                break
            c.create_text(x + pad, ry, anchor="w", text=str(label),
                          fill=theme.MUTED, font=lf)
            c.create_text(x + w - pad, ry, anchor="e",
                          text=ellipsize(str(value), vf, budget),
                          fill=theme.FOCAL, font=vf)

    def _draw_spark(self, panel, x: int, y: int, w: int, h: int, tone: str):
        c = self.canvas
        c.create_line(x, y + h, x + w, y + h, fill=theme.HOLO_DIM)
        pts = spark_points(panel.spark, x, y, w, h)
        if len(pts) >= 4:
            # the reactor's two-stroke fake glow: wide dim under, narrow bright
            c.create_line(*pts, fill=theme.GLOW_UNDER, width=max(2, px(3)),
                          smooth=True)
            c.create_line(*pts, fill=tone, width=max(1, px(1)), smooth=True)


# --------------------------------------------------------------- helper
def board_enabled(get_option: Optional[Callable]) -> bool:
    """`console.board` from assistant.json, defaulting to on. A missing or
    raising option reader must not make the verb dead — the Board is a
    view, and refusing to draw one is never the safer failure."""
    if not callable(get_option):
        return True
    try:
        value = get_option("console.board")
    except Exception:                       # noqa: BLE001 - config boundary
        log.debug("board: option read failed", exc_info=True)
        return True
    return True if value is None else bool(value)
