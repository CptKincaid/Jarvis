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
"""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from typing import Optional

from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.console_mode import ACTIVE, AMBIENT, STANDBY, dim
from jarvis.ui.widgets import ellipsize, px, ui_display, ui_mono

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
QUIET_INK = theme.WARN      # the ember the whole slab takes in quiet hours

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
        """(head, body, faint) for the slab's tone, already dimmed. Quiet
        hours take the whole palette ember; away drops the contrast."""
        if tone == "quiet":
            head, body, faint = QUIET_INK, dim(QUIET_INK, 0.7), theme.FAINT
        elif tone == "away":
            head, body, faint = theme.MUTED, theme.MUTED, theme.FAINT
        else:
            head, body, faint = theme.FOCAL, theme.INK, theme.MUTED
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
        self.create_text(w // 2, cy, anchor="center", text=clock_text(now),
                         fill=head, font=ui_display(CLOCK_SIZE, "semibold"))
        self.create_text(w // 2, cy + px(CLOCK_BOX) // 2 + px(14),
                         anchor="center", text=f"{meridiem(now)}  ·  "
                                               f"{date_text(now)}",
                         fill=faint, font=ui_display(theme.SIZE_CAPTION))
        rows = standby_rows(self._room)
        top = cy + px(CLOCK_BOX) // 2 + px(48)
        self._draw_rows(rows, w, h, top, body, faint, centered=True)

    def _draw_ambient(self, w, h, head, body, faint) -> None:
        now = datetime.now()
        self.create_text(px(PAD), px(PAD) + px(6), anchor="w",
                         text=clock_text(now), fill=head,
                         font=ui_display(AMBIENT_CLOCK, "semibold"))
        self.create_text(w - px(PAD), px(PAD) + px(6), anchor="e",
                         text=date_text(now), fill=faint,
                         font=ui_display(theme.SIZE_CAPTION))
        y = px(PAD) + px(34)
        self.create_line(px(PAD), y, w - px(PAD), y, fill=dim(theme.HOLO_DIM,
                                                              self._dim))
        self._draw_rows(room_rows(self._room), w, h, y + px(18), body, faint)
        frac = gpu_fraction(self._room)
        if frac is not None:
            self._draw_gpu(w, h, frac, head, faint)

    def _draw_rows(self, rows, w, h, top, body, faint,
                   centered: bool = False) -> None:
        rows = visible_rows(rows, h, px(ROW_H), top)
        if self._reveal is not None:
            rows = rows[:max(0, int(self._reveal))]
        lf = ui_display(theme.SIZE_CAPTION, "semibold")
        vf = ui_mono(theme.SIZE_CAPTION)
        budget = max(px(60), w - 2 * px(PAD) - px(72))
        for i, (label, value) in enumerate(rows):
            y = top + i * px(ROW_H)
            text = ellipsize(str(value), vf, budget)
            if centered:
                self.create_text(w // 2, y, anchor="center",
                                 text=f"{label}   {text}", fill=body, font=vf)
                continue
            self.create_text(px(PAD), y, anchor="w", text=label, fill=faint,
                             font=lf)
            self.create_text(w - px(PAD), y, anchor="e", text=text,
                             fill=body, font=vf)

    def _draw_gpu(self, w, h, frac: float, head, faint) -> None:
        """GPU load as a bar on the slab's bottom edge — a number would be
        a readout; a bar is a glance."""
        bw = min(px(GPU_BAR_W), max(px(40), w - 2 * px(PAD)))
        x0 = px(PAD)
        y = h - px(PAD)
        self.create_line(x0, y, x0 + bw, y, fill=faint, width=max(1, px(2)))
        self.create_line(x0, y, x0 + int(bw * frac), y, fill=head,
                         width=max(1, px(2)))


__all__ = ["RoomSlab", "clock_text", "date_text", "gpu_fraction", "meridiem",
           "room_rows", "slab_tone", "standby_rows", "visible_rows",
           "ACTIVE", "AMBIENT", "STANDBY"]
