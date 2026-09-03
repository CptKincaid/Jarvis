"""The tab row under the wordmark: one row, any number of surfaces.

Hunter, 2026-09-03, verbatim: "just make the app larger if we are running
out of header room or better yet, make a little tab to click thats
underneath jarvis, that can be the area that has multiple tabs".

So the strip is its OWN ROW, directly beneath the header rule::

    JARVIS                                   [READY] [SENSING]
    ----------------------------------------------------------
     [ CHAT ]  [ SENSORS ]                      <- this module
    ----------------------------------------------------------
     ...the stage...

WHY NOT THE HEADER. tests/test_header_fit.py MEASURES the header's width
budget at his 920-px window: 312 px are left for chips and the state pill
plus the sensing badge already spend 299 of them. A `[SENSORS]` chip up
there would clip the sensing badge, which is the privacy readout, and
tkPack UNMAPS a child it cannot fit (jarvis/ui/views.header_spans) -- an
absent badge and a badge reading SENSING must not look the same. His own
ruling was "dont make jarvis smaller", and a wider window only moves the
problem one tab along. A row of its own has the whole width and grows.

WHY IT MUST HOLD MORE THAN TWO. It ships with CHAT and SENSORS and more
are coming, so the strip owns no list of its own: a surface is added with
one ``add()`` call naming the key, the word and what to show::

    self.tabs.add("logs", "LOGS", select=self._show_logs)

Nothing else changes -- not this module, not the fit test, which is
parametrised over 2, 3 and 4 tabs and measures the real widths.

WHAT SELECTION LOOKS LIKE, and why it is not a focus ring. The console
sits on his desk unfocused most of the day, so the selected tab is drawn
with three channels that owe nothing to focus: a brighter word (FOCAL vs
MUTED), a lit underline the width of the tab, and a faint raised ground.
Keyboard FOCUS is the separate highlight ring every other widget in the
tree uses. A tab that only looked selected while the window had focus
would be unreadable in exactly the state he leaves it in.

KEYBOARD. Every tab takes focus in the normal traversal order; Left and
Right walk the row, Return and space select. Selecting is explicit rather
than automatic on focus, so arrowing past SENSORS does not start its poll
thread on the way by.

NOT IN STANDBY. The console hides the whole row with the footer and
selects CHAT on the way in (main_window._set_tabs_hidden): the quiet mode
is "a clock and nothing else", and a lit tab over that clock is one click
from starting a poll thread behind it.

Everything above the widget is a pure function of MEASURED text widths, so
the row's FIT is answered with no display at all -- the same split
jarvis/ui/views.py and tests/test_header_fit.py make. Its BEHAVIOUR is
not: tests/test_tab_strip.py starts a private Xvfb of its own, builds the
real row on it and drives it, because a grep for "<Return>" passes on code
whose binding has been deleted.
"""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.views import header_spans, spans_clipped
from jarvis.ui.widgets import px, ui_display

log = get_logger("ui.tab_strip")

# Design units, at the 96-dpi baseline; px() scales them by S.
TAB_PAD_X = 12          # ink to the tab's own edge, each side
TAB_PAD_Y = 6           # above and below the word
TAB_GAP = 6             # between two tabs
UNDERLINE = 2           # the selected tab's lit rule
STRIP_PAD_Y = 4         # above and below the whole row

TAB_SIZE = theme.SIZE_CAPTION   # a type SIZE, not a colour: look-independent


@dataclass(frozen=True)
class Tab:
    """One surface in the row. ``select`` is called when this tab becomes
    the selected one, ``leave`` when it stops being it -- so a page that
    holds a resource (the SENSORS page holds a poll thread) is started and
    stopped by the strip and by nothing else."""
    key: str
    text: str
    select: Optional[Callable] = None
    leave: Optional[Callable] = None


# ------------------------------------------------------------- the geometry
# Every tab is a Canvas with highlightthickness 1 (its keyboard focus ring),
# so its REQUESTED size is the drawn box plus a border a side. Counting that
# here is what makes tab_width()/strip_height() equal winfo_reqwidth() and
# winfo_reqheight() to the pixel -- measured under a private Xvfb at S=1.0
# and S=2.0 in both looks, and pinned in tests/test_tab_strip.py.
BORDER = 1              # device px, NOT a design unit: a focus ring is 1 px


def _int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def tab_box_w(text_w: int) -> int:
    """The DRAWN box of a tab -- what the Canvas is configured to. The
    border is outside it, which is why this and ``tab_width`` differ."""
    return _int(text_w) + 2 * px(TAB_PAD_X)


def tab_box_h(line_h: int) -> int:
    """The drawn box's height. The underline is INSIDE it, so a selected
    tab and an unselected one are exactly the same size and the row does
    not twitch on a click."""
    return _int(line_h) + 2 * px(TAB_PAD_Y) + max(1, px(UNDERLINE))


def tab_width(text_w: int) -> int:
    """A tab's REQUESTED width from its measured text width (pure) --
    winfo_reqwidth(), which is what the packer is handed."""
    return tab_box_w(text_w) + 2 * BORDER


def tab_height(line_h: int) -> int:
    """A tab's requested height from the font's linespace (pure)."""
    return tab_box_h(line_h) + 2 * BORDER


def strip_height(line_h: int) -> int:
    """What the whole row costs the stage, in device px (pure).

    This is the number that has to come out of the SENSORS page's budget,
    and it is the row's own pad above and below plus the hairline it hangs
    the stage off -- not a guess at "about 40 px".
    """
    return tab_height(line_h) + 2 * px(STRIP_PAD_Y) + max(1, px(1))


def strip_items(widths) -> list:
    """[(key, requested_w, pad_left, pad_right)] in PACKING order, the
    shape ``views.header_spans`` takes.

    ``widths`` is [(key, requested_w)] -- already tab widths, because the
    live strip measures them off the widgets. The trailing gap is on the
    RIGHT of every tab, which is how they are packed, so a tab added at
    the end changes nothing about the ones before it.
    """
    return [(key, int(w), 0, px(TAB_GAP)) for key, w in (widths or ())]


def row_width(total_w: int) -> int:
    """The width the tabs actually get: the window less the row's own pad
    a side (``self._row`` is packed padx=theme.PAD)."""
    return max(0, int(total_w) - 2 * theme.PAD)


def strip_spans(total_w: int, widths) -> list:
    """Where each tab lands and how wide it is actually DRAWN (pure).

    Left-packed children only, through the same transcription of Tk 8.6's
    packer the header fit test uses, so "does the row fit" is answered by
    the same arithmetic in both places rather than by two guesses.
    """
    return header_spans(row_width(total_w), strip_items(widths), [])


def strip_clipped(total_w: int, widths) -> list:
    """[(key, drawn_px, requested_px)] for every tab the packer had to cut
    down, 0 drawn px meaning Tk unmapped it outright. Empty means the row
    fits. This is the row's whole fitting assertion -- the same one
    tests/test_header_fit.py makes about the header, and it exists because
    tkPack does not overlap an over-subscribed row, it TRUNCATES the last
    child packed and then stops drawing it (jarvis/ui/views.header_spans).
    """
    return spans_clipped(strip_spans(total_w, widths))


def tabs_that_fit(total_w: int, tab_w: int) -> int:
    """How many tabs of one width the row holds at ``total_w`` (pure).
    The answer to "can it hold a third", asked as a number."""
    width = _int(tab_w)
    if width <= 0:                        # a tab of no width is not a tab
        return 0
    return max(0, row_width(total_w) // (width + px(TAB_GAP)))


# ============================================================ the Tk surface
class _Tab(tk.Canvas):
    """One clickable tab. Not a ``RoundButton``: a button has no selected
    state, and the three channels that mark this one (word, underline,
    ground) all have to survive an unfocused window."""

    def __init__(self, parent, text: str, command: Optional[Callable] = None,
                 bg=None):
        bg = bg or parent.cget("bg")
        self._font = ui_display(TAB_SIZE, "semibold")
        try:
            tw = tkfont.Font(font=self._font).measure(text)
            th = tkfont.Font(font=self._font).metrics("linespace")
        except Exception:                     # noqa: BLE001 - no font metrics
            tw, th = px(7) * max(1, len(text)), px(TAB_SIZE + 6)
        # NOT _w / _h: tkinter.Misc uses `_w` for the widget's Tcl pathname
        # and super().__init__ overwrites it, so the first draw compared a
        # string with an int. RoundButton names its own `_btn_w` for the
        # same reason. tests/test_tab_strip.py scans for the whole family.
        self._box_w, self._box_h = tab_box_w(tw), tab_box_h(th)
        super().__init__(parent, width=self._box_w, height=self._box_h, bg=bg,
                         highlightthickness=1, bd=0, takefocus=1,
                         cursor="hand2")
        self._text = text
        self.command = command
        self._selected = False
        self._hovered = False
        self._draw()
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))
        self.bind("<ButtonRelease-1>", lambda e: self.invoke())
        self.bind("<Return>", lambda e: self.invoke())
        self.bind("<space>", lambda e: self.invoke())
        self.bind("<FocusIn>", lambda e: self._ring(True))
        self.bind("<FocusOut>", lambda e: self._ring(False))
        self.bind("<Configure>", self._on_configure, add=True)
        self._ring(False)

    # ------------------------------------------------------------- state
    @property
    def text(self) -> str:
        return self._text

    @property
    def selected(self) -> bool:
        return self._selected

    def set_selected(self, value: bool) -> None:
        self._selected = bool(value)
        self._draw()

    def invoke(self) -> None:
        if self.command:
            self.command()

    def _hover(self, on: bool) -> None:
        self._hovered = bool(on)
        self._draw()

    def _ring(self, on: bool) -> None:
        """The keyboard focus ring, read at call time so a look switch
        lands. Deliberately a DIFFERENT channel from selection."""
        try:
            self.configure(highlightbackground=self.cget("bg"),
                           highlightcolor=theme.CYAN if on else self.cget("bg"))
        except tk.TclError:                   # noqa: BLE001 - a dead widget
            log.debug("tab strip: focus ring failed", exc_info=True)

    def _on_configure(self, event) -> None:
        key = (event.width, event.height)
        if getattr(self, "_cfg_last", None) == key:
            return
        self._cfg_last = key
        self._draw()

    # ------------------------------------------------------------- paint
    def _draw(self) -> None:
        self.delete("all")
        w = max(self.winfo_width(), self._box_w)
        h = max(self.winfo_height(), self._box_h)
        rule = max(1, px(UNDERLINE))
        if self._selected:
            fill, fg = theme.RAISED, theme.FOCAL
        elif self._hovered:
            fill, fg = theme.RAISED, theme.INK
        else:
            fill, fg = "", theme.MUTED
        if fill:
            self.create_rectangle(0, 0, w, h - rule, fill=fill, outline="")
        self.create_text(w // 2, (h - rule) // 2, text=self._text, fill=fg,
                         font=self._font)
        if self._selected:
            # The channel that survives an unfocused window: a lit rule the
            # full width of the tab, on the row's own bottom edge.
            self.create_rectangle(0, h - rule, w, h, fill=theme.CYAN,
                                  outline="")


class TabStrip(tk.Frame):
    """The row itself. Add a surface with one line::

        strip.add("sensors", "SENSORS", select=page.show, leave=page.hide)

    ``select(key)`` is idempotent -- selecting the tab that is already
    selected does nothing at all -- so a page whose own close path syncs
    the strip back cannot recurse through it.
    """

    def __init__(self, parent, bg=None, on_change: Optional[Callable] = None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, bg=bg)
        self._bg = bg
        self._on_change = on_change
        self._specs: dict = {}
        self._widgets: dict = {}
        self._order: list = []
        self._selected = ""
        self._row = tk.Frame(self, bg=bg)
        self._row.pack(fill="x", padx=theme.PAD, pady=px(STRIP_PAD_Y))
        # The hairline the stage hangs off, so the row reads as a shelf
        # rather than as two buttons floating over the reactor.
        self._rule = tk.Frame(self, bg=theme.LINE, height=max(1, px(1)))
        self._rule.pack(fill="x", side="bottom")

    # --------------------------------------------------------------- API
    @property
    def keys(self) -> tuple:
        return tuple(self._order)

    @property
    def selected(self) -> str:
        return self._selected

    def widget(self, key: str):
        return self._widgets.get(key)

    def add(self, key: str, text: str, select: Optional[Callable] = None,
            leave: Optional[Callable] = None):
        """One more surface in the row. The FIRST tab added is selected,
        because a strip with nothing selected has no state to show and the
        stage underneath it would be nobody's."""
        key = str(key)
        if key in self._specs:
            raise ValueError("tab %r is already in the strip" % key)
        spec = Tab(key=key, text=str(text), select=select, leave=leave)
        self._specs[key] = spec
        self._order.append(key)
        tab = _Tab(self._row, spec.text, command=lambda k=key: self.select(k),
                   bg=self._bg)
        tab.pack(side="left", padx=(0, px(TAB_GAP)))
        tab.bind("<Left>", lambda e, k=key: self._step(k, -1))
        tab.bind("<Right>", lambda e, k=key: self._step(k, +1))
        self._widgets[key] = tab
        if len(self._order) == 1:
            self.select(key, run=False)
        return tab

    def select(self, key: str, run: bool = True) -> bool:
        """Make ``key`` the selected tab. Returns whether it changed.

        ``run`` False paints the selection without calling the callbacks --
        the first tab's surface is already the one on screen at build time,
        and calling show() on it would be a second, redundant open.
        """
        key = str(key)
        if key not in self._specs or key == self._selected:
            return False
        was, self._selected = self._selected, key
        for k, tab in self._widgets.items():
            tab.set_selected(k == key)
        if not run:
            return True
        self._call(self._specs.get(was), "leave")
        self._call(self._specs.get(key), "select")
        if callable(self._on_change):
            try:
                self._on_change(key)
            except Exception:                 # noqa: BLE001 - a callback
                log.exception("tab strip: on_change failed")
        return True

    def focus_selected(self) -> None:
        tab = self._widgets.get(self._selected)
        if tab is not None:
            try:
                tab.focus_set()
            except tk.TclError:               # noqa: BLE001 - a dead widget
                log.debug("tab strip: focus failed", exc_info=True)

    # ----------------------------------------------------------- measured
    def measured_widths(self) -> list:
        """[(key, requested_w)] off the LIVE widgets, for the fit check the
        console runs at build time. Not the same thing as ``strip_items``:
        that one is pure and takes text widths, this one asks Tk."""
        out = []
        for key in self._order:
            tab = self._widgets.get(key)
            try:
                out.append((key, int(tab.winfo_reqwidth())))
            except Exception:                 # noqa: BLE001 - an unmapped tab
                out.append((key, 0))
        return out

    def clipped(self, total_w: int) -> list:
        """Which tabs the packer had to cut at ``total_w`` -- the live
        version of ``strip_clipped``, measured rather than assumed."""
        return strip_clipped(total_w, self.measured_widths())

    # ---------------------------------------------------------- internals
    def _call(self, spec: Optional[Tab], which: str) -> None:
        fn = getattr(spec, which, None) if spec is not None else None
        if not callable(fn):
            return
        try:
            fn()
        except Exception:                     # noqa: BLE001 - a surface
            log.exception("tab strip: %s on %s failed", which,
                          getattr(spec, "key", "?"))

    def _step(self, key: str, delta: int) -> str:
        """Left/Right move FOCUS along the row and stop at the ends. They
        do not select: arrowing past SENSORS must not start its poll."""
        if key not in self._order:
            return "break"
        i = min(len(self._order) - 1, max(0, self._order.index(key) + delta))
        tab = self._widgets.get(self._order[i])
        if tab is not None:
            try:
                tab.focus_set()
            except tk.TclError:               # noqa: BLE001 - a dead widget
                log.debug("tab strip: focus failed", exc_info=True)
        return "break"
