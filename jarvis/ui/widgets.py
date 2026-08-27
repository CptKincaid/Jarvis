"""Canvas-drawn flat widgets for the Jarvis V3 UI.

Primitives: round_rect / RoundedField, Card, RoundButton, Toggle (animated),
Meter, Chip, Tooltip (timing ported from voice_input_gui.py 968-1009), Toast,
ellipsize helper. All colors and fonts come from jarvis.ui.theme tokens ONLY.

Nothing in this module constructs a Tk root; widgets are created under a
parent the caller owns, so importing this file never requires a display.
"""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.ui import theme

log = get_logger("ui.widgets")

# Font sizing: Tk's Xft backend renders point sizes at the X server's DPI
# and ignores `tk scaling` (on this 4K box: 162dpi → everything 1.7x the
# spec). The V3 spec couples its type scale (26/15/13/11) to a fixed pixel
# layout, so fonts are rendered in PIXEL units (negative Tk sizes) at the
# 96-dpi baseline: px = size * 4/3. set_font_scale() multiplies that.
# Families still come from theme.
_FONT_SCALE = [4.0 / 3.0]

# Global UI scale factor S (HiDPI). Every hardcoded pixel dimension in
# jarvis/ui goes through px(); fonts are scaled ONLY via set_font_scale
# (wired automatically by set_scale) so they are never double-scaled.
_SCALE = [1.0]


def set_font_scale(factor: float):
    _FONT_SCALE[0] = (4.0 / 3.0) * max(0.5, min(3.0, factor))


def set_scale(factor: float):
    """Establish the global UI scale S once at startup (MainWindow), before
    any widgets are built. Also wires the font scale to S."""
    _SCALE[0] = max(0.5, min(3.0, float(factor)))
    set_font_scale(_SCALE[0])


def get_scale() -> float:
    return _SCALE[0]


def px(v) -> int:
    """Design-unit pixels → device pixels at the global UI scale S."""
    return round(v * _SCALE[0])


def ui_font(size: int, weight: str = "normal") -> tuple:
    family = theme.font(size, weight)[0]
    return (family, -max(8, round(size * _FONT_SCALE[0])), weight)


def ui_mono(size: int) -> tuple:
    family = theme.mono(size)[0]
    return (family, -max(8, round(size * _FONT_SCALE[0])), "normal")


def ui_display(size: int, weight: str = "normal") -> tuple:
    """Display face (Rajdhani → body fallback) in pixel units. weight:
    normal | semibold | bold (semibold maps to a named Rajdhani face)."""
    family, _sz, tk_weight = theme.display(size, weight)
    return (family, -max(8, round(size * _FONT_SCALE[0])), tk_weight)


# ---------------------------------------------------------------- helpers
def _hex_rgb(color: str) -> tuple:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def round_rect(canvas: tk.Canvas, x1, y1, x2, y2, radius=None, **kw):
    """Draw a smoothed rounded rectangle polygon on a canvas; returns item id.
    radius defaults to theme.RADIUS at call time (already S-scaled)."""
    if radius is None:
        radius = theme.RADIUS
    radius = max(1, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


def chamfer_rect(canvas: tk.Canvas, x1, y1, x2, y2, cut=None, **kw):
    """HUD panel shape: rectangle with 45°-cut corners; returns item id.
    cut defaults to theme.CHAMFER at call time (already S-scaled)."""
    if cut is None:
        cut = theme.CHAMFER
    cut = max(2, min(cut, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [
        x1 + cut, y1,
        x2 - cut, y1,
        x2, y1 + cut,
        x2, y2 - cut,
        x2 - cut, y2,
        x1 + cut, y2,
        x1, y2 - cut,
        x1, y1 + cut,
    ]
    return canvas.create_polygon(pts, smooth=False, **kw)


def measure(font_spec, text: str) -> int:
    """Pixel width of `text` in EXACTLY the font a canvas/label renders
    with. Measures the raw font description via `font measure`: wrapping
    the spec in tkfont.Font() round-trips a pixel size (-24) through
    `font actual`, which reports rounded POINTS, so the named font came
    out ~30% larger than the rendered one (and leaked a font per call)."""
    root = tk._get_default_root("measure text")
    return int(root.tk.call("font", "measure", font_spec, text))


def ellipsize(text: str, font_spec, max_px: int) -> str:
    """Trim text with a trailing ellipsis so it fits in max_px pixels."""
    try:
        if measure(font_spec, text) <= max_px:
            return text
        while text and measure(font_spec, text + "…") > max_px:
            text = text[:-1]
    except Exception:
        return text
    return text + "…"


# -------------------------------------------------------------- BarGradient
class BarGradient:
    """Barely-perceptible pre-rendered horizontal gradient ground for a bar
    Frame (header / command bar / status strip) so no region reads as a
    flat fill. A placed, lowered Canvas carries one PhotoImage that is
    rebuilt ONLY on debounced resize settle (no per-frame work). Children
    still sitting on the bar's flat base color are re-tinted to the
    gradient color sampled at their center so their solid label/canvas
    backgrounds don't punch flat holes in the ground; children that manage
    their own bg (hairlines, etc.) are left alone. The blend range stays
    at 3-5% max — atmosphere under the content, never competing with it.

    stops: [(x_fraction, blend_fraction), ...] sorted by x, linearly
    interpolated; blend is base→tint (default tint theme.CYAN)."""

    def __init__(self, bar, base: str, stops, tint: Optional[str] = None):
        self.bar = bar
        self._base = base
        self._tint = tint or theme.CYAN
        self._stops = list(stops)
        self._photo = None
        self._pil = None            # the rendered ground (PIL) for slices
        self._img = None
        self._job = None
        self._key = None
        self._tinted: dict = {}     # child -> last color we applied
        self._slices: dict = {}     # canvas child -> its ground slice photo
        self._watched: set = set()  # canvas children re-sliced on resize
        self.canvas = tk.Canvas(bar, bg=base, highlightthickness=0, bd=0)
        self.canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)
        tk.Misc.lower(self.canvas)             # ground below every sibling
        bar.bind("<Configure>", self._schedule, add=True)
        self._schedule()

    def _schedule(self, _e=None):
        if self._job is not None:
            try:
                self.bar.after_cancel(self._job)
            except tk.TclError:
                pass
        self._job = self.bar.after(200, self._rebuild)

    def _f_at(self, xf: float) -> float:
        s = self._stops
        if xf <= s[0][0]:
            return s[0][1]
        for (x0, f0), (x1, f1) in zip(s, s[1:]):
            if xf <= x1:
                return f0 + (f1 - f0) * (xf - x0) / max(1e-6, x1 - x0)
        return s[-1][1]

    def _color_at(self, xf: float) -> str:
        f = self._f_at(max(0.0, min(1.0, xf)))
        b, t = _hex_rgb(self._base), _hex_rgb(self._tint)
        return "#%02x%02x%02x" % tuple(
            int(bc + (tc - bc) * f) for bc, tc in zip(b, t))

    def _rebuild(self):
        self._job = None
        if not self.bar.winfo_exists():
            return
        w, h = self.bar.winfo_width(), self.bar.winfo_height()
        if w < 4 or h < 4:
            return
        if self._key != (w, h):
            self._key = (w, h)
            try:
                from PIL import Image, ImageTk
                row = Image.new("RGB", (w, 1))
                row.putdata([_hex_rgb(self._color_at(x / max(1, w - 1)))
                             for x in range(w)])
                self._pil = row.resize((w, h), Image.NEAREST)
                self._photo = ImageTk.PhotoImage(self._pil)
            except Exception:
                log.exception("bar gradient render failed")
                return
            if self._img is None:
                self._img = self.canvas.create_image(
                    0, 0, anchor="nw", image=self._photo)
                # owners may draw their own items on this canvas (the
                # status strip does) — the ground stays underneath them
                self.canvas.tag_lower(self._img)
            else:
                self.canvas.itemconfigure(self._img, image=self._photo)
        self._retint(w)

    def _retint(self, w: int):
        for child in self.bar.winfo_children():
            if child is self.canvas:
                continue
            try:
                cur = child.cget("bg")
            except tk.TclError:
                continue
            if cur != self._base and cur != self._tinted.get(child):
                continue            # widget owns its bg — leave it alone
            x = child.winfo_x() + child.winfo_width() / 2.0
            col = self._color_at(x / max(1, w))
            try:
                child.configure(bg=col)
                if int(child.cget("highlightthickness")) > 0:
                    child.configure(highlightbackground=col)
            except (tk.TclError, ValueError):
                continue
            self._tinted[child] = col
            # Canvas children (wordmark, state pill, corner brackets) get
            # the EXACT slice of the ground under them as a lowered image
            # item, so a wide canvas over a sloping gradient never shows
            # as a flat slab; re-sliced whenever the child resizes.
            if isinstance(child, tk.Canvas) and \
                    getattr(child, "BARGRAD_SLICE", True):
                self._slice(child)
                if child not in self._watched:
                    self._watched.add(child)
                    child.bind("<Configure>",
                               lambda e, ch=child: self.bar.after_idle(
                                   lambda: self._slice(ch)),
                               add=True)

    def _slice(self, child):
        if self._pil is None:
            return
        try:
            if not child.winfo_exists():
                return
            x0, y0 = child.winfo_x(), child.winfo_y()
            cw, ch = child.winfo_width(), child.winfo_height()
            pw, ph = self._pil.size
            x0 = max(0, min(x0, pw - 1))
            y0 = max(0, min(y0, ph - 1))
            x1, y1 = min(pw, x0 + cw), min(ph, y0 + ch)
            if x1 - x0 < 1 or y1 - y0 < 1:
                return
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(self._pil.crop((x0, y0, x1, y1)))
            child.delete("bargrad")
            child.create_image(0, 0, anchor="nw", image=photo,
                               tags=("bargrad",))
            child.tag_lower("bargrad")
            self._slices[child] = photo
        except Exception:
            log.debug("bar gradient slice failed", exc_info=True)


def _focus_ring(widget: tk.Widget, parent_bg: str):
    """Visible keyboard focus: 1px CYAN_DIM outline via Tk highlight border."""
    widget.configure(
        highlightthickness=max(1, px(1)),
        highlightbackground=parent_bg,
        highlightcolor=theme.CYAN_DIM,
    )


# ------------------------------------------------------------------- Card
class Card(tk.Canvas):
    """HUD panel card: chamfered 45°-cut corners, hairline EDGE border,
    brighter corner accent strokes on the top-left and bottom-right.

    Content goes into `self.body` (a tk.Frame). The card stretches to its
    parent's width when packed with fill="x"; height follows the body.
    `set_left_rule(color, fraction)` draws a 2px vertical rule at the left
    edge whose length maps fraction (0..1) of the body height.
    `set_edge_glow()` stacks faint vertical lines inside the left edge
    (JARVIS reply cards).
    """

    def __init__(self, parent, fill=theme.RAISED, radius=None,
                 pad=12, bg=None, **kw):
        bg = bg or parent.cget("bg")
        # explicit size: Tk's canvas defaults are "10c"x"7c" (640x447 on
        # HiDPI), which distorts layout before the first sync
        kw.setdefault("width", px(100))
        kw.setdefault("height", px(40))
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, **kw)
        self._fill = fill
        # radius kept for API compat; chamfer cut comes from theme.CHAMFER
        self._radius = theme.RADIUS if radius is None else radius
        self._pad = px(pad)
        self._rule: Optional[tuple] = None       # (color, fraction)
        self._glow: Optional[tuple] = None       # left-edge glow colors
        self.body = tk.Frame(self, bg=fill)
        # window origin at the SCALED pad (the card is sized for 2*_pad, so
        # an unscaled origin would hug the top-left at S > 1)
        self._win = self.create_window(self._pad, self._pad, anchor="nw",
                                       window=self.body)
        self._last = (0, 0)
        self.body.bind("<Configure>", self._on_body, add=True)
        self.bind("<Configure>", self._on_self, add=True)

    def _on_body(self, event):
        # Guard: <Configure> also fires on pure moves; skip no-op redraws
        # (interactive window resizes deliver these per pixel step).
        key = (event.width, event.height)
        if getattr(self, "_body_last", None) == key:
            return
        self._body_last = key
        self.sync()

    def sync(self):
        """Fit the card height to the body's requested height. Also called
        by owners after reflowing content (e.g. wraplength changes) since
        unmapped bodies get no <Configure> events."""
        want_h = self.body.winfo_reqheight() + 2 * self._pad
        if int(self.cget("height")) != want_h:
            self.configure(height=want_h)
        self._redraw()

    def _on_self(self, event):
        key = (event.width, event.height)
        if getattr(self, "_self_last", None) == key:
            return
        self._self_last = key
        inner_w = max(1, event.width - 2 * self._pad)
        self.itemconfigure(self._win, width=inner_w)
        self._redraw()

    def set_left_rule(self, color: str, fraction: float):
        self._rule = (color, max(0.0, min(1.0, fraction)))
        self._redraw()

    def set_edge_glow(self, colors: Optional[tuple] = None):
        """Bright holographic left edge (JARVIS reply cards): a narrow
        bright core with stacked dimmer strokes trailing inward — the
        two-stroke fake-glow treatment, so the card visibly sits inside
        the luminous world."""
        self._glow = colors or (theme.ARC_BRIGHT, theme.RAMP60,
                                theme.GLOW_UNDER)
        self._redraw()

    def set_fill(self, fill: str):
        self._fill = fill
        self.body.configure(bg=fill)
        self._redraw()

    def _redraw(self):
        w = self.winfo_width()
        h = self.body.winfo_reqheight() + 2 * self._pad
        if w <= 2 or h <= 2:
            return
        self._last = (w, h)
        self.delete("chrome")
        cut = theme.CHAMFER
        # Film rule: never a full bright border (the neon-box tell). The
        # panel outline is a dim hairline; brightness lives only in the
        # two corner-bracket accents.
        item = chamfer_rect(self, 1, 1, w - 2, h - 2, cut,
                            fill=self._fill, outline=theme.RAMP33, width=1,
                            tags="chrome")
        self.tag_lower(item)
        # translucent-glass read: 1px lighter inner top-edge highlight —
        # the catch-light that makes the panel a lit glass slab
        self.create_line(1 + cut, 2, w - 2 - cut, 2,
                         fill=theme.GLASS_EDGE, width=1, tags="chrome")
        # bracket accents: trace border + chamfer at TL and BR only, with
        # the two-stroke underlay glow (wide dim beneath narrow bright) so
        # the corners read as lit, not drawn
        alen = px(14)
        aw = max(1, px(1))
        tl = (1, 1 + cut + alen, 1, 1 + cut, 1 + cut, 1, 1 + cut + alen, 1)
        br = (w - 2, h - 2 - cut - alen, w - 2, h - 2 - cut,
              w - 2 - cut, h - 2, w - 2 - cut - alen, h - 2)
        for pts in (tl, br):
            self.create_line(*pts, fill=theme.GLOW_UNDER, width=px(3),
                             tags="chrome")
            self.create_line(*pts, fill=theme.ARC_BRIGHT, width=aw,
                             tags="chrome")
        if self._glow:
            for i, color in enumerate(self._glow):
                x = 3 + i * 2
                self.create_line(x, 1 + cut, x, h - 2 - cut,
                                 fill=color, width=2, tags="chrome")
        if self._rule:
            color, frac = self._rule
            rule_h = int((h - 2 * self._pad) * frac)
            if rule_h > 2:
                y0 = (h - rule_h) // 2
                self.create_rectangle(0, y0, max(2, px(2)), y0 + rule_h,
                                      fill=color, outline="", tags="chrome")


# ------------------------------------------------------------- RoundButton
class RoundButton(tk.Canvas):
    """Flat rounded button with hover/active states and visible focus.

    kind: "default" (RAISED fill, INK text), "accent" (CYAN_SOFT fill, CYAN
    text), "ghost" (transparent, MUTED text). Pass text or a glyph char.
    """

    _KINDS = {
        "default": dict(fill=theme.RAISED, hover=theme.LINE,
                        active=theme.CYAN_SOFT, fg=theme.INK),
        "accent": dict(fill=theme.CYAN_SOFT, hover=theme.CYAN_SOFT,
                       active=theme.CYAN_SOFT, fg=theme.CYAN),
        "ghost": dict(fill="", hover=theme.RAISED,
                      active=theme.CYAN_SOFT, fg=theme.MUTED),
    }
    BARGRAD_SLICE = False      # redraws with delete("all"); small + on the
                               # flat part of the bar ground anyway

    def __init__(self, parent, text="", command: Optional[Callable] = None,
                 kind="default", size=theme.SIZE_LABEL, weight="normal",
                 width=None, height=None, pad_x=14, pad_y=7, bg=None):
        bg = bg or parent.cget("bg")
        self._font = ui_font(size, weight)
        pad_x, pad_y = px(pad_x), px(pad_y)      # callers pass design units
        try:
            tw = tkfont.Font(font=self._font).measure(text)
            th = tkfont.Font(font=self._font).metrics("linespace")
        except Exception:
            tw, th = px(8) * len(text), px(size + 6)
        self._btn_w = (px(width) if width else 0) or (tw + 2 * pad_x)
        self._btn_h = (px(height) if height else 0) or (th + 2 * pad_y)
        super().__init__(parent, width=self._btn_w, height=self._btn_h, bg=bg,
                         highlightthickness=1, bd=0, takefocus=1, cursor="hand2")
        _focus_ring(self, bg)
        self._spec = dict(self._KINDS.get(kind, self._KINDS["default"]))
        self._text = text
        self.command = command
        self._state = "normal"
        self._hovered = False
        self._pressed = False
        self._draw()
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Return>", lambda e: self.invoke())
        self.bind("<space>", lambda e: self.invoke())
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        key = (event.width, event.height)
        if getattr(self, "_cfg_last", None) == key:
            return
        self._cfg_last = key
        self._draw()

    def set_text(self, text: str):
        self._text = text
        self._draw()

    def set_enabled(self, enabled: bool):
        self._state = "normal" if enabled else "disabled"
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def invoke(self):
        if self._state != "normal":
            return
        if self.command:
            self.command()

    def _hover(self, on):
        self._hovered = on
        self._draw()

    def _press(self, _e):
        if self._state != "normal":
            return
        self._pressed = True
        self.focus_set()
        self._draw()

    def _release(self, e):
        was = self._pressed
        self._pressed = False
        self._draw()
        if was and 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height():
            self.invoke()

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(), self._btn_w)
        h = max(self.winfo_height(), self._btn_h)
        if self._state == "disabled":
            fill, fg = theme.RAISED, theme.FAINT
        elif self._pressed:
            fill, fg = self._spec["active"], self._spec["fg"]
        elif self._hovered:
            fill, fg = self._spec["hover"], self._spec["fg"]
        else:
            fill, fg = self._spec["fill"], self._spec["fg"]
        if fill:
            inset = max(1, px(1))
            round_rect(self, inset, inset, w - inset - 1, h - inset - 1,
                       theme.RADIUS, fill=fill, outline="")
        self.create_text(w // 2, h // 2, text=self._text, fill=fg,
                         font=self._font)


# ------------------------------------------------------------------ Toggle
class Toggle(tk.Canvas):
    """Animated on/off switch (150ms slide). Replaces checkbuttons.

    API: get(), set(value, animate=True), .command called with new bool on
    user interaction. Keyboard: space/Return toggles; focus ring visible.
    """

    W, H = 40, 22                     # design units; instances scale by S
    STEPS, STEP_MS = 6, 25            # 6 * 25ms = 150ms animation

    def __init__(self, parent, value=False, command: Optional[Callable] = None,
                 bg=None):
        bg = bg or parent.cget("bg")
        self.W, self.H = px(type(self).W), px(type(self).H)
        super().__init__(parent, width=self.W, height=self.H, bg=bg,
                         highlightthickness=1, bd=0, takefocus=1,
                         cursor="hand2")
        _focus_ring(self, bg)
        self.command = command
        self._value = bool(value)
        self._pos = 1.0 if value else 0.0     # knob position 0..1
        self._anim = None
        self._draw()
        self.bind("<ButtonRelease-1>", lambda e: self._toggle())
        self.bind("<space>", lambda e: self._toggle())
        self.bind("<Return>", lambda e: self._toggle())

    def get(self) -> bool:
        return self._value

    def set(self, value: bool, animate: bool = True):
        value = bool(value)
        if value == self._value and (self._pos in (0.0, 1.0)):
            return
        self._value = value
        if animate:
            self._animate()
        else:
            self._pos = 1.0 if value else 0.0
            self._draw()

    def _toggle(self):
        self.focus_set()
        self.set(not self._value)
        if self.command:
            try:
                self.command(self._value)
            except Exception:
                log.exception("toggle command failed")

    def _animate(self):
        if self._anim:
            try:
                self.after_cancel(self._anim)
            except Exception:
                pass
            self._anim = None
        target = 1.0 if self._value else 0.0
        step = (target - self._pos) / self.STEPS

        def tick(n=self.STEPS):
            self._pos = target if n <= 1 else self._pos + step
            self._draw()
            if n > 1 and self.winfo_exists():
                self._anim = self.after(self.STEP_MS, tick, n - 1)
            else:
                self._anim = None
        tick()

    def _draw(self):
        self.delete("all")
        on_f = self._pos
        track = theme.CYAN_SOFT if on_f > 0.5 else theme.LINE
        knob = theme.CYAN if on_f > 0.5 else theme.MUTED
        round_rect(self, px(1), px(3), self.W - px(2), self.H - px(4),
                   radius=(self.H - px(7)) / 2, fill=track, outline="")
        r = (self.H - px(10)) / 2
        cx = px(4) + r + on_f * (self.W - 2 * (px(4) + r))
        cy = self.H / 2 - 0.5
        self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=knob, outline="")


# ------------------------------------------------------------------- Meter
class Meter(tk.Canvas):
    """Thin horizontal level bar. set(value 0..1)."""

    def __init__(self, parent, width=120, height=4, color=theme.CYAN, bg=None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, width=px(width), height=px(height), bg=bg,
                         highlightthickness=0, bd=0)
        self._color = color
        self._value = 0.0
        self.bind("<Configure>", self._on_configure)
        self._draw()

    def _on_configure(self, event):
        key = (event.width, event.height)
        if getattr(self, "_cfg_last", None) == key:
            return
        self._cfg_last = key
        self._draw()

    def set(self, value: float, color: Optional[str] = None):
        self._value = max(0.0, min(1.0, float(value)))
        if color:
            self._color = color
        self._draw()

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 2)
        h = max(self.winfo_height(), 2)
        self.create_rectangle(0, 0, w, h, fill=theme.LINE, outline="")
        fw = int(w * self._value)
        if fw > 0:
            self.create_rectangle(0, 0, fw, h, fill=self._color, outline="")


# -------------------------------------------------------------------- Chip
class Chip(tk.Canvas):
    """Small RAISED pill with caption text (e.g. the model chip)."""

    def __init__(self, parent, text="", fg=theme.FAINT, fill=theme.RAISED,
                 size=theme.SIZE_CAPTION, mono=False, bg=None):
        bg = bg or parent.cget("bg")
        self._font = ui_mono(size) if mono else ui_display(size)
        self._fg = fg
        self._fill = fill
        self._text = text
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0)
        self._draw()

    def set_text(self, text: str, fg: Optional[str] = None):
        self._text = text
        if fg:
            self._fg = fg
        self._draw()

    def _draw(self):
        self.delete("all")
        try:
            tw = tkfont.Font(font=self._font).measure(self._text)
            th = tkfont.Font(font=self._font).metrics("linespace")
        except Exception:
            tw, th = px(7) * len(self._text), px(14)
        w, h = tw + px(18), th + px(6)
        self.configure(width=w, height=h)
        round_rect(self, 0, 0, w - 1, h - 1, radius=(h - 1) / 2,
                   fill=self._fill, outline="")
        self.create_text(w // 2, h // 2, text=self._text, fill=self._fg,
                         font=self._font)


# --------------------------------------------------------------- StatePill
class StatePill(tk.Canvas):
    """The single app-state site: a chamfered RAISED slab holding a state
    dot + the state word (display SIZE_LABEL semibold). The slab hugs the
    CURRENT word (re-measured on every set_state) — packed on the right,
    it grows leftward into the empty header centre, so the gear beside it
    never moves; words are never ellipsized.
    set_state(word, dot_color, word_color)."""

    WORDS = ("READY", "LISTENING…", "THINKING…", "SPEAKING", "ERROR")
    HEIGHT, PAD_X, DOT, GAP = 26, 12, 8, 6     # design units

    def __init__(self, parent, bg=None):
        bg = bg or parent.cget("bg")
        self._font = ui_display(theme.SIZE_LABEL, "semibold")
        # (never `self._w` / `self._h`: tkinter's Misc uses _w for the Tcl
        # path name)
        self._pill_h = px(self.HEIGHT)
        self._pill_w = 0
        self._dot_color = theme.CYAN_DIM
        self._word_text = self.WORDS[0]
        self._word_color = theme.FOCAL
        super().__init__(parent, width=1, height=self._pill_h, bg=bg,
                         highlightthickness=0, bd=0)
        self._dot = None
        self._word = None
        self._fit(self._word_text)

    def _measure(self, word: str) -> int:
        try:
            return measure(self._font, word)
        except Exception:
            return px(8) * len(word)

    def _fit(self, word: str):
        """Resize the slab to `word` and rebuild its chrome (the canvas
        width changes, so the chamfer/catch-light must be redrawn)."""
        pill_h = self._pill_h
        pill_w = (self._measure(word) + 2 * px(self.PAD_X) + px(self.DOT)
                  + px(self.GAP))
        if pill_w == self._pill_w and self._word is not None:
            return
        self._pill_w = pill_w
        self.delete("pill")        # keeps the bar-gradient ground slice
        self.configure(width=pill_w)
        chamfer_rect(self, 0, 0, pill_w - 1, pill_h - 1, cut=px(6),
                     fill=theme.RAISED, outline="", tags=("pill",))
        self.create_line(px(6) + 1, 1, pill_w - px(6) - 2, 1,
                         fill=theme.GLASS_EDGE, width=1, tags=("pill",))
        r = px(self.DOT) / 2.0
        dx, dy = px(self.PAD_X) + r, pill_h / 2.0
        self._dot = self.create_oval(dx - r, dy - r, dx + r, dy + r,
                                     fill=self._dot_color, outline="",
                                     tags=("pill",))
        self._word = self.create_text(
            px(self.PAD_X) + px(self.DOT) + px(self.GAP), pill_h // 2,
            anchor="w", text=self._word_text, fill=self._word_color,
            font=self._font, tags=("pill",))

    def set_state(self, word: str, dot_color: str, word_color: str):
        self._word_text, self._dot_color = word, dot_color
        self._word_color = word_color
        self._fit(word)
        self.itemconfigure(self._word, text=word, fill=word_color)
        self.itemconfigure(self._dot, fill=dot_color)

    def set_dot(self, color: str):
        self._dot_color = color
        self.itemconfigure(self._dot, fill=color)


# ----------------------------------------------------------------- Tooltip
class Tooltip:
    """Hover tooltip. Timing/behavior ported from voice_input_gui.py 968-1009:
    400ms show delay, positioned at widget rootx+20 / below widget +4,
    auto-hide after 8s, hidden on Leave and ButtonPress. New theme colors.
    """

    def __init__(self, widget, text, tag: Optional[str] = None):
        """`tag`: bind to a canvas item tag on `widget` (a Canvas) instead
        of the whole widget — hover tooltips for canvas-drawn controls."""
        self.widget = widget
        self.text = text
        self._tw = None
        self._after_id = None
        if tag is not None:
            widget.tag_bind(tag, "<Enter>", self._schedule_show, add=True)
            widget.tag_bind(tag, "<Leave>", self._hide, add=True)
            widget.tag_bind(tag, "<ButtonPress>", self._hide, add=True)
        else:
            widget.bind("<Enter>", self._schedule_show, add=True)
            widget.bind("<Leave>", self._hide, add=True)
            widget.bind("<ButtonPress>", self._hide, add=True)

    def set_text(self, text: str):
        self.text = text

    def _schedule_show(self, event=None):
        self._hide()  # Kill any existing tooltip first
        self._after_id = self.widget.after(400, self._show)

    def _show(self, event=None):
        self._after_id = None
        if not self.widget.winfo_exists() or not self.text:
            return
        x = self.widget.winfo_rootx() + px(20)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + px(4)
        self._tw = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        try:
            tw.attributes("-topmost", True)
        except tk.TclError:
            log.debug("tooltip: -topmost unsupported")
        tw.configure(bg=theme.LINE)          # 1px border via padding
        label = tk.Label(
            tw, text=self.text, font=ui_font(theme.SIZE_CAPTION),
            bg=theme.RAISED, fg=theme.INK,
            padx=px(8), pady=px(4), wraplength=px(320), justify="left",
        )
        label.pack(padx=max(1, px(1)), pady=max(1, px(1)))
        # Auto-hide after 8 seconds as safety net
        self.widget.after(8000, self._hide)

    def _hide(self, event=None):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                log.debug("tooltip: after_cancel failed", exc_info=True)
            self._after_id = None
        if self._tw:
            try:
                self._tw.destroy()
            except tk.TclError:
                pass
            self._tw = None


# ------------------------------------------------------------------- Toast
class Toast:
    """Transient bottom-center notice on a container (usually the root)."""

    _KIND_FG = {"ok": theme.OK, "info": theme.INK,
                "warn": theme.WARN, "error": theme.ERR}

    def __init__(self, container):
        self.container = container
        self._frame = None
        self._after_id = None

    def show(self, text: str, kind: str = "info", ms: int = 1800):
        self.hide()
        fg = self._KIND_FG.get(kind, theme.INK)
        self._frame = tk.Frame(self.container, bg=theme.LINE)
        tk.Label(self._frame, text=text, font=ui_font(theme.SIZE_LABEL),
                 bg=theme.RAISED, fg=fg, padx=px(14), pady=px(7)).pack(
            padx=max(1, px(1)), pady=max(1, px(1)))
        self._frame.place(relx=0.5, rely=1.0, y=-px(104), anchor="s")
        self._frame.lift()
        self._after_id = self.container.after(ms, self.hide)

    def hide(self):
        if self._after_id:
            try:
                self.container.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._frame:
            try:
                self._frame.destroy()
            except tk.TclError:
                pass
            self._frame = None
