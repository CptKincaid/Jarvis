"""The carry chip: what his closed hand is holding, in the console header.

The safety lane's ruling: a GRAB plays a 115 ms tone and shows a chip that
names what it picked up -- the tone is the instant, the chip is the WHAT.
He is looking at the screen when he does this (he described reaching AT
the screen), so a small header indicator is guaranteed to be seen and costs
nothing when it is wrong. It sits left of the sensing badge
(jarvis/ui/sensing_badge.py is the precedent: a header chip, not a
transcript card -- a transient carry is not conversation).

STATES, and what each looks like:

  holding   a closed-hand glyph, the subject's name tail-truncated at 28
            characters, and a 2 px rule under it that DEPLETES over the
            carry's cap -- ``CastGesture.carry_cap_s()``, 4.1 s at the
            7.5 fps the camera delivers, never the 8 s wall-clock backstop
            alone (MEASURED: drawn over 8 s the rule was half gone when the
            frame cap put the thing down) -- so he can see it is about to
            be put down. The backstop is the chip's own timer: when no
            frame has arrived to end the carry by ``backstop_s``, the chip
            puts it down itself through ``on_dismiss``.
  thrown    the name with an arrow in the fling's direction, for 900 ms
  landed    "-> BOARD" / "-> HPCOMPUTER" in the ok colour, then fades
  held      the target in the warning colour: it did not get there, and
            Jarvis has said why; the payload is on the board. A veto, a
            refusal and an empty hand are NOT this state -- nothing was
            kept anywhere, so they read ``dropped``
  dropped   the rule goes muted and the word becomes "dropped" for 900 ms,
            then the chip goes. A TIMEOUT LOOKS IDENTICAL TO A DELIBERATE
            DROP, because it is the same outcome.

Click = put it down (the free fifth cancel). Every decision that a
screenshot review would argue about is a pure function here (``chip_word``,
``chip_tone``, ``chip_colors``, ``rule_fraction``), so
tests/test_carry_chip.py asserts them without a window; the widget itself
is thin and is never constructed in the suite.

Every colour is read from ``theme`` at call time, the trap named at the top
of jarvis/ui/theme.py.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.widgets import chamfer_rect, measure, px, ui_display

log = get_logger("ui.carry")

STATE_IDLE = "idle"
STATE_HOLDING = "holding"
STATE_THROWN = "thrown"
STATE_LANDED = "landed"
STATE_HELD = "held"
STATE_DROPPED = "dropped"

MAX_CHARS = 28            # the way views.py tail-truncates a partial
FLASH_MS = 900            # thrown / dropped / landed / held linger
TICK_MS = 100             # the depleting rule's repaint cadence
GLYPH_HOLD = "◉"
ARROWS = {"left": "←", "right": "→", "up": "↑", "down": "↓"}


def chip_name(name: str, max_chars: int = MAX_CHARS) -> str:
    """The subject's spoken name, tail-truncated with an ellipsis."""
    text = " ".join(str(name or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max(1, max_chars - 1)].rstrip() + "…"


def chip_word(state: str, name: str = "", direction: str = "",
              target: str = "") -> str:
    """The chip's text for a state. Pure."""
    short = chip_name(name)
    if state == STATE_HOLDING:
        return "%s %s" % (GLYPH_HOLD, short) if short else GLYPH_HOLD
    if state == STATE_THROWN:
        arrow = ARROWS.get(str(direction or "").lower(), "→")
        return "%s %s" % (short, arrow) if short else arrow
    if state == STATE_LANDED:
        return "→ %s" % str(target or "board").upper()
    if state == STATE_HELD:
        return "held · %s" % str(target or "").upper() if target else "held"
    if state == STATE_DROPPED:
        return "dropped"
    return ""


def chip_tone(state: str) -> str:
    """holding -> live, thrown/landed -> ok, held -> warn, dropped -> muted."""
    if state == STATE_HOLDING:
        return "live"
    if state in (STATE_THROWN, STATE_LANDED):
        return "ok"
    if state == STATE_HELD:
        return "warn"
    return "muted"


def chip_colors(tone: str) -> dict:
    """ink / edge / rule for a tone, in the CURRENT look."""
    ok = getattr(theme, "OK", theme.CYAN)
    muted = getattr(theme, "MUTED", theme.INK)
    if tone == "live":
        return {"ink": theme.FOCAL, "edge": theme.CYAN, "rule": theme.CYAN_DIM}
    if tone == "ok":
        return {"ink": theme.FOCAL, "edge": ok, "rule": ok}
    if tone == "warn":
        return {"ink": theme.INK, "edge": theme.WARN, "rule": theme.WARN}
    return {"ink": muted, "edge": theme.GLASS_EDGE, "rule": muted}


def rule_fraction(now: float, started: float, ttl_s: float) -> float:
    """How much of the carry's TTL is left, 1.0 -> 0.0, clamped."""
    if ttl_s <= 0.0:
        return 0.0
    left = 1.0 - (float(now) - float(started)) / float(ttl_s)
    return max(0.0, min(1.0, left))


def backstop_due(now: float, started: float, backstop_s: float) -> bool:
    """Has the carry outlived its wall-clock cap with no frame to end it?
    0 (or less) means no backstop was asked for, never "due at once"."""
    if backstop_s <= 0.0:
        return False
    return (float(now) - float(started)) > float(backstop_s)


class CarryChip(tk.Canvas):
    """The header chip. Hidden (never packed) until a grab; shown left of
    the sensing badge; goes away on its own after a throw or a drop.

    ``on_dismiss`` is the click -- it puts the carry down through the
    courier, which then tells this chip so via ``dropped``.
    """

    HEIGHT, PAD_X, RULE = 26, 10, 2       # design units, == SensingBadge

    def __init__(self, parent, bg=None,
                 on_dismiss: Optional[Callable[[], None]] = None,
                 now: Optional[Callable[[], float]] = None):
        bg = bg or parent.cget("bg")
        self._font = ui_display(theme.SIZE_CAPTION, "semibold")
        self._pill_h = px(self.HEIGHT)
        self._state = STATE_IDLE
        self._text = ""
        self._started = 0.0
        self._ttl = 0.0
        self._backstop = 0.0
        self._shown = False
        self._after_id = None
        self._on_dismiss = on_dismiss
        import time as _time
        self._now = now or _time.monotonic
        super().__init__(parent, width=1, height=self._pill_h, bg=bg,
                         highlightthickness=0, bd=0)
        self.bind("<ButtonPress-1>", self._clicked, add=True)

    # ----------------------------------------------------------- reads
    @property
    def state(self) -> str:
        return self._state

    @property
    def shown(self) -> bool:
        return self._shown

    # --------------------------------------------------------- the API
    def hold(self, name: str, ttl_s: float = 8.0,
             backstop_s: Optional[float] = None) -> None:
        self._state = STATE_HOLDING
        self._text = chip_word(STATE_HOLDING, name)
        self._started = self._now()
        self._ttl = float(ttl_s)
        self._backstop = float(backstop_s) if backstop_s else 0.0
        self._show()
        self._fit()
        self._schedule(TICK_MS, self._tick)

    def thrown(self, direction: str = "") -> None:
        name = self._text[2:] if self._text.startswith(GLYPH_HOLD) else ""
        self._flash(STATE_THROWN, chip_word(STATE_THROWN, name, direction))

    def landed(self, target: str = "") -> None:
        self._flash(STATE_LANDED, chip_word(STATE_LANDED, target=target))

    def held(self, target: str = "") -> None:
        self._flash(STATE_HELD, chip_word(STATE_HELD, target=target),
                    linger_ms=FLASH_MS * 3)

    def dropped(self, toward: str = "") -> None:
        self._flash(STATE_DROPPED, chip_word(STATE_DROPPED))

    def clear(self) -> None:
        self._cancel_timer()
        self._state = STATE_IDLE
        self._text = ""
        self._hide()

    # ------------------------------------------------------------ guts
    def _flash(self, state: str, text: str, linger_ms: int = FLASH_MS) -> None:
        self._cancel_timer()
        self._state = state
        self._text = text
        self._ttl = 0.0
        self._show()
        self._fit()
        self._schedule(linger_ms, self.clear)

    def _show(self) -> None:
        if self._shown:
            return
        self._shown = True
        try:
            self.pack(side="right", padx=(0, theme.PAD_S))
        except tk.TclError:
            log.debug("carry chip pack on a dead window", exc_info=True)

    def _hide(self) -> None:
        if not self._shown:
            return
        self._shown = False
        try:
            self.pack_forget()
        except tk.TclError:
            log.debug("carry chip unpack on a dead window", exc_info=True)

    def _schedule(self, ms: int, fn) -> None:
        self._cancel_timer()
        try:
            self._after_id = self.after(ms, fn)
        except tk.TclError:
            self._after_id = None

    def _cancel_timer(self) -> None:
        aid, self._after_id = self._after_id, None
        if aid is not None:
            try:
                self.after_cancel(aid)
            except tk.TclError:
                pass

    def _tick(self) -> None:
        if self._state != STATE_HOLDING:
            return
        if backstop_due(self._now(), self._started, self._backstop):
            # No frame came to end it. The wall-clock cap, from here.
            self._clicked()
            return
        self._draw_rule()
        self._schedule(TICK_MS, self._tick)

    def _clicked(self, _event=None) -> None:
        if self._state != STATE_HOLDING:
            return
        fn = self._on_dismiss
        if callable(fn):
            try:
                fn()
            except Exception:                    # noqa: BLE001 - callback boundary
                log.exception("carry chip dismiss failed")

    def _measure(self, word: str) -> int:
        try:
            return measure(self._font, word)
        except Exception:                        # noqa: BLE001 - no metrics
            return px(7) * len(word)

    def _fit(self) -> None:
        colors = chip_colors(chip_tone(self._state))
        pill_h = self._pill_h
        pill_w = self._measure(self._text) + 2 * px(self.PAD_X)
        self.delete("chip")
        try:
            self.configure(width=pill_w)
        except tk.TclError:
            return
        if theme.LOOK == "holo":
            chamfer_rect(self, 0, 0, pill_w - 1, pill_h - 1, cut=px(6),
                         fill="", outline=colors["edge"], width=1,
                         tags=("chip",))
        else:
            chamfer_rect(self, 0, 0, pill_w - 1, pill_h - 1, cut=px(6),
                         fill=theme.RAISED, outline="", tags=("chip",))
        self.create_text(px(self.PAD_X), pill_h // 2, anchor="w",
                         text=self._text, fill=colors["ink"],
                         font=self._font, tags=("chip",))
        self._draw_rule()

    def _draw_rule(self) -> None:
        self.delete("rule")
        if self._state != STATE_HOLDING or self._ttl <= 0.0:
            return
        colors = chip_colors(chip_tone(self._state))
        frac = rule_fraction(self._now(), self._started, self._ttl)
        try:
            width = int(self.cget("width"))
        except tk.TclError:
            return
        x0 = px(6)
        x1 = x0 + int((width - 2 * px(6)) * frac)
        y = self._pill_h - px(self.RULE) - 1
        if x1 > x0:
            self.create_line(x0, y, x1, y, fill=colors["rule"],
                             width=max(1, px(self.RULE)), tags=("rule",))


__all__ = [
    "ARROWS", "CarryChip", "FLASH_MS", "GLYPH_HOLD", "MAX_CHARS",
    "STATE_DROPPED", "STATE_HELD", "STATE_HOLDING", "STATE_IDLE",
    "STATE_LANDED", "STATE_THROWN", "backstop_due", "chip_colors",
    "chip_name", "chip_tone", "chip_word", "rule_fraction",
]
