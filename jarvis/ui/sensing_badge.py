"""The console's sensing indicator -- "am I being watched", at a glance.

Offline mode (jarvis/sensing.py) is a privacy control, and a privacy
control that is only audible is half a control: he has to be able to LOOK
at the console and know. So the badge sits in the header beside the state
pill, and it is present in every state -- there is no "nothing to show"
case, because the absence of a badge and a badge saying SENSING would be
indistinguishable from across the room.

THREE STATES, TOLD APART THREE WAYS. The word carries it in text
("SENSING" / "CAMERA OFF" / "OFFLINE"), the dot carries it in colour
(cyan live, amber restricted), and OFFLINE additionally carries it in
SHAPE -- a hollow dot, i.e. nothing lit. Colour alone would not survive a
dimmed monitor or a colour-blind glance, and this is the one readout in
the app where being wrong is not a cosmetic bug.

Everything a screenshot review would argue about is a pure function here
(``badge_tone`` / ``badge_word`` / ``badge_colors`` / ``badge_caption``),
so tests/test_sensing_badge.py asserts legibility in BOTH looks without
creating a window -- the house pattern, and the standing rule since the
2026-08-26 desktop freeze came out of UI window churn.

Every colour is read from ``theme`` at CALL time: ``select_look`` re-derives
the tokens at runtime, so a captured value would freeze the import-time
look (the trap named at the top of jarvis/ui/theme.py).
"""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from typing import Any

from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.widgets import chamfer_rect, measure, px, ui_display

log = get_logger("ui.sensing")

TONE_ON = "on"
TONE_CURFEW = "curfew"
TONE_OFF = "off"

WORDS = {TONE_ON: "SENSING", TONE_CURFEW: "CAMERA OFF", TONE_OFF: "OFFLINE"}


def normalise(obj: Any) -> dict:
    """A SensingState, a SensingChanged event or a status dict -> one dict.

    Three shapes reach the badge (the policy directly on the 5 s pass, the
    bus event on a spoken switch, and a plain dict from a diagnostic), and
    a badge that understood only one of them would silently stop updating
    down the path nobody tested.
    """
    if isinstance(obj, dict):
        data = obj
        get = data.get
    else:
        def get(key, default=None):
            return getattr(obj, key, default)
    return {"camera": bool(get("camera", True)),
            "radar": bool(get("radar", True)),
            "offline": bool(get("offline", False)),
            "reason": str(get("reason", "") or ""),
            "until": get("until", None),
            "curfew": get("curfew", None),
            "persisted": bool(get("persisted", True))}


def badge_tone(state: Any) -> str:
    s = normalise(state)
    if s["offline"] or not (s["camera"] or s["radar"]):
        return TONE_OFF
    if not s["camera"]:
        return TONE_CURFEW
    return TONE_ON


def badge_word(state: Any) -> str:
    return WORDS[badge_tone(state)]


def badge_colors(tone: str) -> dict:
    """dot / ink / edge for a tone, in the CURRENT look.

    The ink is always a light text token (FOCAL or INK) because both looks
    ground on a dark blue: cyan here would be structure colour used as
    text, which the film budget reserves for chrome and which loses against
    classic's lifted ground.
    """
    if tone == TONE_OFF:
        return {"dot": theme.WARN, "ink": theme.FOCAL, "edge": theme.WARN,
                "filled": False}
    if tone == TONE_CURFEW:
        return {"dot": theme.WARN, "ink": theme.INK, "edge": theme.WARN,
                "filled": True}
    return {"dot": theme.CYAN, "ink": theme.FOCAL, "edge": theme.GLASS_EDGE,
            "filled": True}


def _clock_words(hm) -> str:
    from jarvis.quiet import fmt_clock
    return fmt_clock(int(hm[0]), int(hm[1]))


def badge_caption(state: Any) -> str:
    """The second line -- the tooltip, and the status strip's detail. Says
    WHY and, when there is one, until when."""
    s = normalise(state)
    parts = []
    if s["reason"] == "failsafe":
        parts.append("no saved state; off until you say otherwise")
    elif s["reason"] == "timed" and s["until"]:
        end = datetime.fromtimestamp(float(s["until"]))
        parts.append("camera + radar off until %s"
                     % _clock_words((end.hour, end.minute)))
    elif s["offline"]:
        parts.append("camera + radar off until you say otherwise")
    elif s["reason"] == "curfew" and s["curfew"]:
        parts.append("camera off until %s (curfew)" % _clock_words(s["curfew"][1]))
    elif s["curfew"]:
        parts.append("camera + radar on; curfew at %s"
                     % _clock_words(s["curfew"][0]))
    else:
        parts.append("camera + radar on")
    if not s["persisted"]:
        parts.append("not saved — it won't hold across a restart")
    return " · ".join(parts)


class SensingBadge(tk.Canvas):
    """The header chip. Mirrors StatePill's chrome exactly (outlined capsule
    in holo, filled slab in classic) so it cannot be the one widget in the
    header that renders wrong in one of the looks."""

    HEIGHT, PAD_X, DOT, GAP = 26, 10, 8, 6      # design units, == StatePill

    def __init__(self, parent, bg=None):
        bg = bg or parent.cget("bg")
        self._font = ui_display(theme.SIZE_CAPTION, "semibold")
        self._pill_h = px(self.HEIGHT)
        self._pill_w = 0
        self._tone = TONE_ON
        self._word_text = WORDS[TONE_ON]
        super().__init__(parent, width=1, height=self._pill_h, bg=bg,
                         highlightthickness=0, bd=0)
        self._dot = None
        self._word = None
        self.caption = ""
        self._fit()

    def _measure(self, word: str) -> int:
        try:
            return measure(self._font, word)
        except Exception:  # noqa: BLE001 - no font metrics without a root
            return px(7) * len(word)

    def _fit(self):
        colors = badge_colors(self._tone)
        pill_h = self._pill_h
        pill_w = (self._measure(self._word_text) + 2 * px(self.PAD_X)
                  + px(self.DOT) + px(self.GAP))
        self._pill_w = pill_w
        self.delete("badge")
        self.configure(width=pill_w)
        if theme.LOOK == "holo":
            chamfer_rect(self, 0, 0, pill_w - 1, pill_h - 1, cut=px(6),
                         fill="", outline=colors["edge"], width=1,
                         tags=("badge",))
        else:
            chamfer_rect(self, 0, 0, pill_w - 1, pill_h - 1, cut=px(6),
                         fill=theme.RAISED, outline="", tags=("badge",))
            self.create_line(px(6) + 1, 1, pill_w - px(6) - 2, 1,
                             fill=colors["edge"], width=1, tags=("badge",))
        r = px(self.DOT) / 2.0
        dx, dy = px(self.PAD_X) + r, pill_h / 2.0
        self._dot = self.create_oval(
            dx - r, dy - r, dx + r, dy + r,
            fill=colors["dot"] if colors["filled"] else "",
            outline="" if colors["filled"] else colors["dot"],
            width=max(1, px(1)), tags=("badge",))
        self._word = self.create_text(
            px(self.PAD_X) + px(self.DOT) + px(self.GAP), pill_h // 2,
            anchor="w", text=self._word_text, fill=colors["ink"],
            font=self._font, tags=("badge",))

    def set_state(self, state: Any) -> None:
        """Show `state` (a SensingState, a SensingChanged or a status dict)."""
        tone, word = badge_tone(state), badge_word(state)
        self.caption = badge_caption(state)
        if tone == self._tone and word == self._word_text:
            return
        self._tone, self._word_text = tone, word
        self._fit()
