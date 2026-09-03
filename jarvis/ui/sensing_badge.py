"""The console's sensing indicator -- "am I being watched", at a glance.

Offline mode (jarvis/sensing.py) is a privacy control, and a privacy
control that is only audible is half a control: he has to be able to LOOK
at the console and know. So the badge sits in the header beside the state
pill, and it is present in every state -- there is no "nothing to show"
case, because the absence of a badge and a badge saying SENSING would be
indistinguishable from across the room.

THREE STATES, TOLD APART THREE WAYS. The word carries it in text
("SENSING" / "CAM OFF" / "OFFLINE"), the dot carries it in colour
(cyan live, amber restricted), and the dot carries it AGAIN in SHAPE:
a full disc when everything is lit, a HALF disc under the curfew (half
the sensors lit -- the radar is on, the camera is not), a hollow ring
when nothing is. Colour alone would not survive a dimmed monitor or a
colour-blind glance; and CAM OFF and OFFLINE are both amber and both
seven glyphs, so without the shape the only thing telling them apart
across the room was the word. This is the one readout in the app where
being wrong is not a cosmetic bug.

WHAT 2026-09-03 TOOK, AND WHAT IT REFUSED TO TAKE. The header ran out of
room (tests/test_header_fit.py: 312 px for this chip and the state pill
together, 485 asked) and he ruled the wordmark out of paying for it. So
this chip gave up chip padding (PAD_X 10 -> 4, GAP 6 -> 4, measured on
Xvfb as the geometry at which the pill's full words still fit at every
scale) and one word -- "CAMERA OFF" -> "CAM OFF", seven glyphs like the
other two -- and NOTHING else: same SIZE_CAPTION face, same dot, same
capsule, all three tell-apart channels intact, so what a glance from
across the room sees is unchanged. The unabbreviated reason ("camera off
until 7 am (curfew)") stays on ``badge_caption``, which MainWindow feeds
the badge's tooltip on every refresh. The pill paid the rest of the
bill; its state is also on the reactor, the transcript and its own dot,
and this one is not.

The two chips now share a face and a geometry, so which one is the
privacy readout is told by the CHROME -- and honestly about how much each
channel carries (contrasts computed from the theme tokens; the pixels
confirmed off a private Xvfb at S=1 and S=2, 2026-09-03):

  * HOLO, the glance-level tell: the pill wears a BRIGHT corner tick on
    its left cut (widgets.StatePill._fit) and the badge deliberately
    does NOT -- its cut is its plain outline. BRIGHT is 9.5:1 against
    the ground; a tick drawn in any token the badge owns would sit at
    1.31:1 (CYAN) / 1.18:1 (WARN) against the pill's, i.e. two ticks a
    hue apart, which would erase the asymmetry rather than add a
    channel. So the badge stays tick-less on purpose, and
    tests/test_sensing_badge.py pins that it draws none. How much that
    buys, measured off the pixels: the pill's top-left corner (an 8 x 8
    design-unit box) averages 1.68:1 brighter than the badge's at S=2
    and 1.17:1 at S=1 -- a modest tell, and the best the tokens allow.
  * BOTH looks, the hue-level channel: the badge's edge (capsule outline
    in holo, the 1 px catch-light in classic) is never the pill's
    GLASS_EDGE -- CYAN_DIM while live, WARN while restricted. Live, that
    is a hue shift at 1.17:1 (holo) / 1.41:1 (classic) on a hairline: a
    token-level distinction a test can read, not one a glance can.
    Restricted, the amber edge is unmistakable in either look.
  * CLASSIC has no tick vocabulary (it is the frozen 08-31 console), so
    there the two chips ARE visual twins while live; the word and the
    dot carry it, as they did before 2026-09-03.

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

# Seven glyphs in every tone: the chip barely twitches between states,
# so it reads as one steady mark rather than a thing that jumps.
WORDS = {TONE_ON: "SENSING", TONE_CURFEW: "CAM OFF", TONE_OFF: "OFFLINE"}


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


def sensing_failsafe_state() -> dict:
    """What the badge shows when there is NO policy to ask.

    A header that still reads SENSING because the owner failed to
    construct is the console asserting the one thing nobody can check.
    The app hands a denying stand-in to the sensors in that case
    (jarvis/sensing.py ``DENIED``), so the honest badge is the same
    fail-safe: everything off, and the caption says why.
    """
    return {"camera": False, "radar": False, "offline": True,
            "reason": "failsafe", "until": None, "curfew": None,
            "persisted": True}


def badge_tone(state: Any) -> str:
    s = normalise(state)
    if s["offline"] or not (s["camera"] or s["radar"]):
        return TONE_OFF
    if not s["camera"]:
        return TONE_CURFEW
    return TONE_ON


def badge_word(state: Any) -> str:
    return WORDS[badge_tone(state)]


DOT_DISC = "disc"      # everything lit
DOT_HALF = "half"      # half the sensors lit (radar on, camera off)
DOT_RING = "ring"      # nothing lit


def badge_colors(tone: str) -> dict:
    """dot / ink / edge / shape for a tone, in the CURRENT look.

    The ink is always a light text token (FOCAL or INK) because both looks
    ground on a dark blue: cyan here would be structure colour used as
    text, which the film budget reserves for chrome and which loses against
    classic's lifted ground.

    The edge is never StatePill's GLASS_EDGE (CYAN_DIM is the accent
    outline the holo buttons already use, so it is vocabulary the eye has
    met). Live, that is a hue shift on a hairline -- 1.17:1 holo, 1.41:1
    classic, computed -- so it is the token-level tell, not the glance
    one; the glance tell in holo is the corner tick the pill has and this
    chip does not (module docstring). Restricted, the WARN edge reads.

    ``shape`` is the third tell-apart channel, per tone: DISC / HALF /
    RING. CAM OFF and OFFLINE are both amber and both seven glyphs, so
    the shape is what separates them without reading.
    """
    if tone == TONE_OFF:
        return {"dot": theme.WARN, "ink": theme.FOCAL, "edge": theme.WARN,
                "shape": DOT_RING}
    if tone == TONE_CURFEW:
        return {"dot": theme.WARN, "ink": theme.INK, "edge": theme.WARN,
                "shape": DOT_HALF}
    return {"dot": theme.CYAN, "ink": theme.FOCAL, "edge": theme.CYAN_DIM,
            "shape": DOT_DISC}


def _clock_words(hm) -> str:
    from jarvis.quiet import fmt_clock
    return fmt_clock(int(hm[0]), int(hm[1]))


def badge_caption(state: Any) -> str:
    """The badge's tooltip (its only consumer: MainWindow feeds it to
    Tooltip.set_text on every refresh). Says WHY and, when there is one,
    until when."""
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

    HEIGHT, PAD_X, DOT, GAP = 26, 4, 8, 4       # design units, == StatePill

    def __init__(self, parent, bg=None):
        bg = bg or parent.cget("bg")
        self._font = self.font()
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

    @classmethod
    def font(cls) -> tuple:
        """The chip face, read from theme at CALL time. Identical to
        StatePill.font(): the two header chips are one type size."""
        return ui_display(theme.SIZE_CAPTION, "semibold")

    @classmethod
    def chip_w(cls, text_w: int) -> int:
        """Capsule width around `text_w` px of word (pure). One formula
        for the drawn chip and the budgeted chip."""
        return text_w + 2 * px(cls.PAD_X) + px(cls.DOT) + px(cls.GAP)

    @classmethod
    def widest_w(cls) -> int:
        """Chip width for the WIDEST of the three words ('CAM OFF'), in
        the current look/scale.

        The header budgets around this rather than the current word: the
        badge is packed LAST in the header, so when the bar runs out of
        room the badge is the child Tk CLIPS -- and, when the cavity is
        gone entirely, the child Tk stops drawing (views.header_spans, a
        transcription of tkPack.c). That is the 2026-09-02 "the word
        sensing is underneath the ready symbol": 124 px of 168, its word
        sheared off flush against the wordmark, and 41 px of 214 with the
        pill on LISTENING…. Budgeting for the widest keeps it from coming
        back at 21:00 when the curfew turns the word into CAM OFF -- and
        an unmapped privacy badge is the failure this class exists to
        prevent, since absence and SENSING must never look the same.
        """
        font = cls.font()
        try:
            text_w = max(measure(font, word) for word in WORDS.values())
        except Exception:  # noqa: BLE001 - no font metrics without a root
            text_w = px(7) * max(len(word) for word in WORDS.values())
        return cls.chip_w(text_w)

    def _measure(self, word: str) -> int:
        try:
            return measure(self._font, word)
        except Exception:  # noqa: BLE001 - no font metrics without a root
            return px(7) * len(word)

    def _fit(self):
        colors = badge_colors(self._tone)
        pill_h = self._pill_h
        pill_w = self.chip_w(self._measure(self._word_text))
        self._pill_w = pill_w
        self.delete("badge")
        self.configure(width=pill_w)
        if theme.LOOK == "holo":
            # the outline only -- NO corner tick. The pill's bright tick
            # is what tells the two chips apart at a glance in holo, and
            # a tick here in the badge's own tokens would sit 1.2-1.3:1
            # from it (computed; module docstring). Measured, not guessed.
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
        shape = colors["shape"]
        # disc: a filled dot. ring: the outline only. half: the outline
        # plus its left half filled (a chord from 12 o'clock round to 6),
        # so the three shapes read full / half / empty at any size.
        self._dot = self.create_oval(
            dx - r, dy - r, dx + r, dy + r,
            fill=colors["dot"] if shape == DOT_DISC else "",
            outline="" if shape == DOT_DISC else colors["dot"],
            width=max(1, px(1)), tags=("badge",))
        if shape == DOT_HALF:
            self.create_arc(dx - r, dy - r, dx + r, dy + r, start=90,
                            extent=180, style="chord", fill=colors["dot"],
                            outline="", tags=("badge",))
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
