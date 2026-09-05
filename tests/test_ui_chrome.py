"""Tk-free tests for the live-screen chrome helpers of the holo look
(2026-09-01 blue-holographic overhaul, lane B): the placeholder chooser
that stops the command bar clipping its hint, the status-strip telemetry
elision that ends the wake-word / CPU collision, the tracked-caps labels,
the per-look card recipe, and the def-time → call-time token fixes in
jarvis.ui.widgets. No root is created: only pure functions, static
methods and signatures are exercised."""
import inspect
import os
import tempfile

import pytest

os.environ.setdefault("JARVIS_LOG_DIR", tempfile.mkdtemp(prefix="jarvis-ui-"))
os.environ.setdefault("JARVIS_ASSISTANT_CONFIG",
                      os.path.join(tempfile.mkdtemp(prefix="jarvis-ui-cfg-"),
                                   "assistant.json"))

from jarvis.ui import theme  # noqa: E402
from jarvis.ui.views import (TELEMETRY_LEVELS, CommandBar,  # noqa: E402
                             StatusStrip, card_look, fit_placeholder,
                             fmt_load, fmt_temps_compact, levels_for_look,
                             plan_strip, plan_telemetry, telemetry_segments,
                             tracked)
from jarvis.ui.widgets import (Card, Chip, Meter, RoundButton,  # noqa: E402
                               Toast)


@pytest.fixture(autouse=True)
def _restore_look():
    """Leave the theme in its import-time state (holo, scale 1.0)."""
    yield
    theme.apply_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)


# Measured 2026-09-01 on :99 at JARVIS_UI_SCALE 2.0, the 460-design-px
# minimum window (920 dev px): display font ('Chakra Petch', -30) for the
# placeholder, ('Chakra Petch SemiBold', -24) strip labels, ('JetBrains
# Mono', -24) strip values. Kept as literals so the tests stay Tk-free.
LONG, SHORT, PLAIN = (CommandBar.PLACEHOLDER_HOT, CommandBar.PLACEHOLDER_HOT_SHORT,
                      CommandBar.PLACEHOLDER)
PLACEHOLDER_WS = [(LONG, 631), (SHORT, 369), (PLAIN, 315)]
ENTRY_AT_920 = 648 - 100          # field width minus px(50) at S=2
WAKE_END_AT_920 = 32 + 208 + 16   # PAD + "● WAKE WORD ON" + PAD_S = 256
LEVEL0 = [("CPU", 218), ("GPU", 218), ("MEMORY", 274)]   # sum 710 → x 210
LEVEL1 = [("CPU", 134), ("GPU", 134), ("MEMORY", 162)]   # sum 430 → x 490
LEVEL2 = [("MEMORY", 162)]                               # sum 162 → x 758
LEVELS = [LEVEL0, LEVEL1, LEVEL2]

# MEASURED 2026-09-03 on Xvfb :95 at JARVIS_UI_SCALE 2.0, the same faces
# and the same method as the 09-01 pass above ('Chakra Petch SemiBold' -24
# labels, 'JetBrains Mono' -24 values): labels CPU/GPU 48, MEM 52, MEMORY
# 100; one mono character 14. A segment is PAD_S + value + PAD_S, plus
# label + 6-unit gap when it has a label, plus PAD on the last one --
# StatusStrip._layout's own formula. The strip lives in a shell packed
# padx=1, so his 920-px window gives it 918, not 920.
#
# IDLE ('cpu 53° 7% · gpu 44° 0%', '47.5/122 GB'):
#   0 full     CPU '53°C · 7%' 218 | GPU 218 | MEM '47.5/122 GB' 282 = 718
#   1 compact  CPU '53° 7%'    176 | GPU 176 |     '47.5/122 GB' 218 = 570
#   2 load     CPU '7%'        120 | GPU 120 |     '47.5/122 GB' 218 = 458
#   3 minimal                                      '47.5/122 GB' 218 = 218
STRIP_W = 918
LIVE = [[("CPU", 218), ("GPU", 218), ("MEMORY", 282)],
        [("CPU", 176), ("GPU", 176), ("MEMORY", 218)],
        [("CPU", 120), ("GPU", 120), ("MEMORY", 218)],
        [("MEMORY", 218)]]
# HOT ('cpu 100° 100% · gpu 88° 100%', '121.7/122 GB') -- a three-digit
# temperature under full load, on a pool over 100 GB in use:
#   0 full  260 | 246 | 296 = 802     2 load  148 | 148 | 232 = 528
#   1 comp  218 | 204 | 232 = 654     3 min               232 = 232
HOT = [[("CPU", 260), ("GPU", 246), ("MEMORY", 296)],
       [("CPU", 218), ("GPU", 204), ("MEMORY", 232)],
       [("CPU", 148), ("GPU", 148), ("MEMORY", 232)],
       [("MEMORY", 232)]]
WAKE_END_OFF = 32 + 16 + 12 + 192 + 16   # PAD ring gap 'WAKE WORD OFF' PAD_S
WAKE_END_CLASSIC = 32 + 219 + 16         # classic: '○ WAKE WORD OFF', no ring


# ------------------------------------------------------- placeholder
def test_fit_placeholder_picks_the_longest_that_fits():
    # today's clip: 631 px of text into a 548-px entry → the short form
    assert fit_placeholder(ENTRY_AT_920 - 12, PLACEHOLDER_WS) == SHORT
    # a wide window keeps the full sentence
    assert fit_placeholder(1400, PLACEHOLDER_WS) == LONG
    # exact fit counts as fitting
    assert fit_placeholder(631, PLACEHOLDER_WS) == LONG
    assert fit_placeholder(630, PLACEHOLDER_WS) == SHORT


def test_fit_placeholder_falls_back_to_the_shortest_and_survives_empty():
    # nothing fits → the shortest, never '' (the field must still hint)
    assert fit_placeholder(200, PLACEHOLDER_WS) == PLAIN
    assert fit_placeholder(500, []) == ""
    assert fit_placeholder(500, None) == ""


def test_placeholder_variants_are_distinct_and_mention_the_wake_word():
    assert LONG != SHORT != PLAIN
    assert "Jarvis" in LONG and "Jarvis" in SHORT and "Jarvis" not in PLAIN
    assert len(SHORT) < len(LONG)


# ---------------------------------------------------------- telemetry
def test_plan_telemetry_reproduces_and_fixes_the_920_px_collision():
    # BEFORE: the wake-word segment ended at 256 px and the full cluster
    # started at 920 - 710 = 210 px — 46 px of overlap that plan_strip
    # never saw (no project → nothing to yield).
    assert plan_strip(920, WAKE_END_AT_920, 130, 14, 0, LEVEL0) == (0, [])
    assert 920 - sum(w for _n, w in LEVEL0) < WAKE_END_AT_920
    # AFTER: level 1 (compact) starts at 490 px, 234 px clear of the text
    level, chars, hidden = plan_telemetry(920, WAKE_END_AT_920, 130, 14, 0,
                                          LEVELS)
    assert (level, chars, hidden) == (1, 0, [])
    assert 920 - sum(w for _n, w in LEVEL1) - WAKE_END_AT_920 == 234


def test_plan_telemetry_keeps_the_full_cluster_when_it_fits():
    # a 1400-px window (700 design px) has room for everything
    assert plan_telemetry(1400, WAKE_END_AT_920, 130, 14, 0, LEVELS) == (0, 0, [])
    # the classic cut-off: exactly enough room stays at level 0
    assert plan_telemetry(WAKE_END_AT_920 + 710, WAKE_END_AT_920, 130, 14, 0,
                          LEVELS)[0] == 0
    assert plan_telemetry(WAKE_END_AT_920 + 709, WAKE_END_AT_920, 130, 14, 0,
                          LEVELS)[0] == 1


def test_plan_telemetry_steps_down_to_memory_only_and_never_past_it():
    # 500 px: even the compact cluster (430) collides with 256 → minimal
    assert plan_telemetry(500, WAKE_END_AT_920, 130, 14, 0, LEVELS) == (2, 0, [])
    # absurd: nothing fits; the narrowest level is still the answer
    assert plan_telemetry(300, WAKE_END_AT_920, 130, 14, 0, LEVELS)[0] == 2
    assert plan_telemetry(300, WAKE_END_AT_920, 130, 14, 0, [LEVEL0]) == (0, 0, [])


def test_plan_telemetry_leaves_every_layout_that_fitted_before_alone():
    # with a project the old plan already resolved this width by hiding
    # MEMORY and keeping six chip characters; the level stays 0 and the
    # plan_strip answer is returned verbatim — classic renders as today
    old = plan_strip(920, WAKE_END_AT_920, 130, 14, 6, LEVEL0)
    assert old == (6, ["MEMORY"])
    assert plan_telemetry(920, WAKE_END_AT_920, 130, 14, 6, LEVELS) == (0,) + old


def test_both_looks_now_plan_with_the_whole_ladder():
    """2026-09-02: classic gets the elision too.

    It used to plan with level 0 alone -- the 09-01 review's same-minute
    A/B found the elision to be the ONE classic pixel change outside the
    holo sphere, and classic is the fallback that renders the 08-31
    console token for token, FOUND 08-26 overprint included. Answering
    "is that free or used?" broke that contract by itself: the level-0
    memory value changed in BOTH looks ('47.5 GB' -> '47.5/122 GB'), so
    classic's cluster grew 710 -> 718 px and its overprint on his 918-px
    strip went 59 px -> 67. Given a classic 8 px MORE broken or a classic
    that elides, the elision wins."""
    for look in ("classic", "holo"):
        theme.select_look(look)
        assert levels_for_look(LIVE) == LIVE
    # and the overprint the contract used to preserve is gone: classic at
    # his window plans the compact rung instead of printing CPU over OFF
    assert plan_telemetry(STRIP_W, WAKE_END_CLASSIC, 130, 14, 0,
                          levels_for_look(LIVE, "classic")) == (1, 0, [])
    assert sum(w for _n, w in LIVE[0]) > STRIP_W - WAKE_END_CLASSIC
    assert sum(w for _n, w in LIVE[1]) <= STRIP_W - WAKE_END_CLASSIC
    # explicit look no longer changes the answer; empty still survives
    assert levels_for_look(LIVE, "classic") == levels_for_look(LIVE, "holo")
    assert levels_for_look([], "classic") == [] and levels_for_look(None) == []


def test_telemetry_segments_per_level():
    temps, mem = "cpu 53° 7% · gpu 44° 0%", "47.5/122 GB"
    assert telemetry_segments(temps, mem, 0) == [
        ("CPU", "53°C · 7%"), ("GPU", "44°C · 0%"), ("MEM", "47.5/122 GB")]
    assert telemetry_segments(temps, mem, 1) == [
        ("CPU", "53° 7%"), ("GPU", "44° 0%"), ("", "47.5/122 GB")]
    assert telemetry_segments(temps, mem, 2) == [
        ("CPU", "7%"), ("GPU", "0%"), ("", "47.5/122 GB")]
    assert telemetry_segments(temps, mem, 3) == [("", "47.5/122 GB")]
    # the ladder has four rungs and the last one is the floor
    assert TELEMETRY_LEVELS == 4 == len(LIVE)
    assert telemetry_segments(temps, mem, 9) == \
        telemetry_segments(temps, mem, TELEMETRY_LEVELS - 1)
    # the compact line reads 'CPU 53° 7% · GPU 44° 0% · 47.5/122 GB'
    assert " · ".join(f"{lab} {val}".strip()
                      for lab, val in telemetry_segments(temps, mem, 1)) \
        == "CPU 53° 7% · GPU 44° 0% · 47.5/122 GB"
    # ...and the load line 'CPU 7% · GPU 0% · 47.5/122 GB'
    assert " · ".join(f"{lab} {val}".strip()
                      for lab, val in telemetry_segments(temps, mem, 2)) \
        == "CPU 7% · GPU 0% · 47.5/122 GB"
    # unknowns
    assert telemetry_segments("", "", 0) == [("CPU", "--"), ("GPU", "--"),
                                             ("MEM", "--")]
    assert telemetry_segments("gpu 38°", None, 1) == [("CPU", "--"),
                                                      ("GPU", "38°"), ("", "--")]
    # a segment with no percentage has only its temperature to give
    assert telemetry_segments("gpu 38°", None, 2) == [("CPU", "--"),
                                                      ("GPU", "38°"), ("", "--")]


def test_the_compact_level_keeps_the_number_that_actually_moves():
    """2026-09-02, verbatim: "i dont think the CPU and GPU are updating".
    They were: the strip elides to level 1 at his 920-px window, and level
    1 dropped the PERCENTAGE and kept only the temperature -- which on this
    box drifts a degree or two over minutes. The compact rung reads as a
    frozen strip because the only fast-moving number had been elided away."""
    temps = "cpu 53° 7% · gpu 44° 0%"
    compact = [v for _lab, v in telemetry_segments(temps, "47.5/122 GB", 1)]
    assert "7%" in compact[0] and "0%" in compact[1]
    # a temperature-only rung is what it replaces
    assert compact[0] != fmt_temps_compact("cpu 53° 7%") == "53°"
    # and the utilisation survives every rung below it but the floor
    for level in range(1, TELEMETRY_LEVELS - 1):
        values = [v for _lab, v in telemetry_segments(temps, "47.5 GB", level)]
        assert "7%" in values[0] and "0%" in values[1]


def test_fmt_temps_compact():
    assert fmt_temps_compact("cpu 53° 7%") == "53°"
    assert fmt_temps_compact("gpu 38°") == "38°"
    assert fmt_temps_compact("") == "--"
    assert fmt_temps_compact(None) == "--"
    assert StatusStrip.SEGMENTS == ("CPU", "GPU", "MEMORY")


def test_the_livelier_cluster_still_respects_the_width_budget():
    """What the extra characters cost, against the budget the elision
    system exists to respect. The percentage and the /total are paid for
    out of level 1's slack (430 -> 570 px of the 650 free at his window),
    and level 0 still fits the 520-design-unit DEFAULT window, so no
    layout that showed the full cluster before loses it."""
    assert WAKE_END_OFF == 268 and WAKE_END_CLASSIC == 267
    assert plan_telemetry(STRIP_W, WAKE_END_OFF, 130, 14, 0, LIVE)[0] == 1
    assert STRIP_W - sum(w for _n, w in LIVE[1]) - WAKE_END_OFF == 80
    # the default window keeps the full cluster, with 52 px to spare
    assert plan_telemetry(1038, WAKE_END_OFF, 130, 14, 0, LIVE)[0] == 0
    assert 1038 - sum(w for _n, w in LIVE[0]) - WAKE_END_OFF == 52


def test_a_three_digit_temperature_under_load_never_blanks_the_strip():
    """The rung the load level exists for.

    Keeping the utilisation at level 1 spent most of its slack: at 100°
    and 100% on a pool over 100 GB the compact rung asks 654 px of the 650
    his strip leaves, so the plan drops a rung -- and until 2026-09-02 the
    rung below was memory ALONE. CPU and GPU would have vanished at
    exactly the load the whole change exists to surface. The load rung
    catches it with 122 px to spare, and drops the temperatures instead of
    the numbers that move."""
    free = STRIP_W - WAKE_END_OFF
    assert free == 650
    assert sum(w for _n, w in HOT[1]) == 654 > free      # compact tips
    assert sum(w for _n, w in HOT[2]) == 528 <= free     # load holds
    assert plan_telemetry(STRIP_W, WAKE_END_OFF, 130, 14, 0, HOT)[0] == 2
    hot = "cpu 100° 100% · gpu 88° 100%"
    assert telemetry_segments(hot, "121.7/122 GB", 2) == [
        ("CPU", "100%"), ("GPU", "100%"), ("", "121.7/122 GB")]
    # what the three-rung ladder would have shown instead: memory alone
    three_rung = [HOT[0], HOT[1], HOT[3]]
    assert plan_telemetry(STRIP_W, WAKE_END_OFF, 130, 14, 0, three_rung)[0] == 2
    assert telemetry_segments(hot, "121.7/122 GB", 3) == [("", "121.7/122 GB")]


def test_fmt_load_keeps_the_percentage_and_falls_back_to_the_temperature():
    assert fmt_load("cpu 53° 7%") == "7%"
    assert fmt_load("gpu 44° 100%") == "100%"
    assert fmt_load("gpu 38°") == "38°"        # no percentage to keep
    assert fmt_load("") == "--" and fmt_load(None) == "--"
    assert fmt_load("gpu unknown") == "--"


# --------------------------------------------------------- captions
def test_tracked_caps():
    assert tracked("JARVIS") == "J A R V I S"
    assert tracked("you") == "Y O U"
    assert tracked("wake word") == "W A K E W O R D"
    assert tracked("") == ""


# ---------------------------------------------------------- card look
def test_card_look_is_a_frame_in_holo_and_todays_slab_in_classic():
    theme.select_look("holo")
    j, u, g, p = (card_look(r) for r in ("jarvis", "user", "partial", "progress"))
    assert j["style"] == u["style"] == g["style"] == p["style"] == "frame"
    assert j["fill"] == u["fill"] == theme.TV_BG      # drawn ON the display
    assert j["edge"] == theme.GLASS_EDGE and j["accent"] == theme.BRIGHT
    assert u["edge"] == theme.LINE                    # YOU turns dimmer
    assert g.get("dash")                              # the listening ghost
    assert "dash" not in j and "dash" not in u
    theme.select_look("classic")
    assert card_look("jarvis") == dict(fill=theme.RAISED, style="slab")
    assert card_look("user") == dict(fill=theme.RAISED, style="slab")
    assert card_look("partial") == dict(fill=theme.SURFACE, style="slab")
    assert card_look("progress") == dict(fill=theme.SURFACE, style="slab")
    assert card_look("approval") == dict(fill=theme.RAISED, style="slab")
    assert card_look("briefing") == dict(fill=theme.RAISED, style="slab")


# ------------------------------------------------ call-time widget tokens
def test_widget_signatures_no_longer_capture_theme_tokens():
    # a def-time default froze the import-time look; the resolved value
    # now comes from theme at construction (None → theme.X in __init__)
    assert inspect.signature(Card.__init__).parameters["fill"].default is None
    assert inspect.signature(Meter.__init__).parameters["color"].default is None
    chip = inspect.signature(Chip.__init__).parameters
    assert chip["fg"].default is None and chip["fill"].default is None
    assert chip["size"].default is None
    assert inspect.signature(RoundButton.__init__).parameters["size"].default is None
    assert not hasattr(RoundButton, "_KINDS")
    assert not hasattr(Toast, "_KIND_FG")


def test_card_meter_chip_defaults_resolve_from_the_selected_look(monkeypatch):
    """The resolved-default path each __init__ takes (no root here, so
    the staticmethods stand in for construction): built after
    select_look("holo") they carry HOLO tokens, after "classic" TODAY's,
    and the two really differ — the def-time capture handed every widget
    the import-time look whatever the user chose."""
    theme.select_look("classic")
    c_fill, c_color = Card.resolve_fill(), Meter.resolve_color()
    c_chip = Chip.resolve_style()
    assert c_fill == theme.RAISED and c_color == theme.CYAN
    assert c_chip == (theme.FAINT, theme.RAISED, theme.SIZE_CAPTION)
    theme.select_look("holo")
    assert Card.resolve_fill() == theme.RAISED != c_fill
    # CYAN is the one anchor both looks share, so prove call-time reading
    # directly: a retinted token is what the next Meter gets
    assert Meter.resolve_color() == theme.CYAN == c_color
    monkeypatch.setattr(theme, "CYAN", "#0000ff")
    assert Meter.resolve_color() == "#0000ff"
    monkeypatch.undo()
    h_chip = Chip.resolve_style()
    assert h_chip == (theme.FAINT, theme.RAISED, theme.SIZE_CAPTION)
    assert h_chip[:2] != c_chip[:2] and h_chip[2] == c_chip[2]  # size invariant
    # an explicit argument always wins over the theme
    assert Card.resolve_fill("#123456") == "#123456"
    assert Meter.resolve_color("#abcdef") == "#abcdef"
    assert Chip.resolve_style("#111111", "#222222", 9) == ("#111111", "#222222", 9)
    # the classic values are the 08-31 tokens from the oracle
    # (tests/fixtures/theme_tokens_85d5066.json)
    assert c_fill == "#183748" and c_color == "#35e0ff"
    assert c_chip[0] == "#61788f"


def test_round_button_kinds_follow_the_selected_look():
    theme.select_look("classic")
    classic = RoundButton._kinds()
    assert classic["default"]["fill"] == theme.RAISED
    assert classic["accent"]["fill"] == theme.CYAN_SOFT
    assert classic["default"]["outline"] == "" and classic["ghost"]["fill"] == ""
    classic_raised = theme.RAISED
    theme.select_look("holo")
    holo = RoundButton._kinds()
    # holo buttons are outlined rings: empty idle fill, an outline tone
    assert holo["default"]["fill"] == "" and holo["accent"]["fill"] == ""
    assert holo["default"]["outline"] == theme.GLASS_EDGE
    assert holo["accent"]["outline"] == theme.CYAN_DIM
    assert holo["ghost"]["outline"] == ""
    # and the classic dict really was built from classic tokens
    assert classic["default"]["fill"] == classic_raised != theme.RAISED


def test_toast_kind_colour_is_read_per_call():
    theme.select_look("classic")
    assert Toast._kind_fg("warn") == theme.WARN
    assert Toast._kind_fg("ok") == theme.OK
    assert Toast._kind_fg("nonsense") == theme.INK
    theme.select_look("holo")
    assert Toast._kind_fg("info") == theme.INK
    assert Toast._kind_fg("error") == theme.ERR


# =====================================================================
# 2026-09-03 ui-polish (the 09-03 panel, ~/scratch-0903/ui-synthesis.md)
# =====================================================================
from jarvis.ui.board import (EMPTY_PANEL_DEFAULT, EMPTY_PANEL_TEXT,  # noqa: E402
                             empty_panel_text)
from jarvis.ui import views as views_mod  # noqa: E402
from jarvis.ui.views import (SNAP_FRAGMENT, SettingsDrawer,  # noqa: E402
                             snap_hidden, speaker_color)
from jarvis.ui.widgets import StatePill  # noqa: E402


# ------------------------------------------------ U01 transcript top snap
def test_snap_hides_the_small_fragment_the_view_cuts_and_keeps_the_rest():
    """The 09-03 shots, as arithmetic at S=2 (SNAP_FRAGMENT 96 -> 192 px).
    Four cards stacked 16 px apart; the view's top edge lands 178 px above
    the bottom of the first (08b's 'draft from Tuesday…' fragment, no head
    row): hidden. Everything below is whole: shown."""
    tops = [16, 316, 532, 748]
    heights = [284, 200, 200, 120]
    view_top = 16 + 284 - 178
    assert snap_hidden(tops, heights, view_top, 192) == [True, False, False, False]
    # a fragment taller than the floor stays -- a long reply half on
    # screen is still worth reading
    assert snap_hidden(tops, heights, 16 + 284 - 200, 192) == [False] * 4
    # the empty bracket of 23-claude-task (10 px left) and the halved
    # 'Here's your morning, sir.' of 09 (40 px) both go
    assert snap_hidden(tops, heights, 16 + 284 - 10, 192)[0] is True
    assert snap_hidden(tops, heights, 16 + 284 - 40, 192)[0] is True


def test_snap_never_hides_the_card_the_view_rests_on_or_a_whole_card():
    # one card taller than the view: it is the last card, it stays
    assert snap_hidden([16], [1400], 900, 192) == [False]
    # a card ENTIRELY above the edge is off-screen, not a fragment; Tk
    # clips it and it must stay mapped for the scroll back up
    assert snap_hidden([16, 316], [284, 100], 320, 192) == [False, False]
    # the edge exactly on a boundary cuts nothing
    assert snap_hidden([16, 316], [284, 100], 316, 192) == [False, False]
    assert snap_hidden([], [], 0, 192) == []
    assert SNAP_FRAGMENT == 96


# The wiring, not the arithmetic: a wheel tick and a relayout must both run
# the snap, or snap_hidden is a pure function nobody calls (the 09-04 review
# found the two tests above still passing with every _apply_snap() call
# deleted). No Tk: the view is built with __new__ over a fake canvas that
# records item states, and the fragment card must come out "hidden".
class _FakeCanvas:
    def __init__(self, width, height, tops=None):
        self.w, self.h = width, height
        self.pos = dict(tops or {})           # win id -> y
        self.state = {}                        # win id -> last state
        self.width = {}
        self.region_h = None
        self.top = 0

    # geometry ---------------------------------------------------------
    def winfo_exists(self):
        return True

    def update_idletasks(self):
        pass

    def winfo_width(self):
        return self.w

    def winfo_height(self):
        return self.h

    def coords(self, item, *xy):
        if xy:
            self.pos[item] = xy[1]
            return None
        return [0, self.pos[item]]

    def itemconfigure(self, item, **kw):
        if "state" in kw:
            self.state[item] = kw["state"]
        if "width" in kw:
            self.width[item] = kw["width"]

    def configure(self, **kw):
        if "scrollregion" in kw:
            self.region_h = kw["scrollregion"][3]

    # scrolling --------------------------------------------------------
    def yview_moveto(self, frac):
        self.top = int(max(0, (self.region_h or self.h) - self.h) * frac)

    def yview_scroll(self, n, _unit):
        self.top = max(0, self.top + n * 10)

    def yview(self):
        return (0.0, 1.0)

    def canvasy(self, y):
        return self.top + y


class _FakeCard:
    def __init__(self, height, text="body"):
        self.h = height
        self._text = text

    def sync(self):
        pass

    def winfo_reqheight(self):
        return self.h

    def cget(self, _key):
        return self._text


def _snap_view(heights, canvas):
    from jarvis.ui.views import TranscriptView
    view = TranscriptView.__new__(TranscriptView)      # no tk.Frame.__init__
    view.canvas = canvas
    view._cards = []
    for i, h in enumerate(heights):
        card = _FakeCard(h)
        view._cards.append([card, card, "jarvis", 100 + i, h])
    view._partial = None
    view._pinned = True
    view._layout_job = None
    view._schedule_dots = lambda: None
    view._card_geo = lambda role, W=None, text="": (0, 400, 380)
    return view


def test_a_wheel_tick_runs_the_snap_and_unmaps_the_fragment(monkeypatch):
    theme.select_look("holo")
    monkeypatch.setattr(views_mod, "px", lambda v: v * 2)   # S=2 arithmetic
    tops = {100: 16, 101: 316, 102: 532, 103: 748}
    canvas = _FakeCanvas(800, 700, tops)
    view = _snap_view([284, 200, 200, 120], canvas)
    canvas.top = 16 + 284 - 178 - 20          # one tick above the 08b cut
    view._wheel(1)                             # +20 px: lands the cut at -178
    assert canvas.top == 16 + 284 - 178
    assert canvas.state == {100: "hidden", 101: "normal", 102: "normal",
                            103: "normal"}, canvas.state
    # scrolling up unpins nothing here (yview says pinned), but a taller
    # remainder is a card worth reading: it comes back
    view._wheel(-3)
    assert canvas.state[100] == "normal"


def test_a_relayout_runs_the_snap_after_stacking_the_cards(monkeypatch):
    """A new card arrives, the stack is rebuilt top-anchored and pinned to
    the bottom, and the first card's 40 px remainder (09's halved 'Here's
    your morning, sir.') must be unmapped BY THE RELAYOUT -- it is the one
    code path a new reply always takes."""
    theme.select_look("holo")
    monkeypatch.setattr(views_mod, "px", lambda v: v * 2)
    heights = [284, 200, 200, 120]
    gap = theme.PAD_S
    total = sum(heights) + gap * 3
    # viewport sized so the view's top edge lands 40 px above the bottom
    # of the first card once pinned to the bottom of the scrollregion
    H = (gap + total + gap) - (gap + 284 - 40)
    canvas = _FakeCanvas(800, H)
    view = _snap_view(heights, canvas)
    view._relayout()
    assert canvas.top == gap + 284 - 40
    assert canvas.state[100] == "hidden", canvas.state
    assert [canvas.state[i] for i in (101, 102, 103)] == ["normal"] * 3
    # classic never snaps: the same relayout maps every card
    theme.select_look("classic")
    view._relayout()
    assert all(s == "normal" for s in canvas.state.values())
    theme.select_look("holo")


# ------------------------------------------- U09 the standby clock
# MEASURED 2026-09-04 on Xvfb :95 (scripts/ui_shots.py, holo, S=2, the
# tree before this fix), rows below the stage of 17-standby: the clock
# band peaks at luminance 159, its caption at 131, the row VALUES at 173
# -- the clock was the third brightest text on its own slab, because the
# holo head was CORE_BANDS[2] (#a8e9ff, L 221) under INK (#e9f2fb, L 241).
# The 09-03 panel shot (ui-shots-int) reads 129 / 106 / 140, same order.
def _lum(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_standby_clock_outranks_the_rows_and_its_caption_in_holo():
    from jarvis.ui import ambient
    from jarvis.ui.console_mode import DIM_FLOOR, dim
    theme.select_look("holo")
    head, body, faint = ambient.slab_ink("normal")
    assert head == theme.FOCAL and faint == theme.MUTED
    assert _lum(head) > _lum(body) > _lum(faint)
    # the ladder survives the standby dim at every factor down to the floor
    # (dim is a blend toward the ground, so the order is what matters)
    for f in (1.0, 0.7, DIM_FLOOR):
        assert _lum(dim(head, f)) > _lum(dim(body, f)) > _lum(dim(faint, f))
    # and the top of the ladder really is the top: no token brighter
    assert _lum(theme.FOCAL) >= max(_lum(theme.INK), _lum(theme.MUTED),
                                    _lum(theme.CYAN))


def test_holo_standby_caption_is_the_date_alone_and_classic_keeps_its_am():
    from datetime import datetime
    from jarvis.ui import ambient
    at = datetime(2026, 9, 4, 16, 26)
    assert ambient.standby_caption(at, look="holo") == \
        ambient.tracked("FRIDAY 4 SEPTEMBER")
    assert "PM" not in ambient.standby_caption(at, look="holo")
    # classic is frozen to the 85d5066 oracle: meridiem, dots, untracked
    assert ambient.standby_caption(at, look="classic") == \
        "PM  ·  FRIDAY 4 SEPTEMBER"
    theme.select_look("classic")
    assert ambient.standby_caption(at) == "PM  ·  FRIDAY 4 SEPTEMBER"
    theme.select_look("holo")
    assert ambient.standby_caption(at) == ambient.tracked("FRIDAY 4 SEPTEMBER")


def test_the_meridiem_sits_inline_after_the_digits_as_one_centred_group():
    from jarvis.ui.ambient import clock_line, meridiem_bottom
    # digits 240 px, gap 20, 'PM' 60: the 320 px group is centred on 460
    dx, mx = clock_line(460, 240, 60, 20)
    assert (dx, mx) == (460 - 160, 460 - 160 + 240 + 20)
    assert dx + 240 + 20 == mx                 # inline, one gap after
    assert (dx + (mx + 60)) // 2 == 460        # the GROUP is centred
    # baselines meet: a 120 px linespace with a 96 px ascent centred on
    # y=500 has its baseline at 500 - 60 + 96 = 536; a 9 px descent on
    # the smaller face puts its south anchor at 545
    assert meridiem_bottom(500, 120, 96, 9) == 545


# --------------------------------------------------- U03 label > meta
def test_speaker_label_lifts_one_step_in_holo_and_keeps_classic():
    theme.select_look("holo")
    assert speaker_color("jarvis") == theme.CYAN
    assert speaker_color("you") == theme.MUTED
    # the stamp stays FAINT (views._card_head) and the label is no longer
    # the stamp's colour on either card
    assert theme.FAINT not in (speaker_color("jarvis"), speaker_color("you"))
    theme.select_look("classic")
    assert speaker_color("jarvis") == theme.CYAN_DIM
    assert speaker_color("you") == theme.FAINT
    # explicit look beats the current one
    assert speaker_color("you", "holo") == theme.MUTED


# --------------------------------------- U14 footer with an active project
# MEASURED 2026-09-03 on Xvfb :92 at JARVIS_UI_SCALE 2.0, the rig's
# readings ('cpu 52° 7% · gpu 44° 2%', '26.8 GB'): wake segment ends at
# 257 ('WAKE WORD ON' 181 + ring/gaps), PROJECT fixed part 150, one mono
# character 14; level 0 CPU/GPU 218 each and MEM 226, level 1 176/176/162,
# level 2 120/120/162, level 3 162.
WAKE_END_23 = 257
PROJ_FIXED = 150
CHAR_W = 14
LEVELS_23 = [[("CPU", 218), ("GPU", 218), ("MEMORY", 226)],
             [("CPU", 176), ("GPU", 176), ("MEMORY", 162)],
             [("CPU", 120), ("GPU", 120), ("MEMORY", 162)],
             [("MEMORY", 162)]]


def test_a_dropped_project_chip_gives_the_memory_figure_back():
    """The 09-03 shots 23/24: 'CPU 52°C · 7% | GPU 44°C · 2%' -- level 0,
    MEMORY gone, and no PROJECT chip either. plan_strip yielded MEMORY for
    a chip that then still did not fit (75 px free = 5 characters, one
    under the six-character floor) and returned (0, ['MEMORY']): the chip
    dropped and the segment it evicted never came back."""
    before_free = STRIP_W - WAKE_END_23 - (218 + 218) - PROJ_FIXED
    assert before_free // CHAR_W == 5 < 6
    assert plan_strip(STRIP_W, WAKE_END_23, PROJ_FIXED, CHAR_W, 6,
                      LEVELS_23[0]) == (0, [])
    level, chars, hidden = plan_telemetry(STRIP_W, WAKE_END_23, PROJ_FIXED,
                                          CHAR_W, 6, LEVELS_23)
    # the ladder re-plans with the whole cluster and lands on the compact
    # rung with the chip -- either the chip or the memory figure, never
    # neither
    assert (level, chars, hidden) == (1, 6, ["MEMORY"])
    assert chars > 0 or "MEMORY" not in hidden
    # ...and the °-form no longer flips with the project: level 1 is what
    # the same strip plans with NO project
    assert plan_telemetry(STRIP_W, WAKE_END_23, PROJ_FIXED, CHAR_W, 0,
                          LEVELS_23)[0] == 1


def test_every_plan_shows_the_chip_or_the_memory_figure():
    for w in range(600, 1400, 7):
        for slug_len in (0, 3, 6, 12):
            level, chars, hidden = plan_telemetry(w, WAKE_END_23, PROJ_FIXED,
                                                  CHAR_W, slug_len, LEVELS_23)
            assert chars > 0 or "MEMORY" not in hidden, (w, slug_len)
            # a plan that shows no chip hides nothing on its behalf
            if chars == 0:
                assert hidden == []


# ------------------------------------------------- U16 the tracking rule
def test_caption_tracks_surface_labels_only_and_only_in_holo():
    theme.select_look("holo")
    assert theme.caption("Voice ID", surface=True) == "V O I C E   I D"
    assert theme.caption("weather", surface=False) == "WEATHER"
    assert theme.caption("alarm") == "A L A R M"
    assert theme.tracked_caps("Jarvis") == "J A R V I S"
    theme.select_look("classic")
    assert theme.caption("Voice ID", surface=True) == "VOICE ID"
    assert theme.caption("weather", surface=False) == "WEATHER"
    assert theme.caption("", surface=True) == ""


# ------------------------------------------ U05 the pill's dot has a shape
def test_state_pill_has_a_ring_shape_and_set_state_defaults_to_the_disc():
    sig = inspect.signature(StatePill.set_state)
    assert sig.parameters["shape"].default == StatePill.DOT_DISC == "disc"
    assert StatePill.DOT_RING == "ring"


# ------------------------------------------------ U02 the toast can dock
def test_toast_is_an_overlay_until_docked():
    t = Toast.__new__(Toast)
    Toast.__init__(t, container=None)
    assert not t.docked
    t.dock(host="shell", before="reactor")
    assert t.docked and t._host == "shell" and t._before == "reactor"
    assert Toast.STRIP_H == 34
    theme.select_look("holo")
    assert Toast._kind_dot("error") == theme.ERR
    assert Toast._kind_dot("warn") == theme.WARN
    assert Toast._kind_dot("ok") == theme.CYAN
    assert Toast._kind_dot("info") == theme.CYAN_DIM


# --------------------------------------------- U15 an empty board panel
def test_empty_panels_state_their_emptiness():
    assert empty_panel_text("focus") == "nothing in focus"
    assert empty_panel_text("sessions") == "no sessions"
    assert empty_panel_text("no-such-panel") == EMPTY_PANEL_DEFAULT
    assert empty_panel_text(None) == EMPTY_PANEL_DEFAULT
    for key, text in EMPTY_PANEL_TEXT.items():
        assert text and text == text.lower(), key      # a line, not a label


# ------------------------------------------- settings slider geometry
def test_the_holo_slider_row_fits_with_its_value_inline():
    """The value used to be drawn ABOVE a stock Tk scale, centred on the
    knob, so at the low end it overhung into the label ('Silence timeout
    (s)2.5', shot 14). It is a Label on the row's own baseline now, to the
    RIGHT of the track, so the row is a straight width budget.

    MEASURED on :94 at S=2, 2026-09-05: inner width 576, the widest label
    ('Silence timeout (s)') 310, the value ('0.015' in the caption mono
    face) 70. The arithmetic is here so a longer label or a longer track
    cannot silently collide again."""
    label_w, inner, value_w = 310, 576, 70
    row = (2 * SettingsDrawer.SLIDER_GAP_HOLO
           + 2 * SettingsDrawer.SLIDER_LEN_HOLO
           + 2 * SettingsDrawer.SLIDER_VALUE_GAP_HOLO + value_w)
    assert label_w + row <= inner, (label_w + row, inner)
    assert SettingsDrawer.SLIDER_VALUE_CHARS >= len("0.015")
    # the old stock-scale geometry really did fill the row to within 2 px
    assert inner - label_w - 2 * SettingsDrawer.SCALE_LEN_CLASSIC == 6


# The rule applied to the two row-key sites that still tracked (2026-09-04):
# both drawn on a fake canvas, no Tk.
def _fake_measure(_font, text):
    return 7 * len(text or "")


def test_engine_card_keys_and_load_readout_are_plain_caps_in_holo(monkeypatch):
    from jarvis.ui import reactor
    theme.select_look("holo")
    monkeypatch.setattr(reactor, "measure", _fake_measure)

    class _Stage:
        _draw_card = reactor.Reactor._draw_card
        _draw_holo_decor = reactor.Reactor._draw_holo_decor

        def __init__(self):
            self._decor = {}
            self.texts = []

        def create_line(self, *a, **k):
            return 1

        create_polygon = create_arc = create_oval = create_line

        def create_text(self, *a, **k):
            self.texts.append(k.get("text"))
            return 1

    st = _Stage()
    st._draw_holo_decor(918, 600, 300, 300)
    st._draw_card(918, 300)
    keys = [t for t in st.texts if t]
    labels = [lab for lab, _key in reactor.CARD_ROWS]
    assert keys[-len(labels):] == labels                  # HEAR, not H E A R
    assert reactor.GAUGE_LABEL in keys
    assert reactor.tracked(reactor.GAUGE_LABEL) not in keys
    assert all(" " not in k for k in labels)
    # the per-row value budget is still cut from the label actually drawn
    assert set(st._decor["card_budget"]) == {key for _lab, key in reactor.CARD_ROWS}


def test_standby_and_ambient_row_keys_are_plain_caps_in_holo(monkeypatch):
    from jarvis.ui import ambient
    theme.select_look("holo")
    monkeypatch.setattr(ambient, "measure", _fake_measure)
    ambient._MEASURE_CACHE.clear()

    class _Slab:
        _draw_framed_rows = ambient.RoomSlab._draw_framed_rows
        _draw_rows = ambient.RoomSlab._draw_rows

        def __init__(self):
            self._reveal = None
            self._dim = 1.0
            self.texts = []

        def _draw_frame(self, *a):
            pass

        def _pad_x(self):
            return 20

        def create_line(self, *a, **k):
            return 1

        def create_text(self, *a, **k):
            self.texts.append(k.get("text"))
            return 1

    rows = [("NEXT", "Lab report"), ("DUE", "tomorrow"), ("OUTSIDE", "72°")]
    try:
        slab = _Slab()
        slab._draw_framed_rows(rows, 900, 1400, 300, "#fff", "#888")
        assert slab.texts[::2] == ["NEXT", "DUE", "OUTSIDE"]
        slab = _Slab()
        slab._draw_rows(rows, 900, 1400, 300, "#fff", "#888")
        assert slab.texts[::2] == ["NEXT", "DUE", "OUTSIDE"]
        assert ambient.tracked("NEXT") == "N E X T"        # the helper is untouched
    finally:
        ambient._MEASURE_CACHE.clear()
