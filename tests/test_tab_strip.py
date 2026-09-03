"""The tab row's width and height budget, tested display-free.

2026-09-03, verbatim: "just make the app larger if we are running out of
header room or better yet, make a little tab to click thats underneath
jarvis, that can be the area that has multiple tabs".

WHY THE ROW AND NOT THE HEADER, in numbers. tests/test_header_fit.py
measures 312 px left for chips at his 920-px window and the state pill
plus the sensing badge already spend 299 of them -- 13 px spare. A
`[SENSORS]` chip there clips the sensing badge, and tkPack does not
overlap an over-subscribed bar, it TRUNCATES the last child packed and
then UNMAPS it (jarvis/ui/views.header_spans, transcribed from Tk 8.6's
generic/tkPack.c). An absent badge and a badge reading SENSING must not
look the same. Widening the window past 920 was ruled out too: his width
is deliberate, and it only moves the problem one tab along.

The row has the WHOLE width instead. At his window it spends 292 of 854
usable px on two tabs and holds FIVE of the widest word before anything
clips -- which is what "designed from the start to hold more than two"
has to mean if it is going to mean anything.

EVERY WIDTH BELOW WAS MEASURED on a private Xvfb (:95, never his :1) at
JARVIS_UI_SCALE 1.0 and 2.0 in BOTH looks, off the real widgets'
winfo_reqwidth / winfo_reqheight, and kept here as literals so this file
stays Tk-free -- the same house pattern as tests/test_header_fit.py. The
two looks measured IDENTICALLY: the type scale is look-independent, so
the row costs the same in holo and classic.

    S=1.0   CHAT text 32 px  SENSORS 58   linespace 17
            tab 58 / 84      row height 42
    S=2.0   CHAT text 60 px  SENSORS 108  linespace 32
            tab 110 / 158    row height 80      <- his scale
"""
import os
import tempfile

import pytest

os.environ.setdefault("JARVIS_LOG_DIR", tempfile.mkdtemp(prefix="jarvis-tab-"))
os.environ.setdefault("JARVIS_ASSISTANT_CONFIG",
                      os.path.join(tempfile.mkdtemp(prefix="jarvis-tab-cfg-"),
                                   "assistant.json"))

from jarvis.ui import tab_strip as ts  # noqa: E402
from jarvis.ui import theme  # noqa: E402
from jarvis.ui.widgets import px  # noqa: E402

HIS_W = 918             # his 920-px window, less the shell's 1-px inset
DEFAULT_W = 1038        # the 520-design-unit default window

# MEASURED text widths of the display face at SIZE_CAPTION semibold
# ('Chakra Petch SemiBold'), the face the header chips and the wordmark
# use. Both looks, identical.
TEXT_W = {
    1.0: {"CHAT": 32, "SENSORS": 58, "LOGS": 32, "CAMERA": 50,
          "SETTINGS": 58, "ROOMS": 44},
    2.0: {"CHAT": 60, "SENSORS": 108, "LOGS": 61, "CAMERA": 94,
          "SETTINGS": 111, "ROOMS": 84},
}
LINESPACE = {1.0: 17, 2.0: 32}
# what the real widgets requested, at the same two scales
TAB_REQ_W = {1.0: {"CHAT": 58, "SENSORS": 84}, 2.0: {"CHAT": 110, "SENSORS": 158}}
TAB_REQ_H = {1.0: 33, 2.0: 62}
STRIP_REQ_H = {1.0: 42, 2.0: 80}

SHIPPED = ("CHAT", "SENSORS")


@pytest.fixture(autouse=True)
def _restore_look():
    """Leave the theme in its import-time state (holo, scale 1.0)."""
    yield
    theme.apply_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)


def _at(scale: float, look: str = "holo"):
    from jarvis.ui.widgets import set_scale
    theme.select_look(look)
    theme.apply_scale(scale)
    set_scale(scale)


def _widths(words, scale: float):
    """[(key, requested tab width)] for a row of these words."""
    return [(w.lower(), ts.tab_width(TEXT_W[scale][w])) for w in words]


# ------------------------------------------ the pure geometry IS the widget
@pytest.mark.parametrize("look", ("holo", "classic"))
@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_the_pure_size_is_the_size_the_widget_asks_tk_for(scale, look):
    """The whole point of a display-free fit test is that the arithmetic
    it does is the arithmetic Tk does. These four numbers are what the
    real widgets returned from winfo_reqwidth / winfo_reqheight under
    Xvfb; if the module's padding changes and these do not, the test is
    measuring its own opinion."""
    _at(scale, look)
    for word in SHIPPED:
        assert ts.tab_width(TEXT_W[scale][word]) == TAB_REQ_W[scale][word]
    assert ts.tab_height(LINESPACE[scale]) == TAB_REQ_H[scale]
    assert ts.strip_height(LINESPACE[scale]) == STRIP_REQ_H[scale]


def test_the_border_is_outside_the_drawn_box():
    """A tab is a Canvas with a 1-px focus ring, and the ring is NOT part
    of the width Tk is configured with. Conflating the two made the first
    build 2 px wider than it measured."""
    _at(2.0)
    assert ts.tab_width(60) == ts.tab_box_w(60) + 2 * ts.BORDER
    assert ts.tab_height(32) == ts.tab_box_h(32) + 2 * ts.BORDER


def test_a_selected_tab_is_exactly_as_big_as_an_unselected_one():
    """The underline lives INSIDE the box, so the row does not twitch when
    he clicks and nothing below it moves."""
    _at(2.0)
    assert ts.tab_box_h(32) >= px(ts.UNDERLINE)
    assert ts.tab_height(32) == TAB_REQ_H[2.0]      # one height, both states


# ---------------------------------------------------- the row fits, in numbers
@pytest.mark.parametrize("look", ("holo", "classic"))
@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_the_two_shipped_tabs_fit_his_window_in_both_looks(scale, look):
    _at(scale, look)
    assert ts.strip_clipped(HIS_W, _widths(SHIPPED, scale)) == []
    assert ts.strip_clipped(DEFAULT_W, _widths(SHIPPED, scale)) == []


def test_a_third_tab_is_a_one_liner_and_still_fits():
    """The requirement. Adding LOGS changes no width, no pad and no test
    of the two before it -- the trailing gap is on the right of every tab,
    so a tab appended at the end moves nothing."""
    _at(2.0)
    two = ts.strip_spans(HIS_W, _widths(SHIPPED, 2.0))
    three = ts.strip_spans(HIS_W, _widths(SHIPPED + ("LOGS",), 2.0))
    assert three[:2] == two
    assert ts.strip_clipped(HIS_W, _widths(SHIPPED + ("LOGS",), 2.0)) == []


def test_four_tabs_of_the_longest_words_he_is_likely_to_want_still_fit():
    _at(2.0)
    words = ("CHAT", "SENSORS", "SETTINGS", "CAMERA")
    assert ts.strip_clipped(HIS_W, _widths(words, 2.0)) == []


def test_the_row_holds_five_of_the_widest_word_at_his_window():
    """The number that answers "will it take more than two". Five SENSORS-
    width tabs fit at 918 px and the sixth would be clipped -- so the row
    is not a two-tab strip with a third bolted on."""
    _at(2.0)
    wide = ts.tab_width(TEXT_W[2.0]["SENSORS"])
    assert ts.tabs_that_fit(HIS_W, wide) == 5
    five = [("t%d" % i, wide) for i in range(5)]
    assert ts.strip_clipped(HIS_W, five) == []
    assert ts.strip_clipped(HIS_W, five + [("t5", wide)]) != []


def test_the_row_reports_a_tab_it_had_to_cut_rather_than_hiding_it():
    """Same assertion the header makes, and for the same reason: tkPack
    unmaps what it cannot fit, so "nothing collides" says nothing and
    "nothing is clipped" is the whole check."""
    _at(2.0)
    wide = ts.tab_width(TEXT_W[2.0]["SENSORS"])
    cut = ts.strip_clipped(300, [("a", wide), ("b", wide), ("c", wide)])
    assert [name for name, _drawn, _req in cut] == ["b", "c"]
    assert cut[-1][1] == 0                      # unmapped outright


def test_the_row_pays_for_its_own_padding_out_of_the_window():
    _at(2.0)
    assert ts.row_width(HIS_W) == HIS_W - 2 * theme.PAD
    assert ts.row_width(0) == 0


# ------------------------------------------------- what the stage gives up
def test_what_the_row_costs_the_stage_is_a_measured_number():
    """80 device px at his scale. The SENSORS page has to still fit in
    what is left, and it does BECAUSE it gave up its own tab row, which
    cost more than this (measured on the photo rig, 09-03)."""
    _at(2.0)
    assert ts.strip_height(LINESPACE[2.0]) == 80
    _at(1.0)
    assert ts.strip_height(LINESPACE[1.0]) == 42


def test_the_row_costs_the_same_in_both_looks():
    """The type scale is look-independent, so a look switch may not move
    the stage under him."""
    for scale in (1.0, 2.0):
        heights = set()
        for look in ("holo", "classic"):
            _at(scale, look)
            heights.add(ts.strip_height(LINESPACE[scale]))
        assert len(heights) == 1, heights


# ----------------------------------------------------------- odd arguments
def test_a_tab_with_no_measurable_text_still_has_a_size():
    _at(1.0)
    assert ts.tab_width(None) == 2 * px(ts.TAB_PAD_X) + 2 * ts.BORDER
    assert ts.tab_width(-40) == ts.tab_width(0)
    assert ts.tab_height("nonsense") > 0


def test_an_empty_row_is_not_a_clipped_row():
    _at(2.0)
    assert ts.strip_spans(HIS_W, []) == []
    assert ts.strip_clipped(HIS_W, None) == []
    assert ts.tabs_that_fit(HIS_W, 0) == 0


# -------------------------------------------------- the module's own habits
def test_the_strip_reads_the_theme_at_call_time():
    """tests/test_theme_look.py parametrises over every UI module and would
    catch a def-time capture, but this row is BELOW the wordmark and a
    frozen colour there is a frozen look on the most visible chrome in the
    app. Asserted here too, against the same scanner."""
    from tests.test_theme_look import _def_time_theme_captures
    from pathlib import Path
    hits = _def_time_theme_captures(Path(ts.__file__))
    assert hits == []


def test_no_attribute_shadows_a_tk_internal_of_the_same_name():
    """`self._w = ...` before super().__init__ cost the first build a
    TypeError on every draw: tkinter.Misc uses `_w` for the widget's Tcl
    pathname and overwrote it, so `max(self.winfo_width(), self._w)`
    compared a string with an int. RoundButton names its own `_btn_w` for
    exactly this reason. An AST scan is the standing guard."""
    import ast
    import tkinter as tk
    reserved = {"_w", "_name", "_tclCommands", "tk", "master", "children",
                "widgetName"} | {n for n in vars(tk.Misc) if not n.startswith("__")}
    tree = ast.parse(open(ts.__file__, encoding="utf-8").read())
    clashes = set()
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for node in ast.walk(cls):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign)
                       else [])
            for t in targets:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self" and t.attr in reserved):
                    clashes.add("%s.%s" % (cls.name, t.attr))
    assert not clashes, sorted(clashes)


# --------------------------------------------- selection, focus, keyboard
def _source() -> str:
    return open(ts.__file__, encoding="utf-8").read()


def _body(name: str) -> str:
    """One method's source, comments stripped -- a docstring may SAY what
    the code does not do."""
    import re
    src = _source()
    start = src.index("    def %s(self" % name)
    rest = src[start + 10:]
    end = rest.index("\n    def ") if "\n    def " in rest else len(rest)
    return re.sub(r"(?m)#.*$", "", src[start:start + 10 + end])


def test_the_selected_tab_is_marked_by_something_other_than_focus():
    """He leaves the console unfocused on his desk all day. Selection is
    three channels that owe nothing to focus -- a brighter word, a lit
    underline and a raised ground -- and the keyboard focus RING is a
    separate one, so the two states are never confused."""
    draw = _body("_draw")
    assert "self._selected" in draw
    for channel in ("theme.FOCAL", "theme.CYAN", "theme.RAISED"):
        assert channel in draw, channel
    # the ring is drawn nowhere near the selection
    assert "highlightcolor" not in draw and "focus" not in draw.lower()
    ring = _body("_ring")
    assert "highlightcolor" in ring and "self._selected" not in ring


def test_every_tab_is_reachable_and_operable_from_the_keyboard():
    src = _source()
    assert "takefocus=1" in src
    for key in ("<Return>", "<space>", "<Left>", "<Right>"):
        assert key in src, key


def test_the_arrow_keys_move_focus_and_do_not_select():
    """Arrowing past SENSORS must not start its poll thread on the way by,
    so Left/Right move focus only; Return or space is the commitment."""
    step = _body("_step")
    assert "focus_set" in step and "self.select(" not in step
    assert 'return "break"' in step          # Tk's own traversal stays out


def test_selecting_the_tab_that_is_already_selected_does_nothing():
    """The SENSORS page's own hide() syncs the strip back to CHAT, and the
    strip's CHAT press is what called hide(). Without the guard that is a
    loop."""
    select = _body("select")
    assert "key == self._selected" in select and "return False" in select


def test_adding_a_surface_is_one_call_that_carries_its_own_lifecycle():
    """`select` and `leave` are on the tab, so the strip starts and stops
    the page's poll thread and nothing else does -- which is what keeps
    "only while it is on screen" true now that flipping in and out is one
    click instead of a function key."""
    import inspect
    sig = inspect.signature(ts.TabStrip.add)
    assert list(sig.parameters) == ["self", "key", "text", "select", "leave"]
    assert sig.parameters["select"].default is None
    assert sig.parameters["leave"].default is None
    assert ts.Tab("k", "T").select is None


def test_the_console_wires_its_tabs_with_one_line_each():
    """The requirement, read off main_window: a third surface is one more
    add() and nothing else. If this ever needs a second line per tab, the
    strip has stopped being the thing he asked for."""
    import re
    from pathlib import Path
    src = Path(ts.__file__).with_name("main_window.py").read_text()
    body = src[src.index("    def _fill_tabs"):src.index("    def _build_sensors")]
    adds = re.findall(r"strip\.add\(", body)
    assert len(adds) == 2, adds
    # and each one is a single statement, not a block
    assert 'strip.add("chat", "CHAT")' in body


def test_f9_and_the_tab_cannot_disagree_about_what_is_on_screen():
    """F9 stays as the shortcut but goes THROUGH the strip, so the lit tab
    and the surface are one state, not two that drift."""
    from pathlib import Path
    src = Path(ts.__file__).with_name("main_window.py").read_text()
    body = src[src.index("    def sensors_toggle"):src.index("    def _build_footer")]
    assert "strip.select(" in body
    assert "<F9>" in src
    # and the page's own hide() puts the strip back
    assert "_sensors_closed" in src
