"""The tab row: its width and height budget, and the widget's behaviour.

TWO HALVES, and they are tested two different ways. The FIT is pure
arithmetic over measured text widths and is answered with no display at
all. The BEHAVIOUR -- selection, the keyboard, what the row does when the
console goes quiet -- is driven against the REAL widget on a private Xvfb
this file starts for itself (see the `display` fixture), because the first
version of these tests grepped this module's own source and would have
passed on code whose bindings had been deleted.

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

EVERY WIDTH BELOW WAS MEASURED on a private Xvfb (never his :1) at
JARVIS_UI_SCALE 1.0 and 2.0 in BOTH looks, off the real widgets'
winfo_reqwidth / winfo_reqheight, and kept here as literals -- and
test_the_pure_size_is_what_the_real_widget_asks_tk_for now RE-MEASURES
them against live widgets every run, so the transcription cannot rot. The
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
from jarvis.ui.main_window import MainWindow as _MW  # noqa: E402
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


# ============== selection, focus, keyboard -- THE REAL WIDGET, MEASURED
# These used to GREP this module's source. That is not a test of
# behaviour: `assert "<Return>" in src` passes if the binding is deleted
# and the string survives in a comment, and it says nothing about whether
# Return reaches invoke(). So the row is BUILT and DRIVEN -- in a process
# of its own (tests/live_tab_strip.py), because a Tk root may not be
# created in here: CLAUDE.md's convention is "No Tk in unit tests", and a
# first attempt at doing it inline ABORTED the whole suite (2026-09-03,
# SIGABRT at 87%, the faulthandler dump 100 threads deep with no Python
# frame to blame). The harness MEASURES; every assertion is below, against
# the pure functions and the theme tokens, so no comparison is ever "the
# driver said it was fine".
HARNESS = "tests.live_tab_strip"


@pytest.fixture(scope="module")
def observed():
    """Run the harness once and hand back what it saw.

    It starts a private Xvfb of its own and builds every root with
    ``tk.Tk(screenName=...)``, so this works whether or not DISPLAY is set
    and never touches his :1 either way.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, "-m", HARNESS], cwd=str(repo),
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, (done.stdout[-2000:], done.stderr[-2000:])
    lines = [ln for ln in done.stdout.splitlines() if ln.strip()]
    assert lines, done.stderr[-2000:]
    data = json.loads(lines[-1])
    if "error" in data:
        pytest.fail("the live row raised:\n" + data["error"])
    if "skip" in data:
        pytest.skip(data["skip"])
    return data


def _step(observed, name: str) -> dict:
    hits = [s for s in observed["transcript"] if s["step"] == name]
    assert len(hits) == 1, "%s: %d entries" % (name, len(hits))
    return hits[0]


# --------------------------------------------------- the arithmetic is live
@pytest.mark.parametrize("look", ("holo", "classic"))
@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_the_pure_size_is_what_the_real_widget_asks_tk_for(observed, scale,
                                                           look):
    """The literals at the top of this file are a transcription of a
    measurement, and a transcription rots. Here are the same four numbers
    off widgets built for this run, compared with the pure functions --
    and with the literals, when the display face this box resolves is the
    one they were measured with."""
    _at(scale, look)
    got = observed["geometry"]["%s-%s" % (scale, look)]
    for word in SHIPPED:
        key = word.lower()
        assert got["req_w"][key] == ts.tab_width(got["text_w"][word]), word
        assert got["req_h"][key] == ts.tab_height(got["linespace"]), word
    assert got["strip_req_h"] == ts.strip_height(got["linespace"])
    if got["display_face"] == "Chakra Petch":     # the face the literals used
        assert got["text_w"] == {w: TEXT_W[scale][w] for w in SHIPPED}
        assert got["linespace"] == LINESPACE[scale]
        assert got["strip_req_h"] == STRIP_REQ_H[scale]
        assert dict(got["measured_widths"]) == \
            {w.lower(): TAB_REQ_W[scale][w] for w in SHIPPED}
    assert got["clipped_918"] == []


def test_the_row_measures_its_own_tabs_and_reports_what_it_had_to_cut(observed):
    """``clipped()`` is the row's whole fitting assertion and what
    _fill_tabs logs at build time. Four real tabs, and the pure packer is
    checked against the live one on the same widths."""
    _at(2.0)
    got = observed["four_tabs"]
    assert got["keys"] == ["chat", "sensors", "settings", "camera"]
    widths = [tuple(p) for p in got["measured_widths"]]
    assert all(w > 0 for _k, w in widths)
    assert got["clipped_918"] == []
    assert [list(c) for c in ts.strip_clipped(918, widths)] == got["clipped_918"]
    cut = got["clipped_300"]
    assert [k for k, _d, _r in cut] == ["sensors", "settings", "camera"]
    assert cut[-1][1] == 0                        # unmapped outright
    assert [list(c) for c in ts.strip_clipped(300, widths)] == cut


# ------------------------------------------------------------- selection
def test_the_first_tab_added_is_selected_and_its_surface_is_not_run(observed):
    """The stage under the first tab is already what is on screen at build
    time, so calling show() on it would be a second, redundant open."""
    built = _step(observed, "built")
    assert built["selected"] == "chat"
    assert built["flags"] == {"chat": True, "sensors": False}
    assert built["calls"] == [] and built["changes"] == []


def test_clicking_a_tab_shows_its_surface_and_hides_the_one_before_it(observed):
    """The lifecycle the SENSORS page's poll thread hangs off: leave THEN
    select, in that order, so the old surface releases before the new one
    takes anything."""
    on = _step(observed, "click-sensors")
    assert on["calls"] == ["chat:leave", "sensors:select"]
    assert on["changes"] == ["sensors"]
    assert on["selected"] == "sensors"
    assert on["flags"] == {"chat": False, "sensors": True}
    back = _step(observed, "click-chat")
    assert back["calls"] == ["sensors:leave", "chat:select"]
    assert back["selected"] == "chat"


def test_clicking_the_tab_that_is_already_selected_does_nothing_at_all(observed):
    """The SENSORS page's own hide() syncs the strip back to CHAT, and the
    strip's CHAT press is what called hide(). Without the guard that is a
    loop -- so a press on an already-lit tab must run no callback."""
    again = _step(observed, "click-sensors-again")
    assert again["calls"] == [] and again["changes"] == []
    assert again["selected"] == "sensors"
    same = _step(observed, "select-same-returns")
    assert same["returned"] == [False, False]     # already selected; unknown key
    assert same["calls"] == []


def test_the_selected_tab_is_marked_by_channels_that_owe_nothing_to_focus(observed):
    """He leaves the console unfocused on his desk all day. The selected
    tab is a brighter word, a lit underline and a raised ground; the
    keyboard focus RING is a different channel. Read off the canvas with
    the focus parked on the UNSELECTED tab."""
    _at(2.0, "holo")
    paint = observed["paint"]
    tokens = paint["tokens"]
    assert tokens["FOCAL"] == theme.FOCAL and tokens["CYAN"] == theme.CYAN
    text = [fill for kind, fill in paint["selected_items"] if kind == "text"]
    rects = [fill for kind, fill in paint["selected_items"] if kind == "rectangle"]
    assert text == [tokens["FOCAL"]]
    assert tokens["CYAN"] in rects                # the underline
    assert tokens["RAISED"] in rects              # the raised ground
    off_text = [f for k, f in paint["unselected_items"] if k == "text"]
    off_rects = [f for k, f in paint["unselected_items"] if k == "rectangle"]
    assert off_text == [tokens["MUTED"]] and off_rects == []
    # the ring is on the FOCUSED tab, which is the unselected one here
    assert paint["ring"]["sensors"] == tokens["CYAN"]
    assert paint["ring"]["chat"] == paint["bg"]["chat"]
    # ONE tab in both states is exactly the same size: the underline is
    # inside the box, so a click must not move anything under the row.
    assert paint["chat_size_selected"] == paint["chat_size_unselected"]
    assert [f for k, f in paint["chat_items_unselected"] if k == "text"] == \
        [tokens["MUTED"]]


# -------------------------------------------------------------- keyboard
@pytest.mark.parametrize("key", ("<Return>", "<space>"))
def test_return_and_space_select_the_tab_that_has_focus(observed, key):
    """Every tab takes focus in the normal traversal order and either key
    commits. Both are pressed on the real widget; neither is grepped."""
    before = _step(observed, "focus-sensors" + key)
    assert before["focus"] == "sensors" and before["selected"] == "chat"
    assert before["takefocus"] == "1"
    after = _step(observed, "press" + key)
    assert after["selected"] == "sensors"
    assert after["calls"] == ["chat:leave", "sensors:select"]


def test_the_arrow_keys_move_focus_along_the_row_and_do_not_select(observed):
    """Arrowing past SENSORS must not start its poll thread on the way by:
    Left and Right move FOCUS, and Return or space is the commitment. They
    stop at the ends rather than wrapping."""
    right = _step(observed, "right-from-chat")
    assert right["focus"] == "sensors"
    assert right["selected"] == "chat" and right["calls"] == []
    left = _step(observed, "left-from-sensors")
    assert left["focus"] == "chat" and left["calls"] == []
    assert _step(observed, "left-at-the-start")["focus"] == "chat"
    assert _step(observed, "right-at-the-end")["focus"] == "sensors"
    for name in ("right-from-chat", "left-from-sensors", "left-at-the-start",
                 "right-at-the-end"):
        assert _step(observed, name)["calls"] == [], name


def test_focus_selected_puts_the_keyboard_on_the_lit_tab(observed):
    got = _step(observed, "focus-selected")
    assert got["selected"] == "sensors" and got["focus"] == "sensors"


# ------------------------------------------------------- a surface that fails
def test_a_surface_whose_callback_raises_does_not_take_the_row_with_it(observed):
    """A diagnostics page that fails to open must leave the console
    usable: the strip logs and carries on, and the lit tab still matches
    what the press asked for."""
    got = observed["raises"]
    assert got["returned"] is True
    assert got["selected"] == "sensors"
    assert got["flags"] == {"chat": False, "sensors": True}
    assert got["calls"] == ["chat:leave", "sensors:select"]
    assert got["changes"] == ["sensors"]


# ------------------------------------------ the shape of the wiring
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


# ================================================== the row in STANDBY
# _set_footer_hidden's own words: standby makes the panel "a clock and
# nothing else". The strip is packed into the shell ABOVE the stage, so
# nothing the footer does reaches it -- photographed 2026-09-03, CHAT and
# SENSORS at full brightness over the dimmed clock, and one click on
# SENSORS there started the poll thread behind it. F9 could already do
# that; a tab makes it a one-click accident.
#
# Driven through the SHIPPING MainWindow methods on a stand-in, the way
# tests/test_found_standby_drag_snapback.py drives the same seam -- not
# by reading the source, which cannot tell whether the call is reached.
class _Root:
    def __init__(self):
        self.x, self.y = 100, 200

    def winfo_x(self):
        return self.x

    def winfo_y(self):
        return self.y

    def geometry(self, spec=None):
        return "920x1440+%d+%d" % (self.x, self.y)


class _Strip:
    """Records what the console does to the row. ``select`` runs the
    leave/select callbacks the way the real strip does."""

    def __init__(self, page=None):
        self._keys = ("chat", "sensors")
        self.selected = "chat"
        self.packed = True
        self.pack_kw = []
        self.forgets = 0
        self.page = page

    @property
    def keys(self):
        return self._keys

    def select(self, key, run=True):
        if key == self.selected:
            return False
        was, self.selected = self.selected, key
        if run and self.page is not None:
            self.page.hide() if was == "sensors" else None
            self.page.show() if key == "sensors" else None
        return True

    def pack(self, **kw):
        self.packed = True
        self.pack_kw.append(kw)

    def pack_forget(self):
        self.packed = False
        self.forgets += 1


class _Page:
    """The SENSORS surface: open/shut plus whether it is polling."""

    def __init__(self):
        self.polling = False
        self.shown = 0
        self.hidden = 0

    def show(self):
        self.polling, self.shown = True, self.shown + 1

    def hide(self):
        self.polling, self.hidden = False, self.hidden + 1


class _Console:
    """The console's mode seam with nothing else attached. The methods are
    taken UNBOUND from MainWindow, so these tests exercise the shipping
    code path rather than a re-description of it.

    ``reactor`` and ``transcript`` are deliberately bare objects:
    _on_console_mode guards both with ``except AttributeError``, so a stub
    that answers neither takes the same path a console whose widgets are
    not built yet takes."""

    _on_console_mode = _MW._on_console_mode
    _set_tabs_hidden = _MW._set_tabs_hidden

    def __init__(self, page=None):
        self.root = _Root()
        self.sensors = page
        self.tabs = _Strip(page)
        self.reactor = object()
        self.transcript = object()
        self.room = type("R", (), {"set_mode": lambda self, m: None})()
        self.preview = None
        self.preview_worker = None
        self._standby_origin = None
        self._standby_drift = (0, 0)
        self._footer_hidden = False
        self._tabs_hidden = False
        self.footer_calls = []

    def _set_footer_hidden(self, hidden):
        """Stubbed: the real one pack_forgets two widgets that are not
        built here. Recorded, because the assertion is that the row goes
        with the footer rather than staying behind it."""
        self.footer_calls.append(bool(hidden))
        self._footer_hidden = bool(hidden)

    def _move_to(self, x, y):
        self.root.x, self.root.y = int(x), int(y)

    def _preview_apply(self, mode):
        pass

    def mode(self, mode):
        self._on_console_mode(mode)


def test_standby_takes_the_tab_row_away_with_the_footer():
    """A row of lit tabs over a dimmed clock is new chrome in the quiet
    mode, and it is CLICKABLE: the strip is above the stage, so
    _set_footer_hidden never reached it."""
    from jarvis.ui.console_mode import ACTIVE, STANDBY
    con = _Console()
    assert con.tabs.packed
    con.mode(STANDBY)
    assert con.footer_calls == [True]
    assert not con.tabs.packed, "the tab row is still on screen in standby"
    con.mode(ACTIVE)
    assert con.tabs.packed
    assert con.tabs.pack_kw[-1].get("side") == "top"
    # back UNDER THE WORDMARK, not at the bottom of the shell's top stack:
    # pack() with no anchor appends, which would put the row under the
    # transcript.
    assert con.tabs.pack_kw[-1].get("before") is con.reactor


def test_a_sensors_page_open_when_standby_arrives_stops_polling():
    """The half that costs him something real. Hiding the row alone would
    leave the poll thread running behind the clock -- against a radar the
    curfew may have just powered down."""
    from jarvis.ui.console_mode import ACTIVE, STANDBY
    page = _Page()
    con = _Console(page)
    con.tabs.select("sensors")
    assert page.polling
    con.mode(STANDBY)
    assert not page.polling, "the SENSORS page kept polling behind the clock"
    assert con.tabs.selected == "chat"
    # and coming back does NOT reopen it: he left standby, not a page
    con.mode(ACTIVE)
    assert not page.polling and page.shown == 1


def test_a_page_opened_by_f9_with_no_strip_is_shut_by_standby_too():
    """The strip is optional chrome (_build_tabs returns None if it fails)
    and F9 still works without it, so the quiet mode cannot rely on the
    strip being there to stop the poll."""
    from jarvis.ui.console_mode import STANDBY
    page = _Page()
    con = _Console(page)
    con.tabs = None
    page.show()
    con.mode(STANDBY)
    assert not page.polling
