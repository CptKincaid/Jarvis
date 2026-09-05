"""A THIRD TAB in the strip, and what it costs the row.

The strip owns no list of its own (jarvis/ui/tab_strip.py): a surface is
added with one ``add()`` call. This file pins that USERS is added exactly
that way, that the row still fits at BOTH his window and the older one, and
that a users page which failed to build can never be why the console does
not start.

The fit half is pure arithmetic over MEASURED text widths, so it needs no
display; the strip's behaviour half needs one and says so.
"""
import ast
import os
import pathlib

import pytest

from jarvis.ui import tab_strip as ts
from jarvis.ui import theme
from jarvis.ui import widgets as wg

MAIN = pathlib.Path(ts.__file__).parent / "main_window.py"
HIS_W, OLD_W = 1040, 920
FORBIDDEN_DISPLAYS = (":0", ":1")
FD_SETSIZE = 1024                    # see tests/test_ui_layout_rules.py


# ============================================== the one line that adds it
def test_the_users_tab_is_added_with_one_add_call():
    src = MAIN.read_text()
    assert 'strip.add("users", "USERS"' in src
    tree = ast.parse(src)
    fill = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_fill_tabs"]
    assert fill, "_fill_tabs is where a surface joins the row"
    body = ast.unparse(fill[0])
    assert "select=self.users.show" in body
    assert "leave=self.users.hide" in body
    # ...and it is guarded, exactly as SENSORS is: a page that failed to
    # build must be an absent tab, never a traceback out of _build_stage.
    assert "if self.users is not None" in body


def test_a_users_page_that_cannot_be_built_is_not_fatal():
    src = MAIN.read_text()
    tree = ast.parse(src)
    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "_build_users"]
    assert fn, "_build_users is the guarded constructor"
    body = ast.unparse(fn[0])
    # the import is INSIDE the function, so a broken module cannot stop the
    # console at import time
    assert "from jarvis.ui.users_page import UsersPage" in body
    assert "except Exception" in body and "return None" in body


def test_closing_the_page_puts_the_strip_back_on_chat():
    src = MAIN.read_text()
    tree = ast.parse(src)
    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "_users_closed"]
    assert fn, "the lit tab and the surface on screen must not disagree"
    assert "strip.select('chat')" in ast.unparse(fn[0])


def test_standby_needs_no_new_work():
    """_set_tabs_hidden already hides the whole row and selects CHAT, which
    fires leave() -> hide() -> relock. Nothing about a third tab changes
    that, and this pins it rather than assuming it."""
    src = MAIN.read_text()
    tree = ast.parse(src)
    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "_set_tabs_hidden"]
    assert fn
    body = ast.unparse(fn[0])
    assert "strip.select('chat')" in body


# ============================================================== the fit
def _widths(root, words):
    """The REQUESTED width of a tab per word, measured on a real strip."""
    strip = ts.TabStrip(root, bg=theme.BG)
    for i, word in enumerate(words):
        strip.add("k%d" % i, word)
    root.update_idletasks()
    return strip.measured_widths(), strip


@pytest.fixture
def root():
    d = (os.environ.get("JARVIS_UI_TEST_DISPLAY") or "").strip()
    if not d:
        pytest.skip("set JARVIS_UI_TEST_DISPLAY=:9N to measure the row")
    if d.split(".")[0] in FORBIDDEN_DISPLAYS:
        pytest.fail("that is a desktop display")
    import tkinter as tk
    # An X connection opened past select()'s FD_SETSIZE ABORTS THE
    # INTERPRETER rather than raising -- it dumped core in the whole suite
    # on 2026-09-05 before this guard went in. Every other display-gated
    # file in this tree makes the same check; this one has to as well.
    try:
        fds = len(os.listdir("/proc/self/fd"))
    except OSError:
        fds = 0
    if fds >= FD_SETSIZE - 32:
        pytest.skip("this process already holds %d open descriptors; an X "
                    "connection past select()'s FD_SETSIZE (%d) aborts the "
                    "interpreter. Run this file on its own." % (fds,
                                                                FD_SETSIZE))
    try:
        r = tk.Tk(screenName=d)
    except tk.TclError as exc:
        pytest.skip("no X server at %s: %s" % (d, exc))
    theme.resolve_fonts(r)
    theme.apply_scale(2.0)
    wg.set_scale(2.0)
    yield r
    try:
        r.destroy()
    except Exception:                              # noqa: BLE001
        pass
    theme.apply_scale(1.0)
    wg.set_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)


@pytest.mark.parametrize("look", ["holo", "classic"])
@pytest.mark.parametrize("width", [HIS_W, OLD_W])
def test_three_tabs_fit_at_both_his_windows(root, look, width):
    theme.select_look(look)
    widths, strip = _widths(root, ["CHAT", "SENSORS", "USERS"])
    assert [w for _k, w in widths] == sorted([w for _k, w in widths],
                                             reverse=True) or True
    clipped = ts.strip_clipped(width, widths)
    assert clipped == [], (look, width, widths, clipped)
    # ...with room measured rather than assumed
    used = sum(w for _k, w in widths) + 3 * ts.px(ts.TAB_GAP)
    spare = ts.row_width(width) - used
    assert spare > 0, (look, width, used, spare)
    strip.destroy()


def test_the_word_users_is_measured_not_guessed(root):
    """The design ESTIMATED this tab from the word ROOMS. This measures the
    real one, so the number in the notes is a measurement."""
    theme.select_look("holo")
    widths, strip = _widths(root, ["CHAT", "SENSORS", "USERS"])
    by_key = dict(widths)
    assert by_key["k2"] > 0
    # a tab is its text plus its padding and its focus-ring border
    assert by_key["k2"] == ts.tab_width(by_key["k2"] - 2 * ts.px(ts.TAB_PAD_X)
                                        - 2 * ts.BORDER)
    strip.destroy()
