"""FOUND 2026-09-02 (Hunter, by voice): "if jarvis is in standby mode and i
try and drag him he jumps back to where he was being dragged from".

THE STALE ANCHOR IS THE BUG.  Standby runs a burn-in walk by moving the
WHOLE window -- the reactor's decor is baked at fixed canvas coordinates,
so the window is the only thing that can drift (console_mode.py:37-42).
`_on_console_mode` captures `_standby_origin` once on the way in, and every
tick sets the window to `origin + drift_offset(elapsed)`: a position
recomputed from scratch each second, deliberately, so a late tick lands
where it should instead of accumulating error (console_mode.py:138-152).

His drag writes `root.geometry("+x+y")` straight onto the window
(`_move_drag`) and left that anchor untouched, so BOTH ends of standby
threw the drag away:

  * the NEXT DRIFT TICK -- at most a second later -- recomputed from the
    stale origin and yanked the window back to within a few pixels of
    where the drag began.  This is the jump he reported.
  * LEAVING STANDBY called `_move_to(*_standby_origin)` ("undo the burn-in
    walk") and put the window back on the pre-drag anchor for good.  That
    is also the position `_on_close` saves as `window_geometry` -- it stops
    the console creeping across the desk one quit at a time, and it saves
    AFTER `modes.stop()` has restored the origin -- so the reposition did
    not even survive to the next launch.

Re-anchoring to the raw dropped position would not fix it either: the
window carries the drift displacement, so `origin = where he dropped it`
makes the very next tick jump by the whole accumulated offset.  The anchor
has to be `dropped position - the drift currently baked into the window`,
which is precisely the last offset `_on_console_drift` applied.

Display-free, the way tests/test_holo_geometry.py is: the shipping methods
are taken UNBOUND off MainWindow and driven against a fake root that
records geometry strings.  No Tk root is created -- a real toplevel on the
live display is the window churn behind the 2026-08-26 desktop freeze.
"""
import re
from datetime import datetime
from types import SimpleNamespace

import pytest

from jarvis.ui import console_mode as cm
from jarvis.ui.main_window import MainWindow

# device px, straight off the fake root below
START = (100, 200)
SIZE = (520, 880)

_GEOM = re.compile(r"(?:(\d+)x(\d+))?(?:([+-]\d+)([+-]\d+))?$")


class _Root:
    """The four calls the drag/standby path makes on `self.root`.

    `geometry()` splits size from position the way the real Tk geometry
    manager does -- "+x+y" moves without resizing, "WxH" resizes without
    moving -- because that split is the whole reason `_grip_drag` (a SE
    resize) cannot disturb the standby anchor while `_move_drag` can.
    """

    def __init__(self, x=START[0], y=START[1], w=SIZE[0], h=SIZE[1]):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.specs = []                  # every geometry string, in order

    # position readers (_move_start and the standby capture use these)
    def winfo_x(self):
        return self.x

    def winfo_y(self):
        return self.y

    winfo_rootx = winfo_x                # borderless: no frame offset
    winfo_rooty = winfo_y

    def geometry(self, spec=None):
        if spec is None:
            return f"{self.w}x{self.h}+{self.x}+{self.y}"
        self.specs.append(spec)
        m = _GEOM.fullmatch(spec)
        assert m, f"not a Tk geometry string: {spec!r}"
        if m.group(1):
            self.w, self.h = int(m.group(1)), int(m.group(2))
        if m.group(3):
            self.x, self.y = int(m.group(3)), int(m.group(4))

    @property
    def pos(self):
        return (self.x, self.y)


class _Event:
    """A <B1-Motion> / <ButtonPress-1> event: the handlers read only the
    root-relative pointer position."""

    def __init__(self, x_root, y_root):
        self.x_root, self.y_root = x_root, y_root


class _Console:
    """The drag + standby seam of MainWindow with nothing else attached.

    The methods are taken unbound from MainWindow, so these tests exercise
    the shipping code rather than a re-implementation of it.  `reactor` and
    `transcript` are deliberately bare objects: `_on_console_mode` guards
    both with `except AttributeError`, so a stub that does not answer
    `set_speed_scale` / `pause_atmo` takes the same path a console whose
    widgets are not built yet takes.
    """

    _move_start = MainWindow._move_start
    _move_drag = MainWindow._move_drag
    _grip_drag = MainWindow._grip_drag
    _move_to = MainWindow._move_to
    _on_console_drift = MainWindow._on_console_drift
    _on_console_mode = MainWindow._on_console_mode

    def __init__(self):
        self.root = _Root()
        self._drag_off = None
        self._geom_ts = 0.0
        self._standby_origin = None
        self._standby_drift = (0, 0)
        self._min_w, self._min_h = 460, 720
        self.modes = None
        self.room = SimpleNamespace(set_mode=lambda mode: None)
        self.reactor = object()
        self.transcript = object()
        self.modes_seen = []

    def _set_footer_hidden(self, hidden):
        self.modes_seen.append(hidden)

    # ---------------------------------------------------------- helpers
    def drag_to(self, x, y):
        """Press on the header where the window is now, then move the
        pointer by the offset that lands the window's top-left on (x, y).

        `_geom_ts` is zeroed first: `_move_drag` drops any motion inside
        16 ms of the last one (the 60 Hz cap), and two calls in a test are
        microseconds apart, not the tens of ms a human drag delivers.
        """
        px, py = self.root.pos
        press = (px + 40, py + 12)       # somewhere on the header
        self._geom_ts = 0.0
        self._move_start(_Event(*press))
        self._geom_ts = 0.0
        self._move_drag(_Event(press[0] + (x - px), press[1] + (y - py)))


def _standby_at(origin_drift=(0, 0)):
    """A console parked in standby with `origin_drift` already applied --
    i.e. the state a drift tick leaves behind."""
    con = _Console()
    con._on_console_mode(cm.STANDBY)
    con._on_console_drift(*origin_drift)
    return con


# ------------------------------------------------- path 1: the drift tick
def test_the_next_drift_tick_does_not_yank_a_dragged_window_back():
    """His report, as positions.  The window is at (100, 200); one minute
    of standby has walked it to (103, 203); he drags it to (163, 233).  The
    tick two minutes later must land it at (169, 238) -- his position plus
    the six px the walk moved on -- and NOT at (109, 208), which is the old
    anchor plus the same offset, i.e. back where the drag started."""
    assert cm.drift_offset(60.0) == (3, 3)      # the walk at 3 px/min…
    assert cm.drift_offset(180.0) == (9, 8)     # …two minutes further on
    con = _standby_at((3, 3))
    assert con.root.pos == (103, 203)

    con.drag_to(163, 233)
    assert con.root.pos == (163, 233)           # the drag itself works today

    con._on_console_drift(9, 8)
    assert con.root.pos == (169, 238), (
        "the drift tick recomputed from the pre-drag anchor and threw the "
        "window back to (109, 208)")


def test_a_drag_before_the_first_drift_tick_still_holds():
    """Entering standby applies a (0, 0) offset, so the window can be
    dragged before the walk has moved it at all.  From (100, 200) to
    (140, 260), then the first real tick (3, 3): (143, 263)."""
    con = _standby_at()
    assert con.root.pos == START

    con.drag_to(140, 260)
    con._on_console_drift(3, 3)
    assert con.root.pos == (143, 263)


def test_dragging_twice_in_standby_keeps_both_moves():
    """Each drag re-anchors against the drift standing at that moment, so
    the second one does not inherit the first one's error."""
    con = _standby_at((3, 3))
    con.drag_to(163, 233)
    con._on_console_drift(6, 5)                 # a tick between the drags
    assert con.root.pos == (166, 235)

    con.drag_to(400, 300)
    con._on_console_drift(9, 8)
    assert con.root.pos == (403, 303)           # +3, +5 on from (6, 5)


def test_a_drift_tick_mid_drag_does_not_fight_the_cursor():
    """The 1 Hz tick can land between two motion events.  The drag is
    absolute (pointer minus the grab offset), so the next motion puts the
    window back under his cursor whatever the tick did -- and the anchor it
    leaves behind must be the one that belongs to that final position."""
    con = _standby_at((3, 3))
    px, py = con.root.pos
    con._geom_ts = 0.0
    con._move_start(_Event(px + 40, py + 12))

    con._geom_ts = 0.0
    con._move_drag(_Event(px + 90, py + 42))    # half way: window at +50,+30
    con._on_console_drift(4, 4)                 # a tick, mid-drag
    con._geom_ts = 0.0
    con._move_drag(_Event(px + 140, py + 72))   # he keeps going: +100,+60

    assert con.root.pos == (203, 263)           # under the cursor, exactly
    con._on_console_drift(9, 8)
    assert con.root.pos == (208, 267)           # (9, 8) on from the (4, 4)


# ---------------------------------------------- path 2: leaving standby
def test_waking_from_standby_undoes_the_walk_but_not_his_move():
    """`_move_to(*_standby_origin)` is right and stays: what standby applied
    must come off.  What HE applied must not.  Dragged from (103, 203) to
    (163, 233) while the walk stood at (3, 3), the woken window belongs at
    (160, 230) -- the original (100, 200) plus his exact +60/+30 -- and that
    is the geometry `_on_close` goes on to save."""
    con = _standby_at((3, 3))
    con.drag_to(163, 233)

    con._on_console_mode(cm.ACTIVE)
    assert con.root.pos == (160, 230), (
        "waking restored the pre-drag anchor and discarded the reposition")
    assert con._standby_origin is None
    # _on_close saves root.geometry() AFTER modes.stop() has woken the
    # console, so this string is what the next launch reads back.
    assert con.root.geometry() == "520x880+160+230"


def test_a_second_standby_stint_anchors_where_he_left_it():
    """The origin is captured fresh on the way in, so the drag survives a
    full standby -> active -> standby round trip."""
    con = _standby_at((3, 3))
    con.drag_to(163, 233)
    con._on_console_mode(cm.ACTIVE)
    con._on_console_drift(0, 0)                 # the tick outside standby

    con._on_console_mode(cm.STANDBY)
    assert con._standby_origin == (160, 230)
    con._on_console_drift(3, 3)
    assert con.root.pos == (163, 233)


# ------------------------------------------- the paths that must not move
def test_a_drag_outside_standby_is_untouched():
    """Active mode has no anchor at all; the drag is a bare geometry write
    and must stay one."""
    con = _Console()
    con._on_console_drift(0, 0)                 # what every non-standby tick sends
    assert con.root.pos == START
    assert con.root.specs == []                 # …and it moves nothing

    con.drag_to(300, 400)
    assert con.root.pos == (300, 400)
    assert con._standby_origin is None
    assert con.root.specs == ["+300+400"]


def test_a_resize_in_standby_leaves_the_anchor_alone():
    """`_grip_drag` is the SE corner: it writes WxH only, and the window's
    top-left does not move, so the anchor is still correct.  Asserted rather
    than assumed -- a resize that quietly re-anchored would drift the console
    by the offset on the next tick."""
    con = _standby_at((3, 3))
    con._geom_ts = 0.0
    con._grip_drag(_Event(con.root.x + 700, con.root.y + 1000))
    assert con.root.specs[-1] == "700x1000"
    assert con.root.pos == (103, 203)           # unmoved
    assert con._standby_origin == START

    con._on_console_drift(9, 8)
    assert con.root.pos == (109, 208)
    assert con.root.geometry() == "700x1000+109+208"


# -------------------------------------------- against the real state machine
def test_the_real_console_walks_on_from_where_he_put_it():
    """The same story driven by the shipping ConsoleModes on a frozen
    clock, so the offsets come from the machine rather than from literals
    in this file."""
    con = _Console()
    now = [1000.0]
    modes = cm.ConsoleModes(
        after=lambda ms, fn: None,              # no Tk loop; tick() by hand
        idle_fn=lambda: 9999.0,                 # the desk has been empty
        on_mode=con._on_console_mode,
        on_drift=con._on_console_drift,
        option=lambda key, default: default,
        clock=lambda: now[0],
        now=datetime.now)
    con.modes = modes

    assert modes.tick() == cm.STANDBY
    assert con._standby_origin == START
    now[0] += 60.0
    modes.tick()
    assert con.root.pos == (103, 203)

    con.drag_to(163, 233)
    now[0] += 120.0
    modes.tick()
    assert con.root.pos == (169, 238)

    # …and waking gives him back exactly the window he dropped, minus the
    # walk: (100, 200) + his (60, 30).
    modes.stop()
    assert modes.mode == cm.ACTIVE
    assert con.root.pos == (160, 230)


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
