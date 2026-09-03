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

That subtraction has a consequence a reviewer caught on the way in: a drop
made ENTIRELY on screen can anchor at a NEGATIVE coordinate (park it flush
in the corner while the walk is at its rightmost and the un-drifted home is
off the edge).  The anchor is right to go there and is deliberately not
clamped -- see the corner-park test below -- but the position it saves is
spelled "+-37+-17", and `_pick_geometry` rejected exactly that, handing him
the default window back.  The last four tests cover that round trip.

Display-free, the way tests/test_holo_geometry.py is: the shipping methods
are taken UNBOUND off MainWindow and driven against a fake root that
records geometry strings.  No Tk root is created -- a real toplevel on the
live display is the window churn behind the 2026-08-26 desktop freeze.
"""
import re
from datetime import datetime
from types import SimpleNamespace

import pytest

from jarvis.config import CONFIG
from jarvis.ui import console_mode as cm
from jarvis.ui.main_window import MainWindow

# device px, straight off the fake root below
START = (100, 200)
SIZE = (520, 880)

# Only the form BOTH writers actually produce -- `f"+{x}+{y}"` -- with the
# coordinates allowed to be negative, because that is what real Tk takes:
# measured on a scratch Xvfb, `geometry("+-34+-21")` is accepted, `geometry()`
# echoes "520x880+-34+-21" and `winfo_x()` is -34.  "-34-21" is deliberately
# NOT accepted here: to Tk that is right/bottom gravity and lands near the far
# corner (846, 699 on a 1400x1600 screen), so a writer that ever emitted it
# would be moving the window somewhere else entirely and must fail loudly.
_GEOM = re.compile(r"(?:(\d+)x(\d+))?(?:\+(-?\d+)\+(-?\d+))?$")


class _Root:
    """The handful of calls the drag/standby/geometry path makes on
    `self.root`.

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

    def winfo_screenheight(self):
        return 1600                      # _default_geometry clamps h to 90%

    def geometry(self, spec=None):
        if spec is None:
            return f"{self.w}x{self.h}+{self.x}+{self.y}"
        self.specs.append(spec)
        m = _GEOM.fullmatch(spec)
        assert m, f"not a Tk geometry string: {spec!r}"
        if m.group(1) is not None:
            self.w, self.h = int(m.group(1)), int(m.group(2))
        if m.group(3) is not None:       # "0" is a legal coordinate
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
    _preview_apply = MainWindow._preview_apply
    _pick_geometry = MainWindow._pick_geometry
    _default_geometry = MainWindow._default_geometry
    # 2026-09-03: standby hides the tab row with the footer and shuts a
    # SENSORS page left open behind the clock. Bound, not stubbed -- the
    # console with no strip built is a state the shipping method handles
    # (`tabs` is None when _build_tabs failed), so it is worth taking that
    # path here rather than pretending the call does not happen.
    _set_tabs_hidden = MainWindow._set_tabs_hidden

    def __init__(self):
        self.root = _Root()
        self._drag_off = None
        self._geom_ts = 0.0
        self._standby_origin = None
        self._standby_drift = (0, 0)
        self._tabs_hidden = False
        self.tabs = None                 # _build_tabs failed / not built yet
        self._min_w, self._min_h = 460, 720
        self.modes = None
        self.room = SimpleNamespace(set_mode=lambda mode: None)
        self.reactor = object()
        self.transcript = object()
        # The camera pane, unbuilt -- the same state a console takes before
        # _build_footer has run, and the path _preview_apply returns from
        # at once. Nothing on the drag/anchor path touches it.
        self.preview = None
        self.preview_worker = None

    def _set_footer_hidden(self, hidden):
        """Stubbed, not recorded: the real one pack_forgets two widgets and
        nothing on the drag/anchor path reads it back."""

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


# ------------------------------ the anchor may legitimately land off-screen
def test_a_corner_park_anchors_off_screen_and_still_reloads_next_launch(
        monkeypatch):
    r"""The re-anchor subtracts the drift, so a drop made ENTIRELY on screen
    can put the anchor at a negative coordinate: parked flush at (5, 4) while
    the walk stood at its rightmost (+42, +21), the un-drifted home is
    (-37, -17).

    That anchor is deliberately NOT clamped. Clamping it to (0, 0) would
    desynchronise it from the drift the window is actually wearing and the
    very next tick would jump the window by the difference -- the snap-back
    this whole file is about, in miniature. Clamping the WAKE instead would
    break a console he has deliberately parked half off the edge, which
    a414152 preserves across standby. So the negative position is allowed to
    exist, and the job is to make it survive a quit.

    `_on_close` saves it in Tk's own spelling -- sign of the gravity first,
    then the coordinate, "+-37+-17" (measured on a scratch Xvfb:
    `geometry("+-34+-21")` is accepted, echoes "520x880+-34+-21",
    `winfo_x()` is -34). `_pick_geometry` used to reject exactly that string,
    because `[+-]\d+` cannot match "+-37", and fall back to the DEFAULT
    geometry -- so one corner park cost him his saved SIZE as well as his
    position at the next launch.
    """
    peak = cm.drift_offset(1190.0)               # the walk at its rightmost
    assert peak == (42, 21)
    assert peak[0] == cm.DRIFT_RADIUS == max(          # …genuinely the far
        cm.drift_offset(t)[0] for t in range(0, 6000))  # side of the circle
    con = _standby_at(peak)
    assert con.root.pos == (142, 221)
    con._geom_ts = 0.0                           # …on a panel he had resized
    con._grip_drag(_Event(con.root.x + 700, con.root.y + 1000))

    con.drag_to(5, 4)                            # flush into the corner
    assert con._standby_origin == (-37, -17)     # his drop minus the walk

    con._on_console_mode(cm.ACTIVE)              # what _on_close does first
    saved = con.root.geometry()
    assert saved == "700x1000+-37+-17"

    monkeypatch.setattr(CONFIG, "window_geometry", saved)
    assert con._pick_geometry() == saved         # verbatim: size AND position
    assert saved != con._default_geometry()      # …which the fallback loses


def test_a_left_edge_drag_while_awake_reloads_with_his_size(monkeypatch):
    """The same `_pick_geometry` hole, reached the way a414152 already could:
    `_move_drag` writes "+-20+300" the moment he drags the panel's left edge
    past x=0 with the console awake, and that is what gets saved. Nothing to
    do with standby -- which is why the fix is in the reader, not the writer.
    """
    con = _Console()
    con.drag_to(-20, 300)
    assert con.root.specs == ["+-20+300"]        # verbatim, as Tk wants it
    monkeypatch.setattr(CONFIG, "window_geometry", con.root.geometry())
    assert con._pick_geometry() == "520x880+-20+300"


def test_a_negative_position_survives_the_pre_hidpi_rescale(monkeypatch):
    """A pre-HiDPI size is replaced by the scaled default and the POSITION is
    kept -- the negative form has to come through that branch intact too."""
    con = _Console()
    monkeypatch.setattr(CONFIG, "window_geometry", "300x400+-37+-17")
    assert con._pick_geometry() == con._default_geometry() + "+-37+-17"


def test_geometry_that_is_not_a_geometry_still_falls_back(monkeypatch):
    """The negative control for the widened pattern: it must not have turned
    into "accept anything"."""
    con = _Console()
    for junk in ("", "garbage", "520x880+", "520x880+-+3", "520x-880+1+2",
                 "520x880+1+2 "):
        monkeypatch.setattr(CONFIG, "window_geometry", junk)
        assert con._pick_geometry() == con._default_geometry(), junk


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
