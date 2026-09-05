"""CLASSIC IS FROZEN AT jarvis-v3 4b7d373. This file fails if it drifts.

WHY THIS FILE EXISTS. Hunter, 2026-09-05 02:05: "can we also clean up the
sensors tab and settings area a bit. they look a little unprofessional and
some of text doesnt sit right." He asked about the console he runs, which
is the HOLO look (``theme.DEFAULT_LOOK``). The 09-05 relayout was applied
to BOTH looks and changed the one he never mentioned:

    MEASURED on the photo rig (scripts/ui_shots.py, private Xvfb, S=2),
    classic, against the jarvis-v3 tip:

        frame                     920x1440        1040x1760 (his window)
        26-sensors               175,076 px          202,322 px
        27-sensors-fault         175,222 px          204,647 px
        25-header-worst            1,117 px
        14-settings, crop x300+  13,055 px

    of 1,324,800 px a frame, i.e. 13.2% of the classic sensors page.

So the sensors-page relayout, the reworded reason lines, the amber rules
and the 1-px ink fix are all gated on the look at CALL time
(``sensors_page.restyled``, ``theme.LOOK`` in ``widgets``), and this file
pins the frozen side. It is deliberately written in terms a diff against
the v3 tip can be checked against: the exact sentences, the exact widget
tree, the exact draw geometry.

THE PRICE, STATED. Classic still carries the bug he photographed -- at
920x1440 with both rooms in their worst state the last band row and SAVE
run off the bottom of the stage. That is pinned below
(``test_classic_still_clips_at_his_window_because_it_is_frozen``) so it is
a recorded decision rather than a silent regression. He runs holo.

The Tk half runs only when ``JARVIS_UI_TEST_DISPLAY`` names a private X
display, exactly as tests/test_ui_layout_rules.py does, and refuses his
desktop displays outright.
"""
import os

import pytest

from jarvis.ui import sensors_page as sp
from jarvis.ui import theme
from jarvis.ui import widgets as wg

FORBIDDEN_DISPLAYS = (":0", ":1")
FD_SETSIZE = 1024                    # see tests/test_ui_layout_rules.py
FONT_GLOBALS = ("_FAMILY", "_FAMILY_MONO", "_HAS_DISPLAY", "_DISPLAY")


def font_globals() -> dict:
    return {k: getattr(theme, k) for k in FONT_GLOBALS}


@pytest.fixture(autouse=True)
def _restore_look():
    """theme.resolve_fonts() is a one-way door and select_look() is global;
    every test here changes both, so both are handed back."""
    fonts = font_globals()
    yield
    theme.apply_scale(1.0)
    wg.set_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)
    for k, v in fonts.items():
        setattr(theme, k, v)


# ===================================================== the gate itself
def test_the_gate_reads_the_look_at_call_time():
    """A look token captured at def time freezes the import-time look --
    the exact defect tests/test_theme_look.py exists to catch. The gate is
    a function, and it answers differently after select_look()."""
    theme.select_look("holo")
    assert sp.restyled() is True
    theme.select_look("classic")
    assert sp.restyled() is False
    # and it can be asked about a look it is not currently in
    assert sp.restyled("holo") is True
    assert sp.restyled("classic") is False


def test_the_module_captures_no_look_at_import_time():
    """No module-level name may hold "holo"/"classic" as a decision. The
    look is read inside functions, never bound beside them."""
    import ast
    src = open(sp.__file__).read()
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        text = ast.dump(node)
        assert "theme.LOOK" not in text and "'LOOK'" not in text, \
            ast.unparse(node)


# ================================================ the frozen sentences
def test_the_standing_caption_is_the_v3_sentence_in_classic():
    theme.select_look("classic")
    assert sp.restart_note() == sp.RESTART_NOTE_V3
    assert sp.restart_note() == ("edits apply at the next Jarvis restart — "
                                 "nothing reloads the config")
    theme.select_look("holo")
    assert sp.restart_note() == sp.RESTART_NOTE
    assert sp.restart_note() == "applies at the next Jarvis restart"


def _ladders():
    data = {"zones": {"enabled": True, "rooms": [
        {"name": "office", "enabled": True, "camera_zone": "at the desk",
         "bands": [{"name": "empty space", "near_m": 0.75, "far_m": 2.25},
                   {"name": "at the desk", "near_m": 2.25, "far_m": 3.75}]},
        {"name": "kitchen", "enabled": True, "camera_zone": "",
         "bands": [{"name": "the kitchen", "near_m": 0.75, "far_m": 3.0}]},
    ]}}

    def get_option(key, default=None):
        node = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    return sp.read_ladders(get_option)


def _face():
    return sp.camera_view({"live": True, "faces": 1,
                           "face": {"name": "hunterp", "id_score": 0.71,
                                    "id_ran": True}})


def _blind():
    return sp.camera_view({"live": False, "reason": "sensing"})


def test_every_frozen_reason_line_is_the_v3_sentence():
    """The nine sentences fuse() can print, verbatim off the v3 tip. A
    reworded line is a changed pixel on a page he did not ask about."""
    theme.select_look("classic")
    lad = _ladders()
    office, kitchen = lad.for_room("office"), lad.for_room("kitchen")
    F, B = _face(), _blind()

    def why(**kw):
        kw.setdefault("camera", B)
        kw.setdefault("overrules", True)
        return sp.fuse(**kw).why

    assert why(present=True, distance_m=3.0, zmap=None) == (
        "there is no ladder for this room in zones.rooms, so a range "
        "cannot be given a name")
    assert why(present=True, distance_m=3.0, zmap=office) == \
        "a target inside 'at the desk'"
    assert why(present=True, distance_m=None, zmap=office) == (
        "presence with no usable range — someone is in the room, the "
        "band is unknown")
    assert why(present=True, distance_m=5.9, zmap=office) == (
        "5.90 m is in a gap between the bands, so the room is the only "
        "honest answer")
    assert why(present=False, distance_m=None, zmap=office) == \
        "the radar reads empty"
    assert why(present=None, distance_m=None, zmap=office) == \
        "neither leg has an opinion — this is not an empty room"
    # the camera clause is APPENDED with a semicolon, which is the clause
    # that made the line wrap into a second grey row
    assert why(present=True, distance_m=3.0, zmap=office, camera=F,
               overrules=False) == (
        "a target inside 'at the desk'; the camera sees a face but is not "
        "allowed to overrule")
    assert why(present=True, distance_m=2.0, zmap=kitchen, camera=F) == (
        "a target inside 'the kitchen'; the camera sees a face but "
        "'kitchen' has no camera zone in zones.rooms")
    assert why(present=True, distance_m=3.0, zmap=office, camera=F) == \
        "the camera can see a face; the radar's range is not asked"
    assert why(present=None, distance_m=None, zmap=office, camera=F) == \
        "the radar has no opinion; the camera can see a face"


def test_a_silent_radar_is_still_named_in_classic_when_a_face_was_seen():
    """v3's source line only checked saw_face, so a face in a room with no
    camera zone printed "NO OPINION · radar" under a radar that had said
    nothing. Fixed in holo; frozen in classic."""
    lad = _ladders()
    theme.select_look("classic")
    v = sp.fuse(present=None, distance_m=None, camera=_face(),
                zmap=lad.for_room("kitchen"), overrules=True)
    assert v.zone == sp.NO_OPINION and v.source == "radar"
    theme.select_look("holo")
    v = sp.fuse(present=None, distance_m=None, camera=_face(),
                zmap=lad.for_room("kitchen"), overrules=True)
    assert v.zone == sp.NO_OPINION and v.source == ""


def test_no_opinion_is_amber_in_classic_and_faint_in_holo():
    theme.select_look("classic")
    assert sp.presence_words(None) == ("NO OPINION", sp.TONE_WARN)
    theme.select_look("holo")
    assert sp.presence_words(None) == ("NO OPINION", sp.TONE_FAINT)
    # the two states that never moved
    for look in ("classic", "holo"):
        theme.select_look(look)
        assert sp.presence_words(True) == ("PRESENT", sp.TONE_OK)
        assert sp.presence_words(False) == ("EMPTY", sp.TONE_MUTED)


# ================================== the unrecognised face: a decision
def test_an_unrecognised_face_is_not_a_fault_in_holo():
    """UNCOVERED BY ANY TEST until now, and it broke the page's one rule.
    sensors_page's own docstrings say three times that amber is reserved
    for a fault and is "the only orange thing on the page" -- and a camera
    that saw a face it could not name painted "FACE UNKNOWN 0.42" amber
    beside it. A camera that reports a person and a confidence HAS
    answered; nothing is broken, so nothing is amber, and it reads like
    the other two answered-but-no-identity lines. Classic is frozen.
    """
    view = sp.camera_view({"live": True, "faces": 1,
                           "face": {"name": "", "id_score": 0.42,
                                    "id_ran": True}})
    theme.select_look("holo")
    text, tone = sp.camera_text(view)
    assert text == "FACE  UNKNOWN  0.42"
    assert tone == sp.TONE_MUTED
    theme.select_look("classic")
    assert sp.camera_text(view)[1] == sp.TONE_WARN


def test_amber_on_the_holo_page_means_a_fault_and_nothing_else():
    """The whole camera vocabulary, every state, in the look he runs: not
    one of them may be amber. The fault line is the only orange thing."""
    theme.select_look("holo")
    states = (
        {},                                                  # no camera
        {"live": False, "reason": "sensing"},                # off
        {"live": True, "faces": 0},                          # nobody
        {"live": True, "faces": 1, "face": {"id_ran": False}},
        {"live": True, "faces": 1,
         "face": {"name": "", "id_score": 0.42, "id_ran": True}},
        {"live": True, "faces": 1,
         "face": {"name": "hunterp", "id_score": 0.71, "id_ran": True}},
    )
    for st in states:
        present = bool(st)
        tone = sp.camera_text(sp.camera_view(st or None, present=present))[1]
        assert tone != sp.TONE_WARN, st
    # and the fault line still is amber, so the rule has a subject
    row = sp.RoomRow(name="office", label="the office",
                     presence_word="NO OPINION", presence_tone=sp.TONE_FAINT,
                     distance_text=sp.DASH, distance_m=None,
                     rtt_text=sp.DASH,
                     fault="no answer from the sensor (2 in a row)",
                     camera_text="NO FACE", camera_tone=sp.TONE_MUTED,
                     verdict=sp.Verdict(sp.NO_OPINION, "NO OPINION", "",
                                        "neither leg has an opinion"))
    lines = sp.explanation_lines(row)
    assert lines[-1][1] == sp.TONE_WARN


# ====================================================== on a display
def _display() -> str:
    d = (os.environ.get("JARVIS_UI_TEST_DISPLAY") or "").strip()
    if not d:
        return ""
    base = d.split(".")[0]
    if base in FORBIDDEN_DISPLAYS or base.split(":")[-1] in ("0", "1"):
        pytest.fail("JARVIS_UI_TEST_DISPLAY=%r is a desktop display; this "
                    "file builds windows and will not open one on his "
                    "screen" % d)
    return d


STAGE_W, STAGE_H_WITH_PANE, SCALE = 920, 852, 2.0


@pytest.fixture
def root():
    display = _display()
    if not display:
        pytest.skip("set JARVIS_UI_TEST_DISPLAY=:9N (a private Xvfb) to run "
                    "the measured frozen-classic tests")
    import tkinter as tk
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
        r = tk.Tk(screenName=display)
    except tk.TclError as exc:
        pytest.skip("no X server at %s: %s" % (display, exc))
    r.geometry("%dx%d+0+0" % (STAGE_W, 1440))
    theme.resolve_fonts(r)
    theme.apply_scale(SCALE)
    wg.set_scale(SCALE)
    yield r
    try:
        r.destroy()
    except Exception:                      # noqa: BLE001 - teardown
        pass


def _fake_services():
    from types import SimpleNamespace
    specs = [{"name": "office", "url": "http://192.0.2.10",
              "label": "the office"},
             {"name": "kitchen", "url": "http://192.0.2.11",
              "label": "the kitchen"}]
    data = {
        "presence": {"room_sensor_enabled": True, "rooms": specs,
                     "camera_overrules": True},
        "camera": {"room": "office"},
        "zones": {"enabled": True, "rooms": [
            {"name": "office", "enabled": True, "camera_zone": "at the desk",
             "bands": [{"name": "empty space", "near_m": 0.75, "far_m": 2.25},
                       {"name": "at the desk", "near_m": 2.25,
                        "far_m": 3.75}]},
            {"name": "kitchen", "enabled": True, "camera_zone": "",
             "bands": [{"name": "the kitchen", "near_m": 0.75, "far_m": 3.0},
                       {"name": "at the door", "near_m": 3.0,
                        "far_m": 3.75}]},
        ]},
    }

    def get_option(key, default=None):
        node = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    return SimpleNamespace(get_option=get_option, sensing=None)


def _worst_case_rows(page):
    readings = [sp.Reading(name=s.name, label=s.label, url=s.url, present=None,
                           status={"url": s.url, "paused": True,
                                   "retry_in_s": 29.0, "fails": 4},
                           at=1.0)
                for s in page.specs]
    camera = {"live": False, "reason": "sensing",
              "detail": "CAMERA OFF · curfew until 7 am"}
    return sp.page_rows(readings, camera, page.ladders, page.overrules,
                        camera_room=page.camera_room)


def _build_page(root, look: str, height: int = STAGE_H_WITH_PANE,
                worst: bool = True):
    import tkinter as tk
    theme.select_look(look)
    host = tk.Frame(root, width=STAGE_W, height=height)
    host.pack_propagate(False)
    host.pack()
    page = sp.SensorsPage(host, services=_fake_services())
    page.place(in_=host, x=0, y=0, relwidth=1.0, height=height)
    root.update_idletasks()
    if worst:
        for row in _worst_case_rows(page):
            page.apply_row(row)
    root.update_idletasks()
    return page


def test_the_classic_page_has_no_scrolling_view_at_all(root):
    """The whole scroll apparatus -- the canvas, the body window, the cyan
    thumb, the wheel grab -- is holo's. v3 packed one plain column."""
    page = _build_page(root, "classic")
    assert page._canvas is None and page._body is None
    assert page._thumb is None
    assert page.overflow_px() == 0        # never scrolls, never claims to
    assert page._rooms_bands == []
    # and the wheel is never grabbed, so a hidden page cannot eat scroll
    page._grab_wheel()
    assert not root.bind_all("<Button-4>")


def test_the_classic_overrule_toggle_is_the_last_row_of_the_band_list(root):
    """v3 put the one setting on this page at the end of the tune column,
    which is where it scrolled off. Holo moved it beside SAVE. Frozen."""
    page = _build_page(root, "classic")
    tune = page._bands[-1]["lo"].master.master
    assert page._overrule.master.master is tune
    assert page._overrule.master is not page._foot
    # SAVE and the caption are still below it -- checked by PACK ORDER,
    # not by geometry: at his window this classic column overflows the
    # stage and the foot is clipped, which is the frozen bug itself
    # (test_classic_still_clips_at_his_window_because_it_is_frozen).
    kids = list(page.winfo_children())
    assert kids.index(page._foot) > kids.index(tune)


def test_a_classic_room_block_is_the_v3_five_widget_tree(root):
    """Header: name + PRESENCE + range + rtt. Then two captioned sub-rows
    (camera, verdict) and two sentences packed straight onto the block."""
    import tkinter.font as tkfont
    page = _build_page(root, "classic")
    block = page._blocks["office"]
    assert block.source is None           # holo's ", · camera" chip
    assert block._fault_row is None       # holo's captioned fault row
    assert block.presence.master is block.name.master   # both in the head
    assert block.why.master is block      # not in a captioned row
    assert block.fault.master is block
    # v3 drew the LEG VALUES in the header's own size, so the camera line
    # shouted as loudly as the answer; holo drops them to the caption size,
    # which is the "two sizes in a block" rule.
    def size(w):
        return abs(tkfont.Font(font=w.cget("font")).cget("size"))
    assert size(block.camera) == size(block.name)
    assert size(block.why) < size(block.camera)
    # the caption column is 14 characters wide, not holo's 9
    caption = block.camera.master.winfo_children()[0]
    assert caption.cget("width") == 14
    # and the range is monospace whatever it says, dash included
    mono = theme.mono(1)[0]
    assert block.distance.cget("text") == sp.DASH
    assert tkfont.Font(font=block.distance.cget("font")).cget("family") == mono


def test_a_classic_band_bar_draws_a_full_track_with_two_end_stops(root):
    """v3's strip: one line the full width and an end stop at each end. The
    holo bar draws the band's SHARE of the room instead, and no end stops.
    Measured off the canvas items, never off a picture."""
    page = _build_page(root, "classic")
    bar = page._bands[0]["bar"]
    bar.set_span((0.25, 0.5))             # holo's span: no ink in classic
    root.update_idletasks()
    lines = [i for i in bar.find_all() if bar.type(i) == "line"]
    assert len(lines) == 3                # the rail and the two end stops
    assert not [i for i in bar.find_all() if bar.type(i) == "rectangle"]
    rail = bar.coords(lines[0])
    pad = wg.px(6)
    assert rail[0] == pytest.approx(pad)
    assert rail[2] == pytest.approx(max(bar.winfo_width(), wg.px(40)) - pad)


def test_a_classic_button_and_toggle_keep_the_v3_ink_edges(root):
    """The 1-px fix (canvas_size) moved every button and toggle in the tree
    by up to 2 px: 13,055 px in the classic drawer crop and all 1,117 px of
    the classic chat header, MEASURED. It is holo's now."""
    import tkinter as tk
    for look, holo in (("classic", False), ("holo", True)):
        theme.select_look(look)
        host = tk.Frame(root, bg="#000")
        host.pack()
        btn = wg.RoundButton(host, text="SAVE", kind="default", bg="#000")
        btn.pack()
        tog = wg.Toggle(host, value=True, bg="#000")
        tog.pack()
        root.update_idletasks()
        btn._draw()
        tog._draw()
        # the button's ring: holo lays it out on the INTERIOR width, which
        # is 2 px narrower than winfo_width() (the focus ring, both sides)
        ring = [i for i in btn.find_all() if btn.type(i) != "text"][0]
        right = max(btn.coords(ring)[0::2])
        interior = wg.canvas_size(btn, btn._btn_w, btn._btn_h)[0]
        outer = max(btn.winfo_width(), btn._btn_w)
        assert outer > interior, (outer, interior)
        assert right == pytest.approx((interior if holo else outer)
                                      - max(1, wg.px(1)) - 1), look
        # the toggle's track ends on that same inset in holo, on W - px(2)
        # in classic -- the 6 px the control column was out of true
        track = tog.find_all()[0]
        want = (tog.W - max(1, wg.px(1)) - 1) if holo else (tog.W - wg.px(2))
        assert max(tog.coords(track)[0::2]) == pytest.approx(want), look
        host.destroy()


def test_classic_still_clips_at_his_window_because_it_is_frozen(root):
    """THE PRICE OF FREEZING CLASSIC, recorded as a fact rather than left
    to be rediscovered. In the worst case he photographed -- 920x1440, the
    camera pane packed, both rooms dark with a fault line and a reason line
    -- the classic page asks for more height than the stage has, so the
    last band row and SAVE run off the bottom. That is exactly the bug the
    09-05 pass fixed, and the fix is holo's, because holo is what he runs
    (theme.DEFAULT_LOOK) and he never asked for classic to change.

    If this test ever fails, classic has been RELAID OUT: either revisit
    the freeze with him or re-measure the whole frozen set.
    """
    page = _build_page(root, "classic")
    stage_bottom = page.winfo_rooty() + STAGE_H_WITH_PANE
    save = page._save_btn
    # Tk does not draw what it has no room for: the button comes back
    # unmapped (winfo_rooty 0, height 1) rather than merely low down, which
    # is what "SAVE is gone entirely" looked like on his photograph.
    off = (not save.winfo_ismapped()
           or save.winfo_rooty() + save.winfo_height() > stage_bottom)
    assert off, ("classic now FITS his window -- the frozen v3 layout did "
                 "not. Something relaid the classic page out.")
    # holo, the same case, fits with room to spare
    holo = _build_page(root, "holo")
    assert holo.overflow_px() == 0
    hsave = holo._save_btn
    assert hsave.winfo_rooty() + hsave.winfo_height() <= \
        holo.winfo_rooty() + holo.winfo_height()


def _build_drawer(root, look: str):
    import tkinter as tk
    from jarvis.ui import views
    theme.select_look(look)
    host = tk.Frame(root, width=STAGE_W, height=1440)
    host.pack_propagate(False)
    host.pack()
    drawer = views.SettingsDrawer(host, services=None)
    drawer.place(in_=host, relx=1.0, y=0, x=0, anchor="ne", relheight=1.0,
                 width=drawer.WIDTH)
    root.update_idletasks()
    return drawer


def test_the_classic_knightfall_row_still_overruns_because_it_is_frozen(root):
    """THE PRICE OF FREEZING CLASSIC, second entry (2026-09-05).

    Hunter, on the merged tip: "the knightfall text doesnt fit in its
    slot". MEASURED on :93 at S=2: the row asks for 676 px of a 576 px
    slot, so Tk squeezes the Open button from 125 px to 25 and draws the
    word as a sliver; the button under it asks for 578 and is clipped at
    576. The fix is HOLO's -- holo is what he runs (theme.DEFAULT_LOOK) --
    and classic keeps the 09-04 row, wording, widths and overflow, exactly
    as it keeps the clipped sensors page above.

    If this test ever fails, the classic drawer has been relaid out:
    revisit the freeze with him or re-measure the whole frozen set."""
    from jarvis.ui.views import SettingsDrawer
    drawer = _build_drawer(root, "classic")
    row = drawer._knightfall_entry.master
    assert row.winfo_reqwidth() > row.winfo_width(), (row.winfo_reqwidth(),
                                                      row.winfo_width())
    open_btn = drawer._knightfall_open
    assert open_btn.winfo_width() < open_btn.winfo_reqwidth()
    # the v3 words and the v3 box, untouched
    label = [k for k in row.winfo_children()
             if k.winfo_class() == "Label"][0]
    assert label.cget("text") == "Knightfall code"
    assert drawer._knightfall_new._text == "Email me a new Knightfall code"
    assert int(drawer._knightfall_entry.cget("width")) == 12
    assert label.cget("text") != SettingsDrawer.KNIGHTFALL_LABEL_HOLO
    # ...and holo, the same row, fits with room to spare
    holo = _build_drawer(root, "holo")
    hrow = holo._knightfall_entry.master
    assert hrow.winfo_reqwidth() <= hrow.winfo_width()
    assert holo._knightfall_open.winfo_width() >= \
        holo._knightfall_open.winfo_reqwidth()
