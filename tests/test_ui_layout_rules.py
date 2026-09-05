"""Layout rules for the SENSORS page and the settings drawer (2026-09-05).

Hunter: "can we also clean up the sensors tab and settings area a bit.
they look a little unprofessional and some of text doesnt sit right."

Two halves. The first is Tk-free and always runs: the pure rules the
widgets render from -- the slider's number formatting and snapping, the
one-line budget on every explanation sentence, which explanation line may
be amber, and which readout is drawn in the monospace face. The second
builds the REAL widgets on a display and measures them, and runs ONLY when
``JARVIS_UI_TEST_DISPLAY`` names a private X display (the photo rig's own,
e.g. ``:94``). It refuses his desktop displays the way the rig does: a
window opened on ``:1`` is a window on the screen he is using.
"""
import os
import re

import pytest

from jarvis.ui import sensors_page as sp
from jarvis.ui import theme
from jarvis.ui import widgets as wg

FORBIDDEN_DISPLAYS = (":0", ":1")

# select(2)'s fd_set, which libX11 still uses. An X connection whose socket
# lands on a descriptor at or above this cannot be select()ed, and glibc's
# fortify check turns that into a bare abort() -- the process dies with no
# Python frame and no X error message.
#
# MEASURED 2026-09-05, and it is not hypothetical here: run under the WHOLE
# suite in one process this file dumped core on the first test that builds a
# root. The suite holds 1194 open descriptors by the time it reaches this
# file (583 through tests/test_[a-m]*.py, where the same tests pass), and a
# 12-line pytest file that opens 1100 /dev/null handles and then calls
# tk.Tk() reproduces the abort on its own with nothing of Jarvis loaded.
#
# So the guard is a SKIP with the reason on it rather than a crash: the leak
# is the suite's, the wall is libX11's, and neither is something a layout
# test can fix. Run this file on its own (or with the other UI files) and it
# measures; run it at the end of 11,000 tests and it says why it will not.
FD_SETSIZE = 1024


def _open_fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:                                # not Linux; assume room
        return 0


# theme.resolve_fonts() is a ONE-WAY door: it returns early once _FAMILY is
# set, so the first test here that builds a root leaves theme._HAS_DISPLAY
# True for the rest of the process -- and tests/fixtures/theme_tokens_*.json
# pins it False. MEASURED 2026-09-05: `pytest tests/test_ui_layout_rules.py
# tests/test_theme_look.py` gave 2 failed with the diff exactly
# {'_HAS_DISPLAY': (False, True)} and no colour token drifted; the reverse
# order gave 68 passed. Alphabetical order hides it and the canonical suite
# skips this file, but this file's own docstring tells you to run it with
# the other UI files, which is the failing order.
FONT_GLOBALS = ("_FAMILY", "_FAMILY_MONO", "_HAS_DISPLAY", "_DISPLAY")


def font_globals() -> dict:
    return {k: getattr(theme, k) for k in FONT_GLOBALS}


def set_font_globals(saved: dict) -> None:
    for k, v in saved.items():
        setattr(theme, k, v)


@pytest.fixture(autouse=True)
def _restore_look():
    fonts = font_globals()
    yield
    theme.apply_scale(1.0)
    wg.set_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)
    set_font_globals(fonts)


# ============================================================ Tk-free rules
# ---------------------------------------------------- the drawer's slider
def test_slider_formats_a_value_the_way_the_stock_scale_did():
    """The three drawer rows read 0.015 / 2.5 / 0.30 on the 09-05 shot;
    the holo slider must print the same digits for the same resolution."""
    assert wg.slider_format(0.015, 0.001) == "0.015"
    assert wg.slider_format(2.5, 0.5) == "2.5"
    assert wg.slider_format(0.3, 0.05) == "0.30"
    assert wg.slider_format(3, 1) == "3"
    assert wg.slider_format(0.25, 0.25) == "0.25"
    # digits come from the RESOLUTION, so a value that lands on a round
    # number does not shorten and make the row jump
    assert wg.slider_format(0.3, 0.001) == "0.300"


def test_slider_formatting_never_raises_on_junk():
    """It runs inside a drag and inside bind_config."""
    assert wg.slider_format(None, 0.05) == "0.00"
    assert wg.slider_format("x", 0.5) == "0.0"
    assert wg.slider_format(float("nan"), 0.05) == "0.00"
    assert wg.slider_decimals(0) == 2
    assert wg.slider_decimals("x") == 2


def test_slider_snaps_to_the_resolution_and_clamps_to_the_range():
    assert wg.slider_snap(0.0163, 0.005, 0.05, 0.001) == pytest.approx(0.016)
    assert wg.slider_snap(2.74, 2.0, 20.0, 0.5) == pytest.approx(2.5)
    assert wg.slider_snap(2.76, 2.0, 20.0, 0.5) == pytest.approx(3.0)
    assert wg.slider_snap(-1.0, 0.1, 0.9, 0.05) == pytest.approx(0.1)
    assert wg.slider_snap(9.0, 0.1, 0.9, 0.05) == pytest.approx(0.9)
    # junk never raises: it lands on the low end
    assert wg.slider_snap("x", 0.1, 0.9, 0.05) == pytest.approx(0.1)
    assert wg.slider_snap(float("nan"), 0.1, 0.9, 0.05) == pytest.approx(0.1)
    assert wg.slider_snap(None, 0.1, 0.9, 0.05) == pytest.approx(0.1)
    # a zero step is a free value, not a division by zero
    assert wg.slider_snap(0.37, 0.0, 1.0, 0) == pytest.approx(0.37)


def test_holo_slider_rows_fit_the_drawer_with_the_value_inline():
    """MEASURED 2026-09-05 on :94 at S=2: the widest slider label
    ('Silence timeout (s)') is 310 px and the drawer's inner width 576.
    The value sits INLINE at the row's right now, so the track, the two
    gaps and a fixed-width value must fit beside that label. The value box
    is 5 characters ('0.015' is the widest the drawer shows), 70 px in the
    caption mono face."""
    from jarvis.ui.views import SettingsDrawer
    label_w, inner, value_w = 310, 576, 70
    row = (2 * SettingsDrawer.SLIDER_GAP_HOLO
           + 2 * SettingsDrawer.SLIDER_LEN_HOLO
           + 2 * SettingsDrawer.SLIDER_VALUE_GAP_HOLO + value_w)
    assert label_w + row <= inner, (label_w + row, inner)
    assert SettingsDrawer.SLIDER_VALUE_CHARS >= len("0.015")


# ----------------------------------------------- the explanation lines
WHY_MAX = 56    # characters: one line at the caption face in the value
                # column of his 920-px window at S=2 (measured ~59 fit)


def _opts():
    rooms = [{"name": "office", "enabled": True, "camera_zone": "at the desk",
              "bands": [{"name": "empty space", "near_m": 0.75, "far_m": 2.25},
                        {"name": "at the desk", "near_m": 2.25, "far_m": 3.75}]},
             {"name": "kitchen", "enabled": True, "camera_zone": "",
              "bands": [{"name": "the kitchen", "near_m": 0.75, "far_m": 3.0},
                        {"name": "at the door", "near_m": 3.0, "far_m": 3.75}]}]
    data = {"zones": {"enabled": True, "rooms": rooms}}

    def get_option(key, default=None):
        node = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    return get_option


def _fuse_matrix():
    """Every verdict this page can reach, in both rooms (the office has a
    camera zone, the kitchen has none) and with the toggle both ways."""
    ladders = sp.read_ladders(_opts())
    face = sp.camera_view({"live": True, "faces": 1,
                           "face": {"name": "hunterp", "id_score": 0.71,
                                    "id_ran": True}})
    blank = sp.camera_view({})
    out = []
    for room in ("office", "kitchen"):
        zmap = ladders.for_room(room)
        for present in (True, False, None):
            for distance in (1.0, 3.0, 3.9, 5.5, None):
                for cam in (face, blank):
                    for overrules in (True, False):
                        out.append(sp.fuse(present=present,
                                           distance_m=distance, camera=cam,
                                           zmap=zmap, overrules=overrules))
    out.append(sp.fuse(present=True, distance_m=3.0, camera=blank, zmap=None,
                       overrules=True))
    return out


def test_every_why_line_is_one_short_line():
    """The 09-05 shot had 'camera · the camera can see a face; the radar's
    range is / not asked' wrapping mid-phrase, and the sentence GREW a
    semicolon clause about the camera on top of that. Every reason the
    fusion can give now fits one line, and the SOURCE is no longer
    prefixed onto it (it sits beside the verdict word)."""
    seen = set()
    for v in _fuse_matrix():
        assert len(v.why) <= WHY_MAX, (len(v.why), v.why)
        assert not v.why.startswith(("camera ·", "radar ·")), v.why
        assert "\n" not in v.why
        seen.add(v.why)
    assert len(seen) >= 7, seen          # the matrix really does vary


def test_every_fault_line_is_one_short_line():
    cases = [
        {"url": "http://192.0.2.10", "fails": 2},
        {"url": "http://192.0.2.10", "paused": True, "retry_in_s": 29.0},
        {"url": "http://192.0.2.10", "paused": True},
        {"url": "http://192.0.2.10", "blocked": "offline"},
        {"url": "http://192.0.2.10", "blocked": "stopped"},
        {"url": "http://192.0.2.10", "blocked": "policy"},
        {"url": "http://192.0.2.10"},
        {"url": ""},
    ]
    for status in cases:
        line = sp.fault_line(status)
        assert 0 < len(line) <= WHY_MAX, (len(line), line)


def test_only_the_fault_line_wears_amber_among_the_explanation_lines():
    """Grey and orange with no legend (the 09-05 shot). The rule: the
    reason line is muted, the fault line is the ONE amber line, and it is
    there only when a leg had no opinion."""
    ladders = sp.read_ladders(_opts())
    ok = sp.Reading(name="office", present=True, distance_m=1.42, rtt_ms=0.4)
    row = sp.page_rows([ok], {}, ladders, True, camera_room="office")[0]
    lines = sp.explanation_lines(row)
    assert [tone for _t, tone in lines] == [sp.TONE_FAINT]
    dark = sp.Reading(name="office", present=None,
                      status={"url": "http://192.0.2.10", "fails": 2})
    row = sp.page_rows([dark], {}, ladders, True, camera_room="office")[0]
    lines = sp.explanation_lines(row)
    assert [tone for _t, tone in lines] == [sp.TONE_FAINT, sp.TONE_WARN]
    assert "2 in a row" in lines[1][0]
    assert all(tone != sp.TONE_WARN for _t, tone in lines[:-1])


def test_the_verdict_line_names_its_source_beside_the_word():
    ladders = sp.read_ladders(_opts())
    face = {"live": True, "faces": 1,
            "face": {"name": "hunterp", "id_score": 0.71, "id_ran": True}}
    ok = sp.Reading(name="office", present=True, distance_m=1.42, rtt_ms=0.4)
    row = sp.page_rows([ok], face, ladders, True, camera_room="office")[0]
    assert row.verdict.source == "camera"
    assert sp.source_text(row.verdict) == "· camera"
    dark = sp.Reading(name="office", present=None,
                      status={"url": "http://192.0.2.10", "fails": 2})
    row = sp.page_rows([dark], {}, ladders, True, camera_room="office")[0]
    assert row.verdict.source == ""
    assert sp.source_text(row.verdict) == ""


def test_a_dash_is_a_word_and_a_number_is_a_readout():
    """The KITCHEN header's '—' was drawn in the monospace face beside a
    sans header (the 09-05 shot). Monospace is for NUMBERS only."""
    assert sp.readout_is_numeric(sp.fmt_m(1.42)) is True
    assert sp.readout_is_numeric(sp.fmt_ms(0.4)) is True
    assert sp.readout_is_numeric(sp.fmt_ms(154)) is True
    assert sp.readout_is_numeric(sp.fmt_m(None)) is False
    assert sp.readout_is_numeric(sp.DASH) is False
    assert sp.readout_is_numeric("") is False
    assert sp.readout_is_numeric(None) is False


def test_the_standing_captions_are_short_and_plain():
    """The house voice: a caption, not a sentence."""
    assert len(sp.RESTART_NOTE) <= 40, sp.RESTART_NOTE
    assert "restart" in sp.RESTART_NOTE.lower()
    assert "saved" not in sp.RESTART_NOTE.lower()
    assert sp.RESTART_NOTE == sp.RESTART_NOTE.strip()
    assert not sp.RESTART_NOTE.endswith(".")


# ------------------------------------------------- the band bars are SPANS
def test_a_band_bar_is_its_share_of_the_rooms_whole_ladder():
    """ROUND 1 FIX. The bars were a full-width track with two end stops and
    a marker that is absent whenever the live range is outside the band --
    so on the 09-05 frames one of four bars carried a mark and on the fault
    frame none of four did, and four bare tracks read as four sliders with
    the handle gone (his words: the sliders "show no handle").

    They are readouts, so they now draw WHAT THEY KNOW: each band's own
    extent, positioned on the room's whole ladder. Two bands of one room
    start at different x by construction, which is the thing a slider can
    never do."""
    assert sp.room_scale([(0.75, 2.25), (2.25, 3.75)]) == (0.75, 3.75)
    assert sp.band_span(0.75, 2.25, 0.75, 3.75) == (0.0, 0.5)
    assert sp.band_span(2.25, 3.75, 0.75, 3.75) == (0.5, 1.0)
    assert sp.band_span(3.0, 3.75, 0.75, 3.0 + 0.75) == (0.75, 1.0)


def test_a_gap_in_the_ladder_is_drawn_as_a_gap():
    """fuse() has a whole rule for "the range is in a gap between the
    bands"; until now the page could not SHOW him one."""
    spans = sp.spans_for_room([(0.75, 2.0), (2.5, 3.75)])
    assert spans[0][1] < spans[1][0]
    assert spans[0][0] == 0.0 and spans[1][1] == 1.0


def test_a_span_never_divides_by_a_degenerate_ladder():
    assert sp.band_span(1.0, 1.0, 1.0, 1.0) is None
    assert sp.band_span(None, 2.0, 0.0, 3.0) is None
    assert sp.band_span(1.0, 2.0, 3.0, 3.0) is None
    assert sp.spans_for_room([]) == ()
    assert sp.spans_for_room([(None, None)]) == (None,)
    assert sp.spans_for_room([("x", 2.0), (1.0, 2.0)]) == (None, (0.0, 1.0))
    # one band is the whole scale
    assert sp.spans_for_room([(0.75, 3.75)]) == ((0.0, 1.0),)


def test_a_typed_band_moves_its_own_bar_and_nothing_else():
    """He edits the numbers in the boxes; the ladder under them follows
    what he typed, so a band he is widening grows while he types."""
    before = sp.spans_for_room([(0.75, 2.25), (2.25, 3.75)])
    after = sp.spans_for_room([(0.75, 3.0), (3.0, 3.75)])
    assert after[0][1] > before[0][1]
    assert after[1][0] > before[1][0]


# --------------------------------------------------------- amber is a fault
def test_no_opinion_is_not_amber_anywhere_in_a_room_block():
    """His brief: "reserve orange for a fault only ... make it the only
    orange". The reason lines were fixed in round 0 and the COUNT did not
    move: a dark room still painted the state word amber twice (the header
    verdict and the radar leg), so the fault frame was two stacked blocks
    of orange. A missing answer is now the FAINTEST thing on the row, not
    the loudest; the fault line under it says why, and it is the only
    amber thing on the page."""
    assert sp.presence_words(None) == ("NO OPINION", sp.TONE_FAINT)
    assert sp.presence_words(True)[1] == sp.TONE_OK
    assert sp.presence_words(False)[1] == sp.TONE_MUTED
    assert sp.verdict_tone(sp.NO_OPINION) == sp.TONE_FAINT
    assert sp.verdict_tone(sp.NO_LADDER) == sp.TONE_FAINT
    assert sp.verdict_tone("at the desk") == sp.TONE_OK
    assert sp.TONE_WARN not in (sp.presence_words(None)[1],
                                sp.verdict_tone(sp.NO_OPINION))


# ------------------------------------------------------ the house voice
def test_no_reason_line_prints_python_quotes_or_a_config_key():
    """MEASURED under pytest on round 0's tip: the RULE_BAND reason -- the
    ordinary line, shown whenever the radar places him and the camera does
    not overrule -- rendered `a target inside 'at the desk'`, and the
    no-camera-zone line rendered `'office' has no camera zone in
    zones.rooms`. %r puts Python's quotes and a dotted config key in the
    house-voice column on a normal day."""
    quoted = re.compile(r"""(^|\s)['"]|['"](\s|$)""")
    for v in _fuse_matrix():
        assert not quoted.search(v.why), v.why      # an apostrophe is fine
        assert "zones." not in v.why, v.why
        assert "_" not in v.why, v.why


def test_no_opinion_names_no_source_even_when_the_camera_was_looking():
    """fuse() printed "NO OPINION · radar" when the radar had been silent:
    a face in a room whose camera_zone is blank set saw_face, and the
    source line only checked saw_face. Neither leg answered, so neither
    leg is named."""
    ladders = sp.read_ladders(_opts())
    face = sp.camera_view({"live": True, "faces": 1,
                           "face": {"name": "hunterp", "id_score": 0.71,
                                    "id_ran": True}})
    v = sp.fuse(present=None, distance_m=None, camera=face,
                zmap=ladders.for_room("kitchen"), overrules=True)
    assert v.zone == sp.NO_OPINION
    assert v.source == ""
    assert sp.source_text(v) == ""
    # and the radar is still named when the radar really did answer
    v = sp.fuse(present=True, distance_m=3.0, camera=face,
                zmap=ladders.for_room("kitchen"), overrules=True)
    assert v.source == "radar"

def test_this_file_hands_the_font_globals_back_the_way_it_found_them():
    """Otherwise it corrupts the classic oracle for every file after it."""
    before = font_globals()
    theme._FAMILY, theme._HAS_DISPLAY = "Somefont", True
    assert font_globals() != before
    set_font_globals(before)
    assert font_globals() == before
    assert set(FONT_GLOBALS) <= set(vars(theme))


def test_the_holo_band_bar_marker_is_a_needle_not_a_handle():
    """The 09-05 brief read the band bars as sliders with missing handles.
    They are READOUTS: the marker is the live range and is absent when the
    range is outside the band (band_fraction), which is correct. What was
    wrong is that a round dot on a track reads as a slider knob. In holo
    the marker is a needle; classic keeps its dot."""
    assert sp.marker_shape("holo") == "needle"
    assert sp.marker_shape("classic") == "dot"
    theme.select_look("holo")
    assert sp.marker_shape() == "needle"
    theme.select_look("classic")
    assert sp.marker_shape() == "dot"


# ===================================================== on a private display
def _display() -> str:
    """The private display to build on, or "" to skip. Refuses his
    desktop displays outright (the rig's rule, tests/test_ui_shots.py)."""
    d = (os.environ.get("JARVIS_UI_TEST_DISPLAY") or "").strip()
    if not d:
        return ""
    base = d.split(".")[0]
    if base in FORBIDDEN_DISPLAYS or base.split(":")[-1] in ("0", "1"):
        pytest.fail("JARVIS_UI_TEST_DISPLAY=%r is a desktop display; the "
                    "layout tests build windows and will not open one on "
                    "his screen" % d)
    return d


# MEASURED 2026-09-05 off the photo rig's frames (holo, S=2, 920x1440, the
# camera pane packed): the page spans from the tab strip's rule (y=208) to
# the pane's top (y=1060).
STAGE_W, STAGE_H_WITH_PANE = 920, 852
SCALE = 2.0


@pytest.fixture
def root():
    display = _display()
    if not display:
        pytest.skip("set JARVIS_UI_TEST_DISPLAY=:9N (a private Xvfb) to run "
                    "the measured layout tests")
    import tkinter as tk
    fds = _open_fds()
    if fds >= FD_SETSIZE - 32:                     # headroom for Tk's own
        pytest.skip("this process already holds %d open descriptors; an X "
                    "connection past select()'s FD_SETSIZE (%d) aborts the "
                    "interpreter (see the note at the top of this file). Run "
                    "this file on its own." % (fds, FD_SETSIZE))
    try:
        r = tk.Tk(screenName=display)
    except tk.TclError as exc:
        pytest.skip("no X server at %s: %s" % (display, exc))
    # MAPPED, not withdrawn: a withdrawn root lays nothing out, so every
    # winfo_height() below would be 1 and every assertion would pass for
    # the wrong reason. The display is private (checked above), so a real
    # window here is a window nobody sees.
    r.geometry("%dx%d+0+0" % (STAGE_W, 1440))
    theme.resolve_fonts(r)
    theme.apply_scale(SCALE)
    wg.set_scale(SCALE)
    yield r
    try:
        r.destroy()
    except Exception:  # noqa: BLE001 - teardown
        pass


def _fake_services(rooms: int = 2):
    from types import SimpleNamespace
    specs = [{"name": "office", "url": "http://192.0.2.10", "label": "the office"},
             {"name": "kitchen", "url": "http://192.0.2.11", "label": "the kitchen"}]
    data = {
        "presence": {"room_sensor_enabled": True, "rooms": specs[:rooms],
                     "camera_overrules": True},
        "camera": {"room": "office"},
        "zones": {"enabled": True, "rooms": [
            {"name": "office", "enabled": True, "camera_zone": "at the desk",
             "bands": [{"name": "empty space", "near_m": 0.75, "far_m": 2.25},
                       {"name": "at the desk", "near_m": 2.25, "far_m": 3.75}]},
            {"name": "kitchen", "enabled": True, "camera_zone": "",
             "bands": [{"name": "the kitchen", "near_m": 0.75, "far_m": 3.0},
                       {"name": "at the door", "near_m": 3.0, "far_m": 3.75}]},
        ][:rooms]},
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
    """State 27 of the rig: every room dark with a fault line AND a reason
    line, the camera off for the curfew. The tallest the page can be."""
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


@pytest.mark.parametrize("look", ["holo", "classic"])
def test_nothing_on_the_sensors_page_is_clipped_at_his_window(root, look):
    """The 09-05 shot: 'camera overrules radar' cut off under the camera
    pane, SAVE and its caption gone entirely. At his window with the pane
    packed and BOTH rooms in their worst state the page must FIT -- and
    the foot (SAVE, the overrule toggle, the caption, the age) is pinned
    outside the scroll, so it is on screen whatever the body does."""
    page = _build_page(root, look)
    assert page.overflow_px() == 0, page.overflow_px()
    view_bottom = page._canvas.winfo_rooty() + page._canvas.winfo_height()
    for band in page._bands:
        row = band["lo"].master
        assert row.winfo_rooty() + row.winfo_height() <= view_bottom, \
            band["name"]
    page_bottom = page.winfo_rooty() + page.winfo_height()
    for name in ("_save_btn", "_overrule", "_note", "_age"):
        w = getattr(page, name)
        assert w.winfo_ismapped(), name
        assert w.winfo_rooty() + w.winfo_height() <= page_bottom, name


def test_the_page_scrolls_rather_than_clips_when_it_cannot_fit(root):
    """A ladder with more bands, or a third sensor, or a shorter window:
    the body scrolls, the thumb says so, and SAVE never scrolls away."""
    page = _build_page(root, "holo", height=420)
    assert page.overflow_px() > 0
    assert page._thumb.winfo_ismapped()
    save = page._save_btn
    assert save.winfo_rooty() + save.winfo_height() <= \
        page.winfo_rooty() + page.winfo_height()
    # and it really moves
    before = page._canvas.canvasy(0)
    page._scroll(3)
    root.update_idletasks()
    assert page._canvas.canvasy(0) > before
    # a page that fits shows no thumb at all
    fits = _build_page(root, "holo")
    assert not fits._thumb.winfo_ismapped()


def test_a_room_block_uses_two_sizes_and_mono_only_for_numbers(root):
    import tkinter.font as tkfont
    page = _build_page(root, "holo")
    block = page._blocks["office"]
    sizes = set()
    for lbl in (block.name, block.presence, block.camera, block.verdict,
                block.why, block.fault):
        sizes.add(abs(tkfont.Font(font=lbl.cget("font")).cget("size")))
    assert len(sizes) == 2, sizes
    mono = theme.mono(1)[0]
    for lbl in (block.name, block.presence, block.camera, block.verdict,
                block.why, block.fault, block.source):
        assert tkfont.Font(font=lbl.cget("font")).cget("family") != mono
    # the dash is a word, drawn in the header's own face
    assert block.distance.cget("text") == sp.DASH
    assert tkfont.Font(font=block.distance.cget("font")).cget("family") != mono
    block.apply(sp.page_rows(
        [sp.Reading(name="office", present=True, distance_m=1.42, rtt_ms=0.4)],
        {}, page.ladders, True, camera_room="office")[0])
    assert tkfont.Font(font=block.distance.cget("font")).cget("family") == mono
    assert tkfont.Font(font=block.rtt.cget("font")).cget("family") == mono


def test_a_room_block_puts_every_detail_line_on_one_label_column(root):
    """The reason line was indented by a hand-guessed px(96) while the two
    leg values sat behind a fixed-width caption -- so nothing lined up."""
    page = _build_page(root, "holo")
    root.update_idletasks()
    block = page._blocks["office"]
    xs = {name: b.camera.winfo_x() for name, b in page._blocks.items()}
    assert len(set(xs.values())) == 1, xs
    assert block.why.winfo_x() == block.camera.winfo_x() \
        == block.presence.winfo_x() == block.fault.winfo_x()


def test_the_band_editor_rows_share_one_baseline(root):
    page = _build_page(root, "holo")
    root.update_idletasks()

    def centre(w):
        return w.winfo_y() + w.winfo_height() / 2

    for band in page._bands:
        lo, hi, bar = band["lo"], band["hi"], band["bar"]
        assert abs(centre(lo) - centre(bar)) <= 2, band["name"]
        assert abs(centre(hi) - centre(bar)) <= 2, band["name"]
        assert abs(centre(band["unit"]) - centre(hi)) <= 2, band["name"]
    units = {b["unit"].winfo_x() for b in page._bands}
    assert len(units) == 1, units


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


def _rows_of(drawer):
    """Every row frame in the drawer's sections, in order."""
    out = []
    for box in drawer._inner.winfo_children():
        for row in box.winfo_children():
            if row.winfo_class() == "Frame":
                out.append(row)
    return out


def test_holo_slider_rows_put_the_value_inline_on_the_rows_baseline(root):
    """The value floated ABOVE a stock Tk scale (the 09-05 shot). Now it is
    a label at the right of the track whose centre line is the row's."""
    drawer = _build_drawer(root, "holo")
    seen = 0
    for row in _rows_of(drawer):
        sliders = [k for k in row.winfo_children() if isinstance(k, wg.Slider)]
        if not sliders:
            continue
        seen += 1
        slider = sliders[0]
        assert slider.label is not None and slider.label.master is row
        label = slider.label
        c_row = row.winfo_height() / 2
        assert abs(label.winfo_y() + label.winfo_height() / 2 - c_row) <= 2
        assert abs(slider.winfo_y() + slider.winfo_height() / 2 - c_row) <= 2
        assert label.winfo_x() >= slider.winfo_x() + slider.winfo_width() - 1
        assert label.cget("text") == wg.slider_format(slider.get(), slider.res)
    assert seen == 3, seen


def test_holo_button_rows_sit_in_the_control_column(root):
    """Buttons sat under the label column and broke the two-column rhythm
    (the 09-05 shot). In holo a button row's button is right-aligned with
    the toggles' right edge."""
    drawer = _build_drawer(root, "holo")
    right_edges = set()
    buttons = 0
    for row in _rows_of(drawer):
        for kid in row.winfo_children():
            if isinstance(kid, (wg.Toggle, wg.RoundButton)):
                right_edges.add(kid.winfo_x() + kid.winfo_width())
                buttons += isinstance(kid, wg.RoundButton)
    assert buttons >= 5, buttons
    assert len(right_edges) == 1, right_edges


def test_classic_keeps_the_stock_scale_and_the_left_aligned_button(root):
    import tkinter as tk
    drawer = _build_drawer(root, "classic")
    rows = _rows_of(drawer)
    scales = [k for row in rows for k in row.winfo_children()
              if isinstance(k, tk.Scale)]
    assert len(scales) == 3
    assert all(not isinstance(k, wg.Slider) for row in rows
               for k in row.winfo_children())
    # classic buttons pack straight into the box, left-anchored, as before
    boxes = drawer._inner.winfo_children()
    direct = [k for box in boxes for k in box.winfo_children()
              if isinstance(k, wg.RoundButton)]
    assert len(direct) >= 5
    for btn in direct:
        assert btn.winfo_x() == 0, btn.winfo_x()


def test_the_drawer_keeps_its_width_and_its_section_order(root):
    """Another lane (knightfall) adds rows to Privacy and fields to
    Services through these same helpers, so the sections may not be
    reordered under it."""
    from jarvis.ui import views
    drawer = _build_drawer(root, "holo")
    assert drawer.WIDTH == wg.px(views.SettingsDrawer.WIDTH)
    titles = [w.cget("text") for w in drawer._inner.winfo_children()
              if w.winfo_class() == "Label"]
    plain = [t.replace(" ", "") for t in titles]
    assert plain == ["AUDIO", "RECOGNITION", "VOICEID", "SPEECH",
                     "INTELLIGENCE", "ASSISTANT", "PRIVACY", "SYSTEM"]


# MEASURED 2026-09-05 from the rig at the app's OWN default geometry
# (main_window.DEFAULT_W/H = 520x880 design units, i.e. 1040x1760 at S=2 --
# the size he actually runs). The stage under the tab strip is 320 px
# taller there than in the 920x1440 window the brief named.
STAGE_H_HIS_WINDOW = 1172


def test_the_foot_follows_the_content_at_the_size_he_runs(root):
    """ROUND 1 FIX. Pinning the foot to the bottom of the frame traded the
    clipped row for a VOID: measured on his own default geometry, 408 px
    of nothing between the last band row and SAVE -- 23% of the window --
    because the page fits with room to spare there. The page is one column
    now: the foot follows the content, and the slack falls off the bottom
    where a finished page ends."""
    page = _build_page(root, "holo", height=STAGE_H_HIS_WINDOW)
    body_bottom = page._body.winfo_rooty() + page._body.winfo_height()
    gap = page._save_btn.winfo_rooty() - body_bottom
    assert 0 <= gap <= wg.px(24), gap
    assert page.overflow_px() == 0
    assert not page._thumb.winfo_ismapped()
    # and nothing has fallen off the bottom
    page_bottom = page.winfo_rooty() + page.winfo_height()
    for name in ("_save_btn", "_overrule", "_note", "_age"):
        w = getattr(page, name)
        assert w.winfo_ismapped(), name
        assert w.winfo_rooty() + w.winfo_height() <= page_bottom, name


def test_the_foot_is_still_pinned_below_the_fold_when_the_page_scrolls(root):
    """The other half of the same rule: when the body is taller than the
    room it has, the view stops at the foot and the body scrolls under it.
    SAVE never scrolls away."""
    page = _build_page(root, "holo", height=420)
    assert page.overflow_px() > 0
    save_top = page._save_btn.winfo_rooty()
    view_bottom = page._canvas.winfo_rooty() + page._canvas.winfo_height()
    assert view_bottom <= save_top


def test_every_band_row_draws_its_own_span(root):
    """The state he photographed: no live range in either room, so no
    marker anywhere. Every bar must still carry ink of its own, and the
    two bands of one room must start at different x -- which is what a
    span is and a slider track is not."""
    page = _build_page(root, "holo")
    root.update_idletasks()
    lefts = {}
    for band in page._bands:
        bar = band["bar"]
        assert bar.find_all(), band["name"]
        x0, x1 = bar.span_px()
        assert x1 > x0, band["name"]
        lefts.setdefault(band["room"], []).append(x0)
    for room, xs in lefts.items():
        assert len(set(xs)) == len(xs), (room, xs)


def test_typing_a_band_bound_redraws_that_rooms_ladder(root):
    """The boxes are the editor; the bars are the picture of what is in
    them. Widening a band under his hands widens its bar and pushes the
    next band along, so he can see the ladder he is typing.

    The redraw is bound to <KeyRelease> and <FocusOut> on both boxes; the
    BINDING is asserted rather than driven, because an Xvfb with no window
    manager gives the toplevel no input focus and delivers no key event at
    all (measured on :94: event_generate("<KeyRelease-0>", when="now") on
    a mapped, focused entry fires nothing). What the binding calls is
    driven directly."""
    page = _build_page(root, "holo")
    root.update_idletasks()
    first, second = page._bands[0], page._bands[1]
    for row in (first, second):
        for box in (row["lo"], row["hi"]):
            assert set(box.bind()) >= {"<KeyRelease>", "<FocusOut>"}, box
    before = (first["bar"].span_px()[1], second["bar"].span_px()[0])
    first["hi"].delete(0, "end")
    first["hi"].insert(0, "3.00")
    page._resync_spans()
    root.update_idletasks()
    assert first["bar"].span_px()[1] > before[0]
    # the neighbour has not moved, so the OVERLAP he has just typed (0.75
    # to 3.00 over 2.25 to 3.75) is on the page -- band_edits would refuse
    # that pair on SAVE and now he can see it before he presses it
    assert second["bar"].span_px()[0] == before[1]
    assert second["bar"].span_px()[0] < first["bar"].span_px()[1]
    # the live mark is measured against the same bounds the bar was drawn
    # from, so the needle cannot drift off the bar while he types
    assert (first["near_m"], first["far_m"]) == (0.75, 3.0)
    # a box holding junk drops its own bar and leaves the room's alone
    first["hi"].delete(0, "end")
    first["hi"].insert(0, "abc")
    page._resync_spans()
    root.update_idletasks()
    assert first["span"] is None
    assert second["span"] is not None


def test_the_live_range_marks_the_band_it_is_in_and_no_other(root):
    page = _build_page(root, "holo")
    page._open = True            # apply() only paints an open page; show()
    # would start the poller, and this measures the paint, not the poll.
    page.apply([sp.Reading(name="office", label="the office", present=True,
                           distance_m=3.10, rtt_ms=0.4)])
    root.update_idletasks()
    marked = [b["name"] for b in page._bands if b["bar"].marked()]
    assert marked == ["at the desk"], marked


def test_only_the_fault_line_is_amber_in_a_dark_room_block(root):
    """Counting amber ink on the fault frame: 6 lines before, and the
    round-0 pass moved it to 6. The state word is the faintest thing on a
    dark row now, so the fault line is the only amber one -- one per room,
    two on the page."""
    page = _build_page(root, "holo")
    for block in page._blocks.values():
        amber = [w for w in (block.name, block.presence, block.camera,
                             block.verdict, block.source, block.why,
                             block.distance, block.rtt)
                 if str(w.cget("fg")) == theme.WARN]
        assert amber == [], [w.cget("text") for w in amber]
        assert str(block.fault.cget("fg")) == theme.WARN
        assert block.fault.winfo_ismapped()


def test_the_drawer_controls_share_one_ink_edge(root):
    """The author's stated rule was "every button's right edge is the
    toggles' right edge"; MEASURED on the frames it was 6 px out --
    buttons' ink at x=1005, toggles' at 999 -- because RoundButton drew
    its ring from winfo_width(), which counts the 2 px highlight border on
    each side, so the ring was 4 px too wide and its right edge was
    CLIPPED by the canvas. Widget geometry agreed; the ink did not."""
    drawer = _build_drawer(root, "holo")
    edges = {}
    for row in _rows_of(drawer):
        for kid in row.winfo_children():
            if isinstance(kid, (wg.Toggle, wg.RoundButton)):
                box = kid.bbox("all")
                assert box, kid
                bd = int(kid.cget("highlightthickness") or 0)
                edges.setdefault(type(kid).__name__, set()).add(
                    kid.winfo_width() - (box[2] + bd))
    assert set(edges) == {"Toggle", "RoundButton"}, edges
    every = set().union(*edges.values())
    assert max(every) - min(every) <= 1, edges

def test_the_rows_keep_one_vertical_rhythm(root):
    """Every row in a section box is a row frame with the same pady, so a
    button row is not taller than its neighbours by more than the button's
    own padding (the 'larger gap after Calibrate noise')."""
    drawer = _build_drawer(root, "holo")
    heights = [r.winfo_height() for r in _rows_of(drawer)]
    assert heights
    assert max(heights) - min(heights) <= wg.px(14), (min(heights),
                                                      max(heights))
