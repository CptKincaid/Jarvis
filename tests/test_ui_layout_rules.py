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

import pytest

from jarvis.ui import sensors_page as sp
from jarvis.ui import theme
from jarvis.ui import widgets as wg

FORBIDDEN_DISPLAYS = (":0", ":1")


@pytest.fixture(autouse=True)
def _restore_look():
    yield
    theme.apply_scale(1.0)
    wg.set_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)


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


def test_the_rows_keep_one_vertical_rhythm(root):
    """Every row in a section box is a row frame with the same pady, so a
    button row is not taller than its neighbours by more than the button's
    own padding (the 'larger gap after Calibrate noise')."""
    drawer = _build_drawer(root, "holo")
    heights = [r.winfo_height() for r in _rows_of(drawer)]
    assert heights
    assert max(heights) - min(heights) <= wg.px(14), (min(heights),
                                                      max(heights))
