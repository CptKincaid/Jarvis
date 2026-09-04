"""The camera pane's geometry, words and colours (jarvis/ui/preview.py).

DISPLAY-FREE, the way tests/test_holo_geometry.py and
tests/test_found_standby_drag_snapback.py are: the layout is pure functions,
and the one method that draws is taken UNBOUND off ``CameraPreview`` and run
against a canvas that records what it was asked to do. **No test here creates
a Tk root or a toplevel** -- UI window churn on the live display is what froze
his desktop on 2026-08-26, and a camera pane is not worth a second one.

NO TEST LOOKS AT A PICTURE EITHER. Nothing below asserts on pixel content;
what is checked is where a box lands, which word is printed, whether the
overlay is hidden, and whether the ink has enough contrast to read. The image
itself is represented by a sentinel object, because the pane's contract with
it is "put this on that canvas item" and nothing more.

Both LOOKS are exercised. Classic is the fallback Hunter asked for and the
pane has to be legible in it too, so every colour question is asked twice.
"""
import re
from types import SimpleNamespace

import pytest

from jarvis import campreview as cp
from jarvis.ui import preview as pv
from jarvis.ui import theme
from jarvis.ui.console_mode import ACTIVE, AMBIENT, STANDBY
from jarvis.ui.main_window import MainWindow
from jarvis.ui.preview import CameraPreview
from jarvis.ui.widgets import get_scale, px, set_scale


@pytest.fixture(autouse=True)
def _restore():
    """The look and the UI scale are module globals; put both back."""
    scale = get_scale()
    yield
    set_scale(scale)
    theme.select_look(theme.DEFAULT_LOOK)
    theme.apply_scale(scale)


def face(x=100.0, y=50.0, w=300.0, h=300.0, conf=0.93, yaw=14.1,
         attending=True, landmarks_ok=True, name="", id_score=0.0,
         id_ran=False):
    return cp.PreviewFace(conf=conf, x=x, y=y, w=w, h=h, yaw_deg=yaw,
                          attending=attending, landmarks_ok=landmarks_ok,
                          name=name, id_score=id_score, id_ran=id_ran)


def live(*faces, cap=(1280, 720)):
    return cp.PreviewShot(image=object(), faces=tuple(faces), cap_w=cap[0],
                          cap_h=cap[1], reason=cp.REASON_LIVE, seq=1)


# ------------------------------------------------------------ the fit
def test_a_sixteen_by_nine_camera_fills_the_box_exactly():
    """His LifeCam grants 1280x720, and the pane's box is 16:9, so the
    common case has no letterbox at all."""
    assert pv.fit_box(1280, 720, 320, 180) == (0, 0, 320, 180)


def test_a_box_that_is_a_pixel_off_sixteen_by_nine_letterboxes_by_a_pixel():
    """The pane's design units land on 136x76, which is 1.789 rather than
    1.778. The picture keeps the CAMERA's aspect and gives back the pixel."""
    ox, oy, w, h = pv.fit_box(1280, 720, 272, 152)
    assert (w, h) == (270, 152) and (ox, oy) == (1, 0)


def test_a_four_by_three_camera_is_letterboxed_not_squashed():
    """A 1.33x horizontal squash would put every face box slightly off the
    face it belongs to, which is the one thing a tracking overlay may not
    do. The pillars are centred, so the picture is not shoved to one side."""
    ox, oy, w, h = pv.fit_box(640, 480, 272, 152)
    assert (w, h) == (203, 152)                 # height-bound, aspect kept
    assert oy == 0 and ox == (272 - 203) // 2   # pillarboxed, centred
    assert abs(w / h - 640 / 480) < 0.01


def test_a_portrait_source_letterboxes_the_other_way():
    ox, oy, w, h = pv.fit_box(480, 640, 272, 152)
    assert w < 272 and h == 152
    assert ox > 0 and oy == 0


def test_a_camera_that_reported_nothing_still_gives_a_usable_box():
    assert pv.fit_box(0, 0, 272, 152) == (0, 0, 272, 152)


def test_the_drawn_fit_is_read_off_the_picture_that_actually_arrived():
    """One source of truth. Two independent computations of the same
    rectangle are two that can disagree, and here disagreeing means boxes
    a few pixels off the faces they belong to."""
    box = (272, 152)
    letterboxed = SimpleNamespace(width=203, height=152)   # a 4:3 camera
    assert pv.image_fit(letterboxed, box) == pv.fit_box(640, 480, *box)
    assert pv.image_fit(None, box) == (0, 0, 272, 152)     # nothing to measure


# ------------------------------------------------------- the face boxes
def test_a_detection_maps_into_the_picture_by_the_same_numbers_it_is_drawn_at():
    fit = pv.fit_box(1280, 720, 320, 180)
    x0, y0, x1, y1 = pv.face_rect(face(x=640, y=360, w=320, h=180), 1280, 720,
                                  fit)
    assert (x0, y0) == (160, 90)                # exactly the middle
    assert (x1 - x0, y1 - y0) == (80, 45)       # 1280->320 is 4:1


def test_the_letterbox_offset_moves_the_boxes_with_the_picture():
    """The overlay and the image share ``fit_box``, so a pillarboxed 4:3
    camera cannot end up with its boxes glued to the pane instead of to the
    picture."""
    fit = pv.fit_box(640, 480, 320, 180)
    x0, _y0, _x1, _y1 = pv.face_rect(face(x=0, y=0, w=64, h=48), 640, 480, fit)
    assert x0 == fit[0] > 0


def test_a_face_leaving_the_frame_is_clamped_rather_than_dropped():
    """Half out of frame is exactly when a tracking overlay earns its keep;
    a rectangle that vanished at the edge would read as a lost detection."""
    fit = pv.fit_box(1280, 720, 320, 180)
    x0, y0, x1, y1 = pv.face_rect(face(x=1200, y=680, w=300, h=300), 1280,
                                  720, fit)
    assert x1 <= 320 and y1 <= 180
    assert x0 >= 0 and y0 >= 0
    assert x1 > x0 and y1 > y0                  # still visible


def test_a_tiny_face_never_collapses_to_a_dot():
    fit = pv.fit_box(1280, 720, 320, 180)
    x0, y0, x1, y1 = pv.face_rect(face(x=10, y=10, w=1, h=1), 1280, 720, fit)
    assert x1 - x0 >= pv.MIN_BOX and y1 - y0 >= pv.MIN_BOX


def test_the_attention_brackets_stay_inside_the_box_they_bracket():
    rect = (10, 20, 60, 70)
    arms = pv.bracket_points(rect, 8)
    assert len(arms) == 4
    for arm in arms:
        xs, ys = arm[0::2], arm[1::2]
        assert min(xs) >= rect[0] and max(xs) <= rect[2]
        assert min(ys) >= rect[1] and max(ys) <= rect[3]


def test_the_bracket_arm_shrinks_rather_than_crossing_a_small_box():
    for arm in pv.bracket_points((0, 0, 8, 8), 40):
        xs = arm[0::2]
        assert max(xs) - min(xs) <= 4           # never past the midpoint


# --------------------------------------------------------------- the words
@pytest.mark.parametrize("reason,word", [
    (cp.REASON_LIVE, "LIVE"),
    (cp.REASON_DISABLED, "OFF"),
    (cp.REASON_SENSING, "CAMERA OFF"),
    (cp.REASON_PIPELINE, "NO CAMERA"),
    (cp.REASON_NO_FRAME, "NO SIGNAL"),
])
def test_every_state_has_a_word_of_its_own(reason, word):
    """"Off because you said so" and "broken" need opposite responses from
    him, so they must not share a readout."""
    assert pv.state_word(cp.blank(reason)) == word


def test_a_dark_pane_prints_the_reason_it_is_dark():
    shot = cp.blank(cp.REASON_SENSING, "curfew until 7 am")
    assert pv.state_word(shot) == "CAMERA OFF"
    assert pv.attention_line(shot) == ("curfew until 7 am", pv.TONE_OFF)


def test_the_verdict_row_says_which_of_the_three_it_is():
    assert pv.attention_line(live(face(attending=True)))[0] == pv.ATTEND_YES
    assert pv.attention_line(live(face(attending=False)))[0] == pv.ATTEND_NO
    assert pv.attention_line(live())[0] == pv.ATTEND_NONE


def test_the_verdict_is_carried_by_the_word_and_not_only_by_a_colour():
    """The rule sensing_badge settled on: colour alone does not survive a
    dimmed monitor or a colour-blind glance, and this is a readout where
    being wrong is not cosmetic."""
    yes, _ = pv.attention_line(live(face(attending=True)))
    no, _ = pv.attention_line(live(face(attending=False)))
    assert yes != no
    assert yes.strip() and no.strip()


def test_the_readouts_print_the_measured_numbers():
    rows = pv.readout_rows(live(face(conf=0.93, yaw=54.1)))
    assert rows[0] == ("CONF", "0.93")
    assert rows[1] == ("YAW", "+54°")


def test_a_yaw_with_no_landmarks_is_a_dash_not_a_zero():
    """0 deg means "facing me". A degenerate row that read as 0 would put
    LOOKING AT JARVIS on a detection with no eyes in it."""
    rows = pv.readout_rows(live(face(yaw=0.0, landmarks_ok=False)))
    assert rows[1] == ("YAW", "—")


def test_extra_faces_are_counted_beside_the_subject():
    rows = pv.readout_rows(live(face(w=300), face(w=100, attending=False)))
    assert "2 faces" in rows[0][1]


def test_an_empty_frame_reads_as_zero_faces_not_as_a_blank_row():
    rows = pv.readout_rows(live())
    assert rows[0] == ("FACES", "0")


def test_a_dark_pane_leaves_the_number_rows_blank_rather_than_labelled():
    """A 0 under CONF would be a measurement, and a labelled row with
    nothing in it reads as one that came back blank. There is no
    measurement; the state word and the reason line carry it."""
    assert pv.readout_rows(cp.blank(cp.REASON_SENSING)) == (("", ""), ("", ""))


# ------------------------------------------------------------ the colours
def _lum(color: str) -> float:
    color = color.lstrip("#")
    parts = []
    for i in (0, 2, 4):
        v = int(color[i:i + 2], 16) / 255.0
        parts.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2]


def _contrast(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


@pytest.mark.parametrize("look", theme.LOOKS)
def test_every_readout_is_legible_in_both_looks(look):
    """Classic is the fallback he asked for, so the pane has to read in it
    too. 4.5:1 is the WCAG AA bar for normal text; these are the values he
    reads off the pane, not decoration."""
    theme.select_look(look)
    ground = theme.BG
    for tone in (pv.TONE_LIVE, pv.TONE_AWAY, pv.TONE_OFF):
        assert _contrast(pv.tone_ink(tone), ground) >= 4.5, (look, tone)
    assert _contrast(theme.FOCAL, ground) >= 4.5        # the values
    assert _contrast(theme.MUTED, ground) >= 4.5        # the labels
    # The heading is a static caption naming the pane, not a value, so it
    # sits at the large-text bar rather than the body one.
    assert _contrast(theme.RAIL, ground) >= 3.0


@pytest.mark.parametrize("look", theme.LOOKS)
def test_the_three_tones_are_told_apart_by_colour_as_well_as_by_word(look):
    theme.select_look(look)
    inks = {pv.tone_ink(t) for t in (pv.TONE_LIVE, pv.TONE_AWAY, pv.TONE_OFF)}
    assert len(inks) == 3


@pytest.mark.parametrize("look", theme.LOOKS)
def test_a_face_box_has_an_edge_whatever_is_behind_it(look):
    """A stroke over live video is a stroke over an unknown background, and
    the two brightest tokens in the palette are exactly the ones that vanish
    against a lit wall. The dark halo underneath is what guarantees an
    edge."""
    theme.select_look(look)
    for attending in (True, False):
        assert _contrast(pv.box_ink(attending), pv.halo_ink()) >= 4.5
    # The OTHER faces too. A secondary box that is not legible over its own
    # halo is indistinguishable from a detection the detector missed, and
    # the first spelling of this (RAMP33) measured 2.2:1 / 2.3:1.
    assert _contrast(pv.box_ink(False, primary=False), pv.halo_ink()) >= 4.5
    assert _contrast(pv.box_ink(True, primary=False), pv.halo_ink()) >= 4.5


@pytest.mark.parametrize("look", theme.LOOKS)
def test_the_tracked_face_is_told_from_the_others_by_brightness(look):
    theme.select_look(look)
    assert _lum(pv.box_ink(True)) > _lum(pv.box_ink(False))
    assert _lum(pv.box_ink(False)) > _lum(pv.box_ink(False, primary=False))


def test_every_colour_is_read_at_call_time_and_not_frozen_at_import():
    """select_look re-derives the tokens at runtime; a module constant would
    freeze the import-time look -- the trap at the top of jarvis/ui/theme.py.
    """
    theme.select_look("holo")
    holo = (pv.chrome_ink(), pv.tone_ink(pv.TONE_OFF), pv.halo_ink())
    theme.select_look("classic")
    classic = (pv.chrome_ink(), pv.tone_ink(pv.TONE_OFF), pv.halo_ink())
    assert holo[0] != classic[0]        # GLASS_EDGE is derived per look
    assert holo[2] != classic[2]        # the two grounds


@pytest.mark.parametrize("look", theme.LOOKS)
def test_the_picture_box_has_a_visible_outline_in_both_looks(look):
    """The outline is the ONLY thing separating the box from the panel:
    SURFACE sits just 1.05:1 (holo) / 1.15:1 (classic) above the ground, so
    a chrome token that does not clear the fill leaves the pane's dark
    state -- its default, and its state until the vision lane lands -- as
    readouts floating on nothing. Classic LINE measured 1.44:1, which was
    the bug this replaced."""
    theme.select_look(look)
    assert pv.chrome_ink() == theme.GLASS_EDGE   # 1px strokes ARE the chrome
    assert _contrast(pv.chrome_ink(), theme.SURFACE) >= 3.0
    assert _contrast(pv.chrome_ink(), theme.BG) >= 3.0


# ---------------------------------------------------------- the surfaces
@pytest.mark.parametrize("mode", [AMBIENT, STANDBY])
def test_the_pane_is_absent_in_ambient_and_standby(mode):
    """Both are surfaces for a room nobody is talking to -- the panel is a
    dimmed clock drifting to avoid burn-in. A live camera view on either
    would be a lit lens for an empty chair."""
    assert pv.pane_visible(mode, True) is False


def test_the_pane_is_present_in_the_active_console_when_it_is_switched_on():
    assert pv.pane_visible(ACTIVE, True) is True
    assert pv.pane_visible(ACTIVE, False) is False


def test_the_band_scales_with_the_ui_scale():
    set_scale(1.0)
    theme.apply_scale(1.0)
    one = pv.band_height()
    set_scale(2.0)
    theme.apply_scale(2.0)
    assert pv.band_height() == pytest.approx(one * 2, abs=2)
    assert pv.worker_box() == (px(pv.PANE_W), px(pv.PANE_H))


# ------------------------------------------------------------- painting
class FakeCanvas:
    """Every canvas call the pane makes, recorded. No Tk anywhere."""

    # A made-up but FIXED text metric, so a test can say where a chip's
    # ground should land without measuring a real font. The pane asks the
    # canvas for the bbox of the text it just set precisely so it does not
    # have to guess this itself; what is under test is what it does with the
    # answer, not the answer.
    CHAR_W, LINE_H = 6, 12

    def __init__(self):
        self.state = {}
        self.coords_of = {}
        self.calls = []

    def itemconfigure(self, item, **kw):
        self.calls.append((item, kw))
        self.state.setdefault(item, {}).update(kw)

    itemconfig = itemconfigure

    def coords(self, item, *args):
        self.coords_of[item] = tuple(args)

    def bbox(self, item):
        """Tk's ``bbox`` for a text item, at this canvas's fake metric."""
        at = self.coords_of.get(item)
        conf = self.state.get(item, {})
        if not at or not conf.get("text"):
            return None
        x, y = at[0], at[1]
        w = len(conf["text"]) * self.CHAR_W
        if conf.get("anchor") == "sw":
            return (x, y - self.LINE_H, x + w, y)
        return (x, y, x + w, y + self.LINE_H)

    def shown(self, item) -> bool:
        return self.state.get(item, {}).get("state") == "normal"


def test_a_long_reason_is_trimmed_to_the_width_the_band_actually_has(
        monkeypatch):
    """A pipeline failure carries the exception text ("...No module named
    'jarvis.camera'"), which at S=1 is wider than the 520-unit console. A
    reason he cannot read is the same as no reason.

    The trimming itself is widgets.ellipsize (it measures in the real font
    and so needs a window); what is checked here is the ROOM handed to it,
    which is the part this module gets to be wrong about.
    """
    seen = {}

    def fake(text, _font, max_px):
        seen["room"] = max_px
        return text[:8] + "\u2026"
    monkeypatch.setattr(pv, "ellipsize", fake)
    ns = pane()
    ns._col_x = 165
    ns.canvas.winfo_width = lambda: 520
    long = "the camera lane is not installed (No module named 'jarvis.camera')"
    c = paint(ns, cp.blank(cp.REASON_PIPELINE, long))
    assert seen["room"] == 520 - 165 - theme.PAD
    assert c.state["verdict"]["text"] == "the came\u2026"


def test_a_pane_with_no_window_yet_prints_the_reason_whole():
    """_fit_text measures, and measuring needs a window. Before there is
    one the honest fallback is the untrimmed sentence, never a blank."""
    ns = pane()
    ns.canvas.winfo_width = lambda: (_ for _ in ()).throw(RuntimeError("no"))
    assert CameraPreview._fit_text(ns, "offline mode") == "offline mode"


def pane(scale=1.0, shown=True):
    """A ``CameraPreview`` stand-in carrying only what ``_paint`` touches.

    The shipping methods are taken UNBOUND and driven against it -- the
    house pattern, and the reason no test in this file has a window.
    """
    set_scale(scale)
    theme.apply_scale(scale)
    box = (px(pv.PANE_W), px(pv.PANE_H))
    ns = SimpleNamespace(
        canvas=FakeCanvas(), _origin=(10, 8), _box=box,
        _fit=(0, 0, box[0], box[1]), _img="img", _state="state",
        _verdict="verdict", _photo=None, _photo_size=(0, 0), _seq=-1,
        _rows=[("l0", "v0"), ("l1", "v1")],
        _shadows=["s0", "s1", "s2"], _boxes=["b0", "b1", "b2"],
        _brackets=["k0", "k1", "k2", "k3"],
        _name="name", _namebg="namebg",
        images=[])
    ns._clear_picture = lambda: CameraPreview._clear_picture(ns)
    ns._draw_faces = lambda shot: CameraPreview._draw_faces(ns, shot)
    ns._draw_name = lambda f, rect: CameraPreview._draw_name(ns, f, rect)
    ns._place_name = lambda t, x, y: CameraPreview._place_name(ns, t, x, y)
    ns._fit_text = lambda text: CameraPreview._fit_text(ns, text)
    ns._col_x = ns._origin[0] + box[0] + px(pv.COL_GAP)

    def show(image):
        # Mimics the real _show_image's canvas effects without PIL or Tk:
        # it either reveals the picture item or clears the pane.
        ns.images.append(image)
        if not shown:
            ns._clear_picture()
            return False
        ns.canvas.itemconfigure(ns._img, state="normal")
        return True
    ns._show_image = show
    return ns


def paint(ns, shot):
    CameraPreview._paint(ns, shot)
    return ns.canvas


def test_a_dark_pane_hides_the_picture_and_every_box():
    """A frozen last frame under the words "camera off" would be the console
    showing him a view it is claiming not to have."""
    ns = pane()
    c = paint(ns, cp.blank(cp.REASON_SENSING, "offline mode"))
    assert c.state["img"]["state"] == "hidden"
    for item in ns._boxes + ns._shadows + ns._brackets:
        assert c.state[item]["state"] == "hidden"
    assert c.state["state"]["text"] == "CAMERA OFF"
    assert c.state["verdict"]["text"] == "offline mode"
    assert ns.images == []                       # nothing was even converted


def test_a_pillarboxed_picture_takes_its_overlay_with_it():
    """The 4:3 case end to end: the picture is inset, and so are the boxes."""
    ns = pane()
    shot = cp.PreviewShot(image=SimpleNamespace(width=ns._box[0] - 40,
                                                height=ns._box[1]),
                          faces=(face(x=0, y=0, w=64, h=48),),
                          cap_w=640, cap_h=480, reason=cp.REASON_LIVE, seq=1)
    c = paint(ns, shot)
    ox, oy = ns._origin
    assert ns._fit[0] == 20                       # centred, 20 px each side
    assert c.coords_of[ns._img] == (ox + 20, oy)  # the picture is inset…
    assert c.coords_of["b0"][0] == ox + 20        # …and so is its overlay


def test_a_live_frame_shows_the_picture_and_the_tracked_box():
    ns = pane()
    c = paint(ns, live(face()))
    assert ns.images and ns.images[0] is not None
    assert c.state["img"]["state"] == "normal"
    assert c.shown("b0") and c.shown("s0")
    assert not c.shown("b1")                     # only one face was found
    assert c.state["b0"]["outline"] == theme.FOCAL


def test_the_brackets_close_only_when_he_is_inside_the_cone():
    ns = pane()
    c = paint(ns, live(face(attending=True)))
    assert all(c.shown(k) for k in ns._brackets)
    ns = pane()
    c = paint(ns, live(face(attending=False)))
    assert not any(c.shown(k) for k in ns._brackets)
    assert c.state["b0"]["outline"] == theme.CYAN


def test_the_boxes_are_drawn_inside_the_picture_and_offset_by_its_origin():
    ns = pane()
    c = paint(ns, live(face(x=640, y=360, w=320, h=180)))
    x0, y0, x1, y1 = c.coords_of["b0"]
    ox, oy = ns._origin
    assert ox <= x0 < x1 <= ox + ns._box[0]
    assert oy <= y0 < y1 <= oy + ns._box[1]
    assert c.coords_of["s0"] == c.coords_of["b0"]   # the halo tracks the box


def test_extra_faces_are_drawn_dimmer_than_the_subject():
    ns = pane()
    c = paint(ns, live(face(w=300), face(x=0, y=0, w=80, h=80,
                                         attending=False)))
    assert c.shown("b0") and c.shown("b1") and not c.shown("b2")
    assert c.state["b0"]["outline"] != c.state["b1"]["outline"]


def test_a_frame_that_will_not_convert_leaves_the_pane_dark_not_stale():
    ns = pane(shown=False)
    c = paint(ns, live(face()))
    assert c.state["img"]["state"] == "hidden"
    for item in ns._boxes:
        assert c.state[item]["state"] == "hidden"


# --------------------------------------------------- who it is looking at
def test_the_tracked_face_is_labelled_with_who_it_is():
    """His words, 2026-09-03: "lets have the identity of the person its
    tracking next to their name, small but readable" -- WHO, and only who.

    The first build of this read "identity" as the match score and drew
    "HUNTER 0.74" over his face. Seeing it live the same day he said it
    should not be there, and he is right: the caption over a face is the
    ANSWER, and the evidence behind the answer belongs on a surface opened
    to read numbers. The score is still on the SENSORS page."""
    ns = pane()
    c = paint(ns, live(face(name="hunter", id_score=0.74, id_ran=True)))
    assert c.state["name"]["text"] == "HUNTER"
    assert c.shown("name") and c.shown("namebg")
    assert c.state["name"]["fill"] == theme.FOCAL


def test_a_face_that_matched_nothing_says_unknown_rather_than_going_blank():
    """Three states, not two: no chip is "identity is not running", UNKNOWN
    is "it ran and nobody in the gallery is this person". Drawing them the
    same way would leave him unable to tell one from the other."""
    ns = pane()
    c = paint(ns, live(face(name="", id_score=0.21, id_ran=True)))
    assert c.state["name"]["text"] == "UNKNOWN"
    assert c.state["name"]["fill"] == theme.MUTED     # a lesser claim
    assert c.shown("name")


def test_a_face_nobody_asked_about_gets_no_chip_at_all():
    ns = pane()
    c = paint(ns, live(face()))
    assert not c.shown("name") and not c.shown("namebg")


def test_only_the_tracked_face_is_named():
    """One subject, the same rule the attention verdict follows: one
    embedding per identity tick, on the face at the desk."""
    ns = pane()
    c = paint(ns, live(face(w=300, name="hunter", id_score=0.74, id_ran=True),
                       face(x=0, y=0, w=80, h=80)))
    assert c.state["name"]["text"] == "HUNTER"
    assert len([1 for item in ("name", "namebg") if c.shown(item)]) == 2


def test_the_chip_carries_its_own_ground_under_its_word():
    """Text over live video is text over an unknown background -- the rule
    this pane already made when it put the numbers BESIDE the picture. He
    asked for this one next to the face, so it brings a ground with it."""
    ns = pane()
    c = paint(ns, live(face(name="hunter", id_score=0.74, id_ran=True)))
    bg = c.coords_of["namebg"]
    text = c.coords_of["name"]
    pad = px(pv.NAME_PAD)
    assert bg[0] == text[0] - pad and bg[1] <= text[1]
    assert bg[2] > text[0] and bg[3] > bg[1]          # it encloses the word


def test_the_chip_sits_under_the_box_it_belongs_to():
    ns = pane()
    named = face(x=100, y=50, w=200, h=200, name="hunter", id_score=0.74,
                 id_ran=True)
    c = paint(ns, live(named))
    box = c.coords_of["b0"]
    text = c.coords_of["name"]
    assert text[0] == box[0]                          # left-aligned to it
    assert text[1] >= box[3]                          # …and below it


def test_the_chip_flips_above_a_box_that_is_against_the_bottom():
    """A caption that ran off the picture would be a readout with no ground
    under it, which is the one thing the chip exists to prevent."""
    box = (px(pv.PANE_W), px(pv.PANE_H))
    picture = (10, 8, box[0], box[1])
    rect = (20, box[1] - 4, 60, box[1] + 8)           # hard against the floor
    x, y, anchor = pv.name_anchor(rect, picture, 3, 12)
    assert anchor == "sw" and y <= rect[1]


def test_a_face_that_fills_the_frame_tucks_the_chip_inside_the_bottom():
    """At a desk the person nearest the lens IS the box that fills the
    frame, so this is the common case rather than an edge one."""
    picture = (0, 0, 200, 100)
    x, y, anchor = pv.name_anchor((0, 0, 200, 100), picture, 3, 12)
    assert anchor == "sw"
    assert 0 <= y <= 100


def test_the_chip_is_pulled_back_inside_the_picture():
    """A chip anchored to a face at the right edge would otherwise run into
    the readout column, where the pane's OTHER numbers live."""
    ns = pane()
    named = face(x=1200, y=50, w=80, h=80, name="hunter", id_score=0.74,
                 id_ran=True)
    c = paint(ns, live(named))
    ox, _oy = ns._origin
    right = ox + ns._fit[0] + ns._fit[2]
    assert c.coords_of["namebg"][2] <= right
    assert c.coords_of["name"][0] < c.coords_of["b0"][0]   # it moved left


def test_the_same_chip_at_the_same_place_is_not_re_set_or_re_measured():
    """Between two detections the boxes are carried forward, so the chip
    lands at the same point picture after picture. Re-setting the text and
    asking the canvas to measure it again was two canvas operations per
    repaint for an answer it already had."""
    ns = pane()
    measured = []
    real_bbox = ns.canvas.bbox
    ns.canvas.bbox = lambda item: measured.append(item) or real_bbox(item)
    shot = live(face(name="hunter", id_score=0.74, id_ran=True))

    def text_sets():
        return [kw for item, kw in ns.canvas.calls
                if item == "name" and "text" in kw]

    paint(ns, shot)
    sets, meas = len(text_sets()), len(measured)
    assert sets >= 1 and meas >= 1
    first = (ns.canvas.coords_of["name"], ns.canvas.coords_of["namebg"])
    paint(ns, shot)                              # the carried box
    assert len(text_sets()) == sets              # nothing re-set
    assert len(measured) == meas                 # nothing re-measured
    assert ns.canvas.shown("name") and ns.canvas.shown("namebg")
    assert (ns.canvas.coords_of["name"], ns.canvas.coords_of["namebg"]) \
        == first
    paint(ns, live(face(x=140, name="hunter", id_score=0.74, id_ran=True)))
    assert len(measured) > meas                  # a moved face: measured
    paint(ns, live(face(name="hunter", id_score=0.31, id_ran=True)))
    assert len(text_sets()) > sets + 1           # a new score: re-set


def test_a_chip_shifted_off_the_edge_is_put_back_when_the_shift_goes():
    """A cached placement puts nothing down, so the item stays where the
    LAST frame's edge shift moved it. The shift is tracked so the next frame
    that needs none puts the word back under its ground."""
    ns = pane()
    edge = face(x=1200, y=50, w=80, h=80, name="hunter", id_score=0.74,
                id_ran=True)
    paint(ns, live(edge))
    shifted = ns.canvas.coords_of["name"][0]
    assert shifted < ns.canvas.coords_of["b0"][0]        # it moved left
    # The same chip, same anchor point, but the picture is now wider than
    # the box fit allowed before: no shift is needed, and the item must be
    # placed at the unshifted x rather than left where the shift put it.
    ns._name_last = ((ns._name_last[0][0], ns._name_last[0][1],
                      ns._name_last[0][2]), ns._name_last[1])
    ns._fit = (0, 0, ns._box[0] * 4, ns._box[1])
    paint(ns, live(edge))
    x = ns.canvas.coords_of["name"][0]
    bg = ns.canvas.coords_of["namebg"]
    assert bg[0] <= x <= bg[2]                           # word on its ground


def test_no_ground_means_no_word():
    """Before there is a window the canvas cannot measure, so there is no
    ground to draw -- and a word without one is a word over live video."""
    ns = pane()
    ns.canvas.bbox = lambda _item: None
    c = paint(ns, live(face(name="hunter", id_score=0.74, id_ran=True)))
    assert not c.shown("name") and not c.shown("namebg")


def test_a_dark_pane_takes_the_name_with_it():
    """A name left over a picture the pane is claiming not to have would be
    the console asserting who is in a room it says it cannot see."""
    ns = pane()
    paint(ns, live(face(name="hunter", id_score=0.74, id_ran=True)))
    c = paint(ns, cp.blank(cp.REASON_SENSING, "offline mode"))
    assert not c.shown("name") and not c.shown("namebg")


def test_an_empty_frame_takes_the_name_with_it_too():
    ns = pane()
    paint(ns, live(face(name="hunter", id_score=0.74, id_ran=True)))
    c = paint(ns, live())
    assert not c.shown("name") and not c.shown("namebg")


@pytest.mark.parametrize("look", theme.LOOKS)
def test_the_name_is_legible_over_its_own_ground_in_both_looks(look):
    """Small but READABLE, and in classic too. 4.5:1 is the WCAG AA bar for
    normal text; the chip's ground is the darkest value in the palette
    precisely so both inks clear it whatever is behind the picture."""
    theme.select_look(look)
    known = face(name="hunter", id_score=0.74, id_ran=True)
    unknown = face(name="", id_score=0.21, id_ran=True)
    for f in (known, unknown):
        assert _contrast(pv.identity_ink(f), pv.chip_ink()) >= 4.5, look
    # …and the two are told apart by brightness as well as by the word, the
    # same three-ways rule the attention verdict follows.
    assert _lum(pv.identity_ink(known)) > _lum(pv.identity_ink(unknown))


def test_the_chips_colours_are_read_at_call_time_and_not_frozen():
    theme.select_look("holo")
    holo = pv.chip_ink()
    theme.select_look("classic")
    assert pv.chip_ink() != holo


def test_the_chip_scales_with_the_ui_scale():
    """"Small but readable" is a statement about apparent size, so the chip
    is derived from the FONT rather than from a design constant that could
    drift out of step with it at a scale nobody measured at."""
    set_scale(1.0)
    theme.apply_scale(1.0)
    one = pv.name_line_h()
    set_scale(2.0)
    theme.apply_scale(2.0)
    assert pv.name_line_h() > one


def test_a_chip_that_would_cover_the_face_sheds_its_score_first():
    """Measured on the real font files: "HUNTER 0.74" is 45% of the picture
    in the console's condensed display face but 66% in the DejaVu fallback,
    and this pane cannot guarantee which one Tk resolved. So the rule is a
    measurement, not a font -- and the SCORE goes before the NAME does,
    because the name is what he asked for."""
    ns = pane()
    ns.canvas.CHAR_W = 12                        # a wide fallback face
    c = paint(ns, live(face(name="hunter", id_score=0.74, id_ran=True)))
    assert c.state["name"]["text"] == "HUNTER"   # the score, not the name
    assert c.shown("name") and c.shown("namebg")


def test_the_score_is_never_drawn_over_the_picture_however_much_room_there_is():
    """The caption used to carry the score and DROP it when the pane was
    narrow, so the same face read "HUNTER 0.74" or "HUNTER" depending on
    the width -- which is how he noticed it at all. Width must not change
    what the caption says."""
    for char_w in (4, 8, 16):
        ns = pane()
        ns.canvas.CHAR_W = char_w
        c = paint(ns, live(face(name="hunter", id_score=0.74, id_ran=True)))
        assert c.state["name"]["text"] == "HUNTER", char_w
        assert "0.74" not in c.state["name"]["text"], char_w


def test_a_name_too_wide_even_on_its_own_is_trimmed_not_run_off_the_edge(
        monkeypatch):
    """A long enrolled label at a large UI scale. Half a name he can see
    beats a whole one he cannot."""
    wide = 40                                    # px per character
    monkeypatch.setattr(pv, "ellipsize",
                        lambda t, _f, room: t[:max(0, room // wide - 1)] + "…")
    ns = pane()
    ns.canvas.CHAR_W = wide
    c = paint(ns, live(face(name="hunter", id_score=0.74, id_ran=True)))
    assert c.state["name"]["text"] == "HU…"      # trimmed, not clipped
    ox, _oy = ns._origin
    left, right = ox + ns._fit[0], ox + ns._fit[0] + ns._fit[2]
    assert left <= c.coords_of["namebg"][0]
    assert c.coords_of["namebg"][2] <= right     # …and it fits the picture


def test_the_chip_is_a_caption_not_a_banner():
    """"Small but readable" cuts both ways. The picture is 76 design units
    tall and 136 wide, so a chip that took a third of it would hide the face
    it is naming -- which is why it is the pane's CONDENSED display face at
    the one annotation size, not the wider UI face."""
    assert pv.name_font() == pv.ui_display(theme.SIZE_CAPTION, "semibold")
    for scale in (1.0, 2.0):
        set_scale(scale)
        theme.apply_scale(scale)
        assert pv.name_line_h() <= px(pv.PANE_H) // 3, scale


def test_the_score_he_reads_is_the_one_that_was_measured():
    """He reads this pane to understand behaviour. 0.74 against his measured
    p50 of 0.739 and a 0.363 bar is checkable; a bare name is not."""
    assert pv.identity_text(face(name="hunter", id_score=0.739,
                                 id_ran=True)) == "HUNTER 0.74"
    assert pv.identity_text(face(id_ran=False)) == ""
    assert pv.identity_text(None) == ""


# ------------------------------------- letting go of the last frame
class FakePhoto:
    """``ImageTk.PhotoImage``'s seam. It HOLDS the picture, which is the
    whole point of the two tests below: a photo that is merely hidden is a
    frame the process -- and Tk's image store -- is still keeping."""

    def __init__(self, image):
        self.image = image

    def paste(self, image):
        self.image = image


def photo_pane(monkeypatch, scale=1.0):
    """``pane()``, but running the REAL ``_show_image`` against a stub
    PhotoImage. The default harness stubs that method out, so ``_photo``
    -- the one surviving reference to a frame -- is never exercised by it.

    No picture exists here either: what is handed in is an object with a
    width and a height, because nothing in this file may look at pixels.
    """
    ns = pane(scale=scale)
    ns._show_image = lambda image: CameraPreview._show_image(ns, image)
    monkeypatch.setattr("PIL.ImageTk.PhotoImage", FakePhoto)
    return ns


def picture(ns):
    return SimpleNamespace(width=ns._box[0], height=ns._box[1])


def test_going_dark_releases_the_frame_and_does_not_merely_hide_it(
        monkeypatch):
    """A frozen last frame under the words "camera off" would be the console
    showing him a view it is claiming not to have -- and hiding the canvas
    item does not stop it being one. PhotoImage keeps the picture in a Tk
    image buffer for as long as anything references it, so the reference has
    to go too."""
    ns = photo_pane(monkeypatch)
    img = picture(ns)
    paint(ns, cp.PreviewShot(image=img, faces=(face(),), cap_w=1280,
                             cap_h=720, reason=cp.REASON_LIVE, seq=1))
    assert ns._photo is not None and ns._photo.image is img   # it IS held
    assert ns._photo_size == ns._box

    c = paint(ns, cp.blank(cp.REASON_SENSING, "offline mode"))
    assert c.state["img"]["state"] == "hidden"
    assert c.state["img"]["image"] == ""     # …and Tk is not left the name
    assert ns._photo is None, "the last frame is still in the Tk image store"
    assert ns._photo_size == (0, 0)


def test_the_switch_going_off_clears_a_pane_that_will_never_paint_again(
        monkeypatch):
    """The OFF paths do not repaint: MainWindow._preview_apply stops the
    pane, stops the capture and pack_forgets the widget. A stop that only
    cancelled the timer would unmap a pane still displaying -- and still
    holding -- the frame he just switched off, and on the next pack it would
    be remapped visible."""
    ns = photo_pane(monkeypatch)
    paint(ns, cp.PreviewShot(image=picture(ns), faces=(face(),), cap_w=1280,
                             cap_h=720, reason=cp.REASON_LIVE, seq=7))
    assert ns._photo is not None
    cancelled = []
    ns._job = "after#1"
    ns.after_cancel = cancelled.append
    ns._clear_picture = lambda: CameraPreview._clear_picture(ns)

    CameraPreview.stop(ns)

    assert cancelled == ["after#1"]              # the poll is off…
    assert ns._photo is None                     # …and so is the picture
    assert ns.canvas.state["img"]["state"] == "hidden"
    assert ns.canvas.state["img"]["image"] == ""
    for item in ns._boxes + ns._shadows + ns._brackets:
        assert ns.canvas.state[item]["state"] == "hidden"
    # The next shot always repaints: a poll that found the same sequence
    # number and returned early would leave the pane dark over a live one.
    assert ns._seq == -1


def test_stopping_a_pane_that_was_never_started_is_still_a_clear(monkeypatch):
    """stop() used to return early when there was no timer. The clear has to
    happen anyway -- quit and the standby edge both reach it that way."""
    ns = photo_pane(monkeypatch)
    paint(ns, cp.PreviewShot(image=picture(ns), cap_w=1280, cap_h=720,
                             reason=cp.REASON_LIVE, seq=1))
    ns._job = None
    ns._clear_picture = lambda: CameraPreview._clear_picture(ns)
    CameraPreview.stop(ns)
    assert ns._photo is None


def test_a_pane_whose_canvas_has_gone_still_stops_cleanly(monkeypatch):
    """stop() runs on the quit path, after the toplevel may already be on
    its way out. A clear that raised there would take the teardown with it."""
    ns = photo_pane(monkeypatch)
    paint(ns, cp.PreviewShot(image=picture(ns), cap_w=1280, cap_h=720,
                             reason=cp.REASON_LIVE, seq=1))

    def gone(*_a, **_kw):
        raise RuntimeError("the canvas is destroyed")
    ns.canvas.itemconfigure = gone
    ns._job = None
    ns._clear_picture = lambda: CameraPreview._clear_picture(ns)
    CameraPreview.stop(ns)                       # must not raise
    assert ns._photo is None                     # …and it still let go


def test_a_poll_that_finds_the_same_frame_repaints_nothing():
    """What makes polling at twice the capture rate affordable next to a
    60 fps animation loop: a no-op poll is an int compare."""
    ns = pane()
    ns._paint = lambda shot: painted.append(shot)
    painted = []
    ns.worker = SimpleNamespace(latest=lambda: live(face()))
    CameraPreview.refresh(ns)
    CameraPreview.refresh(ns)
    CameraPreview.refresh(ns)
    assert len(painted) == 1


def test_a_pane_with_no_worker_does_nothing_at_all():
    ns = pane()
    ns.worker = None
    CameraPreview.refresh(ns)                    # must not raise
    assert ns.canvas.calls == []


# ------------------------------------------- the window's show/hide switch
class FakePane:
    def __init__(self):
        self.packs = 0
        self.forgets = 0
        self.started = 0
        self.stopped = 0

    def pack(self, **_kw):
        self.packs += 1

    def pack_forget(self):
        self.forgets += 1

    def start(self, *_a):
        self.started += 1

    def stop(self):
        self.stopped += 1


class FakeWorker:
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.started_with = []
        self.joined = []

    def start(self, enabled=None):
        self.starts += 1
        self.started_with.append(enabled)
        return True

    def stop(self, timeout=2.0, join=True):
        self.stops += 1
        self.joined.append(join)


def window(mode=ACTIVE, enabled=True):
    ns = SimpleNamespace(
        preview=FakePane(), preview_worker=FakeWorker(),
        modes=SimpleNamespace(mode=mode), _preview_shown=False,
        _console_option=lambda key, default=None:
            enabled if key == cp.OPTION_ENABLED else default)
    ns._preview_enabled = lambda: MainWindow._preview_enabled(ns)
    ns._preview_apply = lambda m=None, enabled=None: \
        MainWindow._preview_apply(ns, m, enabled)
    return ns


def test_going_to_standby_hides_the_pane_and_stops_the_capture():
    """Hiding and stopping are ONE decision. A hidden pane whose thread kept
    grabbing would be a lit camera serving a screen nobody is looking at."""
    ns = window()
    ns._preview_apply()
    assert ns.preview.packs == 1 and ns.preview_worker.starts == 1
    ns._preview_apply(STANDBY)
    assert ns.preview.forgets == 1
    assert ns.preview_worker.stops == 1
    assert ns.preview.stopped == 1               # the repaint timer too
    # …and it does NOT wait for the device here. This runs on the Tk thread
    # and the edge recurs every 45 s of quiet; a 130 ms grab (his measured
    # LifeCam figure) would be eight consecutive 60 Hz slots of frozen
    # console. The deny is synchronous either way -- only the handback moves.
    assert ns.preview_worker.joined == [False]


def test_the_toggle_off_stops_the_capture_even_in_the_active_console():
    ns = window(enabled=False)
    ns._preview_apply()
    assert ns.preview.packs == 0
    assert ns.preview_worker.starts == 0
    assert ns.preview_worker.stops == 1          # and it is told to release
    # The pane is stopped too, and THAT is what drops the picture: the
    # widget is unmapped without ever painting again.
    assert ns.preview.stopped == 1


def test_the_switch_is_idempotent_so_a_mode_tick_does_not_repack():
    ns = window()
    for _ in range(4):
        ns._preview_apply()
    assert ns.preview.packs == 1
    assert ns.preview.forgets == 0


def test_a_console_with_no_pane_at_all_survives_the_mode_change():
    """A camera must not be able to stop the console from working."""
    ns = window()
    ns.preview = None
    ns._preview_apply(STANDBY)                   # must not raise


def test_the_worker_is_stopped_even_when_the_pane_refuses():
    class Angry(FakePane):
        def pack_forget(self):
            raise RuntimeError("no")
    ns = window()
    ns._preview_apply()
    ns.preview = Angry()
    ns.preview.__dict__["stop"] = lambda: None
    ns._preview_apply(STANDBY)


def test_the_toggles_new_value_is_used_rather_than_re_read():
    """SettingsDrawer._set_option writes assistant.json on a daemon thread
    and echoes to the console immediately. Re-reading here would race that
    write and could act on the value he just changed away from -- and the
    WORKER re-read it too, so a stale False would pack the pane and then
    refuse to capture behind it, saying "OFF -- preview off"."""
    ns = window(enabled=False)               # what the file still says
    ns._preview_apply(enabled=True)          # what he just clicked
    assert ns.preview.packs == 1
    assert ns.preview_worker.starts == 1
    assert ns.preview_worker.started_with == [True]   # …handed on, not re-read


def test_a_capture_that_refuses_to_start_still_leaves_the_pane_polling():
    """The two used to share one try, so a worker that raised skipped
    pane.start() -- which is what clears the band on its first poll. The
    pane would be remapped showing whatever it had last, with no timer left
    to fix it."""
    class Angry(FakeWorker):
        def start(self, enabled=None):
            raise RuntimeError("no thread for you")
    ns = window()
    ns.preview_worker = Angry()
    ns._preview_apply()
    assert ns.preview.packs == 1
    assert ns.preview.started == 1


def test_the_pane_polls_at_twice_the_configured_capture_rate():
    """A pane polling at 83 ms for a capture running at 10 fps would hold
    every other frame back a full period."""
    ns = pane()
    ns._job = None
    ns.worker = SimpleNamespace(fps=10.0, latest=lambda: live())
    ns._tick = lambda: None
    CameraPreview.start(ns)
    assert ns._interval == cp.poll_ms(10.0) == 50
    ns2 = pane()
    ns2.worker = SimpleNamespace(latest=lambda: live())   # an older worker
    ns2._job = None
    ns2._tick = lambda: None
    CameraPreview.start(ns2)
    # …and a worker that cannot say falls back to the CONFIGURED default
    # rather than to a literal, which is how the pane ended up polling for
    # 6 fps under a capture that had been raised.
    assert ns2._interval == cp.poll_ms(cp.DEFAULT_FPS)


# --------------------------------------------------- the settings row
def test_the_settings_row_writes_the_key_the_console_listens_for():
    """One key, spelled once: jarvis/campreview.py owns it, the drawer
    imports it and the window compares against the same import."""
    from jarvis.ui.main_window import CAMERA_PREVIEW_OPTION
    from jarvis.ui.views import CAMERA_PREVIEW_OPTION as DRAWER_KEY
    assert CAMERA_PREVIEW_OPTION == DRAWER_KEY == cp.OPTION_ENABLED
    assert cp.OPTION_ENABLED == "camera.preview"


def test_the_row_is_re_read_when_the_drawer_opens_even_with_no_policy():
    """The camera preview is an assistant.json option and does not need a
    sensing owner to be read, so a box whose policy failed to construct
    must still show the truth about the pane."""
    from jarvis.ui.views import SettingsDrawer
    seen = []
    ns = SimpleNamespace(
        services=None,
        _preview_toggle=SimpleNamespace(
            set=lambda value, animate=True: seen.append((value, animate))),
        _get_option=lambda key, default=False: key == cp.OPTION_ENABLED,
        _sensing=lambda: None)
    SettingsDrawer._refresh_privacy(ns)
    assert seen == [(True, False)]           # read back, and NOT animated —
    # animate=False also means Toggle.command is not called, so opening the
    # drawer cannot write the switch it just read.


def test_the_echo_is_the_same_callback_the_config_rows_use():
    from jarvis.ui.views import SettingsDrawer
    told = []
    ns = SimpleNamespace(on_config_change=lambda n, v: told.append((n, v)))
    SettingsDrawer._echo(ns, cp.OPTION_ENABLED, True)
    assert told == [(cp.OPTION_ENABLED, True)]
    # …and an unwired console is not an error
    SettingsDrawer._echo(SimpleNamespace(on_config_change=None),
                         cp.OPTION_ENABLED, True)


def test_services_declares_the_camera_feed_the_app_will_hand_over():
    """build_ui_services keeps only the fields this dataclass declares, so
    an app half that passes camera_feed would have it silently dropped --
    and the preview would build a SECOND gated feed, which takes the curfew
    away from the app's (SensingPolicy.attach replaces by name)."""
    import dataclasses

    from jarvis.ui.main_window import Services
    names = {f.name for f in dataclasses.fields(Services)}
    assert "camera_feed" in names
    assert Services().camera_feed is None
    assert cp.resolve_feed(Services(camera_feed="the app's"), None, None,
                           None) == ("the app's", "", False)


# ---------------------------------------------------------- the privacy rule
def test_no_function_in_the_pane_hands_a_picture_back_out():
    """The image goes ONE way: capture thread -> this canvas. A helper that
    returned one would be the first step to a second consumer."""
    source = (pv.__file__ and open(pv.__file__).read()) or ""
    assert "return self._photo" not in source
    assert not re.search(r"def \w+\([^)]*\)\s*->\s*(Image|ImageTk)", source)
    assert "shot.image" in source                # it is only ever handed on
