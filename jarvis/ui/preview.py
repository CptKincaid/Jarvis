"""The camera pane: a small live view with the tracking drawn on it.

Hunter, 2026-09-02: *"lets add a small camera with visable tracking on the
jarvis app but make me be able to turn if off in settings"*.

WHERE IT SITS, AND WHY THERE. The console is header / reactor stage /
transcript / command bar / status strip, and the pane is packed
``side="bottom"`` so it lands directly ABOVE the command bar -- a band the
width of the panel, ~76 design px of picture with its readouts beside it.
The three alternatives were each worse in a specific way:

* inside the reactor stage. The stage's decor is baked at fixed canvas
  coordinates and rebuilt only on a size settle; a live child in there is
  exactly the widget churn behind the 2026-08-26 desktop freeze, and the
  pane would be competing for the same canvas the 60 fps loop redraws.
* over the transcript, the way ``RoomSlab`` is. The slab is allowed to
  cover the conversation because it only appears when there IS no
  conversation; a camera pane doing it would hide what he is reading.
* in the header, beside the sensing badge. There is room for a 20 px chip
  there and nothing else, and a 20 px preview is a coloured smudge.

Above the command bar it is glanceable without being in the way, it costs
the transcript ~76 design px instead of the reactor's stage, and hiding it
is the same ``pack_forget`` the footer already does in standby.

WHAT THE OVERLAY DRAWS, AND WHY IT IS NOT JUST A BOX. Measured on his
LifeCam on 2026-09-02: looking at his screen reads p50 **54.1 deg** of head
yaw, looking at the camera **14.1 deg**, against a 20 deg attention cone.
That is a real, wide separation, so the pane can honestly show whether he is
addressing Jarvis rather than merely that a face exists. The verdict is
carried THREE ways, because this pane will be read at a glance from across a
desk and colour alone does not survive a dimmed monitor or a colour-blind
glance -- the same rule ``jarvis/ui/sensing_badge.py`` settled on:

  * the WORD ("LOOKING AT JARVIS" / "LOOKING AWAY"),
  * the box COLOUR (focal white against dim cyan),
  * the box SHAPE (corner brackets close on the tracked face when it is
    inside the cone; a plain hairline rectangle when it is not).

The numbers -- confidence, yaw in degrees -- are printed BESIDE the picture,
never on it. Text over live video is text over an unknown background, and
the one readout in this pane that has to be legible in every frame is the
one saying whether the camera is on.

THE ONE THING THAT IS ON THE PICTURE IS THE NAME, and it brings a ground with
it. Hunter, 2026-09-03: *"lets have the identity of the person its tracking
next to their name, small but readable"* -- a caption beside the picture would
not have answered that, because with two faces in frame it would not say WHICH
one it meant. So the identity chip is drawn against the tracked box, and the
rule above is honoured rather than waived: the word sits on a filled rectangle
in the window ground (``chip_ink``), sized from the canvas's own measurement of
the text, so its contrast is a fact about two palette tokens rather than a hope
about what is behind him. It reads "HUNTER 0.74" or "UNKNOWN 0.21", and it is
ABSENT when identity did not run -- three states, because "not looking" and
"do not recognise you" are opposite facts about the same picture.

Nothing the chip says is a permission. See jarvis/campreview.py: identity may
remove capability or add a name and may never grant one, and the structural
guarantee is that nothing outside jarvis/ui/ can see a PreviewFace at all.

WHAT IT COSTS THE 60 fps LOOP. On a scratch Xvfb at his UI scale (S=2.0), a
real reactor animating on 16.67 ms slot boundaries, 30 s windows after a 12 s
warm-up, with a synthetic 1280x720 source (2026-09-02; no camera was opened) --

  A  reactor alone                     0 late slots / 1801   max late 1.6-4.3 ms
  B  reactor + this pane at 6 fps    0-1 late slots / 1802   max late 1.4-3.8 ms
  C  the same capture ON the Tk loop 744-771 / ~1800 (41-43%)  max late 137 ms

C is the counterfactual this whole design exists to avoid, and it is not a
near miss: two slots in five late is the console stopping dead several
refreshes at a time. B is indistinguishable from A across five paired runs in
both orders (a "late slot" is one starting >8 ms after its boundary; 0 or 1 in
1800 either way), because the only Tk-thread work per frame is one photo paste
plus a couple of dozen coords / itemconfigure calls: **0.32-0.39 ms mean,
0.88 ms worst observed**, against a 16.67 ms slot.

THAT TABLE COULD NOT BE RE-RUN FOR THE 15 fps CHANGE, and pretending otherwise
would be worse than saying so: there is no Xvfb on this box, and the only
display is the one his LIVE console is running on -- window churn there is what
froze his desktop on 2026-08-26. So what was measured instead (2026-09-03) is
the thing that table scales with, on the same display-free harness the suite
uses:

  * the Tk-thread WORK per repaint, in canvas operations -- 24 before this
    change, 26 with identity off, 30 with a name on screen (one photo paste,
    the boxes, the brackets, and the chip's measure-and-place). Against the
    0.32-0.39 ms measured for ~24 operations that is ~0.4-0.5 ms mean and
    ~1.1 ms worst: an eighth of the 8 ms late bar, a fifteenth of a slot. A
    repaint cannot make a slot late on its own arithmetic.
  * the RATE those land at, which is the DEVICE's delivered rate and not the
    configured one: a repaint happens only when a new shot has landed, and
    his LifeCam delivers 3.7-7.5 fps at 1280x720 whatever is asked (measured
    2026-09-03; see jarvis/campreview.py's module docstring for the table).
    So the pane's whole claim on the Tk thread is ~1.5-3 ms per second of the
    1000 ms the mainloop has, rising to ~15 ms if the light ever lets the
    camera reach the 30 fps cap. What changes is how MANY slots carry an
    extra half millisecond, not whether any of them can blow their budget.
  * the POLL rate, which does follow the configured ceiling
    (``campreview.poll_ms``): 12 a second at the old 6 fps default, 15 at
    the 7.5 default, 30 at his configured 15, 50 at the 30 cap. A poll that
    finds nothing new -- most of them, at the device's real rate -- is one
    attribute read and an int compare on ``PreviewShot.seq``.

The residual risk is stated rather than measured away: repaints at the
device's rate are that many chances a second to land on a slot that was
already tight for another reason, and the cap allows thirty where the old
default was six. The reactor prints ``avatar: late slots N/M`` on its own
window; if that starts moving after this, ``camera.preview_fps`` is the dial
and it goes DOWN as well as up.

A BLACK RECTANGLE IS NOT AN ANSWER. When there is no picture the pane says
which of the reasons it is -- his toggle, offline mode, the curfew (with the
hour it lifts), the fail-safe, a camera that is not plugged in, a vision lane
that is not installed. "camera off (curfew until 7 am)" is checkable; a dark
box is indistinguishable from a broken feature, and he would have no way to
tell which he was looking at.

NO PIXELS REACH ANYTHING BUT THIS CANVAS, AND NOT FOR LONGER THAN THE VIEW
IS ON. The image handed in has already been reduced to the pane's own box by
``jarvis/campreview.shrink`` on the capture thread; this module converts it to
a Tk photo and pastes it into ONE canvas item. Nothing is saved, nothing is
published, no function here returns an image, and
``tests/test_camera_preview.py`` greps this file for the ways a frame could
reach a disk. When the view goes off -- his toggle, the sensing owner, the
ambient surface, quit -- ``_clear_picture`` drops the photo as well as hiding
it, because a hidden ``PhotoImage`` is a frame the process is still holding.

EVERY COLOUR IS READ AT CALL TIME. ``select_look`` re-derives the tokens at
runtime, so a module constant such as ``BOX = theme.CYAN`` would freeze the
import-time look -- the trap named at the top of jarvis/ui/theme.py. Both
looks are checked by test, because the pane has to be legible in the classic
fallback too.
"""
from __future__ import annotations

import time
import tkinter as tk
from typing import Optional

from jarvis.campreview import (DEFAULT_FPS, DETAIL_DETECTOR_FAILED, MAX_FACES,
                               REASON_DISABLED,
                               REASON_LIVE, REASON_NO_FRAME, REASON_PIPELINE,
                               REASON_SENSING, REASON_WAITING, PreviewShot,
                               poll_ms)
# Re-exported, not used here: the letterbox is computed on the CAPTURE
# thread (campreview.grab), and this module derives the drawn rectangle
# from the picture that actually arrived (image_fit). It lives beside the
# rest of the pane's geometry for anyone reading this file, and
# tests/test_camera_preview.py checks the two agree.
from jarvis.campreview import fit_box  # noqa: F401
from jarvis.logs import get_logger
from jarvis.ui import theme
from jarvis.ui.widgets import (ellipsize, frame_rect, px, ui_display,
                               ui_font, ui_mono)

log = get_logger("ui.preview")

# Design units at the 96-dpi baseline; every one goes through px().
PANE_W, PANE_H = 136, 76          # the picture box -- 16:9 to within a pixel
PAD_Y = 8                         # band padding above and below the box
COL_GAP = 14                      # picture -> readout column
LABEL_W = 54                      # readout label column width
ROW_H = 16                        # readout row pitch
HEAD_DROP = 2                     # heading baseline inside the box top
BRACKET_ARM = 5                   # attention bracket arm, design units
MIN_BOX = 6                       # a face box never draws smaller than this
NAME_GAP = 3                      # tracked box -> identity chip
NAME_PAD = 3                      # chip padding around its text
# How much of the picture's width the chip may take before it drops its
# score. Measured, not chosen: see identity_name().
NAME_MAX_FRAC = 0.6

# The three tones the readouts use. Word first, colour second, shape third
# -- see the module docstring.
TONE_LIVE = "live"
TONE_AWAY = "away"
TONE_OFF = "off"

STATE_WORDS = {
    REASON_LIVE: "LIVE",
    REASON_DISABLED: "OFF",
    REASON_SENSING: "CAMERA OFF",
    REASON_PIPELINE: "NO CAMERA",
    REASON_NO_FRAME: "NO SIGNAL",
    REASON_WAITING: "…",
}

ATTEND_YES = "LOOKING AT JARVIS"
ATTEND_NO = "LOOKING AWAY"
ATTEND_NONE = "NO FACE IN FRAME"
# A LIVE picture the detector could not look at. Two words that need
# opposite responses from him -- "the detector fell over" and "nobody is
# there" -- used to print the same NO FACE IN FRAME / FACES 0 (F56).
ATTEND_FAILED = "DETECTOR FAILED"
ATTEND_NO_DETECTOR = "NO DETECTOR"
# What a face that matched nothing is called. Not blank, and not the last
# name seen: "asked, and nobody in the gallery is this person" is an answer
# and has to look like one.
IDENT_UNKNOWN = "UNKNOWN"


# ------------------------------------------------------------ pure layout
def band_height() -> int:
    """The whole band's height in device pixels, at the current UI scale."""
    return px(PANE_H) + 2 * px(PAD_Y)


def image_fit(image, box: tuple) -> tuple:
    """``(ox, oy, w, h)`` for a picture already scaled by the capture thread,
    centred in the pane's box. Falls back to the whole box when there is no
    image to measure."""
    try:
        w, h = int(image.width), int(image.height)
    except Exception:                       # noqa: BLE001 - not an image
        return 0, 0, int(box[0]), int(box[1])
    w, h = max(1, min(int(box[0]), w)), max(1, min(int(box[1]), h))
    return (int(box[0]) - w) // 2, (int(box[1]) - h) // 2, w, h


def face_rect(face, cap_w: int, cap_h: int, fit: tuple) -> tuple:
    """A detection in CAPTURE pixels -> ``(x0, y0, x1, y1)`` inside the
    picture, clamped to it.

    Clamped rather than dropped: a face half out of frame is exactly when a
    tracking overlay earns its keep, and a rectangle that vanished at the
    edge would read as a lost detection.
    """
    ox, oy, w, h = fit
    if cap_w <= 0 or cap_h <= 0:
        return ox, oy, ox, oy
    sx, sy = w / float(cap_w), h / float(cap_h)
    x0 = ox + face.x * sx
    y0 = oy + face.y * sy
    x1, y1 = x0 + face.w * sx, y0 + face.h * sy
    lo_x, hi_x, lo_y, hi_y = ox, ox + w, oy, oy + h
    x0, x1 = max(lo_x, min(hi_x, x0)), max(lo_x, min(hi_x, x1))
    y0, y1 = max(lo_y, min(hi_y, y0)), max(lo_y, min(hi_y, y1))
    if x1 - x0 < MIN_BOX:
        x1 = min(hi_x, x0 + MIN_BOX)
    if y1 - y0 < MIN_BOX:
        y1 = min(hi_y, y0 + MIN_BOX)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def bracket_points(rect: tuple, arm: int) -> tuple:
    """The four corner brackets for the ATTENDING shape.

    Shape, not just colour: this is the difference the pane is read for, and
    ``sensing_badge`` already established that the state which matters must
    not be carried by a hue alone.
    """
    x0, y0, x1, y1 = rect
    arm = max(2, min(int(arm), (x1 - x0) // 2 or 2, (y1 - y0) // 2 or 2))
    return (
        (x0, y0 + arm, x0, y0, x0 + arm, y0),
        (x1 - arm, y0, x1, y0, x1, y0 + arm),
        (x1, y1 - arm, x1, y1, x1 - arm, y1),
        (x0 + arm, y1, x0, y1, x0, y1 - arm),
    )


def identity_text(face) -> str:
    """"HUNTER 0.74" / "UNKNOWN 0.21", or "" when identity did not run.

    NOT DRAWN OVER THE PICTURE ANY MORE, and kept only because the score is
    worth having a formatter for. The preview caption is ``identity_name``:
    the name alone. See the comment at the caption ladder for why.

    THREE STATES, NOT TWO, and the empty string is the one that is easy to
    get wrong. No chip means identity was not ASKED -- ``camera.identity`` is
    off, the SFace weights are missing, nobody is enrolled, or this is not
    the tracked face. UNKNOWN means it was asked and nothing matched. A pane
    that drew those the same way would leave him unable to tell "Jarvis is
    not looking" from "Jarvis does not know you", which are opposite facts
    about the same picture.

    The SCORE rides along because he reads this pane to understand
    behaviour, not to be reassured by it: 0.74 against his measured p50 of
    0.739 and a 0.363 bar is a number he can check, and an UNKNOWN sitting
    at 0.35 tells him the bar is the problem rather than the camera.
    """
    who = identity_name(face)
    if not who:
        return ""
    return "%s %.2f" % (who, float(getattr(face, "id_score", 0.0)))


def identity_name(face) -> str:
    """The chip WITHOUT its score -- what it falls back to when the full
    caption would cover the face it is naming.

    The score is dropped before the name is, and that ordering is the whole
    point: the name is what he asked for and the score is the evidence
    behind it. Measured on the real font files 2026-09-03 at S=2, against
    the 272 px picture: "HUNTER 0.74" is 122 px in Rajdhani SemiBold (the
    console's display face, 45%) but 179 px in the DejaVu fallback (66%),
    which is a dark bar across two thirds of a small preview. One rule
    covers both rather than a font this pane cannot guarantee.
    """
    if face is None or not getattr(face, "id_ran", False):
        return ""
    name = (getattr(face, "name", "") or "").strip()
    return name.upper() if name else IDENT_UNKNOWN


def identity_ink(face) -> str:
    """The chip's text colour. A NAME is a focal value -- the pane's
    brightest ink, the same step the numbers use. An UNKNOWN is MUTED: still
    over the 4.5:1 bar on the chip's own ground, but visibly a lesser claim,
    so the two never have to be read letter by letter to be told apart."""
    known = bool(face is not None and (getattr(face, "name", "") or "").strip())
    return theme.FOCAL if known else theme.MUTED


def chip_ink() -> str:
    """The chip's fill: the window ground, the darkest value the palette has.

    THE CHIP EXISTS BECAUSE OF A RULE THIS PANE ALREADY MADE. Text over live
    video is text over an unknown background, and the module docstring's
    answer was to keep every readout beside the picture. He asked for this
    one to sit next to the face, so instead of relaxing the rule the text
    brings its own ground with it -- the same trick the face boxes use with
    their dark halo, and it is what lets the contrast be a fact rather than
    a hope about what is behind him.
    """
    return theme.BG


def name_font() -> tuple:
    """The chip's face: the pane's DISPLAY face at the one annotation size.

    Display rather than the UI face, for two reasons that point the same
    way. It is what this pane already uses for its other verdicts (the state
    word, LOOKING AT JARVIS), and a name is a verdict rather than a label.
    And it is CONDENSED -- at the same legible size it covers noticeably
    less of a 136-unit-wide picture than the UI face would, which is the
    whole difference between "small but readable" and a caption that hides
    the face it is captioning.
    """
    return ui_display(theme.SIZE_CAPTION, "semibold")


def name_line_h() -> int:
    """The chip's height in device pixels, derived from the FONT rather than
    from a design constant, so it cannot drift out of step with the text it
    has to enclose at a scale nobody measured at."""
    return abs(int(name_font()[1])) + 2 * px(NAME_PAD)


def name_anchor(rect: tuple, picture: tuple, gap: int, line_h: int) -> tuple:
    """``(x, y, anchor)`` for the identity chip: under the tracked box.

    Under, not over: the chip is a caption on the face and a caption sits
    below its subject. It flips ABOVE when the box is against the bottom of
    the picture, and tucks inside the bottom edge when the face fills the
    frame -- which at a desk is the common case for the person nearest the
    lens, so it is not an edge case worth getting wrong.

    ``picture`` is ``(x, y, w, h)`` of the drawn frame in canvas
    coordinates; the chip never leaves it, because a caption hanging in the
    panel beside the picture is a readout with no ground under it.
    """
    x0, y0, _x1, y1 = rect
    px0, py0, _pw, ph = picture
    bottom = py0 + ph
    if y1 + gap + line_h <= bottom:
        return x0, y1 + gap, "nw"
    if y0 - gap - line_h >= py0:
        return x0, y0 - gap, "sw"
    return x0, bottom - gap, "sw"


def state_word(shot: PreviewShot) -> str:
    return STATE_WORDS.get(shot.reason, "OFF")


def state_tone(shot: PreviewShot) -> str:
    if shot.reason != REASON_LIVE:
        return TONE_OFF
    face = shot.primary
    return TONE_LIVE if (face is not None and face.attending) else TONE_AWAY


def attention_line(shot: PreviewShot) -> tuple:
    """``(text, tone)`` for the verdict row.

    A live shot with no faces and a ``detail`` is one where nothing could
    be LOOKED FOR -- the detector raised, or there is none -- and that is
    printed as such, in the warning ink, rather than as an empty room.
    """
    if shot.reason != REASON_LIVE:
        return shot.detail or "", TONE_OFF
    face = shot.primary
    if face is None:
        if shot.detail.startswith(DETAIL_DETECTOR_FAILED):
            return ATTEND_FAILED, TONE_OFF
        if shot.detail:
            return ATTEND_NO_DETECTOR, TONE_OFF
        return ATTEND_NONE, TONE_AWAY
    return (ATTEND_YES, TONE_LIVE) if face.attending else (ATTEND_NO,
                                                           TONE_AWAY)


def readout_rows(shot: PreviewShot) -> tuple:
    """``((label, value), ...)`` for the two number rows.

    Empty strings rather than "-" when there is nothing measured: a dash
    reads as a value that came back zero, and the row above already says
    whether anything is being measured at all.
    """
    face = shot.primary
    if shot.reason != REASON_LIVE:
        # Not "FACES  --": a labelled row with nothing in it reads as a
        # measurement that came back blank. There is no measurement; the
        # state word and the reason line below carry the whole story.
        return (("", ""), ("", ""))
    if face is None:
        # "0" is a count; a detector that could not look counted nothing,
        # and the row says so by being blank (see attention_line).
        return (("FACES", "" if shot.detail else "0"), ("YAW", ""))
    conf = ("CONF", "%.2f" % face.conf)
    yaw = ("YAW", "%+.0f°" % face.yaw_deg if face.landmarks_ok else "—")
    if len(shot.faces) > 1:
        conf = ("CONF", "%.2f  (%d faces)" % (face.conf, len(shot.faces)))
    return (conf, yaw)


def tone_ink(tone: str) -> str:
    """Text colour for a tone, in the CURRENT look. Both looks ground on a
    dark blue, so every one of these is a light ink -- cyan here would be
    structure colour used as text, which loses against classic's lifted
    ground (the rule sensing_badge.badge_colors states)."""
    if tone == TONE_LIVE:
        return theme.FOCAL
    if tone == TONE_AWAY:
        return theme.MUTED
    return theme.WARN


def box_ink(attending: bool, primary: bool = True) -> str:
    """The face rectangle's stroke. The tracked face is focal white when it
    is inside the cone and structure cyan when it is not; every other face
    is drawn in the dimmed cyan, so the subject is obvious at a glance
    while the others are still VISIBLE.

    Dimmed, not faint. The first spelling of this used RAMP33, which
    measures 2.2:1 (holo) / 2.3:1 (classic) against the dark halo drawn
    under it -- below any legibility bar, so a secondary box would read as
    a smudge or as nothing at all, which is indistinguishable from the
    detector having missed that face. CYAN_DIM is 5.9:1 / 5.2:1 and still
    well under CYAN's 12.5:1 / 11.0:1, so the hierarchy the pane is read
    for survives at both ends.
    """
    if not primary:
        return theme.CYAN_DIM
    return theme.FOCAL if attending else theme.CYAN


def halo_ink() -> str:
    """The dark underlay a face box is drawn on top of. The window ground in
    both looks -- the darkest value the palette has, which is what a halo
    over an unknown video background needs to be."""
    return theme.BG


def chrome_ink() -> str:
    """The picture box's 1px outline -- GLASS_EDGE in BOTH looks.

    This branched on the look at first, on the reasoning that classic's
    lifted ground wanted the quieter separator token. Measured, that is
    backwards: classic LINE is 1.44:1 against the box fill (SURFACE) and
    1.65:1 against the canvas ground, while GLASS_EDGE is 3.25:1 / 3.73:1.
    And classic needs the outline MORE, not less -- its SURFACE step is
    only 1.15:1 above BG, so in the pane's dark state (its default, and its
    state until the vision lane lands) the outline is the only thing
    separating the picture box from the panel. At 1.44:1 there would be no
    box at all: readouts floating on the ground, which is exactly the "hole
    in the panel" the filled SURFACE is there to prevent.
    """
    return theme.GLASS_EDGE


def pane_visible(mode: str, enabled: bool) -> bool:
    """Show the pane only in the ACTIVE console.

    Ambient and standby are the surfaces for a room nobody is talking to --
    the transcript is behind a dimmed clock and the panel is drifting to
    avoid burn-in. A live camera view on either would be a lit lens for an
    empty chair, and the worker is stopped rather than merely hidden, so
    "not visible" really does mean "not capturing".
    """
    from jarvis.ui.console_mode import ACTIVE      # noqa: PLC0415 - no cycle
    return bool(enabled) and mode == ACTIVE


# ---------------------------------------------------------- the widget
class CameraPreview(tk.Frame):
    """The band itself: one canvas, and items that are created ONCE.

    Nothing here creates or destroys a canvas item after ``_build``, and the
    photo is pasted into rather than replaced. That is deliberate and it is
    the performance design, and raising the rate is what made it matter: at
    15 fps a per-frame ``create_image`` / ``delete`` pair would hand Tk a new
    XImage nine hundred times a minute, on the same thread the reactor is
    trying to hit 16.67 ms slot boundaries on. Updating is ``coords`` +
    ``itemconfigure`` + one ``paste``, and the identity chip joins that
    discipline rather than being the one thing recreated per frame.

    The widget never captures. It polls ``PreviewWorker.latest()`` on its
    own ``after`` chain at twice the capture rate (``campreview.poll_ms``),
    and a poll that finds the same ``seq`` returns after an int compare.
    """

    def __init__(self, parent, worker=None):
        super().__init__(parent, bg=theme.BG)
        self.worker = worker
        self._seq = -1
        self._photo = None
        self._photo_size = (0, 0)
        self._job = None
        self._fit = (0, 0, px(PANE_W), px(PANE_H))
        self._box = (px(PANE_W), px(PANE_H))
        self._interval = poll_ms(DEFAULT_FPS)
        # The identity chip's last placement, so a chip that has not moved
        # is not re-set and re-measured on every picture -- see _place_name.
        self._name_last = None
        self._name_anchor = None
        self._name_shift = 0
        self._build()

    # ------------------------------------------------------------ build
    def _build(self) -> None:
        w, h = self._box
        # The 1px rule packs FIRST so the packer gives it its row before the
        # expanding canvas claims the rest; it makes the band read as its
        # own strip above the command bar rather than as loose text on the
        # ground.
        tk.Frame(self, bg=theme.LINE, height=max(1, px(1))).pack(
            fill="x", side="bottom")
        self.canvas = tk.Canvas(self, height=band_height(), bg=theme.BG,
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        # theme.PAD is already S-scaled by theme.apply_scale; PAD_Y is a
        # design unit and goes through px(). Scaling PAD again here is the
        # double-scale trap widgets.py's header warns about.
        x0, y0 = theme.PAD, px(PAD_Y)
        self._origin = (x0, y0)

        # The picture's ground. A filled SURFACE step rather than the window
        # ground: an empty box has to read as a switched-off screen, not as
        # a hole in the panel.
        self.canvas.create_rectangle(x0, y0, x0 + w, y0 + h,
                                     fill=theme.SURFACE, outline="",
                                     tags="ground")
        self._img = self.canvas.create_image(x0, y0, anchor="nw",
                                             tags="picture")
        # Face boxes and the attention brackets, created hidden and never
        # recreated. MAX_FACES rectangles, four bracket lines for the
        # tracked face only -- three boxes is a glance; a fourth face at his
        # desk is a poster on the wall.
        #
        # Each box gets a DARK UNDERLAY drawn first and one pixel wider. A
        # stroke over live video is a stroke over an unknown background, and
        # the two brightest tokens in the palette (FOCAL, CYAN) are exactly
        # the ones that vanish against a lit wall or a window behind him.
        # The halo is the same two-stroke idea theme.GLOW_UNDER/ARC_BRIGHT
        # exports for the reactor arcs, inverted: dark under bright, so the
        # box has an edge whatever is behind it.
        self._shadows = [self.canvas.create_rectangle(
            0, 0, 0, 0, outline=theme.BG, width=max(2, px(3)),
            state="hidden", tags="overlay") for _ in range(MAX_FACES)]
        self._boxes = [self.canvas.create_rectangle(
            0, 0, 0, 0, outline=theme.CYAN, width=max(1, px(1)),
            state="hidden", tags="overlay") for _ in range(MAX_FACES)]
        self._brackets = [self.canvas.create_line(
            0, 0, 0, 0, 0, 0, fill=theme.FOCAL, width=max(1, px(2)),
            state="hidden", tags="overlay") for _ in range(4)]

        if theme.LOOK == "holo":
            frame_rect(self.canvas, x0, y0, x0 + w, y0 + h, fill="",
                       outline=chrome_ink(), notch=px(6), tags="chrome")
        else:
            self.canvas.create_rectangle(x0, y0, x0 + w, y0 + h, fill="",
                                         outline=chrome_ink(),
                                         width=max(1, px(1)), tags="chrome")

        # The readout column.
        cx = self._col_x = x0 + w + px(COL_GAP)
        cy = y0 + px(HEAD_DROP)
        self._head = self.canvas.create_text(
            cx, cy, anchor="nw", text="CAMERA",
            font=ui_display(theme.SIZE_CAPTION, "semibold"), fill=theme.RAIL)
        self._state = self.canvas.create_text(
            cx + px(LABEL_W), cy, anchor="nw", text="OFF",
            font=ui_display(theme.SIZE_CAPTION, "semibold"), fill=theme.WARN)
        self._rows = []
        for i in range(2):
            ry = cy + px(ROW_H) * (i + 1)
            label = self.canvas.create_text(
                cx, ry, anchor="nw", text="", font=ui_font(theme.SIZE_CAPTION),
                fill=theme.MUTED)
            value = self.canvas.create_text(
                cx + px(LABEL_W), ry, anchor="nw", text="",
                font=ui_mono(theme.SIZE_CAPTION), fill=theme.FOCAL)
            self._rows.append((label, value))
        self._verdict = self.canvas.create_text(
            cx, cy + px(ROW_H) * 3, anchor="nw", text="",
            font=ui_display(theme.SIZE_CAPTION, "semibold"), fill=theme.MUTED)

        # The identity chip: a ground and a word, created LAST so canvas
        # stacking (creation order) puts them over the picture, the boxes and
        # the brackets. Two items, created once and moved -- the same
        # paste-don't-replace discipline the rest of this pane keeps, because
        # they now move fifteen times a second rather than six.
        self._namebg = self.canvas.create_rectangle(
            0, 0, 0, 0, fill=chip_ink(), outline="", state="hidden",
            tags="overlay")
        self._name = self.canvas.create_text(
            0, 0, anchor="nw", text="", state="hidden", font=name_font(),
            fill=theme.FOCAL, tags="overlay")

    # ----------------------------------------------------------- polling
    def start(self, ms: Optional[int] = None) -> None:
        """Begin polling the worker. Idempotent.

        The interval comes from the WORKER's configured rate, not from a
        constant, and it is twice that rate (``campreview.poll_ms``): 67 ms
        at the 7.5 default, 33 ms at his configured 15, 20 ms at the 30 cap.
        The configured rate is a CEILING the device may not reach -- his
        LifeCam delivers 3.7-7.5 fps at 1280x720 whatever is asked (measured
        2026-09-03) -- and polling at the ceiling rather than the delivered
        rate is the right side to err on: a poll that finds nothing new is
        an int compare, while a poll slower than the device holds every
        other frame back a full period.
        """
        if self._job is not None:
            return
        rate = getattr(self.worker, "fps", None)
        self._interval = int(ms or poll_ms(rate if rate else DEFAULT_FPS))
        self._tick()

    def stop(self) -> None:
        """Stop polling AND let go of the frame on screen.

        The clear belongs here rather than only in ``_paint`` because the
        OFF paths never paint again: ``MainWindow._preview_apply`` stops
        the pane, stops the capture and then ``pack_forget``s the widget,
        so a stop that only cancelled the timer would unmap a pane still
        displaying -- and still holding -- the last frame, with no poll
        left to clear it. Putting it here covers the settings toggle, the
        ambient/standby transition and quit in one place.

        ``_seq`` is reset so the next shot always repaints: after a clear,
        a poll that found the same sequence number and returned early
        would leave the pane dark over a live capture.
        """
        job, self._job = self._job, None
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:                # noqa: BLE001 - a dead window
                log.debug("preview: after_cancel on a dead widget",
                          exc_info=True)
        self._seq = -1
        self._clear_picture()

    def _tick(self) -> None:
        try:
            self.refresh()
        except Exception:                    # noqa: BLE001 - the loop lives
            log.exception("preview: a repaint failed")
        try:
            self._job = self.after(self._interval, self._tick)
        except Exception:                    # noqa: BLE001 - a dead window
            self._job = None

    def refresh(self) -> None:
        """One poll. Cheap when nothing changed -- an attribute read and an
        int compare -- which is what makes polling at twice the capture rate
        affordable next to a 60 fps animation loop."""
        worker = self.worker
        if worker is None:
            return
        shot = worker.latest()
        if shot.seq == self._seq:
            return
        self._seq = shot.seq
        t0 = time.monotonic()
        self._paint(shot)
        if shot.live:
            # THE ONE STAGE THE CAPTURE THREAD CANNOT TIME. The repaint runs
            # here, on the Tk thread, and is posted back to the worker so the
            # once-a-minute ``campreview:`` line carries ``draw`` beside
            # ``grab`` and ``detect`` -- a slow pane and a slow camera are
            # then two different columns instead of one complaint. A number,
            # never the shot; a worker without the hook (a stub) is skipped.
            note = getattr(worker, "note_stage", None)
            if callable(note):
                try:
                    note("draw", (time.monotonic() - t0) * 1000.0)
                except Exception:            # noqa: BLE001 - a diagnostic
                    log.debug("preview: could not post the draw time",
                              exc_info=True)

    # ---------------------------------------------------------- painting
    def _paint(self, shot: PreviewShot) -> None:
        """Draw one shot. Touches ``self.canvas`` and ``self._photo`` only,
        so the suite drives it against a recording stand-in rather than a
        real toplevel (the house pattern since the 2026-08-26 freeze)."""
        ox, oy = self._origin
        tone = state_tone(shot)
        self.canvas.itemconfigure(self._state, text=state_word(shot),
                                  fill=tone_ink(tone))
        for (label, value), (text, val) in zip(self._rows, readout_rows(shot)):
            self.canvas.itemconfigure(label, text=text, fill=theme.MUTED)
            self.canvas.itemconfigure(value, text=val, fill=theme.FOCAL)
        verdict, vtone = attention_line(shot)
        self.canvas.itemconfigure(self._verdict, text=self._fit_text(verdict),
                                  fill=tone_ink(vtone))

        if not shot.live:
            # Clear the picture as well as the boxes: a frozen last frame
            # under the words "camera off" would be the console showing him
            # a view it is claiming not to have.
            self._clear_picture()
            return

        # The fit is taken FROM THE IMAGE THAT ARRIVED, not recomputed from
        # the capture geometry. The capture thread already letterboxed it
        # (campreview.grab -> fit_box), and two independent computations of
        # the same rectangle are two that can disagree -- which here would
        # mean face boxes drawn a few pixels off the faces they belong to,
        # the one thing a tracking overlay may not do.
        self._fit = image_fit(shot.image, self._box)
        if not self._show_image(shot.image):
            # No picture means no overlay. Boxes hanging in an empty ground
            # would be the pane asserting a detection on a frame it could
            # not show him -- exactly the mismatch this feature is for.
            return
        fx, fy = ox + self._fit[0], oy + self._fit[1]
        self.canvas.coords(self._img, fx, fy)
        self._draw_faces(shot)

    def _fit_text(self, text: str) -> str:
        """Trim the reason line to the width the band actually has.

        A pipeline failure carries the exception text ("...(No module named
        'jarvis.camera')"), which at S=1 is wider than the 520-unit console.
        A sentence that ran off the panel would be a stated reason he cannot
        read, which is the same as no reason at all.
        """
        try:
            room = int(self.canvas.winfo_width()) - self._col_x - theme.PAD
            if room <= 0:
                return text
            return ellipsize(text, ui_display(theme.SIZE_CAPTION, "semibold"),
                             room)
        except Exception:                    # noqa: BLE001 - no window yet
            return text

    def _clear_picture(self) -> None:
        """Hide the picture AND release its pixels.

        Hiding alone is not enough, and the difference is the whole point
        of this method. ``ImageTk.PhotoImage`` keeps the reduced frame in a
        Tk image buffer for as long as anything references it, so a merely
        hidden item would leave the last frame he was shown alive in the
        process -- and in Tk's image store, with the canvas item still
        pointing at it -- for the life of the app, underneath the words
        "camera off". Dropping the last Python reference is what makes
        ``PhotoImage.__del__`` delete the Tk image and free the buffer.

        The item is pointed at "" BEFORE the reference goes, so Tk is not
        left holding the name of an image that is being deleted.

        The cost is one ``PhotoImage`` rebuild per on/off cycle, not per
        frame, so the paste-don't-replace design in ``_show_image`` is
        untouched.
        """
        try:
            self.canvas.itemconfigure(self._img, image="", state="hidden")
        except Exception:                    # noqa: BLE001
            log.debug("preview: could not hide the picture", exc_info=True)
        self._photo = None
        self._photo_size = (0, 0)
        try:
            for item in (self._shadows + self._boxes + self._brackets
                         + [self._namebg, self._name]):
                self.canvas.itemconfigure(item, state="hidden")
        except Exception:                    # noqa: BLE001 - a dead window
            log.debug("preview: could not hide the overlay", exc_info=True)

    def _show_image(self, image) -> bool:
        """Install the frame. Returns whether there is a picture to place.

        The Tk photo is created ONCE per size and pasted into afterwards.
        Replacing it every frame would churn a Tk image object at the
        capture rate on the thread the reactor's slot loop lives on, and
        ``ConvCost`` in jarvis/ui/avatar_clock.py exists because that
        conversion is the most expensive thing the frame loop does.
        """
        if image is None:
            self._clear_picture()
            return False
        try:
            from PIL import ImageTk                # noqa: PLC0415 - lazy
            size = (int(image.width), int(image.height))
            if self._photo is None or self._photo_size != size:
                self._photo = ImageTk.PhotoImage(image)
                self._photo_size = size
                self.canvas.itemconfigure(self._img, image=self._photo)
            else:
                self._photo.paste(image)
        except Exception:                          # noqa: BLE001 - PIL/Tk edge
            log.debug("preview: the frame would not convert", exc_info=True)
            self._clear_picture()
            return False
        self.canvas.itemconfigure(self._img, state="normal")
        return True

    def _draw_faces(self, shot: PreviewShot) -> None:
        ox, oy = self._origin
        faces = shot.faces[:MAX_FACES]
        primary = shot.primary
        attending = bool(primary is not None and primary.attending)
        for i, (shadow, item) in enumerate(zip(self._shadows, self._boxes)):
            if i >= len(faces):
                self.canvas.itemconfigure(shadow, state="hidden")
                self.canvas.itemconfigure(item, state="hidden")
                continue
            face = faces[i]
            x0, y0, x1, y1 = face_rect(face, shot.cap_w, shot.cap_h,
                                       self._fit)
            for it in (shadow, item):
                self.canvas.coords(it, ox + x0, oy + y0, ox + x1, oy + y1)
            self.canvas.itemconfigure(shadow, state="normal",
                                      outline=halo_ink())
            self.canvas.itemconfigure(
                item, state="normal",
                outline=box_ink(face.attending, face is primary))
        if primary is None:
            self._draw_name(None, None)
            for item in self._brackets:
                self.canvas.itemconfigure(item, state="hidden")
            return
        rect = face_rect(primary, shot.cap_w, shot.cap_h, self._fit)
        rect = (rect[0] + ox, rect[1] + oy, rect[2] + ox, rect[3] + oy)
        self._draw_name(primary, rect)
        if not attending:
            for item in self._brackets:
                self.canvas.itemconfigure(item, state="hidden")
            return
        pts = bracket_points(rect, px(BRACKET_ARM))
        for item, arm in zip(self._brackets, pts):
            self.canvas.coords(item, *arm)
            self.canvas.itemconfigure(item, state="normal", fill=theme.FOCAL)

    def _draw_name(self, face, rect) -> None:
        """The identity chip, beside the tracked box. Hidden when there is
        nothing honest to put in it.

        THE GROUND IS MEASURED, NOT ASSUMED. The chip's rectangle is sized
        from the canvas's own ``bbox`` of the text rather than from a guessed
        character width, because a ground a few pixels short of its word puts
        the last letter over live video -- and the whole reason for the chip
        is that a letter over live video has no guaranteed contrast. If the
        bbox cannot be had (no window yet), the TEXT is hidden too: no
        ground, no word. Same trade ``_show_image`` makes when a frame will
        not convert -- dark rather than half-drawn.

        AND IT SHEDS THE SCORE BEFORE IT COVERS THE FACE. The picture is
        272 px wide at his scale; a caption over 60% of that is a bar, not a
        label. The measurement is why this is a loop rather than a constant:
        the console's display face is condensed and fits both words, the
        fallback face does not, and this pane cannot guarantee which one Tk
        resolved. The common case still measures once.
        """
        if not identity_name(face) or rect is None:
            self.canvas.itemconfigure(self._namebg, state="hidden")
            self.canvas.itemconfigure(self._name, state="hidden")
            return
        ox, oy = self._origin
        fx, fy, fw, fh = self._fit
        picture = (ox + fx, oy + fy, fw, fh)
        pad = px(NAME_PAD)
        x, y, anchor = name_anchor(rect, picture, px(NAME_GAP), name_line_h())
        if anchor != getattr(self, "_name_anchor", None):
            # The canvas measures the text FROM its anchor, so a flip from
            # under the box to above it changes the bbox for the same word
            # at the same point. The placement cache is dropped with it.
            self._name_last = None
            self._name_anchor = anchor
        self.canvas.itemconfigure(self._name, anchor=anchor,
                                  fill=identity_ink(face), state="normal")
        room = fw - 2 * pad
        box = None
        # THE NAME ONLY. The score used to ride in front of the name here,
        # dropping out on a narrow pane -- which is why he saw "HUNTER 0.51"
        # sometimes and "HUNTER" other times over the same face. He asked
        # for it gone: the caption over his own face is an ANSWER, and the
        # evidence behind it belongs on a surface he opens to read numbers.
        # The score is still on the SENSORS page (ui/sensors_page.py), which
        # is exactly such a surface, so nothing was lost by removing it here.
        for text, bar in ((identity_name(face), room),):
            box = self._place_name(text, x, y)
            if box is None:
                self.canvas.itemconfigure(self._name, state="hidden")
                self.canvas.itemconfigure(self._namebg, state="hidden")
                return
            if (box[2] - box[0]) + 2 * pad <= bar:
                break
        else:
            # Even the bare word does not fit -- a long enrolled label at a
            # large UI scale. Trimmed rather than clamped off the edge: half
            # a name he can see beats a whole one he cannot.
            box = self._place_name(
                ellipsize(identity_name(face), name_font(), room), x, y)
            if box is None:
                self.canvas.itemconfigure(self._name, state="hidden")
                self.canvas.itemconfigure(self._namebg, state="hidden")
                return
        # Kept inside the picture. A chip anchored to a face at the right
        # edge would otherwise run into the readout column, which is where
        # the pane's OTHER numbers live -- two unrelated readouts touching.
        shift = min(0, (picture[0] + fw - pad) - box[2])
        shift = max(shift, (picture[0] + pad) - box[0])
        if shift or getattr(self, "_name_shift", 0):
            # Placed at x + shift whenever the item is not already there:
            # this frame needs a shift, or the last one had one and a cached
            # placement (which put nothing down) left the item where that
            # shift moved it.
            self.canvas.coords(self._name, x + shift, y)
        self._name_shift = shift
        if shift:
            box = (box[0] + shift, box[1], box[2] + shift, box[3])
        self.canvas.coords(self._namebg, box[0] - pad, box[1] - pad,
                           box[2] + pad, box[3] + pad)
        self.canvas.itemconfigure(self._namebg, state="normal",
                                  fill=chip_ink())

    def _place_name(self, text: str, x: int, y: int):
        """Put the word down and ask the canvas how wide it came out.
        ``None`` when it cannot be measured, which is a window that does not
        exist yet rather than a failure.

        CACHED ON THE LAST PLACEMENT. Between two detections the boxes are
        carried forward, so the same word lands at the same point picture
        after picture; re-setting the text and asking the canvas to measure
        it again was two of the ~30 canvas operations a repaint costs, at
        the picture rate, for an answer it already had. A hit puts nothing
        down -- the item still carries that text at that point, because
        nothing but this method sets either (``_draw_name`` tracks the one
        shift it applies afterwards). The anchor is part of what the canvas
        measures; ``_draw_name`` drops the cache when it changes.
        """
        last = getattr(self, "_name_last", None)
        if last is not None and last[0] == (text, x, y):
            return last[1]
        self.canvas.itemconfigure(self._name, text=text)
        self.canvas.coords(self._name, x, y)
        try:
            box = self.canvas.bbox(self._name)
        except Exception:                    # noqa: BLE001 - no window yet
            return None
        self._name_last = ((text, x, y), box) if box is not None else None
        return box

    # ------------------------------------------------------------ teardown
    def destroy(self):                       # pragma: no cover - Tk teardown
        self.stop()
        super().destroy()


def build_preview(parent, worker=None) -> Optional[CameraPreview]:
    """The pane, or None with the reason logged. A camera preview must not
    be able to stop the console from being built."""
    try:
        return CameraPreview(parent, worker=worker)
    except Exception:                        # noqa: BLE001 - a UI must survive
        log.exception("preview: the camera pane could not be built")
        return None


def worker_box() -> tuple:
    """The size the capture thread should scale to: the pane's picture box
    in DEVICE pixels, so the resize happens once, off the Tk thread, at
    exactly the size that will be shown."""
    return (px(PANE_W), px(PANE_H))
