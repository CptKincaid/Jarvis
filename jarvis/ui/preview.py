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

WHAT IT COSTS THE 60 fps LOOP, MEASURED. On a scratch Xvfb at his UI scale
(S=2.0), a real reactor animating on 16.67 ms slot boundaries, 30 s windows
after a 12 s warm-up, with a synthetic 1280x720 source carrying the LifeCam's
measured 130 ms grab latency (scratchpad harness; no camera was opened) --

  A  reactor alone                     0 late slots / 1801   max late 1.6-4.3 ms
  B  reactor + this pane at 6 fps    0-1 late slots / 1802   max late 1.4-3.8 ms
  C  the same capture ON the Tk loop 744-771 / ~1800 (41-43%)  max late 137 ms

C is the counterfactual this whole design exists to avoid, and it is not a
near miss: two slots in five late and a 137 ms stall is the console stopping
dead eight refreshes at a time, six times a second. B is indistinguishable
from A across five paired runs in both orders (a "late slot" is one starting
>8 ms after its boundary; 0 or 1 in 1800 either way), because the only
Tk-thread work per frame is one photo paste plus a dozen coords /
itemconfigure calls: **0.32-0.39 ms mean, 0.88 ms worst observed**, against a
16.67 ms slot. The 1.5 ms resize of the full frame happens on the capture
thread, not here.

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

import tkinter as tk
from typing import Optional

from jarvis.campreview import (MAX_FACES, REASON_DISABLED, REASON_LIVE,
                               REASON_NO_FRAME, REASON_PIPELINE,
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


def state_word(shot: PreviewShot) -> str:
    return STATE_WORDS.get(shot.reason, "OFF")


def state_tone(shot: PreviewShot) -> str:
    if shot.reason != REASON_LIVE:
        return TONE_OFF
    face = shot.primary
    return TONE_LIVE if (face is not None and face.attending) else TONE_AWAY


def attention_line(shot: PreviewShot) -> tuple:
    """``(text, tone)`` for the verdict row."""
    if shot.reason != REASON_LIVE:
        return shot.detail or "", TONE_OFF
    face = shot.primary
    if face is None:
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
        return (("FACES", "0"), ("YAW", ""))
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
    the performance design: at 6 fps a per-frame ``create_image`` /
    ``delete`` pair would hand Tk a new XImage sixty times a minute, on the
    same thread the reactor is trying to hit 16.67 ms slot boundaries on.
    Updating is ``coords`` + ``itemconfigure`` + one ``paste``.

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
        self._interval = poll_ms(6.0)
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

    # ----------------------------------------------------------- polling
    def start(self, ms: Optional[int] = None) -> None:
        """Begin polling the worker. Idempotent.

        The interval comes from the WORKER's configured rate, not from a
        constant: a pane polling at 83 ms for a capture running at 10 fps
        would hold every other frame back a full period.
        """
        if self._job is not None:
            return
        rate = getattr(self.worker, "fps", None)
        self._interval = int(ms or poll_ms(rate if rate else 6.0))
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
        self._paint(shot)

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
            for item in self._shadows + self._boxes + self._brackets:
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
        if primary is None or not attending:
            for item in self._brackets:
                self.canvas.itemconfigure(item, state="hidden")
            return
        rect = face_rect(primary, shot.cap_w, shot.cap_h, self._fit)
        pts = bracket_points((rect[0] + ox, rect[1] + oy,
                              rect[2] + ox, rect[3] + oy), px(BRACKET_ARM))
        for item, arm in zip(self._brackets, pts):
            self.canvas.coords(item, *arm)
            self.canvas.itemconfigure(item, state="normal", fill=theme.FOCAL)

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
