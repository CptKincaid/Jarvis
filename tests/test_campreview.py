"""The camera preview's capture side (jarvis/campreview.py).

WHAT THESE TESTS ARE ALLOWED TO DO, which is the point of the file. Hunter's
standing rule is that nothing may look at what his camera sees, and the
preview is the one feature in Jarvis whose whole job is to put a frame on a
screen. So the suite drives GEOMETRY, STATE and TIMING and nothing else:

* no test opens a device -- every frame here is a small array this process
  generated, and every feed is a stub that counts its own calls;
* no test asserts on pixel CONTENT. The only thing ever asserted about an
  image is its SIZE, because that is a layout fact; a test that had to look
  at a picture to pass would be a test that made looking normal;
* one test greps the two modules for the ways a frame could reach a disk,
  because a review of print statements does not survive the next edit and a
  grep the suite runs does.

The three behaviours that MUST hold, and each has a test that fails loudly:
the toggle off means no capture at all, a sensing denial means no capture and
a STATED reason, and the Tk thread is never the thread that waits 130 ms for
a frame.
"""
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from jarvis import campreview as cp

REPO = Path(__file__).resolve().parent.parent


def _code_only(path: Path) -> str:
    """The file with every comment and string literal removed.

    The grep below has to look at CODE. These two modules discuss `imwrite`
    and `save` at length in their docstrings -- explaining that they do not
    call them is most of why the rule survives -- so a naive grep would be a
    test that punishes documenting the rule.
    """
    import io
    import tokenize
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


# ------------------------------------------------------------ fake pieces
def frame(w=64, h=36, value=90):
    """A frame-shaped array this process made up. Not a picture of anything;
    the detector is a stub, so nothing here is ever looked at."""
    return np.full((h, w, 3), value, dtype=np.uint8)


class FakeFeed:
    """``jarvis.camera.CameraFeed``'s contract, and nothing else: a frame or
    None meaning no opinion, plus a close that can be counted."""

    def __init__(self, frames=None, delay=0.0, raises=False):
        self._frames = list(frames) if frames is not None else None
        self.delay = delay
        self.raises = raises
        self.captures = 0
        self.closes = 0
        self.lens = None

    def capture(self):
        self.captures += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise OSError("the camera went away")
        if self._frames is None:
            return frame()
        return self._frames.pop(0) if self._frames else None

    def close(self):
        self.closes += 1


class FakeDetector:
    """YuNet's seam: ``input_size`` plus ``detect`` returning (N, 15) rows.

    The rows are made up here, in DETECT pixels, so the scale-back-up
    arithmetic in the pipeline is exercised against numbers whose right
    answer is known by hand.
    """

    name = "fake"

    def __init__(self, rows=None, input_size=(32, 18)):
        self.input_size = input_size
        self.rows = rows
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        return self.rows


def row(x, y, w, h, conf=0.9, nose_dx=0.0):
    """One YuNet row: box, five landmarks, score. The eyes sit level and one
    interocular apart, so ``nose_dx`` (a fraction of that distance) is the
    only thing driving the yaw -- which is how the tests below can name an
    expected angle without re-implementing the head model."""
    r = [0.0] * 15
    r[0], r[1], r[2], r[3] = x, y, w, h
    eye = w * 0.4
    cx, cy = x + w / 2.0, y + h * 0.4
    r[4], r[5] = cx - eye / 2.0, cy          # person's right eye
    r[6], r[7] = cx + eye / 2.0, cy          # person's left eye
    r[8], r[9] = cx + nose_dx * eye, cy      # nose tip
    r[14] = conf
    return r


class FakePipeline:
    """A ``PreviewPipeline`` stand-in for the worker tests: it records what
    it was asked for and hands back whatever the test wants."""

    def __init__(self, shot=None, delay=0.0, feed=object()):
        self.feed = feed
        self.reason = ""
        self.grabs = 0
        self.closes = 0
        self.delay = delay
        self._shot = shot

    def grab(self, box, seq=0):
        self.grabs += 1
        if self.delay:
            time.sleep(self.delay)
        if self._shot is not None:
            return cp.PreviewShot(image=self._shot.image,
                                  faces=self._shot.faces,
                                  cap_w=self._shot.cap_w,
                                  cap_h=self._shot.cap_h,
                                  reason=self._shot.reason, seq=seq)
        return cp.PreviewShot(image=object(), cap_w=box[0], cap_h=box[1],
                              reason=cp.REASON_LIVE, seq=seq)

    def close(self):
        self.closes += 1


class State:
    """A ``SensingState``-shaped object. Duck-typed the way the header badge
    reads it (jarvis/ui/sensing_badge.normalise), which is the whole reason
    the pane and the badge cannot disagree about why the lens is shut."""

    def __init__(self, camera=True, radar=True, offline=False, reason="",
                 until=None, curfew=None, persisted=True):
        self.camera, self.radar, self.offline = camera, radar, offline
        self.reason, self.until = reason, until
        self.curfew, self.persisted = curfew, persisted


class Policy:
    def __init__(self, state=None, raises=False):
        self._state = state or State()
        self.raises = raises
        self.reads = 0

    def state(self):
        self.reads += 1
        if self.raises:
            raise RuntimeError("the policy is broken")
        return self._state


def options(**kw):
    """A ``services.get_option``-shaped reader over a dict."""
    def get(key, default=None):
        return kw.get(key, default)
    return get


class Clock:
    """A hand-wound monotonic clock.

    The pipeline now runs THREE cadences off one clock -- pictures, boxes,
    names -- so a test that wants two consecutive detections has to say so by
    advancing time. Driving it by hand is also the only way to assert a
    cadence at all: against the real clock, two grabs a microsecond apart are
    indistinguishable from a cadence that is broken.
    """

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> float:
        self.t += float(seconds)
        return self.t


# ------------------------------------------------------- the toggle is OFF
def test_the_toggle_defaults_off_and_off_means_no_thread_and_no_capture():
    pipe = FakePipeline()
    w = cp.PreviewWorker(get_option=options(), pipeline=pipe)
    assert cp.preview_enabled(options()) is False
    assert w.start() is False
    assert w.running is False
    assert pipe.grabs == 0
    assert w.latest().reason == cp.REASON_DISABLED
    assert w.latest().image is None


def test_switching_the_toggle_off_stops_the_thread_and_closes_the_device():
    pipe = FakePipeline()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy())
    assert w.start() is True
    for _ in range(200):
        if pipe.grabs:
            break
        time.sleep(0.005)
    assert pipe.grabs >= 1
    w.set_enabled(False)
    assert w.running is False
    assert pipe.closes == 1                    # the device was RELEASED
    grabbed = pipe.grabs
    time.sleep(0.05)
    assert pipe.grabs == grabbed               # and nothing grabs afterwards
    assert w.latest().reason == cp.REASON_DISABLED


# --------------------------------------------------- sensing says no
@pytest.mark.parametrize("state,expected", [
    (State(camera=False, reason="curfew", curfew=((21, 0), (7, 0))),
     "curfew until 7"),
    (State(camera=False, radar=False, offline=True, reason="offline"),
     "offline mode"),
    (State(camera=False, radar=False, offline=True, reason="failsafe",
           persisted=True), "sensing state unknown"),
])
def test_a_sensing_denial_means_no_capture_and_a_stated_reason(state, expected):
    pipe = FakePipeline()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy(state))
    shot = w.cycle()
    assert pipe.grabs == 0                     # the device was never asked
    assert pipe.closes == 1                    # and one already open is shut
    assert shot.reason == cp.REASON_SENSING
    assert shot.image is None
    assert expected in shot.detail
    # A black rectangle is not an answer: the pane has something to print.
    assert len(shot.detail) > 4


def test_no_sensing_owner_at_all_is_the_failsafe_not_a_live_picture():
    """A console that showed a picture because the policy failed to
    construct would be asserting the one thing nobody can check."""
    pipe = FakePipeline()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=None)
    shot = w.cycle()
    assert shot.reason == cp.REASON_SENSING
    assert "unknown" in shot.detail
    assert pipe.grabs == 0


def test_a_policy_that_raises_is_read_as_a_denial():
    pipe = FakePipeline()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy(raises=True))
    shot = w.cycle()
    assert shot.reason == cp.REASON_SENSING
    assert pipe.grabs == 0


def test_the_curfew_edge_closes_a_device_that_is_already_open():
    """The window he is looking at when 21:00 arrives. Nobody speaks, the
    clock moves, and the next cycle must both stop grabbing AND release."""
    pipe = FakePipeline()
    policy = Policy(State())
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=policy)
    assert w.cycle().live is True
    policy._state = State(camera=False, reason="curfew",
                          curfew=((21, 0), (7, 0)))
    shot = w.cycle()
    assert shot.reason == cp.REASON_SENSING
    assert pipe.closes == 1
    assert pipe.grabs == 1                     # no grab on the denied pass


def test_camera_allowed_reads_anything_it_cannot_parse_as_a_no():
    assert cp.camera_allowed(State()) is True
    assert cp.camera_allowed(State(camera=False)) is False
    assert cp.camera_allowed(State(offline=True)) is False
    assert cp.camera_allowed({"camera": True}) is True
    # sensing_badge.normalise defaults a missing `camera` to True, which is
    # right for a badge and would be a fail-ONLINE here.
    assert cp.camera_allowed(object()) is False
    assert cp.camera_allowed({}) is False
    assert cp.camera_allowed(None) is False


# ------------------------------------------------ nothing to capture WITH
def test_a_missing_vision_lane_names_itself_instead_of_showing_black():
    def make():
        return cp.PreviewPipeline(None, reason="the camera lane is not "
                                               "installed (no module)")
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         sensing=Policy(), make_pipeline=make)
    shot = w.cycle()
    assert shot.reason == cp.REASON_PIPELINE
    assert "not installed" in shot.detail
    assert shot.image is None


def test_a_feed_that_declines_reads_as_no_signal_not_as_an_empty_room():
    """The VSS lesson: a detector that fell back silently reported zero
    faces, and zero faces with everything fine is what an empty room looks
    like. 'I could not look' has to be spelled differently."""
    pipe = cp.PreviewPipeline(FakeFeed(frames=[]), detector=FakeDetector())
    shot = pipe.grab((32, 18))
    assert shot.reason == cp.REASON_NO_FRAME
    assert shot.faces == ()
    assert shot.live is False


def test_a_feed_that_raises_is_a_reason_not_a_crash():
    pipe = cp.PreviewPipeline(FakeFeed(raises=True), detector=FakeDetector())
    assert pipe.grab((32, 18)).reason == cp.REASON_NO_FRAME


# --------------------------------------------------------- the live path
# The pipeline's OWN contract -- the scale back up to capture pixels, the
# ordering, the MAX_FACES cap, the tracker -- is exercised here against a
# stub `observe`, so it is tested on a tree that has no vision lane at all.
# The geometry that stub stands in for belongs to jarvis/visionrig.py and is
# tested where it lives; the block at the bottom of this file joins the two
# whenever that lane is installed.
class Obs:
    """What ``visionrig.observe`` returns, reduced to the fields the preview
    reads. Duck-typed on purpose: the preview must not care whether a future
    observation grows a field."""

    def __init__(self, row, sx, sy, landmarks_ok=True):
        self.conf = float(row[14])
        self.x, self.y = float(row[0]) * sx, float(row[1]) * sy
        self.w, self.h = float(row[2]) * sx, float(row[3]) * sy
        self.yaw_deg = float(row[8]) - (self.x + self.w / 2.0)
        self.landmarks_ok = landmarks_ok


def observer(calls=None, landmarks_ok=True):
    def observe(row, _lens, sx, sy, _head):
        if calls is not None:
            calls.append((sx, sy))
        return Obs(row, sx, sy, landmarks_ok=landmarks_ok)
    return observe


class FakeTracker:
    def __init__(self, verdicts=None):
        self.seen = []
        self._verdicts = list(verdicts or [])
        self.dwell_s = 0.0

    def update(self, yaw):
        self.seen.append(yaw)
        return self._verdicts.pop(0) if self._verdicts else False


def test_a_live_frame_carries_boxes_in_capture_pixels_and_a_sized_image():
    """The only thing asserted about the image is its SIZE. That is a layout
    fact; its content is his room and is nobody's business."""
    det = FakeDetector(rows=[row(4, 2, 8, 8, conf=0.93)], input_size=(32, 18))
    scales = []
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det,
                              observe=observer(scales))
    shot = pipe.grab((16, 9))
    assert shot.live is True
    assert det.calls == 1
    assert shot.cap_w == 64 and shot.cap_h == 36
    assert (shot.image.width, shot.image.height) == (16, 9)
    # detect 32x18 -> capture 64x36 is exactly 2x in BOTH axes, and the two
    # scales are handed over separately because they can disagree.
    assert scales == [(2.0, 2.0)]
    face = shot.primary
    assert (face.x, face.y, face.w, face.h) == (8.0, 4.0, 16.0, 16.0)
    assert face.conf == pytest.approx(0.93)


def test_a_squashed_detect_size_is_carried_as_two_scales_not_averaged():
    """A 4:3 detect size against a 16:9 capture is a 1.33x horizontal squash
    of every face in the frame. Averaging the two scales would hide it."""
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 24))
    scales = []
    cp.PreviewPipeline(FakeFeed(), detector=det,
                       observe=observer(scales)).grab((16, 9))
    assert scales == [(2.0, 1.5)]


def test_the_biggest_face_is_the_subject_and_only_it_gets_the_verdict():
    """Size, not confidence: at a desk the nearest face is the one at the
    desk, and a confident 20 px face across the room is not the subject."""
    det = FakeDetector(rows=[row(0, 0, 4, 4, conf=0.99),
                             row(8, 4, 10, 10, conf=0.70)],
                       input_size=(32, 18))
    tracker = FakeTracker([True])
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, tracker=tracker,
                              observe=observer())
    shot = pipe.grab((16, 9))
    assert len(shot.faces) == 2
    assert shot.primary.conf == pytest.approx(0.70)     # the BIG one
    assert shot.faces[0] is shot.primary                # sorted by area
    assert shot.faces[0].attending is True
    assert shot.faces[1].attending is False             # never the others
    assert len(tracker.seen) == 1                       # one head, one tracker
    # …and this is NOT the rule visionrig.Rig uses (it takes the most
    # confident). What the two share is the GEOMETRY that produces the
    # numbers, not the choice of subject, so the divergence is stated where
    # a reader of either will meet it rather than left to be discovered by
    # comparing two files.
    assert "Rig" in (cp.PreviewShot.primary.__doc__ or "")


def test_more_faces_than_the_pane_can_draw_are_dropped_not_averaged():
    rows = [row(i, 0, 4 + i, 4 + i) for i in range(6)]
    pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(rows=rows),
                              observe=observer())
    assert len(pipe.grab((16, 9)).faces) == cp.MAX_FACES


def test_the_cap_keeps_the_face_at_the_desk_not_the_three_most_confident():
    """The cap is applied AFTER the size sort. YuNet hands its rows back
    score-descending, so slicing first would keep by CONFIDENCE: three 20 px
    faces across the room would survive and the 360 px one at the desk --
    the subject, by this class's own rule -- would be thrown away, taking
    the attention verdict onto the wrong head."""
    rows = [row(0, 0, 2, 2, conf=0.99), row(3, 0, 2, 2, conf=0.98),
            row(6, 0, 2, 2, conf=0.97), row(9, 0, 9, 9, conf=0.80)]
    det = FakeDetector(rows=rows, input_size=(32, 18))
    tracker = FakeTracker([True])
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, tracker=tracker,
                              observe=observer())
    shot = pipe.grab((16, 9))
    assert len(shot.faces) == cp.MAX_FACES
    assert shot.primary.conf == pytest.approx(0.80)   # the BIG one survived
    assert shot.faces[0] is shot.primary
    assert shot.faces[0].attending is True            # …and it is the subject
    # The reduction is bounded too, so a detector with a runaway top_k
    # cannot cost the capture thread a visible amount of arithmetic.
    assert cp.MAX_ROWS >= 16


def test_a_frame_with_no_face_tells_the_tracker_so_rather_than_going_quiet():
    """A gap has to END the dwell. A tracker that simply stopped being
    called would keep reporting the attention state he walked away from."""
    tracker = FakeTracker([True])
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    clock = Clock()
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, tracker=tracker,
                              observe=observer(), now=clock)
    assert pipe.grab((16, 9)).primary.attending is True
    det.rows = []
    clock.tick(1.0)                              # …and a detection later
    assert pipe.grab((16, 9)).faces == ()
    assert tracker.seen[-1] is None


def test_a_row_the_geometry_refuses_is_skipped_not_fatal():
    def explode(*_a):
        raise ValueError("degenerate landmarks")
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    shot = cp.PreviewPipeline(FakeFeed(), detector=det,
                              observe=explode).grab((16, 9))
    assert shot.live is True and shot.faces == ()


def test_no_detector_is_a_reason_beside_the_picture_never_zero_faces():
    pipe = cp.PreviewPipeline(FakeFeed(), detector=None,
                              reason="yunet.onnx is missing")
    shot = pipe.grab((16, 9))
    assert shot.live is True                    # there IS a picture
    assert shot.faces == ()
    assert "missing" in shot.detail             # …and it says why no boxes


# ------------------------------------------ the three rates are not one
def test_the_detector_does_not_run_on_every_picture():
    """The whole point of the rate split. At 15 fps a detection per picture
    is 15 x 2 ms a second for a box that moves at the speed of a head at a
    desk; at 8 Hz it is 16 ms and the box is at most 125 ms behind."""
    clock = Clock()
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock)
    for _ in range(15):                          # one second of pictures
        pipe.grab((16, 9))
        clock.tick(1.0 / 15.0)
    # 8 Hz against a 15 fps picture rate lands on every other frame.
    assert 7 <= det.calls <= 8, det.calls


def test_a_picture_rate_below_the_detect_rate_still_detects_every_frame():
    """The cadence is a FLOOR on the period, not a divider on the frames: at
    2 fps there is no sense in skipping detections, and the arithmetic must
    not quietly stop the boxes updating on a slow camera."""
    clock = Clock()
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock)
    for _ in range(4):
        pipe.grab((16, 9))
        clock.tick(0.5)
    assert det.calls == 4


def test_the_box_is_carried_between_detections_rather_than_blinking():
    """A box that vanished on the frames between detections would strobe at
    the difference of the two rates -- worse than the lag it replaced."""
    clock = Clock()
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock)
    first = pipe.grab((16, 9))
    clock.tick(1.0 / 15.0)                       # the next picture, no detect
    second = pipe.grab((16, 9))
    assert det.calls == 1
    assert second.live is True
    assert second.faces == first.faces           # the same boxes, carried


def test_a_detection_that_finds_nobody_empties_the_pane_at_once():
    """Carrying forward is for the frames BETWEEN detections. A detection
    that came back empty is an answer, and holding the last box over it
    would draw a face on an empty chair."""
    clock = Clock()
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock)
    assert pipe.grab((16, 9)).faces
    det.rows = []
    clock.tick(0.2)
    assert pipe.grab((16, 9)).faces == ()
    clock.tick(1.0 / 15.0)
    assert pipe.grab((16, 9)).faces == ()        # and it stays empty


def test_a_camera_that_changes_mode_drops_the_boxes_it_cannot_place():
    """A carried face is in the CAPTURE pixels of the frame it came from, so
    a mode change makes it WRONG rather than stale -- boxes drawn on the
    wrong part of a differently sized picture."""
    clock = Clock()
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    feed = FakeFeed(frames=[frame(64, 36), frame(32, 18), frame(32, 18)])
    pipe = cp.PreviewPipeline(feed, detector=det, observe=observer(),
                              now=clock)
    assert pipe.grab((16, 9)).cap_w == 64
    clock.tick(1.0 / 15.0)                       # too soon for a detection…
    shot = pipe.grab((16, 9))                    # …but the mode changed
    assert shot.cap_w == 32
    assert det.calls == 2                        # so it detected anyway
    assert shot.faces                            # in the NEW geometry


def test_a_detector_that_raises_clears_the_boxes_rather_than_freezing_them():
    """A frozen overlay over a live picture is the pane asserting a
    detection it did not make."""
    clock = Clock()

    class Angry(FakeDetector):
        def detect(self, _frame):
            self.calls += 1
            raise RuntimeError("the detector fell over")

    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock)
    assert pipe.grab((16, 9)).faces
    pipe.detector = Angry(input_size=(32, 18))
    clock.tick(1.0)
    assert pipe.grab((16, 9)).faces == ()


# ------------------------------------------------------------ who it is
class FakeRecogniser:
    """``facedetect.SFaceRecogniser``'s seam: ``min_conf`` plus ``embed``.

    It records the rows it was asked about, which is how the tests below can
    assert the ONE rule that cannot bend -- that no embedding is ever taken
    from a detection under the detector's own bar.
    """

    def __init__(self, vec=(1.0, 2.0, 3.0), min_conf=0.6, raises=False):
        self.min_conf = min_conf
        self.vec = vec
        self.raises = raises
        self.rows = []

    def embed(self, _frame, row):
        self.rows.append(list(row))
        if self.raises:
            raise ValueError("that is not a face")
        return self.vec


class FakeGallery:
    """``facegallery.FaceGallery.match``'s seam, and nothing else. The pane
    never saves, so there is nothing else to stand in for."""

    def __init__(self, answers=None, raises=False):
        self._answers = list(answers or [])
        self.raises = raises
        self.calls = 0

    def match(self, _vec):
        self.calls += 1
        if self.raises:
            raise RuntimeError("the gallery is unreadable")
        if not self._answers:
            return "hunter", 0.74
        return self._answers.pop(0) if len(self._answers) > 1 \
            else self._answers[0]


def identified(clock, answers=None, recogniser=None, rows=None,
               **kw):
    """A pipeline with identity wired, on a hand-wound clock."""
    det = FakeDetector(rows=rows if rows is not None else [row(4, 2, 8, 8)],
                       input_size=(32, 18))
    pipe = cp.PreviewPipeline(
        FakeFeed(), detector=det, observe=observer(), now=clock,
        recogniser=recogniser if recogniser is not None else FakeRecogniser(),
        gallery=FakeGallery(answers), identity_min=0.363, min_conf=0.6, **kw)
    return pipe, det


def test_the_tracked_face_carries_the_name_of_whoever_it_is():
    """His words, 2026-09-03: "lets have the identity of the person its
    tracking next to their name, small but readable"."""
    clock = Clock()
    pipe, _ = identified(clock, [("hunter", 0.74)])
    clock.tick(1.0)
    face = pipe.grab((16, 9)).primary
    clock.tick(1.0)                              # one agreeing reading more
    face = pipe.grab((16, 9)).primary
    assert face.name == "hunter"
    assert face.id_score == pytest.approx(0.74)
    assert face.id_ran is True


def test_identity_is_never_computed_from_a_detection_under_the_bar():
    """THE rule. SFace answers confidently on things that are not faces --
    unrelated non-face crops match each other at 0.66-0.92 against a 0.363
    "same person" bar -- so an embedding from a weak detection is not a weak
    answer, it is a confident wrong one. The embedding is not a second
    opinion on whether this is a face."""
    clock = Clock()
    rec = FakeRecogniser(min_conf=0.6)
    pipe, _ = identified(clock, rows=[row(4, 2, 8, 8, conf=0.4)],
                         recogniser=rec)
    for _ in range(6):
        clock.tick(1.0)
        shot = pipe.grab((16, 9))
    assert rec.rows == []                        # never asked
    # …and it is not a face AT ALL. camera.min_conf was read, stored and
    # never compared against anything until 2026-09-03, so the detector's
    # own 0.3 floor was the only bar: a door frame drew a box at 0.48
    # against his configured 0.6 and got named. His bar is applied now.
    assert shot.primary is None
    assert shot.faces == ()


def test_identity_runs_at_its_own_cadence_not_at_the_detection_rate():
    """10.4 ms an embedding, measured 2026-09-03 -- five to seven times a
    detection. At the picture rate it would cost more than everything else
    in this module put together."""
    clock = Clock()
    rec = FakeRecogniser()
    pipe, det = identified(clock, recogniser=rec)
    for _ in range(15):                          # one second of pictures
        pipe.grab((16, 9))
        clock.tick(1.0 / 15.0)
    assert 7 <= det.calls <= 8                   # boxes at 8 Hz…
    assert 2 <= len(rec.rows) <= 3               # …names at 2 Hz


def test_the_embedding_is_taken_from_the_subjects_own_row():
    """The row, not the rectangle: SFace aligns its crop from the five
    landmarks. Handing it the wrong row would embed the face across the
    room and put that name on the head at the desk."""
    clock = Clock()
    rec = FakeRecogniser()
    rows = [row(0, 0, 2, 2, conf=0.99), row(9, 4, 9, 9, conf=0.80)]
    pipe, _ = identified(clock, rows=rows, recogniser=rec)
    clock.tick(1.0)
    shot = pipe.grab((16, 9))
    assert shot.primary.conf == pytest.approx(0.80)      # the BIG one
    assert rec.rows[0][14] == pytest.approx(0.80)        # and its row


def test_a_face_that_matches_nothing_reads_as_unknown_not_as_blank():
    """Say it honestly. A face under the bar is an ANSWER, and it must not
    read as "identity is not running" and must not read as the last name."""
    clock = Clock()
    pipe, _ = identified(clock, [("hunter", 0.21)])      # under 0.363
    clock.tick(1.0)
    face = pipe.grab((16, 9)).primary
    assert face.id_ran is True                   # it was asked…
    assert face.name == ""                       # …and the answer was no
    assert face.id_score == pytest.approx(0.21)  # with the score to show


def test_a_single_odd_reading_cannot_rename_the_person_on_screen():
    """A chip that alternates hunter / unknown a hundred milliseconds apart
    is unreadable, and it says two contradictory things about the person in
    the chair. Changing a held verdict costs evidence."""
    clock = Clock()
    pipe, _ = identified(clock, [("hunter", 0.74), ("hunter", 0.74),
                                 ("hunter", 0.10), ("hunter", 0.74)])
    names = []
    for _ in range(4):
        clock.tick(1.0)
        names.append(pipe.grab((16, 9)).primary.name)
    assert names == ["", "hunter", "hunter", "hunter"]


def test_two_agreeing_readings_are_enough_to_take_a_name_off_again():
    """Slow to put a name on is cosmetic. Slow to take one OFF is the pane
    telling him something false about who Jarvis thinks is there, so once
    something is held both directions cost the same evidence."""
    clock = Clock()
    pipe, _ = identified(clock, [("hunter", 0.74), ("hunter", 0.74),
                                 ("", 0.10), ("", 0.10)])
    seen = []
    for _ in range(4):
        clock.tick(1.0)
        face = pipe.grab((16, 9)).primary
        seen.append((face.name, face.id_ran))
    assert seen == [("", False),                 # a name costs two readings
                    ("hunter", True), ("hunter", True),
                    ("", True)]                  # …and so does losing one


def test_the_name_goes_when_the_face_does_rather_than_lingering():
    """An empty frame is precisely the window in which a different person
    sits down -- the same reason Eye's SessionIdentity.room_empty exists."""
    clock = Clock()
    pipe, det = identified(clock, [("hunter", 0.74)])
    for _ in range(2):
        clock.tick(1.0)
        pipe.grab((16, 9))
    assert pipe.grab((16, 9)).primary.name == "hunter"
    det.rows = []
    clock.tick(1.0)
    assert pipe.grab((16, 9)).faces == ()
    det.rows = [row(4, 2, 8, 8)]
    clock.tick(1.0)
    assert pipe.grab((16, 9)).primary.name == ""     # a fresh question


def test_a_held_name_expires_rather_than_sitting_over_an_unchecked_face():
    """A subject that drops under HIS bar stops being a face, and the name
    goes with it.

    This used to keep the box and expire only the name, on the argument
    that a weak detection is still a detection. Seeing it live on
    2026-09-03 he called it "capturing random objects": the pane drew boxes
    on his wall at 0.48 against a configured 0.6. The bar the class always
    said it applied is applied."""
    clock = Clock()
    rec = FakeRecogniser(min_conf=0.6)
    det = FakeDetector(rows=[row(4, 2, 8, 8, conf=0.9)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock, recogniser=rec,
                              gallery=FakeGallery([("hunter", 0.74)]),
                              identity_min=0.363, min_conf=0.6)
    for _ in range(2):
        clock.tick(1.0)
        pipe.grab((16, 9))
    assert pipe.grab((16, 9)).primary.name == "hunter"
    det.rows = [row(4, 2, 8, 8, conf=0.4)]           # too weak to embed
    clock.tick(cp.IDENT_HOLD_S + 0.1)
    assert pipe.grab((16, 9)).primary is None        # gone, name and all


def test_an_embedding_that_fails_leaves_the_last_verdict_alone():
    """A crop that would not align is a failure of the machinery, not an
    answer. Overwriting a held verdict with a verdict nobody reached would
    be the pane inventing an "unknown"."""
    clock = Clock()
    pipe, _ = identified(clock, [("hunter", 0.74)])
    for _ in range(2):
        clock.tick(1.0)
        pipe.grab((16, 9))
    assert pipe.grab((16, 9)).primary.name == "hunter"
    pipe.recogniser = FakeRecogniser(raises=True)
    clock.tick(0.6)
    assert pipe.grab((16, 9)).primary.name == "hunter"


def test_a_gallery_that_raises_is_a_missing_answer_not_a_crash():
    clock = Clock()
    pipe, _ = identified(clock)
    pipe.gallery = FakeGallery(raises=True)
    clock.tick(1.0)
    face = pipe.grab((16, 9)).primary
    assert face.id_ran is False and face.name == ""


@pytest.mark.parametrize("vec", [(0.0, 0.0, 0.0), (float("nan"), 1.0, 2.0),
                                 ()])
def test_a_degenerate_embedding_is_a_missing_answer_not_an_unknown(vec):
    """A crop of nothing embeds to NaNs; a blank one to zeros. The gallery
    answers ("", 0.0) for both -- the same shape as "asked, nobody matched"
    -- and the pane would print UNKNOWN 0.00 over a face nobody was able to
    ask about. That is a machinery failure and is treated like one: the
    held verdict is left alone, and from cold there is no chip at all."""
    clock = Clock()
    pipe, _ = identified(clock, [("hunter", 0.74)])
    for _ in range(2):
        clock.tick(1.0)
        pipe.grab((16, 9))
    assert pipe.grab((16, 9)).primary.name == "hunter"
    pipe.recogniser = FakeRecogniser(vec=vec)
    clock.tick(0.6)
    assert pipe.grab((16, 9)).primary.name == "hunter"   # left alone
    gallery = FakeGallery([("", 0.0)])
    cold = cp.PreviewPipeline(FakeFeed(), observe=observer(), now=Clock(1.0),
                              detector=FakeDetector(rows=[row(4, 2, 8, 8)],
                                                    input_size=(32, 18)),
                              recogniser=FakeRecogniser(vec=vec),
                              gallery=gallery)
    face = cold.grab((16, 9)).primary
    assert face.id_ran is False                        # no chip, not UNKNOWN
    assert gallery.calls == 0                          # never even asked


def test_the_identity_gate_has_its_own_floor_under_the_detector_dial():
    """camera.min_conf decides how many boxes get DRAWN, and its documented
    direction of travel is down (0.7 -> 0.6, "because a miss is silent").
    Borrowing it as the identity gate let a dial about what the pane shows
    decide who gets NAMED: measured, min_conf 0.0 let a 0.25 detection
    produce a chip reading HUNTER 0.71 over a smudge. The gate keeps its own
    floor, and a HIGHER detector bar still raises it."""
    clock = Clock()
    rec = FakeRecogniser(min_conf=0.0)
    det = FakeDetector(rows=[row(4, 2, 8, 8, conf=0.25)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock, recogniser=rec, gallery=FakeGallery(),
                              identity_min=0.363, min_conf=0.0)
    assert pipe.id_conf == cp.DEFAULT_MIN_CONF
    for _ in range(4):
        clock.tick(1.0)
        face = pipe.grab((16, 9)).primary
    assert face is not None and face.conf == pytest.approx(0.25)  # drawn…
    assert rec.rows == [] and face.id_ran is False              # …not named
    high = cp.PreviewPipeline(FakeFeed(), observe=observer(), now=clock,
                              detector=FakeDetector(
                                  rows=[row(4, 2, 8, 8, conf=0.7)],
                                  input_size=(32, 18)),
                              recogniser=rec, gallery=FakeGallery(),
                              identity_min=0.363, min_conf=0.8)
    assert high.id_conf == 0.8
    clock.tick(1.0)
    high.grab((16, 9))
    assert rec.rows == []                        # 0.7 is under HIS 0.8 bar


# ------------------------------------------- the name belongs to a face
def _two_faces(swap_every: int = 1, fps: float = 15.0, seconds: float = 10.0):
    """Two people at similar distance, synthetic. Face A (hunter) at x=2 and
    face B (a stranger) at x=18 in detect pixels, sizes 8/9 swapping so that
    "largest" -- the pane's subject rule -- flips between them. The gallery
    answers by WHICH ROW was embedded, so a name over the other face can
    only come from the hold lending it. Returns (frames, named, wrong)."""
    class Det(FakeDetector):
        def __init__(self):
            super().__init__(input_size=(32, 18))
            self.i = 0

        def detect(self, _frame):
            self.calls += 1
            big_a = (self.i // swap_every) % 2 == 0
            self.i += 1
            wa, wb = (9, 8) if big_a else (8, 9)
            return [row(2, 2, wa, wa), row(18, 2, wb, wb)]

    class Rec(FakeRecogniser):
        last = None

        def embed(self, _frame, r):
            self.last = "A" if r[0] == 2 else "B"
            return super().embed(_frame, r)

    class Gal:
        def match(self, _vec):
            return ("hunter", 0.74) if rec.last == "A" else ("", 0.20)

    clock, det, rec = Clock(), Det(), Rec()
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock, recogniser=rec, gallery=Gal(),
                              identity_min=0.363, min_conf=0.6)
    frames = int(round(fps * seconds))
    named = wrong = 0
    for _ in range(frames):
        p = pipe.grab((16, 9)).primary
        if p.id_ran and p.name == "hunter":
            named += 1
            if p.x >= 36.0:                      # B's box, in capture px
                wrong += 1
        clock.tick(1.0 / fps)
    return frames, named, wrong


@pytest.mark.parametrize("swap_every", [1, 3])
def test_a_name_never_walks_onto_a_face_it_was_not_computed_from(swap_every):
    """THE two-person case. Measured on the unbound hold (dba972a), same
    harness: 142 of 150 frames carried a chip and 70 of them put hunter's
    name and score on the stranger's box, never converging -- each reading
    about the other face reset the disagreement counter and refreshed the
    timestamp, so neither hysteresis rule could fire. Bound to the box it
    was computed from: 0 wrong, and the name is still there for the face
    it belongs to (72 named frames flipping every picture, 36 every third).
    """
    frames, named, wrong = _two_faces(swap_every)
    assert frames == 150
    assert wrong == 0
    assert named >= 30                           # the feature still works


def test_a_failed_detection_clears_the_name_so_the_next_face_cannot_inherit_it():
    """A detection that COULD NOT be made is not weaker evidence of "nobody
    is there" than one that came back empty. The boxes were already dropped
    on this path; the NAME was not, and a different person sitting down in
    the same spot 0.2 s later inherited it (measured: the stranger's chip
    read name='hunter' id_ran=True)."""
    clock = Clock()
    pipe, det = identified(clock, [("hunter", 0.74)])
    for _ in range(2):
        clock.tick(1.0)
        pipe.grab((16, 9))
    assert pipe.grab((16, 9)).primary.name == "hunter"

    class Angry(FakeDetector):
        def detect(self, _frame):
            self.calls += 1
            raise RuntimeError("the detector fell over")

    pipe.detector = Angry(input_size=(32, 18))
    clock.tick(0.2)
    assert pipe.grab((16, 9)).faces == ()
    pipe.detector = det                          # a face at the SAME place
    clock.tick(0.6)
    face = pipe.grab((16, 9)).primary
    assert face.id_ran is False and face.name == ""   # a fresh question


def test_a_row_set_the_geometry_cannot_reduce_clears_the_name_too():
    """The other early return in _faces: no visionrig to reduce the rows
    with. Same rule, same reason."""
    clock = Clock()
    pipe, det = identified(clock, [("hunter", 0.74)])
    for _ in range(2):
        clock.tick(1.0)
        pipe.grab((16, 9))
    assert pipe.grab((16, 9)).primary.name == "hunter"
    keep = pipe._observe                          # noqa: SLF001
    pipe._observe = None                          # noqa: SLF001
    pipe._observer = lambda: None                 # noqa: SLF001
    clock.tick(0.2)
    assert pipe.grab((16, 9)).faces == ()
    del pipe._observer
    pipe._observe = keep                          # noqa: SLF001
    clock.tick(0.6)
    face = pipe.grab((16, 9)).primary
    assert face.id_ran is False and face.name == ""


def test_a_capture_mode_change_drops_the_name_with_the_boxes():
    """A held verdict is expressed in the capture pixels its subject's box
    was; a camera that renegotiates its mode while somebody new sits down
    is exactly the window in which the last name is wrong."""
    clock = Clock()
    feed = FakeFeed(frames=[frame(64, 36)] * 3 + [frame(32, 18)] * 2)
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(feed, detector=det, observe=observer(),
                              now=clock, recogniser=FakeRecogniser(),
                              gallery=FakeGallery([("hunter", 0.74)]),
                              identity_min=0.363, min_conf=0.6)
    for _ in range(2):
        clock.tick(1.0)
        pipe.grab((16, 9))
    assert pipe.grab((16, 9)).primary.name == "hunter"
    clock.tick(0.1)
    shot = pipe.grab((16, 9))                    # the mode changed
    assert shot.cap_w == 32 and shot.faces
    assert shot.primary.id_ran is False          # the name did not carry


def test_identity_off_means_no_name_and_no_embedding_at_all():
    """With camera.identity off, recogniser_from_config returns None and
    nothing about his face is computed, let alone shown."""
    clock = Clock()
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock)
    clock.tick(1.0)
    face = pipe.grab((16, 9)).primary
    assert face.id_ran is False and face.name == ""


def test_a_threshold_typed_wrong_cannot_blank_the_pane():
    """build_pipeline promises never to raise. A camera setting typed as a
    word rather than a number must fall back, not take the preview out --
    the same rule every other read at this boundary follows."""
    assert cp._float_option(options(**{cp.OPTION_MIN_CONF: "high"}),
                            cp.OPTION_MIN_CONF, 0.6) == 0.6
    assert cp._float_option(options(**{cp.OPTION_IDENTITY_MIN: None}),
                            cp.OPTION_IDENTITY_MIN, 0.363) == 0.363
    assert cp._float_option(options(**{cp.OPTION_MIN_CONF: "0.8"}),
                            cp.OPTION_MIN_CONF, 0.6) == 0.8


def test_nobody_enrolled_is_no_chip_rather_than_unknown_forever():
    """An empty gallery is the state BEFORE he enrols, not an error, and
    "UNKNOWN" on every face forever would read as a broken recogniser."""
    assert cp.resolve_identity(_CfgOff()) == (None, None)


class _CfgOff:
    def get(self, key, default=None):
        return False if key == "camera.identity" else default


# ------------------------------------------------- the hold, on its own
def test_the_first_honest_unknown_is_shown_at_once():
    clock = Clock()
    hold = cp.IdentityHold(now=clock)
    assert hold.observe("", 0.1) == ("", 0.1, True)


def test_a_name_needs_agreement_before_it_takes_the_screen():
    clock = Clock()
    hold = cp.IdentityHold(agree=2, now=clock)
    assert hold.observe("hunter", 0.74)[2] is False
    assert hold.observe("hunter", 0.75) == ("hunter", 0.75, True)


def test_two_different_candidates_in_a_row_agree_about_nothing():
    """A run of disagreeing readings is noise, not evidence, and must not
    accumulate into a verdict."""
    clock = Clock()
    hold = cp.IdentityHold(agree=2, now=clock)
    hold.observe("hunter", 0.7)
    hold.observe("guest", 0.7)
    assert hold.held()[2] is False


def test_a_verdict_nobody_confirms_expires():
    clock = Clock()
    hold = cp.IdentityHold(agree=1, hold_s=1.5, now=clock)
    hold.observe("hunter", 0.74)
    clock.tick(1.4)
    assert hold.held() == ("hunter", 0.74, True)
    clock.tick(0.2)
    assert hold.held() == ("", 0.0, False)


def test_clearing_forgets_everything_including_the_candidate():
    clock = Clock()
    hold = cp.IdentityHold(agree=2, now=clock)
    hold.observe("hunter", 0.74)
    hold.clear()
    hold.observe("hunter", 0.74)
    assert hold.held()[2] is False               # the run started again


A_BOX, B_BOX = (10.0, 10.0, 100.0, 100.0), (300.0, 10.0, 100.0, 100.0)


def test_a_verdict_is_withheld_from_another_face_and_kept_for_its_own():
    """The verdict BELONGS to the face it was computed from. Another face
    asking for it gets nothing -- not a wrong name -- and the hold is not
    thrown away for the asking: the face it belongs to is usually largest
    again on the next picture."""
    clock = Clock()
    hold = cp.IdentityHold(agree=2, now=clock)
    hold.observe("hunter", 0.74, A_BOX)
    assert hold.observe("hunter", 0.75, A_BOX) == ("hunter", 0.75, True)
    assert hold.held(B_BOX) == ("", 0.0, False)      # withheld
    assert hold.held(A_BOX) == ("hunter", 0.75, True)   # …and still his
    assert hold.held() == ("hunter", 0.75, True)     # no geometry: as before


def test_a_reading_about_a_different_face_replaces_the_subject():
    """A READING moves the subject; a stamp request does not. The hold is
    "who the identity tick last looked at", and it just looked elsewhere."""
    clock = Clock()
    hold = cp.IdentityHold(agree=2, now=clock)
    hold.observe("hunter", 0.74, A_BOX)
    hold.observe("hunter", 0.74, A_BOX)
    assert hold.observe("", 0.2, B_BOX) == ("", 0.2, True)   # unknown, at once
    assert hold.held(A_BOX) == ("", 0.0, False)              # hunter's is gone
    assert hold.held(B_BOX) == ("", 0.2, True)


def test_two_readings_about_two_faces_are_not_two_agreeing_readings():
    clock = Clock()
    hold = cp.IdentityHold(agree=2, now=clock)
    hold.observe("hunter", 0.74, A_BOX)
    assert hold.observe("hunter", 0.74, B_BOX)[2] is False   # a new run
    assert hold.observe("hunter", 0.74, B_BOX)[2] is True    # two about B
    assert hold.held(A_BOX) == ("", 0.0, False)


def test_a_head_that_drifts_at_a_desk_is_still_the_same_face():
    """SUBJECT_IOU at 0.3: a lean of a tenth of the box is well inside it, a
    face beside it scores 0, and a stranger stepping in front at twice the
    linear size scores 0.25 even perfectly concentric."""
    clock = Clock()
    hold = cp.IdentityHold(agree=1, now=clock)
    box = (100.0, 100.0, 200.0, 200.0)
    hold.observe("hunter", 0.74, box)
    assert hold.held((120.0, 110.0, 200.0, 200.0))[0] == "hunter"   # a lean
    assert hold.held((350.0, 100.0, 200.0, 200.0))[2] is False      # beside
    assert hold.held((0.0, 0.0, 400.0, 400.0))[2] is False          # in front
    assert hold.held((0.0, 0.0, 0.0, 0.0))[0] == "hunter"   # no geometry: yes


# ------------------------------------------------------ the Tk thread
def test_the_capture_thread_is_the_one_that_waits_for_the_frame():
    """THE reason this feature has a thread at all. A grab costs 130 ms on
    his LifeCam (measured 2026-09-02) and the console animates on 16.67 ms
    slot boundaries, so a read on the Tk thread would stall eight
    consecutive frames every time. Here a reader is never made to wait: the
    slot is a lock-protected attribute and the grab is somewhere else.
    """
    pipe = FakePipeline(delay=0.20)
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy())
    w.start()
    try:
        worst = 0.0
        deadline = time.monotonic() + 0.45
        reads = 0
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            w.latest()
            worst = max(worst, time.monotonic() - t0)
            reads += 1
        assert reads > 100                     # the reader really did spin
        assert pipe.grabs >= 1                 # …across at least one grab
        # One 60 Hz slot is 16.67 ms; a read must not be a fraction of it.
        assert worst < 0.005, "latest() blocked for %.1f ms" % (worst * 1000)
    finally:
        w.stop()


def test_the_worker_hands_over_the_latest_shot_and_drops_the_ones_between():
    """No queue: a preview frame two frames old has no value, and a queue
    that filled while the UI was busy would hand back a backlog to paint."""
    pipe = FakePipeline()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy())
    seqs = [w.cycle().seq for _ in range(4)]
    assert seqs == [1, 2, 3, 4]
    assert w.latest().seq == 4                 # only the newest survives


def test_stop_releases_the_device_even_when_the_grab_is_still_in_flight():
    """A privacy control that can be postponed indefinitely by a wedged
    camera is not one -- the same ordering CameraFeed._close_raw settled."""
    pipe = FakePipeline(delay=1.5)
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy())
    w.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    w.stop(timeout=0.2)
    assert time.monotonic() - t0 < 1.0
    assert pipe.closes >= 1


def test_the_ui_thread_is_not_made_to_wait_for_a_wedged_camera_to_let_go():
    """stop() runs on the TK THREAD -- from _preview_apply, at every
    ACTIVE->AMBIENT edge, i.e. every 45 s of quiet. A 130 ms grab there is
    eight consecutive 60 Hz slots and the console visibly stops dead, which
    is the symptom this module's threading exists to avoid. So the console
    passes join=False: the deny still happens before stop() returns, only
    the device handback moves off the caller."""
    pipe = FakePipeline(delay=1.5)
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy())
    w.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    w.stop(join=False)
    blocked = time.monotonic() - t0
    # One 60 Hz slot is 16.67 ms. The call must not be a fraction of it.
    assert blocked < 0.005, "stop() blocked the caller %.1f ms" % (
        blocked * 1000)
    assert w.running is False                  # …and the deny is immediate
    assert w.latest().reason == cp.REASON_DISABLED
    assert w.latest().image is None
    for _ in range(300):                       # the device still comes back
        if pipe.closes:
            break
        time.sleep(0.01)
    assert pipe.closes >= 1


def test_a_frame_that_arrives_after_the_switch_is_dropped_not_stored():
    """"stop() means no frame is held" has to survive a grab that outlasts
    the bounded join, not only a 130 ms one. The capture thread finishes its
    in-flight read after the switch went off; that frame must not land in
    the slot."""
    started, release = threading.Event(), threading.Event()

    class WedgedPipeline(FakePipeline):
        def grab(self, box, seq=0):
            started.set()
            release.wait(3.0)
            return cp.PreviewShot(image=object(), cap_w=box[0], cap_h=box[1],
                                  reason=cp.REASON_LIVE, seq=seq)

    pipe = WedgedPipeline()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy())
    w.start()
    assert started.wait(2.0)
    w.stop(timeout=0.2)                        # times out on the wedged grab
    assert w.latest().image is None
    release.set()                              # now the grab comes back
    time.sleep(0.15)
    assert w.latest().reason == cp.REASON_DISABLED
    assert w.latest().image is None, "a frame was held after stop()"


def campreview_threads() -> int:
    return sum(1 for t in threading.enumerate()
               if t.is_alive() and t.name == "campreview")


def test_a_stop_that_timed_out_cannot_be_revived_by_the_next_start():
    """The stop event is PER GENERATION and is never cleared.

    A single shared flag would be un-set by the next start(), reviving the
    thread still inside the wedged read -- so two of them would drive one
    device for the life of the process, racing in jarvis/eye.py's lockless
    capture() (where one can reach release() while the other is inside
    read()) and in this object's own non-atomic counters. Each wedge and
    restart would add another permanent thread.

    The successor is held on the capture lease rather than refused, so the
    pane comes back by itself the moment the wedged read returns -- and it
    waits there, not on the Tk thread.
    """
    inside, seen = [], []
    gate = threading.Lock()
    release = threading.Event()

    class SharedFeedPipeline(FakePipeline):
        """ONE object across both generations, the way a real
        services.camera_feed is: _close_pipeline drops the reference but
        the device behind it is the same device."""

        def grab(self, box, seq=0):
            with gate:
                inside.append(threading.current_thread().ident)
                seen.append(len(inside))
            try:
                release.wait(2.0)
                return cp.PreviewShot(image=object(), cap_w=box[0],
                                      cap_h=box[1], reason=cp.REASON_LIVE,
                                      seq=seq)
            finally:
                with gate:
                    inside.pop()

    pipe = SharedFeedPipeline()
    before = campreview_threads()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         make_pipeline=lambda: pipe, sensing=Policy())
    try:
        w.start()
        for _ in range(400):
            if seen:
                break
            time.sleep(0.005)
        assert seen, "the first capture never reached the device"
        w.stop(timeout=0.05)                   # the join times out
        first = w._thread                      # noqa: SLF001 - the point
        assert first is None                   # running says False…
        w.start()                              # …and he switches it back on
        second = w._thread                     # noqa: SLF001
        assert second is not None
        time.sleep(0.15)                       # plenty of time to double up
        assert max(seen) == 1, \
            "two capture threads were inside the device at once"
        release.set()                          # the wedged read comes back
        for _ in range(400):                   # …and the successor takes over
            if len(seen) > 1:
                break
            time.sleep(0.005)
        assert len(seen) > 1, "the pane never came back after the wedge"
        assert max(seen) == 1
    finally:
        release.set()
        w.stop(timeout=2.0)
    for _ in range(400):                       # no thread is left behind
        if campreview_threads() <= before:
            break
        time.sleep(0.005)
    assert campreview_threads() <= before


def test_the_last_pass_of_a_stopped_run_cannot_overwrite_the_live_one():
    """A superseded generation finishing its last pass must not write its
    stale verdict over the picture the new one is publishing."""
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=FakePipeline(), sensing=Policy())
    old = w._stop                              # noqa: SLF001 - the point
    w.start()                                  # mints a NEW generation
    live = w.cycle()
    assert live.live is True
    w._publish(cp.blank(cp.REASON_DISABLED), old)   # noqa: SLF001
    assert w.latest() is live                  # the old run was ignored
    w.stop()


def test_the_value_he_just_clicked_beats_a_config_that_has_not_landed():
    """SettingsDrawer._set_option writes assistant.json on a daemon thread
    and echoes to the console synchronously, so the ON path holds the new
    value while a re-read here could still see the old one -- and a stale
    False would pack the pane and then refuse to capture behind it."""
    stale = options()                          # what the file still says
    w = cp.PreviewWorker(get_option=stale, pipeline=FakePipeline(),
                         sensing=Policy())
    assert w.start() is False                  # …without the override
    assert w.start(enabled=True) is True       # …and with it
    try:
        assert w.latest().reason != cp.REASON_DISABLED
    finally:
        w.stop()
    # The OFF direction never re-reads at all, so it cannot race either.
    w.set_enabled(False)
    assert w.running is False


# ---------------------------------------------------------- the rate caps
def test_the_capture_rate_is_capped_at_the_granted_mode_not_at_a_fixed_bug():
    """The ceiling is the MODE's granted nominal, 30 -- read back from the
    driver by scripts/camera_mode_probe.py (2026-09-03): 1280x720 MJPG,
    granted in either set order. Above it there is nothing to fetch. The
    APP does not reach it: 7.5 fps with him at the desk (his log), and the
    most the probe has reached is 15 with the driver's default buffers
    (module docstring). The request is a ceiling the device decides
    whether to honour, not a rate.
    """
    assert cp.preview_fps(options()) == cp.DEFAULT_FPS
    assert cp.preview_fps(options(**{cp.OPTION_FPS: 60})) == cp.MAX_FPS
    assert cp.MAX_FPS == 30.0
    assert cp.preview_fps(options(**{cp.OPTION_FPS: 0})) == cp.MIN_FPS
    assert cp.preview_fps(options(**{cp.OPTION_FPS: "nonsense"})) == \
        cp.DEFAULT_FPS


def test_the_default_is_the_rate_the_mode_delivered_not_the_one_it_advertises():
    """An earlier version of this test asserted DEFAULT_FPS == 15 and called
    the device's 7.5 fps an artefact of a fixed 1920x1080 bug. It was not
    (measured; see the module docstring): 7.5 is what the app got from
    1280x720 with him at the desk at 10 requested (his 15 was clamped to
    10 by the build then running; a second version of this docstring said
    "10 or 15", which the log does not support) and again at 15 later that
    day, so that is the default. His config's own 15 still passes through
    unclamped -- a ceiling above the delivered rate costs nothing but a
    parked read -- and the clamp that once turned it into 10 is not coming
    back. assistant_config.DEFAULTS carries the same number; the test
    below pins them equal."""
    assert cp.DEFAULT_FPS == 7.5
    assert cp.MIN_FPS <= cp.DEFAULT_FPS <= cp.MAX_FPS
    assert cp.preview_fps(options(**{cp.OPTION_FPS: 15.0})) == 15.0


def test_the_ui_poll_sits_well_clear_of_the_60_hz_slot():
    """The pane polls on its own after-chain, not on the reactor's grid. It
    has to be slow enough to be invisible next to a 16.67 ms slot and fast
    enough that a frame is not held back a whole period -- at the default,
    at his configured 15, and at the 30 ceiling, where the 20 ms floor is
    what keeps it off the slot."""
    for fps in (1.0, 6.0, cp.DEFAULT_FPS, 10.0, 15.0, 30.0):
        ms = cp.poll_ms(fps)
        assert ms >= 20                              # never near a slot
        assert ms <= 1000.0 / fps                    # …and never a frame late
    assert cp.poll_ms(6.0) == 83
    assert cp.poll_ms(cp.DEFAULT_FPS) == 67
    assert cp.poll_ms(15.0) == 33
    assert cp.poll_ms(30.0) == 20                    # the floor, not 17


@pytest.mark.parametrize("fps,stride", [(0.0, 1), (2.0, 1), (7.5, 1),
                                        (10.0, 1), (15.0, 2), (30.0, 4)])
def test_the_detect_cadence_is_every_nth_picture(fps, stride):
    """N = max(1, round(picture_fps / DETECT_FPS)). A wall-clock period was
    INERT at the 7.5 fps the device delivers (133 ms is already past 125)
    and quantised to every second picture at 10 -- 5 Hz boxes, slower than
    the pane managed before the split."""
    assert cp.detect_stride(fps) == stride


@pytest.mark.parametrize("fps,lo,hi", [(7.5, 15, 15), (10.0, 20, 20),
                                       (15.0, 15, 16), (30.0, 14, 16)])
def test_the_detect_cadence_follows_the_measured_picture_rate(fps, lo, hi):
    """Driven with pictures at the device's rate, not the configured one:
    every picture at 7.5 and 10 delivered fps, every second at 15, every
    fourth at 30. Two seconds of pictures each."""
    clock = Clock()
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock)
    for _ in range(int(round(fps * 2.0))):
        pipe.grab((16, 9))
        clock.tick(1.0 / fps)
    assert lo <= det.calls <= hi, det.calls


# ------------------------------------------------------------- resolving
def test_the_app_feed_wins_and_a_second_one_is_never_built():
    """SensingPolicy.attach REPLACES a device by name, so a preview that
    built its own feed beside the app's would silently take the curfew away
    from the app's lens. The app's feed is therefore not a preference."""
    built = []

    def build(_cfg, _policy):
        built.append(1)
        return object(), ""

    services = type("S", (), {"camera_feed": "the app's feed"})()
    feed, reason, owned = cp.resolve_feed(services, options(), Policy(), build)
    assert feed == "the app's feed"
    assert reason == ""
    assert built == []
    assert owned is False              # BORROWED -- see close(), below


def test_with_no_app_feed_the_preview_builds_one_through_the_gated_path():
    seen = {}

    def build(cfg, policy):
        seen["hfov"] = cfg.get("camera.hfov_deg", 0.0)
        seen["policy"] = policy
        return "a gated feed", ""

    policy = Policy()
    feed, reason, owned = cp.resolve_feed(
        None, options(**{"camera.hfov_deg": 65.6}), policy, build)
    assert feed == "a gated feed"
    assert seen["hfov"] == 65.6            # the config reaches camera.build
    assert seen["policy"] is policy        # …and so does the sensing owner
    assert owned is True                   # we opened it, we close it


def test_no_sensing_owner_means_no_feed_is_built_at_all():
    built = []
    feed, reason, owned = cp.resolve_feed(
        None, options(), None, lambda *_a: (built.append(1), None)[1])
    assert feed is None
    assert owned is False
    assert "sensing" in reason
    assert built == []


def test_a_build_that_explodes_is_a_reason_not_an_exception():
    def build(_cfg, _policy):
        raise RuntimeError("v4l2 said no")
    feed, reason, owned = cp.resolve_feed(None, options(), Policy(), build)
    assert feed is None
    assert owned is False
    assert "v4l2 said no" in reason


def test_the_apps_own_feed_is_never_closed_by_the_preview():
    """``CameraFeed.close()`` bumps the gate epoch and releases the raw
    device, so a preview that closed the app's lens would discard another
    consumer's in-flight read and force a reopen -- on every ambient
    transition, every curfew edge and every toggle-off. The preview stops
    READING it; whether it shuts is the sensing owner's ruling."""
    feed = FakeFeed()
    pipe = cp.PreviewPipeline(feed, owned=False)
    pipe.close()
    assert feed.closes == 0                 # borrowed: left to its owner
    assert pipe.feed is None                # …but this object stops reading


def test_a_feed_the_preview_built_itself_is_the_preview_to_close():
    feed = FakeFeed()
    pipe = cp.PreviewPipeline(feed, owned=True)
    pipe.close()
    assert feed.closes == 1


def test_a_curfew_edge_does_not_shut_the_lens_the_app_handed_over():
    """The failure mode this pairs with: the sensing branch calls
    _close_pipeline on every denial, so a borrowed feed would be closed
    from under the app the first time the curfew engaged."""
    feed = FakeFeed()
    services = type("S", (), {"camera_feed": feed})()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         services=services,
                         sensing=Policy(State(camera=True)),
                         make_pipeline=lambda: cp.build_pipeline(
                             services, options(), Policy(), feed=feed,
                             owned=False))
    assert w.cycle().live is True
    w.sensing = Policy(State(camera=False, reason="curfew",
                             curfew=((21, 0), (7, 0))))
    shot = w.cycle()
    assert shot.reason == cp.REASON_SENSING
    assert shot.image is None
    assert feed.closes == 0                 # …and the app still has its lens
    assert feed.captures == 1               # the preview stopped reading it


# ---------------------------------------------------------- the reduction
def test_the_frame_is_reduced_to_the_pane_box_before_it_crosses_over():
    """Scaled on the capture thread, at the size that will be shown -- so
    the Tk thread never resamples 2.7 MB and no consumer can keep the full
    frame by keeping a reference."""
    img = cp.shrink(frame(1280, 720), (136, 76))
    assert (img.width, img.height) == (136, 76)
    assert img.mode == "RGB"


def test_a_four_by_three_camera_arrives_letterboxed_not_squashed():
    """The aspect decision is made on the CAPTURE thread, so the picture
    that crosses over is already the size it will be drawn at. Stretching
    it into the pane's 16:9 box would squash every face 1.33x and put every
    overlay box slightly off the face it belongs to."""
    pipe = cp.PreviewPipeline(FakeFeed(frames=[frame(640, 480)]),
                              detector=FakeDetector(rows=[]),
                              observe=observer())
    img = pipe.grab((272, 152)).image
    assert (img.width, img.height) == (203, 152)
    assert abs(img.width / img.height - 640 / 480) < 0.01


def test_something_that_is_not_a_frame_reduces_to_nothing_rather_than_raising():
    assert cp.shrink(np.zeros((4, 4), dtype=np.uint8), (8, 8)) is None
    assert cp.shrink(None, (8, 8)) is None


# ------------------------------------------------------- the privacy rules
def test_no_shot_summary_can_carry_pixels():
    """``numbers_only`` is the mechanical half of the rule: a log line or a
    status dict cannot pick the image up by accident, because the only
    field carrying one is excluded by name."""
    shot = cp.PreviewShot(image=object(), faces=(cp.PreviewFace(0.9, 1, 2, 3,
                                                                4),),
                          cap_w=1280, cap_h=720, reason=cp.REASON_LIVE)
    data = shot.numbers_only()
    assert "image" not in data
    for key, value in data.items():
        assert isinstance(value, (int, float, str, bool, dict)), key
    assert set(data["face"]) == {"conf", "x", "y", "w", "h", "yaw_deg",
                                 "attending", "landmarks_ok", "eye_px",
                                 "name", "id_score", "id_ran"}
    assert data["hand"] == {}


def test_the_status_dict_a_diagnostic_would_print_is_numbers_and_strings():
    w = cp.PreviewWorker(get_option=options())
    for value in w.status().values():
        assert isinstance(value, (int, float, str, bool, dict))


@pytest.mark.parametrize("name", ["jarvis/campreview.py", "jarvis/ui/preview.py",
                                  # the hand stage rides grab() on the same
                                  # frame, and the tracker under it is the
                                  # only other code that sees the array
                                  "jarvis/handstage.py", "jarvis/handpose.py"])
def test_no_code_path_writes_a_frame_anywhere(name):
    """The rule, enforced mechanically. Reviewing for an imwrite does not
    survive the next edit; a grep the suite runs does.

    Hunter: *"i dont want you to look at anything the camera sees without my
    explicit permission."* The preview is rendered on HIS screen, so the
    frame may reach one Tk canvas and nothing else -- not a file, not a
    socket, not a bus event.
    """
    source = _code_only(REPO / name)
    forbidden = (
        r"\bimwrite\b", r"\bimencode\b", r"\.save\s*\(", r"\btofile\b",
        r"\bopen\s*\([^)]*['\"]w", r"\bNamedTemporaryFile\b",
        r"\bmkstemp\b", r"\bwrite_bytes\b", r"\bwrite_text\b",
        r"\bsocket\b", r"\brequests\b", r"\burlopen\b", r"\bbus\.publish\b",
    )
    for pattern in forbidden:
        assert not re.search(pattern, source), \
            "%s: %r is a way a frame could leave the process" % (name, pattern)


# ------------------------------------------------- the enrolment tap
# The one seam by which a full frame leaves this module, added 2026-09-04 so
# that face enrolment can run without killing Jarvis to free /dev/video0.
# These tests are the reason it is safe: the tap is offered only while an
# enrolment is blocked asking for a frame, and it is DENIED at the single
# point every camera handback converges on.
class RecordingTap:
    """jarvis.enroltap.FrameTap's surface, counting instead of holding."""

    def __init__(self, want=True, raises=False):
        self.want = bool(want)
        self.raises = bool(raises)
        self.offered = 0
        self.denied = []

    def offer(self, frame):
        self.offered += 1
        if self.raises:
            raise RuntimeError("the tap is broken")
        return self.want

    def deny(self, reason):
        self.denied.append(str(reason))


def test_the_grab_offers_the_full_frame_to_a_tap_that_is_set():
    """The frame the tap gets is the CAPTURE frame, not the pane's 160x90
    thumbnail. Enrolment's quality bar wants a 112 px face, which no
    thumbnail can carry, so an offer made after the shrink would be a
    feature that silently never produced a usable sample."""
    tap = RecordingTap()
    pipe = cp.PreviewPipeline(FakeFeed(frames=[frame(640, 360)]),
                              detector=FakeDetector(), observe=observer())
    pipe.tap = tap
    shot = pipe.grab((16, 9))
    assert tap.offered == 1
    assert shot.cap_w == 640 and shot.cap_h == 360   # it really was the big one


def test_a_pipeline_with_no_tap_offers_nothing():
    """The normal state, and it must cost the capture thread nothing."""
    pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(),
                              observe=observer())
    assert pipe.tap is None
    pipe.grab((16, 9))          # must not raise


def test_a_tap_that_raises_cannot_take_the_preview_down():
    """The pane going dark because a face was being enrolled is precisely
    the failure this whole lane exists to avoid."""
    tap = RecordingTap(raises=True)
    pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(),
                              observe=observer())
    pipe.tap = tap
    shot = pipe.grab((16, 9))
    assert shot.reason == cp.REASON_LIVE
    assert tap.offered == 1


def test_no_frame_is_offered_when_the_feed_gave_none():
    """A missed capture is not a frame, and must not reach the tap as one."""
    tap = RecordingTap()
    pipe = cp.PreviewPipeline(FakeFeed(frames=[]), detector=FakeDetector(),
                              observe=observer())
    pipe.tap = tap
    pipe.grab((16, 9))
    assert tap.offered == 0


def test_closing_the_pipeline_denies_the_tap():
    """THE PRIVACY EDGE. _close_pipeline is the one point the sensing deny,
    the curfew edge, stop() and _release all converge on, so the deny is
    written once there rather than at four call sites that could drift.

    An enrolment reading through a camera that has been handed back must be
    told, not left to time out: run_enrolment treats a False read as fatal
    and ends the run, which is what actually shuts the lens down.
    """
    tap = RecordingTap()
    worker = cp.PreviewWorker(make_pipeline=lambda: cp.PreviewPipeline(
        FakeFeed(), detector=FakeDetector(), observe=observer()))
    worker._pipeline = worker._make()
    worker.set_tap(tap)
    assert worker._pipeline.tap is tap
    worker._close_pipeline()
    assert len(tap.denied) == 1
    assert tap.denied[0]                     # it says WHY, not just that
    # ...and the worker lets go of it, so a second close does not re-deny a
    # tap that belongs to a run which has already finished.
    worker._close_pipeline()
    assert len(tap.denied) == 1


class RecordingHands:
    """jarvis.handstage.HandStage's cancel surface, counting."""

    def __init__(self, raises=False):
        self.raises = raises
        self.cancelled = []

    def cancel(self, why="cancelled"):
        self.cancelled.append(str(why))
        if self.raises:
            raise RuntimeError("the stage is broken")


def test_closing_the_pipeline_puts_the_hand_down():
    """The gesture engine's carry caps run only when a frame arrives to
    test them, so a carry whose frames stopped HERE -- the ACTIVE->AMBIENT
    edge every 45 s of quiet, the curfew, the settings toggle -- stayed
    live for as long as the silence lasted (MEASURED: 60 s, and a spoken
    throw then took the stale subject). Every handback converges on
    _close_pipeline, so the hand is put down there, once."""
    hands = RecordingHands()
    worker = cp.PreviewWorker(hands=hands, make_pipeline=lambda: cp.PreviewPipeline(
        FakeFeed(), detector=FakeDetector(), observe=observer()))
    worker._pipeline = worker._make()
    worker._close_pipeline()
    assert hands.cancelled == ["preview stopped"]
    # stop() with no thread running takes the same door.
    worker.stop()
    assert hands.cancelled == ["preview stopped", "preview stopped"]


def test_a_stage_that_cannot_cancel_does_not_break_the_handback():
    """The close is a privacy edge and a teardown path: a stage with no
    cancel, or one that raises, must not stop the device going back."""
    feed = FakeFeed()
    worker = cp.PreviewWorker(hands=object(), pipeline=cp.PreviewPipeline(
        feed, detector=FakeDetector(), observe=observer()))
    worker._close_pipeline()
    assert feed.closes == 1
    feed = FakeFeed()
    hands = RecordingHands(raises=True)
    worker = cp.PreviewWorker(hands=hands, pipeline=cp.PreviewPipeline(
        feed, detector=FakeDetector(), observe=observer()))
    worker._close_pipeline()
    assert feed.closes == 1 and hands.cancelled == ["preview stopped"]


def test_a_sensing_deny_reaches_the_tap_through_the_close():
    """The realistic path: the curfew starts halfway through an enrolment.
    cycle() refuses before opening anything and closes the pipeline, and the
    enrolment finds out from that."""
    tap = RecordingTap()
    worker = cp.PreviewWorker(
        sensing=Policy(State(camera=False, reason="curfew")),
        make_pipeline=lambda: cp.PreviewPipeline(
            FakeFeed(), detector=FakeDetector(), observe=observer()))
    worker._pipeline = worker._make()
    worker.set_tap(tap)
    shot = worker.cycle()
    assert shot.reason == cp.REASON_SENSING
    assert tap.denied, "an enrolment was left reading a camera it had lost"


def test_a_pipeline_built_mid_run_inherits_the_tap():
    """A capture restarted under a live enrolment must keep delivering, or
    the run stalls against an object nothing hands frames to."""
    tap = RecordingTap()
    # A PERMISSIVE POLICY IS REQUIRED, and that is itself the rule holding:
    # a worker with no sensing owner fails CLOSED, closes its pipeline and
    # denies the tap rather than capturing for an authority it cannot reach.
    worker = cp.PreviewWorker(
        sensing=Policy(State()),
        make_pipeline=lambda: cp.PreviewPipeline(
            FakeFeed(), detector=FakeDetector(), observe=observer()))
    worker.set_tap(tap)                  # no pipeline exists yet
    worker.cycle()                       # ...which builds one
    assert worker._pipeline.tap is tap
    assert tap.offered == 1


def test_setting_no_tap_leaves_the_preview_alone():
    tap = RecordingTap()
    worker = cp.PreviewWorker(
        sensing=Policy(State()),
        make_pipeline=lambda: cp.PreviewPipeline(
            FakeFeed(), detector=FakeDetector(), observe=observer()))
    worker._pipeline = worker._make()
    worker.set_tap(tap)
    worker.set_tap(None)
    assert worker._pipeline.tap is None
    worker.cycle()
    assert tap.offered == 0


def test_the_pipeline_keeps_no_frame_after_the_grab_returns():
    """Nothing about a frame survives on the object: not the array, not a
    crop, not an embedding. The same property visionrig.Rig holds."""
    pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(),
                              observe=observer())
    pipe.grab((16, 9))
    for value in vars(pipe).values():
        assert not isinstance(value, np.ndarray)


def test_a_recognised_face_cannot_grant_anything_because_nothing_reads_it():
    """THE CAPABILITY RULE, mechanically. Identity may REMOVE capability or
    ADD a name; it must never GRANT one.

    A name computed here is display-only, and the reason is structural
    rather than careful: the only modules that import this one are the pane
    that draws it and the window that packs the pane. The wake gate
    (jarvis/eye.resolve_wake), the commander and the bus cannot see a
    PreviewFace at all, so there is no seam by which "the camera recognised
    him" could become "so let him". Eye reads its OWN identity from its own
    lane and is untouched by this file.

    Written as a grep because a reviewer's assurance does not survive the
    next edit and this does: the day somebody imports campreview into
    jarvis/app.py to save a round trip, this fails and asks why.

    2026-09-03: jarvis/ui/sensors_page.py joined the list. It takes ONE
    thing from here -- REASON_WORDS, the sentence for a camera that is not
    running -- so the two surfaces use one vocabulary for the same states
    rather than inventing a second. It does fuse a recognised face into a
    zone verdict ("AT THE DESK"), and that verdict is DISPLAY-ONLY by the
    same structural argument: the page is constructed by main_window and by
    nothing else, it publishes no event, and no module imports it to read
    one. If the arrival greeting ever wants that verdict, the fusion moves
    to a module of its own with its own capability argument -- it does not
    get read off a Tk page.
    """
    importers = set()
    for path in sorted((REPO / "jarvis").rglob("*.py")):
        if path.name == "campreview.py":
            continue
        if re.search(r"^\s*(from|import)\s+jarvis[\s.]*\S*campreview",
                     path.read_text(), re.M):
            importers.add(path.relative_to(REPO).as_posix())
    assert importers <= {"jarvis/ui/preview.py", "jarvis/ui/main_window.py",
                         "jarvis/ui/views.py",
                         "jarvis/ui/sensors_page.py"}, importers
    # The sensors page's verdict really is display-only: only the window
    # that packs it imports it. (A MENTION does not count -- the config's
    # own comment names the file as where the zone model is argued.)
    readers = {p.relative_to(REPO).as_posix()
               for p in sorted((REPO / "jarvis").rglob("*.py"))
               if p.name != "sensors_page.py"
               and re.search(r"^\s*(from|import)\s+\S*sensors_page|"
                             r"^\s*from\s+\S+\s+import\s+[^\n]*\bsensors_page\b",
                             p.read_text(), re.M)}
    assert readers <= {"jarvis/ui/main_window.py"}, readers
    # …and the class itself hands nothing out but a shot: no callback, no
    # sink, no publish. The pane POLLS; nothing here pushes.
    #
    # set_tap JOINED THIS LIST ON 2026-09-04 AND IT IS INSIDE THE CONTRACT,
    # which is worth writing down rather than merely widening the set. It is
    # not a push: the tap is PULL-based, so the capture thread hands a frame
    # over only while an enrolment is blocked in read() asking for one, and
    # with no tap set it is an attribute read. It is not a leak either --
    # jarvis/enroltap.FrameTap is a single slot cleared on take, on deny, on
    # abort and on release, it never touches a disk, and it is denied at
    # _close_pipeline, the one point every camera handback converges on. And
    # crucially it does not weaken the sentence above it: the tap reads
    # through the preview's OWN gated CameraFeed, so the curfew, offline mode
    # and SensingPolicy still own the lens. What it exists for is the reverse
    # of a capability grant -- it lets face ENROLMENT happen without killing
    # Jarvis, and enrolment writes an embedding, it does not read one.
    #
    # note_stage (2026-09-04) is the other direction entirely: a FLOAT IN.
    # The Tk pane posts how long its repaint took so the minute log line
    # can print ``draw`` beside ``grab``; nothing comes back out of it, it
    # holds no frame and no callback, and StageStats only ever stores
    # floats. ``stages`` is that store, readable for the log and the suite.
    api = {n for n in dir(cp.PreviewWorker) if not n.startswith("_")}
    assert api == {"start", "stop", "set_enabled", "latest", "status",
                   "cycle", "fps", "running", "set_tap", "LOG_EVERY_S",
                   "note_stage"}, api


def test_the_gallery_is_only_ever_read_never_written():
    """jarvis/facegallery.py exists because a test destroyed his voiceprint
    by SAVING over it. A UI pane is the last thing that should be able to
    touch the store his face lives in, so this module has no code path that
    writes one -- not save, not add, not purge, not forget."""
    source = _code_only(REPO / "jarvis" / "campreview.py")
    for verb in ("save", "add", "purge", "forget", "rollback", "enroll"):
        assert not re.search(r"\.\s*%s\s*\(" % verb, source), verb
    assert re.search(r"\.\s*match\s*\(", source)   # reading is all it does


# ------------------------------------------------------------- the words
def test_every_reason_has_something_to_say():
    """A pane that went dark without a sentence would be indistinguishable
    from a broken feature."""
    for reason in (cp.REASON_DISABLED, cp.REASON_SENSING, cp.REASON_PIPELINE,
                   cp.REASON_NO_FRAME, cp.REASON_WAITING):
        assert cp.REASON_WORDS[reason]
        assert cp.blank(reason).detail


def test_the_pane_and_the_header_badge_read_the_same_state_object():
    """Two privacy readouts two inches apart telling different stories is
    worse than one, so the detail is derived from the badge's own
    normaliser rather than from a second reading of the policy."""
    from jarvis.ui.sensing_badge import badge_caption
    state = State(camera=False, reason="curfew", curfew=((21, 0), (7, 0)))
    assert "7" in cp.sensing_detail(state)
    assert "7" in badge_caption(state)
    assert cp.sensing_detail(State()) == ""     # nothing to say when it is on


# --------------------------------------- with the vision lane installed
# jarvis/visionrig.py and jarvis/facemodels.py arrive with the camera lane
# (branch vision-bringup). Until they land these skip; once they do, they
# are the join that proves the preview reads the SAME geometry his numbers-
# only self-check prints, rather than a second implementation of it.
def test_the_yaw_and_the_cone_come_from_the_rigs_own_geometry():
    pytest.importorskip("jarvis.visionrig")
    pytest.importorskip("jarvis.facemodels")
    from jarvis.facemodels import Lens
    from jarvis.visionrig import AttentionTracker, HeadModel
    det = FakeDetector(rows=None, input_size=(32, 18))
    tracker = AttentionTracker(20.0, 0.0)
    clock = Clock()
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det,
                              lens=Lens(64, 36, 65.6), head=HeadModel(0.35),
                              tracker=tracker, now=clock)
    # Nose on the eye midpoint -> yaw 0 -> inside a 20 deg cone.
    det.rows = [row(4, 2, 8, 8, nose_dx=0.0)]
    head = pipe.grab((16, 9)).primary
    assert head.yaw_deg == pytest.approx(0.0, abs=0.01)
    assert head.attending is True
    # A ratio well past 0.35 * tan(20 deg) -> outside it. His measured
    # separation is 54 deg (screen) against 14 deg (camera).
    det.rows = [row(4, 2, 8, 8, nose_dx=0.5)]
    clock.tick(1.0)                              # a fresh detection, not the
    away = pipe.grab((16, 9)).primary            # one carried forward
    assert abs(away.yaw_deg) > 20.0
    assert away.attending is False


def test_build_pipeline_never_opens_a_device_when_the_camera_is_off():
    """camera.enabled is False by default, so the honest end state is a
    pipeline with no feed and a reason -- not a VideoCapture."""
    pytest.importorskip("jarvis.camera")
    pipe = cp.build_pipeline(None, options(), Policy())
    assert pipe.feed is None
    assert pipe.reason


def test_a_strong_detection_with_landmarks_that_are_not_a_face_is_never_embedded():
    """The wall. MEASURED on his own camera, 2026-09-03: a door frame was
    detected, embedded and named -- "id hunter 0.39", "0.43", "0.42" in the
    log against a 0.363 bar, while his real face that evening scored 0.51
    and 0.57. The confidence bar does not catch it, because the wall
    cleared 0.6. SFace aligns its crop from the five landmarks, so a row
    whose landmarks are not a face's hands it nonsense -- and SFace answers
    nonsense CONFIDENTLY, not weakly. The geometry is the guard."""
    clock = Clock()
    rec = FakeRecogniser(min_conf=0.6)
    det = FakeDetector(rows=[row(4, 2, 8, 8, conf=0.95)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det,
                              observe=observer(landmarks_ok=False),
                              now=clock, recogniser=rec,
                              gallery=FakeGallery([("hunter", 0.74)]),
                              identity_min=0.363, min_conf=0.6)
    for _ in range(6):
        clock.tick(1.0)
        shot = pipe.grab((16, 9))
    assert rec.rows == []                     # never embedded
    assert shot.primary is not None           # still drawn: it IS a detection
    assert shot.primary.conf == pytest.approx(0.95)
    assert shot.primary.id_ran is False       # and never named
    assert shot.primary.name == ""


# ------------------------------------------------- per-stage timing (09-04)
# WHY THESE EXIST. On 2026-09-03 the preview line read ``3.7 fps  grab 268
# ms`` for a whole evening and the one number on it was the WHOLE cycle, so
# three agents ruled out USB, the GPU and the app's own loop before anybody
# timed the device read alone. The stages are timed separately now and the
# minute line prints each one's p50, with ``-`` for a stage that did not run.
# Every clock here is hand-wound: a fake that advances the clock INSIDE its
# call is how a stage's cost is known to the millisecond, and the reason a
# real device is never needed to test the instrument.
class TickingFeed(FakeFeed):
    """A feed whose capture takes exactly ``cost`` seconds of the fake clock
    -- the device's frame interval, without a device."""

    def __init__(self, clock, cost, **kw):
        super().__init__(**kw)
        self.clock, self.cost = clock, cost

    def capture(self):
        self.clock.tick(self.cost)
        return super().capture()


class TickingDetector(FakeDetector):
    def __init__(self, clock, cost, **kw):
        super().__init__(**kw)
        self.clock, self.cost = clock, cost

    def detect(self, frame):
        self.clock.tick(self.cost)
        return super().detect(frame)


class TickingRecogniser(FakeRecogniser):
    def __init__(self, clock, cost, **kw):
        super().__init__(**kw)
        self.clock, self.cost = clock, cost

    def embed(self, frame, row):
        self.clock.tick(self.cost)
        return super().embed(frame, row)


def test_stage_stats_take_a_p50_per_stage_and_say_absent_for_one_that_never_ran():
    st = cp.StageStats()
    for ms in (10.0, 268.0, 12.0):
        st.note("grab", ms)
    st.extend({"shrink": 1.5, "detect": "3.0"})
    st.note("nonsense", "not a number")            # ignored, never raises
    p = st.p50s()
    assert p["grab"] == 12.0                       # the median, not the mean
    assert p["shrink"] == 1.5 and p["detect"] == 3.0
    assert "embed" not in p and "nonsense" not in p
    assert st.counts()["grab"] == 3
    assert st.p50s(reset=True)["grab"] == 12.0
    assert st.p50s() == {}                         # reset really resets


def test_stage_stats_keep_a_bounded_window_so_a_silent_log_cannot_grow_them():
    st = cp.StageStats()
    for i in range(cp.STAGE_KEEP + 100):
        st.note("grab", float(i))
    assert st.counts()["grab"] == cp.STAGE_KEEP


def test_the_stage_line_prints_every_stage_in_order_with_dash_for_did_not_run():
    line = cp.stage_line({"grab": 267.9, "detect": 3.14, "shrink": 1.8,
                          "draw": 2.06}, cycles=222)
    assert line == ("stages p50 ms [222 cycles]: grab 268  detect 3.1  "
                    "landmarks -  embed -  hand -  offer -  shrink 1.8  "
                    "draw 2.1"), line
    # a stage nobody declared still prints, after the known ones
    assert cp.stage_line({"odd": 4.0}).endswith("draw -  odd 4.0")


def test_a_live_shot_carries_the_device_read_separately_from_the_whole_cycle():
    """The device read is 268 ms, the detector 3 ms: the cycle says 271 and
    the stages say which of the two it is. That sentence is the entire
    reason the instrument exists."""
    clock = Clock()
    feed = TickingFeed(clock, 0.268)
    det = TickingDetector(clock, 0.003, rows=[row(4, 2, 8, 8)],
                          input_size=(32, 18))
    pipe = cp.PreviewPipeline(feed, detector=det, observe=observer(),
                              now=clock)
    shot = pipe.grab((16, 9))
    assert shot.live
    assert shot.stage_ms["grab"] == pytest.approx(268.0)
    assert shot.stage_ms["detect"] == pytest.approx(3.0)
    assert shot.stage_ms["landmarks"] == pytest.approx(0.0)   # no clock cost
    assert shot.stage_ms["shrink"] == pytest.approx(0.0)
    assert shot.grab_ms == pytest.approx(271.0)                # the whole pass
    # stages that had nothing to do are ABSENT, not zero
    for name in ("embed", "hand", "offer", "draw"):
        assert name not in shot.stage_ms, name
    assert set(shot.stage_ms) <= set(cp.STAGES)
    assert all(isinstance(v, float) for v in shot.stage_ms.values())


def test_detect_and_landmarks_are_timed_only_on_the_cycles_they_run():
    clock = Clock()
    det = TickingDetector(clock, 0.003, rows=[row(4, 2, 8, 8)],
                          input_size=(32, 18))
    # a feed that costs no clock: the picture interval is what the ticks
    # between grabs say (15 fps), so the 8 Hz cadence lands on every other
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det,
                              observe=observer(), now=clock)
    first = pipe.grab((16, 9))
    clock.tick(1.0 / 15.0)
    second = pipe.grab((16, 9))                  # carried boxes, no detect
    assert det.calls == 1
    assert first.stage_ms["detect"] == pytest.approx(3.0)
    assert "landmarks" in first.stage_ms
    assert "detect" not in second.stage_ms and "landmarks" not in second.stage_ms
    assert second.stage_ms["grab"] == pytest.approx(0.0)   # still measured


def test_the_embedding_is_its_own_column_and_is_not_counted_as_landmarks():
    clock = Clock()
    rec = TickingRecogniser(clock, 0.0104)
    pipe, _det = identified(clock, recogniser=rec)
    shot = pipe.grab((16, 9))
    assert len(rec.rows) == 1                    # one embedding was taken
    assert shot.stage_ms["embed"] == pytest.approx(10.4)
    # _faces was measured whole; the embed inside it is taken back out
    assert shot.stage_ms["landmarks"] == pytest.approx(0.0)
    assert shot.stage_ms["grab"] == pytest.approx(0.0)


def test_the_offer_and_the_hand_stage_are_timed_when_and_only_when_wired():
    clock = Clock()

    class Tap:
        def __init__(self):
            self.offers = 0

        def offer(self, _frame):
            self.offers += 1
            clock.tick(0.002)

    class Hands:
        def observe(self, *_a):
            clock.tick(0.0056)
            return None

    pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(rows=[]),
                              observe=observer(), now=clock, hands=Hands())
    bare = pipe.grab((16, 9))
    assert bare.stage_ms["hand"] == pytest.approx(5.6)
    assert "offer" not in bare.stage_ms
    pipe.tap = Tap()
    tapped = pipe.grab((16, 9))
    assert tapped.stage_ms["offer"] == pytest.approx(2.0)
    assert tapped.stage_ms["hand"] == pytest.approx(5.6)


def test_a_missed_frame_still_reports_how_long_the_miss_took_nothing_else():
    """A feed that answers None after a 2 s wait is a stuck camera, and
    that wait is the number worth having; the blank shot carries no image
    and no stages, so nothing downstream mistakes it for a picture."""
    clock = Clock()
    feed = TickingFeed(clock, 2.0, frames=[])
    pipe = cp.PreviewPipeline(feed, detector=FakeDetector(rows=[]),
                              observe=observer(), now=clock)
    shot = pipe.grab((16, 9))
    assert not shot.live and shot.stage_ms == {}
    assert pipe._cycle_ms["grab"] == pytest.approx(2000.0)


def test_a_stalled_grab_reaches_the_minute_line_through_the_worker(caplog):
    """The reviewer's blind spot, closed one level up. The pipeline keeps a
    missed grab's cost in ``_cycle_ms`` and -- deliberately -- hands over a
    blank shot with NO stages, so nothing downstream mistakes it for a
    picture. But the worker folded only ``shot.stage_ms`` into the minute
    window, so a device that took two seconds to say nothing never appeared
    on the one line this instrument exists to write. The worker now folds
    the miss's grab time itself; the shot contract is untouched."""
    clock = Clock(100.0)
    feed = TickingFeed(clock, 2.0, frames=[])
    pipe = cp.PreviewPipeline(feed, detector=FakeDetector(rows=[]),
                              observe=observer(), now=clock)
    w = cp.PreviewWorker(get_option=options(**{"camera.preview": True}),
                         sensing=Policy(), pipeline=pipe, now=clock)
    w._logged = clock()                          # the minute has just begun
    shot = w.cycle()
    assert shot.reason == cp.REASON_NO_FRAME, shot.reason
    assert not shot.live and shot.stage_ms == {}      # the contract stands
    p50s = w.stages.p50s()
    assert p50s.get("grab") == pytest.approx(2000.0), p50s
    assert w._stage_cycles == 1
    # ...and it is the LINE that matters: the stall is printed, not merely
    # held. Close the minute and read what the log says.
    clock.tick(cp.PreviewWorker.LOG_EVERY_S)
    with caplog.at_level(logging.INFO, logger="jarvis.campreview"):
        w.cycle()
    lines = [r.getMessage() for r in caplog.records
             if r.name == "jarvis.campreview" and "stages p50" in r.getMessage()]
    assert lines and "grab 2000" in lines[-1], lines


def test_numbers_only_carries_the_stage_costs_as_floats_keyed_by_name():
    shot = cp.PreviewShot(image=object(), reason=cp.REASON_LIVE,
                          stage_ms={"grab": 268.0, "shrink": 1.8})
    data = shot.numbers_only()
    assert data["stage_ms"] == {"grab": 268.0, "shrink": 1.8}
    assert "image" not in data
    assert all(isinstance(v, float) for v in data["stage_ms"].values())


def test_the_minute_line_prints_the_p50_of_every_stage_and_resets(caplog):
    """The log line the next such evening is answered from. The cycle's
    whole cost keeps its place on the line (renamed ``cycle`` -- it was
    printed as ``grab`` and read as the device read alone for a whole
    evening), then the per-stage p50s follow in STAGES order, with ``-``
    for a stage that did not run this minute, and the Tk thread's ``draw``
    arrives through note_stage."""
    clock = Clock(100.0)
    shots = iter([
        cp.PreviewShot(image=object(), cap_w=1280, cap_h=720,
                       reason=cp.REASON_LIVE, seq=1, grab_ms=271.0,
                       stage_ms={"grab": 266.0, "detect": 3.0, "shrink": 2.0}),
        cp.PreviewShot(image=object(), cap_w=1280, cap_h=720,
                       reason=cp.REASON_LIVE, seq=2, grab_ms=270.0,
                       stage_ms={"grab": 270.0, "shrink": 1.6}),
        cp.PreviewShot(image=object(), cap_w=1280, cap_h=720,
                       reason=cp.REASON_LIVE, seq=3, grab_ms=268.0,
                       stage_ms={"grab": 268.0, "detect": 3.4, "shrink": 1.8}),
        cp.PreviewShot(image=object(), cap_w=1280, cap_h=720,
                       reason=cp.REASON_LIVE, seq=4, grab_ms=268.0,
                       stage_ms={"grab": 268.0, "shrink": 1.8}),
    ])

    class Pipe(FakePipeline):
        def grab(self, box, seq=0):
            self.grabs += 1
            return next(shots)

    w = cp.PreviewWorker(get_option=options(**{"camera.preview": True}),
                         sensing=Policy(), pipeline=Pipe(), now=clock)
    w._logged = clock()                          # the minute has just begun
    for _ in range(3):
        w.cycle()
        clock.tick(0.27)
    w.note_stage("draw", 2.1)
    w.note_stage("draw", 2.3)
    w.note_stage("draw", 1.9)
    clock.tick(cp.PreviewWorker.LOG_EVERY_S)     # …and now it is over
    with caplog.at_level(logging.INFO, logger="jarvis.campreview"):
        w.cycle()                                # the 4th shot closes the minute
    lines = [r.getMessage() for r in caplog.records
             if r.name == "jarvis.campreview" and "stages p50" in r.getMessage()]
    assert lines, [r.getMessage() for r in caplog.records]
    line = lines[-1]
    assert "live  faces 0  1280x720  " in line and "cycle 268 ms" in line, line
    assert "grab 268  detect 3.2  landmarks -  embed -  hand -  offer -  " \
           "shrink 1.8  draw 2.1" in line, line
    assert "[4 cycles]" in line, line
    # and the window is reset: the next minute starts empty
    assert w.stages.p50s() == {}


def test_a_stage_number_posted_from_the_pane_is_a_float_in_and_nothing_out():
    w = cp.PreviewWorker(get_option=options())
    assert w.note_stage("draw", 2.5) is None
    assert w.stages.p50s() == {"draw": 2.5}
    w.note_stage("draw", "garbage")             # never raises
    assert w.stages.counts() == {"draw": 1}


# ---------------------------------------------- the 09-03 preview findings
# What the verify lenses found on preview-polish (P01-P14, F56) and what
# the thaw closed on 2026-09-04. Each test here fails with its fix reverted.
def _norm(text: str) -> str:
    """Docstrings wrap where they wrap; the claims are checked as prose."""
    return " ".join(text.split())


def test_the_two_default_rates_are_one_number():
    """P04. AssistantConfig deep-merges DEFAULTS into every config, so
    DEFAULTS['camera']['preview_fps'] is the default a fresh install runs
    at and campreview.DEFAULT_FPS is reached only with no config at all.
    They were 15.0 and 7.5 for a day, which made every "at the 7.5 default"
    sentence in this module a statement about a default the app never
    used. AssistantConfig() here is DEFAULTS -- no file is read."""
    from jarvis.assistant_config import DEFAULTS, AssistantConfig
    assert DEFAULTS["camera"]["preview_fps"] == cp.DEFAULT_FPS
    cfg = AssistantConfig()
    assert cp.preview_fps(cfg.get) == cp.DEFAULT_FPS
    assert cp.poll_ms(cp.preview_fps(cfg.get)) == cp.poll_ms(cp.DEFAULT_FPS)


def test_the_record_says_what_was_asked_of_the_device():
    """P03/P11. Four places said the device delivered 7.4-7.6 fps "whether
    10 or 15 was requested". Every 'capturing at up to' line in that log
    reads 6.0 or 10.0: the build running then (59bb901) clamped his 15 to
    10. The record has to say what was measured."""
    cam_src = _norm((REPO / "jarvis" / "campreview.py").read_text())
    probe_src = _norm((REPO / "scripts" / "camera_mode_probe.py").read_text())
    for text in (cam_src, probe_src):
        assert "whether 10 or 15" not in text
        assert "7.4-7.6 at 15" not in text
    assert "59bb901" in cam_src and "clamped" in cam_src


def test_the_rate_story_is_marked_measured_or_inferred_where_it_is_each():
    """P09, P10, P12. The stale-picture reading of the 11 ms grab is an
    inference (nothing timed frame age); cv2's CAP_PROP_BUFFERSIZE read-back
    is OpenCV's own stored request, not the driver's count; and the probe's
    3.8 fps was the probe's configuration in that light, not the device's
    ceiling -- the A/B that settled it (c228a01) and the app's failure to
    gain from it (9ba1c56) both belong in the record."""
    doc = _norm(cp.__doc__)
    cam = _norm((REPO / "jarvis" / "camera.py").read_text())
    assert "AN INFERENCE, NOT A MEASUREMENT" in doc
    assert "it was a frame that had been waiting" not in doc
    assert "not observable through cv2" in doc
    assert "not observable through cv2" in cam
    assert "set(1) accepted and read back 1" not in doc
    assert "the rate is the device's own" not in cam
    assert "NOT THE DEVICE'S CEILING" in doc
    assert "7.6 fps / 132" in doc and "9ba1c56" in doc


def test_the_hold_docstrings_state_the_measured_bounds():
    """P01, P06, P07, P14: the sentences that promised more than the code
    does are gone, and the measured bounds stand in their place."""
    src = _norm((REPO / "jarvis" / "campreview.py").read_text())
    assert "A face LEAVING clears it immediately" not in src          # P01
    assert "REPLACED IN PLACE" in src and "1.47 s" in src
    assert "0.8 s" in src and "from cold" in src                      # P14
    assert "well below where two separate faces could land" not in src  # P06
    assert "expected, not measured" in src and "800 ms" in src
    assert "every 20th 136 -> 112" in src                             # P07


def test_a_verdict_goes_to_the_face_that_best_matches_its_box():
    """P02. ``_same`` asks a yes/no question of ONE box, and two boxes can
    both qualify: a stranger leaning into his shoulder overlaps the held
    box by more than SUBJECT_IOU, and then "largest" decides which of them
    wears his name. The verdict goes to the best match among the faces in
    the picture; a tie goes to the caller."""
    clock = Clock()
    hold = cp.IdentityHold(agree=2, now=clock)
    hold.observe("hunter", 0.74, A_BOX)
    hold.observe("hunter", 0.75, A_BOX)
    leaning = (60.0, 10.0, 100.0, 100.0)            # IoU 0.333 with A_BOX
    assert cp.iou(A_BOX, leaning) == pytest.approx(1 / 3)
    assert hold.held(leaning) == ("hunter", 0.75, True)     # alone: qualifies
    assert hold.held(leaning, others=(A_BOX,)) == ("", 0.0, False)  # withheld
    assert hold.held(A_BOX, others=(leaning,)) == ("hunter", 0.75, True)
    assert hold.held(A_BOX, others=(A_BOX,)) == ("hunter", 0.75, True)  # tie
    assert hold.held(A_BOX, others=(None,)) == ("hunter", 0.75, True)


class _ByRowGallery:
    """Answers by WHICH face the embedding came from, so a wrong name is a
    name over the other person's box and nothing else."""

    def match(self, vec):
        return ("hunter", 0.8) if vec[0] else ("alice", 0.8)


class _ByRowRecogniser(FakeRecogniser):
    def embed(self, _frame, row):
        self.rows.append(list(row))
        return (1.0, 0.0, 0.0) if row[0] < 6 else (0.0, 1.0, 0.0)


@pytest.mark.parametrize("fps", [15.0, 7.5])
def test_two_overlapping_faces_never_wear_each_others_names(fps):
    """P02, through the pipeline. Two faces overlapping a third of their
    area -- inside what the detectors' own NMS lets through
    (facedetect.NMS_THRESHOLD 0.3, faceinsight.NMS_IOU 0.4) -- with
    "largest" flipping every picture. Measured without the best-match rule
    (synthetic harness, 2026-09-03): about half of all chips wrong and
    never converging. With it: none."""
    from jarvis import faceinsight as fi
    from jarvis import facedetect as fd
    a, b = (2.0, 2.0, 10.0, 10.0), (7.0, 2.0, 10.0, 10.0)
    assert fd.NMS_THRESHOLD <= cp.iou(a, b) <= fi.NMS_IOU
    clock = Clock()
    det = FakeDetector(input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock, recogniser=_ByRowRecogniser(),
                              gallery=_ByRowGallery(), identity_min=0.363,
                              min_conf=0.6)
    wrong = named = 0
    for i in range(150):
        big, small = (a, b) if i % 2 == 0 else (b, a)
        det.rows = [row(big[0], big[1], big[2] + 0.2, big[3] + 0.2),
                    row(small[0], small[1], small[2], small[3])]
        face = pipe.grab((16, 9)).primary
        clock.tick(1.0 / fps)
        if face is None or not face.id_ran or not face.name:
            continue
        named += 1
        is_a = face.x < 12                    # capture pixels: a at 4, b at 14
        if (face.name == "hunter") != is_a:
            wrong += 1
    assert wrong == 0, (wrong, named)
    assert named > 0                          # the rule withholds; it does not blank


def test_the_detect_stride_keeps_its_footing_at_a_half_integer_ratio():
    """P08. At exactly 12 pictures a second the ratio to DETECT_FPS is 1.5
    and float noise in the EMA flips round() between 1 and 2 from second to
    second -- measured [8, 6, 11, 12, 7, 6, 6, 6, 11, 11] detections a
    second with zero jitter. Unreachable on his camera (30/n) but any 12 or
    20 fps device would show it. The stride in force is kept until the
    ratio has moved past the boundary by a margin."""
    assert cp.detect_stride(12.0, 8.0, prev=1) == 1
    assert cp.detect_stride(12.0, 8.0, prev=2) == 2
    assert cp.detect_stride(13.0, 8.0, prev=1) == 2      # 1.625: past it
    assert cp.detect_stride(11.0, 8.0, prev=2) == 1      # 1.375: past it
    assert cp.detect_stride(15.0, 8.0, prev=1) == 2      # his 15 still steps
    assert cp.detect_stride(0.0, 8.0, prev=3) == 1       # unknown rate: learn
    clock = Clock()
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, observe=observer(),
                              now=clock)
    per_second = []
    for _second in range(10):
        before = det.calls
        for _ in range(12):
            pipe.grab((16, 9))
            clock.tick(1.0 / 12.0)
        per_second.append(det.calls - before)
    steady = per_second[2:]
    assert max(steady) - min(steady) <= 1, per_second


def test_a_frame_in_flight_when_stop_returns_never_lands_in_the_slot():
    """P13. stop() sets the SAME event the in-flight generation publishes
    under, and _publish checked identity only -- so a grab that passed
    cycle()'s post-grab check a few instructions before stop() returned
    put a live frame in the slot AFTER "no frame is held" had been
    promised. Replayed here deterministically: the stop lands between the
    post-grab check and the publish (inside the rate measurement)."""
    pipe = FakePipeline()
    w = cp.PreviewWorker(get_option=options(**{cp.OPTION_ENABLED: True}),
                         pipeline=pipe, sensing=Policy())

    def stop_here(_at):
        w.stop(join=False)
        return 0.0
    w._measure_fps = stop_here                 # noqa: SLF001 - the window
    shot = w.cycle()
    assert shot.live is True                   # the cycle did compute one
    assert w.latest().live is False            # …and the slot never saw it
    assert w.latest().reason == cp.REASON_DISABLED
    assert pipe.closes == 1


def test_a_detector_that_raises_is_said_on_the_pane_not_drawn_as_an_empty_room(
        caplog):
    """F56. A detector that raised produced a LIVE shot with no faces and no
    detail, which the pane printed as NO FACE IN FRAME / FACES 0 -- an empty
    room, with his face in the picture -- and only a debug line. Now the
    failure is a sentence in ``detail``, a counter, and ONE warning per
    change of state rather than one per frame."""
    import logging
    clock = Clock()

    class Angry(FakeDetector):
        def detect(self, _frame):
            self.calls += 1
            raise RuntimeError("onnx fell over")

    pipe = cp.PreviewPipeline(FakeFeed(), detector=Angry(input_size=(32, 18)),
                              observe=observer(), now=clock)
    with caplog.at_level(logging.WARNING, logger="jarvis.campreview"):
        shot = pipe.grab((16, 9))
        clock.tick(1.0)
        again = pipe.grab((16, 9))
    assert shot.live is True and shot.faces == ()
    assert shot.detail.startswith(cp.DETAIL_DETECTOR_FAILED)
    assert "RuntimeError" in shot.detail
    assert again.detail == shot.detail
    assert pipe.detector_errors == 2
    assert shot.numbers_only()["detail"] == shot.detail      # a log can say it
    warned = [r for r in caplog.records if r.name == "jarvis.campreview"
              and r.levelno >= logging.WARNING]
    assert len(warned) == 1, [r.getMessage() for r in warned]
    # Recovery is one more warning, and the sentence goes.
    pipe.detector = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    clock.tick(1.0)
    with caplog.at_level(logging.WARNING, logger="jarvis.campreview"):
        back = pipe.grab((16, 9))
    assert back.detail == "" and back.faces
    warned = [r for r in caplog.records if r.name == "jarvis.campreview"
              and r.levelno >= logging.WARNING]
    assert len(warned) == 2
