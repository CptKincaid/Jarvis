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

    def __init__(self, row, sx, sy):
        self.conf = float(row[14])
        self.x, self.y = float(row[0]) * sx, float(row[1]) * sy
        self.w, self.h = float(row[2]) * sx, float(row[3]) * sy
        self.yaw_deg = float(row[8]) - (self.x + self.w / 2.0)
        self.landmarks_ok = True


def observer(calls=None):
    def observe(row, _lens, sx, sy, _head):
        if calls is not None:
            calls.append((sx, sy))
        return Obs(row, sx, sy)
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
    assert shot.primary.id_ran is False          # …and no chip claims one


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
    """Identity stops being computed whenever the subject drops under the
    detector's bar. The box is still drawn -- a weak detection is still a
    detection -- but the NAME on it has to age out."""
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
    face = pipe.grab((16, 9)).primary
    assert face.id_ran is False                      # no chip at all
    assert face.conf == pytest.approx(0.4)           # …but still a box


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
    DEVICE does not reach it: the same probe measured 3.7-3.9 fps in every
    30 fps mode it has (640x480 included, both formats), at 02:36 and again
    at 07:17, and his own log 7.5 with him at the desk. The request is a
    ceiling the device decides whether to honour, not a rate.
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
    (measured; see the module docstring): 7.5 is what 1280x720 delivered
    with him at the desk whether 10 or 15 was requested, so that is the
    default. His config's own 15 still passes through unclamped -- a ceiling
    above the delivered rate costs nothing but a parked read -- and the
    clamp that once turned it into 10 is not coming back."""
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
                                 "attending", "landmarks_ok",
                                 "name", "id_score", "id_ran"}


def test_the_status_dict_a_diagnostic_would_print_is_numbers_and_strings():
    w = cp.PreviewWorker(get_option=options())
    for value in w.status().values():
        assert isinstance(value, (int, float, str, bool, dict))


@pytest.mark.parametrize("name", ["jarvis/campreview.py", "jarvis/ui/preview.py"])
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
    """
    importers = set()
    for path in sorted((REPO / "jarvis").rglob("*.py")):
        if path.name == "campreview.py":
            continue
        if re.search(r"^\s*(from|import)\s+jarvis[\s.]*\S*campreview",
                     path.read_text(), re.M):
            importers.add(path.relative_to(REPO).as_posix())
    assert importers <= {"jarvis/ui/preview.py", "jarvis/ui/main_window.py",
                         "jarvis/ui/views.py"}, importers
    # …and the class itself hands nothing out but a shot: no callback, no
    # sink, no publish. The pane POLLS; nothing here pushes.
    api = {n for n in dir(cp.PreviewWorker) if not n.startswith("_")}
    assert api == {"start", "stop", "set_enabled", "latest", "status",
                   "cycle", "fps", "running", "LOG_EVERY_S"}, api


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
