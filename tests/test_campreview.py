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


def test_more_faces_than_the_pane_can_draw_are_dropped_not_averaged():
    rows = [row(i, 0, 4 + i, 4 + i) for i in range(6)]
    pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(rows=rows),
                              observe=observer())
    assert len(pipe.grab((16, 9)).faces) == cp.MAX_FACES


def test_a_frame_with_no_face_tells_the_tracker_so_rather_than_going_quiet():
    """A gap has to END the dwell. A tracker that simply stopped being
    called would keep reporting the attention state he walked away from."""
    tracker = FakeTracker([True])
    det = FakeDetector(rows=[row(4, 2, 8, 8)], input_size=(32, 18))
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det, tracker=tracker,
                              observe=observer())
    assert pipe.grab((16, 9)).primary.attending is True
    det.rows = []
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


# ---------------------------------------------------------- the rate caps
def test_the_capture_rate_is_capped_under_what_the_device_delivers():
    """~7.5 fps measured. Asking for 30 does not buy 30 frames, it buys a
    thread that is always inside a 130 ms blocking read."""
    assert cp.preview_fps(options()) == cp.DEFAULT_FPS
    assert cp.preview_fps(options(**{cp.OPTION_FPS: 60})) == cp.MAX_FPS
    assert cp.MAX_FPS <= 10.0
    assert cp.preview_fps(options(**{cp.OPTION_FPS: 0})) == cp.MIN_FPS
    assert cp.preview_fps(options(**{cp.OPTION_FPS: "nonsense"})) == \
        cp.DEFAULT_FPS


def test_the_ui_poll_sits_well_clear_of_the_60_hz_slot():
    """The pane polls on its own after-chain, not on the reactor's grid. It
    has to be slow enough to be invisible next to a 16.67 ms slot and fast
    enough that a frame is not held back a whole period."""
    for fps in (1.0, 6.0, 10.0):
        ms = cp.poll_ms(fps)
        assert ms >= 20                              # never near a slot
        assert ms <= 1000.0 / fps                    # …and never a frame late
    assert cp.poll_ms(6.0) == 83


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
    feed, reason = cp.resolve_feed(services, options(), Policy(), build)
    assert feed == "the app's feed"
    assert reason == ""
    assert built == []


def test_with_no_app_feed_the_preview_builds_one_through_the_gated_path():
    seen = {}

    def build(cfg, policy):
        seen["hfov"] = cfg.get("camera.hfov_deg", 0.0)
        seen["policy"] = policy
        return "a gated feed", ""

    policy = Policy()
    feed, reason = cp.resolve_feed(None, options(**{"camera.hfov_deg": 65.6}),
                                   policy, build)
    assert feed == "a gated feed"
    assert seen["hfov"] == 65.6            # the config reaches camera.build
    assert seen["policy"] is policy        # …and so does the sensing owner


def test_no_sensing_owner_means_no_feed_is_built_at_all():
    built = []
    feed, reason = cp.resolve_feed(None, options(), None,
                                   lambda *_a: (built.append(1), None)[1])
    assert feed is None
    assert "sensing" in reason
    assert built == []


def test_a_build_that_explodes_is_a_reason_not_an_exception():
    def build(_cfg, _policy):
        raise RuntimeError("v4l2 said no")
    feed, reason = cp.resolve_feed(None, options(), Policy(), build)
    assert feed is None
    assert "v4l2 said no" in reason


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
                                 "attending", "landmarks_ok"}


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
    pipe = cp.PreviewPipeline(FakeFeed(), detector=det,
                              lens=Lens(64, 36, 65.6), head=HeadModel(0.35),
                              tracker=tracker)
    # Nose on the eye midpoint -> yaw 0 -> inside a 20 deg cone.
    det.rows = [row(4, 2, 8, 8, nose_dx=0.0)]
    head = pipe.grab((16, 9)).primary
    assert head.yaw_deg == pytest.approx(0.0, abs=0.01)
    assert head.attending is True
    # A ratio well past 0.35 * tan(20 deg) -> outside it. His measured
    # separation is 54 deg (screen) against 14 deg (camera).
    det.rows = [row(4, 2, 8, 8, nose_dx=0.5)]
    away = pipe.grab((16, 9)).primary
    assert abs(away.yaw_deg) > 20.0
    assert away.attending is False


def test_build_pipeline_never_opens_a_device_when_the_camera_is_off():
    """camera.enabled is False by default, so the honest end state is a
    pipeline with no feed and a reason -- not a VideoCapture."""
    pytest.importorskip("jarvis.camera")
    pipe = cp.build_pipeline(None, options(), Policy())
    assert pipe.feed is None
    assert pipe.reason
