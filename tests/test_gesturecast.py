"""Grab, throw, and what happens next -- the wiring (jarvis/handstage.py,
jarvis/gesturecast.py, the hook in jarvis/campreview.py).

The engine (jarvis/gesture.py) and the sinks (jarvis/cast.py) each have
their own suite. This one drives the whole chain the way the console does:
a frame this process made up goes into ``PreviewPipeline.grab()`` (or
straight into ``HandStage.observe``), a FAKE tracker hands back landmark
rows built from tests/test_gesture's synthetic hand, and what comes out is
checked as NUMBERS, TONES, CHIP CALLS and SPOKEN LINES. No camera is opened,
no frame is looked at, no window is made, no socket is touched and no
thread is started: every outward edge of the courier is a recorder.

What is pinned, and why each needs a test rather than a promise:

* OFF means the tracker is never built. Not "runs and discards".
* NO OPINION is ``None``: missing models and a raising tracker answer
  None, never ``present=False``.
* Attention is LATCHED, the face baseline is REQUIRED, a question on the
  floor blocks a grab and ends a carry.
* The grab tone plays BEFORE the spoken subject; a fling to a named side
  lands on the board (the map ships empty) and the untaught line is said
  ONCE; a taught side routes to HPCOMPUTER and is HELD out loud with the
  live reason, the payload falling back to the board.
* Open in place is a drop. A sentence puts a carry down quietly.
* Nothing that leaves the stage can carry a pixel, and no array survives
  on the stage or the courier after a call.
* The frame counters follow the DELIVERED rate, not the configured one.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis import board
from jarvis import campreview as cp
from jarvis import cast as cast_mod
from jarvis import gesturecast as gc_mod
from jarvis import handstage as hs
from jarvis.cast import BOARD_LINE, NOTHING_LINE, UNTAUGHT_LINE, CastSubject
from jarvis.gesture import CastState
from jarvis.gesturecast import GestureCast
from jarvis.handstage import HandShot
from jarvis.visionrig import assert_numbers_only
from tests.test_campreview import FakeDetector, FakeFeed, observer, row as face_row
from tests.test_gesture import EYE_PX, GOOD, H, W, _curls, hand3d, project, seg

FPS = 7.5
STEP = 1.0 / FPS
SUBJECT = "the thesis draft"


# ------------------------------------------------ nothing reaches out
@pytest.fixture(autouse=True)
def _no_network_no_xdotool(monkeypatch):
    reached = []

    def boom(*a, **kw):
        reached.append((a, kw))
        raise AssertionError("the suite must not touch the network or spawn xdotool")

    monkeypatch.setattr(cast_mod.socket, "socket", boom)
    monkeypatch.setattr(cast_mod.socket, "getaddrinfo", boom)
    monkeypatch.setattr(cast_mod, "focused_window_title", boom)
    yield
    assert reached == []


# --------------------------------------------------------------- fakes
class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


class Options:
    def __init__(self, **kw):
        self.data = {"gesture.enabled": True}
        self.data.update(kw)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        return True


class FakeTracker:
    """``detect`` answers the rows the test queued. It never keeps the
    frame, and it asserts it was handed an array, not a picture object."""

    def __init__(self):
        self.rows = ()
        self.calls = 0

    def detect(self, frame):
        assert isinstance(frame, np.ndarray)
        self.calls += 1
        return self.rows


class Chip:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def record(*args):
            self.calls.append((name,) + tuple(args))
        return record

    def names(self):
        return [c[0] for c in self.calls]


class FakeCommander:
    def __init__(self):
        self.open = False
        self.stashed = []
        self._last_document = None

    def question_open(self):
        return self.open

    def stash_destructive(self, run, line):
        self.stashed.append((run, line))


def face(attending=True, eye_px=EYE_PX, landmarks_ok=True):
    return cp.PreviewFace(conf=0.9, x=560.0, y=200.0, w=160.0, h=200.0,
                          attending=attending, landmarks_ok=landmarks_ok,
                          eye_px=eye_px)


def lm_row(img):
    return SimpleNamespace(lm=np.asarray(img, dtype=np.float32), conf=0.9)


def build(clock=None, screen=SUBJECT, track=None, **kw):
    """The courier, its stage and every recorder, wired the way the app
    and the console wire them but with nothing real behind any edge."""
    clock = clock or Clock()
    rec = SimpleNamespace(spoken=[], tones=[], shows=[], chip=Chip(),
                          transfers=[])
    opts = kw.pop("opts", None) or Options()
    commander = FakeCommander()
    providers = {
        "document": lambda: None,
        "track": (lambda: CastSubject("track", track, at=clock())) if track
        else (lambda: None),
        "screen": (lambda: CastSubject("screen", screen, at=clock())) if screen
        else (lambda: None),
    }
    kw.setdefault("probe", lambda: (False, "no port answered", ""))
    kw.setdefault("transfer", lambda dev: rec.transfers.append(dev))
    kw.setdefault("preview_fps", FPS)
    courier = GestureCast(get_option=opts.get, set_option=opts.set,
                          commander=commander, spotify=None,
                          speak=rec.spoken.append, earcon=rec.tones.append,
                          board_show=lambda: rec.shows.append(1),
                          worker=lambda fn: fn(), now=clock, wall=clock,
                          providers=providers, **kw)
    courier.attach_ui(chip=rec.chip, post=lambda fn: fn())
    tracker = FakeTracker()
    stage = courier.stage(get_option=opts.get, make_tracker=lambda: tracker)
    return SimpleNamespace(courier=courier, stage=stage, tracker=tracker,
                           rec=rec, opts=opts, commander=commander,
                           clock=clock, frame=np.zeros((36, 64, 3), np.uint8))


def drive(rig, scen, faces=lambda t: (face(),), lead=4, step=STEP,
          until=None):
    """A test_gesture trajectory through the stage, frame by frame, with
    ``lead`` face-only frames first so the baseline exists. Returns the
    shots; stops early when ``until(shot)`` says so."""
    dur, traj = scen
    shots = []
    seq = 0
    for _ in range(lead):
        rig.tracker.rows = ()
        shots.append(rig.stage.observe(rig.frame, faces(0.0), W, H, seq))
        rig.clock.t += step
        seq += 1
    t = 0.0
    while t < dur:
        pos, curl, pi, ya, ro = traj(t)
        img = project(hand3d(_curls(curl)), pos, pi, ya, ro)
        rig.tracker.rows = (lm_row(img),) if img is not None else ()
        shot = rig.stage.observe(rig.frame, faces(t), W, H, seq)
        shots.append(shot)
        if until is not None and shot is not None and until(shot):
            break
        t += step
        rig.clock.t += step
        seq += 1
    return shots


def events(shots):
    return [s.event for s in shots if s is not None and s.event]


def open_in_place():
    """Reach in, close, hold, then open WITHOUT moving: the drop case."""
    return (0.55 + 0.20 + 0.5 + 0.4 + 0.5, lambda t: seg(t, [
        (0.55, (140, 300, 780), (10, 10, 430), 0.10, 0.06, 30, 0, 0),
        (0.20, (10, 10, 430), (10, 10, 425), 0.06, 1.00, 35, 0, 0),
        (0.50, (10, 10, 425), (15, 5, 425), 1.00, 1.00, 35, 0, 0),
        (0.40, (15, 5, 425), (18, 8, 425), 1.00, 0.06, 35, 0, 0),
        (0.50, (18, 8, 425), (18, 8, 425), 0.06, 0.06, 35, 0, 0)]))


# ============================================================ the stage
class TestTheSwitch:
    def test_off_by_default_means_the_tracker_is_never_built(self):
        rig = build(opts=Options(**{"gesture.enabled": False}))
        made = []
        rig.stage._make = lambda: made.append(1) or rig.tracker
        shots = drive(rig, GOOD["his-left"])
        assert all(s is None for s in shots)
        assert made == [] and rig.tracker.calls == 0
        assert rig.rec.tones == [] and rig.rec.spoken == []

    def test_the_default_config_ships_it_off(self):
        from jarvis.assistant_config import DEFAULTS
        assert DEFAULTS["gesture"]["enabled"] is False
        assert DEFAULTS["gesture"]["sinks"] == {}
        assert hs.gesture_enabled(None) is False

    def test_flipping_it_off_mid_carry_puts_the_thing_down(self):
        rig = build()
        drive(rig, GOOD["his-left"], until=lambda s: s.event == "grab")
        assert rig.courier.carrying and rig.courier.held is not None
        rig.opts.data["gesture.enabled"] = False
        assert rig.stage.observe(rig.frame, (face(),), W, H, 99) is None
        assert rig.courier.machine.state is not CastState.CARRYING
        assert rig.courier.held is None
        assert rig.rec.tones[-1] == "held-back"


class TestThePreconditions:
    def test_not_attending_means_the_tracker_does_not_run(self):
        rig = build()
        shots = drive(rig, GOOD["his-left"],
                      faces=lambda t: (face(attending=False),))
        assert rig.tracker.calls == 0
        assert all(s is not None and not s.armed for s in shots)
        assert shots[-1].reason == hs.REASON_UNARMED
        assert events(shots) == []

    def test_attention_is_latched_for_the_configured_window(self):
        rig = build()
        rig.tracker.rows = ()
        s = rig.stage.observe(rig.frame, (face(attending=True),), W, H, 0)
        assert s.armed and rig.tracker.calls == 1
        rig.clock.t += 2.5
        s = rig.stage.observe(rig.frame, (face(attending=False),), W, H, 1)
        assert s.armed and rig.tracker.calls == 2
        rig.clock.t += 1.0                          # 3.5 s since attending
        s = rig.stage.observe(rig.frame, (face(attending=False),), W, H, 2)
        assert not s.armed and rig.tracker.calls == 2

    def test_two_faces_do_not_arm_it(self):
        rig = build()
        s = rig.stage.observe(rig.frame, (face(), face()), W, H, 0)
        assert not s.armed and rig.tracker.calls == 0

    def test_no_face_baseline_means_no_scale_and_no_grab(self):
        rig = build()
        shots = drive(rig, GOOD["his-left"],
                      faces=lambda t: (face(eye_px=0.0),))
        assert rig.tracker.calls > 0
        assert events(shots) == []
        assert all(s.reach == 0.0 for s in shots if s is not None)
        assert rig.stage.status()["baseline_samples"] == 0

    def test_the_baseline_needs_three_samples_and_is_a_median(self):
        rig = build(opts=Options())
        assert rig.stage._baseline((face(eye_px=80.0),), 0.0) == 0.0
        assert rig.stage._baseline((face(eye_px=90.0),), 0.1) == 0.0
        assert rig.stage._baseline((face(eye_px=400.0),), 0.2) == 90.0
        # ...and it forgets a face older than the window
        assert rig.stage._baseline((), 0.2 + hs.FACE_WINDOW_S + 0.1) == 0.0

    def test_a_question_on_the_floor_blocks_a_grab(self):
        rig = build()
        rig.commander.open = True
        shots = drive(rig, GOOD["his-left"])
        assert events(shots) == []
        assert rig.tracker.calls == 0
        assert shots[-1].reason == hs.REASON_QUESTION

    def test_a_question_opening_mid_carry_puts_it_down(self):
        rig = build()
        drive(rig, GOOD["his-left"], until=lambda s: s.event == "grab")
        assert rig.courier.carrying
        rig.commander.open = True
        s = rig.stage.observe(rig.frame, (face(),), W, H, 500)
        assert not rig.courier.carrying and s.reason == hs.REASON_QUESTION
        assert rig.rec.tones[-1] == "held-back"


class TestNoOpinion:
    def test_missing_models_answer_none_and_are_looked_for_again_later(self):
        rig = build()
        tries = []

        def make():
            tries.append(rig.clock.t)
            raise RuntimeError("palm_detection_mediapipe_2023feb.onnx: missing")
        rig.stage._make = make
        assert rig.stage.observe(rig.frame, (face(),), W, H, 0) is None
        assert "missing" in rig.stage.tracker_reason
        rig.clock.t += 5.0
        assert rig.stage.observe(rig.frame, (face(),), W, H, 1) is None
        assert len(tries) == 1
        rig.clock.t += hs.RETRY_MODELS_S
        rig.stage.observe(rig.frame, (face(),), W, H, 2)
        assert len(tries) == 2

    def test_a_raising_tracker_is_no_opinion_not_no_hand(self):
        rig = build()

        def boom(frame):
            raise RuntimeError("ORT fell over")
        rig.tracker.detect = boom
        assert rig.stage.observe(rig.frame, (face(),), W, H, 0) is None

    def test_a_raising_stage_leaves_the_picture_untouched(self):
        class Bad:
            def observe(self, *a):
                raise RuntimeError("no")
        pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(),
                                  observe=observer(), hands=Bad())
        shot = pipe.grab((16, 9))
        assert shot.live and shot.hand is None


# ======================================================== the gesture
class TestTheGesture:
    def test_a_fling_to_his_left_grabs_then_lands_on_the_board(self):
        rig = build()
        shots = drive(rig, GOOD["his-left"])
        assert events(shots) == ["grab", "throw"]
        assert rig.rec.tones == ["heard-you", "done"]
        assert rig.rec.spoken == ["Holding %s, sir." % SUBJECT, BOARD_LINE,
                                  UNTAUGHT_LINE]
        assert rig.rec.shows == [1]
        assert rig.rec.chip.names() == ["hold", "thrown", "landed"]
        assert rig.rec.chip.calls[0] == ("hold", SUBJECT, 8.0)
        assert rig.rec.chip.calls[1] == ("thrown", "left")
        assert rig.courier.held is None
        rec = rig.courier.recent()
        assert rec["status"] == "landed" and rec["target"] == "the board"
        assert rec["spoken"] == SUBJECT and rec["by"] == "gesture"

    def test_the_grab_tone_comes_before_the_spoken_subject(self):
        order = []
        rig = build()
        rig.courier._earcon_fn = lambda n: order.append(("tone", n))
        rig.courier._speak_fn = lambda t: order.append(("say", t))
        drive(rig, GOOD["his-left"], until=lambda s: s.event == "grab")
        assert order[0] == ("tone", "heard-you")
        assert order[1] == ("say", "Holding %s, sir." % SUBJECT)

    def test_a_fling_to_his_right_is_named_right(self):
        rig = build()
        shots = drive(rig, GOOD["his-right"])
        assert events(shots) == ["grab", "throw"]
        assert rig.rec.chip.calls[1] == ("thrown", "right")

    def test_the_untaught_line_is_said_once_per_session(self):
        rig = build()
        drive(rig, GOOD["his-left"])
        drive(rig, GOOD["his-right"])
        assert rig.rec.spoken.count(UNTAUGHT_LINE) == 1
        assert rig.rec.tones == ["heard-you", "done"] * 2

    def test_the_board_line_is_silent_when_the_console_is_on_top(self):
        rig = build(console_visible=lambda: True)
        drive(rig, GOOD["his-left"])
        assert BOARD_LINE not in rig.rec.spoken
        assert rig.rec.shows == [1]

    def test_a_taught_side_goes_to_hpcomputer_and_is_held_out_loud(self):
        rig = build(opts=Options(**{"gesture.sinks": {"left": "hpcomputer"}}))
        shots = drive(rig, GOOD["his-left"])
        assert events(shots) == ["grab", "throw"]
        assert rig.rec.tones == ["heard-you", "warning"]
        held = [s for s in rig.rec.spoken if "isn't answering" in s]
        assert len(held) == 1 and "no port answered" in held[0]
        assert UNTAUGHT_LINE not in rig.rec.spoken
        assert rig.rec.shows == [1]                 # fell back to the board
        assert rig.rec.chip.calls[-1] == ("held", "HPCOMPUTER")
        rec = rig.courier.recent()
        assert rec["status"] == "held" and rec["target"] == "HPCOMPUTER"

    def test_the_other_side_still_goes_to_the_board(self):
        rig = build(opts=Options(**{"gesture.sinks": {"left": "hpcomputer"}}))
        drive(rig, GOOD["his-right"])
        assert rig.rec.tones == ["heard-you", "done"]
        assert UNTAUGHT_LINE not in rig.rec.spoken

    def test_opening_the_hand_where_it_is_is_a_drop(self):
        rig = build()
        shots = drive(rig, open_in_place())
        assert events(shots) == ["grab", "drop"]
        assert rig.rec.tones == ["heard-you", "held-back"]
        assert rig.rec.shows == []
        assert rig.rec.chip.names() == ["hold", "dropped"]
        assert rig.courier.held is None
        assert BOARD_LINE not in rig.rec.spoken

    def test_nothing_to_hold_is_one_tone_then_a_sentence(self):
        rig = build(screen=None)
        shots = drive(rig, GOOD["his-left"])
        assert events(shots) == ["drop"]
        assert rig.rec.tones == ["held-back"] and rig.rec.spoken == []
        assert rig.rec.chip.calls == []
        drive(rig, GOOD["his-left"])               # inside 10 s
        assert rig.rec.spoken == [NOTHING_LINE]

    def test_the_grab_can_be_told_not_to_speak(self):
        rig = build(opts=Options(**{"gesture.speak_grab": False}))
        drive(rig, GOOD["his-left"], until=lambda s: s.event == "grab")
        assert rig.rec.tones == ["heard-you"]
        assert not any(s.startswith("Holding") for s in rig.rec.spoken)
        assert rig.rec.chip.calls[0] == ("hold", SUBJECT, 8.0)

    def test_someone_else_in_frame_vetoes_the_throw_silently(self):
        rig = build(identity=lambda: "guest")
        drive(rig, GOOD["his-left"])
        assert rig.rec.tones == ["heard-you"]
        assert rig.rec.shows == []
        assert rig.courier.recent()["status"] == "vetoed"
        assert UNTAUGHT_LINE not in rig.rec.spoken

    def test_a_sentence_puts_a_live_carry_down_quietly(self):
        rig = build()
        drive(rig, GOOD["his-left"], until=lambda s: s.event == "grab")
        tones = list(rig.rec.tones)
        rig.courier.spoken_over()
        assert not rig.courier.carrying and rig.courier.held is None
        assert rig.rec.tones == tones
        assert rig.rec.chip.calls[-1][0] == "dropped"
        rig.courier.spoken_over()                   # idle: a no-op
        assert rig.rec.chip.calls[-1][0] == "dropped"


class TestTheRate:
    def test_the_counters_follow_the_delivered_rate_not_the_configured(self):
        rig = build(preview_fps=15.0)             # his live config's ask
        for i in range(12):
            rig.stage.observe(rig.frame, (face(),), W, H, i)
            rig.clock.t += STEP                    # ...but 7.5 delivered
        assert 7.0 < rig.stage.fps_measured < 8.0
        assert rig.courier.machine.t.dwell_frames == 3
        assert abs(rig.courier.machine.stall_s - 3.0 / 7.5) < 0.02

    def test_a_faster_feed_scales_the_frame_counters_up(self):
        rig = build()
        for i in range(12):
            rig.stage.observe(rig.frame, (face(),), W, H, i)
            rig.clock.t += 1.0 / 15.0
        assert rig.courier.machine.t.dwell_frames == 6
        assert abs(rig.courier.machine.stall_s - 0.2) < 0.01


# ================================================================ purity
def _arrays_in(obj, seen=None, depth=0):
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 6:
        return []
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return [obj]
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _arrays_in(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            out += _arrays_in(v, seen, depth + 1)
    elif hasattr(obj, "__dict__"):
        out += _arrays_in(vars(obj), seen, depth + 1)
    return out


class TestNothingLeaks:
    def test_no_array_survives_on_the_stage_or_the_courier(self):
        rig = build()
        drive(rig, GOOD["his-left"])
        rig.tracker.rows = ()
        assert _arrays_in(rig.stage) == []
        assert _arrays_in(rig.courier) == []
        assert _arrays_in(rig.courier.machine) == []

    def test_the_frame_is_not_retained_by_a_call(self):
        rig = build()
        frame = np.zeros((36, 64, 3), np.uint8)
        before = sys.getrefcount(frame)
        rig.tracker.rows = (lm_row(project(hand3d(_curls(0.1)),
                                           (10, 10, 430), 30, 0, 0)),)
        rig.stage.observe(frame, (face(),), W, H, 0)
        assert sys.getrefcount(frame) == before

    def test_everything_out_is_numbers_and_strings(self):
        rig = build()
        shots = drive(rig, GOOD["his-left"])
        for s in shots:
            if s is not None:
                assert_numbers_only(s.as_dict())
        assert_numbers_only(rig.stage.status())
        assert_numbers_only(rig.courier.status())
        shot = cp.PreviewShot(image=object(), faces=(face(),), cap_w=W,
                              cap_h=H, reason=cp.REASON_LIVE, hand=shots[-1])
        data = shot.numbers_only()
        assert "image" not in data and data["hand"]["state"] == "cooldown"
        assert_numbers_only(data)

    def test_the_hand_shot_has_no_field_that_could_hold_an_array(self):
        import dataclasses
        for f in dataclasses.fields(HandShot):
            assert f.type in ("bool", "float", "str"), f.name


# ================================================== through the pipeline
class TestThePipelineHook:
    def test_grab_hands_the_stage_the_frame_and_carries_scalars(self):
        rig = build()
        rows = [face_row(4, 4, 8, 8, conf=0.95)]
        inner = observer()

        def observe(row, lens, sx, sy, head):
            # visionrig.observe carries eye_px (scaled to capture px); the
            # campreview stub does not, so it is added here to pin that
            # _faces hands it through to the stage untouched.
            obs = inner(row, lens, sx, sy, head)
            obs.eye_px = 3.2 * sx
            return obs
        pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(rows),
                                  observe=observe, hands=rig.stage)
        shot = pipe.grab((16, 9), seq=3)
        assert shot.live and isinstance(shot.hand, HandShot)
        assert shot.faces[0].eye_px == pytest.approx(3.2 * 2.0)
        assert rig.stage.frames == 1
        assert shot.numbers_only()["hand"]["armed"] in (True, False)
        for value in vars(pipe).values():
            assert not isinstance(value, np.ndarray)

    def test_the_worker_carries_the_hand_through_its_rebuild(self):
        rig = build()
        pipe = cp.PreviewPipeline(FakeFeed(), detector=FakeDetector(),
                                  observe=observer(), hands=rig.stage)
        w = cp.PreviewWorker(get_option=lambda k, d=None: {
            "camera.preview": True}.get(k, d), pipeline=pipe,
            sensing=SimpleNamespace(state=lambda: {"camera": True,
                                                   "radar": True,
                                                   "offline": False}),
            hands=rig.stage)
        shot = w.cycle()
        assert shot.live and isinstance(shot.hand, HandShot)
        assert "gesture" in w.status()
        assert_numbers_only(w.status())

    def test_build_pipeline_passes_the_stage_through(self):
        rig = build()
        pipe = cp.build_pipeline(feed=FakeFeed(), hands=rig.stage)
        assert pipe.hands is rig.stage
        pipe = cp.build_pipeline(feed=None, hands=rig.stage,
                                 get_option=lambda k, d=None: d,
                                 build=lambda cfg, pol: (None, "no camera"))
        assert pipe.hands is rig.stage and pipe.feed is None


# ================================================================ voice
class TestByVoice:
    def test_throw_this_on_the_board(self):
        rig = build()
        line, status = rig.courier.throw_by_voice("the board")
        assert (line, status) == (BOARD_LINE, "landed")
        assert rig.rec.tones == ["done"] and rig.rec.shows == [1]
        assert rig.courier.recent()["by"] == "voice"

    def test_throw_this_on_hpcomputer_is_held_with_the_live_reason(self):
        rig = build()
        line, status = rig.courier.throw_by_voice("hp computer")
        assert status == "held"
        assert "HPCOMPUTER isn't answering" in line and "no port answered" in line
        assert rig.rec.tones == ["warning"] and rig.rec.shows == [1]
        assert rig.rec.chip.calls[-1] == ("held", "HPCOMPUTER")

    def test_a_playing_track_still_reaches_hpcomputer(self):
        rig = build(track="Blue in Green")
        line, status = rig.courier.throw_by_voice("the pc")
        assert status == "landed"
        assert rig.rec.transfers == [cast_mod.HPCOMPUTER_DEVICE]
        assert "on HPCOMPUTER" in line and rig.rec.tones == ["done"]

    def test_an_unknown_target_is_refused_by_name(self):
        rig = build()
        line, status = rig.courier.throw_by_voice("the fridge")
        assert status == "refused" and "the fridge" in line
        assert rig.rec.tones == [] and rig.rec.shows == []

    def test_a_voice_throw_takes_the_carried_thing_and_ends_the_carry(self):
        rig = build()
        drive(rig, GOOD["his-left"], until=lambda s: s.event == "grab")
        tones = list(rig.rec.tones)
        line, status = rig.courier.throw_by_voice("the board")
        assert status == "landed" and not rig.courier.carrying
        assert rig.rec.tones == tones + ["done"]    # no held-back in between

    def test_drop_it(self):
        rig = build()
        assert rig.courier.drop_by_voice() == NOTHING_LINE
        drive(rig, GOOD["his-left"], until=lambda s: s.event == "grab")
        assert rig.courier.drop_by_voice() == gc_mod.DROPPED_LINE
        assert not rig.courier.carrying and rig.courier.held is None
        assert rig.rec.tones == ["heard-you", "held-back"]
        assert rig.rec.chip.calls[-1][0] == "dropped"

    def test_what_am_i_holding(self):
        rig = build()
        assert rig.courier.holding_line() == NOTHING_LINE
        drive(rig, GOOD["his-left"], until=lambda s: s.event == "grab")
        assert rig.courier.holding_line() == "Holding %s, sir." % SUBJECT

    def test_teaching_a_side_and_asking_it_back(self):
        rig = build()
        assert rig.courier.side_line("hpcomputer") == \
            gc_mod.SIDE_UNKNOWN_LINE.format(target="HPCOMPUTER")
        line = rig.courier.teach("right", "hp computer")
        assert line == "Right is HPCOMPUTER from now on, sir."
        assert rig.opts.data["gesture.sinks"] == {"right": "hpcomputer"}
        assert rig.courier.side_line("the pc") == "HPCOMPUTER is on your right, sir."
        rig.courier.teach("left", "hpcomputer")    # moves, never doubles
        assert rig.opts.data["gesture.sinks"] == {"left": "hpcomputer"}
        assert rig.courier.teach("right", "the fridge") == \
            cast_mod.TAUGHT_UNKNOWN_LINE.format(name="the fridge")

    def test_a_read_back_sink_only_proposes_and_runs_on_the_yes(self):
        rig = build()

        class Asking:
            name, label = "handoff", "the page"
            reversible, needs_identity, wants_bytes = False, False, False

            def __init__(self):
                self.delivered = 0

            def available(self):
                return True, ""

            def needs_readback(self, subject=None):
                return True

            def deliver(self, subject):
                self.delivered += 1
                return cast_mod.landed_result("Served, sir.", sink=self.name)
        sink = Asking()
        rig.courier.registry["handoff"] = sink
        line, status = rig.courier.throw_by_voice("the page")
        assert status == "proposed"
        assert line == cast_mod.READBACK_LINE.format(
            What=cast_mod._cap(SUBJECT), target="the page")
        assert sink.delivered == 0 and rig.rec.tones == []
        run, stashed_line = rig.commander.stashed[-1]
        assert stashed_line == line
        res = run()
        assert sink.delivered == 1
        assert res.handled and res.speak and res.reply == "Served, sir."

    def test_without_a_commander_nothing_can_hold_a_yes_so_it_refuses(self):
        rig = build()
        rig.courier._commander = None

        class Asking:
            name, label = "handoff", "the page"
            reversible, needs_identity, wants_bytes = False, False, False

            def available(self):
                return True, ""

            def needs_readback(self, subject=None):
                return True

            def deliver(self, subject):
                raise AssertionError("must not deliver on a wave")
        rig.courier.registry["handoff"] = Asking()
        line, status = rig.courier.throw_by_voice("the page")
        assert status == "refused" and line == cast_mod.NO_PROPOSE_LINE


# ================================================================ board
class TestTheBoardSlab:
    def test_the_last_throw_is_a_slab_on_the_board(self):
        rig = build()
        drive(rig, GOOD["his-left"])
        st = board.board_state(cast=rig.courier.recent, now=rig.clock.t)
        assert st.keys == board.PANEL_ORDER + ("cast",)
        p = st.get("cast")
        assert ("STATUS", "LANDED") in p.rows and ("WHAT", SUBJECT) in p.rows
        assert p.tone == "ok" and SUBJECT in p.line
        rig.clock.t += gc_mod.RECENT_TTL_S + 1
        assert rig.courier.recent() is None
        st = board.board_state(cast=rig.courier.recent, now=rig.clock.t)
        assert "cast" not in st.keys


# ========================================================== the config
class TestTheConfig:
    def test_thresholds_read_key_by_key_and_coerce(self):
        get = Options(**{"gesture.reach_min": "2.5", "gesture.dwell_frames": 4.0,
                         "gesture.target_sectors": ["Left"],
                         "gesture.carry_max_s": 6}).get
        t = hs.thresholds_from_options(get)
        assert t.reach_min == 2.5 and t.dwell_frames == 4
        assert isinstance(t.dwell_frames, int)
        assert t.target_sectors == ("left",) and t.carry_max_s == 6.0

    def test_a_value_that_will_not_coerce_is_skipped_not_fatal(self):
        get = Options(**{"gesture.dwell_frames": "three"}).get
        assert hs.thresholds_from_options(get).dwell_frames == 3

    def test_the_stage_and_the_self_check_read_the_same_keys(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "scripts" / "gesture_selfcheck.py"
        spec = importlib.util.spec_from_file_location("gesture_selfcheck", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        opts = Options(**{"gesture.reach_min": 2.5})
        assert mod.thresholds_from_config(opts) == hs.thresholds_from_options(opts.get)

    def test_the_courier_builds_with_no_edges_at_all(self):
        """The app must be able to construct it before anything else
        exists: no probe, no thread, no device, nothing spoken."""
        courier = GestureCast()
        assert courier.machine.state is CastState.IDLE
        assert set(courier.registry) == {"board", "hpcomputer", "handoff"}
        assert courier.holding_line() == NOTHING_LINE
        assert courier.recent() is None
        assert_numbers_only(courier.status())
