"""Display-free pins on scripts/ui_shots.py, the console photo rig.

The rig builds the REAL MainWindow on a private Xvfb and drives it through
its states with bus events. Nothing here opens a display: these tests pin
the three things about the rig that must stay true whatever it is asked to
photograph --

  * its Services carry NO camera feed and a sensing stand-in that cannot
    reach a device (Hunter, 2026-09-02: "I dont want you to look at anything
    the camera sees without my explicit permission");
  * its source never imports cv2 or the vision lane, never names a video
    device node, and never constructs the real capture worker;
  * its state list is the one its docstring promises, in order, with the
    one state this tree cannot show marked as skipped.

No Tk root is created. The module is imported from its path; the heavy
imports inside it (jarvis.ui.main_window) happen only in the helpers that
need them.
"""
import importlib.util
import os
import re
import threading

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "ui_shots.py")


@pytest.fixture(scope="module")
def rig():
    spec = importlib.util.spec_from_file_location("ui_shots_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def source():
    with open(SCRIPT, encoding="utf-8") as fh:
        return fh.read()


# ------------------------------------------------------------ the states
def _docstring_states(doc: str) -> list:
    # `08b`: a lettered state sits between its bare number and the next
    return [(m.group(1), m.group(2))
            for m in re.finditer(r"^  (\d\d[a-z]?) +([a-z][a-z-]*)\b", doc, re.M)]


def test_the_state_list_is_the_one_the_docstring_promises(rig):
    listed = _docstring_states(rig.__doc__)
    assert listed, "the docstring carries no numbered state list"
    assert [(n, s) for n, s, _why in rig.STATES] == listed


def test_the_states_are_numbered_in_order_and_the_skips_say_why(rig):
    numbers = [n for n, _s, _w in rig.STATES]
    # 01..25 without a gap; a lettered state (08b) follows its bare number
    ints = [int(n[:2]) for n in numbers]
    assert ints == sorted(ints)
    assert sorted(set(ints)) == list(range(1, max(ints) + 1))
    for i, n in enumerate(numbers):
        if len(n) > 2:
            assert numbers[i - 1] == n[:2], f"{n} does not follow {n[:2]}"
    assert "08b" in numbers
    skipped = {n: why for n, _s, why in rig.STATES if why}
    assert set(skipped) == {"13"}
    for why in skipped.values():
        assert "59bb901" in why
    # the docstring marks the same one as SKIPPED, nothing else
    doc_skips = re.findall(r"^  (\d\d[a-z]?) +\S+\s+SKIPPED", rig.__doc__, re.M)
    assert sorted(doc_skips) == ["13"]


def test_skip_reasons_are_looked_up_by_number_not_by_position(rig):
    # STATES[12] stopped being 13-timer the day 08b went in
    assert rig.skip_reason("13") == rig.STATES[13][2]
    assert rig.STATES[13][:2] == ("13", "timer")
    assert rig.skip_reason("24") == ""
    assert rig.skip_reason("99") == ""


def test_the_cleared_shot_speaks_the_commanders_own_line(rig):
    from jarvis.commander import TRANSCRIPT_CLEAR_LINE
    assert rig.TRANSCRIPT_CLEAR_LINE == TRANSCRIPT_CLEAR_LINE


def test_shot_filenames_are_number_dash_slug(rig):
    assert rig.shot_filename("03", "listening") == "03-listening.png"
    names = [rig.shot_filename(n, s) for n, s, _w in rig.STATES]
    assert len(set(names)) == len(names)


# ---------------------------------------------------------- the services
def test_the_rigs_services_carry_no_camera_feed(rig):
    svc = rig.build_services()
    assert svc.camera_feed is None
    # the sensing stand-in: a state to read and a curfew, nothing that can
    # attach, stop or resume a device
    st = svc.sensing.state()
    assert isinstance(st.camera, bool) and isinstance(st.offline, bool)
    assert svc.sensing.curfew() == ((21, 0), (7, 0))
    for verb in ("attach", "disable", "enable", "set_curfew", "release"):
        assert not hasattr(svc.sensing, verb), verb
    assert isinstance(svc.desk_idle_s(), float)
    assert isinstance(svc.room_state(gpu_pct=None), dict)


def test_every_service_callable_is_harmless(rig):
    svc = rig.build_services()
    for name in ("start_recording", "stop_recording", "cancel_recording",
                 "toggle_hotword", "open_terminal", "alarm_action",
                 "approval_answer", "uncertain_answer", "board_closed"):
        getattr(svc, name)("x", True, 10)
    assert svc.get_option("console.board") is True
    assert svc.get_option("camera.preview", False) is False
    assert svc.get_option("no.such.key", "dflt") == "dflt"
    assert [c[0] for c in svc._rig_calls][:2] == ["start_recording", "stop_recording"]


def test_the_sensing_stand_in_covers_the_three_badge_states(rig):
    # The TONES, not the words: the words are the badge's own (ui-telemetry
    # shortened CAMERA OFF to CAM OFF for the header fit, and
    # tests/test_sensing_badge.py pins them); the stand-in's job is to reach
    # all three states through both the state() read and the event.
    from jarvis.ui.sensing_badge import (TONE_CURFEW, TONE_OFF, TONE_ON,
                                         badge_tone, badge_word)
    s = rig.RigSensing("on")
    assert badge_tone(s.state()) == badge_tone(s.event()) == TONE_ON
    s.set("curfew")
    assert badge_tone(s.state()) == badge_tone(s.event()) == TONE_CURFEW
    s.set("offline")
    assert badge_tone(s.state()) == badge_tone(s.event()) == TONE_OFF
    words = set()
    for mode in ("on", "curfew", "offline"):
        s.set(mode)
        words.add(badge_word(s.state()))
    assert len(words) == 3, words


# -------------------------------------------------- the capture stand-in
def test_the_stand_in_worker_never_starts_a_thread_and_never_goes_live_on_its_own(rig):
    from jarvis import campreview as cp
    w = rig.RigPreviewWorker(get_option=None, sensing=None, services=None,
                             box=(272, 152))
    before = threading.active_count()
    assert w.start(enabled=True) is True
    assert threading.active_count() == before
    shot = w.latest()
    assert shot.reason == cp.REASON_SENSING and shot.image is None
    assert not shot.live
    w.stop(join=False)
    assert w.latest().reason == cp.REASON_DISABLED
    assert threading.active_count() == before
    # the pane's poll interval comes from fps; it must be a real number
    assert 1.0 <= w.fps <= 10.0
    assert "image" not in w.status()


def test_the_only_live_picture_is_one_this_script_draws(rig):
    from jarvis import campreview as cp
    box = (272, 152)
    shot = rig.synthetic_shot(box)
    assert shot.live and shot.reason == cp.REASON_LIVE
    assert shot.image.width <= box[0] and shot.image.height <= box[1]
    assert len(shot.faces) == 1
    face = shot.primary
    assert face.conf == pytest.approx(0.74) and face.attending
    assert "image" not in shot.numbers_only()
    w = rig.RigPreviewWorker(box=box)
    w.show(shot)
    assert w.latest().seq == 1 and w.latest().live
    w.show(shot)
    assert w.latest().seq == 2


# ---------------------------------------------------- the radar stand-in
def test_the_radar_stand_in_is_a_transport_and_never_a_socket(rig):
    radar = rig.RigRadar()
    body = radar("http://192.0.2.10/binary_sensor/Presence", 3.0)
    assert rig.__dict__.get("socket") is None            # none is imported
    from jarvis.roomsensor import parse_state
    assert parse_state(body) is True
    from jarvis.ui.sensors_page import parse_distance_cm
    cm = parse_distance_cm(radar("http://192.0.2.10/sensor/Detection%20distance",
                                 3.0))
    assert cm == 142.0
    # a host it does not know behaves exactly as an unflashed ESP32 does
    with pytest.raises(OSError):
        radar("http://192.0.2.11/binary_sensor/Presence", 3.0)
    with pytest.raises(OSError):
        radar("http://192.168.50.51/binary_sensor/Presence", 3.0)


def test_the_rigs_two_rooms_are_test_net_literals_and_not_his_lan(rig):
    # RFC 5737 TEST-NET-1: reserved for documentation, routed nowhere.
    for spec in rig.OPTIONS["presence.rooms"]:
        assert spec["url"].startswith("http://192.0.2."), spec
    assert "192.168." not in open(SCRIPT, encoding="utf-8").read()


def test_the_sensors_page_gets_its_transport_before_it_is_ever_shown(source):
    # The page builds a SensorPoller with roomsensor's REAL http client in
    # its __init__; the rig must replace it before show() polls anything.
    swap = source.index("win.sensors.poller = sensors.SensorPoller")
    assert swap < source.index("win.sensors_toggle()")
    assert "get=self.radar" in source


def test_the_named_face_state_leaves_the_anonymous_one_alone(rig):
    plain = rig.synthetic_shot((272, 152))
    named = rig.synthetic_shot((272, 152), name="hunterp", id_score=0.71)
    assert plain.primary.name == "" and plain.primary.id_ran is False
    assert named.primary.name == "hunterp" and named.primary.id_ran is True
    assert named.primary.conf == plain.primary.conf == pytest.approx(0.74)


def test_the_import_blocker_refuses_cv2_and_the_vision_lane(rig):
    blocker = rig._LensBlocker()
    for name in ("cv2", "cv2.data", "jarvis.camera", "jarvis.eye",
                 "jarvis.facedetect", "jarvis.visionrig"):
        with pytest.raises(ImportError):
            blocker.find_spec(name)
    assert blocker.find_spec("json") is None
    assert blocker.find_spec("jarvis.campreview") is None
    assert "cv2" in rig.BLOCKED_MODULES and "jarvis.camera" in rig.BLOCKED_MODULES


# ------------------------------------------------------------ the source
def test_the_source_never_imports_cv2_or_the_camera_lane(source):
    assert not re.search(r"^\s*(import|from)\s+cv2\b", source, re.M)
    assert not re.search(r"^\s*import\s+jarvis\.(camera|eye|facedetect|visionrig)\b",
                         source, re.M)
    assert not re.search(r"^\s*from\s+jarvis\.(camera|eye|facedetect|visionrig)\b",
                         source, re.M)
    assert not re.search(r"^\s*from\s+jarvis\s+import\s+[^\n]*\b(camera|eye|"
                         r"facedetect|visionrig)\b", source, re.M)


def test_the_source_never_names_a_video_device_and_never_opens_a_capture(source):
    assert "/dev/" + "video" not in source
    assert "VideoCapture" not in source
    assert "grab(" not in source.replace("ImageGrab.grab(", "")
    assert "capture()" not in source


def test_the_source_never_constructs_the_real_preview_worker(source):
    # the script constructs no worker itself -- the window does, through the
    # patched name -- so any PreviewWorker( call it carries must be the
    # stand-in's, and none at all is the expected case
    calls = re.findall(r"(\w*PreviewWorker)\(", source)
    assert set(calls) <= {"RigPreviewWorker"}, calls
    # ...and the stand-in is installed BEFORE the window is built
    assert source.index("campreview.PreviewWorker = RigPreviewWorker") \
        < source.index("mw.create(services)")


def test_the_source_asserts_its_own_guards(source):
    assert "services.camera_feed is None" in source
    assert "isinstance(win.preview_worker, RealWorker)" in source
    assert "blocked modules loaded" in source
    assert "CONFIG.save = lambda: None" in source


# ------------------------------------------------- his desktop is off limits
def test_the_rig_refuses_his_desktop_displays(rig):
    """ensure_display() attaches to ANY live server at --display, so
    `--display :1` would have put the console on his real desktop."""
    for d in (":0", ":1", ":0.0", ":1.0", "localhost:1", "localhost:1.0", ""):
        assert not rig.display_allowed(d), d
        with pytest.raises(SystemExit):
            rig.refuse_live_display(d)
    for d in (":91", ":92", ":99", ":10", ":1a"):
        assert rig.display_allowed(d), d
        rig.refuse_live_display(d)
    assert rig.FORBIDDEN_DISPLAYS == (":0", ":1")


def test_main_refuses_a_live_display_before_starting_anything(rig, tmp_path,
                                                                monkeypatch):
    touched = []
    monkeypatch.setattr(rig, "_display_alive", lambda d: touched.append(d) or True)
    monkeypatch.setattr(rig, "run_child", lambda a: touched.append("child") or 0)
    for argv in (["--display", ":1", "--out", str(tmp_path)],
                 ["--display", ":0.0", "--out", str(tmp_path)],
                 ["--child", "--look", "holo", "--display", ":1",
                  "--out", str(tmp_path)]):
        with pytest.raises(SystemExit) as exc:
            rig.main(argv)
        assert "his desktop" in str(exc.value)
    assert touched == []
    assert "refuse_live_display(display)" in open(rig.__file__).read() \
        or "refuse_live_display(args.display)" in open(rig.__file__).read()
