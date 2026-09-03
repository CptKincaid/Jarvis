"""Tk-free tests for the console's SENSORS page (jarvis/ui/sensors_page.py).

The page is a diagnostic surface: he opens it to watch the radar and the
camera disagree while he tunes the distance bands. So every rule it renders
is a pure function here -- the URL the distance comes from, the fusion that
turns a presence bit and a range into a zone, the plain-language fault line
for a read that had no opinion, and the validation on the numbers he types
-- and the widget is only a renderer. No root is created and no socket is
opened: the poller's transport is injected, and the one test that exercises
it counts the requests that were NOT sent.

THE TWO RULES THIS FILE EXISTS TO HOLD.

* ``read() -> None`` is NO OPINION and must never render as an empty room.
  A false "nobody there" is what makes Jarvis go quiet on him, and a page
  whose whole purpose is to show him what the sensors think would be the
  worst possible place to introduce that lie.
* The page must never hand ``SensingPolicy`` to its own ``RoomSensor``s.
  ``SensingPolicy.attach`` replaces BY NAME (jarvis/rooms.py says so at
  length), so a second radar attaching as "radar" would silently take the
  curfew away from the app's real one. The page asks ``allowed(RADAR)``
  itself instead, and the assertion is that no request was sent.
"""
import os
import tempfile

import pytest

os.environ.setdefault("JARVIS_LOG_DIR", tempfile.mkdtemp(prefix="jarvis-ui-"))
os.environ.setdefault("JARVIS_ASSISTANT_CONFIG",
                      os.path.join(tempfile.mkdtemp(prefix="jarvis-ui-cfg-"),
                                   "assistant.json"))

from jarvis.roomfabric import RoomSpec  # noqa: E402
from jarvis.ui import sensors_page as sp  # noqa: E402
from jarvis.ui import theme  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_look():
    """Leave the theme in its import-time state (holo, scale 1.0)."""
    yield
    theme.apply_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)


OFFICE = RoomSpec(name="office", url="http://192.168.50.51", label="the office",
                  primary=True)
KITCHEN = RoomSpec(name="kitchen", url="http://192.168.50.52", label="the kitchen")
BANDS = sp.Bands(0.8, 1.8, 1.8, 4.5)


def reading(**kw):
    base = dict(name="office", label="the office", url=OFFICE.url,
                status={"url": OFFICE.url, "blocked": "", "paused": False,
                        "fails": 0, "cooldown_s": 30.0})
    base.update(kw)
    return sp.Reading(**base)


# ------------------------------------------------------------- the URL rule
def test_the_distance_endpoint_is_the_entity_name_and_not_the_object_id():
    # The 00d5b7e lesson: web_server v2 serves an entity at its NAME,
    # percent-encoded. /sensor/detection_distance is a silent 404.
    assert sp.distance_url("http://192.168.50.51") == \
        "http://192.168.50.51/sensor/Detection%20distance"
    assert "detection_distance" not in sp.DISTANCE_PATH


def test_the_distance_endpoint_replaces_a_presence_path_he_pasted():
    # presence.room_sensor_url may already carry the presence entity; the
    # distance is a DIFFERENT entity on the same device, so the path is
    # replaced rather than appended to.
    assert sp.distance_url("http://192.168.50.51/binary_sensor/Presence") == \
        "http://192.168.50.51/sensor/Detection%20distance"
    assert sp.distance_url("http://192.168.50.51/?x=1#f").endswith(
        "/sensor/Detection%20distance")


def test_an_address_with_no_scheme_has_no_distance_endpoint():
    for bad in ("", "   ", "192.168.50.51", "ftp://192.168.50.51", "://x"):
        assert sp.distance_url(bad) == ""


# -------------------------------------------------------- reading a distance
def test_a_distance_body_reads_the_typed_value_and_falls_back_to_the_state():
    assert sp.parse_distance_cm(
        '{"id":"sensor/Detection distance","value":142,"state":"142 cm"}') == 142.0
    # a firmware that omits value: the display string still carries it
    assert sp.parse_distance_cm('{"state":"142 cm"}') == 142.0
    assert sp.parse_distance_cm({"value": 83.5}) == 83.5
    assert sp.parse_distance_cm(b'{"value": 12}') == 12.0


def test_a_distance_that_is_not_a_number_is_no_opinion_and_never_zero():
    for body in ("", "<html>404</html>", '{"value": null}', '{"state":"unknown"}',
                 '{"value": "nan"}', "[1,2]", None, '{"value": -5}'):
        assert sp.parse_distance_cm(body) is None, body
    # zero centimetres is what the LD2410 reports with no target; it is
    # inside the 0.75 m blind zone, so it is not a distance either
    assert sp.parse_distance_cm('{"value": 0}') is None


# ------------------------------------------------------------- the two bands
def test_the_desk_band_wins_where_the_two_bands_touch():
    assert sp.zone_for_distance(1.8, BANDS) == sp.ZONE_DESK
    assert sp.zone_for_distance(0.8, BANDS) == sp.ZONE_DESK
    assert sp.zone_for_distance(1.81, BANDS) == sp.ZONE_ROOM
    assert sp.zone_for_distance(4.5, BANDS) == sp.ZONE_ROOM


def test_a_distance_outside_both_bands_has_no_zone_of_its_own():
    assert sp.zone_for_distance(0.4, BANDS) == ""
    assert sp.zone_for_distance(5.9, BANDS) == ""
    assert sp.zone_for_distance(None, BANDS) == ""


def test_bands_missing_from_the_config_fall_back_to_the_measured_coverage():
    bands = sp.read_bands(lambda key, default=None: default)
    assert (bands.desk_lo, bands.desk_hi) == sp.DEFAULT_DESK_BAND
    assert (bands.room_lo, bands.room_hi) == sp.DEFAULT_ROOM_BAND
    # 4.5 m is what the tuned gates measured, not a guess
    assert bands.room_hi == 4.5


def test_bands_that_arrive_backwards_or_broken_are_repaired_not_dropped():
    # Reading the FILE is generous: a hand-edit that swapped the ends must
    # not cost him the page. (Typing into the page is not -- see below.)
    cfg = {sp.OPTION_DESK_BAND: [1.8, 0.8],
           sp.OPTION_ROOM_BAND: ["nonsense", 4.0]}
    bands = sp.read_bands(lambda key, default=None: cfg.get(key, default))
    assert bands.desk_lo == 0.8 and bands.desk_hi == 1.8
    assert (bands.room_lo, bands.room_hi) == sp.DEFAULT_ROOM_BAND


def test_the_band_marker_is_a_fraction_of_its_band_and_nothing_outside_it():
    assert sp.band_fraction(1.3, 0.8, 1.8) == pytest.approx(0.5)
    assert sp.band_fraction(0.8, 0.8, 1.8) == 0.0
    assert sp.band_fraction(1.8, 0.8, 1.8) == 1.0
    assert sp.band_fraction(4.0, 0.8, 1.8) is None
    assert sp.band_fraction(None, 0.8, 1.8) is None
    assert sp.band_fraction(1.0, 1.0, 1.0) is None      # a zero-width band


# ------------------------------------------------------------- the fault line
def test_a_read_with_no_opinion_is_never_rendered_as_an_empty_room():
    words = sp.fault_line({"url": OFFICE.url, "blocked": "", "paused": False,
                           "fails": 0, "cooldown_s": 30.0})
    assert words
    for lie in ("nobody", "empty", "away", "no one"):
        assert lie not in words.lower(), words


def test_the_fault_line_names_offline_mode_rather_than_a_dead_sensor():
    line = sp.fault_line({"url": OFFICE.url, "blocked": "offline"})
    assert "offline" in line.lower()
    assert "no answer" not in line.lower()
    assert "policy" in sp.fault_line({"url": OFFICE.url,
                                      "blocked": "policy"}).lower()
    assert "stopped" in sp.fault_line({"url": OFFICE.url,
                                       "blocked": "stopped"}).lower()


def test_the_fault_line_says_when_the_breaker_will_try_again():
    line = sp.fault_line({"url": OFFICE.url, "blocked": "", "paused": True,
                          "fails": 3, "cooldown_s": 60.0})
    assert "60" in line and "again" in line.lower()


def test_a_room_with_no_address_says_so_rather_than_no_answer():
    line = sp.fault_line({"url": "", "blocked": ""})
    assert "address" in line.lower()
    assert "answer" not in line.lower()


def test_a_healthy_read_has_no_fault_line_at_all():
    assert sp.fault_line({"url": OFFICE.url, "blocked": "", "paused": False,
                          "fails": 0}, present=True) == ""
    assert sp.fault_line({"url": OFFICE.url, "blocked": "", "paused": False,
                          "fails": 0}, present=False) == ""


# -------------------------------------------------------------- the camera
def test_a_camera_that_was_never_asked_differs_from_one_that_did_not_know_him():
    asked = sp.camera_view({"live": True, "faces": 1, "running": True,
                            "face": {"name": "", "id_score": 0.21,
                                     "id_ran": True}})
    never = sp.camera_view({"live": True, "faces": 1, "running": True,
                            "face": {"name": "", "id_score": 0.0,
                                     "id_ran": False}})
    assert asked.asked and not never.asked
    assert sp.camera_text(asked)[0] != sp.camera_text(never)[0]
    assert "UNKNOWN" in sp.camera_text(asked)[0]
    assert "identity" in sp.camera_text(never)[0].lower()
    # both still see a face, so both put a person at the desk
    assert asked.zone == never.zone == sp.ZONE_DESK


def test_the_camera_row_says_why_it_is_dark_rather_than_showing_nothing():
    off = sp.camera_view({"live": False, "faces": 0, "running": False,
                          "reason": "sensing",
                          "detail": "CAMERA OFF · curfew until 7 am"})
    text, tone = sp.camera_text(off)
    assert "curfew" in text
    assert tone in sp.TONES
    # ...and with no status at all it says the camera is not wired, not that
    # nobody is there
    blank_text, _ = sp.camera_text(sp.camera_view({}))
    assert "nobody" not in blank_text.lower()


def test_a_room_with_no_camera_says_so_rather_than_camera_off():
    none = sp.camera_view({}, present=False)
    text, _tone = sp.camera_text(none)
    assert "no camera" in text.lower()
    assert none.zone == ""


def test_a_named_face_carries_the_name_and_the_score_he_tunes_against():
    view = sp.camera_view({"live": True, "faces": 1, "running": True,
                           "face": {"name": "hunterp", "id_score": 0.712,
                                    "id_ran": True}})
    text, tone = sp.camera_text(view)
    assert "hunterp" in text and "0.71" in text
    assert tone == sp.TONE_OK


# --------------------------------------------------------------- the fusion
def test_the_camera_overrules_the_radar_when_he_lets_it():
    # His words: "camera recognition overrules sensor detection since he can
    # literally see me at my desk".
    cam = sp.camera_view({"live": True, "faces": 1, "running": True,
                          "face": {"name": "hunterp", "id_score": 0.71,
                                   "id_ran": True}})
    v = sp.fuse(present=True, distance_m=3.4, camera=cam, bands=BANDS,
                overrules=True)
    assert v.zone == sp.ZONE_DESK and v.source == "camera"
    assert v.word == sp.ZONE_WORDS[sp.ZONE_DESK]


def test_the_camera_does_not_overrule_when_the_toggle_is_off():
    cam = sp.camera_view({"live": True, "faces": 1, "running": True,
                          "face": {"name": "hunterp", "id_score": 0.71,
                                   "id_ran": True}})
    v = sp.fuse(present=True, distance_m=3.4, camera=cam, bands=BANDS,
                overrules=False)
    assert v.zone == sp.ZONE_ROOM and v.source == "radar"
    v = sp.fuse(present=False, distance_m=None, camera=cam, bands=BANDS,
                overrules=False)
    assert v.zone == sp.ZONE_AWAY and "camera" in v.why.lower()


def test_the_camera_is_the_only_leg_with_an_opinion_when_the_radar_has_none():
    # Nothing is being OVERRULED here -- the radar abstained -- so the
    # camera answers even with the toggle off.
    cam = sp.camera_view({"live": True, "faces": 1, "running": True,
                          "face": {"name": "hunterp", "id_score": 0.71,
                                   "id_ran": True}})
    for overrules in (True, False):
        v = sp.fuse(present=None, distance_m=None, camera=cam, bands=BANDS,
                    overrules=overrules)
        assert v.zone == sp.ZONE_DESK and v.source == "camera"


def test_two_legs_with_no_opinion_is_no_opinion_and_not_an_empty_room():
    v = sp.fuse(present=None, distance_m=None, camera=sp.camera_view({}),
                bands=BANDS, overrules=True)
    assert v.zone == sp.ZONE_UNKNOWN
    assert v.word == sp.ZONE_WORDS[sp.ZONE_UNKNOWN]
    assert "nobody" not in v.word.lower()
    assert v.source == ""


def test_presence_with_an_unknown_distance_is_in_the_room_not_at_the_desk():
    # The radar answers "someone is in the room" and only coarsely "how far".
    # With no range at all the honest answer is the room, never the desk.
    v = sp.fuse(present=True, distance_m=None, camera=sp.camera_view({}),
                bands=BANDS, overrules=True)
    assert v.zone == sp.ZONE_ROOM
    v = sp.fuse(present=True, distance_m=5.8, camera=sp.camera_view({}),
                bands=BANDS, overrules=True)
    assert v.zone == sp.ZONE_ROOM


def test_a_target_inside_the_desk_band_is_at_the_desk_without_a_camera():
    v = sp.fuse(present=True, distance_m=1.4, camera=sp.camera_view({}),
                bands=BANDS, overrules=True)
    assert v.zone == sp.ZONE_DESK and v.source == "radar"


# ------------------------------------------------------------ the whole row
def test_the_page_lists_every_configured_room_in_config_order():
    rows = sp.page_rows([reading(name="office"),
                         reading(name="kitchen", label="the kitchen")],
                        camera_status={}, bands=BANDS, overrules=True,
                        camera_room="office")
    assert [r.name for r in rows] == ["office", "kitchen"]
    # the camera lives in ONE room; the other says so
    assert "no camera" in rows[1].camera_text.lower()


def test_a_row_for_an_unreachable_sensor_still_renders_every_field():
    row = sp.page_rows([reading(present=None, distance_m=None, rtt_ms=None,
                                status={"url": OFFICE.url, "blocked": "",
                                        "paused": True, "fails": 3,
                                        "cooldown_s": 30.0})],
                       camera_status={}, bands=BANDS, overrules=True,
                       camera_room="office")[0]
    assert row.presence_word == "NO OPINION"
    assert row.distance_text == sp.DASH and row.rtt_text == sp.DASH
    assert row.fault and row.verdict.zone == sp.ZONE_UNKNOWN


def test_a_present_row_prints_metres_and_milliseconds_the_way_he_reads_them():
    row = sp.page_rows([reading(present=True, distance_m=1.42, rtt_ms=62.4)],
                       camera_status={}, bands=BANDS, overrules=True,
                       camera_room="office")[0]
    assert row.presence_word == "PRESENT"
    assert row.distance_text == "1.4 m"
    assert row.rtt_text == "62 ms"
    assert row.fault == ""


# --------------------------------------------------------------- the poller
class _Policy:
    """Only the one method the page is allowed to use."""

    def __init__(self, yes=True):
        self.yes = yes
        self.asked = 0

    def allowed(self, kind):
        self.asked += 1
        return self.yes


def test_the_poller_asks_the_policy_before_it_opens_a_socket():
    # "the readings are ignored" is not the same promise as "the radar was
    # not polled", and only the second one is worth anything to him.
    sent = []

    def get(url, timeout):
        sent.append(url)
        return '{"value": true}'

    policy = _Policy(yes=False)
    poller = sp.SensorPoller([OFFICE], policy=policy, get=get)
    rows = poller.poll_once()
    assert sent == []
    assert policy.asked >= 1
    assert rows[0].present is None
    assert "offline" in rows[0].status["blocked"]


def test_the_poller_never_hands_the_policy_to_its_own_room_sensors():
    # SensingPolicy.attach replaces BY NAME: a second "radar" would take the
    # curfew away from the app's real sensor.
    class _Trap(_Policy):
        def attach(self, *a, **k):
            raise AssertionError("the page attached to the sensing policy")

    poller = sp.SensorPoller([OFFICE], policy=_Trap(),
                             get=lambda url, t: '{"value": false}')
    rows = poller.poll_once()
    assert rows[0].present is False


def test_one_poll_reads_presence_and_then_the_distance_from_the_same_device():
    seen = []

    def get(url, timeout):
        seen.append(url)
        if "binary_sensor" in url:
            return '{"id":"binary_sensor/Presence","value":true,"state":"ON"}'
        return '{"id":"sensor/Detection distance","value":142,"state":"142 cm"}'

    row = sp.SensorPoller([OFFICE], get=get).poll_once()[0]
    assert row.present is True
    assert row.distance_m == pytest.approx(1.42)
    assert row.rtt_ms is not None and row.rtt_ms >= 0.0
    assert [u.rsplit("/", 1)[-1] for u in seen] == \
        ["Presence", "Detection%20distance"]


def test_a_sensor_that_has_no_opinion_is_not_asked_for_a_distance():
    # The breaker's promise is that a dead ESP32 costs the poll loop zero
    # syscalls per tick; a second request for its range would undo it.
    seen = []

    def get(url, timeout):
        seen.append(url)
        raise OSError("no route to host")

    row = sp.SensorPoller([OFFICE], get=get).poll_once()[0]
    assert row.present is None and row.distance_m is None
    assert len(seen) == 1
    assert row.rtt_ms is not None          # the request WAS sent and it failed


def test_a_room_with_no_url_is_listed_but_never_polled():
    seen = []
    blank = RoomSpec(name="kitchen", url="", label="the kitchen")
    rows = sp.SensorPoller([blank],
                           get=lambda u, t: seen.append(u)).poll_once()
    assert seen == []
    assert rows[0].name == "kitchen" and rows[0].present is None
    assert "address" in sp.fault_line(rows[0].status).lower()


# ------------------------------------------------------------------ saving
def test_saving_writes_the_three_dotted_keys_and_nothing_else():
    edits, err = sp.band_edits("0.8", "1.8", "1.8", "4.5", True)
    assert err == ""
    assert set(edits) == {sp.OPTION_DESK_BAND, sp.OPTION_ROOM_BAND,
                          sp.OPTION_CAMERA_OVERRULES}
    assert edits[sp.OPTION_DESK_BAND] == [0.8, 1.8]
    assert edits[sp.OPTION_ROOM_BAND] == [1.8, 4.5]
    assert edits[sp.OPTION_CAMERA_OVERRULES] is True
    for key in edits:
        assert key.startswith("presence.")


def test_a_band_edit_that_is_not_a_number_is_refused_with_a_reason():
    edits, err = sp.band_edits("nought point eight", "1.8", "1.8", "4.5", True)
    assert edits == {}
    assert "number" in err.lower() and "desk" in err.lower()


def test_a_band_typed_backwards_is_refused_rather_than_silently_swapped():
    # Reading the file repairs it; a person typing gets told. Two different
    # situations, deliberately two different answers.
    edits, err = sp.band_edits("1.8", "0.8", "1.8", "4.5", True)
    assert edits == {} and "desk" in err.lower()
    edits, err = sp.band_edits("0.8", "1.8", "4.5", "1.8", True)
    assert edits == {} and "room" in err.lower()


def test_a_band_beyond_what_the_radar_can_see_is_refused():
    edits, err = sp.band_edits("0.8", "1.8", "1.8", "9.0", True)
    assert edits == {}
    assert "6" in err            # gates 0..8 at 0.75 m each
    assert sp.MAX_BAND_M == 6.0


def test_the_bands_carry_the_hardware_notes_he_would_otherwise_learn_the_hard_way():
    notes = sp.band_notes(sp.Bands(0.5, 1.4, 1.4, 4.5))
    joined = " ".join(notes).lower()
    assert "0.75" in joined          # nothing at all is detected inside it
    assert "1.5" in joined           # no STILL target inside it
    assert sp.band_notes(BANDS) == ()


def test_a_round_trip_of_zero_is_never_confused_with_one_that_was_not_sent():
    # The dash means "no request left this process"; a real, very fast poll
    # must not borrow it, and must not print 0 either.
    assert sp.fmt_ms(None) == sp.DASH
    assert sp.fmt_ms(0.4) == "<1 ms"
    assert sp.fmt_ms(62.4) == "62 ms"
    assert sp.fmt_ms(1186) == "1186 ms"          # the worst measured sample
    assert sp.fmt_ms(-1) == sp.DASH


def test_the_standing_caption_does_not_claim_a_save_that_has_not_happened():
    # It sits under an UNPRESSED save button, so it may not say "saved".
    assert "saved" not in sp.RESTART_NOTE.lower()
    assert "saved" in sp.SAVED_NOTE.lower()
    assert "restart" in sp.SAVED_NOTE.lower()


def test_the_page_says_a_restart_is_needed_because_nothing_reloads_the_config():
    # AssistantConfig.reload_if_changed has no callers; the page must say so
    # rather than pretend the edit is live.
    assert "restart" in sp.RESTART_NOTE.lower()
    # An AST scan, not a grep: jarvis/app.py and jarvis/address.py both
    # mention the method in a COMMENT saying exactly this, and a grep counts
    # those as callers.
    import ast
    import pathlib
    pkg = pathlib.Path(sp.__file__).parent.parent
    callers = []
    for path in sorted(pkg.rglob("*.py")):
        if path.name == "assistant_config.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and \
                    isinstance(node.func, ast.Attribute) and \
                    node.func.attr == "reload_if_changed":
                callers.append("%s:%d" % (path.name, node.lineno))
    assert callers == [], (
        "reload_if_changed grew a caller (%s); RESTART_NOTE may now be a lie"
        % callers)


def test_saving_goes_through_set_option_so_the_write_stays_atomic_and_0600():
    # AssistantConfig.save() is the only atomic 0600 writer (mkstemp in the
    # same directory, fsync, os.replace, chmod 0600); the page must not open
    # the file itself. An AST scan, so a docstring may still SAY what the
    # module does not do and a method named is_open is not a false hit.
    import ast
    called = set()
    for node in ast.walk(ast.parse(open(sp.__file__, encoding="utf-8").read())):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name):
            called.add(fn.id)
        elif isinstance(fn, ast.Attribute):
            called.add(fn.attr)
    for forbidden in ("open", "dump", "dumps", "write_text", "write_bytes",
                      "_write_private", "mkstemp", "replace"):
        assert forbidden not in called, forbidden
    assert "set_option" in open(sp.__file__, encoding="utf-8").read()


# ------------------------------------------------------- the widget, safely
def test_no_attribute_shadows_a_method_of_the_same_name():
    """``self._camera = {}`` in __init__ shadowed ``def _camera(self)`` and
    turned every repaint into "TypeError: dict object is not callable" --
    found only by running the photo rig, because the pure functions this
    file is otherwise made of never touch the widget. An AST scan is the
    cheap standing guard for the whole class."""
    import ast
    tree = ast.parse(open(sp.__file__, encoding="utf-8").read())
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        methods = {n.name for n in cls.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        clashes = set()
        for node in ast.walk(cls):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for t in targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "self" and t.attr in methods):
                    clashes.add(t.attr)
        assert not clashes, "%s: %s shadow methods of the same name" % (
            cls.name, sorted(clashes))


def test_opening_the_page_never_polls_on_the_tk_thread():
    """A sensor that has gone away costs 3 s per room, and "the sensor has
    gone away" is exactly the state he opens this page in. The first pass
    must come off the poll thread."""
    import re as _re
    src = open(sp.__file__, encoding="utf-8").read()
    body = src[src.index("    def show(self)"):src.index("    def hide(self)")]
    body = _re.sub(r"(?m)#.*$", "", body)      # the comment SAYS poll_once
    assert "poll_once" not in body
    assert "self.poller.start(" in body


def test_the_row_before_the_first_poll_does_not_read_as_an_empty_room():
    row = sp.waiting_row(OFFICE)
    assert row.presence_word == sp.WAITING_WORD
    assert row.distance_text == row.rtt_text == sp.DASH
    assert row.verdict.zone == sp.ZONE_UNKNOWN
    for lie in ("nobody", "empty", "present"):
        assert lie not in row.presence_word.lower()


def test_a_toggle_repaints_from_the_readings_already_on_screen():
    # Flicking "camera overrules radar" must show the change at once, not a
    # poll later, and must not blank the rows in the meantime.
    src = open(sp.__file__, encoding="utf-8").read()
    body = src[src.index("def _on_overrule"):src.index("def save(")]
    assert "self.apply(self._last)" in body


# ------------------------------------------------------------- the boundary
def test_the_page_never_names_a_lens_or_a_microphone():
    src = open(sp.__file__, encoding="utf-8").read()
    assert "/dev/" + "video" not in src
    assert "VideoCapture" not in src
    assert "cv2" not in src
    # the camera reaches this page as NUMBERS from PreviewWorker.status()
    assert ".image" not in src and "PIL" not in src


def test_the_camera_only_ever_arrives_as_the_numbers_only_status_dict():
    from jarvis.campreview import PreviewShot
    keys = set(PreviewShot().numbers_only())
    assert "image" not in keys
    # every key the page reads must exist on that dict (plus the ones
    # PreviewWorker.status adds)
    for key in ("faces", "reason", "detail", "live", "face"):
        assert key in keys, key
