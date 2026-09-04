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
import copy
import os
import tempfile
import threading
import time

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

# HIS LADDERS, 2026-09-03, as assistant.json holds them. The office desk is
# the FARTHER band -- he sits at 3.13 m median and the near space is empty
# -- which is the fact the page's own two-band model had backwards. The
# kitchen has no lens, and its camera_zone says so by being blank.
OFFICE_ROOM = {"name": "office", "enabled": True, "camera_zone": "at the desk",
               "bands": [{"name": "empty space", "near_m": 0.75, "far_m": 2.25},
                         {"name": "at the desk", "near_m": 2.25, "far_m": 3.75}]}
KITCHEN_ROOM = {"name": "kitchen", "enabled": True, "camera_zone": "",
                "bands": [{"name": "the kitchen", "near_m": 0.75, "far_m": 3.0},
                          {"name": "at the door", "near_m": 3.0, "far_m": 3.75}]}


def opts(**over):
    """A ``get_option`` over a dotted dict, the shape the page is handed."""
    data = {"zones": {"enabled": True,
                      "rooms": [copy.deepcopy(OFFICE_ROOM),
                                copy.deepcopy(KITCHEN_ROOM)]}}
    for key, value in over.items():
        node = data
        parts = key.replace("__", ".").split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if value is _GONE:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value

    def get_option(key, default=None):
        node = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node
    return get_option


_GONE = object()


def ladders(**over):
    return sp.read_ladders(opts(**over))


LADDERS = ladders()


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


# ------------------------------------------------------ ONE zone model
# The merge of zone-log and sensors-page left TWO zone models in the tree
# that disagreed, and the disagreement was MEASURED: "the console calls 4 of
# 10 points 'AT THE DESK' from the radar alone, while the zone log has no
# desk band at all and reaches 'at the desk' only through the camera". They
# were different config keys -- presence.desk_band_m here, zones.rooms there
# -- so git saw no conflict. These pin the page onto the ONE model that
# survived five rounds of review and matches his measured room.
def test_the_page_places_a_range_with_the_zone_logs_own_ladder():
    zmap = LADDERS.for_room("office")
    assert [b.name for b in zmap.bands] == ["empty space", "at the desk"]
    # and it is literally the same object type the log places a reading with
    assert isinstance(zmap, sp.ZoneMap)
    assert zmap.place(3.13) == "at the desk"     # where he actually sits
    assert zmap.place(1.0) == "empty space"


def test_the_desk_is_the_FARTHER_band_in_his_office_not_the_nearer_one():
    """The two-band model this page used to carry assumed the desk was the
    NEARER band. His office is the other way round -- the radar sits on the
    desk aimed out across the room, so the near space is empty and he reads
    at 3.13 m median (min 3.00, max 3.30, measured 2026-09-03)."""
    zmap = LADDERS.for_room("office")
    near, far = zmap.bands
    assert near.name == "empty space" and far.name == "at the desk"
    assert far.near_m > near.near_m


def test_each_room_is_placed_by_its_own_ladder_and_never_by_another_rooms():
    assert LADDERS.for_room("kitchen").place(3.19) == "at the door"
    assert LADDERS.for_room("office").place(3.19) == "at the desk"


def test_a_room_with_no_ladder_says_so_rather_than_borrowing_a_geometry():
    got = ladders(zones__rooms=[copy.deepcopy(OFFICE_ROOM)])
    assert got.for_room("kitchen") is None
    v = sp.fuse(present=True, distance_m=3.2, camera=sp.camera_view({}),
                zmap=None, overrules=True)
    assert v.zone == sp.NO_LADDER
    assert "zones.rooms" in v.why


def test_a_ladder_the_zone_model_refuses_is_refused_here_too_and_named():
    """jarvis/zones.py refuses a broken ladder rather than substituting a
    working one, and the page must not substitute either -- a verdict on
    screen that the log will not write is the disagreement all over again."""
    broken = copy.deepcopy(OFFICE_ROOM)
    broken["bands"][1]["near_m"] = 1.0        # now it overlaps "empty space"
    got = ladders(zones__rooms=[broken])
    assert got.for_room("office") is None
    joined = " ".join(got.notes)
    assert "zones.rooms[0].bands" in joined and "overlap" in joined


def test_the_bands_the_page_shows_are_the_bands_the_verdict_uses():
    """The whole claim of this pass, as one assertion: for every metre
    across the ladder, what the page prints is what zones.verdict says."""
    from jarvis import zones as zn
    zmap = LADDERS.for_room("office")
    for cm in range(0, 500, 7):
        metres = cm / 100.0
        v = sp.fuse(present=True, distance_m=metres, camera=sp.camera_view({}),
                    zmap=zmap, overrules=True)
        expected = zn.verdict(zmap, presence=True, distance_m=metres)
        assert v.zone == expected.zone, metres
        assert v.word == expected.zone.upper(), metres


def test_the_page_and_the_zone_log_ask_the_SAME_question_of_the_config():
    """The class fix, as one assertion. scripts/zone_log.py places a
    reading with ``zones.zone_map_for(cfg, room)``; the page places it with
    ``read_ladders(...).for_room(room)``. Over the same config they must be
    the same ladder, band for band -- if they can differ at all the two
    models are back."""
    from jarvis import zones as zn
    get_option = opts()
    cfg = sp._CfgView(get_option)
    got = sp.read_ladders(get_option)
    for room in ("office", "kitchen"):
        theirs = zn.zone_map_for(cfg, room)
        ours = got.for_room(room)
        assert theirs is not None and ours is not None, room
        assert ours.bands == theirs.bands, room
        assert ours.camera_zone == theirs.camera_zone, room
    # and a room the log refuses is refused here too, not substituted
    broken = copy.deepcopy(OFFICE_ROOM)
    broken["bands"][0]["far_m"] = "near"
    bad = opts(zones__rooms=[broken, copy.deepcopy(KITCHEN_ROOM)])
    assert zn.zone_map_for(sp._CfgView(bad), "office") is None
    assert sp.read_ladders(bad).for_room("office") is None


def test_two_lists_of_room_names_that_do_not_join_are_named_on_screen():
    """HIS CONFIG TONIGHT. presence.rooms is empty, so roomfabric falls
    back to the singular keys and polls ONE room called "room" at the
    office address -- while zones.rooms names "office" and "kitchen". They
    join on the room NAME, so nothing joins, every row reads NO ZONE LADDER
    and nothing anywhere says why."""
    singular = RoomSpec(name="room", url=OFFICE.url, label="room", primary=True)
    notes = sp.name_mismatch_note([singular], LADDERS)
    assert len(notes) == 1
    for want in ("'room'", "presence.rooms", "zones.rooms", "'office'",
                 "'kitchen'", "restart"):
        assert want in notes[0], want
    # silent when they DO join, and silent when there is no ladder at all
    assert sp.name_mismatch_note([OFFICE, KITCHEN], LADDERS) == ()
    assert sp.name_mismatch_note([singular], ladders(zones__rooms=[])) == ()
    assert sp.name_mismatch_note([], LADDERS) == ()


def test_the_band_marker_is_a_fraction_of_its_band_and_nothing_outside_it():
    assert sp.band_fraction(1.3, 0.8, 1.8) == pytest.approx(0.5)
    assert sp.band_fraction(0.8, 0.8, 1.8) == 0.0
    assert sp.band_fraction(1.8, 0.8, 1.8) == 1.0
    assert sp.band_fraction(4.0, 0.8, 1.8) is None
    assert sp.band_fraction(None, 0.8, 1.8) is None
    assert sp.band_fraction(1.0, 1.0, 1.0) is None      # a zero-width band


# ------------------------------------------------- the superseded two bands
def test_the_old_two_band_keys_are_gone_from_the_shipped_defaults():
    """A key nothing reads but DEFAULTS still ships looks live: it appears
    in every config, it has plausible numbers in it, and there is no way to
    tell it from one that drives something."""
    from jarvis.assistant_config import DEFAULTS
    presence = DEFAULTS["presence"]
    assert "desk_band_m" not in presence
    assert "room_band_m" not in presence
    assert presence["camera_overrules"] is True     # this one is still live


def test_this_page_is_the_only_reader_of_the_superseded_keys_left():
    """Two models disagreed because two files read two different keys. One
    reader is what makes that impossible, so this counts READERS -- string
    constants in real code -- and lets a comment or a docstring say what
    the key used to be, which is the audit trail."""
    import ast
    import pathlib as _p
    root = _p.Path(sp.__file__).resolve().parents[2]
    hits = []
    for path in sorted(root.glob("jarvis/**/*.py")):
        if path.name == "sensors_page.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "band_m" in node.value and node.value not in docs):
                hits.append("%s:%d" % (path.name, node.lineno))
    assert hits == [], hits


def test_bands_are_carried_across_when_there_is_no_zone_ladder_at_all():
    """His numbers must not be dropped on the floor by a migration."""
    got = ladders(zones__rooms=_GONE,
                  presence__desk_band_m=[2.25, 3.75],
                  presence__room_band_m=[0.75, 2.25])
    zmap = got.for_room("office")
    assert zmap is not None
    assert [(b.name, b.near_m, b.far_m) for b in zmap.bands] == [
        ("in the room", 0.75, 2.25), ("at the desk", 2.25, 3.75)]
    assert got.source("office") == sp.LEGACY_DESK_BAND
    joined = " ".join(got.notes)
    assert "carried across" in joined and "zones.rooms" in joined


def test_where_both_models_exist_zones_rooms_wins_and_the_page_says_which():
    got = ladders(presence__desk_band_m=[0.8, 1.8],
                  presence__room_band_m=[1.8, 4.5])
    assert got.source("office") == sp.OPTION_ROOMS
    assert [b.name for b in got.for_room("office").bands] == ["empty space",
                                                              "at the desk"]
    joined = " ".join(got.notes)
    assert "presence.desk_band_m" in joined and "NOT" in joined
    assert "SAVE removes them" in joined


def test_a_superseded_band_that_cannot_be_carried_across_is_named_not_swapped():
    """MEASURED 2026-09-03: presence.room_band_m = [1.8, 8.0] rendered as
    1.8/4.5 with no note anywhere, and one press of SAVE wrote the 4.5 over
    his 8.0. There is no default left to swap in -- these keys are a
    migration source now -- so it is said out loud and carried across as
    nothing."""
    for junk in ([1.8, 8.0], [80, 180], ["nonsense", 4.0], [], "1.8-4.5"):
        got = ladders(zones__rooms=_GONE, presence__room_band_m=junk,
                      presence__desk_band_m=[2.25, 3.75])
        joined = " ".join(got.notes)
        assert "presence.room_band_m" in joined, junk
        assert "not usable" in joined, junk
        zmap = got.for_room("office")
        assert [b.name for b in zmap.bands] == ["at the desk"], junk


def test_a_superseded_pair_that_was_only_reordered_is_carried_without_a_note():
    """Swapping the ends loses nothing, so it stays silent: a note for every
    generosity would train him to ignore the ones that cost data."""
    got = ladders(zones__rooms=_GONE, presence__desk_band_m=[3.75, 2.25])
    zmap = got.for_room("office")
    assert (zmap.bands[0].near_m, zmap.bands[0].far_m) == (2.25, 3.75)
    assert not [n for n in got.notes if "not usable" in n]


def test_superseded_keys_absent_from_the_config_are_not_a_complaint():
    assert ladders().notes == ()
    assert ladders().legacy is None


def test_two_superseded_bands_that_overlap_are_refused_rather_than_merged():
    got = ladders(zones__rooms=_GONE, presence__desk_band_m=[1.0, 3.0],
                  presence__room_band_m=[0.75, 2.0])
    assert got.legacy is None
    assert "do not describe a ladder" in " ".join(got.notes)


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
    # retry_in_s is the seconds LEFT on the sensor's own deadline. It used
    # to read cooldown_s, which roomsensor._failed has ALREADY doubled by
    # the time status() is asked -- see the 09-03 verifier test below.
    line = sp.fault_line({"url": OFFICE.url, "blocked": "", "paused": True,
                          "fails": 3, "cooldown_s": 60.0, "retry_in_s": 30.0})
    assert "30" in line and "again" in line.lower()


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
    # both still SEE a face; which place that means is the room's
    # camera_zone in zones.rooms, not anything this file decides
    assert asked.sees_a_face is never.sees_a_face is True


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
    assert none.sees_a_face is False


def test_a_named_face_carries_the_name_and_the_score_he_tunes_against():
    view = sp.camera_view({"live": True, "faces": 1, "running": True,
                           "face": {"name": "hunterp", "id_score": 0.712,
                                    "id_ran": True}})
    text, tone = sp.camera_text(view)
    assert "hunterp" in text and "0.71" in text
    assert tone == sp.TONE_OK


# --------------------------------------------------------------- the fusion
def _face(name="hunterp"):
    return sp.camera_view({"live": True, "faces": 1, "running": True,
                           "face": {"name": name, "id_score": 0.71,
                                    "id_ran": True}})


def _office():
    return LADDERS.for_room("office")


def test_the_camera_overrules_the_radar_when_he_lets_it():
    # His words: "camera recognition overrules sensor detection since he can
    # literally see me at my desk". The zone it names is the ROOM'S
    # camera_zone out of the config, not a word this file owns.
    v = sp.fuse(present=True, distance_m=1.0, camera=_face(), zmap=_office(),
                overrules=True)
    assert v.zone == "at the desk" and v.source == "camera"
    assert v.word == "AT THE DESK"


def test_the_camera_does_not_overrule_when_the_toggle_is_off():
    v = sp.fuse(present=True, distance_m=1.0, camera=_face(), zmap=_office(),
                overrules=False)
    assert v.zone == "empty space" and v.source == "radar"
    v = sp.fuse(present=False, distance_m=None, camera=_face(), zmap=_office(),
                overrules=False)
    assert v.zone == sp.ABSENT and "camera" in v.why.lower()


def test_the_camera_is_the_only_leg_with_an_opinion_when_the_radar_has_none():
    # Nothing is being OVERRULED here -- the radar abstained -- so the
    # camera answers even with the toggle off.
    for overrules in (True, False):
        v = sp.fuse(present=None, distance_m=None, camera=_face(),
                    zmap=_office(), overrules=overrules)
        assert v.zone == "at the desk" and v.source == "camera"


def test_a_recognised_face_cannot_name_a_place_in_a_room_with_no_lens():
    """His kitchen entry carries camera_zone "" -- there is no camera in
    the kitchen. The camera rule cannot fire for it however the toggle is
    set, and the verdict falls through to the radar."""
    kitchen = LADDERS.for_room("kitchen")
    assert kitchen.has_camera is False
    v = sp.fuse(present=True, distance_m=3.19, camera=_face(), zmap=kitchen,
                overrules=True)
    assert v.zone == "at the door" and v.source == "radar"
    assert "no camera zone" in v.why


def test_two_legs_with_no_opinion_is_no_opinion_and_not_an_empty_room():
    v = sp.fuse(present=None, distance_m=None, camera=sp.camera_view({}),
                zmap=_office(), overrules=True)
    assert v.zone == sp.NO_OPINION
    assert v.word == "NO OPINION"
    assert "nobody" not in v.word.lower() and "empty" not in v.word.lower()
    assert v.source == ""


def test_presence_with_an_unknown_distance_is_unplaced_and_never_a_guess():
    # The radar answers "someone is in the room" and only coarsely "how
    # far". With no range at all the honest answer names no band.
    v = sp.fuse(present=True, distance_m=None, camera=sp.camera_view({}),
                zmap=_office(), overrules=True)
    assert v.zone == sp.UNPLACED
    # and a range in a declared gap is the same answer, with a reason that
    # says which
    v = sp.fuse(present=True, distance_m=5.8, camera=sp.camera_view({}),
                zmap=_office(), overrules=True)
    assert v.zone == sp.UNPLACED and "gap" in v.why


def test_a_target_inside_a_band_is_that_band_without_any_camera():
    v = sp.fuse(present=True, distance_m=3.13, camera=sp.camera_view({}),
                zmap=_office(), overrules=True)
    assert v.zone == "at the desk" and v.source == "radar"
    assert "at the desk" in v.why


# ------------------------------------------------------------ the whole row
def test_the_page_lists_every_configured_room_in_config_order():
    rows = sp.page_rows([reading(name="office"),
                         reading(name="kitchen", label="the kitchen")],
                        camera_status={}, ladders=LADDERS, overrules=True,
                        camera_room="office")
    assert [r.name for r in rows] == ["office", "kitchen"]
    # the camera lives in ONE room; the other says so
    assert "no camera" in rows[1].camera_text.lower()


def test_a_row_for_an_unreachable_sensor_still_renders_every_field():
    row = sp.page_rows([reading(present=None, distance_m=None, rtt_ms=None,
                                status={"url": OFFICE.url, "blocked": "",
                                        "paused": True, "fails": 3,
                                        "cooldown_s": 30.0})],
                       camera_status={}, ladders=LADDERS, overrules=True,
                       camera_room="office")[0]
    assert row.presence_word == "NO OPINION"
    assert row.distance_text == sp.DASH and row.rtt_text == sp.DASH
    assert row.fault and row.verdict.zone == sp.NO_OPINION


def test_a_present_row_prints_metres_and_milliseconds_the_way_he_reads_them():
    row = sp.page_rows([reading(present=True, distance_m=1.42, rtt_ms=62.4)],
                       camera_status={}, ladders=LADDERS, overrules=True,
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
def _edits(office=(("empty space", "0.75", "2.25"),
                   ("at the desk", "2.25", "3.75"))):
    return [sp.BandEdit("office", name, lo, hi) for name, lo, hi in office]


def test_saving_writes_the_two_dotted_keys_and_nothing_else():
    edits, err = sp.band_edits(_edits(), LADDERS, True)
    assert err == ""
    assert set(edits) == {sp.OPTION_ROOMS, sp.OPTION_CAMERA_OVERRULES}
    assert edits[sp.OPTION_CAMERA_OVERRULES] is True
    office = [r for r in edits[sp.OPTION_ROOMS] if r["name"] == "office"][0]
    assert office["bands"] == [
        {"name": "empty space", "near_m": 0.75, "far_m": 2.25},
        {"name": "at the desk", "near_m": 2.25, "far_m": 3.75}]


def test_the_write_is_a_merge_so_a_room_this_page_never_showed_survives():
    """AssistantConfig REPLACES a list rather than merging it, so a save
    built from the widgets would delete every room the page was not
    showing -- his kitchen, tonight, since presence.rooms is empty and the
    fabric polls one room called "room"."""
    edits, err = sp.band_edits(_edits(), LADDERS, True)
    assert err == ""
    kitchen = [r for r in edits[sp.OPTION_ROOMS] if r["name"] == "kitchen"]
    assert kitchen == [KITCHEN_ROOM]          # byte for byte, camera_zone ""


def test_saving_keeps_the_camera_zone_a_band_edit_has_no_business_touching():
    edits, _err = sp.band_edits(_edits(), LADDERS, True)
    rooms = {r["name"]: r for r in edits[sp.OPTION_ROOMS]}
    assert rooms["office"]["camera_zone"] == "at the desk"
    assert rooms["kitchen"]["camera_zone"] == ""      # no lens; stays no lens
    assert rooms["office"]["enabled"] is True


def test_a_room_with_only_carried_across_bands_is_APPENDED_by_the_save():
    """That is what completes the migration: the numbers stop living in the
    superseded keys and start living in zones.rooms."""
    carried = ladders(zones__rooms=_GONE, presence__desk_band_m=[2.25, 3.75],
                      presence__room_band_m=[0.75, 2.25])
    edits, err = sp.band_edits(
        [sp.BandEdit("office", "in the room", "0.75", "2.25"),
         sp.BandEdit("office", "at the desk", "2.25", "3.75")], carried, True)
    assert err == ""
    assert [r["name"] for r in edits[sp.OPTION_ROOMS]] == ["office"]
    assert edits[sp.OPTION_ROOMS][0]["bands"][1]["near_m"] == 2.25


def test_a_band_edit_that_is_not_a_number_is_refused_with_a_reason():
    edits, err = sp.band_edits(
        [sp.BandEdit("office", "at the desk", "two point two five", "3.75")],
        LADDERS, True)
    assert edits == {}
    assert "number" in err.lower() and "at the desk" in err


def test_a_band_typed_backwards_is_refused_rather_than_silently_swapped():
    # Carrying the old keys across repairs a swap; a person typing gets
    # told. Two different situations, deliberately two different answers.
    edits, err = sp.band_edits(
        [sp.BandEdit("office", "at the desk", "3.75", "2.25")], LADDERS, True)
    assert edits == {} and "at the desk" in err
    assert "near end" in err


def test_a_band_beyond_what_the_radar_can_see_is_refused():
    edits, err = sp.band_edits(
        [sp.BandEdit("office", "at the desk", "2.25", "9.0")], LADDERS, True)
    assert edits == {}
    assert "6" in err            # gates 0..8 at 0.75 m each
    assert sp.MAX_BAND_M == 6.0


def test_a_ladder_that_would_not_parse_is_refused_HERE_not_by_the_log_later():
    """The page must not save a geometry jarvis/zones.py will refuse: that
    is how a config reaches the state where the page shows one thing and
    the record says another. The judge is the zone model itself."""
    edits, err = sp.band_edits(
        [sp.BandEdit("office", "empty space", "0.75", "3.00"),
         sp.BandEdit("office", "at the desk", "2.25", "3.75")], LADDERS, True)
    assert edits == {}
    assert "overlap" in err and "office" in err


def test_a_zones_rooms_of_the_wrong_shape_refuses_the_save_by_name():
    """The save has to refuse, not proceed: with the section poisoned the
    page is showing no bands, and a merge from an empty page would replace
    whatever is in his file with an empty list."""
    got = ladders(zones__rooms="office")
    assert got.blocked and "zones.rooms" in got.blocked
    edits, err = sp.band_edits(_edits(), got, True)
    assert edits == {} and "zones.rooms" in err
    # ... and the same for every other way the section can be wrong
    for section in ("nonsense", 7, [], {"enabled": "no"},
                    {"dwell_s": "three", "rooms": [copy.deepcopy(OFFICE_ROOM)]}):
        got = ladders(zones=section)
        assert got.blocked, section
        assert sp.band_edits(_edits(), got, True)[0] == {}, section


def test_an_empty_page_saves_the_toggle_and_never_an_empty_band_list():
    """A list REPLACES rather than merges in AssistantConfig, so a save
    with no band rows on screen must not write zones.rooms at all."""
    edits, err = sp.band_edits([], LADDERS, False)
    assert err == "" and set(edits) == {sp.OPTION_CAMERA_OVERRULES}
    assert edits[sp.OPTION_CAMERA_OVERRULES] is False


def test_the_bands_carry_the_hardware_notes_he_would_otherwise_learn_the_hard_way():
    from jarvis.zones import Band, ZoneMap
    notes = sp.band_notes(ZoneMap("office", (Band("under the desk", 0.5, 1.4),)))
    joined = " ".join(notes).lower()
    assert "0.75" in joined          # nothing at all is detected inside it
    assert "1.5" in joined           # no STILL target inside it
    assert "under the desk" in joined            # WHICH band, by name
    assert sp.band_notes(LADDERS.for_room("office")) == ()
    assert sp.band_notes(None) == ()


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
    assert row.verdict.zone == sp.NO_OPINION
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


# =====================================================================
# The 2026-09-03 verifier pass. Every test below reproduces a finding
# that was MEASURED against this branch before it was repaired; the
# comment on each one is the failure it saw.
# =====================================================================

# ------------------------------------------------- MAJOR: a failed save
class _Services:
    """A services stand-in with the shape jarvis/app.py actually has:
    ``set_option`` RETURNS False on failure and does not raise."""

    def __init__(self, answer=True, raises=None):
        self.answer, self.raises = answer, raises
        self.calls = []

    def set_option(self, key, value):
        self.calls.append((key, value))
        if self.raises is not None:
            raise self.raises
        return self.answer


EDITS = {sp.OPTION_ROOMS: [copy.deepcopy(OFFICE_ROOM)],
         sp.OPTION_CAMERA_OVERRULES: True}


def test_a_write_that_returns_false_is_never_reported_as_saved():
    """THE WORST THING ON THE BRANCH. jarvis/app.py set_option catches
    internally and RETURNS False -- it does not raise -- so the page's
    old `try: fn(...) except: log` on a daemon thread could not see a
    failure at all, and set SAVED_NOTE unconditionally on the next line.
    He restarts expecting new bands, gets the old ones, and nothing on
    screen said so. Nothing in this codebase may report a failure as a
    success."""
    svc = _Services(answer=False)
    failed = sp.write_options(svc.set_option, EDITS)
    assert set(failed) == set(EDITS)
    text, tone = sp.save_note(failed)
    assert text != sp.SAVED_NOTE
    assert "saved" not in text.lower() or "not saved" in text.lower()
    assert tone == sp.TONE_ERR


def test_a_write_that_raises_is_never_reported_as_saved_either():
    svc = _Services(raises=OSError("read-only file system"))
    failed = sp.write_options(svc.set_option, EDITS)
    assert set(failed) == set(EDITS)
    assert sp.save_note(failed)[0] != sp.SAVED_NOTE


def test_the_failure_note_names_the_key_that_would_not_write():
    text, _tone = sp.save_note((sp.OPTION_ROOMS,))
    assert sp.OPTION_ROOMS in text
    assert sp.OPTION_CAMERA_OVERRULES not in text


def test_one_key_failing_is_still_a_failure_and_not_a_partial_success():
    def half(key, value):
        return key != sp.OPTION_ROOMS
    failed = sp.write_options(half, EDITS)
    assert failed == (sp.OPTION_ROOMS,)
    assert sp.save_note(failed)[0] != sp.SAVED_NOTE


class _Store:
    """A services stand-in over a real dotted dict, so a write and the read
    that follows it are the same file."""

    def __init__(self, data=None):
        self.data = data if data is not None else {}
        self.cleared = []

    def get_option(self, key, default=None):
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def set_option(self, key, value):
        node = self.data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        return True

    def unset_option(self, key):
        node = self.data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                return False
        if parts[-1] not in node:
            return False
        node.pop(parts[-1])
        self.cleared.append(key)
        return True


def test_a_save_retires_the_superseded_keys_instead_of_leaving_them_looking_live():
    """A key nothing reads is indistinguishable from one that drives
    something, and that is how the two zone models disagreed for a day."""
    svc = _Store({"presence": {"desk_band_m": [2.25, 3.75],
                               "room_band_m": [0.75, 2.25]}})
    notes = sp.retire_legacy(svc.get_option, svc.unset_option)
    assert notes == ()
    assert sorted(svc.cleared) == [sp.LEGACY_DESK_BAND, sp.LEGACY_ROOM_BAND]
    assert svc.get_option(sp.LEGACY_DESK_BAND) is None


def test_retiring_keys_that_are_not_there_writes_nothing_and_says_nothing():
    svc = _Store({})
    assert sp.retire_legacy(svc.get_option, svc.unset_option) == ()
    assert svc.cleared == []


def test_keys_that_cannot_be_removed_are_SAID_rather_than_left_in_silence():
    # An older app, or a services stand-in with no unset_option. Leaving a
    # live-looking lie in his file quietly is the one outcome this may not
    # have.
    svc = _Store({"presence": {"desk_band_m": [2.25, 3.75]}})
    notes = sp.retire_legacy(svc.get_option, None)
    assert len(notes) == 1
    assert sp.LEGACY_DESK_BAND in notes[0] and "by hand" in notes[0]
    # and a clear that silently does not take is named too
    stuck = _Store({"presence": {"desk_band_m": [2.25, 3.75]}})
    notes = sp.retire_legacy(stuck.get_option, lambda key: False)
    assert len(notes) == 1 and sp.LEGACY_DESK_BAND in notes[0]


def test_a_clear_that_raises_is_a_note_and_never_a_crash_on_the_tk_thread():
    svc = _Store({"presence": {"room_band_m": [0.75, 2.25]}})

    def boom(key):
        raise OSError("read-only file system")

    notes = sp.retire_legacy(svc.get_option, boom)
    assert len(notes) == 1 and sp.LEGACY_ROOM_BAND in notes[0]


def test_a_save_and_the_read_after_it_agree_end_to_end():
    """The migration as one round trip: a config with only the superseded
    keys, the bands carried across, the save, and then the SAME reader
    finding them in zones.rooms with the old keys gone."""
    svc = _Store({"presence": {"desk_band_m": [2.25, 3.75],
                               "room_band_m": [0.75, 2.25]}})
    before = sp.read_ladders(svc.get_option)
    assert before.source("office") == sp.LEGACY_DESK_BAND
    edits, err = sp.band_edits(
        [sp.BandEdit("office", b.name, b.near_m, b.far_m)
         for b in before.for_room("office").bands], before, True)
    assert err == ""
    assert sp.write_options(svc.set_option, edits) == ()
    assert sp.retire_legacy(svc.get_option, svc.unset_option) == ()
    after = sp.read_ladders(svc.get_option)
    assert after.source("office") == sp.OPTION_ROOMS
    assert [(b.name, b.near_m, b.far_m) for b in after.for_room("office").bands] \
        == [("in the room", 0.75, 2.25), ("at the desk", 2.25, 3.75)]
    assert after.notes == ()                 # nothing left to complain about
    assert svc.get_option(sp.LEGACY_DESK_BAND) is None


def test_a_write_that_says_nothing_at_all_still_counts_as_saved():
    """A services object whose set_option returns None (the drawer's own
    stand-ins do) must not be read as a failure: only an explicit False
    is one, because that is the value jarvis/app.py returns."""
    assert sp.write_options(lambda k, v: None, EDITS) == ()
    assert sp.save_note(())[0] == sp.SAVED_NOTE


def test_the_save_is_written_inline_and_not_handed_to_a_daemon_thread():
    """Two reasons the thread had to go: its result could not reach the
    note, and a daemon thread means SAVE-then-quit can be killed
    mid-write. Three set_option calls are an in-memory edit plus one
    atomic os.replace."""
    import re as _re
    src = open(sp.__file__, encoding="utf-8").read()
    body = src[src.index("    def save(self)"):src.index("    def band_values")]
    body = _re.sub(r"(?m)#.*$", "", body)
    assert "Thread(" not in body and "daemon" not in body
    assert "write_options(" in body and "save_note(" in body


# -------------------------------------------- MAJOR: the leaked threads
class _Blocking:
    """A transport that parks inside the request until the test lets it
    go -- the widest form of the race, which is an in-flight HTTP read."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, url, timeout):
        self.calls += 1
        self.entered.set()
        self.release.wait(5.0)
        return '{"value": true, "state": "ON"}'


def _live_poll_threads() -> int:
    return sum(1 for t in threading.enumerate()
               if t.name == sp.POLL_THREAD_NAME and t.is_alive())


class _Slow:
    """A transport that takes 0.3 s, the way an ESP32 mid-answer does. The
    verifier's probe used 0.6 s against a real worst round trip of 1186 ms;
    the point is only that stop() lands while a request is in flight."""

    def __init__(self, delay=0.3):
        self.delay, self.calls = delay, 0
        self.entered = threading.Event()

    def __call__(self, url, timeout):
        self.calls += 1
        self.entered.set()
        time.sleep(self.delay)
        return '{"value": true, "state": "ON"}'


def _settle(before: int, timeout: float = 5.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline and _live_poll_threads() > before:
        time.sleep(0.02)
    return _live_poll_threads()


def test_closing_and_reopening_the_page_leaves_one_poll_loop_not_four():
    """MEASURED before the repair: four open/close cycles gave live
    "sensors-page" threads 1, 2, 3, 4 -- four loops all hitting one ESP32
    at 1 Hz, four presence + four distance requests a second against a
    device whose p90 is 154 ms, and four repaints a second posted to Tk.

    The cause: stop() dropped the Thread handle without joining and
    start() then CLEARED the SHARED stop flag out from under the thread
    still inside get(), so that thread never saw the stop and looped
    forever. The race window is the whole in-flight HTTP read -- widest
    for an unreachable sensor, which is exactly the state the page exists
    to be opened in. A tab is far easier to flip in and out of than F9,
    so this only gets worse the moment the strip ships.

    Nothing is settled between the toggles ON PURPOSE: letting each
    orphan notice the flag and exit is precisely what does not happen
    when he flips the tab quickly, and the bug would not reproduce.

    What this one asserts is CONVERGENCE and silence -- one loop left
    over, and no request at all once the page is shut. The stronger
    property, that there is never a second loop even for an instant, is
    test_flipping_the_tab_never_puts_two_poll_loops_on_one_sensor: the
    first repair passed this test while running four loops at once.
    """
    before = _live_poll_threads()
    slow = _Slow(delay=0.4)
    poller = sp.SensorPoller([OFFICE], get=slow)
    try:
        for _toggle in range(3):
            slow.entered.clear()
            poller.start(lambda rows: None, post=lambda fn: None)
            assert slow.entered.wait(5.0), "the poll thread never started"
            assert _live_poll_threads() <= before + 1, (
                "a second poll loop started beside the one still in a request")
            poller.stop()
        # ... and now he leaves it open, which is the state that costs him
        slow.entered.clear()
        poller.start(lambda rows: None, post=lambda fn: None)
        assert slow.entered.wait(5.0)
        assert _settle(before + 1, timeout=4.0) == before + 1, (
            "more than one poll loop is hitting the room sensor")
    finally:
        poller.stop()
    assert _settle(before) == before
    settled = slow.calls
    time.sleep(1.2)                       # more than a poll interval
    assert slow.calls == settled, "a closed page kept hitting the ESP32"


def test_a_stopped_poller_never_polls_again_even_mid_request():
    """The orphan must not come back to life when the page is reopened:
    it gets its own stop flag, so a later start() cannot clear it."""
    gate = _Blocking()
    poller = sp.SensorPoller([OFFICE], get=gate)
    poller.start(lambda rows: None, post=lambda fn: None)
    assert gate.entered.wait(5.0)
    poller.stop()
    poller.start(lambda rows: None, post=lambda fn: None)   # reopened
    poller.stop()
    gate.release.set()
    deadline = time.time() + 5.0
    while time.time() < deadline and _live_poll_threads():
        time.sleep(0.05)
    settled = gate.calls
    time.sleep(0.6)
    assert gate.calls == settled, "a stopped poller polled again"


def test_stop_reports_whether_the_thread_actually_ended():
    poller = sp.SensorPoller([OFFICE], get=lambda u, t: '{"value": false}')
    poller.start(lambda rows: None, post=lambda fn: None)
    assert poller.stop() is True


# ------------------------------ MINOR: the retry countdown was 2x and static
def test_the_retry_countdown_is_the_wait_that_is_left_not_the_next_one():
    """MEASURED with a frozen clock: at the pass where the breaker opened
    the page said 'trying again in 60 s' while the real wait was 30 s --
    and jarvis.log said 30 s for the same event. roomsensor._failed sets
    _skip_until from the CURRENT cooldown and THEN doubles it, so
    status()['cooldown_s'] is the NEXT one. The page reads the sensor's
    own deadline instead, so the number is right AND counts down."""
    line = sp.fault_line({"url": OFFICE.url, "paused": True, "fails": 3,
                          "cooldown_s": 60.0, "retry_in_s": 30.0})
    assert "30" in line and "60" not in line
    # and it MOVES: the same breaker one repaint later
    later = sp.fault_line({"url": OFFICE.url, "paused": True, "fails": 3,
                           "cooldown_s": 60.0, "retry_in_s": 11.0})
    assert "11" in later and "30" not in later


def test_a_status_with_no_deadline_drops_the_number_rather_than_guessing():
    line = sp.fault_line({"url": OFFICE.url, "paused": True, "fails": 3,
                          "cooldown_s": 60.0})
    assert "60" not in line and "again" in line.lower()


def test_the_sensor_publishes_the_wait_that_is_left_and_the_log_agrees():
    """The fix at the source: roomsensor.status() now carries the seconds
    LEFT, so the page and the one warning line in jarvis.log can no longer
    disagree by 2x about the same sensor."""
    from jarvis import roomsensor
    clock = {"t": 100.0}

    def boom(url, timeout):
        raise OSError("down")

    s = roomsensor.RoomSensor("http://192.0.2.10", get=boom,
                              now=lambda: clock["t"], fail_after=1,
                              cooldown_s=30.0)
    s.read()
    assert s.status()["retry_in_s"] == pytest.approx(30.0)
    assert s.status()["cooldown_s"] == 60.0          # the NEXT one, unchanged
    clock["t"] += 20.0
    assert s.status()["retry_in_s"] == pytest.approx(10.0)   # it counts down
    clock["t"] += 20.0
    assert s.status()["retry_in_s"] == 0.0


# ------------------ MINOR: an out-of-range band was swapped in silence
# The four tests that were here rode on the page's own two-band model and
# on its silent fall back to a built-in default pair. Both are gone: the
# bands are zones.rooms now, and the superseded keys are a migration source
# with no default to fall back TO. What the findings were about -- a value
# in his file that is thrown away without a word, and a SAVE that then
# writes the substitute over it -- is held by the migration tests up in
# "the superseded two bands", which assert the same thing against the model
# that survived.


# ------------- MINOR: the empty state named only one of the two switches
def test_the_empty_page_names_the_master_switch_when_that_is_what_is_off():
    """roomfabric.room_specs() returns [] at its FIRST line when
    presence.room_sensor_enabled is false, whatever the URL says. The old
    line always blamed presence.room_sensor_url -- and after 00d5b7e
    ('URL set, master switch still off') that is the likely next state,
    so the page would point at the one key that is already right."""
    cfg = {"presence.room_sensor_enabled": False,
           "presence.room_sensor_url": "http://192.0.2.10"}
    line = sp.empty_state_line(lambda k, d=None: cfg.get(k, d))
    assert "presence.room_sensor_enabled" in line
    assert "presence.room_sensor_url" not in line


def test_the_empty_page_names_the_address_keys_when_the_switch_is_on():
    cfg = {"presence.room_sensor_enabled": True}
    line = sp.empty_state_line(lambda k, d=None: cfg.get(k, d))
    assert "presence.room_sensor_url" in line
    assert "presence.room_sensor_enabled" not in line


def test_the_empty_page_says_the_fix_is_a_hand_edit_and_a_restart():
    """Nothing on this page can set either switch, so telling him the key
    without telling him where it lives is half an answer."""
    for enabled in (True, False):
        line = sp.empty_state_line(lambda k, d=None, e=enabled: (
            e if k == "presence.room_sensor_enabled" else d))
        assert "assistant.json" in line and "restart" in line.lower()


# ------------------------------------------ MINOR: no staleness signal
def test_the_page_says_how_old_the_numbers_are():
    """A diagnostics page he reads while walking the room had no way to
    tell him the numbers had stopped moving: Reading.at was captured on
    every poll and never rendered."""
    assert sp.age_text(1000.0, 1000.0) == "updated just now"
    assert "2 s" in sp.age_text(1002.4, 1000.0)
    assert "1 min" in sp.age_text(1000.0 + 65, 1000.0)
    assert sp.age_text(1000.0, 0.0) == ""          # nothing has landed yet
    assert sp.age_text(1000.0, 2000.0) == "updated just now"   # clock skew


def test_two_failed_hops_to_the_tk_thread_stop_the_loop_loudly():
    """Reproduced by driving the page with root.update() instead of
    mainloop: every self.after() raised 'main thread is not in main loop',
    the page kept polling, and nothing on screen or at INFO said why. The
    rows cannot be repainted from a thread that cannot reach Tk, so the
    honest answer is to stop and say so."""
    posts = []

    def dead_post(fn):
        posts.append(fn)
        raise RuntimeError("main thread is not in main loop")

    poller = sp.SensorPoller([OFFICE], get=lambda u, t: '{"value": true}')
    poller.start(lambda rows: None, post=dead_post, interval_s=0.2)
    deadline = time.time() + 5.0
    while time.time() < deadline and _live_poll_threads():
        time.sleep(0.05)
    assert not _live_poll_threads(), "the loop kept polling into a dead Tk"
    assert len(posts) == sp.POST_FAILS_MAX
    assert poller.post_failed is True
    poller.stop()


# ------------------------ NIT: the camera was pinned to whichever room is primary
def test_the_camera_belongs_to_the_room_a_key_says_it_does():
    """roomfabric forces `primary` onto the FIRST entry when none is
    marked, so a presence.rooms list that happened to lead with the
    kitchen would have shown the office camera's recognised face against
    the kitchen row. camera.room is the key that ties the lens to a room;
    primary is the default when he has not said."""
    assert sp.camera_room_name(lambda k, d=None: {"camera.room": "kitchen"}
                               .get(k, d), [OFFICE, KITCHEN]) == "kitchen"
    assert sp.camera_room_name(lambda k, d=None: d, [OFFICE, KITCHEN]) == "office"
    # a name that is not a configured room is ignored rather than silently
    # giving every row "no camera in this room"
    assert sp.camera_room_name(lambda k, d=None: {"camera.room": "garage"}
                               .get(k, d), [OFFICE, KITCHEN]) == "office"


# ------------------------------- NIT: fault_line raised on a non-integer
def test_a_status_dict_with_junk_in_it_cannot_blank_the_whole_page():
    """page_rows calls fault_line inside the repaint, so a ValueError
    there takes out every row, not one field."""
    line = sp.fault_line({"url": OFFICE.url, "fails": "x"})
    assert line and "opinion" in line.lower()
    assert sp.fault_line({"url": OFFICE.url, "fails": 2.9})


# ------------- MAJOR (round 2): the leak was terminal, but not singular
class _Counting:
    """A slow transport that counts how many poll threads are inside a
    request AT THE SAME TIME -- the number that matters to the ESP32,
    which does not care whether the extra caller will eventually die."""

    def __init__(self, delay: float = 0.4):
        self.delay = float(delay)
        self.calls = 0
        self.inside = 0
        self.peak = 0
        self.entered = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, url, timeout):
        with self._lock:
            self.calls += 1
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        self.entered.set()
        time.sleep(self.delay)
        with self._lock:
            self.inside -= 1
        return '{"value": true, "state": "ON"}'


class _PeakThreads:
    """Samples the live poll-thread count from the side, because the
    interesting moment is DURING the flipping, not after it settles."""

    def __init__(self, hz: float = 400.0):
        self.peak = _live_poll_threads()
        self._gap = 1.0 / float(hz)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="poll-thread-sampler")

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _live_poll_threads())
            self._stop.wait(self._gap)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(2.0)
        self.peak = max(self.peak, _live_poll_threads())
        return False


def test_flipping_the_tab_never_puts_two_poll_loops_on_one_sensor():
    """MEASURED on the first repair, which is why this test exists: the
    per-run stop flag made the orphans TERMINAL but not absent. stop()
    released the Thread handle under the lock and joined afterwards, so
    start()'s "is a thread already alive" guard read None while the old
    thread was still inside get() and built a second loop beside it --
    four concurrent "sensors-page" threads while flipping, six with a
    request that never returns.

    The assertion is CONCURRENCY, not convergence: one thread in the
    transport at a time, and one live poll thread at a time, sampled
    while the flipping happens rather than after it stops.
    """
    before = _live_poll_threads()
    tap = _Counting(delay=0.4)
    poller = sp.SensorPoller([OFFICE], get=tap)
    try:
        with _PeakThreads() as peak:
            for _flip in range(5):
                tap.entered.clear()
                poller.start(lambda rows: None, post=lambda fn: None)
                assert tap.entered.wait(5.0), "the poll thread never started"
                poller.stop()             # nothing settles: that is the point
            tap.entered.clear()
            poller.start(lambda rows: None, post=lambda fn: None)
            assert tap.entered.wait(5.0)
            time.sleep(0.6)               # a pass and a bit, still flipping-fresh
        assert tap.peak == 1, (
            "%d poll threads were inside the transport at once" % tap.peak)
        assert peak.peak <= before + 1, (
            "%d live poll threads, only %d may exist"
            % (peak.peak, before + 1))
    finally:
        poller.stop()
    assert _settle(before) == before


class _FailingGate:
    """Parks inside the request and then fails -- the state the office
    radar is in today, and the one that arms the breaker."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = True
        self.calls = 0

    def __call__(self, url, timeout):
        self.calls += 1
        self.entered.set()
        if self.block:
            self.release.wait(5.0)
        raise OSError("unreachable")


def test_a_flip_during_an_outage_trips_the_breaker_once_not_twice():
    """MEASURED on the first repair: ``_sensors`` is keyed by room name and
    outlives a run, so an orphan and its replacement held the SAME
    RoomSensor. One round of failures then doubled the cooldown twice --
    30 s -> 120 s where the design says 60 s -- and roomsensor._failed()
    takes no lock, so the "N in a row" the fault line quotes was inflated
    too. The office radar is unplugged today, which is exactly the state
    that reaches this.
    """
    gate = _FailingGate()
    poller = sp.SensorPoller([OFFICE], get=gate)
    gate.block = False
    row = None
    for _pass in range(2):                # two of the three it takes
        row = poller.poll_once()[0]
    status = row.status
    assert status["fails"] == 2, status
    assert status["cooldown_s"] == 30.0, status
    assert gate.calls == 2

    gate.block = True
    gate.entered.clear()
    try:
        poller.start(lambda rows: None, post=lambda fn: None)
        assert gate.entered.wait(5.0), "the poll thread never started"
        poller.stop()                     # he flips to CHAT...
        poller.start(lambda rows: None, post=lambda fn: None)   # ...and back
        time.sleep(0.3)                   # long enough for a second loop to enter
        gate.release.set()
        time.sleep(0.5)
    finally:
        poller.stop()
    _settle(0)
    status = poller.poll_once()[0].status
    assert status["cooldown_s"] == 60.0, (
        "one round of failures doubled the breaker to %.0f s" %
        status["cooldown_s"])


# ---------- NIT (round 2): the bands were repaired, this key was not
def test_a_camera_overrules_value_the_page_cannot_use_is_named_not_coerced():
    """The same silent coercion the bands had, left on the one other key
    this page WRITES. `return True if value is None else bool(value)` makes
    the string "false" True -- bool("false") is True -- so a hand edit of
    `"presence.camera_overrules": "false"` rendered the toggle ON, showed
    him the opposite of what he had typed, and one press of SAVE wrote
    `true` over it. The bands got a second return value and a named note
    for exactly this; this key now gets the same.
    """
    assert bool("false") is True          # the coercion, stated
    assert sp.read_overrules_noted(lambda k, d=None: {}.get(k, d)) == (True, "")
    for value in (True, False):
        assert sp.read_overrules_noted(
            lambda k, d=None, v=value: {sp.OPTION_CAMERA_OVERRULES: v}.get(k, d)
        ) == (value, "")
    for junk in ("false", "off", 0, 1, [], "yes"):
        got, note = sp.read_overrules_noted(
            lambda k, d=None, v=junk: {sp.OPTION_CAMERA_OVERRULES: v}.get(k, d))
        assert got is True, junk
        assert note and "SAVE" in note and repr(junk)[:20] in note, junk
    # and it reaches the screen the way a bad band does
    assert sp.read_overrules(lambda k, d=None:
                             {sp.OPTION_CAMERA_OVERRULES: "false"}.get(k, d)) is True


def test_a_broken_overrules_note_rides_with_the_band_notes():
    """The page's own notes list is what carries it to the screen: the
    band repair put its sentence in config_notes, and this one has to land
    in the same place or nothing renders it."""
    src = open(sp.__file__, encoding="utf-8").read()
    init = src[src.index("    def __init__(self, host, services=None"):
               src.index("    # ------------------------------------------------------------- config")]
    assert "read_overrules_noted(" in init
    assert "self.config_notes" in init.split("read_overrules_noted(")[1]


def test_a_stop_lands_inside_one_request_not_at_the_end_of_the_pass():
    """What an orphan still costs, as a number. The run check used to be
    once per pass, so a stop that landed inside the office presence read
    was still followed by the office DISTANCE read and both of the
    kitchen's -- four requests to devices nobody was going to paint, one
    of which is unplugged. MEASURED with the transport parked inside the
    first read: 4 requests after the stop before this, 1 after.
    """
    gate = _Blocking()
    poller = sp.SensorPoller([OFFICE, KITCHEN], get=gate)
    try:
        poller.start(lambda rows: None, post=lambda fn: None)
        assert gate.entered.wait(5.0), "the poll thread never started"
        assert gate.calls == 1, "the parked request is the office presence read"
        poller.stop()                     # he flips to CHAT mid-request
        gate.release.set()
    finally:
        _settle(0)
        poller.stop()
    time.sleep(0.4)
    assert gate.calls == 1, (
        "%d requests went out after the stop" % gate.calls)


def test_a_reopen_inside_a_dead_hop_does_not_label_the_new_run_stopped():
    """One level up from "the loop stops when it cannot reach Tk".

    ``post_failed`` is what puts "the poll loop stopped — reopen the page"
    on the age line (SensorsPage._tick), so it belongs to the RUN whose hop
    died. Raising it unconditionally while disarming only the matching run
    meant a reopen that landed INSIDE the failing hop -- stop() then
    start(), which is one flip of the tab -- left the flag set over a run
    that was polling and repainting perfectly well: the page read STOPPED
    at a live surface.

    Deterministic, not timed: the reopen happens on a helper thread that
    is JOINED before the hop raises, so the interleaving is not a race the
    test hopes for.
    """
    painted, hops = [], []

    def reopen():
        poller.stop(timeout_s=0.0)        # he flips to CHAT...
        poller.start(lambda rows: painted.append(rows),
                     post=lambda fn: fn(), interval_s=0.2)   # ...and back

    def dying_post(fn):
        hops.append(1)
        if len(hops) == sp.POST_FAILS_MAX:
            hand = threading.Thread(target=reopen, name="reopen")
            hand.start()
            hand.join(5.0)
            assert not hand.is_alive(), "the reopen never finished"
        raise RuntimeError("main thread is not in main loop")

    poller = sp.SensorPoller([OFFICE], get=lambda u, t: '{"value": true}')
    try:
        poller.start(lambda rows: None, post=dying_post, interval_s=0.2)
        deadline = time.time() + 5.0
        while time.time() < deadline and not painted:
            time.sleep(0.02)
        assert painted, "the reopened run never repainted"
        assert poller.running, "the old run's failure disarmed the new one"
        assert poller.post_failed is False, (
            "the age line would read 'the poll loop stopped' at a surface "
            "that is polling")
    finally:
        poller.stop()
    assert _settle(0) == 0
