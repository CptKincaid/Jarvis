"""The SENSOR SETUP sheet: everything above the widget, and then the widget.

Hunter, 2026-09-05: "i know i want a more ease of access on configuring
sensors through the UI and adding news ones that way". Until now the SENSORS
page could edit the distance BANDS and nothing else; the device itself --
which network it joins, what address it takes, how far it must see, which
pins the radar is wired to -- was a hand-edited JSON file.

THE THING THIS FILE EXISTS TO PIN. That JSON file holds his Wi-Fi PSK and
the device's OTA password, and this sheet renders to PNG on a photo rig
whose frames get looked at. So:

* no secret is ever RENDERED -- asserted over the whole widget tree, every
  label's text and every entry's contents, after a full open/edit/save;
* no secret is ever LOGGED -- asserted over caplog at DEBUG, which is the
  level a stray "%r" of the form would land at;
* an EMPTY secret box means "leave the stored one alone", asserted against
  the bytes on disk;
* the two secret boxes are ``show="•"``, created empty every time the sheet
  opens, never pre-filled and never bound to config -- the ``_knightfall_row``
  pattern jarvis/ui/views.py already states in its own words.

Every name, address, SSID and password here is INVENTED and every test
writes into ``tmp_path``. The Tk half runs only when
``JARVIS_UI_TEST_DISPLAY`` names a private Xvfb and refuses his desktop
displays outright, exactly as tests/test_ui_layout_rules.py does.
"""
import json
import logging
import os
from pathlib import Path

import pytest

from jarvis import sensorprofile as sp
from jarvis.ui import sensor_setup as ss

PSK = "invented-psk-not-his-9317"
OTA = "invented-ota-not-his-4482"
SSID = "PRETEND-NET-5G"
SAT_PW = "invented-satellite-pw-6620"
IP = "192.0.2.10"
NEW_IP = "192.0.2.30"
FORBIDDEN_DISPLAYS = (":0", ":1")
FD_SETSIZE = 1024


def _write_raw(directory: Path, room: str, **over) -> Path:
    data = {"room": room, "ssid": SSID, "password": PSK, "ota_password": OTA,
            "ip": IP, "dhcp": False, "mac": "", "gateway": "192.0.2.1",
            "subnet": "255.255.255.0", "board": "esp32dev",
            "rx_pin": "GPIO16", "tx_pin": "GPIO17", "preset": "desk",
            "mount_note": "across the room", "nearest_m": 2.0, "range_m": 3.5,
            "still": True, "timeout_s": 10, "serial_port": "/dev/ttyUSB0",
            "flashed": True, "notes": ""}
    data.update(over)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("%s.json" % room)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


# ============================================== what the sheet starts from
def test_the_form_starts_from_the_profile_with_both_secrets_empty(tmp_path):
    _write_raw(tmp_path, "office")
    form = ss.form_for("office", tmp_path)
    assert form["ip"] == IP and form["ssid"] == SSID
    assert form["nearest_m"] == "2.00" and form["range_m"] == "3.50"
    assert form["timeout_s"] == "10" and form["still"] is True
    # THE TWO SECRET BOXES START EMPTY, always. Not the stored value, not a
    # row of bullets its LENGTH -- a masked string 14 wide tells you the PSK
    # is 14 characters.
    assert form["password"] == "" and form["ota_password"] == ""
    assert form["has_password"] is True and form["has_ota_password"] is True
    assert PSK not in json.dumps(form) and OTA not in json.dumps(form)


def test_a_new_room_starts_from_the_defaults_and_a_blank_name(tmp_path):
    form = ss.form_for("", tmp_path)
    assert form["room"] == "" and form["ip"] == ""
    assert form["board"] == sp.DEFAULTS["board"]
    assert form["has_password"] is False and form["has_ota_password"] is False
    assert form["preset"] == sp.DEFAULTS["preset"]


def test_a_secret_is_a_word_never_a_shape():
    assert ss.secret_word(True) == "set"
    assert ss.secret_word(False) == "not set"
    # never a masked string, never a hash, never a last-four
    assert "•" not in ss.secret_word(True)
    assert not any(ch.isdigit() for ch in ss.secret_word(True))


def test_picking_a_mount_fills_the_geometry_and_shows_its_note():
    fill = ss.preset_fill("desk")
    preset = sp.PRESETS["desk"]
    assert fill["nearest_m"] == "2.00" and fill["range_m"] == "3.50"
    assert fill["timeout_s"] == "10" and fill["still"] is True
    assert fill["mount_note"] == preset.mount
    # the doorway is honestly a motion sensor and the fill says so
    assert ss.preset_fill("doorway")["still"] is False
    assert ss.preset_fill("nonsense") == {}


# ================================================ where each room stands
def _cfg(rooms=None, zones=None, enabled=True):
    data = {"presence": {"room_sensor_enabled": enabled,
                         "rooms": rooms if rooms is not None else []},
            "zones": {"enabled": True,
                      "rooms": zones if zones is not None else []}}

    def get_option(key, default=None):
        node = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    return get_option, data


def test_a_room_reports_which_of_the_four_states_it_is_in(tmp_path):
    _write_raw(tmp_path, "office")
    _write_raw(tmp_path, "den")
    get_option, _ = _cfg(
        rooms=[{"name": "office", "url": "http://%s" % IP, "primary": True}],
        zones=[{"name": "office", "bands": [{"name": "b", "near_m": 0.75,
                                             "far_m": 3.0}]}])
    states = {s.room: s for s in ss.room_states(get_option, tmp_path)}
    assert set(states) == {"office", "den"}
    office = states["office"]
    assert (office.has_profile, office.polled, office.has_ladder,
            office.primary) == (True, True, True, True)
    den = states["den"]
    assert (den.has_profile, den.polled, den.has_ladder) == (True, False, False)
    assert "no ladder" in ss.state_line(den)
    assert "polled" in ss.state_line(office) and "ladder" in ss.state_line(office)


def test_the_master_switch_being_off_means_no_room_is_polled(tmp_path):
    """roomfabric.room_specs() returns [] at its FIRST line when the master
    switch is false, whatever the address says. A page that showed "polled"
    there would be showing him a room nothing reads."""
    _write_raw(tmp_path, "office")
    get_option, _ = _cfg(rooms=[{"name": "office", "url": "http://%s" % IP}],
                         enabled=False)
    state = ss.room_states(get_option, tmp_path)[0]
    assert state.listed is True and state.polled is False
    assert "switched off" in ss.state_line(state)


def test_a_room_in_the_config_with_no_profile_is_still_listed(tmp_path):
    """He may have written presence.rooms by hand. The picker must show it
    or the sheet looks like it lost his room."""
    get_option, _ = _cfg(rooms=[{"name": "kitchen", "url": "http://%s" % IP}])
    states = ss.room_states(get_option, tmp_path)
    assert [s.room for s in states] == ["kitchen"]
    assert states[0].has_profile is False


# ==================================================== the merge, not a rebuild
def test_the_presence_write_carries_a_satellites_password_byte_for_byte():
    """THE assistant.json TRAP. AssistantConfig.get() does NOT redact, so
    reading presence.rooms hands back the satellite's real basic-auth
    password. It is carried through the merge and never bound to anything."""
    raw = [{"name": "kitchen", "url": "http://192.0.2.11", "username": "jarvis",
            "password": SAT_PW, "sensors": ["presence"], "lease_ttl_s": 120},
           {"name": "office", "url": "http://192.0.2.9", "primary": True}]
    out = ss.merged_presence_rooms(raw, "office",
                                   {"url": "http://%s" % NEW_IP,
                                    "label": "office", "primary": True,
                                    "enabled": True, "timeout_s": 1.5})
    kitchen = [e for e in out if e["name"] == "kitchen"][0]
    assert kitchen["password"] == SAT_PW and kitchen["username"] == "jarvis"
    assert kitchen["sensors"] == ["presence"] and kitchen["lease_ttl_s"] == 120
    office = [e for e in out if e["name"] == "office"][0]
    assert office["url"] == "http://%s" % NEW_IP
    # the ORIGINAL list is untouched: the write edits a copy
    assert raw[1]["url"] == "http://192.0.2.9"


def test_a_room_that_is_not_in_the_list_is_appended_not_substituted():
    raw = [{"name": "kitchen", "url": "http://192.0.2.11", "password": SAT_PW}]
    out = ss.merged_presence_rooms(raw, "den", {"url": "http://%s" % NEW_IP,
                                                "label": "den"})
    assert [e["name"] for e in out] == ["kitchen", "den"]
    assert out[0]["password"] == SAT_PW
    assert out[1]["enabled"] is True


def test_only_one_room_can_be_the_primary_one():
    """roomfabric forces primary onto the FIRST entry when none is marked,
    so two primaries is a silent coin toss about which room the camera and
    the speaker belong to."""
    raw = [{"name": "kitchen", "url": "http://192.0.2.11", "primary": True,
            "password": SAT_PW},
           {"name": "office", "url": "http://192.0.2.9"}]
    out = ss.merged_presence_rooms(raw, "office", {"url": "http://%s" % NEW_IP,
                                                   "primary": True})
    by = {e["name"]: e for e in out}
    assert by["office"]["primary"] is True
    assert by["kitchen"]["primary"] is False
    assert by["kitchen"]["password"] == SAT_PW    # and nothing else moved


def test_not_marking_it_primary_leaves_every_other_room_alone():
    raw = [{"name": "kitchen", "url": "http://192.0.2.11", "primary": True}]
    out = ss.merged_presence_rooms(raw, "den", {"url": "http://%s" % NEW_IP})
    assert out[0]["primary"] is True


def test_a_config_whose_rooms_are_not_a_list_is_refused_not_overwritten():
    out = ss.merged_presence_rooms({"office": "yes"}, "den", {"url": "x"})
    assert out is None
    assert ss.merged_presence_rooms(None, "den", {"url": "x"}) == [
        {"name": "den", "url": "x", "enabled": True}]


def test_a_starter_ladder_is_one_band_from_the_blind_zone_to_the_far_edge():
    values, why = sp.validate({"room": "den", "preset": "room",
                               "nearest_m": "1.6", "range_m": "4.5",
                               "timeout_s": "10", "still": True, "dhcp": False,
                               "ip": IP, "gateway": "192.0.2.1",
                               "subnet": "255.255.255.0", "mac": "",
                               "ssid": SSID, "board": "esp32dev",
                               "rx_pin": "GPIO16", "tx_pin": "GPIO17"})
    assert why == ""
    bands = ss.starter_bands(values)
    assert bands == [{"name": "den", "near_m": 0.75, "far_m": 4.5}]
    # and jarvis/zones.py must accept it, or the room is saved unplaceable
    from jarvis.zones import Band, ZoneMap
    ZoneMap("den", tuple(Band(b["name"], b["near_m"], b["far_m"])
                         for b in bands), "")


def test_the_zone_write_keeps_every_room_it_was_not_shown():
    raw = [{"name": "office", "enabled": True, "camera_zone": "at the desk",
            "bands": [{"name": "empty space", "near_m": 0.75, "far_m": 2.25}]}]
    out = ss.merged_zone_rooms(raw, "den", [{"name": "den", "near_m": 0.75,
                                             "far_m": 4.5}])
    assert [e["name"] for e in out] == ["office", "den"]
    assert out[0]["camera_zone"] == "at the desk"
    assert out[0]["bands"][0]["name"] == "empty space"
    assert out[1]["camera_zone"] == ""      # no lens in this room, and it says so


# ================================================ what a save actually writes
def _values(**over):
    form = {"room": "den", "preset": "room", "nearest_m": "1.6",
            "range_m": "4.5", "timeout_s": "10", "still": True, "dhcp": False,
            "ip": NEW_IP, "gateway": "192.0.2.1", "subnet": "255.255.255.0",
            "mac": "", "ssid": SSID, "board": "esp32dev", "rx_pin": "GPIO16",
            "tx_pin": "GPIO17", "serial_port": "/dev/ttyUSB0", "notes": ""}
    form.update(over)
    values, why = sp.validate(form)
    assert why == "", why
    return values


def test_ticking_poll_writes_the_room_into_presence_rooms():
    get_option, _ = _cfg(rooms=[{"name": "office", "url": "http://192.0.2.9"}])
    edits, notes = ss.config_writes(_values(), get_option, poll=True,
                                    primary=False, ladder=False)
    assert edits["presence.rooms"][-1]["name"] == "den"
    # the WHOLE presence URL, entity path and all, built through
    # roomsensor.entity_path -- never a bare address that a later reader has
    # to guess the entity for.
    assert edits["presence.rooms"][-1]["url"] == \
        "http://%s/binary_sensor/Presence" % NEW_IP
    assert "presence.room_sensor_enabled" not in edits   # already on
    assert "zones.rooms" not in edits


def test_adding_the_first_sensor_turns_the_master_switch_on_and_says_so():
    """Without this, adding a sensor from the UI leaves it unpolled with no
    on-screen way to fix it: roomfabric returns [] at its first line."""
    get_option, _ = _cfg(rooms=[], enabled=False)
    edits, notes = ss.config_writes(_values(), get_option, poll=True,
                                    primary=True, ladder=False)
    assert edits["presence.room_sensor_enabled"] is True
    assert any("switched room sensing on" in n for n in notes)


def test_not_ticking_poll_writes_the_profile_only():
    get_option, _ = _cfg(rooms=[])
    edits, notes = ss.config_writes(_values(), get_option, poll=False,
                                    primary=False, ladder=False)
    assert edits == {}
    assert any("not polled" in n for n in notes)


def test_ticking_the_ladder_writes_one_band_for_the_new_room():
    get_option, _ = _cfg(rooms=[], zones=[])
    edits, _ = ss.config_writes(_values(), get_option, poll=True,
                                primary=False, ladder=True)
    assert edits["zones.rooms"] == [{"name": "den", "enabled": True,
                                     "camera_zone": "",
                                     "bands": [{"name": "den", "near_m": 0.75,
                                                "far_m": 4.5}]}]


def test_a_room_that_already_has_a_ladder_is_never_given_a_second_one():
    get_option, _ = _cfg(rooms=[], zones=[
        {"name": "den", "bands": [{"name": "b", "near_m": 0.75, "far_m": 2.0}]}])
    edits, notes = ss.config_writes(_values(), get_option, poll=True,
                                    primary=False, ladder=True)
    assert "zones.rooms" not in edits
    assert any("already has" in n for n in notes)


def test_the_saved_sentence_separates_this_page_from_the_running_jarvis():
    """THE MOST LIKELY WAY TO MISLEAD HIM. After an add, THIS PAGE polls the
    new sensor at once -- he can walk in front of it and watch the numbers
    move -- while the presence lane that decides whether Jarvis thinks he is
    home does not until the next start. Two sentences, at the point of the
    add."""
    line = ss.saved_line("den", polled=True)
    assert "this page" in line and "restart" in line
    assert "den" in line
    assert "restart" not in ss.saved_line("den", polled=False)


# ================================================== CHECK, and it never writes
class _Radar:
    """A transport stand-in. ``RoomSensor`` takes ``get`` as its seam, so
    this replaces the socket entirely: no request leaves the process and no
    address on his LAN is named (192.0.2.x is RFC 5737 TEST-NET-1)."""

    def __init__(self, present=True, move=5, still=5, dead=False):
        self.present, self.move, self.still, self.dead = present, move, still, dead
        self.gets, self.posts = [], []

    def get(self, url, timeout):
        self.gets.append(url)
        if self.dead:
            raise OSError("no route to host")
        if "binary_sensor" in url:
            return '{"value": %s}' % ("true" if self.present else "false")
        if "Max%20move%20gate" in url:
            return '{"value": %d}' % self.move
        if "Max%20still%20gate" in url:
            return '{"value": %d}' % self.still
        return '{"value": 5}'

    def post(self, url, timeout):
        self.posts.append(url)


def test_check_reports_presence_the_round_trip_and_the_gates():
    radar = _Radar(present=True, move=6, still=6)
    line = ss.check_device(_values(), get=radar.get)
    assert "SOMEONE" in line and "ms" in line
    assert "move 6 / still 6" in line and "ok" in line
    assert radar.posts == []           # READ ONLY, structurally


def test_check_names_a_gate_mismatch_in_words_not_as_a_tick():
    """The measured 2026-09-03 failure: a power cycle left a sensor seeing
    75 cm and nothing said so. It is named, with both numbers."""
    radar = _Radar(present=False, move=0, still=0)
    line = ss.check_device(_values(), get=radar.get)
    assert "empty" in line
    assert "move 0 / still 0" in line and "profile wants 6 / 6" in line
    assert "0.75 m" in line             # what gate 0 actually covers
    assert "does not match" in line


def test_check_on_a_device_that_does_not_answer_says_so_plainly():
    radar = _Radar(dead=True)
    line = ss.check_device(_values(), get=radar.get)
    assert "no answer" in line and NEW_IP in line
    assert "powered" in line


def test_check_refuses_a_profile_with_no_address_without_sending_anything():
    radar = _Radar()
    line = ss.check_device(_values(dhcp=True, ip=""), get=radar.get)
    assert "no address" in line
    assert radar.gets == []


# ============================== TUNE, the one control that writes to hardware
def test_tune_writes_each_number_and_reads_every_one_of_them_back():
    radar = _Radar(move=0, still=0)

    def post(url, timeout):
        radar.posts.append(url)
        if "Max%20move%20gate" in url:
            radar.move = 6
        if "Max%20still%20gate" in url:
            radar.still = 6

    lines = ss.tune_device(_values(), get=radar.get, post=post, sleep=lambda s: None)
    assert len(radar.posts) == 3        # absence delay + the two gates
    text = "\n".join(lines)
    assert "Max move gate" in text and "0 -> 6" in text and "ok" in text
    assert all("value=" in u for u in radar.posts)


def test_tune_says_NOT_APPLIED_when_the_readback_disagrees():
    """The read-back is the whole point: an ESPHome rename would otherwise
    leave the write silently ignored and this claiming a tuning it never
    applied."""
    radar = _Radar(move=0, still=0)
    lines = ss.tune_device(_values(), get=radar.get, post=lambda u, t: None,
                           sleep=lambda s: None)
    text = "\n".join(lines)
    assert "NOT APPLIED" in text and "asked 6" in text


def test_tune_sends_nothing_at_all_without_an_address():
    radar = _Radar()
    lines = ss.tune_device(_values(dhcp=True, ip=""), get=radar.get,
                           post=radar.post, sleep=lambda s: None)
    assert radar.posts == [] and radar.gets == []
    assert any("no address" in ln for ln in lines)


def test_every_url_this_lane_builds_goes_through_entity_path():
    """THE 404 BUG MUST NOT COME BACK. web_server v2 serves an entity at its
    NAME percent-encoded, not at the snake_case object_id; every URL in this
    tree was built from an object_id once and every poll and every tune
    write was a silent 404."""
    from jarvis import roomsensor
    radar = _Radar()
    ss.check_device(_values(), get=radar.get)
    ss.tune_device(_values(), get=radar.get, post=radar.post,
                   sleep=lambda s: None)
    for url in radar.gets + radar.posts:
        assert "%20" in url or url.endswith("/binary_sensor/Presence")
        assert "_gate" not in url and "absence_delay" not in url.lower()
    assert roomsensor.entity_path("number", "Max move gate") == \
        "/number/Max%20move%20gate"
    assert any(roomsensor.entity_path("number", "Absence delay") + "/set?value=10"
               in u for u in radar.posts)
    assert any(roomsensor.entity_path("binary_sensor", "Presence") in u
               for u in radar.gets)


def test_the_flash_command_is_handed_over_whole_and_printed_too():
    """The page cannot flash a board -- that is USB serial and this account
    is not in `dialout`. A clipboard that did not land must not be a dead
    end, so the command is PRINTED as well as copied."""
    calls, held = [], {}

    def run(argv, **kw):
        """A stand-in for the X clipboard. NOTHING in this suite may touch
        the real one -- 2026-09-03 a diagnostic wrote a sentinel into it
        while he was pasting."""
        calls.append(argv)
        if "-o" in argv:
            class R:
                returncode = 0
                stdout = held.get("text", b"")
            return R()
        held["text"] = kw.get("input", b"")

        class W:
            returncode = 0
        return W()
    text, ok = ss.copy_flash_command("den", run=run)
    assert "room_sensor.py flash den" in text
    assert calls and calls[0][0] == "xclip"
    assert ok is True


def test_the_secrets_never_appear_in_anything_this_module_returns(tmp_path,
                                                                  caplog):
    """Belt and braces over every string the sheet can put on screen."""
    _write_raw(tmp_path, "office")
    with caplog.at_level(logging.DEBUG):
        form = ss.form_for("office", tmp_path)
        get_option, _ = _cfg(rooms=[{"name": "office", "url": "http://%s" % IP,
                                     "password": SAT_PW}])
        states = ss.room_states(get_option, tmp_path)
        lines = [ss.state_line(s) for s in states]
        lines.append(ss.saved_line("office", polled=True))
        lines.append(sp.address_line(sp.validate(dict(form, room="office",
                                                      password="", ota_password=""))[0]))
        edits, notes = ss.config_writes(_values(room="office"), get_option,
                                        poll=True, primary=False, ladder=False)
        lines.extend(notes)
    # EVERY STRING THIS SHEET CAN PUT ON SCREEN, and the log with it.
    blob = json.dumps([form, lines], default=str)
    for secret in (PSK, OTA, SAT_PW):
        assert secret not in blob
        assert secret not in caplog.text
    # ...and the satellite's password SURVIVES in the value handed to
    # set_option, because that is the merge promise: it goes straight into
    # the atomic 0600 writer, never onto a widget and never into a message.
    assert SAT_PW in json.dumps(edits)


# ============================================================ on a display
def _display() -> str:
    d = (os.environ.get("JARVIS_UI_TEST_DISPLAY") or "").strip()
    if not d:
        return ""
    base = d.split(".")[0]
    if base in FORBIDDEN_DISPLAYS or base.split(":")[-1] in ("0", "1"):
        pytest.fail("JARVIS_UI_TEST_DISPLAY=%r is a desktop display; this "
                    "file builds windows and will not open one on his "
                    "screen" % d)
    return d


HIS_W, HIS_H, SCALE = 1040, 1760, 2.0      # the window he actually runs
SMALL_W, SMALL_H = 920, 1440


FONT_GLOBALS = ("_FAMILY", "_FAMILY_MONO", "_HAS_DISPLAY", "_DISPLAY")


@pytest.fixture(autouse=True)
def _restore_look():
    """theme.resolve_fonts() is a ONE-WAY DOOR and select_look() is global.

    Without this, the size tokens this file leaves behind fail
    tests/test_theme_look.py's oracle comparison in the NEXT file of the
    same run -- which is how a green file makes a green file red. The same
    fixture tests/test_ui_classic_frozen.py carries, for the same reason.
    """
    from jarvis.ui import theme
    from jarvis.ui import widgets as wg
    fonts = {k: getattr(theme, k) for k in FONT_GLOBALS}
    yield
    theme.apply_scale(1.0)
    wg.set_scale(1.0)
    theme.select_look(theme.DEFAULT_LOOK)
    for k, v in fonts.items():
        setattr(theme, k, v)


@pytest.fixture
def root():
    from jarvis.ui import theme
    from jarvis.ui import widgets as wg
    display = _display()
    if not display:
        pytest.skip("set JARVIS_UI_TEST_DISPLAY=:9N (a private Xvfb) to run "
                    "the measured sheet tests")
    import tkinter as tk
    try:
        fds = len(os.listdir("/proc/self/fd"))
    except OSError:
        fds = 0
    if fds >= FD_SETSIZE - 32:
        pytest.skip("this process already holds %d open descriptors" % fds)
    try:
        r = tk.Tk(screenName=display)
    except tk.TclError as exc:
        pytest.skip("no X server at %s: %s" % (display, exc))
    r.geometry("%dx%d+0+0" % (HIS_W, HIS_H))
    theme.resolve_fonts(r)
    theme.apply_scale(SCALE)
    wg.set_scale(SCALE)
    theme.select_look("holo")
    yield r
    try:
        r.destroy()
    except Exception:                      # noqa: BLE001 - teardown
        pass


class _Services:
    """get_option / set_option over a plain dict. No file, no app."""

    def __init__(self, data=None):
        self.data = data if data is not None else {
            "presence": {"room_sensor_enabled": True, "rooms": [
                {"name": "office", "url": "http://%s/binary_sensor/Presence" % IP,
                 "primary": True}]},
            "zones": {"enabled": True, "rooms": [
                {"name": "office", "enabled": True, "camera_zone": "",
                 "bands": [{"name": "at the desk", "near_m": 0.75,
                            "far_m": 3.5}]}]}}
        self.writes = []
        self.sensing = None

    def get_option(self, key, default=None):
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_option(self, key, value):
        self.writes.append((key, value))
        node = self.data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        return True


def _sheet(root, tmp_path, services=None, radar=None):
    """The sheet over a host frame, with every transport injected and its
    worker made synchronous so a test never races a thread."""
    import tkinter as tk
    from jarvis.ui import sensor_setup as mod
    host = tk.Frame(root, width=HIS_W, height=HIS_H)
    host.pack_propagate(False)
    host.pack()
    radar = radar or _Radar()
    sheet = mod.SetupSheet(host, services=services or _Services(),
                           directory=tmp_path, get=radar.get, post=radar.post,
                           run=lambda *a, **kw: None, sleep=lambda s: None)
    sheet._work = lambda fn, done: done(fn())      # synchronous, no thread
    sheet.show()
    root.update_idletasks()
    return sheet, radar


def _tree_text(widget) -> str:
    """Every string the sheet can put in front of him: label text, button
    text, and the contents of every entry."""
    out = []
    stack = [widget]
    while stack:
        w = stack.pop()
        stack.extend(w.winfo_children())
        for option in ("text",):
            try:
                out.append(str(w.cget(option)))
            except Exception:              # noqa: BLE001 - not that kind
                pass
        try:
            out.append(str(w.get()))       # every Entry, masked or not
        except Exception:                  # noqa: BLE001 - not an Entry
            pass
        try:
            for item in w.find_all():      # every canvas text item
                out.append(str(w.itemcget(item, "text")))
        except Exception:                  # noqa: BLE001 - not a Canvas
            pass
    return "\n".join(out)


def test_the_secret_boxes_are_masked_empty_and_never_prefilled(root, tmp_path):
    _write_raw(tmp_path, "office")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    root.update_idletasks()
    for entry in (sheet._secret["password"], sheet._secret["ota_password"]):
        assert entry.cget("show") == "•"
        assert entry.get() == ""
        assert not entry.cget("textvariable")     # never bound to anything
    # what IS shown is a WORD
    assert ss.SET_WORD in _tree_text(sheet)
    assert "•••" not in _tree_text(sheet)


def test_no_secret_reaches_the_widget_tree_or_the_log(root, tmp_path, caplog):
    """The whole tree, after a real open-edit-save, and caplog at DEBUG --
    the level a stray %r of the form would land at."""
    _write_raw(tmp_path, "office")
    services = _Services()
    with caplog.at_level(logging.DEBUG):
        sheet, _ = _sheet(root, tmp_path, services)
        sheet.select("office")
        root.update_idletasks()
        sheet._field["ip"].delete(0, "end")
        sheet._field["ip"].insert(0, NEW_IP)
        sheet.save()
        root.update_idletasks()
        text = _tree_text(sheet)
    for secret in (PSK, OTA):
        assert secret not in text
        assert secret not in caplog.text
    assert "saved" in sheet.result().lower()


def test_saving_with_an_empty_secret_box_leaves_the_stored_value(root,
                                                                 tmp_path):
    path = _write_raw(tmp_path, "office")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    sheet._field["ip"].delete(0, "end")
    sheet._field["ip"].insert(0, NEW_IP)
    sheet.save()
    raw = json.loads(path.read_text())
    assert raw["password"] == PSK and raw["ota_password"] == OTA
    assert raw["ip"] == NEW_IP
    assert (os.stat(path).st_mode & 0o777) == 0o600


def test_typing_one_secret_replaces_only_that_one(root, tmp_path):
    path = _write_raw(tmp_path, "office")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    sheet._secret["password"].insert(0, "a-third-invented-psk")
    sheet.save()
    raw = json.loads(path.read_text())
    assert raw["password"] == "a-third-invented-psk"
    assert raw["ota_password"] == OTA
    # and the box is cleared after a successful save, so it cannot be read
    # back off the screen by the next person to walk past
    assert sheet._secret["password"].get() == ""


@pytest.mark.parametrize("field,bad,word", [
    ("ip", "192.0.2", "not an address"),
    ("ip", "8.8.8.8", "home network"),
    ("rx_pin", "GPIO99", "GPIO0 to GPIO39"),
    ("rx_pin", "17", "GPIO"),
    ("range_m", "1.0", "must be beyond"),
    ("nearest_m", "0.5", "blind zone"),
    ("nearest_m", "banana", "not a number"),
])
def test_a_bad_value_is_refused_on_screen_and_nothing_is_written(
        root, tmp_path, field, bad, word):
    path = _write_raw(tmp_path, "office")
    before = path.read_bytes()
    services = _Services()
    sheet, _ = _sheet(root, tmp_path, services)
    sheet.select("office")
    sheet._field[field].delete(0, "end")
    sheet._field[field].insert(0, bad)
    sheet.save()
    root.update_idletasks()
    assert word in sheet.result()
    assert path.read_bytes() == before        # refused, never repaired
    assert services.writes == []


def test_the_sheet_sends_nothing_to_any_device_until_he_presses_check(
        root, tmp_path):
    _write_raw(tmp_path, "office")
    sheet, radar = _sheet(root, tmp_path)
    sheet.select("office")
    sheet.save()
    root.update_idletasks()
    assert radar.gets == [] and radar.posts == []
    sheet.check()
    root.update_idletasks()
    assert radar.gets and radar.posts == []   # CHECK reads, never writes


def test_tune_writes_to_his_hardware_only_on_the_second_press(root, tmp_path):
    """It is the only control in this lane that writes to the radar, so it
    arms on one press and fires on the next."""
    _write_raw(tmp_path, "office")
    sheet, radar = _sheet(root, tmp_path)
    sheet.select("office")
    sheet.tune()
    root.update_idletasks()
    assert radar.posts == []
    assert "again" in sheet.result().lower()
    sheet.tune()
    root.update_idletasks()
    assert len(radar.posts) == 3


def test_the_copy_button_never_touches_the_real_clipboard(root, tmp_path):
    """His clipboard and his X selections are HIS. Everything goes through
    the injected runner -- 2026-09-03 a diagnostic wrote a sentinel into it
    while he was pasting."""
    _write_raw(tmp_path, "office")
    seen = []
    sheet, _ = _sheet(root, tmp_path)
    sheet._run = lambda argv, **kw: seen.append(argv) or type(
        "R", (), {"returncode": 1, "stdout": b""})()
    sheet.select("office")
    sheet.copy_command()
    root.update_idletasks()
    assert all(a[0] == "xclip" for a in seen)
    # the command is on screen whatever the clipboard did
    assert "room_sensor.py flash office" in sheet.result()


def test_the_sheet_says_what_it_cannot_do(root, tmp_path):
    """Whatever it leaves out, it SAYS it leaves out and what he does
    instead. Flashing is USB serial and this account has no dialout."""
    _write_raw(tmp_path, "office")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    root.update_idletasks()
    text = _tree_text(sheet)
    assert "dialout" in text
    assert "flash" in text.lower()


def test_adding_a_new_room_writes_the_profile_and_the_room_list(root,
                                                                tmp_path):
    services = _Services()
    sheet, _ = _sheet(root, tmp_path, services)
    sheet.select("")                      # + NEW
    for field, value in (("room", "den"), ("ip", NEW_IP), ("ssid", SSID)):
        sheet._field[field].delete(0, "end")
        sheet._field[field].insert(0, value)
    sheet._poll.set(True, animate=False)
    sheet._ladder.set(True, animate=False)
    sheet._secret["password"].insert(0, "invented-den-psk")
    sheet.save()
    root.update_idletasks()
    raw = json.loads((tmp_path / "den.json").read_text())
    assert raw["room"] == "den" and raw["password"] == "invented-den-psk"
    keys = dict(services.writes)
    assert [e["name"] for e in keys["presence.rooms"]] == ["office", "den"]
    assert any(e["name"] == "den" for e in keys["zones.rooms"])
    assert "restart" in sheet.result()


def test_a_new_room_may_not_be_given_a_name_the_config_shapes_disagree_on(
        root, tmp_path):
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("")
    sheet._field["room"].insert(0, "Sam's room")
    sheet._field["ip"].delete(0, "end")
    sheet._field["ip"].insert(0, NEW_IP)
    sheet._field["ssid"].insert(0, SSID)
    sheet.save()
    assert "lower-case" in sheet.result()
    assert not list(tmp_path.glob("*.json"))


# ==================================================== it fits HIS window
@pytest.mark.parametrize("size", [(HIS_W, HIS_H), (SMALL_W, SMALL_H)])
def test_the_sheet_fits_the_stage_at_both_sizes(root, tmp_path, size):
    """RENDERED AT 1040x1760, the app default and the window he actually
    runs, as well as at 920x1440. A drawer row reviewed twice at 920x1440
    overflowed its slot at his real size on 2026-09-05 and he found it in
    ten minutes."""
    import tkinter as tk
    from jarvis.ui import sensor_setup as mod
    _write_raw(tmp_path, "office")
    _write_raw(tmp_path, "kitchen")
    width, height = size
    root.geometry("%dx%d+0+0" % (width, height))
    stage = int(height * 852 / 1440)      # the stage this page is placed over
    host = tk.Frame(root, width=width, height=stage)
    host.pack_propagate(False)
    host.pack()
    sheet = mod.SetupSheet(host, services=_Services(), directory=tmp_path)
    sheet.show()
    root.update_idletasks()
    sheet.select("office")
    root.update_idletasks()
    # THE FOOT IS NEVER CLIPPED: every button he needs is inside the stage,
    # and so are the three switches that decide whether Jarvis reads the room
    # at all -- they are the payoff of adding a sensor from the UI and they
    # are not going below the fold of a form that scrolls.
    for name, button in dict(sheet.buttons(), **sheet.switches()).items():
        bottom = button.winfo_rooty() - host.winfo_rooty() + \
            button.winfo_height()
        assert bottom <= stage, (name, bottom, stage, size)
        assert button.winfo_width() > 0
    # and the body either fits or SAYS it scrolls
    assert sheet.overflow_px() >= 0
    sheet.hide()


def test_the_chosen_mount_chip_is_actually_marked(root, tmp_path):
    """The picked preset must LOOK picked. RoundButton had no way to change
    its kind after construction, so the first version of this swallowed an
    AttributeError and the chips never highlighted -- silently, which is the
    worst way for a control to be wrong."""
    from jarvis.ui.widgets import RoundButton
    assert callable(getattr(RoundButton, "set_kind", None))
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("")
    sheet.pick_preset("doorway")
    root.update_idletasks()
    # in holo an accent button is a RING, not a fill: the difference is the
    # outline and the text colour (widgets.RoundButton._kinds).
    marks = {key: (btn._spec["outline"], btn._spec["fg"])
             for key, btn in sheet._preset_btn.items()}
    assert marks["doorway"] != marks["desk"]
    assert sheet.form()["preset"] == "doorway"
    sheet.pick_preset("desk")
    root.update_idletasks()
    assert (sheet._preset_btn["desk"]._spec["outline"],
            sheet._preset_btn["desk"]._spec["fg"]) == marks["doorway"]


def test_the_board_rows_are_folded_away_until_he_asks_for_them(root, tmp_path):
    """The pins, the board id and the USB port are set once and never again.
    They are behind a fold so the rows he uses -- name, mount, geometry,
    address, Wi-Fi, and the switches that make Jarvis read the room -- fit
    closer to the top of a sheet that scrolls."""
    _write_raw(tmp_path, "office")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    root.update_idletasks()
    assert not sheet._field["rx_pin"].winfo_ismapped()
    folded = sheet.overflow_px()
    # the values are still THERE, so a save carries them
    assert sheet._field["rx_pin"].get() == "GPIO16"
    assert sheet.form()["board"] == "esp32dev"
    sheet.toggle_board()
    root.update_idletasks()
    assert sheet._field["rx_pin"].winfo_ismapped()
    assert sheet.overflow_px() > folded


def test_the_three_switches_are_pinned_beside_save_not_at_the_end_of_a_form(
        root, tmp_path):
    """"poll this room" is what makes the whole exercise worth anything, and
    it was at the bottom of a body that scrolls by 598 px at his own window
    (measured on :94, S=2). It sits in the pinned foot with SAVE."""
    _write_raw(tmp_path, "office")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    root.update_idletasks()
    assert set(sheet.switches()) == {"poll", "primary", "ladder"}
    foot = sheet._buttons["save"].master.master
    for name, toggle in sheet.switches().items():
        assert toggle.winfo_ismapped(), name
        assert str(toggle).startswith(str(foot)), name


@pytest.mark.parametrize("size", [(HIS_W, HIS_H), (SMALL_W, SMALL_H)])
def test_no_foot_row_is_wider_than_the_window_at_either_size(root, tmp_path,
                                                             size):
    """MEASURED on :94 at S=2, 2026-09-05, with both rooms present::

           row              width    920 window    his 1040 window
           switches          618 px   856 avail     976 avail
           CHECK/TUNE/COPY   612 px
           SAVE/CLOSE        361 px

    One row of five buttons was ~780 px, which fitted at 1040 and not at
    920 -- the same defect as this morning's drawer row, in the other
    direction. This is why the foot is three rows.
    """
    import tkinter as tk
    from jarvis.ui import sensor_setup as mod
    from jarvis.ui import theme
    _write_raw(tmp_path, "office")
    _write_raw(tmp_path, "kitchen")
    width, height = size
    root.geometry("%dx%d+0+0" % (width, height))
    host = tk.Frame(root, width=width, height=int(height * 852 / 1440))
    host.pack_propagate(False)
    host.pack()
    sheet = mod.SetupSheet(host, services=_Services(), directory=tmp_path)
    sheet.show()
    sheet.select("office")
    root.update_idletasks()
    avail = width - 2 * theme.PAD
    rows = {"switches": sheet._switches["poll"].master,
            "devices": sheet._buttons["check"].master,
            "save": sheet._buttons["save"].master}
    for name, row in rows.items():
        used = sum(c.winfo_width() for c in row.winfo_children())
        assert used <= avail, (name, used, avail, size)
    sheet.hide()


# ================================================= the button on the page
def test_the_sensors_page_offers_setup_in_holo_and_never_in_classic(root,
                                                                    tmp_path):
    """CLASSIC IS PIXEL-FROZEN at the jarvis-v3 tip and the 09-05 relayout
    already moved it by 202,322 px at his own window size, so every new
    control here is holo's -- the look he actually runs. In classic the
    button is simply absent and NOTHING is added in its place: a caption
    saying "setup lives in holo" would itself be moved pixels."""
    import tkinter as tk
    from jarvis.ui import sensors_page as page_mod
    from jarvis.ui import theme
    for look, wanted in (("holo", True), ("classic", False)):
        theme.select_look(look)
        host = tk.Frame(root, width=HIS_W, height=800)
        host.pack_propagate(False)
        host.pack()
        page = page_mod.SensorsPage(host, services=_Services())
        page.place(in_=host, x=0, y=0, relwidth=1.0, height=800)
        root.update_idletasks()
        assert (page.setup_btn is not None) is wanted, look
        if wanted:
            # left of SAVE, in the pinned foot, where the row has slack
            assert page.setup_btn.master is page._save_btn.master
            assert page.setup_btn.winfo_rootx() < page._save_btn.winfo_rootx()
        host.destroy()


def test_pressing_setup_opens_the_sheet_over_the_page(root, tmp_path):
    import tkinter as tk
    from jarvis.ui import sensors_page as page_mod
    _write_raw(tmp_path, "office")
    host = tk.Frame(root, width=HIS_W, height=900)
    host.pack_propagate(False)
    host.pack()
    page = page_mod.SensorsPage(host, services=_Services(),
                                profile_dir=tmp_path)
    page.place(in_=host, x=0, y=0, relwidth=1.0, height=900)
    root.update_idletasks()
    assert page.setup is None                 # built on the first press only
    page.open_setup()
    root.update_idletasks()
    assert page.setup is not None and page.setup.is_open
    assert "office" in _tree_text(page.setup).lower()
    page.setup.hide()


def test_a_room_added_from_the_sheet_appears_on_the_page_without_a_restart(
        root, tmp_path):
    """The page polls the new sensor at once -- he can walk in front of it
    and watch the numbers move -- while Jarvis's own presence lane does not
    until the next start. Both halves are said on the result line; this
    pins the half the page owns."""
    import tkinter as tk
    from jarvis.ui import sensors_page as page_mod
    services = _Services()
    host = tk.Frame(root, width=HIS_W, height=900)
    host.pack_propagate(False)
    host.pack()
    page = page_mod.SensorsPage(host, services=services, profile_dir=tmp_path)
    page.place(in_=host, x=0, y=0, relwidth=1.0, height=900)
    root.update_idletasks()
    assert [s.name for s in page.specs] == ["office"]
    page.open_setup()
    sheet = page.setup
    sheet._work = lambda fn, done: done(fn())
    sheet.select("")
    for field, value in (("room", "den"), ("ip", NEW_IP), ("ssid", SSID)):
        sheet._field[field].delete(0, "end")
        sheet._field[field].insert(0, value)
    sheet._poll.set(True, animate=False)
    sheet._ladder.set(True, animate=False)
    sheet.save()
    root.update_idletasks()
    assert "den" in sheet.result()
    assert [s.name for s in page.specs] == ["office", "den"]
    assert "den" in page._blocks
    assert page.ladders.for_room("den") is not None
    sheet.hide()


def test_the_empty_page_points_at_setup_in_holo_and_at_the_file_in_classic():
    """A page that says "edit assistant.json and restart" is the answer this
    lane exists to replace -- in the look that HAS the button. Classic is
    frozen, so it keeps the old sentence word for word."""
    from jarvis.ui import sensors_page as page_mod
    from jarvis.ui import theme
    try:
        for look, wanted, unwanted in (("holo", "SETUP", "assistant.json"),
                                       ("classic", "assistant.json", "SETUP")):
            theme.select_look(look)
            for enabled in (True, False):
                line = page_mod.empty_state_line(
                    lambda k, d=None, e=enabled: (
                        e if k == "presence.room_sensor_enabled" else d))
                assert wanted in line, (look, enabled)
                assert unwanted not in line, (look, enabled)
    finally:
        theme.select_look(theme.DEFAULT_LOOK)


def test_a_dhcp_room_with_no_address_yet_is_not_claimed_to_be_polled():
    """He can save a profile before the router has handed the device an
    address. presence.rooms takes the entry, but roomfabric SKIPS a room
    with no url -- so "this page starts reading it now" would be a lie, and
    the one thing this surface must not do is claim a sensor is being read
    when it is not."""
    get_option, _ = _cfg(rooms=[])
    values = _values(dhcp=True, ip="")
    assert values["ip"] == ""
    edits, notes = ss.config_writes(values, get_option, poll=True,
                                    primary=False, ladder=False)
    assert edits["presence.rooms"][-1]["url"] == ""
    assert any("no address" in n for n in notes)
    assert any("cannot poll" in n for n in notes)
    assert "restart" not in ss.saved_line("den", polled=False)


# =========================== defects found by LOOKING at the rendered frame
def test_every_wrapping_paragraph_actually_wraps(root, tmp_path):
    """PHOTOGRAPHED at 1040x1760 on :94: the line explaining the three
    switches ran off the right edge mid-word -- "...primary — where he is by
    default; zone ladd". It was not in the rewrap list. Every label that can
    be longer than the sheet is wrapped to the sheet's width, and this test
    walks the tree rather than naming them, so the next one added is covered
    too."""
    _write_raw(tmp_path, "office")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    root.update_idletasks()
    width = sheet.winfo_width()
    assert width > 200
    import tkinter.font as tkfont
    stack, long = [sheet], []
    while stack:
        w = stack.pop()
        stack.extend(w.winfo_children())
        try:
            text = str(w.cget("text"))
            wrap = int(w.cget("wraplength"))
        except Exception:                  # noqa: BLE001 - not a Label
            continue
        if not text or wrap:
            continue
        font = tkfont.Font(font=w.cget("font"))
        if font.measure(text) > width:
            long.append(text[:60])
    assert long == [], long


def test_the_picker_line_does_not_run_two_rooms_together(root, tmp_path):
    """PHOTOGRAPHED: "kitchen: no profile · polled · ladder · office: profile
    · polled · ladder · primary" -- the separator between ROOMS was the same
    dot as the separator between a room's own facts, so it read as one
    run-on list of nine things."""
    _write_raw(tmp_path, "office")
    _write_raw(tmp_path, "kitchen")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    root.update_idletasks()
    labels = [w for w in sheet._picker.winfo_children()
              if w.winfo_class() == "Label"]
    assert labels
    text = labels[-1].cget("text")
    assert text.count("\n") >= 1          # one room per line
    for line in text.splitlines():
        assert line.count(":") == 1, line


def test_the_sheet_covers_the_page_and_not_the_header_and_tabs(root, tmp_path):
    """PHOTOGRAPHED: the sheet took the WHOLE window -- wordmark, status
    pill, sensing badge and the tab row all gone. It is a page-level surface,
    not a takeover, so it lands on exactly the box the SENSORS page is
    placed in and the console's own furniture stays visible."""
    import tkinter as tk
    from jarvis.ui import sensors_page as page_mod
    _write_raw(tmp_path, "office")
    host = tk.Frame(root, width=HIS_W, height=HIS_H)
    host.pack_propagate(False)
    host.pack()
    strip = tk.Frame(host, height=200, bg="#123")   # stands in for the header
    strip.pack(fill="x")
    stage = tk.Frame(host, bg="#000")
    stage.pack(fill="both", expand=True)
    page = page_mod.SensorsPage(host, services=_Services(),
                                profile_dir=tmp_path, cover=(stage,))
    page.place(in_=host, **page.place_box())
    root.update_idletasks()
    page.open_setup()
    root.update_idletasks()
    sheet = page.setup
    assert sheet.winfo_rooty() == page.winfo_rooty()
    assert sheet.winfo_height() == page.winfo_height()
    assert sheet.winfo_rooty() > host.winfo_rooty()   # the header survives
    sheet.hide()


def test_a_locked_room_box_keeps_the_consoles_own_colours(root, tmp_path):
    """PHOTOGRAPHED: the disabled room entry came back in Tk's default
    light-grey, a white box in a dark console."""
    _write_raw(tmp_path, "office")
    sheet, _ = _sheet(root, tmp_path)
    sheet.select("office")
    root.update_idletasks()
    entry = sheet._field["room"]
    assert str(entry.cget("state")) == "disabled"
    assert str(entry.cget("disabledbackground")) == str(entry.cget("bg"))
    assert str(entry.cget("disabledforeground")) != "" 
