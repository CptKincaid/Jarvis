"""The room-sensor PROFILE store, and the promise that the UI above it can
never see a secret.

WHAT THIS FILE PINS. ``jarvis/sensorprofile.py`` is the only thing in the
package that opens ``~/.config/jarvis/room-sensors/<room>.json``. That file
holds his Wi-Fi PSK and the device's OTA password, and the sheet that edits
it renders to PNG on a photo rig whose frames get looked at. So the design
is not "be careful in the UI", it is a MODULE BOUNDARY:

* ``read_public`` returns every field EXCEPT the two secrets, plus two
  booleans saying whether each is set. The secret NAMES are not keys of the
  returned dict at all, so there is nothing above this line to leak.
* ``write`` re-opens the file itself and carries an unchanged secret across
  BELOW that line: an empty box means "leave it alone", and the old value
  never enters a variable the UI can see.
* the writer logs the room and the NAMES of the fields it changed. Never a
  value, and never an exception's text -- a JSON decoder's message quotes
  the bytes it choked on, and those bytes are his PSK.

EVERY NAME, ADDRESS, SSID AND PASSWORD IN THIS FILE IS INVENTED, and every
test writes into ``tmp_path``. Nothing here reads the real profile
directory; the one test that asks where that directory IS asserts only that
it follows ``PATHS.ASSISTANT_CONFIG``, which the suite has already
redirected.
"""
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import pytest

from jarvis import sensorprofile as sp

REPO = Path(__file__).resolve().parent.parent

# Invented. Not his.
PSK = "invented-psk-not-his-9317"
OTA = "invented-ota-not-his-4482"
SSID = "PRETEND-NET-5G"
IP = "192.0.2.10"           # RFC 5737 TEST-NET-1: never routed anywhere


def _script():
    """scripts/room_sensor.py loaded by path -- scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "room_sensor_script_profile", REPO / "scripts" / "room_sensor.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # @dataclass resolves through here
    spec.loader.exec_module(mod)
    return mod


def _write_raw(directory: Path, room: str, **over) -> Path:
    """A whole profile on disk, secrets and all, the way the script writes
    one. Used as the STARTING STATE for the merge tests."""
    data = {"room": room, "ssid": SSID, "password": PSK, "ota_password": OTA,
            "ip": IP, "dhcp": False, "mac": "aa:bb:cc:dd:ee:ff",
            "gateway": "192.0.2.1", "subnet": "255.255.255.0",
            "board": "esp32dev", "rx_pin": "GPIO16", "tx_pin": "GPIO17",
            "preset": "desk", "mount_note": "across the room", "nearest_m": 2.0,
            "range_m": 3.5, "still": True, "timeout_s": 10,
            "serial_port": "/dev/ttyUSB0", "flashed": True, "notes": ""}
    data.update(over)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("%s.json" % room)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


# =================================================== the shape of a profile
def test_the_field_list_is_exactly_the_scripts_dataclass():
    """PARITY. scripts/room_sensor.py owns the file format and is
    deliberately standalone-runnable, so it is not edited by this lane. Two
    shapes for one file is how a field gets written by one and dropped by
    the other, so the drift fails here instead."""
    rs = _script()
    assert tuple(rs.Profile.__dataclass_fields__) == sp.PROFILE_FIELDS


def test_the_defaults_are_the_scripts_defaults():
    rs = _script()
    prof = rs.Profile(room="pretend")
    for field in sp.PROFILE_FIELDS:
        if field == "room":
            continue
        assert sp.DEFAULTS[field] == getattr(prof, field), field


def test_the_two_secrets_are_named_and_are_not_public():
    assert sp.SECRET_FIELDS == ("password", "ota_password")
    for name in sp.SECRET_FIELDS:
        assert name in sp.PROFILE_FIELDS
        assert name not in sp.PUBLIC_FIELDS
    # the SSID is deliberately NOT a secret: it is broadcast by the router,
    # and showing it is the only way he can answer "is this profile pointed
    # at the right network".
    assert "ssid" in sp.PUBLIC_FIELDS


def test_the_profile_directory_follows_the_assistant_config(tmp_path,
                                                            monkeypatch):
    """Derived from PATHS.ASSISTANT_CONFIG, never from Path.home(), so the
    suite's redirect carries it and no test can reach the real profiles."""
    from jarvis.config import PATHS
    monkeypatch.setattr(PATHS, "ASSISTANT_CONFIG",
                        tmp_path / "cfg" / "assistant.json")
    assert sp.profile_dir() == tmp_path / "cfg" / "room-sensors"


def test_the_three_slug_rules_agree_on_a_room_name():
    """FOUR CONFIG SHAPES, ONE ROOM NAME: the profile filename, the
    presence.rooms key and the ESPHome hostname are each derived by a
    different rule in a different file. ``room_ok`` is what the sheet
    refuses a new name with, and it refuses exactly the names on which the
    three disagree."""
    from jarvis import roomfabric
    rs = _script()
    for name in ("office", "kitchen", "back-room", "den2"):
        assert sp.room_ok(name) == "", name
        stem = sp.profile_path(name, Path("/nowhere")).stem
        assert stem == roomfabric.room_name(name)
        assert rs.Profile(room=name).device_name == "jarvis-" + stem
    for name in ("", "   ", "Sam's room", "office/kitchen", "UPPER",
                 "two words", "back room"):
        assert sp.room_ok(name) != "", name


# ========================================================= reading, narrowly
def test_read_public_returns_no_secret_under_any_key(tmp_path):
    _write_raw(tmp_path, "office")
    got = sp.read_public("office", tmp_path)
    assert set(got) == set(sp.PUBLIC_FIELDS) | {"has_password",
                                                "has_ota_password"}
    for value in got.values():
        assert value != PSK and value != OTA
    assert PSK not in json.dumps(got) and OTA not in json.dumps(got)
    assert got["has_password"] is True and got["has_ota_password"] is True
    assert got["ssid"] == SSID and got["ip"] == IP


def test_read_public_says_not_set_for_a_blank_secret(tmp_path):
    _write_raw(tmp_path, "office", password="", ota_password="")
    got = sp.read_public("office", tmp_path)
    assert got["has_password"] is False and got["has_ota_password"] is False


def test_read_public_fills_missing_fields_with_the_defaults(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "spare.json").write_text('{"room": "spare", "ip": "192.0.2.9"}')
    got = sp.read_public("spare", tmp_path)
    assert got["ip"] == "192.0.2.9"
    assert got["board"] == sp.DEFAULTS["board"]
    assert got["has_password"] is False


def test_read_public_is_none_for_every_unusable_file(tmp_path):
    assert sp.read_public("missing", tmp_path) is None
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "broken.json").write_text("{not json at all")
    assert sp.read_public("broken", tmp_path) is None
    (tmp_path / "alist.json").write_text("[1, 2, 3]")
    assert sp.read_public("alist", tmp_path) is None


def test_a_broken_profile_is_reported_without_echoing_its_bytes(tmp_path,
                                                               caplog):
    """A JSON decoder's message quotes the offending line, and that line
    may be the PSK. The refusal names the file and says nothing else."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "office.json").write_text(
        '{"password": "%s" NOT-JSON}' % PSK)
    with caplog.at_level(logging.DEBUG):
        assert sp.read_public("office", tmp_path) is None
        with pytest.raises(sp.ProfileError) as exc:
            sp.write("office", {"ip": IP}, directory=tmp_path)
    assert "not valid JSON" in str(exc.value)
    assert PSK not in str(exc.value)
    assert PSK not in caplog.text


def test_list_rooms_is_the_profiles_on_disk(tmp_path):
    assert sp.list_rooms(tmp_path) == ()
    _write_raw(tmp_path, "office")
    _write_raw(tmp_path, "kitchen")
    (tmp_path / "build").mkdir()                 # the script's render dir
    (tmp_path / "office.json.bak-0903").write_text("{}")   # a stale backup
    assert sp.list_rooms(tmp_path) == ("kitchen", "office")


# ================================================== writing, and the merge
def test_an_empty_secret_box_leaves_the_stored_value_untouched(tmp_path):
    """THE RULE THE WHOLE SHEET RESTS ON. Empty means "leave it alone",
    never "clear it" -- and the carried-across value never leaves the
    writer."""
    _write_raw(tmp_path, "office")
    sp.write("office", {"ip": "192.0.2.77"}, password="", ota_password=None,
             directory=tmp_path)
    raw = json.loads((tmp_path / "office.json").read_text())
    assert raw["password"] == PSK and raw["ota_password"] == OTA
    assert raw["ip"] == "192.0.2.77"


def test_a_typed_secret_replaces_the_stored_one(tmp_path):
    _write_raw(tmp_path, "office")
    sp.write("office", {}, password="second-invented-psk", directory=tmp_path)
    raw = json.loads((tmp_path / "office.json").read_text())
    assert raw["password"] == "second-invented-psk"
    assert raw["ota_password"] == OTA          # the one he did not type


def test_a_new_profile_starts_from_the_defaults(tmp_path):
    sp.write("den", {"ssid": SSID, "ip": IP, "preset": "room"},
             password=PSK, ota_password=OTA, directory=tmp_path)
    raw = json.loads((tmp_path / "den.json").read_text())
    assert set(raw) == set(sp.PROFILE_FIELDS)
    assert raw["room"] == "den" and raw["board"] == sp.DEFAULTS["board"]
    assert raw["flashed"] is False


def test_the_writer_ignores_a_secret_smuggled_through_the_public_dict(tmp_path):
    """The UI never holds one, but the public dict is the sheet's own
    widget values and a later edit could add a key. The writer takes the
    two secrets ONLY through their keyword arguments."""
    _write_raw(tmp_path, "office")
    sp.write("office", {"password": "smuggled", "ota_password": "smuggled"},
             directory=tmp_path)
    raw = json.loads((tmp_path / "office.json").read_text())
    assert raw["password"] == PSK and raw["ota_password"] == OTA


def test_the_file_keeps_0600_and_the_directory_0700(tmp_path):
    home = tmp_path / "fresh"
    path = sp.write("den", {"ssid": SSID}, password=PSK, directory=home)
    assert (os.stat(path).st_mode & 0o777) == 0o600
    assert (os.stat(home).st_mode & 0o777) == 0o700
    # and a rewrite does not widen it
    os.chmod(path, 0o644)
    sp.write("den", {"ssid": SSID}, directory=home)
    assert (os.stat(path).st_mode & 0o777) == 0o600


def test_a_partial_write_cannot_corrupt_an_existing_profile(tmp_path,
                                                            monkeypatch):
    """The temp file is in the SAME directory and os.replace is atomic, so
    a failure anywhere before it leaves the old bytes exactly where they
    were -- including the secrets the old file holds."""
    _write_raw(tmp_path, "office")
    before = (tmp_path / "office.json").read_bytes()

    def boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        sp.write("office", {"ip": "192.0.2.99"}, password="new",
                 directory=tmp_path)
    assert (tmp_path / "office.json").read_bytes() == before
    assert json.loads(before)["password"] == PSK
    # and no half-written temp file is left behind for anyone to read
    assert sorted(p.name for p in tmp_path.iterdir()) == ["office.json"]


def test_the_writer_logs_the_names_it_changed_and_never_a_value(tmp_path,
                                                               caplog):
    _write_raw(tmp_path, "office")
    with caplog.at_level(logging.DEBUG):
        sp.write("office", {"ip": "192.0.2.55", "ssid": SSID},
                 password="another-invented-psk", ota_password=OTA,
                 directory=tmp_path)
    assert PSK not in caplog.text
    assert "another-invented-psk" not in caplog.text and OTA not in caplog.text
    assert "password" in caplog.text and "office" in caplog.text


def test_a_profile_round_trips_through_read_and_write(tmp_path):
    """Everything he can edit survives a save and a re-open, and the two
    secrets he cannot see survive with it."""
    _write_raw(tmp_path, "office")
    got = sp.read_public("office", tmp_path)
    got["nearest_m"] = 2.5
    got["range_m"] = 4.0
    got["notes"] = "on the shelf"
    sp.write("office", got, directory=tmp_path)
    again = sp.read_public("office", tmp_path)
    assert again["nearest_m"] == 2.5 and again["range_m"] == 4.0
    assert again["notes"] == "on the shelf"
    assert again["has_password"] is True and again["has_ota_password"] is True
    for field in sp.PUBLIC_FIELDS:
        if field in ("nearest_m", "range_m", "notes"):
            continue
        assert again[field] == got[field], field
    raw = json.loads((tmp_path / "office.json").read_text())
    assert raw["password"] == PSK and raw["ota_password"] == OTA


# ================================================ what a bad value is told
def _form(**over) -> dict:
    form = {"room": "den", "preset": "room", "nearest_m": "1.6",
            "range_m": "4.5", "timeout_s": "10", "still": True,
            "dhcp": False, "ip": IP, "gateway": "192.0.2.1",
            "subnet": "255.255.255.0", "mac": "", "ssid": SSID,
            "board": "esp32dev", "rx_pin": "GPIO16", "tx_pin": "GPIO17",
            "serial_port": "/dev/ttyUSB0", "notes": ""}
    form.update(over)
    return form


def test_a_good_form_validates_into_typed_values():
    values, why = sp.validate(_form())
    assert why == ""
    assert values["nearest_m"] == 1.6 and values["range_m"] == 4.5
    assert values["timeout_s"] == 10 and values["still"] is True
    assert set(values) <= set(sp.PUBLIC_FIELDS)
    assert "password" not in values and "ota_password" not in values


@pytest.mark.parametrize("ip,word", [
    ("192.0.2", "not an address"),
    ("kitchen.local", "not an address"),
    ("8.8.8.8", "not on a home network"),
    ("", "needs an address"),
])
def test_a_bad_address_is_refused_with_the_reason(ip, word):
    values, why = sp.validate(_form(ip=ip))
    assert values == {}
    assert word in why and "address" in why
    # a hostname is refused BY NAME: an mDNS answer is one poisoned packet
    # away from pointing this at someone else's box
    if ip == "kitchen.local":
        assert "hostname" in why


def test_dhcp_drops_the_address_requirement_but_not_the_check():
    values, why = sp.validate(_form(dhcp=True, ip=""))
    assert why == "" and values["ip"] == ""
    values, why = sp.validate(_form(dhcp=True, ip="8.8.8.8"))
    assert values == {} and "home network" in why


@pytest.mark.parametrize("pin,word", [
    ("GPIO99", "GPIO0 to GPIO39"),
    ("16", "GPIO"),
    ("", "GPIO"),
    ("GPIO 16", "GPIO"),
])
def test_a_bad_pin_number_is_refused_with_the_reason(pin, word):
    values, why = sp.validate(_form(rx_pin=pin))
    assert values == {}
    assert word in why and "rx pin" in why


def test_the_two_pins_may_not_be_the_same_pin():
    values, why = sp.validate(_form(rx_pin="GPIO17", tx_pin="GPIO17"))
    assert values == {} and "same pin" in why


@pytest.mark.parametrize("near,far,still,word", [
    ("0.5", "3.0", True, "blind zone"),
    ("1.0", "3.0", True, "STILL floor"),
    ("3.0", "3.0", True, "must be beyond"),
    ("3.0", "2.0", True, "must be beyond"),
    ("x", "3.0", True, "is not a number"),
    ("1.6", "", True, "is not a number"),
])
def test_a_bad_range_is_refused_with_the_reason(near, far, still, word):
    values, why = sp.validate(_form(nearest_m=near, range_m=far, still=still))
    assert values == {}, why
    assert word in why


def test_motion_only_is_what_makes_a_doorway_legal():
    """Inside the 1.5 m still floor the LD2410 is a PIR. The sheet does not
    repair that silently -- it refuses, and the tick is his answer."""
    values, why = sp.validate(_form(nearest_m="0.8", range_m="2.0",
                                    still=True))
    assert values == {} and "STILL floor" in why
    values, why = sp.validate(_form(nearest_m="0.8", range_m="2.0",
                                    still=False))
    assert why == "" and values["still"] is False


def test_a_range_past_the_module_is_warned_and_a_wall_reach_is_warned():
    values, why = sp.validate(_form(range_m="7.0"))
    assert why == "" and values["range_m"] == 7.0
    notes = sp.geometry_notes(values)
    assert any("ceiling" in n for n in notes)
    assert any("stud wall" in n for n in sp.geometry_notes(
        sp.validate(_form(range_m="5.5"))[0]))
    assert sp.geometry_notes(sp.validate(_form())[0]) == ()


def test_an_empty_ssid_is_refused_because_it_cannot_be_flashed():
    values, why = sp.validate(_form(ssid=""))
    assert values == {} and "network name" in why


def test_a_new_room_may_not_take_an_existing_profiles_name():
    values, why = sp.validate(_form(room="office"), known_rooms=("office",))
    assert values == {} and "already" in why
    # editing THAT room is not a clash with itself
    values, why = sp.validate(_form(room="office"), known_rooms=("office",),
                              editing="office")
    assert why == ""


@pytest.mark.parametrize("mac", ["aa:bb:cc", "zz:bb:cc:dd:ee:ff", "aabbccddeeff"])
def test_a_bad_mac_is_refused(mac):
    values, why = sp.validate(_form(mac=mac))
    assert values == {} and "MAC" in why


def test_a_blank_mac_is_fine_because_the_router_does_not_need_one():
    values, why = sp.validate(_form(mac=""))
    assert why == "" and values["mac"] == ""


def test_a_bad_timeout_is_refused():
    values, why = sp.validate(_form(timeout_s="soon"))
    assert values == {} and "absence delay" in why
    values, why = sp.validate(_form(timeout_s="0"))
    assert values == {} and "absence delay" in why


# ===================================================== the consequence lines
def test_the_gate_line_names_both_gates_in_gates_and_in_metres():
    """A gate number is not a distance he can stand at, so the line carries
    both. The METRES are the ones jarvis/sensorcheck.py's fault message
    uses -- a max gate of 5 means gates 0..5 are on, which is coverage to
    4.50 m. scripts/room_sensor.py's describe() says 3.75 for the same
    gate (it multiplies by the gate rather than the gate above it); the
    spelling that matters is the one the failure message will use, because
    the two must not disagree in front of him."""
    from jarvis import sensorcheck as sc
    values, _ = sp.validate(_form(nearest_m="2.0", range_m="3.5"))
    line = sp.gate_line(values)
    assert "move gate 5" in line and "4.50 m" in line
    assert "still gate 5" in line
    assert sc.gate_range(5) == "0.00-4.50 m"


def test_the_still_gate_never_starts_before_the_floor():
    """A still-capable mount never asks for less than gate 2: gates 0 and 1
    have no rest sensitivity at all. Asked of ``gates`` directly -- through
    ``validate`` the case is unreachable, because a range short enough to
    want gate 1 puts the nearest body inside the blind zone."""
    assert sp.gates({"range_m": 0.7, "still": True}) == (1, 2)
    assert sp.gates({"range_m": 0.7, "still": False}) == (1, 1)
    assert "still gate 2" in sp.gate_line({"range_m": 0.7, "still": True})


def test_the_gates_agree_with_the_script_and_with_sensorcheck():
    rs = _script()
    from jarvis import sensorcheck as sc
    for near, far, still in ((2.0, 3.5, True), (1.6, 4.5, True),
                             (0.8, 2.0, False), (1.6, 6.0, True)):
        values, why = sp.validate(_form(nearest_m=str(near),
                                        range_m=str(far), still=still))
        assert why == "", why
        prof = rs.Profile(room="pretend", nearest_m=near, range_m=far,
                          still=still)
        assert sp.gates(values) == (prof.max_move_gate, prof.max_still_gate)
        assert sp.gates(values) == sc.wanted_gates({"range_m": far,
                                                    "still": still})


def test_the_address_line_is_the_url_jarvis_will_actually_read():
    """Built through roomsensor.entity_path, so the entity NAME rule that
    made every poll a silent 404 cannot come back here."""
    from jarvis import roomsensor
    values, _ = sp.validate(_form(ip=IP))
    line = sp.address_line(values)
    assert line.endswith("http://%s/binary_sensor/Presence" % IP)
    assert roomsensor.entity_path("binary_sensor", "Presence") in line
    assert "Presence" in line and "/presence" not in line
    # DHCP with no address yet says so rather than printing "http:///"
    values, _ = sp.validate(_form(dhcp=True, ip=""))
    assert "the router" in sp.address_line(values)


def test_the_flash_command_is_this_checkouts_script_with_the_room_in_it():
    cmd = sp.flash_command("den")
    assert cmd.endswith("room_sensor.py flash den")
    assert str(REPO) in cmd


def test_the_presets_are_the_scripts_presets():
    rs = _script()
    assert tuple(sp.PRESETS) == tuple(rs.PRESETS)
    for key, preset in sp.PRESETS.items():
        other = rs.PRESETS[key]
        assert (preset.nearest_m, preset.range_m, preset.still,
                preset.timeout_s) == (other.nearest_m, other.range_m,
                                      other.still, other.timeout_s)
        assert preset.mount == other.mount and preset.blurb == other.blurb


# ============================================ one spelling of the path rule
def test_sensorcheck_reads_the_path_rule_from_here():
    """``jarvis/sensorcheck.py`` had its own copy of profile_dir /
    profile_path. Two copies of "where the profile lives" is how a boot
    check and a setup sheet come to disagree about which file they are
    talking about, so there is one, here, and the check imports it."""
    from jarvis import sensorcheck as sc
    assert sc.profile_dir is sp.profile_dir
    assert sc.profile_path is sp.profile_path
    # and read_profile still returns its deliberately narrow TWO fields:
    # that is a separate promise its own callers rest on.
    with pytest.MonkeyPatch.context() as mp:
        directory = Path(__file__).resolve().parent
        assert sc.read_profile("nothing-here", directory) is None
        del mp
