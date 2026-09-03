"""The pure half of scripts/room_sensor.py: gate arithmetic, the mount rules,
the URL the profile produces, and the rendered YAML.

No serial port, no network, no ESPHome, no camera. Every distance number
asserted here comes from Hi-Link's datasheet by way of
docs/room-sensor-bedroom.md §0, and the two that matter are the ones the
walkthrough never mentioned: nothing at all inside 0.75 m, and no STILL
detection inside 1.5 m. A mount that ignores the second one is a PIR, which
is what the whole feature exists to avoid -- so ``check_geometry`` refusing it
is the behaviour under test, not a nicety.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "room_sensor_script", REPO / "scripts" / "room_sensor.py")
rs = importlib.util.module_from_spec(SPEC)
# Registered BEFORE exec: @dataclass resolves its annotations through
# sys.modules[cls.__module__], so a module that is not in there yet blows up
# in dataclasses._is_type rather than anywhere near the code under test.
sys.modules[SPEC.name] = rs
SPEC.loader.exec_module(rs)


# ------------------------------------------------------------------- gates
@pytest.mark.parametrize("metres,gate", [
    (0.75, 1),      # exactly one band
    (1.0, 2),       # 0.75-1.5 reaches it
    (1.5, 2),
    (2.0, 3),
    (3.0, 4),       # gate 3 stops AT 3.0; reaching it needs gate 4
    (3.75, 5),
    (6.0, 8),
    (99.0, 8),      # clamped at the module's own ceiling
])
def test_a_gate_covers_seven_hundred_and_fifty_millimetres(metres, gate):
    assert rs.gate_for_metres(metres) == gate


def test_a_distance_must_be_positive():
    with pytest.raises(ValueError):
        rs.gate_for_metres(0)


def test_hi_links_worked_example_still_holds():
    """Gates 3 and 4 together cover 2.25-3.75 m -- the manufacturer's own
    example, and the reason gate N starts at 0.75*N rather than ending there.
    That is what puts the still floor at 1.5 m and not 0.75 m."""
    assert 3 * rs.GATE_M == pytest.approx(2.25)
    assert 5 * rs.GATE_M == pytest.approx(3.75)
    assert rs.STILL_FLOOR_M == pytest.approx(2 * rs.GATE_M)


# ------------------------------------------------------------------- mounts
def _levels(near, far, still=True):
    return [lvl for lvl, _ in rs.check_geometry(near, far, still)]


def test_a_chair_inside_the_still_floor_is_refused():
    """The failure this sensor was bought to avoid: 1.2 m from the chair, he
    stops fidgeting, and gates 0-1 have no rest sensitivity to hold him."""
    assert "stop" in _levels(1.2, 3.0, still=True)


def test_the_same_mount_is_allowed_when_it_admits_it_is_motion_only():
    assert "stop" not in _levels(1.2, 3.0, still=False)


def test_inside_the_blind_zone_nothing_helps():
    """Under 0.75 m the module detects nothing at all, so --motion-only does
    not rescue it either."""
    assert "stop" in _levels(0.5, 2.0, still=False)


def test_a_far_edge_behind_the_subject_is_refused():
    assert "stop" in _levels(2.0, 1.5, still=True)


def test_a_sane_desk_mount_passes():
    assert _levels(2.0, 3.5, still=True) == []


def test_a_long_range_warns_about_the_wall_but_does_not_refuse():
    lv = _levels(2.0, 5.5, still=True)
    assert lv and "stop" not in lv


# ------------------------------------------------------------------ profile
def test_the_room_names_the_device_and_never_the_entity():
    """web_server builds the URL path from the ENTITY name, so the room has to
    live in the device name or presence.room_sensor_url moves under Jarvis."""
    p = rs.Profile(room="Office", ip="192.168.50.60")
    assert p.device_name == "jarvis-office"
    assert p.presence_url == "http://192.168.50.60/binary_sensor/presence"


def test_object_ids_match_the_template_names():
    assert rs.object_id("Presence") == "presence"
    assert rs.object_id("Absence delay") == "absence_delay"
    assert rs.object_id("Max move gate") == "max_move_gate"
    assert rs.object_id("Max still gate") == "max_still_gate"


def test_a_still_capable_mount_never_asks_for_a_gate_below_the_still_floor():
    p = rs.Profile(room="nook", range_m=1.0, still=True)
    assert p.max_move_gate == 2 and p.max_still_gate == 2


def test_a_motion_only_mount_may_use_the_near_gates():
    p = rs.Profile(room="hall", range_m=1.0, still=False)
    assert p.max_still_gate == p.max_move_gate


def test_a_profile_round_trips_and_is_private(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "PROFILE_DIR", tmp_path / "room-sensors")
    p = rs.Profile(room="office", ssid="net", password="hunter2",
                   ip="192.168.50.60", preset="desk", nearest_m=2.0, range_m=3.5)
    path = p.save()
    assert path.stat().st_mode & 0o777 == 0o600, "the Wi-Fi PSK is in this file"
    again = rs.Profile.load("office")
    assert again.ssid == "net" and again.ip == "192.168.50.60"
    assert again.max_move_gate == 5


def test_a_missing_profile_says_what_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "PROFILE_DIR", tmp_path / "room-sensors")
    with pytest.raises(SystemExit):
        rs.Profile.load("nowhere")


def test_describe_never_prints_the_password():
    p = rs.Profile(room="office", ssid="net", password="hunter2",
                   ip="192.168.50.60")
    assert "hunter2" not in p.describe()


# ------------------------------------------------------------------- render
def test_the_rendered_yaml_carries_this_room_and_keeps_every_ruling():
    p = rs.Profile(room="office", ssid="net", password="pw", ota_password="ota",
                   ip="192.168.50.60")
    out = rs.render_yaml(p)
    assert 'device_name: "jarvis-office"' in out
    assert 'static_ip: "192.168.50.60"' in out
    assert "YOUR_WIFI_SSID" not in out, "a placeholder survived into the render"
    # The decisions the template documents must all still be in the render.
    assert "\napi:" not in out, "the api: block reboots the device every 15 min"
    assert "has_target:" in out and "name: Presence" in out
    # `throttle:` was REMOVED from the ld2410 component upstream; it must not
    # come back on the component, and the presence bit must stay unthrottled
    # because that delay is arrival latency Jarvis pays.
    assert "throttle: 250ms" not in out
    presence_block = out.split("binary_sensor:")[1].split("sensor:")[0]
    assert "throttle" not in presence_block
    assert "baud_rate: 256000" in out
    assert "tx_pin: GPIO17" in out and "rx_pin: GPIO16" in out


def test_a_wrover_moves_the_pins_off_the_psram():
    p = rs.Profile(room="office", ssid="n", password="p", ip="192.168.50.60",
                   board="esp-wrover-kit", rx_pin="GPIO32", tx_pin="GPIO33")
    out = rs.render_yaml(p)
    assert "rx_pin: GPIO32" in out and "tx_pin: GPIO33" in out
    assert "board: esp-wrover-kit" in out


def test_the_template_itself_keeps_its_placeholders():
    """The rendered copy goes to ~/.config with the PSK in it; the file in the
    repo must never learn his network."""
    text = rs.TEMPLATE.read_text()
    assert "YOUR_WIFI_SSID" in text and "YOUR_WIFI_PASSWORD" in text


def test_the_build_path_is_outside_the_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "BUILD_DIR", tmp_path / "build")
    p = rs.Profile(room="office", ip="192.168.50.60")
    assert rs.REPO not in rs.build_path(p).parents


def test_a_dhcp_reservation_drops_the_manual_ip_block():
    """A reservation and a manual_ip block are two answers to one question:
    the router hands out the reserved address and the device ignores it,
    which looks exactly like a reservation that never took."""
    p = rs.Profile(room="office", ssid="n", password="p", ip="192.168.50.60",
                   dhcp=True, mac="5c:01:3b:be:ef:78")
    out = rs.render_yaml(p)
    assert "manual_ip:" not in out
    assert "static_ip:" not in out.split("wifi:")[1].split("captive_portal")[0]
    assert "ap:" in out and "captive_portal:" in out, "the rest of wifi: survived"
    assert "ssid: ${wifi_ssid}" in out


def test_a_static_address_keeps_the_block():
    p = rs.Profile(room="office", ssid="n", password="p", ip="192.168.50.60")
    out = rs.render_yaml(p)
    assert "manual_ip:" in out and "static_ip: ${static_ip}" in out


def test_describe_names_the_reservation_it_depends_on():
    p = rs.Profile(room="office", ip="192.168.50.60", dhcp=True,
                   mac="5c:01:3b:be:ef:78")
    assert "5c:01:3b:be:ef:78" in p.describe()


# ------------------------------------------------------------------- presets
def test_every_preset_is_a_mount_that_passes_its_own_rules():
    for key, preset in rs.PRESETS.items():
        levels = _levels(preset.nearest_m, preset.range_m, preset.still)
        assert "stop" not in levels, "%s: %s" % (key, levels)


def test_the_doorway_preset_admits_it_is_motion_only():
    assert rs.PRESETS["doorway"].still is False
    assert rs.PRESETS["desk"].still is True
    assert rs.PRESETS["desk"].nearest_m >= rs.STILL_FLOOR_M


def test_install_config_is_a_no_op_without_yes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rs, "PROFILE_DIR", tmp_path / "room-sensors")
    monkeypatch.setattr(rs, "ASSISTANT_JSON", tmp_path / "assistant.json")
    rs.Profile(room="office", ip="192.168.50.60").save()
    (tmp_path / "assistant.json").write_text(json.dumps({"presence": {"enabled": True}}))
    args = type("A", (), {"room": "office", "yes": False, "fast": False})()
    assert rs.cmd_install_config(args) == 0
    assert "nothing written" in capsys.readouterr().out
    assert json.loads((tmp_path / "assistant.json").read_text())["presence"] == {"enabled": True}
