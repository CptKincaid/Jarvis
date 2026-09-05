"""The room-sensor DEVICE profile: read it without its secrets, write it
without losing them.

WHY THIS MODULE EXISTS. A room sensor is configured in three places, and
until now the UI could only reach the third:

1. ``~/.config/jarvis/room-sensors/<room>.json`` -- the DEVICE profile.
   Which network it joins, what address it takes, which pins the radar is
   wired to, how far it must see. Hand-edited JSON, owned by the ``Profile``
   dataclass in ``scripts/room_sensor.py``.
2. ``presence.rooms[]`` in assistant.json -- what the app actually polls.
3. ``zones.rooms[]`` -- the distance bands, which ``jarvis/ui/sensors_page``
   has edited since 2026-09-03.

Hunter, 2026-09-05: "i know i want a more ease of access on configuring
sensors through the UI and adding news ones that way". This is the store
the new sheet edits (1), reusing the rules ``scripts/room_sensor.py``
already argues rather than inventing a second set.

**THE PROFILE HOLDS TWO SECRETS AND THE UI ABOVE THIS LINE MUST NEVER SEE
EITHER.** ``password`` is his Wi-Fi PSK; ``ota_password`` is what stops a
neighbour reflashing the device. The sheet renders to PNG on a photo rig
whose frames are looked at, so "be careful in the UI" is not a design. The
design is a MODULE BOUNDARY, and it is three rules:

* ``read_public`` returns every field EXCEPT those two, plus ``has_password``
  and ``has_ota_password``. The secret NAMES are not keys of the returned
  dict at all -- there is nothing above this line to leak, render or log.
  This is the same narrow-return precedent ``jarvis/sensorcheck.py`` already
  states in its own words.
* ``write`` re-opens the file ITSELF and carries an unchanged secret across
  down here. **AN EMPTY BOX MEANS "LEAVE IT ALONE", NEVER "CLEAR IT"**, and
  the carried-across value never enters a variable the caller can reach. The
  two secrets are taken only through keyword arguments, so a secret
  smuggled into the public dict is ignored rather than written.
* nothing here logs a VALUE. The writer logs the room and the NAMES of the
  fields that changed, and a file that will not parse is reported as "the
  profile file is not valid JSON" with the decoder's message DROPPED -- a
  JSON error quotes the bytes it choked on, and those bytes are his PSK.

THE SSID IS NOT A SECRET AND IS SHOWN ON PURPOSE. The router broadcasts it,
and showing it is the only way he can answer "is this profile pointed at
the right network". Stated here so nobody later helpfully masks it and
breaks the one check that needs it.

WHAT IS NOT HERE. ``scripts/room_sensor.py`` still owns flashing, the
ESPHome render and the ``tune`` write, and it is deliberately
standalone-runnable, so it is not imported and not edited: this module is a
second implementation of the SHAPE, and ``tests/test_sensorprofile.py``
pins the field list, the defaults, the presets and the gate arithmetic to
the script's so the two cannot drift.

ONE ROOM NAME, FOUR CONFIG SHAPES. The profile filename, the
``presence.rooms`` key, the ``zones.rooms`` key and the ESPHome hostname are
each derived from the room name by a DIFFERENT rule in a different file.
They agree for "office" and "kitchen" and they do not agree for "Sam's
room". ``room_ok`` refuses a new name unless all of them land on the same
string, which is cheaper than discovering later that three of the four
point at a room that does not exist.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jarvis.config import PATHS
from jarvis.logs import get_logger
from jarvis.roomsensor import entity_path

log = get_logger("sensorprofile")


class ProfileError(RuntimeError):
    """A profile that cannot be read or written, said WITHOUT quoting the
    file. The message is safe to put on screen; the bytes are not."""


# --------------------------------------------------------------- the radar
# Every one of these is Hi-Link's, not ours, and every one is already argued
# in scripts/room_sensor.py and docs/room-sensor-bedroom.md. Repeated rather
# than imported because scripts/ is not an importable package; the parity
# test crosses the two, so a drift fails the suite.
GATE_M = 0.75                 # one LD2410 distance gate
MAX_GATE = 8                  # gates 0..8, so 6.0 m
MAX_RANGE_M = GATE_M * MAX_GATE
BLIND_M = 0.75                # nothing at all is detected inside this
STILL_FLOOR_M = 1.5           # no STILL detection inside this (gates 0, 1)
STILL_FLOOR_GATE = 2
DEFAULT_TIMEOUT_S = 5

# ------------------------------------------------------------- the network
DEFAULT_GATEWAY = "192.168.50.1"
DEFAULT_SUBNET = "255.255.255.0"

# THE ENTITY NAME IS "Presence" IN EVERY ROOM, and the ROOM goes in the
# DEVICE name. web_server v2 serves an entity at its NAME percent-encoded,
# so renaming it per room moves the endpoint and breaks the poll -- which is
# exactly the bug that made presence fail silently for a day. Every URL this
# lane builds goes through roomsensor.entity_path and nothing spells a path
# a second time.
PRESENCE_ENTITY = "Presence"
PRESENCE_PATH = entity_path("binary_sensor", PRESENCE_ENTITY)

# --------------------------------------------------------------- the shape
# EXACTLY scripts/room_sensor.py's Profile, in its order. Pinned by test.
PROFILE_FIELDS = ("room", "ssid", "password", "ota_password", "ip", "dhcp",
                  "mac", "gateway", "subnet", "board", "rx_pin", "tx_pin",
                  "preset", "mount_note", "nearest_m", "range_m", "still",
                  "timeout_s", "serial_port", "flashed", "notes")
SECRET_FIELDS = ("password", "ota_password")
PUBLIC_FIELDS = tuple(f for f in PROFILE_FIELDS if f not in SECRET_FIELDS)

DEFAULTS: Dict[str, Any] = {
    "room": "", "ssid": "", "password": "", "ota_password": "", "ip": "",
    "dhcp": False, "mac": "", "gateway": DEFAULT_GATEWAY,
    "subnet": DEFAULT_SUBNET, "board": "esp32dev", "rx_pin": "GPIO16",
    "tx_pin": "GPIO17", "preset": "room", "mount_note": "", "nearest_m": 1.6,
    "range_m": 4.5, "still": True, "timeout_s": DEFAULT_TIMEOUT_S,
    "serial_port": "/dev/ttyUSB0", "flashed": False, "notes": "",
}


# -------------------------------------------------------------- the mounts
@dataclass(frozen=True)
class Preset:
    """A preset is a MOUNT, not a room: where the thing physically sits, how
    far away the nearest body it must hold is, and how far the far edge of
    what it should see is."""
    key: str
    blurb: str
    mount: str
    nearest_m: float
    range_m: float
    still: bool = True
    timeout_s: int = DEFAULT_TIMEOUT_S


PRESETS = {p.key: p for p in (
    Preset(
        key="desk",
        blurb="he sits still at a desk; the radar watches the CHAIR from across the room",
        mount=("On a shelf or wall ACROSS the room from the chair, 1.0-1.5 m up, "
               "pointed at where his chest is when he is sitting. NOT on the "
               "monitor: at 0.6 m the chair is inside the 1.5 m still floor and "
               "he vanishes the moment he stops fidgeting -- the exact failure "
               "this sensor was bought to avoid."),
        nearest_m=2.0, range_m=3.5, timeout_s=10),
    Preset(
        key="bedroom",
        blurb="holds a sleeper: ceiling, flat, above the sternum",
        mount=("Ceiling, face parallel to it, pointing straight down, directly "
               "over your sternum (~0.6-0.7 m out from the headboard wall, on "
               "the bed centreline; over YOUR side on a shared bed). Ceiling "
               "height minus mattress-plus-0.2 m must be >= 1.55 m. "
               "See docs/room-sensor-bedroom.md -- the nightstand does not work."),
        nearest_m=1.6, range_m=2.5, timeout_s=30),
    Preset(
        key="room",
        blurb="general living space: is anybody in here",
        mount=("1.0-1.5 m up, on a wall or shelf, pointed ACROSS the room like a "
               "small speaker. Keep the far gate inside the room: the LD2410 "
               "reads through plasterboard and will hold presence ON from the "
               "corridor."),
        nearest_m=1.6, range_m=4.5, timeout_s=10),
    Preset(
        key="doorway",
        blurb="a threshold: motion only, and it says so",
        mount=("Beside or above the doorway, pointed along the walk-through. "
               "Anything this close is a MOTION sensor -- there is no still "
               "detection inside 1.5 m -- which is fine for a threshold and "
               "wrong for a chair."),
        nearest_m=0.8, range_m=2.0, still=False, timeout_s=3),
)}


# ---------------------------------------------------------------- the name
# Lower-case letters, digits and hyphens, starting on a letter or a digit.
# Deliberately narrower than any of the four slug rules rather than the
# intersection of them: a name that is already canonical is a name all four
# agree on, and the alternative is a box that quietly renames what he typed.
_NAME_RX = re.compile(r"^[a-z0-9][a-z0-9-]*$")
NAME_MAX = 24


def room_ok(name: Any) -> str:
    """"" when ``name`` is a room name every lane will spell the same way,
    else the sentence to show him.

    THE FOUR SHAPES. ``profile_path`` slugs on ``[^a-z0-9-]``,
    ``roomfabric.room_name`` keeps spaces and underscores,
    ``Profile.device_name`` slugs on ``[^a-z0-9]``, and ``zones._room_key``
    collapses and lower-cases. "office" and "kitchen" survive all four;
    "Sam's room" becomes four different strings, three of which name a room
    that does not exist.
    """
    text = str(name or "").strip()
    if not text:
        return "the room needs a name"
    if len(text) > NAME_MAX:
        return "the room name is longer than %d characters" % NAME_MAX
    if not _NAME_RX.match(text):
        return ("the room name must be lower-case letters, digits and "
                "hyphens (like \"back-room\"): %r is spelled differently by "
                "the profile file, the presence list and the device "
                "hostname, and three of the four would name a room that "
                "does not exist" % text)
    return ""


def device_name(room: str) -> str:
    """The ESPHome hostname and OTA target. The ROOM lives HERE, never in
    the entity name."""
    return "jarvis-" + re.sub(r"[^a-z0-9]+", "-", str(room).lower()).strip("-")


def presence_url(ip: str) -> str:
    """The URL Jarvis will poll, or "" when there is no address yet."""
    text = str(ip or "").strip()
    return "http://%s%s" % (text, PRESENCE_PATH) if text else ""


# ------------------------------------------------------------- where it is
def profile_dir() -> Path:
    """Where ``scripts/room_sensor.py`` keeps the profiles.

    Derived from ``PATHS.ASSISTANT_CONFIG`` rather than from ``Path.home()``
    so that the suite's redirect of the config directory carries this with
    it -- no test may read the real profiles, which hold his Wi-Fi PSK.
    """
    return PATHS.ASSISTANT_CONFIG.parent / "room-sensors"


def profile_path(room: str, directory: Optional[Path] = None) -> Path:
    """``<dir>/<room>.json``, slugged exactly as ``Profile.path_for``."""
    slug = re.sub(r"[^a-z0-9-]+", "-", str(room).lower()).strip("-")
    return Path(directory if directory is not None else profile_dir()) / \
        ("%s.json" % slug)


def list_rooms(directory: Optional[Path] = None) -> tuple:
    """Every room that HAS a profile, in name order. Never raises.

    This is the sheet's room picker, and it deliberately lists rooms the app
    is not polling: a half-finished sensor is exactly the one he opens this
    surface to finish.
    """
    where = Path(directory if directory is not None else profile_dir())
    try:
        names = sorted(p.stem for p in where.glob("*.json") if p.is_file())
    except OSError:
        log.debug("sensor profiles: %s is unreadable", where, exc_info=True)
        return ()
    return tuple(n for n in names if room_ok(n) == "")


# ------------------------------------------------------------- the reading
def _raw(room: str, directory: Optional[Path]) -> Optional[dict]:
    """The file as a dict, or None. NEVER quotes the file in an error."""
    path = profile_path(room, directory)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except (UnicodeDecodeError, ValueError):
        # DELIBERATELY no exc_info and no message: a JSON decoder quotes the
        # line it choked on, and that line may be the PSK.
        log.warning("sensor profiles: %s is not valid JSON", path)
        raise ProfileError("the profile file is not valid JSON: %s" % path)
    return data if isinstance(data, dict) else None


def read_public(room: str, directory: Optional[Path] = None) -> Optional[dict]:
    """Every profile field EXCEPT the two secrets, plus whether each is set.

    ``None`` for a missing file, a directory, a permission error or a file
    that is not a JSON object -- all of which mean "there is no profile
    here", which is what the sheet's ``+ NEW`` is for.

    The returned dict has no ``password`` key and no ``ota_password`` key at
    all. That is the whole secrets design: what the UI cannot hold, it
    cannot render, log or photograph.
    """
    try:
        data = _raw(room, directory)
    except ProfileError:
        return None
    if data is None:
        return None
    out = {field: data.get(field, DEFAULTS[field]) for field in PUBLIC_FIELDS}
    out["room"] = str(data.get("room") or room)
    out["has_password"] = bool(str(data.get("password") or "").strip())
    out["has_ota_password"] = bool(str(data.get("ota_password") or "").strip())
    return out


# ------------------------------------------------------------- the writing
def _write_private(path: Path, text: str) -> None:
    """Atomic write at 0600: temp file in the SAME directory (created 0600
    by mkstemp), fsync, os.replace, chmod to be sure.

    The same helper ``jarvis/assistant_config.py`` uses, and the reason a
    half-written profile cannot exist: the old bytes are replaced in one
    step or not at all. No backup copy is kept -- an extra copy of a PSK is
    a liability and os.replace is already atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=".room-sensor-", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def write(room: str, public: dict, *, password: Optional[str] = None,
          ota_password: Optional[str] = None,
          directory: Optional[Path] = None) -> Path:
    """Merge ``public`` into the room's profile and write it. Returns the path.

    **THE MERGE HAPPENS HERE, BELOW THE REDACTION LINE.** The stored secrets
    are read in this function and, unless a non-empty replacement was passed
    for one, written straight back out -- they are never returned, logged or
    handed to a caller. ``password=""`` and ``password=None`` both mean
    "leave it alone", because an empty box is a box he did not type in.

    ``public`` may only set ``PUBLIC_FIELDS``; anything else in it (a secret
    smuggled in by a later edit, ``has_password`` coming straight back from
    ``read_public``) is ignored rather than written.
    """
    why = room_ok(room)
    if why:
        raise ProfileError(why)
    path = profile_path(room, directory)
    stored = _raw(room, directory) or {}
    blob = dict(DEFAULTS)
    blob.update({k: v for k, v in stored.items() if k in PROFILE_FIELDS})
    blob["room"] = str(room)
    changed = []
    for field in PUBLIC_FIELDS:
        if field == "room" or field not in (public or {}):
            continue
        if blob.get(field) != public[field]:
            changed.append(field)
        blob[field] = public[field]
    for field, typed in (("password", password),
                         ("ota_password", ota_password)):
        if isinstance(typed, str) and typed.strip():
            blob[field] = typed
            changed.append(field)
        # else: the byte already on disk is carried forward untouched and
        # never leaves this function.
    _write_private(path, json.dumps({k: blob[k] for k in PROFILE_FIELDS},
                                    indent=2, sort_keys=True) + "\n")
    # NAMES, never values. "password" here means the field was replaced.
    log.info("sensor profiles: wrote %s (%s)", room,
             ", ".join(changed) if changed else "no field changed")
    return path


# ------------------------------------------------------------- the geometry
def gate_for_metres(metres: float) -> int:
    """The smallest gate whose band REACHES ``metres``.

    Gate N spans 0.75*N .. 0.75*(N+1), so 3.0 m needs gate 4 (3.0-3.75), not
    gate 3 which stops exactly at it. Clamped to 1..8. Byte for byte the
    rule in ``scripts/room_sensor.py`` and ``jarvis/sensorcheck.py``.
    """
    if metres <= 0:
        raise ValueError("a distance must be positive")
    return max(1, min(MAX_GATE, math.ceil(float(metres) / GATE_M)))


def gates(values: dict) -> Tuple[int, int]:
    """(max move gate, max still gate) for a validated form.

    The still gate can never start before 1.5 m, so a still-capable mount
    never asks for less than gate 2 -- ``Profile.max_still_gate``.
    """
    move = gate_for_metres(float(values.get("range_m") or 0.0))
    still = max(STILL_FLOOR_GATE, move) if values.get("still", True) else move
    return move, still


def gate_line(values: dict) -> str:
    """The consequence line: both gates, in GATES and in METRES.

    In metres because a gate number is not a distance he can stand at, and
    the whole 2026-09-03 failure was a device on gate 0 -- 75 cm -- looking
    like an empty room.

    THE METRES ARE ``sensorcheck.gate_range``'s, not ``describe()``'s. A max
    gate of 5 means gates 0..5 are enabled, so the device covers 0.00 to
    4.50 m; ``scripts/room_sensor.py:describe`` prints 3.75 for the same
    gate because it multiplies by the gate rather than by the gate above it.
    The two spellings differ by one band, and the one that has to be right
    is the one the FAULT message uses -- ``sensorcheck`` is what will one
    day tell him this device no longer matches this profile, and a setup
    sheet that had said 3.75 would look like it was talking about a
    different device.
    """
    move, still = gates(values)
    return ("move gate %d (covers to %.2f m), still gate %d (covers to %.2f m)"
            % (move, (move + 1) * GATE_M, still, (still + 1) * GATE_M))


def address_line(values: dict) -> str:
    """The exact URL Jarvis will read, or what is standing in the way."""
    url = presence_url(values.get("ip") or "")
    if url:
        return "Jarvis will read %s" % url
    if values.get("dhcp"):
        return ("no address yet: the router hands one out, then put it here "
                "so Jarvis knows where to look")
    return "no address yet"


def check_geometry(nearest_m: float, range_m: float, still: bool) -> list:
    """Every reason this mount will disappoint him, in the script's words.

    ``[(level, message)]``; "stop" is refused, "warn" is shown and allowed.
    Byte for byte ``scripts/room_sensor.py:check_geometry`` -- these are
    Hi-Link's numbers and the argument for them is in
    docs/room-sensor-bedroom.md, gate by gate.
    """
    out = []
    if nearest_m < BLIND_M:
        out.append(("stop", "%.2f m is inside the module's blind zone (%.2f m). "
                            "It will not see anything there at all."
                            % (nearest_m, BLIND_M)))
    elif nearest_m < STILL_FLOOR_M and still:
        out.append(("stop", "%.2f m is inside the 1.5 m STILL floor: gates 0 and 1 "
                            "have no rest sensitivity, so a body that stops moving "
                            "disappears. That is a PIR, which is what you were "
                            "avoiding. Move the sensor further from the chair, or "
                            "tick motion only if this really is a doorway."
                            % nearest_m))
    if range_m > MAX_RANGE_M:
        out.append(("warn", "%.2f m is past the module's %.1f m ceiling; the far "
                            "gate will be clamped to %d (%.2f m)."
                            % (range_m, MAX_RANGE_M, MAX_GATE, MAX_RANGE_M)))
    if range_m <= nearest_m:
        out.append(("stop", "the far edge (%.2f m) must be beyond the nearest body "
                            "(%.2f m)." % (range_m, nearest_m)))
    if range_m >= 5.0:
        out.append(("warn", "at %.2f m the far gate reaches through most stud "
                            "walls. If presence sticks ON when the room is empty, "
                            "lower the far edge until it stops seeing the "
                            "corridor." % range_m))
    return out


def geometry_notes(values: dict) -> tuple:
    """The WARNINGS for a validated form: shown, never refused."""
    if not values:
        return ()
    return tuple(msg for level, msg in check_geometry(
        float(values.get("nearest_m") or 0.0),
        float(values.get("range_m") or 0.0),
        bool(values.get("still", True))) if level == "warn")


# ----------------------------------------------------------- the validation
_PIN_RX = re.compile(r"^GPIO(\d{1,2})$")
_MAC_RX = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$", re.I)
MAX_PIN = 39                  # the ESP32's own ceiling


def _number(label: str, raw) -> tuple:
    text = str(raw).strip()
    try:
        num = float(text)
    except (TypeError, ValueError):
        return None, "%s: %r is not a number" % (label, text)
    if not math.isfinite(num):
        return None, "%s: %r is not a number" % (label, text)
    return num, ""


def _is_private_literal(host: str) -> bool:
    """A plain IP literal a home network can own.

    ``not is_global``, the same rule and for the same reason as
    ``jarvis/rooms.is_private_ip``: ``is_private`` is False for
    100.64.0.0/10 (carrier-grade NAT, which some home routers hand out) and
    True for 203.0.113.0/24, so the obvious test is wrong in both
    directions.
    """
    try:
        addr = ipaddress.ip_address(str(host or "").strip())
    except ValueError:
        return False
    if addr.is_unspecified or addr.is_multicast:
        return False
    return not addr.is_global


def _address(label: str, raw, *, required: bool) -> tuple:
    text = str(raw or "").strip()
    if not text:
        if required:
            return None, ("%s: Jarvis needs an address to poll. Give the "
                          "sensor one, or tick DHCP if the router reserves "
                          "it." % label)
        return "", ""
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return None, ("%s: %r is not an address. It must be a plain IPv4 "
                      "literal like 192.168.50.60 -- a hostname is refused "
                      "here, because an mDNS answer is one poisoned packet "
                      "away from pointing Jarvis at someone else's box."
                      % (label, text))
    if not _is_private_literal(text):
        return None, ("%s: %s is not on a home network. Jarvis reaches the "
                      "sensor over plain HTTP on the LAN, so it must be a "
                      "private address." % (label, text))
    return text, ""


def _pin(label: str, raw) -> tuple:
    text = str(raw or "").strip()
    hit = _PIN_RX.match(text)
    if not hit:
        return None, ("%s: %r is not a pin. Write it the way the board "
                      "prints it, GPIO16." % (label, text))
    num = int(hit.group(1))
    if num > MAX_PIN:
        return None, ("%s: %s is out of range -- this board has GPIO0 to "
                      "GPIO%d." % (label, text, MAX_PIN))
    return "GPIO%d" % num, ""


def validate(form: dict, *, known_rooms=(), editing: str = "") -> tuple:
    """(the profile fields to write, "") or ({}, why it is refused).

    PURE, and refused rather than repaired -- the same rule
    ``sensors_page._typed`` states: a person who has just typed something
    wrong should be told, not silently corrected. Nothing is written unless
    every field passes, so a refusal never leaves half a profile on disk.

    The refusals are the ones ``scripts/room_sensor.py`` already argues, in
    the same words, plus the ones a UI adds: a room name the four config
    shapes would spell differently, an address that is not a private
    literal, and a pin that is not a pin.
    """
    form = dict(form or {})
    room = str(form.get("room") or "").strip()
    why = room_ok(room)
    if why:
        return {}, why
    if room != str(editing or "").strip() and room in tuple(known_rooms or ()):
        return {}, ("there is already a profile for %r. Pick it from the row "
                    "above to edit it, or choose another name." % room)

    values: Dict[str, Any] = {"room": room}

    near, why = _number("nearest body", form.get("nearest_m"))
    if why:
        return {}, why
    far, why = _number("far edge", form.get("range_m"))
    if why:
        return {}, why
    delay, why = _number("absence delay", form.get("timeout_s"))
    if why:
        return {}, why
    if delay < 1 or delay > 3600:
        return {}, ("absence delay: %g s is outside 1 to 3600 s" % delay)
    still = bool(form.get("still", True))
    stops = [msg for level, msg in check_geometry(near, far, still)
             if level == "stop"]
    if stops:
        return {}, stops[0]
    values.update(nearest_m=round(near, 2), range_m=round(far, 2),
                  still=still, timeout_s=int(round(delay)))

    preset = str(form.get("preset") or "").strip()
    if preset not in PRESETS:
        return {}, "mount: %r is not one of %s" % (preset,
                                                   ", ".join(sorted(PRESETS)))
    values["preset"] = preset
    values["mount_note"] = str(form.get("mount_note")
                               or PRESETS[preset].mount)

    dhcp = bool(form.get("dhcp", False))
    values["dhcp"] = dhcp
    ip, why = _address("address", form.get("ip"), required=not dhcp)
    if why:
        return {}, why
    values["ip"] = ip
    if dhcp:
        # A DHCP RESERVATION and a manual_ip block are two answers to one
        # question, and render_yaml drops the block. The two fields go with
        # it rather than being written and ignored.
        values["gateway"] = DEFAULTS["gateway"]
        values["subnet"] = DEFAULTS["subnet"]
    else:
        gateway, why = _address("gateway", form.get("gateway"), required=True)
        if why:
            return {}, why
        values["gateway"] = gateway
        subnet = str(form.get("subnet") or "").strip()
        try:
            ipaddress.IPv4Address(subnet)
        except ValueError:
            return {}, ("subnet: %r is not a mask. It looks like "
                        "255.255.255.0." % subnet)
        values["subnet"] = subnet

    mac = str(form.get("mac") or "").strip()
    if mac and not _MAC_RX.match(mac):
        return {}, ("MAC: %r is not a MAC address. It has six pairs, like "
                    "a4:cf:12:9b:07:e1 -- read it with  room_sensor.py mac "
                    "<room>." % mac)
    values["mac"] = mac.lower()

    ssid = str(form.get("ssid") or "").strip()
    if not ssid:
        return {}, ("Wi-Fi: the network name is empty. A profile with no "
                    "network cannot be flashed onto anything.")
    values["ssid"] = ssid

    rx, why = _pin("rx pin", form.get("rx_pin"))
    if why:
        return {}, why
    tx, why = _pin("tx pin", form.get("tx_pin"))
    if why:
        return {}, why
    if rx == tx:
        return {}, ("rx pin and tx pin are the same pin (%s). The LD2410 is "
                    "wired CROSSED: its TX goes to the board's rx pin and its "
                    "RX to the tx pin." % rx)
    values["rx_pin"], values["tx_pin"] = rx, tx

    board = str(form.get("board") or "").strip()
    if not board:
        return {}, "board: the board id is empty (the usual one is esp32dev)"
    values["board"] = board
    values["serial_port"] = str(form.get("serial_port")
                                or DEFAULTS["serial_port"]).strip()
    values["notes"] = str(form.get("notes") or "")
    return values, ""


# ------------------------------------------------------------ the hand-over
def script_path() -> Path:
    """``scripts/room_sensor.py`` in THIS checkout, so a command the sheet
    prints is the script that goes with the code that printed it."""
    return Path(__file__).resolve().parent.parent / "scripts" / "room_sensor.py"


def flash_command(room: str) -> str:
    """The exact line to paste. The UI cannot flash a board -- that is USB
    serial, and this account is not in ``dialout`` -- so it hands him the
    command instead of pretending."""
    return "%s flash %s" % (script_path(), room)


def mac_command(room: str) -> str:
    return "%s mac %s" % (script_path(), room)


def tune_command(room: str) -> str:
    return "%s tune %s" % (script_path(), room)


__all__ = ["BLIND_M", "DEFAULTS", "DEFAULT_TIMEOUT_S", "GATE_M", "MAX_GATE",
           "MAX_RANGE_M", "NAME_MAX", "PRESENCE_ENTITY", "PRESENCE_PATH",
           "PRESETS", "PROFILE_FIELDS", "PUBLIC_FIELDS", "Preset",
           "ProfileError", "SECRET_FIELDS", "STILL_FLOOR_M", "address_line",
           "check_geometry", "device_name", "flash_command", "gate_for_metres",
           "gate_line", "gates", "geometry_notes", "list_rooms", "mac_command",
           "presence_url", "profile_dir", "profile_path", "read_public",
           "room_ok", "script_path", "tune_command", "validate", "write"]
