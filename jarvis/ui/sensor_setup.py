"""SENSOR SETUP: add and configure a room sensor without opening a terminal.

Hunter, 2026-09-05: "i know i want a more ease of access on configuring
sensors through the UI and adding news ones that way". Until today
``jarvis/ui/sensors_page.py`` could edit the distance BANDS and nothing
else; the device itself -- which network it joins, what address it takes,
how far it must see, which pins the radar is wired to -- was a hand-edited
``~/.config/jarvis/room-sensors/<room>.json``. This is the sheet that edits
it, placed over the SENSORS page with the same overlay mechanic that page
uses over the console's stage.

**WHAT IT DOES NOT DO, SAID ON SCREEN RATHER THAN LEFT TO BE DISCOVERED.**

* **It cannot flash a new board.** Flashing a fresh ESP32 is a USB serial
  operation and this account cannot open the port at all -- ``/dev/ttyUSB0``
  is ``root:dialout 0660`` and his groups do not include ``dialout``
  (measured 2026-09-03: ``open()`` -> EACCES). Nothing running as Jarvis can
  get past that. The sheet hands him the exact command with the profile
  filled in, and the one-line group fix.
* **It cannot compile firmware.** That is a multi-minute PlatformIO build,
  and a button that freezes the console for four minutes is worse than no
  button. Same answer: it hands over the command.
* **It cannot read a MAC off a board** (esptool, same serial port), so it
  accepts one he types.
* **It cannot show him a Wi-Fi password he has forgotten.** Not once, not to
  check. It says whether one is SET and lets him replace it. That is a
  deliberate refusal, not a missing feature.
* **It cannot rename a room.** The name is the profile filename, the
  ``presence.rooms`` key, the ``zones.rooms`` key AND the ESPHome hostname;
  renaming would silently orphan three of the four.
* **A change is not live until Jarvis restarts.**
  ``AssistantConfig.reload_if_changed`` still has no callers. What IS
  immediate is this page's own poll, and the difference is the most likely
  way this surface could mislead him, so ``saved_line`` says both halves in
  one breath rather than leaving the standing foot note to carry it.

**THE SECRETS ARE UNREACHABLE FROM HERE, STRUCTURALLY.** Nothing in this
module ever holds the Wi-Fi PSK or the OTA password:
``jarvis/sensorprofile.py`` is the only reader, it returns "set"/"not set"
booleans in place of the values, and the keep-the-old-one merge happens
inside its writer, below that line. The two entries are ``show="•"``,
created empty on every open, never bound to config and never read except by
the SAVE press -- the pattern ``jarvis/ui/views.py:_knightfall_row`` already
states in its own words. An empty box means "leave it alone".

And one trap worth naming out loud: ``AssistantConfig.get()`` does NOT
redact, so reading ``presence.rooms`` hands back a satellite's real
``web_server`` basic-auth password. This module edits only the radar fields
of an entry and carries ``username``/``password``/``sensors``/``lease_ttl_s``
through byte for byte, never into a widget and never into a message.

WHAT IS TESTED WITHOUT A DISPLAY. Everything above the widget, the same
split ``sensors_page`` and ``views`` make: the form, the merges, the
consequence lines, CHECK and TUNE against an injected transport.
"""
from __future__ import annotations

import time
import tkinter as tk
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jarvis import sensorprofile as sp
from jarvis.logs import get_logger
from jarvis.roomsensor import (DEFAULT_TIMEOUT_S, RoomSensor, _get_default,
                               _post_default, entity_path)
from jarvis.sensorcheck import (MOVE_GATE_ENTITY, STILL_GATE_ENTITY,
                                gate_range, read_gates)
from jarvis.ui import theme
from jarvis.ui.widgets import RoundButton, Toggle, px, ui_display, ui_mono

log = get_logger("sensorsetup")

# ------------------------------------------------------------------ the keys
OPTION_ROOMS = "presence.rooms"
OPTION_ENABLED = "presence.room_sensor_enabled"
OPTION_ZONES = "zones.rooms"

# The number entities TUNE writes, in the order it writes them. Names, not
# object_ids: web_server v2 serves an entity at its NAME percent-encoded, and
# building these from an object_id is the bug that made every poll and every
# tune write a silent 404 for a day. entity_path owns the rule.
DELAY_ENTITY = "Absence delay"
TUNE_SETTLE_S = 0.4          # the module writes its own NVM before it reads

CHECK_TIMEOUT_S = DEFAULT_TIMEOUT_S
BLIND_BAND_M = sp.BLIND_M    # a starter ladder begins at the blind zone

# The words. Kept here so a test can assert on them without a display.
SET_WORD, UNSET_WORD = "set", "not set"


# ------------------------------------------------------------ what he starts with
def form_for(room: str, directory: Optional[Path] = None) -> dict:
    """The sheet's starting values for a room, or a blank new one.

    **BOTH SECRET BOXES START EMPTY AND ALWAYS WILL.** Not the stored value,
    and not a row of bullets its length either -- a masked string 14 wide
    tells you the PSK is 14 characters. ``has_password`` /
    ``has_ota_password`` are the only thing said about them.
    """
    public = sp.read_public(room, directory) if str(room or "").strip() else None
    if public is None:
        public = {field: sp.DEFAULTS[field] for field in sp.PUBLIC_FIELDS}
        public["room"] = str(room or "").strip()
        public["mount_note"] = sp.PRESETS[sp.DEFAULTS["preset"]].mount
        public["has_password"] = False
        public["has_ota_password"] = False
    form = dict(public)
    form["nearest_m"] = "%.2f" % float(public.get("nearest_m") or 0.0)
    form["range_m"] = "%.2f" % float(public.get("range_m") or 0.0)
    form["timeout_s"] = "%d" % int(public.get("timeout_s") or 0)
    form["still"] = bool(public.get("still", True))
    form["dhcp"] = bool(public.get("dhcp", False))
    form["password"] = ""
    form["ota_password"] = ""
    return form


def secret_word(is_set: Any) -> str:
    """What is shown beside a secret box: a WORD, never a shape. Never a
    masked string of the real length, never a hash, never a last four."""
    return SET_WORD if bool(is_set) else UNSET_WORD


def preset_fill(key: str) -> dict:
    """The geometry a mount preset implies, as the sheet's own strings.

    A preset is a MOUNT, not a room: where the thing physically sits, how
    far the nearest body it must hold is, and how far the far edge is.
    ``still`` False means this position is honestly a motion sensor and the
    profile says so instead of pretending.
    """
    preset = sp.PRESETS.get(str(key or ""))
    if preset is None:
        return {}
    return {"nearest_m": "%.2f" % preset.nearest_m,
            "range_m": "%.2f" % preset.range_m,
            "timeout_s": "%d" % preset.timeout_s,
            "still": bool(preset.still),
            "mount_note": preset.mount}


# --------------------------------------------------------- where a room stands
@dataclass(frozen=True)
class RoomState:
    """The four things he needs to know about one room at a glance: is there
    a profile, is it in the app's room list, is the app actually polling it,
    and does it have a zone ladder."""
    room: str
    has_profile: bool = False
    listed: bool = False
    polled: bool = False
    has_ladder: bool = False
    primary: bool = False
    url: str = ""


def _opt(get_option: Optional[Callable], key: str, default=None):
    if not callable(get_option):
        return default
    try:
        value = get_option(key, default)
    except Exception:                     # noqa: BLE001 - config boundary
        log.debug("sensor setup: %s unreadable", key, exc_info=True)
        return default
    return default if value is None else value


def _entries(get_option: Optional[Callable]) -> list:
    raw = _opt(get_option, OPTION_ROOMS, None)
    return [e for e in raw if isinstance(e, dict)] \
        if isinstance(raw, (list, tuple)) else []


def _key(name: Any) -> str:
    from jarvis.roomfabric import room_name
    return room_name(name)


def room_states(get_option: Optional[Callable],
                directory: Optional[Path] = None) -> tuple:
    """Every room he has a profile for OR a config entry for, in name order.

    Rooms with no profile are listed on purpose: he may have written
    ``presence.rooms`` by hand, and a picker that hid those would look like
    the sheet had lost his room.
    """
    master = bool(_opt(get_option, OPTION_ENABLED, False))
    entries = {_key(e.get("name")): e for e in _entries(get_option)
               if _key(e.get("name"))}
    zoned = {_key(e.get("name")) for e in
             (_opt(get_option, OPTION_ZONES, None) or [])
             if isinstance(e, dict) and e.get("bands")}
    profiles = set(sp.list_rooms(directory))
    out = []
    for room in sorted(profiles | set(entries)):
        entry = entries.get(room) or {}
        url = str(entry.get("url") or "").strip()
        out.append(RoomState(
            room=room,
            has_profile=room in profiles,
            listed=room in entries,
            # THE MASTER SWITCH IS PART OF "polled". roomfabric.room_specs()
            # returns [] at its FIRST line when it is false, whatever the
            # address says, so a row that said "polled" there would be
            # showing him a room nothing reads.
            polled=bool(master and url and entry.get("enabled", True)),
            has_ladder=room in zoned,
            primary=bool(entry.get("primary", False)),
            url=url))
    return tuple(out)


def state_line(state: RoomState) -> str:
    """One room's row in the picker, in plain words."""
    parts = ["profile" if state.has_profile else "no profile"]
    if state.polled:
        parts.append("polled")
    elif state.listed:
        parts.append("in the room list but room sensing is switched off"
                     if state.url else "in the room list, no address")
    else:
        parts.append("not polled")
    parts.append("ladder" if state.has_ladder else "no ladder")
    if state.primary:
        parts.append("primary")
    return " · ".join(parts)


# ------------------------------------------------------------------- the merges
def merged_presence_rooms(raw, room: str, fields: dict) -> Optional[list]:
    """``presence.rooms`` with ONE room's radar fields changed. None when
    the config holds something that is not a list of rooms.

    THE WRITE IS A MERGE, NEVER A REBUILD. ``AssistantConfig`` REPLACES a
    list rather than merging it, so a save built from the widgets would
    silently delete every room this sheet was not showing -- including a
    leased satellite, whose entry carries a password this module must never
    touch. Only ``name``, ``label``, ``url``, ``power_url``, ``timeout_s``,
    ``primary`` and ``enabled`` are ever written; ``username``, ``password``,
    ``sensors``, ``lease_ttl_s`` and every key a later Jarvis adds are
    carried through byte for byte.

    The one field this touches on OTHER rooms is ``primary``, and only to
    clear it: ``roomfabric`` forces primary onto the first entry when none
    is marked, so two primaries is a silent coin toss about which room the
    camera and the speaker belong to.
    """
    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)):
        return None
    want = _key(room)
    fields = {k: v for k, v in (fields or {}).items()
              if k in ("label", "url", "power_url", "timeout_s", "primary",
                       "enabled")}
    primary = bool(fields.get("primary", False))
    out: List[Any] = []
    seen = False
    for entry in raw:
        if not isinstance(entry, dict):
            out.append(entry)             # not ours to judge; not ours to lose
            continue
        fresh = dict(entry)
        if _key(entry.get("name")) == want:
            fresh.update(fields)
            fresh["name"] = want
            seen = True
        elif primary and fresh.get("primary"):
            fresh["primary"] = False
        out.append(fresh)
    if not seen:
        fresh = {"name": want}
        fresh.update(fields)
        fresh.setdefault("enabled", True)
        out.append(fresh)
    return out


def starter_bands(values: dict) -> list:
    """One band, from the blind zone to the far edge, named after the room.

    A room with no ladder gets no verdict at all from ``jarvis/zones.py``,
    so a sensor added here would be polled and unplaceable. One honest band
    is better than three invented ones: he can split it on the SENSORS page
    in a second, and the bands are what that page is for.
    """
    return [{"name": str(values.get("room") or "room"),
             "near_m": BLIND_BAND_M,
             "far_m": round(float(values.get("range_m") or 0.0), 2)}]


def merged_zone_rooms(raw, room: str, bands: list) -> Optional[list]:
    """``zones.rooms`` with one room's ladder ADDED. Same merge discipline.

    ``camera_zone`` is "" -- a real answer meaning there is no lens in this
    room (jarvis/zones.py, ``ZoneMap.has_camera``). A new room does not get
    the built-in default handed to it, because that would claim a camera it
    does not have.
    """
    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)):
        return None
    want = _key(room)
    out = [dict(e) if isinstance(e, dict) else e for e in raw]
    for entry in out:
        if isinstance(entry, dict) and _key(entry.get("name")) == want:
            entry["bands"] = list(bands)
            return out
    out.append({"name": want, "enabled": True, "camera_zone": "",
                "bands": list(bands)})
    return out


# ------------------------------------------------------------- what a save writes
def config_writes(values: dict, get_option: Optional[Callable], *,
                  poll: bool, primary: bool, ladder: bool) -> tuple:
    """(the dotted keys to write, the sentences to show). Never raises.

    The PROFILE is written by ``sensorprofile.write`` and is not in here:
    these are the assistant.json edits that make the app read the room.
    """
    room = str(values.get("room") or "").strip()
    notes: List[str] = []
    edits: Dict[str, Any] = {}
    if not poll:
        notes.append("saved to the profile only — this room is not polled by "
                     "Jarvis. Tick “poll this room” to add it.")
        return edits, tuple(notes)
    url = sp.presence_url(values.get("ip"))
    if not url:
        # He saved a profile before the router handed the device an address.
        # The entry is written so the room exists, but roomfabric SKIPS a
        # room with no url, so nothing here may claim it is being read.
        notes.append("no address yet, so Jarvis cannot poll this room: put "
                     "the address the router gave it into this sheet and "
                     "save again")
    rooms = merged_presence_rooms(_opt(get_option, OPTION_ROOMS, None), room,
                                  {"url": url, "label": room,
                                   "primary": bool(primary),
                                   "enabled": True})
    if rooms is None:
        notes.append("%s in assistant.json is not a list of rooms — fix that "
                     "by hand before saving from here" % OPTION_ROOMS)
        return {}, tuple(notes)
    edits[OPTION_ROOMS] = rooms
    if not bool(_opt(get_option, OPTION_ENABLED, False)):
        # WITHOUT THIS, adding a first sensor from the UI leaves it unpolled
        # with no on-screen way to fix it. roomfabric.room_specs() returns []
        # at its first line when this is false, whatever the address says.
        edits[OPTION_ENABLED] = True
        notes.append("switched room sensing on — it was off, and nothing is "
                     "polled at all while it is")
    if ladder:
        zoned = {_key(e.get("name")) for e in
                 (_opt(get_option, OPTION_ZONES, None) or [])
                 if isinstance(e, dict) and e.get("bands")}
        if _key(room) in zoned:
            notes.append("%s already has a zone ladder; it was left alone"
                         % room)
        else:
            bands = merged_zone_rooms(_opt(get_option, OPTION_ZONES, None),
                                      room, starter_bands(values))
            if bands is None:
                notes.append("%s in assistant.json is not a list of rooms — "
                             "the ladder was not written" % OPTION_ZONES)
            else:
                edits[OPTION_ZONES] = bands
                notes.append("started a zone ladder for %s: one band, %.2f m "
                             "to %.2f m. Split it on this page."
                             % (room, BLIND_BAND_M,
                                float(values.get("range_m") or 0.0)))
    return edits, tuple(notes)


def saved_line(room: str, *, polled: bool) -> str:
    """The sentence at the point of the add, and it says BOTH halves.

    After a save this PAGE polls the new sensor immediately -- he can walk
    in front of it and watch the numbers move -- while the presence lane
    that decides whether Jarvis thinks he is home does not until the next
    start. That is genuinely useful and it is also the most likely way this
    surface could mislead him, so it is one sentence with both facts in it
    rather than a standing note at the bottom of the page.
    """
    if not polled:
        return "saved %s — the profile only; Jarvis does not poll this room" % room
    return ("saved %s — this page starts reading it now; Jarvis's own "
            "presence lane uses it at the next restart" % room)


# ------------------------------------------------------------------- the device
def _refuse_post(url: str, timeout: float) -> None:
    """CHECK is read-only as a property of the object, not as a promise in a
    comment -- the same backstop ``jarvis/sensorcheck.py`` uses."""
    raise RuntimeError("CHECK is read-only; it will not POST to %s" % url)


def check_device(values: dict, *, get: Optional[Callable] = None,
                 now: Callable[[], float] = time.monotonic) -> str:
    """Read presence, the round trip and the device's two gates. NEVER writes.

    A gate mismatch is named IN WORDS with both numbers and the metres they
    cover, because that is the measured 2026-09-03 failure: after a power
    cycle a sensor came back on gate 0 -- 75 cm -- and a blinded radar and an
    empty room are the same reading.
    """
    url = sp.presence_url(values.get("ip"))
    if not url:
        return "no address yet — nothing to check"
    sensor = RoomSensor(url, timeout_s=CHECK_TIMEOUT_S,
                        get=get or _get_default, post=_refuse_post,
                        fail_after=1)
    start = now()
    present = sensor.read()
    ms = (now() - start) * 1000.0
    if present is None:
        return ("no answer from %s — is it powered and on the Wi-Fi?"
                % (values.get("ip") or url))
    head = "presence %s, %.0f ms" % ("SOMEONE" if present else "empty", ms)
    move, still = read_gates(sensor)
    want_move, want_still = sp.gates(values)
    if move is None and still is None:
        return "%s · the device would not say what its gates are" % head
    got = "move %s / still %s" % ("?" if move is None else move,
                                  "?" if still is None else still)
    want = "profile wants %d / %d" % (want_move, want_still)
    agree = ((move is None or move == want_move)
             and (still is None or still == want_still))
    if agree:
        return "%s · gates %s, %s — ok" % (head, got, want)
    seen = move if move is not None else still
    return ("%s · gates %s, %s — the device does not match its profile: it "
            "covers %s, and presence past that reads as an EMPTY ROOM. Press "
            "TUNE." % (head, got, want, gate_range(seen)))


def _number_url(ip: str, entity: str, value=None) -> str:
    """``http://<ip>/number/<entity name, percent-encoded>[/set?value=]``."""
    base = "http://%s%s" % (str(ip).strip(), entity_path("number", entity))
    if value is None:
        return base
    return "%s/set?value=%s" % (base, urllib.parse.quote(str(value), safe=""))


def _number_value(body) -> Optional[float]:
    import json as _json
    try:
        data = _json.loads(body if isinstance(body, str)
                           else body.decode("utf-8", "replace"))
    except Exception:                     # noqa: BLE001 - any shape but ours
        return None
    if isinstance(data, dict):
        value = data.get("value")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None
    try:
        return float(data)
    except (TypeError, ValueError):
        return None


def tune_device(values: dict, *, get: Optional[Callable] = None,
                post: Optional[Callable] = None,
                sleep: Callable[[float], Any] = time.sleep) -> tuple:
    """Push the geometry into the radar's NVM and READ EVERY VALUE BACK.

    THE ONLY THING IN THIS LANE THAT WRITES TO HIS HARDWARE, and the sheet
    arms it on one press and fires on the second. ``jarvis/sensorcheck.py``
    refuses to ever write, for three stated reasons -- a stale profile can
    overwrite a good device, the readback lies shortly after power-up, and a
    boot-path write is unwatched. Two of those are answered by this being a
    deliberate press with the read-back printed; the third (a stale profile)
    is answered by the profile having just been saved from the form in front
    of him.

    The read-back is the whole point: an ESPHome rename would otherwise
    leave the write silently ignored and this claiming a tuning it never
    applied.
    """
    ip = str(values.get("ip") or "").strip()
    if not ip:
        return ("no address yet — nothing to tune",)
    getter = get or _get_default
    poster = post or _post_default
    move, still = sp.gates(values)
    wanted = ((DELAY_ENTITY, int(values.get("timeout_s") or 0)),
              (MOVE_GATE_ENTITY, move), (STILL_GATE_ENTITY, still))
    lines = []
    for entity, value in wanted:
        try:
            before = _number_value(getter(_number_url(ip, entity),
                                          CHECK_TIMEOUT_S))
            poster(_number_url(ip, entity, value), CHECK_TIMEOUT_S)
            sleep(TUNE_SETTLE_S)          # the module writes its own NVM
            after = _number_value(getter(_number_url(ip, entity),
                                         CHECK_TIMEOUT_S))
        except Exception as exc:          # noqa: BLE001 - every failure is a line
            log.debug("sensor setup: %s could not be tuned", entity,
                      exc_info=True)
            lines.append("%s — could not be set (%s)"
                         % (entity, type(exc).__name__))
            continue
        ok = after is not None and abs(after - float(value)) < 0.51
        lines.append("%s %s -> %s  %s"
                     % (entity, _short(before), _short(after),
                        "ok" if ok else "NOT APPLIED (asked %s)" % value))
    return tuple(lines)


def _short(value) -> str:
    if value is None:
        return "?"
    return "%d" % value if float(value).is_integer() else "%.2f" % value


# ------------------------------------------------------------- the hand-over
def copy_flash_command(room: str, run: Optional[Callable] = None) -> tuple:
    """(the command, did the clipboard hold it). It is PRINTED either way.

    Through ``enrolentry.to_clipboard``, which takes an injected runner, so
    no test and no probe ever touches his real X selection -- 2026-09-03 a
    diagnostic wrote a sentinel into his clipboard while he was pasting.
    True means "it was there when I looked", never "it will be there when
    you paste", which is why the command goes on screen as well.
    """
    from jarvis.enrolentry import to_clipboard
    text = sp.flash_command(room)
    try:
        ok = bool(to_clipboard(text, run=run))
    except Exception:                     # noqa: BLE001 - a convenience
        log.debug("sensor setup: the clipboard hook raised", exc_info=True)
        ok = False
    return text, ok


DIALOUT_FIX = ("this account is not in the dialout group, so nothing running "
               "as Jarvis can open the serial port. Once, in a terminal:  "
               "sudo usermod -aG dialout $USER  then log out and back in.")





# ============================================================== the widget
# Everything above this line is a pure function and is tested with no
# display, the same split jarvis/ui/sensors_page.py and jarvis/ui/views.py
# make. Everything below draws.

CAPTION_W = 11                # the label column, in characters
ENTRY_W = 16                  # the wide entries (ssid, address, notes)
NUM_W = 6                     # the numeric ones
WIRING = ("LD2410 TX -> the rx pin, RX -> the tx pin (CROSSED), VCC -> 5V, "
          "GND -> GND")
CANNOT = ("This page cannot flash a board: that is USB serial, and %s "
          "Press COPY FLASH COMMAND and run it in a terminal. It also cannot "
          "compile firmware or read a MAC off a board (same port), and it "
          "will never show you a Wi-Fi password — only whether one is set."
          % DIALOUT_FIX)
NOT_LIVE = ("Saved settings reach Jarvis's presence lane at the next restart; "
            "this page reads the sensor straight away.")
ARM_WORD = "TUNE writes to the radar itself. Press TUNE again to do it."


class SetupSheet(tk.Frame):
    """The sensor setup surface, placed over the SENSORS page.

    WHY IT IS A SHEET AND NOT A ROW IN THE PAGE. The room block's header
    already packs name + presence + distance + round trip at fixed widths,
    and the page's measured body budget is 690 px into a 719-px viewport at
    920x1440 -- 29 px spare. A fifth control on that row is the exact class
    of change that overflowed his real window on 2026-09-05 and was found in
    ten minutes. So this is its own overlay with its own scroll and its own
    pinned foot, opened by one button in the page's foot, where there is
    slack.

    THE FOOT IS PINNED TO THE BOTTOM and the body scrolls under it. A SAVE
    button that scrolls away is a save button he cannot find, and the same
    goes for the result line -- which is the only place this surface ever
    answers him.
    """

    def __init__(self, host, services=None, directory: Optional[Path] = None,
                 on_saved: Optional[Callable] = None,
                 get: Optional[Callable] = None,
                 post: Optional[Callable] = None,
                 run: Optional[Callable] = None,
                 sleep: Optional[Callable] = None,
                 box: Optional[Callable] = None):
        super().__init__(host, bg=theme.TV_BG)
        self.host = host
        self.services = services
        # WHERE IT LANDS. A callable returning place() kwargs -- the SENSORS
        # page hands over its own place_box, so the sheet covers exactly what
        # the page covers. Without it the sheet took the WHOLE window and the
        # wordmark, the status pill, the sensing badge and the tab row all
        # went with it (photographed at 1040x1760). This is a page-level
        # surface, not a takeover.
        self._box = box
        self._dir = directory
        self._on_saved = on_saved
        self._get, self._post_fn, self._run = get, post, run
        self._sleep = sleep
        self._open = False
        self._room = ""                   # "" is the + NEW row
        self._field: Dict[str, tk.Entry] = {}
        self._secret: Dict[str, tk.Entry] = {}
        self._secret_lbl: Dict[str, tk.Label] = {}
        self._preset = sp.DEFAULTS["preset"]
        self._preset_btn: Dict[str, Any] = {}
        self._buttons: Dict[str, Any] = {}      # the foot only
        self._board_btn = None
        # Every label that can be longer than the sheet. _rewrap walks THIS,
        # so a paragraph added later is wrapped without anyone remembering.
        self._wrapped: List[Any] = []
        self._canvas = None
        self._body = None
        self._thumb = None
        # TUNE is the one control that writes to his hardware, so it arms on
        # one press and fires on the second -- the RestartArm pattern
        # jarvis/ui/views.py already uses for the other irreversible button.
        from jarvis.ui.views import RestartArm
        self._arm = RestartArm(8.0)
        self._build()
        self.select("")

    # ------------------------------------------------------------ config
    def _get_option(self, key: str, default=None):
        fn = getattr(self.services, "get_option", None) if self.services else None
        if fn is None:
            return default
        try:
            value = fn(key, default)
        except Exception:                 # noqa: BLE001 - config boundary
            log.exception("sensor setup: get_option %s failed", key)
            return default
        return default if value is None else value

    # ------------------------------------------------------------- build
    def _build(self) -> None:
        bg = theme.TV_BG
        # ---- the foot FIRST and pinned to the bottom: SAVE and the result
        # line may never scroll away.
        foot = tk.Frame(self, bg=bg)
        foot.pack(side="bottom", fill="x", padx=theme.PAD,
                  pady=(theme.PAD_S, theme.PAD_S))
        # THE THREE SWITCHES LIVE IN THE FOOT, not at the end of the form.
        # They are what makes the whole exercise worth anything -- "poll this
        # room" is the difference between a saved file and a sensor Jarvis
        # reads -- and the body below still scrolls, by 534 px at his own
        # 1040x1760 window and 755 px at 920x1440 (measured on :94, S=2). It
        # was 598 and 819 with these three rows still in it, which is where a
        # control goes to be missed.
        switches = tk.Frame(foot, bg=bg)
        switches.pack(fill="x", pady=(0, px(6)))
        self._switches: Dict[str, Toggle] = {}
        for key, caption in (("poll", "poll this room"),
                             ("primary", "primary"),
                             ("ladder", "zone ladder")):
            tk.Label(switches, text=caption,
                     font=ui_display(theme.SIZE_CAPTION), fg=theme.MUTED,
                     bg=bg, bd=0, padx=0, pady=0).pack(side="left")
            toggle = Toggle(switches, value=False, bg=bg)
            toggle.pack(side="left", padx=(px(6), px(16)))
            self._switches[key] = toggle
        self._poll = self._switches["poll"]
        self._primary = self._switches["primary"]
        self._ladder = self._switches["ladder"]

        top = tk.Frame(foot, bg=bg)
        top.pack(fill="x")
        self._buttons["save"] = RoundButton(
            top, text="SAVE PROFILE", kind="default", size=theme.SIZE_CAPTION,
            bg=bg, pad_y=5, command=self.save)
        self._buttons["save"].pack(side="left")
        # EVERY BUTTON ON THIS SHEET IS OUTLINED (kind "default"): 8 of its
        # 11 were bare words on 2026-09-06 -- CLOSE, CHECK, TUNE, COPY FLASH
        # COMMAND, BOARD & WIRING and every room chip but the chosen one --
        # and to him a bare word is not a button. The sheet is holo's, so
        # classic's frozen ink is untouched.
        self._buttons["close"] = RoundButton(
            top, text="CLOSE", kind="default", size=theme.SIZE_CAPTION, bg=bg,
            pad_y=5, command=self.hide)
        self._buttons["close"].pack(side="right")
        # A SECOND ROW for the device buttons. One row of five was measured
        # at ~780 px into a 920-px window with the page's own padding either
        # side: it fitted at his 1040 and not at 920, which is the shape of
        # this morning's defect in the other direction.
        mid = tk.Frame(foot, bg=bg)
        mid.pack(fill="x", pady=(px(6), 0))
        for key, label, command in (("check", "CHECK", self.check),
                                    ("tune", "TUNE", self.tune),
                                    ("copy", "COPY FLASH COMMAND",
                                     self.copy_command)):
            self._buttons[key] = RoundButton(
                mid, text=label, kind="default", size=theme.SIZE_CAPTION,
                bg=bg, pad_y=5, command=command)
            self._buttons[key].pack(side="left", padx=(0, px(6)))
        self._result = tk.Label(foot, text=NOT_LIVE,
                                font=ui_display(theme.SIZE_CAPTION),
                                fg=theme.FAINT, bg=bg, anchor="w",
                                justify="left", bd=0, padx=0, pady=0)
        self._result.pack(fill="x", pady=(px(6), 0))
        self._wrapped.append(self._result)

        # ---- the scrolling body
        view = tk.Frame(self, bg=bg)
        view.pack(side="top", fill="both", expand=True)
        self._canvas = tk.Canvas(view, bg=bg, highlightthickness=0, bd=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._thumb = tk.Frame(self._canvas, bg=theme.CYAN_DIM,
                               width=max(2, px(2)))
        self._body = tk.Frame(self._canvas, bg=bg)
        self._body_win = self._canvas.create_window(0, 0, anchor="nw",
                                                    window=self._body)
        self._canvas.bind("<Configure>", lambda e: self._sync_view(), add=True)
        self._body.bind("<Configure>", lambda e: self._sync_view(), add=True)
        self._canvas.bind("<Enter>", self._grab_wheel, add=True)
        self._canvas.bind("<Leave>", self._drop_wheel, add=True)
        self._build_body(self._body, bg)
        self.bind("<Configure>", self._rewrap, add=True)

    def _build_body(self, body, bg) -> None:
        pad = theme.PAD
        tk.Label(body, text=theme.caption("ROOM SENSOR SETUP", surface=False),
                 font=ui_display(theme.SIZE_CAPTION, "semibold"),
                 fg=theme.FAINT, bg=bg, anchor="w", bd=0, padx=0,
                 pady=0).pack(fill="x", padx=pad, pady=(px(8), px(4)))
        self._picker = tk.Frame(body, bg=bg)
        self._picker.pack(fill="x", padx=pad)

        row = self._row(body, "room")
        self._field["room"] = self._entry(row, width=ENTRY_W)
        self._field["room"].pack(side="left")
        self._room_note = tk.Label(row, font=ui_display(theme.SIZE_CAPTION),
                                   fg=theme.FAINT, bg=bg, anchor="w", bd=0,
                                   padx=0, pady=0)
        self._room_note.pack(side="left", padx=(px(8), 0))

        # IN THE ORDER A NEW SENSOR NEEDS THEM. MEASURED 2026-09-06 at
        # 920x1440 on the + NEW form: the address, gateway/mask, mac, wi-fi,
        # password and ota code -- everything a new device actually needs
        # -- sat below the 572-px fold, under a 128-px mount paragraph and
        # the gates line; at his 1040 the ota box was hidden. Address and
        # network come first now; where the thing sits and how far it
        # watches come after, because those have defaults and a network
        # password does not.
        row = self._row(body, "address")
        self._field["ip"] = self._entry(row, width=ENTRY_W)
        self._field["ip"].pack(side="left")
        self._dhcp = Toggle(row, value=False, bg=bg)
        self._dhcp.pack(side="right")
        tk.Label(row, text="router reserves it",
                 font=ui_display(theme.SIZE_CAPTION), fg=theme.MUTED, bg=bg,
                 bd=0, padx=0, pady=0).pack(side="right", padx=(0, px(6)))
        self._dhcp.command = self._on_dhcp
        self._static = self._row(body, "gateway")
        self._field["gateway"] = self._entry(self._static, width=ENTRY_W)
        self._field["gateway"].pack(side="left")
        tk.Label(self._static, text="mask", font=ui_display(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=bg, bd=0, padx=0,
                 pady=0).pack(side="left", padx=px(6))
        self._field["subnet"] = self._entry(self._static, width=ENTRY_W)
        self._field["subnet"].pack(side="left")
        row = self._row(body, "mac")
        self._field["mac"] = self._entry(row, width=ENTRY_W)
        self._field["mac"].pack(side="left")
        tk.Label(row, text="only for a DHCP reservation",
                 font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT, bg=bg,
                 bd=0, padx=0, pady=0).pack(side="left", padx=(px(6), 0))
        self._addr = tk.Label(body, font=ui_mono(theme.SIZE_CAPTION),
                              fg=theme.CYAN_DIM, bg=bg, anchor="w",
                              justify="left", bd=0, padx=0, pady=0)
        self._addr.pack(fill="x", padx=pad, pady=(px(2), px(6)))
        self._wrapped.append(self._addr)

        row = self._row(body, "wi-fi")
        self._field["ssid"] = self._entry(row, width=ENTRY_W)
        self._field["ssid"].pack(side="left")
        # SHORT ENOUGH FOR ITS SLOT. The caption read "network name (not a
        # secret; the router shouts it)": 531 px in the 449-px slot beside
        # the box at 920x1440, cut at "the ro" (MEASURED 2026-09-06). On a
        # line of its own it fitted but cost 34 px, which put the ota box
        # 22 px under the fold on the + NEW form at 920; the shorter
        # caption keeps the row and the fold (ota box 517-554 in 566).
        tk.Label(row, text="network name (not a secret)",
                 font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT, bg=bg,
                 bd=0, padx=0, pady=0).pack(side="left", padx=(px(6), 0))
        self._secret_row(body, "password", "password")
        self._secret_row(body, "ota_password", "ota code")

        row = self._row(body, "mount")
        for key in sorted(sp.PRESETS):
            btn = RoundButton(row, text=key.upper(), kind="default",
                              size=theme.SIZE_CAPTION, bg=bg, pad_x=8,
                              pad_y=4,
                              command=lambda k=key: self.pick_preset(k))
            btn.pack(side="left", padx=(0, px(4)))
            self._preset_btn[key] = btn
        self._mount = tk.Label(body, font=ui_display(theme.SIZE_CAPTION),
                               fg=theme.MUTED, bg=bg, anchor="w",
                               justify="left", bd=0, padx=0, pady=0)
        self._mount.pack(fill="x", padx=pad, pady=(px(2), px(4)))
        self._wrapped.append(self._mount)

        row = self._row(body, "watches")
        self._field["nearest_m"] = self._entry(row, width=NUM_W)
        self._field["nearest_m"].pack(side="left")
        tk.Label(row, text="m to", font=ui_display(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=bg, bd=0, padx=0,
                 pady=0).pack(side="left", padx=px(4))
        self._field["range_m"] = self._entry(row, width=NUM_W)
        self._field["range_m"].pack(side="left")
        tk.Label(row, text="m", font=ui_display(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=bg, bd=0, padx=0,
                 pady=0).pack(side="left", padx=(px(4), px(12)))
        self._still = Toggle(row, value=True, bg=bg)
        self._still.pack(side="right")
        tk.Label(row, text="holds a still body",
                 font=ui_display(theme.SIZE_CAPTION), fg=theme.MUTED, bg=bg,
                 bd=0, padx=0, pady=0).pack(side="right", padx=(0, px(6)))
        self._still.command = lambda v: self._recompute()

        row = self._row(body, "absence")
        self._field["timeout_s"] = self._entry(row, width=NUM_W)
        self._field["timeout_s"].pack(side="left")
        tk.Label(row, text="s before the room reads empty",
                 font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT, bg=bg,
                 bd=0, padx=0, pady=0).pack(side="left", padx=(px(4), 0))
        self._gates = tk.Label(body, font=ui_mono(theme.SIZE_CAPTION),
                               fg=theme.CYAN_DIM, bg=bg, anchor="w", bd=0,
                               padx=0, pady=0)
        self._gates.pack(fill="x", padx=pad, pady=(px(2), px(6)))
        self._wrapped.append(self._gates)

        # ---- THE BOARD IS FOLDED AWAY. Pins, board id and USB port are set
        # once, at the bench, and never again; the rows he actually uses are
        # above them and this sheet already scrolls (measured 2026-09-05 on
        # :94, S=2: the body wants 1551 px into an 815-px viewport at his own
        # 1040x1760 window). Folding these four rows is 268 px of that back.
        fold = tk.Frame(body, bg=bg)
        fold.pack(fill="x", padx=pad, pady=(px(4), 0))
        # NOT in ``buttons()``: that is the FOOT, whose contract is that
        # every one of its controls is on screen without scrolling. This chip
        # is a body control and scrolls with the rows it folds.
        self._board_btn = RoundButton(
            fold, text="BOARD & WIRING", kind="default",
            size=theme.SIZE_CAPTION, bg=bg, pad_x=8, pad_y=4,
            command=self.toggle_board)
        self._board_btn.pack(side="left")
        tk.Label(fold, text="set once, at the bench",
                 font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT, bg=bg,
                 bd=0, padx=0, pady=0).pack(side="left", padx=(px(8), 0))
        self._board = tk.Frame(body, bg=bg)
        row = self._row(self._board, "board")
        self._field["board"] = self._entry(row, width=ENTRY_W)
        self._field["board"].pack(side="left")
        row = self._row(self._board, "pins")
        for key, caption in (("rx_pin", "rx"), ("tx_pin", "tx")):
            tk.Label(row, text=caption, font=ui_display(theme.SIZE_CAPTION),
                     fg=theme.FAINT, bg=bg, bd=0, padx=0,
                     pady=0).pack(side="left", padx=(0, px(4)))
            self._field[key] = self._entry(row, width=NUM_W + 2)
            self._field[key].pack(side="left", padx=(0, px(8)))
        row = self._row(self._board, "usb port")
        self._field["serial_port"] = self._entry(row, width=ENTRY_W)
        self._field["serial_port"].pack(side="left")
        wiring = tk.Label(self._board, text=WIRING,
                          font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT,
                          bg=bg, anchor="w", justify="left", bd=0, padx=0,
                          pady=0)
        wiring.pack(fill="x", padx=pad, pady=(px(2), px(6)))
        self._wrapped.append(wiring)

        row = self._row(body, "notes")
        self._field["notes"] = self._entry(row, width=ENTRY_W * 2)
        self._field["notes"].pack(side="left")

        tk.Frame(body, bg=theme.LINE, height=max(1, px(1))).pack(
            fill="x", padx=pad, pady=px(6))
        switch_note = tk.Label(
            body, text=("poll this room — Jarvis reads it for presence; "
                        "primary — where he is by default; zone ladder — one "
                        "band, so a reading in this room can be placed. The "
                        "three switches are beside SAVE."),
            font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT, bg=bg,
            anchor="w", justify="left", bd=0, padx=0, pady=0)
        switch_note.pack(fill="x", padx=pad, pady=(px(2), px(4)))
        self._wrapped.append(switch_note)
        self._cannot = tk.Label(body, text=CANNOT,
                                font=ui_display(theme.SIZE_CAPTION),
                                fg=theme.FAINT, bg=bg, anchor="w",
                                justify="left", bd=0, padx=0, pady=0)
        self._cannot.pack(fill="x", padx=pad, pady=(px(8), px(10)))
        self._wrapped.append(self._cannot)

    def _row(self, parent, caption: str) -> tk.Frame:
        bg = theme.TV_BG
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", padx=theme.PAD, pady=px(3))
        tk.Label(row, text=caption, font=ui_display(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=bg, anchor="w", width=CAPTION_W, bd=0,
                 padx=0, pady=0).pack(side="left")
        return row

    def _entry(self, parent, width: int = ENTRY_W) -> tk.Entry:
        # disabledbackground/foreground EXPLICITLY: Tk's defaults are a
        # light-grey box with grey text, which is a white hole in a dark
        # console (photographed on the locked room name at 1040x1760).
        return tk.Entry(parent, width=width, font=ui_mono(theme.SIZE_CAPTION),
                        fg=theme.INK, bg=theme.SURFACE,
                        disabledbackground=theme.SURFACE,
                        disabledforeground=theme.MUTED,
                        insertbackground=theme.CYAN, relief="flat",
                        highlightthickness=1, highlightbackground=theme.LINE,
                        highlightcolor=theme.CYAN_DIM, bd=0)

    def _secret_row(self, parent, key: str, caption: str) -> None:
        """A WRITE-ONLY row: masked, empty on every open, never bound to
        config and never read except by the SAVE press. Beside it, a WORD --
        never a masked string of the real length, which would tell anyone
        looking how long the PSK is."""
        bg = theme.TV_BG
        row = self._row(parent, caption)
        entry = tk.Entry(row, show="•", width=ENTRY_W,
                         font=ui_mono(theme.SIZE_CAPTION), fg=theme.INK,
                         bg=theme.SURFACE, insertbackground=theme.CYAN,
                         relief="flat", highlightthickness=1,
                         highlightbackground=theme.LINE,
                         highlightcolor=theme.CYAN_DIM, bd=0)
        entry.pack(side="left")
        self._secret[key] = entry
        self._secret_lbl[key] = tk.Label(
            row, font=ui_display(theme.SIZE_CAPTION), fg=theme.MUTED, bg=bg,
            anchor="w", bd=0, padx=0, pady=0)
        self._secret_lbl[key].pack(side="left", padx=(px(8), 0))
        tk.Label(row, text="— leave empty to keep it",
                 font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT, bg=bg,
                 bd=0, padx=0, pady=0).pack(side="left", padx=(px(6), 0))

    def _rewrap(self, event=None) -> None:
        """Wrap every paragraph to the sheet's width.

        BY WALKING, not by name. The first version listed four labels and
        the fifth -- the line explaining the three switches -- ran off the
        right edge mid-word at 1040x1760 ("...zone ladd"), photographed on
        :94. A list is a thing to forget to add to; ``_wrapped`` is the set
        of labels that were built as paragraphs, and every one of them is
        wrapped here.
        """
        width = max(px(200), int(self.winfo_width()) - 2 * theme.PAD)
        for label in self._wrapped:
            try:
                label.configure(wraplength=width)
            except Exception:             # noqa: BLE001 - torn down
                return

    # ------------------------------------------------------- the picker
    def _rebuild_picker(self) -> None:
        for child in list(self._picker.winfo_children()):
            child.destroy()
        bg = theme.TV_BG
        states = room_states(self._get_option, self._dir)
        row = tk.Frame(self._picker, bg=bg)
        row.pack(fill="x")
        for index, state in enumerate(states):
            if index and index % 4 == 0:
                row = tk.Frame(self._picker, bg=bg)
                row.pack(fill="x", pady=(px(3), 0))
            # Every chip is a button (outlined); the chosen one is accent.
            RoundButton(row, text=state.room.upper(),
                        kind="accent" if state.room == self._room
                        else "default",
                        size=theme.SIZE_CAPTION, bg=bg, pad_x=8, pad_y=4,
                        command=lambda r=state.room: self.select(r)
                        ).pack(side="left", padx=(0, px(4)))
        RoundButton(row, text="+ NEW",
                    kind="accent" if not self._room else "default",
                    size=theme.SIZE_CAPTION, bg=bg, pad_x=8, pad_y=4,
                    command=lambda: self.select("")).pack(side="left")
        # ONE ROOM PER LINE. Joined with the same "·" the facts inside a
        # room use, two rooms read as one run-on list of nine things --
        # photographed at 1040x1760 on :94.
        line = "\n".join("%s: %s" % (s.room, state_line(s)) for s in states)
        note = tk.Label(self._picker,
                        text=line or "no room sensors configured yet",
                        font=ui_display(theme.SIZE_CAPTION), fg=theme.FAINT,
                        bg=bg, anchor="w", justify="left", bd=0, padx=0,
                        pady=0)
        note.pack(fill="x", pady=(px(3), px(6)))
        self._wrapped.append(note)

    # -------------------------------------------------------- the form
    def select(self, room: str) -> None:
        """Show one room's profile, or a blank new one. Clears both secret
        boxes: they are never pre-filled, on any path."""
        self._room = str(room or "").strip()
        form = form_for(self._room, self._dir)
        for key, entry in self._field.items():
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, str(form.get(key, "")))
        # THE NAME IS LOCKED WHEN EDITING. It is the profile filename, the
        # presence.rooms key, the zones.rooms key AND the ESPHome hostname;
        # renaming here would silently orphan three of the four.
        if self._room:
            self._field["room"].configure(state="disabled")
            self._room_note.configure(text="a room cannot be renamed here")
        else:
            self._room_note.configure(text="lower-case letters, digits, "
                                           "hyphens")
            # A NEW room starts with the cursor in its name: ADD A SENSOR
            # lands here, and the first thing it asks for is the one box
            # that has no default.
            try:
                self._field["room"].focus_set()
            except Exception:             # noqa: BLE001 - torn down
                log.debug("sensor setup: the room box could not take focus",
                          exc_info=True)
        for key in ("password", "ota_password"):
            self._secret[key].delete(0, "end")
        self._still.set(bool(form.get("still", True)), animate=False)
        self._dhcp.set(bool(form.get("dhcp", False)), animate=False)
        state = {s.room: s for s in room_states(self._get_option,
                                                self._dir)}.get(self._room)
        self._poll.set(bool(state and state.polled), animate=False)
        self._primary.set(bool(state and state.primary), animate=False)
        self._ladder.set(False, animate=False)
        self._preset = str(form.get("preset") or sp.DEFAULTS["preset"])
        self._secrets_word(form.get("has_password"),
                           form.get("has_ota_password"))
        self._arm.expire()
        self._rebuild_picker()
        self._recompute()

    def _secrets_word(self, has_password, has_ota) -> None:
        self._secret_lbl["password"].configure(text=secret_word(has_password))
        self._secret_lbl["ota_password"].configure(text=secret_word(has_ota))

    def pick_preset(self, key: str) -> None:
        """A mount preset fills the geometry and shows its note. It does not
        write anything: SAVE does."""
        fill = preset_fill(key)
        if not fill:
            return
        self._preset = key
        for field in ("nearest_m", "range_m", "timeout_s"):
            self._field[field].delete(0, "end")
            self._field[field].insert(0, fill[field])
        self._still.set(bool(fill["still"]), animate=False)
        self._recompute()

    def _on_dhcp(self, value: bool) -> None:
        """A DHCP reservation and a manual_ip block are two answers to one
        question, and the renderer drops the block. So the two fields go with
        it rather than sitting there looking as though they are used."""
        self._recompute()

    def form(self) -> dict:
        """What is typed, as strings. The two secrets are NOT in here."""
        out = {key: entry.get() for key, entry in self._field.items()}
        out["room"] = out.get("room") or self._room
        out["preset"] = self._preset
        out["still"] = bool(self._still.get())
        out["dhcp"] = bool(self._dhcp.get())
        out["mount_note"] = sp.PRESETS[self._preset].mount \
            if self._preset in sp.PRESETS else ""
        return out

    def _recompute(self) -> None:
        """Repaint the consequence lines from what is typed right now."""
        for key, btn in self._preset_btn.items():
            btn.set_kind("accent" if key == self._preset else "default")
        if self._dhcp.get():
            self._static.pack_forget()
        else:
            self._static.pack(fill="x", padx=theme.PAD, pady=px(3),
                              after=self._field["ip"].master)
        self._mount.configure(text=sp.PRESETS[self._preset].mount
                              if self._preset in sp.PRESETS else "")
        values, why = sp.validate(self.form(),
                                  known_rooms=sp.list_rooms(self._dir),
                                  editing=self._room)
        if why:
            self._gates.configure(text="", fg=theme.FAINT)
            self._addr.configure(text=why, fg=theme.WARN)
            return
        notes = geometry_notes_text(values)
        self._gates.configure(text=sp.gate_line(values) + notes,
                              fg=theme.WARN if notes else theme.CYAN_DIM)
        self._addr.configure(text=sp.address_line(values), fg=theme.CYAN_DIM)

    # ------------------------------------------------------- the result
    def result(self) -> str:
        try:
            return str(self._result.cget("text"))
        except Exception:                 # noqa: BLE001 - torn down
            return ""

    def _say(self, text: str, tone: str = "faint") -> None:
        colour = {"err": theme.ERR, "warn": theme.WARN,
                  "ok": theme.FOCAL}.get(tone, theme.FAINT)
        try:
            self._result.configure(text=text, fg=colour)
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("sensor setup: the result line is gone", exc_info=True)

    # --------------------------------------------------------- the save
    def save(self) -> None:
        """Validate, write the profile, write the config, and say WHICH of
        those happened. Nothing is written unless everything validates."""
        values, why = sp.validate(self.form(),
                                  known_rooms=sp.list_rooms(self._dir),
                                  editing=self._room)
        if why:
            self._say(why, "err")
            return
        room = values["room"]
        try:
            sp.write(room, values,
                     password=self._secret["password"].get(),
                     ota_password=self._secret["ota_password"].get(),
                     directory=self._dir)
        except (OSError, sp.ProfileError) as exc:
            # The message names the FILE and never quotes it.
            self._say("the profile was not written: %s" % exc, "err")
            return
        for key in ("password", "ota_password"):
            self._secret[key].delete(0, "end")
        # POLLED means Jarvis will actually read it: the switch AND an
        # address. roomfabric skips a room with no url, so the switch alone
        # is not enough to say so.
        poll = bool(self._poll.get())
        reads = poll and bool(sp.presence_url(values.get("ip")))
        edits, notes = config_writes(values, self._get_option, poll=poll,
                                     primary=bool(self._primary.get()),
                                     ladder=bool(self._ladder.get()))
        failed = ()
        if edits:
            from jarvis.ui.sensors_page import write_options
            fn = getattr(self.services, "set_option", None) \
                if self.services else None
            if fn is None:
                notes = notes + ("assistant settings are not wired, so the "
                                 "room list was not changed",)
            else:
                failed = write_options(fn, edits)
        self._room = room
        public = sp.read_public(room, self._dir) or {}
        self._secrets_word(public.get("has_password"),
                           public.get("has_ota_password"))
        self._field["room"].configure(state="disabled")
        self._rebuild_picker()
        line = saved_line(room, polled=reads and not failed)
        if failed:
            line = ("NOT SAVED to assistant.json — %s would not write; the "
                    "profile itself was saved" % ", ".join(failed))
        self._say(" ".join((line,) + tuple(notes)),
                  "err" if failed else "ok")
        if self._on_saved and not failed:
            try:
                self._on_saved(room)
            except Exception:             # noqa: BLE001 - a callback
                log.exception("sensor setup: on_saved failed")

    # ------------------------------------------------------- the device
    def _work(self, fn, done) -> None:
        """Off the Tk thread and back. A synchronous read of a sensor that
        has gone away costs up to 3 s of frozen console, and that is exactly
        the state this sheet gets opened in."""
        import threading

        def run():
            try:
                out = fn()
            except Exception as exc:      # noqa: BLE001 - every failure is a line
                log.debug("sensor setup: the device call failed", exc_info=True)
                out = "that did not work (%s)" % type(exc).__name__
            try:
                self.after(0, lambda: done(out))
            except Exception:             # noqa: BLE001 - torn down
                log.debug("sensor setup: could not reach Tk", exc_info=True)
        threading.Thread(target=run, daemon=True,
                         name="sensor-setup").start()

    def check(self) -> None:
        values, why = sp.validate(self.form(),
                                  known_rooms=sp.list_rooms(self._dir),
                                  editing=self._room)
        if why:
            self._say(why, "err")
            return
        self._say("reading %s…" % values["ip"])
        self._work(lambda: check_device(values, get=self._get),
                   lambda text: self._say(text))

    def tune(self) -> None:
        """Arm, then fire. The only control here that writes to his radar."""
        values, why = sp.validate(self.form(),
                                  known_rooms=sp.list_rooms(self._dir),
                                  editing=self._room)
        if why:
            self._say(why, "err")
            return
        if self._arm.press() == "armed":
            self._say(ARM_WORD, "warn")
            return
        self._say("tuning %s…" % values["ip"])
        kw = {} if self._sleep is None else {"sleep": self._sleep}
        self._work(lambda: "\n".join(tune_device(values, get=self._get,
                                                 post=self._post_fn, **kw)),
                   lambda text: self._say(text))

    def copy_command(self) -> None:
        room = str(self.form().get("room") or "").strip()
        why = sp.room_ok(room)
        if why:
            self._say(why, "err")
            return
        text, ok = copy_flash_command(room, run=self._run)
        # PRINTED whether or not the clipboard took it: the X clipboard has
        # no storage and a live process owns the selection, so "it was there
        # when I looked" is the strongest thing that can honestly be said.
        self._say("%s   (%s)" % (text, "copied" if ok
                                 else "the clipboard would not take it — "
                                      "type it from here"))

    def toggle_board(self) -> None:
        """Show or hide the board rows. The FIELDS exist either way, so a
        save carries what they hold whether or not he has looked at them."""
        if self._board.winfo_ismapped():
            self._board.pack_forget()
        else:
            self._board.pack(fill="x", after=self._board_btn.master)
        self._sync_view()

    def switches(self) -> dict:
        """The three foot toggles, by name. Same contract as ``buttons``:
        none of them ever scrolls out of reach."""
        return dict(self._switches)

    def buttons(self) -> dict:
        """The FOOT buttons, whose contract is that none of them ever
        scrolls out of reach."""
        return dict(self._buttons)

    # ------------------------------------------------------- the scroll
    def _sync_view(self) -> None:
        if self._canvas is None:
            return
        try:
            self._canvas.itemconfigure(self._body_win,
                                       width=self._canvas.winfo_width())
            self._canvas.configure(
                scrollregion=self._canvas.bbox("all") or (0, 0, 0, 0))
        except Exception:                 # noqa: BLE001 - torn down
            return
        self._sync_thumb()

    def overflow_px(self) -> int:
        """How much of the body is below the fold (0 = it all fits)."""
        if self._canvas is None or self._body is None:
            return 0
        try:
            return max(0, self._body.winfo_reqheight()
                       - max(1, self._canvas.winfo_height()))
        except Exception:                 # noqa: BLE001 - torn down
            return 0

    def _sync_thumb(self) -> None:
        """A 2 px strip on the right edge, shown ONLY when there is
        something below the fold: a page that scrolls with no mark saying so
        is a page whose bottom rows he has no reason to look for."""
        over = self.overflow_px()
        try:
            if over <= 0:
                self._thumb.place_forget()
                self._canvas.yview_moveto(0.0)
                return
            height = max(1, self._canvas.winfo_height())
            span = max(px(20), int(height * height / (height + over)))
            top = self._canvas.canvasy(0)
            frac = max(0.0, min(1.0, top / float(over)))
            self._thumb.place(relx=1.0, x=-px(3), anchor="nw",
                              y=int(frac * (height - span)), height=span)
            # LIFTED. A Canvas stacks its children in creation order and the
            # body window is created after the thumb and fills the width, so
            # without this the mark is placed and invisible -- photographed
            # at 1040x1760, a sheet with the OTA row, the board fold and the
            # "what this cannot do" paragraph below the fold and nothing on
            # screen suggesting they were there.
            self._thumb.lift()
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("sensor setup: the thumb could not be placed",
                      exc_info=True)

    def _scroll(self, units: int) -> None:
        if self._canvas is None or self.overflow_px() <= 0:
            return
        self._canvas.yview_scroll(units, "units")
        self._sync_thumb()

    def _grab_wheel(self, _e=None) -> None:
        self._canvas.bind_all("<Button-4>", lambda e: self._scroll(-3))
        self._canvas.bind_all("<Button-5>", lambda e: self._scroll(3))

    def _drop_wheel(self, _e=None) -> None:
        try:
            self._canvas.unbind_all("<Button-4>")
            self._canvas.unbind_all("<Button-5>")
        except Exception:                 # noqa: BLE001 - torn down
            pass

    # ------------------------------------------------------ open / shut
    @property
    def is_open(self) -> bool:
        return self._open

    def show(self) -> None:
        if self._open:
            return
        self._open = True
        box = dict(x=0, y=0, relwidth=1.0, relheight=1.0)
        if callable(self._box):
            try:
                box = dict(self._box())
            except Exception:             # noqa: BLE001 - an unmapped page
                log.debug("sensor setup: the page's box is unreadable",
                          exc_info=True)
        self.place(in_=self.host, **box)
        self.lift()
        self.select(self._room)
        self._sync_view()

    def hide(self) -> None:
        if not self._open:
            return
        self._open = False
        self._drop_wheel()
        self._arm.expire()
        try:
            self.place_forget()
        except Exception:                 # noqa: BLE001 - a dead widget
            log.debug("sensor setup: unplace failed", exc_info=True)

    def toggle(self) -> None:
        self.hide() if self._open else self.show()


def geometry_notes_text(values: dict) -> str:
    notes = sp.geometry_notes(values)
    return ("  —  " + "  ".join(notes)) if notes else ""


__all__ = ["ARM_WORD", "BLIND_BAND_M", "CANNOT", "DIALOUT_FIX", "NOT_LIVE",
           "OPTION_ENABLED", "OPTION_ROOMS", "OPTION_ZONES", "RoomState",
           "SET_WORD", "SetupSheet", "UNSET_WORD", "WIRING", "check_device",
           "config_writes", "copy_flash_command", "form_for",
           "geometry_notes_text", "merged_presence_rooms", "merged_zone_rooms",
           "preset_fill", "room_states", "saved_line", "secret_word",
           "starter_bands", "state_line", "tune_device"]
