"""The SENSORS page: what the radar thinks, what the camera thinks, and
which of them won.

He asked for "a small GUI" inside the console -- not a browser tab and not
a second window -- to watch presence while he tunes it::

    JARVIS                            [READY] [SENSING]
    ---------------------------------------------------
     [ CHAT ]  [ SENSORS ]      <- jarvis/ui/tab_strip.py
    ---------------------------------------------------
     office        * PRESENT 1.4 m
       camera      * FACE  hunterp
       verdict     AT THE DESK
     desk band  [0.8]---#---[1.8] m
     room band  [1.8]--#----[4.5] m
     camera overrules radar   [x]
     [SAVE]                        updated 2 s ago

The tab row is a MODULE OF ITS OWN, one row under the wordmark, and this
page draws none of it: 2026-09-03, his words, "make a little tab to click
thats underneath jarvis, that can be the area that has multiple tabs".
The strip owns show()/hide(), which is also what keeps "polls only while
it is on screen" true now that flipping in and out is one click -- and
STANDBY goes through the same seam: the console hides the row and selects
CHAT (main_window._set_tabs_hidden), so a page left open when he walks
away is shut rather than polling behind the clock (measured on the photo
rig, state 28: 0 requests across the standby dwell).

WHAT THE HARDWARE CAN ACTUALLY TELL HIM, because the page must not imply
more. The office sensor is an HLK-LD2410C on an ESP32 behind ESPHome's
``web_server`` v2. It reports a presence bit, a moving/still bit and a
DISTANCE in centimetres. It reports no angle and no x/y, so it cannot tell
"at the desk" from "at the bookshelf" when both sit at the same range --
that is structural, not a tuning problem. The zone model is therefore
distance BANDS, and the camera overrules: if the eye recognises him in its
cone he is at the desk whatever the radar's range says (his words: "camera
recognition overrules sensor detection since he can literaly see me at my
desk"). The radar answers "someone is in the room" and, coarsely, "how far".

MEASURED on the live office radar, 2026-09-03 (do not re-derive):
presence ON in 63 of 63 samples over 45 s while he walked the room, tracking
0.83 m to 3.6 m; 0 of 78 in an empty room; poll round trip min 46 / median
62 / p90 154 / max 1186 ms with the gates tuned to 4.5 m of coverage. Those
numbers are why ``DEFAULT_ROOM_BAND`` tops out at 4.5 and why this page
polls on ``roomsensor.DEFAULT_TIMEOUT_S`` (3.0 s) rather than the 1.5 s in
``presence.room_sensor_timeout_s``: a diagnostic surface exists to SHOW him
the 1186 ms sample, and a 1.5 s timeout would render it as a fault.

FIVE RULES THIS FILE HOLDS, all of them learned elsewhere in the tree:

* **None is NO OPINION, never an empty room.** ``RoomSensor.read()``
  returns True / False / None and the third one means the leg abstained --
  offline mode, a tripped breaker, a 404, HTML from the wrong URL. A page
  that painted that as "nobody there" would be teaching him to trust the
  exact lie that makes Jarvis go quiet on him. Every unknown here renders
  as NO OPINION with a plain-language reason under it.
* **The page never hands ``SensingPolicy`` to its own sensors.**
  ``SensingPolicy.attach`` replaces BY NAME (jarvis/rooms.py argues this at
  length), so a second sensor attaching as "radar" would silently take the
  curfew away from the app's real one. ``SensorPoller`` asks
  ``policy.allowed(RADAR)`` itself and, when the answer is no, issues no
  request at all -- the assertion the test makes is that nothing was sent,
  because "the readings are ignored" is a weaker promise than "the radar
  was not polled".
* **The camera arrives as NUMBERS.** ``PreviewWorker.status()`` is built
  from ``PreviewShot.numbers_only()``, which excludes the image by name.
  This module never touches a frame, never names a device node and never
  imports the vision lane; ``camera_view`` reads a dict of counts, a name
  and a cosine score. Hunter, 2026-09-03: "i dont want you to have access
  to private or sensitive info or access to camera or mic, it all feeds
  through jarvis first."
* **Saving goes through ``services.set_option``.** That lands in
  ``AssistantConfig.set`` -> ``save()``, which is the one atomic 0600
  writer (mkstemp in the same directory, fsync, os.replace, chmod). This
  file never opens assistant.json.
* **An edit is not live.** ``AssistantConfig.reload_if_changed`` has no
  callers, so a band he changes here takes effect at the next start. The
  page SAYS so (``RESTART_NOTE``) instead of pretending otherwise.
* **The fused verdict is DISPLAY-ONLY.** It combines a recognised face
  with a range, and the capability rule (tests/test_campreview.py) is that
  identity may remove capability or add a name and must never GRANT one.
  Structurally it cannot here: ``main_window`` is the only importer of this
  module, the page publishes no event, and nothing reads ``fuse()``. If the
  arrival greeting ever wants this verdict, ``fuse`` moves to a module of
  its own with its own argument -- it does not get read off a Tk page.

Everything above the widget is a pure function, so the whole surface is
tested with no display (tests/test_ui_sensors_page.py) -- the same split
jarvis/ui/views.py and jarvis/ui/console_mode.py make.
"""
from __future__ import annotations

import json
import math
import threading
import time
import tkinter as tk
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from jarvis.logs import get_logger
from jarvis.roomsensor import DEFAULT_TIMEOUT_S, RoomSensor, entity_path
from jarvis.sensing import RADAR
from jarvis.ui import theme
from jarvis.ui.widgets import (RoundButton, Toggle, px, ui_display, ui_font,
                               ui_mono)

log = get_logger("ui.sensors_page")

# ---------------------------------------------------------------- the words
DASH = "—"                    # every unknown scalar; never "0" and never ""

ZONE_DESK = "desk"
ZONE_ROOM = "room"
ZONE_AWAY = "away"
ZONE_UNKNOWN = "unknown"
ZONE_WORDS = {
    ZONE_DESK: "AT THE DESK",
    ZONE_ROOM: "IN THE ROOM",
    ZONE_AWAY: "NOBODY THERE",
    # Deliberately not "EMPTY": this is the state where BOTH legs abstained,
    # and the whole point of the None contract is that it is not absence.
    ZONE_UNKNOWN: "NO OPINION",
}

# Tone KEYS, not colours: a colour resolved out here would freeze the
# import-time look (tests/test_theme_look.py scans this file for exactly
# that). tone_color() resolves one at paint time.
TONE_OK, TONE_WARN, TONE_ERR, TONE_MUTED, TONE_FAINT = (
    "ok", "warn", "err", "muted", "faint")
TONES = (TONE_OK, TONE_WARN, TONE_ERR, TONE_MUTED, TONE_FAINT)

# The standing caption, and the one shown after a write. They are separate
# because the standing one sat under an unpressed SAVE button reading
# "saved to assistant.json" -- which is a claim, not a caption.
RESTART_NOTE = "edits apply at the next Jarvis restart — nothing reloads the config"
SAVED_NOTE = "saved — restart Jarvis to apply it"
NOT_WIRED_NOTE = "assistant settings not wired"

# ------------------------------------------------------------- the geometry
# The distance entity, addressed by its NAME exactly as the presence one is:
# web_server v2 serves an entity at its name, percent-encoded, NOT at the
# snake_case object_id. Getting that wrong is a silent 404 -- it cost this
# tree every poll and every tune write until 00d5b7e. entity_path() owns the
# rule; nothing here spells a path a second time.
DISTANCE_ENTITY = "Detection distance"
DISTANCE_PATH = entity_path("sensor", DISTANCE_ENTITY)

# The bands, in metres. The desk band's floor sits just outside the LD2410's
# 0.75 m blind zone and its ceiling just inside the 4.5 m the tuned gates
# reach; both numbers come from scripts/room_sensor.py (GATE_M 0.75, gates
# 0..8) and from the 09-03 walk measured in the module docstring.
DEFAULT_DESK_BAND = (0.8, 1.8)
DEFAULT_ROOM_BAND = (1.8, 4.5)
MAX_BAND_M = 6.0              # 8 gates x 0.75 m: the module's own ceiling
BLIND_M = 0.75                # nothing at all is detected inside this
STILL_FLOOR_M = 1.5           # no STILL target inside this (gates 0 and 1)

OPTION_DESK_BAND = "presence.desk_band_m"
OPTION_ROOM_BAND = "presence.room_band_m"
OPTION_CAMERA_OVERRULES = "presence.camera_overrules"

POLL_S = 1.0                  # while the page is on screen, and only then
POLL_THREAD_NAME = "sensors-page"
# How long stop() waits for the loop to end. Short on purpose: it is called
# from hide(), which is the Tk thread, and the loop is almost always parked
# in stop.wait() and returns at once. Correctness does not rest on the join
# -- it rests on the per-run flag, which no later start() can clear.
JOIN_S = 0.25
POST_FAILS_MAX = 2            # consecutive failed hops to the Tk thread
AGE_TICK_MS = 1000            # how often the "updated N s ago" line redraws


@dataclass(frozen=True)
class Bands:
    """The two distance bands, in metres, already repaired."""
    desk_lo: float
    desk_hi: float
    room_lo: float
    room_hi: float


def tone_color(tone: str) -> str:
    """A tone key -> the theme colour, read NOW so a look switch lands."""
    return {TONE_OK: theme.FOCAL, TONE_WARN: theme.WARN, TONE_ERR: theme.ERR,
            TONE_MUTED: theme.MUTED, TONE_FAINT: theme.FAINT}.get(
                tone, theme.MUTED)


# ------------------------------------------------------------ the transport
def distance_url(value: str) -> str:
    """The Detection-distance endpoint for a device address, or "".

    The path is REPLACED rather than appended to, because
    ``presence.room_sensor_url`` may already carry the presence entity
    (normalize_url appends that one when the address has no path) and the
    distance is a different entity on the same device. The scheme stays
    required for the same reason it is next door: a bare "192.168.50.51"
    is a typo, and guessing at it would fail once a poll as a timeout
    instead of once at startup as a line he can read.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return ""
    return urllib.parse.urlunsplit(
        parts._replace(path=DISTANCE_PATH, query="", fragment=""))


def parse_distance_cm(body: Any) -> Optional[float]:
    """An ESPHome sensor body -> centimetres, or None for "I cannot tell".

    Generous about the shape for the same reason ``roomsensor.parse_state``
    is -- he may point this at a Home Assistant template later -- but strict
    about what counts as a distance. ``{"value": null}`` (no target),
    a NaN, a negative number and ZERO are all None: 0 cm is what the module
    reports with nothing in the beam, and it is inside the 0.75 m blind
    zone besides, so rendering it as "0.0 m" would draw a man standing on
    the sensor.
    """
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, dict):
        return _distance_from_mapping(body)
    if isinstance(body, bool):
        return None                      # a flag is not a range
    if isinstance(body, (int, float)):
        return _finite_cm(body)
    text = str(body or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return _finite_cm(_number_in(text))
    if isinstance(data, dict):
        return _distance_from_mapping(data)
    if isinstance(data, bool) or not isinstance(data, (int, float)):
        return None
    return _finite_cm(data)


def _distance_from_mapping(data: dict) -> Optional[float]:
    # "value" is the typed field and wins; "state" is ESPHome's display
    # string ("142 cm") and is the fallback for a firmware that omits it.
    if "value" in data:
        got = data["value"]
        if isinstance(got, (int, float)) and not isinstance(got, bool):
            return _finite_cm(got)
    if "state" in data:
        return _finite_cm(_number_in(str(data["state"])))
    return None


def _number_in(text: str) -> Optional[float]:
    """The leading number in "142 cm", or None for "unknown" / "nan"."""
    head = text.strip().split()
    if not head:
        return None
    try:
        return float(head[0])
    except (TypeError, ValueError):
        return None


def _finite_cm(value) -> Optional[float]:
    try:
        cm = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cm) or cm <= 0.0:
        return None
    return cm


# ----------------------------------------------------------------- the bands
def _pair(value, default: tuple) -> tuple:
    """One configured band -> ((lo, hi), was_replaced).

    READING THE FILE IS GENEROUS. A hand-edit that swapped the ends is
    ordered rather than dropped, and anything that is not two usable
    numbers falls back to the default for that band alone: one broken band
    must not cost him the page. Typing INTO the page is the other way round
    (band_edits refuses and says why), because a person who just typed
    something wrong should be told, not silently corrected.

    But the generosity is no longer SILENT, which is the half that costs
    him data: the second return says the value in the file was thrown
    away, so ``read_bands_noted`` can name it on screen. Measured
    2026-09-03: ``presence.room_band_m = [1.8, 8.0]`` rendered as 1.8/4.5
    with no note anywhere, and one press of SAVE wrote the 4.5 over his
    8.0. REORDERING is not a replacement and says nothing -- nothing was
    lost, and a note for every kindness teaches him to ignore the ones
    that matter.
    """
    try:
        lo, hi = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return default, True
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return default, True
    lo, hi = min(lo, hi), max(lo, hi)
    if lo < 0.0 or hi > MAX_BAND_M or lo >= hi:
        return default, True
    return (lo, hi), False


def _band_note(label: str, raw, default: tuple) -> str:
    """The sentence for a band the config could not use. It names the
    band, quotes what was in the file and says what SAVE would do with the
    substitute, because that is the press that destroys the original."""
    return ("%s: %s in assistant.json is not usable (two numbers, 0 to "
            "%.0f m, near end first) — showing the default %.2f–%.2f m, "
            "and SAVE would write that over it"
            % (label, _short(raw), MAX_BAND_M, default[0], default[1]))


def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def read_bands_noted(get_option: Optional[Callable]) -> tuple:
    """(the two bands, repaired) and (what the config lost doing it).

    Never raises. The notes tuple is empty when the file was usable as
    written -- and when it was merely reordered, which loses nothing.
    """
    def opt(key, default):
        if not callable(get_option):
            return default
        try:
            value = get_option(key, default)
        except Exception:                 # noqa: BLE001 - config boundary
            log.debug("sensors page: %s unreadable", key, exc_info=True)
            return default
        return default if value is None else value

    notes = []
    out = []
    for label, key, default in (("desk band", OPTION_DESK_BAND,
                                 DEFAULT_DESK_BAND),
                                ("room band", OPTION_ROOM_BAND,
                                 DEFAULT_ROOM_BAND)):
        raw = opt(key, None)
        if raw is None:
            out.append(default)           # absent is not a complaint
            continue
        pair, replaced = _pair(raw, default)
        out.append(pair)
        if replaced:
            log.warning("sensors page: %s is not usable (%s); using %s",
                        key, _short(raw), default)
            notes.append(_band_note(label, raw, default))
    return Bands(out[0][0], out[0][1], out[1][0], out[1][1]), tuple(notes)


def read_bands(get_option: Optional[Callable]) -> Bands:
    """The two bands from assistant.json, repaired. Never raises."""
    return read_bands_noted(get_option)[0]


def empty_state_line(get_option: Optional[Callable]) -> str:
    """What the page says when no room is configured.

    It BRANCHES, because ``roomfabric.room_specs()`` returns [] at its
    FIRST line whenever ``presence.room_sensor_enabled`` is false, whatever
    the address says. The old line always blamed
    ``presence.room_sensor_url`` -- and since 00d5b7e was about getting
    that URL right, "URL set, master switch still off" is the likely next
    state, in which the page would have pointed him at the one key that
    was already correct.

    Nothing on this page can set either switch, so the line says where
    they live and that a restart is what applies them.
    """
    enabled = False
    if callable(get_option):
        try:
            enabled = bool(get_option("presence.room_sensor_enabled", False))
        except Exception:                 # noqa: BLE001 - config boundary
            log.debug("sensors page: the master switch is unreadable",
                      exc_info=True)
    where = " — edit ~/.config/jarvis/assistant.json and restart Jarvis"
    if not enabled:
        return ("no room sensors: presence.room_sensor_enabled is off, so "
                "nothing is polled whatever the address says" + where)
    return ("no room sensors: presence.room_sensor_url and presence.rooms "
            "are both empty" + where)


def camera_room_name(get_option: Optional[Callable], specs) -> str:
    """Which room the lens is in.

    ``camera.room`` is the key that ties the camera to a room; the PRIMARY
    room is the default when he has not said. Reading primary ALONE was
    geometry inferred from an unrelated key: roomfabric forces ``primary``
    onto the first entry when none is marked, so a ``presence.rooms`` list
    that happened to lead with the kitchen would have shown the office
    camera's recognised face against the kitchen row and declared AT THE
    DESK there. A name that is not a configured room is ignored rather
    than obeyed, because obeying it gives EVERY row "no camera in this
    room" and hides the camera leg entirely.
    """
    names = [getattr(s, "name", "") for s in (specs or ())]
    want = ""
    if callable(get_option):
        try:
            want = str(get_option("camera.room", "") or "").strip()
        except Exception:                 # noqa: BLE001 - config boundary
            log.debug("sensors page: camera.room unreadable", exc_info=True)
    if want and want in names:
        return want
    if want:
        log.warning("sensors page: camera.room is %r, which is not a "
                    "configured room; using the primary one", want)
    return next((getattr(s, "name", "") for s in (specs or ())
                 if getattr(s, "primary", False)), "")


def read_overrules_noted(get_option: Optional[Callable]) -> tuple:
    """(whether the camera may overrule the radar) and (what the config
    lost getting there). Never raises.

    Defaults to TRUE: it is what he asked for in the first place, and the
    toggle exists so he can watch the fusion with it off, not because off
    is the norm. Absent is not a complaint, so it carries no note.

    ONLY A REAL BOOLEAN COUNTS, and that is the repair. ``bool(value)`` on
    anything else is the same silent coercion the bands had: a hand edit of
    ``"presence.camera_overrules": "false"`` is TRUE to bool(), so the
    toggle rendered ON, showed him his own setting inverted, and one press
    of SAVE wrote ``true`` over the word he typed. A number is refused for
    the same reason -- 0/1 for a switch is a guess about which of the two
    he meant, and this page destroys the original on SAVE.
    """
    if not callable(get_option):
        return True, ""
    try:
        value = get_option(OPTION_CAMERA_OVERRULES, None)
    except Exception:                     # noqa: BLE001 - config boundary
        log.debug("sensors page: %s unreadable", OPTION_CAMERA_OVERRULES,
                  exc_info=True)
        return True, ""
    if value is None:                     # absent: the default, silently
        return True, ""
    if isinstance(value, bool):
        return value, ""
    log.warning("sensors page: %s is not usable (%s); using true",
                OPTION_CAMERA_OVERRULES, _short(value))
    return True, ("camera overrules radar: %s in assistant.json is not "
                  "usable (true or false) — showing the default on, and "
                  "SAVE would write that over it" % _short(value))


def read_overrules(get_option: Optional[Callable]) -> bool:
    """Whether the camera is allowed to overrule the radar. Never raises."""
    return read_overrules_noted(get_option)[0]


def zone_for_distance(metres, bands: Bands) -> str:
    """Which band a range falls in: desk, room, or "" (neither).

    The desk band is checked FIRST, so where the two bands touch (1.8 m in
    his sketch) the nearer one wins. That is the safer way round: calling a
    man at the boundary "at the desk" costs a greeting he was going to get
    anyway, while calling him "in the room" would hold back the thing the
    desk zone exists to trigger.
    """
    if metres is None:
        return ""
    try:
        m = float(metres)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(m):
        return ""
    if bands.desk_lo <= m <= bands.desk_hi:
        return ZONE_DESK
    if bands.room_lo <= m <= bands.room_hi:
        return ZONE_ROOM
    return ""


def band_fraction(metres, lo: float, hi: float) -> Optional[float]:
    """Where a range sits inside a band, 0..1, or None when it is outside
    it (the marker is then simply not drawn -- a clamped marker parked on
    an end would read as a reading that is in the band)."""
    if metres is None or hi <= lo:
        return None
    try:
        m = float(metres)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(m) or m < lo or m > hi:
        return None
    return (m - lo) / (hi - lo)


def band_notes(bands: Bands) -> tuple:
    """Hardware facts his numbers just walked into, in his language.

    Both come from the LD2410's own geometry (scripts/room_sensor.py, which
    refuses to flash a profile that violates them): the module detects
    NOTHING inside 0.75 m, and it detects no STILL target inside 1.5 m --
    so a desk band that lives entirely below 1.5 m reads a motionless man
    as an empty room, which is the PIR failure the radar was chosen to
    avoid. Notes, not refusals: it is his wall and his desk.
    """
    out = []
    if bands.desk_lo < BLIND_M:
        out.append("the radar detects nothing inside %.2f m, so the bottom "
                   "of the desk band is dead" % BLIND_M)
    if bands.desk_hi <= STILL_FLOOR_M:
        out.append("no STILL target is reported inside %.1f m, so a "
                   "motionless man at the desk may read as empty" % STILL_FLOOR_M)
    return tuple(out)


def band_edits(desk_lo, desk_hi, room_lo, room_hi, overrules) -> tuple:
    """(the dotted keys to write, "") or ({}, why it was refused).

    Refuses rather than repairs -- see ``_pair`` for why the two directions
    differ. Every message names WHICH band, because two rows of four
    identical-looking boxes is exactly where a wrong one hides.
    """
    values = {}
    for label, lo_raw, hi_raw, key in (
            ("desk band", desk_lo, desk_hi, OPTION_DESK_BAND),
            ("room band", room_lo, room_hi, OPTION_ROOM_BAND)):
        pair = []
        for raw in (lo_raw, hi_raw):
            try:
                num = float(str(raw).strip())
            except (TypeError, ValueError):
                return {}, "%s: %r is not a number" % (label, str(raw).strip())
            if not math.isfinite(num):
                return {}, "%s: %r is not a number" % (label, str(raw).strip())
            if num < 0.0 or num > MAX_BAND_M:
                return {}, ("%s: %.2f m is outside what the radar can see "
                            "(0 to %.0f m)" % (label, num, MAX_BAND_M))
            pair.append(round(num, 2))
        if pair[0] >= pair[1]:
            return {}, ("%s: the near end (%.2f) must be less than the far "
                        "end (%.2f)" % (label, pair[0], pair[1]))
        values[key] = pair
    values[OPTION_CAMERA_OVERRULES] = bool(overrules)
    return values, ""


# ----------------------------------------------------------------- the write
def write_options(set_option: Callable, edits: dict) -> tuple:
    """Write every edit. Returns the keys that would NOT write, in order.

    THE ONE RULE THIS FUNCTION EXISTS FOR: in the real app the write does
    not raise. ``jarvis/app.py`` set_option catches internally and RETURNS
    False, so a caller that only guarded with try/except could not see a
    failure at all -- and the page's old save() dispatched these onto a
    daemon thread and then set "saved — restart Jarvis to apply it"
    unconditionally on the next line, before the write had even happened.
    He would restart expecting new bands, get the old ones, and nothing on
    screen would have said so.

    Only an explicit ``False`` is a failure. ``None`` is not: several
    services stand-ins in this tree return nothing at all, and reading
    silence as a failure would put a red line under a write that worked.

    Called INLINE, on the Tk thread. Three set_option calls are an
    in-memory edit plus one atomic os.replace (mkstemp in the same
    directory, fsync, replace, chmod 0600), and a daemon thread would also
    mean SAVE-then-quit could be killed mid-write.
    """
    failed = []
    for key, value in (edits or {}).items():
        try:
            ok = set_option(key, value)
        except Exception:                 # noqa: BLE001 - config boundary
            log.exception("sensors page: set_option %s failed", key)
            failed.append(key)
            continue
        if ok is False:
            log.warning("sensors page: %s would not write", key)
            failed.append(key)
    return tuple(failed)


def save_note(failed) -> tuple:
    """(the line under the SAVE button, a tone key) for a write's outcome."""
    keys = tuple(failed or ())
    if not keys:
        return SAVED_NOTE, TONE_FAINT
    return ("NOT SAVED — %s would not write; assistant.json still holds the "
            "old value" % ", ".join(keys)), TONE_ERR


# ---------------------------------------------------------------- the camera
@dataclass(frozen=True)
class CameraView:
    """What the eye says about one room, as numbers.

    ``present`` is whether there is a camera in that room AT ALL. "there is
    no camera in the kitchen" and "the kitchen camera is off" are different
    claims and only one of them is a guarantee -- the same distinction
    jarvis/rooms.py draws between ABSENT and OFF.

    ``asked`` is ``PreviewFace.id_ran``: False means identity was never
    run (the switch is off, the gallery is empty), True with an empty
    ``name`` means it WAS run and nothing matched. One boolean is the whole
    difference between "not looking" and "do not recognise you", and a page
    that showed them the same way would be unreadable exactly when he needs
    to read it.
    """
    present: bool = True
    live: bool = False
    running: bool = False
    faces: int = 0
    name: str = ""
    score: float = 0.0
    asked: bool = False
    reason: str = ""
    detail: str = ""

    @property
    def zone(self) -> str:
        """The camera's own verdict. A face in the cone is a person at the
        desk -- the lens points at the desk, which is the whole reason it
        is allowed to overrule a range. WHO that is rides separately in
        ``name``; this page does not fold identity into geometry."""
        return ZONE_DESK if (self.present and self.live and self.faces) else ""


def camera_view(status: Any, present: bool = True) -> CameraView:
    """``PreviewWorker.status()`` -> a CameraView. Never raises.

    The argument is the numbers-only dict (built from
    ``PreviewShot.numbers_only``, which excludes the image by name). An
    empty dict is the honest no-camera-wired case and renders as such.
    """
    data = status if isinstance(status, dict) else {}
    face = data.get("face") or {}
    if not isinstance(face, dict):
        face = {}
    try:
        score = float(face.get("id_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        faces = int(data.get("faces") or 0)
    except (TypeError, ValueError):
        faces = 0
    return CameraView(
        present=bool(present),
        live=bool(data.get("live")),
        running=bool(data.get("running")),
        faces=max(0, faces),
        name=str(face.get("name") or ""),
        score=score,
        asked=bool(face.get("id_ran")),
        reason=str(data.get("reason") or ""),
        detail=str(data.get("detail") or ""))


def camera_text(view: CameraView) -> tuple:
    """(the line, a tone key) for the camera row. Never blank."""
    if not view.present:
        return "%s no camera in this room" % DASH, TONE_FAINT
    if not view.live:
        # campreview already writes the sentence the pane prints ("CAMERA
        # OFF · curfew until 7 am"); reuse it rather than invent a second
        # vocabulary for the same states.
        return (view.detail or _reason_words(view.reason)
                or "camera not running"), TONE_FAINT
    if not view.faces:
        return "NO FACE", TONE_MUTED
    if not view.asked:
        return "FACE %s identity not running" % DASH, TONE_MUTED
    if not view.name:
        return "FACE  UNKNOWN  %.2f" % view.score, TONE_WARN
    return "FACE  %s  %.2f" % (view.name, view.score), TONE_OK


def _reason_words(reason: str) -> str:
    """campreview's own sentence for a reason code, imported lazily: that
    module pulls the vision lane's config helpers, and a diagnostics page
    must not be the reason a console fails to start."""
    if not reason:
        return ""
    try:
        from jarvis.campreview import REASON_WORDS
    except Exception:                     # noqa: BLE001 - optional lane
        return reason
    return REASON_WORDS.get(reason, reason)


# ---------------------------------------------------------------- the fusion
@dataclass(frozen=True)
class Verdict:
    """The fused zone, plus WHICH leg produced it and why. The source and
    the reason are on screen because the page exists for him to watch the
    override happen while he tunes, not merely to see the answer."""
    zone: str
    word: str
    source: str            # "camera" | "radar" | "" (nobody had an opinion)
    why: str


def fuse(*, present: Optional[bool], distance_m, camera: CameraView,
         bands: Bands, overrules: bool) -> Verdict:
    """The one fusion rule, as a pure function.

    ORDER, and the argument for it:

    1. The camera, when he lets it. His words: "camera recognition
       overrules sensor detection since he can literaly see me at my desk".
    2. The radar's presence bit, placed by its range. Presence with NO
       usable range is IN THE ROOM, never at the desk: the sensor's honest
       claim is "someone is here", and inventing the nearer zone out of a
       missing number is the failure mode this page was built to expose.
    3. Presence False is NOBODY THERE -- but only because a leg actually
       said so.
    4. Both legs silent is NO OPINION. Not an empty room. The whole None
       contract in jarvis/roomsensor.py exists for this line.

    With the toggle OFF and the radar abstaining, the camera still answers
    (step 5): nothing is being overruled when the other leg did not speak,
    and refusing the only evidence in the room would be a different lie.
    """
    cam_zone = camera.zone if camera is not None else ""
    if cam_zone and overrules:
        return Verdict(ZONE_DESK, ZONE_WORDS[ZONE_DESK], "camera",
                       "the camera can see a face; the radar's range is not asked")
    if present is True:
        zone = zone_for_distance(distance_m, bands)
        if zone == ZONE_DESK:
            why = "a target inside the desk band"
        elif zone == ZONE_ROOM:
            why = "a target inside the room band"
        else:
            zone = ZONE_ROOM
            why = ("presence with no usable range — someone is in the room, "
                   "the band is unknown")
        if cam_zone and not overrules:
            why += "; the camera sees a face but is not allowed to overrule"
        return Verdict(zone, ZONE_WORDS[zone], "radar", why)
    if present is False:
        why = "the radar reads empty"
        if cam_zone:
            why += ("; the camera sees a face but is not allowed to overrule"
                    if not overrules else "")
        return Verdict(ZONE_AWAY, ZONE_WORDS[ZONE_AWAY], "radar", why)
    if cam_zone:
        return Verdict(ZONE_DESK, ZONE_WORDS[ZONE_DESK], "camera",
                       "the radar has no opinion; the camera can see a face")
    return Verdict(ZONE_UNKNOWN, ZONE_WORDS[ZONE_UNKNOWN], "",
                   "neither leg has an opinion — this is not an empty room")


# ------------------------------------------------------------- the readings
@dataclass(frozen=True)
class Reading:
    """One poll of one room sensor. Numbers and a status dict, nothing else."""
    name: str
    label: str = ""
    url: str = ""
    present: Optional[bool] = None
    distance_m: Optional[float] = None
    rtt_ms: Optional[float] = None
    status: dict = field(default_factory=dict)
    at: float = 0.0


def fmt_m(value) -> str:
    """Metres to one decimal, or the dash. One decimal because the gate is
    0.75 m wide: a second digit would claim a precision the module does not
    have."""
    try:
        m = float(value)
    except (TypeError, ValueError):
        return DASH
    if not math.isfinite(m):
        return DASH
    return "%.1f m" % m


def fmt_ms(value) -> str:
    """A round trip in whole milliseconds, or the dash.

    The dash is not zero: it is what a poll that was never SENT looks like
    (offline mode, or the breaker skipping the socket entirely), and the
    difference between "the device answered instantly" and "nobody asked
    it" is the whole point of the column. A real sub-millisecond round trip
    therefore prints "<1 ms" rather than "0 ms" -- measured against the live
    office radar the floor is 46 ms, so on real hardware this only ever
    fires for an injected transport.
    """
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return DASH
    if not math.isfinite(ms) or ms < 0.0:
        return DASH
    if ms < 1.0:
        return "<1 ms"
    return "%d ms" % round(ms)


def fmt_s(value) -> str:
    try:
        return "%d s" % round(float(value))
    except (TypeError, ValueError):
        return DASH


def _int_or(value, default: int = 0) -> int:
    """Every other scalar in fault_line is defended; this one was not, and
    ``int(data.get("fails") or 0)`` raised ValueError on a non-integer.
    page_rows calls fault_line INSIDE the repaint, so that would have
    blanked the whole page rather than one field."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def age_text(now: float, at: float) -> str:
    """"updated 2 s ago", or "" when nothing has landed yet.

    The page had no staleness signal at all: ``Reading.at`` was captured on
    every poll and never rendered, so a diagnostics surface he reads while
    walking the room could not tell him the numbers had STOPPED moving.
    It is driven by the page's own 1 s Tk tick rather than by the poll, so
    a wedged transport makes the number climb instead of freezing it.

    A reading from the future (a clock step) reads as "just now" rather
    than as a negative age: the honest claim there is that it is fresh.
    """
    try:
        gap = float(now) - float(at)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(gap) or float(at) <= 0.0:
        return ""
    if gap < 1.5:
        return "updated just now"
    if gap < 60.0:
        return "updated %d s ago" % round(gap)
    if gap < 3600.0:
        return "updated %d min ago" % int(gap // 60)
    return "updated over an hour ago"


def presence_words(present: Optional[bool]) -> tuple:
    """(word, tone) for the presence bit. NO OPINION is its own state and
    wears the warning tone, so it can never be mistaken for EMPTY at a
    glance from a desk chair."""
    if present is True:
        return "PRESENT", TONE_OK
    if present is False:
        return "EMPTY", TONE_MUTED
    return "NO OPINION", TONE_WARN


def fault_line(status: Any, present: Optional[bool] = None) -> str:
    """Why this leg had no opinion, in his language. "" when it had one.

    The vocabulary is deliberately never "empty", "away" or "nobody": those
    words belong to a leg that ANSWERED. Everything here is a reason the
    question could not be asked or could not be understood.
    """
    if present is not None:
        return ""
    data = status if isinstance(status, dict) else {}
    if not data.get("url"):
        return "no address configured for this room"
    blocked = str(data.get("blocked") or "")
    if blocked == "offline":
        return "not polled — offline mode is on"
    if blocked == "stopped":
        return "not polled — sensing is stopped"
    if blocked == "policy":
        return "not polled — the sensing policy would not answer"
    if blocked:
        return "not polled — %s" % blocked
    if data.get("paused"):
        # retry_in_s, NOT cooldown_s: roomsensor._failed arms the breaker
        # from the current cooldown and THEN doubles it, so cooldown_s is
        # the NEXT wait -- exactly 2x the truth, and static for the whole
        # wait. Measured 09-03: the page said 60 s while jarvis.log said
        # 30 s for the same event. A status dict that carries no deadline
        # (a stand-in from a future caller) drops the number rather than
        # guessing at one.
        left = data.get("retry_in_s")
        if left is None:
            return "no answer — trying again shortly"
        return "no answer — trying again in %s" % fmt_s(left)
    fails = _int_or(data.get("fails"), 0)
    if fails:
        return "no answer from the sensor (%d in a row)" % fails
    return "no opinion — nothing that reads as presence came back"


@dataclass(frozen=True)
class RoomRow:
    """One room, fully rendered. The widget sets labels from this and makes
    no decisions of its own."""
    name: str
    label: str
    presence_word: str
    presence_tone: str
    distance_text: str
    distance_m: Optional[float]
    rtt_text: str
    fault: str
    camera_text: str
    camera_tone: str
    verdict: Verdict


def page_rows(readings, camera_status: Any, bands: Bands, overrules: bool,
              camera_room: str = "") -> list:
    """Every configured room, in config order, ready to paint."""
    out = []
    for r in readings or ():
        cam = camera_view(camera_status,
                          present=bool(camera_room) and r.name == camera_room)
        word, tone = presence_words(r.present)
        text, cam_tone = camera_text(cam)
        out.append(RoomRow(
            name=r.name, label=r.label or r.name,
            presence_word=word, presence_tone=tone,
            distance_text=fmt_m(r.distance_m) if r.distance_m is not None else DASH,
            distance_m=r.distance_m,
            rtt_text=fmt_ms(r.rtt_ms),
            fault=fault_line(r.status, r.present),
            camera_text=text, camera_tone=cam_tone,
            verdict=fuse(present=r.present, distance_m=r.distance_m,
                         camera=cam, bands=bands, overrules=overrules)))
    return out


WAITING_WORD = "READING…"


def waiting_row(spec) -> RoomRow:
    """The row shown before the first poll lands. Every value is a dash and
    the state says READING, because a blank row on a presence page reads as
    an empty room."""
    return RoomRow(name=getattr(spec, "name", ""), label=_spoken(spec),
                   presence_word=WAITING_WORD, presence_tone=TONE_MUTED,
                   distance_text=DASH, distance_m=None, rtt_text=DASH,
                   fault="", camera_text=DASH, camera_tone=TONE_FAINT,
                   verdict=Verdict(ZONE_UNKNOWN, ZONE_WORDS[ZONE_UNKNOWN], "",
                                   "the first poll has not come back yet"))


def blocked_reason(policy: Any) -> str:
    """Why the radar may not be polled right now ("" = it may).

    Mirrors ``RoomSensor.blocked`` deliberately, INCLUDING the rule that a
    policy which raises counts as forbidden: a decision we could not make
    is not permission. What it does not do is attach -- see the module
    docstring.
    """
    if policy is None:
        return ""
    try:
        return "" if policy.allowed(RADAR) else "offline"
    except Exception:                     # noqa: BLE001 - a broken policy is not a yes
        log.debug("sensors page: the sensing policy failed", exc_info=True)
        return "policy"


class SensorPoller:
    """Polls every configured room WHILE THE PAGE IS ON SCREEN, and only then.

    A diagnostics page that kept a thread alive behind a hidden surface
    would be a second poll loop competing with the fabric's own for the
    same ESP32 -- and on a device whose worst measured round trip is
    1186 ms, that is not free. ``start()``/``stop()`` are called from
    show()/hide() and nowhere else.

    ONE THREAD AT A TIME, and it is an invariant rather than a hope: a run
    is a generation number, the thread that is still finishing the last
    run's request ADOPTS the next one, and only that thread ever releases
    its own handle. See ``start()`` for the two shapes of the bug this
    replaces -- both measured, both on this branch.

    ``get`` is the transport seam (the same one ``RoomSensor`` exposes);
    the default is roomsensor's own, so there is exactly ONE HTTP client in
    the tree for these devices, with one 4 KB cap and one parser to review.
    """

    def __init__(self, specs, *, policy: Any = None,
                 get: Optional[Callable] = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 now: Optional[Callable[[], float]] = None):
        self.specs = list(specs or ())
        self.policy = policy
        self._get = get
        self.timeout_s = float(timeout_s)
        self._now = now or time.perf_counter
        # Keyed by room and deliberately OUTLIVING a run: the breaker state
        # of a radar that is unplugged is worth keeping across a tab flip.
        # That is only safe because exactly one thread ever touches them --
        # see start(); when two did, one round of failures doubled the
        # cooldown twice (measured 30 s -> 120 s, 2026-09-03).
        self._sensors: dict = {}
        # ONE condition guards the whole run state. A Condition and not a
        # Lock because stop() has to be able to cut the loop's sleep short.
        self._cond = threading.Condition(threading.Lock())
        self._thread: Optional[threading.Thread] = None
        self._armed = False               # is a run on?
        self._gen = 0                     # which run; bumped by start/stop
        self._on_rows: Optional[Callable] = None
        self._post: Optional[Callable] = None
        self._interval = POLL_S
        self.post_failed = False

    # ------------------------------------------------------------ one pass
    def _sensor(self, spec) -> RoomSensor:
        sensor = self._sensors.get(spec.name)
        if sensor is None:
            # policy is NOT passed: SensingPolicy.attach replaces by name.
            kw = {} if self._get is None else {"get": self._get}
            sensor = RoomSensor(spec.url, timeout_s=self.timeout_s, **kw)
            self._sensors[spec.name] = sensor
        return sensor

    def _distance(self, spec) -> Optional[float]:
        url = distance_url(spec.url)
        if not url:
            return None
        get = self._get
        if get is None:
            from jarvis import roomsensor
            get = roomsensor._get_default          # the module's ONE transport
        try:
            body = get(url, self.timeout_s)
        except Exception:                 # noqa: BLE001 - every failure is unknown
            log.debug("sensors page: %s unreachable", url, exc_info=True)
            return None
        cm = parse_distance_cm(body)
        return None if cm is None else cm / 100.0

    def poll_once(self, alive: Optional[Callable] = None) -> tuple:
        """One pass over every room. Never raises.

        ``alive`` is the poll thread's own "is my run still the current
        one". It is checked before every request, not once per pass, so a
        stop costs at most the ONE read already in flight. MEASURED with
        his two rooms and a transport parked inside the first read: 4
        requests went out after the stop before this, 1 after.
        """
        rows = []
        for spec in self.specs:
            if alive is not None and not alive():
                break
            sensor = self._sensor(spec)
            blocked = blocked_reason(self.policy)
            status = dict(sensor.status())
            status["blocked"] = blocked or status.get("blocked") or ""
            if not sensor.configured or blocked:
                # No request is sent -- that is the assertion, not that the
                # answer would have been ignored.
                rows.append(Reading(name=spec.name, label=_spoken(spec),
                                    url=spec.url, status=status,
                                    at=time.time()))
                continue
            before = sensor.reads
            t0 = self._now()
            present = sensor.read()
            sent = sensor.reads > before
            rtt = (self._now() - t0) * 1000.0 if sent else None
            # The SECOND request of the room, and gated by the same run
            # check: a stop that lands inside the presence read must not be
            # followed by a distance read nobody will ever paint.
            distance = (self._distance(spec)
                        if present is not None
                        and (alive is None or alive()) else None)
            status = dict(sensor.status())
            status["blocked"] = ""
            rows.append(Reading(name=spec.name, label=_spoken(spec),
                                url=spec.url, present=present,
                                distance_m=distance, rtt_ms=rtt,
                                status=status, at=time.time()))
        return tuple(rows)

    # -------------------------------------------------------------- thread
    def start(self, on_rows: Callable, post: Optional[Callable] = None,
              interval_s: float = POLL_S) -> None:
        """Poll on a thread of our own; hand results back through ``post``
        (the Tk thread hop). Idempotent.

        THE RUN IS A NUMBER, NOT A THREAD. ``start()`` arms a new
        generation; the poll thread reads that generation at the top of
        every pass. So when a start arrives while the last run's thread is
        still inside ``get(url, 3.0)``, that thread ADOPTS the new run --
        no second thread is created, and the two of them can never hold
        one ``RoomSensor``. The handle is released by the thread itself,
        under this same lock, so start() sees either a live thread (hand
        it the run) or none (make one) and never a gap between the two.

        TWO MEASURED BUGS THIS REPLACES, both on this branch.
        d41b52a shared ONE stop Event and cleared it on every start, so a
        thread inside a blocking read had its flag taken away and looped
        forever: 6 live "sensors-page" threads that never drained.
        dddaeeb gave each run its own Event but ``stop()`` released the
        Thread handle BEFORE joining it, so this guard read None while the
        old thread was still in the request and built a second loop beside
        it: peak 4 concurrent threads on one ESP32 while flipping the tab,
        and one round of failures doubled that room's breaker twice --
        cooldown 30 s -> 120 s where the design says 60 s.
        """
        with self._cond:
            if (self._armed and self._thread is not None
                    and self._thread.is_alive()):
                return                    # already running: idempotent
            self._on_rows = on_rows
            self._post = post
            self._interval = max(0.2, float(interval_s))
            self._gen += 1
            self._armed = True
            self.post_failed = False
            if self._thread is not None and self._thread.is_alive():
                # Still finishing the previous run's request. It picks this
                # run up on its way round rather than dying beside a
                # replacement that polls the same sensor at the same time.
                self._cond.notify_all()
                return
            thread = threading.Thread(target=self._worker, daemon=True,
                                      name=POLL_THREAD_NAME)
            self._thread = thread
        thread.start()

    def _worker(self) -> None:
        """The one poll thread. It serves RUNS, not one run.

        It exits -- and releases its own handle, under the lock ``start()``
        holds -- the moment it comes round to find no run armed. That is
        the whole of the lifecycle: there is no handle for stop() to drop
        early and no Event for start() to clear out from under a blocking
        read.

        ``post`` failing twice in a row ENDS the run, loudly. The hop is
        the only way to the Tk thread, so a loop that cannot make it can
        no longer repaint anything -- it would poll the ESP32 forever
        behind a surface frozen on stale numbers, with nothing at INFO
        saying why (measured: every self.after() raising "main thread is
        not in main loop" while the page kept polling).
        """
        try:
            self._serve()
        finally:
            # Belt and braces for a way out this loop has no name for (an
            # unexpected raise): the handle must not outlive the thread, or
            # `running` would say yes with nobody polling. Guarded on
            # identity so it cannot clobber a thread start() has already
            # put in its place.
            with self._cond:
                if self._thread is threading.current_thread():
                    self._thread = None

    def _serve(self) -> None:
        misses, served = 0, None
        while True:
            with self._cond:
                if not self._armed:
                    # The handle goes back HERE and nowhere else, so a
                    # start() under this lock can never mistake a thread on
                    # its way out for a live one, or a live one for gone.
                    self._thread = None
                    return
                gen = self._gen
                on_rows, post, interval = self._on_rows, self._post, self._interval
            if gen != served:             # a new run: its own miss count
                misses, served = 0, gen
            try:
                rows = self.poll_once(alive=lambda g=gen: self._current(g))
            except Exception:             # noqa: BLE001 - a diagnostic page
                log.exception("sensors page: poll failed")
                rows = ()
            if not self._current(gen):
                continue                  # stopped or replaced mid-pass
            try:
                (post or (lambda fn: fn()))(lambda r=rows: on_rows(r))
                misses = 0
            except Exception:             # noqa: BLE001 - a dead widget
                misses += 1
                if misses >= POST_FAILS_MAX:
                    log.warning("sensors page: %d repaints in a row could not "
                                "reach the Tk thread; the poll loop is "
                                "stopping rather than polling behind a frozen "
                                "surface", misses, exc_info=True)
                    # BOTH under the gen check: post_failed belongs to the
                    # RUN that could not reach Tk. If a start() has already
                    # armed a newer run (he reopened the page while this
                    # pass was in the hop), that run has its own hop and its
                    # own flag -- raising this one would put "the poll loop
                    # stopped -- reopen the page" (SensorsPage._tick) on a
                    # surface that is polling perfectly well.
                    with self._cond:
                        if self._gen == gen:
                            self.post_failed = True
                            self._armed = False
                    continue              # round to the top, which exits
                log.debug("sensors page: repaint failed", exc_info=True)
            with self._cond:
                if self._armed and self._gen == gen:
                    self._cond.wait(interval)   # a stop cuts this short

    def _current(self, gen: int) -> bool:
        """Is the run this pass belongs to still the one that is armed?"""
        with self._cond:
            return self._armed and self._gen == gen

    def stop(self, timeout_s: float = JOIN_S) -> bool:
        """End the run. True when the poll thread was gone by ``timeout_s``.

        The run is disarmed and its generation retired under the lock, and
        the thread reads that generation rather than an Event of its own,
        so there is nothing here to clear out from under a blocking read.
        ``stop()`` does NOT clear ``self._thread``: the thread does that
        itself on its way out, which is what lets a start() arriving mid
        request find the live thread and hand it the new run instead of
        starting a second one beside it.

        False is not a leak. The thread is bounded by the one request it is
        already inside (``poll_once`` re-checks the run between rooms), it
        cannot poll again for this run, and the next run reuses it.
        """
        with self._cond:
            self._armed = False
            self._gen += 1
            thread = self._thread
            self._cond.notify_all()
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(max(0.0, float(timeout_s)))
        alive = thread.is_alive()
        if alive:
            # Said out loud so a wedged transport is visible rather than
            # inferred. It sends at most the request it is already in.
            log.info("sensors page: the poll thread is still inside a "
                     "request; it will exit when that returns")
        return not alive

    @property
    def running(self) -> bool:
        """Is a run ARMED -- i.e. will another poll go out? Not "is a
        thread alive": after stop() the thread may still be finishing the
        request it was in, and that is not the page polling."""
        with self._cond:
            return self._armed


def _spoken(spec) -> str:
    return getattr(spec, "label", "") or getattr(spec, "name", "")


# =========================================================== the Tk surface
class _BandBar(tk.Canvas):
    """``[0.8]---#---[1.8] m``: the band, with the live range on it.

    The marker is simply ABSENT when the reading is outside the band (see
    ``band_fraction``) -- a marker clamped to an end would read as a
    reading that is in the band, which is the one thing this strip must
    never say.
    """

    H = 18                                # design units

    def __init__(self, parent, bg=None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, height=px(type(self).H), bg=bg,
                         highlightthickness=0, bd=0)
        self._frac: Optional[float] = None
        self.bind("<Configure>", lambda e: self._draw(), add=True)

    def set(self, fraction: Optional[float]) -> None:
        self._frac = fraction
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w = max(self.winfo_width(), px(40))
        h = max(self.winfo_height(), px(8))
        y = h / 2
        pad = px(6)
        self.create_line(pad, y, w - pad, y, fill=theme.LINE,
                         width=max(1, px(1)))
        for x in (pad, w - pad):          # the two end stops
            self.create_line(x, y - px(4), x, y + px(4), fill=theme.RAMP60,
                             width=max(1, px(1)))
        if self._frac is None:
            return
        x = pad + (w - 2 * pad) * max(0.0, min(1.0, float(self._frac)))
        r = px(4)
        self.create_oval(x - r, y - r, x + r, y + r, fill=theme.FOCAL,
                         outline="")


class _RoomBlock(tk.Frame):
    """One room: the sensor line, the camera line, the verdict line, and a
    fault line that is packed only when there is one."""

    def __init__(self, parent, bg: str):
        super().__init__(parent, bg=bg)
        self._bg = bg
        head = tk.Frame(self, bg=bg)
        head.pack(fill="x")
        self.name = tk.Label(head, font=ui_display(theme.SIZE_LABEL, "semibold"),
                             fg=theme.INK, bg=bg, anchor="w", width=12)
        self.name.pack(side="left")
        self.rtt = tk.Label(head, font=ui_mono(theme.SIZE_CAPTION),
                            fg=theme.FAINT, bg=bg, anchor="e", width=8)
        self.rtt.pack(side="right")
        self.distance = tk.Label(head, font=ui_mono(theme.SIZE_LABEL),
                                 fg=theme.FOCAL, bg=bg, anchor="e", width=8)
        self.distance.pack(side="right")
        self.presence = tk.Label(head, font=ui_display(theme.SIZE_LABEL),
                                 fg=theme.MUTED, bg=bg, anchor="w")
        self.presence.pack(side="left")

        self.camera = self._sub("camera")
        self.verdict = self._sub("verdict")
        self.fault = tk.Label(self, font=ui_font(theme.SIZE_CAPTION),
                              fg=theme.WARN, bg=bg, anchor="w",
                              justify="left")
        self.why = tk.Label(self, font=ui_font(theme.SIZE_CAPTION),
                            fg=theme.FAINT, bg=bg, anchor="w", justify="left")
        self.why.pack(fill="x", padx=(px(96), 0))
        self.bind("<Configure>", self._wrap, add=True)

    def _wrap(self, event) -> None:
        """Tk labels do not wrap without an explicit pixel width, and these
        two carry the only long sentences on the page. Without this the
        reason line was cut mid-word at his window ("...the radar's range
        is n") -- measured on the photo rig, 2026-09-03."""
        width = max(px(120), int(event.width) - px(104))
        for label in (self.fault, self.why):
            label.configure(wraplength=width)

    def _sub(self, caption: str) -> tk.Label:
        row = tk.Frame(self, bg=self._bg)
        row.pack(fill="x")
        tk.Label(row, text=caption, font=ui_font(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=self._bg, anchor="w",
                 width=14).pack(side="left", padx=(px(16), 0))
        value = tk.Label(row, font=ui_font(theme.SIZE_LABEL), fg=theme.MUTED,
                         bg=self._bg, anchor="w")
        value.pack(side="left")
        return value

    def apply(self, row: RoomRow) -> None:
        self.name.configure(text=row.label.upper())
        self.presence.configure(text=row.presence_word,
                                fg=tone_color(row.presence_tone))
        self.distance.configure(text=row.distance_text)
        self.rtt.configure(text=row.rtt_text)
        self.camera.configure(text=row.camera_text,
                              fg=tone_color(row.camera_tone))
        self.verdict.configure(
            text=row.verdict.word,
            fg=theme.FOCAL if row.verdict.zone != ZONE_UNKNOWN else theme.WARN)
        self.why.configure(text=("%s · %s" % (row.verdict.source,
                                              row.verdict.why)
                                 if row.verdict.source else row.verdict.why))
        if row.fault:
            self.fault.configure(text=row.fault)
            self.fault.pack(fill="x", padx=(px(96), 0))
        else:
            self.fault.pack_forget()


class SensorsPage(tk.Frame):
    """The SENSORS surface, placed over the console's stage.

    WHERE THE TAB IS. Not here and not in the header. The header has 13 px
    to spare at his 920-px window (tests/test_header_fit.py measures it),
    so a `[SENSORS]` chip up there costs him the sensing badge, which is
    the privacy readout. It is a ROW OF ITS OWN under the wordmark --
    jarvis/ui/tab_strip.py -- and this page no longer draws the two fake
    tabs it used to: two rows of tabs on one screen is a question about
    which of them is in charge.

    WHY IT COVERS THE REACTOR TOO. MEASURED at his window and scale
    (920x1440, S=2.0): the transcript panel alone is ~520 device px and
    this page asks for 773, so placed over the transcript the bands, the
    overrule toggle and SAVE were simply cut off the bottom -- the first
    photo rig pass showed exactly that. ``cover`` is the widgets whose
    union it should span (the reactor and the transcript); the reactor is
    decoration, and a man reading a sensor page is not looking at it.
    MEASURED 2026-09-03 on the real window at 920x1440, S=2.0, in BOTH
    looks (the earlier note gave holo's numbers and called them both)::

                 stage span      this page      spare
        holo     1124 -> 1044    868 -> 773     256 -> 271
        classic  1136 -> 1056    868 -> 773     268 -> 283

    So there is MORE headroom than before the strip, in both: dropping
    this page's own tab row (95 px) paid for the strip (80 px, identical
    in the two looks) with 15 px over.

    Nothing here polls until ``show()`` and nothing keeps polling after
    ``hide()``. After 09-03 that is a generation number rather than a stop
    flag, and there is exactly ONE poll thread at a time (see
    ``SensorPoller.start``) -- a tab is far easier to flip in and out of
    than a function key was, and two loops on one RoomSensor doubled that
    room's breaker twice. STANDBY shuts the page as well: the console
    hides the tab row with the footer and selects CHAT on the way
    (main_window._set_tabs_hidden), so nothing polls behind the clock.
    """

    def __init__(self, host, services=None, camera_status=None,
                 cover=(), on_close: Optional[Callable] = None):
        super().__init__(host, bg=theme.TV_BG)
        self.host = host
        self.cover = tuple(w for w in (cover or ()) if w is not None)
        self.services = services
        self._camera_status = camera_status
        self._on_close = on_close
        self._open = False
        self._blocks: dict = {}
        self._rows: list = []
        self._last: tuple = ()            # the readings the rows were painted from
        self._tick_id = None

        self.specs = self._room_specs()
        self.bands, self.config_notes = read_bands_noted(self._get_option)
        # The toggle gets the same treatment as the bands: a value the file
        # holds that this page cannot use is NAMED, because SAVE writes the
        # substitute over it.
        self.overrules, overrule_note = read_overrules_noted(self._get_option)
        if overrule_note:
            self.config_notes = tuple(self.config_notes) + (overrule_note,)
        # camera.room, falling back to the primary room -- NOT primary
        # alone, which was geometry inferred from an unrelated key.
        self.camera_room = camera_room_name(self._get_option, self.specs)
        self.poller = SensorPoller(self.specs,
                                   policy=getattr(services, "sensing", None))
        self._build()

    # ------------------------------------------------------------- config
    def _get_option(self, key: str, default=None):
        fn = getattr(self.services, "get_option", None) if self.services else None
        if fn is None:
            return default
        try:
            value = fn(key, default)
        except Exception:                 # noqa: BLE001 - config boundary
            log.exception("sensors page: get_option %s failed", key)
            return default
        return default if value is None else value

    def _room_specs(self) -> list:
        """Every configured room, through the fabric's own reader -- so the
        page lists exactly what the app would poll, singular keys and all."""
        try:
            from jarvis.roomfabric import room_specs
        except Exception:                 # noqa: BLE001 - optional lane
            log.exception("sensors page: the room fabric is unavailable")
            return []
        try:
            specs = room_specs(_CfgView(self._get_option))
        except Exception:                 # noqa: BLE001 - config boundary
            log.exception("sensors page: room specs unreadable")
            return []
        return list(specs)

    # -------------------------------------------------------------- build
    def _build(self) -> None:
        bg = theme.TV_BG
        # NO tab row of its own any more. The console has a real one
        # (jarvis/ui/tab_strip.py) directly under the wordmark, which is
        # what he asked for, and two rows of tabs on one screen is a
        # question about which of them is in charge. Dropping it also pays
        # for the strip: it cost this page more vertical than the strip
        # costs the stage (measured, scripts/ui_shots.py).
        body = tk.Frame(self, bg=bg)
        body.pack(fill="both", expand=True, padx=theme.PAD,
                  pady=(theme.PAD_S, 0))
        if not self.specs:
            tk.Label(body, text=empty_state_line(self._get_option),
                     font=ui_font(theme.SIZE_LABEL), fg=theme.FAINT, bg=bg,
                     anchor="w", justify="left",
                     wraplength=px(420)).pack(fill="x")
        for spec in self.specs:
            block = _RoomBlock(body, bg)
            # px(4), not PAD_S: with a fault line AND a reason line on both
            # rooms (state 27 of the photo rig) the column was ~4 px taller
            # than the stage and Tk clipped the last caption's descenders.
            block.pack(fill="x", pady=(0, px(4)))
            self._blocks[spec.name] = block

        tk.Frame(self, bg=theme.LINE, height=max(1, px(1))).pack(
            fill="x", padx=theme.PAD, pady=theme.PAD_S)
        tune = tk.Frame(self, bg=bg)
        tune.pack(fill="x", padx=theme.PAD)
        self._desk = self._band_row(tune, "desk band",
                                    self.bands.desk_lo, self.bands.desk_hi)
        self._room = self._band_row(tune, "room band",
                                    self.bands.room_lo, self.bands.room_hi)

        row = tk.Frame(tune, bg=bg)
        row.pack(fill="x", pady=px(4))
        tk.Label(row, text="camera overrules radar",
                 font=ui_font(theme.SIZE_LABEL), fg=theme.MUTED, bg=bg,
                 anchor="w").pack(side="left")
        self._overrule = Toggle(row, value=self.overrules, bg=bg)
        self._overrule.pack(side="right")
        self._overrule.command = self._on_overrule

        foot = tk.Frame(self, bg=bg)
        foot.pack(fill="x", padx=theme.PAD, pady=(theme.PAD_S, theme.PAD))
        # SAVE and the age share ONE row; the caption gets the next one to
        # itself. Packing all three into `foot` put the caption between the
        # button and the age and cut it mid-word -- photographed 09-03,
        # which is the same trap that put this caption below the button in
        # the first place.
        row = tk.Frame(foot, bg=bg)
        row.pack(fill="x")
        RoundButton(row, text="SAVE", kind="default", size=theme.SIZE_CAPTION,
                    bg=bg, command=self.save).pack(side="left")
        # The staleness readout, on the SAVE row so it costs no height. It
        # is driven by the page's own 1 s tick, not by the poll, so a poll
        # that has stopped makes this number CLIMB rather than freeze --
        # which is the whole point of putting an age on a numbers page.
        self._age = tk.Label(row, font=ui_font(theme.SIZE_CAPTION),
                             fg=theme.FAINT, bg=bg, anchor="e")
        self._age.pack(side="right", padx=(px(12), 0))
        # BELOW the button, not beside it: beside it the caption had ~750 px
        # of a 920-px window and was cut mid-word on the photo rig, and the
        # refusal message a bad band edit puts here is longer still.
        self._note = tk.Label(foot, text=RESTART_NOTE,
                              font=ui_font(theme.SIZE_CAPTION), fg=theme.FAINT,
                              bg=bg, anchor="w", justify="left")
        self._note.pack(fill="x", pady=(px(4), 0))
        self._notes = tk.Label(self, font=ui_font(theme.SIZE_CAPTION),
                               fg=theme.WARN, bg=bg, anchor="w",
                               justify="left")
        self.bind("<Configure>", lambda e: [
            w.configure(wraplength=max(px(160), int(e.width) - 2 * theme.PAD))
            for w in (self._notes, self._note)], add=True)
        self._show_notes()

    def _band_row(self, parent, caption: str, lo: float, hi: float) -> dict:
        bg = theme.TV_BG
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=px(4))
        tk.Label(row, text=caption, font=ui_font(theme.SIZE_LABEL),
                 fg=theme.MUTED, bg=bg, anchor="w",
                 width=11).pack(side="left")
        tk.Label(row, text="m", font=ui_font(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=bg).pack(side="right", padx=(px(4), 0))
        hi_e = self._entry(row, hi)
        hi_e.pack(side="right")
        lo_e = self._entry(row, lo)
        lo_e.pack(side="left", padx=(0, px(6)))
        bar = _BandBar(row, bg=bg)
        bar.pack(side="left", fill="x", expand=True, padx=(0, px(6)))
        return {"lo": lo_e, "hi": hi_e, "bar": bar}

    def _entry(self, parent, value: float) -> tk.Entry:
        e = tk.Entry(parent, width=5, justify="center",
                     font=ui_mono(theme.SIZE_LABEL), fg=theme.INK,
                     bg=theme.SURFACE, insertbackground=theme.CYAN,
                     relief="flat", highlightthickness=1,
                     highlightbackground=theme.LINE,
                     highlightcolor=theme.CYAN_DIM, bd=0)
        e.insert(0, "%.2f" % value)
        return e

    # --------------------------------------------------------- open / shut
    def toggle(self) -> None:
        self.hide() if self._open else self.show()

    @property
    def is_open(self) -> bool:
        """Named ``is_open`` and not ``open``: a method called ``open`` on a
        widget shadows the builtin at every call site that reads it."""
        return self._open

    def place_box(self) -> dict:
        """The place() kwargs that cover ``cover``, or the whole host.

        Measured off the live geometry rather than assumed, so the page
        still lands correctly when the footer is hidden (standby) or the
        camera pane is packed or unpacked underneath it.
        """
        if not self.cover:
            return dict(x=0, y=0, relwidth=1.0, relheight=1.0)
        try:
            self.host.update_idletasks()
            tops = [w.winfo_y() for w in self.cover]
            bottoms = [w.winfo_y() + w.winfo_height() for w in self.cover]
            top, height = min(tops), max(bottoms) - min(tops)
        except Exception:                 # noqa: BLE001 - an unmapped widget
            log.debug("sensors page: cover geometry unreadable", exc_info=True)
            return dict(x=0, y=0, relwidth=1.0, relheight=1.0)
        if height < 1:
            return dict(x=0, y=0, relwidth=1.0, relheight=1.0)
        return dict(x=0, y=top, relwidth=1.0, height=height)

    def show(self) -> None:
        if self._open:
            return
        self._open = True
        self.place(in_=self.host, **self.place_box())
        self.lift()
        # The first pass comes off the POLL THREAD, never from here. A
        # synchronous poll_once() on the Tk thread would freeze the console
        # for up to 3 s per room on a sensor that has gone away, which is
        # exactly the state this page exists to be opened in. The rows show
        # their waiting text for one tick instead; the same rule DeskWatch
        # and _probe_room follow, and for the same reason.
        self.waiting()
        self.poller.start(self.apply, post=self._post)
        self._tick()

    def hide(self) -> None:
        if not self._open:
            return
        self._open = False
        self.poller.stop()
        self._untick()
        try:
            self.place_forget()
        except Exception:                 # noqa: BLE001 - a dead widget
            log.debug("sensors page: unplace failed", exc_info=True)
        if self._on_close:
            try:
                self._on_close()
            except Exception:             # noqa: BLE001 - a callback
                log.exception("sensors page: on_close failed")

    def _post(self, fn) -> None:
        """Hop to the Tk thread. RAISES when the hop cannot be made.

        It used to swallow the failure at DEBUG, which is how a page whose
        every repaint raised "main thread is not in main loop" kept
        polling behind a surface stuck on READING… with nothing at INFO
        saying why. The poll loop counts these now and stops after
        POST_FAILS_MAX in a row, loudly -- a loop that cannot reach Tk
        cannot repaint anything, so continuing is only cost.
        """
        self.after(0, fn)

    # ---------------------------------------------------------- the age tick
    def _tick(self) -> None:
        """Repaint the age of the newest reading, once a second, on the Tk
        thread. Independent of the poll ON PURPOSE (see ``age_text``)."""
        self._tick_id = None
        if not self._open:
            return
        newest = max((r.at for r in self._last if r.at), default=0.0)
        text = age_text(time.time(), newest)
        if self.poller.post_failed:
            text = "the poll loop stopped — reopen the page"
        try:
            self._age.configure(text=text)
            self._tick_id = self.after(AGE_TICK_MS, self._tick)
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("sensors page: age tick failed", exc_info=True)

    def _untick(self) -> None:
        tick, self._tick_id = self._tick_id, None
        if tick is None:
            return
        try:
            self.after_cancel(tick)
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("sensors page: age tick cancel failed", exc_info=True)

    # ------------------------------------------------------------ painting
    def waiting(self) -> None:
        """What the page says in the tick before the first poll lands.

        Not blank, and emphatically not EMPTY: an unlabelled row on a
        presence page reads as "nobody there", which is the one thing this
        surface may never accidentally say.
        """
        for spec in self.specs:
            self.apply_row(waiting_row(spec))

    def apply_row(self, row: RoomRow) -> None:
        block = self._blocks.get(row.name)
        if block is not None:
            block.apply(row)

    def apply(self, readings) -> None:
        if not self._open:
            return
        self._last = tuple(readings or ())
        rows = page_rows(self._last, self._camera(), self.bands, self.overrules,
                         camera_room=self.camera_room)
        self._rows = rows
        for row in rows:
            self.apply_row(row)
        here = next((r.distance_m for r in rows
                     if r.name == self.camera_room), None)
        if here is None and rows:
            here = rows[0].distance_m
        self._desk["bar"].set(band_fraction(here, self.bands.desk_lo,
                                            self.bands.desk_hi))
        self._room["bar"].set(band_fraction(here, self.bands.room_lo,
                                            self.bands.room_hi))

    def _camera(self) -> dict:
        """The camera's numbers-only status, or {} -- which renders as "not
        running", never as an empty room."""
        fn = self._camera_status
        if not callable(fn):
            return {}
        try:
            got = fn()
        except Exception:                 # noqa: BLE001 - provider boundary
            log.debug("sensors page: camera status failed", exc_info=True)
            return {}
        return got if isinstance(got, dict) else {}

    def _show_notes(self) -> None:
        # config_notes FIRST: "the file holds a value I could not use" is
        # the one that costs him data if he presses SAVE past it.
        notes = tuple(self.config_notes) + band_notes(self.bands)
        if notes:
            self._notes.configure(text="\n".join(notes))
            self._notes.pack(fill="x", padx=theme.PAD, pady=(0, theme.PAD_S))
        else:
            self._notes.pack_forget()

    # -------------------------------------------------------------- saving
    def _on_overrule(self, value: bool) -> None:
        """The toggle repaints the fusion at once so he can WATCH the
        override stop happening; the write waits for SAVE, like the bands."""
        self.overrules = bool(value)
        # Repaint from the readings already on screen rather than waiting a
        # second for the next poll: the point of the toggle is that he can
        # SEE the override stop happening the moment he flicks it.
        self.apply(self._last)

    def save(self) -> None:
        """Validate, write, and say WHICH of those three happened.

        The write is inline and its result is read: see ``write_options``
        for why a thread and a bare try/except could not tell a failed
        save from a successful one. A band the config could not use is
        re-checked here too, because a successful write clears it.
        """
        edits, err = self.band_values()
        if err:
            self._note.configure(text=err, fg=theme.ERR)
            return
        self.bands = Bands(*(edits[OPTION_DESK_BAND] + edits[OPTION_ROOM_BAND]))
        self.overrules = bool(edits[OPTION_CAMERA_OVERRULES])
        fn = getattr(self.services, "set_option", None) if self.services else None
        if fn is None:
            self._note.configure(text=NOT_WIRED_NOTE, fg=theme.WARN)
            return
        failed = write_options(fn, edits)
        if not failed:
            # What he typed is now what the file holds, so the "the config
            # had a value I could not use" notes are spent.
            self.config_notes = ()
        self._show_notes()
        text, tone = save_note(failed)
        self._note.configure(text=text, fg=tone_color(tone))

    def band_values(self) -> tuple:
        """What is currently typed, validated. Split out so the refusal
        message is testable without a display."""
        return band_edits(self._desk["lo"].get(), self._desk["hi"].get(),
                          self._room["lo"].get(), self._room["hi"].get(),
                          self._overrule.get())


class _CfgView:
    """``cfg.get(dotted, default)`` over a ``get_option`` callable, so
    ``roomfabric.room_specs`` can read the live config without this page
    loading assistant.json itself (it holds his app passwords and tokens;
    the console already has a redacting reader and this goes through it)."""

    def __init__(self, get_option: Optional[Callable]):
        self._get = get_option

    def get(self, key: str, default=None):
        if not callable(self._get):
            return default
        try:
            value = self._get(key, default)
        except Exception:                 # noqa: BLE001 - config boundary
            return default
        return default if value is None else value
