"""The SENSORS page: what the radar thinks, what the camera thinks, and
which of them won.

He asked for "a small GUI" inside the console -- not a browser tab and not
a second window -- to watch presence while he tunes it::

    JARVIS   [chat] [SENSORS] [*]
    ---------------------------------
     office        * PRESENT 1.4 m
       camera      * FACE  hunterp
       verdict     AT THE DESK
     desk band  [0.8]---#---[1.8] m
     room band  [1.8]--#----[4.5] m
     camera overrules radar   [x]

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
    """One configured band -> an ordered (lo, hi), or the default.

    READING THE FILE IS GENEROUS. A hand-edit that swapped the ends is
    ordered rather than dropped, and anything that is not two usable
    numbers falls back to the default for that band alone: one broken band
    must not cost him the page. Typing INTO the page is the other way round
    (band_edits refuses and says why), because a person who just typed
    something wrong should be told, not silently corrected.
    """
    try:
        lo, hi = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return default
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return default
    lo, hi = min(lo, hi), max(lo, hi)
    if lo < 0.0 or hi > MAX_BAND_M or lo >= hi:
        return default
    return (lo, hi)


def read_bands(get_option: Optional[Callable]) -> Bands:
    """The two bands from assistant.json, repaired. Never raises."""
    def opt(key, default):
        if not callable(get_option):
            return default
        try:
            value = get_option(key, default)
        except Exception:                 # noqa: BLE001 - config boundary
            log.debug("sensors page: %s unreadable", key, exc_info=True)
            return default
        return default if value is None else value

    desk = _pair(opt(OPTION_DESK_BAND, list(DEFAULT_DESK_BAND)),
                 DEFAULT_DESK_BAND)
    room = _pair(opt(OPTION_ROOM_BAND, list(DEFAULT_ROOM_BAND)),
                 DEFAULT_ROOM_BAND)
    return Bands(desk[0], desk[1], room[0], room[1])


def read_overrules(get_option: Optional[Callable]) -> bool:
    """Whether the camera is allowed to overrule the radar. Defaults to
    TRUE: it is what he asked for in the first place, and the toggle exists
    so he can watch the fusion with it off, not because off is the norm."""
    if not callable(get_option):
        return True
    try:
        value = get_option(OPTION_CAMERA_OVERRULES, True)
    except Exception:                     # noqa: BLE001 - config boundary
        log.debug("sensors page: %s unreadable", OPTION_CAMERA_OVERRULES,
                  exc_info=True)
        return True
    return True if value is None else bool(value)


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
        return "no answer — trying again in %s" % fmt_s(data.get("cooldown_s"))
    fails = int(data.get("fails") or 0)
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
        self._sensors: dict = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

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

    def poll_once(self) -> tuple:
        """One pass over every room. Never raises."""
        rows = []
        for spec in self.specs:
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
            distance = self._distance(spec) if present is not None else None
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
        (the Tk thread hop). Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    rows = self.poll_once()
                except Exception:         # noqa: BLE001 - a diagnostic page
                    log.exception("sensors page: poll failed")
                    rows = ()
                if self._stop.is_set():
                    return
                try:
                    (post or (lambda fn: fn()))(lambda r=rows: on_rows(r))
                except Exception:         # noqa: BLE001 - a dead widget
                    log.debug("sensors page: repaint failed", exc_info=True)
                self._stop.wait(max(0.2, float(interval_s)))

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="sensors-page")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread = None


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

    WHY NOT A HEADER TAB. The header has 13 px to spare at his 920-px
    window -- tests/test_header_fit.py measures it: 299 of the 312 left for
    chips is spent by the state pill and the sensing badge, and the ruling
    that got it there was his own ("dont make jarvis smaller, just make
    ready and sensing smaller to fit"). A `[SENSORS]` tab up there would
    cost him the sensing badge, which is the privacy readout. So the tab
    strip lives at the top of the PAGE and the header is untouched. If he
    would rather have it in the header, something up there has to go, and
    that is his call to make, not one to make for him.

    WHY IT COVERS THE REACTOR TOO. MEASURED at his window and scale
    (920x1440, S=2.0): the transcript panel alone is ~420 device px and
    this page asks for ~940, so placed over the transcript the bands, the
    overrule toggle and SAVE were simply cut off the bottom -- the first
    photo rig pass showed exactly that. ``cover`` is the widgets whose
    union it should span (the reactor and the transcript); the reactor is
    decoration, and a man reading a sensor page is not looking at it.

    Nothing here polls until ``show()`` and nothing keeps polling after
    ``hide()``.
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

        self.specs = self._room_specs()
        self.bands = read_bands(self._get_option)
        self.overrules = read_overrules(self._get_option)
        self.camera_room = next((s.name for s in self.specs
                                 if getattr(s, "primary", False)), "")
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
        tabs = tk.Frame(self, bg=bg)
        tabs.pack(fill="x", padx=theme.PAD, pady=(theme.PAD_S, 0))
        RoundButton(tabs, text="CHAT", kind="ghost", size=theme.SIZE_CAPTION,
                    bg=bg, command=self.hide).pack(side="left")
        RoundButton(tabs, text="SENSORS", kind="accent",
                    size=theme.SIZE_CAPTION, bg=bg,
                    command=lambda: None).pack(side="left", padx=(px(6), 0))
        RoundButton(tabs, text="✕", kind="ghost", size=theme.SIZE_CAPTION,
                    pad_x=8, pad_y=5, bg=bg,
                    command=self.hide).pack(side="right")
        tk.Frame(self, bg=theme.LINE, height=max(1, px(1))).pack(
            fill="x", padx=theme.PAD, pady=(theme.PAD_S, theme.PAD_S))

        body = tk.Frame(self, bg=bg)
        body.pack(fill="both", expand=True, padx=theme.PAD)
        if not self.specs:
            tk.Label(body, text="no room sensors configured — "
                              "presence.room_sensor_url is empty",
                     font=ui_font(theme.SIZE_LABEL), fg=theme.FAINT, bg=bg,
                     anchor="w", justify="left").pack(fill="x")
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
        RoundButton(foot, text="SAVE", kind="default", size=theme.SIZE_CAPTION,
                    bg=bg, command=self.save).pack(anchor="w")
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

    def hide(self) -> None:
        if not self._open:
            return
        self._open = False
        self.poller.stop()
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
        """Hop to the Tk thread. A page that has been closed drops it."""
        try:
            self.after(0, fn)
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("sensors page: post failed", exc_info=True)

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
        notes = band_notes(self.bands)
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
        edits, err = self.band_values()
        if err:
            self._note.configure(text=err, fg=theme.ERR)
            return
        self.bands = Bands(*(edits[OPTION_DESK_BAND] + edits[OPTION_ROOM_BAND]))
        self.overrules = bool(edits[OPTION_CAMERA_OVERRULES])
        self._show_notes()
        fn = getattr(self.services, "set_option", None) if self.services else None
        if fn is None:
            self._note.configure(text="assistant settings not wired",
                                 fg=theme.WARN)
            return

        def run():
            for key, value in edits.items():
                try:
                    fn(key, value)
                except Exception:         # noqa: BLE001 - config boundary
                    log.exception("sensors page: set_option %s failed", key)
        threading.Thread(target=run, daemon=True, name="sensors-save").start()
        self._note.configure(text=SAVED_NOTE, fg=theme.FAINT)

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
