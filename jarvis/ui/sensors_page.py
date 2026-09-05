"""The SENSORS page: what the radar thinks, what the camera thinks, and
which of them won.

He asked for "a small GUI" inside the console -- not a browser tab and not
a second window -- to watch presence while he tunes it::

    JARVIS                            [READY] [SENSING]
    ---------------------------------------------------
     [ CHAT ]  [ SENSORS ]      <- jarvis/ui/tab_strip.py
    ---------------------------------------------------
     office        * PRESENT 3.1 m
       camera      * FACE  hunterp
       verdict     AT THE DESK
     empty space  [0.75]-#-----[2.25] m
     at the desk  [2.25]----#--[3.75] m
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

ONE ZONE MODEL, AND IT IS NOT THIS FILE'S (2026-09-03). Until today this
page had a zone model of its own -- ``presence.desk_band_m`` and
``presence.room_band_m``, two bands, the desk assumed to be the NEARER one
-- while ``jarvis/zones.py`` and the zone log had an N-band ladder with
names, gaps and a camera zone under ``zones.rooms``. Different keys, so git
saw no conflict when the two branches merged, and the disagreement was
MEASURED rather than argued: the console called 4 of 10 points "AT THE
DESK" from the radar alone, and the zone log had no desk band at all and
could reach "at the desk" only through the camera. In his actual office the
two-band model was BACKWARDS -- the near space is empty and the desk is at
3.13 m median (measured on the live radar, 2026-09-03).

So the page reads and writes ``zones.rooms`` and nothing else: the bands
come out of ``zones.read_zones`` (the same validator ``scripts/zone_log.py``
uses) and the verdict is ``zones.verdict``, so what he edits here is what
places a reading in the record. The band rows are named by the band and
grouped by room, because "desk band" and "room band" are not what his
ladders are called. His kitchen has no lens and says so -- a blank
``camera_zone`` means there is no camera in the room, and the camera rule
cannot fire for it.

THE OLD KEYS ARE A MIGRATION SOURCE AND NOTHING ELSE. They are gone from
``assistant_config.DEFAULTS``, this file is the only reader left, and a
config that still has them gets its numbers carried across rather than
dropped: where there is no ``zones.rooms`` ladder for a room they become
its bands, where there IS one ``zones.rooms`` wins and the page says which
it used, and a successful SAVE writes the bands into ``zones.rooms`` and
REMOVES the old keys -- a superseded key left in the file looks exactly
like a live one, which is how two models disagreed for a day.

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
  file never opens assistant.json. Retiring the superseded keys goes
  through ``services.unset_option`` for the same reason, and where that is
  not wired the keys stay and the page SAYS they stay.
* **The write is a MERGE of ``zones.rooms``, never a rebuild.** A list
  replaces rather than merges in ``AssistantConfig``, so a save that built
  the list from the widgets would silently delete every room this page was
  not showing. ``band_edits`` edits a copy of what the file holds and
  touches only the bands of the rooms on screen.
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

import copy
import json
import math
import threading
import time
import tkinter as tk
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from jarvis import zones as zn
from jarvis.logs import get_logger
from jarvis.roomsensor import DEFAULT_TIMEOUT_S, RoomSensor, entity_path
from jarvis.sensing import RADAR
from jarvis.ui import theme
# ABSENT and UNPLACED are re-exported rather than used here: they are the
# three verdicts that are not a band, they are SPELLED in jarvis/zones.py,
# and a page that wrote its own copy of the words is how the two models
# drifted apart in the first place.
from jarvis.zones import (ABSENT, NO_OPINION,  # noqa: F401 - re-exported
                          UNPLACED, Band, ZoneMap)
from jarvis.ui.widgets import (RoundButton, Toggle, canvas_size, px,
                               ui_display, ui_font, ui_mono)

log = get_logger("ui.sensors_page")

# ---------------------------------------------------------------- the words
DASH = "—"                    # every unknown scalar; never "0" and never ""

# THE ZONES ARE THE CONFIG'S, NOT THIS FILE'S. There is no ZONE_DESK here
# any more: a zone is whatever ``zones.rooms[].bands[].name`` calls it, and
# the words on screen are that name in capitals. The page used to carry its
# own four-word vocabulary over a TWO-band model of its own, and that model
# disagreed with the one the zone log writes -- see the module docstring.
# The three verdicts that are not a band come straight from jarvis/zones.py
# so there is one spelling of each in the tree.
NO_LADDER = "no zone ladder"     # this page only: the room has no bands

# Tone KEYS, not colours: a colour resolved out here would freeze the
# import-time look (tests/test_theme_look.py scans this file for exactly
# that). tone_color() resolves one at paint time.
TONE_OK, TONE_WARN, TONE_ERR, TONE_MUTED, TONE_FAINT = (
    "ok", "warn", "err", "muted", "faint")
TONES = (TONE_OK, TONE_WARN, TONE_ERR, TONE_MUTED, TONE_FAINT)

# The standing caption, and the one shown after a write. They are separate
# because the standing one sat under an unpressed SAVE button reading
# "saved to assistant.json" -- which is a claim, not a caption.
# One short line each: they sit in a caption column beside a button, and
# the 09-05 pass found the standing one long enough to wrap there.
RESTART_NOTE = "applies at the next Jarvis restart"
SAVED_NOTE = "saved — restart Jarvis to apply it"
NOT_WIRED_NOTE = "assistant settings not wired"
# What jarvis-v3 printed here. CLASSIC IS FROZEN (see `restyled`), and this
# caption is on the frame, so the old sentence has to still exist.
RESTART_NOTE_V3 = ("edits apply at the next Jarvis restart — nothing "
                   "reloads the config")


# ------------------------------------------------------------ the look gate
def restyled(look: Optional[str] = None) -> bool:
    """True when the 2026-09-05 relayout draws this page; False for the
    look that must render EXACTLY as jarvis-v3 4b7d373 did.

    WHY THIS EXISTS. He asked for the SENSORS tab and the settings area to
    be cleaned up. He did not ask for the classic look to change, and the
    09-05 relayout changed it by 175,076 px of 1,324,800 at 920x1440 and
    202,322 px at the window he actually runs (1040x1760) -- MEASURED on
    the photo rig, frames 25/26/27. So classic is FROZEN at the v3 tip and
    every 09-05 improvement is holo's, which is the look he runs
    (theme.DEFAULT_LOOK).

    A CORRECTION TO THE RIG NOTE THIS WORK STARTED FROM. Frames 25, 26 and
    27 were said to be the rig's byte-stable frames. 25 and 27 are; 26 is
    NOT. MEASURED both ways: two renders of the SAME v3 tree differ by 268
    px at y418..435 x936..962 at 1040x1760, and two renders of the same
    FIXED tree differ by 309 px at y448..465 x816..842 at 920x1440 -- the
    round-trip readout, which is a live measurement and flips a digit
    between runs. It sometimes matches by luck, which is how it came to be
    called stable. A 0-px claim about frame 26 needs the same-code A/B
    beside it.

    READ AT CALL TIME, never captured at def time: a look token captured
    when the module is imported freezes the import-time look, and
    tests/test_theme_look.py scans this file for exactly that.
    """
    return (look or theme.LOOK) == "holo"


def restart_note() -> str:
    """The standing caption under SAVE. Read at CALL time, so the frozen
    classic look keeps the sentence jarvis-v3 printed."""
    return RESTART_NOTE if restyled() else RESTART_NOTE_V3

# ------------------------------------------------------------- the geometry
# The distance entity, addressed by its NAME exactly as the presence one is:
# web_server v2 serves an entity at its name, percent-encoded, NOT at the
# snake_case object_id. Getting that wrong is a silent 404 -- it cost this
# tree every poll and every tune write until 00d5b7e. entity_path() owns the
# rule; nothing here spells a path a second time.
DISTANCE_ENTITY = "Detection distance"
DISTANCE_PATH = entity_path("sensor", DISTANCE_ENTITY)

MAX_BAND_M = 6.0              # 8 gates x 0.75 m: the module's own ceiling
BLIND_M = zn.BLIND_M          # nothing at all is detected inside this
STILL_FLOOR_M = zn.STILL_FLOOR_M   # no STILL target inside this (gates 0, 1)

# THE ONE KEY THE BANDS LIVE IN. Written whole rather than per-room, because
# a band list is a list and AssistantConfig replaces lists rather than
# merging them; band_edits therefore edits a COPY of what the file holds and
# hands back the whole thing, so a room this page never showed keeps its
# ladder byte for byte.
OPTION_ROOMS = zn.ROOMS_KEY                  # "zones.rooms"
OPTION_CAMERA_OVERRULES = "presence.camera_overrules"

# THE SUPERSEDED KEYS. A two-band model that assumed the desk was the NEARER
# band, which in his office is backwards -- the near space is empty and the
# desk is at 3.13 m. They are read HERE and nowhere else in the tree, once,
# as a migration source: a config that has them and no usable zones.rooms
# gets its numbers carried across rather than dropped, and SAVE writes them
# into zones.rooms and then REMOVES them, so the file stops holding a key
# that looks live and drives nothing. They are absent from
# assistant_config.DEFAULTS, so a value in either means his file still has
# it.
LEGACY_DESK_BAND = "presence.desk_band_m"
LEGACY_ROOM_BAND = "presence.room_band_m"
LEGACY_KEYS = (LEGACY_DESK_BAND, LEGACY_ROOM_BAND)
# What the two legacy bands were called on screen, and what they become as
# named bands when they are carried across.
LEGACY_NEAR_NAME = "at the desk"
LEGACY_FAR_NAME = "in the room"

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
class Ladders:
    """Every room's ladder, plus where each one came from and what the
    config lost getting here.

    ``maps`` is keyed by room key (``zones._room_key``) and holds
    ``jarvis.zones.ZoneMap`` -- the SAME object the zone log places a
    reading with, which is the whole point of this pass. ``legacy`` is the
    one ladder carried across from the superseded two-band keys, used for
    any room that has none of its own. ``raw`` is ``zones.rooms`` exactly
    as the file holds it, which is what SAVE edits a copy of.
    """
    maps: dict = field(default_factory=dict)
    legacy: Optional[ZoneMap] = None
    raw: Any = None
    notes: tuple = ()
    refused: frozenset = frozenset()      # room keys jarvis/zones.py refused
    blocked: str = ""                     # why SAVE may not touch the bands

    def for_room(self, room: str) -> Optional[ZoneMap]:
        """The ladder that places a reading in ``room``, or None.

        The room's own entry wins; the carried-across legacy ladder stands
        in only where there is no entry at all. A room whose entry is
        REFUSED gets None, never the legacy one -- jarvis/zones.py refuses
        a broken ladder rather than substituting a working one, and a page
        that substituted here would be showing him a verdict the log will
        not write.
        """
        key = zn._room_key(room)
        got = self.maps.get(key)
        if got is not None:
            return got
        return None if key in self.refused else self.legacy

    def source(self, room: str) -> str:
        """Which config key this room's bands came from ("" = none)."""
        key = zn._room_key(room)
        if key in self.maps:
            return OPTION_ROOMS
        if key not in self.refused and self.legacy is not None:
            return LEGACY_DESK_BAND
        return ""


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


# ---------------------------------------------------------------- the bands
def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _legacy_pair(value) -> Optional[tuple]:
    """One superseded band -> (near, far), or None for "not usable".

    REORDERING IS FREE, EVERYTHING ELSE IS REFUSED. A hand-edit that
    swapped the ends loses nothing and is ordered silently. Anything that
    is not two usable metres -- centimetres, a word, a range past what the
    radar can see -- is None, and the caller NAMES it.

    It used to fall back to a built-in default pair, which is the disease
    this whole pass is closing: measured 2026-09-03,
    ``presence.room_band_m = [1.8, 8.0]`` rendered as 1.8/4.5 with no note
    anywhere and one press of SAVE wrote the 4.5 over his 8.0. There is no
    default left to fall back TO -- the keys are gone from DEFAULTS and
    these are a migration source, not a live model -- so a value that
    cannot be carried across is said out loud and carried across as
    nothing.
    """
    try:
        lo, hi = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    lo, hi = min(lo, hi), max(lo, hi)
    if lo < 0.0 or hi > MAX_BAND_M or lo >= hi:
        return None
    return lo, hi


def _opt(get_option: Optional[Callable], key: str, default=None):
    """One config read that never raises and never turns null into a
    value."""
    if not callable(get_option):
        return default
    try:
        value = get_option(key, default)
    except Exception:                     # noqa: BLE001 - config boundary
        log.debug("sensors page: %s unreadable", key, exc_info=True)
        return default
    return default if value is None else value


def legacy_ladder(get_option: Optional[Callable]) -> tuple:
    """(the ladder the superseded keys describe, notes). Never raises.

    ``(None, ())`` when neither key is in his file, which is the normal
    case and not a complaint. The two bands become NAMED bands -- the near
    one keeps the word the old page put on screen for it -- so that what is
    carried into ``zones.rooms`` reads like the rest of the ladder rather
    than like a migration artefact.

    THE ORDER IS TAKEN FROM THE NUMBERS, NOT FROM THE KEY NAMES. The old
    model assumed the desk was the NEARER band; in his office it is the
    FARTHER one (the near space is empty and he sits at 3.13 m). So
    ``desk_band_m`` keeps the name "at the desk" wherever its numbers put
    it, and ZoneMap sorts the ladder.
    """
    raw = {key: _opt(get_option, key) for key in LEGACY_KEYS}
    if all(value is None for value in raw.values()):
        return None, ()
    notes, bands = [], []
    for key, name in ((LEGACY_DESK_BAND, LEGACY_NEAR_NAME),
                      (LEGACY_ROOM_BAND, LEGACY_FAR_NAME)):
        value = raw[key]
        if value is None:
            continue
        pair = _legacy_pair(value)
        if pair is None:
            log.warning("sensors page: %s is not usable (%s); it is not "
                        "carried across", key, _short(value))
            notes.append("%s in assistant.json is not usable (%s) — two "
                         "numbers, 0 to %.0f m — so it is NOT carried into "
                         "%s and nothing on this page uses it"
                         % (key, _short(value), MAX_BAND_M, OPTION_ROOMS))
            continue
        bands.append(Band(name, pair[0], pair[1]))
    if not bands:
        return None, tuple(notes)
    try:
        zmap = ZoneMap("room", tuple(bands), zn.DEFAULT_CAMERA_ZONE)
    except ValueError as exc:
        # Two bands that overlap are not a ladder, and jarvis/zones.py is
        # the one judge of that -- the page must not accept a geometry the
        # log would refuse.
        notes.append("%s and %s in assistant.json do not describe a ladder "
                     "(%s), so neither is carried into %s"
                     % (LEGACY_DESK_BAND, LEGACY_ROOM_BAND, exc, OPTION_ROOMS))
        return None, tuple(notes)
    return zmap, tuple(notes)


def read_ladders(get_option: Optional[Callable]) -> Ladders:
    """Every room's bands, through jarvis/zones.py and nothing else.

    ONE ZONE MODEL. ``zones.read_zones`` is THE validator for that section
    -- the same call ``scripts/zone_log.py`` makes -- so a ladder this page
    shows is a ladder the log will write against, and a ladder it refuses
    is refused here too, by name, rather than quietly replaced with
    something that looks like it works.

    What this adds on top is only reporting: a refused room becomes a
    sentence on screen instead of a line in the log he is not reading, and
    the superseded two-band keys are carried across for a room that has no
    entry of its own. Where both exist, ``zones.rooms`` WINS and the page
    says so -- one of them has to, and it is the one that matches his
    measured room.
    """
    cfg = _CfgView(get_option)
    try:
        parsed = zn.read_zones(cfg)
    except Exception:                     # noqa: BLE001 - config boundary
        log.exception("sensors page: the zones section could not be read")
        parsed = None
    notes: list = []
    maps: dict = {}
    refused: set = set()
    raw = None
    # A section that got itself wrong ABOVE the room level poisons
    # everything: there are no entries to salvage, and a SAVE from here
    # would replace whatever is in the file with a list built out of a page
    # showing nothing. Refuse the write and say which key to go and fix.
    blocked = ""
    if parsed is None:
        blocked = ("the %s section could not be read at all; fix it in "
                   "assistant.json before saving from here" % zn.SECTION_KEY)
    elif parsed.poisoned:
        blocked = ("%s — fix that in assistant.json before saving from here"
                   % (sorted(parsed.refused.values())[0] if parsed.refused
                      else "the %s section is unusable" % zn.SECTION_KEY))
    if parsed is not None:
        maps = dict(parsed.rooms)
        refused = set(parsed.room_refusals)
        # From the validator, NOT a dotted read of our own: jarvis/zones.py
        # is the only door onto that section, for a writer as much as for a
        # reader.
        raw = parsed.raw_rooms
        for path, why in sorted(parsed.refused.items()):
            notes.append("%s — that room records nothing and is not shown "
                         "with a zone until it is fixed" % why)
    legacy, legacy_notes = legacy_ladder(get_option)
    notes.extend(legacy_notes)
    if legacy is not None and maps:
        # Both models are in his file. zones.rooms wins; say which, and say
        # that SAVE clears the loser rather than leaving it looking live.
        notes.append("%s and %s are still in assistant.json and are NOT "
                     "read: the bands come from %s. SAVE removes them."
                     % (LEGACY_DESK_BAND, LEGACY_ROOM_BAND, OPTION_ROOMS))
    elif legacy is not None:
        notes.append("the bands were carried across from %s and %s; SAVE "
                     "writes them into %s and removes the old keys"
                     % (LEGACY_DESK_BAND, LEGACY_ROOM_BAND, OPTION_ROOMS))
    return Ladders(maps=maps, legacy=legacy, raw=raw, notes=tuple(notes),
                   refused=frozenset(refused), blocked=blocked)


def name_mismatch_note(specs, ladders: Ladders) -> tuple:
    """The sentence for "these two lists do not use the same room names".

    The page polls the rooms in ``presence.rooms`` (or the singular
    ``presence.room_sensor_*`` keys, which name the room "room") and places
    them with the ladders in ``zones.rooms``. The join is the room NAME, and
    when it does not join there is no verdict for any row and nothing in
    the log to say why -- which is exactly the state his config is in
    tonight: ``presence.rooms`` is empty, so one room called "room" is
    polled, while ``zones.rooms`` names "office" and "kitchen".

    Silent when every polled room has a ladder, and silent when there are
    no ladders at all (that is a different sentence, and read_ladders
    already has it).
    """
    polled = [getattr(s, "name", "") for s in (specs or ())
              if getattr(s, "name", "")]
    if not polled or not ladders.maps:
        return ()
    missing = [name for name in polled if ladders.for_room(name) is None]
    if not missing:
        return ()
    return ("%s %s polled but %s no ladder: %s names %s and %s names %s. "
            "They join on the room NAME — make the two lists agree, and "
            "restart."
            % (" and ".join(repr(m) for m in missing),
               "are" if len(missing) > 1 else "is",
               "have" if len(missing) > 1 else "has",
               "presence.rooms", ", ".join(repr(n) for n in polled),
               OPTION_ROOMS, ", ".join(repr(n) for n in sorted(ladders.maps))),)


def zone_word(zone: str) -> str:
    """The verdict, as the page prints it: the band's own name, in capitals.

    ``NO OPINION`` is deliberately not "EMPTY" and NOBODY is deliberately
    not a band name -- both come from jarvis/zones.py, where the argument
    that a non-answer is not absence was already settled.
    """
    return str(zone or "").upper()


def band_notes(zmap: Optional[ZoneMap]) -> tuple:
    """Hardware facts his bands just walked into, in his language.

    Both come from the LD2410's own geometry (scripts/room_sensor.py, which
    refuses to flash a profile that violates them): the module detects
    NOTHING inside 0.75 m, and no STILL target inside 1.5 m -- so a band
    that lives entirely below 1.5 m reads a motionless man as an empty
    room, which is the PIR failure the radar was chosen to avoid. Notes,
    not refusals: it is his wall and his desk.

    NAMED PER BAND, because the ladder is his and "the desk band" is no
    longer a thing this file knows about. The band a note is about is the
    band he has to go and change.
    """
    if zmap is None:
        return ()
    out = []
    for band in zmap.bands:
        if band.near_m < BLIND_M:
            out.append("%r starts at %.2f m, inside the %.2f m the radar "
                       "detects nothing at all in" % (band.name, band.near_m,
                                                      BLIND_M))
        if band.far_m <= STILL_FLOOR_M:
            out.append("%r ends at %.2f m and no STILL target is reported "
                       "inside %.1f m, so a motionless man there may read as "
                       "an empty room" % (band.name, band.far_m, STILL_FLOOR_M))
    return tuple(out)


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


def _finite(value) -> Optional[float]:
    """A float, or None for anything a typed box or a config can hold."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def room_scale(pairs) -> Optional[tuple]:
    """(lo, hi): the whole ladder a room's bands are drawn on, or None
    when nothing usable was given."""
    ok = [(a, b) for a, b in ((_finite(x), _finite(y)) for x, y in pairs or ())
          if a is not None and b is not None and b > a]
    if not ok:
        return None
    return min(a for a, _b in ok), max(b for _a, b in ok)


def band_span(near, far, lo, hi) -> Optional[tuple]:
    """Where one band sits on its room's scale, as (x0, x1) fractions.

    THIS IS WHY THE BARS STOPPED BEING SLIDERS (2026-09-05, round 1). Each
    bar used to be a full-width track from the band's near edge to its far
    edge with the live range marked on it -- and the mark is absent
    whenever the range is outside the band, which is most of the time.
    MEASURED on the 09-05 frames: one of four bars carried a mark on the
    live shot and NONE of four on the fault shot, so what he saw was four
    identical tracks with two end stops and nothing between them, i.e.
    four sliders with the handle missing (his words). A bar drawn as the
    band's SHARE of the room instead is ink he can read with no live
    reading at all, the bands step across the row like the ladder they
    are, and a gap in the ladder is a gap on the page.
    """
    near, far, lo, hi = (_finite(near), _finite(far), _finite(lo),
                         _finite(hi))
    if None in (near, far, lo, hi) or hi <= lo or far <= near:
        return None
    span = hi - lo
    x0 = max(0.0, min(1.0, (near - lo) / span))
    x1 = max(0.0, min(1.0, (far - lo) / span))
    return (x0, x1) if x1 > x0 else None


def spans_for_room(pairs) -> tuple:
    """Every band of ONE room, in order, as spans on that room's scale.

    Fed from the typed boxes as well as from the config, so a band he is
    widening grows under his hands; a box holding junk drops that one bar
    and leaves the rest of the ladder alone.
    """
    pairs = tuple(pairs or ())
    scale = room_scale(pairs)
    if scale is None:
        return tuple(None for _ in pairs)
    return tuple(band_span(a, b, scale[0], scale[1]) for a, b in pairs)


@dataclass(frozen=True)
class BandEdit:
    """One band as the page holds it while he types in it.

    The NAME rides along unchanged: this page tunes the numbers, and a
    band's name is what the zone log will call the place he was standing.
    """
    room: str
    name: str
    lo: str
    hi: str


def _typed(label: str, raw) -> tuple:
    """One typed metre value -> (number, "") or (None, why it is refused).

    Typing is REFUSED rather than repaired, which is the opposite of how
    the superseded keys were READ, and deliberately so: a person who has
    just typed something wrong should be told, not silently corrected.
    """
    text = str(raw).strip()
    try:
        num = float(text)
    except (TypeError, ValueError):
        return None, "%s: %r is not a number" % (label, text)
    if not math.isfinite(num):
        return None, "%s: %r is not a number" % (label, text)
    if num < 0.0 or num > MAX_BAND_M:
        return None, ("%s: %.2f m is outside what the radar can see "
                      "(0 to %.0f m)" % (label, num, MAX_BAND_M))
    return round(num, 2), ""


def band_edits(edits, ladders: Ladders, overrules) -> tuple:
    """(the dotted keys to write, "") or ({}, why it was refused).

    ``edits`` is the ``BandEdit`` rows the page is showing, in order. The
    result is exactly TWO keys -- ``zones.rooms`` and the overrule toggle
    -- because there is one zone model now and its bands live in one place.

    THE WRITE IS A MERGE, NOT A REBUILD. It starts from ``ladders.raw``,
    which is ``zones.rooms`` as the file holds it, and replaces only the
    ``bands`` of the rooms this page actually showed. A room he has
    configured but is not polling, a room's ``camera_zone``, its
    ``enabled`` flag and every key a later Jarvis might add all survive
    untouched -- and a room that only had the carried-across legacy bands
    is APPENDED, which is what completes the migration.

    THE GEOMETRY IS JUDGED BY jarvis/zones.py AND NOT HERE. Every room's
    new ladder is built as a ``ZoneMap`` before anything is written, so a
    pair of overlapping bands is refused on screen instead of being saved
    and then refused by the log -- which is how a config gets into the
    state where the page shows one thing and the record says another.
    """
    rows = list(edits or ())
    if ladders.blocked:
        return {}, ladders.blocked
    if not rows:
        # Nothing on screen to save. Writing the bands anyway would replace
        # his list with one built out of an empty page.
        return {OPTION_CAMERA_OVERRULES: bool(overrules)}, ""
    values: dict = {}
    for row in rows:
        label = "%s / %s" % (row.room, row.name)
        pair = []
        for raw in (row.lo, row.hi):
            num, err = _typed(label, raw)
            if err:
                return {}, err
            pair.append(num)
        if pair[0] >= pair[1]:
            return {}, ("%s: the near end (%.2f) must be less than the far "
                        "end (%.2f)" % (label, pair[0], pair[1]))
        values.setdefault(row.room, []).append(
            {"name": row.name, "near_m": pair[0], "far_m": pair[1]})
    for room, bands in values.items():
        try:
            ZoneMap(room, tuple(Band(b["name"], b["near_m"], b["far_m"])
                                for b in bands),
                    _camera_zone_of(ladders, room))
        except ValueError as exc:
            return {}, "%s: %s" % (room, exc)
    raw = ladders.raw
    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)):
        return {}, ("%s in assistant.json is %s, not a list of rooms — fix "
                    "that by hand before saving from here"
                    % (OPTION_ROOMS, type(raw).__name__))
    return {OPTION_ROOMS: _merged_rooms(raw, values, ladders),
            OPTION_CAMERA_OVERRULES: bool(overrules)}, ""


def _camera_zone_of(ladders: Ladders, room: str) -> str:
    """The room's camera zone, carried through an edit unchanged.

    "" is a real answer and means the room has no lens (jarvis/zones.py,
    ``ZoneMap.has_camera``); it must survive a band edit rather than being
    replaced with the built-in default, which is the exact bug this pass
    closed one file over.
    """
    zmap = ladders.for_room(room)
    return "" if zmap is None else zmap.camera_zone


def _merged_rooms(raw, bands_by_room: dict, ladders: Ladders) -> list:
    """``zones.rooms`` with only the edited rooms' bands replaced."""
    out = []
    seen = set()
    for entry in raw:
        if not isinstance(entry, dict):
            out.append(copy.deepcopy(entry))   # not ours to judge; not ours
            continue                           # to lose either
        key = zn._room_key(entry.get("name"))
        fresh = dict(entry)
        if key in bands_by_room:
            fresh["bands"] = copy.deepcopy(bands_by_room[key])
            seen.add(key)
        out.append(fresh)
    for key, bands in bands_by_room.items():
        if key in seen:
            continue
        # The migration write: a room that had no entry of its own gets one,
        # carrying the camera zone the ladder was showing.
        out.append({"name": key, "enabled": True,
                    "camera_zone": _camera_zone_of(ladders, key),
                    "bands": copy.deepcopy(bands)})
    return out


# ----------------------------------------------------------------- the write
def retire_legacy(get_option: Optional[Callable],
                  unset_option: Optional[Callable]) -> tuple:
    """Remove the superseded band keys once their numbers are in
    ``zones.rooms``. Returns the notes that are LEFT to show.

    A key that has been superseded and is merely IGNORED still sits in his
    file looking live, with plausible numbers in it and no way to tell it
    from one that drives something -- which is exactly how two zone models
    disagreed for a day without anyone noticing. So it goes, on the save
    that carried its numbers across.

    Where the services object has no ``unset_option`` (an older app, a
    stand-in) the keys STAY and the page SAYS they stay. Silently leaving a
    live-looking lie in the file is the one outcome this function exists to
    prevent, so it may not be the quiet failure mode either.

    Never raises: a write that will not happen is a note, not a crash on
    the Tk thread.
    """
    held = [key for key in LEGACY_KEYS if _opt(get_option, key) is not None]
    if not held:
        return ()
    if not callable(unset_option):
        return ("%s %s still in assistant.json and no longer read; the bands "
                "live in %s now — remove them by hand"
                % (" and ".join(held), "are" if len(held) > 1 else "is",
                   OPTION_ROOMS),)
    for key in held:
        try:
            unset_option(key)
        except Exception:                 # noqa: BLE001 - config boundary
            log.exception("sensors page: %s would not clear", key)
    left = [key for key in held if _opt(get_option, key) is not None]
    if left:
        return ("%s could not be removed from assistant.json; %s no longer "
                "read — the bands live in %s"
                % (" and ".join(left), "they are" if len(left) > 1 else "it is",
                   OPTION_ROOMS),)
    return ()


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
    def sees_a_face(self) -> bool:
        """Whether the eye has an opinion at all right now.

        It no longer returns a ZONE. Naming the place was this page's own
        two-band model talking: WHICH place a recognised face means is the
        room's ``camera_zone`` in ``zones.rooms``, which is "at the desk"
        in his office and DELIBERATELY BLANK in his kitchen -- a room with
        no lens. WHO it is rides separately in ``name``; this page does not
        fold identity into geometry.
        """
        return bool(self.present and self.live and self.faces)



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
        # AN UNRECOGNISED FACE IS NOT A FAULT (2026-09-05, round 2). This
        # line was amber and untested -- the one state on the page that
        # broke the rule the round-1 pass wrote three docstrings about
        # ("amber is reserved for a fault ... it is the only orange thing
        # on the page"). A camera that sees a face it cannot name HAS
        # answered: it reports a person and the confidence it reached, and
        # the score beside the word already says how sure it is. Nothing is
        # broken, so nothing is amber; it reads like the other two
        # answered-but-no-identity lines (NO FACE, identity not running).
        # Classic is frozen at v3, which painted it amber.
        return ("FACE  UNKNOWN  %.2f" % view.score,
                TONE_MUTED if restyled() else TONE_WARN)
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


def _fuse_v3(v, *, saw_face: bool, speak: bool, distance_m, zmap) -> Verdict:
    """The reason line and the source EXACTLY as jarvis-v3 4b7d373 wrote
    them, for the frozen classic look (``restyled``).

    Kept as its own function rather than as branches inside ``fuse`` so
    the frozen sentences are one block that can be diffed against the v3
    tip, and so nothing here can drift when the holo wording changes
    again. Two real differences live in here besides the wording: the
    camera clause was APPENDED to the rule's sentence with a semicolon
    (which is what made the line wrap), and RULE_SILENT still named the
    radar whenever a face had been seen.
    """
    if v.rule == zn.RULE_BAND:
        why = "a target inside %r" % v.zone
    elif v.rule == zn.RULE_UNPLACED:
        why = ("presence with no usable range — someone is in the room, the "
               "band is unknown" if distance_m is None else
               "%.2f m is in a gap between the bands, so the room is the "
               "only honest answer" % distance_m)
    elif v.rule == zn.RULE_EMPTY:
        why = "the radar reads empty"
    else:
        why = "neither leg has an opinion — this is not an empty room"
    if saw_face and not speak:
        why += "; the camera sees a face but is not allowed to overrule"
    elif saw_face:
        why += ("; the camera sees a face but %r has no camera zone in %s"
                % (zmap.room, OPTION_ROOMS))
    source = "" if v.rule == zn.RULE_SILENT and not saw_face else "radar"
    return Verdict(v.zone, zone_word(v.zone), source, why)


def fuse(*, present: Optional[bool], distance_m, camera: CameraView,
         zmap, overrules: bool) -> Verdict:
    """The one fusion rule -- and it is ``jarvis.zones.verdict``, not a
    second copy of it living on a Tk page.

    THAT IS THE WHOLE POINT OF THIS FUNCTION. Until 2026-09-03 this page
    had a fusion of its own over a two-band model of its own, and the
    merge agent measured the two disagreeing: the console called 4 of 10
    points "AT THE DESK" from the radar alone while the zone log had no
    desk band at all and could only reach "at the desk" through the camera.
    They were different config keys, so nothing conflicted and nothing
    complained. What is left here is the ADAPTER: it decides whether the
    camera gets to speak, and jarvis/zones.py decides what the answer is.

    THE ORDER, which is that module's and is argued there:

    1. The camera, when he lets it AND the room has one. His words:
       "camera recognition overrules sensor detection since he can literaly
       see me at my desk". A room whose ``camera_zone`` is blank has no
       lens and the rule cannot fire for it at all (``ZoneMap.has_camera``).
    2. The radar's presence bit, placed by its band. Presence with NO
       usable range is "in the room, unplaced" -- never the nearest band.
       Inventing a place out of a missing number is the failure this page
       was built to expose.
    3. Presence False is "not in the room", and only because a leg said so.
    4. Both legs silent is "no opinion". Not an empty room.

    With the toggle OFF and the radar abstaining, the camera still answers:
    nothing is being overruled when the other leg did not speak, and
    refusing the only evidence in the room would be a different lie.

    A room with no ladder gets NO LADDER and no invented bands. The page
    can still say PRESENT and print a range; what it cannot honestly do is
    name a place.
    """
    saw_face = bool(camera is not None and camera.sees_a_face)
    new = restyled()
    if zmap is None:
        return Verdict(NO_LADDER, zone_word(NO_LADDER), "",
                       "no bands are set for this room" if new else
                       "there is no ladder for this room in %s, so a range "
                       "cannot be given a name" % OPTION_ROOMS)
    speak = saw_face and (overrules or present is None)
    opinion = zn.CameraOpinion(known=True, label=camera.name) if speak else None
    v = zn.verdict(zmap, presence=present, distance_m=distance_m,
                   camera=opinion)
    if v.rule == zn.RULE_CAMERA:
        if new:
            why = ("the camera sees a face; the radar is not asked"
                   if present is not None else
                   "the radar is silent; the camera sees a face")
        else:
            why = ("the camera can see a face; the radar's range is not asked"
                   if present is not None else
                   "the radar has no opinion; the camera can see a face")
        return Verdict(v.zone, zone_word(v.zone), "camera", why)
    if not new:
        return _fuse_v3(v, saw_face=saw_face, speak=speak,
                        distance_m=distance_m, zmap=zmap)
    # ONE SHORT LINE (2026-09-05). These sentences used to run to 75
    # characters and then GREW a semicolon clause about the camera, so at
    # his window they wrapped mid-phrase into a second grey line under a
    # block that already had four. The reason line is now one line, and
    # when the camera saw a face and still did not win, THAT is the line --
    # it is the surprising fact, and the verdict word already carries what
    # the radar decided.
    if saw_face and not speak:
        why = "the camera sees a face but may not overrule"
    elif saw_face:
        # It was allowed and still did not win, which leaves exactly one
        # reason: this room's camera_zone is blank, i.e. the config says
        # there is no lens here (jarvis/zones.py, ZoneMap.has_camera).
        why = "the camera sees a face; this room has no camera zone"
    elif v.rule == zn.RULE_BAND:
        # NOT "a target inside %r": the verdict word beside it already
        # names the band, and %r put Python's quotes on the page.
        why = "the range falls inside this band"
    elif v.rule == zn.RULE_UNPLACED:
        why = ("someone is in the room; the range is unknown"
               if distance_m is None else
               "%.2f m is in a gap between the bands" % distance_m)
    elif v.rule == zn.RULE_EMPTY:
        why = "the radar reads empty"
    else:
        why = "neither leg has an opinion — not an empty room"
    # RULE_SILENT is "neither leg answered", and a face the camera was not
    # allowed to use is not an answer: naming the radar there printed
    # "NO OPINION · radar" under a radar that had said nothing (measured
    # under pytest 09-05, a room whose camera_zone is blank).
    source = "" if v.rule == zn.RULE_SILENT else "radar"
    return Verdict(v.zone, zone_word(v.zone), source, why)


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
    """(word, tone) for the presence bit.

    NO OPINION is FAINT, not amber (2026-09-05, round 1). His brief:
    "reserve orange for a fault only ... make it the only orange." The
    round-0 pass fixed the explanation lines and the amber count did not
    move -- MEASURED on the frames, 3 amber lines before and 3 after on
    the live shot, 6 and 6 on the fault shot -- because a dark room still
    shouted the state word twice, once in the header and once on the radar
    leg. Three tones already tell the three states apart (PRESENT reads
    live, EMPTY reads muted, NO OPINION reads faintest of all), and the
    amber fault line directly under it says WHY there is no answer.
    """
    if present is True:
        return "PRESENT", TONE_OK
    if present is False:
        return "EMPTY", TONE_MUTED
    # Classic is frozen at v3, which shouted this one amber (`restyled`).
    return "NO OPINION", TONE_FAINT if restyled() else TONE_WARN


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


def page_rows(readings, camera_status: Any, ladders: Ladders, overrules: bool,
              camera_room: str = "") -> list:
    """Every configured room, in config order, ready to paint.

    The ladder is looked up PER ROOM, so the kitchen is placed by the
    kitchen's bands and the office by the office's -- and a room with no
    ladder says so instead of borrowing another room's geometry.
    """
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
                         camera=cam, zmap=ladders.for_room(r.name),
                         overrules=overrules)))
    return out



# ------------------------------------------------- what a room block draws
# THE TYPE RULES (2026-09-05). His words: "they look a little unprofessional
# and some of text doesnt sit right." Three type styles were fighting inside
# one room block -- a bold display header beside a large display word beside
# a large MONOSPACE range beside a small grey caption -- and the two
# explanation sentences mixed grey and orange with no legend for which was
# which. The rules the widget renders by, kept out here so they are tested
# with no display:
#
#   * ONE sans face for every label and every word; the monospace face is
#     for NUMERIC READOUTS ONLY (readout_is_numeric).
#   * TWO sizes in a block: the header line, and the detail lines.
#   * ONE muted style for the reason line. AMBER IS RESERVED FOR A FAULT,
#     so on this page the fault line is the only orange thing (see
#     explanation_lines) and it appears only when a leg had no opinion.
#     MEASURED off the rendered frames at 920x1440 (amber ink runs in the
#     page body, anti-aliasing filtered): the live shot went 3 -> 3 -> 1
#     across the two passes and the fault shot 6 -> 6 -> 2, which is one
#     fault line per dark room and nothing else. The first pass fixed the
#     reason lines and moved the count not at all, because the STATE WORD
#     was amber twice over in a dark room (presence_words, verdict_tone).
#   * The verdict's SOURCE sits beside the verdict word, not glued onto the
#     front of the reason sentence.


def readout_is_numeric(text: str) -> bool:
    """True when a readout is a NUMBER and may be drawn in the monospace
    face. The dash is a word: the KITCHEN header drew its '—' placeholder
    in mono beside a sans header, which is the sort of thing that reads as
    a broken font rather than as a missing value."""
    return any(ch.isdigit() for ch in (text or ""))


def verdict_tone(zone: str) -> str:
    """The tone the fused word wears. A zone that was NAMED is the page's
    answer and reads live; a zone nobody could name is the faintest thing
    on the row. Neither is amber -- see presence_words."""
    return TONE_FAINT if zone in (NO_OPINION, NO_LADDER) else TONE_OK


def source_text(verdict: Verdict) -> str:
    """The leg that answered, as it is drawn AFTER the verdict word
    ("AT THE DESK · camera"), or "" when neither leg spoke."""
    return ("· %s" % verdict.source) if getattr(verdict, "source", "") else ""


def explanation_lines(row: "RoomRow") -> tuple:
    """((text, tone), ...) for the lines under a room's readouts.

    At most two, in this order: the reason, always, in the muted tone; the
    fault, only when there is one, in the warning tone. Nothing else on
    this page is amber, so orange means "a leg could not answer" and needs
    no legend.
    """
    out = [(row.verdict.why, TONE_FAINT)]
    if row.fault:
        out.append((row.fault, TONE_WARN))
    return tuple(out)


def marker_shape(look: Optional[str] = None) -> str:
    """"needle" | "dot": how a band bar draws the live range.

    The bars are READOUTS, not sliders -- the marker is absent when the
    range is outside the band (band_fraction), which is the one honest
    thing the strip can do. Changing the marker was not enough on its own,
    though: MEASURED on the rendered frames, 0 of 4 bars carried a solid
    span and only the one bar holding the live range carried a mark at
    all, so four bare tracks still read as four broken sliders. What fixed
    that is the SPAN (band_span, _BandBar); the marker shape is the second
    half of it. In holo it is a needle across the band; classic keeps the
    dot it was pinned with.
    """
    return "needle" if (look or theme.LOOK) == "holo" else "dot"


WAITING_WORD = "READING…"


def waiting_row(spec) -> RoomRow:
    """The row shown before the first poll lands. Every value is a dash and
    the state says READING, because a blank row on a presence page reads as
    an empty room."""
    return RoomRow(name=getattr(spec, "name", ""), label=_spoken(spec),
                   presence_word=WAITING_WORD, presence_tone=TONE_MUTED,
                   distance_text=DASH, distance_m=None, rtt_text=DASH,
                   fault="", camera_text=DASH, camera_tone=TONE_FAINT,
                   verdict=Verdict(NO_OPINION, zone_word(NO_OPINION), "",
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
    """``——[####]————`` : one band's own stretch of its room's ladder.

    NOT A SLIDER, AND IT NO LONGER LOOKS LIKE ONE (2026-09-05, round 1).
    It used to draw a full-width track with an end stop at each end and a
    marker for the live range -- and ``band_fraction`` returns None
    whenever that range is outside this band, which is most of the time
    and ALWAYS when nothing is being read. MEASURED on the 09-05 frames:
    one of four bars carried a marker on the live shot, none of four on
    the fault shot. Four bare tracks with two end stops read as four
    sliders whose handle had gone missing, which is exactly what he
    reported.

    So the bar draws what it actually knows. The rail is the room's WHOLE
    ladder; the solid bar is this band's share of it (``band_span``); the
    live range is a mark inside the bar, and only inside the bar that owns
    it. Two bands of one room therefore start at different x -- a shape no
    slider can have -- a gap in the ladder shows as a gap, and every row
    carries ink with no reading at all.
    """

    H = 18                                # design units
    BAR = 4                               # the solid band, in design units

    def __init__(self, parent, bg=None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, height=px(type(self).H), bg=bg,
                         highlightthickness=0, bd=0)
        self._frac: Optional[float] = None
        self._span: Optional[tuple] = None
        self.bind("<Configure>", lambda e: self._draw(), add=True)

    def set_span(self, span: Optional[tuple]) -> None:
        """Where this band sits on the room's scale, or None when the
        ladder cannot be read (a box holding junk while he types)."""
        self._span = span
        self._draw()

    def set(self, fraction: Optional[float]) -> None:
        """Where the live range sits INSIDE this band, or None when it is
        outside it -- a mark parked on an end would read as a reading that
        is in the band, which is the one thing this strip must never say."""
        self._frac = fraction
        self._draw()

    def marked(self) -> bool:
        return self._frac is not None and self._span is not None

    def _rail(self) -> tuple:
        w = canvas_size(self, px(40))[0]
        pad = px(6)
        return pad, max(pad + px(8), w - pad)

    def span_px(self) -> tuple:
        """(x0, x1) of the solid bar in device px -- what the tests measure
        instead of looking at the picture."""
        x0, x1 = self._rail()
        if not self._span:
            return x0, x0
        return (x0 + (x1 - x0) * self._span[0],
                x0 + (x1 - x0) * self._span[1])

    def _draw(self) -> None:
        self.delete("all")
        if not restyled():
            return self._draw_v3()
        h = canvas_size(self, px(40), px(8))[1]
        y = h / 2
        x0, x1 = self._rail()
        stroke = max(1, px(1))
        self.create_line(x0, y, x1, y, fill=theme.LINE, width=stroke)
        if not self._span:
            return
        bx0, bx1 = self.span_px()
        bar = max(2, px(type(self).BAR))
        self.create_rectangle(bx0, y - bar / 2, bx1, y + bar / 2,
                              fill=theme.RAMP60, outline="")
        if self._frac is None:
            return
        x = bx0 + (bx1 - bx0) * max(0.0, min(1.0, float(self._frac)))
        if marker_shape() == "needle":
            self.create_line(x, y - px(6), x, y + px(6), fill=theme.FOCAL,
                             width=max(1, px(2)))
            return
        r = px(4)
        self.create_oval(x - r, y - r, x + r, y + r, fill=theme.FOCAL,
                         outline="")

    def _draw_v3(self) -> None:
        """``[0.8]---#---[1.8] m``: the strip EXACTLY as jarvis-v3 drew it
        -- a full-width track, an end stop at each end, and a round dot
        where the live range falls. Classic is frozen (``restyled``).

        The span is not drawn here at all: ``set_span`` still records it
        (``span_px`` is what the tests measure), it simply has no ink in
        this look, which is the shape v3 shipped.
        """
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
    """One room, in five lines at most and in TWO type sizes.

    THE SHAPE (2026-09-05)::

        OFFICE  AT THE DESK · camera            1.4 m   <1 ms
          radar     PRESENT
          camera    FACE hunterp 0.71
          the camera sees a face; the radar is not asked
          no answer from the sensor (2 in a row)          <- amber, faults only

    The HEADER carries the answer (the fused zone) and the two numbers; the
    detail lines carry the two LEGS, named symmetrically, on one label
    column; the reason is one muted line and the fault is the only amber
    thing on the page. Before today the header carried PRESENT (a leg) in
    the display face beside a monospace range, the verdict sat in a detail
    row in a third size, and the reason line was prefixed with the source
    and ran long enough to wrap into a second grey line.
    """

    CAP_W = 9                 # the label column, in caption-face characters

    def __init__(self, parent, bg: str):
        super().__init__(parent, bg=bg)
        self._bg = bg
        # WHICH TREE THIS BLOCK IS. Read once, HERE -- a block is built
        # after create() has selected the look and is never re-looked, and
        # apply() has to talk to the widgets that actually exist. It is a
        # call-time read of theme.LOOK, not a def-time capture.
        self._restyled = restyled()
        if not self._restyled:
            self._build_v3(bg)
            return
        head = tk.Frame(self, bg=bg)
        head.pack(fill="x")
        flat = dict(bg=bg, bd=0, padx=0, pady=0)     # Tk's default 1px
        # border + 1px pady on EVERY label is 4 px a line, and this block
        # is five lines twice over.
        self.name = tk.Label(head, font=ui_display(theme.SIZE_LABEL, "semibold"),
                             fg=theme.INK, anchor="w",
                             width=type(self).CAP_W, **flat)
        self.name.pack(side="left")
        self.rtt = tk.Label(head, font=ui_mono(theme.SIZE_CAPTION),
                            fg=theme.FAINT, anchor="e", width=7, **flat)
        self.rtt.pack(side="right")
        self.distance = tk.Label(head, font=ui_mono(theme.SIZE_LABEL),
                                 fg=theme.FOCAL, anchor="e", width=7, **flat)
        self.distance.pack(side="right", padx=(0, px(8)))
        self.verdict = tk.Label(head, font=ui_display(theme.SIZE_LABEL),
                                fg=theme.MUTED, anchor="w", **flat)
        self.verdict.pack(side="left")
        # The leg that answered, beside the word rather than glued to the
        # front of the reason sentence ("camera · the camera can see...").
        self.source = tk.Label(head, font=ui_display(theme.SIZE_CAPTION),
                               fg=theme.FAINT, anchor="w", **flat)
        self.source.pack(side="left", padx=(px(8), 0))

        _, self.presence = self._line("radar")
        _, self.camera = self._line("camera")
        _, self.why = self._line("")
        # The fault row is the only one that comes and goes: it is packed by
        # apply() when a leg had no opinion and forgotten when it does.
        self._fault_row, self.fault = self._line("", tone=theme.WARN)
        self._fault_row.pack_forget()
        self.bind("<Configure>", self._wrap, add=True)

    # ------------------------------------------------- the frozen classic
    def _build_v3(self, bg: str) -> None:
        """The block EXACTLY as jarvis-v3 4b7d373 built it: the room name
        and the PRESENCE word in the header beside a monospace range, the
        camera and the verdict in two captioned sub-rows in a third size,
        and the reason and fault packed straight onto the block on a
        hand-set px(96) indent. Classic is frozen (``restyled``)."""
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

        self.camera = self._sub_v3("camera")
        self.verdict = self._sub_v3("verdict")
        self.fault = tk.Label(self, font=ui_font(theme.SIZE_CAPTION),
                              fg=theme.WARN, bg=bg, anchor="w",
                              justify="left")
        self.why = tk.Label(self, font=ui_font(theme.SIZE_CAPTION),
                            fg=theme.FAINT, bg=bg, anchor="w", justify="left")
        self.why.pack(fill="x", padx=(px(96), 0))
        # There is no `source` label in this tree: v3 glued the source onto
        # the front of the reason sentence. apply_v3 does the same.
        self.source = None
        self._fault_row = None
        self.bind("<Configure>", self._wrap, add=True)

    def _sub_v3(self, caption: str) -> tk.Label:
        row = tk.Frame(self, bg=self._bg)
        row.pack(fill="x")
        tk.Label(row, text=caption, font=ui_font(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=self._bg, anchor="w",
                 width=14).pack(side="left", padx=(px(16), 0))
        value = tk.Label(row, font=ui_font(theme.SIZE_LABEL), fg=theme.MUTED,
                         bg=self._bg, anchor="w")
        value.pack(side="left")
        return value

    def _apply_v3(self, row: RoomRow) -> None:
        self.name.configure(text=row.label.upper())
        self.presence.configure(text=row.presence_word,
                                fg=tone_color(row.presence_tone))
        self.distance.configure(text=row.distance_text)
        self.rtt.configure(text=row.rtt_text)
        self.camera.configure(text=row.camera_text,
                              fg=tone_color(row.camera_tone))
        self.verdict.configure(
            text=row.verdict.word,
            fg=(theme.WARN if row.verdict.zone in (NO_OPINION, NO_LADDER)
                else theme.FOCAL))
        self.why.configure(text=("%s · %s" % (row.verdict.source,
                                              row.verdict.why)
                                 if row.verdict.source else row.verdict.why))
        if row.fault:
            self.fault.configure(text=row.fault)
            self.fault.pack(fill="x", padx=(px(96), 0))
        else:
            self.fault.pack_forget()

    def _line(self, caption: str, tone: Optional[str] = None) -> tuple:
        """One detail line: a fixed caption column and a value beside it.

        Every line goes through here -- INCLUDING the two that have no
        caption -- so the reason and the fault land on exactly the same x
        as the two leg values instead of on a hand-guessed indent.
        """
        row = tk.Frame(self, bg=self._bg)
        row.pack(fill="x")
        tk.Label(row, text=caption, font=ui_display(theme.SIZE_CAPTION),
                 fg=theme.FAINT, bg=self._bg, anchor="w", bd=0, padx=0,
                 pady=0, width=type(self).CAP_W).pack(side="left",
                                                      padx=(px(10), 0))
        value = tk.Label(row, font=ui_display(theme.SIZE_CAPTION),
                         fg=(tone or theme.MUTED), bg=self._bg, anchor="w",
                         justify="left", bd=0, padx=0, pady=0)
        value.pack(side="left")
        return row, value

    def _wrap(self, event) -> None:
        """Tk labels do not wrap without an explicit pixel width, and these
        two carry the only sentences on the page. They are one short line
        each by construction now (fuse / fault_line), but a wider band name
        or a longer blocked reason must still wrap at the column rather
        than be cut mid-word ("...the radar's range is n", measured on the
        photo rig 2026-09-03)."""
        width = max(px(120), int(event.width)
                    - (px(110) if self._restyled else px(104)))
        for label in (self.fault, self.why):
            label.configure(wraplength=width)

    def apply(self, row: RoomRow) -> None:
        if not self._restyled:
            return self._apply_v3(row)
        self.name.configure(text=row.label.upper())
        self.presence.configure(text=row.presence_word,
                                fg=tone_color(row.presence_tone))
        # MONOSPACE IS FOR NUMBERS. A dash is a word and is drawn in the
        # header's own face; the moment a real range lands it becomes a
        # readout again (readout_is_numeric).
        for label, text, size in ((self.distance, row.distance_text,
                                   theme.SIZE_LABEL),
                                  (self.rtt, row.rtt_text,
                                   theme.SIZE_CAPTION)):
            label.configure(text=text,
                            font=(ui_mono(size) if readout_is_numeric(text)
                                  else ui_display(size)))
        self.camera.configure(text=row.camera_text,
                              fg=tone_color(row.camera_tone))
        self.verdict.configure(text=row.verdict.word,
                               fg=tone_color(verdict_tone(row.verdict.zone)))
        self.source.configure(text=source_text(row.verdict))
        lines = explanation_lines(row)
        self.why.configure(text=lines[0][0], fg=tone_color(lines[0][1]))
        if len(lines) > 1:
            self.fault.configure(text=lines[1][0], fg=tone_color(lines[1][1]))
            self._fault_row.pack(fill="x")
        else:
            self._fault_row.pack_forget()


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

    WHAT IT COSTS AND WHERE IT GOES, RE-MEASURED 2026-09-05 on :94 at
    920x1440, S=2, holo, with the camera pane packed and BOTH rooms in
    their worst state (a fault line AND a reason line on each). The stage
    is 852 px; the pinned foot takes 133 of it; the scrolling body asks
    for 690 into a 719-px viewport::

        before 09-05   body 1043 px into 852   -> 191 px CUT OFF
        after  09-05   body  690 px into 719   -> fits, 29 px spare

    The 353 px came off in four places, in order of size: the block was
    retyped to two sizes (a 46-px header line and 32-px detail lines, from
    50 + 5x36); the overrule toggle moved into the pinned foot; the band
    rows lost the display-size label and Tk's default label border; and the
    rule between the two halves lost half its air.

    Past that -- a third sensor, or a ladder with more bands -- the body
    SCROLLS (``overflow_px``, ``_sync_thumb``), and SAVE, the age, the
    caption and the overrule toggle stay pinned below it. Roughly 37 px a
    band and 174 px a room, so the second screen starts somewhere past
    three rooms.

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
        # THE SCROLLING VIEW IS HOLO'S. Classic is frozen at jarvis-v3
        # (``restyled``), which packed one plain column, so every method
        # that drives the scroll checks for the canvas rather than
        # assuming it. None, not missing: a getattr() default would hide a
        # real build failure.
        self._canvas = None
        self._body = None
        self._thumb = None
        self._foot = None
        self._save_btn = None
        self._rooms_bands: list = []

        self.specs = self._room_specs()
        # ONE zone model: the ladders come from zones.rooms through
        # jarvis/zones.py's own validator, so what he edits here is what
        # the zone log places a reading with.
        self.ladders = read_ladders(self._get_option)
        self.config_notes = tuple(self.ladders.notes)
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
        """ONE COLUMN: the body, then the foot, then whatever is left over.

        Hunter, 2026-09-05: the last band row ("camera overrules radar")
        was cut off at the bottom, hidden behind the camera pane, and SAVE
        with it. Measured on :94 at S=2 in his 920x1440 window with the
        pane packed, both rooms dark and every line drawn: the page asked
        for 1043 px into an 852-px stage.

        Two things fix it, and both are needed. The block and band rows
        were retyped and retightened so his config FITS (measured below);
        and what is left is a scrolling view, so a ladder with more bands
        or a third sensor scrolls instead of vanishing. SAVE, the age and
        the caption sit OUTSIDE the scroll -- a save button that scrolls
        away is a save button he cannot find.

        AND THE FOOT FOLLOWS THE CONTENT (round 1). Pinning it to the
        bottom of the frame traded the clipped row for a VOID: measured at
        the geometry he actually runs (main_window's own default, 520x880
        design units, i.e. 1040x1760 at S=2), 408 px of nothing between the
        last band row and SAVE -- 23% of the window -- because his config
        FITS there with room to spare. So the scrolling view is sized to
        its content and capped at the room the foot leaves it
        (``_avail_px``): when the page fits, the foot sits under the last
        band row and the slack falls off the bottom, where a finished page
        ends; when it does not, the view stops at the foot and the body
        scrolls under it, exactly as before.

        AND THE SLACK STAYS THERE (round 2, a decision). RE-MEASURED off
        the rig at his window, counting blank rows in the ink: this page
        has ONE run of 428 blank rows, y976..1403, i.e. 24% of the window,
        and it is below the last line. The v3 tip had 198 blank rows at
        y622..819 -- the same slack, in the MIDDLE, which is the shape he
        photographed. Filling 428 px would mean growing rows the content
        does not need (and rows that jump whenever a fault line appears),
        or shrinking the page to its content, which uncovers the
        transcript this page is placed over (``cover``). So the rule that
        is pinned is the SHAPE: one run of slack, and it is the last thing
        on the page (tests/test_ui_layout_rules.py,
        test_the_pages_only_slack_is_at_the_bottom_where_a_page_ends).
        """
        if not restyled():
            return self._build_v3()
        bg = theme.TV_BG
        # ---- the foot. BUILT here and PACKED after the view, so it lands
        # directly under the content instead of at the frame's bottom edge.
        self._foot = foot = tk.Frame(self, bg=bg)
        # SAVE and the age share ONE row; the caption gets the next one to
        # itself. Packing all three into `foot` put the caption between the
        # button and the age and cut it mid-word -- photographed 09-03,
        # which is the same trap that put this caption below the button in
        # the first place.
        act = tk.Frame(foot, bg=bg)
        act.pack(fill="x")
        self._save_btn = RoundButton(act, text="SAVE", kind="default",
                                     size=theme.SIZE_CAPTION, bg=bg,
                                     pad_y=5, command=self.save)
        self._save_btn.pack(side="left")
        # THE OVERRULE TOGGLE LIVES HERE, not at the end of the band list.
        # It was the last row of a scrolling column and it is the row he
        # photographed cut in half behind the camera pane; it is also not a
        # band -- it is the one SETTING on this page, so it belongs beside
        # the button that writes it, where it cannot be scrolled away.
        self._overrule = Toggle(act, value=self.overrules, bg=bg)
        self._overrule.pack(side="right")
        self._overrule.command = self._on_overrule
        tk.Label(act, text="camera overrules radar",
                 font=ui_display(theme.SIZE_CAPTION), fg=theme.MUTED, bg=bg,
                 anchor="e", bd=0, padx=0, pady=0).pack(side="right",
                                                        padx=(0, px(10)))
        line = tk.Frame(foot, bg=bg)
        line.pack(fill="x", pady=(px(4), 0))
        # The staleness readout, sharing the caption row so it costs no
        # height. It is driven by the page's own 1 s tick, not by the poll,
        # so a poll that has stopped makes this number CLIMB rather than
        # freeze -- which is the whole point of putting an age on a numbers
        # page.
        self._age = tk.Label(line, font=ui_display(theme.SIZE_CAPTION),
                             fg=theme.FAINT, bg=bg, anchor="e", bd=0,
                             padx=0, pady=0)
        self._age.pack(side="right", padx=(px(12), 0))
        # BELOW the button, not beside it: beside it the caption had ~750 px
        # of a 920-px window and was cut mid-word on the photo rig, and the
        # refusal message a bad band edit puts here is longer still.
        self._note = tk.Label(line, text=restart_note(),
                              font=ui_display(theme.SIZE_CAPTION),
                              fg=theme.FAINT, bg=bg, anchor="w",
                              justify="left", bd=0, padx=0, pady=0)
        self._note.pack(side="left", fill="x", expand=True)
        self._notes = tk.Label(foot, font=ui_display(theme.SIZE_CAPTION),
                               fg=theme.WARN, bg=bg, anchor="w",
                               justify="left", bd=0, padx=0, pady=0)

        # ---- the scrolling view, packed FIRST and sized to its content
        view = tk.Frame(self, bg=bg)
        view.pack(side="top", fill="x")
        foot.pack(side="top", fill="x", padx=theme.PAD,
                  pady=(theme.PAD_S, px(12)))
        self._canvas = tk.Canvas(view, bg=bg, highlightthickness=0, bd=0,
                                 height=px(40))
        self._canvas.pack(side="left", fill="x", expand=True)
        # A 2px cyan strip on the canvas' right edge, shown ONLY when there
        # is something below the fold. A page that scrolls with no mark
        # saying so is a page whose bottom rows he has no reason to look for.
        self._thumb = tk.Frame(self._canvas, bg=theme.CYAN_DIM,
                               width=max(2, px(2)))
        self._body = tk.Frame(self._canvas, bg=bg)
        self._body_win = self._canvas.create_window(
            0, 0, anchor="nw", window=self._body)
        self._canvas.bind("<Configure>", lambda e: self._sync_view(), add=True)
        self._body.bind("<Configure>", lambda e: self._sync_view(), add=True)
        self._canvas.bind("<Enter>", self._grab_wheel, add=True)
        self._canvas.bind("<Leave>", self._drop_wheel, add=True)

        body = tk.Frame(self._body, bg=bg)
        body.pack(fill="both", expand=True, padx=theme.PAD,
                  pady=(theme.PAD_S, 0))
        if not self.specs:
            tk.Label(body, text=empty_state_line(self._get_option),
                     font=ui_display(theme.SIZE_LABEL), fg=theme.FAINT, bg=bg,
                     anchor="w", justify="left",
                     wraplength=px(420)).pack(fill="x")
        for spec in self.specs:
            block = _RoomBlock(body, bg)
            block.pack(fill="x", pady=(0, px(6)))
            self._blocks[spec.name] = block

        tk.Frame(self._body, bg=theme.LINE, height=max(1, px(1))).pack(
            fill="x", padx=theme.PAD, pady=px(6))
        tune = tk.Frame(self._body, bg=bg)
        tune.pack(fill="x", padx=theme.PAD)
        # ONE ROW PER BAND, named by the band, grouped by room. The two
        # fixed rows that used to live here were the page's own two-band
        # model; his office ladder has its own names and his kitchen has
        # different ones again, and a page that could only edit "desk" and
        # "room" could not edit either of them.
        self._bands = []                  # in the order they are packed
        self._rooms_bands = []            # the same rows, grouped by room:
        # a band's bar is drawn on ITS ROOM's scale, so the group is what
        # the redraw works on (_resync_spans).
        first = True
        for spec in self.specs:
            zmap = self.ladders.for_room(spec.name)
            if zmap is None:
                tk.Label(tune, text="%s: no ladder in %s — add one and "
                                    "restart" % (_spoken(spec), OPTION_ROOMS),
                         font=ui_display(theme.SIZE_CAPTION), fg=theme.WARN,
                         bg=bg, anchor="w").pack(fill="x")
                continue
            if len(self.specs) > 1:
                # A room label needs air ABOVE it and none below: with the
                # same pady on both sides it read as a caption for the row
                # above as easily as for the rows under it (09-05).
                tk.Label(tune, text=theme.caption(_spoken(spec), surface=False),
                         font=ui_display(theme.SIZE_CAPTION, "semibold"),
                         fg=theme.FAINT, bg=bg, anchor="w", bd=0, padx=0,
                         pady=0).pack(
                    fill="x", pady=(0 if first else px(10), px(3)))
                first = False
            room = zn._room_key(spec.name)
            rows = [self._band_row(tune, room, band) for band in zmap.bands]
            self._bands.extend(rows)
            self._rooms_bands.append(rows)
        self._resync_spans()

        self.bind("<Configure>", lambda e: [
            w.configure(wraplength=max(px(160), int(e.width) - 2 * theme.PAD))
            for w in (self._notes, self._note)], add=True)
        self._show_notes()

    # ------------------------------------------------- the frozen classic
    def _build_v3(self) -> None:
        """The page EXACTLY as jarvis-v3 4b7d373 packed it: one plain
        column with no scrolling view, the overrule toggle as the last row
        of the tune list, and the foot pinned under it.

        CLASSIC IS FROZEN, AND THAT HAS A PRICE (``restyled``). This is the
        layout he photographed with "camera overrules radar" cut off under
        the camera pane at 920x1440 in the worst case, and the fix for that
        is holo's. He runs holo (theme.DEFAULT_LOOK) and he did not ask for
        classic to change; the 09-05 relayout moved it by 175,076 px at
        920x1440 and 202,322 px at his own window, MEASURED on the rig.
        """
        bg = theme.TV_BG
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
            block.pack(fill="x", pady=(0, px(4)))
            self._blocks[spec.name] = block

        tk.Frame(self, bg=theme.LINE, height=max(1, px(1))).pack(
            fill="x", padx=theme.PAD, pady=theme.PAD_S)
        tune = tk.Frame(self, bg=bg)
        tune.pack(fill="x", padx=theme.PAD)
        self._bands = []                  # in the order they are packed
        for spec in self.specs:
            zmap = self.ladders.for_room(spec.name)
            if zmap is None:
                tk.Label(tune, text="%s: no ladder in %s — add one and "
                                    "restart" % (_spoken(spec), OPTION_ROOMS),
                         font=ui_font(theme.SIZE_CAPTION), fg=theme.WARN,
                         bg=bg, anchor="w").pack(fill="x")
                continue
            if len(self.specs) > 1:
                tk.Label(tune, text=_spoken(spec).upper(),
                         font=ui_font(theme.SIZE_CAPTION), fg=theme.FAINT,
                         bg=bg, anchor="w").pack(fill="x")
            for band in zmap.bands:
                self._bands.append(self._band_row(tune, zn._room_key(spec.name),
                                                  band))

        row = tk.Frame(tune, bg=bg)
        row.pack(fill="x", pady=px(4))
        tk.Label(row, text="camera overrules radar",
                 font=ui_font(theme.SIZE_LABEL), fg=theme.MUTED, bg=bg,
                 anchor="w").pack(side="left")
        self._overrule = Toggle(row, value=self.overrules, bg=bg)
        self._overrule.pack(side="right")
        self._overrule.command = self._on_overrule

        self._foot = foot = tk.Frame(self, bg=bg)
        foot.pack(fill="x", padx=theme.PAD, pady=(theme.PAD_S, theme.PAD))
        row = tk.Frame(foot, bg=bg)
        row.pack(fill="x")
        self._save_btn = RoundButton(row, text="SAVE", kind="default",
                                     size=theme.SIZE_CAPTION, bg=bg,
                                     command=self.save)
        self._save_btn.pack(side="left")
        self._age = tk.Label(row, font=ui_font(theme.SIZE_CAPTION),
                             fg=theme.FAINT, bg=bg, anchor="e")
        self._age.pack(side="right", padx=(px(12), 0))
        self._note = tk.Label(foot, text=restart_note(),
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

    # ------------------------------------------------------- the scroll
    def _sync_view(self) -> None:
        """Keep the scroll region, the body's width and the thumb honest.

        Called on every <Configure> of either the viewport or the body, so
        a fault line appearing or a room block growing re-measures rather
        than leaving a stale region behind.
        """
        if self._canvas is None:          # frozen classic (``restyled``)
            return
        try:
            width = self._canvas.winfo_width()
            self._canvas.itemconfigure(self._body_win, width=width)
            # HEIGHT FOLLOWS THE CONTENT, capped at the room the foot
            # leaves. Set only when it CHANGES: a canvas that reconfigures
            # itself on its own <Configure> would loop.
            want = min(self._body.winfo_reqheight(), self._avail_px())
            if want > 0 and want != self._canvas.winfo_reqheight():
                self._canvas.configure(height=want)
            self._canvas.configure(
                scrollregion=self._canvas.bbox("all") or (0, 0, 0, 0))
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("sensors page: scroll region unreadable", exc_info=True)
            return
        self._sync_thumb()

    def _avail_px(self) -> int:
        """How much height the scrolling view may take: the page, less
        what the foot needs. 0 before the page has been laid out, and 0
        for the frozen classic look, which has no scrolling view."""
        if self._foot is None or self._canvas is None:
            return 0
        try:
            height = self._page_h()
            if height <= 1:
                return 0
            foot = (self._foot.winfo_reqheight() + theme.PAD_S + px(12))
            return max(px(40), height - foot)
        except Exception:                 # noqa: BLE001 - torn down
            return 0

    def _page_h(self) -> int:
        """The frame's own height, which is what it was PLACED with -- a
        frame whose children are packed reports its request until the
        geometry manager has run once."""
        try:
            return max(int(self.winfo_height()), 1)
        except Exception:                 # noqa: BLE001 - torn down
            return 1

    def _sync_thumb(self) -> None:
        if self._canvas is None:          # frozen classic (``restyled``)
            return
        view_h = self._canvas.winfo_height()
        over = self.overflow_px()
        if over <= 0 or view_h <= 1:
            try:
                self._thumb.place_forget()
                self._canvas.yview_moveto(0.0)
            except Exception:             # noqa: BLE001 - torn down
                log.debug("sensors page: thumb hide failed", exc_info=True)
            return
        content = view_h + over
        top = self._canvas.canvasy(0)
        frac = max(0.08, view_h / float(content))
        try:
            self._thumb.place(relx=1.0, x=-px(3), anchor="nw",
                              y=int(top / content * view_h),
                              height=max(px(12), int(frac * view_h)))
            self._thumb.lift()
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("sensors page: thumb place failed", exc_info=True)

    def overflow_px(self) -> int:
        """How much taller the body is than the viewport. 0 = it all fits.

        The number this page was built around: at his window, with his
        config and both rooms in their worst state, it must be 0
        (tests/test_ui_layout_rules.py measures it on a private display).
        """
        if self._canvas is None:          # frozen classic: it never scrolls
            return 0
        try:
            room = self._avail_px() or self._canvas.winfo_height()
            return max(0, self._body.winfo_reqheight() - room)
        except Exception:                 # noqa: BLE001 - torn down
            return 0

    def _scroll(self, units: int) -> None:
        if self.overflow_px() <= 0:
            return
        try:
            self._canvas.yview_scroll(units, "units")
        except Exception:                 # noqa: BLE001 - torn down
            log.debug("sensors page: scroll failed", exc_info=True)
            return
        self._sync_thumb()

    def _grab_wheel(self, _e=None) -> None:
        if self._canvas is None:
            return
        self._canvas.bind_all("<Button-4>", lambda e: self._scroll(-2))
        self._canvas.bind_all("<Button-5>", lambda e: self._scroll(2))

    def _drop_wheel(self, _e=None) -> None:
        if self._canvas is None:
            return
        for seq in ("<Button-4>", "<Button-5>"):
            try:
                self._canvas.unbind_all(seq)
            except Exception:             # noqa: BLE001 - torn down
                log.debug("sensors page: wheel unbind failed", exc_info=True)

    def _resync_spans(self) -> None:
        """Redraw every room's bars from what is CURRENTLY in its boxes.

        Called at build and on every keystroke in a band box, so the
        ladder under his hands follows what he typed rather than what the
        config held when the page opened. Junk in one box drops that one
        bar (spans_for_room) instead of the room's whole ladder.
        """
        for rows in self._rooms_bands:
            pairs = [(r["lo"].get(), r["hi"].get()) for r in rows]
            spans = spans_for_room(pairs)
            for row, span, (lo, hi) in zip(rows, spans, pairs):
                row["span"] = span
                row["bar"].set_span(span)
                # The live mark is drawn INSIDE the span, so the bounds it
                # is measured against have to be the same ones the span
                # was drawn from -- otherwise the needle drifts off the
                # bar while he is typing. Unparseable boxes keep the
                # config's numbers.
                lo, hi = _finite(lo), _finite(hi)
                if lo is not None and hi is not None and hi > lo:
                    row["near_m"], row["far_m"] = lo, hi

    def _band_row(self, parent, room: str, band) -> dict:
        """``at the desk  [2.25] ——[####]—— [3.75] m`` on ONE baseline.

        The unit is a column of its own (a fixed width, packed to the far
        right) rather than a label that floats off the end of whatever the
        number happened to be -- the "m"s were ragged on the 09-05 shot.
        """
        bg = theme.TV_BG
        if not restyled():
            return self._band_row_v3(parent, room, band)
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=px(2))
        tk.Label(row, text=band.name, font=ui_display(theme.SIZE_CAPTION),
                 fg=theme.MUTED, bg=bg, anchor="w", width=18, bd=0, padx=0,
                 pady=0).pack(side="left")
        unit = tk.Label(row, text="m", font=ui_display(theme.SIZE_CAPTION),
                        fg=theme.FAINT, bg=bg, anchor="w", width=2, bd=0,
                        padx=0, pady=0)
        unit.pack(side="right", padx=(px(6), 0))
        hi_e = self._entry(row, band.far_m)
        hi_e.pack(side="right")
        lo_e = self._entry(row, band.near_m)
        lo_e.pack(side="left", padx=(0, px(8)))
        bar = _BandBar(row, bg=bg)
        bar.pack(side="left", fill="x", expand=True, padx=(0, px(8)))
        for entry in (lo_e, hi_e):
            entry.bind("<KeyRelease>", lambda _e: self._resync_spans(),
                       add=True)
            entry.bind("<FocusOut>", lambda _e: self._resync_spans(),
                       add=True)
        return {"room": room, "name": band.name, "lo": lo_e, "hi": hi_e,
                "bar": bar, "unit": unit, "span": None,
                "near_m": band.near_m, "far_m": band.far_m}

    def _band_row_v3(self, parent, room: str, band) -> dict:
        """The band row EXACTLY as jarvis-v3 packed it: a display-size band
        name in a 13-character column, the unit floating off the end of
        whatever the number happened to be, and no live redraw while he
        types. Classic is frozen (``restyled``)."""
        bg = theme.TV_BG
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", pady=px(4))
        tk.Label(row, text=band.name, font=ui_font(theme.SIZE_LABEL),
                 fg=theme.MUTED, bg=bg, anchor="w",
                 width=13).pack(side="left")
        unit = tk.Label(row, text="m", font=ui_font(theme.SIZE_CAPTION),
                        fg=theme.FAINT, bg=bg)
        unit.pack(side="right", padx=(px(4), 0))
        hi_e = self._entry(row, band.far_m)
        hi_e.pack(side="right")
        lo_e = self._entry(row, band.near_m)
        lo_e.pack(side="left", padx=(0, px(6)))
        bar = _BandBar(row, bg=bg)
        bar.pack(side="left", fill="x", expand=True, padx=(0, px(6)))
        return {"room": room, "name": band.name, "lo": lo_e, "hi": hi_e,
                "bar": bar, "unit": unit, "span": None,
                "near_m": band.near_m, "far_m": band.far_m}

    def _entry(self, parent, value: float) -> tk.Entry:
        size = theme.SIZE_CAPTION if restyled() else theme.SIZE_LABEL
        e = tk.Entry(parent, width=5, justify="center",
                     font=ui_mono(size), fg=theme.INK,
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
        # bind_all is GLOBAL: a wheel binding left behind would scroll a
        # hidden page from anywhere in the console.
        self._drop_wheel()
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
        rows = page_rows(self._last, self._camera(), self.ladders,
                         self.overrules, camera_room=self.camera_room)
        self._rows = rows
        for row in rows:
            self.apply_row(row)
        # The marker goes on the bands of the room the reading came FROM.
        # It used to take one distance and put it on both bars whatever
        # room it was measured in, which with two sensors draws the kitchen
        # range on the office ladder.
        here = {zn._room_key(row.name): row.distance_m for row in rows}
        for band in self._bands:
            band["bar"].set(band_fraction(here.get(band["room"]),
                                          band["near_m"], band["far_m"]))

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
        notes = tuple(self.config_notes)
        notes += name_mismatch_note(self.specs, self.ladders)
        for spec in self.specs:
            notes += band_notes(self.ladders.for_room(spec.name))
        if notes:
            self._notes.configure(text="\n".join(notes))
            if restyled():
                self._notes.pack(fill="x", pady=(px(4), 0))
            else:                         # frozen classic (``restyled``)
                self._notes.pack(fill="x", padx=theme.PAD,
                                 pady=(0, theme.PAD_S))
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
        self.overrules = bool(edits[OPTION_CAMERA_OVERRULES])
        fn = getattr(self.services, "set_option", None) if self.services else None
        if fn is None:
            self._note.configure(text=NOT_WIRED_NOTE, fg=theme.WARN)
            return
        failed = write_options(fn, edits)
        if not failed:
            # What he typed is now what the file holds, so the "the config
            # had a value I could not use" notes are spent -- and the
            # superseded keys have been carried across, so they go.
            self.ladders = Ladders(maps=self.ladders.maps,
                                   legacy=self.ladders.legacy,
                                   raw=edits[OPTION_ROOMS],
                                   refused=self.ladders.refused)
            self.config_notes = retire_legacy(
                self._get_option,
                getattr(self.services, "unset_option", None)
                if self.services else None)
            self._reread_ladders()
        self._show_notes()
        text, tone = save_note(failed)
        self._note.configure(text=text, fg=tone_color(tone))

    def _reread_ladders(self) -> None:
        """Re-read the ladders from the config after a successful write, so
        the verdicts and the band bars follow what was just saved.

        Not a reload of the whole config -- ``reload_if_changed`` still has
        no callers and ``RESTART_NOTE`` is still true for the rest of the
        app. This is the in-memory ``AssistantConfig`` the write just went
        through answering the same question again.
        """
        try:
            self.ladders = read_ladders(self._get_option)
        except Exception:                 # noqa: BLE001 - config boundary
            log.exception("sensors page: the ladders could not be re-read")

    def band_values(self) -> tuple:
        """What is currently typed, validated. Split out so the refusal
        message is testable without a display."""
        return band_edits([BandEdit(b["room"], b["name"], b["lo"].get(),
                                    b["hi"].get()) for b in self._bands],
                          self.ladders, self._overrule.get())


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
