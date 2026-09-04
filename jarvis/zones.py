"""Zones: WHERE in the room, as far as this hardware can honestly say --
and a written record of it before anything is allowed to act on it.

He asked for the record first, in those words: "Just log it first."
Nothing in this module speaks, greets, dims a panel or publishes an event.
It turns two cheap facts -- a presence bit with a distance, and whether the
camera recognised a face -- into a named zone, holds that name still while
the numbers wobble, and appends one JSONL line per COMMITTED change.

WHAT THE RADAR CAN AND CANNOT TELL YOU. The office sensor is an
HLK-LD2410C on an ESP32 (``jarvis/roomsensor.py``). It reports a presence
bit, a moving-target bit, a still-target bit and a distance in centimetres.
It reports NO ANGLE and NO X/Y, so it cannot tell "at the desk" from "at
the bookshelf" when both sit at the same range. That is structural, not a
tuning problem, and it is why a zone here is a DISTANCE BAND and why the
camera has to be allowed to overrule it.

THE PRECEDENCE, which is the whole model, in his words: "camera
recognition overrules sensor detection since he can literally see me at my
desk". So::

    camera recognises a known face   -> the room's camera zone   ("at the desk")
      ... but only in a room that HAS a camera; a blank ``camera_zone``
      means there is no lens here and the radar decides (``has_camera``)
    presence True, distance in a band-> that band's name
    presence True, distance in none  -> "in the room, unplaced"
    presence False                   -> "not in the room"
    read() returned None             -> "no opinion"   NEVER "not in the room"

Two asymmetries in there are deliberate and both are copied from
``jarvis/presence.py``, where the same argument was already settled.

* **The camera may only ever ADD.** A lens that recognised nobody is not
  evidence of an empty chair -- it may be closed by the 21:00 curfew, by
  offline mode, or he may simply be turned away -- so ``known=False`` falls
  through to the radar rather than vetoing it. Only recognition wins.
* **"No opinion" is not absence.** A timeout, a 404, an open breaker or
  offline mode all give None, and collapsing that into "not in the room"
  is how Jarvis goes quiet on a man sitting three feet away.

WHY THE NEAREST BAND IS NOT THE DESK -- the thing that would have been got
wrong by guessing. The office profile
(``~/.config/jarvis/room-sensors/office.json``) records the mount as "on
the desk at the BACK edge, aimed OUT across the room at the door -- not at
the chair". Sitting in his chair he is BEHIND the module and inside its
0.75 m blind zone, where it detects nothing at all; and the first 1.5 m in
front of it has no STILL-target sensitivity (gates 0 and 1), so a
motionless body there disappears the way it would from a PIR. A naive
"desk = nearest band" would therefore have named the one place the radar
is structurally blind. The nearest band is the floor IN FRONT of the desk
-- where he stands up, or where someone else stands to talk to him -- and
the chair itself belongs to the camera. That is not a preference; it is
the mount.

THE OFFICE LADDER, gate-aligned. One distance gate is 0.75 m
(``scripts/room_sensor.py`` GATE_M), so every edge below is a multiple of
it: an edge inside a gate is finer than the sensor and would be a fiction.
The device is tuned to max move gate 6 / max still gate 6, i.e. 4.5 m, with
a 10 s absence delay (measured 2026-09-03).

    0.00 - 0.75 m   GAP: the module's blind zone. Nothing is detected.
    0.75 - 1.50 m   "just off the desk"   -- MOTION ONLY; a still body here
                                             is invisible (gates 0-1).
    1.50 - 3.00 m   "the middle of the room"
    3.00 - 4.50 m   "by the door"         -- the measured walk topped out at
                                             3.6 m, which is inside this band.
    4.50 m +        GAP: past the configured far gate; anything reported out
                    there is reaching through a stud wall.

Both gaps are declared rather than accidental (``ZoneMap.gaps()``), and a
reading in one is "in the room, unplaced" -- a real answer, not an error.

A BAD LADDER IS REFUSED, NOT SUBSTITUTED. The ladder above is the built-in
one, and it stands in for exactly one case: A CONFIG THAT SAID NOTHING AT
ALL about rooms, meaning no ``zones.rooms`` key, which is a config written
before this section existed. A config that TRIED to say something and got
it wrong is not silence and gets nothing -- not a room whose bands do not
parse, not one switched off in its own entry, not one named twice, not one
simply missing from the list, and NOT a ``zones.rooms`` that is there but
is not a list (an unwrapped room object, a map keyed by name, a string, a
number, an explicit null). ``zone_map_for`` returns None for all of those
and names the offending key (``rejected_rooms``), and None means RECORD
NOTHING. The alternative is the one failure this whole log exists to
avoid: he edits his bands, mistypes them, and the record goes on being
written against the bands he thought he had replaced -- looking, line after
plausible line, like it worked.

The second pass of this module claimed that rule and did not implement it:
it told "no key" from "a LIST", so the commonest slip in that file --
dropping the ``[ ]`` around his one room -- landed on the built-in ladder
with no log line at any level. ``AssistantConfig._deep_merge`` replaces on
a type mismatch, so the unwrapped dict really does beat the DEFAULTS list,
and the whole ladder above is served out of DEFAULTS. Absence is now the
only silence.

HYSTERESIS, because a band edge WILL chatter. A body straddling the 1.50 m
edge crosses it many times a minute, and a memoryless model would write a
line per sample. A change therefore commits only after the new zone has
held for ``dwell_s``. The default is 3.0 s, and it is a CHOICE argued from
measured numbers rather than a measurement itself:

* the poll cadence is 2.0 s (``presence.rooms_poll_s``), so 3.0 s means
  three consecutive agreeing reads and a worst-case 4.0 s to commit;
* a walker at ~1.2 m/s is inside the narrowest 0.75 m band for ~0.6 s, so
  a pass-through cannot commit anything -- which is the point, the log
  records where he DWELLS, not every gate he crosses;
* the radar's own absence delay is already 10 s, so 3.0 s adds nothing
  material to the departure edge;
* the poll round trip is 46 / 62 / 154 / 1186 ms (min / median / p90 / max,
  25 samples, 2026-09-03), so one slow poll cannot skip a dwell window.

NOT measured: the actual chatter rate at a real band edge. The office
radar was off the network when this was written (no ARP reply at
192.168.50.51 over four tries, while the gateway answered in 2.6-8.2 ms),
so ``dwell_s`` is config and ``scripts/zone_log.py`` exists to gather the
evidence that would settle it.

THE RECORD. ``ZoneLog`` appends one JSON object per committed change to
``~/.local/state/jarvis/zones.jsonl`` -- XDG state, not ``PATHS.LOG_DIR``,
because /tmp is wiped at every boot on this box and the point of a record
is that it outlives one. The directory is 0700 and the file 0600. It
rotates at ``log_max_bytes`` (1 MB) keeping one generation, so 2 MB is the
hard ceiling.

LINE SIZES: ONE CLAIM, AND IT IS ENFORCED RATHER THAN MEASURED. Three
passes of this file quoted a "typical" office line size and a records-per-MB
figure derived from it, and a verifier failed to reproduce the figure all
three times. It is DELETED rather than restated a fourth time -- the numbers
are not repeated here even to disown them -- and what is left is a bound the
code holds up:

    ``ZoneLog.append`` REFUSES a line over ``MAX_LINE_BYTES`` (640 bytes)
    rather than writing it, and counts the refusal the way it counts an
    unwritable disk.

That makes the ceiling arithmetic on two constants instead of a claim
about how well somebody enumerated the space: a file of ``max_bytes``
holds at least ``max_bytes // MAX_LINE_BYTES`` records -- 1,562 at the
1 MB default -- and the whole log cannot pass ``(keep + 1) * max_bytes``,
2 MB, whatever the config says. The widest record the capped fields can
actually build is not stated here as a number: it is COMPUTED over the
whole space by the suite's
``test_the_widest_record_the_fields_allow_is_computed_not_asserted``, and
asserted to sit under the limit, so a field added later moves the test's
answer instead of making this paragraph a lie. What a real office
day actually writes is NOT MEASURED -- the office radar has been off the
network throughout (2026-09-03) -- and ``scripts/zone_log.py`` is the
instrument that would settle it.

The ceiling needs both halves. Room and band names come out of the config
and are otherwise arbitrary strings, so they are capped
(``MAX_NAME_CHARS``) IN BYTES AS WELL AS IN CHARACTERS: ``json.dumps``
escapes non-ASCII, so a 64-CHARACTER Chinese name is 384 bytes on the line
and 64 emoji are 768, and the character-only cap the second pass shipped
let those straight through.

**NO CAMERA FRAME, NO CROP AND NO EMBEDDING EVER GOES IN THAT FILE.** The
record is built from a fixed tuple of fields (``RECORD_FIELDS``), the
camera's whole contribution is ``{"known": bool, "label": str}``, and the
label is truncated at 64 characters and 64 escaped bytes. This module
contains NO camera code whatsoever: an opinion is handed in by the caller,
so nothing here can open a lens even by accident.

OFFLINE MODE reaches this for free and must keep doing so. The distance is
read through ``RoomSensor.read_distance``, which asks the same
``SensingPolicy`` before the same socket, so while sensing is denied the
radar is not polled at all -- ``sensor.reads`` staying at zero is the
assertion, exactly as next door. A distance read of our own over plain
urllib would have been a hole straight through that promise.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from jarvis.config import PATHS
from jarvis.logs import get_logger
from jarvis.roomsensor import DETECTION_ENTITY, MOVING_ENTITY, STILL_ENTITY

log = get_logger("zones")

# From scripts/room_sensor.py, which owns the flashing and the tuning.
# Repeated rather than imported because scripts/ is not an importable
# package; if these ever disagree, that file is right and this is stale.
GATE_M = 0.75            # one LD2410 distance gate
BLIND_M = 0.75           # nothing at all is detected inside this
STILL_FLOOR_M = 1.5      # no STILL-target detection inside this (gates 0, 1)

# The verdicts that are not a band. Spelled the way they would be spoken,
# because the day something does read them out they should not need
# translating.
UNPLACED = "in the room, unplaced"
ABSENT = "not in the room"
NO_OPINION = "no opinion"

# WHICH RULE decided, recorded on every line so a surprising verdict can be
# traced without re-deriving it.
RULE_CAMERA = "camera"
RULE_BAND = "band"
RULE_UNPLACED = "unplaced"
RULE_EMPTY = "radar-empty"
RULE_SILENT = "radar-silent"

DEFAULT_CAMERA_ZONE = "at the desk"
# A room's camera_zone, BLANK: there is no lens in this room, so the camera
# rule cannot fire for it and every verdict is the radar's. It is a real
# answer -- his kitchen is exactly this -- and not a missing one; see
# BLANK_TEXT_MEANS at the foot of this module for the rule it belongs to.
NO_CAMERA = ""
# The office is the only room with a lens. A room whose camera watches
# something else must set its own "camera_zone" in the config.
OFFICE_CAMERA_ZONE = DEFAULT_CAMERA_ZONE
OFFICE_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("just off the desk", 0.75, 1.5),
    ("the middle of the room", 1.5, 3.0),
    ("by the door", 3.0, 4.5),
)

DEFAULT_DWELL_S = 3.0
DEFAULT_MAX_BYTES = 1_000_000     # at least DEFAULT_MAX_BYTES //
                                  # MAX_LINE_BYTES = 1,562 records of ANY
                                  # shape, which is arithmetic on the two
                                  # constants and not an estimate
DEFAULT_KEEP = 1                  # so the ceiling is 2 x DEFAULT_MAX_BYTES
MAX_LABEL_CHARS = 64              # a NAME. Nothing the lens saw, ever.
# Room and band names come out of the config and are arbitrary strings, so
# they are capped like the face label -- at 64 characters AND at 64 bytes
# once JSON-escaped, because a 64-character CJK or emoji name is 384 or 768
# bytes on the line and the character-only cap did not bound it at all.
MAX_NAME_CHARS = 64
MAX_LINE_BYTES = 640              # ENFORCED in ZoneLog.append: a record
                                  # over this is refused, not written, so
                                  # the (keep + 1) * max_bytes ceiling is
                                  # not a claim about how well anyone
                                  # enumerated the space. No worst-case
                                  # record size is quoted here on purpose
                                  # -- three passes quoted one and got it
                                  # wrong; the suite COMPUTES it over the
                                  # whole space instead and asserts it is
                                  # under this. It is also the floor under
                                  # max_bytes, so a cap can never be set
                                  # below one record.

# The exact keys a record may carry. A fixed tuple rather than whatever the
# caller passed, so there is nowhere for a frame, a crop or an embedding to
# be smuggled into the file.
RECORD_FIELDS = ("at", "iso", "room", "old", "new", "rule", "distance_m",
                 "presence", "moving", "still", "camera", "held_s")


def default_log_path() -> Path:
    """``~/.local/state/jarvis/zones.jsonl`` (JARVIS_STATE_DIR in tests)."""
    return PATHS.STATE_DIR / "zones.jsonl"


def _flat(text: Any) -> str:
    """One line, whitespace collapsed, and nothing else done to it."""
    return " ".join(str(text or "").split())


def _json_cost(text: str) -> int:
    """The bytes ``text`` costs on a record line, quotes excluded.

    ``json.dumps`` escapes to ASCII, so the line is pure ASCII and a
    character's byte cost is its ESCAPED length: 1 for most of Latin-1,
    2 for a quote or a backslash, 6 for a CJK character (``\\uXXXX``) and
    12 for an emoji (a surrogate pair). Escaping is per character, so the
    costs add.
    """
    return len(json.dumps(text)) - 2


def _short(text: Any, limit: int = MAX_NAME_CHARS) -> str:
    """One collapsed line of at most ``limit`` characters AND at most
    ``limit`` bytes once it is on a record line.

    EVERY name that can reach a record goes through here -- the room, the
    band, the face label -- which is what makes a record's maximum size
    computable instead of estimated.

    BOTH caps, because the first pass capped only characters and the
    ceiling it then claimed is a byte figure: 64 CJK characters are 384
    bytes on the line and 64 emoji are 768, which walked straight past
    ``MAX_LINE_BYTES`` and past the ``(keep + 1) * max_bytes`` ceiling with
    it. Truncation is per character so a multi-byte character is never
    split in half.
    """
    flat = _flat(text)[:limit]
    if _json_cost(flat) <= limit:
        return flat                      # the ASCII case, in one call
    out, used = [], 0
    for ch in flat:
        cost = _json_cost(ch)
        if used + cost > limit:
            break
        out.append(ch)
        used += cost
    return "".join(out)


# ------------------------------------------------------------------ camera
@dataclass(frozen=True)
class CameraOpinion:
    """What the eye is willing to say. A NAME and a yes/no, and nothing
    else -- there is deliberately no field an image could live in.

    ``known`` False means the lens looked and recognised nobody, which is
    NOT a claim that the chair is empty (it may be closed by the curfew, he
    may be turned away), so it never vetoes the radar. ``None`` in place of
    the whole object means the camera had no opinion at all.
    """
    known: bool
    label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "label", _short(self.label, MAX_LABEL_CHARS))
        object.__setattr__(self, "known", bool(self.known))

    def as_record(self) -> dict:
        return {"known": self.known, "label": self.label}

    @classmethod
    def from_identify(cls, label: str,
                      score: float = 0.0) -> Optional["CameraOpinion"]:
        """``jarvis/eye.py``'s ``identify()`` result -> an opinion.

        That method returns ``(label, score)`` and uses ``("", 0.0)`` for
        every way it can decline -- under the detection bar, no gallery, a
        recogniser that raised, a score below ``match_min`` -- on its own
        stated grounds that "a consumer that has to tell them apart will
        get one of them wrong". So an empty label becomes None here, "no
        opinion", rather than ``known=False``: claiming the eye LOOKED and
        recognised nobody would invent the one distinction eye.py
        deliberately refuses to draw.

        It takes the RESULT and never a frame. Nothing in this module has
        ever held an image and nothing in it ever will.
        """
        name = " ".join(str(label or "").split())
        if not name:
            return None
        return cls(known=True, label=name)


# ------------------------------------------------------------------- bands
@dataclass(frozen=True)
class Band:
    """One named distance band, half-open: ``near_m <= d < far_m``.

    Half-open so two touching bands neither overlap nor leave a hairline
    gap at the shared edge -- 1.50 m belongs to exactly one of them.
    """
    name: str
    near_m: float
    far_m: float

    def __post_init__(self):
        object.__setattr__(self, "name", _short(self.name))

    def holds(self, metres: Optional[float]) -> bool:
        if metres is None:
            return False
        try:
            value = float(metres)
        except (TypeError, ValueError):
            return False
        return self.near_m <= value < self.far_m


@dataclass(frozen=True)
class ZoneMap:
    """The bands of ONE room, plus the name the camera's verdict carries.

    Built eagerly and validated eagerly: overlapping bands are a config
    error worth raising on, because a reading that two bands both claim has
    no defensible answer. Gaps are the opposite -- they are legitimate and
    named (``gaps()``), and a reading inside one is "unplaced".

    **A BLANK ``camera_zone`` MEANS THERE IS NO LENS IN THIS ROOM**
    (``has_camera``), and it is stored exactly as given rather than
    replaced. It used to be ``camera_zone or DEFAULT_CAMERA_ZONE``, which
    is the same disease four rounds of review chased one level up: a value
    that TRIED to say something -- "" is how a room says it has no camera
    -- was silently swapped for the built-in one, so his lensless kitchen
    described itself as "at the desk". Worse, "   " is truthy and walked
    past the ``or`` entirely, collapsed to "" in ``_short``, and the camera
    rule then wrote a NAMELESS zone into the record while a band with no
    name was refused outright. Blank is now one answer with one meaning at
    every level: see ``BLANK_TEXT_MEANS``.
    """
    room: str
    bands: Tuple[Band, ...]
    camera_zone: str = DEFAULT_CAMERA_ZONE

    def __post_init__(self):
        # The room is capped FIRST, because every complaint below is built
        # from it: capping afterwards let a 300-character room name reach
        # the log line and the refusal string whole.
        object.__setattr__(self, "room", _short(self.room))
        bands = tuple(sorted(self.bands, key=lambda b: (b.near_m, b.far_m)))
        if not bands:
            raise ValueError("%s has no bands" % (self.room or "a room"))
        seen = set()
        for band in bands:
            if not str(band.name).strip():
                raise ValueError("a band in %s has no name" % self.room)
            if band.name in seen:
                raise ValueError("%s names %r twice" % (self.room, band.name))
            seen.add(band.name)
            if not band.far_m > band.near_m:
                raise ValueError("%s: %r ends at %.2f m, which is not past its "
                                 "start at %.2f m"
                                 % (self.room, band.name, band.far_m, band.near_m))
        for lo, hi in zip(bands, bands[1:]):
            if hi.near_m < lo.far_m:
                raise ValueError("%s: %r and %r overlap between %.2f and %.2f m"
                                 % (self.room, lo.name, hi.name,
                                    hi.near_m, lo.far_m))
        object.__setattr__(self, "bands", bands)
        # NOT `or DEFAULT_CAMERA_ZONE`. _short collapses whitespace, so
        # every blank spelling -- "", "   ", "\t\n" -- lands on the one
        # value that means "no camera in this room".
        object.__setattr__(self, "camera_zone", _short(self.camera_zone))

    @property
    def has_camera(self) -> bool:
        """Is there a lens in this room at all?

        False is a GUARANTEE and not an outage: "there is no camera in the
        kitchen" and "the kitchen camera is off" are different claims, the
        same distinction jarvis/rooms.py draws between ABSENT and OFF. A
        room with no camera can never take the camera rule, so its verdict
        is the radar's, always.
        """
        return bool(self.camera_zone)

    @classmethod
    def office(cls) -> "ZoneMap":
        """The live office geometry. See the module docstring for why the
        chair is not in here."""
        return cls("office", tuple(Band(*b) for b in OFFICE_BANDS),
                   OFFICE_CAMERA_ZONE)

    def place(self, metres: Optional[float]) -> Optional[str]:
        """The band a distance falls in, or None for "no band claims it"."""
        for band in self.bands:
            if band.holds(metres):
                return band.name
        return None

    def gaps(self) -> Tuple[Tuple[float, Optional[float]], ...]:
        """Every distance no band claims, as (near, far) pairs.

        The last pair's far edge is None, meaning "and everything beyond".
        The first is normally the module's 0.75 m blind zone. Declared so
        the holes in the ladder can be read off rather than inferred from
        the arithmetic -- a gap he did not intend is a bug, and a bug you
        can print is one you can find.
        """
        out: list = []
        if self.bands[0].near_m > 0.0:
            out.append((0.0, self.bands[0].near_m))
        for lo, hi in zip(self.bands, self.bands[1:]):
            if hi.near_m > lo.far_m:
                out.append((lo.far_m, hi.near_m))
        out.append((self.bands[-1].far_m, None))
        return tuple(out)

    def describe(self) -> str:
        """The ladder as text, for a script or the console.

        A room with no lens says so in the camera row rather than printing
        a blank one: an empty first line reads as a camera whose zone
        nobody named, which is the state that no longer exists.
        """
        rows = ["  %-24s camera" % self.camera_zone] if self.has_camera else \
               ["  (no camera in this room -- every verdict is the radar's)"]
        edges: list = []
        for lo, hi in self.gaps():
            edges.append((lo, hi if hi is not None else math.inf, None))
        for band in self.bands:
            edges.append((band.near_m, band.far_m, band.name))
        for near, far, name in sorted(edges):
            far_text = "  inf" if far == math.inf else "%5.2f" % far
            rows.append("  %5.2f - %s m   %s"
                        % (near, far_text, name or "(gap -> %s)" % UNPLACED))
        return "\n".join(rows)


# ----------------------------------------------------------------- verdict
@dataclass(frozen=True)
class Verdict:
    """One fused answer, plus everything that went into it."""
    room: str
    zone: str
    rule: str
    distance_m: Optional[float] = None
    presence: Optional[bool] = None
    moving: Optional[bool] = None
    still: Optional[bool] = None
    camera: Optional[CameraOpinion] = None


def verdict(zmap: ZoneMap, *, presence: Optional[bool],
            distance_m: Optional[float] = None,
            camera: Optional[CameraOpinion] = None,
            moving: Optional[bool] = None,
            still: Optional[bool] = None) -> Verdict:
    """Fuse one camera opinion and one radar reading into a named zone.

    **THE CAMERA OVERRULES THE RADAR.** If the eye recognises him in its
    cone he is at the desk whatever the radar's range says -- including
    when the radar says the room is empty, which is the normal case, since
    the module is on the desk pointing away from the chair.

    Everything else is the radar, in the order given in the module
    docstring. None from the radar is "no opinion" and never absence.
    """
    common = dict(room=zmap.room, distance_m=distance_m, presence=presence,
                  moving=moving, still=still, camera=camera)
    if camera is not None and camera.known and zmap.has_camera:
        # ``has_camera`` is the whole of the guard and it is here rather
        # than at the caller: this is a public function, a recognised face
        # can be handed to it for any room, and a room with no lens must
        # never answer with a place that does not exist -- nor with the
        # nameless zone a blank camera_zone used to produce.
        return Verdict(zone=zmap.camera_zone, rule=RULE_CAMERA, **common)
    if presence is False:
        return Verdict(zone=ABSENT, rule=RULE_EMPTY, **common)
    if presence is not True:
        # None, "", 0, "unknown" -- anything that is not a straight yes or
        # no. Only an explicit False may say the room is empty; everything
        # else is "no opinion", because collapsing a non-answer into
        # absence is how Jarvis goes quiet on a man sitting three feet
        # away. RoomSensor.read() only ever gives True/False/None today;
        # this is the guard for the next reader.
        return Verdict(zone=NO_OPINION, rule=RULE_SILENT, **common)
    name = zmap.place(distance_m)
    if name is None:
        return Verdict(zone=UNPLACED, rule=RULE_UNPLACED, **common)
    return Verdict(zone=name, rule=RULE_BAND, **common)


# -------------------------------------------------------------- the record
@dataclass(frozen=True)
class Transition:
    """One COMMITTED zone change. What gets written, and all of it."""
    room: str
    old: str
    new: str
    rule: str
    at: float                       # unix seconds, for plotting
    iso: str                        # the same instant, for reading
    held_s: float                   # how long the new zone held before commit
    distance_m: Optional[float] = None
    presence: Optional[bool] = None
    moving: Optional[bool] = None
    still: Optional[bool] = None
    camera: Optional[CameraOpinion] = None

    def as_record(self) -> dict:
        """The JSON line. Built field by field from RECORD_FIELDS -- never
        from the caller's dict -- so nothing unexpected can reach the file.

        Every name is capped HERE as well as at its source. ``ZoneMap`` and
        ``ZoneTracker`` already cap what they hold, but ``Transition`` is
        constructible by anyone, and this is the last gate before the file:
        capping at the gate is what makes the record's size a property of
        the format rather than of every caller's good behaviour. ``iso`` is
        deliberately not capped -- a mangled timestamp is worse than a long
        one, and ``ZoneLog.append`` refuses an over-long line outright.
        """
        rec = {
            "at": round(float(self.at), 3),
            "iso": self.iso,
            "room": _short(self.room),
            "old": _short(self.old),
            "new": _short(self.new),
            "rule": _short(self.rule),
            "distance_m": (None if self.distance_m is None
                           else round(float(self.distance_m), 2)),
            "presence": self.presence,
            "moving": self.moving,
            "still": self.still,
            "camera": None if self.camera is None else self.camera.as_record(),
            "held_s": round(float(self.held_s), 2),
        }
        return {k: rec[k] for k in RECORD_FIELDS}


class ZoneLog:
    """Append-only JSONL, 0600, capped, and it never raises at the caller.

    Rotation is a single ``os.replace`` to ``<name>.1`` when the next line
    would take the file past ``max_bytes``; the previous generation is
    overwritten, so the total on disk is at most ``(keep + 1) * max_bytes``
    -- 2 MB at the defaults. That bound holds because ``max_bytes`` is
    clamped up to ``MAX_LINE_BYTES`` and NO LINE MAY EXCEED IT: names are
    capped in bytes so a record cannot get there through the config, and
    ``append`` refuses one that does anyway. So ``max_bytes`` holds at
    least ``max_bytes // MAX_LINE_BYTES`` records -- 1,562 at the default
    -- which is arithmetic on two constants rather than an average anyone
    had to measure.

    Every failure -- an unwritable directory, a full disk, a path that
    cannot be opened at all, a line past the limit, a transition carrying a
    field that is not a number -- is counted and logged, and ``append``
    returns False. IT NEVER RAISES AT THE CALLER: a logging feature may not
    take the poll loop down with it, and that includes while it is still
    building the line.
    """

    def __init__(self, path: Optional[os.PathLike | str] = None,
                 max_bytes: int = DEFAULT_MAX_BYTES, keep: int = DEFAULT_KEEP):
        self.path = Path(path) if path else default_log_path()
        try:
            self.max_bytes = max(MAX_LINE_BYTES, int(max_bytes))
        except (TypeError, ValueError):
            self.max_bytes = DEFAULT_MAX_BYTES
        # 0 or 1, nothing else: a second kept generation buys nothing a
        # larger max_bytes does not, and every extra file is one more
        # copy of where he was and when.
        self.keep = 1 if keep else 0
        self.writes = 0
        self.failures = 0
        self._warned = False

    def append(self, transition: Transition) -> bool:
        try:
            # INSIDE the try, which is the whole promise. This was built
            # above it, so a Transition carrying a field that is not a
            # number -- as_record calls float() on three of them -- raised
            # ValueError at the CALLER, i.e. inside the poll loop, which is
            # exactly what "it never raises at the caller" says it will
            # not do. The named exceptions are the ones a bad field can
            # actually produce: float()/round() give ValueError,
            # TypeError or OverflowError, json.dumps gives TypeError or
            # ValueError, and an object that is not a Transition at all
            # gives AttributeError.
            blob = (json.dumps(transition.as_record()) + "\n").encode("utf-8")
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            self.failures += 1
            if not self._warned:
                log.warning("zones: a transition could not be turned into a "
                            "record (%s); it was NOT recorded", exc)
                self._warned = True
            else:
                log.debug("zones: unrecordable transition", exc_info=True)
            return False
        if len(blob) > MAX_LINE_BYTES:
            # THE CEILING IS ENFORCED, NOT ENUMERATED. Every name is capped
            # so this cannot be reached through the config; it is the
            # backstop for a field that is not a name, and it is what makes
            # "(keep + 1) * max_bytes" a promise rather than the result of
            # one afternoon's enumeration. A record that would break the
            # bound is refused and counted, the same as an unwritable disk.
            self.failures += 1
            if not self._warned:
                log.warning("zones: a %d-byte record is past the %d-byte "
                            "line limit and was NOT recorded", len(blob),
                            MAX_LINE_BYTES)
                self._warned = True
            return False
        try:
            self._ensure_dir()
            self._rotate_if_needed(len(blob))
            # 0600 at CREATE time, not chmod after: the record says who was
            # where and when, and it must never exist world-readable, not
            # even for the microsecond between two syscalls.
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, blob)
            finally:
                os.close(fd)
        except (OSError, ValueError) as exc:
            # ValueError as well as OSError: a path carrying a null byte
            # makes os.stat and os.open raise ValueError, and that one is
            # reachable straight from zones.log_path -- so catching only
            # OSError here let a config value take the poll loop down,
            # which is the one thing this class promises it will not do.
            self.failures += 1
            if not self._warned:
                log.warning("zones: %s cannot be written (%s); transitions "
                            "are not being recorded", self.path, exc)
                self._warned = True
            else:
                log.debug("zones: append to %s failed", self.path, exc_info=True)
            return False
        self.writes += 1
        self._warned = False
        return True

    def _ensure_dir(self) -> None:
        parent = self.path.parent
        if not parent.is_dir():
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            return                       # no file yet: nothing to rotate
        if size == 0 or size + incoming <= self.max_bytes:
            return
        if self.keep <= 0:
            os.unlink(self.path)
            return
        # os.replace carries the 0600 mode across, so the kept generation is
        # as private as the live file.
        os.replace(self.path, self.path.with_name(self.path.name + ".1"))


# ------------------------------------------------------------- hysteresis
class ZoneTracker:
    """One room's committed zone, with the dwell that stops it chattering.

    ``observe()`` is called once per poll and returns a ``Transition`` ONLY
    on a committed change -- None every other time, including while a
    candidate is still serving its dwell. That is the whole anti-chatter
    guarantee: a distance oscillating across a band edge never lets one
    candidate hold long enough, so twenty crossings produce zero lines
    instead of twenty.

    The dwell applies to EVERY rule, the camera included, and to "no
    opinion" as well: a single dropped poll must not write a line, because
    ``roomsensor``'s own breaker needs three failures in a row before it
    even calls the sensor down.

    Two clocks on purpose. ``now`` is monotonic and decides the dwell, so a
    clock jump cannot commit or block a change; ``wall`` is wall time and
    only ever lands in the record, where an NTP step is a cosmetic problem
    rather than a logic one.
    """

    def __init__(self, room: str, zmap: ZoneMap, *,
                 dwell_s: float = DEFAULT_DWELL_S,
                 now: Optional[Callable[[], float]] = None,
                 wall: Optional[Callable[[], float]] = None,
                 log: Optional[ZoneLog] = None):
        self.room = _short(room or zmap.room)
        self.zmap = zmap
        try:
            self.dwell_s = max(0.0, float(dwell_s))
        except (TypeError, ValueError):
            self.dwell_s = DEFAULT_DWELL_S
        self._now = now or time.monotonic
        self._wall = wall or time.time
        self.log = log
        self.zone: str = NO_OPINION
        self.last: Optional[Verdict] = None
        self._cand: Optional[Tuple[Verdict, float]] = None

    @property
    def pending(self) -> Optional[str]:
        """The zone waiting out its dwell, if any. Diagnostics only."""
        return None if self._cand is None else self._cand[0].zone

    def observe(self, *, presence: Optional[bool],
                distance_m: Optional[float] = None,
                camera: Optional[CameraOpinion] = None,
                moving: Optional[bool] = None,
                still: Optional[bool] = None) -> Optional[Transition]:
        v = verdict(self.zmap, presence=presence, distance_m=distance_m,
                    camera=camera, moving=moving, still=still)
        self.last = v
        if v.zone == self.zone:
            self._cand = None            # the wobble came home; forget it
            return None
        if self._cand is None or self._cand[0].zone != v.zone:
            self._cand = (v, self._now())
        held = self._now() - self._cand[1]
        if held < self.dwell_s:
            return None
        old, self.zone = self.zone, v.zone
        self._cand = None
        at = self._wall()
        change = Transition(
            room=self.room, old=old, new=v.zone, rule=v.rule, at=at,
            iso=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(at)),
            held_s=held, distance_m=v.distance_m, presence=v.presence,
            moving=v.moving, still=v.still, camera=v.camera)
        log.info("zone: %s %s -> %s (%s, %s)", self.room, old, v.zone, v.rule,
                 "no distance" if v.distance_m is None
                 else "%.2f m" % v.distance_m)
        if self.log is not None:
            self.log.append(change)
        return change


# ---------------------------------------------------------------- watcher
class ZoneWatcher:
    """A room sensor, a map, a tracker and the log, driven one poll at a
    time by somebody else's loop.

    There is no thread here on purpose: this branch logs and nothing more,
    so the cadence belongs to whatever eventually owns it
    (``scripts/zone_log.py`` today, the app later).

    ``camera`` is a zero-argument callable returning a ``CameraOpinion`` or
    None. THIS MODULE NEVER OPENS A LENS -- it cannot, there is no camera
    code in it -- and a hook that raises is treated as no opinion, so a
    closed gate or a broken face pipeline costs the zone log nothing.

    The distance is read only when the radar says someone is there, which
    keeps an empty room at one request per poll.
    """

    def __init__(self, room: str, sensor: Any, zmap: ZoneMap, *,
                 dwell_s: float = DEFAULT_DWELL_S,
                 log_file: Optional[ZoneLog] = None,
                 camera: Optional[Callable[[], Optional[CameraOpinion]]] = None,
                 now: Optional[Callable[[], float]] = None,
                 wall: Optional[Callable[[], float]] = None,
                 read_bits: bool = True):
        self.room = _short(room or zmap.room)
        self.sensor = sensor
        self.zmap = zmap
        self.read_bits = bool(read_bits)
        self._camera = camera
        self.log_file = log_file if log_file is not None else ZoneLog()
        self.tracker = ZoneTracker(self.room, zmap, dwell_s=dwell_s, now=now,
                                   wall=wall, log=self.log_file)

    @property
    def zone(self) -> str:
        return self.tracker.zone

    def poll(self) -> Optional[Transition]:
        """One reading. Returns a Transition only on a committed change.

        **ZERO IS NOT A PLACE.** ``parse_cm`` says so in as many words: 0
        is a real reading meaning "no target of THIS kind", so it is
        treated here exactly like a missing one -- it falls through to the
        next entity, and if nothing is left it is recorded as no distance
        at all rather than as a man standing on the module. The first pass
        guarded the fallback with ``distance is None`` and so skipped it
        for a detection entity publishing 0, which is precisely the case
        below.

        UNVERIFIED, and it must stay labelled that way until the hardware
        answers: what the office LD2410 actually publishes on "Detection
        distance" when only a STILL target is present. The device was off
        the network when this was written (2026-09-03), so whether that
        entity goes null, goes 0 or keeps the last value has NOT been
        measured. All three are handled -- null and 0 both fall through to
        the still distance -- but "handled" here means "reasoned about",
        not "observed". ``scripts/zone_log.py`` is the instrument that
        settles it.
        """
        presence = self.sensor.read()
        distance = moving = still = None
        if presence is True:
            distance = self.sensor.read_distance(DETECTION_ENTITY)
            if self.read_bits:
                moving_m = self.sensor.read_distance(MOVING_ENTITY)
                still_m = self.sensor.read_distance(STILL_ENTITY)
                moving = None if moving_m is None else moving_m > 0.0
                still = None if still_m is None else still_m > 0.0
                if not distance:
                    # The detection entity is the primary, but a firmware
                    # that publishes neither a number nor anything at all
                    # for it must not cost the band: the STILL distance
                    # comes first because a man sitting is the case the
                    # zones exist for.
                    distance = (still_m if still_m else None) or \
                               (moving_m if moving_m else None)
            distance = distance or None       # 0.0 is "no target", not 0 m
        return self.tracker.observe(presence=presence, distance_m=distance,
                                    camera=self._camera_opinion(),
                                    moving=moving, still=still)

    def _camera_opinion(self) -> Optional[CameraOpinion]:
        """The eye's opinion, or None -- and NOT ASKED AT ALL in a room the
        config says has no camera.

        ``verdict`` already refuses the camera rule for such a room, so
        this is not what makes the answer right; it is what stops a
        lensless room paying for the hook, and it is one fewer path by
        which anything here can reach the vision lane. His kitchen has no
        camera, and now nothing in the kitchen's poll goes near one.
        """
        if self._camera is None or not self.zmap.has_camera:
            return None
        try:
            got = self._camera()
        except Exception:  # noqa: BLE001 - a broken eye is not a verdict
            log.debug("zones: the camera hook failed", exc_info=True)
            return None
        return got if isinstance(got, CameraOpinion) else None


# ----------------------------------------------------------------- config
# ONE VALIDATOR FOR THE WHOLE SECTION, and nothing reads a zones value
# around it.
#
# Three passes of this module fixed one level and left the level above it
# open: a malformed BAND list fell back to the built-in ladder, then a
# ``zones.rooms`` of the wrong SHAPE fell back, then a ``zones`` KEY of the
# wrong shape fell back -- and ``zones.enabled`` was read with ``bool()``,
# so "false", "no", "off", "0" and null all left recording switched ON.
# Every one of those was the same disease treated one level at a time.
#
# THE RULE, and it has no exceptions and no special levels:
#
#     A config that TRIES to say something about zones and gets it wrong is
#     REFUSED BY NAME and records nothing. Only a key that is entirely
#     ABSENT falls back to a default.
#
# So the shape of the whole section is DECLARED below as data
# (``SECTION_SHAPE``, ``ROOM_SHAPE``, ``BAND_SHAPE``), one walker
# (``_Shape``) checks every level against it, and ``read_zones`` is the
# only door: ``dwell_s``, ``log_path``, ``log_max_bytes``, ``log_keep``,
# ``zone_maps``, ``rejected_rooms`` and ``zone_map_for`` are all views over
# its result. A level added to the declaration without a check is
# impossible -- the declaration IS the check -- and
# ``tests/test_zones.py`` walks the same declaration as a cross product of
# every level against every wrong shape, so a level added to the config and
# not to the declaration fails the suite instead of shipping.
#
# WHAT REFUSING COSTS. A refusal ABOVE the room level (the section itself,
# ``enabled``, ``dwell_s``, the log keys, or a ``rooms`` that is not a
# list) poisons everything: there are no entries to salvage and no defaults
# it would be honest to use, so every room records nothing. A refusal
# INSIDE one room entry costs that room and no other, exactly as one bad
# room does in ``roomfabric.room_specs``.
#
# SHAPE IS MOST OF IT AND NOT ALL OF IT. The tables answer "is this the
# right KIND of thing"; two values pass that and are still unusable, and
# both are refused by name at the same level as any other wrong value: a
# ``log_path`` that is text and cannot become a path ("~nobody/x", or one
# carrying a null byte -- ``read_zones``), and a band list whose geometry
# does not describe a ladder (``ZoneMap`` raises, ``_read_rooms`` catches).
#
# WHAT IT DOES NOT DO, declared rather than overlooked: a key the tables
# below do not name is left alone. A config written by a LATER Jarvis must
# not be refused wholesale by an earlier one, so an unknown key is not an
# error here -- only a declared key of the wrong shape is.
_MISSING = object()      # the key is not in the config at all
_REQUIRED = object()     # ... and there is no default for it, so that is fatal

SECTION_KEY = "zones"
ROOMS_KEY = "zones.rooms"


def _is_number(value) -> bool:
    """A number this module can actually use as a distance or a size.

    Three ways a "number" is not one. ``bool`` is an ``int`` in Python and
    ``"near_m": true`` is not a distance. NaN and inf survive ``float()``
    and then poison every comparison that places a body in a band. And an
    integer big enough -- ``10 ** 400`` in a hand-edited file -- makes
    ``float()`` itself raise OverflowError, which is how this check
    escaped as an exception at the caller in its first draft.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


# The vocabulary of shapes, spelled the way the refusal will read.
SHAPES: Dict[str, Callable[[Any], bool]] = {
    "a mapping": lambda v: isinstance(v, dict),
    "a list": lambda v: isinstance(v, (list, tuple)),
    "true or false": lambda v: isinstance(v, bool),
    "a number": _is_number,
    "text": lambda v: isinstance(v, str),
}

# (key, shape, default). ``_REQUIRED`` means there is no sane default, so
# the key being absent is itself a refusal; ``_MISSING`` as a default means
# "absent is a real answer the caller handles" -- true of ``rooms`` alone,
# where absence is the one surviving bridge to the built-in ladder.
SECTION_SHAPE = (
    ("enabled", "true or false", True),
    ("dwell_s", "a number", DEFAULT_DWELL_S),
    ("log_path", "text", ""),
    ("log_max_bytes", "a number", DEFAULT_MAX_BYTES),
    ("log_keep", "a number", DEFAULT_KEEP),
    ("rooms", "a list", _MISSING),
)
ROOM_SHAPE = (
    ("name", "text", _REQUIRED),
    ("enabled", "true or false", True),
    ("camera_zone", "text", DEFAULT_CAMERA_ZONE),
    ("bands", "a list", _REQUIRED),
)
BAND_SHAPE = (
    ("name", "text", _REQUIRED),
    ("near_m", "a number", _REQUIRED),
    ("far_m", "a number", _REQUIRED),
)

# WHAT A PRESENT-BUT-BLANK TEXT VALUE MEANS, declared per key.
#
# Shape is one way a value can be wrong and blankness is another, and it is
# the one four rounds of review did not reach. The tables above answer "is
# this the right KIND of thing"; "" and "   " are text, so they pass, and
# what happened next was decided by whichever ``or`` happened to be on the
# path -- which is how ``camera_zone: ""`` became "at the desk" in a room
# with no lens, and how ``camera_zone: "   "`` became a zone with NO NAME
# in the record while a band with no name was refused outright. Two
# opposite answers to the same question, neither of them written down.
#
# So it is written down. A blank text value is a value that TRIED to say
# something, and every declared text key says here what it says:
#
#   REFUSED               -- blank is a mistake; the room records nothing
#                            and the dotted path is named
#   anything else         -- blank is an ANSWER, and this is the answer
#
# ``tests/test_zones.py`` walks the three shape tables and asserts every
# "text" key appears here AND that the code does what the entry claims, so
# a text key added to a table without a decision about blankness fails the
# suite instead of shipping with a silent substitution behind it.
REFUSED = "refused: the room records nothing and the key is named"
BLANK_TEXT_MEANS = {
    "zones.log_path": "the default log path -- \"\" is the shipped value",
    "zones.rooms[].name": REFUSED,
    "zones.rooms[].camera_zone": ("the room has NO camera, so the camera "
                                  "rule cannot fire for it and the radar "
                                  "decides (ZoneMap.has_camera)"),
    "zones.rooms[].bands[].name": REFUSED,
}


def _kind(value) -> str:
    """What the config actually put there, in the words ``SHAPES`` uses."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true or false"
    if isinstance(value, float) and not math.isfinite(value):
        return "%r, which is not a finite number" % value
    if isinstance(value, int) and not _is_number(value):
        return "a whole number too large to use"
    for want, ok in SHAPES.items():
        if ok(value):
            return want
    return "a %s" % type(value).__name__


class _Shape:
    """The walker. Checks one value against one declared shape and files
    the failure under the DOTTED CONFIG PATH, which is the thing he has to
    go and edit.

    It never raises and never returns a substitute: a value that fails is
    absent from the result and present in ``refused``, and the caller
    decides how far up that reaches.
    """

    def __init__(self):
        self.refused: Dict[str, str] = {}

    def refuse(self, path: str, why: str, *, deliberate: bool = False) -> bool:
        # setdefault: the FIRST reason is the one to fix, and a later,
        # vaguer complaint about the same key must not overwrite it.
        #
        # ``deliberate`` is the one refusal that is not a mistake: a room
        # switched off in its own entry. It still records nothing and is
        # still named, but telling him to go and fix a switch he set on
        # purpose -- at ERROR, every poll -- would be wrong.
        if path not in self.refused:
            self.refused[path] = "%s %s" % (path, why)
            if deliberate:
                log.info("zones: %s, so it records nothing", self.refused[path])
            else:
                log.error("zones: %s -- it records nothing until that is "
                          "fixed; it is NOT falling back to the built-in "
                          "ladder", self.refused[path])
        return False

    def value(self, path: str, raw, want: str) -> Tuple[Any, bool]:
        """One value that is not read out of a mapping: the section, a room
        entry, a band entry."""
        if SHAPES[want](raw):
            return raw, True
        return None, self.refuse(path, "is %s; %s was expected"
                                 % (_kind(raw), want))

    def field(self, holder: dict, path: str, key: str, want: str,
              default) -> Tuple[Any, bool]:
        raw = holder.get(key, _MISSING)
        where = "%s.%s" % (path, key)
        if raw is _MISSING:
            if default is _REQUIRED:
                return None, self.refuse(where, "is missing")
            return default, True       # ABSENT is the only silence there is
        return self.value(where, raw, want)

    def block(self, holder: dict, path: str, shape) -> Tuple[dict, bool]:
        """Every declared field of one mapping.

        It does not stop at the first bad field: one run of the instrument
        should name everything that is wrong, not send him back round the
        loop for the next one.
        """
        out, ok = {}, True
        for key, want, default in shape:
            out[key], good = self.field(holder, path, key, want, default)
            ok = good and ok
        return out, ok


@dataclass(frozen=True)
class ZonesConfig:
    """The whole ``zones`` section after one validation pass.

    ``refused`` is keyed by DOTTED CONFIG PATH and is the canonical answer;
    ``room_refusals`` is the same set of messages keyed by the room they
    cost, for the rooms whose name could be read at all. Both are built in
    the one walk, so they cannot drift apart.
    """
    enabled: bool
    dwell_s: float
    log_path: Path
    log_max_bytes: int
    log_keep: int
    rooms: Dict[str, ZoneMap]
    refused: Dict[str, str]
    room_refusals: Dict[str, str]
    bridge: bool                    # zones.rooms was ABSENT: the built-in
    poisoned: bool                  # a refusal ABOVE the room level

    def why(self, room: str) -> str:
        """Why ``room`` records nothing, or "" when it does record.

        The section-level reason wins, because when the section is poisoned
        that is the only thing worth telling him -- a per-room complaint
        underneath it would send him to the wrong line of the file.
        """
        if self.poisoned:
            return sorted(self.refused.values())[0] if self.refused else ""
        if not self.enabled:
            return "%s.enabled is false; nothing is recorded" % SECTION_KEY
        key = _room_key(room)
        if key in self.rooms:
            return ""
        if key in self.room_refusals:
            return self.room_refusals[key]
        if not self.bridge:
            return "%r is not in %s" % (key, ROOMS_KEY)
        return ""          # no rooms key at all: zone_map_for owns the bridge


def _cfg_raw(cfg, key: str):
    """The value at ``key``, or ``_MISSING`` when the key is absent.

    ``AssistantConfig.get`` returns its default only for a key that is not
    there, and the value -- None included -- for a key that is, so passing
    a sentinel as the default is what tells the two apart. Every other
    dotted read of a zones key has been deleted: this is the only one, and
    it asks for the SECTION, so a ``zones`` that is a string or a number
    cannot hide behind a dotted lookup that quietly answers "absent" for
    every key under it.
    """
    get = getattr(cfg, "get", None)
    if not callable(get):
        return _MISSING
    try:
        return get(key, _MISSING)
    except Exception:  # noqa: BLE001 - a broken config must not cost the log
        log.debug("zones: cfg.get(%s) failed", key, exc_info=True)
        return _MISSING


def _room_key(name: Any) -> str:
    """A room's lookup name: collapsed, lower-cased and capped.

    The same function builds the key in ``read_zones`` and looks it up in
    ``zone_map_for``, so the two cannot disagree about what counts as the
    same room. Lower-cased BEFORE the cap, not after, so that the key is
    exactly ``_short`` of the lower-cased name and the clash check below is
    looking at the same string the lookup will.
    """
    return _short(_flat(name).lower())


def _cap_clash(flats) -> str:
    """The complaint for two names that DIFFER but become one once capped.

    ``_short`` can turn two distinct config names into a single one, and
    the duplicate check downstream would then refuse the room for a
    repetition the user never wrote -- sending him looking for a second
    entry that is not in his file. Refusing is still right; the reason has
    to be the true one.

    It has to say what ACTUALLY happened, which is why the length is
    computed and not quoted: ``_short`` cuts at ``MAX_NAME_CHARS``
    characters AND at ``MAX_NAME_CHARS`` escaped bytes, so two names in
    CJK collide after ten characters, not sixty-four. The previous message
    said "share their first 64 characters" whatever the alphabet, which was
    simply untrue for his non-ASCII case.

    ``flats`` are whitespace-collapsed and, for room names, lower-cased
    already, so a name that differs only in case is a real duplicate here
    and not a clash. Empty string when there is no such pair.
    """
    seen: Dict[str, str] = {}
    for flat in flats:
        key = _short(flat)
        if seen.setdefault(key, flat) != flat:
            return ("two names are the same once each is cut to fit one "
                    "record: both become %r, which is %d characters and %d "
                    "escaped bytes, and a name is cut at %d of each -- they "
                    "differ only past that"
                    % (key, len(key), _json_cost(key), MAX_NAME_CHARS))
    return ""


def _read_bands(shape: _Shape, raw, where: str) -> Optional[Tuple[Band, ...]]:
    """Every band of one room, or None when the room records nothing."""
    if not raw:
        shape.refuse(where, "is empty; a room with no bands has no ladder")
        return None
    out, ok = [], True
    flats = []
    for j, entry in enumerate(raw):
        at = "%s[%d]" % (where, j)
        band, good = shape.value(at, entry, "a mapping")
        if not good:
            ok = False
            continue
        fields, good = shape.block(band, at, BAND_SHAPE)
        if not good:
            ok = False
            continue
        flats.append(_flat(fields["name"]))
        out.append((at, fields))
    if not ok:
        return None
    clash = _cap_clash(flats)
    if clash:
        shape.refuse(where, "cannot be used: %s" % clash)
        return None
    return tuple(Band(f["name"], float(f["near_m"]), float(f["far_m"]))
                 for _, f in out)


def _read_rooms(shape: _Shape, raw) -> Tuple[Dict[str, ZoneMap],
                                             Dict[str, str]]:
    """The usable ladders and the per-room refusals, in config order.

    One bad room takes itself and nothing else, exactly as in
    ``roomfabric.room_specs``.
    """
    maps: Dict[str, ZoneMap] = {}
    room_refusals: Dict[str, str] = {}
    full: Dict[str, str] = {}        # room key -> the name before capping
    first: Dict[str, str] = {}       # room key -> the path that claimed it

    def refuse(key: str, path: str, why: str,
               deliberate: bool = False) -> None:
        shape.refuse(path, why, deliberate=deliberate)
        if key:
            room_refusals.setdefault(key, shape.refused[path])

    def under(path: str) -> str:
        """The first refusal filed at or below ``path``. Never empty at the
        call sites below -- they are only reached after one was filed --
        but written so that a future one cannot IndexError on the log."""
        found = sorted(v for k, v in shape.refused.items()
                       if k == path or k.startswith(path + ".")
                       or k.startswith(path + "["))
        return found[0] if found else "%s is unusable" % path

    for i, entry in enumerate(raw):
        where = "%s[%d]" % (ROOMS_KEY, i)
        room, ok = shape.value(where, entry, "a mapping")
        if not ok:
            continue
        fields, ok = shape.block(room, where, ROOM_SHAPE)
        flat = _flat(fields["name"] or "").lower()
        name = _short(flat)
        if not ok:
            # The name may itself have been the wrong shape, in which case
            # there is no room key to file this under and the dotted path
            # in ``shape.refused`` is the whole answer.
            if name:
                room_refusals.setdefault(name, under(where))
            continue
        if not name:
            refuse("", "%s.name" % where, "is empty")
            continue
        if name in maps or name in room_refusals:
            # Two ladders both claiming one room have no defensible answer,
            # so NEITHER is used. Which of the two it is matters: a name he
            # wrote twice and a name the cap folded into another send him
            # looking for different things.
            maps.pop(name, None)
            clash = _cap_clash([full.get(name, flat), flat])
            refuse(name, where,
                   ("and %s cannot both be the room: %s"
                    % (first.get(name, ROOMS_KEY), clash)) if clash else
                   ("names %r again -- %s already does, and two ladders "
                    "cannot both be the room"
                    % (name, first.get(name, ROOMS_KEY))))
            continue
        full[name], first[name] = flat, where
        if not fields["enabled"]:
            refuse(name, "%s.enabled" % where,
                   "is false, so the room is switched off", deliberate=True)
            continue
        bands = _read_bands(shape, fields["bands"], "%s.bands" % where)
        if bands is None:
            room_refusals.setdefault(name, under("%s.bands" % where))
            continue
        try:
            maps[name] = ZoneMap(name, bands, fields["camera_zone"])
        except ValueError as exc:
            # The geometry, not the shape: overlapping bands, a band that
            # ends before it starts, one name used twice. Shape is the
            # table's job and this is the only thing left.
            refuse(name, "%s.bands" % where, "does not describe a ladder (%s)"
                   % exc)
    return maps, room_refusals


def read_zones(cfg) -> ZonesConfig:
    """THE validator. Walk the whole declared shape of ``zones`` once.

    Everything else in this module that wants a zones value asks this and
    nothing else, so there is no path by which a value reaches the log
    without having been checked.
    """
    shape = _Shape()
    raw = _cfg_raw(cfg, SECTION_KEY)
    if raw is _MISSING:
        # The config predates this section entirely. That, and only that,
        # is silence, and silence is the one thing the built-in office
        # ladder may still answer.
        return ZonesConfig(True, DEFAULT_DWELL_S, default_log_path(),
                           DEFAULT_MAX_BYTES, DEFAULT_KEEP, {}, {}, {},
                           bridge=True, poisoned=False)

    def nothing() -> ZonesConfig:
        """A section that got itself wrong above the room level. No value
        out of it is trustworthy, so none is used and nothing records."""
        return ZonesConfig(False, DEFAULT_DWELL_S, default_log_path(),
                           DEFAULT_MAX_BYTES, DEFAULT_KEEP, {},
                           dict(shape.refused), {}, bridge=False,
                           poisoned=True)

    section, ok = shape.value(SECTION_KEY, raw, "a mapping")
    if not ok:
        return nothing()
    values, ok = shape.block(section, SECTION_KEY, SECTION_SHAPE)
    if not ok:
        return nothing()

    rooms_raw = values["rooms"]
    bridge = rooms_raw is _MISSING
    maps: Dict[str, ZoneMap] = {}
    room_refusals: Dict[str, str] = {}
    if values["enabled"] and not bridge:
        maps, room_refusals = _read_rooms(shape, rooms_raw)
    if not values["enabled"]:
        # The master switch. Off is an ANSWER, not a refusal: there is
        # nothing to go and fix, so nothing is named -- but a built-in
        # ladder that answered underneath it would not be a switch, so the
        # bridge is closed here too.
        bridge = False
    text = values["log_path"].strip()
    try:
        if "\0" in text:
            # A null byte makes os.open raise ValueError rather than
            # OSError, so it would have walked past ZoneLog's handler too.
            raise ValueError("it contains a null byte")
        chosen = Path(text).expanduser() if text else default_log_path()
    except (RuntimeError, ValueError) as exc:
        # THE SAME RULE, ONE LEVEL DOWN FROM SHAPE. "~nosuchuser/zones.jsonl"
        # is text, so the table passes it, and then expanduser RAISES
        # RuntimeError -- which took the exception out through read_zones to
        # the caller instead of refusing the key. Shape is not the only way a
        # value can be wrong, and a wrong value is refused BY NAME at every
        # level including this one.
        shape.refuse("%s.log_path" % SECTION_KEY,
                     "is text but cannot be used as a path (%s)" % exc)
        return nothing()
    return ZonesConfig(
        enabled=values["enabled"],
        dwell_s=max(0.0, float(values["dwell_s"])),
        log_path=chosen,
        log_max_bytes=max(MAX_LINE_BYTES, int(values["log_max_bytes"])),
        log_keep=1 if values["log_keep"] else 0,
        rooms=maps, refused=dict(shape.refused), room_refusals=room_refusals,
        bridge=bridge, poisoned=False)


# The views. Each one is ``read_zones`` and a field: there is deliberately
# no second way to read any of these keys.
def dwell_s(cfg) -> float:
    return read_zones(cfg).dwell_s


def log_max_bytes(cfg) -> int:
    return read_zones(cfg).log_max_bytes


def log_keep(cfg) -> int:
    """1 to keep one rotated generation (the default), 0 to discard it."""
    return read_zones(cfg).log_keep


def log_path(cfg) -> Path:
    return read_zones(cfg).log_path


def zone_maps(cfg) -> Dict[str, ZoneMap]:
    """Every configured room's USABLE ladder, in config order.

    A room the config names but this module refuses is absent from here and
    present in ``rejected_rooms``; the two together are the whole picture.
    """
    return read_zones(cfg).rooms


def rejected_rooms(cfg) -> Dict[str, str]:
    """Everything in the ``zones`` section that will record NOTHING, keyed
    by the DOTTED CONFIG PATH that is wrong -- which is the line he has to
    go and edit.

    Empty is the healthy answer, and anything in here is worth printing at
    the top of a run. A room whose entry could be read far enough to know
    its name also appears in ``read_zones(cfg).room_refusals`` under that
    name; the path is the canonical key because a room whose NAME is the
    broken thing has no other.
    """
    return read_zones(cfg).refused


def zone_map_for(cfg, room: str) -> Optional[ZoneMap]:
    """The ladder for ONE room, or None -- and None means RECORD NOTHING.

    Every way to get None, and every one of them deliberate:

    * anything in the section is the wrong shape -- ``zones`` itself,
      ``enabled``, ``dwell_s``, a log key, ``rooms``, a room entry, a room
      field, a band entry or a band field. The config TRIED to say
      something about zones and got it wrong, and a wrong answer is refused
      rather than overruled;
    * ``log_path`` is text and still cannot be a path -- a home directory
      that does not exist, a null byte. Shape is not the only way to be
      wrong, and this level is not exempt either;
    * ``zones.enabled`` is false -- the master switch, and an off switch a
      built-in default could walk around would not be one;
    * the room's own entry says ``enabled: false``;
    * the room's entry is duplicated, or its geometry does not describe a
      ladder -- it is REFUSED BY NAME, never replaced by the shipped one. A
      typo in his band list must not leave the log being written against
      the old bands and looking like it worked;
    * the config HAS a ``zones.rooms`` list and this room is not in it.

    The single fallback left is a config written BEFORE this section
    existed: NO ``zones`` SECTION AT ALL, or a section with no ``rooms``
    key, still gets the built-in office ladder, the same way roomfabric
    falls back to the singular ``presence.room_sensor_*`` keys. Absence is
    the only silence. Note that ``AssistantConfig`` serves the ``zones``
    block out of DEFAULTS, so in the running app that bridge is never the
    path taken -- the office ladder always arrives through the config,
    which is exactly why an edit to it has to be refused loudly rather than
    replaced.
    """
    zones = read_zones(cfg)
    name = _room_key(room)
    got = zones.rooms.get(name)
    if got is not None:
        return got
    why = zones.why(name)
    if why:
        log.error("zones: %r records nothing: %s", name, why)
        return None
    if zones.bridge and name == "office":
        return ZoneMap.office()
    log.warning("zones: there is no ladder for %r; it records nothing", name)
    return None


__all__ = ["ABSENT", "BAND_SHAPE", "BLANK_TEXT_MEANS", "BLIND_M", "Band",
           "CameraOpinion", "NO_CAMERA", "REFUSED",
           "DEFAULT_CAMERA_ZONE", "DEFAULT_DWELL_S", "GATE_M",
           "MAX_LABEL_CHARS", "MAX_LINE_BYTES",
           "MAX_NAME_CHARS", "NO_OPINION", "OFFICE_BANDS", "RECORD_FIELDS",
           "ROOMS_KEY", "ROOM_SHAPE", "RULE_BAND", "RULE_CAMERA", "RULE_EMPTY",
           "RULE_SILENT", "RULE_UNPLACED", "SECTION_KEY", "SECTION_SHAPE",
           "SHAPES", "STILL_FLOOR_M", "Transition", "UNPLACED", "Verdict",
           "ZoneLog", "ZoneMap", "ZoneTracker", "ZoneWatcher", "ZonesConfig",
           "default_log_path", "dwell_s", "log_keep", "log_max_bytes",
           "log_path", "read_zones", "rejected_rooms", "verdict",
           "zone_map_for", "zone_maps"]
