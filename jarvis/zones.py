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
hard ceiling. MEASURED line sizes: 242 bytes with no camera opinion, 272
with one, 330 at the 64-character label ceiling -- so 1 MB is 3,000 to
4,100 transitions, which at the dwell above is far more than a day.

**NO CAMERA FRAME, NO CROP AND NO EMBEDDING EVER GOES IN THAT FILE.** The
record is built from a fixed tuple of fields (``RECORD_FIELDS``), the
camera's whole contribution is ``{"known": bool, "label": str}``, and the
label is truncated at 64 characters. This module contains NO camera code
whatsoever: an opinion is handed in by the caller, so nothing here can open
a lens even by accident.

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
# The office is the only room with a lens. A room whose camera watches
# something else must set its own "camera_zone" in the config.
OFFICE_CAMERA_ZONE = DEFAULT_CAMERA_ZONE
OFFICE_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("just off the desk", 0.75, 1.5),
    ("the middle of the room", 1.5, 3.0),
    ("by the door", 3.0, 4.5),
)

DEFAULT_DWELL_S = 3.0
DEFAULT_MAX_BYTES = 1_000_000     # 3,000-4,100 records; measured, see above
DEFAULT_KEEP = 1                  # so the ceiling is 2 x DEFAULT_MAX_BYTES
MAX_LINE_BYTES = 512              # the worst measured line is 330; this is
                                  # the floor under max_bytes, so a cap can
                                  # never be set below one record
MAX_LABEL_CHARS = 64              # a NAME. Nothing the lens saw, ever.

# The exact keys a record may carry. A fixed tuple rather than whatever the
# caller passed, so there is nowhere for a frame, a crop or an embedding to
# be smuggled into the file.
RECORD_FIELDS = ("at", "iso", "room", "old", "new", "rule", "distance_m",
                 "presence", "moving", "still", "camera", "held_s")


def default_log_path() -> Path:
    """``~/.local/state/jarvis/zones.jsonl`` (JARVIS_STATE_DIR in tests)."""
    return PATHS.STATE_DIR / "zones.jsonl"


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
        text = " ".join(str(self.label or "").split())[:MAX_LABEL_CHARS]
        object.__setattr__(self, "label", text)
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
    """
    room: str
    bands: Tuple[Band, ...]
    camera_zone: str = DEFAULT_CAMERA_ZONE

    def __post_init__(self):
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
        object.__setattr__(self, "camera_zone",
                           str(self.camera_zone or DEFAULT_CAMERA_ZONE))

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
        """The ladder as text, for a script or the console."""
        rows = ["  %-24s camera" % self.camera_zone]
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
    if camera is not None and camera.known:
        return Verdict(zone=zmap.camera_zone, rule=RULE_CAMERA, **common)
    if presence is None:
        return Verdict(zone=NO_OPINION, rule=RULE_SILENT, **common)
    if not presence:
        return Verdict(zone=ABSENT, rule=RULE_EMPTY, **common)
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
        from the caller's dict -- so nothing unexpected can reach the file."""
        rec = {
            "at": round(float(self.at), 3),
            "iso": self.iso,
            "room": self.room,
            "old": self.old,
            "new": self.new,
            "rule": self.rule,
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
    -- 2 MB at the defaults. Records measure 242-330 bytes (measured, see
    the module docstring), so 1 MB is 3,000-4,100 of them.

    Every failure -- an unwritable directory, a full disk -- is counted and
    logged once at debug, and ``append`` returns False. A logging feature
    may not take the poll loop down with it.
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
        line = json.dumps(transition.as_record()) + "\n"
        blob = line.encode("utf-8")
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
        except OSError as exc:
            self.failures += 1
            if not self._warned:
                log.warning("zones: %s is not writable (%s); transitions are "
                            "not being recorded", self.path, exc)
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
        self.room = str(room or zmap.room)
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
        self.room = str(room or zmap.room)
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
        """One reading. Returns a Transition only on a committed change."""
        presence = self.sensor.read()
        distance = moving = still = None
        if presence:
            distance = self.sensor.read_distance(DETECTION_ENTITY)
            if self.read_bits:
                moving_m = self.sensor.read_distance(MOVING_ENTITY)
                still_m = self.sensor.read_distance(STILL_ENTITY)
                moving = None if moving_m is None else moving_m > 0.0
                still = None if still_m is None else still_m > 0.0
                if distance is None:
                    # The detection entity is the primary, but a firmware
                    # that does not publish it must not cost the band: the
                    # STILL distance comes first because a man sitting is
                    # the case the zones exist for.
                    distance = (still_m if still_m else None) or \
                               (moving_m if moving_m else None)
        return self.tracker.observe(presence=presence, distance_m=distance,
                                    camera=self._camera_opinion(),
                                    moving=moving, still=still)

    def _camera_opinion(self) -> Optional[CameraOpinion]:
        if self._camera is None:
            return None
        try:
            got = self._camera()
        except Exception:  # noqa: BLE001 - a broken eye is not a verdict
            log.debug("zones: the camera hook failed", exc_info=True)
            return None
        return got if isinstance(got, CameraOpinion) else None


# ----------------------------------------------------------------- config
def _cfg_get(cfg, key: str, default=None):
    get = getattr(cfg, "get", None)
    if not callable(get):
        return default
    try:
        value = get(key, default)
    except Exception:  # noqa: BLE001 - a broken config must not cost the log
        log.debug("zones: cfg.get(%s) failed", key, exc_info=True)
        return default
    return default if value is None else value


def dwell_s(cfg) -> float:
    try:
        return max(0.0, float(_cfg_get(cfg, "zones.dwell_s", DEFAULT_DWELL_S)))
    except (TypeError, ValueError):
        return DEFAULT_DWELL_S


def log_max_bytes(cfg) -> int:
    try:
        return max(MAX_LINE_BYTES,
                   int(_cfg_get(cfg, "zones.log_max_bytes", DEFAULT_MAX_BYTES)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES


def log_keep(cfg) -> int:
    """1 to keep one rotated generation (the default), 0 to discard it."""
    try:
        return 1 if int(_cfg_get(cfg, "zones.log_keep", DEFAULT_KEEP)) else 0
    except (TypeError, ValueError):
        return DEFAULT_KEEP


def log_path(cfg) -> Path:
    text = str(_cfg_get(cfg, "zones.log_path", "") or "").strip()
    return Path(text).expanduser() if text else default_log_path()


def zone_maps(cfg) -> Dict[str, ZoneMap]:
    """Every configured room's ladder, in config order.

    Shaped after ``roomfabric.room_specs``: a LIST of labelled entries, and
    a broken one is SKIPPED with a warning rather than raised on -- one
    room with a typo in it must not take the others down. ``zones.enabled``
    is the master switch and turns the whole record off.
    """
    if not bool(_cfg_get(cfg, "zones.enabled", True)):
        return {}
    raw = _cfg_get(cfg, "zones.rooms", None)
    if not isinstance(raw, (list, tuple)):
        return {}
    out: Dict[str, ZoneMap] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if not bool(entry.get("enabled", True)):
            continue
        name = " ".join(str(entry.get("name") or "").split()).lower()
        if not name or name in out:
            log.warning("zones: a room entry has no name, or repeats one; "
                        "skipped")
            continue
        try:
            bands = tuple(Band(str(b["name"]), float(b["near_m"]),
                               float(b["far_m"]))
                          for b in (entry.get("bands") or []))
            out[name] = ZoneMap(name, bands,
                                str(entry.get("camera_zone") or
                                    DEFAULT_CAMERA_ZONE))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("zones: %s has unusable bands (%s); skipped", name, exc)
    return out


def zone_map_for(cfg, room: str) -> Optional[ZoneMap]:
    """The ladder for ONE room, or None.

    The config's entry wins; a box whose config predates this section still
    gets the built-in office ladder, the same way roomfabric falls back to
    the singular ``presence.room_sensor_*`` keys. ``zones.enabled`` False
    is None for every room, the fallback included -- an off switch a
    built-in default could walk around would not be one.
    """
    if not bool(_cfg_get(cfg, "zones.enabled", True)):
        return None
    name = " ".join(str(room or "").split()).lower()
    got = zone_maps(cfg).get(name)
    if got is not None:
        return got
    return ZoneMap.office() if name == "office" else None


__all__ = ["ABSENT", "BLIND_M", "Band", "CameraOpinion", "DEFAULT_DWELL_S",
           "GATE_M", "MAX_LABEL_CHARS", "MAX_LINE_BYTES", "NO_OPINION",
           "OFFICE_BANDS", "RECORD_FIELDS", "RULE_BAND", "RULE_CAMERA",
           "RULE_EMPTY", "RULE_SILENT", "RULE_UNPLACED", "STILL_FLOOR_M",
           "Transition", "UNPLACED", "Verdict", "ZoneLog", "ZoneMap",
           "ZoneTracker", "ZoneWatcher", "default_log_path", "dwell_s",
           "log_keep", "log_max_bytes", "log_path", "verdict",
           "zone_map_for", "zone_maps"]
