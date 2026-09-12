"""The sensor fabric: three rooms fused into the three answers he asks for.

``jarvis/roomsensor.py`` answers one question about ONE room -- "does the
room see a person right now" -- and holds no history. ``jarvis/rooms.py``
(the satellite lane) answers "may that remote sensor run, and is it
actually obeying". THIS module is the layer above both: N named rooms,
read on one fast cadence, fused.

    WHICH ROOM is he in          -> where()      "office", and how sure
    IS HE ANYWHERE in the house  -> anywhere()   True / False / None
    IS SOMEONE ELSE here         -> others()     -- refused. See below.

It owns no transport. A "room" here is a name plus ANY object with
``read() -> True | False | None``: a ``RoomSensor`` on a plain ESPHome
box, a ``rooms.Satellite`` behind a lease, or a fake in a test. That is
deliberate -- the fusion is the part that is hard to get right, and it
should not be re-derived once per transport.

WHAT AN LD2410 CAN AND CANNOT TELL YOU, because the fusion is only honest
if this is stated first.

* It reports ONE presence bit. Not a count, not an identity. A room that
  reads occupied holds one person or four, and nothing here can tell those
  apart. ``others()`` therefore returns None with a reason rather than a
  guess -- "someone else is here" needs a different sensor, and inventing
  it from radar would be the kind of confident lie that makes a house
  assistant useless.
* It sees THROUGH plasterboard (docs/room-sensor.md section 9). Two
  sensors either side of one wall can both see the SAME body, so "two
  rooms occupied" is not evidence of two people either. It is evidence of
  one person near a wall until the gates are trimmed.
* Its OFF edge is late by design: the module's own "absence delay" holds a
  target for a factory 5 s after the last sign of life. Arrival is fast,
  departure is mushy, and every timer below is built around that asymmetry
  rather than against it.

THE FOUR TIMERS, and the one job each has. All are seconds, all are
config, and the defaults are argued from the hardware:

* ``enter_hold_s`` = 2.0 -- a new room must hold occupied this long before
  it may become the ACTIVE room. A doorway pass-through at walking pace is
  about a second inside the beam, so two seconds is the line between
  "walked through" and "walked in".
* ``leave_hold_s`` = 8.0 -- the active room may read empty this long and
  still be believed. The radar's own absence delay is already 5 s and
  cannot be undercut from here; 8 s is that plus poll jitter. Anything
  shorter re-litigates the device's timer and flaps every time he leans
  out of the beam.
* ``switch_min_s`` = 6.0 -- a floor on how often the active room may change
  at all. This is the doorway anti-flap: standing in a doorway BOTH sensors
  see him, and newest-edge alone would ping-pong every poll.
* ``stale_after_s`` = 90.0 -- when the last known room stops being named.
  Between ``leave_hold_s`` and this, ``where()`` still names the room but
  says ``stale``; after it, nothing is named. It is NEVER "he is out":
  absence stays the phone's verdict on the phone's own grace.

Worst case handoff office -> kitchen: 2 s of enter hold plus up to one
poll, so about 4 s, and never more than one room change per 6 s.

WHAT ONE BAD ROOM MAY NOT DO. ``anywhere()`` returns False only when EVERY
room answered False. One room whose breaker is open makes the house
"unknown", never "empty" -- with three sensors "all empty" is a claim
about coverage we no longer have. Unknown falls through to the phone
probe, which is exactly the behaviour this box had before any sensor was
bought. A room stuck ON is the opposite risk (a pedestal fan inside the
beam is the documented failure) and would make the house occupied for
ever, so a room whose presence bit has run continuously for
``stuck_after_h`` is dropped from both answers with one warning line and
taken back the moment it reads false.

A ROOM LOCKED ON ITS OWN FURNITURE is the third case, and his office is
one: sensor -> open space -> the back of his chair -> him -> desk and
monitors -> a cement wall, with nowhere else to mount it. Measured
2026-09-11 (``jarvis/roomstill.py`` has the numbers): the presence bit read
OCCUPIED 527 of 527 samples with the flat EMPTY, so the bit carries nothing
about him in that room and the 12 h latch above would have made the house
occupied for ever. The still DISTANCE does carry him -- a body breathes
and shifts, a desk does not -- so every poll that reads occupied also reads
``Still distance`` and feeds one ``StillWindow`` per room; when the reading
has not moved 20 cm in 180 s the room is READ AS EMPTY, and the corrected
bit is what ``observe()``, ``readings()``, ``anywhere()`` and the vote all
see. The asymmetry is roomstill's: only a FIXTURE verdict may take an
occupancy away; a missing or unreadable distance keeps the sensor's word.
The window is keyed to the radar's RAW run and never to its own verdict --
otherwise the corrected False would clear the evidence that produced it
and the office would flap. ``still_check: false`` on a room's entry
switches the filter off for that room.

OFFLINE MODE (jarvis/sensing.py) governs the fabric through the SAME
policy object every sensor already asks, with one adapter in between.
``SensingPolicy.attach`` is keyed by device NAME and replaces a duplicate,
so three ``RoomSensor``s built against one policy register as ONE device
and only the last one's power is cut -- measured on 2026-09-02, not
feared. ``RoomPolicy`` namespaces the attach ("office radar", "kitchen
radar") while passing ``allowed()`` straight through, so every radar is
registered, every one is stopped, and the spoken line reads "THE OFFICE
RADAR AND THE KITCHEN RADAR are down" out of the commander's existing
joiner with no change there. A single-sensor config keeps the bare name
"radar", so the sentence he hears today does not change under him.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from jarvis import roomstill
from jarvis.logs import get_logger
from jarvis.roomsensor import STILL_ENTITY

log = get_logger("roomfabric")

DEFAULT_ROOM_NAME = "room"      # the single-sensor fallback's name
DEFAULT_POLL_S = 2.0
DEFAULT_ENTER_HOLD_S = 2.0
DEFAULT_LEAVE_HOLD_S = 8.0
DEFAULT_SWITCH_MIN_S = 6.0
DEFAULT_STALE_AFTER_S = 90.0
DEFAULT_STUCK_AFTER_H = 12.0

CERTAIN, STALE, UNKNOWN = "certain", "stale", "unknown"

# Why others() cannot answer. Said in words because the answer is None and
# a bare None reads like a bug rather than a boundary.
NO_COUNT_REASON = ("an LD2410 reports one presence bit per room, not a "
                   "count, and it sees through walls; two rooms occupied "
                   "is one person near a wall until proven otherwise")


# --------------------------------------------------------------- config
def _flag(value, default: bool = True) -> bool:
    """A boolean the way a hand-edited JSON file spells it. ``bool("false")``
    is True, and ``still_check`` is the one switch the docs tell him to type
    himself, so the usual spellings of off are honoured."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def _cfg_get(cfg, key, default=None):
    get = getattr(cfg, "get", None)
    if not callable(get):
        return default
    try:
        value = get(key, default)
    except Exception:  # noqa: BLE001 - a broken config must not cost the legs
        log.debug("roomfabric: cfg.get(%s) failed", key, exc_info=True)
        return default
    return default if value is None else value


@dataclass(frozen=True)
class RoomSpec:
    """One room's configuration. ``name`` is the identifier the rest of the
    app matches on; ``label`` is what gets spoken."""
    name: str
    url: str = ""
    label: str = ""
    power_url: str = ""
    timeout_s: float = 1.5
    primary: bool = False
    # The still-distance filter (module docstring, jarvis/roomstill.py). On
    # for every radar by his ruling of 2026-09-12; ``still_check: false``
    # on the entry switches one room off.
    still_check: bool = True

    @property
    def spoken(self) -> str:
        return self.label or self.name


def room_name(value: Any) -> str:
    """THE spelling of a room name, for every lane that reads
    ``presence.rooms``: this fabric, the satellite lease
    (``jarvis/rooms.py``) and the speaker router (``jarvis/roomaudio.py``).

    Case-folded, whitespace collapsed, punctuation dropped: a room name is
    an identifier he types into a JSON file by hand, and "Kitchen " must
    not be a second room. Until 2026-09-04 the fabric slugged and the other
    two lanes only stripped, so the same word typed once became "kitchen"
    for presence and "Kitchen" for the voice, and an announcement routed
    on the fabric's occupancy landed in the office with the reason "no
    room has an opinion" (F02). One function, imported by the other two,
    is the only version of that guarantee that cannot drift.
    ``jarvis/arrival.py`` keeps a COPY on purpose (it may import nothing
    that owns a thread) and its test pins the copy to this one.
    """
    text = " ".join(str(value or "").split()).lower()
    return "".join(ch if (ch.isalnum() or ch in " -_") else "" for ch in text).strip()


_slug = room_name      # the fabric's first name for it; app.py still says it


def room_entries(cfg) -> list:
    """``presence.rooms`` as THE list, read once and the same way for
    every lane.

    Each entry comes back as a COPY with its ``name`` normalised by
    ``room_name``. Entries that are not dicts, are switched off, have no
    name, or repeat a name are dropped HERE (the duplicate with a warning),
    so no lane can keep a room another lane dropped. Deliberately NOT
    applied here: ``presence.room_sensor_enabled`` (the fabric's master
    switch -- the speaker router does not need a radar to have a speaker)
    and the url requirement (a speaker-only room has a ``say_url`` and no
    radar). Never raises.
    """
    raw = _cfg_get(cfg, "presence.rooms", None)
    out: list = []
    seen: set = set()
    for entry in raw if isinstance(raw, (list, tuple)) else ():
        if not isinstance(entry, dict):
            continue
        if not bool(entry.get("enabled", True)):
            continue
        name = room_name(entry.get("name") or "")
        if not name:
            if entry.get("url") or entry.get("say_url"):
                log.warning("roomfabric: a room entry has a url and no name; "
                            "skipped")
            continue
        if name in seen:
            log.warning("roomfabric: two rooms are both called %r; the "
                        "second is skipped", name)
            continue
        seen.add(name)
        out.append(dict(entry, name=name))
    return out


def room_specs(cfg) -> list:
    """Every configured room, in config order.

    Shaped after ``jarvis/tools/mail.py:mail_accounts`` -- this codebase's
    existing plural: a LIST of labelled entries under ``presence.rooms``,
    and while it is absent or empty the singular ``presence.room_sensor_*``
    keys are used instead, so a config written before three rooms existed
    keeps working with no edit at all. An entry missing a url, or switched
    off, is SKIPPED rather than raised on: one unfinished room must not
    take the others down with it.

    ``presence.room_sensor_enabled`` stays the master switch over the whole
    fabric, so docs/room-sensor.md section 10 ("turning it off") remains
    true word for word with three sensors on the wall.
    """
    if not bool(_cfg_get(cfg, "presence.room_sensor_enabled", False)):
        return []
    raw = _cfg_get(cfg, "presence.rooms", None)
    default_timeout = _timeout(_cfg_get(cfg, "presence.room_sensor_timeout_s", 1.5))
    if not isinstance(raw, (list, tuple)) or not raw:
        url = str(_cfg_get(cfg, "presence.room_sensor_url", "") or "").strip()
        if not url:
            return []
        return [RoomSpec(name=DEFAULT_ROOM_NAME, url=url,
                         label=DEFAULT_ROOM_NAME, primary=True,
                         timeout_s=default_timeout,
                         power_url=str(_cfg_get(
                             cfg, "presence.room_sensor_power_url", "") or "").strip(),
                         still_check=_flag(_cfg_get(
                             cfg, "presence.room_sensor_still_check", None), True))]
    out: list = []
    for entry in room_entries(cfg):
        url = str(entry.get("url") or "").strip()
        if not url:
            continue      # unfinished, or a speaker-only room (roomaudio's)
        name = entry["name"]
        out.append(RoomSpec(
            name=name,
            url=url,
            label=str(entry.get("label") or name).strip() or name,
            power_url=str(entry.get("power_url") or "").strip(),
            timeout_s=_timeout(entry.get("timeout_s", default_timeout)),
            primary=bool(entry.get("primary", False)),
            still_check=_flag(entry.get("still_check"), True),
        ))
    if out and not any(r.primary for r in out):
        # The room the Spark is in, where he is by default and whose speaker
        # is already the right one. The first wins when he did not say.
        out[0] = replace(out[0], primary=True)
    return out


def _timeout(value, default: float = 1.5) -> float:
    try:
        return max(0.1, float(value))
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------- the adapter
class RoomPolicy:
    """One room's view of the sensing policy: ``allowed`` straight through,
    ``attach`` under a room-specific NAME.

    ``SensingPolicy.attach`` drops any device already registered under the
    same name, so three ``RoomSensor``s handed the same policy collapse to
    one and offline mode cuts the power of only the last one built. That is
    a privacy hole, not a tidiness problem, and this small shim is the
    whole fix -- no edit to jarvis/sensing.py or jarvis/roomsensor.py,
    which the offline-mode lane owns.

    The name is ``"<label> radar"`` and not ``"radar:<name>"`` because the
    commander SPEAKS these words: ``_sensing_join`` turns them into "the
    office radar and the kitchen radar", which is a sentence. A colon is
    not.
    """

    def __init__(self, policy, label: str, plural: bool = True):
        self._policy = policy
        self.label = label
        self.plural = plural

    def device_name(self, kind: str) -> str:
        # A single-sensor install keeps the bare "radar" so the sentence he
        # hears today does not change under him.
        return "%s %s" % (self.label, kind) if self.plural else str(kind)

    def allowed(self, kind) -> bool:
        return bool(self._policy.allowed(kind))

    def attach(self, name, stop, present=None, resume=None) -> None:
        attach = getattr(self._policy, "attach", None)
        if callable(attach):
            attach(self.device_name(name), stop, present, resume)

    def __getattr__(self, item):        # state(), status(), disable()...
        return getattr(self._policy, item)


# ------------------------------------------------------------- one room
@dataclass
class Room:
    """A spec, its reader, and the little history the fusion needs.

    The reader itself still holds none: this is the only object that knows
    an edge happened, which keeps ``roomsensor.py`` the one-question module
    its docstring promises."""
    spec: RoomSpec
    sensor: Any
    value: Optional[bool] = None
    true_since: float = 0.0        # when the current run of True began
    false_since: float = 0.0       # when the current run of False began
    last_true: float = 0.0         # the last moment it saw anybody
    last_answer: float = 0.0       # the last moment it had an opinion at all
    stuck: bool = False
    gap_limit_s: float = 30.0      # a silence longer than this ends the run
    # THE WALK-OUT INSTRUMENT. On 2026-09-06 the kitchen produced no room
    # line at all while he walked out of the flat, and the log could not
    # tell "the kitchen never saw him" from "we never got an answer from
    # the kitchen" -- opposite diagnoses that looked identical. These are
    # what separate them, and they are counts, not durations, because at a
    # 2.0 s poll a one-poll run measures 0.0 s and inventing 0.8 would be
    # a number with no source.
    true_polls: int = 0            # consecutive True polls in the run
    none_polls: int = 0            # cumulative unanswered polls
    none_run: int = 0              # consecutive unanswered polls right now
    none_since: float = 0.0        # when the current silence began
    was_active: bool = False       # did THIS run ever become the active room
    glimpse: Optional[tuple] = None   # (polls, seconds) pending a log line
    gap_said: bool = False
    # THE STILL-DISTANCE FILTER (module docstring). ``raw`` is what the
    # radar's bit said; ``value`` is that bit after the window has had its
    # say. ``still`` is None for a room switched off in config.
    raw: Optional[bool] = None
    still: Any = None
    still_verdict: str = "unknown"
    still_blind: bool = False          # raw on, no readable distance for a window
    still_last_read: float = 0.0       # when the window last got a reading
    fixture_run: bool = False          # this poll proved the whole run a fixture
    _raw_run: Optional[bool] = field(default=None, repr=False)
    _run_last_true_before: float = field(default=0.0, repr=False)
    _run_had_person: bool = field(default=False, repr=False)
    _run: Optional[bool] = field(default=None, repr=False)
    _warned_stuck: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        # Only a reader that can give a distance gets a window; the satellite
        # lane's readers cannot, and a window that can never fill would only
        # ever say "blind".
        if self.still is None and getattr(self.spec, "still_check", True) \
                and callable(getattr(self.sensor, "read_distance", None)):
            self.still = roomstill.StillWindow()

    @property
    def name(self) -> str:
        return self.spec.name

    def observe(self, value: Optional[bool], now: float) -> None:
        """One reading. ``None`` is no opinion and NOT a transition.

        ``_run`` is the last DEFINITE reading and is what the edge clocks
        key off, so a two-poll network hiccup does not reset the enter hold
        and make him "arrive" in the room he never left. A silence longer
        than ``gap_limit_s`` does end the run, though: an ESP32 that was
        unreachable for an afternoon is not still holding the morning's
        edge, and letting it would hand the stuck detector a thirteen-hour
        run that never happened.
        """
        if value is None:
            self.value = None
            self.none_polls += 1
            self.none_run += 1
            if not self.none_since:
                self.none_since = now
            return
        # A DEFINITE answer ends any silence. ``none_run`` is what the
        # unanswered line counts and ``gap_said`` is what keeps it to one
        # line per silence rather than one per poll.
        self.none_since, self.none_run, self.gap_said = 0.0, 0, False
        if self.last_answer and now - self.last_answer > self.gap_limit_s:
            # The run is ended by the SILENCE, not by an edge, so no
            # glimpse is recorded: we do not know what happened in it.
            self._run = None
            self.true_polls = 0
        self.last_answer = now
        if value:
            if self._run is not True:
                self.true_since = now
                self.true_polls = 1
                self.was_active = False
            else:
                self.true_polls += 1
            self.last_true = now
        else:
            if self._run is True and not self.was_active:
                # A TRUE RUN THAT ENDED WITHOUT EVER BECOMING THE ACTIVE
                # ROOM -- which is what a pass-through looks like. Measured
                # honestly: last_true - true_since, so a single poll is
                # 0.0 s and says so.
                self.glimpse = (self.true_polls,
                                max(0.0, self.last_true - self.true_since))
            self.true_polls = 0
            if self._run is not False:
                self.false_since = now
                if self.stuck:
                    log.info("roomfabric: %s is reading empty again; back in "
                             "the picture", self.name)
                self.stuck, self._warned_stuck = False, False
        self._run = self.value = value

    def check_stuck(self, now: float, after_s: float) -> None:
        if self.value is True and self.true_since and \
                now - self.true_since >= after_s and not self.stuck:
            self.stuck = True
            if not self._warned_stuck:
                log.warning("roomfabric: %s has read occupied for %.0f h "
                            "without a break; ignoring it until it clears (a "
                            "fan or a curtain inside the beam is the usual "
                            "cause -- see docs/room-sensor.md section 9)",
                            self.name, (now - self.true_since) / 3600.0)
                self._warned_stuck = True

    def status(self) -> dict:
        st: dict = {}
        try:
            st.update(self.sensor.status())
        except Exception:  # noqa: BLE001 - a diagnostic must not raise
            log.debug("roomfabric: %s status failed", self.name, exc_info=True)
        # The room's own truth goes LAST: RoomSensor.status() carries a
        # ``value`` of its own (the raw bit) and must not overwrite the
        # corrected one.
        st.update({"room": self.name, "value": self.value, "raw": self.raw,
                   "stuck": self.stuck})
        if self.still is not None:
            try:
                st["still"] = {"verdict": self.still_verdict,
                               "spread_cm": self.still.spread(),
                               "samples": self.still.samples,
                               "blind": self.still_blind,
                               "corrected": self.raw is True and self.value is False}
            except Exception:  # noqa: BLE001 - a diagnostic must not raise
                log.debug("roomfabric: %s still status failed", self.name,
                          exc_info=True)
        return st


@dataclass(frozen=True)
class Where:
    """The fused answer to "which room". ``room`` is "" when nothing is
    known; ``confidence`` is certain / stale / unknown."""
    room: str = ""
    label: str = ""
    confidence: str = UNKNOWN
    since: float = 0.0
    occupied: tuple = ()      # every room reading True right now
    unknown: tuple = ()       # every room with no opinion (down, or blocked)

    @property
    def known(self) -> bool:
        return bool(self.room)


# ---------------------------------------------------------- the fabric
class RoomFabric:
    """N rooms, one tick, three answers. Thread-safe; owns no thread of its
    own unless ``start()`` is called."""

    def __init__(self, rooms: list, now: Callable[[], float] = time.monotonic,
                 enter_hold_s: float = DEFAULT_ENTER_HOLD_S,
                 leave_hold_s: float = DEFAULT_LEAVE_HOLD_S,
                 switch_min_s: float = DEFAULT_SWITCH_MIN_S,
                 stale_after_s: float = DEFAULT_STALE_AFTER_S,
                 stuck_after_h: float = DEFAULT_STUCK_AFTER_H,
                 poll_s: float = DEFAULT_POLL_S,
                 publish: Optional[Callable] = None):
        self.rooms = list(rooms)
        self._now = now
        self.enter_hold_s = max(0.0, float(enter_hold_s))
        self.leave_hold_s = max(0.0, float(leave_hold_s))
        self.switch_min_s = max(0.0, float(switch_min_s))
        self.stale_after_s = max(0.0, float(stale_after_s))
        self.stuck_after_s = max(60.0, float(stuck_after_h) * 3600.0)
        self.poll_s = max(0.25, float(poll_s))
        self._publish = publish
        self._lock = threading.RLock()
        self._active = ""
        self._active_since = 0.0
        self._switched_at = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # THE BOOT RACE, MEASURED. The sentinel's first tick beat the
        # fabric's first stored reading by 1.4 s on 2026-09-06, and the
        # verdict that came out of it was a false AWAY on a man standing in
        # the flat: Room.value starts None, so the rooms leg read
        # UNREACHABLE and cell 24 is away. ``polls`` and ``ready`` let the
        # voter WAIT, bounded, for one honest answer instead of voting on a
        # blank.
        self.polls = 0
        self.glimpses = 0
        self.unanswered = 0
        self._ready = threading.Event()
        for r in self.rooms:
            # Long enough that a couple of dropped polls are a hiccup, short
            # enough that a real outage does not preserve a stale edge.
            r.gap_limit_s = max(self.leave_hold_s, 4.0 * self.poll_s)
        # The still window is 180 s of wall clock with a 50-reading density
        # floor: above 3.6 s a poll it can never fill and the filter would be
        # silently inert. Say so once rather than let the ghost stand quietly.
        if any(r.still is not None for r in self.rooms) and \
                roomstill.WINDOW_S / self.poll_s < roomstill.MIN_SAMPLES:
            log.warning("roomfabric: presence.rooms_poll_s is %.1f s, so the "
                        "still-distance window (%.0f s) can hold at most %d "
                        "readings against a floor of %d -- the fixture check "
                        "can never judge at this cadence; poll every %.1f s "
                        "or faster", self.poll_s, roomstill.WINDOW_S,
                        int(roomstill.WINDOW_S / self.poll_s) + 1,
                        roomstill.MIN_SAMPLES,
                        roomstill.WINDOW_S / roomstill.MIN_SAMPLES)

    # ------------------------------------------------------------- names
    def __len__(self) -> int:
        return len(self.rooms)

    @property
    def configured(self) -> bool:
        return any(getattr(r.sensor, "configured", True) for r in self.rooms)

    def room(self, name: str):
        for r in self.rooms:
            if r.name == name:
                return r
        return None

    # -------------------------------------------------------------- tick
    def tick(self) -> Where:
        """Read every room once and re-resolve. One GET per room; a room
        whose breaker is open or whose radar is offline costs nothing."""
        now = self._now()
        for r in self.rooms:
            try:
                raw = r.sensor.read()
            except Exception:  # noqa: BLE001 - one room may not break the rest
                log.debug("roomfabric: %s read failed", r.name, exc_info=True)
                raw = None
            value = self._still_filter(r, raw, now)
            r.observe(value, now)
            if r.fixture_run:
                # The run that just ended was the furniture's from its
                # first poll; it was no pass-through, so no glimpse line
                # and no RoomGlimpsed for it.
                r.glimpse, r.fixture_run = None, False
            r.check_stuck(now, self.stuck_after_s)
            self._note_edges(r, now)
        self.polls += 1
        if self.ready:
            self._ready.set()
        return self._resolve(now)

    # ------------------------------------------ the still-distance filter
    def _still_filter(self, room, raw: Optional[bool], now: float) -> Optional[bool]:
        """The radar's bit after its still distance has had its say (module
        docstring; the numbers are jarvis/roomstill.py's). Never raises: the
        distance is a second GET, and its failure costs the room nothing but
        the correction."""
        room.raw = raw
        win = room.still
        if win is None:
            return raw
        if raw is None:
            return None                  # no opinion is not evidence either way
        if raw is False:
            # THE RAW RUN ENDED. Only the sensor's own word ends a run --
            # never the FIXTURE verdict, or the corrected False would erase
            # the evidence that produced it and the room would flap.
            win.clear()
            room.still_verdict = roomstill.UNKNOWN
            room.still_blind = False
            room._raw_run = False
            return False
        # raw is True. The silence rule is observe()'s, word for word: a gap
        # past the limit ended the run, so the old readings go with it.
        started = room._raw_run is not True
        if room.last_answer and now - room.last_answer > room.gap_limit_s:
            win.clear()
            room.still_verdict = roomstill.UNKNOWN
            room.still_blind = False
            started = True
        room._raw_run = True
        if started:
            # Remember what the hint knew BEFORE this run: if the run turns
            # out to be the furniture's from its first poll, its True
            # readings were never a sighting (last_seen_room's 2026-09-06
            # stuck-room lesson, applied to its twin).
            room._run_last_true_before = room.last_true
            room._run_had_person = False
            room.still_last_read = now
        cm = self._still_cm(room)
        if cm is not None:
            win.add(cm, now)
            room.still_last_read = now
            if room.still_blind:
                room.still_blind = False
                log.info("roomfabric: %s: the still distance can see again; "
                         "the fixture check is back", room.name)
        elif not room.still_blind and now - room.still_last_read > win.window_s:
            room.still_blind = True
            log.info("roomfabric: %s: no still distance for %.0f s; the "
                     "fixture check is blind, reading the bit as-is",
                     room.name, now - room.still_last_read)
        # Age the window by the clock whether or not a reading came: a dead
        # distance entity must not freeze the last verdict for ever.
        win.expire(now)
        verdict = win.verdict()
        if verdict == roomstill.PERSON:
            room._run_had_person = True
        if verdict != room.still_verdict:
            self._note_verdict(room, verdict, win)
            if verdict == roomstill.FIXTURE and not room._run_had_person:
                room.last_true = room._run_last_true_before
                room.fixture_run = True
            room.still_verdict = verdict
        return roomstill.occupancy_is_real(verdict, True)

    @staticmethod
    def _still_cm(room) -> Optional[float]:
        """``Still distance`` in CENTIMETRES, or None. ``read_distance``
        reports METRES; the window was calibrated in cm on his recordings,
        and an unconverted 0.48 m would read as "under 20" and make the man
        at his desk a fixture. A reader with no distance (the satellite
        lane's) simply never gets a verdict."""
        read = getattr(room.sensor, "read_distance", None)
        if not callable(read):
            return None
        try:
            metres = read(STILL_ENTITY)
        except Exception:  # noqa: BLE001 - a second GET may not cost the first
            log.debug("roomfabric: %s still distance failed", room.name,
                      exc_info=True)
            return None
        if metres is None:
            return None
        try:
            return float(metres) * 100.0
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _note_verdict(room, verdict: str, win) -> None:
        """One line when a room becomes a fixture and one when a BODY moves
        in it again. UNKNOWN -> PERSON is every ordinary arrival and says
        nothing, and FIXTURE -> UNKNOWN (the evidence aged out or the run
        ended) is not "moving again" -- that sentence is only true of a
        PERSON verdict."""
        spread = win.spread()
        if verdict == roomstill.FIXTURE:
            log.info("roomfabric: %s reads occupied but its still distance has "
                     "moved only %.0f cm in %.0f s (%d readings) -- a fixture, "
                     "not a body; reading it empty",
                     room.name, spread if spread is not None else 0.0,
                     win.window_s, win.samples)
        elif verdict == roomstill.PERSON and room.still_verdict == roomstill.FIXTURE:
            log.info("roomfabric: %s: the still distance is moving again "
                     "(%.0f cm in %.0f s) -- a body; reading it occupied",
                     room.name, spread if spread is not None else 0.0,
                     win.window_s)

    # -------------------------------------------------- the privacy switch
    @property
    def blocked(self) -> str:
        """Why the WHOLE house cannot be sensed right now ("" = it can be).

        Every configured room has to be blocked before the house is: one
        radar still allowed to look is still a leg, and reporting the house
        blind while a room can see would throw away the only evidence there
        is. The reason returned is the first room's own word ("offline",
        "policy", "stopped"), because presence.py logs it and the rooms are
        all blocked by the same policy in practice.

        IT LIVES HERE, NOT ONLY ON ``HouseView``, BECAUSE THE VOTER NEEDS
        IT. ``ThreeLegProbe`` holds the fabric and not the view, and on
        2026-09-06 at 15:10:57 that gap cost a false away: "camera off for
        ten minutes" switched the radars off with the lens, the rooms leg
        went UNREACHABLE, and the voter read a leg HE had switched off as a
        leg answering "no". One rule, one implementation, two readers.
        """
        rooms = self.rooms
        if not rooms:
            return ""
        reasons = [str(getattr(r.sensor, "blocked", "") or "") for r in rooms]
        return reasons[0] if all(reasons) else ""

    # ------------------------------------------------- the boot readiness
    @property
    def ready(self) -> bool:
        """Has every configured room answered at least once? An
        unconfigured fabric is trivially ready: there is nothing to wait
        for, and a wait that never ends is worse than a blank vote."""
        with self._lock:
            rooms = [r for r in self.rooms
                     if getattr(r.sensor, "configured", True)]
            return all(r.last_answer > 0.0 for r in rooms) if rooms else True

    def wait_ready(self, timeout_s: float) -> bool:
        """Block, BOUNDED, until every room has answered once. True if it
        did. Called on the presence daemon thread and nowhere else -- the
        fabric's own thread is what sets the event, so there is no cycle
        and no way for this to outlive its timeout."""
        if self.ready:
            return True
        try:
            return bool(self._ready.wait(max(0.0, float(timeout_s))))
        except (TypeError, ValueError):
            return self.ready

    # ------------------------------------------------- the walk-out lines
    def _note_edges(self, room, now: float) -> None:
        """THE TWO LINES THAT WOULD HAVE ANSWERED 2026-09-06 IN THE LOG.

        A GLIMPSE -- a True run that ended without ever becoming the active
        room. That is what a pass-through looks like, and it is invisible
        today: ``RoomChanged`` is published only after the enter hold, so
        his walk to the front door left no trace at all and the departure
        sequence never left the desk.

        AN UNANSWERED ROOM -- polls that came back None for longer than the
        gap limit. "The kitchen never saw him" and "we never got an answer
        from the kitchen" are OPPOSITE diagnoses and they looked identical
        in his log. Never raises: a log line is not worth a poll.
        """
        pending, room.glimpse = room.glimpse, None
        if pending is not None:
            polls, run_s = pending
            self.glimpses += 1
            log.info("roomfabric: %s glimpsed -- %d poll%s occupied, %.1f s "
                     "measured (poll %.1f s), under the %.1f s enter hold; "
                     "not the active room", room.name, polls,
                     "" if polls == 1 else "s", run_s, self.poll_s,
                     self.enter_hold_s)
            self._publish_glimpse(room, polls, run_s)
        if room.none_since and not room.gap_said and \
                now - room.none_since >= room.gap_limit_s:
            room.gap_said = True
            self.unanswered += 1
            log.info("roomfabric: %s has not answered for %.0f s (%d polls); "
                     "we do not know whether anybody is in it, which is not "
                     "the same as the room seeing nobody",
                     room.name, now - room.none_since, room.none_run)

    def _publish_glimpse(self, room, polls: int, run_s: float) -> None:
        if self._publish is None:
            return
        try:
            from jarvis.events import RoomGlimpsed
            self._publish(RoomGlimpsed(
                room=room.name, label=room.spec.spoken, polls=int(polls),
                run_s=float(run_s), hold_s=float(self.enter_hold_s),
                poll_s=float(self.poll_s), at=time.time()))
        except Exception:  # noqa: BLE001 - the bus must not break the poll
            log.debug("roomfabric: glimpse publish failed", exc_info=True)

    # -------------------------------------------------------- the fusion
    def _candidates(self, now: float) -> list:
        return [r for r in self.rooms
                if r.value is True and not r.stuck
                and now - r.true_since >= self.enter_hold_s]

    def _rank(self, room) -> tuple:
        # Newest ON edge first: the room he walked INTO is the room whose
        # bit turned over most recently. Then the primary room, which is a
        # tie-break and nothing more -- two rooms cannot share an edge time
        # unless they were observed in the same tick.
        return (room.true_since, 1 if room.spec.primary else 0)

    def _resolve(self, now: float) -> Where:
        with self._lock:
            cands = self._candidates(now)
            cur = self.room(self._active) if self._active else None
            if cur is not None and cur.stuck:
                # The active room turned out to be a fan. Drop it BEFORE the
                # switch floor is consulted, or the floor would defend a
                # room we have just decided is not a person -- and the stale
                # clock never fires, because last_true keeps moving.
                log.info("roomfabric: %s is stuck on; no longer the active "
                         "room", cur.name)
                self._active, self._active_since, cur = "", now, None
            if cands:
                best = max(cands, key=self._rank)
                if best.name != self._active:
                    held = self._active and \
                        (now - self._switched_at) < self.switch_min_s
                    if not held:
                        self._set_active(best, now)
            elif cur is not None and cur.last_true and \
                    now - cur.last_true >= self.stale_after_s:
                # Nothing sees him and the last sighting has gone cold. This
                # is never "he is out" -- that is the phone's verdict on its
                # own grace -- only "I no longer know the room".
                log.debug("roomfabric: no room has seen anyone since %.0fs; "
                          "dropping %s", now - cur.last_true, cur.name)
                self._active, self._active_since = "", now
            return self._where(now)

    def _set_active(self, room, now: float) -> None:
        # The run has become the active room, so its end is an ordinary
        # departure from a room and never a glimpse.
        room.was_active = True
        previous, self._active = self._active, room.name
        self._active_since = self._switched_at = now
        log.info("roomfabric: %s%s", room.name,
                 " (from %s)" % previous if previous else "")
        if self._publish is not None:
            try:
                from jarvis.events import RoomChanged
                self._publish(RoomChanged(room=room.name, label=room.spec.spoken,
                                          previous=previous, at=time.time()))
            except Exception:  # noqa: BLE001 - the bus must not break the poll
                log.debug("roomfabric: publish failed", exc_info=True)

    def _where(self, now: float) -> Where:
        occupied = tuple(r.name for r in self.rooms
                         if r.value is True and not r.stuck)
        unknown = tuple(r.name for r in self.rooms if r.value is None)
        cur = self.room(self._active) if self._active else None
        if cur is None or cur.stuck:
            return Where(occupied=occupied, unknown=unknown)
        if cur.value is True:
            conf = CERTAIN
        elif cur.value is False and now - cur.false_since < self.leave_hold_s:
            # Inside the leave hold the room is still believed: the radar's
            # own 5 s absence delay means a false here is as likely to be him
            # leaning out of the beam as him leaving the room.
            conf = CERTAIN
        else:
            conf = STALE
        return Where(room=cur.name, label=cur.spec.spoken, confidence=conf,
                     since=self._active_since, occupied=occupied,
                     unknown=unknown)

    # -------------------------------------------------------- the answers
    def where(self) -> Where:
        """Which room, and how sure. Does not read the sensors: call
        ``tick()`` (or ``start()``) for that."""
        with self._lock:
            return self._where(self._now())

    def last_seen_room(self, skip=()) -> tuple:
        """(room name, seconds since it last saw anybody) -- ("", None) if
        no room has ever seen anyone.

        THE SOURCE FOR THE BEDROOM HINT, and deliberately not
        ``where().room``. The active room is dropped after
        ``stale_after_s`` (90 s) measured from its last_true, and with the
        office's own 10 s absence delay on top the active room survives
        about 100 seconds after he leaves it. Every bedroom trip that
        matters is longer than that, so ``where()`` would hand back "" at
        exactly the moment the hint is needed.

        ``Room.last_true`` has none of that: it is per room, stamped on
        every tick that sees anybody, never cleared while the process
        lives, and unbounded in age. Zero new state, zero new polling.
        See ``presencevote.bedroom_split`` for what the hint buys.

        A STUCK OR FAULTED ROOM IS NOT A HINT, and until 2026-09-06 it was.
        At 14:05:47.969 cell 12 printed "the last room to see anybody was
        the office ... so he never left the office" at the exact instant
        the rooms leg had computed CLEAR *because* the office was faulted:
        this method took ``max(last_true)`` over EVERY room, and a latched
        office's ``last_true`` is always about zero seconds old. One
        verdict, two contradictory readings of one sensor. ``skip`` is the
        caller's own faulted set -- the stuck DETECTOR's, which the fabric
        cannot see -- on top of the fabric's own ``stuck``.
        """
        drop = {_slug(r) for r in (skip or ())}
        with self._lock:
            now = self._now()
            live = [r for r in self.rooms
                    if r.last_true and not r.stuck
                    and _slug(r.name) not in drop]
            best = max(live, key=lambda r: r.last_true, default=None)
            if best is None:
                return ("", None)
            return (best.name, now - best.last_true)

    def readings(self) -> dict:
        """``{room: True | False | None}`` as of the last tick -- the radar's
        bit after the still-distance filter (``Room.raw`` keeps the bit
        itself), and otherwise RAW.

        Stuck rooms are NOT filtered here. There is exactly one place that
        drops a room -- ``presencevote.rooms_leg(faulted=...)`` -- and two
        dropping mechanisms is how one of them ends up forgotten. Use
        ``stuck_rooms()`` for the fabric's own half of that set.
        """
        with self._lock:
            return {r.name: r.value for r in self.rooms}

    def stuck_rooms(self) -> frozenset:
        """The rooms the fabric's own duration detector has given up on."""
        with self._lock:
            return frozenset(r.name for r in self.rooms if r.stuck)

    def anywhere(self) -> Optional[bool]:
        """Is anyone in the house? True / False / None.

        FALSE ONLY WHEN EVERY ROOM ANSWERED. With one sensor "empty" was a
        statement about the one room we could see; with three it is a
        statement about coverage, and a room whose breaker is open means we
        do not have it. So a single unknown room makes the whole answer
        None, which falls through to the phone probe -- exactly the
        behaviour this box had before any sensor was bought.
        """
        with self._lock:
            live = [r for r in self.rooms
                    if getattr(r.sensor, "configured", True) and not r.stuck]
            if not live:
                return None
            if any(r.value is True for r in live):
                return True
            if all(r.value is False for r in live):
                return False
            return None

    def others(self) -> Optional[bool]:
        """Is someone here who is not him? **Radar cannot answer this.**

        Always None, on purpose, and ``others_reason`` says why in words a
        spoken line can use. The bit per room carries no count and no
        identity, and because the beam goes through plasterboard even "two
        rooms at once" is not two people. The legs that CAN answer it are
        named in the design: an ARP roster of known phones (works today, no
        new hardware), the speaker gate on a voice, and the face
        recognition he wants later.
        """
        return None

    @property
    def others_reason(self) -> str:
        return NO_COUNT_REASON

    def status(self) -> dict:
        w = self.where()
        return {"active": w.room, "confidence": w.confidence,
                "occupied": list(w.occupied), "unknown": list(w.unknown),
                "anywhere": self.anywhere(),
                "rooms": [r.status() for r in self.rooms]}

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.configured:
            log.info("roomfabric: no room sensor configured; fabric idle")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="roomfabric",
                                        daemon=True)
        self._thread.start()
        log.info("roomfabric: watching %s every %.1fs",
                 ", ".join(r.name for r in self.rooms), self.poll_s)

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("roomfabric: tick failed")
            self._stop.wait(self.poll_s)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------- the presence leg
    def house_view(self):
        return HouseView(self)


class HouseView:
    """The fabric wearing ``RoomSensor``'s interface.

    ``jarvis/presence.py`` composes ONE object with ``read`` / ``configured``
    / ``paused`` / ``blocked`` / ``status``, and its asymmetry (a room
    seeing somebody beats a sleeping phone; a room seeing nobody never
    beats a phone that answers) is already the right rule for three rooms
    as well as one. So the fabric is handed to it wearing that shape and
    ``RoomOrPhone`` needs no change whatever: read() is the house-level
    answer, and the None that means "I cannot see the whole house" lands on
    the module's existing dark-safe path.

    **``blocked`` IS PART OF THAT SHAPE AND WAS MISSING.** It shipped
    without one on 2026-09-03 and the cost was silent: presence's
    ``_blacked_out()`` reads ``sensor.blocked`` inside a broad ``except``,
    so a leg without the attribute made offline mode look like "nothing is
    blocked" and the sentinel froze on its last verdict for the length of
    the blackout instead of going to "unknown". On a rooms-only box (no
    phone leg -- his) that IS the dark-safe path. Anything else wearing
    this interface owes the same property; the pin is
    tests/test_sensing.py::test_a_rooms_only_box_goes_unknown_when_sensing_is_off.
    """

    def __init__(self, fabric: RoomFabric):
        self.fabric = fabric

    @property
    def url(self) -> str:
        return ", ".join(r.spec.url for r in self.fabric.rooms if r.spec.url)

    @property
    def configured(self) -> bool:
        return self.fabric.configured

    @property
    def paused(self) -> bool:
        return all(getattr(r.sensor, "paused", False) for r in self.fabric.rooms) \
            if self.fabric.rooms else True

    @property
    def blocked(self) -> str:
        """Why the WHOLE house cannot be sensed right now ("" = it can be).

        ONE RULE, ONE IMPLEMENTATION: see ``RoomFabric.blocked``, which the
        three-leg voter also reads. This wrapper exists because ``blocked``
        is part of the ``RoomSensor`` shape presence.py composes against,
        and it shipped without one on 2026-09-03 -- see the class docstring
        for what that silence cost.
        """
        return self.fabric.blocked

    @property
    def last_value(self) -> Optional[bool]:
        return self.fabric.anywhere()

    def read(self) -> Optional[bool]:
        """The house's answer.

        The fabric's own thread polls every two seconds because a room
        handoff has to be quick; the presence sentinel polls every sixty
        because a phone has nothing to say in a hurry. When the fast loop is
        running this returns what it last saw rather than reading the
        sensors AGAIN on the sentinel's thread -- two callers polling one
        RoomSensor would double the traffic and race the breaker's counters
        for nothing. With no thread (a test, or a build that only wants the
        presence leg) it reads.
        """
        if not self.fabric.running:
            self.fabric.tick()
        return self.fabric.anywhere()

    def status(self) -> dict:
        return self.fabric.status()


# ------------------------------------------------------------- building
def build(cfg, policy=None, sensor_factory: Optional[Callable] = None,
          publish: Optional[Callable] = None) -> Optional[RoomFabric]:
    """The fabric from config, or None when no room is configured.

    Never raises and never polls: a bad entry costs its own room and
    nothing else, exactly as ``presence._make_sensor`` promises for one.
    ``sensor_factory(spec, room_policy)`` is the transport seam -- the
    default builds a ``RoomSensor``, and the satellite lane's ``Satellite``
    drops in through the same hole (see ``from_readers``).
    """
    specs = room_specs(cfg)
    if not specs:
        return None
    if sensor_factory is None:
        sensor_factory = _default_factory
    plural = len(specs) > 1
    rooms: list = []
    for spec in specs:
        rp = RoomPolicy(policy, spec.spoken, plural) if policy is not None else None
        try:
            sensor = sensor_factory(spec, rp)
        except Exception:  # noqa: BLE001 - one room may not cost the others
            log.exception("roomfabric: %s could not be built", spec.name)
            continue
        if not getattr(sensor, "configured", True):
            log.warning("roomfabric: %s url %r is not an http(s) URL; skipped",
                        spec.name, spec.url)
            continue
        rooms.append(Room(spec=spec, sensor=sensor))
    if not rooms:
        return None
    log.info("roomfabric: %s", ", ".join(
        "%s %s" % (r.name, getattr(r.sensor, "url", "")) for r in rooms))
    return RoomFabric(
        rooms, publish=publish,
        enter_hold_s=_num(cfg, "presence.rooms_enter_hold_s", DEFAULT_ENTER_HOLD_S),
        leave_hold_s=_num(cfg, "presence.rooms_leave_hold_s", DEFAULT_LEAVE_HOLD_S),
        switch_min_s=_num(cfg, "presence.rooms_switch_min_s", DEFAULT_SWITCH_MIN_S),
        stale_after_s=_num(cfg, "presence.rooms_stale_after_s", DEFAULT_STALE_AFTER_S),
        stuck_after_h=_num(cfg, "presence.rooms_stuck_after_h", DEFAULT_STUCK_AFTER_H),
        poll_s=_num(cfg, "presence.rooms_poll_s", DEFAULT_POLL_S))


def _default_factory(spec: RoomSpec, room_policy):
    from jarvis import roomsensor as _rs
    kw = {}
    if room_policy is not None:
        kw["policy"] = room_policy
    if spec.power_url:
        kw["power_url"] = spec.power_url
    return _rs.RoomSensor(spec.url, timeout_s=spec.timeout_s, **kw)


def from_readers(pairs, **kw) -> Optional[RoomFabric]:
    """A fabric over readers something else already built.

    ``pairs`` is ``[(RoomSpec | name, reader), ...]`` where a reader is
    anything with ``read() -> True | False | None`` -- the satellite lane's
    ``rooms.Satellite``, a ``RoomSensor``, a stub. The fusion has no opinion
    about the wire, and there should be exactly one implementation of it.
    """
    rooms: list = []
    for i, (spec, reader) in enumerate(pairs):
        if not isinstance(spec, RoomSpec):
            spec = RoomSpec(name=_slug(spec) or "room%d" % i,
                            label=str(spec), primary=(i == 0))
        rooms.append(Room(spec=spec, sensor=reader))
    return RoomFabric(rooms, **kw) if rooms else None


def _num(cfg, key: str, default: float) -> float:
    try:
        return float(_cfg_get(cfg, key, default))
    except (TypeError, ValueError):
        return default
