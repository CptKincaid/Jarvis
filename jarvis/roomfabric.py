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

from jarvis.logs import get_logger

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

    @property
    def spoken(self) -> str:
        return self.label or self.name


def _slug(value: Any) -> str:
    text = " ".join(str(value or "").split()).lower()
    return "".join(ch if (ch.isalnum() or ch in " -_") else "" for ch in text).strip()


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
                             cfg, "presence.room_sensor_power_url", "") or "").strip())]
    out: list = []
    seen: set = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if not bool(entry.get("enabled", True)):
            continue
        url = str(entry.get("url") or "").strip()
        name = _slug(entry.get("name") or "")
        if not url or not name or name in seen:
            if url and not name:
                log.warning("roomfabric: a room entry has a url and no name; "
                            "skipped")
            elif name in seen:
                log.warning("roomfabric: two rooms are both called %r; the "
                            "second is skipped", name)
            continue
        seen.add(name)
        out.append(RoomSpec(
            name=name,
            url=url,
            label=str(entry.get("label") or name).strip() or name,
            power_url=str(entry.get("power_url") or "").strip(),
            timeout_s=_timeout(entry.get("timeout_s", default_timeout)),
            primary=bool(entry.get("primary", False)),
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
    _run: Optional[bool] = field(default=None, repr=False)
    _warned_stuck: bool = field(default=False, repr=False)

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
            return
        if self.last_answer and now - self.last_answer > self.gap_limit_s:
            self._run = None
        self.last_answer = now
        if value:
            if self._run is not True:
                self.true_since = now
            self.last_true = now
        elif self._run is not False:
            self.false_since = now
            if self.stuck:
                log.info("roomfabric: %s is reading empty again; back in the "
                         "picture", self.name)
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
        st = {"room": self.name, "value": self.value, "stuck": self.stuck}
        try:
            st.update(self.sensor.status())
        except Exception:  # noqa: BLE001 - a diagnostic must not raise
            log.debug("roomfabric: %s status failed", self.name, exc_info=True)
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
        for r in self.rooms:
            # Long enough that a couple of dropped polls are a hiccup, short
            # enough that a real outage does not preserve a stale edge.
            r.gap_limit_s = max(self.leave_hold_s, 4.0 * self.poll_s)

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
                value = r.sensor.read()
            except Exception:  # noqa: BLE001 - one room may not break the rest
                log.debug("roomfabric: %s read failed", r.name, exc_info=True)
                value = None
            r.observe(value, now)
            r.check_stuck(now, self.stuck_after_s)
        return self._resolve(now)

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
    / ``paused`` / ``status``, and its asymmetry (a room seeing somebody
    beats a sleeping phone; a room seeing nobody never beats a phone that
    answers) is already the right rule for three rooms as well as one. So
    the fabric is handed to it wearing that shape and ``RoomOrPhone`` needs
    no change whatever: read() is the house-level answer, and the None that
    means "I cannot see the whole house" lands on the module's existing
    dark-safe path.
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
