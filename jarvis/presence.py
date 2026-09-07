"""Presence sentinel: is Hunter home? His phone on the Wi-Fi says so.

``probe(ip, mac, run)`` is pure: it reads ``ip -4 neigh`` and calls the
phone present when its entry is REACHABLE (or DELAY -- the kernel is
re-confirming a neighbour it heard from moments ago). Anything else --
STALE, FAILED, no entry -- means "not sure", and a single unprivileged
``ping -c1 -W1`` settles it. STALE is NEVER absence: on this box every
neighbour but the router shows STALE between conversations.

Bluetooth is deliberately not a leg: no phone is paired to the Spark and a
phone does not stay "Connected: yes" to a Linux host unless an audio or HID
profile is in use, so ``bluetoothctl info`` would say "away" all day.

A SECOND LEG, off by default: ``presence.room_sensor_url`` points at an
ESP32 + LD2410 mmWave module over plain HTTP (``jarvis/roomsensor.py``,
``docs/room-sensor.md``). The two legs compose as an OR with one asymmetry
that IS the design: **the room seeing someone is positive evidence and
beats a sleeping phone; the room seeing nobody is not absence** -- he may
be in the kitchen -- so an empty room never overrides a phone that
answers. Absence therefore remains exactly what it was, the phone's
verdict on the same grace, and the sensor can only ever make him home
sooner. With no URL configured the sentinel polls the same ``probe``
function object it always did.

N ROOMS, when ``presence.rooms`` is configured: the leg becomes
``roomfabric.HouseView`` rather than one ``RoomSensor``, and ``RoomOrPhone``
below needs no change at all -- the view wears the same
``read() -> True | False | None`` **and the same ``blocked``**, which is
what carries the offline-mode path below into the multi-room case (it
shipped without one, and ``_blacked_out`` swallowed the AttributeError);
the asymmetry above is already the right rule for three rooms as well as
one. The fabric also publishes
``RoomChanged``, which is what lets the app treat the KITCHEN as the front
door (jarvis/arrival.py, app._on_room_changed). With ``presence.rooms``
empty the singular path below runs unchanged, which is the configuration
on this box today.

OFFLINE MODE (jarvis/sensing.py) takes the radar leg away and NOTHING
else: ``RoomSensor.read`` returns None while sensing is denied, which is
the module's existing "no opinion" path, so the composition degrades to
exactly what it was before the sensor was bought -- the phone's verdict on
the same grace. The phone probe itself is not governed: it reads the
kernel's ARP table for an address he configured, and is not a sensor
pointed at the room.

On a SENSOR-ONLY box, though, "no opinion" leaves nothing at all, and the
sentinel goes to **unknown** rather than holding its last verdict. Holding
is right for a breaker outage measured in seconds; an offline-mode
blackout lasts until he speaks, and a verdict frozen that long is being
asserted, not held. A frozen "away" makes jarvis/quiet.py answer "you're
out" and swallow every proactive line while he is sitting in the room; a
frozen "home" speaks into an empty one and then never says "welcome back".
Unknown is the state both of those consumers already blank (the ambient
WHERE row prints nothing, ``is_home()`` answers True), so nothing new has
to learn about offline mode to degrade honestly.

``PresenceSentinel`` polls the probe on a daemon thread (health.Watchdog's
start / stop-with-join form so a ping in flight cannot outlive quit) and
publishes ``Presence`` on the bus at each transition. Away has hysteresis:
the phone must be unseen for ``presence.away_after_min`` before he is
"out" -- an iPhone drops off Wi-Fi power-save for minutes at a time, and a
few-minute grace would flap. Home is immediate. ``is_home()`` answers True
until the probe has ever said otherwise (a misconfigured probe must not
mute him), and the consumers (jarvis/quiet.py, the app's "Welcome back,
sir") read that.
"""
from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from typing import Callable, Optional

from jarvis.events import Presence, bus
from jarvis.logs import get_logger

log = get_logger("presence")

DEFAULT_AWAY_MIN = 12
DEFAULT_POLL_S = 60.0
DEFAULT_AWAY_POLL_S = 10.0     # see the poll_s docstring: arrival must be prompt
# How long the FIRST vote may wait for the room fabric to complete one poll.
# MEASURED NEED: on 2026-09-06 the fabric started at 10:49:02.4 and named the
# office at 10:49:04.9, while the sentinel's first tick published a verdict at
# 10:49:03.5 -- 1.4 s before the rooms had anything to say. Room.value starts
# None, so the leg read UNREACHABLE and cell 24 is AWAY. Three poll periods at
# his rooms_poll_s of 2.0 covers that race four times over, and it is bounded
# so a fabric that never answers costs six seconds ONCE and then votes without
# the rooms, honestly and out loud.
DEFAULT_BOOT_GRACE_S = 6.0
PING_TIMEOUT_S = 3.0
PRESENT_STATES = ("REACHABLE", "DELAY", "PERMANENT")
WELCOME_LINE = "Welcome back, sir."

# A leg that does not answer "are you blocked?" at all, told apart from
# one that answers "no". See PresenceSentinel._blacked_out.
_MISSING = object()

_NEIGH_RX = re.compile(
    r"^(?P<ip>\S+)\s+dev\s+(?P<dev>\S+)(?:\s+lladdr\s+(?P<mac>[0-9a-f:]+))?"
    r"(?:\s+\S+)*?\s+(?P<state>[A-Z]+)\s*$", re.I)


def parse_neigh(text: str) -> list[dict]:
    """``ip -4 neigh`` lines -> [{ip, dev, mac, state}]."""
    out = []
    for line in (text or "").splitlines():
        m = _NEIGH_RX.match(line.strip())
        if not m:
            continue
        out.append({"ip": m.group("ip"), "dev": m.group("dev"),
                    "mac": (m.group("mac") or "").lower(),
                    "state": m.group("state").upper()})
    return out


def _run_default(argv, timeout):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def default_gateway(run: Callable = _run_default) -> str:
    """The default route's next hop, or "". Used only as a canary."""
    try:
        res = run(["ip", "-4", "route", "show", "default"], timeout=PING_TIMEOUT_S)
        toks = (getattr(res, "stdout", "") or "").split()
        return toks[toks.index("via") + 1] if "via" in toks else ""
    except Exception:  # noqa: BLE001 - a canary may not cost the leg
        log.debug("presence: default route lookup failed", exc_info=True)
        return ""


def probe_state(ip: str = "", mac: str = "",
                run: Callable = _run_default) -> Optional[bool]:
    """UNKNOWN, NEVER AWAY. ``True`` / ``False`` / ``None``.

    ``probe()`` below cannot say "I could not ask" -- it returns a plain
    bool, and its own comment about ip(8) ("a missing/timed-out ip(8) is
    'unknown'") is contradicted by the line under it, which falls through
    to the ping and returns False on any exception. False means AWAY, so
    every one of these reads as "he left the flat": the Spark's own Wi-Fi
    drops, the router reboots, ip(8) or ping is missing from PATH, the
    subnet changes. That is a confident false away, which is the worse of
    the two failures -- it mutes him while he is sitting in the room.

    THE RULE HERE: distinguish the two failures concretely.

      * the ping SUBPROCESS ran and got no reply -> ``False``. A real
        "asked, and no answer".
      * it raised, timed out, or could not be launched at all -> ``None``.
      * ``ip -4 neigh`` itself failed -> ``None``, before anything else.
      * no address configured at all -> ``None``. An unconfigured leg says
        nothing about where he is; the old ``probe`` answered False here,
        which is a leg voting on a question it was never wired to see.

    THE GATEWAY CANARY, and it costs a packet only on the negative path.
    His default gateway is the neighbour that answers; if the GATEWAY does
    not reply either, the network is down, not the man, and the leg is
    ``None`` regardless of what the phone did. Measured on his box
    2026-09-05: 6 neighbour rows, states REACHABLE and STALE only, the
    gateway REACHABLE. (The brief said the gateway was the ONLY REACHABLE
    row; it is one of three. The canary does not depend on that.)

    ``PresenceSentinel.tick`` already handles a None correctly -- it holds
    everything, including the boot grace clock, and counts nothing towards
    the away grace. The plumbing to receive an honest unknown was already
    there; only the leg refused to send one.
    """
    ip, mac = (ip or "").strip(), (mac or "").strip().lower()
    if not ip and not mac:
        return None
    try:
        res = run(["ip", "-4", "neigh"], timeout=PING_TIMEOUT_S)
        rows = parse_neigh(getattr(res, "stdout", "") or "")
    except Exception:  # noqa: BLE001 - could not ask, so do not answer
        log.debug("presence: ip neigh failed", exc_info=True)
        return None
    for row in rows:
        if (ip and row["ip"] == ip) or (mac and row["mac"] == mac):
            if row["state"] in PRESENT_STATES:
                return True
            if not ip:
                ip = row["ip"]                  # MAC-only config: ping what ARP knows
    if not ip:
        # A MAC that ARP has never seen. There is nothing to ping, and
        # "no address to ping" is not evidence that he is out.
        return None
    try:
        res = run(["ping", "-c", "1", "-W", "1", ip], timeout=PING_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - TimeoutExpired, FileNotFoundError...
        log.debug("presence: ping %s could not run", ip, exc_info=True)
        return None
    if getattr(res, "returncode", 1) == 0:
        return True
    gateway = default_gateway(run)
    if not gateway:
        return None
    try:
        res = run(["ping", "-c", "1", "-W", "1", gateway], timeout=PING_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        return None
    if getattr(res, "returncode", 1) != 0:
        log.info("presence: the phone did not answer and neither did the "
                 "gateway; the network is down, not him")
        return None
    return False


def probe(ip: str = "", mac: str = "", run: Callable = _run_default) -> bool:
    """True when the phone answers. ``run(argv, timeout=)`` is the seam
    (tests pass a fake that returns an object with returncode/stdout).

    KEPT AS A BOOL for every existing caller. It flattens the honest
    unknown that ``probe_state`` can now express -- deliberately, because
    changing this function's type would change what "away" means for the
    single-leg path that is live on other boxes. The three-leg path uses
    ``probe_state``.
    """
    ip, mac = (ip or "").strip(), (mac or "").strip().lower()
    if not ip and not mac:
        return False
    try:
        res = run(["ip", "-4", "neigh"], timeout=PING_TIMEOUT_S)
        rows = parse_neigh(getattr(res, "stdout", "") or "")
    except Exception:  # noqa: BLE001 - a missing/timed-out ip(8) is "unknown"
        log.debug("presence: ip neigh failed", exc_info=True)
        rows = []
    for row in rows:
        if (ip and row["ip"] == ip) or (mac and row["mac"] == mac):
            if row["state"] in PRESENT_STATES:
                return True
            if not ip:
                ip = row["ip"]                  # MAC-only config: ping what ARP knows
    if not ip:
        return False                             # no address to ping
    try:
        res = run(["ping", "-c", "1", "-W", "1", ip], timeout=PING_TIMEOUT_S)
        return getattr(res, "returncode", 1) == 0
    except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired included
        log.debug("presence: ping %s failed", ip, exc_info=True)
        return False


def _cfg_get(cfg, key, default=None):
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            value = get(key, default)
            return default if value is None else value
        except Exception:  # noqa: BLE001
            log.debug("presence: cfg.get(%s) failed", key, exc_info=True)
    return default


def _seconds(value, default: float, floor_s: float = 60.0) -> float:
    """A window in seconds from a number, or ``default``. NaN, zero and
    negatives are nonsense and fall back; a window is floored at a minute,
    the same floor ``away_after_s`` and ``arrival.mic_silence_s`` apply."""
    try:
        s = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(s) or s <= 0.0:
        return float(default)
    return max(floor_s, s)


def _minutes(value):
    """Minutes -> seconds, or None for anything that is not a number."""
    try:
        return float(value) * 60.0
    except (TypeError, ValueError):
        return None


def _make_sensor(cfg, policy=None):
    """The room-sensor leg from config, or None. Never raises, never polls.

    Built ONCE, at construction: ``AssistantConfig.reload_if_changed`` has
    no callers, so a config edit needs a restart -- and a sensor that
    appeared mid-run would want a state re-evaluation nobody asked for.

    ``policy`` is the sensing owner (jarvis/sensing.py). It is handed to
    the SENSOR, not consulted here: offline mode has to stop the poll at
    the wire, and a check in this function would only decide whether the
    leg exists at start-up.
    """
    raw = str(_cfg_get(cfg, "presence.room_sensor_url", "") or "").strip()
    if not bool(_cfg_get(cfg, "presence.room_sensor_enabled", False)):
        if raw:
            log.info("presence: room_sensor_url is set but "
                     "presence.room_sensor_enabled is false; phone probe only")
        return None
    if not raw:
        return None
    try:
        from jarvis import roomsensor
        sensor = roomsensor.RoomSensor(
            raw, timeout_s=_cfg_get(cfg, "presence.room_sensor_timeout_s",
                                    roomsensor.DEFAULT_TIMEOUT_S),
            policy=policy,
            power_url=str(_cfg_get(cfg, "presence.room_sensor_power_url",
                                   "") or "").strip())
    except Exception:  # noqa: BLE001 - a missing module must not cost the phone leg
        log.exception("presence: room sensor could not be built")
        return None
    if not sensor.configured:
        log.warning("presence: room_sensor_url %r is not an http(s) URL; "
                    "phone probe only", raw)
        return None
    log.info("presence: room sensor %s", sensor.url)
    return sensor


def _make_fabric(cfg, policy=None):
    """The MULTI-ROOM leg (jarvis/roomfabric.py), or None. Never raises.

    Built only when ``presence.rooms`` is a non-empty list, so a config
    written for one radar takes the single-sensor path above byte for byte
    and this feature cannot regress the box that is live today.

    Two things come with it, and the second is the point. ``HouseView``
    wears ``RoomSensor``'s interface, so ``RoomOrPhone`` composes the whole
    house exactly as it composed one room -- any room seeing him beats a
    sleeping phone, no room seeing him never beats a phone that answers.
    And the fabric publishes ``RoomChanged``, which is what makes the
    KITCHEN a door sensor: the app greets on that event after a whole-home
    absence (app._on_room_changed), rather than waiting for the phone's
    radio to answer an ARP.
    """
    raw = _cfg_get(cfg, "presence.rooms", None)
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    try:
        from jarvis import roomfabric
        return roomfabric.build(cfg, policy=policy, publish=bus.publish)
    except Exception:  # noqa: BLE001 - a broken fabric may not cost the phone
        log.exception("presence: the room fabric could not be built")
        return None


class RoomOrPhone:
    """``(ip, mac) -> True | False | None`` -- the two legs, composed.

    The room is asked FIRST because it is the cheap one (a LAN GET, ~5 ms,
    against a ``ping`` that costs up to a second) and because a hit there
    ends the tick: while he is in the room the phone is never probed at
    all. Then:

    * room says someone -> ``True``, and the phone is not consulted. This
      is the arrival win: the radar sees him on the doorstep whether or
      not his phone's radio has woken up to answer an ARP.
    * room says nobody -> the phone decides, unchanged. An empty room is
      not an empty flat.
    * room has no opinion (offline, garbage, breaker open) -> the phone
      decides, unchanged. THIS is the dark-safe path, and it is why
      ``RoomSensor.read`` returns None rather than False.
    * room says nobody AND there is no phone leg to fall back to ->
      ``False``, the honest answer for a sensor-only install.
    * no opinion AND no phone leg -> ``None``, meaning "this tick knows
      nothing". ``tick`` holds the state rather than counting it towards
      the away grace: a sensor-only install whose sensor died must never
      drift into "away" and mute him.
    """

    def __init__(self, sensor, phone: Callable = probe):
        self.sensor, self.phone = sensor, phone

    def __call__(self, ip: str = "", mac: str = "") -> Optional[bool]:
        seen = None
        try:
            seen = self.sensor.read()
        except Exception:  # noqa: BLE001 - the room must not break the phone leg
            log.debug("presence: room sensor read failed", exc_info=True)
        if seen:
            return True
        if ip or mac:
            # NOT bool(): a phone leg that can say "I could not ask" must
            # be able to say it through here. ``bool()`` flattened a None
            # to False at the last line, which is the confident false away
            # this whole composition exists to avoid. A leg that still
            # returns a plain bool is unaffected.
            answer = self.phone(ip, mac)
            return None if answer is None else bool(answer)
        return False if seen is False else None


class ThreeLegProbe:
    """HIS three-leg voter, wearing the sentinel's probe shape.

    ``(ip, mac) -> True | False | None`` exactly like ``RoomOrPhone``, so it
    drops into ``PresenceSentinel`` at the seam that was always there and
    the sentinel's thread, cadence, hysteresis and event publishing are
    untouched. The mapping is the whole interface:

        HOME, BED   -> True    he is in the flat (BED is a HOME sub-state)
        AWAY        -> False   established out
        UNKNOWN     -> None    the sentinel HOLDS: it does not count a None
                               towards the away grace, and it does not
                               start the boot grace clock on one

    THE GRACE IS THE SENTINEL'S, NOT THE VOTER'S. ``phone_leg`` is called
    with ``grace_s=0.0`` on purpose. ``away_after_min`` (12 min) is applied
    once, by ``PresenceSentinel.tick``, against ``last_seen``; if the voter
    ALSO held a grace they would stack and "away" would take 24 minutes --
    twelve of which he would spend standing in his hallway un-greeted. So
    the voter answers "right now", and the sentinel decides how long a
    "no" has to persist before it means anything. ``grace_s`` is exposed
    so a test can pin that.

    WHY THE PHONE IS ASKED EVERY TICK NOW, and this is the fix for the
    proximate cause of 2026-09-05. ``RoomOrPhone.__call__`` opens with
    ``if seen: return True`` -- while ANY room reads occupied the phone is
    never probed at all. So the latched office did not merely outvote the
    phone, it PREVENTED THE PHONE FROM VOTING. Had the phone been asked it
    would have said "no answer", the grace would have expired at ~20:35:10
    and the sentinel would have been strictly away when he opened the door
    at 20:43:13. Here all three legs are read on every tick and the vote is
    taken over all three. The cost is one ARP read plus at most one ping
    per poll -- 60 s while he is home.

    THE FOURTH INPUT, for cell 6 only. ``mic()`` returns SECONDS SINCE THE
    MICROPHONE LAST COMPLETED A TURN -- ``TurnLedger.idle_s()``, the number
    ``arrival.departure_ready`` already vetoes on -- or None. It is a
    number, never audio, and ``_mic_leg`` below is pinned to stay that way.
    ``recency_s`` is how fresh an agreement with an occupied run has to be
    for that run to still count as him (``presence.corroboration_recency_min``);
    ``mic_window_s`` is how long a turn goes on proving he is in the flat
    (``presence.departure_mic_silence_min`` -- the departure veto's own
    number, reused rather than re-invented).
    """

    def __init__(self, fabric=None, cfg=None, stuck=None,
                 phone: Callable = probe_state, eye: Optional[Callable] = None,
                 mic: Optional[Callable] = None,
                 desk: Optional[Callable] = None,
                 desk_age: Optional[Callable] = None,
                 look: Optional[Callable] = None,
                 away_s: Optional[Callable] = None,
                 sensing_off: Optional[Callable] = None,
                 door_room: str = "kitchen", desk_room: str = "office",
                 recency_s: Optional[float] = None,
                 mic_window_s: Optional[float] = None,
                 desk_window_s: Optional[float] = None,
                 grace_s: Optional[float] = None,
                 boot_grace_s: Optional[float] = None,
                 now: Callable[[], float] = time.monotonic):
        from jarvis import presencevote as pv
        self.fabric = fabric
        self._cfg = cfg
        self.stuck = stuck
        self.phone = phone
        self.eye = eye
        self.mic = mic
        # SECONDS SINCE THE KEYBOARD OR MOUSE MOVED, or None. The MIC'S
        # TWIN, and the fix for 15:28:13 and 17:07:28 on 2026-09-06 -- two
        # "away (cell 6)" lines with him at the desk, his phone napping,
        # the camera leg dark and the mic quiet past its window. He had
        # typed a command at 15:17 and no leg was reading it.
        # ``deskpresence.DeskSentinel.idle_s`` on the live box; a number,
        # never a keystroke, and there is nothing on this path that could
        # carry one.
        self.desk = desk
        # HOW OLD THAT NUMBER IS, or None on a box that cannot say.
        #
        # THE BELT OVER THE SEVENTH GUARD-ONE-HALF DEFECT, 2026-09-06.
        # ``DeskSentinel.idle_s`` handed out a cache written only on a
        # SUCCESSFUL poll, so a dead idle monitor left the last good number
        # standing for ever -- measured, 400 no-signal polls (3 h 20 min)
        # and it still answered 5.0 s, which is DESK_AT, which vetoes away
        # in every cell; at the sentinel, rooms clear and his phone gone
        # from t=0, AWAY NEVER FIRED in six hours. The sentinel now expires
        # its own reading, and ``presencevote.desk_leg`` refuses a stale one
        # even if a different reader lets one through.
        self.desk_age = desk_age
        # "IS SENSING SWITCHED OFF?" -- a callable, or None to read it off
        # the fabric. See ``_sensing_off``. Named ``_fn`` because the
        # method of that name is what the vote calls.
        self._sensing_off_fn = sensing_off
        # ARM THE LENS AND WAIT, BOUNDED -- jarvis/eyeloop.EyeLoop.wait_for_look.
        # Called for the two cells his rule 1 cannot reach (see __call__).
        self.look = look
        # Seconds the house has been AWAY, or None. The bedroom split needs
        # it because its office branch is a CONTINUITY argument and
        # continuity is unavailable after an absence.
        self.away_s = away_s
        self.door_room = door_room
        self.desk_room = desk_room
        # THE PHONE'S OWN GRACE, AND IT WAS 0.0 UNTIL 2026-09-06.
        #
        # ``phone_leg`` was written to read a False inside the grace as
        # UNKNOWN -- "an iPhone drops off Wi-Fi power-save for minutes at a
        # time and a napping radio is not a departure" -- and this class
        # set ``grace_s = 0.0`` and never wrote it again, so ``0.0 >= 0.0``
        # made the FIRST unanswered ping a hard PHONE_NO. The guard was
        # built and never applied to the leg it was built for.
        #
        # The cost was measured, not imagined: at 10:49:03 on 2026-09-06,
        # five seconds after a restart with him standing in the flat, the
        # voter published "away (cell 24) -- phone-only away past the
        # grace". There was no grace. The sentence was false.
        #
        # It is ``presence.away_after_min`` (720 s on his box) because that
        # is the number that already means "long enough that a silent radio
        # is a departure", and re-deriving it here is how two clocks drift
        # apart. ``presence.phone_grace_s`` overrides it.
        # An EXPLICIT grace is honoured verbatim, zero included: a test
        # that means "no grace, I am exercising a different cell" must be
        # able to say so, and _seconds treats zero as nonsense (rightly --
        # a config that says 0 is a typo, not a policy).
        if grace_s is not None:
            self.grace_s = max(0.0, float(grace_s))
        else:
            self.grace_s = _seconds(
                _cfg_get(cfg, "presence.phone_grace_s", None),
                self._away_after_s(), floor_s=0.0)
        # How long the FIRST vote may wait for the room fabric's first
        # poll. Three poll periods at his rooms_poll_s of 2.0. The race it
        # closes was measured at 1.4 s.
        self.boot_grace_s = DEFAULT_BOOT_GRACE_S if boot_grace_s is None \
            else max(0.0, float(boot_grace_s))
        self.recency_s = _seconds(recency_s, pv.DEFAULT_RECENCY_S)
        self.mic_window_s = _seconds(mic_window_s, pv.DEFAULT_MIC_WINDOW_S)
        # ONE NUMBER, ONE MEANING -- the same discipline the mic window
        # follows. The desk window IS the corroboration recency window: a
        # keystroke is independent agreement, and that key already means
        # "how fresh does agreement have to be to keep saying he is here".
        #
        # NOT ``presence.desk_away_after_min``, which was the first choice
        # and is the wrong question -- it asks whether the CHAIR is empty.
        # RE-MEASURED IN THIS TREE on a fake clock, rooms clear, his phone
        # gone and his last keystroke at the moment he walked out: the leg
        # off gives away at 11.8 min, the shipped 900 s window 15.0, and the
        # 1500 s window 25.0. (The 36.8 this comment used to claim is no
        # longer reproducible -- it predates Verdict.witness_s_ago, when the
        # desk window and the sentinel's grace still stacked.) So every
        # errand under about twenty-five minutes would have earned
        # silence on his return. At the recency window it lands at ~15.
        self.desk_window_s = _seconds(
            desk_window_s if desk_window_s is not None else self.recency_s,
            pv.DEFAULT_DESK_WINDOW_S)
        self.verdict = None
        self.legs: dict = {}
        self._said = ""
        # True while the ONLY thing holding this leg back from PHONE_NO is
        # its own grace: the phone was asked and did not answer. The
        # sentinel reads it so the two graces cannot STACK -- see
        # PresenceSentinel.tick.
        self.grace_running = False
        # How long ago the evidence behind the last HOME actually landed.
        # Zero unless a witness veto is holding the away off -- see
        # ``__call__`` and PresenceSentinel.tick.
        self.seen_s_ago = 0.0
        self._now = now
        self._started = None       # first call, on the monotonic clock
        self._last_yes = None      # the last tick the phone actually answered
        self._rooms_waited = False
        self._rooms_said = False
        # Said ONCE, not every poll: the desk room is not a configured
        # sensor, or a room is occupied and it is not the one the lens can
        # see. Both are reasons a look did NOT happen, and a refusal with
        # no line on disk is the 2026-09-05 defect in miniature.
        self._desk_room_said = False
        self._other_room_said = False

    def _away_after_s(self) -> float:
        """``presence.away_after_min`` in seconds -- the sentinel's own
        number, read the same way it reads it."""
        try:
            mins = float(_cfg_get(self._cfg, "presence.away_after_min",
                                  DEFAULT_AWAY_MIN))
        except (TypeError, ValueError):
            mins = DEFAULT_AWAY_MIN
        return max(60.0, mins * 60.0)

    # ------------------------------------------------------- the legs
    def _wait_for_rooms(self) -> None:
        """ONCE, on the first vote: let the fabric finish one poll.

        Bounded, on the presence daemon thread, and it cannot deadlock --
        the fabric's own thread is what sets the event and this waits on it
        with a timeout. It is deliberately NOT in ``PresenceSentinel.start``:
        that runs on the startup thread and would delay the whole boot.

        Only when the fabric's thread is actually running. With no thread
        (a test, or a build that only wants the leg) ``_rooms_leg`` ticks
        the fabric itself a few lines below, so there is nothing to wait for.
        """
        if self._rooms_waited:
            return
        self._rooms_waited = True
        fabric = self.fabric
        if fabric is None or self.boot_grace_s <= 0.0:
            return
        if not getattr(fabric, "running", False):
            return
        wait = getattr(fabric, "wait_ready", None)
        if not callable(wait):
            return
        try:
            if wait(self.boot_grace_s):
                return
        except Exception:  # noqa: BLE001 - a broken fabric is not a verdict
            log.debug("presence: the boot wait failed", exc_info=True)
            return
        if not self._rooms_said:
            self._rooms_said = True
            log.info("presence: the rooms had not answered within %.1fs of "
                     "the first vote; voting without the rooms",
                     self.boot_grace_s)

    def _rooms_leg(self):
        """(leg, last_room, age, desk_room_occupied). Never raises.

        THE FOURTH VALUE IS WHAT THE LENS IS ALLOWED TO OPEN FOR, added
        2026-09-06. The leg itself is a HOUSE answer -- ROOMS_ON means SOME
        room reads occupied -- and the burst used to be armed off exactly
        that. But the lens is in the OFFICE and his flat is a corridor, so
        him in the kitchen with the office empty armed a burst to look at a
        room he was not in: MEASURED at 90 bursts, 2250 frames and 90 device
        opens per hour of votes. This says whether the DESK ROOM itself --
        the room the lens can actually see -- reads occupied.
        """
        from jarvis import presencevote as pv
        if self.fabric is None:
            return pv.ROOMS_UNREACHABLE, "", None, False
        self._wait_for_rooms()
        try:
            # The same rule ``HouseView.read`` follows, and for the same
            # reason: when the fabric's own 2 s thread is running we read
            # what it last saw rather than polling every ESP32 a second
            # time on the sentinel's thread. With no thread (a test, or a
            # build that only wants the presence leg) we tick it ourselves
            # -- without this the readings are whatever the last tick left,
            # which on a never-ticked fabric is None for every room.
            if not getattr(self.fabric, "running", False):
                self.fabric.tick()
            readings = self.fabric.readings()
            faulted = set(self.fabric.stuck_rooms())
        except Exception:  # noqa: BLE001 - a broken fabric is not a verdict
            log.debug("presence: the fabric could not be read", exc_info=True)
            return pv.ROOMS_UNREACHABLE, "", None, False
        if self.stuck is not None:
            try:
                for room, value in readings.items():
                    self.stuck.observe(room, value)
                faulted |= set(self.stuck.faulted())
            except Exception:  # noqa: BLE001
                log.debug("presence: the stuck detector failed", exc_info=True)
        # THE HINT MUST NOT NAME A SENSOR THE VOTE HAS ALREADY THROWN OUT.
        # At 14:05:47.969 cell 12 printed "the last room to see anybody was
        # the office ... so he never left the office", and at that instant
        # rooms_leg had computed CLEAR precisely BECAUSE the office was
        # faulted. One verdict, two contradictory readings of one sensor.
        # last_seen_room takes max(last_true) over every room, and the
        # latched office's last_true was ~0 s old.
        try:
            last_room, age = self.fabric.last_seen_room(skip=faulted)
        except TypeError:      # a fabric from before the skip existed
            try:
                last_room, age = self.fabric.last_seen_room()
                from jarvis.presencevote import _slug as _sl
                if _sl(last_room) in {_sl(r) for r in faulted}:
                    last_room, age = "", None
            except Exception:  # noqa: BLE001
                last_room, age = "", None
        except Exception:  # noqa: BLE001
            last_room, age = "", None
        return (pv.rooms_leg(readings, faulted=faulted), last_room, age,
                self._desk_room_on(readings, faulted))

    def _desk_room_on(self, readings, faulted) -> bool:
        """Does the room the LENS IS IN read occupied right now?

        Same faulted-drop as ``rooms_leg`` so the two cannot disagree about
        which sensors are in play. A desk room that is not a configured
        sensor at all answers False and says so ONCE at INFO -- a box that
        cannot tell the lens there is something to look at must not guess,
        and a silently dark leg is the failure this file keeps paying for.
        """
        from jarvis import presencevote as pv
        try:
            key = pv._slug(self.desk_room)
            drop = {pv._slug(r) for r in (faulted or ())}
            live = {pv._slug(r): v for r, v in dict(readings or {}).items()
                    if pv._slug(r) not in drop}
            names = dict(readings or {})
            if names and key not in {pv._slug(r) for r in names}:
                # Only once the fabric has actually answered: an empty
                # readings dict is a fabric that has not polled yet, not a
                # box whose desk room has no sensor.
                if not self._desk_room_said:
                    self._desk_room_said = True
                    log.info("presence: the lens is in %r, which is not one "
                             "of the configured rooms (%s), so a room change "
                             "cannot arm a look. presence.desk_room names it.",
                             self.desk_room,
                             ", ".join(sorted(names)))
                return False
            return live.get(key) is True
        except Exception:  # noqa: BLE001 - never cost the vote
            log.debug("presence: the desk room could not be read", exc_info=True)
            return False

    def _camera_leg(self):
        """A NAME AND A COUNT, never a frame.

        ``eye()`` returns ``(identity, faces, live)``. No eye at all -- and
        that is his box today, because nothing in the tree ever assigns
        ``services.camera_feed`` -- is BLIND, which is "could not look" and
        never "looked and saw nobody".
        """
        from jarvis import presencevote as pv
        if self.eye is None:
            return pv.CAM_BLIND
        try:
            identity, faces, live = self.eye()
        except Exception:  # noqa: BLE001 - a broken eye is not a verdict
            log.debug("presence: the eye could not be asked", exc_info=True)
            return pv.CAM_BLIND
        return pv.camera_leg(identity=identity, faces=faces, live=live)

    def _mic_leg(self):
        """(leg, seconds ago, WHY). ONE NUMBER OFF THE LEDGER, never audio.

        ``mic()`` is ``TurnLedger.idle_s`` on the live box: seconds since
        the microphone last completed a turn, or None if it never has. No
        reader at all -- any box where the app has not wired the ledger in
        -- is UNKNOWN, which never votes; a reader that raises is the same.
        Nothing here opens a device or touches a sample.

        THE THIRD RETURN VALUE IS THE 2026-09-06 WORDING FIX. MIC_UNKNOWN
        has three quite different causes and ``cell6`` printed "the mic
        ledger could not be read" for all of them -- including the one that
        is true at EVERY boot, where the ledger is perfectly readable and
        simply holds no turn yet (``TurnLedger.idle_s`` returns None until
        the first one). Every cell-6 line in his log on 09-06, 11:05:19
        included, printed the wrong one of the three. The leg VALUE is
        unchanged in all three -- it must not vote either way -- and only
        the sentence differs.
        """
        from jarvis import presencevote as pv
        if self.mic is None:
            return pv.MIC_UNKNOWN, None, pv.MIC_WHY_NO_READER
        try:
            s_ago = self.mic()
        except Exception:  # noqa: BLE001 - a broken ledger is not a verdict
            log.debug("presence: the mic ledger could not be read", exc_info=True)
            return pv.MIC_UNKNOWN, None, pv.MIC_WHY_UNREADABLE
        if s_ago is None:
            return pv.MIC_UNKNOWN, None, pv.MIC_WHY_NO_TURN
        leg = pv.mic_leg(s_ago=s_ago, window_s=self.mic_window_s)
        if leg == pv.MIC_UNKNOWN:
            return leg, None, pv.MIC_WHY_NONSENSE
        return leg, float(s_ago), ""

    def _desk_leg(self):
        """(leg, seconds ago). ONE NUMBER OFF THE IDLE MONITOR.

        THE MIC'S TWIN, and written directly beneath it on purpose: these
        two answer the same question -- "has anything actually had him in
        the flat?" -- and a change to one that is not made to the other is
        the defect shape this repo has paid for four times in two days.

        ``desk()`` is ``DeskSentinel.idle_s`` on the live box: seconds
        since the last keyboard or mouse event off Mutter's idle monitor,
        or None when there is no signal. NOT A KEYSTROKE, not a window
        title, not a command -- a duration, and nothing on this path could
        carry anything else.

        No reader, a reader that raises, or a nonsense number is
        DESK_UNKNOWN, which votes nothing. An IDLE desk is DESK_IDLE, which
        also votes nothing: he reads papers at that desk and the bedroom
        has no sensor, so an empty chair is not an empty flat. Only
        DESK_AT ever changes a verdict, and only ever away -> home.
        """
        from jarvis import presencevote as pv
        if self.desk is None:
            return pv.DESK_UNKNOWN, None
        try:
            idle = self.desk()
        except Exception:  # noqa: BLE001 - a broken monitor is not a verdict
            log.debug("presence: the desk idle monitor could not be read",
                      exc_info=True)
            return pv.DESK_UNKNOWN, None
        leg = pv.desk_leg(idle_s=idle, window_s=self.desk_window_s,
                          age_s=self._desk_age())
        if leg != pv.DESK_AT:
            return leg, None
        try:
            return leg, float(idle)
        except (TypeError, ValueError):     # unreachable via desk_leg
            return pv.DESK_UNKNOWN, None

    def _desk_age(self):
        """HOW OLD the idle number is, or None when the box cannot say.

        Read immediately after the value, off the same sentinel, so the two
        come from the same poll in every case that matters. A box with no
        age reader keeps voting exactly as it did -- the SENTINEL's own
        expiry is the guard there and this is the belt.
        """
        fn = self.desk_age
        if not callable(fn):
            return None
        try:
            age = fn()
        except Exception:  # noqa: BLE001 - a broken monitor is not a verdict
            log.debug("presence: the desk reading's age could not be read",
                      exc_info=True)
            return None
        return None if age is None else age

    def _sensing_off(self) -> bool:
        """Has HE switched the sensors off? Never raises.

        A leg the privacy switch darkened did not fail to see him -- it was
        not asked -- and on 2026-09-06 at 15:10:57 the voter read exactly
        that as a phone-only away, with him at the desk talking to it.

        The rooms' OWN word is the source (``RoomFabric.blocked``, which is
        non-empty only while EVERY configured room is blocked), because the
        radar is the leg the switch actually removes from the vote: a
        camera that cannot look is already CAM_BLIND and already votes
        nothing. An injected ``sensing_off`` callable overrides it, which
        is how a box with a different set of governed sensors says so.
        """
        get = self._sensing_off_fn
        if callable(get):
            try:
                return bool(get())
            except Exception:  # noqa: BLE001 - a broken switch is not a vote
                log.debug("presence: the sensing switch could not be read",
                          exc_info=True)
                return False
        fabric = self.fabric
        if fabric is None:
            return False
        try:
            return bool(getattr(fabric, "blocked", ""))
        except Exception:  # noqa: BLE001 - a fabric from before `blocked`
            return False

    def _agreed_s_ago(self):
        """Seconds since anything independent last agreed with an OPEN
        occupied run -- the freshest across rooms -- or None with no run on
        the books.

        A run nothing has agreed with YET is aged from its start: a man who
        just sat down has a young run and his phone has not woken, and that
        is not a fault. What this must never do is answer "ever": the first
        cut asked ``corroborated_s_ago is not None`` here, which stays true
        for the rest of the run after one phone hit, so a radar that saw him
        and then latched kept him "home" for as long as it took the
        45-minute stuck fault to drop the room. Measured: a trip of an hour
        was not greeted. The window is applied in ``presencevote.cell6``.

        THE THIRD STATE, ADDED 2026-09-06 AND MEASURED FROM THE LIVE FILE.
        A run that was ALREADY LATCHED when Jarvis started has no honest
        age at all. His roomruns.json held exactly one run that morning --
        ``office started 10:50:03, corroborated 14:17:53`` -- and the run
        on the books BEGAN 65 SECONDS AFTER BOOT, because a radar latched
        since before the restart was re-aged to zero on its first
        post-boot observation. That is why cell 6 at 10:50:04 said
        "something else agreed with that run 1 s ago" and granted a full
        15-minute HOME window to a sensor nothing had agreed with, and why
        "away" did not arrive until 11:05:19 -- about fourteen minutes
        after he actually walked out.

        Neither available answer is right. Ageing from boot invents a
        number; answering None reads in ``cell6`` as HOME for ever. So the
        flag travels beside the age and the cell holds UNKNOWN on it.
        Returns ``(seconds, pre_existing)``.
        """
        if self.stuck is None:
            return None, False
        try:
            ages, pre = [], False
            for row in self.stuck.status().values():
                since = row.get("corroborated_s_ago")
                if row.get("pre_existing") and since is None:
                    # The latch may be seconds or hours old and the box
                    # genuinely does not know. It contributes NO age.
                    pre = True
                    continue
                run_s = row.get("run_s")
                if not run_s:
                    continue
                ages.append(float(run_s if since is None else since))
            return (min(ages) if ages else None), pre
        except Exception:  # noqa: BLE001 - unreadable history is not a fault
            log.debug("presence: the run history could not be read", exc_info=True)
            return None, False

    # -------------------------------------------------------- the vote
    def __call__(self, ip: str = "", mac: str = "") -> Optional[bool]:
        from jarvis import presencevote as pv
        now = self._now()
        if self._started is None:
            self._started = now
        rooms, last_room, age, desk_room_on = self._rooms_leg()
        camera = self._camera_leg()
        mic, mic_s_ago, mic_reason = self._mic_leg()
        desk, desk_s_ago = self._desk_leg()
        try:
            answer = self.phone(ip, mac) if (ip or mac) else None
        except Exception:  # noqa: BLE001 - the phone must not break the vote
            log.debug("presence: the phone probe failed", exc_info=True)
            answer = None
        if answer is True:
            self._last_yes = now
        # HOW LONG THE PHONE HAS BEEN SILENT, which this class never
        # measured. See ``grace_s`` in __init__ for the 10:49:03 line that
        # a hardcoded 0.0 produced. A phone that has NEVER answered is aged
        # from the first vote, not from zero.
        anchor = self._last_yes if self._last_yes is not None else self._started
        phone = pv.phone_leg(answer=answer, unseen_s=max(0.0, now - anchor),
                             grace_s=self.grace_s)
        # THE TWO GRACES MUST NOT STACK, and in exactly one case they did.
        # Normally they cannot: the sentinel measures ``away_after_s`` from
        # ``last_seen``, which froze the moment the phone stopped
        # answering -- the same instant this leg's clock started -- so
        # "away" lands at the same minute it always did (MEASURED: the
        # 26-minute desk residual in tests/test_presence_recency.py is
        # unchanged by this grace).
        #
        # The exception is a start where the phone has NEVER answered.
        # ``last_seen`` is None then, the sentinel falls back to
        # ``_started_at``, and ``_started_at`` is only set on a non-None
        # answer -- so it began when THIS grace expired and away took 25
        # minutes instead of 13 (measured on a fake clock, both ways).
        # This flag lets the sentinel start its own clock while the phone
        # is definitively silent, without weakening the rule it has for a
        # leg that has said NOTHING (a sensor-only box whose sensor is
        # down), which is a different thing and still holds everything.
        self.grace_running = (answer is False and phone == pv.PHONE_UNKNOWN)

        # THE TWO CELLS WHERE THE LENS EARNS "NUMBER ONE". Rooms and phone
        # are both known by now, and a room reading occupied with a phone
        # that is silent (cell 6) or unaskable (cell 9) is exactly where
        # his rule 1 has no precondition today -- a camera that could not
        # look is not evidence of an empty room, so the rules fall silent
        # and history has to break the tie. Every cell-6 line in his log on
        # 2026-09-06 (10:50:04, 11:05:19, 14:07:49) would have been decided
        # by the lens instead. Bounded inside the producer; a timeout costs
        # nothing, because BLIND is what the leg already answers.
        #
        # AND ONLY FOR THE ROOM THE LENS IS IN, since 2026-09-06. The arm
        # used to read ROOMS_ON, which is a HOUSE answer: him in the kitchen
        # with the office empty opened the device to look at a room he was
        # not in. MEASURED with a fake opener and synthetic arrays -- office
        # empty, kitchen occupied, phone unaskable, one hour of votes: 90
        # bursts, 2250 frames, 90 device opens, and every one of them scored
        # as "looked and saw nobody" FOR THE WHOLE FLAT, because CAM_LOOKED
        # carries no room. It is his lamp and his lens; the leg gains
        # nothing from a room it cannot see.
        #
        # A LATCHED DESK ROOM STILL ARMS, and that is deliberate: a latched
        # office IS cell 6, and the lens is the one leg that can end that
        # argument. That look is the cost of the leg doing its job.
        if (camera == pv.CAM_BLIND and rooms == pv.ROOMS_ON
                and phone in (pv.PHONE_NO, pv.PHONE_UNKNOWN)
                and callable(self.look)):
            if not desk_room_on:
                if not self._other_room_said:
                    self._other_room_said = True
                    log.info("presence: a room reads occupied but %r does "
                             "not, and the lens is in %r -- not opening it "
                             "to look at a room it cannot see",
                             self.desk_room, self.desk_room)
            else:
                try:
                    if self.look(why="the desk room is occupied and the "
                                     "phone is quiet"):
                        camera = self._camera_leg()
                except Exception:  # noqa: BLE001 - a broken lens is not a verdict
                    log.debug("presence: the lens could not be armed",
                              exc_info=True)

        if self.stuck is not None:
            # Corroboration: something independent agreed that somebody is
            # here. House-level and conservative -- see jarvis/stuckroom.py.
            try:
                if phone == pv.PHONE_YES:
                    self.stuck.corroborate("phone")
                if camera == pv.CAM_SAW:
                    self.stuck.corroborate("camera")
                if mic == pv.MIC_HEARD and mic_s_ago is not None:
                    # Stamped AT THE TURN, not at this poll -- a turn nine
                    # minutes ago is not an agreement now. The app stamps
                    # each wake as it happens too; this is the same fact
                    # from the other side of a restart, and an older stamp
                    # is ignored, so the two never double-count. Without it
                    # the 45-minute fault dropped a room the mic was
                    # vouching for and cell 15 called him away at the desk.
                    self.stuck.corroborate("mic", ago=mic_s_ago)
            except Exception:  # noqa: BLE001
                log.debug("presence: corroboration failed", exc_info=True)

        # Cell 6 needs to know how RECENTLY any occupied room's run was
        # agreed with -- not whether it ever was. Read only when a room is
        # on; the other 26 cells never see it.
        agreed_s_ago, agreed_pre = (self._agreed_s_ago()
                                    if rooms == pv.ROOMS_ON else (None, False))

        # HIS OWN PRIVACY SWITCH, read once and handed to both paths. A
        # leg he switched off is BLIND, never a NO -- 15:10:57.
        off = self._sensing_off()

        if not (ip or mac):
            # No phone leg configured AT ALL -- a sensor-only install, not
            # his box. See presencevote.decide_rooms_only for why P2 has
            # to yield when there is nothing to outvote the radar with.
            verdict = pv.decide_rooms_only(rooms=rooms, camera=camera,
                                           mic=mic, mic_s_ago=mic_s_ago,
                                           desk=desk, desk_s_ago=desk_s_ago,
                                           sensing_off=off)
        else:
            verdict = pv.decide(phone=phone, camera=camera, rooms=rooms,
                                agreed_s_ago=agreed_s_ago,
                                agreed_pre_existing=agreed_pre,
                                recency_s=self.recency_s,
                                mic=mic, mic_s_ago=mic_s_ago,
                                mic_reason=mic_reason,
                                desk=desk, desk_s_ago=desk_s_ago,
                                sensing_off=off,
                                last_room=last_room, last_room_age_s=age,
                                door_room=self.door_room,
                                desk_room=self.desk_room,
                                away_s=self._away_now())
        self.verdict = verdict
        self.legs = {"phone": phone, "camera": camera, "rooms": rooms,
                     "mic": mic, "desk": desk}
        self._log(verdict)
        # WHEN THE EVIDENCE ACTUALLY LANDED, for the sentinel's away clock.
        # 0.0 -- "now" -- for every ordinary verdict; the witness's own age
        # when a stale keystroke or turn is what withheld an away. See
        # presencevote.Verdict.witness_s_ago for the measured 26.8 -> 15.0
        # this saves, and for the half of it cell 6 was still missing on
        # 2026-09-06.
        try:
            self.seen_s_ago = max(0.0, float(verdict.witness_s_ago or 0.0))
        except (TypeError, ValueError):
            self.seen_s_ago = 0.0
        if verdict.state == pv.UNKNOWN:
            return None
        return verdict.state != pv.AWAY

    def _away_now(self):
        """Seconds the house has been AWAY, or None. Never raises.

        The bedroom split's office branch is a CONTINUITY argument -- "he
        never left the office and the radar dropped a still body" -- and
        continuity is unavailable after an absence. At 14:05:47.969 on
        2026-09-06 the voter printed that sentence and then, at .970,
        "presence: home (returned)" for a man who had been out two and a
        half hours. Both lines came from the same tick.
        """
        get = self.away_s
        if not callable(get):
            return None
        try:
            out = get()
        except Exception:  # noqa: BLE001 - the sentinel must not cost the vote
            return None
        try:
            return None if out is None else max(0.0, float(out))
        except (TypeError, ValueError):
            return None

    def _log(self, verdict) -> None:
        """The reason, at INFO, once per change.

        This is the line that did not exist on 2026-09-05. The whole day's
        log held seven "presence: home" lines, zero "away" lines and no
        word about why the door did not open -- so "why did he not greet
        me" had no answer anywhere on disk.
        """
        key = "%s/%s" % (verdict.cell, verdict.state)
        if self._said == key:
            return
        self._said = key
        log.info("presence: %s (cell %d) -- %s [phone %s, camera %s, rooms %s, "
                 "mic %s, desk %s]",
                 verdict.state, verdict.cell, verdict.reason,
                 self.legs.get("phone"), self.legs.get("camera"),
                 self.legs.get("rooms"), self.legs.get("mic"),
                 self.legs.get("desk"))


def make_probe(sensor, phone: Callable = probe) -> Callable:
    """The probe the sentinel polls. No sensor -> the phone probe ITSELF,
    so an unconfigured box runs the identical code path it ran before this
    module existed."""
    return phone if sensor is None else RoomOrPhone(sensor, phone)


class PresenceSentinel:
    """See the module docstring. ``state`` is 'home' | 'away' | 'unknown'."""

    # The leg-and-reason already named in the unreadable-`blocked` ERROR
    # (see _warn_no_blocked). A class default so a sentinel built by a test
    # with object.__new__ still answers the question.
    _no_blocked_warned = ""

    def __init__(self, cfg, publish: Callable = bus.publish,
                 probe_fn: Optional[Callable] = None,
                 now: Callable[[], float] = time.time, poll_s: Optional[float] = None,
                 policy=None):
        self._cfg = cfg
        self._publish = publish
        # The room sensor is composed IN here rather than wired in app.py:
        # probe_fn was always the injection point, and an explicit one
        # (every test) still wins outright. With presence.rooms configured
        # the whole FABRIC is the leg and the singular sensor is not built
        # at all -- two owners polling one ESP32 would double its traffic
        # and race the breaker's counters for nothing (roomfabric.HouseView
        # says the same about its own two callers).
        self.fabric = _make_fabric(cfg, policy)
        self.sensor = (self.fabric.house_view() if self.fabric is not None
                       else _make_sensor(cfg, policy))
        # HIS THREE-LEG VOTER, when there is a fabric to read rooms from.
        # ``presence.three_legs`` is OFF BY DEFAULT. Turning it on makes the
        # camera, then the phone, then the room sensors vote on "away" in
        # his order, so a room reading occupied no longer stops the phone
        # being asked and a latched radar cannot cost him the greeting.
        # Off, the 2026-09-05 composition (RoomOrPhone) runs byte for byte.
        # It ships off because it changes what "away" means on a live box,
        # and a merge plus a restart must not decide that for him.
        self.stuck = None
        self.legs = None
        if self.fabric is not None and \
                bool(_cfg_get(cfg, "presence.three_legs", False)):
            self.legs = self._build_legs(cfg)
        self._probe = (probe_fn if probe_fn is not None
                       else (self.legs if self.legs is not None
                             else make_probe(self.sensor)))
        self._now = now
        self._poll_s = poll_s
        self.home: Optional[bool] = None
        self.since: float = 0.0
        self.last_seen: Optional[float] = None
        self._started_at: Optional[float] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _build_legs(self, cfg):
        """The three-leg probe, or None. Never raises: a broken voter must
        cost the FEATURE, not the presence leg the box already had."""
        try:
            from jarvis import stuckroom
            self.stuck = stuckroom.StuckRooms(publish=bus.publish)
        except Exception:  # noqa: BLE001 - the fault board is optional
            log.exception("presence: the stuck-room detector could not be "
                          "built; the voter runs without corroboration")
            self.stuck = None
        from jarvis import presencevote as pv
        # ONE EDIT: presence.corroboration_recency_min, in minutes. The
        # default is derived, not measured -- pv.RECENCY_S_PROVENANCE says
        # from what -- and his own short-trip length is the right value.
        recency_s = _seconds(
            _minutes(_cfg_get(cfg, "presence.corroboration_recency_min", None)),
            pv.DEFAULT_RECENCY_S)
        # The mic window is the departure veto's own number
        # (presence.departure_mic_silence_min), read through arrival's own
        # reader so the two can never drift apart.
        try:
            from jarvis import arrival
            mic_window_s = arrival.mic_silence_s(
                lambda key, default=None: _cfg_get(cfg, key, default))
        except Exception:  # noqa: BLE001 - the default is the same number
            mic_window_s = None
        # A DEFAULT TAKEN FOR HIM, AND IT NOW DECIDES MORE THAN IT DID.
        # ``presence.desk_room`` is UNSET on his box, so "office" is a code
        # default -- and since 2026-09-06 it is also the room whose radar
        # may OPEN THE LENS. The key was a tie-breaker for the bedroom
        # split when it was chosen; it is a privacy-shaped decision now, so
        # the boot log names it rather than letting it stay implicit.
        desk_room = _cfg_get(cfg, "presence.desk_room", None)
        if desk_room in (None, ""):
            from jarvis.arrival import DEFAULT_DESK_ROOM
            desk_room = DEFAULT_DESK_ROOM
            log.info("presence: presence.desk_room is unset, so %r is a CODE "
                     "DEFAULT -- it names the room whose radar may arm the "
                     "camera and the room the bedroom split reasons from. "
                     "Set presence.desk_room to overrule it.", desk_room)
        desk_room = str(desk_room)
        try:
            legs = ThreeLegProbe(
                fabric=self.fabric, cfg=cfg, stuck=self.stuck,
                # The bedroom split needs to know a return from continuity;
                # only the sentinel holds that. See ThreeLegProbe._away_now.
                away_s=self._away_for_legs,
                door_room=str(_cfg_get(cfg, "presence.door_room", "kitchen")),
                # ``presence.desk_room``, NOT ``presence.desk``.
                # ``presence.desk`` is deskpresence.py's BOOLEAN switch and
                # it is True on his box, so reading it here gave the room
                # name "True" and the bedroom split lost its office branch
                # outright. tests/test_presence_desk_key.py pins it.
                desk_room=desk_room,
                recency_s=recency_s, mic_window_s=mic_window_s)
        except Exception:  # noqa: BLE001
            log.exception("presence: the three-leg voter could not be built; "
                          "falling back to the room-or-phone composition")
            return None
        log.info("presence: three-leg voter active (camera, phone, rooms; "
                 "the mic breaks cell 6) -- an agreement counts for %d min, "
                 "a turn for %d min; set presence.three_legs false to go back",
                 int(legs.recency_s // 60), int(legs.mic_window_s // 60))
        # SAID OUT LOUD because it CHANGES WHAT AWAY MEANS on a live box.
        # Until 2026-09-06 the phone leg's grace was hardcoded 0.0 and the
        # first unanswered ping was a hard "no" -- so "away" could arrive a
        # few seconds after a restart, which is exactly what happened at
        # 10:49:03. With a real grace it cannot arrive for %d minutes after
        # a start where the phone has never answered. That is the intended,
        # safe outcome (nothing is greeted and nothing is muted while the
        # state is unknown), and it is his to overrule with
        # presence.phone_grace_s.
        log.info("presence: the phone leg now waits %d min before a silent "
                 "radio counts as away (presence.phone_grace_s); the first "
                 "unanswered ping used to be a hard no, and the verdict "
                 "sentence that said otherwise was false",
                 int(legs.grace_s // 60))
        return legs

    def _away_for_legs(self):
        """Seconds the house has been AWAY, or None. Read WITHOUT the lock
        being held by the caller: ``tick`` calls the probe outside it."""
        with self._lock:
            if self.home is not False or not self.since:
                return None
            since = self.since
        try:
            return max(0.0, self._now() - since)
        except Exception:  # noqa: BLE001 - a broken clock is not a verdict
            return None

    @property
    def verdict(self):
        """The last three-leg verdict, or None on the old path. Carries the
        reason, so a consumer can say WHY rather than only what."""
        return getattr(self.legs, "verdict", None)

    def corroborate(self, source: str = "") -> None:
        """The mic heard him (or any other independent leg agreed).

        The app feeds the turn ledger in here. It is the third
        corroboration source in jarvis/stuckroom.py and the one neither the
        phone nor the camera can supply.
        """
        if self.stuck is None:
            return
        try:
            self.stuck.corroborate(source or "mic")
        except Exception:  # noqa: BLE001
            log.debug("presence: corroborate failed", exc_info=True)

    # ------------------------------------------------------------ config
    @property
    def phone_ip(self) -> str:
        return str(_cfg_get(self._cfg, "presence.phone_ip", "") or "").strip()

    @property
    def phone_mac(self) -> str:
        return str(_cfg_get(self._cfg, "presence.phone_mac", "") or "").strip().lower()

    @property
    def configured(self) -> bool:
        """Enabled, and at least one leg to stand on."""
        return bool(_cfg_get(self._cfg, "presence.enabled", True)) and \
            bool(self.phone_ip or self.phone_mac or self.sensor is not None)

    @property
    def away_after_s(self) -> float:
        try:
            mins = float(_cfg_get(self._cfg, "presence.away_after_min", DEFAULT_AWAY_MIN))
        except (TypeError, ValueError):
            mins = DEFAULT_AWAY_MIN
        return max(60.0, mins * 60.0)

    @property
    def poll_s(self) -> float:
        """Seconds until the next probe -- FASTER while he is out.

        Arrival is detected on the next poll, so at the 60 s default "the
        room notices the door" is in practice "the room notices up to a
        minute after he sat down", by which time he may already be
        mid-utterance and a staged arrival cue reads as late rather than
        composed. Polling every ~10 s costs one ping per 10 s and only
        while the house is empty; once he is home the slow cadence is back,
        because a present phone has nothing to tell us in a hurry.
        """
        if self._poll_s is not None:
            return float(self._poll_s)
        key = "presence.poll_s_away" if self.state == "away" else "presence.poll_s"
        default = DEFAULT_AWAY_POLL_S if key.endswith("_away") else DEFAULT_POLL_S
        try:
            return max(5.0, float(_cfg_get(self._cfg, key, default)))
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------- state
    @property
    def state(self) -> str:
        with self._lock:
            return "unknown" if self.home is None else ("home" if self.home else "away")

    def is_home(self) -> bool:
        """True unless the probe has established he is out."""
        with self._lock:
            return self.home is not False

    # -------------------------------------------------------------- tick
    def tick(self) -> Optional[Presence]:
        """One probe; returns the Presence event published, if any."""
        if not self.configured:
            return None
        now = self._now()
        try:
            answer = self._probe(self.phone_ip, self.phone_mac)
        except Exception:  # noqa: BLE001 - the loop must survive anything
            log.exception("presence: probe failed")
            return None
        if answer is None:
            # No evidence AT ALL this tick (a sensor-only install whose
            # sensor is down). Hold everything -- including the boot grace
            # clock, which must start when the first real answer arrives,
            # not when the first blank one does. The one exception is a
            # blackout with no end: see _blacked_out.
            if self._blacked_out():
                self._forget()
            elif getattr(self._probe, "grace_running", False):
                # ...and the OTHER exception, added 2026-09-06. The voter
                # is holding purely because its own phone grace has not
                # expired, which means the phone WAS asked and did not
                # answer. That is evidence, not a blank, and starting the
                # clock here is what stops the two graces stacking into 25
                # minutes on a start where the phone has never answered.
                # last_seen is deliberately NOT touched: he has not been
                # seen, and pretending otherwise would delay away instead.
                with self._lock:
                    if self._started_at is None:
                        self._started_at = now
            return None
        present = bool(answer)
        event = None
        with self._lock:
            if self._started_at is None:
                # The grace clock starts at boot, not at the first miss: a
                # phone asleep at startup would otherwise be "away" on tick
                # one and earn a spurious "Welcome back" on tick two.
                self._started_at = now
                self.last_seen = now if present else None
            if present:
                # STAMPED WHEN HE WAS ACTUALLY SEEN, not when we polled. It
                # is "now" for every ordinary answer; a witness veto -- a
                # keystroke or a turn that is minutes old and is all that
                # is holding an away off -- says how old, and stamping THAT
                # is what stops its window and this grace from stacking.
                # MEASURED IN THIS TREE: rooms clear, phone gone, his
                # last keystroke at the moment he walked out -> away at
                # 15.0 min stamped, 26.8 unstamped. (A keystroke ALREADY 11
                # min old when he leaves is a different scenario and lands
                # at 11.8 either way -- an earlier version of this comment
                # attached the numbers to that one, wrongly.)
                seen_ago = 0.0
                try:
                    seen_ago = max(0.0, float(
                        getattr(self._probe, "seen_s_ago", 0.0) or 0.0))
                except (TypeError, ValueError):
                    seen_ago = 0.0
                self.last_seen = now - seen_ago
                if self.home is not True:
                    returned = self.home is False
                    self.home, self.since = True, now
                    event = Presence(home=True, since=now, returned=returned)
            else:
                anchor = self.last_seen if self.last_seen is not None else self._started_at
                if self.home is not False and now - anchor >= self.away_after_s:
                    self.home, self.since = False, now
                    event = Presence(home=False, since=now, returned=False)
        if event is not None:
            log.info("presence: %s%s", "home" if event.home else "away",
                     " (returned)" if event.returned else "")
            try:
                self._publish(event)
            except Exception:  # noqa: BLE001
                log.exception("presence: publish failed")
        return event

    def _blacked_out(self) -> bool:
        """True while the ONLY leg is a sensor that offline mode switched off.

        A breaker-open sensor is deliberately NOT this: that outage lasts
        30 s and holding the last verdict across it is correct. Offline mode
        lasts until he says otherwise, and there is no honest way to keep
        answering a question nothing has been able to observe for hours.

        A LEG THAT CANNOT SAY WHETHER IT IS BLOCKED IS LOUD -- however it
        fails to say it. This is the privacy path, and it has now gone
        quiet twice in two different ways. First the whole read sat inside
        a bare ``except Exception`` logged at debug, so when the multi-room
        leg arrived without the attribute (2026-09-03) the AttributeError
        was swallowed and the dark-safe path simply stopped existing.
        Then the fix made a MISSING ``blocked`` loud and left the OTHER
        branch of its own ``if`` exactly as it was: a ``blocked`` that
        RAISED -- an unreadable policy file, a bug in a future leg -- was
        still caught, still debug, still silent. Same bug, other half.

        So there is no longer a branch to forget. One ``try`` asks the leg
        the question; ANY answer that is not a usable one -- the attribute
        absent, the property raising, the value refusing ``bool()`` -- is
        the same event, reported at ERROR by ``_warn_no_blocked`` with the
        leg and the reason named. It still returns False (inventing
        "blacked out" from a broken leg would blank presence on every bug)
        but it can no longer do so quietly, and the ERROR is said ONCE per
        leg-and-reason: ``poll_s`` is 60 s, and one line an hour for ever
        is how a real error gets filtered out of a log.

        Caught, not raised, either way. ``tick()`` has no guard of its own
        (only ``_loop`` does), so an exception escaping here costs the
        whole poll -- this blackout's own ``_forget()`` included.
        """
        if self.phone_ip or self.phone_mac:
            return False
        sensor = self.sensor
        if sensor is None:
            return False
        try:
            blocked = getattr(sensor, "blocked", _MISSING)
            if blocked is _MISSING:
                raise AttributeError("the leg has no `blocked`")
            return bool(blocked)
        except Exception as exc:  # noqa: BLE001 - provider boundary
            self._warn_no_blocked(type(sensor).__name__, exc)
            return False

    def _warn_no_blocked(self, leg: str, exc: BaseException) -> None:
        """The unreadable-``blocked`` ERROR, once per leg and reason.

        Keyed on the reason as well as the leg so a leg that starts
        failing a NEW way says so, while the one that is simply wired
        wrong stays one line rather than one a minute.
        """
        key = "%s/%s" % (leg, type(exc).__name__)
        if getattr(self, "_no_blocked_warned", "") == key:
            return
        self._no_blocked_warned = key
        log.error("presence: the %s leg cannot say whether it is blocked "
                  "(%s: %s), so offline mode cannot reach the sentinel; "
                  "holding the last verdict", leg, type(exc).__name__, exc)

    def _forget(self) -> None:
        """Back to "no opinion", without publishing a transition.

        There is no ``Presence(unknown)`` to publish -- ``home`` is a bool
        on the wire -- and inventing one would be the confident false away
        this module exists to avoid. Consumers read ``state`` and
        ``is_home()``, and both already read unknown as "say nothing".
        """
        with self._lock:
            if self.home is None and self._started_at is None:
                return                 # already unknown; say it once
            log.info("presence: sensing is off and there is no phone leg; "
                     "presence is unknown until it is back")
            # The grace clock goes too: nothing was learned during the
            # blackout, so "away" must be earned again from the first real
            # answer rather than from a last_seen older than the switch.
            self.home, self.since = None, 0.0
            self.last_seen, self._started_at = None, None

    # ------------------------------------------------------------ thread
    def _fabric_call(self, what: str) -> None:
        """start / stop the room fabric, if there is one. The sentinel owns
        its leg's thread the way it owns its own: a fabric left running
        after stop() would keep three ESP32s polled into the teardown."""
        fn = getattr(self.fabric, what, None)
        if not callable(fn):
            return
        try:
            fn()
        except Exception:  # noqa: BLE001 - the sentinel still runs without it
            log.exception("presence: room fabric %s failed", what)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.configured:
            log.info("presence: no phone_ip / phone_mac / room sensor configured; "
                     "sentinel idle")
            return
        # BEFORE the sentinel's own thread: the fabric's 2 s cadence is what
        # notices the door, and its first RoomChanged should not have to
        # wait on a 60 s phone poll.
        self._fabric_call("start")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="presence",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._fabric_call("stop")
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("presence: tick failed")
            self._stop.wait(max(0.01, self.poll_s))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
