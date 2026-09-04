"""Does the room radar still cover the room? Asked once, at startup.

MEASURED TWICE ON REAL HARDWARE, 2026-09-03. After a power cycle a sensor
came back with its distance gates at 0. Gate 0 is 0.00-0.75 m, so the
device sees three quarters of a metre and nothing beyond it. The office
looked fine because the entity readback lied for a while; the kitchen
genuinely went blind past 75 cm, which is why he measured 41% presence at
63 cm and then 0% at 1.5 m and we both spent an afternoon on the mount.
Re-running ``scripts/room_sensor.py tune <room>`` fixed it both times.

**THE MECHANISM IS NOT KNOWN AND NOTHING HERE INVENTS ONE.** Not the
firmware, not the NVM, not the power rail -- it was not measured and it is
not guessed at, and no message this module writes claims a cause. What IS
certain is the consequence, and the consequence is the whole reason this
file exists:

    presence degrades to a 75 cm sensor and NOTHING SAYS SO. He sits at
    3.13 m. A blinded radar and an empty room are the same reading, and
    "the room is empty" is exactly the lie the presence lane was built to
    stop Jarvis telling itself.

WHAT IT DOES. For every configured room sensor it reads the device's own
``Max move gate`` and ``Max still gate`` and compares them with what that
room's PROFILE asks for -- ``~/.config/jarvis/room-sensors/<room>.json``,
which ``scripts/room_sensor.py`` owns and which is the same file ``tune``
writes from. A confirmed disagreement is one loud line naming the room,
what the gates should be, what they are, in metres as well as gates, and
the exact command that fixes it.

VERIFIED AGAINST THE LIVE HARDWARE, 2026-09-03, read-only GETs::

    office   192.168.50.51   profile range_m 3.0 still -> wants gate 4 / 4
             device answered  move 4  still 4   -> ok   (171 ms, 2 requests)
    kitchen  192.168.50.52   profile range_m 3.0 still -> wants gate 4 / 4
             device answered  move 4  still 4   -> ok   (418 ms, 2 requests)

So the paths that matter on a healthy box are measured: the entity names
resolve, the bodies parse, the arithmetic agrees with the profile, and a
matching device says nothing at all. THE MISMATCH PATH IS NOT MEASURED
AGAINST HARDWARE and cannot honestly be: reproducing it means writing gate
0 to his device, and the whole argument below is that this code does not
write to his device. It is covered against the fake transport
(tests/test_sensorcheck.py) and against the two numbers the failure
actually produced.

IT REPORTS. IT DOES NOT HEAL, and that is a decision rather than an
omission:

* he tuned these by hand tonight and the tuning is fragile -- a re-apply
  from a stale profile would overwrite a good device with an old file, and
  the profile is not the authority on what he last set at the device;
* the readback is UNRELIABLE shortly after the device powers up (that is
  why the office "looked fine"), so a self-heal is a write triggered by a
  reading we already know can lie;
* a write to his hardware from the boot path is a write nobody watched. A
  wrong report costs him one log line; a wrong write costs him the sensor.

So the sensors it builds are handed a transport that REFUSES to post
(``ReadOnly``): report-only is a property of the object here, not a promise
in a comment.

THE FRESH-DEVICE PROBLEM, and how it is handled. A single 0 immediately
after power-up is NOT proof of a reset -- it is the state the office was in
when it was fine. So a disagreement buys a SECOND READ after
``CONFIRM_DELAY_S`` and nothing else, and only two reads that agree WITH
EACH OTHER and disagree with the profile are loud (``MISMATCH``). Two reads
that disagree with each other are ``UNSETTLED``: said once, quietly, with
the command in it, because a device still making its mind up is not
evidence in either direction. And the whole check starts ``START_DELAY_S``
after boot for the same reason.

EVERYTHING DEGRADES, and none of it is an alarm: no profile for the room
(nothing to compare, and an invented expectation would be a false alarm
every boot), a profile that is not JSON, no network, a 404 on one or both
entities, a body that is not a number, no address for the room, the master
switch off, and offline mode -- which is asked BEFORE the socket, so while
sensing is denied not one request leaves this process.

IT IS NOT ON THE BOOT PATH. ``start()`` returns a daemon thread
immediately; the app holds the handle and never joins it, and an exception
anywhere inside costs the check and nothing else.

**THE POLICY IS NEVER HANDED TO A SENSOR.** ``SensingPolicy.attach``
replaces BY NAME (jarvis/rooms.py argues it at length), so a second sensor
attaching as "radar" would silently take the curfew away from the app's
real one. This asks ``policy.allowed(RADAR)`` itself, exactly as
``jarvis/ui/sensors_page.py`` does, and a policy that RAISES counts as
denied: a decision we could not make is not permission.

NO LENS, NO MIC, NO SECRETS. It reads two integers off a range sensor. The
profile file also holds his Wi-Fi PSK and OTA password; ``read_profile``
takes the two geometry fields out of it and returns those, so nothing else
is ever held, logged or passed on.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jarvis.config import PATHS
from jarvis.logs import get_logger
from jarvis.roomsensor import DEFAULT_TIMEOUT_S, RoomSensor
from jarvis.sensing import RADAR

log = get_logger("sensorcheck")

# From scripts/room_sensor.py, which owns the flashing and the tuning.
# Repeated rather than imported because scripts/ is not an importable
# package; if these ever disagree, that file is right and this is stale --
# and tests/test_sensorcheck.py loads the script by path and crosses the
# two, so a drift fails the suite instead of raising a false alarm on every
# boot.
GATE_M = 0.75            # one LD2410 distance gate
MAX_GATE = 8             # gates 0..8, so 6.0 m
STILL_FLOOR_GATE = 2     # no STILL detection inside 1.5 m, i.e. gates 0 and 1

# The two entities, by NAME, because web_server v2 serves an entity at its
# name percent-encoded and not at the snake_case object_id. jarvis/
# roomsensor.entity_path owns the rule; nothing here spells a path.
MOVE_GATE_ENTITY = "Max move gate"
STILL_GATE_ENTITY = "Max still gate"
GATE_ENTITIES = (MOVE_GATE_ENTITY, STILL_GATE_ENTITY)

# Long enough that a device which booted with the house is past the window
# where its readback lies, short enough that he is told before he has
# stopped looking at the log. NOT measured -- the readback's settling time
# was never instrumented, only observed to exist -- so these are choices,
# and the confirm read is what makes a wrong choice harmless rather than a
# false alarm.
START_DELAY_S = 20.0
CONFIRM_DELAY_S = 20.0
TIMEOUT_S = DEFAULT_TIMEOUT_S

# The four answers. Only one of them is loud.
OK = "ok"                # the device agrees with the profile
MISMATCH = "mismatch"    # two reads agree with each other and not the profile
UNSETTLED = "unsettled"  # two reads disagree with each other: no evidence
UNCHECKED = "unchecked"  # no profile, no answer, no address, sensing denied

THREAD_NAME = "sensor-gate-check"


class ReadOnly(RuntimeError):
    """Raised if anything ever tries to POST through a check's sensor.

    ``RoomSensor`` only posts from ``stop()``/``resume()``, which are only
    reached through ``SensingPolicy.attach``, which this module never
    calls. This is the structural backstop that makes "it cannot write to
    his device" a property of the object rather than a claim about the call
    graph.
    """


def _refuse_post(url: str, timeout: float) -> None:
    raise ReadOnly("the gate check is read-only; it will not POST to %s" % url)


# ------------------------------------------------------------------- gates
def gate_for_metres(metres: float) -> int:
    """The smallest gate whose band REACHES ``metres``.

    Gate N spans 0.75*N .. 0.75*(N+1), so 3.0 m needs gate 4 (3.0-3.75),
    not gate 3 which stops exactly at it. Clamped to 1..8. Byte for byte
    the rule in ``scripts/room_sensor.py``; see the note on the constants.
    """
    if metres <= 0:
        raise ValueError("a distance must be positive")
    return max(1, min(MAX_GATE, math.ceil(float(metres) / GATE_M)))


def gate_range(gate: Optional[int]) -> str:
    """A gate as the metres it actually covers, which is the number he
    reads the room in. Gate 0 is "0.00-0.75 m", and that is the whole
    finding: three quarters of a metre where the profile asked for four
    times that."""
    if gate is None:
        return "unknown"
    return "0.00-%.2f m" % ((int(gate) + 1) * GATE_M)


def _positive(value) -> Optional[float]:
    """A number this module can use as a distance. bool is an int in
    Python and ``"range_m": true`` is not a range; NaN and inf survive
    float() and then poison every comparison."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        out = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def wanted_gates(profile: Optional[dict]) -> Optional[Tuple[int, int]]:
    """(max move gate, max still gate) the profile asks for, or None.

    The still gate can never start before 1.5 m, so a still-capable mount
    never asks for less than gate 2 -- ``Profile.max_still_gate``. None
    means the profile does not say, and a profile that does not say is not
    an expectation to hold the device to.
    """
    if not isinstance(profile, dict):
        return None
    metres = _positive(profile.get("range_m"))
    if metres is None:
        return None
    move = gate_for_metres(metres)
    still = max(STILL_FLOOR_GATE, move) if profile.get("still", True) else move
    return move, still


# ---------------------------------------------------------------- profiles
def profile_dir() -> Path:
    """Where ``scripts/room_sensor.py`` keeps the profiles.

    Derived from ``PATHS.ASSISTANT_CONFIG`` rather than from ``Path.home()``
    so that the suite's redirect of the config directory carries this with
    it -- the tests must never read the real profiles, which hold his
    Wi-Fi PSK.
    """
    return PATHS.ASSISTANT_CONFIG.parent / "room-sensors"


def profile_path(room: str, directory: Optional[Path] = None) -> Path:
    """``<dir>/<room>.json``, slugged exactly as ``Profile.path_for``."""
    slug = re.sub(r"[^a-z0-9-]+", "-", str(room).lower()).strip("-")
    return Path(directory if directory is not None else profile_dir()) / \
        ("%s.json" % slug)


def read_profile(room: str, directory: Optional[Path] = None) -> Optional[dict]:
    """The TWO geometry fields of a room's profile, or None.

    ``range_m`` and ``still`` and nothing else. The file also holds his
    Wi-Fi PSK and OTA password, and the narrow return is what makes it
    impossible for either to be logged, compared or passed on from here.
    Never raises: a missing file, a directory, a permission error or a
    file that is not JSON at all is None, which is "not checked".
    """
    path = profile_path(room, directory)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        log.debug("sensor check: no usable profile at %s", path, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    return {"range_m": data.get("range_m"), "still": data.get("still", True)}


# ----------------------------------------------------------------- results
@dataclass(frozen=True)
class RoomCheck:
    """One room's answer. ``loud`` is the only thing a caller has to read
    to know whether it is worth his attention."""
    room: str
    state: str
    want_move: Optional[int] = None
    want_still: Optional[int] = None
    got_move: Optional[int] = None
    got_still: Optional[int] = None
    detail: str = ""

    @property
    def loud(self) -> bool:
        return self.state == MISMATCH

    def message(self) -> str:
        """What to say, or "" for a room with nothing to say.

        An OK room is silent on purpose: a line every boot saying the
        sensor is fine is a line he stops reading, and then the one that
        matters is in the middle of it.
        """
        if self.state == OK:
            return ""
        if self.state == UNCHECKED:
            return "room sensor %s: not checked -- %s" % (self.room, self.detail)
        head = ("room sensor %s does NOT match its profile" if self.loud else
                "room sensor %s did not settle on an answer") % self.room
        parts = []
        for label, want, got in (("max move gate", self.want_move, self.got_move),
                                 ("max still gate", self.want_still,
                                  self.got_still)):
            if got is None:
                continue
            parts.append("%s is %d (%s) and the profile asks for %d (%s)"
                         % (label, got, gate_range(got), want, gate_range(want)))
        body = "; ".join(parts) or self.detail
        return ("%s: %s. Presence past %s reads as an EMPTY ROOM and nothing "
                "else says so. Fix:  %s"
                % (head, body, gate_range(self.got_move
                                          if self.got_move is not None
                                          else self.got_still),
                   fix_command(self.room)))


def fix_command(room: str) -> str:
    """The exact line to paste. Built from THIS checkout so it is the
    script that goes with the code that printed it."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "room_sensor.py"
    return "%s tune %s" % (script, room)


# ------------------------------------------------------------- the reading
def _as_gate(value) -> Optional[int]:
    """A read-back number as a gate, or None. ESPHome serves these as
    floats; a value outside 0..MAX_GATE is not a gate this module knows how
    to talk about and is treated as no answer rather than as a fault."""
    if value is None:
        return None
    try:
        gate = int(round(float(value)))
    except (OverflowError, TypeError, ValueError):
        return None
    return gate if 0 <= gate <= MAX_GATE else None


def read_gates(sensor) -> Tuple[Optional[int], Optional[int]]:
    """The device's own two gates, each None when it would not say."""
    return (_as_gate(sensor.read_number(MOVE_GATE_ENTITY)),
            _as_gate(sensor.read_number(STILL_GATE_ENTITY)))


def _verdict(want: Tuple[int, int],
             got: Tuple[Optional[int], Optional[int]]) -> str:
    """OK / MISMATCH / UNCHECKED for ONE reading against the profile.

    Only the entities that ANSWERED are compared: a firmware missing one of
    the two must not silence the other. Nothing answering at all is
    UNCHECKED, never a mismatch -- a device that is not talking is not a
    device that has been reset.
    """
    pairs = [(w, g) for w, g in zip(want, got) if g is not None]
    if not pairs:
        return UNCHECKED
    return OK if all(w == g for w, g in pairs) else MISMATCH


def _blocked(policy) -> str:
    """Why the radar may not be polled right now ("" = it may).

    Mirrors ``RoomSensor.blocked`` INCLUDING the rule that a policy which
    raises counts as forbidden. What it does not do is attach.
    """
    if policy is None:
        return ""
    try:
        return "" if policy.allowed(RADAR) else "offline mode is on"
    except Exception:  # noqa: BLE001 - a broken policy is not a yes
        log.debug("sensor check: the sensing policy failed", exc_info=True)
        return "the sensing policy would not answer"


def _sensor_for(url: str, get: Optional[Callable] = None) -> RoomSensor:
    """One read-only sensor for one device.

    No policy (attach replaces by name), no power_url, and a post that
    raises. ``fail_after=1`` because this asks each entity once or twice
    and then goes away: the breaker's job here is only to stop the second
    read of a device that did not answer the first.
    """
    kw: Dict[str, Any] = {} if get is None else {"get": get}
    return RoomSensor(url, timeout_s=TIMEOUT_S, post=_refuse_post,
                      fail_after=1, **kw)


def check_room(room: str, url: str, want: Tuple[int, int], *,
               get: Optional[Callable] = None,
               sleep: Callable[[float], Any] = time.sleep,
               confirm_delay_s: float = CONFIRM_DELAY_S) -> RoomCheck:
    """One room: read, and read again before saying anything loud."""
    sensor = _sensor_for(url, get)
    first = read_gates(sensor)
    state = _verdict(want, first)
    if state == OK:
        return RoomCheck(room, OK, want[0], want[1], first[0], first[1])
    if state == UNCHECKED:
        return RoomCheck(room, UNCHECKED, want[0], want[1],
                         detail="the device did not answer for its gates")
    # It disagreed once. That buys a second read and nothing else: the
    # readback is unreliable shortly after the device powers up, and the
    # office was in exactly this state while it was perfectly fine.
    sleep(max(0.0, float(confirm_delay_s)))
    second = read_gates(sensor)
    again = _verdict(want, second)
    if again == OK:
        return RoomCheck(room, OK, want[0], want[1], second[0], second[1],
                         detail="the first readback disagreed and the second "
                                "did not")
    if again == UNCHECKED:
        return RoomCheck(room, UNCHECKED, want[0], want[1],
                         detail="it stopped answering before the second read")
    if second != first:
        return RoomCheck(room, UNSETTLED, want[0], want[1], second[0], second[1],
                         detail="two reads gave two different answers")
    return RoomCheck(room, MISMATCH, want[0], want[1], second[0], second[1])


def rooms_to_check(cfg) -> List[Any]:
    """Every room the app would actually poll. Never raises.

    A pure config read -- no file, no socket -- so it is cheap enough for
    ``start()`` to ask it on the CALLING thread and skip the thread
    entirely when there is nothing to check. That matters beyond tidiness:
    without it every construction of ``JarvisApp`` spawns a thread that
    sleeps ``START_DELAY_S``, and the suite constructs a great many.
    """
    try:
        from jarvis.roomfabric import room_specs
        return [s for s in (room_specs(cfg) or ())
                if getattr(s, "name", "") and getattr(s, "url", "")]
    except Exception:  # noqa: BLE001 - a check may not cost the caller
        log.debug("sensor check: the room fabric is unavailable", exc_info=True)
        return []


def check_all(cfg, *, get: Optional[Callable] = None, policy: Any = None,
              profile_dir: Optional[Path] = None,
              sleep: Callable[[float], Any] = time.sleep,
              confirm_delay_s: float = CONFIRM_DELAY_S) -> List[RoomCheck]:
    """Every configured room sensor, checked once. Never raises.

    Rooms come from ``roomfabric.room_specs`` -- the fabric's own reader --
    so this checks exactly what the app polls, singular keys and all, and a
    room the app would not poll is not checked or complained about.
    """
    blocked = _blocked(policy)
    out: List[RoomCheck] = []
    for spec in rooms_to_check(cfg):
        room = getattr(spec, "name", "")
        url = getattr(spec, "url", "")
        want = wanted_gates(read_profile(room, profile_dir))
        if want is None:
            # room_sensor.py owns the profiles. A room that was never tuned
            # through it has nothing to compare against, and an invented
            # expectation would be a false alarm every boot.
            out.append(RoomCheck(room, UNCHECKED,
                                 detail="there is no tuning profile at %s"
                                        % profile_path(room, profile_dir)))
            continue
        if blocked:
            # BEFORE the socket. "the readings are ignored" is a weaker
            # promise than "the radar was not polled".
            out.append(RoomCheck(room, UNCHECKED, want[0], want[1],
                                 detail="offline mode: %s" % blocked))
            continue
        try:
            out.append(check_room(room, url, want, get=get, sleep=sleep,
                                  confirm_delay_s=confirm_delay_s))
        except Exception:  # noqa: BLE001 - a check may not cost the caller
            log.debug("sensor check: %s could not be checked", room,
                      exc_info=True)
            out.append(RoomCheck(room, UNCHECKED, want[0], want[1],
                                 detail="the check itself failed"))
    _report(out)
    return out


def _report(checks: List[RoomCheck]) -> None:
    """Say it once, at the level the finding deserves.

    A confirmed mismatch is at ERROR: it is a fault that silently disables
    presence for the room, and it names the one command that fixes it.
    Everything else is at INFO or below -- an unsettled device and a
    missing profile are both "I could not tell", and a line at WARNING for
    either would train him to skim the level that matters.
    """
    for check in checks:
        text = check.message()
        if not text:
            continue
        (log.error if check.loud else log.info)("%s", text)


# ------------------------------------------------------------------ thread
def start(cfg, *, policy: Any = None, delay_s: float = START_DELAY_S,
          sleep: Callable[[float], Any] = time.sleep,
          **kw) -> Optional[threading.Thread]:
    """Run the check on a daemon thread after ``delay_s``. NEVER blocks.

    ``None`` when there is nothing to check -- no configured room sensor,
    or a config that cannot be read at all. That is decided HERE, on the
    calling thread, out of a pure config read: a box with no radar must not
    pay a thread and a 20-second sleep to discover it has no radar.

    The delay is the fresh-device problem again: a readback taken the
    instant the house comes back up can lie, so the first read waits, and
    the confirm read inside ``check_room`` covers the case where the wait
    was not long enough. The thread is a daemon and nobody joins it -- an
    exception anywhere inside costs the check and not the boot.
    """
    if not rooms_to_check(cfg):
        return None
    def run() -> None:
        try:
            if delay_s:
                sleep(float(delay_s))
            check_all(cfg, policy=policy, sleep=sleep, **kw)
        except Exception:  # noqa: BLE001 - a check may not cost the boot
            log.debug("sensor check: the gate check failed", exc_info=True)

    thread = threading.Thread(target=run, daemon=True, name=THREAD_NAME)
    thread.start()
    return thread


__all__ = ["CONFIRM_DELAY_S", "GATE_ENTITIES", "GATE_M", "MAX_GATE",
           "MISMATCH", "MOVE_GATE_ENTITY", "OK", "ReadOnly", "RoomCheck",
           "START_DELAY_S", "STILL_FLOOR_GATE", "STILL_GATE_ENTITY",
           "UNCHECKED", "UNSETTLED", "check_all", "check_room", "fix_command",
           "gate_for_metres", "gate_range", "profile_dir", "profile_path",
           "read_gates", "read_profile", "start", "wanted_gates"]
