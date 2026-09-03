"""Room sensor: does the ROOM see a person, right now?

``jarvis/presence.py`` answers "is he in the building" from his phone's
Wi-Fi address, and that answer is slow and occasionally a lie: an iPhone
drops off Wi-Fi power-save for minutes, so the honest away grace is twelve
minutes and the arrival edge lands whenever the radio next wakes. This
module reads a second, faster leg -- an ESP32 + HLK-LD2410 mmWave module
sitting in the room, exposed over the LAN by ESPHome's ``web_server``.

**mmWave, not PIR, and the difference is the whole point.** A PIR fires on
MOTION: it says "gone" the moment he stops moving, so a man reading in a
chair is an empty room within a minute. The LD2410 is a PRESENCE radar --
it reports a still target from chest movement and breathing -- so "nobody
there" survives him sitting still, which is what "is he home" actually
needs.

TRANSPORT: one HTTP GET of one entity, stdlib only, no broker to install
and no new dependency in the shared venv. ESPHome's ``web_server`` (version
2, which is what both templates ask for) serves each entity as JSON at its
NAME, percent-encoded -- NOT at the snake_case object_id, and NOT at the
``id:`` the YAML gives it. MEASURED against the live office radar,
2026-09-03::

    $ curl http://192.168.50.51/binary_sensor/Presence
    {"id":"binary_sensor/Presence","value":true,"state":"ON"}
    $ curl -o /dev/null -w '%{http_code}\n' http://192.168.50.51/binary_sensor/presence
    404
    $ curl 'http://192.168.50.51/sensor/Moving%20distance'
    {"id":"sensor/Moving distance","value":42,"state":"42 cm"}

This module and ``scripts/room_sensor.py`` both built the object_id form,
so every poll and every tune write was a 404: presence never once fired,
and it degraded to "no opinion" silently, exactly as a missing sensor
would. Build a path with ``entity_path()`` and nothing else.

THREE RULES the rest of the app depends on.

* ``read()`` returns ``True`` / ``False`` / ``None``, and ``None`` is "no
  opinion", NEVER "empty". A timeout, a refused connection, a 404, HTML
  from the wrong URL -- all of it is None, and the caller falls back to
  the phone leg exactly as if this module did not exist. A sensor that
  goes offline must degrade to phone-only; a false "away" makes Jarvis
  hold his proactive speech and go quiet on him, which is the worst
  outcome available here.
* **The breaker is a latency guarantee, not an optimisation.** After
  ``fail_after`` consecutive failures the sensor is skipped entirely for a
  growing cooldown, so a dead ESP32 costs the poll loop ZERO milliseconds
  and zero syscalls per tick rather than a timeout apiece. That is what
  makes "unplug it and nothing changes" literally true.
* **Silence when it is down.** One warning when the breaker opens, one
  info line when it recovers; every other failure is debug. A sensor that
  has been unplugged for a week must not write a line a minute into the
  log the user reads first.

OFFLINE MODE (2026-09-02, jarvis/sensing.py). A ``policy`` may be handed
in, and then it is asked BEFORE the socket: while sensing is denied
``read()`` issues no request at all, which is why ``reads`` is the
assertion the test makes -- "the readings are ignored" is not the same
promise as "the radar was not polled", and only the second one is worth
anything to him. ``power_url`` goes one step further and cuts the device:
ESPHome serves a switch at ``POST /switch/<name>/turn_off``, so with a
GPIO holding the LD2410's supply (the optional block in
scripts/esphome/jarvis-room-sensor.yaml) "offline" means the radar stops
radiating, not merely that nobody is listening.

**Without that wire the honest limit is that we stop asking**, and that is
the configuration on this box today: nothing is flashed, no MOSFET is
wired, so ``stop()`` returns ``sensing.POLLING_ONLY`` -- truthy, because
the stop did everything this process can do, and NOT ``True``, because the
LD2410 keeps radiating and keeps serving presence to anyone on the LAN.
The spoken confirmation renders that bucket as "I've stopped reading the
radar, but its power isn't switched", which is the sentence he would
actually want at 11pm; "the radar is down" would be a promise.

Nothing here knows about presence, hysteresis or the bus: this module
answers one question and holds no history, so the composition (and the
decision about what to do when the two legs disagree) lives in
``jarvis/presence.py`` where the hysteresis it has to respect already is.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from jarvis.logs import get_logger
from jarvis.sensing import POLLING_ONLY, RADAR

log = get_logger("roomsensor")

# The entity whose name IS the URL. Renaming it in the YAML moves the
# endpoint, so the name lives here as a constant and the path is derived
# from it -- never spelled out a second time.
PRESENCE_ENTITY = "Presence"


def entity_path(domain: str, name: str) -> str:
    """The web_server path for an entity, from its NAME.

    ``entity_path("binary_sensor", "Presence") == "/binary_sensor/Presence"``
    and ``entity_path("sensor", "Moving distance")`` percent-encodes the
    space. This is the only place that rule is written down; see the
    measurement in the module docstring for why it is the name and not the
    object_id.
    """
    return "/%s/%s" % (domain, urllib.parse.quote(str(name), safe=""))


DEFAULT_ENTITY_PATH = entity_path("binary_sensor", PRESENCE_ENTITY)

# The three DISTANCE entities, for jarvis/zones.py. Names, not object_ids,
# for the reason spelled out above; ESPHome's ld2410 publishes them in
# CENTIMETRES ({"id":"sensor/Moving distance","value":42,"state":"42 cm"},
# measured 2026-09-03) and read_distance() returns metres.
DETECTION_ENTITY = "Detection distance"
MOVING_ENTITY = "Moving distance"
STILL_ENTITY = "Still distance"
# MEASURED against the live office radar, 25 polls at 4 Hz, 2026-09-03:
# min 46 ms, median 62 ms, p90 154 ms, MAX 1186 ms. The old value here was
# 1.5 s with the comment "LAN round trip is ~5 ms; this is pure paranoia" --
# it was neither. An ESP32 in Wi-Fi modem-sleep parks a round trip for the
# best part of a second, so 1.5 s was 1.3x the worst sample, not 300x it.
# 3.0 s is ~2.5x the worst seen. A slow poll is not costly: read() returns
# None, the phone leg answers, and only DEFAULT_FAIL_AFTER in a row is a fault.
DEFAULT_TIMEOUT_S = 3.0
DEFAULT_FAIL_AFTER = 3      # transients are free; three in a row is a fault
DEFAULT_COOLDOWN_S = 30.0
MAX_COOLDOWN_S = 300.0
MAX_BYTES = 4096            # an entity is ~60 bytes; anything else is wrong
USER_AGENT = "jarvis-roomsensor/1"

# "242 cm", "242", "2.42 cm" -- the leading number of a state string.
_CM_RX = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*(?:cm)?$", re.I)

_TRUE_WORDS = frozenset({"on", "true", "yes", "1", "occupied", "present",
                         "detected", "home", "active"})
_FALSE_WORDS = frozenset({"off", "false", "no", "0", "clear", "empty",
                          "unoccupied", "away", "none", "idle"})


def normalize_url(value: str, entity_path: str = DEFAULT_ENTITY_PATH) -> str:
    """The configured URL, or "" when it is not one.

    The scheme is REQUIRED (a bare "192.168.50.60" is rejected rather than
    guessed at) because the config value is also the thing a typo lands
    in, and a silently-invented URL would fail as a timeout every poll
    instead of as one loud line at startup. A URL with no path gets the
    default entity path appended, so pasting the device address with the
    scheme is enough.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return ""
    path = parts.path or ""
    if path in ("", "/"):
        parts = parts._replace(path=entity_path)
    return urllib.parse.urlunsplit(parts)


def parse_state(body: Any) -> Optional[bool]:
    """An ESPHome entity body -> True / False / None ("I cannot tell").

    Deliberately generous about the shape, because the point of an HTTP
    leg is that he can point it at something else later (a Home Assistant
    template, a shell one-liner behind netcat) without touching Python:
    ``{"value": true}``, ``{"state": "ON"}``, a bare ``true``, or the
    plain text ``ON`` all read the same. Anything else -- an HTML error
    page, an empty body, a number that is not a flag -- is None, which the
    caller treats as no opinion rather than as absence.
    """
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, bool):
        return body
    if isinstance(body, (int, float)):
        return bool(body)
    if isinstance(body, dict):
        return _from_mapping(body)
    text = str(body or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return _from_word(text)
    # Deliberately NOT a blind recursion: json.loads("[1,2]") gives a list
    # whose str() parses back to a list, which recursed forever.
    if isinstance(data, str):
        return _from_word(data)
    if isinstance(data, (bool, int, float, dict)):
        return parse_state(data)
    return None


def parse_cm(body: Any) -> Optional[float]:
    """An ESPHome distance entity -> centimetres, or None.

    ``{"value": 242}`` is the typed field and wins; ``{"state": "242 cm"}``
    is the display string and is the fallback for a firmware that omits
    value. A null value (the radar has no target) is None, and so is a
    negative number, HTML from a wrong URL, or anything else -- None here
    means "no distance", never zero, because zero is a real reading that
    means "no target of this kind" and the two must not merge.
    """
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, bool):
        return None                       # a flag is not a distance
    if isinstance(body, (int, float)):
        return float(body) if body >= 0 else None
    if isinstance(body, dict):
        for key in ("value", "state"):
            if key in body:
                got = parse_cm(body[key])
                if got is not None:
                    return got
        return None
    text = str(body or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        pass
    else:
        if isinstance(data, (bool, int, float, dict)):
            return parse_cm(data)
        if isinstance(data, str):
            text = data
        else:
            return None
    m = _CM_RX.match(text.strip())
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if value >= 0 else None


def _from_mapping(data: dict) -> Optional[bool]:
    # "value" is the typed field and wins; "state" is ESPHome's display
    # string ("ON") and is the fallback for a firmware that omits value.
    for key in ("value", "state"):
        if key in data:
            got = parse_state(data[key])
            if got is not None:
                return got
    return None


def _from_word(text: str) -> Optional[bool]:
    word = text.strip().strip('"').strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return None


def _get_default(url: str, timeout: float) -> str:
    """The one transport. ``RoomSensor(get=...)`` is the test seam."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - LAN http
        return resp.read(MAX_BYTES).decode("utf-8", "replace")


def _post_default(url: str, timeout: float) -> None:
    """ESPHome switches take a POST with no body. ``RoomSensor(post=...)``
    is the seam; a failure RAISES so the caller can say the radar may
    still be powered."""
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - LAN http
        resp.read(MAX_BYTES)


class RoomSensor:
    """One mmWave endpoint. ``read()`` is the whole interface."""

    def __init__(self, url: str, timeout_s: float = DEFAULT_TIMEOUT_S,
                 get: Callable[[str, float], Any] = _get_default,
                 now: Optional[Callable[[], float]] = None,
                 fail_after: int = DEFAULT_FAIL_AFTER,
                 cooldown_s: float = DEFAULT_COOLDOWN_S,
                 policy: Any = None, power_url: str = "",
                 post: Optional[Callable[[str, float], Any]] = None):
        self.url = normalize_url(url)
        try:
            self.timeout_s = max(0.1, float(timeout_s))
        except (TypeError, ValueError):
            self.timeout_s = DEFAULT_TIMEOUT_S
        self._get = get
        self._now = now or time.monotonic  # monotonic: immune to clock jumps
        self.fail_after = max(1, int(fail_after or DEFAULT_FAIL_AFTER))
        self._base_cooldown = max(1.0, float(cooldown_s or DEFAULT_COOLDOWN_S))
        self._cooldown = self._base_cooldown
        self._fails = 0
        self._skip_until = 0.0
        self._down = False              # has the breaker's warning been logged?
        self.last_value: Optional[bool] = None
        self.last_ok: Optional[float] = None
        self.reads = 0                  # requests actually sent (the breaker's proof)
        # Offline mode. The policy is the ONE authority (jarvis/sensing.py);
        # _stopped is only the no-policy case, so a stop() on a bare sensor
        # still means something.
        self._policy = policy
        self._stopped = False
        self.power_url = str(power_url or "").strip().rstrip("/")
        self._post = post or _post_default
        self._power_warned = False      # one loud line per outage, not per retry
        # The DISTANCE leg (jarvis/zones.py) gets its OWN failure counter,
        # deliberately separate from the presence breaker above. A distance
        # entity that 404s -- a renamed entity, a firmware without
        # detection_distance -- would otherwise open the SHARED breaker and
        # take PRESENCE down with it, and presence is the one thing this
        # module exists to provide. So a bad distance costs the distance
        # and nothing else: three failures in a row and it stops asking for
        # a cooldown, with one warning line.
        self._dist_fails = 0
        self._dist_skip_until = 0.0
        self._dist_down = False
        attach = getattr(policy, "attach", None)
        if callable(attach):
            attach(RADAR, self.stop, present=lambda: self.configured,
                   resume=self.resume)

    @property
    def configured(self) -> bool:
        return bool(self.url)

    @property
    def blocked(self) -> str:
        """Why sensing is forbidden right now ("" = it is not).

        A policy that RAISES counts as forbidden. That is the fail-safe
        reaching the wire: a decision we could not make is not permission.
        """
        policy = self._policy
        if policy is not None:
            try:
                return "" if policy.allowed(RADAR) else "offline"
            except Exception:  # noqa: BLE001 - a broken policy is not a yes
                log.exception("room sensor: the sensing policy failed; "
                              "treating the radar as offline")
                return "policy"
        return "stopped" if self._stopped else ""

    @property
    def paused(self) -> bool:
        """True while read() will not touch the network -- the breaker is
        open, or offline mode forbids the radar entirely."""
        return bool(self.blocked) or self._skip_until > self._now()

    # ------------------------------------------------------ offline mode
    def _power(self, on: bool) -> bool:
        """Flip the device's own power switch, when one is wired."""
        url = "%s/turn_%s" % (self.power_url, "on" if on else "off")
        try:
            self._post(url, self.timeout_s)
        except Exception:  # noqa: BLE001 - every transport failure is a NO
            if not self._power_warned:
                # SensingPolicy.enforce retries a failed stop on every pass,
                # so a traceback per attempt would bury the log this module
                # promises to stay quiet in when the device is unreachable.
                log.exception("room sensor: %s failed; the radar may still "
                              "be powered", url)
                self._power_warned = True
            else:
                log.debug("room sensor: %s failed again", url, exc_info=True)
            return False
        self._power_warned = False
        log.info("room sensor: radar powered %s via %s",
                 "on" if on else "off", url)
        return True

    def stop(self):
        """Stop sensing. ``True`` only when the radar is ACTUALLY down.

        With no ``power_url`` the most this process can do is never ask
        again, and that is reported as ``sensing.POLLING_ONLY`` rather than
        as success: the module keeps radiating and keeps answering the LAN,
        and "the radar is down" would be a promise nobody kept. Truthy, so a
        caller that only wants "did the stop work" still reads yes.
        """
        self._stopped = True
        return self._power(False) if self.power_url else POLLING_ONLY

    def resume(self) -> bool:
        """Undo stop(). True when the radar is back."""
        self._stopped = False
        return self._power(True) if self.power_url else True

    def read(self) -> Optional[bool]:
        """True (someone is in the room) / False (nobody) / None (no opinion)."""
        if not self.configured or self.paused:
            return None
        try:
            body = self._get(self.url, self.timeout_s)
        except Exception as exc:  # noqa: BLE001 - every transport failure is "unknown"
            self.reads += 1
            self._failed("unreachable", exc)
            return None
        self.reads += 1
        value = parse_state(body)
        if value is None:
            # Garbage counts against the breaker too: a device serving the
            # wrong URL will serve it forever, and polling HTML every ten
            # seconds for a week is the same waste as polling a dead host.
            self._failed("unreadable", repr(body)[:120])
            return None
        self._ok()
        self.last_value = value
        return value

    def read_distance(self, entity: str = DETECTION_ENTITY) -> Optional[float]:
        """The named distance entity in METRES, or None.

        ESPHome publishes these in centimetres; the conversion happens here
        so no caller has to remember it. None is "no distance" -- offline
        mode, an open breaker, a timeout, a 404, a null value -- and is
        NEVER 0.0, which is a real reading meaning "no target of this kind"
        and is what the moving/still bits are derived from.

        It asks the same ``blocked``/``paused`` gate as ``read()`` BEFORE
        the socket, so offline mode covers it for free: while sensing is
        denied no request is sent and ``reads`` does not move. A distance
        read of anyone's own over urllib would have been a hole straight
        through that promise, which is why this lives here and not in
        jarvis/zones.py.
        """
        if not self.configured or self.paused:
            return None
        if self._dist_skip_until > self._now():
            return None
        url = urllib.parse.urlunsplit(
            urllib.parse.urlsplit(self.url)._replace(
                path=entity_path("sensor", entity), query="", fragment=""))
        try:
            body = self._get(url, self.timeout_s)
        except Exception as exc:  # noqa: BLE001 - every transport failure is "unknown"
            self.reads += 1
            self._dist_failed(entity, exc)
            return None
        self.reads += 1
        cm = parse_cm(body)
        if cm is None:
            self._dist_failed(entity, repr(body)[:120])
            return None
        self._dist_fails, self._dist_skip_until = 0, 0.0
        if self._dist_down:
            log.info("room sensor %s: %s is back", self.url, entity)
            self._dist_down = False
        return cm / 100.0

    def _dist_failed(self, entity: str, detail: Any) -> None:
        """Counts against the DISTANCE leg only -- never the presence
        breaker. See the note in __init__."""
        self._dist_fails += 1
        if self._dist_fails < self.fail_after:
            log.debug("room sensor distance %s: %s (%s)", entity, self.url, detail)
            return
        self._dist_skip_until = self._now() + self._base_cooldown
        if not self._dist_down:
            log.warning("room sensor %s: %r is unreadable (%s); zones will be "
                        "unplaced, presence is unaffected",
                        self.url, entity, detail)
            self._dist_down = True
        else:
            log.debug("room sensor distance %s: %s; retrying in %.0fs",
                      entity, self.url, self._base_cooldown)

    # ------------------------------------------------------------ breaker
    def _failed(self, kind: str, detail: Any) -> None:
        self._fails += 1
        if self._fails < self.fail_after:
            log.debug("room sensor %s: %s (%s)", kind, self.url, detail)
            return
        self._skip_until = self._now() + self._cooldown
        if not self._down:
            # The ONE line a dead sensor is allowed. Says what happens
            # next, because "unreachable" alone reads like a lost feature.
            log.warning("room sensor %s: %s (%s); phone probe only, "
                        "retrying in %.0fs", kind, self.url, detail, self._cooldown)
            self._down = True
        else:
            log.debug("room sensor %s: %s; retrying in %.0fs",
                      kind, self.url, self._cooldown)
        self._cooldown = min(self._cooldown * 2.0, MAX_COOLDOWN_S)

    def _ok(self) -> None:
        if self._down:
            log.info("room sensor %s: back", self.url)
        self._fails, self._skip_until, self._down = 0, 0.0, False
        self._cooldown = self._base_cooldown
        self.last_ok = self._now()

    def status(self) -> dict:
        """For the console / a diagnostic script; never parsed by the app."""
        return {"url": self.url, "value": self.last_value, "fails": self._fails,
                "paused": self.paused, "cooldown_s": self._cooldown,
                "reads": self.reads, "last_ok": self.last_ok,
                "blocked": self.blocked,
                "power_url": self.power_url}
