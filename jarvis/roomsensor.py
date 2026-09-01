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
and no new dependency in the shared venv. ESPHome's ``web_server`` serves
each entity as JSON at ``/<domain>/<object_id>``::

    $ curl http://192.168.50.60/binary_sensor/presence
    {"id":"binary_sensor-presence","value":true,"state":"ON"}

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

Nothing here knows about presence, hysteresis or the bus: this module
answers one question and holds no history, so the composition (and the
decision about what to do when the two legs disagree) lives in
``jarvis/presence.py`` where the hysteresis it has to respect already is.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from jarvis.logs import get_logger

log = get_logger("roomsensor")

# ESPHome names the endpoint after the entity: a binary_sensor called
# "Presence" is /binary_sensor/presence. The YAML in
# scripts/esphome/jarvis-room-sensor.yaml uses exactly this name.
DEFAULT_ENTITY_PATH = "/binary_sensor/presence"
DEFAULT_TIMEOUT_S = 1.5     # LAN round trip is ~5 ms; this is pure paranoia
DEFAULT_FAIL_AFTER = 3      # transients are free; three in a row is a fault
DEFAULT_COOLDOWN_S = 30.0
MAX_COOLDOWN_S = 300.0
MAX_BYTES = 4096            # an entity is ~60 bytes; anything else is wrong
USER_AGENT = "jarvis-roomsensor/1"

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


class RoomSensor:
    """One mmWave endpoint. ``read()`` is the whole interface."""

    def __init__(self, url: str, timeout_s: float = DEFAULT_TIMEOUT_S,
                 get: Callable[[str, float], Any] = _get_default,
                 now: Optional[Callable[[], float]] = None,
                 fail_after: int = DEFAULT_FAIL_AFTER,
                 cooldown_s: float = DEFAULT_COOLDOWN_S):
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

    @property
    def configured(self) -> bool:
        return bool(self.url)

    @property
    def paused(self) -> bool:
        """True while the breaker is open: read() will not touch the network."""
        return self._skip_until > self._now()

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
                "reads": self.reads, "last_ok": self.last_ok}
