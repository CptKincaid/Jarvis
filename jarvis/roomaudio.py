"""Which room Jarvis speaks in, and how he knows the voice landed.

Design: ``docs/multiroom-audio.md``. This module is the AUDIO half of the
three-room build and deliberately owns nothing the sensing half owns.
``jarvis/rooms.py`` is the satellite mesh -- the lease, the power
confirmation, and ``Satellite.read()``'s per-room "is someone in THIS
room". This module CONSUMES that as ``{room: True | False | None}`` and
never computes one; ``occupancy_from`` is the adapter and it is the only
thing here that knows the mesh's shape.

The two modules share the config block (``rooms.satellites``), the URL
trust rule (``rooms.check_url``: an http(s) PRIVATE IP LITERAL, never a
hostname) and the room names. They do not share a file, because a
satellite's radar and a satellite's speaker are allowed to be two
different boxes in one room -- the ESP32 on port 80 and the Pi on 8765.

TWO IDEAS, AND THEY ARE SEPARABLE.

``VoiceRouter`` is the policy, and it is PURE -- a dict in, a ``Decision``
out, no clock but the one it is handed and no network at all. The rule::

    an ANSWER goes back where the question came from;
    an ANNOUNCEMENT goes where he is;
    when nobody knows where he is it goes to `here`, never everywhere;
    when he is out it is HELD, exactly as jarvis/quiet.py holds it today.

Three refusals are load-bearing:

* **``None`` is no opinion, never "empty".** ``RoomSensor.read`` already
  makes that distinction and ``presence.RoomOrPhone`` already honours it;
  a router that read an offline radar as an empty room would relocate his
  assistant every time a sensor was unplugged.
* **Never two rooms at once**, except an alarm. Two speakers saying the
  same sentence forty milliseconds apart is the comb filter that makes
  multi-room sound broken, and an announcement has one listener.
* **Never broadcast on ignorance.** Unknown resolves to ``here``, which is
  the room the Spark is in. Shouting in three rooms because a sensor is
  down is how a house wakes someone at two in the morning.

``RoomSpeaker`` is one remote room's ``/say`` endpoint, and it exists for
the reason ``jarvis/soundbar.py`` exists. On 2026-08-30 a dead Bluetooth
speaker made PipeWire move the default sink to the HDMI monitor and Jarvis
talked into it for hours; ``pactl`` saw a perfectly healthy sink the whole
time, because it can see a SINK and not a SPEAKER. Multiplied by three
rooms that is three ways to talk to a wall, so every remote room owes a
RECEIPT: ``played_ms`` short of the clip's duration means it was cut off,
and a ``peak`` at the floor means it played into a muted output -- the
remote equivalent of the monitor behind the desk, and the one signal
``pactl`` never had. Two bad receipts in a row and the room is dropped as
a target rather than spoken into hopefully.

The transport discipline is copied from ``jarvis/roomsensor.py`` on
purpose: stdlib ``urllib``, a hard timeout, and a breaker that after
``fail_after`` consecutive failures skips the room entirely for a growing
cooldown -- so an unplugged kitchen costs the loop ZERO syscalls per tick
instead of a timeout apiece, and one warning line instead of one a minute.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from jarvis import rooms as roommesh
from jarvis.logs import get_logger

log = get_logger("roomaudio")

# The three classes of speech. They route differently and the difference
# is the whole policy; see docs/multiroom-audio.md section 2.
ANSWER = "answer"        # a reply to something he just said
ANNOUNCE = "announce"    # proactive: a timer, a reminder, a heads-up
ALARM = "alarm"          # the one line whose job is to be heard anywhere
KINDS = (ANSWER, ANNOUNCE, ALARM)

DEFAULT_HERE = "office"
# How recently a room must have heard a turn to win a two-occupied-rooms
# tie. Five minutes: long enough to cover a question, a reply and a
# follow-up, short enough that yesterday's kitchen turn does not decide
# tonight's reminder.
DEFAULT_RECENCY_S = 300.0

DEFAULT_TIMEOUT_S = 4.0     # a wav POST, not a sensor read: bigger than 1.5
DEFAULT_FAIL_AFTER = 2      # a speaker gets fewer chances than a sensor
DEFAULT_COOLDOWN_S = 30.0
MAX_COOLDOWN_S = 300.0
MAX_BYTES = 4096            # a receipt is ~80 bytes; anything else is wrong
USER_AGENT = "jarvis-roomaudio/1"
SAY_PATH = "/say"
# A reply that played less than half of what was sent did not land. Two of
# those in a row is the silence audit that the 2026-08-30 incident had no
# way to make: hours of speech, nothing emitted, nothing comparing the two.
MIN_PLAYED_RATIO = 0.5


def say_url(value: Any, path: str = SAY_PATH) -> str:
    """The configured ``/say`` endpoint, or "".

    The trust rule is ``rooms.check_url``'s, not a second one: an
    ``http(s)://`` PRIVATE IP LITERAL, no hostnames (an mDNS answer is one
    poisoned packet away from pointing his voice at somebody else's box)
    and no public addresses. Reusing it means there is ONE definition of
    "a satellite address I will talk to" for the whole three-room build,
    rather than a strict one on the privacy path and a lax one on the
    audio path -- and this endpoint carries the same bearer token, so the
    lax one would be the hole.

    An empty value is legal and MEANINGFUL: the room has no speaker of its
    own. Only ``rooms.here`` -- the room the Spark is in -- speaks without
    a URL, out of the existing ``paplay`` path.
    """
    base = roommesh.check_url(value)
    if not base:
        return ""
    parts = urllib.parse.urlsplit(base)
    if (parts.path or "") in ("", "/"):
        parts = parts._replace(path=path)
    return urllib.parse.urlunsplit(parts)


@dataclass(frozen=True)
class Room:
    """One room's audio identity, keyed by the SAME name
    ``rooms.RoomSpec`` uses so the two halves cannot disagree about what a
    room is called -- that name is what the spoken line says out loud.

    ``url`` empty means the room has no speaker of its own. The one room
    allowed to be targeted anyway is ``rooms.here``, which is this box: the
    existing ``paplay`` path. That is what makes a one-room config a
    no-op, and why the office is a room like any other rather than a
    special case carved around.
    """
    name: str
    url: str = ""
    private: bool = False   # never a broadcast target, never gets the digest
    enabled: bool = True


@dataclass(frozen=True)
class Decision:
    """Where a line goes, and why. ``reason`` is for the log and the
    console; nothing parses it."""
    rooms: tuple = ()
    reason: str = ""
    held: bool = False

    def __bool__(self) -> bool:
        return bool(self.rooms)

    @property
    def room(self) -> str:
        """The single target, or "" -- the common case, since only an
        alarm ever names more than one."""
        return self.rooms[0] if len(self.rooms) == 1 else ""


def rooms_from_config(cfg) -> tuple:
    """``({name: Room}, here)`` from assistant.json. Never raises.

    Reads the SAME ``rooms.satellites`` list ``rooms.specs_from_config``
    reads, plus two audio-only keys per entry -- ``say_url`` and
    ``private`` -- and the house-level ``rooms.here``. One list, so a room
    cannot exist for the radar and not for the voice, and a room name
    cannot be spelled two ways.

    ``here`` is always present as a Room even when it is not in the
    satellite list: the Spark's own room has no satellite, and it is the
    fallback every other rule falls through to.

    Tolerant on purpose. A satellite with a bad ``say_url`` keeps its room
    (announcements for it fall back to ``here``, so he still hears them)
    rather than vanishing, because a silently missing room is a line he
    never hears and never finds out about. The mistake is logged once.
    """
    get = getattr(cfg, "get", None)
    if not callable(get):
        return {}, DEFAULT_HERE
    try:
        here = str(get("rooms.here", DEFAULT_HERE) or DEFAULT_HERE).strip()
        enabled = bool(get("rooms.enabled", False))
        raw = get("rooms.satellites", []) or []
    except Exception:  # noqa: BLE001 - a broken config must not lose his voice
        log.exception("roomaudio: config unreadable; speaking here only")
        return {DEFAULT_HERE: Room(name=DEFAULT_HERE)}, DEFAULT_HERE
    here = here or DEFAULT_HERE
    out = {here: Room(name=here)}
    if not enabled:
        return out, here
    for entry in raw if isinstance(raw, (list, tuple)) else ():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "") or "").strip()
        if not name:
            continue
        want = str(entry.get("say_url", "") or "").strip()
        url = say_url(want)
        if want and not url:
            log.warning("roomaudio: %s say_url %r is not a private http(s) "
                        "address; that room falls back to %s", name, want, here)
        out[name] = Room(name=name, url=("" if name == here else url),
                         private=bool(entry.get("private", False)),
                         enabled=bool(entry.get("enabled", True)))
    return out, here


def occupancy_from(mesh) -> dict:
    """``rooms.RoomMesh`` -> ``{room: True | False | None}``.

    The ONLY thing in this module that knows the mesh's shape, and it is
    deliberately defensive: the mesh is another lane's file, and a router
    that raised because an attribute was renamed would take his voice down
    with it.

    NOT for the reply path. ``Satellite.read()`` polls, and a spoken turn
    may not wait on a LAN round trip per room (jarvis/aside.py's rule).
    Call it on the mesh's own thread and hand the snapshot to
    ``VoiceRouter.observe``; ``route`` then costs nothing.
    """
    out: dict = {}
    for sat in list(getattr(mesh, "satellites", ()) or ()):
        name = str(getattr(getattr(sat, "spec", None), "name", "") or "").strip()
        if not name:
            continue
        try:
            out[name] = sat.read()
        except Exception:  # noqa: BLE001 - one bad room must not lose the rest
            log.debug("roomaudio: %s read failed", name, exc_info=True)
            out[name] = None
    return out


# --------------------------------------------------------------- the policy
class VoiceRouter:
    """The room policy. Pure: no I/O, no thread, no bus.

    ``occupancy`` is ``{room: True | False | None}`` from the sensing side.
    ``unhealthy`` is the set of rooms whose speaker has stopped answering
    (see ``RoomSpeaker``); they are never targets, and a demoted room falls
    back rather than swallowing the line.
    """

    def __init__(self, rooms: Optional[dict] = None, here: str = DEFAULT_HERE,
                 recency_s: float = DEFAULT_RECENCY_S,
                 now: Callable[[], float] = time.monotonic):
        self.rooms: dict = dict(rooms or {})
        self._here = str(here or DEFAULT_HERE).strip() or DEFAULT_HERE
        try:
            self.recency_s = max(0.0, float(recency_s))
        except (TypeError, ValueError):
            self.recency_s = DEFAULT_RECENCY_S
        self._now = now
        self._heard: dict = {}          # room -> monotonic seconds
        self._occ: dict = {}            # the last snapshot the mesh handed us
        self.unhealthy: set = set()

    # ------------------------------------------------------------- rooms
    @property
    def here(self) -> str:
        """The fallback room. Falls through to any usable room, then to ""
        -- a router with nothing configured must not name a room that does
        not exist."""
        if self._usable(self._here):
            return self._here
        for name in self.rooms:
            if self._usable(name):
                return name
        return ""

    def _usable(self, name: str) -> bool:
        """A room may be spoken in when it exists, is enabled, is not
        demoted, and has somewhere for the voice to come out -- its own
        ``/say`` endpoint, or being ``here`` (this box's own sink)."""
        room = self.rooms.get(name)
        if not room or not room.enabled or name in self.unhealthy:
            return False
        return bool(room.url) or name == self._here

    def heard(self, room: str, at: Optional[float] = None) -> None:
        """Record that a turn happened in ``room``. This is the ONLY
        evidence the router has about which of two occupied rooms holds
        him, so every transport that delivers a turn should call it."""
        name = str(room or "").strip()
        if name:
            self._heard[name] = self._now() if at is None else float(at)

    def observe(self, occupancy: Optional[dict] = None) -> None:
        """Take the mesh's latest snapshot. Called on the MESH's thread, so
        that ``route`` -- which runs inside a spoken turn -- never waits on
        a LAN round trip. Same rule as ``soundbar.status_line``: the answer
        comes from the last tick's reading, not from a fresh probe."""
        self._occ = dict(occupancy or {})

    def demote(self, room: str) -> None:
        self.unhealthy.add(str(room or "").strip())

    def promote(self, room: str) -> None:
        self.unhealthy.discard(str(room or "").strip())

    # ------------------------------------------------------------- policy
    def route(self, kind: str = ANNOUNCE, source_room: str = "",
              occupancy: Optional[dict] = None, home: bool = True) -> Decision:
        """Where this line goes. See the module docstring for the rule."""
        if kind == ALARM:
            # Everywhere, private rooms INCLUDED: an alarm you cannot hear
            # from the bedroom is not an alarm. Private only ever excludes
            # a room from an ANNOUNCEMENT broadcast, and there is no such
            # thing -- announcements are single-room by rule.
            targets = tuple(n for n in self.rooms if self._usable(n))
            return Decision(targets or ((self.here,) if self.here else ()),
                            "alarm: every room")
        if kind == ANSWER:
            asked = str(source_room or "").strip()
            if self._usable(asked):
                # NEVER re-routed on a sensor reading. He asked two seconds
                # ago and he is still standing there; a radar that has not
                # caught up must not move the answer away from his ear.
                return Decision((asked,), "answered where it was asked")
            here = self.here
            return Decision((here,) if here else (),
                            "no source room; answered here")
        # ---- an announcement: it belongs to the person, not to the room.
        if not home:
            # The existing rule, unchanged (quiet.hold_when_away). An empty
            # house is not an invitation to shout in three rooms.
            return Decision((), "he's out", held=True)
        occ = occupancy if isinstance(occupancy, dict) else self._occ
        seen = [n for n, v in occ.items() if v is True and self._usable(n)]
        if len(seen) == 1:
            return Decision((seen[0],), "the only occupied room")
        if len(seen) > 1:
            pick = self._most_recent(seen)
            if pick:
                return Decision((pick,), "the occupied room he last spoke in")
            here = self.here
            if here in seen:
                return Decision((here,), "two rooms occupied; here by default")
            return Decision((sorted(seen)[0],), "two rooms occupied; no recent turn")
        # Nobody seen. Two very different situations, and conflating them
        # is the bug: every room saying False is EVIDENCE, every room
        # saying None is IGNORANCE. Both land on `here`, but only the first
        # earns a reason that reads like a fact.
        here = self.here
        if not here:
            return Decision((), "no room configured", held=True)
        if occ and all(v is False for v in occ.values()):
            return Decision((here,), "no room sees him; here by default")
        return Decision((here,), "no room has an opinion; here by default")

    def _most_recent(self, names: Iterable[str]) -> str:
        """The room among ``names`` with the newest turn inside the recency
        window, or "" when none of them is fresh enough to be evidence."""
        now = self._now()
        best, best_at = "", None
        for name in names:
            at = self._heard.get(name)
            if at is None or (self.recency_s and now - at > self.recency_s):
                continue
            if best_at is None or at > best_at:
                best, best_at = name, at
        return best


# ------------------------------------------------------------- the receipt
@dataclass
class Receipt:
    """What a room says it actually did with a clip. ``landed`` is the one
    question ``pactl`` could never answer."""
    ok: bool = False
    played_ms: float = 0.0
    expected_ms: float = 0.0
    peak: Optional[float] = None
    error: str = ""

    @property
    def landed(self) -> bool:
        if not self.ok:
            return False
        if self.peak is not None and self.peak <= 0.0:
            # It ran, and nothing came out: a muted or dead output. This is
            # the HDMI monitor of 2026-08-30, and it is the whole reason a
            # receipt carries a level and not just a duration.
            return False
        if self.expected_ms > 0:
            return self.played_ms >= self.expected_ms * MIN_PLAYED_RATIO
        return self.played_ms > 0


def parse_receipt(body: Any, expected_ms: float = 0.0) -> Receipt:
    """A ``/say`` response body -> a Receipt. Anything unreadable is a
    FAILED receipt, never an assumed success: an endpoint that answers with
    an HTML error page has not played anything."""
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    data: Any = body
    if isinstance(body, str):
        try:
            data = json.loads(body.strip() or "null")
        except (ValueError, TypeError):
            return Receipt(expected_ms=expected_ms, error="unreadable")
    if not isinstance(data, dict):
        return Receipt(expected_ms=expected_ms, error="unreadable")
    def _num(key: str) -> Optional[float]:
        if key not in data:
            return None
        try:
            return float(data[key])
        except (TypeError, ValueError):
            return None
    peak = _num("peak")
    return Receipt(ok=bool(data.get("ok", False)),
                   played_ms=_num("played_ms") or 0.0,
                   expected_ms=expected_ms, peak=peak,
                   error=str(data.get("error", "") or ""))


def _post_default(url: str, payload: bytes, timeout: float,
                  headers: Optional[dict] = None) -> str:
    """The one transport. ``RoomSpeaker(post=...)`` is the test seam; no
    test in this suite may touch a live LAN."""
    req = urllib.request.Request(url, data=payload, method="POST",
                                 headers={"User-Agent": USER_AGENT,
                                          "Content-Type": "audio/wav",
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - LAN http
        return resp.read(MAX_BYTES).decode("utf-8", "replace")


class RoomSpeaker:
    """One remote room's ``/say``. ``say(wav, expected_ms)`` is the whole
    interface; ``healthy`` is what the router reads.

    The breaker is ``roomsensor.RoomSensor``'s, for its reason: a room that
    has been unplugged for a week must cost the reply path nothing and must
    write one line, not one a minute.
    """

    def __init__(self, name: str, url: str, token: str = "",
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 post: Callable[..., Any] = _post_default,
                 now: Optional[Callable[[], float]] = None,
                 fail_after: int = DEFAULT_FAIL_AFTER,
                 cooldown_s: float = DEFAULT_COOLDOWN_S):
        self.name = str(name or "").strip()
        self.url = say_url(url)
        self.token = str(token or "").strip()
        try:
            self.timeout_s = max(0.1, float(timeout_s))
        except (TypeError, ValueError):
            self.timeout_s = DEFAULT_TIMEOUT_S
        self._post = post
        self._now = now or time.monotonic       # immune to clock jumps
        self.fail_after = max(1, int(fail_after or DEFAULT_FAIL_AFTER))
        self._base_cooldown = max(1.0, float(cooldown_s or DEFAULT_COOLDOWN_S))
        self._cooldown = self._base_cooldown
        self._fails = 0
        self._skip_until = 0.0
        self._down = False
        self.posts = 0                          # the breaker's proof
        self.last: Optional[Receipt] = None

    @property
    def configured(self) -> bool:
        return bool(self.url)

    @property
    def paused(self) -> bool:
        return self._skip_until > self._now()

    @property
    def healthy(self) -> bool:
        return self.configured and not self.paused

    def say(self, wav: bytes, expected_ms: float = 0.0) -> Receipt:
        """POST one clip and read the receipt back.

        A room that is not configured, or whose breaker is open, returns a
        failed receipt WITHOUT touching the network -- the caller falls
        back to ``here`` exactly as if the room were not there.
        """
        if not self.configured:
            return Receipt(expected_ms=expected_ms, error="unconfigured")
        if self.paused:
            return Receipt(expected_ms=expected_ms, error="breaker")
        headers = {"Authorization": "Bearer %s" % self.token} if self.token else {}
        try:
            body = self._post(self.url, bytes(wav or b""), self.timeout_s,
                              headers)
        except Exception as exc:  # noqa: BLE001 - every transport failure is a NO
            self.posts += 1
            self._failed("unreachable", exc, expected_ms)
            return Receipt(expected_ms=expected_ms, error="unreachable")
        self.posts += 1
        receipt = parse_receipt(body, expected_ms)
        self.last = receipt
        if not receipt.landed:
            # A room that ANSWERS but did not play counts against the
            # breaker too. "The endpoint is up" is not the promise; "the
            # sound came out" is, and that is the promise 2026-08-30 broke.
            self._failed("silent", receipt.error or "no audio", expected_ms)
            return receipt
        self._ok()
        return receipt

    # ------------------------------------------------------------ breaker
    def _failed(self, kind: str, detail: Any, expected_ms: float) -> None:
        self._fails += 1
        if self._fails < self.fail_after:
            log.debug("room %s %s: %s (%s)", self.name, kind, self.url, detail)
            return
        self._skip_until = self._now() + self._cooldown
        if not self._down:
            log.warning("room %s %s: %s (%s); speaking here instead, "
                        "retrying in %.0fs", self.name, kind, self.url,
                        detail, self._cooldown)
            self._down = True
        else:
            log.debug("room %s %s; retrying in %.0fs", self.name, kind,
                      self._cooldown)
        self._cooldown = min(self._cooldown * 2.0, MAX_COOLDOWN_S)

    def _ok(self) -> None:
        if self._down:
            log.info("room %s: back", self.name)
        self._fails, self._skip_until, self._down = 0, 0.0, False
        self._cooldown = self._base_cooldown

    def status(self) -> dict:
        """For the console and a diagnostic script; never parsed by the app."""
        last = self.last
        return {"room": self.name, "url": self.url, "healthy": self.healthy,
                "paused": self.paused, "fails": self._fails,
                "cooldown_s": self._cooldown, "posts": self.posts,
                "played_ms": last.played_ms if last else None,
                "peak": last.peak if last else None,
                "landed": last.landed if last else None}


__all__ = ["ANSWER", "ANNOUNCE", "ALARM", "KINDS", "Room", "Decision",
           "Receipt", "RoomSpeaker", "VoiceRouter", "occupancy_from",
           "parse_receipt", "rooms_from_config", "say_url"]
