"""Rooms: what a satellite in ANOTHER room is allowed to sense, and what
it will actually confirm.

``jarvis/sensing.py`` (2026-09-02) is the one sensing owner and it enforces
at the device: ``CameraGate`` never opens the lens while sensing is denied,
``RoomSensor.read`` issues no request at all. That works because those
devices are on THIS computer. A satellite in the kitchen is a separate
computer on the LAN, and the promise does not survive the trip:

    **Jarvis cannot guarantee a remote camera or radar is off. If the
    network drops it can stop TRUSTING the data; it cannot stop the
    sensing.**

Fail-to-offline is not something a central controller can do to a machine
it cannot reach. So this module does not try. It changes who makes the
promise:

* **Jarvis is the REQUESTER.** It holds a lease open on each satellite and
  renews it while ``SensingPolicy`` says that kind of sensor may run.
* **The satellite is the ENFORCER.** Its firmware powers the sensor down
  when the lease expires (``scripts/esphome/jarvis-satellite.yaml``:
  a ``script`` in ``mode: restart`` with a delay, plus
  ``restore_mode: ALWAYS_OFF`` so a reboot comes up not sensing).

A network drop, a Jarvis crash, a ``kill -9``, a power cut on the Spark,
a satellite reboot -- every one of them stops the renewal, and every one of
them therefore ends in a powered-down sensor without Jarvis doing anything.
That is fail-to-offline made a property of the DEVICE rather than a promise
by the controller, and it is the only version of the promise that is true.

WHAT THE LEASE DOES NOT BUY, said now rather than discovered later:

* the curfew edge and "offline mode" are sent as an explicit REVOKE, and
  the lease is only the backstop. If the revoke does not land, a remote
  sensor keeps running for up to one TTL (``lease_ttl_s``, 90 s by
  default). The console says so; it does not round it down to "off";
* a satellite that has been reflashed, or whose firmware is lying, can
  report anything. No software check here detects that. The answer is an
  LED wired across the sensor's SWITCHED SUPPLY -- a light no firmware can
  turn on -- and a physical switch. See docs/multiroom-privacy.md;
* "unreachable" is not "off". It is also not "on". It is UNKNOWN, and this
  module refuses to collapse it into either, because a green tick on a
  device nobody can reach is the exact lie the feature exists to prevent.

HOW IT COMPOSES WITH WHAT IS ALREADY THERE, without touching it.

* ``jarvis/roomsensor.py`` is reused unmodified, once for
  ``/binary_sensor/presence`` and once per governed sensor for
  ``/binary_sensor/<kind>_powered``.
  Its circuit breaker, its 4 KB cap and its None-on-anything-odd rule are
  the containment for a satellite that has been compromised, and reusing it
  means there is one HTTP parser to review rather than two. The ``get``
  seam it already exposes is where the auth header and the redirect refusal
  go, so no edit is needed there either.
* ``policy`` is NOT handed to those RoomSensors. ``SensingPolicy.attach``
  replaces by NAME, so six sensors attaching as ``"radar"`` would leave one
  attached and five unstoppable. ``Satellite`` asks ``policy.allowed(RADAR)``
  itself before it reads (so an offline satellite is never polled -- the
  ``reads`` counter is the assertion, same as next door) and attaches ONCE
  under a room-qualified name, ``"kitchen radar"``. That name is what the
  spoken confirmation already says out loud: "I couldn't stop the bedroom
  radar, so it may still be running."
* the PRESENCE fusion is not here. ``jarvis/roomfabric.py`` owns it -- N
  named rooms, four timers argued from the LD2410's late OFF edge, and the
  refusal to answer "is someone ELSE here" from a sensor that reports one
  bit. A ``Satellite`` wears ``read() -> True | False | None``, which is
  the only shape that fabric wants, so it drops straight in as a room's
  reader. ``RoomMesh.read`` / ``mesh_probe`` below are the degenerate
  one-line version for a box with no fabric wired; prefer the fabric.
  ``PrivacyView`` here is deliberately NOT ``roomfabric.HouseView``: that
  one answers "who is where", this one answers "what is allowed to look".

TRUST. A satellite is untrusted input. A URL must be ``http://`` plus a
PRIVATE IP LITERAL: no hostnames (an mDNS answer is one poisoned packet
away from pointing the lease at somebody else's box) and no public
addresses (the mirror of ``jarvis/webapp.py``'s refusal to bind one).
Redirects are refused outright -- a compromised satellite answering 302 to
``http://127.0.0.1:8765`` would otherwise turn this poll loop into an SSRF
gadget against the phone client.
"""
from __future__ import annotations

import base64
import ipaddress
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from jarvis.logs import get_logger
from jarvis.sensing import CAMERA, RADAR

log = get_logger("rooms")

# The third row the console shows and the policy deliberately does NOT
# govern. jarvis/sensing.py owns camera + radar and stops at the
# microphone on purpose (offline mode is spoken off again). A satellite
# mic is the same argument one room over, so it is not gated here either
# -- but it IS displayed, because "offline mode left the kitchen
# microphone live" must be visible rather than inferred.
MIC = "mic"
KINDS = (CAMERA, RADAR, MIC)

# Per (room, sensor) readout. UNKNOWN is a first-class state and never a
# synonym for OFF.
LIVE = "live"           # the device says this sensor is powered, recently
OFF = "off"             # the device says it is not, recently
UNKNOWN = "unknown"     # no fresh answer: unreachable, stale, unparseable
DISAGREE = "disagree"   # we asked for OFF and the device says it is ON
ABSENT = "absent"       # no such sensor in that room

ALLOW = "allow"
DENY = "deny"

# 90 s is a compromise with two ends. Shorter means a satellite that loses
# contact goes dark sooner (good) and that a Jarvis restart costs presence
# in every other room for longer (bad, and he restarts it daily). The
# camera's own TTL is shorter because a lens is worth more than a radar.
DEFAULT_LEASE_TTL_S = 90.0
DEFAULT_CAMERA_TTL_S = 20.0
DEFAULT_RENEW_S = 25.0          # ~TTL/3.5: two lost packets are survivable
DEFAULT_STALE_AFTER_S = 90.0    # older than this is UNKNOWN, not the last value
DEFAULT_TIMEOUT_S = 1.5
REVOKE_TRIES = 3                # then stop asking and let the lease expire

DEFAULT_PRESENCE_PATH = "/binary_sensor/presence"
# ONE LEASE PER SENSOR KIND, not one per room. The 21:00 curfew closes the
# lens and deliberately leaves the radar up (jarvis/sensing.py: the radar
# makes no image, so taking it down at night costs presence for no privacy).
# A single room-wide lease could not express that -- stopping renewal would
# take the radar with the camera -- so ``{kind}`` is substituted into each
# path and each kind counts down on its own.
DEFAULT_POWERED_PATH = "/binary_sensor/{kind}_powered"
DEFAULT_RENEW_PATH = "/button/{kind}_lease_renew/press"
DEFAULT_REVOKE_PATH = "/button/{kind}_lease_revoke/press"

MAX_BYTES = 4096
USER_AGENT = "jarvis-rooms/1"


# ----------------------------------------------------------------- trust
def is_private_ip(host: str) -> bool:
    """True for an address a home network can own, given as an IP LITERAL.

    ``not is_global``, the same rule and for the same reason as
    ``jarvis/webapp.py:192``: ``is_private`` is False for 100.64.0.0/10
    (carrier-grade NAT, which some home routers hand out) and True for
    203.0.113.0/24, so the obvious test is wrong in both directions.
    Written out here rather than imported because this runs in a poll loop
    and webapp.py is the whole phone client; the comment is the link.

    A hostname is False, deliberately. Resolution would happen at poll
    time against whatever DNS or mDNS answers, and this URL is where
    Jarvis POSTs privacy commands.
    """
    text = (host or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    if addr.is_unspecified or addr.is_multicast:
        return False
    return not addr.is_global


def check_url(value: str) -> str:
    """The satellite base URL, or "" with the reason logged.

    Enforced here rather than at the first request so a typo is one loud
    line at start-up instead of a timeout every minute forever.
    """
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https"):
        log.warning("rooms: %r has no http(s) scheme; ignored", text)
        return ""
    host = parts.hostname or ""
    if not is_private_ip(host):
        log.warning("rooms: %r is not a private IP literal; ignored. A "
                    "satellite URL must be http://<private ip>[:port] -- "
                    "no hostnames, no public addresses", text)
        return ""
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect.

    urllib follows them by default. A satellite that has been compromised
    could answer 302 ``http://127.0.0.1:8765/api/say`` and turn this poll
    loop into a request-forgery gadget aimed at the phone client, which is
    bound to a private address precisely because it trusts private peers.
    Returning None makes urllib raise, which every caller here reads as a
    failure, which reads as UNKNOWN.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        log.warning("rooms: refused a %s redirect to %r", code, newurl)
        return None


def make_transport(username: str = "", password: str = "") -> tuple:
    """``(get, post)`` for one satellite: no redirects, capped, optionally
    authenticated.

    BASIC rather than digest, pre-sent. ESPHome's ``web_server: auth:``
    wraps every registered handler (AuthMiddlewareHandler in
    esphome/components/web_server_base), so both work -- but digest costs a
    401 round trip on EVERY request through urllib, doubling the poll, and
    the attacker it defends against (a passive sniffer inside the WPA2
    session) is not the one on this threat model. The one that matters is
    an active host on the LAN, and basic stops him exactly as well. The
    real mitigation is the VLAN; see docs/multiroom-privacy.md.
    """
    opener = urllib.request.build_opener(_NoRedirect)
    headers = {"User-Agent": USER_AGENT}
    if username or password:
        raw = ("%s:%s" % (username, password)).encode("utf-8")
        headers["Authorization"] = "Basic " + \
            base64.b64encode(raw).decode("ascii")

    def _request(url: str, timeout: float, method: str):
        req = urllib.request.Request(url, data=b"" if method == "POST" else None,
                                     method=method, headers=headers)
        with opener.open(req, timeout=timeout) as resp:  # noqa: S310 - LAN http
            return resp.read(MAX_BYTES).decode("utf-8", "replace")

    def get(url: str, timeout: float) -> str:
        return _request(url, timeout, "GET")

    def post(url: str, timeout: float) -> str:
        return _request(url, timeout, "POST")

    return get, post


# ---------------------------------------------------------------- config
@dataclass(frozen=True)
class RoomSpec:
    """One satellite, as configured. ``sensors`` is what is physically in
    that room; a kind that is not listed renders ABSENT rather than OFF,
    because "there is no camera in the kitchen" and "the kitchen camera is
    off" are different claims and only one of them is a guarantee."""
    name: str
    url: str
    sensors: tuple = (RADAR,)
    username: str = ""
    password: str = ""
    presence_path: str = DEFAULT_PRESENCE_PATH
    powered_path: str = DEFAULT_POWERED_PATH
    renew_path: str = DEFAULT_RENEW_PATH
    revoke_path: str = DEFAULT_REVOKE_PATH
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S

    @property
    def configured(self) -> bool:
        return bool(self.name and self.url)


def spec_from_dict(data: Any) -> Optional[RoomSpec]:
    """One ``rooms.satellites[]`` entry -> a RoomSpec, or None.

    Never raises: a satellite that cannot be understood is dropped with a
    line in the log, exactly as a bad room_sensor_url is next door. A
    half-parsed satellite would be worse than none -- it would appear in
    the console as a room that is being watched over.
    """
    if not isinstance(data, dict):
        return None
    name = str(data.get("name", "") or "").strip()
    url = check_url(data.get("url", ""))
    if not name or not url:
        if name or data.get("url"):
            log.warning("rooms: satellite %r has no usable url; dropped", name)
        return None
    kinds = tuple(k for k in (data.get("sensors") or (RADAR,))
                  if k in KINDS)
    if not kinds:
        log.warning("rooms: satellite %r lists no known sensor; dropped", name)
        return None
    try:
        ttl = max(5.0, float(data.get("lease_ttl_s", DEFAULT_LEASE_TTL_S)))
    except (TypeError, ValueError):
        ttl = DEFAULT_LEASE_TTL_S
    return RoomSpec(
        name=name, url=url, sensors=kinds,
        username=str(data.get("username", "") or ""),
        password=str(data.get("password", "") or ""),
        presence_path=str(data.get("presence_path") or DEFAULT_PRESENCE_PATH),
        powered_path=str(data.get("powered_path") or DEFAULT_POWERED_PATH),
        renew_path=str(data.get("renew_path") or DEFAULT_RENEW_PATH),
        revoke_path=str(data.get("revoke_path") or DEFAULT_REVOKE_PATH),
        lease_ttl_s=ttl)


def specs_from_config(cfg) -> tuple:
    """Every enabled satellite from ``rooms.satellites``. Never raises."""
    get = getattr(cfg, "get", None)
    if not callable(get):
        return ()
    try:
        if not bool(get("rooms.enabled", False)):
            return ()
        raw = get("rooms.satellites", []) or []
    except Exception:  # noqa: BLE001 - a broken config must not open a lens
        log.exception("rooms: config unreadable; no satellites")
        return ()
    out = []
    seen = set()
    for entry in raw if isinstance(raw, (list, tuple)) else ():
        spec = spec_from_dict(entry)
        if spec is None:
            continue
        if spec.name in seen:
            log.warning("rooms: two satellites are called %r; the second is "
                        "dropped", spec.name)
            continue
        seen.add(spec.name)
        out.append(spec)
    return tuple(out)


# ----------------------------------------------------------------- views
@dataclass(frozen=True)
class SensorView:
    """One (room, sensor) readout, as the console and the spoken line see
    it. Everything a UI would argue about is decided here, so the widget
    is a renderer and the test does not need a window."""
    room: str
    kind: str
    state: str
    intent: str = ALLOW
    age_s: Optional[float] = None       # since the last confirmation
    lease_left_s: Optional[float] = None
    unexpected: bool = False            # asked to sense, reports off
    detail: str = ""

    @property
    def confirmed(self) -> bool:
        """True only when the DEVICE said so, recently. Inference never
        sets this: 'its lease must have expired by now' is a caption, not
        a confirmation, and the two must not share a colour."""
        return self.state in (LIVE, OFF)


def derive_state(intent: str, powered: Optional[bool], age_s: Optional[float],
                 stale_after_s: float = DEFAULT_STALE_AFTER_S) -> str:
    """(what we asked for, what the device said, how old) -> a state.

    Pure, and the whole honesty rule lives in the first branch: no answer,
    or an answer older than the freshness window, is UNKNOWN. Holding the
    last value would mean a satellite that was unplugged while OFF renders
    as a confirmed OFF forever, which is the most comfortable lie
    available here and therefore the one worth refusing hardest.
    """
    if powered is None or age_s is None or age_s > max(1.0, stale_after_s):
        return UNKNOWN
    if powered:
        return DISAGREE if intent == DENY else LIVE
    return OFF


TONES = {LIVE: ("LIVE", "cyan", True),
         OFF: ("OFF", "warn", True),
         UNKNOWN: ("UNKNOWN", "muted", False),
         DISAGREE: ("STILL ON", "error", True),
         ABSENT: ("", "muted", False)}


def chip(view: SensorView) -> tuple:
    """``(word, tone, filled)`` for one readout.

    Three axes on purpose, the same rule jarvis/ui/sensing_badge.py sets:
    the word carries it in text, the tone in colour, and FILLED carries it
    in shape. UNKNOWN is hollow and muted -- it must not be mistakable for
    OFF at a glance across a dimmed room, which a second amber dot would
    be. STILL ON is the only red in this readout because it is the only
    state that means a privacy instruction did not take.
    """
    return TONES.get(view.state, TONES[UNKNOWN])


def _mins(seconds: Optional[float]) -> str:
    if seconds is None:
        return "never"
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return "%d s ago" % int(seconds)
    if seconds < 5400:
        return "%d min ago" % int(round(seconds / 60.0))
    return "%.1f h ago" % (seconds / 3600.0)


def caption(view: SensorView) -> str:
    """The second line: what is known, how old it is, and -- when the
    device cannot be reached -- what the LEASE implies, marked as an
    inference rather than dressed up as a reading."""
    if view.state == ABSENT:
        return "no %s in the %s" % (view.kind, view.room)
    if view.state == DISAGREE:
        return ("asked to stop %s ago and still reporting on"
                % _mins(view.age_s).replace(" ago", ""))
    if view.state == UNKNOWN:
        head = "unreachable; last heard %s" % _mins(view.age_s)
        if view.lease_left_s is not None and view.lease_left_s <= 0:
            return head + " — its lease has expired, so it should have " \
                          "powered down unless its firmware was changed"
        if view.lease_left_s is not None:
            return head + " — lease runs out in %d s" % int(view.lease_left_s)
        return head
    if view.state == OFF and view.unexpected:
        return "reports off, but nothing asked it to stop (%s)" \
            % _mins(view.age_s)
    return "%s, confirmed %s" % ("sensing" if view.state == LIVE
                                 else "off at the device", _mins(view.age_s))


@dataclass(frozen=True)
class PrivacyView:
    """Every room, every sensor, one snapshot. ``rows`` is ordered: the
    local room first, then satellites in config order, so the console does
    not reshuffle between passes."""
    rows: tuple = ()
    at: float = 0.0

    def worst(self) -> str:
        """The single state a one-chip roll-up may show.

        DISAGREE beats UNKNOWN beats LIVE beats OFF. Note that LIVE beats
        OFF and not the other way round: a roll-up that showed OFF while
        one room was still sensing would be the lie again, one level up.
        """
        for state in (DISAGREE, UNKNOWN, LIVE, OFF):
            if any(r.state == state for r in self.rows):
                return state
        return OFF

    def rooms(self) -> tuple:
        out = []
        for row in self.rows:
            if row.room not in out:
                out.append(row.room)
        return tuple(out)

    def by_state(self, state: str) -> tuple:
        return tuple(r for r in self.rows if r.state == state)

    def trustworthy(self) -> bool:
        """True when every configured sensor answered recently. The
        console's "all confirmed" tick is allowed only here."""
        return bool(self.rows) and all(
            r.state != UNKNOWN for r in self.rows if r.state != ABSENT)


def _phrase(rows: tuple) -> str:
    """('kitchen','radar'), ('kitchen','camera') -> 'the kitchen radar and
    camera'; across rooms it keeps the rooms apart."""
    order: list = []
    per: dict = {}
    for row in rows:
        if row.room not in per:
            per[row.room] = []
            order.append(row.room)
        per[row.room].append(row.kind)
    parts = []
    for room in order:
        kinds = per[room]
        if len(kinds) == 1:
            parts.append("the %s %s" % (room, kinds[0]))
        else:
            parts.append("the %s %s and %s" % (room, ", ".join(kinds[:-1]),
                                               kinds[-1]))
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def spoken_status(view: PrivacyView) -> str:
    """"Are you watching?", answered per room.

    The unreachable rooms are named FIRST and never rounded down to "off":
    the whole point of asking is the room he cannot see, and "everything's
    off, sir" with a kitchen nobody has heard from in four minutes is the
    answer this module exists to make impossible.
    """
    if not view.rows:
        return "There are no rooms configured, sir."
    parts = []
    bad = view.by_state(DISAGREE)
    unknown = view.by_state(UNKNOWN)
    live = view.by_state(LIVE)
    off = view.by_state(OFF)
    if bad:
        parts.append("%s %s still reporting on after I asked %s to stop."
                     % (_phrase(bad)[0].upper() + _phrase(bad)[1:],
                        "are" if len(bad) > 1 else "is",
                        "them" if len(bad) > 1 else "it"))
    if unknown:
        expired = all(r.lease_left_s is not None and r.lease_left_s <= 0
                      for r in unknown)
        parts.append("I can't reach %s. %s" % (
            _phrase(unknown),
            "The lease has run out, so it should be off — but I can't "
            "confirm that." if expired else
            "I don't know what it's doing."))
    if live:
        parts.append("%s %s sensing."
                     % (_phrase(live)[0].upper() + _phrase(live)[1:],
                        "are" if len(live) > 1 else "is"))
    if off and not (bad or unknown):
        parts.append("%s %s off."
                     % (_phrase(off)[0].upper() + _phrase(off)[1:],
                        "are" if len(off) > 1 else "is"))
    elif off:
        parts.append("%s %s off." % (_phrase(off)[0].upper() + _phrase(off)[1:],
                                     "are" if len(off) > 1 else "is"))
    return " ".join(parts) if parts else "Nothing is configured to sense, sir."


# ------------------------------------------------------------- satellite
@dataclass
class _Lease:
    """One sensor kind's lease on one satellite. Per KIND rather than per
    room because the 21:00 curfew closes the lens and leaves the radar up;
    a room-wide lease could not say that."""
    kind: str
    intent: str = DENY               # nothing is leased until it is asked for
    until: Optional[float] = None    # our copy of the device's countdown
    powered: Optional[bool] = None   # what the device last said
    confirmed_at: Optional[float] = None
    revoke_tries: int = 0


class Satellite:
    """One room's device: a lease per sensor, and the two things it is asked.

    ``read()`` is the presence answer and it is deliberately conservative:
    it returns None unless the device has JUST confirmed the radar is
    powered. A presence reading from a sensor we cannot confirm is running
    is a reading from an unknown moment, and None ("no opinion") is the
    path jarvis/presence.py already handles safely.
    """

    def __init__(self, spec: RoomSpec, policy: Any = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 get: Optional[Callable] = None,
                 post: Optional[Callable] = None,
                 now: Callable[[], float] = time.time):
        from jarvis import roomsensor
        self.spec = spec
        self.policy = policy
        self.timeout_s = max(0.1, float(timeout_s))
        self._now = now
        default_get, default_post = make_transport(spec.username, spec.password)
        self._get = get or default_get
        self._post = post or default_post
        self._lock = threading.RLock()
        # NOT given the policy: SensingPolicy.attach replaces by NAME, so
        # every satellite attaching as "radar" would leave one stoppable and
        # the rest silently un-stoppable. The gate is applied in read()
        # instead, and the attach below is room-qualified -- which is also
        # exactly what the spoken confirmation needs to name.
        self.presence = roomsensor.RoomSensor(
            spec.url + spec.presence_path, timeout_s=self.timeout_s,
            get=self._get, now=now)
        self.leases: dict = {}
        self.powered: dict = {}
        for kind in spec.sensors:
            if kind == MIC:
                continue                 # not governed; see the module head
            self.leases[kind] = _Lease(kind)
            self.powered[kind] = roomsensor.RoomSensor(
                spec.url + spec.powered_path.format(kind=kind),
                timeout_s=self.timeout_s, get=self._get, now=now)
        attach = getattr(policy, "attach", None)
        if callable(attach):
            for kind in self.leases:
                attach("%s %s" % (spec.name, kind),
                       (lambda k=kind: self.revoke(k)),
                       present=(lambda: self.spec.configured),
                       resume=(lambda k=kind: self.renew(k)))

    # ------------------------------------------------------------ gates
    def allowed(self, kind: str = RADAR) -> bool:
        """May this kind sense, per the house policy? A policy that RAISES
        is a NO -- the fail-safe reaching one room further out."""
        policy = self.policy
        if policy is None:
            return True
        try:
            return bool(policy.allowed(kind))
        except Exception:  # noqa: BLE001 - a broken policy is not permission
            log.exception("rooms: the sensing policy failed; %s %s treated as "
                          "offline", self.spec.name, kind)
            return False

    def has(self, kind: str) -> bool:
        return kind in self.spec.sensors

    @property
    def governed(self) -> tuple:
        """The kinds this room leases -- camera and radar, never the mic."""
        return tuple(self.leases)

    # ------------------------------------------------------------ lease
    def _press(self, path: str) -> bool:
        try:
            self._post(self.spec.url + path, self.timeout_s)
        except Exception:  # noqa: BLE001 - every transport failure is a NO
            log.debug("rooms: %s%s failed", self.spec.name, path, exc_info=True)
            return False
        return True

    def renew(self, kind: str = RADAR) -> bool:
        """Push one sensor's lease out by a TTL. False when it did not land
        -- and a lease that did not land is NOT extended here, so our copy
        of the countdown keeps running toward the device powering itself
        down, which is what the device is really doing."""
        lease = self.leases.get(kind)
        if lease is None:
            return True                  # nothing here to lease
        ok = self._press(self.spec.renew_path.format(kind=kind))
        with self._lock:
            lease.intent = ALLOW
            lease.revoke_tries = 0
            if ok:
                lease.until = self._now() + self.spec.lease_ttl_s
        if not ok:
            log.warning("rooms: %s %s did not take the lease renewal",
                        self.spec.name, kind)
        return ok

    def revoke(self, kind: str = RADAR) -> bool:
        """Stop one sensor now, rather than waiting out its lease.

        This is what the curfew edge and "offline mode" send. The lease is
        the BACKSTOP for when this cannot be delivered; it is not the
        mechanism, because up to 90 s of extra camera is not what he means
        by 21:00.
        """
        lease = self.leases.get(kind)
        if lease is None:
            return True
        ok = self._press(self.spec.revoke_path.format(kind=kind))
        with self._lock:
            lease.intent = DENY
            lease.revoke_tries = 0 if ok else lease.revoke_tries + 1
            if ok:
                lease.until = self._now()
        if not ok:
            log.warning("rooms: %s %s refused the revoke (try %d)",
                        self.spec.name, kind, lease.revoke_tries)
        return ok

    # SensingPolicy.attach hooks are bound per kind in __init__, so
    # Outcome.stopped / .failed name "kitchen camera", not just the room.
    def stop(self) -> bool:
        """Every governed sensor in this room off. True only if all of them
        took it -- a partial stop is a failure, and must be said as one."""
        return all([self.revoke(k) for k in self.governed])

    def resume(self) -> bool:
        return all([self.renew(k) for k in self.governed])

    # ------------------------------------------------------- confirming
    def confirm(self, kind: str = RADAR) -> Optional[bool]:
        """Ask the device what it is actually doing. None = it did not say.

        Nothing here trusts the ANSWER for privacy purposes -- a satellite
        that has been reflashed will say whatever its author wants -- but a
        satellite that has merely crashed, hung or dropped off the Wi-Fi
        tells the truth by not answering, and that is the case this catches.
        """
        sensor = self.powered.get(kind)
        if sensor is None:
            return None
        value = sensor.read()
        with self._lock:
            lease = self.leases[kind]
            if value is not None:
                lease.powered = bool(value)
                lease.confirmed_at = self._now()
            return value

    def confirm_all(self) -> None:
        for kind in self.governed:
            self.confirm(kind)

    def view(self, kind: str = RADAR,
             stale_after_s: float = DEFAULT_STALE_AFTER_S) -> SensorView:
        with self._lock:
            if not self.has(kind):
                return SensorView(self.spec.name, kind, ABSENT)
            if kind == MIC:
                # Not leased, not governed, and honest about it: offline
                # mode leaves microphones alone by design, and a row that
                # quietly showed OFF would be the lie one room over.
                return SensorView(self.spec.name, kind, UNKNOWN,
                                  detail="the microphone is not governed by "
                                         "offline mode")
            lease = self.leases[kind]
            now = self._now()
            age = None if lease.confirmed_at is None \
                else now - lease.confirmed_at
            left = None if lease.until is None else lease.until - now
            state = derive_state(lease.intent, lease.powered, age,
                                 stale_after_s)
            return SensorView(self.spec.name, kind, state,
                              intent=lease.intent, age_s=age, lease_left_s=left,
                              unexpected=(state == OFF
                                          and lease.intent == ALLOW))

    # ------------------------------------------------------------- read
    def read(self) -> Optional[bool]:
        """Is someone in THIS room? True / False / None (no opinion).

        Two gates before the socket, and the ``reads`` counters on both
        RoomSensors are the assertion that they held: offline mode means
        the satellite is not polled at all, not that its answer is
        discarded. And a presence reading is only returned when the device
        has confirmed the radar is powered in the same breath -- otherwise
        the number describes a moment nobody can date.
        """
        if not self.has(RADAR) or not self.allowed(RADAR):
            return None
        if self.confirm(RADAR) is not True:
            return None
        return self.presence.read()

    def status(self) -> dict:
        with self._lock:
            return {"room": self.spec.name, "url": self.spec.url,
                    "sensors": list(self.spec.sensors),
                    "presence_reads": self.presence.reads,
                    "leases": {k: {"intent": v.intent, "until": v.until,
                                   "powered": v.powered,
                                   "confirmed_at": v.confirmed_at,
                                   "reads": self.powered[k].reads}
                               for k, v in self.leases.items()}}


# ------------------------------------------------------------------ mesh
class RoomMesh:
    """Every satellite, one renewal thread, one view.

    The loop is the honest half of the design: on each pass it asks the
    house policy what each kind may do and either renews or revokes, then
    reads back what the device says. It does NOT stop confirming a room it
    has revoked -- a device that keeps reporting ON after being told to
    stop is the one thing that must reach his eyes, and a loop that stopped
    looking would never see it.
    """

    def __init__(self, specs=(), policy: Any = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 renew_s: float = DEFAULT_RENEW_S,
                 stale_after_s: float = DEFAULT_STALE_AFTER_S,
                 now: Callable[[], float] = time.time,
                 satellites=None):
        self.policy = policy
        self.renew_s = max(2.0, float(renew_s))
        self.stale_after_s = max(2.0, float(stale_after_s))
        self._now = now
        self.satellites: list = list(satellites) if satellites is not None \
            else [Satellite(s, policy=policy, timeout_s=timeout_s, now=now)
                  for s in specs]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_config(cls, cfg, policy: Any = None, **kw) -> "RoomMesh":
        get = getattr(cfg, "get", None)
        def _num(key, default):
            try:
                return float(get(key, default)) if callable(get) else default
            except (TypeError, ValueError):
                return default
        return cls(specs_from_config(cfg), policy=policy,
                   renew_s=_num("rooms.renew_s", DEFAULT_RENEW_S),
                   stale_after_s=_num("rooms.stale_after_s",
                                      DEFAULT_STALE_AFTER_S),
                   timeout_s=_num("rooms.timeout_s", DEFAULT_TIMEOUT_S), **kw)

    @property
    def configured(self) -> bool:
        return bool(self.satellites)

    # -------------------------------------------------------------- tick
    def tick(self) -> None:
        """One renewal pass. Never raises: a satellite that throws must not
        cost the others their lease, which would take out presence in a
        room that was working fine."""
        for sat in list(self.satellites):
            try:
                self._tick_one(sat)
            except Exception:  # noqa: BLE001 - one room must not sink the rest
                log.exception("rooms: %s tick failed", sat.spec.name)

    def _tick_one(self, sat: "Satellite") -> None:
        """One satellite, one kind at a time.

        Per KIND is the whole reason the leases are separate: at 21:00 the
        camera's lease stops being renewed and the radar's does not, which
        is the ruling jarvis/sensing.py already made for the local box,
        carried one room out.
        """
        for kind in sat.governed:
            lease = sat.leases[kind]
            if sat.allowed(kind):
                sat.renew(kind)
            elif lease.intent != DENY or lease.revoke_tries < REVOKE_TRIES:
                # Retry a revoke that did not land, then stop asking: past
                # three tries the network is the problem, the lease is
                # already counting down at the device, and hammering it
                # only fills the log. The room goes UNKNOWN on the console,
                # which is the true statement.
                sat.revoke(kind)
            # Confirm even a kind we have given up revoking: a device still
            # reporting ON after being told to stop is the one thing that
            # has to reach his eyes, and a loop that stopped looking would
            # never see it.
            sat.confirm(kind)

    def view(self) -> PrivacyView:
        rows = []
        for sat in self.satellites:
            for kind in KINDS:
                if sat.has(kind):
                    rows.append(sat.view(kind, self.stale_after_s))
        return PrivacyView(rows=tuple(rows), at=self._now())

    # ------------------------------------------------------------- read
    def read(self) -> Optional[bool]:
        """Is anyone in ANY room? The N-room form of RoomOrPhone's rule.

        Someone seen anywhere is True and ends it. Nobody-anywhere is False
        ONLY when every room answered; one silent room makes the house
        None, because a satellite that dropped off the Wi-Fi is not
        evidence that its room is empty -- and a false "away" is what makes
        Jarvis go quiet on him for the evening.
        """
        answers = []
        for sat in list(self.satellites):
            try:
                answers.append(sat.read())
            except Exception:  # noqa: BLE001
                log.debug("rooms: %s read failed", sat.spec.name, exc_info=True)
                answers.append(None)
        if any(a is True for a in answers):
            return True
        if answers and all(a is False for a in answers):
            return False
        return None

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.configured:
            log.info("rooms: no satellites configured; mesh idle")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="rooms",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Quit. The lease is deliberately NOT renewed on the way out and
        NOT revoked either: a clean shutdown and a crash must end in the
        same place, or the crash path is the one nobody tests."""
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("rooms: tick failed")
            self._stop.wait(self.renew_s)


def mesh_probe(mesh: "RoomMesh", phone: Optional[Callable] = None) -> Callable:
    """The probe ``PresenceSentinel`` polls, extended to N rooms.

    THE FALLBACK, not the recommendation. ``jarvis/roomfabric.py`` does
    this properly -- enter/leave holds, a doorway anti-flap, a stuck-room
    guard -- and a ``Satellite`` plugs into it directly because it already
    wears ``read() -> True | False | None``. Use this only on a box where
    the fabric is not wired.

    Same asymmetry as ``jarvis.presence.RoomOrPhone`` and for the same
    reason: a room seeing someone is positive evidence and beats a sleeping
    phone; a room seeing nobody is not an empty flat.
    """
    from jarvis import presence as presence_mod
    phone_fn = phone if phone is not None else presence_mod.probe

    def probe(ip: str = "", mac: str = "") -> Optional[bool]:
        seen = mesh.read()
        if seen:
            return True
        if ip or mac:
            return bool(phone_fn(ip, mac))
        return False if seen is False else None

    return probe
