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
PING_TIMEOUT_S = 3.0
PRESENT_STATES = ("REACHABLE", "DELAY", "PERMANENT")
WELCOME_LINE = "Welcome back, sir."

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


def probe(ip: str = "", mac: str = "", run: Callable = _run_default) -> bool:
    """True when the phone answers. ``run(argv, timeout=)`` is the seam
    (tests pass a fake that returns an object with returncode/stdout)."""
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


class PresenceSentinel:
    """See the module docstring. ``state`` is 'home' | 'away' | 'unknown'."""

    def __init__(self, cfg, publish: Callable = bus.publish, probe_fn: Callable = probe,
                 now: Callable[[], float] = time.time, poll_s: Optional[float] = None):
        self._cfg = cfg
        self._publish = publish
        self._probe = probe_fn
        self._now = now
        self._poll_s = poll_s
        self.home: Optional[bool] = None
        self.since: float = 0.0
        self.last_seen: Optional[float] = None
        self._started_at: Optional[float] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ config
    @property
    def phone_ip(self) -> str:
        return str(_cfg_get(self._cfg, "presence.phone_ip", "") or "").strip()

    @property
    def phone_mac(self) -> str:
        return str(_cfg_get(self._cfg, "presence.phone_mac", "") or "").strip().lower()

    @property
    def configured(self) -> bool:
        return bool(_cfg_get(self._cfg, "presence.enabled", True)) and \
            bool(self.phone_ip or self.phone_mac)

    @property
    def away_after_s(self) -> float:
        try:
            mins = float(_cfg_get(self._cfg, "presence.away_after_min", DEFAULT_AWAY_MIN))
        except (TypeError, ValueError):
            mins = DEFAULT_AWAY_MIN
        return max(60.0, mins * 60.0)

    @property
    def poll_s(self) -> float:
        if self._poll_s is not None:
            return float(self._poll_s)
        try:
            return max(5.0, float(_cfg_get(self._cfg, "presence.poll_s", DEFAULT_POLL_S)))
        except (TypeError, ValueError):
            return DEFAULT_POLL_S

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
            present = bool(self._probe(self.phone_ip, self.phone_mac))
        except Exception:  # noqa: BLE001 - the loop must survive anything
            log.exception("presence: probe failed")
            return None
        event = None
        with self._lock:
            if self._started_at is None:
                # The grace clock starts at boot, not at the first miss: a
                # phone asleep at startup would otherwise be "away" on tick
                # one and earn a spurious "Welcome back" on tick two.
                self._started_at = now
                self.last_seen = now if present else None
            if present:
                self.last_seen = now
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

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.configured:
            log.info("presence: no phone_ip / phone_mac configured; sentinel idle")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="presence",
                                        daemon=True)
        self._thread.start()

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
                log.exception("presence: tick failed")
            self._stop.wait(max(0.01, self.poll_s))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
