"""Offline mode: the ONE answer to "am I allowed to sense right now".

Hunter, 2026-09-02: *"It also needs to have a Jarvis offline mode and it
will shutdown/disable sensors and cameras"*, controlled *"by a voice
command that says offline mode or deactivate presence or something of that
nature"*, with the curfew window changeable *"with voice command or UI
settings buttons"*.

WHAT IS GOVERNED, AND WHAT DELIBERATELY IS NOT.

* ``camera`` -- the lens. Off while offline, and off every night for the
  curfew (21:00-07:00 by default, ``sensing.curfew.*``).
* ``radar``  -- the ESP32 + LD2410 mmWave module (jarvis/roomsensor.py).
  Off while offline. NOT off for the curfew: the curfew is about a lens in
  the room, and the radar produces no image, so switching it off at night
  would cost presence for nothing.
* **The microphone is not here at all.** It is his explicit choice --
  offline mode is spoken off again, so gating the mic would make the
  switch one-way. ``SENSORS`` names only the two sensors above, an unknown
  kind is DENIED rather than allowed, and ``tests/test_sensing.py`` holds
  the recorder / hotword / transcriber to importing nothing from here.
* The phone's Wi-Fi probe (jarvis/presence.py ``probe``) is not governed
  either: it reads the kernel's ARP table for an address he configured. It
  is not a sensor pointed at the room, and treating it as one would take
  away the leg that still works while the radar is down.

THE FAIL-SAFE, WHICH IS THE WHOLE POINT. He chose FAIL TO OFFLINE over
both alternatives (persist-and-trust, fail-online), in his words because
privacy beats convenience. So:

* the state file is read at construction, and **anything that is not a
  readable, parseable, unambiguous "online" starts OFFLINE** -- missing,
  empty, truncated, wrong shape, unreadable, ``{"offline": "maybe"}``;
* a first run therefore starts offline. That is not a bug: on a box where
  nothing is wired yet the honest state IS "not sensing", and one spoken
  sentence ("come back online") clears it for good;
* a ``disable()`` whose write failed reports ``persisted=False`` so the
  spoken line can say the switch will not survive a restart, rather than
  promising something the disk refused.

AN OPEN-ENDED OFFLINE DOES NOT EXPIRE; A TIMED ONE DOES. He did not say.
The safer reading is that "go offline" means until he says otherwise -- a
switch that healed itself overnight would reopen a lens he never asked to
reopen -- while "no cameras for the next two hours" carries its own end
and is honoured even across a restart, because the end time is his
instruction and not a guess.

Full walkthrough, including the wiring for a real radar power cut:
``docs/offline-mode.md``.

ENFORCEMENT IS AT THE DEVICE. ``CameraGate`` never calls the opener while
sensing is denied (a consumer that merely dropped frames would still have
a lit camera light in the room), and ``RoomSensor.read`` issues no HTTP
request at all -- its ``reads`` counter is the proof. ``disable()`` also
STOPS what is already running, through the stoppers each device attached,
and reports which ones actually stopped, which failed, and which were
never there, so the spoken confirmation states what happened instead of
what was intended.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis.logs import get_logger

log = get_logger("sensing")

CAMERA = "camera"
RADAR = "radar"
SENSORS = (CAMERA, RADAR)

# Why sensing is off, most authoritative first. The order is the order
# state() resolves them in, and the spoken lines key off it.
REASON_FAILSAFE = "failsafe"   # the state could not be read; assume the worst
REASON_OFFLINE = "offline"     # he said so, open-endedly
REASON_TIMED = "timed"         # he said so, with an end time
REASON_CURFEW = "curfew"       # the nightly camera window

DEFAULT_CURFEW_START = (21, 0)
DEFAULT_CURFEW_END = (7, 0)
CURFEW_ENABLED_KEY = "sensing.curfew.enabled"
CURFEW_START_KEY = "sensing.curfew.start"
CURFEW_END_KEY = "sensing.curfew.end"

# The settings-drawer pickers (jarvis/ui/views.py). Half-hours only, and
# only the hours a night curfew is worth having -- a free-text field in a
# slide-over is how you end up with "9pm" in a value that must parse as
# HH:MM. The two lists are disjoint on purpose: equal ends read as "no
# curfew" in curfew(), which a picker must not be able to produce.
CURFEW_START_CHOICES = ("19:00", "20:00", "20:30", "21:00", "21:30",
                        "22:00", "22:30", "23:00")
CURFEW_END_CHOICES = ("05:00", "05:30", "06:00", "06:30", "07:00",
                      "07:30", "08:00", "09:00")

STATE_VERSION = 1
MAX_STATE_BYTES = 4096         # the record is ~120 bytes; anything else is wrong


# ------------------------------------------------------------ clock helpers
def parse_hhmm(value: Any) -> Optional[tuple]:
    """``"21:00"`` -> ``(21, 0)``; anything else -> None.

    Deliberately strict: this is the value a settings button and a voice
    command both write, and a lenient parser that read "21" as nine in the
    evening would be inventing a privacy window.
    """
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return (hour, minute)
    return None


def fmt_hhmm(hm: tuple) -> str:
    return "%02d:%02d" % (int(hm[0]), int(hm[1]))


def in_window(start: tuple, end: tuple, now: tuple) -> bool:
    """Is ``now`` inside [start, end)? Wrapping windows are the normal case
    here -- his curfew is 21:00 -> 07:00 -- so the wrap is not a corner."""
    s = start[0] * 60 + start[1]
    e = end[0] * 60 + end[1]
    t = now[0] * 60 + now[1]
    if s == e:
        return False
    return s <= t < e if s < e else (t >= s or t < e)


# ------------------------------------------------------------------ records
@dataclass(frozen=True)
class SensingState:
    """What every sensor and every display reads. Never constructed by
    hand outside this module: ``SensingPolicy.state()`` is the only source."""
    camera: bool
    radar: bool
    offline: bool                     # the manual switch (or the fail-safe)
    reason: str                       # "" when everything may sense
    until: Optional[float] = None     # a timed offline's end, epoch seconds
    curfew: Optional[tuple] = None    # ((h, m), (h, m)) or None when off
    failsafe: bool = False
    persisted: bool = True            # False once a save has failed

    @property
    def sensing(self) -> bool:
        return bool(self.camera or self.radar)


@dataclass(frozen=True)
class Outcome:
    """What a switch actually DID -- the material for a spoken line that
    states what happened rather than what was asked for."""
    state: SensingState
    stopped: tuple = ()               # devices that were present and stopped
    resumed: tuple = ()               # devices brought back by enable()
    failed: tuple = ()                # present, and the switch did not work
    absent: tuple = ()                # attached but not physically there
    persisted: bool = True


@dataclass
class _Device:
    name: str
    stop: Callable[[], Any]
    present: Callable[[], bool] = field(default=lambda: True)
    resume: Optional[Callable[[], Any]] = None


# ------------------------------------------------------------------- policy
class SensingPolicy:
    """The single sensing-state owner. One per process; see the module
    docstring for the rules it enforces.

    ``cfg`` is the AssistantConfig (only ``get`` / ``set`` are used, and a
    missing config simply means the shipped curfew). ``path`` is the state
    file; ``now`` is the clock seam every test drives.
    """

    def __init__(self, cfg=None, path: Optional[os.PathLike | str] = None,
                 now: Callable[[], float] = time.time):
        self._cfg = cfg
        self._now = now
        self._lock = threading.RLock()
        self._devices: list[_Device] = []
        self._persisted = True
        if path is None:
            from jarvis.config import PATHS
            path = PATHS.MEMORY_DIR / "sensing.json"
        self.path = Path(path)
        self._offline, self._until, self._failsafe = self._load()

    # ------------------------------------------------------------ loading
    def _load(self) -> tuple:
        """(offline, until, failsafe). EVERY failure path lands offline."""
        try:
            raw = self.path.read_text(encoding="utf-8")[:MAX_STATE_BYTES]
        except FileNotFoundError:
            log.info("sensing: no state at %s; starting OFFLINE until told "
                     "otherwise", self.path)
            return True, None, True
        except OSError as exc:
            log.warning("sensing: state %s unreadable (%s); starting OFFLINE",
                        self.path, exc)
            return True, None, True
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as exc:
            log.warning("sensing: state %s is corrupt (%s); starting OFFLINE",
                        self.path, exc)
            return True, None, True
        if not isinstance(data, dict) or not isinstance(data.get("offline"), bool):
            log.warning("sensing: state %s has no usable 'offline' flag "
                        "(%.60r); starting OFFLINE", self.path, data)
            return True, None, True
        until = data.get("until")
        if until is not None:
            try:
                until = float(until)
            except (TypeError, ValueError):
                log.warning("sensing: state %s has an unusable 'until' (%r); "
                            "starting OFFLINE open-endedly", self.path, until)
                return True, None, True
        offline = bool(data["offline"])
        if offline and until is not None and until <= self._now():
            # His instruction carried its own end and the end has passed.
            # Honouring it is not a silent re-enable: it is what "for the
            # next two hours" meant, and the running process would have
            # done exactly this had it stayed up.
            log.info("sensing: the timed offline ended at %s; online", until)
            return False, None, False
        return offline, (until if offline else None), False

    # -------------------------------------------------------------- saving
    def _save(self) -> bool:
        record = {"version": STATE_VERSION, "offline": bool(self._offline),
                  "until": self._until, "saved": self._now()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            # NOT fatal in-process: the running Jarvis still obeys the
            # switch. But the next start would read the old file, so the
            # caller has to be able to say so out loud.
            log.exception("sensing: state could not be saved to %s", self.path)
            self._persisted = False
            return False
        self._persisted = True
        return True

    # -------------------------------------------------------------- curfew
    def _cfg_get(self, key: str, default=None):
        get = getattr(self._cfg, "get", None)
        if not callable(get):
            return default
        try:
            value = get(key, default)
        except Exception:  # noqa: BLE001 - a broken config must not open a lens
            log.debug("sensing: cfg.get(%s) failed", key, exc_info=True)
            return default
        return default if value is None else value

    def curfew(self) -> Optional[tuple]:
        """``((h, m), (h, m))`` or None when the curfew is switched off.

        A malformed value falls back to the SHIPPED window rather than to
        "no curfew": a typo in assistant.json must not quietly delete a
        privacy control.
        """
        if not bool(self._cfg_get(CURFEW_ENABLED_KEY, True)):
            return None
        start = parse_hhmm(self._cfg_get(CURFEW_START_KEY, ""))
        end = parse_hhmm(self._cfg_get(CURFEW_END_KEY, ""))
        if start is None or end is None or start == end:
            if self._cfg is not None:
                log.debug("sensing: curfew %r-%r unusable; using the default",
                          self._cfg_get(CURFEW_START_KEY, ""),
                          self._cfg_get(CURFEW_END_KEY, ""))
            return DEFAULT_CURFEW_START, DEFAULT_CURFEW_END
        return start, end

    def curfew_active(self, now: Optional[float] = None) -> bool:
        win = self.curfew()
        if win is None:
            return False
        dt = datetime.fromtimestamp(self._now() if now is None else now)
        return in_window(win[0], win[1], (dt.hour, dt.minute))

    def set_curfew(self, start: Optional[tuple], end: Optional[tuple]) -> bool:
        """Both ends at once, or ``(None, None)`` to switch the curfew off.
        Returns True when the config took the write."""
        setter = getattr(self._cfg, "set", None)
        if not callable(setter):
            return False
        try:
            if start is None or end is None:
                setter(CURFEW_ENABLED_KEY, False)
                return True
            setter(CURFEW_START_KEY, fmt_hhmm(start))
            setter(CURFEW_END_KEY, fmt_hhmm(end))
            setter(CURFEW_ENABLED_KEY, True)
        except Exception:  # noqa: BLE001
            log.exception("sensing: the curfew window could not be saved")
            return False
        return True

    # --------------------------------------------------------------- state
    def _expire(self) -> None:
        """A timed offline whose end has passed comes back, in place."""
        if self._offline and self._until is not None and \
                self._until <= self._now():
            log.info("sensing: the timed offline has ended; online")
            self._offline, self._until, self._failsafe = False, None, False
            self._save()

    def state(self) -> SensingState:
        with self._lock:
            self._expire()
            curfew = self.curfew()
            if self._offline:
                reason = (REASON_FAILSAFE if self._failsafe else
                          REASON_TIMED if self._until is not None else
                          REASON_OFFLINE)
                return SensingState(camera=False, radar=False, offline=True,
                                    reason=reason, until=self._until,
                                    curfew=curfew, failsafe=self._failsafe,
                                    persisted=self._persisted)
            if self.curfew_active():
                return SensingState(camera=False, radar=True, offline=False,
                                    reason=REASON_CURFEW, curfew=curfew,
                                    persisted=self._persisted)
            return SensingState(camera=True, radar=True, offline=False,
                                reason="", curfew=curfew,
                                persisted=self._persisted)

    def allowed(self, kind: Optional[str]) -> bool:
        """May ``kind`` sense right now? An unknown kind is DENIED -- a
        sensor this object has never heard of is exactly the one nobody
        thought about, and guessing "yes" is how a lens stays live."""
        st = self.state()
        if kind == CAMERA:
            return st.camera
        if kind == RADAR:
            return st.radar
        return False

    def status(self) -> dict:
        """For the console, the badge and a diagnostic script."""
        st = self.state()
        return {"camera": st.camera, "radar": st.radar, "offline": st.offline,
                "reason": st.reason, "until": st.until,
                "failsafe": st.failsafe, "persisted": st.persisted,
                "curfew": ("%s-%s" % (fmt_hhmm(st.curfew[0]),
                                      fmt_hhmm(st.curfew[1])))
                if st.curfew else ""}

    # ------------------------------------------------------------- devices
    def attach(self, name: str, stop: Callable[[], Any],
               present: Optional[Callable[[], bool]] = None,
               resume: Optional[Callable[[], Any]] = None) -> None:
        """Register a device's OWN stop (and, when it has one, the way
        back). ``present`` says whether the thing physically exists, so a
        box with no camera is never told one was shut down."""
        with self._lock:
            self._devices = [d for d in self._devices if d.name != name]
            self._devices.append(
                _Device(name, stop, present or (lambda: True), resume))

    def _switch_devices(self, on: bool) -> tuple:
        """Stop (or resume) every attached device. Returns
        (acted, failed, absent) -- the material for an honest spoken line."""
        acted, failed, absent = [], [], []
        for dev in list(self._devices):
            try:
                here = bool(dev.present())
            except Exception:  # noqa: BLE001
                log.debug("sensing: %s presence check failed", dev.name,
                          exc_info=True)
                here = True            # assume it is there and try anyway
            if not here:
                absent.append(dev.name)
                continue
            fn = dev.resume if on else dev.stop
            if fn is None:
                continue               # nothing to undo; it was never stopped
            try:
                ok = fn()
            except Exception:  # noqa: BLE001 - one bad device must not skip the rest
                log.exception("sensing: %s could not be %s", dev.name,
                              "resumed" if on else "stopped")
                failed.append(dev.name)
                continue
            (acted if ok is not False else failed).append(dev.name)
        if failed:
            log.warning("sensing: %s, but these did not follow: %s",
                        "online" if on else "offline", ", ".join(failed))
        return tuple(acted), tuple(failed), tuple(absent)

    # -------------------------------------------------------------- switch
    def disable(self, until: Optional[float] = None,
                source: str = "voice") -> Outcome:
        """Go offline. ``until`` is an epoch second for a bounded request."""
        with self._lock:
            self._offline, self._failsafe = True, False
            self._until = float(until) if until else None
            persisted = self._save()
            stopped, failed, absent = self._switch_devices(False)
        log.info("sensing: OFFLINE (%s%s); stopped=%s failed=%s", source,
                 "" if self._until is None else " until %.0f" % self._until,
                 stopped, failed)
        return Outcome(self.state(), stopped=stopped, failed=failed,
                       absent=absent, persisted=persisted)

    def enable(self, source: str = "voice") -> Outcome:
        """Come back online, clearing the fail-safe as well as the switch.

        The devices are told, not just the flag: a radar whose power was
        cut has to be switched back on, and a resume that FAILED has to be
        spoken -- "back online" with a dead radar is the same class of lie
        as "offline" with a live one.
        """
        with self._lock:
            self._offline, self._until, self._failsafe = False, None, False
            persisted = self._save()
            resumed, failed, absent = self._switch_devices(True)
        log.info("sensing: ONLINE (%s); resumed=%s failed=%s", source,
                 resumed, failed)
        return Outcome(self.state(), resumed=resumed, failed=failed,
                       absent=absent, persisted=persisted)


# --------------------------------------------------------------- the camera
class CameraGate:
    """The camera-side interface the vision lane will implement against.

    There is no camera on this box yet, so this is the CONTRACT and the
    enforcement point, not a driver: ``opener()`` is whatever finally opens
    the device (``cv2.VideoCapture(0)``, a v4l2 handle), ``closer(device)``
    releases it. The gate's promise is narrow and testable -- **while
    sensing is denied the opener is never called** -- which is why the test
    for it uses a fake device that records its own opens.

    A policy that RAISES is treated as offline. That is the fail-safe
    reaching all the way to the device: a broken decision must not be read
    as permission.
    """

    def __init__(self, policy, opener: Callable[[], Any],
                 closer: Optional[Callable[[Any], Any]] = None,
                 name: str = CAMERA,
                 present: Optional[Callable[[], bool]] = None):
        self.policy = policy
        self.name = name
        self._opener = opener
        self._closer = closer
        self._device = None
        self._lock = threading.RLock()
        attach = getattr(policy, "attach", None)
        if callable(attach):
            attach(name, self.release, present)

    @property
    def is_open(self) -> bool:
        return self._device is not None

    def allowed(self) -> bool:
        try:
            return bool(self.policy.allowed(self.name))
        except Exception:  # noqa: BLE001 - a broken policy is not permission
            log.exception("sensing: the policy failed; %s stays shut", self.name)
            return False

    def open(self):
        """The device, or None when sensing is denied. Never opens while
        denied, and closes an already-open device the moment it is."""
        with self._lock:
            if not self.allowed():
                if self._device is not None:
                    self.release()
                return None
            if self._device is None:
                self._device = self._opener()
            return self._device

    def release(self) -> bool:
        """Close the device. True when there is nothing left open."""
        with self._lock:
            dev, self._device = self._device, None
            if dev is None:
                return True
            if self._closer is None:
                closer = getattr(dev, "release", None) or getattr(dev, "close", None)
            else:
                closer = lambda: self._closer(dev)  # noqa: E731 - one call site
            if not callable(closer):
                return True
            try:
                closer()
            except Exception:  # noqa: BLE001
                log.exception("sensing: %s could not be released", self.name)
                self._device = dev          # still open; say so
                return False
            return True

    def status(self) -> dict:
        return {"name": self.name, "open": self.is_open,
                "allowed": self.allowed()}
