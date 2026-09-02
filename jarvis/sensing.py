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

ENFORCEMENT IS AT THE DEVICE, AND IT RUNS ON THE CLOCK. ``CameraGate``
never calls the opener while sensing is denied (a consumer that merely
dropped frames would still have a lit camera light in the room), and
``RoomSensor.read`` issues no HTTP request at all -- its ``reads`` counter
is the proof. ``disable()`` also STOPS what is already running, through
the stoppers each device attached.

But the 21:00 curfew edge arrives with NOBODY having said anything, and a
lens opened at 20:59 is still a lit camera at 21:01 if the only guard is
``open()``. So ``enforce()`` walks the attached devices and stops the ones
whose permission has just gone away (and resumes the ones it stopped once
it comes back), and ``start()`` runs it on a daemon thread owned by this
object -- deliberately NOT from the console's 5 s pass, because a lens
must not stay open because the UI thread died or the window was never
built.

WHAT THE SPOKEN LINE MAY CLAIM. A switch reports which devices actually
stopped, which failed, which were never there -- and which could only be
stopped AS FAR AS THIS PROCESS REACHES (``POLLING_ONLY``). That last
bucket is the live configuration today: with no ESPHome power switch
wired, "offline" means Jarvis stops asking the radar, while the LD2410
keeps radiating and keeps serving presence to the LAN. Saying "the radar
is down" for that would be the exact overclaim this module exists to
prevent.
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

# How often the clock-driven guard walks the devices. The curfew is
# minute-granular, so this bounds how long a lens opened at 20:59 can stay
# open past 21:00; the pass itself is a clock read and a dict lookup, so
# the cost of making that bound small is nil.
DEFAULT_ENFORCE_S = 15.0


class _PollingOnly:
    """The return value of a ``stop()`` that could only stop the POLLING.

    TRUTHY, so a caller that just wants "did the stop work" still reads it
    as yes, but distinguishable from ``True`` where the difference is the
    whole point: the radar has no power switch wired on this box, so
    "offline" stops Jarvis asking and leaves the LD2410 radiating. The
    spoken line has to be able to say which of the two happened.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return "POLLING_ONLY"


POLLING_ONLY = _PollingOnly()


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
    # Stopped only as far as this process reaches: the polling stopped, the
    # device is still powered and still sensing the room. NOT a success --
    # it is the live radar configuration, and the one the spoken line would
    # otherwise dress up as "the radar is down".
    partial: tuple = ()
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
        # A SECOND lock, held only while device callbacks run and always
        # taken BEFORE _lock, never with _lock already held. It stops a
        # spoken switch and the clock guard interleaving their stops and
        # resumes on the same device, and it is separate from _lock
        # precisely because CameraGate.release must not be called under the
        # lock CameraGate.open asks for (that pair deadlocks).
        self._switch_lock = threading.RLock()
        self._devices: list[_Device] = []
        # name -> "we believe this device is running". enforce() acts only
        # on a CHANGE, so a device that refused to stop is retried on the
        # next pass instead of being written off as done.
        self._driven: dict[str, bool] = {}
        self._last_failed: tuple = ()
        self._persisted = True
        self._thread: Optional[threading.Thread] = None
        self._stop_ev = threading.Event()
        self._interval_s = DEFAULT_ENFORCE_S
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
        except Exception as exc:  # noqa: BLE001 - see below; NOTHING may escape
            # Deliberately blanket. A state file whose bytes are not valid
            # UTF-8 (an interrupted write, a bad block) raises
            # UnicodeDecodeError, which is a ValueError and slipped past an
            # `except OSError` -- and a constructor that raises is caught by
            # app._construct, leaving self.sensing None, the radar polling
            # ungoverned and the badge reading SENSING. That is a fail-ONLINE
            # on exactly the corrupt input his ruling names, so every way
            # this read can go wrong lands on the same fail-safe.
            log.warning("sensing: state %s unreadable (%s: %s); starting "
                        "OFFLINE", self.path, type(exc).__name__, exc)
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

    def _cfg_write(self, values: dict) -> bool:
        """Write several dotted keys and REPORT whether the file took them.

        ``AssistantConfig.set`` returns False rather than raising when the
        save fails, so the old three-``set`` version confirmed a window it
        may never have written -- and a refusal in the middle left the
        persisted start and end disagreeing. ``update`` is one save, so the
        pair cannot land half-written.
        """
        update = getattr(self._cfg, "update", None)
        if callable(update):
            return bool(update(dict(values)))
        setter = getattr(self._cfg, "set", None)
        if not callable(setter):
            return False
        ok = True
        for key, value in values.items():
            ok = (setter(key, value) is not False) and ok
        return ok

    def set_curfew(self, start: Optional[tuple], end: Optional[tuple]) -> bool:
        """Both ends at once, or ``(None, None)`` to switch the curfew off.
        Returns True when the config took the write."""
        try:
            if start is None or end is None:
                return self._cfg_write({CURFEW_ENABLED_KEY: False})
            return self._cfg_write({CURFEW_START_KEY: fmt_hhmm(start),
                                    CURFEW_END_KEY: fmt_hhmm(end),
                                    CURFEW_ENABLED_KEY: True})
        except Exception:  # noqa: BLE001
            log.exception("sensing: the curfew window could not be saved")
            return False

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

    def _may_run(self, state: SensingState, name: str) -> bool:
        """May the device attached under ``name`` run in ``state``?

        An attached name this module has never heard of is governed by the
        manual switch ALONE. That is deliberately not the rule ``allowed()``
        uses for an unknown sensor KIND (which is denied): there, "deny"
        costs nothing, whereas here it would stop a device the nightly
        camera window was never meant to touch and never resume it.
        """
        if name == CAMERA:
            return state.camera
        if name == RADAR:
            return state.radar
        return not state.offline

    def _switch_devices(self, on: bool, devices=None,
                        state: Optional[SensingState] = None) -> tuple:
        """Stop (or resume) devices. Returns (acted, failed, absent, partial)
        -- the material for a line that says what happened.

        NEVER call this while holding ``self._lock``: the stoppers are
        other objects' methods and they take their own locks (CameraGate
        takes its gate lock and then asks the policy), so running them
        under this one is the gate->policy / policy->gate cycle that
        deadlocked a camera open against a spoken "offline mode".
        """
        acted, failed, absent, partial = [], [], [], []
        for dev in list(self._devices if devices is None else devices):
            try:
                here = bool(dev.present())
            except Exception:  # noqa: BLE001
                log.debug("sensing: %s presence check failed", dev.name,
                          exc_info=True)
                here = True            # assume it is there and try anyway
            if not here:
                absent.append(dev.name)
                continue
            if on and state is not None and not self._may_run(state, dev.name):
                # "Back online" does not mean "open the lens": the nightly
                # curfew can still be running, and this is the one place a
                # privacy window could be silently overridden.
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
            if ok is False:
                failed.append(dev.name)
            elif isinstance(ok, _PollingOnly):
                partial.append(dev.name)
            else:
                acted.append(dev.name)
        if failed and tuple(failed) != self._last_failed:
            # Once per CHANGE of the failing set: enforce() retries a failed
            # stop every pass (privacy), and a warning a pass would bury the
            # log a dead radar is supposed to stay quiet in.
            log.warning("sensing: %s, but these did not follow: %s",
                        "online" if on else "offline", ", ".join(failed))
        self._last_failed = tuple(failed)
        return tuple(acted), tuple(failed), tuple(absent), tuple(partial)

    def _remember(self, devices, wanted, failed) -> None:
        """Record what each device was driven to, so ``enforce`` acts on
        transitions. A device that FAILED keeps its old value, which is what
        makes the next pass retry it."""
        with self._lock:
            for dev in devices:
                if dev.name in failed:
                    continue
                self._driven[dev.name] = bool(wanted(dev.name))

    # -------------------------------------------------------------- switch
    def disable(self, until: Optional[float] = None,
                source: str = "voice") -> Outcome:
        """Go offline. ``until`` is an epoch second for a bounded request."""
        with self._switch_lock:
            with self._lock:
                end = None
                if until:
                    try:
                        end = float(until)
                    except (TypeError, ValueError):
                        end = None
                if end is not None and end <= self._now():
                    # An end already past would be expired by the very next
                    # state() read, so this method would DENY and then
                    # report "allowed" in the same breath. Fail toward
                    # privacy: off open-endedly, and the spoken line reads
                    # out.state.until to say the bound was dropped.
                    log.warning("sensing: 'until %.0f' is not in the future; "
                                "going offline open-endedly", end)
                    end = None
                self._offline, self._failsafe, self._until = True, False, end
                persisted = self._save()
                # Taken HERE, under the lock, so a concurrent enable()
                # cannot interleave into the outcome "go offline" reports.
                state = self.state()
                devices = list(self._devices)
            stopped, failed, absent, partial = self._switch_devices(False,
                                                                    devices)
            self._remember(devices, lambda _n: False, failed)
        log.info("sensing: OFFLINE (%s%s); stopped=%s partial=%s failed=%s",
                 source, "" if end is None else " until %.0f" % end,
                 stopped, partial, failed)
        return Outcome(state, stopped=stopped, failed=failed, absent=absent,
                       partial=partial, persisted=persisted)

    def enable(self, source: str = "voice") -> Outcome:
        """Come back online, clearing the fail-safe as well as the switch.

        The devices are told, not just the flag: a radar whose power was
        cut has to be switched back on, and a resume that FAILED has to be
        spoken -- "back online" with a dead radar is the same class of lie
        as "offline" with a live one.
        """
        with self._switch_lock:
            with self._lock:
                self._offline, self._until, self._failsafe = False, None, False
                persisted = self._save()
                state = self.state()
                devices = list(self._devices)
            resumed, failed, absent, _p = self._switch_devices(True, devices,
                                                              state)
            self._remember(devices, lambda n: self._may_run(state, n), failed)
        log.info("sensing: ONLINE (%s); resumed=%s failed=%s", source,
                 resumed, failed)
        return Outcome(state, resumed=resumed, failed=failed, absent=absent,
                       persisted=persisted)

    # --------------------------------------------------- the clock-driven guard
    def enforce(self) -> Outcome:
        """Make the devices match the CURRENT verdict, whoever changed it.

        This is the half of enforcement that does not need anyone to speak.
        ``CameraGate`` only re-checks permission inside ``open()``, so a
        lens opened at 20:59 is still a lit camera at 21:01 unless something
        walks the devices on the clock; that is this. It stops what has just
        lost permission and resumes what has just got it back, and it acts
        only on the CHANGE, so a device already down is not re-stopped every
        pass.
        """
        with self._switch_lock:
            with self._lock:
                state = self.state()
                devices = list(self._devices)
                want = {d.name: self._may_run(state, d.name) for d in devices}
                off = [d for d in devices
                       if self._driven.get(d.name, True) and not want[d.name]]
                on = [d for d in devices
                      if not self._driven.get(d.name, True) and want[d.name]]
            stopped = resumed = absent = partial = ()
            failed: tuple = ()
            if off:
                stopped, failed, absent, partial = self._switch_devices(False,
                                                                        off)
            if on:
                resumed, f2, a2, _p = self._switch_devices(True, on, state)
                failed, absent = failed + f2, absent + a2
            if off or on:
                self._remember(off + on, lambda n: want[n], failed)
                log.info("sensing: enforced %s; stopped=%s resumed=%s "
                         "partial=%s failed=%s", state.reason or "clear",
                         stopped, resumed, partial, failed)
        return Outcome(state, stopped=stopped, resumed=resumed, failed=failed,
                       absent=absent, partial=partial,
                       persisted=self._persisted)

    def start(self, interval_s: float = DEFAULT_ENFORCE_S) -> None:
        """Run ``enforce()`` on a daemon thread until ``stop()``.

        Owned here rather than driven from ``MainWindow``'s 5 s pass on
        purpose: the curfew has to close a lens whether or not the console
        is up, and a privacy control that depends on a Tk thread being
        alive is not one.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            self._interval_s = max(0.01, float(interval_s))
        except (TypeError, ValueError):
            self._interval_s = DEFAULT_ENFORCE_S
        self._stop_ev.clear()
        self._thread = threading.Thread(target=self._watch, name="sensing",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the guard thread. Does NOT come back online: quitting is not
        consent, and the state file is what the next start reads."""
        self._stop_ev.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _watch(self) -> None:
        while not self._stop_ev.is_set():
            try:
                self.enforce()
            except Exception:  # noqa: BLE001 - the guard must outlive anything
                log.exception("sensing: enforcement pass failed")
            self._stop_ev.wait(self._interval_s)


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
        """The device, or None when sensing is denied.

        Never opens while denied, and closes a device this gate is still
        holding. It cannot close one on its OWN, though -- nothing here
        runs between two calls -- so the guard that shuts a lens when the
        21:00 curfew arrives with nobody speaking is ``SensingPolicy.enforce``
        on the policy's thread, which calls ``release`` below.
        """
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


# ------------------------------------------------- when there is no policy
class _DeniedPolicy:
    """The stand-in for a policy that could not be built AT ALL.

    ``app._construct`` swallows a constructor failure and hands back None,
    and a governed sensor whose ``policy`` is None falls back to "nobody is
    stopping me" -- a fail-ONLINE reached by the one path the fail-safe
    inside SensingPolicy cannot cover, because SensingPolicy is the thing
    that did not exist. So a sensor is handed THIS instead: it denies
    everything, for good, and only a restart that builds a real policy
    changes that.
    """

    def allowed(self, kind: Optional[str]) -> bool:
        return False

    def state(self) -> SensingState:
        return SensingState(camera=False, radar=False, offline=True,
                            reason=REASON_FAILSAFE, failsafe=True)

    def status(self) -> dict:
        return SensingPolicy.status(self)

    def attach(self, name, stop, present=None, resume=None) -> None:
        """Accepted and dropped: there is no owner to run the stoppers."""

    def curfew(self):
        return None


DENIED = _DeniedPolicy()
