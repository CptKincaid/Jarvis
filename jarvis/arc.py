"""The arc: one name for the hour of the house.

Seven phases -- pre-dawn, waking, working, afternoon, dusk, evening, night
-- derived from real sunrise/sunset, the wall clock, the quiet policy,
presence and a running focus block. Published as ``ArcChanged`` on the bus
and consumed by whoever wants it (room tone, the panel, a briefing that
wants to know whether it is greeting him or seeing him off).

INVARIANTS -- read these before extending the module.

* **The arc is a STATE SOURCE with no side effects.** It publishes one
  event on an accepted transition and does nothing else. It never speaks,
  never touches the UI, never writes a preference. Everything that is felt
  lives in a consumer.
* **quiet.py owns quiet, not the arc.** ``QuietPolicy`` already is a
  time+presence+calendar policy with deliberate asymmetries (free_until
  beats presence; can_speak defers mid-capture). The arc CONSUMES
  ``quiet.reason()`` and must never re-derive quiet hours nor gate speech
  itself, or two policies disagree at 23:00. It also never calls a setter
  on the config -- ``tests/test_arc.py`` asserts that.
* **Solar time is computed in-process.** ``weather.py``'s sunrise/sunset
  arrive over the network and need ``home_location`` lat/lon, which
  defaults to empty; an arc keyed off it would silently fall back to fixed
  hours on every cold boot and every network blip. ``sun_times()`` is the
  NOAA sunrise equation against stdlib datetime -- no network, no weather
  import, accurate to about a minute (checked against published tables for
  College Station and New York at both solstices). A forecast that is
  ALREADY cached may override it (``cached_sun_times``); nothing here ever
  triggers a fetch.
* **Publish only on an accepted transition, never per tick.** The
  hysteresis is a minimum dwell (``MIN_DWELL_S``), a tolerance band around
  the solar boundaries (``BAND_S``) and a forward-only rule within a local
  day, so a cached forecast refreshing with a sunset 20 minutes later
  cannot flap dusk back to afternoon.

``phase_at()`` is a pure function of its arguments -- ``previous`` /
``since`` / ``previous_forced`` come in rather than being held -- so the
whole engine is testable with a frozen clock.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from jarvis.events import ArcChanged, bus
from jarvis.logs import get_logger

log = get_logger("arc")

# Ordered: the day runs through these and (within one local date) may only
# advance. The order is the hysteresis, so do not reorder casually.
PHASES = ("pre-dawn", "waking", "working", "afternoon", "dusk", "evening", "night")
_ORDER = {name: i for i, name in enumerate(PHASES)}

# A phase holds for at least this long before the clock may advance it.
MIN_DWELL_S = 20 * 60.0
# Tolerance around a SOLAR boundary: a sunset that moves by less than this
# between two reads cannot move the phase.
BAND_S = 15 * 60.0

# An empty or hushed house has one mood, and it is not a warm one: nothing
# brightens, nothing chimes. quiet.py already answers "you're out" for an
# away phone, so away and quiet collapse to the same state by construction
# rather than by coincidence.
NEUTRAL_PHASE = "night"

# Fixed anchors for the two boundaries the sun does not set.
PRE_DAWN_HOUR = 4          # the night gives way to pre-dawn here
AFTERNOON_HOUR = 12
NIGHT_HOUR = 22

# Solar offsets. Waking starts a little before the sun is up and runs about
# two hours; the working boundary is clamped so a June sunrise does not put
# "working" at 07:30 nor a December one at 10:30.
WAKING_LEAD = timedelta(minutes=25)
WAKING_RUN = timedelta(hours=2)
WORK_EARLIEST = dtime(7, 30)
WORK_LATEST = dtime(9, 30)
DUSK_LEAD = timedelta(minutes=60)     # dusk opens before the sun is down
EVENING_LAG = timedelta(minutes=30)   # and closes after it

# Used only when no coordinates are configured and no forecast is cached.
FALLBACK_SUNRISE = dtime(6, 45)
FALLBACK_SUNSET = dtime(19, 45)

TICK_S = 60.0
STATE_NAME = "arc_state.json"


# ------------------------------------------------------------------ solar
J2000 = 2451545.0
_J2000_ORDINAL = date(2000, 1, 1).toordinal()
# Standard refraction-corrected solar disc centre at apparent sunrise.
_ZENITH_DEG = -0.833
_OBLIQUITY_DEG = 23.4397


def _from_julian(j: float) -> datetime:
    """Julian date -> aware UTC datetime (JD 2451545.0 is 2000-01-01 12:00Z)."""
    return datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=j - J2000)


def sun_times(lat: float, lon: float, day: date,
              tz=None) -> tuple[Optional[datetime], Optional[datetime]]:
    """(sunrise, sunset) for a local date, as aware datetimes in ``tz``.

    The NOAA sunrise equation, stdlib only -- no network and no weather
    import, so the arc is a function of the clock and nothing else.
    ``(None, None)`` above the polar circles, where the sun neither rises
    nor sets that day; callers fall back to fixed hours.
    """
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None, None
    n = day.toordinal() - _J2000_ORDINAL
    j_star = n - lon / 360.0                     # mean solar time at this meridian
    mean_anom = (357.5291 + 0.98560028 * j_star) % 360.0
    m_rad = math.radians(mean_anom)
    centre = (1.9148 * math.sin(m_rad) + 0.0200 * math.sin(2 * m_rad)
              + 0.0003 * math.sin(3 * m_rad))
    ecl_lon = (mean_anom + centre + 180.0 + 102.9372) % 360.0
    l_rad = math.radians(ecl_lon)
    j_transit = (J2000 + j_star + 0.0053 * math.sin(m_rad)
                 - 0.0069 * math.sin(2 * l_rad))
    sin_dec = math.sin(l_rad) * math.sin(math.radians(_OBLIQUITY_DEG))
    dec = math.asin(sin_dec)
    phi = math.radians(lat)
    try:
        cos_omega = ((math.sin(math.radians(_ZENITH_DEG)) - math.sin(phi) * sin_dec)
                     / (math.cos(phi) * math.cos(dec)))
    except ZeroDivisionError:                    # exactly at a pole
        return None, None
    if abs(cos_omega) > 1.0:
        return None, None                        # polar day or polar night
    omega = math.degrees(math.acos(cos_omega))
    rise = _from_julian(j_transit - omega / 360.0)
    sset = _from_julian(j_transit + omega / 360.0)
    if tz is None:
        return rise.astimezone(), sset.astimezone()
    return rise.astimezone(tz), sset.astimezone(tz)


def cached_sun_times(cfg, day: date, tz=None):
    """Sunrise/sunset from an ALREADY-CACHED forecast, or (None, None).

    An override, never a source: this reads ``weather``'s in-process cache
    and never fetches, so a cold boot or a dead network costs nothing. The
    local NOAA answer is what the arc actually runs on.
    """
    try:
        from jarvis.tools import weather as weather_mod
    except Exception:                            # noqa: BLE001 - optional
        return None, None
    try:
        loc = cfg.get("home_location") if hasattr(cfg, "get") else None
        lat, lon = (loc or {}).get("lat"), (loc or {}).get("lon")
        if lat is None or lon is None:
            return None, None
        key = weather_mod._key(float(lat), float(lon))
        with weather_mod._CACHE_LOCK:
            hit = weather_mod._CACHE.get(key)
        if not hit:
            return None, None
        daily = (hit[1] or {}).get("daily") or {}
        times = list(daily.get("time") or [])
        idx = times.index(day.isoformat())
        rise = weather_mod._iso((daily.get("sunrise") or [])[idx])
        sset = weather_mod._iso((daily.get("sunset") or [])[idx])
    except Exception:                            # noqa: BLE001 - cache shape
        log.debug("arc: cached sun times unavailable", exc_info=True)
        return None, None
    if rise is None or sset is None:
        return None, None
    zone = tz or datetime.now().astimezone().tzinfo
    # open-meteo returns location-local wall time with timezone=auto and no
    # offset; stamp it with the local zone rather than guessing UTC.
    rise = rise if rise.tzinfo else rise.replace(tzinfo=zone)
    sset = sset if sset.tzinfo else sset.replace(tzinfo=zone)
    return rise, sset


# -------------------------------------------------------------- the bands
def boundaries(day: date, *, sunrise: Optional[datetime] = None,
               sunset: Optional[datetime] = None, tz=None) -> list[tuple[str, datetime, bool]]:
    """The day's phase boundaries as (phase, starts_at, is_solar).

    Strictly increasing by construction: each boundary is clamped to at
    least a minute after the one before it, so a freak sunrise/sunset (high
    latitude, a garbled cache) can never produce an unorderable day.
    """
    zone = tz or (sunrise.tzinfo if sunrise is not None else None) \
        or datetime.now().astimezone().tzinfo

    def _at(t: dtime) -> datetime:
        return datetime.combine(day, t, tzinfo=zone)

    def _anchor(when: Optional[datetime], fallback: dtime) -> datetime:
        # The solar CLOCK TIME re-anchored onto `day`, not the datetime as
        # handed in: a sunrise carried over from yesterday's cache would
        # otherwise sit before every boundary and collapse the whole day
        # into the clamp.
        if when is None:
            return _at(fallback)
        return _at(when.astimezone(zone).timetz().replace(tzinfo=None))

    rise = _anchor(sunrise, FALLBACK_SUNRISE)
    sset = _anchor(sunset, FALLBACK_SUNSET)
    work = rise + WAKING_RUN
    work = min(max(work, _at(WORK_EARLIEST)), _at(WORK_LATEST))
    raw = [
        ("pre-dawn", _at(dtime(PRE_DAWN_HOUR, 0)), False),
        ("waking", rise - WAKING_LEAD, True),
        ("working", work, True),
        ("afternoon", _at(dtime(AFTERNOON_HOUR, 0)), False),
        ("dusk", sset - DUSK_LEAD, True),
        ("evening", sset + EVENING_LAG, True),
        ("night", _at(dtime(NIGHT_HOUR, 0)), False),
    ]
    out: list[tuple[str, datetime, bool]] = []
    for name, when, solar in raw:
        if out and when <= out[-1][1]:
            when = out[-1][1] + timedelta(minutes=1)
        out.append((name, when, solar))
    return out


def band_at(now: datetime, *, sunrise: Optional[datetime] = None,
            sunset: Optional[datetime] = None) -> str:
    """The phase the clock and the sun alone would name (no overrides)."""
    for name, when, _solar in reversed(boundaries(now.date(), sunrise=sunrise,
                                                  sunset=sunset, tz=now.tzinfo)):
        if now >= when:
            return name
    return "night"                               # before 04:00: still last night


def _band_start(now: datetime, phase: str, sunrise, sunset):
    """(starts_at, is_solar) for `phase` on now's date, or (None, False)."""
    for name, when, solar in boundaries(now.date(), sunrise=sunrise,
                                        sunset=sunset, tz=now.tzinfo):
        if name == phase:
            return when, solar
    return None, False


def forced_phase(quiet_reason: str = "", presence_state: str = "",
                 focus_phase: str = "") -> str:
    """The phase an override demands, or "" when the clock may decide.

    A focus block outranks everything: a man deliberately working at
    midnight is working, and the room should not act like it is bedtime at
    him. Otherwise a hushed house (quiet hours, DND, a meeting) or an empty
    one collapses to the neutral phase.
    """
    if str(focus_phase or "").strip().lower() == "block":
        return "working"
    if str(quiet_reason or "").strip():
        return NEUTRAL_PHASE
    if str(presence_state or "").strip().lower() == "away":
        return NEUTRAL_PHASE
    return ""


def phase_at(now: datetime, *, sunrise: Optional[datetime] = None,
             sunset: Optional[datetime] = None, quiet_reason: str = "",
             presence_state: str = "", focus_phase: str = "",
             previous: Optional[str] = None, since: Optional[float] = None,
             previous_forced: bool = False) -> str:
    """The phase of the house. Pure: state arrives as arguments.

    ``previous`` / ``since`` (a POSIX timestamp) / ``previous_forced`` are
    the caller's memory of the last accepted phase. Hysteresis applies only
    to the clock-and-sun band; an override is a discrete fact about the
    house and takes effect at once, in both directions.
    """
    forced = forced_phase(quiet_reason, presence_state, focus_phase)
    if forced:
        return forced
    band = band_at(now, sunrise=sunrise, sunset=sunset)
    if previous is None or previous not in _ORDER or previous_forced:
        # No memory, or the memory was an override that has just lifted:
        # the room must not lag behind the reason it was hushed.
        return band
    if band == previous:
        return band
    prev_day = None
    if since is not None:
        try:
            prev_day = datetime.fromtimestamp(float(since), tz=now.tzinfo).date()
        except (TypeError, ValueError, OSError, OverflowError):
            prev_day = None
    same_day = prev_day is not None and prev_day == now.date()
    if same_day and _ORDER[band] < _ORDER[previous]:
        # Forward-only within a day: a forecast refreshing with a later
        # sunset must not walk dusk back to afternoon.
        return previous
    if since is not None and same_day:
        try:
            held = now.timestamp() - float(since)
        except (TypeError, ValueError):
            held = MIN_DWELL_S
        if held < MIN_DWELL_S:
            return previous                      # minimum dwell not yet served
    start, solar = _band_start(now, band, sunrise, sunset)
    if solar and start is not None and (now - start).total_seconds() < BAND_S:
        # Inside the tolerance band around a solar boundary: the sun has to
        # be past it by BAND_S before the phase is allowed to move.
        return previous
    return band


# ------------------------------------------------------------ the engine
class Arc:
    """Names the hour on a daemon thread and publishes each transition.

    Nothing else. See the module docstring's invariants: no speech, no UI,
    no config writes.
    """

    def __init__(self, cfg, publish: Callable = bus.publish,
                 now: Callable[[], float] = time.time,
                 quiet=None, presence=None, focus=None,
                 state_path: Optional[Path] = None, tick_s: float = TICK_S,
                 sun_times_fn: Callable = sun_times):
        self._cfg = cfg
        self._publish = publish
        self._now = now
        self._quiet = quiet
        self._presence = presence
        self._focus = focus
        self._state_path = Path(state_path) if state_path else None
        self.tick_s = float(tick_s)
        self._sun_times = sun_times_fn
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.phase: str = ""
        self.since: float = 0.0
        self._forced = False
        self._sun_cache: tuple = ()               # (date, sunrise, sunset)
        self._load()

    # ------------------------------------------------------------ config
    def _get(self, key: str, default=None):
        get = getattr(self._cfg, "get", None)
        if not callable(get):
            return default
        try:
            value = get(key, default)
        except Exception:                         # noqa: BLE001 - config boundary
            log.debug("arc: cfg.get(%s) failed", key, exc_info=True)
            return default
        return default if value is None else value

    @property
    def enabled(self) -> bool:
        return bool(self._get("arc.enabled", True))

    # ------------------------------------------------------------- state
    def _load(self) -> None:
        if not self._state_path:
            return
        try:
            if not self._state_path.exists():
                return
            data = json.loads(self._state_path.read_text())
        except (OSError, ValueError):
            log.debug("arc state unreadable", exc_info=True)
            return
        if not isinstance(data, dict):
            return
        phase = data.get("phase")
        if phase in _ORDER:
            self.phase = phase
            try:
                self.since = float(data.get("since") or 0.0)
            except (TypeError, ValueError):
                self.since = 0.0
            self._forced = bool(data.get("forced"))

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps({"phase": self.phase, "since": self.since,
                                       "forced": self._forced}))
            os.replace(tmp, self._state_path)     # atomic: never half a state
        except OSError:
            log.debug("arc state save failed", exc_info=True)

    # ------------------------------------------------------------ inputs
    @staticmethod
    def _deref(source):
        """The input, or what a late-binding callable resolves to.

        app.py builds the arc before the focus session and the services
        exist, so it hands over ``lambda: self.focus`` rather than a member
        that is still None. None of the real inputs (QuietPolicy,
        PresenceSentinel, FocusSession) is callable, so this cannot be
        confused by one.
        """
        try:
            return source() if callable(source) else source
        except Exception:            # noqa: BLE001 - a late binding may not be ready
            return None

    def _quiet_reason(self) -> str:
        fn = getattr(self._deref(self._quiet), "reason", None)
        if not callable(fn):
            return ""
        try:
            return str(fn() or "")
        except Exception:                         # noqa: BLE001 - policy boundary
            log.debug("arc: quiet.reason failed", exc_info=True)
            return ""

    def _presence_state(self) -> str:
        try:
            return str(getattr(self._deref(self._presence), "state", "") or "")
        except Exception:                         # noqa: BLE001
            return ""

    def _focus_phase(self) -> str:
        try:
            return str(getattr(self._deref(self._focus), "phase", "") or "")
        except Exception:                         # noqa: BLE001
            return ""

    def sun(self, day: date, tz=None):
        """(sunrise, sunset) for a local date, cached for the day."""
        if self._sun_cache and self._sun_cache[0] == day:
            return self._sun_cache[1], self._sun_cache[2]
        rise = sset = None
        loc = self._get("home_location") or {}
        lat, lon = loc.get("lat"), loc.get("lon")
        if lat is not None and lon is not None:
            try:
                rise, sset = self._sun_times(lat, lon, day, tz)
            except Exception:                     # noqa: BLE001 - arithmetic boundary
                log.debug("arc: solar computation failed", exc_info=True)
                rise = sset = None
        if rise is None or sset is None:
            # No coordinates (home_location defaults to empty) or polar:
            # a cached forecast is the only other local answer.
            c_rise, c_sset = cached_sun_times(self._cfg, day, tz)
            rise, sset = c_rise or rise, c_sset or sset
        self._sun_cache = (day, rise, sset)
        return rise, sset

    # -------------------------------------------------------------- tick
    def tick(self) -> Optional[ArcChanged]:
        """Name the hour; publish and return the event on a transition."""
        if not self.enabled:
            return None
        now = datetime.fromtimestamp(self._now()).astimezone()
        rise, sset = self.sun(now.date(), now.tzinfo)
        # Read each input ONCE per tick: quiet.reason() walks the calendar,
        # and two reads a tick could disagree mid-transition.
        reason, presence, focus = (self._quiet_reason(), self._presence_state(),
                                   self._focus_phase())
        forced = forced_phase(reason, presence, focus)
        with self._lock:
            previous, since, prev_forced = self.phase, self.since, self._forced
        phase = phase_at(now, sunrise=rise, sunset=sset, quiet_reason=reason,
                         presence_state=presence, focus_phase=focus,
                         previous=previous or None, since=since or None,
                         previous_forced=prev_forced)
        if phase == previous:
            if bool(forced) != prev_forced:
                # Same name, different reason (quiet lifted at 23:30 and the
                # clock says night anyway): remember WHY, so the next lift
                # does not look like an override falling away.
                with self._lock:
                    self._forced = bool(forced)
                self._save()
            return None
        stamp = now.timestamp()
        with self._lock:
            self.phase, self.since, self._forced = phase, stamp, bool(forced)
        self._save()
        event = ArcChanged(phase=phase, previous=previous, since=stamp,
                           forced=bool(forced),
                           sunrise=rise.timestamp() if rise is not None else 0.0,
                           sunset=sset.timestamp() if sset is not None else 0.0)
        log.info("arc: %s -> %s%s", previous or "(none)", phase,
                 f" (forced by {reason or presence or focus})" if forced else "")
        try:
            self._publish(event)
        except Exception:                         # noqa: BLE001
            log.exception("arc: publish failed")
        return event

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        # Alive-guard + clear, like deadlines.py: a stopped instance can be
        # started again (tests, a config reload).
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="arc")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("arc tick failed")
            if self._stop.wait(self.tick_s):
                return
