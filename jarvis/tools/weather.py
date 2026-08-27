"""Weather tool (spec 2026-08-26, section 6.3) on Open-Meteo, no API key.

One forecast call (current + hourly + daily, US units, the location's own
timezone, 7 days) cached 10 minutes per rounded (lat, lon).  ``format_weather``
renders the four spoken shapes — now / today / tomorrow / week — as compact
plain text the local model rephrases.  The location comes from
``location.geocode`` when the model names a place and from
``location.resolve_home`` otherwise.  Every request of this module,
geocoding included, goes through the module-level ``_fetch`` seam.
"""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import date, datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from jarvis.logs import get_logger
from jarvis.tools import location as _location
from jarvis.tools.location import (Location, cache_dir, cfg_get, clock_words,
                                   geocode, http_get, is_home_word,
                                   resolve_home, setup_line)
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.weather")

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
    "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,"
    "precipitation,relative_humidity_2m"
    "&hourly=temperature_2m,precipitation_probability,weather_code"
    "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
    "precipitation_probability_max,sunrise,sunset"
    "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    "&timezone=auto&forecast_days=7")
CACHE_TTL = 600                 # seconds a forecast stays fresh
FETCH_TIMEOUT = 6
UNREACHABLE = "weather service unreachable"
WHENS = ("now", "today", "tomorrow", "week")

# WMO 4677 weather interpretation codes -> plain words.
WMO_CODES = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}
SNOW_CODES = {71, 73, 75, 77, 85, 86}
_PRECIP_WORDS = ("rain", "drizzle", "showers", "snow", "thunder", "fog", "hail")


def _fetch(url: str, timeout: float = FETCH_TIMEOUT, headers: Optional[dict] = None) -> bytes:
    """Test seam: every request of this module goes through here."""
    return http_get(url, timeout=timeout, headers=headers)


# ------------------------------------------------------------------ words
def describe_code(code, daytime: bool = False) -> str:
    """WMO code -> words; clear skies by day are "sunny"."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "unknown conditions"
    if daytime and code == 0:
        return "sunny"
    if daytime and code == 1:
        return "mostly sunny"
    return WMO_CODES.get(code, "unsettled")


def _joiner(words: str) -> str:
    """"88 and sunny" / "79 with showers"."""
    return "with" if any(w in words for w in _PRECIP_WORDS) else "and"


def _rnd(value) -> Optional[int]:
    try:
        return int(math.floor(float(value) + 0.5))
    except (TypeError, ValueError):
        return None


def _pct(value) -> str:
    n = _rnd(value)
    return f"{n}%" if n is not None else "unknown"


def _iso(text: str, tz=None) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    if tz is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def _tz(data: dict):
    name = (data or {}).get("timezone") or ""
    try:
        return ZoneInfo(name) if name else None
    except Exception:  # noqa: BLE001 - odd zone names from the service
        return None


def _hm(text: str) -> str:
    dt = _iso(text)
    return clock_words(dt) if dt else "unknown"


def _daily(data: dict, i: int) -> dict:
    daily = data.get("daily") or {}
    out = {}
    for key, values in daily.items():
        try:
            out[key] = values[i]
        except (IndexError, TypeError):
            out[key] = None
    return out


def _rain_words(day: dict) -> str:
    what = "snow" if _rnd(day.get("weather_code")) in SNOW_CODES else "rain"
    return f"{_pct(day.get('precipitation_probability_max'))} chance of {what}"


def _day_label(day: date, today: date, capital: bool) -> str:
    if day == today:
        word = "today"
    elif day == today + timedelta(days=1):
        word = "tomorrow"
    else:
        return day.strftime("%A")
    return word.capitalize() if capital else word


def _is_daytime(when: datetime, day: dict) -> bool:
    rise, sset = _iso(day.get("sunrise")), _iso(day.get("sunset"))
    if not rise or not sset:
        return 6 <= when.hour < 20
    naive = when.replace(tzinfo=None)
    return rise.replace(tzinfo=None) <= naive <= sset.replace(tzinfo=None)


def forecast_now(data: dict) -> Optional[datetime]:
    """The forecast's own clock (location-local, aware when a tz is known)."""
    return _iso((data.get("current") or {}).get("time"), _tz(data))


def _local_now(data: dict, now: Optional[datetime]) -> datetime:
    tz = _tz(data)
    if now is None:
        return forecast_now(data) or datetime.now(tz)
    if now.tzinfo is not None and tz is not None:
        return now.astimezone(tz)
    return now


def format_weather(data: dict, when: str = "now", now: datetime = None) -> str:
    """Compact plain text for one of now / today / tomorrow / week."""
    when = coerce_when(when)
    now = _local_now(data, now)
    today = now.date()
    day0 = _daily(data, 0)
    if when == "now":
        cur = data.get("current") or {}
        words = describe_code(cur.get("weather_code"), _is_daytime(now, day0))
        return (f"{_rnd(cur.get('temperature_2m'))}°F and {words}, feels like "
                f"{_rnd(cur.get('apparent_temperature'))}, wind "
                f"{_rnd(cur.get('wind_speed_10m'))} mph; high "
                f"{_rnd(day0.get('temperature_2m_max'))}, low "
                f"{_rnd(day0.get('temperature_2m_min'))}, {_rain_words(day0)}.")
    if when in ("today", "tomorrow"):
        i = 0 if when == "today" else 1
        day = _daily(data, i)
        if day.get("temperature_2m_max") is None:
            return f"no forecast for {when}"
        return (f"{when.capitalize()}: high {_rnd(day.get('temperature_2m_max'))}, "
                f"low {_rnd(day.get('temperature_2m_min'))}, "
                f"{describe_code(day.get('weather_code'), True)}, "
                f"{_rain_words(day)}, sunset {_hm(day.get('sunset'))}.")
    clauses = []
    for i, text in enumerate((data.get("daily") or {}).get("time") or []):
        if i >= 7:
            break
        day = _daily(data, i)
        d = _iso(text)
        if d is None or day.get("temperature_2m_max") is None:
            continue
        words = describe_code(day.get("weather_code"), True)
        clauses.append(f"{_day_label(d.date(), today, not clauses)} "
                       f"{_rnd(day.get('temperature_2m_max'))} {_joiner(words)} {words}")
    return (", ".join(clauses) + ".") if clauses else "no forecast for the week"


# ----------------------------------------------------------------- cache
_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple, tuple[float, dict]] = {}
_clock = time.time      # test seam for the cache clock


def _key(lat: float, lon: float) -> tuple:
    return (round(float(lat), 2), round(float(lon), 2))


def forecast_url(lat: float, lon: float) -> str:
    return FORECAST_URL.format(lat=f"{float(lat):.4f}", lon=f"{float(lon):.4f}")


def fetch_forecast(lat: float, lon: float, fetch: Callable = None, now: float = None) -> dict:
    """The cached forecast for (lat, lon); raises when the service fails."""
    fetch = fetch or _fetch
    now = _clock() if now is None else now
    key = _key(lat, lon)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]
    data = json.loads(fetch(forecast_url(lat, lon), timeout=FETCH_TIMEOUT))
    if not isinstance(data, dict) or data.get("error") or "current" not in data:
        raise ValueError(f"unexpected forecast body: {str(data)[:80]}")
    with _CACHE_LOCK:
        _CACHE[key] = (now, data)
    return data


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def coerce_when(value) -> str:
    """Loose model values -> one of WHENS ("this week" -> "week")."""
    text = " ".join(str(value or "").split()).lower().strip(" .?!")
    if not text:
        return "now"
    if text in WHENS:
        return text
    if "tomorrow" in text or text in ("tmrw", "tmr", "next day"):
        return "tomorrow"
    if "week" in text or "7 day" in text or "seven day" in text or "forecast" in text:
        return "week"
    if "today" in text or "rest of the day" in text or "tonight" in text or \
            "this evening" in text or "this afternoon" in text:
        return "today"
    if "now" in text or "current" in text or "outside" in text or "at the moment" in text:
        return "now"
    return "now"


# ----------------------------------------------------------------- tools
def resolve_place(cfg, location, cache_path, fetch: Callable) -> tuple:
    """(Location | None, ToolResult | None): a named place is geocoded,
    home comes from the config / IP lookup; the second slot carries the
    excuse when nothing could be resolved."""
    place = " ".join(str(location or "").split())
    if not is_home_word(place):
        loc = geocode(place, fetch=fetch, cache_path=cache_path)
        if loc is None:
            return None, ToolResult(text=f"I couldn't find a place called {place}",
                                    ok=False)
        return loc, None
    loc = resolve_home(cfg, cache_path, fetch=fetch)
    if loc is None:
        if cfg_get(cfg, "location_lookup", True) is False:
            line = setup_line(cfg, "home_location")
            return None, ToolResult(text=line, ok=False, speak=line)
        return None, ToolResult(text="location lookup unreachable", ok=False)
    return loc, None


def make_tools(cfg, services) -> list[ToolSpec]:
    cache_path = cache_dir() / "location.json"

    def get_weather(when="now", location="", **_) -> ToolResult:
        when = coerce_when(when)
        loc, excuse = resolve_place(cfg, location, cache_path, _fetch)
        if excuse is not None:
            return excuse
        try:
            data = fetch_forecast(loc.lat, loc.lon, fetch=_fetch)
        except Exception as exc:  # noqa: BLE001 - one-clause failure text
            log.warning("forecast for %s failed: %s", loc.label(), exc)
            return ToolResult(text=UNREACHABLE, ok=False)
        text = format_weather(data, when)
        if not is_home_word(location):
            text = f"In {loc.label()}: {text}"
        return ToolResult(text=text)

    return [ToolSpec(
        name="get_weather",
        description="Weather now, today, tomorrow or this week, at home or in a city.",
        parameters={"type": "object", "properties": {
            "when": {"type": "string", "enum": list(WHENS),
                     "description": "now, today, tomorrow or week"},
            "location": {"type": "string",
                         "description": "City name; omit for home."}}},
        handler=get_weather)]
