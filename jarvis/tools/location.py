"""Location and time tools (spec 2026-08-26, section 6.2).

`resolve_home` answers "where is Hunter": the configured ``home_location``
when it carries lat/lon, a geocode of its city when only a name is set,
else an IP lookup (ipapi.co, then ip-api.com) cached for 24 h in
``~/.cache/jarvis/location.json``.  `geocode` turns a spoken place name
into coordinates + timezone through Open-Meteo's geocoder (cached per name,
in memory and in the same file).  The tools ``get_location`` and
``get_time`` build on those; ``get_time`` renders the clock through
``zoneinfo`` for a named city and falls back to ``datetime.now()`` for home.

Every HTTP request goes through the module-level ``_fetch`` seam (tests
monkeypatch it); nothing here blocks longer than LOOKUP_BUDGET_S (4 s).
The other tool modules import ``http_get`` / the config helpers from here
so there is exactly one urllib wrapper in the package.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.location")

IPAPI_URL = "https://ipapi.co/json/"
IP_API_URL = "http://ip-api.com/json/"
GEOCODE_URL = ("https://geocoding-api.open-meteo.com/v1/search"
               "?name={q}&count=1&language=en")
LOCATION_CACHE_TTL = 24 * 3600      # seconds an IP lookup stays valid
LOOKUP_BUDGET_S = 4.0               # hard wall-clock cap for any lookup
USER_AGENT = "Jarvis/3 (+https://github.com/hunterp/Jarvis)"
HOME_WORDS = {"", "home", "here", "local", "my location", "current",
              "current location", "where i am", "my city", "none", "null"}

# Persona excuses when AssistantConfig (section 10.2) is not available yet.
DEFAULT_SETUP_LINES = {
    "home_location": ("I'll need your home location set up, sir; the notes "
                      "are in docs/assistant-setup.md."),
    "google_ical": ("I'll need your Google calendar link set up, sir; the "
                    "notes are in docs/assistant-setup.md."),
    "icloud": ("I'll need your iCloud calendar set up, sir; the notes are "
               "in docs/assistant-setup.md."),
}


# ------------------------------------------------------------------ HTTP
class Response(bytes):
    """``bytes`` with the HTTP status and (lower-cased) headers attached, so
    a plain-bytes stand-in from a test still satisfies every caller."""
    status: int = 200
    headers: dict


def http_get(url: str, timeout: float = 6, headers: Optional[dict] = None) -> Response:
    """The one urllib wrapper of the tools package.  Raises on transport
    errors and on HTTP errors other than 304 (which comes back as an empty
    Response with ``status == 304`` for conditional GETs)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "*/*",
                      **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = Response(resp.read())
            body.status = int(resp.status or 200)
            body.headers = {k.lower(): v for k, v in resp.headers.items()}
            return body
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            body = Response(b"")
            body.status = 304
            body.headers = {k.lower(): v for k, v in (exc.headers or {}).items()}
            return body
        raise


def _fetch(url: str, timeout: float = 6, headers: Optional[dict] = None) -> bytes:
    """Test seam: every request of this module goes through here."""
    return http_get(url, timeout=timeout, headers=headers)


# ------------------------------------------------------------ config glue
def cfg_get(cfg, dotted: str, default=None):
    """Read ``a.b.c`` from an AssistantConfig (``.get(dotted)``), a dict, or
    any attribute-bearing stand-in — whichever the caller wired in."""
    if cfg is None:
        return default
    if not isinstance(cfg, dict):
        getter = getattr(cfg, "get", None)
        if callable(getter):
            try:
                value = getter(dotted, None)
            except TypeError:
                value = None
            if value is not None:
                return value
    cur = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def setup_line(cfg, section: str) -> str:
    """``cfg.setup_line(section)`` when the config offers it, else the
    matching default persona line."""
    fn = getattr(cfg, "setup_line", None)
    if callable(fn):
        try:
            line = fn(section)
            if line:
                return str(line)
        except Exception:  # noqa: BLE001 - config stand-ins vary
            log.debug("setup_line(%s) failed on %r", section, type(cfg))
    return DEFAULT_SETUP_LINES.get(
        section, f"I'll need {section.replace('_', ' ')} set up, sir; the "
                 "notes are in docs/assistant-setup.md.")


def cache_dir() -> Path:
    """``PATHS.CACHE_DIR`` (env JARVIS_CACHE_DIR in tests) or ~/.cache/jarvis."""
    try:
        from jarvis.config import PATHS
        path = getattr(PATHS, "CACHE_DIR", None)
        if path:
            return Path(path)
    except Exception:  # noqa: BLE001 - config import must never break a tool
        pass
    return Path(os.environ.get("JARVIS_CACHE_DIR") or
                (Path.home() / ".cache" / "jarvis"))


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - a bad cache is just a miss
        log.warning("cache %s unreadable: %s", path, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - cache writes are best effort
        log.warning("cache %s not written: %s", path, exc)


# --------------------------------------------------------------- Location
@dataclass
class Location:
    city: str = ""
    region: str = ""
    country: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    tz: str = ""
    source: str = ""        # config | geocode | ip | cache

    def label(self) -> str:
        """"Chicago, Illinois" / "Tokyo, Japan" / "Chicago"."""
        second = self.region if self.region and self.region != self.city \
            else self.country
        return ", ".join(p for p in (self.city, second) if p)

    def tzinfo(self):
        if not self.tz:
            return None
        try:
            return ZoneInfo(self.tz)
        except Exception:  # noqa: BLE001 - unknown zone names from services
            log.warning("unknown timezone %r", self.tz)
            return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Location":
        loc = cls()
        for key in loc.__dataclass_fields__:
            if key in data:
                setattr(loc, key, data[key])
        return loc


def system_tz():
    """The machine's zone (America/Chicago on the Spark) as a tzinfo."""
    return datetime.now().astimezone().tzinfo


def _system_tz_name() -> str:
    tz = system_tz()
    key = getattr(tz, "key", "")
    if key:
        return key
    try:
        target = os.readlink("/etc/localtime")
        if "zoneinfo/" in target:
            return target.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    return ""


def _num(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ipapi(data: dict) -> Optional[Location]:
    if not isinstance(data, dict) or data.get("error"):
        return None
    lat, lon = _num(data.get("latitude")), _num(data.get("longitude"))
    if lat is None or lon is None:
        return None
    return Location(city=str(data.get("city") or ""),
                    region=str(data.get("region") or ""),
                    country=str(data.get("country_name") or ""),
                    lat=lat, lon=lon, tz=str(data.get("timezone") or ""),
                    source="ip")


def _parse_ip_api(data: dict) -> Optional[Location]:
    if not isinstance(data, dict) or data.get("status") != "success":
        return None
    lat, lon = _num(data.get("lat")), _num(data.get("lon"))
    if lat is None or lon is None:
        return None
    return Location(city=str(data.get("city") or ""),
                    region=str(data.get("regionName") or data.get("region") or ""),
                    country=str(data.get("country") or ""),
                    lat=lat, lon=lon, tz=str(data.get("timezone") or ""),
                    source="ip")


IP_SERVICES = ((IPAPI_URL, _parse_ipapi), (IP_API_URL, _parse_ip_api))


def lookup_ip(fetch: Callable = None, budget_s: float = LOOKUP_BUDGET_S) -> Optional[Location]:
    """ipapi.co first, ip-api.com second, never more than ``budget_s`` in
    total.  A rate-limit / error body counts as a failure and falls through."""
    fetch = fetch or _fetch
    deadline = time.monotonic() + budget_s
    for url, parse in IP_SERVICES:
        remaining = deadline - time.monotonic()
        if remaining < 0.5:
            break
        try:
            raw = fetch(url, timeout=min(3.0, remaining))
            loc = parse(json.loads(raw or b"{}"))
        except Exception as exc:  # noqa: BLE001 - try the next service
            log.warning("ip lookup %s failed: %s", url, exc)
            continue
        if loc is not None:
            return loc
        log.warning("ip lookup %s gave no usable answer", url)
    return None


def resolve_home(cfg, cache_path=None, fetch: Callable = None, now: float = None) -> Optional[Location]:
    """Where home is.  Precedence: config lat/lon -> geocoded config city
    -> cached IP lookup (< 24 h) -> fresh IP lookup -> stale cache -> None.
    ``cfg.location_lookup = false`` disables the network entirely."""
    fetch = fetch or _fetch
    cache_path = Path(cache_path) if cache_path else cache_dir() / "location.json"
    now = time.time() if now is None else now
    home = cfg_get(cfg, "home_location") or {}
    if not isinstance(home, dict):
        home = {}
    lat, lon = _num(home.get("lat")), _num(home.get("lon"))
    city = str(home.get("city") or "").strip()
    if lat is not None and lon is not None:
        return Location(city=city, region=str(home.get("region") or ""),
                        country=str(home.get("country") or ""), lat=lat,
                        lon=lon, tz=str(home.get("tz") or "") or _system_tz_name(),
                        source="config")
    if city:
        query = ", ".join(p for p in (city, str(home.get("region") or "")) if p)
        loc = geocode(query, fetch=fetch, cache_path=cache_path) or \
            (geocode(city, fetch=fetch, cache_path=cache_path) if query != city else None)
        if loc is not None:
            loc.source = "config"
            return loc
        log.warning("home city %r could not be geocoded", city)
    lookup = cfg_get(cfg, "location_lookup", True)
    if lookup is False or (isinstance(lookup, str) and lookup.lower() in ("false", "0", "no")):
        return None
    cached = _read_json(cache_path)
    entry = cached.get("home") if isinstance(cached.get("home"), dict) else None
    if entry and now - float(entry.get("fetched_at") or 0) < LOCATION_CACHE_TTL:
        loc = Location.from_dict(entry)
        loc.source = "ip"
        return loc
    loc = lookup_ip(fetch)
    if loc is not None:
        cached["home"] = {**loc.to_dict(), "fetched_at": now}
        _write_json(cache_path, cached)
        return loc
    if entry:
        log.warning("ip lookup failed; using the stale cached location")
        loc = Location.from_dict(entry)
        loc.source = "ip"
        return loc
    return None


# ---------------------------------------------------------------- geocode
_GEOCODE_LOCK = threading.Lock()
_GEOCODE_CACHE: dict[str, Optional[dict]] = {}


def _geocode_key(name: str) -> str:
    return " ".join(str(name or "").split()).lower()


def _parse_geocode(data: dict) -> Optional[Location]:
    results = (data or {}).get("results") or []
    if not results:
        return None
    r = results[0]
    lat, lon = _num(r.get("latitude")), _num(r.get("longitude"))
    if lat is None or lon is None:
        return None
    return Location(city=str(r.get("name") or ""),
                    region=str(r.get("admin1") or ""),
                    country=str(r.get("country") or ""),
                    lat=lat, lon=lon, tz=str(r.get("timezone") or ""),
                    source="geocode")


def geocode(name: str, fetch: Callable = None, cache_path=None,
            timeout: float = LOOKUP_BUDGET_S) -> Optional[Location]:
    """Open-Meteo geocoder, first hit only; cached per normalised name in
    memory and in the location cache file (misses are remembered for the
    process only, so a typo does not poison the disk cache)."""
    fetch = fetch or _fetch
    key = _geocode_key(name)
    if not key:
        return None
    with _GEOCODE_LOCK:
        if key in _GEOCODE_CACHE:
            hit = _GEOCODE_CACHE[key]
            return Location.from_dict(hit) if hit else None
    cache_path = Path(cache_path) if cache_path else cache_dir() / "location.json"
    cached = _read_json(cache_path)
    disk = cached.get("geocode") if isinstance(cached.get("geocode"), dict) else {}
    if isinstance(disk.get(key), dict):
        with _GEOCODE_LOCK:
            _GEOCODE_CACHE[key] = disk[key]
        return Location.from_dict(disk[key])
    url = GEOCODE_URL.format(q=urllib.parse.quote(key))
    try:
        loc = _parse_geocode(json.loads(fetch(url, timeout=timeout) or b"{}"))
    except Exception as exc:  # noqa: BLE001 - the tool explains the miss
        log.warning("geocode %r failed: %s", key, exc)
        return None
    with _GEOCODE_LOCK:
        _GEOCODE_CACHE[key] = loc.to_dict() if loc else None
    if loc is not None:
        cached = _read_json(cache_path)
        cached.setdefault("geocode", {})
        if isinstance(cached["geocode"], dict):
            cached["geocode"][key] = loc.to_dict()
            _write_json(cache_path, cached)
    return loc


def clear_caches() -> None:
    """Tests: forget every in-memory geocode answer."""
    with _GEOCODE_LOCK:
        _GEOCODE_CACHE.clear()


# ------------------------------------------------------------------ time
def clock_words(dt: datetime) -> str:
    """4:05 pm / 12:00 am — the 12-hour form every tool uses."""
    hour = dt.hour % 12 or 12
    return f"{hour}:{dt.minute:02d} {'am' if dt.hour < 12 else 'pm'}"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def date_words(dt: datetime) -> str:
    """Wednesday the 26th of August."""
    return f"{dt.strftime('%A')} the {_ordinal(dt.day)} of {dt.strftime('%B')}"


def format_time(dt: datetime, place: str = "") -> str:
    """"It's 4:05 pm in Tokyo, sir." / "It's 4:05 pm, sir." """
    where = f" in {place}" if place else ""
    return f"It's {clock_words(dt)}{where}, sir."


def now_in(loc: Optional[Location], now: datetime = None) -> datetime:
    """The current wall clock at ``loc`` (its tz) or at home (system tz)."""
    tz = loc.tzinfo() if loc is not None else None
    if now is None:
        return datetime.now(tz) if tz is not None else datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()
    return now.astimezone(tz) if tz is not None else now.astimezone(system_tz())


def is_home_word(text) -> bool:
    return " ".join(str(text or "").split()).lower().strip(" .?!") in HOME_WORDS


# ----------------------------------------------------------------- tools
def make_tools(cfg, services) -> list[ToolSpec]:
    cache_path = cache_dir() / "location.json"

    def get_location(**_) -> ToolResult:
        loc = resolve_home(cfg, cache_path, fetch=_fetch)
        if loc is None or not loc.label():
            lookup = cfg_get(cfg, "location_lookup", True)
            if lookup is False:
                line = setup_line(cfg, "home_location")
                return ToolResult(text=line, ok=False, speak=line)
            return ToolResult(text="location lookup unreachable", ok=False)
        origin = ("from your settings" if loc.source in ("config", "geocode")
                  else "from your network address")
        return ToolResult(text=f"You're in {loc.label()}, sir ({origin}).")

    def get_time(location="", **_) -> ToolResult:
        place = " ".join(str(location or "").split())
        if is_home_word(place):
            home = resolve_home(cfg, cache_path, fetch=_fetch)
            dt = now_in(home) if home is not None and home.tz else \
                datetime.now().astimezone()
            return ToolResult(text=f"{format_time(dt)} Today is {date_words(dt)}.")
        loc = geocode(place, fetch=_fetch, cache_path=cache_path)
        if loc is None or not loc.tz:
            return ToolResult(text=f"I couldn't find a place called {place}",
                              ok=False)
        dt = now_in(loc)
        return ToolResult(text=f"{format_time(dt, loc.city)} It's "
                               f"{date_words(dt)} there.")

    return [
        ToolSpec(
            name="get_time",
            description="Current time and date at home or in a named city.",
            parameters={"type": "object", "properties": {
                "location": {"type": "string",
                             "description": "City name; omit for home."}}},
            handler=get_time),
        ToolSpec(
            name="get_location",
            description="Where Hunter is right now (home city and region).",
            parameters={"type": "object", "properties": {}},
            handler=get_location),
    ]
