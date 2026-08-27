"""Tests for jarvis.tools.location and jarvis.tools.weather (spec 6.2/6.3).

Unit tests never touch the network: every request goes through the
modules' ``_fetch`` seams, which are replaced with canned JSON.  Caches are
written to tmp paths only (tests/conftest.py also redirects JARVIS_CACHE_DIR
and JARVIS_ASSISTANT_CONFIG).  Live tests run only with JARVIS_LIVE=1 and
are read-only against Open-Meteo / the geocoder.
"""
import json
import os
import time
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import jarvis.tools.location as location
import jarvis.tools.weather as weather
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec

CHI = ZoneInfo("America/Chicago")
NOW = datetime(2026, 8, 26, 9, 0, tzinfo=CHI)          # a Wednesday


# ------------------------------------------------------------- fixtures
class FakeCfg:
    """Stand-in for AssistantConfig (section 10.2): get(dotted), setup_line."""

    def __init__(self, **over):
        self.data = {"home_location": {"city": "", "region": "", "lat": None, "lon": None},
                     "location_lookup": True, "units": "us"}
        self.data.update(over)

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def setup_line(self, section):
        return f"I'll need your {section} set up, sir; the notes are in docs/assistant-setup.md."


IPAPI_OK = {"ip": "1.2.3.4", "city": "Chicago", "region": "Illinois",
            "country_name": "United States", "latitude": 41.85,
            "longitude": -87.65, "timezone": "America/Chicago"}
IPAPI_LIMITED = {"error": True, "reason": "RateLimited"}
IP_API_OK = {"status": "success", "country": "United States", "regionName": "Texas",
             "city": "Tyler", "lat": 32.3251, "lon": -95.2981,
             "timezone": "America/Chicago"}
GEO_TOKYO = {"results": [{"name": "Tokyo", "latitude": 35.6895, "longitude": 139.69171,
                          "country": "Japan", "admin1": "Tokyo",
                          "timezone": "Asia/Tokyo"}]}
GEO_NONE = {"generationtime_ms": 0.3}


def geo_route(url, timeout=None, headers=None):
    """The geocoder knows Tokyo and nothing else."""
    return GEO_TOKYO if "tokyo" in url.lower() else GEO_NONE

FORECAST = {
    "latitude": 41.85, "longitude": -87.65, "timezone": "America/Chicago",
    "current": {"time": "2026-08-26T09:00", "temperature_2m": 72.4,
                "apparent_temperature": 75.1, "weather_code": 2,
                "wind_speed_10m": 8.3, "precipitation": 0.0,
                "relative_humidity_2m": 60},
    "hourly": {"time": ["2026-08-26T09:00"], "temperature_2m": [72.4],
               "precipitation_probability": [10], "weather_code": [2]},
    "daily": {
        "time": ["2026-08-26", "2026-08-27", "2026-08-28", "2026-08-29",
                 "2026-08-30", "2026-08-31", "2026-09-01"],
        "weather_code": [2, 80, 0, 3, 95, 61, 1],
        "temperature_2m_max": [85.2, 79.4, 88.0, 90.6, 84.0, 77.7, 82.0],
        "temperature_2m_min": [63.6, 60.2, 66.0, 70.0, 68.0, 61.0, 60.0],
        "precipitation_probability_max": [10, 60, 0, 20, 90, 70, 5],
        "sunrise": ["2026-08-26T06:10", "2026-08-27T06:11", "2026-08-28T06:12",
                    "2026-08-29T06:13", "2026-08-30T06:14", "2026-08-31T06:15",
                    "2026-09-01T06:16"],
        "sunset": ["2026-08-26T19:42", "2026-08-27T19:40", "2026-08-28T19:39",
                   "2026-08-29T19:37", "2026-08-30T19:36", "2026-08-31T19:34",
                   "2026-09-01T19:33"]},
}


class Fetcher:
    """Canned-response ``_fetch``: routes by URL prefix, records calls."""

    def __init__(self, routes):
        self.routes = routes        # prefix -> dict | bytes | Exception
        self.calls = []

    def __call__(self, url, timeout=6, headers=None):
        self.calls.append((url, timeout, headers))
        for prefix, answer in self.routes.items():
            if url.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                if callable(answer):
                    answer = answer(url, timeout, headers)
                if isinstance(answer, (bytes, bytearray)):
                    return bytes(answer)
                return json.dumps(answer).encode()
        raise AssertionError(f"unexpected fetch {url}")

    def urls(self):
        return [c[0] for c in self.calls]


@pytest.fixture(autouse=True)
def _clean_caches():
    location.clear_caches()
    weather.clear_cache()
    yield
    location.clear_caches()
    weather.clear_cache()


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "location.json"


# ------------------------------------------------------ resolve_home
def test_home_from_config_lat_lon_never_fetches(cache):
    cfg = FakeCfg(home_location={"city": "Chicago", "region": "Illinois",
                                 "lat": 41.85, "lon": -87.65, "tz": "America/Chicago"})
    fetch = Fetcher({})
    loc = location.resolve_home(cfg, cache, fetch=fetch)
    assert (loc.city, loc.region, loc.lat, loc.lon, loc.tz, loc.source) == \
        ("Chicago", "Illinois", 41.85, -87.65, "America/Chicago", "config")
    assert loc.label() == "Chicago, Illinois"
    assert fetch.calls == [] and not cache.exists()


def test_home_from_config_city_is_geocoded(cache):
    cfg = FakeCfg(home_location={"city": "Tokyo", "region": "", "lat": None, "lon": None})
    fetch = Fetcher({location.GEOCODE_URL.split("?")[0]: GEO_TOKYO})
    loc = location.resolve_home(cfg, cache, fetch=fetch)
    assert loc.source == "config" and loc.tz == "Asia/Tokyo" and loc.lat == 35.6895
    assert len(fetch.calls) == 1 and "name=tokyo" in fetch.urls()[0]


def test_ipapi_first_then_cached_24h(cache):
    fetch = Fetcher({location.IPAPI_URL: IPAPI_OK, location.IP_API_URL: IP_API_OK})
    loc = location.resolve_home(FakeCfg(), cache, fetch=fetch, now=1000.0)
    assert (loc.city, loc.region, loc.source) == ("Chicago", "Illinois", "ip")
    assert fetch.urls() == [location.IPAPI_URL]
    saved = json.loads(cache.read_text())
    assert saved["home"]["city"] == "Chicago" and saved["home"]["fetched_at"] == 1000.0
    # hit: 23 h later nothing is fetched
    fetch2 = Fetcher({})
    again = location.resolve_home(FakeCfg(), cache, fetch=fetch2, now=1000.0 + 23 * 3600)
    assert again.city == "Chicago" and fetch2.calls == []
    # miss: 25 h later a new lookup runs
    fetch3 = Fetcher({location.IPAPI_URL: IPAPI_LIMITED, location.IP_API_URL: IP_API_OK})
    later = location.resolve_home(FakeCfg(), cache, fetch=fetch3, now=1000.0 + 25 * 3600)
    assert later.city == "Tyler" and fetch3.urls() == [location.IPAPI_URL, location.IP_API_URL]


def test_ipapi_rate_limited_falls_back_to_ip_api(cache):
    fetch = Fetcher({location.IPAPI_URL: IPAPI_LIMITED, location.IP_API_URL: IP_API_OK})
    loc = location.resolve_home(FakeCfg(), cache, fetch=fetch)
    assert loc.label() == "Tyler, Texas" and loc.tz == "America/Chicago"
    assert fetch.urls() == [location.IPAPI_URL, location.IP_API_URL]


def test_ipapi_exception_falls_back_and_total_failure_is_none(cache):
    fetch = Fetcher({location.IPAPI_URL: OSError("dns"), location.IP_API_URL: IP_API_OK})
    assert location.resolve_home(FakeCfg(), cache, fetch=fetch).city == "Tyler"
    dead = Fetcher({location.IPAPI_URL: OSError("dns"), location.IP_API_URL: OSError("dns")})
    assert location.resolve_home(FakeCfg(), cache.with_name("x.json"), fetch=dead) is None
    # a stale cache still beats nothing
    stale = location.resolve_home(FakeCfg(), cache, fetch=dead, now=time.time() + 48 * 3600)
    assert stale is not None and stale.city == "Tyler"


def test_lookup_disabled_skips_network(cache):
    fetch = Fetcher({})
    assert location.resolve_home(FakeCfg(location_lookup=False), cache, fetch=fetch) is None
    assert fetch.calls == []


def test_lookup_never_exceeds_budget(cache, monkeypatch):
    """Both services time out: the second attempt gets only what is left of
    the 4 s budget, and a spent budget skips it entirely."""
    clock = [0.0]
    monkeypatch.setattr(location.time, "monotonic", lambda: clock[0])

    def slow(url, timeout, headers):
        clock[0] += timeout
        raise TimeoutError("slow")
    fetch = Fetcher({location.IPAPI_URL: slow, location.IP_API_URL: slow})
    assert location.lookup_ip(fetch) is None
    timeouts = [c[1] for c in fetch.calls]
    assert timeouts[0] == 3.0 and abs(timeouts[1] - 1.0) < 1e-6
    assert sum(timeouts) <= location.LOOKUP_BUDGET_S


# --------------------------------------------------------------- geocode
def test_geocode_memory_and_disk_cache(cache):
    fetch = Fetcher({location.GEOCODE_URL.split("?")[0]: GEO_TOKYO})
    a = location.geocode("Tokyo", fetch=fetch, cache_path=cache)
    b = location.geocode("  tokyo ", fetch=fetch, cache_path=cache)
    assert a.tz == "Asia/Tokyo" and b.lat == a.lat and len(fetch.calls) == 1
    assert json.loads(cache.read_text())["geocode"]["tokyo"]["city"] == "Tokyo"
    location.clear_caches()          # a new process reads the disk copy
    c = location.geocode("TOKYO", fetch=Fetcher({}), cache_path=cache)
    assert c.city == "Tokyo" and c.source == "geocode"


def test_geocode_miss_and_failure(cache):
    fetch = Fetcher({location.GEOCODE_URL.split("?")[0]: GEO_NONE})
    assert location.geocode("Xyzzy", fetch=fetch, cache_path=cache) is None
    assert location.geocode("Xyzzy", fetch=fetch, cache_path=cache) is None
    assert len(fetch.calls) == 1            # the miss is remembered in memory
    assert "geocode" not in json.loads(cache.read_text() or "{}") if cache.exists() else True
    assert location.geocode("", fetch=fetch, cache_path=cache) is None
    boom = Fetcher({location.GEOCODE_URL.split("?")[0]: OSError("down")})
    assert location.geocode("Paris", fetch=boom, cache_path=cache) is None


# ------------------------------------------------------------------ time
def test_time_words():
    assert location.clock_words(datetime(2026, 8, 26, 16, 5)) == "4:05 pm"
    assert location.clock_words(datetime(2026, 8, 26, 0, 7)) == "12:07 am"
    assert location.clock_words(datetime(2026, 8, 26, 12, 0)) == "12:00 pm"
    assert location.date_words(datetime(2026, 8, 26)) == "Wednesday the 26th of August"
    assert location.date_words(datetime(2026, 9, 1)) == "Tuesday the 1st of September"
    assert location.date_words(datetime(2026, 9, 22)) == "Tuesday the 22nd of September"
    assert location.date_words(datetime(2026, 9, 3)) == "Thursday the 3rd of September"
    assert location.date_words(datetime(2026, 9, 11)) == "Friday the 11th of September"
    assert location.format_time(datetime(2026, 8, 26, 16, 5), "Tokyo") == \
        "It's 4:05 pm in Tokyo, sir."
    assert location.format_time(datetime(2026, 8, 26, 9, 0)) == "It's 9:00 am, sir."


def test_now_in_converts_zones():
    tokyo = location.Location(city="Tokyo", tz="Asia/Tokyo")
    dt = location.now_in(tokyo, NOW)
    assert (dt.hour, dt.day, str(dt.tzinfo)) == (23, 26, "Asia/Tokyo")
    home = location.now_in(None, NOW)
    assert home.utcoffset() == NOW.astimezone().utcoffset()
    bad = location.Location(city="X", tz="Not/AZone")
    assert bad.tzinfo() is None


# --------------------------------------------------------------- tools
def _location_registry(cfg, fetch, monkeypatch, tmp_path):
    monkeypatch.setattr(location, "_fetch", fetch)
    monkeypatch.setattr(location, "cache_dir", lambda: tmp_path)
    reg = ToolRegistry()
    reg.register_many(location.make_tools(cfg, SimpleNamespace()))
    return reg


def test_tool_specs_follow_the_contract():
    specs = location.make_tools(FakeCfg(), SimpleNamespace()) + \
        weather.make_tools(FakeCfg(), SimpleNamespace())
    assert [s.name for s in specs] == ["get_time", "get_location", "get_weather"]
    for spec in specs:
        assert isinstance(spec, ToolSpec) and callable(spec.handler)
        assert len(spec.description.split()) <= 20
        assert spec.schema()["function"]["name"] == spec.name
    when = specs[2].parameters["properties"]["when"]
    assert when["enum"] == ["now", "today", "tomorrow", "week"]


def test_get_location_wording(monkeypatch, tmp_path):
    reg = _location_registry(FakeCfg(), Fetcher({location.IPAPI_URL: IPAPI_OK}),
                             monkeypatch, tmp_path)
    r = reg.call("get_location", {})
    assert r.ok and r.text == "You're in Chicago, Illinois, sir (from your network address)."
    cfg = FakeCfg(home_location={"city": "Austin", "region": "Texas", "lat": 30.27, "lon": -97.74})
    reg = _location_registry(cfg, Fetcher({}), monkeypatch, tmp_path / "b")
    assert reg.call("get_location", {"extra": 1}).text == \
        "You're in Austin, Texas, sir (from your settings)."


def test_get_location_excuses(monkeypatch, tmp_path):
    dead = Fetcher({location.IPAPI_URL: OSError("x"), location.IP_API_URL: OSError("x")})
    reg = _location_registry(FakeCfg(), dead, monkeypatch, tmp_path)
    r = reg.call("get_location", {})
    assert not r.ok and r.text == "location lookup unreachable" and r.speak is None
    reg = _location_registry(FakeCfg(location_lookup=False), Fetcher({}), monkeypatch,
                             tmp_path / "off")
    r = reg.call("get_location", {})
    assert not r.ok and r.text.startswith("I'll need your home_location set up, sir")
    assert r.speak == r.text


def test_get_time_home_and_city(monkeypatch, tmp_path):
    fetch = Fetcher({location.GEOCODE_URL.split("?")[0]: geo_route,
                     location.IPAPI_URL: IPAPI_OK})
    reg = _location_registry(FakeCfg(), fetch, monkeypatch, tmp_path)
    fixed = datetime(2026, 8, 26, 14, 5, tzinfo=CHI)

    class FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)
    monkeypatch.setattr(location, "datetime", FixedDT)
    r = reg.call("get_time", {})
    assert r.ok and r.text == "It's 2:05 pm, sir. Today is Wednesday the 26th of August."
    r = reg.call("get_time", {"location": "Tokyo"})
    assert r.text == "It's 4:05 am in Tokyo, sir. It's Thursday the 27th of August there."
    assert reg.call("get_time", {"location": "here"}).text.startswith("It's 2:05 pm, sir.")
    r = reg.call("get_time", {"location": "Nowhere-ville"})
    assert not r.ok and r.text == "I couldn't find a place called Nowhere-ville"


def test_get_time_uses_home_timezone_from_config(monkeypatch, tmp_path):
    cfg = FakeCfg(home_location={"city": "Tokyo", "lat": 35.68, "lon": 139.69, "tz": "Asia/Tokyo"})
    reg = _location_registry(cfg, Fetcher({}), monkeypatch, tmp_path)
    fixed = datetime(2026, 8, 26, 14, 5, tzinfo=CHI)

    class FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)
    monkeypatch.setattr(location, "datetime", FixedDT)
    assert reg.call("get_time", {}).text.startswith("It's 4:05 am, sir.")


def test_cfg_get_and_setup_line_accept_any_config_shape():
    ns = SimpleNamespace(home_location={"lat": 1.0}, icloud=SimpleNamespace(apple_id="a"))
    assert location.cfg_get(ns, "home_location.lat") == 1.0
    assert location.cfg_get(ns, "icloud.apple_id") == "a"
    assert location.cfg_get(ns, "missing.key", "d") == "d"
    assert location.cfg_get({"a": {"b": 2}}, "a.b") == 2
    assert location.cfg_get(None, "a", 3) == 3
    assert location.setup_line(ns, "google_ical") == location.DEFAULT_SETUP_LINES["google_ical"]
    assert location.setup_line(FakeCfg(), "icloud").startswith("I'll need your icloud set up")


# ---------------------------------------------------------------- weather
def test_forecast_url_matches_spec():
    url = weather.forecast_url(41.85, -87.65)
    assert url.startswith("https://api.open-meteo.com/v1/forecast?latitude=41.8500&longitude=-87.6500&")
    for part in ("current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,"
                 "precipitation,relative_humidity_2m",
                 "hourly=temperature_2m,precipitation_probability,weather_code",
                 "daily=weather_code,temperature_2m_max,temperature_2m_min,"
                 "precipitation_probability_max,sunrise,sunset",
                 "temperature_unit=fahrenheit", "wind_speed_unit=mph",
                 "precipitation_unit=inch", "timezone=auto", "forecast_days=7"):
        assert f"&{part}" in url


def test_wmo_words():
    assert weather.describe_code(0) == "clear"
    assert weather.describe_code(0, daytime=True) == "sunny"
    assert weather.describe_code(2) == "partly cloudy"
    assert weather.describe_code(61) == "light rain"
    assert weather.describe_code(95) == "thunderstorms"
    assert weather.describe_code(999) == "unsettled"
    assert weather.describe_code(None) == "unknown conditions"


def test_format_weather_exact_strings():
    assert weather.format_weather(FORECAST, "now", NOW) == \
        "72°F and partly cloudy, feels like 75, wind 8 mph; high 85, low 64, 10% chance of rain."
    assert weather.format_weather(FORECAST, "today", NOW) == \
        "Today: high 85, low 64, partly cloudy, 10% chance of rain, sunset 7:42 pm."
    assert weather.format_weather(FORECAST, "tomorrow", NOW) == \
        "Tomorrow: high 79, low 60, light showers, 60% chance of rain, sunset 7:40 pm."
    assert weather.format_weather(FORECAST, "week", NOW) == (
        "Today 85 and partly cloudy, tomorrow 79 with light showers, Friday 88 and sunny, "
        "Saturday 91 and overcast, Sunday 84 with thunderstorms, Monday 78 with light rain, "
        "Tuesday 82 and mostly sunny.")


def test_format_weather_uses_forecast_clock_and_night_words():
    night = json.loads(json.dumps(FORECAST))
    night["current"]["time"] = "2026-08-26T22:30"
    night["current"]["weather_code"] = 0
    assert weather.format_weather(night, "now").startswith("72°F and clear,")
    # an aware `now` in another zone is converted to the forecast's zone
    tokyo_now = datetime(2026, 8, 27, 1, 0, tzinfo=ZoneInfo("Asia/Tokyo"))   # = Aug 26 11:00 CDT
    assert weather.format_weather(FORECAST, "week", tokyo_now).startswith("Today 85")
    snow = json.loads(json.dumps(FORECAST))
    snow["daily"]["weather_code"][1] = 73
    assert "chance of snow" in weather.format_weather(snow, "tomorrow", NOW)


def test_coerce_when():
    for raw, want in (("this week", "week"), ("Week", "week"), ("tmrw", "tomorrow"),
                      ("Tomorrow?", "tomorrow"), ("right now", "now"), ("", "now"),
                      (None, "now"), ("today", "today"), ("tonight", "today"),
                      ("7 day forecast", "week"), ("currently", "now"), (5, "now")):
        assert weather.coerce_when(raw) == want, raw


def test_forecast_cache_10_minutes(monkeypatch):
    fetch = Fetcher({"https://api.open-meteo.com/v1/forecast": FORECAST})
    weather.fetch_forecast(41.851, -87.649, fetch=fetch, now=0.0)
    weather.fetch_forecast(41.854, -87.651, fetch=fetch, now=599.0)   # same rounded key
    assert len(fetch.calls) == 1
    weather.fetch_forecast(41.851, -87.649, fetch=fetch, now=601.0)
    assert len(fetch.calls) == 2
    weather.fetch_forecast(35.68, 139.69, fetch=fetch, now=601.0)     # another place
    assert len(fetch.calls) == 3


def _weather_registry(cfg, fetch, monkeypatch, tmp_path):
    monkeypatch.setattr(weather, "_fetch", fetch)
    monkeypatch.setattr(weather, "cache_dir", lambda: tmp_path)
    reg = ToolRegistry()
    reg.register_many(weather.make_tools(cfg, SimpleNamespace()))
    return reg


def test_get_weather_tool_home_and_city(monkeypatch, tmp_path):
    fetch = Fetcher({"https://api.open-meteo.com/v1/forecast": FORECAST,
                     location.GEOCODE_URL.split("?")[0]: geo_route,
                     location.IPAPI_URL: IPAPI_OK})
    cfg = FakeCfg(home_location={"city": "Chicago", "region": "Illinois",
                                 "lat": 41.85, "lon": -87.65})
    reg = _weather_registry(cfg, fetch, monkeypatch, tmp_path)
    r = reg.call("get_weather", {"when": "today"})
    assert r.ok and r.text == "Today: high 85, low 64, partly cloudy, 10% chance of rain, sunset 7:42 pm."
    r = reg.call("get_weather", {})                       # defaults to now
    assert r.text.startswith("72°F and partly cloudy")
    r = reg.call("get_weather", {"when": "this week", "location": "Tokyo", "junk": True})
    assert r.ok and r.text.startswith("In Tokyo, Japan: Today 85 and partly cloudy")
    assert [u for u in fetch.urls() if "forecast" in u][-1].startswith(
        "https://api.open-meteo.com/v1/forecast?latitude=35.6895&longitude=139.6917")
    r = reg.call("get_weather", {"when": "now", "location": "Xyzzy"})
    assert not r.ok and r.text == "I couldn't find a place called Xyzzy"


def test_get_weather_failures(monkeypatch, tmp_path):
    fetch = Fetcher({"https://api.open-meteo.com/v1/forecast": OSError("down"),
                     location.IPAPI_URL: IPAPI_OK})
    reg = _weather_registry(FakeCfg(), fetch, monkeypatch, tmp_path)
    r = reg.call("get_weather", {"when": "now"})
    assert r == ToolResult(text="weather service unreachable", ok=False)
    bad = Fetcher({"https://api.open-meteo.com/v1/forecast": {"error": True, "reason": "x"},
                   location.IPAPI_URL: IPAPI_OK})
    reg = _weather_registry(FakeCfg(), bad, monkeypatch, tmp_path / "b")
    assert reg.call("get_weather", {"when": "now"}).text == "weather service unreachable"
    reg = _weather_registry(FakeCfg(location_lookup=False), Fetcher({}), monkeypatch,
                            tmp_path / "c")
    r = reg.call("get_weather", {"when": "today"})
    assert not r.ok and r.speak == r.text and "home_location" in r.text
    dead = Fetcher({location.IPAPI_URL: OSError("x"), location.IP_API_URL: OSError("x")})
    reg = _weather_registry(FakeCfg(), dead, monkeypatch, tmp_path / "d")
    assert reg.call("get_weather", {"when": "today"}).text == "location lookup unreachable"


def test_registry_call_for_briefing_shape(monkeypatch, tmp_path):
    """W4's briefing calls registry.call('get_weather', {'when': 'today'})."""
    fetch = Fetcher({"https://api.open-meteo.com/v1/forecast": FORECAST,
                     location.IPAPI_URL: IPAPI_OK})
    reg = _weather_registry(FakeCfg(), fetch, monkeypatch, tmp_path)
    r = reg.call("get_weather", {"when": "today"})
    assert r.ok and r.text.startswith("Today: high 85")


# ------------------------------------------------------------------ live
@pytest.mark.skipif(not os.environ.get("JARVIS_LIVE"), reason="JARVIS_LIVE=1 only")
def test_live_geocode_and_forecast(tmp_path):
    tokyo = location.geocode("Tokyo", cache_path=tmp_path / "loc.json")
    assert tokyo and tokyo.tz == "Asia/Tokyo"
    data = weather.fetch_forecast(tokyo.lat, tokyo.lon)
    text = weather.format_weather(data, "now")
    assert "°F" in text and "feels like" in text and text.endswith("chance of rain.")
    assert weather.format_weather(data, "week").count(",") == 6


@pytest.mark.skipif(not os.environ.get("JARVIS_LIVE"), reason="JARVIS_LIVE=1 only")
def test_live_ip_lookup(tmp_path):
    loc = location.resolve_home(FakeCfg(), tmp_path / "loc.json")
    assert loc is not None and loc.lat is not None and loc.tz
