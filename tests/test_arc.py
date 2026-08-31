"""The arc (jarvis/arc.py): a pure state engine with a frozen clock.

Every boundary, the hysteresis (dwell, the solar tolerance band, the
forward-only rule) and the three overrides. The engine has to be provably a
function of its inputs before it is allowed a thread, so most of this file
never constructs Arc at all.
"""
from datetime import date, datetime, timedelta, timezone

import pytest

from jarvis import arc as arc_mod
from jarvis.arc import (BAND_S, MIN_DWELL_S, NEUTRAL_PHASE, PHASES, Arc, band_at,
                        boundaries, forced_phase, phase_at, sun_times)

TZ = timezone(timedelta(hours=-5))          # a fixed offset: no tzdata needed
LAT, LON = 30.628, -96.334                  # College Station, TX
DAY = date(2026, 8, 30)


def at(h, m=0, day=DAY):
    return datetime(day.year, day.month, day.day, h, m, tzinfo=TZ)


@pytest.fixture
def sun():
    """A fixed sunrise/sunset for DAY: 07:00 and 19:51 local."""
    return at(7, 0), at(19, 51)


# ------------------------------------------------------------------ solar
def test_sun_times_matches_published_tables():
    """NOAA formula, stdlib only. Checked against published sunrise/sunset
    for New York at both solstices -- within two minutes is the accuracy
    the formula claims and all the arc needs."""
    tz = timezone(timedelta(hours=-4))      # EDT
    rise, sset = sun_times(40.7128, -74.0060, date(2026, 6, 21), tz)
    assert rise.strftime("%H:%M") in ("05:24", "05:25", "05:26")
    assert sset.strftime("%H:%M") in ("20:30", "20:31", "20:32")
    tz = timezone(timedelta(hours=-5))      # EST
    rise, sset = sun_times(40.7128, -74.0060, date(2026, 12, 21), tz)
    assert rise.strftime("%H:%M") in ("07:15", "07:16", "07:17", "07:18")
    assert sset.strftime("%H:%M") in ("16:30", "16:31", "16:32")


def test_sun_times_needs_no_network_and_no_weather_import(monkeypatch):
    """The whole point of correction 1: an arc keyed off weather.py would
    degrade to fixed hours on every cold boot. Nothing in sun_times may
    reach the network."""
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("sun_times went to the network"))
    assert sun_times(LAT, LON, DAY, TZ)[0] is not None


def test_sun_times_is_none_in_polar_night():
    """Above the arctic circle in December the sun never rises; callers
    fall back to fixed hours rather than crashing on acos()."""
    assert sun_times(78.2, 15.6, date(2026, 12, 21), TZ) == (None, None)


def test_sun_times_rejects_nonsense_coordinates():
    assert sun_times("north", None, DAY, TZ) == (None, None)
    assert sun_times(200.0, 0.0, DAY, TZ) == (None, None)


# ------------------------------------------------------------- the bands
def test_the_seven_boundaries_in_order(sun):
    rise, sset = sun
    cases = [
        (at(2, 0), "night"),        # before 04:00 is still last night
        (at(4, 30), "pre-dawn"),
        (at(6, 45), "waking"),      # sunrise 07:00 less the 25 min lead
        (at(9, 0), "working"),      # sunrise + 2 h
        (at(13, 0), "afternoon"),
        (at(19, 0), "dusk"),        # sunset 19:51 less the 60 min lead
        (at(20, 40), "evening"),    # sunset + 30 min
        (at(22, 30), "night"),
    ]
    for when, expected in cases:
        assert band_at(when, sunrise=rise, sunset=sset) == expected, when


def test_every_band_name_is_a_declared_phase(sun):
    rise, sset = sun
    seen = {band_at(at(0, 0) + timedelta(minutes=10 * i), sunrise=rise, sunset=sset)
            for i in range(6 * 24)}
    assert seen == set(PHASES)


def test_boundaries_are_strictly_increasing_even_at_absurd_latitudes():
    """A garbled cache or a high-latitude midsummer can put sunset before
    sunrise; the day must still be orderable."""
    rise, sset = at(23, 30), at(1, 0)
    times = [when for _n, when, _s in boundaries(DAY, sunrise=rise, sunset=sset, tz=TZ)]
    assert times == sorted(times) and len(set(times)) == len(times)


def test_the_working_boundary_is_clamped_against_a_freak_sunrise():
    """A June sunrise must not put "working" at 07:30, nor December's at
    10:30: the clamp is what keeps the phase meaning what it says."""
    early = band_at(at(7, 45), sunrise=at(4, 30), sunset=at(21, 0))
    late = band_at(at(9, 45), sunrise=at(8, 30), sunset=at(17, 0))
    assert early == "working" and late == "working"


def test_no_coordinates_falls_back_to_fixed_hours():
    """home_location defaults to empty; the arc still names the hour."""
    assert band_at(at(13, 0), sunrise=None, sunset=None) == "afternoon"
    assert band_at(at(5, 0), sunrise=None, sunset=None) == "pre-dawn"


# --------------------------------------------------------------- overrides
def test_quiet_reason_forces_night(sun):
    rise, sset = sun
    assert phase_at(at(13, 0), sunrise=rise, sunset=sset,
                    quiet_reason="do not disturb until 3:00 pm") == "night"


def test_a_focus_block_pins_working_even_at_midnight(sun):
    """A man deliberately working at midnight is working; the room must not
    act like it is bedtime at him."""
    rise, sset = sun
    assert phase_at(at(0, 30), sunrise=rise, sunset=sset,
                    focus_phase="block") == "working"


def test_a_focus_block_outranks_quiet_hours(sun):
    rise, sset = sun
    assert phase_at(at(23, 30), sunrise=rise, sunset=sset,
                    quiet_reason="quiet hours until 7:00 am",
                    focus_phase="block") == "working"


def test_away_collapses_to_the_neutral_phase(sun):
    rise, sset = sun
    assert phase_at(at(13, 0), sunrise=rise, sunset=sset,
                    presence_state="away") == NEUTRAL_PHASE


def test_unknown_presence_is_not_away(sun):
    """PresenceSentinel.state is "unknown" until the probe answers, and a
    misconfigured probe must not hush the house all day."""
    rise, sset = sun
    assert phase_at(at(13, 0), sunrise=rise, sunset=sset,
                    presence_state="unknown") == "afternoon"


def test_forced_phase_reports_why():
    assert forced_phase(focus_phase="block") == "working"
    assert forced_phase(quiet_reason="a meeting") == NEUTRAL_PHASE
    assert forced_phase(presence_state="away") == NEUTRAL_PHASE
    assert forced_phase() == ""
    assert forced_phase(focus_phase="break") == ""


# -------------------------------------------------------------- hysteresis
def test_a_sunset_that_shifts_twenty_minutes_does_not_flap(sun):
    """The named risk: a cached forecast refreshes with a different sunset
    between two ticks. The phase must not walk backwards."""
    rise, sset = sun
    now = at(19, 5)                                     # just inside dusk
    first = phase_at(now, sunrise=rise, sunset=sset)
    assert first == "dusk"
    since = at(19, 4).timestamp()
    # Same instant, sunset now 20 minutes later: dusk would not have opened.
    later = phase_at(now, sunrise=rise, sunset=sset + timedelta(minutes=20),
                     previous="dusk", since=since)
    assert later == "dusk"


def test_the_phase_only_advances_within_a_day(sun):
    rise, sset = sun
    since = at(19, 30).timestamp()
    assert phase_at(at(19, 40), sunrise=rise, sunset=sset,
                    previous="evening", since=since) == "evening"


def test_a_new_day_may_start_over_at_pre_dawn(sun):
    """night (index 6) -> pre-dawn (index 0) is backwards by index but
    forwards in time; the forward-only rule is per local date."""
    rise, sset = sun
    since = at(22, 10, day=DAY).timestamp()
    tomorrow = date(2026, 8, 31)
    assert phase_at(at(4, 30, day=tomorrow), sunrise=rise, sunset=sset,
                    previous="night", since=since) == "pre-dawn"


def test_a_stale_sunrise_from_another_date_still_names_the_day(sun):
    """phase_at is public and pure, so it can be handed yesterday's cached
    sunrise. The solar CLOCK time is what matters; re-anchoring it on the
    day under test keeps the boundaries in their proper places."""
    rise, sset = sun                              # both dated 2026-08-30
    tomorrow = date(2026, 8, 31)
    assert band_at(at(13, 0, day=tomorrow), sunrise=rise, sunset=sset) == "afternoon"
    assert band_at(at(5, 0, day=tomorrow), sunrise=rise, sunset=sset) == "pre-dawn"


def test_minimum_dwell_holds_a_phase_that_has_only_just_started(sun):
    rise, sset = sun
    since = at(11, 55).timestamp()               # working adopted 5 min ago
    assert phase_at(at(12, 0), sunrise=rise, sunset=sset,
                    previous="working", since=since) == "working"
    served = at(12, 0).timestamp() - MIN_DWELL_S - 1
    assert phase_at(at(12, 0), sunrise=rise, sunset=sset,
                    previous="working", since=served) == "afternoon"


def test_the_solar_tolerance_band_delays_a_solar_boundary_only(sun):
    """A solar boundary needs BAND_S past it; a fixed clock boundary such
    as noon does not move and so needs no tolerance."""
    rise, sset = sun
    old = at(19, 0).timestamp() - MIN_DWELL_S - 1
    just_in = at(18, 55)                          # dusk opened at 18:51
    assert phase_at(just_in, sunrise=rise, sunset=sset,
                    previous="afternoon", since=old) == "afternoon"
    past = at(18, 51) + timedelta(seconds=BAND_S + 60)
    assert phase_at(past, sunrise=rise, sunset=sset,
                    previous="afternoon", since=old) == "dusk"
    # noon is fixed: no band, only the dwell
    assert phase_at(at(12, 1), sunrise=rise, sunset=sset,
                    previous="working",
                    since=at(12, 1).timestamp() - MIN_DWELL_S - 1) == "afternoon"


def test_leaving_an_override_returns_to_the_clock_at_once(sun):
    """A hushed room that becomes free again must not lag 20 minutes behind
    the reason it was hushed, so previous_forced skips the dwell."""
    rise, sset = sun
    since = at(12, 58).timestamp()
    assert phase_at(at(13, 0), sunrise=rise, sunset=sset, previous="night",
                    since=since, previous_forced=True) == "afternoon"
    # without the flag the same call is held by the dwell
    assert phase_at(at(13, 0), sunrise=rise, sunset=sset, previous="night",
                    since=since) == "night"


def test_an_unknown_previous_phase_is_ignored(sun):
    rise, sset = sun
    assert phase_at(at(13, 0), sunrise=rise, sunset=sset,
                    previous="brunch", since=at(12, 0).timestamp()) == "afternoon"


# ------------------------------------------------------------- the engine
class FakeCfg:
    """A config that records any write. The arc must never make one."""

    def __init__(self, data=None):
        self._data = data or {}
        self.writes = []

    def get(self, key, default=None):
        node = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def set(self, key, value):
        self.writes.append((key, value))


class FakeQuiet:
    def __init__(self, reason=""):
        self._reason = reason

    def reason(self, now=None):
        return self._reason


def _arc(tmp_path, cfg=None, clock=None, **kw):
    published = []
    a = Arc(cfg or FakeCfg({"home_location": {"lat": LAT, "lon": LON}}),
            publish=published.append,
            now=clock or (lambda: at(13, 0).timestamp()),
            state_path=tmp_path / "arc.json", **kw)
    return a, published


def test_tick_publishes_once_per_transition_never_per_tick(tmp_path):
    a, published = _arc(tmp_path)
    assert a.tick() is not None
    assert a.phase == "afternoon"
    assert a.tick() is None and a.tick() is None
    assert len(published) == 1


def test_the_arc_never_writes_to_the_config(tmp_path):
    """Invariant 3 of the module docstring: briefing.verbosity and the quiet
    keys are user-set by voice, and an arc that wrote one would silently
    overwrite a spoken preference."""
    cfg = FakeCfg({"home_location": {"lat": LAT, "lon": LON}})
    a, _pub = _arc(tmp_path, cfg=cfg)
    a.tick()
    a.tick()
    assert cfg.writes == []


def test_the_arc_asks_quiet_and_never_re_derives_it(tmp_path):
    """quiet.py is the owner. The arc reads reason() and obeys it."""
    a, published = _arc(tmp_path, quiet=FakeQuiet("a meeting until 2:00 pm"))
    ev = a.tick()
    assert ev.phase == "night" and ev.forced is True


def test_state_survives_a_restart(tmp_path):
    a, _pub = _arc(tmp_path)
    a.tick()
    b, published = _arc(tmp_path)
    assert b.phase == "afternoon" and b.since == a.since
    assert b.tick() is None and published == []


def test_a_disabled_arc_does_nothing(tmp_path):
    cfg = FakeCfg({"arc": {"enabled": False},
                   "home_location": {"lat": LAT, "lon": LON}})
    a, published = _arc(tmp_path, cfg=cfg)
    assert a.tick() is None and published == []
    a.start()
    assert a._thread is None


def test_a_failing_quiet_policy_does_not_stop_the_arc(tmp_path):
    class Boom:
        def reason(self, now=None):
            raise RuntimeError("policy down")

    a, published = _arc(tmp_path, quiet=Boom())
    assert a.tick().phase == "afternoon"


def test_the_engine_holds_the_solar_answer_for_the_day(tmp_path, monkeypatch):
    calls = []

    def counted(lat, lon, day, tz=None):
        calls.append(day)
        return at(7, 0, day), at(19, 51, day)

    a, _pub = _arc(tmp_path, sun_times_fn=counted)
    a.tick()
    a.tick()
    assert len(calls) == 1


def test_a_cached_forecast_is_an_override_and_never_a_fetch(tmp_path, monkeypatch):
    """cached_sun_times reads what weather.py already holds; it must never
    call fetch_forecast."""
    from jarvis.tools import weather as weather_mod
    monkeypatch.setattr(weather_mod, "fetch_forecast",
                        lambda *a, **k: pytest.fail("the arc fetched a forecast"))
    cfg = FakeCfg({"home_location": {"lat": LAT, "lon": LON}})
    assert arc_mod.cached_sun_times(cfg, DAY, TZ) == (None, None)


def test_stop_is_safe_before_start(tmp_path):
    a, _pub = _arc(tmp_path)
    a.stop()
    assert a._thread is None
