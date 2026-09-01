"""Room sensor (jarvis/roomsensor.py) and its composition into the presence
sentinel (jarvis/presence.py).

Nothing here opens a socket: ``RoomSensor(get=...)`` is the transport seam
the way ``probe(run=...)`` is the subprocess seam next door. The whole
point of the feature is the ASYMMETRY -- the room seeing someone beats a
sleeping phone, the room seeing nobody never beats a phone that answers --
plus the dark-safe promise: with no sensor configured, or with one that is
unplugged, the sentinel behaves exactly as it did before this existed.
"""
from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from jarvis import presence as presence_mod
from jarvis import roomsensor as rs
from jarvis.events import Presence
from jarvis.presence import PresenceSentinel, RoomOrPhone, make_probe, probe
from jarvis.roomsensor import RoomSensor, normalize_url, parse_state

ESPHOME_ON = '{"id":"binary_sensor-presence","value":true,"state":"ON"}'
ESPHOME_OFF = '{"id":"binary_sensor-presence","value":false,"state":"OFF"}'


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def tick(self, s):
        self.t += s


class Http:
    """Fake transport: records URLs, answers from a script."""

    def __init__(self, *answers):
        self.answers, self.urls, self.timeouts = list(answers), [], []

    def __call__(self, url, timeout):
        self.urls.append(url)
        self.timeouts.append(timeout)
        answer = self.answers[0] if len(self.answers) == 1 else self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def calls(self):
        return len(self.urls)


def _sensor(*answers, clock=None, **kw):
    clock = clock or Clock()
    s = RoomSensor("http://10.0.0.9/binary_sensor/presence", get=Http(*answers),
                   now=clock.now, **kw)
    s.clock = clock
    return s


# ----------------------------------------------------------------- URL
def test_a_url_needs_a_scheme_and_gains_the_default_entity_path():
    assert normalize_url("http://10.0.0.9") == "http://10.0.0.9/binary_sensor/presence"
    assert normalize_url("http://10.0.0.9/") == "http://10.0.0.9/binary_sensor/presence"
    assert normalize_url("http://10.0.0.9/binary_sensor/room") == \
        "http://10.0.0.9/binary_sensor/room"
    # No guessing: a bare host, a typo or a stray number is NOT a URL, and
    # inventing one would fail as a timeout every poll instead of a line.
    assert normalize_url("10.0.0.9") == "" and normalize_url("0.1") == ""
    assert normalize_url("") == "" and normalize_url(None) == ""


# --------------------------------------------------------------- parse
@pytest.mark.parametrize("body,want", [
    (ESPHOME_ON, True), (ESPHOME_OFF, False),
    ('{"state":"ON"}', True), ('{"state":"OFF"}', False),   # no "value" field
    ("true", True), ("false", False), ("ON", True), ("off", False),
    ("1", True), ("0", False), (b'{"value":true}', True),
    ('{"value":1}', True), ('{"value":0}', False),
])
def test_every_shape_an_esphome_entity_answers_in(body, want):
    assert parse_state(body) is want


@pytest.mark.parametrize("body", [
    "", "   ", "<html><body>404</body></html>", "{not json", "[1,2,3]",
    '{"id":"binary_sensor-presence"}', "maybe", None, '{"value":"maybe"}',
])
def test_garbage_is_no_opinion_never_absence(body):
    assert parse_state(body) is None


# -------------------------------------------------------------- sensor
def test_a_seeing_sensor_reads_true_with_a_tight_timeout():
    s = _sensor(ESPHOME_ON)
    assert s.read() is True
    assert s._get.urls == ["http://10.0.0.9/binary_sensor/presence"]
    assert s._get.timeouts == [rs.DEFAULT_TIMEOUT_S]      # a dead host cannot stall the loop
    assert s.status()["value"] is True


def test_an_empty_room_reads_false():
    assert _sensor(ESPHOME_OFF).read() is False


def test_an_unreachable_sensor_is_none_not_false():
    for boom in (socket.timeout("timed out"), OSError("no route"),
                 ValueError("nonsense")):
        assert _sensor(boom).read() is None


def test_a_sensor_serving_garbage_is_none_not_false():
    assert _sensor("<html>ESPHome</html>").read() is None


def test_an_unconfigured_sensor_never_calls_the_transport():
    http = Http(ESPHOME_ON)
    s = RoomSensor("not-a-url", get=http)
    assert not s.configured and s.read() is None and http.calls == 0


# ------------------------------------------------------------- breaker
def test_a_dead_sensor_stops_costing_the_poll_loop_anything():
    """The latency guarantee: after fail_after strikes no request is even
    attempted until the cooldown expires, so an unplugged ESP32 costs the
    tick zero milliseconds rather than a timeout apiece."""
    s = _sensor(OSError("down"), fail_after=3, cooldown_s=30.0)
    for _ in range(3):
        assert s.read() is None
    assert s._get.calls == 3 and s.paused
    for _ in range(10):
        assert s.read() is None
    assert s._get.calls == 3                       # not one more syscall


def test_the_cooldown_expires_retries_once_and_backs_off_further():
    s = _sensor(OSError("down"), fail_after=1, cooldown_s=30.0)
    assert s.read() is None and s._get.calls == 1 and s.paused
    s.clock.tick(31)
    assert not s.paused
    assert s.read() is None and s._get.calls == 2   # one retry, then paused again
    assert s.paused
    s.clock.tick(31)
    assert s.paused                                 # 30 -> 60: it backed off


def test_a_sensor_that_comes_back_resets_the_breaker():
    s = _sensor(OSError("down"), OSError("down"), ESPHOME_ON, ESPHOME_OFF,
                fail_after=3, cooldown_s=30.0)
    assert s.read() is None and s.read() is None and not s.paused
    assert s.read() is True
    assert s._fails == 0 and s._cooldown == 30.0 and s.last_ok is not None
    assert s.read() is False


# ------------------------------------------------------------- compose
class Phone:
    def __init__(self, answer=True):
        self.answer, self.calls = answer, []

    def __call__(self, ip="", mac=""):
        self.calls.append((ip, mac))
        return self.answer


def _composed(sensor_answer, phone_answer=True):
    sensor = SimpleNamespace(read=lambda: sensor_answer)
    phone = Phone(phone_answer)
    return RoomOrPhone(sensor, phone), phone


def test_the_room_seeing_someone_beats_a_sleeping_phone():
    """The disagreement decision, direction one: positive evidence wins,
    and the phone is not even asked (the ping is the expensive leg)."""
    fn, phone = _composed(True, phone_answer=False)
    assert fn("10.0.0.2", "") is True
    assert phone.calls == []


def test_an_empty_room_never_overrides_a_phone_that_answers():
    """Direction two: he may be in the kitchen. Absence stays the phone's
    verdict, on the same grace it always had."""
    fn, phone = _composed(False, phone_answer=True)
    assert fn("10.0.0.2", "") is True
    assert phone.calls == [("10.0.0.2", "")]


def test_an_empty_room_and_a_silent_phone_is_absence():
    fn, _ = _composed(False, phone_answer=False)
    assert fn("10.0.0.2", "") is False


def test_a_sensor_with_no_opinion_degrades_to_the_phone_exactly():
    for answer in (None,):
        fn, phone = _composed(answer, phone_answer=True)
        assert fn("10.0.0.2", "") is True and phone.calls == [("10.0.0.2", "")]
        fn, phone = _composed(answer, phone_answer=False)
        assert fn("10.0.0.2", "") is False and phone.calls == [("10.0.0.2", "")]


def test_a_sensor_that_raises_still_lets_the_phone_answer():
    boom = SimpleNamespace(read=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    phone = Phone(True)
    assert RoomOrPhone(boom, phone)("10.0.0.2", "") is True
    assert phone.calls == [("10.0.0.2", "")]


def test_sensor_only_install_answers_from_the_room_and_holds_when_blind():
    fn, phone = _composed(False)
    assert fn("", "") is False and phone.calls == []      # nobody in the room
    fn, _ = _composed(True)
    assert fn("", "") is True
    fn, _ = _composed(None)
    assert fn("", "") is None            # blind: NOT absence, hold the state


def test_no_sensor_means_the_original_probe_function_object():
    """Dark-safe at the identity level: nothing wraps the phone probe."""
    assert make_probe(None) is probe
    assert isinstance(make_probe(SimpleNamespace(read=lambda: None)), RoomOrPhone)


# ------------------------------------------------------ config plumbing
class DictCfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


PHONE = {"presence.phone_ip": "10.0.0.2"}
SENSOR = {"presence.room_sensor_enabled": True,
          "presence.room_sensor_url": "http://10.0.0.9"}


def test_the_sensor_is_off_until_both_keys_are_set():
    assert presence_mod._make_sensor(DictCfg(PHONE)) is None
    assert presence_mod._make_sensor(DictCfg({**PHONE, **SENSOR,
                                              "presence.room_sensor_enabled": False})) is None
    assert presence_mod._make_sensor(DictCfg({"presence.room_sensor_enabled": True})) is None
    bad = DictCfg({"presence.room_sensor_enabled": True,
                   "presence.room_sensor_url": "10.0.0.9"})
    assert presence_mod._make_sensor(bad) is None          # no scheme, no guessing


def test_a_configured_sensor_is_built_with_its_timeout():
    s = presence_mod._make_sensor(DictCfg({**SENSOR, "presence.room_sensor_timeout_s": 0.4}))
    assert isinstance(s, RoomSensor) and s.timeout_s == 0.4
    assert s.url == "http://10.0.0.9/binary_sensor/presence"


def test_an_unconfigured_sentinel_polls_the_untouched_phone_probe():
    """The dark path, end to end: same object, same call, no wrapper."""
    s = PresenceSentinel(DictCfg(PHONE), publish=lambda e: None)
    assert s.sensor is None and s._probe is probe and s.configured


def test_a_sensor_alone_is_enough_to_run_the_sentinel():
    s = PresenceSentinel(DictCfg(SENSOR), publish=lambda e: None)
    assert s.sensor is not None and s.configured and isinstance(s._probe, RoomOrPhone)


# ------------------------------------------------ sentinel integration
def _wired(cfg, *answers, phone_answer=False, clock=None):
    """A sentinel built the way app.py builds it (no probe_fn), with the
    two hardware seams swapped underneath: the sensor's HTTP and the
    phone's subprocess. Proves the WIRING, not a hand-made probe."""
    clock = clock or Clock()
    events = []
    s = PresenceSentinel(DictCfg(cfg), publish=events.append, now=clock.now)
    http = Http(*answers) if answers else None
    if s.sensor is not None:
        s.sensor._get = http
        s.sensor._now = clock.now
    phone = Phone(phone_answer)
    if isinstance(s._probe, RoomOrPhone):
        s._probe.phone = phone
    else:
        s._probe = phone
    s.clock, s.events, s.http, s.phone = clock, events, http, phone
    return s


def test_the_room_seeing_him_makes_him_home_on_the_first_tick():
    s = _wired({**PHONE, **SENSOR}, ESPHOME_ON, phone_answer=False)
    ev = s.tick()
    assert isinstance(ev, Presence) and ev.home and s.state == "home"
    assert s.phone.calls == []                    # the ping never ran


def test_an_empty_room_with_an_answering_phone_is_still_home():
    s = _wired({**PHONE, **SENSOR}, ESPHOME_OFF, phone_answer=True)
    assert s.tick().home and s.state == "home"
    s.clock.tick(3600)
    assert s.tick() is None and s.state == "home"  # no flap, no second event


def test_an_offline_sensor_is_indistinguishable_from_having_none():
    """The property that matters most: unplug it and today's behaviour is
    what is left -- the phone still decides, on the same grace."""
    s = _wired({**PHONE, **SENSOR}, OSError("down"), phone_answer=True)
    assert s.tick().home
    s.phone.answer = False
    for _ in range(11):
        s.clock.tick(60)
        assert s.tick() is None and s.is_home()    # the 12 min grace, unchanged
    s.clock.tick(60)
    ev = s.tick()
    assert ev is not None and ev.home is False and s.state == "away"
    # The breaker, not the clock, is what keeps this cheap: 13 ticks cost
    # a handful of requests (three strikes, then one retry per growing
    # cooldown), while the phone leg runs on every tick exactly as before.
    assert 3 <= s.http.calls <= 8 < 13
    assert len(s.phone.calls) == 13


def test_a_sensor_serving_garbage_falls_back_to_the_phone():
    s = _wired({**PHONE, **SENSOR}, "<html>oops</html>", phone_answer=True)
    assert s.tick().home and s.phone.calls == [("10.0.0.2", "")]


def test_the_arrival_transition_fires_exactly_once():
    """away -> home is ONE Presence(returned=True); the ticks either side
    publish nothing, which is what stops a second 'Welcome back, sir'."""
    answers = [ESPHOME_OFF] * 14 + [ESPHOME_ON] * 3
    s = _wired({**PHONE, **SENSOR}, *answers, phone_answer=False)
    s.tick()
    for _ in range(13):
        s.clock.tick(60)
        s.tick()
    assert s.state == "away"
    s.clock.tick(10)
    ev = s.tick()
    assert ev is not None and ev.home and ev.returned
    s.clock.tick(10)
    assert s.tick() is None                        # still home, nothing published
    s.clock.tick(10)
    assert s.tick() is None
    returns = [e for e in s.events if e.returned]
    assert len(returns) == 1


def test_a_sensor_only_install_going_blind_never_declares_him_away():
    """No phone leg and no sensor answer = no evidence. Holding 'home' is
    the only safe answer: a false away holds his proactive speech."""
    s = _wired(SENSOR, ESPHOME_ON, *([OSError("down")] * 5))
    assert s.tick().home
    for _ in range(60):
        s.clock.tick(60)
        assert s.tick() is None
    assert s.state == "home" and s.is_home()
    # An hour of ticks, and the dead sensor was asked a dozen-odd times:
    # the cooldown grows to 300 s, so the cost per tick tends to zero.
    assert 4 <= s.http.calls <= 20 < 61


def test_a_sensor_only_install_that_is_blind_from_boot_stays_unknown():
    s = _wired(SENSOR, OSError("down"))
    assert s.tick() is None and s.state == "unknown" and s.is_home()
    assert s._started_at is None                   # the grace clock has not started
