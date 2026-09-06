"""The instrument he runs himself. Numbers only.

No camera, no microphone, no frame, no recording. It reads his two ESP32s
over plain HTTP and prints counts, distances, gates and latencies. It must
never write to a sensor: no OTA, no config push, no reboot, no gate change.
"""
from __future__ import annotations

import pytest

from jarvis import roomprobe


def _samples(presence, still=None, moving=None, still_d=None):
    out = []
    n = len(presence)
    for i in range(n):
        out.append(roomprobe.Sample(
            at=float(i * 2),
            presence=presence[i],
            still_target=(still[i] if still else None),
            moving_target=(moving[i] if moving else None),
            still_cm=(still_d[i] if still_d else None),
            moving_cm=None, detect_cm=None, ms=150.0))
    return out


# ------------------------------------------------------- never clears
def test_a_room_that_never_clears_is_reported_as_never_clearing():
    r = roomprobe.summarise("office", _samples([True] * 30))
    assert r["never_clears"] is True
    assert r["on_fraction"] == 1.0
    assert r["samples"] == 30


def test_a_room_that_clears_even_once_is_not_reported_as_latched():
    r = roomprobe.summarise("office", _samples([True] * 29 + [False]))
    assert r["never_clears"] is False


def test_a_clear_room_is_not_latched():
    r = roomprobe.summarise("kitchen", _samples([False] * 30))
    assert r["never_clears"] is False
    assert r["on_fraction"] == 0.0


def test_the_window_length_is_reported_so_the_claim_is_bounded():
    """"occupied for 60 seconds" is a much weaker claim than "for an hour",
    and the report must not let the two be confused."""
    r = roomprobe.summarise("office", _samples([True] * 30))
    assert r["window_s"] == pytest.approx(58.0)


def test_no_samples_at_all_makes_no_claim():
    r = roomprobe.summarise("office", [])
    assert r["never_clears"] is None
    assert r["samples"] == 0


# --------------------------------------------------------- distances
def test_the_still_distance_is_reported_as_median_min_and_max():
    r = roomprobe.summarise("office", _samples([True] * 5,
                                               still_d=[300, 302, 303, 301, 305]))
    assert r["still_cm"]["median"] == 302
    assert r["still_cm"]["min"] == 300
    assert r["still_cm"]["max"] == 305
    assert r["still_cm"]["n"] == 5


def test_a_distance_nothing_reported_is_absent_rather_than_zero():
    r = roomprobe.summarise("office", _samples([True] * 3))
    assert r["still_cm"] is None


def test_moving_and_still_are_counted_separately():
    r = roomprobe.summarise("office", _samples(
        [True] * 4, still=[True, True, False, True],
        moving=[False, True, False, False]))
    assert r["still_on"] == 3
    assert r["moving_on"] == 1


# -------------------------------------------- against HIS configured window
def test_a_target_beyond_his_configured_range_is_flagged():
    """His office profile watches to 3.00 m. A body measured at 3.03 m is
    outside the window he configured and is held only by the gate above
    it -- which is worth telling him, because it is a one-line fix."""
    r = roomprobe.summarise("office", _samples([True] * 3, still_d=[303, 302, 304]),
                            profile={"nearest_m": 1.6, "range_m": 3.0})
    assert r["outside_window"] is True
    assert r["window_m"] == (1.6, 3.0)


def test_a_target_inside_the_window_is_not_flagged():
    r = roomprobe.summarise("office", _samples([True] * 3, still_d=[250, 240, 260]),
                            profile={"nearest_m": 1.6, "range_m": 3.0})
    assert r["outside_window"] is False


def test_a_target_nearer_than_the_near_gate_is_flagged_too():
    r = roomprobe.summarise("office", _samples([True] * 3, still_d=[100, 110, 90]),
                            profile={"nearest_m": 1.6, "range_m": 3.0})
    assert r["outside_window"] is True


def test_with_no_profile_no_window_claim_is_made():
    r = roomprobe.summarise("office", _samples([True] * 3, still_d=[303]*3))
    assert r["outside_window"] is None
    assert r["window_m"] is None


# ------------------------------------------------------------ health
def test_errors_and_latency_are_reported():
    s = _samples([True] * 3)
    s.append(roomprobe.Sample(at=6.0, error="TimeoutError"))
    r = roomprobe.summarise("office", s)
    assert r["errors"] == 1
    assert r["median_ms"] == 150


def test_a_window_of_nothing_but_errors_makes_no_occupancy_claim():
    s = [roomprobe.Sample(at=float(i), error="ConnectionError") for i in range(5)]
    r = roomprobe.summarise("office", s)
    assert r["never_clears"] is None
    assert r["errors"] == 5


# ------------------------------------------------------------ render
def test_the_report_is_text_and_names_the_room():
    r = roomprobe.summarise("office", _samples([True] * 30, still_d=[303]*30),
                            profile={"nearest_m": 1.6, "range_m": 3.0})
    text = roomprobe.render(r)
    assert "office" in text
    assert "303" in text
    assert "NEVER CLEARED" in text.upper()


def test_the_report_says_so_plainly_when_the_room_is_healthy():
    text = roomprobe.render(roomprobe.summarise("kitchen",
                                                _samples([False] * 30)))
    assert "kitchen" in text
    assert "NEVER CLEARED" not in text.upper()


def test_the_report_carries_no_image_or_audio_field():
    r = roomprobe.summarise("office", _samples([True] * 5))
    banned = ("frame", "image", "jpeg", "png", "audio", "wav", "mic",
              "camera", "snapshot")
    for key in r:
        assert not any(b in key.lower() for b in banned), key


# ------------------------------------------------------- read-only
def test_the_prober_only_ever_issues_reads():
    """No OTA, no config push, no reboot, no gate change. His sensors are
    live hardware in his flat and this instrument may look, never touch."""
    calls = []

    def opener(url, timeout=0):
        calls.append(url)
        class R:
            def read(self_inner):
                return b'{"value": true, "state": "ON"}'
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False
        return R()
    roomprobe.read_entity("http://10.0.0.1", "binary_sensor", "Presence",
                          opener=opener)
    assert len(calls) == 1
    assert calls[0].startswith("http://10.0.0.1/binary_sensor/")
    # Every ESPHome write path is a POST to .../<id>/turn_on|set|restart.
    for bad in ("turn_on", "turn_off", "/set", "restart", "ota"):
        assert bad not in calls[0]


def test_the_poll_interval_has_a_floor_of_two_seconds():
    assert roomprobe.MIN_INTERVAL_S >= 2.0
    assert roomprobe.clamp_interval(0.1) >= 2.0
    assert roomprobe.clamp_interval(5.0) == 5.0


def test_the_window_is_hard_capped():
    assert roomprobe.clamp_seconds(10 ** 9) <= roomprobe.MAX_SECONDS
    assert roomprobe.clamp_seconds(60) == 60


def test_the_entity_url_uses_the_name_not_the_object_id():
    """MEASURED, and it cost a silent failure once: ESPHome web_server v2
    serves an entity at its NAME. Every guessed object_id URL was a 404 and
    presence failed silently."""
    url = roomprobe.entity_url("http://10.0.0.1", "sensor", "Still distance")
    assert url == "http://10.0.0.1/sensor/Still%20distance"
    assert "still_distance" not in url
