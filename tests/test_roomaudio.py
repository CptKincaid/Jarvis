"""Multi-room speech routing and the per-room receipt (jarvis/roomaudio.py).

The policy tests are pure -- no clock but the injected one, no network at
all -- and the RoomSpeaker tests drive a fake ``post``. Nothing here may
touch a live LAN or a live PipeWire.

Design: docs/multiroom-audio.md.
"""
from __future__ import annotations

import pytest

from jarvis.roomaudio import (ALARM, ANNOUNCE, ANSWER, Receipt, Room,
                              RoomSpeaker, VoiceRouter, occupancy_from,
                              parse_receipt, rooms_from_config, say_url)


class Cfg:
    """The two ``get`` calls rooms_from_config makes, and nothing else."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


# ------------------------------------------------------------------ config
def _cfg(satellites, here="office", enabled=True):
    return Cfg(**{"rooms.here": here, "rooms.enabled": enabled,
                  "rooms.satellites": satellites})


def test_here_always_exists_even_with_no_satellites():
    """The Spark's own room has no satellite, and it is the room every
    other rule falls through to."""
    rooms, here = rooms_from_config(_cfg([], enabled=False))
    assert here == "office" and rooms["office"].url == ""


def test_the_satellite_list_is_shared_with_the_sensing_half():
    """One list, so a room cannot exist for the radar and not for the
    voice, and a room name cannot be spelled two ways."""
    rooms, here = rooms_from_config(_cfg([
        {"name": "kitchen", "url": "http://192.168.50.61",
         "say_url": "http://192.168.50.61:8765"},
        {"name": "bedroom", "url": "http://192.168.50.62", "private": True},
    ]))
    assert set(rooms) == {"office", "kitchen", "bedroom"}
    assert rooms["kitchen"].url == "http://192.168.50.61:8765/say"
    assert rooms["bedroom"].url == ""          # radar but no speaker yet
    assert rooms["bedroom"].private is True


def test_a_room_whose_say_url_is_rejected_survives_as_a_name():
    """It falls back to `here`, so he still hears the line; a silently
    missing room is one he never hears and never finds out about."""
    rooms, _ = rooms_from_config(_cfg([
        {"name": "kitchen", "url": "http://192.168.50.61",
         "say_url": "http://kitchen.local:8765"},          # a hostname
    ]))
    assert "kitchen" in rooms and rooms["kitchen"].url == ""


def test_disabled_rooms_config_speaks_here_only():
    rooms, here = rooms_from_config(_cfg(
        [{"name": "kitchen", "url": "http://192.168.50.61",
          "say_url": "http://192.168.50.61:8765"}], enabled=False))
    assert set(rooms) == {"office"} and here == "office"


def test_an_unreadable_config_still_leaves_him_a_room():
    class Boom:
        def get(self, key, default=None):
            raise RuntimeError("no")

    rooms, here = rooms_from_config(Boom())
    assert here in rooms


def test_say_url_takes_the_privacy_lanes_trust_rule_not_a_looser_one():
    """rooms.check_url: an http(s) PRIVATE IP LITERAL. This endpoint
    carries the same bearer token, so a laxer rule here would be the hole."""
    assert say_url("http://192.168.50.61") == "http://192.168.50.61/say"
    assert say_url("http://192.168.50.61:8765/play") == \
        "http://192.168.50.61:8765/play"
    assert say_url("http://kitchen.local") == ""        # a hostname
    assert say_url("http://8.8.8.8") == ""              # public
    assert say_url("192.168.50.61") == ""               # no scheme
    assert say_url("") == "" and say_url(None) == ""


# ------------------------------------------------------------------ policy
def _router(**kw):
    rooms = {n: Room(name=n, url=("" if n == "office" else "http://10.0.0.9/say"),
                     private=(n == "bedroom"))
             for n in ("office", "kitchen", "bedroom")}
    clock = {"t": 1000.0}
    r = VoiceRouter(rooms, here="office", now=lambda: clock["t"], **kw)
    return r, clock


def test_an_answer_goes_back_where_it_was_asked():
    r, _ = _router()
    d = r.route(ANSWER, source_room="kitchen", occupancy={"office": True})
    assert d.room == "kitchen"


def test_an_answer_is_never_rerouted_by_a_sensor():
    """He asked two seconds ago and he is still standing there. A radar
    that has not caught up must not move the answer away from his ear."""
    r, _ = _router()
    d = r.route(ANSWER, source_room="kitchen",
                occupancy={"kitchen": False, "office": True})
    assert d.room == "kitchen"


def test_an_answer_from_an_unknown_room_lands_here():
    r, _ = _router()
    assert r.route(ANSWER, source_room="garage").room == "office"
    assert r.route(ANSWER, source_room="").room == "office"


def test_a_timer_set_in_the_office_fires_where_he_is():
    """The awkward case, and the reason announcements are not tied to the
    room that created them: the timer belongs to the person."""
    r, _ = _router()
    d = r.route(ANNOUNCE, source_room="office",
                occupancy={"office": False, "kitchen": True})
    assert d.room == "kitchen"


def test_an_announcement_is_held_when_he_is_out_not_broadcast():
    r, _ = _router()
    d = r.route(ANNOUNCE, occupancy={"office": False}, home=False)
    assert d.held is True
    assert d.rooms == ()


def test_no_opinion_anywhere_lands_here_and_never_everywhere():
    """None is "no opinion", NEVER "empty" -- roomsensor.read's rule. A
    router that broadcast on ignorance would shout in three rooms every
    time a sensor was unplugged."""
    r, _ = _router()
    d = r.route(ANNOUNCE, occupancy={"office": None, "kitchen": None})
    assert d.rooms == ("office",)
    assert "no room has an opinion" in d.reason


def test_positively_empty_rooms_read_differently_from_silent_ones():
    r, _ = _router()
    d = r.route(ANNOUNCE, occupancy={"office": False, "kitchen": False})
    assert d.rooms == ("office",)
    assert "no room sees him" in d.reason


def test_two_occupied_rooms_pick_the_one_he_last_spoke_in():
    r, clock = _router()
    r.heard("office", at=900.0)
    r.heard("kitchen", at=980.0)
    d = r.route(ANNOUNCE, occupancy={"office": True, "kitchen": True})
    assert d.room == "kitchen"
    assert len(d.rooms) == 1          # never two: that is the comb filter


def test_a_stale_turn_is_not_evidence():
    """Yesterday's kitchen turn must not decide tonight's reminder."""
    r, clock = _router(recency_s=300.0)
    r.heard("kitchen", at=100.0)      # 900 s ago
    d = r.route(ANNOUNCE, occupancy={"office": True, "kitchen": True})
    assert d.room == "office"         # here, by default


def test_an_alarm_goes_everywhere_including_a_private_room():
    """An alarm you cannot hear from the bedroom is not an alarm."""
    r, _ = _router()
    d = r.route(ALARM, occupancy={"office": False})
    assert set(d.rooms) == {"office", "kitchen", "bedroom"}


def test_a_demoted_room_is_never_a_target():
    r, _ = _router()
    r.demote("kitchen")
    assert r.route(ANSWER, source_room="kitchen").room == "office"
    assert r.route(ANNOUNCE, occupancy={"kitchen": True}).room == "office"
    assert "kitchen" not in r.route(ALARM).rooms
    r.promote("kitchen")
    assert r.route(ANSWER, source_room="kitchen").room == "kitchen"


def test_here_falls_through_when_the_configured_room_is_gone():
    r = VoiceRouter({"kitchen": Room(name="kitchen", url="http://10.0.0.9/say")},
                    here="office")
    assert r.here == "kitchen"
    assert VoiceRouter({}, here="office").here == ""


def test_a_room_with_no_speaker_is_not_a_target():
    """The bedroom has a radar and no speaker yet: it can be seen in, not
    spoken in."""
    r = VoiceRouter({"office": Room(name="office"),
                     "bedroom": Room(name="bedroom")}, here="office")
    assert r.route(ANNOUNCE, occupancy={"bedroom": True}).room == "office"
    assert r.route(ANSWER, source_room="bedroom").room == "office"


def test_observe_lets_route_run_without_touching_the_network():
    """route() runs inside a spoken turn; the snapshot is taken on the
    mesh's own thread, soundbar.status_line's rule."""
    r, _ = _router()
    r.observe({"kitchen": True})
    assert r.route(ANNOUNCE).room == "kitchen"


def test_occupancy_from_survives_a_mesh_that_changed_shape():
    class Sat:
        def __init__(self, name, value):
            self.spec = type("S", (), {"name": name})()
            self._v = value

        def read(self):
            if isinstance(self._v, Exception):
                raise self._v
            return self._v

    mesh = type("M", (), {"satellites": [Sat("kitchen", True),
                                         Sat("bedroom", RuntimeError("x"))]})()
    assert occupancy_from(mesh) == {"kitchen": True, "bedroom": None}
    assert occupancy_from(object()) == {}


def test_a_router_with_no_rooms_holds_rather_than_naming_one():
    d = VoiceRouter({}).route(ANNOUNCE, occupancy={})
    assert d.held is True and d.rooms == ()


# ----------------------------------------------------------------- receipt
def test_a_receipt_that_played_nothing_did_not_land():
    assert Receipt(ok=True, played_ms=0.0, expected_ms=3000).landed is False


def test_a_receipt_with_a_dead_level_did_not_land():
    """The HDMI monitor of 2026-08-30: the player ran, the sink was fine,
    and nothing came out of a speaker. pactl could not see this."""
    assert Receipt(ok=True, played_ms=3000, expected_ms=3000, peak=0.0
                   ).landed is False
    assert Receipt(ok=True, played_ms=3000, expected_ms=3000, peak=0.3
                   ).landed is True


def test_a_truncated_playback_did_not_land():
    assert Receipt(ok=True, played_ms=400, expected_ms=3000).landed is False
    assert Receipt(ok=True, played_ms=2900, expected_ms=3000).landed is True


@pytest.mark.parametrize("body", ["", "<html>oops</html>", "[1,2]", "null",
                                  b"not json"])
def test_an_unreadable_body_is_a_failed_receipt_not_an_assumed_success(body):
    assert parse_receipt(body, expected_ms=1000).landed is False


def test_parse_receipt_reads_the_normal_shape():
    r = parse_receipt('{"ok": true, "played_ms": 3120, "peak": 0.31}', 3000)
    assert r.ok and r.played_ms == 3120 and r.peak == 0.31 and r.landed


# ------------------------------------------------------------ RoomSpeaker
class FakePost:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, payload, timeout, headers=None):
        self.calls.append((url, len(payload), headers or {}))
        reply = self.replies.pop(0) if self.replies else '{"ok": true, "played_ms": 3000}'
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_a_good_reply_lands_and_carries_the_token():
    post = FakePost('{"ok": true, "played_ms": 3000, "peak": 0.4}')
    sp = RoomSpeaker("kitchen", "http://10.0.0.5", token="abc", post=post)
    r = sp.say(b"RIFF....", expected_ms=3000)
    assert r.landed is True and sp.healthy is True
    assert post.calls[0][0] == "http://10.0.0.5/say"
    assert post.calls[0][2]["Authorization"] == "Bearer abc"


def test_the_breaker_stops_touching_the_network_after_two_failures():
    """An unplugged kitchen must cost the reply path ZERO syscalls, not a
    timeout apiece -- roomsensor.RoomSensor's rule and its assertion."""
    post = FakePost(OSError("no route"), OSError("no route"), OSError("boom"))
    sp = RoomSpeaker("kitchen", "http://10.0.0.5", post=post,
                     now=lambda: 0.0, fail_after=2)
    assert sp.say(b"x", 100).landed is False
    assert sp.say(b"x", 100).landed is False
    assert sp.paused is True
    assert sp.say(b"x", 100).error == "breaker"
    assert sp.posts == 2                 # the third never reached the wire


def test_a_room_that_answers_but_plays_nothing_counts_against_the_breaker():
    """"The endpoint is up" is not the promise; "the sound came out" is."""
    silent = '{"ok": true, "played_ms": 0}'
    post = FakePost(silent, silent)
    sp = RoomSpeaker("kitchen", "http://10.0.0.5", post=post,
                     now=lambda: 0.0, fail_after=2)
    sp.say(b"x", 3000)
    sp.say(b"x", 3000)
    assert sp.healthy is False


def test_the_breaker_recovers_on_a_good_receipt():
    clock = {"t": 0.0}
    post = FakePost(OSError("down"), '{"ok": true, "played_ms": 3000}')
    sp = RoomSpeaker("kitchen", "http://10.0.0.5", post=post,
                     now=lambda: clock["t"], fail_after=1, cooldown_s=30.0)
    assert sp.say(b"x", 3000).landed is False
    assert sp.paused is True
    clock["t"] = 31.0
    assert sp.say(b"x", 3000).landed is True
    assert sp.healthy is True and sp._cooldown == 30.0


def test_an_unconfigured_room_never_touches_the_network():
    post = FakePost()
    sp = RoomSpeaker("kitchen", "not-a-url", post=post)
    assert sp.configured is False
    assert sp.say(b"x", 100).error == "unconfigured"
    assert post.calls == []


def test_status_is_a_dict_the_console_can_render():
    sp = RoomSpeaker("kitchen", "http://10.0.0.5",
                     post=FakePost('{"ok": true, "played_ms": 3000}'))
    sp.say(b"x", 3000)
    st = sp.status()
    assert st["room"] == "kitchen" and st["landed"] is True and st["posts"] == 1
