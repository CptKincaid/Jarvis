"""Rooms (jarvis/rooms.py): the satellite lease, and the refusal to call an
unreachable device "off".

Nothing here opens a socket -- ``Satellite(get=..., post=...)`` is the
transport seam, the same shape ``RoomSensor(get=...)`` next door uses. The
three properties under test are the three the feature is for:

* offline mode stops the POLL, not just the trust (the ``reads`` counters);
* a device we cannot reach is UNKNOWN and never OFF, in the state model,
  in the chip, in the caption and in the spoken line;
* a lease that does not land is not extended, so the device's own countdown
  keeps running toward powering itself down.
"""
from __future__ import annotations

import logging
import urllib.parse

import pytest

from jarvis import rooms
from jarvis.rooms import (ABSENT, ALLOW, DENY, DISAGREE, LIVE, MIC, OFF,
                          UNKNOWN, PrivacyView, RoomMesh, RoomSpec, Satellite,
                          SensorView, caption, check_url, chip, derive_state,
                          is_private_ip, make_transport, mesh_probe,
                          spec_from_dict, specs_from_config, spoken_status)
from jarvis.sensing import CAMERA, RADAR

ON = '{"id":"binary_sensor-presence","value":true,"state":"ON"}'
OFF_BODY = '{"id":"binary_sensor-presence","value":false,"state":"OFF"}'


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def tick(self, s):
        self.t += s


class Wire:
    """Fake transport. ``bodies`` maps a path suffix to a body (or an
    Exception to raise); ``posts`` records every button press."""

    def __init__(self, bodies=None, post_ok=True):
        self.bodies = dict(bodies or {})
        self.gets: list = []
        self.posts: list = []
        self.post_ok = post_ok

    def get(self, url, timeout):
        self.gets.append(url)
        for suffix, body in self.bodies.items():
            if url.endswith(suffix):
                if isinstance(body, Exception):
                    raise body
                return body
        raise OSError("404 %s" % url)

    def post(self, url, timeout):
        self.posts.append(url)
        if not self.post_ok:
            raise OSError("unreachable")
        return ""


class Policy:
    """Stands in for SensingPolicy: the two methods rooms.py uses."""

    def __init__(self, camera=True, radar=True, raises=False):
        self.camera, self.radar, self.raises = camera, radar, raises
        self.attached: dict = {}

    def allowed(self, kind):
        if self.raises:
            raise RuntimeError("policy is broken")
        return self.camera if kind == CAMERA else \
            self.radar if kind == RADAR else False

    def attach(self, name, stop, present=None, resume=None):
        self.attached[name] = (stop, present, resume)


def make_sat(policy=None, wire=None, clock=None, sensors=(RADAR,), **kw):
    wire = wire or Wire({"/binary_sensor/Presence": ON,
                         "/binary_sensor/Radar%20powered": ON})
    clock = clock or Clock()
    spec = RoomSpec(name="kitchen", url="http://192.168.50.61",
                    sensors=sensors, **kw)
    sat = Satellite(spec, policy=policy, get=wire.get, post=wire.post,
                    now=clock.now)
    return sat, wire, clock


# ------------------------------------------------------------------ trust
@pytest.mark.parametrize("host,ok", [
    ("192.168.50.61", True), ("10.0.0.4", True), ("172.16.9.9", True),
    # CGNAT: is_private says False for this and some home routers hand it
    # out, which is why the rule is `not is_global` (jarvis/webapp.py:192)
    ("100.64.1.1", True), ("127.0.0.1", True), ("169.254.3.3", True),
    ("8.8.8.8", False), ("1.1.1.1", False), ("93.184.216.34", False),
    ("0.0.0.0", False), ("224.0.0.1", False),
    ("jarvis-kitchen.local", False), ("", False), ("kitchen", False)])
def test_is_private_ip(host, ok):
    assert is_private_ip(host) is ok


def test_check_url_requires_a_private_literal():
    assert check_url("http://192.168.50.61") == "http://192.168.50.61"
    assert check_url("http://192.168.50.61:8080/") == "http://192.168.50.61:8080"
    # a hostname is one poisoned mDNS answer from being somebody else's box
    assert check_url("http://jarvis-kitchen.local") == ""
    assert check_url("http://8.8.8.8") == ""
    assert check_url("192.168.50.61") == ""            # no scheme
    assert check_url("") == ""


def test_redirects_are_refused():
    """A compromised satellite must not be able to point the poll loop at
    the phone client (jarvis/webapp.py trusts private peers)."""
    handler = rooms._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {},
                                    "http://127.0.0.1:8765/api/say") is None


def test_transport_sends_basic_auth_and_no_credentials_without_one():
    captured = {}

    class FakeOpener:
        def open(self, req, timeout=None):
            captured["headers"] = dict(req.header_items())
            captured["method"] = req.get_method()

            class R:
                def read(self, n):
                    return b"ON"

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False
            return R()

    real = rooms.urllib.request.build_opener
    rooms.urllib.request.build_opener = lambda *a: FakeOpener()
    try:
        get, post = make_transport("jarvis", "hunter2")
        assert get("http://192.168.50.61/x", 1.0) == "ON"
        keys = {k.lower(): v for k, v in captured["headers"].items()}
        assert keys["Authorization".lower()].startswith("Basic ")
        post("http://192.168.50.61/y", 1.0)
        assert captured["method"] == "POST"
        get2, _ = make_transport()
        get2("http://192.168.50.61/x", 1.0)
        keys = {k.lower() for k in captured["headers"]}
        assert "authorization" not in keys
    finally:
        rooms.urllib.request.build_opener = real


# ------------------------------------------------------------ state model
def test_derive_state_truth_table():
    assert derive_state(ALLOW, True, 1.0) == LIVE
    assert derive_state(ALLOW, False, 1.0) == OFF
    assert derive_state(DENY, False, 1.0) == OFF
    # asked to stop, still reporting on -- the one red state
    assert derive_state(DENY, True, 1.0) == DISAGREE
    # no answer at all
    assert derive_state(ALLOW, None, None) == UNKNOWN
    assert derive_state(DENY, None, None) == UNKNOWN


def test_a_stale_confirmation_is_unknown_not_the_last_value():
    """A satellite unplugged while OFF must not render as a confirmed OFF
    for ever. That is the most comfortable lie available here."""
    assert derive_state(DENY, False, age_s=10.0, stale_after_s=90.0) == OFF
    assert derive_state(DENY, False, age_s=91.0, stale_after_s=90.0) == UNKNOWN
    assert derive_state(ALLOW, True, age_s=91.0, stale_after_s=90.0) == UNKNOWN


def test_unknown_differs_from_off_in_word_and_in_shape():
    """Colour alone does not survive a dimmed monitor: sensing_badge.py's
    rule, one level down."""
    off_word, off_tone, off_filled = chip(SensorView("kitchen", RADAR, OFF))
    unk_word, unk_tone, unk_filled = chip(SensorView("kitchen", RADAR, UNKNOWN))
    assert off_word != unk_word
    assert off_tone != unk_tone
    assert off_filled is True and unk_filled is False
    live_word, live_tone, _ = chip(SensorView("kitchen", RADAR, LIVE))
    assert live_tone not in (off_tone, unk_tone)
    assert chip(SensorView("kitchen", RADAR, DISAGREE))[1] == "error"


def test_caption_marks_the_lease_inference_as_an_inference():
    view = SensorView("kitchen", RADAR, UNKNOWN, intent=DENY, age_s=240.0,
                      lease_left_s=-30.0)
    text = caption(view)
    assert "unreachable" in text
    assert "should have powered down" in text
    # it must not claim a confirmation it does not have
    assert "confirmed" not in text
    live = caption(SensorView("kitchen", RADAR, LIVE, age_s=2.0))
    assert "confirmed" in live


def test_absent_is_not_off():
    view = SensorView("kitchen", CAMERA, ABSENT)
    assert "no camera in the kitchen" == caption(view)
    assert chip(view)[0] == ""


# -------------------------------------------------------------- the view
def _rows(*pairs):
    return PrivacyView(rows=tuple(SensorView(r, k, s) for r, k, s in pairs))


def test_worst_prefers_the_uncomfortable_answer():
    assert _rows(("office", RADAR, OFF)).worst() == OFF
    assert _rows(("office", RADAR, OFF), ("kitchen", RADAR, LIVE)).worst() == LIVE
    assert _rows(("office", RADAR, LIVE),
                 ("kitchen", RADAR, UNKNOWN)).worst() == UNKNOWN
    assert _rows(("office", RADAR, UNKNOWN),
                 ("kitchen", RADAR, DISAGREE)).worst() == DISAGREE


def test_trustworthy_only_when_every_room_answered():
    assert _rows(("office", RADAR, OFF), ("kitchen", RADAR, OFF)).trustworthy()
    assert not _rows(("office", RADAR, OFF),
                     ("kitchen", RADAR, UNKNOWN)).trustworthy()
    assert _rows(("office", RADAR, OFF),
                 ("kitchen", CAMERA, ABSENT)).trustworthy()
    assert not PrivacyView().trustworthy()


def test_spoken_status_names_the_room_it_cannot_reach_first():
    view = PrivacyView(rows=(
        SensorView("office", CAMERA, OFF, intent=DENY, age_s=2.0),
        SensorView("office", RADAR, OFF, intent=DENY, age_s=2.0),
        SensorView("kitchen", RADAR, UNKNOWN, intent=DENY, age_s=240.0,
                   lease_left_s=-30.0)))
    line = spoken_status(view)
    assert line.index("can't reach") < line.index("off.")
    assert "kitchen" in line
    assert "can't confirm" in line


def test_spoken_status_flags_a_device_that_ignored_the_order():
    view = PrivacyView(rows=(
        SensorView("bedroom", RADAR, DISAGREE, intent=DENY, age_s=3.0),))
    line = spoken_status(view)
    assert "still reporting on" in line
    assert line.lower().startswith("the bedroom radar")


def test_spoken_status_with_nothing_configured():
    assert "no rooms configured" in spoken_status(PrivacyView())


# ------------------------------------------------------------- satellite
def test_offline_mode_stops_the_poll_not_just_the_trust():
    policy = Policy(radar=False)
    sat, wire, _ = make_sat(policy=policy)
    assert sat.read() is None
    assert sat.presence.reads == 0 and sat.powered[RADAR].reads == 0
    assert wire.gets == []


def test_a_broken_policy_is_not_permission():
    sat, wire, _ = make_sat(policy=Policy(raises=True))
    assert sat.read() is None
    assert wire.gets == []


def test_presence_is_none_until_the_radar_is_confirmed_powered():
    wire = Wire({"/binary_sensor/Presence": ON,
                 "/binary_sensor/Radar%20powered": OFF_BODY})
    sat, wire, _ = make_sat(policy=Policy(), wire=wire)
    assert sat.read() is None
    # the presence entity was never asked: an unconfirmed sensor's reading
    # describes a moment nobody can date
    assert sat.presence.reads == 0


def test_presence_is_none_when_the_device_does_not_answer():
    wire = Wire({"/binary_sensor/Presence": ON})   # powered path 404s
    sat, _, _ = make_sat(policy=Policy(), wire=wire)
    assert sat.read() is None


def test_presence_reads_through_when_the_radar_is_confirmed():
    sat, wire, _ = make_sat(policy=Policy())
    assert sat.read() is True
    assert sat.presence.reads == 1 and sat.powered[RADAR].reads == 1


def test_each_room_attaches_under_its_own_name():
    """SensingPolicy.attach replaces by NAME. Two satellites attaching as
    "radar" would leave one of them unstoppable, and the spoken line could
    not name the room that refused."""
    policy = Policy()
    make_sat(policy=policy, sensors=(RADAR, CAMERA))
    assert set(policy.attached) == {"kitchen radar", "kitchen camera"}
    spec = RoomSpec(name="bedroom", url="http://192.168.50.62")
    wire = Wire()
    Satellite(spec, policy=policy, get=wire.get, post=wire.post)
    assert "bedroom radar" in policy.attached
    assert "kitchen radar" in policy.attached


def test_a_microphone_is_shown_but_not_governed():
    """Shown, not attached, and -- since F05 -- its own state rather than
    UNKNOWN, because UNKNOWN is the word for a device that did not answer."""
    policy = Policy()
    sat, _, _ = make_sat(policy=policy, sensors=(RADAR, MIC))
    assert MIC not in " ".join(policy.attached)
    view = sat.view(MIC)
    assert view.state == rooms.UNGOVERNED
    assert view.state != UNKNOWN
    assert "not governed by offline mode" in view.detail


def test_renew_extends_the_lease_and_a_failed_renew_does_not():
    clock = Clock()
    wire = Wire({"/binary_sensor/Radar%20powered": ON})
    sat, _, _ = make_sat(policy=Policy(), wire=wire, clock=clock)
    assert sat.renew() is True
    assert sat.leases[RADAR].until == pytest.approx(
        clock.t + sat.spec.lease_ttl_s)
    before = sat.leases[RADAR].until
    wire.post_ok = False
    clock.tick(30)
    assert sat.renew() is False
    assert sat.leases[RADAR].until == before   # the countdown keeps running


def test_revoke_sets_the_lease_to_now():
    clock = Clock()
    sat, wire, _ = make_sat(policy=Policy(), clock=clock)
    sat.renew()
    assert sat.revoke() is True
    assert sat.leases[RADAR].until == clock.t
    assert sat.leases[RADAR].intent == DENY
    assert wire.posts[-1].endswith("/button/Radar%20lease%20revoke/press")


def test_stop_reports_failure_so_the_spoken_line_can():
    wire = Wire(post_ok=False)
    sat, _, _ = make_sat(policy=Policy(), wire=wire)
    assert sat.stop() is False


def test_view_says_disagree_when_a_revoked_device_still_reports_on():
    clock = Clock()
    sat, _, _ = make_sat(policy=Policy(), clock=clock)
    sat.revoke()
    sat.confirm()                    # the device answers ON anyway
    assert sat.view(RADAR).state == DISAGREE


def test_view_goes_unknown_once_the_confirmation_is_stale():
    clock = Clock()
    sat, _, _ = make_sat(policy=Policy(), clock=clock)
    sat.renew()
    sat.confirm()
    assert sat.view(RADAR).state == LIVE
    clock.tick(600)
    view = sat.view(RADAR)
    assert view.state == UNKNOWN
    assert view.lease_left_s < 0     # the lease is long gone


# ------------------------------------------------------------------ mesh
def make_mesh(policy, *names, wire=None, clock=None):
    clock = clock or Clock()
    wire = wire or Wire({"/binary_sensor/Presence": ON,
                         "/binary_sensor/Radar%20powered": ON})
    sats = []
    for i, name in enumerate(names):
        spec = RoomSpec(name=name, url="http://192.168.50.%d" % (61 + i))
        sats.append(Satellite(spec, policy=policy, get=wire.get,
                              post=wire.post, now=clock.now))
    return RoomMesh(policy=policy, satellites=sats, now=clock.now), wire, clock


def test_tick_renews_while_allowed_and_revokes_at_the_curfew_edge():
    policy = Policy()
    mesh, wire, _ = make_mesh(policy, "kitchen", "bedroom")
    mesh.tick()
    assert sum(1 for u in wire.posts if u.endswith("renew/press")) == 2
    policy.radar = False
    wire.posts.clear()
    mesh.tick()
    assert sum(1 for u in wire.posts if u.endswith("revoke/press")) == 2


def test_a_revoke_that_never_lands_is_retried_then_left_to_the_lease():
    policy = Policy(radar=False)
    wire = Wire({"/binary_sensor/Radar%20powered": ON}, post_ok=False)
    mesh, wire, _ = make_mesh(policy, "kitchen", wire=wire)
    for _ in range(6):
        mesh.tick()
    revokes = [u for u in wire.posts if u.endswith("revoke/press")]
    assert len(revokes) == rooms.REVOKE_TRIES
    # and it keeps CONFIRMING, so a device that ignored the order is seen
    assert mesh.satellites[0].powered[RADAR].reads >= 1


def test_the_mesh_keeps_confirming_a_room_it_has_revoked():
    policy = Policy(radar=False)
    mesh, wire, _ = make_mesh(policy, "kitchen")
    mesh.tick()
    assert any("/binary_sensor/Radar%20powered" in u for u in wire.gets)
    assert mesh.view().rows[0].state == DISAGREE


def test_one_bad_satellite_does_not_cost_the_others_their_lease():
    class Boom(Satellite):
        def renew(self):
            raise RuntimeError("nope")

    policy = Policy()
    clock, wire = Clock(), Wire({"/binary_sensor/Radar%20powered": ON})
    bad = Boom(RoomSpec("kitchen", "http://192.168.50.61"), policy=policy,
               get=wire.get, post=wire.post, now=clock.now)
    good = Satellite(RoomSpec("bedroom", "http://192.168.50.62"), policy=policy,
                     get=wire.get, post=wire.post, now=clock.now)
    mesh = RoomMesh(policy=policy, satellites=[bad, good], now=clock.now)
    mesh.tick()
    assert good.leases[RADAR].until is not None


def test_mesh_read_fuses_the_way_room_or_phone_does():
    policy = Policy()
    seen = {"kitchen": True, "bedroom": False}

    class Fake(Satellite):
        def read(self):
            return seen[self.spec.name]

    wire = Wire()
    sats = [Fake(RoomSpec(n, "http://192.168.50.6%d" % i), policy=policy,
                 get=wire.get, post=wire.post)
            for i, n in enumerate(("kitchen", "bedroom"))]
    mesh = RoomMesh(policy=policy, satellites=sats)
    assert mesh.read() is True                 # anyone, anywhere
    seen["kitchen"] = False
    assert mesh.read() is False                # every room answered "nobody"
    seen["kitchen"] = None
    assert mesh.read() is None                 # one silent room -> no opinion


def test_mesh_probe_never_lets_an_empty_room_beat_a_phone():
    policy = Policy()

    class Fake(Satellite):
        def read(self):
            return False

    wire = Wire()
    sat = Fake(RoomSpec("kitchen", "http://192.168.50.61"), policy=policy,
               get=wire.get, post=wire.post)
    mesh = RoomMesh(policy=policy, satellites=[sat])
    probe = mesh_probe(mesh, phone=lambda ip, mac: True)
    assert probe("192.168.50.9", "") is True
    probe_no_phone = mesh_probe(mesh, phone=lambda ip, mac: False)
    assert probe_no_phone("192.168.50.9", "") is False
    # no phone leg at all: the room's "nobody" is the honest answer
    assert probe_no_phone("", "") is False


def test_mesh_probe_holds_when_nothing_knows_anything():
    policy = Policy()

    class Fake(Satellite):
        def read(self):
            return None

    wire = Wire()
    sat = Fake(RoomSpec("kitchen", "http://192.168.50.61"), policy=policy,
               get=wire.get, post=wire.post)
    mesh = RoomMesh(policy=policy, satellites=[sat])
    assert mesh_probe(mesh, phone=lambda i, m: False)("", "") is None


# ---------------------------------------------------------------- config
class Cfg:
    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        value = self.data.get(key, default)
        return default if value is None else value


def test_spec_from_dict_drops_what_it_cannot_trust():
    assert spec_from_dict({"name": "kitchen",
                           "url": "http://192.168.50.61"}) is not None
    assert spec_from_dict({"name": "kitchen",
                           "url": "http://kitchen.local"}) is None
    assert spec_from_dict({"name": "", "url": "http://192.168.50.61"}) is None
    assert spec_from_dict({"name": "k", "url": "http://192.168.50.61",
                           "sensors": ["thermal"]}) is None
    assert spec_from_dict("kitchen") is None


def test_specs_from_config_is_off_by_default_and_dedups():
    cfg = Cfg({"presence.rooms": [
        {"name": "kitchen", "url": "http://192.168.50.61",
         "sensors": ["radar"]}]})
    assert specs_from_config(cfg) == ()        # the master switch is off
    cfg = Cfg({"presence.room_sensor_enabled": True, "presence.rooms": [
        {"name": "kitchen", "url": "http://192.168.50.61",
         "sensors": ["radar"]},
        {"name": "Kitchen", "url": "http://192.168.50.62",
         "sensors": ["radar"]},                # the same room, spelled twice
        {"name": "bedroom", "url": "http://8.8.8.8", "sensors": ["radar"]},
        {"name": "study", "url": "http://192.168.50.63",
         "sensors": ["radar", "mic"]}]})
    specs = specs_from_config(cfg)
    assert [s.name for s in specs] == ["kitchen", "study"]
    assert specs[1].sensors == (RADAR, MIC)


def test_specs_from_config_survives_a_broken_config():
    class Broken:
        def get(self, key, default=None):
            raise RuntimeError("no")
    assert specs_from_config(Broken()) == ()
    assert specs_from_config(object()) == ()


def test_mesh_from_config_with_nothing_configured_is_idle():
    mesh = RoomMesh.from_config(Cfg({}))
    assert not mesh.configured
    assert mesh.view().rows == ()
    assert mesh.read() is None
    mesh.start()                       # must not raise, must not spawn
    assert mesh._thread is None


def test_back_online_inside_the_curfew_leaves_the_lens_off():
    """"Offline mode" then "back online" at 22:00: the mesh tick checked
    allowed() before renewing, but the SensingPolicy RESUME hook went
    straight to renew(), pressed camera_lease_renew, set the intent to
    ALLOW and had the spoken line name "kitchen camera" as resumed --
    inside the curfew (F01, reproduced 2026-09-03). The gate now lives in
    renew() itself, and a refused resume is neither claimed nor a failure."""
    policy = Policy(camera=False, radar=True)   # i.e. curfew_active()
    wire = Wire({"/binary_sensor/Presence": ON,
                 "/binary_sensor/Radar%20powered": ON,
                 "/binary_sensor/Camera%20powered": OFF_BODY})
    sat, wire, _ = make_sat(policy=policy, wire=wire, sensors=(RADAR, CAMERA))
    wire.posts.clear()
    assert sat.resume() is True                     # the policy's resume hook
    assert any(u.endswith("/button/Radar%20lease%20renew/press") for u in wire.posts)
    assert not any("Camera%20lease%20renew" in u for u in wire.posts)
    assert sat.leases[CAMERA].intent != "allow"
    from jarvis.sensing import Declined
    assert isinstance(sat.renew(CAMERA), Declined)


def test_the_curfew_closes_the_lens_and_leaves_the_radar_up():
    """The ruling jarvis/sensing.py made for the local box, carried one
    room out: at 21:00 the camera stops being renewed and the radar does
    not. A single room-wide lease could not express that -- which is why
    there is one lease per KIND."""
    policy = Policy(camera=False, radar=True)   # i.e. curfew_active()
    wire = Wire({"/binary_sensor/Presence": ON,
                 "/binary_sensor/Radar%20powered": ON,
                 "/binary_sensor/Camera%20powered": OFF_BODY})
    sat, wire, _ = make_sat(policy=policy, wire=wire, sensors=(RADAR, CAMERA))
    mesh = RoomMesh(policy=policy, satellites=[sat])
    mesh.tick()
    assert any(u.endswith("/button/Radar%20lease%20renew/press") for u in wire.posts)
    assert any(u.endswith("/button/Camera%20lease%20revoke/press")
               for u in wire.posts)
    assert not any("Camera%20lease%20renew" in u for u in wire.posts)
    states = {r.kind: r.state for r in mesh.view().rows}
    assert states[RADAR] == LIVE and states[CAMERA] == OFF
    # ...and presence still works during the curfew, which is the point
    assert sat.read() is not None


def test_offline_mode_takes_both_kinds_down_in_every_room():
    policy = Policy(camera=False, radar=False)
    wire = Wire({"/binary_sensor/Radar%20powered": OFF_BODY,
                 "/binary_sensor/Camera%20powered": OFF_BODY})
    sat, wire, _ = make_sat(policy=policy, wire=wire, sensors=(RADAR, CAMERA))
    mesh = RoomMesh(policy=policy, satellites=[sat])
    mesh.tick()
    revoked = {u.rsplit("/button/", 1)[1] for u in wire.posts}
    assert revoked == {"Radar%20lease%20revoke/press",
                       "Camera%20lease%20revoke/press"}
    assert all(r.state == OFF for r in mesh.view().rows)
    assert mesh.view().trustworthy()


def test_stop_is_true_only_when_every_sensor_in_the_room_took_it():
    """SensingPolicy.Outcome.failed is what the spoken line reads from, so
    a partial stop must not report success."""
    calls = []

    class Half(Satellite):
        def revoke(self, kind=RADAR):
            calls.append(kind)
            return kind != CAMERA

    wire = Wire()
    sat = Half(RoomSpec("kitchen", "http://192.168.50.61",
                        sensors=(RADAR, CAMERA)), get=wire.get, post=wire.post)
    assert sat.stop() is False
    assert set(calls) == {RADAR, CAMERA}       # both were tried, not short-cut


# ------------------------------------------------------------- one list
def test_room_name_is_the_one_normaliser_the_three_lanes_share():
    """'Kitchen', ' kitchen ' and 'Kitchen!' are one room, not three. The
    fabric slugged, the satellite lane stripped and the audio lane only
    stripped, so the same word in the same file named two different rooms
    (F02, reproduced 2026-09-03). One function, and the other two lanes
    import it rather than carry a copy."""
    from jarvis import roomfabric
    assert rooms.room_name is roomfabric.room_name is roomfabric._slug
    for word in ("Kitchen", "  Kitchen  ", "Kitchen!"):
        assert rooms.room_name(word) == "kitchen"
    assert rooms.room_name("Front Room") == "front room"
    assert rooms.room_name(None) == ""


def test_the_three_lanes_read_one_list_and_agree_on_the_names():
    """presence.rooms is THE list: the one DEFAULTS declares and the one the
    fabric, the sensors page, zone_log and arrival already read. Before
    this the satellite and audio lanes read rooms.satellites, declared
    nowhere, so a config with the documented list got a fabric with no
    leases and an audio lane that knew one room (F02)."""
    from jarvis import roomaudio, roomfabric
    cfg = Cfg({"presence.room_sensor_enabled": True,
               "presence.rooms": [
                   {"name": "Office", "url": "http://192.168.50.51",
                    "primary": True},
                   {"name": "Kitchen ", "url": "http://192.168.50.61",
                    "sensors": ["radar"],
                    "say_url": "http://192.168.50.61:8765"}]})
    fabric_names = [s.name for s in roomfabric.room_specs(cfg)]
    lease_names = [s.name for s in specs_from_config(cfg)]
    audio, here = roomaudio.rooms_from_config(cfg)
    assert fabric_names == ["office", "kitchen"]
    assert lease_names == ["kitchen"]          # the one that declares sensors
    assert set(audio) == {"office", "kitchen"} and here == "office"
    assert audio["kitchen"].url == "http://192.168.50.61:8765/say"


def test_a_plain_room_sensor_is_never_leased():
    """The radars on the wall today run jarvis-room-sensor.yaml: no lease
    buttons. An entry that does not list sensors is the fabric's room and
    this lane must not press anything on it -- every press would be a 404
    and the console would call a working radar UNKNOWN."""
    cfg = Cfg({"presence.room_sensor_enabled": True,
               "presence.rooms": [
                   {"name": "office", "url": "http://192.168.50.51"},
                   {"name": "kitchen", "url": "http://192.168.50.61",
                    "sensors": []}]})
    assert specs_from_config(cfg) == ()
    assert not RoomMesh.from_config(cfg).configured


def test_the_master_switch_turns_the_lease_lane_off_with_the_fabric():
    """presence.room_sensor_enabled is the one switch over every room lane:
    with it off nobody reads a radar, so a leased one would be a powered
    sensor nobody is listening to."""
    cfg = Cfg({"presence.room_sensor_enabled": False,
               "presence.rooms": [{"name": "kitchen",
                                   "url": "http://192.168.50.61",
                                   "sensors": ["radar"]}]})
    assert specs_from_config(cfg) == ()


def test_the_rooms_block_holds_the_timers_and_no_second_list():
    """DEFAULTS declares every key a lane reads, and exactly ONE room list.
    rooms.satellites, rooms.here and rooms.enabled are gone: the list is
    presence.rooms, `here` is its primary entry, and `sensors` is the
    opt-in."""
    from jarvis.assistant_config import DEFAULTS
    block = DEFAULTS["rooms"]
    assert set(block) == {"renew_s", "stale_after_s", "timeout_s"}
    assert block["timeout_s"] == rooms.DEFAULT_TIMEOUT_S == 3.0
    assert DEFAULTS["presence"]["rooms"] == []
    mesh = RoomMesh.from_config(Cfg({"rooms.renew_s": 40.0}))
    assert mesh.renew_s == 40.0


def test_a_satellite_password_is_masked_the_way_a_gmail_one_is():
    """The YAML tells him to paste the web_server password into the room's
    entry and says it is masked. F03 (merged 2026-09-03) put that entry
    under rooms.satellites; it moved to presence.rooms with the list and
    must still be masked in repr(cfg) and scrubbed out of a log line."""
    from jarvis.assistant_config import SECRET_LIST_FIELDS, AssistantConfig
    assert ("presence.rooms", "password") in SECRET_LIST_FIELDS
    cfg = AssistantConfig({"presence": {"rooms": [
        {"name": "kitchen", "url": "http://192.168.50.61",
         "sensors": ["radar"], "username": "jarvis",
         "password": "sat-secret"}]}})
    assert "sat-secret" in cfg.secret_values()
    assert "sat-secret" not in repr(cfg)
    assert "sat-secret" not in cfg.scrub("GET / with sat-secret failed")


# ------------------------------------------------------------- the urls
def test_every_url_is_the_entity_name_form_the_firmware_serves():
    """ESPHome web_server v2 serves an entity at its NAME, percent-encoded
    (measured on the live office radar 2026-09-03: /binary_sensor/Presence
    -> 200, /binary_sensor/Presence -> 404). Every path this module built
    was the object_id form, so no press would ever have landed and a
    satellite that answered would have read UNKNOWN. The rule is
    roomsensor.entity_path's, spelled once."""
    from jarvis import roomsensor
    spec = RoomSpec(name="kitchen", url="http://192.168.50.61",
                    sensors=(RADAR, CAMERA))
    assert spec.presence_path() == roomsensor.DEFAULT_ENTITY_PATH \
        == "/binary_sensor/Presence"
    assert spec.powered_path(RADAR) == "/binary_sensor/Radar%20powered"
    assert spec.renew_path(RADAR) == "/button/Radar%20lease%20renew/press"
    assert spec.revoke_path(CAMERA) == "/button/Camera%20lease%20revoke/press"
    sat, wire, _ = make_sat(sensors=(RADAR, CAMERA))
    sat.renew(RADAR)
    sat.revoke(CAMERA)
    sat.confirm(RADAR)
    sat.read()
    assert wire.gets and wire.posts
    for url in wire.gets + wire.posts:
        path = url[len(spec.url):]
        # encoded exactly once, and a NAME (capitalised), never an object_id
        assert path == urllib.parse.quote(urllib.parse.unquote(path),
                                          safe="/"), url
        assert path.split("/")[2][0].isupper(), url


def test_the_entity_names_are_the_ones_the_satellite_yaml_declares():
    """The code and the firmware are two files that must agree on four
    strings. A scan of scripts/esphome/jarvis-satellite.yaml for its
    `name:` values holds them to each other, so renaming an entity on one
    side fails here rather than as a 404 on the wall."""
    import pathlib
    import re
    yaml = (pathlib.Path(rooms.__file__).resolve().parents[1]
            / "scripts" / "esphome" / "jarvis-satellite.yaml").read_text()
    names = {m.strip() for m in re.findall(
        r'^\s*name:\s*"?([^"\n#]+?)"?\s*(?:#.*)?$', yaml, re.M)}
    for want in (rooms.DEFAULT_PRESENCE_ENTITY,
                 rooms.entity_name(rooms.DEFAULT_POWERED_ENTITY, RADAR),
                 rooms.entity_name(rooms.DEFAULT_RENEW_ENTITY, RADAR),
                 rooms.entity_name(rooms.DEFAULT_REVOKE_ENTITY, RADAR)):
        assert want in names, (want, sorted(names))


def test_entity_name_fills_either_spelling_and_never_raises():
    assert rooms.entity_name("{Kind} powered", RADAR) == "Radar powered"
    assert rooms.entity_name("{kind}_powered", CAMERA) == "camera_powered"
    assert rooms.entity_name("{what} powered", RADAR) == "{what} powered"
    assert rooms.entity_name("", RADAR) == ""


def test_the_press_timeout_is_the_measured_poll_timeout():
    """rooms.py carried its own 1.5 s under the '~5 ms LAN' claim that
    roomsensor's measurement retired (max round trip seen 1186 ms). One
    number, imported, so a press waits as long as a poll does."""
    from jarvis import roomsensor
    assert rooms.DEFAULT_TIMEOUT_S == roomsensor.DEFAULT_TIMEOUT_S == 3.0
    sat, _, _ = make_sat()
    assert sat.timeout_s == 3.0
    assert sat.presence.timeout_s == 3.0


# ---------------------------------------------------------- press breaker
def test_a_dead_satellite_stops_pressing_and_says_so_once(caplog):
    """An unplugged kitchen paid a POST timeout per kind per tick and wrote
    a WARNING every tick -- 3,456 lines a day into the log worth reading
    first (F04). The GET side already had RoomSensor's breaker; the POST
    side had none."""
    wire = Wire(post_ok=False)
    clock = Clock()
    sat, wire, clock = make_sat(policy=Policy(), wire=wire, clock=clock)
    mesh = RoomMesh(policy=Policy(), satellites=[sat], now=clock.now)
    with caplog.at_level(logging.DEBUG, logger="jarvis.rooms"):
        for _ in range(20):
            mesh.tick()
            clock.tick(25.0)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) <= 2, [r.message for r in warnings]
    assert len(wire.posts) < 20, len(wire.posts)
    assert sat.presses < 20


def test_the_press_breaker_recovers_and_says_so(caplog):
    wire = Wire(post_ok=False)
    clock = Clock()
    sat, wire, clock = make_sat(policy=Policy(), wire=wire, clock=clock)
    for _ in range(rooms.PRESS_FAIL_AFTER):
        sat.renew()
    assert sat.press_paused is True
    assert sat.renew() is False              # no request at all while open
    presses = sat.presses
    assert len(wire.posts) == presses
    clock.tick(rooms.PRESS_COOLDOWN_S + 1.0)
    wire.post_ok = True
    with caplog.at_level(logging.INFO, logger="jarvis.rooms"):
        assert sat.renew() is True
    assert sat.press_paused is False
    assert any("back" in r.message for r in caplog.records)


# ------------------------------------------------------------- ungoverned
def test_a_listed_microphone_reads_ungoverned_not_unreachable():
    """A satellite with sensors: [radar, mic] made "Are you watching?"
    answer "I can't reach the kitchen mic", rolled the whole house up to
    UNKNOWN and never showed all-confirmed -- while the mic was reachable
    and deliberately ungoverned (F05)."""
    policy = Policy()
    sat, _, _ = make_sat(policy=policy, sensors=(RADAR, MIC))
    assert MIC not in " ".join(policy.attached)
    view = sat.view(MIC)
    assert view.state == rooms.UNGOVERNED
    assert "not governed by offline mode" in view.detail
    assert "not governed" in caption(view)
    word, _, filled = chip(view)
    assert word != chip(SensorView("kitchen", MIC, UNKNOWN))[0]
    assert filled is True                    # it is a known fact, not a gap


def test_an_ungoverned_mic_does_not_make_the_house_unknown():
    sat, _, _ = make_sat(policy=Policy(), sensors=(RADAR, MIC))
    mesh = RoomMesh(policy=Policy(), satellites=[sat])
    mesh.tick()                              # renew the radar, then confirm it
    view = mesh.view()
    states = {r.kind: r.state for r in view.rows}
    assert states[RADAR] == LIVE and states[MIC] == rooms.UNGOVERNED
    assert view.trustworthy() is True
    assert view.worst() == LIVE
    spoken = spoken_status(view)
    assert "can't reach" not in spoken
    assert "not governed by offline mode" in spoken


def test_ungoverned_beats_off_in_the_roll_up():
    """A one-chip roll-up that said OFF while a microphone was live would
    be the same lie one level up."""
    view = PrivacyView(rows=(SensorView("kitchen", RADAR, OFF),
                             SensorView("kitchen", MIC, rooms.UNGOVERNED)))
    assert view.worst() == rooms.UNGOVERNED


# ------------------------------------------------------------ still-on age
def test_the_still_on_caption_ages_from_the_revoke_not_the_confirmation():
    """caption() read view.age_s, which is refreshed by every mesh tick, so
    a satellite that had been refusing for an hour read "asked to stop 0 s
    ago" forever (F06)."""
    clock = Clock()
    wire = Wire({"/binary_sensor/Radar%20powered": ON})
    sat, wire, clock = make_sat(wire=wire, clock=clock)
    assert sat.revoke() is True
    clock.tick(7200.0)
    sat.confirm(RADAR)                       # a fresh confirmation: still ON
    view = sat.view(RADAR, stale_after_s=1e9)
    assert view.state == DISAGREE
    assert view.age_s == pytest.approx(0.0)          # last heard: just now
    assert view.revoked_age_s == pytest.approx(7200.0)
    assert caption(view) == "asked to stop 2.0 h ago and still reporting on"
    # and with the fix reverted -- ageing from the confirmation -- it read:
    assert "0 s" not in caption(view)


def test_a_renewed_lease_forgets_the_old_revoke():
    clock = Clock()
    wire = Wire({"/binary_sensor/Radar%20powered": ON})
    sat, wire, clock = make_sat(policy=Policy(), wire=wire, clock=clock)
    assert sat.revoke() is True
    clock.tick(60.0)
    assert sat.renew() is True
    assert sat.leases[RADAR].revoked_at is None
    clock.tick(5.0)
    sat.confirm(RADAR)
    assert sat.view(RADAR).state == LIVE
    assert sat.view(RADAR).revoked_age_s is None
