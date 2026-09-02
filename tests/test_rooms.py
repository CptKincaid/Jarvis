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
    wire = wire or Wire({"/binary_sensor/presence": ON,
                         "/binary_sensor/radar_powered": ON})
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
    wire = Wire({"/binary_sensor/presence": ON,
                 "/binary_sensor/radar_powered": OFF_BODY})
    sat, wire, _ = make_sat(policy=Policy(), wire=wire)
    assert sat.read() is None
    # the presence entity was never asked: an unconfirmed sensor's reading
    # describes a moment nobody can date
    assert sat.presence.reads == 0


def test_presence_is_none_when_the_device_does_not_answer():
    wire = Wire({"/binary_sensor/presence": ON})   # powered path 404s
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
    policy = Policy()
    sat, _, _ = make_sat(policy=policy, sensors=(RADAR, MIC))
    assert MIC not in " ".join(policy.attached)
    view = sat.view(MIC)
    assert view.state == UNKNOWN
    assert "not governed by offline mode" in view.detail


def test_renew_extends_the_lease_and_a_failed_renew_does_not():
    clock = Clock()
    wire = Wire({"/binary_sensor/radar_powered": ON})
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
    assert wire.posts[-1].endswith("/button/radar_lease_revoke/press")


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
    wire = wire or Wire({"/binary_sensor/presence": ON,
                         "/binary_sensor/radar_powered": ON})
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
    wire = Wire({"/binary_sensor/radar_powered": ON}, post_ok=False)
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
    assert any("/binary_sensor/radar_powered" in u for u in wire.gets)
    assert mesh.view().rows[0].state == DISAGREE


def test_one_bad_satellite_does_not_cost_the_others_their_lease():
    class Boom(Satellite):
        def renew(self):
            raise RuntimeError("nope")

    policy = Policy()
    clock, wire = Clock(), Wire({"/binary_sensor/radar_powered": ON})
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
    cfg = Cfg({"rooms.enabled": False, "rooms.satellites": [
        {"name": "kitchen", "url": "http://192.168.50.61"}]})
    assert specs_from_config(cfg) == ()
    cfg = Cfg({"rooms.enabled": True, "rooms.satellites": [
        {"name": "kitchen", "url": "http://192.168.50.61"},
        {"name": "kitchen", "url": "http://192.168.50.62"},
        {"name": "bedroom", "url": "http://8.8.8.8"},
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


def test_the_curfew_closes_the_lens_and_leaves_the_radar_up():
    """The ruling jarvis/sensing.py made for the local box, carried one
    room out: at 21:00 the camera stops being renewed and the radar does
    not. A single room-wide lease could not express that -- which is why
    there is one lease per KIND."""
    policy = Policy(camera=False, radar=True)   # i.e. curfew_active()
    wire = Wire({"/binary_sensor/presence": ON,
                 "/binary_sensor/radar_powered": ON,
                 "/binary_sensor/camera_powered": OFF_BODY})
    sat, wire, _ = make_sat(policy=policy, wire=wire, sensors=(RADAR, CAMERA))
    mesh = RoomMesh(policy=policy, satellites=[sat])
    mesh.tick()
    assert any(u.endswith("/button/radar_lease_renew/press") for u in wire.posts)
    assert any(u.endswith("/button/camera_lease_revoke/press")
               for u in wire.posts)
    assert not any("camera_lease_renew" in u for u in wire.posts)
    states = {r.kind: r.state for r in mesh.view().rows}
    assert states[RADAR] == LIVE and states[CAMERA] == OFF
    # ...and presence still works during the curfew, which is the point
    assert sat.read() is not None


def test_offline_mode_takes_both_kinds_down_in_every_room():
    policy = Policy(camera=False, radar=False)
    wire = Wire({"/binary_sensor/radar_powered": OFF_BODY,
                 "/binary_sensor/camera_powered": OFF_BODY})
    sat, wire, _ = make_sat(policy=policy, wire=wire, sensors=(RADAR, CAMERA))
    mesh = RoomMesh(policy=policy, satellites=[sat])
    mesh.tick()
    revoked = {u.rsplit("/button/", 1)[1] for u in wire.posts}
    assert revoked == {"radar_lease_revoke/press", "camera_lease_revoke/press"}
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
