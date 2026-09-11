"""GREET AT THE DOOR, ASK AT THE DESK -- the WIRING (jarvis/app.py).

tests/test_arrival_desk.py pins the rules; this pins what the app does
with them: that the greeting still lands at the door with the catch-up
held back, that the held lines are NOT drained there, that either leg
delivers them at his desk and only one of them does, that a box which
cannot settle him keeps today's behaviour exactly, and that every honest
way this degrades leaves the backlog where it is.

Built the way tests/test_arrival_app.py builds it: a real JarvisApp
through ``object.__new__`` with only the sinks stubbed. No Tk, no mic, no
socket, no radar and NO LENS -- the camera reaches this feature as an
identity label and nothing in this file has ever held a frame.
"""
import logging
from types import SimpleNamespace

import pytest

from jarvis import app as app_mod
from jarvis import arrival as arrival_mod
from jarvis.config import CONFIG
from jarvis.presence import WELCOME_LINE

DESK = arrival_mod.DEFAULT_DESK_ZONE
HELD = ["While you were out, sir:", "The build passed, sir."]


class Cfg:
    def __init__(self, data=None):
        self.data = data or {}
        self.user_name = "Hunter"

    def get(self, key, default=None):
        return self.data.get(key, default)


class Quiet:
    """quiet.py's two seams this cue uses -- the reversible drain and the
    reason it is holding its tongue."""

    def __init__(self, frags=(), reason=""):
        self.frags = list(frags)
        self.put_backs = 0
        self._reason = reason

    def reason(self):
        return self._reason

    def take_fragments(self):
        taken, self.frags = self.frags, []
        done = []

        def put_back():
            if done:
                return 0
            done.append(True)
            self.put_backs += 1
            self.frags = list(taken) + list(self.frags)
            return len(taken)
        return list(taken), put_back

    def release_fragments(self):
        return self.take_fragments()[0]


class Verdict:
    """jarvis/zones.Verdict's shape, and only the three fields this reads."""

    def __init__(self, zone, rule="band", room="office"):
        self.zone, self.rule, self.room = zone, rule, room


def make_app(cfg=None, quiet=None, services=None, legs=("radar", "camera")):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = Cfg(cfg)
    a._init_assistant_state()
    a.tts = SimpleNamespace(spoken=[])
    a.tts.speak = a.tts.spoken.append
    a.quiet = quiet if quiet is not None else Quiet([])
    a.services = services if services is not None else \
        SimpleNamespace(panel_wake=None, briefing_offer=None)
    a._away_since = 0.0
    a._door = arrival_mod.DoorWatch()
    a._desk = arrival_mod.DeskWatch()
    a.presence = None
    # WHICH LEGS ARE LIVE is a question about hardware and config, and it
    # has its own tests below; every test that is about the SPLIT states
    # its answer rather than building a fake camera to imply it.
    if legs is not None:
        a._settle_legs = lambda: tuple(legs)
    return a


@pytest.fixture(autouse=True)
def _talkback():
    was = CONFIG.talkback
    CONFIG.talkback = True
    yield
    CONFIG.talkback = was


@pytest.fixture(autouse=True)
def _no_earcon(monkeypatch):
    """The door plan plays a sound; nothing in this file wants a speaker."""
    monkeypatch.setattr(app_mod.earcons, "play", lambda *a, **kw: True)


# ==================================================================
# AT THE DOOR: he is greeted, and NOT asked about his mail
# ==================================================================
def test_the_door_greets_him_and_holds_the_question_back():
    a = make_app(quiet=Quiet(HELD))
    a._unread_count = lambda: 3
    a._greet_return("room:kitchen")
    assert a.tts.spoken == [WELCOME_LINE], \
        "the catch-up was still asked in the hallway"
    assert a._desk.armed is True, "nothing owes him the catch-up"


def test_THE_HELD_LINES_ARE_NOT_DRAINED_AT_THE_DOOR():
    """The failure this branch's ancestor was caught on twice. They are
    what he missed while he was out and there is no second copy of them
    anywhere: deferring the catch-up must not TOUCH them, not even
    take-and-give-back."""
    a = make_app(quiet=Quiet(HELD))
    a._unread_count = lambda: 3
    a._greet_return("room:kitchen")
    assert a.quiet.frags == HELD, "the door destroyed the backlog"
    assert a.quiet.put_backs == 0, "it was taken and handed back, not left alone"


def test_the_ledger_at_the_door_records_a_greeting_and_no_catch_up(caplog):
    a = make_app(quiet=Quiet(HELD))
    with caplog.at_level(logging.INFO, logger="jarvis.arrival"):
        a._greet_return("room:kitchen")
    ledger = [r.getMessage() for r in caplog.records if " -> " in r.getMessage()]
    assert ledger == ["arrival (room:kitchen): panel -> earcon -> greeting"]
    assert "the catch-up is owed at his desk (radar and camera)" in caplog.text


def test_a_panel_only_quiet_arrival_owes_him_nothing():
    """He came home mid-meeting: the panel comes up, nothing is spoken,
    and the policy owns the backlog. A catch-up deferred here would ask
    him about his mail at his desk with no welcome ever having happened."""
    a = make_app(quiet=Quiet(HELD, reason="a meeting"))
    a._greet_return("room:kitchen")
    assert a.tts.spoken == [] and a._desk.armed is False


# ==================================================================
# ...UNLESS NOTHING COULD EVER DELIVER IT
# ==================================================================
def test_with_NO_LEG_LIVE_the_catch_up_is_still_asked_at_the_door():
    """The live box today: no zone source attached and no camera feed. A
    deferral nothing can fire is the catch-up silently never happening, so
    the cue stays exactly the one that shipped."""
    a = make_app(quiet=Quiet(HELD), legs=())
    a._unread_count = lambda: 3
    a._greet_return("room:kitchen")
    assert a.tts.spoken[0] == WELCOME_LINE
    assert "3 unread emails" in a.tts.spoken[1]
    assert a._desk.armed is False
    assert a.quiet.frags == [], "the backlog was not read out with the offer"


def test_the_split_can_be_switched_off_with_both_legs_live():
    a = make_app({"presence.settle_at_desk": False}, quiet=Quiet([]))
    a._unread_count = lambda: 3
    a._greet_return("room:kitchen")
    assert "3 unread emails" in a.tts.spoken[1] and a._desk.armed is False


def test_an_app_with_no_desk_watch_at_all_keeps_the_old_cue():
    a = make_app(quiet=Quiet([]), legs=None)
    del a._desk
    a._unread_count = lambda: 3
    a._greet_return("room:kitchen")
    assert "3 unread emails" in a.tts.spoken[1]


# ==================================================================
# AT THE DESK: either leg delivers it, and only one of them does
# ==================================================================
def _home(unread=3, quiet=None):
    a = make_app(quiet=quiet if quiet is not None else Quiet([]))
    a._unread_count = lambda: unread
    a._greet_return("room:kitchen")
    a.tts.spoken.clear()
    return a


def test_the_radar_leg_delivers_the_offer_at_his_desk():
    a = _home()
    assert a._settle(room="office", verdict=Verdict(DESK)) == "radar"
    assert "3 unread emails" in a.tts.spoken[0]
    assert a.tts.spoken[0].endswith("?"), "it delivered instead of offering"


def test_the_camera_leg_delivers_it_with_no_radar_at_all():
    """His back is to the module and his face is to the lens, and both of
    those are normal. An identity LABEL -- never a frame."""
    a = _home()
    assert a._settle(room="office", camera="Hunter") == "camera"
    assert "3 unread emails" in a.tts.spoken[0]


def test_BOTH_LEGS_ASK_HIM_ONCE():
    a = _home()
    legs = [a._settle(room="office", verdict=Verdict(DESK), camera="Hunter")
            for _ in range(20)]
    assert [leg for leg in legs if leg] == ["camera"]
    assert len(a.tts.spoken) == 1, "he was asked %d times" % len(a.tts.spoken)


def test_walking_THROUGH_the_office_does_not_spend_the_catch_up():
    """He crosses to the bedroom. The radar places him in the empty
    space, nothing is said, and the question is still owed."""
    a = _home()
    assert a._settle(room="office", verdict=Verdict("empty space")) == ""
    assert a.tts.spoken == [] and a._desk.armed is True


def test_a_settle_with_nothing_armed_says_nothing():
    """He got up for coffee and sat back down. No homecoming, no
    catch-up."""
    a = make_app(quiet=Quiet(HELD))
    a._unread_count = lambda: 3
    assert a._settle(room="office", verdict=Verdict(DESK)) == ""
    assert a.tts.spoken == [] and a.quiet.frags == HELD


def test_THE_BACKLOG_HELD_AT_THE_DOOR_IS_READ_OUT_AT_THE_DESK():
    """The whole point of deferring rather than dropping: the lines he
    missed travel from the hallway to the chair and are spoken there."""
    a = _home(quiet=Quiet(HELD))
    a._settle(room="office", camera="Hunter")
    said = a.tts.spoken[0]
    assert "The build passed" in said and "3 unread emails" in said
    assert a.quiet.frags == [] and a.quiet.put_backs == 0


def test_the_desk_offer_is_parked_on_the_protocol_the_commander_answers():
    """The same rung as the door's offer and the first-wake briefing: a
    "yes" must not mean different things in different rooms."""
    a = _home()
    a._settle(room="office", camera="Hunter")
    offer = a.services.briefing_offer
    assert isinstance(offer, dict) and callable(offer["deliver"])
    assert a._followup_after_speech is True, "nobody was listening for the yes"


def test_the_log_says_WHICH_LEG_decided(caplog):
    a = _home()
    with caplog.at_level(logging.INFO, logger="jarvis.arrival"):
        a._settle(room="office", verdict=Verdict(DESK))
    assert "by the radar" in caplog.text


def test_a_camera_ruled_verdict_is_not_credited_to_the_radar(caplog):
    """zones.verdict lets the camera overrule the radar and the result
    still carries the desk zone. Reading the zone alone would put the
    radar in the log for a decision the lens made."""
    a = _home()
    with caplog.at_level(logging.INFO, logger="jarvis.arrival"):
        a._settle(room="office", verdict=Verdict(DESK, rule="camera"))
    assert "by the camera" in caplog.text


# ==================================================================
# IF HE NEVER REACHES THE DESK
# ==================================================================
def test_he_goes_straight_to_bed_and_the_lines_are_STILL_HELD():
    """Straight past the office, or the module unplugged, or the lens
    inside its 21:00-07:00 curfew. Nothing was taken at the door, so the
    backlog is exactly where quiet.py's own next tick will find it."""
    a = _home(quiet=Quiet(HELD))
    for room in ("kitchen", "bedroom", ""):
        assert a._settle(room=room, camera="Hunter") == ""
    assert a.quiet.frags == HELD and a.quiet.put_backs == 0
    assert a._desk.armed is True, "the catch-up was quietly given up on"


def test_a_deferred_catch_up_DOES_NOT_EXPIRE():
    """It is a live read of the mailbox and the fault board taken at
    delivery, so there is nothing captured at the door to go stale -- and
    the only thing an expiry could do is throw away lines that are his."""
    a = _home(quiet=Quiet(HELD))
    for _ in range(500):
        a._settle(room="kitchen", camera="Hunter")
    assert a._desk.armed is True
    a._settle(room="office", camera="Hunter")
    assert "The build passed" in a.tts.spoken[0]


def test_a_QUIET_HOUR_at_the_desk_leaves_it_OWED_and_says_nothing():
    """The one new way this feature could speak over a hold: at the door
    a quiet house is panel-only, but a DEFERRED catch-up arrives an hour
    later and _say is called for the digest with proactive=False."""
    a = _home(quiet=Quiet(HELD))
    a.quiet._reason = "quiet hours"
    assert a._settle(room="office", camera="Hunter") == ""
    assert a.tts.spoken == []
    assert a.quiet.frags == HELD and a.quiet.put_backs == 0
    assert a._desk.armed is True
    # ...and it is delivered when the window closes.
    a.quiet._reason = ""
    assert a._settle(room="office", camera="Hunter") == "camera"
    assert "The build passed" in a.tts.spoken[0]


def test_a_settle_that_had_NOTHING_TO_SAY_does_not_stay_owed():
    """No mail, no fault, no backlog. There is nothing to ask about, so
    the catch-up is done rather than owed for ever."""
    a = _home(unread=0, quiet=Quiet([]))
    assert a._settle(room="office", camera="Hunter") == "camera"
    assert a.tts.spoken == [] and a._desk.armed is False


# ==================================================================
# THE ROOM-CHANGE TRIGGER
# ==================================================================
def _room(room, previous=""):
    from jarvis.events import RoomChanged
    return RoomChanged(room=room, previous=previous)


def test_the_kitchen_greets_and_the_office_then_asks():
    """His flat end to end: front door -> kitchen -> office, and the two
    halves of the cue land in the two rooms."""
    a = make_app(quiet=Quiet(HELD))
    a._unread_count = lambda: 1
    a.presence = SimpleNamespace(state="away")
    a._eye_identity = lambda: "Hunter"
    a._on_room_changed(_room("kitchen"))
    assert a.tts.spoken == [WELCOME_LINE] and a.quiet.frags == HELD
    a._on_room_changed(_room("office", previous="kitchen"))
    assert "The build passed" in a.tts.spoken[1]
    assert "1 unread email" in a.tts.spoken[1]


def test_a_room_change_with_no_camera_opinion_settles_nothing():
    """The lens off, or inside the curfew: _eye_identity is "" and the
    office room alone is not the desk band."""
    a = _home(quiet=Quiet(HELD))
    a._eye_identity = lambda: ""
    a._on_room_changed(_room("office", previous="kitchen"))
    assert a.tts.spoken == [] and a._desk.armed is True


def test_a_broken_eye_costs_the_observation_and_not_the_subscriber():
    a = _home(quiet=Quiet(HELD))

    def boom():
        raise RuntimeError("the eye exploded")

    a._eye_identity = boom
    a._on_room_changed(_room("office", previous="kitchen"))
    assert a._desk.armed is True and a.quiet.frags == HELD


# ==================================================================
# WHICH LEGS ARE LIVE -- read at the door, and it may not overstate
# ==================================================================
def test_a_box_with_no_zone_source_and_no_camera_has_no_legs():
    a = make_app(legs=None)
    assert a._settle_legs() == ()


def test_a_camera_feed_is_the_camera_leg():
    """UPDATED 2026-09-06 and deliberately TIGHTENED. A feed alone was
    enough while nothing on the tree ever attached one -- the leg could
    not be counted, so the gate could not be wrong. Now that the app owns
    a feed, all three of the things that would actually have to publish
    this leg are checked: the feed, the producer that looks through it,
    and a gallery that can put a NAME to what it sees. A leg counted here
    that cannot fire is his mail question silently never being asked."""
    a = make_app(legs=None,
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          camera_feed=object(),
                                          eyeloop=object()))
    a._camera_can_name_cache = True
    assert a._settle_legs() == ("camera",)


def test_a_feed_with_no_producer_is_not_a_camera_leg():
    """A feed nothing looks through never publishes a reading, so
    _eye_identity answers "" for ever and the catch-up would wait for a
    settle that cannot arrive."""
    a = make_app(legs=None,
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          camera_feed=object()))
    a._camera_can_name_cache = True
    assert a._settle_legs() == ()


def test_a_camera_that_cannot_name_anybody_is_not_a_camera_leg():
    """This leg is delivered by a NAME. With identity off, nobody
    enrolled, or an enrolment belonging to a model that is no longer the
    active one (a real state on this box since the 09-03 backend swap),
    the producer counts faces and names nobody -- which never settles."""
    a = make_app(legs=None,
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          camera_feed=object(),
                                          eyeloop=object()))
    a._camera_can_name_cache = False
    assert a._settle_legs() == ()


def test_a_zone_source_with_NO_ZONES_CONFIG_is_not_a_leg():
    """A verdict lane attached to a room with no map never names the desk
    band, so deferring on the strength of it would lose the catch-up."""
    a = make_app(legs=None,
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          zone_source=object()))
    assert a._settle_legs() == ()


def test_a_zone_source_with_the_desk_band_configured_IS_a_leg():
    a = make_app({"zones": {"rooms": [{"name": "office", "bands": [
                     {"name": "empty space", "near_m": 0.75, "far_m": 2.25},
                     {"name": DESK, "near_m": 2.25, "far_m": 3.75}]}]}},
                 legs=None,
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          zone_source=object()))
    assert a._settle_legs() == ("radar",)


def test_a_zone_source_whose_map_names_no_DESK_band_is_not_a_leg():
    a = make_app({"zones": {"rooms": [{"name": "office", "bands": [
                     {"name": "somewhere", "near_m": 0.75, "far_m": 3.75}]}]}},
                 legs=None,
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          zone_source=object()))
    assert a._settle_legs() == ()


def test_the_desk_zone_is_the_one_zones_py_names():
    """arrival.py copies the name rather than importing zones.py, which
    owns a log file and a config reader. This is what pins the copy."""
    from jarvis import zones as zones_mod
    assert arrival_mod.DEFAULT_DESK_ZONE == zones_mod.DEFAULT_CAMERA_ZONE
    # ...and the leg names are the rule names, so a Verdict's own `rule`
    # can be compared to them without a translation table in between.
    assert arrival_mod.LEG_CAMERA == zones_mod.RULE_CAMERA
    assert arrival_mod.LEG_RADAR != zones_mod.RULE_CAMERA


# ==================================================================
# THE ENTRY POINTS THE TWO LANES CALL
# ==================================================================
def test_a_zone_lane_delivers_through_note_zone_verdict():
    a = _home()
    assert a.note_zone_verdict(Verdict(DESK)) == "radar"
    assert "3 unread emails" in a.tts.spoken[0]


def test_a_camera_lane_delivers_through_note_camera_identity():
    a = _home()
    assert a.note_camera_identity("Hunter") == "camera"
    assert "3 unread emails" in a.tts.spoken[0]


def test_note_camera_identity_takes_a_NAME_and_a_score_and_nothing_else():
    """The hard boundary, asserted rather than asserted-in-a-comment: the
    camera-side entry point has no parameter a frame could be passed
    through."""
    import inspect
    params = set(inspect.signature(app_mod.JarvisApp.note_camera_identity)
                 .parameters)
    assert params == {"self", "label", "score", "room"}


# ==================================================================
# A DELIVERY THE WORKER DROPS IS STILL OWED
# ==================================================================
# With a mailbox configured the catch-up finishes on a thread, and
# _arrival_catch_up_stale can decide by then that a turn owns the floor.
# At the door that drop costs the last word of a cue that has already
# spoken. At his desk it is the WHOLE delivery -- and he was asked nothing
# at all -- so the arm goes back and his next reading asks again.
FAKE_ACCOUNT = {"label": "test", "address": "someone@example.com",
                "app_password": "not-a-real-password"}


def _gated_mailbox(gate):
    def fetch_unread(*a, **kw):
        gate.wait(timeout=10.0)
        return [SimpleNamespace(sender="Canvas", subject="Lab 3 graded")]
    return fetch_unread


def _settling_on_a_worker(monkeypatch, gate, held=()):
    import jarvis.tools.mail as mail_mod
    monkeypatch.setattr(mail_mod, "fetch_unread", _gated_mailbox(gate))
    a = make_app({"gmail.accounts": [FAKE_ACCOUNT]}, quiet=Quiet(list(held)))
    a._greet_return("room:kitchen")
    a.tts.spoken.clear()
    assert a._settle(room="office", camera="Hunter") == "camera"
    return a


def _finish(a, gate):
    gate.set()
    a._arrival_catch_up_thread.join(timeout=10.0)
    assert not a._arrival_catch_up_thread.is_alive()


def test_a_desk_delivery_the_worker_DROPS_is_owed_again(monkeypatch):
    import threading as _threading
    gate = _threading.Event()
    a = _settling_on_a_worker(monkeypatch, gate, held=HELD)
    a._dispatch_gen += 1                    # he asked Jarvis something
    _finish(a, gate)
    assert a.tts.spoken == []
    assert a.quiet.frags == HELD and a.quiet.put_backs == 0
    assert a._desk.armed is True, "the catch-up was dropped, not deferred"


def test_a_desk_delivery_that_SPOKE_is_not_owed_again(monkeypatch):
    import threading as _threading
    gate = _threading.Event()
    a = _settling_on_a_worker(monkeypatch, gate, held=HELD)
    _finish(a, gate)
    assert "The build passed" in a.tts.spoken[0]
    assert a._desk.armed is False


def test_the_re_armed_delivery_actually_lands_next_time(monkeypatch):
    import threading as _threading
    gate = _threading.Event()
    a = _settling_on_a_worker(monkeypatch, gate, held=HELD)
    a._dispatch_gen += 1
    _finish(a, gate)
    # ...he stops talking, and the radar commits his desk band again.
    gate.clear()
    assert a._settle(room="office", verdict=Verdict(DESK)) == "radar"
    _finish(a, gate)
    assert "The build passed" in a.tts.spoken[0]


def test_the_door_does_not_re_arm_on_a_drop(monkeypatch):
    """Only the deferred delivery passes a `missed` hook. At the door the
    catch-up ran as part of the cue and a drop is the documented, costless
    outcome -- re-arming there would ask him at his desk about a question
    the hallway had already put to him."""
    import threading as _threading
    gate = _threading.Event()
    import jarvis.tools.mail as mail_mod
    monkeypatch.setattr(mail_mod, "fetch_unread", _gated_mailbox(gate))
    a = make_app({"gmail.accounts": [FAKE_ACCOUNT]}, quiet=Quiet(HELD), legs=())
    a._greet_return("room:kitchen")
    a._dispatch_gen += 1
    _finish(a, gate)
    assert a._desk.armed is False


# ==================================================================
# THE NAP GATE, WIRED (jarvis/app.py)
# ==================================================================
def test_a_phone_return_before_the_confirm_closed_says_nothing():
    """Seven times in two days, measured: away, then home(returned) ten
    seconds later, then "Welcome back, sir" to a man who never left."""
    a = make_app(quiet=Quiet(HELD))
    a._departure_pending = True
    a._greet_return("phone")
    assert a.tts.spoken == [], "the false welcome is still spoken"
    assert a.quiet.frags == HELD, "a nap drained the backlog he was owed"


def test_the_same_return_once_the_confirm_has_closed_is_greeted():
    a = make_app(quiet=Quiet(HELD))
    a._departure_pending = False
    a._greet_return("phone")
    assert a.tts.spoken == [WELCOME_LINE]


def test_a_door_return_is_greeted_even_with_the_confirm_still_armed():
    """The radar saw a body. Only the radio is doubted."""
    a = make_app(quiet=Quiet(HELD))
    a._departure_pending = True
    a._greet_return("room:kitchen")
    assert a.tts.spoken == [WELCOME_LINE]


def test_the_refused_nap_is_logged_by_name(caplog):
    a = make_app(quiet=Quiet(HELD))
    a._departure_pending = True
    with caplog.at_level(logging.INFO, logger="jarvis.app"):
        a._greet_return("phone")
    assert any("nap" in r.getMessage() for r in caplog.records), \
        "it refused anonymously, which is the defect this lane exists for"


def test_cancelling_an_armed_confirm_reports_that_it_was_pending():
    a = make_app()
    a._departure_timer = SimpleNamespace(cancel=lambda: None)
    assert a._cancel_departure() is True
    assert a._departure_pending is True
    assert a._cancel_departure() is False
    assert a._departure_pending is False
