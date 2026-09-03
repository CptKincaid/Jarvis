"""The arrival cue as the APP composes it (jarvis/app.py).

tests/test_arrival_kitchen.py pins the rules; this pins the WIRING -- that
the greeting reads the calendar cache and not a network, that the offer
rides the same thinned burst as the quiet digest, that the offer is parked
on the one protocol the commander already answers, and that every one of
those degrades to "Welcome back, sir" when its source is absent.

Built the way tests/test_address.py builds it: a real JarvisApp through
``object.__new__`` with only the sinks stubbed. No Tk, no mic, no socket.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from jarvis import app as app_mod
from jarvis import arrival as arrival_mod
from jarvis.config import CONFIG
from jarvis.presence import WELCOME_LINE


class Cfg:
    def __init__(self, data=None):
        self.data = data or {}
        self.user_name = "Hunter"

    def get(self, key, default=None):
        return self.data.get(key, default)


class Quiet:
    """quiet.py's one seam the arrival cue uses."""

    def __init__(self, frags=()):
        self.frags = list(frags)

    def release_fragments(self):
        frags, self.frags = self.frags, []
        return frags


def make_app(cfg=None, quiet=None, services=None):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = Cfg(cfg)
    a._init_assistant_state()
    a.tts = SimpleNamespace(spoken=[])
    a.tts.speak = a.tts.spoken.append
    a.quiet = quiet
    a.services = services if services is not None else SimpleNamespace(panel_wake=None)
    a._away_since = 0.0
    a._door = arrival_mod.DoorWatch()
    a.presence = None
    return a


@pytest.fixture(autouse=True)
def _talkback():
    was = CONFIG.talkback
    CONFIG.talkback = True
    yield
    CONFIG.talkback = was


def ev(title, start, end):
    return SimpleNamespace(title=title, start=start, end=end, all_day=False,
                           location="", calendar="")


class Calendar:
    """The parked CalendarSource's two attributes the greeting reads. It
    counts events() calls so "no network on the doorstep" is an assertion
    and not a hope."""

    def __init__(self, events=(), configured=True, boom=False):
        self._events = list(events)
        self.configured = configured
        self.boom = boom
        self.reads = 0

    def events(self):
        self.reads += 1
        if self.boom:
            raise RuntimeError("caldav is unreachable")
        return list(self._events)


# ==================================================================
# "Welcome back from X"
# ==================================================================
def test_with_no_calendar_wired_the_greeting_is_the_plain_line():
    a = make_app()
    a._away_since = 1_000.0
    assert a._welcome_text() == WELCOME_LINE


def test_with_no_recorded_departure_the_calendar_is_not_even_asked():
    """Jarvis restarted while he was out: there is no absence to match, and
    asking would only invite a wrong answer."""
    cal = Calendar([ev("Advisor meeting", datetime.now().astimezone(),
                       datetime.now().astimezone())])
    a = make_app(services=SimpleNamespace(calendar=cal))
    assert a._welcome_text() == WELCOME_LINE and cal.reads == 0


def test_an_event_that_covered_the_absence_is_named_in_the_greeting():
    now = datetime.now().astimezone()
    cal = Calendar([ev("the dentist", now - timedelta(hours=2),
                       now - timedelta(minutes=10))])
    a = make_app(services=SimpleNamespace(calendar=cal))
    a._away_since = (now - timedelta(hours=2, minutes=10)).timestamp()
    assert a._welcome_text() == "Welcome back from the dentist, sir."


def test_a_short_errand_does_not_name_a_long_absence_at_the_app_edge():
    """The verifier's app-level repro, pinned where it actually bit: four
    hours away, one four-minute calendar entry that ended sixteen minutes
    before he walked in, and the greeting said "Welcome back from take the
    bins out, sir." The absence-coverage half of the rule (arrival.outing)
    is what makes it the plain line again."""
    now = datetime.now().astimezone()
    cal = Calendar([ev("take the bins out", now - timedelta(minutes=20),
                       now - timedelta(minutes=16))])
    a = make_app(services=SimpleNamespace(calendar=cal))
    a._away_since = (now - timedelta(hours=4)).timestamp()
    assert a._welcome_text() == WELCOME_LINE


def test_an_unreachable_calendar_costs_the_name_and_not_the_greeting():
    now = datetime.now().astimezone()
    cal = Calendar([], boom=True)
    a = make_app(services=SimpleNamespace(calendar=cal))
    a._away_since = (now - timedelta(hours=2)).timestamp()
    assert a._welcome_text() == WELCOME_LINE and cal.reads == 1


def test_an_unconfigured_calendar_is_never_read():
    cal = Calendar([], configured=False)
    a = make_app(services=SimpleNamespace(calendar=cal))
    a._away_since = 1_000.0
    assert a._welcome_text() == WELCOME_LINE and cal.reads == 0


def test_the_outing_can_be_switched_off_without_losing_the_welcome():
    now = datetime.now().astimezone()
    cal = Calendar([ev("the dentist", now - timedelta(hours=2),
                       now - timedelta(minutes=10))])
    a = make_app({"presence.arrival_outing": False},
                 services=SimpleNamespace(calendar=cal))
    a._away_since = (now - timedelta(hours=2, minutes=10)).timestamp()
    assert a._welcome_text() == WELCOME_LINE and cal.reads == 0


def test_the_greeting_step_speaks_whatever_welcome_text_decided():
    now = datetime.now().astimezone()
    cal = Calendar([ev("Organic Chemistry", now - timedelta(hours=2),
                       now - timedelta(minutes=5))])
    a = make_app(services=SimpleNamespace(calendar=cal, panel_wake=None))
    a._away_since = (now - timedelta(hours=2, minutes=5)).timestamp()
    assert arrival_mod.run(["greeting"], a._arrival_actions()) == ["greeting"]
    assert a.tts.spoken == ["Welcome back from Organic Chemistry, sir."]


# ==================================================================
# The catch-up OFFERS
# ==================================================================
def test_with_no_mailbox_and_a_clear_board_nothing_is_offered():
    """The dark-safe path, and the one the live box is on today: no
    gmail account configured means fetch_unread raises before a socket is
    opened, and the count is None rather than zero."""
    a = make_app()
    assert a._unread_count() is None
    assert a._arrival_offer_line() == ""


def test_the_offer_rides_the_same_burst_as_the_quiet_digest():
    """ONE thinning pass over welcome + digest + offer. Two passes is how
    the address thinning was got wrong the first time (jarvis/address.py),
    and the burst ledger comment in _arrival_actions is about exactly this.
    """
    a = make_app(quiet=Quiet(["While you were out, sir:", "The build passed, sir."]))
    a._unread_count = lambda: 3
    done = arrival_mod.run(["greeting", "catch-up"], a._arrival_actions())
    assert done == ["greeting", "catch-up"]
    assert a.tts.spoken[0] == WELCOME_LINE
    tail = a.tts.spoken[1]
    assert "3 unread emails" in tail and tail.endswith("?")
    # The welcome already addressed him; the burst carries ONE more "sir"
    # at most, and the offer's own is the one that survives the thinning.
    assert " ".join(a.tts.spoken).count("sir") <= 2


def test_the_offer_speaks_even_when_there_was_no_backlog_to_release():
    """Nothing was held (he was out in the daytime), but there are three
    unread emails: that is still worth a question."""
    a = make_app(quiet=Quiet([]))
    a._unread_count = lambda: 3
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == ["catch-up"]
    assert "3 unread emails" in a.tts.spoken[0]


def test_nothing_to_say_still_reports_the_step_as_not_run():
    a = make_app(quiet=Quiet([]))
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == []
    assert a.tts.spoken == []


def test_the_offer_is_parked_on_the_one_protocol_the_commander_answers():
    """services.briefing_offer + Commander._try_briefing_offer, not a
    second offer of its own: a "yes" must not mean different things on
    different rungs."""
    a = make_app(quiet=Quiet([]), services=SimpleNamespace(panel_wake=None,
                                                           briefing_offer=None))
    a._unread_count = lambda: 2
    arrival_mod.run(["catch-up"], a._arrival_actions())
    offer = a.services.briefing_offer
    assert isinstance(offer, dict) and callable(offer["deliver"])
    assert offer["made_at"] > 0
    # A question nobody listens for is the stuck-listen bug in miniature.
    assert a._followup_after_speech is True


def test_nothing_is_parked_when_nothing_was_offered():
    a = make_app(quiet=Quiet(["While you were out, sir:", "The build passed, sir."]),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert a.services.briefing_offer is None


def test_only_an_error_is_major_enough_to_meet_him_at_the_door():
    warn = SimpleNamespace(kind="warn", line="Memory is getting tight.", text="")
    a = make_app(services=SimpleNamespace(panel_wake=None,
                                          faults=SimpleNamespace(current=warn)))
    assert a._major_line() == ""
    err = SimpleNamespace(kind="error", line="The disk is full.", text="")
    a.services.faults = SimpleNamespace(current=err)
    assert a._major_line() == "The disk is full."


def test_the_major_clause_with_no_mailbox_is_a_statement_not_a_question():
    """The nonsense offer, at the app's edge. With no mailbox and an error
    standing this used to be "The disk is full. Shall I go through it,
    sir?" -- and the yes it invited spoke that sentence back."""
    err = SimpleNamespace(kind="error", line="The disk is full.", text="")
    a = make_app(services=SimpleNamespace(panel_wake=None,
                                          faults=SimpleNamespace(current=err)))
    line = a._arrival_offer_line()
    assert line == "The disk is full."
    assert not line.endswith("?") and "unread" not in line


def test_a_fault_only_line_is_spoken_but_nothing_is_parked():
    """A statement is not an offer: parking one would leave a "yes"
    hanging on nothing, and open the mic for an answer to no question."""
    err = SimpleNamespace(kind="error", line="Ollama is unreachable.", text="")
    a = make_app(quiet=Quiet([]),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          faults=SimpleNamespace(current=err)))
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == ["catch-up"]
    assert a.tts.spoken == ["Ollama is unreachable."]
    assert a.services.briefing_offer is None
    assert a._followup_after_speech is False


def test_the_delivery_does_not_read_the_fault_the_offer_already_spoke():
    """He says yes and hears the sentence he just heard. The offer speaks
    the fault; the delivery is about the MAIL."""
    import jarvis.tools.mail as mail_mod
    err = SimpleNamespace(kind="error", line="Ollama is unreachable.", text="")
    mails = [SimpleNamespace(sender="Canvas", subject="Lab 3 graded")]
    a = make_app(quiet=Quiet([]),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          faults=SimpleNamespace(current=err)))
    a._unread_count = lambda: len(mails)
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert "Ollama is unreachable." in a.tts.spoken[0]
    a.tts.spoken.clear()
    orig = mail_mod.fetch_unread
    mail_mod.fetch_unread = lambda *args, **kw: list(mails)
    try:
        assert a.services.briefing_offer["deliver"]() is True
    finally:
        mail_mod.fetch_unread = orig
    said = " ".join(a.tts.spoken)
    assert "Ollama" not in said, "the fault was read back a second time"
    assert "Canvas, Lab 3 graded." in said


def test_a_fault_that_appeared_AFTER_the_offer_is_still_delivered():
    """The narrow case the `said` carry-through has to keep: the delivery
    skips only the sentence the offer actually spoke, not every fault."""
    a = make_app()
    later = SimpleNamespace(kind="error", line="The disk is full.", text="")
    a.services = SimpleNamespace(panel_wake=None,
                                 faults=SimpleNamespace(current=later))
    assert a._deliver_arrival_catch_up(said="Ollama is unreachable.") is True
    assert "The disk is full." in " ".join(a.tts.spoken)


def test_the_offer_can_be_switched_off():
    a = make_app({"presence.arrival_offer": False})
    a._unread_count = lambda: 9
    assert a._arrival_offer_line() == ""


def test_the_delivery_reads_senders_and_subjects_and_only_after_a_yes():
    """The yes is what buys the contents. Until then not one subject line
    has been spoken -- which is the whole shape of the ruling."""
    mails = [SimpleNamespace(sender="Dr Villalobos", subject="thesis draft"),
             SimpleNamespace(sender="Canvas", subject="Lab 3 graded"),
             SimpleNamespace(sender="Bank", subject="statement ready"),
             SimpleNamespace(sender="Spam", subject="you have won")]
    a = make_app(quiet=Quiet([]), services=SimpleNamespace(panel_wake=None,
                                                           briefing_offer=None))
    a._unread_count = lambda: len(mails)
    arrival_mod.run(["catch-up"], a._arrival_actions())
    offered = " ".join(a.tts.spoken)
    assert "thesis draft" not in offered and "Villalobos" not in offered

    import jarvis.tools.mail as mail_mod
    a.tts.spoken.clear()
    orig = mail_mod.fetch_unread
    mail_mod.fetch_unread = lambda *args, **kw: list(mails)
    try:
        assert a.services.briefing_offer["deliver"]() is True
    finally:
        mail_mod.fetch_unread = orig
    said = " ".join(a.tts.spoken)
    assert "Dr Villalobos, thesis draft." in said
    assert "you have won" not in said and "And 1 more." in said


def test_a_delivery_with_nothing_left_to_say_reports_false():
    """_try_briefing_offer owns telling him a yes found nothing; a delivery
    that silently says nothing and claims success is the worse failure."""
    a = make_app()
    assert a._deliver_arrival_catch_up() is False


# ==================================================================
# The kitchen as the door, at the app's edge
# ==================================================================
def test_a_room_change_while_he_is_home_greets_nobody():
    a = make_app()
    a.presence = SimpleNamespace(state="home")
    greeted = []
    a._greet_return = greeted.append
    a._on_room_changed(SimpleNamespace(room="kitchen", previous="office"))
    assert greeted == []


def test_the_kitchen_after_an_absence_greets_through_the_shared_greeter():
    a = make_app()
    a.presence = SimpleNamespace(state="away")
    greeted = []
    a._greet_return = greeted.append
    a._on_room_changed(SimpleNamespace(room="kitchen", previous=""))
    assert greeted == ["room:kitchen"]
    # The office lighting up on the way through is the same walk.
    a._on_room_changed(SimpleNamespace(room="office", previous="kitchen"))
    a._on_room_changed(SimpleNamespace(room="kitchen", previous="office"))
    assert greeted == ["room:kitchen"]


def test_a_fresh_boot_with_him_at_his_desk_is_not_an_arrival():
    """presence is "unknown" until the first probe lands, and unknown is
    not away. Greeting on it would welcome him home every restart."""
    a = make_app()
    a.presence = SimpleNamespace(state="unknown")
    greeted = []
    a._greet_return = greeted.append
    a._on_room_changed(SimpleNamespace(room="kitchen", previous=""))
    assert greeted == []


def test_a_departure_stamps_when_he_was_last_seen_and_re_arms_the_door():
    """_away_since is what "welcome back from X" matches against, and
    presence.last_seen is the honest departure -- ev.since is twelve
    minutes later on the default grace."""
    a = make_app()
    a.presence = SimpleNamespace(state="away", last_seen=1_000.0)
    a._cancel_departure = lambda: None
    a._arm_departure = lambda ev: None
    a._door._fired = True
    a._on_presence(SimpleNamespace(home=False, returned=False, since=1_720.0))
    assert a._away_since == 1_000.0
    assert a._door.observe(room="kitchen", away=True) is True


def test_a_departure_with_no_last_seen_falls_back_to_the_event():
    a = make_app()
    a.presence = SimpleNamespace(state="away", last_seen=None)
    a._cancel_departure = lambda: None
    a._arm_departure = lambda ev: None
    a._on_presence(SimpleNamespace(home=False, returned=False, since=1_720.0))
    assert a._away_since == 1_720.0


# ==================================================================
# The leg underneath it: presence.rooms builds the fabric
# ==================================================================
def test_configured_rooms_become_the_presence_leg():
    """The kitchen is only a door sensor if something POLLS it. With
    presence.rooms set the leg is roomfabric's HouseView -- which is what
    publishes RoomChanged -- and not one RoomSensor."""
    from jarvis.presence import PresenceSentinel
    p = PresenceSentinel(Cfg({
        "presence.room_sensor_enabled": True,
        "presence.rooms": [
            {"name": "office", "url": "http://192.168.50.51", "primary": True},
            {"name": "kitchen", "url": "http://192.168.50.52"}]}))
    assert p.fabric is not None and len(p.fabric) == 2
    assert type(p.sensor).__name__ == "HouseView"
    assert p.configured is True


def test_a_single_sensor_config_still_takes_the_old_path_byte_for_byte():
    """The box that is live today has one radar and no `rooms` list; this
    feature may not change what it builds."""
    from jarvis.presence import PresenceSentinel
    p = PresenceSentinel(Cfg({"presence.room_sensor_enabled": True,
                              "presence.room_sensor_url": "http://192.168.50.51"}))
    assert p.fabric is None
    assert type(p.sensor).__name__ == "RoomSensor"


def test_a_broken_rooms_list_costs_the_fabric_and_not_the_phone_leg():
    from jarvis.presence import PresenceSentinel
    p = PresenceSentinel(Cfg({"presence.room_sensor_enabled": True,
                              "presence.rooms": ["not a room"],
                              "presence.phone_ip": "192.168.50.9"}))
    assert p.fabric is None and p.sensor is None and p.configured is True


def test_stopping_the_sentinel_stops_the_fabric_with_it():
    """A fabric left running would keep polling every ESP32 in the flat
    into the teardown."""
    from jarvis.presence import PresenceSentinel
    p = PresenceSentinel(Cfg({}))
    calls = []
    p.fabric = SimpleNamespace(start=lambda: calls.append("start"),
                               stop=lambda: calls.append("stop"))
    p.stop()
    assert calls == ["stop"]


# ==================================================================
# The catch-up must not freeze the window it is greeting him through
# ==================================================================
# `bus.attach_tk(root)` makes publish() queue and drain() run from the UI's
# `_pump`, so every subscriber -- `_on_presence`, `_on_room_changed`, and
# the whole arrival cue behind them -- executes on the Tk MAIN thread. The
# unread count is an IMAP round trip (IMAP_TIMEOUT 15 s a mailbox;
# mail.py records a measured 8.1 s across his three accounts), so paying
# it inline froze the window and every event behind it at the moment he
# walked in.
FAKE_ACCOUNT = {"label": "test", "address": "someone@example.com",
                "app_password": "not-a-real-password"}


def _slow_mailbox(seconds=2.0):
    import time as _time

    def fetch_unread(*args, **kw):
        _time.sleep(seconds)
        return [SimpleNamespace(sender="Canvas", subject="Lab 3 graded")]
    return fetch_unread


def test_a_slow_mailbox_does_not_block_the_arrival_cue(monkeypatch):
    """MEASURED, not argued: the step returns in well under the fetch it
    is waiting on, and the offer still gets spoken once the worker lands."""
    import time as _time

    import jarvis.tools.mail as mail_mod
    monkeypatch.setattr(mail_mod, "fetch_unread", _slow_mailbox(2.0))
    a = make_app({"gmail.accounts": [FAKE_ACCOUNT]}, quiet=Quiet([]),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    assert a._arrival_mail_is_remote() is True
    started = _time.monotonic()
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == ["catch-up"]
    blocked_for = _time.monotonic() - started
    assert blocked_for < 0.5, f"the pump thread was held for {blocked_for:.2f}s"
    assert a.tts.spoken == [], "the fetch was paid on the calling thread"
    a._arrival_catch_up_thread.join(timeout=10.0)
    assert not a._arrival_catch_up_thread.is_alive()
    assert "1 unread email" in a.tts.spoken[0]
    assert a.services.briefing_offer is not None


def test_with_no_mailbox_the_step_stays_inline_and_synchronous():
    """No account configured means fetch_unread raises before a socket is
    opened, so there is nothing to move and the cue stays end-to-end
    synchronous -- which is what every assertion on tts.spoken the line
    after run() depends on."""
    a = make_app(quiet=Quiet(["While you were out, sir:", "The build passed, sir."]))
    assert a._arrival_mail_is_remote() is False
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == ["catch-up"]
    assert a.tts.spoken and getattr(a, "_arrival_catch_up_thread", None) is None


def test_the_offer_is_switched_off_before_a_worker_is_ever_started():
    a = make_app({"presence.arrival_offer": False,
                  "gmail.accounts": [FAKE_ACCOUNT]})
    assert a._arrival_mail_is_remote() is False


# ==================================================================
# The 60 s window starts when he can answer, not when we decided to ask
# ==================================================================
def test_the_offer_ttl_is_stamped_after_the_digest_is_spoken():
    """BRIEFING_OFFER_TTL_S (60 s) is measured off `made_at`. The question
    is parked BEFORE the burst goes out so a TTS failure cannot leave it on
    the floor -- but a long quiet-hours backlog in front of it then ate
    most of the window he had to say yes."""
    import time as _time
    stamps = []
    a = make_app(quiet=Quiet(["While you were out, sir:", "The build passed, sir."]),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    a._unread_count = lambda: 3
    spoken = a.tts.speak

    def slow_speak(text):
        stamps.append(a.services.briefing_offer["made_at"])
        _time.sleep(0.2)                 # stand-in for the digest being read
        spoken(text)

    a.tts.speak = slow_speak
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert stamps, "the offer was not parked before the words went out"
    after = a.services.briefing_offer["made_at"]
    assert after - stamps[0] >= 0.2, "the TTL still ran during the digest"
    assert after <= _time.time()


def test_a_first_wake_offer_that_replaced_ours_keeps_its_own_clock():
    """Re-stamping must touch only the offer we parked; whoever owns the
    slot now owns its TTL."""
    a = make_app(services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    ours = {"made_at": 100.0, "deliver": lambda: True}
    theirs = {"made_at": 200.0, "deliver": lambda: True}
    a.services.briefing_offer = theirs
    a._restamp_offer(ours)
    assert theirs["made_at"] == 200.0 and ours["made_at"] == 100.0
    a._restamp_offer(None)
    assert a.services.briefing_offer is theirs


def test_a_door_room_that_names_no_configured_room_is_reported(caplog):
    """A door room that matches nothing is otherwise SILENT: the trigger
    just never fires and there is no error anywhere to find."""
    import logging
    a = make_app({"presence.door_room": "hallway",
                  "presence.room_sensor_enabled": True,
                  "presence.rooms": [{"name": "office", "url": "http://10.0.0.9"},
                                     {"name": "kitchen", "url": "http://10.0.0.8"}]})
    a._door = arrival_mod.DoorWatch(door="hallway")
    with caplog.at_level(logging.WARNING):
        a._warn_door_room_names_nothing()
    assert any("hallway" in r.getMessage() for r in caplog.records)


def test_a_box_with_no_rooms_list_is_not_warned_at_every_boot():
    """The live configuration today: the whole feature is inert by design,
    and a warning every boot would be noise."""
    import logging
    a = make_app({"presence.door_room": "hallway"})
    a._door = arrival_mod.DoorWatch(door="hallway")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    log = logging.getLogger("jarvis.app")
    log.addHandler(handler)
    try:
        a._warn_door_room_names_nothing()
    finally:
        log.removeHandler(handler)
    assert [r for r in records if r.levelno >= logging.WARNING] == []


def test_a_door_room_that_does_match_is_silent(caplog):
    import logging
    a = make_app({"presence.door_room": "Kitchen!",
                  "presence.room_sensor_enabled": True,
                  "presence.rooms": [{"name": "office", "url": "http://10.0.0.9"},
                                     {"name": "kitchen", "url": "http://10.0.0.8"}]})
    a._door = arrival_mod.DoorWatch(door="Kitchen!")
    with caplog.at_level(logging.WARNING):
        a._warn_door_room_names_nothing()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
