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
    """quiet.py's one seam the arrival cue uses, with the SAME contract as
    QuietPolicy.take_fragments: a drain the caller can undo. The arrival
    catch-up may decide not to speak long after it asked, and a stand-in
    that could only drain one-way would let a test claim the backlog was
    safe when the real policy's was being destroyed."""

    def __init__(self, frags=()):
        self.frags = list(frags)
        self.put_backs = 0

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
    assert a._arrival_offer_fragments() == ([], "")


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
    frags, major = a._arrival_offer_fragments()
    line = " ".join(frags)
    assert line == "The disk is full." and major == "The disk is full."
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
    assert a._arrival_offer_fragments() == ([], "")


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
# unread count is an IMAP round trip, and how long one of those can take is
# NOT MEASURED and has no bound this tree can quote: IMAP_TIMEOUT is the
# SOCKET timeout, so it bounds one blocking call and never bounded a fetch.
# That is exactly why it cannot be paid inline, where it froze the window
# and every event behind it at the moment he walked in.
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


# ==================================================================
# ...and it must not speak into whatever he is doing when it lands
# ==================================================================
# Taking the fetch off the pump thread bought the window back and sold the
# ATOMICITY of the last arrival step: the cue returns, and the offer is
# spoken and parked whenever the mailbox happens to answer -- however long
# that is, which nothing here measures or bounds. Inline that could not
# happen. A catch-up that lands after he has asked Jarvis something else is
# not late, it is WRONG, and silence is the right outcome for a stale
# digest: nothing here is news that keeps. How late is TOO late is the one
# part that is bounded, and by a check rather than a claim -- see the
# lateness tests at the end of this file.
def _gated_mailbox(gate, seconds=10.0):
    def fetch_unread(*args, **kw):
        gate.wait(seconds)
        return [SimpleNamespace(sender="Canvas", subject="Lab 3 graded")]
    return fetch_unread


def _catching_up(monkeypatch, gate, held=()):
    import jarvis.tools.mail as mail_mod
    monkeypatch.setattr(mail_mod, "fetch_unread", _gated_mailbox(gate))
    a = make_app({"gmail.accounts": [FAKE_ACCOUNT]}, quiet=Quiet(held),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == ["catch-up"]
    return a


def _finish(a, gate):
    gate.set()
    a._arrival_catch_up_thread.join(timeout=10.0)
    assert not a._arrival_catch_up_thread.is_alive()


def _said_nothing(a):
    assert a.tts.spoken == [], "the stale catch-up spoke anyway"
    assert a.services.briefing_offer is None, "a stale question was parked"
    assert a._followup_after_speech is False, "a mic was opened for it"


def test_a_catch_up_that_lands_after_he_has_TAKEN_A_TURN_is_dropped(monkeypatch):
    """He walked in, the mailbox took its time, and he asked Jarvis
    something in the meantime. The answer to THAT is the floor; the
    doorstep question arriving on top of it is the failure."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate)
    a._dispatch_gen = getattr(a, "_dispatch_gen", 0) + 1   # one turn, as _dispatch does
    _finish(a, gate)
    _said_nothing(a)


def test_a_catch_up_that_lands_during_a_LIVE_turn_is_dropped(monkeypatch):
    """The floor is busy right now: _turn_busy is exactly the guard
    _after_speech makes before it puts anything of its own out."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate)
    a._turn_busy = _threading.Event()
    a._turn_busy.set()
    _finish(a, gate)
    _said_nothing(a)


def test_a_catch_up_from_a_SUPERSEDED_arrival_is_dropped(monkeypatch):
    """He came back, left again and came back again inside one slow fetch.
    The first cue's digest belongs to an arrival that is over."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate)
    a._arrival_gen = getattr(a, "_arrival_gen", 0) + 1
    _finish(a, gate)
    _said_nothing(a)


def test_a_catch_up_that_lands_on_a_QUIET_house_still_speaks(monkeypatch):
    """The control: nothing changed while the mailbox answered, so the
    offer is spoken and parked exactly as it always was. A guard that
    drops the ordinary case is worse than the bug."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate)
    _finish(a, gate)
    assert "1 unread email" in a.tts.spoken[0]
    assert a.services.briefing_offer is not None
    assert a._followup_after_speech is True


def test_the_follow_up_mic_is_armed_only_once_the_question_is_QUEUED():
    """`_park_arrival_offer` set `_followup_after_speech` before `_say`
    had queued a word. On the worker that is a real gap: the welcome's own
    falling edge can drain on the Tk thread inside it, and `_after_speech`
    then sees the flag with `tts.pending == 0` and opens the mic before the
    question has been asked."""
    a = make_app(quiet=Quiet([]), services=SimpleNamespace(panel_wake=None,
                                                           briefing_offer=None))
    a._unread_count = lambda: 3
    seen, spoken = [], a.tts.speak

    def watch(text):
        seen.append(a._followup_after_speech)
        spoken(text)

    a.tts.speak = watch
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert seen == [False], "the mic was armed before the question was queued"
    assert a._followup_after_speech is True


# ==================================================================
# The cue's ledger says what actually happened
# ==================================================================
def test_the_ledger_says_STARTED_when_the_catch_up_went_to_a_worker(monkeypatch):
    """`arrival.run`'s contract is "returns those that actually ran", and
    on the worker path the step has only STARTED -- it may yet turn out to
    have nothing to say, and log so. Two log lines contradicting each
    other is worse than either."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate)
    assert a._arrival_ledger(["greeting", "catch-up"]) == \
        "greeting -> catch-up (started)"
    _finish(a, gate)


def test_the_ledger_says_plain_catch_up_when_it_ran_INLINE():
    a = make_app(quiet=Quiet(["While you were out, sir:", "The build passed, sir."]))
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert a._arrival_ledger(["greeting", "catch-up"]) == "greeting -> catch-up"
    assert a._arrival_ledger([]) == ""


def test_the_deferred_mark_does_not_survive_into_the_NEXT_cue(monkeypatch):
    """One cue's worker must not label the next cue's inline step."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate)
    _finish(a, gate)
    a.assistant.data.pop("gmail.accounts")
    a.quiet = Quiet(["While you were out, sir:"])
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert a._arrival_ledger(["catch-up"]) == "catch-up"


# ==================================================================
# A DROPPED CATCH-UP MUST NOT COST HIM THE BACKLOG
# ==================================================================
# The repair above was right to drop a stale digest and wrong in how it
# did it: the held lines were drained atomically on the pump thread BEFORE
# the worker started, so when the worker decided the catch-up was stale
# those lines were GONE. They are the things he missed while he was out
# and there is no second copy of them anywhere.
#
# Two rules hold it now and these tests are the whole class, not the one
# reported case: nothing is taken until the guard has passed, and every
# path out of speak_catch_up that did not queue the words puts it back.
HELD = ["While you were out, sir:", "The build passed, sir."]


def _boom(*a, **kw):
    raise RuntimeError("the floor probe exploded")


def _turn_taken(a):
    a._dispatch_gen = getattr(a, "_dispatch_gen", 0) + 1


def _turn_open(a):
    import threading as _threading
    a._turn_busy = _threading.Event()
    a._turn_busy.set()


def _transcribing(a):
    import threading as _threading
    a._audio_busy = _threading.Event()
    a._audio_busy.set()


def _mic_open(a):
    a.recorder = SimpleNamespace(recording=True)


def _newer_arrival(a):
    a._arrival_gen = getattr(a, "_arrival_gen", 0) + 1


def _guard_itself_broken(a):
    # A GUARD THAT FAILS OPEN IS NOT A GUARD. Every read in
    # _arrival_catch_up_stale used to sit under one bare `except` that fell
    # through to "" -- and "" is PERMISSION to speak. So: break a read and
    # the question does not get asked.
    a._turn_busy = SimpleNamespace(is_set=_boom)


DROPS = [_turn_taken, _turn_open, _transcribing, _mic_open, _newer_arrival,
         _guard_itself_broken]


@pytest.mark.parametrize("move", DROPS, ids=lambda f: f.__name__)
def test_a_dropped_catch_up_leaves_the_QUIET_BACKLOG_INTACT(monkeypatch, move):
    """Every way the digest can be dropped, against a real backlog."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate, held=HELD)
    move(a)
    _finish(a, gate)
    _said_nothing(a)
    assert a.quiet.frags == HELD, "the dropped catch-up destroyed the backlog"


@pytest.mark.parametrize("move", DROPS, ids=lambda f: f.__name__)
def test_a_dropped_catch_up_does_not_even_TAKE_the_backlog(monkeypatch, move):
    """Not "took it and gave it back" -- never taken. The guard is asked
    before anything that cannot be undone, so the ordinary drop leaves the
    policy's own falling edge to read the lines out on its next tick."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate, held=HELD)
    move(a)
    _finish(a, gate)
    assert a.quiet.frags == HELD and a.quiet.put_backs == 0, \
        "it was taken and handed back, not left alone"


@pytest.mark.parametrize("move", DROPS, ids=lambda f: f.__name__)
def test_the_backlog_a_dropped_catch_up_left_is_STILL_THERE_NEXT_TIME(
        monkeypatch, move):
    """The point of keeping it: the next cue reads it out."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate, held=HELD)
    move(a)
    _finish(a, gate)
    # ...and he walks in again, this time with nothing in the way.
    a.assistant.data.pop("gmail.accounts")
    a._dispatch_gen = getattr(a, "_dispatch_gen", 0)
    a._turn_busy = a._audio_busy = None
    a.recorder = None
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert a.tts.spoken and "The build passed" in a.tts.spoken[-1]
    assert a.quiet.frags == []


def test_a_catch_up_that_DOES_speak_consumes_the_backlog_exactly_once(monkeypatch):
    """The control, and the other half of the class: a digest that WAS
    spoken must not be put back, or he hears it twice."""
    import threading as _threading
    gate = _threading.Event()
    a = _catching_up(monkeypatch, gate, held=HELD)
    _finish(a, gate)
    assert "The build passed" in a.tts.spoken[0]
    assert "1 unread email" in a.tts.spoken[0]
    assert a.quiet.frags == [] and a.quiet.put_backs == 0


def test_a_digest_that_THINS_AWAY_TO_NOTHING_puts_the_backlog_back():
    """The exit nobody was looking at: taken, and then there was nothing
    left to say. It returns "" like an empty backlog does, and "" must not
    mean the lines were spent."""
    a = make_app(quiet=Quiet(HELD),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    a._unread_count = lambda: 3
    a._thin_address = lambda frags: ["" for _ in frags]
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == []
    assert a.tts.spoken == [] and a.quiet.frags == HELD
    assert a.quiet.put_backs == 1


def test_a_TTS_FAILURE_puts_the_backlog_back_too():
    """A digest that could not be spoken is worth repeating; one that was
    destroyed is not recoverable."""
    a = make_app(quiet=Quiet(HELD),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    a._unread_count = lambda: 3
    a.tts.speak = _boom
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == []
    assert a.quiet.frags == HELD and a.quiet.put_backs == 1


def test_a_ONE_WAY_quiet_stand_in_is_declared_rather_than_losing_lines(caplog):
    """A policy that has only the one-way drain cannot give anything back.
    That is a degraded guarantee, and the whole bug above was a degraded
    guarantee nobody said out loud."""
    import logging

    class OneWay:
        def __init__(self):
            self.frags = list(HELD)

        def release_fragments(self):
            frags, self.frags = self.frags, []
            return frags

    a = make_app(quiet=OneWay(),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    with caplog.at_level(logging.WARNING):
        assert a._take_held_fragments()[0] == HELD
    assert any("OneWay" in r.getMessage() and r.levelno >= logging.WARNING
               for r in caplog.records)


# ==================================================================
# The park and the follow-up mic are one invariant
# ==================================================================
def test_a_TTS_FAILURE_still_leaves_a_WINDOW_to_answer_the_parked_question():
    """Parking before the words is deliberate: a TTS failure must not leave
    a question on the floor with nothing listening for the answer. Arming
    the mic after `_say` -- which is right, the welcome's own falling edge
    could otherwise open it early -- put the arm on the far side of the one
    call that can fail, so a TTS failure produced exactly the outcome the
    park exists to prevent. The card is already published; he can read the
    question, and now he can answer it."""
    a = make_app(quiet=Quiet([]),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    a._unread_count = lambda: 3
    a.tts.speak = _boom
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert a.services.briefing_offer is not None, "the question was not parked"
    assert a._followup_after_speech is True, "parked with no window to answer in"


def test_nothing_is_armed_when_nothing_was_PARKED_even_on_a_TTS_failure():
    """A fault-only line is a statement; there is no question, so there is
    no window -- the `finally` must not arm one anyway."""
    board = SimpleNamespace(current=SimpleNamespace(kind="error",
                                                    line="The disk is full."))
    a = make_app(quiet=Quiet([]),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None,
                                          faults=board))
    a._unread_count = lambda: 0
    a.tts.speak = _boom
    arrival_mod.run(["catch-up"], a._arrival_actions())
    assert a.services.briefing_offer is None
    assert a._followup_after_speech is False


# ==================================================================
# How late the question may be is a CHECK, not a claim
# ==================================================================
# "IMAP_TIMEOUT: 15 s a mailbox" was quoted as the worst case three times
# in this branch and was never one -- it is the SOCKET timeout, so it
# bounds one blocking call, not a fetch and not the step. The bound is now
# enforced where it can be true by construction.
def test_a_catch_up_that_overruns_the_LATENESS_BOUND_is_dropped(monkeypatch):
    """MEASURED against the clock, not argued: the bound is set below the
    fetch and the digest does not get spoken."""
    import jarvis.tools.mail as mail_mod
    monkeypatch.setattr(app_mod, "ARRIVAL_CATCH_UP_LATENESS_S", 0.2)
    monkeypatch.setattr(mail_mod, "fetch_unread", _slow_mailbox(0.5))
    a = make_app({"gmail.accounts": [FAKE_ACCOUNT]}, quiet=Quiet(HELD),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == ["catch-up"]
    a._arrival_catch_up_thread.join(timeout=10.0)
    assert not a._arrival_catch_up_thread.is_alive()
    _said_nothing(a)
    assert a.quiet.frags == HELD


def test_a_catch_up_INSIDE_the_lateness_bound_still_speaks(monkeypatch):
    """The control. A guard that drops the ordinary case is worse than the
    bug it was written for."""
    import jarvis.tools.mail as mail_mod
    monkeypatch.setattr(app_mod, "ARRIVAL_CATCH_UP_LATENESS_S", 5.0)
    monkeypatch.setattr(mail_mod, "fetch_unread", _slow_mailbox(0.05))
    a = make_app({"gmail.accounts": [FAKE_ACCOUNT]}, quiet=Quiet(HELD),
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    arrival_mod.run(["catch-up"], a._arrival_actions())
    a._arrival_catch_up_thread.join(timeout=10.0)
    assert "The build passed" in a.tts.spoken[0]
    assert a.services.briefing_offer is not None


# ==================================================================
# The guard fails CLOSED
# ==================================================================
def test_the_guard_says_DROP_when_it_cannot_tell():
    """"" is permission. A guard whose own breakage grants the thing it
    exists to withhold is the wrong way round, and there is now exactly one
    `return ""` in it -- the last statement of the try."""
    import time as _time
    a = make_app()
    a._arrival_gen = a._dispatch_gen = 0
    a._turn_busy = SimpleNamespace(is_set=_boom)
    assert a._arrival_catch_up_stale(0, 0, _time.monotonic()) != ""


def test_the_guard_says_SPEAK_when_nothing_has_moved():
    import time as _time
    a = make_app()
    a._arrival_gen = a._dispatch_gen = 0
    assert a._arrival_catch_up_stale(0, 0, _time.monotonic()) == ""


def test_the_guard_drops_anything_older_than_the_bound():
    import time as _time
    a = make_app()
    a._arrival_gen = a._dispatch_gen = 0
    started = _time.monotonic() - (app_mod.ARRIVAL_CATCH_UP_LATENESS_S + 1.0)
    why = a._arrival_catch_up_stale(0, 0, started)
    assert "late" in why


# ==================================================================
# The same class, against the REAL QuietPolicy
# ==================================================================
# Everything above vets the drop against the `Quiet` stand-in at the top of
# this file -- and that stand-in was rewritten by the same change as the
# fix. On its own it proves the two AGREE, not that his backlog survives.
# These wire jarvis.quiet.QuietPolicy itself: its own deque, its own
# take_fragments, its own put_back.
def _real_quiet(lines=HELD):
    from jarvis.quiet import QuietPolicy
    p = QuietPolicy(Cfg())
    p.set_dnd(600)                      # he is out; the lines are held
    for line in lines:
        assert p.hold(line, "message") is True
    return p


def _texts(policy):
    return [text for _, text, _ in policy.held]


def _catching_up_for_real(monkeypatch, gate, policy):
    """As `_catching_up`, but the REAL policy is wired BEFORE the cue runs
    -- which is the whole point. Attached afterwards it would never be the
    thing the pump thread reached for, and the old one-way drain would sail
    past a test that looked like it was watching it."""
    import jarvis.tools.mail as mail_mod
    monkeypatch.setattr(mail_mod, "fetch_unread", _gated_mailbox(gate))
    a = make_app({"gmail.accounts": [FAKE_ACCOUNT]}, quiet=policy,
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == ["catch-up"]
    return a


@pytest.mark.parametrize("move", DROPS, ids=lambda f: f.__name__)
def test_a_dropped_catch_up_leaves_a_REAL_policys_backlog_intact(monkeypatch,
                                                                 move):
    import threading as _threading
    gate = _threading.Event()
    p = _real_quiet()
    a = _catching_up_for_real(monkeypatch, gate, p)
    move(a)
    _finish(a, gate)
    _said_nothing(a)
    assert _texts(p) == list(HELD), "the dropped catch-up destroyed the backlog"


def test_the_REAL_backlog_a_dropped_catch_up_left_is_read_out_NEXT_TIME(
        monkeypatch):
    """Kept is worth nothing if nothing ever says it."""
    import threading as _threading
    gate = _threading.Event()
    p = _real_quiet()
    a = _catching_up_for_real(monkeypatch, gate, p)
    a._dispatch_gen = getattr(a, "_dispatch_gen", 0) + 1       # he took a turn
    _finish(a, gate)
    _said_nothing(a)
    # ...and he walks in again, with no mailbox in the way (inline, atomic).
    a.assistant.data.pop("gmail.accounts")
    a._dispatch_gen = getattr(a, "_dispatch_gen", 0)
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == ["catch-up"]
    assert a.tts.spoken and "The build passed" in a.tts.spoken[-1]
    assert p.held == []


def test_a_TTS_FAILURE_gives_a_REAL_policy_its_lines_back_and_they_are_SPOKEN():
    """The put_back path itself, end to end: taken from the real deque,
    handed back to it, and read out by the policy's OWN next tick -- which
    only happens because put_back re-arms the falling edge the take
    consumed."""
    said = []
    p = _real_quiet()
    p._say = said.append
    a = make_app(quiet=p,
                 services=SimpleNamespace(panel_wake=None, briefing_offer=None))
    a._unread_count = lambda: 3
    a.tts.speak = _boom
    assert arrival_mod.run(["catch-up"], a._arrival_actions()) == []
    assert a.tts.spoken == [] and _texts(p) == list(HELD)
    # The take consumed the falling edge those lines belonged to; put_back
    # re-armed it, so the policy's own next tick reads them out.
    p._set("quiet.dnd_until", 0)               # the window is over
    text = p.tick()
    assert "The build passed" in text and said == [text]


# ==================================================================
# A NAP IS NOT A RETURN (_on_presence)
# ==================================================================
def test_a_ten_second_round_trip_neither_greets_nor_burns_the_power_up():
    """His measured failure, 2026-09-11: away at 11:32:26, home(returned)
    at 11:32:36. Ten seconds is one presence.poll_s_away; his phone
    answered the next poll. Nothing left and nothing came back."""
    a = make_app()
    a._departure_timer = SimpleNamespace(cancel=lambda: None)
    swept, greeted = [], []
    a._maybe_power_up = lambda why: swept.append(why)
    a._greet_return = lambda src: greeted.append(src)
    a._on_presence(SimpleNamespace(home=True, returned=True, since=2_000.0))
    assert greeted == [], "it welcomed a man who had not moved"
    assert swept == [], "a napping radio burned the once-a-day sweep"


def test_a_return_after_the_confirm_closed_still_greets_and_sweeps():
    a = make_app()
    a._departure_timer = None
    swept, greeted = [], []
    a._maybe_power_up = lambda why: swept.append(why)
    a._greet_return = lambda src: greeted.append(src)
    a._on_presence(SimpleNamespace(home=True, returned=True, since=2_000.0))
    assert greeted == ["phone"] and swept == ["presence"]
