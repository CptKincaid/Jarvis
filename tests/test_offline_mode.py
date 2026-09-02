"""The spoken half of offline mode (jarvis/commander.py).

His words for what he wanted to be able to say: *"a voice command that
says offline mode or deactivate presence or something of that nature"* --
so this is a FAMILY, tested the way the house tests families: a phrase
table for the positives, an explicit negative table (he says a great many
sentences containing "off" and "stop"), and handler tests that assert the
SPOKEN line states what actually happened.
"""
import types

import pytest

import jarvis.commander as C
from jarvis import sensing


# ------------------------------------------------------------------ fixtures
class FakeCfg:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, dotted, default=None):
        value = self.data.get(dotted, default)
        return default if value is None else value

    def set(self, dotted, value):
        self.data[dotted] = value
        return True


def _clock(hour: int, minute: int = 0):
    import datetime as _dt
    return _dt.datetime(2026, 9, 2, hour, minute).timestamp()


@pytest.fixture
def policy(tmp_path):
    p = sensing.SensingPolicy(cfg=FakeCfg(), path=tmp_path / "sensing.json",
                              now=lambda: _clock(12))
    p.enable()
    return p


def _commander(policy=None, **kw):
    c = object.__new__(C.Commander)
    c.services = types.SimpleNamespace(sensing=policy, assistant=None,
                                       memory=None, timekeeper=None, **kw)
    return c


def _run(cmd_name, commander, text):
    """Dispatch `text` through the ONE registry entry named `cmd_name`."""
    cmd = next(c for c in C.REGISTRY if c.name == cmd_name)
    m = cmd.matcher(text)
    assert m, f"{cmd_name} did not match {text!r}"
    return cmd.handler(commander, text, m)


# ------------------------------------------------------------------- the OFF
OFF_PHRASES = [
    "offline mode",                       # his word, bare
    "jarvis, offline mode",
    "go offline",
    "offline mode on",
    "enable offline mode",
    "switch to offline mode",
    "engage privacy mode",
    "privacy mode",
    "privacy mode on",
    "deactivate presence",                # his word, verbatim
    "disable presence",
    "deactivate the presence sensor",
    "turn off the sensors",               # his word, verbatim
    "turn off the camera",
    "shut down the camera",
    "kill the radar",
    "turn the camera off",
    "switch the sensors off",
    "stop watching",                      # his word, verbatim
    "stop watching me",
    "stop watching the room",
    "stop looking at me",
    "don't watch me",
    "close your eyes",
    "look away",
    "no more cameras",
    "cameras off",
    "sensors off",
    # 2026-09-02 review: natural neighbours of his own phrasings that fell
    # through to the router and did nothing. A privacy order must never
    # reach a model that cannot switch a sensor, so a leading politeness
    # word is not allowed to cost him the switch.
    "please stop watching",
    "can you stop watching",
    "could you turn off the camera",
    "turn off the camera and the radar",
    "cameras down",
    "sensors down",
    "go dark",
    "no sensors tonight",
    "no cameras today",
]

# Every one of these is something he plausibly says, and every one of them
# would be a serious bug: the camera going off because he asked for silence,
# or the lights, or the end of a timer.
NOT_OFF_PHRASES = [
    "stop",
    "stop it",
    "stop talking",
    "stop reading",
    "stop the timer",
    "stop the room tone",
    "stop watching netflix",
    "stop the music",
    "turn off the lights",
    "turn off the music",
    "turn off quiet hours",
    "turn off do not disturb",
    "turn off the room tone",
    "turn the lights off",
    "power down the workshop",
    "lights out",
    "no more asides",
    "shut up",
    "never mind",
    "cancel that",
    "close the board",
    "end the session",
    "turn off my morning briefing",
    "what's off camera in that photo",
    "turn off the screen",
    # ...and the blast radius of the leading-politeness and "and" widenings
    "please stop",
    "please stop the timer",
    "can you stop the music",
    "can you turn off the lights",
    "could you turn the lights down",
    "please turn off do not disturb",
    "turn off the lights and the music",
    "go dark mode",
    "screen down",
    "music off",
    "tonight",
    "today",
]


@pytest.mark.parametrize("phrase", OFF_PHRASES)
def test_the_offline_family_matches_his_phrasings(phrase):
    assert C._SENSING_OFF_RX.match(phrase), phrase


@pytest.mark.parametrize("phrase", NOT_OFF_PHRASES)
def test_the_offline_family_leaves_everything_else_alone(phrase):
    assert not C._SENSING_OFF_RX.match(phrase), phrase
    assert not C._SENSING_ON_RX.match(phrase), phrase


# -------------------------------------------------------------------- the ON
ON_PHRASES = [
    "come back online",
    "go online",
    "back online",
    "online mode",
    "exit offline mode",
    "cancel offline mode",
    "turn off offline mode",
    "offline mode off",
    "leave privacy mode",
    "privacy mode off",
    "reactivate presence",
    "enable presence",
    "turn the sensors back on",
    "turn on the camera",
    "switch the camera back on",
    "start watching again",
    "start watching the room",
    "open your eyes",
    "sensors back on",
    "resume sensing",
    "start sensing",
    "wake the sensors",
    "wake up the camera",
    "you can watch again",
    "please turn the camera back on",
]

NOT_ON_PHRASES = [
    "turn on the lights",
    "turn the music back on",
    "start watching the game",
    "wake up the workshop",
    "wake me at seven",
    "resume the timer",
    "resume the music",
    "start a focus session",
    "start the timer",
    "start the music",
    "turn on my morning briefing",
    "turn on the lights and the fan",
]


@pytest.mark.parametrize("phrase", ON_PHRASES)
def test_the_online_family_matches(phrase):
    assert C._SENSING_ON_RX.match(phrase), phrase


@pytest.mark.parametrize("phrase", NOT_ON_PHRASES)
def test_the_online_family_leaves_everything_else_alone(phrase):
    assert not C._SENSING_ON_RX.match(phrase), phrase
    assert not C._SENSING_OFF_RX.match(phrase), phrase


# ---------------------------------------------------------------- the STATUS
STATUS_PHRASES = [
    "are you watching",
    "are you watching me",
    "are you offline",
    "are you in privacy mode",
    "is offline mode on",
    "is the camera on",
    "are the sensors off",
    "sensing status",
    "sensor status",
    "what's the camera status",
    "when is the camera curfew",
]


@pytest.mark.parametrize("phrase", STATUS_PHRASES)
def test_the_status_question_matches_and_never_switches_anything(phrase):
    assert C._SENSING_STATUS_RX.match(phrase), phrase
    assert not C._SENSING_OFF_RX.match(phrase), phrase
    assert not C._SENSING_ON_RX.match(phrase), phrase


def test_asking_does_not_change_the_state(policy):
    c = _commander(policy)
    res = _run("sensing status", c, "are you watching")
    assert res.handled and res.speak
    assert policy.allowed(sensing.CAMERA) is True


# --------------------------------------------------------- the spoken switch
def test_going_offline_says_what_actually_stopped(policy):
    stopped = []
    policy.attach("camera", lambda: stopped.append("cam") or True)
    policy.attach("radar", lambda: True)
    res = _run("sensing off", _commander(policy), "offline mode")
    assert res.handled and res.speak
    assert policy.allowed(sensing.CAMERA) is False
    low = res.reply.lower()
    assert "offline" in low
    assert "camera" in low and "radar" in low
    assert "microphone" in low, "the mic staying live is the point of the switch"


def test_going_offline_names_a_device_it_could_not_stop(policy):
    policy.attach("radar", lambda: False)
    res = _run("sensing off", _commander(policy), "stop watching")
    low = res.reply.lower()
    assert "couldn't stop the radar" in low or "could not stop the radar" in low
    assert "still" in low, "he has to be told it may still be running"


def test_going_offline_never_claims_a_camera_that_is_not_there(policy):
    """There is no camera on this box yet. Saying "the camera is down"
    would be the exact promise he ruled out."""
    policy.attach("camera", lambda: True, present=lambda: False)
    policy.attach("radar", lambda: True)
    res = _run("sensing off", _commander(policy), "offline mode")
    assert "camera" not in res.reply.lower()
    assert "radar" in res.reply.lower()


def test_going_offline_with_nothing_attached_says_so(policy):
    res = _run("sensing off", _commander(policy), "offline mode")
    assert res.handled
    assert "nothing" in res.reply.lower()


def _wire_radar(policy, power_url="", posts=None):
    """A real RoomSensor attached to `policy`, with or without the optional
    ESPHome power switch."""
    from jarvis import roomsensor
    return roomsensor.RoomSensor(
        "http://10.0.0.9/binary_sensor/presence",
        get=lambda url, timeout: '{"value": true}', policy=policy,
        power_url=power_url,
        post=lambda url, timeout: (posts if posts is not None else []).append(url))


def test_the_line_tells_a_radar_that_is_down_from_one_that_is_merely_unpolled(
        policy, tmp_path):
    """He has NOT flashed the ESP32 and has not run a MOSFET, so the
    unpowered case is the ONLY one that exists on his box today -- and it
    used to produce the byte-identical "The radar is down." Stopping the
    polling is not stopping the radar: the LD2410 keeps radiating and keeps
    serving presence to anyone on the LAN.
    """
    _wire_radar(policy)                      # no power switch: the live case
    unpolled = _run("sensing off", _commander(policy), "offline mode").reply

    posts = []
    wired = sensing.SensingPolicy(cfg=FakeCfg(), path=tmp_path / "s2.json",
                                  now=lambda: _clock(12))
    wired.enable()
    _wire_radar(wired, "http://10.0.0.9/switch/radar_power", posts)
    powered = _run("sensing off", _commander(wired), "offline mode").reply

    assert posts == ["http://10.0.0.9/switch/radar_power/turn_off"]
    assert unpolled != powered, "the two configurations must not sound alike"
    assert "the radar is down" in powered.lower()
    assert "the radar is down" not in unpolled.lower()
    assert "power isn't switched" in unpolled.lower()
    assert "still sensing the room" in unpolled.lower()
    assert "nothing was sensing" not in unpolled.lower()


def test_a_bound_that_was_dropped_is_not_spoken_as_one(policy):
    """disable() refuses an `until` that is not in the future (the next
    state() read would expire it), so the head has to follow the OUTCOME
    and not the request -- otherwise "offline until 11 am" is said over a
    switch with no end at all."""
    out = policy.disable(until=_clock(12) - 5)
    line = C._sensing_off_line(out, when_text="11 am")
    assert "until" not in line.lower()
    assert line.startswith("Offline, sir.")


def test_switching_the_curfew_off_reports_a_refused_write(policy, tmp_path):
    """Same honesty rule as the set-window branch twenty lines below it:
    set_curfew returns False when the config did not take the write."""
    class Refusing(FakeCfg):
        def set(self, dotted, value):
            return False

    p = sensing.SensingPolicy(cfg=Refusing(), path=tmp_path / "s.json",
                              now=lambda: _clock(12))
    p.enable()
    res = _run("sensing curfew", _commander(p), "turn off the camera curfew")
    low = res.reply.lower()
    assert "couldn't" in low or "could not" in low


def test_a_switch_that_could_not_be_saved_is_spoken(tmp_path):
    """A write that failed means the next start reads the OLD file, i.e.
    online. He has to hear that, not a bare "offline, sir"."""
    home = tmp_path / "state"
    p = sensing.SensingPolicy(cfg=FakeCfg(), path=home / "s.json",
                              now=lambda: _clock(12))
    p.enable()                       # creates the directory
    home.chmod(0o500)                # ...and now nothing new may be written
    try:
        res = _run("sensing off", _commander(p), "offline mode")
        assert "restart" in res.reply.lower()
        assert p.allowed(sensing.CAMERA) is False
    finally:
        home.chmod(0o700)


def test_coming_back_online_is_spoken_with_the_curfew_it_lands_in(tmp_path):
    p = sensing.SensingPolicy(cfg=FakeCfg(), path=tmp_path / "s.json",
                              now=lambda: _clock(22))
    p.disable()
    res = _run("sensing on", _commander(p), "come back online")
    low = res.reply.lower()
    assert "curfew" in low and "camera" in low
    assert p.allowed(sensing.RADAR) is True
    assert p.allowed(sensing.CAMERA) is False, "the curfew is not overridden"


def test_the_handler_says_so_when_there_is_no_policy_wired():
    """Not a silent fall-through to the model: a privacy command handed to
    a language model is a privacy command that did nothing."""
    res = _run("sensing off", _commander(None), "offline mode")
    assert res.handled and res.speak
    assert "can't" in res.reply.lower() or "cannot" in res.reply.lower()


# ------------------------------------------------------- the timed offline
HOLD_PHRASES = {
    "keep the camera off until noon": "until",
    "keep the cameras off for two hours": "for",
    "no cameras for the next two hours": "for",
    "no cameras until seven": "until",
    "turn the camera off for an hour": "for",
    "go offline for thirty minutes": "for",
    "offline mode until ten": "until",
    "leave the sensors off until six": "until",
    # the bounded forms of phrasings whose BARE form already worked -- the
    # gap the review found, and the one where falling through to the router
    # is worst: he asked for a window and got nothing at all.
    "stop watching for ten minutes": "for",
    "stop watching me for an hour": "for",
    "camera off for an hour": "for",
    "sensors off for the night": "for",
    "offline until morning": "until",
    "please stop watching for ten minutes": "for",
}


@pytest.mark.parametrize("phrase,mode", sorted(HOLD_PHRASES.items()))
def test_the_timed_offline_matches_and_captures_its_when(phrase, mode):
    m = C._SENSING_HOLD_RX.match(phrase)
    assert m, phrase
    assert C._sensing_hold_mode(m) == mode
    assert C._sensing_hold_when(m).strip(), phrase


def test_a_timed_offline_sets_an_end_and_says_it(policy):
    res = _run("sensing hold", _commander(policy), "no cameras for the next two hours")
    assert policy.allowed(sensing.CAMERA) is False
    st = policy.state()
    assert st.until is not None and st.until > _clock(12)
    assert "until" in res.reply.lower()


def test_a_timed_offline_whose_when_is_gibberish_goes_off_anyway_and_says_so(policy):
    """Failing toward MORE privacy, out loud. Guessing the time would
    quietly reopen the lens early; refusing would leave it open now."""
    res = _run("sensing hold", _commander(policy),
               "keep the camera off until the thing on tuesday")
    assert policy.allowed(sensing.CAMERA) is False
    assert policy.state().until is None, "an unparsed end must not be invented"
    low = res.reply.lower()
    assert "didn't catch" in low or "did not catch" in low


# ------------------------------------------------------------- the schedule
CURFEW_PHRASES = [
    "camera curfew from nine to seven",
    "set the camera curfew from ten to six",
    "change the curfew to nine to seven",
    "no cameras from ten at night to six in the morning",
    "start the camera curfew at ten tonight",
    "extend the camera curfew until noon",
    "push the curfew to nine",
    "turn off the camera curfew",
    "cancel the camera curfew",
]


@pytest.mark.parametrize("phrase", CURFEW_PHRASES)
def test_the_curfew_family_matches(phrase):
    assert C._SENSING_CURFEW_RX.match(phrase), phrase


def test_setting_the_window_by_voice_writes_both_ends(policy):
    res = _run("sensing curfew", _commander(policy),
               "camera curfew from ten to six")
    assert policy.curfew() == ((22, 0), (6, 0))
    assert "10 pm" in res.reply and "6 am" in res.reply


def test_moving_only_the_start_keeps_the_end(policy):
    """"Start the camera curfew at ten tonight" is one edge, not a window."""
    _run("sensing curfew", _commander(policy), "camera curfew from nine to seven")
    _run("sensing curfew", _commander(policy),
         "start the camera curfew at ten tonight")
    assert policy.curfew() == ((22, 0), (7, 0))


def test_extending_only_the_end_keeps_the_start(policy):
    _run("sensing curfew", _commander(policy), "camera curfew from nine to seven")
    _run("sensing curfew", _commander(policy), "extend the camera curfew until noon")
    assert policy.curfew() == ((21, 0), (12, 0))


def test_the_curfew_can_be_switched_off_by_voice(policy):
    res = _run("sensing curfew", _commander(policy), "turn off the camera curfew")
    assert policy.curfew() is None
    assert "off" in res.reply.lower()


def test_an_unparseable_curfew_time_changes_nothing_and_says_so(policy):
    before = policy.curfew()
    res = _run("sensing curfew", _commander(policy),
               "start the camera curfew at whenever")
    assert policy.curfew() == before
    assert res.handled and res.speak


# ------------------------------------------------------------------ wiring
def test_the_family_is_in_the_registry_and_in_tier_one():
    names = {c.name for c in C.REGISTRY}
    tier1 = {c.name for c in C.ASSISTANT_TIER1}
    for name in ("sensing off", "sensing on", "sensing status",
                 "sensing hold", "sensing curfew"):
        assert name in names, name
        # The hotword eats the wake word, so a privacy command arrives with
        # no "jarvis" left to strip; without Tier 1 membership it would
        # reach the router and be answered by a model that cannot switch
        # anything off.
        assert name in tier1, name


def test_the_privacy_family_outranks_every_other_matcher():
    """"stop watching" and "turn off the sensors" must reach this handler
    even if some future entry starts claiming "stop"/"turn off" -- so the
    family sits at the very top of the registry."""
    order = [c.name for c in C.REGISTRY]
    first = [n for n in order[:5]]
    for name in ("sensing status", "sensing curfew", "sensing hold",
                 "sensing on", "sensing off"):
        assert name in first, (name, first)


def test_nothing_in_the_family_can_reach_the_microphone():
    """Belt and braces on his one explicit exception: no handler here may
    touch the recorder, the hotword or CONFIG.hotword."""
    import inspect
    for fn in (C._h_sensing_off, C._h_sensing_on, C._h_sensing_status,
               C._h_sensing_hold, C._h_sensing_curfew):
        src = inspect.getsource(fn)
        for forbidden in ("hotword", "recorder", "mic", "listen"):
            assert forbidden not in src, (fn.__name__, forbidden)
