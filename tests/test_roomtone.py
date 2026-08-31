"""Room tone (jarvis/roomtone.py).

Mic safety is the design, so most of this file is about when the bed is
SILENT: off by default, off during quiet, off the instant the microphone
opens, and off again if a hold is ever dropped on the floor.
"""
import wave

import pytest

from jarvis import roomtone
from jarvis.roomtone import BEDS, PHASES, RoomTone, bake, bake_all, desired_phase


class FakeCfg:
    def __init__(self, data=None):
        self._data = data or {}

    def get(self, key, default=None):
        node = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def set(self, key, value):
        node = self._data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


class FakeProc:
    def __init__(self):
        self.written = bytearray()
        self.killed = False
        self.stdin = self
        self._closed = False

    # stdin surface
    def write(self, data):
        if self._closed:
            raise BrokenPipeError("stream is gone")
        self.written += data

    def flush(self):
        pass

    def close(self):
        self._closed = True

    # process surface
    def poll(self):
        return 1 if self.killed else None

    def terminate(self):
        self.killed = True
        self._closed = True


@pytest.fixture
def player(tmp_path):
    procs = []

    def popen(argv, **kw):
        procs.append((argv, FakeProc()))
        return procs[-1][1]
    return procs, popen


def _tone(tmp_path, popen, data=None, **kw):
    return RoomTone(FakeCfg(data if data is not None else {"ambience": {"room_tone": True}}),
                    popen=popen, directory=tmp_path, **kw)


class Phase:
    def __init__(self, phase):
        self.phase = phase


class Quiet:
    def __init__(self, reason=""):
        self._reason = reason

    def reason(self, now=None):
        return self._reason


# ------------------------------------------------------------------- beds
def test_there_is_one_bed_per_arc_phase():
    from jarvis.arc import PHASES as ARC_PHASES
    assert set(PHASES) == set(ARC_PHASES)


def test_the_beds_thin_out_towards_the_edges_of_the_day():
    """Pre-dawn is almost nothing and night thins to a drone; the work
    hours carry the most. That shape IS the feature."""
    def energy(phase):
        _hz, drone, noise, _tilt = BEDS[phase]
        return drone + noise
    assert energy("pre-dawn") < energy("waking") < energy("working")
    assert energy("night") < energy("evening") < energy("dusk")


def test_a_loop_is_seamless_enough_not_to_click():
    """The ends have to meet: a 12 s bed with an audible seam is a metronome
    you cannot stop hearing."""
    import numpy as np
    for phase in PHASES:
        sig = np.asarray(roomtone._loop_samples(phase))
        seam = abs(int(sig[0]) - int(sig[-1]))
        assert seam < 0.05 * int(np.abs(sig).max()), phase


def test_a_bed_is_never_loud():
    import numpy as np
    for phase in PHASES:
        sig = np.asarray(roomtone._loop_samples(phase))
        assert int(np.abs(sig).max()) < 0.5 * 32767, phase


def test_the_loops_are_deterministic():
    """Baked once and cached forever, so the same seed must give the same
    bytes -- otherwise a cached loop is a loop nobody chose."""
    assert roomtone._loop_samples("dusk").tobytes() == \
        roomtone._loop_samples("dusk").tobytes()


def test_bake_writes_one_wav_per_phase(tmp_path):
    made = bake_all(tmp_path)
    assert set(made) == set(PHASES)
    with wave.open(str(made["night"])) as wf:
        assert wf.getnchannels() == 1 and wf.getframerate() == roomtone.SAMPLE_RATE
        assert wf.getnframes() == int(roomtone.SAMPLE_RATE * roomtone.LOOP_S)


def test_bake_does_not_resynthesize_an_existing_loop(tmp_path):
    """Correction 6: never synthesize on a tick, and certainly not while
    Whisper is decoding."""
    path = bake("night", tmp_path)
    path.write_bytes(b"already here")
    assert bake("night", tmp_path).read_bytes() == b"already here"


def test_an_unknown_phase_bakes_nothing(tmp_path):
    assert bake("brunch", tmp_path) is None


# ----------------------------------------------------------- the policy
def test_the_bed_is_off_by_default():
    assert desired_phase(enabled=False, arc_phase="working") == ""


def test_the_bed_follows_the_arc():
    for phase in PHASES:
        assert desired_phase(enabled=True, arc_phase=phase) == phase


def test_quiet_silences_the_bed():
    """quiet.py owns hours, DND, meetings and an empty house -- and being
    wrong here is audible all night."""
    assert desired_phase(enabled=True, arc_phase="working",
                         quiet_reason="quiet hours until 7:00 am") == ""


def test_an_empty_house_silences_the_bed():
    """Which also lets the Bluetooth speaker sleep."""
    assert desired_phase(enabled=True, arc_phase="working",
                         presence_state="away") == ""


def test_a_mute_silences_the_bed():
    assert desired_phase(enabled=True, arc_phase="working", muted=True) == ""


def test_a_long_silence_stops_the_bed():
    """Auto-stop with no presence: the phone may simply be unconfigured."""
    assert desired_phase(enabled=True, arc_phase="working",
                         mic_idle_s=1_800.0, away_stop_s=1_200.0) == ""
    assert desired_phase(enabled=True, arc_phase="working",
                         mic_idle_s=60.0, away_stop_s=1_200.0) == "working"


def test_an_unknown_arc_phase_plays_nothing():
    assert desired_phase(enabled=True, arc_phase="") == ""
    assert desired_phase(enabled=True, arc_phase="brunch") == ""


# ---------------------------------------------------------- the stream
def test_a_tick_brings_the_bed_up_once(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    assert tone.tick() == "night"
    assert tone.tick() == "night"
    assert len(procs) == 1


def test_the_stream_is_a_dedicated_named_paplay(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    argv = procs[0][0]
    assert argv[0] == "paplay" and f"--stream-name={roomtone.STREAM_NAME}" in argv
    assert "--raw" in argv and f"--rate={roomtone.SAMPLE_RATE}" in argv


def test_the_volume_is_near_subliminal_by_default(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    assert f"--volume={int(roomtone.DEFAULT_VOLUME * 65536)}" in procs[0][0]


def test_a_phase_change_swaps_the_bed(tmp_path, player):
    procs, popen = player
    arc = Phase("night")
    tone = _tone(tmp_path, popen, arc=arc)
    tone.tick()
    arc.phase = "waking"
    assert tone.tick() == "waking"
    assert procs[0][1].killed is True and len(procs) == 2


def test_turning_it_off_takes_the_bed_down(tmp_path, player):
    procs, popen = player
    cfg = FakeCfg({"ambience": {"room_tone": True}})
    tone = RoomTone(cfg, popen=popen, directory=tmp_path, arc=Phase("night"))
    tone.tick()
    cfg.set("ambience.room_tone", False)
    assert tone.tick() == ""
    assert procs[0][1].killed is True


def test_a_dead_player_is_restarted(tmp_path, player):
    """PulseAudio restarts, paplay dies; the room should come back."""
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    procs[0][1].terminate()
    assert tone.tick() == "night" and len(procs) == 2


def test_a_missing_paplay_is_not_an_error(tmp_path):
    def popen(argv, **kw):
        raise FileNotFoundError("no paplay here")
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    assert tone.tick() == ""


def test_pump_writes_the_loop_into_the_stream(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    assert tone._pump() is True
    expected = int(roomtone.SAMPLE_RATE * roomtone.LOOP_S) * 2
    assert len(procs[0][1].written) == expected


# ------------------------------------------------------------ mic safety
def test_the_mute_is_immediate_not_deferred(tmp_path, player):
    """A flag a tick would notice later is not good enough: the capture has
    already started by then."""
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    tone.mute()
    assert procs[0][1].killed is True and tone.playing == ""


def test_the_wake_word_mutes_before_the_mic_opens(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    tone.on_wake()
    assert tone.muted is True and procs[0][1].killed is True


def test_a_capture_keeps_the_bed_down_until_it_ends(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    tone.on_wake()
    tone.on_mic_open()
    assert tone.tick() == ""
    tone.on_mic_close()
    assert tone.muted is False and tone.tick() == "night"


def test_a_streaming_reply_does_not_latch_the_bed_off(tmp_path, player):
    """SpeakingState streams at ~12 Hz while Jarvis talks; a depth counter
    would take twelve increments and one decrement and never recover."""
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()

    class Speaking:
        def __init__(self, active):
            self.active = active

    for _ in range(12):
        tone.on_speaking(Speaking(True))
    tone.on_speaking(Speaking(False))
    assert tone.muted is False and tone.tick() == "night"


def test_a_reply_that_ends_does_not_release_a_capture(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))

    class Speaking:
        def __init__(self, active):
            self.active = active

    tone.on_mic_open()
    tone.on_speaking(Speaking(True))
    tone.on_speaking(Speaking(False))
    assert tone.muted is True         # the capture's own hold survives


def test_a_hold_nobody_released_expires(tmp_path, player):
    """An aborted capture must not mute the room until the next restart --
    silently, which is the worst way for ambience to fail."""
    procs, popen = player
    now = [1000.0]
    tone = _tone(tmp_path, popen, arc=Phase("night"), now=lambda: now[0])
    tone.mute()
    assert tone.want() == ""
    now[0] += roomtone.MAX_HOLD_S + 1
    assert tone.want() == "night"


def test_settle_takes_the_bed_down_silently(tmp_path, player):
    """The departure cue. It says nothing; it just stops."""
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    tone.settle()
    assert procs[0][1].killed is True and tone.playing == ""


def test_stop_takes_the_stream_with_it(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    tone.stop()
    assert procs[0][1].killed is True


def test_a_broken_pipe_is_normal(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"))
    tone.tick()
    procs[0][1].close()
    assert tone._pump() is False and tone.playing == ""


def test_quiet_is_read_through_the_policy_not_re_derived(tmp_path, player):
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"), quiet=Quiet("a meeting"))
    assert tone.tick() == "" and procs == []


def test_a_failing_quiet_policy_does_not_start_a_bed_by_accident(tmp_path, player):
    class Boom:
        def reason(self, now=None):
            raise RuntimeError("policy down")
    procs, popen = player
    tone = _tone(tmp_path, popen, arc=Phase("night"), quiet=Boom())
    assert tone.tick() == "night"      # a broken gate must not mute the room forever


# ----------------------------------------------------- the voice switch
def test_room_tone_on_and_off_by_voice():
    from jarvis.commander import _ROOM_TONE_RX
    for phrase in ("room tone on", "turn on the room tone", "start the room tone",
                   "jarvis, room tone on", "ambience on", "play the room tone"):
        m = _ROOM_TONE_RX.match(phrase)
        assert m is not None, phrase
        assert (m.group("state1") or m.group("state2") or m.group("verb2")), phrase
    for phrase in ("room tone off", "turn off the room tone", "stop the room tone",
                   "kill the room tone", "ambience off"):
        assert _ROOM_TONE_RX.match(phrase) is not None, phrase


def test_asking_about_the_room_tone_does_not_start_it():
    """A bare "room tone" is a question; acting on it would start audio he
    never asked for."""
    from jarvis.commander import _ROOM_TONE_RX
    m = _ROOM_TONE_RX.match("is the room tone on")
    assert m is not None and m.group("status")
    bare = _ROOM_TONE_RX.match("room tone")
    assert bare is not None
    assert not (bare.group("state1") or bare.group("state2") or bare.group("verb2"))


def _commander(cfg, tone=None):
    """A bare commander over a namespace with just what the handler needs."""
    import types

    from jarvis.commander import Commander
    c = object.__new__(Commander)
    c.services = types.SimpleNamespace(assistant=cfg, memory=None, roomtone=tone)
    return c


def test_the_switch_persists_the_preference():
    from jarvis.commander import _ROOM_TONE_RX, _h_room_tone
    cfg = FakeCfg({"ambience": {"room_tone": False}})
    c = _commander(cfg)
    res = _h_room_tone(c, "room tone on", _ROOM_TONE_RX.match("room tone on"))
    assert res.handled and res.speak and cfg.get("ambience.room_tone") is True
    res = _h_room_tone(c, "room tone off", _ROOM_TONE_RX.match("room tone off"))
    assert res.handled and cfg.get("ambience.room_tone") is False


def test_turning_it_off_by_voice_stops_the_bed_now(tmp_path, player):
    """Not at the next tick: he asked for silence."""
    from jarvis.commander import _ROOM_TONE_RX, _h_room_tone
    procs, popen = player
    cfg = FakeCfg({"ambience": {"room_tone": True}})
    tone = RoomTone(cfg, popen=popen, directory=tmp_path, arc=Phase("night"))
    tone.tick()
    _h_room_tone(_commander(cfg, tone), "room tone off",
                 _ROOM_TONE_RX.match("room tone off"))
    assert procs[0][1].killed is True and tone.playing == ""


def test_the_status_question_reports_without_switching():
    from jarvis.commander import _ROOM_TONE_RX, _h_room_tone
    cfg = FakeCfg({"ambience": {"room_tone": False}})
    res = _h_room_tone(_commander(cfg), "is the room tone on",
                       _ROOM_TONE_RX.match("is the room tone on"))
    assert res.handled and "off" in res.reply.lower()
    assert cfg.get("ambience.room_tone") is False
