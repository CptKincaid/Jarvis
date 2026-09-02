"""The Room Mixer (jarvis/mixer.py).

The fixture is a REAL ``pactl list sink-inputs`` dump taken on this box
while librespot was playing AND Jarvis was speaking, so both streams are
present at once -- and both report ``application.process.binary = "pacat"``,
which is precisely why the exemption is by PID and never by name.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis import mixer as mx

FIXTURE = Path(__file__).parent / "fixtures" / "pactl_sink_inputs.txt"
DUMP = FIXTURE.read_text()

SPOTIFY_PID = 3132645          # sink-input #92, pacat, "Spotify (Spark)"
JARVIS_PID = 3708595           # sink-input #2790, paplay, a tts_cache wav


class FakeRun:
    """The subprocess seam, recording argv and answering listings."""

    def __init__(self, dump=DUMP, rc=0):
        self.dump = dump
        self.rc = rc
        self.calls: list[list] = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        listing = argv[:3] == ["pactl", "list", "sink-inputs"]
        return SimpleNamespace(returncode=self.rc,
                               stdout=self.dump if listing else "",
                               stderr="")

    @property
    def writes(self):
        return [c for c in self.calls if "set-sink-input-volume" in c]


def mixer(tmp_path, run=None, remote=None, **cfg):
    settings = {"audio.duck": True, "audio.duck_level": 30,
                "audio.duck_ramp_ms": 0}
    settings.update(cfg)
    conf = SimpleNamespace(get=lambda k, d=None: settings.get(k, d))
    return mx.RoomMixer(cfg=conf, run=run or FakeRun(),
                        state_path=tmp_path / "mixer.json",
                        registry=mx.PidRegistry(),
                        ppid_of=lambda pid: None,
                        sleep=lambda s: None, now=lambda: 1000.0,
                        remote=remote)


class FakeRemote:
    """A stand-in for jarvis.tools.spotify.SpotifyTool's duck()/unduck(),
    plus the two read-only bits the mixer takes from it: the name of the
    device it ducked and its cached music state."""

    def __init__(self, ok=True, boom=False, name="HPCOMPUTER", playing=False,
                 during_duck=None):
        self.ok, self.boom = ok, boom
        self.name, self.playing = name, playing
        self.during_duck = during_duck       # runs INSIDE duck(): the latency seam
        self.ducked: list = []
        self.unducked = 0

    def duck(self, pct):
        if self.boom:
            raise RuntimeError("spotify is down")
        if self.during_duck is not None:
            self.during_duck()
        self.ducked.append(pct)
        return self.ok

    def unduck(self):
        self.unducked += 1
        return True

    @property
    def ducked_device(self):
        return self.name if len(self.ducked) > self.unducked else None

    def music_playing(self):
        return self.playing


# ---------------------------------------------------------------- parsing
def test_parse_sink_inputs_reads_both_live_streams():
    inputs = mx.parse_sink_inputs(DUMP)
    assert [i["index"] for i in inputs] == [92, 2790]
    spotify, jarvis = inputs
    assert spotify["pid"] == SPOTIFY_PID
    assert spotify["app_name"] == "pacat"
    assert spotify["media_name"] == "Spotify (Spark)"
    assert spotify["volume_pct"] == 100 and spotify["corked"] is False
    assert spotify["restore_key"] == "sink-input-by-application-name:pacat"
    assert jarvis["pid"] == JARVIS_PID and jarvis["app_name"] == "paplay"
    assert jarvis["sink"] == spotify["sink"] == "102"
    # node.name is what the filter-chain's playback stream is known by
    # (see AEC_PLAYBACK_NAME); on these two it merely echoes the binary.
    assert spotify["node_name"] == "pacat" and jarvis["node_name"] == "paplay"


def test_parse_sink_inputs_empty_and_garbage():
    assert mx.parse_sink_inputs("") == []
    assert mx.parse_sink_inputs("no streams here\n") == []


def test_parse_takes_the_loudest_channel():
    text = ("Sink Input #1\n\tVolume: front-left: 3 / 40% / -8 dB,   "
            "front-right: 4 / 70% / -3 dB\n\t        balance 0.00\n")
    assert mx.parse_sink_inputs(text)[0]["volume_pct"] == 70


# ---------------------------------------------------------------- planning
def test_plan_duck_moves_spotify_and_leaves_jarvis_alone():
    inputs = mx.parse_sink_inputs(DUMP)
    plan = mx.plan_duck(inputs, exempt_pids={JARVIS_PID}, floor_pct=30)
    assert plan == [["pactl", "set-sink-input-volume", "92", "30%"]]


def test_plan_duck_never_names_a_sink():
    inputs = mx.parse_sink_inputs(DUMP)
    for argv in mx.plan_duck(inputs, exempt_pids={JARVIS_PID}):
        assert "set-sink-volume" not in argv and "set-sink-mute" not in argv


def test_duck_targets_skips_corked_and_already_quiet():
    inputs = [{"index": 1, "pid": 10, "volume_pct": 100, "corked": True},
              {"index": 2, "pid": 11, "volume_pct": 20, "corked": False},
              {"index": 3, "pid": 12, "volume_pct": 55, "corked": False}]
    assert [i["index"] for i in mx.duck_targets(inputs, (), 30)] == [3]


def _stream(index, pid, **props):
    base = {"index": index, "pid": pid, "volume_pct": 100, "corked": False,
            "app_name": "", "media_name": "", "node_name": ""}
    base.update(props)
    return base


@pytest.mark.parametrize("field", ["node_name", "media_name", "app_name"])
def test_duck_targets_exempts_the_aec_playback_stream_by_name(field):
    """With echo cancellation on, Jarvis's voice leaves the box as a
    sink-input owned by the filter-chain's pipewire process, NOT by Jarvis
    or any PID he spawned, so the PID registry cannot see it -- and a
    30 % Jarvis is the one thing the mixer exists to prevent.  The stream
    is known only by its name, which may land in any of the three name
    properties depending on how the chain was declared."""
    aec = _stream(7, 1234, **{field: mx.AEC_PLAYBACK_NAME})
    music = _stream(8, 5678, app_name="pacat", media_name="Spotify (Spark)")
    assert [i["index"] for i in mx.duck_targets([aec, music], (), 30)] == [8]
    # Exact name only: a stream merely mentioning it is somebody else's.
    near = _stream(9, 1234, **{field: mx.AEC_PLAYBACK_NAME + ".monitor"})
    assert [i["index"] for i in mx.duck_targets([near], (), 30)] == [9]
    assert mx.AEC_PLAYBACK_NAME == "jarvis_aec_playback"


def test_plan_restore_puts_the_original_back():
    saved = [{"index": 92, "volume_pct": 100}, {"index": 7}]
    assert mx.plan_restore(saved) == [
        ["pactl", "set-sink-input-volume", "92", "100%"]]


def test_ramp_ends_exactly_on_the_target():
    assert mx.ramp(100, 30, 4) == [82, 65, 48, 30]
    assert mx.ramp(30, 100, 4)[-1] == 100
    assert mx.ramp(50, 50, 1) == [50]


# ------------------------------------------------------------- own streams
def test_descends_from_walks_the_parent_chain():
    tree = {500: 400, 400: 300, 300: 1}
    assert mx.descends_from(500, 300, tree.get)
    assert mx.descends_from(300, 300, tree.get)
    assert not mx.descends_from(500, 999, tree.get)


def test_descends_from_survives_a_lying_proc():
    assert not mx.descends_from(5, 1, lambda pid: 5)      # self-parent
    assert not mx.descends_from("x", 1, lambda pid: 1)
    assert not mx.descends_from(5, 1, lambda pid: None)


def test_registry_exempts_a_player_jarvis_spawned(tmp_path):
    m = mixer(tmp_path)
    inputs = mx.parse_sink_inputs(DUMP)
    assert JARVIS_PID not in m.exempt_pids(inputs)         # not yet registered
    m._registry.add(JARVIS_PID)
    assert JARVIS_PID in m.exempt_pids(inputs)
    m._registry.discard(JARVIS_PID)
    assert JARVIS_PID not in m.exempt_pids(inputs)


def test_a_descendant_of_this_process_is_exempt_without_registering(tmp_path):
    import os
    m = mixer(tmp_path)
    m._ppid_of = lambda pid: os.getpid() if pid == SPOTIFY_PID else None
    assert SPOTIFY_PID in m.exempt_pids(mx.parse_sink_inputs(DUMP))


def test_module_level_registry_round_trips():
    mx.register_own_pid(424242)
    assert 424242 in mx.OWN_PIDS.snapshot()
    mx.forget_own_pid(424242)
    assert 424242 not in mx.OWN_PIDS.snapshot()


# --------------------------------------------------------------- ducking
def test_duck_lowers_only_the_music_stream(tmp_path):
    run = FakeRun()
    m = mixer(tmp_path, run=run)
    m._registry.add(JARVIS_PID)
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    assert run.writes, "nothing was ducked"
    assert {c[2] for c in run.writes} == {"92"}
    assert run.writes[-1] == ["pactl", "set-sink-input-volume", "92", "30%"]


def test_duck_never_touches_the_master_sink(tmp_path):
    run = FakeRun()
    m = mixer(tmp_path, run=run)
    m._registry.add(JARVIS_PID)
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    m.on_speaking(SimpleNamespace(active=False))
    m.pump()
    for argv in run.calls:
        assert "set-sink-volume" not in argv
        assert "set-sink-mute" not in argv
        assert "set-default-sink" not in argv


def test_restore_returns_the_stream_to_its_original(tmp_path):
    run = FakeRun()
    m = mixer(tmp_path, run=run)
    m._registry.add(JARVIS_PID)
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    m.on_speaking(SimpleNamespace(active=False))
    m.pump()
    assert run.writes[-1] == ["pactl", "set-sink-input-volume", "92", "100%"]
    assert not (tmp_path / "mixer.json").exists()


def test_speaking_state_repeats_do_not_re_duck(tmp_path):
    run = FakeRun()
    m = mixer(tmp_path, run=run)
    m._registry.add(JARVIS_PID)
    for _ in range(12):                       # SpeakingState arrives ~12 Hz
        m.on_speaking(SimpleNamespace(active=True, amplitude=0.4))
        m.pump()
    listings = [c for c in run.calls if c[:3] == ["pactl", "list", "sink-inputs"]]
    assert len(listings) == 1


def test_recording_holds_the_duck_through_the_end_of_speech(tmp_path):
    run = FakeRun()
    m = mixer(tmp_path, run=run)
    m._registry.add(JARVIS_PID)
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    m.on_recording_started()
    m.pump()
    m.on_speaking(SimpleNamespace(active=False))
    m.pump()
    assert run.writes[-1][-1] == "30%", "the mic is still open"
    m.on_recording_stopped()
    m.pump()
    assert run.writes[-1][-1] == "100%"


def test_duck_off_by_config_writes_nothing(tmp_path):
    run = FakeRun()
    m = mixer(tmp_path, run=run, **{"audio.duck": False})
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    assert run.writes == []


def test_floor_level_is_configurable_and_clamped(tmp_path):
    assert mixer(tmp_path, **{"audio.duck_level": 55}).floor_pct == 55
    assert mixer(tmp_path, **{"audio.duck_level": 0}).floor_pct == mx.MIN_TOUCH_PCT
    assert mixer(tmp_path, **{"audio.duck_level": 400}).floor_pct == 95
    assert mixer(tmp_path, **{"audio.duck_level": "loud"}).floor_pct == 30


def test_stands_down_when_nothing_local_is_playing(tmp_path):
    """Spotify on his phone: pactl sees only Jarvis's own stream."""
    run = FakeRun()
    m = mixer(tmp_path, run=run)
    m._registry.add(JARVIS_PID)
    m._registry.add(SPOTIFY_PID)              # pretend both are ours
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    assert run.writes == []
    m.on_speaking(SimpleNamespace(active=False))
    m.pump()
    assert run.writes == []


def test_a_pactl_outage_is_survivable(tmp_path):
    def boom(argv, **kw):
        raise FileNotFoundError("pactl")
    m = mixer(tmp_path, run=boom)
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()                                   # must not raise
    m.on_speaking(SimpleNamespace(active=False))
    m.pump()


# ---------------------------------------------------- stream-restore heal
def test_state_file_is_written_before_the_ramp(tmp_path):
    calls = []

    def run(argv, **kw):
        calls.append((list(argv), (tmp_path / "mixer.json").exists()))
        listing = argv[:3] == ["pactl", "list", "sink-inputs"]
        return SimpleNamespace(returncode=0, stdout=DUMP if listing else "")

    m = mixer(tmp_path, run=run)
    m._registry.add(JARVIS_PID)
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    writes = [(a, existed) for a, existed in calls if "set-sink-input-volume" in a]
    assert writes and all(existed for _, existed in writes), \
        "a crash mid-ramp would leave nothing to heal from"


def test_heal_restores_a_stream_a_crashed_run_left_ducked(tmp_path):
    state = tmp_path / "mixer.json"
    state.write_text('{"t": 1000.0, "streams": [{"index": 92, '
                     '"volume_pct": 100, "restore_key": '
                     '"sink-input-by-application-name:pacat", "app_name": "pacat"}]}')
    ducked = DUMP.replace("front-left: 65536 / 100% / 0.00 dB,   "
                          "front-right: 65536 / 100% / 0.00 dB",
                          "front-left: 19660 / 30% / -20 dB,   "
                          "front-right: 19660 / 30% / -20 dB", 1)
    run = FakeRun(dump=ducked)
    m = mixer(tmp_path, run=run)
    assert m.heal() == 1
    assert run.writes == [["pactl", "set-sink-input-volume", "92", "100%"]]
    assert not state.exists()


def test_heal_keeps_an_entry_whose_stream_has_not_come_back(tmp_path):
    """The poison is keyed by application NAME: a librespot restarted
    tomorrow inherits it, so the entry must outlive this boot."""
    state = tmp_path / "mixer.json"
    state.write_text('{"t": 1000.0, "streams": [{"index": 92, '
                     '"volume_pct": 100, "restore_key": '
                     '"sink-input-by-application-name:librespot"}]}')
    run = FakeRun()
    m = mixer(tmp_path, run=run)
    assert m.heal() == 0
    assert run.writes == []
    assert state.exists()


def test_heal_leaves_a_volume_he_has_since_raised_himself(tmp_path):
    state = tmp_path / "mixer.json"
    state.write_text('{"t": 1000.0, "streams": [{"index": 92, '
                     '"volume_pct": 100, "restore_key": '
                     '"sink-input-by-application-name:pacat"}]}')
    run = FakeRun()                            # the live dump is at 100 %
    m = mixer(tmp_path, run=run)
    assert m.heal() == 0
    assert run.writes == []


def test_stale_entries_expire(tmp_path):
    state = tmp_path / "mixer.json"
    state.write_text('{"t": 0, "streams": [{"index": 92, "volume_pct": 100, '
                     '"restore_key": "sink-input-by-application-name:x", '
                     '"t": 0}]}')
    m = mixer(tmp_path, run=FakeRun())
    m._now = lambda: mx.STALE_MAX_S * 3
    assert m.heal() == 0
    assert not state.exists()


# ---------------------------------------------------------------- thread
def test_start_and_stop_are_joinable_and_restore(tmp_path):
    run = FakeRun()
    m = mixer(tmp_path, run=run)
    m._registry.add(JARVIS_PID)
    m.start()
    try:
        m.on_speaking(SimpleNamespace(active=True))
        for _ in range(200):
            if run.writes:
                break
            import time as _t
            _t.sleep(0.01)
        assert run.writes, "the worker never ducked"
    finally:
        m.stop()
    assert m._thread is not None and not m._thread.is_alive()
    assert run.writes[-1][-1] == "100%"


def test_stop_without_start_is_safe(tmp_path):
    mixer(tmp_path).stop()


@pytest.mark.parametrize("pid", [None, "", "nope"])
def test_registry_ignores_rubbish(pid):
    reg = mx.PidRegistry()
    reg.add(pid)
    reg.discard(pid)
    assert reg.snapshot() == set()


# ------------------------------------------------- #72 the remote Connect duck
# "He also cant discern my voice from the vocalists in the music."  On the
# evening of the test the ONLY local sink-input was the idle librespot pipe --
# the music he could hear was on HPCOMPUTER, a Spotify Connect device, where
# pactl reaches nothing.  So `mixer: ducked 1 stream(s) to 30%` sat in the log
# while the music stayed exactly where it was.  The Connect device's own
# volume is the only handle on that music, and pactl cannot tell the mixer
# whether it is needed -- so it is pulled on every hold, local duck or not.
def test_remote_duck_fires_when_there_is_nothing_local(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    remote = FakeRemote()
    run = FakeRun(dump="")                       # no sink-inputs at all
    m = mixer(tmp_path, run=run, remote=remote)
    m.on_recording_started()
    m.pump()
    assert remote.ducked == [30]
    assert run.writes == []                      # nothing local was touched
    m.on_recording_stopped()
    m.pump()
    assert remote.unducked == 1


def test_remote_duck_restore_edge_fires_without_a_local_duck(tmp_path, monkeypatch):
    """A remote-only duck leaves self._ducked empty; without the remote flag
    in pump() the restore edge never fires and his Spotify volume stays at
    30% of where he left it."""
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    remote = FakeRemote()
    m = mixer(tmp_path, run=FakeRun(dump=""), remote=remote)
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    m.pump()                                     # a second pass must not re-duck
    assert remote.ducked == [30]
    m.on_speaking(SimpleNamespace(active=False))
    m.pump()
    assert remote.unducked == 1


def test_remote_duck_fires_despite_an_uncorked_local_target(tmp_path, monkeypatch,
                                                            caplog):
    """The 2026-09-01 incident in one test.  The real dump has the librespot
    pipe (#92: pacat, uncorked, 100%) even though the Spark had never been
    picked as a device and the pipe carried silence; "local streams win"
    meant the remote duck never fired ONCE that evening.  Both must go down,
    both must come back."""
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    remote = FakeRemote()
    run = FakeRun()                              # the real dump: 2 streams
    m = mixer(tmp_path, run=run, remote=remote)
    m._registry.add(JARVIS_PID)                  # the paplay is Jarvis himself
    m.on_recording_started()
    with caplog.at_level("INFO", logger="jarvis.mixer"):
        m.pump()
    assert {w[2] for w in run.writes} == {"92"}  # the pipe was ducked locally...
    assert remote.ducked == [30]                 # ...AND the Connect device
    assert ("mixer: ducked the Spotify Connect device HPCOMPUTER to 30% of its "
            "volume") in caplog.text
    m.on_recording_stopped()
    with caplog.at_level("INFO", logger="jarvis.mixer"):
        m.pump()
    assert remote.unducked == 1
    assert run.writes[-1] == ["pactl", "set-sink-input-volume", "92", "100%"]
    assert "mixer: restored the Spotify Connect device HPCOMPUTER" in caplog.text


def test_remote_duck_fires_on_speaking_as_well_as_recording(tmp_path, monkeypatch):
    """Both edges hold the room; the vocalist is as much a problem for his
    reply being heard as for his question being understood."""
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    remote = FakeRemote()
    m = mixer(tmp_path, run=FakeRun(), remote=remote)
    m.on_speaking(SimpleNamespace(active=True))
    m.pump()
    assert remote.ducked == [30]
    m.on_speaking(SimpleNamespace(active=False))
    m.pump()
    assert remote.unducked == 1


def test_remote_is_asked_once_per_hold_not_once_per_second(tmp_path, monkeypatch):
    """No active Connect device -> duck() says False.  The worker wakes every
    second while a hold is up, and the two Web API calls a remote duck costs
    must not be repeated on each wake-up; a NEW hold may ask again."""
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    remote = FakeRemote(ok=False)
    m = mixer(tmp_path, run=FakeRun(dump=""), remote=remote)
    m.on_recording_started()
    for _ in range(4):
        m.pump()
    assert remote.ducked == [30]
    m.on_recording_stopped()
    m.pump()
    m.on_recording_started()
    m.pump()
    assert remote.ducked == [30, 30]


def test_a_duck_that_lands_after_the_hold_ended_is_restored(tmp_path, monkeypatch):
    """The two API calls take 0.3-0.6 s; a short "Jarvis, stop" can be over
    before they return.  The edge that ended the hold already woke the
    worker, so the very next pump has to see the late duck and lift it."""
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    m = mixer(tmp_path, run=FakeRun(dump=""))
    remote = FakeRemote(during_duck=m.on_recording_stopped)
    m.set_remote(remote)
    m.on_recording_started()
    m.pump()                                     # duck lands, hold already gone
    assert remote.ducked == [30] and remote.unducked == 0
    assert m.music_playing()                     # a duck in force IS music
    m.pump()                                     # the wake the edge queued
    assert remote.unducked == 1
    assert not m.music_playing()


def test_a_duck_that_lands_during_stop_is_restored(tmp_path, monkeypatch):
    """stop() restores before raising the flag, so a duck still in flight is
    invisible to it; whoever finishes last must put the device back."""
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    m = mixer(tmp_path, run=FakeRun(dump=""))
    remote = FakeRemote(during_duck=m.stop)
    m.set_remote(remote)
    m.on_recording_started()
    m.pump()
    assert remote.ducked == [30]
    assert remote.unducked == 1


def test_music_playing_is_the_remotes_cache_or_a_duck_in_force(tmp_path):
    """The wake gate asks this ~10 times a second when a wake fires; it must
    be a cache read, never a request, and False when nothing is wired."""
    assert mixer(tmp_path).music_playing() is False
    remote = FakeRemote(playing=True)
    m = mixer(tmp_path, remote=remote)
    assert m.music_playing() is True
    remote.playing = False
    assert m.music_playing() is False


def test_music_playing_survives_a_broken_remote(tmp_path):
    class Broken:
        def music_playing(self):
            raise RuntimeError("token expired")
    m = mixer(tmp_path, remote=Broken())
    assert m.music_playing() is False
    m = mixer(tmp_path, remote=object())         # no such method at all
    assert m.music_playing() is False


def test_remote_duck_is_suppressed_by_room_control_off(tmp_path, monkeypatch):
    """conftest sets JARVIS_ROOM_CONTROL=0 because the suite builds the REAL
    app; a remote duck must not reach across the network and turn down music
    he is actually listening to."""
    monkeypatch.setenv("JARVIS_ROOM_CONTROL", "0")
    remote = FakeRemote()
    m = mixer(tmp_path, run=FakeRun(dump=""), remote=remote)
    m.on_recording_started()
    m.pump()
    assert remote.ducked == []


def test_a_broken_remote_never_breaks_the_mixer(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    remote = FakeRemote(boom=True)
    m = mixer(tmp_path, run=FakeRun(dump=""), remote=remote)
    m.on_recording_started()
    m.pump()
    m.on_recording_stopped()
    m.pump()
    assert remote.unducked == 0                  # nothing to put back


def test_stop_restores_the_remote_too(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    remote = FakeRemote()
    m = mixer(tmp_path, run=FakeRun(dump=""), remote=remote)
    m.on_recording_started()
    m.pump()
    m.stop()
    assert remote.unducked == 1


def test_no_remote_configured_is_the_old_stand_down(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    run = FakeRun(dump="")
    m = mixer(tmp_path, run=run)
    m.on_recording_started()
    m.pump()
    m.on_recording_stopped()
    m.pump()
    assert run.writes == []


def test_set_remote_wires_the_ducker_after_construction(tmp_path, monkeypatch):
    """app.py builds the mixer before the tools, so services.spotify cannot be
    passed to __init__; the wiring has to be a post-hoc call (#72)."""
    monkeypatch.delenv("JARVIS_ROOM_CONTROL", raising=False)
    remote = FakeRemote()
    m = mixer(tmp_path, run=FakeRun(dump=""))
    m.set_remote(remote)
    m.on_recording_started()
    m.pump()
    assert remote.ducked == [30]
