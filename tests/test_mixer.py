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


def mixer(tmp_path, run=None, **cfg):
    settings = {"audio.duck": True, "audio.duck_level": 30,
                "audio.duck_ramp_ms": 0}
    settings.update(cfg)
    conf = SimpleNamespace(get=lambda k, d=None: settings.get(k, d))
    return mx.RoomMixer(cfg=conf, run=run or FakeRun(),
                        state_path=tmp_path / "mixer.json",
                        registry=mx.PidRegistry(),
                        ppid_of=lambda pid: None,
                        sleep=lambda s: None, now=lambda: 1000.0)


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
