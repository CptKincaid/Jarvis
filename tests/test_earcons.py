"""The earcon lexicon (jarvis/earcons.py) and the beeps it replaced."""
import wave

import pytest

from jarvis import earcons


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    earcons.set_config(None)
    earcons.reset_cooldowns()
    monkeypatch.setattr(earcons.PATHS, "MEMORY_DIR", tmp_path)
    yield
    earcons.set_config(None)
    earcons.reset_cooldowns()


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


def _play_log():
    calls = []

    def run(argv):
        calls.append(argv)
        return True
    return calls, run


# --------------------------------------------------------------- the family
def test_the_lexicon_is_the_six_named_tones():
    assert set(earcons.NAMES) == {"heard-you", "thinking", "held-back",
                                  "done", "warning", "arrival"}


def test_every_tone_is_built_from_the_one_family():
    """The accent: 220/330/440/660/880/1320 -- a root, its fifth and its
    octave, which are exactly the three pitches the room already used."""
    family = {earcons.ROOT_LOW, earcons.FIFTH_LOW, earcons.ROOT,
              earcons.FIFTH, earcons.OCTAVE, earcons.TWELFTH}
    for name, (notes, _gain) in earcons.TONES.items():
        assert {f for f, _ms in notes} <= family, name


def test_the_old_pitches_survived_as_the_family_root_fifth_and_octave():
    assert (earcons.ROOT, earcons.FIFTH, earcons.OCTAVE) == (440.0, 660.0, 880.0)


def test_every_tone_is_under_four_hundred_milliseconds():
    for name, (notes, _gain) in earcons.TONES.items():
        assert sum(ms for _f, ms in notes) <= 400.0, name


# ------------------------------------------------------------------ render
def test_render_writes_a_playable_mono_wav(tmp_path):
    path = earcons.render("arrival")
    assert path.exists() and path.parent.name == "earcons"
    with wave.open(str(path)) as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
        assert wf.getframerate() == earcons.SAMPLE_RATE
        assert wf.getnframes() > 0


def test_render_is_deterministic():
    first = earcons.render("done", force=True).read_bytes()
    second = earcons.render("done", force=True).read_bytes()
    assert first == second


def test_render_does_not_rewrite_an_existing_file():
    path = earcons.render("done")
    path.write_bytes(b"RIFF-not-really")
    assert earcons.render("done").read_bytes() == b"RIFF-not-really"


def test_the_envelope_starts_and_ends_at_silence():
    """A hard edge clicks. The raised cosine is the reason the six tones
    sound like one instrument."""
    import numpy as np
    notes, gain = earcons.TONES["heard-you"]
    sig = np.asarray(earcons._samples(notes, gain))
    assert abs(int(sig[0])) < 200 and abs(int(sig[-1])) < 200
    assert int(np.abs(sig).max()) > 5000       # and it is audible in between


def test_render_all_bakes_the_whole_lexicon(tmp_path):
    made = earcons.render_all(tmp_path / "audition")
    assert set(made) == set(earcons.NAMES)
    assert all(p.exists() for p in made.values())


def test_an_unknown_name_renders_nothing():
    assert earcons.render("fanfare") is None
    assert earcons.play("fanfare") is False


# ------------------------------------------------------------------ aliases
def test_the_three_historical_kinds_map_into_the_family():
    assert earcons.resolve("start") == "heard-you"
    assert earcons.resolve("stop") == "done"
    assert earcons.resolve("nudge") == "held-back"


def test_recorder_play_beep_now_goes_through_the_lexicon(monkeypatch):
    """One beep system, not two: the failure the lexicon exists to prevent."""
    from jarvis import recorder as recorder_mod
    seen = []
    monkeypatch.setattr(earcons, "play", lambda name: seen.append(name))
    recorder_mod.play_beep("nudge")
    assert seen == ["nudge"]


def test_recorder_init_beeps_still_resolves_its_three_kinds(monkeypatch):
    from jarvis import recorder as recorder_mod
    monkeypatch.setattr(recorder_mod, "_BEEP_FILES", {})
    recorder_mod._init_beeps()
    assert set(recorder_mod._BEEP_FILES) == {"start", "stop", "nudge"}


# --------------------------------------------------------------- the gate
def test_the_gate_is_sound_earcons_and_not_config_sound():
    """CONFIG.sound is False by default and means the old chimes; hanging
    the lexicon off it would ship it mute."""
    calls, run = _play_log()
    earcons.set_config(FakeCfg({"sound": {"earcons": False}}))
    assert earcons.play("arrival", run=run) is False and calls == []
    earcons.set_config(FakeCfg({"sound": {"earcons": True}}))
    assert earcons.play("arrival", run=run) is True and calls


def test_no_config_installed_still_plays():
    """A script or a test gets sound, not silence."""
    calls, run = _play_log()
    assert earcons.play("done", run=run) is True and calls


def test_the_volume_rides_on_paplay():
    calls, run = _play_log()
    earcons.set_config(FakeCfg({"sound": {"volume": 0.25}}))
    earcons.play("done", run=run)
    assert calls[0][0] == "paplay" and calls[0][1] == f"--volume={int(0.25 * 65536)}"


def test_aplay_is_the_fallback_when_paplay_is_missing():
    calls = []

    def run(argv):
        calls.append(argv)
        return argv[0] != "paplay"
    assert earcons.play("done", run=run) is True
    assert [c[0] for c in calls] == ["paplay", "aplay"]


def test_playback_never_raises():
    def boom(argv):
        raise RuntimeError("pulse is gone")
    assert earcons.play("done", run=boom) is False


# ------------------------------------------------------------- rate limit
def test_a_repeat_of_the_same_tone_is_dropped(monkeypatch):
    """The false-wake metronome: HotwordDetected fires on false wakes and a
    noisy room must not turn the bloom into a beat."""
    now = [100.0]
    monkeypatch.setattr(earcons, "_clock", lambda: now[0])
    calls, run = _play_log()
    earcons.set_config(FakeCfg({"sound": {"cooldown_s": 4}}))
    assert earcons.play("heard-you", run=run) is True
    now[0] += 1.0
    assert earcons.play("heard-you", run=run) is False
    now[0] += 4.0
    assert earcons.play("heard-you", run=run) is True
    assert len(calls) == 2


def test_a_different_tone_is_not_swallowed_by_the_cooldown(monkeypatch):
    """Why the limit is per-name: "done" closes a capture two seconds after
    "heard-you" opened it, and one cue across all causes would eat it."""
    now = [100.0]
    monkeypatch.setattr(earcons, "_clock", lambda: now[0])
    calls, run = _play_log()
    earcons.set_config(FakeCfg({"sound": {"cooldown_s": 30}}))
    assert earcons.play("heard-you", run=run) is True
    now[0] += 2.0
    assert earcons.play("done", run=run) is True
    assert len(calls) == 2


def test_two_tones_may_not_stack(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(earcons, "_clock", lambda: now[0])
    calls, run = _play_log()
    assert earcons.play("heard-you", run=run) is True
    now[0] += earcons.MIN_GAP_S / 2
    assert earcons.play("done", run=run) is False


def test_a_broken_cooldown_setting_falls_back_to_the_default(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(earcons, "_clock", lambda: now[0])
    calls, run = _play_log()
    earcons.set_config(FakeCfg({"sound": {"cooldown_s": "soon"}}))
    assert earcons.play("done", run=run) is True
    now[0] += 1.0
    assert earcons.play("done", run=run) is False


def test_a_caller_may_shorten_the_same_tone_cooldown_for_its_own_play(monkeypatch):
    """The gesture (jarvis/gesturecast.py) plays heard-you at the pace of a
    hand: grab-drop-grab inside 4 s. The wake-word path passes nothing and
    keeps the config value, so a false-wake metronome is still swallowed."""
    now = [100.0]
    monkeypatch.setattr(earcons, "_clock", lambda: now[0])
    calls, run = _play_log()
    assert earcons.play("heard-you", run, cooldown_s=0.6)
    now[0] += 1.0
    assert not earcons.play("heard-you", run)              # the 4 s default
    assert earcons.play("heard-you", run, cooldown_s=0.6)  # the gesture's own
    now[0] += 0.3
    assert not earcons.play("heard-you", run, cooldown_s=0.6)
    assert len(calls) == 2


def test_a_broken_per_call_cooldown_falls_back_to_the_default(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(earcons, "_clock", lambda: now[0])
    calls, run = _play_log()
    assert earcons.play("done", run, cooldown_s="soon")
    now[0] += 1.0
    assert not earcons.play("done", run, cooldown_s="soon")
    assert len(calls) == 1
