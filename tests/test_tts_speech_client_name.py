"""Speech streams carry their own application.name (2026-09-01 loudness fix).

WirePlumber restores per-stream volume keyed by application.name; every paplay
shares "paplay", and the timer chime's --volume=32768 (0.125 linear) was being
restored onto Jarvis's speech. Pin the name on BOTH playback paths."""
import subprocess

from jarvis import tts as tts_mod


class _Proc:
    def __init__(self, cmd, **kw):
        _Proc.cmds.append(cmd)
        self.returncode = 0
    def poll(self):
        return 0
    def wait(self, timeout=None):
        return 0
    def kill(self):
        pass
    def terminate(self):
        pass


def test_file_playback_names_the_speech_client(monkeypatch):
    _Proc.cmds = []
    monkeypatch.setattr(subprocess, "Popen", _Proc)
    monkeypatch.setattr(tts_mod.CONFIG, "playback_device", "", raising=False)
    t = tts_mod.TTS.__new__(tts_mod.TTS)
    t._stop_flag = False
    t._mark_audio = lambda: None
    t._play_proc = None
    t._play("/nonexistent/x.wav")
    assert _Proc.cmds, "no player spawned"
    assert _Proc.cmds[0][:2] == ["paplay", f"--client-name={tts_mod.SPEECH_CLIENT_NAME}"]
    assert tts_mod.SPEECH_CLIENT_NAME != "paplay"


def test_other_players_and_chimes_keep_their_names():
    # timekeeper's chime and the earcons stay plain paplay + --volume: only
    # streams WITHOUT an explicit --volume are exposed to the restore key.
    import inspect
    from jarvis import earcons, roomtone
    from jarvis.tools import timekeeper
    for mod in (earcons, roomtone, timekeeper):
        assert "--volume" in inspect.getsource(mod)
        assert "jarvis-speech" not in inspect.getsource(mod)
