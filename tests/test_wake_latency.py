"""The gap between the wake word and an open mic is speech thrown away.

_on_hotword sat behind a fixed `threading.Timer(0.2, ...)`. Measured across 27
real detections the wake-to-recording gap was 216-298 ms (median 259), so that
timer was most of it, and the user reported the start of every sentence being
clipped. The wait is only there so the start chime is not recorded back
through the mic -- which matters solely when a chime plays.

The thread hop is a separate concern and must survive: start() acquires the
mic arbiter, which pauses the hotword, and _on_hotword runs ON the hotword
listener thread. Pausing that thread from inside itself would wedge the
listener, so the call must never be made inline.
"""
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis.app import JarvisApp
from jarvis.config import CONFIG


class _Rec:
    recording = False

    def __init__(self):
        self.started = 0

    def start(self):
        self.started += 1


class _Clear:
    def is_set(self):
        return False


@pytest.fixture
def app(monkeypatch):
    a = object.__new__(JarvisApp)
    a.recorder = _Rec()
    a._audio_busy = _Clear()
    a._turn_busy = _Clear()
    a.turns = SimpleNamespace(mark=lambda *args, **kw: None)   # the ledger's wake mark
    monkeypatch.setattr(app_mod, "play_beep", lambda *a, **k: None)
    return a


def _timers(monkeypatch):
    """Record every Timer(delay, fn) without letting one actually fire."""
    seen = []

    class FakeTimer:
        def __init__(self, delay, fn):
            seen.append((delay, fn))
            self.fn = fn

        def start(self):
            pass

    monkeypatch.setattr(app_mod.threading, "Timer", FakeTimer)
    return seen


def test_no_chime_means_no_wait(app, monkeypatch):
    monkeypatch.setattr(CONFIG, "sound", False)
    seen = _timers(monkeypatch)
    app._on_hotword(0.9)
    assert len(seen) == 1
    assert seen[0][0] == 0.0, f"{seen[0][0]}s of speech dropped for no chime"


def test_a_chime_still_gets_its_guard(app, monkeypatch):
    """Otherwise the mic records the beep and Whisper transcribes it."""
    monkeypatch.setattr(CONFIG, "sound", True)
    seen = _timers(monkeypatch)
    app._on_hotword(0.9)
    assert seen[0][0] == app._WAKE_BEEP_GUARD_S > 0


def test_start_is_never_called_inline(app, monkeypatch):
    """It would pause the hotword from within the hotword's own thread."""
    monkeypatch.setattr(CONFIG, "sound", False)
    seen = _timers(monkeypatch)
    app._on_hotword(0.9)
    assert app.recorder.started == 0, "recorder.start ran on the listener thread"
    seen[0][1]()                                    # what the timer would run
    assert app.recorder.started == 1


def test_a_busy_turn_still_refuses_to_open_a_second_capture(app, monkeypatch):
    """Removing the delay must not weaken the one-utterance-at-a-time guard."""
    monkeypatch.setattr(CONFIG, "sound", False)
    app._audio_busy = SimpleNamespace(is_set=lambda: True)
    monkeypatch.setattr(app_mod.bus, "publish", lambda *a, **k: None)
    seen = _timers(monkeypatch)
    app._on_hotword(0.9)
    assert seen == [] and app.recorder.started == 0
