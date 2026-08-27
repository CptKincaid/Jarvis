"""Jarvis must not hear himself talk.

`jarvis/hotword.py:8-10` documents the contract: "every mic consumer
(recording, calibration, enrollment, training, TTS talk-back) pauses the
hotword stream via ``arbiter.acquire(owner)``". Every consumer did so except
TTS, so the always-on wake word listened straight through Jarvis's own
speech and could trigger on it.

These tests pin the fix to the behaviour that matters:
  - the hotword is paused *for the duration of* a spoken burst, not merely
    bracketed by balanced events,
  - a burst of chunks pauses ONCE (per-chunk pause/resume would thrash the
    audio stream, which blanks ~400 ms on every resume),
  - and every exit path releases -- a leaked acquire leaves Jarvis
    permanently deaf, which is far worse than the bug being fixed.

The real MicArbiter is used, not a fake: the re-entrancy and the
pause/resume callback wiring are the parts most likely to break.
"""
from __future__ import annotations

import threading
import time

import pytest

from jarvis.recorder import MicArbiter
from jarvis.tts import TTS


def wait_until(pred, timeout=3.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.01)
    return pred()


@pytest.fixture
def arbiter():
    """Real MicArbiter with the hotword callbacks recorded."""
    arb = MicArbiter()
    arb.events = []
    arb.register_hotword(lambda: arb.events.append("pause"),
                         lambda: arb.events.append("resume"))
    return arb


@pytest.fixture
def tts(arbiter, monkeypatch):
    """Edge TTS wired to the arbiter, with synthesis and playback stubbed."""
    t = TTS(engine="edge", cache=False, arbiter=arbiter)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    return t


def test_hotword_is_paused_while_speaking(tts, arbiter, monkeypatch):
    """Not just balanced events -- actually held for the whole utterance."""
    held = []

    def fake_speak(text):
        held.append(arbiter.held_by)

    monkeypatch.setattr(tts, "_speak_sync", fake_speak)

    tts.speak("good evening")
    assert wait_until(lambda: arbiter.events == ["pause", "resume"]), arbiter.events

    assert held == ["tts"], "hotword was not paused during speech"
    assert arbiter.held_by == ""


def test_burst_pauses_once_not_per_chunk(tts, arbiter, monkeypatch):
    """ReadAloud chunks long text into many speak() calls; each resume
    restarts the capture stream, so a burst must pause exactly once."""
    release = threading.Event()
    started = threading.Event()

    def blocking_speak(text):
        started.set()
        release.wait(3)

    monkeypatch.setattr(tts, "_speak_sync", blocking_speak)

    tts.speak("chunk one")
    assert started.wait(2)
    tts.speak("chunk two")          # queued while chunk one is still playing
    tts.speak("chunk three")
    release.set()

    assert wait_until(lambda: not tts.is_speaking)
    assert arbiter.events == ["pause", "resume"], arbiter.events


def test_interrupt_releases_the_mic(tts, arbiter, monkeypatch):
    """Barge-in must not leave the hotword paused -- that is silent deafness."""
    started = threading.Event()

    def slow_speak(text):
        started.set()
        t0 = time.monotonic()
        while not tts._stop_flag and time.monotonic() - t0 < 3:
            time.sleep(0.01)

    monkeypatch.setattr(tts, "_speak_sync", slow_speak)

    tts.speak("a long line that gets cut off")
    tts.speak("and one queued behind it")
    assert started.wait(2)
    assert arbiter.held_by == "tts"

    assert tts.interrupt() is True
    assert wait_until(lambda: not tts.is_speaking)
    assert wait_until(lambda: arbiter.held_by == "")
    assert arbiter.events == ["pause", "resume"]


def test_synthesis_failure_still_releases(tts, arbiter, monkeypatch):
    """An exception inside the worker must not strand the acquire."""
    def boom(text):
        raise RuntimeError("synth exploded")

    monkeypatch.setattr(tts, "_speak_sync", boom)

    tts.speak("this will fail")
    assert wait_until(lambda: arbiter.events == ["pause", "resume"]), arbiter.events
    assert arbiter.held_by == ""


def test_without_an_arbiter_nothing_breaks(monkeypatch):
    """Standalone use (voice_check, tests, CLI) passes no arbiter."""
    t = TTS(engine="edge", cache=False)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    monkeypatch.setattr(t, "_speak_sync", lambda text: None)

    t.speak("no arbiter here", block=True)
    assert wait_until(lambda: not t.is_speaking)


def test_recorder_and_tts_nest_without_double_resume(tts, arbiter, monkeypatch):
    """Recording while speaking (barge-in windows, enrollment) must pause
    once and resume once -- the arbiter is re-entrant by depth."""
    observed = []

    def speak_then_record(text):
        with arbiter.acquire("record_fixed"):
            observed.append(arbiter.held_by)

    monkeypatch.setattr(tts, "_speak_sync", speak_then_record)

    tts.speak("speaking while the recorder also wants the mic")
    assert wait_until(lambda: arbiter.events == ["pause", "resume"]), arbiter.events

    assert observed == ["tts"], "nested acquire changed ownership"
