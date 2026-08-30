"""The live transcript: words on screen WHILE speaking, not only after.

PartialText, Transcriber.partial() and the UI's ghost card all shipped with
V3; nothing ever called them, so the transcript only filled in once the user
stopped talking. These tests pin the properties that make the missing
producer safe to run against a live recording session: it previews, and it
never disturbs the real transcription that follows.
"""
import numpy as np

import jarvis.app as app_mod
import jarvis.events as events_mod
from jarvis.events import PartialText
from jarvis.recorder import SAMPLE_RATE, MicArbiter, Recorder


def _pipeline_class():
    """The class that owns _partial_loop, without naming the pipeline."""
    for name in dir(app_mod):
        obj = getattr(app_mod, name)
        if isinstance(obj, type) and hasattr(obj, "_partial_loop"):
            return obj
    raise AssertionError("no class owns _partial_loop")


def _capture(monkeypatch):
    """Collect PartialText payloads published on the bus."""
    got = []
    monkeypatch.setattr(
        events_mod.bus, "publish",
        lambda ev: got.append(ev.text) if isinstance(ev, PartialText) else None)
    return got


def test_snapshot_audio_is_side_effect_free(monkeypatch):
    """_finalize_audio publishes Status and logs; the preview must not.

    It is called repeatedly while the user is mid-sentence, so a "No audio
    captured" toast every 0.9 s would be worse than having no preview.
    """
    rec = Recorder(MicArbiter())     # built FIRST: its constructor publishes
    published = []                   # MicState, which is not what we measure
    monkeypatch.setattr(events_mod.bus, "publish", published.append)

    assert rec.snapshot_audio() is None          # empty buffer
    assert published == [], f"published on empty buffer: {published}"

    rec._audio_frames = [np.zeros((1600, 1), dtype=np.float32)]
    assert rec.snapshot_audio() is not None
    assert published == [], f"published while sampling: {published}"


def test_partial_loop_publishes_only_changes(monkeypatch):
    """Whisper returns the same text repeatedly and the card redraws on
    every event, so unchanged text must not be republished."""
    seen = _capture(monkeypatch)
    texts = ["hello", "hello", "hello there", "hello there"]
    state = {"i": 0}

    class Rec:
        recording = True

        def snapshot_audio(self):
            return np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)

    class Tr:
        def partial(self, audio):
            i = min(state["i"], len(texts) - 1)
            state["i"] += 1
            if state["i"] >= len(texts):
                Rec.recording = False
            return texts[i]

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S = 0.01, 0.5
    pipe._partial_loop()

    assert seen == ["hello", "hello there"], seen


def test_partial_loop_survives_a_failing_decode(monkeypatch):
    """A preview must never take the session down with it."""
    _capture(monkeypatch)
    state = {"n": 0}

    class Rec:
        recording = True

        def snapshot_audio(self):
            state["n"] += 1
            if state["n"] > 3:
                Rec.recording = False
            return np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)

    class Tr:
        def partial(self, audio):
            raise RuntimeError("model exploded")

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S = 0.01, 0.5
    pipe._partial_loop()          # must return, not raise


def test_short_audio_is_not_decoded(monkeypatch):
    """Below _PARTIAL_MIN_S whisper largely invents words; watching it type
    nonsense and retract it is worse than waiting."""
    seen = _capture(monkeypatch)
    calls = {"n": 0}
    state = {"n": 0}

    class Rec:
        recording = True

        def snapshot_audio(self):
            state["n"] += 1
            if state["n"] > 3:
                Rec.recording = False
            return np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)

    class Tr:
        def partial(self, audio):
            calls["n"] += 1
            return "should never be asked for"

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S = 0.01, 0.7
    pipe._partial_loop()

    assert calls["n"] == 0, "decoded audio shorter than the minimum"
    assert seen == []
