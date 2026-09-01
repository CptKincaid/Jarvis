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

    # trailing "": the loop retracts its own ghost card as the capture ends
    assert seen == ["hello", "hello there", ""], seen


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


# --------------------------------------------------------- the ghost card
#
# 2026-08-31 voice test, Hunter: "2- presentation or just 2- keeps showing
# up after a response and hes listening back for me", plus a 20:47
# screenshot of a card holding a 100+ digit number. Both are this preview:
# the ghost card had no way to come down except a Transcribed or a
# UserUtterance replacing it, and the two turns that produce NEITHER --
# the follow-up window closing on silence (abort) and a gated-out clip
# ("Too short" / "No audio captured") -- are exactly the turns where the
# loop has been decoding room noise for the whole window.

def test_a_turn_that_never_transcribes_publishes_nothing_to_clear_it(monkeypatch):
    """The cause, pinned: RecordingStopped(reason="abort") -- the
    follow-up window closing on silence -- returns out of
    _on_recording_stopped before any event is published, so nothing the
    UI listens to for "drop the ghost card" ever arrives."""
    from jarvis.events import RecordingStopped

    published = []
    monkeypatch.setattr(events_mod.bus, "publish", published.append)
    pipe = object.__new__(_pipeline_class())
    pipe.recorder = type("Rec", (), {"last_audio": None})()
    pipe._on_recording_stopped(RecordingStopped(reason="abort"))
    assert published == [], published        # no Transcribed, no UserUtterance


def test_the_ghost_card_is_retracted_when_the_mic_shuts(monkeypatch):
    """So the loop retracts its own preview: the LAST PartialText of a
    capture is empty, which is the UI's take-the-card-down signal.

    Without this the noise babble from a silent follow-up window stays on
    screen looking like Jarvis is still listening."""
    seen = _capture(monkeypatch)
    state = {"n": 0}

    class Rec:
        recording = True

        def snapshot_audio(self):
            state["n"] += 1
            if state["n"] > 2:
                Rec.recording = False
            return np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)

    class Tr:
        def partial(self, audio):
            return "drink water every"        # a preview that made it up

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S = 0.01, 0.5
    pipe._partial_loop()

    assert seen[-1] == "", seen
    assert seen.count("") == 1, seen          # one retraction, not a stream


def test_a_capture_that_previewed_nothing_retracts_nothing(monkeypatch):
    """The retraction is not a per-turn broadcast: a turn with no preview
    must not publish an event the UI has to redraw for."""
    seen = _capture(monkeypatch)
    state = {"n": 0}

    class Rec:
        recording = True

        def snapshot_audio(self):
            state["n"] += 1
            if state["n"] > 2:
                Rec.recording = False
            return np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)   # too short

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), object()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S = 0.01, 0.5
    pipe._partial_loop()

    assert seen == [], seen


def test_a_dying_partial_loop_still_retracts_its_ghost(monkeypatch):
    """A loop that raises is the LAST thread that should be allowed to
    leave a preview on screen -- the retraction lives in a finally."""
    seen = _capture(monkeypatch)
    state = {"n": 0}

    class Rec:
        recording = True

        def snapshot_audio(self):
            state["n"] += 1
            if state["n"] > 1:
                raise RuntimeError("mic buffer exploded")
            return np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)

    class Tr:
        def partial(self, audio):
            return "two"

    pipe = object.__new__(_pipeline_class())
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S = 0.01, 0.5
    pipe._partial_loop()                       # must not raise

    assert seen == ["two", ""], seen


# ------------------------------------------------ what the card will draw
#
# The 20:47 screenshot: a card holding "2,000,000,000,000,000,000,..." --
# 100+ digits, one unbreakable "word" that Tk's word wrap cannot split, so
# it also spilled past the card edge. It came from Transcriber.partial()
# on room noise, not from the number formatter or the interval parser
# (nothing on the reply path produced that string), so the card is where
# it has to be stopped.

def test_the_ghost_card_refuses_the_20_47_number():
    from jarvis.ui.views import PARTIAL_MAX_CHARS, partial_display_text

    assert partial_display_text("2,000,000,000,000,000,000,000,000") == ""
    assert partial_display_text("2" * 120) == ""
    assert partial_display_text("It is 2,000,000,000,000 sir") == ""
    # ...his other artifact: a preview with no word in it at all
    assert partial_display_text("2-") == ""
    assert partial_display_text("...") == ""
    assert partial_display_text("  ") == ""
    assert partial_display_text(None) == ""
    # ...while everything a half-said command looks like still previews
    assert partial_display_text("set a reminder to drink water every 45") \
        == "set a reminder to drink water every 45"
    assert partial_display_text("set a timer for 2") == "set a timer for 2"
    assert partial_display_text("call me on 555 1234") == "call me on 555 1234"
    # a runaway preview keeps its TAIL: the newest words are the point
    long = partial_display_text("word " * 200)
    assert len(long) == PARTIAL_MAX_CHARS and long.startswith("…")
    assert long.endswith("word")


def test_empty_partial_text_takes_the_card_down():
    """PartialText("") is the producer's retraction (app._partial_loop
    publishes one when the mic shuts). show_partial used to `return` on
    empty text, which is why nothing could ever take the ghost down."""
    from jarvis.ui.views import TranscriptView

    class Fake:
        def __init__(self):
            self.cleared = 0

        def clear_partial(self):
            self.cleared += 1

    fake = Fake()
    TranscriptView.show_partial(fake, "")
    TranscriptView.show_partial(fake, "2-")          # noise clears it too
    TranscriptView.show_partial(fake, "2,000,000,000,000")
    assert fake.cleared == 3
