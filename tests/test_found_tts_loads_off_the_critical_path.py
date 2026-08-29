"""Regression test: the voice model loaded last, behind everything else.

DEFECT (jarvis/app.py _load_models):

The startup warmer ran strictly in sequence:

    _PRELOAD_DONE.wait() -> brain.warmup() -> transcriber.load()
                         -> tts.load()     -> speaker -> prewarm

XTTS is the ONLY model on the critical path between "the reply exists" and
"the user hears something", and it was fourth -- behind an ollama warmup that
takes 5.5 s to make gemma4:26b resident. From /tmp/vss_voice/jarvis.log,
2026-08-28, on a box whose GPU had already been un-wedged:

    16:41:17.2  app up
    16:41:23.5  torch/whisper/CUDA preloaded
    16:41:28.9  ollama: gemma4:26b resident (load 9.0 s)   <- tts still waiting
    16:41:29.8  whisper on GPU
    16:41:33.7  user asks "What's on my calendar for Monday?"
    16:41:35.2  reply ready
    16:41:42.0  XTTS loaded -- 12 s after the app was otherwise ready
    16:41:46.3  speaking

The voice model had not even begun loading when the user spoke. Nothing in
the warmup depends on ordering, so the one model the user waits on should not
queue behind the ones they do not. Loading it concurrently is safe now that
TTS.load() is serialised (test_found_double_xtts_load).
"""
import threading
import types

import pytest

from jarvis.app import JarvisApp


def _stub_app(events):
    """A minimal stand-in exposing only what _load_models touches."""
    def warmup():
        events["warmup_started"].set()
        # hold the warmup open so anything that queues behind it cannot finish
        assert events["release_warmup"].wait(10), "test never released warmup"

    def tts_load():
        events["tts_started"].set()
        return True

    app = types.SimpleNamespace(
        brain=types.SimpleNamespace(warmup=warmup),
        transcriber=types.SimpleNamespace(load=lambda: "cuda"),
        tts=types.SimpleNamespace(load=tts_load,
                                  prewarm=lambda phrases: None),
        speaker=types.SimpleNamespace(enrolled=False,
                                      load_model=lambda: None),
        _canned_phrases=lambda: [],
    )
    # the real helper, bound to the stub, so the thread body under test is
    # production's and not a reimplementation of it
    app._load_tts = lambda: JarvisApp._load_tts(app)
    return app


@pytest.fixture
def events():
    return {k: threading.Event() for k in
            ("warmup_started", "release_warmup", "tts_started")}


def test_the_voice_model_does_not_queue_behind_the_ollama_warmup(events):
    app = _stub_app(events)
    worker = threading.Thread(target=JarvisApp._load_models, args=(app,),
                              daemon=True)
    worker.start()

    assert events["warmup_started"].wait(5), "_load_models never ran"
    # The warmup is still blocked here. If the voice model is gated behind it,
    # this wait times out -- which is exactly the 12 s the user sat through.
    started = events["tts_started"].wait(5)
    events["release_warmup"].set()
    worker.join(timeout=15)

    assert started, (
        "XTTS did not start loading until the ollama warmup finished; it is "
        "the only model between the reply and the first audible word")


def test_the_warmup_still_completes_everything(events):
    app = _stub_app(events)
    events["release_warmup"].set()          # let it run straight through
    JarvisApp._load_models(app)
    assert events["warmup_started"].is_set()
    assert events["tts_started"].is_set()
