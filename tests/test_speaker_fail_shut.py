"""Speaker verification must not fail OPEN once it is configured.

`verify()` and `filter_segments()` originally accepted the audio on every
error path -- no voiceprint, no model, unextractable embedding -- and logged a
single warning per session. That is the right default for an UNCONFIGURED
system: a fresh install with nothing enrolled must not be mute.

It is the wrong behaviour once a voiceprint exists. At that point the user has
explicitly asked for other voices to be filtered out, and silently passing
every voice because the model failed to load is precisely the failure mode
that let a television reach the commander.

So the policy splits on configuration:
  - nothing enrolled  -> fail OPEN  (unconfigured; accept, warn)
  - voiceprint exists -> fail SHUT  (configured but broken; reject, error)

Typed input is unaffected by either, so a broken model can never lock the user
out of Jarvis entirely.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis.speaker import SpeakerVerifier


@pytest.fixture
def audio():
    rng = np.random.default_rng(0)
    return (rng.standard_normal(16000 * 4) * 0.05).astype(np.float32)


def make(enrolled: bool, threshold: float = 0.4) -> SpeakerVerifier:
    v = SpeakerVerifier(gpu=0, threshold=threshold)
    v._model_loaded = True
    if enrolled:
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = 1.0
        v._embeddings = [emb]
        v._recompute_centroid()
    return v


# ------------------------------------------------------- unconfigured
def test_unconfigured_verify_still_fails_open(audio):
    """No voiceprint: accept, or a fresh install has no working voice path."""
    v = make(enrolled=False)
    ok, score = v.verify(audio)
    assert ok is True and score == 1.0


def test_unconfigured_filter_passes_audio_through(audio):
    v = make(enrolled=False)
    filtered, _ = v.filter_segments(audio)
    assert filtered is not None
    assert len(filtered) == len(audio)


# --------------------------------------------------- configured, broken
def test_enrolled_but_embedding_fails_now_rejects(audio, monkeypatch):
    """The regression: a broken model used to admit every voice."""
    v = make(enrolled=True)
    monkeypatch.setattr(v, "_extract_embedding", lambda a: None)

    ok, score = v.verify(audio)

    assert ok is False, "a configured verifier must not accept on failure"
    assert score == 0.0


def test_enrolled_but_model_CANNOT_load_drops_everything(audio, monkeypatch):
    v = make(enrolled=True)
    v._model_loaded = False
    monkeypatch.setattr(v, "load_model", lambda: False)

    filtered, stats = v.filter_segments(audio)

    assert filtered is None, "configured-but-broken must drop, not pass through"


def test_model_merely_not_loaded_YET_is_loaded_on_demand(audio, monkeypatch):
    """Fail-shut must not block the user during boot.

    The app warms the speaker model on its model-loader thread, but audio can
    arrive first. Rejecting because the load had not happened yet would make
    voice dead for the first ~20 s of every session, which is indistinguishable
    from the feature being broken.
    """
    v = make(enrolled=True)
    v._model_loaded = False
    loads = []

    def fake_load():
        loads.append(1)
        v._model_loaded = True
        return True

    monkeypatch.setattr(v, "load_model", fake_load)
    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0
    monkeypatch.setattr(v, "_extract_embedding", lambda a: emb)

    filtered, _ = v.filter_segments(audio)

    assert loads, "should have attempted the load rather than rejecting"
    assert filtered is not None, "must not fail shut merely because it was cold"


def test_a_failed_load_is_not_retried_every_utterance(audio, monkeypatch):
    """Retrying a broken load costs seconds on every rejection."""
    v = make(enrolled=True)
    v._model_loaded = False
    attempts = []
    monkeypatch.setattr(v, "load_model", lambda: (attempts.append(1), False)[1])

    for _ in range(3):
        v.score(audio)

    assert len(attempts) == 1, f"retried {len(attempts)} times"


def test_error_mid_filter_drops_everything(audio, monkeypatch):
    def boom(chunk):
        raise RuntimeError("cuda gone")

    v = make(enrolled=True)
    monkeypatch.setattr(v, "_extract_embedding", boom)

    filtered, _ = v.filter_segments(audio)

    assert filtered is None


# ------------------------------------------------------------- score()
def test_score_returns_none_when_it_cannot_be_computed(audio, monkeypatch):
    """Callers that want their own fallback need to distinguish 'not you'
    from 'could not tell' -- verify() collapses both into False."""
    v = make(enrolled=True)
    monkeypatch.setattr(v, "_extract_embedding", lambda a: None)
    assert v.score(audio) is None


def test_score_returns_similarity_without_applying_a_policy(audio, monkeypatch):
    v = make(enrolled=True, threshold=0.99)
    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0
    monkeypatch.setattr(v, "_extract_embedding", lambda a: emb)

    s = v.score(audio)

    assert s == pytest.approx(1.0, abs=1e-5), "identical embedding -> similarity 1"
    # ...and it made no accept/reject decision of its own, despite the
    # threshold being set above what verify() would have accepted.
    assert isinstance(s, float)


def test_score_is_none_when_nothing_enrolled(audio):
    assert make(enrolled=False).score(audio) is None
