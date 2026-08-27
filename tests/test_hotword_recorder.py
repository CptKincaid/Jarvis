"""Static-path tests for jarvis.hotword and jarvis.recorder that need no
microphone: the mic arbiter, the no-mic recorder, hotword pause/resume,
and the OpenWakeWord custom-verifier shim."""
import numpy as np
import pytest

from jarvis import hotword as hw
from jarvis.config import MACHINE
from jarvis.events import RecordingStarted, Status, bus
from jarvis.recorder import MicArbiter, Recorder


# ------------------------------------------------------------ arbiter
def test_arbiter_pauses_once_for_nested_acquires():
    calls = []
    arb = MicArbiter()
    arb.register_hotword(lambda: calls.append("pause"),
                         lambda: calls.append("resume"))
    with arb.acquire("recorder"):
        assert arb.held_by == "recorder"
        with arb.acquire("calibrate"):
            assert arb.held_by == "recorder"          # outer owner kept
        assert calls == ["pause"]
    assert calls == ["pause", "resume"]
    assert arb.held_by == ""


def test_arbiter_survives_failing_callbacks():
    arb = MicArbiter()

    def boom():
        raise RuntimeError("x")
    arb.register_hotword(boom, boom)
    with arb.acquire("x"):
        pass
    assert arb.held_by == ""


# ----------------------------------------------------------- recorder
def test_recorder_without_mic_is_a_graceful_noop(monkeypatch):
    monkeypatch.setattr(MACHINE, "has_mic", False)
    events = []
    f1 = bus.subscribe(Status, events.append)
    f2 = bus.subscribe(RecordingStarted, events.append)
    try:
        rec = Recorder(MicArbiter())
        assert rec.mic_available is False
        rec.start()
        assert rec.recording is False
        assert any(isinstance(e, Status) and "No microphone" in e.text
                   for e in events)
        assert not any(isinstance(e, RecordingStarted) for e in events)
        assert len(rec.record_fixed(0.1)) == 0
        assert rec.stop() is None
        assert rec.calibrate_noise() == rec.calibrate_noise()
    finally:
        bus.unsubscribe(Status, f1)
        bus.unsubscribe(RecordingStarted, f2)


def test_recorder_default_mic_always_listed():
    rec = Recorder(MicArbiter())
    assert "Default" in rec.mic_devices and rec.mic_devices["Default"] is None


# ------------------------------------------------------------ hotword
class FakeArbiter:
    def __init__(self):
        self.cbs = None

    def register_hotword(self, pause, resume):
        self.cbs = (pause, resume)


def test_hotword_registers_with_arbiter_and_pause_resume():
    arb = FakeArbiter()
    h = hw.Hotword(arb, lambda: None, lambda score: None)
    assert arb.cbs == (h.pause, h.resume)

    class Stream:
        closed = False

        def stop(self):
            pass

        def close(self):
            Stream.closed = True
    h.active = True
    h._stream = Stream()
    h.pause()
    assert h._paused and h._stream is None and Stream.closed
    h.resume()
    assert not h._paused and h._reopen is True       # loop reopens the stream
    h.stop()
    assert h.active is False


# ---------------------------------------------- custom-verifier shim
class FakeScaler:
    n_features_in_ = 1536


class FakePipeline:
    steps = [("functiontransformer", object()), ("standardscaler", FakeScaler())]

    def __init__(self):
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(np.asarray(X).shape)
        return np.array([[0.2, 0.8]])


class FakeModel:
    def __init__(self):
        self.model_inputs = {"hey_jarvis": 16, "timer": 34}
        self.custom_verifier_models = {}
        self.custom_verifier_threshold = 0.1


def test_verifier_feature_count():
    assert hw.verifier_feature_count(FakePipeline()) == 1536
    assert hw.verifier_feature_count(object()) is None


def test_install_verifier_accepts_matching_size():
    model, pipe = FakeModel(), FakePipeline()
    ok, detail = hw.install_verifier(model, pipe)
    assert ok, detail
    assert model.custom_verifier_threshold == 0.3
    shim = model.custom_verifier_models["hey_jarvis"]
    # Correct size (hey_jarvis: 16 frames x 96) reaches the real verifier.
    out = shim.predict_proba(np.zeros((1, 16, 96)))
    assert out[0][-1] == pytest.approx(0.8) and pipe.calls == [(1, 16, 96)]
    # oww 0.4.0 re-applies the verifier with the *timer* model's 34 frames:
    # the shim answers with the last valid probability instead of raising.
    out = shim.predict_proba(np.zeros((1, 34, 96)))
    assert out[0][-1] == pytest.approx(0.8) and len(pipe.calls) == 1
    assert shim.skipped == 1


def test_install_verifier_rejects_mismatched_size():
    model = FakeModel()
    model.model_inputs["hey_jarvis"] = 34            # a different base model
    ok, detail = hw.install_verifier(model, FakePipeline())
    assert not ok and "1536" in detail and "3264" in detail
    assert model.custom_verifier_models == {}


def test_shim_before_any_valid_call_is_neutral():
    shim = hw.VerifierShim(FakePipeline(), 1536)
    out = shim.predict_proba(np.zeros((1, 34, 96)))
    assert out[0][-1] == 0.0                          # never inflates a score


# ------------------------------------------------- release ordering
def test_stop_releases_the_mic_only_after_finalising(monkeypatch):
    """stop() used to hand the mic back BEFORE finalising the audio.

    The log from 2026-08-27 shows the consequence: "Hotword stream resumed"
    at 58.229 but "Stopped: 28.2s audio" at 58.397 -- the wake word was live
    while the clip it had just captured was still being assembled. Releasing
    last keeps the ordering honest.
    """
    import numpy as np
    from jarvis.config import CONFIG

    order = []

    class Ctx:
        def __exit__(self, *a):
            order.append("release")
            return False

    monkeypatch.setattr(CONFIG, "sound", False)
    rec = Recorder(MicArbiter(), speaker_verifier=None)
    rec.recording = True
    rec._session_ctx = Ctx()
    monkeypatch.setattr(rec, "_join_poll_thread", lambda: None)
    monkeypatch.setattr(rec, "_finalize_audio",
                        lambda: (order.append("finalize"),
                                 np.zeros(16, dtype=np.float32))[1])

    rec.stop()

    assert order == ["finalize", "release"], order
