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


# ------------------------------------------- how long it keeps listening
#
# Measured 2026-08-27 19:46:49: a two-second question held the mic for 12.9 s
# before auto-stopping. Two delays stacked -- a hard-coded 5 s grace where the
# silence clock is erased, then an 8 s silence timeout -- giving a ~13 s floor
# on every utterance however short. The grace is now configurable and both
# defaults suit a wake-word command rather than dictation.
def _recording(monkeypatch, started_ago: float, silent_for=None):
    import jarvis.recorder as rec_mod

    clock = [1000.0]
    monkeypatch.setattr(rec_mod.time, "monotonic", lambda: clock[0])
    r = Recorder(MicArbiter(), speaker_verifier=None)
    r.recording = True
    r._record_start_time = clock[0] - started_ago
    r._silence_start = None if silent_for is None else clock[0] - silent_for
    r._audio_frames = [object()] * 50
    r.stopped_with = None
    monkeypatch.setattr(r, "stop", lambda reason="manual": setattr(r, "stopped_with", reason))
    return r


def test_defaults_suit_a_spoken_command_not_dictation():
    """The DATACLASS defaults, not the loaded CONFIG -- a user's saved
    voice_settings.json overrides them, so this pins what a fresh install gets."""
    import dataclasses
    from jarvis.config import Config
    defaults = {f.name: f.default for f in dataclasses.fields(Config)}
    assert defaults["silence_grace"] <= 2.0, "grace this long makes commands feel stuck"
    assert defaults["silence_timeout"] <= 3.0, "8 s of trailing silence was the complaint"


def test_no_auto_stop_during_the_grace_period(monkeypatch):
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "silence_grace", 1.5)
    monkeypatch.setattr(CONFIG, "silence_timeout", 2.5)
    r = _recording(monkeypatch, started_ago=1.0, silent_for=99)

    assert r._check_silence() is False
    assert r.stopped_with is None
    assert r._silence_start is None, "the grace must reset the silence clock"


def test_auto_stop_once_grace_passed_and_silence_held(monkeypatch):
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "silence_grace", 1.5)
    monkeypatch.setattr(CONFIG, "silence_timeout", 2.5)
    r = _recording(monkeypatch, started_ago=5.0, silent_for=2.6)

    assert r._check_silence() is True
    assert r.stopped_with == "silence"


def test_still_listening_when_silence_is_shorter_than_the_timeout(monkeypatch):
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "silence_grace", 1.5)
    monkeypatch.setattr(CONFIG, "silence_timeout", 2.5)
    r = _recording(monkeypatch, started_ago=5.0, silent_for=1.0)

    assert r._check_silence() is False
    assert r.stopped_with is None


def test_grace_is_configurable_not_hard_coded(monkeypatch):
    """The 5 s was a literal in _check_silence; a long grace must be reachable
    again for dictation without editing code."""
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "silence_grace", 10.0)
    monkeypatch.setattr(CONFIG, "silence_timeout", 2.5)
    r = _recording(monkeypatch, started_ago=6.0, silent_for=99)

    assert r._check_silence() is False, "grace of 10 s must still suppress at 6 s"
