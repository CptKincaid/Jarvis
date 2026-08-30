"""Speech endpointing (jarvis/endpoint.py) and its use by the recorder.

The recorder ended every capture with a 2.5 s energy timer, so every turn
paid 2.5 s of dead air before transcription began. The endpointer stops
CONFIG.endpoint_silence after the last speech chunk instead. These tests
drive it with a scripted probability model; one smoke test uses the real
Silero file when it is present.
"""
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import jarvis.endpoint as ep_mod
import jarvis.recorder as recorder_mod
from jarvis.endpoint import CHUNK, SAMPLE_RATE, VoiceEndpointer
from jarvis.recorder import Recorder


class ScriptedVAD:
    """probability per 512-sample chunk, from a list; last value repeats."""

    def __init__(self, probs):
        self.probs, self.i, self.resets = list(probs), 0, 0

    def __call__(self, chunk):
        assert chunk.shape == (CHUNK,), chunk.shape
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p

    def reset(self):
        self.resets += 1
        self.i = 0


def _chunks(n, rate=SAMPLE_RATE):
    """n chunks' worth of samples at `rate` (any content: the VAD is scripted)."""
    return np.zeros(int(n * CHUNK * rate / SAMPLE_RATE), dtype=np.float32)


def test_silence_then_speech_then_silence():
    vad = ScriptedVAD([0.0] * 10 + [0.9] * 20 + [0.0] * 100)
    e = VoiceEndpointer(model=vad)
    e.feed(_chunks(10), SAMPLE_RATE)
    assert not e.speech_started and e.silence_since_speech is None
    e.feed(_chunks(20), SAMPLE_RATE)
    assert e.speech_started and e.silence_since_speech == 0.0
    e.feed(_chunks(25), SAMPLE_RATE)                  # 25 * 32 ms = 0.8 s
    assert abs(e.silence_since_speech - 0.8) < 1e-9
    assert abs(e.last_speech_seconds - 30 * CHUNK / SAMPLE_RATE) < 1e-9


def test_a_blip_shorter_than_min_speech_does_not_start_speech():
    """The tail of the wake word or a click must not count as the question
    having begun -- otherwise a thinking pause right after "Jarvis" ends the
    capture before a word was said."""
    vad = ScriptedVAD([0.0] * 5 + [0.9] * (ep_mod.MIN_SPEECH_CHUNKS - 1) + [0.0] * 60)
    e = VoiceEndpointer(model=vad)
    e.feed(_chunks(70), SAMPLE_RATE)
    assert not e.speech_started and e.silence_since_speech is None


def test_hysteresis_holds_speech_through_a_borderline_dip():
    vad = ScriptedVAD([0.9] * 6 + [0.4] * 3 + [0.9] * 6 + [0.0] * 10)
    e = VoiceEndpointer(model=vad)
    e.feed(_chunks(25), SAMPLE_RATE)
    # 0.4 is above SPEECH_OFF: the run never broke, so the last speech chunk
    # is the 15th and 10 chunks of silence follow
    assert abs(e.silence_since_speech - 10 * CHUNK / SAMPLE_RATE) < 1e-9


def test_native_rate_audio_is_resampled_and_remainders_carry_over():
    vad = ScriptedVAD([0.9] * 1000)
    e = VoiceEndpointer(model=vad)
    # 44.1 kHz frames of 0.1 s, as the recorder captures them
    for _ in range(10):
        e.feed(np.zeros(4410, dtype=np.float32), 44100)
    # 1.0 s at 16 kHz is 16000 samples, less the resampler's 10 ms hold-back
    held = e._resample._hold * 160 // 441
    emitted = 16000 - held
    assert vad.i == emitted // CHUNK and e._pending.size == emitted % CHUNK
    assert abs(e.audio_seconds - vad.i * CHUNK / SAMPLE_RATE) < 1e-9


def test_reset_clears_state_and_the_model():
    vad = ScriptedVAD([0.9] * 50)
    e = VoiceEndpointer(model=vad)
    e.feed(_chunks(20), SAMPLE_RATE)
    assert e.speech_started
    before = vad.resets                               # __init__ resets once too
    e.reset()
    assert not e.speech_started and e.silence_since_speech is None
    assert e.audio_seconds == 0.0 and vad.resets == before + 1


def test_a_missing_model_file_fails_soft():
    e = VoiceEndpointer(model_path=Path("/nonexistent/silero.jit"))
    assert not e.available
    assert e.warm() is False
    e.feed(_chunks(5), SAMPLE_RATE)                   # no raise
    assert e.silence_since_speech is None


# ------------------------------------------------------------ the recorder
def _recorder(endpointer, rate=16000):
    rec = object.__new__(Recorder)
    rec.recording = True
    rec.endpointer = endpointer
    rec._ep_cursor = 0
    rec._record_rate = rate
    rec._record_start_time = time.monotonic() - 10.0   # well past the grace
    rec._audio_frames = []
    rec._stop_endpoint, rec._stop_dead_air = "", None
    rec._voice_stopped = False
    rec.stops = []

    def stop(reason="manual", endpoint="", dead_air=None):
        rec.stops.append((reason, endpoint, dead_air))
        rec.recording = False
    rec.stop = stop
    return rec


def _push(rec, n_chunks):
    rec._audio_frames.append(_chunks(n_chunks).reshape(-1, 1))


def test_recorder_stops_endpoint_silence_after_the_last_word(monkeypatch):
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_silence", 0.8)
    monkeypatch.setattr(recorder_mod.CONFIG, "silence_grace", 0.0)
    vad = ScriptedVAD([0.9] * 30 + [0.0] * 100)
    rec = _recorder(VoiceEndpointer(model=vad))
    _push(rec, 30)                                    # ~1 s of speech
    assert rec._check_endpoint() is False
    _push(rec, 20)                                    # 0.64 s silence: not yet
    assert rec._check_endpoint() is False
    _push(rec, 6)                                     # 0.83 s: stop
    assert rec._check_endpoint() is True
    (reason, endpoint, dead_air), = rec.stops
    assert (reason, endpoint) == ("silence", "vad") and dead_air >= 0.8


def test_recorder_never_stops_before_speech_was_heard(monkeypatch):
    """Nobody spoke: that is the energy timer's case, not this one."""
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(recorder_mod.CONFIG, "silence_grace", 0.0)
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD([0.0] * 500)))
    _push(rec, 200)                                   # 6.4 s of nothing
    assert rec._check_endpoint() is False and rec.stops == []


def test_recorder_honours_the_grace_period(monkeypatch):
    """The tail of "Jarvis" is speech; a thinking pause after it must not
    end the capture before the question has begun."""
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_silence", 0.8)
    monkeypatch.setattr(recorder_mod.CONFIG, "silence_grace", 1.5)
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD([0.9] * 6 + [0.0] * 100)))
    rec._record_start_time = time.monotonic()         # capture just opened
    _push(rec, 40)                                    # 0.19 s speech + 1.1 s silence
    assert rec._check_endpoint() is False and rec.stops == []
    # ...and the same pause AFTER the grace does end it: the grace is a
    # window, not a notion of whether the question has begun
    rec._record_start_time = time.monotonic() - 1.6
    assert rec._check_endpoint() is True and rec.stops[0][1] == "vad"


def test_recorder_feeds_only_new_frames(monkeypatch):
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    vad = ScriptedVAD([0.0] * 1000)
    rec = _recorder(VoiceEndpointer(model=vad))
    _push(rec, 10)
    rec._check_endpoint()
    _push(rec, 10)
    rec._check_endpoint()
    rec._check_endpoint()                             # nothing new
    assert vad.i == 20


def test_recorder_falls_back_to_the_energy_timer_when_the_vad_breaks(monkeypatch):
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)

    class Broken:
        def feed(self, audio, rate):
            raise RuntimeError("torch exploded")
        silence_since_speech = None
    rec = _recorder(Broken())
    _push(rec, 10)
    assert rec._check_endpoint() is False
    assert rec.endpointer is None, "a broken VAD must be dropped, not retried at 12 Hz"


def test_disabled_by_config(monkeypatch):
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", False)
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD([0.9] * 30 + [0.0] * 100)))
    _push(rec, 130)
    assert rec._check_endpoint() is False and rec.stops == []


# ------------------------------------------------------------ the real model
REAL = ep_mod.MODEL_FILE
CLIP = Path.home() / "voice-training" / "blind7" / "clip01.wav"


@pytest.mark.skipif(not REAL.exists() or not CLIP.exists(),
                    reason="silero file or a speech clip is missing")
def test_the_real_model_hears_speech_and_not_room_tone():
    import soundfile as sf
    e = VoiceEndpointer()
    assert e.warm()
    wav, sr = sf.read(str(CLIP), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    e.feed(wav, sr)
    assert e.speech_started, "real speech not detected"
    spoken_until = e.last_speech_seconds
    assert spoken_until > 0.3
    e.reset()
    rng = np.random.default_rng(0)
    e.feed(rng.normal(0, 0.003, SAMPLE_RATE * 3).astype(np.float32), SAMPLE_RATE)
    assert not e.speech_started, "room tone counted as speech"


@pytest.mark.skipif(not REAL.exists(), reason="silero file missing")
def test_the_real_model_is_cheap_enough_to_run_in_the_poll_loop():
    e = VoiceEndpointer()
    assert e.warm()
    chunk = np.zeros(CHUNK, dtype=np.float32)
    e.feed(chunk, SAMPLE_RATE)
    t0 = time.perf_counter()
    for _ in range(100):
        e.feed(chunk, SAMPLE_RATE)
    per_chunk_ms = (time.perf_counter() - t0) / 100 * 1000
    # a 0.1 s frame is ~3 chunks; the poll tick is 83 ms
    assert per_chunk_ms * 3 < 40, f"{per_chunk_ms:.1f} ms per chunk is too slow for the poll loop"


def test_warm_is_safe_from_several_threads():
    e = VoiceEndpointer(model_path=Path("/nonexistent/silero.jit"))
    ts = [threading.Thread(target=e.warm) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert e._failed


def test_resampling_is_continuous_across_frames():
    """Frame-by-frame resample_poly leaves a discontinuity at every frame
    edge -- a 10 Hz click train under the signal, which is noise to a VAD
    deciding at 0.35/0.5. With the carry, chunked output matches one long
    resample everywhere but the very ends."""
    from scipy.signal import resample_poly
    rate = 44100
    t = np.arange(int(rate * 1.0)) / rate
    x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    whole = resample_poly(x, 160, 441).astype(np.float32)
    r = ep_mod._resampler(rate)
    parts = [r(x[i:i + 4410]) for i in range(0, len(x), 4410)]
    chunked = np.concatenate(parts)
    n = min(len(whole), len(chunked)) - 200
    err = float(np.max(np.abs(whole[100:n] - chunked[100:n])))
    naive = np.concatenate([resample_poly(x[i:i + 4410], 160, 441) for i in range(0, len(x), 4410)]).astype(np.float32)
    naive_err = float(np.max(np.abs(whole[100:n] - naive[100:n])))
    assert err < 1e-3, f"carry resampler drifts from a whole resample by {err}"
    assert naive_err > 10 * err, "the frame-by-frame path was not actually worse"


def test_describe_reports_the_vads_view():
    vad = ScriptedVAD([0.9] * 20 + [0.2] * 10)
    e = VoiceEndpointer(model=vad)
    e.feed(_chunks(30), SAMPLE_RATE)
    d = e.describe()
    assert "started=True" in d and "last_p=0.20" in d and "max_p=0.90" in d
    assert "gap=0.32s" in d


def test_the_energy_timer_stands_down_while_the_vad_hears_speech(monkeypatch):
    """On a real capture the user's RMS sat at 0.008-0.014 against a 0.015
    threshold for a whole clause while the VAD held 0.97-1.00: the energy
    timer counted the clause as silence and ended the capture mid-sentence."""
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_silence", 0.8)
    monkeypatch.setattr(recorder_mod.CONFIG, "silence_timeout", 2.5)
    monkeypatch.setattr(recorder_mod.CONFIG, "silence_grace", 0.0)

    class Vad:
        silence_since_speech = 0.3          # the user is (still) talking
        def describe(self):
            return "vad"
    rec = _recorder(Vad())
    for _ in range(10):                             # > min_frames of 0.1 s blocks
        _push(rec, 3)
    rec._silence_start = time.monotonic() - 5.0     # energy: "quiet for 5 s"
    rec._loud_chunks = 0
    assert rec._check_silence() is False and rec.stops == []
    Vad.silence_since_speech = None                 # nobody has spoken: energy's case
    assert rec._check_silence() is True and rec.stops[0][:2] == ("silence", "energy")
