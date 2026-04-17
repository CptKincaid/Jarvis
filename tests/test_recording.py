"""Tests for RecordingController."""

import numpy as np
import pytest

from jarvis.recording import RecordingController


def _make_audio(seconds=1.0, sr=16000, freq=440.0, amp=0.3):
    """Generate a sine wave numpy array."""
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_noise_gate_zeros_quiet_block():
    rc = RecordingController(sample_rate=16000)
    quiet = np.full(1600, 0.001, dtype=np.float32)  # below 0.005 threshold
    gated = rc.apply_noise_gate(quiet)
    assert np.allclose(gated, 0.0)


def test_noise_gate_preserves_loud_block():
    rc = RecordingController(sample_rate=16000)
    loud = _make_audio(seconds=0.1, amp=0.3)
    gated = rc.apply_noise_gate(loud)
    assert np.allclose(gated, loud)


def test_detect_silence_returns_true_for_quiet_audio():
    rc = RecordingController(sample_rate=16000, silence_threshold=0.04)
    quiet = np.zeros(1600, dtype=np.float32)
    assert rc.is_silent(quiet) is True


def test_detect_silence_returns_false_for_loud_audio():
    rc = RecordingController(sample_rate=16000, silence_threshold=0.04)
    loud = _make_audio(seconds=0.1, amp=0.3)
    assert rc.is_silent(loud) is False


def test_amplitude_callback_fires_with_rms():
    amps = []
    rc = RecordingController(
        sample_rate=16000,
        on_amplitude=lambda a: amps.append(a),
    )
    rc.handle_chunk(_make_audio(seconds=0.1, amp=0.5))
    assert len(amps) == 1
    assert 0.0 < amps[0] <= 1.0


def test_get_audio_returns_empty_when_no_chunks_buffered():
    rc = RecordingController()
    audio = rc.get_audio()
    assert audio.shape == (0,)
    assert audio.dtype == np.float32


def test_get_audio_concatenates_buffered_chunks():
    rc = RecordingController()
    rc.handle_chunk(np.ones(100, dtype=np.float32) * 0.3)
    rc.handle_chunk(np.ones(200, dtype=np.float32) * 0.3)
    rc.handle_chunk(np.ones(150, dtype=np.float32) * 0.3)
    audio = rc.get_audio()
    # Note: first chunk gets noise-gated to zero if below threshold.
    # 0.3 RMS is well above the noise gate so it stays.
    assert len(audio) == 100 + 200 + 150


def test_silence_timeout_triggers_after_sustained_quiet():
    """Silence beyond the timeout should signal stop via handle_chunk's return."""
    import time as real_time

    rc = RecordingController(silence_threshold=0.04, silence_timeout=0.05)
    # First quiet chunk arms the silence timer
    assert rc.handle_chunk(np.zeros(100, dtype=np.float32)) is False
    # Wait past timeout and deliver another quiet chunk
    real_time.sleep(0.1)
    assert rc.handle_chunk(np.zeros(100, dtype=np.float32)) is True


def test_loud_audio_resets_silence_timer():
    import time as real_time

    rc = RecordingController(silence_threshold=0.04, silence_timeout=0.05)
    rc.handle_chunk(np.zeros(100, dtype=np.float32))  # start timer
    real_time.sleep(0.04)
    # Loud chunk resets timer
    rc.handle_chunk(_make_audio(seconds=0.01, amp=0.5))
    # Another quiet chunk — timer was just reset, should NOT hit timeout
    assert rc.handle_chunk(np.zeros(100, dtype=np.float32)) is False


def test_denoise_not_called_during_chunks():
    """Regression: denoise must never run inside the audio-callback path."""
    calls = []
    rc = RecordingController(
        apply_denoise=True,
        denoise_fn=lambda audio, sr: calls.append("denoise") or audio,
    )
    rc.handle_chunk(_make_audio(seconds=0.1, amp=0.3))
    rc.handle_chunk(_make_audio(seconds=0.1, amp=0.3))
    assert calls == []


def test_is_recording_starts_false():
    rc = RecordingController()
    assert rc.is_recording is False
