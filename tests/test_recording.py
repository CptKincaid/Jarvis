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
