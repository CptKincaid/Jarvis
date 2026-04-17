"""Tests for AnimationRenderer."""

import numpy as np

from jarvis.animation import AnimationRenderer, generate_beep


def test_render_active_returns_rgb_frame():
    r = AnimationRenderer(size=128)
    frame = r.render_active(amplitude=0.5)
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8


def test_render_idle_returns_rgb_frame():
    r = AnimationRenderer(size=128)
    frame = r.render_idle(t=0.0)
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8


def test_render_active_zero_amplitude_not_blank():
    r = AnimationRenderer(size=128)
    frame = r.render_active(amplitude=0.0)
    assert frame.sum() > 0  # Base glow is always visible


def test_generate_beep_has_correct_length():
    samples = generate_beep(duration=0.1, sample_rate=16000)
    assert len(samples) == 1600


def test_generate_beep_peaks_at_expected_amplitude():
    samples = generate_beep(duration=0.1, sample_rate=16000, amplitude=0.5)
    assert 0.4 < np.max(np.abs(samples)) <= 0.5
