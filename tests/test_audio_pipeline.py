import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_denoise_reduces_noise():
    from jarvis.audio_pipeline import denoise_audio
    noise = np.random.randn(16000).astype(np.float32) * 0.1
    result = denoise_audio(noise, sr=16000)
    assert result.shape == noise.shape
    assert np.sqrt(np.mean(result**2)) < np.sqrt(np.mean(noise**2))

def test_denoise_preserves_shape():
    from jarvis.audio_pipeline import denoise_audio
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    speech = np.sin(2 * np.pi * 440 * t) * 0.5
    noisy = speech + np.random.randn(16000).astype(np.float32) * 0.05
    result = denoise_audio(noisy, sr=16000)
    assert result.shape == noisy.shape
    assert result.dtype == np.float32

def test_denoise_handles_short_audio():
    from jarvis.audio_pipeline import denoise_audio
    short = np.random.randn(800).astype(np.float32) * 0.1
    result = denoise_audio(short, sr=16000)
    assert result.shape == short.shape

def test_denoise_disabled_returns_original():
    from jarvis.audio_pipeline import denoise_audio
    audio = np.random.randn(16000).astype(np.float32)
    result = denoise_audio(audio, sr=16000, enabled=False)
    np.testing.assert_array_equal(result, audio)

def test_vad_silence_detected():
    from jarvis.audio_pipeline import SileroVAD
    vad = SileroVAD()
    silence = np.zeros(512, dtype=np.float32)
    prob = vad.is_speech(silence, sr=16000)
    assert prob < 0.3

def test_vad_returns_float_in_range():
    from jarvis.audio_pipeline import SileroVAD
    vad = SileroVAD()
    audio = np.random.randn(512).astype(np.float32) * 0.01
    prob = vad.is_speech(audio, sr=16000)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_vad_reusable():
    from jarvis.audio_pipeline import SileroVAD
    vad = SileroVAD()
    silence = np.zeros(512, dtype=np.float32)
    for _ in range(5):
        prob = vad.is_speech(silence, sr=16000)
    assert prob < 0.3

def test_vad_reset():
    from jarvis.audio_pipeline import SileroVAD
    vad = SileroVAD()
    silence = np.zeros(512, dtype=np.float32)
    vad.is_speech(silence, sr=16000)
    vad.reset()
    prob = vad.is_speech(silence, sr=16000)
    assert prob < 0.3
