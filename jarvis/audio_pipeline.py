"""Audio pipeline utilities for Jarvis voice assistant.

Provides mic resolution, noise suppression, and VAD — the foundation
audio processing that all other voice components depend on.
"""

import numpy as np


def resolve_mic_by_name(saved_name, devices=None):
    """Resolve a mic device by name substring (case-insensitive).

    Args:
        saved_name: Saved device name to search for (e.g. "Blue Snowball").
        devices: List of device dicts from sounddevice.query_devices().
                 If None, queries live devices.

    Returns:
        Device index (int) or None if not found.
    """
    if not saved_name or saved_name.lower() in ("default", ""):
        return None

    if devices is None:
        import sounddevice as sd
        devices = sd.query_devices()

    saved_lower = saved_name.lower()

    # Strip "[index] " prefix if present (legacy format)
    if saved_lower.startswith("["):
        bracket_end = saved_lower.find("]")
        if bracket_end > 0:
            saved_lower = saved_lower[bracket_end + 1:].strip()

    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        if saved_lower in d.get("name", "").lower():
            return i

    return None


def denoise_audio(audio, sr=16000, enabled=True):
    """Remove background noise using spectral gating.

    Args:
        audio: numpy float32 array, mono
        sr: sample rate
        enabled: if False, returns audio unchanged

    Returns:
        Denoised numpy float32 array, same shape as input.
    """
    if not enabled or len(audio) < 1600:
        return audio

    try:
        import noisereduce as nr
        reduced = nr.reduce_noise(
            y=audio.astype(np.float32), sr=sr,
            stationary=True, prop_decrease=0.75,
            n_fft=512, hop_length=128,
        )
        return reduced.astype(np.float32)
    except Exception:
        return audio
