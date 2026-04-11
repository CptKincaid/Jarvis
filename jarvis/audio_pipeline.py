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
