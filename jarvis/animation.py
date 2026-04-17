"""AnimationRenderer — orbit reactor animation + beep generation.

Extracted from VoiceInputGUI. Pure numpy; no GUI coupling.
"""

import numpy as np


def generate_beep(
    duration: float = 0.1,
    sample_rate: int = 16000,
    frequency: float = 880.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine-wave beep as float32 PCM samples."""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class AnimationRenderer:
    """Renders the Jarvis arc-reactor orbit as an RGB numpy frame."""

    def __init__(self, size: int = 512):
        self.size = size

    def _base_frame(self) -> np.ndarray:
        """Create an empty dark-blue frame."""
        frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        frame[..., 2] = 8  # very faint blue baseline so frames are never all-zero
        return frame

    def render_active(self, amplitude: float) -> np.ndarray:
        """Render a speaking frame scaled by amplitude (0.0-1.0)."""
        frame = self._base_frame()
        cx = cy = self.size // 2
        amp = max(0.0, min(1.0, amplitude))
        radius = int(self.size * (0.15 + 0.25 * amp))
        intensity = int(80 + 175 * amp)
        self._paint_disc(frame, cx, cy, radius, intensity)
        return frame

    def render_idle(self, t: float) -> np.ndarray:
        """Render a breathing idle frame. t is seconds since app start."""
        amp = 0.3 + 0.2 * np.sin(t * 2.0)
        return self.render_active(amplitude=amp)

    def _paint_disc(self, frame, cx, cy, radius, intensity):
        """In-place blue-cyan disc around (cx, cy)."""
        y, x = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist < radius
        falloff = np.clip(1.0 - dist / max(radius, 1), 0.0, 1.0)
        frame[..., 1] = np.where(mask,
            np.clip(frame[..., 1] + (falloff * intensity * 0.6), 0, 255),
            frame[..., 1]).astype(np.uint8)
        frame[..., 2] = np.where(mask,
            np.clip(frame[..., 2] + (falloff * intensity), 0, 255),
            frame[..., 2]).astype(np.uint8)
