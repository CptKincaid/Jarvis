"""RecordingController — sounddevice stream lifecycle, audio buffering,
silence detection, noise gate. Extracted from VoiceInputGUI.

Does NOT own transcription, TTS, or GUI widgets. Communicates via
callbacks supplied by the caller.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

LOG_DIR = Path("/tmp/vss_voice")

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.04
SILENCE_TIMEOUT = 2.0
NOISE_GATE_THRESHOLD = 0.005


from jarvis.logging import get_logger
_log = get_logger("REC")


class RecordingController:
    """Push-to-talk audio recorder with silence detection and noise gate."""

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        silence_threshold=SILENCE_THRESHOLD,
        silence_timeout=SILENCE_TIMEOUT,
        noise_gate_threshold=NOISE_GATE_THRESHOLD,
        on_amplitude=None,
        on_stopped=None,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_timeout = silence_timeout
        self.noise_gate_threshold = noise_gate_threshold
        self.on_amplitude = on_amplitude
        self.on_stopped = on_stopped

        self._stream = None
        self._frames = []
        self._lock = threading.Lock()
        self._recording = False
        self._silence_start = None

    @property
    def is_recording(self):
        return self._recording

    def apply_noise_gate(self, chunk: np.ndarray) -> np.ndarray:
        """Zero-out blocks whose RMS is below the noise gate threshold."""
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        if rms < self.noise_gate_threshold:
            return np.zeros_like(chunk)
        return chunk

    def is_silent(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        return rms < self.silence_threshold

    def handle_chunk(self, chunk: np.ndarray):
        """Process one audio chunk. Buffers it, fires amplitude callback,
        and returns True iff silence timeout has been reached."""
        gated = self.apply_noise_gate(chunk)
        with self._lock:
            self._frames.append(gated)
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        if self.on_amplitude is not None:
            self.on_amplitude(min(1.0, rms * 4))
        if self.is_silent(chunk):
            now = time.monotonic()
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start >= self.silence_timeout:
                return True
        else:
            self._silence_start = None
        return False

    def start(self, mic_device=None):
        """Open the sounddevice input stream and begin buffering."""
        import sounddevice as sd

        with self._lock:
            self._frames = []
            self._recording = True
            self._silence_start = None

        def _callback(indata, frame_count, time_info, status):
            if not self._recording:
                return
            chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            silence_hit = self.handle_chunk(chunk.astype(np.float32))
            if silence_hit:
                self._recording = False
                if self.on_stopped is not None:
                    self.on_stopped(self.get_audio())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype="float32",
                device=mic_device,
                callback=_callback,
            )
            self._stream.start()
            _log(f"Recording started (device={mic_device})")
        except Exception as e:
            _log(f"Recording start error: {e}")
            self._recording = False
            raise

    def stop(self) -> np.ndarray:
        """Stop the stream and return the captured audio as a single array."""
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        audio = self.get_audio()
        if self.on_stopped is not None:
            self.on_stopped(audio)
        _log(f"Recording stopped ({len(audio)} samples)")
        return audio

    def get_audio(self) -> np.ndarray:
        """Return the buffered audio as a single concatenated float32 array."""
        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._frames)
