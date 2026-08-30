"""Speech endpointing: notice when the user has finished talking.

The recorder used to end every utterance with an energy-based silence timer
(CONFIG.silence_timeout, 2.5 s), so every turn paid 2.5 s of dead air before
transcription even began -- the single largest slice of the wait the user
feels, measured across today's log. Energy cannot be trusted below ~2 s
because breaths, mid-sentence pauses and room noise all look alike to it.

This runs Silero VAD (MIT, 2 MB TorchScript) over the capture as it arrives
and answers one question: how long since the last frame of speech? The
recorder stops at CONFIG.endpoint_silence (0.8 s) of trailing non-speech
once speech has been heard at all. The energy timer stays as the fallback
for a capture in which nobody ever speaks, and the voice-ID stop still
handles a room where someone ELSE keeps talking.

Loaded from ~/.aiws_trainer/silero_vad/silero_vad.jit with plain torch:
no new package in the shared venv (the VSS project lives in it too).
Measured on the GB10: 3.6 ms per 32 ms chunk on CPU, 11% of real time.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("endpoint")

SAMPLE_RATE = 16000
CHUNK = 512                     # what the v5 model expects at 16 kHz (32 ms)
MODEL_FILE = PATHS.AIWS / "silero_vad" / "silero_vad.jit"
SPEECH_ON = 0.5                 # probability above which a chunk is speech
SPEECH_OFF = 0.35               # ...and below which it is not (hysteresis)
MIN_SPEECH_CHUNKS = 4           # ~130 ms of speech before "speech started"


def _resampler(rate: int):
    """44100 -> 16000 by polyphase (exact ratio), cached per rate."""
    if rate == SAMPLE_RATE:
        return lambda x: x
    from scipy.signal import resample_poly
    g = math.gcd(rate, SAMPLE_RATE)
    up, down = SAMPLE_RATE // g, rate // g
    return lambda x: resample_poly(x, up, down).astype(np.float32)


class VoiceEndpointer:
    """Feed audio as it is captured; ask how long since speech.

    Not thread-safe by itself: the recorder feeds it from one poll thread.
    ``model`` is injectable (tests use a callable returning probabilities);
    the default is loaded lazily from MODEL_FILE the first time it is needed,
    off the caller's thread via ``warm()`` if the app calls it early.
    """

    def __init__(self, model=None, model_path: Optional[Path] = None):
        self._model = model
        self._path = Path(model_path) if model_path else MODEL_FILE
        self._load_lock = threading.Lock()
        self._failed = False
        self._resample = None
        self._rate = None
        self.reset()

    # ------------------------------------------------------------ loading
    @property
    def available(self) -> bool:
        return self._model is not None or (not self._failed and self._path.exists())

    def warm(self) -> bool:
        """Load the model now (call from a model-loader thread)."""
        return self._ensure_model() is not None

    def _ensure_model(self):
        if self._model is not None or self._failed:
            return self._model
        with self._load_lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                import torch
                m = torch.jit.load(str(self._path), map_location="cpu")
                m.eval()
                torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
                self._model = _TorchVAD(m)
                log.info("silero VAD loaded from %s", self._path)
            except Exception as exc:          # noqa: BLE001 - reported once
                self._failed = True
                log.warning("silero VAD unavailable (%s); endpointing falls back "
                            "to the energy timer", exc)
        return self._model

    # ------------------------------------------------------------ session
    def reset(self) -> None:
        """New capture: forget everything, including the model's RNN state."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._chunks = 0                 # 512-sample chunks consumed
        self._speech_run = 0
        self._speech_started = False
        self._last_speech_chunk: Optional[int] = None
        self._in_speech = False
        m = self._model
        if m is not None:
            try:
                m.reset()
            except Exception:
                log.debug("VAD state reset failed", exc_info=True)

    def feed(self, audio: np.ndarray, rate: int) -> None:
        """Consume newly captured samples (any length, native rate)."""
        m = self._ensure_model()
        if m is None or audio.size == 0:
            return
        if self._resample is None or self._rate != rate:
            self._resample, self._rate = _resampler(rate), rate
        x = self._resample(np.asarray(audio, dtype=np.float32).flatten())
        buf = np.concatenate([self._pending, x]) if self._pending.size else x
        n = buf.size // CHUNK
        for i in range(n):
            p = float(m(buf[i * CHUNK:(i + 1) * CHUNK]))
            self._chunks += 1
            self._step(p)
        self._pending = buf[n * CHUNK:]

    def _step(self, p: float) -> None:
        # Hysteresis: enter speech above SPEECH_ON, leave below SPEECH_OFF.
        if self._in_speech:
            self._in_speech = p >= SPEECH_OFF
        else:
            self._in_speech = p >= SPEECH_ON
        if self._in_speech:
            self._speech_run += 1
            self._last_speech_chunk = self._chunks
            if self._speech_run >= MIN_SPEECH_CHUNKS:
                self._speech_started = True
        else:
            self._speech_run = 0

    # ------------------------------------------------------------ queries
    @property
    def speech_started(self) -> bool:
        """At least MIN_SPEECH_CHUNKS of continuous speech has been heard."""
        return self._speech_started

    @property
    def audio_seconds(self) -> float:
        return self._chunks * CHUNK / SAMPLE_RATE

    @property
    def last_speech_seconds(self) -> Optional[float]:
        """Position (in capture seconds) of the last speech chunk, or None."""
        if self._last_speech_chunk is None:
            return None
        return self._last_speech_chunk * CHUNK / SAMPLE_RATE

    @property
    def silence_since_speech(self) -> Optional[float]:
        """Seconds of non-speech since the last speech chunk, or None until
        speech has started."""
        if not self._speech_started or self._last_speech_chunk is None:
            return None
        return (self._chunks - self._last_speech_chunk) * CHUNK / SAMPLE_RATE


class _TorchVAD:
    """The JIT model as a plain callable: chunk -> probability."""

    def __init__(self, model):
        import torch
        self._torch = torch
        self._m = model

    def __call__(self, chunk: np.ndarray) -> float:
        with self._torch.no_grad():
            x = self._torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None]
            return float(self._m(x, SAMPLE_RATE).item())

    def reset(self) -> None:
        self._m.reset_states()
