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

# Filler sounds that mean "still thinking". Pure fillers ONLY -- never
# common words like "like", "so", "well", which end real sentences. The one
# list: jarvis.commander imports this name (it owned a copy nothing read).
# Read by trailing_filler() below, which the recorder's filler hold uses to
# decide that the newest partial decode ended on one of these.
FILLER_WORDS = frozenset({
    "uh", "um", "uhh", "umm", "hmm", "hm", "er", "ah", "ehh", "eh",
    "erm", "uhhh", "ummm",
})
# What whisper hangs on a filler: "um," "uh..." "Hmm?" "um -" (the dash and
# the dots can also arrive as their own token, which is why the loop below
# skips a token that strips to nothing).
_EDGE_PUNCT = ".,;:!?…-—–'\"()[]"


def trailing_filler(text) -> str:
    """The last word of ``text`` when it is a filler, else ''.

    Punctuation and case are stripped from the edges of the token, so
    'um,' 'uh...' and 'Hmm?' all count; a final token that is ONLY
    punctuation ('for um ...') is skipped so the word before it is judged.
    Interior punctuation is kept: 'uh-huh' is an answer, not a filler.
    """
    if not text:
        return ""
    for tok in reversed(str(text).split()):
        word = tok.strip(_EDGE_PUNCT).lower()
        if not word:
            continue
        return word if word in FILLER_WORDS else ""
    return ""


class _Resampler:
    """native rate -> 16 kHz by polyphase, continuous across calls.

    resample_poly has no filter state, so resampling each 0.1 s frame on its
    own leaves a discontinuity at every frame edge -- a 10 Hz click train
    under the signal, noise to a VAD deciding at 0.35/0.5. Both edges of a
    polyphase output are unreliable for half the filter length, so each call
    resamples [context + new frame], skips the context's share of the output
    (emitted last time) and holds back the last HOLD native samples (emitted
    next time, once they have a right-hand context). The interior is then
    exactly what one long resample would give; the price is 10 ms of lag.
    """

    def __init__(self, rate: int):
        from scipy.signal import resample_poly
        self._rp = resample_poly
        g = math.gcd(rate, SAMPLE_RATE)
        self.up, self.down = SAMPLE_RATE // g, rate // g
        self.rate = rate
        # resample_poly's kaiser filter is 10*max(up,down) taps to each side
        # in the upsampled domain -> that many / up native samples; round up
        # to a multiple of `down` so the output index stays exact.
        need = math.ceil(10 * max(self.up, self.down) / self.up)
        self._hold = ((need + self.down - 1) // self.down) * self.down
        self._carry = np.zeros(0, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.rate == SAMPLE_RATE:
            return x
        buf = np.concatenate([self._carry, x]) if self._carry.size else x
        out = self._rp(buf, self.up, self.down).astype(np.float32)
        lead = max(0, self._carry.size - self._hold)      # native samples already emitted
        start = lead * self.up // self.down
        end = max(start, (buf.size - self._hold) * self.up // self.down)
        keep = min(buf.size, 2 * self._hold)
        self._carry = buf[-keep:]
        return out[start:end]

    def reset(self):
        self._carry = np.zeros(0, dtype=np.float32)


def _resampler(rate: int):
    return _Resampler(rate)


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
        self.last_prob = 0.0             # the newest chunk's speech probability
        self.max_prob = 0.0              # the loudest speech seen this capture
        if self._resample is not None:
            self._resample.reset()
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
        self.last_prob = p
        if p > self.max_prob:
            self.max_prob = p
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

    def describe(self) -> str:
        """One line for the log when another detector beats this one."""
        gap = self.silence_since_speech
        return ("vad: started=%s last_speech=%s gap=%s last_p=%.2f max_p=%.2f audio=%.1fs"
                % (self._speech_started,
                   "%.2fs" % self.last_speech_seconds if self.last_speech_seconds is not None else "-",
                   "%.2fs" % gap if gap is not None else "-",
                   self.last_prob, self.max_prob, self.audio_seconds))


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
