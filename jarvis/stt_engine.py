"""Speech-to-text engine for Jarvis.

Primary: NVIDIA Parakeet-TDT-0.6B-v2 (1.69% WER, 0.03s/5s audio)
Fallback: faster-whisper small (5.8% WER, 0.2s/5s audio)

Note: NeMo's model loading uses torch internals which may trigger
security scan warnings. This is expected for ML model deserialization.
"""

import os
import tempfile
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("/tmp/vss_voice")


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} [STT] {msg}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "gui_debug.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


@dataclass
class STTResult:
    """Transcription result with confidence scores."""
    text: str
    segments: list = field(default_factory=list)
    language: str = "en"

    @property
    def avg_logprob(self):
        if not self.segments:
            return 0.0
        return sum(lp for _, lp in self.segments) / len(self.segments)


class STTEngine:
    """Speech-to-text with Parakeet primary, Whisper fallback."""

    def __init__(self, gpu=1):
        self.gpu = gpu
        self._model = None
        self._engine_name = None
        self._loaded = False
        self._fallback_model = None

    @property
    def engine_name(self):
        return self._engine_name or "not loaded"

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        if self._loaded:
            return True
        if self._load_parakeet():
            return True
        _log("Parakeet failed, falling back to Whisper small")
        return self._load_whisper()

    def _load_parakeet(self):
        try:
            import nemo.collections.asr as nemo_asr
            _log("Loading Parakeet-TDT-0.6B-v2...")
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-tdt-0.6b-v2"
            )
            self._model = self._model.to(f"cuda:{self.gpu}")
            self._model.set_return_best_hypothesis(True)
            self._engine_name = "Parakeet-TDT"
            self._loaded = True
            # Warm up
            dummy = np.zeros(16000, dtype=np.float32)
            self._transcribe_parakeet(dummy)
            _log(f"Parakeet loaded on CUDA:{self.gpu}")
            return True
        except Exception as e:
            _log(f"Parakeet load error: {e}")
            self._model = None
            return False

    def _load_whisper(self):
        try:
            from faster_whisper import WhisperModel
            _log("Loading Whisper small (fallback)...")
            try:
                self._fallback_model = WhisperModel(
                    "small", device="cuda",
                    device_index=self.gpu, compute_type="float16",
                )
                backend = f"CUDA:{self.gpu}"
            except Exception:
                self._fallback_model = WhisperModel(
                    "small", device="cpu", compute_type="int8",
                )
                backend = "CPU"
            self._engine_name = "Whisper (fallback)"
            self._loaded = True
            _log(f"Whisper small loaded on {backend}")
            return True
        except Exception as e:
            _log(f"Whisper load error: {e}")
            return False

    def transcribe(self, audio_16k):
        """Transcribe 16kHz mono float32 audio -> STTResult."""
        if not self._loaded:
            return STTResult(text="", segments=[], language="en")
        if self._model is not None:
            return self._transcribe_parakeet(audio_16k)
        elif self._fallback_model is not None:
            return self._transcribe_whisper(audio_16k)
        return STTResult(text="", segments=[], language="en")

    def _transcribe_parakeet(self, audio_16k):
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, audio_16k, 16000)
            hypotheses = self._model.transcribe([tmp.name])
            if hypotheses and len(hypotheses) > 0:
                hyp = hypotheses[0]
                if isinstance(hyp, list) and len(hyp) > 0:
                    hyp = hyp[0]
                text = getattr(hyp, 'text', '') or ''
                score = getattr(hyp, 'score', -0.1)
                segments = [(text, score)] if text.strip() else []
                return STTResult(text=text.strip(), segments=segments, language="en")
            return STTResult(text="", segments=[], language="en")
        except Exception as e:
            _log(f"Parakeet transcribe error: {e}")
            return STTResult(text="", segments=[], language="en")
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _transcribe_whisper(self, audio_16k):
        try:
            segments, info = self._fallback_model.transcribe(
                audio_16k, language="en", beam_size=5, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            seg_list = list(segments)
            text = " ".join(seg.text.strip() for seg in seg_list).strip()
            seg_data = [(seg.text, seg.avg_logprob) for seg in seg_list]
            return STTResult(text=text, segments=seg_data, language="en")
        except Exception as e:
            _log(f"Whisper transcribe error: {e}")
            return STTResult(text="", segments=[], language="en")
