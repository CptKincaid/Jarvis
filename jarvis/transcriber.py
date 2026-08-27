"""Whisper transcription for Jarvis V3.

Owns the faster-whisper model and the lock that serializes access between
full transcription and streaming partial previews (the legacy monolith shared
one model between _transcribe_worker and _partial_transcribe_worker via
_partial_lock, voice_input_gui.py:1169).

Machine reality (config.MACHINE): the DGX Spark's ctranslate2 has no CUDA,
but torch cu130 does — so load() prefers the openai-whisper package on CUDA
("GPU fp16") whenever torch.cuda.is_available(). The faster-whisper CPU int8
path is kept as the fallback with unchanged semantics; the legacy CUDA
ctranslate2 attempt (2200-2228) survives only for machines where CUDA
ctranslate2 exists.

This module does transcription ONLY: no command routing, no speaker-segment
filtering (commander/pipeline calls speaker.filter_segments before handing
audio here), no widget access. Pure API — callers publish bus events.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from jarvis.config import CONFIG, MACHINE, PATHS
from jarvis.logs import get_logger

log = get_logger("transcriber")

# Audio sample rate all pipeline audio uses (port: voice_input_gui.py:53)
SAMPLE_RATE = 16000

# Streaming: partial transcription interval, seconds (port: 348)
STREAMING_INTERVAL = 2.0

# Confidence gate: reject transcriptions whose mean segment avg_logprob is
# below this (port: 2602-2604).
MIN_AVG_LOGPROB = -1.5

# Domain vocabulary prompt — biases Whisper toward these terms
# (port verbatim: voice_input_gui.py:78-86)
DEFAULT_VOCAB = (
    "AGV, loaded AGV, empty AGV, forklift, loaded forklift, empty forklift, "
    "pallet, flat pallet, boxes pallet, full pallet, cardboard box, conveyor, "
    "YOLO, VSS, KPI, ReID, SAM, ONNX, CUDA, GPU, RTSP, MJPEG, "
    "zone, dwell time, heatmap, flow rate, counting line, proximity alert, "
    "warehouse, surveillance, detection, tracking, annotation, training, "
    "ChromaDB, knowledge graph, Cosmos VLM, narration, Ollama, "
    "near-miss, safety incident, loading zone, staging area, shipping dock"
)

# Language options (port verbatim: voice_input_gui.py:351-372)
LANGUAGES = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Turkish", "tr"),
    ("Polish", "pl"),
    ("Ukrainian", "uk"),
    ("Vietnamese", "vi"),
    ("Thai", "th"),
    ("Swedish", "sv"),
]
LANG_MAP = dict(LANGUAGES)   # name -> whisper code (None = auto-detect)


# ------------------------------------------------------------------
# Custom vocabulary (persisted to voice_vocab.txt; port: 579-595)
# ------------------------------------------------------------------
def load_vocab() -> str:
    """Load domain vocabulary from user file, or return default."""
    if PATHS.VOCAB_FILE.exists():
        try:
            text = PATHS.VOCAB_FILE.read_text().strip()
            if text:
                return text
        except Exception:
            log.exception("vocab read failed; using default")
    return DEFAULT_VOCAB


def save_vocab(text: str):
    """Save domain vocabulary to user file."""
    PATHS.VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
    PATHS.VOCAB_FILE.write_text(text.strip())
    log.info("Vocabulary saved to %s", PATHS.VOCAB_FILE)


# ------------------------------------------------------------------
# Result
# ------------------------------------------------------------------
@dataclass
class TranscribeResult:
    text: str = ""
    confidence: float = 0.0            # mean segment avg_logprob (0.0 if no segments)
    segments: list = field(default_factory=list)   # [(seg_text, avg_logprob), ...]
    language: str | None = None        # detected/forced language code

    @property
    def accepted(self) -> bool:
        """Confidence gate (port: 2599-2607). No segments -> accept as-is
        (legacy skipped the gate when seg_data was empty)."""
        if not self.segments:
            return True
        return self.confidence >= MIN_AVG_LOGPROB


# ------------------------------------------------------------------
# Transcriber
# ------------------------------------------------------------------
class Transcriber:
    """Single owner of the Whisper model.

    self._lock serializes model access between transcribe() (full) and
    partial() (streaming preview) — port of _partial_lock (1169, 2592, 2716).
    """

    def __init__(self):
        self._model = None
        self._backend = ""
        self._gpu = False                    # True when openai-whisper on CUDA
        self._lock = threading.Lock()        # model access (full + partial)
        self._load_lock = threading.Lock()   # one-time load
        vocab = load_vocab()
        log.info("vocab: %d chars (%s)", len(vocab),
                 "custom file" if PATHS.VOCAB_FILE.exists() else "default")

    # -- vocab ----------------------------------------------------------
    @property
    def vocab(self) -> str:
        """Current vocabulary text (re-read from file, like the legacy
        per-transcription _load_vocab() calls at 2584/2719)."""
        return load_vocab()

    @vocab.setter
    def vocab(self, text: str):
        save_vocab(text)

    # -- model ----------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> str:
        return self._backend

    def load(self) -> str:
        """Load the Whisper model (idempotent, blocking). Returns backend
        string: "GPU fp16" (openai-whisper on CUDA, preferred when torch sees
        a GPU) or "CPU int8" (faster-whisper fallback, unchanged semantics —
        port of _load_model_worker 2200-2228 with the doomed CUDA ctranslate2
        attempt skipped when MACHINE.no_cuda_ct2)."""
        with self._load_lock:
            if self._model is not None:
                return self._backend

            model_size = CONFIG.model
            gpu = CONFIG.gpu

            # GPU first: openai-whisper on torch CUDA (fp16). CONFIG.model
            # names (tiny/base/small/medium/large) are valid in both packages.
            try:
                import torch
                if torch.cuda.is_available():
                    import whisper
                    log.info("Loading model: %s (openai-whisper, cuda)",
                             model_size)
                    model = whisper.load_model(model_size, device="cuda")
                    self._model = model
                    self._backend = "GPU fp16"
                    self._gpu = True
                    log.info("Model loaded on %s", self._backend)
                    return self._backend
            except Exception:
                log.exception(
                    "GPU whisper load failed; falling back to faster-whisper")

            from faster_whisper import WhisperModel

            log.info("Loading model: %s (no_cuda_ct2=%s)",
                     model_size, MACHINE.no_cuda_ct2)

            if MACHINE.no_cuda_ct2:
                model = WhisperModel(model_size, device="cpu",
                                     compute_type="int8")
                backend = "CPU int8"
            else:
                try:
                    model = WhisperModel(
                        model_size, device="cuda",
                        device_index=gpu, compute_type="float16",
                    )
                    backend = f"CUDA:{gpu}"
                except Exception:
                    log.exception("CUDA load failed; falling back to CPU int8")
                    model = WhisperModel(model_size, device="cpu",
                                         compute_type="int8")
                    backend = "CPU int8"

            self._model = model
            self._backend = backend
            self._gpu = False
            log.info("Model loaded on %s", backend)
            return backend

    # -- language -------------------------------------------------------
    def _language(self) -> str | None:
        """Whisper language code for the configured language
        (port: _get_whisper_language 2554-2557; None = auto-detect)."""
        return LANG_MAP.get(CONFIG.language, "en")

    # -- full transcription --------------------------------------------
    def transcribe(self, audio) -> TranscribeResult:
        """Full transcription: beam 5, VAD, vocab prompt, confidence gate.

        Port of _transcribe_worker's whisper section (2581-2609),
        transcription only — speaker filtering and command routing live in
        the pipeline. Loads the model on first use.
        """
        if self._model is None:
            self.load()

        lang = self._language()

        if self._gpu:
            # openai-whisper: expects float32 numpy @16k (exactly what the
            # recorder delivers — no resample). Segments carry avg_logprob
            # with the same semantics as faster-whisper's.
            with self._lock:
                result = self._model.transcribe(
                    audio,
                    initial_prompt=load_vocab(),
                    language=lang,          # None = auto-detect
                    beam_size=5,
                    fp16=True,
                )
            try:
                import torch
                torch.cuda.synchronize()   # idle the driver event thread
            except Exception:
                pass
            seg_list = result.get("segments") or []
            text = " ".join(s["text"].strip() for s in seg_list).strip()
            seg_data = [(s["text"], s["avg_logprob"]) for s in seg_list]
            language = result.get("language")
        else:
            kwargs = dict(
                beam_size=5,
                initial_prompt=load_vocab(),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            if lang is not None:
                kwargs["language"] = lang

            # Lock prevents concurrent model access with partial() (2592)
            with self._lock:
                segments, info = self._model.transcribe(audio, **kwargs)
                seg_list = list(segments)
            text = " ".join(seg.text.strip() for seg in seg_list).strip()
            seg_data = [(seg.text, seg.avg_logprob) for seg in seg_list]
            language = getattr(info, "language", None)

        if seg_data:
            avg_conf = sum(lp for _, lp in seg_data) / len(seg_data)
            log.info("Transcribed: %r (avg_logprob=%.2f)", text, avg_conf)
            if avg_conf < MIN_AVG_LOGPROB:
                log.info("Rejected: confidence too low (%.2f < %s)",
                         avg_conf, MIN_AVG_LOGPROB)
        else:
            avg_conf = 0.0
            log.info("Transcribed: %r", text)

        return TranscribeResult(
            text=text,
            confidence=avg_conf,
            segments=seg_data,
            language=language,
        )

    # -- streaming preview ---------------------------------------------
    def partial(self, audio) -> str:
        """Quick transcription for live preview (no VAD, beam 1).

        Port of _partial_transcribe_worker's whisper section (2710-2724);
        realtime command interception / filler detection / live typing stay
        in the pipeline. Returns "" if the model isn't loaded yet or the
        pass fails — a preview must never block or raise.
        """
        if self._model is None:
            return ""

        with self._lock:
            try:
                lang = self._language()

                if self._gpu:
                    # Greedy decode, no context carry-over, single
                    # temperature (no fallback retries), no timestamp
                    # tokens — fast text-only preview.
                    result = self._model.transcribe(
                        audio,
                        initial_prompt=load_vocab(),
                        language=lang,
                        condition_on_previous_text=False,
                        temperature=0.0,
                        without_timestamps=True,
                        fp16=True,
                    )
                    return (result.get("text") or "").strip()

                kwargs = dict(beam_size=1, initial_prompt=load_vocab())
                if lang is not None:
                    kwargs["language"] = lang

                segments, _ = self._model.transcribe(audio, **kwargs)
                return " ".join(seg.text.strip() for seg in segments).strip()
            except Exception:
                log.exception("partial transcription failed")
                return ""
