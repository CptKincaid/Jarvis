"""TranscriptionPipeline — speaker filter → STT → intent classification.

Extracted from VoiceInputGUI. Holds no GUI state; returns a simple
PipelineResult dataclass.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

LOG_DIR = Path("/tmp/vss_voice")


from jarvis.jarvis_logging import get_logger
_log = get_logger("PIPE")


@dataclass
class PipelineResult:
    text: str
    speaker_authorized: bool
    intent: str | None = None


class TranscriptionPipeline:
    """Runs audio → speaker filter → STT → intent classifier."""

    def __init__(self, stt_engine, speaker_verifier=None, intent_classifier=None):
        self.stt_engine = stt_engine
        self.speaker_verifier = speaker_verifier
        self.intent_classifier = intent_classifier

    def transcribe(self, audio: np.ndarray) -> PipelineResult:
        if self.speaker_verifier is not None:
            try:
                ok = self.speaker_verifier.verify(audio)
            except Exception as e:
                _log(f"Speaker verify error: {e}")
                ok = True  # Fail open on verifier errors (explicit policy)
            if not ok:
                _log("Speaker rejected; discarding transcript")
                return PipelineResult(text="", speaker_authorized=False)

        try:
            stt_result = self.stt_engine.transcribe(audio)
            text = (stt_result.text or "").strip()
        except Exception as e:
            _log(f"STT error: {e}")
            return PipelineResult(text="", speaker_authorized=True)

        intent = None
        if self.intent_classifier is not None and text:
            try:
                intent = self.intent_classifier.classify(text)
            except Exception as e:
                _log(f"Intent classify error: {e}")

        return PipelineResult(
            text=text, speaker_authorized=True, intent=intent,
        )
