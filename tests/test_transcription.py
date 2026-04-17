"""Tests for TranscriptionPipeline."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis.transcription import TranscriptionPipeline, PipelineResult


def _audio():
    return np.zeros(16000, dtype=np.float32)


def test_unauthorized_speaker_returns_empty():
    verifier = MagicMock()
    verifier.verify.return_value = False
    stt = MagicMock()
    pipeline = TranscriptionPipeline(stt_engine=stt, speaker_verifier=verifier)
    result = pipeline.transcribe(_audio())
    assert result.text == ""
    assert result.speaker_authorized is False
    stt.transcribe.assert_not_called()


def test_authorized_speaker_runs_stt():
    verifier = MagicMock()
    verifier.verify.return_value = True
    stt_result = MagicMock(text="hello jarvis")
    stt = MagicMock()
    stt.transcribe.return_value = stt_result
    pipeline = TranscriptionPipeline(stt_engine=stt, speaker_verifier=verifier)
    result = pipeline.transcribe(_audio())
    assert result.text == "hello jarvis"
    assert result.speaker_authorized is True


def test_no_verifier_skips_speaker_check():
    stt_result = MagicMock(text="direct")
    stt = MagicMock()
    stt.transcribe.return_value = stt_result
    pipeline = TranscriptionPipeline(stt_engine=stt, speaker_verifier=None)
    result = pipeline.transcribe(_audio())
    assert result.text == "direct"
    assert result.speaker_authorized is True


def test_intent_classification_applied_when_classifier_provided():
    verifier = MagicMock()
    verifier.verify.return_value = True
    stt_result = MagicMock(text="please build the thing")
    stt = MagicMock()
    stt.transcribe.return_value = stt_result

    intent = MagicMock()
    intent.classify.return_value = "assistant"
    pipeline = TranscriptionPipeline(
        stt_engine=stt, speaker_verifier=verifier, intent_classifier=intent)
    result = pipeline.transcribe(_audio())
    assert result.intent == "assistant"
    intent.classify.assert_called_once_with("please build the thing")


def test_stt_exception_returns_empty_result():
    stt = MagicMock()
    stt.transcribe.side_effect = RuntimeError("model not loaded")
    pipeline = TranscriptionPipeline(stt_engine=stt, speaker_verifier=None)
    result = pipeline.transcribe(_audio())
    assert result.text == ""
    assert result.speaker_authorized is True


def test_speaker_verifier_exception_fails_open():
    """If the verifier crashes we proceed — explicit policy: user isn't
    locked out by a broken model. Worst case: stranger's speech gets through."""
    verifier = MagicMock()
    verifier.verify.side_effect = RuntimeError("embedding failed")
    stt_result = MagicMock(text="the show went on")
    stt = MagicMock()
    stt.transcribe.return_value = stt_result
    pipeline = TranscriptionPipeline(stt_engine=stt, speaker_verifier=verifier)
    result = pipeline.transcribe(_audio())
    assert result.text == "the show went on"
    assert result.speaker_authorized is True


def test_intent_classifier_exception_does_not_block_transcript():
    """A broken intent classifier shouldn't take the transcript with it."""
    verifier = MagicMock()
    verifier.verify.return_value = True
    stt_result = MagicMock(text="hello")
    stt = MagicMock()
    stt.transcribe.return_value = stt_result
    intent = MagicMock()
    intent.classify.side_effect = RuntimeError("corrupt log")
    pipeline = TranscriptionPipeline(
        stt_engine=stt, speaker_verifier=verifier, intent_classifier=intent)
    result = pipeline.transcribe(_audio())
    assert result.text == "hello"
    assert result.intent is None


def test_empty_stt_result_yields_empty_text():
    stt_result = MagicMock(text="")
    stt = MagicMock()
    stt.transcribe.return_value = stt_result
    pipeline = TranscriptionPipeline(stt_engine=stt, speaker_verifier=None)
    result = pipeline.transcribe(_audio())
    assert result.text == ""
