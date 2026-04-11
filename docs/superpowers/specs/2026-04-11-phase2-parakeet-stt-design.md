# Phase 2: Parakeet STT Upgrade — Design Spec

**Date:** 2026-04-11
**Scope:** Replace Whisper small with NVIDIA Parakeet-TDT-0.6B-v2 for speech-to-text
**Project:** Jarvis Voice Assistant (`/home/hunterp/jarvis/`)
**Phase:** 2 of 5 (STT Upgrade)

## Problem Statement

Whisper `small` is the current STT engine. It works but:
- 5.8% WER (word error rate) — misses or mangles words frequently
- 0.208s per 5s audio — acceptable but not instant
- Struggles with domain-specific terms and short utterances

## Solution

Replace with NVIDIA Parakeet-TDT-0.6B-v2:
- 1.69% WER — 3.4x more accurate
- 0.030s per 5s audio (warm) — 7x faster
- 2.4GB VRAM — similar to Whisper small
- English-only (sufficient for this use case)
- NeMo framework already installed and verified compatible

## Architecture

### New Module: `jarvis/stt_engine.py`

Wraps Parakeet with a clean interface matching what the GUI expects:

```python
class STTEngine:
    def load(gpu=1) -> bool
    def transcribe(audio_16k: np.ndarray) -> STTResult
    def is_loaded() -> bool
```

`STTResult` contains:
- `text` (str): full transcription
- `segments` (list): list of (text, avg_logprob) tuples for confidence display
- `language` (str): always "en" for Parakeet

Parakeet requires file input (not raw numpy). The engine handles this internally:
- Writes audio to a temp WAV file
- Calls `model.transcribe([path])`
- Deletes temp file
- Returns STTResult

### Fallback

If Parakeet fails to load (import error, GPU OOM, model download failure):
- Log warning
- Fall back to faster-whisper `small`
- GUI shows "Whisper (fallback)" in status instead of "Parakeet"

### Live Preview (Partial Transcription)

Disabled when using Parakeet. Parakeet transcribes in 0.03s so the final result appears almost instantly — partial preview adds no value. The live preview label stays but shows nothing during recording.

If the user had streaming enabled, it stays in settings but is effectively a no-op with Parakeet.

### Model Selection UI

Remove the model size dropdown (tiny/base/small/medium/large). Replace with a static label showing "Parakeet-TDT" or "Whisper (fallback)" depending on which loaded.

Keep the GPU selector — Parakeet still needs a GPU assignment.

## Files Changed

| File | Changes |
|------|---------|
| `jarvis/stt_engine.py` | NEW — Parakeet wrapper with fallback |
| `jarvis/voice_input_gui.py` | Use stt_engine for transcription, remove model size dropdown, update status display |
| `tests/test_stt_engine.py` | NEW — unit tests for STT engine |

## Files NOT Changed

| File | Reason |
|------|--------|
| `scripts/hotword_daemon.py` | Uses Whisper base for keyword detection, not full STT |
| `jarvis/audio_pipeline.py` | Audio processing unchanged |
| `jarvis/jarvis_tts.py` | TTS is Phase 4 |
| `jarvis/speaker_verification.py` | Speaker model is Phase 3 |

## Dependencies

Already installed:
- `nemo_toolkit[asr]` — verified working, no conflicts with existing packages
- `soundfile` — for temp WAV writing (already installed)

No new packages needed.

## Settings Changes

Removed:
- `model` setting (was: tiny/base/small/medium/large) — Parakeet is one model

Kept:
- `gpu` — which GPU to run on
- `language` — kept in settings but locked to "en" with Parakeet

New:
- `stt_engine` (string, default "parakeet") — "parakeet" or "whisper" for manual override

## Performance

| Metric | Whisper small (before) | Parakeet (after) |
|--------|----------------------|------------------|
| WER | 5.8% | 1.69% |
| Speed (5s audio, warm) | 0.208s | 0.030s |
| Speed (cold start) | ~0.5s | ~1.0s |
| VRAM | ~2GB | 2.4GB |
| Live preview | Yes (streaming) | Disabled (unnecessary) |

## Testing

1. Load Parakeet, transcribe a test audio file, verify text output
2. Force Parakeet load failure, verify Whisper fallback activates
3. Verify confidence scores are returned for GUI display
4. Verify cold start completes within 3 seconds
5. Verify warm transcription completes within 0.1 seconds for 10s audio

## Success Criteria

- Transcription noticeably more accurate (fewer misheard words)
- Response time feels instant (sub-100ms for typical utterances)
- No regression in any other voice feature
- Graceful fallback if Parakeet unavailable
