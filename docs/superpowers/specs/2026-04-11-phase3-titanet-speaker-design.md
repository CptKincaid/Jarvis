# Phase 3: TitaNet Speaker Verification — Design Spec

**Date:** 2026-04-11
**Scope:** Replace SpeechBrain ECAPA-TDNN with NVIDIA TitaNet-Large for speaker verification
**Project:** Jarvis Voice Assistant (`/home/hunterp/jarvis/`)
**Phase:** 3 of 5

## Problem Statement

SpeechBrain ECAPA-TDNN (0.80% EER) scores 0 on short 2-4 second utterances. While Phase 1's mic fix may resolve this, TitaNet-Large is fundamentally more accurate (0.66% EER) and explicitly designed for variable-length inputs including short clips.

## Solution

Swap the model inside `speaker_verification.py`. Same public API, same voiceprint storage format, different underlying model.

### Model Change

| | ECAPA-TDNN (before) | TitaNet-Large (after) |
|---|---|---|
| Framework | SpeechBrain | NeMo (already installed) |
| EER | 0.80% | 0.66% |
| Embedding dim | 192 | 192 |
| Short clip handling | Degraded | Designed for variable-length |
| VRAM | ~400MB | ~1.5GB |
| Load call | `EncoderClassifier.from_hparams()` | `EncDecSpeakerLabelModel.from_pretrained()` |
| Embed call | `encode_batch(tensor)` | `get_embedding(file_path)` |

### Voiceprint Compatibility

Old voiceprints (ECAPA-TDNN embeddings) are NOT compatible with TitaNet embeddings — different embedding spaces. On first load with the new model:
- Check if existing `voiceprint.npz` was created with ECAPA (store a `model_version` marker)
- If incompatible: delete old voiceprint, log warning, prompt user to re-enroll
- New voiceprints include `model_version: "titanet-large"` marker

### API (unchanged)

```python
verifier.load_model()           # Now loads TitaNet instead of ECAPA
verifier.enroll(audio_16k)      # Same
verifier.verify(audio_16k)      # Same  
verifier.filter_segments(...)   # Same
verifier.add_sample(audio_16k)  # Same
verifier.clear()                # Same
```

### Embedding Extraction

TitaNet requires file input (like Parakeet). The `_extract_embedding` method writes a temp WAV, calls `model.get_embedding(path)`, deletes the file.

## Files Changed

| File | Changes |
|------|---------|
| `jarvis/speaker_verification.py` | Replace ECAPA with TitaNet, add voiceprint version check |
| `tests/test_speaker_verification.py` | NEW — unit tests |

## Files NOT Changed

| File | Reason |
|------|--------|
| `jarvis/voice_input_gui.py` | Uses speaker_verification API which is unchanged |
| `jarvis/stt_engine.py` | STT is independent |
| `scripts/hotword_daemon.py` | Speaker check API unchanged |

## Dependencies

Already installed: `nemo_toolkit` (from Phase 2). No new packages.

## Testing

1. Load TitaNet, extract embedding from audio, verify 192-dim output
2. Enroll a sample, verify cosine similarity to self is high
3. Verify two different audio sources produce low similarity
4. Verify old voiceprint is detected and cleared on model change
5. Verify short clips (2-4 seconds) produce valid embeddings (not zero)
