# Phase 1: Audio Pipeline Foundation — Design Spec

**Date:** 2026-04-11
**Scope:** Smart mic persistence, RNNoise in-process noise suppression, Silero VAD silence detection
**Project:** Jarvis Voice Assistant (`/home/hunterp/jarvis/`)
**Phase:** 1 of 5 (Audio Pipeline Foundation)

## Problem Statement

Three root-cause audio issues degrade every downstream component (hotword, speaker verification, transcription, intent detection):

1. **Mic device silently falls back to Default (pipewire virtual mixer)** after reboots, feeding system-mixed audio instead of the physical microphone. This causes Voice ID to score 0 even on the user's own voice.
2. **No noise suppression** — TV/YouTube audio reaches hotword detection and transcription unfiltered, causing false triggers and contaminated recordings.
3. **RMS-based silence detection fails with background audio** — TV keeps the RMS above threshold so recordings never auto-stop, or stop at wrong times.

## Solution

### 1. Smart Mic Persistence

**Save by device name, resolve by name each launch.**

On mic selection in GUI dropdown:
- Store `mic_name` as the device name string (e.g. "USB Audio", "Blue Snowball") in `~/.aiws_trainer/voice_settings.json`
- Store `mic_index` as ephemeral cache (not relied upon across restarts)

On startup (GUI and daemon):
- Scan all input devices via `sounddevice.query_devices()`
- Match `mic_name` against all device names (case-insensitive substring match)
- If exactly one match: use that device index, log the mapping
- If multiple matches: use the first match, log a warning
- If no match: do NOT silently fall back to Default. Instead:
  - Show status bar warning: "Mic 'Blue Snowball' not found — check USB connection"
  - Log the error
  - Disable recording until user selects a new mic

"Test Mic" button in settings:
- Records 3 seconds from the currently selected device
- Plays back through speakers via `paplay` (PipeWire) or `aplay` fallback
- User can hear whether they're recording their own voice or system audio
- Shows RMS level during test for visual confirmation

### 2. RNNoise In-Process Noise Suppression

**New module: `jarvis/audio_pipeline.py`**

Wraps raw mic audio through RNNoise before it reaches any other component.

Implementation:
- Primary: `noisereduce` Python package (spectral gating, pure Python, no C dependencies)
- Fallback: `rnnoise-python` C bindings if available (lower latency, harder to install)
- Either approach produces equivalent denoised output for speech-in-noise scenarios
- Process audio in the mic callback chain: mic → RNNoise → audio buffer
- RNNoise operates on 16kHz mono float32 audio (resample from native mic rate if needed)
- Adds ~10ms latency, runs entirely on CPU
- Negligible CPU overhead (<1% of one core)

Settings toggle:
- "Noise suppression" checkbox in GUI settings (default: ON)
- Persisted in `voice_settings.json` as `noise_suppression: true`
- Hot-toggleable (takes effect on next audio chunk, no restart needed)

The denoised audio feeds into:
- Hotword detection (OpenWakeWord)
- Silero VAD
- Recording buffer (what gets transcribed)
- Speaker verification

### 3. Silero VAD Silence Detection

**Replace RMS-based `_check_silence()` with Silero VAD.**

Model loading:
- `torch.hub.load('snakers4/silero-vad', 'silero_vad')` — cached after first download
- Runs on CPU, <1MB model, <1ms per inference
- Loaded once at GUI startup, reused across all recordings

Silence detection logic:
- Every 100ms during recording, feed the latest audio chunk (1600 samples at 16kHz) to Silero
- Silero returns a speech probability (0.0 to 1.0)
- If speech probability < 0.3 continuously for the silence timeout: auto-stop recording
- If speech probability >= 0.3: reset the silence timer
- Default timeout: 2 seconds (changed from 3)
- User-adjustable via existing slider (1-10 seconds range preserved)

Replaces:
- The RMS-based `_check_silence()` method
- The `_check_speaker_silence()` method (voice-aware silence) — no longer needed since Silero handles speech vs non-speech natively
- The `noise_threshold_var` and calibration button become unnecessary (Silero doesn't need manual calibration)

Does NOT replace:
- The noise gate on the recording audio (keeps keyboard click filtering)
- Speaker verification (that's Phase 3)

### Data Flow

```
Physical Mic (matched by name, NOT "Default")
    |
    v
RNNoise (denoise, CPU, ~10ms)
    |
    v
Audio Buffer (clean audio)
    |
    +--> Silero VAD (speech detection, CPU, <1ms)
    |       |
    |       +--> Auto-stop when no speech for 2s
    |
    +--> Hotword Listener (OpenWakeWord, CPU, 1.5ms)
    |
    +--> Recording → Transcription (Whisper, GPU)
    |
    +--> Speaker Verification (when enabled, GPU)
```

## Files Changed

| File | Changes |
|------|---------|
| `jarvis/audio_pipeline.py` | NEW — RNNoise wrapper, audio processing utilities |
| `jarvis/voice_input_gui.py` | Mic name matching on startup, Silero VAD in silence detection, noise suppression toggle in settings, "Test Mic" button, remove RMS silence check, remove speaker silence check, remove calibration button |
| `scripts/hotword_daemon.py` | Same mic name matching logic, RNNoise pre-filter on hotword audio |

## Files NOT Changed

| File | Reason |
|------|--------|
| `jarvis/jarvis_tts.py` | TTS upgrade is Phase 4 |
| `jarvis/jarvis_brain.py` | No audio pipeline involvement |
| `jarvis/jarvis_agent.py` | No audio pipeline involvement |
| `jarvis/speaker_verification.py` | Speaker model upgrade is Phase 3 |
| `jarvis/orbit_animation.html` | UI animation unchanged |
| `jarvis/orbit_server.py` | Animation server unchanged |

## Dependencies

New Python packages:
- `noisereduce` — spectral gating noise reduction (pip install)
- `silero-vad` — loaded via `torch.hub` (auto-downloads ~1MB model)

No new system-level dependencies. No PipeWire configuration changes.

## Settings Changes

New settings in `voice_settings.json`:
- `mic_name` (string): persistent device name for matching
- `noise_suppression` (bool, default true): enable/disable RNNoise

Modified settings:
- `silence_timeout` default changes from 3.0 to 2.0

Removed/deprecated settings:
- `noise_threshold` — replaced by Silero VAD (no manual calibration needed)

## Testing

After implementation:
1. Restart PC, verify mic auto-resolves by name (not index)
2. Unplug mic, verify warning appears (not silent fallback)
3. Play YouTube loud, verify RNNoise strips TV audio from recording
4. Talk for 5 seconds then stop — verify Silero auto-stops within 2 seconds even with TV playing
5. Use "Test Mic" button — verify you hear your own voice, not system audio

## Success Criteria

- Mic persists correctly across reboots (no silent fallback)
- TV audio at normal volume does NOT prevent silence detection
- TV audio at normal volume does NOT contaminate transcriptions
- Auto-stop works reliably within 2-3 seconds of stopping speech
- No perceptible latency increase from RNNoise (<10ms)
