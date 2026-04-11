# Phase 1: Audio Pipeline Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three root-cause audio issues (mic fallback, no noise suppression, broken silence detection) that degrade every downstream voice component.

**Architecture:** New `audio_pipeline.py` module handles noise suppression and mic resolution. Silero VAD replaces RMS-based silence detection in the GUI. All changes are in-process Python with no system-level audio config.

**Tech Stack:** sounddevice, noisereduce, torch (Silero VAD via torch.hub), numpy, scipy

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `jarvis/audio_pipeline.py` | CREATE | Mic name resolution, noise suppression wrapper, Silero VAD wrapper |
| `jarvis/voice_input_gui.py` | MODIFY | Mic name matching, Silero VAD silence check, denoise toggle, Test Mic button, remove calibration |
| `scripts/hotword_daemon.py` | MODIFY | Mic name matching, noise suppression on hotword audio |
| `tests/test_mic_resolution.py` | CREATE | Unit tests for mic name resolution |
| `tests/test_audio_pipeline.py` | CREATE | Unit tests for noise suppression and Silero VAD |

---

### Task 1: Install Dependencies

**Files:** None (pip install only)

- [ ] **Step 1: Install noisereduce**

```bash
source /home/hunterp/vss_env/bin/activate
python3 -m pip install noisereduce
```

Expected: Successfully installed noisereduce

- [ ] **Step 2: Verify Silero VAD downloads**

```bash
source /home/hunterp/vss_env/bin/activate
python3 -c "
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
print('Silero VAD loaded OK')
print('Model type:', type(model))
"
```

Expected: Silero VAD loaded OK

- [ ] **Step 3: Verify noisereduce works**

```bash
source /home/hunterp/vss_env/bin/activate
python3 -c "
import noisereduce as nr
import numpy as np
audio = np.random.randn(16000).astype(np.float32) * 0.1
reduced = nr.reduce_noise(y=audio, sr=16000, stationary=True)
print('Input RMS:', round(np.sqrt(np.mean(audio**2)), 4))
print('Output RMS:', round(np.sqrt(np.mean(reduced**2)), 4))
print('noisereduce OK')
"
```

Expected: Output RMS lower than input RMS

- [ ] **Step 4: Commit**

```bash
cd /home/hunterp/jarvis && git add -A && git commit -m "chore: verify Phase 1 deps (noisereduce, silero-vad)"
```

---

### Task 2: Create audio_pipeline.py with Mic Resolution

**Files:**
- Create: `jarvis/audio_pipeline.py`
- Create: `tests/test_mic_resolution.py`

- [ ] **Step 1: Write failing tests for mic resolution**

Create `tests/test_mic_resolution.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_resolve_mic_by_name_exact():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [
        {"name": "HDA NVidia: HDMI 0", "max_input_channels": 0},
        {"name": "USB Audio: Blue Snowball", "max_input_channels": 2},
        {"name": "pipewire", "max_input_channels": 64},
    ]
    assert resolve_mic_by_name("Blue Snowball", devices) == 1

def test_resolve_mic_by_name_substring():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [
        {"name": "HDA NVidia", "max_input_channels": 0},
        {"name": "USB Audio: - (hw:2,0)", "max_input_channels": 2},
        {"name": "default", "max_input_channels": 64},
    ]
    assert resolve_mic_by_name("USB Audio", devices) == 1

def test_resolve_mic_case_insensitive():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [{"name": "Blue Snowball iCE", "max_input_channels": 1}]
    assert resolve_mic_by_name("blue snowball", devices) == 0

def test_resolve_mic_not_found():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [{"name": "pipewire", "max_input_channels": 64}]
    assert resolve_mic_by_name("Blue Snowball", devices) is None

def test_resolve_mic_skips_output_only():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [
        {"name": "Blue Snowball Output", "max_input_channels": 0},
        {"name": "Blue Snowball Input", "max_input_channels": 2},
    ]
    assert resolve_mic_by_name("Blue Snowball", devices) == 1

def test_resolve_mic_default_returns_none():
    from jarvis.audio_pipeline import resolve_mic_by_name
    assert resolve_mic_by_name("Default", []) is None
    assert resolve_mic_by_name(None, []) is None
    assert resolve_mic_by_name("", []) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/hunterp/jarvis && python3 -m pytest tests/test_mic_resolution.py -v
```

Expected: All 6 FAIL (ImportError)

- [ ] **Step 3: Implement audio_pipeline.py with resolve_mic_by_name**

Create `jarvis/audio_pipeline.py`:

```python
"""Audio pipeline utilities for Jarvis voice assistant.

Provides mic resolution, noise suppression, and VAD — the foundation
audio processing that all other voice components depend on.
"""

import numpy as np


def resolve_mic_by_name(saved_name, devices=None):
    """Resolve a mic device by name substring (case-insensitive).

    Args:
        saved_name: Saved device name to search for (e.g. "Blue Snowball").
        devices: List of device dicts from sounddevice.query_devices().
                 If None, queries live devices.

    Returns:
        Device index (int) or None if not found.
    """
    if not saved_name or saved_name.lower() in ("default", ""):
        return None

    if devices is None:
        import sounddevice as sd
        devices = sd.query_devices()

    saved_lower = saved_name.lower()

    # Strip "[index] " prefix if present (legacy format)
    if saved_lower.startswith("["):
        bracket_end = saved_lower.find("]")
        if bracket_end > 0:
            saved_lower = saved_lower[bracket_end + 1:].strip()

    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        if saved_lower in d.get("name", "").lower():
            return i

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/hunterp/jarvis && python3 -m pytest tests/test_mic_resolution.py -v
```

Expected: All 6 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/hunterp/jarvis && git add jarvis/audio_pipeline.py tests/test_mic_resolution.py
git commit -m "feat: add mic name resolution to audio_pipeline"
```

---

### Task 3: Add Noise Suppression to audio_pipeline.py

**Files:**
- Modify: `jarvis/audio_pipeline.py`
- Create: `tests/test_audio_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_audio_pipeline.py`:

```python
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_denoise_reduces_noise():
    from jarvis.audio_pipeline import denoise_audio
    noise = np.random.randn(16000).astype(np.float32) * 0.1
    result = denoise_audio(noise, sr=16000)
    assert result.shape == noise.shape
    assert np.sqrt(np.mean(result**2)) < np.sqrt(np.mean(noise**2))

def test_denoise_preserves_shape():
    from jarvis.audio_pipeline import denoise_audio
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    speech = np.sin(2 * np.pi * 440 * t) * 0.5
    noisy = speech + np.random.randn(16000).astype(np.float32) * 0.05
    result = denoise_audio(noisy, sr=16000)
    assert result.shape == noisy.shape
    assert result.dtype == np.float32

def test_denoise_handles_short_audio():
    from jarvis.audio_pipeline import denoise_audio
    short = np.random.randn(800).astype(np.float32) * 0.1
    result = denoise_audio(short, sr=16000)
    assert result.shape == short.shape

def test_denoise_disabled_returns_original():
    from jarvis.audio_pipeline import denoise_audio
    audio = np.random.randn(16000).astype(np.float32)
    result = denoise_audio(audio, sr=16000, enabled=False)
    np.testing.assert_array_equal(result, audio)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/hunterp/jarvis && python3 -m pytest tests/test_audio_pipeline.py -v
```

Expected: All 4 FAIL (ImportError)

- [ ] **Step 3: Add denoise_audio to audio_pipeline.py**

Append to `jarvis/audio_pipeline.py`:

```python
def denoise_audio(audio, sr=16000, enabled=True):
    """Remove background noise using spectral gating.

    Args:
        audio: numpy float32 array, mono
        sr: sample rate
        enabled: if False, returns audio unchanged

    Returns:
        Denoised numpy float32 array, same shape as input.
    """
    if not enabled or len(audio) < 1600:
        return audio

    try:
        import noisereduce as nr
        reduced = nr.reduce_noise(
            y=audio.astype(np.float32), sr=sr,
            stationary=True, prop_decrease=0.75,
            n_fft=512, hop_length=128,
        )
        return reduced.astype(np.float32)
    except Exception:
        return audio
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/hunterp/jarvis && python3 -m pytest tests/test_audio_pipeline.py -v
```

Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/hunterp/jarvis && git add jarvis/audio_pipeline.py tests/test_audio_pipeline.py
git commit -m "feat: add noise suppression to audio_pipeline"
```

---

### Task 4: Add Silero VAD Wrapper to audio_pipeline.py

**Files:**
- Modify: `jarvis/audio_pipeline.py`
- Modify: `tests/test_audio_pipeline.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_audio_pipeline.py`:

```python
def test_vad_silence_detected():
    from jarvis.audio_pipeline import SileroVAD
    vad = SileroVAD()
    silence = np.zeros(1600, dtype=np.float32)
    prob = vad.is_speech(silence, sr=16000)
    assert prob < 0.3

def test_vad_returns_float_in_range():
    from jarvis.audio_pipeline import SileroVAD
    vad = SileroVAD()
    audio = np.random.randn(1600).astype(np.float32) * 0.01
    prob = vad.is_speech(audio, sr=16000)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_vad_reusable():
    from jarvis.audio_pipeline import SileroVAD
    vad = SileroVAD()
    silence = np.zeros(1600, dtype=np.float32)
    for _ in range(5):
        prob = vad.is_speech(silence, sr=16000)
    assert prob < 0.3

def test_vad_reset():
    from jarvis.audio_pipeline import SileroVAD
    vad = SileroVAD()
    silence = np.zeros(1600, dtype=np.float32)
    vad.is_speech(silence, sr=16000)
    vad.reset()  # Should not crash
    prob = vad.is_speech(silence, sr=16000)
    assert prob < 0.3
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd /home/hunterp/jarvis && python3 -m pytest tests/test_audio_pipeline.py::test_vad_silence_detected -v
```

Expected: FAIL (ImportError — SileroVAD not defined)

- [ ] **Step 3: Add SileroVAD class to audio_pipeline.py**

Append to `jarvis/audio_pipeline.py`:

```python
import torch


class SileroVAD:
    """Silero VAD wrapper for speech detection.

    Replaces RMS-based silence threshold. Handles TV/music background
    noise correctly. Runs on CPU, <1ms per call.
    """

    def __init__(self):
        self._model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._model, _ = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', trust_repo=True,
        )
        self._model.eval()
        self._loaded = True

    def is_speech(self, audio_chunk, sr=16000):
        """Check if audio chunk contains speech.

        Args:
            audio_chunk: numpy float32 array, ~100ms (1600 samples at 16kHz)
            sr: sample rate

        Returns:
            Speech probability (0.0 to 1.0). Above 0.3 is likely speech.
        """
        self._ensure_loaded()

        if isinstance(audio_chunk, np.ndarray):
            tensor = torch.from_numpy(audio_chunk).float()
        else:
            tensor = audio_chunk

        if tensor.dim() > 1:
            tensor = tensor.squeeze()

        with torch.no_grad():
            prob = self._model(tensor, sr)

        return float(prob)

    def reset(self):
        """Reset model state between recordings."""
        if self._model is not None:
            self._model.reset_states()
```

- [ ] **Step 4: Run all tests**

```bash
cd /home/hunterp/jarvis && python3 -m pytest tests/test_audio_pipeline.py -v
```

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/hunterp/jarvis && git add jarvis/audio_pipeline.py tests/test_audio_pipeline.py
git commit -m "feat: add Silero VAD wrapper to audio_pipeline"
```

---

### Task 5: Integrate Smart Mic into GUI

**Files:**
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Add `_get_mic_raw_name()` helper method**

Add to the VoiceInputGUI class (near `_detect_mics`):

```python
def _get_mic_raw_name(self):
    """Get raw device name (without index prefix) for persistence."""
    display = self.mic_var.get()
    if display == "Default":
        return None
    if display.startswith("["):
        bracket_end = display.find("]")
        if bracket_end > 0:
            return display[bracket_end + 2:].strip()
    return display
```

- [ ] **Step 2: Add `mic_name` to `_save_settings()`**

Find `"mic": self.mic_var.get(),` in the save data dict. Add after it:

```python
            "mic_name": self._get_mic_raw_name(),
```

- [ ] **Step 3: Add mic name resolution to `_load_settings()`**

Find `# Restore pinned target window name` in `_load_settings()`. Add before it:

```python
            # Resolve mic by name (survives reboots/USB re-enumeration)
            saved_mic_name = data.get("mic_name")
            if saved_mic_name:
                from jarvis.audio_pipeline import resolve_mic_by_name
                import sounddevice as sd
                idx = resolve_mic_by_name(saved_mic_name, sd.query_devices())
                if idx is not None:
                    for display, dev_idx in self._mic_devices.items():
                        if dev_idx == idx:
                            self.mic_var.set(display)
                            _log(f"Mic resolved: '{saved_mic_name}' -> [{idx}]")
                            break
                else:
                    _log(f"WARNING: Mic '{saved_mic_name}' not found")
                    self.root.after(1000, lambda n=saved_mic_name: self._set_status(
                        "Mic not found", "#da3633", f"'{n}' — check USB"))
```

- [ ] **Step 4: Update mic fallback in `_start_recording()`**

Find the block `if mic_name not in self._mic_devices:` and replace:

```python
        if mic_name not in self._mic_devices:
            from jarvis.audio_pipeline import resolve_mic_by_name
            import sounddevice as sd
            raw_name = self._get_mic_raw_name()
            resolved = resolve_mic_by_name(raw_name, sd.query_devices()) if raw_name else None
            if resolved is not None:
                mic_idx = resolved
                _log(f"Mic resolved by name: '{raw_name}' -> [{resolved}]")
            else:
                _log(f"WARNING: Mic not found, recording disabled")
                self._set_status("Mic not found", "#da3633", "Check USB connection")
                return
```

- [ ] **Step 5: Commit**

```bash
cd /home/hunterp/jarvis && git add jarvis/voice_input_gui.py
git commit -m "feat: smart mic persistence by device name"
```

---

### Task 6: Add Test Mic Button

**Files:**
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Add button in settings UI after mic dropdown**

Find the mic combobox (`cb_mic`) pack line. Add after it:

```python
        test_mic_btn = self._holo_btn(row2, "Test Mic", self._test_mic)
        test_mic_btn.pack(side="left", padx=(6, 0))
        _Tooltip(test_mic_btn,
                 "Records 3s and plays back through speakers.\n"
                 "Verify you hear YOUR voice, not system audio.")
```

- [ ] **Step 2: Add `_test_mic()` method**

```python
def _test_mic(self):
    """Record 3 seconds from selected mic and play back."""
    if self.recording:
        self._set_status("Stop recording first", self.YELLOW, "")
        return
    if self.hotword_var.get() and self._hotword._stream:
        self._hotword.pause()

    mic_idx = self._mic_devices.get(self.mic_var.get())
    self._set_status("Testing mic...", self.ACCENT, "Speak now (3 seconds)")

    def _worker():
        import sounddevice as sd
        import wave, tempfile, subprocess, os
        try:
            dev_info = sd.query_devices(mic_idx, 'input')
            rate = int(dev_info['default_samplerate'])
        except Exception:
            rate = 44100
        try:
            audio = sd.rec(int(3 * rate), samplerate=rate,
                           channels=1, dtype='float32', device=mic_idx)
            sd.wait()
            rms = float(np.sqrt(np.mean(audio ** 2)))
            _log(f"Test mic: RMS={rms:.4f}")
            self.root.after(0, lambda: self._set_status(
                "Playing back...", self.ACCENT, f"RMS: {rms:.4f}"))
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            for cmd in [["paplay", tmp.name], ["aplay", "-q", tmp.name]]:
                try:
                    subprocess.run(cmd, timeout=10, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            os.unlink(tmp.name)
            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, f"Mic test done (RMS: {rms:.4f})"))
        except Exception as e:
            _log(f"Test mic error: {e}")
            self.root.after(0, lambda: self._set_status("Error", "#da3633", str(e)[:40]))
        finally:
            if self.hotword_var.get():
                self.root.after(0, self._hotword.resume)

    threading.Thread(target=_worker, daemon=True).start()
```

- [ ] **Step 3: Commit**

```bash
cd /home/hunterp/jarvis && git add jarvis/voice_input_gui.py
git commit -m "feat: add Test Mic playback button"
```

---

### Task 7: Integrate Noise Suppression into GUI

**Files:**
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Add settings variable**

Near other setting vars (around line 1200), add:

```python
        self.noise_suppress_var = tk.BooleanVar(value=True)
```

- [ ] **Step 2: Add to persistence**

In `_load_settings()` mapping: `"noise_suppression": (self.noise_suppress_var, bool),`

In `_save_settings()` data: `"noise_suppression": self.noise_suppress_var.get(),`

In `_setup_auto_save()` var list: add `self.noise_suppress_var`

- [ ] **Step 3: Add to audio callback in `_start_recording()`**

Before the callback function, cache the setting:

```python
        noise_suppress_enabled = self.noise_suppress_var.get()
```

Inside `audio_callback`, after the noise gate block, add:

```python
            if noise_suppress_enabled and len(chunk) >= 1600:
                from jarvis.audio_pipeline import denoise_audio
                chunk = denoise_audio(chunk.flatten(), sr=native_rate).reshape(-1, 1)
```

- [ ] **Step 4: Add toggle in settings UI (row 4)**

```python
        chk_denoise = tk.Checkbutton(
            row4, text="Denoise", variable=self.noise_suppress_var, **chk_style,
        )
        chk_denoise.pack(side="left", padx=(0, 4))
        _Tooltip(chk_denoise,
                 "Removes TV/music/ambient noise before transcription.\n"
                 "Uses spectral gating. Default: ON.")
```

- [ ] **Step 5: Commit**

```bash
cd /home/hunterp/jarvis && git add jarvis/voice_input_gui.py
git commit -m "feat: integrate noise suppression into recording"
```

---

### Task 8: Replace RMS Silence with Silero VAD

**Files:**
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Load Silero VAD at startup**

In `__init__`, after the hotword listener creation:

```python
        from jarvis.audio_pipeline import SileroVAD
        self._silero_vad = SileroVAD()
```

- [ ] **Step 2: Change SILENCE_TIMEOUT default**

Change `SILENCE_TIMEOUT = 3.0` to:

```python
SILENCE_TIMEOUT = 2.0
```

- [ ] **Step 3: Replace `_check_silence()` method**

Replace the entire method with:

```python
def _check_silence(self):
    if not self.recording:
        return

    # Hard cap
    if self._record_start_time:
        elapsed = time.monotonic() - self._record_start_time
        if elapsed >= self.MAX_RECORDING_SECONDS:
            _log(f"Max recording time ({self.MAX_RECORDING_SECONDS}s)")
            self._voice_stopped = True
            try:
                self._stop_and_transcribe()
            except Exception as e:
                _log(f"Max recording stop error: {e}")
                self.recording = False
                self._reset_button()
            return

    # Skip first 2 seconds
    if self._record_start_time and (time.monotonic() - self._record_start_time) < 2.0:
        self.root.after(100, self._check_silence)
        return

    # Silero VAD speech detection
    try:
        if len(self._audio_frames) >= 2:
            recent = np.concatenate(self._audio_frames[-2:], axis=0).flatten()
            rate = getattr(self, '_record_rate', 16000)
            if rate != 16000:
                from scipy.signal import resample
                new_len = int(len(recent) * 16000 / rate)
                recent = resample(recent, new_len).astype(np.float32)
            chunk = recent[-1600:] if len(recent) >= 1600 else recent
            speech_prob = self._silero_vad.is_speech(chunk, sr=16000)
        else:
            speech_prob = 1.0
    except Exception:
        speech_prob = 1.0

    timeout = self.silence_var.get()

    if speech_prob < 0.3:
        if self._silence_start is None:
            self._silence_start = time.monotonic()
        elif (time.monotonic() - self._silence_start) >= timeout:
            _log(f"Auto-stop: no speech for {timeout}s (prob={speech_prob:.2f})")
            self._voice_stopped = True
            try:
                self._stop_and_transcribe()
            except Exception as e:
                _log(f"Auto-stop error: {e}")
                self.recording = False
                self._reset_button()
                if self.hotword_var.get():
                    self._hotword.resume()
            return
    else:
        self._silence_start = None

    self.root.after(100, self._check_silence)
```

- [ ] **Step 4: Remove `_check_speaker_silence()` method and its call**

Delete the entire `_check_speaker_silence` method.

In `_start_recording()`, find and remove: `self._check_speaker_silence()`

- [ ] **Step 5: Remove calibration UI and methods**

In the settings UI, remove the calibration button, noise threshold label, and threshold display.

Delete methods: `_calibrate_noise`, `_calibrate_worker`

Keep the silence timeout slider (it still controls Silero's wait time).

- [ ] **Step 6: Reset Silero between recordings**

In `_start_recording()`, add before recording begins:

```python
        self._silero_vad.reset()
```

- [ ] **Step 7: Commit**

```bash
cd /home/hunterp/jarvis && git add jarvis/voice_input_gui.py
git commit -m "feat: replace RMS silence detection with Silero VAD"
```

---

### Task 9: Integrate into Hotword Daemon

**Files:**
- Modify: `scripts/hotword_daemon.py`

- [ ] **Step 1: Update `_resolve_mic()` to use audio_pipeline**

Replace the function:

```python
def _resolve_mic(mic_name):
    import sounddevice as sd
    if not mic_name or mic_name == "Default":
        return None
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = sd.query_devices()
    idx = resolve_mic_by_name(mic_name, devices)
    if idx is not None:
        _log(f"Mic resolved: '{mic_name}' -> [{idx}]")
        return idx
    # Legacy index format fallback
    if mic_name.startswith("["):
        try:
            idx = int(mic_name.split("]")[0][1:])
            sd.query_devices(idx, 'input')
            return idx
        except Exception:
            pass
    _log(f"WARNING: Mic '{mic_name}' not found")
    return None
```

- [ ] **Step 2: Load `mic_name` from settings**

In `_load_settings()`, add: `"mic_name": data.get("mic_name", data.get("mic")),`

In `run_daemon()`, use: `mic_idx = _resolve_mic(settings.get("mic_name") or settings.get("mic"))`

- [ ] **Step 3: Add noise suppression to hotword audio**

Before the hotword prediction call, denoise the audio:

```python
        from jarvis.audio_pipeline import denoise_audio
        raw = denoise_audio(raw, sr=16000)
```

- [ ] **Step 4: Commit**

```bash
cd /home/hunterp/jarvis && git add scripts/hotword_daemon.py
git commit -m "feat: smart mic + denoise in hotword daemon"
```

---

### Task 10: Final Integration Test

**Files:** None (testing only)

- [ ] **Step 1: Run all unit tests**

```bash
cd /home/hunterp/jarvis && python3 -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Restart daemon**

```bash
pkill -f "hotword_daemon.py" 2>/dev/null; sleep 1
source /home/hunterp/vss_env/bin/activate
nohup python3 /home/hunterp/jarvis/scripts/hotword_daemon.py --daemon > /dev/null 2>&1 &
sleep 3 && tail -5 /tmp/vss_voice/hotword_daemon.log
```

Expected: Mic resolves by name in log

- [ ] **Step 3: Launch GUI and verify**

1. Correct mic selected (not "Default")
2. "Test Mic" plays back your voice
3. Say "Jarvis", speak 3-4 seconds, stop — auto-stops within 2s even with TV
4. Close and reopen — mic persists

- [ ] **Step 4: Final commit**

```bash
cd /home/hunterp/jarvis && git add -A
git commit -m "feat: Phase 1 complete — audio pipeline foundation

- Smart mic persistence (name-based, survives reboots)
- Noise suppression (noisereduce spectral gating)
- Silero VAD silence detection (replaces RMS threshold)
- Test Mic playback button

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
