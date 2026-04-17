# Linux HUD Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Jarvis-Win modular architecture (pywebview HUD + `config.py` + commander + 15-file split) onto the Linux repo while keeping the Linux AI stack (Parakeet, TitaNet, Kokoro/F5, intent learning, dual-threshold wake word).

**Architecture:** One branch `linux-hud` off `jarvis-review-fixes`. Phase 1 extracts backend modules (`config.py`, `desktop.py`, `hotword.py`, `commander.py`) without touching the live path. Phase 2 builds the pywebview HUD (`app.py` + `frontend/`) as a parallel entry point. Phase 3 adds engineering hygiene (TODO.md, docs layout). Tkinter GUI stays as `--legacy` fallback indefinitely.

**Tech Stack:** Python 3.12, pywebview (GTK WebKit2), HTML/CSS/JS HUD, sounddevice, OpenWakeWord, NeMo (Parakeet + TitaNet), Kokoro, F5-TTS, xdotool/wmctrl/xclip for desktop automation, pytest.

**Reference:**
- Spec: `docs/superpowers/specs/2026-04-16-linux-hud-port-design.md`
- Architecture source: github.com/CptKincaid/Jarvis-Win (master branch)

---

## Task 0: Branch safety + worktree prep

**Files:**
- None (git operations only)

- [ ] **Step 1: Push current branch to origin**

```bash
cd /home/hunterp/jarvis
git push -u origin jarvis-review-fixes
```

Expected: branch pushed, GitHub shows 30+ commits ahead of main.

- [ ] **Step 2: Create feature branch**

```bash
git checkout -b linux-hud
git status
```

Expected: `On branch linux-hud`, clean working tree.

---

# PHASE 1 — Backend Foundations

Four new modules extracted and tested in isolation. `voice_input_gui.py` is then updated to import from them with zero behavior change.

## Task 1: `jarvis/config.py`

**Files:**
- Create: `jarvis/config.py`
- Create: `tests/test_config.py`

**Responsibility:** Every path, GPU detection, feature flag, and model name lives here. Every other module imports from `config`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Tests for centralized config module."""

from pathlib import Path

import pytest


def test_appdata_dir_is_under_home():
    from jarvis import config
    assert str(config.APPDATA_DIR).startswith(str(Path.home()))


def test_paths_exist_after_import():
    from jarvis import config
    assert config.APPDATA_DIR.is_dir()
    assert config.TEMP_DIR.is_dir()
    assert config.MEMORY_DIR.is_dir()
    assert config.SCREEN_DIR.is_dir()


def test_preserves_existing_aiws_trainer_location():
    from jarvis import config
    assert config.APPDATA_DIR == Path.home() / ".aiws_trainer"
    assert config.TEMP_DIR == Path("/tmp/vss_voice")


def test_gpu_mode_is_one_of_known_tiers():
    from jarvis import config
    assert config.GPU_MODE in {"24gb", "16gb", "8gb", "cpu"}


def test_vram_gb_is_non_negative_float():
    from jarvis import config
    assert isinstance(config.VRAM_GB, (int, float))
    assert config.VRAM_GB >= 0


def test_feature_flags_consistent_with_gpu_mode():
    from jarvis import config
    if config.GPU_MODE == "24gb":
        assert config.ENABLE_XTTS is True
        assert config.ENABLE_F5 is True
        assert config.ENABLE_SPEAKER_VERIFY is True
    elif config.GPU_MODE == "cpu":
        assert config.ENABLE_XTTS is False
        assert config.ENABLE_F5 is False


def test_whisper_model_name_valid():
    from jarvis import config
    assert config.WHISPER_MODEL in {"tiny", "base", "small", "medium"}


def test_log_uses_shared_logger():
    from jarvis import config
    assert callable(config.log)


def test_claude_bin_resolves_to_expanded_path():
    from jarvis import config
    assert "~" not in config.CLAUDE_BIN
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /home/hunterp/vss_env/bin/activate
python -m pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.config'`.

- [ ] **Step 3: Create `jarvis/config.py`**

```python
"""Centralized configuration — paths, GPU mode, feature flags, model config.

All modules import from here. No hardcoded paths outside this file.

Preserves existing data locations:
  APPDATA_DIR = ~/.aiws_trainer/   (NOT ~/.config/Jarvis/ — would lose data)
  TEMP_DIR    = /tmp/vss_voice/    (matches current log + IPC locations)
"""

import os
import subprocess
from pathlib import Path

from jarvis.jarvis_logging import get_logger

APPDATA_DIR = Path.home() / ".aiws_trainer"
TEMP_DIR = Path("/tmp/vss_voice")
SETTINGS_FILE = APPDATA_DIR / "voice_settings.json"
MEMORY_DIR = APPDATA_DIR / "jarvis_memory"
AGENT_DATA_DIR = APPDATA_DIR / "jarvis_data"
VOICEPRINT_FILE = APPDATA_DIR / "voiceprint.npz"
VOICE_REF = APPDATA_DIR / "jarvis_voice_ref.wav"
VOICE_REF_DIR = APPDATA_DIR / "jarvis_reference"
WAKEWORD_VERIFIER = APPDATA_DIR / "hey_jarvis_verifier.pkl"
INTENT_LOG = APPDATA_DIR / "intent_log.json"
SPEAK_QUEUE_KEY = APPDATA_DIR / "speak_queue.key"

LOG_FILE = TEMP_DIR / "gui_debug.log"
SPEAK_QUEUE_FILE = TEMP_DIR / "speak_queue.txt"
SCREEN_DIR = Path("/tmp/vss_screen")


def _ensure_dirs():
    """Create all managed directories. Called at import time."""
    for d in (APPDATA_DIR, TEMP_DIR, MEMORY_DIR, AGENT_DATA_DIR,
              VOICE_REF_DIR, SCREEN_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


_ensure_dirs()


def _detect_vram_gb() -> float:
    """Return max single-GPU VRAM in GiB via nvidia-smi, or 0 on failure."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return 0.0
        lines = [int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip()]
        if not lines:
            return 0.0
        return max(lines) / 1024
    except Exception:
        return 0.0


VRAM_GB = _detect_vram_gb()

if VRAM_GB >= 22:
    GPU_MODE = "24gb"
elif VRAM_GB >= 14:
    GPU_MODE = "16gb"
elif VRAM_GB >= 6:
    GPU_MODE = "8gb"
else:
    GPU_MODE = "cpu"


ENABLE_XTTS = GPU_MODE in {"24gb", "16gb"}
ENABLE_F5 = GPU_MODE == "24gb"
ENABLE_SPEAKER_VERIFY = GPU_MODE in {"24gb", "16gb", "8gb"}
ENABLE_PARAKEET = GPU_MODE in {"24gb", "16gb"}
ENABLE_HOTWORD = True

WHISPER_MODEL = {
    "24gb": "small",
    "16gb": "small",
    "8gb": "base",
    "cpu": "tiny",
}[GPU_MODE]

STT_GPU = 1 if GPU_MODE in {"24gb", "16gb"} else None
TTS_GPU = 1 if ENABLE_XTTS else None
SPEAKER_GPU = 1 if ENABLE_SPEAKER_VERIFY else None

CLAUDE_BIN = str(Path.home() / ".npm-global" / "bin" / "claude")
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"

log = get_logger("CONFIG")
log(f"GPU_MODE={GPU_MODE} VRAM={VRAM_GB:.1f}GiB "
    f"ENABLE_XTTS={ENABLE_XTTS} ENABLE_F5={ENABLE_F5} "
    f"ENABLE_SPEAKER_VERIFY={ENABLE_SPEAKER_VERIFY}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis/config.py tests/test_config.py
git commit -m "feat(config): centralize paths, GPU detection, feature flags

Port config.py pattern from Jarvis-Win. Linux path conventions
(~/.aiws_trainer/, /tmp/vss_voice/) preserve all existing user
data — voiceprint, memory, habits, settings stay in place.

GPU tier detection: 24gb (dual-3090) / 16gb / 8gb / cpu.
Feature flags derive from tier — XTTS, F5, speaker verify toggle
automatically.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `jarvis/desktop.py`

**Files:**
- Create: `jarvis/desktop.py`
- Create: `tests/test_desktop.py`

**Responsibility:** All xdotool/wmctrl/xclip operations live here. Other modules never shell out to these tools directly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_desktop.py`:

```python
"""Tests for DesktopController (xdotool/wmctrl/xclip abstraction)."""

from unittest.mock import MagicMock, patch

import pytest

from jarvis.desktop import DesktopController


def _fake_run(stdout="", returncode=0):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    return r


@patch("jarvis.desktop.subprocess.run")
def test_get_active_window_title_uses_xdotool(mock_run):
    mock_run.return_value = _fake_run(stdout="Terminal — bash\n")
    d = DesktopController()
    title = d.get_active_window_title()
    assert title == "Terminal — bash"
    args, kwargs = mock_run.call_args
    assert "xdotool" in args[0][0]
    assert "getactivewindow" in args[0]


@patch("jarvis.desktop.subprocess.run")
def test_list_windows_parses_wmctrl_output(mock_run):
    wmctrl_output = (
        "0x01234567  0 host  Firefox\n"
        "0x02345678  0 host  VS Code\n"
    )
    mock_run.return_value = _fake_run(stdout=wmctrl_output)
    d = DesktopController()
    windows = d.list_windows()
    assert len(windows) == 2
    assert windows[0] == ("0x01234567", "Firefox")
    assert windows[1] == ("0x02345678", "VS Code")


@patch("jarvis.desktop.subprocess.run")
def test_list_windows_falls_back_to_xdotool_when_wmctrl_missing(mock_run):
    mock_run.side_effect = [
        FileNotFoundError("wmctrl"),
        _fake_run(stdout="12345\n67890\n"),
        _fake_run(stdout="Firefox\n"),
        _fake_run(stdout="VS Code\n"),
    ]
    d = DesktopController()
    windows = d.list_windows()
    assert windows == [("12345", "Firefox"), ("67890", "VS Code")]


@patch("jarvis.desktop.subprocess.run")
def test_switch_to_window_matches_substring(mock_run):
    wmctrl_output = (
        "0x01234567  0 host  Firefox\n"
        "0x02345678  0 host  Visual Studio Code\n"
    )
    mock_run.side_effect = [
        _fake_run(stdout=wmctrl_output),
        _fake_run(returncode=0),
    ]
    d = DesktopController()
    ok = d.switch_to_window("code")
    assert ok is True
    assert mock_run.call_args_list[1].args[0][:2] == ["wmctrl", "-ia"]


@patch("jarvis.desktop.subprocess.run")
def test_type_text_calls_xdotool_type(mock_run):
    mock_run.return_value = _fake_run()
    d = DesktopController()
    d.type_text("Hello world")
    args, _ = mock_run.call_args
    assert args[0][0] == "xdotool"
    assert args[0][1] == "type"
    assert "Hello world" in args[0]


@patch("jarvis.desktop.subprocess.run")
def test_send_key_calls_xdotool_key(mock_run):
    mock_run.return_value = _fake_run()
    d = DesktopController()
    d.send_key("Return")
    args, _ = mock_run.call_args
    assert args[0] == ["xdotool", "key", "Return"]


@patch("jarvis.desktop.subprocess.run")
def test_copy_to_clipboard_pipes_to_xclip(mock_run):
    mock_run.return_value = _fake_run()
    d = DesktopController()
    d.copy_to_clipboard("some text")
    args, kwargs = mock_run.call_args
    assert args[0][0] == "xclip"
    assert kwargs.get("input") == "some text"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_desktop.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.desktop'`.

- [ ] **Step 3: Create `jarvis/desktop.py`**

```python
"""Linux Desktop Automation — xdotool/wmctrl/xclip abstraction.

Ports Jarvis-Win's DesktopController API to Linux tools.
"""

import subprocess
from typing import Optional

from jarvis.jarvis_logging import get_logger

_log = get_logger("DESKTOP")


class DesktopController:
    """Linux desktop automation via command-line tools."""

    def get_active_window_title(self) -> str:
        try:
            r = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            return r.stdout.strip()
        except Exception as e:
            _log(f"get_active_window_title error: {e}")
            return ""

    def list_windows(self) -> list[tuple[str, str]]:
        """Return [(wid, title)] for all visible windows."""
        try:
            r = subprocess.run(
                ["wmctrl", "-l"], capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                windows = []
                for line in r.stdout.strip().splitlines():
                    parts = line.split(None, 3)
                    if len(parts) >= 4:
                        wid, _, _, title = parts
                        if title and len(title) > 1:
                            windows.append((wid, title))
                return windows
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            r = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--name", ""],
                capture_output=True, text=True, timeout=3,
            )
            windows = []
            for wid in r.stdout.strip().splitlines():
                wid = wid.strip()
                if not wid:
                    continue
                try:
                    name_r = subprocess.run(
                        ["xdotool", "getwindowname", wid],
                        capture_output=True, text=True, timeout=1,
                    )
                    name = name_r.stdout.strip()
                    if name and len(name) > 1:
                        windows.append((wid, name))
                except Exception:
                    pass
            return windows
        except Exception as e:
            _log(f"list_windows error: {e}")
            return []

    def switch_to_window(self, name: str) -> bool:
        name_lower = name.lower()
        for wid, title in self.list_windows():
            if name_lower in title.lower():
                try:
                    subprocess.run(
                        ["wmctrl", "-ia", wid],
                        capture_output=True, timeout=2,
                    )
                    return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    try:
                        subprocess.run(
                            ["xdotool", "windowactivate", "--sync", wid],
                            capture_output=True, timeout=2,
                        )
                        return True
                    except Exception as e:
                        _log(f"switch_to_window fallback error: {e}")
                        return False
        return False

    def type_text(self, text: str, auto_enter: bool = False) -> None:
        try:
            subprocess.run(
                ["xdotool", "type", "--delay", "10", text],
                capture_output=True, timeout=30,
            )
            if auto_enter:
                self.send_key("Return")
        except Exception as e:
            _log(f"type_text error: {e}")

    def send_key(self, key: str) -> None:
        try:
            subprocess.run(
                ["xdotool", "key", key],
                capture_output=True, timeout=2,
            )
        except Exception as e:
            _log(f"send_key error: {e}")

    def copy_to_clipboard(self, text: str) -> None:
        try:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text, text=True, capture_output=True, timeout=2,
            )
        except Exception as e:
            _log(f"copy_to_clipboard error: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_desktop.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis/desktop.py tests/test_desktop.py
git commit -m "feat(desktop): Linux xdotool/wmctrl/xclip abstraction

Consolidates all scattered subprocess calls for window management,
keyboard automation, and clipboard into a single DesktopController.
wmctrl-first, xdotool-fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `jarvis/hotword.py`

**Files:**
- Create: `jarvis/hotword.py`
- Create: `tests/test_hotword.py`

**Responsibility:** Extract `HotwordListener` from `voice_input_gui.py` and decouple from the GUI via explicit callbacks. OpenWakeWord loading + dual-threshold confirmation preserved verbatim.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hotword.py`:

```python
"""Tests for HotwordListener (decoupled from GUI)."""

from unittest.mock import MagicMock

import pytest

from jarvis.hotword import HotwordListener


def _make_listener(**overrides):
    defaults = dict(
        mic_name="Default",
        mic_devices={"Default": None},
        is_recording=lambda: False,
        is_tts_speaking=lambda: False,
        is_model_ready=lambda: True,
        on_hotword_detected=MagicMock(),
        schedule=lambda delay_ms, fn: fn(),
    )
    defaults.update(overrides)
    return HotwordListener(**defaults)


def test_listener_starts_inactive():
    hl = _make_listener()
    assert hl.active is False


def test_start_marks_active():
    hl = _make_listener()
    hl.start()
    assert hl.active is True
    hl.stop()


def test_stop_marks_inactive():
    hl = _make_listener()
    hl.start()
    hl.stop()
    assert hl.active is False


def test_pause_clears_stream():
    hl = _make_listener()
    hl._stream = MagicMock()
    hl.pause()
    assert hl._stream is None


def test_resume_with_no_stream_sets_reopen_flag_when_active():
    hl = _make_listener()
    hl.active = True
    hl._stream = None
    hl.resume()
    assert hl._reopen is True


def test_resume_calls_start_when_inactive():
    hl = _make_listener()
    hl.active = False
    hl.start = MagicMock(side_effect=lambda: setattr(hl, "active", True))
    hl._stream = None
    hl.resume()
    hl.start.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_hotword.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.hotword'`.

- [ ] **Step 3: Create `jarvis/hotword.py`**

```python
"""HotwordListener — OpenWakeWord 'hey jarvis' detection with dual-threshold
confirmation. Extracted from voice_input_gui.py and decoupled from the GUI.
"""

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from jarvis import config
from jarvis.jarvis_logging import get_logger

_log = get_logger("HOTWORD")

CHANNELS = 1


class HotwordListener:
    """Always-on wake word listener using OpenWakeWord (CPU, ~1.5ms/prediction)."""

    THRESHOLD = 0.2
    CONFIRM_THRESHOLD = 0.15
    CONFIRM_WINDOW = 1.0

    def __init__(
        self,
        mic_name: str,
        mic_devices: dict,
        is_recording: Callable[[], bool],
        is_tts_speaking: Callable[[], bool],
        is_model_ready: Callable[[], bool],
        on_hotword_detected: Callable[[], None],
        schedule: Callable[[int, Callable[[], None]], None],
        verifier_path: Optional[Path] = None,
    ):
        self.mic_name = mic_name
        self.mic_devices = mic_devices
        self.is_recording = is_recording
        self.is_tts_speaking = is_tts_speaking
        self.is_model_ready = is_model_ready
        self.on_hotword_detected = on_hotword_detected
        self.schedule = schedule
        self.verifier_path = verifier_path or config.WAKEWORD_VERIFIER

        self.active = False
        self._stream = None
        self._model = None
        self._reopen = False
        self._pending_hotword: Optional[float] = None
        self._resume_cooldown: float = 0.0

    def start(self):
        if self.active:
            return
        self.active = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        _log("Hotword listener started")

    def stop(self):
        self.active = False
        self._close_stream()
        _log("Hotword listener stopped")

    def pause(self):
        self._close_stream()
        _log("Hotword stream paused (mic released)")

    def resume(self):
        if self._stream:
            return
        if self.active:
            self._reopen = True
            _log("Hotword stream will resume")
        else:
            self.start()
            _log("Hotword listener restarted fresh")

    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _listen_loop(self):
        import sounddevice as sd

        if self._model is None:
            try:
                from openwakeword.model import Model
                self._model = Model()
                _log(f"OpenWakeWord loaded: {list(self._model.models.keys())}")

                if self.verifier_path.exists():
                    import joblib
                    self._model.custom_verifier_models["hey_jarvis"] = \
                        joblib.load(str(self.verifier_path))
                    self._model.custom_verifier_threshold = 0.3
                    _log(f"Custom wake word verifier attached: {self.verifier_path}")
            except Exception as e:
                _log(f"OpenWakeWord load error: {e}")
                self.active = False
                return

        self._reopen = False
        mic_idx = self.mic_devices.get(self.mic_name)

        try:
            dev_info = sd.query_devices(mic_idx, 'input')
            native_rate = int(dev_info['default_samplerate'])
        except Exception:
            native_rate = 44100
        _log(f"Hotword mic rate: {native_rate}Hz")

        chunk_samples = int(native_rate * 0.08)
        buf = deque(maxlen=int(native_rate * 2))

        def callback(indata, frame_count, time_info, status):
            if self.active and not self.is_recording():
                chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                buf.extend(chunk.tolist())

        def _open_stream():
            try:
                self._stream = sd.InputStream(
                    samplerate=native_rate, channels=CHANNELS,
                    dtype="float32", device=mic_idx,
                    callback=callback,
                    blocksize=chunk_samples,
                )
                self._stream.start()
                return True
            except Exception as e:
                _log(f"Hotword stream error: {e}")
                self._stream = None
                return False

        if not _open_stream():
            self.active = False
            return

        while self.active:
            time.sleep(0.08)

            if self._reopen and not self.is_recording() and not self._stream:
                self._reopen = False
                buf.clear()
                _open_stream()
                _log("Hotword stream resumed")
                if self._model:
                    self._model.reset()
                self._resume_cooldown = time.monotonic() + 2.0

            if self.is_recording() or not self._stream:
                continue

            if self.is_tts_speaking():
                buf.clear()
                continue

            if time.monotonic() < self._resume_cooldown:
                buf.clear()
                continue

            if len(buf) < chunk_samples:
                continue

            raw = np.array(list(buf)[-chunk_samples:], dtype=np.float32)
            if native_rate != 16000:
                from scipy.signal import resample as scipy_resample
                new_len = int(len(raw) * 16000 / native_rate)
                raw = scipy_resample(raw, new_len).astype(np.float32)
            audio_int16 = (raw * 32767).astype(np.int16)

            try:
                predictions = self._model.predict(audio_int16)
            except Exception:
                continue

            score = max(
                predictions.get("hey_jarvis", 0.0),
                predictions.get("hey_mycroft", 0.0) * 0.7,
            )
            if score >= 0.1:
                _log(f"Hotword score: {score:.3f}")

            now = time.monotonic()
            if score >= self.THRESHOLD:
                _log(f"Hotword detected (strong, score={score:.3f})")
                buf.clear()
                self._model.reset()
                self._pending_hotword = None
                self._fire_detected()
                time.sleep(1.5)
            elif score >= self.CONFIRM_THRESHOLD:
                if (self._pending_hotword
                        and (now - self._pending_hotword) < self.CONFIRM_WINDOW):
                    _log(f"Hotword confirmed (2 frames, score={score:.3f})")
                    buf.clear()
                    self._model.reset()
                    self._pending_hotword = None
                    self._fire_detected()
                    time.sleep(1.5)
                else:
                    self._pending_hotword = now
            else:
                if (self._pending_hotword
                        and (now - self._pending_hotword) > self.CONFIRM_WINDOW):
                    self._pending_hotword = None

    def _fire_detected(self):
        if self.is_model_ready():
            self.schedule(0, self.on_hotword_detected)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_hotword.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis/hotword.py tests/test_hotword.py
git commit -m "feat(hotword): extract HotwordListener with explicit callback interface

Decoupled from VoiceInputGUI via callables. Preserves dual-threshold
confirmation logic (Phase 5 wake-word hardening).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `jarvis/commander.py`

**Files:**
- Create: `jarvis/commander.py`
- Create: `tests/test_commander.py`

**Responsibility:** Intent router on top of `dispatcher.CommandDispatcher`. Adds dictation-mode toggle and brain routing for unmatched text.

- [ ] **Step 1: Write the failing test**

Create `tests/test_commander.py`:

```python
"""Tests for Commander (routes text to handlers / dictation / brain)."""

from unittest.mock import MagicMock

import pytest

from jarvis.commander import Commander
from jarvis.dispatcher import CommandDispatcher, CommandHandler


def _make_commander(handlers=None, brain=None, desktop=None, tts=None):
    brain = brain or MagicMock()
    dispatcher = CommandDispatcher(handlers=handlers or [], brain=brain)
    return Commander(
        dispatcher=dispatcher,
        desktop=desktop or MagicMock(),
        tts=tts or MagicMock(),
    )


def test_empty_text_returns_false():
    c = _make_commander()
    assert c.process("") is False
    assert c.process("   ") is False


def test_start_dictation_toggles_mode_on():
    tts = MagicMock()
    c = _make_commander(tts=tts)
    assert c.process("start dictation") is True
    assert c.dictation_mode is True
    tts.speak.assert_called()


def test_stop_dictation_toggles_mode_off():
    c = _make_commander()
    c.dictation_mode = True
    assert c.process("stop dictation") is True
    assert c.dictation_mode is False


def test_dictation_mode_types_text_instead_of_dispatching():
    desktop = MagicMock()
    c = _make_commander(desktop=desktop)
    c.dictation_mode = True
    c.process("the quick brown fox")
    desktop.type_text.assert_called_once_with("the quick brown fox", auto_enter=False)


def test_handler_match_routes_to_handler():
    called = []
    handlers = [
        CommandHandler(r"^show time$", lambda m, ctx: called.append("time") or True),
    ]
    c = _make_commander(handlers=handlers)
    assert c.process("show time") is True
    assert called == ["time"]


def test_unmatched_text_falls_through_to_brain():
    brain = MagicMock()
    c = _make_commander(brain=brain)
    c.process("what is the meaning of life")
    brain.handle.assert_called_once()


def test_dictation_toggles_are_case_insensitive():
    c = _make_commander()
    c.process("START DICTATION")
    assert c.dictation_mode is True
    c.process("Stop Dictation")
    assert c.dictation_mode is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_commander.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.commander'`.

- [ ] **Step 3: Create `jarvis/commander.py`**

```python
"""Commander — intent router on top of the CommandDispatcher registry.

Adds dictation-mode toggle and brain fallback for unmatched text.
"""

from jarvis.dispatcher import CommandDispatcher
from jarvis.jarvis_logging import get_logger

_log = get_logger("COMMANDER")


class Commander:
    """Routes transcribed text through dictation → dispatcher → brain."""

    DICTATION_ON = {
        "start dictation", "dictation mode", "start dictation mode",
        "begin dictation", "begin dictation mode",
    }
    DICTATION_OFF = {
        "stop dictation", "end dictation", "stop dictation mode",
        "end dictation mode",
    }

    def __init__(self, dispatcher: CommandDispatcher, desktop=None, tts=None):
        self.dispatcher = dispatcher
        self.desktop = desktop
        self.tts = tts
        self.dictation_mode = False

    def process(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        text = text.strip()
        lower = text.lower().rstrip(".!?")
        _log(f"Processing: {text[:60]}")

        if lower in self.DICTATION_ON:
            self.dictation_mode = True
            self._speak("Dictation mode enabled. I will type everything "
                        "you say. Say stop dictation to end.")
            return True
        if lower in self.DICTATION_OFF:
            self.dictation_mode = False
            self._speak("Dictation mode disabled.")
            return True

        if self.dictation_mode:
            if self.desktop:
                self.desktop.type_text(text, auto_enter=False)
            return True

        self.dispatcher.handle(text, ctx={"lower": lower})
        return True

    def _speak(self, text: str):
        if self.tts is not None:
            try:
                self.tts.speak(text)
            except Exception as e:
                _log(f"TTS error in _speak: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_commander.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis/commander.py tests/test_commander.py
git commit -m "feat(commander): intent router with dictation mode

Wraps CommandDispatcher with dictation-mode toggle. Unmatched text
falls through to the brain via dispatcher.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Wire new modules into `voice_input_gui.py` (no-op)

**Files:**
- Modify: `jarvis/voice_input_gui.py`

**Responsibility:** Replace the inline `HotwordListener` class with a thin adapter around `jarvis.hotword.HotwordListener`. Instantiate `DesktopController`. No user-visible behavior change.

- [ ] **Step 1: Confirm current tests pass before editing**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 2: In `jarvis/voice_input_gui.py`, add imports (after existing `from jarvis.*` block)**

```python
from jarvis.desktop import DesktopController
from jarvis.hotword import HotwordListener as _NewHotwordListener
```

- [ ] **Step 3: Replace the inline `class HotwordListener:` (around line 785-1006) with a thin adapter**

Replace the entire `class HotwordListener:` block with:

```python
class HotwordListener:
    """Thin adapter — delegates to jarvis.hotword.HotwordListener."""

    THRESHOLD = _NewHotwordListener.THRESHOLD
    CONFIRM_THRESHOLD = _NewHotwordListener.CONFIRM_THRESHOLD
    CONFIRM_WINDOW = _NewHotwordListener.CONFIRM_WINDOW

    def __init__(self, gui):
        self.gui = gui
        self._inner = _NewHotwordListener(
            mic_name=gui.mic_var.get(),
            mic_devices=gui._mic_devices,
            is_recording=lambda: gui.recording,
            is_tts_speaking=lambda: bool(
                getattr(gui, '_tts', None) and gui._tts.is_speaking
            ),
            is_model_ready=lambda: gui.model_loaded,
            on_hotword_detected=self._on_hotword,
            schedule=lambda delay_ms, fn: gui.root.after(delay_ms, fn),
        )

    @property
    def active(self):
        return self._inner.active

    def start(self):
        self._inner.mic_name = self.gui.mic_var.get()
        self._inner.mic_devices = self.gui._mic_devices
        self._inner.start()

    def stop(self):
        self._inner.stop()

    def pause(self):
        self._inner.pause()

    def resume(self):
        self._inner.resume()

    def _on_hotword(self):
        if not self.gui.recording and self.gui.model_loaded:
            self.pause()
            self.gui._set_status("Hotword!", self.gui.BLUE, "Starting recording...")
            if self.gui.sound_var.get():
                threading.Thread(
                    target=_play_beep, args=(_BEEP_START,), daemon=True
                ).start()
            self.gui.root.after(200, self.gui._start_recording)
```

- [ ] **Step 4: Add `self._desktop = DesktopController()` to `VoiceInputGUI.__init__`**

Near the end of `__init__`, before the last few lines, insert:

```python
        self._desktop = DesktopController()
```

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: all tests still pass.

- [ ] **Step 6: Manual syntax + import check**

```bash
python -c "import jarvis.voice_input_gui; print('import OK')"
```

Expected: `import OK`.

- [ ] **Step 7: Commit**

```bash
git add jarvis/voice_input_gui.py
git commit -m "refactor: wire voice_input_gui.py to new backend modules

Replace inline HotwordListener class with a thin adapter around
jarvis.hotword.HotwordListener. Instantiate DesktopController in
__init__ for future use by command handlers.

No behavior change — all callbacks wired to preserve the gui._hotword
API surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# PHASE 2 — HUD

Build the pywebview frontend as a parallel entry point. Tkinter GUI is unaffected.

## Task 6: `frontend/` — port HTML/CSS/JS from Jarvis-Win

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/css/hud.css`
- Create: `frontend/js/app.js`
- Create: `frontend/js/reactor.js`

**Responsibility:** Port the web HUD from Jarvis-Win verbatim.

- [ ] **Step 1: Fetch the four frontend files from Jarvis-Win**

```bash
mkdir -p /home/hunterp/jarvis/frontend/css /home/hunterp/jarvis/frontend/js

for pair in \
    "frontend/index.html frontend/index.html" \
    "frontend/css/hud.css frontend/css/hud.css" \
    "frontend/js/app.js frontend/js/app.js" \
    "frontend/js/reactor.js frontend/js/reactor.js"; do
    remote=$(echo $pair | cut -d' ' -f1)
    local=$(echo $pair | cut -d' ' -f2)
    gh api "repos/CptKincaid/Jarvis-Win/contents/$remote" \
        --jq '.content' | base64 -d > "/home/hunterp/jarvis/$local"
    echo "$local: $(wc -l < /home/hunterp/jarvis/$local) lines"
done
```

Expected: 4 files created with non-zero line counts.

- [ ] **Step 2: Verify the frontend files are valid**

```bash
python -c "
from pathlib import Path
for f in ['frontend/index.html', 'frontend/css/hud.css',
          'frontend/js/app.js', 'frontend/js/reactor.js']:
    p = Path(f)
    assert p.exists(), f'missing: {f}'
    content = p.read_text()
    assert len(content) > 100, f'too short: {f}'
    if f.endswith('.html'):
        assert '<html' in content
    elif f.endswith('.css'):
        assert '{' in content and '}' in content
    elif f.endswith('.js'):
        assert 'function' in content or 'const' in content or '=>' in content
print('all 4 frontend files valid')
"
```

Expected: `all 4 frontend files valid`.

- [ ] **Step 3: Commit**

```bash
git add frontend/
git commit -m "feat(hud): port frontend HTML/CSS/JS from Jarvis-Win

Four files copied verbatim from the Windows repo — the web-layer
code is platform-independent. Bridge wiring lands in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `jarvis/app.py` — pywebview launcher + `JarvisAPI` bridge

**Files:**
- Create: `jarvis/app.py`
- Create: `tests/test_jarvis_api.py`

**Responsibility:** pywebview entry point that launches the HUD, plus the `JarvisAPI` class JS calls via `pywebview.api.*`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_jarvis_api.py`:

```python
"""Tests for JarvisAPI (the JS↔Python bridge class)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def api():
    from jarvis.app import JarvisAPI
    return JarvisAPI()


def test_initial_status(api):
    status = api.get_status()
    assert status["status"] == "Initializing"
    assert status["recording"] is False
    assert status["user_text"] == ""
    assert status["jarvis_text"] == ""


def test_get_amplitude_returns_float(api):
    amp = api.get_amplitude()
    assert isinstance(amp, float)
    assert 0.0 <= amp <= 1.0


def test_set_user_text_updates_status(api):
    api._set_user_text("hello world")
    assert api.get_status()["user_text"] == "hello world"


def test_set_jarvis_text_updates_status(api):
    api._set_jarvis_text("all systems nominal")
    assert api.get_status()["jarvis_text"] == "all systems nominal"


def test_set_status_updates_status_and_color(api):
    api._set_status("Recording", "#22d3ee")
    status = api.get_status()
    assert status["status"] == "Recording"
    assert status["status_color"] == "#22d3ee"


def test_init_modules_is_idempotent(api):
    with patch.object(api, "_load_backend_modules") as mock_load:
        r1 = api.init_modules()
        r2 = api.init_modules()
        assert r1["ok"] is True
        assert r2.get("skipped") is True
        mock_load.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_jarvis_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.app'`.

- [ ] **Step 3: Create `jarvis/app.py`**

```python
"""Jarvis HUD — pywebview launcher with a JS↔Python bridge.

Usage:
    python -m jarvis.app           # HUD mode (default)
    python -m jarvis.app --legacy  # fall back to Tkinter GUI
"""

import argparse
import os
import sys
import threading
from pathlib import Path

from jarvis import config
from jarvis.jarvis_logging import get_logger

_log = get_logger("APP")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"


class JarvisAPI:
    """JS-callable API exposing Jarvis modules to the web frontend."""

    def __init__(self):
        self._desktop = None
        self._memory = None
        self._tts = None
        self._recorder = None
        self._pipeline = None
        self._stt = None
        self._speaker = None
        self._brain = None
        self._agent = None
        self._commander = None
        self._hotword = None

        self._state_lock = threading.Lock()
        self._status = "Initializing"
        self._status_color = "#ffaa00"
        self._status_detail = ""
        self._recording = False
        self._audio_level = 0.0
        self._user_text = ""
        self._jarvis_text = ""
        self._modules_loaded = False

    def init_modules(self):
        if self._modules_loaded:
            return {"ok": True, "skipped": True}
        try:
            self._load_backend_modules()
            self._modules_loaded = True
            self._set_status("Ready", "#22c55e")
            _log("Modules loaded")
            return {"ok": True}
        except Exception as e:
            _log(f"init_modules error: {e}")
            return {"ok": False, "error": str(e)}

    def get_status(self):
        with self._state_lock:
            return {
                "status": self._status,
                "status_color": self._status_color,
                "status_detail": self._status_detail,
                "recording": self._recording,
                "user_text": self._user_text,
                "jarvis_text": self._jarvis_text,
                "gpu_info": f"{config.GPU_MODE} | {config.VRAM_GB:.1f}GB | "
                            f"{config.WHISPER_MODEL}",
            }

    def get_amplitude(self):
        return float(self._audio_level)

    def start_recording(self):
        if not self._modules_loaded:
            return {"ok": False, "error": "modules not loaded"}
        if self._recording:
            return {"ok": False, "error": "already recording"}
        try:
            self._recording = True
            self._set_status("Listening...", "#22d3ee", "Speak now")
            self._recorder.start()
            return {"ok": True}
        except Exception as e:
            self._recording = False
            _log(f"start_recording error: {e}")
            return {"ok": False, "error": str(e)}

    def stop_recording(self):
        if not self._recording:
            return {"ok": False, "error": "not recording"}
        try:
            audio = self._recorder.stop()
            self._recording = False
            self._set_status("Processing...", "#facc15")
            threading.Thread(
                target=self._transcribe_and_dispatch,
                args=(audio,), daemon=True,
            ).start()
            return {"ok": True}
        except Exception as e:
            self._recording = False
            _log(f"stop_recording error: {e}")
            return {"ok": False, "error": str(e)}

    def close_window(self):
        _log("close_window called")
        try:
            if self._hotword:
                self._hotword.stop()
            if self._recorder and self._recording:
                self._recorder.stop()
        except Exception as e:
            _log(f"close_window cleanup error: {e}")
        try:
            import webview
            for w in webview.windows:
                w.destroy()
        except Exception:
            pass

    def _set_status(self, status, color, detail=""):
        with self._state_lock:
            self._status = status
            self._status_color = color
            self._status_detail = detail

    def _set_user_text(self, text):
        with self._state_lock:
            self._user_text = text

    def _set_jarvis_text(self, text):
        with self._state_lock:
            self._jarvis_text = text

    def _on_amplitude(self, amp):
        self._audio_level = amp

    def _load_backend_modules(self):
        from jarvis.desktop import DesktopController
        from jarvis.memory import JarvisMemory
        from jarvis.jarvis_tts import JarvisTTS
        from jarvis.recording import RecordingController
        from jarvis.transcription import TranscriptionPipeline
        from jarvis.stt_engine import STTEngine
        from jarvis.speaker_verification import SpeakerVerifier
        from jarvis.jarvis_brain import JarvisBrain
        from jarvis.jarvis_agent import JarvisAgent
        from jarvis.dispatcher import CommandDispatcher
        from jarvis.commander import Commander

        self._desktop = DesktopController()
        self._memory = JarvisMemory()
        self._tts = JarvisTTS(engine="kokoro" if config.VRAM_GB < 22 else "f5")

        self._recorder = RecordingController(
            sample_rate=16000,
            on_amplitude=self._on_amplitude,
        )

        self._stt = STTEngine(gpu=config.STT_GPU)
        threading.Thread(target=self._stt.load, daemon=True).start()

        self._speaker = (SpeakerVerifier(gpu=config.SPEAKER_GPU)
                          if config.ENABLE_SPEAKER_VERIFY else None)

        self._pipeline = TranscriptionPipeline(
            stt_engine=self._stt,
            speaker_verifier=self._speaker,
        )

        self._agent = JarvisAgent()
        self._brain = JarvisBrain()

        dispatcher = CommandDispatcher(handlers=[], brain=self._brain)
        self._commander = Commander(
            dispatcher=dispatcher,
            desktop=self._desktop,
            tts=self._tts,
        )

    def _transcribe_and_dispatch(self, audio):
        try:
            result = self._pipeline.transcribe(audio)
            if not result.text:
                self._set_status("Ready", "#22c55e", "No speech detected")
                return
            self._set_user_text(result.text)
            self._set_status("Thinking...", "#a855f7")
            self._commander.process(result.text)
        except Exception as e:
            _log(f"_transcribe_and_dispatch error: {e}")
            self._set_status("Error", "#ef4444", str(e)[:40])


def _launch_legacy():
    _log("Legacy mode — launching Tkinter GUI")
    os.execvp(
        sys.executable,
        [sys.executable, "-m", "jarvis.voice_input_gui"] + sys.argv[1:],
    )


def main():
    parser = argparse.ArgumentParser(description="Jarvis HUD launcher")
    parser.add_argument("--legacy", action="store_true",
                        help="Run the Tkinter legacy GUI instead of HUD")
    args, _rest = parser.parse_known_args()

    if args.legacy:
        _launch_legacy()
        return

    import webview

    api = JarvisAPI()
    window = webview.create_window(
        "Jarvis",
        str(INDEX_HTML),
        js_api=api,
        width=460, height=720,
        frameless=True, easy_drag=True,
        on_top=False,
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_jarvis_api.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 6: Import sanity check**

```bash
python -c "from jarvis.app import JarvisAPI, main; print('import OK')"
```

Expected: `import OK`.

- [ ] **Step 7: Commit**

```bash
git add jarvis/app.py tests/test_jarvis_api.py
git commit -m "feat(hud): pywebview launcher + JarvisAPI bridge

Entry point for 'python -m jarvis.app'. Opens a frameless pywebview
window and exposes JarvisAPI as the JS-callable bridge. JarvisAPI
lazy-loads all backend modules on first JS call.

--legacy flag execs into the Tkinter GUI for fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Manual HUD smoke test

**Files:**
- None (runtime verification only)

- [ ] **Step 1: Launch the HUD**

```bash
source /home/hunterp/vss_env/bin/activate
python -m jarvis.app
```

Expected: a frameless window appears with the Jarvis HUD. The reactor canvas is visible. Status starts as "Initializing" then switches to "Ready".

- [ ] **Step 2: Click Record, speak a short phrase, stop recording**

Expected:
- Status: "Listening..." while recording
- Status: "Processing..." after stop
- "You:" panel populates with transcript
- "Jarvis:" panel populates with response
- TTS plays audio response

- [ ] **Step 3: Close the window via the × button**

Expected: clean exit, no error dialogs, process terminates within 3s.

- [ ] **Step 4: Relaunch with `--legacy` and smoke-test the Tkinter GUI**

```bash
python -m jarvis.app --legacy
```

Expected: the original Tkinter GUI opens. Record/stop/transcribe still works.

- [ ] **Step 5: Document any observed bugs in TODO.md (next task)**

---

# PHASE 3 — Housekeeping

## Task 9: `TODO.md` + `CLAUDE.md` updates

**Files:**
- Create: `TODO.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create `TODO.md`**

```markdown
# Jarvis — TODO

## Open bugs
_(none at port time — add HUD smoke-test findings here)_

## Backlog (not blocking)

### Architecture follow-ups
- Wire `CommandDispatcher` registry up to actual commands (progressive migration of `_check_quick_command`)
- Switch `hotword_daemon.py` to launch `python -m jarvis.app --auto-record` once the HUD has an `--auto-record` flag
- Move `jarvis_brain.py` and `jarvis_agent.py` to consume `config.py` paths rather than their current inline `APPDATA_DIR` definitions
- Re-enable the reverted security fixes from `jarvis-review-fixes` (shell allowlist, HMAC speak queue) once confidence in the HUD path is established

### "Extension of me" / Claude tool use
- Replace `claude -p --output-format text` with Claude Code agent invocation so Jarvis can edit/create files directly rather than just speaking responses
- Add `[EDIT file:path]` / `[WRITE file:path]` / `[READ file:path]` action verbs to the brain grammar
- Persist conversation history across sessions (lightweight JSON ring buffer in `MEMORY_DIR`)

## Architecture docs
- `docs/superpowers/specs/` — design docs
- `docs/superpowers/plans/` — implementation plans
- `CLAUDE.md` — stack and running instructions
```

- [ ] **Step 2: Update `CLAUDE.md`**

Replace the entire contents of `CLAUDE.md` with:

```markdown
# CLAUDE.md — Jarvis AI Voice Assistant

## Project Overview
Personal AI voice assistant with desktop control, speaker verification, XTTS/F5/Kokoro voice synthesis, and proactive intelligence. Built on Python with CUDA GPU acceleration. Two frontends: pywebview HUD (default) and Tkinter GUI (legacy fallback).

## Stack
Python 3.12 | CUDA 12.6 (2× RTX 3090) | pywebview HUD (HTML/CSS/JS) | Tkinter legacy | NVIDIA Parakeet-TDT STT | NVIDIA TitaNet-Large speaker verification | Kokoro 82M + F5-TTS + Edge TTS + XTTS v2 | OpenWakeWord | Ollama | Claude CLI

## Project Structure
```
jarvis/
  app.py                  # MAIN ENTRY: pywebview HUD launcher + JarvisAPI bridge
  voice_input_gui.py      # LEGACY: Tkinter GUI, runnable via --legacy
  config.py               # Centralized paths, GPU mode, feature flags
  commander.py            # Intent router + dictation mode
  dispatcher.py           # Regex command registry
  desktop.py              # xdotool/wmctrl/xclip abstraction
  recording.py            # Mic stream + silence detection
  transcription.py        # Speaker filter → STT → intent
  animation.py            # Reactor frame generation
  hotword.py              # OpenWakeWord dual-threshold detection
  stt_engine.py           # Parakeet primary / Whisper fallback
  speaker_verification.py # TitaNet-Large voiceprint
  jarvis_tts.py           # Kokoro / F5 / Edge / XTTS
  jarvis_brain.py         # Ollama + Claude CLI hybrid
  jarvis_agent.py         # Proactive intelligence
  memory.py               # Persistent JSON memory
  context.py              # Git + window + system context
  jarvis_logging.py       # Shared prefix-bound logger
  jarvis_speak_queue.py   # File-based TTS IPC
  speak_queue_auth.py     # HMAC-signed queue (not wired in main path)

frontend/
  index.html              # HUD markup
  css/hud.css             # Glassmorphism styles
  js/app.js               # pywebview bridge polling loop
  js/reactor.js           # Canvas2D arc-reactor animation

scripts/
  hotword_daemon.py       # Systemd-style wake word listener (launches GUI)
  screen_capture.py       # Ctrl+Shift+S screenshot daemon

tests/                    # pytest — 100+ tests, all must stay green
docs/superpowers/
  specs/                  # design documents
  plans/                  # implementation plans
TODO.md                   # bug tracking + backlog
```

## Running
```bash
source /home/hunterp/vss_env/bin/activate
python -m jarvis.app          # HUD (default)
python -m jarvis.app --legacy # Tkinter fallback
python scripts/hotword_daemon.py --daemon  # Background wake word
```

## Key Patterns
- All paths flow through `jarvis.config` — no hardcoded `/home/hunterp/` literals
- User data in `~/.aiws_trainer/` (voiceprint, memory, settings, voice ref)
- Temp files in `/tmp/vss_voice/` (logs, speak queue)
- GPU features auto-toggle via `config.GPU_MODE` (24gb / 16gb / 8gb / cpu)
- YOLO/Whisper on GPU 0 (legacy), Parakeet/XTTS/SpeechBrain on GPU 1
- Tkinter vars are NOT thread-safe — always cache `.get()` before passing to audio callbacks
- Debug logs: `/tmp/vss_voice/gui_debug.log`

## Testing
```bash
python -m pytest tests/ -q
```
All tests must pass on every commit.
```

- [ ] **Step 3: Commit**

```bash
git add TODO.md CLAUDE.md
git commit -m "docs: TODO.md + updated CLAUDE.md for new architecture

TODO.md tracks open bugs and backlog (including the deferred
'extension of me' Claude tool-use work). CLAUDE.md reflects the
new module layout and HUD entry point.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Final verification + merge to `main`

- [ ] **Step 1: Full test run**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Grep for regressions**

```bash
grep -rn "/home/hunterp" jarvis/ --include="*.py" | head
grep -rn "self._whisper_model.transcribe" jarvis/ --include="*.py" | head
grep -c "def _log" jarvis/*.py | grep -v ":0"
```

Expected: zero matches for the first two; third shows only `jarvis_logging.py` if anything.

- [ ] **Step 3: Line count check**

```bash
wc -l jarvis/*.py
```

Expected: `voice_input_gui.py` should still compile but is no longer the primary entry point.

- [ ] **Step 4: HUD final smoke test (from Task 8)**

Repeat Task 8 steps after all other phases are complete.

- [ ] **Step 5: Merge**

```bash
git checkout main
git merge --no-ff linux-hud -m "Merge branch 'linux-hud'

Port Jarvis-Win's modular architecture to Linux:
- pywebview HUD replaces Tkinter as default frontend
- jarvis/config.py centralizes all paths and GPU detection
- jarvis/commander.py routes intent (with dictation mode)
- jarvis/desktop.py abstracts xdotool/wmctrl/xclip
- jarvis/hotword.py extracts HotwordListener with callback interface
- frontend/ hosts the web HUD (HTML/CSS/JS + Canvas2D reactor)

Tkinter GUI kept as --legacy fallback. AI stack unchanged
(Parakeet + TitaNet + Kokoro/F5 + intent learning + dual-threshold
wake word).

Specs: docs/superpowers/specs/2026-04-16-linux-hud-port-design.md
Plan: docs/superpowers/plans/2026-04-16-linux-hud-port.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Push to GitHub**

```bash
git push origin main
git push origin linux-hud
```

---

## Summary

- Task 0: branch prep (0 commits — setup only)
- Phase 1 (Tasks 1-5): 5 backend modules + tests (5 commits)
- Phase 2 (Tasks 6-8): HUD frontend + app.py + smoke test (2 commits + manual verify)
- Phase 3 (Tasks 9-10): TODO, CLAUDE.md, merge (2 commits)

Total: ~9 commits, 5 new Python modules + 4 web files + 2 doc files.
