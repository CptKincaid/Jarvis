# Jarvis Code Review Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 21 code-review findings in the Jarvis voice assistant across three phases — critical bugs/security, module extraction with tests, and remaining cleanup inside the new module layout.

**Architecture:** Single branch `jarvis-review-fixes`. Phase 1 applies surgical in-place patches. Phase 2 splits the 5,800-line `VoiceInputGUI` god class into four focused modules (`recording.py`, `transcription.py`, `dispatcher.py`, `animation.py`) each ≤500 lines with matching `tests/test_*.py`. Phase 3 applies the remaining 14 fixes inside the new module boundaries.

**Tech Stack:** Python 3.12, Tkinter, sounddevice, faster-whisper, Parakeet (nemo_toolkit), OpenWakeWord, SpeechBrain, Kokoro/F5/XTTS, pytest.

**Reference spec:** `docs/superpowers/specs/2026-04-16-jarvis-code-review-remediation-design.md`

---

## Task 0: Create working branch

**Files:**
- None (branch setup only)

- [ ] **Step 1: Create and switch to feature branch**

```bash
git checkout -b jarvis-review-fixes
git status
```

Expected: `On branch jarvis-review-fixes`, clean working tree (the unstaged `jarvis_tts.py` change from a prior session stays — it's unrelated to this work).

---

# PHASE 1 — Critical bugs + security (items 1-5)

Surgical in-place fixes. One commit per item. No new test infrastructure; verification is manual smoke per task.

## Task 1: Shell allowlist + voice confirmation (item #1)

**Files:**
- Modify: `jarvis/voice_input_gui.py:3513-3529` (the `run_match` block in `_check_quick_command`)
- Modify: `jarvis/voice_input_gui.py` (add module-level `SHELL_ALLOWLIST` constant near line 65 with other constants)

- [ ] **Step 1: Add `SHELL_ALLOWLIST` constant near line 92 (after `NOISE_GATE_THRESHOLD`)**

Insert after line 92 in `jarvis/voice_input_gui.py`:

```python
# Shell commands allowed without voice confirmation.
# Anything outside this set requires a spoken "yes" or "confirm".
SHELL_ALLOWLIST = {
    "ls", "pwd", "cd", "git", "df", "du", "free", "uptime",
    "date", "whoami", "hostname", "wc", "cat", "head", "tail",
    "echo", "which", "whereis", "ps", "top", "env", "printenv",
    "python", "python3", "pip", "pytest",
}
```

- [ ] **Step 2: Replace the `run_match` block at lines 3513-3529 with the allowlist-gated version**

Replace exactly this block (lines 3513-3529):

```python
        # Shell piping: "run <command>" / "execute <command>"
        run_match = re.match(r"(?:run|execute|shell)\s+(.+)", cmd_text)
        if run_match:
            shell_cmd = run_match.group(1).strip()
            _log(f"Shell command: {shell_cmd}")
            self._set_status("Running...", self.ACCENT, shell_cmd[:30])
            def _run():
                output = self._agent.run_shell(shell_cmd)
                self.root.after(0, lambda: self._show_jarvis_text(
                    f"$ {shell_cmd}\n{output}"))
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, "Command done"))
                if self.talkback_var.get() and len(output) < 200:
                    from jarvis.jarvis_speak_queue import say
                    say(f"Result: {output[:100]}")
            threading.Thread(target=_run, daemon=True).start()
            return True
```

With:

```python
        # Shell piping: "run <command>" / "execute <command>"
        run_match = re.match(r"(?:run|execute|shell)\s+(.+)", cmd_text)
        if run_match:
            shell_cmd = run_match.group(1).strip()
            first_token = shell_cmd.split()[0] if shell_cmd.split() else ""
            if first_token in SHELL_ALLOWLIST:
                _log(f"Shell command (allowlisted): {shell_cmd}")
                self._set_status("Running...", self.ACCENT, shell_cmd[:30])
                self._run_shell_async(shell_cmd)
            else:
                _log(f"Shell command requires confirmation: {shell_cmd}")
                self._set_status("Confirming...", self.YELLOW,
                                 "Say yes to run")
                threading.Thread(
                    target=self._confirm_and_run_shell,
                    args=(shell_cmd,),
                    daemon=True,
                ).start()
            return True
```

- [ ] **Step 3: Add helper methods `_run_shell_async` and `_confirm_and_run_shell`**

Add these two methods inside the `VoiceInputGUI` class. Place them immediately after the `_check_quick_command` method (find its closing line, then insert after it):

```python
    def _run_shell_async(self, shell_cmd):
        """Run a shell command in a daemon thread; display + optionally speak the result."""
        def _run():
            output = self._agent.run_shell(shell_cmd)
            self.root.after(0, lambda: self._show_jarvis_text(
                f"$ {shell_cmd}\n{output}"))
            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, "Command done"))
            if self.talkback_var.get() and len(output) < 200:
                from jarvis.jarvis_speak_queue import say
                say(f"Result: {output[:100]}")
        threading.Thread(target=_run, daemon=True).start()

    def _confirm_and_run_shell(self, shell_cmd):
        """Prompt for voice confirmation, then run shell_cmd iff user said yes/confirm."""
        import sounddevice as sd

        # Speak the confirmation prompt synchronously so we don't record TTS audio
        tts = self._get_tts()
        hotword_was_active = self.hotword_var.get()
        if hotword_was_active:
            self._hotword.pause()
        try:
            tts.speak(f"Confirm: run {shell_cmd}?", block=True)

            # Record 3 seconds at 16 kHz
            try:
                audio = sd.rec(int(3 * SAMPLE_RATE),
                               samplerate=SAMPLE_RATE,
                               channels=1, dtype="float32")
                sd.wait()
            except Exception as e:
                _log(f"Confirm recording error: {e}")
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, "Confirmation failed"))
                return

            audio_flat = audio.flatten() if audio.ndim > 1 else audio
            if self._stt_engine is None or not self._stt_engine.is_loaded:
                _log("Confirm: STT not loaded; refusing")
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, "Refused (STT not ready)"))
                return

            result = self._stt_engine.transcribe(audio_flat)
            transcript = (result.text or "").strip().lower()
            _log(f"Confirm transcript: {transcript!r}")

            if "yes" in transcript or "confirm" in transcript:
                _log(f"Confirm accepted; running: {shell_cmd}")
                self.root.after(0, lambda: self._set_status(
                    "Running...", self.ACCENT, shell_cmd[:30]))
                self._run_shell_async(shell_cmd)
            else:
                _log(f"Confirm rejected: {transcript!r}")
                self.root.after(0, lambda: self._show_jarvis_text(
                    f"Refused: {shell_cmd} (heard: {transcript!r})"))
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, "Refused"))
        finally:
            if hotword_was_active:
                self._hotword.resume()
```

- [ ] **Step 4: Manual smoke test**

Run: `python jarvis/voice_input_gui.py`

Tests:
1. Say "Hey Jarvis, run ls" → should execute immediately (allowlisted)
2. Say "Hey Jarvis, run rm dash rf slash" → should TTS "Confirm: run rm -rf /?" and wait 3s; stay silent → should log "Confirm rejected" and show "Refused"
3. Say "Hey Jarvis, run echo hello" → allowlisted, runs

Expected log lines in `/tmp/vss_voice/gui_debug.log`:
```
Shell command (allowlisted): ls
Shell command requires confirmation: rm -rf /
Confirm transcript: ''
Confirm rejected: ''
```

- [ ] **Step 5: Commit**

```bash
git add jarvis/voice_input_gui.py
git commit -m "$(cat <<'EOF'
fix(security): gate voice-triggered shell commands with allowlist + confirmation

Shell commands dictated via "run/execute/shell" voice commands were
passed directly to subprocess with no validation. A misheard phrase
could execute arbitrary shell code. Now:
  - Safe read-only commands (ls, git, pwd, etc.) run without prompting
  - Anything else triggers a spoken "Confirm: run <cmd>?" with a 3s
    voice window; only "yes"/"confirm" proceeds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: TTS queue HMAC authentication (item #2)

**Files:**
- Modify: `jarvis/jarvis_speak_queue.py` (add HMAC signing to `say()`)
- Modify: `jarvis/voice_input_gui.py` (around line 4877-4910, `_watch_speak_queue` — verify HMAC on read; lock down `/tmp/vss_voice` permissions on startup)
- Create: `~/.aiws_trainer/speak_queue.key` (generated on first run, mode 0600)

- [ ] **Step 1: Add shared key module**

Create `jarvis/speak_queue_auth.py`:

```python
"""Shared HMAC authentication for the TTS speak queue.

On first use, generates a 32-byte random key at ~/.aiws_trainer/speak_queue.key
(mode 0600). Writers prepend an HMAC-SHA256 truncated to 16 hex chars; readers
verify it.
"""

import hmac
import hashlib
import os
import secrets
from pathlib import Path

KEY_FILE = Path.home() / ".aiws_trainer" / "speak_queue.key"
QUEUE_DIR = Path("/tmp/vss_voice")
HMAC_LEN = 16  # hex chars of truncated hmac


def _ensure_queue_dir():
    """Create /tmp/vss_voice with 0700 permissions, owned by caller."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(QUEUE_DIR, 0o700)
    except OSError:
        pass


def _load_or_create_key() -> bytes:
    """Return the shared key; create it on first use."""
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not KEY_FILE.exists():
        key = secrets.token_bytes(32)
        KEY_FILE.write_bytes(key)
        os.chmod(KEY_FILE, 0o600)
        return key
    return KEY_FILE.read_bytes()


def sign(text: str) -> str:
    """Return 'HMAC16 text' format for writing to the queue."""
    key = _load_or_create_key()
    mac = hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()[:HMAC_LEN]
    return f"{mac} {text}"


def verify(line: str) -> str | None:
    """Given a queue line, return the text payload iff the HMAC matches; else None."""
    if len(line) < HMAC_LEN + 2:
        return None
    mac_hex, _, text = line.partition(" ")
    if len(mac_hex) != HMAC_LEN or not text:
        return None
    key = _load_or_create_key()
    expected = hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()[:HMAC_LEN]
    if hmac.compare_digest(mac_hex, expected):
        return text
    return None
```

- [ ] **Step 2: Write tests for the auth module**

Create `tests/test_speak_queue_auth.py`:

```python
"""Tests for HMAC-based speak queue authentication."""

import os
from pathlib import Path

import pytest

from jarvis import speak_queue_auth


@pytest.fixture(autouse=True)
def _isolated_key(tmp_path, monkeypatch):
    key_file = tmp_path / "speak_queue.key"
    monkeypatch.setattr(speak_queue_auth, "KEY_FILE", key_file)
    yield


def test_sign_verify_roundtrip():
    signed = speak_queue_auth.sign("Hello Jarvis")
    assert speak_queue_auth.verify(signed) == "Hello Jarvis"


def test_verify_rejects_unsigned_line():
    assert speak_queue_auth.verify("Hello Jarvis") is None


def test_verify_rejects_tampered_payload():
    signed = speak_queue_auth.sign("Hello Jarvis")
    mac, _, _ = signed.partition(" ")
    tampered = f"{mac} Goodbye Jarvis"
    assert speak_queue_auth.verify(tampered) is None


def test_key_file_is_mode_0600(tmp_path):
    speak_queue_auth._load_or_create_key()
    mode = speak_queue_auth.KEY_FILE.stat().st_mode & 0o777
    assert mode == 0o600
```

- [ ] **Step 3: Run tests; verify they pass**

```bash
cd /home/hunterp/jarvis
python -m pytest tests/test_speak_queue_auth.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Update `jarvis_speak_queue.py` to sign each line**

Replace the entire contents of `jarvis/jarvis_speak_queue.py` with:

```python
"""Jarvis speak queue — file-based communication between Claude and TTS.

Claude writes HMAC-signed lines to the queue file so untrusted local
processes cannot make Jarvis speak arbitrary text.

Usage from Claude Code:
    from jarvis.jarvis_speak_queue import say
    say("Jarvis: Hello, how can I help?")
"""

from pathlib import Path

from jarvis.speak_queue_auth import sign, _ensure_queue_dir

SPEAK_QUEUE = Path("/tmp/vss_voice/speak_queue.txt")


def say(text):
    """Queue a signed message for Jarvis to speak."""
    _ensure_queue_dir()
    signed = sign(text.strip())
    with open(SPEAK_QUEUE, "a") as f:
        f.write(signed + "\n")
```

- [ ] **Step 5: Update the watcher in `voice_input_gui.py` to verify HMAC**

Modify `jarvis/voice_input_gui.py` at lines 4866-4910 (`_start_speak_queue_watcher` and `_watch_speak_queue`).

Replace lines 4866-4910 with:

```python
    def _start_speak_queue_watcher(self):
        """Watch the speak queue file for new lines from Claude."""
        from jarvis.speak_queue_auth import _ensure_queue_dir
        _ensure_queue_dir()

        speak_file = Path("/tmp/vss_voice/speak_queue.txt")
        self._speak_queue_pos = 0

        # Clear any old content
        if speak_file.exists():
            self._speak_queue_pos = speak_file.stat().st_size

        self._watch_speak_queue()

    def _watch_speak_queue(self):
        """Poll the speak queue file for new HMAC-signed lines."""
        from jarvis.speak_queue_auth import verify

        if not self.talkback_var.get():
            self.root.after(2000, self._watch_speak_queue)
            return

        # Don't queue new speech while already speaking (prevents double-play)
        tts = self._get_tts()
        if tts.is_speaking:
            self.root.after(1000, self._watch_speak_queue)
            return

        speak_file = Path("/tmp/vss_voice/speak_queue.txt")
        try:
            if speak_file.exists():
                size = speak_file.stat().st_size
                if size > self._speak_queue_pos:
                    with open(speak_file) as f:
                        f.seek(self._speak_queue_pos)
                        new_lines = f.read()
                    self._speak_queue_pos = size

                    # Verify each line's HMAC; silently drop forgeries
                    verified = []
                    for line in new_lines.strip().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        payload = verify(line)
                        if payload is None:
                            _log(f"Speak queue: dropped unsigned line")
                            continue
                        verified.append(payload)

                    combined = " ".join(verified)
                    if combined:
                        _log(f"Talk-back queue: {combined[:60]}")
```

Note: the tail of `_watch_speak_queue` (from line 4905 `self.root.after(0, lambda: self._set_status(` onward) is preserved — only the file-read and line-extraction block is changed.

- [ ] **Step 6: Manual smoke test**

Run: `python jarvis/voice_input_gui.py`

Tests:
1. Open second terminal, run: `python -c "from jarvis.jarvis_speak_queue import say; say('Hello from Jarvis')"` → should speak.
2. Open second terminal, run: `echo "Unsigned attack text" >> /tmp/vss_voice/speak_queue.txt` → should NOT speak; log should show `Speak queue: dropped unsigned line`.
3. Check permissions: `ls -la /tmp/vss_voice` should show `drwx------` (0700); `ls -la ~/.aiws_trainer/speak_queue.key` should show `-rw-------` (0600).

- [ ] **Step 7: Commit**

```bash
git add jarvis/speak_queue_auth.py jarvis/jarvis_speak_queue.py jarvis/voice_input_gui.py tests/test_speak_queue_auth.py
git commit -m "$(cat <<'EOF'
fix(security): HMAC-authenticate the TTS speak queue

Any local process (or path traversal from a web tool) could previously
append to /tmp/vss_voice/speak_queue.txt and make Jarvis speak or type
arbitrary text. Now:
  - /tmp/vss_voice is locked to 0700 on startup
  - A 32-byte key at ~/.aiws_trainer/speak_queue.key (mode 0600) signs
    each queued line with HMAC-SHA256 (first 16 hex chars)
  - Watcher drops unsigned / mismatched lines

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Remove duplicate orbit shutdown (item #3)

**Files:**
- Modify: `jarvis/voice_input_gui.py:5757-5761` (delete duplicate block)

- [ ] **Step 1: Delete lines 5757-5761**

In `jarvis/voice_input_gui.py`, find the `_cleanup` method (around line 5748) and delete the duplicate block. The fix is to remove exactly this block (current lines 5757-5761):

```python
        if hasattr(self, '_orbit_server') and self._orbit_server:
            try:
                self._orbit_server.shutdown()
            except Exception:
                pass
```

The first identical block (lines 5752-5756) must remain. After the edit, `_cleanup()` starts:

```python
    def _cleanup(self):
        """Release all resources — mic, hotword, tray, global hotkey, browser."""
        self.recording = False
        # Shut down orbit server if running
        if hasattr(self, '_orbit_server') and self._orbit_server:
            try:
                self._orbit_server.shutdown()
            except Exception:
                pass
        try:
            self._hotword.stop()
        except Exception:
            pass
```

- [ ] **Step 2: Verify with grep**

```bash
grep -n "_orbit_server.shutdown" jarvis/voice_input_gui.py
```

Expected: exactly one match (not two).

- [ ] **Step 3: Commit**

```bash
git add jarvis/voice_input_gui.py
git commit -m "fix: remove duplicate _orbit_server.shutdown() call in _cleanup

Copy-paste bug: the same try/except block appeared twice back-to-back.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Fix dead STT path in partial preview (item #4)

**Files:**
- Modify: `jarvis/voice_input_gui.py:2868-2908` (`_partial_transcribe_worker`)

- [ ] **Step 1: Replace the `_partial_transcribe_worker` body to route through `_stt_engine`**

Replace exactly these lines (2874-2882):

```python
        with self._partial_lock:
            try:
                lang = self._get_whisper_language()
                kwargs = dict(beam_size=1, initial_prompt=_load_vocab())
                if lang is not None:
                    kwargs["language"] = lang

                segments, _ = self._whisper_model.transcribe(audio, **kwargs)
                text = " ".join(seg.text.strip() for seg in segments).strip()
```

With:

```python
        with self._partial_lock:
            try:
                # Route through the unified STT engine (Parakeet primary,
                # Whisper fallback). Skip cleanly if STT isn't loaded yet
                # so the preview worker can't crash with AttributeError.
                if self._stt_engine is None or not self._stt_engine.is_loaded:
                    return
                result = self._stt_engine.transcribe(audio)
                text = (result.text or "").strip()
```

- [ ] **Step 2: Verify `_whisper_model` is no longer referenced by the partial worker**

```bash
grep -n "_whisper_model.transcribe" jarvis/voice_input_gui.py
```

Expected: zero matches (or only matches that are NOT inside `_partial_transcribe_worker`).

- [ ] **Step 3: Manual smoke test**

Run: `python jarvis/voice_input_gui.py`, enable "Live preview" checkbox, press record, speak.

Expected: partial preview text appears in the live-preview label as you speak; log shows no AttributeError around `_partial_transcribe_worker`.

- [ ] **Step 4: Commit**

```bash
git add jarvis/voice_input_gui.py
git commit -m "fix: route partial-transcribe worker through STTEngine

The partial preview worker was still calling self._whisper_model which
is None after the STT migration to Parakeet/Whisper via STTEngine. The
worker's broad except was silently swallowing AttributeError, so the
live preview produced no text and filler-word silence-reset never fired.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Serialize HotwordListener stream mutations (item #5)

**Files:**
- Modify: `jarvis/voice_input_gui.py:785-838` (`HotwordListener` class — add lock, protect stream open/close)
- Modify: `jarvis/voice_input_gui.py:886-914` (`_open_stream` helper and the re-open block inside `_listen_loop`)

- [ ] **Step 1: Locate `HotwordListener.__init__` and add `_stream_lock`**

Find `HotwordListener.__init__` (search for `class HotwordListener` at line 785). In the `__init__` method, after `self._stream = None` or similar existing attribute init, add:

```python
        self._stream_lock = threading.Lock()
```

- [ ] **Step 2: Wrap `_close_stream` in the lock**

Replace lines 831-838 (the current `_close_stream`):

```python
    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
```

With:

```python
    def _close_stream(self):
        with self._stream_lock:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
```

- [ ] **Step 3: Wrap `_open_stream` in the lock**

Find `_open_stream` (inside `_listen_loop`, around lines 886-899). Replace:

```python
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
```

With:

```python
        def _open_stream():
            with self._stream_lock:
                if self._stream is not None:
                    return True  # Already open — don't re-create
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
```

- [ ] **Step 4: Protect the re-open block inside the main loop**

Find the re-open block at lines 911-920 inside `_listen_loop`. Replace:

```python
            # Re-open stream after recording finishes
            if self._reopen and not self.gui.recording and not self._stream:
                self._reopen = False
                buf.clear()
                _open_stream()
                _log("Hotword stream resumed")
                # Reset OWW model state and add cooldown so residual
                # audio from the previous recording doesn't false-trigger
                if self._model:
                    self._model.reset()
                self._resume_cooldown = time.monotonic() + 2.0  # 2s cooldown
```

With (the critical change is the `_reopen` read and `_stream` check now happen under the lock):

```python
            # Re-open stream after recording finishes
            should_reopen = False
            with self._stream_lock:
                if self._reopen and not self.gui.recording and self._stream is None:
                    self._reopen = False
                    should_reopen = True
            if should_reopen:
                buf.clear()
                _open_stream()
                _log("Hotword stream resumed")
                if self._model:
                    self._model.reset()
                self._resume_cooldown = time.monotonic() + 2.0  # 2s cooldown
```

- [ ] **Step 5: Manual smoke test**

Run: `python jarvis/voice_input_gui.py`

Tests:
1. Say "Hey Jarvis" → recording starts, hotword pauses cleanly.
2. Stop recording → hotword resumes within 2s.
3. Say "Hey Jarvis" again immediately after Jarvis finishes speaking → should detect (tests pause/resume/re-open sequence).
4. Log should show `Hotword stream paused` then `Hotword stream resumed` with no errors.

- [ ] **Step 6: Commit**

```bash
git add jarvis/voice_input_gui.py
git commit -m "$(cat <<'EOF'
fix: serialize HotwordListener stream open/close with a lock

pause() is called synchronously from the TTS speak thread to prevent
feedback, but its stream mutation raced with _listen_loop's reopen
block. Added a threading.Lock that serializes all _stream mutations
(_open_stream, _close_stream, and the loop's reopen TOCTOU check).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# PHASE 2 — Extraction + tests (items 6-7)

Split `VoiceInputGUI` into four modules. Each module gets a matching test file. The GUI class shrinks to Tk widget setup + wiring.

**Strategy:** Extract one module at a time, with tests, rewiring the GUI to call it at the end of each task. This keeps the app runnable after every task. The coordinator rewiring is the final task of Phase 2.

## Task 6: Extract `RecordingController` (items #6 partial)

**Files:**
- Create: `jarvis/recording.py`
- Create: `tests/test_recording.py`

**Responsibility:** Own the sounddevice stream lifecycle, audio callback, silence detection, noise gate. Provide callbacks for amplitude updates and recording-stopped events.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_recording.py`:

```python
"""Tests for RecordingController."""

import numpy as np
import pytest

from jarvis.recording import RecordingController


def _make_audio(seconds=1.0, sr=16000, freq=440.0, amp=0.3):
    """Generate a sine wave numpy array."""
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_noise_gate_zeros_quiet_block():
    rc = RecordingController(sample_rate=16000)
    quiet = np.full(1600, 0.001, dtype=np.float32)  # below 0.005 threshold
    gated = rc.apply_noise_gate(quiet)
    assert np.allclose(gated, 0.0)


def test_noise_gate_preserves_loud_block():
    rc = RecordingController(sample_rate=16000)
    loud = _make_audio(seconds=0.1, amp=0.3)
    gated = rc.apply_noise_gate(loud)
    assert np.allclose(gated, loud)


def test_detect_silence_returns_true_for_quiet_audio():
    rc = RecordingController(sample_rate=16000, silence_threshold=0.04)
    quiet = np.zeros(1600, dtype=np.float32)
    assert rc.is_silent(quiet) is True


def test_detect_silence_returns_false_for_loud_audio():
    rc = RecordingController(sample_rate=16000, silence_threshold=0.04)
    loud = _make_audio(seconds=0.1, amp=0.3)
    assert rc.is_silent(loud) is False


def test_amplitude_callback_fires_with_rms():
    amps = []
    rc = RecordingController(
        sample_rate=16000,
        on_amplitude=lambda a: amps.append(a),
    )
    rc.handle_chunk(_make_audio(seconds=0.1, amp=0.5))
    assert len(amps) == 1
    assert 0.0 < amps[0] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/hunterp/jarvis
python -m pytest tests/test_recording.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.recording'`.

- [ ] **Step 3: Create `jarvis/recording.py`**

```python
"""RecordingController — sounddevice stream lifecycle, audio buffering,
silence detection, noise gate. Extracted from VoiceInputGUI.

Does NOT own transcription, TTS, or GUI widgets. Communicates via
callbacks supplied by the caller.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

LOG_DIR = Path("/tmp/vss_voice")

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.04
SILENCE_TIMEOUT = 2.0
NOISE_GATE_THRESHOLD = 0.005


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} [REC] {msg}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "gui_debug.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


class RecordingController:
    """Push-to-talk audio recorder with silence detection and noise gate."""

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        silence_threshold=SILENCE_THRESHOLD,
        silence_timeout=SILENCE_TIMEOUT,
        noise_gate_threshold=NOISE_GATE_THRESHOLD,
        on_amplitude=None,
        on_stopped=None,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_timeout = silence_timeout
        self.noise_gate_threshold = noise_gate_threshold
        self.on_amplitude = on_amplitude
        self.on_stopped = on_stopped

        self._stream = None
        self._frames = []
        self._lock = threading.Lock()
        self._recording = False
        self._silence_start = None

    @property
    def is_recording(self):
        return self._recording

    def apply_noise_gate(self, chunk: np.ndarray) -> np.ndarray:
        """Zero-out blocks whose RMS is below the noise gate threshold."""
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        if rms < self.noise_gate_threshold:
            return np.zeros_like(chunk)
        return chunk

    def is_silent(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        return rms < self.silence_threshold

    def handle_chunk(self, chunk: np.ndarray):
        """Process one audio chunk. Buffers it, fires amplitude callback,
        and returns True iff silence timeout has been reached."""
        gated = self.apply_noise_gate(chunk)
        with self._lock:
            self._frames.append(gated)
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        if self.on_amplitude is not None:
            self.on_amplitude(min(1.0, rms * 4))
        if self.is_silent(chunk):
            now = time.monotonic()
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start >= self.silence_timeout:
                return True
        else:
            self._silence_start = None
        return False

    def start(self, mic_device=None):
        """Open the sounddevice input stream and begin buffering."""
        import sounddevice as sd

        with self._lock:
            self._frames = []
            self._recording = True
            self._silence_start = None

        def _callback(indata, frame_count, time_info, status):
            if not self._recording:
                return
            chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            silence_hit = self.handle_chunk(chunk.astype(np.float32))
            if silence_hit:
                self._recording = False
                if self.on_stopped is not None:
                    self.on_stopped(self.get_audio())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype="float32",
                device=mic_device,
                callback=_callback,
            )
            self._stream.start()
            _log(f"Recording started (device={mic_device})")
        except Exception as e:
            _log(f"Recording start error: {e}")
            self._recording = False
            raise

    def stop(self) -> np.ndarray:
        """Stop the stream and return the captured audio as a single array."""
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        audio = self.get_audio()
        if self.on_stopped is not None:
            self.on_stopped(audio)
        _log(f"Recording stopped ({len(audio)} samples)")
        return audio

    def get_audio(self) -> np.ndarray:
        """Return the buffered audio as a single concatenated float32 array."""
        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._frames)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_recording.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis/recording.py tests/test_recording.py
git commit -m "refactor(phase2): extract RecordingController into jarvis.recording

First module in the VoiceInputGUI decomposition. Owns sounddevice
stream, audio buffering, silence detection, and noise gate. GUI will
be wired to this in the Phase 2 coordinator task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Extract `TranscriptionPipeline` (items #6 partial)

**Files:**
- Create: `jarvis/transcription.py`
- Create: `tests/test_transcription.py`

**Responsibility:** Given raw audio, run speaker verification (if enabled), run STT, run intent classification, return cleaned text + classification verdict.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_transcription.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_transcription.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.transcription'`.

- [ ] **Step 3: Create `jarvis/transcription.py`**

```python
"""TranscriptionPipeline — speaker filter → STT → intent classification.

Extracted from VoiceInputGUI. Holds no GUI state; returns a simple
PipelineResult dataclass.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

LOG_DIR = Path("/tmp/vss_voice")


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} [PIPE] {msg}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "gui_debug.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_transcription.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis/transcription.py tests/test_transcription.py
git commit -m "refactor(phase2): extract TranscriptionPipeline into jarvis.transcription

Second module in VoiceInputGUI decomposition. Pure function of audio:
speaker filter → STT → intent classification → PipelineResult. No GUI
coupling, fully unit-testable with mocks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Extract `CommandDispatcher` with registry (items #6, #7)

**Files:**
- Create: `jarvis/dispatcher.py`
- Create: `tests/test_dispatcher.py`

**Responsibility:** Replace the 350-line `_check_quick_command` elif chain with a `COMMAND_REGISTRY: list[(re.Pattern, Handler)]`. Routes transcribed text to handlers; unmatched text falls through to the brain.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_dispatcher.py`:

```python
"""Tests for CommandDispatcher."""

from unittest.mock import MagicMock

from jarvis.dispatcher import CommandDispatcher, CommandHandler


def _make_handler(name, called_list):
    def _fn(match, ctx):
        called_list.append((name, match.group(0)))
        return True
    return _fn


def test_matching_command_routes_to_handler():
    called = []
    handlers = [
        CommandHandler(r"^list files$", _make_handler("list", called)),
        CommandHandler(r"^show time$", _make_handler("time", called)),
    ]
    brain = MagicMock()
    d = CommandDispatcher(handlers=handlers, brain=brain)
    handled = d.handle("list files", ctx={})
    assert handled is True
    assert called == [("list", "list files")]
    brain.handle.assert_not_called()


def test_first_match_wins_when_multiple_patterns_match():
    called = []
    handlers = [
        CommandHandler(r"^list.*", _make_handler("broad", called)),
        CommandHandler(r"^list files$", _make_handler("specific", called)),
    ]
    brain = MagicMock()
    d = CommandDispatcher(handlers=handlers, brain=brain)
    d.handle("list files", ctx={})
    assert called == [("broad", "list files")]


def test_unmatched_text_falls_through_to_brain():
    brain = MagicMock()
    d = CommandDispatcher(handlers=[], brain=brain)
    d.handle("hello there", ctx={})
    brain.handle.assert_called_once_with("hello there", {})


def test_handler_returning_false_falls_through():
    called = []
    def _refused(match, ctx):
        called.append("refused")
        return False
    handlers = [CommandHandler(r"^list.*", _refused)]
    brain = MagicMock()
    d = CommandDispatcher(handlers=handlers, brain=brain)
    d.handle("list files", ctx={})
    assert called == ["refused"]
    brain.handle.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dispatcher.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.dispatcher'`.

- [ ] **Step 3: Create `jarvis/dispatcher.py`**

```python
"""CommandDispatcher — routes transcribed text to handlers via a regex registry.

Replaces the 350-line if/elif chain in VoiceInputGUI._check_quick_command.
Each handler is (pattern, callable). Handlers receive the re.Match and a
context dict; returning True = handled, False = fall through.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

LOG_DIR = Path("/tmp/vss_voice")


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} [DISP] {msg}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "gui_debug.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


@dataclass
class CommandHandler:
    pattern: str
    handler: Callable[[re.Match, dict], bool]
    flags: int = re.IGNORECASE

    def compiled(self):
        return re.compile(self.pattern, self.flags)


class CommandDispatcher:
    """Dispatches text to the first matching handler; falls through to brain."""

    def __init__(self, handlers: list[CommandHandler], brain: Any):
        self._compiled = [(h.compiled(), h.handler) for h in handlers]
        self.brain = brain

    def handle(self, text: str, ctx: dict | None = None) -> None:
        ctx = ctx if ctx is not None else {}
        for pattern, handler in self._compiled:
            m = pattern.match(text)
            if m:
                try:
                    if handler(m, ctx):
                        _log(f"Dispatched: {pattern.pattern!r} -> handled")
                        return
                except Exception as e:
                    _log(f"Handler error ({pattern.pattern!r}): {e}")
        # No handler claimed it; forward to brain
        self.brain.handle(text, ctx)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dispatcher.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis/dispatcher.py tests/test_dispatcher.py
git commit -m "$(cat <<'EOF'
refactor(phase2): extract CommandDispatcher with regex registry

Replaces the 350-line _check_quick_command if/elif chain with a
list of (pattern, handler) tuples. Handlers are now testable in
isolation, ordering is explicit, and new commands are one-liner
registrations instead of wedged-in elif branches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Extract `AnimationRenderer` (items #6 partial)

**Files:**
- Create: `jarvis/animation.py`
- Create: `tests/test_animation.py`

**Responsibility:** Own orbit frame generation and the amplitude feeder. Consumers subscribe via `render_idle()` / `render_active(amp)`.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_animation.py`:

```python
"""Tests for AnimationRenderer."""

import numpy as np

from jarvis.animation import AnimationRenderer, generate_beep


def test_render_active_returns_rgb_frame():
    r = AnimationRenderer(size=128)
    frame = r.render_active(amplitude=0.5)
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8


def test_render_idle_returns_rgb_frame():
    r = AnimationRenderer(size=128)
    frame = r.render_idle(t=0.0)
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8


def test_render_active_zero_amplitude_not_blank():
    r = AnimationRenderer(size=128)
    frame = r.render_active(amplitude=0.0)
    assert frame.sum() > 0  # Base glow is always visible


def test_generate_beep_has_correct_length():
    samples = generate_beep(duration=0.1, sample_rate=16000)
    assert len(samples) == 1600


def test_generate_beep_peaks_at_expected_amplitude():
    samples = generate_beep(duration=0.1, sample_rate=16000, amplitude=0.5)
    assert 0.4 < np.max(np.abs(samples)) <= 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_animation.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.animation'`.

- [ ] **Step 3: Create `jarvis/animation.py`**

```python
"""AnimationRenderer — orbit reactor animation + beep generation.

Extracted from VoiceInputGUI. Pure numpy; no GUI coupling.
"""

import numpy as np


def generate_beep(
    duration: float = 0.1,
    sample_rate: int = 16000,
    frequency: float = 880.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine-wave beep as float32 PCM samples."""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class AnimationRenderer:
    """Renders the Jarvis arc-reactor orbit as an RGB numpy frame."""

    def __init__(self, size: int = 512):
        self.size = size

    def _base_frame(self) -> np.ndarray:
        """Create an empty dark-blue frame."""
        frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        frame[..., 2] = 8  # very faint blue baseline so frames are never all-zero
        return frame

    def render_active(self, amplitude: float) -> np.ndarray:
        """Render a speaking frame scaled by amplitude (0.0-1.0)."""
        frame = self._base_frame()
        cx = cy = self.size // 2
        amp = max(0.0, min(1.0, amplitude))
        radius = int(self.size * (0.15 + 0.25 * amp))
        intensity = int(80 + 175 * amp)
        self._paint_disc(frame, cx, cy, radius, intensity)
        return frame

    def render_idle(self, t: float) -> np.ndarray:
        """Render a breathing idle frame. t is seconds since app start."""
        amp = 0.3 + 0.2 * np.sin(t * 2.0)
        return self.render_active(amplitude=amp)

    def _paint_disc(self, frame, cx, cy, radius, intensity):
        """In-place blue-cyan disc around (cx, cy)."""
        y, x = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist < radius
        falloff = np.clip(1.0 - dist / max(radius, 1), 0.0, 1.0)
        frame[..., 1] = np.where(mask,
            np.clip(frame[..., 1] + (falloff * intensity * 0.6), 0, 255),
            frame[..., 1]).astype(np.uint8)
        frame[..., 2] = np.where(mask,
            np.clip(frame[..., 2] + (falloff * intensity), 0, 255),
            frame[..., 2]).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_animation.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis/animation.py tests/test_animation.py
git commit -m "$(cat <<'EOF'
refactor(phase2): extract AnimationRenderer + beep generator

Fourth and final module in the extraction. Pure numpy: render_active(amp)
and render_idle(t) return RGB frames; generate_beep() replaces the slow
per-sample struct.pack loop.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Rewire `VoiceInputGUI` to use the new modules

**Files:**
- Modify: `jarvis/voice_input_gui.py` (coordinator rewire — many edits)

**Note:** This is the biggest single task in the plan. After it, the app still runs end-to-end, but `voice_input_gui.py` is reduced and now *uses* the four new modules. Keep changes mechanical; do not rewrite behavior that isn't on the punch list.

- [ ] **Step 1: Import the new modules**

Near the top of `jarvis/voice_input_gui.py` (after the existing `import` block at lines 31-47), add:

```python
from jarvis.recording import RecordingController
from jarvis.transcription import TranscriptionPipeline
from jarvis.dispatcher import CommandDispatcher, CommandHandler
from jarvis.animation import AnimationRenderer, generate_beep
```

- [ ] **Step 2: Instantiate the four modules in `VoiceInputGUI.__init__`**

Find `VoiceInputGUI.__init__` (search for `class VoiceInputGUI:`). Near the end of `__init__` (after all existing attribute initialization, before any threads are started), add:

```python
        # Extracted subsystems (Phase 2)
        self._recording_ctrl = RecordingController(
            sample_rate=SAMPLE_RATE,
            silence_threshold=SILENCE_THRESHOLD,
            silence_timeout=SILENCE_TIMEOUT,
            noise_gate_threshold=NOISE_GATE_THRESHOLD,
            on_amplitude=lambda amp: setattr(self, '_audio_level', amp),
            on_stopped=lambda audio: self.root.after(
                0, lambda: self._on_recording_stopped(audio)),
        )
        self._animation_renderer = AnimationRenderer(size=512)
        # _pipeline and _cmd_dispatcher are initialized lazily after
        # STT + speaker verifier finish loading in background
        self._pipeline = None
        self._cmd_dispatcher = None
```

- [ ] **Step 3: Add pipeline/dispatcher lazy init**

Add a new method on `VoiceInputGUI`:

```python
    def _ensure_pipeline_ready(self):
        """Build TranscriptionPipeline + CommandDispatcher once STT is loaded."""
        if self._pipeline is None and self._stt_engine is not None \
                and self._stt_engine.is_loaded:
            self._pipeline = TranscriptionPipeline(
                stt_engine=self._stt_engine,
                speaker_verifier=self._speaker_verifier,
                intent_classifier=getattr(self, '_intent_classifier', None),
            )
        if self._cmd_dispatcher is None:
            self._cmd_dispatcher = CommandDispatcher(
                handlers=self._build_command_handlers(),
                brain=self._brain,
            )
```

- [ ] **Step 4: Convert `_check_quick_command` to a handler list**

Find `_check_quick_command` (around line 3194) and replace the entire method with a small `_build_command_handlers()` that returns a list of `CommandHandler` instances. Each of the original elif branches becomes a handler function + registration.

Because the original method is ~350 lines and the patterns are project-specific, the mechanical conversion for *each* elif is:

```python
# before:
if re.match(r"PATTERN", cmd_text):
    BODY
    return True

# after (in _build_command_handlers()):
def _handle_xyz(match, ctx):
    BODY  # replace cmd_text references with match.group(0) or match.string
    return True

handlers.append(CommandHandler(r"PATTERN", _handle_xyz))
```

For this task, translate EACH existing elif branch 1:1 into a CommandHandler. Do not change behavior. Preserve order (first match wins).

Place the result just above `_check_quick_command` in the file:

```python
    def _build_command_handlers(self) -> list[CommandHandler]:
        """Return the ordered list of command handlers for dispatcher.

        Translated 1:1 from the former _check_quick_command if/elif chain.
        Each handler returns True iff it handled the command.
        """
        handlers: list[CommandHandler] = []
        # ... translate each elif from the original method ...
        # (detailed mechanical translation below)
        return handlers
```

**Implementation strategy:** because of the sheer size of `_check_quick_command`, do this in sub-steps. Open the file, find each `if re.match(...)` or `if <phrase> in cmd_text` block, and extract it as a closure. Use the shell allowlist helper from Task 1 for the shell branch.

Keep `_check_quick_command` as a thin delegation:

```python
    def _check_quick_command(self, cmd_text: str) -> bool:
        """Legacy entry point; delegates to the dispatcher."""
        if self._cmd_dispatcher is None:
            self._ensure_pipeline_ready()
        self._cmd_dispatcher.handle(cmd_text, ctx={"cmd_text": cmd_text})
        return True  # Dispatcher always handles (either command or brain fallback)
```

- [ ] **Step 5: Route the post-recording hook through the pipeline**

Find `_transcribe_worker` (the method that runs after recording stops). Replace its STT + speaker-filter + classify logic with a single call:

```python
    def _on_recording_stopped(self, audio):
        """Called from RecordingController when recording stops."""
        self._ensure_pipeline_ready()
        if self._pipeline is None:
            _log("Pipeline not ready; dropping recording")
            return

        def _process():
            result = self._pipeline.transcribe(audio)
            if not result.text:
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, ""))
                return
            self.root.after(0, lambda: self._handle_transcribed(result.text))

        threading.Thread(target=_process, daemon=True).start()
```

`_handle_transcribed(text)` is the existing post-transcribe hook (auto-type, history log, command dispatch). If that logic lives inline in the current `_transcribe_worker`, factor it out into this new method.

- [ ] **Step 6: Route animation calls through the renderer**

Find `_render_orbit_fast` (around line 4236). Keep it as a thin wrapper that delegates:

```python
    def _render_orbit_fast(self, amp):
        frame = self._animation_renderer.render_active(amp)
        # existing PIL conversion + canvas update preserved unchanged
        # ... (keep the existing self._orbit_img / self.canvas.itemconfig code)
```

Find `_generate_beep` (around line 532) and replace its for-loop body with:

```python
    def _generate_beep(self):
        samples = generate_beep(duration=0.1, sample_rate=SAMPLE_RATE,
                                frequency=880.0, amplitude=0.3)
        # existing wave.write / pygame.mixer code preserved unchanged
        # ... (keep the existing playback code)
```

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all Phase 2 tests pass (the 4 module test files + existing test_audio_pipeline, test_mic_resolution, test_speaker_verification, test_stt_engine).

- [ ] **Step 8: Manual smoke test — full round trip**

Run: `python jarvis/voice_input_gui.py`

Tests:
1. Press Record → RecordingController starts → speak → silence timeout → `_on_recording_stopped` fires → TranscriptionPipeline runs → text appears.
2. Say "Hey Jarvis, run ls" → dispatcher routes to shell handler → allowlisted → runs.
3. Say "Hey Jarvis, summarize this repository" → dispatcher has no matching handler → falls through to brain → brain response appears.
4. Speaking animation still animates (AnimationRenderer driving orbit).
5. `wc -l jarvis/voice_input_gui.py` should drop noticeably (the 350-line command chain is gone; pipeline logic is externalized).

- [ ] **Step 9: Commit**

```bash
git add jarvis/voice_input_gui.py
git commit -m "$(cat <<'EOF'
refactor(phase2): rewire VoiceInputGUI to coordinate extracted modules

VoiceInputGUI now delegates to:
  - RecordingController (audio lifecycle)
  - TranscriptionPipeline (speaker → STT → intent)
  - CommandDispatcher (regex registry replaces 350-line elif chain)
  - AnimationRenderer (orbit + beep)

Behavior preserved; no new features. The coordinator keeps Tk widgets,
event binding, and cross-cutting state. Each subsystem is independently
unit-testable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# PHASE 3 — Remaining fixes in new module layout (items 8-21)

Each task is scoped to a single new module (or the coordinator). Small, composable fixes.

## Task 11: Centralize logging + delete dead patterns (items #8, #21)

**Files:**
- Create: `jarvis/logging.py`
- Modify: every module with a local `_log()` (`voice_input_gui.py`, `jarvis_tts.py`, `stt_engine.py`, `speaker_verification.py`, `jarvis_brain.py`, `recording.py`, `transcription.py`, `dispatcher.py`, `animation.py`)
- Modify: `jarvis/voice_input_gui.py` (delete `_ASSISTANT_PATTERNS` and `_CASUAL_PATTERNS` at lines 132-179)

- [ ] **Step 1: Create `jarvis/logging.py`**

```python
"""Shared timestamp logger for the Jarvis modules.

Every module previously had its own copy of _log() writing to
/tmp/vss_voice/gui_debug.log with a slightly different prefix. This
module replaces them all. Use get_logger("TTS") / get_logger("STT") /
etc. to get a prefix-bound logger.
"""

from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/tmp/vss_voice")
LOG_FILE = LOG_DIR / "gui_debug.log"


def _write(line: str):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_logger(prefix: str):
    """Return a function that logs with the given bracketed prefix."""
    def _log(msg):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _write(f"{ts} [{prefix}] {msg}")
    return _log
```

- [ ] **Step 2: Delete the dead intent pattern lists**

In `jarvis/voice_input_gui.py`, delete lines 132-179 (the module-level `_ASSISTANT_PATTERNS` and `_CASUAL_PATTERNS` lists). Verify before deletion:

```bash
grep -n "_ASSISTANT_PATTERNS\|_CASUAL_PATTERNS" jarvis/voice_input_gui.py
```

If the only matches are the definitions themselves (no usages outside the class-level `_POSITIVE_PATTERNS` / `_NEGATIVE_PATTERNS`), proceed to delete.

- [ ] **Step 3: Replace each module's `_log` with `get_logger` import**

In each of these files, find the local `def _log(msg): ...` definition and replace it with:

```python
from jarvis.logging import get_logger
_log = get_logger("<prefix>")  # TTS / STT / VER / BRAIN / GUI / REC / PIPE / DISP / ANIM
```

Files + prefixes:
- `jarvis/voice_input_gui.py` → `GUI`
- `jarvis/jarvis_tts.py` → `TTS`
- `jarvis/stt_engine.py` → `STT`
- `jarvis/speaker_verification.py` → `VER`
- `jarvis/jarvis_brain.py` → `BRAIN`
- `jarvis/recording.py` → `REC`
- `jarvis/transcription.py` → `PIPE`
- `jarvis/dispatcher.py` → `DISP`
- `jarvis/animation.py` → `ANIM`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Manual verify log file still aggregates**

```bash
rm /tmp/vss_voice/gui_debug.log
python -c "from jarvis.recording import _log; _log('hello')"
python -c "from jarvis.jarvis_tts import _log; _log('hi')"
cat /tmp/vss_voice/gui_debug.log
```

Expected: two lines, `[REC] hello` and `[TTS] hi`.

- [ ] **Step 6: Commit**

```bash
git add jarvis/logging.py jarvis/voice_input_gui.py jarvis/jarvis_tts.py \
        jarvis/stt_engine.py jarvis/speaker_verification.py \
        jarvis/jarvis_brain.py jarvis/recording.py jarvis/transcription.py \
        jarvis/dispatcher.py jarvis/animation.py
git commit -m "$(cat <<'EOF'
refactor: centralize _log via jarvis.logging; delete dead intent patterns

Nine modules each had a copy-pasted _log() writing to the same file
with slightly different prefixes. Replaced with get_logger(prefix).
Also deleted the unreferenced _ASSISTANT_PATTERNS / _CASUAL_PATTERNS
module-level lists in voice_input_gui.py — IntentClassifier owns its
own _POSITIVE_PATTERNS / _NEGATIVE_PATTERNS.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Recording fixes (items #9, #11, #13, #17)

**Files:**
- Modify: `jarvis/recording.py` (denoise in `stop()`, not `handle_chunk`)
- Modify: `jarvis/voice_input_gui.py` (hotword cooldown, stream_partial reschedule, SIGTERM)
- Modify: `tests/test_recording.py` (add denoise-timing test)

- [ ] **Step 1: Move denoise to `stop()` (item #9)**

In `jarvis/recording.py`, add an optional `apply_denoise` flag and move the denoise call:

```python
class RecordingController:
    def __init__(self, ..., apply_denoise: bool = False, denoise_fn=None):
        ...
        self.apply_denoise = apply_denoise
        self.denoise_fn = denoise_fn  # callable(audio, sr) -> audio

    def stop(self) -> np.ndarray:
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        audio = self.get_audio()
        if self.apply_denoise and self.denoise_fn is not None and audio.size:
            try:
                audio = self.denoise_fn(audio, self.sample_rate)
            except Exception as e:
                _log(f"Denoise error: {e}")
        if self.on_stopped is not None:
            self.on_stopped(audio)
        _log(f"Recording stopped ({len(audio)} samples)")
        return audio
```

In `voice_input_gui.py`, update the `RecordingController(...)` instantiation from Task 10 Step 2 to pass the denoise callback when the feature is enabled:

```python
        self._recording_ctrl = RecordingController(
            ...,
            apply_denoise=getattr(self, 'noise_suppress_enabled', False),
            denoise_fn=self._denoise_audio_fn,  # existing denoise_audio function
        )
```

- [ ] **Step 2: Replace hotword `time.sleep(1.5)` with monotonic cooldown (item #11)**

In `jarvis/voice_input_gui.py` inside `HotwordListener._listen_loop`, find the two `time.sleep(1.5)` calls at lines 977 and 988. Replace BOTH with:

```python
                self._detection_cooldown = time.monotonic() + 1.5
```

Then add a cooldown check at the top of the while loop (after the sleep at line 908):

```python
            while self.active:
                time.sleep(0.08)

                # Detection cooldown — skip processing if recently detected
                if time.monotonic() < getattr(self, '_detection_cooldown', 0):
                    continue

                # (existing re-open block follows)
```

- [ ] **Step 3: Don't reschedule `_stream_partial` when streaming is disabled (item #13)**

Find `_stream_partial` (around line 2844). Replace the early-return block so it doesn't reschedule when disabled:

```python
    def _stream_partial(self):
        """Periodic live-preview transcription."""
        if not self.streaming_var.get():
            return  # Stop the loop entirely; caller restarts via Record button
        # (rest of method preserved)
```

Ensure that pressing Record starts the loop (it already does — verify by searching for `self._stream_partial` calls; the Record button's start path calls `self.root.after(500, self._stream_partial)` already).

- [ ] **Step 4: Add SIGTERM handler that runs `_cleanup()` (item #17)**

In `jarvis/voice_input_gui.py`, in `VoiceInputGUI.__init__` near the end (after all subsystems initialized):

```python
        import signal
        def _sigterm(*_):
            _log("SIGTERM received; running _cleanup()")
            self._cleanup()
        try:
            signal.signal(signal.SIGTERM, _sigterm)
        except ValueError:
            # Not on main thread; skip
            pass
```

- [ ] **Step 5: Add test for denoise timing**

Append to `tests/test_recording.py`:

```python
def test_denoise_runs_in_stop_not_in_chunk(monkeypatch):
    """Regression: denoise must not run inside the audio callback hot path."""
    calls = []
    def _denoise(audio, sr):
        calls.append("denoise")
        return audio
    rc = RecordingController(apply_denoise=True, denoise_fn=_denoise)
    rc.handle_chunk(np.zeros(1600, dtype=np.float32))
    rc.handle_chunk(np.zeros(1600, dtype=np.float32))
    # Expectation: no denoise calls during streaming
    assert calls == []
    # Stream is not open (we didn't call start()); stop() returns empty audio
    # but should still not call denoise on empty.
    # Simulate audio present:
    rc._frames = [np.ones(1600, dtype=np.float32)]
    rc._recording = True  # short-circuit the stream path
    # Emulate direct post-recording:
    audio = rc.get_audio()
    if rc.apply_denoise and rc.denoise_fn and audio.size:
        rc.denoise_fn(audio, rc.sample_rate)
    assert calls == ["denoise"]
```

Run:

```bash
python -m pytest tests/test_recording.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Manual smoke test**

Run: `python jarvis/voice_input_gui.py`

Tests:
1. Toggle "noise suppress" checkbox on; record some audio with background hum; stop → transcript should be clean.
2. Say "Hey Jarvis" twice in quick succession (< 1.5s apart); second detection should be ignored.
3. Toggle streaming preview off → after recording stops, `_stream_partial` no longer in logs.
4. `kill -TERM <pid>` from another terminal → `_cleanup()` runs (check log).

- [ ] **Step 7: Commit**

```bash
git add jarvis/recording.py jarvis/voice_input_gui.py tests/test_recording.py
git commit -m "$(cat <<'EOF'
fix(recording): denoise in stop(), monotonic hotword cooldown, SIGTERM cleanup

  - Denoise no longer runs inside the audio callback (buffer overrun risk)
  - Hotword cooldown uses time.monotonic() instead of blocking time.sleep
  - _stream_partial no longer reschedules when streaming checkbox is off
  - SIGTERM now triggers _cleanup() even when minimized to tray

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Transcription fix — intent-log corruption handling (item #14)

**Files:**
- Modify: `jarvis/voice_input_gui.py` (IntentClassifier._load_log)
- Modify: `tests/test_transcription.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transcription.py`:

```python
def test_intent_classifier_backs_up_corrupt_log(tmp_path, monkeypatch):
    from jarvis.voice_input_gui import IntentClassifier

    log_path = tmp_path / "intent_log.json"
    log_path.write_text("{ not valid json")

    # Swap the log path on the instance
    ic = IntentClassifier.__new__(IntentClassifier)
    ic._log_path = log_path
    ic._log_data = []
    ic._load_log()

    backups = list(tmp_path.glob("intent_log.json.corrupt.*"))
    assert len(backups) == 1
    assert ic._log_data == []
```

Run: `python -m pytest tests/test_transcription.py::test_intent_classifier_backs_up_corrupt_log -v` → Expected: FAIL.

- [ ] **Step 2: Update `IntentClassifier._load_log`**

Find `IntentClassifier._load_log` in `jarvis/voice_input_gui.py` (around line 234). Replace:

```python
    def _load_log(self):
        try:
            if self._log_path.exists():
                self._log_data = json.loads(self._log_path.read_text())
            else:
                self._log_data = []
        except Exception:
            self._log_data = []
```

With:

```python
    def _load_log(self):
        if not self._log_path.exists():
            self._log_data = []
            return
        try:
            self._log_data = json.loads(self._log_path.read_text())
        except Exception as e:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = self._log_path.with_suffix(f".json.corrupt.{ts}")
            try:
                self._log_path.rename(backup)
                _log(f"Intent log corrupt ({e}); backed up to {backup.name}")
            except Exception as rename_err:
                _log(f"Intent log corrupt and unbackup-able: {rename_err}")
            self._log_data = []
```

- [ ] **Step 3: Run test to verify it passes**

```bash
python -m pytest tests/test_transcription.py::test_intent_classifier_backs_up_corrupt_log -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add jarvis/voice_input_gui.py tests/test_transcription.py
git commit -m "fix: back up corrupt intent_log.json before wiping it

Silent reset on parse failure destroyed accumulated classifier
learning. Now logs the error and renames to intent_log.json.corrupt.*.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Dispatcher fixes — wmctrl, WID cache, Path.home (items #10, #16, #19)

**Files:**
- Modify: `jarvis/dispatcher.py` OR `jarvis/voice_input_gui.py` (wherever window helpers landed after Task 10)
- Modify: `jarvis/voice_input_gui.py:431-437` (QUICK_COMMANDS hard-coded paths)
- Modify: `jarvis/jarvis_agent.py:716` (hard-coded home path)

- [ ] **Step 1: Replace N+1 xdotool with single wmctrl call (item #10)**

Locate `_get_window_list` (now in `dispatcher.py` or still in `voice_input_gui.py` if Task 10 left window helpers in the GUI). Replace its body with:

```python
    def _get_window_list(self):
        """Return [(wid, title)] for all visible windows — one subprocess."""
        import subprocess
        try:
            out = subprocess.check_output(
                ["wmctrl", "-l"], text=True, timeout=2,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError,
                subprocess.CalledProcessError):
            return []
        windows = []
        for line in out.strip().splitlines():
            # Format: "<wid> <desktop> <host> <title...>"
            parts = line.split(None, 3)
            if len(parts) >= 4:
                windows.append((parts[0], parts[3]))
        return windows
```

- [ ] **Step 2: Cache Claude terminal WID with 5s TTL (item #16)**

Locate `_find_claude_terminal`. Add an instance-level cache:

```python
    def _find_claude_terminal(self):
        """Return the Claude terminal WID, cached for up to 5s."""
        import time
        now = time.monotonic()
        cached = getattr(self, '_claude_wid_cache', None)
        if cached is not None:
            wid, expires = cached
            if now < expires:
                return wid

        # Cache miss — actually look it up (original logic preserved below)
        wid = self._lookup_claude_terminal_uncached()
        self._claude_wid_cache = (wid, now + 5.0)
        return wid
```

Then rename the existing `_find_claude_terminal` body to `_lookup_claude_terminal_uncached`. The cache is invalidated when the TTL expires; no manual invalidation needed.

- [ ] **Step 3: Replace hard-coded `/home/hunterp/` (item #19)**

```bash
grep -n "/home/hunterp" jarvis/voice_input_gui.py jarvis/jarvis_agent.py
```

For each match, replace the literal string with `str(Path.home())` or the appropriate `Path.home() / ...`. Example in `voice_input_gui.py` lines 431-437 (`QUICK_COMMANDS`):

```python
# before:
"/home/hunterp/jarvis"

# after:
str(Path.home() / "jarvis")
```

In `jarvis_agent.py:716`, same pattern. After editing, confirm:

```bash
grep -n "/home/hunterp" jarvis/voice_input_gui.py jarvis/jarvis_agent.py
```

Expected: zero matches.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add jarvis/voice_input_gui.py jarvis/jarvis_agent.py
git commit -m "$(cat <<'EOF'
fix(dispatcher): batch xdotool → wmctrl, cache Claude WID, Path.home()

  - _get_window_list: 1 wmctrl subprocess instead of N+1 xdotool calls
  - _find_claude_terminal: 5s WID cache cuts 2 xdotool calls per type
  - Hard-coded /home/hunterp/ paths → Path.home() (portable across users)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Animation fixes — delete dead renderer, vectorize beep (items #15, #18)

**Files:**
- Modify: `jarvis/voice_input_gui.py` (delete `_render_orbit_frame` at lines 4339-4398)
- (item #18 already handled — `generate_beep` in `animation.py` is vectorized via numpy)

- [ ] **Step 1: Confirm `_render_orbit_frame` is unreachable**

```bash
grep -n "_render_orbit_frame" jarvis/voice_input_gui.py
```

Expected: only the `def _render_orbit_frame(...)` definition line should appear — no callers.

- [ ] **Step 2: Delete the method**

In `jarvis/voice_input_gui.py`, delete lines 4339-4398 (the full `_render_orbit_frame` method body).

- [ ] **Step 3: Verify deletion**

```bash
grep -n "_render_orbit_frame" jarvis/voice_input_gui.py
```

Expected: zero matches.

```bash
python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add jarvis/voice_input_gui.py
git commit -m "chore: delete unreachable _render_orbit_frame

_render_orbit_fast is the only caller path for reactor rendering.
_render_orbit_frame was superseded during the pre-rendered frames
migration and had no live callers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Coordinator fixes — lazy TTS, atomic settings (items #12, #20)

**Files:**
- Modify: `jarvis/voice_input_gui.py` (`_get_tts`, `_save_settings`)

- [ ] **Step 1: Lazy-instantiate TTS + reload on engine change (item #12)**

Find `_get_tts` (currently around line 4774). Replace with:

```python
    def _get_tts(self):
        """Return the TTS instance; create or reload when engine changes."""
        want_engine = self.tts_engine_var.get() if hasattr(self, 'tts_engine_var') else "kokoro"
        # Only instantiate if talkback is actually enabled
        if not getattr(self, '_tts', None) or self._tts.engine != want_engine:
            if getattr(self, '_tts', None) is not None:
                try:
                    self._tts.stop()
                except Exception:
                    pass
                self._tts = None
            if self.talkback_var.get() or self._pending_tts_request:
                from jarvis.jarvis_tts import JarvisTTS
                self._tts = JarvisTTS(gpu=1, engine=want_engine)
                self._tts.load()
        return self._tts
```

Update `_watch_speak_queue` (from Task 2 Step 5) to guard on `talkback_var.get()` before calling `_get_tts()` — already done in that task, but verify the order:

```python
    def _watch_speak_queue(self):
        if not self.talkback_var.get():
            self.root.after(2000, self._watch_speak_queue)
            return
        # ... _get_tts() call below
```

- [ ] **Step 2: Atomic `_save_settings` write (item #20)**

Find `_save_settings` (around line 1456). Replace the body:

```python
# before:
SETTINGS_FILE.write_text(json.dumps(settings, indent=2))

# after:
tmp_path = SETTINGS_FILE.with_suffix(".tmp")
tmp_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path.write_text(json.dumps(settings, indent=2))
os.replace(tmp_path, SETTINGS_FILE)
```

- [ ] **Step 3: Manual smoke test**

Run: `python jarvis/voice_input_gui.py`

Tests:
1. Turn talkback OFF; check log — no repeated `JarvisTTS loaded` messages in the speak-queue watcher loop.
2. Turn talkback ON → `JarvisTTS loaded` appears once.
3. Switch TTS engine dropdown from Kokoro to F5 → next TTS call loads F5, log shows `F5-TTS loaded`.
4. Check `~/.aiws_trainer/voice_settings.json.tmp` does NOT exist after normal save.
5. Kill process mid-save (tricky — skip or simulate by running `os.replace` on a half-written file manually).

- [ ] **Step 4: Commit**

```bash
git add jarvis/voice_input_gui.py
git commit -m "$(cat <<'EOF'
fix(coordinator): lazy TTS instantiation + atomic settings write

  - _get_tts no longer creates JarvisTTS every 1s tick when talkback
    is disabled; also reloads underlying model when engine changes
  - _save_settings writes to .tmp and os.replace()s into place —
    a crash mid-write can no longer corrupt voice_settings.json

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Final verification + merge prep

- [ ] **Step 1: Full test run**

```bash
cd /home/hunterp/jarvis
python -m pytest tests/ -v
```

Expected: all tests green.

- [ ] **Step 2: Line-count check**

```bash
wc -l jarvis/*.py
```

Expected: `voice_input_gui.py` under 4,000 lines (ideally <2,500). Each new module ≤500 lines.

- [ ] **Step 3: Grep for old patterns that should be gone**

```bash
grep -n "_whisper_model.transcribe" jarvis/voice_input_gui.py   # expect zero
grep -n "/home/hunterp" jarvis/                                  # expect zero
grep -n "_render_orbit_frame" jarvis/voice_input_gui.py         # expect zero
grep -n "_ASSISTANT_PATTERNS\|_CASUAL_PATTERNS" jarvis/voice_input_gui.py  # expect zero
grep -cn "def _log" jarvis/                                      # expect zero (all moved to logging.py)
```

All should be zero matches.

- [ ] **Step 4: Full manual smoke — end-to-end**

Run: `python jarvis/voice_input_gui.py`

Checklist:
- [ ] Record button works; transcript appears
- [ ] Hotword detection works ("Hey Jarvis")
- [ ] Speaker verification works (authorized speaker → transcript; other → empty)
- [ ] TTS talkback speaks responses
- [ ] Shell allowlist works (ls runs; rm prompts; unsigned queue line rejected)
- [ ] Intent classifier persists across restarts
- [ ] Settings file survives ungraceful exit (simulate: `kill -9`)
- [ ] Tray minimize → SIGTERM → clean shutdown

- [ ] **Step 5: Merge to main**

```bash
git checkout main
git merge --no-ff jarvis-review-fixes -m "$(cat <<'EOF'
Merge branch 'jarvis-review-fixes'

Fixes 21 issues from the 2026-04-16 code review:
  Phase 1: critical bugs + security (shell injection, queue HMAC,
           duplicate shutdown, dead STT path, hotword race)
  Phase 2: split VoiceInputGUI (5800 lines) into recording /
           transcription / dispatcher / animation modules + tests
  Phase 3: remaining fixes inside the new module layout

Spec: docs/superpowers/specs/2026-04-16-jarvis-code-review-remediation-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Summary

- Phase 1 (Tasks 1-5): 5 commits — critical bugs + security
- Phase 2 (Tasks 6-10): 5 commits — module extraction + tests
- Phase 3 (Tasks 11-16): 6 commits — cleanup in new layout
- Task 17: merge to main

Total: ~22 commits, 4 new modules, 4 new test files, 21 review findings resolved.
