"""Recorder for Jarvis V3 — mic streams, silence auto-stop, calibration, beeps.

Ports the audio-critical paths of voice_input_gui.py verbatim (this machine has
no microphone, so the constants and order of operations must move untouched):

- beep synthesis / playback            (monolith 526-573)
- noise calibration                    (monolith 2262-2340)
- 1s restart debounce                  (monolith 2360-2364)
- audio callback + silence detection   (monolith 2403-2434)
- resample to 16k                      (monolith 2477-2485)
- stop finalization, 60s cap, gate     (monolith 2487-2541)
- silence / speaker-silence auto-stop  (monolith 4295-4441)
- fixed-duration recording             (monolith 5000-5054)

Differences from the monolith, per the V3 spec:
- No tkinter. Auto-stop runs on the recorder's own poll thread and publishes
  RecordingStopped instead of calling _stop_and_transcribe.
- Thresholds (noise_threshold, noise_gate flag) are cached BEFORE the audio
  callback closure is created — fixes the off-thread Tk variable read.
- All ad-hoc getattr state is an explicit __init__ field.
- No-mic machines: start() is a graceful no-op with MicState/Status events.
"""
from __future__ import annotations

import io
import math
import struct
import subprocess
import threading
import time
import wave
from collections import deque
from contextlib import contextmanager

import numpy as np

from jarvis.config import CONFIG, MACHINE, PATHS
from jarvis.events import (
    AudioLevel,
    MicState,
    RecordingStarted,
    RecordingStopped,
    Status,
    bus,
)
from jarvis.logs import get_logger

log = get_logger("recorder")

# Audio constants — monolith 53-56, 89, 92 (verbatim)
SAMPLE_RATE = 16000
CHANNELS = 1
NOISE_GATE_THRESHOLD = 0.005
WAVEFORM_BARS = 64
MAX_RECORDING_SECONDS = 60      # monolith 4299 — hard cap to prevent memory issues


# ------------------------------------------------------------------
# Sound generation (no external files needed) — port of monolith 526-573
# ------------------------------------------------------------------
def _generate_beep(freq=880, duration_ms=120, volume=0.3):
    """Generate a short beep as WAV bytes."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        for i in range(n_samples):
            t = i / SAMPLE_RATE
            env = min(i / 200, 1.0) * min((n_samples - i) / 200, 1.0)
            sample = int(volume * env * 32767 * math.sin(2 * math.pi * freq * t))
            w.writeframes(struct.pack("<h", max(-32767, min(32767, sample))))
    return buf.getvalue()


_BEEP_FILES: dict[str, str] = {}
_BEEP_LOCK = threading.Lock()


def _init_beeps():
    """Pre-generate beep WAV files to /tmp (module-owned)."""
    with _BEEP_LOCK:
        if _BEEP_FILES:
            return
        try:
            PATHS.LOG_DIR.mkdir(parents=True, exist_ok=True)
            start_path = PATHS.LOG_DIR / "beep_start.wav"
            stop_path = PATHS.LOG_DIR / "beep_stop.wav"
            if not start_path.exists():
                start_path.write_bytes(_generate_beep(freq=880, duration_ms=100))
            if not stop_path.exists():
                stop_path.write_bytes(_generate_beep(freq=660, duration_ms=150))
            _BEEP_FILES["start"] = str(start_path)
            _BEEP_FILES["stop"] = str(stop_path)
        except Exception:
            log.exception("beep init failed")


def play_beep(kind: str):
    """Play the 'start' or 'stop' beep asynchronously via paplay/aplay."""
    _init_beeps()
    path = _BEEP_FILES.get(kind)
    if not path:
        return
    for cmd in [["paplay", path], ["aplay", "-q", path]]:
        try:
            subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return
        except FileNotFoundError:
            continue


# ------------------------------------------------------------------
# Noise gate — port of monolith 601-610 (verbatim)
# ------------------------------------------------------------------
def _apply_noise_gate(audio, threshold=NOISE_GATE_THRESHOLD, block_size=1600):
    """Zero out blocks of audio below RMS threshold."""
    result = audio.copy()
    for i in range(0, len(result), block_size):
        block = result[i:i + block_size]
        rms = np.sqrt(np.mean(block ** 2))
        if rms < threshold:
            result[i:i + block_size] = 0.0
    return result


# ------------------------------------------------------------------
# Mic arbiter — THE single owner of the microphone
# ------------------------------------------------------------------
class MicArbiter:
    """Serializes mic ownership. The hotword listener registers pause/resume
    callbacks; every other consumer wraps its mic use in acquire(), which
    pauses the hotword stream and resumes it on exit. Re-entrant: nested
    acquires pause once and resume once."""

    def __init__(self):
        self._lock = threading.RLock()
        self._depth = 0
        self._owner = ""
        self._pause_cb = None
        self._resume_cb = None

    def register_hotword(self, pause_cb, resume_cb):
        with self._lock:
            self._pause_cb = pause_cb
            self._resume_cb = resume_cb

    @contextmanager
    def acquire(self, owner: str):
        """Pause hotword for the duration of the block; resume on exit."""
        with self._lock:
            self._depth += 1
            first = self._depth == 1
            if first:
                self._owner = owner
                if self._pause_cb is not None:
                    try:
                        self._pause_cb()
                    except Exception:
                        log.exception("hotword pause failed (owner=%s)", owner)
        try:
            yield self
        finally:
            with self._lock:
                self._depth = max(0, self._depth - 1)
                last = self._depth == 0
                if last:
                    self._owner = ""
                    if self._resume_cb is not None:
                        try:
                            self._resume_cb()
                        except Exception:
                            log.exception("hotword resume failed (owner=%s)", owner)

    @property
    def held_by(self) -> str:
        with self._lock:
            return self._owner


# ------------------------------------------------------------------
# Recorder
# ------------------------------------------------------------------
class Recorder:
    """Owns recording sessions. start()/stop()/abort() for the main pipeline;
    record_fixed() for enrollment/wake-training; calibrate_noise() for the
    noise-floor calibration flow. Publishes RecordingStarted/RecordingStopped,
    AudioLevel (~12Hz), MicState, and Status events. Never touches the UI.

    On auto-stop (silence / speaker-silence / 60s cap) the poll thread calls
    stop(reason=...); the finalized 16k mono float32 audio is stored in
    self.last_audio and RecordingStopped(reason) is published — the pipeline
    wiring reads recorder.last_audio on that event.
    """

    _POLL_S = 0.083            # AudioLevel ~12Hz
    _SILENCE_POLL_S = 0.3      # monolith 4344: root.after(300, ...)
    _SPEAKER_POLL_S = 3.0      # monolith 4390: root.after(3000, ...)

    def __init__(self, arbiter: MicArbiter, speaker_verifier=None):
        self._arbiter = arbiter
        # Optional jarvis.speaker.SpeakerVerifier for voice-aware auto-stop;
        # may be injected later by assigning recorder.speaker_verifier.
        self.speaker_verifier = speaker_verifier

        # Explicit state inventory (was scattered getattr state in the monolith)
        self.recording = False
        self.last_audio: np.ndarray | None = None
        self._stream = None
        self._audio_frames: list = []
        self._audio_level = 0.0
        self._waveform_buffer = deque(maxlen=WAVEFORM_BARS)
        self._record_rate = SAMPLE_RATE          # was getattr'd (monolith 2479)
        self._loud_chunks = 0                    # was getattr'd (monolith 2421)
        self._last_record_start = 0.0            # was hasattr'd (monolith 2360)
        self._record_start_time = 0.0
        self._silence_start = None
        self._speaker_silence_start = None       # Voice-ID silence tracker
        self._speaker_silence_misses = 0         # was getattr'd (monolith 4413)
        self._voice_id_match = False             # exposed for UI/reactor
        self._voice_stopped = False              # manual/auto stop — skip restart

        self._poll_thread: threading.Thread | None = None
        self._stop_lock = threading.Lock()
        self._session_ctx = None                 # held arbiter context

        self._mic_devices: dict[str, int | None] = {"Default": None}
        self._detect_mics()
        self._publish_mic_state()

    # -- mic enumeration (port of monolith 1282-1293) -------------------
    def _detect_mics(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self._mic_devices = {"Default": None}
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    name = f"[{i}] {d['name']}"
                    self._mic_devices[name] = i
        except Exception as e:
            log.warning("Mic detection error: %s", e)
            self._mic_devices = {"Default": None}

    @property
    def mic_available(self) -> bool:
        return bool(MACHINE.has_mic)

    @property
    def mic_devices(self) -> dict:
        return dict(self._mic_devices)

    def _publish_mic_state(self):
        name = MACHINE.mic_names[0] if MACHINE.mic_names else ""
        bus.publish(MicState(available=self.mic_available, device_name=name))

    def _resolve_mic(self):
        """Configured mic name -> device index; fall back to default
        (port of monolith 2370-2377)."""
        mic_name = CONFIG.mic
        if mic_name not in self._mic_devices:
            log.info("Saved mic %r not found, using default", mic_name)
            return None
        return self._mic_devices.get(mic_name)

    def _native_rate(self, mic_idx) -> int:
        """Device native sample rate; many mics only do 44100/48000
        (port of monolith 2379-2384)."""
        try:
            import sounddevice as sd
            dev_info = sd.query_devices(mic_idx, 'input')
            return int(dev_info['default_samplerate'])
        except Exception:
            return 44100

    # -- resample (port of monolith 2477-2485, verbatim math) -----------
    def _resample_to_16k(self, audio):
        """Resample audio from recording rate to 16kHz for Whisper."""
        rate = self._record_rate
        if rate == SAMPLE_RATE:
            return audio
        from scipy.signal import resample
        new_len = int(len(audio) * SAMPLE_RATE / rate)
        resampled = resample(audio, new_len).astype(np.float32)
        return resampled

    # -- session control -------------------------------------------------
    def start(self):
        """Begin a recording session. No-op (with warn Status) when no mic."""
        import_err = None
        if not self.mic_available:
            self._publish_mic_state()
            bus.publish(Status(text="No microphone detected", kind="warn"))
            log.warning("start() ignored: no input devices")
            return

        # Guard against rapid restart loop (port of monolith 2358-2364)
        if self.recording:
            return
        if self._last_record_start:
            elapsed = time.monotonic() - self._last_record_start
            if elapsed < 1.0:
                return
        self._last_record_start = time.monotonic()

        try:
            import sounddevice as sd
        except Exception as e:
            import_err = e
        if import_err is not None:
            log.exception("sounddevice unavailable")
            bus.publish(Status(text=f"Audio backend error: {import_err}", kind="error"))
            return

        # Pause hotword for the whole session (replaces the scattered
        # hotword pause at monolith 2367-2368)
        self._session_ctx = self._arbiter.acquire("recorder")
        self._session_ctx.__enter__()

        mic_idx = self._resolve_mic()
        native_rate = self._native_rate(mic_idx)
        self._record_rate = native_rate
        log.info("Mic native rate: %sHz", native_rate)

        # Reset session state (port of monolith 2388-2398)
        self._audio_frames = []
        self._silence_start = None
        self._loud_chunks = 0
        self._speaker_silence_start = None
        self._speaker_silence_misses = 0
        self._voice_id_match = False
        self._voice_stopped = False
        self._waveform_buffer.clear()
        self.last_audio = None

        # Cache thresholds for the audio thread BEFORE creating the closure
        # (monolith 2400-2401 cached only silence_thresh and still read the
        # noise-gate Tk var inside the callback — both cached here).
        silence_thresh = CONFIG.noise_threshold
        gate_on = CONFIG.noise_gate

        # Audio callback (port of monolith 2403-2434, verbatim logic)
        def audio_callback(indata, frame_count, time_info, status):
            if not self.recording:
                return

            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()

            # Measure RMS on original audio BEFORE noise gate (for silence detection)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            self._audio_level = min(rms * 10, 1.0)

            # Silence detection uses original signal
            # Only reset silence timer on sustained sound (2+ consecutive loud chunks)
            # This prevents single noise spikes from restarting the countdown
            if rms < silence_thresh:
                self._loud_chunks = 0
                if self._silence_start is None:
                    self._silence_start = time.monotonic()
            else:
                self._loud_chunks = self._loud_chunks + 1
                if self._loud_chunks >= 2:
                    self._silence_start = None

            # Noise gate: zero out quiet blocks (after silence check)
            if gate_on and rms < NOISE_GATE_THRESHOLD:
                chunk[:] = 0.0

            self._audio_frames.append(chunk.reshape(-1, 1))

            # Waveform samples
            step = max(1, len(chunk) // 4)
            for val in chunk[::step]:
                self._waveform_buffer.append(float(val))

        # Set recording state BEFORE opening mic (mic open can block briefly)
        self.recording = True
        self._record_start_time = time.monotonic()
        bus.publish(RecordingStarted())
        bus.publish(Status(text="Listening...", kind="busy"))

        if CONFIG.sound:
            threading.Thread(target=play_beep, args=("start",), daemon=True).start()

        # Recorder-owned poll thread replaces the Tk-after chains
        # (_update_waveform / _check_silence / _check_speaker_silence)
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="recorder-poll")
        self._poll_thread.start()

        # Open mic stream (may block briefly)
        try:
            self._stream = sd.InputStream(
                samplerate=native_rate, channels=CHANNELS,
                dtype="float32", device=mic_idx,
                callback=audio_callback,
                blocksize=int(native_rate * 0.1),
            )
            self._stream.start()
        except Exception as e:
            log.exception("Mic open error")
            self.recording = False
            bus.publish(Status(text=f"Mic error: {str(e)[:50]}", kind="error"))
            self._join_poll_thread()
            self._release_session()
            bus.publish(RecordingStopped(reason="abort"))
            return

        log.info("Recording started")

    def stop(self, reason: str = "manual") -> np.ndarray | None:
        """End the session; finalize audio (resample, cap, gate — port of
        monolith 2487-2541). Publishes RecordingStopped(reason). Returns 16k
        mono float32 audio, or None if nothing usable was captured. The same
        array is stored in self.last_audio for event-driven consumers."""
        with self._stop_lock:
            if not self.recording:
                return None
            self.recording = False

        self._audio_level = 0.0
        self._voice_stopped = True

        if CONFIG.sound:
            threading.Thread(target=play_beep, args=("stop",), daemon=True).start()

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.warning("mic stream close failed", exc_info=True)
            self._stream = None

        self._join_poll_thread()
        self._release_session()

        audio = self._finalize_audio()
        self.last_audio = audio
        bus.publish(RecordingStopped(reason=reason))
        return audio

    def abort(self):
        """Discard the current session without producing audio."""
        with self._stop_lock:
            if not self.recording:
                return
            self.recording = False

        self._audio_level = 0.0
        self._voice_stopped = True

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.warning("mic stream close failed", exc_info=True)
            self._stream = None

        self._join_poll_thread()
        self._release_session()

        self._audio_frames = []
        self.last_audio = None
        bus.publish(RecordingStopped(reason="abort"))
        log.info("Recording aborted")

    def _release_session(self):
        ctx, self._session_ctx = self._session_ctx, None
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                log.exception("arbiter release failed")

    def _join_poll_thread(self):
        t = self._poll_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._poll_thread = None

    def _finalize_audio(self) -> np.ndarray | None:
        """Snapshot frames -> 16k float32 (port of monolith 2508-2541)."""
        # Snapshot audio frames (callback thread may still be draining)
        frames = list(self._audio_frames)
        if not frames:
            bus.publish(Status(text="No audio captured", kind="info"))
            return None

        try:
            audio_raw = np.concatenate(frames, axis=0).flatten()
        except ValueError:
            log.error("audio frames empty after snapshot")
            bus.publish(Status(text="No audio captured", kind="info"))
            return None

        audio = self._resample_to_16k(audio_raw)
        duration = len(audio) / SAMPLE_RATE
        log.info("Stopped: %.1fs audio", duration)

        # Cap audio at 60 seconds to prevent freezes
        max_samples = SAMPLE_RATE * 60
        if len(audio) > max_samples:
            log.info("Audio capped from %.1fs to 60s", duration)
            audio = audio[:max_samples]
            duration = 60.0

        if duration < 0.3:
            bus.publish(Status(text="Too short", kind="info"))
            return None

        # Apply noise gate to final audio
        if CONFIG.noise_gate:
            audio = _apply_noise_gate(audio)

        return audio

    # -- poll loop: AudioLevel + auto-stop -------------------------------
    def _poll_loop(self):
        """Publishes AudioLevel ~12Hz; runs the silence check every 300ms and
        the speaker-silence check every 3s (replaces the monolith's
        root.after chains at 2450-2452 / 4344 / 4390)."""
        next_silence = 0.0
        next_speaker = 0.0
        while self.recording:
            now = time.monotonic()
            waveform = [min(abs(v), 1.0) for v in self._waveform_buffer]
            bus.publish(AudioLevel(level=self._audio_level, waveform=waveform))
            if now >= next_silence:
                next_silence = now + self._SILENCE_POLL_S
                if self._check_silence():
                    return
            if now >= next_speaker:
                next_speaker = now + self._SPEAKER_POLL_S
                self._check_speaker_silence()
            time.sleep(self._POLL_S)

    def _check_silence(self) -> bool:
        """Port of monolith 4301-4344. Returns True when it stopped the
        session (the poll loop then exits)."""
        if not self.recording:
            return True

        # Hard cap on recording length
        if self._record_start_time:
            elapsed = time.monotonic() - self._record_start_time
            if elapsed >= MAX_RECORDING_SECONDS:
                log.info("Max recording time reached (%ss)", MAX_RECORDING_SECONDS)
                self._voice_stopped = True
                try:
                    self.stop(reason="cap")
                except Exception:
                    log.exception("Max recording stop error")
                    self.recording = False
                return True

        timeout = CONFIG.silence_timeout
        min_frames = int(SAMPLE_RATE * 0.5 / (SAMPLE_RATE * 0.1))
        # Don't auto-stop in the first 5 seconds of recording (grace period)
        if self._record_start_time and (time.monotonic() - self._record_start_time) < 5.0:
            self._silence_start = None
            self._loud_chunks = 0
            return False
        if (self._silence_start is not None
                and (time.monotonic() - self._silence_start) >= timeout
                and len(self._audio_frames) > min_frames):
            log.info("Auto-stop on silence (%ss timeout)", timeout)
            self._voice_stopped = True
            try:
                self.stop(reason="silence")
            except Exception:
                log.exception("Auto-stop error")
                self.recording = False
                bus.publish(Status(text="Auto-stop error", kind="error"))
            return True

        return False

    def _check_speaker_silence(self):
        """Voice-aware silence: auto-stop when the user hasn't spoken for the
        timeout, even if background audio (TV/YouTube) keeps making noise.
        Port of monolith 4346-4390; verify runs on a helper thread exactly as
        before so the poll loop keeps publishing levels."""
        if not self.recording:
            return
        verifier = self.speaker_verifier
        if not CONFIG.speaker_verify or verifier is None:
            return   # Voice ID not active — regular silence detection handles it
        try:
            if not verifier.enrolled:
                return
        except Exception:
            log.exception("speaker verifier enrolled-check failed")
            return

        now = time.monotonic()
        # Don't check for the first 3 seconds of recording
        if self._record_start_time and (now - self._record_start_time) < 3.0:
            return

        # Grab the last 3 seconds of audio
        rate = self._record_rate
        samples_needed = int(rate * 3.0)
        frames = list(self._audio_frames)
        if not frames:
            return

        recent = np.concatenate(frames[-max(1, samples_needed // int(rate * 0.1)):],
                                axis=0).flatten()
        # Resample to 16kHz for speaker check
        audio_16k = self._resample_to_16k(recent)

        def _check():
            try:
                is_match, score = verifier.verify(audio_16k)
            except Exception:
                log.warning("speaker silence verify failed", exc_info=True)
                return
            self._on_speaker_silence_result(is_match, score)

        threading.Thread(target=_check, daemon=True).start()

    def _on_speaker_silence_result(self, is_match, score):
        """Handle result of periodic speaker check during recording.
        Port of monolith 4392-4441 (thresholds and miss-count verbatim)."""
        if not self.recording:
            return

        # Use a relaxed threshold for silence detection — we don't want to
        # cut the user off mid-sentence on a borderline score. The strict
        # threshold is applied later by the segment filter.
        relaxed_threshold = CONFIG.speaker_threshold * 0.70
        is_probably_user = score >= relaxed_threshold
        self._voice_id_match = is_probably_user

        if is_probably_user:
            # User is likely speaking — reset speaker silence timer
            if self._speaker_silence_start is not None:
                log.info("Voice-ID silence reset (score=%.3f >= relaxed %.3f)",
                         score, relaxed_threshold)
            self._speaker_silence_start = None
            self._speaker_silence_misses = 0
        else:
            # Probably not the user — count consecutive misses
            self._speaker_silence_misses += 1

            # Require 2 consecutive misses before starting silence timer
            # (one borderline check shouldn't trigger a stop)
            if self._speaker_silence_misses < 2:
                log.info("Voice-ID miss 1/2 (score=%.3f), waiting for confirmation",
                         score)
                return

            if self._speaker_silence_start is None:
                self._speaker_silence_start = time.monotonic()
                log.info("Voice-ID silence started (score=%.3f, 2 consecutive misses)",
                         score)

            elapsed = time.monotonic() - self._speaker_silence_start
            timeout = CONFIG.silence_timeout
            if elapsed >= timeout and len(self._audio_frames) > 5:
                log.info("Voice-ID auto-stop: no user voice for %.1fs "
                         "(last score=%.3f)", elapsed, score)
                self._voice_stopped = True
                try:
                    self.stop(reason="silence")
                except Exception:
                    log.exception("Voice-ID auto-stop error")
                    self.recording = False

    # -- fixed-duration recording (port of monolith 5000-5054) -----------
    def record_fixed(self, seconds: float) -> np.ndarray:
        """Blocking capture of `seconds` of audio at the mic's native rate,
        resampled to 16k mono float32. Used for enrollment / wake-word
        training. Returns an empty array on failure. Runs under the arbiter
        (hotword paused for the duration)."""
        if not self.mic_available:
            bus.publish(Status(text="No microphone detected", kind="warn"))
            return np.zeros(0, dtype=np.float32)
        if self.recording:
            bus.publish(Status(text="Stop recording first", kind="warn"))
            return np.zeros(0, dtype=np.float32)

        try:
            import sounddevice as sd
        except Exception as e:
            log.exception("sounddevice unavailable")
            bus.publish(Status(text=f"Audio backend error: {e}", kind="error"))
            return np.zeros(0, dtype=np.float32)

        with self._arbiter.acquire("record_fixed"):
            mic_idx = self._resolve_mic()
            native_rate = self._native_rate(mic_idx)

            frames: list = []

            def cb(indata, frame_count, time_info, status):
                chunk = indata[:, 0].copy() if indata.ndim > 1 \
                    else indata.flatten().copy()
                frames.append(chunk)

            try:
                stream = sd.InputStream(
                    samplerate=native_rate, channels=CHANNELS,
                    dtype="float32", device=mic_idx,
                    callback=cb, blocksize=int(native_rate * 0.1),
                )
                stream.start()
                time.sleep(float(seconds))
                stream.stop()
                stream.close()
            except Exception as e:
                log.exception("Fixed recording error")
                bus.publish(Status(text=f"Recording failed: {str(e)[:50]}",
                                   kind="error"))
                return np.zeros(0, dtype=np.float32)

        if not frames:
            bus.publish(Status(text="No audio captured", kind="warn"))
            return np.zeros(0, dtype=np.float32)

        audio_raw = np.concatenate(frames).flatten()
        # Resample to 16kHz (verbatim math, monolith 5048-5053)
        if native_rate != SAMPLE_RATE:
            from scipy.signal import resample
            new_len = int(len(audio_raw) * SAMPLE_RATE / native_rate)
            audio = resample(audio_raw, new_len).astype(np.float32)
        else:
            audio = audio_raw
        return audio

    # -- noise calibration (port of monolith 2262-2338) -------------------
    def calibrate_noise(self) -> float:
        """Blocking: sample background noise for 3 seconds and set the
        threshold above it. Saves to CONFIG.noise_threshold and returns the
        new value (current value on failure). Hotword paused via arbiter."""
        current = CONFIG.noise_threshold
        if self.recording:
            bus.publish(Status(text="Stop recording first", kind="warn"))
            return current
        if not self.mic_available:
            bus.publish(Status(text="No microphone detected", kind="warn"))
            return current

        try:
            import sounddevice as sd
        except Exception as e:
            log.exception("sounddevice unavailable")
            bus.publish(Status(text=f"Audio backend error: {e}", kind="error"))
            return current

        bus.publish(Status(text="Calibrating — be quiet for 3 seconds",
                           kind="busy"))

        with self._arbiter.acquire("calibrate"):
            mic_idx = self._resolve_mic()
            native_rate = self._native_rate(mic_idx)

            samples: list = []

            def cb(indata, frames, t, status):
                rms = float(np.sqrt(np.mean(indata ** 2)))
                samples.append(rms)

            try:
                stream = sd.InputStream(
                    samplerate=native_rate, channels=CHANNELS,
                    dtype="float32", device=mic_idx,
                    callback=cb, blocksize=int(native_rate * 0.1),
                )
                stream.start()
                time.sleep(3.0)
                stream.stop()
                stream.close()
            except Exception as e:
                log.exception("Calibration error")
                bus.publish(Status(text=f"Calibration failed: {e}", kind="error"))
                return current

        if not samples:
            bus.publish(Status(text="No audio captured", kind="error"))
            return current

        arr = np.array(samples)
        p99 = float(np.percentile(arr, 99))
        # Set threshold 50% above the 99th percentile of background noise
        new_threshold = round(p99 * 1.5, 4)
        # Clamp to reasonable range
        new_threshold = max(0.01, min(0.15, new_threshold))

        log.info("Calibration: mean=%.4f p99=%.4f -> threshold=%s",
                 arr.mean(), p99, new_threshold)

        CONFIG.update(noise_threshold=new_threshold)
        bus.publish(Status(
            text=f"Noise floor: {p99:.4f} → threshold: {new_threshold}",
            kind="ok"))
        return new_threshold

    # -- misc state ------------------------------------------------------
    @property
    def voice_id_match(self) -> bool:
        """Latest periodic speaker-check verdict (UI/reactor indicator)."""
        return self._voice_id_match

    @property
    def audio_level(self) -> float:
        return self._audio_level

    @property
    def voice_stopped(self) -> bool:
        """True when the last stop was manual/auto (skip continuous restart)."""
        return self._voice_stopped
