#!/usr/bin/env python3
"""Always-on hotword daemon — listens for wake words and launches Voice Input GUI.

Runs as a lightweight background service using OpenWakeWord (CPU, ~1.5ms/check).
When it hears "Hey Jarvis", it optionally verifies the speaker against the
enrolled voiceprint before launching the GUI.

Usage:
    python scripts/utilities/hotword_daemon.py           # Run in foreground
    python scripts/utilities/hotword_daemon.py --daemon   # Run daemonized

Systemd service: ~/.config/systemd/user/aiws-hotword.service
"""

import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JARVIS_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = Path(sys.executable)  # Use whatever python launched us
GUI_SCRIPT = JARVIS_DIR / "jarvis" / "voice_input_gui.py"
SETTINGS_FILE = Path.home() / ".aiws_trainer" / "voice_settings.json"
VOICEPRINT_FILE = Path.home() / ".aiws_trainer" / "voiceprint.npz"
LOG_DIR = Path("/tmp/vss_voice")
LOG_FILE = LOG_DIR / "hotword_daemon.log"

CHANNELS = 1
HOTWORD_COOLDOWN = 3.0
HOTWORD_THRESHOLD = 0.3  # OpenWakeWord confidence threshold (lowered for "Jarvis" without "Hey")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _log(msg):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def _load_settings():
    defaults = {"mic": None, "gpu": 1, "speaker_verify": False}
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text())
            return {
                "mic": data.get("mic", defaults["mic"]),
                "mic_name": data.get("mic_name", data.get("mic")),
                "gpu": data.get("gpu", defaults["gpu"]),
                "speaker_verify": data.get("speaker_verify", False),
                "speaker_threshold": data.get("speaker_threshold", 0.40),
            }
    except Exception as e:
        _log(f"Settings load error: {e}")
    return defaults


def _resolve_mic(mic_name):
    import sounddevice as sd
    if not mic_name or mic_name == "Default":
        return None
    # Name-based resolution (survives reboots)
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = sd.query_devices()
    idx = resolve_mic_by_name(mic_name, devices)
    if idx is not None:
        _log(f"Mic resolved: '{mic_name}' -> [{idx}]")
        return idx
    # Legacy: try index extraction
    if mic_name.startswith("["):
        try:
            idx = int(mic_name.split("]")[0][1:])
            sd.query_devices(idx, 'input')
            return idx
        except Exception:
            pass
    _log(f"WARNING: Mic '{mic_name}' not found")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and mic_name in d["name"]:
            return i
    return None


# ---------------------------------------------------------------------------
# GUI management
# ---------------------------------------------------------------------------
def _is_gui_running():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "voice_input_gui.py"],
            capture_output=True, text=True, timeout=2,
        )
        pids = result.stdout.strip().splitlines()
        own_pid = str(os.getpid())
        return any(p.strip() != own_pid for p in pids if p.strip())
    except Exception:
        return False


def _find_gui_window():
    try:
        result = subprocess.run(
            ["xdotool", "search", "--name", "Voice Input"],
            capture_output=True, text=True, timeout=2,
        )
        wids = result.stdout.strip().splitlines()
        return wids[0].strip() if wids else None
    except Exception:
        return None


def _launch_or_focus_gui():
    if _is_gui_running():
        wid = _find_gui_window()
        if wid:
            _log(f"Focusing existing GUI window {wid} and pressing F5")
            try:
                subprocess.run(
                    ["xdotool", "windowactivate", "--sync", wid],
                    capture_output=True, timeout=2,
                )
                time.sleep(0.2)
                subprocess.run(
                    ["xdotool", "key", "--window", wid, "F5"],
                    capture_output=True, timeout=2,
                )
            except Exception as e:
                _log(f"Focus/F5 error: {e}")
            return
        return

    _log("Launching Voice Input GUI with --auto-record...")
    try:
        env = os.environ.copy()
        env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
        subprocess.Popen(
            [str(VENV_PYTHON), str(GUI_SCRIPT), "--auto-record"],
            cwd=str(JARVIS_DIR), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _log("GUI launched with auto-record")
    except Exception as e:
        _log(f"Launch error: {e}")


def _notify(title, body=""):
    try:
        subprocess.Popen(
            ["notify-send", "-i", "audio-input-microphone", "-t", "2000", title, body],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Speaker verification (optional — checks if hotword came from the user)
# ---------------------------------------------------------------------------
class _SpeakerChecker:
    """Lightweight speaker check for the daemon. Loads ECAPA on demand."""

    def __init__(self, gpu, threshold):
        self.gpu = gpu
        self.threshold = threshold
        self._model = None
        self._centroid = None

    def load(self):
        """Load voiceprint and model."""
        if not VOICEPRINT_FILE.exists():
            _log("No voiceprint file — speaker check disabled")
            return False
        try:
            data = np.load(VOICEPRINT_FILE)
            embeddings = [data[k] for k in sorted(data.files)]
            if not embeddings:
                return False
            self._centroid = np.mean(embeddings, axis=0)
            _log(f"Voiceprint loaded: {len(embeddings)} samples")
        except Exception as e:
            _log(f"Voiceprint load error: {e}")
            return False

        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": f"cuda:{self.gpu}"},
                savedir=str(Path.home() / ".aiws_trainer" / "speaker_model"),
            )
            _log(f"Speaker model loaded on CUDA:{self.gpu}")
            return True
        except Exception as e:
            _log(f"Speaker model error: {e}")
            return False

    def is_user(self, audio_16k_float):
        """Check if audio matches voiceprint. Returns (is_match, score)."""
        if self._model is None or self._centroid is None:
            return True, 1.0  # No model = accept all
        if len(audio_16k_float) < 16000:  # Need at least 1s
            return True, 1.0
        try:
            import torch
            waveform = torch.tensor(audio_16k_float, dtype=torch.float32).unsqueeze(0)
            waveform = waveform.to(f"cuda:{self.gpu}")
            with torch.no_grad():
                emb = self._model.encode_batch(waveform)
            emb = emb.squeeze().cpu().numpy()
            score = float(np.dot(emb, self._centroid) / (
                np.linalg.norm(emb) * np.linalg.norm(self._centroid)))
            is_match = score >= self.threshold
            _log(f"Speaker check: score={score:.3f} threshold={self.threshold} "
                 f"{'MATCH' if is_match else 'REJECT'}")
            return is_match, score
        except Exception as e:
            _log(f"Speaker check error: {e}")
            return True, 1.0


# Wake word phrases
HOTWORD_PHRASES = {
    "jarvis", "hey jarvis", "hey jarv",
    "jarvas", "jarves", "jarvus", "service jarvis",
    "nervous", "harvest",
    "hey claude", "hey cloud", "hey claud",
    "okay claude", "ok claude",
}

HOTWORD_WINDOW = 2.0
HOTWORD_INTERVAL = 0.8
HOTWORD_MODEL = "base"
SILENCE_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Main listener loop (Whisper-based with custom phrases)
# ---------------------------------------------------------------------------
def run_daemon():
    """Main hotword listening loop using Whisper for custom keyword detection."""
    import sounddevice as sd
    from scipy.signal import resample as scipy_resample
    from faster_whisper import WhisperModel

    settings = _load_settings()
    mic_idx = _resolve_mic(settings.get("mic_name") or settings.get("mic"))
    gpu = settings.get("gpu", 1)

    _log(f"Hotword daemon starting (mic={mic_idx}, gpu={gpu})")
    _notify("Hotword Daemon", "Always-on voice activation ready")

    # Detect native sample rate
    try:
        dev_info = sd.query_devices(mic_idx, 'input')
        native_rate = int(dev_info['default_samplerate'])
    except Exception:
        native_rate = 44100
    _log(f"Mic native rate: {native_rate}Hz")

    # Load Whisper for hotword detection
    _log(f"Loading Whisper '{HOTWORD_MODEL}' for hotword detection...")
    try:
        model = WhisperModel(
            HOTWORD_MODEL, device="cuda",
            device_index=gpu, compute_type="float16",
        )
        _log(f"Whisper loaded on CUDA:{gpu}")
    except Exception as e:
        _log(f"GPU failed ({e}), falling back to CPU")
        model = WhisperModel(HOTWORD_MODEL, device="cpu", compute_type="int8")
        _log("Whisper loaded on CPU")

    # Load speaker verifier if enabled
    speaker_checker = None
    if settings.get("speaker_verify"):
        speaker_checker = _SpeakerChecker(gpu, settings.get("speaker_threshold", 0.40))
        if not speaker_checker.load():
            speaker_checker = None

    # Audio buffer
    buf = deque(maxlen=int(native_rate * HOTWORD_WINDOW))

    def callback(indata, frame_count, time_info, status):
        chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        buf.extend(chunk.tolist())

    # Open mic stream
    try:
        stream = sd.InputStream(
            samplerate=native_rate, channels=CHANNELS,
            dtype="float32", device=mic_idx,
            callback=callback,
            blocksize=int(native_rate * 0.2),
        )
        stream.start()
        _log("Mic stream open, listening for hotwords...")
    except Exception as e:
        _log(f"Mic error: {e}")
        return

    last_trigger = 0

    while True:
        time.sleep(HOTWORD_INTERVAL)

        # Skip if GUI is already running
        if _is_gui_running():
            buf.clear()
            continue

        if len(buf) < native_rate * 1.0:
            continue

        audio_raw = np.array(list(buf), dtype=np.float32)
        rms = np.sqrt(np.mean(audio_raw ** 2))
        if rms < SILENCE_THRESHOLD:
            continue

        # Resample to 16kHz for Whisper
        if native_rate != 16000:
            new_len = int(len(audio_raw) * 16000 / native_rate)
            audio = scipy_resample(audio_raw, new_len).astype(np.float32)
        else:
            audio = audio_raw

        # Denoise before hotword check
        from jarvis.audio_pipeline import denoise_audio
        audio = denoise_audio(audio, sr=16000)

        # Quick transcription — no VAD filter for short wake words
        try:
            segments, _ = model.transcribe(
                audio, language="en", beam_size=3, vad_filter=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip().lower()

            if not text:
                continue

            matched = False
            for phrase in HOTWORD_PHRASES:
                if phrase in text:
                    now = time.monotonic()
                    if now - last_trigger < HOTWORD_COOLDOWN:
                        break

                    _log(f"HOTWORD DETECTED: {text!r}")

                    # Speaker verification
                    if speaker_checker:
                        verify_audio = audio_raw
                        if native_rate != 16000:
                            vlen = int(len(verify_audio) * 16000 / native_rate)
                            verify_audio = scipy_resample(
                                verify_audio, vlen).astype(np.float32)
                        is_user, spk_score = speaker_checker.is_user(verify_audio)
                        if not is_user:
                            _log(f"Speaker rejected ({spk_score:.3f}), ignoring")
                            buf.clear()
                            break

                    _notify("Voice Activated!", f"Heard: {phrase}")
                    buf.clear()
                    last_trigger = now
                    matched = True

                    # Release mic
                    stream.stop()
                    stream.close()
                    _log("Mic released for GUI")

                    _launch_or_focus_gui()

                    # Wait while GUI is running
                    time.sleep(5.0)
                    while _is_gui_running():
                        time.sleep(2.0)
                    _log("GUI closed, resuming daemon listening")
                    time.sleep(1.0)

                    # Reopen mic
                    try:
                        buf.clear()
                        stream = sd.InputStream(
                            samplerate=native_rate, channels=CHANNELS,
                            dtype="float32", device=mic_idx,
                            callback=callback,
                            blocksize=int(native_rate * 0.2),
                        )
                        stream.start()
                        time.sleep(1.5)
                        _log("Mic stream reopened, listening again...")
                    except Exception as e:
                        _log(f"Mic reopen error: {e}")
                        return

                    break

            if not matched and text and len(text) > 2:
                _log(f"Heard: {text!r} (no hotword match)")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if "--daemon" in sys.argv:
        _log("Starting hotword daemon (background)")
    else:
        _log("Starting hotword daemon (foreground)")
        print("\n  Always-on hotword daemon (OpenWakeWord)")
        print("  Say 'Hey Jarvis' to open Voice Input")
        print("  Press Ctrl+C to quit\n")

    run_daemon()


if __name__ == "__main__":
    main()
