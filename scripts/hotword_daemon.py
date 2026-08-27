#!/usr/bin/env python3
"""Always-on hotword daemon — listens for wake words and launches Voice Input GUI.

Runs as a lightweight background service using OpenWakeWord (CPU, ~1.5ms/check).
When it hears "Hey Jarvis", it optionally verifies the speaker against the
enrolled voiceprint before launching the GUI.

Usage:
    python scripts/hotword_daemon.py           # Run in foreground
    python scripts/hotword_daemon.py --daemon   # Run daemonized

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
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))   # so `import jarvis.*` works

_VSS_PYTHON = Path.home() / "vss_env" / "bin" / "python"
VENV_PYTHON = _VSS_PYTHON if _VSS_PYTHON.exists() else Path(sys.executable)
GUI_SCRIPT = REPO_ROOT / "jarvis" / "voice_input_gui.py"
SETTINGS_FILE = Path.home() / ".aiws_trainer" / "voice_settings.json"
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
    defaults = {"mic": None, "gpu": 0, "speaker_verify": False}
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text())
            return {
                "mic": data.get("mic", defaults["mic"]),
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
    if mic_name.startswith("["):
        try:
            idx = int(mic_name.split("]")[0][1:])
            sd.query_devices(idx, 'input')
            return idx
        except Exception:
            pass
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
            cwd=str(REPO_ROOT), env=env,
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
def _load_speaker_verifier(gpu, threshold):
    """Build a jarvis.speaker.SpeakerVerifier for the daemon, or None."""
    try:
        from jarvis.speaker import SpeakerVerifier
    except Exception as e:
        _log(f"jarvis.speaker unavailable ({e}) — speaker check disabled")
        return None
    try:
        verifier = SpeakerVerifier(gpu=gpu, threshold=threshold)
        verifier.load()
        if not verifier.enrolled:
            _log("No voiceprint enrolled — speaker check disabled")
            return None
        if not verifier.load_model():
            _log("Speaker model load failed — speaker check disabled")
            return None
        return verifier
    except Exception as e:
        _log(f"Speaker verifier init error: {e}")
        return None


# Wake word phrases
HOTWORD_PHRASES = {
    "jarvis", "hey jarvis", "hey jarv",
    "jarvas", "jarves", "jarvus", "service jarvis",
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
    mic_idx = _resolve_mic(settings.get("mic"))
    gpu = settings.get("gpu", 0)

    _log(f"Hotword daemon starting (mic={mic_idx}, gpu={gpu})")
    _notify("Hotword Daemon", "Always-on voice activation ready")

    # Detect native sample rate
    try:
        dev_info = sd.query_devices(mic_idx, 'input')
        native_rate = int(dev_info['default_samplerate'])
    except Exception:
        native_rate = 44100
    _log(f"Mic native rate: {native_rate}Hz")

    # Load Whisper for hotword detection.
    # GB10 has no CUDA-capable ctranslate2 — go straight to CPU int8.
    _log(f"Loading Whisper '{HOTWORD_MODEL}' for hotword detection...")
    model = WhisperModel(HOTWORD_MODEL, device="cpu", compute_type="int8")
    _log("Whisper loaded on CPU (int8)")

    # Load speaker verifier if enabled
    speaker_checker = None
    if settings.get("speaker_verify"):
        speaker_checker = _load_speaker_verifier(
            gpu, settings.get("speaker_threshold", 0.40))

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
                        is_user, spk_score = speaker_checker.verify(verify_audio)
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
