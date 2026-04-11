# CLAUDE.md — Jarvis AI Voice Assistant

## Project Overview
Personal AI voice assistant with desktop control, speaker verification, XTTS voice cloning, and proactive intelligence. Built on Python/Tkinter with CUDA GPU acceleration.

## Stack
Python 3.12 | CUDA 12.6 (2x RTX 3090) | Tkinter GUI | Whisper STT | XTTS v2 TTS | SpeechBrain | OpenWakeWord

## Project Structure
```
jarvis/                    # Main package
  voice_input_gui.py       # Main GUI (~4800 lines)
  jarvis_tts.py            # XTTS/Edge TTS engine
  jarvis_agent.py          # Intelligence layer (screen, habits, memory, workflows)
  jarvis_speak_queue.py    # File-based TTS queue
  speaker_verification.py  # ECAPA-TDNN voiceprint system
  orbit_server.py          # Animation HTTP server (unused, kept for reference)
  orbit_animation.html     # Browser animation (unused, kept for reference)
scripts/
  hotword_daemon.py        # Always-on wake word listener (systemd service)
  screen_capture.py        # Ctrl+Shift+S screenshot daemon
```

## Running
```bash
source /home/hunterp/vss_env/bin/activate
python jarvis/voice_input_gui.py          # Main GUI
python scripts/hotword_daemon.py --daemon  # Wake word daemon
```

## Key Patterns
- All GUI runs on venv Python (PIL ImageTk requirement)
- YOLO/Whisper on GPU0, XTTS/SpeechBrain on GPU1
- Settings persist to ~/.aiws_trainer/voice_settings.json
- TTS speak queue: write to /tmp/vss_voice/speak_queue.txt
- Debug logs: /tmp/vss_voice/gui_debug.log
