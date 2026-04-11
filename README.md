# Jarvis — AI Voice Assistant

Personal AI voice assistant with desktop control, speaker verification, XTTS voice cloning, and proactive intelligence.

## Features

- **Voice Input** — Whisper STT with speaker verification and Voice ID
- **JARVIS Voice** — XTTS v2 voice clone from Paul Bettany reference
- **Desktop Control** — window switching, tabs, scrolling, clicking, keyboard shortcuts
- **Hotword Detection** — "Jarvis" / "Hey Jarvis" wake word (Whisper + OpenWakeWord)
- **Smart Commands** — 50+ voice commands for system control, git, files, search
- **Proactive Alerts** — GPU temp, disk space monitoring
- **Habit Learning** — learns your command patterns and suggests actions
- **Contextual Memory** — remember/recall facts across sessions
- **Arc Reactor UI** — animated Tkinter GUI with cyan theme

## Quick Start

```bash
source /home/hunterp/vss_env/bin/activate
python jarvis/voice_input_gui.py
```

## Hotword Daemon

```bash
python scripts/hotword_daemon.py --daemon
```

## Voice Commands

Say "Jarvis" or hold Ctrl+Shift+R, then:

- "switch to Opera" / "open terminal"
- "scroll down" / "next tab" / "close tab"
- "commit" / "run tests" / "check GPU"
- "what time is it" / "what's the weather"
- "remember that..." / "recall..."
- "remind me in 30 minutes to..."
- "deploy" / "morning" / "training check"
- "good night"

## Requirements

- Python 3.12+ with CUDA
- 2x RTX 3090 (or any CUDA GPU)
- Whisper, SpeechBrain, Piper/XTTS, OpenWakeWord
- Linux with X11 (xdotool)
