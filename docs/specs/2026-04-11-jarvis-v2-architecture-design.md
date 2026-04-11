# Jarvis V2 — Tiered Intelligence Architecture

**Date:** 2026-04-11
**Scope:** Full architectural redesign of Jarvis voice assistant
**Priority:** Intelligence > Reliability > Polish > New Features

## Problem Statement

Jarvis v1 was built in a single marathon session. All features (~70 voice commands, TTS, speaker verification, desktop control, brain routing) are crammed into `voice_input_gui.py` (5,600 lines). The system works but has:

- Monolithic architecture — everything in one file, hard to maintain
- Inconsistent intelligence — some commands are smart, others are dumb pattern matching
- No persistent memory across sessions
- Context blindness — the brain doesn't know what you're working on
- Reliability issues — recording cutoffs, hotword misses, animation lag
- No autonomy — can't chain complex multi-step tasks without explicit instructions

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│                   GUI Layer                  │
│  gui.py — Arc Reactor UI, buttons, layout   │
├─────────────────────────────────────────────┤
│              Audio Pipeline                  │
│  recorder.py ──► transcriber.py             │
│  (mic, silence)   (Whisper, speaker verify) │
├─────────────────────────────────────────────┤
│             Command Router                   │
│  commander.py — parses intent, routes to:   │
├──────┬───────────┬──────────────────────────┤
│ T1   │    T2     │          T3              │
│Local │  Ollama   │     Claude CLI           │
│<0.5s │  ~2-3s    │      ~5-15s              │
│      │           │                          │
│Time  │  Chat     │  Multi-step tasks        │
│Cmds  │  Summarize│  Code analysis           │
│Desktop│ Explain  │  Autonomous workflows    │
│System │ Simple Q │  Debug, deploy, refactor │
├──────┴───────────┴──────────────────────────┤
│            Context Engine                    │
│  context.py — feeds ALL tiers with:         │
│  git state, screen, files, memory, habits   │
├─────────────────────────────────────────────┤
│              Output Layer                    │
│  tts.py — XTTS/Edge speech                  │
│  desktop.py — xdotool actions               │
│  speak_queue.py — file-based TTS queue      │
└─────────────────────────────────────────────┘
```

## Module Breakdown

### 1. gui.py — UI Layer (~800 lines)
**Purpose:** Arc Reactor display, buttons, status bar, settings panel, dual transcription boxes.

Contains:
- Tkinter window setup, color palette, theme
- Arc Reactor particle orbit animation (PIL rendering)
- Status bar, record button, settings toggles
- "You" and "Jarvis" transcription boxes
- Collapsible settings panel
- No audio, no AI logic — pure UI

Communicates with other modules via callbacks and state variables.

### 2. recorder.py — Audio Recording (~400 lines)
**Purpose:** Mic management, silence detection, push-to-talk.

Contains:
- sounddevice mic stream open/close
- Audio callback (frames, RMS, waveform buffer)
- Silence auto-stop with grace period
- Voice-ID silence detection (speaker-aware)
- Push-to-talk (Ctrl+Shift+R) support
- Max recording cap (60s)
- Noise gate, calibration

Outputs: raw audio numpy array when recording stops.

### 3. transcriber.py — Speech-to-Text (~500 lines)
**Purpose:** Whisper transcription + speaker verification + confidence gating.

Contains:
- faster-whisper model loading and transcription
- Segment-level speaker filtering (keep user's voice, drop TV)
- Confidence gate (reject low avg_logprob)
- Partial streaming transcription (live preview)
- Voice command detection in partials (filler words, stop phrases)

Inputs: raw audio. Outputs: transcribed text + confidence data.

### 4. commander.py — Command Router (~600 lines)
**Purpose:** Parse transcribed text, decide what to do, route to correct tier.

Contains:
- Intent classifier (learned, with "Was this for me?" prompt)
- Quick command matching (70+ patterns)
- Desktop control parser (window, tab, scroll, click, chain)
- Workflow executor (deploy, morning, training check)
- Dictation mode toggle
- Reminder/timer system
- Voice notes, clipboard, file finder

Decision logic:
1. Is it a direct command? → Tier 1 (execute locally)
2. Is it a simple question? → Tier 2 (Ollama)
3. Is it complex/needs reasoning? → Tier 3 (Claude)
4. Uncertain? → ask the user

### 5. brain.py — Tiered Intelligence (~400 lines)
**Purpose:** Route to Ollama or Claude with rich context.

**Tier 2 (Ollama):**
- System prompt with Jarvis personality
- Context injection from context engine
- Temperature/token control for concise responses
- Model: llama3.2 (local, fast)

**Tier 3 (Claude CLI):**
- `claude -p` with structured action format
- Rich context: conversation history, git state, screen info
- Structured output: [SPEAK], [RUN], [TYPE], [WINDOW], [SILENT]
- 120s timeout with graceful fallback
- Conversation history maintained (last 20 exchanges)

**Autonomy features:**
- Multi-step task execution (run command → analyze output → decide next)
- Self-correction (if command fails, try alternative)
- Proactive suggestions based on context

### 6. context.py — Context Engine (~500 lines)
**Purpose:** Build and maintain a rich context snapshot for all tiers.

Gathers (lazily, cached 30s):
- **Git state:** branch, changed files, recent commits, ahead/behind
- **Active window:** name, geometry, app type
- **Recent files:** last 5 modified in project
- **Conversation history:** last 10 exchanges
- **Screen capture:** latest screenshot path
- **System state:** GPU util/temp, disk, uptime, running processes
- **User memory:** saved notes, habits, preferences
- **Error context:** recent errors from logs

Provides a `get_context(detail_level)` method:
- `"minimal"` — just active window + last command (for Tier 1)
- `"standard"` — + git state, recent files, conversation (for Tier 2)
- `"full"` — everything including screen + errors (for Tier 3)

### 7. tts.py — Voice Synthesis (existing, ~240 lines)
Already separated. Dual engine: Edge (fast) + XTTS (JARVIS clone).

### 8. desktop.py — Desktop Control (~300 lines)
**Purpose:** All xdotool operations extracted from voice_input_gui.py.

Contains:
- Window switching, focusing, listing
- Tab control (next, prev, new, close)
- Scrolling, clicking (left, right, double)
- Keyboard shortcuts (copy, paste, undo, save, find)
- Multi-monitor window movement
- OCR text finding and clicking (when tesseract available)
- App launching
- Media controls (volume, play/pause)

### 9. hotword.py — Wake Word Detection (~200 lines)
**Purpose:** OpenWakeWord listener extracted from voice_input_gui.py.

The GUI's internal hotword listener, cleaned up. OpenWakeWord for the GUI, Whisper for the daemon (separate process).

### 10. memory.py — Persistent Memory (~300 lines)
**Purpose:** Long-term memory across sessions.

Stores in `~/.aiws_trainer/jarvis_memory/`:
- **Conversation summaries** — compressed history of past sessions
- **User preferences** — learned from interactions
- **Project knowledge** — facts about repos, file purposes
- **Habit patterns** — command frequency by time/day
- **Voice notes** — timestamped text memos
- **Intent feedback** — "Was this for me?" learning data

Provides: `remember(key, value)`, `recall(query)`, `get_habits()`, `get_preferences()`

## Intelligence Features

### Persistent Conversation Memory
- Each session's conversation is summarized and stored
- On startup, the last 3 session summaries are loaded into context
- "Jarvis, what were we working on last time?" works

### Autonomous Multi-Step Tasks
- "Jarvis, deploy" → runs tests → checks results → if pass: commit → push → create PR
- Each step's output feeds into the next step's decision
- If a step fails, Jarvis explains why and suggests alternatives
- User can interrupt at any point

### Screen-Aware Context
- Before answering complex questions, Jarvis captures the screen
- Knows which app you're in, what file is open
- "Jarvis, fix this" — analyzes visible error on screen

### Proactive Assistance
- Notices patterns: "You usually run tests before committing. Shall I?"
- GPU temp warnings, disk space alerts
- Training completion notifications
- Reminds you of things you asked to be reminded about

### Personality
- MCU Jarvis: professional, concise, slightly warm, dry humor
- Adapts formality based on context (casual chat vs. technical work)
- Never robotic — varies phrasing, uses natural transitions
- Remembers user's name and preferences

## Reliability Improvements

### Recording Stability
- Max 60s recording cap (prevents freezes)
- 5-second grace period before silence can trigger
- Separate audio thread — never blocks UI
- Graceful error recovery in all audio paths

### Hotword Detection
- Whisper-based daemon (custom phrases: "Jarvis", "Hey Jarvis")
- OpenWakeWord in GUI (low latency when GUI is open)
- Speaker verification on hotword (blocks TV/YouTube triggers)
- Hotword pauses during TTS playback (no feedback loops)

### Animation Performance
- Pre-rendered arc reactor frames (no per-frame numpy glow)
- PIL particles rendered in background thread
- 12fps during recording, 16fps idle
- Never blocks audio pipeline

## Migration Plan

1. Create new module files alongside existing voice_input_gui.py
2. Extract code module-by-module, keeping voice_input_gui.py working at each step
3. Replace voice_input_gui.py internals with imports from new modules
4. Once all code is extracted, voice_input_gui.py becomes a thin launcher
5. Test end-to-end after each extraction step

## What Does NOT Change

- User-facing behavior (all 70+ commands still work)
- Settings file format and location
- Hotword daemon (separate process, already clean)
- Screen capture script (already separate)
- TTS module (already separate)
- Voice reference files and voiceprint data
