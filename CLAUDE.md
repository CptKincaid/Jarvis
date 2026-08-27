# CLAUDE.md — Jarvis AI Voice Assistant

## Project Overview
Personal voice assistant: wake word → speaker-verified capture → Whisper →
intent routing → local LLM, Claude Code sessions, or a tool. Desktop control,
XTTS voice cloning, timers/alarms, calendar, mail, Spotify, briefings.

## Stack
Python 3.12 (aarch64) | NVIDIA GB10, single GPU (DGX Spark) | Tkinter UI |
openWakeWord 0.4.0 | Whisper | SpeechBrain ECAPA-TDNN | edge-tts / XTTS v2

## Running
```bash
python -m jarvis.app                 # the app (needs DISPLAY, uses :1 here)
python -m pytest -q                  # 1506 tests, ~75 s
ruff check jarvis/ scripts/ tests/
```
Always the venv: `~/vss_env/bin/python`. Never system python — PIL/ImageTk,
torch and speechbrain all live in the venv.

## Architecture (V3)
~31k lines across 32 top-level modules plus `tools/`, `ui/`, `channels/`,
communicating over an event bus (`jarvis/events.py`). Nothing calls the UI
directly; modules publish events and the window subscribes.

```
app.py            wiring, audio pipeline, services   commander.py  intent + routing
recorder.py       mic arbiter, capture, VAD          router.py     local/claude/ask
hotword.py        openWakeWord + speaker gate        brain.py      local LLM (ollama)
transcriber.py    whisper                            speaker.py    ECAPA verification
tts.py            edge/XTTS + playback               claude_session.py  tmux sessions
events.py         the bus + event dataclasses        approvals.py  MCP permission broker
ui/               borderless Tk console              tools/        weather, calendar, notes,
channels/         discord                                          mail, spotify, timekeeper
```

**Voice path:** `hotword` → `recorder` → `RecordingStopped` →
`app._process_audio` → `Transcribed` + `UserUtterance` → `commander.handle`.
Speech happens ONLY through `JarvisApp._say` (talkback-gated); `JarvisReply`
events are display-only.

## Things that will bite you

**The mic arbiter is the single owner of the microphone.** Every consumer —
recording, calibration, enrolment, wake-word training, and TTS talk-back —
wraps its use in `arbiter.acquire(owner)`, which pauses the hotword. It is a
re-entrant depth counter, NOT a mutex: nesting is fine, but it will not stop
two consumers using the mic at once. TTS holds it for a whole spoken burst, so
Jarvis cannot hear himself; block on the TTS before recording after speaking.

**Speaker verification gates two layers, and they fail in opposite
directions.** The wake-word gate (`hotword._speaker_ok`) fails OPEN — an
unwakeable assistant is worse than an over-eager one. The transcript gate
(`app._process_audio` → `speaker.filter_segments`) fails SHUT once a
voiceprint exists, because silently accepting every voice is how a television
reached the commander. Nothing enrolled → both fail open, so a fresh box is
never mute.

**`speaker.load()` must be called or both gates are dead.** `__init__` does
not self-load. A real app start logs `voiceprint loaded: N samples`; if that
line is missing the feature is off no matter what the settings say.

**`MACHINE.has_mic` cannot trust sounddevice alone.** While PipeWire holds a
USB mic, PortAudio cannot probe `hw:N` and lists only `pipewire`/`default` —
and with NO mic those same virtual devices still appear at 44100 Hz carrying
silence. `/proc/asound/pcm` is the tiebreak. Check `arecord -l`, not
sounddevice, when diagnosing.

**One utterance at a time.** `recorder.recording` is already False during the
~20 s transcription pass; `app._audio_busy` is the guard that matters.

**The intent classifier only runs on `source == "voice"` without a "jarvis"
prefix.** A false NO is silent and unrecoverable — only UNCERTAIN prompts, so
`log_feedback` never learns from a wrongly-dropped command. When measuring it,
point `IntentClassifier.INTENT_LOG` at a temp file; the real log makes results
look better than a clean install.

**Whisper backend differs by device.** GPU takes the openai-whisper path;
`vad_filter` exists only on the CPU faster-whisper branch, so there is no VAD
on the GPU path. ctranslate2 has no aarch64 CUDA wheel, hence the fallback.

## State lives outside the repo
```
~/.aiws_trainer/voice_settings.json   engine, thresholds, speaker_verify
~/.aiws_trainer/voiceprint.npz        ECAPA embeddings (scripts/enroll_voice.py)
~/.aiws_trainer/intent_log.json       intent classifier feedback
~/.config/jarvis/assistant.json       user, location, calendar/mail/spotify creds
/tmp/vss_voice/jarvis.log             the log worth reading first
/tmp/vss_voice/speak_queue.txt        write a line here to make Jarvis speak
```

## Conventions
- Tests use the real modules; only hardware is stubbed (see
  `tests/test_app_wiring.py`). No Tk in unit tests.
- Ports from the old monolith carry the original line numbers in comments —
  keep them, they are the audit trail.
- Comments explain WHY, especially where a fix looks arbitrary.
