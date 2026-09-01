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

**Ollama runs ONE model at a time.** `OLLAMA_MAX_LOADED_MODELS=1` in
`/etc/systemd/system/ollama.service.d/10-residency.conf` — the guard added
after the 2026-08-28 unified-memory power-off. So asking for a SECOND model
does not add one, it EVICTS the chat model, and gemma4:26b then costs ~7 s to
reload. Nothing logs an "unload", which makes this invisible from Jarvis's
side. Two real bugs came from it in one day: memory recall embedding every
query on the reply path, and the test suite firing 24 `/api/embed` calls per
run straight at the live server. **The reply path may not ask Ollama for any
model other than the chat model**, and `tests/conftest.py` now refuses
connections to port 11434 outright.

**The test suite runs against the LIVE box, so conftest is a firewall.** It
already redirects every writable path (log dir, memory dir, assistant config,
cache, intent log, docs index) and blocks the room controls, because a scene
test would dim the screen he is reading. Two more were added on 2026-08-31
after both reached him: audio out (four tests spawned a real `paplay` into his
speakers — `earcons.play` resolves its runner at CALL time so the firewall can
replace it) and Ollama (above). **Anything that can reach hardware, the
network or his desktop belongs in that fixture before it belongs in a test.**

**A hand edit to `assistant.json` does NOTHING until Jarvis restarts.**
`AssistantConfig.reload_if_changed()` exists and has no callers, so the
running process keeps its boot-time copy. Change a setting, restart, verify —
in that order, or you will debug a setting that was never applied.

## State lives outside the repo
```
~/.aiws_trainer/voice_settings.json   engine, thresholds, speaker_verify
~/.aiws_trainer/voiceprint.npz        ECAPA embeddings (scripts/enroll_voice.py)
~/.aiws_trainer/intent_log.json       intent classifier feedback
~/.config/jarvis/assistant.json       user, location, calendar/mail/spotify creds
/tmp/vss_voice/jarvis.log             the log worth reading first
/tmp/vss_voice/speak_queue.txt        write a line here to make Jarvis speak
~/.aiws_trainer/jarvis_memory/        timekeeper.db, notes.db, dossier/headsup state
~/.config/autostart/jarvis.desktop    starts Jarvis 15 s after login
```
`/tmp` is WIPED AT BOOT on this box (a tmpfiles `D /tmp` rule), so everything
under `/tmp/vss_voice` is scratch: the log, the pid file, the sockets and the
speak queue are all recreated. Nothing that must survive a reboot may live
there. What DOES survive: the autostart entry above plus the user units
`jarvis-f5`, `jarvis-spotify`, `tailscaled` and `haymaker-digest.timer`, all
enabled with lingering on, and `ollama` as a system service.

## Conventions
- Tests use the real modules; only hardware is stubbed (see
  `tests/test_app_wiring.py`). No Tk in unit tests.
- Ports from the old monolith carry the original line numbers in comments —
  keep them, they are the audit trail.
- Comments explain WHY, especially where a fix looks arbitrary.
