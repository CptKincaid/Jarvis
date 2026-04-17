# Linux HUD Port — Design

**Date:** 2026-04-16
**Status:** Approved
**Scope:** Port the Jarvis-Win modular architecture (pywebview HUD, `config.py`, commander pattern, 15-file split) onto the Linux repo while keeping the Linux-side AI stack (Parakeet STT, TitaNet speaker verification, Kokoro/F5 TTS, intent classifier, dual-threshold wake word).

## Context

Two sibling repos:

- **`CptKincaid/Jarvis`** (this repo, Linux) — monolithic 5,800-line `voice_input_gui.py` + 14 sibling modules. Newer AI: Parakeet (1.69% WER), TitaNet (0.66% EER), Kokoro 82M + F5-TTS, intent classifier with feedback learning, dual-threshold wake word.
- **`CptKincaid/Jarvis-Win`** (Windows) — 15 focused modules, centralized `config.py`, `commander.py` router, pywebview HUD with web frontend (`frontend/index.html` + `css/hud.css` + `js/app.js`, `js/reactor.js`). Older AI: Whisper, ECAPA-TDNN, Edge/XTTS.

The Win repo is strictly better architecturally. The Linux repo is strictly better AI-wise. This spec merges the two: Win's architecture + Linux's models.

## Goals

1. Replace the 5,800-line Tkinter god class with a pywebview HUD plus 15 focused modules, matching Jarvis-Win's layout.
2. Keep every Linux AI subsystem unchanged — no behavior regression in STT, speaker verification, TTS, hotword, or intent classification.
3. Keep Tkinter runnable as a `--legacy` fallback indefinitely (user decision: B).
4. Centralize every hardcoded path in `config.py` using XDG-compatible Linux conventions (`~/.config/Jarvis/`, `/tmp/jarvis/`).
5. Extend the existing pytest suite (86 passing tests today) and keep it green after every commit.

## Non-Goals

- No AI model changes (Parakeet/TitaNet/Kokoro/F5 all stay).
- No new voice commands or user-facing features.
- No full migration of `_check_quick_command` elif chain to `CommandDispatcher.try_handle()` — that's deferred from the earlier remediation plan and remains deferred here.
- No Claude Code tool-use integration ("extension of me" coding workflows) — that's the next project.
- No security boundary changes (shell allowlist, HMAC queue auth stay as the `main`-reverted stubs until the user opts back in).

## Final File Layout

```
jarvis/
  app.py                  # NEW — pywebview launcher + JarvisAPI bridge
  config.py               # NEW — centralized paths, VRAM detect, feature flags
  commander.py            # NEW — intent router wrapping dispatcher.py
  desktop.py              # NEW — xdotool/xprop/wmctrl abstraction
  hotword.py              # NEW — HotwordListener extracted from voice_input_gui.py
  recording.py            # EXISTS — Phase 2 module, already on branch
  transcription.py        # EXISTS — Phase 2 module
  dispatcher.py           # EXISTS — wrapped by commander.py
  animation.py            # EXISTS — Phase 2 module; powers Canvas2D reactor
  jarvis_logging.py       # EXISTS — shared logger
  jarvis_tts.py           # UNCHANGED — Kokoro/F5/Edge/XTTS
  stt_engine.py           # UNCHANGED — Parakeet/Whisper
  speaker_verification.py # UNCHANGED — TitaNet
  speak_queue_auth.py     # EXISTS — kept for future re-enable
  jarvis_speak_queue.py   # EXISTS — currently plain file-based IPC (unchanged)
  jarvis_brain.py         # UNCHANGED — Ollama + Claude backends
  jarvis_agent.py         # UNCHANGED — proactive intelligence
  memory.py               # UNCHANGED
  context.py              # UNCHANGED
  voice_input_gui.py      # LEGACY — kept for `--legacy` fallback

frontend/
  index.html              # NEW — HUD markup (port from Jarvis-Win)
  css/hud.css             # NEW — glassmorphism styles
  js/app.js               # NEW — pywebview bridge polling loop
  js/reactor.js           # NEW — Canvas2D arc-reactor

tests/                    # EXISTS — all current tests kept green
  + tests for new modules (config, commander, desktop, hotword, app/JarvisAPI)

docs/superpowers/
  specs/                  # this file lives here
  plans/                  # implementation plans

TODO.md                   # NEW — structured bug + backlog tracking
CLAUDE.md                 # UPDATED — new architecture reflected
```

## Module Contracts

Each new module is independently testable and has a narrow interface.

### `jarvis/config.py`

- **Exports:** path constants (`APPDATA_DIR`, `TEMP_DIR`, `SETTINGS_FILE`, `MEMORY_DIR`, `VOICEPRINT_FILE`, `VOICE_REF`, `LOG_FILE`, `SPEAK_QUEUE_FILE`, `SCREEN_DIR`), `GPU_MODE`, `VRAM_GB`, `WHISPER_MODEL`, `ENABLE_XTTS`, `ENABLE_SPEAKER_VERIFY`, `CLAUDE_BIN`, `OLLAMA_URL`, `OLLAMA_MODEL`, `log()`.
- **Depends on:** `os`, `pathlib`, `subprocess` (for nvidia-smi).
- **Linux path conventions:**
  - `APPDATA_DIR` = `~/.aiws_trainer/` (preserves existing data — must not break current enrollment, habits, memory)
  - `TEMP_DIR` = `/tmp/vss_voice/` (preserves existing log aggregation)
  - Paths are resolved via `Path.home()` and `Path("/tmp")` — no hardcoded `/home/hunterp`.
- **VRAM detection:** shell `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`, returns MiB; converts to GiB.
- **GPU mode tiers:**
  - `"24gb"` (≥ 22 GiB) — dual-3090 machine; all features on, Kokoro + F5 + XTTS available
  - `"16gb"` (≥ 14 GiB) — XTTS + speaker verify on, F5 off
  - `"8gb"` (≥ 6 GiB) — speaker verify on, XTTS/F5 off, Whisper base
  - `"cpu"` (< 6 GiB) — minimal features

### `jarvis/commander.py`

- **Exports:** `Commander` class.
- **Wraps:** existing `dispatcher.CommandDispatcher`.
- **Adds:** dictation-mode toggle ("start dictation" / "stop dictation" phrases), routing hook for unhandled text to `brain.think()`.
- **Interface:**
  ```python
  cmd = Commander(desktop=..., brain=..., tts=..., memory=..., agent=...)
  handled = cmd.process(text)  # returns bool
  ```
- **Dictation mode:** when enabled, every transcribed utterance is typed into the active window via `desktop.type_text(auto_enter=False)` instead of being routed to brain.

### `jarvis/desktop.py`

- **Exports:** `DesktopController` class.
- **Methods:**
  - `get_active_window_title() -> str` (via `xdotool getactivewindow getwindowname`)
  - `list_windows() -> list[(wid, title)]` (single `wmctrl -l`, falls back to xdotool)
  - `switch_to_window(name: str) -> bool` (matches by substring)
  - `type_text(text: str, auto_enter: bool = False)` (via `xdotool type`)
  - `send_key(key: str)` (via `xdotool key`)
  - `copy_to_clipboard(text: str)` (via `xclip` or `xsel`)
- **Replaces:** all scattered subprocess calls in `voice_input_gui.py` for xdotool/wmctrl/xclip.
- **Fallbacks:** when wmctrl missing, xdotool path is used (same dual-path my earlier fix installed).

### `jarvis/hotword.py`

- **Exports:** `HotwordListener` class.
- **Source:** existing `HotwordListener` class inside `voice_input_gui.py:785-1020`. Move verbatim, swap the `gui` coupling for explicit callbacks:
  ```python
  hl = HotwordListener(
      mic_idx=..., sample_rate=...,
      on_detected=callable,          # fires on wake word
      is_recording=callable,         # returns bool, pauses detection
      is_tts_speaking=callable,      # returns bool, prevents feedback
      model_dir=Path,                # for custom verifier
  )
  hl.start(); hl.pause(); hl.resume(); hl.stop()
  ```
- **Preserves:** dual-threshold confirmation logic from Phase 5 wake-word hardening.

### `jarvis/app.py`

- **Exports:** `main()` entry point; `JarvisAPI` class.
- **Launches:** frameless pywebview window pointing at `frontend/index.html`.
- **`JarvisAPI` bridges:**
  - `init_modules()` — lazy-loads all backend modules
  - `start_recording()` / `stop_recording()` — drives `RecordingController`
  - `get_status()` — returns `{status, status_color, audio_level, user_text, jarvis_text, gpu_info}` for JS polling (200ms interval)
  - `get_amplitude()` — real-time audio level for reactor
  - `toggle_setting(name, value)` — persists to `config.SETTINGS_FILE`
  - `close_window()` — runs cleanup
- **Legacy flag:** `python -m jarvis.app --legacy` execs `python -m jarvis.voice_input_gui` instead.

### `frontend/index.html` + CSS + JS

- **Port verbatim** from Jarvis-Win (minimal adaptation needed — web is web).
- **Polling loop:** `app.js` calls `pywebview.api.get_status()` every 200 ms, updates DOM.
- **Reactor animation:** `reactor.js` runs Canvas2D draw loop synced to amplitude from `api.get_amplitude()`.
- **Settings panel:** standard checkboxes driving `api.toggle_setting()`.

## Phased Approach

### Phase 1 — Foundations (config + commander + hotword + desktop)

Safe, pure backend extractions. Each new module gets tests; no GUI changes yet. `voice_input_gui.py` is updated to import from the new modules but its runtime behavior is unchanged.

- **P1.1** `config.py` + tests
- **P1.2** `desktop.py` + tests
- **P1.3** `hotword.py` + tests (extract `HotwordListener`)
- **P1.4** `commander.py` + tests (wraps `dispatcher.py`)
- **P1.5** Update `voice_input_gui.py` to delegate to the new modules — no user-visible change

**Phase 1 gate:** all 86 current tests + new tests green. Tkinter GUI still fully functional.

### Phase 2 — HUD (app.py + frontend/)

Everything up to here hasn't shipped a new UI. This phase does.

- **P2.1** Port Jarvis-Win's `frontend/index.html`, `css/hud.css`, `js/app.js`, `js/reactor.js` verbatim
- **P2.2** Build `jarvis/app.py` with `JarvisAPI` — wire to backend modules
- **P2.3** Add `python -m jarvis.app --legacy` fallback flag
- **P2.4** Manual parity test: record/stop/transcribe/speak all work in HUD

**Phase 2 gate:** HUD completes one full voice round-trip; Tkinter path still works.

### Phase 3 — Housekeeping

- **P3.1** `TODO.md` with bug tracking template
- **P3.2** Update `CLAUDE.md` to describe new architecture
- **P3.3** `docs/superpowers/plans/` index

## Testing Strategy

- **Unit tests** for every new module under `tests/`:
  - `test_config.py` — path resolution, VRAM detection, feature flag derivation
  - `test_commander.py` — dictation toggle, dispatch fallthrough, brain routing
  - `test_desktop.py` — mocked subprocess calls; verify commands issued
  - `test_hotword.py` — mocked OpenWakeWord; verify dual-threshold logic
  - `test_jarvis_api.py` — `JarvisAPI` methods with mocked backend
- **Integration test:** `test_voice_roundtrip.py` — synthetic audio → transcription → commander → speak-queue; uses recorded WAV and mocked TTS.
- **No Selenium / browser automation** for the HUD — manual smoke only. Reasonable tradeoff: the frontend is 3 small files that can be read top-to-bottom.

## Branching and Rollback

- Branch: `linux-hud` off the current `jarvis-review-fixes` HEAD.
- One commit per phase sub-step; easy per-commit revert.
- Phase 1 can merge independently if Phase 2 slips — no user-visible change, all backend improvements.
- **Before merging to `main`:**
  - Push current `jarvis-review-fixes` to GitHub (safety)
  - Run full `pytest tests/`
  - Manual smoke: HUD voice round-trip; legacy Tkinter voice round-trip
  - CLAUDE.md + TODO.md updated

## Explicit Out-of-Scope Items

- pywebview-to-Electron migration (pywebview is sufficient on Linux).
- Real-time reactor animation via WebSocket (200ms polling is fine for MVP).
- Mobile/remote HUD rendering.
- `desktop.py` parity with Jarvis-Win's `pyautogui`/`pywinauto` APIs (we use xdotool — sufficient for current feature set).
- Restoring the shell allowlist, HMAC queue, or any of the other Phase 1 security fixes that were reverted during the bug hunt — those are their own re-enable decision.
- Any Phase 2 items from the earlier `jarvis-review-fixes` plan that were not yet shipped (partial-transcribe fix, atomic writes, `get_logger` migration are kept; full `_check_quick_command` → registry migration stays deferred).
