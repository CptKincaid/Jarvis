# Jarvis Code Review Remediation — Design

**Date:** 2026-04-16
**Status:** Approved
**Scope:** Fix 21 issues found in a comprehensive review of `jarvis/voice_input_gui.py` and sibling modules.

## Context

A full code review of the Jarvis voice assistant (primarily the 5,800-line `voice_input_gui.py`) surfaced 21 issues across four tiers:

- **Critical (5):** shell-command injection via transcribed speech; unauthenticated TTS queue; duplicate orbit shutdown; dead STT path in partial preview; hotword pause race.
- **High (5):** 5,800-line god class; 350-line if/elif dispatcher; dead/duplicate intent patterns; denoise on the audio callback thread; N+1 xdotool subprocesses.
- **Medium (7):** blocking `time.sleep` in hotword loop; unconditional TTS instantiation on every poll tick; `_stream_partial` reschedules when disabled; silent intent-log corruption; dead orbit frame renderer; uncached Claude terminal WID; SIGTERM path leaks resources.
- **Low (4):** sample-by-sample beep generation; hard-coded `/home/hunterp/` paths; non-atomic settings write; five copies of `_log()`.

This spec decomposes the work into three sequential phases.

## Goals

1. Remove the two security issues (shell injection, TTS queue trust) before any further Jarvis use.
2. Reduce `voice_input_gui.py` from 5,800 lines to a thin coordinator (<1,500 lines) by extracting four focused modules.
3. Add unit test coverage for the extracted modules (currently zero on GUI/TTS code).
4. Eliminate all 21 review findings without introducing new features or behavior changes.

## Non-Goals

- No changes to TTS engine selection, wake word detection thresholds, or speaker verification accuracy.
- No removal of the orbit HTTP server code (already marked unused).
- No new user-facing features.

## Phased Approach

### Phase 1 — Critical bugs + security (items 1-5)

Surgical, in-place fixes. No module moves, no new test scaffolding. One commit per item so each is individually revertible.

| # | File:Line | Change |
|---|---|---|
| 1 | `voice_input_gui.py:3514` | Add `SHELL_ALLOWLIST = {"ls","pwd","git","df","free","uptime","date","whoami","hostname","wc","cat","head","tail"}`. Parse `shell_cmd` first token; if ∈ allowlist, run immediately. Else TTS-speak `"Confirm: run <cmd>?"`, record 3 seconds of mic audio, transcribe via `STTEngine.transcribe()`, accept only if the transcript contains `"yes"` or `"confirm"` (case-insensitive); otherwise refuse. |
| 2 | `voice_input_gui.py:4889`, `jarvis_speak_queue.py` | `os.makedirs("/tmp/vss_voice", mode=0o700, exist_ok=True)` + `os.chmod` on every startup. Generate `~/.aiws_trainer/speak_queue.key` (32 random bytes, mode 0600) if missing. All queue writers prepend `hmac_sha256(key, line).hexdigest()[:16] + " "`. Watcher verifies HMAC and drops unsigned/mismatched lines. |
| 3 | `voice_input_gui.py:5757-5761` | Delete duplicate `_orbit_server.shutdown()` block (4 lines). |
| 4 | `voice_input_gui.py:2881` | Replace `self._whisper_model.transcribe(audio, **kwargs)` with `self._stt_engine.transcribe(audio)` (returns `STTResult`; use `.text`). Guard entry with `if not self._stt_engine or not self._stt_engine.is_loaded: return` so the worker no-ops safely when STT hasn't finished loading yet. |
| 5 | `voice_input_gui.py:815-838` (`HotwordListener`) | Add `self._stream_lock = threading.Lock()` to `HotwordListener.__init__`. Wrap `_open_stream()`, `_close_stream()`, and the loop's stream-reopen block (line 911-920) in `with self._stream_lock:`. Keeps the existing synchronous `pause()` call (needed to prevent TTS feedback) but serializes the stream mutations so `pause()` cannot race with `resume()` or the loop's reopen. |

**Phase 1 verification (manual smoke):**
- Say "run ls" → allowlisted, runs.
- Say "run rm -rf /" → blocked, TTS prompts for confirmation, times out without match → refused.
- Say "run rm -rf /" then "yes" within 5s → runs (documented risk the user accepted).
- Restart; confirm talkback still works and partial preview still shows text.
- External process writes unsigned line to queue → ignored.

### Phase 2 — Extraction + tests (items 6-7)

Split `VoiceInputGUI` into four focused modules. The GUI class remains as a thin coordinator responsible for Tk widgets, event wiring, and holding references to the four subsystems.

| New module | Responsibility | Current-file sources |
|---|---|---|
| `jarvis/recording.py` (`RecordingController`) | sounddevice stream lifecycle, audio callback, silence detection, noise gate, silence-reset logic | `audio_callback`, `_stream`, `_audio_frames`, silence timer, noise gate |
| `jarvis/transcription.py` (`TranscriptionPipeline`) | speaker filter → STT → `IntentClassifier` → cleanup; owns partial-transcribe worker | `_stt_engine`, `_speaker_verifier`, `IntentClassifier`, `_partial_transcribe_worker` |
| `jarvis/dispatcher.py` (`CommandDispatcher`) | routes transcribed text to quick commands, desktop actions, brain. Replaces the 350-line `_check_quick_command` elif chain with `COMMAND_REGISTRY: list[tuple[re.Pattern, Callable]]`. | all `_check_quick_command` branches, `QUICK_COMMANDS`, desktop helpers like `_find_claude_terminal`, `_get_window_list` |
| `jarvis/animation.py` (`AnimationRenderer`) | orbit frame generation, waveform, amplitude feeder | `_render_orbit_fast`, waveform canvas updates, amplitude polling |

**Coordinator shape after extraction:**

```python
class VoiceInputGUI:
    def __init__(self, root):
        # Tk widgets setup...
        self.recorder = RecordingController(
            sample_rate=SAMPLE_RATE,
            on_amplitude=self._on_amplitude,
            on_stopped=self._on_recording_stopped,
        )
        self.pipeline = TranscriptionPipeline(
            gpu=DEFAULT_GPU,
            speaker_verifier=self._speaker_verifier,
        )
        self.dispatcher = CommandDispatcher(
            agent=self._agent,
            tts_provider=self._get_tts,
            brain=self._brain,
        )
        self.animation = AnimationRenderer(canvas=self.canvas)
```

**Data flow:**
`RecordingController.start()` → audio chunks → buffered → `on_stopped` callback → `TranscriptionPipeline.transcribe(audio)` → text → `CommandDispatcher.handle(text)` → side effects (TTS, typing, brain call). `AnimationRenderer` subscribes to `RecordingController.on_amplitude` for reactor-glow updates.

**Test coverage added during extraction:**

| Test file | Coverage target |
|---|---|
| `tests/test_recording.py` | Synthetic numpy audio exercises silence detection thresholds, noise gate, callback fires the expected number of times. |
| `tests/test_transcription.py` | Mock `_stt_engine` + mock speaker verifier; assert (a) non-authorized speaker → empty result, (b) authorized speaker + known-good audio → expected text, (c) intent classifier integrated. |
| `tests/test_dispatcher.py` | `COMMAND_REGISTRY` dispatch: each known command pattern routes to the correct mock handler; unknown text falls through to `brain.handle`. |
| `tests/test_animation.py` | Amplitude input → orbit frame numpy array shape/dtype invariants; beep generation produces correct sample count and peak amplitude. |

Each new module ≤500 lines; each test file ≥3 passing tests.

**Phase 2 verification:** `pytest tests/` all green; manual smoke of full record → transcribe → dispatch flow through the new modules.

### Phase 3 — Remaining fixes (items 8-21)

All items are small and land inside the new module boundaries established in Phase 2. Grouped by the module they land in.

**`recording.py`:**
- #9 Move `denoise_audio` out of the sounddevice callback into a post-recording step in `RecordingController.stop()`.
- #11 Replace `time.sleep(1.5)` in hotword loop with a `time.monotonic()` cooldown timestamp (matches the pattern already at line 920).
- #13 Don't reschedule `_stream_partial` when `streaming_var.get()` is False.
- #17 Add a SIGTERM handler that runs `_cleanup()` even when the GUI is minimized to tray.

**`transcription.py`:**
- #14 `IntentClassifier._load_log` logs `f"Intent log corrupt: {e}"` and renames the corrupt file to `intent_log.json.corrupt.<timestamp>` before resetting.

**`dispatcher.py`:**
- #10 Replace N+1 `xdotool` subprocesses in `_get_window_list` with a single `wmctrl -l` invocation.
- #16 Cache Claude terminal WID with 5-second TTL in `_find_claude_terminal`.
- #19 Replace hard-coded `/home/hunterp/` strings with `Path.home()`.

**`animation.py`:**
- #15 Delete unreachable `_render_orbit_frame` (~60 lines).
- #18 Vectorize `_generate_beep` using `numpy.sin` instead of a Python `for` loop over `struct.pack`.

**`voice_input_gui.py` coordinator:**
- #12 `_get_tts()` guards on `talkback_var.get()` before instantiating; tracks last-used engine name and only recreates `JarvisTTS` when the name changes.
- #20 `_save_settings` writes to `SETTINGS_FILE.with_suffix(".tmp")` then `os.replace()` for atomicity.

**New `jarvis/logging.py`:**
- #21 Centralized `_log()` with shared file handler; all modules import from it.
- #8 Remove the unreferenced module-level `_ASSISTANT_PATTERNS` and `_CASUAL_PATTERNS` lists in `voice_input_gui.py`.

**Phase 3 verification:** `pytest tests/` all green; manual smoke of full flow.

## Branching & Rollback

- Single branch: `jarvis-review-fixes`.
- Commits grouped per phase; each item in Phase 1 and Phase 3 is its own commit for clean revert.
- Phase 2 extraction lands as 4-5 commits (one per extracted module + one for the coordinator rewrite + one for the test files).
- If a phase regresses after merge, `git revert <merge-commit>` rolls back that phase only.
- Branch only merges to `main` after all three phases pass verification.

## Explicit Out-of-Scope

- TTS engine behavior changes (Kokoro/F5/XTTS selection stays as-is).
- Wake word detection threshold tuning.
- Speaker verification accuracy work.
- Orbit HTTP server code (unused but left alone).
- New user-facing features.
