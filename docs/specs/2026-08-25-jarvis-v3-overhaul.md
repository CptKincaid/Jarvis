# Jarvis V3 — Full Overhaul

**Date:** 2026-08-25
**Supersedes:** `2026-04-11-jarvis-v2-architecture-design.md` (keeps its goals — tiered intelligence, context, memory, reliability — drops its stale machine assumptions and its extract-in-place migration)
**Machine reality:** DGX Spark (GB10, aarch64, single GPU, unified memory — `nvidia-smi` memory reads N/A, use `/proc/meminfo`). Whisper is **CPU int8 only** (no aarch64 CUDA ctranslate2). TTS default **edge**. openwakeword **pinned 0.4.0** with a local custom-verifier patch that must survive verbatim. `claude` binary resolves from PATH (`~/.local/bin/claude`), NOT `~/.npm-global`. **The machine may have no microphone** — the app must be fully usable with text input alone.

## Principles

1. **Event-driven core.** Modules publish events on a bus; the UI subscribes. No module outside `jarvis/ui/` may import tkinter or touch a widget. The bus marshals onto the Tk main thread internally.
2. **Single owner per resource.** The mic has one arbiter. The Whisper model lives in one class with its own lock. Settings live in one Config object. One log setup. One memory store.
3. **Port, don't rewrite, audio-critical code.** Silence detection, resampling, hotword prediction, speaker-verify math move with their exact constants and order of operations (line refs below). There is no microphone on this machine to re-verify with.
4. **Config is the source of truth.** Tk variables in the UI are *views* of Config, not the store. `~/.aiws_trainer/voice_settings.json` stays the file (back-compat: read old keys, preserve unknown keys on save).
5. **Honest failure.** No new blanket `except: pass`. Errors log with context and surface as status events. Fail-open paths (speaker verify) log loudly.
6. **Commands are data.** The ~35-branch dispatcher becomes an ordered registry. Handlers return a result object; they never touch widgets.

## Package layout

```
jarvis/
  app.py            # entry point + wiring (WP2, do not create in WP1)
  config.py         # [FOUNDATION, exists] Config dataclass, Paths, MachineProfile
  events.py         # [FOUNDATION, exists] Bus, event dataclasses, Tk marshalling
  logs.py           # [FOUNDATION, exists] get_logger() — rotating file @ /tmp/vss_voice/jarvis.log
  recorder.py       # mic streams, silence auto-stop, calibration, beeps, MicArbiter
  transcriber.py    # Whisper mgmt (CPU), full + partial transcription, vocab
  hotword.py        # OpenWakeWord listener (0.4.0 patch preserved verbatim)
  tts.py            # dual-engine TTS: timeout, internal queue, amplitude events
  speak_queue.py    # in-process say() + external-file watcher (evolves jarvis_speak_queue)
  speaker.py        # SpeakerVerifier (atomic save, no GUI import) — evolves speaker_verification.py
  commander.py      # IntentClassifier + command registry + routing (CommandResult)
  brain.py          # tiers T2/T3, resolved claude bin, cancellation — evolves jarvis_brain.py
  context.py        # context engine, absorbs jarvis_agent probes — evolves in place
  memory.py         # single persistent store, absorbs agent memory — evolves in place
  desktop.py        # xdotool: windows, tabs, typing, claude-terminal, screenshots
  workflows.py      # deploy/morning/training-check + persisted reminders
  ui/
    __init__.py     # [FOUNDATION, exists]
    theme.py        # [FOUNDATION, exists] design tokens + font resolution
    widgets.py      # canvas-drawn flat widgets: RoundButton, Card, Toggle, Meter, Tooltip
    reactor.py      # the new reactor visualization (state machine: idle/listen/think/speak)
    views.py        # TranscriptView (chat), CommandBar, SettingsDrawer, StatusStrip
    main_window.py  # window assembly, tray, global hotkeys, event subscriptions
  voice_input_gui.py  # LEGACY 5612-line monolith — READ-ONLY source for ports during WP1;
                      # becomes a 3-line shim to app.main() in WP2. Do NOT edit in WP1.
scripts/
  hotword_daemon.py   # fix path math, reuse jarvis.speaker, CPU whisper default
  screen_capture.py   # fix unguarded log dir
tests/                # pytest; pure-logic only (no audio hardware, no X11, no models)
```

**Deleted in WP2** (do not port): `orbit_server.py`, `orbit_animation.html`, `jarvis_agent.py` (absorbed — see context/memory/brain), `jarvis_tts.py`/`jarvis_speak_queue.py`/`speaker_verification.py` (renamed/evolved), monolith dead code (`_render_orbit_frame` 4181-4282, dup cleanup 5569-5573, `_ASSISTANT_PATTERNS`/`_CASUAL_PATTERNS` 132-160, legacy `HOTWORD_PHRASES` consts 64-75).

## Threading model

- Tk main thread runs the UI only.
- Worker threads (audio callback, transcribe, brain, tts, hotword, watchers) never touch widgets or Tk variables. They call `bus.publish(Event(...))` — thread-safe; the bus queues and drains on the Tk thread via `root.after` (see `events.py`).
- Reading config from worker threads is safe (plain dataclass attrs). Writing config happens on the Tk thread (UI) or via `config.update(**kw)` which is lock-guarded.
- Blocking subprocesses get timeouts. Brain calls are cancellable (`brain.cancel()` kills the subprocess).

## Event catalog (events.py — already written; consult the file)

`AudioLevel(level, waveform)` ~12Hz during recording · `RecordingStarted/RecordingStopped(reason)` · `PartialText(text)` · `Transcribed(text, confidence, speaker_score, accepted, reject_reason)` · `HotwordDetected(score)` · `MicState(available, device_name)` · `Status(text, kind)` kind∈{ok,info,busy,warn,error} · `UserUtterance(text, source)` source∈{voice,typed} · `JarvisReply(text, speak)` · `BrainState(state)` state∈{idle,thinking} · `SpeakingState(active, amplitude)` amplitude streamed ~12Hz while speaking · `ReminderFired(text)` · `AppQuit()`

## Module contracts

Every module gets: `from jarvis.config import CONFIG, PATHS, MACHINE` · `from jarvis.events import bus, <events>` · `from jarvis.logs import get_logger`. Port-from references are `voice_input_gui.py` line numbers (the file mapping digest at `/tmp/claude-1000/-home-hunterp/e9a1b620-9b7f-45d3-b67c-6ecfba3bf4c3/tasks/wlvvyfeuc.output` has the full anatomy).

### recorder.py
```python
class MicArbiter:            # THE single mic owner
    def acquire(self, owner: str) -> AbstractContextManager  # pauses hotword, resumes on exit
    def register_hotword(self, pause_cb, resume_cb)
class Recorder:
    def __init__(self, arbiter)
    def start(self)          # begins a recording session (publishes RecordingStarted)
    def stop(self, reason="manual") -> np.ndarray | None   # 16k mono float32; publishes RecordingStopped
    def abort(self)
    def record_fixed(self, seconds: float) -> np.ndarray   # for enrollment/wake-training (port 5015-5054)
    def calibrate_noise(self) -> float                     # port 2262-2340
    @property def recording(self) -> bool
def play_beep(kind)          # port beep synth/playback 526-573 (module-owned temp wavs)
```
Port verbatim: audio callback with silence detection 2403-2434 (cache `noise_gate` and thresholds BEFORE the callback closure — fixes the off-thread Tk read), silence/speaker-silence auto-stop logic 4301-4434 (runs on recorder's own poll thread now, publishes stop), resample 2477-2485, 60s cap, 1s restart debounce 2360-2364, grace period. All ad-hoc `getattr` state (`_loud_chunks`, `_record_rate`, `_last_record_start`) becomes explicit `__init__` fields. Mic enumeration + **no-mic detection**: if no input devices, publish `MicState(available=False)` and make `start()` a no-op with a warn Status. Waveform/level publishing via `AudioLevel`.

### transcriber.py
```python
class Transcriber:
    def __init__(self)                       # loads vocab; model lazy
    def load(self) -> str                    # returns backend "CPU int8"; skip doomed CUDA on MACHINE.no_cuda_ct2
    def transcribe(self, audio) -> TranscribeResult   # full: segments, text, avg_logprob, language
    def partial(self, audio) -> str          # streaming preview (shares model, self._lock — port 2710-2760)
    vocab: property                          # load/save voice_vocab.txt (port 576-595)
class TranscribeResult: text, confidence, segments, language
```
Port: model load fallback (2200-2228, but go straight to CPU when `MACHINE.no_cuda_ct2`), confidence gate values (2604), initial_prompt vocab injection, LANGUAGES table 351-372. Speaker segment filtering stays OUT (commander/pipeline calls `speaker.filter_segments`).

### hotword.py
Port `HotwordListener` 785-963 → class `Hotword` with the same detection loop. **CARRY THE 0.4.0 PATCH VERBATIM (lines 841-858):** bare `Model()`, then inject joblib-loaded `~/.aiws_trainer/hey_jarvis_verifier.pkl` into `model.custom_verifier_models['hey_jarvis']`, `custom_verifier_threshold=0.3`. Constructor takes `(arbiter, get_mic_index: Callable, on_detect: Callable)` — no GUI refs. `start/stop/pause/resume`; `pause/resume` registered with MicArbiter so ALL former call sites (recording, calibration, enrollment, training, TTS talk-back — 8+ scattered sites) go through `arbiter.acquire()`. Score rule verbatim: `max(hey_jarvis, hey_mycroft*0.7) >= 0.3`, 1.5s debounce, buf.clear+model.reset on fire. Also port the wake-word verifier TRAINING flow (5178-5260) here as `train_verifier(samples) -> path` (pure logic; UI drives it).

### tts.py  (evolves jarvis_tts.py — keep engine internals, fix reliability)
```python
class TTS:
    def speak(self, text, block=False)   # enqueue; internal worker drains queue (no more silent drops)
    def stop(self)                       # stop current + clear queue
    engine: property                     # "edge" | "xtts"
```
Fixes: `asyncio.wait_for(...,15)` timeout on edge synth (193-205); `finally:` unlink of temp wav; amplitude envelope published as `SpeakingState(active=True, amplitude=…)` events ~12Hz (replaces the GUI polling `_current_amp` cross-thread); on speak start/end publish SpeakingState — main_window pauses hotword via arbiter on these events. Playback chain paplay→pw-play→aplay unchanged.

### speak_queue.py
`say(text)` — in-process: calls the app's TTS queue DIRECTLY (no /tmp detour). Keep writing `/tmp/vss_voice/speak_queue.txt` only as the EXTERNAL interface: a watcher thread tails the file for messages from other processes (VSS etc.). Fix truncation: `if size < pos: pos = 0`. Truncate the file after reading. The watcher publishes to the same TTS queue.

### speaker.py  (evolves speaker_verification.py)
Same API (`enroll_from_audio`, `verify`, `filter_segments`, `add_sample`, `is_enrolled`, threshold logic) with: **no `voice_input_gui` import** (use `logs.get_logger`); atomic save (`np.savez` to `.tmp` + `os.replace`); gpu default 0 and CPU fallback on this machine; fail-open kept but logged at WARNING with a `Status(kind=warn)` event the first time per session.

### commander.py
```python
@dataclass CommandResult: handled: bool; reply: str|None; speak: bool; status: str|None; done: bool
class Commander:
    def __init__(self, services)  # services: desktop, workflows, brain, memory, context, tts, agent-free
    def handle(self, text: str, source: str) -> CommandResult   # full routing pipeline
class IntentClassifier: ...       # port 163-345 unchanged (persisted intent_log.json path unchanged)
REGISTRY: list[Command]           # ordered; Command(name, matcher, handler, needs)
```
Port ALL tables (VOICE_COMMANDS, ACTION_COMMANDS, SCREENSHOT_PHRASES, TARGET_PATTERN, STOP/FILLER phrases, QUICK_COMMANDS, DESKTOP_ACTIONS 377-480) and `_apply_voice_commands` 483-510. Convert `_check_quick_command`'s ~35 branches (3036-3485) to REGISTRY entries **preserving order exactly** (precedence is load-bearing: 'find file' regex before 'find'; answer-question fallback before QUICK_COMMANDS table). Handlers call services and return CommandResult — zero widget access, zero inline speak-queue imports (speak via `result.speak`/`services.tts`). QUICK_COMMANDS shell strings: replace hardcoded `/home/hunterp/vss_env` with `PATHS.vss_env`. Routing order (port from `_transcribe_worker` 2559-2680 + `_on_transcription` 2820-2940): dictation → desktop → quick/registry → intent classify (uncertain → UI prompt via event) → jarvis-mode brain → fallback type-to-window. "remember X" routes to `memory.remember` (PERSISTENT — fixes the data-loss bug); "recall" to `memory.recall`.

### brain.py  (evolves jarvis_brain.py in place)
Keep: tier prompts, [SPEAK]/[RUN]/[TYPE]/[WINDOW]/[SILENT]/[DONE] protocol, Ollama /api/generate, think() worker model. Fix: `CLAUDE_BIN = os.environ.get("JARVIS_CLAUDE_BIN") or shutil.which("claude")` (kill all three ~/.npm-global sites); JARVIS_SYSTEM hardware text → "NVIDIA DGX Spark (GB10, unified memory)"; Ollama request timeout 30s + connect-refused → friendly Status; `cancel()` kills in-flight subprocess and clears `_busy`; busy-guard auto-expires (no permanent deafness); `_is_local_question` ≤5-word rule also requires no action-verb prefix (route action phrases to commander first — commander already runs before brain). ContextEngine + Memory are INJECTED (constructor args), not privately constructed. Wire `execute_autonomous` (already implemented, 128-152/266-340) to the registry: "deploy"/"autonomous:" phrases → `brain.execute_autonomous` (spec's marquee feature goes live).

### context.py  (evolves in place, absorbs jarvis_agent probes)
Fix `project_dir` → `Path(__file__).resolve().parents[1]`; GB10: GPU util from nvidia-smi utilization only, memory from `/proc/meminfo`; absorb from jarvis_agent.py the SUPERIOR versions of: screen capture to `/tmp/vss_screen/latest.png` (73-103), window tracking. Session memories: `load_recent_sessions()` from memory into the standard/full context. Delete the gathered-but-never-rendered keys mismatch (screen/processes rendered or not gathered).

### memory.py  (evolves in place — becomes the ONLY store)
Absorb jarvis_agent's habits/notes/remember-recall; one dir `~/.aiws_trainer/jarvis_memory/`; on first run migrate `~/.aiws_trainer/jarvis_data/*` into it (merge habit counts, copy notes; leave the old dir renamed `.migrated`). `save_session(summary)` called from app shutdown; single habit log (kill the double logging).

### desktop.py
Port: `_parse_desktop_action` 3586-3660 (mappings become module table), `_execute_desktop_actions` 3665-3790 (sleeps stay, runs on caller thread), window list (BATCHED: one `xdotool search` + loop `getwindowname` in one subprocess via `xargs` or wmctrl-style single call), claude-terminal heuristics `_is_claude_title` 4446 + `_find_claude_terminal` + `_type_text` 4519 + target pinning, screenshot capture (merge `_take_screenshot` + `_type_then_screenshot` into one with optional text), app launching, media keys. API: plain functions + a small `DesktopControl` class holding target-window state. No widgets; target pin changes publish Status.

### workflows.py
Port workflow definitions + runner (deploy/morning/training check 3486-3546 area) minus widget calls (Status events + CommandResult text). Reminders: persisted to `~/.aiws_trainer/jarvis_memory/reminders.json`, restored on boot, cancellable, fire → `ReminderFired` + tts.

### scripts/hotword_daemon.py (repair in place)
`VENV_PYTHON = sys.executable`… actually: derive repo root `Path(__file__).resolve().parents[1]`, GUI script `<root>/jarvis/voice_input_gui.py`, venv python from `<root>`-adjacent known path or `shutil.which('python3')` under `~/vss_env` — use `/home/hunterp/vss_env/bin/python` resolved via config default. Replace `_SpeakerChecker` with `from jarvis.speaker import SpeakerVerifier`. Whisper `device="cpu", compute_type="int8"` directly on this machine. Prune 'nervous'/'harvest' from phrase list. Keep the rest as-is (it works when paths are right).

## UI design (jarvis/ui/) — full redesign, Tkinter

**Direction:** *mission console* — keeps the JARVIS cyan-on-ink identity but executed with restraint: one accent, flat borderless surfaces, generous spacing, a reactor that communicates state, and a conversation you can read.

### Tokens (theme.py — already written; consult the file)
Ground ladder: `BG #06090f` · `SURFACE #0b1119` · `RAISED #101826` · `LINE #182234`. Ink: `INK #e8f0f8` · `MUTED #8fa3b8` · `FAINT #55677d`. Accent: `CYAN #35e0ff` · `CYAN_DIM #1899bd` · `CYAN_SOFT #0e2f42` (fills). Semantic: `OK #4ade80` · `WARN #ffb454` · `ERR #f8556d`. States: listening=CYAN, thinking=WARN(amber), speaking=OK(soft green), idle=CYAN_DIM. Font stack resolved at runtime: Inter → Liberation Sans → DejaVu Sans (theme.font(size, weight)); mono: DejaVu Sans Mono. Type scale: 26/15/13/11. Spacing: 8px grid (pad = 16 standard, 24 section).

### Window (520×880 default, min 460×720, resizable; title **"Jarvis"**)
Top→bottom:
1. **Header** (56px): wordmark "JARVIS" (26 bold cyan) left; right-aligned status cluster — colored state dot + status text (13 MUTED) + model chip ("small · CPU", 11 FAINT in RAISED pill). Text must ellipsize, never clip.
2. **Reactor stage** (flexible, min 260px): borderless canvas, BG ground. The reactor (reactor.py): layered PIL-prerendered base (concentric thin arc rings, tick marks, radial glow) + live canvas overlays. States: *idle* — slow 8s rotation, core breathing ±6%; *listening* — ring expands with `AudioLevel.level`, 64-bar circular waveform around the core from `AudioLevel.waveform`; *thinking* — amber orbital comet sweep; *speaking* — core pulses with `SpeakingState.amplitude` ripple rings. Framerate 12fps via `after(83)`; all frames composed off-thread when possible (port the pre-render pattern 3900-4180, discard `_render_orbit_frame` dead path).
3. **Conversation** (TranscriptView): vertically scrolling chat log on BG. Each utterance a flat Card (RAISED, 10px radius, 12px pad): speaker label ("YOU" 11 FAINT / "JARVIS" 11 CYAN_DIM), body 15 INK, timestamp 11 FAINT right. User cards carry a 2px left rule in CYAN_DIM whose length maps confidence. Click card → copy (toast Status). Autoscroll pinned to bottom unless user scrolled up. Replaces the fixed You/Jarvis boxes + history tab.
4. **Command bar** (64px, SURFACE): rounded entry (RAISED field, placeholder "Type a command — or say 'Jarvis'") + circular mic button (44px): idle=CYAN ring on RAISED, recording=filled CYAN with stop glyph, disabled (no mic)=FAINT ring + tooltip "No microphone detected". Enter → `UserUtterance(text, source='typed')`. Mic → recorder start/stop (identical pipeline).
5. **Status strip** (28px): left hotword state ("● wake word on" 11), center transient status messages, right CPU/GPU temp chip (from context, 30s refresh).
6. **Settings** (gear top-right): SettingsDrawer slides over from right (width 320, RAISED, scrollable), grouped: *Audio* (mic picker, noise gate+calibrate, silence timeout, sounds) · *Recognition* (model size, language, vocab editor, streaming, confidence review) · *Voice ID* (enroll, verify toggle, threshold, wake-word training) · *Speech* (talkback, engine) · *Intelligence* (jarvis mode, auto-type, smart target picker, live write) · *System* (hotword, tray, auto-enter). Custom Toggle widgets, not checkbuttons. Every control binds Config ↔ widget (trace saves stay).
7. **Tray + hotkeys:** port TrayIcon (615-689) and GlobalHotkey (695-779) into main_window.py wiring — same keys (Ctrl+Shift+V toggle, Ctrl+Shift+R PTT, F5 record — F5 binding MUST stay for hotword_daemon's synthetic keypress).

### widgets.py primitives
`RoundRect` helpers (canvas polygon smoothing), `Card`, `RoundButton(text|glyph, kind)` hover/active states, `Toggle` (animated 150ms), `Meter` (thin bar), `Chip`, `Tooltip` (port timing/behavior 968-1009, new colors), `Toast`. Keyboard focus visible (1px CYAN_DIM outline). All colors from theme tokens ONLY.

## Config schema (config.py — already written; consult the file)
Same JSON file/keys as today (`model, gpu, mic, language, auto_type, continuous, sound, review, voice_cmds, noise_gate, streaming, hotword, smart_target, auto_enter, live_write, talkback, jarvis_mode, tts_engine, speaker_verify, speaker_threshold, target_name, silence_timeout, noise_threshold`) + new: `window_geometry`. Unknown keys preserved on save. `MACHINE`: `no_cuda_ct2`, `gpu_count`, `has_mic`, `is_gb10`, `claude_bin` — detected once at boot, cheap.

## Testing (tests/, pytest, must pass with no audio/X/model deps)
- `test_commander.py`: registry precedence table-driven (the known landmines: 'find file' vs 'find', fallback order, dictation toggle), `_apply_voice_commands` punctuation/backspace, CommandResult contract, remember→memory.remember routing.
- `test_config.py`: load old-format JSON → defaults filled, unknown keys survive round-trip.
- `test_memory.py`: remember/recall/habits round-trip in tmpdir; jarvis_data migration merge.
- `test_events.py`: publish from thread → drained in order on fake main loop.
- `test_context.py`: format_for_prompt renders every gathered key (tmp git repo fixture).
- `test_intent.py`: classifier YES/NO/UNCERTAIN on the documented examples + feedback learning.

## Verification checklist (WP3)
1. `pytest` green. 2. `python -m jarvis.app` boots on :1 with no mic present → MicState(available=False), mic button disabled, typed command "what time is it" gets a reply. 3. Whisper loads CPU int8. 4. Hotword thread starts and loads the 0.4.0 verifier patch (log line). 5. TTS speaks a queued line (edge) with amplitude events. 6. Tray, hotkeys, settings drawer, vocab editor open. 7. Screenshot before/after comparison. 8. `git diff --stat` reviewed; legacy shim in place; old entry command still works.
