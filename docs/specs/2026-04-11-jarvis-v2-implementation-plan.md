# Jarvis V2 Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the 5,600-line voice_input_gui.py monolith into 10 focused modules, add a Context Engine for intelligent context injection, upgrade the brain with tiered routing, and add persistent memory — while keeping the system working at every step.

**Architecture:** Extract-and-replace migration. Each task creates a new module, moves code from voice_input_gui.py into it, replaces the old code with imports, and verifies end-to-end. The GUI becomes a thin orchestrator that wires modules together.

**Tech Stack:** Python 3.12, Tkinter, faster-whisper, SpeechBrain ECAPA-TDNN, OpenWakeWord, Piper/XTTS TTS, Ollama (llama3.2), Claude CLI, xdotool, PIL, numpy, sounddevice

---

## Phase 1: Foundation Modules (no behavior change)

### Task 1: Extract desktop.py — Desktop Control

**Files:**
- Create: `jarvis/desktop.py`
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Create jarvis/desktop.py with all xdotool operations**

Extract from voice_input_gui.py:
- `_execute_desktop_actions()` → `execute_actions(actions)`
- `_parse_desktop_action()` → `parse_action(text)`
- `_find_claude_terminal()` → `find_claude_terminal()`
- `_get_window_list()` → `get_window_list()`
- `_type_text()` → `type_text(text, target_wid, target_name, auto_enter, live_write)`
- `_live_type_partial()` → `live_type_partial(text, state)`
- All DESKTOP_ACTIONS dict and parsing logic
- Window switching, tab control, scroll, click, keyboard shortcuts
- Multi-monitor `move_window_to_monitor()`
- `click_on_text()` from jarvis_agent.py

The module should be a class `DesktopController` with no Tkinter dependencies.

```python
"""jarvis/desktop.py — Desktop automation via xdotool."""
import re
import subprocess
import time
import threading
from pathlib import Path

class DesktopController:
    def __init__(self):
        self._target_wid = None
        self._target_name = None

    def execute_actions(self, actions):
        """Execute a chain of (action_type, data) tuples."""
        ...

    def parse_action(self, text):
        """Parse a single text command into an action tuple."""
        ...

    def type_text(self, text, target_wid=None, target_name=None,
                  auto_enter=False):
        """Type text into target window."""
        ...

    def switch_window(self, name):
        """Find and focus a window by name."""
        ...

    def get_window_list(self):
        """List all visible windows as (wid, name) tuples."""
        ...

    def find_claude_terminal(self):
        """Find Claude Code terminal by spinner title."""
        ...

    # ... tab, scroll, click, media, shortcuts, multi-monitor
```

- [ ] **Step 2: Verify desktop.py imports and runs standalone**

Run:
```bash
cd /home/hunterp/jarvis && python3 -c "from jarvis.desktop import DesktopController; d = DesktopController(); print('OK', d.get_window_list()[:2])"
```
Expected: prints OK with window list

- [ ] **Step 3: Replace voice_input_gui.py desktop code with imports**

In voice_input_gui.py:
- Add `from jarvis.desktop import DesktopController`
- In `__init__`: `self._desktop = DesktopController()`
- Replace all `self._execute_desktop_actions(...)` with `self._desktop.execute_actions(...)`
- Replace `self._type_text(...)` calls with `self._desktop.type_text(...)`
- Replace `self._parse_desktop_action(...)` with `self._desktop.parse_action(...)`
- Remove the extracted methods and DESKTOP_ACTIONS dict from voice_input_gui.py

- [ ] **Step 4: Launch GUI and test basic functionality**

Run: `python3 jarvis/voice_input_gui.py`
Test: Say "Jarvis, what time is it" — should still work. Check logs for errors.

- [ ] **Step 5: Commit**

```bash
cd /home/hunterp/jarvis && git add -A
git commit -m "refactor: extract desktop.py — desktop control module"
```

---

### Task 2: Extract recorder.py — Audio Recording

**Files:**
- Create: `jarvis/recorder.py`
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Create jarvis/recorder.py**

Extract from voice_input_gui.py:
- `_start_recording()` audio parts (stream open, callback, frames)
- `_stop_and_transcribe()` audio stop parts (stream close, frame concatenation)
- `_check_silence()` and `_check_speaker_silence()`
- `_calibrate_noise()` and `_calibrate_worker()`
- `_resample_to_16k()`
- Audio callback function
- Silence detection state (_silence_start, _loud_chunks, etc.)
- MAX_RECORDING_SECONDS constant
- Noise gate logic

```python
"""jarvis/recorder.py — Audio recording with silence detection."""
import numpy as np
import time
import threading
from collections import deque

SAMPLE_RATE = 16000
MAX_RECORDING_SECONDS = 60

class AudioRecorder:
    def __init__(self, mic_idx=None, native_rate=44100,
                 noise_threshold=0.02, silence_timeout=8.0):
        self.recording = False
        self._audio_frames = []
        self._audio_level = 0.0
        self._stream = None
        # ... state vars

    def start(self, on_audio_level=None):
        """Start recording. Returns immediately, records in background."""
        ...

    def stop(self):
        """Stop recording. Returns (audio_raw, native_rate)."""
        ...

    def resample_to_16k(self, audio_raw):
        """Resample from native rate to 16kHz for Whisper."""
        ...

    @property
    def audio_level(self):
        """Current RMS audio level (0-1) for animation."""
        return self._audio_level
```

- [ ] **Step 2: Verify recorder.py compiles**

```bash
python3 -c "from jarvis.recorder import AudioRecorder; print('OK')"
```

- [ ] **Step 3: Replace voice_input_gui.py recording code with imports**

- [ ] **Step 4: Test recording end-to-end**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: extract recorder.py — audio recording module"
```

---

### Task 3: Extract hotword.py — Wake Word Detection

**Files:**
- Create: `jarvis/hotword.py`
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Create jarvis/hotword.py**

Extract the `HotwordListener` class from voice_input_gui.py. Remove dependency on `self.gui` — use callbacks instead.

```python
"""jarvis/hotword.py — OpenWakeWord hotword detection."""
from openwakeword.model import Model
import sounddevice as sd
import numpy as np
import threading
import time
from pathlib import Path

class HotwordListener:
    THRESHOLD = 0.3

    def __init__(self, mic_idx=None, on_hotword=None):
        self._on_hotword = on_hotword  # callback, replaces self.gui reference
        ...

    def start(self): ...
    def stop(self): ...
    def pause(self): ...
    def resume(self): ...
```

- [ ] **Step 2: Verify hotword.py compiles**
- [ ] **Step 3: Replace voice_input_gui.py hotword code with imports**
- [ ] **Step 4: Test hotword detection**
- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: extract hotword.py — wake word detection"
```

---

### Task 4: Extract commander.py — Command Router

**Files:**
- Create: `jarvis/commander.py`
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Create jarvis/commander.py**

Extract from voice_input_gui.py:
- `IntentClassifier` class
- `_check_quick_command()` → `check_quick_command(text, agent, desktop)`
- `_check_desktop_command()` → `check_desktop_command(text, desktop)`
- All QUICK_COMMANDS, LOCAL_PATTERNS dicts
- Reminder system
- Voice notes commands
- Clipboard commands
- File finder commands
- All the "Jarvis, ..." pattern matching

```python
"""jarvis/commander.py — Command parsing and routing."""

class Commander:
    def __init__(self, desktop, agent, brain, tts_speak=None):
        self.desktop = desktop
        self.agent = agent
        self.brain = brain
        self._speak = tts_speak
        self.intent = IntentClassifier()

    def process(self, text):
        """Process transcribed text. Returns (handled: bool, result)."""
        # Check desktop commands
        # Check quick commands
        # Check agent commands (remember, recall, etc.)
        # Check local questions
        # Route to brain (Ollama or Claude)
        ...

class IntentClassifier:
    """Learned intent classifier — is this for Jarvis or background?"""
    ...
```

- [ ] **Step 2: Verify commander.py compiles**
- [ ] **Step 3: Replace voice_input_gui.py command handling with imports**
- [ ] **Step 4: Test commands end-to-end**
- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: extract commander.py — command routing"
```

---

### Task 5: Extract transcriber.py — Speech-to-Text

**Files:**
- Create: `jarvis/transcriber.py`
- Modify: `jarvis/voice_input_gui.py`

- [ ] **Step 1: Create jarvis/transcriber.py**

Extract from voice_input_gui.py:
- `_load_model_worker()` → model loading
- `_transcribe_worker()` → full transcription pipeline
- `_partial_transcribe_worker()` → streaming partial
- Speaker verification integration (segment filtering)
- Confidence gate
- Voice command stripping
- Screenshot phrase detection

```python
"""jarvis/transcriber.py — Whisper transcription + speaker verification."""

class Transcriber:
    def __init__(self, model_size="small", gpu=1, speaker_verifier=None):
        self._model = None
        self._speaker = speaker_verifier
        ...

    def load_model(self):
        """Load Whisper model onto GPU."""
        ...

    def transcribe(self, audio_16k):
        """Full transcription with speaker filtering + confidence gate.
        Returns (text, seg_data, flags)."""
        ...

    def transcribe_partial(self, audio_16k):
        """Quick partial transcription for live preview."""
        ...
```

- [ ] **Step 2: Verify transcriber.py compiles**
- [ ] **Step 3: Replace voice_input_gui.py transcription code with imports**
- [ ] **Step 4: Test transcription end-to-end**
- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: extract transcriber.py — speech-to-text"
```

---

## Phase 2: Intelligence Upgrades

### Task 6: Create context.py — Context Engine

**Files:**
- Create: `jarvis/context.py`
- Modify: `jarvis/jarvis_brain.py`

- [ ] **Step 1: Create jarvis/context.py**

```python
"""jarvis/context.py — Rich context for all intelligence tiers."""
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

class ContextEngine:
    CACHE_TTL = 30  # seconds

    def __init__(self, project_dir="/home/hunterp/jarvis"):
        self._cache = {}
        self._cache_times = {}
        self._conversation = []
        self._project_dir = project_dir

    def get_context(self, detail="standard"):
        """Get context snapshot at specified detail level.

        'minimal' — active window + last command
        'standard' — + git, recent files, conversation
        'full' — + screen, errors, system state, memory
        """
        ctx = {}
        ctx["time"] = datetime.now().strftime("%I:%M %p, %A %B %d %Y")
        ctx["active_window"] = self._get_active_window()

        if detail in ("standard", "full"):
            ctx["git"] = self._get_git_state()
            ctx["recent_files"] = self._get_recent_files()
            ctx["conversation"] = self._conversation[-10:]

        if detail == "full":
            ctx["system"] = self._get_system_state()
            ctx["errors"] = self._get_recent_errors()
            ctx["screen"] = self._get_screen_info()

        return ctx

    def add_exchange(self, user_text, jarvis_response):
        """Add a conversation exchange."""
        self._conversation.append({
            "time": datetime.now().isoformat(),
            "user": user_text,
            "jarvis": jarvis_response,
        })
        self._conversation = self._conversation[-20:]

    def format_for_prompt(self, ctx):
        """Format context dict into a text prompt section."""
        parts = [f"Time: {ctx['time']}"]
        if ctx.get("active_window"):
            parts.append(f"Active window: {ctx['active_window']}")
        if ctx.get("git"):
            g = ctx["git"]
            parts.append(f"Git: {g.get('branch','')} | {g.get('changed',0)} changed files")
        if ctx.get("conversation"):
            parts.append("Recent conversation:")
            for ex in ctx["conversation"][-3:]:
                parts.append(f"  User: {ex['user'][:60]}")
                parts.append(f"  Jarvis: {ex['jarvis'][:60]}")
        if ctx.get("system"):
            parts.append(f"System: {ctx['system']}")
        return "\n".join(parts)

    # Private methods for each data source (cached)
    def _cached(self, key, getter):
        now = time.monotonic()
        if (key in self._cache and
                now - self._cache_times.get(key, 0) < self.CACHE_TTL):
            return self._cache[key]
        value = getter()
        self._cache[key] = value
        self._cache_times[key] = now
        return value

    def _get_git_state(self):
        return self._cached("git", self._fetch_git)

    def _fetch_git(self):
        try:
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5,
                cwd=self._project_dir,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
                cwd=self._project_dir,
            ).stdout.strip()
            changed = len(status.splitlines()) if status else 0
            last = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                capture_output=True, text=True, timeout=5,
                cwd=self._project_dir,
            ).stdout.strip()
            return {"branch": branch, "changed": changed, "last_commit": last}
        except Exception:
            return {}

    def _get_active_window(self):
        return self._cached("window", lambda: subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip() if subprocess else "unknown")

    def _get_recent_files(self):
        return self._cached("files", lambda: subprocess.run(
            ["find", self._project_dir, "-maxdepth", "3", "-type", "f",
             "-not", "-path", "*/.*", "-not", "-path", "*/__pycache__/*",
             "-printf", "%T@ %p\n"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[-5:] if subprocess else [])

    def _get_system_state(self):
        return self._cached("system", self._fetch_system)

    def _fetch_system(self):
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            return f"GPU: {gpu}"
        except Exception:
            return "GPU info unavailable"

    def _get_recent_errors(self):
        try:
            log = Path("/tmp/vss_voice/gui_debug.log")
            if log.exists():
                lines = log.read_text().splitlines()[-50:]
                errors = [l for l in lines if "error" in l.lower() or "exception" in l.lower()]
                return errors[-5:] if errors else []
        except Exception:
            pass
        return []

    def _get_screen_info(self):
        return self._cached("screen", lambda: "/tmp/vss_screen/latest.png")
```

- [ ] **Step 2: Verify context.py compiles and works**

```bash
python3 -c "
import sys; sys.path.insert(0, '/home/hunterp/jarvis')
from jarvis.context import ContextEngine
ctx = ContextEngine()
result = ctx.get_context('standard')
print('Git:', result.get('git'))
print('Window:', result.get('active_window'))
print(ctx.format_for_prompt(result)[:200])
"
```

- [ ] **Step 3: Integrate context into jarvis_brain.py**

Update `_query_ollama()` and `_query_claude_peers()` to inject context:
```python
# In JarvisBrain.__init__:
from jarvis.context import ContextEngine
self._context = ContextEngine()

# In _query_ollama:
ctx = self._context.get_context("standard")
ctx_text = self._context.format_for_prompt(ctx)
system = JARVIS_SYSTEM.format(time=ctx["time"]) + f"\n\nContext:\n{ctx_text}"

# In _query_claude_peers:
ctx = self._context.get_context("full")
ctx_text = self._context.format_for_prompt(ctx)
prompt = f"...\n\nContext:\n{ctx_text}\n\nUser: {user_input}"
```

- [ ] **Step 4: Test context-aware responses**

```bash
python3 -c "
import sys; sys.path.insert(0, '/home/hunterp/jarvis')
from jarvis.jarvis_brain import JarvisBrain
brain = JarvisBrain()
actions = brain._query_ollama('what am I working on right now')
print(actions)
"
```
Should mention git branch and recent files.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: context engine — rich context injection for all tiers"
```

---

### Task 7: Create memory.py — Persistent Memory

**Files:**
- Create: `jarvis/memory.py`
- Modify: `jarvis/jarvis_brain.py`
- Modify: `jarvis/jarvis_agent.py`

- [ ] **Step 1: Create jarvis/memory.py**

```python
"""jarvis/memory.py — Persistent memory across sessions."""
import json
import time
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path.home() / ".aiws_trainer" / "jarvis_memory"

class JarvisMemory:
    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._facts = self._load("facts.json", {})
        self._habits = self._load("habits.json", [])
        self._preferences = self._load("preferences.json", {})
        self._session_summaries = self._load("sessions.json", [])
        self._intent_log = self._load("intent_log.json", [])

    def _load(self, filename, default):
        path = MEMORY_DIR / filename
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return default

    def _save(self, filename, data):
        try:
            (MEMORY_DIR / filename).write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    # Facts
    def remember(self, key, value):
        self._facts[key] = {"value": value, "time": datetime.now().isoformat()}
        self._save("facts.json", self._facts)

    def recall(self, query):
        query_lower = query.lower()
        matches = []
        for key, entry in self._facts.items():
            if query_lower in key.lower() or query_lower in str(entry["value"]).lower():
                matches.append({"key": key, **entry})
        return matches[:5]

    # Habits
    def log_habit(self, command, context=None):
        self._habits.append({
            "time": datetime.now().isoformat(),
            "hour": datetime.now().hour,
            "day": datetime.now().strftime("%A"),
            "command": command,
            "context": context,
        })
        self._habits = self._habits[-500:]
        self._save("habits.json", self._habits)

    def suggest_by_habit(self):
        if len(self._habits) < 10:
            return None
        hour = datetime.now().hour
        counts = {}
        for h in self._habits:
            if abs(h.get("hour", -1) - hour) <= 1:
                cmd = h["command"]
                counts[cmd] = counts.get(cmd, 0) + 1
        if counts:
            best = max(counts, key=counts.get)
            if counts[best] >= 3:
                return best
        return None

    # Preferences
    def set_preference(self, key, value):
        self._preferences[key] = value
        self._save("preferences.json", self._preferences)

    def get_preference(self, key, default=None):
        return self._preferences.get(key, default)

    # Session summaries
    def save_session_summary(self, summary):
        self._session_summaries.append({
            "time": datetime.now().isoformat(),
            "summary": summary,
        })
        self._session_summaries = self._session_summaries[-20:]
        self._save("sessions.json", self._session_summaries)

    def get_recent_sessions(self, n=3):
        return self._session_summaries[-n:]

    # Intent learning
    def log_intent(self, text, is_for_assistant):
        self._intent_log.append({
            "text": text,
            "label": "yes" if is_for_assistant else "no",
        })
        self._intent_log = self._intent_log[-500:]
        self._save("intent_log.json", self._intent_log)

    def get_intent_log(self):
        return self._intent_log

    # Voice notes (consolidate from agent)
    def save_note(self, text):
        notes_dir = MEMORY_DIR / "notes"
        notes_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = notes_dir / f"note_{ts}.txt"
        path.write_text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n{text}\n")
        return str(path)

    def get_notes(self, n=5):
        notes_dir = MEMORY_DIR / "notes"
        if not notes_dir.exists():
            return []
        files = sorted(notes_dir.glob("note_*.txt"), reverse=True)[:n]
        return [{"file": f.name, "content": f.read_text().strip()[:100]} for f in files]
```

- [ ] **Step 2: Verify memory.py works**

```bash
python3 -c "
import sys; sys.path.insert(0, '/home/hunterp/jarvis')
from jarvis.memory import JarvisMemory
m = JarvisMemory()
m.remember('test', 'this is a test')
print(m.recall('test'))
m.log_habit('check gpu')
print(m.suggest_by_habit())
"
```

- [ ] **Step 3: Integrate memory into brain and context engine**

In context.py, add memory to full context:
```python
def __init__(self, ...):
    from jarvis.memory import JarvisMemory
    self._memory = JarvisMemory()
```

In brain.py, log conversations and use memory for context.

- [ ] **Step 4: Test memory persistence**

Run the test twice — second run should recall data from first.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: persistent memory — facts, habits, sessions, notes"
```

---

### Task 8: Upgrade brain.py — Autonomous Multi-Step Tasks

**Files:**
- Modify: `jarvis/jarvis_brain.py`

- [ ] **Step 1: Add autonomous task execution to brain**

Add a method that chains multiple steps with decision-making between them:

```python
def execute_autonomous(self, task_description, callback=None):
    """Execute a multi-step task autonomously.

    Each step:
    1. Query Claude with task + results so far
    2. Parse structured response
    3. Execute [RUN] actions, collect output
    4. Feed output back into next query
    5. Repeat until Claude says [DONE] or max steps reached
    """
    MAX_STEPS = 10
    results = []
    current_task = task_description

    for step in range(MAX_STEPS):
        prompt = f"""Task: {task_description}
Step {step + 1}/{MAX_STEPS}
Previous results:
{json.dumps(results[-3:], indent=2) if results else 'None yet'}

What should I do next? Respond with structured commands.
If the task is complete, respond with [DONE] message."""

        actions = self._query_claude_peers(prompt)

        for action_type, action_data in actions:
            if action_type == "DONE":
                if callback:
                    callback([("SPEAK", action_data)])
                return results
            elif action_type == "RUN":
                output = subprocess.run(
                    action_data, shell=True, capture_output=True,
                    text=True, timeout=30,
                ).stdout.strip()[:500]
                results.append({"step": step, "command": action_data,
                               "output": output})
            elif action_type == "SPEAK":
                if callback:
                    callback([("SPEAK", action_data)])

    return results
```

- [ ] **Step 2: Wire autonomous mode to commander**

In commander.py, detect "deploy", "debug this", etc. and route to `brain.execute_autonomous()`.

- [ ] **Step 3: Test autonomous deploy workflow**

```bash
# Say: "Jarvis, deploy"
# Expected: runs tests → reports results → asks to commit → commits
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: autonomous multi-step task execution"
```

---

## Phase 3: GUI Refactor

### Task 9: Extract gui.py — Clean UI Layer

**Files:**
- Create: `jarvis/gui.py`
- Modify: `jarvis/voice_input_gui.py` (becomes thin launcher)

- [ ] **Step 1: Create jarvis/gui.py with all Tkinter code**

Extract:
- Color palette constants
- `_build_ui()` — all widget creation
- Arc Reactor animation (particle orbit, pre-rendered frames)
- `_update_orbit_idle()`, `_update_waveform()`, `_render_orbit_fast()`
- `_compose_frame()`, `_apply_frame()`
- `_update_speaking_animation()`
- Status bar updates
- Settings panel (collapsible)
- Dual transcription boxes
- Button styling (`_holo_btn`)
- TTK theme configuration
- Window management (minimize to tray, close, etc.)

```python
"""jarvis/gui.py — Jarvis Arc Reactor GUI."""

class JarvisGUI:
    # Color palette
    BG = "#0a0e14"
    CARD_BG = "#1a2332"
    # ...

    def __init__(self, root, on_record_toggle=None, on_close=None):
        self.root = root
        self._on_record_toggle = on_record_toggle
        # ... build UI, no audio/AI logic

    def set_status(self, label, color, detail=""):
        ...

    def show_user_text(self, text, seg_data=None):
        ...

    def show_jarvis_text(self, text):
        ...

    def set_audio_level(self, level):
        """Update arc reactor animation amplitude."""
        ...

    def set_recording(self, recording):
        """Update UI state for recording on/off."""
        ...
```

- [ ] **Step 2: Update voice_input_gui.py to be a thin orchestrator**

```python
"""jarvis/voice_input_gui.py — Thin orchestrator wiring all modules."""

class VoiceInputApp:
    def __init__(self, root):
        self.gui = JarvisGUI(root, on_record_toggle=self._toggle)
        self.recorder = AudioRecorder(...)
        self.transcriber = Transcriber(...)
        self.commander = Commander(...)
        self.brain = JarvisBrain()
        self.desktop = DesktopController()
        self.context = ContextEngine()
        self.memory = JarvisMemory()
        self.tts = JarvisTTS()
        self.hotword = HotwordListener(on_hotword=self._on_hotword)
        # Wire everything together
```

- [ ] **Step 3: Test full end-to-end**
- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: extract gui.py — clean UI layer, thin orchestrator"
```

---

### Task 10: End-to-End Validation

**Files:** None new — testing only

- [ ] **Step 1: Test hotword detection**

Say "Jarvis" with GUI open → should start recording

- [ ] **Step 2: Test push-to-talk**

Hold Ctrl+Shift+R → speak → release → should transcribe and respond

- [ ] **Step 3: Test Jarvis Mode (Ollama)**

Enable Jarvis Mode → say "Jarvis, what time is it" → should respond via TTS

- [ ] **Step 4: Test Jarvis Mode (Claude)**

Say "Jarvis, explain the context engine we built" → should route to Claude, respond with project context

- [ ] **Step 5: Test desktop control**

Say "Jarvis, switch to Opera and scroll down" → should chain commands

- [ ] **Step 6: Test memory persistence**

Say "Jarvis, remember that training uses batch size 16" → close GUI → reopen → say "Jarvis, recall training" → should remember

- [ ] **Step 7: Test autonomous workflow**

Say "Jarvis, deploy" → should run multi-step workflow autonomously

- [ ] **Step 8: Test talk-back with animation**

Say something → Jarvis responds → arc reactor should pulse during speech, then idle

- [ ] **Step 9: Final commit and push**

```bash
git add -A && git commit -m "test: end-to-end validation of Jarvis V2 architecture"
git push
```
