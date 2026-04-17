"""Voice Input GUI — push-to-talk speech-to-text with local Whisper.

Records from microphone, transcribes with faster-whisper on GPU,
and types the result into the active terminal via xdotool.

Features:
    - Push-to-talk with big Record/Stop button
    - Live waveform visualization
    - Real-time partial transcription (streaming preview)
    - Voice commands (new line, period, comma, delete that, etc.)
    - Review-before-typing mode (edit text before sending)
    - Continuous mode (auto-restart after transcription)
    - Custom vocabulary for warehouse domain terms
    - Multi-language support (90+ languages)
    - Sound feedback (beep on record start/stop)
    - System tray icon with recording state
    - Keyboard shortcuts (F5 / Space to toggle)
    - Auto-stop on silence detection
    - Noise gate (filter keyboard/fan noise)
    - Session log export

Usage:
    # As standalone:
    python -m jarvis.voice_input_gui

    # From main launcher (Toplevel):
    from jarvis.voice_input_gui import VoiceInputGUI
    gui = VoiceInputGUI(tk.Toplevel(root))
"""

import io
import json
import os
import re
import sys
import math
import time
import wave
import struct
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np

from jarvis.recording import RecordingController
from jarvis.transcription import TranscriptionPipeline
from jarvis.dispatcher import CommandDispatcher, CommandHandler
from jarvis.animation import AnimationRenderer, generate_beep as _generate_beep_samples

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.04
SILENCE_TIMEOUT = 2.0
DEFAULT_MODEL = "small"
DEFAULT_GPU = 1
LOG_DIR = Path("/tmp/vss_voice")
SESSION_LOG_DIR = Path.home() / ".aiws_trainer"
VOCAB_FILE = SESSION_LOG_DIR / "voice_vocab.txt"
SETTINGS_FILE = SESSION_LOG_DIR / "voice_settings.json"

# Hotword settings
HOTWORD_PHRASES = {
    # Claude variants
    "hey claude", "hey cloud", "hey claud", "a claude",
    "okay claude", "ok claude", "yo claude",
    # Jarvis variants (common Whisper mishearings)
    "jarvis", "hey jarvis", "hey jarv",
    "jarvas", "jarves", "jarvus", "service jarvis",
    "nervous", "harvest",  # common mishearings of "jarvis"
}
HOTWORD_WINDOW = 2.0   # seconds of audio to check
HOTWORD_INTERVAL = 0.8  # seconds between checks

# Domain vocabulary prompt — biases Whisper toward these terms
DEFAULT_VOCAB = (
    "AGV, loaded AGV, empty AGV, forklift, loaded forklift, empty forklift, "
    "pallet, flat pallet, boxes pallet, full pallet, cardboard box, conveyor, "
    "YOLO, VSS, KPI, ReID, SAM, ONNX, CUDA, GPU, RTSP, MJPEG, "
    "zone, dwell time, heatmap, flow rate, counting line, proximity alert, "
    "warehouse, surveillance, detection, tracking, annotation, training, "
    "ChromaDB, knowledge graph, Cosmos VLM, narration, Ollama, "
    "near-miss, safety incident, loading zone, staging area, shipping dock"
)

# Waveform display settings
WAVEFORM_BARS = 64

# Noise gate: RMS below this in a block → zero it out
NOISE_GATE_THRESHOLD = 0.005

# Shell commands allowed without voice confirmation.
# Anything outside this set requires a spoken "yes" or "confirm".
SHELL_ALLOWLIST = {
    "ls", "pwd", "cd", "git", "df", "du", "free", "uptime",
    "date", "whoami", "hostname", "wc", "cat", "head", "tail",
    "echo", "which", "whereis", "ps", "top", "env", "printenv",
    "python", "python3", "pip", "pytest",
}


def _draw_circle(frame, cx, cy, r, color, alpha):
    """Draw a circle outline onto a numpy RGB frame."""
    S = frame.shape[0]
    theta = np.linspace(0, 2 * np.pi, max(20, r * 4), endpoint=False)
    xs = (cx + r * np.cos(theta)).astype(int)
    ys = (cy + r * np.sin(theta)).astype(int)
    mask = (xs >= 0) & (xs < S) & (ys >= 0) & (ys < S)
    xs, ys = xs[mask], ys[mask]
    for ch in range(3):
        current = frame[ys, xs, ch].astype(np.float32)
        frame[ys, xs, ch] = np.clip(
            current + (color[ch] - current) * alpha, 0, 255
        ).astype(np.uint8)


def _draw_filled_circle(frame, cx, cy, r, color, alpha):
    """Draw a filled circle onto a numpy RGB frame."""
    S = frame.shape[0]
    y1, y2 = max(0, cy - r), min(S, cy + r + 1)
    x1, x2 = max(0, cx - r), min(S, cx + r + 1)
    if y1 >= y2 or x1 >= x2:
        return
    yy, xx = np.ogrid[y1 - cy:y2 - cy, x1 - cx:x2 - cx]
    mask = (xx * xx + yy * yy) <= r * r
    region = frame[y1:y2, x1:x2]
    for ch in range(3):
        current = region[:, :, ch].astype(np.float32)
        blended = current + (color[ch] - current) * alpha * mask
        region[:, :, ch] = np.clip(blended, 0, 255).astype(np.uint8)

# Filler sounds that indicate thinking — reset silence timer when detected
# Only pure filler sounds, NOT common words like "like", "so", "well"
FILLER_WORDS = {"uh", "um", "uhh", "umm", "hmm", "hm", "er", "ah", "ehh", "eh",
                "erm", "uhhh", "ummm"}

class IntentClassifier:
    """Learns whether speech is directed at the assistant or is background chat.

    Three-tier system:
    - CONFIDENT_YES: clearly a command/question → type immediately
    - CONFIDENT_NO: clearly side conversation → discard silently
    - UNCERTAIN: ask user "Was this for me?" → log answer to improve

    Logged examples are stored in ~/.aiws_trainer/intent_log.json and used
    to train a simple text classifier that improves over time.
    """

    INTENT_LOG = Path.home() / ".aiws_trainer" / "intent_log.json"
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"

    # Patterns that strongly suggest assistant-directed speech
    _POSITIVE_PATTERNS = [
        "how do", "how can", "can you", "could you", "would you", "what is",
        "what are", "what's", "where is", "where's", "why is", "why does",
        "is there", "are there", "do you", "tell me", "show me", "explain",
        "implement", "fix", "create", "make", "build", "add", "remove",
        "delete", "update", "change", "modify", "run", "check", "test",
        "start", "stop", "open", "close", "save", "commit", "push",
        "install", "deploy", "debug", "refactor", "write",
        "jarvis", "claude", "hey claude",
        "the code", "the file", "the bug", "the error", "the gui",
        "the config", "the model", "the script", "the function",
        "this file", "this code", "this bug",
        "let's", "let me", "i want", "i need", "i'd like",
        "go ahead", "please", "take a look", "look at",
        "check the screen", "screenshot", "take a screenshot",
        "take screenshot",
    ]

    # Patterns that suggest casual/side conversation
    _NEGATIVE_PATTERNS = [
        "bless her", "bless him", "oh my god", "that's crazy",
        "no way", "for real", "i know right", "lol", "haha",
        "she said", "he said", "they said", "she's", "he's",
        "dude", "bro", "man ", "yo ",
    ]

    def __init__(self):
        self._log_data = []  # List of {"text": ..., "label": "yes"/"no"}
        self._learned_positive = set()  # Phrases learned as positive
        self._learned_negative = set()  # Phrases learned as negative
        self._load_log()

    def _load_log(self):
        """Load logged intent examples from disk."""
        if not self.INTENT_LOG.exists():
            return
        try:
            self._log_data = json.loads(self.INTENT_LOG.read_text())
            # Build learned pattern sets from logged examples
            for entry in self._log_data:
                text_lower = entry["text"].lower()
                words = text_lower.split()
                label = entry["label"]
                # Extract 2-3 word ngrams as learned patterns
                for n in (2, 3):
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i:i + n])
                        if label == self.YES:
                            self._learned_positive.add(ngram)
                            self._learned_negative.discard(ngram)
                        else:
                            self._learned_negative.add(ngram)
                            self._learned_positive.discard(ngram)
        except Exception as e:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = self.INTENT_LOG.with_suffix(f".json.corrupt.{ts}")
            try:
                self.INTENT_LOG.rename(backup)
                _log(f"Intent log corrupt ({e}); backed up to {backup.name}")
            except Exception as rename_err:
                _log(f"Intent log corrupt and unbackup-able: {rename_err}")
            self._log_data = []

    def _save_log(self):
        """Save intent log to disk."""
        try:
            self.INTENT_LOG.parent.mkdir(parents=True, exist_ok=True)
            self.INTENT_LOG.write_text(json.dumps(self._log_data, indent=2))
        except Exception:
            pass

    def log_feedback(self, text, is_for_assistant):
        """Record user feedback on whether text was directed at assistant."""
        label = self.YES if is_for_assistant else self.NO
        self._log_data.append({"text": text, "label": label})

        # Keep log manageable (last 500 entries)
        if len(self._log_data) > 500:
            self._log_data = self._log_data[-500:]

        # Update learned patterns
        words = text.lower().split()
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                if is_for_assistant:
                    self._learned_positive.add(ngram)
                    self._learned_negative.discard(ngram)
                else:
                    self._learned_negative.add(ngram)
                    self._learned_positive.discard(ngram)

        self._save_log()

    def classify(self, text):
        """Classify text as YES, NO, or UNCERTAIN.

        Returns (classification, confidence) where confidence is 0-1.
        """
        if not text or len(text.strip()) < 3:
            return self.NO, 1.0

        lower = text.strip().lower()
        words = lower.split()
        pos_score = 0
        neg_score = 0

        # --- Rule-based signals ---

        # Very short reactions (1-3 words)
        if len(words) <= 3:
            if any(lower.startswith(p) for p in (
                    "run", "fix", "check", "stop", "test", "take",
                    "commit", "push", "show", "open", "screenshot",
                    "save", "deploy", "start", "build", "install")):
                return self.YES, 0.9
            if "screenshot" in lower:
                return self.YES, 0.9
            if lower.endswith("?"):
                return self.YES, 0.8
            return self.NO, 0.8

        # Strong positive patterns
        for pattern in self._POSITIVE_PATTERNS:
            if pattern in lower:
                pos_score += 2

        # Strong negative patterns
        for pattern in self._NEGATIVE_PATTERNS:
            if pattern in lower:
                neg_score += 2

        # Questions
        if lower.rstrip().endswith("?"):
            pos_score += 1.5

        # 3rd person pronouns (talking about others)
        other_pronouns = {"she", "he", "they", "her", "him", "them", "his"}
        pronoun_count = sum(1 for w in words if w in other_pronouns)
        neg_score += pronoun_count * 0.5

        # Long text without positive signals
        if len(words) >= 8 and pos_score == 0:
            neg_score += 1

        # --- Learned patterns from feedback ---
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                if ngram in self._learned_positive:
                    pos_score += 1.5
                if ngram in self._learned_negative:
                    neg_score += 1.5

        # --- Decision ---
        total = pos_score + neg_score
        if total == 0:
            # No signals either way — uncertain
            return self.UNCERTAIN, 0.5

        pos_ratio = pos_score / total

        if pos_ratio >= 0.7:
            return self.YES, pos_ratio
        elif pos_ratio <= 0.3:
            return self.NO, 1 - pos_ratio
        else:
            return self.UNCERTAIN, 0.5

    @property
    def num_examples(self):
        return len(self._log_data)

# Streaming: partial transcription interval (seconds)
STREAMING_INTERVAL = 2.0

# Language options
LANGUAGES = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Turkish", "tr"),
    ("Polish", "pl"),
    ("Ukrainian", "uk"),
    ("Vietnamese", "vi"),
    ("Thai", "th"),
    ("Swedish", "sv"),
]

# ------------------------------------------------------------------
# Voice commands — spoken phrase → replacement
# ------------------------------------------------------------------
VOICE_COMMANDS = [
    # Punctuation
    (r"\b(?:period|full stop)\b", "."),
    (r"\bcomma\b", ","),
    (r"\b(?:question mark)\b", "?"),
    (r"\b(?:exclamation mark|exclamation point)\b", "!"),
    (r"\bcolon\b", ":"),
    (r"\bsemicolon\b", ";"),
    (r"\b(?:dash|hyphen)\b", "-"),
    (r"\bellipsis\b", "..."),
    (r"\b(?:open paren|open parenthesis|left paren)\b", "("),
    (r"\b(?:close paren|close parenthesis|right paren)\b", ")"),
    (r"\b(?:open quote|open quotes|begin quote)\b", '"'),
    (r"\b(?:close quote|close quotes|end quote)\b", '"'),
    (r"\b(?:single quote|apostrophe)\b", "'"),
    # Whitespace / structure
    (r"\b(?:new line|newline|line break)\b", "\n"),
    (r"\btab\b(?:\s+(?:key|character))?", "\t"),
    # Editing (special actions handled separately)
    (r"\b(?:backspace|back space)\b", "\x08"),
]

# Special action commands (not simple replacements)
ACTION_COMMANDS = {
    "delete that": "delete_last_sentence",
    "scratch that": "delete_last_sentence",
    "undo that": "delete_last_sentence",
    "clear all": "clear_all",
    "select all": "select_all",
}

# Screenshot trigger phrases — stripped from text, triggers capture after typing
SCREENSHOT_PHRASES = [
    "and take a screenshot", "and take screenshot", "and screenshot",
    "take a screenshot", "take screenshot", "capture screen",
    "screen capture", "screenshot",
]

# Voice targeting patterns — "target X", "switch to X", "type in X", "go to X"
TARGET_PATTERN = re.compile(
    r"^(?:target|switch to|type in|go to|focus|open)\s+(.+)$",
    re.IGNORECASE,
)
# Reset target back to auto
TARGET_RESET_PHRASES = {"target auto", "target claude", "reset target", "target default"}

# Voice phrases that stop recording (stripped from final transcription)
STOP_RECORDING_PHRASES = {
    "end recording", "stop recording", "stop listening",
    "done recording", "finish recording",
}

# Quick voice commands — "Jarvis, commit" etc.
_VSS = str(Path.home() / "vss_env")
QUICK_COMMANDS = {
    "commit": f"cd {_VSS} && git add -A && git status -s",
    "run tests": f"cd {_VSS} && python scripts/agents/run_all.py --quick",
    "check gpu": "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv",
    "check disk": "df -h / /storage 2>/dev/null",
    "check logs": "tail -20 /tmp/vss_voice/gui_debug.log",
    "system status": "uptime && free -h && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
}

# Desktop control commands — parsed from natural language
DESKTOP_ACTIONS = {
    # Window management
    "switch to": "window",
    "go to": "window",
    "open": "window",
    "focus": "window",
    "close window": "close",
    "minimize": "minimize",
    "maximize": "maximize",
    "full screen": "fullscreen",
    # Mouse/scroll
    "scroll up": "scroll_up",
    "scroll down": "scroll_down",
    "scroll left": "scroll_left",
    "scroll right": "scroll_right",
    "click": "click",
    "double click": "double_click",
    "right click": "right_click",
    # Tabs
    "next tab": "next_tab",
    "previous tab": "prev_tab",
    "new tab": "new_tab",
    "close tab": "close_tab",
    # System
    "volume up": "vol_up",
    "volume down": "vol_down",
    "mute": "mute",
    "play": "media_play",
    "pause": "media_pause",
    # Keyboard shortcuts
    "copy": "copy",
    "paste": "paste",
    "undo": "undo",
    "redo": "redo",
    "save": "save",
    "select all": "select_all",
    "find": "find",
}

# Reminder storage
REMINDERS_FILE = Path.home() / ".aiws_trainer" / "jarvis_reminders.json"


def _apply_voice_commands(text):
    """Apply voice command replacements to transcribed text."""
    result = text

    # Check for action commands first (full phrase match)
    text_lower = result.strip().lower()
    for phrase, action in ACTION_COMMANDS.items():
        if text_lower == phrase or text_lower.endswith(phrase):
            return f"__ACTION__{action}"

    # Apply punctuation/whitespace replacements
    for pattern, replacement in VOICE_COMMANDS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Clean up spaces around punctuation and newlines
    result = re.sub(r"\s+([.,!?;:)\]])", r"\1", result)
    result = re.sub(r"([([\[])\s+", r"\1", result)
    result = re.sub(r"\s*\n\s*", "\n", result)

    # Handle backspace markers
    while "\x08" in result:
        idx = result.index("\x08")
        if idx > 0:
            result = result[:idx - 1] + result[idx + 1:]
        else:
            result = result[1:]

    return result.strip()


from jarvis.logging import get_logger
_log = get_logger("GUI")


# ------------------------------------------------------------------
# Sound generation (no external files needed)
# ------------------------------------------------------------------
def _generate_beep(freq=880, duration_ms=120, volume=0.3):
    """Generate a short beep as WAV bytes (vectorized)."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    i = np.arange(n_samples)
    t = i / SAMPLE_RATE
    env = np.minimum(i / 200, 1.0) * np.minimum((n_samples - i) / 200, 1.0)
    samples = (volume * env * 32767 * np.sin(2 * np.pi * freq * t))
    samples = np.clip(samples, -32767, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


_BEEP_START = None
_BEEP_STOP = None


def _init_beeps():
    """Pre-generate beep WAV files to /tmp."""
    global _BEEP_START, _BEEP_STOP
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start_path = LOG_DIR / "beep_start.wav"
    stop_path = LOG_DIR / "beep_stop.wav"

    if not start_path.exists():
        start_path.write_bytes(_generate_beep(freq=880, duration_ms=100))
    if not stop_path.exists():
        stop_path.write_bytes(_generate_beep(freq=660, duration_ms=150))

    _BEEP_START = str(start_path)
    _BEEP_STOP = str(stop_path)


def _play_beep(path):
    """Play a WAV file asynchronously using paplay or aplay."""
    if not path:
        return
    for cmd in [["paplay", path], ["aplay", "-q", path]]:
        try:
            subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return
        except FileNotFoundError:
            continue


# ------------------------------------------------------------------
# Custom vocabulary (persisted to file)
# ------------------------------------------------------------------
def _load_vocab():
    """Load domain vocabulary from user file, or return default."""
    if VOCAB_FILE.exists():
        try:
            text = VOCAB_FILE.read_text().strip()
            if text:
                return text
        except Exception:
            pass
    return DEFAULT_VOCAB


def _save_vocab(text):
    """Save domain vocabulary to user file."""
    VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_FILE.write_text(text.strip())
    _log(f"Vocabulary saved to {VOCAB_FILE}")


# ------------------------------------------------------------------
# Noise gate
# ------------------------------------------------------------------
def _apply_noise_gate(audio, threshold=NOISE_GATE_THRESHOLD, block_size=1600):
    """Zero out blocks of audio below RMS threshold."""
    result = audio.copy()
    for i in range(0, len(result), block_size):
        block = result[i:i + block_size]
        rms = np.sqrt(np.mean(block ** 2))
        if rms < threshold:
            result[i:i + block_size] = 0.0
    return result


# ------------------------------------------------------------------
# System tray icon (optional — graceful fallback)
# ------------------------------------------------------------------
class TrayIcon:
    """System tray icon with recording state indicator."""

    def __init__(self, gui):
        self.gui = gui
        self.icon = None
        self._available = False
        self._setup()

    def _setup(self):
        try:
            import pystray
            from PIL import Image, ImageDraw

            self._pystray = pystray
            self._Image = Image
            self._ImageDraw = ImageDraw
            self._available = True
            _log("System tray: available")
        except Exception as e:
            _log(f"System tray: not available ({e})")

    def start(self):
        if not self._available:
            return

        image = self._make_icon(self.gui.GREEN)
        menu = self._pystray.Menu(
            self._pystray.MenuItem("Show/Hide", self._toggle_window, default=True),
            self._pystray.MenuItem("Toggle Record", self._toggle_record),
            self._pystray.Menu.SEPARATOR,
            self._pystray.MenuItem("Quit", self._quit),
        )

        self.icon = self._pystray.Icon(
            "voice_input", image, "Voice Input - Ready", menu
        )
        threading.Thread(target=self.icon.run, daemon=True).start()
        _log("Tray icon started")

    def update_state(self, recording):
        if not self._available or not self.icon:
            return
        color = "#0891b2" if recording else self.gui.GREEN
        title = "Voice Input - Recording..." if recording else "Voice Input - Ready"
        try:
            self.icon.icon = self._make_icon(color)
            self.icon.title = title
        except Exception:
            pass

    def stop(self):
        if self.icon:
            try:
                self.icon.stop()
            except Exception:
                pass

    def _make_icon(self, color):
        img = self._Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = self._ImageDraw.Draw(img)
        draw.rounded_rectangle([20, 8, 44, 38], radius=8, fill=color)
        draw.arc([14, 20, 50, 50], start=0, end=180, fill=color, width=3)
        draw.line([32, 50, 32, 58], fill=color, width=3)
        draw.line([22, 58, 42, 58], fill=color, width=3)
        return img

    def _toggle_window(self):
        self.gui.root.after(0, self.gui._toggle_visibility)

    def _toggle_record(self):
        self.gui.root.after(0, self.gui._toggle_recording)

    def _quit(self):
        self.gui.root.after(0, self.gui._on_close)


# ------------------------------------------------------------------
# Global hotkey listener (Xlib — works even when GUI unfocused)
# ------------------------------------------------------------------
class GlobalHotkey:
    """Register Ctrl+Shift+V system-wide using Xlib record extension."""

    def __init__(self, gui):
        self.gui = gui
        self._thread = None
        self._available = False

    def start(self):
        try:
            from Xlib import X, XK, display
            from Xlib.ext import record
            self._available = True
        except ImportError:
            _log("Global hotkey: Xlib not available")
            return

        self._thread = threading.Thread(target=self._listener, daemon=True)
        self._thread.start()
        _log("Global hotkey listener started (Ctrl+Shift+V)")

    def _listener(self):
        from Xlib import X, XK, display
        from Xlib.ext import record
        from Xlib.protocol import rq

        local_dpy = display.Display()
        record_dpy = display.Display()
        ctrl_held = False
        shift_held = False

        def callback(reply):
            nonlocal ctrl_held, shift_held
            if reply.category != record.FromServer or reply.client_swapped:
                return
            data = reply.data
            while len(data):
                event, data = rq.EventField(None).parse_binary_value(
                    data, record_dpy.display, None, None
                )
                keycode = event.detail
                keysym = local_dpy.keycode_to_keysym(keycode, 0)

                if event.type == X.KeyPress:
                    if keysym in (XK.XK_Control_L, XK.XK_Control_R):
                        ctrl_held = True
                    elif keysym in (XK.XK_Shift_L, XK.XK_Shift_R):
                        shift_held = True
                    elif keysym == XK.XK_v and ctrl_held and shift_held:
                        _log("Global hotkey: Ctrl+Shift+V")
                        self.gui.root.after(0, self.gui._toggle_recording)
                    elif keysym == XK.XK_r and ctrl_held and shift_held:
                        # Push-to-talk: start recording on key press
                        if not self.gui.recording:
                            _log("Push-to-talk: key down")
                            self.gui._voice_stopped = True  # Don't auto-restart
                            self.gui.root.after(0, self.gui._start_recording)
                elif event.type == X.KeyRelease:
                    if keysym in (XK.XK_Control_L, XK.XK_Control_R):
                        ctrl_held = False
                    elif keysym in (XK.XK_Shift_L, XK.XK_Shift_R):
                        shift_held = False
                    elif keysym == XK.XK_r and self.gui.recording:
                        # Push-to-talk: stop on key release
                        _log("Push-to-talk: key up")
                        self.gui._voice_stopped = True
                        self.gui.root.after(0, self.gui._stop_and_transcribe)

        ctx = record_dpy.record_create_context(
            0, [record.AllClients],
            [{"core_requests": (0, 0), "core_replies": (0, 0),
              "ext_requests": (0, 0, 0, 0), "ext_replies": (0, 0, 0, 0),
              "delivered_events": (0, 0),
              "device_events": (X.KeyPress, X.KeyRelease),
              "errors": (0, 0), "client_started": False, "client_died": False}],
        )
        try:
            record_dpy.record_enable_context(ctx, callback)
        except Exception as e:
            _log(f"Global hotkey error: {e}")
        finally:
            try:
                record_dpy.record_free_context(ctx)
            except Exception:
                pass


# ------------------------------------------------------------------
# Hotword listener (OpenWakeWord — lightweight, reliable)
# ------------------------------------------------------------------
class HotwordListener:
    """Always-on wake word listener using OpenWakeWord (CPU, ~1.5ms/prediction).

    Much more reliable than the old Whisper-based approach which ran full
    transcription every 0.8s and often missed short wake words in noise.
    """

    THRESHOLD = 0.2       # Primary trigger threshold
    CONFIRM_THRESHOLD = 0.15  # Confirmation threshold (lower)
    CONFIRM_WINDOW = 1.0      # Seconds to look for confirmation frame

    def __init__(self, gui):
        self.gui = gui
        self.active = False
        self._stream = None
        self._model = None
        self._reopen = False
        self._stream_lock = threading.Lock()

    def start(self):
        if self.active:
            return
        self.active = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        _log("Hotword listener started")

    def stop(self):
        self.active = False
        self._close_stream()
        _log("Hotword listener stopped")

    def pause(self):
        """Release the mic stream so recording can use it."""
        self._close_stream()
        _log("Hotword stream paused (mic released)")

    def resume(self):
        """Re-open the mic stream after recording finishes."""
        if self._stream:
            return
        if self.active:
            self._reopen = True
            _log("Hotword stream will resume")
        else:
            self.start()
            _log("Hotword listener restarted fresh")

    def _close_stream(self):
        with self._stream_lock:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def _listen_loop(self):
        import sounddevice as sd

        # Load OpenWakeWord model (CPU, tiny) with custom verifier if available
        if self._model is None:
            try:
                from openwakeword.model import Model
                # Load base model first, then attach custom verifier after
                self._model = Model()
                _log(f"OpenWakeWord loaded: {list(self._model.models.keys())}")

                verifier_path = Path.home() / ".aiws_trainer" / "hey_jarvis_verifier.pkl"
                if verifier_path.exists():
                    import joblib
                    self._model.custom_verifier_models["hey_jarvis"] = joblib.load(str(verifier_path))
                    self._model.custom_verifier_threshold = 0.3
                    _log(f"Custom wake word verifier attached: {verifier_path}")
            except Exception as e:
                _log(f"OpenWakeWord load error: {e}")
                self.active = False
                return

        self._reopen = False
        mic_name = self.gui.mic_var.get()
        mic_idx = self.gui._mic_devices.get(mic_name)

        # Detect native sample rate
        try:
            dev_info = sd.query_devices(mic_idx, 'input')
            native_rate = int(dev_info['default_samplerate'])
        except Exception:
            native_rate = 44100
        _log(f"Hotword mic rate: {native_rate}Hz")

        # OpenWakeWord needs 16kHz int16 chunks of 1280 samples (80ms)
        # We'll collect audio in a buffer and resample
        chunk_samples = int(native_rate * 0.08)  # 80ms at native rate
        oww_chunk_size = 1280  # 80ms at 16kHz

        buf = deque(maxlen=int(native_rate * 2))  # 2s rolling buffer

        def callback(indata, frame_count, time_info, status):
            if self.active and not self.gui.recording:
                chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                buf.extend(chunk.tolist())

        def _open_stream():
            with self._stream_lock:
                if self._stream is not None:
                    return True  # Already open — don't re-create
                try:
                    self._stream = sd.InputStream(
                        samplerate=native_rate, channels=CHANNELS,
                        dtype="float32", device=mic_idx,
                        callback=callback,
                        blocksize=chunk_samples,
                    )
                    self._stream.start()
                    return True
                except Exception as e:
                    _log(f"Hotword stream error: {e}")
                    self._stream = None
                    return False

        if not _open_stream():
            self.active = False
            return

        last_check = 0

        while self.active:
            time.sleep(0.08)  # Check every 80ms (matches OWW chunk size)

            # Re-open stream after recording finishes
            should_reopen = False
            with self._stream_lock:
                if self._reopen and not self.gui.recording and self._stream is None:
                    self._reopen = False
                    should_reopen = True
            if should_reopen:
                buf.clear()
                _open_stream()
                _log("Hotword stream resumed")
                if self._model:
                    self._model.reset()
                self._resume_cooldown = time.monotonic() + 2.0  # 2s cooldown

            if self.gui.recording or not self._stream:
                continue

            # Skip detection while TTS is speaking (prevents feedback loop)
            tts = getattr(self.gui, '_tts', None)
            if tts and tts.is_speaking:
                buf.clear()
                continue

            # Skip detection during post-resume cooldown
            if time.monotonic() < getattr(self, '_resume_cooldown', 0):
                buf.clear()
                continue

            # Need at least 80ms of audio
            if len(buf) < chunk_samples:
                continue

            # Extract latest chunk and resample to 16kHz int16
            raw = np.array(list(buf)[-chunk_samples:], dtype=np.float32)

            if native_rate != 16000:
                from scipy.signal import resample as scipy_resample
                new_len = int(len(raw) * 16000 / native_rate)
                raw = scipy_resample(raw, new_len).astype(np.float32)

            # Convert float32 [-1, 1] to int16 for OWW
            audio_int16 = (raw * 32767).astype(np.int16)

            # Predict — ~1.5ms on CPU
            try:
                predictions = self._model.predict(audio_int16)
            except Exception:
                continue

            # Check hey_jarvis and hey_mycroft (similar sound)
            score = max(
                predictions.get("hey_jarvis", 0.0),
                predictions.get("hey_mycroft", 0.0) * 0.7,
            )

            # Log near-misses for debugging
            if score >= 0.1:
                _log(f"Hotword score: {score:.3f}")

            # Dual-threshold confirmation: need one frame above THRESHOLD
            # OR two frames above CONFIRM_THRESHOLD within CONFIRM_WINDOW
            now = time.monotonic()
            # Detection cooldown — skip processing if recently detected.
            # Non-blocking (monotonic timestamp) so the loop stays responsive
            # to self.active=False.
            if time.monotonic() < getattr(self, '_detection_cooldown', 0):
                continue

            if score >= self.THRESHOLD:
                # Strong detection — trigger immediately
                _log(f"Hotword detected (strong, score={score:.3f})")
                buf.clear()
                self._model.reset()
                self._pending_hotword = None
                self.gui.root.after(0, self._on_hotword)
                self._detection_cooldown = time.monotonic() + 1.5
            elif score >= self.CONFIRM_THRESHOLD:
                # Weak detection — need confirmation
                pending = getattr(self, '_pending_hotword', None)
                if pending and (now - pending) < self.CONFIRM_WINDOW:
                    # Second weak frame within window — confirmed
                    _log(f"Hotword confirmed (2 frames, score={score:.3f})")
                    buf.clear()
                    self._model.reset()
                    self._pending_hotword = None
                    self.gui.root.after(0, self._on_hotword)
                    self._detection_cooldown = time.monotonic() + 1.5
                else:
                    # First weak frame — start confirmation window
                    self._pending_hotword = now
            else:
                # Below both thresholds — clear pending
                if getattr(self, '_pending_hotword', None):
                    if (now - self._pending_hotword) > self.CONFIRM_WINDOW:
                        self._pending_hotword = None

    def _on_hotword(self):
        """Called when hotword is detected. Releases mic first so recording can use it."""
        if not self.gui.recording and self.gui.model_loaded:
            self.pause()
            self.gui._set_status("Hotword!", self.gui.BLUE, "Starting recording...")
            if self.gui.sound_var.get():
                threading.Thread(target=_play_beep, args=(_BEEP_START,), daemon=True).start()
            self.gui.root.after(200, self.gui._start_recording)


# ------------------------------------------------------------------
# Tooltip helper
# ------------------------------------------------------------------
class _Tooltip:
    """Hover tooltip for any widget."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tw = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule_show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule_show(self, event=None):
        self._hide()  # Kill any existing tooltip first
        self._after_id = self.widget.after(400, self._show)

    def _show(self, event=None):
        self._after_id = None
        if not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tw = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        label = tk.Label(
            tw, text=self.text, font=("Arial", 9),
            bg="#1a2332", fg="#d4e5f7", relief=tk.SOLID, borderwidth=1,
            padx=8, pady=4, wraplength=320, justify="left",
        )
        label.pack()
        # Auto-hide after 8 seconds as safety net
        self.widget.after(8000, self._hide)

    def _hide(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tw:
            self._tw.destroy()
            self._tw = None


# Setting descriptions (used for tooltips)
SETTING_TIPS = {
    "model": (
        "Whisper AI model size. Larger = more accurate but slower.\n"
        "  tiny: ~1GB VRAM, fastest, lower accuracy\n"
        "  base: ~1GB VRAM, fast, decent for clear speech\n"
        "  small: ~2GB VRAM, good balance (recommended)\n"
        "  medium: ~5GB VRAM, very accurate\n"
        "  large-v3: ~10GB VRAM, best accuracy"
    ),
    "gpu": (
        "Which GPU to run Whisper on.\n"
        "Default: GPU 1 (avoids conflict with YOLO on GPU 0).\n"
        "Set to 0 if you only have one GPU."
    ),
    "mic": (
        "Microphone input device.\n"
        "'Default' uses the system default mic.\n"
        "Use a specific device if the default picks up\n"
        "the wrong input (e.g. webcam vs USB mic)."
    ),
    "lang": (
        "Language for transcription.\n"
        "'English' is fastest and most accurate for English speech.\n"
        "'Auto-detect' lets Whisper guess the language\n"
        "(slightly slower, use for multilingual input)."
    ),
    "auto_type": (
        "Auto-type: Automatically types transcribed text into\n"
        "the active window using xdotool after transcription.\n"
        "Disable if you want to copy/paste manually instead."
    ),
    "review": (
        "Review mode: Shows transcription in an editable text box.\n"
        "You can fix mistakes before pressing Enter/Send to type it.\n"
        "Disables auto-type when enabled."
    ),
    "continuous": (
        "Continuous mode: Automatically restarts recording after\n"
        "each transcription. Great for dictating long text.\n"
        "Recording loops until you press Stop."
    ),
    "sounds": (
        "Sound feedback: Plays a beep when recording starts\n"
        "and stops, so you know the mic is active without\n"
        "looking at the screen."
    ),
    "voice_cmds": (
        "Voice commands: Converts spoken words to punctuation\n"
        "and actions. Say 'period', 'comma', 'new line',\n"
        "'question mark', 'delete that', etc.\n"
        "Disable for verbatim transcription."
    ),
    "noise_gate": (
        "Noise gate: Silences audio blocks below a volume\n"
        "threshold. Filters out keyboard clicks, fan noise,\n"
        "and ambient room sound for cleaner transcription."
    ),
    "streaming": (
        "Live preview: Shows partial transcription while you\n"
        "are still speaking. Uses a fast lightweight pass\n"
        "every 2 seconds. The final result may differ slightly."
    ),
    "hotword": (
        "Hey Claude: Always-on wake word detection.\n"
        "Say 'Hey Claude' to start recording hands-free.\n"
        "Uses continuous mic listening with lightweight\n"
        "inference — increases CPU/GPU usage when enabled."
    ),
    "smart_target": (
        "Smart target: Automatically finds open Claude Code\n"
        "terminal windows and types text into them.\n"
        "Detects Claude by its spinner character in the title.\n"
        "Falls back to the currently focused window if\n"
        "no Claude terminal is found."
    ),
    "reload": (
        "Reload model: Unloads the current Whisper model\n"
        "and loads the selected model/GPU. Use after\n"
        "changing the Model or GPU settings."
    ),
    "vocab": (
        "Edit vocabulary: Custom domain terms that bias\n"
        "Whisper toward recognizing warehouse-specific\n"
        "words like AGV, forklift, pallet, etc."
    ),
    "auto_enter": (
        "Auto-enter: Automatically presses Enter after\n"
        "typing the transcribed text. Useful for sending\n"
        "messages in Claude Code or chat windows without\n"
        "having to press Enter manually."
    ),
    "silence_timeout": (
        "Silence timeout: How many seconds of no speech\n"
        "before recording automatically stops and transcribes.\n"
        "Higher values give you more time to pause and think.\n"
        "Range: 1-10 seconds."
    ),
    "calibrate": (
        "Calibrate: Samples 3 seconds of background noise\n"
        "from your mic and sets the silence threshold just\n"
        "above it. Run this if recording doesn't auto-stop\n"
        "or stops too easily. Be quiet during calibration."
    ),
    "speaker_verify": (
        "Voice ID: Only transcribe audio that matches your\n"
        "enrolled voiceprint. Filters out TV, YouTube, and\n"
        "other people's voices. Click 'Enroll' to record\n"
        "your voice samples first (3-5 samples recommended)."
    ),
}


# ------------------------------------------------------------------
# Main GUI
# ------------------------------------------------------------------
class VoiceInputGUI:
    """Tkinter GUI for voice-to-text input."""

    # Color palette — Cyan Arc Reactor (Jarvis theme)
    BG = "#0a0e14"
    CARD_BG = "#1a2332"
    BORDER = "#1e3a4f"
    TEXT = "#d4e5f7"
    MUTED = "#4a6a8a"
    ACCENT = "#06b6d4"
    ACCENT_LIGHT = "#67e8f9"
    GREEN = "#3fb950"
    YELLOW = "#d29922"
    BLUE = "#06b6d4"

    def __init__(self, root, on_close_callback=None, auto_record=False):
        self.root = root
        self.on_close_callback = on_close_callback
        self._auto_record = auto_record  # Start recording as soon as model loads
        self._daemon_launched = auto_record  # Persistent: skip continuous restart
        self.root.title("Voice Input")
        self.root.configure(bg=self.BG)
        self.root.geometry("440x900")
        self.root.resizable(True, True)
        self.root.minsize(400, 700)

        # State
        self.recording = False
        self.processing = False
        self.model_loaded = False
        self.model_loading = False
        self._whisper_model = None
        self._stt_engine = None
        self._stream = None
        self._audio_frames = []
        self._silence_start = None
        self._audio_level = 0.0
        self._waveform_buffer = deque(maxlen=WAVEFORM_BARS)
        self._history = []
        self._session_log = []  # (timestamp, text) tuples
        self._partial_text = ""
        self._last_partial_time = 0
        self._partial_lock = threading.Lock()
        self._record_start_time = None
        self._last_audio = None  # Last recorded audio for playback
        self._voice_stopped = False  # Set by "end recording" to skip continuous restart
        self._target_wid = None  # Pinned window target (None = auto)
        self._target_name = None  # Display name of pinned target

        # Settings vars
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.gpu_var = tk.IntVar(value=DEFAULT_GPU)
        self.mic_var = tk.StringVar(value="Default")
        self.lang_var = tk.StringVar(value="English")
        self.auto_type_var = tk.BooleanVar(value=True)
        self.continuous_var = tk.BooleanVar(value=False)
        self.sound_var = tk.BooleanVar(value=True)
        self.review_var = tk.BooleanVar(value=False)
        self.voice_cmds_var = tk.BooleanVar(value=True)
        self.noise_gate_var = tk.BooleanVar(value=True)
        self.noise_suppress_var = tk.BooleanVar(value=True)
        self.streaming_var = tk.BooleanVar(value=True)
        self.hotword_var = tk.BooleanVar(value=False)
        self.smart_target_var = tk.BooleanVar(value=True)
        self.auto_enter_var = tk.BooleanVar(value=False)
        self.live_write_var = tk.BooleanVar(value=False)
        self.talkback_var = tk.BooleanVar(value=False)
        self.jarvis_mode_var = tk.BooleanVar(value=False)  # Claude brain mode
        self.tts_engine_var = tk.StringVar(value="edge")  # "edge" or "xtts"
        self.speaker_verify_var = tk.BooleanVar(value=False)
        self.speaker_threshold_var = tk.DoubleVar(value=0.35)
        self.silence_var = tk.DoubleVar(value=SILENCE_TIMEOUT)
        self.noise_threshold_var = tk.DoubleVar(value=SILENCE_THRESHOLD)
        self._last_segments = []  # (text, avg_logprob) for confidence display

        self._mic_devices = {}
        self._detect_mics()

        # Language lookup
        self._lang_map = {name: code for name, code in LANGUAGES}

        # Load persisted settings (overrides defaults above)
        self._load_settings()

        _init_beeps()

        self._configure_ttk_theme()
        self._build_ui()
        self._bind_keys()

        # Auto-save settings on any change
        self._setup_auto_save()

        # Show voiceprint count if file exists
        _vp_file = Path.home() / ".aiws_trainer" / "voiceprint.npz"
        if _vp_file.exists():
            try:
                data = np.load(_vp_file)
                n = len(data.files)
                self._voiceprint_label.config(
                    text=f"{n} sample{'s' if n != 1 else ''}",
                    fg=self.GREEN,
                )
            except Exception as e:
                _log(f"Voiceprint label load error: {e}")

        self._load_model_async()

        # System tray
        self._tray = TrayIcon(self)
        self._tray.start()

        # Global hotkey (Ctrl+Shift+V system-wide)
        self._global_hotkey = GlobalHotkey(self)
        self._global_hotkey.start()

        # Hotword listener (started on demand via checkbox)
        self._hotword = HotwordListener(self)

        # Silero VAD for speech detection (replaces RMS silence threshold)
        from jarvis.audio_pipeline import SileroVAD
        self._silero_vad = SileroVAD()

        # Jarvis TTS
        self._tts = None

        # Intent classifier — learns from user feedback
        self._intent_classifier = IntentClassifier()

        # Jarvis Agent — intelligence layer
        from jarvis.jarvis_agent import JarvisAgent
        self._agent = JarvisAgent()

        # Jarvis Brain — Claude-powered intelligence
        from jarvis.jarvis_brain import JarvisBrain
        self._brain = JarvisBrain()

        # Speaker verification (lazy-loaded when enabled)
        self._speaker_verifier = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # SIGTERM handler so _cleanup runs even when minimized to tray
        # (tray path skips _cleanup intentionally for fast re-wake, so
        # without this the hotword stream + hotkey context leak on kill).
        import signal
        def _sigterm(*_):
            _log("SIGTERM received; running _cleanup()")
            try:
                self._cleanup()
            finally:
                os._exit(0)
        try:
            signal.signal(signal.SIGTERM, _sigterm)
        except (ValueError, OSError):
            # Not on main thread or signal not supported; skip
            pass

        # Start speak queue watcher for talk-back
        self._start_speak_queue_watcher()

        # Start proactive monitoring (GPU temp, disk space alerts)
        if self.talkback_var.get():
            self._agent.start_monitoring(speak_func=True)

        # Pre-load TTS engine in background if talk-back is on
        if self.talkback_var.get():
            def _preload_tts():
                tts = self._get_tts()
                tts.load()
            threading.Thread(target=_preload_tts, daemon=True).start()
        self.root.bind("<Destroy>", self._on_window_destroy)

    # ------------------------------------------------------------------
    # Mic detection
    # ------------------------------------------------------------------
    def _detect_mics(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self._mic_devices = {"Default": None}
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    name = f"[{i}] {d['name']}"
                    self._mic_devices[name] = i
        except Exception as e:
            _log(f"Mic detection error: {e}")
            self._mic_devices = {"Default": None}

    def _get_mic_raw_name(self):
        """Get raw device name (without index prefix) for persistence."""
        display = self.mic_var.get()
        if display == "Default":
            return None
        if display.startswith("["):
            bracket_end = display.find("]")
            if bracket_end > 0:
                return display[bracket_end + 2:].strip()
        return display

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------
    def _load_settings(self):
        """Load settings from JSON file, applying over defaults."""
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            _log(f"Loading settings from {SETTINGS_FILE}")

            mapping = {
                "model": (self.model_var, str),
                "gpu": (self.gpu_var, int),
                "mic": (self.mic_var, str),
                "language": (self.lang_var, str),
                "auto_type": (self.auto_type_var, bool),
                "continuous": (self.continuous_var, bool),
                "sound": (self.sound_var, bool),
                "review": (self.review_var, bool),
                "voice_cmds": (self.voice_cmds_var, bool),
                "noise_gate": (self.noise_gate_var, bool),
                "noise_suppression": (self.noise_suppress_var, bool),
                "streaming": (self.streaming_var, bool),
                "hotword": (self.hotword_var, bool),
                "smart_target": (self.smart_target_var, bool),
                "auto_enter": (self.auto_enter_var, bool),
                "live_write": (self.live_write_var, bool),
                "talkback": (self.talkback_var, bool),
                "jarvis_mode": (self.jarvis_mode_var, bool),
                "tts_engine": (self.tts_engine_var, str),
                "speaker_verify": (self.speaker_verify_var, bool),
                "speaker_threshold": (self.speaker_threshold_var, float),
                "silence_timeout": (self.silence_var, float),
                # target_name loaded manually below
                "noise_threshold": (self.noise_threshold_var, float),
            }

            for key, (var, typ) in mapping.items():
                if key in data:
                    try:
                        var.set(typ(data[key]))
                    except (ValueError, TypeError):
                        pass

            # Resolve mic by name (survives reboots/USB re-enumeration)
            saved_mic_name = data.get("mic_name")
            if saved_mic_name:
                from jarvis.audio_pipeline import resolve_mic_by_name
                import sounddevice as sd
                idx = resolve_mic_by_name(saved_mic_name, sd.query_devices())
                if idx is not None:
                    for display, dev_idx in self._mic_devices.items():
                        if dev_idx == idx:
                            self.mic_var.set(display)
                            _log(f"Mic resolved: '{saved_mic_name}' -> [{idx}]")
                            break
                else:
                    _log(f"WARNING: Mic '{saved_mic_name}' not found")
                    self.root.after(1000, lambda n=saved_mic_name: self._set_status(
                        "Mic not found", "#da3633", f"'{n}' — check USB"))

            # Restore pinned target window name
            saved_target = data.get("target_name")
            if saved_target:
                self._target_name = saved_target
                # Will try to find matching window after UI is built
                self.root.after(2000, lambda: self._restore_target(saved_target))

        except Exception as e:
            _log(f"Settings load error: {e}")

    def _save_settings(self, *_args):
        """Save current settings to JSON file."""
        data = {
            "model": self.model_var.get(),
            "gpu": self.gpu_var.get(),
            "mic": self.mic_var.get(),
            "mic_name": self._get_mic_raw_name(),
            "language": self.lang_var.get(),
            "auto_type": self.auto_type_var.get(),
            "continuous": self.continuous_var.get(),
            "sound": self.sound_var.get(),
            "review": self.review_var.get(),
            "voice_cmds": self.voice_cmds_var.get(),
            "noise_gate": self.noise_gate_var.get(),
            "noise_suppression": self.noise_suppress_var.get(),
            "streaming": self.streaming_var.get(),
            "hotword": self.hotword_var.get(),
            "smart_target": self.smart_target_var.get(),
            "auto_enter": self.auto_enter_var.get(),
            "live_write": self.live_write_var.get(),
            "talkback": self.talkback_var.get(),
            "jarvis_mode": self.jarvis_mode_var.get(),
            "tts_engine": self.tts_engine_var.get(),
            "speaker_verify": self.speaker_verify_var.get(),
            "speaker_threshold": self.speaker_threshold_var.get(),
            "target_name": self._target_name,
            "silence_timeout": self.silence_var.get(),
            "noise_threshold": self.noise_threshold_var.get(),
        }
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: write to .tmp then rename so a crash mid-write
            # cannot leave a truncated settings.json that fails to parse.
            tmp_path = SETTINGS_FILE.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2))
            os.replace(tmp_path, SETTINGS_FILE)
        except Exception as e:
            _log(f"Settings save error: {e}")

    def _setup_auto_save(self):
        """Register trace callbacks so any setting change auto-saves."""
        for var in (
            self.model_var, self.mic_var, self.lang_var,
            self.auto_type_var, self.continuous_var, self.sound_var,
            self.review_var, self.voice_cmds_var, self.noise_gate_var, self.noise_suppress_var,
            self.streaming_var, self.hotword_var, self.smart_target_var,
            self.auto_enter_var, self.live_write_var,
            self.talkback_var, self.jarvis_mode_var, self.tts_engine_var,
            self.speaker_verify_var, self.speaker_threshold_var,
            self.silence_var, self.noise_threshold_var, self.gpu_var,
        ):
            var.trace_add("write", self._save_settings)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------
    def _bind_keys(self):
        self.root.bind("<F5>", lambda e: self._toggle_recording())
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", lambda e: self._minimize_to_tray())
        self.root.bind("<Return>", self._on_enter)

    def _on_space(self, event):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        self._toggle_recording()

    def _on_enter(self, event):
        """In review mode, Enter sends the text."""
        if self.review_var.get() and not self.recording and not self.processing:
            self._send_reviewed_text()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _configure_ttk_theme(self):
        """Apply holographic dark theme to ttk widgets (combobox, scrollbar)."""
        style = ttk.Style()
        style.theme_use("clam")  # clam is the most customizable ttk theme

        # Combobox
        style.configure("TCombobox",
            fieldbackground="#0c1822",
            background="#132637",
            foreground="#67e8f9",
            arrowcolor="#06b6d4",
            bordercolor="#1a3050",
            lightcolor="#1a3050",
            darkcolor="#1a3050",
            selectbackground="#1a3050",
            selectforeground="#67e8f9",
        )
        style.map("TCombobox",
            fieldbackground=[("readonly", "#0c1822"), ("focus", "#0e1e2e")],
            foreground=[("readonly", "#67e8f9")],
            bordercolor=[("focus", "#06b6d4")],
            arrowcolor=[("focus", "#67e8f9")],
        )
        # Combobox dropdown list
        self.root.option_add("*TCombobox*Listbox.background", "#0c1822")
        self.root.option_add("*TCombobox*Listbox.foreground", "#67e8f9")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#1a3050")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#67e8f9")

        # Scrollbar
        style.configure("TScrollbar",
            background="#132637",
            troughcolor="#0a0e14",
            arrowcolor="#06b6d4",
            bordercolor="#1a3050",
        )

    def _holo_btn(self, parent, text, command, **kwargs):
        """Create a holographic-styled button with cyan border highlight."""
        btn = tk.Button(
            parent, text=text, font=kwargs.pop("font", ("Arial", 8)),
            bg="#0c1822", fg="#67e8f9",
            activebackground="#132637", activeforeground="#67e8f9",
            relief=tk.FLAT, cursor="hand2", bd=0,
            highlightthickness=1, highlightbackground="#1a3050",
            highlightcolor="#06b6d4",
            command=command, **kwargs,
        )
        return btn

    def _build_ui(self):
        # Main scrollable area
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=self.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self._scroll_frame = tk.Frame(canvas, bg=self.BG)

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        self._canvas_window = canvas.create_window(
            (0, 0), window=self._scroll_frame, anchor="nw"
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        # Stretch inner frame to canvas width on resize
        def _on_canvas_configure(event):
            canvas.itemconfigure(self._canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_linux_scroll(event):
            if event.num == 4:
                canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                canvas.yview_scroll(3, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_linux_scroll)
        canvas.bind_all("<Button-5>", _on_linux_scroll)

        parent = self._scroll_frame

        # Header — JARVIS branding
        header = tk.Frame(parent, bg=self.BG)
        header.pack(fill="x", padx=24, pady=(18, 0))

        tk.Label(
            header, text="JARVIS",
            font=("Arial", 18, "bold"), bg=self.BG, fg=self.ACCENT,
        ).pack(side="left")

        tk.Label(
            header, text="VOICE ASSISTANT",
            font=("Arial", 9), bg=self.BG, fg="#1e3a4f",
        ).pack(side="left", padx=(10, 0), pady=(4, 0))

        min_btn = tk.Button(
            header, text="\u2014", font=("Arial", 10),
            bg=self.BG, fg="#1e3a4f", relief=tk.FLAT,
            cursor="hand2", width=3, bd=0,
            command=self._minimize_to_tray,
        )
        min_btn.pack(side="right")

        # Status bar — rounded feel with glow dot
        status_frame = tk.Frame(parent, bg=self.CARD_BG,
                                highlightbackground=self.BORDER, highlightthickness=1)
        status_frame.pack(fill="x", padx=24, pady=(14, 0))

        status_inner = tk.Frame(status_frame, bg=self.CARD_BG)
        status_inner.pack(fill="x", padx=16, pady=10)

        self.status_dot = tk.Label(
            status_inner, text="\u25CF", font=("Arial", 12),
            bg=self.CARD_BG, fg=self.YELLOW,
        )
        self.status_dot.pack(side="left", padx=(0, 8))

        self.status_label = tk.Label(
            status_inner, text="Loading model...",
            font=("Arial", 11, "bold"), bg=self.CARD_BG, fg=self.TEXT,
        )
        self.status_label.pack(side="left")

        self.status_detail = tk.Label(
            status_inner, text="",
            font=("Arial", 8), bg=self.CARD_BG, fg=self.MUTED,
        )
        self.status_detail.pack(side="right")

        # Record button — holographic outlined
        btn_frame = tk.Frame(parent, bg=self.BG)
        btn_frame.pack(fill="x", padx=24, pady=(16, 0))

        self.record_btn = tk.Button(
            btn_frame, text="R E C O R D",
            font=("Arial", 14, "bold"),
            bg="#0c1822", fg=self.MUTED,
            activebackground="#0c1822", activeforeground=self.MUTED,
            relief=tk.FLAT, cursor="hand2",
            state=tk.DISABLED, bd=0, pady=14,
            highlightthickness=1, highlightbackground="#1a3050",
            highlightcolor="#06b6d4",
            command=self._toggle_recording,
        )
        self.record_btn.pack(fill="x")

        tk.Label(
            btn_frame, text="F5 \u00b7 SPACE \u00b7 \"JARVIS\"",
            font=("Arial", 7), bg=self.BG, fg="#1e3a4f",
        ).pack(pady=(4, 0))

        # Waveform display — Particle Orbit
        wave_frame = tk.Frame(parent, bg=self.BG)
        wave_frame.pack(fill="x", padx=24, pady=(10, 0))

        self.live_preview_label = tk.Label(
            wave_frame, text="",
            font=("Arial", 9, "italic"), bg=self.BG, fg="#4a6a8a",
        )
        self.live_preview_label.pack(fill="x", pady=(0, 2))

        self._orbit_size = 220
        self.waveform_canvas = tk.Canvas(
            wave_frame, width=self._orbit_size, height=self._orbit_size,
            bg="#0a0e14", highlightbackground=self.BORDER, highlightthickness=1,
        )
        self.waveform_canvas.pack(pady=(0, 4))

        # PIL-rendered orbit — single image on canvas
        from PIL import ImageTk
        self._orbit_time = 0.0
        self._orbit_img_id = self.waveform_canvas.create_image(
            self._orbit_size // 2, self._orbit_size // 2, anchor="center",
        )
        self._orbit_photo = None
        self._orbit_pending = None
        self._orbit_rendering = False

        # Particle state
        self._orbit_particles = []
        for ring in range(5):
            count = 12 + ring * 6
            for i in range(count):
                self._orbit_particles.append({
                    "ring": ring,
                    "angle": (i / count) * math.pi * 2 + ring * 0.3,
                    "base_r": 18 + ring * 18,
                    "speed": (0.3 + ring * 0.1) * (1 if i % 2 == 0 else -1),
                    "size": 1.5 + (i % 3) * 0.5,
                    "phase": (i * 0.7) % (math.pi * 2),
                })

        self._orbit_bg = (10, 14, 20)

        # Pre-render the arc reactor center as a static image (never changes shape,
        # only brightness) — huge perf win, no numpy glow per frame
        self._prerender_reactor_frames()

        # Start idle animation
        self._update_orbit_idle()

        # --- Transcription: You ---
        you_frame = tk.Frame(parent, bg=self.CARD_BG,
                             highlightbackground=self.BORDER, highlightthickness=1)
        you_frame.pack(fill="x", padx=24, pady=(12, 0))

        you_header = tk.Frame(you_frame, bg=self.CARD_BG)
        you_header.pack(fill="x", padx=12, pady=(8, 0))

        tk.Label(
            you_header, text="You",
            font=("Arial", 10, "bold"), bg=self.CARD_BG, fg=self.GREEN,
        ).pack(side="left")

        # Review mode: Send button
        self.send_btn = self._holo_btn(
            you_header, "Send", self._send_reviewed_text,
            font=("Arial", 9, "bold"),
        )
        # Hidden by default, shown when review mode is on

        self.play_btn = self._holo_btn(
            you_header, "\u25B6 Play", self._play_last_audio,
        )
        self.play_btn.pack(side="right", padx=(4, 0))

        self.copy_btn = self._holo_btn(
            you_header, "Copy", self._copy_last,
        )
        self.copy_btn.pack(side="right", padx=(4, 0))

        self.output_text = tk.Text(
            you_frame, font=("Consolas", 10),
            bg="#0a0e14", fg=self.TEXT,
            insertbackground=self.TEXT,
            relief=tk.FLAT, wrap=tk.WORD,
            padx=12, pady=6, height=3,
        )
        self.output_text.pack(fill="x", padx=8, pady=(4, 4))
        self.output_text.config(state=tk.DISABLED)

        # Text tags for confidence coloring
        self.output_text.tag_configure("conf_high", foreground=self.GREEN)
        self.output_text.tag_configure("conf_med", foreground=self.YELLOW)
        self.output_text.tag_configure("conf_low", foreground="#da3633")

        # --- Transcription: Jarvis ---
        jarvis_frame = tk.Frame(parent, bg=self.CARD_BG,
                                highlightbackground=self.BORDER, highlightthickness=1)
        jarvis_frame.pack(fill="x", padx=24, pady=(6, 0))

        jarvis_header = tk.Frame(jarvis_frame, bg=self.CARD_BG)
        jarvis_header.pack(fill="x", padx=12, pady=(8, 0))

        tk.Label(
            jarvis_header, text="Jarvis",
            font=("Arial", 10, "bold"), bg=self.CARD_BG, fg=self.ACCENT,
        ).pack(side="left")

        self.jarvis_copy_btn = self._holo_btn(
            jarvis_header, "Copy", self._copy_jarvis,
        )
        self.jarvis_copy_btn.pack(side="right", padx=(4, 0))

        self.jarvis_text = tk.Text(
            jarvis_frame, font=("Consolas", 10),
            bg="#0a0e14", fg=self.ACCENT,
            insertbackground=self.ACCENT,
            relief=tk.FLAT, wrap=tk.WORD,
            padx=12, pady=6, height=3,
        )
        self.jarvis_text.pack(fill="x", padx=8, pady=(4, 4))
        self.jarvis_text.config(state=tk.DISABLED)

        # Confidence bar
        conf_bar = tk.Frame(you_frame, bg=self.CARD_BG)
        conf_bar.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(
            conf_bar, text="Confidence", font=("Arial", 8),
            bg=self.CARD_BG, fg="#4a6a8a",
        ).pack(side="left")

        self.conf_canvas = tk.Canvas(
            conf_bar, height=8, bg="#0d1a28", highlightthickness=0,
        )
        self.conf_canvas.pack(side="left", fill="x", expand=True, padx=(6, 0))

        self.conf_label = tk.Label(
            conf_bar, text="", font=("Arial", 8),
            bg=self.CARD_BG, fg="#4a6a8a",
        )
        self.conf_label.pack(side="right", padx=(6, 0))

        # Collapsible settings toggle
        self._settings_visible = False
        toggle_frame = tk.Frame(parent, bg=self.BG)
        toggle_frame.pack(fill="x", padx=24, pady=(10, 0))

        self._settings_arrow = tk.Label(
            toggle_frame, text="\u25B6", font=("Arial", 9),
            bg=self.BG, fg="#4a6a8a", cursor="hand2",
        )
        self._settings_arrow.pack(side="left")

        settings_toggle_label = tk.Label(
            toggle_frame, text="Settings",
            font=("Arial", 9), bg=self.BG, fg="#4a6a8a", cursor="hand2",
        )
        settings_toggle_label.pack(side="left", padx=(4, 0))

        for w in (self._settings_arrow, settings_toggle_label):
            w.bind("<Button-1>", lambda e: self._toggle_settings())

        # Settings container (hidden by default)
        self._settings_container = tk.Frame(parent, bg=self.BG)
        # Don't pack yet — starts hidden

        settings_frame = tk.Frame(self._settings_container, bg=self.CARD_BG,
                                  highlightbackground=self.BORDER, highlightthickness=1)
        settings_frame.pack(fill="x", padx=24, pady=(6, 0))

        settings_inner = tk.Frame(settings_frame, bg=self.CARD_BG)
        settings_inner.pack(fill="x", padx=16, pady=10)

        # Row 1: Model + GPU + Reload
        row1 = tk.Frame(settings_inner, bg=self.CARD_BG)
        row1.pack(fill="x", pady=(0, 5))

        lbl_model = tk.Label(row1, text="Model", font=("Arial", 10),
                             bg=self.CARD_BG, fg=self.MUTED)
        lbl_model.pack(side="left")
        _Tooltip(lbl_model, SETTING_TIPS["model"])
        cb_model = ttk.Combobox(
            row1, textvariable=self.model_var,
            values=["tiny", "base", "small", "medium", "large-v3"],
            state="readonly", width=9,
        )
        cb_model.pack(side="left", padx=(6, 10))
        _Tooltip(cb_model, SETTING_TIPS["model"])

        lbl_gpu = tk.Label(row1, text="GPU", font=("Arial", 10),
                           bg=self.CARD_BG, fg=self.MUTED)
        lbl_gpu.pack(side="left")
        _Tooltip(lbl_gpu, SETTING_TIPS["gpu"])
        spn_gpu = tk.Spinbox(
            row1, textvariable=self.gpu_var, from_=0, to=3,
            width=3, font=("Arial", 10),
            bg="#0c1822", fg="#67e8f9", buttonbackground="#132637",
            highlightbackground="#1a3050", highlightcolor="#06b6d4",
            highlightthickness=1, relief=tk.FLAT, bd=0,
            insertbackground="#67e8f9",
        )
        spn_gpu.pack(side="left", padx=(6, 10))
        _Tooltip(spn_gpu, SETTING_TIPS["gpu"])

        reload_btn = self._holo_btn(row1, "Reload", self._reload_model)
        reload_btn.pack(side="right")
        _Tooltip(reload_btn, SETTING_TIPS["reload"])

        # Row 2: Mic + Language
        row2 = tk.Frame(settings_inner, bg=self.CARD_BG)
        row2.pack(fill="x", pady=(0, 5))

        lbl_mic = tk.Label(row2, text="Mic", font=("Arial", 10),
                           bg=self.CARD_BG, fg=self.MUTED)
        lbl_mic.pack(side="left")
        _Tooltip(lbl_mic, SETTING_TIPS["mic"])
        cb_mic = ttk.Combobox(
            row2, textvariable=self.mic_var,
            values=list(self._mic_devices.keys()),
            state="readonly", width=24,
        )
        cb_mic.pack(side="left", padx=(6, 10))
        _Tooltip(cb_mic, SETTING_TIPS["mic"])

        test_mic_btn = self._holo_btn(row2, "Test Mic", self._test_mic)
        test_mic_btn.pack(side="left", padx=(0, 10))
        _Tooltip(test_mic_btn,
                 "Records 3s and plays back through speakers.\n"
                 "Verify you hear YOUR voice, not system audio.")

        lbl_lang = tk.Label(row2, text="Lang", font=("Arial", 10),
                            bg=self.CARD_BG, fg=self.MUTED)
        lbl_lang.pack(side="left")
        _Tooltip(lbl_lang, SETTING_TIPS["lang"])
        cb_lang = ttk.Combobox(
            row2, textvariable=self.lang_var,
            values=[name for name, _ in LANGUAGES],
            state="readonly", width=11,
        )
        cb_lang.pack(side="left", padx=(6, 0))
        _Tooltip(cb_lang, SETTING_TIPS["lang"])

        # Section label
        tk.Label(
            settings_inner, text="SETTINGS",
            font=("Arial", 7), bg=self.CARD_BG, fg="#1e3a4f",
        ).pack(anchor="w", pady=(6, 4))

        # Row 3: Toggles line 1
        row3 = tk.Frame(settings_inner, bg=self.CARD_BG)
        row3.pack(fill="x", pady=(0, 4))

        chk_style = dict(
            font=("Arial", 8), bg="#0c1822", fg="#3a5a7a",
            selectcolor="#0e2a40", activebackground="#0e2a40",
            activeforeground="#67e8f9",
            indicatoron=0, padx=10, pady=4,
            bd=0, relief=tk.FLAT, overrelief=tk.FLAT,
            highlightthickness=1, highlightbackground="#1a3050",
            highlightcolor="#06b6d4",
        )

        chk_auto = tk.Checkbutton(
            row3, text="Auto-type", variable=self.auto_type_var,
            command=self._on_auto_type_toggle, **chk_style,
        )
        chk_auto.pack(side="left", padx=(0, 4))
        _Tooltip(chk_auto, SETTING_TIPS["auto_type"])

        chk_review = tk.Checkbutton(
            row3, text="Review", variable=self.review_var,
            command=self._on_review_toggle, **chk_style,
        )
        chk_review.pack(side="left", padx=(0, 4))
        _Tooltip(chk_review, SETTING_TIPS["review"])

        chk_cont = tk.Checkbutton(
            row3, text="Continuous", variable=self.continuous_var, **chk_style,
        )
        chk_cont.pack(side="left", padx=(0, 4))
        _Tooltip(chk_cont, SETTING_TIPS["continuous"])

        chk_snd = tk.Checkbutton(
            row3, text="Sounds", variable=self.sound_var, **chk_style,
        )
        chk_snd.pack(side="left", padx=(0, 4))
        _Tooltip(chk_snd, SETTING_TIPS["sounds"])

        chk_enter = tk.Checkbutton(
            row3, text="Auto-enter", variable=self.auto_enter_var, **chk_style,
        )
        chk_enter.pack(side="left", padx=(0, 4))
        _Tooltip(chk_enter, SETTING_TIPS["auto_enter"])

        # Row 4: Toggles line 2
        row4 = tk.Frame(settings_inner, bg=self.CARD_BG)
        row4.pack(fill="x", pady=(0, 2))

        chk_live = tk.Checkbutton(
            row4, text="Auto-write", variable=self.live_write_var, **chk_style,
        )
        chk_live.pack(side="left", padx=(0, 4))
        _Tooltip(chk_live,
                 "Auto-write: Types words into the target window\n"
                 "in real-time as you speak. The final transcription\n"
                 "replaces the partial text when recording stops.")

        chk_talk = tk.Checkbutton(
            row4, text="Talk-back", variable=self.talkback_var, **chk_style,
        )
        chk_talk.pack(side="left", padx=(0, 4))
        _Tooltip(chk_talk,
                 "Talk-back: Jarvis reads Claude's responses aloud.\n"
                 "Toggle the voice engine with the button next to this.")

        self._tts_engine_btn = self._holo_btn(
            row4, "Edge", self._toggle_tts_engine,
        )
        self._tts_engine_btn.pack(side="left", padx=(0, 4))
        self._update_tts_engine_btn()
        _Tooltip(self._tts_engine_btn,
                 "Toggle voice engine:\n"
                 "Edge = fast (~1s), British neural voice\n"
                 "JARVIS = slower (~2s), cloned JARVIS voice")

        add_clip_btn = self._holo_btn(
            row4, "Add Voice Clip", self._add_voice_clip,
        )
        add_clip_btn.pack(side="left", padx=(0, 4))
        _Tooltip(add_clip_btn,
                 "Paste a YouTube URL with clean JARVIS dialogue\n"
                 "to improve the XTTS voice clone quality.")

        # Row 5: Jarvis Mode
        row5_mode = tk.Frame(settings_inner, bg=self.CARD_BG)
        row5_mode.pack(fill="x", pady=(4, 0))

        chk_jmode = tk.Checkbutton(
            row5_mode, text="Jarvis Mode", variable=self.jarvis_mode_var,
            **chk_style,
        )
        chk_jmode.pack(side="left", padx=(0, 4))
        _Tooltip(chk_jmode,
                 "Jarvis Mode: Claude IS Jarvis's brain.\n"
                 "Messages go to Claude CLI instead of typing\n"
                 "into a terminal. Jarvis thinks, speaks, and\n"
                 "executes actions autonomously.")

        chk_vcmd = tk.Checkbutton(
            row4, text="Voice cmds", variable=self.voice_cmds_var, **chk_style,
        )
        chk_vcmd.pack(side="left", padx=(0, 4))
        _Tooltip(chk_vcmd, SETTING_TIPS["voice_cmds"])

        chk_ng = tk.Checkbutton(
            row4, text="Noise gate", variable=self.noise_gate_var, **chk_style,
        )
        chk_ng.pack(side="left", padx=(0, 4))
        _Tooltip(chk_ng, SETTING_TIPS["noise_gate"])

        chk_denoise = tk.Checkbutton(
            row4, text="Denoise", variable=self.noise_suppress_var, **chk_style,
        )
        chk_denoise.pack(side="left", padx=(0, 4))
        _Tooltip(chk_denoise,
                 "Removes TV/music/ambient noise before transcription.\n"
                 "Uses spectral gating. Default: ON.")

        chk_stream = tk.Checkbutton(
            row4, text="Live preview", variable=self.streaming_var, **chk_style,
        )
        chk_stream.pack(side="left", padx=(0, 4))
        _Tooltip(chk_stream, SETTING_TIPS["streaming"])

        chk_hw = tk.Checkbutton(
            row4, text="Hey Claude", variable=self.hotword_var,
            command=self._on_hotword_toggle, **chk_style,
        )
        chk_hw.pack(side="left", padx=(0, 4))
        _Tooltip(chk_hw, SETTING_TIPS["hotword"])

        # Voice ID section label
        tk.Label(
            settings_inner, text="VOICE ID",
            font=("Arial", 7), bg=self.CARD_BG, fg="#1e3a4f",
        ).pack(anchor="w", pady=(8, 4))

        # Row 4a: Voice ID (speaker verification)
        row4a = tk.Frame(settings_inner, bg=self.CARD_BG)
        row4a.pack(fill="x", pady=(4, 2))

        chk_spk = tk.Checkbutton(
            row4a, text="Voice ID", variable=self.speaker_verify_var,
            command=self._on_speaker_verify_toggle, **chk_style,
        )
        chk_spk.pack(side="left", padx=(0, 4))
        _Tooltip(chk_spk, SETTING_TIPS["speaker_verify"])

        self._enroll_btn = self._holo_btn(
            row4a, "Enroll Voice", self._start_enrollment,
        )
        self._enroll_btn.pack(side="left", padx=(0, 4))
        _Tooltip(self._enroll_btn,
                 "Record a voice sample to build your voiceprint.\n"
                 "Speak normally for 10-20 seconds. Repeat 3-5 times\n"
                 "for best accuracy. Your voiceprint is stored locally.")

        self._voiceprint_label = tk.Label(
            row4a, text="No voiceprint", font=("Arial", 9),
            bg=self.CARD_BG, fg=self.MUTED,
        )
        self._voiceprint_label.pack(side="left", padx=(4, 0))

        clear_vp_btn = self._holo_btn(row4a, "Clear", self._clear_voiceprint)
        clear_vp_btn.pack(side="left", padx=(4, 0))
        _Tooltip(clear_vp_btn, "Delete all enrolled voice samples")

        train_ww_btn = self._holo_btn(
            row4a, "Train Wake Word", self._start_wakeword_training)
        train_ww_btn.pack(side="left", padx=(4, 0))
        _Tooltip(train_ww_btn,
                 "Record yourself saying 'Hey Jarvis' 10 times\n"
                 "to train a custom wake word model tuned to\n"
                 "YOUR voice. Much more accurate than the default.")

        # Voice ID confidence threshold slider
        row4a2 = tk.Frame(settings_inner, bg=self.CARD_BG)
        row4a2.pack(fill="x", pady=(0, 2))

        lbl_conf = tk.Label(
            row4a2, text="Voice ID confidence", font=("Arial", 9),
            bg=self.CARD_BG, fg=self.MUTED,
        )
        lbl_conf.pack(side="left")
        _Tooltip(lbl_conf,
                 "How strict speaker matching should be.\n"
                 "Lower = more permissive (may let others through)\n"
                 "Higher = stricter (may reject you if voice differs)\n"
                 "0.25 = relaxed, 0.40 = balanced, 0.60 = strict")

        self._spk_thresh_label = tk.Label(
            row4a2, text=f"{self.speaker_threshold_var.get():.2f}",
            font=("Arial", 9, "bold"), bg=self.CARD_BG, fg=self.TEXT,
            width=4,
        )
        self._spk_thresh_label.pack(side="right", padx=(4, 0))

        spk_slider = tk.Scale(
            row4a2, from_=0.15, to=0.70, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.speaker_threshold_var,
            bg=self.CARD_BG, fg=self.TEXT, highlightthickness=0,
            troughcolor="#0d1a28", sliderrelief=tk.FLAT,
            showvalue=False, length=120,
            command=self._on_speaker_threshold_change,
        )
        spk_slider.pack(side="right")
        _Tooltip(spk_slider,
                 "Drag to adjust. Test with Voice ID enabled:\n"
                 "status bar shows match score for each recording.")

        # Row 4b: Window target
        row4b = tk.Frame(settings_inner, bg=self.CARD_BG)
        row4b.pack(fill="x", pady=(4, 2))

        lbl_tgt = tk.Label(
            row4b, text="Target", font=("Arial", 10),
            bg=self.CARD_BG, fg=self.MUTED,
        )
        lbl_tgt.pack(side="left")
        _Tooltip(lbl_tgt, SETTING_TIPS["smart_target"])

        self._target_display = tk.Label(
            row4b, text="Auto (Claude)",
            font=("Arial", 9, "bold"), bg="#0d1a28", fg=self.GREEN,
            padx=6, pady=2,
        )
        self._target_display.pack(side="left", padx=(6, 4))
        _Tooltip(self._target_display,
                 "Current type target. 'Auto' finds Claude terminals.\n"
                 "Click 'Pick' to select a specific window.")

        pick_btn = self._holo_btn(row4b, "Pick", self._pick_window)
        pick_btn.pack(side="left", padx=(0, 4))
        _Tooltip(pick_btn,
                 "Click, then click on any window to set it as\n"
                 "the type target. Your cursor becomes a crosshair.")

        self._window_combo_var = tk.StringVar(value="")
        self._window_combo = ttk.Combobox(
            row4b, textvariable=self._window_combo_var,
            state="readonly", width=16,
        )
        self._window_combo.pack(side="left", padx=(0, 4))
        self._window_combo.bind("<<ComboboxSelected>>", self._on_window_selected)
        _Tooltip(self._window_combo,
                 "Select a window from the list.\n"
                 "Click the refresh button to update.")

        refresh_btn = self._holo_btn(
            row4b, "\u21BB", self._refresh_window_list, font=("Arial", 9),
        )
        refresh_btn.pack(side="left", padx=(0, 4))
        _Tooltip(refresh_btn, "Refresh the list of open windows")

        reset_btn = self._holo_btn(row4b, "Auto", self._reset_target)
        reset_btn.pack(side="left")
        _Tooltip(reset_btn, "Reset target to Auto (Claude terminal)")

        # Populate window list on startup
        self.root.after(1000, self._refresh_window_list)

        # Row 5: Vocab editor button
        row5 = tk.Frame(settings_inner, bg=self.CARD_BG)
        row5.pack(fill="x", pady=(4, 0))

        vocab_btn = self._holo_btn(row5, "Edit Vocabulary", self._open_vocab_editor)
        vocab_btn.pack(side="left")
        _Tooltip(vocab_btn, SETTING_TIPS["vocab"])

        # Row 6: Silence timeout + Noise calibration
        row6 = tk.Frame(settings_inner, bg=self.CARD_BG)
        row6.pack(fill="x", pady=(4, 0))

        lbl_silence = tk.Label(row6, text="Silence", font=("Arial", 10),
                               bg=self.CARD_BG, fg=self.MUTED)
        lbl_silence.pack(side="left")
        _Tooltip(lbl_silence, SETTING_TIPS["silence_timeout"])

        self._silence_value_label = tk.Label(
            row6, text=f"{self.silence_var.get():.1f}s",
            font=("Arial", 9), bg=self.CARD_BG, fg=self.TEXT, width=4,
        )
        self._silence_value_label.pack(side="left", padx=(2, 0))

        silence_scale = tk.Scale(
            row6, from_=1.0, to=10.0, resolution=0.5, orient=tk.HORIZONTAL,
            variable=self.silence_var, showvalue=False, length=100,
            bg=self.CARD_BG, fg=self.TEXT, troughcolor="#0d1a28",
            highlightthickness=0, sliderlength=15,
            command=lambda v: self._silence_value_label.config(text=f"{float(v):.1f}s"),
        )
        silence_scale.pack(side="left", padx=(4, 8))
        _Tooltip(silence_scale, SETTING_TIPS["silence_timeout"])

        # Noise threshold display
        lbl_thresh = tk.Label(row6, text="Noise", font=("Arial", 10),
                              bg=self.CARD_BG, fg=self.MUTED)
        lbl_thresh.pack(side="left")

        self._thresh_label = tk.Label(
            row6, text=f"{self.noise_threshold_var.get():.3f}",
            font=("Arial", 9, "bold"), bg=self.CARD_BG, fg=self.GREEN,
        )
        self._thresh_label.pack(side="left", padx=(2, 4))

        calibrate_btn = self._holo_btn(row6, "Calibrate", self._calibrate_noise)
        calibrate_btn.pack(side="left")
        _Tooltip(calibrate_btn, SETTING_TIPS["calibrate"])

        # History + export (inside collapsible container)
        hist_frame = tk.Frame(self._settings_container, bg=self.BG)
        hist_frame.pack(fill="x", padx=24, pady=(8, 0))

        hist_header = tk.Frame(hist_frame, bg=self.BG)
        hist_header.pack(fill="x")

        tk.Label(
            hist_header, text="History", font=("Arial", 9, "bold"),
            bg=self.BG, fg=self.MUTED,
        ).pack(side="left")

        self.log_count_label = tk.Label(
            hist_header, text="0 entries",
            font=("Arial", 8), bg=self.BG, fg="#4a6a8a",
        )
        self.log_count_label.pack(side="left", padx=(8, 0))

        self._holo_btn(hist_header, "Export", self._export_log).pack(side="right")

        self.history_label = tk.Label(
            hist_frame, text="No transcriptions yet",
            font=("Arial", 9), bg=self.BG, fg="#4a6a8a",
            anchor="w", justify="left",
        )
        self.history_label.pack(fill="x")

        # Footer
        tk.Label(
            parent,
            text="F5 / Space = Record  |  Esc = Minimize  |  Enter = Send (review)  |  Ctrl+Shift+V = Global (active)",
            font=("Arial", 8), bg=self.BG, fg="#4a6a8a",
        ).pack(pady=(6, 10))

    # ------------------------------------------------------------------
    # Toggle callbacks
    # ------------------------------------------------------------------
    def _toggle_settings(self):
        """Show/hide the settings panel."""
        if self._settings_visible:
            self._settings_container.pack_forget()
            self._settings_arrow.config(text="\u25B6")  # right arrow
            self._settings_visible = False
        else:
            self._settings_container.pack(fill="x", after=self._settings_arrow.master)
            self._settings_arrow.config(text="\u25BC")  # down arrow
            self._settings_visible = True

    def _on_review_toggle(self):
        """When review mode is toggled, update UI state."""
        if self.review_var.get():
            self.auto_type_var.set(False)
            self.output_text.config(state=tk.NORMAL, bg="#0d1a2e")
            self.send_btn.pack(side="right", padx=(4, 0))
        else:
            self.output_text.config(state=tk.DISABLED, bg="#0a0e14")
            self.send_btn.pack_forget()

    def _on_auto_type_toggle(self):
        """When auto-type is enabled, disable review mode."""
        if self.auto_type_var.get():
            self.review_var.set(False)
            self.output_text.config(state=tk.DISABLED, bg="#0a0e14")
            self.send_btn.pack_forget()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model_async(self):
        self.model_loading = True
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        try:
            from jarvis.stt_engine import STTEngine

            gpu = self.gpu_var.get()
            _log(f"Loading STT engine on GPU {gpu}")

            self._stt_engine = STTEngine(gpu=gpu)
            loaded = self._stt_engine.load()
            if not loaded:
                raise RuntimeError("No STT engine available")

            engine_name = self._stt_engine.engine_name
            self.model_loaded = True
            self.model_loading = False
            _log(f"STT engine loaded: {engine_name}")

            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, f"{engine_name} on CUDA:{gpu}"))
            self.root.after(0, lambda: self.record_btn.config(
                state=tk.NORMAL, bg="#0c1822", fg="#67e8f9",
                activebackground="#132637", activeforeground="#67e8f9"))

            # Pre-load speaker verifier if Voice ID enabled (before auto-record)
            if self.speaker_verify_var.get():
                verifier = self._get_speaker_verifier()
                verifier.load_model()
                self.root.after(0, self._update_voiceprint_label)

            # Auto-record mode (launched by hotword daemon)
            if self._auto_record:
                self._auto_record = False  # One-shot
                # Don't start hotword yet — it will start after recording finishes
                self.root.after(200, self._start_recording)
                _log("Auto-record triggered by hotword daemon")
            elif self.hotword_var.get():
                # Normal launch: start hotword listener
                self.root.after(100, self._hotword.start)
                _log("Auto-started hotword listener from saved settings")

        except Exception as e:
            self.model_loading = False
            _log(f"Model load error: {e}")
            self.root.after(0, lambda: self._set_status(
                f"Error: {e}", "#da3633", ""))

    def _reload_model(self):
        if self.recording or self.processing:
            return
        self._whisper_model = None
        self.model_loaded = False
        self.record_btn.config(state=tk.DISABLED, bg="#132637", fg=self.MUTED)
        self._set_status("Loading model...", self.YELLOW, "")
        self._load_model_async()

    def _test_mic(self):
        """Record 3 seconds from selected mic and play back."""
        if self.recording:
            self._set_status("Stop recording first", self.YELLOW, "")
            return
        if self.hotword_var.get() and self._hotword._stream:
            self._hotword.pause()

        mic_idx = self._mic_devices.get(self.mic_var.get())
        self._set_status("Testing mic...", self.ACCENT, "Speak now (3 seconds)")

        def _worker():
            import sounddevice as sd
            import wave, tempfile, os
            try:
                dev_info = sd.query_devices(mic_idx, 'input')
                rate = int(dev_info['default_samplerate'])
            except Exception:
                rate = 44100
            try:
                audio = sd.rec(int(3 * rate), samplerate=rate,
                               channels=1, dtype='float32', device=mic_idx)
                sd.wait()
                rms = float(np.sqrt(np.mean(audio ** 2)))
                _log(f"Test mic: RMS={rms:.4f}")
                self.root.after(0, lambda: self._set_status(
                    "Playing back...", self.ACCENT, f"RMS: {rms:.4f}"))
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(rate)
                    wf.writeframes((audio * 32767).astype(np.int16).tobytes())
                for cmd in [["paplay", tmp.name], ["aplay", "-q", tmp.name]]:
                    try:
                        subprocess.run(cmd, timeout=10, check=True,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                        break
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue
                os.unlink(tmp.name)
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, f"Mic test done (RMS: {rms:.4f})"))
            except Exception as e:
                _log(f"Test mic error: {e}")
                self.root.after(0, lambda: self._set_status(
                    "Error", "#da3633", str(e)[:40]))
            finally:
                if self.hotword_var.get():
                    self.root.after(0, self._hotword.resume)

        threading.Thread(target=_worker, daemon=True).start()

    def _calibrate_noise(self):
        """Sample background noise for 3 seconds and set threshold above it."""
        if self.recording:
            self._set_status("Stop recording first", self.YELLOW, "")
            return

        # Pause hotword listener so it doesn't hold the mic
        if self.hotword_var.get() and self._hotword._stream:
            self._hotword.pause()

        self._set_status("Calibrating...", self.BLUE, "Be quiet for 3 seconds")
        threading.Thread(target=self._calibrate_worker, daemon=True).start()

    def _calibrate_worker(self):
        """Background thread: sample mic for 3s, compute noise floor, set threshold."""
        import sounddevice as sd

        mic_name = self.mic_var.get()
        mic_idx = self._mic_devices.get(mic_name)

        try:
            dev_info = sd.query_devices(mic_idx, 'input')
            native_rate = int(dev_info['default_samplerate'])
        except Exception:
            native_rate = 44100

        samples = []

        def cb(indata, frames, t, status):
            rms = float(np.sqrt(np.mean(indata ** 2)))
            samples.append(rms)

        try:
            stream = sd.InputStream(
                samplerate=native_rate, channels=CHANNELS,
                dtype="float32", device=mic_idx,
                callback=cb, blocksize=int(native_rate * 0.1),
            )
            stream.start()
            time.sleep(3.0)
            stream.stop()
            stream.close()
        except Exception as e:
            _log(f"Calibration error: {e}")
            self.root.after(0, lambda: self._set_status(
                "Error", "#da3633", f"Calibration failed: {e}"))
            # Resume hotword
            if self.hotword_var.get():
                self.root.after(0, self._hotword.resume)
            return

        if not samples:
            self.root.after(0, lambda: self._set_status(
                "Error", "#da3633", "No audio captured"))
            if self.hotword_var.get():
                self.root.after(0, self._hotword.resume)
            return

        arr = np.array(samples)
        p99 = float(np.percentile(arr, 99))
        # Set threshold 50% above the 99th percentile of background noise
        new_threshold = round(p99 * 1.5, 4)
        # Clamp to reasonable range
        new_threshold = max(0.01, min(0.15, new_threshold))

        _log(f"Calibration: mean={arr.mean():.4f} p99={p99:.4f} → threshold={new_threshold}")

        def _apply():
            self.noise_threshold_var.set(new_threshold)
            self._thresh_label.config(text=f"{new_threshold:.3f}")
            self._set_status("Ready", self.GREEN,
                             f"Noise floor: {p99:.4f} → threshold: {new_threshold}")
            # Resume hotword
            if self.hotword_var.get():
                self._hotword.resume()

        self.root.after(0, _apply)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _toggle_recording(self):
        if self.processing:
            return
        if not self.model_loaded:
            return
        if not self.recording:
            self._start_recording()
        else:
            self._voice_stopped = True  # Manual stop — don't auto-restart
            self._stop_and_transcribe()

    def _start_recording(self):
        import sounddevice as sd

        # Guard against rapid restart loop
        if self.recording:
            return
        if hasattr(self, '_last_record_start'):
            elapsed = time.monotonic() - self._last_record_start
            if elapsed < 1.0:
                return
        self._last_record_start = time.monotonic()

        # Pause hotword listener to free the mic device
        if self.hotword_var.get() and self._hotword._stream:
            self._hotword.pause()

        mic_name = self.mic_var.get()
        mic_idx = self._mic_devices.get(mic_name)

        # If saved mic no longer exists, try resolving by name
        if mic_name not in self._mic_devices:
            from jarvis.audio_pipeline import resolve_mic_by_name
            import sounddevice as sd
            raw_name = self._get_mic_raw_name()
            resolved = resolve_mic_by_name(raw_name, sd.query_devices()) if raw_name else None
            if resolved is not None:
                mic_idx = resolved
                _log(f"Mic resolved by name: '{raw_name}' -> [{resolved}]")
            else:
                _log(f"WARNING: Mic not found, recording disabled")
                self._set_status("Mic not found", "#da3633", "Check USB connection")
                return

        # Detect the device's native sample rate (many mics only do 44100/48000)
        try:
            dev_info = sd.query_devices(mic_idx, 'input')
            native_rate = int(dev_info['default_samplerate'])
        except Exception:
            native_rate = 44100
        self._record_rate = native_rate
        _log(f"Mic native rate: {native_rate}Hz")

        self._audio_frames = []
        self._silence_start = None
        self._loud_chunks = 0
        self._speaker_silence_start = None  # Voice-ID silence tracker
        self._last_speaker_check = 0        # Monotonic time of last check
        self._voice_id_match = False         # Visual indicator for orbit
        self._live_typed_text = ""           # What's been typed so far (live writing)
        self._live_typed_chars = 0           # Character count typed into terminal
        self._waveform_buffer.clear()
        self._partial_text = ""
        self._last_partial_time = time.monotonic()

        # Cache threshold for audio thread (Tkinter vars aren't thread-safe)
        silence_thresh = self.noise_threshold_var.get()
        noise_suppress_enabled = self.noise_suppress_var.get()

        def audio_callback(indata, frame_count, time_info, status):
            if not self.recording:
                return

            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()

            # Measure RMS on original audio BEFORE noise gate (for silence detection)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            self._audio_level = min(rms * 10, 1.0)

            # Silence detection uses original signal
            # Only reset silence timer on sustained sound (2+ consecutive loud chunks)
            # This prevents single noise spikes from restarting the 3s countdown
            if rms < silence_thresh:
                self._loud_chunks = 0
                if self._silence_start is None:
                    self._silence_start = time.monotonic()
            else:
                self._loud_chunks = getattr(self, '_loud_chunks', 0) + 1
                if self._loud_chunks >= 2:
                    self._silence_start = None

            # Noise gate: zero out quiet blocks (after silence check)
            if self.noise_gate_var.get() and rms < NOISE_GATE_THRESHOLD:
                chunk[:] = 0.0

            # Noise suppression (spectral gating — removes TV/ambient)
            if noise_suppress_enabled and len(chunk) >= 1600:
                from jarvis.audio_pipeline import denoise_audio
                chunk = denoise_audio(chunk.flatten(), sr=native_rate).reshape(-1, 1)

            self._audio_frames.append(chunk.reshape(-1, 1))

            # Waveform samples
            step = max(1, len(chunk) // 4)
            for val in chunk[::step]:
                self._waveform_buffer.append(float(val))

        # Set recording state and start animation BEFORE opening mic
        # (mic open can block briefly and cause frame stutter)
        self.recording = True
        self._record_start_time = time.monotonic()
        self.record_btn.config(
            text="\u23F9  S T O P  0:00", bg="#0c1822", fg="#22d3ee",
            activebackground="#132637",
        )
        self._set_status("Listening...", self.ACCENT, "Speak now")
        self._tray.update_state(True)

        if self.sound_var.get():
            threading.Thread(target=_play_beep, args=(_BEEP_START,), daemon=True).start()

        self._update_waveform()
        self._check_silence()
        self._silero_vad.reset()
        self._update_timer()

        # Open mic stream (may block briefly — animation already running)
        try:
            self._stream = sd.InputStream(
                samplerate=native_rate, channels=CHANNELS,
                dtype="float32", device=mic_idx,
                callback=audio_callback,
                blocksize=int(native_rate * 0.1),
            )
            self._stream.start()
        except Exception as e:
            _log(f"Mic open error: {e}")
            self.recording = False
            self._set_status("Mic Error", "#da3633", str(e)[:50])
            self._reset_button()
            return

        # Start streaming partial transcription
        if self.streaming_var.get():
            self._stream_partial()

        _log("Recording started")

    def _resample_to_16k(self, audio):
        """Resample audio from recording rate to 16kHz for Whisper."""
        rate = getattr(self, '_record_rate', SAMPLE_RATE)
        if rate == SAMPLE_RATE:
            return audio
        from scipy.signal import resample
        new_len = int(len(audio) * SAMPLE_RATE / rate)
        resampled = resample(audio, new_len).astype(np.float32)
        return resampled

    def _stop_and_transcribe(self):
        if not self.recording:
            return

        self.recording = False
        self._audio_level = 0.0
        self._tray.update_state(False)

        if self.sound_var.get():
            threading.Thread(target=_play_beep, args=(_BEEP_STOP,), daemon=True).start()

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self.live_preview_label.config(text="")

        # Snapshot audio frames (callback thread may still be draining)
        frames = list(self._audio_frames)
        if not frames:
            self._set_status("Ready", self.GREEN, "No audio captured")
            self._reset_button()
            return

        try:
            audio_raw = np.concatenate(frames, axis=0).flatten()
        except ValueError:
            _log("Error: audio frames empty after snapshot")
            self._set_status("Ready", self.GREEN, "No audio captured")
            self._reset_button()
            return
        self._last_audio = audio_raw.copy()
        audio = self._resample_to_16k(audio_raw)
        duration = len(audio) / SAMPLE_RATE
        _log(f"Stopped: {duration:.1f}s audio")

        # Cap audio at 60 seconds to prevent freezes
        max_samples = SAMPLE_RATE * 60
        if len(audio) > max_samples:
            _log(f"Audio capped from {duration:.1f}s to 60s")
            audio = audio[:max_samples]
            duration = 60.0

        if duration < 0.3:
            self._set_status("Ready", self.GREEN, "Too short")
            self._reset_button()
            return

        # Apply noise gate to final audio
        if self.noise_gate_var.get():
            audio = _apply_noise_gate(audio)

        self.processing = True
        self.record_btn.config(
            text="TRANSCRIBING...", state=tk.DISABLED,
            bg="#0c1822", fg=self.MUTED,
        )
        self._set_status("Transcribing...", self.YELLOW, f"{duration:.1f}s audio")

        threading.Thread(
            target=self._transcribe_worker, args=(audio,), daemon=True
        ).start()

    def _get_whisper_language(self):
        """Get the language code for Whisper from current selection."""
        lang_name = self.lang_var.get()
        return self._lang_map.get(lang_name, "en")

    def _transcribe_worker(self, audio):
        try:
            # Speaker verification — filter to only user's voice segments
            # Skip segment filter if the periodic speaker check already
            # confirmed our voice during recording (saves ~0.5s)
            if self.speaker_verify_var.get() and self._speaker_verifier is not None:
                duration_sec = len(audio) / SAMPLE_RATE
                voice_confirmed = getattr(self, '_voice_id_match', False)

                if voice_confirmed and duration_sec < 15:
                    # Short recording, voice already verified — skip segment filter
                    _log(f"Speaker skip: voice confirmed during recording, "
                         f"{duration_sec:.1f}s < 15s")
                else:
                    filtered, stats = self._speaker_verifier.filter_segments(audio)
                    if filtered is None:
                        avg = (sum(stats["scores"]) / len(stats["scores"])
                               if stats["scores"] else 0.0)
                        _log(f"Speaker rejected all {stats['total']} segments "
                             f"(avg score={avg:.3f})")
                        self.root.after(0, lambda a=avg: self._on_transcription(
                            "", None, speaker_rejected=True, speaker_score=a))
                        return
                    if stats["total"] > 0:
                        kept_pct = stats["matched"] / stats["total"] * 100
                        orig_sec = len(audio) / SAMPLE_RATE
                        filt_sec = len(filtered) / SAMPLE_RATE
                        _log(f"Speaker filter: kept {stats['matched']}/{stats['total']} "
                             f"segments ({kept_pct:.0f}%), "
                             f"{orig_sec:.1f}s -> {filt_sec:.1f}s audio")
                    audio = filtered

            # Transcribe using STT engine (Parakeet or Whisper fallback)
            with self._partial_lock:
                result = self._stt_engine.transcribe(audio)
            text = result.text
            seg_data = result.segments

            # Confidence gate — reject low-confidence transcriptions
            if seg_data:
                avg_conf = sum(lp for _, lp in seg_data) / len(seg_data)
                _log(f"Transcribed: {text!r} (avg_logprob={avg_conf:.2f})")
                if avg_conf < -1.5:
                    _log(f"Rejected: confidence too low ({avg_conf:.2f} < -1.5)")
                    self.root.after(0, lambda a=avg_conf: self._on_transcription(
                        "", None, speaker_rejected=True,
                        speaker_score=a))
                    return
            else:
                _log(f"Transcribed: {text!r}")

            # Dictation mode — type directly, don't send to Claude
            if getattr(self, '_dictation_mode', False):
                if "end dictation" in text.lower():
                    self._dictation_mode = False
                    _log("Dictation mode ended")
                    self.root.after(0, lambda: self._show_jarvis_text(
                        "Dictation mode: OFF"))
                    self.root.after(0, lambda: self._on_transcription(
                        "[Dictation ended]", None))
                    return
                # Type text directly without submitting
                if self.auto_type_var.get():
                    threading.Thread(
                        target=lambda t=text: subprocess.run(
                            ["xdotool", "type", "--clearmodifiers", "--delay", "5", t + " "],
                            timeout=10, capture_output=True,
                        ), daemon=True,
                    ).start()
                self.root.after(0, lambda t=text: self._on_transcription(t, seg_data))
                return

            # Desktop control commands ("Jarvis, switch to opera and scroll down")
            if self._check_desktop_command(text):
                self.root.after(0, lambda t=text, s=seg_data: self._on_transcription(t, s))
                return

            # Quick voice commands ("Jarvis, commit" etc.)
            if self._check_quick_command(text):
                self.root.after(0, lambda t=text, s=seg_data: self._on_transcription(t, s))
                return

            # Intent check — is this directed at the assistant?
            intent, conf = self._intent_classifier.classify(text)
            if intent == IntentClassifier.NO:
                _log(f"Ignored (background chat, conf={conf:.2f}): {text!r}")
                self.root.after(0, lambda: self._on_transcription(
                    "", None, speaker_rejected=True, speaker_score=0.0))
                return
            elif intent == IntentClassifier.UNCERTAIN:
                _log(f"Uncertain intent (conf={conf:.2f}): {text!r}")
                self.root.after(0, lambda t=text, s=seg_data: (
                    self._show_intent_prompt(t, s)))
                return

            # Apply voice commands
            if self.voice_cmds_var.get():
                text = _apply_voice_commands(text)

            # Strip stop-recording phrases from end of text
            for phrase in STOP_RECORDING_PHRASES:
                idx = text.lower().rfind(phrase)
                if idx >= 0 and idx > len(text) - len(phrase) - 5:
                    text = text[:idx].rstrip(" ,.-")
                    break

            # Check for screenshot trigger — strip phrase, flag for after typing
            wants_screenshot = False
            text_lower = text.lower()
            for phrase in SCREENSHOT_PHRASES:
                idx = text_lower.rfind(phrase)
                if idx >= 0:
                    text = (text[:idx] + text[idx + len(phrase):]).strip().rstrip(" ,.-")
                    wants_screenshot = True
                    _log(f"Screenshot requested, remaining text: {text!r}")
                    break

            self.root.after(0, lambda: self._on_transcription(
                text, seg_data, screenshot=wants_screenshot))
        except Exception as e:
            _log(f"Transcription error: {e}")
            self.root.after(0, lambda: self._on_transcription_error(str(e)))

    # ------------------------------------------------------------------
    # Streaming partial transcription
    # ------------------------------------------------------------------
    def _stream_partial(self):
        """Periodically transcribe accumulated audio for a live preview."""
        if not self.recording or not self.streaming_var.get():
            return
        # Parakeet is fast enough that partial preview is unnecessary;
        # stop the reschedule loop entirely rather than burning a slot
        # every 500ms while the engine is active.
        if self._stt_engine and "Parakeet" in (self._stt_engine.engine_name or ""):
            return

        now = time.monotonic()
        if now - self._last_partial_time >= STREAMING_INTERVAL and len(self._audio_frames) > 5:
            self._last_partial_time = now
            # Grab a copy of current audio and resample for Whisper
            try:
                audio_raw = np.concatenate(self._audio_frames, axis=0).flatten()
                rate = getattr(self, '_record_rate', SAMPLE_RATE)
                if len(audio_raw) > rate * 0.5:  # At least 0.5s
                    audio_16k = self._resample_to_16k(audio_raw)
                    threading.Thread(
                        target=self._partial_transcribe_worker,
                        args=(audio_16k,),
                        daemon=True,
                    ).start()
            except Exception:
                pass

        self.root.after(500, self._stream_partial)

    def _partial_transcribe_worker(self, audio):
        """Quick transcription for live preview (no VAD, smaller beam).

        Also detects voice commands (targeting, actions) mid-speech
        and executes them immediately without waiting for recording to stop.
        """
        with self._partial_lock:
            try:
                # Route through the unified STT engine (Parakeet primary,
                # Whisper fallback). Skip cleanly if STT isn't loaded yet
                # so the preview worker can't crash with AttributeError.
                if self._stt_engine is None or not self._stt_engine.is_loaded:
                    return
                result = self._stt_engine.transcribe(audio)
                text = (result.text or "").strip()

                if text and self.recording:
                    self._partial_text = text

                    # Check for real-time voice commands
                    if self.voice_cmds_var.get() and self._check_realtime_command(text):
                        return  # Command handled, recording stopped

                    # Check for filler sounds — reset silence timer (user is thinking)
                    # Only match if the LAST word is exactly a filler sound
                    last_word = text.lower().split()[-1].strip(".,!?") if text.strip() else ""
                    if last_word in FILLER_WORDS:
                        self._silence_start = None
                        self._loud_chunks = 0
                        _log(f"Filler detected, silence timer reset: {last_word!r}")

                    # Show preview (truncated)
                    preview = text[-60:] if len(text) > 60 else text
                    self.root.after(0, lambda: self.live_preview_label.config(
                        text=f"... {preview}"))

                    # Live writing — type new words into target
                    if self.live_write_var.get() and self.auto_type_var.get():
                        self.root.after(0, lambda t=text: self._live_type_partial(t))
            except Exception:
                pass

    def _check_realtime_command(self, text):
        """Check partial transcription for commands to execute immediately.

        Returns True if a command was detected and handled (recording will stop).
        """
        text_lower = text.strip().lower().rstrip(".")

        # Check for target reset phrases
        if text_lower in TARGET_RESET_PHRASES:
            _log(f"Realtime command: target reset from {text!r}")
            self.root.after(0, self._abort_and_reset_target)
            return True

        # Check for targeting commands
        match = TARGET_PATTERN.match(text_lower)
        if match:
            query = match.group(1).strip().rstrip(".")
            _log(f"Realtime command: target {query!r} from {text!r}")
            self.root.after(0, lambda: self._abort_and_target(query))
            return True

        # Check for stop-recording phrases ("end recording", "stop recording")
        for phrase in STOP_RECORDING_PHRASES:
            if text_lower.endswith(phrase):
                _log(f"Realtime command: stop recording from {text!r}")
                self._voice_stopped = True  # Skip continuous restart
                self.root.after(0, self._stop_and_transcribe)
                return True

        # Check for action commands (delete that, clear all, etc.)
        for phrase, action in ACTION_COMMANDS.items():
            if text_lower == phrase or text_lower.endswith(phrase):
                _log(f"Realtime command: action {action} from {text!r}")
                self.root.after(0, lambda a=action: self._abort_and_action(a))
                return True

        return False

    def _abort_recording(self):
        """Stop recording without transcribing (command already handled)."""
        self.recording = False
        self._audio_level = 0.0
        self._tray.update_state(False)
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.live_preview_label.config(text="")
        self._reset_button()
        # Resume hotword listener now that mic is free
        if self.hotword_var.get():
            self._hotword.resume()

    def _abort_and_reset_target(self):
        """Abort recording and reset target to auto."""
        self._abort_recording()
        self._reset_target()

    def _abort_and_target(self, query):
        """Abort recording and switch target to matching window."""
        self._abort_recording()
        self._voice_target_window(query)

    def _abort_and_action(self, action):
        """Abort recording and execute an action command."""
        self._abort_recording()
        self._handle_action(action)

    # ------------------------------------------------------------------
    # Transcription result handling
    # ------------------------------------------------------------------
    def _on_transcription(self, text, seg_data=None, screenshot=False,
                          speaker_rejected=False, speaker_score=0.0):
        self.processing = False
        self._reset_button()
        # Resume hotword listener now that mic is free
        if self.hotword_var.get():
            self._hotword.resume()

        # Speaker verification rejected this audio
        if speaker_rejected:
            self._set_status("Ready", self.YELLOW,
                             f"Voice ID: not you (score={speaker_score:.2f})")
            self._maybe_continuous_restart()
            return

        # Handle voice targeting commands
        if text and self.voice_cmds_var.get():
            text_lower = text.strip().lower().rstrip(".")
            if text_lower in TARGET_RESET_PHRASES:
                self._reset_target()
                self._maybe_continuous_restart()
                return
            match = TARGET_PATTERN.match(text_lower)
            if match:
                query = match.group(1).strip().rstrip(".")
                self._voice_target_window(query)
                self._maybe_continuous_restart()
                return

        # Handle action commands
        if text.startswith("__ACTION__"):
            action = text.replace("__ACTION__", "")
            self._handle_action(action)
            self._maybe_continuous_restart()
            return

        if text:
            # Show in output box
            if self.review_var.get():
                # Review mode: editable (no confidence coloring)
                self.output_text.config(state=tk.NORMAL, bg="#0d1a2e")
                self.output_text.delete("1.0", tk.END)
                self.output_text.insert("1.0", text)
                self._set_status("Review", self.BLUE, "Edit and press Enter/Send")
                return
            elif seg_data:
                # Show with confidence coloring
                self._display_with_confidence(seg_data)
            else:
                self.output_text.config(state=tk.NORMAL)
                self.output_text.delete("1.0", tk.END)
                self.output_text.insert("1.0", text)
                self.output_text.config(state=tk.DISABLED)

            # Log to session
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._session_log.append((ts, text))
            self.log_count_label.config(text=f"{len(self._session_log)} entries")

            # Add to history
            ts_short = datetime.now().strftime("%H:%M")
            self._history.insert(0, f"[{ts_short}] {text[:55]}{'...' if len(text) > 55 else ''}")
            self._history = self._history[:8]
            self.history_label.config(text="\n".join(self._history))

            # Log command for habit learning
            self._agent.log_command(text[:50])

            # Jarvis Mode — send to Claude brain instead of terminal
            if self.jarvis_mode_var.get():
                self._set_status("Thinking...", self.ACCENT, "Jarvis is thinking")
                self._speaking_animation = True
                self.root.after(0, self._update_speaking_animation)
                self._brain.think(text, callback=self._on_brain_response)
                self._maybe_continuous_restart()
                return

            # Intent enhancement — add context if relevant
            type_text = text
            enhanced = self._agent.interpret_intent(text)
            if enhanced:
                type_text = enhanced
                _log(f"Intent enhanced: +{len(enhanced) - len(text)} chars context")

            # Auto-type, then screenshot if requested
            if self.auto_type_var.get():
                if screenshot:
                    threading.Thread(
                        target=self._type_then_screenshot, args=(type_text,), daemon=True
                    ).start()
                else:
                    threading.Thread(
                        target=self._type_text, args=(type_text,), daemon=True
                    ).start()
            elif screenshot:
                # No auto-type but screenshot requested
                threading.Thread(target=self._take_screenshot, daemon=True).start()

            self._set_status("Ready", self.GREEN, f"Typed {len(text)} chars")

            # Passive voiceprint learning — add confirmed recording
            if (self.speaker_verify_var.get()
                    and self._speaker_verifier is not None
                    and self._speaker_verifier.enrolled
                    and self._last_audio is not None):
                audio_16k = self._resample_to_16k(self._last_audio)
                threading.Thread(
                    target=self._speaker_verifier.add_sample,
                    args=(audio_16k,), daemon=True,
                ).start()

            self._maybe_continuous_restart()
        else:
            self._set_status("Ready", self.GREEN, "No speech detected")
            # Don't auto-restart on empty — prevents infinite loop

    def _show_intent_prompt(self, text, seg_data):
        """Show 'Was this for me?' prompt for uncertain intent."""
        self.processing = False
        self._reset_button()
        if self.hotword_var.get():
            self._hotword.resume()

        # Store pending text for if user says Yes
        self._pending_intent_text = text
        self._pending_intent_seg_data = seg_data

        # Show text in output box
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.config(state=tk.DISABLED)

        # Show intent prompt in status area
        self._set_status("Was this for me?", self.YELLOW, text[:50])

        # Create Yes/No buttons in a prompt frame
        if hasattr(self, '_intent_prompt_frame'):
            self._intent_prompt_frame.destroy()

        self._intent_prompt_frame = tk.Frame(
            self._scroll_frame, bg=self.CARD_BG,
            highlightbackground=self.ACCENT, highlightthickness=1,
        )
        self._intent_prompt_frame.pack(fill="x", padx=24, pady=(6, 0),
                                       after=self.waveform_canvas.master)

        tk.Label(
            self._intent_prompt_frame, text="Was this meant for me?",
            font=("Arial", 10, "bold"), bg=self.CARD_BG, fg=self.TEXT,
        ).pack(side="left", padx=(12, 8), pady=8)

        self._holo_btn(
            self._intent_prompt_frame, "Yes — Send it",
            lambda: self._resolve_intent(True),
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=(0, 6), pady=8)

        self._holo_btn(
            self._intent_prompt_frame, "No — Background",
            lambda: self._resolve_intent(False),
        ).pack(side="left", padx=(0, 12), pady=8)

    def _resolve_intent(self, is_for_assistant):
        """Handle user's Yes/No response to intent prompt."""
        text = getattr(self, '_pending_intent_text', "")
        seg_data = getattr(self, '_pending_intent_seg_data', None)

        # Log feedback for learning
        self._intent_classifier.log_feedback(text, is_for_assistant)
        n = self._intent_classifier.num_examples
        _log(f"Intent feedback: {'YES' if is_for_assistant else 'NO'} "
             f"for {text!r} (total logged: {n})")

        # Remove prompt UI
        if hasattr(self, '_intent_prompt_frame'):
            self._intent_prompt_frame.destroy()

        if is_for_assistant:
            # Send the text as if it was a normal transcription
            self._on_transcription(text, seg_data)
        else:
            self._set_status("Ready", self.GREEN, "Ignored — learned for next time")

    def _on_transcription_error(self, error):
        self.processing = False
        self._reset_button()
        self._set_status("Error", "#da3633", error[:40])
        # Resume hotword listener now that mic is free
        if self.hotword_var.get():
            self._hotword.resume()

    def _handle_action(self, action):
        """Handle special voice command actions."""
        if action == "delete_last_sentence":
            # Remove last typed text via backspaces
            _log("Action: delete last sentence")
            self._set_status("Ready", self.GREEN, "Deleted last input")
        elif action == "clear_all":
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.config(state=tk.DISABLED)
            self._set_status("Ready", self.GREEN, "Cleared")
        elif action == "select_all":
            self._set_status("Ready", self.GREEN, "Select all")
        elif action == "screenshot":
            _log("Action: take screenshot")
            self._set_status("Capturing...", self.BLUE, "Taking screenshot")
            threading.Thread(target=self._take_screenshot, daemon=True).start()

    def _check_quick_command(self, text):
        """Check if text is a quick voice command. Returns True if handled."""
        lower = text.strip().lower().rstrip(".")

        # Check "jarvis, <command>" pattern
        for prefix in ("jarvis ", "jarvis, ", "hey jarvis ", "hey jarvis, "):
            if lower.startswith(prefix):
                cmd_text = lower[len(prefix):].strip()
                break
        else:
            return False

        # Check for "go back" (contextual memory)
        if cmd_text in ("go back", "previous window", "last window"):
            prev = self._agent.get_last_window()
            if prev:
                _log(f"Go back to: {prev}")
                threading.Thread(
                    target=self._execute_desktop_actions,
                    args=([("window", prev)],), daemon=True,
                ).start()
                return True

        # Check for "click on <text>" (screen awareness)
        click_match = re.match(r"click (?:on |the )?(.+)", cmd_text)
        if click_match:
            target = click_match.group(1).strip()
            _log(f"Click on text: {target}")
            self._set_status("Looking...", self.ACCENT, f"Finding '{target}'")
            threading.Thread(
                target=self._agent_click_text,
                args=(target,), daemon=True,
            ).start()
            return True

        # Check for "what's on screen" / "describe screen"
        if any(p in cmd_text for p in ("what's on screen", "describe screen",
                                        "what do you see", "look at screen")):
            info = self._agent.analyze_screen()
            if info:
                self._show_jarvis_text(f"Active: {info['active_window']}")
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say(f"You are currently in {info['active_window']}.")
            return True

        # Check for workflow commands: "deploy", "morning", "training check"
        workflow = (self._agent.get_workflow(cmd_text) or
                    self._agent.DEFAULT_WORKFLOWS.get(cmd_text))
        if workflow:
            _log(f"Workflow: {cmd_text} ({len(workflow)} steps)")
            self._set_status("Workflow...", self.ACCENT, cmd_text)
            threading.Thread(
                target=self._run_workflow,
                args=(cmd_text, workflow), daemon=True,
            ).start()
            return True

        # Check for "suggest" / "what should I do"
        if any(p in cmd_text for p in ("suggest", "what should i do",
                                        "any suggestions")):
            suggestion = self._agent.suggest_command()
            if suggestion:
                msg = f"Based on your habits, you usually run '{suggestion}' around this time."
                self._show_jarvis_text(msg)
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say(msg)
            else:
                msg = "I don't have enough data yet to make suggestions. Keep using voice commands and I'll learn your patterns."
                self._show_jarvis_text(msg)
            return True

        # Check for "remember <something>"
        remember_match = re.match(r"remember (?:that )?(.+)", cmd_text)
        if remember_match:
            note = remember_match.group(1).strip()
            self._agent.remember("user_note", note)
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(f"Noted. I'll remember that.")
            self._show_jarvis_text(f"Remembered: {note}")
            return True

        # Check for "recall" / "what did I say about"
        recall_match = re.match(r"(?:recall|what did i say about|remember about)\s+(.+)", cmd_text)
        if recall_match:
            query = recall_match.group(1).strip()
            results = self._agent.recall(query)
            if results:
                text = "\n".join(f"- {r['value']}" for r in results[:3])
                self._show_jarvis_text(f"I recall:\n{text}")
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say(f"I recall: {results[0]['value']}")
            else:
                self._show_jarvis_text("I don't have anything stored about that.")
            return True

        # "what's open" / "list windows"
        if cmd_text in ("what's open", "whats open", "list windows",
                        "show windows", "what windows are open"):
            windows = self._get_window_list()
            names = [n for _, n in windows[:10]]
            text = "\n".join(f"- {n}" for n in names)
            self._show_jarvis_text(f"Open windows:\n{text}")
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(f"You have {len(names)} windows open. {', '.join(names[:4])}")
            return True

        # "launch <app>"
        launch_match = re.match(r"launch\s+(.+)", cmd_text)
        if launch_match:
            app = launch_match.group(1).strip()
            _log(f"Launching: {app}")
            try:
                subprocess.Popen(
                    [app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self._show_jarvis_text(f"Launched {app}")
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say(f"Launching {app}.")
            except FileNotFoundError:
                # Try xdg-open for .desktop apps
                try:
                    subprocess.Popen(
                        ["gtk-launch", app],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    self._show_jarvis_text(f"Could not find '{app}'")
            return True

        # "type <text>" — type raw text into active window
        type_match = re.match(r"type\s+(.+)", cmd_text)
        if type_match:
            to_type = type_match.group(1).strip()
            threading.Thread(
                target=lambda: subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "5", to_type],
                    timeout=10, capture_output=True,
                ), daemon=True,
            ).start()
            self._show_jarvis_text(f"Typed: {to_type}")
            return True

        # "what's in my clipboard" / "read clipboard"
        if any(p in cmd_text for p in ("clipboard", "what did i copy",
                                        "read clipboard")):
            try:
                r = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=2,
                )
                clip = r.stdout.strip()[:200]
                self._show_jarvis_text(f"Clipboard: {clip}")
                if self.talkback_var.get() and clip:
                    from jarvis.jarvis_speak_queue import say
                    say(f"Your clipboard contains: {clip[:100]}")
            except Exception:
                self._show_jarvis_text("Could not read clipboard")
            return True

        # "search for <query>" — open web search
        search_match = re.match(r"(?:search|google|look up)\s+(?:for\s+)?(.+)", cmd_text)
        if search_match:
            query = search_match.group(1).strip()
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._show_jarvis_text(f"Searching: {query}")
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(f"Searching for {query}.")
            return True

        # "start a timer for X minutes"
        timer_match = re.match(
            r"(?:start a |set a )?timer\s+(?:for\s+)?(\d+)\s*(minutes?|mins?|seconds?|hours?)",
            cmd_text)
        if timer_match:
            amount = int(timer_match.group(1))
            unit = timer_match.group(2).lower()
            if unit.startswith("hour"):
                seconds = amount * 3600
            elif unit.startswith("min"):
                seconds = amount * 60
            else:
                seconds = amount
            self._set_reminder(seconds, f"Timer for {amount} {unit}")
            return True

        # "good night" / "shut down" / "go to sleep"
        if cmd_text in ("good night", "goodnight", "go to sleep", "shut down jarvis"):
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say("Good night sir. I'll be here when you need me.")
            return True

        # "what's running" / "heavy processes"
        if any(p in cmd_text for p in ("what's running", "whats running",
                                        "heavy processes", "top processes")):
            procs = self._agent.list_heavy_processes()
            text = "\n".join(f"- {p['cmd']} (CPU:{p['cpu']}% MEM:{p['mem']}%)"
                             for p in procs)
            self._show_jarvis_text(f"Top processes:\n{text}")
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(f"Top process is {procs[0]['cmd']} using {procs[0]['cpu']} percent CPU." if procs else "No heavy processes.")
            return True

        # "git status" / "what's changed"
        if any(p in cmd_text for p in ("git status", "what's changed",
                                        "whats changed", "repo status")):
            info = self._agent.git_summary()
            if info:
                text = (f"Branch: {info['branch']}\n"
                        f"Changed: {info['changed_files']} files\n"
                        f"Last: {info['last_commit']}\n"
                        f"Ahead: {info['commits_ahead']} commits")
                self._show_jarvis_text(text)
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say(f"On branch {info['branch']}. {info['changed_files']} changed files. {info['commits_ahead']} commits ahead of remote.")
            return True

        # "check network" / "am I online"
        if any(p in cmd_text for p in ("check network", "am i online",
                                        "internet", "connectivity")):
            net = self._agent.check_connectivity()
            status = "Online" if net.get("internet") else "Offline"
            latency = f", {net.get('latency_ms', '?')}ms" if net.get("internet") else ""
            ollama = "running" if net.get("ollama") else "not running"
            text = f"Internet: {status}{latency}\nOllama: {ollama}"
            self._show_jarvis_text(text)
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(f"You are {status.lower()}{latency}. Ollama is {ollama}.")
            return True

        # "find file <name>"
        find_match = re.match(r"find (?:file |files? )?(.+)", cmd_text)
        if find_match:
            name = find_match.group(1).strip()
            files = self._agent.find_file(name)
            if files:
                text = "\n".join(f"- {f}" for f in files)
                self._show_jarvis_text(f"Found:\n{text}")
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say(f"Found {len(files)} files matching {name}.")
            else:
                self._show_jarvis_text(f"No files matching '{name}'")
            return True

        # "recent files" / "what was I working on"
        if any(p in cmd_text for p in ("recent files", "what was i working on",
                                        "last edited", "recently modified")):
            files = self._agent.recent_files()
            text = "\n".join(f"- {Path(f).name}" for f in files)
            self._show_jarvis_text(f"Recently modified:\n{text}")
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                names = ", ".join(Path(f).name for f in files[:3])
                say(f"Most recently modified: {names}.")
            return True

        # Clipboard history: "show clipboard" / "clipboard history"
        if any(p in cmd_text for p in ("show clipboard", "clipboard history",
                                        "last copies", "paste history")):
            items = self._agent.get_clipboard_history()
            if items:
                text = "\n".join(f"{i+1}. {c['text'][:60]}"
                                 for i, c in enumerate(items))
                self._show_jarvis_text(f"Clipboard history:\n{text}")
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say(f"You have {len(items)} items in clipboard history.")
            else:
                self._show_jarvis_text("Clipboard history is empty.")
            return True

        # "paste the one before last" / "paste item 2"
        paste_match = re.match(r"paste (?:item |number )?(\d+|before last|previous)", cmd_text)
        if paste_match:
            idx_str = paste_match.group(1)
            idx = 1 if idx_str in ("before last", "previous") else int(idx_str) - 1
            result = self._agent.paste_from_history(idx)
            if result:
                self._show_jarvis_text(f"Pasted: {result}")
            return True

        # Voice notes: "take a note <text>" / "note that <text>"
        note_match = re.match(r"(?:take a note|note that|note)\s+(.+)", cmd_text)
        if note_match:
            note = note_match.group(1).strip()
            path = self._agent.save_voice_note(note)
            self._show_jarvis_text(f"Note saved: {note}")
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say("Note saved.")
            return True

        # "show notes" / "read notes"
        if any(p in cmd_text for p in ("show notes", "read notes", "my notes",
                                        "list notes", "voice notes")):
            notes = self._agent.list_voice_notes()
            if notes:
                text = "\n".join(f"- {n['content']}" for n in notes)
                self._show_jarvis_text(f"Recent notes:\n{text}")
            else:
                self._show_jarvis_text("No voice notes yet.")
            return True

        # Shell piping: "run <command>" / "execute <command>"
        run_match = re.match(r"(?:run|execute|shell)\s+(.+)", cmd_text)
        if run_match:
            shell_cmd = run_match.group(1).strip()
            first_token = shell_cmd.split()[0] if shell_cmd.split() else ""
            if first_token in SHELL_ALLOWLIST:
                _log(f"Shell command (allowlisted): {shell_cmd}")
                self._set_status("Running...", self.ACCENT, shell_cmd[:30])
                self._run_shell_async(shell_cmd)
            else:
                _log(f"Shell command requires confirmation: {shell_cmd}")
                self._set_status("Confirming...", self.YELLOW,
                                 "Say yes to run")
                threading.Thread(
                    target=self._confirm_and_run_shell,
                    args=(shell_cmd,),
                    daemon=True,
                ).start()
            return True

        # "count lines in <file>"
        count_match = re.match(r"count (?:the )?lines? in (.+)", cmd_text)
        if count_match:
            filename = count_match.group(1).strip()
            output = self._agent.run_shell(f"wc -l {filename} 2>/dev/null || find {Path.home()} -name '{filename}' -exec wc -l {{}} + 2>/dev/null | tail -1")
            self._show_jarvis_text(output)
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(output)
            return True

        # Multi-monitor: "move to other screen" / "other monitor"
        if any(p in cmd_text for p in ("other screen", "other monitor",
                                        "move to monitor", "next screen",
                                        "next monitor")):
            success = self._agent.move_window_to_monitor("next")
            if success:
                self._show_jarvis_text("Window moved to other monitor.")
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say("Done.")
            else:
                self._show_jarvis_text("Could not move window. Single monitor?")
            return True

        # Dictation mode: "dictate" — continuous typing without Claude
        if cmd_text in ("dictate", "start dictation", "dictation mode"):
            self._dictation_mode = True
            self._show_jarvis_text("Dictation mode: ON\nSay 'end dictation' to stop.")
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say("Dictation mode active. I'll type everything you say directly. Say end dictation to stop.")
            return True

        # Conditional trigger: "when gpu drops below 50 notify me"
        trigger_match = re.match(r"when (.+?)(?:,?\s*(?:notify me|tell me|alert me|let me know))", cmd_text)
        if trigger_match:
            condition = trigger_match.group(1).strip()
            self._agent.set_trigger(condition, f"Alert: {condition}")
            self._show_jarvis_text(f"Trigger set: when {condition}")
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(f"I'll notify you when {condition}.")
            return True

        # Text transformation: "make that uppercase" / "fix the grammar"
        if any(p in cmd_text for p in ("make that uppercase", "uppercase that",
                                        "make that lowercase", "lowercase that")):
            try:
                r = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=2,
                )
                clip = r.stdout.strip()
                if "upper" in cmd_text:
                    result = clip.upper()
                else:
                    result = clip.lower()
                proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                )
                proc.communicate(input=result.encode(), timeout=2)
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                    timeout=2, capture_output=True,
                )
                self._show_jarvis_text(f"Transformed: {result[:50]}")
            except Exception:
                pass
            return True

        # Direct questions Jarvis can answer locally
        answer = self._agent.answer_question(cmd_text)
        if answer:
            self._show_jarvis_text(answer)
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(answer)
            return True

        # Check for quick commands
        for phrase, shell_cmd in QUICK_COMMANDS.items():
            if phrase in cmd_text:
                _log(f"Quick command: {phrase} → {shell_cmd[:50]}")
                self._set_status("Running...", self.ACCENT, phrase)
                threading.Thread(
                    target=self._run_quick_command,
                    args=(phrase, shell_cmd), daemon=True,
                ).start()
                return True

        # Check for reminder: "remind me in X minutes to Y"
        remind_match = re.search(
            r"remind me in (\d+)\s*(minutes?|mins?|hours?|seconds?)\s+(?:to\s+)?(.+)",
            cmd_text, re.IGNORECASE,
        )
        if remind_match:
            amount = int(remind_match.group(1))
            unit = remind_match.group(2).lower()
            task = remind_match.group(3).strip()

            if unit.startswith("hour"):
                seconds = amount * 3600
            elif unit.startswith("min"):
                seconds = amount * 60
            else:
                seconds = amount

            self._set_reminder(seconds, task)
            return True

        return False

    def _run_shell_async(self, shell_cmd):
        """Run a shell command in a daemon thread; display + optionally speak the result."""
        def _run():
            output = self._agent.run_shell(shell_cmd)
            self.root.after(0, lambda: self._show_jarvis_text(
                f"$ {shell_cmd}\n{output}"))
            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, "Command done"))
            if self.talkback_var.get() and len(output) < 200:
                from jarvis.jarvis_speak_queue import say
                say(f"Result: {output[:100]}")
        threading.Thread(target=_run, daemon=True).start()

    def _confirm_and_run_shell(self, shell_cmd):
        """Prompt for voice confirmation, then run shell_cmd iff user said yes/confirm."""
        import sounddevice as sd

        tts = self._get_tts()
        hotword_was_active = self.hotword_var.get()
        if hotword_was_active:
            self._hotword.pause()
        try:
            tts.speak(f"Confirm: run {shell_cmd}?", block=True)

            try:
                audio = sd.rec(int(3 * SAMPLE_RATE),
                               samplerate=SAMPLE_RATE,
                               channels=1, dtype="float32")
                sd.wait()
            except Exception as e:
                _log(f"Confirm recording error: {e}")
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, "Confirmation failed"))
                return

            audio_flat = audio.flatten() if audio.ndim > 1 else audio
            if self._stt_engine is None or not self._stt_engine.is_loaded:
                _log("Confirm: STT not loaded; refusing")
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, "Refused (STT not ready)"))
                return

            result = self._stt_engine.transcribe(audio_flat)
            transcript = (result.text or "").strip().lower()
            _log(f"Confirm transcript: {transcript!r}")

            if "yes" in transcript or "confirm" in transcript:
                _log(f"Confirm accepted; running: {shell_cmd}")
                self.root.after(0, lambda: self._set_status(
                    "Running...", self.ACCENT, shell_cmd[:30]))
                self._run_shell_async(shell_cmd)
            else:
                _log(f"Confirm rejected: {transcript!r}")
                self.root.after(0, lambda: self._show_jarvis_text(
                    f"Refused: {shell_cmd} (heard: {transcript!r})"))
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN, "Refused"))
        finally:
            if hotword_was_active:
                self._hotword.resume()

    def _run_quick_command(self, name, shell_cmd):
        """Execute a quick command and speak the result."""
        try:
            result = subprocess.run(
                shell_cmd, shell=True, capture_output=True,
                text=True, timeout=30,
            )
            output = result.stdout.strip()[:300]
            _log(f"Quick command result: {output[:60]}")

            # Show in Jarvis text box
            self.root.after(0, lambda: self._show_jarvis_text(
                f"[{name}]\n{output}"))

            # Speak a summary if talk-back is on
            if self.talkback_var.get():
                summary = f"Command {name} complete."
                from jarvis.jarvis_speak_queue import say
                say(summary)

            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, f"{name} done"))
        except Exception as e:
            _log(f"Quick command error: {e}")
            self.root.after(0, lambda: self._set_status(
                "Error", "#da3633", str(e)[:40]))

    def _set_reminder(self, seconds, task):
        """Set a timed reminder."""
        _log(f"Reminder set: '{task}' in {seconds}s")
        mins = seconds // 60
        self._set_status("Ready", self.GREEN,
                         f"Reminder set: {mins}m — {task[:30]}")

        if self.talkback_var.get():
            from jarvis.jarvis_speak_queue import say
            say(f"Reminder set for {mins} minutes. I will remind you to {task}.")

        def _remind():
            time.sleep(seconds)
            _log(f"Reminder fired: {task}")
            # Desktop notification
            try:
                subprocess.Popen(
                    ["notify-send", "-u", "critical", "-i", "appointment-soon",
                     "Jarvis Reminder", task],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            # Speak it
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(f"Sir, this is your reminder. {task}")
            self.root.after(0, lambda: self._set_status(
                "Reminder!", self.ACCENT, task[:50]))
            self.root.after(0, lambda: self._show_jarvis_text(
                f"Reminder: {task}"))

        threading.Thread(target=_remind, daemon=True).start()

    def _check_desktop_command(self, text):
        """Check if text contains desktop control commands. Returns True if handled."""
        lower = text.strip().lower().rstrip(".")

        # Must start with "jarvis" prefix
        for prefix in ("jarvis ", "jarvis, ", "hey jarvis ", "hey jarvis, "):
            if lower.startswith(prefix):
                cmd_text = lower[len(prefix):].strip()
                break
        else:
            return False

        # Split on "and then", "then", "and", commas for chained commands
        parts = re.split(r'\s+and then\s+|\s+then\s+|\s+and\s+|,\s*', cmd_text)
        parts = [p.strip() for p in parts if p.strip()]

        if not parts:
            return False

        actions = []
        for part in parts:
            action = self._parse_desktop_action(part)
            if action:
                actions.append(action)

        if not actions:
            return False

        _log(f"Desktop commands: {actions}")
        self._set_status("Executing...", self.ACCENT,
                         f"{len(actions)} command{'s' if len(actions) > 1 else ''}")

        threading.Thread(
            target=self._execute_desktop_actions,
            args=(actions,), daemon=True,
        ).start()
        return True

    def _parse_desktop_action(self, text):
        """Parse a single desktop action from text."""
        text = text.strip().lower()

        # Window switching: "switch to opera", "open terminal"
        for phrase in ("switch to ", "go to ", "focus ", "open "):
            if text.startswith(phrase):
                target = text[len(phrase):].strip()
                return ("window", target)

        # Tab controls
        if "next tab" in text:
            return ("key", "ctrl+Tab")
        if "previous tab" in text or "prev tab" in text:
            return ("key", "ctrl+shift+Tab")
        if "new tab" in text:
            return ("key", "ctrl+t")
        if "close tab" in text:
            return ("key", "ctrl+w")

        # Scroll
        if "scroll down" in text:
            amount = 5
            m = re.search(r'(\d+)', text)
            if m:
                amount = int(m.group(1))
            return ("scroll", "down", amount)
        if "scroll up" in text:
            amount = 5
            m = re.search(r'(\d+)', text)
            if m:
                amount = int(m.group(1))
            return ("scroll", "up", amount)

        # Click
        if "double click" in text:
            return ("click", "double")
        if "right click" in text:
            return ("click", "right")
        if "click" in text:
            return ("click", "left")

        # Window management
        if "minimize" in text:
            return ("key", "super+h")
        if "maximize" in text:
            return ("key", "super+Up")
        if "full screen" in text or "fullscreen" in text:
            return ("key", "F11")
        if "close window" in text:
            return ("key", "alt+F4")

        # Media
        if "volume up" in text:
            return ("key", "XF86AudioRaiseVolume")
        if "volume down" in text:
            return ("key", "XF86AudioLowerVolume")
        if "mute" in text:
            return ("key", "XF86AudioMute")
        if text in ("play", "pause", "play pause"):
            return ("key", "XF86AudioPlay")

        # Keyboard shortcuts
        shortcuts = {
            "copy": "ctrl+c", "paste": "ctrl+v", "undo": "ctrl+z",
            "redo": "ctrl+shift+z", "save": "ctrl+s", "select all": "ctrl+a",
            "find": "ctrl+f",
        }
        for phrase, keys in shortcuts.items():
            if phrase in text:
                return ("key", keys)

        # Wait/pause
        m = re.match(r'wait\s+(\d+)', text)
        if m:
            return ("wait", int(m.group(1)))

        return None

    def _execute_desktop_actions(self, actions):
        """Execute a chain of desktop actions with xdotool."""
        for i, action in enumerate(actions):
            if action[0] == "window":
                target = action[1]
                _log(f"Desktop: switch to '{target}'")
                # Find window by name
                try:
                    result = subprocess.run(
                        ["xdotool", "search", "--name", target],
                        capture_output=True, text=True, timeout=2,
                    )
                    wids = result.stdout.strip().splitlines()
                    if wids:
                        subprocess.run(
                            ["xdotool", "windowactivate", "--sync", wids[0].strip()],
                            capture_output=True, timeout=2,
                        )
                        self.root.after(0, lambda t=target: self._set_status(
                            "Ready", self.GREEN, f"Switched to {t}"))
                    else:
                        _log(f"Desktop: window '{target}' not found")
                except Exception as e:
                    _log(f"Desktop window error: {e}")
                time.sleep(0.3)

            elif action[0] == "key":
                keys = action[1]
                _log(f"Desktop: key {keys}")
                try:
                    subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", keys],
                        capture_output=True, timeout=2,
                    )
                except Exception as e:
                    _log(f"Desktop key error: {e}")
                time.sleep(0.2)

            elif action[0] == "scroll":
                direction = action[1]
                amount = action[2] if len(action) > 2 else 5
                btn = "5" if direction == "down" else "4"
                _log(f"Desktop: scroll {direction} x{amount}")
                for _ in range(amount):
                    try:
                        subprocess.run(
                            ["xdotool", "click", "--clearmodifiers", btn],
                            capture_output=True, timeout=2,
                        )
                    except Exception:
                        pass
                    time.sleep(0.05)

            elif action[0] == "click":
                click_type = action[1]
                _log(f"Desktop: {click_type} click")
                try:
                    if click_type == "double":
                        subprocess.run(
                            ["xdotool", "click", "--clearmodifiers", "--repeat", "2",
                             "--delay", "100", "1"],
                            capture_output=True, timeout=2,
                        )
                    elif click_type == "right":
                        subprocess.run(
                            ["xdotool", "click", "--clearmodifiers", "3"],
                            capture_output=True, timeout=2,
                        )
                    else:
                        subprocess.run(
                            ["xdotool", "click", "--clearmodifiers", "1"],
                            capture_output=True, timeout=2,
                        )
                except Exception as e:
                    _log(f"Desktop click error: {e}")
                time.sleep(0.2)

            elif action[0] == "wait":
                secs = action[1]
                _log(f"Desktop: wait {secs}s")
                time.sleep(secs)

        self.root.after(0, lambda: self._set_status(
            "Ready", self.GREEN, f"Executed {len(actions)} commands"))

        if self.talkback_var.get():
            from jarvis.jarvis_speak_queue import say
            say(f"Done. Executed {len(actions)} commands.")

    def _agent_click_text(self, target):
        """Use screen awareness to find and click text."""
        success = self._agent.click_on_text(target)
        if success:
            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, f"Clicked '{target}'"))
        else:
            self.root.after(0, lambda: self._set_status(
                "Ready", self.YELLOW, f"Could not find '{target}' on screen"))
            if self.talkback_var.get():
                from jarvis.jarvis_speak_queue import say
                say(f"I could not find {target} on screen.")

    def _run_workflow(self, name, steps):
        """Execute a multi-step workflow."""
        results = []
        for step_type, step_data in steps:
            if step_type == "shell":
                try:
                    r = subprocess.run(
                        step_data, shell=True, capture_output=True,
                        text=True, timeout=60,
                    )
                    results.append(r.stdout.strip()[:200])
                    _log(f"Workflow step: {step_data[:40]} → OK")
                except Exception as e:
                    results.append(f"Error: {e}")
            elif step_type == "speak":
                if self.talkback_var.get():
                    from jarvis.jarvis_speak_queue import say
                    say(step_data)
            elif step_type == "key":
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", step_data],
                    capture_output=True, timeout=2,
                )
            elif step_type == "wait":
                time.sleep(int(step_data))
            time.sleep(0.3)

        output = "\n".join(r for r in results if r)
        self.root.after(0, lambda: self._show_jarvis_text(
            f"[{name}]\n{output[:500]}"))
        self.root.after(0, lambda: self._set_status(
            "Ready", self.GREEN, f"Workflow '{name}' complete"))
        self._agent.log_command(f"workflow:{name}")

    def _on_brain_response(self, actions):
        """Handle structured response from Claude brain."""
        _log(f"Brain response: {len(actions)} actions")
        spoken_parts = []
        gui_parts = []

        def _execute():
            for action_type, action_data in actions:
                if action_type == "SPEAK":
                    spoken_parts.append(action_data)
                    gui_parts.append(action_data)
                elif action_type == "SILENT":
                    gui_parts.append(action_data)
                elif action_type == "RUN":
                    _log(f"Brain RUN: {action_data[:50]}")
                    output = self._agent.run_shell(action_data)
                    gui_parts.append(f"$ {action_data}\n{output}")
                elif action_type == "TYPE":
                    subprocess.run(
                        ["xdotool", "type", "--clearmodifiers",
                         "--delay", "5", action_data],
                        timeout=10, capture_output=True,
                    )
                elif action_type == "WINDOW":
                    self._execute_desktop_actions([("window", action_data)])
                elif action_type == "CLICK":
                    self._agent.click_on_text(action_data)
                time.sleep(0.2)

            # Update GUI
            jarvis_text = "\n".join(gui_parts)
            self.root.after(0, lambda: self._show_jarvis_text(jarvis_text))

            # Speak
            if spoken_parts and self.talkback_var.get():
                full_speech = " ".join(spoken_parts)
                from jarvis.jarvis_speak_queue import say
                say(full_speech)
            else:
                self._speaking_animation = False

            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, ""))

        threading.Thread(target=_execute, daemon=True).start()

    def _show_jarvis_text(self, text):
        """Update the Jarvis transcription box."""
        self.jarvis_text.config(state=tk.NORMAL)
        self.jarvis_text.delete("1.0", tk.END)
        self.jarvis_text.insert("1.0", text)
        self.jarvis_text.config(state=tk.DISABLED)

    def _type_then_screenshot(self, text):
        """Take screenshot first, then type user's text with screenshot reference."""
        # Capture screen first
        try:
            from PIL import ImageGrab
            capture_dir = Path("/tmp/vss_screen")
            capture_dir.mkdir(parents=True, exist_ok=True)
            latest = capture_dir / "latest.png"
            ts = datetime.now().strftime("%H%M%S")
            timestamped = capture_dir / f"screen_{ts}.png"
            img = ImageGrab.grab()
            img.save(str(latest))
            img.save(str(timestamped))
            captures = sorted(capture_dir.glob("screen_*.png"))
            for old in captures[:-20]:
                old.unlink(missing_ok=True)
            _log(f"Screenshot saved: {latest}")
            try:
                subprocess.Popen(
                    ["notify-send", "-i", "camera-photo", "-t", "2000",
                     "Screenshot Captured", f"Saved to {latest}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass
        except Exception as e:
            _log(f"Screenshot error: {e}")

        # Type user's text with screenshot context
        # Prepend "check the screen" so Claude knows to look at the screenshot
        combined = f"check the screen, {text}" if text else "check the screen"
        self._type_text(combined)

    def _take_screenshot(self):
        """Capture screen and send 'check the screen' to Claude terminal."""
        try:
            from PIL import ImageGrab
            from pathlib import Path

            capture_dir = Path("/tmp/vss_screen")
            capture_dir.mkdir(parents=True, exist_ok=True)
            latest = capture_dir / "latest.png"
            ts = datetime.now().strftime("%H%M%S")
            timestamped = capture_dir / f"screen_{ts}.png"

            img = ImageGrab.grab()
            img.save(str(latest))
            img.save(str(timestamped))

            # Keep only last 20
            captures = sorted(capture_dir.glob("screen_*.png"))
            for old in captures[:-20]:
                old.unlink(missing_ok=True)

            _log(f"Screenshot saved: {latest}")

            # Notify — use urgency=critical so it's not missed
            try:
                subprocess.Popen(
                    ["notify-send", "-u", "normal", "-i", "camera-photo",
                     "-t", "3000",
                     "Screenshot Captured",
                     f"Saved to {latest}\nSending to Claude..."],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass

            self.root.after(0, lambda: self._set_status(
                "Screenshot!", self.GREEN, f"Captured → {latest.name}"))

            # Type "check the screen" into Claude terminal
            self._type_text("check the screen")

            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, "Screenshot taken + sent to Claude"))
        except Exception as e:
            _log(f"Screenshot error: {e}")
            self.root.after(0, lambda: self._set_status(
                "Error", "#da3633", f"Screenshot failed: {e}"))

    # ------------------------------------------------------------------
    # Review mode: send edited text
    # ------------------------------------------------------------------
    def _send_reviewed_text(self):
        """Send the reviewed/edited text from the output box."""
        if not self.review_var.get():
            return

        text = self.output_text.get("1.0", tk.END).strip()
        if not text:
            return

        # Log it
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._session_log.append((ts, text))
        self.log_count_label.config(text=f"{len(self._session_log)} entries")

        ts_short = datetime.now().strftime("%H:%M")
        self._history.insert(0, f"[{ts_short}] {text[:55]}{'...' if len(text) > 55 else ''}")
        self._history = self._history[:8]
        self.history_label.config(text="\n".join(self._history))

        # Type it
        threading.Thread(
            target=self._type_text, args=(text,), daemon=True
        ).start()

        # Clear
        self.output_text.delete("1.0", tk.END)
        self._set_status("Ready", self.GREEN, f"Sent {len(text)} chars")

        self._maybe_continuous_restart()

    def _maybe_continuous_restart(self):
        """Restart recording if continuous mode is on."""
        if self._voice_stopped:
            self._voice_stopped = False
            return
        if self._daemon_launched:
            return  # Daemon mode: record once, then wait for next hotword
        if self.continuous_var.get() and self.model_loaded and not self.processing:
            if not self.review_var.get():  # Don't auto-restart in review mode
                self.root.after(1500, self._start_recording)

    def _reset_button(self):
        if self.model_loaded:
            self.record_btn.config(
                text="R E C O R D",
                state=tk.NORMAL,
                bg="#0c1822", fg="#67e8f9",
                activebackground="#132637", activeforeground="#67e8f9",
            )
        else:
            self.record_btn.config(
                text="R E C O R D",
                state=tk.DISABLED,
                bg="#0c1822", fg=self.MUTED,
                activebackground="#0c1822", activeforeground=self.MUTED,
            )

    # ------------------------------------------------------------------
    # Particle Orbit waveform
    # ------------------------------------------------------------------
    def _prerender_reactor_frames(self):
        """Pre-render arc reactor center at 10 brightness levels.

        This is the expensive part (numpy radial gradients). By doing it
        once at startup, each animation frame just pastes a pre-made image.
        """
        from PIL import Image
        S = self._orbit_size
        cx, cy = S // 2, S // 2 - 5
        bg_r, bg_g, bg_b = self._orbit_bg

        self._reactor_frames = []  # 10 levels: amp 0.0 to 0.9

        for level in range(10):
            amp = level / 10.0
            frame = np.full((S, S, 3), [bg_r, bg_g, bg_b], dtype=np.uint8)

            # Ambient glow
            glow_r = int(45 + amp * 35)
            y_c, x_c = np.ogrid[-cy:S - cy, -cx:S - cx]
            dist_sq = x_c * x_c + y_c * y_c
            glow_mask = dist_sq < glow_r * glow_r
            if glow_mask.any():
                dist = np.sqrt(dist_sq[glow_mask].astype(np.float32))
                intensity = (1.0 - dist / glow_r) * (0.12 + amp * 0.25)
                intensity = np.clip(intensity, 0, 1)
                for ch, tgt in enumerate([6, 182, 212]):
                    cur = frame[glow_mask, ch].astype(np.float32)
                    frame[glow_mask, ch] = np.clip(
                        cur + (tgt - cur) * intensity, 0, 255
                    ).astype(np.uint8)

            # Concentric rings
            ra = 0.25 + amp * 0.35
            _draw_circle(frame, cx, cy, 38, (6, 182, 212), ra * 0.3)
            _draw_circle(frame, cx, cy, 32, (6, 182, 212), ra * 0.5)
            _draw_circle(frame, cx, cy, 24, (6, 182, 212), ra * 0.7)
            _draw_circle(frame, cx, cy, 16, (103, 232, 249), ra)

            # Core glow
            core_r = max(6, int(7 + amp * 3))
            core_glow_r = core_r + 12
            dg_y1 = max(0, cy - core_glow_r)
            dg_y2 = min(S, cy + core_glow_r + 1)
            dg_x1 = max(0, cx - core_glow_r)
            dg_x2 = min(S, cx + core_glow_r + 1)
            yy, xx = np.ogrid[dg_y1 - cy:dg_y2 - cy, dg_x1 - cx:dg_x2 - cx]
            d_dist = np.sqrt((xx * xx + yy * yy).astype(np.float32))
            d_mask = d_dist < core_glow_r
            region = frame[dg_y1:dg_y2, dg_x1:dg_x2]
            d_int = np.clip((1 - d_dist / core_glow_r) * (0.5 + amp * 0.5), 0, 1)
            for ch, tgt in enumerate([103, 232, 249]):
                cur = region[:, :, ch].astype(np.float32)
                cur[d_mask] += (tgt - cur[d_mask]) * d_int[d_mask]
                region[:, :, ch] = np.clip(cur, 0, 255).astype(np.uint8)
            _draw_filled_circle(frame, cx, cy, core_r, (150, 240, 255), 0.95)

            self._reactor_frames.append(Image.fromarray(frame, "RGB"))

        _log(f"Pre-rendered {len(self._reactor_frames)} reactor frames")

    def _update_orbit_idle(self):
        """Idle animation."""
        if self.recording:
            return
        self._orbit_time += 0.025
        amp = 0.25 + math.sin(self._orbit_time * 0.6) * 0.08
        self._render_orbit_fast(amp)
        self.root.after(60, self._update_orbit_idle)

    def _update_waveform(self):
        """Recording animation."""
        if not self.recording:
            self.root.after(100, self._update_orbit_idle)
            return
        self._orbit_time += 0.03
        amp = min(getattr(self, '_audio_level', 0.0), 0.73)
        self._render_orbit_fast(amp)
        self.root.after(80, self._update_waveform)

    def _render_orbit_fast(self, amp):
        """Fast orbit render: paste pre-rendered reactor + draw particles with PIL."""
        if self._orbit_rendering:
            return
        self._orbit_rendering = True

        # Update particle angles (main thread, fast)
        for p in self._orbit_particles:
            speed_mult = 0.015 + amp * 0.2
            p["angle"] += p["speed"] * speed_mult

        particles_snap = [(p["ring"], p["angle"], p["base_r"], p["size"],
                           p["phase"]) for p in self._orbit_particles]
        t = self._orbit_time

        def _render():
            try:
                img = self._compose_frame(amp, t, particles_snap)
                self._orbit_pending = img
            except Exception:
                pass
            self._orbit_rendering = False
            self.root.after(0, self._apply_frame)

        threading.Thread(target=_render, daemon=True).start()

    def _compose_frame(self, amp, t, particles):
        """Compose frame: pre-rendered reactor + PIL particles."""
        from PIL import ImageDraw

        S = self._orbit_size
        cx, cy = S // 2, S // 2 - 5

        # Pick pre-rendered reactor frame by amplitude
        level = min(9, max(0, int(amp * 10)))
        img = self._reactor_frames[level].copy()

        # Draw rotating segment dots on the reactor
        draw = ImageDraw.Draw(img.convert("RGBA") if img.mode != "RGBA"
                              else img)
        # Convert to RGBA for particle alpha
        if img.mode != "RGBA":
            img = img.convert("RGBA")
            draw = ImageDraw.Draw(img)

        seg_count = 10
        seg_r = 32
        seg_a = int((0.5 + amp * 0.4) * 255)
        for i in range(seg_count):
            angle = (i / seg_count) * math.pi * 2 + t * 0.5
            sx = cx + math.cos(angle) * seg_r
            sy = cy + math.sin(angle) * seg_r
            draw.ellipse([sx - 2, sy - 2, sx + 2, sy + 2],
                         fill=(103, 232, 249, seg_a))

        # Particles — small, fast, sporadic
        for ring, angle, base_r, size, phase in particles:
            wobble = math.sin(t * 2.5 + phase) * 2
            r = base_r + wobble
            jx = math.sin(t * 3.7 + phase * 2.3) * (2 + amp * 5)
            jy = math.cos(t * 4.1 + phase * 1.7) * (2 + amp * 5)
            x = cx + math.cos(angle) * r + jx
            y = cy + math.sin(angle) * r + jy

            alpha = 0.15 + amp * 0.75 * (1 - ring * 0.1)
            sz = max(1, size * (0.5 + amp * 0.5))
            a = int(alpha * 255)

            if ring % 2 == 0:
                color = (6, 182, 212, a)
            else:
                color = (103, 232, 249, int(a * 0.8))

            draw.ellipse([x - sz, y - sz, x + sz, y + sz], fill=color)

        # Voice ID ring
        if self.recording and self.speaker_verify_var.get():
            voice_match = getattr(self, '_voice_id_match', False)
            vr = S // 2 - 8
            if voice_match:
                draw.ellipse([cx - vr, cy - vr, cx + vr, cy + vr],
                             outline=(63, 185, 80, 100), width=1)
            else:
                draw.ellipse([cx - vr, cy - vr, cx + vr, cy + vr],
                             outline=(80, 40, 40, 50), width=1)

        return img.convert("RGB")

    def _apply_frame(self):
        """Apply rendered frame to canvas."""
        if self._orbit_pending is None:
            return
        from PIL import ImageTk
        try:
            self._orbit_photo = ImageTk.PhotoImage(self._orbit_pending)
            self.waveform_canvas.itemconfigure(
                self._orbit_img_id, image=self._orbit_photo)
        except tk.TclError:
            S = self._orbit_size
            self._orbit_img_id = self.waveform_canvas.create_image(
                S // 2, S // 2, anchor="center", image=self._orbit_photo)
        self._orbit_pending = None

    # ------------------------------------------------------------------
    # Recording timer
    # ------------------------------------------------------------------
    def _update_timer(self):
        if not self.recording or self._record_start_time is None:
            return
        elapsed = time.monotonic() - self._record_start_time
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        self.record_btn.config(text=f"\u23F9  S T O P  {mins}:{secs:02d}")
        self.root.after(500, self._update_timer)

    # ------------------------------------------------------------------
    # Silence auto-stop
    # ------------------------------------------------------------------
    MAX_RECORDING_SECONDS = 60  # Hard cap to prevent memory issues

    def _check_silence(self):
        """Auto-stop recording when Silero VAD detects no speech."""
        if not self.recording:
            return

        # Hard cap on recording length
        if self._record_start_time:
            elapsed = time.monotonic() - self._record_start_time
            if elapsed >= self.MAX_RECORDING_SECONDS:
                _log(f"Max recording time ({self.MAX_RECORDING_SECONDS}s)")
                self._voice_stopped = True
                try:
                    self._stop_and_transcribe()
                except Exception as e:
                    _log(f"Max recording stop error: {e}")
                    self.recording = False
                    self._reset_button()
                return

        # Skip first 2 seconds (let speech establish)
        if self._record_start_time and (time.monotonic() - self._record_start_time) < 2.0:
            self.root.after(100, self._check_silence)
            return

        # Silero VAD speech detection
        try:
            if len(self._audio_frames) >= 2:
                recent = np.concatenate(self._audio_frames[-2:], axis=0).flatten()
                rate = getattr(self, '_record_rate', 16000)
                if rate != 16000:
                    from scipy.signal import resample
                    new_len = int(len(recent) * 16000 / rate)
                    recent = resample(recent, new_len).astype(np.float32)
                chunk = recent[-512:] if len(recent) >= 512 else recent
                speech_prob = self._silero_vad.is_speech(chunk, sr=16000)
            else:
                speech_prob = 1.0
        except Exception:
            speech_prob = 1.0

        timeout = self.silence_var.get()

        if speech_prob < 0.3:
            if self._silence_start is None:
                self._silence_start = time.monotonic()
            elif (time.monotonic() - self._silence_start) >= timeout:
                _log(f"Auto-stop: no speech for {timeout}s (prob={speech_prob:.2f})")
                self._voice_stopped = True
                try:
                    self._stop_and_transcribe()
                except Exception as e:
                    _log(f"Auto-stop error: {e}")
                    self.recording = False
                    self._reset_button()
                    if self.hotword_var.get():
                        self._hotword.resume()
                return
        else:
            self._silence_start = None

        self.root.after(100, self._check_silence)

    def _check_speaker_silence(self):
        """Voice-aware silence: auto-stop when user hasn't spoken for timeout,
        even if background audio (TV/YouTube) is still making noise.

        Runs every 3 seconds, checks the last 3 seconds of audio against
        the voiceprint. If it doesn't match, starts a silence timer.
        """
        if not self.recording:
            return
        if not self.speaker_verify_var.get() or self._speaker_verifier is None:
            # Voice ID not active — regular silence detection handles it
            self.root.after(3000, self._check_speaker_silence)
            return
        if not self._speaker_verifier.enrolled:
            self.root.after(3000, self._check_speaker_silence)
            return

        now = time.monotonic()
        # Don't check for the first 5 seconds of recording — the initial
        # audio after hotword detection is unreliable (mix of wake word +
        # start of sentence scores low on speaker verification)
        if self._record_start_time and (now - self._record_start_time) < 5.0:
            self.root.after(3000, self._check_speaker_silence)
            return

        # Grab the last 3 seconds of audio
        rate = getattr(self, '_record_rate', SAMPLE_RATE)
        samples_needed = int(rate * 3.0)
        frames = list(self._audio_frames)
        if not frames:
            self.root.after(3000, self._check_speaker_silence)
            return

        recent = np.concatenate(frames[-max(1, samples_needed // int(rate * 0.1)):],
                                axis=0).flatten()
        # Resample to 16kHz for speaker check
        audio_16k = self._resample_to_16k(recent)

        def _check():
            try:
                is_match, score = self._speaker_verifier.verify(audio_16k)
            except Exception:
                return
            self.root.after(0, lambda: self._on_speaker_silence_result(is_match, score))

        threading.Thread(target=_check, daemon=True).start()
        self.root.after(3000, self._check_speaker_silence)

    def _on_speaker_silence_result(self, is_match, score):
        """Handle result of periodic speaker check during recording."""
        if not self.recording:
            return

        # Use a relaxed threshold for silence detection — we don't want to
        # cut the user off mid-sentence on a borderline score. The strict
        # threshold is applied later by the segment filter.
        relaxed_threshold = self.speaker_threshold_var.get() * 0.70
        is_probably_user = score >= relaxed_threshold
        self._voice_id_match = is_probably_user

        if is_probably_user:
            # User is likely speaking — reset speaker silence timer
            if self._speaker_silence_start is not None:
                _log(f"Voice-ID silence reset (score={score:.3f} >= "
                     f"relaxed {relaxed_threshold:.3f})")
            self._speaker_silence_start = None
            self._speaker_silence_misses = 0
        else:
            # Probably not the user — count consecutive misses
            self._speaker_silence_misses = getattr(
                self, '_speaker_silence_misses', 0) + 1

            # Require 2 consecutive misses before starting silence timer
            # (one borderline check shouldn't trigger a stop)
            if self._speaker_silence_misses < 2:
                _log(f"Voice-ID miss 1/2 (score={score:.3f}), waiting for confirmation")
                return

            if self._speaker_silence_start is None:
                self._speaker_silence_start = time.monotonic()
                _log(f"Voice-ID silence started (score={score:.3f}, "
                     f"2 consecutive misses)")

            elapsed = time.monotonic() - self._speaker_silence_start
            timeout = self.silence_var.get()
            if elapsed >= timeout and len(self._audio_frames) > 5:
                _log(f"Voice-ID auto-stop: no user voice for {elapsed:.1f}s "
                     f"(last score={score:.3f})")
                self._voice_stopped = True
                try:
                    self._stop_and_transcribe()
                except Exception as e:
                    _log(f"Voice-ID auto-stop error: {e}")
                    self.recording = False
                    self._reset_button()
                    if self.hotword_var.get():
                        self._hotword.resume()

    # ------------------------------------------------------------------
    # Text injection
    # ------------------------------------------------------------------
    @staticmethod
    def _is_claude_title(name):
        """Detect Claude Code terminal by its spinner-prefixed title."""
        return bool(name) and ord(name[0]) > 127 and len(name) > 1 and name[1] == ' '

    _claude_wid_cache: tuple | None = None  # (wid, expires_monotonic)

    @classmethod
    def _find_claude_terminal(cls):
        """Find the Claude Code terminal window ID, or None (5s TTL cache)."""
        now = time.monotonic()
        cached = cls._claude_wid_cache
        if cached is not None:
            wid, expires = cached
            if now < expires:
                return wid

        wid = cls._lookup_claude_terminal_uncached()
        cls._claude_wid_cache = (wid, now + 5.0)
        return wid

    @staticmethod
    def _lookup_claude_terminal_uncached():
        """Uncached lookup — scans terminals for Claude's spinner-prefixed title."""
        try:
            result = subprocess.run(
                ["xdotool", "search", "--class", "terminal"],
                capture_output=True, text=True, timeout=2,
            )
            wids = result.stdout.strip().splitlines()
            for wid in wids:
                wid = wid.strip()
                if not wid:
                    continue
                name_result = subprocess.run(
                    ["xdotool", "getwindowname", wid],
                    capture_output=True, text=True, timeout=2,
                )
                name = name_result.stdout.strip()
                if VoiceInputGUI._is_claude_title(name):
                    _log(f"Found Claude terminal: WID={wid} name={name!r}")
                    return wid
        except Exception as e:
            _log(f"Claude terminal search error: {e}")
        return None

    def _live_type_partial(self, text):
        """Type new words from partial transcription into target window."""
        if not self.recording or not text:
            return

        old = self._live_typed_text
        # Find what's new — only type the difference
        if text.startswith(old):
            new_part = text[len(old):]
        elif old and text[:len(old) // 2] == old[:len(old) // 2]:
            # Partial changed in the middle — Whisper re-interpreted.
            # Backspace old text and retype. This is the tricky case.
            new_part = None  # Signal to replace
        else:
            new_part = text  # Completely different — type fresh

        if new_part is None:
            # Need to backspace and retype — do it in background
            def _replace():
                if self._live_typed_chars > 0:
                    # Backspace all previously typed characters
                    subprocess.run(
                        ["xdotool", "key", "--clearmodifiers"]
                        + ["BackSpace"] * self._live_typed_chars,
                        timeout=5, capture_output=True,
                    )
                # Type the new text
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "3", text],
                    timeout=10, capture_output=True,
                )
                self._live_typed_chars = len(text)
                self._live_typed_text = text
            threading.Thread(target=_replace, daemon=True).start()
        elif new_part.strip():
            def _append():
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "3", new_part],
                    timeout=5, capture_output=True,
                )
                self._live_typed_chars += len(new_part)
                self._live_typed_text = text
            threading.Thread(target=_append, daemon=True).start()

    def _type_text(self, text):
        """Type text into the target window.

        Priority: pinned target > auto-detect Claude terminal > active window.
        """
        time.sleep(0.2)

        target_wid = None
        target_name = "active window"

        # 1. Pinned target (user picked a specific window)
        if self._target_wid:
            target_wid = self._target_wid
            short = (self._target_name or "")[:30]
            target_name = short or f"window {target_wid}"

        # 2. Auto-detect Claude terminal
        elif self.smart_target_var.get():
            wid = self._find_claude_terminal()
            if wid:
                target_wid = wid
                target_name = "Claude terminal"
            else:
                _log("No Claude terminal found, typing into active window")

        # Focus the target window if we have one
        if target_wid:
            try:
                subprocess.run(
                    ["xdotool", "windowactivate", "--sync", target_wid],
                    capture_output=True, text=True, timeout=2,
                )
                time.sleep(0.15)
            except Exception as e:
                _log(f"Window focus error: {e}")
                target_name = "active window"

        try:
            # If live writing was active, backspace the partial first
            if self.live_write_var.get() and self._live_typed_chars > 0:
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers"]
                    + ["BackSpace"] * self._live_typed_chars,
                    timeout=10, capture_output=True,
                )
                self._live_typed_chars = 0
                self._live_typed_text = ""
                time.sleep(0.05)

            # Type the final text
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "5", text],
                timeout=10, capture_output=True,
            )
            # Auto-press Enter if enabled
            if self.auto_enter_var.get():
                time.sleep(0.05)
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "Return"],
                    timeout=2, capture_output=True,
                )
                _log(f"Text typed + Enter into {target_name}")
            else:
                _log(f"Text typed into {target_name}")
            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, f"Typed {len(text)} chars → {target_name}"))
        except Exception as e:
            _log(f"xdotool error: {e}")

    # ------------------------------------------------------------------
    # Talk-back TTS
    # ------------------------------------------------------------------
    def _get_tts(self):
        """Lazy-load TTS engine; recreate when the engine selection changes.

        Avoids instantiating the full JarvisTTS model on every speak-queue
        poll tick when talkback is off. Also fully reloads the underlying
        model (not just the engine attribute) when the user switches
        engines in the dropdown — previously the attribute changed but the
        loaded weights did not.
        """
        want_engine = self.tts_engine_var.get()
        if self._tts is not None and self._tts.engine == want_engine:
            return self._tts
        if self._tts is not None:
            try:
                self._tts.stop()
            except Exception:
                pass
            self._tts = None
        from jarvis.jarvis_tts import JarvisTTS
        self._tts = JarvisTTS(engine=want_engine)
        return self._tts

    def _add_voice_clip(self):
        """Prompt for a YouTube URL and download as JARVIS voice reference."""
        from tkinter import simpledialog

        url = simpledialog.askstring(
            "Add Voice Clip",
            "Paste a YouTube URL with clean JARVIS dialogue:",
            parent=self.root,
        )
        if not url or "youtube" not in url and "youtu.be" not in url:
            return

        self._set_status("Downloading...", self.YELLOW, "Fetching voice clip")

        def _download():
            try:
                import subprocess
                ref_dir = Path.home() / ".aiws_trainer" / "jarvis_reference"
                ref_dir.mkdir(parents=True, exist_ok=True)

                # Count existing clips
                existing = list(ref_dir.glob("clip_*.wav"))
                idx = len(existing)
                out = ref_dir / f"clip_{idx:02d}.wav"

                result = subprocess.run(
                    [sys.executable, "-m", "yt_dlp",
                     "--extract-audio", "--audio-format", "wav",
                     "-o", str(out).replace(".wav", ".%(ext)s"),
                     "--no-playlist", url],
                    capture_output=True, text=True, timeout=60,
                )

                if result.returncode == 0 and out.exists():
                    # Use as the new voice reference (or combine with existing)
                    import shutil
                    main_ref = Path.home() / ".aiws_trainer" / "jarvis_voice_ref.wav"
                    shutil.copy2(out, main_ref)
                    _log(f"Voice clip added: {out}")
                    self.root.after(0, lambda: self._set_status(
                        "Ready", self.GREEN,
                        f"Voice clip #{idx} added! Restart for XTTS to use it."))
                else:
                    _log(f"Download failed: {result.stderr[-200:]}")
                    self.root.after(0, lambda: self._set_status(
                        "Error", "#da3633", "Download failed"))
            except Exception as e:
                _log(f"Voice clip error: {e}")
                self.root.after(0, lambda: self._set_status(
                    "Error", "#da3633", str(e)[:40]))

        threading.Thread(target=_download, daemon=True).start()

    def _toggle_tts_engine(self):
        """Switch between Edge TTS and XTTS JARVIS voice."""
        if self.tts_engine_var.get() == "edge":
            self.tts_engine_var.set("xtts")
        else:
            self.tts_engine_var.set("edge")
        self._update_tts_engine_btn()
        # Update the TTS instance if it exists
        if self._tts is not None:
            self._tts.engine = self.tts_engine_var.get()
        self._save_settings()

    def _update_tts_engine_btn(self):
        """Update the engine button text."""
        eng = self.tts_engine_var.get()
        if eng == "xtts":
            self._tts_engine_btn.config(text="JARVIS")
        else:
            self._tts_engine_btn.config(text="Edge")

    def _update_speaking_animation(self):
        """Animate the arc reactor synced to Jarvis speech audio."""
        if not getattr(self, '_speaking_animation', False):
            _log("Speaking animation ended, returning to idle")
            self._audio_level = 0.0
            self.root.after(200, self._update_orbit_idle)
            return
        if self.recording:
            return
        self._orbit_time += 0.03
        # Read real amplitude from TTS engine
        tts = self._get_tts()
        amp = 0.15 + tts.current_amplitude * 0.58  # Scale to 0.15-0.73 range
        self._render_orbit_fast(amp)
        self.root.after(80, self._update_speaking_animation)

    def _start_speak_queue_watcher(self):
        """Watch the speak queue file for new lines from Claude."""
        from jarvis.speak_queue_auth import _ensure_queue_dir
        _ensure_queue_dir()

        speak_file = Path("/tmp/vss_voice/speak_queue.txt")
        self._speak_queue_pos = 0

        # Clear any old content
        if speak_file.exists():
            self._speak_queue_pos = speak_file.stat().st_size

        self._watch_speak_queue()

    def _watch_speak_queue(self):
        """Poll the speak queue file for new HMAC-signed lines."""
        from jarvis.speak_queue_auth import verify

        if not self.talkback_var.get():
            self.root.after(2000, self._watch_speak_queue)
            return

        # Don't queue new speech while already speaking (prevents double-play)
        tts = self._get_tts()
        if tts.is_speaking:
            self.root.after(1000, self._watch_speak_queue)
            return

        speak_file = Path("/tmp/vss_voice/speak_queue.txt")
        try:
            if speak_file.exists():
                size = speak_file.stat().st_size
                if size > self._speak_queue_pos:
                    with open(speak_file) as f:
                        f.seek(self._speak_queue_pos)
                        new_lines = f.read()
                    self._speak_queue_pos = size

                    verified = []
                    for line in new_lines.strip().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        payload = verify(line)
                        if payload is None:
                            _log("Speak queue: dropped unsigned line")
                            continue
                        verified.append(payload)

                    combined = " ".join(verified)
                    if combined:
                        _log(f"Talk-back queue: {combined[:60]}")
                        self.root.after(0, lambda: self._set_status(
                            "Speaking...", self.ACCENT, "Jarvis"))
                        self._speaking_animation = True
                        self.root.after(0, self._update_speaking_animation)

                        def _show_transcript(t=combined):
                            self.jarvis_text.config(state=tk.NORMAL)
                            self.jarvis_text.delete("1.0", tk.END)
                            self.jarvis_text.insert("1.0", t)
                            self.jarvis_text.config(state=tk.DISABLED)
                            ts_short = datetime.now().strftime("%H:%M")
                            self._history.insert(0,
                                f"[{ts_short}] Jarvis: {t[:45]}...")
                            self._history = self._history[:8]
                            self.history_label.config(
                                text="\n".join(self._history))
                        self.root.after(0, _show_transcript)

                        def _speak_and_stop(t=combined):
                            # Pause hotword BEFORE speaking (synchronous,
                            # not via root.after) to prevent feedback loop
                            if self.hotword_var.get():
                                self._hotword.pause()
                                time.sleep(0.3)  # Let mic fully release

                            try:
                                tts.speak(t, block=True)
                            finally:
                                self._speaking_animation = False
                                # Wait for echo to die before resuming hotword
                                if self.hotword_var.get():
                                    time.sleep(1.5)
                                    self._hotword.resume()
                                self.root.after(0, lambda: self._set_status(
                                    "Ready", self.GREEN, ""))
                                _log("Talk-back: speech done, hotword resumed after 1.5s cooldown")

                        threading.Thread(target=_speak_and_stop,
                                         daemon=True).start()
        except Exception as e:
            _log(f"Speak queue error: {e}")

        self.root.after(1000, self._watch_speak_queue)

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------
    def _copy_last(self):
        was_disabled = str(self.output_text.cget("state")) == "disabled"
        self.output_text.config(state=tk.NORMAL)
        text = self.output_text.get("1.0", tk.END).strip()
        if was_disabled:
            self.output_text.config(state=tk.DISABLED)
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("Ready", self.GREEN, "Copied to clipboard")

    def _copy_jarvis(self):
        self.jarvis_text.config(state=tk.NORMAL)
        text = self.jarvis_text.get("1.0", tk.END).strip()
        self.jarvis_text.config(state=tk.DISABLED)
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("Ready", self.GREEN, "Copied Jarvis response")

    # ------------------------------------------------------------------
    # Audio playback
    # ------------------------------------------------------------------
    def _play_last_audio(self):
        """Play back the last recorded audio clip."""
        if self._last_audio is None:
            self._set_status("Ready", self.YELLOW, "No recording to play")
            return
        threading.Thread(target=self._play_audio_worker, daemon=True).start()

    def _play_audio_worker(self):
        """Save last audio to temp WAV and play it."""
        try:
            wav_path = LOG_DIR / "last_recording.wav"
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            rate = getattr(self, '_record_rate', SAMPLE_RATE)
            audio_int16 = (self._last_audio * 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(audio_int16.tobytes())
            duration = len(self._last_audio) / rate
            self.root.after(0, lambda: self._set_status(
                "Playing...", self.BLUE, f"{duration:.1f}s"))
            _play_beep(str(wav_path))
            # Wait for playback to roughly finish before resetting status
            time.sleep(min(duration + 0.3, 30))
            self.root.after(0, lambda: self._set_status(
                "Ready", self.GREEN, "Playback done"))
        except Exception as e:
            _log(f"Playback error: {e}")
            self.root.after(0, lambda: self._set_status(
                "Ready", self.YELLOW, f"Playback failed"))

    # ------------------------------------------------------------------
    # Session log export
    # ------------------------------------------------------------------
    def _export_log(self):
        """Export session transcription log to a file."""
        if not self._session_log:
            self._set_status("Ready", self.YELLOW, "No entries to export")
            return

        default_name = f"voice_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        default_dir = str(SESSION_LOG_DIR)

        filepath = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Voice Log",
        )

        if not filepath:
            return

        try:
            with open(filepath, "w") as f:
                f.write("AIWS Voice Input — Session Log\n")
                f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Entries: {len(self._session_log)}\n")
                f.write("=" * 60 + "\n\n")
                for ts, text in self._session_log:
                    f.write(f"[{ts}]\n{text}\n\n")

            self._set_status("Ready", self.GREEN, f"Exported {len(self._session_log)} entries")
            _log(f"Session log exported to {filepath}")
        except Exception as e:
            self._set_status("Error", "#da3633", f"Export failed: {e}")

    # ------------------------------------------------------------------
    # Window visibility (for tray)
    # ------------------------------------------------------------------
    def _toggle_visibility(self):
        if self.root.state() == "withdrawn":
            self.root.deiconify()
            self.root.lift()
        else:
            self.root.withdraw()

    def _minimize_to_tray(self):
        if self._tray._available:
            self.root.withdraw()
        else:
            self.root.iconify()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def _set_status(self, text, dot_color, detail=""):
        self.status_label.config(text=text)
        self.status_dot.config(fg=dot_color)
        self.status_detail.config(text=detail)

    # ------------------------------------------------------------------
    # Hotword toggle
    # ------------------------------------------------------------------
    def _on_hotword_toggle(self):
        if self.hotword_var.get():
            self._hotword.start()
            self._set_status("Ready", self.GREEN, "Hotword listening")
        else:
            self._hotword.stop()
            self._set_status("Ready", self.GREEN, "Hotword off")

    # ------------------------------------------------------------------
    # Speaker verification
    # ------------------------------------------------------------------
    def _get_speaker_verifier(self):
        """Lazy-load the speaker verifier."""
        if self._speaker_verifier is None:
            from jarvis.speaker_verification import SpeakerVerifier
            self._speaker_verifier = SpeakerVerifier(
                gpu=self.gpu_var.get(),
                threshold=self.speaker_threshold_var.get(),
            )
            self._speaker_verifier.load()
        return self._speaker_verifier

    def _on_speaker_threshold_change(self, val=None):
        """Update the threshold label and sync to verifier."""
        thresh = self.speaker_threshold_var.get()
        self._spk_thresh_label.config(text=f"{thresh:.2f}")
        if self._speaker_verifier is not None:
            self._speaker_verifier.threshold = thresh

    def _on_speaker_verify_toggle(self):
        if self.speaker_verify_var.get():
            self._set_status("Loading...", self.YELLOW, "Loading speaker model")

            def _load():
                verifier = self._get_speaker_verifier()
                ok = verifier.load_model()
                def _done():
                    if ok:
                        self._update_voiceprint_label()
                        if verifier.enrolled:
                            self._set_status("Ready", self.GREEN, "Voice ID active")
                        else:
                            self._set_status("Ready", self.YELLOW,
                                             "Voice ID on — enroll your voice first")
                    else:
                        self.speaker_verify_var.set(False)
                        self._set_status("Error", "#da3633", "Speaker model failed")
                self.root.after(0, _done)

            threading.Thread(target=_load, daemon=True).start()
        else:
            self._set_status("Ready", self.GREEN, "Voice ID off")

    def _update_voiceprint_label(self):
        """Update the voiceprint sample count label."""
        verifier = self._get_speaker_verifier()
        if verifier.enrolled:
            n = verifier.num_samples
            self._voiceprint_label.config(
                text=f"{n} sample{'s' if n != 1 else ''}",
                fg=self.GREEN,
            )
        else:
            self._voiceprint_label.config(text="No voiceprint", fg=self.MUTED)

    def _start_enrollment(self):
        """Record a voice enrollment sample."""
        if self.recording:
            self._set_status("Stop recording first", self.YELLOW, "")
            return

        # Ensure model is loaded
        verifier = self._get_speaker_verifier()
        if not verifier._model_loaded:
            self._set_status("Loading...", self.YELLOW, "Loading speaker model")

            def _load_then_enroll():
                verifier.load_model()
                self.root.after(0, self._run_enrollment)

            threading.Thread(target=_load_then_enroll, daemon=True).start()
        else:
            self._run_enrollment()

    # Guided enrollment prompts — each captures a different vocal quality
    _ENROLL_PROMPTS = [
        ("Normal voice", "Talk naturally about anything — your day, what you're working on"),
        ("Read aloud", "Read something on your screen out loud, like you're presenting"),
        ("Quieter voice", "Speak a bit softer, like talking to someone nearby"),
        ("Louder / excited", "Speak up, like explaining something you're excited about"),
        ("Commands & questions", "Try short commands and questions: 'Jarvis, show me the logs'"),
    ]

    def _run_enrollment(self):
        """Start the enrollment recording flow with guided prompts."""
        import sounddevice as sd

        # Pause hotword listener
        if self.hotword_var.get() and self._hotword._stream:
            self._hotword.pause()

        verifier = self._get_speaker_verifier()
        n = verifier.num_samples

        # Pick the appropriate guided prompt
        prompt_idx = n % len(self._ENROLL_PROMPTS)
        prompt_title, prompt_hint = self._ENROLL_PROMPTS[prompt_idx]

        self._enroll_btn.config(state=tk.DISABLED, text="Recording...")
        self._set_status(f"Sample #{n + 1}: {prompt_title}", self.BLUE, prompt_hint)

        mic_name = self.mic_var.get()
        mic_idx = self._mic_devices.get(mic_name)

        try:
            dev_info = sd.query_devices(mic_idx, 'input')
            native_rate = int(dev_info['default_samplerate'])
        except Exception:
            native_rate = 44100

        frames = []

        def cb(indata, frame_count, time_info, status):
            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
            frames.append(chunk)

        def _record():
            try:
                stream = sd.InputStream(
                    samplerate=native_rate, channels=CHANNELS,
                    dtype="float32", device=mic_idx,
                    callback=cb, blocksize=int(native_rate * 0.1),
                )
                stream.start()
                # Record for 15 seconds with countdown + prompt reminders
                import time as _time
                for i in range(15):
                    _time.sleep(1.0)
                    remaining = 14 - i
                    # Show prompt for first 3 seconds, then countdown
                    if remaining > 11:
                        hint = prompt_hint
                    else:
                        hint = f"{prompt_title} — {remaining}s left"
                    self.root.after(0, lambda h=hint: self._set_status(
                        "Enrolling...", self.BLUE, h))
                stream.stop()
                stream.close()
            except Exception as e:
                _log(f"Enrollment recording error: {e}")
                self.root.after(0, lambda: self._enrollment_done(None))
                return

            if not frames:
                self.root.after(0, lambda: self._enrollment_done(None))
                return

            audio_raw = np.concatenate(frames).flatten()
            # Resample to 16kHz
            if native_rate != SAMPLE_RATE:
                from scipy.signal import resample
                new_len = int(len(audio_raw) * SAMPLE_RATE / native_rate)
                audio = resample(audio_raw, new_len).astype(np.float32)
            else:
                audio = audio_raw
            self.root.after(0, lambda: self._enrollment_done(audio))

        threading.Thread(target=_record, daemon=True).start()

    def _enrollment_done(self, audio_16k):
        """Process enrollment audio on main thread."""
        self._enroll_btn.config(state=tk.NORMAL, text="Enroll Voice")

        # Resume hotword
        if self.hotword_var.get():
            self._hotword.resume()

        if audio_16k is None:
            self._set_status("Error", "#da3633", "Enrollment failed")
            return

        verifier = self._get_speaker_verifier()
        ok, count = verifier.enroll(audio_16k)
        if ok:
            self._update_voiceprint_label()
            if count < 3:
                # Show next prompt hint
                next_idx = count % len(self._ENROLL_PROMPTS)
                next_title = self._ENROLL_PROMPTS[next_idx][0]
                self._set_status("Ready", self.GREEN,
                                 f"Sample {count} saved! Next: {next_title} "
                                 f"({3 - count} more recommended)")
            elif count < 5:
                self._set_status("Ready", self.GREEN,
                                 f"{count} samples — good! {5 - count} more for best accuracy")
            else:
                self._set_status("Ready", self.GREEN,
                                 f"{count} samples — excellent voiceprint!")
        else:
            self._set_status("Error", "#da3633", "Audio too short or model error")

    def _clear_voiceprint(self):
        """Clear all enrolled voice data."""
        verifier = self._get_speaker_verifier()
        verifier.clear()
        self._update_voiceprint_label()
        self._set_status("Ready", self.GREEN, "Voiceprint cleared")

    def _start_wakeword_training(self):
        """Record samples of user saying 'Hey Jarvis' to train custom wake word model."""
        if self.recording:
            self._set_status("Stop recording first", self.YELLOW, "")
            return

        if self.hotword_var.get() and self._hotword._stream:
            self._hotword.pause()

        self._ww_samples = []
        self._ww_sample_count = 0
        self._ww_total = 10
        self._set_status("Wake Word Training", self.ACCENT,
                         f"Say 'Hey Jarvis' clearly (0/{self._ww_total})")
        self.root.after(500, self._record_wakeword_sample)

    def _record_wakeword_sample(self):
        """Record a single 3-second 'Hey Jarvis' sample."""
        import sounddevice as sd

        if self._ww_sample_count >= self._ww_total:
            self._train_wakeword_model()
            return

        mic_name = self.mic_var.get()
        mic_idx = self._mic_devices.get(mic_name)

        try:
            dev_info = sd.query_devices(mic_idx, 'input')
            native_rate = int(dev_info['default_samplerate'])
        except Exception:
            native_rate = 44100

        frames = []

        def cb(indata, frame_count, time_info, status):
            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
            frames.append(chunk)

        n = self._ww_sample_count + 1
        self._set_status("Recording...", self.ACCENT,
                         f"Say 'Hey Jarvis' now ({n}/{self._ww_total})")

        if self.sound_var.get():
            threading.Thread(target=_play_beep, args=(_BEEP_START,), daemon=True).start()

        def _record():
            try:
                stream = sd.InputStream(
                    samplerate=native_rate, channels=CHANNELS,
                    dtype="float32", device=mic_idx,
                    callback=cb, blocksize=int(native_rate * 0.1),
                )
                stream.start()
                time.sleep(3.0)
                stream.stop()
                stream.close()
            except Exception as e:
                _log(f"Wake word recording error: {e}")
                self.root.after(0, lambda: self._set_status(
                    "Error", "#da3633", str(e)[:40]))
                return

            if self.sound_var.get():
                _play_beep(_BEEP_STOP)

            if frames:
                audio = np.concatenate(frames).flatten()
                if native_rate != 16000:
                    from scipy.signal import resample
                    new_len = int(len(audio) * 16000 / native_rate)
                    audio = resample(audio, new_len).astype(np.float32)
                audio_int16 = (audio * 32767).astype(np.int16)
                self._ww_samples.append(audio_int16)

            self._ww_sample_count += 1
            time.sleep(0.5)
            self.root.after(0, self._record_wakeword_sample)

        threading.Thread(target=_record, daemon=True).start()

    def _train_wakeword_model(self):
        """Train custom verifier model from recorded samples."""
        self._set_status("Training...", self.YELLOW,
                         "Building custom wake word model...")

        def _train():
            try:
                import joblib
                import scipy.io.wavfile as wav_io
                from openwakeword.model import Model as OWWModel
                from openwakeword.custom_verifier_model import (
                    get_reference_clip_features, train_verifier_model)

                save_dir = Path.home() / ".aiws_trainer" / "wakeword_training"
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save positive samples as WAV files
                pos_dir = save_dir / "positive"
                pos_dir.mkdir(exist_ok=True)
                for i, sample in enumerate(self._ww_samples):
                    path = pos_dir / f"hey_jarvis_{i:02d}.wav"
                    wav_io.write(str(path), 16000, sample)

                pos_files = sorted(str(p) for p in pos_dir.glob("*.wav"))

                # Generate negative samples (silence + noise)
                neg_dir = save_dir / "negative"
                neg_dir.mkdir(exist_ok=True)
                for i in range(10):
                    noise = (np.random.randn(16000 * 3) * 1000).astype(np.int16)
                    path = neg_dir / f"noise_{i:02d}.wav"
                    wav_io.write(str(path), 16000, noise)

                neg_files = sorted(str(p) for p in neg_dir.glob("*.wav"))

                _log(f"Training custom verifier: {len(pos_files)} positive, "
                     f"{len(neg_files)} negative")

                oww = OWWModel()
                model_name = "hey_jarvis"

                pos_features = np.vstack([
                    get_reference_clip_features(f, oww, model_name,
                                                threshold=0.3, N=3)
                    for f in pos_files
                ])

                neg_features = np.vstack([
                    get_reference_clip_features(f, oww, model_name,
                                                threshold=0.0, N=1)
                    for f in neg_files
                ])

                _log(f"Features: {pos_features.shape[0]} positive, "
                     f"{neg_features.shape[0]} negative")

                all_features = np.vstack((pos_features, neg_features))
                all_labels = np.array(
                    [1] * pos_features.shape[0]
                    + [0] * neg_features.shape[0])

                model = train_verifier_model(all_features, all_labels)

                # Save with joblib (sklearn standard)
                model_path = (Path.home() / ".aiws_trainer"
                              / "hey_jarvis_verifier.pkl")
                joblib.dump(model, str(model_path))
                _log(f"Custom verifier saved: {model_path}")

                self.root.after(0, lambda: self._set_status(
                    "Ready", self.GREEN,
                    f"Custom wake word trained! "
                    f"({pos_features.shape[0]} features)"))

            except Exception as e:
                _log(f"Wake word training error: {e}")
                self.root.after(0, lambda: self._set_status(
                    "Error", "#da3633",
                    f"Training failed: {str(e)[:40]}"))

            if self.hotword_var.get():
                self.root.after(0, self._hotword.resume)

        threading.Thread(target=_train, daemon=True).start()

    # ------------------------------------------------------------------
    # Window target system
    # ------------------------------------------------------------------
    def _get_window_list(self):
        """Get list of (wid, name) for all visible windows (single wmctrl call)."""
        windows = []
        own_wid = None
        try:
            own_wid_int = self.root.winfo_id()
            # wmctrl returns hex WIDs (0x0...); xdotool uses decimal.
            # Convert our own decimal WID to the hex wmctrl emits.
            own_wid = f"0x{own_wid_int:08x}"
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["wmctrl", "-l"], capture_output=True, text=True, timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            _log(f"Window list error (wmctrl missing/timeout): {e}")
            return []
        if result.returncode != 0:
            return []

        for line in result.stdout.strip().splitlines():
            # Format: "<wid> <desktop> <host> <title...>"
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            wid, _, _, name = parts
            if own_wid and wid.lower() == own_wid.lower():
                continue
            if name and len(name) > 1:
                windows.append((wid, name))
        return windows

    def _refresh_window_list(self):
        """Refresh the window dropdown with current open windows."""
        windows = self._get_window_list()
        self._window_map = {}  # display_name -> wid
        display_names = ["Auto (Claude)"]
        for wid, name in windows:
            short = name[:40] + "..." if len(name) > 40 else name
            display_names.append(short)
            self._window_map[short] = (wid, name)
        self._window_combo["values"] = display_names
        _log(f"Window list refreshed: {len(windows)} windows")

    def _on_window_selected(self, event=None):
        """Handle window selection from dropdown."""
        selection = self._window_combo.get()
        if selection == "Auto (Claude)" or not selection:
            self._reset_target()
            return
        entry = self._window_map.get(selection)
        if entry:
            wid, name = entry
            self._target_wid = wid
            self._target_name = name
            short = name[:25] + "..." if len(name) > 25 else name
            self._target_display.config(text=short, fg=self.BLUE)
            self._set_status("Ready", self.GREEN, f"Target: {short}")
            self._save_settings()
            _log(f"Target set to WID={wid} name={name!r}")

    def _pick_window(self):
        """Let user click on any window to select it as target."""
        self._set_status("Pick window...", self.YELLOW, "Click on any window")
        self.root.iconify()  # Minimize so user can click other windows

        def _do_pick():
            try:
                result = subprocess.run(
                    ["xdotool", "selectwindow"],
                    capture_output=True, text=True, timeout=30,
                )
                wid = result.stdout.strip()
                if wid:
                    name_result = subprocess.run(
                        ["xdotool", "getwindowname", wid],
                        capture_output=True, text=True, timeout=2,
                    )
                    name = name_result.stdout.strip() or f"Window {wid}"
                    self._target_wid = wid
                    self._target_name = name
                    short = name[:25] + "..." if len(name) > 25 else name
                    def _apply(s=short):
                        self._target_display.config(text=s, fg=self.BLUE)
                        self._set_status("Ready", self.GREEN, f"Target: {s}")
                    self.root.after(0, _apply)
                    _log(f"Picked target WID={wid} name={name!r}")
            except Exception as e:
                _log(f"Pick window error: {e}")
                self.root.after(0, lambda: self._set_status(
                    "Ready", self.YELLOW, "Pick cancelled"))
            finally:
                self.root.after(100, self.root.deiconify)

        threading.Thread(target=_do_pick, daemon=True).start()

    def _reset_target(self):
        """Reset to auto-detect Claude terminal."""
        self._target_wid = None
        self._target_name = None
        self._target_display.config(text="Auto (Claude)", fg=self.GREEN)
        self._window_combo_var.set("")
        self._set_status("Ready", self.GREEN, "Target: Auto (Claude)")
        self._save_settings()
        _log("Target reset to Auto (Claude)")

    def _restore_target(self, name):
        """Re-find a previously pinned target window by name."""
        windows = self._get_window_list()
        for wid, wname in windows:
            if name.lower() in wname.lower():
                self._target_wid = wid
                self._target_name = name
                short = name[:25]
                self._target_display.config(text=short, fg="#67e8f9")
                _log(f"Restored target: {name} (WID {wid})")
                return
        _log(f"Could not restore target '{name}', using Auto")

    def _voice_target_window(self, query):
        """Find a window matching the spoken query and pin it as target.

        Matches by case-insensitive substring against window titles.
        E.g. "opera" matches "Opera Browser", "terminal" matches
        "hunterp@...: ~/vss_env".
        """
        _log(f"Voice target search: {query!r}")
        windows = self._get_window_list()

        # Score windows: prefer exact word boundary matches over substrings
        best_wid = None
        best_name = None
        best_score = 0

        for wid, name in windows:
            name_lower = name.lower()
            # Exact start-of-title match
            if name_lower.startswith(query):
                score = 3
            # Word boundary match
            elif re.search(r'\b' + re.escape(query) + r'\b', name_lower):
                score = 2
            # Substring match
            elif query in name_lower:
                score = 1
            else:
                continue

            if score > best_score:
                best_score = score
                best_wid = wid
                best_name = name

        if best_wid:
            self._target_wid = best_wid
            self._target_name = best_name
            short = best_name[:25] + "..." if len(best_name) > 25 else best_name
            self._target_display.config(text=short, fg=self.BLUE)
            self._set_status("Ready", self.GREEN, f"Target → {short}")
            _log(f"Voice targeted: WID={best_wid} name={best_name!r}")
        else:
            self._set_status("Ready", self.YELLOW, f"No window matching '{query}'")
            _log(f"Voice target: no match for {query!r}")

    # ------------------------------------------------------------------
    # Custom vocabulary editor
    # ------------------------------------------------------------------
    def _open_vocab_editor(self):
        """Open a popup window to edit the domain vocabulary."""
        editor = tk.Toplevel(self.root)
        editor.title("Edit Vocabulary")
        editor.geometry("460x400")
        editor.configure(bg=self.BG)
        editor.transient(self.root)

        tk.Label(
            editor, text="Domain Vocabulary",
            font=("Arial", 14, "bold"), bg=self.BG, fg=self.ACCENT,
        ).pack(padx=16, pady=(12, 4))

        tk.Label(
            editor,
            text="Comma-separated terms that bias Whisper toward your domain.\n"
                 "These words will be recognized more accurately.",
            font=("Arial", 9), bg=self.BG, fg=self.MUTED,
            justify="left",
        ).pack(padx=16, pady=(0, 8), anchor="w")

        text_widget = tk.Text(
            editor, font=("Consolas", 10),
            bg="#1a2332", fg=self.TEXT,
            insertbackground=self.TEXT,
            wrap=tk.WORD, padx=10, pady=8,
        )
        text_widget.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # Load current vocab
        current = _load_vocab()
        text_widget.insert("1.0", current)

        btn_frame = tk.Frame(editor, bg=self.BG)
        btn_frame.pack(fill="x", padx=16, pady=(0, 12))

        def _save():
            new_vocab = text_widget.get("1.0", tk.END).strip()
            _save_vocab(new_vocab)
            self._set_status("Ready", self.GREEN, "Vocabulary saved")
            editor.destroy()

        def _reset():
            text_widget.delete("1.0", tk.END)
            text_widget.insert("1.0", DEFAULT_VOCAB)

        self._holo_btn(btn_frame, "Reset to Default", _reset).pack(side="left")
        self._holo_btn(
            btn_frame, "Save", _save, font=("Arial", 9, "bold"), width=12,
        ).pack(side="right")

    # ------------------------------------------------------------------
    # Confidence display
    # ------------------------------------------------------------------
    def _display_with_confidence(self, segments_data):
        """Display transcription with per-segment confidence coloring.

        segments_data: list of (text, avg_logprob) tuples
        """
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)

        total_logprob = 0
        total_segs = 0

        for text, logprob in segments_data:
            if not text.strip():
                continue

            # avg_logprob: 0 = perfect, -1 = very bad
            # > -0.3 = high confidence, -0.3 to -0.7 = medium, < -0.7 = low
            if logprob > -0.3:
                tag = "conf_high"
            elif logprob > -0.7:
                tag = "conf_med"
            else:
                tag = "conf_low"

            self.output_text.insert(tk.END, text.strip() + " ", tag)
            total_logprob += logprob
            total_segs += 1

        self.output_text.config(state=tk.DISABLED)

        # Update confidence bar
        if total_segs > 0:
            avg = total_logprob / total_segs
            # Normalize: -1.0 → 0%, 0.0 → 100%
            pct = max(0, min(100, int((avg + 1.0) * 100)))
            self._update_conf_bar(pct)

    def _update_conf_bar(self, pct):
        """Draw the confidence percentage bar."""
        self.conf_canvas.delete("all")
        w = self.conf_canvas.winfo_width()
        h = self.conf_canvas.winfo_height()
        fill_w = int(w * pct / 100)

        if pct >= 70:
            color = self.GREEN
        elif pct >= 40:
            color = self.YELLOW
        else:
            color = "#da3633"

        if fill_w > 0:
            self.conf_canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="")

        self.conf_label.config(text=f"{pct}%")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _on_close(self):
        # If launched by daemon, minimize to tray instead of closing
        # so next "Jarvis" is instant (no model reload)
        if self._daemon_launched and self._tray._available:
            self._minimize_to_tray()
            _log("Minimized to tray (daemon mode — staying alive for fast wake)")
            return
        self._cleanup()
        if self.on_close_callback:
            self.on_close_callback()
        else:
            self.root.destroy()

    def _on_window_destroy(self, event):
        """Force-kill entire process when root window is destroyed (e.g. xkill)."""
        if event.widget is not self.root:
            return
        self._cleanup()
        os._exit(0)

    def _cleanup(self):
        """Release all resources — mic, hotword, tray, global hotkey, browser."""
        self.recording = False
        # Shut down orbit server if running
        if hasattr(self, '_orbit_server') and self._orbit_server:
            try:
                self._orbit_server.shutdown()
            except Exception:
                pass
        try:
            self._hotword.stop()
        except Exception:
            pass
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        try:
            self._global_hotkey.stop()
        except Exception:
            pass
        try:
            self._tray.stop()
        except Exception:
            pass


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    auto_record = "--auto-record" in sys.argv
    root = tk.Tk()
    root.title("Voice Input")
    app = VoiceInputGUI(root, on_close_callback=root.destroy, auto_record=auto_record)
    try:
        root.mainloop()
    except (tk.TclError, KeyboardInterrupt):
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
