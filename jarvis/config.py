"""Configuration, paths, and machine profile for Jarvis V3.

CONFIG is the single source of truth for settings. The UI binds Tk variables
to it as views. File format is backward-compatible with V1's
~/.aiws_trainer/voice_settings.json — old keys map 1:1 and unknown keys are
preserved across save().
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path

from jarvis.logs import get_logger

log = get_logger("config")


# ---------------------------------------------------------------- paths
class PATHS:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    AIWS = Path.home() / ".aiws_trainer"
    LOG_DIR = Path(os.environ.get("JARVIS_LOG_DIR") or "/tmp/vss_voice")
    SCREEN_DIR = Path("/tmp/vss_screen")
    SETTINGS_FILE = AIWS / "voice_settings.json"
    VOCAB_FILE = AIWS / "voice_vocab.txt"
    # JARVIS_MEMORY_DIR keeps the test suite (tests/conftest.py) out of the
    # user's real ~/.aiws_trainer/jarvis_memory: the timekeeper / notes /
    # claude-project state, the typed history and the memory store all hang
    # off it. Unset in production, so the live app is unaffected.
    MEMORY_DIR = Path(os.environ.get("JARVIS_MEMORY_DIR") or
                      (AIWS / "jarvis_memory"))
    # migrated into MEMORY_DIR once; JARVIS_LEGACY_DIR keeps the suite off the
    # user's real ~/.aiws_trainer/jarvis_data.
    LEGACY_AGENT_DIR = Path(os.environ.get("JARVIS_LEGACY_DIR") or
                            (AIWS / "jarvis_data"))
    VOICEPRINT = AIWS / "voiceprint.npz"
    HEY_JARVIS_VERIFIER = AIWS / "hey_jarvis_verifier.pkl"
    SPEAK_QUEUE = LOG_DIR / "speak_queue.txt"
    VOICE_REF = AIWS / "jarvis_voice_ref.wav"
    VSS_ENV = Path.home() / "vss_env"
    REMINDERS = MEMORY_DIR / "reminders.json"
    # -- personal assistant (spec 2026-08-26, section 3.2) --------------
    # Env overrides exist so the test suite (tests/conftest.py) can firewall
    # the real config file and cache; state files live under MEMORY_DIR,
    # runtime files (socket, task streams, MCP config) under LOG_DIR.
    ASSISTANT_CONFIG = Path(os.environ.get("JARVIS_ASSISTANT_CONFIG") or
                            (Path.home() / ".config" / "jarvis" / "assistant.json"))
    TIMEKEEPER_DB = MEMORY_DIR / "timekeeper.db"
    NOTES_DB = MEMORY_DIR / "notes.db"
    CLAUDE_PROJECTS = MEMORY_DIR / "claude_projects.json"
    CACHE_DIR = Path(os.environ.get("JARVIS_CACHE_DIR") or
                     (Path.home() / ".cache" / "jarvis"))
    APPROVALS_SOCK = LOG_DIR / "approvals.sock"
    CLAUDE_TASK_DIR = LOG_DIR / "claude"
    MCP_CONFIG = LOG_DIR / "mcp_jarvis.json"
    AUTOSTART_DESKTOP = Path.home() / ".config" / "autostart" / "jarvis.desktop"


# ------------------------------------------------------------- settings
@dataclass
class Config:
    model: str = "small"
    gpu: int = 0
    mic: str = "Default"
    language: str = "English"
    auto_type: bool = True
    continuous: bool = True
    sound: bool = False
    review: bool = False
    voice_cmds: bool = True
    noise_gate: bool = True
    streaming: bool = True
    hotword: bool = True
    smart_target: bool = True
    auto_enter: bool = True
    live_write: bool = False
    talkback: bool = True
    jarvis_mode: bool = True
    tts_engine: str = "edge"
    speaker_verify: bool = False
    speaker_threshold: float = 0.4
    target_name: str = ""
    # Tuned 2026-08-27 for wake-word commands. At 5.0/8.0 a two-second
    # question held the mic for ~13 s before auto-stopping. Raise both for
    # dictation, where long pauses are normal and being cut off is worse
    # than waiting.
    silence_timeout: float = 2.5
    silence_grace: float = 1.5     # no auto-stop this soon after start
    noise_threshold: float = 0.015
    window_geometry: str = ""

    def __post_init__(self):
        self._extra: dict = {}
        self._lock = threading.Lock()
        self._save_cb = None      # app wires this to debounce disk writes

    # -- persistence ---------------------------------------------------
    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        try:
            data = json.loads(PATHS.SETTINGS_FILE.read_text())
        except Exception:
            return cfg
        known = {f.name: f.type for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                try:
                    setattr(cfg, key, type(getattr(cfg, key))(value))
                except Exception:
                    log.warning("bad settings value %s=%r; using default", key, value)
            else:
                cfg._extra[key] = value
        return cfg

    def save(self):
        with self._lock:
            data = {f.name: getattr(self, f.name) for f in fields(self)}
            data.update(self._extra)
        try:
            PATHS.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PATHS.SETTINGS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, PATHS.SETTINGS_FILE)
        except Exception:
            log.exception("settings save failed")

    def update(self, **kw):
        """Thread-safe field update + save."""
        with self._lock:
            for key, value in kw.items():
                setattr(self, key, value)
        self.save()


# ------------------------------------------------------- machine profile
def _alsa_has_capture(path: Path = Path("/proc/asound/pcm")) -> bool:
    """True when the kernel reports at least one capture-capable PCM.

    /proc/asound/pcm lists one line per PCM device, suffixed "playback N"
    and/or "capture N". With no microphone attached there are no capture lines
    at all -- the same fact `arecord -l` reports, without the subprocess.
    """
    try:
        return any("capture" in line for line in path.read_text().splitlines())
    except Exception:
        return False


@dataclass
class MachineProfile:
    no_cuda_ct2: bool = True
    gpu_count: int = 0
    gpu_name: str = ""
    is_gb10: bool = False
    has_mic: bool = False
    mic_names: list = field(default_factory=list)
    claude_bin: str = ""
    arch: str = ""

    @classmethod
    def detect(cls) -> "MachineProfile":
        m = cls(arch=platform.machine())
        try:
            import ctranslate2
            m.no_cuda_ct2 = ctranslate2.get_cuda_device_count() == 0
        except Exception:
            m.no_cuda_ct2 = True
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3)
            names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            m.gpu_count = len(names)
            m.gpu_name = names[0] if names else ""
            m.is_gb10 = "GB10" in m.gpu_name
        except Exception:
            pass
        try:
            import sounddevice as sd
            for dev in sd.query_devices():
                if dev.get("max_input_channels", 0) > 0 and \
                        dev.get("name", "") not in ("default", "pipewire", "pulse"):
                    m.mic_names.append(dev["name"])
        except Exception:
            pass
        # pipewire/default expose virtual inputs even with NO hardware mic, so
        # a name match alone would claim a mic that isn't there. But the
        # converse also happens: while PipeWire holds a USB mic exclusively,
        # PortAudio cannot probe hw:N and enumerates only those same virtual
        # names -- capture through them works, yet every name gets filtered and
        # voice would be disabled with a live mic plugged in. ALSA's own PCM
        # table is authoritative in both directions, so it breaks the tie.
        m.has_mic = bool(m.mic_names) or _alsa_has_capture()
        # A GNOME autostart launch may not have ~/.local/bin on PATH, so
        # which() can come back empty at login and every Claude submit would
        # get the setup line; that install location is the last resort
        # (mirrors assistant_config._claude_bin).
        local = Path.home() / ".local" / "bin" / "claude"
        m.claude_bin = os.environ.get("JARVIS_CLAUDE_BIN") or \
            shutil.which("claude") or \
            (str(local) if os.access(local, os.X_OK) else "")
        log.info("machine: %s", m)
        return m


CONFIG = Config.load()
MACHINE = MachineProfile.detect()
