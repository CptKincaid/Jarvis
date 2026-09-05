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
    # Names taught by voice ("my advisor's name is spelled P-E-Y-R-O-V-I")
    # and "add X to your vocabulary"; one per line, append-only. Read by
    # jarvis/vocab.py into the Whisper prompt alongside VOCAB_FILE.
    NAMES_FILE = AIWS / "voice_names.txt"
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
    # JARVIS_VOICEPRINT keeps the suite off the user's enrolled voice.
    # On 2026-09-02 a test built a real SpeakerVerifier and called
    # enroll_from_audio, whose save() writes this global: it replaced his
    # 6-sample pool with two copies of the fixture's constant vector
    # (every element 0.07216878), against which his own enrolment clips
    # score 0.055-0.125 under a 0.30 threshold -- voice dead at the next
    # restart. Unset in production, so the live app is unaffected.
    VOICEPRINT = Path(os.environ.get("JARVIS_VOICEPRINT") or
                      (AIWS / "voiceprint.npz"))
    # JARVIS_FACE_GALLERY keeps the suite off his enrolled FACE, for exactly
    # the reason the line above exists: a face embedding is biometric data
    # about one person and, unlike a password, cannot be re-issued. A
    # DIRECTORY rather than a file because the gallery is generational --
    # jarvis/facegallery.py never overwrites, so a bad write costs one
    # generation instead of the enrolment, which is the part the voiceprint
    # did not have on 2026-09-02. Unset in production.
    FACE_GALLERY = Path(os.environ.get("JARVIS_FACE_GALLERY") or
                        (AIWS / "face_gallery"))
    # The MULTI-SPEAKER voice gallery (jarvis/voicegallery.py). A DIRECTORY,
    # generational, and deliberately NOT voiceprint.npz: writing several
    # people's vectors into that file would let any build with the old
    # single-speaker loader pool them into one centroid and admit all of them
    # as him (measured -- see speaker.KNOWN_VOICEPRINT_FORMATS). voiceprint.npz
    # stays exactly where it is as the rollback; this is the new store, with
    # its own format namespace.
    #
    # JARVIS_VOICE_GALLERY exists for the same reason JARVIS_VOICEPRINT does,
    # and it was added BEFORE the store could be written to rather than after
    # a test destroyed it: on 2026-09-02 that lesson cost him his enrolment.
    VOICE_GALLERY = Path(os.environ.get("JARVIS_VOICE_GALLERY") or
                         (AIWS / "voice_gallery"))
    HEY_JARVIS_VERIFIER = AIWS / "hey_jarvis_verifier.pkl"
    SPEAK_QUEUE = LOG_DIR / "speak_queue.txt"
    # Chosen by ear 2026-08-28: 35.8s built from three different Fish
    # Audio renderings of Bettany. The old film clip is kept beside it
    # (jarvis_voice_ref.wav) so this is a one-line rollback.
    VOICE_REF = AIWS / "jarvis_voice_ref_fish3.wav"
    # F5-TTS reference. Flow-matching infill needs the reference TEXT
    # as well as the audio, so the transcript sits beside the wav.
    VOICE_REF_F5 = AIWS / "jarvis_voice_ref_f5.wav"
    VOICE_REF_F5_TEXT = AIWS / "jarvis_voice_ref_f5.txt"
    # f5-tts cannot share vss_env: installing it there removes fastapi,
    # which VSS's own api.py needs. It gets its own venv and a sidecar.
    F5_PYTHON = Path.home() / ".local/share/jarvis-f5/venv/bin/python"
    F5_SOCK = LOG_DIR / "f5.sock"
    # Breeze-TTS-2, quantized (int4 MLPs, group-32 depth, bf16 attention and
    # text encoder), pre-quantized so it loads in 7.9 s instead of 38 s. Its
    # own venv and source tree for the same reason F5 has them, and its own
    # sidecar for a stronger one: the one-time CUDA-graph capture transiently
    # demands ~18.3 GB, which must never happen inside the Tk app. It shares
    # F5's reference clip and transcript (VOICE_REF_F5 / VOICE_REF_F5_TEXT) --
    # that is the pair the round-11 blind test rated 4.71.
    BREEZE_PYTHON = (Path.home() /
                     "voice-training/engines/breeze/venv/bin/python")
    BREEZE_REPO = Path.home() / "voice-training/engines/breeze-q4/repo"
    BREEZE_CKPT = Path.home() / "voice-training/engines/breeze-q4/ckpt-q4"
    BREEZE_SOCK = LOG_DIR / "breeze.sock"
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
    # jarvis/cmdsock.py: ask Jarvis from a shell (python -m jarvis.ask)
    COMMAND_SOCK = LOG_DIR / "command.sock"
    # jarvis/dayreview.py: one digest per day; MEMORY_DIR because LOG_DIR
    # is tmpfs and the review is meant to outlive the boot that wipes it
    REVIEWS_DIR = MEMORY_DIR / "reviews"
    CLAUDE_TASK_DIR = LOG_DIR / "claude"
    MCP_CONFIG = LOG_DIR / "mcp_jarvis.json"
    AUTOSTART_DESKTOP = Path.home() / ".config" / "autostart" / "jarvis.desktop"
    # jarvis/zones.py: the zone TRANSITION LOG. XDG state rather than
    # LOG_DIR because /tmp is wiped at every boot on this box (a tmpfiles
    # `D /tmp` rule) and the whole point of a record is that it outlives
    # one. JARVIS_STATE_DIR keeps the suite (tests/conftest.py) out of the
    # real ~/.local/state/jarvis, which says who was in which room and
    # when. The directory is created 0700 and the file 0600 by ZoneLog.
    STATE_DIR = Path(os.environ.get("JARVIS_STATE_DIR") or
                     (Path.home() / ".local" / "state" / "jarvis"))
    # jarvis/identity.py: WHO Jarvis knows -- one row per person, with the
    # salted hashes of the two fallbacks (the spoken passphrase and the
    # typed override code). Beside zones.jsonl and NOT inside
    # assistant.json on purpose: that object is deep-copied into reports
    # and prints redacted() from its __repr__, so the strongest way to
    # keep a hash out of a log is to keep it out of that object. Written
    # 0600 in a 0700 directory by Registry.save.
    OWNER_REGISTRY = STATE_DIR / "people.json"


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
    speaker_threshold: float = 0.3      # measured 2026-08-29; see speaker.DEFAULT_THRESHOLD
    target_name: str = ""
    # Tuned 2026-08-27 for wake-word commands. At 5.0/8.0 a two-second
    # question held the mic for ~13 s before auto-stopping. Raise both for
    # dictation, where long pauses are normal and being cut off is worse
    # than waiting.
    silence_timeout: float = 2.5
    silence_grace: float = 1.5     # no auto-stop this soon after start
    # Speech endpointing (jarvis/endpoint.py): stop this long after the VAD
    # last heard speech, instead of waiting out silence_timeout. 0.8 s is the
    # usual assistant figure; raise it if a thinking pause mid-question cuts
    # you off, lower it if the wait after a question feels long.
    endpoint_vad: bool = True
    endpoint_silence: float = 0.8
    # The filler hold (jarvis/recorder.py, Hunter 08-31: "um/uh should buy
    # him more time"). When the live preview's newest decode ends on a
    # filler ("set a timer for, um…") and the VAD heard nothing after it,
    # the stop waits filler_hold_s beyond endpoint_silence -- so a thinking
    # pause after an um is 2.3 s, not 0.8 s. A hold is one such pause; at
    # most filler_max_holds of them per capture, then the ordinary stop.
    # Raise filler_hold_s if he still gets cut off after an um, lower it if
    # the wait after "…um" feels long; 0 holds nothing. The 60 s cap and the
    # 2.5 s energy timer are untouched (the energy timer can still end a
    # held pause first: the turn line then says stop=energy).
    filler_hold: bool = True
    filler_hold_s: float = 1.5
    filler_max_holds: int = 3
    # Add "Um, uh, hmm, er." to the PREVIEW's initial_prompt so whisper
    # writes fillers down instead of dropping them (it is trained on clean
    # transcripts). The final transcribe() never carries it, so commands
    # stay clean. UNMEASURED, so it ships OFF: scripts/filler_probe.py
    # compares takes with it on and off; turn it on only if that says so.
    filler_prompt_hint: bool = False
    # After Jarvis answers, keep listening this long for a follow-up with no
    # wake word ("...and Tuesday?"). Needs the VAD (it must know that nothing
    # was said); 0 turns it off. Every follow-up is still speaker-verified.
    followup_window: float = 4.0
    # Speak each sentence of a model reply as it is generated instead of
    # waiting for the whole reply.
    stream_replies: bool = True
    # PulseAudio/PipeWire sink for speech ("" = the default). Set to the
    # echo-cancelling sink so barge-in hears you over Jarvis.
    playback_device: str = ""
    # Barge-in: keep the wake word live while Jarvis speaks so "Jarvis, stop"
    # cuts him off. The speaker gate keeps his own voice from waking him.
    barge_in: bool = True
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
