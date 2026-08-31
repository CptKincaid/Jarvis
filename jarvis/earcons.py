"""The earcon lexicon: six short tones in one family, so the room has an
accent instead of a drawer of unrelated beeps.

This is a REFACTOR, not greenfield. Jarvis already had a synthesized beep
vocabulary in ``recorder.py`` -- start 880 Hz, stop 660 Hz, nudge 440 Hz --
and the 440 nudge was already a semantic earcon ("I heard you but I am not
answering"). Those three pitches turn out to be a root, its fifth and its
octave, so the family here is *derived from what the room already sounds
like* rather than invented beside it: 220/440 root, 330/660 fifth,
880/1320 octave and twelfth, one timbre, one 10 ms raised-cosine envelope.
``recorder.play_beep`` now delegates here, so there is exactly one beep
system.

The six names and what publishes them today:

===========  =========================================================
heard-you    the wake word was accepted and the mic is opening
             (recorder "start")
done         a capture closed cleanly (recorder "stop")
held-back    "I heard you and I have nothing to answer" -- the nudge
             policy, and the lineage of the old 440 Hz tone
arrival      he came back and the room straightened up (jarvis/arrival.py)
thinking     RENDERED, DELIBERATELY UNWIRED
warning      RENDERED, DELIBERATELY UNWIRED
===========  =========================================================

The last two are rendered so the lexicon is complete and auditionable, and
wired to nothing on purpose. ``BrainState`` flips constantly and the
reactor already shows thinking visually (ui/reactor.py); an assistant that
chirps at every internal state transition reads as a machine with a
nervous tic, not a butler with an accent. And a grave warning tone whose
only candidate publisher is the health watchdog -- whose lines are
PROACTIVE and therefore held by quiet hours -- would sound at 3 a.m.
exactly when the speech it accompanies was suppressed. Give either one a
real publisher before wiring it.

Gating: ``sound.earcons`` in assistant.json (default true), NOT
``CONFIG.sound`` -- that flag is False by default and means "the old
start/stop chimes", so hanging the lexicon off it would ship it mute. The
config arrives through ``set_config()`` at app boot, the way
``channels.notify.set_quiet_gate`` does; with no config installed the
lexicon plays (a test or a script gets sound, not silence).

Rate limiting lives HERE, not at the call sites, but it is per-name rather
than one cue across all causes: a shared global cooldown would swallow the
"done" tone that closes a capture two seconds after "heard-you" opened it,
which is a worse failure than the one it prevents. A repeat of the SAME
tone inside ``sound.cooldown_s`` is dropped -- that is the false-wake
metronome the rule exists for -- and ``MIN_GAP_S`` stops two different
tones stacking on top of each other.

Rendering is deterministic and happens once: the WAVs live under
``PATHS.MEMORY_DIR/earcons`` (not LOG_DIR -- /tmp is wiped at boot on this
box) and are only written when missing. ``scripts/make_earcons.py`` bakes
them ahead of time and can dump them anywhere for an audition.
"""
from __future__ import annotations

import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("earcons")

SAMPLE_RATE = 22050
ENVELOPE_MS = 10.0           # raised-cosine attack and release, per tone
MIN_GAP_S = 0.25             # two DIFFERENT tones may not stack
DEFAULT_COOLDOWN_S = 4.0     # a repeat of the SAME tone is dropped inside this
DEFAULT_VOLUME = 0.5

# The family. 440 was the old nudge, 660 the old stop, 880 the old start:
# root, fifth, octave. Everything else is those three transposed.
ROOT_LOW, FIFTH_LOW = 220.0, 330.0
ROOT, FIFTH, OCTAVE, TWELFTH = 440.0, 660.0, 880.0, 1320.0

# name -> (notes as (hz, ms), gain). Every tone is under 400 ms.
TONES: dict[str, tuple[tuple[tuple[float, float], ...], float]] = {
    # rising fifth -> octave: "I have you", landing on the old start pitch
    "heard-you": (((FIFTH, 70.0), (OCTAVE, 110.0)), 0.55),
    # quiet, unresolved, sits under speech
    "thinking": (((ROOT, 90.0), (FIFTH, 90.0)), 0.30),
    # falling to the old 440 nudge: "heard, and nothing to say"
    "held-back": (((FIFTH, 90.0), (ROOT, 190.0)), 0.45),
    # up through the family and out the top
    "done": (((FIFTH, 80.0), (OCTAVE, 80.0), (TWELFTH, 120.0)), 0.50),
    # the only tone below the root, and the only falling one: grave
    "warning": (((FIFTH_LOW, 150.0), (ROOT_LOW, 220.0)), 0.60),
    # the bloom: the whole family in order, warm and unhurried
    "arrival": (((ROOT, 90.0), (FIFTH, 90.0), (OCTAVE, 180.0)), 0.50),
}
NAMES = tuple(TONES)

# The old kinds, kept so recorder.play_beep()'s callers need not change.
ALIASES = {"start": "heard-you", "stop": "done", "nudge": "held-back"}

_lock = threading.Lock()
_last_played: dict[str, float] = {}
_last_any: float = 0.0
_cfg = None
_clock = time.monotonic          # test seam


def set_config(cfg) -> None:
    """Install the assistant config (app boot). None restores "no config"."""
    global _cfg
    _cfg = cfg


def _get(key: str, default):
    get = getattr(_cfg, "get", None)
    if not callable(get):
        return default
    try:
        value = get(key, default)
    except Exception:            # noqa: BLE001 - config boundary
        log.debug("earcons: cfg.get(%s) failed", key, exc_info=True)
        return default
    return default if value is None else value


def enabled() -> bool:
    return bool(_get("sound.earcons", True))


def resolve(name: str) -> str:
    """The lexicon name for a caller's word ("" when there is no such tone)."""
    key = str(name or "").strip().lower().replace("_", "-")
    key = ALIASES.get(key, key)
    return key if key in TONES else ""


# ---------------------------------------------------------------- synthesis
def _samples(notes, gain: float):
    """One tone as int16 PCM (a numpy array).

    Deterministic by construction -- float64 throughout, no RNG, no clock --
    so the same spec always renders the same bytes and a WAV on disk can be
    trusted forever.
    """
    import numpy as np

    edge = max(1, int(SAMPLE_RATE * ENVELOPE_MS / 1000.0))
    chunks = []
    for freq, ms in notes:
        n = max(1, int(SAMPLE_RATE * ms / 1000.0))
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        # One timbre for the whole family: fundamental, a quiet octave and a
        # trace of the twelfth. This is the accent.
        wave_ = (np.sin(2 * np.pi * freq * t)
                 + 0.25 * np.sin(4 * np.pi * freq * t)
                 + 0.08 * np.sin(6 * np.pi * freq * t)) / 1.33
        # Raised cosine at both ends: a hard edge clicks, and a click is the
        # one thing a "voice for the room" cannot have.
        env = np.ones(n, dtype=np.float64)
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(min(edge, n)) / edge)
        env[:len(ramp)] = ramp
        env[n - len(ramp):] = np.minimum(env[n - len(ramp):], ramp[::-1])
        chunks.append(gain * env * wave_)
    sig = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float64)
    return np.clip(sig * 32767.0, -32767, 32767).astype("<i2")


def render(name: str, directory: Optional[Path] = None,
           force: bool = False) -> Optional[Path]:
    """Bake one tone to a WAV and return its path (None for an unknown name).

    Written only when missing, so this is a no-op after the first boot.
    """
    key = resolve(name)
    if not key:
        return None
    out_dir = Path(directory) if directory else (PATHS.MEMORY_DIR / "earcons")
    path = out_dir / f"{key}.wav"
    if path.exists() and not force:
        return path
    notes, gain = TONES[key]
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = _samples(notes, gain).tobytes()
        tmp = path.with_name(path.name + ".tmp")
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(frames)
        tmp.replace(path)        # atomic: paplay never opens a half-written WAV
    except OSError:
        log.debug("earcons: %s could not be rendered", key, exc_info=True)
        return None
    return path


def render_all(directory: Optional[Path] = None, force: bool = False) -> dict[str, Path]:
    """Bake the whole lexicon; returns name -> path for what was written."""
    out = {}
    for name in NAMES:
        path = render(name, directory, force)
        if path is not None:
            out[name] = path
    return out


# ------------------------------------------------------------------ playback
def _volume() -> int:
    try:
        vol = float(_get("sound.volume", DEFAULT_VOLUME))
    except (TypeError, ValueError):
        vol = DEFAULT_VOLUME
    return int(max(0.0, min(1.0, vol)) * 65536)


def _allowed(key: str) -> bool:
    """The shared rate limit. See the module docstring for why it is
    per-name plus a small global gap rather than one cue for all causes."""
    global _last_any
    try:
        cooldown = float(_get("sound.cooldown_s", DEFAULT_COOLDOWN_S))
    except (TypeError, ValueError):
        cooldown = DEFAULT_COOLDOWN_S
    now = _clock()
    with _lock:
        if now - _last_played.get(key, -1e9) < cooldown:
            log.debug("earcon %s suppressed: within %.1fs of the last", key, cooldown)
            return False
        if now - _last_any < MIN_GAP_S:
            return False
        _last_played[key] = now
        _last_any = now
    return True


def _spawn(argv) -> bool:
    try:
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, OSError):
        return False


def play(name: str, run: Optional[Callable] = None) -> bool:
    """Play one earcon, asynchronously. True when a player was launched.

    Never raises: this is called from the hotword listener thread and from
    bus subscribers, where an exception would take the room down with it.
    """
    key = resolve(name)
    if not key or not enabled():
        return False
    # Resolved at CALL time, not bound as a default argument: a default is
    # evaluated at def time, so the test firewall (tests/conftest.py) could
    # not replace it and four tests spawned a REAL paplay into the user's
    # speakers -- twice, on 2026-08-31, while he was sitting at the desk.
    run = _spawn if run is None else run
    try:
        if not _allowed(key):
            return False
        path = render(key)
        if path is None:
            return False
        # paplay first (PipeWire mixes it over whatever else is playing),
        # aplay as the fallback -- the same chain recorder.play_beep used.
        return bool(run(["paplay", f"--volume={_volume()}", str(path)])
                    or run(["aplay", "-q", str(path)]))
    except Exception:            # noqa: BLE001 - a beep must never break a turn
        log.debug("earcons: %s failed to play", key, exc_info=True)
        return False


def play_async(name: str) -> None:
    """play() on its own thread: paplay's spawn costs a few ms and the
    callers are the hotword listener and the Tk bus pump."""
    threading.Thread(target=play, args=(name,), daemon=True,
                     name=f"earcon-{name}").start()


def reset_cooldowns() -> None:
    """Tests and the audition script; never called by the app."""
    global _last_any
    with _lock:
        _last_played.clear()
        _last_any = 0.0
