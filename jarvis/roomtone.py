"""Room tone: a near-subliminal bed that follows the arc.

Pre-dawn is almost nothing; the work hours carry a faint HVAC-and-servers
hum; dusk warms; night thins to a single low drone. You are meant to
notice it only when it stops.

MIC SAFETY IS THE DESIGN, NOT A RISK SECTION. Jarvis is armed for the wake
word continuously, there is no AEC on this box, and a Blue Snowball sits in
the same room as the speaker. A continuous bed therefore enters EVERY
capture and puts three thresholds that were tuned in a quiet room at risk:
the openWakeWord threshold (0.3, with SPEAKER_WAKE_MIN 0.25), the Silero
endpointer (whose own test asserts that room tone is not speech), and the
ECAPA speaker gate at 0.30 -- which has already caused total rejection
once. "Duck during TTS" does not cover any of that. So:

* **The feature is OFF by default** (``ambience.room_tone``) and turned on
  by voice, deliberately, once measured.
* **The mute is hard and immediate.** ``mute()`` kills the stream from the
  caller's own thread rather than setting a flag a tick will notice later.
  It is asserted for the whole of every capture, every wake word and every
  spoken reply, so no bed is ever in the audio Whisper or ECAPA sees.
* **The default volume is 0.05.** Pick the real number from the room's own
  noise floor as the Snowball sees it, and require no regression in
  wake-word FRR or the ECAPA distribution before raising it.

There is no pactl ducking here on purpose. We own the process, so muting it
outright is both stronger than lowering its volume and needs no external
state to restore -- and a duck that has to be undone is a duck that gets
left on when something throws. This is also why the module has no
dependency on a mixer: it ships standalone.

Silence is also a POLICY, not just a mic concern. The bed is quiet whenever
``quiet.reason()`` is non-empty -- quiet hours, DND, a calendar-detected
class or meeting, an empty house -- which additionally means an empty room
stops feeding the Bluetooth speaker and lets it sleep. A long stretch with
no microphone activity stops it too, whatever the phone says.

Loops are baked ONCE to disk under ``PATHS.MEMORY_DIR/roomtone`` (a fixed
RNG seed per phase: deterministic, so a loop cached last month is the loop
this code would make today) and streamed raw into a single dedicated
``paplay`` process. Nothing is ever synthesized on a tick, and certainly
not while Whisper is decoding.
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

log = get_logger("roomtone")

SAMPLE_RATE = 22050
LOOP_S = 12.0                  # long enough not to be heard as a repeat
CHUNK_S = 0.25                 # the write cadence, and so the mute latency
STREAM_NAME = "jarvis-room-tone"
TICK_S = 2.0
DEFAULT_VOLUME = 0.05
DEFAULT_AWAY_STOP_MIN = 20
# Longer than the recorder's 60 s hard cap on a capture: see _expire_holds.
MAX_HOLD_S = 90.0

# One bed per arc phase. (drone hz, drone gain, noise gain, tilt) where
# `tilt` shapes the noise: 1.0 is dark and airless, higher opens it up.
# Nothing here is loud; the loudest bed is a twentieth of full scale before
# the paplay volume is applied on top.
BEDS: dict[str, tuple[float, float, float, float]] = {
    # almost nothing: one very low drone and a breath of air
    "pre-dawn": (48.0, 0.16, 0.05, 0.7),
    # the room starting up
    "waking": (60.0, 0.20, 0.10, 1.3),
    # HVAC and servers, the Spark's own fan implied
    "working": (72.0, 0.26, 0.18, 1.8),
    "afternoon": (72.0, 0.22, 0.15, 1.6),
    # dusk warms: the drone drops a fifth and the air softens
    "dusk": (54.0, 0.26, 0.11, 1.0),
    "evening": (54.0, 0.22, 0.08, 0.9),
    # night thins to a single low drone
    "night": (40.0, 0.18, 0.03, 0.6),
}
PHASES = tuple(BEDS)


def _seed(phase: str) -> int:
    """A stable per-phase seed: the same loop today as last month."""
    return sum((i + 1) * ord(c) for i, c in enumerate(phase)) & 0xFFFF


def _loop_samples(phase: str):
    """One seamless loop as int16 PCM. Deterministic; numpy only."""
    import numpy as np

    drone_hz, drone_gain, noise_gain, tilt = BEDS[phase]
    n = int(SAMPLE_RATE * LOOP_S)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(_seed(phase))

    # The drone: the fundamental plus its fifth, both at frequencies rounded
    # to a whole number of cycles in the loop, so the ends meet exactly and
    # the seam is inaudible.
    def cycles(hz):
        return max(1.0, round(hz * LOOP_S)) / LOOP_S

    sig = np.sin(2 * np.pi * cycles(drone_hz) * t)
    sig += 0.4 * np.sin(2 * np.pi * cycles(drone_hz * 1.5) * t)
    sig *= drone_gain / 1.4

    # The air: white noise smoothed into a low rumble. Built in the
    # frequency domain and made periodic by construction (irfft of a
    # half-spectrum is exactly one period long), which is why there is no
    # crossfade anywhere in this function.
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    # 1/f^2 above a corner that `tilt` moves: dark beds have a low corner.
    corner = 60.0 * tilt
    spec *= 1.0 / (1.0 + (freqs / corner) ** 2)
    spec[0] = 0.0                                  # no DC offset
    air = np.fft.irfft(spec, n)
    peak = float(np.abs(air).max()) or 1.0
    sig += noise_gain * air / peak

    peak = float(np.abs(sig).max()) or 1.0
    # Normalise to the bed's own headroom, never to full scale: this is the
    # difference between ambience and a hum.
    sig = sig / peak * min(1.0, drone_gain + noise_gain)
    return np.clip(sig * 32767.0, -32767, 32767).astype("<i2")


def bake(phase: str, directory: Optional[Path] = None,
         force: bool = False) -> Optional[Path]:
    """Write one phase's loop to a WAV (None for an unknown phase)."""
    if phase not in BEDS:
        return None
    out_dir = Path(directory) if directory else (PATHS.MEMORY_DIR / "roomtone")
    path = out_dir / f"{phase}.wav"
    if path.exists() and not force:
        return path
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = _loop_samples(phase).tobytes()
        tmp = path.with_name(path.name + ".tmp")
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(frames)
        tmp.replace(path)
    except OSError:
        log.debug("roomtone: %s could not be baked", phase, exc_info=True)
        return None
    return path


def bake_all(directory: Optional[Path] = None, force: bool = False) -> dict:
    return {p: path for p in PHASES if (path := bake(p, directory, force)) is not None}


def desired_phase(*, enabled: bool, arc_phase: str = "", quiet_reason: str = "",
                  presence_state: str = "", muted: bool = False,
                  mic_idle_s: Optional[float] = None,
                  away_stop_s: float = DEFAULT_AWAY_STOP_MIN * 60.0) -> str:
    """Which bed should be playing right now ("" for silence).

    Pure, and the whole policy: every reason for silence is here and in one
    order, so there is no second place to look when the room is quiet and
    should not be (or the reverse).
    """
    if not enabled or muted:
        return ""
    if str(quiet_reason or "").strip():
        # quiet.py owns this: hours, DND, a meeting, an empty house. The
        # arc obeys it too, so this is belt and braces -- but the bed is the
        # one consumer where being wrong is audible all night.
        return ""
    if str(presence_state or "").strip().lower() == "away":
        return ""
    if mic_idle_s is not None and mic_idle_s >= away_stop_s:
        # Nobody has said anything for a long time and the phone may simply
        # be unconfigured; let the speaker sleep.
        return ""
    phase = str(arc_phase or "").strip()
    return phase if phase in BEDS else ""


class RoomTone:
    """Owns one paplay stream and the policy that decides what it plays."""

    def __init__(self, cfg, arc=None, quiet=None, presence=None, turns=None,
                 now: Callable[[], float] = time.time,
                 popen: Callable = subprocess.Popen, tick_s: float = TICK_S,
                 directory: Optional[Path] = None):
        self._cfg = cfg
        self._arc = arc
        self._quiet = quiet
        self._presence = presence
        self._turns = turns
        self._now = now
        self._popen = popen
        self.tick_s = float(tick_s)
        self._dir = Path(directory) if directory else None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = None
        self.playing = ""            # the phase currently streaming ("" = silent)
        self._muted: dict[str, float] = {}   # named holds -> when taken
        self._frames: dict[str, bytes] = {}

    # ------------------------------------------------------------ config
    def _get(self, key: str, default=None):
        get = getattr(self._cfg, "get", None)
        if not callable(get):
            return default
        try:
            value = get(key, default)
        except Exception:            # noqa: BLE001 - config boundary
            log.debug("roomtone: cfg.get(%s) failed", key, exc_info=True)
            return default
        return default if value is None else value

    @property
    def enabled(self) -> bool:
        """OFF by default -- see the module docstring. Read every tick, so
        "room tone off" takes effect within one tick with no restart."""
        return bool(self._get("ambience.room_tone", False))

    def _volume(self) -> int:
        try:
            vol = float(self._get("ambience.volume", DEFAULT_VOLUME))
        except (TypeError, ValueError):
            vol = DEFAULT_VOLUME
        return int(max(0.0, min(1.0, vol)) * 65536)

    def _away_stop_s(self) -> float:
        try:
            minutes = float(self._get("ambience.away_stop_min", DEFAULT_AWAY_STOP_MIN))
        except (TypeError, ValueError):
            minutes = DEFAULT_AWAY_STOP_MIN
        return max(60.0, minutes * 60.0)

    # ------------------------------------------------------------ inputs
    @staticmethod
    def _deref(source):
        """The input, or what a late-binding callable resolves to -- app.py
        builds this before the turn ledger exists. See Arc._deref."""
        try:
            return source() if callable(source) else source
        except Exception:        # noqa: BLE001 - a late binding may not be ready
            return None

    def _quiet_reason(self) -> str:
        fn = getattr(self._deref(self._quiet), "reason", None)
        if not callable(fn):
            return ""
        try:
            return str(fn() or "")
        except Exception:            # noqa: BLE001 - policy boundary
            return ""

    def want(self) -> str:
        idle = None
        try:
            turns = self._deref(self._turns)
            idle = turns.idle_s() if turns is not None else None
        except Exception:            # noqa: BLE001
            idle = None
        self._expire_holds()
        with self._lock:
            muted = bool(self._muted)
        return desired_phase(
            enabled=self.enabled,
            arc_phase=str(getattr(self._deref(self._arc), "phase", "") or ""),
            quiet_reason=self._quiet_reason(),
            presence_state=str(getattr(self._deref(self._presence), "state", "") or ""),
            muted=muted, mic_idle_s=idle, away_stop_s=self._away_stop_s())

    # ------------------------------------------------------------- audio
    def _loop_bytes(self, phase: str) -> bytes:
        """The raw frames for one phase, read from the baked WAV once."""
        cached = self._frames.get(phase)
        if cached is not None:
            return cached
        path = bake(phase, self._dir)
        if path is None:
            return b""
        try:
            with wave.open(str(path)) as wf:
                frames = wf.readframes(wf.getnframes())
        except (OSError, wave.Error):
            log.debug("roomtone: %s unreadable", phase, exc_info=True)
            return b""
        self._frames[phase] = frames
        return frames

    def _start_stream(self, phase: str) -> bool:
        argv = ["paplay", "--raw", f"--rate={SAMPLE_RATE}", "--format=s16le",
                "--channels=1", f"--volume={self._volume()}",
                f"--stream-name={STREAM_NAME}"]
        try:
            proc = self._popen(argv, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, OSError):
            log.debug("roomtone: paplay unavailable", exc_info=True)
            return False
        with self._lock:
            self._proc, self.playing = proc, phase
        log.info("roomtone: %s bed up", phase)
        return True

    def _stop_stream(self, why: str = "") -> None:
        with self._lock:
            proc, phase = self._proc, self.playing
            self._proc, self.playing = None, ""
        if proc is None:
            return
        for step in ("terminate", "kill"):
            try:
                getattr(proc, step)()
                break
            except Exception:        # noqa: BLE001 - already gone, or no such method
                continue
        try:
            stdin = getattr(proc, "stdin", None)
            if stdin is not None:
                stdin.close()
        except Exception:            # noqa: BLE001 - a closed pipe is the goal
            pass
        log.info("roomtone: %s bed down%s", phase or "(none)", f" ({why})" if why else "")

    # -------------------------------------------------------------- mute
    def mute(self, owner: str = "mic") -> None:
        """Hard, immediate, from the caller's thread. The recorder, the wake
        word and the TTS all assert this; nothing may be in the bed while
        the microphone is open.

        Holds are NAMED, not counted. SpeakingState streams at ~12 Hz while
        Jarvis talks, so a depth counter would take twelve increments and
        one decrement per reply and latch the bed off for good.
        """
        with self._lock:
            first = not self._muted
            self._muted[owner] = self._now()
        if first:
            self._stop_stream("muted")

    def unmute(self, owner: str = "mic") -> None:
        with self._lock:
            self._muted.pop(owner, None)
        # The tick brings the bed back; nothing is started from here, so an
        # unmute racing a capture cannot re-open audio into it.

    def _expire_holds(self) -> None:
        """Drop a hold nothing ever released.

        RecordingStarted without its RecordingStopped (an aborted capture, a
        crashed consumer) would otherwise mute the room until the next
        restart -- silently, which is the worst way for ambience to fail.
        The cap is longer than the recorder's own 60 s hard limit.
        """
        now = self._now()
        with self._lock:
            stale = [k for k, at in self._muted.items() if now - at > MAX_HOLD_S]
            for key in stale:
                self._muted.pop(key, None)
        for key in stale:
            log.warning("roomtone: %r hold expired after %.0fs; releasing", key, MAX_HOLD_S)

    @property
    def muted(self) -> bool:
        with self._lock:
            return bool(self._muted)

    def settle(self) -> None:
        """The departure cue: the bed drops to its floor, silently."""
        self._stop_stream("he left")

    # ------------------------------------------------------------- ticks
    def tick(self) -> str:
        """Bring the stream in line with the policy; returns what plays."""
        want = self.want()
        with self._lock:
            proc, playing = self._proc, self.playing
        dead = proc is not None and getattr(proc, "poll", lambda: None)() is not None
        if want != playing or dead:
            if playing or dead:
                self._stop_stream("phase change" if want else "policy")
            if want:
                self._start_stream(want)
        return self.playing

    def _pump(self) -> bool:
        """Write one chunk. False when there is nothing streaming."""
        with self._lock:
            proc, phase = self._proc, self.playing
        if proc is None or not phase:
            return False
        frames = self._loop_bytes(phase)
        if not frames:
            self._stop_stream("no loop")
            return False
        step = int(SAMPLE_RATE * CHUNK_S) * 2
        stdin = getattr(proc, "stdin", None)
        if stdin is None:
            return False
        for i in range(0, len(frames), step):
            with self._lock:
                if self._proc is not proc:
                    return False     # muted or switched underneath us
            try:
                stdin.write(frames[i:i + step])
                stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                # mute() closed it, or PulseAudio went away. Either is
                # normal; the next tick decides whether to come back.
                self._stop_stream("stream closed")
                return False
        return True

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="roomtone")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._stop_stream("shutting down")
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
                if self._pump():
                    continue         # writing paces itself against paplay
            except Exception:
                log.exception("roomtone tick failed")
                self._stop_stream("tick failed")
            if self._stop.wait(self.tick_s):
                return
        self._stop_stream("stopped")

    # -------------------------------------------------------- bus hooks
    def on_arc(self, ev) -> None:
        """ArcChanged: the bed follows the hour. The tick does the work, so
        this only wakes it early."""
        log.debug("roomtone: arc -> %s", getattr(ev, "phase", ""))

    def on_wake(self, _ev=None) -> None:
        """HotwordDetected: down BEFORE the mic opens, not with it."""
        self.mute("wake")

    def on_mic_open(self, _ev=None) -> None:
        self.mute("mic")

    def on_mic_close(self, _ev=None) -> None:
        # Both, because a wake word that never became a capture (ignored
        # while transcribing, a rejected speaker) leaves its own hold, and
        # waiting MAX_HOLD_S for that is a minute and a half of dead room.
        self.unmute("mic")
        self.unmute("wake")

    def on_speaking(self, ev) -> None:
        # Its own hold name: a reply that starts mid-capture must not
        # release the capture's hold when it ends.
        if getattr(ev, "amplitude_only", False):
            return          # a mouth-shape tick, not a claim about speech
        if getattr(ev, "active", False):
            self.mute("tts")
        else:
            self.unmute("tts")
