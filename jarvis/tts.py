"""Jarvis TTS — dual-engine text-to-speech (V3).

Evolves jarvis_tts.py: same engine internals (Edge TTS en-GB-RyanNeural fast
path, XTTS v2 voice clone quality path) and the paplay→pw-play→aplay playback
chain, plus the V3 reliability fixes:

- internal FIFO queue drained by a dedicated worker thread — ``speak()``
  enqueues and utterances are never silently dropped while another is playing
- 15s ``asyncio.wait_for`` timeout on edge synthesis
- temp wavs removed in a ``finally`` block
- amplitude envelope published as ``SpeakingState`` events (~12Hz) on the
  event bus, replacing the GUI-polled ``_current_amp``
- ``stop()`` halts current speech and clears the pending queue
- XTTS sentence pipelining: the utterance is split into sentence chunks and
  chunk N+1 is synthesized on a producer thread while chunk N plays, cutting
  time-to-first-audio from ~full-utterance-synthesis to ~one-sentence

Voice & I/O upgrades (audit C):

- ``interrupt()`` — barge-in: cut the current utterance and drop the queue,
  reporting whether anything was actually cut off (the app calls it when
  the user types while Jarvis is talking; "Jarvis, quiet" does the same)
- speech cache (``jarvis.speech_cache``): rendered audio for a phrase is
  kept on disk keyed by engine + voice parameters + spoken text, so repeats
  ("Always, sir.", reminders, the quiet acknowledgement) play instantly and,
  for XTTS, with one fixed rendition; ``prewarm()`` renders a list of
  phrases in the background while nothing is being said
- pronunciation dictionary (``jarvis.pronounce``) applied right before
  synthesis, so "VSS on the GB10" is said "V S S on the G B ten" while the
  transcript keeps the spelling
- ``last_text`` / ``repeat_last()`` for "say again"

Note: Edge TTS writes MP3 data even though the temp file is named .wav;
paplay/pw-play decode it via libsndfile (the aplay fallback would not).

Usage:
    from jarvis.tts import TTS
    tts = TTS()
    tts.speak("Hello sir. All systems operational.")
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import re
import subprocess
import json
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

from jarvis import mixer, pronounce
from jarvis.config import CONFIG, PATHS
from jarvis.events import SpeakingState, Status, bus
from jarvis.logs import get_logger
from jarvis.speech_cache import SpeechCache

log = get_logger("tts")

VOICE_REF = PATHS.VOICE_REF          # ~/.aiws_trainer/jarvis_voice_ref.wav

_ENGINES = ("edge", "xtts", "f5", "fish")

# Which engines need the text massaged before they can say it properly.
# The time and shouted-word rewrites in jarvis/pronounce.py exist because
# XTTS said "six zero pm" for "6:00 pm" and read BIOSENSORS as an acronym.
# Fish s2.1-pro normalises both itself, and the rewrites HURT there: "ay em"
# is voiced as "I'm" (heard 2026-08-28). Verified per engine by listening,
# never assumed.
_ENGINE_NEEDS_TIME_REWRITE = {"edge": True, "xtts": True, "f5": True,
                              "fish": False}
_ENGINE_NEEDS_UNSHOUT = {"edge": True, "xtts": True, "f5": True,
                         "fish": True}

# Voice parameters (unchanged from the V1 engine); they are part of the
# speech-cache key so a tuning change never replays stale audio.
EDGE_VOICE = "en-GB-RyanNeural"
EDGE_RATE = "+5%"
EDGE_PITCH = "-4Hz"
# Settled by listening on 2026-08-28: 1.16 was clipped and robotic, 1.00
# dragged. repetition_penalty 2.0 is XTTS's own default and beat the 5.0 this
# shipped with. These are part of the speech-cache key, so changing one can
# never replay stale audio.
XTTS_PARAMS = dict(speed=1.05, temperature=0.65, top_p=0.85,
                   repetition_penalty=2.0)

# XTTS renders this voice at about -18 LUFS, which is quiet for an assistant
# across a room. Gain is applied to the OUTPUT, where it changes level and
# nothing else -- the reference itself is left exactly as recorded, because
# normalising THAT lost a listening test twice.
#
# CONSTANT, not per-utterance normalisation: _speak_xtts_pipelined renders a
# reply chunk by chunk, and normalising each chunk to its own peak would make
# the volume pump audibly between chunks of a single sentence. A hot chunk is
# pulled back just far enough to stay inside full scale.
# ------------------------------------------------------------- Fish Audio
#
# Hosted s2.1-pro. Chosen after five local engines were rejected by ear on
# 2026-08-28; it is the model that produced the reference the user rated best,
# and it measured 186 ms time-to-first-audio from this box — faster than every
# local option, XTTS's 0.4 s included.
#
# The cost is a network dependency on the assistant's voice, so it is designed
# around rather than ignored: the speech cache is consulted BEFORE the network
# (a repeated line is free and works offline), and any failure falls back to a
# LOCAL engine. Jarvis going silent because the internet did would read as a
# broken assistant.
#
# The free backend is deliberately NOT the default: measured 2218 ms median,
# degrading to 3294 ms over five consecutive calls, and its WebSocket path
# returns 402. Paid is what makes this worth having.
FISH_KEY_FILE = Path.home() / ".config" / "jarvis" / "fish_key"
FISH_MODEL_FILE = Path.home() / ".config" / "jarvis" / "fish_model_id"
FISH_BACKEND = "s2.1-pro"
FISH_TIMEOUT_S = 10.0
FISH_FALLBACK = "f5"          # must stay LOCAL, or an outage is still silence

# Running out of credit is NOT the same failure as the network blinking, and
# must not be handled the same way. A blip should cost one chunk; an exhausted
# balance will fail EVERY chunk forever, so retrying the API before each
# fallback would add a doomed round-trip to every sentence Jarvis ever speaks.
# When we see a payment/quota error we switch to the local engine for good and
# persist it, so the next launch starts local instead of rediscovering this.
_FISH_CREDIT_PHRASES = ("payment required", "out of credit", "insufficient balance",
                        "insufficient credit", "quota exceeded", "no credit")


def _is_out_of_credit(exc: BaseException) -> bool:
    """True when a fish failure means the balance is gone, not the network.

    The status code is authoritative when there is one: fish_audio_sdk raises
    HttpCodeErr with ``.status`` (never ``.response``), and a 429 or a 5xx
    whose body happens to mention "balance" or "credit" must NOT retire the
    engine for good -- retire_fish persists the switch, so one such blip
    would have turned hosted speech off permanently. Only without a status at
    all does the text decide, and then only on explicit payment phrases.
    """
    status = getattr(exc, "status", None)
    if not isinstance(status, int) or isinstance(status, bool):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status == 402
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _FISH_CREDIT_PHRASES)


def _fish_creds():
    """(api_key, reference_id) from disk, or (None, None). Never from source."""
    try:
        key = FISH_KEY_FILE.read_text().strip()
        model = FISH_MODEL_FILE.read_text().strip()
        return (key or None), (model or None)
    except OSError:
        return None, None


def _fish_stream_model(text: str, out_path: str, timeout: float,
                       model: str) -> dict:
    """Same as _fish_stream but against an explicit voice — for A/B testing
    voices without touching the configured one."""
    from fish_audio_sdk import Session, TTSRequest
    key, _ = _fish_creds()
    session = Session(key)
    started = time.monotonic()
    first = None
    with open(out_path, "wb") as fh:
        for chunk in session.tts(
                TTSRequest(text=text, reference_id=model, format="wav",
                           latency="balanced"), backend=FISH_BACKEND):
            if first is None:
                first = time.monotonic() - started
            fh.write(chunk)
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"fish exceeded {timeout}s")
    return {"ok": True, "ttfa": first}


def _fish_iter(text: str, timeout: float) -> Iterator[bytes]:
    """Yield the Fish API's audio bytes for ``text`` as they arrive.

    The network seam for play-while-rendering: the pipelined fish path
    feeds these straight into the player instead of waiting for the whole
    chunk, so the API's measured 186 ms time-to-first-audio finally reaches
    the ear rather than being hidden behind a full-chunk render.
    ``_fish_stream`` (the file seam the tests patch) is a wrapper over it.
    """
    from fish_audio_sdk import Session, TTSRequest
    key, model = _fish_creds()
    if not key or not model:
        raise RuntimeError("fish credentials missing")
    session = Session(key)
    started = time.monotonic()
    # The SDK's httpx client is built with timeout=None, so a connection
    # that delivers no bytes used to block session.tts() forever and the
    # old between-chunks deadline never ran -- the TTS worker wedged with
    # the whole queue behind it. The SDK now iterates on a daemon feeder
    # thread; this generator waits on the queue with the REMAINING
    # deadline, so a silent socket raises instead of hanging.
    q: queue.Queue = queue.Queue(maxsize=64)
    _end = object()

    def _feed():
        try:
            for chunk in session.tts(
                    TTSRequest(text=text, reference_id=model, format="wav",
                               latency="balanced"),
                    backend=FISH_BACKEND):
                q.put(chunk)
                if time.monotonic() - started > timeout:
                    return              # the consumer is raising; stop feeding
            q.put(_end)
        except BaseException as exc:    # noqa: BLE001 - carried to the consumer
            q.put(exc)

    threading.Thread(target=_feed, daemon=True, name="fish-feed").start()
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"fish exceeded {timeout}s")
        try:
            item = q.get(timeout=max(0.05, remaining))
        except queue.Empty:
            raise TimeoutError(f"fish exceeded {timeout}s") from None
        if item is _end:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _fish_stream(text: str, out_path: str, timeout: float) -> dict:
    """Render one chunk via the Fish API into a file. Tests patch this."""
    started = time.monotonic()
    first = None
    with open(out_path, "wb") as fh:
        for chunk in _fish_iter(text, timeout):
            if first is None:
                first = time.monotonic() - started
            fh.write(chunk)
    return {"ok": True, "ttfa": first}


# Play a fish chunk while it is still arriving (paplay reads the wav from
# stdin). Module-level so a test -- or an ear -- can switch the old
# render-then-play path back on for comparison.
FISH_STREAM_PLAYBACK = True

# A RIFF/WAVE header alone is 44 bytes; a stream that died with no more than
# that never reached the speaker and may be re-rendered by the fallback engine.
_WAV_HEADER_BYTES = 44


class _AudioStream:
    """Audio for ONE chunk, arriving from the network while it plays.

    The producer thread appends bytes (``feed``) and tees them into ``path``
    so the speech cache still gets a complete file; the worker thread pulls
    them (``get``) into the player's stdin. Never blocks the producer: the
    buffer is unbounded (a sentence of wav is a few hundred KB), because a
    producer blocked on a full pipe while the previous chunk played would
    trip the fish timeout for no reason.
    """

    def __init__(self, path: str):
        self.path = path
        self.nbytes = 0
        self.failed = False
        self.done = threading.Event()
        self._q: queue.Queue = queue.Queue()

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self.nbytes += len(data)
        self._q.put(data)

    def close(self, failed: bool = False) -> None:
        self.failed = failed
        self.done.set()
        self._q.put(None)

    def get(self, timeout: float | None = None) -> bytes | None:
        """Next bytes, ``None`` at end of stream; ``queue.Empty`` on timeout."""
        return self._q.get(timeout=timeout)

    @property
    def heard(self) -> bool:
        """True once audio (not just a header) has been handed to the player."""
        return self.nbytes > _WAV_HEADER_BYTES


class _LiveEnvelope:
    """RMS envelope of a wav computed from its bytes as they stream past,
    so the reactor pulses for a chunk that has no complete file yet.
    Publishes through the same feeder as the file path (``_run_amp_feeder``);
    ``values()`` yields one amplitude per 80 ms window."""

    def __init__(self):
        self._head = b""
        self._pcm = b""
        self._fmt = None            # (channels, rate, bits) once parsed
        self._window = 0
        self._q: queue.Queue = queue.Queue()
        self._closed = False

    def _parse_header(self) -> bool:
        head = self._head
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return False
        pos = 12
        fmt = None
        while pos + 8 <= len(head):
            cid = head[pos:pos + 4]
            size = int.from_bytes(head[pos + 4:pos + 8], "little")
            if cid == b"fmt " and pos + 8 + 16 <= len(head):
                body = head[pos + 8:pos + 8 + 16]
                channels = int.from_bytes(body[2:4], "little")
                rate = int.from_bytes(body[4:8], "little")
                bits = int.from_bytes(body[14:16], "little")
                fmt = (max(1, channels), max(1, rate), bits)
            elif cid == b"data":
                if fmt is None:
                    return False
                self._fmt = fmt
                self._pcm = head[pos + 8:]
                self._head = b""
                return True
            pos += 8 + size + (size & 1)
        return False

    def push(self, data: bytes) -> None:
        if self._closed or not data:
            return
        if self._fmt is None:
            self._head += data
            if not self._parse_header():
                if len(self._head) > 4096:
                    self._closed = True      # not a wav we understand
                return
        else:
            self._pcm += data
        channels, rate, bits = self._fmt
        if bits != 16:
            return
        frame = 2 * channels
        step = int(rate * 0.08) * frame
        while len(self._pcm) >= step:
            block, self._pcm = self._pcm[:step], self._pcm[step:]
            self._q.put(_rms_int16(block, channels))

    def close(self) -> None:
        self._closed = True
        self._q.put(None)

    def values(self) -> Iterator[float]:
        while True:
            amp = self._q.get()
            if amp is None:
                return
            yield amp


def _rms_int16(block: bytes, channels: int) -> float:
    """Same scaling as the file envelope: soundfile floats in [-1, 1],
    RMS * 4 capped at 1."""
    import numpy as np
    samples = np.frombuffer(block[:len(block) - len(block) % 2], dtype="<i2")
    if channels > 1:
        samples = samples[::channels]
    if not samples.size:
        return 0.0
    rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
    return min(1.0, rms * 4)


# ---------------------------------------------------------------- F5-TTS
#
# F5 runs in a SIDECAR, not in this process. jarvis and VSS share ~/vss_env,
# and installing f5-tts there removes fastapi (VSS's api.py needs it), so F5
# lives in its own venv and we talk to a resident server over a unix socket.
# Loading the model costs seconds, so it stays up for the life of the app.
F5_REF = PATHS.VOICE_REF_F5
F5_REF_TEXT = PATHS.VOICE_REF_F5_TEXT
F5_PYTHON = PATHS.F5_PYTHON
F5_SOCK = PATHS.F5_SOCK
F5_SERVER = PATHS.REPO_ROOT / "scripts" / "f5_server.py"

# Settled by BLIND listening, rounds 1-5, 2026-08-28/29 (see
# ~/voice-training/NOTES.md). This configuration, with the duration floor in
# scripts/f5_server.py and reference clip 0341, scored 5.00 against the real
# hosted Fish control at 5.00 -- straight 5s, indistinguishable in a shuffled
# set.
#
# nfe_step 10, NOT 8. 8 is outside F5's pruned EPSS timestep table, which
# defines schedules only for n in {5, 6, 7, 10, 12, 16}; quality saturates at
# 10 and 12 grazes the latency gate. speed 0.85, not 0.70: 0.70 was chosen by
# ear before the duration floor existed, when slowing the whole utterance was
# the only lever available for short lines. The floor fixes that at its
# source, so the global rate no longer has to compensate -- and 0.85 measured
# as the closest match to the target speaker's articulation.
#
# Both are in the cache key, so changing them invalidates cached audio.
F5_PARAMS = dict(nfe_step=10, speed=0.85)

# The sidecar is meant to be RESIDENT, engine or not: F5 is the local
# fallback for the hosted voice, and a fallback that has to cold-start
# (~180 s on this box) is not a fallback. scripts/systemd/jarvis-f5.service
# (installed by scripts/setup_f5_service.sh) keeps it up across app restarts
# and reboots. When that unit is active it OWNS the socket -- f5_server.py
# unlinks and rebinds the path on start, so a second, Jarvis-spawned copy
# would fight the unit's restarts for it and double the GPU footprint.
F5_UNIT = "jarvis-f5.service"

_f5_proc = None
_f5_lock = threading.Lock()


def _f5_unit_active() -> bool:
    """True when systemd --user reports the sidecar unit up (or coming up).

    "activating" counts: with Type=simple the unit is "active" the moment
    the python process exists, long before the model is resident, so this
    is only ever a hint that someone else owns the socket -- readiness is
    still _f5_alive(). Any failure to ask (no systemctl, no user bus, a
    timeout) is "not active": we would rather risk a duplicate sidecar than
    never start one.
    """
    try:
        res = subprocess.run(["systemctl", "--user", "is-active", F5_UNIT],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return res.stdout.strip() in ("active", "activating", "reloading")


def _f5_request(payload: dict, timeout: float = 120) -> dict:
    """One request/response over the sidecar socket."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(F5_SOCK))
        sock.sendall(json.dumps(payload).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            part = sock.recv(65536)
            if not part:
                break
            buf += part
    return json.loads(buf.decode("utf-8") or "{}")


def _f5_alive() -> bool:
    try:
        return bool(_f5_request({"ping": True}, timeout=5).get("ready"))
    except Exception:
        return False


def _ensure_f5_server(startup_timeout: float = 180) -> bool:
    """Adopt a running sidecar, or start one. True when it answers a ping.

    Three cases, in order: the socket already answers (a systemd-owned or
    earlier Jarvis-spawned server -- adopted as-is); the systemd unit is
    active but not yet answering (it is loading the model -- WAIT for it,
    never spawn beside it); nothing owns the socket (spawn our own).
    """
    global _f5_proc
    with _f5_lock:
        if _f5_alive():
            return True
        if _f5_unit_active():
            return _wait_for_f5_unit(startup_timeout)
        for path in (F5_PYTHON, F5_SERVER, F5_REF, F5_REF_TEXT):
            if not Path(path).exists():
                log.error("f5 sidecar cannot start, missing: %s", path)
                return False
        F5_SOCK.parent.mkdir(parents=True, exist_ok=True)
        log.info("starting f5 sidecar (%s is not active)", F5_UNIT)
        try:
            _f5_proc = subprocess.Popen(
                [str(F5_PYTHON), str(F5_SERVER), "--socket", str(F5_SOCK),
                 "--ref", str(F5_REF), "--ref-text", str(F5_REF_TEXT),
                 "--nfe", str(F5_PARAMS["nfe_step"]),
                 "--speed", str(F5_PARAMS["speed"])],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            log.exception("f5 sidecar failed to spawn")
            return False
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if _f5_proc.poll() is not None:
                log.error("f5 sidecar exited during startup (rc=%s)",
                          _f5_proc.returncode)
                return False
            if _f5_alive():
                log.info("f5 sidecar ready")
                return True
            time.sleep(0.5)
        log.error("f5 sidecar did not become ready in %.0fs", startup_timeout)
        return False


def _wait_for_f5_unit(startup_timeout: float) -> bool:
    """The systemd unit owns the socket: poll until it answers or dies.

    Re-checks the unit every few seconds so a unit that crashed (or that
    the user stopped) does not cost the whole startup budget -- and so we
    do not then spawn our own copy underneath a unit that systemd is about
    to restart (Restart=always).
    """
    log.info("f5 sidecar is owned by %s; waiting for it", F5_UNIT)
    deadline = time.monotonic() + startup_timeout
    next_unit_check = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _f5_alive():
            log.info("f5 sidecar ready (%s)", F5_UNIT)
            return True
        if time.monotonic() >= next_unit_check:
            if not _f5_unit_active():
                log.error("%s stopped while we were waiting for it", F5_UNIT)
                return False
            next_unit_check = time.monotonic() + 5
        time.sleep(0.5)
    log.error("%s did not answer in %.0fs", F5_UNIT, startup_timeout)
    return False


def retire_own_f5_sidecar() -> bool:
    """Stop a Jarvis-spawned sidecar that the systemd unit has superseded.

    Only ours (``_f5_proc``), and only when the unit is active: once the
    unit is up it has taken the socket path, so our copy answers nobody and
    just holds a few GB of the unified pool. Without the unit our copy is
    deliberately left running -- it is what makes the next launch warm.
    Called at app shutdown; never touches a systemd-owned process.
    """
    global _f5_proc
    # Bounded acquire: the warm thread holds _f5_lock across a cold start
    # (up to 180 s) and quit() calls this synchronously on the UI thread.
    # Skipping retirement on contention is safe -- leaving our sidecar
    # resident is the documented default, and mid-cold-start there is
    # nothing to retire yet.
    if not _f5_lock.acquire(timeout=2):
        log.info("f5 retire skipped: the warm-up holds the lock")
        return False
    try:
        proc = _f5_proc
        if proc is None or proc.poll() is not None:
            return False
        if not _f5_unit_active():
            return False
        log.info("stopping our f5 sidecar (pid %s): %s owns the socket now",
                 proc.pid, F5_UNIT)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            log.exception("could not stop our f5 sidecar")
            return False
        _f5_proc = None
        return True
    finally:
        _f5_lock.release()


OUTPUT_GAIN = 1.34               # ~+2.5 dB; peak 0.72 -> ~0.97
_CEILING = 0.99


def apply_output_gain(wav):
    """Raise the rendered level without touching timbre or dynamics."""
    import numpy as np
    out = np.asarray(wav, dtype=np.float32) * OUTPUT_GAIN
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > _CEILING:          # rare outlier: scale it just under the rail
        out *= _CEILING / peak
    return out


class TTS:
    """Dual-engine TTS: Edge TTS (fast) or XTTS v2 (quality).

    ``speak()`` enqueues; a daemon worker thread serializes synthesis and
    playback. SpeakingState(active=True/False) marks utterance start/end
    (only at burst boundaries — back-to-back queued utterances do not flap
    active off/on between items); SpeakingState(active=True, amplitude=…)
    streams the envelope at ~12Hz while audio plays.
    """

    MAX_SPEAK_LENGTH = 500

    def __init__(self, gpu: int = 0, engine: str = "edge",
                 cache: bool = True, cache_dir: Path | str | None = None,
                 pronunciation: bool = True, arbiter=None):
        self._xtts = None
        self._gpu = gpu
        self._engine = engine if engine in _ENGINES else "edge"
        self._stop_flag = False
        self._speaking = False           # burst state (queue non-empty → done)
        self._burst_announced = False    # SpeakingState(active=True) sent yet?
        self._amp_playing = False
        self._amp_gen = 0                # generation token: one per chunk feeder
        self._current_amp = 0.0
        self._player: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()   # (text, done_event)
        self._arbiter = arbiter          # MicArbiter; None when standalone
        self._mic_hold = None            # live acquire() for this burst
        self.cache: Optional[SpeechCache] = (
            SpeechCache(cache_dir) if cache else None)
        self._pronunciation = pronunciation
        self._synth_lock = threading.Lock()    # one XTTS inference at a time
        self._load_lock = threading.Lock()     # one XTTS LOAD at a time
        self._prewarm_thread: threading.Thread | None = None
        self._f5_warm_thread: threading.Thread | None = None
        self.last_text = ""                    # last cleaned utterance queued
        self.interrupts = 0                    # barge-ins that cut speech
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="tts-worker")
        self._worker.start()

    # ------------------------------------------------------------ player
    # The live player, behind a property so the Room Mixer (jarvis/mixer.py)
    # learns its PID wherever in this file it is set. The Mixer exempts
    # Jarvis's own streams by PID and NEVER by name: paplay and the pacat
    # librespot pipes music through both report application.name/binary
    # "pacat", so a name-based exemption would duck his voice under his
    # voice -- or fail to duck the music at all.
    @property
    def _play_proc(self) -> "subprocess.Popen | None":
        return self._player

    @_play_proc.setter
    def _play_proc(self, proc: "subprocess.Popen | None") -> None:
        prev = getattr(self, "_player", None)
        if prev is not None and prev is not proc:
            mixer.forget_own_pid(getattr(prev, "pid", None))
        self._player = proc
        if proc is not None:
            # getattr, not proc.pid: the test doubles for the player are
            # bare objects, and a missing pid must not break playback.
            mixer.register_own_pid(getattr(proc, "pid", None))

    # ------------------------------------------------------------ engine
    @property
    def engine(self) -> str:
        return self._engine

    @engine.setter
    def engine(self, value: str):
        if value not in _ENGINES:
            raise ValueError(f"engine must be one of {_ENGINES}, got {value!r}")
        self._engine = value

    # ------------------------------------------------------------ public
    def load(self) -> bool:
        """Load the TTS engine. Edge needs no preload; XTTS loads the model.

        Serialised: the startup warmer (app._load_models) and the speak path
        (_speak_sync) both call this, and an unguarded ``self._xtts is None``
        check let both run a full 1.8 GB load concurrently -- the user then
        waited out the SECOND one (11.07 s of silence after the reply was
        ready, 2026-08-28). The check is repeated under the lock so a caller
        that waited returns as soon as the winner has finished priming.
        """
        if self._engine == "edge":
            return True
        if self._engine == "fish":
            key, model = _fish_creds()
            if key and model:
                # Hosted voice: nothing to load, but its LOCAL fallback must
                # be warm NOW, not at the first outage (see warm_f5_fallback).
                self.warm_f5_fallback()
                return True
            log.warning("fish credentials missing (%s) — falling back to %s",
                        FISH_KEY_FILE, FISH_FALLBACK)
            self._engine = FISH_FALLBACK
            return self.load()
        if self._engine == "f5":
            if _ensure_f5_server():
                return True
            return self._f5_unavailable()
        if self._xtts is not None:
            return True
        with self._load_lock:
            return self._load_xtts_locked()

    def warm_f5_fallback(self) -> Optional[threading.Thread]:
        """Bring the F5 sidecar up in the background while another engine
        is in use.

        The hosted voice's fallback is F5, and _ensure_f5_server used to be
        called for the first time by the first FAILING fish chunk -- which
        then paid the sidecar's ~180 s cold start, i.e. an outage was still
        a silent assistant. The unit (jarvis-f5.service) is the real answer;
        this covers a box where it is not installed, and adopts the unit's
        server when it is. Daemon thread, once per instance, never blocks
        load(): a warm-up is an optimisation, not a reason to wait.
        """
        if FISH_FALLBACK != "f5":
            return None
        t = self._f5_warm_thread
        if t is not None and t.is_alive():
            return t

        def _run():
            if _ensure_f5_server():
                log.info("f5 fallback is warm")
                return
            # Loud on purpose: the hosted voice still works, so nothing
            # else would tell the user their outage plan is cold.
            log.warning("f5 fallback could NOT be warmed -- a fish outage "
                        "would cost a cold start or drop to another engine")
            bus.publish(Status(text="Local voice fallback (F5) is not running",
                               kind="warn"))

        t = threading.Thread(target=_run, daemon=True, name="tts-f5-warm")
        self._f5_warm_thread = t
        t.start()
        return t

    def _f5_unavailable(self) -> bool:
        """The configured local voice will not come up. Degrade LOUDLY and
        stay local where possible.

        This used to drop straight to edge -- Microsoft's cloud -- with one
        log line, which on a box whose whole point is a local voice is the
        wrong default twice over: it sends every utterance off the machine
        and nobody notices until the accent changes. XTTS is local and
        needs only its reference clip, so it is tried first; edge is the
        last resort and is announced as the cloud engine it is. Returns
        load()'s verdict for the engine actually chosen, so an utterance is
        not discarded when the replacement can speak (fish's contract).
        """
        if VOICE_REF.exists():
            log.error("f5 sidecar unavailable — falling back to xtts (local)")
            bus.publish(Status(text="F5 voice down — using XTTS (local)",
                               kind="warn"))
            self._engine = "xtts"
        else:
            log.error("f5 sidecar unavailable and no XTTS reference at %s — "
                      "falling back to edge (CLOUD)", VOICE_REF)
            bus.publish(Status(text="F5 voice down — using edge (cloud voice)",
                               kind="warn"))
            self._engine = "edge"
        return self.load()

    def release_f5_sidecar(self) -> bool:
        """App-shutdown hook: see retire_own_f5_sidecar()."""
        try:
            return retire_own_f5_sidecar()
        except Exception:
            log.exception("f5 sidecar release failed")
            return False

    def _load_xtts_locked(self) -> bool:
        if self._engine == "edge":       # a failed load may have fallen back
            return True
        if self._xtts is not None:       # another thread got there first
            return True
        if not VOICE_REF.exists():
            log.warning("voice reference not found: %s", VOICE_REF)
            return False
        try:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            # coqui-tts 0.27 expects a helper transformers 5.x removed.
            import torch
            # Leave cores for the UI during the load burst (inference is GPU).
            try:
                torch.set_num_threads(max(4, (os.cpu_count() or 8) - 4))
            except Exception:
                pass
            import transformers.pytorch_utils as _tpu
            if not hasattr(_tpu, "isin_mps_friendly"):
                _tpu.isin_mps_friendly = (
                    lambda elements, test_elements: torch.isin(elements, test_elements))
            from TTS.api import TTS as _CoquiTTS
            xtts = _CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._xtts = xtts.to(f"cuda:{self._gpu}")
            try:
                # leave no async work outstanding: the driver's
                # cuda-EvtHandlr thread polls while any is pending
                torch.cuda.synchronize()
            except Exception:
                pass
            log.info("XTTS v2 loaded on CUDA:%d", self._gpu)
            self._prime_voice()
            return True
        except Exception:
            # Loud, like _f5_unavailable: edge sends every utterance off the
            # machine, and a log line is the one place nobody looks. load()'s
            # edge branch returns before touching _load_lock, so the
            # recursion cannot deadlock.
            log.exception("XTTS load error — falling back to edge engine")
            bus.publish(Status(text="XTTS load failed — using edge (cloud voice)",
                               kind="warn"))
            self._engine = "edge"
            return self.load()

    def _prime_voice(self):
        """Compute the speaker conditioning latents once, at load.

        XTTS caches them internally, so only the FIRST synthesis pays: measured
        on this box, first call 2.21 s against 0.97 s warm, with
        get_conditioning_latents alone costing 1.37 s. prewarm() used to prime
        them as a side effect, but it skips phrases already in the speech cache
        -- and with a warm cache it renders nothing at all, so the first real
        utterance of every session paid the cold path. Doing it here is
        explicit and does not depend on the cache being empty.

        Best effort: a prime is an optimisation, never a reason not to speak.
        """
        model = getattr(getattr(self._xtts, "synthesizer", None),
                        "tts_model", None)
        if model is None or not hasattr(model, "get_conditioning_latents"):
            return
        t0 = time.monotonic()
        try:
            model.get_conditioning_latents(audio_path=[str(VOICE_REF)])
            # Latents alone are NOT the cold cost: measured on this box, priming
            # them took 1.59 s and left the first utterance at 2.09 s against
            # 2.21 s unprimed. The rest is CUDA kernel / autoregressive warm-up
            # in the GPT and vocoder, which only a real synthesis touches.
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                self._synth_xtts("Ready.", tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            log.info("XTTS voice primed in %.2fs", time.monotonic() - t0)
        except Exception:
            log.exception("voice prime failed (first reply will be slower)")

    def speak(self, text: str, block: bool = False
              ) -> Optional[threading.Event]:
        """Enqueue text for speech; the worker drains FIFO (no silent drops).

        Args:
            text: text to speak (cleaned/truncated before synthesis)
            block: if True, wait until this utterance finishes (or is stopped)

        Returns the utterance's done event (set when it has played, been
        skipped or been dropped) so a caller queuing many lines -- the
        reader -- can tell which one is being spoken; None when nothing
        was queued.
        """
        if not text or not text.strip():
            return None
        text = self._clean_for_speech(text)
        if not text:
            return None
        self.last_text = text
        done = threading.Event()
        self._q.put((text, done))
        if block:
            done.wait()
        return done

    def stop(self):
        """Stop current speech and clear the pending queue."""
        self._stop_flag = True
        self._clear_queue()
        self._terminate_playback()

    def skip_current(self) -> bool:
        """Cut the utterance being spoken; everything queued behind it still
        plays. This is what "skip" means mid read-aloud: one chunk, not the
        document. ``_stop_flag`` is already per-utterance (``_speak_sync``
        resets it at the top of each item), so a skip is a stop without the
        queue clear. False when nothing is being spoken."""
        if not self._speaking:
            return False
        self._stop_flag = True
        self._terminate_playback()
        return True

    def _terminate_playback(self):
        proc = self._play_proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                log.exception("failed to terminate playback process")

    def interrupt(self) -> bool:
        """Barge-in. Stops playback and drops queued utterances like
        ``stop()``; returns True when something was actually cut off (so
        the caller can decide whether an acknowledgement is warranted)."""
        was_talking = self._speaking or not self._q.empty()
        self.stop()
        if was_talking:
            self.interrupts += 1
            log.info("speech interrupted (barge-in #%d)", self.interrupts)
        return was_talking

    def repeat_last(self) -> bool:
        """Say the last queued utterance again ("say again"). False when
        nothing has been said yet."""
        if not self.last_text:
            return False
        self.speak(self.last_text)
        return True

    @property
    def busy(self) -> bool:
        """Speaking now OR lines still queued -- the was_talking predicate
        interrupt() uses. For callers that need "is Jarvis mid-burst":
        SpeakingState now marks first AUDIO (the honest ledger mark), so
        during the render window the event says idle while this says busy."""
        return bool(self._speaking or not self._q.empty())

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def pending(self) -> int:
        """Utterances queued behind the current one."""
        return self._q.qsize()

    @property
    def current_amplitude(self) -> float:
        """Last published speech amplitude (0-1). Prefer SpeakingState events."""
        return self._current_amp

    # ----------------------------------------------------------- prewarm
    def prewarm(self, phrases: Iterable[str], block: bool = False
                ) -> Optional[threading.Thread]:
        """Render ``phrases`` into the speech cache in the background so
        their first real use plays instantly. Runs one phrase at a time,
        yields whenever real speech is queued, skips phrases already
        cached, and is a no-op without a cache or a loadable engine."""
        if self.cache is None:
            return None
        todo = [self._clean_for_speech(p) for p in phrases if p and p.strip()]
        todo = [t for t in todo if t]
        if not todo:
            return None
        if self._prewarm_thread is not None and self._prewarm_thread.is_alive():
            log.info("prewarm already running; ignoring %d phrase(s)", len(todo))
            return self._prewarm_thread

        def _run():
            if not self.load():
                return
            engine = self._engine
            done = 0
            for text in todo:
                spoken = self._pronounce(text)
                items = (self._split_sentences(spoken) if engine == "xtts"
                         else [spoken])
                for item in items:
                    key = self._cache_key(engine, item)
                    if self.cache.get(key) is not None:
                        continue
                    # Yield to real speech: never compete for the engine.
                    while self._speaking or not self._q.empty():
                        time.sleep(0.25)
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.close()
                    try:
                        if engine == "fish":
                            self._synth_fish(item, tmp.name)
                        elif engine == "f5":
                            self._synth_f5(item, tmp.name)
                        elif engine == "xtts" and self._xtts is not None:
                            self._synth_xtts(item, tmp.name)
                        else:
                            self._synth_edge(item, tmp.name)
                        if self.cache.put(key, tmp.name) is not None:
                            done += 1
                    except Exception:
                        log.warning("prewarm synth failed: %.60s", item,
                                    exc_info=True)
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass
            log.info("prewarm: %d phrase chunk(s) rendered (%s)", done, engine)

        self._prewarm_thread = threading.Thread(
            target=_run, daemon=True, name="tts-prewarm")
        self._prewarm_thread.start()
        if block:
            self._prewarm_thread.join()
        return self._prewarm_thread

    # ------------------------------------------------------------ worker
    def _clear_queue(self):
        while True:
            try:
                _, done = self._q.get_nowait()
            except queue.Empty:
                return
            done.set()

    def _acquire_mic(self):
        """Pause the hotword for the duration of a spoken burst.

        hotword.py's stated contract is that every mic consumer -- including
        "TTS talk-back" -- pauses the stream via ``arbiter.acquire(owner)``.
        TTS was the one consumer that never did, so the always-on wake word
        listened straight through Jarvis's own voice.

        Held across the whole burst rather than per chunk: ReadAloud splits
        long text into many speak() calls, and every resume restarts the
        capture stream (openWakeWord blanks ~400 ms after reset()).
        """
        if CONFIG.barge_in:
            # Barge-in: the wake word stays live while Jarvis speaks, so
            # "Jarvis, stop" can cut him off. His own voice cannot wake him:
            # the speaker gate scores the TTS voice at -0.03..-0.06 against
            # the enrolled voiceprint (measured 2026-08-29).
            self._mic_hold = None
            return
        if self._arbiter is None or self._mic_hold is not None:
            return
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(self._arbiter.acquire("tts"))
        except Exception:
            log.exception("mic acquire for TTS failed -- speaking uncovered")
            return
        self._mic_hold = stack

    def _release_mic(self):
        """Resume the hotword. Must run on EVERY exit path: a leaked acquire
        leaves Jarvis permanently deaf, which is worse than hearing himself."""
        stack, self._mic_hold = self._mic_hold, None
        if stack is None:
            return
        try:
            stack.close()
        except Exception:
            log.exception("mic release after TTS failed")

    def _worker_loop(self):
        while True:
            text, done = self._q.get()
            try:
                if not self._speaking:
                    self._speaking = True
                    self._burst_announced = False
                    self._acquire_mic()
                    # SpeakingState(active=True) is NOT published here. It is
                    # the app's "audio" mark, and the turn ledger's "wait"
                    # ends on it -- publishing before _speak_sync renders
                    # anything put the mark 0.4-0.8 s ahead of the first
                    # sound (live log 2026-08-29: "turn:" landed 13-90 ms
                    # after "speaking (fish)", before any paplay existed).
                    # _mark_audio() sends it when playback actually starts.
                self._speak_sync(text)
            except Exception:
                log.exception("TTS worker error")
            finally:
                done.set()
                if self._q.empty():
                    self._speaking = False
                    # Release BEFORE publishing: if a subscriber raises out of
                    # this finally the worker thread dies, and a stranded
                    # acquire would mute the hotword for the whole session.
                    self._release_mic()
                    if not self._burst_announced:
                        # Nothing played (synthesis failed, or stopped before
                        # playback): subscribers pair the falling edge with a
                        # rising one, so give them the edge -- late, but the
                        # turn still closes and the follow-up mic still opens.
                        self._mark_audio()
                    bus.publish(SpeakingState(active=False, amplitude=0.0))

    def _mark_audio(self):
        """Publish the burst's SpeakingState(active=True) once, at the moment
        the first audio is handed to a player -- the honest 'audio' mark."""
        if self._burst_announced:
            return
        self._burst_announced = True
        bus.publish(SpeakingState(active=True, amplitude=0.0))

    def _pronounce(self, text: str) -> str:
        """Apply the pronunciation dictionary (never fails speech)."""
        if not self._pronunciation:
            return text
        try:
            eng = self._engine
            return pronounce.apply(
                text,
                rewrite_times=_ENGINE_NEEDS_TIME_REWRITE.get(eng, True),
                unshout_words=_ENGINE_NEEDS_UNSHOUT.get(eng, True)) or text
        except Exception:
            log.exception("pronunciation apply failed")
            return text

    def _cache_key(self, engine: str, spoken: str) -> str:
        if engine == "fish":
            _key, model = _fish_creds()
            return SpeechCache.key("fish", spoken, model=model or "none",
                                   backend=FISH_BACKEND)
        if engine == "f5":
            try:
                st = F5_REF.stat()
                ref = f"{st.st_size}:{int(st.st_mtime)}"
            except OSError:
                ref = "none"
            return SpeechCache.key("f5", spoken, ref=ref, **F5_PARAMS)
        if engine == "xtts":
            try:
                st = VOICE_REF.stat()
                ref = f"{st.st_size}:{int(st.st_mtime)}"
            except OSError:
                ref = "none"
            return SpeechCache.key("xtts", spoken, ref=ref, **XTTS_PARAMS)
        return SpeechCache.key("edge", spoken, voice=EDGE_VOICE,
                               rate=EDGE_RATE, pitch=EDGE_PITCH)

    def _cached(self, engine: str, spoken: str) -> Optional[str]:
        if self.cache is None:
            return None
        try:
            hit = self.cache.get(self._cache_key(engine, spoken))
        except Exception:
            log.exception("speech cache lookup failed")
            return None
        return str(hit) if hit is not None else None

    def _store(self, engine: str, spoken: str, path: str) -> None:
        if self.cache is None:
            return
        try:
            self.cache.put(self._cache_key(engine, spoken), path)
        except Exception:
            log.exception("speech cache store failed")

    def _speak_sync(self, text: str):
        """Synthesize and play one utterance (runs on the worker thread)."""
        if not self.load():
            return
        self._stop_flag = False

        spoken = self._pronounce(text)
        # load() may have fallen back to edge, so re-check the engine here.
        if self._engine in ("f5", "fish"):
            log.info("speaking (%s): %.60s", self._engine, text)
            self._speak_pipelined(spoken, self._engine)
            return
        if self._engine == "xtts" and self._xtts is not None:
            log.info("speaking (xtts): %.60s", text)
            self._speak_pipelined(spoken, "xtts")
            return

        cached = self._cached("edge", spoken)
        log.info("speaking (edge%s): %.60s", ", cached" if cached else "", text)
        path = cached
        tmp_name = None
        try:
            if path is None:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                tmp_name = tmp.name
                self._synth_edge(spoken, tmp_name)
                self._store("edge", spoken, tmp_name)
                path = tmp_name

            if self._stop_flag:
                return

            self._start_amp_feeder(path)
            self._play(path)
            log.info("speech complete")
        finally:
            self._amp_playing = False
            self._current_amp = 0.0
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
                except OSError:
                    log.exception("temp wav unlink failed: %s", tmp_name)

    def _speak_xtts_pipelined(self, text: str):
        """Back-compat alias for the XTTS arm of _speak_pipelined."""
        return self._speak_pipelined(text, "xtts")

    def _speak_pipelined(self, text: str, engine: str = "xtts"):
        """Sentence pipelining: synthesize chunk N+1 while chunk N plays.

        Shared by the two local cloning engines (xtts, f5); only the
        per-chunk synth call and the cache namespace differ.

        A producer thread synthesizes sentence chunks to per-chunk temp wavs
        (or takes them from the speech cache) and feeds them through a
        queue; this (worker) thread plays each chunk via the usual
        paplay→pw-play→aplay chain, streaming amplitude events per chunk.
        stop() aborts synthesis at the next sentence boundary and playback
        immediately; queued-but-unplayed temp wavs are drained + unlinked
        (cached files are never unlinked). Returns only after the LAST chunk
        finishes (blocking semantics).
        """
        synth = {"f5": self._synth_f5, "fish": self._synth_fish}.get(
            engine, self._synth_xtts)
        chunks = self._split_sentences(text)
        wav_q: queue.Queue = queue.Queue()
        _DONE = object()
        streaming = engine == "fish" and FISH_STREAM_PLAYBACK

        def _stream_fish(sent: str, tmp_name: str) -> bool:
            """Play-while-rendering for one fish chunk. Hands an _AudioStream
            to the consumer BEFORE the first byte arrives and tees the bytes
            into ``tmp_name`` for the cache. Returns False when the chunk
            must be rendered by the fallback engine instead (it failed before
            any audio reached the player); raises nothing."""
            stream = _AudioStream(tmp_name)
            wav_q.put(stream)
            try:
                with open(tmp_name, "wb") as fh:
                    for data in _fish_iter(sent, FISH_TIMEOUT_S):
                        if self._stop_flag:
                            break
                        fh.write(data)
                        stream.feed(data)
            except Exception as exc:
                log.exception("fish stream failed: %.60s", sent)
                if _is_out_of_credit(exc):
                    self.retire_fish("out of credit")
                stream.close(failed=True)
                # Audio already reached the ear: re-rendering the sentence
                # locally would say the first half twice. Lose the tail of
                # this one chunk instead; the next chunk goes local.
                if stream.heard:
                    log.warning("fish stream cut mid-chunk; not re-speaking")
                    return True
                return False
            if not self._stop_flag:
                # Store before close(): the consumer unlinks the tee file as
                # soon as it has played, and close() is what lets it finish.
                self._store(engine, sent, tmp_name)
            stream.close()
            return True

        def _producer():
            try:
                for sent in chunks:
                    if self._stop_flag:
                        break
                    cached = self._cached(engine, sent)
                    if cached is not None:
                        wav_q.put((cached, False))
                        continue
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False)
                    tmp.close()
                    if streaming and self._engine == "fish":
                        if _stream_fish(sent, tmp.name):
                            continue
                        # failed before any audio: fall back below, into a
                        # fresh temp file (the consumer owns the first one)
                        tmp = tempfile.NamedTemporaryFile(
                            suffix=".wav", delete=False)
                        tmp.close()
                        try:
                            log.warning("falling back to %s for this chunk",
                                        FISH_FALLBACK)
                            if self.load_fallback():
                                self._synth_f5(sent, tmp.name)
                                wav_q.put((tmp.name, True))
                                continue
                        except Exception:
                            log.exception("fallback synth failed too")
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass
                        continue
                    used = engine            # who actually rendered it
                    try:
                        if engine == "fish" and self._engine != "fish":
                            # retired mid-reply: straight to the local
                            # engine, no doomed API round-trip first
                            used = "f5"
                            self._synth_f5(sent, tmp.name)
                        else:
                            synth(sent, tmp.name)
                    except Exception as exc:
                        log.exception("%s chunk synth failed: %.60s",
                                      engine, sent)
                        if engine == "fish":
                            # the network died mid-reply: finish locally
                            # rather than dropping the rest of the sentence.
                            # An exhausted balance is different: retire fish
                            # for good so later chunks and later launches do
                            # not each pay a doomed API round-trip first.
                            try:
                                if _is_out_of_credit(exc):
                                    self.retire_fish("out of credit")
                                log.warning("falling back to %s for this chunk",
                                            FISH_FALLBACK)
                                if self.load_fallback():
                                    self._synth_f5(sent, tmp.name)
                                    wav_q.put((tmp.name, True))
                                    continue
                            except Exception:
                                log.exception("fallback synth failed too")
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass
                        continue
                    if not self._stop_flag:
                        # under the RENDERING engine's key: F5 audio filed
                        # as "fish" would replay in the wrong voice from
                        # cache once the account recovers
                        self._store(used, sent, tmp.name)
                    wav_q.put((tmp.name, True))
            finally:
                wav_q.put(_DONE)

        producer = threading.Thread(
            target=_producer, daemon=True, name=f"tts-{engine}-synth")
        producer.start()
        try:
            while True:
                item = wav_q.get()      # producer always ends with _DONE
                if item is _DONE:
                    break
                if isinstance(item, _AudioStream):
                    try:
                        if not self._stop_flag:
                            self._play_stream(item)
                    finally:
                        # Played to the end -> the producer has closed the
                        # stream and _store has copied the tee, so the wait
                        # is instant. Stopped mid-stream -> do NOT wait on a
                        # stalled network; unlinking a file the producer
                        # still holds open is fine, and a stopped chunk is
                        # never stored anyway.
                        if not self._stop_flag:
                            item.done.wait(timeout=FISH_TIMEOUT_S + 5)
                        try:
                            os.unlink(item.path)
                        except FileNotFoundError:
                            pass
                        except OSError:
                            log.exception("temp wav unlink failed: %s",
                                          item.path)
                    continue
                path, owned = item
                try:
                    if not self._stop_flag:
                        self._start_amp_feeder(path)
                        self._play(path)
                finally:
                    if owned:
                        try:
                            os.unlink(path)
                        except FileNotFoundError:
                            pass
                        except OSError:
                            log.exception("temp wav unlink failed: %s", path)
            if not self._stop_flag:
                log.info("speech complete")
        finally:
            self._amp_playing = False
            self._current_amp = 0.0
            producer.join(timeout=60)

    # ------------------------------------------------------- sentence split
    _ABBREV_TAIL = re.compile(
        r'(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|fig|dept'
        r'|est|approx|min|max|e\.g|i\.e)|\b[A-Za-z])[.;]$', re.IGNORECASE)

    _MAX_CHUNK_CHARS = 160
    # A chunk may be at most this many times the length of the one before it.
    # Measured on the un-wedged GB10 (2026-08-28): XTTS renders at RTF ~0.29,
    # i.e. ~0.053 s of audio and ~0.0154 s of synthesis per character, so a
    # chunk breaks even against its predecessor's playback at 3.44x. 2.5x
    # keeps a margin for XTTS's run-to-run sampling variance.
    _CHUNK_GROWTH = 2.5

    def _split_sentences(self, text: str, min_chars: int = 20,
                         max_chars: int | None = None) -> list[str]:
        """Split text into chunks for pipelined synthesis.

        Splits on [.!?;]+whitespace, keeps common abbreviations (Mr. / e.g. /
        single initials) attached, and merges fragments shorter than
        ``min_chars`` forward (a short trailing fragment merges backward).

        Chunks longer than ``max_chars`` are then broken again at commas.
        Sentence-only splitting starved playback: a short opening sentence
        followed by one long comma-separated list meant chunk 2 took longer to
        synthesise than chunk 1 took to play, and the gap was audible
        (2026-08-28 13:02). Smaller chunks keep the producer ahead.
        """
        parts = re.split(r'(?<=[.!?;])\s+', text)
        max_chars = self._MAX_CHUNK_CHARS if max_chars is None else max_chars
        chunks: list[str] = []
        buf = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            buf = f"{buf} {part}" if buf else part
            if self._ABBREV_TAIL.search(buf) or len(buf) < min_chars:
                continue                 # merge forward into the next part
            chunks.append(buf)
            buf = ""
        if buf:
            if chunks and len(buf) < min_chars:
                chunks[-1] = f"{chunks[-1]} {buf}"   # tiny tail merges back
            else:
                chunks.append(buf)

        # Second pass: a chunk far longer than the one before it starves
        # playback, so break the long ones again at commas.
        #
        # The cap is GRADUATED, not fixed. A uniform max_chars still ran dry
        # when a short opener led (measured 2026-08-28 after the GPU cold
        # drain: "Here is your briefing." buys 1.57s of audio, but the 153-char
        # chunk behind it needs 2.40s to synthesise -- an audible ~1s hole in
        # 5 of 5 reps). What bounds a chunk is the playback its PREDECESSOR
        # buys, so the allowance starts small and grows with each chunk
        # emitted, converging on max_chars once there is enough cover.
        def limit_after(prev: str | None) -> int:
            if prev is None:
                return max_chars
            return max(min_chars, min(max_chars,
                                      int(self._CHUNK_GROWTH * len(prev))))

        out: list[str] = []
        for chunk in chunks:
            limit = limit_after(out[-1] if out else None)
            if len(chunk) <= limit:
                out.append(chunk)
                continue
            piece = ""
            for part in re.split(r"(?<=,)\s+", chunk):
                candidate = f"{piece} {part}".strip() if piece else part
                if piece and len(candidate) > limit:
                    out.append(piece)
                    piece = part
                    # the chunk just emitted buys cover for the next one
                    limit = limit_after(out[-1])
                else:
                    piece = candidate
            if piece:
                if out and len(piece) < min_chars:
                    out[-1] = f"{out[-1]} {piece}"
                else:
                    out.append(piece)
        return out or [text]

    # --------------------------------------------------------- envelope
    def _start_amp_feeder(self, wav_path: str):
        """Extract the amplitude envelope and stream it as SpeakingState
        events at ~12Hz (one 80ms chunk per event) while audio plays."""
        try:
            import numpy as np
            import soundfile as sf
            audio_data, sr = sf.read(wav_path)
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]
            # RMS amplitude per 80ms chunk (verbatim from jarvis_tts.py)
            chunk_size = int(sr * 0.08)
            envelope = []
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                envelope.append(min(1.0, rms * 4))
        except Exception:
            log.exception("amplitude envelope extraction failed")
            return
        self._run_amp_feeder(iter(envelope))

    def _run_amp_feeder(self, source: Iterator[float]):
        """Stream ``source`` (one value per 80 ms window) as SpeakingState
        amplitude events. The file path hands over a precomputed list; the
        fish streaming path hands over a generator fed as bytes arrive."""
        # The feeder's first tick is a SpeakingState(active=True): make sure
        # the burst's rising edge is the mark, a few ms before the player.
        self._mark_audio()
        self._amp_gen += 1
        gen = self._amp_gen              # a newer chunk's feeder supersedes us
        self._amp_playing = True

        def _feed_amp():
            for amp in source:
                if (gen != self._amp_gen or not self._amp_playing
                        or self._stop_flag):
                    break
                self._current_amp = amp
                bus.publish(SpeakingState(active=True, amplitude=amp))
                time.sleep(0.08)
            if gen == self._amp_gen:     # don't stomp a newer chunk's feeder
                self._current_amp = 0.0
                self._amp_playing = False
                bus.publish(SpeakingState(active=True, amplitude=0.0))

        threading.Thread(target=_feed_amp, daemon=True,
                         name="tts-amp-feeder").start()

    # ---------------------------------------------------------- playback
    def _play(self, wav_path: str):
        """Play a wav via paplay → pw-play → aplay (chain order unchanged).

        Uses Popen + poll so stop() can interrupt playback; per-player
        timeout stays 30s, non-zero exit falls through to the next player.
        """
        dev = (CONFIG.playback_device or "").strip()
        # An explicit sink (the echo-cancelling one) for the two players
        # that can take one; aplay is the no-PipeWire fallback.
        for cmd in [["paplay", *(["--device", dev] if dev else []), wav_path],
                    ["pw-play", *(["--target", dev] if dev else []), wav_path],
                    ["aplay", "-q", wav_path]]:
            if self._stop_flag:
                return
            self._mark_audio()           # the honest "audio" mark: at Popen
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                continue
            self._play_proc = proc
            try:
                if not self._wait_player(proc, deadline_s=30):
                    return
            finally:
                self._play_proc = None
            if proc.returncode == 0:
                return
            # non-zero exit → try the next player in the chain

    def _wait_player(self, proc: subprocess.Popen, deadline_s: float) -> bool:
        """Poll ``proc`` until it exits. False when it was cut short by
        stop()/skip_current() or the deadline (and has been terminated)."""
        deadline = time.monotonic() + deadline_s
        while proc.poll() is None:
            if self._stop_flag or time.monotonic() > deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                return False
            time.sleep(0.05)
        return True

    def _play_stream(self, stream: _AudioStream):
        """Play a chunk's audio as it arrives: paplay reads the wav from
        stdin, so the first bytes sound while the rest is still rendering.

        Waits for the FIRST bytes before spawning anything, so a chunk that
        failed before producing audio (the fallback engine renders it next)
        costs no player and no false "audio" mark. Only paplay reads a wav
        from a pipe reliably; without it, or if it dies at once (no sound
        server), the complete tee file goes through the usual chain.
        """
        first = None
        while first is None:
            if self._stop_flag:
                return
            try:
                first = stream.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:            # closed with no audio at all
                return
        dev = (CONFIG.playback_device or "").strip()
        cmd = ["paplay", *(["--device", dev] if dev else [])]
        started = time.monotonic()
        self._mark_audio()
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            self._play_stream_from_file(stream)
            return
        self._play_proc = proc
        envelope = _LiveEnvelope()
        self._run_amp_feeder(envelope.values())
        data: bytes | None = first
        try:
            while data is not None:
                if self._stop_flag:
                    break
                try:
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError):
                    break                # the player died or was terminated
                envelope.push(data)
                while True:
                    if self._stop_flag:
                        data = None
                        break
                    try:
                        data = stream.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue
            try:
                proc.stdin.close()
            except OSError:
                pass
            # Deadline covers the buffered audio still in the sound server
            # plus a slow stream; the file path allows 30 s per player.
            cut = not self._wait_player(proc, deadline_s=FISH_TIMEOUT_S + 30)
        finally:
            self._play_proc = None
            envelope.close()
        if cut or self._stop_flag or proc.returncode == 0:
            return
        if time.monotonic() - started < 0.5:
            # died immediately (no server for paplay, say): nothing was
            # heard, so the complete file can safely go down the chain
            log.warning("paplay stdin playback failed (rc=%s); using the "
                        "file chain", proc.returncode)
            self._play_stream_from_file(stream)

    def _play_stream_from_file(self, stream: _AudioStream):
        """Fallback: wait for the tee file to be complete, then play it
        exactly like a rendered chunk."""
        if not stream.done.wait(timeout=FISH_TIMEOUT_S + 5):
            return
        while True:                      # drain what the pipe path would have
            try:
                if stream.get(timeout=0) is None:
                    break
            except queue.Empty:
                break
        if stream.failed or self._stop_flag:
            return
        self._start_amp_feeder(stream.path)
        self._play(stream.path)

    # -------------------------------------------------------- synthesis
    def retire_fish(self, reason: str):
        """Switch to the local engine permanently and remember the choice.

        Called when fish fails in a way that will not recover on its own
        (an exhausted balance). Idempotent: safe to call on every chunk of a
        reply that is failing repeatedly.
        """
        if self._engine == FISH_FALLBACK:
            return
        log.warning("fish retired (%s) — switching to %s permanently",
                    reason, FISH_FALLBACK)
        self._engine = FISH_FALLBACK
        try:
            CONFIG.tts_engine = FISH_FALLBACK
            CONFIG.save()
            log.info("persisted tts_engine=%s", FISH_FALLBACK)
        except Exception:
            # a failed save must not take the voice down; the session is
            # already local, only the persistence is lost.
            log.exception("could not persist tts_engine")

    def load_fallback(self) -> bool:
        """Bring the local fallback engine up (used when fish fails)."""
        if FISH_FALLBACK == "f5":
            return _ensure_f5_server()
        return True

    def _synth_fish(self, text: str, out_path: str):
        """Render one chunk through the hosted API. Raises on failure so the
        caller can fall back locally rather than emitting silence."""
        res = _fish_stream(text, out_path, FISH_TIMEOUT_S)
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("fish returned no audio")
        return res

    def _synth_f5(self, text: str, out_path: str):
        """Render one chunk through the sidecar. Raises on failure — silence
        from Jarvis reads as a broken assistant, so this must not be swallowed
        here; the caller decides whether to fall back."""
        resp = _f5_request({"text": text, "out": out_path,
                            "nfe": F5_PARAMS["nfe_step"],
                            "speed": F5_PARAMS["speed"]})
        if not resp.get("ok"):
            raise RuntimeError(f"f5 synthesis failed: {resp.get('error')}")
        return resp

    def _synth_edge(self, text: str, out_path: str):
        """Synthesize with Edge TTS (fast, ~1s warm). 15s hard timeout.
        (The stream is MP3 whatever the suffix; see module docstring.)"""
        import edge_tts
        communicate = edge_tts.Communicate(
            text, EDGE_VOICE,
            rate=EDGE_RATE, pitch=EDGE_PITCH,
        )
        # edge_tts is async — run in a private event loop with a timeout so
        # a dead network can't hang the worker forever.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                asyncio.wait_for(communicate.save(out_path), timeout=15))
        finally:
            loop.close()

    def _synth_xtts(self, text: str, out_path: str):
        """Synthesize one text (typically a single sentence chunk) with XTTS
        v2 voice clone into ``out_path``. Splits internally as a safety net
        for oversized inputs; the pipelined path feeds it per-chunk. One
        inference at a time (the prewarm thread shares the model)."""
        import numpy as np
        import soundfile as sf

        # Split into sentences for streaming
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                     if s.strip()]
        if not sentences:
            sentences = [text]

        all_wav = []
        with self._synth_lock:
            for sent in sentences:
                if self._stop_flag:
                    break
                wav = self._xtts.tts(
                    text=sent,
                    speaker_wav=str(VOICE_REF),
                    language="en",
                    **XTTS_PARAMS,
                )
                all_wav.append(np.array(wav))
            try:
                import torch
                torch.cuda.synchronize()      # idle the driver event thread
            except Exception:
                pass

        if all_wav:
            sf.write(out_path, apply_output_gain(np.concatenate(all_wav)),
                     24000)

    # ---------------------------------------------------------- cleaning
    def _clean_for_speech(self, text: str) -> str:
        """Clean text for natural speech output (verbatim from jarvis_tts)."""
        text = re.sub(r'```[\s\S]*?```', ' code block omitted ', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'(/[a-zA-Z0-9_./\-]+){3,}', ' file path omitted ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > self.MAX_SPEAK_LENGTH:
            cut = text[:self.MAX_SPEAK_LENGTH].rfind('.')
            if cut > self.MAX_SPEAK_LENGTH // 2:
                text = text[:cut + 1]
            else:
                text = text[:self.MAX_SPEAK_LENGTH] + "..."

        return text
