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
from typing import Iterable, Optional

from jarvis import pronounce
from jarvis.config import CONFIG, PATHS
from jarvis.events import SpeakingState, bus
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
_FISH_CREDIT_MARKERS = ("402", "payment required", "insufficient", "quota",
                        "out of credit", "credit", "balance", "billing")


def _is_out_of_credit(exc: BaseException) -> bool:
    """True when a fish failure means the balance is gone, not the network."""
    text = f"{type(exc).__name__}: {exc}".lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 402:
        return True
    # "credit"/"balance" are broad, so require an explicit payment signal too
    return any(m in text for m in _FISH_CREDIT_MARKERS)


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
    started = time.monotonic(); first = None
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


def _fish_stream(text: str, out_path: str, timeout: float) -> dict:
    """Render one chunk via the Fish API. The network seam — tests patch this."""
    from fish_audio_sdk import Session, TTSRequest
    key, model = _fish_creds()
    if not key or not model:
        raise RuntimeError("fish credentials missing")
    session = Session(key)
    started = time.monotonic()
    first = None
    with open(out_path, "wb") as fh:
        for chunk in session.tts(
                TTSRequest(text=text, reference_id=model, format="wav",
                           latency="balanced"),
                backend=FISH_BACKEND):
            if first is None:
                first = time.monotonic() - started
            fh.write(chunk)
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"fish exceeded {timeout}s")
    return {"ok": True, "ttfa": first}


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

_f5_proc = None
_f5_lock = threading.Lock()


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
    """Start the sidecar if it is not already answering. True when ready."""
    global _f5_proc
    with _f5_lock:
        if _f5_alive():
            return True
        for path in (F5_PYTHON, F5_SERVER, F5_REF, F5_REF_TEXT):
            if not Path(path).exists():
                log.error("f5 sidecar cannot start, missing: %s", path)
                return False
        F5_SOCK.parent.mkdir(parents=True, exist_ok=True)
        log.info("starting f5 sidecar")
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
        self._amp_playing = False
        self._amp_gen = 0                # generation token: one per chunk feeder
        self._current_amp = 0.0
        self._play_proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()   # (text, done_event)
        self._arbiter = arbiter          # MicArbiter; None when standalone
        self._mic_hold = None            # live acquire() for this burst
        self.cache: Optional[SpeechCache] = (
            SpeechCache(cache_dir) if cache else None)
        self._pronunciation = pronunciation
        self._synth_lock = threading.Lock()    # one XTTS inference at a time
        self._load_lock = threading.Lock()     # one XTTS LOAD at a time
        self._prewarm_thread: threading.Thread | None = None
        self.last_text = ""                    # last cleaned utterance queued
        self.interrupts = 0                    # barge-ins that cut speech
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="tts-worker")
        self._worker.start()

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
                return True
            log.warning("fish credentials missing (%s) — falling back to %s",
                        FISH_KEY_FILE, FISH_FALLBACK)
            self._engine = FISH_FALLBACK
            return self.load()
        if self._engine == "f5":
            if _ensure_f5_server():
                return True
            log.warning("f5 sidecar unavailable — falling back to edge")
            self._engine = "edge"
            return False
        if self._xtts is not None:
            return True
        with self._load_lock:
            return self._load_xtts_locked()

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
            log.exception("XTTS load error — falling back to edge engine")
            self._engine = "edge"
            return False

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

    def speak(self, text: str, block: bool = False):
        """Enqueue text for speech; the worker drains FIFO (no silent drops).

        Args:
            text: text to speak (cleaned/truncated before synthesis)
            block: if True, wait until this utterance finishes (or is stopped)
        """
        if not text or not text.strip():
            return
        text = self._clean_for_speech(text)
        if not text:
            return
        self.last_text = text
        done = threading.Event()
        self._q.put((text, done))
        if block:
            done.wait()

    def stop(self):
        """Stop current speech and clear the pending queue."""
        self._stop_flag = True
        self._clear_queue()
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
                    self._acquire_mic()
                    bus.publish(SpeakingState(active=True, amplitude=0.0))
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
                    bus.publish(SpeakingState(active=False, amplitude=0.0))

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
                    try:
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
                        self._store(engine, sent, tmp.name)
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

        self._amp_gen += 1
        gen = self._amp_gen              # a newer chunk's feeder supersedes us
        self._amp_playing = True

        def _feed_amp():
            for amp in envelope:
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
        for cmd in [["paplay", wav_path], ["pw-play", wav_path],
                    ["aplay", "-q", wav_path]]:
            if self._stop_flag:
                return
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                continue
            self._play_proc = proc
            try:
                deadline = time.monotonic() + 30
                while proc.poll() is None:
                    if self._stop_flag or time.monotonic() > deadline:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=2)
                        return
                    time.sleep(0.05)
            finally:
                self._play_proc = None
            if proc.returncode == 0:
                return
            # non-zero exit → try the next player in the chain

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
