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
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from jarvis import pronounce
from jarvis.config import PATHS
from jarvis.events import SpeakingState, bus
from jarvis.logs import get_logger
from jarvis.speech_cache import SpeechCache

log = get_logger("tts")

VOICE_REF = PATHS.VOICE_REF          # ~/.aiws_trainer/jarvis_voice_ref.wav

_ENGINES = ("edge", "xtts")

# Voice parameters (unchanged from the V1 engine); they are part of the
# speech-cache key so a tuning change never replays stale audio.
EDGE_VOICE = "en-GB-RyanNeural"
EDGE_RATE = "+5%"
EDGE_PITCH = "-4Hz"
XTTS_PARAMS = dict(speed=1.16, temperature=0.65, top_p=0.85,
                   repetition_penalty=5.0)


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
        """Load the TTS engine. Edge needs no preload; XTTS loads the model."""
        if self._engine == "edge":
            return True
        if self._xtts is not None:
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
            return True
        except Exception:
            log.exception("XTTS load error — falling back to edge engine")
            self._engine = "edge"
            return False

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
                        if engine == "xtts" and self._xtts is not None:
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
            return pronounce.apply(text) or text
        except Exception:
            log.exception("pronunciation apply failed")
            return text

    def _cache_key(self, engine: str, spoken: str) -> str:
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
        if self._engine == "xtts" and self._xtts is not None:
            log.info("speaking (xtts): %.60s", text)
            self._speak_xtts_pipelined(spoken)
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
        """XTTS sentence pipelining: synthesize chunk N+1 while chunk N plays.

        A producer thread synthesizes sentence chunks to per-chunk temp wavs
        (or takes them from the speech cache) and feeds them through a
        queue; this (worker) thread plays each chunk via the usual
        paplay→pw-play→aplay chain, streaming amplitude events per chunk.
        stop() aborts synthesis at the next sentence boundary and playback
        immediately; queued-but-unplayed temp wavs are drained + unlinked
        (cached files are never unlinked). Returns only after the LAST chunk
        finishes (blocking semantics).
        """
        chunks = self._split_sentences(text)
        wav_q: queue.Queue = queue.Queue()
        _DONE = object()

        def _producer():
            try:
                for sent in chunks:
                    if self._stop_flag:
                        break
                    cached = self._cached("xtts", sent)
                    if cached is not None:
                        wav_q.put((cached, False))
                        continue
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False)
                    tmp.close()
                    try:
                        self._synth_xtts(sent, tmp.name)
                    except Exception:
                        log.exception("XTTS chunk synth failed: %.60s", sent)
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass
                        continue
                    if not self._stop_flag:
                        self._store("xtts", sent, tmp.name)
                    wav_q.put((tmp.name, True))
            finally:
                wav_q.put(_DONE)

        producer = threading.Thread(
            target=_producer, daemon=True, name="tts-xtts-synth")
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
        out: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                out.append(chunk)
                continue
            piece = ""
            for part in re.split(r"(?<=,)\s+", chunk):
                candidate = f"{piece} {part}".strip() if piece else part
                if piece and len(candidate) > max_chars:
                    out.append(piece)
                    piece = part
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
            sf.write(out_path, np.concatenate(all_wav), 24000)

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
