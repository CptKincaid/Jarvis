"""Recorder for Jarvis V3 — mic streams, silence auto-stop, calibration, beeps.

Ports the audio-critical paths of voice_input_gui.py verbatim (this machine has
no microphone, so the constants and order of operations must move untouched):

- beep playback                        (monolith 526-573; the synth itself
  moved to jarvis/earcons.py on 2026-08-30, see the note above _init_beeps)
- noise calibration                    (monolith 2262-2340)
- 1s restart debounce                  (monolith 2360-2364)
- audio callback + silence detection   (monolith 2403-2434)
- resample to 16k                      (monolith 2477-2485)
- stop finalization, 60s cap, gate     (monolith 2487-2541)
- silence / speaker-silence auto-stop  (monolith 4295-4441)
- fixed-duration recording             (monolith 5000-5054)

Differences from the monolith, per the V3 spec:
- No tkinter. Auto-stop runs on the recorder's own poll thread and publishes
  RecordingStopped instead of calling _stop_and_transcribe.
- Thresholds (noise_threshold, noise_gate flag) are cached BEFORE the audio
  callback closure is created — fixes the off-thread Tk variable read.
- All ad-hoc getattr state is an explicit __init__ field.
- No-mic machines: start() is a graceful no-op with MicState/Status events.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import contextmanager

import numpy as np

from jarvis.config import CONFIG, MACHINE
from jarvis.endpoint import trailing_filler
from jarvis.spelling import spelling_run
from jarvis.events import (
    AudioLevel,
    MicState,
    RecordingStarted,
    RecordingStopped,
    Status,
    bus,
)
from jarvis.logs import get_logger

log = get_logger("recorder")

# Audio constants — monolith 53-56, 89, 92 (verbatim)
SAMPLE_RATE = 16000
CHANNELS = 1
# 0.0025, not 0.005: his surviving speech blocks measure RMS 0.0083-0.0115
# and quiet word-tails were being zeroed by the old gate (2026-08-31 study).
NOISE_GATE_THRESHOLD = 0.0025
WAVEFORM_BARS = 64
MAX_RECORDING_SECONDS = 60      # monolith 4299 — hard cap to prevent memory issues
# The longest a follow-up capture may wait for the first word (lecture
# notes): half the hard cap, so a sentence started at the end of the wait
# still has 30 s before the cap cuts it.
MAX_FOLLOWUP_WINDOW_S = MAX_RECORDING_SECONDS / 2
# THE BACKSTOP (2026-09-02).  Every ordinary exit a capture has -- the VAD
# endpoint, the energy timer, the follow-up window, even the 60 s cap above
# -- is checked on ONE thread, _poll_loop.  There is no second one.  If that
# thread dies, is starved, or never starts, the mic stays open, the arbiter
# stays held (which pauses the wake word: Jarvis goes deaf), the mixer stays
# ducked and the board stays on "Listening…" -- with nothing logged.  This
# timer is the only exit that does not depend on that thread.  The grace
# keeps it strictly a backstop: the 60 s cap always gets there first.
WATCHDOG_GRACE_S = 5.0
# THE FILLER HOLD (Hunter, 08-31: "um/uh should buy him more time"). The
# live preview reports its newest decode through note_partial(); when that
# text ends on a filler AND the decoded span reaches to within
# FILLER_SLACK_S of the last speech the VAD heard, the um was the last
# thing said and _check_endpoint waits CONFIG.filler_hold_s beyond
# CONFIG.endpoint_silence. 0.6 s of slack, because the span can end inside
# the um's own tail (whisper writes "um" from its first half), the VAD's
# hysteresis hangs on a chunk or two after the voice fades, and the
# resampler lags 10 ms -- while a real word spoken after the um moves the
# last-speech mark past the slack, and the next preview (0.9 s cadence)
# clears the hold anyway. A wrong hold costs one filler_hold_s once; a
# missed one cuts him off mid-sentence. Only the trailing filler is kept:
# the recorder never stores the words.
FILLER_SLACK_S = 0.6
# THE DECODE THAT WILL SAY IS STILL RUNNING (09-06). When the stop is due
# and a decode is in flight whose snapshot was taken AFTER the last speech
# -- so it holds the whole of the burst that just ended -- the recorder
# waits for it instead of stopping past it, at most this much beyond
# endpoint_silence. The speculative pass starts 0.3 s into every pause and
# takes 0.24 s at his p50 and 0.46 s at his p90 (n=238), which with the
# preview thread's 0.08 s poll lands it 0.54-0.84 s in: the p90 tail
# missed the 0.8 s stop, and that tail is where the first spelled letter
# was lost 2-5 times in 9 (tests/test_spelling_survival.py). What the wait
# costs an ORDINARY turn: nothing measurable, because the real path waits
# for that same decode after the stop anyway (JarvisApp._take_speculation
# joins a pass in flight; the final transcribe queues behind a greedy one
# on the model lock), so the words reach the commander at the same
# moment; the ledger's dead-air reads the wait. The cap is what bounds a
# decode that has hung: below the 2.0 s spelling hold and the 2.5 s
# energy timer, so it can end nothing later than they would.
DECODE_WAIT_MAX_S = 1.0
# A snapshot may fall this far short of the last speech chunk and still be
# taken to hold the burst: the endpointer scores 32 ms chunks and the last
# tenth of a spelled character is its decay.
DECODE_COVER_SLACK_S = 0.1


# ------------------------------------------------------------------
# Non-speech cues — the three historical beeps, now the earcon lexicon
# (jarvis/earcons.py).  The synth that used to live here (port of monolith
# 526-573) was retired on 2026-08-30: it and earcons.py were two beep
# systems with different timbres and envelopes, which is exactly the
# failure a shared "voice for the room" exists to prevent.
# ------------------------------------------------------------------
_BEEP_FILES: dict[str, str] = {}
_BEEP_LOCK = threading.Lock()


def _init_beeps():
    """Resolve the three historical beep kinds to earcon WAVs.

    These three pitches -- start 880, stop 660, nudge 440 -- were the whole
    of Jarvis's non-speech vocabulary, and they are a root, a fifth and an
    octave. jarvis/earcons.py adopted them as its family and re-skinned
    them with one shared timbre and envelope; this function now only maps
    the old kind names onto it, so there is exactly ONE beep system rather
    than two with different accents.
    """
    with _BEEP_LOCK:
        if _BEEP_FILES:
            return
        try:
            from jarvis import earcons
            for kind in ("start", "stop", "nudge"):
                path = earcons.render(kind)
                if path is not None:
                    _BEEP_FILES[kind] = str(path)
        except Exception:
            log.exception("beep init failed")


def play_beep(kind: str):
    """Play the 'start', 'stop' or 'nudge' cue asynchronously.

    Delegates to the earcon lexicon, which owns the rate limiting and the
    ``sound.earcons`` gate; the paplay -> aplay chain is unchanged.
    """
    try:
        from jarvis import earcons
        earcons.play(kind)
    except Exception:
        log.exception("beep playback failed")


# ------------------------------------------------------------------
# Noise gate — port of monolith 601-610 (verbatim)
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
# Mic arbiter — THE single owner of the microphone
# ------------------------------------------------------------------
class MicArbiter:
    """Serializes mic ownership. The hotword listener registers pause/resume
    callbacks; every other consumer wraps its mic use in acquire(), which
    pauses the hotword stream and resumes it on exit. Re-entrant: nested
    acquires pause once and resume once."""

    def __init__(self):
        self._lock = threading.RLock()
        self._depth = 0
        self._owner = ""
        self._pause_cb = None
        self._resume_cb = None

    def register_hotword(self, pause_cb, resume_cb):
        with self._lock:
            self._pause_cb = pause_cb
            self._resume_cb = resume_cb

    @contextmanager
    def acquire(self, owner: str):
        """Pause hotword for the duration of the block; resume on exit."""
        with self._lock:
            self._depth += 1
            first = self._depth == 1
            if first:
                self._owner = owner
                if self._pause_cb is not None:
                    try:
                        self._pause_cb()
                    except Exception:
                        log.exception("hotword pause failed (owner=%s)", owner)
        try:
            yield self
        finally:
            with self._lock:
                self._depth = max(0, self._depth - 1)
                last = self._depth == 0
                if last:
                    self._owner = ""
                    if self._resume_cb is not None:
                        try:
                            self._resume_cb()
                        except Exception:
                            log.exception("hotword resume failed (owner=%s)", owner)

    @property
    def held_by(self) -> str:
        with self._lock:
            return self._owner


# ------------------------------------------------------------------
# Recorder
# ------------------------------------------------------------------
class Recorder:
    """Owns recording sessions. start()/stop()/abort() for the main pipeline;
    record_fixed() for enrollment/wake-training; calibrate_noise() for the
    noise-floor calibration flow. Publishes RecordingStarted/RecordingStopped,
    AudioLevel (~12Hz), MicState, and Status events. Never touches the UI.

    On auto-stop (silence / speaker-silence / 60s cap) the poll thread calls
    stop(reason=...); the finalized 16k mono float32 audio is stored in
    self.last_audio and RecordingStopped(reason) is published — the pipeline
    wiring reads recorder.last_audio on that event.
    """

    _POLL_S = 0.083            # AudioLevel ~12Hz
    _SILENCE_POLL_S = 0.3      # monolith 4344: root.after(300, ...)
    # Was 3.0, a literal port of the monolith's root.after(3000, ...) Tk
    # chain rather than a cost decision: an ECAPA verify measures ~5 ms, so
    # 1 Hz is under 1% duty. At 3 s the voice-ID stop could not win. Measured
    # on a real capture with music playing: first check at 3 s, misses at 6 s
    # and 9 s, so the 2.5 s timer could not even START before 9 s, and the
    # energy detector -- which music defeats -- ran the recording to 11.9 s.
    _SPEAKER_POLL_S = 1.0
    _SPEAKER_WARMUP_S = 1.5    # ECAPA needs ~1 s of audio to say anything
    _SPEAKER_WINDOW_S = 2.0    # the trailing audio judged; also the lag before
                               # a stopped speaker can register as gone
    # The filler hold's per-capture state, as CLASS defaults so a bare
    # Recorder (tests build one with object.__new__) reads as "no partial
    # yet, no holds". start() resets them through _reset_filler_hold().
    _latest_partial = None        # (filler, spell_n, audio_end_s, wallclock) of the newest decode
    _filler_holds = 0             # distinct pauses held this capture
    _filler_hold_key = None       # last_speech_seconds of the pause being held
    _spell_holds = 0              # ...and the same two for the SPELLING hold,
    _spell_hold_key = None        # counted apart (jarvis/spelling.py)
    _capture_id = 0               # bumped by every start(); see note_partial
    _decoding = None              # (audio_end_s, wallclock) of a decode in flight; note_decoding
    _decode_wait_key = None       # last_speech_seconds of the pause a wait was logged for
    # The commander is waiting on an ADDRESS (it asked "What is it?" or
    # read a draft back): a bare spelled answer is held from its FIRST
    # character (jarvis.spelling.spelling_run's address_owed). Read ONCE
    # per capture, at start(), from the probe the app installs -- never
    # mid-capture, so a question that expires while he is spelling cannot
    # drop the hold under him. None -> False: a bare Recorder owes nothing.
    _address_owed = False
    address_owed_probe = None     # () -> bool, set by the app (JarvisApp)

    def __init__(self, arbiter: MicArbiter, speaker_verifier=None):
        self._arbiter = arbiter
        # Optional jarvis.speaker.SpeakerVerifier for voice-aware auto-stop;
        # may be injected later by assigning recorder.speaker_verifier.
        self.speaker_verifier = speaker_verifier
        # Optional jarvis.endpoint.VoiceEndpointer, injected by the app once
        # the model is warm. None -> the energy timer alone ends captures.
        self.endpointer = None
        self._ep_cursor = 0                      # frames already fed to it
        self._followup = False                   # opened without a wake word
        self._followup_window = 0.0              # 0 -> CONFIG.followup_window
        self._stop_endpoint = ""                 # what ended the last capture
        self._stop_dead_air = None               # ...and how long it waited

        # Explicit state inventory (was scattered getattr state in the monolith)
        self.recording = False
        self.last_audio: np.ndarray | None = None
        self._stream = None
        self._audio_frames: list = []
        self._audio_level = 0.0
        self._waveform_buffer = deque(maxlen=WAVEFORM_BARS)
        self._record_rate = SAMPLE_RATE          # was getattr'd (monolith 2479)
        self._loud_chunks = 0                    # was getattr'd (monolith 2421)
        self._last_record_start = 0.0            # was hasattr'd (monolith 2360)
        self._record_start_time = 0.0
        self._silence_start = None
        self._speaker_silence_start = None       # Voice-ID silence tracker
        self._speaker_silence_misses = 0         # was getattr'd (monolith 4413)
        self._voice_id_match = False             # exposed for UI/reactor
        self._voice_stopped = False              # manual/auto stop — skip restart

        self._poll_thread: threading.Thread | None = None
        self._stop_lock = threading.Lock()
        self._session_ctx = None                 # held arbiter context
        # The backstop timer and its cap (see WATCHDOG_GRACE_S). An attribute
        # rather than a constant so a test can shorten it.
        self.watchdog_cap_s = MAX_RECORDING_SECONDS + WATCHDOG_GRACE_S
        self._watchdog: threading.Timer | None = None

        self._mic_devices: dict[str, int | None] = {"Default": None}
        self._detect_mics()
        self._publish_mic_state()

    # -- mic enumeration (port of monolith 1282-1293) -------------------
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
            log.warning("Mic detection error: %s", e)
            self._mic_devices = {"Default": None}

    @property
    def followup(self) -> bool:
        """True while (and after) a session opened without a wake word
        (start(followup=True)). Read by the app's nudge policy: a follow-up
        window that hears nothing is a normal outcome, not a lost turn."""
        return bool(self._followup)

    @property
    def mic_available(self) -> bool:
        return bool(MACHINE.has_mic)

    @property
    def mic_devices(self) -> dict:
        return dict(self._mic_devices)

    def _publish_mic_state(self):
        name = MACHINE.mic_names[0] if MACHINE.mic_names else ""
        bus.publish(MicState(available=self.mic_available, device_name=name))

    def _resolve_mic(self):
        """Configured mic name -> device index; fall back to default
        (port of monolith 2370-2377)."""
        mic_name = CONFIG.mic
        if mic_name not in self._mic_devices:
            log.info("Saved mic %r not found, using default", mic_name)
            return None
        return self._mic_devices.get(mic_name)

    def _native_rate(self, mic_idx) -> int:
        """Device native sample rate; many mics only do 44100/48000
        (port of monolith 2379-2384)."""
        try:
            import sounddevice as sd
            dev_info = sd.query_devices(mic_idx, 'input')
            return int(dev_info['default_samplerate'])
        except Exception:
            return 44100

    # -- resample (port of monolith 2477-2485, verbatim math) -----------
    def _resample_to_16k(self, audio):
        """Resample audio from recording rate to 16kHz for Whisper."""
        rate = self._record_rate
        if rate == SAMPLE_RATE:
            return audio
        from scipy.signal import resample
        new_len = int(len(audio) * SAMPLE_RATE / rate)
        resampled = resample(audio, new_len).astype(np.float32)
        return resampled

    # -- session control -------------------------------------------------
    @staticmethod
    def clamp_window(window) -> float:
        """A follow-up wait in seconds: 0 (use CONFIG.followup_window) for
        None / junk, never above MAX_FOLLOWUP_WINDOW_S."""
        try:
            return min(MAX_FOLLOWUP_WINDOW_S, max(0.0, float(window or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def start(self, followup: bool = False, window: float = None):
        """Begin a recording session. No-op (with warn Status) when no mic.

        ``followup=True`` opens the mic without a wake word right after a
        reply: if the VAD hears no speech within CONFIG.followup_window the
        session is aborted quietly (no "No audio captured", no transcript).
        ``window`` overrides that wait for one capture (lecture notes keep
        the mic open longer between sentences); it is capped at
        MAX_FOLLOWUP_WINDOW_S so the 60 s hard cap still leaves room to talk.
        """
        import_err = None
        if not self.mic_available:
            self._publish_mic_state()
            bus.publish(Status(text="No microphone detected", kind="warn"))
            log.warning("start() ignored: no input devices")
            return

        # Guard against rapid restart loop (port of monolith 2358-2364)
        if self.recording:
            return
        if self._last_record_start:
            elapsed = time.monotonic() - self._last_record_start
            if elapsed < 1.0:
                return
        self._last_record_start = time.monotonic()

        try:
            import sounddevice as sd
        except Exception as e:
            import_err = e
        if import_err is not None:
            log.exception("sounddevice unavailable")
            bus.publish(Status(text=f"Audio backend error: {import_err}", kind="error"))
            return

        # Pause hotword for the whole session (replaces the scattered
        # hotword pause at monolith 2367-2368)
        self._session_ctx = self._arbiter.acquire("recorder")
        self._session_ctx.__enter__()

        mic_idx = self._resolve_mic()
        native_rate = self._native_rate(mic_idx)
        self._record_rate = native_rate
        log.info("Mic native rate: %sHz", native_rate)

        # Reset session state (port of monolith 2388-2398)
        self._audio_frames = []
        self._silence_start = None
        self._loud_chunks = 0
        self._speaker_silence_start = None
        self._speaker_silence_misses = 0
        self._voice_id_match = False
        self._voice_stopped = False
        self._waveform_buffer.clear()
        self.last_audio = None
        self._ep_cursor = 0
        self._stop_endpoint, self._stop_dead_air = "", None
        self._reset_filler_hold()
        self._address_owed = self._probe_address_owed()
        self._followup = bool(followup)
        self._followup_window = self.clamp_window(window)
        if self.endpointer is not None:
            try:
                self.endpointer.reset()
            except Exception:
                log.debug("endpointer reset failed", exc_info=True)

        # Cache thresholds for the audio thread BEFORE creating the closure
        # (monolith 2400-2401 cached only silence_thresh and still read the
        # noise-gate Tk var inside the callback — both cached here).
        silence_thresh = CONFIG.noise_threshold
        gate_on = CONFIG.noise_gate

        # Audio callback (port of monolith 2403-2434, verbatim logic)
        def audio_callback(indata, frame_count, time_info, status):
            if not self.recording:
                return

            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()

            # Measure RMS on original audio BEFORE noise gate (for silence detection)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            self._audio_level = min(rms * 10, 1.0)

            # Silence detection uses original signal
            # Only reset silence timer on sustained sound (2+ consecutive loud chunks)
            # This prevents single noise spikes from restarting the countdown
            if rms < silence_thresh:
                self._loud_chunks = 0
                if self._silence_start is None:
                    self._silence_start = time.monotonic()
            else:
                self._loud_chunks = self._loud_chunks + 1
                if self._loud_chunks >= 2:
                    self._silence_start = None

            # Noise gate: zero out quiet blocks (after silence check)
            if gate_on and rms < NOISE_GATE_THRESHOLD:
                chunk[:] = 0.0

            self._audio_frames.append(chunk.reshape(-1, 1))

            # Waveform samples
            step = max(1, len(chunk) // 4)
            for val in chunk[::step]:
                self._waveform_buffer.append(float(val))

        # Set recording state BEFORE opening mic (mic open can block briefly)
        self.recording = True
        self._record_start_time = time.monotonic()
        self._arm_watchdog()
        bus.publish(RecordingStarted())
        bus.publish(Status(text="Listening...", kind="busy"))

        if CONFIG.sound:
            threading.Thread(target=play_beep, args=("start",), daemon=True).start()

        # Recorder-owned poll thread replaces the Tk-after chains
        # (_update_waveform / _check_silence / _check_speaker_silence)
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="recorder-poll")
        self._poll_thread.start()

        # Open mic stream (may block briefly)
        try:
            self._stream = sd.InputStream(
                samplerate=native_rate, channels=CHANNELS,
                dtype="float32", device=mic_idx,
                callback=audio_callback,
                blocksize=int(native_rate * 0.1),
            )
            self._stream.start()
        except Exception as e:
            log.exception("Mic open error")
            self.recording = False
            self._cancel_watchdog()
            bus.publish(Status(text=f"Mic error: {str(e)[:50]}", kind="error"))
            self._join_poll_thread()
            self._release_session()
            bus.publish(RecordingStopped(reason="abort"))
            return

        log.info("Recording started")

    # -- the backstop ----------------------------------------------------
    def _arm_watchdog(self):
        """One timer per session, off the poll thread (see WATCHDOG_GRACE_S)."""
        self._cancel_watchdog()
        try:
            cap = float(self.watchdog_cap_s)
        except (TypeError, ValueError):
            cap = MAX_RECORDING_SECONDS + WATCHDOG_GRACE_S
        if cap <= 0:
            return
        timer = threading.Timer(cap, self._watchdog_fire)
        timer.daemon = True
        timer.name = "recorder-watchdog"
        self._watchdog = timer
        timer.start()

    def _cancel_watchdog(self):
        timer, self._watchdog = self._watchdog, None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:                    # noqa: BLE001
                log.debug("watchdog cancel failed", exc_info=True)

    def _watchdog_fire(self):
        """Nothing ended this capture. End it here, and say so out loud."""
        if not self.recording:
            return
        held = self._arbiter.held_by or "recorder"
        elapsed = time.monotonic() - (self._record_start_time or time.monotonic())
        log.warning("recorder watchdog: the capture held by %r has run %.0fs "
                    "with no endpoint, no timer and no cap; forcing it shut",
                    held, elapsed)
        try:
            self.stop(reason="watchdog", endpoint="watchdog")
        except Exception:                        # noqa: BLE001
            log.exception("recorder watchdog: stop() failed; forcing the release")
        finally:
            # stop() clears both of these on every path it survives; if it
            # did not get that far, this does it by hand.
            if self.recording or self._session_ctx is not None:
                self._force_release("watchdog")

    def _force_release(self, reason: str = "watchdog"):
        """Put the world back with no dependence on anything above.

        The mic is shut, the arbiter handed back (a stranded acquire leaves
        Jarvis permanently deaf) and RecordingStopped published -- that one
        event is what lifts the mixer's duck and clears "Listening…" from
        the board.  Never a happy path: every step is independently guarded.
        """
        self.recording = False
        self._audio_level = 0.0
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:                    # noqa: BLE001
                log.warning("mic stream close failed", exc_info=True)
        try:
            self._release_session()
        except Exception:                        # noqa: BLE001
            log.exception("arbiter release failed")
        bus.publish(RecordingStopped(reason=reason, endpoint=reason,
                                     followup=self._followup))

    def stop(self, reason: str = "manual", endpoint: str = "",
             dead_air: float | None = None) -> np.ndarray | None:
        """End the session; finalize audio (resample, cap, gate — port of
        monolith 2487-2541). Publishes RecordingStopped(reason). Returns 16k
        mono float32 audio, or None if nothing usable was captured. The same
        array is stored in self.last_audio for event-driven consumers.

        ``endpoint`` / ``dead_air`` say which detector ended the capture and
        how long it waited. They are recorded HERE, under the lock, by the
        caller that wins: three detectors on three threads can each decide
        to stop within the same window, and a loser writing shared fields
        after the winner used to be what got published.
        """
        with self._stop_lock:
            if not self.recording:
                return None
            self.recording = False
            self._stop_endpoint = endpoint or reason
            self._stop_dead_air = dead_air
            t_stop = time.monotonic()        # the decision, not the teardown

        self._cancel_watchdog()
        self._audio_level = 0.0
        self._voice_stopped = True
        audio = None
        # EVERYTHING below is inside a finally. Before 2026-09-02 it was a
        # happy path: anything that raised (a thread that cannot be spawned,
        # a raising _finalize_audio) skipped the publish, and RecordingStopped
        # is the ONE event that lifts the mixer's duck and clears "Listening…"
        # from the board. A capture that ends without it looks, from every
        # seat in the room, exactly like one that never ended.
        try:
            if CONFIG.sound:
                threading.Thread(target=play_beep, args=("stop",),
                                 daemon=True).start()

            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    log.warning("mic stream close failed", exc_info=True)
                self._stream = None

            self._join_poll_thread()

            # Finalise BEFORE handing the mic back. Releasing first resumed
            # the wake word while the clip it had just captured was still
            # being assembled -- log 2026-08-27: "Hotword stream resumed" at
            # 58.229, "Stopped: 28.2s audio" at 58.397. try/finally because a
            # release skipped by a raising _finalize_audio would leave Jarvis
            # deaf.
            try:
                audio = self._finalize_audio()
            finally:
                self._release_session()
            self.last_audio = audio
            if audio is not None and os.environ.get("JARVIS_DEBUG_AUDIO") == "1":
                self._dump_capture(audio)
        finally:
            self._release_session()          # idempotent; a no-op above
            bus.publish(RecordingStopped(reason=reason,
                                         endpoint=self._stop_endpoint or reason,
                                         dead_air_s=self._stop_dead_air,
                                         t=t_stop, followup=self._followup,
                                         filler_holds=self._filler_holds,
                                         spell_holds=self._spell_holds))
        return audio

    def abort(self):
        """Discard the current session without producing audio."""
        with self._stop_lock:
            if not self.recording:
                return
            self.recording = False

        self._cancel_watchdog()
        self._audio_level = 0.0
        self._voice_stopped = True

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.warning("mic stream close failed", exc_info=True)
            self._stream = None

        self._join_poll_thread()
        self._release_session()

        self._audio_frames = []
        self.last_audio = None
        bus.publish(RecordingStopped(reason="abort", followup=self._followup,
                                     filler_holds=self._filler_holds,
                                     spell_holds=self._spell_holds))
        log.info("Recording aborted")

    def _dump_capture(self, audio_16k):
        """JARVIS_DEBUG_AUDIO=1: keep the last capture as a wav, tagged with
        what ended it, so an endpointing miss can be replayed offline through
        the VAD instead of guessed at. Off by default: it records the room."""
        try:
            import soundfile as sf
            from jarvis.config import PATHS
            out = PATHS.LOG_DIR / f"capture_last_{self._stop_endpoint or 'manual'}.wav"
            sf.write(out, audio_16k, SAMPLE_RATE)
            log.info("capture dump: %s", out)
        except Exception:
            log.debug("capture dump failed", exc_info=True)

    def _release_session(self):
        ctx, self._session_ctx = self._session_ctx, None
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                log.exception("arbiter release failed")

    def _join_poll_thread(self):
        t = self._poll_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._poll_thread = None

    def snapshot_audio(self) -> "np.ndarray | None":
        """Current buffer as 16 kHz float32, or None if there is nothing yet.

        Pure and side-effect free, unlike _finalize_audio: no Status events,
        no logging, no noise gate, no minimum-length rule. It is called
        repeatedly WHILE the user is still speaking to drive the live
        transcript, so it must never publish, never raise, and never disturb
        the session it is sampling.
        """
        frames = list(self._audio_frames)      # the callback may still append
        if not frames:
            return None
        try:
            raw = np.concatenate(frames, axis=0).flatten()
        except ValueError:
            return None
        if raw.size == 0:
            return None
        try:
            return self._resample_to_16k(raw)
        except Exception:
            log.debug("partial resample failed", exc_info=True)
            return None

    def snapshot_final(self) -> "np.ndarray | None":
        """snapshot_audio() shaped exactly as _finalize_audio would shape it
        (60 s cap, 0.3 s minimum, noise gate), still with no side effects.

        For the app's speculative transcription: a decode started during the
        endpoint silence is only worth reusing if it saw the SAME clip the
        normal path will see, and snapshot_audio() alone skips the cap and
        the gate (the gate zeroes quiet 100 ms blocks, which changes what
        whisper is given).
        """
        audio = self.snapshot_audio()
        if audio is None:
            return None
        return self._shape_final(audio)

    @staticmethod
    def _shape_final(audio: np.ndarray) -> "np.ndarray | None":
        """The final-clip rules shared by _finalize_audio and
        snapshot_final: cap at 60 s, drop under 0.3 s, noise-gate."""
        audio = audio[:SAMPLE_RATE * MAX_RECORDING_SECONDS]
        if len(audio) / SAMPLE_RATE < 0.3:
            return None
        if CONFIG.noise_gate:
            audio = _apply_noise_gate(audio)
        return audio

    def _finalize_audio(self) -> np.ndarray | None:
        """Snapshot frames -> 16k float32 (port of monolith 2508-2541)."""
        # Snapshot audio frames (callback thread may still be draining)
        frames = list(self._audio_frames)
        if not frames:
            bus.publish(Status(text="No audio captured", kind="info"))
            return None

        try:
            audio_raw = np.concatenate(frames, axis=0).flatten()
        except ValueError:
            log.error("audio frames empty after snapshot")
            bus.publish(Status(text="No audio captured", kind="info"))
            return None

        audio = self._resample_to_16k(audio_raw)
        duration = len(audio) / SAMPLE_RATE
        log.info("Stopped: %.1fs audio", duration)

        # Cap audio at 60 seconds to prevent freezes
        if len(audio) > SAMPLE_RATE * MAX_RECORDING_SECONDS:
            log.info("Audio capped from %.1fs to %ds", duration, MAX_RECORDING_SECONDS)

        # 60 s cap, 0.3 s minimum, noise gate -- shared with snapshot_final()
        # so a speculative decode and the real one see the same clip.
        shaped = self._shape_final(audio)
        if shaped is None:
            bus.publish(Status(text="Too short", kind="info"))
        return shaped

    # -- poll loop: AudioLevel + auto-stop -------------------------------
    def _poll_loop(self):
        """Publishes AudioLevel ~12Hz; runs the silence check every 300ms and
        the speaker-silence check every 3s (replaces the monolith's
        root.after chains at 2450-2452 / 4344 / 4390)."""
        next_silence = 0.0
        next_speaker = 0.0
        try:
            while self.recording:
                now = time.monotonic()
                waveform = [min(abs(v), 1.0) for v in self._waveform_buffer]
                bus.publish(AudioLevel(level=self._audio_level, waveform=waveform))
                if self._check_endpoint():
                    return
                if now >= next_silence:
                    next_silence = now + self._SILENCE_POLL_S
                    if self._check_silence():
                        return
                if now >= next_speaker:
                    next_speaker = now + self._SPEAKER_POLL_S
                    self._check_speaker_silence()
                time.sleep(self._POLL_S)
        except Exception:
            # A raising check used to kill this thread with `recording` still
            # True and the arbiter still held: the capture could then never
            # end (even the 60 s cap lives here). End it instead.
            log.exception("poll loop error; ending the capture")
            try:
                self.stop(reason="error")
            except Exception:
                log.exception("stop after poll error failed")
                self.recording = False

    def _check_endpoint(self) -> bool:
        """VAD endpointing: stop CONFIG.endpoint_silence after the user's last
        word. Returns True when it stopped the session.

        Runs every poll tick (~12 Hz) on the frames captured since the last
        tick, so a stop lands within ~100 ms of the threshold. Only ever
        stops AFTER speech has been heard: a capture in which nobody speaks
        is left to the energy timer. silence_grace still applies, so a pause
        that ends within the grace of the capture opening (the tail of the
        wake word, then thinking) cannot end it; a pause of endpoint_silence
        AFTER the grace does, and raising endpoint_silence is the remedy if
        that cuts a slow talker off.

        THE TWO HOLDS: once the gap has reached endpoint_silence and the
        stop is otherwise due, _hold_extra asks whether the newest preview
        decode ended on "um"/"uh" as the LAST thing heard -- if so the stop
        waits CONFIG.filler_hold_s longer (at most filler_max_holds pauses
        per capture) -- or on a SPELLING RUN (jarvis/spelling.py), which
        waits CONFIG.spell_hold_s for at most spell_max_holds pauses. One
        seam, two counters: a man spelling an address pauses between every
        character, and 0.8 s of quiet is what he leaves between two letters.
        LIMIT (the filler): the preview runs at 0.9 s cadence, the
        endpoint at 0.8 s, and the speculative pass parks the preview from
        0.3 s into a pause -- so a filler said after the last snapshot may
        never have been decoded when the stop is due. The recorder does not
        decode the tail itself (a decode on every turn's stop is latency
        every turn pays): it stops as before. scripts/filler_probe.py
        measures how often that happens. The SPELLING hold does not share
        that limit any more: the speculative pass -- which runs 0.3 s into
        EVERY pause, pause-synchronous rather than on the 0.9 s clock --
        reports a decode that ends on a run through note_speculative()
        (09-06), so a letter is seen by the pass the pause itself starts.
        The 60 s cap and the energy timer are untouched -- the energy timer
        (2.5 s of quiet) can end a held pause before the 2.3 s hold does;
        the ledger's stop= says which.
        """
        ep = self.endpointer
        if ep is None or not self.recording or not CONFIG.endpoint_vad:
            return False
        frames = self._audio_frames
        n = len(frames)
        if n > self._ep_cursor:
            fresh = frames[self._ep_cursor:n]
            self._ep_cursor = n
            try:
                ep.feed(np.concatenate(fresh, axis=0).flatten(), self._record_rate)
            except Exception:
                # WARNING, not debug: this silently returns every capture to
                # the 2.5 s energy timer for the rest of the session.
                log.warning("endpointer failed; energy timer only for this session",
                            exc_info=True)
                self.endpointer = None
                return False
        gap = ep.silence_since_speech
        if gap is None:
            window = getattr(self, "_followup_window", 0.0) or CONFIG.followup_window
            if self._followup and self._record_start_time and \
                    (time.monotonic() - self._record_start_time) >= window:
                log.info("follow-up: nothing said in %.1fs; closing quietly", window)
                self.abort()
                return True
            return False
        if gap < CONFIG.endpoint_silence:
            return False
        if self._record_start_time and \
                (time.monotonic() - self._record_start_time) < CONFIG.silence_grace:
            return False
        if ep.audio_seconds < 0.5:
            return False                      # by audio, not by frame count
        if (CONFIG.filler_hold or CONFIG.spell_hold) and \
                gap < CONFIG.endpoint_silence + self._hold_extra(ep, gap):
            return False
        held = ", ".join(
            filter(None, ("%d filler hold(s)" % self._filler_holds
                          if self._filler_holds else "",
                          "%d spelling hold(s)" % self._spell_holds
                          if self._spell_holds else "")))
        log.info("Auto-stop: %.2fs after the last word (vad%s)", gap,
                 ", " + held if held else "")
        self._voice_stopped = True
        try:
            self.stop(reason="silence", endpoint="vad", dead_air=gap)
        except Exception:
            log.exception("endpoint stop error")
            self.recording = False
        return True

    # -- the holds: filler and spelling -------------------------------------
    @property
    def capture_id(self) -> int:
        """Which capture is open. The preview reads this BEFORE it
        snapshots the buffer and hands it back to note_partial, so a decode
        that outlives its own capture can be told apart from this one's."""
        return self._capture_id

    def note_decoding(self, audio_end_s: float, capture_id) -> None:
        """A decode has just STARTED on a snapshot ending at ``audio_end_s``
        capture seconds (JarvisApp._partial_loop and _maybe_speculate,
        after the snapshot and before the model is called). The note that
        follows it -- note_partial or note_speculative, on the same
        capture -- clears it. While it stands and the stop is due,
        _hold_extra waits for it (DECODE_WAIT_MAX_S) when the snapshot
        covers the pause: a decode that holds the whole of the last burst
        is about to say whether it was a letter, and stopping 40 ms
        before it lands is how the first letter was lost. Same stamp rule
        as every other note: a capture that turned over drops it."""
        if capture_id != self._capture_id:
            return
        self._decoding = (float(audio_end_s), time.monotonic())

    def note_partial(self, text: str, audio_end_s: float,
                     capture_id) -> None:
        """The live preview's newest decode. Called by the app's partial
        loop after EVERY decode (not only a changed one -- a hold must
        survive whisper returning the same "…um" twice), with the capture
        position, in seconds, that the decoded span ends at. Only the
        trailing filler (jarvis.endpoint.trailing_filler) and the LENGTH
        of the trailing spelling run (jarvis.spelling.spelling_run) are
        kept; the recorder never sees or stores the words -- and a spelled
        local part is half an address, so the run is kept as a COUNT and
        logged as a count, never as the characters. Safe from any thread:
        one tuple assignment, read whole by _hold_extra.

        ``capture_id`` is the capture the AUDIO came from -- Recorder.
        capture_id read before the snapshot. A decode takes hundreds of ms,
        so the capture it started in can stop and the NEXT one open before
        it returns; that note carries the old capture's words and the old
        capture's position, and it held the new capture's first pause
        (MEASURED 09-05 at shipped defaults: `filler hold 1/3: 'um' at
        1.0s` on a capture with no filler in it). The round-1 span bounds
        cannot see it -- a SHORT previous capture leaves end_s inside both
        of them.

        Identity, NOT a flag: guarding on self.recording instead reads a
        different moment from the snapshot, and it would drop the note a
        decode makes after the stop -- which must still arrive, or a stale
        um keeps holding the mic (the partial loop's design item 4).
        The stamp is REQUIRED and there is no vouching path: a note is
        used only when its stamp equals the open capture's id, so an
        unstamped None is simply not equal and is dropped. A default that
        meant "trust me" is how this hole would be reopened by the next
        caller that forgets.
        """
        if capture_id != self._capture_id:
            return                    # a decode that outlived its capture
        self._decoding = None         # it landed: nothing to wait for
        self._latest_partial = (trailing_filler(text),
                                spelling_run(text, self._address_owed),
                                float(audio_end_s), time.monotonic())

    def note_speculative(self, text: str, audio_end_s: float,
                         capture_id) -> None:
        """The SPECULATIVE pass's decode (JarvisApp._maybe_speculate), for
        the spelling hold ONLY -- and only ever ADDING a hold, never
        clearing one.

        WHY IT REPORTS AT ALL (09-06). The greedy preview was the only
        thing feeding spelling_run, and the speculative pass parks it: a
        full decode from 0.3 s into every pause pushes the next greedy
        snapshot 0.9 s past its own end. Modelled on this recorder's real
        poll cadence with his log's speculative decode times (n=238, p50
        0.24 s, p90 0.46 s), the first spelled letter after "send an email
        to" survived 4 of 9 preview phases and the whole seven-character
        address 0-3 of 9 -- the pass was hiding the very letters the hold
        needed. The pass is also the RIGHT reporter: it starts because the
        pause started, so it lands at 0.3 s + its own latency into every
        gap between letters, on no clock; and if no more speech follows,
        its text is exactly what the commander will be handed.

        WHY ONLY ADDING. The filler hold deliberately does not hear this
        pass (its clean full decode is the one most likely to have dropped
        the um). The same asymmetry applies here in the other direction: a
        pass whose text does NOT end on a run says nothing, so it can never
        clear a hold the greedy preview set, and the greedy stays the one
        thing that can. The failure mode that leaves is a hold outliving
        its cause by one greedy interval -- a longer wait, never a lost
        letter. The stamp rule is note_partial's, unchanged.
        """
        if capture_id != self._capture_id:
            return                    # a decode that outlived its capture
        self._decoding = None         # it landed, whatever it said
        n = spelling_run(text, self._address_owed)
        if not n:
            return
        self._latest_partial = ("", n, float(audio_end_s), time.monotonic())

    def _probe_address_owed(self) -> bool:
        """Ask the app's probe whether an address is owed, for this capture.
        Guarded: a probe that raises is a probe that said no."""
        probe = getattr(self, "address_owed_probe", None)
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception:             # noqa: BLE001 - a hold is best effort
            log.debug("address_owed probe failed", exc_info=True)
            return False

    def _reset_filler_hold(self) -> None:
        """New capture: a new id, then no partial and no holds.

        The id moves FIRST, and the order is load-bearing: a note landing
        between the two statements must read the NEW id and be dropped.
        Clearing first would leave a window in which the old capture's id
        still matched and its note survived the reset."""
        self._capture_id += 1
        self._latest_partial = None
        self._decoding = None
        self._decode_wait_key = None
        self._filler_holds = 0
        self._filler_hold_key = None
        self._spell_holds = 0
        self._spell_hold_key = None

    def _hold_extra(self, ep, gap: float) -> float:
        """Seconds to add to endpoint_silence for the pause the VAD is in
        now (`gap` seconds old), from the newest preview decode:

        * CONFIG.filler_hold_s when it ended on a filler ("...um"), or
        * CONFIG.spell_hold_s when it ended on a SPELLING RUN
          (jarvis.spelling.spelling_run -- he is saying an address one
          character at a time and the pause between two characters is
          longer than the 0.8 s endpoint),

        in both cases only when the decoded span reached to within
        FILLER_SLACK_S of the last speech; else 0.

        ONE seam, deliberately, rather than a parallel one: the two holds
        share the span bounds, the once-per-pause key and the changes-the-
        outcome rule below, so a fix to any of that is a fix to both. They
        keep SEPARATE counters and caps -- an utterance has at most three
        ums but an address has sixteen characters, and the ledger's
        holds=N must go on meaning ums.

        The filler is checked first and wins a tie. A decode ending
        "...q-z-v, um" is a man who has stopped spelling to think, and
        1.5 s is the figure that was measured for thinking.

        A hold is counted ONCE per pause (keyed on the last-speech
        position), never per poll tick and never again for a later
        re-decode of the same pause; at most CONFIG.filler_max_holds /
        CONFIG.spell_max_holds pauses per capture, after which this
        returns 0 and the ordinary stop happens. Logs each hold once: the
        filler word and the wait, or -- because a spelled local part is
        half an address -- the character COUNT and the wait, never the
        characters.

        THE SPAN BOUNDS, from measurements on 09-05 and now shared by both
        holds, neither of which can stop a capture sooner than the branch
        already did. The span may not end more than FILLER_SLACK_S before
        the last speech (something was said after it that was never
        decoded), nor more than FILLER_SLACK_S past the audio the
        endpointer has been fed. The original guard was one-sided
        (`end_s < last - FILLER_SLACK_S`), so note_partial("um", 999.0) --
        a preview decode landing after a stop, carrying the OLD capture's
        position -- bought a hold in a pause it never covered. The slack on
        this side too: the preview snapshots the buffer between poll ticks
        and feed() consumes whole 512-sample chunks, so a legitimate end_s
        runs a fraction of a second ahead. The counting rule lives in
        _counted_hold.

        THE DECODE IN FLIGHT (09-06, DECODE_WAIT_MAX_S above): when the
        stop is due and a decode announced through note_decoding is still
        running on a snapshot that holds the whole of the last burst, the
        stop waits for it, bounded, rather than beating it by tens of
        milliseconds. MEASURED on this recorder's real poll cadence with
        the app's preview thread modelled as it runs (the greedy preview
        at 0.9 s, the speculative pass 0.3 s into every pause parking it,
        his log's speculative decode times, tests/test_spelling_survival.
        py): with the greedy preview as the only reporter, the first
        spelled letter after "send an email to" survived 4 of 9 preview
        phases and the whole seven-character address at his 0.99 s pace
        0 of 9; with the speculative pass reporting (note_speculative)
        but no wait, the whole address survived 9, 7, 9, 4, 8 and 4 of 9
        at greedy 0.2/0.4/0.6 s x pass p50/p90 -- the p90 pass landing
        40 ms after the stop; with the wait, 9 of 9 in every cell, and
        the same with the pass switched off by config (9, 8, 9 of 9). A
        first attempt at this presumed a short burst after a counted hold
        to be the next character without any decode (a burst-length
        rule); measured against the wait it bought nothing in any cell
        and could newly hold a short word said after a run, so it is not
        here. The filler hold is untouched by any of this."""
        latest = self._latest_partial
        last = ep.last_speech_seconds
        if last is None:
            return 0.0
        fresh = False
        if latest is not None:
            filler, spell_n, end_s, _wall = latest
            # The span may not end more than FILLER_SLACK_S before the last
            # speech (speech followed it: the tail was never decoded), nor
            # more than FILLER_SLACK_S past the audio this endpointer has
            # been fed (a partial from a capture it never heard).
            fresh = (end_s >= last - FILLER_SLACK_S
                     and end_s <= ep.audio_seconds + FILLER_SLACK_S)
        if fresh:
            if filler and CONFIG.filler_hold:
                return self._counted_hold(
                    gap, last, float(CONFIG.filler_hold_s),
                    "_filler_holds", "_filler_hold_key", CONFIG.filler_max_holds,
                    "filler hold %d/%d: %r at %.1fs, waiting %.1fs",
                    (filler, last, float(CONFIG.filler_hold_s)))
            if spell_n and CONFIG.spell_hold:
                return self._counted_hold(
                    gap, last, float(CONFIG.spell_hold_s),
                    "_spell_holds", "_spell_hold_key", CONFIG.spell_max_holds,
                    "spelling hold %d/%d: %d characters at %.1fs, waiting %.1fs",
                    (spell_n, last, float(CONFIG.spell_hold_s)))
        # A decode is STILL RUNNING on a snapshot that holds the whole of
        # the burst that just ended: it is about to say, so the stop waits
        # for it (bounded). Before the fresh-word verdict below on purpose:
        # a note landing before this decode started is older than it.
        wait = self._decode_wait(ep, last)
        if wait:
            return wait
        # Either a decode covers this pause and ends on a WORD -- the run,
        # if there was one, is over -- or no decode covers it and none is
        # running on it: the ordinary stop.
        return 0.0

    def _decode_wait(self, ep, last: float) -> float:
        """DECODE_WAIT_MAX_S while a decode noted by note_decoding is in
        flight for this capture and its snapshot reaches the last speech
        (within DECODE_COVER_SLACK_S), else 0. Only with the spelling hold
        on: it exists for the letter the stop would otherwise beat, and
        with the hold off the recorder stops exactly as before. Not a
        hold and not counted as one -- the ledger's holds mean what they
        meant -- but logged once per pause at DEBUG, so a week of turns
        can say how often the stop waited on a decode."""
        if not CONFIG.spell_hold:
            return 0.0
        dec = self._decoding
        if dec is None:
            return 0.0
        end_s, _wall = dec
        if end_s < last - DECODE_COVER_SLACK_S:
            return 0.0            # snapshotted before the burst ended: it cannot say
        if end_s > ep.audio_seconds + FILLER_SLACK_S:
            return 0.0            # from audio this endpointer never heard
        if self._decode_wait_key != last:
            self._decode_wait_key = last
            log.debug("stop waits on a decode in flight (snapshot %.2fs, last speech %.2fs)",
                      end_s, last)
        return float(DECODE_WAIT_MAX_S)

    def _counted_hold(self, gap: float, last: float, extra: float,
                      count_attr: str, key_attr: str, cap: int,
                      msg: str, args: tuple) -> float:
        """The half of a hold that is the same for both kinds: count it
        once per pause, refuse past the cap, refuse when it would delay
        nothing, and log it once.

        A hold is counted and logged only when it CHANGES the outcome. One
        starved poll tick arriving with the gap already past
        endpoint_silence + extra stops on that tick; incrementing there
        made RecordingStopped.filler_holds -- and the ledger's holds=N,
        the number the design added so a week of turns could say how often
        the hold fired -- an upper bound rather than a count of holds that
        DELAYED a stop. A hold already counted for this pause stays
        counted on the tick it expires."""
        if getattr(self, key_attr) != last:
            if getattr(self, count_attr) >= cap:
                return 0.0
            if gap >= CONFIG.endpoint_silence + extra:
                return 0.0        # already past the hold: it would delay nothing
            setattr(self, count_attr, getattr(self, count_attr) + 1)
            setattr(self, key_attr, last)
            log.info(msg, getattr(self, count_attr), cap, *args)
        return extra

    def _check_silence(self) -> bool:
        """Port of monolith 4301-4344. Returns True when it stopped the
        session (the poll loop then exits)."""
        if not self.recording:
            return True

        # Hard cap on recording length
        if self._record_start_time:
            elapsed = time.monotonic() - self._record_start_time
            if elapsed >= MAX_RECORDING_SECONDS:
                log.info("Max recording time reached (%ss)", MAX_RECORDING_SECONDS)
                self._voice_stopped = True
                try:
                    self.stop(reason="cap")
                except Exception:
                    log.exception("Max recording stop error")
                    self.recording = False
                return True

        timeout = CONFIG.silence_timeout
        min_frames = int(SAMPLE_RATE * 0.5 / (SAMPLE_RATE * 0.1))
        # Grace period: no auto-stop this soon after the session opened, so a
        # slow start ("Jarvis... uh...") is not clipped. Was a hard-coded 5.0,
        # which together with an 8 s timeout put a ~13 s floor under every
        # utterance no matter how short.
        grace = CONFIG.silence_grace
        if self._record_start_time and \
                (time.monotonic() - self._record_start_time) < grace:
            self._silence_start = None
            self._loud_chunks = 0
            return False
        # Snapshot: the audio callback sets _silence_start back to None on
        # two loud chunks, from its own thread, and a re-read after the test
        # raised TypeError here -- which killed the poll thread.
        start = self._silence_start
        if (start is not None
                and (time.monotonic() - start) >= timeout
                and len(self._audio_frames) > min_frames):
            ep = self.endpointer
            if ep is not None and CONFIG.endpoint_vad:
                # The VAD still hears the user: stand down. Quiet speech reads
                # as silence here -- on a real capture the user's RMS ran at
                # 0.008-0.014 against a 0.015 threshold for a whole clause
                # while the VAD held 0.97-1.00 -- so this timer was ending
                # captures mid-sentence. The VAD stops them itself,
                # endpoint_silence after the last word; the 60 s cap remains.
                gap = ep.silence_since_speech
                if gap is not None and gap < CONFIG.endpoint_silence:
                    return False
            log.info("Auto-stop on silence (%ss timeout)", timeout)
            if self.endpointer is not None and CONFIG.endpoint_vad:
                # The VAD was live and the energy timer still won: say why.
                try:
                    log.info("energy beat the %s", self.endpointer.describe())
                except Exception:
                    log.debug("vad describe failed", exc_info=True)
            self._voice_stopped = True
            try:
                self.stop(reason="silence", endpoint="energy",
                          dead_air=time.monotonic() - start)
            except Exception:
                log.exception("Auto-stop error")
                self.recording = False
                bus.publish(Status(text="Auto-stop error", kind="error"))
            return True

        return False

    def _check_speaker_silence(self):
        """Voice-aware silence: auto-stop when the user hasn't spoken for the
        timeout, even if background audio (TV/YouTube) keeps making noise.
        Port of monolith 4346-4390; verify runs on a helper thread exactly as
        before so the poll loop keeps publishing levels."""
        if not self.recording:
            return
        verifier = self.speaker_verifier
        if not CONFIG.speaker_verify or verifier is None:
            return   # Voice ID not active — regular silence detection handles it
        try:
            if not verifier.enrolled:
                return
        except Exception:
            log.exception("speaker verifier enrolled-check failed")
            return
        if getattr(verifier, "_model_failed", False):
            return          # a dead model is not worth asking once a second

        # One check in flight at a time. At 1 Hz a slow verify could otherwise
        # stack threads that all mutate the miss counters and may all call
        # stop(); a check that is late is simply skipped, the next poll asks.
        lock = getattr(self, "_speaker_check_lock", None)
        if lock is None:
            lock = self._speaker_check_lock = threading.Lock()
        if not lock.acquire(blocking=False):
            return

        try:
            self._start_speaker_check(lock, verifier)
        except BaseException:
            # Anything raised between acquire and the thread's own finally
            # (concatenate, resample) would otherwise hold the lock for the
            # life of the Recorder and silence every later voice-ID check.
            lock.release()
            raise

    def _start_speaker_check(self, lock, verifier):
        now = time.monotonic()
        if (self._record_start_time
                and (now - self._record_start_time) < self._SPEAKER_WARMUP_S):
            lock.release()
            return

        rate = self._record_rate
        samples_needed = int(rate * self._SPEAKER_WINDOW_S)
        frames = list(self._audio_frames)
        if not frames:
            lock.release()
            return

        # Walk back until the window is filled rather than assuming a frame
        # size: blocksize is 0.1 s today, but the window is now a constant and
        # this must not silently judge the wrong span if either ever changes.
        tail, got = [], 0
        for chunk in reversed(frames):
            tail.append(chunk)
            got += len(chunk)
            if got >= samples_needed:
                break
        recent = np.concatenate(list(reversed(tail)), axis=0).flatten()[-samples_needed:]
        # Resample to 16kHz for speaker check
        audio_16k = self._resample_to_16k(recent)

        def _check():
            try:
                try:
                    is_match, score = verifier.verify(audio_16k)
                except Exception:
                    log.warning("speaker silence verify failed", exc_info=True)
                    return
                self._on_speaker_silence_result(is_match, score)
            finally:
                lock.release()

        threading.Thread(target=_check, daemon=True).start()

    def _on_speaker_silence_result(self, is_match, score):
        """Handle result of periodic speaker check during recording.
        Port of monolith 4392-4441 (thresholds and miss-count verbatim)."""
        if not self.recording:
            return

        # Use a relaxed threshold for silence detection — we don't want to
        # cut the user off mid-sentence on a borderline score. The strict
        # threshold is applied later by the segment filter.
        relaxed_threshold = CONFIG.speaker_threshold * 0.70
        is_probably_user = score >= relaxed_threshold
        self._voice_id_match = is_probably_user

        if is_probably_user:
            # User is likely speaking — reset speaker silence timer
            if self._speaker_silence_start is not None:
                log.info("Voice-ID silence reset (score=%.3f >= relaxed %.3f)",
                         score, relaxed_threshold)
            self._speaker_silence_start = None
            self._speaker_silence_misses = 0
        else:
            # Probably not the user — count consecutive misses
            self._speaker_silence_misses += 1

            # Require 2 consecutive misses before starting silence timer
            # (one borderline check shouldn't trigger a stop)
            if self._speaker_silence_misses < 2:
                log.info("Voice-ID miss 1/2 (score=%.3f), waiting for confirmation",
                         score)
                return

            if self._speaker_silence_start is None:
                self._speaker_silence_start = time.monotonic()
                log.info("Voice-ID silence started (score=%.3f, 2 consecutive misses)",
                         score)

            elapsed = time.monotonic() - self._speaker_silence_start
            timeout = CONFIG.silence_timeout
            if elapsed >= timeout and len(self._audio_frames) > 5:
                log.info("Voice-ID auto-stop: no user voice for %.1fs "
                         "(last score=%.3f)", elapsed, score)
                self._voice_stopped = True
                try:
                    self.stop(reason="silence", endpoint="voice_id", dead_air=elapsed)
                except Exception:
                    log.exception("Voice-ID auto-stop error")
                    self.recording = False

    # -- fixed-duration recording (port of monolith 5000-5054) -----------
    def record_fixed(self, seconds: float) -> np.ndarray:
        """Blocking capture of `seconds` of audio at the mic's native rate,
        resampled to 16k mono float32. Used for enrollment / wake-word
        training. Returns an empty array on failure. Runs under the arbiter
        (hotword paused for the duration)."""
        if not self.mic_available:
            bus.publish(Status(text="No microphone detected", kind="warn"))
            return np.zeros(0, dtype=np.float32)
        if self.recording:
            bus.publish(Status(text="Stop recording first", kind="warn"))
            return np.zeros(0, dtype=np.float32)

        try:
            import sounddevice as sd
        except Exception as e:
            log.exception("sounddevice unavailable")
            bus.publish(Status(text=f"Audio backend error: {e}", kind="error"))
            return np.zeros(0, dtype=np.float32)

        with self._arbiter.acquire("record_fixed"):
            mic_idx = self._resolve_mic()
            native_rate = self._native_rate(mic_idx)

            frames: list = []

            def cb(indata, frame_count, time_info, status):
                chunk = indata[:, 0].copy() if indata.ndim > 1 \
                    else indata.flatten().copy()
                frames.append(chunk)

            try:
                stream = sd.InputStream(
                    samplerate=native_rate, channels=CHANNELS,
                    dtype="float32", device=mic_idx,
                    callback=cb, blocksize=int(native_rate * 0.1),
                )
                stream.start()
                time.sleep(float(seconds))
                stream.stop()
                stream.close()
            except Exception as e:
                log.exception("Fixed recording error")
                bus.publish(Status(text=f"Recording failed: {str(e)[:50]}",
                                   kind="error"))
                return np.zeros(0, dtype=np.float32)

        if not frames:
            bus.publish(Status(text="No audio captured", kind="warn"))
            return np.zeros(0, dtype=np.float32)

        audio_raw = np.concatenate(frames).flatten()
        # Resample to 16kHz (verbatim math, monolith 5048-5053)
        if native_rate != SAMPLE_RATE:
            from scipy.signal import resample
            new_len = int(len(audio_raw) * SAMPLE_RATE / native_rate)
            audio = resample(audio_raw, new_len).astype(np.float32)
        else:
            audio = audio_raw
        return audio

    # -- noise calibration (port of monolith 2262-2338) -------------------
    def calibrate_noise(self) -> float:
        """Blocking: sample background noise for 3 seconds and set the
        threshold above it. Saves to CONFIG.noise_threshold and returns the
        new value (current value on failure). Hotword paused via arbiter."""
        current = CONFIG.noise_threshold
        if self.recording:
            bus.publish(Status(text="Stop recording first", kind="warn"))
            return current
        if not self.mic_available:
            bus.publish(Status(text="No microphone detected", kind="warn"))
            return current

        try:
            import sounddevice as sd
        except Exception as e:
            log.exception("sounddevice unavailable")
            bus.publish(Status(text=f"Audio backend error: {e}", kind="error"))
            return current

        bus.publish(Status(text="Calibrating — be quiet for 3 seconds",
                           kind="busy"))

        with self._arbiter.acquire("calibrate"):
            mic_idx = self._resolve_mic()
            native_rate = self._native_rate(mic_idx)

            samples: list = []

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
                log.exception("Calibration error")
                bus.publish(Status(text=f"Calibration failed: {e}", kind="error"))
                return current

        if not samples:
            bus.publish(Status(text="No audio captured", kind="error"))
            return current

        arr = np.array(samples)
        p99 = float(np.percentile(arr, 99))
        # Set threshold 50% above the 99th percentile of background noise
        new_threshold = round(p99 * 1.5, 4)
        # Clamp to reasonable range
        new_threshold = max(0.01, min(0.15, new_threshold))

        log.info("Calibration: mean=%.4f p99=%.4f -> threshold=%s",
                 arr.mean(), p99, new_threshold)

        CONFIG.update(noise_threshold=new_threshold)
        bus.publish(Status(
            text=f"Noise floor: {p99:.4f} → threshold: {new_threshold}",
            kind="ok"))
        return new_threshold

    # -- misc state ------------------------------------------------------
    @property
    def voice_id_match(self) -> bool:
        """Latest periodic speaker-check verdict (UI/reactor indicator)."""
        return self._voice_id_match

    @property
    def audio_level(self) -> float:
        return self._audio_level

    @property
    def voice_stopped(self) -> bool:
        """True when the last stop was manual/auto (skip continuous restart)."""
        return self._voice_stopped
