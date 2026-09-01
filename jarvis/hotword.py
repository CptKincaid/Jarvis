"""OpenWakeWord hotword listener for Jarvis V3.

Ported from voice_input_gui.py HotwordListener (lines 785-963) with the
openwakeword 0.4.0 custom-verifier patch preserved verbatim (841-858), plus
the wake-word verifier training flow (5178-5260) as train_verifier().

No GUI references: the constructor takes the MicArbiter, a mic-index getter,
and an on_detect callback. pause()/resume() are registered with the arbiter so
every mic consumer (recording, calibration, enrollment, training, TTS
talk-back) pauses the hotword stream via ``arbiter.acquire(owner)``.

Audit C fix: the injected verifier is wrapped in ``VerifierShim`` and its
feature size checked at load (``install_verifier``). Found with synthetic
"Hey Jarvis" clips: oww 0.4.0's predict loop hands the hey_jarvis verifier
the timer/weather models' feature windows too and sklearn raised
"X has 3264 features, but StandardScaler is expecting 1536" — only on a
genuine wake word, which the listen loop then swallowed, so the hotword
could never fire while ~/.aiws_trainer/hey_jarvis_verifier.pkl existed.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from jarvis.config import CONFIG
from jarvis.events import HotwordDetected, Status, bus
from jarvis.logs import get_logger

log = get_logger("hotword")

CHANNELS = 1  # port: voice_input_gui.py:54
OWW_EMBEDDING_DIM = 96        # openwakeword feature windows are (frames, 96)
VERIFIER_THRESHOLD = 0.3      # port: voice_input_gui.py:854


def verifier_feature_count(verifier):
    """Flattened feature count a trained verifier expects (sklearn
    Pipeline: from its StandardScaler / final estimator), else None."""
    steps = getattr(verifier, "steps", None) or []
    for _, step in reversed(list(steps)):
        n = getattr(step, "n_features_in_", None)
        if n:
            return int(n)
    n = getattr(verifier, "n_features_in_", None)
    return int(n) if n else None


class VerifierShim:
    """Guards a custom verifier against the openwakeword 0.4.0 predict loop.

    ``Model.predict`` re-applies every custom verifier once per LOADED
    model, passing ``get_features(self.model_inputs[mdl])`` — the frame
    count of the model being iterated, not the verifier's own. With the
    default set (alexa/hey_mycroft/hey_jarvis/timer/weather) the hey_jarvis
    verifier (16 x 96 = 1536 features) is also handed the timer (34-frame)
    and weather (22-frame) windows and raises. The shim answers mis-sized
    calls with the last valid probability (0.0 before any valid call —
    never inflating a score) and delegates right-sized calls."""

    def __init__(self, inner, n_features: int):
        self.inner = inner
        self.n_features = int(n_features)
        self.calls = 0
        self.skipped = 0
        self._last = np.array([[1.0, 0.0]])

    def predict_proba(self, X):
        arr = np.asarray(X)
        per_sample = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else arr.size
        if per_sample != self.n_features:
            self.skipped += 1
            return self._last
        self.calls += 1
        self._last = self.inner.predict_proba(X)
        return self._last


def install_verifier(model, verifier, name: str = "hey_jarvis",
                     threshold: float = VERIFIER_THRESHOLD):
    """Inject a trained verifier into an oww Model after construction (the
    0.4.0 patch), guarded by ``VerifierShim`` and a feature-size check.
    Returns (ok, detail); on a mismatch nothing is installed."""
    frames = (getattr(model, "model_inputs", None) or {}).get(name)
    if not frames:
        return False, f"base model {name!r} not loaded"
    expected = int(frames) * OWW_EMBEDDING_DIM
    got = verifier_feature_count(verifier)
    if got is not None and got != expected:
        return False, (f"verifier expects {got} features but {name} yields "
                       f"{expected} ({frames} x {OWW_EMBEDDING_DIM}); retrain "
                       f"it (Settings > Voice ID > Train wake word)")
    model.custom_verifier_models[name] = VerifierShim(verifier, expected)
    model.custom_verifier_threshold = threshold
    return True, f"{got or expected} features, threshold {threshold}"


def wake_hit(predictions, threshold, unverified_threshold):
    """Is this frame a wake hit, and at what score?

    `hey_jarvis` is vetted by the trained custom verifier. `hey_mycroft` is
    NOT: openWakeWord gates verifiers on the parent model name
    (``custom_verifier_models.get(parent_model)``) and ours is registered under
    hey_jarvis only, so the mycroft head reaches the comparison with nothing
    vetting it -- a raw 0.4286 used to be enough. It keeps its own, stricter
    bar so the unvetted path cannot fire as easily as the vetted one.
    """
    jarvis = float(predictions.get("hey_jarvis", 0.0) or 0.0)
    mycroft = float(predictions.get("hey_mycroft", 0.0) or 0.0) * 0.7
    hit = jarvis >= threshold or mycroft >= unverified_threshold
    return hit, max(jarvis, mycroft)


# How much buffered audio the speaker gate wants before it will judge a wake.
# Below speaker.MIN_AUDIO_SECONDS it cannot score at all -- _extract_embedding
# returns None, score() returns None, and _speaker_ok fails OPEN. 1.5 s, not
# the verifier's 1.0: the wake buffer is scored on whatever speech it holds,
# and the 2026-08-31 simulation put false rejects at 19.6% with a 1.0 s floor
# against 6.2% at 1.5 s (+0.6pp false accepts).
#
# This is a preference, NOT a veto. It was written as one on 2026-08-28 after
# a wake fired on 0.349 s of audio at 16:41:32.959 -- but that wake was not a
# person. It was openWakeWord re-firing the PREVIOUS wake word out of feature
# state a pause does not clear (see reset_oww_stream), which is why it landed
# at 0.349 s: the arithmetic, not a speaker. Vetoing short wakes suppressed
# that symptom and cost real ones instead -- 52 logged "wake held" lines, none
# of which ever recovered.
WAKE_MIN_AUDIO_SECONDS = 1.5

# How long a detected wake may be HELD BACK waiting for the ring buffer to
# reach WAKE_MIN_AUDIO_SECONDS, so the speaker gate scores real audio instead
# of abstaining. Only ever paid when the shortfall fits inside it: a wake on a
# freshly cleared buffer is admitted at once. Holding one indefinitely -- which
# is what the 2026-08-28 `continue` did -- is a fail-SHUT decision in the one
# layer CLAUDE.md documents as fail-OPEN, and openWakeWord's activation decays
# within a few 80 ms frames, so a wake not acted on quickly is simply lost.
WAKE_HOLD_MAX_SECONDS = 0.6


def wake_audio_sufficient(n_samples: int, native_rate: int,
                          min_seconds: float = WAKE_MIN_AUDIO_SECONDS) -> bool:
    """Is there enough buffered audio for the speaker gate to judge a wake?"""
    return n_samples >= int(native_rate * min_seconds)


def wake_hold_seconds(n_samples: int, native_rate: int,
                      min_seconds: float = WAKE_MIN_AUDIO_SECONDS,
                      max_wait: float = WAKE_HOLD_MAX_SECONDS) -> float:
    """How long to wait for the buffer to fill, or 0.0 to decide NOW.

    The ring buffer fills in real time, so the wait is exactly the shortfall.
    Waiting is worth it only when that shortfall fits inside `max_wait`, which
    is set to cover the near-misses actually seen in jarvis.log -- the 0.88 s
    and 0.96 s holds, the only two of 52 that were a genuinely part-filled
    buffer rather than a freshly cleared one. (The 0.96 s hold at 15:44:11
    completed 107 ms later against the 1.0 s floor of the day and scored
    -0.053, a real verdict on a real voice.) Past `max_wait` the wake is
    admitted immediately with the speaker gate abstaining: an unwakeable
    assistant is worse than an over-eager one, and the transcript gate
    (app._process_audio -> speaker.filter_segments) still fails SHUT behind it.
    """
    shortfall = int(native_rate * min_seconds) - int(n_samples)
    if shortfall <= 0:
        return 0.0                       # already enough: decide now
    wait = shortfall / float(native_rate)
    return wait if wait <= max_wait else 0.0    # too far short: admit now


# ----------------------------------------------------------------------
# openWakeWord stream state
# ----------------------------------------------------------------------
def capture_oww_blank(model):
    """Snapshot an oww model's pristine, silence-filled feature state.

    Taken once, straight after ``Model()`` and before any microphone audio has
    reached it, so it is exactly what a cold start scores against
    (``AudioFeatures.__init__`` seeds the buffer with embeddings OF SILENCE,
    not zeros -- copying it is the only cheap way to get them back).
    Returns None when the model has no 0.4.0-shaped preprocessor, in which
    case ``reset_oww_stream`` degrades to a plain ``Model.reset()``.
    """
    pre = getattr(model, "preprocessor", None)
    if pre is None:
        return None
    try:
        return {
            "feature_buffer": np.array(pre.feature_buffer, copy=True),
            "melspectrogram_buffer": np.array(
                pre.melspectrogram_buffer, copy=True),
        }
    except Exception:
        log.exception("could not snapshot openWakeWord feature state")
        return None


def reset_oww_stream(model, blank=None) -> bool:
    """Forget every trace of PRE-PAUSE audio, not just the score history.

    ``Model.reset()`` clears ONLY ``prediction_buffer`` (openwakeword 0.4.0
    model.py:152-154). The audio itself lives in ``preprocessor.feature_buffer``
    (120 frames, ~10 s) and ``melspectrogram_buffer``, and neither is touched
    by a pause: the stream closes and our ring buffer is cleared, but oww's own
    window is not. Four fresh 80 ms frames after the resume, the 16-frame
    window `hey_jarvis` scores is 12 frames of pre-pause audio plus 4 new ones
    -- the wake word the user said BEFORE the pause -- and it re-fires at 0.98
    with nobody having spoken.

    Measured 2026-08-31 by replaying his own ~/.aiws_trainer/wakeword_training
    clips through the real model: 10 of 30 pause/resume replays re-fired, every
    one at +0.32 s, which is the exact figure in 47 of the 52 "wake held" lines
    in jarvis.log. Returns True when the audio state was really cleared.
    """
    try:
        model.reset()
    except Exception:
        log.exception("openWakeWord reset failed")
    pre = getattr(model, "preprocessor", None)
    if pre is None or not blank:
        return False
    try:
        pre.feature_buffer = np.array(blank["feature_buffer"], copy=True)
        pre.melspectrogram_buffer = np.array(
            blank["melspectrogram_buffer"], copy=True)
        # The melspectrogram is computed from the tail of raw_data_buffer, so
        # stale samples there would bleed three frames of pre-pause audio back
        # into the very first post-resume feature.
        pre.raw_data_buffer.clear()
        pre.accumulated_samples = 0
    except Exception:
        log.exception("could not clear openWakeWord audio state")
        return False
    return True


def frames_agree(history, window, required):
    """True when `required` of the last `window` frames were hits.

    A single 80 ms frame over threshold used to fire the wake word, and one
    transient frame is exactly what a burst of television audio produces.
    """
    recent = list(history)[-window:]
    return sum(1 for h in recent if h) >= required


class Hotword:
    """Always-on wake word listener using OpenWakeWord (CPU, ~1.5ms/prediction).

    Much more reliable than the old Whisper-based approach which ran full
    transcription every 0.8s and often missed short wake words in noise.
    """

    THRESHOLD = 0.3  # Wake word confidence threshold (lowered for "Jarvis" without "Hey")
    UNVERIFIED_THRESHOLD = 0.6   # hey_mycroft head -- the verifier does not cover it
    FRAME_WINDOW = 3             # frames considered for agreement
    FRAMES_REQUIRED = 2          # ...of which this many must be hits
    SPEAKER_WAKE_MIN = 0.25      # deliberately below the transcript threshold:
                                 # wake audio is short and partial, so it scores
                                 # lower than a full utterance. Background voices
                                 # measured near 0.0, leaving ample room.

    def __init__(self, arbiter, get_mic_index: Callable, on_detect: Callable,
                 speaker=None, on_guest: Callable = None):
        """arbiter: jarvis.recorder.MicArbiter (or None for standalone use).
        get_mic_index: () -> int | None (sounddevice input device index).
        on_detect: (score: float) -> None, called from the listener thread.
        """
        self._on_guest = on_guest
        self._arbiter = arbiter
        self._get_mic_index = get_mic_index
        self._on_detect = on_detect
        self._speaker = speaker      # SpeakerVerifier; None = no gating
        self.active = False
        self._stream = None
        self._model = None
        self._reopen = False
        self._paused = False
        self._predict_failures = 0
        # Pristine oww feature state, snapshotted once the model exists; see
        # reset_oww_stream for why Model.reset() alone is not enough.
        self._oww_blank = None
        if arbiter is not None:
            try:
                arbiter.register_hotword(self.pause, self.resume)
            except Exception:
                log.exception("could not register with MicArbiter")

    # -- speaker gate --------------------------------------------------
    def _speaker_ok(self, audio, native_rate) -> bool:
        """Is the buffered utterance the enrolled speaker?

        Fails OPEN, which is the OPPOSITE of the transcript gate in
        jarvis/app.py -- and deliberately so. A wake word that cannot be
        triggered is worse than one that triggers too often, and the
        transcript gate still fails shut behind this. Each layer fails the
        safe way for its own position.
        """
        speaker = getattr(self, "_speaker", None)
        if speaker is None or not getattr(speaker, "is_enrolled", False):
            return True
        if not CONFIG.speaker_verify:
            return True          # one switch governs all speaker gating
        try:
            if native_rate != 16000:
                from scipy.signal import resample
                audio = resample(
                    audio, int(len(audio) * 16000 / native_rate)).astype("float32")
            score = speaker.score(audio)
        except Exception:
            log.exception("wake speaker check failed -- waking anyway")
            return True
        if score is None:
            log.warning("wake speaker check unavailable -- waking anyway")
            return True
        if score < self.SPEAKER_WAKE_MIN:
            log.info("wake suppressed: speaker score %.3f < %.2f",
                     score, self.SPEAKER_WAKE_MIN)
            return False
        return True

    def _warm_speaker(self):
        """The first ECAPA call costs ~1.4 s (CUDA warm-up), every one after
        ~10 ms. Pay it at listener start, not on the first real wake word."""
        speaker = getattr(self, "_speaker", None)
        if speaker is None or not getattr(speaker, "is_enrolled", False):
            return
        try:
            speaker.score(np.zeros(16000, dtype=np.float32))
        except Exception:
            log.exception("speaker warm-up failed (harmless)")

    # -- lifecycle -----------------------------------------------------
    def start(self):
        if self.active:
            return
        self.active = True
        self._paused = False
        threading.Thread(target=self._listen_loop, daemon=True).start()
        log.info("Hotword listener started")

    def stop(self):
        self.active = False
        self._close_stream()
        log.info("Hotword listener stopped")

    def pause(self):
        """Release the mic stream so recording can use it."""
        self._paused = True
        self._close_stream()
        log.info("Hotword stream paused (mic released)")

    def resume(self):
        """Re-open the mic stream after recording finishes."""
        self._paused = False
        if self._stream:
            return
        if self.active:
            self._reopen = True
            log.info("Hotword stream will resume")
        else:
            self.start()
            log.info("Hotword listener restarted fresh")

    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # -- detection loop (port: 838-953) --------------------------------
    def _listen_loop(self):
        import sounddevice as sd

        # Load OpenWakeWord model (CPU, tiny) with custom verifier if available
        if self._model is None:
            try:
                from openwakeword.model import Model
                # oww 0.4.0's custom_verifier_models kwarg expects plain-pickle
                # file paths and mis-matches keys mid-init; inject the joblib
                # object after construction instead.
                self._model = Model()
                verifier_path = Path.home() / ".aiws_trainer" / "hey_jarvis_verifier.pkl"
                if verifier_path.exists():
                    # A bad verifier must never take the base model down.
                    try:
                        import joblib
                        verifier = joblib.load(str(verifier_path))
                        ok, detail = install_verifier(self._model, verifier)
                    except Exception:
                        log.exception("custom verifier load failed")
                        ok, detail = False, "unreadable pickle"
                    if ok:
                        log.info("Custom wake word verifier loaded: %s (%s)",
                                 verifier_path, detail)
                    else:
                        log.warning("Custom wake word verifier NOT used: %s",
                                    detail)
                        bus.publish(Status(
                            text="Wake-word verifier unusable — retrain it",
                            kind="warn"))
                log.info("OpenWakeWord loaded: %s", list(self._model.models.keys()))
            except Exception:
                log.exception("OpenWakeWord load error")
                self.active = False
                return

        # Snapshot the model's silence-filled feature state while it is
        # still pristine -- before the stream opens, so not one microphone
        # sample has reached it. Only once: a later capture would preserve
        # whatever audio was in flight rather than silence.
        if self._oww_blank is None:
            self._oww_blank = capture_oww_blank(self._model)

        self._reopen = False
        mic_idx = self._get_mic_index()

        # Detect native sample rate
        try:
            dev_info = sd.query_devices(mic_idx, 'input')
            native_rate = int(dev_info['default_samplerate'])
        except Exception:
            native_rate = 44100
        log.info("Hotword mic rate: %sHz", native_rate)

        # OpenWakeWord needs 16kHz int16 chunks of 1280 samples (80ms)
        # We'll collect audio in a buffer and resample
        chunk_samples = int(native_rate * 0.08)  # 80ms at native rate

        buf = deque(maxlen=int(native_rate * 2))  # 2s rolling buffer
        recent = deque(maxlen=self.FRAME_WINDOW)  # per-frame hit history
        self._warm_speaker()

        def callback(indata, frame_count, time_info, status):
            if self.active and not self._paused:
                chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                buf.extend(chunk.tolist())

        def _open_stream():
            try:
                self._stream = sd.InputStream(
                    samplerate=native_rate, channels=CHANNELS,
                    dtype="float32", device=mic_idx,
                    callback=callback,
                    blocksize=chunk_samples,
                )
                self._stream.start()
                return True
            except Exception:
                log.exception("Hotword stream error")
                self._stream = None
                return False

        def _fire(utterance, score):
            """Act on a wake the loop has accepted, then re-arm.

            Clears the ring buffer BEFORE the debounce so it refills during
            it, and the oww stream AFTER, because the gap itself is what
            makes oww's window stale (see reset_oww_stream)."""
            buf.clear()
            recent.clear()
            if not self._speaker_ok(utterance, native_rate):
                log.info("Hotword suppressed (score=%.3f): not the enrolled "
                         "speaker", score)
                if self._on_guest is not None:
                    try:
                        self._on_guest(float(score))   # a guest, politely
                    except Exception:
                        log.exception("on_guest callback failed")
                time.sleep(0.5)      # shorter than a real wake's debounce
            else:
                log.info("Hotword detected (score=%.3f)", score)
                bus.publish(HotwordDetected(score=float(score)))
                try:
                    self._on_detect(float(score))
                except Exception:
                    log.exception("on_detect callback failed")
                time.sleep(1.5)  # Debounce
            reset_oww_stream(self._model, self._oww_blank)
            recent.clear()

        if not _open_stream():
            self.active = False
            return

        # A wake whose buffer is nearly long enough to verify, remembered
        # as (deadline, score) while it fills. Never a way to DROP a wake:
        # every pending one is fired below, on its audio or on its deadline.
        pending = None

        while self.active:
            time.sleep(0.08)  # Check every 80ms (matches OWW chunk size)

            # Re-open stream after recording finishes
            if self._reopen and not self._paused and not self._stream:
                self._reopen = False
                buf.clear()
                pending = None          # a wake held across a pause is stale
                _open_stream()
                log.info("Hotword stream resumed")
                # NOT self._model.reset(): that clears the score history only,
                # and oww's ~10 s of audio features survive the pause and
                # re-fire the pre-pause wake word 0.32 s from now.
                if self._model:
                    reset_oww_stream(self._model, self._oww_blank)

            if self._paused or not self._stream:
                continue

            # Honour a held wake the moment its audio is enough OR its grace
            # runs out -- BEFORE the next predict, because openWakeWord's
            # activation decays within a few frames and the old `continue`
            # simply lost it (0 of 50 held wakes in jarvis.log ever recovered
            # within a second; the median gap to the next detection was 46 s).
            if pending is not None:
                deadline, held_score = pending
                if (wake_audio_sufficient(len(buf), native_rate)
                        or time.monotonic() >= deadline):
                    pending = None
                    held = np.array(buf, dtype=np.float32)
                    log.info("wake released on %.2f s of audio (score=%.3f)",
                             len(held) / native_rate, held_score)
                    _fire(held, held_score)
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

            # Predict — ~1.5ms on CPU (measured 3.6-4.9 ms/chunk on GB10)
            try:
                predictions = self._model.predict(audio_int16)
            except Exception:
                # Never silent: a failing predict used to be swallowed here.
                self._predict_failures += 1
                if self._predict_failures <= 3:
                    log.exception("hotword predict failed (%d)",
                                  self._predict_failures)
                if (self._predict_failures == 3
                        and self._model.custom_verifier_models):
                    log.warning("dropping custom wake-word verifier after "
                                "repeated predict failures")
                    self._model.custom_verifier_models = {}
                    bus.publish(Status(
                        text="Wake-word verifier disabled after errors",
                        kind="warn"))
                continue

            # hey_jarvis is verifier-vetted; hey_mycroft is not, so it
            # carries its own stricter bar (see wake_hit).
            hit, score = wake_hit(predictions, self.THRESHOLD,
                                  self.UNVERIFIED_THRESHOLD)
            recent.append(hit)
            if not frames_agree(recent, self.FRAME_WINDOW,
                                self.FRAMES_REQUIRED):
                continue

            # Snapshot the ring buffer BEFORE clearing it -- it holds the
            # utterance that fired, which is what the speaker gate judges.
            utterance = np.array(buf, dtype=np.float32)
            wait = wake_hold_seconds(len(utterance), native_rate)
            if wait > 0.0:
                # Nearly enough: wait the shortfall out so the gate gets a
                # real score instead of abstaining. The wake is remembered,
                # not dropped, and `recent` is cleared so the same frames
                # cannot queue a second one behind it.
                if pending is None:
                    pending = (time.monotonic() + wait, score)
                    log.info("wake held %.2f s: %.2f s buffered, speaker gate "
                             "wants %.1f s", wait, len(utterance) / native_rate,
                             WAKE_MIN_AUDIO_SECONDS)
                # An activation that stays up across several frames must NOT
                # re-arm the deadline -- it is the same wake word, and pushing
                # the deadline forward on every frame is an unbounded wait,
                # which is the bug this whole path exists to remove.
                recent.clear()
                continue
            if not wake_audio_sufficient(len(utterance), native_rate):
                # Too little to ever verify in time. ADMIT it: the speaker
                # gate abstains, exactly as its docstring and CLAUDE.md say
                # this layer must behave. Dropping it here is what cost the
                # user his first "Jarvis" after a reply -- and the transcript
                # gate (app._process_audio -> speaker.filter_segments) still
                # fails SHUT, so this admits a wake, never a stranger's
                # command.
                log.info("wake admitted on %.2f s of audio (score=%.3f): too "
                         "short to verify, speaker gate abstains",
                         len(utterance) / native_rate, score)

            # The buffer is fed from PortAudio's thread, so it can cross the
            # line between the `pending` check above and the snapshot here.
            # This IS the held wake, arriving on its own audio: disarm it, or
            # the deadline fires it a second time after the debounce.
            pending = None
            _fire(utterance, score)


# ----------------------------------------------------------------------
# Wake-word verifier training (port: voice_input_gui.py 5178-5260)
# ----------------------------------------------------------------------
def train_verifier(samples) -> Path:
    """Train a custom hey_jarvis verifier from recorded samples.

    samples: list of 16kHz mono int16 numpy arrays (the UI records them via
    Recorder.record_fixed and converts to int16). Pure logic — no UI, no
    threads; the caller runs this on a worker thread. Returns the saved
    model path; raises on failure.
    """
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
    for i, sample in enumerate(samples):
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

    log.info("Training custom verifier: %d positive, %d negative",
             len(pos_files), len(neg_files))

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

    log.info("Features: %d positive, %d negative",
             pos_features.shape[0], neg_features.shape[0])

    all_features = np.vstack((pos_features, neg_features))
    all_labels = np.array(
        [1] * pos_features.shape[0]
        + [0] * neg_features.shape[0])

    model = train_verifier_model(all_features, all_labels)

    # Save with joblib (sklearn standard)
    model_path = (Path.home() / ".aiws_trainer"
                  / "hey_jarvis_verifier.pkl")
    joblib.dump(model, str(model_path))
    log.info("Custom verifier saved: %s", model_path)
    return model_path
