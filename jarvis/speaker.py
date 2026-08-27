"""Speaker verification using SpeechBrain ECAPA-TDNN.

V3 evolution of speaker_verification.py (same math, same thresholds):
- no reverse import of voice_input_gui (uses jarvis.logs)
- atomic voiceprint save (np.savez to .tmp file handle + os.replace)
- gpu default 0 with automatic CPU fallback (GB10 / no-CUDA machines)
- fail-open verify kept, but logged at WARNING and surfaced once per
  session as a Status(kind="warn") event on the bus

Stores the voiceprint as a set of 192-dim embeddings in a .npz file.
Matching uses cosine similarity against the centroid of all embeddings.

Usage:
    verifier = SpeakerVerifier()          # gpu=0 default, CPU fallback
    verifier.load_model()
    verifier.load()

    ok, n = verifier.enroll_from_audio(audio_16k_np_array)
    is_user, score = verifier.verify(audio_16k_np_array)
    verifier.add_sample(audio_16k_np_array)     # passive learning
"""
from __future__ import annotations

import os
import threading

import numpy as np

from jarvis.config import PATHS
from jarvis.events import Status, bus
from jarvis.logs import get_logger

log = get_logger("speaker")

VOICEPRINT_FILE = PATHS.VOICEPRINT                    # ~/.aiws_trainer/voiceprint.npz
SPEAKER_MODEL_DIR = PATHS.AIWS / "speaker_model"

# Cosine similarity threshold for accepting a speaker match
# Lower = more permissive, higher = stricter
DEFAULT_THRESHOLD = 0.40

# Maximum stored embeddings (oldest beyond this are dropped)
MAX_EMBEDDINGS = 100

# Minimum audio length (seconds) for a useful embedding
MIN_AUDIO_SECONDS = 1.0

SAMPLE_RATE = 16000


class SpeakerVerifier:
    """ECAPA-TDNN speaker verification with enrollment and passive learning."""

    def __init__(self, gpu=0, threshold=DEFAULT_THRESHOLD):
        self.gpu = gpu
        self.threshold = threshold
        self._model = None
        self._embeddings = []       # List of 192-dim numpy arrays
        self._centroid = None       # Mean of all embeddings
        self._lock = threading.Lock()
        self._loaded = False
        self._model_loaded = False
        self._device = None         # resolved by _resolve_device()
        self._warned_fail_open = False   # one Status(warn) per session
        self._model_failed = False       # a failed load is not retried

    # ------------------------------------------------------------ state
    @property
    def is_enrolled(self):
        """Whether we have at least one voiceprint embedding."""
        return len(self._embeddings) > 0

    # Legacy alias (voice_input_gui / hotword_daemon used .enrolled)
    @property
    def enrolled(self):
        return self.is_enrolled

    @property
    def num_samples(self):
        """Number of stored voice samples."""
        return len(self._embeddings)

    # ------------------------------------------------------- fail-open
    def _fail_open(self, reason):
        """Log a fail-open acceptance loudly; publish one warn Status/session."""
        log.warning("speaker verify fail-open (%s) — accepting audio", reason)
        if not self._warned_fail_open:
            self._warned_fail_open = True
            try:
                bus.publish(Status(
                    text=f"Speaker verify inactive ({reason}) — accepting all audio",
                    kind="warn"))
            except Exception:
                log.exception("failed to publish fail-open Status")

    def _fail_shut(self, reason):
        """Reject audio we could not verify, loudly.

        Only reachable once a voiceprint exists: the user has explicitly asked
        for other voices to be filtered, so accepting everything because the
        model broke defeats the feature -- that is exactly how a television
        reached the commander. Published every time rather than once per
        session, because this blocks the voice path and a silent block is the
        experience being avoided. Typed input is unaffected.
        """
        log.error("speaker verify FAILED SHUT (%s) -- rejecting audio; "
                  "typed input still works", reason)
        try:
            bus.publish(Status(
                text=f"Voice blocked: speaker check unavailable ({reason})",
                kind="error"))
        except Exception:
            log.exception("failed to publish fail-shut Status")

    # ---------------------------------------------------------- device
    def _resolve_device(self):
        """Pick cuda:{gpu} when torch CUDA is usable, else cpu."""
        if self._device is not None:
            return self._device
        try:
            import torch
            if torch.cuda.is_available() and 0 <= self.gpu < torch.cuda.device_count():
                self._device = f"cuda:{self.gpu}"
            else:
                self._device = "cpu"
        except Exception:
            log.exception("torch device probe failed; using cpu")
            self._device = "cpu"
        return self._device

    # ----------------------------------------------------------- model
    def load_model(self):
        """Load the ECAPA-TDNN model. Call once at startup.

        Tries the resolved device (cuda:{gpu} when available, else cpu) and
        falls back to cpu when a CUDA load fails.
        """
        if self._model_loaded:
            return True
        device = self._resolve_device()
        for attempt_device in ([device, "cpu"] if device != "cpu" else ["cpu"]):
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": attempt_device},
                    savedir=str(SPEAKER_MODEL_DIR),
                )
                self._device = attempt_device
                self._model_loaded = True
                log.info("speaker model loaded on %s", attempt_device)
                return True
            except Exception:
                log.exception("speaker model load failed on %s", attempt_device)
        return False

    # ----------------------------------------------------- persistence
    def load(self):
        """Load saved voiceprint from disk."""
        if not VOICEPRINT_FILE.exists():
            self._loaded = True
            return
        try:
            data = np.load(VOICEPRINT_FILE)
            self._embeddings = [data[k] for k in sorted(data.files)]
            self._recompute_centroid()
            log.info("voiceprint loaded: %d samples", len(self._embeddings))
        except Exception:
            log.exception("voiceprint load error")
        self._loaded = True

    def save(self):
        """Save voiceprint to disk atomically (.tmp + os.replace)."""
        with self._lock:
            if not self._embeddings:
                return
            try:
                VOICEPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
                arrays = {f"emb_{i:04d}": emb for i, emb in enumerate(self._embeddings)}
                tmp = VOICEPRINT_FILE.with_name(VOICEPRINT_FILE.name + ".tmp")
                # savez appends ".npz" to bare paths; a file handle keeps the
                # exact tmp name so os.replace targets the right file.
                with open(tmp, "wb") as fh:
                    np.savez(fh, **arrays)
                os.replace(tmp, VOICEPRINT_FILE)
            except Exception:
                log.exception("voiceprint save error")

    def _ensure_model(self) -> bool:
        """Load the speaker model on first use, at most once.

        The app warms this on its model-loader thread, but audio can arrive
        before that finishes; now that verification fails SHUT, rejecting
        merely because the load had not happened yet would make voice dead for
        the opening seconds of every session -- indistinguishable from the
        feature being broken. A load that genuinely FAILS is not retried: a
        broken model is almost always permanently broken, and retrying would
        add seconds to every rejection.
        """
        if self._model_loaded:
            return True
        if self._model_failed:
            return False
        if self.load_model():
            return True
        self._model_failed = True
        return False

    # ------------------------------------------------------ embeddings
    def _extract_embedding(self, audio_16k):
        """Extract a 192-dim speaker embedding from 16kHz audio.

        Args:
            audio_16k: numpy array, mono, 16kHz float32

        Returns:
            numpy array of shape (192,) or None on failure
        """
        if not self._ensure_model():
            return None
        if len(audio_16k) < int(SAMPLE_RATE * MIN_AUDIO_SECONDS):
            return None
        try:
            import torch
            # SpeechBrain expects (batch, time) tensor
            waveform = torch.tensor(audio_16k, dtype=torch.float32).unsqueeze(0)
            waveform = waveform.to(self._device or self._resolve_device())
            with torch.no_grad():
                embedding = self._model.encode_batch(waveform)
            # Shape: (1, 1, 192) -> (192,)
            return embedding.squeeze().cpu().numpy()
        except Exception:
            log.exception("embedding extraction error")
            return None

    def _recompute_centroid(self):
        """Recompute the mean embedding (centroid) from all samples."""
        if self._embeddings:
            self._centroid = np.mean(self._embeddings, axis=0)
        else:
            self._centroid = None

    def _cosine_similarity(self, a, b):
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------ enrollment
    def enroll_from_audio(self, audio_16k):
        """Add an enrollment sample. Returns (success, num_total_samples).

        Args:
            audio_16k: numpy array, mono, 16kHz float32
        """
        embedding = self._extract_embedding(audio_16k)
        if embedding is None:
            return False, len(self._embeddings)

        with self._lock:
            self._embeddings.append(embedding)
            self._recompute_centroid()

        self.save()
        log.info("enrolled sample #%d (embedding norm: %.3f)",
                 len(self._embeddings), np.linalg.norm(embedding))
        return True, len(self._embeddings)

    # Legacy alias (voice_input_gui called .enroll)
    enroll = enroll_from_audio

    # ---------------------------------------------------- verification
    def score(self, audio_16k):
        """Cosine similarity to the voiceprint, or None if it cannot be computed.

        Applies no accept/reject policy, unlike verify(), which collapses "not
        you" and "could not tell" into the same False. Callers needing their
        own fallback use this -- the wake-word gate fails OPEN where the
        transcript gate fails SHUT, so it cannot reuse verify()'s verdict.
        """
        if not self.is_enrolled:
            return None
        embedding = self._extract_embedding(audio_16k)
        if embedding is None:
            return None
        with self._lock:
            return float(self._cosine_similarity(embedding, self._centroid))

    def verify(self, audio_16k):
        """Check if audio matches the enrolled voiceprint.

        Args:
            audio_16k: numpy array, mono, 16kHz float32

        Returns:
            (is_match: bool, score: float)
            score is cosine similarity (0-1), higher = more similar

        Fail-open: with no voiceprint, no model, or unextractable audio the
        clip is ACCEPTED (True, 1.0) — logged at WARNING, with one
        Status(kind=warn) event per session.
        """
        if not self.is_enrolled:
            # No voiceprint yet — accept all audio
            self._fail_open("no voiceprint enrolled")
            return True, 1.0

        embedding = self._extract_embedding(audio_16k)
        if embedding is None:
            # Can't extract embedding (model missing, audio too short) — accept
            reason = ("model not loaded" if not self._model_loaded
                      else "no embedding (audio too short?)")
            self._fail_shut(reason)
            return False, 0.0

        with self._lock:
            score = self._cosine_similarity(embedding, self._centroid)

        is_match = score >= self.threshold
        log.info("speaker verify: score=%.3f threshold=%s %s",
                 score, self.threshold, "MATCH" if is_match else "REJECT")
        return is_match, score

    # ------------------------------------------------ passive learning
    def add_sample(self, audio_16k):
        """Passively improve voiceprint with a confirmed recording.

        Only call this after the user has accepted the transcription
        (didn't reject or cancel it).

        Args:
            audio_16k: numpy array, mono, 16kHz float32

        Returns:
            True if sample was added
        """
        embedding = self._extract_embedding(audio_16k)
        if embedding is None:
            return False

        # Only add if it matches current profile (sanity check)
        if self.is_enrolled:
            with self._lock:
                score = self._cosine_similarity(embedding, self._centroid)
            if score < self.threshold:
                log.info("passive sample rejected (score=%.3f < %s)",
                         score, self.threshold)
                return False

        with self._lock:
            self._embeddings.append(embedding)
            # Trim oldest if over limit (keep first 10 enrollment + newest)
            if len(self._embeddings) > MAX_EMBEDDINGS:
                # Keep first 10 (original enrollment) + newest
                keep_first = min(10, len(self._embeddings) // 2)
                keep_recent = MAX_EMBEDDINGS - keep_first
                self._embeddings = (
                    self._embeddings[:keep_first]
                    + self._embeddings[-keep_recent:]
                )
            self._recompute_centroid()

        self.save()
        return True

    # ------------------------------------------------ segment filtering
    def filter_segments(self, audio_16k, window_sec=3.0, hop_sec=1.5):
        """Filter audio to keep only segments matching the enrolled voice.

        Splits audio into overlapping windows, verifies each against the
        voiceprint, and returns only the windows that match. This allows
        the user to speak while a TV/YouTube is playing — their voice
        segments pass, background voices are dropped.

        Args:
            audio_16k: numpy array, mono, 16kHz float32
            window_sec: window size in seconds (default 3.0)
            hop_sec: hop between windows in seconds (default 1.5)

        Returns:
            (filtered_audio, stats_dict)
            filtered_audio: numpy array of concatenated matching segments,
                            or None if no segments matched
            stats_dict: {'total': N, 'matched': M, 'scores': [...]}
        """
        if not self.is_enrolled:
            # Unconfigured: pass through, or voice never works on a fresh box.
            self._fail_open("no voiceprint enrolled")
            return audio_16k, {"total": 0, "matched": 0, "scores": []}
        if not self._ensure_model():
            self._fail_shut("model not loaded")
            return None, {"total": 0, "matched": 0, "scores": []}

        window_samples = int(window_sec * SAMPLE_RATE)
        hop_samples = int(hop_sec * SAMPLE_RATE)
        total_samples = len(audio_16k)

        if total_samples < window_samples:
            # Audio shorter than one window — fall back to whole-clip verify
            is_match, score = self.verify(audio_16k)
            if is_match:
                return audio_16k, {"total": 1, "matched": 1, "scores": [score]}
            return None, {"total": 1, "matched": 0, "scores": [score]}

        windows = []
        positions = []
        pos = 0
        while pos + window_samples <= total_samples:
            windows.append(audio_16k[pos:pos + window_samples])
            positions.append(pos)
            pos += hop_samples
        # Include tail if it's at least 1.5 seconds
        if pos < total_samples and (total_samples - pos) >= int(1.5 * SAMPLE_RATE):
            windows.append(audio_16k[pos:total_samples])
            positions.append(pos)

        # Batch embedding extraction for speed
        scores = []
        matched_mask = []
        try:
            for chunk in windows:
                emb = self._extract_embedding(chunk)
                if emb is not None:
                    with self._lock:
                        score = self._cosine_similarity(emb, self._centroid)
                    scores.append(score)
                    matched_mask.append(score >= self.threshold)
                else:
                    scores.append(0.0)
                    matched_mask.append(False)
        except Exception:
            log.exception("segment verification error")
            self._fail_shut("segment verification error")
            return None, {"total": len(windows), "matched": 0, "scores": []}

        matched_count = sum(matched_mask)
        total_count = len(windows)

        log.info("segment filter: %d/%d windows matched (scores: %s)",
                 matched_count, total_count,
                 ", ".join(f"{s:.2f}" for s in scores))

        if matched_count == 0:
            return None, {"total": total_count, "matched": 0, "scores": scores}

        # Reconstruct audio from matched segments using a mask over the
        # original audio to preserve continuity where possible
        # Mark each sample as "keep" if it belongs to any matched window
        keep = np.zeros(total_samples, dtype=bool)
        for i, pos in enumerate(positions):
            if matched_mask[i]:
                end = min(pos + len(windows[i]), total_samples)
                keep[pos:end] = True

        filtered = audio_16k[keep]

        return filtered, {"total": total_count, "matched": matched_count,
                          "scores": scores}

    # ------------------------------------------------------------ reset
    def clear(self):
        """Delete all voiceprint data."""
        with self._lock:
            self._embeddings.clear()
            self._centroid = None
        try:
            VOICEPRINT_FILE.unlink(missing_ok=True)
            log.info("voiceprint cleared")
        except Exception:
            log.exception("voiceprint clear error")
