"""Speaker verification using NVIDIA TitaNet-Large.

Provides voice enrollment, real-time speaker matching, and passive
voiceprint improvement from confirmed recordings.

Stores voiceprint as a set of 192-dim embeddings in a .npz file.
Matching uses cosine similarity against the centroid of all embeddings.

Model: nvidia/speakerverification_en_titanet_large (0.66% EER)
Previous: SpeechBrain ECAPA-TDNN (0.80% EER) — replaced for better
accuracy on short (2-4 second) utterances.

Usage:
    verifier = SpeakerVerifier(gpu=1)
    verifier.load()

    # Enrollment
    verifier.enroll(audio_16k_np_array)

    # Verification
    is_user, score = verifier.verify(audio_16k_np_array)

    # Passive learning (add confirmed recording to voiceprint)
    verifier.add_sample(audio_16k_np_array)
"""

import threading
import numpy as np
from pathlib import Path

VOICEPRINT_FILE = Path.home() / ".aiws_trainer" / "voiceprint.npz"
SPEAKER_MODEL_DIR = Path.home() / ".aiws_trainer" / "speaker_model"

# Cosine similarity threshold for accepting a speaker match
# Lower = more permissive, higher = stricter
DEFAULT_THRESHOLD = 0.40

# Maximum stored embeddings (oldest beyond this are dropped)
MAX_EMBEDDINGS = 100

# Minimum audio length (seconds) for a useful embedding
MIN_AUDIO_SECONDS = 1.0

SAMPLE_RATE = 16000


from jarvis.logging import get_logger
_log = get_logger("VER")


MODEL_VERSION = "titanet-large"


class SpeakerVerifier:
    """TitaNet-Large speaker verification with enrollment and passive learning."""

    def __init__(self, gpu=1, threshold=DEFAULT_THRESHOLD):
        self.gpu = gpu
        self.threshold = threshold
        self._model = None
        self._embeddings = []      # List of 192-dim numpy arrays
        self._centroid = None       # Mean of all embeddings
        self._lock = threading.Lock()
        self._loaded = False
        self._model_loaded = False

    @property
    def enrolled(self):
        """Whether we have at least one voiceprint embedding."""
        return len(self._embeddings) > 0

    @property
    def num_samples(self):
        """Number of stored voice samples."""
        return len(self._embeddings)

    def load_model(self):
        """Load TitaNet-Large onto GPU. Call once at startup.

        Note: NeMo model loading uses torch internals for deserialization
        which is standard for ML model loading.
        """
        if self._model_loaded:
            return True
        try:
            import nemo.collections.asr as nemo_asr
            self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                "nvidia/speakerverification_en_titanet_large"
            )
            self._model = self._model.to(f"cuda:{self.gpu}")
            self._model_loaded = True
            _log(f"TitaNet-Large loaded on CUDA:{self.gpu}")
            return True
        except Exception as e:
            _log(f"Speaker model load error: {e}")
            return False

    def load(self):
        """Load saved voiceprint from disk.

        Checks model version — clears incompatible voiceprints from
        older models (ECAPA-TDNN embeddings vs TitaNet embeddings).
        numpy's .npz uses safe array serialization, not pickle.
        """
        if not VOICEPRINT_FILE.exists():
            self._loaded = True
            return
        try:
            data = np.load(VOICEPRINT_FILE)
            # Check model version compatibility
            version_file = VOICEPRINT_FILE.with_suffix(".version")
            stored_version = ""
            if version_file.exists():
                stored_version = version_file.read_text().strip()
            if stored_version and stored_version != MODEL_VERSION:
                _log(f"Voiceprint model mismatch: '{stored_version}' vs "
                     f"'{MODEL_VERSION}' — clearing old voiceprint")
                self.clear()
                self._loaded = True
                return
            self._embeddings = [data[k] for k in sorted(data.files)]
            self._recompute_centroid()
            _log(f"Voiceprint loaded: {len(self._embeddings)} samples")
        except Exception as e:
            _log(f"Voiceprint load error: {e}")
        self._loaded = True

    def save(self):
        """Save voiceprint to disk with model version marker."""
        with self._lock:
            if not self._embeddings:
                return
            try:
                VOICEPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
                arrays = {f"emb_{i:04d}": emb for i, emb in enumerate(self._embeddings)}
                np.savez(VOICEPRINT_FILE, **arrays)
                # Write version marker as separate file (avoids pickle in npz)
                version_file = VOICEPRINT_FILE.with_suffix(".version")
                version_file.write_text(MODEL_VERSION)
            except Exception as e:
                _log(f"Voiceprint save error: {e}")

    def _extract_embedding(self, audio_16k):
        """Extract a 192-dim speaker embedding from 16kHz audio.

        Args:
            audio_16k: numpy array, mono, 16kHz float32

        Returns:
            numpy array of shape (192,) or None on failure
        """
        if not self._model_loaded:
            return None
        if len(audio_16k) < int(SAMPLE_RATE * MIN_AUDIO_SECONDS):
            return None
        try:
            import tempfile, os, soundfile as sf
            # TitaNet requires file input
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                sf.write(tmp.name, audio_16k.astype(np.float32), SAMPLE_RATE)
                embedding = self._model.get_embedding(tmp.name)
                # Shape: (1, 192) -> (192,)
                return embedding.squeeze().cpu().numpy()
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        except Exception as e:
            _log(f"Embedding extraction error: {e}")
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

    def enroll(self, audio_16k):
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
        _log(f"Enrolled sample #{len(self._embeddings)} "
             f"(embedding norm: {np.linalg.norm(embedding):.3f})")
        return True, len(self._embeddings)

    def verify(self, audio_16k):
        """Check if audio matches the enrolled voiceprint.

        Args:
            audio_16k: numpy array, mono, 16kHz float32

        Returns:
            (is_match: bool, score: float)
            score is cosine similarity (0-1), higher = more similar
        """
        if not self.enrolled:
            # No voiceprint yet — accept all audio
            return True, 1.0

        embedding = self._extract_embedding(audio_16k)
        if embedding is None:
            # Can't extract embedding (too short, etc.) — accept
            return True, 1.0

        with self._lock:
            score = self._cosine_similarity(embedding, self._centroid)

        is_match = score >= self.threshold
        _log(f"Speaker verify: score={score:.3f} threshold={self.threshold} "
             f"{'MATCH' if is_match else 'REJECT'}")
        return is_match, score

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
        if self.enrolled:
            with self._lock:
                score = self._cosine_similarity(embedding, self._centroid)
            if score < self.threshold:
                _log(f"Passive sample rejected (score={score:.3f} < {self.threshold})")
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
        if not self.enrolled or not self._model_loaded:
            return audio_16k, {"total": 0, "matched": 0, "scores": []}

        window_samples = int(window_sec * SAMPLE_RATE)
        hop_samples = int(hop_sec * SAMPLE_RATE)
        total_samples = len(audio_16k)

        if total_samples < window_samples:
            # Audio shorter than one window — fall back to whole-clip verify
            is_match, score = self.verify(audio_16k)
            if is_match:
                return audio_16k, {"total": 1, "matched": 1, "scores": [score]}
            return None, {"total": 1, "matched": 0, "scores": [score]}

        # Extract embeddings for all windows in a batch
        import torch
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
        except Exception as e:
            _log(f"Segment verification error: {e}")
            return audio_16k, {"total": len(windows), "matched": 0, "scores": []}

        matched_count = sum(matched_mask)
        total_count = len(windows)

        _log(f"Segment filter: {matched_count}/{total_count} windows matched "
             f"(scores: {', '.join(f'{s:.2f}' for s in scores)})")

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

        return filtered, {"total": total_count, "matched": matched_count, "scores": scores}

    def clear(self):
        """Delete all voiceprint data."""
        with self._lock:
            self._embeddings.clear()
            self._centroid = None
        try:
            VOICEPRINT_FILE.unlink(missing_ok=True)
            _log("Voiceprint cleared")
        except Exception as e:
            _log(f"Voiceprint clear error: {e}")
