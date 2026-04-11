import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_extract_embedding_shape():
    from jarvis.speaker_verification import SpeakerVerifier
    v = SpeakerVerifier(gpu=1)
    v.load_model()
    audio = np.random.randn(16000 * 3).astype(np.float32)
    emb = v._extract_embedding(audio)
    assert emb is not None
    assert emb.shape == (192,)


def test_extract_embedding_short_clip():
    from jarvis.speaker_verification import SpeakerVerifier
    v = SpeakerVerifier(gpu=1)
    v.load_model()
    audio = np.random.randn(16000 * 2).astype(np.float32)
    emb = v._extract_embedding(audio)
    assert emb is not None
    assert emb.shape == (192,)
    assert not np.allclose(emb, 0)  # Must NOT be all zeros


def test_self_similarity_high():
    from jarvis.speaker_verification import SpeakerVerifier
    v = SpeakerVerifier(gpu=1)
    v.load_model()
    audio = np.random.randn(16000 * 5).astype(np.float32)
    emb1 = v._extract_embedding(audio)
    emb2 = v._extract_embedding(audio)
    sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    assert sim > 0.95  # Same audio should be near-identical


def test_enroll_and_verify():
    from jarvis.speaker_verification import SpeakerVerifier
    v = SpeakerVerifier(gpu=1)
    v.load_model()
    v.load()
    audio = np.random.randn(16000 * 5).astype(np.float32)
    ok, count = v.enroll(audio)
    assert ok is True
    assert count == 1
    is_match, score = v.verify(audio)
    assert is_match is True
    assert score > 0.9
    v.clear()


def test_too_short_returns_none():
    from jarvis.speaker_verification import SpeakerVerifier
    v = SpeakerVerifier(gpu=1)
    v.load_model()
    short = np.random.randn(8000).astype(np.float32)  # 0.5 seconds
    emb = v._extract_embedding(short)
    assert emb is None  # Below MIN_AUDIO_SECONDS
