"""Live model checks with SYNTHETIC audio (no microphone needed).

Skipped unless JARVIS_LIVE_MODELS=1: needs the network (edge-tts), the
OpenWakeWord models, the SpeechBrain ECAPA checkpoint and torch. Nothing
is played; the voiceprint is written to a temp file, never the real one.

    JARVIS_LIVE_MODELS=1 ~/vss_env/bin/python -m pytest tests/test_voice_models_live.py -q
"""
import asyncio
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("JARVIS_LIVE_MODELS"),
    reason="set JARVIS_LIVE_MODELS=1 to run the synthetic-audio model checks")


def _synth(text, voice, out):
    import edge_tts
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(
            edge_tts.Communicate(text, voice).save(str(out)), timeout=20))
    finally:
        loop.close()


def _load16k(path):
    import librosa
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    _synth("Hey Jarvis.", "en-GB-RyanNeural", d / "hey_ryan.wav")
    _synth("Hey Jarvis.", "en-US-JennyNeural", d / "hey_jenny.wav")
    _synth("The weather is pleasant this afternoon.", "en-GB-RyanNeural",
           d / "neg_ryan.wav")
    _synth("Good afternoon, sir. All systems are operational.",
           "en-GB-RyanNeural", d / "ryan1.wav")
    _synth("The workshop is quiet and the reactor is holding steady.",
           "en-GB-RyanNeural", d / "ryan2.wav")
    _synth("Good afternoon, sir. All systems are operational.",
           "en-US-GuyNeural", d / "guy1.wav")
    return d


def _oww_score(model, path):
    audio = _load16k(path)
    pad = np.zeros(8000, dtype=np.float32)
    a16 = (np.concatenate([pad, audio, pad]) * 32767).astype(np.int16)
    model.reset()
    best = 0.0
    for i in range(0, len(a16) - 1279, 1280):
        p = model.predict(a16[i:i + 1280])
        best = max(best, p.get("hey_jarvis", 0.0), p.get("hey_mycroft", 0.0) * 0.7)
    return best


def test_openwakeword_fires_on_synthetic_hey_jarvis(clips):
    from openwakeword.model import Model
    m = Model()
    assert _oww_score(m, clips / "hey_ryan.wav") >= 0.3
    assert _oww_score(m, clips / "hey_jenny.wav") >= 0.3
    assert _oww_score(m, clips / "neg_ryan.wav") < 0.3


def test_user_verifier_shim_does_not_raise(clips):
    """With the shim installed the (user-voice) verifier is consulted on
    the synthetic wake word without the oww 0.4.0 frame-size crash."""
    from jarvis import hotword as hw
    from jarvis.config import PATHS
    if not PATHS.HEY_JARVIS_VERIFIER.exists():
        pytest.skip("no trained verifier")
    import joblib
    from openwakeword.model import Model
    m = Model()
    ok, detail = hw.install_verifier(m, joblib.load(str(PATHS.HEY_JARVIS_VERIFIER)))
    assert ok, detail
    score = _oww_score(m, clips / "hey_ryan.wav")     # must not raise
    assert 0.0 <= score <= 1.0
    assert m.custom_verifier_models["hey_jarvis"].calls >= 1


def test_whisper_roundtrip(clips):
    from jarvis.transcriber import Transcriber
    tr = Transcriber()
    tr.load()
    res = tr.transcribe(_load16k(clips / "ryan1.wav"))
    assert res.accepted and "operational" in res.text.lower()


def test_ecapa_separates_voices(clips, tmp_path, monkeypatch):
    import jarvis.speaker as spk
    monkeypatch.setattr(spk, "VOICEPRINT_FILE", tmp_path / "vp.npz")
    v = spk.SpeakerVerifier()
    assert v.load_model()
    ok, n = v.enroll_from_audio(_load16k(clips / "ryan1.wav"))
    assert ok and n == 1 and (tmp_path / "vp.npz").exists()
    same, s_same = v.verify(_load16k(clips / "ryan2.wav"))
    other, s_other = v.verify(_load16k(clips / "guy1.wav"))
    assert same and not other and s_same > s_other + 0.3
