"""Silence must not be pooled into a speaker embedding.

ECAPA averages over every frame it is given, so trailing silence does not
merely add nothing -- it pulls the embedding toward the silence embedding,
which scores about -0.10 against an enrolled voiceprint. A real capture on
2026-08-29 held 1.4 s of speech inside a 3.9 s recording (the recorder appends
its 2.5 s silence auto-stop) and scored 0.253 where that voice scores 0.63
clean, so every short utterance was "Rejected (speaker)". These tests pin the
endpoint trim that fixed it, and the fail-safe that keeps it from ever being
the reason a real voice is turned away.
"""
import numpy as np

import threading

from jarvis.speaker import (MIN_SPEECH_SECONDS, SAMPLE_RATE, SpeakerVerifier,
                            speech_bounds, trim_silence)


def _speech(seconds=1.4, amp=0.2, seed=0):
    """Syllable-like speech: broadband noise under a 4 Hz on/off envelope, so
    the clip has the loud/quiet contrast real speech has. Flat noise does not
    -- its quietest tenth sits within 3x of its peak, which the trim rightly
    treats as "cannot tell" and leaves alone (see test_a_flat_clip_...)."""
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    env = np.where(np.cos(2 * np.pi * 4 * t) >= 0, 1.0, 0.1)   # loud at both ends
    return (rng.normal(0, amp, n) * env).astype(np.float32)


def _silence(seconds, amp=1e-4, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0, amp, int(SAMPLE_RATE * seconds)).astype(np.float32)


def _seconds(audio):
    return len(audio) / SAMPLE_RATE


def test_trailing_auto_stop_silence_is_removed():
    """The exact shape that caused the rejection: 1.4 s in a 3.9 s capture."""
    clip = np.concatenate([_speech(1.4), _silence(2.5)])
    out = trim_silence(clip)
    assert _seconds(out) < 1.7, f"kept {_seconds(out):.2f}s of 3.90s"
    assert _seconds(out) > 1.2, "ate into the speech itself"


def test_leading_silence_is_removed():
    """Capture starts before the user does; the wake gate scores that 1 s."""
    clip = np.concatenate([_silence(1.0), _speech(1.4)])
    out = trim_silence(clip)
    assert _seconds(out) < 1.7, f"kept {_seconds(out):.2f}s of 2.40s"


def test_interior_pauses_are_kept():
    """A pause mid-sentence is speech rhythm, and the enrolment audio has it
    too -- splicing it out would widen the mismatch, not close it."""
    clip = np.concatenate([_speech(0.8), _silence(0.5), _speech(0.8, seed=2)])
    assert _seconds(trim_silence(clip)) > 2.0


def test_pure_silence_is_returned_unchanged():
    """Nothing to find. Unchanged is today's behaviour, and the caller still
    rejects it on score -- a trim must never be what fails a gate shut."""
    clip = _silence(3.0)
    assert len(trim_silence(clip)) == len(clip)


def test_quiet_speech_survives():
    """Thresholds are relative to the clip's own peak, so a softly spoken
    utterance is not mistaken for room tone and deleted."""
    clip = np.concatenate([_speech(1.4, amp=0.01), _silence(2.5, amp=1e-5)])
    out = trim_silence(clip)
    assert _seconds(out) < 1.7 and _seconds(out) > 1.2


def test_clean_speech_is_a_no_op():
    """Speech from the first sample to the last: bounds cover the clip."""
    clip = _speech(2.0)
    lo, hi = speech_bounds(clip)
    assert lo <= SAMPLE_RATE * 0.05 and hi >= len(clip) - SAMPLE_RATE * 0.05
    assert _seconds(trim_silence(clip)) > 1.9


def test_a_flat_clip_is_left_alone_and_reported_as_unknown():
    """Constant-level noise has no floor to measure against: the first
    version of the no-op test passed through this path by accident and
    called it a trim. It is a refusal, and it must not shrink anything."""
    rng = np.random.default_rng(0)
    flat = rng.normal(0, 0.2, int(SAMPLE_RATE * 2.0)).astype(np.float32)
    assert speech_bounds(flat) is None
    assert len(trim_silence(flat)) == len(flat)


def test_never_trims_below_the_minimum():
    """Under MIN_SPEECH_SECONDS we cannot tell speech from a click, so the
    audio is handed on whole rather than reduced to a fragment."""
    clip = np.concatenate([_speech(0.1), _silence(2.0)])
    assert speech_bounds(clip) is None
    assert len(trim_silence(clip)) == len(clip), "reduced audio it could not judge"
    assert MIN_SPEECH_SECONDS > 0.1


def test_embedding_path_applies_the_trim(monkeypatch):
    """The trim has to sit at the choke point, or the gates keep scoring
    silence: score(), verify(), enrol and every filter_segments window."""
    import torch

    import jarvis.speaker as speaker_mod

    seen = {}

    class FakeModel:
        def encode_batch(self, waveform):
            seen["samples"] = waveform.shape[-1]
            return torch.zeros(1, 1, 192)

    v = speaker_mod.SpeakerVerifier()
    monkeypatch.setattr(v, "_ensure_model", lambda: True)
    monkeypatch.setattr(v, "_model", FakeModel(), raising=False)
    monkeypatch.setattr(v, "_device", "cpu", raising=False)

    clip = np.concatenate([_speech(1.4), _silence(2.5)])
    assert v._extract_embedding(clip) is not None
    assert seen["samples"] < len(clip), "model saw the untrimmed audio"


def _verifier_with(fake_embed):
    v = object.__new__(SpeakerVerifier)
    vec = np.ones(192, dtype=np.float32) / np.sqrt(192)
    v._embeddings, v._centroid = [vec], vec
    v._lock, v.threshold = threading.Lock(), 0.3
    v._model_failed, v._warned_fail_open = False, True
    v._ensure_model = lambda: True
    v._extract_embedding = fake_embed
    return v


def test_filter_segments_skips_a_window_holding_no_speech():
    """The 2.5 s silence auto-stop leaves a trailing window of room tone. It
    can only ever score the silence embedding, so it is not scored at all.
    The first version compared len(trim_silence(w)) to a minimum -- and
    trim_silence never shrinks a clip it cannot judge, so nothing was ever
    skipped and the commit that claimed it was wrong."""
    calls = []
    def fake_embed(audio):
        calls.append(len(audio) / SAMPLE_RATE)
        return np.ones(192, dtype=np.float32) / np.sqrt(192)
    v = _verifier_with(fake_embed)
    audio = np.concatenate([_speech(1.4), _silence(2.5)])       # 3.9 s capture
    kept, stats = v.filter_segments(audio)
    assert calls == [3.0], f"scored windows of {calls}s; the 2.4 s tail is silence"
    assert stats["total"] == 1 and kept is not None


def test_filter_segments_scores_everything_when_it_cannot_tell():
    """A flat capture has no floor: every window is scored, as before."""
    calls = []
    def fake_embed(audio):
        calls.append(len(audio))
        return np.ones(192, dtype=np.float32) / np.sqrt(192)
    v = _verifier_with(fake_embed)
    rng = np.random.default_rng(1)
    flat = rng.normal(0, 0.2, int(SAMPLE_RATE * 3.9)).astype(np.float32)
    v.filter_segments(flat)
    assert len(calls) == 2


def test_fail_shut_is_paced_not_silenced(monkeypatch):
    """The recorder asks at 1 Hz; a broken model must not toast every second,
    but must keep saying so -- a silent block is the experience avoided."""
    import jarvis.speaker as speaker_mod
    published = []
    monkeypatch.setattr(speaker_mod.bus, "publish", published.append)
    v = object.__new__(SpeakerVerifier)
    clock = {"t": 100.0}
    monkeypatch.setattr(speaker_mod.time, "monotonic", lambda: clock["t"])
    v._fail_shut("model not loaded")
    v._fail_shut("model not loaded")
    v._fail_shut("model not loaded")
    assert len(published) == 1
    clock["t"] += 6.0
    v._fail_shut("model not loaded")
    assert len(published) == 2


def test_a_clip_too_short_to_judge_abstains_without_the_blocked_toast(monkeypatch):
    """Live 21:47:33: a 0.5 s manual-stop clip -- long enough to finalize,
    too short for ECAPA -- failed SHUT and toasted "Voice blocked: speaker
    check unavailable", the line that means the model is down. It is not.

    Since 2026-08-31 the short clip ABSTAINS rather than rejecting: measured
    on Hunter's own clips, FRR@0.30 is 30% at 1.0 s of trimmed speech and 0%
    at 3.0 s, so a score from half a second is a coin flip, and a coin flip
    that silently drops his command is the worse of the two errors. Still no
    embedding, still no toast."""
    import jarvis.speaker as speaker_mod
    published = []
    monkeypatch.setattr(speaker_mod.bus, "publish", published.append)
    v = _verifier_with(lambda audio: (_ for _ in ()).throw(AssertionError("must not embed")))
    v._model_loaded = True
    ok, score = v.verify(_speech(0.5))
    assert ok is True and score == 0.0, "too little speech must fail OPEN"
    assert published == [], f"toasted for a short clip: {published}"


def test_enough_speech_is_still_judged_on_its_score(monkeypatch):
    """The abstain window is a floor, not a bypass: past it the score
    decides exactly as before."""
    import numpy as np
    v = _verifier_with(lambda audio: np.ones(192, dtype=np.float32) / np.sqrt(192))
    v._model_loaded = True
    v._centroid = -np.ones(192, dtype=np.float32) / np.sqrt(192)   # nothing like him
    ok, score = v.verify(_speech(3.0))
    assert ok is False and score < 0.0


def test_a_pre_trim_voiceprint_keeps_its_format_until_re_enrolled(tmp_path, monkeypatch):
    """save() must write the pool's format, not the code's: otherwise the
    first save after today's change would stamp a silence-pooled voiceprint
    as current and the re-enrol warning would vanish."""
    import jarvis.speaker as speaker_mod
    path = tmp_path / "voiceprint.npz"
    monkeypatch.setattr(speaker_mod, "VOICEPRINT_FILE", path)
    vec = np.ones(192, dtype=np.float32) / np.sqrt(192)
    np.savez(path, emb_0000=vec, emb_0001=vec)             # format-1 file: no marker
    v = speaker_mod.SpeakerVerifier()
    v.load()
    assert v.num_samples == 2 and v._format == 1
    v.save()
    assert int(np.load(path)["_format"][0]) == 1, "stamped a stale pool as current"
    # re-enrolment replaces the stale pool instead of mixing into it
    monkeypatch.setattr(v, "_extract_embedding", lambda audio: vec)
    ok, n = v.enroll_from_audio(_speech(2.0))
    assert ok and n == 1 and v._format == speaker_mod.VOICEPRINT_FORMAT
    assert int(np.load(path)["_format"][0]) == speaker_mod.VOICEPRINT_FORMAT
    v2 = speaker_mod.SpeakerVerifier()
    v2.load()
    assert v2.num_samples == 1, "the _format array was loaded as an embedding"
    # passive learning refuses to pollute a stale pool
    v3 = speaker_mod.SpeakerVerifier()
    v3._format, v3._embeddings = 1, [vec]
    assert v3.add_sample(_speech(2.0)) is False
