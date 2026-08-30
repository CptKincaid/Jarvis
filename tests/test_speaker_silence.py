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

from jarvis.speaker import MIN_SPEECH_SECONDS, SAMPLE_RATE, trim_silence


def _speech(seconds=1.4, amp=0.2, seed=0):
    """Broadband noise stands in for speech: the trim is purely energetic."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, amp, int(SAMPLE_RATE * seconds)).astype(np.float32)


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
    clip = _speech(2.0)
    assert _seconds(trim_silence(clip)) > 1.9


def test_never_trims_below_the_minimum():
    """Under MIN_SPEECH_SECONDS we cannot tell speech from a click, so the
    audio is handed on whole rather than reduced to a fragment."""
    clip = np.concatenate([_speech(0.1), _silence(2.0)])
    out = trim_silence(clip)
    assert len(out) == len(clip) or _seconds(out) >= MIN_SPEECH_SECONDS


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
