"""Wake-word gating: frame agreement, the unverified head, and the speaker gate.

Three separate holes let a television wake Jarvis on 2026-08-27 at 12:13:29
(score 0.415):

1. A SINGLE 80 ms frame over threshold fired the wake word. No averaging, no
   N-of-M, no patience -- one transient frame was enough.
2. `hey_mycroft * 0.7` reached the same threshold while being completely
   unverified: the trained custom verifier is registered under `hey_jarvis`
   only (`openwakeword/model.py:235` gates on `custom_verifier_models.get(
   parent_model)`), so a raw mycroft score of 0.3/0.7 = 0.4286 fired with
   nothing vetting it.
3. Nothing checked WHO was speaking, even though a voiceprint now exists and
   scoring 2 s of audio costs ~10 ms once warm.

The speaker gate here deliberately fails OPEN, which is the opposite of the
transcript-level gate in `jarvis/app.py`. A wake word that cannot be triggered
is worse than one that triggers too often, and the transcript gate still fails
shut behind it -- defence in depth, each layer failing the safe way for its own
position.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import hotword as hw


# --------------------------------------------------------- scoring rule
def test_verified_head_fires_at_the_normal_threshold():
    hit, score = hw.wake_hit({"hey_jarvis": 0.31}, 0.30, 0.60)
    assert hit is True
    assert score == pytest.approx(0.31)


def test_unverified_head_needs_the_higher_bar():
    """0.4286 raw mycroft used to be enough; nothing vets that head."""
    preds = {"hey_mycroft": 0.4286}          # * 0.7 == 0.30, the old threshold
    hit, _ = hw.wake_hit(preds, 0.30, 0.60)
    assert hit is False, "unverified head must not fire at the verified bar"


def test_unverified_head_still_fires_when_confident():
    preds = {"hey_mycroft": 0.90}            # * 0.7 == 0.63
    hit, score = hw.wake_hit(preds, 0.30, 0.60)
    assert hit is True
    assert score == pytest.approx(0.63)


def test_below_both_bars_is_silence():
    assert hw.wake_hit({"hey_jarvis": 0.2, "hey_mycroft": 0.4}, 0.30, 0.60)[0] is False


def test_missing_keys_do_not_raise():
    assert hw.wake_hit({}, 0.30, 0.60) == (False, 0.0)


# ------------------------------------------------------ frame agreement
def test_one_lone_frame_does_not_fire():
    """The 12:13:29 failure mode: a single transient frame."""
    assert hw.frames_agree([True], 3, 2) is False
    assert hw.frames_agree([False, True, False], 3, 2) is False


def test_two_of_three_frames_fire():
    assert hw.frames_agree([True, False, True], 3, 2) is True
    assert hw.frames_agree([False, True, True], 3, 2) is True


def test_agreement_only_considers_the_recent_window():
    """An old hit must not combine with a new one across a long gap."""
    history = [True] + [False] * 10 + [True]
    assert hw.frames_agree(history, 3, 2) is False


# ---------------------------------------------------------- speaker gate
class FakeSpeaker:
    def __init__(self, value, enrolled=True):
        self.is_enrolled = enrolled
        self._value = value
        self.calls = 0

    def score(self, audio_16k):
        self.calls += 1
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


@pytest.fixture
def audio():
    return np.zeros(16000 * 2, dtype=np.float32)


def gate(speaker, audio, native_rate=16000, enabled=True):
    hw.CONFIG.speaker_verify = enabled     # one switch governs all gating
    h = hw.Hotword.__new__(hw.Hotword)     # no audio device needed
    h._speaker = speaker
    return h._speaker_ok(audio, native_rate)


@pytest.fixture(autouse=True)
def _restore_config():
    was = hw.CONFIG.speaker_verify
    yield
    hw.CONFIG.speaker_verify = was


def test_enrolled_speaker_passes(audio):
    assert gate(FakeSpeaker(0.72), audio) is True


def test_other_voice_is_suppressed(audio):
    """The TV scored ~0.0 against the enrolled voiceprint."""
    assert gate(FakeSpeaker(0.02), audio) is False


def test_gate_fails_OPEN_when_score_is_unavailable(audio):
    """Opposite policy to app.py's transcript gate, on purpose: a model
    failure must never make Jarvis impossible to wake."""
    assert gate(FakeSpeaker(None), audio) is True


def test_gate_fails_OPEN_when_scoring_raises(audio):
    assert gate(FakeSpeaker(RuntimeError("cuda gone")), audio) is True


def test_no_voiceprint_means_no_gating(audio):
    spk = FakeSpeaker(0.0, enrolled=False)
    assert gate(spk, audio) is True
    assert spk.calls == 0, "must not score when nothing is enrolled"


def test_no_speaker_wired_at_all_is_harmless(audio):
    assert gate(None, audio) is True


def test_wake_gate_is_more_permissive_than_the_transcript_gate():
    """Wake audio is short and partial, so it scores lower than a full
    utterance; gating it at the transcript threshold would reject the user.
    The TV sat near 0.0, so there is ample room below."""
    assert hw.Hotword.SPEAKER_WAKE_MIN < 0.40
    assert hw.Hotword.SPEAKER_WAKE_MIN > 0.10


def test_disabling_speaker_verify_disables_the_wake_gate(audio):
    """One switch, predictable: turning verification off must not leave a
    half-active gate silently suppressing wake words."""
    spk = FakeSpeaker(0.0)
    assert gate(spk, audio, enabled=False) is True
    assert spk.calls == 0
