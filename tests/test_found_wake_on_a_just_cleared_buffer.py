"""Regression test: a wake word accepted on a buffer that had just been cleared.

DEFECT (jarvis/hotword.py _listen_loop):

The listener clears its rolling buffer and resets the OWW model whenever the
mic stream reopens after recording. The wake decision that follows judges
whatever little audio has accumulated since, and the speaker gate behind it
CANNOT run below jarvis.speaker.MIN_AUDIO_SECONDS (1.0 s): _extract_embedding
returns None, score() returns None, and _speaker_ok deliberately fails OPEN.

So in the window right after a resume, the one gate that checks WHO spoke is
structurally bypassed -- it does not reject, it abstains, and abstention means
yes. From /tmp/vss_voice/jarvis.log, 2026-08-28:

    16:41:32.544  Stopped: 4.6s audio          (recording ends)
    16:41:32.610  Hotword stream resumed        (buf.clear(), model.reset())
    16:41:32.959  wake speaker check unavailable -- waking anyway
    16:41:32.959  Hotword detected (score=0.977)
    16:41:32.959  hotword ignored: still transcribing the previous clip

0.349 s of audio -- barely a third of what the gate needs. The app then
correctly refused the wake as busy and put "One moment — still on the last
one" on screen, which is how the user noticed.

The fix is not to make the gate fail shut (a wake word that cannot fire is
worse; see test_hotword_gating) but to refuse to DECIDE until there is enough
audio for the existing gate to do its job.
"""
import numpy as np
import pytest

from jarvis import hotword as hw
from jarvis.hotword import WAKE_MIN_AUDIO_SECONDS as MIN_AUDIO_SECONDS


def test_the_threshold_is_tied_to_what_the_speaker_gate_needs():
    # If these drift apart, the gate silently starts abstaining again.
    from jarvis.speaker import MIN_AUDIO_SECONDS as VERIFY_MIN
    # the wake floor may lead the verifier's, never trail it
    assert hw.WAKE_MIN_AUDIO_SECONDS >= VERIFY_MIN


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_the_logged_failure_is_rejected(rate):
    # 0.349 s, the exact buffer the 16:41:32.959 wake was accepted on
    assert not hw.wake_audio_sufficient(int(rate * 0.349), rate)


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_a_full_buffer_is_accepted(rate):
    # the steady state: the ring buffer holds 2 s whenever the stream is live
    assert hw.wake_audio_sufficient(int(rate * 2.0), rate)


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_exactly_the_gate_minimum_is_enough(rate):
    # The WAKE gate's own floor (1.5 s since 2026-08-31), not the
    # verifier's 1.0 s: a wake buffer is scored on whatever speech it
    # holds, and simulated false rejects were 19.6% at 1.0 s vs 6.2% at 1.5.
    assert hw.wake_audio_sufficient(int(rate * MIN_AUDIO_SECONDS), rate)
    assert not hw.wake_audio_sufficient(int(rate * (MIN_AUDIO_SECONDS - 0.2)), rate)


def test_an_empty_buffer_is_never_enough():
    assert not hw.wake_audio_sufficient(0, 44100)


def test_the_gate_really_does_abstain_below_the_minimum(monkeypatch):
    """The premise of the fix: too-short audio yields None, not a rejection.

    The model is stubbed present so this isolates the length guard -- the
    point is that the gate ABSTAINS on short audio, and _speaker_ok turns
    an abstention into a wake.
    """
    from jarvis.speaker import SpeakerVerifier
    sv = SpeakerVerifier()
    monkeypatch.setattr(sv, "_ensure_model", lambda: True)
    short = np.zeros(int(16000 * 0.349), dtype=np.float32)
    assert sv._extract_embedding(short) is None
