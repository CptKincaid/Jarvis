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
        self.floors = []

    def score(self, audio_16k, min_seconds=None):
        self.calls += 1
        self.floors.append(min_seconds)
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


# ---------------------------------- a rejection needs evidence to reject on
# The gate may only REFUSE him for "that is not you", never for "I could not
# tell".  Two buffers cannot carry that evidence, both measured 2026-09-02
# against his own voiceprint on this box:
#
#   too little speech -- his wake clips hold 0.40-0.88 s of trimmed speech in
#   the 2 s buffer, and one of them (hey_jarvis_04, 0.66 s) scores -0.059,
#   deep inside the impostor band.  Scoring those and rejecting on the result
#   would cost him real wakes.
#
#   a competing bed -- over room noise his own wake word scores -0.074..0.321
#   while the bed ALONE scores -0.114..0.156, and 6 of his 10 land inside the
#   bed's range.  The best bar that exists anywhere is 0.090 and it still
#   refuses 4 of his 10; best-of-windows and a per-buffer lift statistic were
#   both measured and separate no better.
def _speech(seconds, amp=0.2, seed=0):
    """Syllable-like speech: broadband noise under a 4 Hz envelope, so
    speaker.trim_silence can find its endpoints (flat noise it cannot)."""
    rng = np.random.default_rng(seed)
    n = int(16000 * seconds)
    t = np.arange(n) / 16000
    env = np.where(np.cos(2 * np.pi * 4 * t) >= 0, 1.0, 0.1)
    return (rng.normal(0, amp, n) * env).astype(np.float32)


def _buffer(speech_s, seed=0):
    """A 2 s wake buffer holding `speech_s` of speech, the shape the ring
    buffer hands the gate."""
    pad = (2.0 - speech_s) / 2
    quiet = (np.random.default_rng(99).normal(0, 1e-4, int(16000 * pad))
             .astype(np.float32))
    return np.concatenate([quiet, _speech(speech_s, seed=seed), quiet])


def gate_with(speaker, audio, music, oww, native_rate=16000):
    hw.CONFIG.speaker_verify = True
    h = hw.Hotword.__new__(hw.Hotword)
    h._speaker = speaker
    h._music_playing = (lambda: music) if music is not None else None
    return h._speaker_ok(audio, native_rate, oww_score=oww)


def test_the_gate_asks_for_a_number_on_a_short_buffer(audio):
    """It used to take the module default and get None back on 6 of his 10
    own wake clips, so the gate was blind on most wakes and fell open."""
    import jarvis.speaker as speaker_mod
    spk = FakeSpeaker(0.4)
    gate_with(spk, audio, music=False, oww=0.9)
    assert spk.floors == [speaker_mod.MIN_SPEECH_SECONDS]


def test_too_little_speech_abstains_instead_of_refusing_him(audio):
    """hey_jarvis_04: 0.66 s of his own voice scoring -0.059.  Rejecting on
    that number is rejecting on noise."""
    short = _buffer(0.66)
    assert gate_with(FakeSpeaker(-0.059), short, music=False, oww=0.9) is True


def test_enough_speech_is_still_judged_on_the_score(audio):
    """hey_jarvis_09: 2.0 s of trimmed speech at -0.134.  That much audio is
    evidence, so the refusal stands."""
    assert gate_with(FakeSpeaker(-0.134), audio, music=False, oww=0.9) is False


def test_a_good_score_on_a_short_buffer_is_a_positive_recognition(audio):
    """The point of the lower floor: the 6 buffers that used to score None
    land at -0.059..0.273, so the gate can say "that is him" instead of only
    ever failing open on a None."""
    short = _buffer(0.52)
    assert gate_with(FakeSpeaker(0.273), short, music=False, oww=0.9) is True


# ------------------------------------------------- the gate while music plays
# 2026-09-01, Spotify on HPCOMPUTER: his own "Jarvis" scored 0.135 (oww 0.713)
# and 0.158 (oww 0.864) against the 0.25 bar, was refused, and got "I only
# answer to Hunter, sir" -- twice.  The first fix for that was a second bar at
# 0.10; it was wrong, because the bed alone reaches 0.156 and he falls to
# -0.074 over one.  0.10 is in fact the best bar that exists -- and it still
# refuses 4 of his 10.  There is no bar.  There is only "I cannot tell".
def test_his_voice_over_music_is_accepted(audio):
    assert gate_with(FakeSpeaker(0.135), audio, music=True, oww=0.713) is True
    assert gate_with(FakeSpeaker(0.158), audio, music=True, oww=0.864) is True


def test_the_same_score_without_music_is_still_a_stranger(audio):
    """The relief is a property of the room, not the default."""
    assert gate_with(FakeSpeaker(0.135), audio, music=False, oww=0.713) is False
    assert gate_with(FakeSpeaker(0.135), audio, music=None, oww=0.713) is False


def test_music_relief_does_not_wait_for_a_confident_wake_word(audio):
    """The 0.10 bar only applied when oww also cleared 0.6, and 9 of the 30
    suppressions on record sat below that -- hey_jarvis fires from 0.3.  A
    marginal wake word over music is still a wake word the gate cannot judge."""
    assert gate_with(FakeSpeaker(0.135), audio, music=True, oww=0.31) is True
    assert gate_with(FakeSpeaker(-0.098), audio, music=True, oww=0.31) is True


def test_the_cost_of_abstaining_over_music_is_a_false_wake_not_a_refusal(audio):
    """Stated plainly so nobody re-derives the 0.10 bar: music that trips oww
    on its own (-0.084, -0.098) now WAKES him instead of being suppressed.
    That is the trade -- the transcript gate in app.py still fails shut behind
    this one, and an untriggerable wake word is the worse failure."""
    assert gate_with(FakeSpeaker(-0.084), audio, music=True, oww=0.9) is True
    assert gate_with(FakeSpeaker(-0.084), audio, music=False, oww=0.9) is False


def test_abstention_still_fails_open_over_music(audio):
    assert gate_with(FakeSpeaker(None), audio, music=True, oww=0.9) is True


def test_a_broken_music_source_means_no_music(audio):
    def boom():
        raise RuntimeError("spotify token expired")
    hw.CONFIG.speaker_verify = True
    h = hw.Hotword.__new__(hw.Hotword)
    h._speaker = FakeSpeaker(0.135)
    h._music_playing = boom
    assert h._speaker_ok(audio, 16000, oww_score=0.9) is False


def test_there_is_one_bar_and_no_second_one():
    """The music bar was a threshold placed inside an overlap; a reader
    reaching for it again should find nothing there."""
    assert not hasattr(hw.Hotword, "SPEAKER_WAKE_MIN_MUSIC")
    assert hw.Hotword.SPEAKER_WAKE_MIN == 0.25


def test_every_scored_candidate_is_logged_on_one_line(audio, caplog):
    """The calibration record: speaker score, oww score, music state and the
    buffer's ambient level together, so the bar can be re-derived from the
    log instead of from memory."""
    audio = (np.random.default_rng(0).standard_normal(32000) * 0.03).astype(np.float32)
    with caplog.at_level("INFO", logger="jarvis.hotword"):
        gate_with(FakeSpeaker(0.135), audio, music=True, oww=0.713)
        gate_with(FakeSpeaker(0.135), audio, music=False, oww=0.713)
        gate_with(FakeSpeaker(None), audio, music=False, oww=0.713)
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("wake candidate:")]
    assert len(lines) == 3
    assert ("speaker=0.135 speech=2.00s oww=0.713 music=True rms=-30." in lines[0]
            and "-> abstain (music)" in lines[0])
    assert "music=False" in lines[1] and "gate=0.25 -> suppress" in lines[1]
    assert "speaker=none" in lines[2] and "-> abstain" in lines[2]
    # The seconds of speech the score was taken over decide half these
    # verdicts and were invisible until now; that omission cost a day.
    assert all("speech=2.00s" in ln for ln in lines)


def test_ambient_dbfs_reads_the_bed_under_the_voice():
    """Whole-buffer RMS vs the 20th-percentile frame: a word over silence has
    a floor near digital zero, a word over music has a floor near the music."""
    rng = np.random.default_rng(1)
    word = np.zeros(32000, np.float32)
    word[12000:20000] = (rng.standard_normal(8000) * 0.1).astype(np.float32)
    rms, floor = hw.ambient_dbfs(word, 16000)
    assert -30.0 < rms < -20.0
    assert floor == -120.0                       # silence between the words
    bed = (rng.standard_normal(32000) * 0.01).astype(np.float32)
    rms, floor = hw.ambient_dbfs(word + bed, 16000)
    assert -42.0 < floor < -38.0                 # the music, not the word
    assert hw.ambient_dbfs(np.zeros(0, np.float32), 16000) == (-120.0, -120.0)


def test_his_voice_over_music_wakes_him_through_the_real_loop(monkeypatch):
    """End to end through _listen_loop: the 0.135 that was refused on
    2026-09-01, with music known playing, reaches on_detect and never
    on_guest; the same run with the music flag off is the old refusal.

    The callable is passed to the real Hotword CONSTRUCTOR, the way app.py
    passes it, rather than poked onto the instance -- a kwarg the class
    silently ignored would otherwise look exactly like a pass here."""
    from tests.test_wake_after_resume import Harness, _Speaker

    asked = []
    h = Harness(monkeypatch, hit_after=25, speaker=_Speaker(0.135), budget=6.0,
                music_playing=lambda: bool(asked.append(1) or True))
    h.run()
    assert h.detected == [pytest.approx(0.95)]
    assert not h.guests
    assert asked, "the constructor's music_playing was never consulted"

    quiet = Harness(monkeypatch, hit_after=25, speaker=_Speaker(0.135),
                    budget=6.0, music_playing=lambda: False)
    quiet.run()
    assert not quiet.detected
    assert quiet.guests

    # And with nothing wired at all -- a box with no Spotify -- the gate is
    # the strict one it has always been.
    none = Harness(monkeypatch, hit_after=25, speaker=_Speaker(0.135), budget=6.0)
    none.run()
    assert not none.detected and none.guests
