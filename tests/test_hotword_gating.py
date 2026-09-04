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
        """Faithful about the ONE thing the gate's change turns on: the real
        SpeakerVerifier returns None when the TRIMMED audio is shorter than
        the caller's floor, which defaults to MIN_AUDIO_SECONDS.  A stub that
        always hands back a number cannot tell the new gate from a414152 --
        every short buffer scores there too, and a test asserting only the
        bool passes on both trees."""
        import jarvis.speaker as speaker_mod
        self.calls += 1
        self.floors.append(min_seconds)
        if isinstance(self._value, Exception):
            raise self._value
        floor = (speaker_mod.MIN_AUDIO_SECONDS if min_seconds is None
                 else min_seconds)
        if len(speaker_mod.trim_silence(audio_16k)) < 16000 * floor:
            return None
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
#   a competing bed -- but only a LOUD one, and the first writing of this
#   comment left the mix out.  His 10 wake clips mixed with
#   ~/.aiws_trainer/threshold_clips/room, scored against a pool built from
#   ~/.aiws_trainer/threshold_clips/me: the bed ALONE scores -0.035..0.154 at
#   every gain, while he scores -0.095..0.461 at x1 (1 of 10 inside the bed's
#   range, best bar 0.155 refusing 3/10 and admitting 0/12) and -0.040..0.384
#   at x4 (4 of 10 inside, best bar 0.065 refusing 3/10 and admitting 2/12).
#   So a bar separates under a quiet bed and stops separating around x4.  The
#   abstention is chosen for the loud case -- the one that actually refused
#   him on 2026-09-01 -- not because no bar ever works.
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


def verdict_of(caplog, speaker, audio, music, oww, native_rate=16000):
    """(wake?, the one candidate line).  A bare True is not enough to tell
    "that is him" from "I could not tell": below MIN_AUDIO_SECONDS both
    branches return True, so a test that only asserts the bool passes on the
    code that had no accept branch at all (checked against a414152)."""
    caplog.clear()
    with caplog.at_level("INFO", logger="jarvis.hotword"):
        ok = gate_with(speaker, audio, music, oww, native_rate)
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("wake candidate:")]
    assert len(lines) == 1, lines
    return ok, lines[0]


def _bed(n, amp=0.02, seed=7):
    """A room bed: broadband noise under a slow, uneven envelope, so its
    per-frame RMS has the spread real room noise has and flat noise has not.
    That spread is what defeats speaker.trim_silence -- the threshold is
    max(peak*0.08, p10*3), so a bed whose quiet tenth stays low while its
    body clears 8% of his peak lifts the trim's endpoints out to the room."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    env = 0.15 + np.abs(np.sin(2 * np.pi * 0.7 * t) * np.sin(2 * np.pi * 0.23 * t + 1.0))
    return (rng.normal(0, amp, n) * env).astype(np.float32)


def test_the_gate_asks_for_a_number_on_a_short_buffer(audio):
    """It used to take the module default and get None back on 6 of his 10
    own wake clips, so the gate was blind on most wakes and fell open."""
    import jarvis.speaker as speaker_mod
    spk = FakeSpeaker(0.4)
    gate_with(spk, audio, music=False, oww=0.9)
    assert spk.floors == [speaker_mod.MIN_SPEECH_SECONDS]


def test_too_little_speech_abstains_instead_of_refusing_him(caplog):
    """hey_jarvis_04: 0.66 s of his own voice scoring -0.057.  Rejecting on
    that number is rejecting on noise."""
    ok, line = verdict_of(caplog, FakeSpeaker(-0.057), _buffer(0.66),
                          music=False, oww=0.9)
    assert ok is True and "-> abstain (too little speech)" in line


def test_a_short_buffer_that_is_all_speech_abstains_too(caplog):
    """The ring buffer was cleared while he was already saying "Jarvis":
    0.80 s of voice, no quiet either side, buffer length a multiple of the
    20 ms frame (12800 samples, as every 44.1 kHz chunk resamples to).
    speech_bounds spans the whole buffer, so the old slice-length test read
    "trimmed=none" and a -0.057 on his own voice was SUPPRESSED (F48)."""
    audio = _speech(0.80)
    assert len(audio) % 320 == 0
    ok, line = verdict_of(caplog, FakeSpeaker(-0.057), audio,
                          music=False, oww=0.9)
    assert ok is True and "-> abstain (too little speech)" in line
    assert "trimmed=0.80s" in line


def test_enough_speech_is_still_judged_on_the_score(caplog):
    """hey_jarvis_09: 2.0 s of trimmed speech at -0.122.  That much audio is
    evidence, so the refusal stands."""
    ok, line = verdict_of(caplog, FakeSpeaker(-0.122), _buffer(1.6),
                          music=False, oww=0.9)
    assert ok is False and "-> suppress" in line and "trimmed=1.64s" in line


def test_a_good_score_on_a_short_buffer_is_a_positive_recognition(caplog):
    """The point of the lower floor: the 6 buffers that used to score None
    land at -0.057..0.280, so the gate can say "that is him" instead of only
    ever failing open on a None.

    Asserted on the VERDICT, because the bool cannot see it: this buffer
    abstains at any score, so `is True` alone passed unchanged on a414152,
    where there was no accept branch to reach."""
    ok, line = verdict_of(caplog, FakeSpeaker(0.273), _buffer(0.52),
                          music=False, oww=0.9)
    assert ok is True and "-> accept" in line


def test_the_lower_floor_bought_a_log_field_not_a_verdict(caplog):
    """Stated so nobody sells it as more than it is (2026-09-02 review).

    Under MIN_AUDIO_SECONDS of isolated speech the accept branch and the
    abstain branch both wake him, so the score changes only what the line
    says.  That IS worth having -- it is the only calibration record this
    gate will ever produce -- but it is not a decision."""
    good, gline = verdict_of(caplog, FakeSpeaker(0.30), _buffer(0.52),
                             music=False, oww=0.9)
    bad, bline = verdict_of(caplog, FakeSpeaker(-0.30), _buffer(0.52),
                            music=False, oww=0.9)
    assert good is bad is True, "the verdict must not depend on the score here"
    assert "-> accept" in gline and "-> abstain (too little speech)" in bline
    assert "speaker=0.300" in gline and "speaker=-0.300" in bline


def test_a_bed_can_still_refuse_him_on_speech_the_gate_never_measured(caplog):
    """The hole in the invariant, pinned rather than papered over.

    trim_silence's threshold is relative to the clip's own peak and floor, so
    an unflagged bed (a TV, a browser, a phone -- the media signal is Spotify
    ONLY) lifts the endpoints out to the room and the too-little-speech
    abstention quietly disengages.  Measured 2026-09-02 on his own clips:
    hey_jarvis_05 holds 0.52 s of speech and scores 0.277 dry, and reports
    1.46 s at 0.183 under a x4 room bed.  Same speech, opposite verdict.

    Not a regression -- a414152 suppressed these buffers too -- so this test
    pins the CURRENT truth, and fails the day someone closes the gap (the
    floor on the same log line is how: -49.9..-45.2 dBFS quiet against
    -44.0..-38.6 under a x2 bed)."""
    speech = _buffer(0.52)
    bedded = (speech + _bed(len(speech))).astype(np.float32)
    dry_ok, dry_line = verdict_of(caplog, FakeSpeaker(0.183), speech,
                                  music=False, oww=0.9)
    bed_ok, bed_line = verdict_of(caplog, FakeSpeaker(0.183), bedded,
                                  music=False, oww=0.9)
    assert dry_ok is True and "trimmed=0.56s" in dry_line
    assert bed_ok is False and "-> suppress" in bed_line
    assert "trimmed=1.12s" in bed_line, \
        "the bed doubled the reported seconds without adding any speech"


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


def test_every_scored_candidate_is_logged_on_one_line(caplog):
    """The calibration record: speaker score, the seconds the trim isolated,
    oww score, music state and the buffer's ambient level together, so the
    bar can be re-derived from the log instead of from memory."""
    flat = (np.random.default_rng(0).standard_normal(32000) * 0.03).astype(np.float32)
    with caplog.at_level("INFO", logger="jarvis.hotword"):
        gate_with(FakeSpeaker(0.135), flat, music=True, oww=0.713)
        gate_with(FakeSpeaker(0.135), flat, music=False, oww=0.713)
        gate_with(FakeSpeaker(None), flat, music=False, oww=0.713)
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("wake candidate:")]
    assert len(lines) == 3
    assert ("speaker=0.135 trimmed=none oww=0.713 music=True rms=-30." in lines[0]
            and "-> abstain (music)" in lines[0])
    assert "music=False" in lines[1] and "gate=0.25 -> suppress" in lines[1]
    assert "speaker=none" in lines[2] and "-> abstain" in lines[2]


def test_a_buffer_the_trim_could_not_cut_is_not_seconds_of_speech(caplog):
    """speaker.trim_silence returns the audio UNCHANGED when speech_bounds
    finds no endpoints, so the first version of this field printed
    "speech=2.00s" for digital silence and for a sustained tone -- and a
    calibration field that reports a trim failure as two seconds of speech is
    worse than no field.  It now reports what happened: nothing was cut.

    The VERDICT is deliberately unchanged.  An uncut buffer is not evidence
    of too little speech, so silence and a television still fall through to
    the bar instead of being waved past it."""
    for name, buf in (("silence", np.zeros(32000, dtype=np.float32)),
                      ("tone", (0.1 * np.sin(2 * np.pi * 440 *
                                             np.arange(32000) / 16000)
                                ).astype(np.float32))):
        ok, line = verdict_of(caplog, FakeSpeaker(-0.034), buf,
                              music=False, oww=0.9)
        assert "trimmed=none" in line, f"{name}: {line}"
        assert ok is False and "-> suppress" in line, f"{name}: {line}"
        assert "an untrimmable buffer" in caplog.text, \
            "the suppression line must not claim seconds it never measured"


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
