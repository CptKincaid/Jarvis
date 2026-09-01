"""The first "Jarvis" after a reply, and the phantom wake that hid it.

DEFECT (jarvis/hotword.py _listen_loop), reported 2026-08-31:

    "even then I did have to say Jarvis multiple times. When he heard me once
     he kept hearing me but he just didn't listen on the first Jarvis."

/tmp/vss_voice/jarvis.log carried 130 "Hotword detected" against 52 "wake
held: 0.32 s buffered since the stream reopened", and a held wake was simply
dropped by `continue`. NONE of the 52 ever recovered: 0 had a detection within
one second, and the median gap to the next one was 46 s.

TWO faults, in opposite directions, and each one hid the other.

1. FAIL-SHUT WHERE THE CONTRACT SAYS FAIL-OPEN. `continue` discards a wake
   openWakeWord has already declared, betting a later frame will fire again.
   It will not: the activation decays within a few 80 ms frames while the ring
   buffer needs a further 1.18 s to reach WAKE_MIN_AUDIO_SECONDS. CLAUDE.md is
   explicit that this layer fails OPEN -- "an unwakeable assistant is worse
   than an over-eager one" -- and the comment above the `continue` even says
   the gate "abstains (and so admits)". The code did the opposite.

2. THE 0.32 s WAKES WERE NOT A PERSON. openwakeword 0.4.0's Model.reset()
   clears ONLY prediction_buffer (model.py:152-154). The audio lives in
   preprocessor.feature_buffer -- 120 frames, ~10 s -- and a pause does not
   touch it. Four fresh frames after a resume, the 16-frame window hey_jarvis
   scores is 12 frames of PRE-pause audio plus 4 new ones: the wake word the
   user said BEFORE the pause, re-fired at 0.98 with nobody speaking. Replaying
   his own ~/.aiws_trainer/wakeword_training clips through the real model:
   10 of 27 pause/resume replays re-fired, every single one at +0.32 s.

That is why the log's held wakes cluster at 0.32 s and 0.33-0.45 s after
"Hotword stream resumed" -- it is arithmetic, not a speaker. The 2026-08-28
`continue` suppressed that symptom, and the sticky _short_wake_logged latch
then swallowed the log line for every LATER hold in the same turn, which is
exactly where the user's real first "Jarvis" landed.

The fix is both: clear oww's audio state on resume so the phantom cannot fire,
and admit a genuinely short wake with the speaker gate abstaining. The
transcript gate (app._process_audio -> speaker.filter_segments) still fails
SHUT, so admitting a wake never admits a stranger's command.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import hotword as hw


# ====================================================================
# the hold/admit decision
# ====================================================================
@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_the_logged_shape_is_admitted_immediately(rate):
    """0.32 s -- the figure in 47 of the 52 held lines. 1.18 s short of the
    gate, far beyond any grace worth paying: decide now, gate abstains."""
    assert hw.wake_hold_seconds(int(rate * 0.32), rate) == 0.0


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_a_nearly_full_buffer_is_worth_waiting_for(rate):
    """0.96 s, the 15:44:11 hold: it completed 107 ms later and got a real
    score. Waiting buys a verified wake for a fraction of a second."""
    wait = hw.wake_hold_seconds(int(rate * 0.96), rate)
    assert 0.0 < wait <= hw.WAKE_HOLD_MAX_SECONDS
    assert wait == pytest.approx(hw.WAKE_MIN_AUDIO_SECONDS - 0.96, abs=0.01)


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_a_full_buffer_never_waits(rate):
    assert hw.wake_hold_seconds(int(rate * 2.0), rate) == 0.0


def test_the_wait_is_never_longer_than_the_cap():
    """An unbounded wait is how a wake gets lost; the cap is the guarantee."""
    for frac in np.arange(0.0, 1.55, 0.05):
        assert hw.wake_hold_seconds(int(16000 * frac), 16000) <= \
            hw.WAKE_HOLD_MAX_SECONDS


def test_the_cap_is_short_enough_to_stay_conversational():
    # the wake->listen path is already ~1.3 s; a grace near a second would
    # cost more than the abstention it is buying.
    assert 0.0 < hw.WAKE_HOLD_MAX_SECONDS <= 0.6


# ====================================================================
# openWakeWord stream state
# ====================================================================
class _FakePre:
    """Shaped like openwakeword.utils.AudioFeatures."""

    def __init__(self):
        self.feature_buffer = np.zeros((120, 96), dtype=np.float32)
        self.melspectrogram_buffer = np.ones((76, 32), dtype=np.float32)
        self.raw_data_buffer = __import__("collections").deque(maxlen=160000)
        self.accumulated_samples = 0


class _FakeModel:
    def __init__(self, pre=True):
        self.preprocessor = _FakePre() if pre else None
        self.resets = 0

    def reset(self):
        self.resets += 1


def test_blank_state_is_snapshotted_and_restored():
    m = _FakeModel()
    blank = hw.capture_oww_blank(m)
    # ...the pause: a wake word is now sitting in the feature buffer
    m.preprocessor.feature_buffer = np.full((120, 96), 7.0, dtype=np.float32)
    m.preprocessor.melspectrogram_buffer = np.full((90, 32), 3.0, np.float32)
    m.preprocessor.raw_data_buffer.extend([0.5] * 4000)
    m.preprocessor.accumulated_samples = 640

    assert hw.reset_oww_stream(m, blank) is True
    assert np.array_equal(m.preprocessor.feature_buffer,
                          blank["feature_buffer"])
    assert np.array_equal(m.preprocessor.melspectrogram_buffer,
                          blank["melspectrogram_buffer"])
    assert len(m.preprocessor.raw_data_buffer) == 0, \
        "stale samples bleed three frames of pre-pause audio back in"
    assert m.preprocessor.accumulated_samples == 0
    assert m.resets == 1, "the score history must still be reset"


def test_the_snapshot_is_a_copy_not_a_view():
    """A view would track the very audio it is supposed to undo."""
    m = _FakeModel()
    blank = hw.capture_oww_blank(m)
    m.preprocessor.feature_buffer[:] = 9.0
    assert not np.any(blank["feature_buffer"] == 9.0)


def test_reset_degrades_safely_without_a_preprocessor():
    """A future oww must never take the wake word down."""
    m = _FakeModel(pre=False)
    assert hw.capture_oww_blank(m) is None
    assert hw.reset_oww_stream(m, None) is False
    assert m.resets == 1, "still reset what we can"


def test_reset_survives_a_broken_preprocessor():
    m = _FakeModel()
    blank = hw.capture_oww_blank(m)
    m.preprocessor.raw_data_buffer = object()      # no .clear()
    assert hw.reset_oww_stream(m, blank) is False  # reported, not raised


# ====================================================================
# the real shape: driving the listener loop
# ====================================================================
class _Stream:
    def __init__(self, callback):
        self.callback = callback
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.started = False


class _FakeSD:
    """Just enough sounddevice. No device is ever opened."""

    def __init__(self, rate):
        self.rate = rate
        self.streams = []
        self.opens = 0

    def query_devices(self, idx, kind):
        return {"default_samplerate": self.rate}

    def InputStream(self, **kw):
        self.opens += 1
        s = _Stream(kw["callback"])
        self.streams.append(s)
        return s


class _Clock:
    """Virtual time that also delivers the microphone.

    Every sleep advances the clock AND pushes exactly that much audio into
    the open stream, so "the buffer fills in real time" holds in the test the
    same way it holds on the box -- which is the whole premise of the wait.
    """

    def __init__(self, sd, budget=60.0):
        self.sd = sd
        self.t = 1000.0
        self.budget = budget
        self.mute = False       # a microphone that stops delivering

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        self.budget -= seconds
        if self.budget <= 0:
            raise _Done()
        n = int(self.sd.rate * seconds)
        st = self.sd.streams[-1] if self.sd.streams else None
        if st is not None and st.started and n > 0 and not self.mute:
            st.callback(np.zeros((n, 1), dtype=np.float32), n, None, None)


class _Done(Exception):
    """Ends the listener loop without touching Hotword.active."""


class _ScriptedModel:
    """Returns hey_jarvis hits starting `hit_after` predicts past a reset.

    `on_predict` is the seam the arbiter occupies on the real box: a hook that
    can pause and resume the listener partway through the stream, which is the
    only way to reach the resume branch (the loop clears `_reopen` itself on
    entry, so it cannot be pre-set from outside).
    """

    def __init__(self, hit_after=None, pre=True):
        self.preprocessor = _FakePre() if pre else None
        self.hit_after = hit_after
        self.since_reset = 0
        self.predicts = 0
        self.resets = 0
        self.on_predict = None
        self.models = {"hey_jarvis": None}
        self.custom_verifier_models = {}

    def reset(self):
        self.resets += 1
        self.since_reset = 0

    def predict(self, chunk):
        self.predicts += 1
        self.since_reset += 1
        if self.on_predict is not None:
            self.on_predict(self.predicts)
        if self.hit_after is not None and self.since_reset > self.hit_after:
            return {"hey_jarvis": 0.95}
        return {"hey_jarvis": 0.0}


class _Speaker:
    """Mirrors SpeakerVerifier: abstains (None) below MIN_AUDIO_SECONDS."""

    MIN = 1.0

    def __init__(self, value=0.72):
        self.is_enrolled = True
        self.value = value
        self.scored = []

    def score(self, audio_16k):
        self.scored.append(len(audio_16k) / 16000.0)
        if len(audio_16k) < int(16000 * self.MIN):
            return None                    # too short to embed: abstain
        return self.value


class Harness:
    """Drives the real _listen_loop against a fake device and a virtual clock.

    `pause_after` reproduces a turn: at that predict the mic arbiter takes the
    microphone (TTS or recording) and hands it straight back, exactly as
    pause()/resume() are wired to it in Hotword.__init__.
    """

    def __init__(self, monkeypatch, rate=16000, hit_after=None,
                 speaker=None, budget=60.0, pause_after=None,
                 on_pause=None):
        self.sd = _FakeSD(rate)
        self.clock = _Clock(self.sd, budget)
        self.model = _ScriptedModel(hit_after)
        self.detected = []
        self.guests = []
        self.paused_at = None
        self._pause_after = pause_after
        self._on_pause = on_pause
        monkeypatch.setitem(__import__("sys").modules, "sounddevice", self.sd)
        monkeypatch.setattr(hw.time, "sleep", self.clock.sleep, raising=False)
        monkeypatch.setattr(hw.time, "monotonic", self.clock.monotonic,
                            raising=False)
        monkeypatch.setattr(hw.CONFIG, "speaker_verify", True)
        self.speaker = speaker if speaker is not None else _Speaker()
        self.hw = hw.Hotword(None, lambda: 0, self._on_detect,
                             speaker=self.speaker, on_guest=self.guests.append)
        self.hw._model = self.model        # skip the real oww load entirely
        self.hw.active = True
        if pause_after is not None:
            self.model.on_predict = self._maybe_pause

    def _maybe_pause(self, n):
        if self.paused_at is not None or n < self._pause_after:
            return
        self.paused_at = n
        self.hw.pause()                    # the arbiter takes the mic...
        if self._on_pause is not None:
            self._on_pause(self)
        self.clock.sleep(4.0)              # ...for a whole turn...
        self.hw.resume()                   # ...and hands it back

    def _on_detect(self, score):
        self.detected.append(score)
        self.hw.active = False             # one turn is all the test needs

    def run(self):
        try:
            self.hw._listen_loop()
        except _Done:
            pass
        return self


@pytest.fixture(autouse=True)
def _restore_config():
    was = hw.CONFIG.speaker_verify
    yield
    hw.CONFIG.speaker_verify = was


def test_a_short_wake_just_after_a_resume_now_wakes_him(monkeypatch):
    """THE REPORTED BUG, in its measured shape.

    A turn ends, the stream reopens with an empty ring buffer, and four 80 ms
    frames later openWakeWord declares a wake -- 0.32 s of audio, the exact
    figure in 47 of the 52 held lines in jarvis.log. Before the fix the loop
    hit `continue`, the activation decayed, and the wake was gone for good
    (0 of 50 such wakes in the log ever recovered within a second). It must
    wake him.
    """
    h = Harness(monkeypatch, hit_after=2, pause_after=3, budget=12.0).run()

    assert h.paused_at, "the turn never happened; the test proves nothing"
    assert h.detected, ("a wake detected 0.32 s after the resume was dropped; "
                        "this layer fails OPEN (CLAUDE.md)")
    assert h.detected[0] == pytest.approx(0.95)
    # ...and it was admitted by ABSTENTION, not by a speaker verdict
    assert h.speaker.scored[-1] < hw.WAKE_MIN_AUDIO_SECONDS
    assert h.speaker.score(np.zeros(int(16000 * 0.32), np.float32)) is None


def test_the_admitted_wake_carries_only_post_resume_audio(monkeypatch):
    """The gate must never be handed audio from before the pause.

    Pre-roll would let stale speech "verify" a wake it did not come from --
    and after a TTS burst the newest thing in that buffer is the user's LAST
    command, which scores as a match. Clearing on resume is deliberate.
    """
    h = Harness(monkeypatch, hit_after=2, pause_after=3, budget=12.0).run()
    assert h.detected
    assert h.speaker.scored[-1] == pytest.approx(0.32, abs=0.05),         "the wake was judged on audio from before the pause"


def test_openwakeword_audio_state_is_cleared_on_resume(monkeypatch):
    """The phantom's root cause: reset() alone leaves ~10 s of features.

    Whatever oww heard before the pause must be gone, or four frames later it
    scores the pre-pause wake word again and fires with nobody speaking.
    """
    def pollute(h):
        pre = h.model.preprocessor
        pre.feature_buffer = np.full((120, 96), 5.0, np.float32)
        pre.raw_data_buffer.extend([0.3] * 8000)
        pre.accumulated_samples = 640

    h = Harness(monkeypatch, hit_after=None, pause_after=3, budget=12.0,
                on_pause=pollute).run()

    assert h.paused_at
    pre = h.model.preprocessor
    assert not np.any(pre.feature_buffer == 5.0),         ("oww re-fires the pre-pause wake word 0.32 s after every resume "
         "while its feature buffer survives the pause")
    assert len(pre.raw_data_buffer) == 0
    assert pre.accumulated_samples == 0


def test_a_nearly_full_buffer_still_waits_for_a_real_score(monkeypatch):
    """The honest path survives: when the shortfall fits in the grace, the
    wake is HELD, the buffer fills, and the gate scores real audio -- the
    0.88 s / 0.96 s shape, the only two of 52 that were a real part-fill."""
    assert hw.wake_hold_seconds(int(16000 * 1.04), 16000) > 0.0
    h = Harness(monkeypatch, hit_after=11, budget=12.0).run()
    assert h.detected, "a held wake must still fire once its audio arrives"
    assert h.speaker.scored[-1] >= hw.WAKE_MIN_AUDIO_SECONDS,         "the whole point of waiting is a scored wake, not an abstention"


def test_a_held_wake_is_fired_even_if_its_buffer_never_fills(monkeypatch):
    """The deadline is the guarantee. A wake parked waiting for audio that
    stops arriving is the same lost wake in a new place."""
    h = Harness(monkeypatch, hit_after=11, budget=12.0)
    fired = {}

    def freeze(n):
        # the microphone goes quiet the moment the wake is held
        if h.model.since_reset > h.model.hit_after and "at" not in fired:
            fired["at"] = n
            h.clock.mute = True

    h.model.on_predict = freeze
    h.run()
    assert fired, "the wake was never held"
    assert h.detected, "a held wake outlived its deadline and was lost"
    assert h.speaker.scored[-1] < hw.WAKE_MIN_AUDIO_SECONDS,         "it should have been admitted on short audio, gate abstaining"


def test_a_stranger_is_still_suppressed_at_full_length(monkeypatch):
    """Admitting SHORT wakes must not soften the gate when it can judge."""
    h = Harness(monkeypatch, hit_after=25, speaker=_Speaker(0.02),
                budget=6.0).run()
    assert not h.detected, "a scored non-match must still be suppressed"
    assert h.guests, "and offered the guest path"


def test_no_wake_is_ever_invented(monkeypatch):
    """The admit path must not turn silence into a wake."""
    h = Harness(monkeypatch, hit_after=None, budget=5.0).run()
    assert not h.detected
    assert h.model.predicts > 10, "the loop really did run"
