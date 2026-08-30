"""Stopping when the user stops, even while music keeps the room loud.

The energy detector cannot tell speech from a stereo, so background audio
holds a recording open. _check_speaker_silence exists for exactly that -- but
it polled every 3 s, a literal port of the monolith's root.after(3000) chain.
Measured on a real capture with music playing:

    09.667 recording started
    12.681 first check (3 s warm-up)   0.346  -- user talking
    15.752 check                      -0.038  -- miss 1/2
    18.757 check                       0.065  -- miss 2, timer finally starts
    21.661 energy path stopped it at 11.9 s

The 2.5 s voice-ID timer could not even begin before 9 s, so the mechanism
never once fired: all 30 stops in the log came from the energy path. An ECAPA
verify measures ~5 ms, so the interval was never a cost decision.
"""
import threading
import time

import numpy as np

import jarvis.recorder as recorder_mod
from jarvis.recorder import Recorder


def test_the_voice_id_stop_can_beat_the_energy_stop():
    """The arithmetic that decides whether this feature can ever run.

    Worst case to START the timer is warm-up plus the two consecutive misses
    _on_speaker_silence_result requires; the timer then runs for the silence
    timeout. That total must land inside a normal utterance, not after it.
    """
    warmup = Recorder._SPEAKER_WARMUP_S
    poll = Recorder._SPEAKER_POLL_S
    silence_timeout = 2.5                      # CONFIG.silence_timeout default
    start_timer_at = warmup + 2 * poll
    stops_at = start_timer_at + silence_timeout
    assert start_timer_at <= 4.0, f"timer cannot start until {start_timer_at}s"
    assert stops_at <= 7.0, f"voice-ID stop at {stops_at}s is still too late"
    # the old values could not: 3.0 + 2*3.0 = 9.0s before the timer even began
    assert start_timer_at < 3.0 + 2 * 3.0


def test_the_judged_window_is_recent_enough_to_notice_silence():
    """The window is also the lag: a speaker who stopped stays 'present'
    until their voice leaves the trailing span being judged."""
    assert Recorder._SPEAKER_WINDOW_S <= 2.0
    assert Recorder._SPEAKER_WINDOW_S >= 1.0, "ECAPA needs ~1s to say anything"


def _recorder_mid_capture(rate=16000, frame=1024, seconds=12.8):
    """A Recorder past warm-up with `seconds` of `frame`-sized chunks (NOT the
    0.1 s blocksize the old index math assumed)."""
    rec = object.__new__(Recorder)
    rec.recording = True
    rec._record_rate = rate
    rec._record_start_time = None                 # skips the warm-up gate
    rec._audio_frames = [np.ones((frame, 1), dtype=np.float32)] * int(rate * seconds / frame)
    rec._on_speaker_silence_result = lambda m, sc: None
    rec._resample_to_16k = lambda a: a            # already 16 k here
    return rec


class _Verifier:
    enrolled = True
    _model_failed = False

    def __init__(self, hold=None):
        self.seen, self.done, self.hold = [], threading.Event(), hold

    def verify(self, audio):
        self.seen.append(len(audio))
        if self.hold is not None:
            self.hold.wait(2)
        self.done.set()
        return True, 1.0


def test_the_window_is_taken_by_samples_not_by_frame_count(monkeypatch):
    """Drives Recorder._check_speaker_silence itself. The first version of
    this test copied the tail-walk into the test body and asserted on the
    copy, so it passed with the production method stubbed to a no-op."""
    monkeypatch.setattr(recorder_mod.CONFIG, "speaker_verify", True)
    rec = _recorder_mid_capture()
    rec.speaker_verifier = v = _Verifier()
    rec._check_speaker_silence()
    assert v.done.wait(2), "verify never ran"
    want = int(16000 * Recorder._SPEAKER_WINDOW_S)
    assert v.seen == [want], f"judged {v.seen} samples, wanted {want}"


def test_only_one_check_is_in_flight_at_a_time(monkeypatch):
    """At 1 Hz a slow verify must not stack threads that all mutate the miss
    counters and may all call stop(): a late check is skipped, not queued."""
    monkeypatch.setattr(recorder_mod.CONFIG, "speaker_verify", True)
    rec = _recorder_mid_capture()
    gate = threading.Event()
    rec.speaker_verifier = v = _Verifier(hold=gate)
    rec._check_speaker_silence()                  # first: blocks in verify
    for _ in range(50):
        if v.seen:
            break
        time.sleep(0.01)
    rec._check_speaker_silence()                  # second: must be skipped
    rec._check_speaker_silence()
    gate.set()
    assert v.done.wait(2)
    for _ in range(200):                              # until the worker's finally
        if not rec._speaker_check_lock.locked():
            break
        time.sleep(0.01)
    assert not rec._speaker_check_lock.locked()
    assert len(v.seen) == 1, f"{len(v.seen)} verifies ran concurrently"
    rec._check_speaker_silence()                  # released: runs again
    for _ in range(50):
        if len(v.seen) == 2:
            break
        time.sleep(0.01)
    assert len(v.seen) == 2


def test_a_dead_model_is_not_polled(monkeypatch):
    """speaker_verify is forced on so the ONLY thing short-circuiting the
    poll is _model_failed -- without that, a box with the feature off in
    voice_settings.json passed this test for the wrong reason."""
    monkeypatch.setattr(recorder_mod.CONFIG, "speaker_verify", True)
    rec = _recorder_mid_capture()
    v = _Verifier()
    v._model_failed = True
    rec.speaker_verifier = v
    rec._check_speaker_silence()
    assert not v.done.wait(0.3) and v.seen == []
    v._model_failed = False                           # control: the gate under test
    rec._check_speaker_silence()
    assert v.done.wait(2) and v.seen


def test_a_failure_before_the_thread_starts_releases_the_lock(monkeypatch):
    """concatenate/resample raising between acquire and the worker's own
    finally would hold the lock for the life of the Recorder."""
    monkeypatch.setattr(recorder_mod.CONFIG, "speaker_verify", True)
    rec = _recorder_mid_capture()
    rec.speaker_verifier = v = _Verifier()
    rec._resample_to_16k = lambda a: (_ for _ in ()).throw(RuntimeError("scipy exploded"))
    try:
        rec._check_speaker_silence()
    except RuntimeError:
        pass
    assert not rec._speaker_check_lock.locked(), "lock leaked past the failure"
    rec._resample_to_16k = lambda a: a
    rec._check_speaker_silence()
    assert v.done.wait(2)
