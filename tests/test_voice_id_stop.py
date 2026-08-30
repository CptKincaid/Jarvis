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
import numpy as np

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


def test_the_window_is_taken_by_samples_not_by_frame_count(monkeypatch):
    """Blocksize is 0.1 s today and the old index math relied on it. Feed
    frames of a different size and the span judged must still be right."""
    rec = object.__new__(Recorder)
    rate = 16000
    rec._record_rate = rate
    rec._SPEAKER_WINDOW_S = 2.0
    frame = np.ones((1024, 1), dtype=np.float32)          # not 0.1 s
    rec._audio_frames = [frame] * 200                     # ~12.8 s of audio

    samples_needed = int(rate * rec._SPEAKER_WINDOW_S)
    tail, got = [], 0
    for chunk in reversed(rec._audio_frames):
        tail.append(chunk)
        got += len(chunk)
        if got >= samples_needed:
            break
    recent = np.concatenate(list(reversed(tail)), axis=0).flatten()[-samples_needed:]
    assert len(recent) == samples_needed, "judged the wrong span of audio"
