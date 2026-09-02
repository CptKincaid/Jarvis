"""2026-09-02 08:56:18 -- the mixer ducked and never came back up.

    08:56:18.754  jarvis.tts    speech complete
    08:56:18.843  jarvis.mixer  mixer: restored 1 stream(s)
    08:56:19.079  jarvis.mixer  mixer: ducked 1 stream(s) to 30%
    ... nothing for 17 s, until the next utterance's cycle cleared it

The re-duck 236 ms AFTER the falling edge is not a recording starting --
there is no "Recording started" anywhere in that window of the live log.
It is jarvis/tts.py's amplitude feeder. Every tick of `_feed_amp`, and its
tail, published a hard-coded `SpeakingState(active=True)` from the
`tts-amp-feeder` thread, unordered against the worker thread's
`SpeakingState(active=False)`. Whichever landed last won, and nothing ever
clears a re-asserted True: the mixer stays ducked, MainWindow._speaking
stays True (the pill sticks on "Speaking" -- his words), and
JarvisApp._tts_active stays set, which suppresses the listening nudge and
the guest-wake reply. The healthy 08:55:50 utterance shows the other order
from the same code.

THE CONTRACT PINNED HERE: the worker's falling edge is the single authority
on `active`. The feeder's job is the avatar's mouth (amplitude), and once
the burst is closed it says nothing at all -- in EITHER interleaving.
"""
import threading
import time

import pytest

from jarvis.events import SpeakingState, bus
from jarvis.mixer import RoomMixer
from jarvis.tts import TTS


def _feeder(envelope):
    """A real TTS with only the feeder's state, so _run_amp_feeder is the
    shipped function and not a paraphrase of it."""
    tts = TTS.__new__(TTS)
    tts._amp_gen = 0
    tts._amp_playing = False
    tts._current_amp = 0.0
    tts._stop_flag = False
    tts._burst_announced = True
    tts._speaking = True
    tts._burst_closed = False
    tts._edge_lock = threading.Lock()
    tts._mark_audio = lambda: None
    return tts


def _close_burst(tts):
    """Exactly what _worker_loop's finally does at the falling edge."""
    with tts._edge_lock:
        tts._burst_closed = True
        bus.publish(SpeakingState(active=False, amplitude=0.0))


@pytest.fixture
def mixer():
    m = RoomMixer(run=lambda *a, **kw: None)
    bus.subscribe(SpeakingState, m.on_speaking)
    yield m
    bus.unsubscribe(SpeakingState, m.on_speaking)


def test_a_feeder_still_draining_does_not_re_assert_the_burst(mixer):
    """Every LOOP tick is a SpeakingState(active=True) as well as the tail,
    so a feeder that outlives the falling edge re-ducks over and over."""
    tts = _feeder(None)
    tts._run_amp_feeder(iter([0.4] * 32))        # 2.56 s of envelope
    time.sleep(0.5)
    _close_burst(tts)
    time.sleep(2.4)
    assert mixer._holds == set(), (
        "the mixer stayed ducked after 'speech complete' -- he heard the "
        "duck for 17 s and the pill said Speaking")


def test_the_tail_landing_after_the_falling_edge_is_silent(mixer):
    """The 08:56:19.079 ordering, made deterministic: the tail is held
    until after the worker has published the falling edge."""
    released = threading.Event()

    def source():
        yield 0.4
        released.wait(5.0)      # the tail runs only once we say so
        return

    tts = _feeder(None)
    tts._run_amp_feeder(source())
    time.sleep(0.3)
    _close_burst(tts)
    time.sleep(0.1)
    assert mixer._holds == set()             # restored, as the log shows
    released.set()                           # ...and now the tail lands
    time.sleep(0.4)
    assert mixer._holds == set(), (
        "the feeder's tail re-asserted active=True 236 ms after the "
        "falling edge and nothing ever cleared it")


def test_the_falling_edge_wins_when_it_lands_first(mixer):
    """The healthy interleaving keeps working: restore, and stay restored."""
    tts = _feeder(None)
    tts._run_amp_feeder(iter([0.4] * 8))
    time.sleep(0.2)
    assert mixer._holds == {"speaking"}      # ducked while he is talking
    _close_burst(tts)
    time.sleep(0.9)
    assert mixer._holds == set()


def test_a_live_burst_still_ducks_and_the_tail_does_not_end_it(mixer):
    """Mid-burst the tail must NOT be read as a falling edge either: the
    feeder's tail runs at the end of every CHUNK, not of the burst."""
    tts = _feeder(None)
    tts._run_amp_feeder(iter([0.4] * 3))
    time.sleep(0.6)                          # tail has run; burst not closed
    assert mixer._holds == {"speaking"}
    _close_burst(tts)
    time.sleep(0.2)
    assert mixer._holds == set()
