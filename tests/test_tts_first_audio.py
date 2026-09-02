"""The honest first-audio mark, play-while-rendering for Fish, and the
cancel-current-keep-queue primitive the reader's "skip" is built on.

WHY. The turn ledger's "wait" ends on SpeakingState(active=True). Until
2026-08-30 the worker published that BEFORE _speak_sync rendered anything,
so every "wait 1.32 s" the user tuned against was really ~1.7-1.9 s: the
whole first chunk was rendered to a file before paplay ever started, and the
live log showed the "turn:" line landing 13-90 ms after "speaking (fish)".
The mark now goes out when a player is spawned. Fish's 186 ms
time-to-first-audio was likewise never heard, because _fish_stream wrote
the entire chunk to disk first; its bytes now go to paplay's stdin as they
arrive, teed into the file the speech cache needs.

No audio is played: players are faked at the Popen seam.
"""
import io
import os
import threading
import time
import wave

import pytest

from jarvis import tts as tts_mod
from jarvis.events import SpeakingState, bus
from jarvis.tts import TTS


def wav_bytes(seconds=0.4, rate=16000, amp=8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(rate * seconds)
        w.writeframes(b"".join(int(amp * ((i // 40) % 2 * 2 - 1)).to_bytes(
            2, "little", signed=True) for i in range(n)))
    return buf.getvalue()


def write_wav(path, seconds=0.05):
    with open(path, "wb") as fh:
        fh.write(wav_bytes(seconds))


def wait_until(pred, timeout=3.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.01)
    return pred()


class FakeProc:
    """A player that 'exits' once its stdin is closed (or at once for the
    file chain), recording when it was spawned and what it was fed."""
    spawned: list = []

    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self.t = time.monotonic()
        self.fed = bytearray()
        self.returncode = None
        self._closed = kw.get("stdin") is None
        proc = self

        class _Stdin:
            def write(self_, data):
                proc.fed += data

            def close(self_):
                proc._closed = True

        self.stdin = _Stdin()
        FakeProc.spawned.append(self)

    def poll(self):
        if self._closed and self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self._closed = True

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture(autouse=True)
def _fresh_spawns():
    FakeProc.spawned = []
    yield
    FakeProc.spawned = []


@pytest.fixture
def events():
    seen = []
    fn = bus.subscribe(SpeakingState, seen.append)
    try:
        yield seen
    finally:
        bus.unsubscribe(SpeakingState, fn)


# ------------------------------------------------------ the honest mark
def test_speaking_state_rises_when_the_player_starts_not_at_synthesis(
        tmp_path, monkeypatch, events):
    t = TTS(engine="edge", cache=False)
    marks = {}

    def slow_synth(text, out):
        marks["synth"] = time.monotonic()
        time.sleep(0.3)                    # the render the old mark hid
        write_wav(out)

    monkeypatch.setattr(t, "_synth_edge", slow_synth)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    monkeypatch.setattr(tts_mod.subprocess, "Popen", FakeProc)
    t.speak("Was that for me?", block=True)
    assert wait_until(lambda: any(not e.active for e in events))
    rising = next(e for e in events if e.active)
    assert FakeProc.spawned, "no player was spawned"
    popen_t = FakeProc.spawned[0].t
    assert rising.t - marks["synth"] >= 0.25, (
        "the mark landed during synthesis, before any audio")
    assert abs(rising.t - popen_t) < 0.05, "the mark is not at playback start"


def test_a_burst_that_never_plays_still_gets_both_edges(monkeypatch, events):
    """Synthesis failed: no player, but the app pairs a falling edge with a
    rising one (it closes the turn and opens the follow-up mic there)."""
    t = TTS(engine="edge", cache=False)

    def broken(text, out):
        raise RuntimeError("edge is down")

    monkeypatch.setattr(t, "_synth_edge", broken)
    t.speak("hello", block=True)
    assert wait_until(lambda: any(not e.active for e in events))
    actives = [e.active for e in events]
    assert actives.index(True) < actives.index(False)


def test_the_mark_is_once_per_burst_not_per_chunk(tmp_path, monkeypatch, events):
    t = TTS(engine="xtts", cache=False)
    t._xtts = object()
    monkeypatch.setattr(t, "_synth_xtts", lambda text, out: write_wav(out))
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    monkeypatch.setattr(tts_mod.subprocess, "Popen", FakeProc)
    t.speak("The reactor is holding steady this evening. "
            "The workshop is quiet and the coffee is fresh.", block=True)
    assert wait_until(lambda: any(not e.active for e in events))
    assert len(FakeProc.spawned) == 2                 # two chunks played
    assert sum(1 for e in events if e.active) == 1    # one rising edge


# ------------------------------------------------ fish: play while rendering
@pytest.fixture
def fish(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_mod, "_fish_creds", lambda: ("key", "voice"))
    # These tests own the Popen seam for the PLAYER; the F5 warm-up thread
    # (load() on fish) would land its systemctl probe in the same fake and
    # steal spawned[0]. The warm-up has its own tests (test_f5_service.py).
    monkeypatch.setattr(tts_mod.TTS, "warm_f5_fallback", lambda self: None)
    monkeypatch.setattr(tts_mod, "FISH_STREAM_PLAYBACK", True)
    monkeypatch.setattr(tts_mod.subprocess, "Popen", FakeProc)
    t = TTS(engine="fish", cache_dir=tmp_path / "cache")
    return t


def gated_iter(parts, gate: threading.Event, fail_after=None):
    """A fake _fish_iter: yields ``parts`` one at a time, waiting on
    ``gate`` before the LAST one so the test can prove playback began
    before the stream ended; raises after ``fail_after`` parts if set."""
    calls = []

    def _iter(text, timeout):
        calls.append(text)
        for i, part in enumerate(parts):
            if fail_after is not None and i == fail_after:
                raise OSError("connection reset")
            if i == len(parts) - 1:
                gate.wait(3)
            yield part

    _iter.calls = calls
    return _iter


def test_fish_bytes_reach_the_player_before_the_stream_ends(fish, monkeypatch, tmp_path):
    audio = wav_bytes(0.4)
    head, body, tail = audio[:44], audio[44:2000], audio[2000:]
    gate = threading.Event()
    it = gated_iter([head, body, tail], gate)
    monkeypatch.setattr(tts_mod, "_fish_iter", it)
    done = fish.speak("Good evening, sir.")
    # a player exists while the stream is still open (the gate is shut)
    assert wait_until(lambda: bool(FakeProc.spawned))
    proc = FakeProc.spawned[0]
    # reads stdin; the client name keeps the timer chime's saved volume off speech
    assert proc.cmd == ["paplay", f"--client-name={tts_mod.SPEECH_CLIENT_NAME}"]
    assert wait_until(lambda: len(proc.fed) >= len(head) + len(body))
    assert not done.is_set()
    gate.set()
    assert done.wait(3)
    assert bytes(proc.fed) == audio                    # every byte, in order
    assert it.calls == ["Good evening, sir."]
    assert fish.cache.stats()["files"] == 1            # the tee fed the cache
    assert not [p for p in tmp_path.glob("**/*.wav") if "cache" not in str(p)]


def test_fish_repeat_plays_from_cache_without_the_network(fish, monkeypatch):
    gate = threading.Event()
    gate.set()
    it = gated_iter([wav_bytes(0.1)], gate)
    monkeypatch.setattr(tts_mod, "_fish_iter", it)
    played = []
    monkeypatch.setattr(fish, "_play", lambda p: played.append(p))
    monkeypatch.setattr(fish, "_start_amp_feeder", lambda p: None)
    fish.speak("Always, sir.", block=True)
    fish.speak("Always, sir.", block=True)
    assert it.calls == ["Always, sir."]                # one render
    assert len(played) == 1 and "cache" in played[0]  # second from the file


def test_fish_failure_after_audio_is_not_re_spoken_locally(fish, monkeypatch):
    """Half a sentence has already been heard when the network drops: the
    local fallback must NOT render the whole sentence again."""
    audio = wav_bytes(0.4)
    gate = threading.Event()
    gate.set()
    it = gated_iter([audio[:44], audio[44:3000], audio[3000:]], gate,
                    fail_after=2)
    monkeypatch.setattr(tts_mod, "_fish_iter", it)
    f5_calls = []
    monkeypatch.setattr(fish, "load_fallback", lambda: True)
    monkeypatch.setattr(fish, "_synth_f5",
                        lambda text, out: f5_calls.append(text))
    fish.speak("Half of this was heard.", block=True)
    assert FakeProc.spawned and len(FakeProc.spawned[0].fed) == 3000
    assert f5_calls == []
    assert fish.cache.stats()["files"] == 0            # a cut chunk is never cached
    assert fish.engine == "fish"                       # a blip does not retire it


def test_fish_failure_before_any_audio_falls_back_locally(fish, monkeypatch):
    gate = threading.Event()
    gate.set()
    it = gated_iter([b"RIFF"], gate, fail_after=0)     # dies at once
    monkeypatch.setattr(tts_mod, "_fish_iter", it)
    monkeypatch.setattr(fish, "load_fallback", lambda: True)
    f5_calls = []

    def f5(text, out):
        f5_calls.append(text)
        write_wav(out)

    monkeypatch.setattr(fish, "_synth_f5", f5)
    played = []
    monkeypatch.setattr(fish, "_play", lambda p: played.append(os.path.exists(p)))
    monkeypatch.setattr(fish, "_start_amp_feeder", lambda p: None)
    fish.speak("Nothing was heard yet.", block=True)
    assert f5_calls == ["Nothing was heard yet."]
    assert played == [True]
    assert not [p for p in FakeProc.spawned if len(p.cmd) == 1], (
        "a stdin player was spawned for a stream that never produced audio")


def test_stop_mid_stream_cuts_the_player_and_does_not_hang(fish, monkeypatch):
    gate = threading.Event()                           # never opened
    audio = wav_bytes(0.4)
    it = gated_iter([audio[:44], audio[44:2000], audio[2000:]], gate)
    monkeypatch.setattr(tts_mod, "_fish_iter", it)
    done = fish.speak("A line the user talks over.")
    assert wait_until(lambda: FakeProc.spawned and len(FakeProc.spawned[0].fed) >= 2000)
    fish.stop()
    # the consumer drains the producer to its end (so no temp file leaks),
    # and the producer only sees the stop between network chunks: deliver
    # the next one, as a live connection would within milliseconds
    gate.set()
    t0 = time.monotonic()
    assert done.wait(3)
    assert time.monotonic() - t0 < 1.0
    proc = FakeProc.spawned[0]
    assert proc.returncode == -15                      # terminated, not drained
    assert len(proc.fed) < len(audio)                  # the tail never went out
    assert fish.cache.stats()["files"] == 0            # a cut chunk is not cached


def test_the_live_envelope_matches_the_file_envelope():
    """The reactor pulses from bytes as they stream: same 80 ms windows and
    scaling as the soundfile path, whatever the byte boundaries."""
    import numpy as np
    import soundfile as sf
    audio = wav_bytes(0.5)
    data, sr = sf.read(io.BytesIO(audio))
    step = int(sr * 0.08)
    expected = [min(1.0, float(np.sqrt(np.mean(data[i:i + step] ** 2))) * 4)
                for i in range(0, len(data) - step + 1, step)]
    env = tts_mod._LiveEnvelope()
    for i in range(0, len(audio), 777):                # awkward boundaries
        env.push(audio[i:i + 777])
    env.close()
    got = list(env.values())
    assert len(got) == len(expected)
    assert got == pytest.approx(expected, abs=1e-3)


def test_the_file_seam_still_wraps_the_iterator(tmp_path, monkeypatch):
    """tests/test_fish_engine.py patches _fish_stream(text, out, timeout):
    that seam must keep working, and it must be the iterator underneath."""
    monkeypatch.setattr(tts_mod, "_fish_iter",
                        lambda text, timeout: iter([b"RIFF", b"rest"]))
    out = tmp_path / "o.wav"
    res = tts_mod._fish_stream("hi", str(out), 5.0)
    assert res["ok"] and out.read_bytes() == b"RIFFrest"


# ------------------------------------------- skip_current: cancel, keep queue
def test_skip_current_cuts_one_utterance_and_the_next_still_plays(monkeypatch):
    t = TTS(engine="edge", cache=False)
    played, starts = [], []

    def synth(text, out):
        write_wav(out)

    def play(path):
        starts.append(path)
        t0 = time.monotonic()
        while not t._stop_flag and time.monotonic() - t0 < 3:
            time.sleep(0.01)
        played.append(("cut" if t._stop_flag else "full", t.pending))

    monkeypatch.setattr(t, "_synth_edge", synth)
    monkeypatch.setattr(t, "_play", play)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    assert t.skip_current() is False                   # nothing playing
    first = t.speak("first chunk")
    second = t.speak("second chunk")
    assert wait_until(lambda: len(starts) == 1)
    assert t.skip_current() is True
    assert first.wait(2)
    assert wait_until(lambda: len(starts) == 2), "the queued utterance never started"
    assert not second.is_set()                         # still playing, not dropped
    t.stop()
    assert second.wait(2)
    assert played[0] == ("cut", 1)                     # cut with one still queued
    assert len(played) == 2                            # the second one played


def test_a_retired_reply_is_cached_under_the_engine_that_rendered_it(
        tmp_path, monkeypatch, fish):
    """Fish retired mid-reply renders the rest via F5 -- filing that audio
    under the fish cache key replayed the WRONG VOICE from cache once the
    account recovered."""
    t = fish
    monkeypatch.setattr(tts_mod, "FISH_STREAM_PLAYBACK", False)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    monkeypatch.setattr(t, "_synth_f5", lambda text, out: write_wav(out))
    stored = []
    monkeypatch.setattr(t, "_store", lambda engine, text, path: stored.append(engine))
    t._engine = "f5"                     # retire_fish() already flipped it
    t._speak_pipelined("Hello there, sir.", "fish")
    assert stored and set(stored) == {"f5"}, stored
