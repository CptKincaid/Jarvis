"""Tests for jarvis.tts (queue mechanics, no audio) and jarvis.speak_queue."""
import threading
import time

import pytest

import jarvis.speak_queue as sq
from jarvis.events import SpeakingState, bus
from jarvis.tts import TTS


# ------------------------------------------------------------------ tts
def make_tts(monkeypatch, spoken, synth_delay=0.0):
    """TTS with synthesis/playback stubbed out (no mic/speaker on this box)."""
    tts = TTS(engine="edge")

    def fake_speak_sync(text):
        if synth_delay:
            time.sleep(synth_delay)
        if tts._stop_flag:
            return
        spoken.append(text)

    monkeypatch.setattr(tts, "_speak_sync", fake_speak_sync)
    return tts


def test_speak_enqueues_no_drops(monkeypatch):
    spoken = []
    tts = make_tts(monkeypatch, spoken)
    for i in range(5):
        tts.speak(f"line {i}")
    deadline = time.time() + 3
    while len(spoken) < 5 and time.time() < deadline:
        time.sleep(0.02)
    assert spoken == [f"line {i}" for i in range(5)]  # FIFO, none dropped


def test_speak_block_waits(monkeypatch):
    spoken = []
    tts = make_tts(monkeypatch, spoken, synth_delay=0.1)
    tts.speak("blocking line", block=True)
    assert spoken == ["blocking line"]


def test_speaking_state_events(monkeypatch):
    events = []
    fn = bus.subscribe(SpeakingState, events.append)
    try:
        spoken = []
        tts = make_tts(monkeypatch, spoken)
        tts.speak("hello", block=True)
        deadline = time.time() + 2
        while not any(not e.active for e in events) and time.time() < deadline:
            time.sleep(0.02)
        assert any(e.active for e in events)       # start published
        assert any(not e.active for e in events)   # end published
    finally:
        bus.unsubscribe(SpeakingState, fn)


def test_stop_clears_queue(monkeypatch):
    spoken = []
    tts = make_tts(monkeypatch, spoken, synth_delay=0.2)
    tts.speak("first")
    time.sleep(0.05)          # worker picks up "first"
    for i in range(4):
        tts.speak(f"queued {i}")
    tts.stop()
    time.sleep(0.5)
    assert all(not t.startswith("queued") for t in spoken)


def test_clean_for_speech():
    tts = TTS.__new__(TTS)    # no worker thread needed
    out = tts._clean_for_speech("**bold** and `code` and https://x.com/y ok")
    assert out == "bold and and ok"
    long = "word " * 300
    assert len(tts._clean_for_speech(long)) <= TTS.MAX_SPEAK_LENGTH + 3


def test_engine_property():
    tts = TTS.__new__(TTS)
    tts._engine = "edge"
    tts.engine = "xtts"
    assert tts.engine == "xtts"
    with pytest.raises(ValueError):
        tts.engine = "bogus"


# ---------------------------------------------------------- speak_queue
@pytest.fixture
def qfile(tmp_path, monkeypatch):
    path = tmp_path / "speak_queue.txt"
    monkeypatch.setattr(sq, "SPEAK_QUEUE", path)
    sq.set_sink(None)
    yield path
    sq.set_sink(None)


def test_say_uses_sink_directly(qfile):
    got = []
    sq.set_sink(got.append)
    sq.say("  hello sir  ")
    assert got == ["hello sir"]
    assert not qfile.exists()          # no /tmp detour


def test_say_falls_back_to_file(qfile):
    sq.say("offline line")             # no sink wired
    assert qfile.read_text() == "offline line\n"

    def bad_sink(text):
        raise RuntimeError("boom")

    sq.set_sink(bad_sink)
    sq.say("second line")              # sink raises → file fallback
    assert qfile.read_text().splitlines() == ["offline line", "second line"]


def test_watcher_reads_combines_and_truncates(qfile):
    got = []
    sq.set_sink(got.append)
    w = sq.Watcher(qfile)              # not started: poll manually
    qfile.write_text("line one\nline two\n")
    w.poll_once()
    assert got == ["line one line two"]
    assert qfile.stat().st_size == 0   # truncated after read
    w.poll_once()                      # no re-read after truncate
    assert got == ["line one line two"]


def test_watcher_handles_external_truncation(qfile):
    got = []
    sq.set_sink(got.append)
    w = sq.Watcher(qfile)
    qfile.write_text("a much longer earlier message here\n")
    w.poll_once()
    assert got == ["a much longer earlier message here"]
    # External writer truncates and writes something shorter.
    w._pos = 100                       # simulate stale pos beyond new size
    qfile.write_text("short\n")
    w.poll_once()
    assert got[-1] == "short"


def test_watcher_skips_stale_backlog_on_start(qfile):
    got = []
    sq.set_sink(got.append)
    qfile.write_text("stale old content\n")
    w = sq.Watcher(qfile)
    w.start()
    try:
        time.sleep(0.1)
        qfile.write_text("stale old content\nfresh line\n", )
        deadline = time.time() + 3
        while not got and time.time() < deadline:
            time.sleep(0.05)
    finally:
        w.stop()
    assert got == ["fresh line"]


def test_watcher_waits_for_sink(qfile):
    w = sq.Watcher(qfile)
    qfile.write_text("held message\n")
    w.poll_once()                      # no sink → leave content in place
    assert qfile.read_text() == "held message\n"
    got = []
    sq.set_sink(got.append)
    w.poll_once()
    assert got == ["held message"]
