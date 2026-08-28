"""Voice & I/O upgrades in jarvis.tts: barge-in, speech cache, pronunciation,
say-again, prewarm — and the external speak-queue IPC protocol.

No audio is played: synthesis writes a tiny wav, playback is stubbed.
"""
import os
import subprocess
import threading
import time
import wave
from types import SimpleNamespace

import pytest

import jarvis.pronounce as pronounce
import jarvis.speak_queue as sq
from jarvis.pronounce import Pronunciations
from jarvis.tts import TTS


def write_wav(path, seconds=0.05):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


@pytest.fixture(autouse=True)
def _hermetic_pronunciations(tmp_path, monkeypatch):
    """Never read/write the user's real dictionary from tests."""
    monkeypatch.setattr(pronounce, "_default",
                        Pronunciations(path=tmp_path / "pron.json"))


@pytest.fixture
def tts(tmp_path, monkeypatch):
    """Edge-engine TTS with synthesis and playback stubbed."""
    t = TTS(engine="edge", cache_dir=tmp_path / "cache")
    t.synth_calls = []
    t.played = []

    def fake_synth(text, out_path):
        t.synth_calls.append(text)
        write_wav(out_path)

    def fake_play(path):
        t.played.append((path, os.path.exists(path)))

    monkeypatch.setattr(t, "_synth_edge", fake_synth)
    monkeypatch.setattr(t, "_play", fake_play)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    return t


def wait_until(pred, timeout=3.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.01)
    return pred()


# ------------------------------------------------------------ barge-in
def test_interrupt_cuts_current_and_queued(tmp_path, monkeypatch):
    t = TTS(engine="edge", cache=False)
    started = threading.Event()

    def slow_speak(text):
        started.set()
        t0 = time.monotonic()
        while not t._stop_flag and time.monotonic() - t0 < 3:
            time.sleep(0.01)

    monkeypatch.setattr(t, "_speak_sync", slow_speak)
    t.speak("a long sentence that keeps going")
    t.speak("queued behind it")
    assert started.wait(2)
    assert t.is_speaking and t.pending == 1
    assert t.interrupt() is True
    assert wait_until(lambda: not t.is_speaking)
    assert t.pending == 0
    assert t.interrupts == 1
    assert t.interrupt() is False           # nothing left to cut


def test_interrupt_terminates_playback_process(tts, monkeypatch):
    """The real _play polls the player: stop() must terminate it promptly."""
    monkeypatch.undo()                       # get the real _play back
    tts2 = TTS(engine="edge", cache=False)
    monkeypatch.setattr(tts2, "_synth_edge", lambda text, out: write_wav(out))
    monkeypatch.setattr(tts2, "_start_amp_feeder", lambda p: None)
    real_popen = subprocess.Popen

    def sleepy_popen(cmd, **kw):
        return real_popen(["sleep", "5"], **kw)

    monkeypatch.setattr(subprocess, "Popen", sleepy_popen)
    tts2.speak("hello")
    assert wait_until(lambda: tts2._play_proc is not None)
    t0 = time.monotonic()
    tts2.interrupt()
    assert wait_until(lambda: not tts2.is_speaking)
    assert time.monotonic() - t0 < 2.0


# --------------------------------------------------------- speech cache
def test_cache_hit_skips_synthesis_and_keeps_file(tts, tmp_path):
    tts.speak("Always, sir.", block=True)
    tts.speak("Always, sir.", block=True)
    assert tts.synth_calls == ["Always, sir."]         # synthesized once
    assert len(tts.played) == 2
    first, second = tts.played
    assert first[1] and second[1]                       # both existed at play
    assert str(tmp_path / "cache") in second[0]         # replay from cache
    assert os.path.exists(second[0])                    # cached file kept
    assert not os.path.exists(first[0])                 # temp wav unlinked
    assert tts.cache.stats()["hits"] == 1


def test_cache_key_changes_with_text(tts):
    tts.speak("Yes, sir.", block=True)
    tts.speak("No, sir.", block=True)
    assert tts.synth_calls == ["Yes, sir.", "No, sir."]


def test_cache_disabled(tmp_path, monkeypatch):
    t = TTS(engine="edge", cache=False)
    calls = []
    monkeypatch.setattr(t, "_synth_edge", lambda text, out: (calls.append(text), write_wav(out)))
    monkeypatch.setattr(t, "_play", lambda p: None)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    t.speak("Yes, sir.", block=True)
    t.speak("Yes, sir.", block=True)
    assert calls == ["Yes, sir.", "Yes, sir."]


def test_xtts_pipelined_uses_cache_per_chunk(tmp_path, monkeypatch):
    t = TTS(engine="xtts", cache_dir=tmp_path / "cache")
    t._xtts = object()                                   # pretend loaded
    calls, played = [], []
    monkeypatch.setattr(t, "_synth_xtts", lambda text, out: (calls.append(text), write_wav(out)))
    monkeypatch.setattr(t, "_play", lambda p: played.append((p, os.path.exists(p))))
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    text = ("The reactor is holding steady this evening. "
            "The workshop is quiet and the coffee is fresh.")
    t.speak(text, block=True)
    assert len(calls) == 2                               # two sentence chunks
    t.speak(text, block=True)
    assert len(calls) == 2                               # both from cache
    assert len(played) == 4 and all(existed for _, existed in played)
    assert all(os.path.exists(p) for p, _ in played[2:]) # cached kept
    assert not any(os.path.exists(p) for p, _ in played[:2])  # temps gone


# ------------------------------------------------------- pronunciation
def test_pronunciation_applied_at_synthesis_only(tts):
    tts.speak("VSS is up on the GB10.", block=True)
    assert tts.synth_calls == ["V S S is up on the G B ten."]
    assert tts.last_text == "VSS is up on the GB10."      # transcript keeps it


def test_pronunciation_can_be_disabled(tmp_path, monkeypatch):
    t = TTS(engine="edge", cache=False, pronunciation=False)
    calls = []
    monkeypatch.setattr(t, "_synth_edge", lambda text, out: (calls.append(text), write_wav(out)))
    monkeypatch.setattr(t, "_play", lambda p: None)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    t.speak("VSS", block=True)
    assert calls == ["VSS"]


def test_user_pronunciation_reaches_the_voice(tts):
    pronounce.get().add("Peyrovi", "pay-ROH-vee")
    tts.speak("Good evening, Mr Peyrovi.", block=True)
    assert tts.synth_calls == ["Good evening, Mr pay-ROH-vee."]


# ------------------------------------------------------------ say again
def test_repeat_last(tts):
    assert tts.repeat_last() is False
    tts.speak("Right here, sir.", block=True)
    assert tts.repeat_last() is True
    assert wait_until(lambda: len(tts.played) == 2)
    assert tts.synth_calls == ["Right here, sir."]       # second from cache


# -------------------------------------------------------------- prewarm
def test_prewarm_renders_into_cache_then_speech_is_instant(tts):
    th = tts.prewarm(["Always, sir.", "Yes, sir.", "  "], block=True)
    assert th is not None
    assert sorted(tts.synth_calls) == ["Always, sir.", "Yes, sir."]
    assert tts.cache.stats()["files"] == 2
    tts.speak("Yes, sir.", block=True)
    assert len(tts.synth_calls) == 2                     # no new synthesis


def test_prewarm_skips_cached_and_without_cache(tts, tmp_path, monkeypatch):
    tts.speak("Always, sir.", block=True)
    tts.prewarm(["Always, sir."], block=True)
    assert tts.synth_calls == ["Always, sir."]
    t2 = TTS(engine="edge", cache=False)
    assert t2.prewarm(["x"]) is None


# ---------------------------------------- external speak-queue protocol
def test_external_process_protocol_append_line(tmp_path):
    """How ANY other process makes Jarvis speak: append a line to the
    queue file (PATHS.SPEAK_QUEUE, /tmp/vss_voice/speak_queue.txt in the
    live app). The watcher tails it once a second, joins all new lines
    into one utterance, hands it to the TTS sink, and truncates the file.
    Tested against a temp file only."""
    path = tmp_path / "speak_queue.txt"
    got = []
    sq.set_sink(got.append)
    w = sq.Watcher(path)
    w.POLL_S = 0.05
    w.start()
    try:
        subprocess.run(["bash", "-c", f"echo 'Jarvis: build finished' >> '{path}'"],
                       check=True)
        assert wait_until(lambda: got == ["Jarvis: build finished"])
        assert path.stat().st_size == 0                  # truncated after read
        subprocess.run(["bash", "-c", f"printf 'line a\\nline b\\n' >> '{path}'"],
                       check=True)
        assert wait_until(lambda: len(got) == 2)
        assert got[1] == "line a line b"                 # joined utterance
    finally:
        w.stop()
        sq.set_sink(None)


def test_in_process_say_goes_straight_to_sink(tmp_path, monkeypatch):
    monkeypatch.setattr(sq, "SPEAK_QUEUE", tmp_path / "q.txt")
    got = []
    sq.set_sink(got.append)
    try:
        sq.say("Sir, the build is green.")
        assert got == ["Sir, the build is green."]
        assert not (tmp_path / "q.txt").exists()
    finally:
        sq.set_sink(None)


# ------------------------------------------- chunking long sentences
#
# 2026-08-28 13:02: an audible gap between the first and second spoken
# sentence. The pipelined path synthesises chunk N+1 while chunk N plays, but
# it split only on [.!?;] -- so "It is overcast today with a high of 98." (a
# few seconds of audio) was followed by one enormous comma-separated sentence
# listing four classes with building names. Synthesis of chunk 2 took far
# longer than chunk 1 took to play, and playback ran dry.
LONG_SENTENCE = (
    "On Monday you have Biosensors at 9:10 am in Wisenbaker 049, then "
    "Magnetic Resonance Engineering at 12:40 pm in the Emerging Technologies "
    "Building 1003, then Electrical Design Lab Two at 4:10 pm in ETB 1020, "
    "and finally Magnetic Resonance Engineering again at 6:00 pm in Zachry 330.")


def test_a_long_sentence_is_broken_up_for_synthesis(tts):
    chunks = tts._split_sentences(LONG_SENTENCE)
    assert len(chunks) > 1, "one huge chunk starves playback and you hear a gap"
    assert all(len(c) <= 200 for c in chunks), [len(c) for c in chunks]


def test_chunking_preserves_every_word_in_order(tts):
    chunks = tts._split_sentences(LONG_SENTENCE)
    assert " ".join(chunks).split() == LONG_SENTENCE.split()


def test_it_breaks_at_commas_not_mid_word(tts):
    for chunk in tts._split_sentences(LONG_SENTENCE):
        assert chunk == chunk.strip()
        assert not chunk.startswith(","), chunk


def test_ordinary_sentences_are_left_alone(tts):
    text = "You have nothing on today, sir."
    assert tts._split_sentences(text) == [text]


def test_two_normal_sentences_still_split_on_the_full_stop(tts):
    # both halves over min_chars; a shorter tail merges backward by design
    text = ("It is overcast today with a high of 98. "
            "You have nothing else on the calendar today, sir.")
    assert len(tts._split_sentences(text)) == 2


# ------------------------------------------------ priming the voice latents
#
# XTTS caches the speaker conditioning latents internally, so only the FIRST
# synthesis pays for them (measured on the GB10: first call 2.21 s, warm 0.97 s;
# get_conditioning_latents alone is 1.37 s). prewarm() would have primed them as
# a side effect, but it skips phrases already in the speech cache -- and on a
# warm cache it renders nothing at all ("prewarm: 0 phrase chunk(s) rendered"),
# so the first real utterance of every session paid the cold path.
def test_loading_primes_the_speaker_latents(monkeypatch, tmp_path):
    from jarvis.tts import TTS

    calls = []

    class FakeModel:
        def get_conditioning_latents(self, audio_path=None, **kw):
            calls.append(audio_path)
            return ("latent", "embedding")

    t = TTS(engine="xtts", cache=False)
    t._xtts = SimpleNamespace(synthesizer=SimpleNamespace(tts_model=FakeModel()))

    t._prime_voice()

    assert calls, "the first utterance will pay the cold latent cost"


def test_priming_failure_never_breaks_startup(monkeypatch):
    """A prime is an optimisation; it must not stop Jarvis from speaking."""
    from jarvis.tts import TTS

    class Broken:
        def get_conditioning_latents(self, *a, **kw):
            raise RuntimeError("cuda hiccup")

    t = TTS(engine="xtts", cache=False)
    t._xtts = SimpleNamespace(synthesizer=SimpleNamespace(tts_model=Broken()))
    t._prime_voice()          # must not raise


def test_priming_is_a_noop_without_xtts(monkeypatch):
    from jarvis.tts import TTS
    t = TTS(engine="edge", cache=False)
    t._prime_voice()          # must not raise
