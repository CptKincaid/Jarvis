"""Regression test: XTTS was loaded twice, and the user waited for both.

DEFECT (jarvis/tts.py TTS.load):

load() guards with ``if self._xtts is not None: return True`` and nothing
else. Two threads reach it in a normal turn:

  * the startup warmer, jarvis/app.py _load_models() -> self.tts.load()
  * the TTS worker thread, _speak_sync() -> self.load()

Both can pass the None check before either assigns self._xtts, so both run a
full 1.8 GB XTTS load, concurrently, contending for the same GPU. Observed in
/tmp/vss_voice/jarvis.log on 2026-08-28:

    16:41:35.254  chat reply ready (the LLM took 1.45 s)
    16:41:42.005  XTTS v2 loaded on CUDA:0      <- startup warmer
    16:41:45.925  XTTS v2 loaded on CUDA:0      <- speak path, AGAIN
    16:41:46.320  speaking (xtts)

11.07 s of silence after the reply was ready, on a GPU that synthesises this
sentence in well under a second. The duplicate load is the second half of
that; the first half is _load_models() reaching TTS only fourth, behind a
5.5 s ollama warmup (covered by test_found_tts_loads_off_the_critical_path).
"""
import sys
import threading
import time
import types

import pytest

from jarvis.tts import TTS


@pytest.fixture
def fake_coqui(monkeypatch, tmp_path):
    """Stand in for TTS.api, counting how many models get built."""
    built = []

    class FakeCoqui:
        def __init__(self, model_name):
            built.append(model_name)
            time.sleep(0.25)          # a real load is seconds; this is enough
                                      # for a second thread to slip past the check

        def to(self, device):
            return self

    mod = types.ModuleType("TTS.api")
    mod.TTS = FakeCoqui
    pkg = types.ModuleType("TTS")
    pkg.api = mod
    monkeypatch.setitem(sys.modules, "TTS", pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", mod)

    ref = tmp_path / "voice_ref.wav"
    ref.write_bytes(b"RIFF")
    monkeypatch.setattr("jarvis.tts.VOICE_REF", ref)
    return built


def test_concurrent_loads_build_the_model_once(fake_coqui, tmp_path):
    t = TTS(engine="xtts", cache_dir=tmp_path / "cache")
    t._prime_voice = lambda: None          # priming is covered elsewhere

    results, errors = [], []

    def worker():
        try:
            results.append(t.load())
        except Exception as exc:            # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)

    assert not errors, errors
    assert results == [True, True], results
    assert len(fake_coqui) == 1, (
        f"XTTS was built {len(fake_coqui)} times; the second load is pure "
        f"latency the user waits through: {fake_coqui}")


def test_the_second_caller_waits_for_the_model_rather_than_racing_it(
        fake_coqui, tmp_path):
    """A caller that returns True must find a model actually there."""
    t = TTS(engine="xtts", cache_dir=tmp_path / "cache")
    t._prime_voice = lambda: None
    seen = []

    def worker():
        ok = t.load()
        seen.append((ok, t._xtts is not None))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)

    assert seen == [(True, True)] * 3, seen
    assert len(fake_coqui) == 1, fake_coqui
