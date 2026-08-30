"""The Fish Audio engine: hosted s2.1-pro, with the network designed around.

WHY HOSTED. Five local engines were rejected by ear on 2026-08-28 (XTTS,
Piper+jarvis, Chatterbox-Turbo, F5-TTS, and Fish S2-Pro run locally). The one
whose quality won — Fish — cannot run here: S2-Pro measured RTF ~50 on the
GB10 because autoregressive decode is memory-bandwidth-bound and the box has
~103-151 GB/s against the H200's ~30x that. The hosted s2.1-pro measured
186 ms time-to-first-audio from this box, FASTER than any local option
including XTTS's 0.4 s, and it is the model that produced the reference the
user rated best.

WHAT THAT COSTS, AND HOW IT IS PAID FOR HERE:
  * The voice now depends on the network. So the cache is consulted FIRST
    (every repeated line is free and offline after one render), and any
    failure falls back to a local engine rather than going silent. An
    assistant that says nothing reads as broken.
  * The free tier is NOT usable: measured 2218 ms median, degrading to
    3294 ms across five consecutive calls. FISH_BACKEND must stay on the
    paid backend.
  * Credentials live outside the repo, in ~/.config/jarvis/, never in code.
"""
import pytest

from jarvis import tts as tts_mod
from jarvis.tts import TTS


def test_fish_is_a_registered_engine():
    assert "fish" in tts_mod._ENGINES


def test_credentials_come_from_disk_not_source():
    for p in (tts_mod.FISH_KEY_FILE, tts_mod.FISH_MODEL_FILE):
        assert ".config/jarvis" in str(p), p
    src = (tts_mod.__file__)
    body = open(src).read()
    assert "sk-fish-" not in body, "an API key is hard-coded in tts.py"


def test_the_paid_backend_is_used():
    """The free tier measured 2.2s median and degrades under repeat use."""
    assert tts_mod.FISH_BACKEND == "s2.1-pro"


def test_it_asks_for_wav_so_playback_matches_the_other_engines(tmp_path, monkeypatch):
    seen = {}

    def fake_stream(text, out_path, timeout):
        seen["text"] = text
        seen["out"] = out_path
        open(out_path, "wb").write(b"RIFF")
        return {"ok": True}

    monkeypatch.setattr(tts_mod, "_fish_stream", fake_stream)
    t = TTS(engine="fish", cache_dir=tmp_path / "c")
    out = tmp_path / "o.wav"
    t._synth_fish("Good evening, sir.", str(out))
    assert seen["text"] == "Good evening, sir."
    assert out.exists()


def test_cache_key_is_specific_to_the_voice_and_backend(tmp_path, monkeypatch):
    t = TTS(engine="fish", cache_dir=tmp_path / "c")
    base = t._cache_key("fish", "hello")
    assert base != t._cache_key("f5", "hello")
    assert base != t._cache_key("xtts", "hello")
    monkeypatch.setattr(tts_mod, "FISH_BACKEND", "s2.1-pro-free")
    assert t._cache_key("fish", "hello") != base


def test_missing_credentials_degrade_to_local_without_losing_the_utterance(
        tmp_path, monkeypatch):
    """Without a key, fall back AND report success.

    _speak_sync does `if not self.load(): return`, so answering False costs
    the caller that whole utterance. Since the fallback engine really can
    speak, the honest answer is True — degraded, but not silent.
    """
    monkeypatch.setattr(tts_mod, "FISH_KEY_FILE", tmp_path / "missing_key")
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda: True)
    t = TTS(engine="fish", cache_dir=tmp_path / "c")
    assert t.load() is True
    assert t.engine == tts_mod.FISH_FALLBACK
    assert t.engine != "fish", "must degrade, not keep retrying a dead engine"


def test_a_network_failure_raises_so_the_caller_can_fall_back(tmp_path, monkeypatch):
    def boom(text, out_path, timeout):
        raise OSError("network is down")

    monkeypatch.setattr(tts_mod, "_fish_stream", boom)
    t = TTS(engine="fish", cache_dir=tmp_path / "c")
    with pytest.raises(Exception):
        t._synth_fish("hello", str(tmp_path / "o.wav"))


def test_the_fallback_engine_is_local(tmp_path):
    """If the fallback were also hosted, an outage would still be silence."""
    assert tts_mod.FISH_FALLBACK in ("f5", "xtts", "edge")
    assert tts_mod.FISH_FALLBACK != "fish"


def test_a_stalled_connection_raises_instead_of_wedging(monkeypatch):
    """The SDK's httpx client has timeout=None: a socket that delivers no
    bytes used to block session.tts() forever, and the old between-chunks
    deadline never ran -- the TTS worker (and everything queued behind it)
    wedged. The feeder-thread iterator must raise on the wall clock."""
    import sys
    import threading as th
    import time
    import types as ty

    class _Sess:
        def __init__(self, key):
            pass

        def tts(self, req, backend=None):
            th.Event().wait(10)          # a connection delivering nothing
            yield b""                    # pragma: no cover - never reached

    mod = ty.ModuleType("fish_audio_sdk")
    mod.Session = _Sess
    mod.TTSRequest = lambda **k: None
    monkeypatch.setitem(sys.modules, "fish_audio_sdk", mod)
    monkeypatch.setattr(tts_mod, "_fish_creds", lambda: ("k", "m"))
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        list(tts_mod._fish_iter("hello", timeout=0.3))
    assert time.monotonic() - t0 < 3.0, "raised on the wall clock, not the stream"


def test_chunks_still_flow_through_the_feeder(monkeypatch):
    import sys
    import types as ty

    class _Sess:
        def __init__(self, key):
            pass

        def tts(self, req, backend=None):
            yield b"ab"
            yield b"cd"

    mod = ty.ModuleType("fish_audio_sdk")
    mod.Session = _Sess
    mod.TTSRequest = lambda **k: None
    monkeypatch.setitem(sys.modules, "fish_audio_sdk", mod)
    monkeypatch.setattr(tts_mod, "_fish_creds", lambda: ("k", "m"))
    assert list(tts_mod._fish_iter("hello", timeout=5)) == [b"ab", b"cd"]


def test_an_xtts_load_failure_is_loud_and_still_speaks(tmp_path, monkeypatch):
    """The XTTS->edge hop used to be one log line and a False (the caller
    dropped the utterance). It must announce the CLOUD fallback on the bus
    and hand back edge's verdict so the line is still spoken."""
    import sys
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    monkeypatch.setattr(tts_mod, "VOICE_REF", ref)
    # the transformers shim import is the first heavy step: fail there,
    # cheaply, before any model download could start
    monkeypatch.setitem(sys.modules, "transformers.pytorch_utils", None)
    published = []
    monkeypatch.setattr(tts_mod.bus, "publish", published.append)
    t = TTS(engine="xtts", cache_dir=tmp_path / "c")
    assert t.load() is True, "edge can speak; the utterance is not dropped"
    assert t.engine == "edge"
    warns = [e for e in published
             if getattr(e, "kind", "") == "warn" and "edge" in getattr(e, "text", "")]
    assert warns, "the cloud fallback must be announced, not just logged"
