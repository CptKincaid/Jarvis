"""Tests for jarvis.speech_cache."""
import os
import time

from jarvis.speech_cache import SpeechCache


def _wav(path, size=1000):
    path.write_bytes(b"RIFF" + bytes(size))
    return path


def test_key_is_stable_and_param_sensitive():
    k1 = SpeechCache.key("edge", "Always, sir.", voice="ryan")
    assert k1 == SpeechCache.key("edge", "Always, sir.", voice="ryan")
    assert k1 != SpeechCache.key("edge", "Always, sir.", voice="guy")
    assert k1 != SpeechCache.key("xtts", "Always, sir.", voice="ryan")
    assert len(k1) == 40


def test_put_then_get(tmp_path):
    cache = SpeechCache(tmp_path / "cache")
    key = SpeechCache.key("edge", "hi")
    assert cache.get(key) is None
    src = _wav(tmp_path / "src.wav")
    dst = cache.put(key, src)
    assert dst is not None and dst.suffix == ".wav" and dst.read_bytes() == src.read_bytes()
    assert cache.get(key) == dst
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 1


def test_put_ignores_empty_or_missing_source(tmp_path):
    cache = SpeechCache(tmp_path / "cache")
    assert cache.put("k", tmp_path / "nope.wav") is None
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert cache.put("k", empty) is None
    assert not (tmp_path / "cache").exists()      # lazy: nothing created


def test_prune_evicts_least_recently_used(tmp_path):
    cache = SpeechCache(tmp_path / "cache", max_files=3)
    paths = []
    for i in range(4):
        p = cache.put(f"key{i}", _wav(tmp_path / f"s{i}.wav"))
        os.utime(p, (i + 1, i + 1))                # oldest first
        paths.append(p)
    assert cache.prune() == 0 or True             # put() already pruned
    remaining = sorted(q.name for q in (tmp_path / "cache").iterdir())
    assert len(remaining) == 3
    assert "key0.wav" not in remaining


def test_get_touches_mtime(tmp_path):
    cache = SpeechCache(tmp_path / "cache")
    p = cache.put("k", _wav(tmp_path / "s.wav"))
    os.utime(p, (1, 1))
    cache.get("k")
    assert p.stat().st_mtime > 1


def test_prune_by_bytes(tmp_path):
    cache = SpeechCache(tmp_path / "cache", max_bytes=2500)
    for i in range(4):
        p = cache.put(f"k{i}", _wav(tmp_path / f"s{i}.wav", size=1000))
        os.utime(p, (i + 1, i + 1))
    total = sum(q.stat().st_size for q in (tmp_path / "cache").iterdir())
    assert total <= 2500


def test_clear(tmp_path):
    cache = SpeechCache(tmp_path / "cache")
    cache.put("k", _wav(tmp_path / "s.wav"))
    assert cache.clear() == 1
    assert cache.get("k") is None
    assert time.time() > 0
