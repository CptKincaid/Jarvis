"""Transcriber prompt_provider: every initial_prompt — GPU and CPU, full
and partial — goes through _prompt(), and a broken provider degrades to
the legacy vocab file instead of costing the transcription.

The real Transcriber with a fake model object injected (no whisper load);
torch is stubbed in sys.modules so the GPU branch's cuda.synchronize()
never initialises a CUDA context inside the suite.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.config import PATHS
from jarvis.transcriber import DEFAULT_VOCAB, Transcriber


class FakeGpuWhisper:
    """openai-whisper shape: transcribe() returns a dict."""

    def __init__(self):
        self.prompts = []

    def transcribe(self, audio, **kw):
        self.prompts.append(kw.get("initial_prompt"))
        return {"segments": [{"text": "hello", "avg_logprob": -0.3}],
                "language": "en", "text": "hello"}


class FakeFasterWhisper:
    """faster-whisper shape: transcribe() returns (segments, info)."""

    def __init__(self):
        self.prompts = []

    def transcribe(self, audio, **kw):
        self.prompts.append(kw.get("initial_prompt"))
        seg = SimpleNamespace(text="hello", avg_logprob=-0.3)
        return iter([seg]), SimpleNamespace(language="en")


AUDIO = np.zeros(1600, dtype="float32")


@pytest.fixture
def firewall(tmp_path, monkeypatch):
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None)))
    return tmp_path


def _gpu(provider=None):
    tr = Transcriber(prompt_provider=provider)
    tr._model, tr._gpu, tr._backend = FakeGpuWhisper(), True, "GPU fp16"
    return tr


def _cpu(provider=None):
    tr = Transcriber(prompt_provider=provider)
    tr._model, tr._gpu, tr._backend = FakeFasterWhisper(), False, "CPU int8"
    return tr


def test_gpu_transcribe_uses_the_provider(firewall):
    tr = _gpu(lambda: "Peyrovi, BIOSENSORS")
    res = tr.transcribe(AUDIO)
    assert tr._model.prompts == ["Peyrovi, BIOSENSORS"]
    assert res.text == "hello" and res.accepted
    assert res.language == "en"


def test_gpu_partial_uses_the_provider(firewall):
    tr = _gpu(lambda: "Peyrovi, BIOSENSORS")
    assert tr.partial(AUDIO) == "hello"
    assert tr._model.prompts == ["Peyrovi, BIOSENSORS"]


def test_cpu_transcribe_uses_the_provider(firewall):
    tr = _cpu(lambda: "Peyrovi, BIOSENSORS")
    res = tr.transcribe(AUDIO)
    assert tr._model.prompts == ["Peyrovi, BIOSENSORS"]
    assert res.text == "hello" and res.confidence == pytest.approx(-0.3)


def test_cpu_partial_uses_the_provider(firewall):
    tr = _cpu(lambda: "Peyrovi, BIOSENSORS")
    assert tr.partial(AUDIO) == "hello"
    assert tr._model.prompts == ["Peyrovi, BIOSENSORS"]


def test_provider_failure_falls_back_to_the_vocab_file(firewall):
    (firewall / "voice_vocab.txt").write_text("Qwen, Librespot")

    def boom():
        raise RuntimeError("no")

    tr = _gpu(boom)
    tr.transcribe(AUDIO)
    assert tr._model.prompts == ["Qwen, Librespot"]


def test_empty_provider_falls_back_to_default(firewall):
    tr = _cpu(lambda: "   ")
    tr.transcribe(AUDIO)
    assert tr._model.prompts == [DEFAULT_VOCAB]


def test_no_provider_keeps_the_legacy_file_behaviour(firewall):
    (firewall / "voice_vocab.txt").write_text("Qwen")
    tr = _gpu(None)
    tr.transcribe(AUDIO)
    tr.partial(AUDIO)
    assert tr._model.prompts == ["Qwen", "Qwen"]


def test_default_seed_reaches_whisper_when_nothing_configured(firewall):
    tr = _cpu(None)
    tr.transcribe(AUDIO)
    assert tr._model.prompts == [DEFAULT_VOCAB]
    assert "forklift" not in tr._model.prompts[0]
