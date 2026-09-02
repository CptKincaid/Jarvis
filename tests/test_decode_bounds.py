"""Whisper's decode was unbounded, and both halves of it cost Hunter a
nine-second turn.

The ledger line he complained about (/tmp/vss_voice/jarvis.log, 2026-09-02):

    14:44:00.099 turn: wake->mic 58ms . speech 3.65s . dead-air 800ms
                 . stt 5.23s . route 3.19s . wait 9.22s (stop=vad decode=speculative)

Against the good turns from the night before -- stt 63-132 ms, wait
1.30-1.43 s -- the whole difference is stt. Two causes, both measured on
this box through the real Transcriber (whisper "turbo", cuda, fp16, GB10).

1. THE FIRST INFERENCE OF EVERY PROCESS. The weights are preloaded
   ("preloaded torch/whisper + CUDA in 1.2s") but nothing runs a kernel
   until the user speaks, so the first decode pays autotune and the
   caching allocator's first fill. A/B on one real 3.3 s mic capture,
   fresh process each way:

       no warm-up   first user turn  0.951s   (0.944 / 0.870 / 0.927 across runs)
       warm-up      first user turn  0.252s   (0.285 / 0.258 / 0.247)

   ~0.70 s off the first turn after every launch, for a 0.85 s throwaway
   decode paid on the model-loader thread where nobody is waiting.

2. THE UNBOUNDED DECODE. openai-whisper re-decodes a window at
   (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) whenever the greedy pass trips
   logprob_threshold=-1.0 or compression_ratio_threshold=2.4 -- six full
   decodes -- and each pass may emit n_text_ctx//2 = 224 tokens, so a
   repetition loop runs to the cap. Over 20 non-speech clips:

       whisper's defaults   total 49.5s   worst single decode 9.70s
       the bound            total  9.1s   worst single decode 1.00s

   That is the shape of every slow decode in his log -- 9.55s, 5.52s,
   5.18s, 3.48s, four of the six returning character salad -- against
   0.02-0.45 s for every clean one.

NEITHER BOUND IS A TRADE, which is the part that had to be proved: a wrong
transcript that arrives fast is no better than a slow one.

  * 113 real clips (the voice corpus, his own mic captures, and a 20 s
    dictation run) through the shipped Transcriber, unbounded vs bounded:
    IDENTICAL transcript on 113/113, identical median/p95/max decode time,
    113/113 still accepted.
  * On his own failure shape -- 180 clips cut to 0.55/0.85 s and mixed to
    0 dB SNR, where avg_logprob collapses from length bias -- 34 tripped
    the ladder, and one greedy rung was the most ACCURATE of the three
    settings as well as the fastest (WER 0.3882 vs 0.4609 at six rungs and
    0.6257 at two). Whisper throws away a correct short transcript because
    a length-biased mean says so: the same mistake the -0.85 confidence
    gate used to make.
  * the fastest real clip emitted 8.89 tokens/second; the budget allows
    12/s plus 32, and hands Whisper's own 224 back from ~16 s up, so
    dictation of a full window is untouched.

What removing the ladder DOES take away is an accident the confidence gate
was living off: an exhausted ladder returns the temperature-1.0 sample,
whose avg_logprob is deeply negative, and that is how some repetition loops
used to land below MIN_AVG_LOGPROB. On one greedy pass the same clip comes
back at avg_logprob -0.31 and would be dispatched, so the loop is rejected
on Whisper's own designed signal instead -- compression_ratio, 1.64 at worst
over those 113 real clips against 7.55-13.14 on the loops. That gate closes
a hole rather than opening one: whisper's stock decode ACCEPTED five of the
same 20 non-speech clips as "Thank you." / "has already been done."
(avg_logprob -0.69 to -0.98, comfortably above -2.90) and handed them to
the commander.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import jarvis.app as app_mod
from jarvis.config import CONFIG, PATHS
from jarvis.transcriber import (
    MAX_COMPRESSION_RATIO,
    DECODE_TEMPERATURES,
    SAMPLE_RATE,
    WHISPER_SAMPLE_LEN,
    TranscribeResult,
    Transcriber,
    token_budget,
)
from tests.test_app_wiring import build, paths, seams  # noqa: F401  (fixtures)


# --------------------------------------------------------------- fakes
class FakeGpuWhisper:
    """openai-whisper shape: transcribe() returns a dict."""

    def __init__(self, segments=None):
        self.calls: list[dict] = []
        self.audio: list[np.ndarray] = []
        self._segments = segments if segments is not None else [
            {"text": "hello", "avg_logprob": -0.3, "compression_ratio": 1.2}]

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        self.audio.append(audio)
        return {"segments": list(self._segments), "language": "en",
                "text": " ".join(s["text"] for s in self._segments)}


class FakeFasterWhisper:
    """faster-whisper shape: transcribe() returns (segments, info)."""

    def __init__(self, segments=None):
        self.calls: list[dict] = []
        self.audio: list[np.ndarray] = []
        self._segments = segments if segments is not None else [
            SimpleNamespace(text="hello", avg_logprob=-0.3,
                            compression_ratio=1.2)]

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        self.audio.append(audio)
        return iter(list(self._segments)), SimpleNamespace(language="en")


@pytest.fixture
def firewall(tmp_path, monkeypatch):
    """No vocab file, no CUDA context: torch is a stub so the GPU branch's
    cuda.synchronize() cannot initialise a device inside the suite."""
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None)))
    return tmp_path


def _gpu(model=None):
    tr = Transcriber()
    tr._model = model or FakeGpuWhisper()
    tr._gpu, tr._backend = True, "GPU fp16"
    return tr


def _cpu(model=None):
    tr = Transcriber()
    tr._model = model or FakeFasterWhisper()
    tr._gpu, tr._backend = False, "CPU int8"
    return tr


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# ===================================================== the token budget
# The fastest of 113 measured real clips emitted 8.89 tokens/second (a
# 0.9 s cut of "That didn't work, sir"); the slowest full clip needing the
# most tokens was a 20 s run at 69 tokens.
@pytest.mark.parametrize("seconds,tokens_needed", [
    (0.9, 8),        # 0109 cut short: "That didn't work, sir"
    (1.35, 11),      # 0013 "It is five o'clock, sir."
    (3.3, 11),       # his own mic capture, capture_last_vad.wav
    (10.0, 89),      # 8.89 tok/s sustained for ten seconds: nobody does
    (20.0, 69),      # six corpus clips end to end
])
def test_the_token_budget_never_bites_into_real_speech(seconds, tokens_needed):
    assert token_budget(seconds) > tokens_needed * 1.5, (
        "the cap must sit well clear of what speech actually emits, or it "
        "trades accuracy for latency")


def test_a_long_clip_gets_whispers_own_limit_back():
    """Dictation is a 30 s window of continuous speech; the cap must be a
    no-op there, not a truncation."""
    assert token_budget(16.0) == WHISPER_SAMPLE_LEN
    assert token_budget(30.0) == WHISPER_SAMPLE_LEN
    assert token_budget(120.0) == WHISPER_SAMPLE_LEN


def test_the_budget_grows_with_the_clip_and_is_never_zero():
    assert token_budget(0.0) >= 32
    assert token_budget(-5.0) >= 32          # a bad length must not floor it
    values = [token_budget(s) for s in (0.5, 1, 2, 4, 8, 16)]
    assert values == sorted(values)


# ============================================== the bound on the decode
def test_the_gpu_decode_never_climbs_the_temperature_ladder(firewall):
    tr = _gpu()
    tr.transcribe(_audio(3.0))
    assert tr._model.calls[0]["temperature"] == DECODE_TEMPERATURES
    assert len(DECODE_TEMPERATURES) == 1


def test_the_gpu_decode_caps_the_token_run_by_clip_length(firewall):
    tr = _gpu()
    tr.transcribe(_audio(3.0))
    assert tr._model.calls[0]["sample_len"] == token_budget(3.0)
    tr.transcribe(_audio(30.0))
    assert tr._model.calls[1]["sample_len"] == WHISPER_SAMPLE_LEN


def test_the_cpu_decode_is_bounded_the_same_way(firewall):
    """faster-whisper spells the cap max_new_tokens, but a runaway there is
    the same runaway."""
    tr = _cpu()
    tr.transcribe(_audio(3.0))
    kw = tr._model.calls[0]
    assert kw["temperature"] == DECODE_TEMPERATURES
    assert kw["max_new_tokens"] == token_budget(3.0)


def test_the_live_preview_is_bounded_too(firewall):
    """The preview runs several times a second while he is still talking;
    an unbounded pass there holds the model lock the final decode needs."""
    tr = _gpu()
    tr.partial(_audio(2.0))
    assert tr._model.calls[0]["sample_len"] == token_budget(2.0)
    tr = _cpu()
    tr.partial(_audio(2.0))
    assert tr._model.calls[0]["max_new_tokens"] == token_budget(2.0)


# ================================ the loop the shortened ladder lets by
def test_a_repetition_loop_is_rejected_even_when_its_logprob_looks_healthy():
    """Measured bounded output on non-speech: 'and the other day, the other
    day, ...' at avg_logprob -0.42, compression_ratio 8.81. The -2.90
    confidence gate waves that straight through."""
    loop = TranscribeResult(
        text="and the other day, the other day, the other day",
        confidence=-0.42, compression_ratio=8.81,
        segments=[("and the other day, the other day, the other day", -0.42)])
    assert loop.looping is True
    assert loop.accepted is False


@pytest.mark.parametrize("ratio", [0.53, 0.72, 1.20, 1.57, 1.64])
def test_real_speech_compression_ratios_are_nowhere_near_the_gate(ratio):
    """1.64 is the worst of 113 real clips across five decode settings;
    Whisper's own threshold is 2.4."""
    assert ratio < MAX_COMPRESSION_RATIO
    res = TranscribeResult(text="set a timer for eight minutes", confidence=-0.5,
                           compression_ratio=ratio,
                           segments=[("set a timer for eight minutes", -0.5)])
    assert res.looping is False and res.accepted is True


def test_the_gpu_path_carries_the_compression_ratio_out(firewall):
    tr = _gpu(FakeGpuWhisper(segments=[
        {"text": "la la la la", "avg_logprob": -0.2, "compression_ratio": 9.4}]))
    res = tr.transcribe(_audio(3.0))
    assert res.compression_ratio == pytest.approx(9.4)
    assert res.accepted is False


def test_the_cpu_path_carries_the_compression_ratio_out(firewall):
    tr = _cpu(FakeFasterWhisper(segments=[
        SimpleNamespace(text="la la la la", avg_logprob=-0.2,
                        compression_ratio=9.4)]))
    res = tr.transcribe(_audio(3.0))
    assert res.compression_ratio == pytest.approx(9.4)
    assert res.accepted is False


def test_the_worst_segment_is_the_one_that_counts(firewall):
    """One looping segment poisons the turn even when the others are clean;
    a mean would hide it."""
    tr = _gpu(FakeGpuWhisper(segments=[
        {"text": "set a timer", "avg_logprob": -0.3, "compression_ratio": 1.1},
        {"text": "for ten for ten for ten", "avg_logprob": -0.2,
         "compression_ratio": 7.2}]))
    assert tr.transcribe(_audio(4.0)).compression_ratio == pytest.approx(7.2)


def test_a_model_without_a_compression_ratio_still_transcribes(firewall):
    """voice_check and the older faster-whisper builds hand back segments
    with no such field; the gate must fail OPEN, not eat the transcript."""
    tr = _cpu(FakeFasterWhisper(segments=[
        SimpleNamespace(text="what time is it", avg_logprob=-0.3)]))
    res = tr.transcribe(_audio(2.0))
    assert res.text == "what time is it" and res.accepted is True


# ------------------------------------------------ what the app does with it
@pytest.fixture
def app(build):          # noqa: F811
    return build()


def _feed(app, monkeypatch, result):        # noqa: F811
    """Push one decoded clip through _process_audio; return the events."""
    from jarvis.events import Transcribed, bus

    seen: list = []
    dispatched: list = []
    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    monkeypatch.setattr(app, "_decode_clip",
                        lambda audio, verify=True: (audio, {}, False, result))
    monkeypatch.setattr(app, "_dispatch",
                        lambda text, source, **kw: dispatched.append(text))
    monkeypatch.setattr(app, "_say", lambda *a, **k: None)
    bus.subscribe(Transcribed, seen.append)
    try:
        app._process_audio(np.zeros(16000, dtype=np.float32))
        bus.drain()
    finally:
        bus.unsubscribe(Transcribed, seen.append)
    return seen, dispatched


def test_a_looping_transcript_is_reported_as_a_loop_not_as_low_confidence(
        app, monkeypatch):
    """The ledger and the UI both print reject_reason; "confidence" would
    send the next person hunting the threshold again."""
    loop = TranscribeResult(text="and the other day, and the other day",
                            confidence=-0.42, compression_ratio=8.81,
                            segments=[("and the other day", -0.42)])
    seen, dispatched = _feed(app, monkeypatch, loop)
    assert dispatched == []
    assert seen and seen[-1].accepted is False
    assert seen[-1].reject_reason == "looping"


def test_a_loop_is_never_salvaged_into_an_open_question(app, monkeypatch):
    """_salvage_low_confidence exists for a length-biased SHORT transcript.
    A 224-token loop is not that, and Jarvis's open yes/no read-back must
    not consume one."""
    import time as _time
    app.commander._pending_destructive = (
        lambda: None, "Clear all three, sir?", _time.monotonic())
    loop = TranscribeResult(text="yes, yes, yes, yes, yes, yes",
                            confidence=-0.42, compression_ratio=6.0,
                            segments=[("yes, yes, yes, yes, yes, yes", -0.42)])
    seen, dispatched = _feed(app, monkeypatch, loop)
    assert dispatched == []
    assert seen[-1].accepted is False and seen[-1].reject_reason == "looping"


# ============================================== the cold start (cause 1)
def test_warmup_runs_one_throwaway_decode_on_synthetic_audio(firewall):
    tr = _gpu()
    assert tr.warmup() is True
    assert len(tr._model.calls) == 1
    buf = tr._model.audio[0]
    assert isinstance(buf, np.ndarray) and buf.dtype == np.float32
    assert len(buf) > 0 and not buf.any(), \
        "the warm-up must never decode microphone audio"


def test_warmup_never_logs_a_transcript_as_if_it_were_the_user(firewall, caplog):
    """Transcriber.transcribe() logs `Transcribed: ...`; a warm-up that went
    through it would put a hallucination in the log above the real turn."""
    tr = _gpu(FakeGpuWhisper(segments=[
        {"text": "Thank you.", "avg_logprob": -0.5, "compression_ratio": 0.5}]))
    with caplog.at_level("INFO"):
        tr.warmup()
    assert "Transcribed:" not in caplog.text


def test_warmup_publishes_nothing(firewall):
    """It must not disturb the turn ledger, the ghost card or the
    speculative bookkeeping -- so it puts nothing on the bus at all."""
    from jarvis.events import bus

    bus.drain()
    tr = _gpu()
    tr.warmup()
    assert bus._q.qsize() == 0


def test_warmup_is_a_no_op_when_the_model_never_loaded(firewall):
    tr = Transcriber()
    assert tr.warmup() is False


def test_warmup_swallows_a_model_that_raises(firewall):
    class Boom:
        def transcribe(self, audio, **kw):
            raise RuntimeError("CUDA out of memory")

    tr = _gpu(Boom())
    assert tr.warmup() is False        # and, crucially, does not raise


def test_warmup_is_bounded_like_a_real_decode(firewall):
    """A warm-up that hallucinated its way to 224 tokens would ADD to the
    startup it exists to shorten."""
    tr = _gpu()
    tr.warmup()
    kw = tr._model.calls[0]
    assert kw["temperature"] == DECODE_TEMPERATURES
    assert kw["sample_len"] <= token_budget(1.0)


def test_the_model_loader_warms_whisper_before_the_first_turn():
    """The startup thread must run it: preloading the weights without ever
    running a kernel is exactly the state the 0.64 s came out of."""
    order: list[str] = []
    app = types.SimpleNamespace(
        brain=SimpleNamespace(warmup=lambda: order.append("brain")),
        _install_endpointer=lambda: order.append("endpointer"),
        transcriber=SimpleNamespace(
            load=lambda: (order.append("whisper-load"), "cuda")[1],
            warmup=lambda: order.append("whisper-warmup")),
        tts=SimpleNamespace(load=lambda: True, prewarm=lambda phrases: None),
        speaker=SimpleNamespace(enrolled=False, load_model=lambda: None),
        _canned_phrases=lambda: [],
    )
    app._load_tts = lambda: app_mod.JarvisApp._load_tts(app)
    app_mod.JarvisApp._load_models(app)
    assert "whisper-warmup" in order, \
        "nothing runs a whisper kernel until the user speaks"
    assert order.index("whisper-load") < order.index("whisper-warmup")
