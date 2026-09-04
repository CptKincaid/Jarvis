"""Whisper's decode was unbounded on bad audio, and cold on the first turn.

NEITHER OF THOSE IS THE TURN THEY WERE WRITTEN FOR, which is the first thing
to know before trusting the numbers below. The ledger line Hunter complained
about (/tmp/vss_voice/jarvis.log, 2026-09-02):

    14:44:00.099 turn: wake->mic 58ms . speech 3.65s . dead-air 800ms
                 . stt 5.23s . route 3.19s . wait 9.22s (stop=vad decode=speculative)

Against the good turns from the night before -- stt 63-132 ms, wait
1.30-1.43 s -- the whole difference is stt. But THAT decode returned a
correct transcript at avg_logprob -0.62, and whisper only re-decodes below
-1.0 or above compression_ratio 2.4, so no ladder ran on it; measured, the
ladder fires on 0 of 61 real clips and the cold start is 0.65 s, not 5.5 s.
What does stretch a clean 0.27 s decode into seconds is interpreter
contention -- one competing pure-Python spin thread makes it 34.2 s, eight
make it 347 s, transcript byte-identical -- and the reactor logged
"avatar: late slots 219/1801 (max lateness 1793.2 ms)" in the same boot
window. That turn is still open; transcriber._log_slow_decode exists to
settle the next one from its own log line.

What IS measured, on this box through the real Transcriber (whisper
"turbo", cuda, fp16, GB10), are two costs on other turns:

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
and some plain misrecognitions used to land below MIN_AVG_LOGPROB. It cuts
BOTH ways and the honest count is in transcriber.py above
DECODE_TEMPERATURES: on 180 clips cut to 0.55/0.85 s at 0 dB SNR, labelled
against the same cut decoded clean, the ladder rejected 24 results and all
24 were garbage; one rung rejects 0 of them. The loop gate recovers the
repetition half of that and nothing else -- 26 of 28 non-speech clips still
come back "Thank you." at compression_ratio 0.56 and are still accepted,
exactly as before this change. The other half cannot be recovered: at one
rung garbage and faithful transcripts overlap completely (garbage median
-0.83 against a worst faithful -1.04), and no_speech_prob, the obvious
replacement signal, is identically 0.0 on turbo.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import jarvis.app as app_mod
from jarvis.config import CONFIG, PATHS
from jarvis import transcriber as tr_mod
from jarvis.transcriber import (
    DECODE_TEMPERATURES,
    LOOP_RATIO_CEILING,
    SAMPLE_RATE,
    WHISPER_SAMPLE_LEN,
    TranscribeResult,
    Transcriber,
    loop_ratio_limit,
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
    """1.64 is the worst of 113 real clips across five decode settings."""
    res = TranscribeResult(text="set a timer for eight minutes", confidence=-0.5,
                           compression_ratio=ratio, audio_seconds=2.0,
                           segments=[("set a timer for eight minutes", -0.5)])
    assert ratio < loop_ratio_limit(2.0)
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


# ================================== the loop gate scales with the clip
# compression_ratio is gzip over a whole decoded window, so it grows with
# the amount of text in the window: measured on real continuous speech
# through this model, 10 s -> 1.25, 22 s -> 1.66, 28 s -> 1.82, 58 s -> 1.86.
# A flat 2.4 is therefore a different gate on a two-word command than on a
# dictation window, and it is tight enough to reject real insistent speech.
@pytest.mark.parametrize("seconds,ratio,is_loop,what", [
    # loops actually produced by this model, at the length that produced them
    (0.40, 4.63, True,  "clean 0.4s cut -> 'do not disturb, but' x N"),
    (0.40, 3.17, True,  "the same runaway, milder"),
    (0.55, 4.16, True,  "0 dB cut -> 'the rest of the day' x N"),
    (0.85, 3.04, True,  "0 dB cut -> 'TAMU, CUDA' x 6"),
    (0.85, 6.20, True,  "0 dB cut -> 'book' x 11"),
    (2.40, 6.71, True,  "white noise -> 'the rest of the day' x N"),
    (30.0, 6.20, True,  "11 repeated sentences inside one window"),
    # real speech, at the length that produced it
    (0.55, 1.71, False, "worst of 90 clean 0.55s cuts"),
    (0.85, 0.75, False, "worst of 90 clean 0.85s cuts"),
    (2.40, 1.22, False, "worst of 90 clean corpus clips"),
    (28.0, 1.82, False, "28 s of continuous real speech"),
    (58.0, 1.86, False, "58 s of continuous real speech"),
    # a person repeating himself at an assistant that is ignoring him
    (4.00, 2.46, False, "'Turn it up,' five times -- over whisper's own 2.4"),
    (3.50, 3.06, False, "'stop' ten times"),
])
def test_the_loop_gate_reads_real_speech_and_real_loops_apart(
        seconds, ratio, is_loop, what):
    res = TranscribeResult(text="...", confidence=-0.4, audio_seconds=seconds,
                           compression_ratio=ratio, segments=[("...", -0.4)])
    assert res.looping is is_loop, what
    assert res.accepted is not is_loop


def test_the_limit_grows_with_the_clip_and_stops_growing():
    """It has to grow (long windows compress better) and it has to stop
    (a 60 s recording is two windows, not one long ratio)."""
    values = [loop_ratio_limit(s) for s in (0.4, 0.55, 0.85, 2.0, 4.0)]
    assert values == sorted(values)
    assert loop_ratio_limit(4.0) == LOOP_RATIO_CEILING
    assert loop_ratio_limit(60.0) == LOOP_RATIO_CEILING


def test_an_unknown_length_fails_open():
    """decode_clip() is a public seam (jarvis/intercom.py hands a clip in
    over the command socket) and callers build TranscribeResult by hand; a
    missing length must not tighten the gate on them."""
    assert loop_ratio_limit(0.0) == LOOP_RATIO_CEILING
    assert loop_ratio_limit(-3.0) == LOOP_RATIO_CEILING
    res = TranscribeResult(text="turn it up, turn it up, turn it up",
                           confidence=-0.4, compression_ratio=3.0,
                           segments=[("turn it up", -0.4)])
    assert res.looping is False


def test_the_gpu_path_carries_the_clip_length_out(firewall):
    """The gate cannot scale with a length the result never carried."""
    tr = _gpu()
    assert tr.transcribe(_audio(3.25)).audio_seconds == pytest.approx(3.25)


def test_the_cpu_path_carries_the_clip_length_out(firewall):
    tr = _cpu()
    assert tr.transcribe(_audio(3.25)).audio_seconds == pytest.approx(3.25)


def test_a_short_loop_and_a_long_insistent_utterance_do_not_collide(firewall):
    """The one pair a flat threshold cannot separate: a 3.04 loop on a
    0.85 s clip and a 3.06 legitimate ten-fold 'stop' over 3.5 s."""
    tr = _gpu(FakeGpuWhisper(segments=[
        {"text": "TAMU, CUDA, TAMU, CUDA", "avg_logprob": -0.4,
         "compression_ratio": 3.04}]))
    assert tr.transcribe(_audio(0.85)).looping is True
    tr = _gpu(FakeGpuWhisper(segments=[
        {"text": "stop stop stop stop stop", "avg_logprob": -0.4,
         "compression_ratio": 3.06}]))
    assert tr.transcribe(_audio(3.5)).looping is False


def test_the_rejection_log_names_the_limit_it_used(firewall, caplog):
    """The limit moves with the clip now, so the ratio alone is not enough
    to retune from -- the log line has to carry the pair."""
    tr = _gpu(FakeGpuWhisper(segments=[
        {"text": "book book book", "avg_logprob": -0.4,
         "compression_ratio": 6.2}]))
    with caplog.at_level("INFO"):
        tr.transcribe(_audio(0.85))
    assert "compression_ratio=6.20" in caplog.text
    assert "> 2.42 for 0.8s of audio" in caplog.text


# ======================= whisper's loop-contagion guard, with one rung
# whisper/transcribe.py:503 -- `if not condition_on_previous_text or
# result.temperature > 0.5: prompt_reset_since = len(all_tokens)`. With
# DECODE_TEMPERATURES=(0.0,) the second half can never fire, so without the
# first half a looping window is fed verbatim as the prompt for the next
# one. recorder.MAX_RECORDING_SECONDS is 60: two windows.
def test_the_gpu_decode_does_not_feed_a_looping_window_to_the_next(firewall):
    tr = _gpu()
    tr.transcribe(_audio(45.0))
    kw = tr._model.calls[0]
    assert kw["condition_on_previous_text"] is False
    assert kw["carry_initial_prompt"] is True, (
        "turning the conditioning off also drops the vocab prompt from every "
        "window after the first unless it is carried explicitly")


def test_the_cpu_decode_does_not_either(firewall):
    tr = _cpu()
    tr.transcribe(_audio(45.0))
    assert tr._model.calls[0]["condition_on_previous_text"] is False


def test_the_preview_keeps_the_vocab_prompt_across_windows(firewall):
    tr = _gpu()
    tr.partial(_audio(45.0))
    assert tr._model.calls[0]["carry_initial_prompt"] is True


def test_the_cpu_preview_never_climbs_the_ladder_either(firewall):
    """faster_whisper's own default temperature is the SIX-rung ladder, so
    the preview -- which runs several times a second holding the lock the
    final decode waits on -- is unbounded unless it is told otherwise."""
    tr = _cpu()
    tr.partial(_audio(2.0))
    assert tr._model.calls[0]["temperature"] == DECODE_TEMPERATURES


# ============================ the warm-up must give way to a real turn
# app.main() calls start_background() -- which starts the hotword -- before
# when_cycle_live(start_models), and warmup() is the last step of
# _load_models. transcribe() and partial() take the same lock. Seen live:
# jarvis.log.1 "20:56:40.277 Model loaded on GPU fp16" then a decode at
# 20:56:42.395, inside that window.
def test_warmup_does_nothing_once_a_real_decode_has_run(firewall):
    tr = _gpu()
    tr.transcribe(_audio(2.0))
    assert tr._decoded is True
    calls = len(tr._model.calls)
    assert tr.warmup() is False
    assert len(tr._model.calls) == calls, \
        "the kernels are warm; a second decode only costs the user time"


def test_a_preview_also_counts_as_warm(firewall):
    tr = _gpu()
    tr.partial(_audio(2.0))
    assert tr._decoded is True
    assert tr.warmup() is False


def test_warmup_gives_up_rather_than_making_the_mic_path_wait(firewall):
    tr = _gpu()
    assert tr._lock.acquire(blocking=False) is True   # stand in for a turn
    try:
        assert tr.warmup() is False
        assert tr._model.calls == []
    finally:
        tr._lock.release()


def test_warmup_hands_the_lock_back(firewall):
    tr = _gpu()
    assert tr.warmup() is True
    assert tr._lock.acquire(blocking=False) is True, \
        "a warm-up that kept the lock would deadlock the first real turn"
    tr._lock.release()


def test_a_raising_warmup_hands_the_lock_back_too(firewall):
    class Boom:
        def transcribe(self, audio, **kw):
            raise RuntimeError("CUDA out of memory")

    tr = _gpu(Boom())
    assert tr.warmup() is False
    assert tr._lock.acquire(blocking=False) is True
    tr._lock.release()


# ============================== telling a starved decode from a slow one
def test_a_slow_decode_says_whether_the_process_was_starved(caplog):
    """One competing spin thread turns a 0.27 s decode into 34 s with the
    same transcript; whisper doing six re-decodes burns CPU on this thread
    the whole time. "stt 5.23s" cannot tell those apart -- this can."""
    with caplog.at_level("WARNING"):
        tr_mod._log_slow_decode(wall=5.58, lock_wait=0.0, cpu=0.06,
                                seconds=3.3)
    assert "slow decode" in caplog.text
    assert "5.58s wall" in caplog.text and "0.06s cpu" in caplog.text
    assert "starved" in caplog.text


@pytest.mark.parametrize("wall,seconds", [
    (0.31, 3.0),      # warm decode of a real turn
    (0.53, 10.0),     # 10 s of continuous speech
    (0.86, 28.0),     # 28 s
    (2.28, 58.0),     # a full 60 s recording, two windows
])
def test_a_clean_decode_never_cries_wolf(wall, seconds, caplog):
    with caplog.at_level("WARNING"):
        tr_mod._log_slow_decode(wall=wall, lock_wait=0.0, cpu=wall,
                                seconds=seconds)
    assert caplog.text == ""


def test_the_real_decode_is_the_one_being_timed(firewall, monkeypatch):
    """The diagnostic is worthless if it is wired to the wrong number."""
    seen: list = []
    monkeypatch.setattr(tr_mod, "_log_slow_decode",
                        lambda wall, lock_wait, cpu, seconds:
                        seen.append((wall, lock_wait, cpu, seconds)))
    tr = _gpu()
    tr.transcribe(_audio(4.0))
    assert len(seen) == 1
    wall, lock_wait, cpu, seconds = seen[0]
    assert seconds == pytest.approx(4.0)
    assert wall >= 0.0 and lock_wait >= 0.0 and cpu >= 0.0
    assert lock_wait <= wall


def test_warmup_does_not_warm_up_twice(firewall):
    """_load_models calls it once, but the flag means "a kernel has run" --
    a second call has nothing to do and must not take the lock to find out."""
    tr = _gpu()
    assert tr.warmup() is True
    assert tr.warmup() is False
    assert len(tr._model.calls) == 1


# ================= the bound a turn CAN wait on the warm-up (F49)
# The docstring used to say a real turn could never wait on the warm-up.
# The lock is taken non-blocking by warmup() and BLOCKING by transcribe()
# and partial(), so a capture that ends while the throwaway decode holds
# the lock waits for the rest of it -- a whisper kernel cannot be
# abandoned. The honest bound: one silent decode, then the turn decodes
# warm. These pin that bound and the one ordering the app can still avoid.
class SlowFirstDecode(FakeGpuWhisper):
    """The first decode (the warm-up) takes ``hold`` seconds; every one
    after it is instant, which is what a warm kernel looks like."""

    def __init__(self, hold):
        super().__init__()
        self.hold = hold
        self.finished: list[float] = []

    def transcribe(self, audio, **kw):
        import time
        if not self.calls:
            time.sleep(self.hold)
        out = super().transcribe(audio, **kw)
        self.finished.append(time.monotonic())
        return out


def test_a_turn_landing_inside_the_warmup_waits_for_it_and_no_longer(firewall):
    import threading
    import time

    hold = 0.15
    tr = _gpu(SlowFirstDecode(hold))
    warm = threading.Thread(target=tr.warmup, daemon=True)
    warm.start()
    for _ in range(200):                       # until the warm-up holds the lock
        if tr._lock.locked():
            break
        time.sleep(0.005)
    assert tr._lock.locked(), "the warm-up never took the lock"
    t0 = time.monotonic()
    got = tr.transcribe(_audio(2.0))
    waited = time.monotonic() - t0
    warm.join(timeout=2.0)
    assert got.text == "hello", "the turn's decode ran, after the warm-up"
    assert len(tr._model.calls) == 2, "warm-up, then the turn: never dropped"
    assert tr._model.finished[0] <= t0 + waited, \
        "the turn decoded before the warm-up let go of the model"
    assert waited <= hold + 0.5, \
        f"waited {waited:.2f}s on a {hold:.2f}s warm-up: more than one decode"


def _loader_app(order, **extra):
    app = types.SimpleNamespace(
        brain=SimpleNamespace(warmup=lambda: order.append("brain")),
        _install_endpointer=lambda: order.append("endpointer"),
        transcriber=SimpleNamespace(
            load=lambda: (order.append("whisper-load"), "cuda")[1],
            warmup=lambda: order.append("whisper-warmup")),
        tts=SimpleNamespace(load=lambda: True, prewarm=lambda phrases: None),
        speaker=SimpleNamespace(enrolled=False, load_model=lambda: None),
        _canned_phrases=lambda: [],
        **extra,
    )
    app._load_tts = lambda: app_mod.JarvisApp._load_tts(app)
    return app


def test_the_loader_skips_the_warmup_when_a_capture_is_open():
    """A capture already open when the loader gets here has a decode
    seconds away; the throwaway decode would only stand in front of it."""
    import threading
    order: list[str] = []
    app = _loader_app(order, recorder=SimpleNamespace(recording=True),
                      _audio_busy=threading.Event())
    app_mod.JarvisApp._load_models(app)
    assert "whisper-load" in order
    assert "whisper-warmup" not in order


def test_the_loader_skips_the_warmup_while_a_turn_is_being_processed():
    """recorder.recording is already False during the decode pass;
    _audio_busy is the flag that says a turn is in flight (CLAUDE.md)."""
    import threading
    order: list[str] = []
    busy = threading.Event()
    busy.set()
    app = _loader_app(order, recorder=SimpleNamespace(recording=False),
                      _audio_busy=busy)
    app_mod.JarvisApp._load_models(app)
    assert "whisper-warmup" not in order


def test_the_loader_still_warms_when_the_mic_is_idle():
    import threading
    order: list[str] = []
    app = _loader_app(order, recorder=SimpleNamespace(recording=False),
                      _audio_busy=threading.Event())
    app_mod.JarvisApp._load_models(app)
    assert "whisper-warmup" in order
