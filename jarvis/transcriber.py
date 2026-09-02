"""Whisper transcription for Jarvis V3.

Owns the faster-whisper model and the lock that serializes access between
full transcription and streaming partial previews (the legacy monolith shared
one model between _transcribe_worker and _partial_transcribe_worker via
_partial_lock, voice_input_gui.py:1169).

Machine reality (config.MACHINE): the DGX Spark's ctranslate2 has no CUDA,
but torch cu130 does — so load() prefers the openai-whisper package on CUDA
("GPU fp16") whenever torch.cuda.is_available(). The faster-whisper CPU int8
path is kept as the fallback with unchanged semantics; the legacy CUDA
ctranslate2 attempt (2200-2228) survives only for machines where CUDA
ctranslate2 exists.

This module does transcription ONLY: no command routing, no speaker-segment
filtering (commander/pipeline calls speaker.filter_segments before handing
audio here), no widget access. Pure API — callers publish bus events.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from jarvis.config import CONFIG, MACHINE, PATHS
from jarvis.logs import get_logger

log = get_logger("transcriber")

# Audio sample rate all pipeline audio uses (port: voice_input_gui.py:53)
SAMPLE_RATE = 16000

# Streaming: partial transcription interval, seconds (port: 348)
STREAMING_INTERVAL = 2.0

# Confidence gate: reject transcriptions whose mean segment avg_logprob is
# below this (port: 2602-2604). A rejection is spoken ("Say that again,
# sir?") and re-opens the mic; a SECOND one in a row is silent.
#
# -0.85 (set from "genuine utterances on this mic score -0.2..-0.6") was
# measured against long sentences only, and Whisper's avg_logprob is
# length-biased: a two-word clip has few tokens, so the end-of-text token
# dominates the mean and a PERFECT transcript lands near -1. That cost
# Hunter nine features in the 2026-08-31 session -- "belay that" (-0.88),
# "scratch that" (-0.89), "volume 40" (-0.87), "Play my liked songs."
# (-0.97), "max volume" (-2.08), "Full brightness" (-0.99) and, worst, a
# plain "Yes." (-0.95) answering Jarvis's own "Clear all three off your
# shopping list, sir?" -- every one of them transcribed CORRECTLY and
# thrown away.
#
# Re-measured on all 196 `Transcribed:` lines in that session's
# /tmp/vss_voice/jarvis.log (scripts/measure_confidence_gate.py):
#   band              n    real utterance   hallucination
#   >= -0.85        156         156                0
#   -2.90..-0.85     19          18                1   <- ALL rejected before
#   <  -2.90         21           0               21
# The band the old gate threw away holds 15 verbatim commands ("volume
# 40", "Belay that", "scratch that", "Cancel.", "Yes.", "max volume",
# "Play my liked songs.", "Full brightness", ...), 3 imperfect renderings
# of something he really said ("whether tomorrow whether tomorrow"), and
# exactly ONE hallucination: "by Agenda 4.2.6" at -0.94, the clip this
# threshold was tightened for in the first place. Below -2.90 there is not
# one actionable utterance -- two empty strings, repeated-token loops
# ("certain seal seal seal seal seal"), and character salad ("mmmен-4222,
# 902", "쪽 km2mmh gleichzeitig").
#
# So no threshold can separate "Play my liked songs." (-0.97) from "by
# Agenda 4.2.6" (-0.94); they are the same score. -2.90 is the midpoint of
# the real gap in the data (worst true transcript "Yes" at -2.64, best
# hallucination the empty string at -3.15) and it trades that one stray
# noun phrase -- which routes to chat and earns a harmless "I don't
# follow, sir" -- for the 18 real commands the old gate ate.
#
# The gate is also no longer the last word -- app._salvage_low_confidence
# runs a sub-threshold transcript anyway when Jarvis asked the question or
# the words are an exact Tier-1 command.
#
# READ THE BLOCK BELOW DECODE_TEMPERATURES BEFORE TRUSTING ANY OF THIS. All
# 196 lines above came out of a SIX-RUNG decoder, and this module now runs
# one rung. The distribution moved with it: the deep-negative scores that
# made -2.90 a working filter were the temperature-1.0 samples of an
# exhausted ladder, and on one greedy pass the same clips score -2.54 at
# worst. Re-measured, the value is right to KEEP -- nothing separates the
# classes at one rung either -- but it is now a floor against degenerate
# output, not the filter this table describes.
MIN_AVG_LOGPROB = -2.90

# ------------------------------------------------------------------
# The bound on the decode
# ------------------------------------------------------------------
# FIRST, WHAT THESE BOUNDS DO NOT EXPLAIN. They were written for the turn
# Hunter complained about -- /tmp/vss_voice/jarvis.log 14:44:00.099,
# "stt 5.23s ... wait 9.22s", against 63-132 ms of stt on the good turns the
# night before -- and neither one can produce it. That decode returned a
# CORRECT transcript at avg_logprob -0.62, and whisper only re-decodes below
# -1.0 or above compression_ratio 2.4, so the ladder never ran on it; over 61
# real clips it fires 0/61 times. Cold start is 0.65 s, not 5.5 s.
#
# What DOES turn a clean 0.27 s decode into a multi-second one is contention
# for the interpreter: same process, same clip, N pure-Python spin threads
# competing --
#
#     N        0       1       2        4        8
#     decode   0.27s   34.2s   50.5s    122.8s   347.2s
#
# -- byte-identical transcript every time, which is exactly the shape of the
# turn he reported. The reactor logged "avatar: late slots 219/1801 (max
# lateness 1793.2 ms)" inside that same boot window. THE 14:43:56 TURN IS
# STILL UNDIAGNOSED (the 14:29:23 one is not starvation either: its reactor
# window reported 0/900 late slots). _log_slow_decode() below exists so the
# next one is diagnosed from its own log line instead of from a rerun.
#
# What the bounds below DO fix is measured and real: on BAD audio whisper's
# decode is unbounded in two directions, and the first decode of a process
# is cold.
#
#  * THE TEMPERATURE LADDER. openai-whisper re-decodes a window at
#    (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) whenever the greedy pass trips
#    logprob_threshold=-1.0 or compression_ratio_threshold=2.4, and returns
#    whichever rung first passes -- or the temperature-1.0 SAMPLE when none
#    does. Six full decodes of the same audio.
#  * SAMPLE_LEN. Each of those passes may emit n_text_ctx // 2 = 224 tokens,
#    so a repetition loop runs to the cap rather than to the end of the
#    sentence.
#
# Measured here (whisper "turbo", cuda, fp16, GB10) over 20 non-speech clips
# through the real Transcriber.transcribe():
#
#    whisper's own defaults    total 49.5 s   worst single decode 9.70 s
#    the bounds below          total  9.1 s   worst single decode 1.00 s
#
# which is the shape of FOUR of the six slow decodes in his log -- 9.55 s,
# 5.52 s, 5.18 s, 3.48 s, every one of them returning character salad,
# against 0.02-0.45 s for every clean decode. The other two (5.58 s and
# 6.55 s) returned correct transcripts and are the ones above that this
# does not explain.
#
# THE LADDER IS NOT A TRADE. It was measured against the accuracy it is
# supposed to buy, on his own failure shape: 180 corpus clips cut to
# 0.55 s / 0.85 s -- the length that makes avg_logprob collapse, see
# MIN_AVG_LOGPROB above -- and mixed to 0 dB SNR. 34 of them tripped
# logprob_threshold and went up the ladder. WER against the same cut
# decoded clean:
#
#    rungs              WER (all 180)   WER (the 34)   their decode time
#    6 (whisper stock)      0.4609         1.5216         40.19 s
#    2                      0.6257         2.3941         15.25 s
#    1 (greedy only)        0.3882         1.1368          7.15 s
#
# One rung is the most ACCURATE as well as the fastest, and two is the
# worst of the three: a short clip's greedy transcript is usually right and
# merely scores badly, so replacing it with one temperature-0.2 draw is a
# straight loss, while six draws only claw part of it back. Whisper is
# throwing away the right answer because a length-biased mean says so --
# the identical mistake the old -0.85 confidence gate made.
#
# On speech that does NOT trip it there is nothing to trade at all. Over 113
# real clips (the voice corpus, his own mic captures, a 20 s dictation run)
# the bounded decode returned the IDENTICAL transcript 113/113 with the same
# median/p95/max time; over 160 of those degraded to 0 dB and -3 dB SNR the
# ladder fired ZERO times -- every one passed at temperature 0.0, which makes
# one rung and six the same decode by construction -- and WER was 0.1843
# either way.
#
# WHAT IT COSTS, MEASURED IN BOTH DIRECTIONS. An earlier draft of this
# comment claimed the loop gate below "closes a hole rather than opening
# one". Half of that was wrong. Re-measured on the shipped model
# (turbo/cuda/fp16), 180 corpus clips cut to 0.85 s and 0.55 s and mixed to
# 0 dB SNR, each result labelled by WER against THE SAME CUT DECODED CLEAN
# -- not against the full sentence the cut threw away, which is what makes
# every short clip look like a failure:
#
#     rungs               faithful   garbage   rejected by -2.90   garbage
#     6 (whisper stock)      48        132            24             24
#     1 (shipped)            47        133             0              0
#
# The ladder's 24 rejections were 24/24 garbage and cost no faithful
# transcript, so removing it DOES open a hole: on audio this bad the accept
# rate goes 82% -> 100%. A narrow hole -- 82% of that garbage already
# reached the commander -- but a hole, and it is in the residual-risk list.
#
# It cannot be closed by retuning MIN_AVG_LOGPROB, and it cannot be closed
# with no_speech_prob. Both were measured, not assumed:
#
#  * ONE-RUNG SCORES OVERLAP COMPLETELY. Garbage bottoms out at -2.54 (the
#    ladder drove those same clips to -6.67) and its MEDIAN is -0.83, while
#    the worst faithful transcript scores -1.04. A gate at -1.20 would match
#    the ladder's recall (20.3% of garbage, 0/47 faithful lost on this
#    corpus) -- but it would sit 0.16 below the worst faithful score
#    measured, and the two log lines -2.90 was widened FOR ("max volume"
#    -2.08 and "Yes" -2.64, both transcribed correctly) are cases where the
#    ladder DID fire, so their greedy score was never logged and may well
#    land inside that band. Re-opening the 2026-08-31 regression that ate
#    nine correct commands is not worth 20% of the garbage on audio this
#    bad. -2.90 stays, but it is now a floor against degenerate output
#    rather than a working filter, and the comment above it is history.
#  * no_speech_prob -- whisper's own "this window is not speech" signal, and
#    the obvious replacement -- IS IDENTICALLY 0.0 ON TURBO. Silence, noise,
#    hum, real speech; our settings and whisper's own defaults; 28
#    non-speech clips plus a stock-settings probe: 0.0 every time. There is
#    nothing to gate on. (It is a live signal on smaller checkpoints, which
#    is why measuring on the model that actually ships mattered here.)
#  * WORDS PER SECOND does not separate either: faithful and garbage share
#    the same median (3.5/s on the 0.85 s cuts, 3.6/s on the 0.55 s cuts).
#    Only the runaway tail differs, and that is what the loop gate catches.
DECODE_TEMPERATURES = (0.0,)

# Tokens a real utterance can contain. The fastest of those 113 clips
# emitted 8.89 tokens/second ("That didn't work, sir" cut to 0.9 s); the
# longest run needed 69 tokens for 20 s of continuous speech. 12/s plus a
# 32-token allowance for the sot sequence, the timestamps and the odd long
# word keeps at least 1.7x headroom at every length -- and from ~16 s up the
# budget IS Whisper's own 224, so dictation of a full window is untouched.
WHISPER_SAMPLE_LEN = 224        # openai-whisper's n_text_ctx // 2 for turbo
TOKENS_PER_SECOND = 12
TOKEN_FLOOR = 32

# ------------------------------------------------------------------
# The loop gate
# ------------------------------------------------------------------
# compression_ratio is whisper's own "too repetitive" signal, and it has to
# be read out HERE now that the ladder is gone: compression_ratio_threshold
# was the trigger that used to send a looping window back for another draw,
# and the confidence gate was living off the wreckage (an exhausted ladder
# returns the temperature-1.0 sample, whose avg_logprob is deeply negative).
# On one greedy pass the same non-speech clip comes back as "and the rest of
# the day, the rest of the day, ..." at avg_logprob -0.31 and -2.90 waves it
# straight through.
#
# THE LIMIT SCALES WITH THE CLIP, because compression_ratio does. It is gzip
# over a whole decoded window, so it grows with the amount of text in the
# window -- measured on real continuous speech through this model:
#
#     10 s -> 1.25   16 s -> 1.58   22 s -> 1.66   28 s -> 1.82   58 s -> 1.86
#
# (it plateaus past 30 s because whisper computes the ratio per window and
# transcribe() takes the worst window, not the whole transcript). A flat 2.4
# is therefore far tighter on a dictation window than on a two-word command
# -- recorder.MAX_RECORDING_SECONDS is 60 and commander.py has a dictation
# mode -- and it is tight enough to reject real speech: "Turn it up, turn it
# up, turn it up, turn it up, turn it up." measures 2.46 and ten "stop"s
# 3.06. Both are things a person says to an assistant that is ignoring him,
# and collapse_repeats cannot rescue either (it drops whole repeated
# SENTENCES of >= 3 words, split on .!?, so comma-joined repetition inside
# one sentence survives it).
#
# So: 2.0 + 0.5 per second of audio, capped at 4.0 (reached at 4 s), and 4.0
# when the duration is unknown -- the gate fails OPEN. Every measured point
# clears it by at least 1.23x, and the scale is what separates the two cases
# a flat threshold cannot tell apart: a 3.04 loop on a 0.85 s clip and a
# 3.06 legitimate ten-fold "stop" over ~3.5 s.
#
#     audio    limit   worst real speech there   loops measured there
#     0.4 s     2.20   1.71 (at 0.55 s)          3.17, 4.63 "do not
#                                                 disturb, but" x N
#     0.55 s    2.28   1.71 of 90 clean cuts     4.16 x3, 6.59, 7.89
#     0.85 s    2.43   0.75 of 90 clean cuts     3.04, 5.47, 6.20
#     2.4 s     3.20   1.22 of 90 clean clips    6.71 (noise -> "and the
#                                                 rest of the day" x N)
#     >= 4 s    4.00   1.86 at 28-58 s           6.20 (11 repeated
#                                                 sentences)
#
# WHAT THIS DOES NOT DO: it closes the LOOPING SUBSET of non-speech, not
# non-speech. 26 of 28 synthetic non-speech clips decode to "Thank you." at
# compression_ratio 0.56 and avg_logprob -0.39..-1.56 and are accepted
# exactly as they were before -- whisper's stock hallucination on silence is
# still unfiltered, as it was before this change. Only the 2 that actually
# looped are newly rejected.
LOOP_RATIO_FLOOR = 2.0
LOOP_RATIO_PER_SECOND = 0.5
LOOP_RATIO_CEILING = 4.0


def loop_ratio_limit(seconds: float) -> float:
    """compression_ratio above this is a repetition loop, not speech."""
    if not seconds or seconds <= 0:
        return LOOP_RATIO_CEILING       # unknown length -> the loosest limit
    return min(LOOP_RATIO_CEILING,
               LOOP_RATIO_FLOOR + LOOP_RATIO_PER_SECOND * seconds)


def token_budget(seconds: float) -> int:
    """Cap on the tokens one decode pass may emit, from the clip's length."""
    return int(min(WHISPER_SAMPLE_LEN,
                   TOKEN_FLOOR + TOKENS_PER_SECOND * max(0.0, seconds)))


def _seconds(audio) -> float:
    try:
        return len(audio) / float(SAMPLE_RATE)
    except Exception:
        return 0.0


# ------------------------------------------------------------------
# Saying WHY a decode was slow
# ------------------------------------------------------------------
# The 14:43:56 turn at the top of this file is still undiagnosed because
# all the log had was "stt 5.23s". Two very different faults produce that
# line and they live in different modules: whisper actually working
# (re-decoding, running to the token cap) burns CPU on THIS thread the whole
# time, while a starved process does not -- one competing pure-Python spin
# thread stretches a 0.27 s decode to 34 s while its own CPU time barely
# moves, and the transcript comes back identical. So the diagnostic prints
# wall, lock wait, this thread's CPU and the live thread count: cpu far
# below wall, with nothing waiting on the lock, is starvation and the fix is
# somewhere else entirely.
#
# The floor is a function of the clip because a long clip is legitimately
# slow: measured warm on this box, 0.25-0.31 s for a 1-3 s clip, 0.53 s at
# 10 s, 0.86 s at 28 s, 2.28 s at 58 s of continuous speech. 1.0 + 0.05/s
# sits 1.7x-3.4x above every one of those, so a clean decode never trips it.
SLOW_DECODE_FLOOR = 1.0
SLOW_DECODE_PER_SECOND = 0.05


def _log_slow_decode(wall: float, lock_wait: float, cpu: float,
                     seconds: float) -> None:
    if wall < SLOW_DECODE_FLOOR + SLOW_DECODE_PER_SECOND * max(0.0, seconds):
        return
    log.warning(
        "slow decode: %.2fs wall on %.2fs of audio (%.2fs waiting for the "
        "model lock, %.2fs cpu on this thread, %d threads alive) -- cpu far "
        "below wall with no lock wait means the process was starved, not "
        "whisper", wall, seconds, lock_wait, cpu, threading.active_count())

# Default vocabulary prompt — biases Whisper toward the assistant's own
# domain. Replaces the warehouse list ported verbatim from
# voice_input_gui.py:78-86 ("AGV, forklift, pallet, conveyor, ..."):
# with no ~/.aiws_trainer/voice_vocab.txt on this box, every utterance
# was primed with VSS jargon he never says. This seed is the floor;
# jarvis/vocab.py layers his names, corrections, calendar titles and
# Canvas courses on top when app.py wires a prompt_provider in.
DEFAULT_VOCAB = (
    "Jarvis, Hunter, Spark, DGX, GB10, Ollama, Whisper, XTTS, "
    "Claude, Claude Code, Canvas, Spotify, Gmail, Discord, iCloud, TAMU, "
    "calendar, briefing, timer, alarm, reminder, flashcards, quiz, "
    "pomodoro, lecture notes, standup, quiet hours, do not disturb, "
    "syllabus, assignment, announcement, office hours, "
    "terminal, tmux, GPU, CUDA, pytest, playlist, weather"
)

# Language options (port verbatim: voice_input_gui.py:351-372)
LANGUAGES = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Turkish", "tr"),
    ("Polish", "pl"),
    ("Ukrainian", "uk"),
    ("Vietnamese", "vi"),
    ("Thai", "th"),
    ("Swedish", "sv"),
]
LANG_MAP = dict(LANGUAGES)   # name -> whisper code (None = auto-detect)


# ------------------------------------------------------------------
# Custom vocabulary (persisted to voice_vocab.txt; port: 579-595)
# ------------------------------------------------------------------
# Whisper's stutter. On a short clip it sometimes emits the SAME sentence
# twice ("What are on both lists? What are on both lists?" -- reported
# 2026-08-31, feature #31: his question came back at him twice before the
# answer). It is a decode artefact, not speech: nobody says a five-word
# question twice with no pause. Collapsed here, at the one place both the
# speculative pass and the final pass come through, so the card, the log,
# the model and the tier-1 matchers all see the sentence once.
_SENTENCE_RX = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_REPEAT_MIN_WORDS = 3          # "No. No." is emphasis; leave it alone


def collapse_repeats(text: str) -> str:
    """Drop a sentence that is an immediate repeat of the one before it."""
    raw = (text or "").strip()
    if not raw:
        return raw
    parts = [p.strip() for p in _SENTENCE_RX.findall(raw)]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return raw
    kept, dropped = [], 0
    for part in parts:
        key = re.sub(r"[^a-z0-9 ]+", "", part.lower())
        key = " ".join(key.split())
        prev = kept[-1][1] if kept else None
        if key and key == prev and len(key.split()) >= _REPEAT_MIN_WORDS:
            dropped += 1
            continue
        kept.append((part, key))
    if not dropped:
        return raw
    out = " ".join(p for p, _ in kept)
    log.info("collapsed %d repeated sentence(s): %r -> %r", dropped, raw, out)
    return out


def load_vocab() -> str:
    """Load domain vocabulary from user file, or return default."""
    if PATHS.VOCAB_FILE.exists():
        try:
            text = PATHS.VOCAB_FILE.read_text().strip()
            if text:
                return text
        except Exception:
            log.exception("vocab read failed; using default")
    return DEFAULT_VOCAB


def save_vocab(text: str):
    """Save domain vocabulary to user file."""
    PATHS.VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
    PATHS.VOCAB_FILE.write_text(text.strip())
    log.info("Vocabulary saved to %s", PATHS.VOCAB_FILE)
    # The dynamic prompt (jarvis/vocab.py) caches for 60 s; a word learned
    # from a correction must reach the very next attempt, not the one
    # after the TTL. Lazy import: vocab imports this module at top level.
    try:
        from jarvis import vocab as _vocab
        _vocab.clear_cache()
    except Exception:
        pass


# ------------------------------------------------------------------
# Result
# ------------------------------------------------------------------
@dataclass
class TranscribeResult:
    text: str = ""
    confidence: float = 0.0            # mean segment avg_logprob (0.0 if no segments)
    segments: list = field(default_factory=list)   # [(seg_text, avg_logprob), ...]
    language: str | None = None        # detected/forced language code
    # WORST segment compression ratio, not the mean: one looping segment
    # poisons the turn even when the others are clean, and a mean hides it.
    # 0.0 when the backend does not report one -- the gate fails OPEN.
    compression_ratio: float = 0.0
    # Length of the audio that produced this, seconds. The loop gate needs
    # it (see loop_ratio_limit); 0.0 means "unknown", which fails OPEN.
    audio_seconds: float = 0.0

    @property
    def looping(self) -> bool:
        """Whisper ran away repeating itself. See loop_ratio_limit: real
        speech measured 0.33-1.86 at every length tried, the loops 3.04-7.89
        on clips of 0.4-0.85 s, and their avg_logprob (-0.31 to -0.42) is no
        help at all."""
        return self.compression_ratio > loop_ratio_limit(self.audio_seconds)

    @property
    def accepted(self) -> bool:
        """Confidence gate (port: 2599-2607). No segments -> accept as-is
        (legacy skipped the gate when seg_data was empty)."""
        if self.looping:
            return False
        if not self.segments:
            return True
        return self.confidence >= MIN_AVG_LOGPROB


# ------------------------------------------------------------------
# Transcriber
# ------------------------------------------------------------------
class Transcriber:
    """Single owner of the Whisper model.

    self._lock serializes model access between transcribe() (full) and
    partial() (streaming preview) — port of _partial_lock (1169, 2592, 2716).
    """

    def __init__(self, prompt_provider=None):
        self._model = None
        self._backend = ""
        self._gpu = False                    # True when openai-whisper on CUDA
        self._lock = threading.Lock()        # model access (full + partial)
        self._load_lock = threading.Lock()   # one-time load
        # Called before EVERY pass (full and partial) to build the
        # initial_prompt — app.py wires jarvis.vocab.build_prompt in.
        # None keeps the legacy per-call load_vocab() behaviour
        # (voice_check.py and scripts construct Transcriber() bare).
        self._prompt_provider = prompt_provider
        # Set by the first successful decode of either kind. warmup() reads
        # it: once a real turn has run a kernel there is nothing left to
        # warm, and the warm-up must not take the lock that turn's
        # successor needs. Plain bool, written and read under the GIL.
        self._decoded = False
        vocab = load_vocab()
        log.info("vocab: %d chars (%s)%s", len(vocab),
                 "custom file" if PATHS.VOCAB_FILE.exists() else "default",
                 "; dynamic prompt provider wired" if prompt_provider else "")

    # -- vocab ----------------------------------------------------------
    @property
    def vocab(self) -> str:
        """Current vocabulary text (re-read from file, like the legacy
        per-transcription _load_vocab() calls at 2584/2719)."""
        return load_vocab()

    @vocab.setter
    def vocab(self, text: str):
        save_vocab(text)

    def _prompt(self) -> str:
        """initial_prompt for the next pass: the provider's live prompt
        (names, calendar titles, courses — jarvis/vocab.py) when one is
        wired, else the legacy vocab file / default. A provider failure
        must never cost a transcription, so it degrades to load_vocab().
        NOT routed through the ``vocab`` setter: that would persist the
        dynamic prompt to ~/.aiws_trainer on every turn, and the user
        file is a manual layer the builder reads, not a mirror."""
        if self._prompt_provider is not None:
            try:
                text = (self._prompt_provider() or "").strip()
                if text:
                    return text
            except Exception:
                log.exception("prompt provider failed; using vocab file")
        return load_vocab()

    # -- model ----------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> str:
        return self._backend

    def load(self) -> str:
        """Load the Whisper model (idempotent, blocking). Returns backend
        string: "GPU fp16" (openai-whisper on CUDA, preferred when torch sees
        a GPU) or "CPU int8" (faster-whisper fallback, unchanged semantics —
        port of _load_model_worker 2200-2228 with the doomed CUDA ctranslate2
        attempt skipped when MACHINE.no_cuda_ct2)."""
        with self._load_lock:
            if self._model is not None:
                return self._backend

            model_size = CONFIG.model
            gpu = CONFIG.gpu

            # GPU first: openai-whisper on torch CUDA (fp16). CONFIG.model
            # names (tiny/base/small/medium/large) are valid in both packages.
            try:
                import torch
                if torch.cuda.is_available():
                    import whisper
                    log.info("Loading model: %s (openai-whisper, cuda)",
                             model_size)
                    model = whisper.load_model(model_size, device="cuda")
                    self._model = model
                    self._backend = "GPU fp16"
                    self._gpu = True
                    log.info("Model loaded on %s", self._backend)
                    return self._backend
            except Exception:
                log.exception(
                    "GPU whisper load failed; falling back to faster-whisper")

            from faster_whisper import WhisperModel

            log.info("Loading model: %s (no_cuda_ct2=%s)",
                     model_size, MACHINE.no_cuda_ct2)

            if MACHINE.no_cuda_ct2:
                model = WhisperModel(model_size, device="cpu",
                                     compute_type="int8")
                backend = "CPU int8"
            else:
                try:
                    model = WhisperModel(
                        model_size, device="cuda",
                        device_index=gpu, compute_type="float16",
                    )
                    backend = f"CUDA:{gpu}"
                except Exception:
                    log.exception("CUDA load failed; falling back to CPU int8")
                    model = WhisperModel(model_size, device="cpu",
                                         compute_type="int8")
                    backend = "CPU int8"

            self._model = model
            self._backend = backend
            self._gpu = False
            log.info("Model loaded on %s", backend)
            return backend

    # -- cold start -----------------------------------------------------
    # 0.5 s: whisper pads every clip to 30 s anyway, so the length buys
    # nothing; this only has to be long enough to be a legal clip.
    WARMUP_SECONDS = 0.5

    def warmup(self) -> bool:
        """Run one throwaway decode so the USER never pays the cold start.

        The weights are already resident by here (app._preload_heavy_imports
        logs "preloaded torch/whisper + CUDA in 1.2s"), but nothing has run
        a whisper kernel, so the first real inference of the process pays
        autotune and the caching allocator's first fill. A/B on one 3.3 s
        mic capture, fresh process each way, twice:

            no warm-up   first user turn  0.951 s   (0.944 / 0.870 / 0.927)
            warm-up      first user turn  0.252 s   (0.285 / 0.258 / 0.247)

        THE LEDGER IS NOT FREE, and the honest version is: 0.838 s spent
        (0.885 / 0.805 / 0.823) to save 0.651 s (0.923 s mean cold first
        decode against 0.272 s warm). It is net-negative on total machine
        time and it pushes tts.prewarm() and gc.freeze() ~0.85 s later on
        every boot -- in the live log the whole loader tail after
        _install_endpointer() was 23 ms (14:43:44.761 -> "models loaded;
        heap frozen" 14:43:44.784). It is still the right trade, because the
        0.65 s it removes is the only part of the ledger the USER is sitting
        through.

        IT MUST NOT BECOME THE THING A TURN WAITS ON. The mic is already
        live when this runs: app.main() calls start_background() (which
        starts the hotword) before when_cycle_live(start_models), and this
        is the last step of _load_models. transcribe() and partial() take
        the same lock. So the warm-up gives way twice -- it does nothing if
        a real decode already ran, and it takes the lock non-blocking. Seen
        live: jarvis.log.1 has "20:56:40.277 Model loaded on GPU fp16" and a
        decode at 20:56:42.395, right inside that window.

        Silence, never microphone audio, and NOT through transcribe(): that
        would log "Transcribed: 'Thank you.'" -- whisper's stock
        hallucination on silence -- into the log directly above his real
        turn. Nothing is published, nothing is returned, and a failure is
        swallowed: a box that cannot warm up must still be able to listen.

        Returns True when a decode actually ran.
        """
        model = self._model
        if model is None:
            return False
        if self._decoded:
            log.info("whisper warm-up skipped: a real decode already ran")
            return False
        if not self._lock.acquire(blocking=False):
            log.info("whisper warm-up skipped: the mic path holds the model")
            return False
        import numpy as np       # only the warm-up needs it in this module

        audio = np.zeros(int(SAMPLE_RATE * self.WARMUP_SECONDS),
                         dtype=np.float32)
        budget = token_budget(self.WARMUP_SECONDS)
        t0 = time.monotonic()
        try:
            if self._gpu:
                model.transcribe(
                    audio, initial_prompt=self._prompt(),
                    language=self._language(), beam_size=1, fp16=True,
                    temperature=DECODE_TEMPERATURES, sample_len=budget)
            else:
                segments, _ = model.transcribe(
                    audio, beam_size=5, initial_prompt=self._prompt(),
                    temperature=DECODE_TEMPERATURES,
                    max_new_tokens=budget)
                list(segments)              # faster-whisper decodes lazily
        except Exception:
            log.exception("whisper warm-up decode failed")
            return False
        else:
            self._decoded = True     # the flag means "a kernel has run"
        finally:
            self._lock.release()
        log.info("whisper warm-up decode: %.2fs (%s)",
                 time.monotonic() - t0, self._backend or "?")
        return True

    # -- language -------------------------------------------------------
    def _language(self) -> str | None:
        """Whisper language code for the configured language
        (port: _get_whisper_language 2554-2557; None = auto-detect)."""
        return LANG_MAP.get(CONFIG.language, "en")

    # -- full transcription --------------------------------------------
    def transcribe(self, audio) -> TranscribeResult:
        """Full transcription: beam 1, VAD, vocab prompt, confidence gate.

        Beam 1, not 5: measured 2026-08-31 on 12 of Hunter's real clips --
        byte-identical text and WER at beam 1 vs 5 (1.18% verified, 6/6
        names), at HALF the latency (median 0.566 s -> 0.289 s). The
        streaming preview already ran beam 1; this aligns the final pass.

        Port of _transcribe_worker's whisper section (2581-2609),
        transcription only — speaker filtering and command routing live in
        the pipeline. Loads the model on first use.
        """
        if self._model is None:
            self.load()

        lang = self._language()

        seconds = _seconds(audio)
        budget = token_budget(seconds)
        t_start = time.monotonic()

        if self._gpu:
            # openai-whisper: expects float32 numpy @16k (exactly what the
            # recorder delivers — no resample). Segments carry avg_logprob
            # with the same semantics as faster-whisper's.
            with self._lock:
                lock_wait = time.monotonic() - t_start
                cpu0 = time.thread_time()
                result = self._model.transcribe(
                    audio,
                    initial_prompt=self._prompt(),
                    language=lang,          # None = auto-detect
                    beam_size=1,
                    fp16=True,
                    temperature=DECODE_TEMPERATURES,
                    sample_len=budget,
                    # With one rung, whisper's own loop-contagion guard is
                    # dead code: transcribe.py:503 resets the prompt only
                    # when condition_on_previous_text is off OR the rung
                    # that won was above temperature 0.5, and neither can
                    # happen now -- so a first window that loops would be
                    # fed verbatim as the prompt for the second. Reachable:
                    # recorder.MAX_RECORDING_SECONDS is 60, two windows.
                    # carry_initial_prompt keeps the vocab prompt on those
                    # later windows, which turning the conditioning off
                    # would otherwise throw away with it (the reset moves
                    # prompt_reset_since past the initial prompt too).
                    condition_on_previous_text=False,
                    carry_initial_prompt=True,
                )
                cpu = time.thread_time() - cpu0
            try:
                import torch
                torch.cuda.synchronize()   # idle the driver event thread
            except Exception:
                pass
            seg_list = result.get("segments") or []
            text = " ".join(s["text"].strip() for s in seg_list).strip()
            seg_data = [(s["text"], s["avg_logprob"]) for s in seg_list]
            ratio = max((s.get("compression_ratio", 0.0) for s in seg_list),
                        default=0.0)
            language = result.get("language")
        else:
            kwargs = dict(
                beam_size=5,
                initial_prompt=self._prompt(),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                temperature=DECODE_TEMPERATURES,
                max_new_tokens=budget,   # faster-whisper's spelling of the cap
                # Same dead guard as the GPU branch above (faster_whisper
                # transcribe.py:1372, prompt_reset_on_temperature=0.5).
                # faster-whisper has no carry_initial_prompt, so windows
                # past the first lose the vocab bias here -- the lesser evil
                # against a looping window seeding the next one, and this
                # branch is the no-CUDA fallback, not the shipping path.
                condition_on_previous_text=False,
            )
            if lang is not None:
                kwargs["language"] = lang

            # Lock prevents concurrent model access with partial() (2592)
            with self._lock:
                lock_wait = time.monotonic() - t_start
                cpu0 = time.thread_time()
                segments, info = self._model.transcribe(audio, **kwargs)
                seg_list = list(segments)
                cpu = time.thread_time() - cpu0
            text = " ".join(seg.text.strip() for seg in seg_list).strip()
            seg_data = [(seg.text, seg.avg_logprob) for seg in seg_list]
            ratio = max((getattr(seg, "compression_ratio", 0.0) or 0.0
                         for seg in seg_list), default=0.0)
            language = getattr(info, "language", None)

        self._decoded = True             # warmup() has nothing left to do
        _log_slow_decode(time.monotonic() - t_start, lock_wait, cpu, seconds)

        text = collapse_repeats(text)

        if seg_data:
            avg_conf = sum(lp for _, lp in seg_data) / len(seg_data)
            log.info("Transcribed: %r (avg_logprob=%.2f)", text, avg_conf)
            if avg_conf < MIN_AVG_LOGPROB:
                log.info("Rejected: confidence too low (%.2f < %s)",
                         avg_conf, MIN_AVG_LOGPROB)
        else:
            avg_conf = 0.0
            log.info("Transcribed: %r", text)
        limit = loop_ratio_limit(seconds)
        if ratio > limit:
            # Says both numbers, because this is the gate that replaced the
            # deep-negative avg_logprob the six-rung ladder used to produce,
            # and because the limit moves with the clip -- retuning it needs
            # the pair, not just the ratio.
            log.info("Rejected: repetition loop (compression_ratio=%.2f > "
                     "%.2f for %.1fs of audio)", ratio, limit, seconds)

        return TranscribeResult(
            text=text,
            confidence=avg_conf,
            segments=seg_data,
            language=language,
            compression_ratio=ratio,
            audio_seconds=seconds,
        )

    # -- streaming preview ---------------------------------------------
    def partial(self, audio) -> str:
        """Quick transcription for live preview (no VAD, beam 1).

        Port of _partial_transcribe_worker's whisper section (2710-2724);
        realtime command interception / filler detection / live typing stay
        in the pipeline. Returns "" if the model isn't loaded yet or the
        pass fails — a preview must never block or raise.
        """
        if self._model is None:
            return ""

        with self._lock:
            try:
                lang = self._language()

                budget = token_budget(_seconds(audio))

                if self._gpu:
                    # Greedy decode, no context carry-over, single
                    # temperature (no fallback retries) on BOTH backends,
                    # no timestamp tokens — fast text-only preview. The
                    # token cap is new:
                    # the preview runs several times a second while he is
                    # still talking, and an unbounded pass here holds the
                    # model lock the final decode is waiting on.
                    result = self._model.transcribe(
                        audio,
                        initial_prompt=self._prompt(),
                        language=lang,
                        condition_on_previous_text=False,
                        # ...which on its own also drops the vocab prompt
                        # from every window after the first (whisper moves
                        # prompt_reset_since past the initial prompt too);
                        # the preview runs on a rolling buffer that can
                        # cross 30 s during dictation.
                        carry_initial_prompt=True,
                        temperature=DECODE_TEMPERATURES,
                        without_timestamps=True,
                        fp16=True,
                        sample_len=budget,
                    )
                    self._decoded = True
                    return (result.get("text") or "").strip()

                # temperature here too: faster-whisper's own default is the
                # SIX-rung ladder ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), so
                # without this the preview -- which runs several times a
                # second, holding the lock the final decode waits on -- is
                # the one unbounded decode left in the module.
                kwargs = dict(beam_size=1, initial_prompt=self._prompt(),
                              max_new_tokens=budget,
                              temperature=DECODE_TEMPERATURES,
                              condition_on_previous_text=False)
                if lang is not None:
                    kwargs["language"] = lang

                segments, _ = self._model.transcribe(audio, **kwargs)
                text = " ".join(seg.text.strip() for seg in segments).strip()
                self._decoded = True
                return text
            except Exception:
                log.exception("partial transcription failed")
                return ""
