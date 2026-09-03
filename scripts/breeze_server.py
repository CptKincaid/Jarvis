"""Breeze-TTS-2 (Q4) sidecar: holds the quantized model resident, streams
speech over a unix socket.

WHY A SIDECAR AND NOT AN IMPORT
-------------------------------
Same reason as scripts/f5_server.py, twice over. Breeze needs its own venv
(~/voice-training/engines/breeze/venv) and its own source tree
(engines/breeze-q4/repo) on sys.path, neither of which can go into ~/vss_env
without disturbing VSS. And loading is expensive in a way that must be paid
exactly once, at a moment a person chose: measured on this box, 7.3 s to
load the pre-quantized checkpoint plus 22.0 s of CUDA-graph capture, and the
capture transiently demands ~18.3 GB above steady state -- it drove
system-wide MemFree to 2.39 GiB while 25.4 GiB of other tenants (ollama,
F5, Jarvis) were resident. That is the same shape as the 2026-08-28
unified-memory power-off, so it happens here, under a flock, behind a
MemFree gate, and never inside the Jarvis process.

WHY THE READINESS GATE IS LOAD-BEARING
--------------------------------------
int4 WITHOUT CUDA graphs measures RTF 1.131 (MEASUREMENTS.json) -- above
1.0, i.e. it falls behind real time and UNDERRUNS mid-utterance. With the
graphs it is 0.742. If ptxas goes missing (Triton's bundled 12.8 has no
sm_121a) or a CUDA upgrade breaks capture, the model still loads and still
speaks, just too slowly to stream: a working-but-stuttering assistant, which
is worse than a plain one. So the socket answers a ping with the graphs it
ACTUALLY captured, and Jarvis requires them; without them it uses F5.

For the same reason readiness includes a discarded render. Measured: the
first render after capture is 1.509 s to first audio at RTF 1.868 -- it
would underrun on its own. The second is 0.265 s at RTF 0.814. The first
one is paid here.

WHY EVERY RENDER RUNS ON ONE THREAD
----------------------------------
Measured 2026-09-02: a render costs a +16.37 GiB GPU transient the FIRST time
it runs on a given host thread, and 0.0-0.3 GiB every time after that on that
same thread. Connections are served thread-per-connection, so the render used
to pay that on every sentence. It does not any more: RenderWorker holds one
long-lived render thread, warmed at boot inside the GPU flock, and the
connection threads feed it and stream its chunks straight back out. The cost
is in prepare_inputs -- specifically the Mimi encode of the reference clip --
and not in generation; WHY that is per-thread is not known. See RenderWorker
for the arms, and for what was falsified.

PROTOCOL: newline-delimited JSON over a unix socket, one request per
connection (like f5_server.py).

    -> {"ping": true}
    <- {"ok": true, "ready": true, "graphs": true, "detail": {...}}

    compatible (speech cache, prewarm, the phone renderer):
    -> {"text": "Good evening, sir.", "out": "/tmp/x.wav", "gain": 2.8}
    <- {"ok": true, "seconds": 1.37, "wall": 1.02, "first_audio": 0.26}

    streaming (the room):
    -> {"text": "Good evening, sir.", "stream": true, "gain": 2.8}
    <- {"ok": true, "stream": true, "sr": 24000, ...}\n
       then a 44-byte RIFF/WAVE header with placeholder sizes,
       then PCM16 LE mono until the server shuts down its write side.

    failure BEFORE any audio (either mode):
    <- {"ok": false, "error": "..."}\n   and NOT ONE byte of audio,
       so the caller can render that chunk on F5 instead of speaking half
       a sentence twice.

    failure AFTER audio has started: the connection just ends. The caller
    cannot tell that from a clean finish -- everything after the header is
    raw PCM, so no in-band trailer is possible, and on AF_UNIX an abortive
    close is NOT distinguishable either (SO_LINGER 0 gives the peer a clean
    EOF; measured on this box before relying on it). So completion is
    reported out of band, on a second connection:

    -> {"status": "<the id sent with the stream request>"}
    <- {"ok": true, "complete": true, "bytes": 91234, "seconds": 1.9}
    <- {"ok": true, "complete": false, "error": "..."}        (truncated)
    <- {"ok": false, "error": "unknown request id"}           (forgotten)

    NO RECEIPT MEANS INCOMPLETE. A sidecar that died mid-chunk answers
    nothing at all, and the caller must treat that as truncated -- so the
    safe reading is the default one, and caching a clipped sentence takes
    a positive assertion that it is whole.

Run it from the systemd unit (scripts/systemd/jarvis-breeze.service),
installed by scripts/setup_breeze_service.sh. It needs
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas and
BREEZE_TEXT_ENCODER_ATTN=sdpa in the environment; the unit sets both.
"""
import argparse
import collections
import contextlib
import fcntl
import itertools
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
import traceback
import wave
import weakref
from pathlib import Path

import numpy as np

# ------------------------------------------------------------ the pinned rig
#
# This is the configuration Hunter rated 4.71 in the round-11 blind listening
# test, against a hidden hosted-Fish control at 4.64 and the shipped F5 voice
# at 2.79; a 6-clip blind A/B of the QUANTIZED build against the bf16 one he
# had scored 4.73 came out at mean -0.33, inside the round's 1.00 noise floor.
# Every value below is copied from that render's own manifest
# (engines/breeze-q4/out/Q4_rerender/RENDER_INFO.json). Changing any of them
# is a new voice and needs a new blind round -- and each one is in Jarvis's
# speech-cache key (jarvis/tts.py BREEZE_PARAMS) so a change can never replay
# stale audio.
#
# cfg_scale 4.0 is NOT a speed knob. With the graphs on, dropping it buys
# 0.4 %, and it is what makes the voice-direction instruction fire at all.
REPO_DEFAULT = Path("/home/hunterp/voice-training/engines/breeze-q4/repo")
CKPT_DEFAULT = Path("/home/hunterp/voice-training/engines/breeze-q4/ckpt-q4")
GPU_LOCK_DEFAULT = Path("/home/hunterp/voice-training/.gpu.lock")
INSTRUCTION = ("A calm, articulate British assistant speaking conversationally, "
               "with natural variation from sentence to sentence; questions "
               "rise at the end.")
TEMPLATE = "ref_edit_tata"
CFG_SCALE = 4.0
SEED = 1234
# The checkpoint's own generation_config.json value, recorded here and in
# jarvis/tts.py BREEZE_PARAMS so it is visible in the cache key; the server
# does not set it, exactly as the rated render did not.
TEMPERATURE = 0.9
REPETITION_PENALTY = 1.1
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
SAMPLE_RATE = 24000
# Nominal: the pre-quantized checkpoint carries its own attention
# implementation, and this is what the rated render's manifest recorded.
ATTN = "eager"
FAST_CONFIG = "configs/fast.json"

# TWO gates, because the two numbers answer different questions and the
# start is the only dangerous moment this sidecar has.
#
# MemFree is the one that nearly ran out. Measured across a real start on
# this box: MemAvailable never dropped below 41.7 GiB while actual free
# pages hit 2.39 GiB, because ~39 GiB of MemAvailable was reclaimable page
# cache -- and NVRM satisfying pinned allocations by forcing reclaim under
# serverTopLock is exactly the 2026-08-28 deadlock. render_q.py's
# --min-free-gb reads MemAvailable and would have waved that through.
#
# The arithmetic, from that same trace: the resident model costs 13.5 GiB of
# free pages (7.1 GiB attributed to the PID by nvidia-smi plus 6.2 GiB of
# RSS -- on GB10 those are additive, confirmed against the external MemFree
# delta), and graph capture then demands 18.2 GiB MORE, in bursts, on top of
# it. So the trough is roughly (MemFree at start) - 31.7 GiB, and 33 is not a
# comfortable floor, it is the level a real start was MEASURED succeeding
# from: 33.88 GiB in, trough 2.39 GiB. (A companion analysis proposed 25;
# that counts the 18.2 GiB transient but not the 13.5 GiB the model has
# already taken by the time capture runs, and would land at -6.)
#
# MEASURED AGAIN 2026-09-02 16:22-16:23 (13 samples, 5 s apart, with ollama,
# the F5 sidecar, Jarvis and the desktop up): MemFree 19.2-29.2 GiB,
# MemAvailable 63.5-73.5 GiB. An earlier note here claimed idle MemFree was
# 33.5-34.7 GiB; it is not, and the honest consequence is that THIS GATE
# REFUSES ON THE BOX AS IT STANDS TODAY. That is the intended behaviour --
# Jarvis keeps speaking through F5 -- and the number is not the thing to file
# down: 33 is the level a real start was measured surviving from (33.88 GiB
# in, trough 2.39 GiB), and the trough is what the 2026-08-28 power-off was.
# The lever is the 18.6 GiB ollama holds pinned at keep_alive -1
# (`ollama stop <model>`), not this floor.
#
# A refusal is ``return 2``, and the unit sets RestartPreventExitStatus=2 so
# that a deliberate refusal is terminal and legible instead of burning the
# three restarts StartLimitBurst allows and replacing this message with
# "start request repeated too quickly" -- which is what happened to the
# obvious recovery sequence (start, read the message, free memory, start
# again). See scripts/systemd/jarvis-breeze.service.
#
# MemAvailable is the second gate for two reasons: it is the box's standing
# rule for any GPU job here, and it is what says there is page cache left to
# reclaim -- which is the only reason the 2.39 GiB trough above was
# survivable.
MIN_FREE_GB = 33
MIN_AVAILABLE_GB = 60

# "length not known yet" in a RIFF size field -- what ffmpeg writes when its
# output is a pipe. Kept identical to jarvis/tts.py's STREAM_SIZE.
STREAM_SIZE = 0xFFFFFFFF

# TWO deadlines, because the two phases fail differently and one number for
# both was a quarter of an hour of nothing.
#
# READ is short. Every client connects and writes its request line in the same
# breath (jarvis/tts.py _breeze_iter and _breeze_request both connect+sendall),
# so a peer that has connected and written nothing is dead or hostile, not
# slow. Reproduced with the old single 900 s deadline: a client that sent
# b'{"pi' and then held blocked read_request, and -- on the old single-threaded
# accept loop -- every other request queued behind it.
#
# SEND is scripts/f5_server.py's number, and it bounds ONE blocked sendall,
# not a whole generation: the gaps between blocks are the GPU, not the socket,
# and never count against it. 64 KB to a client that is reading is instant, so
# 120 s without progress means the peer stopped reading.
READ_TIMEOUT_S = 15.0
SEND_TIMEOUT_S = 120.0

# Connections are served on their own threads (see serve()) so that a ping or
# a completion receipt is never queued behind a 14 s render. This bounds how
# many may pile up: Jarvis opens one at a time and the phone renderer one
# more, so anything near this is a leak or an attack, and refusing beats
# growing threads without bound.
MAX_CONNECTIONS = 8

# How many stream outcomes to remember for the receipt query above. The
# caller asks for one the moment its stream ends, and only one stream runs
# at a time, so this only has to survive interleaving between the room and
# the phone renderer.
RECEIPTS = 16


# ------------------------------------------------------------------- helpers
def wav_header(rate: int, channels: int = 1, sampwidth: int = 2,
               data_bytes=None) -> bytes:
    """A canonical 44-byte RIFF/WAVE PCM header.

    Byte-for-byte the same builder as jarvis/tts.py's ``wav_header`` --
    tests/test_breeze_sidecar.py asserts the two agree, because the client's
    _LiveEnvelope parses this header and paplay decodes it from stdin, and a
    disagreement here is silence with no error anywhere.
    """
    data = STREAM_SIZE if data_bytes is None else int(data_bytes)
    riff = STREAM_SIZE if data_bytes is None else 36 + int(data_bytes)
    block = int(channels) * int(sampwidth)
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, int(channels), int(rate),
                        int(rate) * block, block, int(sampwidth) * 8) +
            b"data" + struct.pack("<I", data))


def pcm16(audio, gain: float = 1.0) -> bytes:
    """One codec chunk of float32 [-1, 1] as little-endian PCM16.

    The OUTPUT GAIN is applied here, in the one place both socket modes pass
    through, so a streamed chunk and the cached file of the same line come
    out at the same level. Jarvis owns the value (tts.output_gain_for) and
    sends it with the request; it is in the cache key with it.

    Clipped, not peak-normalised: normalising per chunk would make the level
    pump audibly inside one sentence. At the measured Breeze peaks
    (0.210-0.337 over the 14 rated renders) a gain of 2.8 reaches 0.94 and
    never touches the rail.
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if gain != 1.0:
        a = a * float(gain)
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


def mem_gb(field: str = "MemFree", meminfo: str = "/proc/meminfo") -> float:
    """One /proc/meminfo field in GiB, or -1.0 when it cannot be read."""
    try:
        with open(meminfo) as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return -1.0


def free_mem_gb(meminfo: str = "/proc/meminfo") -> float:
    """FREE pages in GiB. Deliberately not MemAvailable -- see MIN_FREE_GB."""
    return mem_gb("MemFree", meminfo)


def graphs_captured(runtime) -> dict:
    """Which CUDA graphs the runtime ACTUALLY holds, not which were asked for.

    ``fast_enabled`` is only the request (it is ``any(...)`` over the config
    flags), and _ensure_graphs quietly falls back to ``prepare_eager()`` for
    any lane whose flag is off -- so the config cannot answer this. The
    objects themselves can: BackboneGraph/DepthDecoderGraph set ``captured``
    at the end of ``capture()``, and each codec lane gets a ``cuda_graph``.
    Read through getattr so a library change degrades to ready=false rather
    than to an AttributeError at ping time.
    """
    backbones = getattr(runtime, "_backbone_graphs", None) or {}
    depth = getattr(runtime, "_depth_decoder_graph", None)
    codec = getattr(runtime, "_codec_runtime", None)
    lanes = tuple(getattr(codec, "lanes", ()) or ()) if codec is not None else ()
    return {
        "fast_enabled": bool(getattr(runtime, "fast_enabled", False)),
        "backbone_decode": bool(backbones) and all(
            bool(getattr(g, "captured", False)) for g in backbones.values()),
        "depth_decoder": bool(getattr(depth, "captured", False)),
        "codec": bool(lanes) and all(
            getattr(lane, "cuda_graph", None) is not None for lane in lanes),
    }


def all_graphs_captured(detail: dict) -> bool:
    """True only when every lane above is genuinely captured."""
    return bool(detail) and all(bool(v) for v in detail.values())


@contextlib.contextmanager
def gpu_lock(path):
    """Hold ~/voice-training/.gpu.lock for the whole load + capture + warm.

    NOT just the load: the entire 18.3 GB transient lives inside the capture
    window, so a lock released after load_model would expose exactly the
    burst it exists to serialise. ``path`` of None is for tests only.
    """
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # SAY SO. This is entered before the only other output in main(),
            # and the unit is Type=simple -- so systemd reports it `active`
            # and Jarvis sees an active unit with no socket, while journalctl
            # shows nothing at all between ExecStartPre and the memory gate
            # for as long as another GPU job holds the lock. Silent-and-queued
            # and hung look identical without this line.
            print(f"breeze: waiting for the GPU lock ({path}) -- another GPU "
                  f"job holds it", flush=True)
            waited = time.perf_counter()
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            print(f"breeze: got the GPU lock after "
                  f"{time.perf_counter() - waited:.0f}s", flush=True)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


# ---------------------------------------------------------- one render thread
# THE +16.37 GiB TRANSIENT, AND WHY THIS CLASS EXISTS.
#
# Measured 2026-09-02 with an external MemFree sampler -- a 1 Hz one misses it
# entirely; the dip is ~620 ms wide and starts ~540 ms into the render, to
# within 5 ms every time. EVERY render served through this socket cost a
# +16.37 GiB GPU transient, fully returned each time (no leak), and INDEPENDENT
# OF THE TEXT: a 20-character line cost the same as a 293-character one.
#
# Two plausible causes were tested and FALSIFIED. max_new_tokens does not size
# it (37 renders, caps 1500 down to 50: 16.373 +/- 0.177 GiB, R^2 0.022 -- so
# the cap stays at 1500, where it belongs; below ~260 frames it guillotines
# real replies, and 5 of the 14 rated lines run past 8 s). Nor is it
# torch.compile/Inductor autotuning: the cache was already warm and 25 renders
# wrote no new files.
#
# What it IS, measured: PER HOST THREAD, paid once on that thread's first
# render. Controlled arms, one resident model, cfg_scale 4.0 throughout --
#
#     a new thread for each render     16.42 / 16.37 / 16.29 / 16.39 GiB
#     ONE persistent worker thread     16.24 GiB on its FIRST render, then
#                                      0.04 / 0.08 / 0.02 / 0.00 / 0.04 / 0.05
#     one new thread, three renders    16.39, then 0.29, 1.47
#     the main thread, 28 renders      0.01 - 0.30 GiB (nvidia-smi: 0.00)
#
# serve() starts a thread per CONNECTION and the render used to run on it, so
# Jarvis paid that 16 GiB on every single sentence -- on a box where the GPU
# memory IS the system memory and where an exhausted pool is what forced the
# 2026-08-28 hard power-off.
#
# WHERE the cost is, measured one level down (probe3/probe4 in
# ~/voice-training/breeze-cap): IT IS NOT THE GENERATION. On a fresh thread,
# prepare_inputs dipped 16.47 / 16.34 / 16.24 GiB while the generation that
# followed it on that same thread dipped 0.01 / 0.07 / 0.11. Inside
# prepare_inputs it is the Mimi audio tokenizer encoding the REFERENCE CLIP
# (encode_prompt_audio): on a brand-new thread that call alone dips 16.47 GiB
# over 620 ms, the HF text tokenisation beside it dips 0.0, and torch's own
# max_memory_allocated goes 6.33 -> 22.65 GB across it -- so the 16 GiB is a
# torch allocation, not something hidden in the driver. Reserved comes back to
# 6.66-6.78 GB after: a warm thread does not HOLD the 16 GiB, it simply never
# asks for it again. That is why one long-lived thread is a fix and not a
# trade.
#
# WHY that one call is per-thread is still NOT KNOWN, and this comment will
# not invent a reason. A generic per-thread CUDA first-touch was ruled out --
# a fresh thread's first torch.zeros, bf16 matmul, sdpa, conv1d and conv2d
# each cost 0.00-0.01 GiB (microthread.py) -- so it is something specific to
# that encoder's first call on a host thread. The consequence for THIS file is
# the same either way, which is why tests/test_breeze_sidecar.py pins the
# THREAD IDENTITY and no story about it.
#
# It does name a better fix for whoever picks this up: the reference clip
# never changes and its encode is deterministic, so memoising
# encode_prompt_audio would remove the 16 GiB outright, boot included. That
# lives in the breeze source tree, not in this file.
#
# Only the RENDER moves. Connections keep their own threads -- a ping or a
# completion receipt must never queue behind a 14 s render, which is the
# measured reason serve() went thread-per-connection in the first place --
# and BreezeService._render_lock still holds generation to one at a time.
# Streaming is preserved chunk by chunk (see __call__): the room's 0.27 s
# time-to-first-audio, which beats the shipped F5 path at 0.53 s, is the whole
# reason this voice is worth the memory, and a worker that returned only
# finished audio would throw it away.

# How many PCM blocks the worker may run ahead of the connection thread that
# is writing them to the socket. Bounded on purpose. Unbounded, a client that
# stopped reading would let the GPU generate a whole utterance into RAM on a
# box with no memory to spare; at 1 it would reimpose exactly the lock-step
# the inline generator had, making the socket's write latency the model's
# problem. Four is small: fast_codec yields one codec frame per block and the
# checkpoint's _frame_rate is 12.5 Hz, so the buffer is ~0.32 s of audio.
RENDER_QUEUE_DEPTH = 4

# A NET, NOT A DEADLINE. The worker reports its own death (_fail_pending), so
# this only catches a worker wedged INSIDE the model, where a connection thread
# would otherwise wait for ever and Jarvis would never fall back to F5. It sits
# far above any real render: the longest rated line is ~14 s of wall clock.
RENDER_STALL_S = 300.0

# The stop pill. A sentinel object, not None, because None is a legal nothing.
_STOP = object()
_DONE = object()

WORKER_GONE = "the breeze render worker is not running"


class _Failed:
    """A render that raised, on its way back to the connection thread."""

    __slots__ = ("exc",)

    def __init__(self, exc):
        self.exc = exc


class _RenderJob:
    """One render in flight between a connection thread and the worker.

    ``chunks`` carries PCM blocks and then exactly one terminal item (_DONE or
    a _Failed). ``cancelled`` is how a connection thread that has gone away --
    barge-in, a dead client, a send timeout -- tells the worker to stop
    generating for nobody.
    """

    __slots__ = ("text", "gain", "chunks", "cancelled")

    def __init__(self, text: str, gain: float, depth: int):
        self.text = text
        self.gain = float(gain)
        self.chunks = queue.Queue(maxsize=depth)
        self.cancelled = threading.Event()


class RenderWorker:
    """Runs every render on ONE long-lived thread, streaming as it goes.

    It is itself a ``render(text, gain) -> iterator of PCM blocks`` callable,
    which is the seam BreezeService and load_engine already used, so nothing
    downstream had to learn what a queue is.
    """

    def __init__(self, render, *, name: str = "breeze-render",
                 depth: int = RENDER_QUEUE_DEPTH,
                 stall_s: float = RENDER_STALL_S):
        self._render = render
        self._name = name
        self._depth = max(1, int(depth))
        self._stall_s = float(stall_s)
        self._inbox: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._thread = None
        self._dead = threading.Event()
        self._current = None

    # ------------------------------------------------------------ lifecycle
    @property
    def alive(self) -> bool:
        """False once the thread has exited or could never be started.

        Read on every ping and before every dispatch: a dead worker must make
        the sidecar not-ready, not make one connection hang.
        """
        return not self._dead.is_set()

    @property
    def ident(self):
        """The thread the model actually runs on, for the log and the tests."""
        return self._thread.ident if self._thread is not None else None

    def start(self):
        """Idempotent. Raises if the worker has already died."""
        with self._lock:
            if self._dead.is_set():
                raise RuntimeError(WORKER_GONE)
            self._start_locked()
        return self

    def _start_locked(self) -> None:
        if self._thread is not None:
            return
        thread = threading.Thread(target=self._loop, name=self._name,
                                  daemon=True)
        try:
            thread.start()
        except RuntimeError:            # out of threads: fail closed, loudly
            self._dead.set()
            raise
        self._thread = thread

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the worker to finish and go. Safe to call twice, or never
        having started."""
        with self._lock:
            thread = self._thread
            if thread is None:
                self._dead.set()
        job = self._current
        if job is not None:
            job.cancelled.set()         # so a long render notices the pill
        self._inbox.put(_STOP)
        if thread is None:
            self._fail_pending(RuntimeError(WORKER_GONE))
            return
        thread.join(timeout)

    # ---------------------------------------------------------- the callable
    def __call__(self, text: str, gain: float = 1.0):
        """``render(text, gain)``: a generator of PCM blocks.

        The blocks are handed over AS THEY ARE PRODUCED -- this is a pipe, not
        a promise. _serve_stream writes each one to the socket the moment it
        arrives, so time-to-first-audio is still the model's first chunk and
        nothing waits for the utterance to finish.
        """
        job = _RenderJob(text, gain, self._depth)
        self._submit(job)
        try:
            while True:
                try:
                    item = job.chunks.get(timeout=self._stall_s)
                except queue.Empty:
                    raise RuntimeError(
                        f"the breeze render worker produced nothing for "
                        f"{self._stall_s:.0f}s") from None
                if item is _DONE:
                    return
                if isinstance(item, _Failed):
                    if not isinstance(item.exc, Exception):
                        # A BaseException -- SystemExit, KeyboardInterrupt --
                        # would blow straight through _serve_stream's
                        # `except Exception`, past handle()'s net and past
                        # _serve_conn's, and the caller would get EOF with no
                        # refusal line: the one outcome this protocol promises
                        # never to produce before audio, and the one that
                        # makes Jarvis speak half a sentence twice. It killed
                        # the worker; it does not get to kill the answer too.
                        raise RuntimeError(
                            f"the breeze render worker died: "
                            f"{type(item.exc).__name__}: {item.exc}")
                    raise item.exc
                yield item
        finally:
            # Whether this ended, raised, or was closed by _close_generator
            # because the client hung up: the worker must stop generating for
            # a consumer that is no longer reading.
            job.cancelled.set()

    def _submit(self, job) -> None:
        # Under the lock, and _loop sets _dead BEFORE it drains, so a job can
        # never be filed with a worker that has just gone and then wait out
        # RENDER_STALL_S for an answer nobody will send.
        with self._lock:
            if self._dead.is_set():
                raise RuntimeError(WORKER_GONE)
            self._start_locked()
            self._inbox.put(job)

    # ------------------------------------------------------------- the loop
    def _loop(self) -> None:
        try:
            while True:
                job = self._inbox.get()
                if job is _STOP:
                    break
                self._current = job
                try:
                    self._run(job)
                finally:
                    self._current = None
        except BaseException:            # it is about to die anyway
            traceback.print_exc()
            print("breeze: the render worker died -- the sidecar is now "
                  "not-ready and Jarvis will speak in F5", file=sys.stderr,
                  flush=True)
        finally:
            # Order matters: _dead first, so _submit under the same lock can
            # never slip a job in behind the drain below.
            self._dead.set()
            self._fail_pending(RuntimeError(WORKER_GONE))

    def _run(self, job) -> None:
        blocks = None
        try:
            blocks = self._render(job.text, job.gain)
            for block in blocks:
                if job.cancelled.is_set():
                    break
                if not block:
                    continue
                if not self._put(job, block):
                    break               # cancelled while waiting for room
            self._put(job, _DONE)
        except Exception as exc:
            # NOT fatal to the worker: one line that blew up must not cost the
            # resident model. The connection thread re-raises this, and if no
            # audio has been sent yet it answers {"ok": false} so the caller
            # renders that chunk on F5.
            self._put(job, _Failed(exc))
        except BaseException as exc:     # hand it back, then die
            self._put(job, _Failed(exc))
            raise
        finally:
            # On the WORKER thread, deliberately. Closing a half-consumed
            # render generator runs its cleanup wherever close() is called
            # from, and that cleanup touches CUDA -- so a barge-in used to
            # tear down GPU state on the connection thread. Now every line of
            # this that reaches the model runs on the one thread.
            _close_generator(blocks)

    def _put(self, job, item) -> bool:
        """Hand one item over, giving up if the consumer has gone. Polls
        rather than blocking so a client that hung up cannot pin the worker
        to a full queue for ever."""
        while not job.cancelled.is_set():
            try:
                job.chunks.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _fail_pending(self, exc) -> None:
        """Every job that will now never run -- the one in flight and every
        one still queued -- gets the error, so no connection thread is left
        waiting on a worker that has gone."""
        stranded = []
        job = self._current
        if job is not None:
            stranded.append(job)
        with self._lock:
            while True:
                try:
                    item = self._inbox.get_nowait()
                except queue.Empty:
                    break
                if item is not _STOP:
                    stranded.append(item)
        for job in stranded:
            self._put(job, _Failed(exc))


# ------------------------------------------------------------- the protocol
class BreezeService:
    """The socket protocol, with the engine behind one callable.

    ``render(text, gain) -> iterator of PCM16 byte blocks`` is the only thing
    this needs from Breeze, which is what lets the whole protocol -- including
    every failure path -- be tested with a fake and no GPU.
    """

    def __init__(self, render=None, sample_rate: int = SAMPLE_RATE,
                 ready: bool = False, graphs: dict | None = None,
                 error: str | None = None, config: dict | None = None):
        # EVERY render goes through the one worker thread -- see RenderWorker
        # for the 16.37 GiB per-thread measurement that is the whole reason.
        # The wrapping happens HERE, not at the call sites, because the defect
        # being fixed was a render running on whichever thread happened to be
        # holding the connection: a service that cannot be constructed without
        # a worker cannot regress to that by accident. load_engine passes a
        # RenderWorker it has already warmed (so the one-off transient is paid
        # at boot, inside the GPU flock) and that one is used as it is.
        if render is None or isinstance(render, RenderWorker):
            self.worker = render
        else:
            self.worker = RenderWorker(render)
        self.render = self.worker
        self.sample_rate = int(sample_rate)
        self.graphs = dict(graphs or {})
        # id -> {"complete": bool, ...} for the receipt query. See the
        # module docstring: EOF alone cannot say whether a stream finished.
        self._receipts: "collections.OrderedDict[str, dict]" = (
            collections.OrderedDict())
        # Connections are served on their own threads now, so both of these
        # are reached concurrently. _render_lock keeps the one resident model
        # to ONE generation at a time -- which is what the old accept-then-
        # handle loop gave for free, and the only part of that serialisation
        # worth keeping. The single render thread would serialise generation
        # on its own, so this lock is now belt AND braces, and it earns its
        # keep: it holds the worker's inbox to one job, so seven queued
        # connections cannot each buy a chunk buffer, and it keeps "one render
        # at a time" a property of the protocol rather than of the worker's
        # internals. _receipt_lock guards the OrderedDict a receipt query
        # reads while a stream is writing it.
        self._render_lock = threading.Lock()
        self._receipt_lock = threading.Lock()
        # Reported on every ping so Jarvis can WARN when the running sidecar
        # does not match the parameters its speech cache is keyed on: the
        # sidecar applies the voice, tts.BREEZE_PARAMS only records it, and a
        # unit hand-started with a different --cfg-scale would otherwise file
        # one voice under another's key in silence.
        self.config = dict(config or {})
        # ready is never a synonym for "loaded": see the module docstring.
        # It is also not a synonym for "was ready at boot" -- see the property
        # below, which folds in whether the render thread is still there.
        self._ready = bool(ready and self.worker is not None
                           and all_graphs_captured(self.graphs))
        self.error = error
        # A service that is dropped -- a test, or any future embedder -- must
        # not leave its worker parked on the inbox for the life of the
        # process. main()'s service lives until exit and calls close() itself.
        self._finalizer = (weakref.finalize(self, self.worker.stop)
                           if self.worker is not None else None)

    @property
    def ready(self) -> bool:
        """Ready AND the render thread is still alive. FAIL CLOSED.

        A dead worker means every render would raise, so the next ping has to
        say not-ready and Jarvis has to speak in the 2.79 voice. The
        alternative -- a sidecar that still claims 4.71 and then errors, or
        worse, hangs the connection -- is a mute assistant.
        """
        return self._ready and self.worker is not None and self.worker.alive

    def close(self) -> None:
        """Stop the render thread. Idempotent."""
        if self.worker is not None:
            self.worker.stop()

    def _why_not(self) -> str | None:
        """The error to report, including one that happened after boot."""
        if self.error:
            return self.error
        if self._ready and self.worker is not None and not self.worker.alive:
            return WORKER_GONE
        return None

    # ------------------------------------------------------------- replies
    def ping(self) -> dict:
        reply = {"ok": True, "ready": self.ready,
                 "graphs": all_graphs_captured(self.graphs),
                 "detail": self.graphs, "sr": self.sample_rate,
                 "config": self.config}
        why = self._why_not()
        if why:
            reply["error"] = why
        return reply

    def receipt(self, request_id: str) -> dict:
        """Did that streamed chunk finish? See the module docstring."""
        with self._receipt_lock:
            got = self._receipts.get(request_id)
        if got is None:
            return {"ok": False, "id": request_id,
                    "error": "unknown request id"}
        return {"ok": True, "id": request_id, **got}

    def _remember(self, request_id, outcome: dict) -> None:
        if not request_id:
            return
        with self._receipt_lock:
            self._receipts[str(request_id)] = outcome
            while len(self._receipts) > RECEIPTS:
                self._receipts.popitem(last=False)

    def _not_ready(self) -> dict:
        return {"ok": False, "error": self._why_not() or
                "breeze sidecar is not ready (CUDA graphs were not captured)"}

    # ------------------------------------------------------------ dispatch
    def handle(self, conn) -> None:
        """Serve exactly one request on ``conn``. Never raises -- and the code
        now says so, not just this line.

        The previous version put only read_request inside a try. Everything
        after it ran bare, so four bytes killed a 13.5 GB resident model with
        no reply to the client. A body of 5, of [], of "x" or of true all
        reached req.get("ping") on a non-dict; {"text": 5} reached .strip() on
        an int; and {"gain": "loud"}, {"gain": null} and {"gain": [1]} all
        reached float(). All eight were reproduced against the real serve()
        loop over a real AF_UNIX socket: no reply, process dead, every later
        connect ECONNREFUSED. With Restart=always/RestartSec=30 each kill
        re-entered the 22 s CUDA-graph capture and its ~18.3 GB transient, and
        three of them left the unit failed until someone ran
        `systemctl --user reset-failed`.

        scripts/f5_server.py -- the file this is modelled on -- opens its try
        BEFORE the first req.get() and answers all of these with
        {"ok": false}. This is that, plus a net in serve() underneath.
        """
        try:
            conn.settimeout(READ_TIMEOUT_S)
        except (AttributeError, OSError):
            pass
        try:
            req = read_request(conn)
        except Exception as exc:                    # never kill the server
            traceback.print_exc()
            _send_json(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        if req is None:
            return
        try:
            self._dispatch(conn, req)
        except Exception as exc:
            # _serve_stream and _serve_file swallow their own failures (the
            # streaming one MUST: half a sentence is already in the room and
            # no JSON line can follow raw PCM), so anything arriving here came
            # from the type checks below, before a byte of audio -- where a
            # refusal line is still exactly the right answer.
            traceback.print_exc()
            _send_json(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _dispatch(self, conn, req) -> None:
        """Route one parsed request. Every field is type-checked before it is
        used, because json.loads returns whatever the peer sent."""
        if not isinstance(req, dict):
            _send_json(conn, {"ok": False,
                              "error": "request must be a JSON object"})
            return
        if req.get("ping"):
            _send_json(conn, self.ping())
            return
        if req.get("status"):
            _send_json(conn, self.receipt(str(req["status"])))
            return
        if not self.ready:
            _send_json(conn, self._not_ready())
            return
        text = req.get("text")
        if not isinstance(text, str) or not text.strip():
            _send_json(conn, {"ok": False, "error": "no text"})
            return
        gain = req.get("gain", 1.0)
        # bool first: it is an int subclass, and {"gain": true} is a mistake,
        # not a request to render at unity.
        if isinstance(gain, bool) or not isinstance(gain, (int, float)):
            _send_json(conn, {"ok": False, "error": "gain must be a number"})
            return
        # Past here the connection generates and sends for as long as the
        # model takes, so the short read deadline is replaced by the send one.
        try:
            conn.settimeout(SEND_TIMEOUT_S)
        except (AttributeError, OSError):
            pass
        # ONE generation at a time on the one resident model, and -- since
        # self.render IS the RenderWorker -- always on the same thread as the
        # last one. A ping or a receipt query never reaches this, which is the
        # point: measured before the split, a 6 s in-flight render made
        # Jarvis's _ensure_breeze_server cost 6.0 s on the speak path because
        # its ping was stuck behind the render in the accept queue.
        with self._render_lock:
            if req.get("stream"):
                self._serve_stream(conn, text.strip(), float(gain),
                                   req.get("id"))
            else:
                self._serve_file(conn, text.strip(), req.get("out"),
                                 float(gain))

    # -------------------------------------------------------------- modes
    def _serve_stream(self, conn, text: str, gain: float,
                      request_id=None) -> None:
        started = time.perf_counter()
        first = None
        nbytes = 0
        blocks = None
        try:
            blocks = self.render(text, gain)
            for block in blocks:
                if not block:
                    continue
                if first is None:
                    first = time.perf_counter() - started
                    _send_json(conn, {"ok": True, "stream": True,
                                      "sr": self.sample_rate, "channels": 1,
                                      "sampwidth": 2,
                                      "first_audio": round(first, 3)})
                    conn.sendall(wav_header(self.sample_rate))
                conn.sendall(block)
                nbytes += len(block)
        except (BrokenPipeError, ConnectionResetError) as exc:
            # The client hung up -- barge-in, or Jarvis exiting. Not an
            # error, and there is nobody left to tell, but the chunk did not
            # finish and the receipt must say so.
            print(f"breeze: client went away mid-stream ({exc})", flush=True)
            _close_generator(blocks)
            self._remember(request_id, {"complete": False, "bytes": nbytes,
                                        "error": f"{type(exc).__name__}: {exc}"})
            return
        except Exception as exc:
            traceback.print_exc()
            _close_generator(blocks)
            self._remember(request_id, {"complete": False, "bytes": nbytes,
                                        "error": f"{type(exc).__name__}: {exc}"})
            if first is None:
                # Nothing was heard: the caller renders this chunk on F5.
                _send_json(conn, {"ok": False,
                                  "error": f"{type(exc).__name__}: {exc}"})
            # Half a sentence is already in the room: nothing more can be
            # said on this connection, and the receipt above is how the
            # caller learns not to cache it.
            return
        _close_generator(blocks)
        if first is None:                            # generated nothing at all
            self._remember(request_id, {"complete": False, "bytes": 0,
                                        "error": "breeze produced no audio"})
            _send_json(conn, {"ok": False, "error": "breeze produced no audio"})
            return
        wall = time.perf_counter() - started
        secs = nbytes / 2 / self.sample_rate
        self._remember(request_id, {"complete": True, "bytes": nbytes,
                                    "seconds": secs, "wall": wall,
                                    "first_audio": first})
        print(f"breeze: streamed {secs:.2f}s in {wall:.2f}s "
              f"(RTF {wall / secs if secs else 0:.3f}, first {first:.3f}s) "
              f"{text[:60]!r}", flush=True)
        with contextlib.suppress(OSError):
            conn.shutdown(socket.SHUT_WR)            # end of audio

    def _serve_file(self, conn, text: str, out, gain: float) -> None:
        # isinstance, not just truthiness: wave.open(str(out)) turned
        # {"out": 5} into {"ok": true} plus a file literally named `5` in the
        # server's cwd -- which load_engine has chdir'd to the Breeze source
        # tree. A wrong-typed path is a refusal, not a stray file.
        if not isinstance(out, str) or not out:
            _send_json(conn, {"ok": False, "error": "no out path"})
            return
        started = time.perf_counter()
        first = None
        parts = []
        blocks = None
        try:
            blocks = self.render(text, gain)
            for block in blocks:
                if not block:
                    continue
                if first is None:
                    first = time.perf_counter() - started
                parts.append(block)
            _close_generator(blocks)
            pcm = b"".join(parts)
            if not pcm:
                raise RuntimeError("breeze produced no audio")
            with wave.open(out, "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(self.sample_rate)
                fh.writeframes(pcm)
        except Exception as exc:
            traceback.print_exc()
            _close_generator(blocks)
            _send_json(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        wall = time.perf_counter() - started
        secs = len(pcm) / 2 / self.sample_rate
        _send_json(conn, {"ok": True, "seconds": secs, "wall": wall,
                          "first_audio": first,
                          "rtf": (wall / secs) if secs else None})


def read_request(conn):
    """One newline-terminated JSON object, or None when the peer said nothing."""
    buf = b""
    while not buf.endswith(b"\n"):
        part = conn.recv(65536)
        if not part:
            break
        buf += part
        if len(buf) > 1_000_000:
            raise ValueError("request too large")
    if not buf.strip():
        return None
    return json.loads(buf.decode("utf-8"))


def _send_json(conn, payload: dict) -> None:
    with contextlib.suppress(OSError):
        conn.sendall(json.dumps(payload).encode() + b"\n")


def _close_generator(blocks) -> None:
    """Stop a half-consumed render so the GPU is not still generating for a
    client that has gone."""
    close = getattr(blocks, "close", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        close()


# ------------------------------------------------------------- the engine
def make_renderer(runtime, tokenizer, audio_tokenizer, model, *, template: str,
                  ref_audio, ref_text: str, instruction: str,
                  cfg_scale: float, seed: int):
    """``render(text, gain)`` over the resident runtime.

    iter_audio_chunks yields HEADERLESS float32 numpy arrays, which is the
    one seam that will silently break the room: both the client's envelope
    parser and paplay's stdin decoder need a canonical PCM16 RIFF. The
    header is written by _serve_stream and the samples are converted here,
    so nothing downstream ever sees a float.
    """
    from breeze_infer.runtime import set_all_seeds
    from breeze_infer.templates import get_template, prepare_inputs

    tmpl = get_template(template)
    counter = itertools.count(1)

    def render(text: str, gain: float = 1.0):
        request = {"id": f"jarvis-{next(counter)}", "text": text,
                   "instruction": instruction, "speaker": "S0",
                   "ref_audio_path": str(ref_audio), "ref_text": ref_text}
        # Same seed every request, as the rated arm did: sampling decides
        # WHICH take you get, not how expressive it is, and a fixed seed is
        # what makes a cached line and a fresh one the same rendition.
        set_all_seeds(seed)
        inputs = prepare_inputs(tokenizer, audio_tokenizer, model, [request],
                                tmpl, guidance_scale=cfg_scale,
                                guidance_scale_ref=None, guidance_scale_ins=None)
        for chunk in runtime.iter_audio_chunks(inputs,
                                               request_id=request["id"]):
            yield pcm16(chunk.audio, gain)

    return render


def load_engine(args):
    """Load, capture, warm. Returns (render, graphs, error, config) -- never
    raises for a graph problem, only for a model that will not load at all.

    ``render`` is a started, warmed RenderWorker: a ``render(text, gain)``
    callable like any other, whose ONE thread has already paid the 16.2 GiB
    first-render transient, here, inside the GPU flock, rather than on
    Hunter's first sentence.
    """
    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)
    # Triton's bundled ptxas (12.8) has no sm_121a, so the depth decoder's
    # torch.compile inside the graph path fails without this. The unit sets
    # both of these; setdefault means a hand-run keeps whatever it was given.
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    # sdpa reproduces the rated arm exactly (the checkpoint's own
    # preferred_attn_implementation for the text encoder).
    os.environ.setdefault("BREEZE_TEXT_ENCODER_ATTN", "sdpa")

    import torch
    from dataclasses import replace
    from breeze_infer.runtime import resolve_device, update_generation_config_for_breeze
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
    from models.quantized_checkpoint import load_quantized_runtime
    from models.warmup_profile import load_warmup_profile

    device = resolve_device()
    t0 = time.perf_counter()
    tokenizer, model, audio_tokenizer, _stats = load_quantized_runtime(
        args.model, device=device, offload_embeddings=False)
    update_generation_config_for_breeze(model)
    load_s = time.perf_counter() - t0
    print(f"breeze: checkpoint resident in {load_s:.1f}s "
          f"({args.model}) free={free_mem_gb():.1f}GB", flush=True)

    config = FastStreamingConfig(
        max_new_tokens=args.max_new_tokens, max_seq_len=MAX_SEQ_LEN,
        fast_all=None, fast_text_encoder=False, fast_backbone_prefill=False,
        fast_backbone_decode=True, fast_depth_decoder=True, fast_codec=True,
        repetition_penalty=args.repetition_penalty)
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config,
                                         tokenizer=tokenizer)

    error = None
    try:
        profile = load_warmup_profile(Path(args.repo) / args.fast_config)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"breeze: graph capture {manifest.get('total_elapsed_ms', 0)/1000:.1f}s",
              flush=True)
    except Exception as exc:
        # NOT fatal here on purpose: a server that answers ready:false is
        # diagnosable from `journalctl` and from a ping, and Jarvis reads it
        # and uses F5. A server that refuses to bind looks identical to a
        # unit that was never installed.
        traceback.print_exc()
        error = f"graph capture failed: {type(exc).__name__}: {exc}"

    graphs = graphs_captured(runtime)
    config = {
        "template": args.template, "instruction": args.instruction,
        "cfg_scale": args.cfg_scale, "seed": args.seed,
        "temperature": getattr(model.generation_config, "temperature", None),
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens, "sr": SAMPLE_RATE,
        "attn": ATTN,
        "text_encoder_attn": os.environ.get("BREEZE_TEXT_ENCODER_ATTN"),
    }
    print(f"breeze: graphs {graphs} free={free_mem_gb():.1f}GB "
          f"alloc={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB",
          flush=True)
    if not all_graphs_captured(graphs):
        error = error or (
            "CUDA graphs were not captured; int4 without graphs measures "
            "RTF 1.131 and underruns mid-utterance")
        return None, graphs, error, config

    ref_text = Path(args.ref_text).read_text().strip()
    render = make_renderer(runtime, tokenizer, audio_tokenizer, model,
                           template=args.template, ref_audio=args.ref,
                           ref_text=ref_text, instruction=args.instruction,
                           cfg_scale=args.cfg_scale, seed=args.seed)

    # THE ONE RENDER THREAD, started here so that every render this process
    # ever serves -- including the two warm ones below -- runs on it. See
    # RenderWorker: the first render on any thread costs a +16.2 GiB transient
    # and later ones on that same thread cost 0.0-0.3 GiB, so this is where
    # that bill is paid: at boot, under the GPU flock, next to the 18.2 GiB
    # capture transient the flock already exists to serialise.
    worker = RenderWorker(render)

    # TWO warm renders, and the first is thrown away without being measured:
    # measured on this box, the first render after capture is 1.509 s to
    # first audio at RTF 1.868 -- it would underrun. The second is the one
    # the numbers describe. A warm render that RAISES is a sidecar that
    # cannot speak, so it fails readiness rather than waiting to fail on
    # Hunter's first reply.
    try:
        worker.start()
        print(f"breeze: render worker on thread {worker.ident}", flush=True)
        for label in ("discarded", "measured"):
            t = time.perf_counter()
            first = None
            n = 0
            for block in worker(args.warm_text, 1.0):
                if first is None:
                    first = time.perf_counter() - t
                n += len(block)
            wall = time.perf_counter() - t
            secs = n / 2 / SAMPLE_RATE
            print(f"breeze: warm ({label}) {secs:.2f}s audio in {wall:.2f}s "
                  f"RTF {wall / secs if secs else 0:.3f} "
                  f"first {first if first is None else round(first, 3)}s",
                  flush=True)
    except Exception as exc:
        traceback.print_exc()
        worker.stop()
        return None, graphs, f"warm render failed: {type(exc).__name__}: {exc}", config

    # WHAT THIS THING ACTUALLY COSTS, on one line in journalctl, after the
    # warm renders rather than before them -- the KV caches and codec buffers
    # are allocated by those, so the print inside the capture block above
    # describes a smaller program. Every other comment in this build quotes
    # 13.5 GB and 18.3 GB from render_q.py's external MemFree deltas, not from
    # this server; MEASUREMENTS.json's own peak_gpu_alloc_gb for this exact Q4
    # config is 6.322. This is the number that settles it.
    #
    # empty_cache() first, and not only for the print: on GB10 the GPU memory
    # IS the system memory, and torch's caching allocator never returns
    # reserved-but-free blocks to the OS on its own -- so whatever the 22 s
    # graph capture left reserved would stay charged against MemFree, and
    # against every other tenant, for the life of the process.
    torch.cuda.empty_cache()
    print(f"breeze: resident free={free_mem_gb():.1f}GB "
          f"alloc={torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB "
          f"reserved={torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB "
          f"peak={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB "
          f"render_thread={worker.ident}",
          flush=True)
    return worker, graphs, None, config


# ------------------------------------------------------------------- serve
def socket_owner(sock_path, timeout: float = 2.0):
    """The ping reply from a sidecar ALREADY listening on ``sock_path``; ``{}``
    when one is listening but did not answer; ``None`` when nothing is there.

    connect() on AF_UNIX succeeds only against a bound, listening socket, so
    this separates a live owner from the stale socket FILE that a wiped /tmp
    and an unclean exit leave behind -- and that distinction is the whole
    point. serve() used to unlink and rebind unconditionally, so a second
    instance silently stole the socket and left the first one alive, resident
    and unreachable: no pidfile, and no way to find it that this box's rules
    permit (matching a process by cmdline text is what caused five self-kills).
    The GPU flock is no defence, because main() releases it at the end of
    load_engine -- so the second instance takes it freely and loads a second
    copy of a 13.5 GB model before it ever reaches the bind.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(sock_path))
    except OSError:
        return None                      # ENOENT or ECONNREFUSED: nobody home
    try:
        sock.sendall(b'{"ping": true}\n')
        buf = b""
        while not buf.endswith(b"\n"):
            part = sock.recv(65536)
            if not part:
                break
            buf += part
        return json.loads(buf.decode("utf-8") or "{}")
    except Exception:
        return {}                        # listening but wedged -- still not ours
    finally:
        with contextlib.suppress(OSError):
            sock.close()


def _refuse_second_instance(sock_path, where: str) -> bool:
    """True (and says why) when a live sidecar already owns ``sock_path``."""
    owner = socket_owner(sock_path)
    if owner is None:
        return False
    print(f"breeze: REFUSING to {where} -- a live sidecar already answers on "
          f"{sock_path} ({owner or 'it did not answer a ping'}). Two instances "
          f"mean two resident copies of a 13.5 GB model and an orphan that "
          f"nothing on this box may go looking for. Stop that one first: "
          f"systemctl --user status jarvis-breeze.service", file=sys.stderr,
          flush=True)
    return True


def _serve_conn(service: BreezeService, conn, live: threading.Semaphore) -> None:
    """One connection, on its own thread -- the last net under handle().

    handle() is supposed to swallow everything; before it did, a malformed
    request line propagated through the bare accept loop and out of main(),
    taking the resident model with it. This catch means no future handler bug
    can do that again.
    """
    try:
        service.handle(conn)
    except Exception:
        traceback.print_exc()
    finally:
        with contextlib.suppress(OSError):
            conn.close()
        live.release()


def serve(sock_path, service: BreezeService) -> int:
    """Accept forever, one THREAD per connection.

    Thread-per-connection rather than the accept-then-handle loop this started
    as, for two measured reasons. A ping or a completion receipt used to queue
    behind an in-flight render -- a 6 s render made Jarvis's
    _ensure_breeze_server cost 6.0 s on the speak path -- and a peer that
    connected and never finished its request line wedged the entire sidecar
    for the connection deadline. Generation itself is still serialised, by
    BreezeService._render_lock, because there is only one resident model.

    These threads do the SOCKET I/O only. The model runs on the one thread
    RenderWorker owns, whatever connection asked for it -- that is what makes
    a render cost 0.04 GiB instead of 16.37.
    """
    if _refuse_second_instance(sock_path, "bind"):
        return 2
    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock_path)
    Path(sock_path).parent.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    os.chmod(sock_path, 0o600)
    srv.listen(MAX_CONNECTIONS)
    print(f"breeze: listening on {sock_path} (ready={service.ready})", flush=True)
    live = threading.Semaphore(MAX_CONNECTIONS)
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            traceback.print_exc()
            print("breeze: the listening socket is gone", file=sys.stderr,
                  flush=True)
            return 1
        if not live.acquire(blocking=False):
            _send_json(conn, {"ok": False, "error": "too many connections"})
            with contextlib.suppress(OSError):
                conn.close()
            continue
        try:
            threading.Thread(target=_serve_conn, args=(service, conn, live),
                             daemon=True, name="breeze-conn").start()
        except RuntimeError:                 # out of threads: do not leak the slot
            traceback.print_exc()
            live.release()
            with contextlib.suppress(OSError):
                conn.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--model", type=Path, default=CKPT_DEFAULT)
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True,
                    help="path to the reference transcript (not the text)")
    ap.add_argument("--instruction", default=INSTRUCTION)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--cfg-scale", type=float, default=CFG_SCALE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--fast-config", default=FAST_CONFIG)
    ap.add_argument("--gpu-lock", default=str(GPU_LOCK_DEFAULT))
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--min-available-gb", type=float, default=MIN_AVAILABLE_GB)
    ap.add_argument("--warm-text", default="Warming up the engine, sir.")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.model, args.repo, Path(args.ref), Path(args.ref_text)):
        if not Path(path).exists():
            print(f"breeze: cannot start, missing {path}", file=sys.stderr)
            return 2

    # Before the load, not merely before the bind: main() releases the GPU
    # flock at the end of load_engine, so a second instance takes it freely
    # and pays for a whole second resident model before serve() would notice.
    if _refuse_second_instance(args.socket, "start"):
        return 2

    engine = graphs = error = config = None
    # The lock covers load AND capture AND both warm renders; only the
    # serving loop runs outside it.
    with gpu_lock(args.gpu_lock):
        free, avail = free_mem_gb(), mem_gb("MemAvailable")
        if free < args.min_free_gb or avail < args.min_available_gb:
            print(f"breeze: REFUSING to start -- MemFree {free:.1f}GB "
                  f"(need {args.min_free_gb}), MemAvailable {avail:.1f}GB "
                  f"(need {args.min_available_gb}). The model takes 13.5GB of "
                  f"free pages and graph capture then needs 18.2GB more. "
                  f"Free some first (ollama holds ~18.6GB pinned) and start "
                  f"the unit again; Jarvis is speaking through F5 meanwhile.",
                  file=sys.stderr)
            return 2
        print(f"breeze: MemFree {free:.1f}GB MemAvailable {avail:.1f}GB at start",
              flush=True)
        # Again, now that the lock is ours. The check above can pass while the
        # other instance is still LOADING -- it holds this flock and has not
        # bound its socket yet -- and without this we would wait out its whole
        # ~30 s start and then load a second 13.5 GB copy before serve()
        # noticed. serve() still checks once more, for the milliseconds
        # between its release of this lock and its bind.
        if _refuse_second_instance(args.socket, "start"):
            return 2
        engine, graphs, error, config = load_engine(args)

    # ``engine`` is load_engine's warmed RenderWorker (or, under a test's
    # monkeypatch, a plain callable BreezeService wraps in one of its own).
    service = BreezeService(render=engine, sample_rate=SAMPLE_RATE,
                            ready=engine is not None, graphs=graphs,
                            error=error, config=config)
    if not service.ready:
        print(f"breeze: NOT READY -- {service._why_not()}", file=sys.stderr,
              flush=True)
    try:
        return serve(args.socket, service)
    finally:
        # serve() only returns when the listening socket is gone or a second
        # instance owns it; either way the render thread should not outlive
        # this call while the process is on its way out.
        service.close()


if __name__ == "__main__":
    sys.exit(main())
