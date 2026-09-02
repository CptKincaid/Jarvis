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
import socket
import struct
import sys
import time
import traceback
import wave
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
# Idle MemFree on this box is 33.5-34.7 GiB, so this gate is genuinely close
# to the line and will refuse on a busy afternoon. That is the intended
# behaviour -- Jarvis keeps speaking through F5 -- and the lever when it
# refuses is the 18.6 GiB ollama has pinned at keep_alive -1, not this
# number.
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

# One request at a time, but a long line is minutes of generation if the
# client stops reading; the socket deadline only bites when nobody is there.
CONN_TIMEOUT_S = 900.0

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
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


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
        self.render = render
        self.sample_rate = int(sample_rate)
        self.graphs = dict(graphs or {})
        # id -> {"complete": bool, ...} for the receipt query. See the
        # module docstring: EOF alone cannot say whether a stream finished.
        self._receipts: "collections.OrderedDict[str, dict]" = (
            collections.OrderedDict())
        # Reported on every ping so Jarvis can WARN when the running sidecar
        # does not match the parameters its speech cache is keyed on: the
        # sidecar applies the voice, tts.BREEZE_PARAMS only records it, and a
        # unit hand-started with a different --cfg-scale would otherwise file
        # one voice under another's key in silence.
        self.config = dict(config or {})
        # ready is never a synonym for "loaded": see the module docstring.
        self.ready = bool(ready and render is not None
                          and all_graphs_captured(self.graphs))
        self.error = error

    # ------------------------------------------------------------- replies
    def ping(self) -> dict:
        reply = {"ok": True, "ready": self.ready,
                 "graphs": all_graphs_captured(self.graphs),
                 "detail": self.graphs, "sr": self.sample_rate,
                 "config": self.config}
        if self.error:
            reply["error"] = self.error
        return reply

    def receipt(self, request_id: str) -> dict:
        """Did that streamed chunk finish? See the module docstring."""
        got = self._receipts.get(request_id)
        if got is None:
            return {"ok": False, "id": request_id,
                    "error": "unknown request id"}
        return {"ok": True, "id": request_id, **got}

    def _remember(self, request_id, outcome: dict) -> None:
        if not request_id:
            return
        self._receipts[str(request_id)] = outcome
        while len(self._receipts) > RECEIPTS:
            self._receipts.popitem(last=False)

    def _not_ready(self) -> dict:
        return {"ok": False, "error": self.error or
                "breeze sidecar is not ready (CUDA graphs were not captured)"}

    # ------------------------------------------------------------ dispatch
    def handle(self, conn) -> None:
        """Serve exactly one request on ``conn``. Never raises."""
        try:
            conn.settimeout(CONN_TIMEOUT_S)
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
        if req.get("ping"):
            _send_json(conn, self.ping())
            return
        if req.get("status"):
            _send_json(conn, self.receipt(str(req["status"])))
            return
        if not self.ready:
            _send_json(conn, self._not_ready())
            return
        text = (req.get("text") or "").strip()
        if not text:
            _send_json(conn, {"ok": False, "error": "no text"})
            return
        gain = float(req.get("gain", 1.0))
        if req.get("stream"):
            self._serve_stream(conn, text, gain, req.get("id"))
        else:
            self._serve_file(conn, text, req.get("out"), gain)

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
        if not out:
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
            with wave.open(str(out), "wb") as fh:
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
    raises for a graph problem, only for a model that will not load at all."""
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

    # TWO warm renders, and the first is thrown away without being measured:
    # measured on this box, the first render after capture is 1.509 s to
    # first audio at RTF 1.868 -- it would underrun. The second is the one
    # the numbers describe. A warm render that RAISES is a sidecar that
    # cannot speak, so it fails readiness rather than waiting to fail on
    # Hunter's first reply.
    try:
        for label in ("discarded", "measured"):
            t = time.perf_counter()
            first = None
            n = 0
            for block in render(args.warm_text, 1.0):
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
        return None, graphs, f"warm render failed: {type(exc).__name__}: {exc}", config
    return render, graphs, None, config


# ------------------------------------------------------------------- serve
def serve(sock_path, service: BreezeService) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock_path)
    Path(sock_path).parent.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    os.chmod(sock_path, 0o600)
    srv.listen(4)
    print(f"breeze: listening on {sock_path} (ready={service.ready})", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            service.handle(conn)
        finally:
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

    render = graphs = error = config = None
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
        render, graphs, error, config = load_engine(args)

    service = BreezeService(render=render, sample_rate=SAMPLE_RATE,
                            ready=render is not None, graphs=graphs,
                            error=error, config=config)
    if not service.ready:
        print(f"breeze: NOT READY -- {error}", file=sys.stderr, flush=True)
    serve(args.socket, service)
    return 0


if __name__ == "__main__":
    sys.exit(main())
