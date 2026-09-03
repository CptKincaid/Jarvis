"""The Breeze-TTS-2 sidecar, its protocol, and the client that streams from it.

WHY. In the round-11 BLIND listening test Hunter rated Breeze 4.71 against a
hidden hosted-Fish control at 4.64 while the shipped F5 voice scored 2.79, and
a blind A/B of the quantized build against the bf16 one came out inside the
round's noise floor. So this is the best voice available and it is local --
but it is also a 13.5 GB resident model whose one-time CUDA-graph capture
transiently demands ~18.3 GB on a box where GPU memory IS system memory.

Two properties are therefore load-bearing and are what most of this file
asserts:

  * IT IS OFF UNTIL HE TURNS IT ON. With tts_engine anything but "breeze",
    every number, table and cache key on the F5 path is what it was before
    (the last section proves it), and nothing here spawns, loads or pings
    anything.
  * READY MEANS THE CUDA GRAPHS WERE CAPTURED. int4 without them measures
    RTF 1.131 -- above real time, so it underruns mid-utterance -- against
    0.742 with them. A sidecar that loaded but did not capture must be
    refused, and Jarvis must speak in the 2.79 voice rather than a broken
    4.71 one.

No GPU, no audio, no display, no network: the server's protocol is driven
through a fake render callable, the client through a real AF_UNIX server in
tmp_path that speaks the protocol, and the players are faked at the Popen
seam.
"""
import fcntl
import importlib.util
import inspect
import io
import json
import os
import queue
import socket
import subprocess
import threading
import time
import wave
from pathlib import Path

import pytest

from jarvis import tts as tts_mod
from jarvis.config import CONFIG, PATHS
from jarvis.tts import TTS

REPO = Path(__file__).resolve().parent.parent
UNIT = REPO / "scripts" / "systemd" / "jarvis-breeze.service"
SETUP = REPO / "scripts" / "setup_breeze_service.sh"


def _load_server():
    """scripts/breeze_server.py, imported by path.

    It keeps every heavy import (torch, the breeze source tree) inside
    main()/load_engine() precisely so this works: the protocol and its
    failure paths are testable with no GPU and no venv switch.
    """
    spec = importlib.util.spec_from_file_location(
        "breeze_server", REPO / "scripts" / "breeze_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load_server()


# ===========================================================================
# helpers
# ===========================================================================
def pcm(nframes: int, value: int = 1000) -> bytes:
    return b"".join(int(value).to_bytes(2, "little", signed=True)
                    for _ in range(nframes))


def wav_bytes(seconds=0.4, rate=24000, amp=8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(rate * seconds)
        w.writeframes(b"".join(int(amp * ((i // 40) % 2 * 2 - 1)).to_bytes(
            2, "little", signed=True) for i in range(n)))
    return buf.getvalue()


def wait_until(pred, timeout=5.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.01)
    return pred()


ALL_GRAPHS = {"fast_enabled": True, "backbone_decode": True,
              "depth_decoder": True, "codec": True}


class FakeProc:
    """A player that 'exits' once its stdin is closed (or at once for the
    file chain), recording when it was spawned and what it was fed."""
    spawned: list = []

    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self.t = time.monotonic()
        self.fed = bytearray()
        self.returncode = None
        self._closed = kw.get("stdin") is None
        proc = self

        class _Stdin:
            def write(self_, data):
                proc.fed += data

            def close(self_):
                proc._closed = True

        self.stdin = _Stdin()
        FakeProc.spawned.append(self)

    def poll(self):
        if self._closed and self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self._closed = True

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture(autouse=True)
def _fresh_spawns():
    FakeProc.spawned = []
    yield
    FakeProc.spawned = []


# ===========================================================================
# 1. the wire format
# ===========================================================================
def test_the_two_wav_header_builders_agree_byte_for_byte():
    """The sidecar writes this header and jarvis/tts.py's _LiveEnvelope
    parses it; paplay decodes the same bytes from stdin. A disagreement here
    is silence with no error anywhere, so it is asserted rather than trusted
    to two copies of the same struct format."""
    for rate in (24000, 16000):
        assert bs.wav_header(rate) == tts_mod.wav_header(rate, 1, 2, None)
        assert bs.wav_header(rate, 1, 2, 400) == tts_mod.wav_header(rate, 1, 2, 400)
    assert len(bs.wav_header(24000)) == tts_mod._WAV_HEADER_BYTES == 44


def test_pcm16_applies_the_gain_and_clips_instead_of_normalising():
    """Per-chunk peak normalisation would make the level pump audibly inside
    one sentence, which is why apply_output_gain is a constant too."""
    quiet = bs.pcm16([0.0, 0.25, -0.25], gain=2.0)
    assert quiet == b"".join(int(v).to_bytes(2, "little", signed=True)
                             for v in (0, 16383, -16383))
    # a hot sample is clipped, not scaled -- and does not drag the chunk down
    hot = bs.pcm16([0.9, 0.1], gain=2.0)
    assert hot[:2] == (32767).to_bytes(2, "little", signed=True)
    assert hot[2:] == (6553).to_bytes(2, "little", signed=True)


def test_the_memory_gate_reads_memfree_not_memavailable(tmp_path):
    """MemAvailable never dropped below 41.7 GiB during a real graph capture
    while actual free pages hit 2.39 GiB, because ~39 GiB of it was
    reclaimable page cache. render_q.py's own gate reads MemAvailable and
    would have waved that through."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       127606644 kB\n"
                       "MemFree:          2097152 kB\n"
                       "MemAvailable:    73400320 kB\n")
    assert bs.free_mem_gb(str(meminfo)) == pytest.approx(2.0, abs=0.01)
    assert bs.mem_gb("MemAvailable", str(meminfo)) == pytest.approx(70.0, abs=0.01)
    assert bs.free_mem_gb(str(tmp_path / "nope")) == -1.0


def test_the_free_page_floor_covers_the_model_and_the_capture():
    """13.5 GiB for the resident model plus 18.2 GiB of capture transient is
    31.7 GiB before the trough reaches zero, so the floor cannot be below
    that -- and a companion analysis proposing 25 would have landed at -6."""
    assert bs.MIN_FREE_GB >= 32
    assert bs.MIN_AVAILABLE_GB >= 60      # the box's standing rule for GPU jobs


# ===========================================================================
# 2. the readiness gate
# ===========================================================================
class _Graph:
    def __init__(self, captured):
        self.captured = captured


class _Lane:
    def __init__(self, graph):
        self.cuda_graph = graph


class _Codec:
    def __init__(self, lanes):
        self.lanes = lanes


class _Runtime:
    """Only the attributes graphs_captured() reads."""
    def __init__(self, backbone=(True, True), depth=True, codec=(object(),),
                 fast=True):
        self._backbone_graphs = {i + 1: _Graph(c) for i, c in enumerate(backbone)}
        self._depth_decoder_graph = _Graph(depth)
        self._codec_runtime = _Codec([_Lane(g) for g in codec])
        self.fast_enabled = fast


def test_graphs_captured_reads_the_objects_not_the_config():
    """fast_enabled is only the REQUEST -- it is any() over the config flags,
    and _ensure_graphs quietly calls prepare_eager() for any lane whose flag
    is off. Only the graph objects know what actually happened."""
    assert bs.all_graphs_captured(bs.graphs_captured(_Runtime()))
    # one backbone bucket captured eagerly
    assert not bs.all_graphs_captured(
        bs.graphs_captured(_Runtime(backbone=(True, False))))
    assert not bs.all_graphs_captured(bs.graphs_captured(_Runtime(depth=False)))
    assert not bs.all_graphs_captured(bs.graphs_captured(_Runtime(codec=(None,))))
    assert not bs.all_graphs_captured(bs.graphs_captured(_Runtime(fast=False)))


def test_graphs_captured_degrades_rather_than_raising_on_a_library_change():
    """A renamed attribute must read as 'not captured' at ping time, not
    explode when Jarvis asks."""
    detail = bs.graphs_captured(object())
    assert detail == {"fast_enabled": False, "backbone_decode": False,
                      "depth_decoder": False, "codec": False}


def test_ready_is_false_without_every_graph():
    """int4 with no graphs is RTF 1.131 -- above real time, so it underruns
    mid-utterance. A model that merely loaded is not ready."""
    svc = bs.BreezeService(render=lambda t, g: iter(()), ready=True,
                           graphs={**ALL_GRAPHS, "codec": False})
    assert svc.ready is False
    assert svc.ping()["graphs"] is False
    assert svc.ping()["ready"] is False
    # and with everything captured it is
    good = bs.BreezeService(render=lambda t, g: iter(()), ready=True,
                            graphs=ALL_GRAPHS)
    assert good.ready is True and good.ping()["graphs"] is True


def test_ready_is_false_without_a_renderer():
    assert bs.BreezeService(render=None, ready=True, graphs=ALL_GRAPHS).ready is False


# ===========================================================================
# 3. the protocol, driven through a fake renderer over a real socketpair
# ===========================================================================
def serve_once(service, request: dict) -> tuple[bytes, list]:
    """Send ``request`` to ``service`` over a socketpair, return every byte
    it wrote back. The service side is a real socket, so the failure paths
    exercise the real sendall/shutdown."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    done = threading.Event()

    def _run():
        try:
            service.handle(b)
        finally:
            b.close()
            done.set()

    threading.Thread(target=_run, daemon=True).start()
    a.sendall(json.dumps(request).encode() + b"\n")
    out = b""
    a.settimeout(5)
    while True:
        try:
            part = a.recv(65536)
        except (OSError, socket.timeout):
            break
        if not part:
            break
        out += part
    a.close()
    done.wait(5)
    return out


def split_reply(raw: bytes) -> tuple[dict, bytes]:
    line, _, rest = raw.partition(b"\n")
    return json.loads(line.decode() or "{}"), rest


def blocks_render(*blocks, fail_after=None):
    """A fake ``render(text, gain)``: yields ``blocks``, optionally raising
    after ``fail_after`` of them."""
    def _render(text, gain=1.0):
        for i, block in enumerate(blocks):
            if fail_after is not None and i == fail_after:
                raise RuntimeError("cuda blew up")
            yield block
    return _render


def ready_service(render):
    return bs.BreezeService(render=render, ready=True, graphs=ALL_GRAPHS,
                            config={"cfg_scale": 4.0})


def test_ping_carries_the_graph_detail_and_the_pinned_config():
    svc = ready_service(blocks_render(pcm(10)))
    head, rest = split_reply(serve_once(svc, {"ping": True}))
    assert head == {"ok": True, "ready": True, "graphs": True,
                    "detail": ALL_GRAPHS, "sr": 24000,
                    "config": {"cfg_scale": 4.0}}
    assert rest == b""


def test_a_render_request_on_a_degraded_sidecar_is_refused_with_no_audio():
    svc = bs.BreezeService(render=blocks_render(pcm(10)), ready=True,
                           graphs={**ALL_GRAPHS, "depth_decoder": False},
                           error="graph capture failed: boom")
    head, rest = split_reply(serve_once(svc, {"text": "Hello.", "stream": True}))
    assert head["ok"] is False
    assert "boom" in head["error"]
    assert rest == b""              # not one byte: the caller uses F5 instead


def test_streaming_sends_the_json_line_then_a_riff_header_then_pcm():
    svc = ready_service(blocks_render(pcm(4), pcm(6)))
    head, rest = split_reply(
        serve_once(svc, {"text": "Good evening.", "stream": True, "id": "r1"}))
    assert head["ok"] is True and head["stream"] is True
    assert head["sr"] == 24000 and head["channels"] == 1 and head["sampwidth"] == 2
    assert rest[:44] == tts_mod.wav_header(24000, 1, 2, None)
    assert rest[44:] == pcm(4) + pcm(6)
    assert svc.receipt("r1") == {"ok": True, "id": "r1", "complete": True,
                                 "bytes": 20, "seconds": 20 / 2 / 24000,
                                 "wall": pytest.approx(svc._receipts["r1"]["wall"]),
                                 "first_audio": pytest.approx(
                                     svc._receipts["r1"]["first_audio"])}


def test_the_json_line_is_withheld_until_the_first_audio_exists():
    """A failure during prepare must cost an error line and NOTHING else, so
    the caller can render that chunk on F5 rather than speaking half a
    sentence twice."""
    svc = ready_service(blocks_render(pcm(4), fail_after=0))
    head, rest = split_reply(
        serve_once(svc, {"text": "Good evening.", "stream": True, "id": "r2"}))
    assert head == {"ok": False, "error": "RuntimeError: cuda blew up"}
    assert rest == b""
    assert svc.receipt("r2")["complete"] is False


def test_a_failure_after_audio_sends_no_error_line_and_leaves_a_receipt():
    """Half a sentence is already in the room. Nothing useful can be said on
    that connection -- everything after the header is raw PCM -- so the
    receipt is how the caller learns not to cache it."""
    svc = ready_service(blocks_render(pcm(4), pcm(4), fail_after=1))
    head, rest = split_reply(
        serve_once(svc, {"text": "Half of this.", "stream": True, "id": "r3"}))
    assert head["ok"] is True                     # audio had already started
    assert rest == tts_mod.wav_header(24000, 1, 2, None) + pcm(4)
    assert svc.receipt("r3")["complete"] is False
    assert "cuda blew up" in svc.receipt("r3")["error"]


def test_a_forgotten_request_id_is_not_a_completion():
    """NO RECEIPT MEANS INCOMPLETE: a sidecar that died mid-chunk answers
    nothing at all, so the safe reading has to be the default one."""
    svc = ready_service(blocks_render(pcm(4)))
    assert svc.receipt("never-seen") == {"ok": False, "id": "never-seen",
                                         "error": "unknown request id"}


def test_receipts_are_bounded():
    svc = ready_service(blocks_render(pcm(2)))
    for i in range(bs.RECEIPTS + 5):
        serve_once(svc, {"text": "x", "stream": True, "id": f"r{i}"})
    assert len(svc._receipts) == bs.RECEIPTS
    assert svc.receipt("r0")["ok"] is False          # evicted, so incomplete
    assert svc.receipt(f"r{bs.RECEIPTS + 4}")["complete"] is True


def test_the_compatible_mode_writes_a_pcm16_wav_the_cache_can_read(tmp_path):
    out = tmp_path / "chunk.wav"
    svc = ready_service(blocks_render(pcm(240), pcm(240)))
    head, rest = split_reply(
        serve_once(svc, {"text": "Always, sir.", "out": str(out)}))
    assert head["ok"] is True and rest == b""
    assert head["seconds"] == pytest.approx(480 / 24000)
    with wave.open(str(out), "rb") as fh:
        assert (fh.getnchannels(), fh.getsampwidth(), fh.getframerate()) == (1, 2, 24000)
        assert fh.getnframes() == 480
    # the phone renderer reads it through the same helper the room does
    assert tts_mod.wav_pcm(str(out))[0] == (24000, 1, 2)


def test_the_compatible_mode_reports_a_failure_instead_of_a_half_file(tmp_path):
    out = tmp_path / "chunk.wav"
    svc = ready_service(blocks_render(pcm(10), fail_after=0))
    head, _ = split_reply(serve_once(svc, {"text": "x", "out": str(out)}))
    assert head["ok"] is False and "cuda blew up" in head["error"]


def test_a_request_with_no_text_or_no_out_is_refused_not_crashed(tmp_path):
    svc = ready_service(blocks_render(pcm(4)))
    assert split_reply(serve_once(svc, {"text": "   "}))[0]["ok"] is False
    assert split_reply(serve_once(svc, {"text": "hi"}))[0]["ok"] is False
    assert split_reply(serve_once(svc, {"nonsense": 1}))[0]["ok"] is False


def test_the_gain_reaches_the_renderer():
    seen = []

    def render(text, gain=1.0):
        seen.append(gain)
        yield pcm(2)

    serve_once(ready_service(render), {"text": "x", "stream": True, "gain": 2.8})
    assert seen == [2.8]


# ===========================================================================
# 4. the pinned rig, in two files
# ===========================================================================
def test_the_rated_configuration_is_the_same_on_both_sides():
    """The sidecar APPLIES these; tts.BREEZE_PARAMS only records them in the
    speech-cache key. A silent disagreement would file one voice under
    another voice's key."""
    p = tts_mod.BREEZE_PARAMS
    assert p["instruction"] == bs.INSTRUCTION == tts_mod.BREEZE_INSTRUCTION
    assert p["template"] == bs.TEMPLATE == "ref_edit_tata"
    assert p["cfg_scale"] == bs.CFG_SCALE == 4.0
    assert p["seed"] == bs.SEED == 1234
    assert p["temperature"] == bs.TEMPERATURE == 0.9
    assert p["repetition_penalty"] == bs.REPETITION_PENALTY == 1.1
    assert p["max_new_tokens"] == bs.MAX_NEW_TOKENS == 1500
    assert p["sr"] == bs.SAMPLE_RATE == 24000
    assert p["attn"] == bs.ATTN == "eager"


def test_cfg_scale_is_not_a_speed_knob():
    """With the graphs on, lowering it buys 0.4% and is what makes the voice
    direction fire at all. Pinned so a future latency pass cannot take it."""
    assert tts_mod.BREEZE_PARAMS["cfg_scale"] == 4.0


# ===========================================================================
# 5. the unit and its installer
# ===========================================================================
def _unit_fields():
    fields = {}
    for line in UNIT.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "[")):
            continue
        k, _, v = line.partition("=")
        fields.setdefault(k.strip(), []).append(v.strip())
    return fields


def _home(path: str) -> str:
    return path.replace("%h", str(Path.home()))


def test_the_unit_limits_its_restarts_harder_than_f5s():
    """Every restart re-enters graph capture, which was measured driving
    system-wide MemFree to 2.39 GiB. A Restart=always loop through that is
    how this box gets powered off."""
    f = _unit_fields()
    assert f["Type"] == ["simple"] and f["Restart"] == ["always"]
    assert f["WantedBy"] == ["default.target"]
    assert int(f["StartLimitBurst"][0]) <= 5
    assert int(f["StartLimitIntervalSec"][0]) >= 600
    assert int(f["RestartSec"][0]) >= 30


def test_the_unit_pins_the_env_graph_capture_needs():
    """Without ptxas the depth decoder's compile fails on sm_121a and you get
    the working-but-too-slow server (RTF 1.131). sdpa reproduces the arm that
    was blind-rated."""
    env = dict(e.split("=", 1) for e in _unit_fields()["Environment"])
    assert env["TRITON_PTXAS_PATH"] == "/usr/local/cuda/bin/ptxas"
    assert env["BREEZE_TEXT_ENCODER_ATTN"] == "sdpa"


def test_the_unit_recreates_the_socket_dir_because_tmp_is_wiped_at_boot():
    pre = _unit_fields()["ExecStartPre"]
    assert any("mkdir -p /tmp/vss_voice" in p for p in pre), pre


def test_the_unit_uses_the_same_paths_as_tts_py():
    """One sidecar, one set of paths. If the unit and PATHS drift apart,
    Jarvis pings one socket while the unit serves another."""
    argv = _home(_unit_fields()["ExecStart"][0]).split()
    assert argv[0] == str(PATHS.BREEZE_PYTHON)
    assert argv[1] == str(Path.home() / "Jarvis" / "scripts" / "breeze_server.py")
    opts = dict(zip(argv[2::2], argv[3::2]))
    assert opts["--model"] == str(PATHS.BREEZE_CKPT)
    assert opts["--repo"] == str(PATHS.BREEZE_REPO)
    assert opts["--ref"] == str(PATHS.VOICE_REF_F5)
    assert opts["--ref-text"] == str(PATHS.VOICE_REF_F5_TEXT)
    # PATHS.BREEZE_SOCK is redirected by conftest; the unit targets the LIVE dir
    assert opts["--socket"] == "/tmp/vss_voice/" + PATHS.BREEZE_SOCK.name
    assert opts["--gpu-lock"].endswith("voice-training/.gpu.lock")


def test_every_unit_flag_is_one_breeze_server_accepts():
    argv = _unit_fields()["ExecStart"][0].split()
    accepted = {a.option_strings[0]
                for a in bs.build_parser()._actions if a.option_strings}
    for flag in argv[2::2]:
        assert flag in accepted, f"breeze_server.py does not take {flag}"


def test_the_installer_does_not_enable_or_start_the_unit():
    """Turning it on means putting a 13.5 GB resident model on the GPU. That
    moment is chosen by a person, which is also what keeps this feature
    opt-in on a box that reboots."""
    src = SETUP.read_text()
    assert "daemon-reload" in src
    # Only what the script RUNS: the closing message quotes both commands on
    # purpose, and a substring search over the whole file would read them as
    # if they were executed.
    ran = "\n".join(ln for ln in src.split("cat <<MSG")[0].splitlines()
                    if not ln.lstrip().startswith("#"))
    assert "systemctl --user enable" not in ran
    assert "systemctl --user start" not in ran
    # ...but it must TELL him both commands, and how to get back
    assert "systemctl --user start $UNIT" in src
    assert "systemctl --user stop $UNIT" in src
    assert 'tts_engine"]="breeze"' in src and 'tts_engine"]="f5"' in src


def test_the_installer_is_executable_and_syntactically_valid():
    assert os.access(SETUP, os.X_OK)
    subprocess.run(["bash", "-n", str(SETUP)], check=True)


# ===========================================================================
# 6. the client: a real AF_UNIX sidecar in tmp_path
# ===========================================================================
class FakeSidecar:
    """A unix-socket server that speaks the breeze protocol.

    ``handler(self, conn, req)`` writes the reply; the helpers below are the
    handlers the tests use. Everything runs in-process on AF_UNIX -- no
    network, no GPU, no venv.
    """

    def __init__(self, path, handler, threaded: bool = False):
        self.path = str(path)
        self.handler = handler
        # The real serve() gives every connection its own thread so a ping is
        # never behind a render. Off by default here so the ordering the other
        # tests assert stays exactly as it was.
        self.threaded = threaded
        self.requests: list = []
        self.receipts: dict = {}
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(8)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            if self.threaded:
                threading.Thread(target=self._one, args=(conn,),
                                 daemon=True).start()
            else:
                self._one(conn)

    def _one(self, conn):
        try:
            buf = b""
            while not buf.endswith(b"\n"):
                part = conn.recv(65536)
                if not part:
                    break
                buf += part
            if buf.strip():
                req = json.loads(buf.decode())
                self.requests.append(req)
                self.handler(self, conn, req)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def send(self, conn, payload):
        conn.sendall(json.dumps(payload).encode() + b"\n")

    def close(self):
        self.srv.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass


def sidecar_handler(blocks, ping=None, fail_after=None, gate=None,
                    refuse=None):
    """The standard handler: answers ping and status, streams ``blocks``.

    ``gate`` is waited on before the LAST block, so a test can prove that
    playback began while the stream was still open.
    """
    reply = ping if ping is not None else {"ok": True, "ready": True,
                                           "graphs": True, "detail": ALL_GRAPHS,
                                           "sr": 24000, "config": {}}

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, reply)
            return
        if req.get("status"):
            got = srv.receipts.get(req["status"])
            srv.send(conn, {"ok": True, **got} if got else
                     {"ok": False, "error": "unknown request id"})
            return
        rid = req.get("id")
        if refuse is not None:
            srv.receipts[rid] = {"complete": False, "error": refuse}
            srv.send(conn, {"ok": False, "error": refuse})
            return
        if not req.get("stream"):
            with wave.open(req["out"], "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(24000)
                fh.writeframes(b"".join(blocks))
            srv.send(conn, {"ok": True, "seconds": 0.1, "wall": 0.05})
            return
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000,
                        "channels": 1, "sampwidth": 2, "first_audio": 0.01})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None))
        n = 44
        for i, block in enumerate(blocks):
            if fail_after is not None and i == fail_after:
                srv.receipts[rid] = {"complete": False, "bytes": n,
                                     "error": "RuntimeError: cuda blew up"}
                return
            if gate is not None and i == len(blocks) - 1:
                gate.wait(5)
            conn.sendall(block)
            n += len(block)
        srv.receipts[rid] = {"complete": True, "bytes": n}
        conn.shutdown(socket.SHUT_WR)

    return handler


@pytest.fixture
def sock_path(tmp_path, monkeypatch):
    path = tmp_path / "breeze.sock"
    monkeypatch.setattr(tts_mod, "BREEZE_SOCK", path)
    return path


@pytest.fixture
def no_spawn(monkeypatch):
    """Jarvis must NEVER start a Breeze sidecar itself: the load costs 7.3 s
    plus 22.0 s of graph capture whose transient was measured at ~18.3 GB.
    That belongs under the GPU flock in the unit, not in a Tk app."""
    calls = []

    def _boom(*a, **kw):
        calls.append(a)
        raise AssertionError("Jarvis must not spawn the breeze sidecar")

    monkeypatch.setattr(tts_mod.subprocess, "Popen", _boom)
    return calls


# ----------------------------------------------------------- discovery
def test_alive_needs_the_graphs_not_just_ready(sock_path):
    srv = FakeSidecar(sock_path, sidecar_handler(
        [], ping={"ok": True, "ready": True, "graphs": False,
                  "detail": {**ALL_GRAPHS, "codec": False}}))
    try:
        assert tts_mod._breeze_alive() is False
    finally:
        srv.close()
    srv = FakeSidecar(sock_path, sidecar_handler([]))
    try:
        assert tts_mod._breeze_alive() is True
    finally:
        srv.close()


def test_alive_is_false_with_no_socket_at_all(sock_path):
    assert tts_mod._breeze_alive() is False


def test_a_running_sidecar_is_adopted_and_never_respawned(sock_path, no_spawn,
                                                          monkeypatch):
    unit_asked = []
    monkeypatch.setattr(tts_mod, "_breeze_unit_active",
                        lambda: unit_asked.append(1) or False)
    srv = FakeSidecar(sock_path, sidecar_handler([]))
    try:
        assert tts_mod._ensure_breeze_server() is True
    finally:
        srv.close()
    assert unit_asked == []          # a live socket answers the question
    assert no_spawn == []


def test_a_degraded_sidecar_is_refused_without_waiting_on_systemd(
        sock_path, no_spawn, monkeypatch):
    """It answered and said its graphs are gone. Waiting for a unit that is
    already up and already wrong would just delay the fallback."""
    monkeypatch.setattr(tts_mod, "_breeze_unit_active",
                        lambda: pytest.fail("must not ask systemd"))
    srv = FakeSidecar(sock_path, sidecar_handler(
        [], ping={"ok": True, "ready": False, "graphs": False,
                  "error": "CUDA graphs were not captured"}))
    try:
        assert tts_mod._ensure_breeze_server() is False
    finally:
        srv.close()


def test_an_activating_unit_is_waited_for_never_raced(sock_path, no_spawn,
                                                      monkeypatch):
    """Type=simple means the unit is 'active' ~30 s before the graphs exist."""
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: True)
    holder = {}

    def _late():
        time.sleep(0.4)
        holder["srv"] = FakeSidecar(sock_path, sidecar_handler([]))

    t = threading.Thread(target=_late, daemon=True)
    t.start()
    try:
        assert tts_mod._ensure_breeze_server(startup_timeout=10) is True
    finally:
        t.join(5)
        holder["srv"].close()
    assert no_spawn == []


def test_a_unit_that_dies_while_we_wait_stops_the_wait(sock_path, no_spawn,
                                                       monkeypatch):
    states = [True]
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: states[0])

    def _die():
        time.sleep(0.2)
        states[0] = False

    threading.Thread(target=_die, daemon=True).start()
    assert tts_mod._wait_for_breeze_unit(30) is False


def test_no_sidecar_and_no_unit_is_a_clean_no(sock_path, no_spawn, monkeypatch):
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: False)
    assert tts_mod._ensure_breeze_server() is False


def test_a_sidecar_whose_config_drifted_is_adopted_but_warned_about(
        sock_path, no_spawn, monkeypatch, caplog):
    """The sidecar applies the voice; BREEZE_PARAMS only keys the cache on
    it. A unit hand-started with --cfg-scale 2 would otherwise file a
    different voice under this voice's key in silence."""
    srv = FakeSidecar(sock_path, sidecar_handler(
        [], ping={"ok": True, "ready": True, "graphs": True,
                  "config": {"cfg_scale": 2.0, "seed": 1234}}))
    try:
        with caplog.at_level("WARNING"):
            assert tts_mod._ensure_breeze_server() is True
    finally:
        srv.close()
    assert any("cfg_scale=2.0" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------- load / fallback
def test_load_falls_back_to_f5_and_does_not_persist_it(sock_path, no_spawn,
                                                       monkeypatch, tmp_path):
    """A cold sidecar recovers on the next start. Writing "f5" to
    voice_settings.json here would silently un-choose Breeze forever after
    one bad boot -- he turned it on with one setting, only he turns it off."""
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: False)
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda *a, **k: True)
    saved = []
    monkeypatch.setattr(CONFIG, "save", lambda: saved.append(CONFIG.tts_engine))
    before = CONFIG.tts_engine
    t = TTS(engine="breeze", cache_dir=tmp_path / "cache")
    assert t.load() is True
    assert t.engine == "f5"
    assert saved == [] and CONFIG.tts_engine == before


def test_load_adopts_a_ready_sidecar_and_keeps_the_engine(sock_path, no_spawn,
                                                          tmp_path, monkeypatch):
    warmed = []
    monkeypatch.setattr(tts_mod.TTS, "warm_f5_fallback",
                        lambda self: warmed.append(1))
    srv = FakeSidecar(sock_path, sidecar_handler([]))
    try:
        t = TTS(engine="breeze", cache_dir=tmp_path / "cache")
        assert t.load() is True
        assert t.engine == "breeze"
    finally:
        srv.close()
    # The chunk fallback is F5, and an outage plan that has to cold-start is
    # not a plan: fish learned that on 2026-08-30 at ~180 s a chunk.
    assert warmed == [1]


# ------------------------------------------------------------- streaming
@pytest.fixture
def breeze(tmp_path, sock_path, monkeypatch, no_spawn):
    monkeypatch.setattr(tts_mod, "BREEZE_STREAM_PLAYBACK", True)
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: False)
    # The F5 fallback is adopted, never probed: these tests own the Popen
    # seam for the PLAYER, and _ensure_f5_server's systemctl call would land
    # in the same fake. It has its own tests (test_f5_service.py).
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda *a, **k: True)
    monkeypatch.setattr(tts_mod.TTS, "warm_f5_fallback", lambda self: None)
    monkeypatch.setattr(tts_mod.subprocess, "Popen", FakeProc)
    return TTS(engine="breeze", cache_dir=tmp_path / "cache")


def test_bytes_reach_the_player_before_the_stream_ends(breeze, sock_path,
                                                       tmp_path):
    """Breeze's 0.268 s engine time-to-first-audio only reaches the ear if
    the first bytes go to paplay's stdin while the rest is still on the GPU.
    Rendering the whole chunk first is 2.12 s at his median utterance."""
    audio = wav_bytes(0.4)[44:]
    body, tail = audio[:2000], audio[2000:]
    gate = threading.Event()
    srv = FakeSidecar(sock_path, sidecar_handler([body, tail], gate=gate))
    try:
        done = breeze.speak("Good evening, sir.")
        assert wait_until(lambda: bool(FakeProc.spawned))
        proc = FakeProc.spawned[0]
        assert proc.cmd == ["paplay", f"--client-name={tts_mod.SPEECH_CLIENT_NAME}"]
        assert wait_until(lambda: len(proc.fed) >= 44 + len(body))
        assert not done.is_set()          # still streaming
        gate.set()
        assert done.wait(10)
        assert bytes(proc.fed) == tts_mod.wav_header(24000, 1, 2, None) + audio
        assert breeze.cache.stats()["files"] == 1
    finally:
        srv.close()
    # cached under the BREEZE key, not some default engine's
    assert breeze.cache.get(breeze._cache_key("breeze", "Good evening, sir.")) \
        is not None
    assert not [p for p in tmp_path.glob("**/*.wav") if "cache" not in str(p)]


def test_a_two_sentence_reply_streams_both_chunks_in_order(breeze, sock_path):
    """Chunk N+1 renders while chunk N plays, and the consumer still hands
    them to the player in order -- the producer runs ahead at RTF 0.742."""
    first, second = wav_bytes(0.2)[44:], wav_bytes(0.3)[44:]
    order = []

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            srv.send(conn, {"ok": True, "complete": True})
            return
        order.append(req["text"])
        block = first if len(order) == 1 else second
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None) + block)
        conn.shutdown(socket.SHUT_WR)

    srv = FakeSidecar(sock_path, handler)
    try:
        breeze.speak("The reactor is holding steady this evening. "
                     "The workshop is quiet and the coffee is fresh.",
                     block=True)
    finally:
        srv.close()
    assert order == ["The reactor is holding steady this evening.",
                     "The workshop is quiet and the coffee is fresh."]
    assert len(FakeProc.spawned) == 2
    assert bytes(FakeProc.spawned[0].fed).endswith(first)
    assert bytes(FakeProc.spawned[1].fed).endswith(second)
    assert breeze.cache.stats()["files"] == 2


def test_a_long_sentence_is_not_comma_split_the_way_f5s_is():
    """The comma pass is what flattens a list into two utterances (measured:
    the pitch RESETS at the comma, +45 Hz mid-clause). Streaming at RTF 0.742
    cannot underrun, so there is nothing to buy by splitting."""
    t = TTS.__new__(TTS)
    t.cache = None
    assert t._chunk_limits("breeze") == (320, 8.0)
    line = ("On the calendar you have Biosensors at nine ten ay em, Magnetic "
            "Resonance Engineering at twelve forty pee em, and at four ten "
            "pee em your Electrical Design Lab presentation.")
    assert t._split_sentences(line, engine="breeze") == [line]


def test_a_refusal_before_any_audio_renders_that_chunk_on_f5(breeze, sock_path):
    srv = FakeSidecar(sock_path, sidecar_handler(
        [], refuse="not ready (CUDA graphs were not captured)"))
    f5 = []
    try:
        breeze._synth_f5 = lambda text, out: (f5.append(text),
                                              open(out, "wb").write(wav_bytes(0.1)))
        breeze.speak("Good evening, sir.", block=True)
    finally:
        srv.close()
    assert f5 == ["Good evening, sir."]
    # filed under the engine that RENDERED it, or it would replay in the
    # wrong voice once breeze is back
    assert breeze.cache.get(breeze._cache_key("f5", "Good evening, sir.")) is not None
    assert breeze.cache.get(breeze._cache_key("breeze", "Good evening, sir.")) is None


def test_a_failure_mid_stream_is_not_re_spoken_and_is_not_cached(breeze,
                                                                 sock_path):
    """Half a sentence has already been heard. Re-rendering it on F5 would
    say the first half twice, in a different voice."""
    audio = wav_bytes(0.4)[44:]
    srv = FakeSidecar(sock_path, sidecar_handler(
        [audio[:3000], audio[3000:]], fail_after=1))
    f5 = []
    try:
        breeze._synth_f5 = lambda text, out: f5.append(text)
        breeze.speak("Half of this was heard.", block=True)
    finally:
        srv.close()
    assert f5 == []
    assert breeze.cache.stats()["files"] == 0
    assert breeze.engine == "breeze"          # a blip does not retire it


def test_a_stream_with_no_receipt_counts_as_truncated(breeze, sock_path):
    """A sidecar that died mid-chunk answers nothing at all, so 'the socket
    reached EOF' cannot mean 'the sentence finished' -- on AF_UNIX an
    abortive close is indistinguishable from a clean one."""
    audio = wav_bytes(0.3)[44:]

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            srv.send(conn, {"ok": False, "error": "unknown request id"})
            return
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None) + audio)
        conn.shutdown(socket.SHUT_WR)      # looks exactly like success

    srv = FakeSidecar(sock_path, handler)
    try:
        breeze.speak("Was that the whole thing?", block=True)
    finally:
        srv.close()
    assert breeze.cache.stats()["files"] == 0


def test_a_sidecar_that_vanishes_mid_stream_does_not_wedge_the_worker(
        breeze, sock_path):
    audio = wav_bytes(0.3)[44:]
    holder = {}

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            srv.send(conn, {"ok": False, "error": "gone"})
            return
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None) + audio[:1000])
        holder["srv"].close()             # the socket file disappears
        conn.close()

    srv = holder["srv"] = FakeSidecar(sock_path, handler)
    try:
        done = breeze.speak("The sidecar is about to die.")
        assert done.wait(20), "the TTS worker wedged"
    finally:
        srv.close()
    assert breeze.cache.stats()["files"] == 0


# --------------------------------------------- the SAME failures, mid-reply
#
# Every fallback above is proved on a ONE-chunk reply, where "fall back" and
# "the reply" are the same event. His replies are not one chunk: the briefing
# is four sentences, and the interesting failures are the ones that land on
# chunk 2 of 4 -- because that is where "fall back to F5 cleanly" and "never
# say a fragment twice" can contradict each other. A fallback that re-renders
# the whole reply says chunk 1 twice; a fallback that gives up says the rest
# not at all. These are the mixed cases, which is the shape Hunter's ship
# condition is actually about.
def _two_chunk_handler(first_block, second_block, *, refuse_nth=None,
                       cut_nth=None):
    """Stream two sentences; refuse (before audio) or cut (after audio) the
    nth of them, 1-based."""
    seen = []

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            got = srv.receipts.get(req["status"])
            srv.send(conn, {"ok": True, **got} if got else
                     {"ok": False, "error": "unknown request id"})
            return
        seen.append(req["text"])
        nth = len(seen)
        rid = req.get("id")
        if nth == refuse_nth:
            srv.receipts[rid] = {"complete": False, "error": "not ready"}
            srv.send(conn, {"ok": False, "error": "not ready"})
            return
        block = first_block if nth == 1 else second_block
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None))
        if nth == cut_nth:
            conn.sendall(block[:len(block) // 2])
            srv.receipts[rid] = {"complete": False, "error": "cuda blew up"}
            return                              # abortive: no receipt, no rest
        conn.sendall(block)
        srv.receipts[rid] = {"complete": True}
        conn.shutdown(socket.SHUT_WR)

    return handler, seen


TWO = ("The reactor is holding steady this evening. "
       "The workshop is quiet and the coffee is fresh.")
TWO_CHUNKS = ["The reactor is holding steady this evening.",
              "The workshop is quiet and the coffee is fresh."]


def test_a_refusal_on_the_second_chunk_does_not_re_speak_the_first(breeze,
                                                                   sock_path):
    """Breeze says sentence 1, then refuses sentence 2 before any audio.

    Only sentence 2 may reach F5. The bug this forecloses is a fallback that
    restarts the REPLY rather than the chunk, which would say sentence 1
    twice -- once in the good voice, once in the 2.79 one."""
    first, second = wav_bytes(0.2)[44:], wav_bytes(0.3)[44:]
    handler, seen = _two_chunk_handler(first, second, refuse_nth=2)
    srv = FakeSidecar(sock_path, handler)
    f5 = []
    try:
        breeze._synth_f5 = lambda text, out: (
            f5.append(text), open(out, "wb").write(wav_bytes(0.3)))
        breeze.speak(TWO, block=True)
    finally:
        srv.close()
    assert seen == TWO_CHUNKS                   # breeze was asked for both
    assert f5 == [TWO_CHUNKS[1]]                # F5 rendered ONLY the second
    assert len(FakeProc.spawned) == 2           # two players, one per sentence
    assert bytes(FakeProc.spawned[0].fed).endswith(first)
    # each sentence filed under the engine that RENDERED it
    assert breeze.cache.get(breeze._cache_key("breeze", TWO_CHUNKS[0])) is not None
    assert breeze.cache.get(breeze._cache_key("f5", TWO_CHUNKS[1])) is not None
    assert breeze.cache.get(breeze._cache_key("breeze", TWO_CHUNKS[1])) is None
    assert breeze.engine == "breeze"            # one bad chunk does not retire it


def test_a_cut_first_chunk_is_not_re_spoken_and_the_reply_still_finishes(
        breeze, sock_path):
    """Sentence 1 is cut mid-stream after part of it was HEARD; sentence 2 is
    fine.

    Two things have to hold at once and they pull opposite ways: the heard
    fragment must NOT be re-rendered on F5 (that is the doubled half
    sentence), and the reply must not stop there either -- a dropped
    sentence 2 is a hole. So: no F5 at all, both players, and only the whole
    sentence is cached."""
    first, second = wav_bytes(0.4)[44:], wav_bytes(0.3)[44:]
    handler, seen = _two_chunk_handler(first, second, cut_nth=1)
    srv = FakeSidecar(sock_path, handler)
    f5 = []
    try:
        breeze._synth_f5 = lambda text, out: (
            f5.append(text), open(out, "wb").write(wav_bytes(0.3)))
        done = breeze.speak(TWO)
        assert done.wait(20), "the TTS worker wedged on the cut chunk"
    finally:
        srv.close()
    assert seen == TWO_CHUNKS
    assert f5 == [], "the heard half sentence must never be re-rendered"
    assert len(FakeProc.spawned) == 2, "sentence 2 was dropped"
    assert bytes(FakeProc.spawned[0].fed).endswith(first[:len(first) // 2])
    assert bytes(FakeProc.spawned[1].fed).endswith(second)
    # the truncated one is not cached; the whole one is
    assert breeze.cache.get(breeze._cache_key("breeze", TWO_CHUNKS[0])) is None
    assert breeze.cache.get(breeze._cache_key("breeze", TWO_CHUNKS[1])) is not None


def test_a_wedged_sidecar_mid_chunk_is_not_re_spoken_either(breeze, sock_path,
                                                            monkeypatch):
    """The same "never say it twice" guard, reached by the OTHER door.

    _stream_breeze has two ways out after audio has been heard: the stream
    ENDS and the completion receipt says it was truncated (the test above),
    or the read itself RAISES. Only the first was covered, and the second is
    the one with the guard in it -- ``if stream.heard: return True`` -- which
    means the branch that actually stops a doubled half sentence had never
    been executed. Verified by mutation: deleting that guard left the whole
    suite green.

    Getting there needs a raise, and on AF_UNIX a peer that dies cannot
    supply one: the module docstring's own measurement is that even SO_LINGER
    0 gives a plain EOF here, which is why the receipt exists at all. What
    does raise is the READ DEADLINE, and that is the realistic shape anyway
    -- a sidecar wedged mid-render (a GPU that stopped answering) rather than
    one that exited. The deadline is dropped to 1 s so the test is not 60."""
    monkeypatch.setattr(tts_mod, "BREEZE_TIMEOUT_S", 1.0)
    first, second = wav_bytes(0.4)[44:], wav_bytes(0.3)[44:]
    seen = []
    wedged = threading.Event()
    stuck_id = {}

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            # The wedged chunk has no receipt to give; the whole one does.
            srv.send(conn, {"ok": False, "error": "still rendering"}
                     if req["status"] == stuck_id.get("id") else
                     {"ok": True, "complete": True})
            return
        seen.append(req["text"])
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        if len(seen) == 1:
            stuck_id["id"] = req.get("id")
            conn.sendall(tts_mod.wav_header(24000, 1, 2, None)
                         + first[:len(first) // 2])
            wedged.wait(15)          # the GPU stopped answering: no EOF, no more
            return
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None) + second)
        conn.shutdown(socket.SHUT_WR)

    srv = FakeSidecar(sock_path, handler, threaded=True)
    f5 = []
    try:
        breeze._synth_f5 = lambda text, out: (
            f5.append(text), open(out, "wb").write(wav_bytes(0.3)))
        done = breeze.speak(TWO)
        assert done.wait(30), "the TTS worker wedged with the sidecar"
    finally:
        wedged.set()
        srv.close()
    assert seen == TWO_CHUNKS
    assert f5 == [], "the heard half sentence must never be re-rendered"
    assert len(FakeProc.spawned) == 2, "sentence 2 was dropped"
    assert bytes(FakeProc.spawned[0].fed).endswith(first[:len(first) // 2])
    assert breeze.cache.get(breeze._cache_key("breeze", TWO_CHUNKS[0])) is None
    assert breeze.cache.get(breeze._cache_key("breeze", TWO_CHUNKS[1])) is not None


def test_the_sidecar_dying_between_two_sentences_finishes_the_reply_on_f5(
        breeze, sock_path):
    """The whole sidecar goes away after sentence 1 -- the socket file with
    it. Sentence 2 has to be spoken by F5, not dropped, and sentence 1 must
    not be repeated."""
    first = wav_bytes(0.2)[44:]      # sentence 2 is never rendered by breeze
    holder = {}
    seen = []

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            srv.send(conn, {"ok": True, "complete": True})
            return
        seen.append(req["text"])
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None) + first)
        conn.shutdown(socket.SHUT_WR)
        holder["srv"].close()               # gone before sentence 2 is asked

    srv = holder["srv"] = FakeSidecar(sock_path, handler)
    f5 = []
    try:
        breeze._synth_f5 = lambda text, out: (
            f5.append(text), open(out, "wb").write(wav_bytes(0.3)))
        done = breeze.speak(TWO)
        assert done.wait(20), "the TTS worker wedged when the sidecar vanished"
    finally:
        srv.close()
    assert seen == [TWO_CHUNKS[0]]
    assert f5 == [TWO_CHUNKS[1]], "the rest of the reply was dropped"
    assert len(FakeProc.spawned) == 2
    assert bytes(FakeProc.spawned[0].fed).endswith(first)   # said once, whole
    assert breeze.cache.get(breeze._cache_key("f5", TWO_CHUNKS[1])) is not None


def test_the_phone_falls_back_to_f5_when_the_sidecar_is_gone(tmp_path,
                                                             monkeypatch):
    """The OTHER renderer. Everything above tests the room.

    Rendition is built from render_engine(), which answers "breeze" without
    probing the socket -- deliberately, because it must not cost a phone a
    round-trip for a line already on disk. So the plan is keyed on breeze
    before anyone knows the sidecar is down, and the fallback has to happen
    somewhere else: stream() calls load() on a miss, load() demotes to F5,
    and _replan() re-keys the whole plan. Without that last step the phone
    would look up breeze keys, miss every one, and render F5 audio into
    them."""
    monkeypatch.setattr(tts_mod, "BREEZE_SOCK", tmp_path / "nothing.sock")
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: False)
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda *a, **k: True)
    monkeypatch.setattr(tts_mod.TTS, "warm_f5_fallback", lambda self: None)
    t = TTS(engine="breeze", cache_dir=tmp_path / "cache")
    f5 = []
    monkeypatch.setattr(t, "_synth_f5", lambda text, out: (
        f5.append(text), open(out, "wb").write(wav_bytes(0.2))))
    monkeypatch.setattr(t, "_synth_breeze", lambda text, out: (
        _ for _ in ()).throw(AssertionError("breeze must not be reached")))

    r = tts_mod.Rendition(t, "Good evening, sir.")
    assert r.engine == "breeze"          # keyed on his choice, not on a probe
    body = r.body()
    assert r.engine == "f5", "the plan was never re-keyed"
    assert f5 == ["Good evening, sir."]
    assert len(body) > tts_mod._WAV_HEADER_BYTES
    # filed under F5, so the room does not later replay it as Breeze
    assert t.cache.get(t._cache_key("f5", "Good evening, sir.")) is not None
    assert t.cache.get(t._cache_key("breeze", "Good evening, sir.")) is None


def test_a_repeat_plays_from_cache_without_touching_the_socket(breeze, sock_path,
                                                               monkeypatch):
    srv = FakeSidecar(sock_path, sidecar_handler([wav_bytes(0.1)[44:]]))
    played = []
    try:
        monkeypatch.setattr(breeze, "_play", lambda p: played.append(p))
        monkeypatch.setattr(breeze, "_start_amp_feeder", lambda p: None)
        breeze.speak("Always, sir.", block=True)
        n = len([r for r in srv.requests if r.get("stream")])
        breeze.speak("Always, sir.", block=True)
        assert len([r for r in srv.requests if r.get("stream")]) == n == 1
    finally:
        srv.close()
    assert len(played) == 1 and "cache" in played[0]


def test_the_whole_chunk_path_still_works_when_streaming_is_off(
        breeze, sock_path, monkeypatch):
    """Not a shipping configuration -- 2.12 s to first audio at his median,
    a 4x regression -- but it is how a listener compares the two paths."""
    monkeypatch.setattr(tts_mod, "BREEZE_STREAM_PLAYBACK", False)
    monkeypatch.setattr(breeze, "_start_amp_feeder", lambda p: None)
    played = []
    monkeypatch.setattr(breeze, "_play", lambda p: played.append(p))
    srv = FakeSidecar(sock_path, sidecar_handler([pcm(2400)]))
    try:
        breeze.speak("Good evening, sir.", block=True)
    finally:
        srv.close()
    assert len(played) == 1
    assert [r for r in srv.requests if r.get("out")]      # the file mode
    assert breeze.cache.stats()["files"] == 1


def test_a_whole_chunk_failure_also_falls_back_to_f5(breeze, sock_path,
                                                     monkeypatch):
    monkeypatch.setattr(tts_mod, "BREEZE_STREAM_PLAYBACK", False)
    monkeypatch.setattr(breeze, "_start_amp_feeder", lambda p: None)
    monkeypatch.setattr(breeze, "_play", lambda p: None)
    f5 = []
    srv = FakeSidecar(sock_path, sidecar_handler([], refuse="boom"))
    try:
        breeze._synth_f5 = lambda text, out: (f5.append(text),
                                              open(out, "wb").write(wav_bytes(0.1)))
        breeze.speak("Good evening, sir.", block=True)
    finally:
        srv.close()
    assert f5 == ["Good evening, sir."]


def test_the_gain_jarvis_keys_on_is_the_gain_the_sidecar_is_told(breeze,
                                                                 sock_path):
    srv = FakeSidecar(sock_path, sidecar_handler([wav_bytes(0.1)[44:]]))
    try:
        breeze.speak("Good evening, sir.", block=True)
    finally:
        srv.close()
    streamed = [r for r in srv.requests if r.get("stream")]
    assert streamed and streamed[0]["gain"] == tts_mod.output_gain_for("breeze")


# ===========================================================================
# 7. the speech cache
# ===========================================================================
def _key(voice, text="Always, sir."):
    return voice._cache_key("breeze", text)


def test_the_breeze_key_carries_everything_that_changes_the_audio(monkeypatch):
    t = TTS.__new__(TTS)
    t.cache = None
    base = _key(t)
    for attr, value in (("cfg_scale", 2.0), ("seed", 7), ("temperature", 0.7),
                        ("repetition_penalty", 1.5), ("max_new_tokens", 900),
                        ("template", "tts_instruction"), ("sr", 16000),
                        ("text_encoder_attn", "eager"),
                        ("instruction", "Speak like a pirate.")):
        params = dict(tts_mod.BREEZE_PARAMS, **{attr: value})
        monkeypatch.setattr(tts_mod, "BREEZE_PARAMS", params)
        assert _key(t) != base, f"{attr} is not in the key"
        monkeypatch.undo()


def test_the_breeze_key_changes_with_the_gain(monkeypatch):
    t = TTS.__new__(TTS)
    t.cache = None
    base = _key(t)
    monkeypatch.setitem(tts_mod.ENGINE_OUTPUT_GAIN, "breeze", 1.9)
    assert _key(t) != base


def test_the_breeze_key_changes_with_the_checkpoint_and_the_reference(
        monkeypatch, tmp_path):
    """int4, group-32 depth and bf16 attention are baked into the export, so
    the manifest's identity IS the quantization; the clip and its transcript
    are the voice."""
    t = TTS.__new__(TTS)
    t.cache = None
    base = _key(t)
    for attr, name in (("BREEZE_CKPT_MANIFEST", "quant_manifest.json"),
                       ("BREEZE_REF", "ref.wav"),
                       ("BREEZE_REF_TEXT", "ref.txt")):
        path = tmp_path / name
        path.write_text("x" * 11)
        monkeypatch.setattr(tts_mod, attr, path)
        assert _key(t) != base, f"{attr} is not in the key"
        monkeypatch.undo()


def test_the_breeze_key_is_not_the_f5_key_for_the_same_line():
    """They share a reference clip, so nothing but the engine name keeps
    F5 audio from replaying as Breeze."""
    t = TTS.__new__(TTS)
    t.cache = None
    assert t._cache_key("breeze", "Always, sir.") != \
        t._cache_key("f5", "Always, sir.")


# ===========================================================================
# 8. OFF BY DEFAULT: the F5 path is what it was
# ===========================================================================
def test_nothing_selects_breeze_on_its_own():
    """It ships opt-in. The engine is only ever "breeze" because
    voice_settings.json says so."""
    from dataclasses import fields
    default = {f.name: f.default for f in fields(type(CONFIG))}["tts_engine"]
    assert default == "edge"
    assert tts_mod.FISH_FALLBACK == "f5"          # fish still retires to F5
    assert tts_mod.BREEZE_FALLBACK == "f5"


# --------------------------------------------- the half-registered engine
#
# tts.py dispatches on the engine NAME in six places and five of them have a
# DEFAULT, so an unregistered name does not raise -- it silently renders on
# XTTS or on cloud Edge and then files the result under the new engine's
# cache key, where it stays. The sixth (ENGINE_OUTPUT_GAIN) defaults to 1.0,
# and that omission has already shipped once: F5 went out at 1.0 instead of
# 2.8 and was an audible ~9 dB regression until somebody noticed by ear.
#
# So this is a CHECKLIST, deliberately written as data. Adding a name to
# _ENGINES fails the first assertion until every site below is filled in,
# which is the only way a default that cannot raise gets a chance to.
_REGISTERED = {
    # engine    pipelined?  its own synth in the two dispatch dicts?  gain
    "edge":    dict(pipelined=False, synth=False, gain=1.0),
    "xtts":    dict(pipelined=True, synth=False, gain=1.34),   # the .get default
    "f5":      dict(pipelined=True, synth=True, gain=2.8),
    "fish":    dict(pipelined=True, synth=True, gain=1.0),
    "breeze":  dict(pipelined=True, synth=True, gain=2.8),
}


def _recording_tts(tmp_path, monkeypatch, engine):
    """A TTS whose four synths only record which of them was called.

    Rendered BEHAVIOURALLY rather than by reading the dispatch dict's source:
    a substring check passes on a half-registered engine whose name merely
    appears somewhere else in the same function, which is how this test first
    failed to catch its own mutant."""
    monkeypatch.setattr(tts_mod, "FISH_STREAM_PLAYBACK", False)
    monkeypatch.setattr(tts_mod, "BREEZE_STREAM_PLAYBACK", False)
    monkeypatch.setattr(tts_mod.subprocess, "Popen", FakeProc)
    t = TTS(engine="edge", cache=False)
    t._engine = engine
    called = []

    def _rec(name):
        def _synth(text, out):
            called.append(name)
            with open(out, "wb") as fh:
                fh.write(wav_bytes(0.05))
        return _synth

    for name in ("f5", "fish", "breeze", "xtts", "edge"):
        monkeypatch.setattr(t, f"_synth_{name}", _rec(name))
    monkeypatch.setattr(t, "_play", lambda p: None)
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    return t, called


@pytest.mark.parametrize("engine", [e for e, w in _REGISTERED.items()
                                    if w["synth"]])
def test_the_pipelined_synth_dispatch_reaches_that_engines_own_synth(
        engine, tmp_path, monkeypatch):
    """_speak_pipelined's dict falls back to _synth_xtts, so an unregistered
    name renders in the WRONG VOICE and is then cached under its own key."""
    t, called = _recording_tts(tmp_path, monkeypatch, engine)
    t._speak_pipelined("Always, sir.", engine)
    assert called == [engine], f"{engine} dispatched to {called}"


@pytest.mark.parametrize("engine", [e for e, w in _REGISTERED.items()
                                    if w["synth"]])
def test_the_rendition_synth_dispatch_reaches_that_engines_own_synth(
        engine, tmp_path, monkeypatch):
    """The same default, on the path the phone takes."""
    t, called = _recording_tts(tmp_path, monkeypatch, engine)
    monkeypatch.setattr(tts_mod.TTS, "render_engine", lambda self: engine)
    r = tts_mod.Rendition(t, "Always, sir.")
    assert r.engine == engine
    path = r._synth("Always, sir.")
    try:
        assert called == [engine], f"{engine} dispatched to {called}"
    finally:
        if path:
            os.unlink(path)


def test_every_engine_is_registered_at_every_dispatch_site():
    """The remaining sites, per engine, from the outside."""
    import inspect
    from jarvis import pronounce
    from jarvis.ui.main_window import MainWindow

    assert set(tts_mod._ENGINES) == set(_REGISTERED), (
        "an engine was added to _ENGINES. Register it in EVERY site this "
        "test checks before adding it here, or it will render as XTTS or "
        "Edge and be cached under its own name.")

    speak_src = inspect.getsource(tts_mod.TTS._speak_sync)
    t = TTS.__new__(TTS)
    t.cache = None

    for engine, want in _REGISTERED.items():
        # 1. the pronunciation rules: an explicit row, never rules_for()'s
        #    everything-on default, which is a guess about an unheard engine
        assert engine in pronounce.ENGINE_RULES, engine
        # 2. the output gain. A missing row is 1.0 and is SILENT.
        assert tts_mod.output_gain_for(engine) == want["gain"], engine
        # 3. the short-line pad
        assert engine in tts_mod._ENGINE_NEEDS_SHORT_PAD, engine
        # 4. the HUD, or the card names a voice that is not speaking
        assert MainWindow._tts_desc(engine) == MainWindow._TTS_DESC[engine]
        # 5. the pipelined arm of _speak_sync -- the branch that decides
        #    whether this engine reaches _speak_pipelined at all
        if want["pipelined"]:
            assert f'"{engine}"' in speak_src, engine

    # 6. a namespace of its own, pairwise: two engines sharing a key would
    #    replay one voice as the other from cache
    keys = {e: t._cache_key(e, "Always, sir.") for e in _REGISTERED}
    assert len(set(keys.values())) == len(keys), keys


def test_the_breeze_gain_matches_f5s_because_they_share_a_reference_clip():
    """2.8, and it must be PINNED rather than inherited from a .get default.

    Breeze clones jarvis_voice_ref_f5.wav -- the same quiet clip (peak 0.303)
    F5 does -- so it comes out at the same level, and 2.8 is the value fitted
    to 400 F5 renders. Matching F5 exactly is also what the A/B needs: the
    cutover changes the voice and not the volume. Dropping the row is not an
    error, it is 1.0, which is the ~9 dB regression F5 already shipped."""
    assert tts_mod.ENGINE_OUTPUT_GAIN["breeze"] == \
        tts_mod.ENGINE_OUTPUT_GAIN["f5"] == 2.8


def test_the_f5_cache_key_formula_is_unchanged():
    """Recomputed from first principles rather than compared to itself: the
    f5 namespace must still be engine + ref stat + gain + nfe/speed and
    nothing else, or every render already on his disk goes stale."""
    from jarvis.speech_cache import SpeechCache
    t = TTS.__new__(TTS)
    t.cache = None
    try:
        st = tts_mod.F5_REF.stat()
        ref = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        ref = "none"
    for line in ("Always, sir.", "Good evening, sir.", "Noted, sir."):
        assert t._cache_key("f5", line) == SpeechCache.key(
            "f5", line, ref=ref, gain=2.8, nfe_step=10, speed=0.85)


def test_the_f5_engine_tables_are_unchanged():
    assert tts_mod.F5_PARAMS == {"nfe_step": 10, "speed": 0.85}
    assert tts_mod.output_gain_for("f5") == 2.8
    assert tts_mod.output_gain_for("xtts") == 1.34
    assert tts_mod.output_gain_for("edge") == 1.0
    assert tts_mod.output_gain_for("fish") == 1.0
    for engine, value in {"edge": False, "xtts": False, "f5": True,
                          "fish": False}.items():
        assert tts_mod._ENGINE_NEEDS_SHORT_PAD[engine] is value


def test_the_other_engines_pronunciation_rules_are_unchanged():
    """The two booleans this used to read (_ENGINE_NEEDS_TIME_REWRITE and
    _ENGINE_NEEDS_UNSHOUT, in tts.py) became one row per engine in
    pronounce.ENGINE_RULES while this branch was in flight. The question is
    the same and so is the answer for every engine that already shipped: the
    three clock/number rules ride together where the old rewrite_times flag
    put them, and unshout is on everywhere. Asserted on the ROWS rather than
    on apply(), because a row is what the old dicts were."""
    from jarvis import pronounce
    for engine in ("edge", "xtts", "f5"):
        rules = pronounce.rules_for(engine)
        assert (rules.marked_times, rules.bare_times, rules.id_digits) == \
            (True, True, True), engine
        assert rules.unshout is True, engine
    fish = pronounce.rules_for("fish")
    assert (fish.marked_times, fish.bare_times, fish.id_digits) == \
        (False, False, False)
    assert fish.unshout is True


def test_the_f5_chunking_and_split_are_unchanged():
    t = TTS.__new__(TTS)
    t.cache = None
    assert t._chunk_limits("f5") == (240, 6.0)
    assert t._chunk_limits("edge") == (160, 2.5)
    assert t._chunk_limits("xtts") == (160, 2.5)
    line = ("Your briefing for today, sir. It is overcast and quite warm, "
            "with a high of ninety-seven. On the calendar you have Biosensors "
            "at nine ten ay em, Magnetic Resonance Engineering at twelve "
            "forty pee em, and at four ten pee em your Electrical Design Lab "
            "presentation. The inbox is quiet, for once.")
    assert t._split_sentences(line, engine="f5") == [
        "Your briefing for today, sir.",
        "It is overcast and quite warm, with a high of ninety-seven.",
        "On the calendar you have Biosensors at nine ten ay em, Magnetic "
        "Resonance Engineering at twelve forty pee em, and at four ten pee em "
        "your Electrical Design Lab presentation.",
        "The inbox is quiet, for once."]


def test_the_fish_stream_deadlines_are_unchanged():
    """The three shared consumer deadlines now travel on the stream so a
    Breeze chunk can have its own. Fish's must still be exactly what they
    were: FISH_TIMEOUT_S."""
    assert tts_mod._AudioStream("x").timeout == tts_mod.FISH_TIMEOUT_S == 10.0
    assert tts_mod._AudioStream("x", timeout=60.0).timeout == 60.0


def test_an_f5_reply_never_touches_the_breeze_code(tmp_path, monkeypatch):
    """The whole feature is inert with the flag unset -- no ping, no socket,
    no systemd query, no import."""
    for name in ("_breeze_request", "_breeze_iter", "_ensure_breeze_server",
                 "_breeze_unit_active", "_breeze_alive"):
        monkeypatch.setattr(tts_mod, name,
                            lambda *a, **k: pytest.fail(f"{name} was called"))
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda *a, **k: True)
    monkeypatch.setattr(tts_mod.subprocess, "Popen", FakeProc)
    t = TTS(engine="f5", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(t, "_start_amp_feeder", lambda p: None)
    monkeypatch.setattr(t, "_synth_f5",
                        lambda text, out: open(out, "wb").write(wav_bytes(0.05)))
    t.speak("Good evening, sir.", block=True)
    assert t.engine == "f5"
    assert t.cache.get(t._cache_key("f5", "Good evening, sir.")) is not None


# ===========================================================================
# 8. a peer must not be able to kill a 13.5 GB resident model
#
# Every case below was reproduced against the REAL serve() loop over a REAL
# AF_UNIX socket: no reply, the process dead, every later connect
# ECONNREFUSED. With Restart=always/RestartSec=30 each kill re-entered the
# 22 s CUDA-graph capture and its ~18.3 GB transient, and three of them left
# the unit failed until someone ran `systemctl --user reset-failed`. Jarvis
# itself only ever sends well-formed dicts, but the enable procedure printed
# by setup_breeze_service.sh asks Hunter to hand-poke this socket with an
# escaped-JSON one-liner.
# ===========================================================================
def start_real_server(path, service):
    """bs.serve() itself, over a real AF_UNIX socket, on a daemon thread.

    The accept loop is a systemd unit's main loop and has no shutdown hook, so
    the thread is left blocked in accept() when the test ends -- daemon, so it
    dies with the interpreter and costs nothing meanwhile.
    """
    out = {}

    def _run():
        out["rc"] = bs.serve(str(path), service)

    threading.Thread(target=_run, daemon=True).start()
    # Not "the path exists": a stale socket FILE may already be sitting there,
    # which is the very thing serve() has to see past.
    assert wait_until(lambda: bs.socket_owner(path) is not None or "rc" in out), \
        "serve() never bound the socket"
    return out


def ask(path, raw: bytes, timeout: float = 5.0) -> bytes:
    """Send raw bytes to a server and read back everything it writes."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(str(path))
    try:
        sock.sendall(raw)
        out = b""
        while True:
            try:
                part = sock.recv(65536)
            except (OSError, socket.timeout):
                break
            if not part:
                break
            out += part
        return out
    finally:
        sock.close()


KILLERS = [
    (b"5\n", "a bare int"),
    (b"[]\n", "an empty list"),
    (b'"x"\n', "a bare string"),
    (b"true\n", "a bare bool"),
    (b"[1, 2, 3]\n", "a list body"),
    (b'{"text": 5, "stream": true}\n', "a non-string text"),
    (b'{"text": "hi", "gain": "loud"}\n', "a string gain"),
    (b'{"text": "hi", "gain": null}\n', "a null gain"),
    (b'{"text": "hi", "gain": [1]}\n', "a list gain"),
]


@pytest.mark.parametrize("raw,what", KILLERS, ids=[k[1] for k in KILLERS])
def test_a_malformed_request_is_refused_and_the_sidecar_survives(
        tmp_path, raw, what):
    path = tmp_path / "breeze.sock"
    rendered = []

    def render(text, gain=1.0):
        rendered.append(text)
        return iter([pcm(10)])

    out = start_real_server(path, ready_service(render))
    head, rest = split_reply(ask(path, raw))
    assert head.get("ok") is False, (what, head)
    assert rest == b""                       # not one byte of audio
    assert rendered == []                    # it never reached the model
    pong, _ = split_reply(ask(path, b'{"ping": true}\n'))
    assert pong["ok"] is True and pong["ready"] is True, what
    assert "rc" not in out, "serve() returned -- the process would be dead"


def test_a_non_object_request_is_named_as_such():
    svc = ready_service(blocks_render(pcm(10)))
    assert split_reply(serve_once(svc, [1, 2, 3]))[0] == {
        "ok": False, "error": "request must be a JSON object"}
    assert split_reply(serve_once(svc, 5))[0]["error"].startswith("request must")
    # ...and it still answers afterwards
    assert split_reply(serve_once(svc, {"ping": True}))[0]["ok"] is True


def test_a_wrong_typed_gain_is_refused_rather_than_coerced():
    """float("2.8") used to work and float("loud") used to kill the process.
    The client always sends a float; anything else is a mistake, and bools are
    checked first because bool is an int subclass."""
    svc = ready_service(blocks_render(pcm(10)))
    for gain in ("loud", None, [1], {"x": 1}, True):
        head, rest = split_reply(
            serve_once(svc, {"text": "hi", "stream": True, "gain": gain}))
        assert head == {"ok": False, "error": "gain must be a number"}, gain
        assert rest == b""
    # the shipped client's own value still renders
    head, rest = split_reply(
        serve_once(svc, {"text": "hi", "stream": True, "gain": 2.8}))
    assert head["ok"] is True and rest


def test_a_wrong_typed_out_path_is_refused_not_silently_written(tmp_path,
                                                                monkeypatch):
    """wave.open(str(out)) turned {"out": 5} into {"ok": true} plus a file
    literally named `5` in the server's cwd -- which load_engine has chdir'd
    to the Breeze source tree."""
    monkeypatch.chdir(tmp_path)
    svc = ready_service(blocks_render(pcm(10)))
    head, _ = split_reply(serve_once(svc, {"text": "hi", "out": 5}))
    assert head == {"ok": False, "error": "no out path"}
    assert list(tmp_path.iterdir()) == []


# ===========================================================================
# 9. one instance, one socket
# ===========================================================================
def test_a_second_instance_refuses_instead_of_stealing_the_socket(tmp_path):
    """serve() used to unlink and rebind unconditionally, so a second process
    took the socket and left the first alive, resident and unreachable -- no
    pidfile, and no way to find it that this box's rules permit (matching a
    process by cmdline text is what caused five self-kills)."""
    path = tmp_path / "breeze.sock"
    start_real_server(path, ready_service(blocks_render(pcm(10))))
    second = bs.BreezeService(render=blocks_render(pcm(10)), ready=True,
                              graphs=ALL_GRAPHS, config={"cfg_scale": 99.0})
    assert bs.serve(str(path), second) == 2
    pong, _ = split_reply(ask(path, b'{"ping": true}\n'))
    assert pong["config"] == {"cfg_scale": 4.0}       # still the FIRST one


def test_a_stale_socket_file_is_not_mistaken_for_a_live_sidecar(tmp_path):
    """/tmp is wiped at boot and an unclean exit leaves the file behind, so
    'the path exists' must not be the test -- connect() is."""
    path = tmp_path / "breeze.sock"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))
    dead.close()
    assert path.exists()
    assert bs.socket_owner(path) is None
    assert bs.socket_owner(tmp_path / "never-existed") is None
    start_real_server(path, ready_service(blocks_render(pcm(10))))
    assert split_reply(ask(path, b'{"ping": true}\n'))[0]["ok"] is True


# ===========================================================================
# 10. the memory gate -- the one line between this sidecar and a repeat of
#     the 2026-08-28 unified-memory power-off
# ===========================================================================
def gate_argv(tmp_path, socket_path=None):
    (tmp_path / "ckpt").mkdir(exist_ok=True)
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "ref.wav").write_bytes(b"RIFF")
    (tmp_path / "ref.txt").write_text("a reference line")
    return ["--socket", str(socket_path or tmp_path / "breeze.sock"),
            "--model", str(tmp_path / "ckpt"),
            "--repo", str(tmp_path / "repo"),
            "--ref", str(tmp_path / "ref.wav"),
            "--ref-text", str(tmp_path / "ref.txt"),
            "--gpu-lock", str(tmp_path / "gpu.lock")]


def flock_taken(path) -> bool:
    """True when an exclusive flock on ``path`` is already held. flock locks
    are per open-file-description, so a second open() in this process sees a
    lock this process holds elsewhere."""
    fh = open(path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        fh.close()


def _fake_meminfo(monkeypatch, free, avail):
    monkeypatch.setattr(bs, "free_mem_gb", lambda *a, **k: free)
    monkeypatch.setattr(bs, "mem_gb",
                        lambda field="MemFree", *a, **k:
                        avail if field == "MemAvailable" else free)


@pytest.mark.parametrize("free,avail,why", [
    (30.0, 73.0, "under the MemFree floor"),
    (40.0, 55.0, "under the MemAvailable floor"),
    (28.4, 71.7, "the box's own reading on 2026-09-02"),
])
def test_the_memory_gate_refuses_before_load_engine_is_called(
        tmp_path, monkeypatch, free, avail, why):
    """The gate worked, but nothing held it in place: a refactor that moved
    the check after load_engine, or turned the `or` into an `and`, passed the
    whole suite. The consequence is a 13.5 GB model plus an 18.2 GB capture
    transient on a box whose GPU memory IS its system memory."""
    loaded = []
    _fake_meminfo(monkeypatch, free, avail)
    monkeypatch.setattr(bs, "load_engine", lambda args: loaded.append(1))
    monkeypatch.setattr(bs, "serve", lambda *a, **k: pytest.fail("it served"))
    assert bs.main(gate_argv(tmp_path)) == 2, why
    assert loaded == [], why


def test_enough_free_pages_and_it_loads_then_serves(tmp_path, monkeypatch):
    _fake_meminfo(monkeypatch, 40.0, 73.0)
    order = []

    def _load(args):
        order.append("load")
        return blocks_render(pcm(10)), ALL_GRAPHS, None, {}

    monkeypatch.setattr(bs, "load_engine", _load)
    monkeypatch.setattr(bs, "serve",
                        lambda path, svc: order.append(("serve", svc.ready)) or 0)
    assert bs.main(gate_argv(tmp_path)) == 0
    assert order == ["load", ("serve", True)]


def test_the_gate_runs_under_the_gpu_flock_and_frees_it_on_the_refusal(
        tmp_path, monkeypatch):
    """The flock has to cover the READING as well as the load, or two starts
    can both read a comfortable MemFree and then both load."""
    argv = gate_argv(tmp_path)
    lock = tmp_path / "gpu.lock"
    seen = []
    monkeypatch.setattr(bs, "free_mem_gb",
                        lambda *a, **k: (seen.append(flock_taken(lock)), 30.0)[1])
    monkeypatch.setattr(bs, "mem_gb", lambda *a, **k: 73.0)
    monkeypatch.setattr(bs, "load_engine", lambda args: pytest.fail("loaded"))
    assert bs.main(argv) == 2
    assert seen == [True]                    # the check ran holding the lock
    assert flock_taken(lock) is False        # ...and it was released


def test_a_missing_path_is_refused_before_anything_is_taken(tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(bs, "load_engine", lambda args: pytest.fail("loaded"))
    argv = gate_argv(tmp_path)
    argv[argv.index("--ref") + 1] = str(tmp_path / "no-such.wav")
    assert bs.main(argv) == 2


def test_main_refuses_a_second_instance_before_it_spends_the_memory(
        tmp_path, monkeypatch):
    """The flock is no defence here: main() releases it at the end of
    load_engine, so a second instance takes it freely and pays for a whole
    second resident model before serve() would ever notice."""
    path = tmp_path / "breeze.sock"
    start_real_server(path, ready_service(blocks_render(pcm(10))))
    _fake_meminfo(monkeypatch, 99.0, 99.0)
    monkeypatch.setattr(bs, "load_engine", lambda args: pytest.fail("loaded"))
    assert bs.main(gate_argv(tmp_path, socket_path=path)) == 2


# ===========================================================================
# 11. deadlines and concurrency: one stalled peer must not be the whole voice
# ===========================================================================
def test_the_read_deadline_is_short_and_the_send_one_is_f5s():
    """One 900 s number for both phases meant a peer that connected and never
    finished its request line held the entire sidecar for fifteen minutes."""
    assert bs.READ_TIMEOUT_S <= 30.0
    assert bs.SEND_TIMEOUT_S == 120.0        # scripts/f5_server.py's number
    assert not hasattr(bs, "CONN_TIMEOUT_S")


def test_a_ping_is_answered_while_a_render_is_in_flight(tmp_path):
    """load() pings on EVERY utterance and must keep doing so (a sidecar can
    die between two replies), so the ping has to be cheap even mid-render.
    Measured on the single-threaded loop: a 6 s in-flight render made
    _ensure_breeze_server cost 6.0 s on the speak path."""
    path = tmp_path / "breeze.sock"
    holding, release = threading.Event(), threading.Event()

    def slow_render(text, gain=1.0):
        holding.set()
        release.wait(20)
        yield pcm(10)

    start_real_server(path, ready_service(slow_render))
    render = threading.Thread(
        target=lambda: ask(path, b'{"text": "a long one", "stream": true}\n', 30),
        daemon=True)
    render.start()
    try:
        assert holding.wait(5)
        started = time.monotonic()
        pong, _ = split_reply(ask(path, b'{"ping": true}\n', 5))
        assert pong["ok"] is True
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
        render.join(20)


def test_a_peer_that_never_finishes_its_line_does_not_wedge_the_sidecar(
        tmp_path):
    path = tmp_path / "breeze.sock"
    start_real_server(path, ready_service(blocks_render(pcm(10))))
    stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stalled.connect(str(path))
    stalled.sendall(b'{"pi')                 # no newline, ever
    try:
        assert split_reply(ask(path, b'{"ping": true}\n', 5))[0]["ok"] is True
    finally:
        stalled.close()


def test_only_one_render_runs_at_a_time_even_with_threaded_connections(
        tmp_path):
    """Threads are for the cheap replies. There is one resident model, and two
    generations on it at once is the thing the accept loop used to prevent."""
    path = tmp_path / "breeze.sock"
    live, peak = [], []

    def render(text, gain=1.0):
        live.append(text)
        peak.append(len(live))
        time.sleep(0.2)
        yield pcm(10)
        live.remove(text)

    start_real_server(path, ready_service(render))
    askers = [threading.Thread(
        target=lambda i=i: ask(path, json.dumps(
            {"text": f"line {i}", "stream": True}).encode() + b"\n", 30),
        daemon=True) for i in range(3)]
    for t in askers:
        t.start()
    for t in askers:
        t.join(30)
    assert len(peak) == 3 and max(peak) == 1, peak


# ===========================================================================
# 11b. THE ONE RENDER THREAD
#
# WHY THIS SECTION EXISTS. Measured 2026-09-02 with an external MemFree
# sampler: a render costs a +16.37 GiB GPU transient the FIRST time it runs on
# a given host thread, and 0.0-0.3 GiB every time after that on the same
# thread. serve() gives each CONNECTION its own thread, so the render used to
# pay that on every sentence -- on the box whose exhausted unified pool forced
# the 2026-08-28 power-off. Arms, one resident model, cfg_scale 4.0
# throughout: a new thread per render 16.42/16.37/16.29/16.39 GiB; ONE
# persistent worker 16.24 on its first render then 0.04/0.08/0.02/0.00/0.04;
# the main thread over 28 renders 0.01-0.30.
#
# The cost was localised (probe3/probe4) to prepare_inputs, and inside it to
# the Mimi encode of the reference clip -- generation on that same fresh
# thread dips 0.01-0.11 GiB -- but WHY that call is per-thread is NOT KNOWN.
# So these tests pin the THREAD IDENTITY, the thing that was measured, and
# nothing about the mechanism. They also pin what the fix must not
# cost: chunks still arrive progressively (0.27 s to first audio, against the
# shipped F5 path's 0.53 s), errors still arrive before any audio, and a dead
# worker still means F5 rather than a hung connection.
# ===========================================================================
def recv_exactly(sock, n: int, timeout: float = 5.0) -> bytes:
    """Exactly ``n`` bytes, or an assertion. Used to prove that audio reached
    the client while the render was still blocked mid-utterance."""
    sock.settimeout(timeout)
    buf = b""
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            break
        buf += part
    assert len(buf) == n, f"wanted {n} bytes, got {len(buf)}"
    return buf


def test_every_render_runs_on_the_one_worker_thread(tmp_path):
    """THE measurement, as a property of the code: seven renders over seven
    connections, one thread identity. Before this, each was a fresh
    connection thread and each paid 16.37 GiB."""
    path = tmp_path / "breeze.sock"
    idents = []
    seen = threading.Lock()

    def render(text, gain=1.0):
        with seen:
            idents.append(threading.get_ident())
        yield pcm(4)

    svc = ready_service(render)
    start_real_server(path, svc)
    for i in range(4):
        ask(path, json.dumps({"text": f"one at a time {i}",
                              "stream": True}).encode() + b"\n", 30)
    askers = [threading.Thread(
        target=lambda i=i: ask(path, json.dumps(
            {"text": f"all at once {i}", "stream": True}).encode() + b"\n", 30),
        daemon=True) for i in range(3)]
    for t in askers:
        t.start()
    for t in askers:
        t.join(30)
    assert len(idents) == 7, idents
    assert len(set(idents)) == 1, f"{len(set(idents))} threads rendered: {idents}"
    assert idents[0] == svc.worker.ident
    assert idents[0] != threading.main_thread().ident


def test_a_plain_render_is_wired_through_a_worker_and_never_doubled():
    """The wrapping is the service's job, not the caller's: the defect was a
    render running on whichever thread held the connection, so a service that
    cannot be built without a worker cannot regress to it."""
    svc = ready_service(blocks_render(pcm(2)))
    assert isinstance(svc.worker, bs.RenderWorker)
    assert svc.render is svc.worker              # the seam is the worker
    already = bs.RenderWorker(blocks_render(pcm(2)))
    assert bs.BreezeService(render=already, ready=True,
                            graphs=ALL_GRAPHS).worker is already
    assert bs.BreezeService(render=None, ready=True,
                            graphs=ALL_GRAPHS).worker is None


def test_nothing_on_the_render_path_starts_a_thread_per_request():
    """The regression, named. A thread created anywhere under handle() is the
    16.37 GiB bill coming back."""
    body = (REPO / "scripts" / "breeze_server.py").read_text()
    service = body.split("class BreezeService", 1)[1].split("def read_request", 1)[0]
    assert "Thread(" not in service
    accept = body.split("def serve(", 1)[1].split("def build_parser", 1)[0]
    assert accept.count("threading.Thread") == 1     # the connection, only


def test_chunks_still_arrive_while_the_render_is_still_running(tmp_path):
    """THE LATENCY THE WORKER MUST NOT COST. Breeze reaches first audio in
    0.27 s against the shipped F5 path's 0.53 s, and that is the whole reason
    this voice is worth its memory. A queue that handed back only a finished
    render would throw it away, so the fake here REFUSES to produce its second
    chunk until the first has been read off the socket."""
    path = tmp_path / "breeze.sock"
    finish = threading.Event()

    def render(text, gain=1.0):
        yield pcm(4, 111)
        assert finish.wait(5), "the second chunk was never asked for"
        yield pcm(4, 222)

    start_real_server(path, ready_service(render))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(str(path))
    try:
        sock.sendall(b'{"text": "two chunks", "stream": true}\n')
        line = b""
        while not line.endswith(b"\n"):
            line += recv_exactly(sock, 1)
        head = json.loads(line.decode())
        assert head["ok"] is True and head["stream"] is True
        # The header and the FIRST chunk, with the render still parked inside
        # the model. If this times out the sidecar buffered the utterance.
        assert recv_exactly(sock, 44) == bs.wav_header(24000)
        assert recv_exactly(sock, 8) == pcm(4, 111)
        finish.set()
        assert recv_exactly(sock, 8) == pcm(4, 222)
    finally:
        finish.set()
        sock.close()


def test_the_worker_does_not_run_far_ahead_of_a_slow_consumer():
    """Bounded on purpose: unbounded, a client that stopped reading would let
    the GPU generate a whole utterance into RAM on a box with none to
    spare."""
    produced = []

    def render(text, gain=1.0):
        for i in range(200):
            produced.append(i)
            yield pcm(2)

    worker = bs.RenderWorker(render)
    blocks = worker("a long one")
    try:
        assert next(blocks) == pcm(2)
        time.sleep(0.3)
        assert len(produced) <= bs.RENDER_QUEUE_DEPTH + 2, len(produced)
    finally:
        blocks.close()
        worker.stop()


def test_a_consumer_that_goes_away_stops_the_generation():
    """Barge-in, or Jarvis exiting. The client's thread closes the generator;
    the GPU must not still be rendering for a room nobody is in."""
    closed = threading.Event()

    def render(text, gain=1.0):
        try:
            for _ in range(1000):
                yield pcm(2)
        finally:
            closed.set()

    worker = bs.RenderWorker(render)
    blocks = worker("a long one")
    try:
        assert next(blocks) == pcm(2)
        blocks.close()
        assert closed.wait(5), "the worker kept generating for a client that had gone"
    finally:
        worker.stop()


def test_a_render_that_raises_crosses_the_thread_as_an_error_before_audio():
    """The thread hop must not turn a refusal into half a sentence: the
    caller renders this chunk on F5 and may not hear the failed one."""
    svc = ready_service(blocks_render(pcm(10), fail_after=0))
    head, rest = split_reply(
        serve_once(svc, {"text": "x", "stream": True, "id": "r9"}))
    assert head["ok"] is False
    assert head["error"] == "RuntimeError: cuda blew up"   # message intact
    assert rest == b""
    assert svc.receipt("r9")["complete"] is False


def test_a_dead_render_worker_makes_the_sidecar_not_ready(tmp_path):
    """FAIL CLOSED. A sidecar whose render thread has gone must say so on the
    next ping -- Jarvis pings on every utterance -- so the reply is spoken in
    the 2.79 voice instead of not at all."""
    path = tmp_path / "breeze.sock"
    svc = ready_service(blocks_render(pcm(4)))
    start_real_server(path, svc)
    ask(path, b'{"text": "hello", "stream": true}\n', 30)   # start the thread
    assert svc.worker.ident is not None
    assert split_reply(ask(path, b'{"ping": true}\n'))[0]["ready"] is True

    svc.worker.stop()
    assert svc.worker.alive is False
    pong, _ = split_reply(ask(path, b'{"ping": true}\n', 5))
    assert pong["ok"] is True and pong["ready"] is False
    assert bs.WORKER_GONE in pong["error"]

    started = time.monotonic()
    head, rest = split_reply(ask(path, b'{"text": "hi", "stream": true}\n', 5))
    assert head["ok"] is False and rest == b""
    assert time.monotonic() - started < 2.0, "the connection hung instead"


def test_a_worker_that_dies_mid_render_answers_and_then_reports_not_ready(
        tmp_path):
    """A BaseException out of the model kills the worker. It must not also
    kill the ANSWER: _serve_stream catches Exception, so a SystemExit crossing
    the thread untouched would reach the client as EOF with no refusal line --
    the one thing the protocol promises never to do before audio."""
    path = tmp_path / "breeze.sock"

    def render(text, gain=1.0):
        raise SystemExit("the venv went away")
        yield pcm(2)                                   # pragma: no cover

    svc = ready_service(render)
    start_real_server(path, svc)
    head, rest = split_reply(ask(path, b'{"text": "hi", "stream": true}\n', 5))
    assert head["ok"] is False and rest == b""
    assert "SystemExit" in head["error"] and "died" in head["error"]
    assert wait_until(lambda: svc.worker.alive is False), "the worker survived"
    pong, _ = split_reply(ask(path, b'{"ping": true}\n', 5))
    assert pong["ok"] is True and pong["ready"] is False
    again, _ = split_reply(ask(path, b'{"text": "again", "stream": true}\n', 5))
    assert again["ok"] is False                        # refused, not queued


def test_concurrent_renders_serialise_and_their_audio_never_interleaves(
        tmp_path):
    """One resident model, one render thread: three clients at once must come
    out as three whole utterances, not three shuffled ones."""
    path = tmp_path / "breeze.sock"
    windows = []
    seen = threading.Lock()

    def render(text, gain=1.0):
        i = int(text.rsplit(" ", 1)[1])
        with seen:
            windows.append(("in", i))
        yield pcm(2, 100 + i)
        time.sleep(0.05)
        yield pcm(2, 200 + i)
        with seen:
            windows.append(("out", i))

    start_real_server(path, ready_service(render))
    got = {}

    def one(i):
        _, rest = split_reply(ask(path, json.dumps(
            {"text": f"line {i}", "stream": True}).encode() + b"\n", 30))
        got[i] = rest

    askers = [threading.Thread(target=one, args=(i,), daemon=True)
              for i in range(3)]
    for t in askers:
        t.start()
    for t in askers:
        t.join(30)
    for i in range(3):
        assert got[i] == bs.wav_header(24000) + pcm(2, 100 + i) + pcm(2, 200 + i)
    assert [w[0] for w in windows] == ["in", "out"] * 3, windows


def test_shutting_the_worker_down_is_clean_and_final():
    rendered = []

    def render(text, gain=1.0):
        rendered.append(text)
        yield pcm(2)

    worker = bs.RenderWorker(render)
    assert list(worker("one")) == [pcm(2)]
    thread = worker._thread
    worker.stop()
    assert thread.is_alive() is False
    assert worker.alive is False
    with pytest.raises(RuntimeError):
        list(worker("two"))
    assert rendered == ["one"]
    worker.stop()                                # idempotent


def test_stopping_a_worker_that_never_started_is_not_an_error():
    worker = bs.RenderWorker(blocks_render(pcm(2)))
    worker.stop()
    assert worker.alive is False


def test_the_boot_warm_renders_run_on_the_worker_thread():
    """The one-off 16.2 GiB is paid at boot, inside the GPU flock, next to the
    18.2 GiB capture transient the flock exists to serialise -- not on
    Hunter's first sentence. This cannot be exercised without the model, so
    the wiring is asserted where it lives."""
    body = (REPO / "scripts" / "breeze_server.py").read_text()
    body = body.split("def load_engine", 1)[1].split("def socket_owner", 1)[0]
    assert "worker = RenderWorker(render)" in body
    warm = body.split('for label in ("discarded", "measured")', 1)[1]
    assert "for block in worker(" in warm
    assert "return worker, graphs, None, config" in body


# ===========================================================================
# 12. the room never waits on bookkeeping
# ===========================================================================
def test_end_audio_releases_the_player_without_ending_the_producer():
    s = tts_mod._AudioStream("x")
    s.feed(b"pcm")
    s.end_audio()
    assert s.get(timeout=0) == b"pcm"
    assert s.get(timeout=0) is None          # the player may close its stdin
    assert not s.done.is_set()               # ...but the producer is not done
    s.end_audio()                            # idempotent: no second sentinel
    s.close()
    assert s.done.is_set()
    with pytest.raises(queue.Empty):
        s.get(timeout=0)


def test_the_fish_stream_contract_is_byte_identical():
    """_AudioStream is shared. Fish must keep the deadline and the 'heard'
    floor it was measured with."""
    s = tts_mod._AudioStream("x")
    assert s.timeout == tts_mod.FISH_TIMEOUT_S == 10.0
    assert s.min_heard == tts_mod._WAV_HEADER_BYTES == 44
    s.feed(b"a" * 45)
    assert s.heard is True
    s.close()                                # one sentinel, as it always was
    assert s.get(timeout=0) == b"a" * 45
    assert s.get(timeout=0) is None
    with pytest.raises(queue.Empty):
        s.get(timeout=0)


def test_the_breeze_audible_floor_is_a_duration_not_the_header_length():
    """44 bytes is the header alone and always fell back correctly; 46 -- the
    header and ONE sample, 0.04 ms -- counted as 'heard', so the clause was
    dropped with nothing audible and nothing cached."""
    assert tts_mod.BREEZE_MIN_HEARD_BYTES == 44 + int(24000 * 2 * 0.02)
    s = tts_mod._AudioStream("x", min_heard=tts_mod.BREEZE_MIN_HEARD_BYTES)
    s.feed(b"a" * 46)
    assert s.heard is False
    s.feed(b"a" * 1000)
    assert s.heard is True


def test_a_stream_cut_just_past_the_header_is_re_rendered_on_f5(breeze,
                                                                sock_path):
    line = "A clause that must not vanish."

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            srv.send(conn, {"ok": False, "error": "unknown request id"})
            return
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None) + pcm(1))
        conn.close()                         # 46 bytes, then gone

    srv = FakeSidecar(sock_path, handler)
    f5 = []
    try:
        breeze._synth_f5 = lambda text, out: (
            f5.append(text), open(out, "wb").write(wav_bytes(0.1)))
        breeze.speak(line, block=True)
    finally:
        srv.close()
    assert f5 == [line]                      # the clause was spoken, on F5
    assert breeze.cache.get(breeze._cache_key("f5", line)) is not None
    assert breeze.cache.get(breeze._cache_key("breeze", line)) is None


def test_a_slow_completion_receipt_does_not_stall_the_reply(
        breeze, sock_path, monkeypatch):
    """The receipt is a SECOND connection, and it used to be the tail of the
    producer's generator: the producer waited for it before rendering the next
    sentence, and the consumer waited for it again (stream.done) before taking
    that sentence off the queue. Measured with a status query blocked for 6 s
    and the deadline cut to 4 s to keep the test short: player 1 at t=0.00 s,
    player 2 at t=6.00 s, for 0.4 s of audio between them. Shipped, the
    deadline was 15 s.

    Three things now bound it. The sidecar answers receipts off its render
    lock (so the CAUSE -- a receipt queued behind a ~14 s phone render -- is
    gone), the round-trip runs on its own thread in parallel with playback and
    with the next chunk's render, and the deadline itself is 2 s. This drives
    the worst case there is: a sidecar that accepts the status connection and
    never answers it at all.
    """
    monkeypatch.setattr(tts_mod, "BREEZE_RECEIPT_TIMEOUT_S", 1.0)
    audio = wav_bytes(0.2)[44:]
    blocked = threading.Event()
    order, asked = [], {}

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            blocked.wait(30)                 # never answers within the deadline
            return
        order.append(req["text"])
        asked[req["text"]] = time.monotonic()
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None) + audio)
        conn.shutdown(socket.SHUT_WR)

    srv = FakeSidecar(sock_path, handler, threaded=True)
    started = time.monotonic()
    try:
        breeze.speak("The reactor is holding steady this evening. "
                     "The workshop is quiet and the coffee is fresh.",
                     block=True)
    finally:
        blocked.set()
        srv.close()
    assert len(order) == 2 and order[0] != order[1]
    # The producer asked for sentence 2 without waiting for sentence 1's
    # receipt -- that round-trip is not on the path between two sentences.
    assert asked[order[1]] - asked[order[0]] < 1.0
    # And the whole reply is bounded by the DEADLINE, not by the sidecar's
    # silence: two chunks at 1 s each, against a 15 s deadline that used to be
    # paid twice, in silence, on the producer AND the consumer.
    assert time.monotonic() - started < 8.0
    assert breeze.cache.stats()["files"] == 0   # no receipt, no cache entry
    assert len(FakeProc.spawned) == 2           # both sentences were spoken


def test_a_truncated_chunk_is_still_never_cached(breeze, sock_path):
    """Moving the receipt off the producer thread must not weaken what it
    decides: no positive 'complete' means no cache entry, and the audio that
    was heard is never re-spoken."""
    audio = wav_bytes(0.3)[44:]

    def handler(srv, conn, req):
        if req.get("ping"):
            srv.send(conn, {"ok": True, "ready": True, "graphs": True})
            return
        if req.get("status"):
            srv.send(conn, {"ok": True, "complete": False, "error": "cut"})
            return
        srv.send(conn, {"ok": True, "stream": True, "sr": 24000})
        conn.sendall(tts_mod.wav_header(24000, 1, 2, None) + audio)
        conn.shutdown(socket.SHUT_WR)

    srv = FakeSidecar(sock_path, handler)
    f5 = []
    try:
        breeze._synth_f5 = lambda text, out: f5.append(text)
        breeze.speak("Was that the whole thing?", block=True)
    finally:
        srv.close()
    assert f5 == []
    assert breeze.cache.stats()["files"] == 0


# ===========================================================================
# 13. starting up without stalling the room
# ===========================================================================
def test_the_speak_path_never_waits_for_a_sidecar_that_is_starting(
        sock_path, no_spawn, monkeypatch, tmp_path):
    """The unit is Type=simple, so systemd reports it active from the moment
    python starts -- before the GPU flock, the memory gate, 7.3 s of load and
    22.0 s of capture. It also holds ~/voice-training/.gpu.lock across all of
    that, so 'active with no socket' is also what QUEUED BEHIND ANOTHER GPU
    JOB looks like, for as long as that job runs. Waiting for it here cost the
    reply the whole budget in silence (measured 4.0 s with it patched to 4;
    shipped it was F5's 180) and then used F5 anyway."""
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: True)
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda *a, **k: True)
    monkeypatch.setattr(tts_mod.TTS, "warm_f5_fallback", lambda self: None)
    warmed = []
    monkeypatch.setattr(tts_mod.TTS, "warm_breeze",
                        lambda self: warmed.append(1))
    t = TTS(engine="breeze", cache_dir=tmp_path / "cache")
    started = time.monotonic()
    assert t.load() is True
    assert time.monotonic() - started < 3.0
    assert t.engine == "f5"                  # speaking, now, in the local voice
    assert warmed == [1]                     # and the good voice is on its way


def test_the_startup_wait_is_its_own_budget_and_off_the_speak_path():
    """180 s was F5's number for F5's cold start. Breeze's unit measures ~30 s
    (7.3 s load + 22.0 s capture)."""
    assert tts_mod.BREEZE_STARTUP_TIMEOUT_S <= 120
    default = inspect.signature(
        tts_mod._ensure_breeze_server).parameters["startup_timeout"].default
    assert default == 0.0                    # the speak path waits for nothing


def test_the_background_warm_puts_the_good_voice_back(sock_path, no_spawn,
                                                      monkeypatch, tmp_path):
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: True)
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda *a, **k: True)
    monkeypatch.setattr(tts_mod.TTS, "warm_f5_fallback", lambda self: None)
    monkeypatch.setattr(tts_mod, "BREEZE_STARTUP_TIMEOUT_S", 15.0)
    t = TTS(engine="breeze", cache_dir=tmp_path / "cache")
    assert t.load() is True and t.engine == "f5"
    srv = FakeSidecar(sock_path, sidecar_handler([]))   # it finally came up
    try:
        assert wait_until(lambda: t.engine == "breeze", 15), \
            "the sidecar answered and nothing picked it back up"
    finally:
        srv.close()
        t._breeze_warm_thread.join(5)


def test_the_background_warm_does_not_override_his_own_choice(
        sock_path, no_spawn, monkeypatch, tmp_path):
    """It only ever RESTORES a demotion we made. If he switched engines while
    it waited, his setting wins."""
    monkeypatch.setattr(tts_mod, "_breeze_unit_active", lambda: True)
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda *a, **k: True)
    monkeypatch.setattr(tts_mod.TTS, "warm_f5_fallback", lambda self: None)
    monkeypatch.setattr(tts_mod, "BREEZE_STARTUP_TIMEOUT_S", 15.0)
    t = TTS(engine="breeze", cache_dir=tmp_path / "cache")
    assert t.load() is True
    t.engine = "edge"
    srv = FakeSidecar(sock_path, sidecar_handler([]))
    try:
        t._breeze_warm_thread.join(15)
        assert t.engine == "edge"
    finally:
        srv.close()


def test_waiting_for_the_gpu_lock_is_not_silent(tmp_path, monkeypatch, capsys):
    """gpu_lock is entered before the only other output in main(), so between
    ExecStartPre and the memory gate journalctl showed NOTHING for as long as
    another GPU job held the lock -- while systemd reported the unit active."""
    lock = tmp_path / "gpu.lock"
    calls = []
    real = fcntl.flock

    def _flock(fd, op):
        calls.append(op)
        if len(calls) == 1:
            raise BlockingIOError("held by another GPU job")
        return real(fd, op)

    monkeypatch.setattr(bs.fcntl, "flock", _flock)
    with bs.gpu_lock(lock):
        pass
    out = capsys.readouterr().out
    assert "waiting for the GPU lock" in out
    assert "got the GPU lock" in out
    assert calls[0] == fcntl.LOCK_EX | fcntl.LOCK_NB
    assert calls[1] == fcntl.LOCK_EX          # blocking, after it said so


# ===========================================================================
# 14. what it costs, and what happens when it refuses
# ===========================================================================
def test_a_deliberate_refusal_does_not_burn_the_units_restarts():
    """breeze_server.py returns 2 for a missing path, for a memory refusal and
    for a second instance -- none of which a restart can fix. Without this,
    systemd counted each refusal as a start attempt, so three in ten minutes
    replaced the (actionable) MemFree message with "start request repeated too
    quickly" and left the unit failed until reset-failed. That is exactly what
    the obvious recovery sequence hits: start, read it, free memory, start."""
    assert _unit_fields()["RestartPreventExitStatus"] == ["2"]


def test_the_installer_says_what_the_gate_needs_and_how_to_clear_a_failure():
    """MemFree measured 19.2-29.2 GiB on 2026-09-02 with ollama, F5, Jarvis
    and the desktop up, against a floor of 33 -- so the refusal is the
    expected first outcome and the message has to carry the lever."""
    src = SETUP.read_text()
    assert f"MemFree >= {bs.MIN_FREE_GB} GB" in src
    assert f"MemAvailable >= {bs.MIN_AVAILABLE_GB} GB" in src
    assert "reset-failed" in src
    assert "ollama stop" in src


def test_the_server_measures_its_own_steady_state_after_the_warm_renders():
    """13.5 GB and 18.3 GB are render_q.py's external MemFree deltas, not this
    server's; MEASUREMENTS.json records peak_gpu_alloc_gb 6.322 for this exact
    Q4 config. The print inside the capture block runs BEFORE the two warm
    renders, which are what allocate the KV caches and codec buffers, and it
    reports a high-water mark rather than what is held. empty_cache() first
    because on GB10 the GPU memory IS the system memory and torch's caching
    allocator never returns reserved-but-free blocks on its own."""
    body = (REPO / "scripts" / "breeze_server.py").read_text()
    body = body.split("def load_engine", 1)[1]
    after_warm = body.split('breeze: warm (', 1)[1]
    assert "torch.cuda.empty_cache()" in after_warm
    assert "breeze: resident" in after_warm
    for field in ("free_mem_gb()", "memory_allocated", "memory_reserved"):
        assert field in after_warm, field
