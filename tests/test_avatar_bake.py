"""Display-free tests for the out-of-process avatar bake kernel + runner."""
import queue
import subprocess
import sys
import time

import numpy as np

from jarvis.ui import avatar_bake
from jarvis.ui.avatar_bake import HEADER, BakeKernel, BakeRunner, hex_rgb

BG, CYAN = hex_rgb("0d1b2a"), hex_rgb("35e0ff")
SIZE, SUP, N = 96, 1, 300
POOL = (0.22, 200.0)


def test_loop_is_bitwise_seamless_at_96px():
    k = BakeKernel(SIZE, SUP, N, POOL, BG, CYAN)
    a = np.asarray(k.frame(0))
    b = np.asarray(k.frame(N))
    assert a.shape == (SIZE, SIZE, 3)
    assert np.array_equal(a, b)
    # and the loop actually moves
    assert not np.array_equal(a, np.asarray(k.frame(1)))


def test_import_pulls_in_neither_tkinter_nor_the_app():
    code = ("import sys, jarvis.ui.avatar_bake; "
            "bad = [m for m in ('tkinter', 'jarvis.events', 'jarvis.logs', "
            "'jarvis.config', 'jarvis.ui.theme') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=avatar_bake._repo_root(), timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr


def test_cli_stream_framing_round_trip():
    ks = [0, 12, 6]
    cmd = [sys.executable, "-m", "jarvis.ui.avatar_bake", "--size", str(SIZE),
           "--sup", str(SUP), "--frames", str(N), "--pool-peak", str(POOL[0]),
           "--pool-r", str(POOL[1]), "--bg", "0d1b2a", "--cyan", "35e0ff",
           "--ks", ",".join(map(str, ks))]
    out = subprocess.run(cmd, capture_output=True, cwd=avatar_bake._repo_root(),
                         timeout=120)
    assert out.returncode == 0, out.stderr.decode(errors="replace")
    nbytes = SIZE * SIZE * 3
    assert len(out.stdout) == len(ks) * (HEADER.size + nbytes)
    kernel = BakeKernel(SIZE, SUP, N, POOL, BG, CYAN)
    pos = 0
    for expect in ks:
        (k,) = HEADER.unpack(out.stdout[pos:pos + HEADER.size])
        pos += HEADER.size
        buf = out.stdout[pos:pos + nbytes]
        pos += nbytes
        assert k == expect
        assert buf == kernel.frame(k).tobytes()    # same bytes as in-process


def _collect(runner, want, timeout=90.0):
    got = {}
    deadline = time.monotonic() + timeout
    while len(got) < want and time.monotonic() < deadline:
        try:
            k, buf = runner.queue.get(timeout=0.5)
        except queue.Empty:
            continue
        got[k] = buf
    return got


def test_runner_falls_back_to_in_process_when_cli_unavailable():
    order = [0, 12, 24, 36]
    r = BakeRunner(SIZE, SUP, N, order, POOL, BG, CYAN, workers=2,
                   python="/nonexistent/python")
    r.start()
    try:
        assert r.mode == "thread"
        got = _collect(r, len(order))
        assert sorted(got) == order
        assert all(len(b) == SIZE * SIZE * 3 for b in got.values())
    finally:
        r.stop()


def test_runner_subprocess_path_delivers_every_frame():
    order = [0, 12, 24, 36, 48, 60]
    r = BakeRunner(SIZE, SUP, N, order, POOL, BG, CYAN, workers=2)
    r.start()
    try:
        assert r.mode == "subprocess"
        got = _collect(r, len(order))
        assert sorted(got) == order
        kernel = BakeKernel(SIZE, SUP, N, POOL, BG, CYAN)
        assert got[12] == kernel.frame(12).tobytes()
    finally:
        r.stop()
