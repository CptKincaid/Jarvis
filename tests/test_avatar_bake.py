"""Display-free tests for the out-of-process avatar bake kernel + runner.

Since 2026-09-01 the kernel has a `look`: "holo" (default, the judged
sphere in holo_kernel.py) and "classic" (the 08-31 streak cloud, whose
frames must stay byte-identical -- CLASSIC_SHA256 below are the hashes of
the pre-overhaul module, recorded from a HEAD d38b493 snapshot)."""
import hashlib
import queue
import subprocess
import sys
import time

import numpy as np
import pytest

from jarvis.ui import avatar_bake
from jarvis.ui.avatar_bake import (DEFAULT_LOOK, HEADER, LOOKS, BakeKernel,
                                   BakeRunner, check_look, hex_rgb)

BG, CYAN = hex_rgb("0d1b2a"), hex_rgb("35e0ff")
SIZE, SUP, N = 96, 1, 300
POOL = (0.22, 200.0)

# sha256 of the classic frame bytes at (size, sup) for k in (0, 37), N=300,
# POOL above, BG/CYAN above -- from the pre-overhaul avatar_bake.py
CLASSIC_SHA256 = {
    (96, 1, 0): "5306299656125ca2456de72436de805c5a72b65bc177b3a78ed2f0a48c038b80",
    (96, 1, 37): "a0a3586afc7be40fc0e8252a14976bb2250d37e481743abc18dce2a4eb450d67",
    (64, 2, 0): "a21343b511bba7568c28381f2b3cafffc565a5f36274df471eab290863228466",
    (64, 2, 37): "8100aa9d41d802acc24a061ff626b897f7eec782735ab0533cfe95fa6a0efcf3",
}


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


# ------------------------------------------------------------------ look
def test_default_look_is_holo_and_the_names_match_theme():
    assert DEFAULT_LOOK == "holo"
    assert LOOKS == ("holo", "classic")
    assert check_look("classic") == "classic"
    assert check_look(" Holo ") == "holo"
    assert check_look("neon") == DEFAULT_LOOK          # lenient, like theme
    assert check_look(None) == DEFAULT_LOOK


def test_holo_kernel_delegates_to_the_judged_sphere():
    from jarvis.ui.holo_kernel import HoloKernel
    k = BakeKernel(64, 1, N, POOL, BG, CYAN, look="holo")
    assert k.look == "holo" and isinstance(k._holo, HoloKernel)
    ref = HoloKernel(64, 1, N, POOL, BG, CYAN, avatar_bake.AV_SEED)
    assert k.frame(3).tobytes() == ref.frame(3).tobytes()
    assert BakeKernel(64, 1, N, POOL, BG, CYAN).look == "holo"   # default


@pytest.mark.parametrize("size,sup", [(96, 1), (64, 2)])
def test_classic_frames_are_byte_identical_to_the_pre_overhaul_bake(size, sup):
    k = BakeKernel(size, sup, N, POOL, BG, CYAN, look="classic")
    assert k.look == "classic" and k._holo is None
    for i in (0, 37):
        digest = hashlib.sha256(np.asarray(k.frame(i)).tobytes()).hexdigest()
        assert digest == CLASSIC_SHA256[(size, sup, i)], (size, sup, i)


def test_runner_passes_the_look_to_its_workers():
    r = BakeRunner(SIZE, SUP, N, [0], POOL, BG, CYAN, look="classic")
    cmd = r._cmd([0, 12])
    assert cmd[cmd.index("--look") + 1] == "classic"
    assert BakeRunner(SIZE, SUP, N, [0], POOL, BG, CYAN).look == "holo"


def test_cli_look_flag_selects_the_classic_kernel():
    cmd = [sys.executable, "-m", "jarvis.ui.avatar_bake", "--size", "64",
           "--sup", "2", "--frames", str(N), "--pool-peak", str(POOL[0]),
           "--pool-r", str(POOL[1]), "--bg", "0d1b2a", "--cyan", "35e0ff",
           "--look", "classic", "--ks", "37"]
    out = subprocess.run(cmd, capture_output=True, cwd=avatar_bake._repo_root(),
                         timeout=120)
    assert out.returncode == 0, out.stderr.decode(errors="replace")
    (k,) = HEADER.unpack(out.stdout[:HEADER.size])
    assert k == 37
    digest = hashlib.sha256(out.stdout[HEADER.size:]).hexdigest()
    assert digest == CLASSIC_SHA256[(64, 2, 37)]
    # an unknown look is a usage error at the CLI (argparse choices)
    bad = subprocess.run(cmd[:-3] + ["neon", "--ks", "0"], capture_output=True,
                         cwd=avatar_bake._repo_root(), timeout=60)
    assert bad.returncode != 0
