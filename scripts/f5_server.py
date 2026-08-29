"""F5-TTS sidecar: holds the model resident, speaks over a unix socket.

WHY A SIDECAR AND NOT AN IMPORT
-------------------------------
f5-tts cannot share Jarvis's environment. Installing it into ~/vss_env
REMOVES fastapi and downgrades rich (checked with `uv pip install --dry-run`
on 2026-08-28), and vss_env hosts the VSS project's API. So F5 lives in its
own venv at ~/.local/share/jarvis-f5/venv and Jarvis talks to it here.

Loading F5 costs seconds, so the model must stay resident: jarvis/tts.py
spawns this once and reuses it for the life of the app.

PROTOCOL: newline-delimited JSON over a unix socket.
    -> {"text": "Good evening, sir.", "out": "/tmp/x.wav"}
    <- {"ok": true, "seconds": 1.37}
    <- {"ok": false, "error": "..."}
    -> {"ping": true}  ->  {"ok": true, "ready": true}
"""
import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

# Settled by listening on 2026-08-28. nfe_step is the whole latency lever
# (flow matching runs a fixed number of denoising steps regardless of text
# length): 32 -> 1.90s, 16 -> 1.01, 12 -> 0.78, 8 -> 0.57 on a short line.
# 8 was the lowest step count without artefacts, and at 8 the user preferred
# speed 0.70 at EVERY chunk length tested (22/43/84/122 chars) -- 0.50 read
# as leaving too long a pause at full stops.
DEFAULTS = {"nfe_step": 8, "speed": 0.7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--nfe", type=int, default=DEFAULTS["nfe_step"])
    ap.add_argument("--speed", type=float, default=DEFAULTS["speed"])
    a = ap.parse_args()

    ref_text = Path(a.ref_text).read_text().strip()

    from f5_tts.api import F5TTS
    t0 = time.perf_counter()
    f5 = F5TTS(device="cuda")
    # One throwaway synthesis: the first call pays CUDA/graph warm-up, and we
    # would rather pay it here than on the user's first spoken reply.
    warm = Path(a.socket).with_suffix(".warm.wav")
    f5.infer(ref_file=a.ref, ref_text=ref_text, gen_text="Ready.",
             nfe_step=a.nfe, speed=a.speed, seed=42, file_wave=str(warm))
    try:
        warm.unlink()
    except OSError:
        pass
    print(f"f5: model resident in {time.perf_counter()-t0:.1f}s", flush=True)

    sock_path = a.socket
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)
    srv.listen(4)
    print(f"f5: listening on {sock_path}", flush=True)

    while True:
        conn, _ = srv.accept()
        try:
            conn.settimeout(120)
            buf = b""
            while not buf.endswith(b"\n"):
                part = conn.recv(65536)
                if not part:
                    break
                buf += part
            if not buf.strip():
                continue
            req = json.loads(buf.decode("utf-8"))
            if req.get("ping"):
                conn.sendall(json.dumps({"ok": True, "ready": True}).encode() + b"\n")
                continue
            t = time.perf_counter()
            wav, sr, _ = f5.infer(
                ref_file=a.ref, ref_text=ref_text, gen_text=req["text"],
                nfe_step=int(req.get("nfe", a.nfe)),
                speed=float(req.get("speed", a.speed)),
                seed=42, file_wave=req["out"])
            resp = {"ok": True, "seconds": len(wav) / sr,
                    "wall": time.perf_counter() - t}
        except Exception as exc:                     # never kill the server
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        try:
            conn.sendall(json.dumps(resp).encode() + b"\n")
        except OSError:
            pass
        finally:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
