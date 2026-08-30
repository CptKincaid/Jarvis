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

# Settled by BLIND listening, rounds 1-5, 2026-08-28/29. This exact
# configuration scored 5.00 against the real hosted Fish control at 5.00 --
# straight 5s, indistinguishable in a shuffled set.
#
# nfe_step 10, not 8: 8 is OUTSIDE F5's pruned EPSS timestep table, which
# defines schedules only for n in {5, 6, 7, 10, 12, 16}. Quality saturates at
# 10; 12 is statistically tied but grazes the RTF gate on a slow draw.
# speed 0.85: at 1.0 this reference paces short lines ~11% fast.
# The other three are F5's own defaults, pinned so a library change cannot
# silently move the voice. Do NOT vary sway_sampling_coef (other values
# collapse) or ode_method (midpoint is strictly dominated).
DEFAULTS = {"nfe_step": 10, "speed": 0.85}
# seed 1234, NOT 42. Every clip in the round-6 blind identification test was
# rendered at 1234 (verified: all six reproduce sha256-identical at that seed,
# waveform corr +1.0000). 42 is a different, never-blind-tested draw of the same
# config -- not measurably worse, but not the voice that was actually approved.
# Sampling controls WHICH tune you get, not how much movement: clip08 (heard as
# "varied") and all five clips heard as "monotone" came from this same seed, so
# expressiveness is not a seed property and re-rolling it will not buy any.
SEED = 1234
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0
CROSS_FADE_DURATION = 0.15
ODE_METHOD = "euler"

# ------------------------------------------------- the short-utterance floor
#
# F5 STARVES SHORT TEXT. utils_infer.py derives the total duration as
#
#     local_speed = 0.3 if len(gen_text.encode()) < 10 else speed
#     duration = ref_len + int(ref_len / ref_text_len * gen_text_len / local_speed)
#
# i.e. the generated span is strictly PROPORTIONAL to gen-text bytes, through
# the origin. Real speech is AFFINE: over all 400 corpus clips this speaker
# obeys  sec = 0.2540 + 0.04988 * bytes  (r = 0.960, n = 400). A proportional
# allotment therefore has slack on long text and starves short text. Measured
# consequence before this fix: "Understood." rendered as literal SILENCE,
# "It is done." as "You", "Always, sir." as "Away" -- 7 of 24 short lines.
#
# The fix is a FLOOR, never a replacement. When the native allotment already
# clears the affine requirement the function returns None and infer() takes
# the exact upstream path, byte for byte. VERIFIED: every long-form render is
# sha256-identical with and without it. It can only lengthen a starved short
# utterance.
FISH_SEC_PER_BYTE = 0.04988      # fitted over all 400 corpus clips
FLOOR_C = 0.45                   # blind-tested against C=0.65; 0.45 won and
                                 # is the narrower intervention (binds below
                                 # 41 bytes rather than 60)
MEL_FPS = 24000 / 256            # hop_length 256 at 24 kHz


def duration_floor(ref_frames, ref_text_bytes, gen_text, speed,
                   C=FLOOR_C, M=FISH_SEC_PER_BYTE):
    """fix_duration for a starved short utterance, else None (exact no-op).

    ref_frames/ref_text_bytes MUST come from the PREPROCESSED reference --
    preprocess_ref_audio_text strips edge silence, appends 50 ms, and appends
    ". " to the text. Using the raw file is a ~4% error.
    """
    gl = len(gen_text.encode("utf-8"))
    if not gl or not ref_text_bytes:
        return None
    local_speed = 0.3 if gl < 10 else speed
    native = int(ref_frames / ref_text_bytes * gl / local_speed) / MEL_FPS
    floor = C + M * gl
    if native >= floor:
        return None
    return (ref_frames + round(floor * MEL_FPS)) / MEL_FPS


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
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    import torchaudio
    t0 = time.perf_counter()
    f5 = F5TTS(device="cuda", ode_method=ODE_METHOD)

    # The floor needs the reference as infer() will actually see it, not as it
    # sits on disk: preprocess strips edge silence, appends 50 ms, and appends
    # ". " to the text. Deriving it here rather than hardcoding keeps the floor
    # correct if the reference clip is ever swapped.
    _pp_audio, _pp_text = preprocess_ref_audio_text(a.ref, ref_text)
    _wav, _sr = torchaudio.load(_pp_audio)
    REF_FRAMES = int(_wav.shape[-1] / _sr * MEL_FPS)
    REF_TEXT_BYTES = len(_pp_text.encode("utf-8"))
    print(f"f5: reference {REF_FRAMES} frames / {REF_TEXT_BYTES} bytes "
          f"({REF_FRAMES / MEL_FPS:.3f}s), floor binds below "
          f"{FLOOR_C / (REF_FRAMES / REF_TEXT_BYTES / MEL_FPS / a.speed - FISH_SEC_PER_BYTE):.0f} bytes",
          flush=True)
    # One throwaway synthesis: the first call pays CUDA/graph warm-up, and we
    # would rather pay it here than on the user's first spoken reply.
    warm = Path(a.socket).with_suffix(".warm.wav")
    f5.infer(ref_file=a.ref, ref_text=ref_text, gen_text="Ready.",
             nfe_step=a.nfe, speed=a.speed, cfg_strength=CFG_STRENGTH,
             sway_sampling_coef=SWAY_SAMPLING_COEF,
             cross_fade_duration=CROSS_FADE_DURATION,
             fix_duration=duration_floor(REF_FRAMES, REF_TEXT_BYTES,
                                         "Ready.", a.speed),
             seed=SEED, file_wave=str(warm))
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
            _speed = float(req.get("speed", a.speed))
            _fd = duration_floor(REF_FRAMES, REF_TEXT_BYTES,
                                 req["text"], _speed)
            wav, sr, _ = f5.infer(
                ref_file=a.ref, ref_text=ref_text, gen_text=req["text"],
                nfe_step=int(req.get("nfe", a.nfe)),
                speed=_speed,
                cfg_strength=CFG_STRENGTH,
                sway_sampling_coef=SWAY_SAMPLING_COEF,
                cross_fade_duration=CROSS_FADE_DURATION,
                fix_duration=_fd,
                seed=SEED, file_wave=req["out"])
            resp = {"ok": True, "seconds": len(wav) / sr,
                    "wall": time.perf_counter() - t,
                    "floored": _fd is not None}
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
