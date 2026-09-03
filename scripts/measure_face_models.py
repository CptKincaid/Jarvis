"""Measure what YuNet + SFace actually cost on this box, on SYNTHETIC frames.

Run:  ~/vss_env/bin/python scripts/measure_face_models.py

WHY SYNTHETIC. Hunter's standing rule is that nothing may look at what his
camera sees. Every number this script prints is therefore produced from frames
it generates itself -- noise, gradients and drawn shapes -- and every number it
prints is a TIMING or a SIZE, never an accuracy. That split is not a
limitation to work around, it is the honest boundary:

* Detector COST is content-independent to within noise. YuNet is a fixed
  fully-convolutional graph: the conv stack does identical work on noise and
  on a face, and only the NMS tail scales with how many boxes survive, which
  at a desk is 0-2. So a synthetic frame measures the real per-frame cost.
* Detector ACCURACY is entirely content-dependent, and a synthetic face is not
  a face. This script will not print a confidence score for a drawn face and
  call it evidence. `--probe-synthetic` measures what a crude drawn face
  actually scores -- 0.33 here, i.e. over a 0.3 floor and comfortably under
  the shipped 0.6 -- which bounds the false-positive side and NOTHING else.
  Whether a real 167 px face clears min_conf is a question only a real camera
  can answer, and it belongs to Hunter to run.

Nothing here opens a capture device. There is no cv2.VideoCapture in this file.
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import statistics
import time

import cv2
import numpy as np

MODEL_DIR = os.path.expanduser("~/.aiws_trainer/models/face")
YUNET = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_INT8 = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar_int8.onnx")
SFACE = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
SFACE_INT8 = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec_int8.onnx")


def rss_mb() -> float:
    """Resident set size in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rss_now_mb() -> float:
    """Current (not peak) RSS in MB, from /proc/self/statm."""
    with open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def synth_frame(w: int, h: int, seed: int = 0) -> np.ndarray:
    """A deterministic synthetic frame. Structured noise, not flat grey:
    a flat frame would let any lazy early-out in the graph fire and would
    under-report the real cost."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    base[:, :, 1] = np.tile(grad, (h, 1))
    return base


def timed(fn, n: int, warmup: int = 5):
    """Return (p50_ms, p95_ms, cpu_ms_mean) over n calls."""
    for _ in range(warmup):
        fn()
    wall, cpu = [], []
    for _ in range(n):
        c0, t0 = time.process_time(), time.perf_counter()
        fn()
        wall.append((time.perf_counter() - t0) * 1000.0)
        cpu.append((time.process_time() - c0) * 1000.0)
    wall.sort()
    return (statistics.median(wall),
            wall[min(len(wall) - 1, int(0.95 * len(wall)))],
            statistics.mean(cpu))


def bench_detect(model: str, threads: int, n: int = 60):
    """Detection cost: resize from a 1280x720 capture, then detect."""
    cv2.setNumThreads(threads)
    rows = []
    cap = synth_frame(1280, 720, seed=1)
    for dw, dh in ((320, 180), (320, 240), (640, 360), (1280, 720)):
        det = cv2.FaceDetectorYN_create(model, "", (dw, dh), 0.6, 0.3, 5000)
        small = cv2.resize(cap, (dw, dh))
        r_p50, r_p95, r_cpu = timed(
            lambda: cv2.resize(cap, (dw, dh)), n)
        d_p50, d_p95, d_cpu = timed(lambda: det.detect(small), n)
        rows.append((f"{dw}x{dh}", r_p50, d_p50, d_p95, r_p50 + d_p50,
                     r_cpu + d_cpu))
    return rows


def bench_recognise(model: str, threads: int, n: int = 40):
    """Embedding cost. alignCrop needs a 15-value detection row
    (x,y,w,h + 5 landmarks + score); we fabricate a plausible one, which is
    valid for TIMING because the warp is a fixed 112x112 similarity transform
    whose cost does not depend on whether a face is really there."""
    cv2.setNumThreads(threads)
    rec = cv2.FaceRecognizerSF_create(model, "")
    frame = synth_frame(1280, 720, seed=2)
    # A 161 px face at the centre -- the size section 9's mount arithmetic
    # produces -- with eyes/nose/mouth at their canonical offsets.
    x, y, w, h = 560.0, 280.0, 161.0, 161.0
    face = np.array([[x, y, w, h,
                      x + 0.31 * w, y + 0.38 * h,   # right eye
                      x + 0.69 * w, y + 0.38 * h,   # left eye
                      x + 0.50 * w, y + 0.58 * h,   # nose tip
                      x + 0.35 * w, y + 0.77 * h,   # right mouth corner
                      x + 0.65 * w, y + 0.77 * h,   # left mouth corner
                      0.99]], dtype=np.float32)
    a_p50, a_p95, a_cpu = timed(lambda: rec.alignCrop(frame, face[0]), n)
    crop = rec.alignCrop(frame, face[0])
    f_p50, f_p95, f_cpu = timed(lambda: rec.feature(crop), n)
    dim = int(np.asarray(rec.feature(crop)).size)
    return a_p50, f_p50, f_p95, a_p50 + f_p50, a_cpu + f_cpu, dim


def _mk_det(path, _=""):
    return cv2.FaceDetectorYN_create(path, "", (320, 180), 0.6, 0.3, 5000)


def _mk_rec(path, _=""):
    return cv2.FaceRecognizerSF_create(path, "")


def cold_load(path, ctor, reps: int = 3):
    """Cold-load wall time and the RSS the loaded model holds."""
    times = []
    for _ in range(reps):
        gc.collect()
        t = time.perf_counter()
        m = ctor(path, "")
        times.append((time.perf_counter() - t) * 1000.0)
        del m
        gc.collect()
    gc.collect()
    before = rss_now_mb()
    keep = ctor(path, "")
    after = rss_now_mb()
    return statistics.median(times), max(0.0, after - before), keep


def probe_synthetic(thr: float = 0.3):
    """What does a DRAWN face score? Bounds the false-positive side only.

    Returns (n_detections, best_score). Measured 2026-09-02: a drawn oval with
    two dots and a mouth arc scores 0.3317 -- it survives a 0.3 floor and is
    rejected by the shipped 0.6. Uniform noise, flat grey and a linear
    gradient score nothing at all at 0.3.
    """
    det = cv2.FaceDetectorYN_create(YUNET, "", (320, 180), thr, 0.3, 5000)
    img = np.full((180, 320, 3), 200, np.uint8)
    cv2.ellipse(img, (160, 90), (34, 45), 0, 0, 360, (170, 150, 140), -1)
    cv2.circle(img, (148, 78), 5, (40, 40, 40), -1)
    cv2.circle(img, (172, 78), 5, (40, 40, 40), -1)
    cv2.ellipse(img, (160, 108), (12, 6), 0, 0, 180, (90, 60, 60), 2)
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return 0, 0.0
    return len(faces), float(max(f[14] for f in faces))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--probe-synthetic", action="store_true")
    args = ap.parse_args()

    print(f"cv2 {cv2.__version__}   cores {os.cpu_count()}   "
          f"models {MODEL_DIR}")
    print()

    print("== cold load (median of 3) + resident cost ==")
    keeps = []
    for name, path, ctor in (
            ("YuNet fp32", YUNET, _mk_det),
            ("YuNet int8", YUNET_INT8, _mk_det),
            ("SFace fp32", SFACE, _mk_rec),
            ("SFace int8", SFACE_INT8, _mk_rec)):
        ms, mb, keep = cold_load(path, ctor)
        keeps.append(keep)
        print(f"  {name:12s} {os.path.getsize(path)/1e6:7.2f} MB on disk   "
              f"load {ms:7.2f} ms   RSS +{mb:6.2f} MB")
    print(f"  process peak RSS with all four resident: {rss_mb():.1f} MB")
    print()

    for th in args.threads:
        print(f"== YuNet fp32 detection, {th} thread(s) ==")
        print(f"  {'detect size':12s} {'resize':>8s} {'detect p50':>11s} "
              f"{'detect p95':>11s} {'chain':>8s} {'cpu-ms':>8s}")
        for size, r, d50, d95, tot, cpu in bench_detect(YUNET, th):
            print(f"  {size:12s} {r:7.2f}m {d50:10.2f}m {d95:10.2f}m "
                  f"{tot:7.2f}m {cpu:7.2f}")
        a, f50, f95, tot, cpu, dim = bench_recognise(SFACE, th)
        print(f"  SFace fp32 ({dim}-D): align {a:.2f} ms + feature p50 "
              f"{f50:.2f} / p95 {f95:.2f} ms = {tot:.2f} ms  "
              f"({cpu:.2f} cpu-ms)")
        a, f50, f95, tot, cpu, dim = bench_recognise(SFACE_INT8, th)
        print(f"  SFace int8 ({dim}-D): align {a:.2f} ms + feature p50 "
              f"{f50:.2f} / p95 {f95:.2f} ms = {tot:.2f} ms  "
              f"({cpu:.2f} cpu-ms)")
        print()

    if args.probe_synthetic:
        print("== synthetic-face probe (false-positive side only) ==")
        for thr in (0.3, 0.5, 0.6):
            n, best = probe_synthetic(thr)
            print(f"  drawn oval @ min_conf {thr}: {n} detection(s), "
                  f"best score {best:.4f}")
        print("  A crude drawn face scores ~0.33: over a 0.3 floor, under the")
        print("  shipped 0.6. That bounds false positives and says NOTHING")
        print("  about whether a real face is found -- only a real camera can")
        print("  answer that, and the measurement belongs to Hunter.")


if __name__ == "__main__":
    main()
