#!/usr/bin/env python3
"""Measure what the echo canceller actually removes, instead of trusting it.

Three subcommands, all offline maths except ``record``, and even that only
RECORDS -- nothing here ever plays audio.  Whatever is coming out of the
speakers while you measure is yours to start (Spotify on the "Spark"
Connect device, or Jarvis speaking), and the doc says why that must be a
person's decision: docs/echo-cancellation.md.

    aec_measure.py record       --seconds 12 --out DIR
        parecord the raw Snowball, jarvis_aec_source and the reference
        (jarvis_aec_sink.monitor) side by side, then print both numbers.
    aec_measure.py attenuation  RAW.wav AEC.wav
        dBFS of each, the attenuation in dB and a per-second table (the
        canceller takes a second or two to converge; watch it climb).
    aec_measure.py lag          MIC.wav REF.wav
        cross-correlation delay of the mic behind the reference: the
        Bluetooth A2DP path adds ~150-300 ms, and that number is the
        starting point for buffer.play_delay in the conf.

Ported 2026-09-01 from the two scratch one-liners used during the live
attempt (measure.py, lag.py).  The first 0.8 s of every file is skipped:
stream start-up plus the canceller's convergence, which would flatter or
damn the average depending on which side it landed.

Target: >= ~10 dB attenuation on music.  The 0.8 dB measured on 2026-09-01
is NOT a result -- the music that day came out of HPCOMPUTER's speakers, so
the canceller had no reference at all.
"""
from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONF = HERE / "99-jarvis-echo-cancel.conf"
SKIP_S = 0.8
# Fallback only; record() reads the pinned mic out of the conf beside this
# file so the two never disagree.
SNOWBALL = "alsa_input.usb-BLUE_MICROPHONE_Blue_Snowball_201506-00.analog-stereo"
AEC_SOURCE = "jarvis_aec_source"
# What entered the canceller.  This is the VIRTUAL sink's monitor, which is
# a different animal from the bluez sink's monitor that measured silent.
AEC_REF = "jarvis_aec_sink.monitor"
RATE = 48000


# ------------------------------------------------------------------ maths

def load_mono(path) -> tuple[np.ndarray, int]:
    """A wav as float64 mono (channels averaged) plus its sample rate."""
    import soundfile as sf
    data, rate = sf.read(str(path))
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(rate)


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return 20.0 * np.log10(rms + 1e-9)


def attenuation(raw: np.ndarray, aec: np.ndarray, rate: int,
                skip_s: float = SKIP_S) -> dict:
    """How much quieter the cancelled signal is than the raw mic.

    Both files are trimmed to the same length after the skip; the per-second
    column is what shows convergence (and a canceller that gave up)."""
    start = int(skip_s * rate)
    raw, aec = raw[start:], aec[start:]
    n = min(len(raw), len(aec))
    raw, aec = raw[:n], aec[:n]
    raw_db, aec_db = dbfs(raw), dbfs(aec)
    per_sec = []
    for i in range(n // rate):
        seg = slice(i * rate, (i + 1) * rate)
        per_sec.append(round(float(dbfs(raw[seg]) - dbfs(aec[seg])), 1))
    return {"raw_db": float(raw_db), "aec_db": float(aec_db),
            "atten_db": float(raw_db - aec_db),
            "per_sec_db": per_sec, "seconds": n / rate}


def lag(mic: np.ndarray, ref: np.ndarray, rate: int, skip_s: float = SKIP_S,
        max_lag_s: float = 1.0) -> dict:
    """Delay of the mic behind the reference, by FFT cross-correlation.

    Positive = the mic hears it AFTER it entered the sink, which is the
    only direction a canceller can work with.  ``ncc`` is the normalised
    peak: below ~0.1 the two files do not contain the same programme and
    the lag is noise -- the reference recording was silent, most likely."""
    start = int(skip_s * rate)
    n = min(len(mic), len(ref))
    a = mic[start:n] - mic[start:n].mean()
    b = ref[start:n] - ref[start:n].mean()
    if len(a) < 2:
        raise ValueError("recordings shorter than the skip window")
    max_lag = min(int(max_lag_s * rate), len(a) - 1)
    size = 2 * len(a)
    xc = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    xc = np.concatenate([xc[-max_lag:], xc[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    k = int(np.argmax(np.abs(xc)))
    energy = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12
    top = np.argsort(np.abs(xc))[-5:][::-1]
    return {"lag_ms": float(lags[k]) / rate * 1000.0,
            "ncc": float(abs(xc[k]) / energy),
            "runners_up_ms": [int(round(lags[t] / rate * 1000.0)) for t in top]}


# -------------------------------------------------------------- recording

def pinned_mic(conf: Path = CONF) -> str:
    """The target.object the conf pins the canceller's capture to."""
    try:
        m = re.search(r'target\.object\s*=\s*"([^"]+)"', conf.read_text())
    except OSError:
        return SNOWBALL
    return m.group(1) if m else SNOWBALL


def sources(run=subprocess.run) -> set[str]:
    try:
        out = run(["pactl", "list", "short", "sources"], capture_output=True,
                  text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.split("\t")[1] for line in out.splitlines() if "\t" in line}


def record_argv(device: str, path: Path) -> list[str]:
    # Mono s16 at the graph rate: every node here runs 48 kHz, so no
    # resampler sits between the measurement and the signal.
    return ["parecord", f"--device={device}", "--file-format=wav",
            f"--rate={RATE}", "--channels=1", "--format=s16le", str(path)]


def record(out_dir: Path, seconds: float, mic: str, aec: str, ref: str) -> dict:
    """Three parecords side by side, stopped together with SIGINT so each
    closes its wav header cleanly.  Returns the three paths."""
    roster = sources()
    missing = [d for d in (mic, aec, ref) if d not in roster]
    if missing:
        raise SystemExit("not in `pactl list short sources`: " + ", ".join(missing)
                         + "\n  is the canceller installed? scripts/audio/aec-install.sh")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"raw": out_dir / "raw.wav", "aec": out_dir / "aec.wav",
             "ref": out_dir / "ref.wav"}
    procs = {}
    try:
        for key, dev in (("raw", mic), ("aec", aec), ("ref", ref)):
            procs[key] = subprocess.Popen(record_argv(dev, paths[key]),
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE)
        time.sleep(seconds)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for key, p in procs.items():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
            err = (p.stderr.read().decode(errors="replace") if p.stderr else "").strip()
            if err:
                print(f"{key}: parecord said: {err}", file=sys.stderr)
    return paths


# ------------------------------------------------------------------- cli

def _print_attenuation(res: dict) -> None:
    print(f"raw {res['raw_db']:.1f} dBFS  aec {res['aec_db']:.1f} dBFS  "
          f"attenuation {res['atten_db']:.1f} dB over {res['seconds']:.1f} s")
    print(f"per-second dB: {res['per_sec_db']}")
    verdict = "meets the >= 10 dB target" if res["atten_db"] >= 10 else "below the ~10 dB target"
    print(f"-> {verdict}")


def _print_lag(res: dict) -> None:
    print(f"lag {res['lag_ms']:.1f} ms (mic after reference)  ncc {res['ncc']:.3f}  "
          f"runners-up {res['runners_up_ms']} ms")
    if res["ncc"] < 0.1:
        print("-> ncc is tiny: the two files do not share a programme; the "
              "reference was probably silent")


def cmd_attenuation(args) -> int:
    raw, r1 = load_mono(args.raw)
    aec, r2 = load_mono(args.aec)
    if r1 != r2:
        raise SystemExit(f"sample rates differ: {r1} vs {r2}")
    _print_attenuation(attenuation(raw, aec, r1, args.skip))
    return 0


def cmd_lag(args) -> int:
    mic, r1 = load_mono(args.mic)
    ref, r2 = load_mono(args.ref)
    if r1 != r2:
        raise SystemExit(f"sample rates differ: {r1} vs {r2}")
    _print_lag(lag(mic, ref, r1, args.skip, args.max_lag))
    return 0


def cmd_record(args) -> int:
    mic = args.mic or pinned_mic()
    print(f"recording {args.seconds:.0f} s from {mic}, {args.aec}, {args.ref} "
          f"-> {args.out}/  (start the music first; nothing is played from here)")
    paths = record(Path(args.out), args.seconds, mic, args.aec, args.ref)
    for key, p in paths.items():
        print(f"  {key}: {p}")
    raw, r1 = load_mono(paths["raw"])
    aec, r2 = load_mono(paths["aec"])
    ref, r3 = load_mono(paths["ref"])
    if not (r1 == r2 == r3):
        raise SystemExit(f"sample rates differ: {r1} {r2} {r3}")
    if len(raw) < r1 * (args.skip + 1):
        raise SystemExit("recording too short to say anything; raise --seconds")
    _print_attenuation(attenuation(raw, aec, r1, args.skip))
    _print_lag(lag(raw, ref, r1, args.skip))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record raw / cancelled / reference, then analyse")
    r.add_argument("--seconds", type=float, default=12.0)
    r.add_argument("--out", default="scratchpad/aec", help="directory for the three wavs")
    r.add_argument("--mic", default=None, help="raw mic source (default: the conf's target.object)")
    r.add_argument("--aec", default=AEC_SOURCE)
    r.add_argument("--ref", default=AEC_REF)
    r.add_argument("--skip", type=float, default=SKIP_S, help="seconds ignored at the start")
    r.set_defaults(fn=cmd_record)

    a = sub.add_parser("attenuation", help="raw vs cancelled dBFS and dB removed")
    a.add_argument("raw")
    a.add_argument("aec")
    a.add_argument("--skip", type=float, default=SKIP_S)
    a.set_defaults(fn=cmd_attenuation)

    g = sub.add_parser("lag", help="delay of the mic behind the reference")
    g.add_argument("mic")
    g.add_argument("ref")
    g.add_argument("--skip", type=float, default=SKIP_S)
    g.add_argument("--max-lag", type=float, default=1.0, help="search window in seconds")
    g.set_defaults(fn=cmd_lag)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
