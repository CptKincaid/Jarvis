#!/usr/bin/env python3
"""Measure a speaker-verification threshold instead of guessing one.

`speaker_threshold` ships at 0.40. Whether that is right for YOUR mic, room and
seating is an empirical question: record some held-out clips of yourself and
some clips of the room (TV, music, other people), score both against the
enrolled voiceprint, and look at where they separate.

Clips are kept on disk so they can be re-scored after re-enrolling without
recording everything again.

Typical run:
    # 1. after scripts/enroll_voice.py, record held-out samples of yourself
    ~/vss_env/bin/python scripts/tune_speaker_threshold.py --record-me 5

    # 2. put the TV / a podcast on, then capture what the room sounds like
    ~/vss_env/bin/python scripts/tune_speaker_threshold.py --record-room 6

    # 3. see the table and the recommendation
    ~/vss_env/bin/python scripts/tune_speaker_threshold.py --sweep

    # 4. write the chosen threshold and switch verification on
    ~/vss_env/bin/python scripts/tune_speaker_threshold.py --apply
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.config import CONFIG, PATHS                      # noqa: E402
from jarvis.recorder import MicArbiter, Recorder             # noqa: E402
from jarvis.speaker import SpeakerVerifier                   # noqa: E402
from jarvis.speaker_tuning import (format_table, recommend,  # noqa: E402
                                   sweep)

CLIPS = PATHS.AIWS / "threshold_clips"
SAMPLE_RATE = 16000
GRID = [round(0.05 * i, 2) for i in range(2, 19)]            # 0.10 .. 0.90


def clip_dir(kind: str) -> Path:
    d = CLIPS / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_clip(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def load_clip(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def record_clips(kind: str, count: int, seconds: float) -> None:
    rec = Recorder(MicArbiter(), speaker_verifier=None)
    if not rec.mic_available:
        print("ERROR: no microphone detected.", file=sys.stderr)
        raise SystemExit(1)

    d = clip_dir(kind)
    existing = len(list(d.glob("*.wav")))
    if kind == "me":
        print(f"Recording {count} HELD-OUT clips of you ({seconds:.0f}s each).")
        print("These are NOT enrolled -- they are the test set. Speak normally.")
    else:
        print(f"Recording {count} clips of the room ({seconds:.0f}s each).")
        print("Get the TV / podcast / music playing at your usual volume now.")
        print("Do NOT talk during these -- they represent everything that is not you.")

    for i in range(count):
        for n in (3, 2, 1):
            print(f"  {n}...", end="", flush=True)
            time.sleep(1)
        print(f" clip {i + 1}/{count} recording...", end="", flush=True)
        audio = rec.record_fixed(seconds)
        if audio is None or len(audio) == 0:
            print(" no audio, skipped.")
            continue
        rms = float(np.sqrt(np.mean(np.square(audio))))
        path = d / f"{kind}_{existing + i:03d}.wav"
        save_clip(path, audio)
        print(f" saved {path.name} (rms={rms:.4f})")

    print(f"\n{len(list(d.glob('*.wav')))} '{kind}' clips on disk.")


def score_clips(v: SpeakerVerifier, kind: str) -> list[float]:
    scores = []
    for path in sorted(clip_dir(kind).glob("*.wav")):
        _, score = v.verify(load_clip(path))
        scores.append(score)
    return scores


def do_sweep(v: SpeakerVerifier, max_far: float):
    if not v.is_enrolled:
        print("ERROR: no voiceprint enrolled -- run scripts/enroll_voice.py first.",
              file=sys.stderr)
        print("       (with no voiceprint, verify() fails OPEN and scores are meaningless)",
              file=sys.stderr)
        raise SystemExit(1)

    print("Scoring clips against the enrolled voiceprint...")
    me = score_clips(v, "me")
    room = score_clips(v, "room")
    print(f"  you : n={len(me)}  " +
          (f"min={min(me):.3f} max={max(me):.3f}" if me else "(none)"))
    print(f"  room: n={len(room)} " +
          (f"min={min(room):.3f} max={max(room):.3f}" if room else "(none)"))

    rows = sweep(me, room, GRID)          # raises if either class is empty
    print("\n" + format_table(rows))
    rec = recommend(rows, max_far=max_far)
    print(f"\nRecommended threshold: {rec.threshold:.2f}")
    print(f"  false accepts (room getting in): {rec.far:.1%}")
    print(f"  false rejects (you shut out)   : {rec.frr:.1%}")
    print(f"  {rec.reason}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record-me", type=int, metavar="N")
    ap.add_argument("--record-room", type=int, metavar="N")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--apply", action="store_true",
                    help="write the recommended threshold and enable speaker_verify")
    ap.add_argument("--max-far", type=float, default=0.0,
                    help="tolerated false-accept rate (default 0 = let nothing through)")
    args = ap.parse_args()

    if args.record_me:
        record_clips("me", args.record_me, args.seconds)
        return 0
    if args.record_room:
        record_clips("room", args.record_room, args.seconds)
        return 0

    if not (args.sweep or args.apply):
        ap.print_help()
        return 0

    v = SpeakerVerifier(gpu=0, threshold=CONFIG.speaker_threshold)
    v.load()
    if not v.load_model():
        print("ERROR: speaker model failed to load.", file=sys.stderr)
        return 1

    rec = do_sweep(v, args.max_far)

    if args.apply:
        if not rec.clean_separation:
            print("\nNOTE: the populations overlap. Applying anyway, but expect")
            print("      either some room audio through or some of your own")
            print("      commands rejected. More takes usually fixes this.")
        # update() is the thread-safe field-set + save in jarvis/config.py
        CONFIG.update(speaker_threshold=rec.threshold, speaker_verify=True)
        print(f"\nWrote speaker_threshold={rec.threshold:.2f} and "
              f"speaker_verify=true to {PATHS.SETTINGS_FILE}")
        print("Restart Jarvis for it to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
